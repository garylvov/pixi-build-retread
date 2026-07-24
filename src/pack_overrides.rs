//! Fix #22: the auto-repaired T1 (conda-as-truth) overrides that fix #20
//! discovered (a conflict originating in a **backend uv-closure** -- a
//! pack's `[package.build.config.retread-wheels]` resolution, computed deep
//! inside the JSON-RPC build backend) are recorded in a workspace-level
//! **ledger** (`.retread/auto-overrides.json`), NOT written into the
//! failing pack's own `pixi.toml`.
//!
//! Why not the pack manifest (fix #20's original sink, superseded here):
//! retread's core "clean minimal manifest" design goal requires that
//! AUTO-generated repairs never mutate a pixi.toml -- only a human editing
//! `[package.build.config.retread-overrides]` themselves should change that
//! file. Fix #20 proved the *closure* needs the override (the pack's uv
//! closure has no other override-consuming table), but writing it into the
//! pack's `pixi.toml` violated the pristine-manifest goal. This module
//! keeps the closure behavior identical while relocating the SINK: the
//! ledger is read back and merged into the pack's `RetreadConfig.overrides`
//! **in memory**, at `Handler::initialize` time (see
//! `merge_ledger_overrides` and its call site in `handler/mod.rs`), so the
//! closure sees the exact same effective overrides it always did -- it just
//! never touches disk to get there.
//!
//! Keying: a pack is identified by its `pixi.toml`'s workspace-relative
//! *directory* (e.g. `pypi-packs/isaac-pack-latest`), which is both stable
//! across a run (unlike an in-process bundle/pack "name" that isn't always
//! available at `initialize` time) and readable in a `git diff` -- the
//! ledger is meant to be committed alongside the workspace, exactly like
//! `.retread/solve-ledger.json`.
//!
//! Run-scoped rollback: a repair run that later EXHAUSTS must not leave a
//! half-written auto override behind (defect #3 parity, carried over from
//! fix #20/#21's pack-manifest snapshot). The single ledger file is
//! snapshotted once per run under `.retread/auto-overrides.json.bak`
//! (plus an `.absent` marker if the ledger didn't exist yet), restored on
//! exhaustion/interrupt/crash, and dropped on convergence.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use toml_edit::{Array, DocumentMut, InlineTable, Item, Value};

use crate::config::RetreadConfig;
use crate::relax::canonical_conda_name;

static ATOMIC_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// One auto-repaired override recorded in the ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoOverrideEntry {
    /// Conda-style spec written verbatim (e.g. `"==2.10.0"`), same format
    /// the manual `retread-overrides` table uses.
    pub spec: String,
    /// The bundle/pack name as the triggering conflict reported it (e.g.
    /// `"isaac-pack-latest"`); informational, not used as a lookup key.
    #[serde(default)]
    pub bundle: String,
    /// Human-readable description of the conflict that caused this
    /// override (the parsed `Conflict`'s ledger `conflict` text) -- the
    /// auditable "why" a `git diff` on the ledger alone can't show.
    #[serde(default)]
    pub provenance: String,
    /// `YYYY-MM-DD` the override was recorded.
    #[serde(default)]
    pub date: String,
}

/// `.retread/auto-overrides.json`: every pack's auto-repaired overrides,
/// keyed by the pack's workspace-relative directory, then by PyPI package
/// name. Deliberately mirrors the shape of a pack's manual
/// `[package.build.config.retread-overrides]` table (map of name -> spec)
/// with provenance metadata alongside.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoOverrideLedger {
    #[serde(default)]
    pub packs: BTreeMap<String, BTreeMap<String, AutoOverrideEntry>>,
    /// Generic fallback engine (ownership-driven repair, un-route
    /// candidate): PyPI names, per pack, that a repair decided must ship
    /// as a PyPI wheel rather than be conda-routed (the doctrine-(v)
    /// cure for a PyPI-vs-conda-forge metadata skew, e.g. run-15's
    /// `moviepy==2.2.1` vs conda-forge's stale `pillow <11.0` cap).
    /// Merged into the pack's `RetreadConfig.keep_pypi` at
    /// `Handler::initialize` time (see [`merge_ledger_overrides`]),
    /// exactly like [`Self::packs`] merges into `config.overrides` -- the
    /// pack's own `pixi.toml` is never touched.
    #[serde(default)]
    pub unrouted: BTreeMap<String, BTreeMap<String, UnrouteEntry>>,
    /// Explicit `retread solve --apply-ledger` decisions to omit a
    /// dependency from the pack's emitted conda requirements. These merge
    /// into `RetreadConfig.drop_deps` in memory; the pack manifest remains
    /// byte-identical.
    #[serde(default)]
    pub dropped_dependencies: BTreeMap<String, BTreeMap<String, DropDependencyEntry>>,
    /// Explicit `retread solve --apply-ledger` decisions to constrain a
    /// transitive root wheel. These merge into
    /// `RetreadConfig.retread_wheels` in memory; the pack manifest remains
    /// byte-identical.
    #[serde(default)]
    pub root_pins: BTreeMap<String, BTreeMap<String, RootPinEntry>>,
}

/// One explicit solve decision to drop a dependency from conda run
/// requirements. The package name is the inner map key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DropDependencyEntry {
    /// The bundle/pack name as the triggering conflict reported it.
    #[serde(default)]
    pub bundle: String,
    /// Human-readable description of the conflict and chosen repair.
    #[serde(default)]
    pub provenance: String,
    /// `YYYY-MM-DD` the decision was recorded.
    #[serde(default)]
    pub date: String,
}

/// One explicit solve decision to pin a transitive root wheel. The package
/// name is the inner map key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootPinEntry {
    /// PEP 440 spec written verbatim into the effective wheel entry.
    pub spec: String,
    /// Output bundle assigned to the effective wheel entry.
    #[serde(default)]
    pub bundle_group: String,
    /// The bundle/pack name as the triggering conflict reported it.
    #[serde(default)]
    pub bundle: String,
    /// Human-readable description of the conflict and chosen repair.
    #[serde(default)]
    pub provenance: String,
    /// `YYYY-MM-DD` the decision was recorded.
    #[serde(default)]
    pub date: String,
}

/// One validated `retread solve --apply-ledger` mutation. The solve driver
/// preflights and de-duplicates the complete audit before handing this batch
/// to [`write_solve_updates`], which serializes every decision with one atomic
/// ledger replacement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SolveLedgerUpdate {
    pub pack_pixi: PathBuf,
    pub bundle: String,
    pub package: String,
    pub provenance: String,
    pub action: SolveLedgerAction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SolveLedgerAction {
    DropDependency,
    Override { spec: String },
    RootPin { spec: String, bundle_group: String },
}

/// One auto-repaired un-route decision recorded in the ledger. Mirrors
/// [`AutoOverrideEntry`]'s provenance fields; carries no `spec` (un-route
/// is a boolean "keep as pypi" decision, not a version).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnrouteEntry {
    /// The bundle/pack name as the triggering conflict reported it.
    #[serde(default)]
    pub bundle: String,
    /// Human-readable description of the conflict that caused this
    /// un-route (the parsed fallback candidate's "why").
    #[serde(default)]
    pub provenance: String,
    /// `YYYY-MM-DD` the un-route was recorded.
    #[serde(default)]
    pub date: String,
}

impl AutoOverrideLedger {
    pub fn load(workspace_dir: &Path) -> Result<Self> {
        let path = ledger_path(workspace_dir);
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
    }

    pub fn write_atomic(&self, workspace_dir: &Path) -> Result<()> {
        let path = ledger_path(workspace_dir);
        let parent = path
            .parent()
            .context("auto-overrides ledger path has no parent directory")?;
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        let bytes = serde_json::to_vec_pretty(self)?;
        let (tmp, mut file) = create_atomic_temp(&path)?;

        if let Err(error) = file
            .write_all(&bytes)
            .with_context(|| format!("failed to write {}", tmp.display()))
            .and_then(|()| {
                file.sync_all()
                    .with_context(|| format!("failed to sync {}", tmp.display()))
            })
        {
            drop(file);
            let _ = std::fs::remove_file(&tmp);
            return Err(error);
        }
        drop(file);

        if let Err(error) = std::fs::rename(&tmp, &path)
            .with_context(|| format!("failed to rename into {}", path.display()))
        {
            let _ = std::fs::remove_file(&tmp);
            return Err(error);
        }

        // The file itself is durable before rename. Persisting the directory
        // entry is supported on Unix; treat directory fsync as best-effort on
        // filesystems that do not permit opening/syncing directories.
        if let Ok(directory) = std::fs::File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    }
}

fn create_atomic_temp(path: &Path) -> Result<(PathBuf, std::fs::File)> {
    let parent = path
        .parent()
        .context("auto-overrides ledger path has no parent directory")?;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("auto-overrides.json");
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    for _ in 0..128 {
        let sequence = ATOMIC_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let tmp = parent.join(format!(
            ".{filename}.tmp-{}-{timestamp}-{sequence}",
            std::process::id()
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
            Ok(file) => return Ok((tmp, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| format!("failed to create {}", tmp.display()));
            }
        }
    }

    anyhow::bail!(
        "failed to allocate a unique temporary ledger file beside {}",
        path.display()
    )
}

pub fn ledger_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(".retread").join("auto-overrides.json")
}

fn snapshot_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir
        .join(".retread")
        .join("auto-overrides.json.bak")
}

fn snapshot_absent_marker(workspace_dir: &Path) -> PathBuf {
    workspace_dir
        .join(".retread")
        .join("auto-overrides.json.bak.absent")
}

/// Workspace-relative, forward-slash-normalized key for a pack: the
/// directory containing its `pixi.toml`. Best-effort canonicalization on
/// both sides so callers can pass either absolute or relative paths as
/// long as they agree on the same workspace root; falls back to the raw
/// path if canonicalization fails (e.g. in tests against paths that don't
/// exist yet).
fn pack_key(workspace_dir: &Path, pack_pixi: &Path) -> String {
    let ws = workspace_dir
        .canonicalize()
        .unwrap_or_else(|_| workspace_dir.to_path_buf());
    let pack_abs = pack_pixi
        .canonicalize()
        .unwrap_or_else(|_| pack_pixi.to_path_buf());
    let pack_dir = pack_abs.parent().unwrap_or(&pack_abs);
    let rel = pack_dir.strip_prefix(&ws).unwrap_or(pack_dir);
    rel.to_string_lossy().replace('\\', "/")
}

fn remove_canonical_package<V>(
    actions: &mut BTreeMap<String, BTreeMap<String, V>>,
    pack: &str,
    package: &str,
) {
    let remove_pack = if let Some(entries) = actions.get_mut(pack) {
        entries.retain(|name, _| canonical_conda_name(name) != package);
        entries.is_empty()
    } else {
        false
    };
    if remove_pack {
        actions.remove(pack);
    }
}

fn has_canonical_package<V>(
    actions: &BTreeMap<String, BTreeMap<String, V>>,
    pack: &str,
    package: &str,
) -> bool {
    actions.get(pack).is_some_and(|entries| {
        entries
            .keys()
            .any(|name| canonical_conda_name(name) == package)
    })
}

fn take_canonical_package<V: Default>(entries: &mut BTreeMap<String, V>, package: &str) -> V {
    let aliases: Vec<String> = entries
        .keys()
        .filter(|name| canonical_conda_name(name) == package)
        .cloned()
        .collect();
    let selected = aliases
        .iter()
        .find(|name| name.as_str() == package)
        .or_else(|| aliases.first())
        .cloned();
    let entry = selected
        .as_ref()
        .and_then(|name| entries.remove(name))
        .unwrap_or_default();
    for alias in aliases {
        entries.remove(&alias);
    }
    entry
}

fn pack_source_metadata_outputs(
    workspace_dir: &Path,
    pack_pixi: &Path,
    triggering_bundle: &str,
    ledger: &AutoOverrideLedger,
) -> BTreeSet<String> {
    let mut outputs = BTreeSet::new();
    if !triggering_bundle.is_empty() {
        outputs.insert(triggering_bundle.to_string());
    }

    let key = pack_key(workspace_dir, pack_pixi);
    if let Ok(text) = std::fs::read_to_string(pack_pixi)
        && let Ok(manifest) = toml::from_str::<toml::Value>(&text)
    {
        if let Some(name) = manifest
            .get("package")
            .and_then(|package| package.get("name"))
            .and_then(toml::Value::as_str)
        {
            outputs.insert(name.to_string());
        }

        let config = manifest
            .get("package")
            .and_then(|package| package.get("build"))
            .and_then(|build| build.get("config"));
        let default_bundle = config
            .and_then(|config| config.get("retread-bundle"))
            .and_then(toml::Value::as_str);
        if let Some(bundle) = default_bundle {
            outputs.insert(bundle.to_string());
        }
        if let Some(wheels) = config
            .and_then(|config| config.get("retread-wheels"))
            .and_then(toml::Value::as_table)
        {
            for (package, entry) in wheels {
                let output = entry
                    .get("bundle")
                    .and_then(toml::Value::as_str)
                    .or(default_bundle)
                    .unwrap_or(package);
                outputs.insert(output.to_string());
            }
        }
    }

    if let Some(entries) = ledger.packs.get(&key) {
        outputs.extend(
            entries
                .values()
                .map(|entry| entry.bundle.clone())
                .filter(|bundle| !bundle.is_empty()),
        );
    }
    if let Some(entries) = ledger.unrouted.get(&key) {
        outputs.extend(
            entries
                .values()
                .map(|entry| entry.bundle.clone())
                .filter(|bundle| !bundle.is_empty()),
        );
    }
    if let Some(entries) = ledger.dropped_dependencies.get(&key) {
        outputs.extend(
            entries
                .values()
                .map(|entry| entry.bundle.clone())
                .filter(|bundle| !bundle.is_empty()),
        );
    }
    if let Some(entries) = ledger.root_pins.get(&key) {
        outputs.extend(entries.values().flat_map(|entry| {
            [entry.bundle.clone(), entry.bundle_group.clone()]
                .into_iter()
                .filter(|bundle| !bundle.is_empty())
        }));
    }

    outputs
}

fn invalidate_pack_source_metadata(
    workspace_dir: &Path,
    pack_pixi: &Path,
    triggering_bundle: &str,
    ledger: &AutoOverrideLedger,
) {
    for output in pack_source_metadata_outputs(workspace_dir, pack_pixi, triggering_bundle, ledger)
    {
        invalidate_pixi_source_metadata(workspace_dir, &output);
    }
}

/// Workspace-relative manifest path used in user-facing diagnostics.
///
/// This shares [`pack_key`] with the ledger writer so an actionable
/// suggestion names the same pack that an applied solve repair would target.
/// It performs no I/O beyond the best-effort canonicalization in `pack_key`.
pub(crate) fn pack_manifest_display_path(workspace_dir: Option<&Path>, pack_pixi: &Path) -> String {
    let filename = pack_pixi
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("pixi.toml");
    if let Some(workspace_dir) = workspace_dir {
        return format!("{}/{filename}", pack_key(workspace_dir, pack_pixi));
    }

    let normalized = pack_pixi.to_string_lossy().replace('\\', "/");
    if let Some(index) = normalized.find("pypi-packs/") {
        return normalized[index..].to_string();
    }
    let pack = pack_pixi
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .unwrap_or("unknown-pack");
    format!("pypi-packs/{pack}/{filename}")
}

/// Render one table-body assignment for a manual `retread-overrides` edit.
///
/// The package/spec pair has exactly the shape consumed by the pack's manual
/// `retread-overrides` table. Omitting the table header is deliberate: the
/// diagnostic tells the user to merge this assignment into the existing table,
/// so it cannot create a duplicate-table TOML error.
pub(crate) fn render_override_toml(package: &str, spec: &str) -> String {
    let mut document = DocumentMut::new();
    document[package] = Item::Value(Value::from(spec));
    document.to_string()
}

/// Render one table-body assignment for a transitive root pin in the failing
/// bundle group.
///
/// Both the package key and version value are formatted by `toml_edit`, so
/// package names that are not valid bare TOML keys are safely quoted. The
/// existing `retread-wheels` table is intentionally not re-declared.
pub(crate) fn render_root_pin_toml(package: &str, spec: &str, bundle_group: &str) -> String {
    let mut document = DocumentMut::new();
    let mut pin = InlineTable::new();
    pin.insert("version", Value::from(spec));
    pin.insert("bundle", Value::from(bundle_group));
    document[package] = Item::Value(Value::InlineTable(pin));
    document.to_string()
}

fn render_drop_deps_toml(package: &str) -> String {
    let mut document = DocumentMut::new();
    let mut drop_deps = Array::new();
    drop_deps.push(package);
    document["retread-drop-deps"] = Item::Value(Value::Array(drop_deps));
    document.to_string()
}

fn append_commented(output: &mut String, text: &str) {
    for line in text.trim_end().lines() {
        output.push_str("# ");
        output.push_str(line);
        output.push('\n');
    }
}

/// Render the primary drop-deps remediation plus one commented override
/// alternative per conflicting requirement.
pub(crate) fn render_drop_deps_with_override_menu(
    package: &str,
    alternatives: &[(String, String)],
) -> String {
    let mut rendered = format!(
        "# Edit the existing [package.build.config] table.\n\
         # Merge `{package}` into any existing retread-drop-deps array:\n"
    );
    rendered.push_str(&render_drop_deps_toml(package));
    for (index, (spec, source)) in alternatives.iter().enumerate() {
        rendered.push('\n');
        rendered.push_str(&format!(
            "# Alternative {}: keep the requirement from:\n",
            index + 1
        ));
        for source_line in source.lines() {
            rendered.push_str("#   ");
            rendered.push_str(source_line);
            rendered.push('\n');
        }
        rendered.push_str(
            "# Under [package.build.config.retread-overrides], add or update \
             (create that table once if absent):\n",
        );
        append_commented(&mut rendered, &render_override_toml(package, spec));
    }
    rendered
}

/// Append (or replace) an auto-repaired override for `pack_pixi`/`package`
/// into the workspace's ledger. This is the fix #22 replacement for fix
/// #20's `write_override`, which wrote directly into the pack's pixi.toml.
pub fn write_override(
    workspace_dir: &Path,
    pack_pixi: &Path,
    bundle: &str,
    package: &str,
    spec: &str,
    provenance: &str,
) -> Result<()> {
    let mut ledger = AutoOverrideLedger::load(workspace_dir)?;
    let key = pack_key(workspace_dir, pack_pixi);
    let package = canonical_conda_name(package);
    remove_canonical_package(&mut ledger.packs, &key, &package);
    remove_canonical_package(&mut ledger.root_pins, &key, &package);
    if !has_canonical_package(&ledger.dropped_dependencies, &key, &package) {
        ledger.packs.entry(key).or_default().insert(
            package,
            AutoOverrideEntry {
                spec: spec.to_string(),
                bundle: bundle.to_string(),
                provenance: provenance.to_string(),
                date: local_date(),
            },
        );
    }
    ledger.write_atomic(workspace_dir)?;
    // Run-12 (deps-from proof): pixi's own per-workspace source-metadata
    // cache (`.pixi/meta-v0/<output>-<hash>/<platform>-<hash>.json`)
    // memoizes each source pack's conda/outputs result keyed on the
    // pack's project model / configuration / backend fingerprints -- the
    // auto-overrides ledger is invisible to ALL of those (that's the
    // point of the ledger sink: the pack's pixi.toml is never touched),
    // so the next `pixi lock` replays the STALE pack render without ever
    // invoking the backend, and the repair is a silent no-op (iteration 3
    // of run 12 hit the byte-identical conflict 6 seconds later). Evict
    // this pack's metadata entries so pixi re-requests conda/outputs; the
    // backend then recomputes with the merged ledger override (and its
    // own conda-outputs memo is correctly keyed on a ledger fingerprint).
    invalidate_pixi_source_metadata(workspace_dir, bundle);
    Ok(())
}

/// Removes pixi's cached source-metadata entries for `bundle` from the
/// workspace's `.pixi/meta-v0/` cache (directories named
/// `<output-name>-<hash>`). Prefix-matched on `<bundle>-`; a false
/// positive (a sibling output whose name extends `bundle`'s) only costs a
/// recompute, never correctness. Best-effort: cache eviction must never
/// fail the caller (a vanished/absent dir is fine).
pub fn invalidate_pixi_source_metadata(workspace_dir: &Path, bundle: &str) {
    let meta_dir = workspace_dir.join(".pixi").join("meta-v0");
    let Ok(entries) = std::fs::read_dir(&meta_dir) else {
        return;
    };
    let prefixes: BTreeSet<String> = [bundle.to_string(), canonical_conda_name(bundle)]
        .into_iter()
        .filter(|bundle| !bundle.is_empty())
        .map(|bundle| format!("{bundle}-"))
        .collect();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if prefixes.iter().any(|prefix| name.starts_with(prefix))
            && let Err(err) = std::fs::remove_dir_all(entry.path())
        {
            eprintln!(
                "retread: could not evict pixi source-metadata cache {}: {err}",
                entry.path().display()
            );
        }
    }
}

/// Append (or replace) an auto-repaired un-route decision for
/// `pack_pixi`/`package` into the workspace's ledger -- the un-route
/// counterpart of [`write_override`]. Same ledger, same cache-eviction
/// follow-up (a pack's `keep-pypi` set changing invalidates its cached
/// conda/outputs render exactly like an overrides change does).
pub fn write_unroute(
    workspace_dir: &Path,
    pack_pixi: &Path,
    bundle: &str,
    package: &str,
    provenance: &str,
) -> Result<()> {
    let mut ledger = AutoOverrideLedger::load(workspace_dir)?;
    let key = pack_key(workspace_dir, pack_pixi);
    ledger.unrouted.entry(key).or_default().insert(
        package.to_string(),
        UnrouteEntry {
            bundle: bundle.to_string(),
            provenance: provenance.to_string(),
            date: local_date(),
        },
    );
    ledger.write_atomic(workspace_dir)?;
    invalidate_pixi_source_metadata(workspace_dir, bundle);
    Ok(())
}

/// Append (or replace) an explicit solve decision to drop `package` from
/// the pack's conda run requirements. The decision is ledger-only:
/// `pack_pixi` is used solely to derive the stable pack key and is never
/// edited.
pub fn write_drop_dependency(
    workspace_dir: &Path,
    pack_pixi: &Path,
    bundle: &str,
    package: &str,
    provenance: &str,
) -> Result<()> {
    let mut ledger = AutoOverrideLedger::load(workspace_dir)?;
    let key = pack_key(workspace_dir, pack_pixi);
    let package = canonical_conda_name(package);

    // A drop is the terminal action for this dependency: retaining an
    // alternative override/root-pin under an equivalent PEP 503 spelling
    // would make the effective repair depend on map iteration order.
    remove_canonical_package(&mut ledger.packs, &key, &package);
    remove_canonical_package(&mut ledger.root_pins, &key, &package);
    remove_canonical_package(&mut ledger.dropped_dependencies, &key, &package);
    ledger.dropped_dependencies.entry(key).or_default().insert(
        package,
        DropDependencyEntry {
            bundle: bundle.to_string(),
            provenance: provenance.to_string(),
            date: local_date(),
        },
    );
    ledger.write_atomic(workspace_dir)?;
    invalidate_pack_source_metadata(workspace_dir, pack_pixi, bundle, &ledger);
    Ok(())
}

/// Append (or replace) an explicit solve decision to pin a transitive root
/// wheel. The decision is ledger-only: `pack_pixi` is used solely to derive
/// the stable pack key and is never edited.
pub fn write_root_pin(
    workspace_dir: &Path,
    pack_pixi: &Path,
    bundle: &str,
    package: &str,
    spec: &str,
    bundle_group: &str,
    provenance: &str,
) -> Result<()> {
    let mut ledger = AutoOverrideLedger::load(workspace_dir)?;
    let key = pack_key(workspace_dir, pack_pixi);
    let mut affected_outputs =
        pack_source_metadata_outputs(workspace_dir, pack_pixi, bundle, &ledger);
    let package = canonical_conda_name(package);
    remove_canonical_package(&mut ledger.root_pins, &key, &package);
    remove_canonical_package(&mut ledger.packs, &key, &package);
    if !has_canonical_package(&ledger.dropped_dependencies, &key, &package) {
        ledger.root_pins.entry(key).or_default().insert(
            package,
            RootPinEntry {
                spec: spec.to_string(),
                bundle_group: bundle_group.to_string(),
                bundle: bundle.to_string(),
                provenance: provenance.to_string(),
                date: local_date(),
            },
        );
    }
    ledger.write_atomic(workspace_dir)?;
    affected_outputs.extend(pack_source_metadata_outputs(
        workspace_dir,
        pack_pixi,
        bundle_group,
        &ledger,
    ));
    for output in affected_outputs {
        invalidate_pixi_source_metadata(workspace_dir, &output);
    }
    Ok(())
}

/// Persist one fully-preflighted solve audit as a single ledger transaction.
///
/// No update is serialized until every item has been folded into memory.
/// Consequently an invalid later proposal cannot leave an earlier proposal
/// committed. Cache eviction happens only after the one atomic ledger rename.
pub(crate) fn write_solve_updates(
    workspace_dir: &Path,
    updates: &[SolveLedgerUpdate],
) -> Result<()> {
    if updates.is_empty() {
        return Ok(());
    }

    let mut ledger = AutoOverrideLedger::load(workspace_dir)?;
    let mut affected_outputs = BTreeSet::new();
    for update in updates {
        affected_outputs.extend(pack_source_metadata_outputs(
            workspace_dir,
            &update.pack_pixi,
            &update.bundle,
            &ledger,
        ));
        let key = pack_key(workspace_dir, &update.pack_pixi);
        let package = canonical_conda_name(&update.package);
        match &update.action {
            SolveLedgerAction::DropDependency => {
                remove_canonical_package(&mut ledger.packs, &key, &package);
                remove_canonical_package(&mut ledger.root_pins, &key, &package);
                remove_canonical_package(&mut ledger.dropped_dependencies, &key, &package);
                ledger.dropped_dependencies.entry(key).or_default().insert(
                    package,
                    DropDependencyEntry {
                        bundle: update.bundle.clone(),
                        provenance: update.provenance.clone(),
                        date: local_date(),
                    },
                );
            }
            SolveLedgerAction::Override { spec } => {
                remove_canonical_package(&mut ledger.packs, &key, &package);
                remove_canonical_package(&mut ledger.root_pins, &key, &package);
                remove_canonical_package(&mut ledger.dropped_dependencies, &key, &package);
                ledger.packs.entry(key).or_default().insert(
                    package,
                    AutoOverrideEntry {
                        spec: spec.clone(),
                        bundle: update.bundle.clone(),
                        provenance: update.provenance.clone(),
                        date: local_date(),
                    },
                );
            }
            SolveLedgerAction::RootPin { spec, bundle_group } => {
                affected_outputs.insert(bundle_group.clone());
                remove_canonical_package(&mut ledger.root_pins, &key, &package);
                remove_canonical_package(&mut ledger.packs, &key, &package);
                remove_canonical_package(&mut ledger.dropped_dependencies, &key, &package);
                ledger.root_pins.entry(key).or_default().insert(
                    package,
                    RootPinEntry {
                        spec: spec.clone(),
                        bundle_group: bundle_group.clone(),
                        bundle: update.bundle.clone(),
                        provenance: update.provenance.clone(),
                        date: local_date(),
                    },
                );
            }
        }
    }

    // Re-scan the final state so output groups introduced by this batch are
    // invalidated along with any groups removed/replaced above.
    for update in updates {
        affected_outputs.extend(pack_source_metadata_outputs(
            workspace_dir,
            &update.pack_pixi,
            &update.bundle,
            &ledger,
        ));
    }
    ledger.write_atomic(workspace_dir)?;
    for output in affected_outputs {
        invalidate_pixi_source_metadata(workspace_dir, &output);
    }
    Ok(())
}

/// Read-only: every PyPI name auto-repaired to un-route for `pack_pixi`.
/// Never fails the caller -- see [`overrides_for_pack`]'s doc comment.
pub fn unrouted_for_pack(
    workspace_dir: &Path,
    pack_pixi: &Path,
) -> std::collections::BTreeSet<String> {
    let ledger = match AutoOverrideLedger::load(workspace_dir) {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "retread: failed to read .retread/auto-overrides.json; \
                 proceeding with no auto un-routes for this pack"
            );
            return std::collections::BTreeSet::new();
        }
    };
    let key = pack_key(workspace_dir, pack_pixi);
    ledger
        .unrouted
        .get(&key)
        .map(|entries| entries.keys().cloned().collect())
        .unwrap_or_default()
}

/// Read-only: every dependency explicitly dropped for `pack_pixi`.
/// Never fails the caller -- see [`overrides_for_pack`]'s doc comment.
pub fn dropped_dependencies_for_pack(
    workspace_dir: &Path,
    pack_pixi: &Path,
) -> std::collections::BTreeSet<String> {
    let ledger = match AutoOverrideLedger::load(workspace_dir) {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "retread: failed to read .retread/auto-overrides.json; \
                 proceeding with no ledgered dropped dependencies for this pack"
            );
            return std::collections::BTreeSet::new();
        }
    };
    let key = pack_key(workspace_dir, pack_pixi);
    ledger
        .dropped_dependencies
        .get(&key)
        .map(|entries| entries.keys().cloned().collect())
        .unwrap_or_default()
}

/// Read-only: every transitive root pin explicitly recorded for
/// `pack_pixi`, including its audit metadata. Never fails the caller -- see
/// [`overrides_for_pack`]'s doc comment.
pub fn root_pins_for_pack(
    workspace_dir: &Path,
    pack_pixi: &Path,
) -> BTreeMap<String, RootPinEntry> {
    let ledger = match AutoOverrideLedger::load(workspace_dir) {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "retread: failed to read .retread/auto-overrides.json; \
                 proceeding with no ledgered root pins for this pack"
            );
            return BTreeMap::new();
        }
    };
    let key = pack_key(workspace_dir, pack_pixi);
    ledger.root_pins.get(&key).cloned().unwrap_or_default()
}

/// Read-only: every auto-repaired override recorded for `pack_pixi`, as a
/// plain `package -> spec` map ready to merge into a `RetreadConfig`.
/// Never fails the caller -- a missing/corrupt ledger degrades to "no auto
/// overrides" rather than aborting the build (the ledger is a cache-like
/// durable record, not a required input).
pub fn overrides_for_pack(workspace_dir: &Path, pack_pixi: &Path) -> BTreeMap<String, String> {
    let ledger = match AutoOverrideLedger::load(workspace_dir) {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "retread: failed to read .retread/auto-overrides.json; \
                 proceeding with no auto overrides for this pack"
            );
            return BTreeMap::new();
        }
    };
    let key = pack_key(workspace_dir, pack_pixi);
    ledger
        .packs
        .get(&key)
        .map(|entries| {
            entries
                .iter()
                .map(|(name, entry)| (name.clone(), entry.spec.clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// Merge this pack's ledger entries into its effective `RetreadConfig`, in
/// memory only. Ledger entries take the same "last write wins" precedence a
/// repeated fix #20 pack-manifest write used to have: if an override/root pin
/// and a user's manual entry share a key, the ledger (the more recently
/// repaired value) wins. Set-like drop-dependency and un-route decisions are
/// appended only when the manual config does not already contain them.
///
/// Called once per pack, from `Handler::initialize`, before the config is
/// stored in `state.config` -- every downstream consumer sees the merged
/// result with no further special-casing, and a ledger change busts the
/// fingerprint exactly like a manifest edit would.
pub fn merge_ledger_overrides(config: &mut RetreadConfig, workspace_dir: &Path, pack_pixi: &Path) {
    let dropped: BTreeSet<String> = dropped_dependencies_for_pack(workspace_dir, pack_pixi)
        .into_iter()
        .map(|package| canonical_conda_name(&package))
        .collect();

    for (package, spec) in overrides_for_pack(workspace_dir, pack_pixi) {
        let package = canonical_conda_name(&package);
        if dropped.contains(&package) {
            continue;
        }
        // Record ledger provenance so the conda run-dep emission's
        // manual-override exemption doesn't mistake this repair-derived
        // pypi override for hand-written intent (run-31 regression: the
        // pack re-emitted an exact `==` conda pin for every ledgered
        // package, undoing the bounded-range emission).
        config
            .ledger_overrides
            .retain(|name| canonical_conda_name(name) != package);
        config
            .overrides
            .retain(|name, _| canonical_conda_name(name) != package);
        config.ledger_overrides.insert(package.clone());
        config.overrides.insert(package, spec);
    }
    // Generic fallback engine's un-route candidate (doctrine (v)): merge
    // ledgered un-routes into `keep_pypi` the same way, so the pack's
    // auto-route sweep (`plan_auto_route_round`) skips these names on its
    // very next render -- see `AutoRouteOptions.keep_pypi` in
    // `uv_closure.rs`, sourced from `config.keep_pypi` in `handler/mod.rs`.
    for package in unrouted_for_pack(workspace_dir, pack_pixi) {
        let package = canonical_conda_name(&package);
        if !config
            .keep_pypi
            .iter()
            .any(|name| canonical_conda_name(name) == package)
        {
            config.keep_pypi.push(package);
        }
    }
    for package in &dropped {
        if !config
            .drop_deps
            .iter()
            .any(|name| canonical_conda_name(name) == *package)
        {
            config.drop_deps.push(package.clone());
        }
    }
    for (package, pin) in root_pins_for_pack(workspace_dir, pack_pixi) {
        let package = canonical_conda_name(&package);
        if dropped.contains(&package) {
            continue;
        }
        let mut entry = take_canonical_package(&mut config.retread_wheels, &package);
        if entry.url.is_some()
            || entry.path.is_some()
            || entry.git.is_some()
            || entry.from.is_some()
        {
            // The Track-2 root-pin proposal deliberately changes this root
            // to an exact index-resolved version. Keeping a URL/path/git
            // source alongside `version` would create a multi-form
            // WheelEntry after manifest validation has already run, and
            // URL form would silently take precedence over the ledger pin.
            // Replace incompatible source forms; ordinary spec entries keep
            // their index/extras and receive only the version/bundle overlay.
            entry = Default::default();
        }
        entry.version = Some(pin.spec);
        entry.bundle = Some(pin.bundle_group);
        config.retread_wheels.insert(package, entry);
    }
}

/// Snapshot the ledger file before this run's first write to it. No-op if
/// a snapshot already exists (first snapshot of the run wins). Records
/// whether the ledger existed at all so rollback can tell "restore old
/// content" from "delete -- there was no ledger before this run" apart.
pub fn ensure_snapshot(workspace_dir: &Path) -> Result<()> {
    let dir = workspace_dir.join(".retread");
    std::fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let bak = snapshot_path(workspace_dir);
    let marker = snapshot_absent_marker(workspace_dir);
    if bak.exists() || marker.exists() {
        return Ok(());
    }
    let path = ledger_path(workspace_dir);
    if path.exists() {
        std::fs::copy(&path, &bak)
            .with_context(|| format!("failed to snapshot {}", path.display()))?;
    } else {
        std::fs::write(&marker, b"")
            .with_context(|| "failed to write ledger-absent marker".to_string())?;
    }
    Ok(())
}

/// Restore the ledger to its pre-run state (or delete it, if it didn't
/// exist before this run), then drop the snapshot. Used on any
/// non-converged exit (exhaustion / interrupt / crash / per-env
/// `--keep-going` restore) -- same call sites that used to call fix #20's
/// per-pack-file `rollback_all`.
pub fn rollback_all(workspace_dir: &Path) -> Result<()> {
    // Evict pixi's source-metadata cache for every pack named in the
    // CURRENT (about-to-be-reverted) ledger: any metadata pixi recomputed
    // during this run reflects overrides that are about to disappear, and
    // pixi's cache key can't see the ledger (see `write_override`), so a
    // stale post-override render would otherwise replay after rollback.
    // Collected before the revert so the run's own additions are included.
    let current_ledger = AutoOverrideLedger::load(workspace_dir).ok();
    let mut touched_bundles: BTreeSet<String> = current_ledger
        .as_ref()
        .map(|ledger| {
            ledger
                .packs
                .values()
                .flat_map(|entries| entries.values().map(|e| e.bundle.clone()))
                .chain(
                    ledger
                        .unrouted
                        .values()
                        .flat_map(|entries| entries.values().map(|e| e.bundle.clone())),
                )
                .chain(
                    ledger
                        .dropped_dependencies
                        .values()
                        .flat_map(|entries| entries.values().map(|e| e.bundle.clone())),
                )
                .chain(ledger.root_pins.values().flat_map(|entries| {
                    entries
                        .values()
                        .flat_map(|e| [e.bundle.clone(), e.bundle_group.clone()])
                }))
                .filter(|b| !b.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if let Some(ledger) = current_ledger.as_ref() {
        for (pack, entries) in &ledger.dropped_dependencies {
            if entries.is_empty() {
                continue;
            }
            let pack_pixi = workspace_dir.join(pack).join("pixi.toml");
            touched_bundles.extend(pack_source_metadata_outputs(
                workspace_dir,
                &pack_pixi,
                "",
                ledger,
            ));
        }
    }
    let bak = snapshot_path(workspace_dir);
    let marker = snapshot_absent_marker(workspace_dir);
    let path = ledger_path(workspace_dir);
    if marker.exists() {
        if path.exists() {
            std::fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
        }
    } else if bak.exists() {
        std::fs::copy(&bak, &path)
            .with_context(|| format!("failed to restore {}", path.display()))?;
    }
    let _ = std::fs::remove_file(&bak);
    let _ = std::fs::remove_file(&marker);
    for bundle in &touched_bundles {
        invalidate_pixi_source_metadata(workspace_dir, bundle);
    }
    Ok(())
}

/// Drop the run's ledger snapshot without restoring -- called once the run
/// converges (the ledger entries are now part of the green lock and stay).
pub fn cleanup_all(workspace_dir: &Path) -> Result<()> {
    let _ = std::fs::remove_file(snapshot_path(workspace_dir));
    let _ = std::fs::remove_file(snapshot_absent_marker(workspace_dir));
    Ok(())
}

/// The exact trailing-comment sentinel fix #20's (now-removed) pixi.toml
/// sink used to tag an auto-written entry, e.g. `torch = "==2.10.0"  #
/// retread:override 2026-07-09`. Only used by [`migrate_pack_toml_entries`]
/// to identify pre-fix-#22 auto entries already committed to a pack's
/// pixi.toml; the current write path never emits this into any manifest.
const LEGACY_OVERRIDE_SENTINEL: &str = "retread:override";

/// One-shot migration for packs that already have fix #20-era auto
/// overrides committed in their `[package.build.config.retread-overrides]`
/// table (tagged with the `# retread:override <date>` sentinel comment).
/// Moves each sentineled entry into the workspace's
/// `.retread/auto-overrides.json` ledger and removes it from `pack_pixi`,
/// leaving any un-sentineled (genuinely manual) entries untouched. Returns
/// the migrated package names.
///
/// Run via `retread migrate-overrides --workspace <dir> --pack <pack
/// pixi.toml>`. Idempotent: running it again on an already-migrated pack
/// finds no sentineled entries and is a no-op.
pub fn migrate_pack_toml_entries(workspace_dir: &Path, pack_pixi: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(pack_pixi)
        .with_context(|| format!("failed to read pack manifest {}", pack_pixi.display()))?;
    let mut doc = text
        .parse::<toml_edit::DocumentMut>()
        .with_context(|| format!("failed to parse pack manifest {}", pack_pixi.display()))?;

    let bundle = doc
        .get("package")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or("unknown-pack")
        .to_string();

    let Some(overrides_tbl) = doc
        .get_mut("package")
        .and_then(|p| p.get_mut("build"))
        .and_then(|b| b.get_mut("config"))
        .and_then(|c| c.get_mut("retread-overrides"))
        .and_then(toml_edit::Item::as_table_like_mut)
    else {
        return Ok(Vec::new());
    };

    let mut migrated = Vec::new();
    let sentineled: Vec<String> = overrides_tbl
        .iter()
        .filter(|(_, item)| {
            item.as_value()
                .map(|v| {
                    v.decor().suffix().is_some_and(|s| {
                        s.as_str()
                            .is_some_and(|s| s.contains(LEGACY_OVERRIDE_SENTINEL))
                    })
                })
                .unwrap_or(false)
        })
        .map(|(k, _)| k.to_string())
        .collect();

    for package in &sentineled {
        let spec = overrides_tbl
            .get(package)
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_default();
        write_override(
            workspace_dir,
            pack_pixi,
            &bundle,
            package,
            &spec,
            "migrated from a pre-fix-#22 pixi.toml `# retread:override` sentinel entry",
        )?;
        overrides_tbl.remove(package);
        migrated.push(package.clone());
    }

    if !migrated.is_empty() {
        write_atomic_toml(pack_pixi, doc.to_string().as_bytes())?;
    }
    Ok(migrated)
}

fn write_atomic_toml(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("toml.retread-tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("failed to write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("failed to rename into {}", path.display()))?;
    Ok(())
}

fn local_date() -> String {
    std::process::Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "1970-01-01".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "retread-auto-overrides-test-{tag}-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn make_pack(ws: &Path, rel: &str) -> PathBuf {
        let pack_dir = ws.join(rel);
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_pixi = pack_dir.join("pixi.toml");
        std::fs::write(
            &pack_pixi,
            "[package]\nname = \"isaac-pack-latest\"\nversion = \"6.0.0\"\n",
        )
        .unwrap();
        pack_pixi
    }

    #[cfg(unix)]
    #[test]
    fn atomic_ledger_write_never_follows_fixed_temp_or_destination_symlinks() {
        use std::os::unix::fs::symlink;

        let tmp = TempDir::new("atomic-symlink");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/atomic-pack");
        let original = std::fs::read(&pack_pixi).unwrap();
        let retread_dir = ws.join(".retread");
        std::fs::create_dir_all(&retread_dir).unwrap();

        // The former predictable temp name was writable through this link.
        // Also make the destination itself a link: rename must replace that
        // directory entry rather than following it.
        let fixed_tmp = ledger_path(ws).with_extension("json.tmp");
        symlink(&pack_pixi, &fixed_tmp).unwrap();
        symlink(&pack_pixi, ledger_path(ws)).unwrap();

        AutoOverrideLedger::default().write_atomic(ws).unwrap();

        assert_eq!(
            std::fs::read(&pack_pixi).unwrap(),
            original,
            "neither an attacker-controlled temp nor destination symlink may modify pixi.toml"
        );
        assert!(
            std::fs::symlink_metadata(&fixed_tmp)
                .unwrap()
                .file_type()
                .is_symlink(),
            "the obsolete fixed temp path must never be opened"
        );
        assert!(
            !std::fs::symlink_metadata(ledger_path(ws))
                .unwrap()
                .file_type()
                .is_symlink(),
            "atomic rename must replace, not follow, the destination symlink"
        );
        assert_eq!(
            AutoOverrideLedger::load(ws).unwrap(),
            AutoOverrideLedger::default()
        );
    }

    #[test]
    fn solve_updates_commit_complete_batch_without_touching_pack_manifests() {
        let tmp = TempDir::new("solve-batch");
        let ws = tmp.path();
        let first = make_pack(ws, "pypi-packs/first");
        let second = make_pack(ws, "pypi-packs/second");
        let first_before = std::fs::read(&first).unwrap();
        let second_before = std::fs::read(&second).unwrap();

        write_solve_updates(
            ws,
            &[
                SolveLedgerUpdate {
                    pack_pixi: first.clone(),
                    bundle: "first-output".into(),
                    package: "NumPy".into(),
                    provenance: "equal-authority conflict".into(),
                    action: SolveLedgerAction::DropDependency,
                },
                SolveLedgerUpdate {
                    pack_pixi: second.clone(),
                    bundle: "second-output".into(),
                    package: "torch".into(),
                    provenance: "graph-root conflict".into(),
                    action: SolveLedgerAction::RootPin {
                        spec: "==2.10.0".into(),
                        bundle_group: "second-output".into(),
                    },
                },
            ],
        )
        .unwrap();

        let ledger = AutoOverrideLedger::load(ws).unwrap();
        assert!(ledger.dropped_dependencies["pypi-packs/first"].contains_key("numpy"));
        assert_eq!(
            ledger.root_pins["pypi-packs/second"]["torch"].spec,
            "==2.10.0"
        );
        assert_eq!(std::fs::read(first).unwrap(), first_before);
        assert_eq!(std::fs::read(second).unwrap(), second_before);
    }

    #[test]
    fn explicit_solve_action_switch_replaces_every_stale_canonical_action() {
        let tmp = TempDir::new("solve-action-switch");
        let ws = tmp.path();
        let pack = make_pack(ws, "pypi-packs/switch");
        let update = |package: &str, action| SolveLedgerUpdate {
            pack_pixi: pack.clone(),
            bundle: "switch-output".into(),
            package: package.into(),
            provenance: "latest Track-2 decision".into(),
            action,
        };

        write_solve_updates(ws, &[update("Demo_Pkg", SolveLedgerAction::DropDependency)]).unwrap();
        write_solve_updates(
            ws,
            &[update(
                "demo-pkg",
                SolveLedgerAction::Override { spec: ">=2".into() },
            )],
        )
        .unwrap();
        let after_override = AutoOverrideLedger::load(ws).unwrap();
        assert!(after_override.packs["pypi-packs/switch"].contains_key("demo-pkg"));
        assert!(
            !after_override
                .dropped_dependencies
                .get("pypi-packs/switch")
                .is_some_and(|entries| entries.contains_key("demo-pkg"))
        );
        assert!(
            !after_override
                .root_pins
                .get("pypi-packs/switch")
                .is_some_and(|entries| entries.contains_key("demo-pkg"))
        );

        write_solve_updates(
            ws,
            &[update(
                "DEMO.PKG",
                SolveLedgerAction::RootPin {
                    spec: "==3".into(),
                    bundle_group: "switch-output".into(),
                },
            )],
        )
        .unwrap();
        let after_root = AutoOverrideLedger::load(ws).unwrap();
        assert!(
            !after_root
                .packs
                .get("pypi-packs/switch")
                .is_some_and(|entries| entries.contains_key("demo-pkg"))
        );
        assert_eq!(
            after_root.root_pins["pypi-packs/switch"]["demo-pkg"].spec,
            "==3"
        );
        assert!(
            !after_root
                .dropped_dependencies
                .get("pypi-packs/switch")
                .is_some_and(|entries| entries.contains_key("demo-pkg"))
        );
    }

    #[test]
    fn diagnostic_manifest_path_tracks_pack_not_output_name() {
        let tmp = TempDir::new("diagnostic-pack-path");
        let pack_pixi = make_pack(tmp.path(), "pypi-packs/robotics-pack");

        assert_eq!(
            pack_manifest_display_path(Some(tmp.path()), &pack_pixi),
            "pypi-packs/robotics-pack/pixi.toml"
        );
        assert_ne!(
            pack_manifest_display_path(Some(tmp.path()), &pack_pixi),
            "pypi-packs/robotics-output/pixi.toml",
            "the output bundle name must not replace the initialized pack identity"
        );
    }

    #[test]
    fn diagnostic_override_menu_merges_into_existing_config_table() {
        let alternatives = vec![
            (
                "==1.26.4".to_string(),
                "wheel `old-extension==1.0.0`".to_string(),
            ),
            (
                ">=2,<3".to_string(),
                "wheel `new-extension==2.0.0`".to_string(),
            ),
        ];
        let menu = render_drop_deps_with_override_menu("numpy", &alternatives);
        assert!(
            !menu.contains("\n[package.build.config"),
            "merge-oriented edit re-declared a table:\n{menu}"
        );

        let manifest = r#"
[package]
name = "robotics-pack"
version = "1.0.0"

[package.build]
backend = { name = "pixi-build-retread", version = "*" }

[package.build.config]
retread-auto-bundle = true

[package.build.config.retread-wheels]
robotics = { version = "==1.0.0", bundle = "robotics-output" }

[package.build.config.retread-overrides]
packaging = ">=24"
"#;
        let merged = manifest.replacen(
            "[package.build.config]\n",
            &format!("[package.build.config]\n{menu}"),
            1,
        );
        let parsed: toml::Value =
            toml::from_str(&merged).expect("merge must not duplicate existing tables");
        assert_eq!(
            parsed["package"]["build"]["config"]["retread-drop-deps"][0].as_str(),
            Some("numpy")
        );
        assert_eq!(
            parsed["package"]["build"]["config"]["retread-auto-bundle"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["package"]["build"]["config"]["retread-wheels"]["robotics"]["bundle"].as_str(),
            Some("robotics-output")
        );
        assert_eq!(
            parsed["package"]["build"]["config"]["retread-overrides"]["packaging"].as_str(),
            Some(">=24")
        );

        for (spec, _) in alternatives {
            let rendered = render_override_toml("numpy", &spec);
            let parsed: toml::Value = toml::from_str(&rendered).unwrap();
            assert_eq!(parsed["numpy"].as_str(), Some(spec.as_str()));
            let commented_entry = format!("# numpy = \"{spec}\"");
            assert!(
                menu.lines().any(|line| line == commented_entry),
                "commented menu omitted the rendered override `{spec}`:\n{menu}"
            );
        }
    }

    #[test]
    fn diagnostic_root_pin_is_parseable_and_safely_quotes_package_key() {
        let rendered = render_root_pin_toml("pin.root", "==2.6.20", "robotics-output");
        let parsed: toml::Value = toml::from_str(&rendered).unwrap();

        assert_eq!(parsed["pin.root"]["version"].as_str(), Some("==2.6.20"));
        assert_eq!(
            parsed["pin.root"]["bundle"].as_str(),
            Some("robotics-output")
        );
        assert!(!rendered.contains("[package.build.config"));
        assert!(
            rendered.contains(
                "\"pin.root\" = { version = \"==2.6.20\", bundle = \"robotics-output\" }"
            ),
            "root package key was not safely quoted in:\n{rendered}"
        );
    }

    #[test]
    fn write_override_evicts_pixi_source_metadata_for_the_pack() {
        // Run-12 regression: pixi's `.pixi/meta-v0/<output>-<hash>/` cache
        // can't see the ledger, so a ledger write MUST evict the pack's
        // cached metadata or the next `pixi lock` replays the stale render
        // and the repair is a silent no-op. Sibling packs' entries stay.
        let tmp = TempDir::new("evict");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/isaac-pack-latest");
        let stale = ws.join(".pixi/meta-v0/isaac-pack-latest-ClbwcsBUfqs");
        let sibling = ws.join(".pixi/meta-v0/other-pack-AbCdEfGhIjK");
        std::fs::create_dir_all(&stale).unwrap();
        std::fs::create_dir_all(&sibling).unwrap();
        std::fs::write(stale.join("linux_64-x.json"), b"{}").unwrap();
        std::fs::write(sibling.join("linux_64-y.json"), b"{}").unwrap();

        write_override(
            ws,
            &pack_pixi,
            "isaac-pack-latest",
            "setuptools",
            ">=68,<81",
            "conflict: workspace range vs auto-routed pin",
        )
        .unwrap();

        assert!(
            !stale.exists(),
            "the repaired pack's stale pixi metadata cache entry must be evicted"
        );
        assert!(
            sibling.exists(),
            "an unrelated pack's metadata cache entry must survive"
        );
    }

    #[test]
    fn write_root_pin_evicts_triggering_bundle_and_target_group_metadata() {
        let tmp = TempDir::new("root-pin-evict");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/isaac-pack-latest");
        write_root_pin(
            ws,
            &pack_pixi,
            "isaac-pack-latest",
            "CMEEL_Boost",
            "==1.87.0",
            "old_output",
            "prior root pin",
        )
        .unwrap();

        let triggering = ws.join(".pixi/meta-v0/isaac-pack-latest-trigger");
        let target_group = ws.join(".pixi/meta-v0/robotics-output-target");
        let prior_group = ws.join(".pixi/meta-v0/old-output-prior");
        let sibling = ws.join(".pixi/meta-v0/other-pack-sibling");
        std::fs::create_dir_all(&triggering).unwrap();
        std::fs::create_dir_all(&target_group).unwrap();
        std::fs::create_dir_all(&prior_group).unwrap();
        std::fs::create_dir_all(&sibling).unwrap();

        write_root_pin(
            ws,
            &pack_pixi,
            "ISAAC_PACK_LATEST",
            "cmeel.boost",
            "==1.88.0",
            "robotics_output",
            "transitive root conflict",
        )
        .unwrap();

        assert!(!triggering.exists());
        assert!(!target_group.exists());
        assert!(!prior_group.exists());
        assert!(sibling.exists());
    }

    #[test]
    fn write_drop_evicts_every_output_metadata_entry_for_multi_output_pack() {
        let tmp = TempDir::new("drop-pack-wide-evict");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/multi-pack");
        std::fs::write(
            &pack_pixi,
            r#"[package]
name = "multi-pack"
version = "1.0.0"

[package.build.config]
retread-bundle = "core-output"

[package.build.config.retread-wheels]
core-wheel = { version = "==1.0.0" }
vision-wheel = { version = "==2.0.0", bundle = "vision-output" }
audio-wheel = { version = "==3.0.0", bundle = "audio-output" }
"#,
        )
        .unwrap();

        let outputs = ["multi-pack", "core-output", "vision-output", "audio-output"];
        for output in outputs {
            std::fs::create_dir_all(ws.join(format!(".pixi/meta-v0/{output}-stale"))).unwrap();
        }
        let sibling = ws.join(".pixi/meta-v0/unrelated-output-still-valid");
        std::fs::create_dir_all(&sibling).unwrap();

        write_drop_dependency(
            ws,
            &pack_pixi,
            "vision-output",
            "shared-dependency",
            "equal-authority conflict",
        )
        .unwrap();

        for output in outputs {
            assert!(
                !ws.join(format!(".pixi/meta-v0/{output}-stale")).exists(),
                "drop must evict stale metadata for pack output {output}"
            );
        }
        assert!(sibling.exists(), "unrelated output metadata must survive");
    }

    #[test]
    fn rollback_all_evicts_pixi_source_metadata_for_touched_packs() {
        // Metadata pixi recomputed DURING the run reflects overrides that
        // rollback removes; those entries must be evicted too or the
        // post-rollback state replays a render of overrides that no
        // longer exist.
        let tmp = TempDir::new("rollback-evict");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/isaac-pack-latest");
        ensure_snapshot(ws).unwrap();
        write_override(
            ws,
            &pack_pixi,
            "isaac-pack-latest",
            "setuptools",
            ">=68,<81",
            "conflict",
        )
        .unwrap();
        // Simulate pixi re-rendering the pack mid-run (a fresh cache
        // entry that bakes in the just-written override).
        let recomputed = ws.join(".pixi/meta-v0/isaac-pack-latest-XyZ123");
        std::fs::create_dir_all(&recomputed).unwrap();

        rollback_all(ws).unwrap();

        assert!(
            !recomputed.exists(),
            "rollback must evict metadata rendered with the reverted overrides"
        );
        // Ledger restored to pre-run (absent) state.
        assert!(!ledger_path(ws).exists());
    }

    #[test]
    fn rollback_evicts_metadata_for_solve_ledger_relaxations() {
        let tmp = TempDir::new("solve-rollback-evict");
        let ws = tmp.path();
        let drop_pack = make_pack(ws, "pypi-packs/drop-pack");
        let pin_pack = make_pack(ws, "pypi-packs/pin-pack");
        std::fs::write(
            &drop_pack,
            r#"[package]
name = "drop-pack"
version = "1.0.0"

[package.build.config.retread-wheels]
primary = { version = "==1.0.0", bundle = "drop-pack" }
secondary = { version = "==1.0.0", bundle = "drop-secondary" }
"#,
        )
        .unwrap();
        ensure_snapshot(ws).unwrap();
        write_drop_dependency(
            ws,
            &drop_pack,
            "drop-pack",
            "numpy",
            "equal-authority conflict",
        )
        .unwrap();
        write_root_pin(
            ws,
            &pin_pack,
            "pin-pack",
            "cmeel-boost",
            "==1.88.0",
            "pin-output",
            "transitive conflict",
        )
        .unwrap();

        let drop_metadata = ws.join(".pixi/meta-v0/drop-pack-recomputed");
        let drop_secondary_metadata = ws.join(".pixi/meta-v0/drop-secondary-recomputed");
        let pin_metadata = ws.join(".pixi/meta-v0/pin-pack-recomputed");
        let pin_group_metadata = ws.join(".pixi/meta-v0/pin-output-recomputed");
        std::fs::create_dir_all(&drop_metadata).unwrap();
        std::fs::create_dir_all(&drop_secondary_metadata).unwrap();
        std::fs::create_dir_all(&pin_metadata).unwrap();
        std::fs::create_dir_all(&pin_group_metadata).unwrap();

        rollback_all(ws).unwrap();

        assert!(!drop_metadata.exists());
        assert!(!drop_secondary_metadata.exists());
        assert!(!pin_metadata.exists());
        assert!(!pin_group_metadata.exists());
        assert!(!ledger_path(ws).exists());
    }

    #[test]
    fn write_override_lands_in_ledger_not_pack_manifest() {
        let tmp = TempDir::new("write");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/isaac-pack-latest");
        let original = std::fs::read_to_string(&pack_pixi).unwrap();

        write_override(
            ws,
            &pack_pixi,
            "isaac-pack-latest",
            "torch",
            "==2.10.0",
            "conflict: torch >=2.11.0 vs conda ==2.10.0",
        )
        .unwrap();

        // Pack manifest is byte-identical -- the auto override never
        // touched it.
        assert_eq!(std::fs::read_to_string(&pack_pixi).unwrap(), original);

        // The ledger recorded it instead.
        let ledger = AutoOverrideLedger::load(ws).unwrap();
        let entry = ledger
            .packs
            .get("pypi-packs/isaac-pack-latest")
            .and_then(|p| p.get("torch"))
            .expect("expected a ledger entry for torch");
        assert_eq!(entry.spec, "==2.10.0");
        assert_eq!(entry.bundle, "isaac-pack-latest");
        assert!(entry.provenance.contains("2.11.0"));
    }

    #[test]
    fn solve_relaxations_land_in_ledger_not_pack_manifest() {
        let tmp = TempDir::new("solve-ledger-write");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/isaac-pack-latest");
        let original = std::fs::read(&pack_pixi).unwrap();

        write_drop_dependency(
            ws,
            &pack_pixi,
            "isaac-pack-latest",
            "numpy",
            "equal-authority conflict between wheel requirements",
        )
        .unwrap();
        write_root_pin(
            ws,
            &pack_pixi,
            "isaac-pack-latest",
            "cmeel-boost",
            "==1.88.0",
            "isaac-pack-latest",
            "transitive root introduced the conflicting numpy pin",
        )
        .unwrap();

        assert_eq!(
            std::fs::read(&pack_pixi).unwrap(),
            original,
            "ledger apply must leave pixi.toml byte-identical"
        );

        let ledger = AutoOverrideLedger::load(ws).unwrap();
        let pack = "pypi-packs/isaac-pack-latest";
        let drop = &ledger.dropped_dependencies[pack]["numpy"];
        assert_eq!(drop.bundle, "isaac-pack-latest");
        assert!(drop.provenance.contains("equal-authority"));
        assert!(!drop.date.is_empty());

        let pin = &ledger.root_pins[pack]["cmeel-boost"];
        assert_eq!(pin.spec, "==1.88.0");
        assert_eq!(pin.bundle_group, "isaac-pack-latest");
        assert_eq!(pin.bundle, "isaac-pack-latest");
        assert!(pin.provenance.contains("transitive root"));
        assert!(!pin.date.is_empty());

        assert!(dropped_dependencies_for_pack(ws, &pack_pixi).contains("numpy"));
        assert_eq!(
            root_pins_for_pack(ws, &pack_pixi)["cmeel-boost"].spec,
            "==1.88.0"
        );
    }

    #[test]
    fn drop_removes_canonical_override_and_root_pin_actions() {
        let tmp = TempDir::new("drop-action-precedence");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/action-pack");

        write_override(
            ws,
            &pack_pixi,
            "action-pack",
            "Shared_Dependency",
            ">=1,<2",
            "earlier override",
        )
        .unwrap();
        write_root_pin(
            ws,
            &pack_pixi,
            "action-pack",
            "shared.dependency",
            "==1.5.0",
            "action-output",
            "earlier root pin",
        )
        .unwrap();
        write_drop_dependency(
            ws,
            &pack_pixi,
            "action-pack",
            "SHARED---DEPENDENCY",
            "chosen drop",
        )
        .unwrap();

        let ledger = AutoOverrideLedger::load(ws).unwrap();
        let pack = "pypi-packs/action-pack";
        assert!(
            ledger.packs.get(pack).is_none_or(BTreeMap::is_empty),
            "drop must remove the same canonical override action"
        );
        assert!(
            ledger.root_pins.get(pack).is_none_or(BTreeMap::is_empty),
            "drop must remove the same canonical root-pin action"
        );
        assert!(
            ledger.dropped_dependencies[pack].contains_key("shared-dependency"),
            "drop key must use the shared canonical package spelling"
        );

        let mut config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": { "unrelated": { "version": "==1.0.0" } }
        }))
        .unwrap();
        merge_ledger_overrides(&mut config, ws, &pack_pixi);
        assert!(!config.overrides.contains_key("shared-dependency"));
        assert!(!config.retread_wheels.contains_key("shared-dependency"));
        assert_eq!(config.drop_deps, vec!["shared-dependency".to_string()]);
    }

    #[test]
    fn root_pin_merges_into_existing_canonical_wheel_key() {
        let tmp = TempDir::new("root-pin-canonical-merge");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/root-pack");
        write_root_pin(
            ws,
            &pack_pixi,
            "root-pack",
            "Shared.Dependency",
            "==2.0.0",
            "root-output",
            "canonical collision",
        )
        .unwrap();

        let mut config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": {
                "shared_dependency": {
                    "version": "==1.0.0",
                    "index": "https://packages.example/simple"
                }
            }
        }))
        .unwrap();
        merge_ledger_overrides(&mut config, ws, &pack_pixi);

        assert_eq!(config.retread_wheels.len(), 1);
        let entry = &config.retread_wheels["shared-dependency"];
        assert_eq!(entry.version.as_deref(), Some("==2.0.0"));
        assert_eq!(entry.bundle.as_deref(), Some("root-output"));
        assert_eq!(
            entry.index.as_deref(),
            Some("https://packages.example/simple")
        );
    }

    #[test]
    fn solve_ledger_relaxations_merge_into_effective_config() {
        let tmp = TempDir::new("solve-ledger-merge");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/isaac-pack-latest");

        write_drop_dependency(
            ws,
            &pack_pixi,
            "isaac-pack-latest",
            "numpy",
            "equal-authority conflict",
        )
        .unwrap();
        write_root_pin(
            ws,
            &pack_pixi,
            "isaac-pack-latest",
            "existing-root",
            "==2.6.20",
            "robotics-output",
            "pin existing transitive root",
        )
        .unwrap();
        write_root_pin(
            ws,
            &pack_pixi,
            "isaac-pack-latest",
            "new-root",
            "==0.9.0",
            "robotics-output",
            "pin previously implicit transitive root",
        )
        .unwrap();
        write_root_pin(
            ws,
            &pack_pixi,
            "isaac-pack-latest",
            "url-root",
            "==4.2.0",
            "robotics-output",
            "replace an incompatible direct-url root",
        )
        .unwrap();

        let mut config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": {
                "existing-root": {
                    "version": "==3.0.0",
                    "index": "https://packages.example/simple",
                    "extras": ["feature"],
                    "bundle": "old-output"
                },
                "url-root": {
                    "url": "https://packages.example/url-root-4.1.0.whl",
                    "sha256": "deadbeef",
                    "bundle": "old-output"
                }
            },
            "retread-drop-deps": ["manual-drop"]
        }))
        .unwrap();

        merge_ledger_overrides(&mut config, ws, &pack_pixi);

        assert_eq!(
            config.drop_deps,
            vec!["manual-drop".to_string(), "numpy".to_string()]
        );
        let existing = &config.retread_wheels["existing-root"];
        assert_eq!(existing.version.as_deref(), Some("==2.6.20"));
        assert_eq!(existing.bundle.as_deref(), Some("robotics-output"));
        assert_eq!(
            existing.index.as_deref(),
            Some("https://packages.example/simple")
        );
        assert_eq!(existing.extras, vec!["feature".to_string()]);

        let added = &config.retread_wheels["new-root"];
        assert_eq!(added.version.as_deref(), Some("==0.9.0"));
        assert_eq!(added.bundle.as_deref(), Some("robotics-output"));
        assert!(added.url.is_none());
        assert!(added.path.is_none());
        assert!(added.git.is_none());
        added
            .validate("new-root")
            .expect("an absent root must be constructed as a valid spec wheel");

        let replaced = &config.retread_wheels["url-root"];
        assert_eq!(replaced.version.as_deref(), Some("==4.2.0"));
        assert_eq!(replaced.bundle.as_deref(), Some("robotics-output"));
        assert!(replaced.url.is_none());
        assert!(replaced.sha256.is_none());
        assert!(replaced.path.is_none());
        assert!(replaced.git.is_none());
        assert!(replaced.from.is_none());
        replaced
            .validate("url-root")
            .expect("the exact ledger root pin must replace incompatible source forms");
    }

    #[test]
    fn old_ledger_without_solve_fields_loads_with_empty_defaults() {
        let tmp = TempDir::new("solve-ledger-backcompat");
        let ws = tmp.path();
        std::fs::create_dir_all(ws.join(".retread")).unwrap();
        std::fs::write(
            ledger_path(ws),
            r#"{
  "packs": {
    "pypi-packs/p": {
      "torch": {
        "spec": "==2.10.0",
        "bundle": "p",
        "provenance": "old ledger",
        "date": "2026-07-01"
      }
    }
  },
  "unrouted": {}
}"#,
        )
        .unwrap();

        let ledger = AutoOverrideLedger::load(ws).unwrap();
        assert!(ledger.dropped_dependencies.is_empty());
        assert!(ledger.root_pins.is_empty());
        assert_eq!(ledger.packs["pypi-packs/p"]["torch"].spec, "==2.10.0");
    }

    #[test]
    fn closure_read_merges_ledger_over_manual_overrides() {
        let tmp = TempDir::new("merge");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/isaac-pack-latest");

        write_override(
            ws,
            &pack_pixi,
            "isaac-pack-latest",
            "torch",
            "==2.10.0",
            "c",
        )
        .unwrap();

        let mut config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": { "mypkg": { "version": "==1.0.0" } },
            "retread-overrides": { "tinyobjloader": "==2.0.0rc13" }
        }))
        .unwrap();

        merge_ledger_overrides(&mut config, ws, &pack_pixi);

        // Manual entry preserved.
        assert_eq!(
            config.overrides.get("tinyobjloader").map(String::as_str),
            Some("==2.0.0rc13")
        );
        // Ledger entry merged in.
        assert_eq!(
            config.overrides.get("torch").map(String::as_str),
            Some("==2.10.0")
        );
    }

    #[test]
    fn ledger_entry_wins_on_key_conflict_with_manual_entry() {
        // Matches the old write_override's unconditional-overwrite
        // semantics: a later auto-repair for a key the user also declared
        // manually replaces the manual value (same as when both lived in
        // the same pixi.toml table).
        let tmp = TempDir::new("conflict");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/isaac-pack-latest");
        write_override(
            ws,
            &pack_pixi,
            "isaac-pack-latest",
            "torch",
            "==2.10.0",
            "c",
        )
        .unwrap();

        let mut config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": { "mypkg": { "version": "==1.0.0" } },
            "retread-overrides": { "torch": "==2.9.0" }
        }))
        .unwrap();
        merge_ledger_overrides(&mut config, ws, &pack_pixi);
        assert_eq!(
            config.overrides.get("torch").map(String::as_str),
            Some("==2.10.0")
        );
    }

    #[test]
    fn fingerprint_changes_when_ledger_changes() {
        let tmp = TempDir::new("fingerprint");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/isaac-pack-latest");
        let base: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": { "mypkg": { "version": "==1.0.0" } }
        }))
        .unwrap();
        let chans = ["conda-forge".to_string()];

        let mut before = base.clone();
        merge_ledger_overrides(&mut before, ws, &pack_pixi);
        let fp_before = crate::courier::config_fingerprint(&before, &chans, "");

        write_override(
            ws,
            &pack_pixi,
            "isaac-pack-latest",
            "torch",
            "==2.10.0",
            "c",
        )
        .unwrap();
        let mut after = base.clone();
        merge_ledger_overrides(&mut after, ws, &pack_pixi);
        let fp_after = crate::courier::config_fingerprint(&after, &chans, "");

        assert_ne!(
            fp_before, fp_after,
            "a ledger change must bust the config fingerprint (cache must not replay stale)"
        );
    }

    #[test]
    fn snapshot_rollback_restores_ledger_absence() {
        let tmp = TempDir::new("rollback-absent");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/p");

        // No ledger exists yet this run.
        ensure_snapshot(ws).unwrap();
        write_override(ws, &pack_pixi, "p", "torch", "==2.10.0", "c").unwrap();
        assert!(ledger_path(ws).exists());

        rollback_all(ws).unwrap();
        assert!(
            !ledger_path(ws).exists(),
            "ledger must be deleted on rollback when it was absent pre-run"
        );
        assert!(!snapshot_path(ws).exists());
        assert!(!snapshot_absent_marker(ws).exists());
    }

    #[test]
    fn snapshot_rollback_restores_prior_ledger_content() {
        let tmp = TempDir::new("rollback-prior");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/p");

        // A prior (converged) run already recorded an entry.
        write_override(ws, &pack_pixi, "p", "numpy", ">=1.24", "prior run").unwrap();
        let prior = AutoOverrideLedger::load(ws).unwrap();

        // This run adds another entry, then exhausts.
        ensure_snapshot(ws).unwrap();
        write_override(ws, &pack_pixi, "p", "torch", "==2.10.0", "this run").unwrap();
        rollback_all(ws).unwrap();

        let after = AutoOverrideLedger::load(ws).unwrap();
        assert_eq!(
            after, prior,
            "rollback must drop only this run's ledger appends"
        );
        assert!(!after.packs["pypi-packs/p"].contains_key("torch"));
        assert!(after.packs["pypi-packs/p"].contains_key("numpy"));
    }

    #[test]
    fn cleanup_keeps_the_written_ledger_entry() {
        let tmp = TempDir::new("cleanup");
        let ws = tmp.path();
        let pack_pixi = make_pack(ws, "pypi-packs/p");
        ensure_snapshot(ws).unwrap();
        write_override(ws, &pack_pixi, "p", "torch", "==2.10.0", "c").unwrap();
        cleanup_all(ws).unwrap();
        assert!(!snapshot_path(ws).exists());
        assert!(!snapshot_absent_marker(ws).exists());
        let ledger = AutoOverrideLedger::load(ws).unwrap();
        assert!(ledger.packs["pypi-packs/p"].contains_key("torch"));
    }

    #[test]
    fn migrate_moves_sentineled_entries_to_ledger_and_leaves_manual_ones() {
        let tmp = TempDir::new("migrate");
        let ws = tmp.path();
        let pack_dir = ws.join("pypi-packs/isaac-pack-latest");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_pixi = pack_dir.join("pixi.toml");
        std::fs::write(
            &pack_pixi,
            "[package]\nname = \"isaac-pack-latest\"\nversion = \"6.0.0\"\n\n\
             [package.build.config.retread-overrides]\n\
             tinyobjloader = \"==2.0.0rc13\"\n\
             torch = \"==2.10.0\"  # retread:override 2026-07-09\n\
             torchvision = \"==0.25.0\"  # retread:override 2026-07-09\n\
             torchaudio = \"==2.10.0\"  # retread:override 2026-07-09\n",
        )
        .unwrap();

        let migrated = migrate_pack_toml_entries(ws, &pack_pixi).unwrap();
        let migrated_set: std::collections::BTreeSet<String> = migrated.into_iter().collect();
        let expected_set: std::collections::BTreeSet<String> =
            ["torch", "torchvision", "torchaudio"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        assert_eq!(migrated_set, expected_set);

        // Manual entry (no sentinel) stays in the pack manifest.
        let pack_text = std::fs::read_to_string(&pack_pixi).unwrap();
        assert!(pack_text.contains("tinyobjloader = \"==2.0.0rc13\""));
        assert!(!pack_text.contains("torch"));
        assert!(!pack_text.contains(LEGACY_OVERRIDE_SENTINEL));

        // Ledger has all three, with correct specs and bundle attribution.
        let overrides = overrides_for_pack(ws, &pack_pixi);
        assert_eq!(overrides.get("torch").map(String::as_str), Some("==2.10.0"));
        assert_eq!(
            overrides.get("torchvision").map(String::as_str),
            Some("==0.25.0")
        );
        assert_eq!(
            overrides.get("torchaudio").map(String::as_str),
            Some("==2.10.0")
        );
        let ledger = AutoOverrideLedger::load(ws).unwrap();
        let entry = &ledger.packs["pypi-packs/isaac-pack-latest"]["torch"];
        assert_eq!(entry.bundle, "isaac-pack-latest");

        // Re-running is a no-op (idempotent).
        let migrated_again = migrate_pack_toml_entries(ws, &pack_pixi).unwrap();
        assert!(migrated_again.is_empty());
    }
}
