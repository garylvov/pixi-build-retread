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

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use toml_edit::{Array, DocumentMut, InlineTable, Item, Table, Value};

use crate::config::RetreadConfig;

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
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let bytes = serde_json::to_vec_pretty(self)?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, bytes)
            .with_context(|| format!("failed to write {}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("failed to rename into {}", path.display()))?;
        Ok(())
    }
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

fn ensure_toml_table<'a>(parent: &'a mut Table, key: &str) -> &'a mut Table {
    if !parent.contains_key(key) {
        let mut table = Table::new();
        table.set_implicit(true);
        parent.insert(key, Item::Table(table));
    }
    parent
        .get_mut(key)
        .and_then(Item::as_table_mut)
        .expect("the inserted TOML item is a table")
}

/// Pure TOML counterpart to [`write_override`].
///
/// The package/spec pair has exactly the shape consumed by the pack's manual
/// `retread-overrides` table, but this helper only renders an in-memory
/// document. It never writes the pack manifest or the auto-overrides ledger.
pub(crate) fn render_override_toml(package: &str, spec: &str) -> String {
    let mut document = DocumentMut::new();
    let package_table = ensure_toml_table(document.as_table_mut(), "package");
    let build_table = ensure_toml_table(package_table, "build");
    let config_table = ensure_toml_table(build_table, "config");
    let overrides_table = ensure_toml_table(config_table, "retread-overrides");
    overrides_table[package] = Item::Value(Value::from(spec));
    document.to_string()
}

/// Render a concrete version constraint for a transitive root in the pack's
/// `retread-wheels` table.
///
/// Both the package key and version value are formatted by `toml_edit`, so
/// package names that are not valid bare TOML keys are safely quoted. This is
/// a pure diagnostic helper and never writes the pack manifest.
pub(crate) fn render_root_pin_toml(package: &str, spec: &str) -> String {
    let mut document = DocumentMut::new();
    let package_table = ensure_toml_table(document.as_table_mut(), "package");
    let build_table = ensure_toml_table(package_table, "build");
    let config_table = ensure_toml_table(build_table, "config");
    let wheels_table = ensure_toml_table(config_table, "retread-wheels");
    let mut pin = InlineTable::new();
    pin.insert("version", Value::from(spec));
    wheels_table.insert(package, Item::Value(Value::InlineTable(pin)));
    document.to_string()
}

fn render_drop_deps_toml(package: &str) -> String {
    let mut document = DocumentMut::new();
    let package_table = ensure_toml_table(document.as_table_mut(), "package");
    let build_table = ensure_toml_table(package_table, "build");
    let config_table = ensure_toml_table(build_table, "config");
    let mut drop_deps = Array::new();
    drop_deps.push(package);
    config_table["retread-drop-deps"] = Item::Value(Value::Array(drop_deps));
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
    let mut rendered = render_drop_deps_toml(package);
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
    ledger.packs.entry(key).or_default().insert(
        package.to_string(),
        AutoOverrideEntry {
            spec: spec.to_string(),
            bundle: bundle.to_string(),
            provenance: provenance.to_string(),
            date: local_date(),
        },
    );
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
    let prefix = format!("{bundle}-");
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if name.starts_with(&prefix)
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

/// Merge this pack's ledger entries into `config.overrides`, in memory
/// only. Ledger entries take the same "last write wins" precedence a
/// repeated fix #20 pack-manifest write used to have: if a ledger entry
/// and a user's manual `retread-overrides` entry share a key, the ledger
/// (the more recently auto-repaired value) wins, matching the byte-for-
/// byte overwrite `write_override`'s old `pixi.toml` sink used to perform.
///
/// Called once per pack, from `Handler::initialize`, before the config is
/// stored in `state.config` -- every downstream consumer (`resolve_all`,
/// `apply_emission`'s `effective.overrides`, and `courier::
/// config_fingerprint`'s `declared_config`) sees the merged result with no
/// further special-casing, and a ledger change busts the fingerprint
/// exactly like a manifest edit would.
pub fn merge_ledger_overrides(config: &mut RetreadConfig, workspace_dir: &Path, pack_pixi: &Path) {
    for (package, spec) in overrides_for_pack(workspace_dir, pack_pixi) {
        // Record ledger provenance so the conda run-dep emission's
        // manual-override exemption doesn't mistake this repair-derived
        // pypi override for hand-written intent (run-31 regression: the
        // pack re-emitted an exact `==` conda pin for every ledgered
        // package, undoing the bounded-range emission).
        config.ledger_overrides.insert(package.clone());
        config.overrides.insert(package, spec);
    }
    // Generic fallback engine's un-route candidate (doctrine (v)): merge
    // ledgered un-routes into `keep_pypi` the same way, so the pack's
    // auto-route sweep (`plan_auto_route_round`) skips these names on its
    // very next render -- see `AutoRouteOptions.keep_pypi` in
    // `uv_closure.rs`, sourced from `config.keep_pypi` in `handler/mod.rs`.
    for package in unrouted_for_pack(workspace_dir, pack_pixi) {
        if !config.keep_pypi.contains(&package) {
            config.keep_pypi.push(package);
        }
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
    let touched_bundles: std::collections::BTreeSet<String> =
        AutoOverrideLedger::load(workspace_dir)
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
                    .filter(|b| !b.is_empty())
                    .collect()
            })
            .unwrap_or_default();
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
    fn diagnostic_override_menu_uses_parseable_pack_override_entries() {
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
        let parsed: toml::Value = toml::from_str(&menu).unwrap();
        assert_eq!(
            parsed["package"]["build"]["config"]["retread-drop-deps"][0].as_str(),
            Some("numpy")
        );

        for (spec, _) in alternatives {
            let rendered = render_override_toml("numpy", &spec);
            let parsed: toml::Value = toml::from_str(&rendered).unwrap();
            assert_eq!(
                parsed["package"]["build"]["config"]["retread-overrides"]["numpy"].as_str(),
                Some(spec.as_str())
            );
            let commented_entry = format!("# numpy = \"{spec}\"");
            assert!(
                menu.lines().any(|line| line == commented_entry),
                "commented menu omitted the rendered override `{spec}`:\n{menu}"
            );
        }
    }

    #[test]
    fn diagnostic_root_pin_is_parseable_and_safely_quotes_package_key() {
        let rendered = render_root_pin_toml("pin.root", "==2.6.20");
        let parsed: toml::Value = toml::from_str(&rendered).unwrap();

        assert_eq!(
            parsed["package"]["build"]["config"]["retread-wheels"]["pin.root"]["version"].as_str(),
            Some("==2.6.20")
        );
        assert!(
            rendered.contains("\"pin.root\" = { version = \"==2.6.20\" }"),
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
