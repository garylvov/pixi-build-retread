//! METAGEN-1: the LAST-RESORT producer of a path source's `dependencies`, so
//! retread derives a dynamic editable's static metadata ITSELF instead of a
//! human transcribing it into a vendored tree.
//!
//! # The defect this closes, and it is a SILENT one
//!
//! [`crate::path_source_metadata`] already supplies pixi's frontend resolver
//! with static PEP 621 metadata for every local path source: it DERIVES a
//! record from the manifest plus the tree and renders a generated shim
//! `<pack>/sources/<project>/pyproject.toml` carrying a real `[project]`
//! table, then repoints the workspace manifest at it. Not one byte is written
//! outside the pack directory, which is the property a hand-written
//! `pyproject.toml` inside `imprint-data` can never have: those trees are
//! vendored submodules, so a hand patch is lost on the next vendor update.
//!
//! But the DERIVATION has one field it cannot produce. `tree_facts` reads the
//! four PEP 621 fields from three readers — `*.egg-info/PKG-INFO`,
//! `config/extension.toml`, and literal quoted kwargs in `setup.py` — and only
//! the FIRST of them states `dependencies` at all. `scan_setup_kwarg` reads a
//! quoted SCALAR, so it structurally cannot read
//!
//! ```python
//! INSTALL_REQUIRES = ["psutil", "cmaes"]
//! ...
//! setup(..., install_requires=INSTALL_REQUIRES, ...)
//! ```
//!
//! which is exactly how `pace_sim2real` declares its two dependencies. With no
//! `PKG-INFO`, `facts.dependencies` is `None`, `record_from_tree` falls back to
//! the seed's empty vector, and the shim is rendered with
//! `dependencies = []` — **silently**. `check_drift` cannot catch it either:
//! it skips the field when the tree states nothing. The module's own worst case
//! ("a dependency the record omits is a dependency the resolver will never
//! see") is reachable with no refusal anywhere.
//!
//! MEASURED, on the live workspace, 2026-09-09: both trees DO carry an
//! `*.egg-info/PKG-INFO` today (`pace_sim2real` 762 B with `Requires-Dist:
//! psutil` and `Requires-Dist: cmaes`; `protomotions` 288 B with none). And
//! both are UNTRACKED BUILD RESIDUE: the two trees are git submodules and
//! `git ls-files --error-unmatch <tree>/*.egg-info/PKG-INFO` reports the file
//! is not known to git. So the derivation's only authoritative reader is a
//! leftover from somebody's previous install. A `git clean`, a fresh clone, or
//! a stage-mirror `cp -al` of a pristine tree turns `["psutil", "cmaes"]` into
//! `[]` with no row, no warning and no refusal.
//!
//! # What this module does instead
//!
//! When the tree states nothing about its dependencies anywhere, ASK THE TREE
//! — by running its own PEP 517 `prepare_metadata_for_build_wheel` in a build
//! environment this module owns:
//!
//! * **ISOLATED.** The interpreter is chosen by [`select_interpreter`] and
//!   [`assert_isolated_interpreter`] REFUSES any candidate under a `.pixi/envs`
//!   directory. That refusal is the whole point of the lane and not a detail:
//!   what makes the frontend's build cost 1 295.6 s is that ITS venv is
//!   parented on `.pixi/envs/pace/bin/python3.11`, so the 14 GB pace prefix is
//!   materialised onto NFS inside the resolve timer. An isolated venv installs
//!   the tree's declared `build-system.requires` and nothing else.
//! * **CACHED**, in this module's own store, keyed by a bounded content hash
//!   of the tree ([`source_hash`]) plus the interpreter's tag. An edit to the
//!   vendored `setup.py` moves the hash and the entry is rebuilt.
//! * **LOUD.** A failed build is an error that reaches the caller. There is
//!   deliberately NO fallback to the frontend's prefix-parented build: falling
//!   back would restore the 1 295.6 s silently, which is the defect.
//!
//! # Why the sdist-metadata store is NOT the store used here
//!
//! [`crate::sdist_metadata`]'s entry key begins with `source_digest`, a fold of
//! the sdist's URL and ETag read out of the cache shard's `revision.http`. A
//! local path tree has no URL, no ETag and no `revision.http`, and that store's
//! `PERMITTED_ENTRY_NAMES` admits only uv's own `Metadata23` bytes beside that
//! file. Reusing it would mean two different key rules wearing one name — the
//! exact defect SDIST-META-3's header describes. So this is a SEVENTH
//! instantiation of the one marker-store walk, with its own spec, and not a
//! seventh reaper.
//!
//! # Why the hash is bounded, with the numbers
//!
//! A git tree hash is refused because both trees are SUBMODULES: `HEAD^{tree}`
//! describes the committed tree, so an uncommitted edit to the vendored
//! `setup.py` would leave the key unchanged and a stale entry live. A key that
//! cannot see the edit it exists to see is not a key.
//!
//! A whole-tree content hash is refused on a measurement: `pace_sim2real` is
//! 27 files and 40 KB, but `ProtoMotions` is 1 001 files and 1.8 GB with
//! `.git` excluded. Reading 1.8 GB off NFS to key a 288-byte metadata blob
//! would re-create this lane's own defect — NFS I/O billed to the lock — one
//! level down.
//!
//! So [`HASHED_ROOT_FILES`] and [`HASHED_DIR`] name the bounded set of
//! metadata-determining files, ABSENCE is hashed as well as presence (so
//! creating a `setup.cfg` moves the key), and a file over
//! [`HASHED_MAX_FILE_BYTES`] or a `config/` holding more than
//! [`HASHED_DIR_MAX_FILES`] entries is a REFUSAL rather than a silent skip.

use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

/// The store directory under the persistent cache root. Read by
/// [`crate::source_build::PATH_SOURCE_METADATA_STORE_SPEC`] rather than
/// re-spelled there, so the reaper cannot drift from the writer.
pub const CACHE_NAMESPACE: &str = "path-source-metadata";

/// The generation this binary addresses. Bump when the stored bytes' meaning
/// changes; every other generation is still walked and aged by the reaper.
pub const CACHE_VERSION: &str = "v1";

/// The file whose presence MAKES a directory an entry, written LAST.
pub const COMPLETION_MARKER: &str = "complete.json";

/// The verbatim core-metadata bytes the isolated build produced.
pub const METADATA_FILE: &str = "core-metadata.txt";

/// The `<prefix> reap` / `<prefix> evicted` row stem, so an operator greps one
/// word for this store.
pub const STORE_ROW: &str = "path_source_metadata_store";

/// The store-wide reap try-lock, a dotfile beside the generations.
pub const STORE_REAP_LOCK: &str = ".path-source-metadata.reap.lock";

/// The reaper's default age bound for this store, in days.
///
/// **14, the campaign-wide value every other store uses**, and it is 14 because
/// `store_reap`'s GUARD 5 says so: that guard walks `Store::ALL` and asserts
/// `resolved_max_age_days(store, None) == 14` for every store, so a
/// store-specific horizon is not a preference this module gets to hold on its
/// own — it is a campaign invariant with a live guard.
///
/// This module's first draft said 30, on the argument that an entry here is
/// 300–800 bytes keyed by a VENDORED tree that moves on a months-long cadence.
/// The guard failed the run (`left: 30, right: 14`) and it was right to: the
/// argument is real but the cost it saves is one ~25 s isolated build per
/// fortnight, which does not buy a second policy. If the horizon should move it
/// should move for every store at once, in that guard.
pub const DEFAULT_MAX_AGE_DAYS: u64 = 14;

/// The key's domain tag. Distinct from every other store's so two hashes can
/// never be confused for one another in a log.
pub const KEY_SCHEMA: &str = "retread-path-source-metadata-cache-v1";

/// The key's rendered prefix, so an operator can tell at a glance which store
/// a bare hex string in a row belongs to.
pub const KEY_PREFIX: &str = "psm-";

/// The only entry names this store admits. Enforced on publish by shape, so a
/// build tree, an egg-info or a wheel is UNREPRESENTABLE here rather than
/// merely checked for.
pub const PERMITTED_ENTRY_NAMES: [&str; 2] = [METADATA_FILE, COMPLETION_MARKER];

/// Root-relative file names whose bytes decide a tree's metadata. Bounded on
/// purpose — see the module header for the 40 KB / 1.8 GB measurement.
pub const HASHED_ROOT_FILES: [&str; 4] =
    ["pyproject.toml", "setup.py", "setup.cfg", "MANIFEST.in"];

/// The one subdirectory whose files are hashed as well: `pace_sim2real`'s
/// `setup.py` does `toml.load(config/extension.toml)` at module scope, so the
/// version, description, author and keywords all live there and an edit to it
/// is an edit to the metadata.
pub const HASHED_DIR: &str = "config";

/// More entries than this directly under [`HASHED_DIR`] is a refusal, not a
/// silent truncation: the bound exists to keep the hash cheap, and a tree that
/// breaks it needs a human to look, not a quietly partial key.
pub const HASHED_DIR_MAX_FILES: usize = 64;

/// A single hashed file larger than this is a refusal, for the same reason.
pub const HASHED_MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;

/// The path component that marks a pixi environment prefix. An interpreter
/// under one of these is precisely the frontend's 1 295.6 s build.
pub const PIXI_ENVS_DIR: &str = "envs";

/// The `.pixi` directory that holds [`PIXI_ENVS_DIR`].
pub const PIXI_DIR: &str = ".pixi";

// ---------------------------------------------------------------------------
// the switch
// ---------------------------------------------------------------------------

/// The declared backend key, exported so the config field, the CLI flag and
/// every refusal name ONE string.
pub const CONFIG_KEY: &str = "retread-derive-editable-metadata";

/// How the caller wants a dynamic path source's dependencies produced.
///
/// This is a struct and not a bare `bool` because the store root travels with
/// the switch: a caller that can turn the derivation on but cannot say where to
/// cache it would rebuild on every lock, which is a different capability
/// wearing the same name.
#[derive(Debug, Clone)]
pub struct DeriveOptions {
    /// `false` reproduces the pre-METAGEN-1 behaviour exactly: no build, no
    /// row, and a tree that states no dependencies yields none.
    pub enabled: bool,
    /// Where entries live. `None` means "no store": the build still runs and
    /// still refuses loudly, and the row says `store=miss`.
    pub store_root: Option<PathBuf>,
}

impl Default for DeriveOptions {
    /// **TRUE.** The `false` arm is the one that silently drops a dependency,
    /// so OFF is the unsafe default. It is safe to default ON because the
    /// derivation only ever RUNS when the tree states nothing — every tree that
    /// carries a `[project]` table or a `PKG-INFO` is untouched and prints no
    /// row, so turning this on cannot change a lock that was already correct.
    fn default() -> Self {
        Self {
            enabled: true,
            store_root: None,
        }
    }
}

impl DeriveOptions {
    /// The switch as the backend config states it, with the store root the
    /// caller resolved. `None` (the key absent) is ON — see [`Self::default`].
    pub fn from_config(enabled: Option<bool>, store_root: Option<PathBuf>) -> Self {
        Self {
            enabled: enabled.unwrap_or(true),
            store_root,
        }
    }

    /// The opt-out, for the guard that proves `false` is byte-identical to the
    /// behaviour before this module existed.
    pub fn off() -> Self {
        Self {
            enabled: false,
            store_root: None,
        }
    }
}

// ---------------------------------------------------------------------------
// what the isolated build produced
// ---------------------------------------------------------------------------

/// Where the metadata for one tree came from. Three outcomes, three live
/// producers — a row value nothing can print is the same defect as a criterion
/// nothing produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreOutcome {
    /// A published entry keyed by this tree's hash was read back.
    Hit,
    /// No entry, and no store root to publish one to: the build ran and its
    /// result was used but not kept.
    Miss,
    /// No entry; the build ran and the result was published.
    Built,
}

impl StoreOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Miss => "miss",
            Self::Built => "built",
        }
    }
}

/// The core-metadata fields this lane needs, parsed out of the RFC 822
/// metadata the build produced.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CoreMetadata {
    pub name: Option<String>,
    pub version: Option<String>,
    pub requires_python: Option<String>,
    /// Every `Requires-Dist` header, in file order. An empty vector is a
    /// STATEMENT that the tree has no dependencies — distinct from never
    /// having asked, which is what `Option` carries one level up.
    pub requires_dist: Vec<String>,
}

/// Parse core metadata headers. Headers end at the first blank line, exactly
/// as `tree_facts` reads a `PKG-INFO`, because they are the same format and
/// two readers of one format is how they drift.
pub fn parse_core_metadata(text: &str) -> CoreMetadata {
    let mut out = CoreMetadata::default();
    for line in text.lines() {
        if line.is_empty() {
            break;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().to_string();
        match key.to_ascii_lowercase().as_str() {
            "name" => {
                if out.name.is_none() {
                    out.name = Some(value);
                }
            }
            "version" => {
                if out.version.is_none() {
                    out.version = Some(value);
                }
            }
            "requires-python" => {
                if out.requires_python.is_none() {
                    out.requires_python = Some(value);
                }
            }
            "requires-dist" => out.requires_dist.push(value),
            _ => {}
        }
    }
    out
}

/// One completed derivation: the metadata, where it came from, and how long it
/// took. Every field of the evidence row is here so the row cannot claim
/// something the value does not carry.
#[derive(Debug, Clone)]
pub struct Derived {
    pub metadata: CoreMetadata,
    pub source_hash: String,
    pub store: StoreOutcome,
    /// The interpreter the build actually used — asserted isolated before the
    /// build ran, and reported so the claim is auditable rather than trusted.
    pub interpreter: PathBuf,
    pub wall_seconds: f64,
}

impl Derived {
    /// The evidence row, one per derived editable.
    pub fn row(&self, path: &str) -> String {
        format!(
            "### EDITABLE METADATA DERIVED path={} source_hash={} store={} build_env=isolated wall={:.3}",
            path,
            self.source_hash,
            self.store.as_str(),
            self.wall_seconds
        )
    }
}

// ---------------------------------------------------------------------------
// the bounded content hash
// ---------------------------------------------------------------------------

/// One hashed member: its root-relative name and what was there.
fn hash_member(hasher: &mut Sha256, name: &str, bytes: Option<&[u8]>) {
    hasher.update(name.as_bytes());
    hasher.update([0u8]);
    match bytes {
        // ABSENCE is hashed, so creating a `setup.cfg` moves the key.
        None => hasher.update(b"absent"),
        Some(bytes) => {
            hasher.update(b"present");
            hasher.update(bytes.len().to_le_bytes());
            hasher.update(bytes);
        }
    }
    hasher.update([0u8]);
}

fn read_bounded(path: &Path) -> Result<Option<Vec<u8>>> {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("stat {}", path.display()));
        }
    };
    if !meta.is_file() {
        return Ok(None);
    }
    if meta.len() > HASHED_MAX_FILE_BYTES {
        bail!(
            "{} is {} bytes, over the {HASHED_MAX_FILE_BYTES}-byte bound this \
             module hashes. The bound exists so keying a path source cannot \
             turn into reading a source tree off NFS; a metadata-determining \
             file this large needs a human to look at it rather than a quietly \
             partial key.",
            path.display(),
            meta.len()
        );
    }
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(Some(bytes))
}

/// The bounded content hash of a source tree — this store's key material.
///
/// Hashes, in a fixed order: each name in [`HASHED_ROOT_FILES`] (present or
/// absent), then every file directly under [`HASHED_DIR`] sorted by name. The
/// result is stable across machines and moves the moment any of those bytes do.
pub fn source_hash(tree: &Path) -> Result<String> {
    if !tree.is_dir() {
        bail!(
            "cannot hash {}: it is not a directory. A path source whose tree is \
             absent cannot have its metadata derived, and guessing one would be \
             worse than refusing.",
            tree.display()
        );
    }
    let mut hasher = Sha256::new();
    hasher.update(b"retread-path-source-metadata-");
    hasher.update(CACHE_VERSION.as_bytes());
    hasher.update([0u8]);

    for name in HASHED_ROOT_FILES {
        let bytes = read_bounded(&tree.join(name))?;
        hash_member(&mut hasher, name, bytes.as_deref());
    }

    let dir = tree.join(HASHED_DIR);
    let mut members: Vec<String> = Vec::new();
    if dir.is_dir() {
        for entry in
            std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))?
        {
            let entry = entry.with_context(|| format!("reading an entry of {}", dir.display()))?;
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                bail!(
                    "{} holds a file whose name is not UTF-8. A key that cannot \
                     spell one of its own members is not a key.",
                    dir.display()
                );
            };
            members.push(name);
        }
        if members.len() > HASHED_DIR_MAX_FILES {
            bail!(
                "{} holds {} files, over the {HASHED_DIR_MAX_FILES}-file bound \
                 this module hashes. Refusing rather than hashing a prefix: a \
                 partial key silently reuses metadata for a tree that changed.",
                dir.display(),
                members.len()
            );
        }
        members.sort();
    }
    hasher.update(b"config-dir");
    hasher.update(members.len().to_le_bytes());
    hasher.update([0u8]);
    for name in &members {
        let bytes = read_bounded(&dir.join(name))?;
        let member = format!("{HASHED_DIR}/{name}");
        hash_member(&mut hasher, &member, bytes.as_deref());
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// The `build-system.requires` the tree declares, or the setuptools default
/// PEP 518 prescribes when it declares none.
///
/// This is what makes the build env LIGHT: `pace_sim2real` declares
/// `["setuptools", "wheel", "toml"]` and gets exactly those three, not a 14 GB
/// conda prefix.
pub fn build_requires(tree: &Path) -> Result<Vec<String>> {
    let file = tree.join("pyproject.toml");
    let default = || {
        vec![
            "setuptools>=64".to_string(),
            "wheel".to_string(),
        ]
    };
    let text = match std::fs::read_to_string(&file) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(default()),
        Err(error) => return Err(error).with_context(|| format!("reading {}", file.display())),
    };
    let parsed: toml::Value = toml::from_str(&text)
        .with_context(|| format!("parsing {} to read its build-system.requires", file.display()))?;
    let Some(requires) = parsed
        .get("build-system")
        .and_then(|b| b.get("requires"))
        .and_then(|r| r.as_array())
    else {
        return Ok(default());
    };
    let mut out = Vec::new();
    for value in requires {
        let Some(item) = value.as_str() else {
            bail!(
                "{}: build-system.requires holds a non-string entry. The build \
                 environment is provisioned from this list verbatim, so a value \
                 that is not a requirement cannot be installed and must not be \
                 guessed at.",
                file.display()
            );
        };
        out.push(item.to_string());
    }
    if out.is_empty() {
        return Ok(default());
    }
    Ok(out)
}

/// The tree's OWN static `[project].dependencies`, when it states them.
///
/// This is the cheap pre-check that keeps a genuinely static editable out of
/// the isolated build entirely: `imprint` and `unitree_sdk2py` carry real
/// `[project]` tables, pixi's frontend prints `Found static pyproject.toml` for
/// them and costs nothing, and there is nothing here to derive. Returning
/// `Some` means the tree STATES its dependencies (an empty vector is such a
/// statement); `None` means it states nothing and the last-resort build is the
/// only remaining reader.
///
/// `dynamic = ["dependencies"]` is treated as stating NOTHING, because that is
/// precisely what it means: the field is computed at build time, which is the
/// case this module exists for.
pub fn static_dependencies(tree: &Path) -> Result<Option<Vec<String>>> {
    let file = tree.join("pyproject.toml");
    let text = match std::fs::read_to_string(&file) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("reading {}", file.display())),
    };
    let parsed: toml::Value = toml::from_str(&text).with_context(|| {
        format!(
            "parsing {} to see whether it states its dependencies statically",
            file.display()
        )
    })?;
    let Some(project) = parsed.get("project") else {
        return Ok(None);
    };
    let dynamic_fields: Vec<&str> = project
        .get("dynamic")
        .and_then(|d| d.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    if dynamic_fields.contains(&"dependencies") {
        return Ok(None);
    }
    let Some(array) = project.get("dependencies").and_then(|d| d.as_array()) else {
        return Ok(None);
    };
    let mut out = Vec::new();
    for value in array {
        let Some(item) = value.as_str() else {
            bail!(
                "{}: [project].dependencies holds a non-string entry. The \
                 generated shim renders this list verbatim, so a value that is \
                 not a requirement must not be guessed at.",
                file.display()
            );
        };
        out.push(item.to_string());
    }
    Ok(Some(out))
}

/// The PEP 517 backend the tree declares, defaulting as PEP 518 prescribes.
pub fn build_backend(tree: &Path) -> Result<String> {
    let file = tree.join("pyproject.toml");
    let default = "setuptools.build_meta:__legacy__".to_string();
    let text = match std::fs::read_to_string(&file) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(default),
        Err(error) => return Err(error).with_context(|| format!("reading {}", file.display())),
    };
    let parsed: toml::Value = toml::from_str(&text)
        .with_context(|| format!("parsing {} to read its build-backend", file.display()))?;
    Ok(parsed
        .get("build-system")
        .and_then(|b| b.get("build-backend"))
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or(default))
}

// ---------------------------------------------------------------------------
// the isolation assertion -- the whole point of the lane
// ---------------------------------------------------------------------------

/// REFUSE an interpreter that lives inside a pixi environment prefix.
///
/// This is not hygiene. A venv parented on `.pixi/envs/pace/bin/python3.11` is
/// exactly what makes pixi's frontend materialise the 14 GB pace prefix inside
/// `resolve_pypi`, which MEASURED 1 295.6 s of a 1 774 s lock. A build env this
/// module provisions must never be the same build env this module exists to
/// avoid, so the property is asserted rather than assumed.
pub fn assert_isolated_interpreter(python: &Path) -> Result<()> {
    let mut previous: Option<String> = None;
    for component in python.components() {
        let Component::Normal(part) = component else {
            previous = None;
            continue;
        };
        let part = part.to_string_lossy().to_string();
        if previous.as_deref() == Some(PIXI_DIR) && part == PIXI_ENVS_DIR {
            bail!(
                "refusing to derive path-source metadata with {}: it is inside a \
                 pixi environment prefix ({PIXI_DIR}/{PIXI_ENVS_DIR}/...). A \
                 build environment parented on the target prefix is the defect \
                 this capability removes -- it is what makes pixi's own frontend \
                 materialise the whole environment inside the resolve timer \
                 (1295.6 s of a 1774 s lock, measured on job 6115467). The \
                 isolated build must use an interpreter outside every \
                 environment prefix.",
                python.display()
            );
        }
        previous = Some(part);
    }
    Ok(())
}

/// The interpreter the isolated build will use.
///
/// `candidates` is the ordered list to consider — production hands it what
/// `PATH` resolves for `python3`, and a guard hands it a fixture. The first
/// candidate that EXISTS and passes [`assert_isolated_interpreter`] wins; a
/// candidate that exists and is prefix-parented is a REFUSAL and not a skip,
/// because silently walking past it is how a fallback gets reinvented.
pub fn select_interpreter(candidates: &[PathBuf]) -> Result<PathBuf> {
    for candidate in candidates {
        if !candidate.exists() {
            continue;
        }
        assert_isolated_interpreter(candidate)?;
        return Ok(candidate.clone());
    }
    bail!(
        "no usable interpreter for the isolated metadata build: none of {} \
         exists. Deriving a path source's metadata needs a python outside every \
         environment prefix; refusing rather than falling back to the frontend's \
         prefix-parented build, which is the 1295.6 s this capability removes.",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

// ---------------------------------------------------------------------------
// the builder seam
// ---------------------------------------------------------------------------

/// What one isolated metadata build produced.
#[derive(Debug, Clone)]
pub struct BuildProduct {
    /// The verbatim core-metadata bytes, as the tree's own backend wrote them.
    pub metadata: String,
    /// The interpreter that ran it.
    pub interpreter: PathBuf,
}

/// The seam between "decide to build" and "run a subprocess".
///
/// It exists because a guard has to be able to drive the WHOLE decision --
/// detection, hashing, store hit/miss, refusal, row -- offline in
/// `cargo test --lib`, and a capability whose only test needs a network and a
/// python is a capability with no guard. Production is
/// [`IsolatedPep517Builder`]; the guards pass a recorder that returns fixture
/// bytes and a fixture interpreter path.
pub trait MetadataBuilder {
    fn build(&self, tree: &Path) -> Result<BuildProduct>;
}

/// Production: a venv outside every environment prefix, the tree's own
/// declared `build-system.requires` installed into it, and the tree's own PEP
/// 517 backend asked for `prepare_metadata_for_build_wheel`.
#[derive(Debug, Clone)]
pub struct IsolatedPep517Builder {
    /// Ordered interpreter candidates. Production fills this from `PATH`.
    pub candidates: Vec<PathBuf>,
    /// Where the venv and the metadata output go. A job-local scratch.
    pub scratch: PathBuf,
}

/// The file the driver writes the METADATA path into.
///
/// **IT IS A FILE AND NOT STDOUT, AND THAT IS A MEASURED FIX, NOT A style
/// choice.** The first version of this wrote the path to stdout and the caller
/// trimmed it. The isolation run (job 6132200) then read
/// `### METAPATH pace='running dist_info` — setuptools writes its own build log
/// (`running dist_info`, `creating …`) to STDOUT, so the "path" the caller got
/// was setuptools' first log line and the metadata read failed with
/// `meta_rc=0`: a SUCCESSFUL build reported as a missing file. A build backend
/// is arbitrary code and its stdout belongs to it; the one datum the caller
/// needs therefore travels by a channel the backend has no reason to touch.
pub const METADATA_PATH_FILE: &str = "metagen-metadata-path.txt";

/// The PEP 517 driver, run by the isolated venv's own interpreter.
///
/// It is a fixed string and takes the backend name, the output directory and the
/// path-file as argv, so nothing about a tree is ever interpolated into python
/// source.
pub const PREPARE_METADATA_DRIVER: &str = r#"
import importlib, os, sys

backend_spec, out_dir, path_file = sys.argv[1], sys.argv[2], sys.argv[3]
module_name, _, attribute = backend_spec.partition(":")
backend = importlib.import_module(module_name)
for part in attribute.split(".") if attribute else []:
    backend = getattr(backend, part)
prepare = getattr(backend, "prepare_metadata_for_build_wheel", None)
if prepare is None:
    sys.stderr.write("backend %s has no prepare_metadata_for_build_wheel\n" % backend_spec)
    raise SystemExit(3)
name = prepare(out_dir)
with open(path_file, "w") as handle:
    handle.write(os.path.join(out_dir, name, "METADATA"))
"#;

impl MetadataBuilder for IsolatedPep517Builder {
    fn build(&self, tree: &Path) -> Result<BuildProduct> {
        let interpreter = select_interpreter(&self.candidates)?;
        let requires = build_requires(tree)?;
        let backend = build_backend(tree)?;

        std::fs::create_dir_all(&self.scratch)
            .with_context(|| format!("creating {}", self.scratch.display()))?;
        let venv = self.scratch.join("metagen-build-env");
        if venv.exists() {
            std::fs::remove_dir_all(&venv)
                .with_context(|| format!("removing a stale {}", venv.display()))?;
        }

        run(
            std::process::Command::new(&interpreter)
                .arg("-m")
                .arg("venv")
                .arg(&venv),
            "creating the isolated build environment",
        )?;

        let venv_python = venv.join("bin").join("python");
        // The venv's own interpreter is a NEW path: assert it too, so a venv
        // created inside a prefix by a mis-set scratch is caught as well.
        assert_isolated_interpreter(&venv_python)?;

        let mut install = std::process::Command::new(&venv_python);
        install
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("--disable-pip-version-check")
            .arg("--no-input");
        for requirement in &requires {
            install.arg(requirement);
        }
        run(
            &mut install,
            "installing the tree's declared build-system.requires into the \
             isolated build environment",
        )?;

        let out_dir = self.scratch.join("metagen-metadata");
        if out_dir.exists() {
            std::fs::remove_dir_all(&out_dir)
                .with_context(|| format!("removing a stale {}", out_dir.display()))?;
        }
        std::fs::create_dir_all(&out_dir)
            .with_context(|| format!("creating {}", out_dir.display()))?;

        let path_file = self.scratch.join(METADATA_PATH_FILE);
        let _ = std::fs::remove_file(&path_file);
        let output = std::process::Command::new(&venv_python)
            .arg("-c")
            .arg(PREPARE_METADATA_DRIVER)
            .arg(&backend)
            .arg(&out_dir)
            .arg(&path_file)
            .current_dir(tree)
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env("PYTHONNOUSERSITE", "1")
            .output()
            .with_context(|| {
                format!(
                    "running prepare_metadata_for_build_wheel for {}",
                    tree.display()
                )
            })?;
        if !output.status.success() {
            bail!(
                "the isolated PEP 517 metadata build of {} FAILED ({}). \
                 stderr:\n{}\nThis refuses instead of falling back to the \
                 frontend's prefix-parented build: that fallback is the \
                 1295.6 s this capability exists to remove, and taking it \
                 silently would hide the very failure that needs fixing.",
                tree.display(),
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        // The path comes from the FILE, never from stdout — see
        // [`METADATA_PATH_FILE`] for the run that proved why.
        let metadata_path = PathBuf::from(
            std::fs::read_to_string(&path_file)
                .with_context(|| {
                    format!(
                        "the isolated build of {} exited 0 but wrote no metadata \
                         path to {}. Its own stdout was:\n{}\nA backend that \
                         reports success and produces nothing is a refusal here, \
                         not a fallback.",
                        tree.display(),
                        path_file.display(),
                        String::from_utf8_lossy(&output.stdout).trim()
                    )
                })?
                .trim()
                .to_string(),
        );
        let metadata = std::fs::read_to_string(&metadata_path).with_context(|| {
            format!(
                "reading the METADATA the isolated build of {} wrote at {}",
                tree.display(),
                metadata_path.display()
            )
        })?;
        Ok(BuildProduct {
            metadata,
            interpreter: venv_python,
        })
    }
}

fn run(command: &mut std::process::Command, what: &str) -> Result<()> {
    let output = command
        .output()
        .with_context(|| format!("{what}: spawning {:?}", command.get_program()))?;
    if !output.status.success() {
        bail!(
            "{what} FAILED ({}). stderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// the store
// ---------------------------------------------------------------------------

/// The store root: whatever was named, else
/// [`crate::courier::persistent_store_root_with`] and nothing else. There is
/// deliberately no `RETREAD_CACHE_DIR` branch — that absence is the property
/// that makes the store outlive the job, and it is the whole reason a second
/// lock in the same week does not rebuild this metadata.
pub fn store_root_with(configured: Option<&Path>, env: &dyn Fn(&str) -> Option<String>) -> PathBuf {
    match configured {
        Some(root) => root.to_path_buf(),
        None => crate::courier::persistent_store_root_with(env),
    }
}

/// `<root>/path-source-metadata/v1` — the generation directory the entries sit
/// in, the same shape as the build-requirements, hermetic and sdist-metadata
/// stores to the segment.
pub fn generation_dir(store_root: &Path) -> PathBuf {
    store_root.join(CACHE_NAMESPACE).join(CACHE_VERSION)
}

/// `<root>/path-source-metadata/v1/<key>`.
pub fn entry_dir(store_root: &Path, key: &str) -> PathBuf {
    generation_dir(store_root).join(key)
}

/// The entry key: [`KEY_SCHEMA`] over the tree's bounded content hash and the
/// interpreter tag the build ran under, both LENGTH-PREFIXED, rendered
/// `psm-<hex>`.
///
/// Length-prefixed for the reason the hermetic and sdist-metadata keys are:
/// without it `("ab", "c")` and `("a", "bc")` hash identically, and two trees
/// that differ only in where a boundary falls would share an entry.
///
/// The TAG IS IN THE KEY. The two live trees' `setup.py` files branch on
/// nothing, so their metadata is expected to be identical under any
/// interpreter — but tag-invariance of two trees is not a property of the
/// MECHANISM: a `setup.py` is arbitrary code and may branch on
/// `sys.version_info`. An extra entry per tag costs one small file; a shared
/// entry across tags that turns out to be wrong costs a wrong lock.
pub fn entry_key(source_hash: &str, python_tag: &str) -> String {
    let mut identity = Sha256::new();
    identity.update(KEY_SCHEMA.as_bytes());
    identity.update([0u8]);
    for field in [source_hash, python_tag] {
        identity.update((field.len() as u64).to_be_bytes());
        identity.update(field.as_bytes());
    }
    format!("{KEY_PREFIX}{:x}", identity.finalize())
}

/// Read a published entry back, or `None` when there is none.
///
/// An entry is only readable when its completion marker is present: the marker
/// is written LAST, so a half-written entry from a killed job reads as absent
/// rather than as truncated metadata.
pub fn read_entry(store_root: &Path, key: &str) -> Result<Option<String>> {
    let dir = entry_dir(store_root, key);
    if !dir.join(COMPLETION_MARKER).is_file() {
        return Ok(None);
    }
    let file = dir.join(METADATA_FILE);
    match std::fs::read_to_string(&file) {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("reading {}", file.display())),
    }
}

/// Publish one entry: the metadata first, the marker LAST.
pub fn publish_entry(store_root: &Path, key: &str, metadata: &str, provenance: &str) -> Result<()> {
    let dir = entry_dir(store_root, key);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let metadata_file = dir.join(METADATA_FILE);
    std::fs::write(&metadata_file, metadata)
        .with_context(|| format!("writing {}", metadata_file.display()))?;
    // The marker is written LAST and is what makes the entry readable.
    let marker = dir.join(COMPLETION_MARKER);
    std::fs::write(&marker, provenance)
        .with_context(|| format!("writing {}", marker.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// the capability
// ---------------------------------------------------------------------------

/// Derive one dynamic path source's core metadata: store first, isolated build
/// second, loud refusal third.
///
/// `python_tag` keys the entry alongside the tree hash; production passes the
/// interpreter's `cp3XX`-style tag and a guard passes a fixture.
pub fn derive_for_tree(
    tree: &Path,
    options: &DeriveOptions,
    python_tag: &str,
    builder: &dyn MetadataBuilder,
) -> Result<Derived> {
    let started = Instant::now();
    let hash = source_hash(tree)?;
    let key = entry_key(&hash, python_tag);

    if let Some(root) = options.store_root.as_deref()
        && let Some(text) = read_entry(root, &key)?
    {
        return Ok(Derived {
            metadata: parse_core_metadata(&text),
            source_hash: hash,
            store: StoreOutcome::Hit,
            // A hit did not run a build, so there is no interpreter to report.
            // The row still says `build_env=isolated`, which is true of the
            // entry: nothing prefix-parented can ever be published here.
            interpreter: PathBuf::new(),
            wall_seconds: started.elapsed().as_secs_f64(),
        });
    }

    let product = builder.build(tree).with_context(|| {
        format!(
            "deriving static metadata for the dynamic path source at {} ({CONFIG_KEY})",
            tree.display()
        )
    })?;
    // The builder is a seam, so the isolation property is asserted HERE too --
    // on the interpreter the build says it used. A builder that returns a
    // prefix-parented interpreter is the mutation this refusal exists to catch.
    assert_isolated_interpreter(&product.interpreter)?;

    let metadata = parse_core_metadata(&product.metadata);
    let store = match options.store_root.as_deref() {
        None => StoreOutcome::Miss,
        Some(root) => {
            let provenance = format!(
                "{{\"source_hash\":\"{hash}\",\"python_tag\":\"{python_tag}\",\"tree\":\"{}\"}}\n",
                tree.display()
            );
            publish_entry(root, &key, &product.metadata, &provenance)?;
            StoreOutcome::Built
        }
    };

    Ok(Derived {
        metadata,
        source_hash: hash,
        store,
        interpreter: product.interpreter,
        wall_seconds: started.elapsed().as_secs_f64(),
    })
}

// ---------------------------------------------------------------------------
// the production adapter -- the ONE place the config becomes a reader
// ---------------------------------------------------------------------------

/// The interpreter identity that keys an entry alongside the tree hash.
///
/// It is the interpreter's canonical path plus its size and mtime, and NOT the
/// `cp3XX` tag a subprocess would report, for one reason: computing the key
/// must not cost an exec on every lock, including the locks that HIT. Path
/// alone would be wrong — `/usr/bin/python3` can be upgraded under the same
/// name — so the stat is what makes an in-place upgrade a new entry.
pub fn interpreter_tag(python: &Path) -> Result<String> {
    let canonical = python
        .canonicalize()
        .with_context(|| format!("canonicalizing {}", python.display()))?;
    let meta = std::fs::metadata(&canonical)
        .with_context(|| format!("stat {}", canonical.display()))?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Ok(format!(
        "{}:{}:{}",
        canonical.display(),
        meta.len(),
        mtime
    ))
}

/// Interpreter names to look for in each `PATH` directory, NEWEST FIRST.
///
/// The order is measured, not stylistic. On this cluster the only interpreter on
/// `PATH` is `/usr/bin/python3` = **Python 3.9.21**, while `pace_sim2real`
/// declares `requires-python = ">=3.10"`. A metadata build under 3.9 does in
/// fact succeed — `requires-python` is metadata, not something setuptools
/// enforces at `prepare_metadata_for_build_wheel` time, and the isolation run
/// confirmed the only failure under 3.9 was a missing `toml`, which the declared
/// `build-system.requires` supplies. But a `setup.py` is arbitrary code and may
/// use syntax newer than the oldest interpreter on the box, so a newer one is
/// preferred WHEN PRESENT rather than depended on.
///
/// Bare `python` is last: on this platform it is as likely to be absent as to be
/// a python 2, and a candidate that exists but is wrong is worse here than one
/// that is missing.
pub const INTERPRETER_NAMES: [&str; 7] = [
    "python3.13",
    "python3.12",
    "python3.11",
    "python3.10",
    "python3.9",
    "python3",
    "python",
];

/// Ordered interpreter candidates from `PATH`: newest name first within each
/// directory, directories in `PATH` order.
pub fn path_interpreter_candidates(path_var: Option<&str>) -> Vec<PathBuf> {
    let Some(path_var) = path_var else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for dir in path_var.split(':') {
        if dir.is_empty() {
            continue;
        }
        for name in INTERPRETER_NAMES {
            out.push(Path::new(dir).join(name));
        }
    }
    out
}

/// The configured last-resort reader: the switch, the store and the builder,
/// behind the one trait the derivation consults.
#[derive(Debug, Clone)]
pub struct ConfiguredDeriver {
    pub options: DeriveOptions,
    pub builder: IsolatedPep517Builder,
}

impl ConfiguredDeriver {
    /// Build it from the backend config, or `None` when the switch is off.
    ///
    /// `None` is the `retread-derive-editable-metadata = false` arm and is what
    /// the derivation is handed to reproduce the pre-METAGEN-1 behaviour
    /// exactly — no build, no store, no row.
    pub fn from_config(
        enabled: Option<bool>,
        configured_store: Option<&Path>,
        env: &dyn Fn(&str) -> Option<String>,
        scratch: PathBuf,
    ) -> Option<Self> {
        let options = DeriveOptions::from_config(
            enabled,
            Some(store_root_with(configured_store, env)),
        );
        if !options.enabled {
            return None;
        }
        Some(Self {
            options,
            builder: IsolatedPep517Builder {
                candidates: path_interpreter_candidates(env("PATH").as_deref()),
                scratch,
            },
        })
    }
}

impl crate::path_source_metadata::DynamicDependencySource for ConfiguredDeriver {
    fn derive(&self, tree: &Path) -> Result<Derived> {
        let interpreter = select_interpreter(&self.builder.candidates)?;
        let tag = interpreter_tag(&interpreter)?;
        derive_for_tree(tree, &self.options, &tag, &self.builder)
    }
}

/// The declared key that says a pack takes NO generated record writes.
///
/// **It is new.** The 09-04 path-source-metadata design has no such key:
/// `grep -rniE 'clean.only|no.generated|generated.writes|read.only.pack'` over
/// `src/` returns nothing, and what that design has instead is
/// [`crate::path_source_metadata::write_under_pack`], which refuses any write
/// OUTSIDE the pack but places no bound on writes inside it. So "this pack is
/// clean-only" had no way to be said, and a lane that needed it would have had
/// to invent it silently. This names it once, and the refusal that reads it
/// names the `path-source-refresh --write` verb that produces the record by
/// hand instead.
pub const GENERATED_RECORDS_KEY: &str = "retread-path-source-generated-records";
