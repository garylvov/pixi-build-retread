//! v2.0.0 courier: the committed install lock.
//!
//! Records every resolution decision the backend made so a consumer can
//! install the bundle's PyPI wheels at link time (via `retread install`,
//! invoked from the conda package's post-link script) WITHOUT re-running
//! the cascade and WITHOUT committing any wheels to git:
//!
//! - `built` wheels (git/path sources, on no index) ship inside the conda
//!   package under `share/retread/wheels/<filename>` and install from
//!   there.
//! - `index` wheels (isaacsim, nvidia, ... -- the multi-GB bulk) are NOT
//!   shipped; they install by fetching their recorded `url` at link time,
//!   on uv's fast hardlink path.
//!
//! The lock is small (KB), human-diffable, and committed next to the pack
//! manifest as `retread-<bundle>.lock.json`. It is the single source of
//! truth the installer reads.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::relax::canonical_conda_name;

/// Git provenance for a source-built wheel (schema 8+).
///
/// Stored in `LockWheel.git_source` for every wheel whose bytes were
/// produced by `build_wheel_from_git`. Carries the RESOLVED 40-char commit
/// SHA (never a branch name or "HEAD") so replay can re-clone the identical
/// commit without reading the live `[retread-wheels]` manifest entry.
///
/// ## POISONING note
///
/// The `rev` field is the RESOLVED SHA at produce time. Replay pins this
/// SHA and builds a byte-stable wheel from it. Only a cascade re-solve
/// (full cold produce) picks up a new branch tip. This is correct: the
/// lock is the contract; moving a branch ref does NOT invalidate the lock
/// (the inputs_hash covers the config rev, not the resolved SHA — adding
/// the resolved SHA to the hash would be circular).
///
/// ## Phase-2 limitation — single-entry per checkout root
///
/// This struct is safe for the Class-1 single-entry case (one
/// `[retread-wheels]` entry per git checkout root). A future multi-entry
/// shared-checkout bundle (e.g. two entries from the same repo at different
/// subdirectories, which would produce a non-trivial `skip_subdirs` at
/// produce time) is NOT covered: the replay synth uses `skip_subdirs=[]`
/// and would produce a non-byte-identical wheel (it over-ships files the
/// produce-time build excluded). This is guarded at replay time (see
/// `materialize_from_lock`) and deferred to Phase 3.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitWheelSource {
    /// Canonical git URL (without the `git+` prefix — the bare clone URL).
    pub url: String,
    /// RESOLVED 40-character commit SHA (never a branch/tag/HEAD ref).
    /// This is `git rev-parse HEAD` after checkout, not the original
    /// `rev` string from the config entry.
    pub rev: String,
    /// Subdirectory within the repo where the Python package lives,
    /// relative to the repo root. `None` means the root (equiv. to ".").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subdirectory: Option<String>,
    /// Extras requested on the originating `[retread-wheels]` entry
    /// (e.g. `["sim"]` for `newton = { from="newton", extras=["sim"] }`).
    /// Not passed to `build_wheel_from_git` (extras drive BFS closure
    /// expansion only, not the wheel build), so the replay can safely
    /// build the synth entry with these extras for BFS parity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extras: Vec<String>,
}

/// Provenance for a wheel built locally from a PyPI sdist because the
/// index ships no target-compatible wheel (e.g. gym). On replay,
/// `materialize_from_lock` re-builds DIRECTLY from the recorded `sdist_url`
/// (the exact resolved https tarball + #sha256), falling back to
/// `resolve_sdist(index, name, version)` only if that URL 404s ->
/// deterministic, portable, manifest-independent. `None` for index-fetched
/// and git-built wheels.
///
/// ## POISONING note
///
/// Like `git_source.rev`, `sdist_source` is NOT folded into `inputs_hash`
/// (same circularity argument: it is a RESULT of resolution, not an input to
/// it; folding it would require resolving to compute the hash that gates
/// resolution). Replay reproduces the RECORDED sdist artifact verbatim. With
/// the stored https URL + #sha256, the only residual risk is "the artifact was
/// deleted/yanked from PyPI" — the same documented pinning floor as every
/// other replay-trusted upstream (git commit, index wheel). The #sha256 is a
/// free integrity check on re-fetch and is likewise NOT in `inputs_hash`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SdistWheelSource {
    /// PEP 503 simple index base URL the sdist was resolved from.
    /// Human-readable provenance + fallback re-resolution key.
    pub index: String,
    /// PEP 503 normalized project name. Fallback re-resolution key.
    pub name: String,
    /// Resolved version (from the built wheel's METADATA). Fallback key.
    pub version: String,
    /// The EXACT resolved sdist URL (e.g.
    /// `https://files.pythonhosted.org/.../<name>-<version>.tar.gz#sha256=<hex>`).
    /// PREFERRED on replay: build straight from this, skipping a re-resolve.
    /// Carries the PEP-503 #sha256 fragment when the index advertised one.
    pub sdist_url: String,
}

/// How a wheel reaches the consumer at install time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Origin {
    /// Shipped inside the conda package (built from a git/path source, on
    /// no index). Installed from `share/retread/wheels/<filename>`.
    Built,
    /// Fetched from `url` at link time (lives on a public/extra index).
    Index,
}

/// One wheel in the bundle's install set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockWheel {
    pub name: String,
    pub version: String,
    pub origin: Origin,
    /// Standardized wheel filename (the basename installed/shipped).
    pub filename: String,
    /// Fetch URL for `Origin::Index` wheels; `None` for `Origin::Built`.
    /// (Index wheels fetch this at install time; Built wheels ship in-package.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// sha256 of the wheel file (index-wheel verification; reproducibility).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// POST-relax Requires-Dist lines recorded for the replay poisoning guard
    /// (schema 5+). Empty when the wheel is `Origin::Index` and unrelaxed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requires_dist: Vec<String>,
    /// True iff this wheel must be shipped inside the conda package
    /// (i.e. it carries the `.injected` infix — it exists on no index).
    /// Recorded so the replay path can cross-check re-materialized wheels
    /// without re-running the full must_ship() filename heuristic.
    #[serde(default)]
    pub must_ship: bool,
    /// Original upstream PyPI URL for `Origin::Built` relax-changed shadow
    /// wheels (schema 6+). These are index wheels whose METADATA was rewritten
    /// by the relax pipeline; they have no `.injected` infix and `must_ship`
    /// is false, but their bytes are NOT available on the original index
    /// (the rewritten shadow is shipped). On replay, `materialize_from_lock`
    /// uses this URL to re-fetch the original bytes and re-apply the relax
    /// rewrite, recreating the shadow without re-running the full BFS/solve.
    ///
    /// `None` for `Origin::Index` wheels (use `url`) and for `Origin::Built`
    /// wheels with `must_ship=true` (source-built; re-materialized from
    /// `[retread-wheels]` config entry or `git_source`, no upstream URL).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_url: Option<String>,
    /// Git provenance for source-built wheels (schema 8+). Present when this
    /// wheel was built from a git source (either a named `[retread-git-sources]`
    /// entry or an inline `git=` entry in `[retread-wheels]`). Carries the
    /// resolved 40-char commit SHA and the clone URL so replay can re-source-
    /// build the wheel without reading the live manifest. `None` for index
    /// wheels and non-git source-built wheels (path/from/url forms).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_source: Option<GitWheelSource>,
    /// Sdist provenance for wheels built from a PyPI sdist (schema 9+).
    /// Present when this wheel was built from an sdist because the index
    /// ships no target-compatible wheel (e.g. gym). Carries the exact
    /// resolved sdist https URL + #sha256 so replay can re-build from the
    /// same tarball manifest-independently. `None` for index-fetched and
    /// git-built wheels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sdist_source: Option<SdistWheelSource>,
}

/// A conda run-dep retread routed to the conda side (a shared transitive
/// the consumer's conda solve provides). Carried so the courier conda
/// package can declare them as its own run-deps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CondaDep {
    pub name: String,
    pub spec: String,
}

/// The committed install lock for one bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetreadLock {
    pub schema: u32,
    pub retread_version: String,
    pub bundle: String,
    pub version: String,
    /// Target python (e.g. "3.11"); the wheel set is python-specific.
    pub python: String,
    /// sha256 over the canonicalized resolution inputs (entry specs +
    /// ordered index chain + relax policy + python + retread version).
    /// Reproducibility gate (req #5) AND the cold-solve replay key (req #4):
    /// `conda/outputs` replays this lock's emitted outputs (skipping the
    /// probe cascade) iff a freshly computed inputs hash matches this.
    #[serde(default)]
    pub inputs_hash: String,
    /// The PEP 508 requirements `retread install` hands to uv (the meta
    /// requirement that drives the closure -- typically the bundle entries
    /// pinned to their resolved versions). uv resolves these against the
    /// shipped find-links wheels + `index_urls`, into the active conda env
    /// (preferring conda-installed dists, so shared transitives stay conda).
    #[serde(default)]
    pub root_requirements: Vec<String>,
    /// Wheels to install at link time (built ship in pkg + index fetched).
    pub wheels: Vec<LockWheel>,
    /// Shared transitives routed to conda (the courier package's run-deps).
    pub conda_run_deps: Vec<CondaDep>,
    /// Index chain for fetching `Origin::Index` wheels + any prerelease
    /// deps, in priority order (entry indexes, then public PyPI).
    pub index_urls: Vec<String>,
    /// Prerelease pins (`name` -> `==X`) uv needs to opt those deps into
    /// prerelease resolution (it only honors them from direct reqs +
    /// overrides). Empty when the bundle pins no prereleases.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub prerelease: BTreeMap<String, String>,
    /// PyPI package names (conda-normalized) that have a conda counterpart
    /// and are therefore routed to the conda side rather than bundled in the
    /// wheel set. Recorded for the build_v1 replay path so it can reconstruct
    /// the correct conda_capable set without re-running the probe cascade.
    /// Sorted for stable JSON output.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conda_capable: Vec<String>,
    /// Canonical resolution-INPUT specs for this bundle, as produced by
    /// `courier_input_specs(config, bundle_name)` — one sorted entry per
    /// `[retread-wheels]` entry in this bundle (key + optional [extras] +
    /// optional version/git-rev/url proxy). Written at lock-produce time;
    /// read by the Part-2 incremental fast-path delta-detector to determine
    /// whether the current manifest diff is a single-dep add.
    ///
    /// NOT in `compute_inputs_hash` (it is the thing the delta-detector
    /// diffs; folding it would make the hash circular). Old schema-9 and
    /// earlier locks lack this field — `#[serde(default)]` returns `vec![]`,
    /// which the incremental path treats as "no prior entry_specs → can't
    /// compute delta → fall back to full cold resolve" (safe).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entry_specs: Vec<String>,
}

/// Schema 10: canonical lock ordering + `entry_specs` field.
///
/// Canonical ordering (`RetreadLock::canonicalize()`):
/// `wheels[]` sorted by (canonical_conda_name, version, origin, filename);
/// `conda_run_deps[]` by (name, spec); `root_requirements[]` and
/// `conda_capable[]` lexicographically; nested `requires_dist[]` and
/// `GitWheelSource.extras[]` lexicographically. This guarantees byte-identical
/// JSON regardless of resolve/discovery order, making lock diffs meaningful.
///
/// New field: `entry_specs: Vec<String>` — the `courier_input_specs` snapshot
/// for the Part-2 incremental delta-detector. `#[serde(default)]` so old locks
/// parse (delta-detector falls back to full cold resolve on empty `entry_specs`).
///
/// Old schema-9 and earlier locks are rejected by the != gate and fall through
/// to full resolve (safe: committed locks must be regenerated). SCHEMA is NOT
/// an epoch bump (output SEMANTICS for identical inputs are unchanged;
/// [emit-epoch-ok]).
pub const SCHEMA: u32 = 10;

/// Bump EMIT_EPOCH in the SAME commit as ANY change that can alter the bytes
/// retread emits for identical manifest inputs (relax/version-selection
/// algorithm; wheel-rewrite logic incl. RECORD/metadata/dist-info/layout;
/// auto-bundle selection or fetching; meta-wheel / courier package format;
/// conda recipe metadata; index-chain merge order; the compute_inputs_hash
/// domain prefix or folded fields). Do NOT bump for docs/tests/logging/
/// refactors with identical output, dep bumps that don't change emitted bytes,
/// or SCHEMA-only lock-format changes. When in doubt, bump it: a needless bump
/// costs one cold solve per pack (cheap, self-healing); a missed bump causes
/// silent stale-cache reuse for consumers (expensive, invisible).
/// SCHEMA = on-disk lock FORMAT; EMIT_EPOCH = emitted-output SEMANTICS for
/// identical inputs -- bump independently.
///
/// Epoch 3: pack-scoped solve_fingerprint -- `inputs_hash` now hashes only the
/// envs that reference the source pack (via discover_outputs_for_source) rather
/// than all envs in the workspace, eliminating false cache misses when unrelated
/// envs change.
///
/// Epoch 4: force-downloaded remote-only relax-changed wheels now go through
/// rewrite_wheel_with (relaxed bytes) instead of being renamed raw (the
/// starts_with(staging_dir) heuristic bug fix). Any pack with a remote-only
/// auto-bundled dep whose Requires-Dist the relax policy changes will emit
/// different (correctly relaxed) shadow bytes.
///
/// Epoch 5: orphan direct-URL Requires-Dist lines (target absent from the
/// resolved bundle closure) are now STRIPPED from emitted wheel METADATA.
/// Fixes uv aborting on bundles that include a wheel with an unconditional
/// git-URL dep not in the bundle (e.g. isaaclab_mimic 1.3.2 robomimic line).
/// Any pack whose wheels carry such orphan URL lines (both isaac-pack and
/// isaac-pack-latest) will emit different (correctly stripped) wheel bytes.
///
/// Epoch 6: confluent constraint-accumulating BFS resolver (Part 1, amendment B):
/// all four version-picking sites now accumulate AND-intersect specifiers and
/// iterate to a name-sorted fixpoint. For the committed locks this produces
/// identical resolved versions (pre-probe: 0 divergences), but the algorithm
/// change is semantically different and gets its own epoch for safety.
///
/// Epoch 7: sibling-aware resolution.  A dep whose canonical name matches
/// another entry in the same bundle group is a "sibling" — provided by that
/// sibling's wheel at install time.  Such deps are now silently dropped in
/// seed_worklist and defended against in the BFS frontier, so they are neither
/// fetched from PyPI nor emitted as run-deps.  This changes resolution
/// semantics for packs with multiple git-source entries (e.g. isaaclab +
/// isaaclab-visualizers from the same repo) where a sibling happens to be
/// listed in Requires-Dist.
///
/// Epoch 8: courier post-link failures are fatal, and the activate guard now
/// verifies both the success marker and installed wheel metadata. Identical
/// manifests emit different post-link/activate scripts, so stale courier
/// packages must be rebuilt.
pub const EMIT_EPOCH: u32 = 8;

impl RetreadLock {
    /// File name for a bundle's lock next to the pack manifest.
    pub fn file_name(bundle: &str) -> String {
        format!("retread-{bundle}.lock.json")
    }

    /// Canonical inputs hash. EVERY producer (the courier staging that
    /// writes the lock) and replayer (`conda/outputs` deciding whether to
    /// skip the cascade) MUST call this -- never hand-roll the digest, or
    /// the two sides disagree and replay silently never fires (or fires
    /// stale). Entry specs are sorted so ordering is not significant; the
    /// index chain order IS significant (it is resolution priority).
    ///
    /// `config_fingerprint` folds in EVERY remaining resolution-affecting
    /// manifest input that is not already covered above -- overrides,
    /// name-map, drop-deps, conda-deps, auto-bundle, build-number, and the
    /// conda channel list. Without it, editing any of those (e.g. genesis's
    /// `retread-name-map`) would leave the hash unchanged and a stale,
    /// POISONED lock would replay. It is produced by the shared
    /// `courier::config_fingerprint` so producer and replayer always agree.
    ///
    /// The hash folds the emit epoch (+ the exact retread version when
    /// `retread-pin-version` is set). Routine retread upgrades that do NOT
    /// change emitted bytes for identical inputs only bump `EMIT_EPOCH` when
    /// the output semantics change, so existing committed locks replay without
    /// a cold re-solve on every upgrade.
    pub fn compute_inputs_hash(
        entry_specs: &[String],
        index_urls: &[String],
        relax: &str,
        python: &str,
        emit_epoch: u32,
        pin_version: Option<&str>,
        config_fingerprint: &str,
    ) -> String {
        use sha2::{Digest, Sha256};
        let mut sorted = entry_specs.to_vec();
        sorted.sort();
        let mut h = Sha256::new();
        h.update(b"retread-inputs-v5\n");
        for s in &sorted {
            h.update(s.as_bytes());
            h.update(b"\n");
        }
        h.update(b"--indexes--\n");
        for u in index_urls {
            h.update(u.as_bytes());
            h.update(b"\n");
        }
        h.update(b"--meta--\n");
        h.update(relax.as_bytes());
        h.update(b"\n");
        h.update(python.as_bytes());
        h.update(b"\n");
        h.update(b"--epoch--\n");
        h.update(emit_epoch.to_le_bytes());
        if let Some(v) = pin_version {
            h.update(b"--pinver--\n");
            h.update(v.as_bytes());
        }
        h.update(b"\n--config--\n");
        h.update(config_fingerprint.as_bytes());
        format!("{:x}", h.finalize())
    }

    /// Read a lock back (consumer side / cold-start replay).
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("reading lock {}", path.display()))?;
        serde_json::from_slice(&bytes).with_context(|| format!("parsing lock {}", path.display()))
    }

    /// Serialize to pretty JSON for committing.
    pub fn to_pretty_json(&self) -> Result<String> {
        // Trailing newline: the committed lock is a normal text file, and the
        // repo's end-of-file-fixer pre-commit hook requires one (a no-newline
        // JSON tripped CI). serde_json::to_string_pretty omits it, so add it.
        let mut s = serde_json::to_string_pretty(self).context("serializing retread lock")?;
        s.push('\n');
        Ok(s)
    }

    /// Idempotency marker file (under `<prefix>/share/retread/`). The
    /// installer writes the lock's content hash here on success and
    /// no-ops when it already matches -- safe to re-run on every relink.
    pub fn marker_name(&self) -> String {
        format!("{}.installed", self.bundle)
    }

    /// Sort all vectors in the lock to their canonical (order-independent)
    /// form so serialized JSON is byte-identical regardless of resolve order.
    /// Called once at serialize time in `courier::stage` (the only write path).
    /// Idempotent: `canonicalize(canonicalize(x)) == canonicalize(x)`.
    ///
    /// Top-level sorts:
    /// - `wheels[]`: (canonical_conda_name(name), version, origin, filename)
    /// - `conda_run_deps[]`: (name, spec)
    /// - `root_requirements[]`: lexicographic
    /// - `conda_capable[]`: lexicographic (supersedes the inline sort in courier::stage)
    /// - `index_urls[]`: NOT sorted — chain order is semantically significant.
    /// - `prerelease`: BTreeMap — already key-ordered.
    ///
    /// Nested sorts (A-1):
    /// - `LockWheel.requires_dist[]`: lexicographic (order-insensitive per §A.5)
    /// - `GitWheelSource.extras[]`: lexicographic
    pub fn canonicalize(&mut self) {
        // Top-level: wheels sorted by (canonical name, version, origin, filename).
        self.wheels.sort_by(|a, b| {
            canonical_conda_name(&a.name)
                .cmp(&canonical_conda_name(&b.name))
                .then_with(|| a.version.cmp(&b.version))
                .then_with(|| {
                    // Origin discriminant: Built < Index for stable ordering.
                    let ord_a = match a.origin {
                        Origin::Built => 0u8,
                        Origin::Index => 1u8,
                    };
                    let ord_b = match b.origin {
                        Origin::Built => 0u8,
                        Origin::Index => 1u8,
                    };
                    ord_a.cmp(&ord_b)
                })
                .then_with(|| a.filename.cmp(&b.filename))
        });

        // Top-level: conda_run_deps sorted by (name, spec).
        self.conda_run_deps
            .sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.spec.cmp(&b.spec)));

        // Top-level: root_requirements lexicographic.
        self.root_requirements.sort();

        // Top-level: conda_capable lexicographic.
        self.conda_capable.sort();

        // Top-level: entry_specs lexicographic (already sorted by
        // courier_input_specs, but sort here defensively so the invariant
        // is enforced in one place regardless of producer).
        self.entry_specs.sort();

        // Nested A-1: requires_dist and GitWheelSource.extras per wheel.
        for wheel in &mut self.wheels {
            wheel.requires_dist.sort();
            if let Some(gs) = &mut wheel.git_source {
                gs.extras.sort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_roundtrips() {
        let lock = RetreadLock {
            schema: SCHEMA,
            retread_version: "2.0.0".into(),
            bundle: "isaac-pack".into(),
            version: "5.1.0".into(),
            python: "3.11".into(),
            inputs_hash: "deadbeef".into(),
            root_requirements: vec!["isaac-pack-pypi==5.1.0".into()],
            wheels: vec![
                LockWheel {
                    // must_ship=true: source-built, no upstream URL.
                    name: "isaaclab".into(),
                    version: "0.51.1".into(),
                    origin: Origin::Built,
                    filename: "isaaclab-0.51.1-py3-none-any.injected.whl".into(),
                    url: None,
                    sha256: None,
                    requires_dist: vec!["numpy>=1.21".into(), "torch>=2.0".into()],
                    must_ship: true,
                    upstream_url: None,
                    git_source: None,
                    sdist_source: None,
                },
                LockWheel {
                    // relax-changed shadow: upstream_url records where to
                    // re-fetch the original for replay re-materialization.
                    name: "skrl".into(),
                    version: "2.1.0".into(),
                    origin: Origin::Built,
                    filename: "skrl-2.1.0-999retread-py3-none-any.whl".into(),
                    url: None,
                    sha256: None,
                    requires_dist: vec!["torch>=2.0,<3".into()],
                    must_ship: false,
                    upstream_url: Some("https://files.pythonhosted.org/skrl-2.1.0.whl".into()),
                    git_source: None,
                    sdist_source: None,
                },
                LockWheel {
                    name: "isaacsim-core".into(),
                    version: "5.1.0.0".into(),
                    origin: Origin::Index,
                    filename: "isaacsim_core-5.1.0.0-cp311-...whl".into(),
                    url: Some("https://pypi.nvidia.com/isaacsim-core/...whl".into()),
                    sha256: Some("abc123".into()),
                    requires_dist: vec![],
                    must_ship: false,
                    upstream_url: None,
                    git_source: None,
                    sdist_source: None,
                },
            ],
            conda_run_deps: vec![CondaDep {
                name: "torchaudio".into(),
                spec: ">=2.7,<3".into(),
            }],
            index_urls: vec![
                "https://pypi.nvidia.com".into(),
                "https://pypi.org/simple/".into(),
            ],
            prerelease: BTreeMap::from([("gmpy2".into(), "==2.1.0a4".into())]),
            conda_capable: vec!["numpy".into(), "torch".into()],
            entry_specs: vec!["isaaclab==0.51.1".into()],
        };
        let json = lock.to_pretty_json().unwrap();
        let back: RetreadLock = serde_json::from_str(&json).unwrap();
        assert_eq!(back.bundle, "isaac-pack");
        assert_eq!(
            back.schema, 10,
            "SCHEMA must be 10 after canonical lock ordering (canonicalize) added"
        );
        assert_eq!(back.wheels.len(), 3);
        // Wheel 0: must_ship source-built, no upstream_url.
        assert_eq!(back.wheels[0].origin, Origin::Built);
        assert!(back.wheels[0].must_ship);
        assert!(back.wheels[0].upstream_url.is_none());
        assert_eq!(
            back.wheels[0].requires_dist,
            vec!["numpy>=1.21", "torch>=2.0"]
        );
        // Wheel 1: relax-changed shadow, upstream_url recorded.
        assert_eq!(back.wheels[1].origin, Origin::Built);
        assert!(!back.wheels[1].must_ship);
        assert_eq!(
            back.wheels[1].upstream_url.as_deref(),
            Some("https://files.pythonhosted.org/skrl-2.1.0.whl")
        );
        // Wheel 2: index wheel, no upstream_url.
        assert_eq!(back.wheels[2].origin, Origin::Index);
        assert!(!back.wheels[2].must_ship);
        assert!(back.wheels[2].upstream_url.is_none());
        assert!(back.wheels[2].requires_dist.is_empty());
        assert_eq!(
            back.prerelease.get("gmpy2").map(String::as_str),
            Some("==2.1.0a4")
        );
        assert_eq!(
            RetreadLock::file_name(&back.bundle),
            "retread-isaac-pack.lock.json"
        );
        assert_eq!(back.inputs_hash, "deadbeef");
        assert_eq!(back.wheels[2].sha256.as_deref(), Some("abc123"));
        assert_eq!(back.conda_capable, vec!["numpy", "torch"]);
        assert_eq!(back.entry_specs, vec!["isaaclab==0.51.1"]);
    }

    /// Schema-4 JSON (no requires_dist / must_ship / conda_capable /
    /// upstream_url) must still deserialize cleanly via serde defaults.
    #[test]
    fn schema4_lock_still_deserializes() {
        let schema4_json = r#"{
            "schema": 4,
            "retread_version": "2.3.1",
            "bundle": "old-pack",
            "version": "1.0.0",
            "python": "3.11",
            "inputs_hash": "oldhash",
            "root_requirements": ["old-pack-pypi==1.0.0"],
            "wheels": [
                {
                    "name": "somewheel",
                    "version": "1.0.0",
                    "origin": "built",
                    "filename": "somewheel-1.0.0-py3-none-any.whl"
                }
            ],
            "conda_run_deps": [],
            "index_urls": ["https://pypi.org/simple/"]
        }"#;
        let lock: RetreadLock = serde_json::from_str(schema4_json).unwrap();
        assert_eq!(lock.schema, 4);
        assert_eq!(lock.bundle, "old-pack");
        // New fields default gracefully.
        assert!(lock.wheels[0].requires_dist.is_empty());
        assert!(!lock.wheels[0].must_ship);
        assert!(lock.wheels[0].upstream_url.is_none());
        assert!(lock.conda_capable.is_empty());
    }

    /// Schema-5 JSON (has requires_dist / must_ship / conda_capable but NOT
    /// upstream_url) must deserialize cleanly — upstream_url defaults to None.
    #[test]
    fn schema5_lock_still_deserializes() {
        let schema5_json = r#"{
            "schema": 5,
            "retread_version": "2.4.0",
            "bundle": "pack",
            "version": "1.0.0",
            "python": "3.11",
            "inputs_hash": "hash5",
            "wheels": [
                {
                    "name": "skrl",
                    "version": "2.1.0",
                    "origin": "built",
                    "filename": "skrl-2.1.0-999retread-py3-none-any.whl",
                    "requires_dist": ["torch>=2.0"],
                    "must_ship": false
                }
            ],
            "conda_run_deps": [],
            "index_urls": ["https://pypi.org/simple/"],
            "conda_capable": ["torch"]
        }"#;
        let lock: RetreadLock = serde_json::from_str(schema5_json).unwrap();
        assert_eq!(lock.schema, 5);
        assert_eq!(lock.bundle, "pack");
        // upstream_url added in schema 6 — defaults to None for schema-5 locks.
        assert!(lock.wheels[0].upstream_url.is_none());
        assert!(!lock.wheels[0].must_ship);
        assert_eq!(lock.wheels[0].requires_dist, vec!["torch>=2.0"]);
    }

    #[test]
    fn inputs_hash_stable_and_order_sensitive() {
        let h1 = RetreadLock::compute_inputs_hash(
            &["b==1".into(), "a==2".into()],
            &[
                "https://pypi.nvidia.com".into(),
                "https://pypi.org/simple/".into(),
            ],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "cfg",
        );
        // entry order does NOT matter (sorted)
        let h2 = RetreadLock::compute_inputs_hash(
            &["a==2".into(), "b==1".into()],
            &[
                "https://pypi.nvidia.com".into(),
                "https://pypi.org/simple/".into(),
            ],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "cfg",
        );
        assert_eq!(h1, h2, "entry-spec order must not change the hash");
        // index order DOES matter (resolution priority)
        let h3 = RetreadLock::compute_inputs_hash(
            &["a==2".into(), "b==1".into()],
            &[
                "https://pypi.org/simple/".into(),
                "https://pypi.nvidia.com".into(),
            ],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "cfg",
        );
        assert_ne!(h1, h3, "index chain order must change the hash");
    }

    #[test]
    fn inputs_hash_covers_config_fingerprint() {
        // The config fingerprint folds in name-map / overrides / drop-deps /
        // conda-deps / build-number / auto-bundle / channels. Changing it
        // MUST change the hash, or a manifest edit replays a poisoned lock.
        let base = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "name-map=opencv-python:py-opencv",
        );
        let changed = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "name-map=opencv-python:opencv",
        );
        assert_ne!(
            base, changed,
            "config fingerprint change must change the inputs hash"
        );
    }

    #[test]
    fn inputs_hash_epoch_changes_hash() {
        // Bumping EMIT_EPOCH must invalidate any committed lock so a cold
        // re-solve is forced on output-semantics changes.
        let h1 = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "cfg",
        );
        let h2 = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            2,
            None,
            "cfg",
        );
        assert_ne!(h1, h2, "changing emit_epoch must change the hash");
    }

    #[test]
    fn inputs_hash_pin_version_differs_from_none() {
        // With retread-pin-version=true, Some("1.0.0") must differ from None.
        let without_pin = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "cfg",
        );
        let with_pin = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            Some("1.0.0"),
            "cfg",
        );
        assert_ne!(
            without_pin, with_pin,
            "pin_version=Some must differ from None"
        );
    }

    #[test]
    fn inputs_hash_pin_version_two_versions_differ_only_when_pinned() {
        // The whole point: two different retread versions produce the SAME hash
        // under pin_version=None (epoch-gated replay) but DIFFERENT hashes
        // under pin_version=Some (strict reproducibility).
        let none_v1 = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "cfg",
        );
        let none_v2 = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "cfg",
        );
        assert_eq!(
            none_v1, none_v2,
            "same epoch, no pin => identical hash regardless of retread version"
        );

        let pinned_v1 = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            Some("1.7.2"),
            "cfg",
        );
        let pinned_v2 = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            Some("1.7.3"),
            "cfg",
        );
        assert_ne!(
            pinned_v1, pinned_v2,
            "different versions under pin_version=Some must differ"
        );
    }

    /// GitWheelSource serializes and deserializes correctly with and without
    /// the optional fields (subdirectory and extras).
    #[test]
    fn git_wheel_source_serde_roundtrip() {
        // Full — with subdirectory and extras.
        let full = GitWheelSource {
            url: "https://github.com/acme/repo.git".into(),
            rev: "abcdef1234567890abcdef1234567890abcdef12".into(),
            subdirectory: Some("packages/core".into()),
            extras: vec!["sim".into(), "dev".into()],
        };
        let json = serde_json::to_string(&full).unwrap();
        let back: GitWheelSource = serde_json::from_str(&json).unwrap();
        assert_eq!(back.url, full.url);
        assert_eq!(back.rev, full.rev);
        assert_eq!(back.subdirectory.as_deref(), Some("packages/core"));
        assert_eq!(back.extras, vec!["sim", "dev"]);

        // Minimal — no subdirectory, no extras: those fields must be absent in JSON.
        let minimal = GitWheelSource {
            url: "https://github.com/acme/repo.git".into(),
            rev: "abcdef1234567890abcdef1234567890abcdef12".into(),
            subdirectory: None,
            extras: vec![],
        };
        let minimal_json = serde_json::to_string(&minimal).unwrap();
        assert!(
            !minimal_json.contains("subdirectory"),
            "subdirectory must be absent when None"
        );
        assert!(
            !minimal_json.contains("extras"),
            "extras must be absent when empty"
        );
        let back_minimal: GitWheelSource = serde_json::from_str(&minimal_json).unwrap();
        assert!(back_minimal.subdirectory.is_none());
        assert!(back_minimal.extras.is_empty());
    }

    /// `SdistWheelSource` serializes and deserializes correctly.
    #[test]
    fn sdist_wheel_source_serde_roundtrip() {
        let src = SdistWheelSource {
            index: "https://pypi.org/simple/".into(),
            name: "gym".into(),
            version: "0.26.2".into(),
            sdist_url: "https://files.pythonhosted.org/packages/gym-0.26.2.tar.gz#sha256=abc"
                .into(),
        };
        let json = serde_json::to_string(&src).unwrap();
        let back: SdistWheelSource = serde_json::from_str(&json).unwrap();
        assert_eq!(back.index, src.index);
        assert_eq!(back.name, src.name);
        assert_eq!(back.version, src.version);
        assert_eq!(back.sdist_url, src.sdist_url);
        // sdist_url must NOT begin with file:// (portability invariant).
        assert!(
            back.sdist_url.starts_with("https://"),
            "sdist_url must be an https URL, not file://"
        );
    }

    /// A schema-8 lock (no sdist_source field) must deserialize cleanly
    /// with sdist_source defaulting to None on each wheel.
    #[test]
    fn schema8_lock_sdist_source_defaults_to_none() {
        let schema8_json = r#"{
            "schema": 8,
            "retread_version": "2.6.0",
            "bundle": "gym-pack",
            "version": "1.0.0",
            "python": "3.11",
            "inputs_hash": "oldhash8",
            "root_requirements": ["gym-pack-pypi==1.0.0"],
            "wheels": [
                {
                    "name": "gym",
                    "version": "0.26.2",
                    "origin": "built",
                    "filename": "gym-0.26.2-999retread-py3-none-any.whl",
                    "requires_dist": ["numpy>=1.21"],
                    "must_ship": false
                }
            ],
            "conda_run_deps": [],
            "index_urls": ["https://pypi.org/simple/"]
        }"#;
        let lock: RetreadLock = serde_json::from_str(schema8_json).unwrap();
        assert_eq!(lock.schema, 8);
        // sdist_source added in schema 9 — must default to None for schema-8 locks.
        assert!(
            lock.wheels[0].sdist_source.is_none(),
            "sdist_source must default to None when absent from schema-8 lock"
        );
        assert!(!lock.wheels[0].must_ship);
    }

    /// A schema-7 lock JSON (no git_source field) must deserialize cleanly
    /// with git_source defaulting to None on each wheel.
    #[test]
    fn schema7_lock_git_source_defaults_to_none() {
        let schema7_json = r#"{
            "schema": 7,
            "retread_version": "2.5.0",
            "bundle": "genesis-pack",
            "version": "1.0.0",
            "python": "3.11",
            "inputs_hash": "oldhash7",
            "root_requirements": ["genesis-pack-pypi==1.0.0"],
            "wheels": [
                {
                    "name": "genesis-world",
                    "version": "1.1.1",
                    "origin": "built",
                    "filename": "genesis_world-1.1.1-999retread-py3-none-any.whl",
                    "requires_dist": ["torch>=2.0"],
                    "must_ship": true
                }
            ],
            "conda_run_deps": [],
            "index_urls": ["https://pypi.org/simple/"]
        }"#;
        let lock: RetreadLock = serde_json::from_str(schema7_json).unwrap();
        assert_eq!(lock.schema, 7);
        assert_eq!(lock.bundle, "genesis-pack");
        // git_source added in schema 8 — must default to None for schema-7 locks.
        assert!(
            lock.wheels[0].git_source.is_none(),
            "git_source must default to None when absent from schema-7 lock"
        );
        assert!(lock.wheels[0].must_ship);
    }

    fn make_test_lock_unordered() -> RetreadLock {
        RetreadLock {
            schema: SCHEMA,
            retread_version: "2.0.0".into(),
            bundle: "test-pack".into(),
            version: "1.0.0".into(),
            python: "3.11".into(),
            inputs_hash: "abc123".into(),
            root_requirements: vec!["torch-req".into(), "numpy-req".into()],
            wheels: vec![
                LockWheel {
                    name: "torch".into(),
                    version: "2.0.0".into(),
                    origin: Origin::Index,
                    filename: "torch-2.0.0-py3-none-any.whl".into(),
                    url: Some("https://pypi.org/torch.whl".into()),
                    sha256: None,
                    requires_dist: vec!["nvidia-cuda>=11.0".into(), "filelock>=3.0".into()],
                    must_ship: false,
                    upstream_url: None,
                    git_source: None,
                    sdist_source: None,
                },
                LockWheel {
                    name: "numpy".into(),
                    version: "1.26.0".into(),
                    origin: Origin::Built,
                    filename: "numpy-1.26.0-py3-none-any.whl".into(),
                    url: None,
                    sha256: None,
                    requires_dist: vec!["packaging".into()],
                    must_ship: true,
                    upstream_url: None,
                    git_source: Some(GitWheelSource {
                        url: "https://github.com/numpy/numpy".into(),
                        rev: "abc123".into(),
                        subdirectory: None,
                        extras: vec!["test".into(), "dev".into()],
                    }),
                    sdist_source: None,
                },
            ],
            conda_run_deps: vec![
                CondaDep {
                    name: "zlib".into(),
                    spec: ">=1.2".into(),
                },
                CondaDep {
                    name: "blas".into(),
                    spec: "*".into(),
                },
            ],
            index_urls: vec!["https://pypi.org/simple/".into()],
            prerelease: BTreeMap::new(),
            conda_capable: vec!["zlib".into(), "blas".into()],
            entry_specs: vec!["torch==2.0.0".into(), "numpy==1.26.0".into()],
        }
    }

    #[test]
    fn entry_specs_roundtrip() {
        // Verify entry_specs is present in JSON and round-trips correctly.
        let mut lock = make_test_lock_unordered();
        lock.canonicalize();
        let json = lock.to_pretty_json().unwrap();
        assert!(
            json.contains("entry_specs"),
            "entry_specs must appear in the JSON when non-empty"
        );
        let back: RetreadLock = serde_json::from_str(&json).unwrap();
        // After canonicalize, entry_specs should be sorted.
        assert_eq!(
            back.entry_specs,
            vec!["numpy==1.26.0", "torch==2.0.0"],
            "entry_specs must be sorted by canonicalize"
        );
    }

    #[test]
    fn entry_specs_default_empty_on_old_lock() {
        // A schema-9 JSON without entry_specs should deserialize with empty vec.
        let old_json = r#"{
            "schema": 9,
            "retread_version": "2.8.0",
            "bundle": "old-pack",
            "version": "1.0.0",
            "python": "3.11",
            "inputs_hash": "abc",
            "root_requirements": ["old-pack-pypi==1.0.0"],
            "wheels": [],
            "conda_run_deps": [],
            "index_urls": ["https://pypi.org/simple/"]
        }"#;
        let lock: RetreadLock = serde_json::from_str(old_json).unwrap();
        assert!(
            lock.entry_specs.is_empty(),
            "old locks without entry_specs must deserialize to empty vec"
        );
    }

    #[test]
    fn canonicalize_is_idempotent() {
        let mut lock = make_test_lock_unordered();
        lock.canonicalize();
        let json1 = lock.to_pretty_json().unwrap();
        lock.canonicalize();
        let json2 = lock.to_pretty_json().unwrap();
        assert_eq!(json1, json2, "canonicalize must be idempotent");
    }

    #[test]
    fn canonicalize_is_permutation_invariant() {
        let mut lock_a = make_test_lock_unordered();
        // Reverse the wheels order.
        lock_a.wheels.reverse();
        lock_a.conda_run_deps.reverse();
        lock_a.canonicalize();

        let mut lock_b = make_test_lock_unordered();
        lock_b.canonicalize();

        assert_eq!(
            lock_b.to_pretty_json().unwrap(),
            lock_a.to_pretty_json().unwrap(),
            "canonicalize must produce identical JSON regardless of input order"
        );
    }

    #[test]
    fn canonicalize_nested_requires_dist() {
        let mut lock = make_test_lock_unordered();
        // Set requires_dist in non-canonical order on the torch wheel (index 0 pre-sort).
        lock.wheels[0].requires_dist = vec!["torch>=2.0".into(), "numpy>=1.21".into()];
        lock.canonicalize();
        // After canonicalize, wheels are sorted by canonical name: numpy < torch.
        // Find the torch wheel by name to check its requires_dist.
        let torch_wheel = lock
            .wheels
            .iter()
            .find(|w| w.name == "torch")
            .expect("torch wheel must exist");
        assert_eq!(
            torch_wheel.requires_dist,
            vec!["numpy>=1.21", "torch>=2.0"],
            "requires_dist must be sorted lexicographically after canonicalize"
        );
    }

    #[test]
    fn canonicalize_inputs_hash_invariant() {
        // Reordering vectors must NOT change inputs_hash.
        let mut lock_a = make_test_lock_unordered();
        lock_a.wheels.reverse();
        let hash_before = RetreadLock::compute_inputs_hash(
            &lock_a.root_requirements,
            &lock_a.index_urls,
            "allow",
            &lock_a.python,
            crate::lock::EMIT_EPOCH,
            None,
            "fingerprint",
        );
        lock_a.canonicalize();
        let hash_after = RetreadLock::compute_inputs_hash(
            &lock_a.root_requirements,
            &lock_a.index_urls,
            "allow",
            &lock_a.python,
            crate::lock::EMIT_EPOCH,
            None,
            "fingerprint",
        );
        assert_eq!(
            hash_before, hash_after,
            "inputs_hash must be invariant under vector reordering"
        );
    }
}
