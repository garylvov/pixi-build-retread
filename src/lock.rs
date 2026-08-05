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
//!   shipped; they install by fetching their recorded direct artifact `url`
//!   and verifying the recorded `sha256`, never by consulting index metadata.
//!
//! The lock is small (KB), human-diffable, and committed next to the pack
//! manifest as `retread-<bundle>.lock.json`. It is the single source of
//! truth the installer reads.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::relax::canonical_conda_name;
use crate::relaxation_record::{
    RelaxationManifest, RelaxationRecord, canonicalize_relaxation_records,
};
use crate::workspace::WorkspaceTargetContract;

fn default_target_subdir() -> String {
    "linux-64".to_owned()
}

fn is_legacy_default_target_subdir(target_subdir: &str) -> bool {
    target_subdir == "linux-64"
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// Exact checkout-root data-injection decision for a Git-built wheel.
///
/// The checkout root itself is derived from [`GitWheelSource::url`] and
/// [`GitWheelSource::rev`].  Persisting only the skipped package subdirectories
/// keeps the lock portable while preserving the producer's exact distinction
/// between checkout-root injection and an intentionally disabled pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub enum GitWheelAutoData {
    /// Phase 1.6 did not run for this Git wheel.
    Disabled,
    /// Phase 1.6 injected non-ignored checkout-root data into this wheel.
    CheckoutRoot {
        /// Package subdirectories whose Python build artifacts are already
        /// carried by sibling wheels. Non-Python data beneath them remains
        /// eligible for the checkout-root data pass.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        skip_subdirectories: Vec<String>,
    },
}

impl GitWheelAutoData {
    /// Validate the portable, replay-trusted portion of this disposition.
    /// Producer capture calls the same contract as replay so a current lock
    /// can never serialize a path that it would later reject.
    pub(crate) fn validate(&self) -> Result<()> {
        use std::path::Component;

        let Self::CheckoutRoot {
            skip_subdirectories,
        } = self
        else {
            return Ok(());
        };
        for skip in skip_subdirectories {
            let path = std::path::Path::new(skip);
            if skip.trim().is_empty()
                || skip != skip.trim()
                || skip.contains('\\')
                || path.is_absolute()
                || path.components().any(|component| {
                    matches!(
                        component,
                        Component::ParentDir | Component::RootDir | Component::Prefix(_)
                    )
                })
            {
                anyhow::bail!("unsafe Git auto-data skip subdirectory `{skip}`");
            }
        }
        Ok(())
    }
}

/// Git provenance for a source-built wheel (schema 8+).
///
/// Stored in `LockWheel.git_source` for every wheel whose bytes were
/// produced by `build_wheel_from_git`. Carries the resolved exact commit
/// object ID (40 or 64 hexadecimal characters; never a branch name or
/// "HEAD") so replay can re-clone the identical commit without reading the
/// live `[retread-wheels]` manifest entry.
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
/// The producer also records the exact phase-1.6 auto-data disposition for
/// each wheel. Replay therefore does not infer carrier/non-carrier behavior
/// from canonical lock order or checkout group size.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitWheelSource {
    /// Canonical git URL (without the `git+` prefix — the bare clone URL).
    pub url: String,
    /// Resolved exact 40- or 64-hex commit object ID (never a
    /// branch/tag/HEAD ref). This is `git rev-parse HEAD` after checkout, not
    /// the original `rev` string from the config entry.
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
    /// Exact phase-1.6 checkout-root data decision made by the producer.
    ///
    /// `Some(Disabled)` is load-bearing: Git wheels discovered transitively
    /// through a PEP 508 URL do not receive the explicit entry's checkout-root
    /// payload. `None` means legacy/unknown provenance and is rejected for a
    /// schema-13 replay rather than guessed from checkout group size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_data: Option<GitWheelAutoData>,
}

fn is_exact_git_commit_object_id(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
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

impl SdistWheelSource {
    /// Validate replay-trusted sdist provenance and return its exact artifact
    /// URL plus the full SHA-256 carried by the URL fragment.
    ///
    /// Schema 12 predates this strict ingress rule, so locks written by an
    /// older schema-12 producer can deserialize but cannot replay when the
    /// fragment is absent, truncated, or malformed. This check must run
    /// before any replay cache, network, or output mutation.
    pub(crate) fn validated_url_and_sha256(
        &self,
        wheel_name: &str,
        wheel_version: &str,
    ) -> Result<(url::Url, String)> {
        use std::str::FromStr as _;

        let source_name = uv_normalize::PackageName::from_str(&self.name)
            .context("invalid sdist source project name")?;
        let locked_name = uv_normalize::PackageName::from_str(wheel_name)
            .context("invalid locked wheel project name")?;
        if source_name != locked_name {
            anyhow::bail!(
                "sdist source project `{}` does not match locked wheel `{wheel_name}`",
                self.name,
            );
        }

        let source_version = uv_pep508::uv_pep440::Version::from_str(&self.version)
            .context("invalid sdist source version")?;
        let locked_version = uv_pep508::uv_pep440::Version::from_str(wheel_version)
            .context("invalid locked wheel version")?;
        if source_version != locked_version {
            anyhow::bail!(
                "sdist source version `{}` does not match locked wheel version `{wheel_version}`",
                self.version,
            );
        }

        let index = url::Url::parse(&self.index).context("invalid sdist source index URL")?;
        if !matches!(index.scheme(), "http" | "https") {
            anyhow::bail!(
                "sdist source index must use http(s), not `{}`",
                index.scheme()
            );
        }

        let artifact = url::Url::parse(&self.sdist_url).context("invalid sdist artifact URL")?;
        if !matches!(artifact.scheme(), "http" | "https") {
            anyhow::bail!(
                "sdist artifact URL must use http(s), not `{}`",
                artifact.scheme()
            );
        }
        let digest = artifact
            .fragment()
            .and_then(|fragment| fragment.strip_prefix("sha256="))
            .filter(|digest| {
                digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
            .map(str::to_ascii_lowercase)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "sdist artifact URL must carry an exact `#sha256=<64 hex>` fragment"
                )
            })?;
        Ok((artifact, digest))
    }
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockWheel {
    pub name: String,
    pub version: String,
    pub origin: Origin,
    /// Standardized wheel filename (the basename installed/shipped).
    pub filename: String,
    /// Direct artifact URL for `Origin::Index` wheels; `None` for `Origin::Built`.
    /// Index wheels fetch this exact URL at install time without index metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// sha256 of the exact wheel file. Required for `Origin::Index` so
    /// install-time replay can direct-fetch and verify without resolving.
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
    /// resolved exact 40- or 64-hex commit object ID and the clone URL so
    /// replay can re-source-build the wheel without reading the live manifest.
    /// `None` for index wheels and non-git source-built wheels
    /// (path/from/url forms).
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

/// SHA-bound final wheel metadata captured after courier mapping (schema 16+).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockAbiContext {
    /// Metadata read from (or, for a remote unchanged index wheel, recorded
    /// alongside) each exact final wheel artifact.
    pub wheels: Vec<LockWheelAbiMetadata>,
}

/// Final wheel metadata bound to the same digest recorded in `LockWheel`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockWheelAbiMetadata {
    pub name: String,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requires_dist: Vec<String>,
}

/// A conda run-dep retread routed to the conda side (a shared transitive
/// the consumer's conda solve provides). Carried so the courier conda
/// package can declare them as its own run-deps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CondaDep {
    pub name: String,
    pub spec: String,
}

/// The committed install lock for one bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetreadLock {
    pub schema: u32,
    pub retread_version: String,
    pub bundle: String,
    pub version: String,
    /// Target python (e.g. "3.11"); the wheel set is python-specific.
    pub python: String,
    /// Conda target subdir whose wheels and routes this lock resolved for.
    ///
    /// Locks written before target-aware resolution were implicitly linux-64,
    /// so a missing field deserializes to that legacy target. The default is
    /// omitted on write to preserve the established linux-64 wire format.
    #[serde(
        default = "default_target_subdir",
        skip_serializing_if = "is_legacy_default_target_subdir"
    )]
    pub target_subdir: String,
    /// Complete, name-independent Pixi workspace target contract used for
    /// resolution. Older locks omit this field and therefore cannot replay as
    /// a rich-profile target; they cold-resolve once instead of aliasing a
    /// different profile on the same conda subdir.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_contract: Option<WorkspaceTargetContract>,
    /// Exact target identity used in the qualified sidecar filename. This
    /// includes canonical environment/profile scope when Pixi supplied an
    /// exact workspace target envelope. The virtual-package contract remains
    /// separately inspectable in `target_contract`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_identity: Option<String>,
    /// Canonical environment/profile provenance needed to reconstruct an
    /// exact scoped target during install and replay. The identity alone is
    /// insufficient because its SHA-256 digest is intentionally irreversible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_scope: Option<crate::workspace::ResolvedWorkspaceTarget>,
    /// Whether `target_scope` was selected by a validated exact workspace
    /// target envelope. Direct inference may produce the same contract and
    /// scope but cannot authorize that scope for co-activated sibling sources.
    #[serde(default, skip_serializing_if = "is_false")]
    pub exact_workspace_envelope: bool,
    /// sha256 over the canonicalized resolution inputs (entry specs +
    /// ordered index chain + relax policy + python + retread version).
    /// Reproducibility gate (req #5) AND the cold-solve replay key (req #4):
    /// `conda/outputs` replays this lock's emitted outputs (skipping the
    /// probe cascade) iff a freshly computed inputs hash matches this.
    #[serde(default)]
    pub inputs_hash: String,
    /// Producer-side meta requirements retained for lock/build parity. Install
    /// replay no longer hands these to uv; it installs `wheels` as explicit
    /// files with `--no-deps`.
    #[serde(default)]
    pub root_requirements: Vec<String>,
    /// Wheels to install at link time (built ship in pkg + index fetched).
    pub wheels: Vec<LockWheel>,
    /// Exact ABI validation context plus SHA-bound final wheel metadata.
    /// Missing on pre-schema-16 locks, which the replay schema gate rejects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abi_context: Option<LockAbiContext>,
    /// Final safe auto-relaxations applied while producing this artifact.
    ///
    /// Replay copies these records into the rebuilt package so a cold and
    /// replayed courier expose the same activate.d warning and JSON payload.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relaxations: Vec<RelaxationRecord>,
    /// Shared transitives routed to conda (the courier package's run-deps).
    pub conda_run_deps: Vec<CondaDep>,
    /// Producer-side index chain used during pack build/solve. Install replay
    /// uses per-wheel direct artifact URLs and never consults this chain.
    pub index_urls: Vec<String>,
    /// Prerelease pins (`name` -> `==X`) uv needs to opt those deps into
    /// prerelease resolution (it only honors them from direct reqs +
    /// overrides). Empty when the bundle pins no prereleases.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub prerelease: BTreeMap<String, String>,
    /// Site-packages-relative shared-library replacement contract applied by
    /// the installer before the GLIBC symbol audit. Values are policy strings;
    /// schema 11 supports `conda-lib`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub shadow_libs: BTreeMap<String, String>,
    /// Compatibility glibc floor recorded from the producer workspace. Exact
    /// target-envelope builds store Pixi's detected value first; direct
    /// inference falls back to the rich profile's explicit declaration. The
    /// installer uses this only as a fallback when the live consumer workspace
    /// manifest is unavailable during post-link.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_glibc: Option<String>,
    /// Effective glibc ceiling used to select the locked wheel set.
    ///
    /// This is replay/cache identity only. Install-time ABI authority remains
    /// the live declaration, with `declared_glibc` as its existing fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution_glibc: Option<String>,
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
    /// Shared content-addressed wheel-store root recorded at BUILD time for
    /// loose bundles (the directory holding `<sha256>/<filename>`). A leading
    /// `~/` means the producer's `$HOME`; the installer expands it against its
    /// OWN `$HOME` (the store default is per-user, and the store is
    /// content-addressed, so a same-path-different-user read either hits the
    /// right bytes or misses safely). Install-side resolution order:
    /// `RETREAD_WHEEL_STORE` env > this field > the shared default
    /// (`courier::retread_wheel_store_root`). `None` for fat bundles and
    /// pre-4.x locks (serde default; the installer falls back to the default
    /// store + legacy candidates).
    ///
    /// NOT part of `inputs_hash` and NOT schema-bumping: this is a byte
    /// LOCATION, never byte content (emit-neutral, like the cache root).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wheel_store: Option<String>,
}

/// Schema 18: ABI-anchor cap-completion relaxation records.
///
/// `RelaxationRecordKind::AbiAnchorCapCompleted` is a new serialized enum
/// value. Older readers must reject the lock instead of interpreting a record
/// set whose complete kind vocabulary they do not understand.
///
/// Schema 17: durable final auto-relaxation records.
///
/// `relaxations` preserves the exact warning payload across courier replay.
/// Older locks cannot prove whether an emitted package was relaxed, so the
/// producer replay gate cold-derives once under the new schema.
///
/// Schema 16: SHA-bound final wheel metadata and replay ABI context.
///
/// `abi_context` records final `Requires-Dist` lines beside the digest of each
/// final wheel. Both replay callers validate those lines against the current
/// live workspace versions, effective overrides, and ABI alias graph.
///
/// Schema 14: complete Pixi workspace target contract.
///
/// `RetreadLock.target_contract` preserves the canonical rich-profile virtual
/// package contract, while `target_scope` preserves canonical environment /
/// profile provenance and `target_identity` binds it to the qualified sidecar
/// filename. Missing contracts, scopes, or exact identities deserialize for
/// compatibility but do not match a scoped contract-qualified resolution
/// target, forcing one cold derive.
///
/// Schema 15: exact workspace-envelope provenance.
///
/// `exact_workspace_envelope` distinguishes an authoritative out-of-band
/// environment/profile selection from direct inference with the same target
/// contract and consumer scope. Resolution, artifact, sidecar, build, and
/// replay identities include this bit, preventing different sibling-lock
/// semantics from aliasing.
///
/// Schema 14: complete workspace target contract and consumer scope.
///
/// `target_contract`, `target_identity`, and `target_scope` bind rich Pixi
/// platform compatibility and environment/profile dependency provenance to a
/// target-qualified sidecar.
///
/// Schema 13: exact Git auto-data replay provenance.
///
/// `GitWheelSource.auto_data` records whether phase 1.6 ran and, when it did,
/// its exact skipped subdirectories. Older locks cannot distinguish an
/// explicit single-entry Git root from a BFS-discovered Git transitive, so
/// producer replay rejects them and cold-derives once.
///
/// Schema 12: install-time pure replay metadata.
///
/// New invariant: every `Origin::Index` wheel must carry a direct artifact
/// `url` and `sha256`; unchanged wheels without a direct URL are shipped as
/// `Origin::Built`. The installer rejects older schemas rather than falling
/// back to resolver-backed uv installs.
///
/// Target identity was added compatibly within schema 12. Legacy locks default
/// to linux-64, and missing resolution metadata fails cold-replay matching
/// closed while remaining installable on linux-64.
///
/// Schema 11: GLIBC runtime contract fields.
///
/// New fields: `shadow_libs` and `declared_glibc`, both serde-defaulted so old
/// locks deserialize. The schema bump forces committed courier locks to rebuild
/// and carry the declared runtime contract explicitly.
///
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
/// On the producer-side `conda/outputs` replay path, old schema-9 and earlier
/// locks are rejected by the != gate and the pack build performs a fresh solve.
/// On the consumer-side install path, old schemas are hard errors: install
/// replay must not fall back to resolver-backed uv. SCHEMA is NOT an epoch bump
/// (output SEMANTICS for identical inputs are unchanged; [emit-epoch-ok]).
pub const SCHEMA: u32 = 18;

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
///
/// Epoch 9: courier packages now ship activate/deactivate LD_LIBRARY_PATH
/// management, guard broken-sentinel logic, and GLIBC shadow-lib metadata.
/// Identical manifests emit different lock JSON and hook scripts.
///
/// Epoch 10: courier locks now carry url+sha256 for every unchanged index
/// wheel, and install-time replay uses explicit wheel files with --no-deps
/// instead of resolver-backed root requirements. Existing packs must cold
/// rebuild once so their locks contain direct-fetch hashes.
///
/// Epoch 11: Rule-2 conda routes are validated jointly against the pack's
/// final emitted conda dependencies and consuming-workspace pins. Conflicting
/// routes move back to the PyPI wheel bundle, changing both lock routing and
/// emitted courier contents for identical manifests.
///
/// Epoch 12: `retread-deps-from` now honors `[tool.uv.sources]` local-path
/// declarations and omits those nonportable roots instead of resolving a
/// same-named registry project. Existing locks must cold-resolve once so their
/// emitted closure reflects the source-aware root set.
///
/// Epoch 13: conda environment deps-from sources now contribute nested pip
/// roots and explicitly mapped advisory floors. Existing locks must
/// cold-resolve so the uv closure reflects the structured YAML inputs.
///
/// Epoch 14: courier recipes no longer create a redundant Python host prefix.
/// Existing locks must rebuild so package metadata is derived from the
/// runtime-only Python requirement while advertised host run exports remain
/// available to pixi.
///
/// Epoch 15: parselmouth candidate families are sorted and deduplicated before
/// route and repair selection. Existing locks must cold-resolve once so their
/// chosen provider cannot depend on hash-map iteration order.
///
/// Epoch 16: resolution identity includes normalized Python minor, target
/// subdir, explicit glibc declaration, and effective wheel ceiling. Locks and
/// rewrite caches cannot replay across target contracts.
/// Epoch 17: final joint-route restoration reuses a compatible wheel already
/// present in the bundle instead of staging a duplicate distribution.
///
/// Epoch 18: conda/outputs run dependencies are canonically ordered on both
/// cold and replay paths, and replay omits Pixi's build-time-injected
/// `python_abi`. This preserves the cold source-package identity instead of
/// advertising a duplicate ABI dependency from the committed build lock.
///
/// Epoch 19: the zip writer upgrade from 2.4 to 6.0 can change rewritten wheel
/// bytes for identical metadata edits. Existing locks and shadow caches must
/// cold-derive once so their recorded content hashes match the new writer.
///
/// Epoch 20: injection and auto-data rewrites normalize every source-wheel ZIP
/// timestamp so identical payloads cannot acquire worktree-specific hashes.
///
/// Epoch 21: every raw source-built wheel is timestamp-normalized before it is
/// hashed or cached, including wheels that need no injection or metadata
/// rewrite. The built-wheel artifact namespace advances to v4 with this epoch.
///
/// Epoch 22: source builds run with PYTHONHASHSEED=0 so PEP 517 projects that
/// derive metadata from Python sets emit stable header ordering. The built-
/// wheel artifact namespace advances to v5 with this epoch.
///
/// Epoch 23: isolated sdist builds constrain setuptools below 81, retaining
/// pkg_resources compatibility for legacy setup.py projects. The built-wheel
/// artifact namespace advances to v6 with this epoch.
///
/// Epoch 24: wheel metadata parsing ignores RFC 822 folded continuation lines
/// before matching identity and dependency headers. Existing locks must cold-
/// derive once so folded License text cannot masquerade as Name, Version, or
/// Requires-Dist metadata.
///
/// Epoch 25: exact-version conda-owned Python distributions are verified by
/// conda ownership rather than wheel RECORD layout. Existing courier packages
/// must rebuild so recordless conda metadata cannot trigger endless activation
/// repair replays.
///
/// Epoch 26: routed BFS overrides and prepared output identity are validated
/// consistently across metadata, cold build, and replay. Existing locks must
/// cold-derive under the complete routed-dependency and output-identity
/// contract.
///
/// Epoch 27: resolution, artifact, sidecar, and build identities include the
/// complete name-independent Pixi workspace target contract plus canonical
/// exact consumer scope. Same-subdir rich profiles such as Pixi-detected
/// linux-64/glibc 2.28 and explicit glibc 2.35 cannot cross-replay, and two
/// same-contract environments with different dependency overlays cannot
/// overwrite one another's emitted courier artifacts.
///
/// Epoch 28: exact workspace-envelope provenance participates in every rich
/// target identity and is persisted in courier locks. Exact and directly
/// inferred targets with otherwise identical contracts and consumer scopes
/// cannot share sibling constraints, artifacts, sidecars, builds, or replay.
///
/// Epoch 29: auto-bundle diagnostics with unsat context are now persisted in lock.
///
/// Epoch 30: unsatisfiable cross-wheel metadata constraints are relaxed inline
/// before PyPI restoration, changing the selected wheel and emitted run list.
///
/// Epoch 31: relaxation is selected strictly and at clause granularity, with a
/// fail-closed ABI-anchor post-check. Identical inputs can therefore emit a
/// narrower dependency or reject an unsafe cached result.
///
/// Epoch 32: every metadata mutation path shares the transitive ABI-alias veto,
/// and the fail-closed post-check also validates embedded wheel metadata.
///
/// Epoch 33: committed-lock replay reconstructs the courier's final metadata
/// mapper and validates those embedded lines with the live ABI aliases,
/// overrides, and workspace pins instead of checking the pre-courier copy.
///
/// Epoch 34: committed locks persist SHA-bound final wheel metadata, and both
/// conda/outputs and conda/build_v1 validate it against the current solved ABI
/// context at their shared ingress before replay can win.
///
/// Epoch 35: exact-version relaxation fails closed for non-zero epochs and
/// prerelease/dev versions instead of emitting a range that drops the epoch or
/// excludes the original prerelease.
///
/// Epoch 36: an overflowing CUDA-family major ceiling now emits a strict
/// same-major wildcard constraint instead of dropping CUDA-major protection.
///
/// Epoch 37: relaxed packages ship an activate.d warning hook and structured
/// relaxation manifest, and courier locks retain that payload across replay.
///
/// Epoch 38: activate.d hooks echo every relaxation regardless of count and
/// always point to the package-bundled JSON manifest for full detail.
///
/// Epoch 39: conflicting precise workspace ABI-anchor patch facts participate
/// in emission relaxation while remaining validation-only, so compatible
/// wheel pins emit a portable patch band.
///
/// Epoch 40: existing identity-fallback auto-routes bypass duplicate metadata
/// fetching so final joint validation sees the complete wheel constraint set;
/// conflicting non-anchor routes may therefore restore a reconciled PyPI wheel.
///
/// Epoch 41: fresh emission auto-completes an open bare-major ABI-anchor lower
/// bound to a canonical within-major cap and records the completion warning.
///
/// Epoch 42: fresh emission widens an effective exact ABI-anchor selection to
/// its within-minor band so independently-built packs can compose on a newer
/// ABI-compatible patch.
///
/// Epoch 43: cold closure resolution may auto-build and ship an exact-pinned,
/// sdist-only dependency when the controlled build proves its wheel is
/// platform-independent.
///
/// Epoch 44: source builds may provision a compiler environment pinned to the
/// newest compatible conda-forge sysroot and emit exact-sysroot manylinux
/// native wheels, expanding the set of bundleable sdist-only dependencies.
///
/// Epoch 46: ABI anchors are excluded from the emit-pypi floor-envelope
/// override table; their per-wheel METADATA constraints ship unchanged and
/// are validated against the workspace's conda pins.
///
/// Epoch 47: git source builds accept a submodule policy, so a source tree
/// carrying gitlinks can emit a wheel built either with its submodules
/// initialized or from the parent tree alone. Both differ from the previous
/// unconditional refusal, and from each other.
///
/// Epoch 48: feature-declared channels outrank inherited workspace channels,
/// and each environment's constraint extraction solves against its own
/// channels rather than the per-output union. Both change which channel
/// serves a given package under strict priority, and therefore what gets
/// emitted.
pub const EMIT_EPOCH: u32 = 48;

fn parse_stored_glibc(value: Option<&str>) -> Option<Option<(u32, u32)>> {
    match value {
        None => Some(None),
        Some(value) => {
            let (major, minor) = value.split_once('.')?;
            if major.is_empty()
                || minor.is_empty()
                || minor.contains('.')
                || !major.bytes().all(|byte| byte.is_ascii_digit())
                || !minor.bytes().all(|byte| byte.is_ascii_digit())
            {
                return None;
            }
            Some(Some((major.parse().ok()?, minor.parse().ok()?)))
        }
    }
}

/// Parse the exact Python grammar permitted at lock ingress.
///
/// Locks record a minor compatibility target. A numeric patch is accepted for
/// compatibility with producers that spell the same target as `3.11.0`, then
/// intentionally discarded. Bare majors, wildcards, suffixes, whitespace,
/// and additional components are malformed rather than aliases.
fn parse_stored_python_minor(value: &str) -> Option<(u32, u32)> {
    let mut parts = value.split('.');
    let major = parts.next()?;
    let minor = parts.next()?;
    let patch = parts.next();
    if parts.next().is_some()
        || major.is_empty()
        || minor.is_empty()
        || !major.bytes().all(|byte| byte.is_ascii_digit())
        || !minor.bytes().all(|byte| byte.is_ascii_digit())
        || patch.is_some_and(|patch| {
            patch.is_empty() || !patch.bytes().all(|byte| byte.is_ascii_digit())
        })
    {
        return None;
    }
    Some((major.parse().ok()?, minor.parse().ok()?))
}

pub(crate) fn normalized_target_python(value: &str) -> Result<String> {
    parse_stored_python_minor(value)
        .map(|(major, minor)| format!("{major}.{minor}"))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "invalid target Python `{value}`; expected major.minor or major.minor.patch"
            )
        })
}

fn stored_python_matches(stored: &str, expected: &str) -> bool {
    matches!(
        (
            parse_stored_python_minor(stored),
            parse_stored_python_minor(expected)
        ),
        (Some(stored), Some(expected)) if stored == expected
    )
}

impl RetreadLock {
    /// File name for a bundle's lock next to the pack manifest.
    pub fn file_name(bundle: &str) -> String {
        format!("retread-{bundle}.lock.json")
    }

    /// Target-qualified lock filename. The full resolution identity prevents
    /// two Python/platform/glibc contracts for one bundle from overwriting or
    /// replaying each other's committed lock.
    pub(crate) fn file_name_for_target(
        bundle: &str,
        target: &crate::pypi::ResolutionTarget,
    ) -> String {
        Self::file_name_for_target_identity(bundle, &target.resolution_identity())
    }

    pub(crate) fn file_name_for_target_identity(bundle: &str, target_identity: &str) -> String {
        format!("retread-{bundle}.target-{target_identity}.lock.json")
    }

    /// Ordered read candidates for an exact target. The target-qualified path
    /// is always first. Only a native linux-64 request may inspect the legacy
    /// bundle-only path; foreign targets never probe it and every write uses
    /// the qualified path.
    pub(crate) fn read_file_names_for_target(
        bundle: &str,
        target: &crate::pypi::ResolutionTarget,
        native_subdir: &str,
    ) -> Vec<String> {
        let mut names = vec![Self::file_name_for_target(bundle, target)];
        if target.conda_subdir() == "linux-64" && native_subdir == "linux-64" {
            names.push(Self::file_name(bundle));
        }
        names
    }

    /// Whether this lock was resolved for `target_subdir`.
    pub fn is_for_target(&self, target_subdir: &str) -> bool {
        self.target_subdir == target_subdir
    }

    /// Reconstruct the immutable target recorded in this lock. Malformed
    /// compatibility metadata is a hard error at lock ingress.
    pub(crate) fn resolution_target(&self) -> Result<crate::pypi::ResolutionTarget> {
        if let Some(identity) = &self.target_identity
            && (identity.len() != 64 || !identity.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            anyhow::bail!("invalid target_identity in retread lock");
        }
        let (python_major, python_minor) = parse_stored_python_minor(&self.python)
            .ok_or_else(|| anyhow::anyhow!("invalid python in retread lock"))?;
        let declared_glibc = parse_stored_glibc(self.declared_glibc.as_deref())
            .ok_or_else(|| anyhow::anyhow!("invalid declared_glibc in retread lock"))?;
        let effective_glibc = parse_stored_glibc(self.resolution_glibc.as_deref())
            .ok_or_else(|| anyhow::anyhow!("invalid resolution_glibc in retread lock"))?;
        if self.exact_workspace_envelope
            && (self.target_contract.is_none() || self.target_scope.is_none())
        {
            anyhow::bail!(
                "exact workspace-envelope provenance requires both target_contract and target_scope"
            );
        }
        let target = crate::pypi::ResolutionTarget::try_from_wheel_target_with_contract(
            crate::pypi::WheelTarget {
                python_version: format!("{python_major}.{python_minor}"),
                conda_subdir: self.target_subdir.clone(),
                max_glibc: effective_glibc,
            },
            declared_glibc,
            self.target_contract.clone(),
        )?;
        match (self.target_scope.clone(), self.exact_workspace_envelope) {
            (Some(scope), true) => target.with_exact_workspace_scope(scope),
            (Some(scope), false) => target.with_workspace_scope(scope),
            (None, false) => Ok(target),
            (None, true) => unreachable!("validated exact envelope scope above"),
        }
    }

    /// Whether this lock matches the complete immutable resolution target.
    ///
    /// Missing or malformed target metadata never aliases a fully specified
    /// target. This makes legacy locks cold-resolve once without preventing
    /// their native linux-64 install replay.
    pub(crate) fn is_for_resolution_target(&self, target: &crate::pypi::ResolutionTarget) -> bool {
        let Ok(locked) = self.resolution_target() else {
            return false;
        };
        let exact_identity_matches = match self.target_identity.as_deref() {
            Some(stored) => {
                stored == target.resolution_identity() && stored == locked.resolution_identity()
            }
            None => target.workspace_scope().is_none() && locked.workspace_scope().is_none(),
        };
        exact_identity_matches
            && stored_python_matches(&self.python, target.python_version())
            && locked.compatibility_identity() == target.compatibility_identity()
    }

    /// Validate every replay-trusted content/provenance field before replay
    /// performs cache, network, or output mutation.
    pub(crate) fn validate_replay_provenance(&self) -> Result<()> {
        for wheel in &self.wheels {
            let sha256 = wheel.sha256.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "locked wheel {}=={} has no final sha256",
                    wheel.name,
                    wheel.version,
                )
            })?;
            if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                anyhow::bail!(
                    "locked wheel {}=={} has an invalid final sha256",
                    wheel.name,
                    wheel.version,
                );
            }

            if let Some(source) = &wheel.sdist_source {
                if wheel.origin != Origin::Built || wheel.must_ship {
                    anyhow::bail!(
                        "locked wheel {}=={} records sdist provenance for an incompatible origin",
                        wheel.name,
                        wheel.version,
                    );
                }
                source
                    .validated_url_and_sha256(&wheel.name, &wheel.version)
                    .with_context(|| {
                        format!(
                            "invalid sdist provenance for locked wheel {}=={}",
                            wheel.name, wheel.version,
                        )
                    })?;
            }
            if let Some(source) = &wheel.git_source
                && source.auto_data.is_none()
            {
                anyhow::bail!(
                    "locked wheel {}=={} is missing its exact Git auto-data disposition",
                    wheel.name,
                    wheel.version,
                );
            }
        }
        Ok(())
    }

    /// Validate the complete, target-bound replay contract without touching
    /// the network, caches, or output filesystem. Replay callers run this for
    /// every entry before materializing the first wheel so malformed later
    /// provenance cannot leave partial work behind.
    pub(crate) fn validate_replay_contract_for_target(
        &self,
        target: &crate::pypi::ResolutionTarget,
        expected_bundle: &str,
    ) -> Result<()> {
        use std::path::Component;

        self.validate_replay_provenance()?;
        if !self.is_for_resolution_target(target) {
            anyhow::bail!("retread lock does not match the requested resolution target");
        }
        if self.bundle != expected_bundle {
            anyhow::bail!(
                "retread lock records bundle `{}`, but replay requested `{expected_bundle}`",
                self.bundle,
            );
        }
        if let Some(manifest) =
            RelaxationManifest::new(self.bundle.clone(), self.relaxations.clone())
        {
            manifest
                .validate_for(expected_bundle, target)
                .context("retread lock records an invalid relaxation warning payload")?;
        }
        let safe_component = |value: &str| {
            !value.trim().is_empty()
                && value == value.trim()
                && !matches!(value, "." | "..")
                && !value.contains('/')
                && !value.contains('\\')
                && std::path::Path::new(value).components().count() == 1
        };
        if !safe_component(&self.bundle) {
            anyhow::bail!("retread lock records an invalid bundle path component");
        }
        self.bundle
            .parse::<uv_normalize::PackageName>()
            .context("retread lock records an invalid bundle package name")?;
        self.version
            .parse::<uv_pep508::uv_pep440::Version>()
            .context("retread lock records an invalid bundle version")?;
        if self.wheels.is_empty() {
            anyhow::bail!("retread lock contains no wheel payload");
        }
        if let Some(recorded) = self.wheel_store.as_deref() {
            let (path, portable) = if let Some(rest) = recorded.strip_prefix("~/") {
                (std::path::Path::new(rest), true)
            } else {
                let path = std::path::Path::new(recorded);
                if !path.is_absolute() {
                    anyhow::bail!("retread lock records a non-portable relative wheel-store path");
                }
                (path, false)
            };
            if path.as_os_str().is_empty()
                || (portable && path.is_absolute())
                || path.components().any(|component| {
                    matches!(component, Component::ParentDir | Component::Prefix(_))
                        || (portable && matches!(component, Component::RootDir))
                })
            {
                anyhow::bail!("retread lock records an unsafe wheel-store path");
            }
        }

        let mut seen_distributions = std::collections::BTreeMap::<String, String>::new();
        let mut seen_filenames = std::collections::BTreeMap::<String, String>::new();
        for wheel in &self.wheels {
            if wheel.name.trim().is_empty()
                || wheel.version.trim().is_empty()
                || wheel.filename.trim().is_empty()
            {
                anyhow::bail!("retread lock contains an incomplete wheel entry");
            }
            let canonical = canonical_conda_name(&wheel.name);
            if let Some(prior) = seen_distributions.insert(canonical, wheel.name.clone()) {
                anyhow::bail!(
                    "retread lock contains duplicate distributions `{prior}` and `{}`",
                    wheel.name,
                );
            }
            let recorded = crate::courier::validate_wheel_filename_for_target(
                &wheel.name,
                &wheel.version,
                &wheel.filename,
                target.wheel_target(),
                "locked replay wheel filename",
            )?;
            if let Some(prior) =
                seen_filenames.insert(recorded.to_ascii_lowercase(), wheel.filename.clone())
            {
                anyhow::bail!(
                    "retread lock contains duplicate wheel filenames `{prior}` and `{}`",
                    wheel.filename,
                );
            }

            let validate_artifact_url = |label: &str, text: &str| -> Result<String> {
                let url = url::Url::parse(text).with_context(|| {
                    format!("invalid {label} for {}=={}", wheel.name, wheel.version)
                })?;
                if !matches!(url.scheme(), "http" | "https" | "file") {
                    anyhow::bail!(
                        "{label} for {}=={} uses unsupported scheme `{}`",
                        wheel.name,
                        wheel.version,
                        url.scheme(),
                    );
                }
                let filename = crate::wheel::wheel_filename_from_url(&url)
                    .with_context(|| format!("invalid wheel filename in {label}"))?;
                crate::courier::validate_wheel_filename_for_target(
                    &wheel.name,
                    &wheel.version,
                    &filename,
                    target.wheel_target(),
                    &format!("locked replay {label} filename"),
                )?;
                Ok(filename)
            };

            match wheel.origin {
                Origin::Index => {
                    if wheel.must_ship
                        || wheel.upstream_url.is_some()
                        || wheel.git_source.is_some()
                        || wheel.sdist_source.is_some()
                    {
                        anyhow::bail!(
                            "index wheel {}=={} records incompatible replay provenance",
                            wheel.name,
                            wheel.version,
                        );
                    }
                    let url = wheel.url.as_deref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "index wheel {}=={} has no locked artifact URL",
                            wheel.name,
                            wheel.version,
                        )
                    })?;
                    let filename = validate_artifact_url("artifact URL", url)?;
                    if filename != wheel.filename {
                        anyhow::bail!(
                            "artifact URL for {}=={} names `{filename}`, but the lock records `{}`",
                            wheel.name,
                            wheel.version,
                            wheel.filename,
                        );
                    }
                }
                Origin::Built => {
                    if wheel.url.is_some() {
                        anyhow::bail!(
                            "built wheel {}=={} unexpectedly records an index URL",
                            wheel.name,
                            wheel.version,
                        );
                    }
                    if let Some(upstream) = wheel.upstream_url.as_deref() {
                        if wheel.must_ship
                            || wheel.git_source.is_some()
                            || wheel.sdist_source.is_some()
                        {
                            anyhow::bail!(
                                "built wheel {}=={} records conflicting upstream provenance",
                                wheel.name,
                                wheel.version,
                            );
                        }
                        let filename = validate_artifact_url("upstream URL", upstream)?;
                        if !crate::courier::wheel_filename_provenance_matches(
                            &wheel.filename,
                            &filename,
                        ) {
                            anyhow::bail!(
                                "upstream URL for {}=={} names different wheel provenance",
                                wheel.name,
                                wheel.version,
                            );
                        }
                    }
                    if let Some(git) = &wheel.git_source {
                        if !wheel.must_ship
                            || wheel.upstream_url.is_some()
                            || wheel.sdist_source.is_some()
                        {
                            anyhow::bail!(
                                "built wheel {}=={} records incompatible git provenance",
                                wheel.name,
                                wheel.version,
                            );
                        }
                        if git.url.trim().is_empty()
                            || git.url != git.url.trim()
                            || git.url.starts_with('-')
                            || git.url.chars().any(char::is_control)
                        {
                            anyhow::bail!(
                                "built wheel {}=={} records an invalid git URL",
                                wheel.name,
                                wheel.version,
                            );
                        }
                        if !is_exact_git_commit_object_id(&git.rev) {
                            anyhow::bail!(
                                "built wheel {}=={} git revision is not an exact 40- or 64-hex commit object ID",
                                wheel.name,
                                wheel.version,
                            );
                        }
                        if let Some(subdirectory) = git.subdirectory.as_deref() {
                            let path = std::path::Path::new(subdirectory);
                            if subdirectory.trim().is_empty()
                                || subdirectory.contains('\\')
                                || path.is_absolute()
                                || path.components().any(|component| {
                                    matches!(
                                        component,
                                        Component::ParentDir
                                            | Component::RootDir
                                            | Component::Prefix(_)
                                    )
                                })
                            {
                                anyhow::bail!(
                                    "built wheel {}=={} records an unsafe git subdirectory",
                                    wheel.name,
                                    wheel.version,
                                );
                            }
                        }
                        let auto_data = git.auto_data.as_ref().ok_or_else(|| {
                            anyhow::anyhow!(
                                "built wheel {}=={} is missing its exact Git auto-data disposition",
                                wheel.name,
                                wheel.version,
                            )
                        })?;
                        auto_data.validate().with_context(|| {
                            format!(
                                "built wheel {}=={} records invalid Git auto-data provenance",
                                wheel.name, wheel.version,
                            )
                        })?;
                    }
                }
            }
        }
        Ok(())
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
    /// name-map, shadow-libs, drop-deps, conda-deps, route-policy,
    /// route-include, auto-bundle, build-number, the conda channel list, and
    /// the workspace solve fingerprint. Without it, editing any of those
    /// (e.g. genesis's `retread-name-map`) would leave the hash unchanged and
    /// a stale, POISONED lock would replay. It is produced by the shared
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
        let target = crate::pypi::ResolutionTarget::from_wheel_target(
            crate::pypi::WheelTarget {
                python_version: python.to_owned(),
                conda_subdir: "linux-64".to_owned(),
                max_glibc: None,
            },
            None,
        );
        Self::compute_inputs_hash_for_target(
            entry_specs,
            index_urls,
            relax,
            &target,
            emit_epoch,
            pin_version,
            config_fingerprint,
        )
    }

    /// Exact-target form of [`Self::compute_inputs_hash`]. Build and replay
    /// paths use this method so the foundation's normalized, full-SHA target
    /// identity participates in every lock gate.
    pub(crate) fn compute_inputs_hash_for_target(
        entry_specs: &[String],
        index_urls: &[String],
        relax: &str,
        target: &crate::pypi::ResolutionTarget,
        emit_epoch: u32,
        pin_version: Option<&str>,
        config_fingerprint: &str,
    ) -> String {
        use sha2::{Digest, Sha256};
        let mut sorted = entry_specs.to_vec();
        sorted.sort();
        let mut h = Sha256::new();
        h.update(b"retread-inputs-v7\n");
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
        h.update(b"--target--\n");
        h.update(target.resolution_identity().as_bytes());
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
        let lock: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing lock {}", path.display()))?;
        if lock.schema == SCHEMA {
            lock.validate_replay_provenance()
                .with_context(|| format!("validating lock {}", path.display()))?;
        }
        Ok(lock)
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
    /// - `relaxations[]`: stable semantic tuple
    /// - `index_urls[]`: NOT sorted — chain order is semantically significant.
    /// - `prerelease`: BTreeMap — already key-ordered.
    ///
    /// Nested sorts (A-1):
    /// - `LockWheel.requires_dist[]`: lexicographic (order-insensitive per §A.5)
    /// - `LockAbiContext.wheels[]`: canonical name + digest
    /// - `LockWheelAbiMetadata.requires_dist[]`: lexicographic
    /// - `RelaxationRecord.involved_wheels[]`: lexicographic
    /// - `GitWheelSource.extras[]`: lexicographic
    /// - `GitWheelSource.auto_data.skip_subdirectories[]`: lexicographic
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
        canonicalize_relaxation_records(&mut self.relaxations);

        // Top-level: entry_specs lexicographic (already sorted by
        // courier_input_specs, but sort here defensively so the invariant
        // is enforced in one place regardless of producer).
        self.entry_specs.sort();

        // Nested A-1: requires_dist and GitWheelSource.extras per wheel.
        for wheel in &mut self.wheels {
            wheel.requires_dist.sort();
            if let Some(gs) = &mut wheel.git_source {
                gs.extras.sort();
                if let Some(GitWheelAutoData::CheckoutRoot {
                    skip_subdirectories,
                }) = &mut gs.auto_data
                {
                    skip_subdirectories.sort();
                    skip_subdirectories.dedup();
                }
            }
        }
        if let Some(context) = &mut self.abi_context {
            context.wheels.sort_by(|left, right| {
                canonical_conda_name(&left.name)
                    .cmp(&canonical_conda_name(&right.name))
                    .then_with(|| left.sha256.cmp(&right.sha256))
            });
            for wheel in &mut context.wheels {
                wheel.requires_dist.sort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_git_commit_object_ids_accept_sha1_and_sha256_only() {
        for accepted in ["a".repeat(40), "A1".repeat(20), "b".repeat(64)] {
            assert!(is_exact_git_commit_object_id(&accepted), "{accepted}");
        }
        for rejected in [
            "a".repeat(39),
            "a".repeat(41),
            "a".repeat(63),
            "a".repeat(65),
            "g".repeat(40),
            "z".repeat(64),
        ] {
            assert!(!is_exact_git_commit_object_id(&rejected), "{rejected}");
        }
    }

    fn target(
        python: &str,
        subdir: &str,
        declared_glibc: Option<(u32, u32)>,
        effective_glibc: Option<(u32, u32)>,
    ) -> crate::pypi::ResolutionTarget {
        crate::pypi::ResolutionTarget::from_wheel_target(
            crate::pypi::WheelTarget {
                python_version: python.into(),
                conda_subdir: subdir.into(),
                max_glibc: effective_glibc,
            },
            declared_glibc,
        )
    }

    fn linux_64_contract(
        declared_glibc: Option<&str>,
        detected_glibc: &str,
    ) -> WorkspaceTargetContract {
        let mut declared_virtual_packages = BTreeMap::new();
        if let Some(glibc) = declared_glibc {
            declared_virtual_packages.insert("glibc".to_string(), glibc.to_string());
        }
        WorkspaceTargetContract {
            subdir: "linux-64".to_string(),
            declared_virtual_packages,
            detected_virtual_packages: BTreeMap::from([
                ("archspec".to_string(), "1=x86_64".to_string()),
                ("glibc".to_string(), detected_glibc.to_string()),
                ("linux".to_string(), "4.18".to_string()),
                ("unix".to_string(), String::new()),
            ]),
        }
    }

    #[test]
    fn lock_roundtrips() {
        let lock = RetreadLock {
            schema: SCHEMA,
            retread_version: "2.0.0".into(),
            bundle: "isaac-pack".into(),
            version: "5.1.0".into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
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
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            resolution_glibc: None,
            conda_capable: vec!["numpy".into(), "torch".into()],
            entry_specs: vec!["isaaclab==0.51.1".into()],
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };
        let json = lock.to_pretty_json().unwrap();
        let back: RetreadLock = serde_json::from_str(&json).unwrap();
        assert_eq!(back.bundle, "isaac-pack");
        assert_eq!(
            back.schema, SCHEMA,
            "lock round-trip must preserve the current schema"
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
        assert!(lock.shadow_libs.is_empty());
        assert!(lock.declared_glibc.is_none());
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
    fn legacy_lock_target_defaults_to_linux_64() {
        let legacy_json = r#"{
            "schema": 12,
            "retread_version": "4.8.8",
            "bundle": "legacy-pack",
            "version": "1.0.0",
            "python": "3.11",
            "wheels": [],
            "conda_run_deps": [],
            "index_urls": []
        }"#;
        let lock: RetreadLock = serde_json::from_str(legacy_json).unwrap();
        assert_eq!(lock.target_subdir, "linux-64");
        assert!(lock.resolution_glibc.is_none());
        assert!(lock.is_for_target("linux-64"));
        assert!(!lock.is_for_target("linux-aarch64"));
        assert!(lock.is_for_resolution_target(&target("3.11.0", "linux-64", None, None)));
        assert!(!lock.is_for_resolution_target(&target("3.11", "linux-64", None, Some((2, 35)))));
    }

    #[test]
    fn lock_python_is_strict_and_normalizes_numeric_patch() {
        let expected_minor = target("3.11", "linux-64", None, None);
        let expected_patch = target("3.11.0", "linux-64", None, None);

        for spelling in ["3.11", "3.11.0"] {
            let mut lock = make_test_lock_unordered();
            lock.python = spelling.into();
            let reconstructed = lock.resolution_target().unwrap();
            assert_eq!(reconstructed.python_version(), "3.11");
            assert!(lock.is_for_resolution_target(&expected_minor));
            assert!(lock.is_for_resolution_target(&expected_patch));
        }

        for malformed in [
            "3",
            "3.",
            ".11",
            "3.11.",
            "3.11.*",
            "3.11rc1",
            "3.11.0rc1",
            "3.11.0.1",
            " 3.11",
            "3.11 ",
        ] {
            let mut lock = make_test_lock_unordered();
            lock.python = malformed.into();
            assert!(
                lock.resolution_target().is_err(),
                "malformed lock Python {malformed:?} was accepted"
            );
            assert!(!lock.is_for_resolution_target(&expected_minor));

            assert!(
                crate::pypi::ResolutionTarget::try_from_parts(malformed, "linux-64", None).is_err(),
                "malformed requested target {malformed:?} was constructed",
            );
        }
    }

    #[test]
    fn target_metadata_roundtrips_without_changing_linux_64_wire_default() {
        let linux = make_test_lock_unordered();
        let linux_json = linux.to_pretty_json().unwrap();
        assert!(!linux_json.contains("\"target_subdir\""));
        assert!(!linux_json.contains("\"resolution_glibc\""));

        let mut arm = linux;
        arm.target_subdir = "linux-aarch64".into();
        arm.declared_glibc = Some("2.35".into());
        arm.resolution_glibc = Some("2.35".into());
        let arm_json = arm.to_pretty_json().unwrap();
        let decoded: RetreadLock = serde_json::from_str(&arm_json).unwrap();
        assert_eq!(decoded.target_subdir, "linux-aarch64");
        assert_eq!(decoded.resolution_glibc.as_deref(), Some("2.35"));
        assert!(decoded.is_for_resolution_target(&target(
            "3.11",
            "linux-aarch64",
            Some((2, 35)),
            Some((2, 35))
        )));
        assert!(!decoded.is_for_resolution_target(&target(
            "3.11",
            "linux-64",
            Some((2, 35)),
            Some((2, 35))
        )));

        arm.resolution_glibc = Some("2.35 trailing".into());
        assert!(!arm.is_for_resolution_target(&target(
            "3.11",
            "linux-aarch64",
            Some((2, 35)),
            Some((2, 35))
        )));
    }

    #[test]
    fn rich_target_contract_roundtrips_and_fails_closed_across_profiles() {
        let p1 = crate::pypi::ResolutionTarget::try_for_contract(
            "3.11",
            linux_64_contract(None, "2.28"),
        )
        .unwrap();
        let p3 = crate::pypi::ResolutionTarget::try_for_contract(
            "3.11",
            linux_64_contract(Some("2.35"), "2.35"),
        )
        .unwrap();

        let mut lock = make_test_lock_unordered();
        lock.target_contract = p1.target_contract().cloned();
        lock.declared_glibc = p1.declared_glibc().map(crate::glibc::format_glibc);
        lock.resolution_glibc = p1.effective_glibc().map(crate::glibc::format_glibc);
        let json = lock.to_pretty_json().unwrap();
        assert!(json.contains("\"target_contract\""));
        let decoded: RetreadLock = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.target_contract, p1.target_contract().cloned());
        assert!(decoded.is_for_resolution_target(&p1));
        assert!(
            !decoded.is_for_resolution_target(&p3),
            "same-subdir p1 and p3 contracts must never share a lock"
        );

        let mut missing_contract = decoded.clone();
        missing_contract.target_contract = None;
        assert!(
            !missing_contract.is_for_resolution_target(&p1),
            "a legacy lock cannot alias a fully specified profile contract"
        );

        let mut mismatched_subdir = decoded;
        mismatched_subdir.target_contract.as_mut().unwrap().subdir = "linux-aarch64".into();
        assert!(mismatched_subdir.resolution_target().is_err());
    }

    #[test]
    fn inputs_hash_uses_full_normalized_target_identity() {
        let hash = |target: &crate::pypi::ResolutionTarget| {
            RetreadLock::compute_inputs_hash_for_target(
                &["demo==1".into()],
                &["https://pypi.org/simple/".into()],
                "patch-then-minor",
                target,
                EMIT_EPOCH,
                None,
                "cfg",
            )
        };
        let arm_35 = target("3.11", "linux-aarch64", Some((2, 35)), Some((2, 35)));
        let arm_35_patch = target("3.11.0", "linux-aarch64", Some((2, 35)), Some((2, 35)));
        let arm_39 = target("3.11", "linux-aarch64", Some((2, 39)), Some((2, 39)));
        let x86_35 = target("3.11", "linux-64", Some((2, 35)), Some((2, 35)));

        assert_eq!(hash(&arm_35), hash(&arm_35_patch));
        assert_ne!(hash(&arm_35), hash(&arm_39));
        assert_ne!(hash(&arm_35), hash(&x86_35));
        assert_eq!(hash(&arm_35).len(), 64);
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

    /// GitWheelSource serializes and deserializes exact auto-data disposition,
    /// while an older lock with no field remains parseable as unknown.
    #[test]
    fn git_wheel_source_serde_roundtrip() {
        // Full — with subdirectory and extras.
        let full = GitWheelSource {
            url: "https://github.com/acme/repo.git".into(),
            rev: "abcdef1234567890abcdef1234567890abcdef12".into(),
            subdirectory: Some("packages/core".into()),
            extras: vec!["sim".into(), "dev".into()],
            auto_data: Some(GitWheelAutoData::CheckoutRoot {
                skip_subdirectories: vec!["packages/core".into()],
            }),
        };
        let json = serde_json::to_string(&full).unwrap();
        let back: GitWheelSource = serde_json::from_str(&json).unwrap();
        assert_eq!(back.url, full.url);
        assert_eq!(back.rev, full.rev);
        assert_eq!(back.subdirectory.as_deref(), Some("packages/core"));
        assert_eq!(back.extras, vec!["sim", "dev"]);
        assert_eq!(back.auto_data, full.auto_data);

        // Minimal — no subdirectory, no extras: those fields must be absent in JSON.
        let minimal = GitWheelSource {
            url: "https://github.com/acme/repo.git".into(),
            rev: "abcdef1234567890abcdef1234567890abcdef12".into(),
            subdirectory: None,
            extras: vec![],
            auto_data: Some(GitWheelAutoData::Disabled),
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
        assert_eq!(back_minimal.auto_data, Some(GitWheelAutoData::Disabled));

        let legacy: GitWheelSource = serde_json::from_str(&format!(
            r#"{{"url":"https://github.com/acme/repo.git","rev":"{}"}}"#,
            "ab".repeat(20),
        ))
        .unwrap();
        assert!(legacy.auto_data.is_none());
    }

    /// `SdistWheelSource` serializes and deserializes correctly.
    #[test]
    fn sdist_wheel_source_serde_roundtrip() {
        let src = SdistWheelSource {
            index: "https://pypi.org/simple/".into(),
            name: "gym".into(),
            version: "0.26.2".into(),
            sdist_url: format!(
                "https://files.pythonhosted.org/packages/gym-0.26.2.tar.gz#sha256={}",
                "ab".repeat(32)
            ),
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
        let (_, digest) = back.validated_url_and_sha256("Gym", "0.26.2.0").unwrap();
        assert_eq!(digest, "ab".repeat(32));
    }

    #[test]
    fn target_qualified_lock_names_cover_the_full_resolution_identity() {
        let x86 = target("3.11", "linux-64", None, Some((2, 35)));
        let x86_patch = target("3.11.0", "linux-64", None, Some((2, 35)));
        let python_minor = target("3.12", "linux-64", None, Some((2, 35)));
        let arm = target("3.11", "linux-aarch64", None, Some((2, 35)));
        let declared = target("3.11", "linux-64", Some((2, 35)), Some((2, 35)));
        let effective = target("3.11", "linux-64", None, Some((2, 39)));

        let name = |target: &crate::pypi::ResolutionTarget| {
            RetreadLock::file_name_for_target("demo", target)
        };
        let x86_name = name(&x86);
        assert!(x86_name.contains(&x86.resolution_identity()));
        assert_eq!(
            x86_name,
            name(&x86_patch),
            "Python patch-equivalent targets must share one lock namespace",
        );

        let distinct: std::collections::BTreeSet<_> = [
            name(&x86),
            name(&python_minor),
            name(&arm),
            name(&declared),
            name(&effective),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            distinct.len(),
            5,
            "Python minor, subdir, declared glibc, and effective glibc must each change the lock namespace",
        );

        assert_eq!(
            RetreadLock::read_file_names_for_target("demo", &x86, "linux-64"),
            vec![x86_name.clone(), RetreadLock::file_name("demo")],
        );
        assert_eq!(
            RetreadLock::read_file_names_for_target("demo", &x86_patch, "linux-64"),
            vec![x86_name, RetreadLock::file_name("demo")],
            "patch-equivalent targets must read the same candidates",
        );
        assert_eq!(
            RetreadLock::read_file_names_for_target("demo", &arm, "linux-64"),
            vec![name(&arm)],
            "foreign targets must never probe the bundle-only legacy lock",
        );
        assert_eq!(
            RetreadLock::read_file_names_for_target("demo", &x86, "linux-aarch64"),
            vec![name(&x86)],
            "linux-64 is foreign on a native aarch64 host and must not probe the legacy lock",
        );
    }

    #[test]
    fn target_qualified_lock_names_partition_exact_consumer_scope() {
        let contract = linux_64_contract(Some("2.35"), "2.35");
        let scoped = |environment: &str| {
            crate::pypi::ResolutionTarget::try_for_contract("3.11", contract.clone())
                .unwrap()
                .with_workspace_scope(crate::workspace::ResolvedWorkspaceTarget {
                    contract: contract.clone(),
                    profiles: vec!["p3".to_string()],
                    environments: vec![environment.to_string()],
                })
                .unwrap()
        };
        let old = scoped("old");
        let new = scoped("new");

        assert_ne!(
            RetreadLock::file_name_for_target("demo", &old),
            RetreadLock::file_name_for_target("demo", &new),
            "same-contract environments may have different dependency overlays and locks"
        );
        assert_eq!(
            old.compatibility_identity(),
            new.compatibility_identity(),
            "the embedded virtual-package target contract remains name-independent"
        );
    }

    #[test]
    fn persisted_scope_is_bound_to_its_exact_target_identity() {
        let contract = linux_64_contract(Some("2.35"), "2.35");
        let unscoped =
            crate::pypi::ResolutionTarget::try_for_contract("3.11", contract.clone()).unwrap();
        let scope = crate::workspace::ResolvedWorkspaceTarget {
            contract: contract.clone(),
            profiles: vec!["p3".to_string()],
            environments: vec!["new".to_string()],
        };
        let scoped = unscoped
            .clone()
            .with_workspace_scope(scope.clone())
            .unwrap();

        let mut lock = make_test_lock_unordered();
        lock.target_contract = Some(contract);
        lock.target_scope = Some(scope);
        lock.target_identity = Some(scoped.resolution_identity());
        lock.declared_glibc = scoped.declared_glibc().map(crate::glibc::format_glibc);
        lock.resolution_glibc = scoped.effective_glibc().map(crate::glibc::format_glibc);
        assert!(lock.is_for_resolution_target(&scoped));

        let mut changed_scope = lock.clone();
        changed_scope.target_scope.as_mut().unwrap().environments = vec!["other".to_string()];
        assert!(
            !changed_scope.is_for_resolution_target(&scoped),
            "a stale identity must not authorize changed persisted scope"
        );

        let mut missing_identity = lock.clone();
        missing_identity.target_identity = None;
        assert!(
            !missing_identity.is_for_resolution_target(&scoped),
            "scoped locks require their exact identity"
        );
        assert!(
            !missing_identity.is_for_resolution_target(&unscoped),
            "persisted scope must not disappear through an unscoped request"
        );

        let mut missing_scope = lock;
        missing_scope.target_scope = None;
        assert!(
            !missing_scope.is_for_resolution_target(&scoped),
            "a scoped identity must also bind reconstructable scope provenance"
        );
    }

    #[test]
    fn persisted_exact_envelope_provenance_roundtrips_and_partitions_replay() {
        let contract = linux_64_contract(Some("2.35"), "2.35");
        let scope = crate::workspace::ResolvedWorkspaceTarget {
            contract: contract.clone(),
            profiles: vec!["p3".to_string()],
            environments: vec!["new".to_string()],
        };
        let inferred = crate::pypi::ResolutionTarget::try_for_contract("3.11", contract.clone())
            .unwrap()
            .with_workspace_scope(scope.clone())
            .unwrap();
        let exact = crate::pypi::ResolutionTarget::try_for_contract("3.11", contract.clone())
            .unwrap()
            .with_exact_workspace_scope(scope.clone())
            .unwrap();
        assert_ne!(inferred.resolution_identity(), exact.resolution_identity());

        let mut lock = make_test_lock_unordered();
        lock.target_contract = Some(contract);
        lock.target_scope = Some(scope);
        lock.exact_workspace_envelope = true;
        lock.target_identity = Some(exact.resolution_identity());
        lock.declared_glibc = exact.declared_glibc().map(crate::glibc::format_glibc);
        lock.resolution_glibc = exact.effective_glibc().map(crate::glibc::format_glibc);

        let reconstructed = lock.resolution_target().unwrap();
        assert!(reconstructed.has_exact_workspace_envelope());
        assert_eq!(
            reconstructed.resolution_identity(),
            exact.resolution_identity()
        );
        assert!(lock.is_for_resolution_target(&exact));
        assert!(!lock.is_for_resolution_target(&inferred));

        let mut missing_authority = lock.clone();
        missing_authority.exact_workspace_envelope = false;
        assert!(!missing_authority.is_for_resolution_target(&exact));

        let mut missing_scope = lock;
        missing_scope.target_scope = None;
        assert!(missing_scope.resolution_target().is_err());
    }

    #[test]
    fn current_lock_replay_provenance_requires_final_and_sdist_hashes() {
        let mut lock = RetreadLock {
            schema: SCHEMA,
            retread_version: "test".into(),
            bundle: "gympack".into(),
            version: "1.0".into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
            inputs_hash: "hash".into(),
            root_requirements: vec![],
            wheels: vec![LockWheel {
                name: "gym".into(),
                version: "0.26.2".into(),
                origin: Origin::Built,
                filename: "gym-0.26.2-999retread-py3-none-any.whl".into(),
                url: None,
                sha256: Some("11".repeat(32)),
                requires_dist: vec![],
                must_ship: false,
                upstream_url: None,
                git_source: None,
                sdist_source: Some(SdistWheelSource {
                    index: "https://pypi.org/simple/".into(),
                    name: "gym".into(),
                    version: "0.26.2".into(),
                    sdist_url: "https://example.com/gym-0.26.2.tar.gz".into(),
                }),
            }],
            conda_run_deps: vec![],
            index_urls: vec![],
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            resolution_glibc: None,
            conda_capable: vec![],
            entry_specs: vec![],
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };

        let err = lock.validate_replay_provenance().unwrap_err();
        assert!(format!("{err:#}").contains("#sha256=<64 hex>"));
        lock.wheels[0].sdist_source.as_mut().unwrap().sdist_url = format!(
            "https://example.com/gym-0.26.2.tar.gz#sha256={}",
            "22".repeat(32)
        );
        lock.validate_replay_provenance().unwrap();
        lock.wheels[0].sha256 = None;
        let err = lock.validate_replay_provenance().unwrap_err();
        assert!(err.to_string().contains("no final sha256"));

        lock.wheels[0].sha256 = Some("11".repeat(32));
        lock.wheels[0].sdist_source = None;
        lock.wheels[0].must_ship = true;
        lock.wheels[0].git_source = Some(GitWheelSource {
            url: "https://github.com/acme/gym.git".into(),
            rev: "33".repeat(20),
            subdirectory: Some("../escape".into()),
            extras: vec![],
            auto_data: Some(GitWheelAutoData::Disabled),
        });
        let target = target("3.11", "linux-64", None, None);
        let err = lock
            .validate_replay_contract_for_target(&target, "gympack")
            .unwrap_err();
        assert!(format!("{err:#}").contains("unsafe git subdirectory"));

        lock.wheels[0].git_source.as_mut().unwrap().subdirectory = Some("packages/gym".into());
        lock.validate_replay_contract_for_target(&target, "gympack")
            .unwrap();

        lock.wheels[0].git_source.as_mut().unwrap().auto_data = None;
        let err = lock.validate_replay_provenance().unwrap_err();
        assert!(err.to_string().contains("Git auto-data disposition"));
        lock.wheels[0].git_source.as_mut().unwrap().auto_data =
            Some(GitWheelAutoData::CheckoutRoot {
                skip_subdirectories: vec!["../escape".into()],
            });
        let err = lock
            .validate_replay_contract_for_target(&target, "gympack")
            .unwrap_err();
        assert!(format!("{err:#}").contains("unsafe Git auto-data"));
        lock.wheels[0].git_source.as_mut().unwrap().auto_data =
            Some(GitWheelAutoData::CheckoutRoot {
                skip_subdirectories: vec!["packages/gym".into()],
            });
        lock.validate_replay_contract_for_target(&target, "gympack")
            .unwrap();

        lock.wheel_store = Some("~//tmp/evil".into());
        let err = lock
            .validate_replay_contract_for_target(&target, "gympack")
            .unwrap_err();
        assert!(err.to_string().contains("unsafe wheel-store path"));
        lock.wheel_store = Some("~/retread/wheels".into());
        lock.validate_replay_contract_for_target(&target, "gympack")
            .unwrap();
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
        assert!(lock.shadow_libs.is_empty());
        assert!(lock.declared_glibc.is_none());
    }

    fn make_test_lock_unordered() -> RetreadLock {
        RetreadLock {
            schema: SCHEMA,
            retread_version: "2.0.0".into(),
            bundle: "test-pack".into(),
            version: "1.0.0".into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
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
                        auto_data: Some(GitWheelAutoData::Disabled),
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
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            resolution_glibc: None,
            conda_capable: vec!["zlib".into(), "blas".into()],
            entry_specs: vec!["torch==2.0.0".into(), "numpy==1.26.0".into()],
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
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
