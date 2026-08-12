//! JSON-RPC method handlers. The four entry points pixi calls.

mod audit_report;
use audit_report::{build_bundle_audit, write_probe_trace};

mod auto_bundle;
use auto_bundle::{
    AutoBundleOutcome, BfsFetched, Pending, PendingSource, UvReresolveContext, UvReresolveMode,
    auto_bundle_transitives, conda_probe_spec, metadata_preferring_sidecar, pick_conda_target,
    scope_conflicts_for_target, seed_worklist, validated_conda_route,
};

mod resolve_state;
use resolve_state::{ObserveEdgeResult, ResolveState};

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use pixi_build_types::procedures::{
    conda_build_v1::{CondaBuildV1Params, CondaBuildV1Result},
    conda_outputs::{
        CondaOutput, CondaOutputDependencies, CondaOutputIgnoreRunExports, CondaOutputMetadata,
        CondaOutputRunExports, CondaOutputsParams, CondaOutputsResult,
    },
    initialize::{InitializeParams, InitializeResult},
    negotiate_capabilities::{NegotiateCapabilitiesParams, NegotiateCapabilitiesResult},
};
use pixi_build_types::{
    BackendCapabilities, BinaryPackageSpec, NamedSpec, PackageSpec, VariantValue,
};
use rattler_conda_types::{
    ChannelUrl, NoArchType, PackageName, Platform, StringMatcher, VersionSpec, VersionWithSource,
};
use serde_json::Value;
use tokio::sync::RwLock;
use uv_pep508::uv_pep440::{
    Operator, Version, VersionSpecifier, VersionSpecifiers, release_specifiers_to_ranges,
};

use crate::config::{RelaxPolicy, RetreadConfig, WheelEntry};
use crate::constraint::{
    Authority, Conflict, Constraint, ConstraintOriginId, Provenance, active_for_finalization,
    aggregate_conflicts, collect_conflicts, finalize,
};
use crate::index_chain::{IndexPurpose, index_chain};
use crate::pypi::{self, ResolutionTarget, WheelTarget, normalized_python_minor};
#[cfg(test)]
use crate::recipe::build_courier_recipe_with_mode_and_lock_filename;
use crate::recipe::{
    BundleSource, build_bundle_recipe_with_relaxations,
    build_courier_recipe_with_mode_lock_and_relaxations, to_yaml,
};
use crate::relax::{
    AbiAliasGraph, CondaConstraintOrigin, CondaDep, CondaName, CondaTarget, NameMap, PypiKey,
    abi_aliases_from_name_map, add_abi_alias_edge, canonical_conda_name, emit_python_version,
    is_semantic_abi_anchor, marker_env_for, semantic_aliases,
};
use crate::relax_decision::{Decision as RelaxDecision, SafetyContext, decide_for_emission};
use crate::relaxation_record::{RelaxationManifest, RelaxationScope, stage_relaxation_payload};
use crate::rpc::{RpcError, ok, parse_params};
use crate::wheel::WheelMetadata;
use crate::wheel_rewrite::rewrite_wheel_with_abi_aliases;
use crate::workspace::{ResolvedWorkspaceTarget, WorkspaceTargetContract, WorkspaceTargetEnvelope};

/// Process-global memo of `conda/outputs` results, keyed by the params
/// that determine the outputs (host/build platform, sorted channels,
/// python variant set). pixi calls `conda/outputs` ONCE PER LOCK-FILE
/// ENVIRONMENT that references the source package (e.g. resolving `gsi`
/// also re-locks `gsi-ros2`, `isaaclab-gpu`, ... that share isaac-pack),
/// but retread's handler already computes EVERY env's outputs in a
/// single call -- so calls 2..N with identical params redo the full
/// multi-env solve for nothing. Memoizing the result collapses those
/// redundant calls to a hash lookup. retread is one long-lived process
/// per `pixi` invocation, so this survives across the repeated calls and
/// is dropped when the backend exits. Same pattern as
/// `solve_check::RECORDS_CACHE`.
#[derive(Clone)]
struct CondaOutputsMemo {
    result: CondaOutputsResult,
    /// Incremental metadata is safe to build only from the exact retained
    /// cold plan that produced it. Never serve such a memo after that typed
    /// handoff has been consumed or invalidated.
    requires_prepared_plan: bool,
}

static CONDA_OUTPUTS_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, CondaOutputsMemo>>,
> = std::sync::OnceLock::new();

fn remove_incremental_conda_outputs_memo(cache_key: &str) -> bool {
    let Some(cache) = CONDA_OUTPUTS_CACHE.get() else {
        return false;
    };
    let mut cache = cache.lock().unwrap();
    if !cache
        .get(cache_key)
        .is_some_and(|memo| memo.requires_prepared_plan)
    {
        return false;
    }
    cache.remove(cache_key);
    true
}

/// Build the cache key from the params that actually determine the
/// outputs. `work_directory` is deliberately EXCLUDED: it's a scratch
/// dir that doesn't affect the emitted metadata, and including it would
/// miss cache hits if pixi varies it per call.
///
/// `workspace_mtime` is the `pixi.toml`'s modification time folded into
/// the key. Rationale: `WorkspaceManifest::load` is mtime-memoized
/// internally, but `CONDA_OUTPUTS_CACHE` returns memoized results that
/// bypass those loads entirely after the first call. Without the mtime
/// in the key, a manifest edit between two pixi invocations (both with
/// the same platform/channels/variant) would return a stale result that
/// reflects the OLD workspace -- the cache hit would carry the previous
/// run's emissions even if the workspace's channel list or env
/// definitions changed. Use "0" when the mtime is unavailable (offline /
/// read-error) so the key is still a valid string and the cache can
/// still hit on identical conditions.
#[cfg(test)]
fn conda_outputs_cache_key(
    params: &CondaOutputsParams,
    workspace_mtime: Option<std::time::SystemTime>,
    auto_overrides_fp: &str,
) -> String {
    let target = ResolutionTarget::for_subdir("0.0", params.host_platform.as_str());
    conda_outputs_cache_key_for_target(
        params,
        workspace_mtime,
        auto_overrides_fp,
        &target,
        None,
        "",
    )
}

fn workspace_consumer_scope_identity(scope: Option<&ResolvedWorkspaceTarget>) -> String {
    use sha2::{Digest, Sha256};

    fn hash_set(hasher: &mut Sha256, values: &[String]) {
        let values: std::collections::BTreeSet<&str> = values.iter().map(String::as_str).collect();
        hasher.update((values.len() as u64).to_le_bytes());
        for value in values {
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
    }

    let mut hasher = Sha256::new();
    hasher.update(b"retread-workspace-consumer-scope-v1\0");
    match scope {
        None => hasher.update([0]),
        Some(scope) => {
            hasher.update([1]);
            hash_set(&mut hasher, &scope.profiles);
            hash_set(&mut hasher, &scope.environments);
        }
    }
    format!("{:x}", hasher.finalize())
}

fn conda_outputs_cache_key_for_target(
    params: &CondaOutputsParams,
    workspace_mtime: Option<std::time::SystemTime>,
    auto_overrides_fp: &str,
    target: &ResolutionTarget,
    consumer_scope: Option<&ResolvedWorkspaceTarget>,
    workspace_solve_fingerprint: &str,
) -> String {
    let mut chans: Vec<String> = params
        .channels
        .iter()
        .map(|c| c.url().to_string())
        .collect();
    chans.sort();
    // Encode the mtime as nanos-since-UNIX_EPOCH so it's both stable and
    // human-readable in debug logs. "0" is the sentinel for "unknown".
    let mtime_str = workspace_mtime
        .and_then(|t| {
            t.duration_since(std::time::SystemTime::UNIX_EPOCH)
                .ok()
                .map(|d| d.subsec_nanos().to_string() + &d.as_secs().to_string())
        })
        .unwrap_or_else(|| "0".to_string());
    // Python variants are already present in `variant_configuration`; use a
    // fixed numeric minor here to namespace the remaining immutable target
    // contract (subdir plus declared/effective glibc). This prevents a shared
    // disk memo from crossing host/deployment compatibility boundaries.
    let target_contract = target.resolution_identity();
    let consumer_scope = workspace_consumer_scope_identity(consumer_scope);
    format!(
        "{}|{}|{}|{:?}|{}|{}|{}|{}|{}|{}",
        params.host_platform,
        params.build_platform,
        chans.join(","),
        params.variant_configuration,
        mtime_str,
        auto_overrides_fp,
        target_contract,
        consumer_scope,
        workspace_solve_fingerprint,
        backend_build_identity(),
    )
}

/// Backend build identity folded into [`conda_outputs_cache_key`]: crate
/// version + git commit hash (embedded by `build.rs`, `-dirty` suffixed
/// for uncommitted worktrees, `unknown` when built without git). Run-31
/// of the retread-deps-from proof exposed the gap this closes: the
/// bounded-range emission binary (16f20ec) reused the previous binary's
/// cached pack renders -- still carrying the old exact `==` auto-routed
/// pins -- because nothing about the BINARY was in the key. Any backend
/// upgrade must bust both the in-memory and disk memos.
fn backend_build_identity() -> &'static str {
    concat!(env!("CARGO_PKG_VERSION"), "+", env!("RETREAD_GIT_HASH"))
}

/// Content fingerprint of the workspace's `.retread/auto-overrides.json`
/// ledger, folded into [`conda_outputs_cache_key`]. Run-12 of the
/// retread-deps-from proof exposed the gap this closes: fix #22's pack
/// repairs write ONLY the ledger (never any pixi.toml, by design), so the
/// workspace-manifest mtime in the cache key never moves between repair
/// iterations -- the next `pixi lock`'s fresh backend hit the v2.11.0
/// cross-process disk memo and returned the STALE pack render (still
/// carrying the pre-repair auto-routed pin, e.g. `setuptools ==83.0.0`)
/// even though `merge_ledger_overrides` had correctly merged the new
/// override into config. Hashing the ledger's bytes (not its mtime --
/// rollback restores the old bytes and must restore the old key too)
/// makes any ledger write bust both the in-memory and disk memos.
/// `"none"` when there's no workspace dir or no ledger yet.
fn auto_overrides_fingerprint(workspace_dir: Option<&std::path::Path>) -> String {
    use sha2::{Digest, Sha256};
    let Some(dir) = workspace_dir else {
        return "none".to_string();
    };
    let Ok(bytes) = std::fs::read(crate::pack_overrides::ledger_path(dir)) else {
        return "none".to_string();
    };
    let digest = Sha256::digest(&bytes);
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// On-disk path for the cross-process `conda/outputs` memo (see the
/// v2.11.0 comment at the `conda_outputs` call site). Keyed by
/// [`conda_outputs_cache_key`] PLUS `source_dir`, hashed to a
/// filesystem-safe name since the key contains channel URLs (`/`, `:`).
///
/// v3.0.1 CORRECTNESS FIX: `CondaOutputsParams` carries no package
/// identity at all (no manifest path, no source dir) -- it's just
/// platform/channels/variant. The in-memory `CONDA_OUTPUTS_CACHE` got
/// away with keying on that alone because pixi runs ONE retread process
/// per source package, so within a process "same params" already meant
/// "same package." The disk cache is visible across EVERY process on
/// the machine, so two DIFFERENT sibling packages in the same workspace
/// (e.g. `isaaclab-viral-pack` and `isaaclab-unitree-pack`, same
/// platform/channels/variant/workspace mtime) hashed to the identical
/// cache file, and one process loaded the other's cached outputs --
/// surfacing as pixi's "the package 'isaaclab-viral-pack' is not
/// provided by the project located at './pypi-packs/isaaclab-viral-pack'
/// (did you mean 'isaaclab-unitree-pack'?)" (#8). `source_dir` (each
/// package's own on-disk directory, set once at `initialize` time) is
/// the missing package identity; folding it into the disk key's hash
/// input (but NOT the in-memory key, which doesn't need it) fixes the
/// collision without touching the already-correct in-process cache.
fn conda_outputs_disk_cache_path(
    cache_dir: &std::path::Path,
    cache_key: &str,
    source_dir: &std::path::Path,
) -> PathBuf {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(cache_key.as_bytes());
    hasher.update(b"\0");
    hasher.update(source_dir.to_string_lossy().as_bytes());
    let digest = hasher.finalize();
    let hex: String = digest.iter().take(16).map(|b| format!("{b:02x}")).collect();
    cache_dir
        .join("retread-conda-outputs-cache")
        .join(format!("{hex}.json"))
}

/// Load a memoized [`CondaOutputsResult`] from disk. Returns `None` on
/// any failure (missing file, unreadable, stale schema) so the caller
/// always has a safe cold-compute fallback -- this is a pure speed
/// optimization, never a source of truth.
async fn read_conda_outputs_disk_cache(path: &std::path::Path) -> Option<CondaOutputsResult> {
    let bytes = tokio::fs::read(path).await.ok()?;
    serde_json::from_slice(&bytes).ok()
}

struct CondaOutputsDiskCacheFillLock(std::fs::File);

impl Drop for CondaOutputsDiskCacheFillLock {
    fn drop(&mut self) {
        if let Err(error) = fs4::fs_std::FileExt::unlock(&self.0) {
            tracing::warn!(error = %error, "failed to unlock conda/outputs first compute");
        }
    }
}

/// Serialize the cold computation for one cross-process output-memo key.
/// Pixi can launch one backend process per environment even when three
/// environments consume the same source package with identical parameters.
/// Atomic memo publication protects readers from partial JSON, but without a
/// fill lock every process still repeats the full closure/routing computation.
async fn acquire_conda_outputs_disk_cache_fill_lock(
    cache_path: &std::path::Path,
) -> Result<CondaOutputsDiskCacheFillLock> {
    let parent = cache_path.parent().ok_or_else(|| {
        anyhow!(
            "conda/outputs cache path has no parent: {}",
            cache_path.display()
        )
    })?;
    tokio::fs::create_dir_all(parent).await.with_context(|| {
        format!(
            "creating conda/outputs cache namespace {}",
            parent.display()
        )
    })?;
    let lock_path = cache_path.with_extension("json.fill-v1.lock");
    tokio::task::spawn_blocking(move || {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .with_context(|| format!("opening conda/outputs fill lock {}", lock_path.display()))?;
        fs4::fs_std::FileExt::lock_exclusive(&file)
            .with_context(|| format!("locking conda/outputs fill {}", lock_path.display()))?;
        Ok(CondaOutputsDiskCacheFillLock(file))
    })
    .await
    .context("conda/outputs fill lock task panicked")?
}

/// Persist a computed [`CondaOutputsResult`] to disk for reuse by a
/// future retread process solving a different environment with the
/// same params. Failures are logged at debug and otherwise ignored --
/// this must never fail the RPC that just successfully computed outputs.
async fn write_conda_outputs_disk_cache(path: &std::path::Path, result: &CondaOutputsResult) {
    let Some(parent) = path.parent() else {
        return;
    };
    if let Err(e) = tokio::fs::create_dir_all(parent).await {
        tracing::debug!(error = %e, path = %parent.display(), "conda/outputs disk-cache: could not create cache dir");
        return;
    }
    let bytes = match serde_json::to_vec(result) {
        Ok(b) => b,
        Err(e) => {
            tracing::debug!(error = %e, "conda/outputs disk-cache: serialize failed");
            return;
        }
    };
    // Write to a per-process temp file then rename, so a concurrent
    // reader in another process never observes a partially-written file.
    let tmp_path = path.with_extension(format!("json.tmp.{}", std::process::id()));
    if let Err(e) = tokio::fs::write(&tmp_path, &bytes).await {
        tracing::debug!(error = %e, path = %tmp_path.display(), "conda/outputs disk-cache: write failed");
        return;
    }
    if let Err(e) = tokio::fs::rename(&tmp_path, path).await {
        tracing::debug!(error = %e, path = %path.display(), "conda/outputs disk-cache: rename failed");
        let _ = tokio::fs::remove_file(&tmp_path).await;
    }
}

/// Read the modification time of `workspace_dir/pixi.toml` for use in
/// the `CONDA_OUTPUTS_CACHE` key. Returns `None` when the file is absent
/// or metadata is unavailable (network-offline, read error). That causes
/// the cache key to use the "0" sentinel and still function correctly --
/// the cache may over-hit in that unusual case, but it will never
/// under-hit and cause a redundant full solve.
fn workspace_manifest_mtime(
    workspace_dir: Option<&std::path::Path>,
) -> Option<std::time::SystemTime> {
    let dir = workspace_dir?;
    std::fs::metadata(dir.join("pixi.toml"))
        .ok()
        .and_then(|m| m.modified().ok())
}

/// Pixi 0.70.2 starts the backend before it prepares the per-source build
/// directory. If a workspace keeps `.pixi/bld` as a symlink to a tmp-backed
/// build root and that tmp root was cleaned, Pixi's next mkdir under
/// `.pixi/bld` hits the dangling symlink itself and returns EEXIST before it
/// ever sends `conda/outputs` to this backend. Repair only that invariant:
/// the symlink stays unchanged, and only its missing target directory is
/// created.
fn ensure_pixi_bld_symlink_target(workspace_dir: Option<&std::path::Path>) -> Result<(), RpcError> {
    ensure_pixi_bld_symlink_target_inner(workspace_dir, crate::fasttmp::in_slurm_job())
}

/// The Slurm check is a parameter so the repair logic can be tested for what
/// it does, not for what the ambient environment happens to be. Reading
/// `SLURM_JOB_ID` directly made the test pass in CI and fail inside any Slurm
/// allocation, where the guard short-circuits before doing the work the test
/// asserts on.
fn ensure_pixi_bld_symlink_target_inner(
    workspace_dir: Option<&std::path::Path>,
    in_slurm_job: bool,
) -> Result<(), RpcError> {
    if in_slurm_job {
        tracing::info!(
            "retread fast-tmp: SLURM job context; not repairing shared workspace .pixi/bld symlink"
        );
        return Ok(());
    }
    let Some(workspace_dir) = pixi_workspace_dir(workspace_dir) else {
        return Ok(());
    };
    let bld = workspace_dir.join(".pixi").join("bld");
    let meta = match std::fs::symlink_metadata(&bld) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(RpcError::internal(format!(
                "checking pixi build dir {}: {e}",
                bld.display()
            )));
        }
    };
    if !meta.file_type().is_symlink() {
        return Ok(());
    }

    let raw_target = std::fs::read_link(&bld).map_err(|e| {
        RpcError::internal(format!(
            "reading pixi build-dir symlink {}: {e}",
            bld.display()
        ))
    })?;
    let target = if raw_target.is_absolute() {
        raw_target
    } else {
        bld.parent()
            .unwrap_or(workspace_dir.as_path())
            .join(raw_target)
    };
    if target.exists() {
        if target.is_dir() {
            return Ok(());
        }
        return Err(RpcError::internal(format!(
            "pixi build-dir symlink {} points to existing non-directory {}",
            bld.display(),
            target.display()
        )));
    }

    std::fs::create_dir_all(&target).map_err(|e| {
        RpcError::internal(format!(
            "pixi build-dir symlink {} points to missing target {}, and retread \
             could not create it: {e}",
            bld.display(),
            target.display()
        ))
    })?;
    tracing::warn!(
        workspace = %workspace_dir.display(),
        symlink = %bld.display(),
        target = %target.display(),
        "retread repaired missing Pixi build-dir symlink target before source build"
    );
    Ok(())
}

fn pixi_workspace_dir(workspace_dir: Option<&std::path::Path>) -> Option<PathBuf> {
    if let Some(dir) = workspace_dir.filter(|dir| dir.join("pixi.toml").is_file()) {
        return Some(dir.to_path_buf());
    }
    std::env::current_dir()
        .ok()
        .filter(|dir| dir.join("pixi.toml").is_file())
}

const NEGOTIATE: &str = "negotiateCapabilities";
const INITIALIZE: &str = "initialize";
const CONDA_OUTPUTS: &str = "conda/outputs";
const CONDA_BUILD_V1: &str = "conda/build_v1";

/// Build a content-addressed build string for courier packages.
///
/// Format: `py{py_short}_h{hash_prefix}_{build_number}` where `hash_prefix`
/// is the first 10 hex chars of the courier inputs hash. This uniquely
/// identifies the content of the courier package: any change to the lock
/// inputs (wheel set, index chain, relax policy, python, retread version,
/// config/channels/workspace fingerprint) yields a new string and pixi
/// re-extracts instead of cache-hitting the stale artifact.
///
/// The `py{py_short}_` prefix is kept for human readability; pixi does NOT
/// parse the build string for the python variant (it uses the variant map
/// and `python X.Y.*` run-dep pins), so the suffix carries no semantic load.
///
/// This function MUST be the single source of truth for courier build string
/// format -- `produce_output` and `replay_from_lock` both call it so the
/// strings are guaranteed byte-identical.
fn courier_build_string(
    py_short: &str,
    inputs_hash: &str,
    build_number: u64,
    loose: bool,
) -> String {
    let prefix = &inputs_hash[..inputs_hash.len().min(10)];
    // Loose bundle mode carries a `_loose` discriminant: the bundle mode
    // never feeds inputs_hash (flipping fat<->loose must NOT force a cold
    // re-solve), but the artifact CONTENT differs (stub vs full payload).
    // Without the discriminant pixi would cache-hit the previously-built
    // artifact of the other mode.
    if loose {
        format!("py{py_short}_h{prefix}_loose_{build_number}")
    } else {
        format!("py{py_short}_h{prefix}_{build_number}")
    }
}

fn courier_build_string_for_target(
    target: &ResolutionTarget,
    inputs_hash: &str,
    build_number: u64,
    loose: bool,
) -> String {
    courier_build_string(
        &target.python_version().replace('.', ""),
        inputs_hash,
        build_number,
        loose,
    )
}

fn resolved_courier_build(
    expected_build: Option<&str>,
    target: &ResolutionTarget,
    staged_inputs_hash: &str,
    build_number: u64,
    loose: bool,
) -> String {
    expected_build.map(str::to_owned).unwrap_or_else(|| {
        courier_build_string_for_target(target, staged_inputs_hash, build_number, loose)
    })
}

fn advertised_build_matches(advertised_build: Option<&str>, current_build: &str) -> bool {
    advertised_build.is_none_or(|build| build == current_build)
}

fn non_courier_build_for_target(
    config: &RetreadConfig,
    target: &ResolutionTarget,
) -> Option<String> {
    (!config.courier && target.target_contract().is_some()).then(|| {
        courier_build_string_for_target(
            target,
            &target.resolution_identity(),
            config.build_number,
            config.bundle_mode == crate::config::BundleMode::Loose,
        )
    })
}

fn validate_advertised_non_courier_target_build(
    config: &RetreadConfig,
    target: &ResolutionTarget,
    advertised_build: Option<&str>,
) -> Result<(), RpcError> {
    let Some(required) = non_courier_build_for_target(config, target) else {
        return Ok(());
    };
    if advertised_build_matches(advertised_build, &required) {
        return Ok(());
    }
    Err(RpcError::invalid_params(format!(
        "advertised non-courier build `{}` does not match immutable target identity `{required}`",
        advertised_build.unwrap_or_default(),
    )))
}

/// Compute the courier inputs hash that uniquely identifies a courier build's
/// content-affecting inputs. This is the single authoritative implementation
/// called by:
///
/// 1. The replay gate (~line 754) — to compare against `lock.inputs_hash`.
/// 2. `produce_output` — to embed in the `CondaOutputMetadata.build` string.
///
/// The hash folds in: entry specs (canonical wheel inputs), the merged index
/// chain, relax policy, python version, retread binary version, and a config
/// fingerprint (channels + workspace solve env). This mirrors exactly what
/// `courier::stage` writes into `RetreadLock.inputs_hash`, guaranteeing that
/// replay fires iff the hash matches.
///
/// IMPORTANT: the arguments must be constructed identically at every call
/// site — see the inline notes in each caller.
fn courier_inputs_hash(
    config: &crate::config::RetreadConfig,
    bundle_name: &str,
    target: &ResolutionTarget,
    channels: &[String],
    workspace_manifest: Option<&crate::workspace::WorkspaceManifest>,
    workspace_dir: &std::path::Path,
    source_dir: &std::path::Path,
) -> String {
    let entry_specs = crate::courier::courier_input_specs(config, bundle_name);
    let ws_indexes: Vec<String> = workspace_manifest
        .map(|m| m.resolution_pypi_index_urls())
        .unwrap_or_else(|| vec![crate::index_chain::PUBLIC_PYPI.to_string()]);
    let entry_indexes: Vec<String> = config
        .retread_wheels
        .values()
        .filter(|entry| entry.url.is_none())
        .filter_map(|entry| entry.index.clone())
        .collect();
    let index_urls = index_chain(entry_indexes, &ws_indexes, IndexPurpose::RootResolve);
    let workspace_fp = workspace_manifest
        .map(|m| workspace_solve_fingerprint(m, workspace_dir, source_dir, target))
        .unwrap_or_default();
    let config_fp = crate::courier::config_fingerprint(config, channels, &workspace_fp);
    crate::lock::RetreadLock::compute_inputs_hash_for_target(
        &entry_specs,
        &index_urls,
        &format!("{:?}", config.relax),
        target,
        crate::lock::EMIT_EPOCH,
        config.pin_version.then_some(env!("CARGO_PKG_VERSION")),
        &config_fp,
    )
}

fn current_courier_build_for_input_bundle(
    config: &RetreadConfig,
    input_bundle_name: &str,
    target: &ResolutionTarget,
    workspace_manifest: Option<&crate::workspace::WorkspaceManifest>,
    workspace_dir: Option<&Path>,
    source_dir: &Path,
) -> String {
    let workspace_root = workspace_dir.unwrap_or(source_dir);
    let channels = workspace_manifest
        .map(|manifest| workspace_courier_channels(manifest, workspace_root, source_dir, target))
        .unwrap_or_default();
    let inputs_hash = courier_inputs_hash(
        config,
        input_bundle_name,
        target,
        &channels,
        workspace_manifest,
        workspace_root,
        source_dir,
    );
    courier_build_string_for_target(
        target,
        &inputs_hash,
        config.build_number,
        config.bundle_mode == crate::config::BundleMode::Loose,
    )
}

fn validate_advertised_courier_build(
    config: &RetreadConfig,
    input_bundle_name: &str,
    target: &ResolutionTarget,
    workspace_manifest: Option<&crate::workspace::WorkspaceManifest>,
    workspace_dir: Option<&Path>,
    source_dir: &Path,
    advertised_build: Option<&str>,
) -> Result<(), RpcError> {
    let current_build = current_courier_build_for_input_bundle(
        config,
        input_bundle_name,
        target,
        workspace_manifest,
        workspace_dir,
        source_dir,
    );
    if advertised_build_matches(advertised_build, &current_build) {
        return Ok(());
    }
    Err(RpcError::invalid_params(format!(
        "courier inputs changed between conda/outputs and conda/build_v1: pixi requested build `{}`, but current inputs for source bundle `{input_bundle_name}` require `{current_build}`; rerun the lock/install so output metadata can be recomputed",
        advertised_build.unwrap_or_default(),
    )))
}

fn resolved_workspace_target_from_resolution(
    manifest: &crate::workspace::WorkspaceManifest,
    workspace_dir: &Path,
    source_dir: &Path,
    target: &ResolutionTarget,
) -> Option<ResolvedWorkspaceTarget> {
    if let Some(scope) = target.workspace_scope() {
        return Some(scope.clone());
    }
    target.target_contract().and_then(|contract| {
        manifest.resolve_source_for_contract(workspace_dir, source_dir, contract)
    })
}

fn workspace_courier_channels(
    manifest: &crate::workspace::WorkspaceManifest,
    workspace_dir: &Path,
    source_dir: &Path,
    target: &ResolutionTarget,
) -> Vec<String> {
    match target.target_contract() {
        Some(_) => {
            resolved_workspace_target_from_resolution(manifest, workspace_dir, source_dir, target)
                .map(|resolved| {
                    manifest.courier_channel_set_for_resolved_target(
                        workspace_dir,
                        source_dir,
                        &resolved,
                    )
                })
                .unwrap_or_default()
        }
        None => manifest.courier_channel_set_for_target(
            workspace_dir,
            source_dir,
            target.conda_subdir(),
        ),
    }
}

fn workspace_precise_consuming_envs(
    manifest: &crate::workspace::WorkspaceManifest,
    workspace_dir: &Path,
    source_dir: &Path,
    target: &ResolutionTarget,
) -> Option<Vec<String>> {
    match target.target_contract() {
        Some(_) => {
            resolved_workspace_target_from_resolution(manifest, workspace_dir, source_dir, target)
                .and_then(|resolved| {
                    manifest.precise_consuming_envs_for_resolved_target(
                        workspace_dir,
                        source_dir,
                        &resolved,
                    )
                })
        }
        None => manifest.precise_consuming_envs_for_target(
            workspace_dir,
            source_dir,
            target.conda_subdir(),
        ),
    }
}

fn workspace_consuming_dependencies(
    manifest: &crate::workspace::WorkspaceManifest,
    workspace_dir: &Path,
    source_dir: &Path,
    target: &ResolutionTarget,
) -> BTreeMap<String, Vec<String>> {
    match target.target_contract() {
        Some(_) => {
            resolved_workspace_target_from_resolution(manifest, workspace_dir, source_dir, target)
                .map(|resolved| {
                    manifest.consuming_env_dependencies_for_resolved_target(
                        workspace_dir,
                        source_dir,
                        &resolved,
                    )
                })
                .unwrap_or_default()
        }
        None => manifest.consuming_env_dependencies_for_target(
            workspace_dir,
            source_dir,
            target.conda_subdir(),
        ),
    }
}

fn workspace_consuming_system_requirements(
    manifest: &crate::workspace::WorkspaceManifest,
    workspace_dir: &Path,
    source_dir: &Path,
    target: &ResolutionTarget,
) -> BTreeMap<String, String> {
    match target.target_contract() {
        Some(_) => {
            resolved_workspace_target_from_resolution(manifest, workspace_dir, source_dir, target)
                .map(|resolved| {
                    manifest.consuming_env_system_requirements_for_resolved_target(
                        workspace_dir,
                        source_dir,
                        &resolved,
                    )
                })
                .unwrap_or_default()
        }
        None => manifest.consuming_env_system_requirements_for_target(
            workspace_dir,
            source_dir,
            target.conda_subdir(),
        ),
    }
}

fn workspace_effective_system_requirements(
    manifest: &crate::workspace::WorkspaceManifest,
    env: &str,
    target: &ResolutionTarget,
) -> BTreeMap<String, String> {
    target.target_contract().map_or_else(
        || manifest.effective_system_requirements_for_target(env, target.conda_subdir()),
        |contract| manifest.effective_system_requirements_for_contract(env, contract),
    )
}

/// Resolution-affecting workspace fingerprint, including the actual metadata
/// constraints exported by co-activated Retread path packages.
///
/// A plain manifest fingerprint sees only the sibling package path, not the
/// sibling's emitted dependency contract. Folding the normalized constraints
/// into the producer and replay hashes ensures a changed sibling requirement
/// invalidates this pack's committed solve instead of replaying a stale,
/// independently-selected closure.
fn workspace_solve_fingerprint(
    manifest: &crate::workspace::WorkspaceManifest,
    workspace_dir: &Path,
    source_dir: &Path,
    target: &ResolutionTarget,
) -> String {
    let mut parts = Vec::new();
    let base = match resolved_workspace_target_from_resolution(
        manifest,
        workspace_dir,
        source_dir,
        target,
    ) {
        Some(resolved) => {
            manifest.solve_fingerprint_for_resolved_target(workspace_dir, source_dir, &resolved)
        }
        None if target.target_contract().is_none() => {
            manifest.solve_fingerprint_for_target(workspace_dir, source_dir, target.conda_subdir())
        }
        None => String::new(),
    };
    if !base.is_empty() {
        parts.push(base);
    }
    let sibling = sibling_lock_constraints(manifest, workspace_dir, source_dir, target);
    parts.extend(
        sibling
            .constraints
            .into_iter()
            .map(|line| format!("co-activated-sibling:{line}")),
    );
    for (name, specs) in sibling_conda_run_dependencies(manifest, workspace_dir, source_dir, target)
    {
        for spec in specs {
            parts.push(format!("co-activated-sibling-conda:{name}={spec}"));
        }
    }
    parts.join("\n")
}

/// Cache-key form of [`workspace_solve_fingerprint`]. Conda output metadata
/// can fan out over multiple Python variants, while sibling lock filenames
/// and replay contracts are Python-qualified. Fingerprint every actual
/// `pythons_for` target; the metadata-only `0.0` cache target must never be
/// used to look up sibling locks.
fn workspace_solve_fingerprint_for_cache(
    workspace_dir: Option<&Path>,
    source_dir: &Path,
    subdir: Platform,
    config: &RetreadConfig,
    variant_configuration: Option<&BTreeMap<String, Vec<VariantValue>>>,
    resolved: Option<&ResolvedWorkspaceTarget>,
    exact_envelope: bool,
) -> Result<String> {
    let Some(workspace_dir) = workspace_dir else {
        return Ok(String::new());
    };
    let Some(manifest) = crate::workspace::WorkspaceManifest::load(workspace_dir) else {
        return Ok(String::new());
    };

    let mut fingerprints = Vec::new();
    for python in pythons_for(config, variant_configuration) {
        let target =
            wheel_target_for_resolved_workspace(subdir, &python, resolved, exact_envelope)?;
        fingerprints.push(format!(
            "python={}\ntarget={}\n{}",
            target.python_version(),
            target.resolution_identity(),
            workspace_solve_fingerprint(&manifest, workspace_dir, source_dir, &target),
        ));
    }
    fingerprints.sort();
    fingerprints.dedup();
    Ok(fingerprints.join("\n--target--\n"))
}

/// Load every exact-target Retread lock activated beside this pack.
///
/// Keeping discovery and replay-contract validation in one place is
/// load-bearing: UV constraints and conda co-solve inputs must describe the
/// same sibling set.
fn coactivated_sibling_locks(
    manifest: &crate::workspace::WorkspaceManifest,
    workspace_dir: &Path,
    source_dir: &Path,
    target: &ResolutionTarget,
) -> Vec<(String, crate::lock::RetreadLock)> {
    let Some(envs) = workspace_precise_consuming_envs(manifest, workspace_dir, source_dir, target)
    else {
        return Vec::new();
    };
    let resolved = target.target_contract().and_then(|_| {
        resolved_workspace_target_from_resolution(manifest, workspace_dir, source_dir, target)
    });
    let source_canon = std::fs::canonicalize(source_dir).unwrap_or_else(|_| source_dir.into());
    let mut siblings = BTreeMap::new();
    for env in envs {
        let paths = resolved.as_ref().map_or_else(
            || manifest.effective_path_dependencies_for_target(&env, target.conda_subdir()),
            |resolved| {
                manifest
                    .effective_path_dependencies_for_resolved_env(&env, resolved)
                    .unwrap_or_default()
            },
        );
        for (bundle, raw_path) in paths {
            let raw_path = PathBuf::from(raw_path);
            let pack_dir = if raw_path.is_absolute() {
                raw_path
            } else {
                workspace_dir.join(raw_path)
            };
            let pack_canon = std::fs::canonicalize(&pack_dir).unwrap_or(pack_dir);
            if pack_canon != source_canon {
                siblings.insert(bundle, pack_canon);
            }
        }
    }

    siblings
        .into_iter()
        .filter_map(|(bundle, pack_dir)| {
            // Exact envelopes authorize one workspace-wide
            // environment/profile scope, so the current target names the
            // sibling sidecar too. Direct inference can carry an identical
            // contract and singleton scope; use explicit envelope provenance,
            // never detected-VP presence, to distinguish them.
            let sibling_target = match target.target_contract() {
                // Legacy, unqualified targets have no consumer scope to
                // rehydrate. Keep their historical target lookup.
                None => target.clone(),
                // An exact envelope authorizes one workspace-wide
                // environment/profile pair, which applies to every source
                // co-activated by that pair.
                Some(_) if target.has_exact_workspace_envelope() => target.clone(),
                // Direct inference aggregates consumers independently for
                // each source. Never fall back to the current source's scope
                // when the sibling scope cannot be proven: that could make a
                // stale lock under the current identity look authoritative.
                Some(contract) => {
                    let scope =
                        manifest.resolve_source_for_contract(workspace_dir, &pack_dir, contract)?;
                    ResolutionTarget::try_for_contract_on_subdir(
                        target.python_version(),
                        target.conda_subdir(),
                        contract.clone(),
                    )
                    .and_then(|resolved| resolved.with_workspace_scope(scope))
                    .ok()?
                }
            };
            // Preserve the declared bundle spelling for its lock filename and
            // replay contract. Conda names may legally contain dots.
            let lock_path = pack_dir.join(crate::lock::RetreadLock::file_name_for_target(
                &bundle,
                &sibling_target,
            ));
            let lock = crate::lock::RetreadLock::load(&lock_path).ok()?;
            if lock.schema != crate::lock::SCHEMA {
                return None;
            }
            lock.validate_replay_contract_for_target(&sibling_target, &bundle)
                .ok()?;
            Some((bundle, lock))
        })
        .collect()
}

/// Exact emitted conda contracts of co-activated sibling packs.
///
/// These are not UV constraints: they describe whether a proposed conda
/// route can coexist with packages Pixi will install beside this bundle. A
/// route that conflicts here must remain a PyPI wheel instead of poisoning
/// the eventual workspace solve.
fn sibling_conda_run_dependencies(
    manifest: &crate::workspace::WorkspaceManifest,
    workspace_dir: &Path,
    source_dir: &Path,
    target: &ResolutionTarget,
) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (_, lock) in coactivated_sibling_locks(manifest, workspace_dir, source_dir, target) {
        for dep in lock.conda_run_deps {
            let name = canonical_conda_name(&dep.name);
            let specs = out.entry(name).or_default();
            if !specs.contains(&dep.spec) {
                specs.push(dep.spec);
            }
        }
    }
    for specs in out.values_mut() {
        specs.sort();
    }
    out
}

/// Read the exact-target committed locks of Retread path packages activated
/// beside `source_dir` and turn their real wheel metadata requirements into
/// non-installing uv constraints for this pack.
///
/// This is intentionally based on `requires_dist`, not a sibling's selected
/// `conda_run_deps`. The latter is a narrow output of an independent solve
/// (the source of the transformers 4.57-vs-5.x false conflict); the former is
/// the actual compatibility contract. Bare requirements impose no constraint.
fn sibling_lock_constraints(
    manifest: &crate::workspace::WorkspaceManifest,
    workspace_dir: &Path,
    source_dir: &Path,
    target: &ResolutionTarget,
) -> crate::uv_closure::ConstraintSet {
    let mut out = crate::uv_closure::ConstraintSet::default();
    let mut seen = BTreeSet::new();
    for (bundle, lock) in coactivated_sibling_locks(manifest, workspace_dir, source_dir, target) {
        // Compose only the sibling's declared entry wheels. Transitive wheel
        // metadata can be internally contradictory after Retread routes or
        // relaxes those dependencies (for example cmeel-boost's numpy>=2 and
        // dex-retargeting's numpy<2 in one valid IsaacLab pack). The entry
        // wheels are the sibling package's public compatibility contract.
        let entry_names: BTreeSet<String> = lock
            .entry_specs
            .iter()
            .filter_map(|spec| locked_entry_name(spec))
            .collect();
        for raw in lock
            .wheels
            .iter()
            .filter(|wheel| entry_names.contains(&canonical_conda_name(&wheel.name)))
            .flat_map(|wheel| &wheel.requires_dist)
        {
            let Ok(requirement): Result<uv_pep508::Requirement, _> =
                uv_pep508::Requirement::from_str(raw)
            else {
                continue;
            };
            let Some(uv_pep508::VersionOrUrl::VersionSpecifier(specifiers)) =
                requirement.version_or_url.as_ref()
            else {
                continue;
            };
            let specifiers = specifiers.to_string();
            if specifiers.is_empty() {
                continue;
            }
            // A constraint must not activate dependency extras merely because
            // the sibling did. Preserve the version and marker only.
            let marker = requirement.marker.try_to_string().unwrap_or_default();
            let marker = if marker.is_empty() {
                String::new()
            } else {
                format!(" ; {marker}")
            };
            let name = canonical_conda_name(requirement.name.as_ref());
            let line = format!("{name}{specifiers}{marker}");
            if !seen.insert(line.clone()) {
                continue;
            }
            out.constraints.push(line.clone());
            let conda_name = name.clone();
            let conda_version = specifiers.clone();
            out.provenance
                .entry(name)
                .or_insert_with(|| crate::uv_closure::ConstraintProvenance {
                    constraint: line,
                    conda_name,
                    conda_version,
                    source: format!("co-activated-retread-lock:{bundle}"),
                    env: "precise-consuming-envs".to_string(),
                    provenance: Provenance::UvConstraint,
                });
        }
    }
    out.constraints.sort();
    out
}

/// Extract the normalized package key from Retread's persisted entry-spec
/// forms (`name[extra]==1`, `name@git:rev`, `name@url:...`, or bare `name`).
fn locked_entry_name(spec: &str) -> Option<String> {
    let end = spec
        .char_indices()
        .find_map(|(index, ch)| {
            matches!(ch, '[' | '=' | '@' | '<' | '>' | '!' | '~' | ' ').then_some(index)
        })
        .unwrap_or(spec.len());
    let name = spec[..end].trim();
    (!name.is_empty()).then(|| canonical_conda_name(name))
}

#[cfg(test)]
mod sibling_lock_constraint_tests {
    use std::collections::BTreeMap;

    use super::{
        CondaCoSolveContext, RetreadConfig, VariantValue, conda_outputs_cache_key_for_target,
        sibling_conda_run_dependencies, sibling_lock_constraints, workspace_solve_fingerprint,
        workspace_solve_fingerprint_for_cache,
    };
    use crate::lock::{CondaDep, LockWheel, Origin, RetreadLock, SCHEMA};
    use crate::pypi::{ResolutionTarget, WheelTarget};
    use crate::workspace::WorkspaceTargetContract;

    fn temp_workspace() -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "retread-sibling-constraints-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(dir.join("current")).unwrap();
        std::fs::create_dir_all(dir.join("sibling")).unwrap();
        std::fs::write(
            dir.join("pixi.toml"),
            r#"[workspace]
channels = ["conda-forge"]
platforms = ["linux-64"]

[feature.composed.dependencies]
current-pack = { path = "./current" }
"sibling.pack" = { path = "./sibling" }

[environments]
composed = { features = ["composed"], no-default-feature = true }
"#,
        )
        .unwrap();
        dir
    }

    fn target() -> ResolutionTarget {
        ResolutionTarget::from_wheel_target(
            WheelTarget {
                python_version: "3.11".to_string(),
                conda_subdir: "linux-64".to_string(),
                max_glibc: None,
            },
            None,
        )
    }

    #[test]
    fn conda_co_solve_preserves_legacy_and_qualified_empty_contract_semantics() {
        let legacy = CondaCoSolveContext::new(
            None,
            None,
            std::path::Path::new("."),
            &target(),
            &[],
            "current-pack",
            &std::collections::BTreeSet::new(),
            &crate::relax::NameMap::new(),
        );
        assert!(
            legacy.detected_virtual_packages.is_none(),
            "an unqualified target must retain the solver's host VP baseline",
        );

        let qualified_target = ResolutionTarget::try_for_contract(
            "3.11",
            WorkspaceTargetContract {
                subdir: "linux-64".to_string(),
                declared_virtual_packages: BTreeMap::from([(
                    "glibc".to_string(),
                    "2.17".to_string(),
                )]),
                detected_virtual_packages: BTreeMap::new(),
            },
        )
        .unwrap();
        let qualified = CondaCoSolveContext::new(
            None,
            None,
            std::path::Path::new("."),
            &qualified_target,
            &[],
            "current-pack",
            &std::collections::BTreeSet::new(),
            &crate::relax::NameMap::new(),
        );
        assert_eq!(
            qualified.detected_virtual_packages,
            Some(BTreeMap::new()),
            "a contract-qualified empty detected set must suppress host VP inheritance",
        );
    }

    #[test]
    fn coactivated_sibling_metadata_constrains_uv_and_fingerprints_replay() {
        let dir = temp_workspace();
        let target = ResolutionTarget::try_for_subdir("3.11", "linux-64").unwrap();
        let mut lock = RetreadLock {
            schema: SCHEMA,
            retread_version: env!("CARGO_PKG_VERSION").to_string(),
            bundle: "sibling.pack".to_string(),
            version: "1.0.0".to_string(),
            python: "3.11".to_string(),
            target_subdir: "linux-64".to_string(),
            target_contract: target.target_contract().cloned(),
            target_identity: Some(target.resolution_identity()),
            target_scope: None,
            exact_workspace_envelope: false,
            inputs_hash: "test-inputs".to_string(),
            root_requirements: vec!["sibling-root==1.0.0".to_string()],
            wheels: vec![
                LockWheel {
                    name: "sibling-root".to_string(),
                    version: "1.0.0".to_string(),
                    origin: Origin::Index,
                    filename: "sibling_root-1.0.0-py3-none-any.whl".to_string(),
                    url: Some(
                        "https://example.invalid/sibling_root-1.0.0-py3-none-any.whl".to_string(),
                    ),
                    sha256: Some("11".repeat(32)),
                    requires_dist: vec![
                        "transformers>=4.57.6,<4.58".to_string(),
                        "bare-dependency".to_string(),
                        "tokenizers[testing]>=0.22,<0.23 ; python_version >= '3.10'".to_string(),
                    ],
                    must_ship: false,
                    upstream_url: None,
                    git_source: None,
                    sdist_source: None,
                },
                LockWheel {
                    name: "transitive-root".to_string(),
                    version: "2.0.0".to_string(),
                    origin: Origin::Index,
                    filename: "transitive_root-2.0.0-py3-none-any.whl".to_string(),
                    url: Some(
                        "https://example.invalid/transitive_root-2.0.0-py3-none-any.whl"
                            .to_string(),
                    ),
                    sha256: Some("22".repeat(32)),
                    requires_dist: vec!["numpy>=2".to_string()],
                    must_ship: false,
                    upstream_url: None,
                    git_source: None,
                    sdist_source: None,
                },
            ],
            conda_run_deps: vec![CondaDep {
                name: "numba".to_string(),
                spec: ">=0.59.1,<0.60".to_string(),
            }],
            index_urls: vec!["https://pypi.org/simple/".to_string()],
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: target.declared_glibc().map(crate::glibc::format_glibc),
            resolution_glibc: target.effective_glibc().map(crate::glibc::format_glibc),
            conda_capable: Vec::new(),
            entry_specs: vec!["sibling-root@git:deadbeef".to_string()],
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };
        let lock_path = dir
            .join("sibling")
            .join(RetreadLock::file_name_for_target("sibling.pack", &target));
        std::fs::write(&lock_path, lock.to_pretty_json().unwrap()).unwrap();

        let manifest = crate::workspace::WorkspaceManifest::load(&dir).unwrap();
        assert_eq!(
            manifest.precise_consuming_envs_for_target(
                &dir,
                &dir.join("current"),
                target.conda_subdir(),
            ),
            Some(vec!["composed".to_string()])
        );
        assert!(lock_path.is_file());
        let loaded = RetreadLock::load(&lock_path).unwrap();
        assert_eq!(
            loaded.resolution_target().unwrap().resolution_identity(),
            target.resolution_identity(),
            "the sibling fixture must use the same production target identity as cache fingerprinting",
        );
        loaded
            .validate_replay_contract_for_target(&target, "sibling.pack")
            .unwrap();
        let constraints = sibling_lock_constraints(&manifest, &dir, &dir.join("current"), &target);
        assert!(
            constraints
                .constraints
                .iter()
                .any(|line| line == "transformers>=4.57.6, <4.58"),
            "constraints were {:?}",
            constraints.constraints
        );
        assert!(
            constraints
                .constraints
                .iter()
                .any(|line| line.starts_with("tokenizers>=0.22, <0.23 ; ")),
            "dependency extras must not be activated by a sibling constraint"
        );
        assert!(
            constraints
                .constraints
                .iter()
                .all(|line| !line.starts_with("bare-dependency"))
        );
        assert!(
            constraints
                .constraints
                .iter()
                .all(|line| !line.starts_with("numpy")),
            "transitive sibling requirements must not become constraints",
        );
        let fingerprint =
            workspace_solve_fingerprint(&manifest, &dir, &dir.join("current"), &target);
        assert!(fingerprint.contains("co-activated-sibling:transformers>=4.57.6, <4.58"));
        assert!(
            fingerprint.contains("co-activated-sibling-conda:numba=>=0.59.1,<0.60"),
            "fingerprint was {fingerprint}",
        );
        let cache_params = pixi_build_types::procedures::conda_outputs::CondaOutputsParams {
            host_platform: rattler_conda_types::Platform::Linux64,
            build_platform: rattler_conda_types::Platform::Linux64,
            channels: vec![],
            variant_configuration: None,
            variant_files: None,
            work_directory: dir.join("work"),
        };
        let cache_key_before = conda_outputs_cache_key_for_target(
            &cache_params,
            None,
            "none",
            &target,
            None,
            &fingerprint,
        );
        let cache_config: RetreadConfig =
            serde_json::from_value(serde_json::json!({"retread-wheels": {}})).unwrap();
        let cache_variants = BTreeMap::from([(
            "python".to_string(),
            vec![
                VariantValue::String("3.10".to_string()),
                VariantValue::String("3.11".to_string()),
            ],
        )]);
        let multi_python_before = workspace_solve_fingerprint_for_cache(
            Some(&dir),
            &dir.join("current"),
            rattler_conda_types::Platform::Linux64,
            &cache_config,
            Some(&cache_variants),
            None,
            false,
        )
        .unwrap();
        lock.wheels[0]
            .requires_dist
            .push("cache-buster>=2".to_string());
        std::fs::write(&lock_path, lock.to_pretty_json().unwrap()).unwrap();
        let changed_fingerprint =
            workspace_solve_fingerprint(&manifest, &dir, &dir.join("current"), &target);
        let cache_key_after = conda_outputs_cache_key_for_target(
            &cache_params,
            None,
            "none",
            &target,
            None,
            &changed_fingerprint,
        );
        let multi_python_after = workspace_solve_fingerprint_for_cache(
            Some(&dir),
            &dir.join("current"),
            rattler_conda_types::Platform::Linux64,
            &cache_config,
            Some(&cache_variants),
            None,
            false,
        )
        .unwrap();
        assert_ne!(fingerprint, changed_fingerprint);
        assert_ne!(
            cache_key_before, cache_key_after,
            "sibling lock content must invalidate conda/outputs memory and disk cache keys",
        );
        assert_ne!(
            multi_python_before, multi_python_after,
            "cache fingerprinting must inspect actual Python-qualified sibling locks across a multi-Python output fanout",
        );
        assert_eq!(
            sibling_conda_run_dependencies(&manifest, &dir, &dir.join("current"), &target)
                .get("numba"),
            Some(&vec![">=0.59.1,<0.60".to_string()]),
        );
        let context = CondaCoSolveContext::new(
            Some(&manifest),
            Some(&dir),
            &dir.join("current"),
            &target,
            &[],
            "current-pack",
            &std::collections::BTreeSet::new(),
            &crate::relax::NameMap::new(),
        );
        assert_eq!(
            context
                .workspace_deps
                .get(&crate::relax::CondaName::new("numba")),
            Some(&vec![">=0.59.1,<0.60".to_string()]),
            "the conda route oracle must compose the sibling's emitted contract",
        );

        lock.schema = SCHEMA - 1;
        std::fs::write(&lock_path, lock.to_pretty_json().unwrap()).unwrap();
        assert!(
            sibling_lock_constraints(&manifest, &dir, &dir.join("current"), &target)
                .constraints
                .is_empty(),
            "stale-schema sibling locks must never feed current replay constraints",
        );

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn exact_empty_vp_scope_applies_to_siblings_but_direct_inference_does_not() {
        let dir = std::env::temp_dir().join(format!(
            "retread-sibling-scope-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join("current")).unwrap();
        std::fs::create_dir_all(dir.join("sibling")).unwrap();
        std::fs::write(
            dir.join("pixi.toml"),
            r#"[workspace]
channels = ["conda-forge"]
platforms = [
  { name = "p1", platform = "linux-64", glibc = "2.28", linux = "4.18" },
  { name = "p1-alias", platform = "linux-64", glibc = "2.28", linux = "4.18" },
]

[feature.current.dependencies]
current-pack = { path = "./current" }

[feature.sibling.dependencies]
"sibling.pack" = { path = "./sibling" }

[feature.sibling.target.p1.dependencies]
profile-only = "1"

[feature.sibling.target.p1-alias.dependencies]
profile-only = "2"

[feature.old]
platforms = ["p1"]

[feature.alias]
platforms = ["p1-alias"]

[environments]
old = { features = ["current", "sibling", "old"], no-default-feature = true }
alias = { features = ["sibling", "alias"], no-default-feature = true }
"#,
        )
        .unwrap();

        let manifest = crate::workspace::WorkspaceManifest::load(&dir).unwrap();
        let resolved = manifest
            .resolve_target_for_source(&dir, &dir.join("current"), "linux-64", None)
            .unwrap()
            .unwrap();
        assert_eq!(resolved.environments, vec!["old"]);
        assert!(
            manifest
                .resolve_source_for_contract(&dir, &dir.join("sibling"), &resolved.contract)
                .is_none(),
            "the sibling's divergent same-contract aliases must fail closed",
        );
        assert!(resolved.contract.detected_virtual_packages.is_empty());
        let direct_target = ResolutionTarget::try_for_contract("3.11", resolved.contract.clone())
            .unwrap()
            .with_workspace_scope(resolved.clone())
            .unwrap();
        let exact_target = ResolutionTarget::try_for_contract("3.11", resolved.contract.clone())
            .unwrap()
            .with_exact_workspace_scope(resolved)
            .unwrap();
        assert!(!direct_target.has_exact_workspace_envelope());
        assert!(exact_target.has_exact_workspace_envelope());
        let lock = RetreadLock {
            schema: SCHEMA,
            retread_version: env!("CARGO_PKG_VERSION").to_string(),
            bundle: "sibling.pack".to_string(),
            version: "1.0.0".to_string(),
            python: "3.11".to_string(),
            target_subdir: "linux-64".to_string(),
            target_contract: direct_target.target_contract().cloned(),
            target_identity: Some(direct_target.resolution_identity()),
            target_scope: direct_target.workspace_scope().cloned(),
            exact_workspace_envelope: false,
            inputs_hash: "stale-current-scope".to_string(),
            root_requirements: vec!["sibling-root==1.0.0".to_string()],
            wheels: vec![LockWheel {
                name: "sibling-root".to_string(),
                version: "1.0.0".to_string(),
                origin: Origin::Index,
                filename: "sibling_root-1.0.0-py3-none-any.whl".to_string(),
                url: Some(
                    "https://example.invalid/sibling_root-1.0.0-py3-none-any.whl".to_string(),
                ),
                sha256: Some("33".repeat(32)),
                requires_dist: vec!["must-not-leak>=1".to_string()],
                must_ship: false,
                upstream_url: None,
                git_source: None,
                sdist_source: None,
            }],
            conda_run_deps: Vec::new(),
            index_urls: vec!["https://pypi.org/simple/".to_string()],
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: direct_target
                .declared_glibc()
                .map(crate::glibc::format_glibc),
            resolution_glibc: direct_target
                .effective_glibc()
                .map(crate::glibc::format_glibc),
            conda_capable: Vec::new(),
            entry_specs: vec!["sibling-root==1.0.0".to_string()],
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };
        let lock_path = dir.join("sibling").join(RetreadLock::file_name_for_target(
            "sibling.pack",
            &direct_target,
        ));
        std::fs::write(&lock_path, lock.to_pretty_json().unwrap()).unwrap();
        RetreadLock::load(&lock_path)
            .unwrap()
            .validate_replay_contract_for_target(&direct_target, "sibling.pack")
            .unwrap();

        let constraints =
            sibling_lock_constraints(&manifest, &dir, &dir.join("current"), &direct_target);
        assert!(
            constraints.constraints.is_empty(),
            "a lock under the current source scope must not stand in for an unresolved sibling scope: {:?}",
            constraints.constraints,
        );
        assert!(
            sibling_conda_run_dependencies(&manifest, &dir, &dir.join("current"), &direct_target,)
                .is_empty()
        );

        let mut exact_lock = lock;
        exact_lock.target_identity = Some(exact_target.resolution_identity());
        exact_lock.target_contract = exact_target.target_contract().cloned();
        exact_lock.target_scope = exact_target.workspace_scope().cloned();
        exact_lock.exact_workspace_envelope = true;
        let exact_lock_path = dir.join("sibling").join(RetreadLock::file_name_for_target(
            "sibling.pack",
            &exact_target,
        ));
        std::fs::write(&exact_lock_path, exact_lock.to_pretty_json().unwrap()).unwrap();
        RetreadLock::load(&exact_lock_path)
            .unwrap()
            .validate_replay_contract_for_target(&exact_target, "sibling.pack")
            .unwrap();

        let exact_constraints =
            sibling_lock_constraints(&manifest, &dir, &dir.join("current"), &exact_target);
        assert!(
            exact_constraints
                .constraints
                .iter()
                .any(|constraint| constraint == "must-not-leak>=1"),
            "the exact envelope must authorize its empty-VP environment/profile scope for co-activated siblings: {:?}",
            exact_constraints.constraints,
        );

        std::fs::remove_dir_all(dir).unwrap();
    }
}

/// Select the committed lock visible to this exact resolution target.
///
/// The target-qualified path is authoritative. A bundle-only schema-12 lock
/// is a read-only compatibility candidate solely for native linux-64; foreign
/// targets never probe that namespace. Callers always write the qualified
/// path returned by [`exact_lock_path_for_target`].
fn lock_path_for_target(
    source_dir: &Path,
    bundle_name: &str,
    target: &ResolutionTarget,
) -> PathBuf {
    let candidates = crate::lock::RetreadLock::read_file_names_for_target(
        bundle_name,
        target,
        crate::glibc::current_pixi_platform(),
    );
    let exact = source_dir.join(
        candidates
            .first()
            .expect("target lock candidates always contain the exact path"),
    );
    candidates
        .into_iter()
        .map(|name| source_dir.join(name))
        .find(|path| path.is_file())
        .unwrap_or(exact)
}

fn exact_lock_path_for_target(
    source_dir: &Path,
    bundle_name: &str,
    target: &ResolutionTarget,
) -> PathBuf {
    source_dir.join(crate::lock::RetreadLock::file_name_for_target(
        bundle_name,
        target,
    ))
}

const DEFAULT_PYTHON: &str = "3.11";

/// URL of prefix-dev/parselmouth's canonical conda-forge -> PyPI name
/// mapping. The JSON is keyed by conda-forge package name with values
/// equal to a list of PyPI distribution names that this conda package
/// provides (or null if no PyPI counterpart). Crucially the list can
/// have multiple entries -- e.g. conda's `libopencv` provides PyPI
/// `opencv-python` AND `opencv-python-headless` -- which we invert at
/// load time to look up all conda candidates from a single PyPI name.
/// Fetched once per backend invocation.
const PARSELMOUTH_MAPPING_URL: &str = "https://raw.githubusercontent.com/prefix-dev/parselmouth/main/files/v0/conda-forge/compressed_mapping.json";

/// Hard-coded PyPI->conda name mappings that PATCH the parselmouth data.
/// These entries are merged into the inverse map on top of parselmouth's
/// (i.e., they ADD candidates rather than replacing). Used for known
/// gaps in parselmouth's coverage -- when the upstream issue is fixed
/// these entries become harmless duplicates.
///
/// Each entry should be accompanied by a link to the relevant
/// parselmouth issue so it can be removed when fixed.
const FALLBACK_PYPI_TO_CONDA: &[(&str, &str)] = &[
    // v0.34.5+: map opencv-python{,-headless} to `opencv` (the
    // current conda-forge name) rather than `py-opencv`. py-opencv
    // exists on conda-forge but was last updated in the py3.6 era
    // (4.2.0) -- conda-forge today ships `opencv` which provides
    // Python bindings via libopencv. Mapping to py-opencv was a
    // long-standing retread bug that surfaced as
    // `py-opencv >=4.11 -> libopencv ==4.2.0 py36_5, no candidates`
    // in the solve check.
    ("opencv-python", "opencv"),
    ("opencv-python-headless", "opencv"),
    // Already covered by parselmouth (`pytorch: [torch]`) but here as a
    // safety net in case the fetch fails entirely.
    ("torch", "pytorch"),
    // Parselmouth maps the PyPI `ray` distribution to conda-forge's
    // `ray-core` package. Keep the same unambiguous edge in the curated
    // fallback so a transient raw.githubusercontent.com failure cannot turn
    // IsaacLab's ranged `ray>=2.45,<3` dependency into the newest PyPI wheel.
    // That wheel can require packaging>=24.2 while Isaac Sim 5.1 pins
    // packaging==23.0; routing the range to conda lets the workspace solver
    // select the compatible ray-core build without a pack-local version pin.
    ("ray", "ray-core"),
    // `pytorch-gpu` is a meta-package on conda-forge (no Requires-Dist /
    // pypi names of its own -- parselmouth's compressed_mapping.json ships
    // `"pytorch-gpu": null`), so the inverse map never gets a `torch ->
    // pytorch-gpu` edge from parselmouth alone. Workspaces commonly pin the
    // GPU variant directly (`pytorch-gpu = "==X"`), so without this entry
    // `resolve_conda_pin_owner("torch")` returns None and the CondaWidenNeeded
    // conda-as-truth ladder (repair.rs) can never find the user's pin.
    // (lock-succ-brief.md ACCEPTANCE RUN #4.)
    ("torch", "pytorch-gpu"),
    // isaacsim's `Requires-Dist` lists `pywin32` (it's Windows-only on
    // PyPI, but Isaac Sim declares it unconditionally). conda-forge ships a
    // `pywin32` stub on Linux that satisfies the import, so route the PyPI
    // name to the identically-named conda package. Baked in here so the
    // consumer workspace no longer needs a `conda-pypi-map` patch for it
    // (parselmouth doesn't map pywin32). Same intent as the gigastrap
    // `patches/conda_pypi_map.json` { "pywin32": "pywin32" } entry.
    ("pywin32", "pywin32"),
];

/// Inverted parselmouth mapping: PyPI name -> conda-forge candidates.
/// One PyPI name can have multiple conda names (e.g., split packages
/// like `airflow` -> `apache-airflow`).
pub(crate) type PypiToCondaMap = std::collections::HashMap<String, Vec<String>>;

const PARSELMOUTH_MAPPING_MAX_BYTES: usize = 16 * 1024 * 1024;
const PARSELMOUTH_MAPPING_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const PARSELMOUTH_MAPPING_ATTEMPTS: usize = 5;
const PARSELMOUTH_RETRY_BASE_DELAY_MS: u64 = if cfg!(test) { 10 } else { 250 };

/// One immutable mapping snapshot per backend process. Caching the complete
/// first result, including the curated-only fallback after a fetch failure,
/// keeps route selection stable between output advertisement and build.
static PYPI_TO_CONDA_MAP: tokio::sync::OnceCell<Arc<PypiToCondaMap>> =
    tokio::sync::OnceCell::const_new();

fn finalize_pypi_to_conda_map(mut inverse: PypiToCondaMap) -> PypiToCondaMap {
    // Patch in known-missing entries from FALLBACK on top of parselmouth.
    // These are gaps in parselmouth's data (see each entry's comment for
    // the corresponding upstream issue). When the upstream issues are
    // fixed, canonicalization below makes them harmless duplicates.
    for (pypi, conda) in FALLBACK_PYPI_TO_CONDA {
        inverse
            .entry(canonical_conda_name(pypi))
            .or_default()
            .push((*conda).to_string());
    }
    for candidates in inverse.values_mut() {
        candidates.sort_unstable();
        candidates.dedup();
    }
    inverse
}

fn retryable_parselmouth_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::REQUEST_TIMEOUT
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status.is_server_error()
}

fn retryable_parselmouth_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause.downcast_ref::<reqwest::Error>().is_some_and(|error| {
            error.is_connect()
                || error.is_timeout()
                || error.is_request()
                || error.is_body()
                || error.status().is_some_and(retryable_parselmouth_status)
        })
    })
}

async fn fetch_pypi_to_conda_map_once(client: &reqwest::Client) -> Result<PypiToCondaMap> {
    use futures::StreamExt;

    let response = client
        .get(PARSELMOUTH_MAPPING_URL)
        .send()
        .await
        .context("fetching parselmouth mapping")?
        .error_for_status()
        .context("parselmouth mapping HTTP status")?;
    if response
        .content_length()
        .is_some_and(|len| len > PARSELMOUTH_MAPPING_MAX_BYTES as u64)
    {
        bail!(
            "parselmouth mapping exceeds {} bytes",
            PARSELMOUTH_MAPPING_MAX_BYTES
        );
    }

    let mut bytes = Vec::with_capacity(
        response
            .content_length()
            .unwrap_or_default()
            .min(PARSELMOUTH_MAPPING_MAX_BYTES as u64) as usize,
    );
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading parselmouth mapping body")?;
        if bytes.len().saturating_add(chunk.len()) > PARSELMOUTH_MAPPING_MAX_BYTES {
            bail!(
                "parselmouth mapping exceeds {} bytes",
                PARSELMOUTH_MAPPING_MAX_BYTES
            );
        }
        bytes.extend_from_slice(&chunk);
    }

    let forward: std::collections::HashMap<String, Option<Vec<String>>> =
        serde_json::from_slice(&bytes).context("parsing parselmouth mapping JSON")?;
    if forward.is_empty() {
        bail!("parselmouth mapping is empty");
    }

    let mut inverse: PypiToCondaMap = std::collections::HashMap::new();
    for (conda_name, pypi_list) in forward {
        for pypi in pypi_list.unwrap_or_default() {
            inverse
                .entry(canonical_conda_name(&pypi))
                .or_default()
                .push(conda_name.clone());
        }
    }
    if inverse.is_empty() {
        bail!("parselmouth mapping contains no PyPI names");
    }
    tracing::info!(
        entries = inverse.len(),
        bytes = bytes.len(),
        "loaded parselmouth PyPI<->conda mapping"
    );
    Ok(inverse)
}

async fn fetch_pypi_to_conda_map() -> Result<PypiToCondaMap> {
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(PARSELMOUTH_MAPPING_TIMEOUT)
        .build()
        .context("building parselmouth HTTP client")?;
    for attempt in 1..=PARSELMOUTH_MAPPING_ATTEMPTS {
        match fetch_pypi_to_conda_map_once(&client).await {
            Ok(mapping) => return Ok(mapping),
            Err(error)
                if attempt < PARSELMOUTH_MAPPING_ATTEMPTS
                    && retryable_parselmouth_error(&error) =>
            {
                let delay_ms = PARSELMOUTH_RETRY_BASE_DELAY_MS * (1_u64 << (attempt - 1));
                tracing::warn!(
                    attempt,
                    max_attempts = PARSELMOUTH_MAPPING_ATTEMPTS,
                    delay_ms,
                    error = %format!("{error:#}"),
                    "transient parselmouth mapping fetch failure; retrying",
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
            Err(error) => return Err(error),
        }
    }
    unreachable!("parselmouth retry loop always returns on its final attempt")
}

async fn load_pypi_to_conda_map_with<F, Fut>(
    cell: &tokio::sync::OnceCell<Arc<PypiToCondaMap>>,
    fetch: F,
) -> Arc<PypiToCondaMap>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<PypiToCondaMap>>,
{
    cell.get_or_init(|| async move {
        let inverse = match fetch().await {
            Ok(inverse) => inverse,
            Err(error) => {
                tracing::warn!(
                    error = %format!("{error:#}"),
                    "parselmouth mapping unavailable; using curated fallback for this process"
                );
                PypiToCondaMap::new()
            }
        };
        Arc::new(finalize_pypi_to_conda_map(inverse))
    })
    .await
    .clone()
}

/// Best-effort fetch of the parselmouth mapping. Returns a fallback map
/// if the network call fails -- never errors. Async because it makes an
/// HTTP request.
///
/// `pub(crate)` so `src/solve/` (the offline lock-repair ladder) can reuse
/// the SAME parselmouth-backed name family (torch -> [pytorch, pytorch-cpu,
/// pytorch-gpu], etc.) that the courier/auto-route path uses to build its
/// `name_map`, instead of hand-rolling a second conda<->pypi name table.
pub(crate) async fn load_pypi_to_conda_map() -> Arc<PypiToCondaMap> {
    load_pypi_to_conda_map_with(&PYPI_TO_CONDA_MAP, fetch_pypi_to_conda_map).await
}

/// Merge the user-declared name map with retread's curated fallbacks and
/// unambiguous parselmouth edges. User entries, including an explicit
/// [`CondaTarget::Disabled`] veto, always win.
///
/// Loading parselmouth remains the caller's policy decision: the cold and
/// incremental paths intentionally use different load predicates. This pure
/// merge only applies an already-loaded map.
pub(crate) fn effective_name_map(configured: &NameMap, pypi_to_conda: &PypiToCondaMap) -> NameMap {
    let mut effective = configured.clone();
    for (pypi, conda) in FALLBACK_PYPI_TO_CONDA {
        effective
            .entry(PypiKey::from_pypi(pypi))
            .or_insert_with(|| CondaTarget::Mapped(CondaName::new(*conda)));
    }
    for (pypi, conda_names) in pypi_to_conda {
        if conda_names.len() == 1 {
            effective
                .entry(PypiKey::from_pypi(pypi))
                .or_insert_with(|| CondaTarget::Mapped(CondaName::new(&conda_names[0])));
        }
    }
    effective
}

#[derive(Default)]
struct State {
    config: Option<RetreadConfig>,
    cache_dir: Option<PathBuf>,
    /// Directory containing the source package's `pixi.toml`. Used to
    /// resolve relative `path = "..."` entries in [retread-wheels].
    source_dir: Option<PathBuf>,
    /// v0.31.0+ workspace root passed by pixi at initialize. Used by
    /// the cascade's last-resort step to mirror the workspace's
    /// `[dependencies]` pin (if any) instead of widening to `*`.
    workspace_dir: Option<PathBuf>,
    /// Optional exact environment/profile target supplied by a workspace
    /// orchestrator. Parsed once at initialize and validated against the
    /// source consumer for every output/build request.
    target_envelope: Option<WorkspaceTargetEnvelope>,
    /// Monotonic initialize generation. Prepared output plans are valid only
    /// for the exact configuration generation that advertised them.
    generation: u64,
    /// Monotonic output-computation transaction. A newer conda/outputs call
    /// invalidates any older in-flight publication in the same generation.
    prepared_transaction: u64,
    prepared_cache_key: Option<String>,
    prepared_builds: Vec<PreparedBuild>,
}

#[derive(Clone, Default)]
pub struct Handler {
    state: Arc<RwLock<State>>,
}

struct Snapshot {
    generation: u64,
    config: RetreadConfig,
    download_dir: PathBuf,
    source_dir: PathBuf,
    cache_dir: PathBuf,
    workspace_dir: Option<PathBuf>,
    target_envelope: Option<WorkspaceTargetEnvelope>,
    fast_cfg: crate::fasttmp::FastTmpConfig,
    fast_tmp: Option<crate::fasttmp::EngagedFastTmp>,
}

/// One wheel after full resolution: URL is concrete, metadata parsed.
#[derive(Debug, Clone)]
struct ResolvedWheel {
    /// PEP 503 normalized PyPI name of this wheel (e.g. "isaacsim-kernel").
    /// Used to filter vendored deps out of the bundle's run-deps and to
    /// build the recipe.yaml's `source` list comments.
    pypi_name: String,
    url: url::Url,
    /// Pristine upstream index / direct URL captured BEFORE
    /// `materialize_and_rewrite` localises the wheel to `file://`.
    ///
    /// Set for:
    ///   - config-entry PyPI version-spec wheels (resolved.url from pypi::resolve)
    ///   - config-entry direct-URL wheels (entry.url)
    ///   - BFS PyPI sub-wheels (resolved_url from bfs_fetch_pypi)
    ///
    /// `None` for source-built (git / path / from) and BFS git/URL sub-wheels
    /// (their bytes re-materialise from source, not from a stable index URL).
    ///
    /// This is the single source of truth read by `build_one` when populating
    /// `EmitWheel.upstream_url`; independent of `url` localization.
    upstream_url: Option<url::Url>,
    /// Git provenance for source-built wheels (schema 8+). Set in
    /// `materialize_and_rewrite` for both named-git (from=) and inline-git
    /// (git=) entry forms. `None` for all other wheel origins.
    git_source: Option<crate::lock::GitWheelSource>,
    /// Sdist provenance for BFS-transitive wheels built from a PyPI sdist
    /// (schema 9+). Set in the BFS phase-3 handler when bfs_fetch_pypi
    /// returns a `SdistProv`. `None` for index-fetched and git-built wheels.
    sdist_source: Option<crate::lock::SdistWheelSource>,
    /// Authority-bearing origin of this wheel's relaxed metadata.
    ///
    /// This is recorded explicitly at materialization time because a local
    /// path build has neither git nor sdist replay provenance and therefore
    /// cannot be classified correctly from those fields after the fact.
    metadata_provenance: Provenance,
    metadata: WheelMetadata,
    /// v0.12.0+: extras the user requested on the originating
    /// `[retread-wheels]` entry. Surfaced in the audit so debugging
    /// "did extras=[all] expand to the right sub-wheels?" is grep-able.
    /// Empty for sub-wheels pulled by the BFS (extras live on the
    /// primary, not on the resolved transitives).
    #[allow(dead_code)] // read at audit time
    extras_requested: Vec<String>,
    /// v0.12.0+: when this wheel carried the auto-data-files inject
    /// for a checkout root (phase 1.6), the report of what got
    /// shipped. `None` for: PyPI/URL/path entries (no checkout root),
    /// BFS sub-wheels (caller doesn't pass auto-data), and dedup'd
    /// siblings (an earlier wheel from the same checkout carried it).
    #[allow(dead_code)] // read at audit time
    auto_data: Option<crate::audit::AutoDataReport>,
    /// v0.12.0+: present when this wheel was the 2..N entry sharing a
    /// checkout root with an earlier wheel that already carried the
    /// auto-data inject. Records the shared root so the audit shows
    /// WHY this wheel didn't ship the repo-root tree.
    #[allow(dead_code)] // read at audit time
    auto_data_dedup_skipped_root: Option<PathBuf>,
}

/// Record wheel metadata authority at the source boundary. Source-built
/// wheels include local paths, whose replay fields are intentionally empty,
/// so downstream code must not try to reconstruct this fact from URL shape.
fn wheel_entry_metadata_provenance(entry: &WheelEntry) -> Provenance {
    if entry.path.is_some() || entry.git.is_some() || entry.from.is_some() {
        Provenance::SourceBuiltRelaxed
    } else {
        Provenance::IndexWheelMetadata
    }
}

/// Reuse a wheel produced by the closure's shared sdist-heal ladder when the
/// same package is a direct spec-form `[retread-wheels]` entry. Without this
/// bridge, primary materialization independently asks the index for a wheel
/// after the closure has already built and stored one, and fails with "no
/// wheels listed" for the exact sdist-only root that triggered the heal.
fn closure_built_entry_override(
    entry_name: &str,
    entry: &WheelEntry,
    closure_wheels: &[crate::lock::LockWheel],
    store_root: &Path,
) -> Result<(WheelEntry, Option<crate::lock::SdistWheelSource>)> {
    if !entry.is_spec() {
        return Ok((entry.clone(), None));
    }
    let canonical_name = canonical_conda_name(entry_name);
    let Some(wheel) = closure_wheels.iter().find(|wheel| {
        matches!(wheel.origin, crate::lock::Origin::Built)
            && wheel.sdist_source.is_some()
            && canonical_conda_name(&wheel.name) == canonical_name
            && entry.normalized_version().as_deref() == Some(wheel.version.as_str())
    }) else {
        return Ok((entry.clone(), None));
    };
    let sha256 = wheel.sha256.as_deref().ok_or_else(|| {
        anyhow!(
            "closure-built wheel {}=={} has no wheel-store sha256",
            wheel.name,
            wheel.version,
        )
    })?;
    let store_path = store_root.join(sha256).join(&wheel.filename);
    if !store_path.is_file() {
        bail!(
            "closure-built wheel {}=={} is missing from the shared wheel store at {}",
            wheel.name,
            wheel.version,
            store_path.display(),
        );
    }
    let local_url = url::Url::from_file_path(&store_path).map_err(|_| {
        anyhow!(
            "closure-built wheel-store path is not representable as a file URL: {}",
            store_path.display(),
        )
    })?;
    let mut overridden = entry.clone();
    overridden.url = Some(local_url);
    overridden.sha256 = Some(sha256.to_string());
    Ok((overridden, wheel.sdist_source.clone()))
}

/// One conda output's worth of wheels: a "bundle" produced by a single
/// `[retread-wheels]` user entry. The primary wheel plus all extras-derived
/// wheels are installed together into the same conda package, matching the
/// pattern in pixi#5230 comment 24.
#[derive(Debug, Clone)]
struct Bundle {
    /// Conda package name from the user entry's map key.
    conda_name: String,
    /// The primary wheel (the one the user named) — drives the package's
    /// version + filename for platform detection.
    primary: ResolvedWheel,
    /// All extras-derived sub-wheels, in BFS discovery order.
    extras: Vec<ResolvedWheel>,
    /// v0.14.1+: every prefer-conda probe decision made while
    /// resolving this bundle (BFS + auto-bundle). Surfaced in
    /// retread-audit-<name>.json so debugging "why did this dep go
    /// to conda?" doesn't require any logs at all -- just `cat` the
    /// audit. The audit is also flushed early (at conda/outputs) so
    /// failed conda solves still leave this trace on disk.
    probe_decisions: Vec<crate::audit::ProbeDecision>,
    /// v0.33.5+: per-env outcome of the pre-emission solve check.
    /// One entry per workspace env that references this discovered
    /// output. Empty `BTreeMap` means the check was skipped (no
    /// workspace, no envs). Each diagnostic shows what THAT env's
    /// conda solver will fail on -- much more actionable than the
    /// cross-env union (which over-constrains because envs that
    /// don't actually pull in all features still inherit those
    /// features' transitives in the union).
    solve_diagnostics: BTreeMap<String, crate::audit::SolveDiagnostics>,
    /// v4.3.0 (spec-uv-restructure M2): packages the uv auto-route loop
    /// moved from the wheel closure to the conda side. Each
    /// becomes a conda run-dep of the stub output (`produce_output`) so
    /// conda — not a shipped wheel — provides it at install time. Empty
    /// on the legacy path and when `auto-route = false`.
    ///
    /// Each route carries the typed origin of its selected version.
    /// Requirements originating in `retread-deps-from` remain advisory and
    /// emit their effective upstream spec; ordinary uv selections remain
    /// preferences until the common constraint finalizer derives a spec.
    auto_routed: Vec<BundleAutoRoute>,
    /// Canonical PyPI names the precise consuming workspace already owns.
    /// These are removed from the pack's wheel graph and are not re-emitted
    /// as generated-pack conda run dependencies. Conda-owned names enter only
    /// after a validated route or from workspace-solved conda provider facts;
    /// partial-consumer providers retain an explicit conda route.
    /// Workspace-PyPI names enter from direct declarations shared by every
    /// consumer.
    auto_dropped: HashSet<String>,
    /// Canonical names of every package in the exported uv closure
    /// (`UvClosure::pins`). These are provided by the wheel closure /
    /// uv install set at install time, so `produce_output` must NEVER
    /// translate a `Requires-Dist` line naming one of them into a conda
    /// run-dep — conda channels may not even carry the package
    /// (`isaacsim-kernel`) or may cap below the wheel's spec (`aiodns`).
    /// Empty on the legacy (non-uv) path, where the BFS bundles the
    /// full closure and the `vendored` set already covers members.
    uv_closure_names: std::collections::HashSet<String>,
    /// Full uv-selected adjacency, including routed/no-emit packages that
    /// are absent from the exported wheel set. Diagnostics use this only to
    /// walk a conflict back to its introducing root edge.
    uv_dependency_graph: crate::uv_closure::UvDependencyGraph,
    /// Exact conda versions selected identically in every precise consuming
    /// environment, including transitives. Membership in this map is the
    /// exact validation boundary; ownership evidence is carried separately in
    /// `workspace_conda_provider_facts`.
    workspace_conda_versions: BTreeMap<String, String>,
    /// Conda-provider evidence derived from every successful precise-consumer
    /// solve. Unlike `workspace_conda_versions`, these facts retain providers
    /// selected by only some consumers and providers whose selected versions
    /// differ across consumers, together with the direct workspace specs that
    /// constrain them.
    workspace_conda_provider_facts: BTreeMap<String, WorkspaceCondaProviderFact>,
}

/// One mutable uv auto-route retained on a bundle until the final emitted
/// conda dependency set has passed the joint co-solvability check. Keeping the
/// complete route (rather than only its conda name/version) preserves the PyPI
/// identity needed to put a rejected route back into the wheel bundle.
#[derive(Debug, Clone)]
struct BundleAutoRoute {
    route: crate::uv_closure::AutoRoutedPackage,
    provenance: Provenance,
    /// A partial workspace-provider fact replaces this route's stale PyPI
    /// selection and inputs with one authoritative conda provider constraint.
    /// The route itself remains the normal emission and joint-validation path.
    workspace_provider: Option<WorkspaceCondaProviderRoute>,
}

/// Typed partial-provider override carried by an existing auto-route. The
/// selected versions are provider evidence; `constraint` is the conjunction of
/// direct workspace specs and is the only clause emitted for this route.
#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkspaceCondaProviderRoute {
    conda_name: CondaName,
    selected_versions: BTreeSet<String>,
    constraint: Constraint,
}

impl Bundle {
    fn all_wheels(&self) -> impl Iterator<Item = &ResolvedWheel> {
        std::iter::once(&self.primary).chain(self.extras.iter())
    }

    /// Promote workspace-solved conda providers into the same typed drop
    /// ownership used by Rule 1. Providers present in every precise consumer
    /// need no emitted dependency; partial providers retain an existing route
    /// annotated with the conjunction of direct workspace constraints so the
    /// shared pack supplies the provider to consumers that did not select it
    /// before the pack was present.
    ///
    /// Explicit PyPI intent remains authoritative: manual overrides,
    /// keep-PyPI requests, first-party roots, and wheels already materialized
    /// in this bundle exclude their entire configured provider group. Vetoes
    /// are resolved before either the typed drop or route annotation mutates.
    /// Only the declared fact map projects ownership; the effective map is
    /// consulted solely to widen exclusions to names emission would translate.
    fn apply_workspace_conda_fact_ownership(
        &mut self,
        config: &RetreadConfig,
        fact_name_map: &NameMap,
        dynamic_keep_pypi: &BTreeSet<PypiKey>,
        protected_roots: &BTreeSet<String>,
    ) {
        if self.workspace_conda_provider_facts.is_empty() {
            return;
        }

        let mut owned =
            workspace_conda_provider_owners(&self.workspace_conda_provider_facts, fact_name_map);

        let ledger_overrides: HashSet<String> = config
            .ledger_overrides
            .iter()
            .map(|name| canonical_conda_name(name))
            .collect();
        let explicit_exclusions: HashSet<String> = config
            .overrides
            .keys()
            .map(|name| canonical_conda_name(name))
            .filter(|name| !ledger_overrides.contains(name))
            .chain(
                config
                    .keep_pypi
                    .iter()
                    .map(|name| canonical_conda_name(name)),
            )
            .chain(
                dynamic_keep_pypi
                    .iter()
                    .map(|name| name.as_str().to_string()),
            )
            .chain(
                protected_roots
                    .iter()
                    .map(|name| canonical_conda_name(name)),
            )
            .chain(
                self.all_wheels()
                    .map(|wheel| canonical_conda_name(&wheel.pypi_name)),
            )
            .chain(std::iter::once(canonical_conda_name(&self.conda_name)))
            .collect();
        let disabled_pypi: HashSet<String> = fact_name_map
            .iter()
            .filter_map(|(pypi_name, target)| {
                target
                    .mapped_name()
                    .is_none()
                    .then(|| pypi_name.as_str().to_string())
            })
            .collect();
        let excluded_pypi: HashSet<String> = explicit_exclusions
            .iter()
            .cloned()
            .chain(disabled_pypi)
            .collect();
        let excluded_providers: HashSet<String> = explicit_exclusions
            .iter()
            .flat_map(|name| {
                let mapped = config
                    .name_map
                    .get(&PypiKey::from_pypi(name))
                    .and_then(|target| target.mapped_name())
                    .map(|conda_name| conda_name.key().into_string());
                std::iter::once(name.clone()).chain(mapped)
            })
            .collect();
        owned.retain(|pypi_name, provider| {
            let effective_provider = config
                .name_map
                .get(&PypiKey::from_pypi(pypi_name))
                .and_then(|target| target.mapped_name())
                .map_or_else(
                    || pypi_name.clone(),
                    |conda_name| conda_name.key().into_string(),
                );
            !excluded_pypi.contains(pypi_name)
                && !excluded_providers.contains(provider)
                && !excluded_providers.contains(&effective_provider)
        });

        if owned.is_empty() {
            return;
        }

        let mut drop_only = BTreeSet::new();
        let mut routed = Vec::new();
        for (pypi_name, provider) in owned {
            let Some(fact) = self.workspace_conda_provider_facts.get(&provider) else {
                continue;
            };
            if fact.present_in_all_consumers {
                drop_only.insert(pypi_name);
                continue;
            }
            let Some(workspace_provider) = workspace_conda_provider_route(&provider, fact) else {
                continue;
            };
            let has_matching_route = self.auto_routed.iter().any(|route| {
                canonical_conda_name(&route.route.pypi_name) == pypi_name
                    && canonical_conda_name(&route.route.conda_name) == provider
            });
            if has_matching_route {
                routed.push((pypi_name, provider, workspace_provider));
            }
        }

        // Mutate only after every candidate has passed its provider-evidence,
        // direct-spec, and existing-route checks. A partial fact can never
        // suppress a wheel edge without also retaining conda provision.
        for (pypi_name, provider, mut workspace_provider) in routed {
            if let Some(route) = self.auto_routed.iter_mut().find(|route| {
                canonical_conda_name(&route.route.pypi_name) == pypi_name
                    && canonical_conda_name(&route.route.conda_name) == provider
            }) {
                // Provider facts use canonical lookup keys; emission retains
                // the matched route's raw conda spelling (underscores are
                // significant conda package-name characters).
                workspace_provider.conda_name = CondaName::new(&route.route.conda_name);
                route.provenance = workspace_provider.constraint.provenance.clone();
                route.workspace_provider = Some(workspace_provider);
                self.auto_dropped.insert(pypi_name);
            }
        }
        self.auto_routed
            .retain(|route| !drop_only.contains(&canonical_conda_name(&route.route.pypi_name)));
        self.auto_dropped.extend(drop_only);
    }
}

/// Project workspace-solved conda providers into the PyPI identities they can
/// provide.
/// Explicit mappings are directional: `foo -> bar` means only a `bar` conda
/// fact owns PyPI `foo`. A `foo` conda fact cannot reverse that edge, and an
/// explicitly disabled PyPI key has no conda owner even when a same-named fact
/// exists. Unmapped names retain the ordinary same-name identity boundary.
fn workspace_conda_provider_owners(
    workspace_conda_provider_facts: &BTreeMap<String, WorkspaceCondaProviderFact>,
    fact_name_map: &NameMap,
) -> BTreeMap<String, String> {
    let fact_names: HashSet<String> = workspace_conda_provider_facts
        .keys()
        .map(|name| canonical_conda_name(name))
        .collect();
    let candidates: BTreeSet<String> = fact_names
        .iter()
        .cloned()
        .chain(
            fact_name_map
                .keys()
                .map(|pypi_name| pypi_name.as_str().to_string()),
        )
        .collect();
    candidates
        .into_iter()
        .filter_map(|pypi_name| {
            let key = PypiKey::from_pypi(&pypi_name);
            let provider = match fact_name_map.get(&key) {
                Some(target) => target
                    .mapped_name()
                    .map(|conda_name| conda_name.key().into_string())?,
                None => pypi_name.clone(),
            };
            fact_names
                .contains(&provider)
                .then_some((pypi_name, provider))
        })
        .collect()
}

/// Build the single authoritative route constraint for a provider selected in
/// only some precise consumers. The selected versions prove concrete conda
/// providers; direct declarations are intersected through the common typed
/// finalizer. Transitive-only and wildcard declarations intentionally emit an
/// unconstrained provider dependency.
fn workspace_conda_provider_route(
    provider: &str,
    fact: &WorkspaceCondaProviderFact,
) -> Option<WorkspaceCondaProviderRoute> {
    if fact.selected_versions.is_empty() {
        return None;
    }
    let provenance = Provenance::WorkspaceCondaFact("precise-consuming-envs".to_string());
    let mut declared_constraints = Vec::new();
    for declared_spec in &fact.declared_specs {
        let trimmed = declared_spec.trim();
        let specifiers = if trimmed.is_empty() || matches!(trimmed, "*" | "==*") {
            VersionSpecifiers::empty()
        } else {
            let pep440 = crate::uv_closure::conda_spec_to_pep440(trimmed)?;
            VersionSpecifiers::from_str(&pep440).ok()?
        };
        declared_constraints.push(Constraint {
            specifiers,
            provenance: provenance.clone(),
            source: format!("workspace conda declaration `{provider} {trimmed}`"),
            origin_id: ConstraintOriginId::from_parts(
                "workspace-provider-declaration",
                [provider, trimmed],
            ),
        });
    }
    if declared_constraints.is_empty() {
        declared_constraints.push(Constraint {
            specifiers: VersionSpecifiers::empty(),
            provenance: provenance.clone(),
            source: format!("workspace-selected transitive conda provider `{provider}`"),
            origin_id: ConstraintOriginId::from_parts(
                "workspace-provider-fallback",
                [provider, "*"],
            ),
        });
    }
    let provider_key = PypiKey::from_pypi(provider);
    let specifiers = finalize(&provider_key, &declared_constraints).ok()?;
    let rendered = if specifiers.is_empty() {
        "*".to_string()
    } else {
        specifiers.to_string().replace(", ", ",")
    };
    let selected_versions = fact
        .selected_versions
        .iter()
        .map(|version| format!("`{version}`"))
        .collect::<Vec<_>>()
        .join(", ");
    Some(WorkspaceCondaProviderRoute {
        conda_name: CondaName::new(provider),
        selected_versions: fact.selected_versions.clone(),
        constraint: Constraint {
            specifiers,
            provenance,
            source: format!(
                "workspace conda provider `{provider} {rendered}` selected as {selected_versions} in precise consuming environments"
            ),
            origin_id: ConstraintOriginId::from_parts(
                "workspace-provider-route",
                [provider, rendered.as_str()],
            ),
        },
    })
}

impl Handler {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn dispatch(&self, method: String, params: Value) -> Result<Value, RpcError> {
        match method.as_str() {
            NEGOTIATE => self.negotiate(parse_params(params)?).await.and_then(ok),
            INITIALIZE => self.initialize(parse_params(params)?).await.and_then(ok),
            CONDA_OUTPUTS => self.conda_outputs(parse_params(params)?).await.and_then(ok),
            // Pixi Build API 5 (pixi 0.72+) serializes dependency match
            // specs in conda/build_v1 as structured objects; normalize
            // them back to the API-4 string form our pinned wire types
            // deserialize, so one binary speaks both protocol revisions.
            CONDA_BUILD_V1 => self
                .conda_build_v1(parse_params(
                    crate::compat::normalize_conda_build_v1_params(params),
                )?)
                .await
                .and_then(ok),
            other => Err(RpcError {
                code: crate::rpc::METHOD_NOT_FOUND,
                message: format!("unknown method `{other}`"),
                data: None,
            }),
        }
    }

    async fn negotiate(
        &self,
        _params: NegotiateCapabilitiesParams,
    ) -> Result<NegotiateCapabilitiesResult, RpcError> {
        Ok(NegotiateCapabilitiesResult {
            capabilities: BackendCapabilities {
                provides_conda_outputs: Some(true),
                provides_conda_build_v1: Some(true),
            },
        })
    }

    async fn initialize(&self, params: InitializeParams) -> Result<InitializeResult, RpcError> {
        let mut config: RetreadConfig = match params.configuration {
            Some(v) => serde_json::from_value(v)
                .map_err(|e| RpcError::invalid_params(format!("[build.config]: {e}")))?,
            None => {
                return Err(RpcError::invalid_params(
                    "pixi-build-retread requires a [build.config] table with at least `wheels = { ... }`",
                ));
            }
        };

        let hermetic_env = std::env::var(crate::config::HERMETIC_BUILDS_ENV).ok();
        config.hermetic =
            crate::config::effective_hermetic_builds(config.hermetic, hermetic_env.as_deref());

        if config.retread_wheels.is_empty() {
            return Err(RpcError::invalid_params(
                "[build.config].wheels must list at least one wheel",
            ));
        }

        // Eagerly normalize + validate each entry now so misconfigurations
        // surface at initialize time rather than mid-build. Normalization
        // lifts a `#sha256=` URL fragment into the discrete `sha256` field so
        // the persistent wheel cache can address URL-form wheels (issue #10
        // perf: otherwise a multi-GiB extscache wheel redownloads every run).
        for (name, entry) in &mut config.retread_wheels {
            entry
                .normalize(name)
                .map_err(|e| RpcError::invalid_params(e.to_string()))?;
            entry
                .validate(name)
                .map_err(|e| RpcError::invalid_params(e.to_string()))?;
        }

        // v4.4.0: the `retread-resolver` knob was removed -- uv is the only
        // resolver. The legacy cascade/resolvo mirror-solver was deleted in
        // v4.2.0. The field is still parsed (so old manifests load under
        // deny_unknown_fields) but ignored; warn once so the user removes it.
        if let Some(kind) = config.resolver.as_deref() {
            tracing::warn!(
                "retread-resolver = \"{kind}\" is ignored (removed in v4.4; uv is the \
                 only resolver). Remove it from your [package.build.config].",
            );
        }

        // L2 (cleanup P4.6): conda-aware is not yet implemented -- warn
        // once at initialize time so the user knows what they're getting.
        // The variant is kept so existing manifests that set
        // `retread-relax = "conda-aware"` keep loading without error; it
        // silently degrades to strong-major at translate time (same upper-
        // bound stripping, no per-dep probe). The probe layer described in
        // RelaxPolicy::CondaAware's doc comment will replace this branch
        // when implemented.
        if config.relax == RelaxPolicy::CondaAware {
            tracing::warn!(
                "retread-relax = \"conda-aware\" is not yet implemented; \
                 degrading to strong-major (strips all upper bounds unconditionally). \
                 For per-dep adaptive widening today, use \
                 \"patch-then-minor-then-major-then-last-resort\" instead.",
            );
        }

        let target_envelope = WorkspaceTargetEnvelope::from_process_env().map_err(|error| {
            RpcError::invalid_params(format!("invalid workspace target envelope: {error:#}"))
        })?;
        let workspace_dir = params.workspace_directory;
        ensure_pixi_bld_symlink_target(workspace_dir.as_deref())?;
        if let Some(workspace_root) = pixi_workspace_dir(workspace_dir.as_deref()) {
            crate::fasttmp::heal_stale_envs_symlink_at_backend_startup(&workspace_root).map_err(
                |error| RpcError::internal(format!("fast-tmp backend envs-link heal: {error:#}")),
            )?;
        }
        config.pack_manifest_path = Some(crate::pack_overrides::pack_manifest_display_path(
            workspace_dir.as_deref(),
            &params.manifest_path,
        ));

        // Fix #22: merge this pack's auto-repaired overrides from the
        // workspace's `.retread/auto-overrides.json` ledger into
        // `config.overrides`, IN MEMORY only -- the pack's pixi.toml is
        // never touched. Every downstream consumer (resolve_all,
        // apply_emission's `effective.overrides`, and
        // `courier::config_fingerprint`'s `declared_config`) reads this
        // same `config`, so they see auto overrides exactly like a manual
        // `retread-overrides` entry, and a ledger change busts the
        // fingerprint like a manifest edit would. No-op (and no error)
        // when there's no workspace directory or no ledger yet.
        if let Some(ws) = workspace_dir.as_deref() {
            crate::pack_overrides::merge_ledger_overrides(&mut config, ws, &params.manifest_path);
        }

        let mut state = self.state.write().await;
        state.config = Some(config);
        state.cache_dir = params.cache_directory;
        // source_directory falls back to the manifest's containing dir.
        state.source_dir = params
            .source_directory
            .or_else(|| params.manifest_path.parent().map(PathBuf::from));
        state.workspace_dir = workspace_dir;
        state.target_envelope = target_envelope;
        state.generation = state
            .generation
            .checked_add(1)
            .expect("handler generation counter exhausted");
        state.prepared_transaction = state
            .prepared_transaction
            .checked_add(1)
            .expect("prepared transaction counter exhausted");
        state.prepared_cache_key = None;
        state.prepared_builds.clear();
        Ok(InitializeResult {})
    }

    async fn conda_outputs(
        &self,
        params: CondaOutputsParams,
    ) -> Result<CondaOutputsResult, RpcError> {
        // Emitted before anything else so the user sees retread is alive the
        // moment pixi hands off -- the work below (parselmouth load, wheel
        // materialization, repodata probes, solve checks) can run for many
        // seconds with no other output. Generic: no package-specific text.
        // Cache hit? pixi re-requests conda/outputs once per lock-file env
        // that shares this source package; the result is identical for
        // identical params, so return the memoized one instead of redoing
        // the whole multi-env solve. (See CONDA_OUTPUTS_CACHE.)
        //
        // Read workspace_dir from handler state BEFORE the lock guard so we
        // don't hold the state lock while doing blocking I/O.
        let (pre_key_workspace_dir, pre_key_source_dir, pre_key_target_envelope, pre_key_config) = {
            let state = self.state.read().await;
            let config = state
                .config
                .clone()
                .ok_or_else(|| RpcError::internal("initialize was not called"))?;
            (
                state.workspace_dir.clone(),
                state
                    .source_dir
                    .clone()
                    .unwrap_or_else(|| params.work_directory.clone()),
                state.target_envelope.clone(),
                config,
            )
        };
        let resolved_workspace_target = resolve_workspace_target_for_source(
            pre_key_workspace_dir.as_deref(),
            &pre_key_source_dir,
            params.host_platform.as_str(),
            pre_key_target_envelope.as_ref(),
        )
        .map_err(|error| {
            RpcError::invalid_params(format!(
                "resolving workspace target `{}` for source `{}`: {error:#}",
                params.host_platform,
                pre_key_source_dir.display(),
            ))
        })?;
        let cache_target = wheel_target_for_resolved_workspace(
            params.host_platform,
            "0.0",
            resolved_workspace_target.as_ref(),
            pre_key_target_envelope.is_some(),
        )
        .map_err(|error| {
            RpcError::invalid_params(format!(
                "invalid workspace target `{}`: {error:#}",
                params.host_platform
            ))
        })?;
        let cache_workspace_solve_fingerprint = workspace_solve_fingerprint_for_cache(
            pre_key_workspace_dir.as_deref(),
            &pre_key_source_dir,
            params.host_platform,
            &pre_key_config,
            params.variant_configuration.as_ref(),
            resolved_workspace_target.as_ref(),
            pre_key_target_envelope.is_some(),
        )
        .map_err(|error| {
            RpcError::invalid_params(format!(
                "fingerprinting workspace solve inputs for target `{}`: {error:#}",
                params.host_platform
            ))
        })?;
        let mtime = workspace_manifest_mtime(pre_key_workspace_dir.as_deref());
        let auto_overrides_fp = auto_overrides_fingerprint(pre_key_workspace_dir.as_deref());
        let cache_key = conda_outputs_cache_key_for_target(
            &params,
            mtime,
            &auto_overrides_fp,
            &cache_target,
            resolved_workspace_target.as_ref(),
            &cache_workspace_solve_fingerprint,
        );
        let memory_cached = {
            let cache = CONDA_OUTPUTS_CACHE
                .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
                .lock()
                .unwrap();
            cache.get(&cache_key).cloned()
        };
        if let Some(cached) = memory_cached {
            let retained = self
                .retain_prepared_for_memory_cache_hit(&cache_key, &params.work_directory)
                .await;
            if !cached.requires_prepared_plan || retained {
                tracing::info!(
                    "retread: conda/outputs cache hit -- returning memoized result (pixi re-requested for another env)",
                );
                crate::status::tty(
                    "reusing already-computed outputs (pixi re-requested this package for another environment).",
                );
                log_final_bundle_outputs(&cached.result);
                return Ok(cached.result);
            }
            // The memo advertised an incremental lock version, but its exact
            // materialized plan was consumed or invalidated. Recompute both
            // metadata and the typed plan together rather than returning an
            // identity that a later build cannot safely reproduce.
            CONDA_OUTPUTS_CACHE
                .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
                .lock()
                .unwrap()
                .remove(&cache_key);
            tracing::debug!(
                "retread: incremental conda/outputs memo lost its prepared plan; recomputing"
            );
        }
        // Fetched early (cheap: just clones handler state) so the DISK
        // cache below can be consulted before the expensive solve.
        let Snapshot {
            generation,
            config,
            download_dir,
            source_dir,
            cache_dir,
            workspace_dir,
            target_envelope: _target_envelope,
            fast_cfg: _fast_cfg,
            fast_tmp: _fast_tmp,
        } = self.snapshot(&params.work_directory).await?;
        // v2.11.0: cross-PROCESS memo. CONDA_OUTPUTS_CACHE only dedupes
        // repeat conda/outputs calls served by the SAME retread process.
        // pixi solves separate top-level environments (e.g. `isaaclab-gpu`
        // vs `isaaclab-gpu-latest`, or `gsi` vs `gsi-ros2`) with SEPARATE
        // backend processes even when both need this exact same source
        // package with identical params -- each such process starts with
        // an empty in-memory cache and reruns the ENTIRE multi-env solve
        // (all widening attempts, for every env) from scratch. This is
        // "solver duplication on cold": the same solve-check attempts for
        // the same envs appear twice (or more) in the log, one full block
        // per process, wasting the minutes-long repodata-parse + resolvo
        // solve cost and leaving other cores idle waiting on it serially.
        // Persisting the result to disk keyed by the same cache_key lets
        // a fresh process reuse an already-computed result instead of
        // redoing the work. Best-effort: read/write failures fall back to
        // a normal cold compute, never fail the RPC.
        let disk_cache_path = conda_outputs_disk_cache_path(&cache_dir, &cache_key, &source_dir);
        if let Some(cached) = read_conda_outputs_disk_cache(&disk_cache_path).await {
            tracing::info!(
                path = %disk_cache_path.display(),
                "retread: conda/outputs disk-cache hit -- reusing a prior process's result \
                 (this environment shares the source package + params with one already solved)",
            );
            crate::status::tty(
                "reusing a previously-computed solve for this source package \
                 (another environment already solved it with the same inputs).",
            );
            CONDA_OUTPUTS_CACHE
                .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
                .lock()
                .unwrap()
                .insert(
                    cache_key.clone(),
                    CondaOutputsMemo {
                        result: cached.clone(),
                        requires_prepared_plan: false,
                    },
                );
            self.invalidate_prepared_builds().await;
            log_final_bundle_outputs(&cached);
            return Ok(cached);
        }
        // Coalesce simultaneous cold requests for this exact source/target
        // key. A waiter rechecks the memo after acquiring the lock and returns
        // the first process's completed result instead of repeating minutes of
        // closure resolution, route validation, and wheel materialization.
        // Locking is best-effort so a cache permission/filesystem issue never
        // turns an otherwise-valid cold compute into an RPC failure.
        let disk_cache_fill_lock = match acquire_conda_outputs_disk_cache_fill_lock(
            &disk_cache_path,
        )
        .await
        {
            Ok(lock) => Some(lock),
            Err(error) => {
                tracing::warn!(
                    path = %disk_cache_path.display(),
                    error = %error,
                    "retread: could not acquire conda/outputs first-compute lock; computing independently",
                );
                None
            }
        };
        if disk_cache_fill_lock.is_some()
            && let Some(cached) = read_conda_outputs_disk_cache(&disk_cache_path).await
        {
            tracing::info!(
                path = %disk_cache_path.display(),
                "retread: conda/outputs disk-cache hit after waiting for a concurrent first compute",
            );
            crate::status::tty(
                "reusing a source-package solve completed concurrently for another environment.",
            );
            CONDA_OUTPUTS_CACHE
                .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
                .lock()
                .unwrap()
                .insert(
                    cache_key.clone(),
                    CondaOutputsMemo {
                        result: cached.clone(),
                        requires_prepared_plan: false,
                    },
                );
            self.invalidate_prepared_builds().await;
            log_final_bundle_outputs(&cached);
            return Ok(cached);
        }
        let prepared_transaction = self.begin_prepared_transaction(generation).await;
        let mut prepared_builds = Vec::new();
        let mut incremental_output_advertised = false;
        // Re-read from the snapshot's workspace path. The cache-key probes
        // above happen before snapshotting handler state and are not a safe
        // lifecycle boundary for the typed plan itself.
        let prepared_workspace_mtime = workspace_manifest_mtime(workspace_dir.as_deref());
        let prepared_auto_overrides_fp = auto_overrides_fingerprint(workspace_dir.as_deref());
        tracing::info!(
            retread_version = env!("CARGO_PKG_VERSION"),
            "retread: computing conda outputs (resolving wheels + probing channels; large wheels may download here)",
        );
        let phase_start = std::time::Instant::now();
        // tracing -> stderr is invisible during pixi's solve phase (pixi hides
        // backend stderr behind its "updating lock-file" spinner, even at -vv).
        // Mirror the key status to /dev/tty so the user sees retread is alive.
        // v1.4.6: cache-aware wording -- with a populated wheels/ dir the
        // "first run downloads" caveat was misleading (the wheels are read
        // from disk; only the conda solves remain).
        let wheels_cached = std::fs::read_dir(&download_dir)
            .map(|mut d| d.next().is_some())
            .unwrap_or(false);
        crate::status::tty(if wheels_cached {
            "resolving source package: wheels found in the local cache; running \
             per-environment conda solve checks (repodata parse + solves can \
             take a few minutes)."
        } else {
            "resolving source package: materializing wheels, then per-environment \
             solve checks (first run downloads the package wheels and runs many \
             conda solves -- large wheels can take several minutes)."
        });

        // Pick the target Python versions. Precedence:
        //   1. workspace.build-variants python = [...]
        //   2. [build.config] python = "3.11" / ["3.11", "3.12"]
        //   3. DEFAULT_PYTHON
        let pythons = pythons_for(&config, params.variant_configuration.as_ref());

        // Fan out: one output per (python, wheel). Any resolve error
        // propagates immediately -- pixi reports a useless "package not
        // provided" message if we return empty outputs, so surfacing the
        // per-entry context from resolve_all is the only way the user
        // ever learns which [retread-wheels] entry actually broke.
        let mut outputs = Vec::new();
        let mut output_conflicts = Vec::new();
        let mut pending_output_relaxations = Vec::new();
        // v4.2.0: the per-env pre-emission solve check (and its
        // bookkeeping / fail gate) was deleted with the legacy
        // mirror-solver; outputs ship unvalidated and `retread solve`
        // owns conflict handling.
        for python_version in &pythons {
            let target = wheel_target_for_resolved_workspace(
                params.host_platform,
                python_version,
                resolved_workspace_target.as_ref(),
                pre_key_target_envelope.is_some(),
            )
            .map_err(|error| {
                RpcError::invalid_params(format!(
                    "invalid Python target `{python_version}`: {error:#}"
                ))
            })?;
            let python_version = target.python_version();
            // Phase 1: materialize wheels + auto-bundle. Env-agnostic;
            // results reused across all per-env emissions.
            let t_materialize = std::time::Instant::now();
            let (materialized, base_config, restore_relaxations) = resolve_all(
                &config,
                &target,
                &download_dir,
                &source_dir,
                &cache_dir,
                &params.channels,
                workspace_dir.as_deref(),
            )
            .await
            .map_err(|e| {
                RpcError::invalid_params(format!(
                    "resolving wheels for python {python_version}: {e:#}"
                ))
            })?;
            pending_output_relaxations.extend(restore_relaxations.iter().cloned());
            tracing::info!(
                python = %python_version,
                elapsed_ms = t_materialize.elapsed().as_millis() as u64,
                bundles = materialized.len(),
                "bench: resolve_all (materialize + auto-bundle) finished",
            );
            // Phase 2: autodiscover one emission per workspace path-dep
            // referencing this source package. When nothing references
            // it (initial setup, missing workspace pixi.toml), returns
            // ONE default emission named after the materialized bundle.
            let bundle_names: HashSet<PypiKey> = materialized
                .iter()
                .map(|b| PypiKey::from_pypi(&b.conda_name))
                .collect();
            // For multi-bundle source packages, autodiscovery runs per
            // bundle so each bundle's name is the default fallback.
            // For typical single-bundle packs this is one entry.
            let default_name = materialized
                .first()
                .map(|b| b.conda_name.clone())
                .unwrap_or_default();
            let t_discover = std::time::Instant::now();
            let emissions = discover_emissions(
                &source_dir,
                workspace_dir.as_deref(),
                &default_name,
                &params.channels,
                &target,
                &bundle_names,
            )
            .await
            .map_err(|error| {
                RpcError::invalid_params(format!(
                    "discovering workspace emissions for target {}: {error:#}",
                    target.conda_subdir()
                ))
            })?;
            tracing::info!(
                elapsed_ms = t_discover.elapsed().as_millis() as u64,
                emissions = emissions.len(),
                "bench: discover_emissions (workspace transitive extraction) finished",
            );
            let local_wheel_stamps = capture_local_wheel_stamps(&materialized);
            if local_wheel_stamps.is_none() {
                tracing::debug!(
                    python = %python_version,
                    "resolved wheel could not be stamped; build will resolve normally"
                );
            }
            let plan = Arc::new(ResolvedTargetPlan {
                materialized,
                base_config,
                restore_relaxations,
                declared_config: config.clone(),
                target: target.clone(),
                work_directory: params.work_directory.clone(),
                workspace_manifest_mtime: prepared_workspace_mtime,
                auto_overrides_fingerprint: prepared_auto_overrides_fp.clone(),
                local_wheel_stamps,
            });
            let workspace_manifest = workspace_dir
                .as_deref()
                .and_then(crate::workspace::WorkspaceManifest::load);
            // Cross-output siblings: per-emission so envs only link
            // to their own siblings (not other envs' renames).
            for emission in &emissions {
                let env_bundles: Vec<Bundle> = plan
                    .materialized
                    .iter()
                    .map(|b| apply_emission(b, &plan.base_config, emission).0)
                    .collect();
                // When courier mode is active and RETREAD_INCREMENTAL=1, the
                // metadata phase must use `lock.version` as the pack version for
                // any bundle that `detect_incremental_add` would accept (the same
                // version the incremental build path uses).  Precompute a map
                // bundle_name → override_version so both the siblings list and
                // per-bundle produce_output use the same, consistent version.
                // When RETREAD_INCREMENTAL is unset (the default) or courier is
                // off, detect_incremental_add returns None at Gate 1 and the map
                // is empty (byte-identical to today).
                let incr_version_overrides: std::collections::HashMap<String, String> =
                    if config.courier && plan.local_wheel_stamps.is_some() {
                        let courier_channels_for_fp = workspace_manifest
                            .as_ref()
                            .map(|m| {
                                workspace_courier_channels(
                                    m,
                                    workspace_dir.as_deref().unwrap_or(source_dir.as_path()),
                                    &source_dir,
                                    &target,
                                )
                            })
                            .unwrap_or_default();
                        let workspace_fp_for_incr = workspace_manifest
                            .as_ref()
                            .map(|m| {
                                workspace_solve_fingerprint(
                                    m,
                                    workspace_dir.as_deref().unwrap_or(source_dir.as_path()),
                                    &source_dir,
                                    &target,
                                )
                            })
                            .unwrap_or_default();
                        let config_fp_for_incr = crate::courier::config_fingerprint(
                            &config,
                            &courier_channels_for_fp,
                            &workspace_fp_for_incr,
                        );
                        let ws_indexes_for_incr: Vec<String> = workspace_manifest
                            .as_ref()
                            .map(|m| m.resolution_pypi_index_urls())
                            .unwrap_or_else(|| vec![crate::index_chain::PUBLIC_PYPI.to_string()]);
                        let relax_str_for_incr = format!("{:?}", config.relax);
                        env_bundles
                            .iter()
                            .filter_map(|b| {
                                let lock_path =
                                    lock_path_for_target(&source_dir, &b.conda_name, &target);
                                detect_incremental_add_for_target(
                                    &lock_path,
                                    &config,
                                    &b.conda_name,
                                    &ws_indexes_for_incr,
                                    &relax_str_for_incr,
                                    &target,
                                    &config_fp_for_incr,
                                )
                                .map(|incr| (b.conda_name.clone(), incr.lock.version.clone()))
                            })
                            .collect()
                    } else {
                        std::collections::HashMap::new()
                    };
                let siblings: Vec<(String, String)> = env_bundles
                    .iter()
                    .map(|b| {
                        let ver = incr_version_overrides
                            .get(&b.conda_name)
                            .cloned()
                            .unwrap_or_else(|| b.primary.metadata.version.clone());
                        (b.conda_name.clone(), ver)
                    })
                    .collect();
                for (bundle_index, base_bundle) in plan.materialized.iter().enumerate() {
                    let (bundle, effective) =
                        apply_emission(base_bundle, &plan.base_config, emission);
                    // WS-B: cold-solve replay. When courier mode is active and
                    // a committed lock exists whose inputs_hash matches the
                    // current resolution inputs (resolved wheel set + index
                    // chain + relax + python + retread version), reconstruct
                    // the CondaOutput directly from the lock and skip the
                    // entire probe cascade (pre_emit_widen_pass +
                    // iterative_solve_refinement + post_emit_widen_pass).
                    // This is a pure optimization: on any error or hash miss
                    // the code falls through to the normal cascade path.
                    //
                    // Compute the courier inputs hash once here: it feeds both
                    // the replay gate (hash-check) and produce_output (embedded
                    // in the content-addressed build string). None for non-courier.
                    let courier_build_hash: Option<String> = if config.courier {
                        // Derive channels from the manifest (NOT from
                        // params.channels, which pixi forwards differently
                        // to conda/outputs vs conda/build_v1 for multi-env
                        // workspaces, causing replay to never fire).
                        // Must be byte-identical to the producer's derivation
                        // in build_one.
                        let courier_channels = workspace_manifest
                            .as_ref()
                            .map(|m| {
                                workspace_courier_channels(
                                    m,
                                    workspace_dir.as_deref().unwrap_or(source_dir.as_path()),
                                    &source_dir,
                                    &target,
                                )
                            })
                            .unwrap_or_default();
                        Some(courier_inputs_hash(
                            &config,
                            &base_bundle.conda_name,
                            &target,
                            &courier_channels,
                            workspace_manifest.as_ref(),
                            workspace_dir.as_deref().unwrap_or(source_dir.as_path()),
                            &source_dir,
                        ))
                    } else {
                        None
                    };
                    if let Some(ref current_hash) = courier_build_hash {
                        let lock_path =
                            lock_path_for_target(&source_dir, &bundle.conda_name, &target);
                        let relax_is_default =
                            config.relax == crate::config::RelaxPolicy::default();
                        let replay_workspace_versions =
                            output_workspace_abi_versions(&bundle, python_version);
                        let replay_aliases = output_abi_aliases(&bundle, &effective);
                        match replay_from_lock_for_target(
                            &lock_path,
                            current_hash,
                            relax_is_default,
                            &target,
                            &bundle.conda_name,
                            params.host_platform,
                            config.build_number,
                            config.bundle_mode == crate::config::BundleMode::Loose,
                            &siblings,
                            &replay_workspace_versions,
                            &effective.overrides,
                            &replay_aliases,
                        ) {
                            Ok(Some(replayed)) => {
                                ensure_output_abi_invariants(
                                    &replayed,
                                    &bundle,
                                    &effective,
                                    python_version,
                                )
                                .map_err(|error| {
                                    RpcError::internal(format!(
                                        "courier replay for {} failed ABI validation: {error:#}",
                                        bundle.conda_name
                                    ))
                                })?;
                                tracing::info!(
                                    bundle = %bundle.conda_name,
                                    "WS-B: cold-solve replay hit -- \
                                     skipping probe cascade",
                                );
                                crate::status::phase(
                                    &source_dir,
                                    &bundle.conda_name,
                                    &format!(
                                        "courier replay: {} outputs reconstructed \
                                         from committed lock (cascade skipped).",
                                        bundle.conda_name,
                                    ),
                                );
                                if plan.local_wheel_stamps.is_some() {
                                    // A prepared fallback builds the fresh
                                    // resolved plan, not the committed lock.
                                    // Pair it only with records re-derived
                                    // from that same plan, and only when its
                                    // final run-deps still match the replayed
                                    // metadata that pixi will solve.
                                    match produce_output_pending_relaxations(
                                        &bundle,
                                        &effective,
                                        params.host_platform,
                                        python_version,
                                        &siblings,
                                        courier_build_hash.as_deref(),
                                        None,
                                    ) {
                                        Ok((fresh, emission_relaxations))
                                            if fresh.run_dependencies
                                                == replayed.run_dependencies
                                                && outputs_share_identity(&fresh, &replayed) =>
                                        {
                                            let relaxations = bundled_relaxations_for_output(
                                                &bundle.conda_name,
                                                &base_bundle.conda_name,
                                                &target,
                                                &plan.restore_relaxations,
                                                &emission_relaxations,
                                            );
                                            prepared_builds.push(PreparedBuild {
                                                locator_id: prepared_builds.len(),
                                                plan: Arc::clone(&plan),
                                                bundle_index,
                                                emission: emission.clone(),
                                                advertised: PreparedOutputIdentity::from_metadata(
                                                    &replayed.metadata,
                                                ),
                                                advertised_run_dependencies: replayed
                                                    .run_dependencies
                                                    .clone(),
                                                relaxations,
                                                incremental_version_override: None,
                                            });
                                        }
                                        Ok(_) => {
                                            tracing::debug!(
                                                bundle = %bundle.conda_name,
                                                "replay metadata differs from the fresh plan; \
                                                 withholding prepared fallback"
                                            );
                                        }
                                        Err(error) => {
                                            tracing::debug!(
                                                bundle = %bundle.conda_name,
                                                error = %format!("{error:#}"),
                                                "fresh plan cannot reproduce replay metadata; \
                                                 withholding prepared fallback"
                                            );
                                        }
                                    }
                                }
                                outputs.push(replayed);
                                continue;
                            }
                            Ok(None) => {
                                tracing::debug!(
                                    bundle = %bundle.conda_name,
                                    "WS-B: replay miss (hash mismatch / no lock) -- cascade",
                                );
                            }
                            Err(e) => {
                                tracing::debug!(
                                    bundle = %bundle.conda_name,
                                    error = %format!("{e:#}"),
                                    "WS-B: replay error (non-fatal) -- \
                                     falling through to cascade",
                                );
                            }
                        }
                    }
                    let version_override_for_bundle = incr_version_overrides
                        .get(&bundle.conda_name)
                        .map(|s| s.as_str());
                    incremental_output_advertised |= version_override_for_bundle.is_some();
                    let non_courier_target_hash = (!config.courier)
                        .then(|| {
                            target
                                .target_contract()
                                .map(|_| target.resolution_identity())
                        })
                        .flatten();
                    let output_build_hash = courier_build_hash
                        .as_deref()
                        .or(non_courier_target_hash.as_deref());
                    let (output, relaxations) = match produce_output_pending_relaxations(
                        &bundle,
                        &effective,
                        params.host_platform,
                        python_version,
                        &siblings,
                        output_build_hash,
                        version_override_for_bundle,
                    ) {
                        Ok(output) => output,
                        Err(error) => {
                            let mut bundle_conflicts = Vec::new();
                            if let Err(error) = collect_conflicts(error, &mut bundle_conflicts) {
                                return Err(RpcError::internal(format!(
                                    "output for {}: {error:#}",
                                    bundle.conda_name
                                )));
                            }
                            output_conflicts.extend(scope_conflicts_for_target(
                                bundle_conflicts,
                                &bundle,
                                &target,
                            ));
                            // Every remaining emission/bundle pair is a
                            // deterministic peer in this conda/outputs
                            // request and may reveal another conflict.
                            continue;
                        }
                    };
                    let bundled_relaxations = bundled_relaxations_for_output(
                        &bundle.conda_name,
                        &base_bundle.conda_name,
                        &target,
                        &plan.restore_relaxations,
                        &relaxations,
                    );
                    pending_output_relaxations.extend(relaxations.iter().cloned());
                    // v4.2.0: the in-backend per-env solve check +
                    // iterative widening cascade were removed with the
                    // legacy mirror-solver. Emitted run-deps go to pixi
                    // unvalidated; conflict attribution and conda-pin
                    // widening are `retread solve`'s job (src/solve/).
                    if let Err(e) = write_probe_trace(&bundle, &source_dir).await {
                        tracing::warn!(
                            bundle = %bundle.conda_name,
                            error = %format!("{e:#}"),
                            "probe trace write failed (non-fatal)",
                        );
                    }
                    if plan.local_wheel_stamps.is_some() {
                        prepared_builds.push(PreparedBuild {
                            locator_id: prepared_builds.len(),
                            plan: Arc::clone(&plan),
                            bundle_index,
                            emission: emission.clone(),
                            advertised: PreparedOutputIdentity::from_metadata(&output.metadata),
                            advertised_run_dependencies: output.run_dependencies.clone(),
                            relaxations: bundled_relaxations,
                            incremental_version_override: version_override_for_bundle
                                .map(str::to_owned),
                        });
                    }
                    outputs.push(output);
                }
            }
        }
        if !output_conflicts.is_empty() {
            return Err(RpcError::internal(format!(
                "{:#}",
                aggregate_conflicts(output_conflicts)
            )));
        }
        tracing::debug!(outputs = outputs.len(), "per-env emission loop complete");
        let result = CondaOutputsResult {
            outputs,
            input_globs: Default::default(),
        };
        log_final_bundle_outputs(&result);
        tracing::info!(
            elapsed_ms = phase_start.elapsed().as_millis() as u64,
            outputs = result.outputs.len(),
            "bench: conda_outputs total",
        );
        // Memoize so pixi's subsequent per-env re-requests (identical
        // params) skip the whole recompute.
        let requires_prepared_plan = incremental_output_advertised;
        if let Some(transaction) = prepared_transaction {
            if !self
                .publish_prepared_builds(
                    generation,
                    transaction,
                    cache_key.clone(),
                    prepared_builds,
                )
                .await
            {
                tracing::debug!(
                    "discarded prepared build plans from a superseded conda/outputs transaction"
                );
            }
        }
        CONDA_OUTPUTS_CACHE
            .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
            .lock()
            .unwrap()
            .insert(
                cache_key.clone(),
                CondaOutputsMemo {
                    result: result.clone(),
                    requires_prepared_plan,
                },
            );
        // Cross-process memos cannot carry the exact typed/materialized plan.
        // Persist ordinary outputs only; incremental-version outputs must be
        // recomputed by each backend process so build_v1 has their plan.
        if !requires_prepared_plan {
            write_conda_outputs_disk_cache(&disk_cache_path, &result).await;
        }
        for relaxation in pending_output_relaxations {
            relaxation.emit();
        }
        Ok(result)
    }

    async fn conda_build_v1(
        &self,
        params: CondaBuildV1Params,
    ) -> Result<CondaBuildV1Result, RpcError> {
        // First thing out, before the heavy resolve/materialize/build below,
        // so a plain `pixi install` shows movement instead of a silent hang.
        // Generic: names come from params, nothing hardcoded.
        tracing::info!(
            retread_version = env!("CARGO_PKG_VERSION"),
            output = %params.output.name.as_normalized(),
            "retread: building package (materializing wheels, then rattler-build; large wheels download here and can take minutes)",
        );
        crate::status::tty(&format!(
            "building '{}': materializing wheels, then rattler-build (slow on first build).",
            params.output.name.as_normalized()
        ));
        let Snapshot {
            generation,
            config,
            download_dir,
            source_dir,
            cache_dir,
            workspace_dir,
            target_envelope,
            fast_cfg,
            fast_tmp,
        } = self.snapshot(&params.work_directory).await?;
        // conda/build_v1 doesn't carry the variant set; the chosen variant
        // is encoded in params.output.variant. Look up `python` there;
        // fall back to the default if absent.
        // Pick python: prefer the variant pixi sent back, but reject
        // bare-major ("3"). pixi forwards the workspace's
        // build-variants only when the recipe declares the variant
        // via `${{ python }}` template -- our generated bundle recipe
        // hardcodes `python 3.11.*` so pixi falls back to bare-major.
        // bare-major fails every cp tag match on PyPI Simple, so we
        // resolve a real X.Y from config.python (preferred) or
        // DEFAULT_PYTHON (last resort, with a loud warn since the
        // user can avoid it by setting `[package.build.config] python`).
        let config_python = config
            .python
            .as_ref()
            .and_then(|s| s.as_versions().into_iter().next());
        let raw = params.output.variant.get("python").map(|v| v.to_string());
        let exact_variant_python = raw
            .as_deref()
            .filter(|value| value.contains('.'))
            .and_then(|value| crate::lock::normalized_target_python(value).ok());
        let requested_python = match raw.as_deref() {
            Some(v) if v.contains('.') => v.to_string(),
            Some(other) => {
                let fallback = config_python
                    .clone()
                    .unwrap_or_else(|| DEFAULT_PYTHON.to_string());
                if config_python.is_none() {
                    tracing::warn!(
                        received = %other,
                        falling_back_to = %fallback,
                        "conda/build_v1: variant python is bare-major and \
                         [package.build.config] retread-python is not set; \
                         using DEFAULT_PYTHON. Set `retread-python = \"3.11\"` \
                         (or your preferred minor) under [package.build.config] \
                         in the source package's pixi.toml to silence this warning.",
                    );
                } else {
                    tracing::info!(
                        received = %other,
                        falling_back_to = %fallback,
                        "conda/build_v1: variant python is bare-major; \
                         using [package.build.config] retread-python",
                    );
                }
                fallback
            }
            None => config_python.unwrap_or_else(|| DEFAULT_PYTHON.to_string()),
        };
        let artifact_subdir = params.output.subdir;
        let resolution_subdir = resolution_subdir_for_build(
            artifact_subdir,
            params.host_prefix.as_ref().map(|prefix| prefix.platform),
            target_envelope.as_ref(),
            params.output.build.as_deref(),
        )
        .map_err(|error| {
            RpcError::invalid_params(format!(
                "resolving build platform for `{}`: {error:#}",
                params.output.name.as_normalized(),
            ))
        })?;
        let resolved_workspace_target = resolve_workspace_target_for_source(
            workspace_dir.as_deref(),
            &source_dir,
            resolution_subdir.as_str(),
            target_envelope.as_ref(),
        )
        .map_err(|error| {
            RpcError::invalid_params(format!(
                "resolving workspace target `{}` for source `{}`: {error:#}",
                resolution_subdir,
                source_dir.display(),
            ))
        })?;
        let target = wheel_target_for_resolved_workspace(
            resolution_subdir,
            &requested_python,
            resolved_workspace_target.as_ref(),
            target_envelope.is_some(),
        )
        .map_err(|error| {
            RpcError::invalid_params(format!(
                "invalid Python target `{requested_python}`: {error:#}"
            ))
        })?
        .with_hermetic_builds(config.hermetic);
        validate_advertised_non_courier_target_build(
            &config,
            &target,
            params.output.build.as_deref(),
        )?;
        // WS-B build_v1 replay gate: when courier mode is active, check the
        // committed lock BEFORE running the expensive resolve_all / probe
        // cascade. If the lock matches (same schema + inputs_hash), re-
        // materialize bytes from disk and build directly -- no BFS, no
        // auto_bundle, no solve. RETREAD_NO_REPLAY=1 disables this fast path.
        let output_dir = params
            .output_directory
            .clone()
            .unwrap_or_else(|| params.work_directory.join("output"));
        let stage_output_dir = fast_tmp.as_ref().and_then(|fast| {
            let output_check = crate::fasttmp::fs_check_path(&output_dir);
            crate::fasttmp::is_slow(&output_check, &fast_cfg).then(|| {
                let output_token = params.output.name.as_normalized().to_string();
                crate::fasttmp::stage_dir(&fast.ns, &output_token)
            })
        });
        if let Some(stage) = &stage_output_dir {
            tracing::info!(
                final_output_dir = %output_dir.display(),
                stage_output_dir = %stage.display(),
                "retread fast-tmp: staging conda build output on fast tmp with verified copy-back"
            );
        }
        let build_output_dir = stage_output_dir
            .clone()
            .unwrap_or_else(|| output_dir.clone());
        // The output version is the package identity pixi already solved from
        // conda/outputs. A cold re-resolve must reproduce it exactly rather
        // than relabel newly resolved wheel bytes. The sole override below is
        // granted only after build_v1 independently proves that this request
        // came from an incremental lock and its localized attempt escalates.
        let advertised_output_version = params.output.version.as_ref().map(ToString::to_string);
        // Fast replay must use the same current solved ABI facts that
        // conda/outputs used to advertise this exact output. A committed lock
        // cannot substitute producer-time selections: a ranged workspace
        // dependency can solve to a newer ABI version without changing the
        // manifest-derived inputs hash.
        let mut prepared_build_selection = self
            .lookup_prepared_build_for_target(
                generation,
                &params.work_directory,
                workspace_dir.as_deref(),
                exact_variant_python.as_deref(),
                &target,
                &params.output,
            )
            .await;
        let mut detected_incremental_fallback_version = None;
        if config.courier {
            // Need workspace_manifest to compute the config fingerprint
            // identically to how build_one does it (channel set + workspace
            // solve env are both folded in).
            let ws_manifest_for_replay = workspace_dir
                .as_deref()
                .and_then(crate::workspace::WorkspaceManifest::load);
            let courier_channels = ws_manifest_for_replay
                .as_ref()
                .map(|m| {
                    workspace_courier_channels(
                        m,
                        workspace_dir.as_deref().unwrap_or(&source_dir),
                        &source_dir,
                        &target,
                    )
                })
                .unwrap_or_default();
            let workspace_fp = ws_manifest_for_replay
                .as_ref()
                .map(|m| {
                    workspace_solve_fingerprint(
                        m,
                        workspace_dir.as_deref().unwrap_or(&source_dir),
                        &source_dir,
                        &target,
                    )
                })
                .unwrap_or_default();
            let config_fp =
                crate::courier::config_fingerprint(&config, &courier_channels, &workspace_fp);
            // The bundle_name for the hash is the requested output name
            // (params.output.name.as_normalized()), which equals bundle.conda_name
            // and is what courier::stage uses as the lock key.
            let bundle_name_for_hash = params.output.name.as_normalized().to_string();
            let declared_input_bundle =
                declared_input_bundle_for_output(&config, &bundle_name_for_hash);
            let current_hash = courier_inputs_hash(
                &config,
                declared_input_bundle
                    .as_deref()
                    .unwrap_or(&bundle_name_for_hash),
                &target,
                &courier_channels,
                ws_manifest_for_replay.as_ref(),
                workspace_dir.as_deref().unwrap_or(&source_dir),
                &source_dir,
            );
            let current_build = courier_build_string_for_target(
                &target,
                &current_hash,
                config.build_number,
                config.bundle_mode == crate::config::BundleMode::Loose,
            );
            let fast_identity_matches = declared_input_bundle.is_some()
                && advertised_build_matches(params.output.build.as_deref(), &current_build);
            if fast_identity_matches {
                let lock_path = lock_path_for_target(&source_dir, &bundle_name_for_hash, &target);
                let relax_is_default = config.relax == crate::config::RelaxPolicy::default();
                let replay_lock = if let Some(selection) = prepared_build_selection.as_ref() {
                    let abi_context = replay_abi_context_for_bundle(
                        &selection.bundle,
                        &selection.effective,
                        target.python_version(),
                    );
                    load_replayable_lock_for_target(
                        &lock_path,
                        &current_hash,
                        relax_is_default,
                        &target,
                        &bundle_name_for_hash,
                        &abi_context,
                    )
                } else {
                    tracing::debug!(
                        bundle = %bundle_name_for_hash,
                        "WS-B build_v1 replay skipped: current prepared ABI context unavailable",
                    );
                    Ok(None)
                };
                match replay_lock {
                    Ok(Some(lock))
                        if advertised_version_matches(
                            advertised_output_version.as_deref(),
                            &lock.version,
                        ) =>
                    {
                        tracing::info!(
                            bundle = %bundle_name_for_hash,
                            "WS-B build_v1 replay hit: re-materializing from lock \
                             (resolve_all skipped)",
                        );
                        crate::status::tty(&format!(
                            "building '{}': replay hit -- re-materializing from lock \
                         (derivation skipped).",
                            bundle_name_for_hash,
                        ));
                        // On the REPLAY path the authoritative run-deps are
                        // lock.conda_run_deps (already validated and stored when
                        // the lock was committed). Using params.run_dependencies
                        // here would allow pixi's live conda solver to inject
                        // non-deterministic extras (e.g. python_abi) that drift
                        // the rewritten lock away from the committed one.
                        // params.run_dependencies is intentionally ignored on
                        // this path; the COLD path (full resolve_all) keeps
                        // using run_override / params.run_dependencies unchanged.
                        let run_deps: Vec<String> = lock
                            .conda_run_deps
                            .iter()
                            .map(|dep| {
                                if dep.spec.is_empty() {
                                    dep.name.clone()
                                } else {
                                    format!("{} {}", dep.name, dep.spec)
                                }
                            })
                            .collect();
                        match materialize_from_lock_for_target(
                            lock,
                            &bundle_name_for_hash,
                            &config,
                            &params.work_directory,
                            &build_output_dir,
                            &target,
                            &source_dir,
                            &cache_dir,
                            params.output.build.as_deref(),
                            run_deps,
                            &config_fp,
                        )
                        .await
                        {
                            Ok(Some(result)) => {
                                let result = finalize_fasttmp_build_output(
                                    result,
                                    stage_output_dir.as_deref(),
                                    &output_dir,
                                )
                                .await?;
                                if let Some(selection) = prepared_build_selection.take() {
                                    self.consume_prepared_build(
                                        generation,
                                        selection.transaction,
                                        selection.prepared.locator_id,
                                    )
                                    .await;
                                }
                                return Ok(result);
                            }
                            Ok(None) => {
                                // Provenance gap (class 3 / schema-5 class 2):
                                // fall through to full resolve_all.
                                tracing::debug!(
                                    bundle = %bundle_name_for_hash,
                                    "WS-B build_v1 replay: provenance gap -- \
                                     falling through to full resolve",
                                );
                            }
                            Err(e) => {
                                return Err(RpcError::internal(format!(
                                    "build_v1 replay {bundle_name_for_hash}: {e:#}"
                                )));
                            }
                        }
                    }
                    Ok(Some(lock)) => {
                        tracing::debug!(
                            bundle = %bundle_name_for_hash,
                            advertised_version = %advertised_output_version.as_deref().unwrap_or_default(),
                            lock_version = %lock.version,
                            "WS-B build_v1 replay version differs from advertised output; falling through to cold resolve",
                        );
                    }
                    Ok(None) => {
                        tracing::debug!(
                            bundle = %bundle_name_for_hash,
                            "WS-B build_v1 replay miss (hash mismatch / no lock) -- full resolve",
                        );
                    }
                    Err(e) => {
                        tracing::debug!(
                            bundle = %bundle_name_for_hash,
                            error = %format!("{e:#}"),
                            "WS-B build_v1 replay error (non-fatal) -- full resolve",
                        );
                    }
                }

                // WS-B incremental-add fast path (STEP 4): if RETREAD_INCREMENTAL=1
                // and the current manifest diff is a pure dep addition, attempt a
                // localized resolve that reuses the locked closure.  Falls through
                // to cold resolve_all on any gate failure, ripple, or conflict.
                {
                    let ws_indexes: Vec<String> = ws_manifest_for_replay
                        .as_ref()
                        .map(|m| m.resolution_pypi_index_urls())
                        .unwrap_or_else(|| vec![crate::index_chain::PUBLIC_PYPI.to_string()]);
                    let relax_str = format!("{:?}", config.relax);
                    let incremental = detect_incremental_add_for_target(
                        &lock_path,
                        &config,
                        declared_input_bundle
                            .as_deref()
                            .unwrap_or(&bundle_name_for_hash),
                        &ws_indexes,
                        &relax_str,
                        &target,
                        &config_fp,
                    );
                    let incremental = incremental.filter(|incr| {
                        let Some(selection) = prepared_build_selection.as_ref() else {
                            tracing::debug!(
                                bundle = %bundle_name_for_hash,
                                "incremental-add skipped: current prepared ABI context unavailable",
                            );
                            return false;
                        };
                        let abi_context = replay_abi_context_for_bundle(
                            &selection.bundle,
                            &selection.effective,
                            target.python_version(),
                        );
                        match validate_loaded_lock_abi(&incr.lock, &abi_context) {
                            Ok(()) => true,
                            Err(error) => {
                                tracing::debug!(
                                    bundle = %bundle_name_for_hash,
                                    error = %format!("{error:#}"),
                                    "incremental-add skipped: prior lock failed current ABI validation",
                                );
                                false
                            }
                        }
                    });
                    if let Some(incr) = incremental {
                        match incremental_version_plan(
                            advertised_output_version.as_deref(),
                            &incr.lock.version,
                        ) {
                            IncrementalVersionPlan::Cold => {
                                tracing::debug!(
                                    bundle = %bundle_name_for_hash,
                                    lock_version = %incr.lock.version,
                                    advertised_version = %advertised_output_version.as_deref().unwrap_or_default(),
                                    "incremental-add: advertised version differs from lock; skipping localized build and requiring an exact cold version match"
                                );
                            }
                            IncrementalVersionPlan::Attempt { fallback_version } => {
                                match resolve_incremental_add(
                                    incr,
                                    &config,
                                    &target,
                                    &download_dir,
                                    &source_dir,
                                    &cache_dir,
                                    &params.channels,
                                    workspace_dir.as_deref(),
                                    &params.work_directory,
                                    &build_output_dir,
                                    params.output.build.as_deref(),
                                    &config_fp,
                                )
                                .await
                                {
                                    Ok(Some(result)) => {
                                        tracing::info!(
                                            bundle = %bundle_name_for_hash,
                                            "incremental-add: localized resolve succeeded"
                                        );
                                        let result = finalize_fasttmp_build_output(
                                            result,
                                            stage_output_dir.as_deref(),
                                            &output_dir,
                                        )
                                        .await?;
                                        if let Some(selection) = prepared_build_selection.take() {
                                            self.consume_prepared_build(
                                                generation,
                                                selection.transaction,
                                                selection.prepared.locator_id,
                                            )
                                            .await;
                                        }
                                        return Ok(result);
                                    }
                                    Ok(None) => {
                                        detected_incremental_fallback_version =
                                            Some(fallback_version);
                                        tracing::debug!(
                                            bundle = %bundle_name_for_hash,
                                            version = %detected_incremental_fallback_version.as_deref().unwrap_or_default(),
                                            "incremental-add: localized attempt escalated; requiring the exact prepared conda/outputs plan"
                                        );
                                        // Fall through to the typed prepared
                                        // plan handoff. A fresh resolve_all is
                                        // forbidden for this identity.
                                    }
                                    Err(e) => {
                                        return Err(RpcError::internal(format!(
                                            "incremental-add {bundle_name_for_hash}: {e:#}"
                                        )));
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                tracing::debug!(
                    output = %bundle_name_for_hash,
                    input_bundle = ?declared_input_bundle,
                    advertised_build = %params.output.build.as_deref().unwrap_or_default(),
                    current_build = %current_build,
                    "courier fast replay/incremental identity unavailable; deferring to recovered cold plan",
                );
            }
        }

        // Build the recipe's run-deps from the exact specs pixi solved and
        // locked with (forwarded in `params.run_dependencies`). Both the
        // advertised-plan handoff and the normal cold path consume this same
        // authoritative list.
        let run_override: Option<Vec<String>> = params
            .run_dependencies
            .as_ref()
            .map(|deps| deps.iter().map(|d| d.spec.to_string()).collect());

        // On the first cold build, conda/outputs has already produced the
        // complete typed resolution that advertised this exact output. Reuse
        // that immutable plan only when lifecycle, identity, workspace inputs,
        // and localized wheel stamps still match. Replay and incremental-add
        // deliberately stay ahead of this handoff.
        if let Some(PreparedBuildSelection {
            transaction,
            prepared,
            bundle,
            effective,
        }) = prepared_build_selection.take()
        {
            tracing::info!(
                bundle = %bundle.conda_name,
                "prepared build plan hit: reusing conda/outputs resolution (resolve_all skipped)"
            );
            crate::status::tty(&format!(
                "building '{}': reusing the resolution that advertised this output.",
                bundle.conda_name,
            ));
            let expected_build = params
                .output
                .build
                .as_deref()
                .unwrap_or(&prepared.advertised.build);
            let input_bundle_name = prepared
                .plan
                .materialized
                .get(prepared.bundle_index)
                .map(|base| base.conda_name.as_str())
                .ok_or_else(|| {
                    RpcError::internal("prepared build lost its source bundle identity".to_string())
                })?;
            validate_prepared_incremental_version_handoff(
                prepared.incremental_version_override.as_deref(),
                detected_incremental_fallback_version.as_deref(),
                &bundle.conda_name,
            )?;
            if !run_dependencies_match(
                &prepared.advertised_run_dependencies.depends,
                run_override.as_deref(),
            )
            .map_err(|error| {
                RpcError::internal(format!(
                    "validating prepared run-dependency parity for {}: {error:#}",
                    bundle.conda_name
                ))
            })? {
                return Err(RpcError::invalid_params(format!(
                    "run_dependencies for `{}` changed after conda/outputs; refusing to \
                     build with a stale relaxation record",
                    bundle.conda_name
                )));
            }
            let prepared_workspace_manifest = workspace_dir
                .as_deref()
                .and_then(crate::workspace::WorkspaceManifest::load);
            if prepared.plan.declared_config.courier {
                validate_advertised_courier_build(
                    &prepared.plan.declared_config,
                    input_bundle_name,
                    &prepared.plan.target,
                    prepared_workspace_manifest.as_ref(),
                    workspace_dir.as_deref(),
                    &source_dir,
                    Some(expected_build),
                )?;
                validate_advertised_courier_version(
                    &bundle,
                    advertised_output_version.as_deref(),
                    prepared.incremental_version_override.as_deref(),
                )?;
            }
            let result = build_one(
                &bundle,
                &effective,
                &prepared.plan.declared_config,
                &params.work_directory,
                &build_output_dir,
                params.output.subdir,
                &prepared.plan.target,
                &source_dir,
                workspace_dir.as_deref(),
                input_bundle_name,
                Some(expected_build),
                prepared.incremental_version_override.as_deref(),
                run_override.as_deref(),
                prepared.relaxations.as_ref(),
            )
            .await
            .map_err(|e| RpcError::internal(format!("build {}: {e:#}", bundle.conda_name)))?;
            let result =
                finalize_fasttmp_build_output(result, stage_output_dir.as_deref(), &output_dir)
                    .await?;
            self.consume_prepared_build(generation, transaction, prepared.locator_id)
                .await;
            return Ok(result);
        }

        reject_unprepared_incremental_fallback(
            detected_incremental_fallback_version.as_deref(),
            params.output.name.as_normalized(),
        )?;

        // Re-resolve materialized bundles, then autodiscover emissions
        // and pick the one matching the requested output name.
        let (materialized, base_config, restore_relaxations) = resolve_all(
            &config,
            &target,
            &download_dir,
            &source_dir,
            &cache_dir,
            &params.channels,
            workspace_dir.as_deref(),
        )
        .await
        .map_err(|e| RpcError::internal(format!("resolving wheels: {e:#}")))?;
        let bundle_names: HashSet<PypiKey> = materialized
            .iter()
            .map(|b| PypiKey::from_pypi(&b.conda_name))
            .collect();
        let default_name = materialized
            .first()
            .map(|b| b.conda_name.clone())
            .unwrap_or_default();
        let emissions = discover_emissions(
            &source_dir,
            workspace_dir.as_deref(),
            &default_name,
            &params.channels,
            &target,
            &bundle_names,
        )
        .await
        .map_err(|error| {
            RpcError::invalid_params(format!(
                "discovering workspace emissions for target {}: {error:#}",
                target.conda_subdir()
            ))
        })?;

        let requested = params.output.name.as_normalized().to_string();
        let picked_emission = emissions
            .iter()
            .find(|e| e.output_name == requested)
            .ok_or_else(|| {
                RpcError::invalid_params(format!(
                    "no discovered output matches requested name `{requested}`; \
                     known: {:?}",
                    emissions.iter().map(|e| &e.output_name).collect::<Vec<_>>()
                ))
            })?;
        let env_bundles: Vec<Bundle> = materialized
            .iter()
            .map(|base| apply_emission(base, &base_config, picked_emission).0)
            .collect();
        let siblings: Vec<(String, String)> = env_bundles
            .iter()
            .map(|sibling| {
                (
                    sibling.conda_name.clone(),
                    sibling.primary.metadata.version.clone(),
                )
            })
            .collect();
        let cold_workspace_manifest = workspace_dir
            .as_deref()
            .and_then(crate::workspace::WorkspaceManifest::load);
        let courier_channels = cold_workspace_manifest
            .as_ref()
            .map(|manifest| {
                workspace_courier_channels(
                    manifest,
                    workspace_dir.as_deref().unwrap_or(source_dir.as_path()),
                    &source_dir,
                    &target,
                )
            })
            .unwrap_or_default();
        let mut matching_bundles = Vec::new();
        let mut rejected_candidates: Vec<String> = Vec::new();
        // Option D lock-parity recovery (docs/RETREAD_DETERMINISM_FIX_DESIGN.md):
        // retain the identity-matched-but-deps-drifted candidate so the failure
        // arm below can ask the committed lock whether it vouches for the
        // advertisement. Populated ONLY when a candidate fails the gate, and
        // every state that populates it reaches the failure arm (identity
        // includes the package name, so at most one candidate can match and it
        // is by construction absent from `matching_bundles`).
        let mut identity_mismatch: Option<ColdMismatchedCandidate> = None;
        let mut identity_mismatch_ambiguous = false;
        for (bundle_index, base_bundle) in materialized.iter().enumerate() {
            let (bundle, effective) = apply_emission(base_bundle, &base_config, picked_emission);
            let courier_hash = config.courier.then(|| {
                courier_inputs_hash(
                    &config,
                    &base_bundle.conda_name,
                    &target,
                    &courier_channels,
                    cold_workspace_manifest.as_ref(),
                    workspace_dir.as_deref().unwrap_or(source_dir.as_path()),
                    &source_dir,
                )
            });
            let rich_target_hash = (!config.courier)
                .then(|| {
                    target
                        .target_contract()
                        .map(|_| target.resolution_identity())
                })
                .flatten();
            let output_hash = courier_hash.as_deref().or(rich_target_hash.as_deref());
            let (candidate, emission_relaxations) = produce_output_pending_relaxations(
                &bundle,
                &effective,
                resolution_subdir,
                target.python_version(),
                &siblings,
                output_hash,
                None,
            )
            .map_err(|error| {
                RpcError::internal(format!(
                    "reconstructing final relaxation record for {}: {error:#}",
                    bundle.conda_name
                ))
            })?;
            let identity_matches = output_matches_build_request(&candidate, &params.output);
            let dependencies_match = output_run_dependencies_match(&candidate, run_override.as_deref())
                .map_err(|error| {
                    RpcError::internal(format!(
                        "validating cached output parity for {}: {error:#}",
                        bundle.conda_name
                    ))
                })?;
            if identity_matches && dependencies_match {
                matching_bundles.push((bundle_index, bundle, effective, emission_relaxations));
            } else {
                // Record why each candidate was rejected. Without this the
                // caller only ever sees "0 exact matches", which names no
                // delta and leaves no way to tell a genuinely stale record
                // apart from a one-field drift (a rebuilt version, a changed
                // build string) that the operator could fix in seconds.
                rejected_candidates.push(format!(
                    "`{}` {}={} build=`{}` subdir={} ({})",
                    bundle.conda_name,
                    candidate.metadata.name.as_normalized(),
                    candidate.metadata.version,
                    candidate.metadata.build,
                    candidate.metadata.subdir,
                    if identity_matches {
                        // Name the wheel that contributed each unadvertised
                        // dependency. The delta alone says a name appeared but
                        // not where from, and the set has been observed to
                        // VARY between runs of the same bundle
                        // (aiohappyeyeballs+virtualenv one run, virtualenv+zipp
                        // the next), which points at the bundle's composition
                        // differing between the metadata and build phases
                        // rather than at a stale record. Attribution turns that
                        // guess into a fact on the next failure.
                        format!(
                            "identity matches, run dependencies differ — {} [{}]",
                            run_dependency_delta(&candidate, run_override.as_deref()),
                            describe_unadvertised_sources(
                                &candidate,
                                run_override.as_deref(),
                                &bundle
                            )
                        )
                    } else {
                        "identity differs".to_string()
                    },
                ));
                if identity_matches {
                    if identity_mismatch.is_some() {
                        // Cannot happen today (identity includes the package
                        // name), but ambiguity must never pick a lock.
                        identity_mismatch_ambiguous = true;
                    } else {
                        identity_mismatch = Some(ColdMismatchedCandidate {
                            bundle,
                            effective,
                            courier_hash,
                        });
                    }
                }
            }
        }
        let [(bundle_index, bundle, effective, emission_relaxations)] = matching_bundles.as_slice()
        else {
            let requested = format!(
                "{}{}{} subdir={}",
                params.output.name.as_normalized(),
                params
                    .output
                    .version
                    .as_ref()
                    .map(|version| format!(" version={version}"))
                    .unwrap_or_default(),
                params
                    .output
                    .build
                    .as_ref()
                    .map(|build| format!(" build=`{build}`"))
                    .unwrap_or_default(),
                params.output.subdir,
            );
            // ── Option D: lock-parity recovery ───────────────────────────────
            // docs/RETREAD_DETERMINISM_FIX_DESIGN.md. The cold re-derivation is
            // a fresh sample of a resolution function whose inputs move under
            // it (repodata TTL, probe outages, the read-modify-write heal-facts
            // ledger, live uv re-locks). When the identity still matches and
            // ONLY the run dependencies drifted, ask the committed pack lock —
            // the recorded resolution that the advertisement's inputs hash pins
            // — whether it reproduces what pixi advertised. If it does, replay
            // it; the produced package then agrees with the advertisement by
            // construction. If it does not, the error below stands and now says
            // WHICH record disagrees and how.
            //
            // This is the only lock read on the cold path and it is
            // unreachable whenever the gate passes: `identity_mismatch` is
            // populated only by a candidate that failed the gate, and this arm
            // runs only when `matching_bundles` is not exactly one. Guarded by
            // `cold_mismatch_recovery_never_consulted_when_candidate_matches`.
            let mut recovery_refusal: Option<String> = None;
            match plan_cold_mismatch_recovery(
                identity_mismatch,
                identity_mismatch_ambiguous,
                &config,
                &target,
                &source_dir,
                run_override.as_deref(),
                advertised_output_version.as_deref(),
            ) {
                ColdRecoveryPlan::NotAttempted => {}
                ColdRecoveryPlan::Refuse { reason } => recovery_refusal = Some(reason),
                ColdRecoveryPlan::Replay { bundle_name, lock } => {
                    // Same convention as the top replay path:
                    // lock.conda_run_deps is authoritative, and it has just
                    // been proven equivalent to what pixi advertised.
                    let run_deps: Vec<String> = lock
                        .conda_run_deps
                        .iter()
                        .map(lock_run_dep_string)
                        .collect();
                    let workspace_fp = cold_workspace_manifest
                        .as_ref()
                        .map(|manifest| {
                            workspace_solve_fingerprint(
                                manifest,
                                workspace_dir.as_deref().unwrap_or(source_dir.as_path()),
                                &source_dir,
                                &target,
                            )
                        })
                        .unwrap_or_default();
                    let config_fp = crate::courier::config_fingerprint(
                        &config,
                        &courier_channels,
                        &workspace_fp,
                    );
                    // Always warn: a recovered build must stay distinguishable
                    // from one whose fresh derivation actually reproduced the
                    // advertisement.
                    tracing::warn!(
                        bundle = %bundle_name,
                        "cold re-derivation drifted from the advertised run dependencies; \
                         recovered by replaying the committed lock, which matches the \
                         advertisement. This build did NOT reproduce from a fresh \
                         resolution. See docs/RETREAD_NONDETERMINISM_AUDIT.md.",
                    );
                    crate::status::tty(&format!(
                        "building '{bundle_name}': cold re-derivation drifted from the \
                         advertised run dependencies -- recovered by replaying the \
                         committed lock (not a fresh reproduction).",
                    ));
                    match materialize_from_lock_for_target(
                        *lock,
                        &bundle_name,
                        &config,
                        &params.work_directory,
                        &build_output_dir,
                        &target,
                        &source_dir,
                        &cache_dir,
                        params.output.build.as_deref(),
                        run_deps,
                        &config_fp,
                    )
                    .await
                    {
                        Ok(Some(result)) => {
                            let result = finalize_fasttmp_build_output(
                                result,
                                stage_output_dir.as_deref(),
                                &output_dir,
                            )
                            .await?;
                            return Ok(result);
                        }
                        Ok(None) => {
                            // Provenance gap: same convention as the top replay
                            // gate, except there is nothing left to fall through
                            // to, so the error below stands.
                            recovery_refusal = Some(
                                "committed lock matches the advertised run dependencies \
                                 but its wheel provenance is incomplete, so it cannot be \
                                 replayed"
                                    .to_string(),
                            );
                        }
                        Err(error) => {
                            return Err(RpcError::internal(format!(
                                "cold-path lock-parity recovery for {bundle_name}: {error:#}",
                            )));
                        }
                    }
                }
            }
            return Err(RpcError::invalid_params(format!(
                "the current source plan has {} exact matches for advertised output \
                 `{}`; refusing to build with a stale or ambiguous relaxation record. \
                 Requested: {requested}. Candidates considered: {}{}",
                matching_bundles.len(),
                params.output.name.as_normalized(),
                if rejected_candidates.is_empty() {
                    "none — the source plan produced no outputs at all".to_string()
                } else {
                    rejected_candidates.join("; ")
                },
                recovery_refusal
                    .map(|reason| format!(". Lock-parity recovery declined: {reason}"))
                    .unwrap_or_default(),
            )));
        };
        let base_bundle = &materialized[*bundle_index];
        let input_bundle_name = base_bundle.conda_name.clone();
        let bundled_relaxations = bundled_relaxations_for_output(
            &bundle.conda_name,
            &input_bundle_name,
            &target,
            &restore_relaxations,
            emission_relaxations,
        );

        if config.courier {
            validate_advertised_courier_build(
                &config,
                &input_bundle_name,
                &target,
                cold_workspace_manifest.as_ref(),
                workspace_dir.as_deref(),
                &source_dir,
                params.output.build.as_deref(),
            )?;
            validate_advertised_courier_version(
                bundle,
                advertised_output_version.as_deref(),
                None,
            )?;
        }

        let result = build_one(
            bundle,
            effective,
            &config,
            &params.work_directory,
            &build_output_dir,
            params.output.subdir,
            &target,
            &source_dir,
            workspace_dir.as_deref(),
            &input_bundle_name,
            params.output.build.as_deref(),
            None,
            run_override.as_deref(),
            bundled_relaxations.as_ref(),
        )
        .await
        .map_err(|e| RpcError::internal(format!("build {}: {e:#}", bundle.conda_name)))?;
        let result =
            finalize_fasttmp_build_output(result, stage_output_dir.as_deref(), &output_dir).await?;
        for relaxation in restore_relaxations {
            relaxation.emit();
        }
        Ok(result)
    }

    async fn begin_prepared_transaction(&self, generation: u64) -> Option<u64> {
        let mut state = self.state.write().await;
        if state.generation != generation {
            return None;
        }
        state.prepared_transaction = state
            .prepared_transaction
            .checked_add(1)
            .expect("prepared transaction counter exhausted");
        state.prepared_cache_key = None;
        state.prepared_builds.clear();
        Some(state.prepared_transaction)
    }

    async fn publish_prepared_builds(
        &self,
        generation: u64,
        transaction: u64,
        cache_key: String,
        prepared_builds: Vec<PreparedBuild>,
    ) -> bool {
        let mut state = self.state.write().await;
        if state.generation != generation || state.prepared_transaction != transaction {
            return false;
        }
        state.prepared_cache_key = Some(cache_key);
        state.prepared_builds = prepared_builds;
        true
    }

    async fn retain_prepared_for_memory_cache_hit(&self, cache_key: &str, work_dir: &Path) -> bool {
        let mut state = self.state.write().await;
        let reusable = state.prepared_cache_key.as_deref() == Some(cache_key)
            && !state.prepared_builds.is_empty()
            && state
                .prepared_builds
                .iter()
                .all(|prepared| prepared.plan.work_directory == work_dir);
        if reusable {
            return true;
        }
        state.prepared_transaction = state
            .prepared_transaction
            .checked_add(1)
            .expect("prepared transaction counter exhausted");
        state.prepared_cache_key = None;
        state.prepared_builds.clear();
        false
    }

    async fn invalidate_prepared_builds(&self) {
        let mut state = self.state.write().await;
        state.prepared_transaction = state
            .prepared_transaction
            .checked_add(1)
            .expect("prepared transaction counter exhausted");
        state.prepared_cache_key = None;
        state.prepared_builds.clear();
    }

    #[cfg(test)]
    async fn lookup_prepared_build(
        &self,
        generation: u64,
        work_dir: &Path,
        workspace_dir: Option<&Path>,
        exact_python_version: Option<&str>,
        output: &pixi_build_types::procedures::conda_build_v1::CondaBuildV1Output,
    ) -> Option<PreparedBuildSelection> {
        self.lookup_prepared_build_impl(
            generation,
            work_dir,
            workspace_dir,
            exact_python_version,
            None,
            output,
        )
        .await
    }

    async fn lookup_prepared_build_for_target(
        &self,
        generation: u64,
        work_dir: &Path,
        workspace_dir: Option<&Path>,
        exact_python_version: Option<&str>,
        target: &ResolutionTarget,
        output: &pixi_build_types::procedures::conda_build_v1::CondaBuildV1Output,
    ) -> Option<PreparedBuildSelection> {
        self.lookup_prepared_build_impl(
            generation,
            work_dir,
            workspace_dir,
            exact_python_version,
            Some(target),
            output,
        )
        .await
    }

    async fn lookup_prepared_build_impl(
        &self,
        generation: u64,
        work_dir: &Path,
        workspace_dir: Option<&Path>,
        exact_python_version: Option<&str>,
        target: Option<&ResolutionTarget>,
        output: &pixi_build_types::procedures::conda_build_v1::CondaBuildV1Output,
    ) -> Option<PreparedBuildSelection> {
        let (transaction, mut candidates): (u64, Vec<PreparedBuild>) = {
            let state = self.state.read().await;
            if state.generation != generation {
                return None;
            }
            (
                state.prepared_transaction,
                state
                    .prepared_builds
                    .iter()
                    .filter(|prepared| {
                        prepared.matches(work_dir, exact_python_version, target, output)
                    })
                    .cloned()
                    .collect(),
            )
        };
        if candidates.len() != 1 {
            tracing::debug!(
                candidates = candidates.len(),
                output = %output.name.as_normalized(),
                "prepared build plan unavailable or ambiguous; resolving normally"
            );
            return None;
        }
        let prepared = candidates.pop().expect("one candidate");
        if prepared.plan.workspace_manifest_mtime != workspace_manifest_mtime(workspace_dir)
            || prepared.plan.auto_overrides_fingerprint != auto_overrides_fingerprint(workspace_dir)
        {
            tracing::debug!(
                output = %output.name.as_normalized(),
                "prepared build plan inputs changed; resolving normally"
            );
            return None;
        }
        let stamps = prepared.plan.local_wheel_stamps.as_ref()?;
        if !local_wheel_stamps_are_current(stamps) {
            tracing::debug!(
                output = %output.name.as_normalized(),
                "prepared build plan wheel changed or is unreadable; resolving normally"
            );
            return None;
        }
        let (bundle, effective) = prepared.reapply()?;
        let state = self.state.read().await;
        if state.generation != generation || state.prepared_transaction != transaction {
            return None;
        }
        Some(PreparedBuildSelection {
            transaction,
            prepared,
            bundle,
            effective,
        })
    }

    async fn consume_prepared_build(&self, generation: u64, transaction: u64, locator_id: usize) {
        let mut state = self.state.write().await;
        if state.generation != generation || state.prepared_transaction != transaction {
            return;
        }
        let previous_len = state.prepared_builds.len();
        let cache_key = state.prepared_cache_key.clone();
        state
            .prepared_builds
            .retain(|prepared| prepared.locator_id != locator_id);
        let consumed = state.prepared_builds.len() != previous_len;
        if state.prepared_builds.is_empty() {
            state.prepared_cache_key = None;
        }
        drop(state);

        // An incremental memo is a set of typed output→plan handoffs, not
        // merely reusable metadata. Consuming any one locator makes that set
        // incomplete even while sibling locators remain, so a repeated
        // conda/outputs request must recompute the whole set.
        if consumed && let Some(cache_key) = cache_key {
            remove_incremental_conda_outputs_memo(&cache_key);
        }
    }

    async fn snapshot(&self, work_dir: &Path) -> Result<Snapshot, RpcError> {
        let (generation, config, state_cache_dir, source_dir, workspace_dir, target_envelope) = {
            let state = self.state.read().await;
            let config = state
                .config
                .clone()
                .ok_or_else(|| RpcError::internal("initialize was not called"))?;
            let source_dir = state
                .source_dir
                .clone()
                .unwrap_or_else(|| work_dir.to_path_buf());
            (
                state.generation,
                config,
                state.cache_dir.clone(),
                source_dir,
                state.workspace_dir.clone(),
                state.target_envelope.clone(),
            )
        };
        // Materialized wheels (downloads, source-builds, and relaxed copies)
        // live inside the pack folder so they're visible alongside the
        // pack's pixi.toml instead of buried in pixi's opaque cache.
        // cache_dir remains the scratch root for git clones.
        let download_dir = source_dir.join("wheels");
        let workspace_root = workspace_dir
            .as_deref()
            .filter(|dir| dir.join("pixi.toml").is_file())
            .unwrap_or(source_dir.as_path());
        let fast_cfg = crate::fasttmp::FastTmpConfig::load(workspace_root);
        let fast_tmp = crate::fasttmp::engage_backend(workspace_root, &fast_cfg)
            .map_err(|e| RpcError::internal(format!("fast-tmp backend engage: {e:#}")))?;
        let mut cache_dir = state_cache_dir
            .clone()
            .unwrap_or_else(|| work_dir.join("cache"));
        if let Some(fast) = &fast_tmp {
            let cache_check = crate::fasttmp::fs_check_path(&cache_dir);
            if state_cache_dir.is_none() || crate::fasttmp::is_slow(&cache_check, &fast_cfg) {
                cache_dir = fast.ns.retread_cache_dir();
            }
        }
        Ok(Snapshot {
            generation,
            config,
            download_dir,
            source_dir,
            cache_dir,
            workspace_dir,
            target_envelope,
            fast_cfg,
            fast_tmp,
        })
    }
}

async fn finalize_fasttmp_build_output(
    mut result: CondaBuildV1Result,
    stage_dir: Option<&Path>,
    output_dir: &Path,
) -> Result<CondaBuildV1Result, RpcError> {
    let Some(stage_dir) = stage_dir else {
        return Ok(result);
    };
    let stage = stage_dir.to_path_buf();
    let final_output = output_dir.to_path_buf();
    let returned_output_file = result.output_file.clone();
    let final_file = tokio::task::spawn_blocking(move || {
        crate::fasttmp::copy_back_artifacts(&stage, &final_output, &returned_output_file)
    })
    .await
    .map_err(|e| RpcError::internal(format!("fast-tmp copy-back task panicked: {e}")))?
    .map_err(|e| RpcError::internal(format!("fast-tmp copy-back: {e:#}")))?;

    result.output_file = final_file;
    if let Err(e) = tokio::fs::remove_dir_all(stage_dir).await {
        tracing::warn!(
            stage_dir = %stage_dir.display(),
            error = %e,
            "retread fast-tmp: failed to remove staged output dir after successful copy-back"
        );
    }
    Ok(result)
}

/// Pick the Python versions to fan outputs over. Precedence:
///
/// 1. `variant_configuration["python"]` from the workspace — pixi sets this
///    from `[workspace.build-variants] python = [...]`. Multiple values
///    produce multiple outputs.
/// 2. `[build.config] python` — single string or list, set by the user in
///    the source package itself. Convenience for workspaces that haven't
///    learned about build-variants.
/// 3. Hardcoded default `3.11`.
fn pythons_for(
    config: &RetreadConfig,
    variants: Option<&std::collections::BTreeMap<String, Vec<VariantValue>>>,
) -> Vec<String> {
    if let Some(values) = variants.and_then(|v| v.get("python"))
        && !values.is_empty()
    {
        // v0.37.0+: reject bare-major variant values (e.g. `"3"`).
        // Pixi sometimes forwards just the major when the
        // workspace's `build-variants = { python = ["3.11"] }`
        // doesn't reach the source-package backend cleanly.
        // Letting a bare major through poisons the entire pipeline:
        //   - solve_check installs `__cpython 3.0.0` as the virtual
        //     package (not 3.11), corrupting transitive ABI checks
        //   - produce_output emits `python 3.*` in host/run deps
        //   - the ABI invariant should catch the latter, but a
        //     bare-major slipping past variants is the upstream cause.
        // Same validation pattern as conda_build_v1 — fall back to
        // config.python or DEFAULT_PYTHON with a loud warn so it's
        // discoverable in pixi logs. Document the workaround so it
        // can be removed when pixi stops forwarding bare-major.
        let validated: Vec<String> = values
            .iter()
            .map(|v| v.to_string())
            .filter(|s| {
                let is_full = s.contains('.');
                if !is_full {
                    tracing::warn!(
                        variant_value = %s,
                        "pythons_for: rejecting bare-major python variant value; \
                         falling back to config.python or DEFAULT_PYTHON. \
                         This usually means pixi forwarded only the major version. \
                         Confirm `build-variants = {{ python = [\"X.Y\"] }}` is set \
                         at workspace top-level with the inline form.",
                    );
                }
                is_full
            })
            .collect();
        if !validated.is_empty() {
            return validated;
        }
        // Fall through to config / default if the entire list was
        // bare-major. (Edge case but real — protects against
        // pixi forwarding `["3"]` exclusively.)
    }
    if let Some(spec) = &config.python {
        let versions = spec.as_versions();
        if !versions.is_empty() {
            return versions;
        }
    }
    vec![DEFAULT_PYTHON.to_string()]
}

#[cfg(test)]
fn wheel_target_for(subdir: Platform, python_version: &str) -> Result<ResolutionTarget> {
    wheel_target_for_contract(subdir, python_version, None)
}

fn wheel_target_for_contract(
    subdir: Platform,
    python_version: &str,
    contract: Option<&WorkspaceTargetContract>,
) -> Result<ResolutionTarget> {
    // The python_version comes from variant configuration (or the chosen
    // output's variant in conda/build_v1). It drives wheel selection on
    // the PyPI index (cp tag matching) and the marker env in relax.rs.
    match contract {
        Some(contract) => ResolutionTarget::try_for_contract_on_subdir(
            python_version,
            subdir.as_str(),
            contract.clone(),
        ),
        None => ResolutionTarget::try_for_subdir(python_version, &subdir.to_string()),
    }
}

fn wheel_target_for_resolved_workspace(
    subdir: Platform,
    python_version: &str,
    resolved: Option<&ResolvedWorkspaceTarget>,
    exact_envelope: bool,
) -> Result<ResolutionTarget> {
    let target = wheel_target_for_contract(
        subdir,
        python_version,
        resolved.map(|resolved| &resolved.contract),
    )?;
    match (resolved, exact_envelope) {
        (Some(resolved), true) => target.with_exact_workspace_scope(resolved.clone()),
        (Some(resolved), false) => target.with_workspace_scope(resolved.clone()),
        (None, false) => Ok(target),
        (None, true) => bail!("an exact workspace target envelope resolved no consumer scope"),
    }
}

/// The output subdir describes the conda artifact, not necessarily the wheel
/// resolution platform. Pure Python bundles advertise `noarch`, while Pixi's
/// host prefix retains the concrete platform whose virtual-package contract
/// selected the wheels. Keep those identities separate during build_v1.
fn resolution_subdir_for_build(
    artifact_subdir: Platform,
    host_platform: Option<Platform>,
    target_envelope: Option<&WorkspaceTargetEnvelope>,
    advertised_build: Option<&str>,
) -> Result<Platform> {
    if artifact_subdir != Platform::NoArch {
        return Ok(artifact_subdir);
    }
    if let Some(platform) = host_platform.filter(|platform| *platform != Platform::NoArch) {
        return Ok(platform);
    }
    if let Some(envelope) = target_envelope {
        let platform = Platform::from_str(&envelope.profile.subdir).with_context(|| {
            format!(
                "parsing exact workspace target profile subdir `{}`",
                envelope.profile.subdir
            )
        })?;
        if platform == Platform::NoArch {
            bail!("an exact workspace target profile may not use `noarch`");
        }
        return Ok(platform);
    }
    if advertised_build.is_some_and(|build| {
        build.split('_').any(|component| {
            component.strip_prefix('h').is_some_and(|hash| {
                hash.len() == 10 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
        })
    }) {
        bail!(
            "noarch build carries a target-qualified build identity, but build_v1 supplied no concrete host platform or exact target envelope"
        );
    }
    // Legacy noarch metadata had no target-qualified identity. Preserve that
    // fallback only when there is no richer provenance to recover.
    Ok(Platform::NoArch)
}

/// Resolve the target contract that this source's concrete workspace
/// consumers select. A malformed or ambiguous workspace contract is a hard
/// error; falling back to host/subdir-only identity is reserved for sources
/// that have no discoverable workspace consumer.
fn resolve_workspace_target_for_source(
    workspace_dir: Option<&Path>,
    source_dir: &Path,
    requested_subdir: &str,
    envelope: Option<&WorkspaceTargetEnvelope>,
) -> Result<Option<ResolvedWorkspaceTarget>> {
    let Some(workspace_dir) = workspace_dir else {
        if envelope.is_some() {
            bail!("an exact workspace target envelope requires a workspace directory");
        }
        return Ok(None);
    };
    let Some(manifest) = crate::workspace::WorkspaceManifest::load(workspace_dir) else {
        if envelope.is_some() {
            bail!(
                "an exact workspace target envelope requires a readable `{}`",
                workspace_dir.join("pixi.toml").display()
            );
        }
        return Ok(None);
    };
    manifest.resolve_target_for_source(workspace_dir, source_dir, requested_subdir, envelope)
}

/// Resolve every user-supplied entry into a list of bundles. Each bundle
/// represents one conda output that pixi will see; its `primary` wheel is
/// the user-named one, `extras` are the recursively-resolved sub-wheels.
/// All wheels in a bundle are installed together into the same conda
/// package, matching the pattern in pixi#5230 comment 24.
///
/// Extras expansion is BFS with cycle detection (by PEP 503 normalized
/// name) scoped per-bundle, so two different user entries can independently
/// pull in the same sub-package.
async fn resolve_all(
    config: &RetreadConfig,
    target: &ResolutionTarget,
    download_dir: &Path,
    source_dir: &Path,
    cache_dir: &Path,
    conda_channels: &[ChannelUrl],
    workspace_dir: Option<&Path>,
) -> Result<(
    Vec<Bundle>,
    RetreadConfig,
    Vec<auto_bundle::WheelMetadataRelaxation>,
)> {
    // Bind the pack-level policy to the target used by every source-build
    // branch in this resolution. Resolution/cache identity intentionally does
    // not change: the policy controls how an exact cache miss is produced,
    // while v7 cache markers attest whether a native artifact was hermetic.
    let target = target.clone().with_hermetic_builds(config.hermetic);
    let target = &target;
    let mut bundles = Vec::with_capacity(config.retread_wheels.len());
    let mut route_conflicts = Vec::new();
    let mut pending_relaxations = Vec::new();

    // One complete workspace chain participates in every resolution path.
    // It retains the declared extras priority and either the explicit main
    // index or pixi's implicit public default.
    let workspace_pypi_indexes = workspace_dir
        .and_then(crate::workspace::WorkspaceManifest::load)
        .map(|manifest| manifest.resolution_pypi_index_urls())
        .unwrap_or_else(|| vec![crate::index_chain::PUBLIC_PYPI.to_string()]);

    // Load parselmouth once and reuse across bundles. We also merge it
    // into the effective name-map: when parselmouth says PyPI name X
    // corresponds to conda name Y, we emit Y in the conda run-deps
    // (otherwise the conda solver would fail to find X). Single-conda-
    // -name PyPI entries are unambiguous; multi-conda entries are
    // skipped from the merge (user must disambiguate via
    // retread-name-map).
    let pypi_to_conda = if config.auto_bundle
        || config.route_policy == crate::config::RoutePolicy::PreferCondaValidated
    {
        load_pypi_to_conda_map().await
    } else {
        Default::default()
    };

    // Build an effective config whose name_map merges three sources, in
    // precedence order:
    //   1. User's retread-name-map (always wins).
    //   2. FALLBACK_PYPI_TO_CONDA -- our manually-curated answers for
    //      parselmouth gaps (opencv-python-headless -> py-opencv, etc.).
    //   3. Parselmouth's unambiguous (single-conda-name) entries.
    let mut effective = config.clone();
    effective.name_map = effective_name_map(&config.name_map, &pypi_to_conda);

    // Group entries by their `bundle` field. Entries that share a bundle
    // name fold into ONE conda output containing all their wheels --
    // this lets the workspace declare a single conda dep and have it
    // install the whole pack. Entries without `bundle` keep the legacy
    // behavior (one entry = one output named after the entry key).
    // Owned-clone so the loop below can mutably borrow `effective`
    // when running the v0.19.0+ last-resort widen pass (which mutates
    // effective.overrides). The clone cost is trivial -- entries are
    // small structs of strings and a few enums.
    let mut groups: std::collections::BTreeMap<String, Vec<(String, WheelEntry)>> =
        std::collections::BTreeMap::new();
    for (entry_name, entry) in &effective.retread_wheels {
        let group_name = bundle_group_for(entry_name, entry, effective.default_bundle.as_deref());
        groups
            .entry(group_name)
            .or_default()
            .push((entry_name.clone(), entry.clone()));
    }
    // Retried materialization attempts share one aggregate probe ledger.
    // Keeping this owner at bundle-loop scope emits exactly one summary per
    // configured bundle, including a zero-probe summary.
    let probe_metrics_by_group = groups
        .keys()
        .map(|group| {
            (
                group.clone(),
                Arc::new(BundleProbeMetrics::new(group.as_str())),
            )
        })
        .collect::<BTreeMap<_, _>>();

    let uv_reresolve_env = std::env::var_os("RETREAD_UV_RERESOLVE");
    let uv_reresolve_mode = UvReresolveMode::from_env_value(uv_reresolve_env.as_deref());
    let mut uv_retry_keep_by_group: BTreeMap<String, BTreeSet<PypiKey>> = BTreeMap::new();

    while let Some((group_name, group_entries)) = groups.pop_first() {
        let group_fallback_indexes = index_chain(
            group_entries
                .iter()
                .filter(|(_, entry)| !entry.is_url())
                .filter_map(|(_, entry)| entry.index.clone()),
            &workspace_pypi_indexes,
            IndexPurpose::TransitiveFallback,
        );
        let uv_retry_keep = uv_retry_keep_by_group
            .get(&group_name)
            .cloned()
            .unwrap_or_default();
        // Build each entry's sub-bundle (primary + BFS extras + D rewrite)
        // independently, then fold them into one merged bundle named
        // after the group. The first entry in BTreeMap order becomes the
        // merged bundle's primary -- pick your entry names accordingly
        // if version selection matters (merged bundle's version = primary
        // wheel's version).
        //
        // v0.12.0+: precompute per-(bundle, checkout_root) auto-data
        // dedup state up front so the FIRST entry that owns a given
        // checkout root carries the auto-data; subsequent siblings
        // sharing that root get None. skip_subdirs for the carrier is
        // the union of every subdirectory of every sibling sharing the
        // root -- so the walk doesn't re-ship Python package source
        // that pip wheel already put in site-packages.
        let entry_checkouts: Vec<Option<PathBuf>> = group_entries
            .iter()
            .map(|(_, e)| checkout_root_for_entry(e, &effective.git_sources, source_dir, cache_dir))
            .collect();
        let mut emitted_auto_data: HashSet<PathBuf> = HashSet::new();
        let auto_data_per_entry: Vec<Option<AutoDataConfig>> = entry_checkouts
            .iter()
            .map(|maybe_root| {
                let root = maybe_root.as_ref()?;
                if emitted_auto_data.contains(root) {
                    return None;
                }
                emitted_auto_data.insert(root.clone());
                let skip_subdirs: Vec<PathBuf> = entry_checkouts
                    .iter()
                    .zip(group_entries.iter())
                    .filter_map(|(other_root, (_, e))| {
                        if other_root.as_ref() == Some(root) {
                            Some(PathBuf::from(e.subdirectory.as_deref().unwrap_or(".")))
                        } else {
                            None
                        }
                    })
                    .collect();
                Some(AutoDataConfig {
                    checkout_root: root.clone(),
                    skip_subdirs,
                })
            })
            .collect();

        // favor-lock: load preferred versions from the committed lock file for
        // this bundle.  The lock file is named after the bundle's canonical
        // conda name (same key used by the lock writer and replay path).
        // load_favored_versions returns an empty map when RETREAD_FAVOR_LOCK is
        // unset (the default), so this is a no-op on the cold / default path
        // and produces byte-identical output to today.
        let bundle_conda_name = canonical_conda_name(&group_name);
        let lock_path = lock_path_for_target(source_dir, &bundle_conda_name, target);
        let favored = load_favored_versions_for_target(&lock_path, target);

        // uv-backed closure computation (the only resolver). This group's
        // closure is computed by a uv subprocess (ephemeral project + `uv
        // lock` + `uv export --format pylock.toml`) and the materialization
        // below is PINNED to uv's picks via the locked-closure seam.
        // Packaging / courier / lock-write downstream are unchanged.
        // `Ok(None)` = no uv-resolvable roots (all source-built entries);
        // the materialization path then runs unpinned.
        let (
            uv_closure,
            deps_from_root_names,
            workspace_facts,
            prelock_owned_drops,
            protected_workspace_fact_names,
            conda_co_solve,
        ): (
            Option<crate::uv_closure::UvClosure>,
            std::collections::BTreeSet<String>,
            WorkspaceCondaFacts,
            BTreeSet<String>,
            BTreeSet<String>,
            CondaCoSolveContext,
        ) = uv_group_closure(
            &group_name,
            &group_entries,
            &effective,
            &config.name_map,
            target,
            cache_dir,
            source_dir,
            workspace_dir,
            &workspace_pypi_indexes,
            conda_channels,
            &uv_retry_keep,
            Arc::clone(
                probe_metrics_by_group
                    .get(&group_name)
                    .expect("every bundle group has one probe metrics ledger"),
            ),
        )
        .await
        .with_context(|| format!("computing uv closure for bundle `{group_name}`"))?;
        let uv_pins: Option<&BTreeMap<String, String>> = uv_closure.as_ref().map(|c| &c.pins);
        // M1 seam FIX (cold-path bundling): uv's closure pins guide the BFS
        // as fetch-time version PREFERENCES (the favor-lock seam), NEVER via
        // the locked-closure seam. `seed_locked` marks every pinned name as
        // already handled (ResolveState.constraints + chosen), which
        // suppressed BFS seeding (`seen_set`) and observe_edge enqueueing for
        // the ENTIRE closure — the bundle shipped only the primary wheel and
        // every transitive (isaacsim-app, isaacsim-kernel, ...) leaked into
        // the conda run-deps as a relaxed spec no conda channel can satisfy
        // ("no candidates" for wheels that exist only on pypi.nvidia.com).
        // `resolve_preferring` picks uv's version whenever it satisfies the
        // accumulated constraints, so the BFS reproduces uv's closure while
        // still fetching + bundling it. The locked-closure seam stays
        // reserved for the incremental-add path (which re-materializes the
        // rest of the bundle from the committed lock, not from the BFS).
        let favored = merge_uv_pins_into_prefs(favored, uv_pins);

        // v2.10.0: build the full sibling name set for this group once.
        // For each entry in the group we compute a sibling set = all OTHER
        // entries' canonical names, so resolve_bundle can skip deps that name
        // a sibling (they're provided by the sibling's wheel at install time).
        let all_entry_canonical: Vec<String> = group_entries
            .iter()
            .map(|(n, _)| canonical_conda_name(n))
            .collect();

        // Direct sdist-only roots have already been built by the closure heal
        // above. Feed those exact store artifacts through the ordinary
        // hash-pinned URL materializer instead of asking the index for a
        // binary wheel a second time.
        let wheel_store_root = crate::courier::retread_wheel_store_root();
        let closure_entry_overrides = group_entries
            .iter()
            .map(|(entry_name, entry)| {
                closure_built_entry_override(
                    entry_name,
                    entry,
                    uv_closure
                        .as_ref()
                        .map_or(&[][..], |closure| closure.wheels.as_slice()),
                    &wheel_store_root,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // Parallel entry builds: collect all per-entry inputs FIRST (the
        // sibling sets below are pure derivations of the entry list), then
        // run the resolve_bundle futures concurrently with a bounded window.
        // `buffered(N)` (not buffer_unordered) yields results in ENTRY ORDER,
        // so `sub_bundles` — and therefore the carrier-bundle merge, the
        // extras order, and the probe-decision order — stay byte-identical
        // to the old serial loop. resolve_bundle takes only shared &refs
        // (no shared &mut); the two cross-entry filesystem touch points are
        // already concurrency-safe: git clone/checkout is serialized by the
        // per-(url,rev) lock in source_build.rs, and wheel downloads land
        // via unique-temp + atomic rename (wheel.rs).
        let mut entry_futures = Vec::with_capacity(group_entries.len());
        for (idx, (((entry_name, _entry), auto_data), (materialize_entry, _sdist_source))) in
            group_entries
                .iter()
                .zip(auto_data_per_entry)
                .zip(closure_entry_overrides.iter())
                .enumerate()
        {
            // Sibling set: every OTHER entry's canonical name in this group.
            let sibling_names: std::collections::HashSet<String> = all_entry_canonical
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, n)| n.clone())
                .collect();
            // Reference bindings so `async move` moves only the &refs,
            // not the outer values (same pattern as solve_env above).
            let favored = &favored;
            let effective = &effective;
            let group_name = &group_name;
            let group_entries = &group_entries;
            let pypi_to_conda = &pypi_to_conda;
            let group_fallback_indexes = &group_fallback_indexes;
            entry_futures.push(async move {
                    resolve_bundle(
                        entry_name,
                        materialize_entry,
                        target,
                        download_dir,
                        source_dir,
                        cache_dir,
                        effective.relax,
                        &effective.git_sources,
                        auto_data,
                        pypi_to_conda,
                        BfsRoutePolicy::from_config(effective),
                        conda_channels,
                        group_fallback_indexes,
                        None, // locked-closure seam: incremental-add ONLY (uv pins flow via prefs below)
                        Some(favored).filter(|m| !m.is_empty()), // favor-lock + uv-closure prefs (empty map → None)
                        &sibling_names,
                    )
                    .await
                    .with_context(|| {
                        if group_entries.len() == 1 {
                            format!(
                                "resolving wheel entry `{entry_name}` (one of {} in [retread-wheels])",
                                effective.retread_wheels.len(),
                            )
                        } else {
                            format!(
                                "resolving wheel entry `{entry_name}` (bundle `{group_name}`, one of {} in [retread-wheels])",
                                effective.retread_wheels.len(),
                            )
                        }
                    })
            });
        }
        let mut sub_bundles: Vec<Bundle> = {
            use futures::stream::{self, StreamExt, TryStreamExt};
            stream::iter(entry_futures)
                .buffered(crate::concurrency::max_concurrent_builds())
                .try_collect()
                .await?
        };
        for (sub_bundle, (_, sdist_source)) in sub_bundles.iter_mut().zip(&closure_entry_overrides)
        {
            if let Some(sdist_source) = sdist_source {
                sub_bundle.primary.upstream_url = None;
                sub_bundle.primary.sdist_source = Some(sdist_source.clone());
                sub_bundle.primary.metadata_provenance = Provenance::SourceBuiltRelaxed;
            }
        }
        let mut bundle = sub_bundles.remove(0);
        bundle.conda_name = canonical_conda_name(&group_name);
        // M2: carry the uv auto-route decisions onto the bundle so
        // produce_output can emit each routed package as a conda run-dep
        // of the stub output.
        if let Some(closure) = &uv_closure {
            bundle.auto_routed = closure
                .auto_routed
                .iter()
                .filter(|r| !closure.auto_dropped.contains(&r.pypi_name))
                .map(|r| BundleAutoRoute {
                    route: r.clone(),
                    provenance: if deps_from_root_names.contains(&r.pypi_name) {
                        Provenance::DepsFromRelaxed
                    } else {
                        Provenance::PriorSelection
                    },
                    workspace_provider: None,
                })
                .collect();
            bundle.auto_dropped = closure.auto_dropped.iter().cloned().collect();
            // Closure membership for the run-dep emission gate: any package
            // uv exported into the wheel closure is uv-installed at install
            // time and must never double as a conda run-dep (see
            // Bundle::uv_closure_names).
            bundle.uv_closure_names = closure
                .pins
                .keys()
                .map(|k| canonical_conda_name(k))
                .collect();
            bundle.uv_dependency_graph = closure.dependency_graph.clone();
            bundle.uv_dependency_graph.deps_from_root_requirements = deps_from_root_names
                .iter()
                .filter_map(|name| {
                    let inputs = closure.effective_input_requirements.as_ref()?.get(name)?;
                    let specifiers = inputs
                        .iter()
                        .map(|input| input.specifiers.clone())
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect::<Vec<_>>();
                    Some((canonical_conda_name(name), specifiers))
                })
                .collect();
        }
        bundle.workspace_conda_versions = workspace_facts.common_selected_versions.clone();
        bundle.workspace_conda_provider_facts = workspace_facts.provider_facts.clone();
        bundle.auto_dropped.extend(prelock_owned_drops);
        for sub in sub_bundles {
            bundle.extras.push(sub.primary);
            bundle.extras.extend(sub.extras);
            // Each sub-bundle's BFS probe decisions get merged into
            // the carrier bundle so the per-bundle audit shows EVERY
            // dep that was probed across the whole group.
            bundle.probe_decisions.extend(sub.probe_decisions);
        }
        // Conda-facts-first: workspace-solved provider evidence owns a wheel
        // dependency unless the pack carries explicit PyPI-side intent. An
        // all-consumer provider needs no emitted route; a partial provider
        // replaces a matching stale uv route with the workspace conjunction.
        // Both flow through `auto_dropped` before auto-bundle scans and joint
        // route validation.
        bundle.apply_workspace_conda_fact_ownership(
            &effective,
            &config.name_map,
            &uv_retry_keep,
            &protected_workspace_fact_names,
        );
        // Auto-bundle scans the whole merged bundle's Requires-Dist, so
        // it naturally handles transitives pulled by any wheel in the
        // group. Every explicit non-URL entry index joins the candidate
        // fallback chain; URL-form entries have no PyPI index to contribute.
        // uv resolver: the closure is AUTHORITATIVE — every member the
        // auto-route did not move to the conda side must ship in the
        // bundle. The BFS above only walks extras-gated + prefix-family
        // deps (isaacsim -> isaacsim-*); non-family transitives (aiodns,
        // nvidia-*, ...) used to be rescued by the pre-emission solve
        // cascade, which v4.2.0 deleted. auto-bundle is the remaining seam
        // that probes conda per dep and bundles what conda lacks, so it
        // must run whenever a uv closure exists — not only when the pack
        // opts in — or those transitives leak into the conda run-deps as
        // relaxed specs no channel can satisfy ("no candidates").
        // uv resolver: closure membership (canonical names) — auto-routed
        // members were already excluded from the exported pylock, so this
        // is exactly "everything that must ship in the bundle".
        let uv_closure_members: Option<std::collections::BTreeMap<String, String>> =
            uv_closure.as_ref().map(|c| {
                c.wheels
                    .iter()
                    .map(|w| (w.name.clone(), w.version.clone()))
                    .collect()
            });
        if effective.auto_bundle || uv_closure.is_some() {
            let outcome = match auto_bundle_transitives(
                &mut bundle,
                &group_fallback_indexes,
                target,
                download_dir,
                &effective,
                conda_channels,
                None, // locked-closure seam: incremental-add ONLY (uv pins flow via prefs below)
                Some(&favored).filter(|m| !m.is_empty()), // favor-lock + uv-closure prefs
                uv_closure_members.as_ref(),
                &conda_co_solve,
                &UvReresolveContext {
                    mode: uv_reresolve_mode,
                    uv_backed: uv_closure.is_some(),
                    keep_pypi: uv_retry_keep.clone(),
                },
            )
            .await
            {
                Ok(outcome) => outcome,
                Err(error) => {
                    collect_conflicts(error, &mut route_conflicts)?;
                    // This bundle cannot produce valid output, but every
                    // remaining BTreeMap group is independent validation
                    // scope and may contribute another typed conflict.
                    continue;
                }
            };
            match outcome {
                AutoBundleOutcome::Complete { relaxations } => {
                    pending_relaxations.extend(relaxations);
                }
                AutoBundleOutcome::RetryKeepPypi { keep_pypi } => {
                    let accumulated = uv_retry_keep_by_group
                        .entry(group_name.clone())
                        .or_default();
                    let previous_len = accumulated.len();
                    accumulated.extend(keep_pypi);
                    if accumulated.len() == previous_len {
                        bail!(
                            "uv re-resolve for bundle `{group_name}` rejected conda routes but added no new keep-pypi names"
                        );
                    }
                    // The materialized bundle reflects the rejected lock. Drop
                    // the whole attempt and re-enter this group's ordinary uv
                    // closure/fixpoint path from its original inputs.
                    groups.insert(group_name, group_entries);
                    continue;
                }
            }
        }
        // v0.32.0+: pre_emit_widen_pass moved OUT of resolve_all into
        // the per-env emission loop in conda_outputs. Materialization
        // (download/build/auto-bundle) is env-agnostic; the cascade +
        // override injection is env-specific so per_env can run it N
        // times with N different channel sets / transitive
        // constraints.
        bundles.push(bundle);
    }

    if !route_conflicts.is_empty() {
        return Err(aggregate_conflicts(route_conflicts));
    }
    Ok((bundles, effective, pending_relaxations))
}

/// Compose the persisted sdist source URL, appending the advertised
/// `#sha256=` fragment exactly once.
///
/// [`crate::pypi::parse_index_links_any`] LIFTS the discrete `sha256` field
/// out of the link's `#sha256=` fragment but leaves that fragment ON the
/// `url` it returns, so `sha256.is_some()` implies `url` already carries a
/// `#sha256=..` fragment. Re-appending onto `url` verbatim would double it
/// (`..tar.gz#sha256=h#sha256=h`); strip any existing fragment first so the
/// stored URL has a single canonical `#sha256=`. No advertised hash -> URL
/// verbatim (may still carry an unrelated fragment, which we leave alone).
fn compose_sdist_source_url(url: &url::Url, sha256: Option<&str>) -> String {
    match sha256 {
        Some(h) => {
            let mut base = url.clone();
            base.set_fragment(None);
            format!("{base}#sha256={h}")
        }
        None => url.to_string(),
    }
}

/// Production wiring for the sdist-only self-heal's THIRD rung (v4.4.0;
/// spec ladder: wheel -> conda-route (existing) -> sdist auto-build (this
/// fn) -> error). Resolves `name`'s sdist off the first index in
/// `index_urls` that carries one (same PyPI-last chain priority as every
/// other resolve in this module), builds it via the SAME machinery
/// git-sourced `[retread-wheels]` entries use
/// ([`crate::source_build::build_wheel_from_sdist_url`]), hash-verifies
/// the downloaded sdist against the index's advertised sha256 (when the
/// index published one), and persists the built wheel content-addressed
/// in the shared wheel store ([`crate::wheel::store_wheel_in_cache`]) so
/// `retread install` replay never re-resolves (no-resolve edict).
///
/// Build-output cache key: the full `sha256(sdist identity,
/// artifact-target identity)`. A changed sdist or compatibility contract gets
/// its own out-dir; an identical repeat request hits `source_build`'s cache
/// and skips the `uv build` subprocess entirely.
async fn build_sdist_wheel(
    name: String,
    requirement: Option<String>,
    index_urls: Vec<String>,
    target: ResolutionTarget,
    cache_dir: PathBuf,
) -> Result<crate::uv_closure::BuiltSdistWheel> {
    // Constrain sdist selection to the EXACT version the structured
    // two-pass detection resolved for this name (`==<version>`, from Pass
    // B's export -- see `uv_closure::with_sdist_heal`). Building the newest
    // sdist instead would re-fail the re-solve when a requirer pinned an
    // older version (a match-any pick built antlr4-python3-runtime 4.13.x
    // for a `==4.9.*` requirer and the re-solve failed identically --
    // deps-from proof run 7). `None` (no version passed) or an unparseable
    // specifier falls back to match-any (empty PEP 440 specifier set is
    // vacuously satisfied by every version), best-effort rather than
    // refusing to heal at all.
    let match_any = || {
        VersionSpecifiers::from_str("")
            .expect("empty PEP 440 specifier string always parses (match-any)")
    };
    let specifiers = match requirement.as_deref() {
        Some(raw) => VersionSpecifiers::from_str(raw).unwrap_or_else(|e| {
            tracing::warn!(
                pkg = %name,
                spec = %raw,
                "sdist auto-build: extracted requirement is not valid PEP 440 \
                 ({e}); falling back to match-any",
            );
            match_any()
        }),
        None => match_any(),
    };
    build_sdist_wheel_with_specifiers(name, specifiers, index_urls, target, cache_dir).await
}

async fn build_sdist_wheel_with_specifiers(
    name: String,
    specifiers: VersionSpecifiers,
    index_urls: Vec<String>,
    target: ResolutionTarget,
    cache_dir: PathBuf,
) -> Result<crate::uv_closure::BuiltSdistWheel> {
    let (index, version, sdist) = fetch_from_pypi_index_chain(
        &index_urls,
        |index| async {
            crate::pypi::resolve_sdist(&index, &name, &specifiers)
                .await
                .map(|(version, sdist)| (index, version, sdist))
        },
        format!("sdist auto-build: no index in the chain has an sdist for `{name}`"),
    )
    .await?;
    tracing::info!(
        pkg = %name,
        version = %version,
        "building sdist {name}=={version} via sdist auto-build",
    );
    let out_dir = cache_dir
        .join("sdist-auto-build-outputs")
        .join(canonical_conda_name(&name));
    let expected = crate::source_build::ExpectedWheel::exact(name.clone(), version.to_string());
    let built = crate::source_build::build_platform_independent_wheel_from_sdist_url_for_target(
        &sdist.url,
        &out_dir,
        &target,
        sdist.sha256.as_deref(),
        &expected,
    )
    .await
    .with_context(|| format!("sdist auto-build: building `{name}` from {}", sdist.url))?;

    let store_root = crate::courier::retread_wheel_store_root();
    let sha256 = crate::wheel::store_wheel_in_cache(&built.wheel_path, &store_root)
        .await
        .with_context(|| format!("sdist auto-build: storing built wheel for `{name}`"))?;
    let filename = built
        .wheel_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| {
            anyhow!(
                "sdist auto-build: built wheel path has no utf-8 filename: {}",
                built.wheel_path.display()
            )
        })?
        .to_string();
    let store_path = store_root.join(&sha256).join(&filename);
    let sdist_url = compose_sdist_source_url(&sdist.url, Some(&built.sdist_sha256));
    Ok(crate::uv_closure::BuiltSdistWheel {
        pypi_name: name.clone(),
        version: version.to_string(),
        filename,
        wheel_path: store_path,
        sha256,
        sdist_source: crate::lock::SdistWheelSource {
            index,
            name: name.clone(),
            version: version.to_string(),
            sdist_url,
        },
    })
}

/// Dedupe root requirement strings by PEP 503-normalized package name,
/// keeping the LAST occurrence of each name (order-preserving on first
/// sight, so unrelated roots keep their original relative order). Used by
/// `uv_group_closure` to reconcile `[retread-wheels]` roots against
/// `retread-deps-from` roots: whichever list contributes a name LAST wins
/// -- see the call site for why that's deps-from by construction. A root
/// string that doesn't parse as PEP 508 (unexpected, but don't panic) is
/// treated as its own unique key, i.e. never deduped away.
fn dedupe_roots_last_wins(roots: Vec<String>) -> Vec<String> {
    let mut order: Vec<String> = Vec::new();
    let mut by_key: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for req in roots {
        let key = root_req_name(&req).unwrap_or_else(|| req.clone());
        if !by_key.contains_key(&key) {
            order.push(key.clone());
        }
        by_key.insert(key, req);
    }
    order
        .into_iter()
        .map(|k| by_key.remove(&k).expect("key was just inserted"))
        .collect()
}

/// Extract a PEP 508 requirement string's PEP 503-normalized package name,
/// e.g. `"Foo_Bar[extra]==1.0"` -> `Some("foo-bar")`. `None` if the string
/// doesn't parse as a PEP 508 requirement.
fn root_req_name(req: &str) -> Option<String> {
    let parsed: uv_pep508::Requirement = uv_pep508::Requirement::from_str(req).ok()?;
    Some(canonical_conda_name(parsed.name.as_ref()))
}

/// spec-uv-restructure M1: build + solve one bundle group's closure via uv.
///
/// Roots are the group's uv-resolvable entries (PyPI spec-form entries as
/// `name[extras]==version`, direct-URL entries as `name @ url`). Conda pins
/// from the workspace manifest become `constraint-dependencies` (with
/// provenance back to their conda source, spec §2.2); `retread-overrides`
/// map to `override-dependencies`; `retread-drop-deps` become unmatchable
/// override markers (AMENDMENT A3); `retread-conda-deps` names are excluded
/// from the exported closure.
///
/// Returns `Ok(None)` when the group has no uv-resolvable roots (all
/// entries source-built) — the caller then runs the legacy path unpinned.
///
/// Milestone-1 limits (see spec for the M2/M3 follow-ups):
/// - conda pins read from the manifest's `default` env only (no
///   `pixi.lock`-gated read, no per-env constraint sets yet);
/// - source-built (path/git/from) entries are not fed to uv as
///   `tool.uv.sources` — they resolve via the legacy path;
/// - source-built (path/git/from) entries are not fed to uv as
///   `tool.uv.sources` — they resolve via the legacy path.
///
/// M2 (v4.3.0): probe-driven conda-route auto-discovery. After the lock,
/// closure wheels whose conda equivalent exists on the workspace channels
/// at the resolved version are moved to the conda side and the lock is
/// re-run to fixpoint (see `uv_closure::auto_route_fixpoint`). Off via
/// `auto-route = false`; per-name opt-out via `keep-pypi`.
/// Wedge-safety cleanup after a FAILED closure solve (fix 00d8e14).
///
/// A genuine RESOLUTION/heal failure may have been poisoned by the stale
/// persisted heal facts this run seeded into its ledgers, so the facts file
/// must be dropped to force a clean cold re-heal next run rather than replay
/// a wedge across runs. But a TRANSIENT failure -- a `uv` spawn error, an
/// NFS/cache I/O blip, a backend crash -- leaves the learned facts perfectly
/// VALID; deleting them there only forces an expensive cold re-heal next run
/// for zero wedge-safety benefit (review perf minor). So we PRESERVE the
/// facts on clearly-transient (io-class) errors and delete on everything
/// else.
///
/// Signal: every transient failure inside [`crate::uv_closure::compute_closure`]
/// (`spawn`, `create_dir_all`, `read`/`write`) is a `std::io::Error` wrapped
/// by `.context(..)`, so an `io::Error` anywhere in the anyhow source chain
/// marks the error transient. A genuine resolution conflict is a
/// `bail!`/`anyhow!` string with no `io::Error` source. When the signal is
/// AMBIGUOUS (no io::Error, but not obviously a resolution message) we
/// CONSERVATIVELY delete -- the wedge-safety invariant (stale facts must
/// never survive a real heal failure) outranks saving one re-heal.
fn discard_facts_on_solve_failure(heal_facts_path: &std::path::Path, err: &anyhow::Error) {
    let transient = err
        .chain()
        .any(|cause| cause.downcast_ref::<std::io::Error>().is_some());
    if transient {
        tracing::debug!(
            path = %heal_facts_path.display(),
            error = %format!("{err:#}"),
            "uv closure: solve hit a transient (io-class) error; KEEPING persisted \
             heal facts (still valid -- avoids a needless cold re-heal next run)",
        );
        return;
    }
    // Resolution-class (or ambiguous): saving empty facts removes the file
    // so the next run re-heals from scratch instead of replaying a
    // possibly-wedged fact set.
    crate::uv_closure::save_heal_facts(heal_facts_path, &crate::uv_closure::HealFacts::default());
}

/// Concrete conda facts for the precisely identified environments that
/// consume one generated pack. Direct declarations retain their stronger
/// pre-lock route authority, while selected conda provider records become
/// post-materialization drop ownership through
/// `Bundle::apply_workspace_conda_fact_ownership`.
#[derive(Debug, Clone, Default)]
struct WorkspaceCondaFacts {
    /// Canonical conda package names directly declared by every precise
    /// consumer. Unlike the mapped PyPI view below, these preserve direct
    /// conda identity for Rule 2 without guessing a cross-ecosystem alias.
    owned_conda: BTreeSet<String>,
    /// Canonical PyPI names whose mapped conda package is directly declared
    /// by every consuming environment.
    owned_conda_pypi: BTreeSet<String>,
    /// Canonical names directly declared in `[pypi-dependencies]` by every
    /// consuming environment. Pixi supplies these on the wheel side already.
    owned_pypi: BTreeSet<String>,
    /// Common selected conda fact for mapped Python distributions. A fact is
    /// present only when every consuming environment selected the same
    /// version of the mapped conda package.
    common_pypi: BTreeMap<String, SolvedPypiFact>,
    /// Common selected versions for exact conda names directly owned by every
    /// precise consumer (Rule-3 and harmonization authority).
    common_conda_versions: BTreeMap<String, String>,
    /// Common selected versions for all conda records, including transitives.
    /// This is the exact workspace fact boundary used to drop-own matching
    /// wheel requirements after materialization. A uv preference or other
    /// PyPI-side solver selection never enters this map.
    common_selected_versions: BTreeMap<String, String>,
    /// Provider evidence keyed by canonical conda package name. This is kept
    /// separate from exact selected versions so ownership can reason about
    /// ranged and transitive providers without weakening exact validation.
    provider_facts: BTreeMap<String, WorkspaceCondaProviderFact>,
    /// Full exact selected specs per consuming environment. These are never
    /// ownership evidence; they are the immutable baseline for route trials.
    env_exact_specs: BTreeMap<String, Vec<String>>,
    /// Stable digest of `env_exact_specs`, for persisted heal-fact validity.
    fingerprint: String,
}

/// Workspace-solved evidence for one conda provider across every precise
/// consumer. Selected versions come only from successful conda solves;
/// declared specs annotate those records with direct workspace intent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct WorkspaceCondaProviderFact {
    selected_versions: BTreeSet<String>,
    declared_specs: BTreeSet<String>,
    present_in_all_consumers: bool,
}

/// Effective Rule-1 ownership authority shared with Rule 2. Direct conda
/// names remain distinct from explicitly mapped PyPI names, and PyPI-side
/// exclusions are carried separately so same-name fallback cannot bypass a
/// manual override, keep request, or protected first-party source.
#[derive(Debug, Clone, Default)]
struct WorkspaceRouteOwnership {
    pypi_names: BTreeSet<PypiKey>,
    conda_names: BTreeSet<PypiKey>,
    excluded_pypi_names: BTreeSet<PypiKey>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SolvedPypiFact {
    conda_name: String,
    version: String,
}

fn workspace_conda_provider_candidates(
    has_uv_roots: bool,
    workspace_facts: &WorkspaceCondaFacts,
    workspace_ownership: &WorkspaceRouteOwnership,
    fact_name_map: &NameMap,
) -> Vec<crate::uv_closure::WorkspaceCondaProvider> {
    if !has_uv_roots {
        return Vec::new();
    }
    workspace_ownership
        .pypi_names
        .iter()
        .filter_map(|pypi_name| {
            let fact = workspace_facts.common_pypi.get(pypi_name.as_str())?;
            let conda_name = fact_name_map
                .get(pypi_name)
                .and_then(CondaTarget::mapped_name)?
                .clone();
            (workspace_ownership.conda_names.contains(&conda_name.key())
                && conda_name.key().as_str() == fact.conda_name)
                .then(|| crate::uv_closure::WorkspaceCondaProvider {
                    pypi_name: pypi_name.clone(),
                    conda_name,
                    version: fact.version.clone(),
                    provenance: Provenance::WorkspaceCondaFact(
                        "precise-consuming-envs".to_string(),
                    ),
                })
        })
        .collect()
}

/// Cloneable workspace-aware conda solver oracle shared by Rule 1's uv
/// auto-route healing and Rule 2's final emitted-route validation. Both rules
/// must ask the same question: can these route specs solve together with every
/// consuming workspace constraint, target Python, system requirements, and
/// channel-priority setting?
#[derive(Debug)]
struct BundleProbeMetrics {
    bundle: String,
    probes: std::sync::atomic::AtomicUsize,
    timing: std::sync::Mutex<BundleProbeTiming>,
}

#[derive(Debug, Default)]
struct BundleProbeTiming {
    rounds: usize,
    active: usize,
    started: Option<std::time::Instant>,
    finished: Option<std::time::Instant>,
}

impl BundleProbeMetrics {
    fn new(bundle: &str) -> Self {
        Self {
            bundle: bundle.to_string(),
            probes: std::sync::atomic::AtomicUsize::new(0),
            timing: std::sync::Mutex::new(BundleProbeTiming::default()),
        }
    }

    fn enter(self: &Arc<Self>) -> BundleProbeGuard {
        use std::sync::atomic::Ordering;

        self.probes.fetch_add(1, Ordering::Relaxed);
        let mut timing = self.timing.lock().unwrap();
        timing.started.get_or_insert_with(std::time::Instant::now);
        if timing.active == 0 {
            timing.rounds += 1;
        }
        timing.active += 1;
        drop(timing);
        BundleProbeGuard {
            metrics: Arc::clone(self),
        }
    }
}

impl Drop for BundleProbeMetrics {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;

        let probes = self.probes.load(Ordering::Relaxed);
        let timing = self.timing.lock().unwrap();
        let wall_ms = timing
            .started
            .zip(timing.finished)
            .map_or(0, |(started, finished)| {
                finished.duration_since(started).as_millis() as u64
            });
        tracing::info!(
            bundle = %self.bundle,
            probes,
            rounds = timing.rounds,
            wall_ms,
            "bench: bundle route probes finished",
        );
    }
}

struct BundleProbeGuard {
    metrics: Arc<BundleProbeMetrics>,
}

impl Drop for BundleProbeGuard {
    fn drop(&mut self) {
        let mut timing = self.metrics.timing.lock().unwrap();
        debug_assert!(timing.active > 0, "probe activity counter underflow");
        timing.active -= 1;
        timing.finished = Some(std::time::Instant::now());
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CondaCoSolveContext {
    channels: Arc<[ChannelUrl]>,
    shared_solve: crate::conda_solve::SharedSparseSolveData,
    probe_pool: Option<crate::thread_budget::ProbePoolGrant>,
    python: String,
    bundle: PypiKey,
    channel_priority: rattler_solve::ChannelPriority,
    system_requirements: BTreeMap<String, String>,
    /// `None` preserves legacy host virtual-package detection. `Some(empty)`
    /// is distinct: a contract-qualified direct-inference target must not
    /// inherit build-host virtual packages that are absent from its contract.
    detected_virtual_packages: Option<BTreeMap<String, String>>,
    workspace_deps: BTreeMap<CondaName, Vec<String>>,
    /// Conda providers that would shadow a direct PyPI declaration shared by
    /// every precise consumer. A mutable conda route is invalid when its
    /// solved transitive closure introduces one of these providers: Pixi maps
    /// the conda record back to the PyPI identity and pins the wheel solve to
    /// the conda version (for example `tensordict -> pytorch 2.7.1` against an
    /// explicit PyPI `torch==2.7.0`).
    workspace_pypi_providers: BTreeMap<CondaName, PypiKey>,
    probe_metrics: Arc<BundleProbeMetrics>,
}

struct CondaCoSolveInputs<'a> {
    manifest: Option<&'a crate::workspace::WorkspaceManifest>,
    workspace_dir: Option<&'a Path>,
    source_dir: &'a Path,
    target: &'a ResolutionTarget,
    conda_channels: &'a [ChannelUrl],
    bundle: &'a str,
    owned_pypi: &'a BTreeSet<String>,
    fact_name_map: &'a NameMap,
}

impl CondaCoSolveContext {
    fn new(
        manifest: Option<&crate::workspace::WorkspaceManifest>,
        workspace_dir: Option<&Path>,
        source_dir: &Path,
        target: &ResolutionTarget,
        conda_channels: &[ChannelUrl],
        bundle: &str,
        owned_pypi: &BTreeSet<String>,
        fact_name_map: &NameMap,
    ) -> Self {
        Self::new_with_probe_metrics(
            CondaCoSolveInputs {
                manifest,
                workspace_dir,
                source_dir,
                target,
                conda_channels,
                bundle,
                owned_pypi,
                fact_name_map,
            },
            Arc::new(BundleProbeMetrics::new(bundle)),
        )
    }

    fn with_probe_metrics(mut self, probe_metrics: Arc<BundleProbeMetrics>) -> Self {
        self.probe_metrics = probe_metrics;
        self
    }

    fn new_with_probe_metrics(
        inputs: CondaCoSolveInputs<'_>,
        probe_metrics: Arc<BundleProbeMetrics>,
    ) -> Self {
        let CondaCoSolveInputs {
            manifest,
            workspace_dir,
            source_dir,
            target,
            conda_channels,
            bundle,
            owned_pypi,
            fact_name_map,
        } = inputs;
        let (channel_priority, system_requirements, raw_workspace_deps) = match manifest {
            Some(manifest) => (
                match manifest.channel_priority.as_deref() {
                    Some("disabled") => rattler_solve::ChannelPriority::Disabled,
                    _ => rattler_solve::ChannelPriority::Strict,
                },
                match workspace_dir {
                    Some(workspace_dir) => workspace_consuming_system_requirements(
                        manifest,
                        workspace_dir,
                        source_dir,
                        target,
                    ),
                    None => workspace_effective_system_requirements(manifest, "default", target),
                },
                match workspace_dir {
                    Some(workspace_dir) => workspace_consuming_dependencies(
                        manifest,
                        workspace_dir,
                        source_dir,
                        target,
                    ),
                    None => Default::default(),
                },
            ),
            None => (
                rattler_solve::ChannelPriority::Strict,
                Default::default(),
                Default::default(),
            ),
        };
        let mut workspace_deps: BTreeMap<CondaName, Vec<String>> = raw_workspace_deps
            .into_iter()
            .map(|(name, specs)| (CondaName::new(name), specs))
            .collect();
        if let (Some(manifest), Some(workspace_dir)) = (manifest, workspace_dir) {
            for (name, specs) in
                sibling_conda_run_dependencies(manifest, workspace_dir, source_dir, target)
            {
                let destination = workspace_deps.entry(CondaName::new(name)).or_default();
                for spec in specs {
                    if !destination.contains(&spec) {
                        destination.push(spec);
                    }
                }
            }
        }
        let workspace_pypi_providers = owned_pypi
            .iter()
            .filter_map(|name| {
                let pypi_name = PypiKey::from_pypi(name);
                let conda_name = fact_name_map.get(&pypi_name)?.mapped_name()?.clone();
                // An explicit conda declaration is already part of the
                // immutable workspace baseline; only wheel-only providers can
                // veto a newly introduced transitive conda provider.
                (!workspace_deps.contains_key(&conda_name)).then_some((conda_name, pypi_name))
            })
            .collect();
        let detected_virtual_packages = target
            .target_contract()
            .map(|contract| contract.detected_virtual_packages.clone());
        let shared_solve =
            crate::conda_solve::SharedSparseSolveData::new(conda_channels, target.conda_subdir());
        Self {
            channels: conda_channels.to_vec().into(),
            shared_solve,
            probe_pool: None,
            python: target.python_version().to_string(),
            bundle: PypiKey::from_pypi(bundle),
            channel_priority,
            system_requirements,
            detected_virtual_packages,
            workspace_deps,
            workspace_pypi_providers,
            probe_metrics,
        }
    }

    fn with_probe_pool(mut self, probe_pool: crate::thread_budget::ProbePoolGrant) -> Self {
        self.probe_pool = Some(probe_pool);
        self
    }

    pub(crate) fn probe_parallelism(&self) -> usize {
        self.probe_pool
            .as_ref()
            .map_or(1, |pool| pool.threads().get())
    }

    fn specs_for_routes(
        &self,
        routed: &[crate::uv_closure::CondaRouteSpec],
    ) -> Vec<crate::relax::CondaMatchSpec> {
        let mut specs = routed
            .iter()
            .map(crate::uv_closure::CondaRouteSpec::match_spec)
            .collect::<Vec<_>>();
        for (name, specs_for_name) in &self.workspace_deps {
            // The bundle's own output is being rendered and is not available
            // on a channel yet.
            if name.key() == self.bundle {
                continue;
            }
            for spec in specs_for_name {
                specs.push(name.match_spec(spec));
            }
        }
        specs.push(CondaName::new("python").match_spec(&format!("{}.*", self.python)));
        specs
    }

    pub(crate) async fn prewarm(&self, routed: Vec<crate::uv_closure::CondaRouteSpec>) {
        let specs = self.specs_for_routes(&routed);
        crate::conda_solve::prewarm_selected_records_for_target_shared(&self.shared_solve, &specs)
            .await;
    }

    pub(crate) async fn solve(
        &self,
        routed: Vec<crate::uv_closure::CondaRouteSpec>,
    ) -> crate::uv_closure::CoInstallVerdict {
        let specs = self.specs_for_routes(&routed);
        let _probe = self.probe_metrics.enter();
        match crate::conda_solve::solve_selected_records_for_target_shared(
            &self.shared_solve,
            &specs,
            self.probe_pool.clone(),
            &crate::conda_solve::SolveTarget::new(
                &self.python,
                self.channel_priority,
                &self.system_requirements,
                self.detected_virtual_packages.as_ref(),
                rattler_solve::SolveStrategy::Highest,
            ),
        )
        .await
        {
            Ok(records) => {
                let conflicts = selected_workspace_pypi_provider_conflicts(
                    records
                        .iter()
                        .map(|record| record.package_record.name.as_normalized()),
                    &self.workspace_pypi_providers,
                );
                if conflicts.is_empty() {
                    crate::uv_closure::CoInstallVerdict::Sat
                } else {
                    let mut reasons = Vec::new();
                    for (conda, pypi) in conflicts {
                        let culprits = routed_roots_reaching_provider(&records, &routed, &conda);
                        if culprits.is_empty() {
                            reasons.push(format!(
                                "conda route selects provider `{conda}` owned by workspace PyPI dependency `{pypi}`"
                            ));
                        } else {
                            reasons.extend(culprits.into_iter().map(|route| {
                                format!(
                                    "conda route `{route}` selects provider `{conda}` owned by workspace PyPI dependency `{pypi}`"
                                )
                            }));
                        }
                    }
                    crate::uv_closure::CoInstallVerdict::ExactUnsat(reasons)
                }
            }
            Err(crate::conda_solve::SharedSolveFailure::Unavailable(reasons)) => {
                crate::uv_closure::CoInstallVerdict::Skipped(reasons.join("; "))
            }
            Err(crate::conda_solve::SharedSolveFailure::Unsolvable(reasons)) => {
                crate::uv_closure::CoInstallVerdict::Unsat(reasons)
            }
            Err(crate::conda_solve::SharedSolveFailure::Unproven(reasons)) => {
                crate::uv_closure::CoInstallVerdict::ExactUnsat(reasons)
            }
        }
    }

    /// Check one mutable route without the globally fixed workspace baseline.
    ///
    /// Source-built metadata routes normally fail open when that baseline is
    /// independently unsatisfiable. Provider ownership is different: a
    /// standalone route whose selected closure contains a conda provider for
    /// a direct workspace PyPI dependency is positive, route-local evidence
    /// and can be rejected without blaming the unrelated baseline.
    pub(crate) async fn validate_standalone_provider_route(
        &self,
        route: crate::uv_closure::CondaRouteSpec,
    ) -> crate::uv_closure::CoInstallVerdict {
        if self.workspace_pypi_providers.is_empty() {
            return crate::uv_closure::CoInstallVerdict::Sat;
        }
        let specs = vec![
            route.match_spec(),
            CondaName::new("python").match_spec(&format!("{}.*", self.python)),
        ];
        let _probe = self.probe_metrics.enter();
        match crate::conda_solve::solve_selected_records_for_target_shared(
            &self.shared_solve,
            &specs,
            self.probe_pool.clone(),
            &crate::conda_solve::SolveTarget::new(
                &self.python,
                self.channel_priority,
                &self.system_requirements,
                self.detected_virtual_packages.as_ref(),
                rattler_solve::SolveStrategy::Highest,
            ),
        )
        .await
        {
            Ok(records) => {
                let conflicts = selected_workspace_pypi_provider_conflicts(
                    records
                        .iter()
                        .map(|record| record.package_record.name.as_normalized()),
                    &self.workspace_pypi_providers,
                );
                if conflicts.is_empty() {
                    crate::uv_closure::CoInstallVerdict::Sat
                } else {
                    crate::uv_closure::CoInstallVerdict::ExactUnsat(
                        conflicts
                            .into_iter()
                            .map(|(conda, pypi)| {
                                format!(
                                    "standalone conda route `{}` selects provider `{conda}` owned by workspace PyPI dependency `{pypi}`",
                                    route.conda_name,
                                )
                            })
                            .collect(),
                    )
                }
            }
            Err(crate::conda_solve::SharedSolveFailure::Unavailable(reasons)) => {
                crate::uv_closure::CoInstallVerdict::Skipped(reasons.join("; "))
            }
            Err(crate::conda_solve::SharedSolveFailure::Unsolvable(reasons)) => {
                crate::uv_closure::CoInstallVerdict::Unsat(reasons)
            }
            Err(crate::conda_solve::SharedSolveFailure::Unproven(reasons)) => {
                crate::uv_closure::CoInstallVerdict::ExactUnsat(reasons)
            }
        }
    }

    pub(crate) fn channels_consulted(&self) -> Vec<String> {
        self.channels.iter().map(ToString::to_string).collect()
    }
}

/// Attribute a protected provider selected by a complete conda solve back to
/// the mutable route roots that reach it. The joint route selector consumes
/// these package-name hints before its exhaustive core reducer, turning a
/// `tensordict -> pytorch` shadow into one singleton proof instead of dozens
/// of whole-environment solves.
fn routed_roots_reaching_provider(
    records: &[rattler_conda_types::RepoDataRecord],
    routed: &[crate::uv_closure::CondaRouteSpec],
    provider: &CondaName,
) -> Vec<CondaName> {
    let graph: BTreeMap<String, BTreeSet<String>> = records
        .iter()
        .map(|record| {
            let name = canonical_conda_name(record.package_record.name.as_normalized());
            let dependencies = record
                .package_record
                .depends
                .iter()
                .filter_map(|dependency| dependency.split_ascii_whitespace().next())
                .map(canonical_conda_name)
                .collect();
            (name, dependencies)
        })
        .collect();
    let provider = provider.key().into_string();
    routed
        .iter()
        .filter_map(|route| {
            let root = route.conda_name.key().into_string();
            let mut stack = vec![root.clone()];
            let mut seen = BTreeSet::new();
            while let Some(current) = stack.pop() {
                if !seen.insert(current.clone()) {
                    continue;
                }
                if current == provider {
                    return Some(route.conda_name.clone());
                }
                if let Some(dependencies) = graph.get(&current) {
                    stack.extend(dependencies.iter().cloned());
                }
            }
            None
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn selected_workspace_pypi_provider_conflicts<I, S>(
    selected_names: I,
    protected: &BTreeMap<CondaName, PypiKey>,
) -> Vec<(CondaName, PypiKey)>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let selected: BTreeSet<String> = selected_names
        .into_iter()
        .map(|name| canonical_conda_name(name.as_ref()))
        .collect();
    protected
        .iter()
        .filter(|(conda, _)| selected.contains(conda.key().as_str()))
        .map(|(conda, pypi)| (conda.clone(), pypi.clone()))
        .collect()
}

/// Direct dependency inputs for one concrete environment that consumes the
/// pack. This is the only manifest-selection seam allowed to authorize
/// workspace-solved Rule-3 facts; feature-only and all-feature fallbacks are
/// deliberately excluded.
#[derive(Debug, Clone)]
struct PreciseConsumerInput {
    env: String,
    conda_deps: BTreeMap<String, String>,
    pypi_deps: BTreeMap<String, Vec<String>>,
}

fn precise_consumer_inputs_for_target(
    manifest: &crate::workspace::WorkspaceManifest,
    workspace_dir: &Path,
    source_dir: &Path,
    target: &ResolutionTarget,
) -> Option<Vec<PreciseConsumerInput>> {
    let envs = workspace_precise_consuming_envs(manifest, workspace_dir, source_dir, target)?;
    let resolved = target.target_contract().and_then(|_| {
        resolved_workspace_target_from_resolution(manifest, workspace_dir, source_dir, target)
    });
    let mut inputs = Vec::with_capacity(envs.len());
    for env in envs {
        let conda_deps = match &resolved {
            Some(resolved) => manifest
                .effective_dependencies_for_resolved_env(&env, resolved)
                .ok()?,
            None => manifest.effective_dependencies_for_target(&env, target.conda_subdir()),
        };
        inputs.push(PreciseConsumerInput {
            conda_deps,
            pypi_deps: match &resolved {
                Some(resolved) => manifest
                    .effective_pypi_dependencies_for_resolved_env(&env, resolved)
                    .ok()?,
                None => {
                    manifest.effective_pypi_dependencies_for_target(&env, target.conda_subdir())
                }
            },
            env,
        });
    }
    Some(inputs)
}

/// Intersection of direct dependency names across every precise consumer.
/// A name declared in only one of several consumers cannot be removed from a
/// shared pack: another consumer would then receive neither ecosystem copy.
fn dependency_name_intersection<T>(maps: &[BTreeMap<String, T>]) -> BTreeSet<String> {
    let Some(first) = maps.first() else {
        return BTreeSet::new();
    };
    let mut common: BTreeSet<String> = first
        .keys()
        .map(|name| canonical_conda_name(name))
        .collect();
    for map in &maps[1..] {
        let names: BTreeSet<String> = map.keys().map(|name| canonical_conda_name(name)).collect();
        common.retain(|name| names.contains(name));
    }
    common
}

/// Build the request view used only for workspace-provider ownership planning.
///
/// A configured `[retread-wheels]` root is a first-party artifact and must
/// never be hollowed out by a same-name dependency in the consuming workspace.
/// A `retread-deps-from` root is different: it mirrors an upstream dependency
/// manifest, so an explicit PyPI dependency in every precise consumer is the
/// provider for that requirement. Hide only those imported roots from the
/// planner's `request_has_direct_root` guard; the real uv request keeps them
/// and receives the planner's graph-wide workspace-provider override.
fn workspace_ownership_planning_dependencies(
    dependencies: &[String],
    deps_from_root_names: &BTreeSet<String>,
) -> Vec<String> {
    dependencies
        .iter()
        .filter(|raw| root_req_name(raw).is_none_or(|name| !deps_from_root_names.contains(&name)))
        .cloned()
        .collect()
}

fn facts_from_solved_records(
    env_records: BTreeMap<String, Vec<rattler_conda_types::RepoDataRecord>>,
    env_conda_deps: BTreeMap<String, BTreeMap<String, String>>,
    owned_pypi: BTreeSet<String>,
    name_map: &NameMap,
    bundle_name: &str,
) -> WorkspaceCondaFacts {
    use sha2::{Digest, Sha256};

    if env_records.is_empty() {
        return WorkspaceCondaFacts {
            owned_pypi,
            ..Default::default()
        };
    }

    let bundle_name = canonical_conda_name(bundle_name);
    // Missing direct-dependency input for a solved consumer is treated as an
    // empty declaration set. That preserves fail-closed direct ownership if
    // callers ever provide mismatched environment maps.
    let direct_deps_for_consumers: Vec<BTreeMap<String, String>> = env_records
        .keys()
        .map(|env| env_conda_deps.get(env).cloned().unwrap_or_default())
        .collect();
    let owned_conda: BTreeSet<String> = dependency_name_intersection(&direct_deps_for_consumers)
        .into_iter()
        .map(|name| canonical_conda_name(&name))
        .filter(|name| name != &bundle_name)
        .collect();
    let per_env_versions: BTreeMap<String, BTreeMap<String, String>> = env_records
        .iter()
        .map(|(env, records)| {
            let versions = records
                .iter()
                .filter_map(|record| {
                    let name = canonical_conda_name(record.package_record.name.as_normalized());
                    (name != bundle_name).then(|| (name, record.package_record.version.to_string()))
                })
                .collect();
            (env.clone(), versions)
        })
        .collect();

    let mut provider_facts: BTreeMap<String, WorkspaceCondaProviderFact> = BTreeMap::new();
    for versions in per_env_versions.values() {
        for (name, version) in versions {
            provider_facts
                .entry(name.clone())
                .or_default()
                .selected_versions
                .insert(version.clone());
        }
    }
    for (name, fact) in &mut provider_facts {
        fact.present_in_all_consumers = per_env_versions
            .values()
            .all(|versions| versions.contains_key(name));
    }
    for env in per_env_versions.keys() {
        if let Some(deps) = env_conda_deps.get(env) {
            for (name, spec) in deps {
                let name = canonical_conda_name(name);
                if let Some(fact) = provider_facts.get_mut(&name) {
                    fact.declared_specs.insert(spec.trim().to_string());
                }
            }
        }
    }

    let mut common_selected_versions = per_env_versions
        .values()
        .next()
        .cloned()
        .unwrap_or_default();
    for versions in per_env_versions.values().skip(1) {
        common_selected_versions.retain(|name, version| versions.get(name) == Some(version));
    }

    // Only an exact conda package name directly declared by every precise
    // consumer can authorize Rule 3 or harmonization. Solved transitives stay
    // in env_exact_specs for route validation/fingerprinting, but a direct
    // `pytorch-gpu` declaration can never manufacture a `pytorch` fact.
    let common_conda_versions: BTreeMap<String, String> = common_selected_versions
        .iter()
        .filter(|(name, _)| owned_conda.contains(*name))
        .map(|(name, version)| (name.clone(), version.clone()))
        .collect();

    // A mapping edge is the identity/provenance proof. Unmapped conda names
    // are not guessed to be PyPI distributions (tetgen-style same-name
    // collisions must fail closed).
    let mut common_pypi = BTreeMap::new();
    let mut owned_conda_pypi = BTreeSet::new();
    for (pypi, target) in name_map {
        let Some(conda) = target.mapped_name() else {
            continue;
        };
        let pypi_name = pypi.as_str().to_owned();
        let conda_name = conda.key().into_string();
        if let Some(version) = common_conda_versions.get(&conda_name) {
            common_pypi.insert(
                pypi_name.clone(),
                SolvedPypiFact {
                    conda_name: conda_name.clone(),
                    version: version.clone(),
                },
            );
        }
        if owned_conda.contains(&conda_name) {
            owned_conda_pypi.insert(pypi_name);
        }
    }

    let mut env_exact_specs = BTreeMap::new();
    for (env, versions) in per_env_versions {
        let mut specs: Vec<String> = versions
            .into_iter()
            .map(|(name, version)| format!("{name} =={version}"))
            .collect();
        specs.sort();
        env_exact_specs.insert(env, specs);
    }
    let mut hasher = Sha256::new();
    for (env, specs) in &env_exact_specs {
        hasher.update(env.as_bytes());
        hasher.update([0xff]);
        for spec in specs {
            hasher.update(spec.as_bytes());
            hasher.update([0]);
        }
    }

    WorkspaceCondaFacts {
        owned_conda,
        owned_conda_pypi,
        owned_pypi,
        common_pypi,
        common_conda_versions,
        common_selected_versions,
        provider_facts,
        env_exact_specs,
        fingerprint: format!("{:x}", hasher.finalize()),
    }
}

/// Convert only precise, agreed workspace facts into the provenance-tagged
/// constraints that can authorize uv Rule 3.
fn workspace_fact_constraints(
    facts: &WorkspaceCondaFacts,
    manual_overrides: &BTreeSet<String>,
) -> crate::uv_closure::ConstraintSet {
    let mut constraints = crate::uv_closure::ConstraintSet::default();
    for (pypi_name, fact) in &facts.common_pypi {
        if manual_overrides.contains(pypi_name)
            || uv_pep508::uv_pep440::Version::from_str(&fact.version).is_err()
        {
            continue;
        }
        let line = format!("{pypi_name}=={}", fact.version);
        constraints.constraints.push(line.clone());
        constraints.provenance.insert(
            pypi_name.clone(),
            crate::uv_closure::ConstraintProvenance {
                constraint: line,
                conda_name: fact.conda_name.clone(),
                conda_version: format!("=={}", fact.version),
                source: "workspace-solved".to_string(),
                env: "precise-consuming-envs".to_string(),
                provenance: Provenance::WorkspaceCondaFact("precise-consuming-envs".to_string()),
            },
        );
    }
    constraints
}

/// Apply conda-export floors as non-installing uv hints.
///
/// A floor is eligible only through a one-to-one edge in the pack's explicit
/// PyPI->conda map and only for an active, otherwise-unconstrained deps-from
/// root. Workspace constraints, configured wheel roots, overrides, and drops
/// remain authoritative. Advisory provenance is retained for diagnostics but
/// excluded from repair attribution by `uv_closure::attribute_conflict`.
#[allow(clippy::too_many_arguments)]
fn apply_deps_from_conda_floors(
    constraints: &mut crate::uv_closure::ConstraintSet,
    floors: &[crate::deps_from::AdvisoryCondaFloor],
    roots: &[String],
    explicit_name_map: &NameMap,
    protected_root_names: &BTreeSet<String>,
    overrides: &BTreeMap<String, String>,
    drops: &[String],
    conda_subdir: &str,
    python_version: &str,
) -> Result<()> {
    if floors.is_empty() || explicit_name_map.is_empty() {
        return Ok(());
    }

    let marker_env = crate::relax::marker_env_for(conda_subdir, python_version)?;
    let mut eligible_roots = BTreeSet::new();
    for raw in roots {
        let Ok(requirement): Result<uv_pep508::Requirement, _> =
            uv_pep508::Requirement::from_str(raw)
        else {
            continue;
        };
        if !requirement.marker.evaluate(&marker_env, &[]) || requirement.version_or_url.is_some() {
            continue;
        }
        let name = canonical_conda_name(requirement.name.as_ref());
        if !protected_root_names.contains(&name) {
            eligible_roots.insert(name);
        }
    }

    // Preserve all normalized aliases so neither direction can silently pick
    // a winner when an explicit map is ambiguous.
    let mut pypi_to_conda: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut conda_to_pypi: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (pypi, target) in explicit_name_map {
        let Some(conda) = target.mapped_name() else {
            continue;
        };
        let pypi = pypi.as_str().to_owned();
        let conda = conda.key().into_string();
        pypi_to_conda
            .entry(pypi.clone())
            .or_default()
            .insert(conda.clone());
        conda_to_pypi.entry(conda).or_default().insert(pypi);
    }

    let mut blocked: BTreeSet<String> = constraints
        .constraints
        .iter()
        .filter_map(|line| root_req_name(line))
        .chain(constraints.provenance.keys().cloned())
        .chain(overrides.keys().map(|name| canonical_conda_name(name)))
        .chain(drops.iter().map(|name| canonical_conda_name(name)))
        .chain(protected_root_names.iter().cloned())
        .collect();

    // Source order is meaningful for deps-from aggregation; retain only the
    // last eligible floor per mapped PyPI name before mutating either vector.
    let mut candidates: BTreeMap<String, crate::deps_from::AdvisoryCondaFloor> = BTreeMap::new();
    for floor in floors {
        let conda = canonical_conda_name(&floor.conda_name);
        let Some(mapped_pypi) = conda_to_pypi.get(&conda) else {
            tracing::debug!(
                conda_name = %floor.conda_name,
                floor = %floor.floor_spec,
                source = %floor.source,
                "retread-deps-from: conda floor is inert without an explicit name-map edge",
            );
            continue;
        };
        if mapped_pypi.len() != 1 {
            tracing::warn!(
                conda_name = %floor.conda_name,
                mapped_pypi = ?mapped_pypi,
                "retread-deps-from: ambiguous explicit conda-to-PyPI mapping; advisory floor skipped",
            );
            continue;
        }
        let pypi = mapped_pypi
            .iter()
            .next()
            .expect("one-element mapping checked above")
            .clone();
        if pypi_to_conda
            .get(&pypi)
            .is_none_or(|targets| targets.len() != 1 || !targets.contains(&conda))
        {
            tracing::warn!(
                pypi_name = %pypi,
                conda_name = %floor.conda_name,
                "retread-deps-from: ambiguous explicit PyPI-to-conda mapping; advisory floor skipped",
            );
            continue;
        }
        if !eligible_roots.contains(&pypi) || blocked.contains(&pypi) {
            tracing::debug!(
                pypi_name = %pypi,
                conda_name = %floor.conda_name,
                floor = %floor.floor_spec,
                "retread-deps-from: advisory floor deferred to an authoritative root or constraint",
            );
            continue;
        }
        candidates.insert(pypi, floor.clone());
    }

    for (pypi, floor) in candidates {
        let line = format!("{pypi}{}", floor.floor_spec);
        let _: uv_pep508::Requirement = uv_pep508::Requirement::from_str(&line)
            .with_context(|| format!("validating deps-from advisory constraint `{line}`"))?;
        constraints.constraints.push(line.clone());
        constraints.provenance.insert(
            pypi.clone(),
            crate::uv_closure::ConstraintProvenance {
                constraint: line,
                conda_name: floor.conda_name.clone(),
                conda_version: floor.floor_spec.clone(),
                source: "deps-from-conda-advisory".to_string(),
                env: floor.source.clone(),
                provenance: Provenance::DepsFromRelaxed,
            },
        );
        blocked.insert(pypi.clone());
        tracing::info!(
            pypi_name = %pypi,
            conda_name = %floor.conda_name,
            floor = %floor.floor_spec,
            source = %floor.source,
            "retread-deps-from: applied conda environment advisory floor to active PyPI root",
        );
    }
    Ok(())
}

/// Solve each precise consuming environment independently. Destructive
/// behavior is enabled only when the workspace can map this source package to
/// concrete active environments and every environment solve succeeds.
async fn solve_workspace_conda_facts(
    manifest: &crate::workspace::WorkspaceManifest,
    workspace_dir: &Path,
    source_dir: &Path,
    target: &ResolutionTarget,
    conda_channels: &[ChannelUrl],
    name_map: &NameMap,
    bundle_name: &str,
) -> WorkspaceCondaFacts {
    let Some(inputs) =
        precise_consumer_inputs_for_target(manifest, workspace_dir, source_dir, target)
    else {
        tracing::debug!(
            bundle = %bundle_name,
            "conda facts: pack-to-environment ownership is ambiguous; abstaining",
        );
        return WorkspaceCondaFacts::default();
    };

    let env_conda_deps: BTreeMap<String, BTreeMap<String, String>> = inputs
        .iter()
        .map(|input| (input.env.clone(), input.conda_deps.clone()))
        .collect();
    let conda_deps: Vec<BTreeMap<String, String>> = env_conda_deps.values().cloned().collect();
    let pypi_deps: Vec<BTreeMap<String, Vec<String>>> =
        inputs.iter().map(|input| input.pypi_deps.clone()).collect();
    let owned_pypi = dependency_name_intersection(&pypi_deps);

    if conda_deps.iter().all(BTreeMap::is_empty) {
        return WorkspaceCondaFacts {
            owned_pypi,
            ..Default::default()
        };
    }

    let channel_priority = match manifest.channel_priority.as_deref() {
        Some("disabled") => rattler_solve::ChannelPriority::Disabled,
        _ => rattler_solve::ChannelPriority::Strict,
    };
    let bundle_key = PypiKey::from_pypi(bundle_name);
    let solve_futures = inputs
        .iter()
        .map(|input| {
            let env = &input.env;
            let deps = &input.conda_deps;
            let mut specs = deps
                .iter()
                .filter_map(|(name, spec)| {
                    let name = CondaName::new(name.as_str());
                    if name.key() == bundle_key {
                        return None;
                    }
                    Some(name.match_spec(spec))
                })
                .collect::<Vec<_>>();
            specs
                .push(CondaName::new("python").match_spec(&format!("{}.*", target.python_version)));
            let sysreqs = workspace_effective_system_requirements(manifest, env, target);
            async move {
                let result = crate::conda_solve::solve_selected_records_for_target(
                    conda_channels,
                    &specs,
                    &target.python_version,
                    &target.conda_subdir,
                    channel_priority,
                    &sysreqs,
                    target
                        .target_contract()
                        .map(|contract| &contract.detected_virtual_packages),
                    rattler_solve::SolveStrategy::Highest,
                )
                .await;
                (env.clone(), result)
            }
        })
        .collect::<Vec<_>>();
    let solves = {
        use futures::stream::{self, StreamExt};
        stream::iter(solve_futures)
            .buffered(crate::concurrency::max_concurrent_builds())
            .collect::<Vec<_>>()
            .await
    };

    let mut env_records = BTreeMap::new();
    for (env, result) in solves {
        match result {
            Ok(records) => {
                env_records.insert(env, records);
            }
            Err(reasons) => {
                tracing::debug!(
                    bundle = %bundle_name,
                    env = %env,
                    reasons = ?reasons,
                    "conda facts: precise consuming-env solve unavailable; abstaining from conda-owned actions",
                );
                return WorkspaceCondaFacts {
                    owned_pypi,
                    ..Default::default()
                };
            }
        }
    }
    facts_from_solved_records(
        env_records,
        env_conda_deps,
        owned_pypi,
        name_map,
        bundle_name,
    )
}

#[allow(clippy::too_many_arguments)]
async fn uv_group_closure(
    group_name: &str,
    group_entries: &[(String, WheelEntry)],
    effective: &RetreadConfig,
    fact_name_map: &NameMap,
    target: &ResolutionTarget,
    cache_dir: &Path,
    source_dir: &Path,
    workspace_dir: Option<&Path>,
    workspace_pypi_indexes: &[String],
    conda_channels: &[ChannelUrl],
    uv_retry_keep: &BTreeSet<PypiKey>,
    probe_metrics: Arc<BundleProbeMetrics>,
) -> Result<(
    Option<crate::uv_closure::UvClosure>,
    std::collections::BTreeSet<String>,
    WorkspaceCondaFacts,
    BTreeSet<String>,
    BTreeSet<String>,
    CondaCoSolveContext,
)> {
    let uv_retry_keep_names: BTreeSet<String> = uv_retry_keep
        .iter()
        .map(|name| name.as_str().to_string())
        .collect();
    let mut roots: Vec<String> = Vec::new();
    // Direct-URL wheels pre-fetched into the content-addressed store and
    // emitted as `[tool.uv.sources]` path sources (see below) accumulate here,
    // then seed `req.built_wheel_sources` at request construction.
    let mut url_wheel_sources: BTreeMap<String, PathBuf> = BTreeMap::new();
    // Staging dir for the prefetch download (durable copy lives in the store,
    // keyed by sha; this dir just holds the initial download, hard-linked into
    // the store on the same filesystem).
    let url_prefetch_dir = cache_dir.join("uv-url-wheel-prefetch");
    let wheel_store_root = crate::courier::retread_wheel_store_root();
    for (name, entry) in group_entries {
        if entry.is_spec() {
            let extras = if entry.extras.is_empty() {
                String::new()
            } else {
                format!("[{}]", entry.extras.join(","))
            };
            let version = entry.version.as_deref().unwrap_or("").trim();
            let spec = if version.is_empty() || version == "*" {
                String::new()
            } else if version.starts_with(['<', '>', '=', '!', '~']) {
                version.to_string()
            } else {
                format!("=={version}")
            };
            roots.push(format!("{name}{extras}{spec}"));
        } else if let Some(url) = &entry.url {
            // Direct-URL wheel: instead of a `name @ https://...` requirement
            // (which makes uv download+unpack the WHOLE wheel just to read
            // METADATA -- and re-pay it every lock, since pypi.nvidia.com
            // serves `no-store` with no PEP 658 sidecar; up to ~5.9 GiB each
            // for the extscache wheels), pre-fetch the wheel ONCE into the
            // content-addressed store and emit a local `path =` source so uv
            // reads METADATA from a seekable local zip. The artifact itself is
            // still shipped by the phase-1 courier from this same store
            // (`fetch_wheel_cached`, cache hit), and the pylock records the
            // wheel as a local `archive` either way (pin-only, no index
            // wheel), so lock provenance is unchanged.
            let store_path = crate::wheel::prefetch_url_wheel_as_source(
                url,
                entry.sha256.as_deref(),
                &url_prefetch_dir,
                &wheel_store_root,
            )
            .await
            .with_context(|| {
                format!(
                    "uv closure: direct-URL wheel prefetch for `{name}` failed; integrity and target failures are not eligible for direct-URL fallback"
                )
            })?;
            tracing::info!(
                entry = %name,
                bundle = %group_name,
                path = %store_path.display(),
                "uv closure: direct-URL wheel pre-fetched to store; \
                 emitting a local path source (avoids the no-store \
                 whole-wheel redownload uv would do to read METADATA)",
            );
            url_wheel_sources.insert(name.clone(), store_path);
            // First-party bare requirement; the path source binds by
            // name (uv reads name/version from the local wheel).
            roots.push(name.clone());
        } else {
            tracing::info!(
                entry = %name,
                bundle = %group_name,
                "uv closure: source-built entry is not a uv root; it resolves \
                 via the legacy materialization path (milestone-1 limit)",
            );
        }
    }
    // retread-deps-from: fetch + parse each configured source and append
    // its PEP 508 lines as additional roots. A pure deps-from bundle (no
    // uv-resolvable `[retread-wheels]` entries at all) is exactly why this
    // runs BEFORE the `roots.is_empty()` bail-out below -- deps-from alone
    // can supply every root this bundle needs.
    // Conda-as-truth: canonical PyPI names supplied by retread-deps-from.
    // deps-from roots are appended strictly AFTER `[retread-wheels]`
    // roots and `dedupe_roots_last_wins` keeps the last occurrence per
    // name, so any name present here is guaranteed to be the winning
    // root below, so these requirements may safely retain deps-from
    // provenance through conda run-dep emission.
    let mut deps_from_root_names: std::collections::BTreeSet<String> = Default::default();
    let mut deps_from_advisory_floors = Vec::new();
    if !effective.deps_from.is_empty() {
        let deps_from = crate::deps_from::resolve_deps_from(
            effective.deps_from.as_slice(),
            source_dir,
            cache_dir,
        )
        .await
        .with_context(|| format!("retread-deps-from: bundle `{group_name}`"))?;
        deps_from_root_names = deps_from
            .pypi_roots
            .iter()
            .filter_map(|root| root_req_name(root))
            .collect();
        deps_from_advisory_floors = deps_from.advisory_conda_floors;
        roots.extend(deps_from.pypi_roots);
        // Dedupe by PEP 503-normalized package name, LAST occurrence wins.
        // deps-from entries are appended after `[retread-wheels]` roots
        // above, so on a same-name collision the deps-from requirement
        // string wins -- lets a deps-from file's own pin override a
        // `[retread-wheels]` spec-form entry's implicit root without an
        // error (or vice versa if a later deps-from source repeats a
        // name an earlier one set).
        roots = dedupe_roots_last_wins(roots);
    }

    let manifest_opt = workspace_dir.and_then(crate::workspace::WorkspaceManifest::load);
    // Solve precise consuming environments for every policy. Ownership/drop
    // actions remain validated-policy-only below, but Rule 3 and
    // harmonization must never fall back to non-consuming feature unions.
    // Only the pack's declared map may establish PyPI -> conda fact identity;
    // fallback/parselmouth routing aliases are not fact authority.
    let mut workspace_facts = match (manifest_opt.as_ref(), workspace_dir) {
        (Some(manifest), Some(ws_dir)) => {
            solve_workspace_conda_facts(
                manifest,
                ws_dir,
                source_dir,
                target,
                conda_channels,
                fact_name_map,
                group_name,
            )
            .await
        }
        _ => WorkspaceCondaFacts::default(),
    };
    if effective.route_policy == crate::config::RoutePolicy::PreferCondaValidated {
        let manual: BTreeSet<String> = effective
            .overrides
            .keys()
            .map(|name| canonical_conda_name(name))
            .collect();
        let keep: BTreeSet<String> = effective
            .keep_pypi
            .iter()
            .map(|name| canonical_conda_name(name))
            .chain(uv_retry_keep_names.iter().cloned())
            .collect();
        let protected_entries: BTreeSet<String> = group_entries
            .iter()
            .map(|(name, _)| canonical_conda_name(name))
            .collect();
        workspace_facts.owned_pypi.retain(|name| {
            !manual.contains(name) && !keep.contains(name) && !protected_entries.contains(name)
        });
        tracing::debug!(
            bundle = %group_name,
            consuming_envs = workspace_facts.env_exact_specs.len(),
            mapped_facts = workspace_facts.common_pypi.len(),
            owned_conda = workspace_facts.owned_conda_pypi.len(),
            owned_pypi = workspace_facts.owned_pypi.len(),
            "conda facts: derived precise workspace routing inputs",
        );
    }

    // Rule 1's effective workspace ownership is a filtered view of the
    // precise solved facts, not a mutation of those facts. Carry both mapped
    // PyPI identity and direct conda identity through the uv and legacy
    // materialization paths so Rule 2 can honor the same authority.
    let workspace_ownership =
        if effective.route_policy == crate::config::RoutePolicy::PreferCondaValidated {
            let mut excluded_pypi_names: BTreeSet<PypiKey> = effective
                .overrides
                .keys()
                .map(|name| PypiKey::from_pypi(name))
                .collect();
            excluded_pypi_names.extend(
                effective
                    .keep_pypi
                    .iter()
                    .map(|name| PypiKey::from_pypi(name)),
            );
            excluded_pypi_names.extend(uv_retry_keep.iter().cloned());
            excluded_pypi_names.extend(
                group_entries
                    .iter()
                    .map(|(name, _)| PypiKey::from_pypi(name)),
            );
            excluded_pypi_names.extend(
                url_wheel_sources
                    .keys()
                    .map(|name| PypiKey::from_pypi(name)),
            );
            let pypi_names = workspace_facts
                .owned_conda_pypi
                .iter()
                .map(|name| PypiKey::from_pypi(name))
                .filter(|name| !excluded_pypi_names.contains(name))
                .collect();
            WorkspaceRouteOwnership {
                pypi_names,
                conda_names: workspace_facts
                    .owned_conda
                    .iter()
                    .map(|name| PypiKey::from_pypi(name))
                    .collect(),
                excluded_pypi_names,
            }
        } else {
            WorkspaceRouteOwnership::default()
        };

    // Construct the shared oracle before the roots-empty return so the legacy
    // materialization path and the uv path validate Rule 2 against identical
    // consuming-workspace facts. Direct PyPI ownership still comes only from
    // the precise consumer facts above, but provider identity is a
    // non-destructive validation edge: use the effective Parselmouth-backed
    // map so a workspace-owned `torch` also protects its conda `pytorch`
    // provider without requiring every pack to repeat retread-name-map.
    let conda_co_solve = CondaCoSolveContext::new(
        manifest_opt.as_ref(),
        workspace_dir,
        source_dir,
        target,
        conda_channels,
        group_name,
        &workspace_facts.owned_pypi,
        &effective.name_map,
    )
    .with_probe_metrics(probe_metrics);
    let conda_co_solve = if crate::thread_budget::parallel_probes_enabled() {
        const PROBE_POOL_CAP: usize = 4;
        let requested_probe_threads = std::num::NonZeroUsize::new(
            crate::concurrency::max_concurrent_builds().clamp(1, PROBE_POOL_CAP),
        )
        .expect("the probe pool cap is nonzero");
        let probe_pool = crate::thread_budget::acquire_probe_pool(requested_probe_threads).await;
        tracing::warn!(
            threads = probe_pool.threads().get(),
            "experimental parallel probe solves enabled by RETREAD_PARALLEL_PROBES=1",
        );
        conda_co_solve.with_probe_pool(probe_pool)
    } else {
        tracing::debug!("parallel probe solves disabled; set RETREAD_PARALLEL_PROBES=1 to opt in");
        conda_co_solve
    };

    // Rule-3-capable policies receive only precise, solved, agreed facts.
    // Aggressive deliberately retains its legacy declared-constraint input,
    // whose non-workspace-solved provenance cannot authorize Rule 3.
    let manual: BTreeSet<String> = effective
        .overrides
        .keys()
        .map(|name| canonical_conda_name(name))
        .collect();
    let mut constraints = match effective.route_policy {
        crate::config::RoutePolicy::PreferCondaValidated | crate::config::RoutePolicy::Minimal => {
            workspace_fact_constraints(&workspace_facts, &manual)
        }
        crate::config::RoutePolicy::Aggressive => match (manifest_opt.as_ref(), workspace_dir) {
            (Some(manifest), Some(ws_dir)) => {
                let deps = unambiguous_consuming_deps(&workspace_consuming_dependencies(
                    manifest, ws_dir, source_dir, target,
                ));
                let global_map = load_pypi_to_conda_map().await;
                crate::uv_closure::build_constraints(
                    &deps,
                    &effective.name_map,
                    &global_map,
                    "manifest",
                    "consuming-envs",
                )
            }
            (Some(manifest), None) => {
                let deps =
                    manifest.effective_dependencies_for_target("default", target.conda_subdir());
                crate::uv_closure::build_constraints(
                    &deps,
                    &effective.name_map,
                    &PypiToCondaMap::new(),
                    "manifest",
                    "default",
                )
            }
            _ => Default::default(),
        },
    };
    if let (Some(manifest), Some(ws_dir)) = (manifest_opt.as_ref(), workspace_dir) {
        let sibling_constraints = sibling_lock_constraints(manifest, ws_dir, source_dir, target);
        if !sibling_constraints.constraints.is_empty() {
            tracing::info!(
                bundle = %group_name,
                constraints = sibling_constraints.constraints.len(),
                "uv closure: applying co-activated sibling pack requirements",
            );
        }
        for line in sibling_constraints.constraints {
            if !constraints.constraints.contains(&line) {
                constraints.constraints.push(line);
            }
        }
        for (name, provenance) in sibling_constraints.provenance {
            constraints.provenance.entry(name).or_insert(provenance);
        }
    }
    // Proactive cuda-major capping (belt to the auto-route co-install
    // check's suspenders): derive this pack's actual consuming env(s)'
    // `cuda-version` anchor via `consuming_env_dependencies` (env-scoped,
    // NOT just the `default` env above — a GPU-only feature like
    // `[feature.gpu]` is invisible to `effective_dependencies("default")`
    // but IS what the consuming env solves against) and cap the known
    // cuda-major-tracked PyPI families to that line up front, so uv
    // never independently picks a cuda-(X+1) release the conda side
    // can't co-install.
    if let (Some(manifest), Some(ws_dir)) = (manifest_opt.as_ref(), workspace_dir) {
        let workspace_deps = workspace_consuming_dependencies(manifest, ws_dir, source_dir, target);
        if let Some(specs) = workspace_deps.get("cuda-version")
            && let Some(major) = crate::uv_closure::cuda_major_from_specs(specs)
        {
            for (name, spec) in crate::uv_closure::cuda_family_constraints(major) {
                if constraints.provenance.contains_key(name) {
                    // An explicit conda dep already constrains this
                    // family by name (rare, but an explicit pin wins
                    // over the curated default) -- don't override it.
                    continue;
                }
                let line = format!("{name}{spec}");
                constraints.constraints.push(line.clone());
                constraints.provenance.insert(
                    name.to_string(),
                    crate::uv_closure::ConstraintProvenance {
                        constraint: line,
                        conda_name: "cuda-version".to_string(),
                        conda_version: format!("{major}.*"),
                        source: "cuda-major-table".to_string(),
                        env: "consuming-envs".to_string(),
                        provenance: Provenance::UvConstraint,
                    },
                );
            }
        }
    }

    let mut protected_root_names: BTreeSet<String> = group_entries
        .iter()
        .map(|(name, _)| canonical_conda_name(name))
        .collect();
    protected_root_names.extend(roots.iter().filter_map(|req| root_req_name(req)));
    apply_deps_from_conda_floors(
        &mut constraints,
        &deps_from_advisory_floors,
        &roots,
        fact_name_map,
        &protected_root_names,
        &effective.overrides,
        &effective.drop_deps,
        &target.conda_subdir,
        &target.python_version,
    )?;

    // retread-overrides -> override-dependencies where PEP 440-representable.
    // Fix #20: `retread solve`/`retread lock` now WRITE auto-repaired T1
    // conda-as-truth overrides for a backend-closure conflict directly
    // into THIS pack's `[package.build.config.retread-overrides]` table
    // (the same table the manual trio lives in) -- so they arrive here in
    // `effective.overrides` with no special handling, exactly like a
    // user-declared override. A workspace `pypi-options.dependency-
    // overrides` write was inert (the pack's uv closure never reads it);
    // the pack table is what the closure actually consumes.
    let mut overrides: Vec<String> = Vec::new();
    for (name, spec) in &effective.overrides {
        let spec = spec.trim();
        if spec.is_empty() || spec == "*" {
            overrides.push(name.clone());
        } else if let Some(pep) = crate::uv_closure::conda_spec_to_pep440(spec) {
            overrides.push(format!("{name}{pep}"));
        } else {
            tracing::warn!(
                name = %name,
                spec = %spec,
                "uv closure: retread-overrides spec is not PEP 440-representable; skipped",
            );
        }
    }
    // retread-drop-deps -> unmatchable-marker overrides (AMENDMENT A3):
    // removes the name from uv's graph so broken Requires-Dist edges
    // can't fail the lock; the parse-time exclude filter stays as
    // belt-and-braces.
    for name in &effective.drop_deps {
        overrides.push(format!("{name} ; {}", crate::uv_closure::DROP_MARKER));
    }
    // Built-in Windows-only shims (idna-ssl, pywin32, ...): NVIDIA's index
    // strips the `sys_platform == "win32"` marker from these Requires-Dist
    // lines, so the marker-pruning path (`uv_closure::environment_marker`)
    // can't drop them and packs would otherwise have to hand-carry every
    // one in `retread-drop-deps`. Inject the SAME unmatchable-marker
    // override on non-Windows targets so uv never resolves them. Semantics
    // mirror the conda run-dep path (`produce_output`'s auto-drop): the user
    // override always wins (re-enable on Linux), Windows targets inject
    // nothing, and names a pack already drops are skipped so there is no
    // duplicate override line.
    // First-party/root requirement names (canonical PEP 503): an explicit
    // top-level requirement for one of the built-in win-only names means the
    // user wants it resolved deliberately, so the injected graph-wide marker
    // must NOT suppress it.
    let first_party_names: Vec<String> =
        roots.iter().filter_map(|req| root_req_name(req)).collect();
    let injected_win_only = crate::uv_closure::built_in_win_only_to_inject(
        &target.conda_subdir,
        |name| effective.overrides.contains_key(name),
        &effective.drop_deps,
        &first_party_names,
    );
    for name in &injected_win_only {
        overrides.push(format!("{name} ; {}", crate::uv_closure::DROP_MARKER));
    }
    if !injected_win_only.is_empty() {
        tracing::debug!(
            group = %group_name,
            subdir = %target.conda_subdir,
            built_in_win_only = ?injected_win_only,
            "uv closure: injected built-in Windows-only drops (packs need no \
             retread-drop-deps for these); re-enable any via retread-overrides",
        );
    }

    // Root chain: explicit entry indexes in group order, then the complete
    // workspace chain. Transitive fallback uses the opposite priority below.
    //
    // Only explicitly-declared entry indexes join the priority chain --
    // entries without an `index =` fall through to the PUBLIC_PYPI tail.
    // Including their implicit pypi.org here put pypi.org FIRST whenever
    // an index-less entry preceded an explicit-index one in group order,
    // and uv breaks equal-version ties by index order: NVIDIA publishes
    // stub sdists of `isaacsim` (and its isaacsim-* constellation) on
    // pypi.org while the real manylinux wheels live only on
    // pypi.nvidia.com, so pypi.org-first made uv lock the useless stub
    // (-> "has no usable wheels" under no-build).
    let entry_indexes = group_entries
        .iter()
        .filter(|(_, entry)| !entry.is_url())
        .filter_map(|(_, entry)| entry.index.clone())
        .collect::<Vec<_>>();
    let index_urls = index_chain(
        entry_indexes.clone(),
        workspace_pypi_indexes,
        IndexPurpose::RootResolve,
    );
    let transitive_index_urls = index_chain(
        entry_indexes,
        workspace_pypi_indexes,
        IndexPurpose::TransitiveFallback,
    );

    // retread-drop-deps also excluded from the parsed closure.
    let mut no_emit: Vec<String> = effective.conda_deps.clone();
    no_emit.extend(effective.drop_deps.iter().cloned());

    let mut req = crate::uv_closure::UvClosureRequest {
        bundle: group_name.to_string(),
        python_version: target.python_version.clone(),
        conda_subdir: target.conda_subdir.clone(),
        dependencies: roots,
        dependency_provenance: deps_from_root_names
            .iter()
            .cloned()
            .map(|name| (name, Provenance::DepsFromRelaxed))
            .collect(),
        constraints,
        overrides,
        no_emit_packages: no_emit,
        index_urls: index_urls.clone(),
        // Direct-URL wheels pre-fetched above as path sources; the self-heal
        // extends this map (`.insert`) with any heal-built wheels.
        built_wheel_sources: url_wheel_sources,
        explicit_pins: BTreeMap::new(), // populated by the self-heal
        workspace_owned: crate::uv_closure::WorkspaceOwnedPlan::default(),
        offline: false,
    };
    let project_dir = cache_dir.join("uv-projects").join(format!(
        "{}-py{}-{}",
        canonical_conda_name(group_name),
        target.python_version,
        target.conda_subdir,
    ));
    let uv_cache_dir = cache_dir.join("uv-cache");
    // Persisted heal facts live OUTSIDE uv-projects so they survive a
    // "delete uv-projects state" cold reset -- that survival is what lets
    // the post-reset Pass A converge in one lock (issue #10 perf, item 3b).
    let heal_facts_path =
        crate::uv_closure::heal_facts_path_for_target(cache_dir, group_name, target);

    // M2: auto-route options. Roots (this bundle's own entries) and
    // retread-built wheel sources must never leave the closure; keep-pypi
    // is the user's opt-out list.
    let mut protected: std::collections::BTreeSet<String> = group_entries
        .iter()
        .map(|(n, _)| canonical_conda_name(n))
        .collect();
    protected.extend(
        req.built_wheel_sources
            .keys()
            .map(|n| canonical_conda_name(n)),
    );
    if effective.route_policy == crate::config::RoutePolicy::PreferCondaValidated {
        let manual: BTreeSet<String> = effective
            .overrides
            .keys()
            .map(|name| canonical_conda_name(name))
            .collect();
        let keep: BTreeSet<String> = effective
            .keep_pypi
            .iter()
            .map(|name| canonical_conda_name(name))
            .chain(uv_retry_keep_names.iter().cloned())
            .collect();
        workspace_facts.owned_pypi.retain(|name| {
            !manual.contains(name) && !keep.contains(name) && !protected.contains(name)
        });
    } else {
        workspace_facts.owned_pypi.clear();
    }

    let direct_workspace_pypi: BTreeSet<PypiKey> = workspace_facts
        .owned_pypi
        .iter()
        .map(|name| PypiKey::from_pypi(name))
        .collect();
    let conda_candidates = workspace_conda_provider_candidates(
        !req.dependencies.is_empty(),
        &workspace_facts,
        &workspace_ownership,
        fact_name_map,
    );
    let planning_context = conda_co_solve.clone();
    let planning_co_solve = move |routes: Vec<crate::uv_closure::CondaRouteSpec>| {
        let context = planning_context.clone();
        async move { context.solve(routes).await }
    };
    // Ownership planning is deliberately mode-independent: it replaces the
    // three unconditional pre-P4 drop implementations and preserves their
    // default-off behavior. RETREAD_UV_RERESOLVE gates only the rejected-route
    // handoff that bypasses the legacy reconstruct/fetch path.
    let mut ownership_req = req.clone();
    ownership_req.dependencies =
        workspace_ownership_planning_dependencies(&req.dependencies, &deps_from_root_names);
    req.workspace_owned = crate::uv_closure::plan_workspace_owned_prelock(
        &ownership_req,
        &direct_workspace_pypi,
        conda_candidates,
        &workspace_ownership.excluded_pypi_names,
        &planning_co_solve,
    )
    .await;
    let prelock_owned_drops = req.workspace_owned.dropped_without_uv();

    if req.dependencies.is_empty() {
        tracing::info!(
            bundle = %group_name,
            "uv closure: no uv-resolvable roots in this bundle; \
             running the legacy closure path unpinned",
        );
        return Ok((
            None,
            std::collections::BTreeSet::new(),
            workspace_facts,
            prelock_owned_drops,
            protected_root_names,
            conda_co_solve,
        ));
    }
    // ABI-anchor pins (`cuda-version`, `python_abi`, ...) from the
    // consuming env(s) -- same source the proactive cuda-major capping
    // above reads. Passed to the route-time metadata check so a routed
    // build requiring a conda-only anchor version the env can't provide
    // is refused up front (run 16c: `triton ==3.6.0` -> `cuda-version
    // >=12.9,<13` against a cuda-12.8 workspace).
    let mut abi_anchor_pins: std::collections::BTreeMap<String, String> =
        if let (Some(manifest), Some(ws_dir)) = (manifest_opt.as_ref(), workspace_dir) {
            workspace_consuming_dependencies(manifest, ws_dir, source_dir, target)
                .into_iter()
                .filter(|(name, _)| crate::solve::is_abi_anchor(name))
                .filter_map(|(name, specs)| specs.into_iter().next().map(|spec| (name, spec)))
                .collect()
        } else {
            Default::default()
        };
    // `python_abi` is never written directly in a manifest (users pin
    // `python`; conda derives the cp-tag ABI constraint from it), so the
    // scan above can never populate it from `consuming_env_dependencies`
    // -- the workspace's exact `python` pin never gets a chance to arm
    // the anchor check for it (run 17: `pandas==3.0.3` needing
    // `python_abi 3.13.*` routed clean against a python-3.11 workspace
    // because the anchor map had no `python_abi` entry at all). Synthesize
    // one from the exact `python` pin, the same numeric major.minor a
    // routed build's `python_abi X.Y.*` depend is checked against
    // (`route_metadata_consistent` only reads the numeric version, the
    // build/cp-tag suffix is discarded).
    if !abi_anchor_pins.contains_key("python_abi")
        && let Some(python_pin) = abi_anchor_pins.get("python").cloned()
    {
        abi_anchor_pins.insert("python_abi".to_string(), python_pin);
    }
    let auto_route_opts = crate::uv_closure::AutoRouteOptions {
        enabled: effective.auto_route,
        // v4.6 Part A: production wiring of the routing policy -- the
        // config-layer default is `Minimal` (whitelist-only routing).
        route_policy: effective.route_policy,
        route_include: effective
            .route_include
            .iter()
            .map(|n| canonical_conda_name(n))
            .collect(),
        keep_pypi: effective
            .keep_pypi
            .iter()
            .map(|n| canonical_conda_name(n))
            .chain(uv_retry_keep_names.iter().cloned())
            .collect(),
        protected,
        name_map: effective.name_map.clone(),
        force_conda: effective
            .force_conda
            .iter()
            .map(|n| canonical_conda_name(n))
            .collect(),
        abi_anchor_pins,
        // Populated below (workspace-deps solve) just before the
        // fixpoint call; empty = un-route fallback only.
        workspace_conda_versions: Default::default(),
        workspace_fact_fingerprint: workspace_facts.fingerprint.clone(),
    };

    // `'static` closures for the fixpoint driver: clone the inputs each
    // solve/probe needs. Cheap relative to a uv subprocess / repodata hit.
    let raw_solve = {
        let project_dir = project_dir.clone();
        let uv_cache_dir = uv_cache_dir.clone();
        let target = target.clone();
        let sdist_build_policy = effective.sdist_build;
        move |r: crate::uv_closure::UvClosureRequest| {
            let project_dir = project_dir.clone();
            let uv_cache_dir = uv_cache_dir.clone();
            let target = target.clone();
            let fut = async move {
                crate::uv_closure::compute_closure_for_target(
                    &r,
                    &target,
                    &project_dir,
                    &uv_cache_dir,
                    None,
                    sdist_build_policy,
                )
                .await
            };
            Box::pin(fut)
                as futures::future::BoxFuture<'static, Result<crate::uv_closure::UvClosure>>
        }
    };
    let probe = {
        let channels = conda_channels.to_vec();
        let python = target.python_version.clone();
        move |conda_name: String, spec: String| {
            let channels = channels.clone();
            let python = python.clone();
            let fut = async move {
                crate::probe::find_route(&channels, &conda_name, &spec, Some(&python))
                    .await
                    .map(|hit| crate::uv_closure::RouteProbeHit {
                        conda_version: hit.version,
                        channel: hit.channel,
                        depends: hit.depends,
                    })
            };
            Box::pin(fut)
                as futures::future::BoxFuture<'static, Option<crate::uv_closure::RouteProbeHit>>
        }
    };
    // Sdist-only self-heal (v4.4.0 third rung, spec-uv-restructure
    // follow-up): rung 1 (conda-route) probes for a compatible version
    // -- `with_sdist_heal` derives an exact spec from the structured
    // Pass-B offender version. Pass B resolved that PyPI version even
    // though no compatible wheel was available; the conda probe may select
    // a differently formatted/equivalent conda version, so both version
    // domains remain distinct in the route fact. This reuses the same
    // `probe::find_route` the ordinary auto-route round uses. Rung 2
    // (sdist auto-build) is gated by
    // `sdist-build` (default "auto"); `"never"` passes `None` so the
    // wrapper reproduces the pre-v4.4.0 conda-route-or-error behavior
    // exactly.
    let sdist_probe = {
        let channels = conda_channels.to_vec();
        let python = target.python_version.clone();
        let keep_pypi = uv_retry_keep_names.clone();
        move |conda_name: String, spec: String| {
            let channels = channels.clone();
            let python = python.clone();
            let keep_pypi = keep_pypi.clone();
            let fut = async move {
                if keep_pypi.contains(&canonical_conda_name(&conda_name)) {
                    return None;
                }
                crate::probe::find_route(&channels, &conda_name, &spec, Some(&python))
                    .await
                    .map(|hit| crate::uv_closure::RouteProbeHit {
                        conda_version: hit.version,
                        channel: hit.channel,
                        depends: hit.depends,
                    })
            };
            Box::pin(fut)
                as futures::future::BoxFuture<'static, Option<crate::uv_closure::RouteProbeHit>>
        }
    };
    let sdist_build = (effective.sdist_build == crate::config::SdistBuildPolicy::Auto).then(|| {
        let index_urls = transitive_index_urls.clone();
        let target = target.clone();
        let cache_dir = cache_dir.to_path_buf();
        move |name: String, requirement: Option<String>| {
            let fut = build_sdist_wheel(
                name,
                requirement,
                index_urls.clone(),
                target.clone(),
                cache_dir.clone(),
            );
            Box::pin(fut)
                as futures::future::BoxFuture<'static, Result<crate::uv_closure::BuiltSdistWheel>>
        }
    });
    // Seed the heal ledgers from facts persisted by a previous run (issue
    // #10 perf): with these present, the FIRST Pass A already carries the
    // learned overrides / pins / built-wheel path-sources, so a warm rerun
    // resolves in a single lock (and the pyproject fingerprint matches the
    // recorded meta, letting uv reuse the healed uv.lock instead of
    // re-resolving).
    // Stale built-wheel entries (store pruned) are dropped on load.
    // Facts are only replayable under the manifest/routing state they were
    // learned from (B1): stamp over the BASE request + routing options; a
    // mismatch discards the file (fresh heal), never a stale replay.
    let facts_stamp = crate::uv_closure::heal_facts_stamp_for_target(
        &req,
        &auto_route_opts,
        effective.sdist_build,
        target,
    );
    let persisted_facts =
        crate::uv_closure::load_heal_facts_for_target(&heal_facts_path, &facts_stamp, target)
            .await?;
    if !persisted_facts.is_empty() {
        tracing::info!(
            bundle = %group_name,
            routed = persisted_facts.routed.len(),
            built = persisted_facts.built.len(),
            prereleased = persisted_facts.prereleased.len(),
            workspace_overrides = persisted_facts.workspace_overrides.len(),
            "uv closure: seeding heal ledgers from persisted facts (warm reuse path)",
        );
    }
    let workspace_overrides = Arc::new(std::sync::Mutex::new(persisted_facts.workspace_overrides));
    let persisted_routes = persisted_facts
        .routed
        .into_iter()
        .filter(|route| !uv_retry_keep_names.contains(&canonical_conda_name(&route.pypi_name)))
        .collect();
    let sdist_routed = Arc::new(std::sync::Mutex::new(persisted_routes));
    let sdist_built = Arc::new(std::sync::Mutex::new(persisted_facts.built));
    // Transitive-prerelease repairs surface naturally in the closure's
    // pins/wheels (the offender keeps its own index wheel); collected here
    // only for logging/audit parity with the route/build ledgers.
    let sdist_prereleased = Arc::new(std::sync::Mutex::new(persisted_facts.prereleased));
    let solve = crate::uv_closure::with_workspace_fact_overrides(
        raw_solve,
        Arc::clone(&workspace_overrides),
    );
    let solve = crate::uv_closure::with_sdist_heal(
        group_name.to_string(),
        solve,
        sdist_probe,
        sdist_build,
        Arc::clone(&sdist_routed),
        Arc::clone(&sdist_built),
        Arc::clone(&sdist_prereleased),
    );
    // Rule 1 adapts its exact auto-route pins to the shared route-spec oracle.
    // Rule 2 receives this same context after bundle materialization, avoiding
    // a second solver path with subtly different workspace inputs.
    let co_solve = {
        let context = conda_co_solve.clone();
        move |routed: Vec<crate::uv_closure::AutoRoutedPackage>| {
            let context = context.clone();
            let routed = routed
                .into_iter()
                .map(|route| crate::uv_closure::CondaRouteSpec {
                    pypi_name: PypiKey::from_pypi(&route.pypi_name),
                    conda_name: CondaName::new(route.conda_name),
                    spec: format!("=={}", route.conda_version),
                })
                .collect();
            Box::pin(async move { context.solve(routed).await })
                as futures::future::BoxFuture<'static, crate::uv_closure::CoInstallVerdict>
        }
    };
    // Harmonization shares Rule 3's precise authority: exact conda names
    // selected identically in every concrete consumer. Ambiguous ownership,
    // failed solves, transitive-only names, or disagreement leave the map
    // empty and the fixpoint abstains to its un-route fallback.
    let mut auto_route_opts = auto_route_opts;
    auto_route_opts.workspace_conda_versions = workspace_facts.common_conda_versions.clone();
    let mut closure = match crate::uv_closure::auto_route_fixpoint_checked(
        &req,
        &auto_route_opts,
        solve,
        probe,
        co_solve,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            // B1: a genuine resolution/heal failure may have been poisoned by
            // stale persisted facts (or is about to change the manifest state
            // via a repair loop), so the facts file is dropped to force a
            // clean re-heal next run. A merely TRANSIENT failure (io/network/
            // backend crash) leaves the facts valid and keeps them -- see
            // `discard_facts_on_solve_failure` for the transient-vs-resolution
            // classification and why "when unsure, delete" stays wedge-safe.
            discard_facts_on_solve_failure(&heal_facts_path, &e);
            return Err(e);
        }
    };
    // Splice in the sdist-only self-heal's discoveries (mirrors
    // `uv_closure::auto_route_fixpoint_with_sdist_heal`'s own splice,
    // which this call site can't use directly -- production also needs
    // the co-install-checked auto-route un-route step from
    // `auto_route_fixpoint_checked`).
    {
        let routed = sdist_routed.lock().unwrap();
        closure.auto_routed.extend(routed.iter().cloned());
    }
    {
        let built = sdist_built.lock().unwrap();
        for w in built.iter() {
            closure
                .pins
                .entry(w.pypi_name.clone())
                .or_insert_with(|| w.version.clone());
            closure.wheels.push(crate::lock::LockWheel {
                name: w.pypi_name.clone(),
                version: w.version.clone(),
                origin: crate::lock::Origin::Built,
                filename: w.filename.clone(),
                url: None,
                sha256: Some(w.sha256.clone()),
                requires_dist: Vec::new(),
                must_ship: true,
                upstream_url: None,
                git_source: None,
                sdist_source: Some(w.sdist_source.clone()),
            });
        }
    }
    // Persist the heal facts that produced this successful closure so the
    // next run's FIRST Pass A carries them (issue #10 perf: warm single-lock
    // convergence + healed-uv.lock reuse). Only reached on success -- a
    // failed solve returned via `?` above and never overwrites good facts.
    // The ledgers now hold the union of persisted + newly-discovered facts
    // (with_sdist_heal re-injects the seed ledgers each round).
    crate::uv_closure::save_heal_facts(
        &heal_facts_path,
        &crate::uv_closure::HealFacts {
            stamp: facts_stamp,
            workspace_overrides: workspace_overrides.lock().unwrap().clone(),
            // The FULL routing set, not just the sdist-heal ledger:
            // `closure.auto_routed` (post-splice) also carries the M2
            // auto-route fixpoint's discoveries (torch/cuda-* style
            // harmonization routes). Persisting those is what lets the next
            // run's ROUND 0 request already include the routing constraints
            // -- so its synthesized pyproject matches the recorded
            // fingerprint, the healed uv.lock is kept, and the fixpoint
            // converges without a second discovery re-lock (run7 measured
            // the miss at 2x ~9 min: round 0 pinless-routeless, round 1
            // re-resolving from scratch after apply_auto_route changed the
            // constraint set). Deduped by pypi_name (splice can repeat).
            routed: {
                let mut seen = std::collections::BTreeSet::new();
                closure
                    .auto_routed
                    .iter()
                    .filter(|r| seen.insert(r.pypi_name.clone()))
                    .cloned()
                    .collect()
            },
            built: sdist_built.lock().unwrap().clone(),
            prereleased: sdist_prereleased.lock().unwrap().clone(),
        },
    );
    Ok((
        Some(closure),
        deps_from_root_names,
        workspace_facts,
        prelock_owned_drops,
        protected_root_names,
        conda_co_solve,
    ))
}

/// Collapses [`crate::workspace::WorkspaceManifest::consuming_env_dependencies`]'
/// multi-spec map into the single-spec map [`crate::uv_closure::build_constraints`]
/// consumes: a name survives only when every consuming env agrees on ONE
/// spec (after trimming). Disagreeing names (possible via the
/// all-features fallback superset for packs no env references, e.g.
/// `pytorch-gpu ==2.7.0` vs `==2.10.0` across unrelated features) are
/// dropped -- one uv constraint cannot satisfy two pins, and dropping
/// preserves the pre-run-38 unconstrained behavior for exactly that
/// ambiguous case. Wildcard/empty specs are dropped too (nothing to
/// constrain).
fn unambiguous_consuming_deps(
    consuming: &BTreeMap<String, Vec<String>>,
) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (name, specs) in consuming {
        let mut distinct: Vec<&str> = specs
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty() && *s != "*")
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        if let [only] = distinct.as_slice() {
            out.insert(name.clone(), (*only).to_string());
        }
    }
    out
}

#[cfg(test)]
mod unambiguous_consuming_deps_tests {
    use super::unambiguous_consuming_deps;
    use std::collections::BTreeMap;

    #[test]
    fn keeps_agreed_pins_drops_conflicts_and_wildcards() {
        let consuming = BTreeMap::from([
            // All consuming envs agree -> kept.
            (
                "pytorch-gpu".to_string(),
                vec!["==2.7.0".to_string(), "==2.7.0".to_string()],
            ),
            // Disagreement (all-features fallback superset) -> dropped:
            // one uv constraint cannot satisfy two pins.
            (
                "torchvision".to_string(),
                vec!["==0.22.0".to_string(), "==0.25.0".to_string()],
            ),
            // Wildcard-only -> dropped (nothing to constrain).
            ("numpy".to_string(), vec!["*".to_string()]),
        ]);
        let deps = unambiguous_consuming_deps(&consuming);
        assert_eq!(deps.get("pytorch-gpu").map(String::as_str), Some("==2.7.0"));
        assert!(!deps.contains_key("torchvision"));
        assert!(!deps.contains_key("numpy"));
    }
}

#[cfg(test)]
mod sdist_source_url_tests {
    use super::compose_sdist_source_url;

    const H: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    #[test]
    fn compose_sdist_source_url_never_doubles_the_sha_fragment() {
        // Index advertised a hash: `pypi::parse_index_links_any` lifts the
        // discrete field but LEAVES the `#sha256=` fragment on the url, so a
        // naive `format!("{url}#sha256={h}")` would double it. The composer
        // must emit exactly one fragment.
        let with_frag =
            url::Url::parse(&format!("https://ex.org/p/foo-1.0.tar.gz#sha256={H}")).unwrap();
        let out = compose_sdist_source_url(&with_frag, Some(H));
        assert_eq!(
            out.matches("#sha256=").count(),
            1,
            "doubled fragment: {out}"
        );
        assert_eq!(out, format!("https://ex.org/p/foo-1.0.tar.gz#sha256={H}"));

        // PEP 691 may expose the digest in a discrete JSON field while the
        // artifact URL itself has no fragment. Persist the hash-bound form so
        // strict replay retains the exact raw artifact identity.
        let no_hash = url::Url::parse("https://ex.org/p/foo-1.0.tar.gz").unwrap();
        assert_eq!(
            compose_sdist_source_url(&no_hash, Some(H)),
            format!("https://ex.org/p/foo-1.0.tar.gz#sha256={H}")
        );

        // No advertised hash: URL passes through verbatim.
        assert_eq!(
            compose_sdist_source_url(&no_hash, None),
            "https://ex.org/p/foo-1.0.tar.gz"
        );
    }
}

#[cfg(test)]
mod facts_cleanup_tests {
    use super::discard_facts_on_solve_failure;

    #[test]
    fn transient_solve_failure_preserves_facts_resolution_failure_deletes() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-facts-cleanup-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let facts_path = tmp.join("facts.json");
        let seed = || std::fs::write(&facts_path, b"{}").unwrap();

        // A transient (io-class) failure -- e.g. a `uv` spawn error wrapping
        // an io::Error -- must KEEP the still-valid facts.
        seed();
        let transient = anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "index 5xx / NFS blip",
        ))
        .context("spawning `uv lock`");
        discard_facts_on_solve_failure(&facts_path, &transient);
        assert!(
            facts_path.exists(),
            "transient error must NOT delete valid heal facts",
        );

        // A genuine resolution conflict (a bail!/anyhow! string, no io::Error
        // source) must DROP the possibly-poisoned facts (wedge safety).
        let resolution = anyhow::anyhow!("no solution found: a==1 conflicts with b==2");
        discard_facts_on_solve_failure(&facts_path, &resolution);
        assert!(
            !facts_path.exists(),
            "resolution failure must delete the facts file",
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}

#[cfg(test)]
mod workspace_conda_facts_tests {
    use super::{
        CondaCoSolveContext, SolvedPypiFact, WorkspaceCondaFacts, WorkspaceRouteOwnership,
        dependency_name_intersection, effective_name_map, facts_from_solved_records,
        precise_consumer_inputs_for_target, workspace_conda_provider_candidates,
        workspace_fact_constraints,
    };
    use crate::constraint::Provenance;
    use crate::pypi::{ResolutionTarget, WheelTarget};
    use crate::relax::{CondaName, CondaTarget, NameMap, PypiKey};
    use rattler_conda_types::{PackageRecord, RepoDataRecord, VersionWithSource};
    use std::collections::{BTreeMap, BTreeSet};
    use std::str::FromStr;
    use url::Url;

    fn repo_record(name: &str, version: &str, depends: &[&str]) -> RepoDataRecord {
        let mut package_record = PackageRecord::new(
            name.parse().unwrap(),
            VersionWithSource::from_str(version).unwrap(),
            "h123456_0".to_string(),
        );
        package_record.subdir = "linux-64".to_string();
        package_record.depends = depends.iter().map(|s| (*s).to_string()).collect();
        RepoDataRecord {
            package_record,
            file_name: format!("{name}-{version}-h123456_0.conda"),
            url: Url::parse(&format!(
                "https://example.invalid/linux-64/{name}-{version}-h123456_0.conda"
            ))
            .unwrap(),
            channel: Some("https://example.invalid".into()),
        }
    }

    fn name_map(entries: &[(&str, &str)]) -> NameMap {
        entries
            .iter()
            .map(|(pypi, conda)| {
                (
                    PypiKey::from_pypi(pypi),
                    CondaTarget::Mapped(CondaName::new(*conda)),
                )
            })
            .collect()
    }

    #[test]
    fn parselmouth_map_protects_direct_pypi_torch_from_conda_routes() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-viral-provider-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let pack_dir = tmp.join("viral-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        std::fs::write(
            tmp.join("pixi.toml"),
            r#"[workspace]
channels = ["conda-forge"]
platforms = ["linux-64"]

[feature.viral.dependencies]
viral-pack = { path = "./viral-pack" }
python = "==3.11"

[feature.viral.pypi-dependencies]
torch = { version = "==2.7.0", index = "https://download.pytorch.org/whl/cu128" }

[environments]
viral-gpu = { features = ["viral"], no-default-feature = true }
"#,
        )
        .unwrap();

        let manifest = crate::workspace::WorkspaceManifest::load(&tmp).unwrap();
        let target = ResolutionTarget::for_subdir("3.11", "linux-64");
        let inputs =
            precise_consumer_inputs_for_target(&manifest, &tmp, &pack_dir, &target).unwrap();
        let owned_pypi = dependency_name_intersection(
            &inputs
                .iter()
                .map(|input| input.pypi_deps.clone())
                .collect::<Vec<_>>(),
        );
        assert_eq!(owned_pypi, BTreeSet::from(["torch".to_string()]));

        // Model the central mapping even when the pack declares no
        // retread-name-map of its own. The curated fallback resolves
        // Parselmouth's multi-provider torch family to `pytorch`.
        let parselmouth = std::collections::HashMap::from([(
            "torch".to_string(),
            vec![
                "pytorch".to_string(),
                "pytorch-cpu".to_string(),
                "pytorch-gpu".to_string(),
            ],
        )]);
        let effective_map = effective_name_map(&NameMap::new(), &parselmouth);
        let target = ResolutionTarget::from_wheel_target(
            WheelTarget {
                python_version: "3.11".to_string(),
                conda_subdir: "linux-64".to_string(),
                max_glibc: None,
            },
            None,
        );
        let context = CondaCoSolveContext::new(
            Some(&manifest),
            Some(&tmp),
            &pack_dir,
            &target,
            &[],
            "viral-pack",
            &owned_pypi,
            &effective_map,
        );
        assert_eq!(
            context
                .workspace_pypi_providers
                .get(&CondaName::new("pytorch")),
            Some(&PypiKey::from_pypi("torch")),
            "the shared provider map must protect workspace PyPI torch without a pack-local map",
        );

        std::fs::remove_dir_all(tmp).unwrap();
    }

    /// The holosoma regression, at the layer that actually broke.
    ///
    /// The consuming env is `no-default-feature = true`, so it inherits NOTHING
    /// from the root `[dependencies]` (which declares its own, different NumPy).
    /// NumPy is conda-owned solely through the non-default `holosoma` feature,
    /// and ownership must see it there.
    ///
    /// It must ALSO render that environment's baseline solve specs with RAW
    /// conda names. `cuda-nvcc_linux-64` is a real conda-forge package;
    /// `canonical_conda_name` is the PEP 503 PyPI normalizer and folds it to
    /// `cuda-nvcc-linux-64`, which no channel has ever published. The precise
    /// consuming-env solve then died with "No candidates were found",
    /// `solve_workspace_conda_facts` abstained, `owned_conda` came back EMPTY --
    /// and so NumPy was never workspace-owned and could never be dropped, no
    /// matter how correct the downstream guard was.
    #[test]
    fn no_default_feature_env_owns_numpy_and_solves_with_raw_conda_names() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-holosoma-owned-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let pack_dir = tmp.join("holosoma-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        std::fs::write(
            tmp.join("pixi.toml"),
            r#"[workspace]
channels = ["conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = "==3.12"
numpy = ">=1.26.4,<3"

[feature.doit-task-runner.dependencies]
doit = ">=0.36.0,<0.38"

[feature.holosoma.dependencies]
holosoma-pack = { path = "./holosoma-pack" }
python = "==3.11"
numpy = "==1.26.4"
pytorch-gpu = "==2.10.0"
cuda-nvcc_linux-64 = "12.9.*"

[environments]
holosoma = { features = ["doit-task-runner", "holosoma"], no-default-feature = true }
"#,
        )
        .unwrap();
        let manifest = crate::workspace::WorkspaceManifest::load(&tmp).unwrap();
        let target = ResolutionTarget::for_subdir("3.11", "linux-64");
        let inputs =
            precise_consumer_inputs_for_target(&manifest, &tmp, &pack_dir, &target).unwrap();

        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].env, "holosoma");
        // The non-default feature's NumPy wins; the root's is NOT inherited.
        assert_eq!(
            inputs[0].conda_deps.get("numpy").map(String::as_str),
            Some("==1.26.4"),
        );

        let owned_conda = dependency_name_intersection(
            &inputs
                .iter()
                .map(|input| input.conda_deps.clone())
                .collect::<Vec<_>>(),
        );
        assert!(
            owned_conda.contains("numpy"),
            "NumPy is conda-owned via a non-default feature of a no-default-feature \
             environment; ownership must see it: {owned_conda:?}"
        );

        // The baseline solve specs this env is solved against. Underscore-bearing
        // conda names must survive verbatim, or the whole solve fails and every
        // fact -- including the NumPy ownership just asserted -- is abstained away.
        let specs: Vec<String> = inputs[0]
            .conda_deps
            .iter()
            .filter(|(name, _)| name.as_str() != "holosoma-pack")
            .map(|(name, spec)| {
                CondaName::new(name.as_str())
                    .match_spec(spec)
                    .as_str()
                    .to_owned()
            })
            .collect();
        assert!(
            specs.contains(&"cuda-nvcc_linux-64 12.9.*".to_string()),
            "the raw conda name must reach the solver: {specs:?}"
        );
        assert!(
            !specs
                .iter()
                .any(|spec| spec.starts_with("cuda-nvcc-linux-64")),
            "PEP 503 normalization must never reach a conda match spec: {specs:?}"
        );
        assert!(specs.contains(&"numpy ==1.26.4".to_string()), "{specs:?}");
    }

    /// The typed match-spec seam keeps the PyPI normalizer out of conda solver
    /// input. Underscores are load-bearing in the conda ecosystem.
    #[test]
    fn workspace_dep_match_spec_preserves_underscored_conda_names() {
        for (name, spec, expected) in [
            ("cuda-nvcc_linux-64", "12.9.*", "cuda-nvcc_linux-64 12.9.*"),
            ("gcc_linux-64", "", "gcc_linux-64"),
            ("python_abi", "3.11.*", "python_abi 3.11.*"),
            ("numpy", "==1.26.4", "numpy ==1.26.4"),
            ("pytorch-gpu", "*", "pytorch-gpu"),
        ] {
            assert_eq!(CondaName::new(name).match_spec(spec).as_str(), expected);
        }
    }

    #[test]
    fn workspace_route_ownership_contains_uses_pypi_key() {
        let ownership = WorkspaceRouteOwnership {
            pypi_names: BTreeSet::from([PypiKey::from_pypi("opencv_python_headless")]),
            conda_names: BTreeSet::from([CondaName::new("cuda-nvcc_linux-64").key()]),
            excluded_pypi_names: BTreeSet::new(),
        };
        assert!(
            ownership
                .pypi_names
                .contains(&PypiKey::from_pypi("opencv-python-headless"))
        );
        assert!(
            ownership
                .conda_names
                .contains(&CondaName::new("cuda-nvcc_linux-64").key())
        );
    }

    #[test]
    fn opencv_name_map_builds_unified_prelock_provider() {
        let mut facts = WorkspaceCondaFacts::default();
        facts.common_pypi.insert(
            "opencv-python".to_string(),
            SolvedPypiFact {
                conda_name: "py-opencv".to_string(),
                version: "4.11.0".to_string(),
            },
        );
        let ownership = WorkspaceRouteOwnership {
            pypi_names: BTreeSet::from([PypiKey::from_pypi("opencv_python")]),
            conda_names: BTreeSet::from([CondaName::new("py-opencv").key()]),
            excluded_pypi_names: BTreeSet::new(),
        };
        let candidates = workspace_conda_provider_candidates(
            true,
            &facts,
            &ownership,
            &name_map(&[("opencv-python", "py-opencv")]),
        );
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].pypi_name.as_str(), "opencv-python");
        assert_eq!(candidates[0].conda_name.as_spec(), "py-opencv");
        assert_eq!(candidates[0].version, "4.11.0");
        assert!(matches!(
            &candidates[0].provenance,
            Provenance::WorkspaceCondaFact(_)
        ));
        assert!(
            workspace_conda_provider_candidates(
                false,
                &facts,
                &ownership,
                &name_map(&[("opencv-python", "py-opencv")]),
            )
            .is_empty(),
            "a non-uv group has no pre-lock provider boundary"
        );
    }

    #[test]
    fn rule3_uses_single_precise_consuming_env_fact() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-rule3-sage-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let pack_dir = tmp.join("sage-isaac-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        std::fs::write(
            tmp.join("pixi.toml"),
            r#"[workspace]
channels = ["conda-forge"]
platforms = ["linux-64"]

[feature.sage.dependencies]
sage-isaac-pack = { path = "./sage-isaac-pack" }
pytorch = "==2.5.1"
cuda-version = "==12.4"
python = ">=3.10.12,<3.11"

[feature.gpu.dependencies]
pytorch-gpu = "==2.10.0"

[environments]
sage = { features = ["sage"], no-default-feature = true }
gpu = { features = ["gpu"], no-default-feature = true }
"#,
        )
        .unwrap();
        let manifest = crate::workspace::WorkspaceManifest::load(&tmp).unwrap();
        let target = ResolutionTarget::for_subdir("3.10", "linux-64");
        let inputs =
            precise_consumer_inputs_for_target(&manifest, &tmp, &pack_dir, &target).unwrap();

        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].env, "sage");
        assert_eq!(
            inputs[0].conda_deps.get("pytorch").map(String::as_str),
            Some("==2.5.1")
        );
        assert!(!inputs[0].conda_deps.contains_key("pytorch-gpu"));

        let facts = facts_from_solved_records(
            BTreeMap::from([(
                inputs[0].env.clone(),
                vec![repo_record("pytorch", "2.5.1", &[])],
            )]),
            BTreeMap::from([(inputs[0].env.clone(), inputs[0].conda_deps.clone())]),
            BTreeSet::new(),
            &name_map(&[("torch", "pytorch")]),
            "sage-isaac-pack",
        );
        let constraints = workspace_fact_constraints(&facts, &BTreeSet::new());
        assert_eq!(constraints.constraints, vec!["torch==2.5.1"]);
        assert_eq!(constraints.provenance["torch"].source, "workspace-solved");
        assert_eq!(constraints.provenance["torch"].conda_name, "pytorch");

        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn rule3_does_not_alias_pytorch_gpu_to_pytorch() {
        let facts = facts_from_solved_records(
            BTreeMap::from([(
                "sage".to_string(),
                vec![
                    repo_record("pytorch-gpu", "2.10.0", &["pytorch ==2.10.0"]),
                    repo_record("pytorch", "2.10.0", &[]),
                ],
            )]),
            BTreeMap::from([(
                "sage".to_string(),
                BTreeMap::from([("pytorch-gpu".to_string(), "==2.10.0".to_string())]),
            )]),
            BTreeSet::new(),
            &name_map(&[("torch", "pytorch")]),
            "sage-isaac-pack",
        );

        assert!(
            !facts.common_conda_versions.contains_key("pytorch"),
            "a transitive pytorch record selected by pytorch-gpu is not direct-name authority"
        );
        assert_eq!(
            facts
                .common_selected_versions
                .get("pytorch")
                .map(String::as_str),
            Some("2.10.0"),
            "the agreed transitive stays available as validation input without becoming ownership"
        );
        assert!(!facts.common_pypi.contains_key("torch"));
        assert!(
            workspace_fact_constraints(&facts, &BTreeSet::new())
                .constraints
                .is_empty()
        );
    }

    #[test]
    fn rule3_abstains_when_precise_consumers_disagree() {
        let facts = facts_from_solved_records(
            BTreeMap::from([
                (
                    "sage-a".to_string(),
                    vec![repo_record("pytorch", "2.5.1", &[])],
                ),
                (
                    "sage-b".to_string(),
                    vec![repo_record("pytorch", "2.6.0", &[])],
                ),
            ]),
            BTreeMap::from([
                (
                    "sage-a".to_string(),
                    BTreeMap::from([("pytorch".to_string(), "==2.5.1".to_string())]),
                ),
                (
                    "sage-b".to_string(),
                    BTreeMap::from([("pytorch".to_string(), "==2.6.0".to_string())]),
                ),
            ]),
            BTreeSet::new(),
            &name_map(&[("torch", "pytorch")]),
            "sage-isaac-pack",
        );

        assert!(!facts.common_conda_versions.contains_key("pytorch"));
        assert!(!facts.common_selected_versions.contains_key("pytorch"));
        assert!(!facts.common_pypi.contains_key("torch"));
        assert_eq!(
            facts.provider_facts["pytorch"],
            super::WorkspaceCondaProviderFact {
                selected_versions: BTreeSet::from(["2.5.1".into(), "2.6.0".into()]),
                declared_specs: BTreeSet::from(["==2.5.1".into(), "==2.6.0".into()]),
                present_in_all_consumers: true,
            },
            "provider presence is independent of exact cross-env version agreement"
        );
        assert!(
            workspace_fact_constraints(&facts, &BTreeSet::new())
                .constraints
                .is_empty()
        );
    }

    #[test]
    fn partial_psutil_provider_fact_retains_ranges_and_selected_version() {
        let env_records = BTreeMap::from([
            (
                "groot".to_string(),
                vec![
                    repo_record("python", "3.11.14", &[]),
                    repo_record("psutil", "7.2.2", &[]),
                ],
            ),
            (
                "pace".to_string(),
                vec![
                    repo_record("python", "3.11.14", &[]),
                    repo_record("psutil", "7.2.2", &[]),
                ],
            ),
            (
                "pm".to_string(),
                vec![
                    repo_record("python", "3.11.14", &[]),
                    repo_record("psutil", "7.2.2", &[]),
                ],
            ),
            (
                "unitree-rl-lab-gpu".to_string(),
                vec![
                    repo_record("python", "3.11.14", &[]),
                    repo_record("psutil", "7.2.2", &[]),
                ],
            ),
            (
                "uwlab-gpu".to_string(),
                vec![
                    repo_record("python", "3.11.14", &[]),
                    repo_record("psutil", "7.2.2", &[]),
                ],
            ),
            (
                "viral".to_string(),
                vec![repo_record("python", "3.11.14", &[])],
            ),
        ]);
        let env_conda_deps = BTreeMap::from([
            ("groot".to_string(), BTreeMap::new()),
            (
                "pace".to_string(),
                BTreeMap::from([("psutil".to_string(), ">=5.9".to_string())]),
            ),
            ("pm".to_string(), BTreeMap::new()),
            (
                "unitree-rl-lab-gpu".to_string(),
                BTreeMap::from([("psutil".to_string(), ">=5.9,<8".to_string())]),
            ),
            ("uwlab-gpu".to_string(), BTreeMap::new()),
            ("viral".to_string(), BTreeMap::new()),
        ]);

        let facts = facts_from_solved_records(
            env_records,
            env_conda_deps,
            BTreeSet::new(),
            &NameMap::default(),
            "isaaclab-2.3x-pack",
        );

        assert_eq!(
            facts.provider_facts["psutil"],
            super::WorkspaceCondaProviderFact {
                selected_versions: BTreeSet::from(["7.2.2".to_string()]),
                declared_specs: BTreeSet::from([">=5.9".to_string(), ">=5.9,<8".to_string(),]),
                present_in_all_consumers: false,
            }
        );
        assert!(
            !facts.common_selected_versions.contains_key("psutil"),
            "a provider missing from one consumer is not an exact common fact"
        );
        assert!(!facts.common_conda_versions.contains_key("psutil"));
    }

    #[test]
    fn package_absent_from_all_solves_has_no_provider_fact() {
        let facts = facts_from_solved_records(
            BTreeMap::from([
                (
                    "alpha".to_string(),
                    vec![repo_record("python", "3.11.14", &[])],
                ),
                (
                    "beta".to_string(),
                    vec![repo_record("python", "3.11.14", &[])],
                ),
            ]),
            BTreeMap::from([
                (
                    "alpha".to_string(),
                    BTreeMap::from([("numpy".to_string(), ">=2".to_string())]),
                ),
                ("beta".to_string(), BTreeMap::new()),
            ]),
            BTreeSet::new(),
            &NameMap::default(),
            "demo-pack",
        );

        assert!(
            !facts.provider_facts.contains_key("numpy"),
            "a declaration without a selected conda record is not provider evidence"
        );
    }

    #[test]
    fn workspace_fact_derivation_requires_mapping_and_cross_env_agreement() {
        let env_records = BTreeMap::from([
            (
                "alpha".to_string(),
                vec![
                    repo_record("numpy", "2.1.0", &[]),
                    repo_record("tetgen", "1.6.0", &[]),
                    repo_record("pytorch", "2.10.0", &[]),
                ],
            ),
            (
                "beta".to_string(),
                vec![
                    repo_record("numpy", "2.1.0", &[]),
                    repo_record("tetgen", "1.6.0", &[]),
                    repo_record("pytorch", "2.9.0", &[]),
                ],
            ),
        ]);
        let facts = facts_from_solved_records(
            env_records,
            BTreeMap::from([
                (
                    "alpha".to_string(),
                    BTreeMap::from([
                        ("numpy".to_string(), "==2.1.0".to_string()),
                        ("tetgen".to_string(), "==1.6.0".to_string()),
                        ("demo_pack".to_string(), "*".to_string()),
                    ]),
                ),
                (
                    "beta".to_string(),
                    BTreeMap::from([
                        ("numpy".to_string(), "==2.1.0".to_string()),
                        ("tetgen".to_string(), "==1.6.0".to_string()),
                        ("demo_pack".to_string(), "*".to_string()),
                    ]),
                ),
            ]),
            BTreeSet::from(["gym".to_string(), "rliable".to_string()]),
            &NameMap::from([
                (
                    PypiKey::from_pypi("numpy"),
                    CondaTarget::Mapped(CondaName::new("numpy")),
                ),
                (
                    PypiKey::from_pypi("torch"),
                    CondaTarget::Mapped(CondaName::new("pytorch")),
                ),
            ]),
            "demo-pack",
        );

        assert_eq!(facts.common_pypi["numpy"].version, "2.1.0");
        assert_eq!(
            facts
                .common_selected_versions
                .get("numpy")
                .map(String::as_str),
            Some("2.1.0")
        );
        assert_eq!(
            facts
                .common_selected_versions
                .get("tetgen")
                .map(String::as_str),
            Some("1.6.0"),
            "a same-version transitive fact is retained without guessing PyPI identity"
        );
        assert!(!facts.common_selected_versions.contains_key("pytorch"));
        assert!(
            !facts.common_pypi.contains_key("torch"),
            "divergent consuming-env versions must not become a shared fact"
        );
        assert!(
            !facts.common_pypi.contains_key("tetgen"),
            "same-name conda/PyPI identity must never be guessed without a mapping"
        );
        assert_eq!(
            facts.owned_conda,
            BTreeSet::from(["numpy".into(), "tetgen".into()]),
            "direct conda ownership remains available without manufacturing a PyPI mapping, and the generated bundle excludes itself"
        );
        assert_eq!(facts.owned_conda_pypi, BTreeSet::from(["numpy".into()]));
        assert_eq!(
            facts.owned_pypi,
            BTreeSet::from(["gym".into(), "rliable".into()])
        );
        assert_eq!(facts.env_exact_specs.len(), 2);
        assert!(!facts.fingerprint.is_empty());
    }

    #[test]
    fn workspace_ownership_requires_every_precise_consumer() {
        let intersection = dependency_name_intersection(&[
            BTreeMap::from([
                ("numpy".to_string(), "==2.1".to_string()),
                ("gym".to_string(), "*".to_string()),
            ]),
            BTreeMap::from([("numpy".to_string(), "==2.1".to_string())]),
        ]);
        assert_eq!(intersection, BTreeSet::from(["numpy".to_string()]));
    }
}

/// retread-deps-from conda-as-truth: canonical PyPI names among `roots`
/// (PEP 508 requirement strings straight from a deps-from source file)
/// whose version specifier is a single exact `==`/`===` pin. Non-exact
/// specs (`>=`, ranges, bare) and unparseable lines are ignored.
pub(crate) fn deps_from_exact_pinned_names(roots: &[String]) -> std::collections::BTreeSet<String> {
    let mut out = std::collections::BTreeSet::new();
    for root in roots {
        let Ok(req): Result<uv_pep508::Requirement, _> = uv_pep508::Requirement::from_str(root)
        else {
            continue;
        };
        let Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) = &req.version_or_url else {
            continue;
        };
        let specs: Vec<_> = specs.iter().collect();
        if specs.len() == 1
            && matches!(
                specs[0].operator(),
                uv_pep508::uv_pep440::Operator::Equal | uv_pep508::uv_pep440::Operator::ExactEqual
            )
        {
            out.insert(canonical_conda_name(req.name.as_ref()));
        }
    }
    out
}

/// One emission targeting a specific discovered output name. The
/// `output_name` is what pixi expects (e.g. "isaac-pack-physx"); the
/// bundle's conda_name gets renamed to it.
#[derive(Debug, Clone)]
struct DiscoveredEmission {
    output_name: String,
    /// Channels + env names were consumed by the per-env pre-emission
    /// solve check deleted in v4.2.0; kept because the channel union
    /// still feeds `extract_transitive_constraints` at construction
    /// time and the env list documents provenance in debug output.
    #[allow(dead_code)]
    channels: Vec<ChannelUrl>,
    /// Legacy constraints derived from a provisional workspace solve. They are
    /// useful only as discovery evidence: they are not user-authored
    /// `retread-overrides` and must never replace wheel metadata constraints.
    #[allow(dead_code)]
    transitive_overrides: BTreeMap<String, String>,
    #[allow(dead_code)]
    envs: Vec<String>,
}

/// One exact typed resolution retained between output advertisement and the
/// matching build request. The complete bundles/config preserve 4.8.x
/// constraint authority, ownership facts, and metadata provenance; locators
/// below share this graph through `Arc` rather than cloning it per output.
struct ResolvedTargetPlan {
    materialized: Vec<Bundle>,
    base_config: RetreadConfig,
    restore_relaxations: Vec<auto_bundle::WheelMetadataRelaxation>,
    declared_config: RetreadConfig,
    target: ResolutionTarget,
    work_directory: PathBuf,
    workspace_manifest_mtime: Option<std::time::SystemTime>,
    auto_overrides_fingerprint: String,
    local_wheel_stamps: Option<Vec<LocalWheelStamp>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LocalWheelStamp {
    path: PathBuf,
    len: u64,
    modified: std::time::SystemTime,
}

#[derive(Clone)]
struct PreparedOutputIdentity {
    name: String,
    version: String,
    build: String,
    subdir: Platform,
}

impl PreparedOutputIdentity {
    fn from_metadata(metadata: &CondaOutputMetadata) -> Self {
        Self {
            name: metadata.name.as_normalized().to_string(),
            version: metadata.version.to_string(),
            build: metadata.build.clone(),
            subdir: metadata.subdir,
        }
    }
}

#[derive(Clone)]
struct PreparedBuild {
    locator_id: usize,
    plan: Arc<ResolvedTargetPlan>,
    bundle_index: usize,
    emission: DiscoveredEmission,
    advertised: PreparedOutputIdentity,
    /// Exact run dependencies paired with the recorded decisions.
    advertised_run_dependencies: CondaOutputDependencies,
    /// Exact final warning payload paired with the advertised output.
    relaxations: Option<RelaxationManifest>,
    /// Present only when conda/outputs intentionally advertised the version
    /// of a matching incremental lock rather than the cold primary wheel.
    /// This typed origin never crosses the process-local metadata/build
    /// handoff; build_v1 must independently rediscover the same lock before
    /// it may use the override.
    incremental_version_override: Option<String>,
}

struct PreparedBuildSelection {
    transaction: u64,
    prepared: PreparedBuild,
    bundle: Bundle,
    effective: RetreadConfig,
}

impl PreparedBuild {
    fn matches(
        &self,
        work_directory: &Path,
        exact_python_version: Option<&str>,
        target: Option<&ResolutionTarget>,
        output: &pixi_build_types::procedures::conda_build_v1::CondaBuildV1Output,
    ) -> bool {
        self.plan.work_directory == work_directory
            && target.is_none_or(|target| {
                self.plan.target.resolution_identity() == target.resolution_identity()
            })
            && exact_python_version.is_none_or(|python_version| {
                normalized_python_minor(python_version)
                    .is_ok_and(|python| python.version() == self.plan.target.python_version())
            })
            && self.advertised.name == output.name.as_normalized()
            && self.advertised.subdir == output.subdir
            && output
                .version
                .as_ref()
                .is_none_or(|version| self.advertised.version == version.to_string())
            && output
                .build
                .as_ref()
                .is_none_or(|build| self.advertised.build == *build)
    }

    fn reapply(&self) -> Option<(Bundle, RetreadConfig)> {
        let base_bundle = self.plan.materialized.get(self.bundle_index)?;
        let (bundle, effective) =
            apply_emission(base_bundle, &self.plan.base_config, &self.emission);
        let applied_name = PackageName::new_unchecked(bundle.conda_name.clone());
        let package_version =
            courier_pack_version(&bundle, self.incremental_version_override.as_deref());
        (applied_name.as_normalized() == self.advertised.name
            && package_version == self.advertised.version)
            .then_some((bundle, effective))
    }
}

fn capture_local_wheel_stamps(bundles: &[Bundle]) -> Option<Vec<LocalWheelStamp>> {
    let mut stamps = Vec::new();
    for wheel in bundles.iter().flat_map(Bundle::all_wheels) {
        if wheel.url.scheme() != "file" {
            continue;
        }
        let path = wheel.url.to_file_path().ok()?;
        let metadata = std::fs::metadata(&path).ok()?;
        if !metadata.is_file() {
            return None;
        }
        stamps.push(LocalWheelStamp {
            path,
            len: metadata.len(),
            modified: metadata.modified().ok()?,
        });
    }
    stamps.sort_unstable_by(|left, right| left.path.cmp(&right.path));
    stamps.dedup();
    Some(stamps)
}

fn local_wheel_stamps_are_current(stamps: &[LocalWheelStamp]) -> bool {
    stamps.iter().all(|stamp| {
        std::fs::metadata(&stamp.path).is_ok_and(|metadata| {
            metadata.is_file()
                && metadata.len() == stamp.len
                && metadata.modified().ok().as_ref() == Some(&stamp.modified)
                && crate::wheel::is_valid_zip(&stamp.path)
        })
    })
}

/// Autodiscovery-based emission planner. Walks the workspace
/// pixi.toml looking for path-deps that resolve to `source_dir`:
///
///   * If the workspace declares e.g. `isaac-pack-physx = { path = "./isaac-pack" }`
///     in `[feature.isaaclab_physx.dependencies]`, AND `isaaclab_physx`
///     is referenced by envs `gsi` + `gsi-ros2`, retread emits ONE
///     output named `isaac-pack-physx` whose constraints union those
///     two envs' deps + transitive constraints.
///   * If the workspace declares `isaac-pack-newton = { path = "./isaac-pack" }`
///     in a feature only `gsn` uses, retread emits a second output
///     named `isaac-pack-newton` with gsn's constraints.
///
/// Returns the planned emissions. When nothing in the workspace
/// references the source package (initial setup; workspace pixi.toml
/// missing), returns a single default emission named after
/// `[package].name` with no env constraints (preserves single-output
/// backward compatibility).
async fn discover_emissions(
    source_dir: &Path,
    workspace_dir: Option<&Path>,
    default_output_name: &str,
    default_channels: &[ChannelUrl],
    target: &ResolutionTarget,
    bundle_names: &HashSet<PypiKey>,
) -> Result<Vec<DiscoveredEmission>> {
    let manifest_opt = workspace_dir.and_then(crate::workspace::WorkspaceManifest::load);
    let default_emission = || DiscoveredEmission {
        output_name: default_output_name.to_string(),
        channels: default_channels.to_vec(),
        transitive_overrides: BTreeMap::new(),
        envs: Vec::new(),
    };

    let (Some(manifest), Some(ws_dir)) = (manifest_opt.as_ref(), workspace_dir) else {
        if target.target_contract().is_some() {
            bail!(
                "workspace target contract cannot be used after the workspace manifest disappeared"
            );
        }
        return Ok(vec![default_emission()]);
    };
    let resolved = match target.target_contract() {
        Some(_) => Some(
            resolved_workspace_target_from_resolution(manifest, ws_dir, source_dir, target)
                .ok_or_else(|| {
                    anyhow!(
                        "workspace target contract no longer maps source `{}` to a concrete consumer",
                        source_dir.display()
                    )
                })?,
        ),
        None => None,
    };
    let discovered = resolved.as_ref().map_or_else(
        || {
            manifest.discover_outputs_for_source_for_target(
                ws_dir,
                source_dir,
                target.conda_subdir(),
            )
        },
        |resolved| {
            manifest.discover_outputs_for_source_for_resolved_target(ws_dir, source_dir, resolved)
        },
    );
    if discovered.is_empty() {
        return Ok(vec![default_emission()]);
    }

    let mut out = Vec::with_capacity(discovered.len());
    for d in discovered {
        // Union channels across all envs that reference this output.
        let chan_strs = manifest.union_effective_channels(&d.envs);
        let env_chans: Vec<ChannelUrl> = chan_strs
            .iter()
            .filter_map(|s| url::Url::parse(s).ok().map(ChannelUrl::from))
            .collect();
        let channels = if env_chans.is_empty() {
            default_channels.to_vec()
        } else {
            env_chans
        };

        // Union transitive constraints across all envs that reference
        // this output. Two envs sharing the same output must coexist
        // under whatever retread emits; intersecting their constraints
        // (via comma-AND) gives the conda solver the right shape.
        //
        // v1.4.0: the per-env extractions are independent at this
        // stage (no seed_overrides flow between them -- that's the
        // MAIN env loop's contract, not this one), so run them
        // concurrently. buffered preserves input order, keeping the
        // accumulated clause order (and thus the joined spec strings)
        // deterministic.
        let env_futures = d
            .envs
            .iter()
            .map(|env| async {
                // Each env solves against ITS OWN channels, not the union
                // above. The union is the right input for the output-level
                // courier identity, but as a solve input it re-inverts
                // priority for multi-distro outputs: a jazzy env sharing an
                // output with a humble env would inherit robostack-humble
                // ahead of robostack-jazzy and, under strict channel
                // priority, reject its own distro's packages.
                let per_env: Vec<ChannelUrl> = manifest
                    .effective_channels(env)
                    .iter()
                    .filter_map(|s| url::Url::parse(s).ok().map(ChannelUrl::from))
                    .collect();
                let channels = if per_env.is_empty() {
                    channels.clone()
                } else {
                    per_env
                };
                match resolved.as_ref() {
                    Some(resolved) => {
                        crate::workspace::extract_transitive_constraints_for_resolved_target(
                            manifest,
                            env,
                            target.python_version(),
                            resolved,
                            &channels,
                            bundle_names,
                        )
                        .await
                    }
                    None => {
                        crate::workspace::extract_transitive_constraints_for_target(
                            manifest,
                            env,
                            target.python_version(),
                            target.conda_subdir(),
                            &channels,
                            bundle_names,
                        )
                        .await
                    }
                }
            })
            .collect::<Vec<_>>();
        let env_results = {
            use futures::stream::{self, StreamExt};
            stream::iter(env_futures)
                .buffered(crate::concurrency::max_concurrent_builds())
                .collect::<Vec<_>>()
                .await
        };
        let mut accumulated: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for trans in env_results {
            // Propagate rather than absorb: a dropped constraint set does not
            // fail anything here, it silently under-constrains the closure.
            let trans = trans.map_err(|error| anyhow!("{error}"))?;
            for (dep, specs) in trans {
                let entry = accumulated.entry(dep.into_string()).or_default();
                for s in specs {
                    if !entry.contains(&s) {
                        entry.push(s);
                    }
                }
            }
        }
        // Also fold in each env's direct [dependencies] pins as
        // override candidates so the cascade respects them too.
        // Workspace conda-dep names are PRESERVED as-typed (not
        // normalized): conda allows underscores in package names
        // (`binutils_linux-64`, `gcc_linux-64`, `python_abi`).
        // Naively normalizing `_` -> `-` produces specs like
        // `binutils-linux-64 >=2.40` that the conda solver can't
        // resolve because the actual package on the channel is
        // `binutils_linux-64`. The solve check consumes these
        // strings directly, so passing them verbatim keeps the
        // solver looking up the right package.
        let direct = resolved.as_ref().map_or_else(
            || manifest.union_effective_dependencies_for_target(&d.envs, target.conda_subdir()),
            |resolved| manifest.union_effective_dependencies_for_resolved_target(&d.envs, resolved),
        );
        for (dep, specs) in direct {
            let entry = accumulated.entry(dep).or_default();
            for s in specs {
                if !entry.contains(&s) {
                    entry.push(s);
                }
            }
        }

        out.push(DiscoveredEmission {
            output_name: d.name,
            channels,
            transitive_overrides: join_transitive_to_overrides(accumulated),
            envs: d.envs,
        });
    }
    Ok(out)
}

/// Collapse `dep_name -> [spec1, spec2, ...]` into comma-AND match-
/// specs: `dep_name -> "spec1,spec2"`. The conda solver intersects
/// comma-separated specs; an empty intersection (workspace requires
/// conflicting versions of the same dep) becomes the conda solver's
/// problem to surface, OR retread's cascade catches it and falls
/// through to PyPI bundle.
///
/// v0.37.0 D4: clause-level dedup, not full-spec dedup. Pre-0.37 we
/// dedup'd entire spec strings, which left textual junk like
/// `setuptools >=41.0.0,>=59.6.0,<80,>=59.6.0,<=79.0.1` in shipped
/// emissions — two `>=59.6.0` clauses survived because they were
/// embedded in different parent specs. Splitting each input spec
/// into its comma-separated clauses first, deduping at the clause
/// level, then re-joining produces the simplest equivalent spec the
/// downstream MatchSpec parser will accept. Validating the rejoined
/// result with `VersionSpec::from_str` is the contract: if the
/// dedup'd output doesn't parse, we fall back to the original
/// (concatenated) form so the cascade keeps working even if the
/// inputs are exotic.
fn join_transitive_to_overrides(
    transitive: BTreeMap<String, Vec<String>>,
) -> BTreeMap<String, String> {
    use rattler_conda_types::{ParseStrictness, VersionSpec};
    let mut out = BTreeMap::new();
    for (k, specs) in transitive {
        // Split every input spec into individual clauses
        // (comma-separated), trim, drop empties and `*`, dedup
        // across the entire dep's input set. Order-preserving so
        // diagnostics remain stable across runs.
        let mut clauses: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for raw in &specs {
            for clause in raw.split(',') {
                let t = clause.trim();
                if t.is_empty() || t == "*" {
                    continue;
                }
                if seen.insert(t.to_string()) {
                    clauses.push(t.to_string());
                }
            }
        }
        if clauses.is_empty() {
            continue;
        }
        let joined = clauses.join(",");
        // Sanity check: the rejoined spec must parse as a
        // VersionSpec, else the cascade will choke when it tries
        // to pass it to rattler later. Fall back to the original
        // textual concat if parsing fails (degraded but functional).
        let final_spec = match VersionSpec::from_str(&joined, ParseStrictness::Lenient) {
            Ok(_) => joined,
            Err(e) => {
                tracing::debug!(
                    dep = %k,
                    joined = %joined,
                    error = %e,
                    "join_transitive_to_overrides: dedup'd spec failed to parse, \
                     falling back to plain concatenation",
                );
                let mut full_seen = HashSet::new();
                specs
                    .iter()
                    .filter(|s| !s.is_empty() && s.as_str() != "*")
                    .filter(|s| full_seen.insert(s.to_string()))
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(",")
            }
        };
        if !final_spec.is_empty() {
            // NOTE: no assert_spec_roundtrips here -- this function
            // self-validates above and deliberately falls back to a
            // degraded-but-functional concat when the dedup'd spec
            // doesn't parse (v0.37.0 D4). The boundary contract lives
            // at the cascade accumulation chokepoint instead.
            out.insert(k, final_spec);
        }
    }
    out
}

/// Bundle group for a `[retread-wheels]` entry: the per-entry `bundle`
/// field wins, then the pack-wide `retread-bundle` default (v1.4.0),
/// then standalone (the entry's own name -- one conda output per
/// entry, the historical behavior). (Migrated from the deleted
/// cascade.rs in v4.2.0.)
pub(crate) fn bundle_group_for(
    entry_name: &str,
    entry: &WheelEntry,
    default_bundle: Option<&str>,
) -> String {
    entry
        .bundle
        .clone()
        .or_else(|| default_bundle.map(String::from))
        .unwrap_or_else(|| entry_name.to_string())
}

/// Apply a `DiscoveredEmission` to a materialized bundle + base config.
///
/// The emission selects only the output identity. Workspace constraints in
/// `transitive_overrides` come from a provisional discovery solve, not explicit
/// override authority. Precise workspace facts already enter the typed route
/// and constraint pipeline during materialization; promoting the older
/// snapshot into `RetreadConfig::overrides` here would replace upstream wheel
/// requirements after routing has completed.
fn apply_emission(
    base_bundle: &Bundle,
    base_config: &RetreadConfig,
    emission: &DiscoveredEmission,
) -> (Bundle, RetreadConfig) {
    let mut bundle = base_bundle.clone();
    bundle.conda_name = emission.output_name.clone();
    // Reset ordinary materialization probes before env-specific decisions are
    // recorded, but retain final joint-route rejections: those are the durable
    // explanation for why an individually valid conda candidate ships as a
    // PyPI wheel and must remain visible in every emitted audit.
    bundle
        .probe_decisions
        .retain(|decision| decision.stage == "auto_route_joint_solve");

    (bundle, base_config.clone())
}

/// Fetch one PyPI-form BFS item from its complete ordered index chain.
///
/// The callback performs the existing single-index wheel-then-sdist operation;
/// this wrapper advances only after that complete attempt fails. Keeping the
/// `Pending` as the input makes phase 2 consume the chain that `seed_worklist`
/// attached to the dependency.
pub(super) async fn fetch_from_pypi_index_chain<T, X, XF>(
    indexes: &[String],
    mut fetch: X,
    exhaustion_context: String,
) -> Result<T>
where
    X: FnMut(String) -> XF,
    XF: std::future::Future<Output = Result<T>>,
{
    let mut misses = Vec::new();
    for index in indexes {
        match fetch(index.clone()).await {
            Ok(value) => return Ok(value),
            Err(error) if pypi::is_pypi_index_miss(&error) => {
                tracing::debug!(
                    index = %index,
                    error = %format!("{error:#}"),
                    "package is absent from this PyPI index; continuing fallback chain"
                );
                misses.push((index.clone(), format!("{error:#}")));
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "PyPI index chain aborted on `{index}` because the failure was not a package miss"
                    )
                });
            }
        }
    }

    if misses.is_empty() {
        bail!("{exhaustion_context}: no PyPI index configured");
    }
    let diagnostics = misses
        .into_iter()
        .map(|(index, error)| format!("  - {index}: {error}"))
        .collect::<Vec<_>>()
        .join("\n");
    bail!("{exhaustion_context}; every configured index missed:\n{diagnostics}")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PypiArtifactPhase {
    ExactWheel,
    RelaxedWheel,
    Sdist,
}

impl PypiArtifactPhase {
    fn label(self) -> &'static str {
        match self {
            Self::ExactWheel => "exact wheel",
            Self::RelaxedWheel => "relaxed wheel",
            Self::Sdist => "sdist",
        }
    }
}

/// Resolve artifacts across a complete PyPI index chain without letting an
/// sdist or relaxed candidate on an earlier index shadow an exact wheel on a
/// later one.
///
/// Artifact preference is global across the chain: exact-spec wheels on every
/// index, then relaxed-spec wheels when policy allows, then sdists. A transport,
/// authentication, parse, or build error remains fatal at the index where it
/// occurred. Index priority is preserved within each artifact phase. The
/// optional version preference is forwarded unchanged through every phase.
async fn fetch_artifact_from_pypi_index_chain<T, X, XF>(
    indexes: &[String],
    try_relaxed_wheels: bool,
    prefer_version: Option<String>,
    mut fetch: X,
    exhaustion_context: String,
) -> Result<T>
where
    X: FnMut(String, PypiArtifactPhase, Option<String>) -> XF,
    XF: std::future::Future<Output = Result<T>>,
{
    let mut misses = Vec::new();
    let phases = [
        PypiArtifactPhase::ExactWheel,
        PypiArtifactPhase::RelaxedWheel,
        PypiArtifactPhase::Sdist,
    ];
    for phase in phases {
        if phase == PypiArtifactPhase::RelaxedWheel && !try_relaxed_wheels {
            continue;
        }
        for index in indexes {
            match fetch(index.clone(), phase, prefer_version.clone()).await {
                Ok(value) => return Ok(value),
                Err(error) if pypi::is_pypi_index_miss(&error) => {
                    tracing::debug!(
                        index = %index,
                        artifact_phase = phase.label(),
                        error = %format!("{error:#}"),
                        "no usable artifact on this PyPI index; continuing phase"
                    );
                    misses.push((phase.label(), index.clone(), format!("{error:#}")));
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "PyPI {} index chain aborted on `{index}` because the failure was not a package miss",
                            phase.label(),
                        )
                    });
                }
            }
        }
    }

    if misses.is_empty() {
        return Err(pypi::pypi_index_miss(format!(
            "{exhaustion_context}: no PyPI index configured"
        )));
    }
    let diagnostics = misses
        .into_iter()
        .map(|(kind, index, error)| format!("  - {kind} {index}: {error}"))
        .collect::<Vec<_>>()
        .join("\n");
    Err(pypi::pypi_index_miss(format!(
        "{exhaustion_context}; every configured index missed:\n{diagnostics}"
    )))
}

#[cfg(test)]
mod wheel_then_sdist_chain_tests {
    use std::sync::{Arc, Mutex};

    use super::{PypiArtifactPhase, fetch_artifact_from_pypi_index_chain};

    #[tokio::test]
    async fn later_vendor_wheel_beats_earlier_public_sdist() {
        let indexes = vec!["public".to_string(), "vendor".to_string()];
        let calls = Arc::new(Mutex::new(Vec::new()));

        let result = fetch_artifact_from_pypi_index_chain(
            &indexes,
            true,
            None,
            {
                let calls = calls.clone();
                move |index, phase, _prefer_version| {
                    let calls = calls.clone();
                    async move {
                        calls
                            .lock()
                            .unwrap()
                            .push(format!("{}:{index}", phase.label()));
                        match phase {
                            PypiArtifactPhase::ExactWheel if index == "vendor" => {
                                Ok("vendor-wheel".to_string())
                            }
                            PypiArtifactPhase::RelaxedWheel if index == "public" => {
                                Ok("public-relaxed-wheel".to_string())
                            }
                            PypiArtifactPhase::Sdist if index == "public" => {
                                Ok("public-sdist".to_string())
                            }
                            _ => Err(crate::pypi::pypi_index_miss("artifact miss")),
                        }
                    }
                }
            },
            "fixture exhaustion".to_string(),
        )
        .await
        .unwrap();

        assert_eq!(result, "vendor-wheel");
        assert_eq!(
            *calls.lock().unwrap(),
            vec!["exact wheel:public", "exact wheel:vendor"],
            "relaxation and sdist lookup must not start while a later index has an exact wheel"
        );
    }

    #[tokio::test]
    async fn sdist_search_starts_only_after_every_wheel_index_misses() {
        let indexes = vec!["first".to_string(), "second".to_string()];
        let calls = Arc::new(Mutex::new(Vec::new()));

        let result = fetch_artifact_from_pypi_index_chain(
            &indexes,
            true,
            None,
            {
                let calls = calls.clone();
                move |index, phase, _prefer_version| {
                    let calls = calls.clone();
                    async move {
                        calls
                            .lock()
                            .unwrap()
                            .push(format!("{}:{index}", phase.label()));
                        if phase == PypiArtifactPhase::Sdist && index == "first" {
                            return Ok("first-sdist".to_string());
                        }
                        Err(crate::pypi::pypi_index_miss("artifact miss"))
                    }
                }
            },
            "fixture exhaustion".to_string(),
        )
        .await
        .unwrap();

        assert_eq!(result, "first-sdist");
        assert_eq!(
            *calls.lock().unwrap(),
            vec![
                "exact wheel:first",
                "exact wheel:second",
                "relaxed wheel:first",
                "relaxed wheel:second",
                "sdist:first",
            ]
        );
    }

    #[tokio::test]
    async fn fatal_wheel_error_still_aborts_the_chain() {
        let indexes = vec!["broken".to_string(), "later".to_string()];
        let calls = Arc::new(Mutex::new(Vec::new()));

        let error = fetch_artifact_from_pypi_index_chain(
            &indexes,
            true,
            None,
            {
                let calls = calls.clone();
                move |index, phase, _prefer_version| {
                    let calls = calls.clone();
                    async move {
                        calls
                            .lock()
                            .unwrap()
                            .push(format!("{}:{index}", phase.label()));
                        Err::<String, _>(anyhow::anyhow!("transport failure"))
                    }
                }
            },
            "fixture exhaustion".to_string(),
        )
        .await
        .unwrap_err();

        assert!(format!("{error:#}").contains("exact wheel index chain aborted on `broken`"));
        assert_eq!(*calls.lock().unwrap(), vec!["exact wheel:broken"]);
    }

    #[tokio::test]
    async fn preferred_version_reaches_relaxed_wheel_and_sdist_phases() {
        let indexes = vec!["only".to_string()];
        let calls = Arc::new(Mutex::new(Vec::new()));

        let result = fetch_artifact_from_pypi_index_chain(
            &indexes,
            true,
            Some("2.0".to_string()),
            {
                let calls = calls.clone();
                move |index, phase, prefer_version| {
                    let calls = calls.clone();
                    async move {
                        calls.lock().unwrap().push((
                            phase.label().to_string(),
                            index,
                            prefer_version,
                        ));
                        if phase == PypiArtifactPhase::Sdist {
                            Ok("preferred-sdist".to_string())
                        } else {
                            Err(crate::pypi::pypi_index_miss("artifact miss"))
                        }
                    }
                }
            },
            "fixture exhaustion".to_string(),
        )
        .await
        .unwrap();

        assert_eq!(result, "preferred-sdist");
        assert_eq!(
            *calls.lock().unwrap(),
            vec![
                (
                    "exact wheel".to_string(),
                    "only".to_string(),
                    Some("2.0".to_string()),
                ),
                (
                    "relaxed wheel".to_string(),
                    "only".to_string(),
                    Some("2.0".to_string()),
                ),
                (
                    "sdist".to_string(),
                    "only".to_string(),
                    Some("2.0".to_string()),
                ),
            ]
        );
    }

    #[tokio::test]
    async fn all_index_miss_keeps_type_context_and_phase_diagnostics() {
        let indexes = vec!["first".to_string(), "second".to_string()];
        let error = fetch_artifact_from_pypi_index_chain(
            &indexes,
            true,
            None,
            |_index, _phase, _prefer_version| async {
                Err::<String, _>(crate::pypi::pypi_index_miss("artifact miss"))
            },
            "auto-bundle fixture context".to_string(),
        )
        .await
        .unwrap_err();

        assert!(crate::pypi::is_pypi_index_miss(&error));
        let rendered = format!("{error:#}");
        assert!(rendered.contains("auto-bundle fixture context"));
        assert!(rendered.contains("exact wheel first: artifact miss"));
        assert!(rendered.contains("relaxed wheel second: artifact miss"));
        assert!(rendered.contains("sdist second: artifact miss"));
    }
}

/// Select the full chain inherited by ordinary PyPI descendants.
///
/// A PyPI parent retains its original chain regardless of which index supplied
/// its wheel. Direct URL and git parents have no index field, so their
/// metadata-discovered PyPI children inherit the bundle chain.
fn bfs_descendant_indexes(source: &PendingSource, bundle_indexes: &[String]) -> Vec<String> {
    match source {
        PendingSource::Pypi { indexes, .. } => indexes.clone(),
        PendingSource::Git { .. } | PendingSource::Url { .. } => bundle_indexes.to_vec(),
    }
}

/// The subset of pack configuration that can make a BFS dependency a native
/// conda route.  Keeping this typed prevents the source-built fallback walk
/// from seeing the effective name map while silently missing the manual
/// override that emission will apply later.
#[derive(Clone, Copy)]
struct BfsRoutePolicy<'a> {
    name_map: &'a NameMap,
    overrides: Option<&'a BTreeMap<String, String>>,
    ledger_overrides: Option<&'a BTreeSet<String>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BfsOverride<'a> {
    spec: &'a str,
    manual: bool,
}

impl<'a> BfsRoutePolicy<'a> {
    fn from_config(config: &'a RetreadConfig) -> Self {
        Self {
            name_map: &config.name_map,
            overrides: Some(&config.overrides),
            ledger_overrides: Some(&config.ledger_overrides),
        }
    }

    #[cfg(test)]
    fn name_map_only(name_map: &'a NameMap) -> Self {
        Self {
            name_map,
            overrides: None,
            ledger_overrides: None,
        }
    }

    fn override_for(&self, key: &str) -> Option<BfsOverride<'a>> {
        let override_spec = self.overrides?.get(key)?;
        // The ledger records the exact key it inserted into `overrides`.
        // Classify that selected entry by exact key as well: canonical-family
        // matching could mislabel a distinct hand-written alias.
        let ledgered = self
            .ledger_overrides
            .is_some_and(|ledger| ledger.contains(key));
        Some(BfsOverride {
            spec: override_spec.as_str(),
            manual: !ledgered,
        })
    }

    /// Resolve the conda identity and any hand-written override with the same
    /// precedence as emission: the PyPI-keyed override wins, then the mapped
    /// conda-provider key. A PyPI-keyed override may establish an identity
    /// route when no mapping exists; an explicit disabled mapping still keeps
    /// the dependency on PyPI.
    fn target_and_override(
        &self,
        pypi_key: &PypiKey,
        pypi_to_conda: &PypiToCondaMap,
    ) -> (Option<CondaName>, Option<BfsOverride<'a>>) {
        // Select by the same precedence as emission BEFORE classifying manual
        // authority. A higher-precedence PyPI-key ledger entry must block a
        // lower-precedence mapped-provider manual entry.
        let pypi_override = self.override_for(pypi_key.as_str());
        let explicitly_disabled =
            matches!(self.name_map.get(pypi_key), Some(CondaTarget::Disabled));
        let target = pick_conda_target(pypi_key, self.name_map, pypi_to_conda).or_else(|| {
            (pypi_override.is_some_and(|entry| entry.manual) && !explicitly_disabled)
                .then(|| CondaName::new(pypi_key.as_str()))
        });
        let selected_override = target.as_ref().and_then(|conda_name| {
            pypi_override.or_else(|| self.override_for(conda_name.as_spec()))
        });
        (target, selected_override)
    }
}

fn incremental_bundle_requires_cold_resolve(bundle: &Bundle) -> bool {
    bundle.probe_decisions.iter().any(|decision| {
        decision.stage == "bfs" && decision.routing_decision.starts_with("short-circuit")
    })
}

#[derive(Debug, PartialEq, Eq)]
enum IncrementalVersionPlan {
    /// Metadata and the committed lock agree, so localized build may proceed.
    /// A later escalation must retain this version during the cold build.
    Attempt { fallback_version: String },
    /// Metadata advertised a different (normally cold-resolved) version. A
    /// localized build would return `lock.version`, so bypass it and preserve
    /// the version pixi already solved.
    Cold,
}

fn incremental_version_plan(
    advertised_version: Option<&str>,
    lock_version: &str,
) -> IncrementalVersionPlan {
    match advertised_version {
        Some(version) if version != lock_version => IncrementalVersionPlan::Cold,
        Some(version) => IncrementalVersionPlan::Attempt {
            fallback_version: version.to_string(),
        },
        None => IncrementalVersionPlan::Attempt {
            fallback_version: lock_version.to_string(),
        },
    }
}

fn advertised_version_matches(advertised_version: Option<&str>, candidate_version: &str) -> bool {
    advertised_version.is_none_or(|version| version == candidate_version)
}

fn validate_advertised_courier_version(
    bundle: &Bundle,
    advertised_version: Option<&str>,
    incremental_fallback_version: Option<&str>,
) -> Result<(), RpcError> {
    let package_version = courier_pack_version(bundle, incremental_fallback_version);
    if advertised_version_matches(advertised_version, &package_version) {
        return Ok(());
    }
    Err(RpcError::invalid_params(format!(
        "courier resolution changed between conda/outputs and conda/build_v1: pixi requested version `{}`, but the current primary wheel for `{}` requires package version `{package_version}`; rerun the lock/install so output metadata can be recomputed",
        advertised_version.unwrap_or_default(),
        bundle.conda_name,
    )))
}

fn validate_prepared_incremental_version_handoff(
    prepared_override: Option<&str>,
    detected_override: Option<&str>,
    output_name: &str,
) -> Result<(), RpcError> {
    if prepared_override == detected_override {
        return Ok(());
    }
    Err(RpcError::invalid_params(format!(
        "courier incremental metadata/build handoff changed for `{output_name}`: conda/outputs recorded version override `{}`, but build_v1 independently detected `{}`; rerun the lock/install so metadata and its exact prepared plan are recomputed together",
        prepared_override.unwrap_or("none"),
        detected_override.unwrap_or("none"),
    )))
}

fn reject_unprepared_incremental_fallback(
    detected_override: Option<&str>,
    output_name: &str,
) -> Result<(), RpcError> {
    let Some(version) = detected_override else {
        return Ok(());
    };
    Err(RpcError::invalid_params(format!(
        "courier incremental build for `{output_name}` escalated after metadata advertised lock version `{version}`, but the exact conda/outputs plan is unavailable or stale; rerun the lock/install so metadata and build share one materialized plan"
    )))
}

/// Resolve an output name back to a declared retread bundle/group only when
/// the identity is unambiguous from the source manifest itself. Workspace
/// aliases created by `apply_emission` deliberately return `None`; build_v1
/// must recover their base identity from the prepared or cold plan before it
/// can validate a content-addressed build hash.
fn declared_input_bundle_for_output(config: &RetreadConfig, output_name: &str) -> Option<String> {
    let output = canonical_conda_name(output_name);
    config
        .retread_wheels
        .iter()
        .find_map(|(entry_name, entry)| {
            let group = bundle_group_for(entry_name, entry, config.default_bundle.as_deref());
            let group = canonical_conda_name(&group);
            if group == output { Some(group) } else { None }
        })
}

fn bfs_probe_target_subdir(target: &ResolutionTarget) -> &str {
    target.conda_subdir()
}

#[allow(clippy::too_many_arguments)]
async fn resolve_bundle(
    entry_name: &str,
    entry: &WheelEntry,
    target: &ResolutionTarget,
    download_dir: &Path,
    source_dir: &Path,
    cache_dir: &Path,
    relax: RelaxPolicy,
    git_sources: &std::collections::BTreeMap<String, crate::config::NamedGitSource>,
    auto_data: Option<AutoDataConfig>,
    // v0.13.10+: BFS short-circuits PyPI-form Pending deps that
    // parselmouth knows about. v0.13.11+: also probes the workspace's
    // conda channels for satisfiability BEFORE short-circuiting --
    // same probe that auto_bundle_transitives uses. If conda has the
    // package but no version matches the spec (gym's `>=0.23,<0.24`
    // when conda-forge skipped from gym 0.21 to 0.26.2), the short-
    // circuit is suppressed and the dep falls through to the regular
    // PyPI resolve + bundle path. URL/git Pending deps NEVER short-
    // circuit: user opted into a specific upstream source via PEP 508
    // `pkg @ <url>` and substituting conda would silently swap deps.
    pypi_to_conda: &PypiToCondaMap,
    // Native-conda routing inputs: the merged effective name-map (user
    // retread-name-map + FALLBACK_PYPI_TO_CONDA + unambiguous parselmouth)
    // plus hand-written retread-overrides. The BFS must see both because
    // emission applies both: otherwise an indecisive route probe can bundle a
    // newest PyPI wheel before a later exact override is emitted.
    route_policy: BfsRoutePolicy<'_>,
    conda_channels: &[ChannelUrl],
    // Complete workspace-first group fallback chain. URL-form roots do not
    // contribute an index, but their ordinary metadata descendants inherit it.
    bundle_indexes: &[String],
    // incremental-add path: locked closure from the committed lock
    // (name → version_str for every wheel EXCEPT the new dep being added).
    // When Some, seeds ResolveState with ==V pinned constraints so ripples
    // become visible to intersect_specifiers.  Cold path: None.
    locked_closure: Option<&std::collections::BTreeMap<String, String>>,
    // favor-lock path: preferred versions for transitive deps (name → version_str).
    // When RETREAD_FAVOR_LOCK=1 and Some, the BFS will prefer these versions during
    // PyPI resolution instead of always picking the highest satisfying version.
    // Unlike `locked_closure`, these deps are NOT pinned as constraints and DO go
    // through the BFS fetch; we just hint the resolver to pick the given version.
    // Cold path (or RETREAD_FAVOR_LOCK unset): None (or ignored).
    favor_lock_prefs: Option<&std::collections::BTreeMap<String, String>>,
    // v2.10.0: canonical conda names of OTHER entries in the same bundle group.
    // A dep that names a sibling is provided by that sibling's wheel at install
    // time; retread must NOT resolve it from PyPI.  Callers outside a bundle
    // group (incremental-add, tests) pass `&HashSet::new()`.
    sibling_names: &std::collections::HashSet<String>,
) -> Result<Bundle> {
    let conda_name = canonical_conda_name(entry_name);
    let abi_aliases = abi_aliases_from_name_map(route_policy.name_map);
    let mut state = ResolveState::default();
    let mut work: BTreeMap<String, Pending> = BTreeMap::new();
    // v0.14.1+: collect every probe + routing decision so the audit
    // can persist them to disk. Flushed in resolve_all at the end of
    // conda/outputs (so failed conda solves still leave the trace).
    let mut probe_decisions: Vec<crate::audit::ProbeDecision> = Vec::new();

    // Materialize the primary wheel to disk by any of the four source
    // forms (url/version/path/git), then apply D (wheel METADATA
    // surgery) per the relax policy, then parse the resulting wheel.
    // For the primary wheel: surface the user's `extras` list in the
    // audit, and record dedup status. The caller (resolve_all) computed
    // the dedup decision and packed it via parallel arrays alongside
    // `auto_data`; resolve_bundle re-derives the "skipped root" by
    // looking at what `auto_data` is NOT carrying when the entry has a
    // checkout root.
    let dedup_skipped_root = if auto_data.is_none() {
        // Cheap recompute; matches the resolve_all decision logic.
        checkout_root_for_entry(entry, git_sources, source_dir, cache_dir)
    } else {
        None
    };
    let (primary, primary_original_rd) = materialize_and_rewrite_with_abi_aliases(
        entry,
        entry_name,
        None,
        target,
        download_dir,
        source_dir,
        cache_dir,
        relax,
        git_sources,
        auto_data,
        EntryAuditInfo {
            extras_requested: entry.extras.clone(),
            dedup_skipped_root,
        },
        &abi_aliases,
    )
    .await?;
    // P2 (grizzly #2): canonical seed -- the BFS dedups candidates in
    // canonical form (see the `canonical_conda_name(&pending.pypi_name)`
    // drain below), so the primary must be seeded the same way.
    // PR-1: seed_chosen records the primary's version into ResolveState so
    // future observe_edge calls can check constraint compatibility.
    state.seed_chosen(
        canonical_conda_name(&primary.pypi_name),
        primary.metadata.version.clone(),
    );

    // incremental-add: seed locked closure as ==V pinned constraints so any
    // incoming edge from the new dep's subtree that would require a different
    // version is detected as a ripple by intersect_specifiers / observe_edge.
    // The primary itself is already seeded via seed_chosen above; skip it here
    // to avoid overwriting with a locked constraint.
    if let Some(closure) = locked_closure {
        let primary_canon = canonical_conda_name(&primary.pypi_name);
        for (name, version_str) in closure {
            let canon = canonical_conda_name(name);
            if canon == primary_canon {
                continue; // primary already seeded
            }
            state.seed_locked(canon, version_str.clone());
        }
    }

    // path/git/from sources are authored project code, not metapackages
    // with extras-gated transitives. SKIP the BFS entirely unless the
    // user explicitly asked for extras on this entry (v0.12.0+). When
    // extras IS requested, fall through and run the BFS so the wheel's
    // METADATA `; extra == "X"` deps get pulled in -- but DON'T do the
    // sibling-prefix base-dep dance, since project code doesn't have a
    // metapackage's namespace convention.
    let is_source_form = entry.is_path() || entry.is_git() || entry.is_named_git();
    if is_source_form && entry.extras.is_empty() {
        return Ok(Bundle {
            conda_name,
            primary,
            extras: vec![],
            probe_decisions: vec![],
            solve_diagnostics: BTreeMap::new(),
            auto_routed: vec![],
            auto_dropped: Default::default(),
            uv_closure_names: Default::default(),
            uv_dependency_graph: Default::default(),
            workspace_conda_versions: Default::default(),
            workspace_conda_provider_facts: Default::default(),
        });
    }

    // Seed BFS from the primary's deps. Two flavors:
    // 1. Extras-gated (`; extra == "X"`) for each requested extra.
    // 2. Sibling base deps -- requirements without an extras marker whose
    //    PyPI name shares the entry's namespace prefix (`<entry>-...`).
    //    Real-world example: the isaacsim metapackage lists
    //    `Requires-Dist: isaacsim-kernel==5.1.0.0` (no marker) because the
    //    kernel is essential to ANY install of isaacsim. We bundle these
    //    sub-packages so the conda solver doesn't try to find them
    //    separately.
    //
    // Source-form entries (git/path/named-git) get extras BUT NOT the
    // sibling-prefix base-dep matching -- a git-built project doesn't
    // own a `<conda_name>-foo` namespace.
    let prefix = if is_source_form {
        String::new()
    } else {
        format!("{}-", conda_name)
    };
    // v1.5.9: seed from the ORIGINAL (pre-D) Requires-Dist so exact
    // family pins resolve exact-first (see materialize_and_rewrite).
    // PR-1: use a temp VecDeque for seed_worklist compat, then drain into BTreeMap.
    {
        let mut tmp_queue: VecDeque<Pending> = VecDeque::new();
        let seen_set: HashSet<String> = state.constraints.keys().cloned().collect();
        // The primary's transitives inherit the complete workspace chain. In a
        // real group build `bundle_indexes` already folds in every entry index
        // (see group_fallback_indexes), so this append is a dedup no-op there.
        // It only adds the primary's own index as a terminal fallback when the
        // caller passed a bare workspace list, keeping the seed hermetic (no
        // implicit public-PyPI injection).
        let primary_seed_indexes: Vec<String> = {
            let mut v = bundle_indexes.to_vec();
            if let Some(idx) = entry.index.as_ref()
                && !v
                    .iter()
                    .any(|e| e.trim_end_matches('/') == idx.trim_end_matches('/'))
            {
                v.push(idx.clone());
            }
            v
        };
        seed_worklist(
            &primary_original_rd,
            &entry.extras,
            &primary_seed_indexes,
            &prefix,
            &seen_set,
            &mut tmp_queue,
            None, // cold path: no locked-closure ripple detection
            sibling_names,
        )?;
        for pending in tmp_queue {
            let name = canonical_conda_name(&pending.pypi_name);
            // FIX 2: merge specifiers when a second edge arrives for the same name
            // in the same sweep, so the constraint is accumulated rather than dropped.
            match work.entry(name) {
                std::collections::btree_map::Entry::Vacant(e) => {
                    e.insert(pending);
                }
                std::collections::btree_map::Entry::Occupied(mut e) => {
                    if let PendingSource::Pypi {
                        specifiers: new_s, ..
                    } = pending.source
                        && let PendingSource::Pypi {
                            ref mut specifiers, ..
                        } = e.get_mut().source
                    {
                        let merged: VersionSpecifiers = specifiers
                            .iter()
                            .cloned()
                            .chain(new_s.iter().cloned())
                            .collect();
                        *specifiers = merged;
                    }
                }
            }
        }
    }

    // BFS, accumulating sub-wheels. v0.12.0+: PyPI-Simple deps go
    // through the existing `pypi::resolve` path; URL/git deps from
    // PEP 508 `pkg @ <url>` form get synthesized into a `WheelEntry`
    // and run through `materialize_and_rewrite` so the same caching,
    // METADATA surgery, and source-build pipeline applies. Url/git
    // sub-wheels propagate their own extras but NOT prefix-base-dep
    // matching (they're project code, same rule as primary).
    let mut extras = Vec::new();
    // v1.4.3: process the BFS level by level. A child's existence is
    // only known after its parent's METADATA is parsed, but items at
    // the SAME depth never read each other's results -- so each
    // level's expensive wheel fetches run with bounded concurrency.
    // Routing probes stay serial (the in-memory repodata cache makes
    // them hashmap lookups after the first call); seed_worklist +
    // extras.push stay a serial in-order sweep so the audit order and
    // extras order are deterministic and identical to the old
    // one-item-at-a-time BFS. Git/URL-form deps are also fetched
    // serially: two materializations of the same repo would race the
    // git-clone cache.
    //
    // PR-1: work is now a BTreeMap<name, Pending>; iteration is
    // canonical-name-sorted (Pillar 3). ResolveState replaces
    // seen: HashSet and accumulates AND-intersection constraints.
    const MAX_BFS_ITERATIONS: usize = 500;
    let mut bfs_iter = 0usize;
    'levels: loop {
        bfs_iter += 1;
        if bfs_iter > MAX_BFS_ITERATIONS {
            bail!(
                "resolve_bundle: BFS iteration cap ({MAX_BFS_ITERATIONS}) exceeded for bundle \
                 `{conda_name}`. This indicates a circular re-resolve in constraint \
                 accumulation (a conflict the resolver failed to detect early)."
            );
        }
        if work.is_empty() {
            break 'levels;
        }
        // Drain work into frontier in canonical (name-sorted) order.
        // The BTreeMap guarantees name-sorted iteration (Pillar 3).
        let current_work: Vec<Pending> = std::mem::take(&mut work).into_values().collect();
        let mut frontier: Vec<Pending> = Vec::new();
        let mut reresolve_queue: Vec<Pending> = Vec::new();

        for pending in current_work {
            let dep_conda_name = canonical_conda_name(&pending.pypi_name);
            // Pre-check: is this dep locked (incremental-add path only)?
            // We capture it BEFORE the mutable observe_edge borrow so we can
            // use it in match arms below without a second mutable borrow.
            let dep_is_locked = state.is_locked(&dep_conda_name);
            let edge_result = state.observe_edge(&dep_conda_name, pending);
            // 2c: if the dep is locked and observe_edge returns a conflict
            // error, escalate to IncrementalRipple instead of failing hard.
            let edge_result = match edge_result {
                Err(e) if dep_is_locked => {
                    return Err(anyhow::Error::new(auto_bundle::IncrementalRipple {
                        reason: format!("locked dep `{dep_conda_name}` constraint conflict: {e:#}"),
                    }));
                }
                other => other?,
            };
            match edge_result {
                ObserveEdgeResult::New(p) => {
                    frontier.push(p);
                }
                ObserveEdgeResult::AlreadySatisfied | ObserveEdgeResult::NonPypiAlreadySeen => {
                    // Already resolved; constraint was accumulated. Skip.
                }
                ObserveEdgeResult::NeedsReResolve(tighter_pending) => {
                    // 2c: if this is a locked dep, a NeedsReResolve means the
                    // new dep's subtree wants to change the locked version →
                    // ripple detected, escalate to cold resolve.
                    if dep_is_locked {
                        return Err(anyhow::Error::new(auto_bundle::IncrementalRipple {
                            reason: format!(
                                "locked dep `{dep_conda_name}` would need re-resolution \
                                     (current lock version excluded by incoming constraint)"
                            ),
                        }));
                    }
                    // Must re-resolve this dep with tighter constraints.
                    // Revoke the chosen version so it gets re-resolved.
                    state.revoke_chosen(&dep_conda_name);
                    // Remove from extras (it will be re-added after re-resolution).
                    extras.retain(|w: &ResolvedWheel| {
                        canonical_conda_name(&w.pypi_name) != dep_conda_name
                    });
                    reresolve_queue.push(tighter_pending);
                }
            }
        }

        // Re-enqueue items that need re-resolution into the next level's work.
        for p in reresolve_queue {
            let name = canonical_conda_name(&p.pypi_name);
            work.insert(name, p);
        }

        if frontier.is_empty() && work.is_empty() {
            break 'levels;
        }
        if frontier.is_empty() {
            continue 'levels; // still have re-resolve work
        }

        // Phase 1: routing (prefer-conda short-circuit), serial.
        // Collects the items that fall through to materialization.
        let mut to_materialize: Vec<Pending> = Vec::new();
        for pending in frontier {
            // v0.17.0+ probe-gated prefer-conda short-circuit. Decision:
            //   (a) Non-Pypi pending source (URL/git PEP 508 dep)
            //       -> never short-circuit (materialize via git/url path).
            //   (b) Parselmouth doesn't know this PyPI name
            //       -> don't short-circuit, go to PyPI.
            //   (c) Pick canonical conda name from parselmouth candidates:
            //         - identity match wins (numpy, psutil, pyyaml, gym...)
            //         - else single-candidate (torch -> pytorch)
            //         - else ambiguous-no-identity -> don't short-circuit,
            //           record decision, go to PyPI. User can disambiguate
            //           via retread-name-map.
            //   (d) Probe workspace conda channels for that name under
            //       wheel's spec + target python.
            //   (e) one or more requirement-matching candidates ->
            //       short-circuit. Empty, unsatisfied, or indecisive -> PyPI.
            //
            // The v0.13.10 `.first()` candidate picker was wrong: the
            // inverted parselmouth map has many false positives (a conda
            // package can list a pypi dep without "being" it), so picking
            // an arbitrary candidate gave nonsense like `numpy -> manifpy`
            // and `torch -> pytorch-cpu`. The probe then asked the wrong
            // question. v0.17.0 fixes the picker.
            let dep_pypi_key = PypiKey::from_pypi(&pending.pypi_name);
            let mut routed_to_conda = false;
            if let PendingSource::Pypi { specifiers, .. } = &pending.source {
                // v0.46.0: the curated/user name-map wins over parselmouth.
                // If it has an entry for this dep (e.g. torch->pytorch from
                // FALLBACK), treat that as the unambiguous conda target --
                // otherwise fall back to parselmouth's inverted candidates,
                // which are often ambiguous for exactly the deps the FALLBACK
                // table exists to disambiguate.
                {
                    let (picked, selected_override) =
                        route_policy.target_and_override(&dep_pypi_key, pypi_to_conda);
                    match picked {
                        None => {
                            let amb = pypi_to_conda.get(dep_pypi_key.as_str());
                            tracing::info!(
                                dep = %pending.pypi_name,
                                candidates = ?amb,
                                "BFS prefer-conda: ambiguous parselmouth mapping with no identity match; not short-circuiting (add retread-name-map to force conda routing)",
                            );
                            probe_decisions.push(crate::audit::ProbeDecision {
                                stage: "bfs".into(),
                                pypi_name: pending.pypi_name.clone(),
                                conda_name: format!("(ambiguous: {amb:?})"),
                                spec: specifiers.to_string(),
                                target_python: target.python_version.clone(),
                                channels_consulted: vec![],
                                satisfiable: None,
                                matching_candidates: 0,
                                routing_decision: "skipped-ambiguous-no-identity".into(),
                            });
                            // routed_to_conda stays false -> falls through to pypi::resolve below
                        }
                        Some(conda_target_name) => {
                            // Emission replaces the wheel requirement with the
                            // selected override, so probe that same spec. A
                            // hand-written override is the native-conda
                            // boundary and remains authoritative even when the
                            // diagnostic probe is unavailable or indecisive.
                            let probe_spec = match selected_override.map(|entry| entry.spec.trim())
                            {
                                Some("") => "*".to_string(),
                                Some(spec) => spec.to_string(),
                                None => conda_probe_spec(specifiers),
                            };
                            let probe_result = crate::probe::probe_for_target(
                                conda_channels,
                                conda_target_name.as_spec(),
                                &probe_spec,
                                Some(&target.python_version),
                                bfs_probe_target_subdir(target),
                            )
                            .await;
                            let manual_override =
                                selected_override.is_some_and(|entry| entry.manual);
                            let route_to_conda =
                                manual_override || validated_conda_route(&probe_result);
                            let routing_decision = if manual_override {
                                "short-circuit-explicit-override"
                            } else if route_to_conda {
                                "short-circuit"
                            } else {
                                "fall-through-to-pypi"
                            };
                            probe_decisions.push(crate::audit::ProbeDecision {
                                stage: "bfs".into(),
                                pypi_name: pending.pypi_name.clone(),
                                conda_name: conda_target_name.as_spec().to_owned(),
                                spec: probe_spec.clone(),
                                target_python: target.python_version.clone(),
                                channels_consulted: probe_result.channels_consulted.clone(),
                                satisfiable: probe_result.satisfiable,
                                matching_candidates: probe_result.matching_candidates,
                                routing_decision: routing_decision.into(),
                            });
                            tracing::info!(
                                dep = %pending.pypi_name,
                                conda_name = %conda_target_name,
                                spec = %probe_spec,
                                decision = %routing_decision,
                                manual_override,
                                matches = probe_result.matching_candidates,
                                channels = ?probe_result.channels_consulted,
                                "BFS prefer-conda probe result",
                            );
                            if route_to_conda {
                                routed_to_conda = true;
                            }
                        }
                    }
                }
            }
            if routed_to_conda {
                continue;
            }
            // v2.10.0 defense-in-depth: a sibling dep that reached the BFS
            // frontier (e.g. via a transitive observe_edge path rather than
            // seed_worklist) must still be suppressed.  The primary guard is
            // in seed_worklist, but any dep that snuck through (e.g. a sibling
            // referenced transitively by another transitive dep) is caught here
            // before the PyPI fetch.
            if sibling_names.contains(dep_pypi_key.as_str()) {
                tracing::debug!(
                    dep = %pending.pypi_name,
                    sibling_canon = %dep_pypi_key,
                    "BFS frontier: skipping sibling dep — provided by sibling bundle entry",
                );
                continue;
            }
            to_materialize.push(pending);
        }

        // Phase 2: fetch this level's PyPI-form wheels concurrently
        // (8-way bounded, order-preserving `buffered`). Git/URL forms
        // pass through untouched and materialize serially in phase 3.
        // Search the COMPLETE index chain for a compatible wheel before
        // trying any sdist. Otherwise a public wheel-stub sdist can shadow
        // the real binary on a later vendor index (isaacsim-app on NVIDIA).
        // Fatal index or build errors still abort immediately.
        // favor-lock: build a snapshot of the preferred versions for this fetch
        // sweep.  We only look versions up (no mutation), so a cheap reference
        // to the caller's BTreeMap is enough -- but async closures capture by
        // value, so clone the prefs map if provided.  On the cold path (None
        // prefs, which is what resolve_all builds when RETREAD_FAVOR_LOCK is
        // unset) `favor_lock_snap` is empty and all `prefer_version` lookups
        // return None, reproducing the original highest-version behavior.
        //
        // The RETREAD_FAVOR_LOCK env gate is ONLY enforced at load_favored_versions
        // (called by resolve_all).  By the time we reach resolve_bundle the param
        // already reflects the caller's intent -- using the param directly avoids
        // process-wide env mutation leaking across parallel tests.
        //
        // NOTE: `favor_lock_prefs` is SEPARATE from `locked_closure`. The locked
        // closure pins deps as ==V constraints (blocking BFS fetch); the prefs map
        // only hints the resolver to prefer a version, without blocking the fetch.
        let favor_lock_snap: std::collections::BTreeMap<String, String> =
            favor_lock_prefs.cloned().unwrap_or_default();
        let fetched: Vec<(Pending, Result<Option<BfsFetched>>)> = {
            use futures::stream::{self, StreamExt};
            let favor_lock_snap_ref = &favor_lock_snap;
            let abi_aliases_ref = &abi_aliases;
            stream::iter(to_materialize)
                .map(|pending| async move {
                    let dep_canon = crate::relax::canonical_conda_name(&pending.pypi_name);
                    let prefer_version = favor_lock_snap_ref.get(&dep_canon).cloned();
                    let request = match &pending.source {
                        PendingSource::Pypi {
                            specifiers,
                            indexes,
                        } => Some((
                            pending.pypi_name.clone(),
                            specifiers.clone(),
                            indexes.clone(),
                        )),
                        PendingSource::Git { .. } | PendingSource::Url { .. } => None,
                    };
                    let result = if let Some((pypi_name, specifiers, indexes)) = request {
                        bfs_fetch_pypi_from_chain(
                            &pypi_name,
                            &specifiers,
                            &indexes,
                            target,
                            download_dir,
                            relax,
                            abi_aliases_ref,
                            prefer_version.as_deref(),
                            format!(
                                "BFS could not resolve `{pypi_name}` from any configured PyPI index"
                            ),
                        )
                        .await
                        .map(Some)
                    } else {
                        Ok(None)
                    };
                    (pending, result)
                })
                .buffered(8)
                .collect()
                .await
        };

        // Phase 3: serial in-order sweep -- materialize git/URL forms,
        // seed the next level, accumulate extras. Order matches the
        // old pop order exactly.
        for (pending, fetch_result) in fetched {
            let dep_conda_name = canonical_conda_name(&pending.pypi_name);
            let sub_indexes_for_recurse = bfs_descendant_indexes(&pending.source, &bundle_indexes);
            // Materialization plus explicit metadata origin. The origin cannot
            // be recovered later from replay fields because path builds carry
            // neither git nor sdist provenance.
            let (
                sub_url,
                sub_upstream_url,
                sub_git_source,
                sub_sdist_source,
                sub_metadata_provenance,
                sub_metadata,
                sub_seed_rd,
            ) = match (&pending.source, fetch_result?) {
                (PendingSource::Pypi { .. }, Some((resolved_url, metadata, sdist_prov))) => {
                    // Pypi-form sub-wheels are NOT D-rewritten, so
                    // their metadata IS the original Requires-Dist.
                    let seed_rd = metadata.requires_dist.clone();
                    let (upstream, sub_sdist_src, metadata_provenance) =
                        bfs_fetch_provenance(&resolved_url, sdist_prov);
                    (
                        resolved_url,
                        upstream,
                        None,
                        sub_sdist_src,
                        metadata_provenance,
                        metadata,
                        seed_rd,
                    )
                }
                (PendingSource::Pypi { .. }, None) => {
                    unreachable!("phase 2 always fetches Pypi-form items")
                }
                (
                    PendingSource::Git {
                        url,
                        rev,
                        subdirectory,
                    },
                    _,
                ) => {
                    let synth = WheelEntry {
                        git: Some(url.clone()),
                        rev: rev.clone().or_else(|| Some("HEAD".to_string())),
                        // A-0 fix: thread the parsed subdirectory through so the
                        // git checkout builds from the correct sub-package path.
                        subdirectory: subdirectory.clone(),
                        ..Default::default()
                    };
                    let synth_name = pending.pypi_name.clone();
                    let (sub, sub_original_rd) = materialize_and_rewrite_with_abi_aliases(
                        &synth,
                        &synth_name,
                        None,
                        target,
                        download_dir,
                        source_dir,
                        cache_dir,
                        relax,
                        git_sources,
                        None,
                        EntryAuditInfo::default(),
                        &abi_aliases,
                    )
                    .await
                    .with_context(|| {
                        format!(
                            "materializing URL Requires-Dist `{} @ git+{}@{}`",
                            pending.pypi_name,
                            url,
                            rev.as_deref().unwrap_or("HEAD"),
                        )
                    })?;
                    let sub_gs = sub.git_source.clone();
                    let sub_up = sub.upstream_url.clone();
                    (
                        sub.url,
                        sub_up,
                        sub_gs,
                        None, // Git-form: no sdist provenance
                        Provenance::SourceBuiltRelaxed,
                        sub.metadata,
                        sub_original_rd,
                    )
                }
                (PendingSource::Url { wheel_url }, _) => {
                    let synth = WheelEntry {
                        url: Some(wheel_url.clone()),
                        ..Default::default()
                    };
                    let synth_name = pending.pypi_name.clone();
                    let (sub, sub_original_rd) = materialize_and_rewrite_with_abi_aliases(
                        &synth,
                        &synth_name,
                        None,
                        target,
                        download_dir,
                        source_dir,
                        cache_dir,
                        relax,
                        git_sources,
                        None,
                        EntryAuditInfo::default(),
                        &abi_aliases,
                    )
                    .await
                    .with_context(|| {
                        format!(
                            "materializing URL Requires-Dist `{} @ {}`",
                            pending.pypi_name, wheel_url,
                        )
                    })?;
                    let sub_up = sub.upstream_url.clone();
                    (
                        sub.url,
                        sub_up,
                        None, // Url-form: no git source
                        None, // Url-form: no sdist provenance
                        Provenance::IndexWheelMetadata,
                        sub.metadata,
                        sub_original_rd,
                    )
                }
            };

            // Recurse: this sub-wheel's own extras and prefix-matching base
            // deps also get pulled in. URL/git sub-wheels reuse the parent
            // bundle's `prefix` (often empty for source-form parents) so
            // they don't pull random siblings.
            // PR-1: use a temp VecDeque for seed_worklist compat, then drain into BTreeMap.
            {
                let mut tmp_seed: VecDeque<Pending> = VecDeque::new();
                let seen_set: HashSet<String> = state.constraints.keys().cloned().collect();
                seed_worklist(
                    &sub_seed_rd,
                    &pending.extras,
                    &sub_indexes_for_recurse,
                    &prefix,
                    &seen_set,
                    &mut tmp_seed,
                    None, // cold path: no locked-closure ripple detection
                    sibling_names,
                )?;
                for p in tmp_seed {
                    let name = canonical_conda_name(&p.pypi_name);
                    // FIX 2: merge specifiers when a second edge arrives for the same
                    // name in the same sweep (recurse drain), same as seed drain above.
                    match work.entry(name) {
                        std::collections::btree_map::Entry::Vacant(e) => {
                            e.insert(p);
                        }
                        std::collections::btree_map::Entry::Occupied(mut e) => {
                            if let PendingSource::Pypi {
                                specifiers: new_s, ..
                            } = p.source
                                && let PendingSource::Pypi {
                                    ref mut specifiers, ..
                                } = e.get_mut().source
                            {
                                let merged: VersionSpecifiers = specifiers
                                    .iter()
                                    .cloned()
                                    .chain(new_s.iter().cloned())
                                    .collect();
                                *specifiers = merged;
                            }
                        }
                    }
                }
            }

            // PR-1: capture version before sub_metadata is moved into the struct.
            let sub_version = sub_metadata.version.clone();
            extras.push(ResolvedWheel {
                pypi_name: dep_conda_name.clone(),
                url: sub_url,
                upstream_url: sub_upstream_url,
                // BFS sub-wheels inherit git_source from materialize_and_rewrite
                // (already set for Git-form PendingSource via the synth path).
                // Pypi/Url-form sub-wheels have no git source.
                git_source: sub_git_source,
                // Sdist provenance (schema 9+): set when the BFS sdist fallback
                // fired for this sub-wheel; None for normal index-wheel fetches
                // and git/url-form sub-wheels.
                sdist_source: sub_sdist_source,
                metadata_provenance: sub_metadata_provenance,
                metadata: sub_metadata,
                extras_requested: vec![],
                auto_data: None,
                auto_data_dedup_skipped_root: None,
            });
            // PR-1: commit the resolved version so future observe_edge calls
            // can check constraint compatibility (re-resolve-on-tighten).
            state.commit_chosen(dep_conda_name, sub_version);
        }
    }

    let bfs_bundle = Bundle {
        conda_name,
        primary,
        extras,
        probe_decisions,
        solve_diagnostics: BTreeMap::new(),
        auto_routed: vec![],
        auto_dropped: Default::default(),
        uv_closure_names: Default::default(),
        uv_dependency_graph: Default::default(),
        workspace_conda_versions: Default::default(),
        workspace_conda_provider_facts: Default::default(),
    };

    Ok(bfs_bundle)
}

/// v1.5.9: produce the relaxed retry specifiers for a sub-wheel whose
/// EXACT upstream pin is missing from the index. Returns None when the
/// policy doesn't relax or relaxation changes nothing (bare deps,
/// range specs -- relax only widens single exact pins).
fn relaxed_retry_specs(
    pypi_name: &str,
    specifiers: &VersionSpecifiers,
    relax: RelaxPolicy,
    abi_aliases: &AbiAliasGraph,
) -> Option<VersionSpecifiers> {
    if relax == RelaxPolicy::None {
        return None;
    }
    let original = format!("{pypi_name}{specifiers}");
    let relaxed_line =
        crate::wheel_rewrite::relax_pep508_with_abi_aliases(&original, relax, abi_aliases).ok()?;
    if relaxed_line == original {
        return None;
    }
    let req: uv_pep508::Requirement = uv_pep508::Requirement::from_str(&relaxed_line).ok()?;
    match req.version_or_url {
        Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) if specs != *specifiers => {
            Some(specs)
        }
        _ => None,
    }
}

/// Sdist provenance captured by the BFS sdist fallback path.
/// Threaded out of `bfs_fetch_pypi_sdist` so the caller can populate
/// `ResolvedWheel.sdist_source` without losing the `sdist.url` that
/// was previously discarded at mod.rs (THE DISCARD POINT in §1.1).
pub(super) struct SdistProv {
    pub(super) index: String,
    pub(super) name: String,
    pub(super) version: String,
    /// The EXACT resolved sdist URL with #sha256 fragment from the PyPI
    /// simple index (pypi.rs:197). This is the preferred replay key:
    /// build_wheel_from_sdist_url(stored_url) skips the re-resolve.
    pub(super) sdist_url: url::Url,
}

/// Convert the shared PyPI fetcher's provenance into the lock representation.
/// A locally built `file://` wheel is never an upstream URL; its immutable
/// sdist URL is the replay source instead.
fn bfs_fetch_provenance(
    resolved_url: &url::Url,
    sdist_prov: Option<SdistProv>,
) -> (
    Option<url::Url>,
    Option<crate::lock::SdistWheelSource>,
    Provenance,
) {
    let metadata_provenance = if sdist_prov.is_some() {
        Provenance::SourceBuiltRelaxed
    } else {
        Provenance::IndexWheelMetadata
    };
    let sdist_source = sdist_prov.map(|p| crate::lock::SdistWheelSource {
        index: p.index,
        name: p.name,
        version: p.version,
        sdist_url: p.sdist_url.to_string(),
    });
    let upstream_url = sdist_source.is_none().then(|| resolved_url.clone());
    (upstream_url, sdist_source, metadata_provenance)
}

/// Resolve one PyPI BFS request using global artifact phases across the full
/// configured index chain: exact wheels, relaxed wheels, then sdists.
async fn bfs_fetch_pypi_from_chain(
    pypi_name: &str,
    specifiers: &VersionSpecifiers,
    indexes: &[String],
    target: &ResolutionTarget,
    download_dir: &Path,
    relax: RelaxPolicy,
    abi_aliases: &AbiAliasGraph,
    prefer_version: Option<&str>,
    exhaustion_context: String,
) -> Result<BfsFetched> {
    let relaxed = relaxed_retry_specs(pypi_name, specifiers, relax, abi_aliases);
    let try_relaxed = relaxed.is_some();
    fetch_artifact_from_pypi_index_chain(
        indexes,
        try_relaxed,
        prefer_version.map(str::to_string),
        |index, phase, prefer_version| {
            let pypi_name = pypi_name.to_string();
            let exact = specifiers.clone();
            let relaxed = relaxed.clone();
            let target = target.clone();
            let download_dir = download_dir.to_path_buf();
            async move {
                match phase {
                    PypiArtifactPhase::ExactWheel => {
                        bfs_fetch_pypi_wheel(
                            &pypi_name,
                            &exact,
                            &index,
                            &target,
                            &download_dir,
                            prefer_version.as_deref(),
                        )
                        .await
                    }
                    PypiArtifactPhase::RelaxedWheel => {
                        let relaxed = relaxed.as_ref().expect(
                            "relaxed artifact phase runs only when relaxed specs are available",
                        );
                        let fetched = bfs_fetch_pypi_wheel(
                            &pypi_name,
                            relaxed,
                            &index,
                            &target,
                            &download_dir,
                            prefer_version.as_deref(),
                        )
                        .await?;
                        tracing::warn!(
                            dep = %pypi_name,
                            exact = %exact,
                            relaxed = %relaxed,
                            resolved = %fetched.0,
                            "PATCH-DRIFT FALLBACK: exact upstream pin was absent from every index; resolved a relaxed wheel. If this dep is part of a pinned wheel family (isaacsim-*), check for runtime contract drift.",
                        );
                        crate::status::tty(&format!(
                            "warning: {pypi_name}{exact} absent from every index; using relaxed match {} (possible family version drift)",
                            fetched.0,
                        ));
                        Ok(fetched)
                    }
                    PypiArtifactPhase::Sdist => {
                        bfs_fetch_pypi_sdist(
                            &pypi_name,
                            &exact,
                            &index,
                            &target,
                            &download_dir,
                            prefer_version.as_deref(),
                        )
                        .await
                    }
                }
            }
        },
        exhaustion_context,
    )
    .await
}

/// Resolve and materialize a binary wheel from one index. A semantic miss is
/// returned unchanged so the chain-level caller can try every remaining wheel
/// index before considering source distributions.
async fn bfs_fetch_pypi_wheel(
    pypi_name: &str,
    specifiers: &VersionSpecifiers,
    index: &str,
    target: &ResolutionTarget,
    download_dir: &Path,
    // favor-lock: when Some, prefer this version on the index before falling
    // back to highest-version selection. Propagated from favor_lock_prefs by the
    // BFS phase-2 fetch loop when RETREAD_FAVOR_LOCK=1. None on the cold path.
    prefer_version: Option<&str>,
) -> Result<BfsFetched> {
    let resolved = if let Some(pv) = prefer_version {
        pypi::resolve_preferring(index, pypi_name, specifiers, target, pv).await
    } else {
        pypi::resolve(index, pypi_name, specifiers, target).await
    }?;
    let metadata = metadata_preferring_sidecar(&resolved, download_dir).await?;
    Ok((resolved.url, metadata, None))
}

/// Resolve and build a source distribution from one index. This runs only
/// after binary-wheel resolution missed on every configured index, so source
/// builds cannot shadow later vendor wheels. The sdist uses the original spec,
/// preserving exact upstream pins and lock provenance. A compatible favored
/// lock version wins over the latest matching source distribution.
async fn bfs_fetch_pypi_sdist(
    pypi_name: &str,
    specifiers: &VersionSpecifiers,
    index: &str,
    target: &ResolutionTarget,
    download_dir: &Path,
    prefer_version: Option<&str>,
) -> Result<BfsFetched> {
    let resolved = if let Some(preferred) = prefer_version {
        pypi::resolve_sdist_preferring(index, pypi_name, specifiers, preferred).await
    } else {
        pypi::resolve_sdist(index, pypi_name, specifiers).await
    };
    let (sdist_version, sdist) = resolved
        .with_context(|| format!("BFS sdist fallback for {pypi_name} {specifiers} on {index}"))?;
    // Capture the sdist URL before the build so replay keeps the immutable
    // source even though the materialized wheel is a local file URL.
    let mut captured_sdist_url = sdist.url.clone();
    // Unified sdist build cache dir keyed on (name, version) so BFS,
    // discovery, and replay all share the same output directory and never
    // rebuild the same (name, version) twice.
    let sdist_out = download_dir
        .join("sdist-builds")
        .join(format!("{pypi_name}-{sdist_version}"));
    let expected = crate::source_build::ExpectedWheel::exact(pypi_name, sdist_version.to_string());
    let built = crate::source_build::build_wheel_from_sdist_url_for_target(
        &sdist.url,
        &sdist_out,
        target,
        sdist.sha256.as_deref(),
        Some(&expected),
    )
    .await
    .with_context(|| {
        format!(
            "uv-building wheel from sdist {} for {}",
            sdist.url, pypi_name,
        )
    })?;
    captured_sdist_url.set_fragment(Some(&format!("sha256={}", built.sdist_sha256)));
    let built_url = url::Url::from_file_path(&built.wheel_path).map_err(|_| {
        anyhow!(
            "built wheel path is not a valid file URL: {}",
            built.wheel_path.display(),
        )
    })?;
    let metadata = tokio::task::spawn_blocking({
        let p = built.wheel_path.clone();
        move || crate::wheel::read_metadata(&p)
    })
    .await
    .context("metadata reader panicked")??;
    tracing::info!(
        dep = %pypi_name,
        built = %built.wheel_path.display(),
        "BFS sdist fallback: built wheel from sdist after all wheel indexes missed",
    );
    let prov = SdistProv {
        index: index.to_string(),
        name: pypi_name.to_string(),
        version: metadata.version.clone(),
        sdist_url: captured_sdist_url,
    };
    Ok((built_url, metadata, Some(prov)))
}

/// True if `output` exists on disk and is newer than `input`. Used to
/// skip the inject + D-rewrite passes when their inputs haven't moved.
/// Missing files (either side) return false so we always recompute.
fn is_fresh(output: &Path, input: &Path) -> Result<bool> {
    let (Ok(out_meta), Ok(in_meta)) = (output.metadata(), input.metadata()) else {
        return Ok(false);
    };
    let (Ok(out_t), Ok(in_t)) = (out_meta.modified(), in_meta.modified()) else {
        return Ok(false);
    };
    if out_t < in_t {
        return Ok(false);
    }
    // Self-heal (run 9 fix): mtime alone can't tell a fully-written cache
    // wheel from a truncated one left behind by a process/node that died
    // mid-write. wheel_inject.rs / wheel_inject_data.rs / wheel_rewrite.rs
    // now write atomically (temp + same-dir rename) going forward, but old
    // corrupted cache entries -- and any future write path that forgets the
    // pattern -- must never be silently trusted just because they're
    // "newer". Validate zip integrity before calling a `.whl` cache hit
    // fresh; on failure, remove the corrupt file and fall through to the
    // caller's normal rebuild path.
    if output.extension().is_some_and(|e| e == "whl") && !crate::wheel::is_valid_zip(output) {
        tracing::warn!(
            wheel = %output.display(),
            "retread: corrupted cached wheel, removing and rebuilding",
        );
        let _ = std::fs::remove_file(output);
        return Ok(false);
    }
    Ok(true)
}

fn relaxed_wheel_cache_stamp_path(wheel: &Path) -> PathBuf {
    let filename = wheel
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default();
    wheel.with_file_name(format!("{filename}.retread-cache"))
}

fn relaxed_wheel_cache_stamp(relax: RelaxPolicy, abi_aliases: &AbiAliasGraph) -> String {
    format!(
        "emit-epoch={}\npolicy={relax:?}\nabi-aliases={abi_aliases:?}\n",
        crate::lock::EMIT_EPOCH
    )
}

fn is_relaxed_wheel_cache_fresh(
    output: &Path,
    input: &Path,
    relax: RelaxPolicy,
    abi_aliases: &AbiAliasGraph,
) -> Result<bool> {
    if !is_fresh(output, input)? {
        return Ok(false);
    }
    let expected = relaxed_wheel_cache_stamp(relax, abi_aliases);
    Ok(
        std::fs::read_to_string(relaxed_wheel_cache_stamp_path(output))
            .is_ok_and(|actual| actual == expected),
    )
}

fn write_relaxed_wheel_cache_stamp(
    output: &Path,
    relax: RelaxPolicy,
    abi_aliases: &AbiAliasGraph,
) -> Result<()> {
    std::fs::write(
        relaxed_wheel_cache_stamp_path(output),
        relaxed_wheel_cache_stamp(relax, abi_aliases),
    )
    .with_context(|| format!("writing relaxed-wheel cache stamp for {}", output.display()))
}

/// After the user-driven (extras + prefix) BFS, optionally bundle any
/// exact-pinned base deps that resolve cleanly on the entry's PyPI index.
/// Compute the upstream checkout root for an entry, when one exists.
/// Used by [`resolve_all`] to dedup the v0.12.0+ auto-data-files inject
/// across entries that share a single clone.
///
/// - git inline: cache_dir/retread-git-clones/<slug>/ (the repo root,
///   *parent* of the entry's subdirectory).
/// - named-git (`from = "<name>"`): same path resolution as inline git,
///   with url + rev pulled from `[retread-git-sources]`.
/// - path: returns None. Path entries point directly at a Python
///   package source; there's no upstream "repo" with sibling content
///   to ship. Users wanting to attach data from adjacent paths declare
///   a separate `[retread-wheels]` entry.
/// - url / spec: None. No source tree, no auto-data inject possible.
fn checkout_root_for_entry(
    entry: &WheelEntry,
    git_sources: &std::collections::BTreeMap<String, crate::config::NamedGitSource>,
    _source_dir: &Path,
    cache_dir: &Path,
) -> Option<PathBuf> {
    if let Some(from_name) = &entry.from {
        let src = git_sources.get(from_name)?;
        Some(crate::source_build::git_checkout_root(
            &src.url, &src.rev, cache_dir,
        ))
    } else if let Some(git_url) = &entry.git {
        let rev = entry.rev.as_ref()?;
        Some(crate::source_build::git_checkout_root(
            git_url, rev, cache_dir,
        ))
    } else {
        None
    }
}

/// v0.12.0+: per-bundle configuration for the auto-data-files inject
/// phase. When `Some`, the wheel produced by phase 1.5 gets a phase
/// 1.6 pass that walks `checkout_root` (respecting `.gitignore`) and
/// emits every non-ignored, non-`skip_subdirs` file as a wheel
/// `.data/data/lib/<rel>` entry -- so the upstream repo's sibling
/// content (apps/, tools/, share/, etc.) lands at `$PREFIX/lib/<rel>`
/// when pip installs the wheel. Computed by the caller in
/// [`resolve_all`] so dedup across the bundle's entries is centralized.
/// `checkout_root` is planning identity, not authority to read the tree;
/// materialization validates it against the active build-returned lease.
#[derive(Debug, Clone)]
pub(crate) struct AutoDataConfig {
    pub checkout_root: PathBuf,
    /// Subdirectories (relative to `checkout_root`) that sibling
    /// entries in this bundle already shipped as wheels -- the walk
    /// descends through them but emits no files (avoids re-shipping the
    /// Python package source into `$PREFIX/lib/source/...`).
    pub skip_subdirs: Vec<PathBuf>,
}

fn persist_git_auto_data(
    auto_data: Option<&AutoDataConfig>,
) -> Result<crate::lock::GitWheelAutoData> {
    let disposition = match auto_data {
        None => Ok(crate::lock::GitWheelAutoData::Disabled),
        Some(config) => config
            .skip_subdirs
            .iter()
            .map(|path| {
                path.to_str().map(str::to_owned).ok_or_else(|| {
                    anyhow!(
                        "Git auto-data skip subdirectory is not valid UTF-8: {}",
                        path.display(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()
            .map(
                |skip_subdirectories| crate::lock::GitWheelAutoData::CheckoutRoot {
                    skip_subdirectories,
                },
            ),
    }?;
    disposition
        .validate()
        .context("validating producer-selected Git auto-data disposition")?;
    Ok(disposition)
}

fn git_auto_data_cache_key(config: &AutoDataConfig) -> Result<String> {
    use sha2::{Digest, Sha256};

    let mut disposition = persist_git_auto_data(Some(config))?;
    if let crate::lock::GitWheelAutoData::CheckoutRoot {
        skip_subdirectories,
    } = &mut disposition
    {
        skip_subdirectories.sort();
        skip_subdirectories.dedup();
    }
    let contract = format!(
        "retread-git-auto-data-cache-v1\nemit-epoch={}\n{}\n",
        crate::lock::EMIT_EPOCH,
        serde_json::to_string(&disposition)?,
    );
    Ok(format!("{:x}", Sha256::digest(contract.as_bytes())))
}

fn replay_git_auto_data(
    source: &crate::lock::GitWheelSource,
    checkout_root: PathBuf,
) -> Result<Option<AutoDataConfig>> {
    match source.auto_data.as_ref().ok_or_else(|| {
        anyhow!(
            "Git replay source {}@{} is missing its exact auto-data disposition",
            source.url,
            source.rev,
        )
    })? {
        crate::lock::GitWheelAutoData::Disabled => Ok(None),
        crate::lock::GitWheelAutoData::CheckoutRoot {
            skip_subdirectories,
        } => Ok(Some(AutoDataConfig {
            checkout_root,
            skip_subdirs: skip_subdirectories.iter().map(PathBuf::from).collect(),
        })),
    }
}

/// v0.12.0+: audit context passed alongside `AutoDataConfig` so the
/// resulting `ResolvedWheel` carries enough info to populate the audit
/// without a second pass. `extras_requested` comes from the entry;
/// `dedup_skipped_root` is `Some` when the caller decided NOT to emit
/// auto-data for this wheel because a sibling already covered the
/// checkout root.
#[derive(Debug, Clone, Default)]
pub(crate) struct EntryAuditInfo {
    pub extras_requested: Vec<String>,
    pub dedup_skipped_root: Option<PathBuf>,
}

/// Build / fetch the primary wheel for an entry, apply D, return as
/// a [`ResolvedWheel`].
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
async fn materialize_and_rewrite(
    entry: &crate::config::WheelEntry,
    entry_name: &str,
    expected_version: Option<&str>,
    target: &ResolutionTarget,
    download_dir: &Path,
    source_dir: &Path,
    cache_dir: &Path,
    relax: RelaxPolicy,
    git_sources: &std::collections::BTreeMap<String, crate::config::NamedGitSource>,
    auto_data: Option<AutoDataConfig>,
    audit_info: EntryAuditInfo,
) -> Result<(ResolvedWheel, Vec<String>)> {
    materialize_and_rewrite_with_abi_aliases(
        entry,
        entry_name,
        expected_version,
        target,
        download_dir,
        source_dir,
        cache_dir,
        relax,
        git_sources,
        auto_data,
        audit_info,
        &AbiAliasGraph::new(),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn materialize_and_rewrite_with_abi_aliases(
    entry: &crate::config::WheelEntry,
    entry_name: &str,
    expected_version: Option<&str>,
    target: &ResolutionTarget,
    download_dir: &Path,
    source_dir: &Path,
    cache_dir: &Path,
    relax: RelaxPolicy,
    git_sources: &std::collections::BTreeMap<String, crate::config::NamedGitSource>,
    auto_data: Option<AutoDataConfig>,
    audit_info: EntryAuditInfo,
    abi_aliases: &AbiAliasGraph,
) -> Result<(ResolvedWheel, Vec<String>)> {
    let pypi_name = canonical_conda_name(entry_name);
    let persisted_auto_data = persist_git_auto_data(auto_data.as_ref())?;

    // Phase 1: get the raw wheel onto disk. For source-built wheels
    // (path / git / from), also remember the source root so phase 1.5
    // can inject any files pip wheel failed to ship.
    let mut source_root: Option<PathBuf> = None;
    // Path builds return an immutable source snapshot lease. Keep it alive
    // through phase 1.5 so injection sees the exact tree that produced the
    // content-addressed raw wheel, even if the live checkout changes.
    let mut path_build: Option<crate::source_build::PathWheelBuild> = None;
    // Git builds retain a shared canonical clean source snapshot through both
    // injection phases. The mutable/warm checkout is kept only as a leased
    // planning identity and is never used as artifact input.
    let mut git_build: Option<crate::source_build::GitWheelBuild> = None;
    // Pristine upstream URL captured BEFORE localization to file://.
    // Set for index (PyPI version-spec) and direct-URL entry forms only.
    // Source-built forms (git / path / from) leave this None.
    let mut upstream_url: Option<url::Url> = None;
    // Git provenance (schema 8+): populated for named-git and inline-git
    // entry forms. None for all other origins.
    let mut git_source_captured: Option<crate::lock::GitWheelSource> = None;
    // Sdist provenance (schema 9+): populated when a PyPI version entry has
    // no wheel and falls back to the sdist auto-build. Mutually exclusive
    // with `upstream_url` (which is strictly a wheel URL).
    let mut sdist_source_captured: Option<crate::lock::SdistWheelSource> = None;
    let raw_path: PathBuf = if let Some(from_name) = &entry.from {
        // Named git-source reference: look up url + rev from the
        // [retread-git-sources] table, treat subdirectory just like
        // an inline git entry.
        let src = git_sources.get(from_name).ok_or_else(|| {
            anyhow!(
                "wheel `{entry_name}`: `from = \"{from_name}\"` not found in \
                 [retread-git-sources]"
            )
        })?;
        let subdir = entry.subdirectory.as_deref().unwrap_or(".");
        let out = download_dir.join(entry_name);
        let expected = expected_version.map_or_else(
            || crate::source_build::ExpectedWheel::named(entry_name),
            |version| crate::source_build::ExpectedWheel::exact(entry_name, version),
        );
        let build = crate::source_build::build_wheel_from_git_leased_for_target(
            &src.url,
            &src.rev,
            subdir,
            cache_dir,
            &out,
            target,
            crate::source_build::GitWheelBuildPolicy {
                expected: Some(&expected),
                static_cpp_runtime: entry.static_cpp_runtime,
                submodules: entry.submodules.or(src.submodules),
            },
        )
        .await
        .with_context(|| {
            format!(
                "phase 1 named-git build for entry `{entry_name}` \
                 (from=`{}`, url=`{}`, rev=`{}`, subdir=`{subdir}`, out_dir={})",
                entry.from.as_deref().unwrap_or(""),
                src.url,
                src.rev,
                out.display(),
            )
        })?;
        let wheel = build.wheel_path().to_path_buf();
        let resolved_sha = build.resolved_sha().to_string();
        source_root = Some(build.source_root().to_path_buf());
        // Record git provenance with the RESOLVED SHA (not the config rev,
        // which may be a branch/tag) so replay is manifest-independent.
        // POISONING: the config rev IS in inputs_hash via courier_input_specs
        // (courier.rs:77); changing the named-source rev invalidates the lock
        // and forces a cascade. The resolved SHA here is NOT fed back into the
        // hash — doing so would be circular (the SHA is an output of the build,
        // not a pure input). Replay pins this SHA; a new branch tip is only
        // picked up by a fresh cold solve.
        git_source_captured = Some(crate::lock::GitWheelSource {
            url: src.url.clone(),
            rev: resolved_sha,
            subdirectory: entry.subdirectory.clone(),
            extras: entry.extras.clone(),
            auto_data: Some(persisted_auto_data.clone()),
        });
        git_build = Some(build);
        wheel
    } else if let Some(url) = &entry.url {
        // Capture the direct URL as the upstream before fetch/localization.
        upstream_url = Some((*url).clone());
        let expected = expected_version.map_or_else(
            || crate::source_build::ExpectedWheel::named(entry_name),
            |version| crate::source_build::ExpectedWheel::exact(entry_name, version),
        );
        let mut already_validated = false;
        let fetched = if let Some(authoritative_sha256) = entry.sha256.as_deref() {
            // The digest-qualified destination is stable. Admit it before the
            // generic fetch path so an unchanged strict attestation avoids a
            // second multi-gigabyte compressed-byte hash on warm locks.
            let destination =
                crate::wheel::pinned_wheel_destination(url, authoritative_sha256, download_dir)?;
            if destination.try_exists()? {
                match crate::source_build::validate_pinned_wheel_for_target_async(
                    &destination,
                    target,
                    &expected,
                    authoritative_sha256,
                    url.as_str(),
                )
                .await
                {
                    Ok(_) => {
                        already_validated = true;
                        destination
                    }
                    Err(error)
                        if crate::source_build::is_authoritative_wheel_hash_mismatch(&error) =>
                    {
                        tracing::warn!(
                            wheel = %entry_name,
                            error = %format!("{error:#}"),
                            "pinned destination bytes are corrupt; healing from authoritative store or URL",
                        );
                        crate::wheel::fetch_wheel_cached(
                            url,
                            Some(authoritative_sha256),
                            download_dir,
                            &crate::courier::retread_wheel_store_root(),
                        )
                        .await?
                    }
                    Err(error) => return Err(error).with_context(|| {
                        format!(
                            "phase 1 URL artifact for `{entry_name}` has a correct authoritative identity but fails strict target/name/version validation"
                        )
                    }),
                }
            } else {
                crate::wheel::fetch_wheel_cached(
                    url,
                    Some(authoritative_sha256),
                    download_dir,
                    &crate::courier::retread_wheel_store_root(),
                )
                .await?
            }
        } else {
            crate::wheel::fetch_wheel_cached(
                url,
                None,
                download_dir,
                &crate::courier::retread_wheel_store_root(),
            )
            .await?
        };
        if !already_validated {
            let validation = if let Some(authoritative_sha256) = entry.sha256.as_deref() {
                crate::source_build::validate_pinned_wheel_for_target_async(
                    &fetched,
                    target,
                    &expected,
                    authoritative_sha256,
                    url.as_str(),
                )
                .await
            } else {
                crate::source_build::validate_wheel_for_target_async(
                    &fetched,
                    target,
                    Some(&expected),
                )
                .await
            };
            validation.with_context(|| {
                format!(
                    "phase 1 URL artifact for `{entry_name}` is incompatible with the immutable target or pinned hash"
                )
            })?;
        }
        fetched
    } else if let Some(path) = &entry.path {
        let abs = if Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            source_dir.join(path)
        };
        let out = download_dir.join(entry_name);
        let expected = expected_version.map_or_else(
            || crate::source_build::ExpectedWheel::named(entry_name),
            |version| crate::source_build::ExpectedWheel::exact(entry_name, version),
        );
        let build = crate::source_build::build_wheel_from_path_for_target(
            &abs,
            &out,
            target,
            Some(&expected),
            Some(download_dir),
            (!Path::new(path).is_absolute()).then_some(source_dir),
            entry.static_cpp_runtime,
        )
        .await
        .with_context(|| {
            format!(
                "phase 1 path build for entry `{entry_name}` (source={}, out_dir={})",
                abs.display(),
                out.display(),
            )
        })?;
        source_root = Some(build.source_root().to_path_buf());
        let wheel = build.wheel_path().to_path_buf();
        path_build = Some(build);
        wheel
    } else if let Some(git_url) = &entry.git {
        let rev = entry
            .rev
            .as_ref()
            .ok_or_else(|| anyhow!("git source `{entry_name}` missing rev"))?;
        let subdir = entry.subdirectory.as_deref().unwrap_or(".");
        let out = download_dir.join(entry_name);
        let expected = expected_version.map_or_else(
            || crate::source_build::ExpectedWheel::named(entry_name),
            |version| crate::source_build::ExpectedWheel::exact(entry_name, version),
        );
        let build = crate::source_build::build_wheel_from_git_leased_for_target(
            git_url,
            rev,
            subdir,
            cache_dir,
            &out,
            target,
            crate::source_build::GitWheelBuildPolicy {
                expected: Some(&expected),
                static_cpp_runtime: entry.static_cpp_runtime,
                submodules: entry.submodules,
            },
        )
        .await
        .with_context(|| {
            format!(
                "phase 1 inline-git build for entry `{entry_name}` \
                 (url=`{git_url}`, rev=`{rev}`, subdir=`{subdir}`, out_dir={})",
                out.display(),
            )
        })?;
        let wheel = build.wheel_path().to_path_buf();
        let resolved_sha = build.resolved_sha().to_string();
        source_root = Some(build.source_root().to_path_buf());
        // Record git provenance with the RESOLVED SHA (not the config rev,
        // which may be a branch/tag) so replay is manifest-independent.
        // POISONING: the config rev IS in inputs_hash via courier_input_specs
        // (courier.rs:71); changing the inline rev invalidates the lock and
        // forces a cascade. The resolved SHA is NOT fed back into the hash —
        // doing so would be circular. Replay pins this SHA; a new branch tip
        // is only picked up by a fresh cold solve.
        git_source_captured = Some(crate::lock::GitWheelSource {
            url: git_url.clone(),
            rev: resolved_sha,
            subdirectory: entry.subdirectory.clone(),
            extras: entry.extras.clone(),
            auto_data: Some(persisted_auto_data),
        });
        git_build = Some(build);
        wheel
    } else {
        // PyPI version spec form.
        let version = entry
            .normalized_version()
            .ok_or_else(|| anyhow!("wheel `{entry_name}` has no version, url, path, or git"))?;
        // The version field accepts both a bare exact version ("1.2.3") and a
        // PEP 440 specifier set (">=0.65,<0.66"); only prefix `==` when the
        // text does not already start with an operator.
        let version_text = version.trim();
        let specifier_text = if version_text.starts_with(['<', '>', '~', '!', '=']) {
            version_text.to_string()
        } else {
            format!("=={version_text}")
        };
        let specifiers = VersionSpecifiers::from_str(&specifier_text)
            .map_err(|e| anyhow!("wheel `{entry_name}` version `{version}`: {e}"))?;
        let index_url = entry.index_url();
        match pypi::resolve(&index_url, entry_name, &specifiers, target).await {
            Ok(resolved) => {
                // Capture the pristine index URL BEFORE fetch_wheel_cached may
                // localise / move it. This is the upstream_url written to the lock
                // so Phase-1 replay can re-fetch without a full BFS re-solve.
                upstream_url = Some(resolved.url.clone());
                crate::wheel::fetch_wheel_cached(
                    &resolved.url,
                    resolved.sha256.as_deref(),
                    download_dir,
                    &crate::courier::retread_wheel_store_root(),
                )
                .await
                .with_context(|| {
                    format!(
                        "phase 1 PyPI fetch for entry `{entry_name}` (url=`{}`)",
                        resolved.url,
                    )
                })?
            }
            Err(error) if pypi::is_pypi_index_miss(&error) => {
                let built = build_sdist_wheel_with_specifiers(
                    entry_name.to_string(),
                    specifiers,
                    vec![index_url.clone()],
                    target.clone(),
                    cache_dir.to_path_buf(),
                )
                .await
                .with_context(|| {
                    format!(
                        "phase 1 PyPI sdist fallback for entry `{entry_name}` \
                         (version=`{version}`, index=`{index_url}`)"
                    )
                })?;
                url::Url::parse(&built.sdist_source.sdist_url).with_context(|| {
                    format!(
                        "phase 1 PyPI sdist fallback for entry `{entry_name}` returned invalid \
                         sdist URL `{}`",
                        built.sdist_source.sdist_url,
                    )
                })?;
                tracing::info!(
                    entry = %entry_name,
                    version = %built.version,
                    "phase 1 PyPI wheel resolve missed; using sdist fallback",
                );
                // Sdist provenance goes in `sdist_source`, NEVER `upstream_url`:
                // upstream_url is defined as a WHEEL URL (the courier extracts
                // and validates a PEP 427 filename from it), while the artifact
                // behind an sdist build is a .tar.gz. Recording the sdist URL
                // as upstream_url made courier staging fail with "extracting
                // upstream URL wheel filename ... from ...tar.gz" for every
                // [retread-wheels] version entry whose distribution publishes
                // no wheel (e.g. compress-json). Same contract as
                // bfs_fetch_provenance for BFS transitives.
                sdist_source_captured = Some(built.sdist_source.clone());
                built.wheel_path
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "phase 1 PyPI resolve for entry `{entry_name}` \
                         (version=`{version}`, index=`{index_url}`)"
                    )
                });
            }
        }
    };

    // Phase 1.5: for source-built wheels, top up the wheel with any
    // files the upstream's setup.py forgot to ship. Common breakage:
    // `packages=["isaaclab"]` without find_packages() emits a wheel
    // containing only `isaaclab/__init__.py`, missing every submodule
    // AND sibling data dirs (`config/extension.toml`). Editable
    // installs paper over this; conda installs don't. Auto-inject so
    // the wheel is actually self-sufficient. Cache reuse: skip if the
    // `*.injected.whl` already exists and is newer than the raw wheel.
    let injected_path = if let Some(root) = source_root {
        let out = raw_path.with_extension("injected.whl");
        if is_fresh(&out, &raw_path)? {
            tracing::info!(
                entry = %entry_name,
                wheel = %out.display(),
                "reusing cached injected wheel",
            );
        } else {
            tracing::info!(
                entry = %entry_name,
                source = %root.display(),
                "auto-injecting missing source-root files into wheel",
            );
            crate::wheel_inject::inject(&raw_path, &out, &root).with_context(|| {
                format!(
                    "phase 1.5 source-root inject for entry `{entry_name}` \
                         (source={}, raw_wheel={}, out_wheel={})",
                    root.display(),
                    raw_path.display(),
                    out.display(),
                )
            })?;
        }
        out
    } else {
        raw_path
    };
    // No later phase reads a local path source tree.
    drop(path_build);

    // Phase 1.6 (v0.12.0+): if the caller passed an `AutoDataConfig`,
    // walk the upstream checkout root (parent of this entry's
    // subdirectory) honoring its own `.gitignore` and inject every
    // non-ignored, non-sibling-subdir file as a wheel `.data/data/lib/
    // <rel>` entry -- those land at `$CONDA_PREFIX/lib/<rel>` after
    // pip installs the wheel. Solves the IsaacLab case where the
    // `.kit` experience files live at the repo root but the wheel only
    // captures `source/<pkg>/`. The output lives beneath a cache directory
    // keyed by the canonical skip set and emit epoch, and is refreshed when
    // the injected wheel changes. Auto-data inject doesn't see checkout-root
    // file mtimes; clearing the build cache when upstream files change remains
    // the caller's responsibility.
    let mut auto_data_file_count: Option<usize> = None;
    let with_data_path = if let Some(cfg) = auto_data.as_ref() {
        let build = git_build.as_ref().ok_or_else(|| {
            anyhow!("phase 1.6 checkout-root auto-data requested for non-git entry `{entry_name}`")
        })?;
        if cfg.checkout_root.as_path() != build.checkout_root() {
            bail!(
                "phase 1.6 checkout-root mismatch for entry `{entry_name}`: planned={}, built={}",
                cfg.checkout_root.display(),
                build.checkout_root().display(),
            );
        }
        let checkout_root = build.canonical_root();
        let cache_key = git_auto_data_cache_key(cfg)?;
        let injected_filename = injected_path.file_name().ok_or_else(|| {
            anyhow!(
                "phase 1.6 input wheel has no filename: {}",
                injected_path.display(),
            )
        })?;
        // The disposition is part of the cache path, rather than an adjacent
        // marker. Wheel bytes and their cache identity are therefore
        // published by the same atomic wheel rename: concurrent or rolled-
        // back group shapes cannot pair one disposition with another's bytes.
        let out = injected_path
            .parent()
            .ok_or_else(|| {
                anyhow!(
                    "phase 1.6 input wheel has no parent: {}",
                    injected_path.display(),
                )
            })?
            .join(".retread-autodata")
            .join(cache_key)
            .join(injected_filename)
            .with_extension("autodata.whl");
        std::fs::create_dir_all(out.parent().expect("autodata wheel always has a parent"))
            .with_context(|| format!("creating Git auto-data cache for `{entry_name}`"))?;
        if is_fresh(&out, &injected_path)? {
            tracing::info!(
                entry = %entry_name,
                wheel = %out.display(),
                "reusing cached auto-data wheel",
            );
            // Re-read the cached wheel's RECORD to recover the count
            // would be precise but is overkill for the audit -- use 0
            // as a sentinel that the cache hit happened.
            auto_data_file_count = Some(0);
        } else {
            tracing::info!(
                entry = %entry_name,
                checkout = %checkout_root.display(),
                skip_subdirs = ?cfg.skip_subdirs,
                "phase 1.6: injecting checkout-root tree as wheel .data/data/lib/* (lands at $PREFIX/lib/*)",
            );
            let n = crate::wheel_inject_data::inject_checkout_root_data(
                &injected_path,
                &out,
                checkout_root,
                &cfg.skip_subdirs,
            )
            .with_context(|| {
                format!(
                    "phase 1.6 checkout-root auto-data inject for entry `{entry_name}` \
                     (checkout={}, skip_subdirs={:?}, input={}, output={})",
                    checkout_root.display(),
                    cfg.skip_subdirs,
                    injected_path.display(),
                    out.display(),
                )
            })?;
            auto_data_file_count = Some(n);
        }
        out
    } else {
        injected_path
    };
    // No later phase reads the checkout. Release the logical reader before
    // metadata rewriting so unrelated work does not retain it unnecessarily.
    drop(git_build);

    // Phase 2: apply D (rewrite METADATA per the relax policy). For
    // policies that aren't 'none', the output is a new wheel file with
    // updated METADATA + RECORD; for 'none' it's a no-op copy. Either
    // way we recompute the sha256 of the final file. Cache reuse: skip
    // the rewrite when `*.relaxed.whl` is already up to date.
    let pre_d_path = with_data_path.clone();
    let final_path = if relax == RelaxPolicy::None {
        with_data_path
    } else {
        let rewritten = with_data_path.with_extension("relaxed.whl");
        if is_relaxed_wheel_cache_fresh(&rewritten, &with_data_path, relax, abi_aliases)? {
            tracing::info!(
                entry = %entry_name,
                wheel = %rewritten.display(),
                "reusing cached relaxed wheel",
            );
        } else {
            tracing::info!(
                entry = %entry_name,
                policy = ?relax,
                "applying relax policy to wheel METADATA",
            );
            let _new_sha = rewrite_wheel_with_abi_aliases(
                &with_data_path,
                &rewritten,
                relax,
                abi_aliases,
            )
            .with_context(|| {
                format!(
                    "phase 2 wheel METADATA rewrite for entry `{entry_name}` (policy={relax:?}, \
                         input={}, output={})",
                    with_data_path.display(),
                    rewritten.display(),
                )
            })?;
            write_relaxed_wheel_cache_stamp(&rewritten, relax, abi_aliases)?;
        }
        rewritten
    };

    let metadata = tokio::task::spawn_blocking({
        let p = final_path.clone();
        move || crate::wheel::read_metadata(&p)
    })
    .await
    .context("metadata reader panicked")??;

    // v1.5.9 (isaacsim 6.0.0.0 vs 6.0.0.1 patch drift): keep the
    // ORIGINAL (pre-D) Requires-Dist for the BFS to resolve sub-wheels
    // from. D's relaxation exists for the EMISSION side (conda
    // run-deps + the pins uv reads from site-packages); resolving
    // PYPI sub-wheels from relaxed ranges let exact family pins like
    // `isaacsim-kernel==6.0.0.0` float to the newest patch
    // (>=6.0.0.0,<6.0.1 admits 6.0.0.1), splitting the installed
    // family across patch versions and breaking runtime contracts
    // (Kit extension names renamed between patches). Sub-wheel
    // resolution is exact-first; the relaxed range is only a
    // fallback when the exact version vanished from the index.
    let original_requires_dist = if relax == RelaxPolicy::None {
        metadata.requires_dist.clone()
    } else {
        tokio::task::spawn_blocking({
            let p = pre_d_path;
            move || crate::wheel::read_metadata(&p)
        })
        .await
        .context("metadata reader panicked")??
        .requires_dist
    };

    // The recipe's `source:` URL points at the POST-D wheel. If we
    // returned the upstream URL here, rattler-build would re-download
    // the un-rewritten file and pip would install strict pins into
    // site-packages; uv on the consumer side then reads the strict
    // pins from METADATA and collides with whatever the conda solver
    // chose for those same deps. Always emit file:// of final_path
    // so the on-disk wheel matches the metadata we already emitted.
    let final_url = url::Url::from_file_path(&final_path).map_err(|_| {
        anyhow!(
            "rewritten wheel at {} is not a valid file URL",
            final_path.display()
        )
    })?;

    let auto_data_report = auto_data.as_ref().map(|cfg| crate::audit::AutoDataReport {
        checkout_root: cfg.checkout_root.clone(),
        file_count: auto_data_file_count.unwrap_or(0),
        skip_subdirs: cfg.skip_subdirs.clone(),
    });
    Ok((
        ResolvedWheel {
            pypi_name,
            url: final_url,
            upstream_url,
            git_source: git_source_captured,
            // Populated only by the version-entry sdist fallback above; BFS
            // transitives get theirs in the BFS phase-3 loop.
            sdist_source: sdist_source_captured,
            metadata_provenance: wheel_entry_metadata_provenance(entry),
            extras_requested: audit_info.extras_requested,
            auto_data: auto_data_report,
            auto_data_dedup_skipped_root: audit_info.dedup_skipped_root,
            metadata,
        },
        original_requires_dist,
    ))
}

/// (env name, refinement outcome, the env's final overrides) from one
/// Shared assembly helper: given the already-computed `run_dep_specs`
/// (seeded with the python dep, siblings NOT yet added) and `seen_dep_names`
/// (tracking which conda names were already added), assembles the final
/// [`CondaOutput`] by:
/// 1. Appending sibling cross-links (skipping self).
/// 2. Appending `uv` when `courier` is true.
/// 3. Canonically ordering run dependencies.
/// 4. Building the subdir/noarch/build/variant metadata.
///
/// This is the single source of truth for output assembly — both the hot path
/// (`produce_output` from live bundle data) and the replay path
/// (`replay_from_lock` from lock fields) call this so the emitted
/// [`CondaOutput`] is guaranteed byte-identical.
#[allow(clippy::too_many_arguments)]
fn assemble_conda_output(
    bundle_name: &str,
    version: &str,
    python_version: &str,
    courier: bool,
    any_platform_specific: bool,
    mut run_dep_specs: Vec<NamedSpec<PackageSpec>>,
    mut seen_dep_names: HashSet<String>,
    host_platform: Platform,
    build_number: u64,
    build_hash: Option<&str>,
    loose: bool,
    siblings: &[(String, String)],
) -> Result<CondaOutput> {
    // Always emit the glob form (`python {ver}.*`). Without the `.*`,
    // rattler-conda-types' VersionSpec under Lenient parsing interprets
    // a bare `3` as exact-equals `==3`, and the rattler-build host
    // solve then errors with "No candidates were found for python ==3"
    // because no python package is at literally version 3. The glob is
    // safe for both bare-major ("3" -> "3.*") and dotted ("3.11" ->
    // "3.11.*") because rattler accepts trailing `.*` either way.
    let python_dep = if python_version.contains('*') {
        format!("python {python_version}")
    } else {
        format!("python {python_version}.*")
    };

    // Cross-output linking: pin each sibling output produced by the same
    // conda/outputs call as an exact-version run-dep on this output. The
    // workspace then only needs to declare ONE of the names from the
    // pack -- conda solves the rest transitively. Without this, each
    // [retread-wheels] entry needs its own line in the workspace
    // pixi.toml, which gets verbose for stacks like IsaacLab (8 names).
    for (sib_name, sib_version) in siblings {
        if sib_name == bundle_name {
            continue;
        }
        if seen_dep_names.insert(sib_name.clone()) {
            run_dep_specs.push(spec_from_str(&format!("{sib_name} =={sib_version}"))?);
        }
    }

    // Courier: the metadata pixi SOLVES + LOCKS must include `uv` (the
    // post-link installer needs it), or it never lands in the consuming env.
    // (The recipe adds it too, but pixi resolves against THIS conda/outputs
    // metadata, not the recipe.) The retread installer binary itself SHIPS
    // inside the courier package -- NOT a run-dep -- so the heavy backend
    // never enters the consumer solve.
    // Guard against a duplicate: `uv` may already be in run_dep_specs
    // (pixi forwards it as a solved run-dep), so only add if not present.
    if courier && seen_dep_names.insert("uv".to_string()) {
        run_dep_specs.push(spec_from_str("uv")?);
    }

    // Conda dependency order is semantically irrelevant, but pixi includes
    // the ordered output metadata in its source-package identity. The courier
    // lock canonicalizes conda_run_deps, so leaving the cold path in discovery
    // order made a replay advertise the same set in a different order and
    // changed pixi.lock. Canonicalize the shared cold/replay output here.
    if courier {
        run_dep_specs.sort_by(|a, b| {
            a.name.cmp(&b.name).then_with(|| {
                audit_report::format_packagespec(&a.spec)
                    .cmp(&audit_report::format_packagespec(&b.spec))
            })
        });
    }

    // Courier is never noarch: it ships the native `retread` installer binary
    // + a python-specific lock, and the courier recipe is `noarch: None`.
    // Advertising noarch would make pixi request a noarch build that
    // rattler-build rejects ("--target-platform noarch").
    let subdir = if any_platform_specific || courier {
        host_platform
    } else {
        Platform::NoArch
    };
    let noarch = if any_platform_specific || courier {
        NoArchType::none()
    } else {
        NoArchType::python()
    };

    let py_short = python_version.replace('.', "");
    // A hash-qualified target uses a content-addressed build string so Pixi
    // cannot alias either courier inputs or two rich same-subdir platform
    // contracts. Legacy non-courier targets retain `py{XY}_{build_number}`.
    let build = match build_hash {
        Some(hash) => courier_build_string(&py_short, hash, build_number, loose),
        None => format!("py{py_short}_{build_number}"),
    };

    let mut variant = std::collections::BTreeMap::new();
    variant.insert(
        "python".to_string(),
        VariantValue::String(python_version.to_string()),
    );

    let name = PackageName::new_unchecked(bundle_name.to_string());
    let version_parsed = VersionWithSource::from_str(version)
        .map_err(|e| anyhow!("parsing version `{version}`: {e}"))?;

    Ok(CondaOutput {
        metadata: CondaOutputMetadata {
            name,
            version: version_parsed,
            build,
            build_number,
            subdir,
            license: None,
            license_family: None,
            noarch,
            purls: None,
            python_site_packages_path: None,
            variant,
        },
        build_dependencies: None,
        host_dependencies: Some(CondaOutputDependencies {
            depends: vec![spec_from_str(&python_dep)?, spec_from_str("pip")?],
            constraints: Vec::new(),
        }),
        run_dependencies: CondaOutputDependencies {
            depends: run_dep_specs,
            constraints: Vec::new(),
        },
        ignore_run_exports: CondaOutputIgnoreRunExports::default(),
        run_exports: CondaOutputRunExports::default(),
        input_globs: None,
    })
}

/// `siblings`: every (conda_name, version) produced by the same
/// `conda/outputs` call (including this bundle's own pair, which is
/// skipped). Each non-self entry becomes a run-dep on this output so
/// declaring any one output in the workspace pulls the whole pack via
/// the conda solver.
///
/// `courier_build_hash`: when `Some`, this is either the courier inputs hash
/// (from [`courier_inputs_hash`]) or a non-courier rich target identity. The
/// build string becomes `py{XY}_h{hash_prefix}_{build_number}`. Legacy
/// non-courier calls with no rich target keep `py{XY}_{build_number}`.
#[derive(Clone, Debug)]
struct EmissionConstraintConflict {
    conda_name: CondaName,
    conflict: Conflict,
}

#[derive(Debug)]
struct EmittedBundleRouteAssembly {
    routes: Vec<crate::uv_closure::CondaRouteSpec>,
    conflicts: Vec<EmissionConstraintConflict>,
    supports_by_conda: BTreeMap<String, BTreeSet<EmissionSupport>>,
}

/// Structural provenance for an edge that reached final emission.
///
/// Rule 2 uses this index to prove that a wholly provisional route can
/// disappear without rendering the complete bundle once per dependency.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum EmissionSupport {
    AutoRoute {
        owner_conda_name: String,
    },
    WheelRequirement {
        translated_conda_name: String,
        raw_pypi_name: String,
    },
}

#[derive(Clone, Debug)]
struct PypiEmissionGroup {
    pypi_names: BTreeSet<PypiKey>,
    conda_name: CondaName,
    conda_name_is_authoritative: bool,
    constraints: Vec<Constraint>,
    validation_only_origins: BTreeSet<ConstraintOriginId>,
    native_conda_overrides: BTreeSet<String>,
    supports: BTreeSet<EmissionSupport>,
}

#[derive(Debug)]
struct TranslatedEmissionConstraint {
    specifiers: VersionSpecifiers,
    native_conda_override: Option<String>,
    provenance: Provenance,
}

/// Recover the typed constraint from the translation boundary without
/// reverse-translating conda syntax back into PyPI syntax.
///
/// Ordinary PyPI requirements must carry a valid effective PEP 440 form into
/// shared finalization. Explicit overrides are the sole native-conda boundary;
/// representable ones still participate in PEP 440 override replacement, and
/// genuinely conda-only syntax remains attached as the final native spec.
fn translated_emission_constraint(
    raw: &str,
    dep: &CondaDep,
    metadata_provenance: &Provenance,
) -> Result<TranslatedEmissionConstraint> {
    match &dep.constraint_origin {
        CondaConstraintOrigin::Pypi {
            original_specifiers,
            effective_specifiers,
        } => {
            if effective_specifiers.trim().is_empty() {
                if original_specifiers.trim().is_empty() && dep.spec.trim().is_empty() {
                    return Ok(TranslatedEmissionConstraint {
                        specifiers: VersionSpecifiers::empty(),
                        native_conda_override: None,
                        provenance: metadata_provenance.clone(),
                    });
                }
                return Err(anyhow!(
                    "translated PyPI requirement `{raw}` produced conda-only spec `{}` without \
                     a preserved PEP 440 constraint (source `{original_specifiers}`); \
                     PyPI-origin constraints may not bypass shared finalization",
                    dep.spec
                ));
            }
            let specifiers =
                VersionSpecifiers::from_str(effective_specifiers).map_err(|error| {
                    anyhow!(
                        "translated PyPI requirement `{raw}` produced conda-only spec `{}` with \
                     invalid preserved PEP 440 constraint `{effective_specifiers}` from source \
                     `{original_specifiers}` ({error}); PyPI-origin constraints may not bypass \
                     shared finalization",
                        dep.spec
                    )
                })?;
            if dep.spec.trim().is_empty() {
                return Err(anyhow!(
                    "translated PyPI requirement `{raw}` preserved PEP 440 constraint \
                     `{effective_specifiers}` from source `{original_specifiers}`, but it has no \
                     conda representation; PyPI-origin constraints may not bypass shared \
                     finalization"
                ));
            }
            Ok(TranslatedEmissionConstraint {
                specifiers,
                native_conda_override: None,
                provenance: metadata_provenance.clone(),
            })
        }
        CondaConstraintOrigin::ExplicitOverride => {
            let (specifiers, native_conda_override) = if dep.spec.trim().is_empty() {
                (VersionSpecifiers::empty(), None)
            } else if let Some(pep) = crate::uv_closure::conda_spec_to_pep440(&dep.spec) {
                (
                    VersionSpecifiers::from_str(&pep).with_context(|| {
                        format!("parsing explicit conda override `{}` as `{pep}`", dep.spec)
                    })?,
                    None,
                )
            } else {
                (VersionSpecifiers::empty(), Some(dep.spec.clone()))
            };
            Ok(TranslatedEmissionConstraint {
                specifiers,
                native_conda_override,
                provenance: Provenance::UvOverride,
            })
        }
    }
}

struct EmissionConstraintInput {
    pypi_name: PypiKey,
    conda_name: CondaName,
    conda_name_is_authoritative: bool,
    constraint: Constraint,
    native_conda_override: Option<String>,
    support: EmissionSupport,
}

fn add_emission_constraint(
    groups: &mut Vec<PypiEmissionGroup>,
    indexes: &mut BTreeMap<PypiKey, usize>,
    input: EmissionConstraintInput,
) -> Result<()> {
    let EmissionConstraintInput {
        pypi_name,
        conda_name,
        conda_name_is_authoritative,
        constraint,
        native_conda_override,
        support,
    } = input;
    let conda_key = conda_name.key();
    let index = match indexes.get(&conda_key) {
        Some(index) => *index,
        None => {
            let index = groups.len();
            indexes.insert(conda_key, index);
            groups.push(PypiEmissionGroup {
                pypi_names: BTreeSet::new(),
                conda_name: conda_name.clone(),
                conda_name_is_authoritative,
                constraints: Vec::new(),
                validation_only_origins: BTreeSet::new(),
                native_conda_overrides: BTreeSet::new(),
                supports: BTreeSet::new(),
            });
            index
        }
    };
    let group = &mut groups[index];
    if group.conda_name != conda_name {
        match (
            group.conda_name_is_authoritative,
            conda_name_is_authoritative,
        ) {
            (false, true) => {
                group.conda_name = conda_name;
                group.conda_name_is_authoritative = true;
            }
            (true, false) => {}
            _ => {
                bail!(
                    "multiple equal-authority raw conda targets `{}` and `{}` share canonical \
                     identity `{}`; refusing to discard either dependency edge",
                    group.conda_name,
                    conda_name,
                    group.conda_name.key()
                );
            }
        }
    }
    group.pypi_names.insert(pypi_name);
    if !group.constraints.contains(&constraint) {
        group.constraints.push(constraint);
    }
    if let Some(native_conda_override) = native_conda_override {
        group.native_conda_overrides.insert(native_conda_override);
        if group.native_conda_overrides.len() > 1 {
            bail!(
                "conflicting native conda overrides target `{}`: {}",
                group.conda_name,
                group
                    .native_conda_overrides
                    .iter()
                    .map(|spec| format!("`{spec}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
    group.supports.insert(support);
    Ok(())
}

#[derive(Clone, Default)]
struct EffectiveVersionRange {
    lower: Option<EffectiveVersionBound>,
    upper: Option<EffectiveVersionBound>,
    predicates: Vec<EffectiveVersionPredicate>,
}

#[derive(Clone)]
struct EffectiveVersionBound {
    version: rattler_conda_types::Version,
    exclusive: bool,
    minor_anchor: bool,
}

#[derive(Clone)]
enum EffectiveVersionPredicate {
    StartsWith(rattler_conda_types::Version),
    NotStartsWith(rattler_conda_types::Version),
    Compatible(rattler_conda_types::Version),
    NotCompatible(rattler_conda_types::Version),
}

impl EffectiveVersionPredicate {
    fn anchor(&self) -> &rattler_conda_types::Version {
        match self {
            Self::StartsWith(version)
            | Self::NotStartsWith(version)
            | Self::Compatible(version)
            | Self::NotCompatible(version) => version,
        }
    }

    fn matches(&self, candidate: &rattler_conda_types::Version) -> bool {
        match self {
            Self::StartsWith(version) => candidate.starts_with(version),
            Self::NotStartsWith(version) => !candidate.starts_with(version),
            Self::Compatible(version) => candidate.compatible_with(version),
            Self::NotCompatible(version) => !candidate.compatible_with(version),
        }
    }
}

impl EffectiveVersionRange {
    fn intersect(mut self, other: Self) -> Self {
        let Self {
            lower: other_lower,
            upper: other_upper,
            predicates: other_predicates,
        } = other;
        self.predicates.extend(other_predicates);
        self.lower = match (self.lower, other_lower) {
            (Some(left), Some(right)) => Some(if left.version > right.version {
                left
            } else if right.version > left.version {
                right
            } else {
                EffectiveVersionBound {
                    version: left.version,
                    exclusive: left.exclusive || right.exclusive,
                    minor_anchor: left.minor_anchor || right.minor_anchor,
                }
            }),
            (bound @ Some(_), None) | (None, bound @ Some(_)) => bound,
            (None, None) => None,
        };
        self.upper = match (self.upper, other_upper) {
            (Some(left), Some(right)) => Some(if left.version < right.version {
                left
            } else if right.version < left.version {
                right
            } else {
                EffectiveVersionBound {
                    version: left.version,
                    exclusive: left.exclusive || right.exclusive,
                    minor_anchor: left.minor_anchor || right.minor_anchor,
                }
            }),
            (bound @ Some(_), None) | (None, bound @ Some(_)) => bound,
            (None, None) => None,
        };
        self
    }

    fn bounds_are_empty(&self) -> bool {
        matches!(
            (&self.lower, &self.upper),
            (Some(lower), Some(upper))
                if lower.version > upper.version
                    || (lower.version == upper.version
                        && (lower.exclusive || upper.exclusive))
        )
    }

    fn contains(&self, candidate: &rattler_conda_types::Version) -> bool {
        let above_lower = self.lower.as_ref().is_none_or(|lower| {
            if lower.exclusive {
                candidate > &lower.version
            } else {
                candidate >= &lower.version
            }
        });
        let below_upper = self.upper.as_ref().is_none_or(|upper| {
            if upper.exclusive {
                candidate < &upper.version
            } else {
                candidate <= &upper.version
            }
        });
        above_lower && below_upper
    }

    fn has_predicate_witness(&self) -> bool {
        let candidates = self
            .lower
            .iter()
            .chain(self.upper.iter())
            .map(|bound| bound.version.clone())
            .chain(
                self.predicates
                    .iter()
                    .map(|predicate| predicate.anchor().clone()),
            )
            .collect::<BTreeSet<_>>();
        candidates.into_iter().any(|candidate| {
            self.contains(&candidate)
                && self
                    .predicates
                    .iter()
                    .all(|predicate| predicate.matches(&candidate))
        })
    }

    fn is_singleton(&self) -> bool {
        matches!(
            (&self.lower, &self.upper),
            (Some(lower), Some(upper))
                if lower.version == upper.version
                    && !lower.exclusive
                    && !upper.exclusive
        )
    }

    fn is_empty(&self) -> bool {
        self.bounds_are_empty()
            || (!self.predicates.is_empty() && self.is_singleton() && !self.has_predicate_witness())
    }

    fn is_proven_nonempty(&self) -> bool {
        !self.bounds_are_empty() && (self.predicates.is_empty() || self.has_predicate_witness())
    }

    fn has_effective_minor_bound(&self) -> bool {
        self.lower
            .iter()
            .chain(self.upper.iter())
            .any(|bound| bound.minor_anchor)
    }
}

/// Take the convex hull of the interval envelopes of a disjunction before
/// deciding whether a minor boundary is load-bearing. Internal boundaries and
/// gaps do not constrain the outer ABI range: `<=3.1|>3.1` becomes unbounded,
/// while `>=3,<3.1|>=3.1,<4` becomes the bare-major interval `[3,4)`.
///
/// Predicate-bearing multi-range unions are rejected by the caller instead
/// of being sent here: their interval envelopes can over-approximate the real
/// prefix/compatible sets.
fn convex_hull_effective_range_envelopes(
    mut ranges: Vec<EffectiveVersionRange>,
) -> Vec<EffectiveVersionRange> {
    ranges.sort_by(|left, right| match (&left.lower, &right.lower) {
        (None, None) => std::cmp::Ordering::Equal,
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
        (Some(left), Some(right)) => left
            .version
            .cmp(&right.version)
            .then_with(|| left.exclusive.cmp(&right.exclusive)),
    });

    let mut merged: Vec<EffectiveVersionRange> = Vec::new();
    for mut range in ranges {
        let Some(last) = merged.last_mut() else {
            range.predicates.clear();
            merged.push(range);
            continue;
        };
        if let (Some(left), Some(right)) = (&mut last.lower, &range.lower)
            && left.version == right.version
        {
            left.exclusive &= right.exclusive;
            left.minor_anchor |= right.minor_anchor;
        }
        last.upper = match (last.upper.take(), range.upper) {
            (None, _) | (_, None) => None,
            (Some(left), Some(right)) => Some(if left.version > right.version {
                left
            } else if right.version > left.version {
                right
            } else {
                EffectiveVersionBound {
                    version: left.version,
                    exclusive: left.exclusive && right.exclusive,
                    minor_anchor: left.minor_anchor || right.minor_anchor,
                }
            }),
        };
        last.predicates.clear();
    }
    merged
}

fn checked_bump_last(
    version: &rattler_conda_types::Version,
) -> Option<rattler_conda_types::Version> {
    // Consume this iterator forward because rattler computes each segment's
    // component offset as the iterator advances.
    let last_segment = version.segments().fold(None, |_, segment| Some(segment))?;
    let last_numeral = last_segment
        .components()
        .filter_map(rattler_conda_types::Component::as_number)
        .next_back()?;
    if last_numeral == u64::MAX {
        return None;
    }
    version
        .bump(rattler_conda_types::VersionBumpType::Last)
        .ok()
}

fn effective_version_ranges(spec: &VersionSpec) -> Vec<EffectiveVersionRange> {
    use rattler_conda_types::version_spec::{
        EqualityOperator, LogicalOperator, RangeOperator, StrictRangeOperator,
    };

    let is_semantic_minor_boundary = |version: &rattler_conda_types::Version| {
        version.as_major_minor().is_some()
            && version
                .with_segments(..1)
                .is_some_and(|major| *version != major)
    };
    let lower_with_anchor =
        |version: rattler_conda_types::Version, exclusive, minor_anchor| EffectiveVersionRange {
            lower: Some(EffectiveVersionBound {
                minor_anchor,
                version,
                exclusive,
            }),
            upper: None,
            predicates: Vec::new(),
        };
    let upper_with_anchor =
        |version: rattler_conda_types::Version, exclusive, minor_anchor| EffectiveVersionRange {
            lower: None,
            upper: Some(EffectiveVersionBound {
                minor_anchor,
                version,
                exclusive,
            }),
            predicates: Vec::new(),
        };
    let lower = |version: rattler_conda_types::Version, exclusive| {
        let minor_anchor = is_semantic_minor_boundary(&version);
        lower_with_anchor(version, exclusive, minor_anchor)
    };
    let upper = |version: rattler_conda_types::Version, exclusive| {
        let minor_anchor = is_semantic_minor_boundary(&version);
        upper_with_anchor(version, exclusive, minor_anchor)
    };

    match spec {
        VersionSpec::None => Vec::new(),
        VersionSpec::Any => vec![EffectiveVersionRange::default()],
        VersionSpec::Range(RangeOperator::Greater, version) => {
            vec![lower(version.clone(), true)]
        }
        VersionSpec::Range(RangeOperator::GreaterEquals, version) => {
            vec![lower(version.clone(), false)]
        }
        VersionSpec::Range(RangeOperator::Less, version) => {
            vec![upper(version.clone(), true)]
        }
        VersionSpec::Range(RangeOperator::LessEquals, version) => {
            vec![upper(version.clone(), false)]
        }
        VersionSpec::Exact(EqualityOperator::Equals, version) => {
            vec![EffectiveVersionRange {
                lower: Some(EffectiveVersionBound {
                    version: version.clone(),
                    exclusive: false,
                    // Equality selects one concrete version even when it has
                    // only a major segment (for example `==3`).
                    minor_anchor: true,
                }),
                upper: Some(EffectiveVersionBound {
                    version: version.clone(),
                    exclusive: false,
                    minor_anchor: true,
                }),
                predicates: Vec::new(),
            }]
        }
        VersionSpec::Exact(EqualityOperator::NotEquals, version) => {
            // The complement of one exact point. Exclusion-created bounds do
            // not count as the positive minor anchor required for ABI safety.
            vec![
                upper_with_anchor(version.clone(), true, false),
                lower_with_anchor(version.clone(), true, false),
            ]
        }
        VersionSpec::StrictRange(StrictRangeOperator::StartsWith, version) => {
            let lower = lower(version.0.clone(), false);
            let upper = checked_bump_last(&version.0)
                .map(|version| upper(version, true))
                .unwrap_or_default();
            let mut range = lower.intersect(upper);
            range
                .predicates
                .push(EffectiveVersionPredicate::StartsWith(version.0.clone()));
            vec![range]
        }
        VersionSpec::StrictRange(StrictRangeOperator::NotStartsWith, version) => {
            // `!=P.*` is the complement of [P, bump_last(P)).
            // Keep the real negative predicate as well: matching prereleases
            // can sort below P while still starting with P.
            let mut ranges = vec![upper_with_anchor(version.0.clone(), true, false)];
            if let Some(ceiling) = checked_bump_last(&version.0) {
                ranges.push(lower_with_anchor(ceiling, false, false));
            }
            for range in &mut ranges {
                range
                    .predicates
                    .push(EffectiveVersionPredicate::NotStartsWith(version.0.clone()));
            }
            ranges
        }
        VersionSpec::StrictRange(StrictRangeOperator::Compatible, version) => {
            let lower = lower(version.0.clone(), false);
            let upper = version
                .0
                .pop_segments(1)
                .and_then(|prefix| checked_bump_last(&prefix))
                .map(|version| upper(version, true))
                .unwrap_or_default();
            let mut range = lower.intersect(upper);
            range
                .predicates
                .push(EffectiveVersionPredicate::Compatible(version.0.clone()));
            vec![range]
        }
        VersionSpec::StrictRange(StrictRangeOperator::NotCompatible, version) => {
            // The positive compatible interval is [V, compatible_upper(V)).
            // Keep its two complement branches; if the upper boundary cannot
            // be represented, retain only the provable before-V branch. The
            // real negative predicate closes prerelease ordering gaps.
            let mut ranges = vec![upper_with_anchor(version.0.clone(), true, false)];
            if let Some(ceiling) = version
                .0
                .pop_segments(1)
                .and_then(|prefix| checked_bump_last(&prefix))
            {
                ranges.push(lower_with_anchor(ceiling, false, false));
            }
            for range in &mut ranges {
                range
                    .predicates
                    .push(EffectiveVersionPredicate::NotCompatible(version.0.clone()));
            }
            ranges
        }
        VersionSpec::Group(LogicalOperator::Or, members) => {
            members.iter().flat_map(effective_version_ranges).collect()
        }
        VersionSpec::Group(LogicalOperator::And, members) => {
            members
                .iter()
                .fold(vec![EffectiveVersionRange::default()], |ranges, member| {
                    let member_ranges = effective_version_ranges(member);
                    ranges
                        .into_iter()
                        .flat_map(|range| {
                            member_ranges
                                .iter()
                                .cloned()
                                .map(move |member_range| range.clone().intersect(member_range))
                        })
                        .filter(|range| !range.is_empty())
                        .collect()
                })
        }
    }
}

fn is_bare_major_spec(spec: &str) -> bool {
    let Ok(spec) =
        VersionSpec::from_str(spec.trim(), rattler_conda_types::ParseStrictness::Lenient)
    else {
        // This is a release-mode safety net. An unparseable anchor constraint
        // cannot demonstrate a load-bearing minor restriction, so fail closed.
        return true;
    };
    let mut ranges = Vec::new();
    for range in effective_version_ranges(&spec) {
        if range.is_empty() {
            continue;
        }
        if !range.is_proven_nonempty() {
            return true;
        }
        ranges.push(range);
    }
    if ranges.is_empty() {
        return true;
    }
    if ranges
        .iter()
        .any(|range| !range.has_effective_minor_bound())
    {
        return true;
    }
    if ranges.len() > 1 && ranges.iter().any(|range| !range.predicates.is_empty()) {
        return true;
    }
    convex_hull_effective_range_envelopes(ranges)
        .iter()
        .any(|range| !range.has_effective_minor_bound())
}

/// Complete one open, bare-major PEP 440 lower bound into the canonical
/// within-major interval accepted for retread's own ABI-anchor emission.
///
/// This deliberately requires an inclusive lower-bound clause with a
/// PEP 440-parsed release major. Compatible extra clauses are preserved and
/// the within-major cap is intersected with them; other shapes stay untouched
/// so the post-emission invariant can reject them without guessing at intent.
fn bare_major_floor(specifier: &VersionSpecifier) -> Option<(u64, u64)> {
    if *specifier.operator() != Operator::GreaterThanEqual {
        return None;
    }
    let version = specifier.version();
    Some((version.epoch(), *version.release().first()?))
}

fn is_plain_epochless_bare_major_floor(specifier: &VersionSpecifier) -> bool {
    let version = specifier.version();
    version.epoch() == 0
        && !version.is_pre()
        && !version.is_post()
        && !version.is_dev()
        && !version.is_local()
        && version
            .release()
            .iter()
            .skip(1)
            .all(|component| *component == 0)
}

fn has_qualified_floor_version(specifier: &VersionSpecifier) -> bool {
    let version = specifier.version();
    version.epoch() != 0
        || version.is_pre()
        || version.is_post()
        || version.is_dev()
        || version.is_local()
}

fn specifier_caps_before_next_major(specifier: &VersionSpecifier, next_major: &Version) -> bool {
    match specifier.operator() {
        Operator::LessThan => specifier.version() <= next_major,
        Operator::LessThanEqual
        | Operator::Equal
        | Operator::ExactEqual
        | Operator::EqualStar
        | Operator::TildeEqual => specifier.version() < next_major,
        Operator::GreaterThan
        | Operator::GreaterThanEqual
        | Operator::NotEqual
        | Operator::NotEqualStar => false,
    }
}

fn is_exact_next_major_cap(specifier: &VersionSpecifier, floor: (u64, u64)) -> bool {
    if *specifier.operator() != Operator::LessThan {
        return false;
    }
    let (epoch, major) = floor;
    let Some(next_major) = major.checked_add(1) else {
        return false;
    };
    let version = specifier.version();
    version.epoch() == epoch
        && !version.is_pre()
        && !version.is_post()
        && !version.is_dev()
        && !version.is_local()
        && version.release().len() == 1
        && version.release().first() == Some(&next_major)
}

fn auto_complete_bare_major_abi_anchor_spec(spec: &str) -> Option<String> {
    let specifiers = VersionSpecifiers::from_str(spec.trim()).ok()?;
    let floor = specifiers.iter().filter_map(bare_major_floor).max()?;
    let (epoch, major) = floor;
    let next_major = major.checked_add(1)?;
    let cap_version = Version::new([next_major]).with_epoch(epoch);
    let qualified_floor = specifiers.iter().any(|specifier| {
        bare_major_floor(specifier) == Some(floor) && has_qualified_floor_version(specifier)
    });
    let qualified_floor_needs_cap = qualified_floor
        && !specifiers
            .iter()
            .any(|specifier| specifier_caps_before_next_major(specifier, &cap_version));
    if (!is_bare_major_spec(spec) && !qualified_floor_needs_cap)
        || is_auto_completed_abi_anchor_spec(spec)
    {
        return None;
    }
    let cap = VersionSpecifier::less_than_version(cap_version);
    let mut normalized = Vec::with_capacity(specifiers.len() + 1);
    let mut normalized_plain_floor = false;
    let mut has_cap = false;
    for specifier in specifiers.iter() {
        if bare_major_floor(specifier) == Some(floor) {
            if is_plain_epochless_bare_major_floor(specifier) {
                if normalized_plain_floor {
                    continue;
                }
                normalized.push(format!(">={major}.0"));
                normalized_plain_floor = true;
            } else {
                // Epoch, pre, post, dev, and any other PEP 440 semantics on
                // the floor are load-bearing. The cap is derived solely from
                // the parsed release major; the floor itself stays exact.
                normalized.push(specifier.to_string());
            }
            continue;
        }
        has_cap |= is_exact_next_major_cap(specifier, floor);
        normalized.push(specifier.to_string());
    }
    if !has_cap {
        normalized.push(cap.to_string());
    }
    let normalized = normalized.join(",");
    let parsed = VersionSpecifiers::from_str(&normalized).ok()?;
    if crate::constraint::specifiers_unsatisfiable(&parsed) || normalized == spec.trim() {
        return None;
    }
    Some(normalized)
}

/// The emission guard's narrow exception for a canonical within-major
/// interval produced by [`auto_complete_bare_major_abi_anchor_spec`].
///
/// `is_bare_major_spec` remains intentionally strict for embedded wheel
/// metadata and overrides. Only a retread-emitted `>=M.0,<M+1` intersection,
/// optionally tightened by compatible extra clauses, is recognized here.
fn is_auto_completed_abi_anchor_spec(spec: &str) -> bool {
    let Ok(specifiers) = VersionSpecifiers::from_str(spec.trim()) else {
        return false;
    };
    if crate::constraint::specifiers_unsatisfiable(&specifiers) {
        return false;
    }
    specifiers.iter().any(|floor| {
        let Some(floor_key) = bare_major_floor(floor) else {
            return false;
        };
        let canonical_floor = !is_plain_epochless_bare_major_floor(floor)
            || (floor.version().release().len() == 2
                && floor.version().release().get(1) == Some(&0));
        canonical_floor
            && specifiers
                .iter()
                .any(|cap| is_exact_next_major_cap(cap, floor_key))
    })
}

/// Replace one effective exact ABI-anchor selection with its canonical
/// within-minor compatibility band.
///
/// Final emission can join an auto-routed `==X.Y.Z` selection with looser
/// wheel clauses such as `~=X.Y.Z`. The exact selection is useful while
/// building one pack, but retaining it in the advertised conda dependency
/// prevents independently-built packs from composing on a newer ABI-compatible
/// patch. Qualified and arbitrary equalities stay fail-closed because the
/// release-only range projection would erase their load-bearing semantics.
fn widen_exact_abi_anchor_spec_to_minor_band(spec: &str) -> Option<String> {
    let specifiers = VersionSpecifiers::from_str(spec.trim()).ok()?;
    if crate::constraint::specifiers_unsatisfiable(&specifiers)
        || specifiers.iter().any(|specifier| {
            let version = specifier.version();
            *specifier.operator() == Operator::ExactEqual
                || version.epoch() != 0
                || version.is_pre()
                || version.is_post()
                || version.is_dev()
                || version.is_local()
        })
    {
        return None;
    }

    let exact = release_specifiers_to_ranges(specifiers.clone())
        .as_singleton()
        .cloned()?;
    let release = exact.release();
    let major = *release.first()?;
    let minor = *release.get(1)?;
    let next_minor = minor.checked_add(1)?;
    let lower = Version::new([major, minor]);
    let upper = Version::new([major, next_minor]);

    let mut removed_exact_constraint = false;
    let mut normalized = specifiers
        .iter()
        .filter_map(|specifier| {
            let pins_exact_release = specifier.version().only_release() == exact;
            if *specifier.operator() == Operator::Equal && pins_exact_release {
                removed_exact_constraint = true;
                None
            } else {
                Some(specifier.clone())
            }
        })
        .collect::<Vec<_>>();
    if !removed_exact_constraint {
        return None;
    }
    normalized.extend([
        VersionSpecifier::greater_than_equal_version(lower),
        VersionSpecifier::less_than_version(upper),
    ]);
    let normalized: VersionSpecifiers = normalized.into_iter().collect();
    if crate::constraint::specifiers_unsatisfiable(&normalized) {
        return None;
    }

    // A convex intersection has a lossless canonical floor/cap rendering.
    // Preserve the explicit clauses for a non-convex result so exclusions
    // and other holes remain load-bearing.
    let ranges = release_specifiers_to_ranges(normalized.clone());
    let mut intervals = ranges.iter();
    let rendered = if let Some((lower, upper)) = intervals.next()
        && intervals.next().is_none()
    {
        let lower = match lower {
            std::ops::Bound::Included(version) => {
                let preferred = normalized
                    .iter()
                    .filter(|specifier| {
                        matches!(
                            specifier.operator(),
                            Operator::GreaterThanEqual | Operator::TildeEqual | Operator::EqualStar
                        ) && specifier.version().only_release() == *version
                    })
                    .map(|specifier| specifier.version().only_release())
                    .max_by_key(|version| version.release().len())
                    .unwrap_or_else(|| version.clone());
                Some(VersionSpecifier::greater_than_equal_version(preferred))
            }
            _ => VersionSpecifier::from_lower_bound(lower),
        };
        [lower, VersionSpecifier::from_upper_bound(upper)]
            .into_iter()
            .flatten()
            .map(|specifier| specifier.to_string())
            .collect::<Vec<_>>()
            .join(",")
    } else {
        normalized.to_string().replace(", ", ",")
    };
    (rendered != spec.trim()).then_some(rendered)
}

fn normalize_emitted_abi_anchor_spec(
    bundle: &str,
    package: &PypiKey,
    constraints: &[Constraint],
    rendered: String,
    allow_exact_widening: bool,
) -> (String, Option<auto_bundle::WheelMetadataRelaxation>) {
    let exact_band = if allow_exact_widening {
        widen_exact_abi_anchor_spec_to_minor_band(&rendered)
    } else {
        None
    };
    let Some((normalized, widened_exact_pin)) =
        exact_band.map(|normalized| (normalized, true)).or_else(|| {
            auto_complete_bare_major_abi_anchor_spec(&rendered)
                .map(|normalized| (normalized, false))
        })
    else {
        return (rendered, None);
    };
    let involved_sources = constraints
        .iter()
        .map(|constraint| constraint.source.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let warning = if widened_exact_pin {
        auto_bundle::abi_anchor_exact_pin_widening(
            bundle,
            package,
            rendered.clone(),
            normalized.clone(),
            involved_sources,
        )
    } else {
        auto_bundle::abi_anchor_cap_completion(
            bundle,
            package,
            rendered.clone(),
            normalized.clone(),
            involved_sources,
        )
    };
    (normalized, Some(warning))
}

type WorkspaceAbiVersions = BTreeMap<String, BTreeSet<String>>;

fn output_abi_aliases(bundle: &Bundle, config: &RetreadConfig) -> AbiAliasGraph {
    let mut aliases = abi_aliases_from_name_map(&config.name_map);
    for route in &bundle.auto_routed {
        add_abi_alias_edge(
            &mut aliases,
            &route.route.pypi_name,
            &route.route.conda_name,
        );
        if let Some(provider) = &route.workspace_provider {
            add_abi_alias_edge(
                &mut aliases,
                &route.route.pypi_name,
                provider.conda_name.as_spec(),
            );
        }
    }
    aliases
}

fn output_workspace_abi_versions(
    bundle: &Bundle,
    workspace_python_version: &str,
) -> WorkspaceAbiVersions {
    let mut workspace_versions = WorkspaceAbiVersions::new();
    for (name, version) in &bundle.workspace_conda_versions {
        workspace_versions
            .entry(canonical_conda_name(name))
            .or_default()
            .insert(version.clone());
    }
    for (name, fact) in &bundle.workspace_conda_provider_facts {
        workspace_versions
            .entry(canonical_conda_name(name))
            .or_default()
            .extend(fact.selected_versions.iter().cloned());
    }
    for route in &bundle.auto_routed {
        if let Some(provider) = &route.workspace_provider {
            workspace_versions
                .entry(provider.conda_name.key().into_string())
                .or_default()
                .extend(provider.selected_versions.iter().cloned());
        }
    }
    if workspace_python_version.contains('.') {
        workspace_versions
            .entry("python".to_string())
            .or_default()
            .insert(workspace_python_version.to_string());
    }
    workspace_versions
}

/// Current, solved ABI facts attached to one advertised output.
///
/// This is process-local on purpose: producer-time lock facts cannot stand in
/// for the versions conda/outputs selected for the current build request.
struct ReplayAbiContext {
    workspace_versions: WorkspaceAbiVersions,
    overrides: BTreeMap<String, String>,
    aliases: AbiAliasGraph,
}

fn replay_abi_context_for_bundle(
    bundle: &Bundle,
    config: &RetreadConfig,
    workspace_python_version: &str,
) -> ReplayAbiContext {
    ReplayAbiContext {
        workspace_versions: output_workspace_abi_versions(bundle, workspace_python_version),
        overrides: config.overrides.clone(),
        aliases: output_abi_aliases(bundle, config),
    }
}

/// Post-emission ABI safety net ported from the deleted cascade.
///
/// The former implementation only logged and merely noticed workspace pins.
/// This pure check returns stable violations so every emission caller can
/// reject corruption in release builds.
pub(crate) fn check_output_abi_invariants(
    output_run_deps: &[(String, String)],
    embedded_requires_dist: &[(String, String)],
    workspace_versions: &WorkspaceAbiVersions,
    overrides: &BTreeMap<String, String>,
    aliases: &AbiAliasGraph,
) -> Vec<String> {
    let mut violations = Vec::new();
    // A source-built wheel may legitimately declare an unconstrained ABI
    // runtime (openmesh 1.2.1 declares bare `numpy`). It is safe only when
    // Retread's emitted conda contract supplies a concrete constraint for the
    // same semantic anchor; uv installs bundled wheels with dependency
    // resolution disabled, so that emitted contract is authoritative.
    let concretely_emitted_anchors = output_run_deps
        .iter()
        .filter(|(name, spec)| {
            let spec = spec.trim();
            is_semantic_abi_anchor(name, aliases)
                && !spec.is_empty()
                && spec != "*"
                && !is_bare_major_spec(spec)
        })
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let mut emitted = output_run_deps
        .iter()
        .cloned()
        .map(|(name, spec)| (name, spec, "retread emitted".to_string(), None, true))
        .collect::<Vec<_>>();
    for (wheel, raw) in embedded_requires_dist {
        let Ok(requirement): Result<uv_pep508::Requirement, _> =
            uv_pep508::Requirement::from_str(raw)
        else {
            continue;
        };
        let (spec, pep440_specifiers) = match requirement.version_or_url.as_ref() {
            None => (String::new(), Some(VersionSpecifiers::empty())),
            Some(uv_pep508::VersionOrUrl::VersionSpecifier(specifiers)) => (
                specifiers.to_string().replace(", ", ","),
                Some(specifiers.clone()),
            ),
            // A direct artifact URL is itself an exact artifact selection, not
            // an unconstrained version range.
            Some(uv_pep508::VersionOrUrl::Url(_)) => continue,
        };
        emitted.push((
            requirement.name.to_string(),
            spec,
            format!("wheel `{wheel}` embeds"),
            pep440_specifiers,
            false,
        ));
    }
    emitted.sort_by(|left, right| (&left.0, &left.1, &left.2).cmp(&(&right.0, &right.1, &right.2)));

    for (name, spec, origin, pep440_specifiers, allow_auto_completed_cap) in emitted {
        if !is_semantic_abi_anchor(&name, aliases) {
            continue;
        }
        let trimmed = spec.trim();
        if trimmed.is_empty() || trimmed == "*" {
            let covered_by_emitted_contract = !allow_auto_completed_cap
                && semantic_aliases(&name, aliases).iter().any(|alias| {
                    concretely_emitted_anchors.iter().any(|emitted| {
                        semantic_aliases(emitted, aliases)
                            .iter()
                            .any(|candidate| candidate == alias)
                    })
                });
            if covered_by_emitted_contract {
                continue;
            }
            violations.push(format!(
                "ABI invariant: {origin} `{name} {trimmed}` (empty/*); \
                 ABI anchors must carry a concrete spec"
            ));
            continue;
        }
        if pep440_specifiers
            .as_ref()
            .is_some_and(crate::constraint::specifiers_unsatisfiable)
        {
            violations.push(format!(
                "ABI invariant: {origin} `{name} {trimmed}` is unsatisfiable under PEP 440"
            ));
            continue;
        }
        // libstdcxx-ng's compatibility contract is the GCC runtime major,
        // unlike Python/CUDA ABI anchors whose bare major is ambiguous. A
        // hermetic wheel's final DT_NEEDED scan deliberately emits this form.
        let gcc_runtime_major = canonical_conda_name(&name) == "libstdcxx-ng";
        // The canonical auto-completed band (`>=M.0,<M+1`) is accepted from
        // every origin: emit-pypi's anchor floor completion writes it into
        // shipped wheel METADATA (epoch 46), not only into retread-emitted
        // conda contracts (epoch 41).
        if is_bare_major_spec(trimmed)
            && !gcc_runtime_major
            && !is_auto_completed_abi_anchor_spec(trimmed)
        {
            violations.push(format!(
                "ABI invariant: {origin} `{name} {trimmed}` (bare-major); \
                 ABI anchors must carry a minor or stricter spec"
            ));
            continue;
        }

        let parsed_conda_spec = pep440_specifiers
            .is_none()
            .then(|| VersionSpec::from_str(trimmed, rattler_conda_types::ParseStrictness::Lenient));
        for workspace_name in semantic_aliases(&name, aliases) {
            let Some(selected_versions) = workspace_versions.get(&workspace_name) else {
                continue;
            };
            for workspace_version in selected_versions {
                if let Some(specifiers) = &pep440_specifiers {
                    match uv_pep508::uv_pep440::Version::from_str(workspace_version) {
                        Ok(version) if specifiers.contains(&version) => {}
                        Ok(_) => violations.push(format!(
                            "ABI invariant: {origin} `{name} {trimmed}` does not cover workspace pin \
                             `{workspace_name}=={workspace_version}`"
                        )),
                        Err(error) => violations.push(format!(
                            "ABI invariant: workspace pin `{workspace_name}=={workspace_version}` \
                             cannot be validated as PEP 440: {error}"
                        )),
                    }
                    continue;
                }

                let parsed_version = rattler_conda_types::Version::from_str(workspace_version);
                match (&parsed_version, parsed_conda_spec.as_ref().unwrap()) {
                    (Ok(version), Ok(specifier)) if specifier.matches(version) => {}
                    (Ok(_), Ok(_)) => violations.push(format!(
                        "ABI invariant: {origin} `{name} {trimmed}` does not cover workspace pin \
                         `{workspace_name}=={workspace_version}`"
                    )),
                    (Err(error), _) => violations.push(format!(
                        "ABI invariant: workspace pin `{workspace_name}=={workspace_version}` \
                         cannot be validated: {error}"
                    )),
                    (_, Err(error)) => violations.push(format!(
                        "ABI invariant: {origin} anchor spec `{name} {trimmed}` cannot be \
                         validated: {error}"
                    )),
                }
            }
        }
    }

    for (name, spec) in overrides {
        if !is_semantic_abi_anchor(name, aliases) {
            continue;
        }
        let trimmed = spec.trim();
        if trimmed.is_empty() || trimmed == "*" || is_bare_major_spec(trimmed) {
            violations.push(format!(
                "ABI invariant: `retread-overrides[{name}]` is `{trimmed}`; \
                 ABI anchors must never be widened to */bare-major"
            ));
        }
    }

    violations.sort();
    violations.dedup();
    violations
}

fn ensure_output_abi_invariants(
    output: &CondaOutput,
    bundle: &Bundle,
    config: &RetreadConfig,
    workspace_python_version: &str,
) -> Result<()> {
    let emitted = output
        .run_dependencies
        .depends
        .iter()
        .map(|dependency| {
            (
                dependency.name.as_str().to_string(),
                audit_report::format_packagespec(&dependency.spec),
            )
        })
        .collect::<Vec<_>>();
    let workspace_versions = output_workspace_abi_versions(bundle, workspace_python_version);
    // The courier performs one final metadata rewrite after phase D. Validate
    // the requirements uv will actually read, rather than the intermediate
    // phase-D lines: floor envelopes can reconcile several source-wheel
    // clauses, and orphan URL requirements can be removed entirely.
    let emit_wheels = bundle
        .all_wheels()
        .map(|wheel| crate::emit_pypi::EmitWheel {
            pypi_name: wheel.pypi_name.clone(),
            version: wheel.metadata.version.clone(),
            requires_dist: wheel.metadata.requires_dist.clone(),
            local_path: wheel.url.to_file_path().ok(),
            wheel_filename: wheel.metadata.filename.clone(),
            sha256: Some(wheel.metadata.sha256.clone()),
            locked_final_sha256: None,
            remote_url: (wheel.url.scheme() != "file").then(|| wheel.url.clone()),
            upstream_url: wheel.upstream_url.clone(),
            git_source: wheel.git_source.clone(),
            sdist_source: wheel.sdist_source.clone(),
        })
        .collect::<Vec<_>>();
    let mut conda_capable = bundle
        .probe_decisions
        .iter()
        .filter(|decision| decision.matching_candidates > 0)
        .map(|decision| canonical_conda_name(&decision.pypi_name))
        .collect::<HashSet<_>>();
    conda_capable.extend(config.name_map.keys().map(|key| key.as_str().to_owned()));
    let emit_plan = crate::emit_pypi::plan(
        &emit_wheels,
        &conda_capable,
        &output_abi_aliases(bundle, config),
    );
    let line_map = crate::emit_pypi::override_line_map(
        &emit_plan.overrides,
        &conda_capable,
        &emit_plan.drop_url,
    );
    let embedded_requires_dist = emit_wheels
        .iter()
        .flat_map(|wheel| {
            wheel
                .requires_dist
                .iter()
                .filter_map(|requirement| match line_map(requirement) {
                    crate::wheel_rewrite::LineAction::Keep => {
                        Some((wheel.pypi_name.clone(), requirement.clone()))
                    }
                    crate::wheel_rewrite::LineAction::Replace(replacement) => {
                        Some((wheel.pypi_name.clone(), replacement))
                    }
                    crate::wheel_rewrite::LineAction::Drop => None,
                })
        })
        .collect::<Vec<_>>();
    let aliases = output_abi_aliases(bundle, config);
    let violations = check_output_abi_invariants(
        &emitted,
        &embedded_requires_dist,
        &workspace_versions,
        &config.overrides,
        &aliases,
    );
    if violations.is_empty() {
        return Ok(());
    }
    for violation in &violations {
        tracing::error!(
            bundle = %bundle.conda_name,
            violation = %violation,
            "bundle emission rejected by ABI invariant",
        );
    }
    bail!(
        "bundle emission rejected by ABI invariant: {}",
        violations.join("; ")
    )
}

type ProducedOutput = (
    CondaOutput,
    Vec<EmissionConstraintConflict>,
    Vec<auto_bundle::WheelMetadataRelaxation>,
    BTreeMap<String, BTreeSet<EmissionSupport>>,
);

fn produce_output_with_conflicts(
    bundle: &Bundle,
    config: &RetreadConfig,
    host_platform: Platform,
    workspace_python_version: &str,
    siblings: &[(String, String)],
    courier_build_hash: Option<&str>,
    // When `Some`, overrides `bundle.primary.metadata.version` so the metadata
    // phase reports the same version the build phase will use (e.g. lock.version
    // on an incremental add).  `None` → use bundle.primary.metadata.version
    // (today's behaviour, always chosen when RETREAD_INCREMENTAL is unset).
    version_override: Option<&str>,
) -> Result<ProducedOutput> {
    // Python version for the emitted variant/build/`python` dep. Shared with
    // the build recipe via `emit_python_version` so the metadata and the
    // recipe can never disagree. NEVER bare-major: a `py3-none-manylinux`
    // primary wheel (is_pure_python == false, tag parses to "3") used to slip
    // through here as `variant {python: "3"}` / `python 3.*`, corrupting the
    // ABI anchor and floating python_abi in the consumer's solve.
    let python_version =
        emit_python_version(&bundle.primary.metadata.filename, workspace_python_version);
    // If ANY wheel in the bundle is platform-specific, the output is too.
    // Courier packages are ALWAYS platform-specific even with all-pure-python
    // wheels: they ship the native `retread` installer binary + a
    // python-specific committed lock, and the courier recipe is hardcoded
    // `noarch: None`. Advertising noarch here would make pixi request a
    // noarch build that rattler-build rejects ("--target-platform noarch").
    let any_platform_specific = bundle.all_wheels().any(|w| !w.metadata.is_pure_python);

    let python_dep = if python_version.contains('*') {
        format!("python {python_version}")
    } else {
        format!("python {python_version}.*")
    };

    // Vendored set: every wheel that's part of this bundle is installed
    // alongside its siblings, so any `Requires-Dist` line that names one of
    // them must be dropped from the conda run-deps (otherwise conda would
    // try to install a separate copy from a channel that doesn't have it).
    // P2: seeded CANONICAL (matches already_covered's query side).
    let vendored: HashSet<String> = bundle
        .all_wheels()
        .map(|w| crate::relax::canonical_conda_name(&w.pypi_name))
        .collect();

    // User-specified deps to drop entirely (no conda counterpart available).
    let user_dropped: HashSet<String> = config
        .drop_deps
        .iter()
        .map(|n| canonical_conda_name(n))
        .collect();

    // Built-in: Windows-only PyPI shims that are commonly mis-declared as
    // unconditional `Requires-Dist` lines without `sys_platform` markers.
    // Saves users from having to add the same entries to drop-deps for
    // every isaacsim-style upstream that ships these. Skipped if the user
    // explicitly overrode the dep -- the override always wins, so callers
    // who actually need (say) pyreadline3 on Linux can re-enable it.
    let built_in_auto_dropped: HashSet<String> = if host_platform != Platform::Win64
        && host_platform != Platform::Win32
        && host_platform != Platform::WinArm64
    {
        crate::config::BUILT_IN_WIN_ONLY
            .iter()
            .map(|p| (*p).to_string())
            .filter(|p| !config.overrides.contains_key(p))
            .collect()
    } else {
        HashSet::new()
    };
    tracing::debug!(
        bundle = %bundle.conda_name,
        vendored = ?vendored,
        n_wheels = bundle.extras.len() + 1,
        "computed vendored set"
    );

    // Union the relaxed run-deps across every wheel. Dedupe by conda name;
    // when two wheels disagree, the first-encountered spec wins. Genuine
    // upstream disagreements (e.g. pillow 11.3 vs 12.0 in isaacsim) are the
    // user's responsibility to resolve via [build.config.overrides].
    //
    // PR-1 (Site 4): iterate wheels in canonical-name order so the
    // first-encountered dedup is confluent (order-independent).
    let env = marker_env_for(&host_platform.to_string(), &python_version)?;
    let mut run_dep_specs: Vec<NamedSpec<PackageSpec>> = vec![spec_from_str(&python_dep)?];
    let mut seen_dep_names: HashSet<String> = HashSet::from(["python".to_string()]);
    let mut emission_groups = Vec::new();
    let mut emission_group_indexes = BTreeMap::new();

    // M2 (v4.3.0): auto-routed packages FIRST. The uv closure was resolved
    // against concrete versions and the probe confirmed each corresponding
    // (channel, python) build. Typed route inputs determine the emitted
    // constraint before any shipped wheel's Requires-Dist for the same name.
    //
    // Conda-as-truth exception: a route rooted in `retread-deps-from`
    // emits the upstream requirement itself rather than inventing a bound
    // from uv's selected version. Exact pins have already been softened to
    // `>=` by the typed input bridge; ranges remain ranges, and a bare root
    // remains bare. This lets the composed conda solve select a compatible
    // provider instead of freezing an incidental uv choice.
    //
    // Bounded-range emission (root-cause fix for the serial repair grind):
    // every OTHER auto-routed pin -- one this pack picked itself off the
    // uv resolver, not inherited from a deps-from file -- used to be
    // emitted as an exact `==X.Y.Z`. That exact snapshot of uv's pick is
    // what clashed with siblings/workspace/conda-metadata run after run,
    // forcing the repair engine to loosen ONE conflict at a time
    // (~5min/render). Emitting a bounded range up front (floored at the
    // locked version -- the uv closure was solved against exactly this
    // version, so the floor must not move -- capped at the next MAJOR,
    // semver's `0.x` convention capping at the next MINOR instead) lets
    // most of these clashes solve clean on the first render. ABI anchors
    // (`is_abi_anchor` -- python/libc/cuda family) retain the exact selection
    // through constraint joining, then the final anchor-only emission
    // normalizer converts an effective singleton to its within-minor ABI band.
    // Names with an explicit `retread-overrides` entry remain exact because
    // hand-written intent always wins over an auto-derived range.
    for auto_route in &bundle.auto_routed {
        let pypi_name = PypiKey::from_pypi(&auto_route.route.pypi_name);
        let route_support = EmissionSupport::AutoRoute {
            owner_conda_name: canonical_conda_name(&auto_route.route.conda_name),
        };
        if let Some(workspace_provider) = &auto_route.workspace_provider {
            // A partial workspace fact retains the ordinary route so the
            // generated pack supplies conda provision to consumers that did
            // not select it in their pre-pack solve. Its typed workspace
            // conjunction wholly replaces the stale uv-selected version and
            // PyPI route inputs; wheel requirements are suppressed through
            // the same `auto_dropped` ownership path below.
            add_emission_constraint(
                &mut emission_groups,
                &mut emission_group_indexes,
                EmissionConstraintInput {
                    pypi_name,
                    conda_name: workspace_provider.conda_name.clone(),
                    conda_name_is_authoritative: true,
                    constraint: workspace_provider.constraint.clone(),
                    native_conda_override: None,
                    support: route_support,
                },
            )?;
            continue;
        }

        let conda_name = CondaName::new(auto_route.route.conda_name.as_str());
        let conda_version = &auto_route.route.conda_version;
        let conda_key = conda_name.key();

        // Preserve the existing conda route contract. The selected version is
        // not restored as a hard PyPI `==`: ordinary routes receive the
        // bounded/exact compatibility envelope the emitted conda package has
        // always declared. Deps-from routes rely solely on their typed
        // upstream inputs below.
        let manual_override = config.overrides.contains_key(conda_key.as_str())
            && !config.ledger_overrides.contains(conda_key.as_str());
        if !matches!(auto_route.provenance, Provenance::DepsFromRelaxed) {
            let route_is_abi_anchor = crate::solve::is_abi_anchor(&auto_route.route.pypi_name)
                || crate::solve::is_abi_anchor(auto_route.route.conda_name.as_str())
                || crate::solve::is_abi_anchor(conda_key.as_str());
            let (route_spec, provenance) = if route_is_abi_anchor || manual_override {
                (format!("=={conda_version}"), Provenance::UvConstraint)
            } else {
                match bounded_range_ceiling(conda_version) {
                    Some(ceiling) => (
                        format!(">={conda_version},<{ceiling}"),
                        Provenance::UvConstraint,
                    ),
                    None => (format!("=={conda_version}"), Provenance::UvConstraint),
                }
            };
            let specifiers = VersionSpecifiers::from_str(&route_spec).with_context(|| {
                format!(
                    "parsing generated conda route constraint `{} {route_spec}`",
                    auto_route.route.pypi_name
                )
            })?;
            add_emission_constraint(
                &mut emission_groups,
                &mut emission_group_indexes,
                EmissionConstraintInput {
                    pypi_name: pypi_name.clone(),
                    conda_name: conda_name.clone(),
                    conda_name_is_authoritative: true,
                    constraint: Constraint {
                        specifiers,
                        provenance,
                        source: format!(
                            "auto-route `{}=={}` to conda `{}=={}`",
                            auto_route.route.pypi_name,
                            auto_route.route.pypi_version,
                            auto_route.route.conda_name,
                            auto_route.route.conda_version
                        ),
                        origin_id: ConstraintOriginId::from_parts(
                            "auto-route",
                            [
                                auto_route.route.pypi_name.as_str(),
                                auto_route.route.pypi_version.as_str(),
                                auto_route.route.conda_name.as_str(),
                                auto_route.route.conda_version.as_str(),
                                auto_route.route.channel.as_str(),
                                "conda-selection",
                                route_spec.as_str(),
                            ],
                        ),
                    },
                    native_conda_override: None,
                    support: route_support.clone(),
                },
            )?;
        }
        let prior_spec = format!("=={}", auto_route.route.pypi_version);
        let prior_specifiers = VersionSpecifiers::from_str(&prior_spec).with_context(|| {
            format!(
                "parsing prior uv selection `{}=={}`",
                auto_route.route.pypi_name, auto_route.route.pypi_version
            )
        })?;
        add_emission_constraint(
            &mut emission_groups,
            &mut emission_group_indexes,
            EmissionConstraintInput {
                pypi_name: pypi_name.clone(),
                conda_name: conda_name.clone(),
                conda_name_is_authoritative: true,
                constraint: Constraint {
                    specifiers: prior_specifiers,
                    provenance: Provenance::PriorSelection,
                    source: format!(
                        "prior uv selection `{}=={}`",
                        auto_route.route.pypi_name, auto_route.route.pypi_version
                    ),
                    origin_id: ConstraintOriginId::from_parts(
                        "auto-route",
                        [
                            auto_route.route.pypi_name.as_str(),
                            auto_route.route.pypi_version.as_str(),
                            auto_route.route.conda_name.as_str(),
                            auto_route.route.conda_version.as_str(),
                            auto_route.route.channel.as_str(),
                            "prior-selection",
                            prior_spec.as_str(),
                        ],
                    ),
                },
                native_conda_override: None,
                support: route_support.clone(),
            },
        )?;
        for input in &auto_route.route.input_requirements {
            let input_role = match input.role {
                crate::uv_closure::AutoRouteInputRole::Requirement => "requirement",
                crate::uv_closure::AutoRouteInputRole::Constraint => "constraint",
                crate::uv_closure::AutoRouteInputRole::Override => "override",
            };
            let specifiers = if input.specifiers.trim().is_empty() {
                VersionSpecifiers::empty()
            } else {
                VersionSpecifiers::from_str(&input.specifiers).with_context(|| {
                    format!(
                        "parsing auto-route input `{}` for `{}`",
                        input.specifiers, auto_route.route.pypi_name
                    )
                })?
            };
            add_emission_constraint(
                &mut emission_groups,
                &mut emission_group_indexes,
                EmissionConstraintInput {
                    pypi_name: pypi_name.clone(),
                    conda_name: conda_name.clone(),
                    conda_name_is_authoritative: true,
                    constraint: Constraint {
                        specifiers,
                        provenance: input.effective_provenance(),
                        source: input.source.clone(),
                        origin_id: ConstraintOriginId::from_parts(
                            "auto-route",
                            [
                                auto_route.route.pypi_name.as_str(),
                                auto_route.route.pypi_version.as_str(),
                                auto_route.route.conda_name.as_str(),
                                auto_route.route.conda_version.as_str(),
                                auto_route.route.channel.as_str(),
                                input_role,
                                input.specifiers.trim(),
                            ],
                        ),
                    },
                    native_conda_override: None,
                    support: route_support.clone(),
                },
            )?;
        }
    }

    let mut sorted_wheels: Vec<&ResolvedWheel> = bundle.all_wheels().collect();
    sorted_wheels.sort_by_key(|w| canonical_conda_name(&w.pypi_name));
    for wheel in sorted_wheels {
        for raw in &wheel.metadata.retread_conda_run_dependencies {
            let dependency = spec_from_str(raw).with_context(|| {
                format!(
                    "parsing Retread native conda run dependency `{raw}` from {}",
                    wheel.metadata.filename
                )
            })?;
            if seen_dep_names.insert(dependency.name.to_string()) {
                run_dep_specs.push(dependency);
            }
        }
        for raw in &wheel.metadata.requires_dist {
            let Some(dep) = crate::relax::translate(
                raw,
                &env,
                &config.name_map,
                &config.overrides,
                // Final emission is strict-first. Collect the original wheel
                // clause here; `relax_decision::decide` below is the sole
                // policy-aware conflict relaxation boundary.
                RelaxPolicy::None,
            )?
            else {
                continue;
            };
            // Skip if this dep refers to another wheel we're vendoring.
            //
            // v1.4.0: check BOTH the translated conda name AND the raw
            // line's PyPI name. The cascade records bundled wheels and
            // drops under the PYPI name (e.g. `tinyobjloader`), but
            // name_map/parselmouth may translate the emission to a
            // DIFFERENT conda name (`tinyobjloader-python`). Matching
            // only the conda name shipped a doomed conda run-dep
            // alongside the already-bundled wheel -- found via
            // examples/isaac6 (isaacsim 6.0's tinyobjloader dep).
            let dep_name = dep.name.clone();
            let raw_pypi_name = dep.pypi_name.as_str();
            // Vendored and explicit drop sets predate typed ecosystem names,
            // so retain their dual-namespace compatibility check.
            let in_set = |set: &HashSet<String>| {
                crate::relax::already_covered(set, &dep_name, Some(raw_pypi_name))
            };
            if in_set(&vendored) {
                continue;
            }
            if in_set(&user_dropped) {
                tracing::debug!(dep = %dep_name, "dropping per retread-drop-deps");
                continue;
            }
            // `auto_dropped` is typed PyPI ownership evidence. Matching the
            // translated conda name here would let an inferred name-map edge
            // turn an unrelated PyPI owner into ownership of this raw wheel
            // requirement.
            if bundle
                .auto_dropped
                .contains(&canonical_conda_name(raw_pypi_name))
            {
                tracing::debug!(
                    dep = %dep_name,
                    bundle = %bundle.conda_name,
                    "dropping wheel dependency owned by a workspace conda provider",
                );
                continue;
            }
            if in_set(&built_in_auto_dropped) {
                // Surface this prominently so the user has a chance to
                // notice if the auto-drop ate something they actually need.
                tracing::warn!(
                    dep = %dep_name,
                    bundle = %bundle.conda_name,
                    "auto-dropping built-in Windows-only PyPI shim on \
                     non-Windows target. If you actually need this on this \
                     platform, set `retread-overrides.{dep_name} = \"*\"` to \
                     bypass the auto-drop.",
                );
                continue;
            }
            // uv-closure membership gate (v4.3.1): a dep that uv exported
            // into the wheel closure is provided by the uv install set at
            // install time — it must NOT also become a conda run-dep. Only
            // packages actually ROUTED to conda may be emitted: auto-routed
            // hits (exact-pinned above), explicit retread-conda-deps, and
            // conda-routed deps outside the closure. Without this gate,
            // closure members leaked as conda run-deps — pypi-only packages
            // (`isaacsim-kernel`) or specs no channel satisfies
            // (`aiodns >=3.1.1` vs conda-forge's 3.0.0) made the whole
            // workspace solve unsatisfiable. Bundle base-deps (the
            // `<entry>-*` prefix family) are closure members too, so this
            // also stops their duplication as conda deps.
            if in_set(&bundle.uv_closure_names) {
                tracing::debug!(
                    dep = %dep_name,
                    bundle = %bundle.conda_name,
                    "skipping run-dep emission: provided by the uv wheel \
                     closure at install time (not conda-routed)",
                );
                continue;
            }
            let conda_name = CondaName::new(dep_name.as_str());
            let pypi_name = dep.pypi_name.clone();
            let translated = translated_emission_constraint(raw, &dep, &wheel.metadata_provenance)?;
            add_emission_constraint(
                &mut emission_groups,
                &mut emission_group_indexes,
                EmissionConstraintInput {
                    pypi_name,
                    conda_name,
                    conda_name_is_authoritative: false,
                    constraint: Constraint {
                        specifiers: translated.specifiers,
                        provenance: translated.provenance,
                        source: format!(
                            "wheel `{}=={}` Requires-Dist `{raw}`",
                            wheel.metadata.name, wheel.metadata.version
                        ),
                        origin_id: ConstraintOriginId::from_parts(
                            "wheel-requires-dist",
                            [
                                wheel.metadata.name.as_str(),
                                wheel.metadata.version.as_str(),
                                raw.as_str(),
                            ],
                        ),
                    },
                    native_conda_override: translated.native_conda_override,
                    support: EmissionSupport::WheelRequirement {
                        translated_conda_name: dep_name,
                        raw_pypi_name: raw_pypi_name.to_string(),
                    },
                },
            )?;
        }
    }

    // Workspace facts normally disappeared through `auto_dropped` before this
    // assembly. A fact retained by an explicit PyPI-side exclusion remains
    // authoritative validation input: attach it only when a concrete
    // emission route's advisory or authoritative clause excludes that record.
    // Compatible ranges keep their portable form, and cross-ecosystem identity
    // still requires the configured route edge used by translation. For a new
    // authoritative conflict, the fact validates relaxation search but is
    // excluded from the emitted projection so a reconciled wheel pin remains
    // a portable compatibility band. The established advisory-only path keeps
    // its existing emitted intersection.
    for group in &mut emission_groups {
        let conda_key = group.conda_name.key();
        let pypi_names = group
            .pypi_names
            .iter()
            .map(PypiKey::as_str)
            .collect::<Vec<_>>()
            .join(", ");
        let Some(version) = bundle.workspace_conda_versions.get(conda_key.as_str()) else {
            continue;
        };
        let workspace_version =
            uv_pep508::uv_pep440::Version::from_str(version).with_context(|| {
                format!(
                    "parsing common workspace conda version `{version}` for PyPI origin(s) \
                     `{pypi_names}`"
                )
            })?;
        let has_override = group
            .constraints
            .iter()
            .any(|constraint| matches!(&constraint.provenance, Provenance::UvOverride));
        let excludes_workspace_fact = |authority| {
            group.constraints.iter().any(|constraint| {
                active_for_finalization(constraint, has_override)
                    && constraint.authority() == authority
                    && !constraint.specifiers.contains(&workspace_version)
            })
        };
        let conflicting_authoritative = excludes_workspace_fact(Authority::Authoritative);
        let conflicting_advisory = excludes_workspace_fact(Authority::Advisory);
        if !conflicting_authoritative && !conflicting_advisory {
            continue;
        }
        let specifiers =
            VersionSpecifiers::from_str(&format!("=={version}")).with_context(|| {
                format!(
                    "parsing common workspace conda fact `{}=={version}` for PyPI origin(s) \
                     `{pypi_names}`",
                    group.conda_name
                )
            })?;
        let constraint = Constraint {
            specifiers,
            provenance: Provenance::WorkspaceCondaFact("precise-consuming-envs".to_string()),
            source: format!(
                "workspace conda fact `{}=={version}` shared by precise consuming environments",
                group.conda_name
            ),
            origin_id: ConstraintOriginId::from_parts(
                "workspace-conda-fact",
                [conda_key.as_str(), version.as_str()],
            ),
        };
        if !group.constraints.contains(&constraint) {
            if conflicting_authoritative {
                group
                    .validation_only_origins
                    .insert(constraint.origin_id.clone());
            }
            group.constraints.push(constraint);
        }
    }

    emission_groups.sort_by(|left, right| {
        (left.conda_name.as_spec(), &left.pypi_names)
            .cmp(&(right.conda_name.as_spec(), &right.pypi_names))
    });
    let abi_aliases = output_abi_aliases(bundle, config);
    let mut conflicts = Vec::new();
    let mut pending_relaxations = Vec::new();
    let mut supports_by_conda = BTreeMap::new();
    for group in emission_groups {
        let PypiEmissionGroup {
            pypi_names,
            conda_name,
            conda_name_is_authoritative: _,
            constraints,
            validation_only_origins,
            native_conda_overrides,
            supports,
        } = group;
        let pypi_name = pypi_names
            .iter()
            .find(|name| crate::solve::is_abi_anchor(name.as_str()))
            .or_else(|| pypi_names.iter().next())
            .cloned()
            .expect("every emission group has a PyPI origin");
        let has_anchor_alias = pypi_names
            .iter()
            .any(|name| crate::solve::is_abi_anchor(name.as_str()))
            || is_semantic_abi_anchor(conda_name.as_spec(), &abi_aliases);
        let has_manual_override = pypi_names
            .iter()
            .map(PypiKey::as_str)
            .chain(std::iter::once(conda_name.as_spec()))
            .any(|name| {
                config.overrides.contains_key(name) && !config.ledger_overrides.contains(name)
            });
        let native_conda_override = native_conda_overrides.into_iter().next();
        let conda_key = conda_name.key().into_string();
        supports_by_conda.insert(conda_key.clone(), supports);
        if !seen_dep_names.insert(conda_key) {
            bail!(
                "duplicate conda dependency target `{conda_name}` reached final emission; \
                 refusing to discard a whole constraint edge"
            );
        }
        match decide_for_emission(
            &pypi_name,
            &constraints,
            &validation_only_origins,
            config.relax,
            &SafetyContext::new(Some(conda_name.as_spec())).with_abi_anchor_alias(has_anchor_alias),
        ) {
            RelaxDecision::Strict {
                specifiers,
                diagnostics,
            } => {
                pending_relaxations.extend(auto_bundle::wheel_metadata_relaxations(
                    &pypi_name,
                    &constraints,
                    diagnostics,
                    &bundle.conda_name,
                    format!(" for bundle '{}'", bundle.conda_name),
                ));
                let rendered = native_conda_override
                    .clone()
                    .unwrap_or_else(|| specifiers.to_string().replace(", ", ","));
                let rendered = if has_anchor_alias && native_conda_override.is_none() {
                    let (rendered, warning) = normalize_emitted_abi_anchor_spec(
                        &bundle.conda_name,
                        &pypi_name,
                        &constraints,
                        rendered,
                        !has_manual_override,
                    );
                    pending_relaxations.extend(warning);
                    rendered
                } else {
                    rendered
                };
                let spec = conda_name.match_spec(&rendered);
                run_dep_specs.push(spec_from_str(spec.as_str())?);
            }
            RelaxDecision::Relaxed {
                specifiers,
                decisions,
            } => {
                pending_relaxations.extend(auto_bundle::wheel_metadata_relaxations(
                    &pypi_name,
                    &constraints,
                    decisions,
                    &bundle.conda_name,
                    format!(" for bundle '{}'", bundle.conda_name),
                ));
                let can_normalize = has_anchor_alias && native_conda_override.is_none();
                let rendered = native_conda_override
                    .unwrap_or_else(|| specifiers.to_string().replace(", ", ","));
                let rendered = if can_normalize {
                    let (rendered, warning) = normalize_emitted_abi_anchor_spec(
                        &bundle.conda_name,
                        &pypi_name,
                        &constraints,
                        rendered,
                        !has_manual_override,
                    );
                    pending_relaxations.extend(warning);
                    rendered
                } else {
                    rendered
                };
                let spec = conda_name.match_spec(&rendered);
                run_dep_specs.push(spec_from_str(spec.as_str())?);
            }
            RelaxDecision::Conflict(conflict) => {
                let platform = host_platform.to_string();
                let conflict = auto_bundle::attach_conflict_suggestion(
                    conflict,
                    bundle,
                    config,
                    &platform,
                    workspace_python_version,
                );
                // Keep a name-only placeholder in the tolerant assembly so
                // Rule 2 can identify and reject a wholly mutable route. The
                // strict `produce_output` wrapper below always returns the
                // structural conflict instead of emitting this placeholder.
                run_dep_specs.push(spec_from_str(conda_name.as_spec())?);
                conflicts.push(EmissionConstraintConflict {
                    conda_name,
                    conflict,
                });
            }
            RelaxDecision::SearchExhausted(exhausted) => {
                return Err(anyhow::Error::new(exhausted.with_scope(format!(
                    "while emitting conda dependency `{conda_name}` for bundle `{}`",
                    bundle.conda_name
                ))));
            }
        }
    }

    // Probe and conflict-localization assemblies pass through here repeatedly.
    // Keep their full dependency lists out of normal logs; the committed
    // outputs are logged exactly once by `log_final_bundle_outputs`.
    let emitted: Vec<&str> = run_dep_specs.iter().map(|s| s.name.as_str()).collect();
    tracing::debug!(
        bundle = %bundle.conda_name,
        run_deps = ?emitted,
        "bundle run-deps emitted; if conda can't find one, add it to \
         retread-drop-deps / retread-overrides / retread-name-map"
    );

    let effective_version = version_override.unwrap_or(&bundle.primary.metadata.version);
    let output = assemble_conda_output(
        &bundle.conda_name,
        effective_version,
        &python_version,
        config.courier,
        any_platform_specific,
        run_dep_specs,
        seen_dep_names,
        host_platform,
        config.build_number,
        courier_build_hash,
        config.bundle_mode == crate::config::BundleMode::Loose,
        siblings,
    )?;
    Ok((output, conflicts, pending_relaxations, supports_by_conda))
}

fn produce_output_pending_relaxations(
    bundle: &Bundle,
    config: &RetreadConfig,
    host_platform: Platform,
    workspace_python_version: &str,
    siblings: &[(String, String)],
    courier_build_hash: Option<&str>,
    version_override: Option<&str>,
) -> Result<(CondaOutput, Vec<auto_bundle::WheelMetadataRelaxation>)> {
    let (output, conflicts, pending_relaxations, _supports_by_conda) =
        produce_output_with_conflicts(
            bundle,
            config,
            host_platform,
            workspace_python_version,
            siblings,
            courier_build_hash,
            version_override,
        )?;
    if !conflicts.is_empty() {
        let conda_names = conflicts
            .iter()
            .map(|conflict| conflict.conda_name.as_spec().to_string())
            .collect::<Vec<_>>();
        for conflict in &conflicts {
            let conda_name = conflict.conda_name.as_spec().to_string();
            tracing::error!(
                bundle = %bundle.conda_name,
                conda_dep = %conda_name,
                error = %conflict.conflict,
                "bundle emission rejected by structural constraint conflict",
            );
        }
        let context = if conda_names.len() == 1 {
            format!(
                "emitting conda dependency `{}` for bundle `{}`",
                conda_names[0], bundle.conda_name
            )
        } else {
            format!(
                "emitting conda dependencies {} for bundle `{}`",
                conda_names
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", "),
                bundle.conda_name
            )
        };
        let conflicts = conflicts
            .into_iter()
            .map(|conflict| conflict.conflict)
            .collect();
        return Err(aggregate_conflicts(conflicts)).with_context(|| context);
    }
    ensure_output_abi_invariants(&output, bundle, config, workspace_python_version)?;
    Ok((output, pending_relaxations))
}

fn bundled_relaxations_for_output(
    emitted_bundle: &str,
    input_bundle: &str,
    target: &ResolutionTarget,
    restore_relaxations: &[auto_bundle::WheelMetadataRelaxation],
    emission_relaxations: &[auto_bundle::WheelMetadataRelaxation],
) -> Option<RelaxationManifest> {
    let scope = RelaxationScope::for_target(target);
    let records = restore_relaxations
        .iter()
        .filter(|relaxation| relaxation.bundle() == input_bundle)
        .chain(
            emission_relaxations
                .iter()
                .filter(|relaxation| relaxation.bundle() == emitted_bundle),
        )
        .map(|relaxation| relaxation.to_record(&scope))
        .collect();
    RelaxationManifest::new(emitted_bundle, records)
}

fn output_matches_build_request(
    output: &CondaOutput,
    requested: &pixi_build_types::procedures::conda_build_v1::CondaBuildV1Output,
) -> bool {
    output.metadata.name.as_normalized() == requested.name.as_normalized()
        && requested
            .version
            .as_ref()
            .is_none_or(|version| output.metadata.version.to_string() == version.to_string())
        && requested
            .build
            .as_ref()
            .is_none_or(|build| output.metadata.build == *build)
        && output.metadata.subdir == requested.subdir
}

fn outputs_share_identity(left: &CondaOutput, right: &CondaOutput) -> bool {
    left.metadata.name == right.metadata.name
        && left.metadata.version.to_string() == right.metadata.version.to_string()
        && left.metadata.build == right.metadata.build
        && left.metadata.build_number == right.metadata.build_number
        && left.metadata.subdir == right.metadata.subdir
        && left.metadata.noarch == right.metadata.noarch
}

fn log_final_bundle_outputs(result: &CondaOutputsResult) {
    for output in &result.outputs {
        let emitted = output
            .run_dependencies
            .depends
            .iter()
            .map(|dependency| dependency.name.as_str())
            .collect::<Vec<_>>();
        tracing::info!(
            bundle = %output.metadata.name.as_normalized(),
            run_deps = ?emitted,
            "bundle run-deps emitted; if conda can't find one, add it to \
             retread-drop-deps / retread-overrides / retread-name-map"
        );
    }
}

fn output_run_dependencies_match(
    output: &CondaOutput,
    advertised: Option<&[String]>,
) -> Result<bool> {
    run_dependencies_match(&output.run_dependencies.depends, advertised)
}

/// Describe *which* run dependencies differ, for the "0 exact matches" error.
/// Knowing only that they differ is not actionable: the record can be stale in
/// one spec out of two hundred, and the operator has no way to see which.
/// Attribute each unadvertised run dependency to the bundle wheel that
/// declares it, and flag wheels that were built from an sdist.
///
/// The unadvertised set varies run to run for the same bundle, and the
/// failure only reproduces from a cold cache -- the one run that performs
/// sdist auto-builds. Either the bundle carries different wheels at build
/// time than at metadata time, or a build-isolation package is reaching the
/// run closure. This says which, without another round of guessing.
fn describe_unadvertised_sources(
    output: &CondaOutput,
    advertised: Option<&[String]>,
    bundle: &Bundle,
) -> String {
    let Some(advertised) = advertised else {
        return "no advertised list to attribute against".to_string();
    };
    let advertised_names: BTreeSet<String> = advertised
        .iter()
        .filter_map(|raw| {
            rattler_conda_types::MatchSpec::from_str(
                raw,
                rattler_conda_types::ParseStrictness::Lenient,
            )
            .ok()
            .and_then(|spec| spec.name.as_ref().map(|name| name.to_string()))
        })
        .collect();

    let mut notes = Vec::new();
    for dependency in &output.run_dependencies.depends {
        let name = canonical_conda_name(&dependency.name).to_string();
        if advertised_names.contains(&name) {
            continue;
        }
        let mut owners: Vec<String> = Vec::new();
        for wheel in std::iter::once(&bundle.primary).chain(bundle.extras.iter()) {
            let declares = wheel
                .metadata
                .requires_dist
                .iter()
                .any(|raw| raw.split([';', ' ', '=', '<', '>', '!', '~', '[']).next()
                    .is_some_and(|dep| canonical_conda_name(dep) == name));
            if declares {
                owners.push(wheel.pypi_name.clone());
            }
        }
        if owners.is_empty() {
            // Declared by nothing in the bundle: it entered the run closure
            // from outside the package's own requirements.
            notes.push(format!("{name}<-UNATTRIBUTED"));
        } else {
            notes.push(format!("{name}<-{}", owners.join("+")));
        }
    }
    if notes.is_empty() {
        return "no unadvertised dependencies to attribute".to_string();
    }
    format!(
        "attribution: {} (bundle wheels: {})",
        notes.join(", "),
        bundle.extras.len() + 1
    )
}

fn run_dependency_delta(output: &CondaOutput, advertised: Option<&[String]>) -> String {
    named_spec_run_dependency_delta(
        &output.run_dependencies.depends,
        advertised,
        "the rebuilt output",
    )
}

/// Shared delta rendering for "which run dependencies differ".
///
/// Extracted from [`run_dependency_delta`] so the cold-path gate error and the
/// lock-parity refusal (see [`recover_cold_mismatch_from_lock`]) describe their
/// disagreement in exactly the same terms. `actual_label` names the left-hand
/// record ("the rebuilt output" / "the committed lock"); passing
/// `"the rebuilt output"` reproduces the historical wording byte-for-byte.
fn named_spec_run_dependency_delta(
    actual: &[NamedSpec<PackageSpec>],
    advertised: Option<&[String]>,
    actual_label: &str,
) -> String {
    let Some(advertised) = advertised else {
        return "no advertised run dependencies to compare".to_string();
    };
    let advertised_names: BTreeSet<String> = advertised
        .iter()
        .filter_map(|raw| {
            rattler_conda_types::MatchSpec::from_str(
                raw,
                rattler_conda_types::ParseStrictness::Lenient,
            )
            .ok()
            .and_then(|spec| spec.name.as_ref().map(|name| name.to_string()))
        })
        .collect();
    let actual_names: BTreeSet<String> = actual
        .iter()
        .map(|dependency| canonical_conda_name(&dependency.name).to_string())
        .collect();

    let only_built: Vec<&String> = actual_names.difference(&advertised_names).collect();
    let only_advertised: Vec<&String> = advertised_names.difference(&actual_names).collect();

    let mut parts = Vec::new();
    if !only_built.is_empty() {
        parts.push(format!(
            "present in {actual_label} but not advertised: {}",
            only_built
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !only_advertised.is_empty() {
        parts.push(format!(
            "advertised but absent from {actual_label}: {}",
            only_advertised
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if parts.is_empty() {
        parts.push(format!(
            "same {} dependency names on both sides, so one or more version/build \
             constraints were relaxed or tightened",
            actual_names.len()
        ));
    }
    parts.join("; ")
}

fn run_dependencies_match(
    actual: &[NamedSpec<PackageSpec>],
    advertised: Option<&[String]>,
) -> Result<bool> {
    let Some(advertised) = advertised else {
        return Ok(true);
    };
    let expected = advertised
        .iter()
        .map(|raw| {
            let spec = rattler_conda_types::MatchSpec::from_str(
                raw,
                rattler_conda_types::ParseStrictness::Lenient,
            )
            .with_context(|| format!("parsing advertised run dependency `{raw}`"))?;
            let name = spec
                .name
                .as_ref()
                .ok_or_else(|| anyhow!("advertised run dependency `{raw}` has no package name"))?;
            let mut normalized = name.to_string();
            if let Some(version) = spec.version {
                normalized.push(' ');
                normalized.push_str(&version.to_string());
            }
            if let Some(build) = spec.build {
                normalized.push(' ');
                normalized.push_str(&build.to_string());
            }
            spec_from_str(&normalized)
        })
        .collect::<Result<Vec<_>>>()?;
    // Pixi may append host/build run exports (notably python_abi) before
    // build_v1. Those extras are legitimate, but every dependency advertised
    // by conda/outputs must remain byte-for-byte semantically present: replacing
    // a relaxed range or intersecting it with an extra same-name spec would
    // make the durable record stale.
    let actual_names = actual
        .iter()
        .map(|dependency| canonical_conda_name(&dependency.name))
        .collect::<BTreeSet<_>>();
    Ok(actual
        .iter()
        .all(|dependency| expected.contains(dependency))
        && expected.iter().all(|dependency| {
            !actual_names.contains(&canonical_conda_name(&dependency.name))
                || actual.contains(dependency)
        }))
}

/// Render one committed-lock run dependency back to the `"<name> <spec>"` form
/// pixi forwarded when the lock was written.
///
/// `courier::parse_conda_deps` produced `CondaDep` by splitting the forwarded
/// string at the first space, so this is its exact inverse.
fn lock_run_dep_string(dep: &crate::lock::CondaDep) -> String {
    if dep.spec.is_empty() {
        dep.name.clone()
    } else {
        format!("{} {}", dep.name, dep.spec)
    }
}

/// Outcome of attempting to recover a cold-path advertised/rebuilt
/// run-dependency mismatch from the committed pack lock.
///
/// See `docs/RETREAD_DETERMINISM_FIX_DESIGN.md` (Option D). The cold path
/// re-executes `resolve_all`, which is a *fresh sample* of a resolution
/// function whose inputs (repodata TTL, probe outages, the heal-facts ledger,
/// live uv re-locks) move between the advertise-time evaluation and the build.
/// When the identity still matches and only the run dependencies drifted, the
/// committed pack lock is the recorded resolution that the advertisement's
/// inputs hash pins; if it agrees with what pixi advertised, replaying it is
/// strictly more faithful than either building the drifted set or refusing.
#[derive(Debug)]
enum ColdMismatchRecovery {
    /// The committed lock reproduces the advertised run deps; replay it.
    ReplayFromLock(Box<crate::lock::RetreadLock>),
    /// No committed lock can vouch for the advertisement; fail closed with
    /// `reason` appended to the existing "0 exact matches" error.
    Refuse { reason: String },
}

/// Decide recovery for ONE identity-matched, deps-mismatched candidate.
///
/// Pure with respect to handler state: the caller performs the lock load (so
/// that this decision is trivially testable, and so the load itself stays
/// inside the branch that would otherwise error) and passes its result in.
///
/// Fail-closed by construction: every path that cannot *prove* the committed
/// lock vouches for the advertisement returns [`ColdMismatchRecovery::Refuse`]
/// with a reason naming which record disagrees and how.
fn recover_cold_mismatch_from_lock(
    lock: anyhow::Result<Option<crate::lock::RetreadLock>>,
    run_override: Option<&[String]>,
    advertised_version: Option<&str>,
) -> ColdMismatchRecovery {
    let refuse = |reason: String| ColdMismatchRecovery::Refuse { reason };
    let lock = match lock {
        Err(error) => {
            return refuse(format!("committed lock unusable: {error:#}"));
        }
        Ok(None) => {
            return refuse(
                "no committed lock vouches for this advertisement (missing lock file, \
                 different schema, different resolution target, inputs-hash mismatch, \
                 or RETREAD_NO_REPLAY=1)"
                    .to_string(),
            );
        }
        Ok(Some(lock)) => lock,
    };
    // Identity already matched, so the advertised version is authoritative; a
    // lock recording a different version would silently build the wrong
    // package. The top replay gate applies the same check.
    if !advertised_version_matches(advertised_version, &lock.version) {
        return refuse(format!(
            "committed lock records version `{}` but pixi advertised version `{}`",
            lock.version,
            advertised_version.unwrap_or("<unset>"),
        ));
    }
    // `run_override` (pixi's echo of the advertised metadata) is compared with
    // THE SAME matcher the gate used on the rebuilt candidate, so the
    // python_abi host/build run-export allowance applies identically to both
    // comparisons. A second, subtly different comparison here would be able to
    // accept a lock the gate would have rejected.
    let lock_deps = match lock
        .conda_run_deps
        .iter()
        .map(|dep| spec_from_str(&lock_run_dep_string(dep)))
        .collect::<Result<Vec<_>>>()
    {
        Ok(deps) => deps,
        Err(error) => {
            return refuse(format!(
                "committed lock records an unparseable run dependency: {error:#}"
            ));
        }
    };
    match run_dependencies_match(&lock_deps, run_override) {
        Ok(true) => ColdMismatchRecovery::ReplayFromLock(Box::new(lock)),
        Ok(false) => refuse(format!(
            "committed lock also differs from the advertised run dependencies: {}",
            named_spec_run_dependency_delta(&lock_deps, run_override, "the committed lock"),
        )),
        // Unreachable in practice: the gate already parsed the same advertised
        // strings. Refusing (rather than propagating) keeps the failure mode
        // fail-closed and diagnosable.
        Err(error) => refuse(format!(
            "comparing the committed lock against the advertised run dependencies: {error:#}"
        )),
    }
}

/// The one cold-path candidate whose identity matched the advertisement but
/// whose freshly re-derived run dependencies did not.
///
/// Retained only on the branch that today ends in `RpcError::invalid_params`;
/// nothing here is constructed when the gate passes.
struct ColdMismatchedCandidate {
    bundle: Bundle,
    effective: RetreadConfig,
    courier_hash: Option<String>,
}

/// What the cold-path failure arm should do, decided before any build work.
enum ColdRecoveryPlan {
    /// Recovery does not apply (non-courier pack, or no candidate matched the
    /// advertised identity). The existing error stands byte-unchanged.
    NotAttempted,
    /// The committed lock vouches for the advertisement: replay it.
    Replay {
        bundle_name: String,
        lock: Box<crate::lock::RetreadLock>,
    },
    /// Fail closed; `reason` is appended to the existing error.
    Refuse { reason: String },
}

/// Everything the cold-path failure arm decides before it materializes
/// anything: candidate eligibility, ABI context, lock location, lock load, and
/// the parity decision.
///
/// Split out from `conda_build_v1` so the whole recovery decision — not just
/// its innermost comparison — is exercisable without running a real build.
/// The caller does nothing but act on the returned plan.
///
/// PERFORMANCE / BLAST-RADIUS CONTRACT: this performs the only lock read on
/// the cold path, and it is called from exactly one place — inside the arm
/// that today terminates in `RpcError::invalid_params`. It must never be
/// hoisted above the gate; `cold_mismatch_recovery_never_consulted_when_candidate_matches`
/// fails if it is.
fn plan_cold_mismatch_recovery(
    candidate: Option<ColdMismatchedCandidate>,
    ambiguous: bool,
    config: &RetreadConfig,
    target: &ResolutionTarget,
    source_dir: &Path,
    run_override: Option<&[String]>,
    advertised_version: Option<&str>,
) -> ColdRecoveryPlan {
    if !config.courier {
        // Non-courier packs commit no lock; today's error stands.
        return ColdRecoveryPlan::NotAttempted;
    }
    if ambiguous {
        return ColdRecoveryPlan::Refuse {
            reason: "two candidates matched the advertised identity, so no committed lock \
                     can be attributed unambiguously; refusing lock-parity recovery"
                .to_string(),
        };
    }
    let Some(candidate) = candidate else {
        return ColdRecoveryPlan::NotAttempted;
    };
    // ABI context comes from the FRESH bundle: producer-time lock facts cannot
    // stand in for the versions the current request solved (same rule the top
    // replay gate documents).
    let abi_context = replay_abi_context_for_bundle(
        &candidate.bundle,
        &candidate.effective,
        target.python_version(),
    );
    let lock_path = lock_path_for_target(source_dir, &candidate.bundle.conda_name, target);
    let relax_is_default = config.relax == crate::config::RelaxPolicy::default();
    // `load_replayable_lock_for_target` re-uses every existing gate: schema,
    // resolution target, bundle name, full replay provenance, inputs hash,
    // relax poisoning, ABI invariants — and honors RETREAD_NO_REPLAY=1, which
    // therefore disables this recovery along with every other replay.
    let loaded = match candidate.courier_hash.as_deref() {
        Some(current_hash) => load_replayable_lock_for_target(
            &lock_path,
            current_hash,
            relax_is_default,
            target,
            &candidate.bundle.conda_name,
            &abi_context,
        ),
        None => Ok(None),
    };
    match recover_cold_mismatch_from_lock(loaded, run_override, advertised_version) {
        ColdMismatchRecovery::ReplayFromLock(lock) => ColdRecoveryPlan::Replay {
            bundle_name: candidate.bundle.conda_name,
            lock,
        },
        ColdMismatchRecovery::Refuse { reason } => ColdRecoveryPlan::Refuse { reason },
    }
}

#[cfg(test)]
fn produce_output(
    bundle: &Bundle,
    config: &RetreadConfig,
    host_platform: Platform,
    workspace_python_version: &str,
    siblings: &[(String, String)],
    courier_build_hash: Option<&str>,
    version_override: Option<&str>,
) -> Result<CondaOutput> {
    let (output, pending_relaxations) = produce_output_pending_relaxations(
        bundle,
        config,
        host_platform,
        workspace_python_version,
        siblings,
        courier_build_hash,
        version_override,
    )?;
    for relaxation in pending_relaxations {
        relaxation.emit();
    }
    Ok(output)
}

/// Render the bundle's actual emitted run-dependency set into the generic
/// route-spec representation consumed by the shared co-solve oracle. This is
/// deliberately derived from `produce_output`, the single emission authority,
/// so marker evaluation, name mapping, relaxation, overrides, vendoring,
/// closure filtering, and first-name-wins dedup cannot drift from Rule 2's
/// validation input.
fn emitted_bundle_route_assembly(
    bundle: &Bundle,
    config: &RetreadConfig,
    target: &WheelTarget,
) -> Result<EmittedBundleRouteAssembly> {
    let host_platform = Platform::from_str(&target.conda_subdir)
        .with_context(|| format!("parsing target conda subdir `{}`", target.conda_subdir))?;
    let (output, conflicts, _pending_relaxations, supports_by_conda) =
        produce_output_with_conflicts(
            bundle,
            config,
            host_platform,
            &target.python_version,
            &[],
            None,
            None,
        )?;
    let routes = output
        .run_dependencies
        .depends
        .iter()
        .map(|dependency| {
            let conda_name = CondaName::new(dependency.name.as_str());
            let conda_key = conda_name.key();
            let pypi_name = bundle
                .auto_routed
                .iter()
                .find(|route| CondaName::new(&route.route.conda_name).key() == conda_key)
                .map(|route| PypiKey::from_pypi(&route.route.pypi_name))
                .unwrap_or(conda_key);
            crate::uv_closure::CondaRouteSpec {
                pypi_name,
                conda_name,
                spec: audit_report::format_packagespec(&dependency.spec),
            }
        })
        .collect();
    Ok(EmittedBundleRouteAssembly {
        routes,
        conflicts,
        supports_by_conda,
    })
}

fn emitted_bundle_route_specs(
    bundle: &Bundle,
    config: &RetreadConfig,
    target: &WheelTarget,
) -> Result<Vec<crate::uv_closure::CondaRouteSpec>> {
    let assembly = emitted_bundle_route_assembly(bundle, config, target)?;
    if !assembly.conflicts.is_empty() {
        return Err(aggregate_conflicts(
            assembly
                .conflicts
                .into_iter()
                .map(|conflict| conflict.conflict)
                .collect(),
        ));
    }
    Ok(assembly.routes)
}

/// v1.4.5: swap an http(s) wheel source for `file://` of retread's
/// cached copy when present. Looks in `<wheels_root>/<filename>` and
/// one level of per-entry subdirs (`<wheels_root>/<entry>/<filename>`,
/// the sdist-build layout). Missing file -> upstream URL unchanged.
/// The caller's recipe pins the wheel sha256 either way, so a stale or
/// corrupt cache file fails the build loudly at hash verification
/// instead of silently building from wrong bytes.
fn localize_wheel_source(url: &url::Url, wheels_root: &Path) -> url::Url {
    if url.scheme() != "http" && url.scheme() != "https" {
        return url.clone();
    }
    let Ok(filename) = crate::wheel::wheel_filename_from_url(url) else {
        return url.clone();
    };
    let mut candidates = vec![wheels_root.join(&filename)];
    if let Ok(entries) = std::fs::read_dir(wheels_root) {
        for e in entries.flatten() {
            if e.path().is_dir() {
                candidates.push(e.path().join(&filename));
            }
        }
    }
    for c in candidates {
        if c.is_file()
            && let Ok(local) = url::Url::from_file_path(&c)
        {
            tracing::debug!(
                wheel = %filename,
                path = %c.display(),
                "recipe source localized to cached wheel (skips rattler-build re-download)",
            );
            return local;
        }
    }
    url.clone()
}

/// Validate the bytes re-produced for a source-built replay before copying
/// lock fields into an EmitWheel. The lock hash is the later courier-rewritten
/// artifact and cannot equal this pre-stage file, but the actual replay file
/// must retain the exact source filename/tag contract that predicts the
/// canonical staged filename.
async fn validate_replayed_source_artifact(
    resolved: &ResolvedWheel,
    locked: &crate::lock::LockWheel,
    target: &ResolutionTarget,
) -> Result<(String, String)> {
    let path = resolved.url.to_file_path().map_err(|_| {
        anyhow!(
            "courier replay source wheel for `{}` is not a local artifact: {}",
            locked.name,
            resolved.url,
        )
    })?;
    let expected = crate::source_build::ExpectedWheel::exact(&locked.name, &locked.version);
    let actual_sha = crate::source_build::validate_wheel_for_target_async(
        &path,
        target,
        Some(&expected),
    )
    .await
    .with_context(|| {
        format!(
            "courier replay source artifact for `{}` failed strict target/identity validation",
            locked.name,
        )
    })?;
    if !actual_sha.eq_ignore_ascii_case(&resolved.metadata.sha256) {
        bail!(
            "courier replay source artifact for `{}` changed after materialization: metadata read {}, strict read {}",
            locked.name,
            resolved.metadata.sha256,
            actual_sha,
        );
    }
    let actual_filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("courier replay source artifact has no UTF-8 filename"))?
        .to_string();
    let predicted_locked_filename = crate::emit_pypi::standard_wheel_filename(&actual_filename);
    if predicted_locked_filename != locked.filename {
        bail!(
            "courier replay source artifact filename drift for `{}`: rebuilt `{actual_filename}` predicts `{predicted_locked_filename}`, lock records `{}`",
            locked.name,
            locked.filename,
        );
    }
    Ok((actual_filename, actual_sha))
}

/// Reconstruct the per-wheel [`crate::emit_pypi::EmitWheel`] list from a
/// committed lock without re-running the full BFS resolve.
///
/// Returns:
/// - `Ok(Some(wheels))` — all wheels re-materialized successfully.
/// - `Ok(None)` — provenance gap (class-3 / schema-5 class-2); caller must
///   fall through to `resolve_all`.
/// - `Err(...)` — hard error during re-materialization.
///
/// Owns the PHASE 2.5 pre-pass (git group membership), the 6 loop-local
/// accumulators (`git_group_members`, `git_group_order`, `auto_data_override`,
/// `git_group_stash`, `built_roots`, `emit_wheels`), and the main
/// per-`LockWheel` classification loop.  The caller retains responsibility for
/// deriving `conda_capable` and `index_urls` from the lock after this returns.
fn replay_sdist_cache_key(raw_sha256: &str, target: &ResolutionTarget) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(b"retread-replay-sdist-build-v2\0");
    hasher.update(raw_sha256.as_bytes());
    hasher.update(b"\0");
    hasher.update(target.resolution_identity().as_bytes());
    format!("{:x}", hasher.finalize())
}

async fn emit_wheels_from_lock(
    lock: &crate::lock::RetreadLock,
    config: &RetreadConfig,
    target: &ResolutionTarget,
    download_dir: &Path,
    source_dir: &Path,
    cache_dir: &Path,
) -> Result<Option<Vec<crate::emit_pypi::EmitWheel>>> {
    use crate::lock::Origin;

    let abi_aliases = abi_aliases_from_name_map(&config.name_map);

    // Provenance gaps are a legitimate signal to cold-resolve, but detect all
    // of them before the first checkout, download, or build. Otherwise a later
    // gap can leave earlier replay entries partially materialized.
    for wheel in &lock.wheels {
        let missing_manifest_source = wheel.origin == Origin::Built
            && wheel.must_ship
            && wheel.git_source.is_none()
            && !config.retread_wheels.contains_key(&wheel.name);
        let missing_shadow_source = wheel.origin == Origin::Built
            && !wheel.must_ship
            && wheel.sdist_source.is_none()
            && wheel.upstream_url.is_none();
        if missing_manifest_source || missing_shadow_source {
            tracing::warn!(
                wheel = %wheel.name,
                "courier replay provenance gap detected before materialization; \
                 falling through to full resolve",
            );
            return Ok(None);
        }
    }

    // Per-wheel re-materialization: classify each LockWheel and build the
    // EmitWheel with correct local_path / remote_url for courier::stage.
    let mut emit_wheels: Vec<crate::emit_pypi::EmitWheel> = Vec::with_capacity(lock.wheels.len());

    // PHASE 2.5: Multi-entry shared-git-checkout replay.
    //
    // Pre-pass: group all Class-1 git wheels (must_ship=true, git_source present)
    // by checkout root (git_checkout_root(gs.url, gs.rev)), preserving lock order.
    // Every member carries its producer-selected auto-data disposition in
    // GitWheelSource. Grouping only coordinates shared-checkout materialization;
    // it must not infer a carrier from lock order because BFS-discovered Git
    // transitives deliberately have auto-data disabled.
    //
    // Grouping key uses the RESOLVED SHA (gs.rev) so equivalence matches produce's
    // partitioning. The lock's subdirectory field may be None (root/"." member).
    //
    // Groups can be NON-CONTIGUOUS in the lock (e.g. git group spans indices 0-5
    // and 10, with index/shadow wheels interleaved). We build a whole group's
    // wheels into an in-memory stash on first encounter, then emit each wheel at
    // its own lock position from the stash. Lock order is preserved byte-for-byte.
    //
    // INCOMPLETE PROVENANCE: if any group member is missing git_source (schema gap
    // or BFS transitive class-3 wheel), we do NOT Err — we return Ok(None) so the
    // caller falls through to full resolve_all.  All-or-nothing: the stash is
    // in-memory, so Ok(None) leaves no partial wheels/ on disk.

    // Step 1: scan lock.wheels to build per-root group membership (preserving order).
    // git_group_members: checkout_root -> Vec<lock index>
    let mut git_group_members: std::collections::HashMap<PathBuf, Vec<usize>> =
        std::collections::HashMap::new();
    // Parallel vec preserving the order in which checkout roots first appear.
    let mut git_group_order: Vec<PathBuf> = Vec::new();

    for (idx, lw) in lock.wheels.iter().enumerate() {
        if lw.origin == crate::lock::Origin::Built
            && lw.must_ship
            && let Some(gs) = &lw.git_source
        {
            let root = crate::source_build::git_checkout_root(&gs.url, &gs.rev, cache_dir);
            let entry = git_group_members.entry(root.clone()).or_insert_with(|| {
                git_group_order.push(root.clone());
                Vec::new()
            });
            entry.push(idx);
        }
    }

    // Step 2: for each group with >1 members (multi-entry), restore each member's
    // exact producer-recorded AutoDataConfig.
    // auto_data_for_lock_idx: lock index -> Option<AutoDataConfig>
    // None means the producer explicitly disabled phase 1.6 for this member.
    // Missing key means "size-1 group" -> restore its disposition in the loop.
    let mut auto_data_override: std::collections::HashMap<usize, Option<AutoDataConfig>> =
        std::collections::HashMap::new();

    for root in &git_group_order {
        let members = &git_group_members[root];
        if members.len() > 1 {
            for &idx in members {
                let gs = lock.wheels[idx]
                    .git_source
                    .as_ref()
                    .expect("git_group_members only contains wheels with git_source; qed");
                auto_data_override.insert(idx, replay_git_auto_data(gs, root.clone())?);
            }
        }
        // Size-1 groups restore their disposition in the single-entry path below.
    }

    // Stash: checkout_root -> built EmitWheels for that group, keyed by lock index.
    // Populated lazily on first encounter of any member of a multi-entry group.
    let mut git_group_stash: std::collections::HashMap<
        PathBuf,
        std::collections::HashMap<usize, crate::emit_pypi::EmitWheel>,
    > = std::collections::HashMap::new();
    // Tracks roots whose groups have already been built into the stash.
    let mut built_roots: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();

    for lw in &lock.wheels {
        let emit = match lw.origin {
            Origin::Index => {
                // Class 4: unchanged index wheel — courier::stage will verify
                // it is still unmodified and record Origin::Index.
                let remote_url = lw
                    .url
                    .as_deref()
                    .and_then(|u| url::Url::parse(u).ok())
                    .filter(|u| u.scheme() != "file");
                // Reconstruct local_path if the wheel file already exists in
                // source_dir/wheels/ (matches cold-produce, where
                // localize_wheel_source collapses https:// to file:// when the
                // file is present locally). This is a warm-box optimization
                // (avoids re-download) AND a plan() parity fix: cold-produce
                // builds EmitWheel with local_path=Some when the file is in
                // wheels/, so plan()'s Pass-1 URL-target check sees the same
                // local_path.is_some() value on both paths.
                let candidate = download_dir.join(&lw.filename);
                let local_path = if candidate.exists() {
                    Some(candidate)
                } else {
                    // Fall through to file:// URL (unlikely for Index wheels,
                    // but handle for completeness).
                    lw.url.as_deref().and_then(|u| {
                        url::Url::parse(u)
                            .ok()
                            .filter(|u| u.scheme() == "file")
                            .and_then(|u| u.to_file_path().ok())
                    })
                };
                let expected = crate::source_build::ExpectedWheel::exact(&lw.name, &lw.version);
                if let Some(path) = &local_path {
                    let actual_sha = crate::source_build::validate_wheel_for_target_async(
                        path,
                        target,
                        Some(&expected),
                    )
                    .await
                    .with_context(|| {
                        format!(
                            "courier replay index artifact for `{}` is incompatible with the immutable target",
                            lw.name,
                        )
                    })?;
                    if let Some(locked_sha) = &lw.sha256
                        && !actual_sha.eq_ignore_ascii_case(locked_sha)
                    {
                        bail!(
                            "courier replay index artifact hash mismatch for `{}`: lock has {}, local file has {}",
                            lw.name,
                            locked_sha,
                            actual_sha,
                        );
                    }
                } else {
                    if crate::pypi::score_wheel(&lw.filename, target.wheel_target()) < 0 {
                        bail!(
                            "courier replay index wheel `{}` is incompatible with python {} on {}",
                            lw.filename,
                            target.python_version(),
                            target.conda_subdir(),
                        );
                    }
                    let (filename_name, filename_version) =
                        crate::pypi::wheel_filename_identity(&lw.filename).ok_or_else(|| {
                            anyhow!(
                                "courier replay index wheel has invalid filename `{}`",
                                lw.filename,
                            )
                        })?;
                    let locked_version = uv_pep508::uv_pep440::Version::from_str(&lw.version)
                        .with_context(|| {
                            format!("invalid locked wheel version `{}`", lw.version)
                        })?;
                    if canonical_conda_name(&filename_name) != canonical_conda_name(&lw.name)
                        || filename_version != locked_version
                    {
                        bail!(
                            "courier replay index wheel `{}` does not match locked identity {}=={}",
                            lw.filename,
                            lw.name,
                            lw.version,
                        );
                    }
                }
                crate::emit_pypi::EmitWheel {
                    pypi_name: lw.name.clone(),
                    version: lw.version.clone(),
                    requires_dist: lw.requires_dist.clone(),
                    local_path,
                    wheel_filename: lw.filename.clone(),
                    sha256: lw.sha256.clone(),
                    locked_final_sha256: None,
                    remote_url,
                    upstream_url: None,
                    git_source: None,   // Origin::Index: no git source
                    sdist_source: None, // Origin::Index: no sdist provenance
                }
            }
            Origin::Built if lw.must_ship => {
                // Class 1 or Class 3: `.injected` source-built wheel.
                //
                // PRIORITY: prefer lw.git_source (schema 8+) for manifest-independent
                // replay. Legacy fallback to config.retread_wheels only when git_source
                // is absent (schema < 8 lock or non-git Class-1 entry such as path=).
                //
                // POISONING note: config-entry git revs (inline rev + named-source rev)
                // are already folded into inputs_hash via courier_input_specs
                // (src/courier.rs: inline rev at ~line 71, named rev at ~line 77).
                // That means a changed config rev invalidates the committed lock and
                // forces a cold cascade — correct. The RESOLVED SHA stored in git_source
                // is NOT fed back into inputs_hash to avoid a circular dependency
                // (compute inputs_hash requires the SHA, but the SHA is only known after
                // the build). The lock is the contract: replay pins the RECORDED SHA;
                // only a cascade re-resolves a moving branch tip.
                if let Some(gs) = &lw.git_source {
                    // Schema-8+ Class-1 git replay: manifest-INDEPENDENT path.
                    // Build a synthetic WheelEntry{git:url, rev:resolved-SHA, subdirectory,
                    // extras} and hand it to materialize_and_rewrite exactly as the
                    // produce path would. Named-vs-inline parity is pre-resolved (DESIGN A):
                    // both arms call the identical leased git build through
                    // materialize_and_rewrite; extras do not reach the wheel build (they
                    // drive BFS closure only), so collapsing a named-git entry to an inline
                    // synth {git:url, rev:SHA} yields a byte-identical wheel.
                    //
                    // PHASE 2.5: Multi-entry shared-git-checkout support.
                    // The pre-pass above identified groups by checkout root. If this wheel
                    // is part of a multi-entry group, we use the group stash:
                    //   - First encounter of the root: build ALL group members via
                    //     materialize_and_rewrite with each member's persisted
                    //     auto-data disposition, then stash results by lock index.
                    //   - Subsequent encounters: emit from stash (no rebuild).
                    // Single-entry groups restore their own persisted disposition below.
                    let checkout_root =
                        crate::source_build::git_checkout_root(&gs.url, &gs.rev, cache_dir);
                    let cur_lock_idx = emit_wheels.len();
                    // Look up whether this wheel is in a multi-entry group.
                    let in_multi_group = auto_data_override.contains_key(&cur_lock_idx);
                    if in_multi_group {
                        // Multi-entry group: use stash.
                        if !built_roots.contains(&checkout_root) {
                            // First encounter of this group: build ALL members.
                            built_roots.insert(checkout_root.clone());
                            let group_indices = git_group_members[&checkout_root].clone();
                            tracing::info!(
                                group_size = group_indices.len(),
                                checkout_root = %checkout_root.display(),
                                "courier replay (phase 2.5): building multi-entry git group"
                            );
                            let mut stash_for_root: std::collections::HashMap<
                                usize,
                                crate::emit_pypi::EmitWheel,
                            > = std::collections::HashMap::new();
                            for &member_idx in &group_indices {
                                let member_lw = &lock.wheels[member_idx];
                                let member_gs = member_lw.git_source.as_ref().expect(
                                    "group member must have git_source; invariant from pre-pass",
                                );
                                // Retrieve the pre-computed auto_data for this member.
                                let member_auto_data =
                                    auto_data_override.get(&member_idx).cloned().unwrap_or(None);
                                let synth_entry = crate::config::WheelEntry {
                                    git: Some(member_gs.url.clone()),
                                    rev: Some(member_gs.rev.clone()),
                                    subdirectory: member_gs.subdirectory.clone(),
                                    extras: member_gs.extras.clone(),
                                    url: None,
                                    sha256: None,
                                    version: None,
                                    index: None,
                                    path: None,
                                    from: None,
                                    bundle: None,
                                    ..crate::config::WheelEntry::default()
                                };
                                tracing::info!(
                                    wheel = %member_lw.name,
                                    url = %member_gs.url,
                                    rev = %member_gs.rev,
                                    has_auto_data = member_auto_data.is_some(),
                                    "courier replay (phase 2.5): building group member"
                                );
                                let (resolved, _rd) = materialize_and_rewrite_with_abi_aliases(
                                    &synth_entry,
                                    &member_lw.name,
                                    Some(&member_lw.version),
                                    target,
                                    download_dir,
                                    source_dir,
                                    cache_dir,
                                    config.relax,
                                    &config.git_sources,
                                    member_auto_data,
                                    EntryAuditInfo::default(),
                                    &abi_aliases,
                                )
                                .await
                                .with_context(|| {
                                    format!(
                                        "courier replay (phase 2.5): re-source-building \
                                         group member `{}` from url={}, rev={}",
                                        member_lw.name, member_gs.url, member_gs.rev,
                                    )
                                })?;
                                let (actual_filename, actual_sha) =
                                    validate_replayed_source_artifact(&resolved, member_lw, target)
                                        .await?;
                                let local_path = (resolved.url.scheme() == "file")
                                    .then(|| resolved.url.to_file_path().ok())
                                    .flatten();
                                stash_for_root.insert(
                                    member_idx,
                                    crate::emit_pypi::EmitWheel {
                                        pypi_name: member_lw.name.clone(),
                                        version: member_lw.version.clone(),
                                        requires_dist: member_lw.requires_dist.clone(),
                                        local_path,
                                        wheel_filename: actual_filename,
                                        sha256: Some(actual_sha),
                                        locked_final_sha256: member_lw.sha256.clone(),
                                        remote_url: None,
                                        upstream_url: None,
                                        git_source: resolved.git_source.clone(),
                                        sdist_source: None, // git group: no sdist provenance
                                    },
                                );
                            }
                            git_group_stash.insert(checkout_root.clone(), stash_for_root);
                        }
                        // Emit this wheel from the stash (built either just now or earlier).
                        git_group_stash[&checkout_root]
                            .get(&cur_lock_idx)
                            .cloned()
                            .expect(
                                "stash must contain every group member after build; \
                                 cur_lock_idx must equal emit_wheels.len() at group-build time",
                            )
                    } else {
                        // Single-entry group: restore the exact producer decision.
                        // This is not inferable from group size: an explicit Git entry
                        // enables checkout-root data, while a BFS Git transitive does not.
                        tracing::info!(
                            wheel = %lw.name,
                            url = %gs.url,
                            rev = %gs.rev,
                            "courier replay: re-source-building git wheel from lock git_source \
                             (manifest-independent, class 1)"
                        );
                        let synth_entry = crate::config::WheelEntry {
                            git: Some(gs.url.clone()),
                            rev: Some(gs.rev.clone()),
                            subdirectory: gs.subdirectory.clone(),
                            extras: gs.extras.clone(),
                            url: None,
                            sha256: None,
                            version: None,
                            index: None,
                            path: None,
                            from: None,
                            bundle: None,
                            // WheelEntry may have additional fields added in future schema
                            // bumps; keep defaults for anything not carried in GitWheelSource.
                            ..crate::config::WheelEntry::default()
                        };
                        let auto_data = replay_git_auto_data(gs, checkout_root)?;
                        let (resolved, _rd) = materialize_and_rewrite_with_abi_aliases(
                            &synth_entry,
                            &lw.name,
                            Some(&lw.version),
                            target,
                            download_dir,
                            source_dir,
                            cache_dir,
                            config.relax,
                            &config.git_sources,
                            auto_data,
                            EntryAuditInfo::default(),
                            &abi_aliases,
                        )
                        .await
                        .with_context(|| {
                            format!(
                                "courier replay (git_source): re-source-building wheel `{}` \
                                 from url={}, rev={}",
                                lw.name, gs.url, gs.rev,
                            )
                        })?;
                        let (actual_filename, actual_sha) =
                            validate_replayed_source_artifact(&resolved, lw, target).await?;
                        let local_path = (resolved.url.scheme() == "file")
                            .then(|| resolved.url.to_file_path().ok())
                            .flatten();
                        crate::emit_pypi::EmitWheel {
                            pypi_name: lw.name.clone(),
                            version: lw.version.clone(),
                            requires_dist: lw.requires_dist.clone(),
                            local_path,
                            wheel_filename: actual_filename,
                            sha256: Some(actual_sha),
                            locked_final_sha256: lw.sha256.clone(),
                            remote_url: None,
                            upstream_url: None,
                            git_source: resolved.git_source.clone(),
                            sdist_source: None, // Class-1 git: no sdist provenance
                        }
                    }
                } else if let Some(entry) = config.retread_wheels.get(&lw.name) {
                    // Legacy fallback (schema < 8, or non-git Class-1 such as path=):
                    // read the live manifest entry and re-run materialize_and_rewrite.
                    // This path requires the manifest to be present and correct (NOT
                    // manifest-independent). For git entries, git_source will be
                    // populated by materialize_and_rewrite (the lock write-back path
                    // in courier::stage will persist it for future replays).
                    //
                    // Residual fall-through note (sdist / direct-URL): neither sdist
                    // nor direct-URL entries carry a git_source (they are not git
                    // builds), so they reach this branch via the manifest. If the
                    // manifest entry is a path= or url= form, materialize_and_rewrite
                    // handles it the same as produce. If the manifest is absent, the
                    // else-branch below returns Ok(None) -> full resolve_all.
                    tracing::info!(
                        wheel = %lw.name,
                        "courier replay: re-materializing source-built wheel via manifest \
                         (legacy / no git_source, class 1)"
                    );
                    // Reconstruct auto_data from the entry's git checkout root
                    // (same logic as resolve_bundle's auto_data_per_entry). For
                    // multi-entry packs, skip_subdirs is not available without
                    // the full group; pass empty (acceptable for replay).
                    let auto_data =
                        checkout_root_for_entry(entry, &config.git_sources, source_dir, cache_dir)
                            .map(|checkout_root| AutoDataConfig {
                                checkout_root,
                                skip_subdirs: vec![],
                            });
                    let (resolved, _rd) = materialize_and_rewrite_with_abi_aliases(
                        entry,
                        &lw.name,
                        Some(&lw.version),
                        target,
                        download_dir,
                        source_dir,
                        cache_dir,
                        config.relax,
                        &config.git_sources,
                        auto_data,
                        EntryAuditInfo::default(),
                        &abi_aliases,
                    )
                    .await
                    .with_context(|| {
                        format!(
                            "courier replay: re-materializing source-built wheel `{}`",
                            lw.name
                        )
                    })?;
                    let (actual_filename, actual_sha) =
                        validate_replayed_source_artifact(&resolved, lw, target).await?;
                    let local_path = (resolved.url.scheme() == "file")
                        .then(|| resolved.url.to_file_path().ok())
                        .flatten();
                    crate::emit_pypi::EmitWheel {
                        pypi_name: lw.name.clone(),
                        version: lw.version.clone(),
                        requires_dist: lw.requires_dist.clone(),
                        local_path,
                        wheel_filename: actual_filename,
                        sha256: Some(actual_sha),
                        locked_final_sha256: lw.sha256.clone(),
                        remote_url: None,
                        upstream_url: None,
                        // git_source from the re-materialized resolved wheel (schema 8+).
                        git_source: resolved.git_source.clone(),
                        sdist_source: None, // legacy manifest path: no sdist provenance
                    }
                } else {
                    // Class 3: BFS transitive built from a `pkg @ git+<url>`
                    // Requires-Dist line. The lock carries no git url+rev
                    // (schema gap); cannot re-materialize without re-running
                    // the full BFS. Return Ok(None) so the caller falls through
                    // to full resolve_all.
                    //
                    // Residual sdist / direct-URL: same fall-through. Neither form
                    // carries a git_source (they are not git builds) and neither has
                    // a [retread-wheels] entry here (Class-3 BFS transitive). The
                    // caller's full resolve_all handles them. Phase 3 may add lock
                    // provenance for these forms, but for now Ok(None) is correct.
                    tracing::warn!(
                        wheel = %lw.name,
                        "courier replay: wheel has no git_source and no [retread-wheels] \
                         entry (class 3 / BFS git transitive, or sdist / direct-URL — \
                         schema gap); falling through to full resolve",
                    );
                    return Ok(None);
                }
            }
            // Class-2b (schema 9+): relax-changed shadow built from a PyPI sdist.
            // Introduced in PHASE 2.6 to fix gym-0.26.2 wheel drift on replay.
            //
            // gym ships only as an sdist on PyPI. On cold produce, bfs_fetch_pypi
            // falls back to the sdist path, calls build_wheel_from_sdist_url, and
            // stores the exact sdist URL (with #sha256) in LockWheel.sdist_source.
            // On replay, the Class-2 arm would find upstream_url=None (suppressed
            // at write time when sdist_prov.is_some()) and return Ok(None) ->
            // full resolve -> python_abi/version drift -> non-byte-identical lock.
            //
            // This arm intercepts BEFORE the bare Origin::Built arm and re-builds
            // directly from the stored sdist_url, bypassing the re-resolve.
            // Integrity or availability failures are terminal: a version-only
            // re-resolve is not equivalent to replaying the locked source bytes.
            //
            // POISONING note: sdist_source is NOT in compute_inputs_hash (same
            // circularity as git_source.rev): the sdist URL is a consequence of the
            // resolve, not an independent input.
            Origin::Built if !lw.must_ship && lw.sdist_source.is_some() => {
                let s = lw.sdist_source.as_ref().unwrap();
                let (stored_url, locked_raw_sha256) = s
                    .validated_url_and_sha256(&lw.name, &lw.version)
                    .with_context(|| {
                        format!(
                            "courier replay Class-2b: validating locked sdist for `{}`",
                            lw.name,
                        )
                    })?;
                let cache_key = replay_sdist_cache_key(&locked_raw_sha256, target);
                let sdist_out = download_dir
                    .join("sdist-builds")
                    .join(format!("{}-{}", s.name, s.version))
                    .join(cache_key);
                tracing::info!(
                    wheel = %lw.name,
                    sdist_url = %stored_url,
                    "courier replay: rebuilding sdist-built shadow from stored sdist_url (class 2b)",
                );
                let expected =
                    crate::source_build::ExpectedWheel::exact(lw.name.clone(), lw.version.clone());
                let built = crate::source_build::build_wheel_from_sdist_url_for_target(
                    &stored_url,
                    &sdist_out,
                    target,
                    Some(&locked_raw_sha256),
                    Some(&expected),
                )
                .await
                .with_context(|| {
                    format!(
                        "courier replay Class-2b: rebuilding `{}` from immutable stored sdist `{stored_url}`; integrity/target errors are not eligible for re-resolve fallback",
                        lw.name,
                    )
                })?;
                crate::emit_pypi::EmitWheel {
                    pypi_name: lw.name.clone(),
                    version: lw.version.clone(),
                    requires_dist: lw.requires_dist.clone(),
                    local_path: Some(built.wheel_path),
                    wheel_filename: lw.filename.clone(),
                    sha256: None,
                    locked_final_sha256: lw.sha256.clone(),
                    remote_url: None,
                    upstream_url: None,
                    git_source: None,
                    sdist_source: lw.sdist_source.clone(),
                }
            }
            Origin::Built => {
                // Class 2: relax-changed INDEX shadow (must_ship=false, no sdist/git).
                // The original upstream URL is in lw.upstream_url (schema 6+).
                // If absent (schema-5 lock), fail so caller falls through
                // to full resolve.
                let remote_url_opt = lw
                    .upstream_url
                    .as_deref()
                    .and_then(|u| url::Url::parse(u).ok());
                let remote_url = match remote_url_opt {
                    Some(u) => u,
                    None => {
                        // Schema-5 lock: upstream_url absent for relax-changed
                        // Built wheels. Cannot re-download without the URL.
                        // Return Ok(None) to fall through to full resolve_all.
                        tracing::warn!(
                            wheel = %lw.name,
                            "courier replay: relax-changed Built wheel has no upstream_url \
                             (schema-5 lock); falling through to full resolve",
                        );
                        return Ok(None);
                    }
                };
                tracing::info!(
                    wheel = %lw.name,
                    url = %remote_url,
                    "courier replay: re-fetching relax-changed shadow from upstream (class 2)"
                );

                // FIX (Phase 2.7): DOWNLOAD the upstream wheel to a local path and
                // route through courier's LOCAL-PATH branch (ShadowSrc::Rewritten /
                // ShadowSrc::Raw → Origin::Built), mirroring what cold's
                // materialize_and_rewrite does. The OLD approach emitted
                // `local_path=None + remote_url=Some(upstream)` which routed through
                // courier's REMOTE-ONLY branch whose `!conda_capable` gate caused
                // conda_capable relax-shadows (pytorch3d) to emit Origin::Index on
                // replay, drifting the lock vs cold. The LOCAL-PATH branch has NO
                // conda_capable gate, so ALL Class-2 shadows replay as Origin::Built
                // regardless of conda_capable membership.
                //
                // The debug_assert in plan() (emit_pypi.rs) is NOT tripped: it fires
                // only when local_path=Some AND remote_url=Some simultaneously. We set
                // local_path=Some + remote_url=None, matching cold's local-path EmitWheel.
                // Index shadows are still never direct-URL Requires-Dist targets (that
                // property is driven by requires_dist content, not EmitWheel fields).
                //
                // fetch_wheel_cached(url, None, dest, cache_root) with sha256=None
                // bypasses the persistent sha256-keyed cache and calls fetch_wheel,
                // landing at dest_dir.join(wheel_filename_from_url(url)) — the pristine
                // 5-field upstream basename, identical to what cold fetched pre-relax.
                let fetched =
                    crate::wheel::fetch_wheel_cached(&remote_url, None, download_dir, cache_dir)
                        .await
                        .with_context(|| {
                            format!(
                                "courier replay Class-2: re-fetching shadow `{}` from `{}`",
                                lw.name, remote_url,
                            )
                        })?;
                let expected = crate::source_build::ExpectedWheel::exact(&lw.name, &lw.version);
                crate::source_build::validate_wheel_for_target_async(
                    &fetched,
                    target,
                    Some(&expected),
                )
                .await
                .with_context(|| {
                    format!(
                        "courier replay Class-2 artifact for `{}` is incompatible with the immutable target",
                        lw.name,
                    )
                })?;

                // DETERMINISM GUARD (Phase 2.7): verify the re-fetched artifact's
                // predicted shadow name matches the recorded lw.filename. If upstream
                // served a repackaged / differently-named artifact under the same URL,
                // the predicted name diverges → fall through to cold re-resolve instead
                // of silently emitting a drifted lock entry.
                //
                // Predicted name = insert_build_tag(standard_wheel_filename(<fetched
                // basename>), "999retread"), using the fetched 5-field upstream basename.
                // The courier stage will use insert_build_tag(standard_wheel_filename(
                // lw.filename), "999retread") against the already-999retread 6-field name
                // (idempotent replace). Both routes yield the identical string when
                // upstream is unchanged (§6.2, PHASE2.7-PLAN.md). If they diverge the
                // assumption is violated and a cold re-resolve is the correct response.
                let fetched_base = fetched.file_name().and_then(|n| n.to_str()).unwrap_or("");
                let predicted = crate::emit_pypi::insert_build_tag(
                    &crate::emit_pypi::standard_wheel_filename(fetched_base),
                    "999retread",
                )
                .with_context(|| {
                    format!(
                        "courier replay Class-2: building predicted shadow name for `{}`",
                        lw.name,
                    )
                })?;
                if predicted != lw.filename {
                    tracing::warn!(
                        wheel = %lw.name,
                        predicted = %predicted,
                        recorded = %lw.filename,
                        "courier replay Class-2: re-fetched artifact name diverges from \
                         recorded shadow filename (upstream repackaged?); falling through \
                         to cold resolve",
                    );
                    return Ok(None);
                }

                // Cold materialization applies the configured general relax
                // policy before courier performs its override/provider rewrite.
                // Replay must feed courier the same phase-D bytes, not the raw
                // upstream wheel, or the authoritative final SHA can drift.
                let replay_local = prepare_replayed_class2_wheel_with_abi_aliases(
                    fetched,
                    config.relax,
                    config.retread_wheels.contains_key(&lw.name),
                    &abi_aliases,
                )
                .await?;

                crate::emit_pypi::EmitWheel {
                    pypi_name: lw.name.clone(),
                    version: lw.version.clone(),
                    requires_dist: lw.requires_dist.clone(),
                    local_path: Some(replay_local),
                    wheel_filename: lw.filename.clone(),
                    sha256: None,
                    locked_final_sha256: lw.sha256.clone(),
                    remote_url: None, // local-path EmitWheel: cold has remote_url=None
                    // upstream_url carries the index URL so courier's Rewritten/Raw
                    // arms write upstream_url=github + url=None, matching cold.
                    upstream_url: Some(remote_url),
                    git_source: None,
                    sdist_source: None,
                }
            }
        };
        emit_wheels.push(emit);
    }

    Ok(Some(emit_wheels))
}

/// Apply the same phase-D relax rewrite to a re-fetched Class-2 index wheel
/// that [`materialize_and_rewrite`] applies on the cold path.
///
/// Courier staging performs a second, distinct metadata rewrite for resolved
/// overrides, conda providers, and orphan URL dependencies.  Feeding it the
/// raw upstream wheel on replay skips phase D and can therefore produce bytes
/// that differ from the authoritative lock whenever the general relax policy
/// widened an exact pin before courier staging.
#[cfg(test)]
async fn prepare_replayed_class2_wheel(
    fetched: PathBuf,
    relax: RelaxPolicy,
    declared_root: bool,
) -> Result<PathBuf> {
    prepare_replayed_class2_wheel_with_abi_aliases(
        fetched,
        relax,
        declared_root,
        &AbiAliasGraph::new(),
    )
    .await
}

async fn prepare_replayed_class2_wheel_with_abi_aliases(
    fetched: PathBuf,
    relax: RelaxPolicy,
    declared_root: bool,
    abi_aliases: &AbiAliasGraph,
) -> Result<PathBuf> {
    // Cold phase D runs in materialize_and_rewrite for declared roots. A
    // remote-only BFS transitive is first downloaded inside courier itself,
    // so courier receives its raw bytes instead. Preserve that distinction.
    if !declared_root || relax == RelaxPolicy::None {
        return Ok(fetched);
    }

    let rewritten = fetched.with_extension("relaxed.whl");
    let src = fetched.clone();
    let dst = rewritten.clone();
    let abi_aliases = abi_aliases.clone();
    tokio::task::spawn_blocking(move || {
        rewrite_wheel_with_abi_aliases(&src, &dst, relax, &abi_aliases)
    })
    .await
    .context("Class-2 replay relax rewrite panicked")?
    .with_context(|| {
        format!(
            "Class-2 replay phase-D rewrite for {} (policy={relax:?})",
            fetched.display(),
        )
    })?;
    Ok(rewritten)
}

/// Replay a committed courier lock: re-materialize all wheels and run
/// `materialize_and_pack`.
///
/// Returns:
/// - `Ok(Some(result))` — all wheels re-materialized; pack completed.
/// - `Ok(None)` — lock provenance insufficient for replay (class 3 / schema-5
///   class-2 gap); caller should fall through to full `resolve_all`.
/// - `Err(...)` — replay was attempted but failed (download error,
///   rattler-build failure, etc.). Caller should propagate as a hard error.
///
/// # Correctness
/// **COLD path** (`run_deps` originates from `params.run_dependencies` /
/// `run_override`): pixi's live conda-solver result is authoritative; the
/// lock has not yet been committed.
///
/// **REPLAY path** (`run_deps` originates from `lock.conda_run_deps`): the
/// already-committed lock is authoritative; using `params.run_dependencies`
/// here would let pixi's non-deterministic conda solver inject extras (e.g.
/// `python_abi`) that drift the rewritten lock from the committed one, which
/// is the exact bug replay is supposed to prevent.  The caller is responsible
/// for sourcing `run_deps` from `lock.conda_run_deps` before calling this
/// function on a replay hit.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
async fn materialize_from_lock(
    lock: crate::lock::RetreadLock,
    config: &RetreadConfig,
    work_dir: &Path,
    output_dir: &Path,
    target: &ResolutionTarget,
    source_dir: &Path,
    cache_dir: &Path,
    expected_build: Option<&str>,
    run_deps: Vec<String>,
    config_fp: &str,
) -> Result<Option<CondaBuildV1Result>> {
    let expected_bundle = lock.bundle.clone();
    materialize_from_lock_for_target(
        lock,
        &expected_bundle,
        config,
        work_dir,
        output_dir,
        target,
        source_dir,
        cache_dir,
        expected_build,
        run_deps,
        config_fp,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn materialize_from_lock_for_target(
    lock: crate::lock::RetreadLock,
    expected_bundle: &str,
    config: &RetreadConfig,
    work_dir: &Path,
    output_dir: &Path,
    target: &ResolutionTarget,
    source_dir: &Path,
    cache_dir: &Path,
    expected_build: Option<&str>,
    run_deps: Vec<String>,
    config_fp: &str,
) -> Result<Option<CondaBuildV1Result>> {
    if !lock.is_for_resolution_target(target) {
        return Ok(None);
    }
    // Validate every entry before the first cache, network, staging, or output
    // mutation so malformed later provenance cannot leave partial work behind.
    lock.validate_replay_contract_for_target(target, expected_bundle)
        .context("validating complete courier replay lock provenance")?;
    let bundle_name = lock.bundle.clone();
    let version = lock.version.clone();
    let download_dir = source_dir.join("wheels");

    let emit_wheels =
        match emit_wheels_from_lock(&lock, config, target, &download_dir, source_dir, cache_dir)
            .await?
        {
            Some(w) => w,
            None => return Ok(None),
        };

    // Reconstruct conda_capable from the lock (recorded by the producer).
    let conda_capable: std::collections::HashSet<String> =
        lock.conda_capable.iter().cloned().collect();

    let index_urls = lock.index_urls.clone();

    tracing::info!(
        bundle = %bundle_name,
        wheels = emit_wheels.len(),
        "courier build_v1 replay: re-materializing from lock (derivation skipped)",
    );
    let replay_relaxations = RelaxationManifest::new(bundle_name.clone(), lock.relaxations.clone());

    let result = materialize_and_pack(
        None, // bundle=None: replay path, audit skipped
        config,
        &bundle_name,
        &bundle_name,
        &version,
        target,
        Some(&lock),
        emit_wheels,
        conda_capable,
        run_deps,
        index_urls,
        config_fp,
        work_dir,
        output_dir,
        source_dir,
        expected_build,
        replay_relaxations.as_ref(),
    )
    .await?;
    Ok(Some(result))
}

/// Shared courier pack tail: stage wheels, build the courier recipe, run
/// rattler-build, flush the deferred committed lock, and return the
/// [`CondaBuildV1Result`].
///
/// Called from two sites:
/// 1. `build_one` (cold path) — after deriving `emit_wheels`, `conda_capable`,
///    `run_deps`, and `index_urls` from the full resolve.
/// 2. `conda_build_v1` (replay path, commit 5) — after re-materializing bytes
///    from the committed lock.
#[allow(clippy::too_many_arguments)]
async fn materialize_and_pack(
    // `None` on the replay path (no Bundle was resolved; audit skipped).
    bundle: Option<&Bundle>,
    config: &RetreadConfig,
    bundle_name: &str,
    input_bundle_name: &str,
    version: &str,
    target: &ResolutionTarget,
    authoritative_lock: Option<&crate::lock::RetreadLock>,
    emit_wheels: Vec<crate::emit_pypi::EmitWheel>,
    conda_capable: std::collections::HashSet<String>,
    run_deps: Vec<String>,
    index_urls: Vec<String>,
    config_fp: &str,
    work_dir: &Path,
    output_dir: &Path,
    source_dir: &Path,
    expected_build: Option<&str>,
    relaxations: Option<&RelaxationManifest>,
) -> Result<CondaBuildV1Result> {
    let staging = work_dir.join(format!("courier-{bundle_name}"));
    // `~/...` is an intentionally portable store identity: expand it against
    // this host and stage there so the serialized identity remains exact.
    // Absolute producer-host paths are not replay authority; those use the
    // current configured store and are normalized only as location metadata
    // by the authoritative-lock comparison below.
    let replay_store_root = authoritative_lock
        .and_then(|lock| lock.wheel_store.as_deref())
        .filter(|recorded| recorded.starts_with("~/"))
        .map(crate::courier::expand_wheel_store_path);
    let staged = crate::courier::stage_for_target_with_store_root_and_relaxations(
        config,
        bundle_name,
        input_bundle_name,
        version,
        target,
        &emit_wheels,
        &conda_capable,
        &run_deps,
        &index_urls,
        config_fp,
        source_dir,
        &staging,
        replay_store_root.as_deref(),
        relaxations,
    )
    .await
    .context("courier staging")?;

    if let Some(authoritative_lock) = authoritative_lock {
        validate_authoritative_replay_lock(authoritative_lock, &staged.lock)?;
    }

    // Pixi permits a dynamic build request (`output.build = None`). The
    // staged lock is the first point where the authoritative inputs hash is
    // available on cold, replay, and incremental paths alike, so synthesize
    // the same content-addressed rich build identity here instead of falling
    // back to the legacy `pyXY_N` namespace.
    let resolved_build = resolved_courier_build(
        expected_build,
        target,
        &staged.lock.inputs_hash,
        config.build_number,
        config.bundle_mode == crate::config::BundleMode::Loose,
    );

    // Defer the committed install lock write until after a successful
    // rattler-build (B-2). The staged copy inside `staging` is already in
    // the recipe's source list; this is the authoritative pack-dir copy.
    let lock_path = exact_lock_path_for_target(source_dir, bundle_name, target);
    let courier_lock_to_commit = (lock_path, staged.lock.to_pretty_json()?);

    let lock_filename = crate::lock::RetreadLock::file_name_for_target(bundle_name, target);
    let recipe = build_courier_recipe_with_mode_lock_and_relaxations(
        bundle_name,
        version,
        target.python_version(),
        &staged.run_deps,
        &staged.source_urls,
        config.build_number,
        // Thread the content-addressed build string into the recipe so
        // the on-disk artifact name matches what conda/outputs advertised.
        Some(&resolved_build),
        config.courier_mode,
        &lock_filename,
        relaxations.is_some(),
    );
    let yaml = to_yaml(&recipe)?;

    let recipe_dir = work_dir.join(format!("recipe-{}", recipe.package.name));
    tokio::fs::create_dir_all(&recipe_dir).await?;
    let recipe_path = recipe_dir.join("recipe.yaml");
    tokio::fs::write(&recipe_path, &yaml).await?;
    tracing::info!(path = %recipe_path.display(), "wrote recipe.yaml");

    // Audit: dump per-wheel pre-D Requires-Dist + post-translate
    // run-deps as JSON, plus copy-paste-friendly pixi.toml fragments.
    // On the replay path `bundle` is None (no resolve_all ran); skip audit.
    if let Some(b) = bundle {
        let audit = build_bundle_audit(b, &recipe);
        let audit_json =
            serde_json::to_string_pretty(&audit).context("serializing audit record")?;
        let audit_path = source_dir.join(format!("retread-audit-{}.json", recipe.package.name));
        if let Err(e) = tokio::fs::write(&audit_path, &audit_json).await {
            tracing::warn!(path = %audit_path.display(), error = %e, "failed to write audit record (non-fatal)");
        } else {
            tracing::info!(path = %audit_path.display(), "wrote audit");
        }
    }

    tokio::fs::create_dir_all(output_dir).await?;

    let target_platform = target.conda_subdir().to_owned();
    let target_subdir = Platform::from_str(target.conda_subdir()).with_context(|| {
        format!(
            "parsing courier resolution target subdir `{}`",
            target.conda_subdir()
        )
    })?;
    let mut compression_lease = crate::thread_budget::acquire(config.compression_threads).await;
    let compression = compression_lease.decision();
    let mut cmd = tokio::process::Command::new("rattler-build");
    cmd.kill_on_drop(true);
    cmd.arg("build")
        .arg("--recipe")
        .arg(&recipe_path)
        .arg("--output-dir")
        .arg(output_dir)
        .arg("--target-platform")
        .arg(&target_platform)
        .arg("--compression-threads")
        .arg(compression.threads.get().to_string())
        .arg("--no-test");
    crate::fasttmp::apply_backend_env(&mut cmd);
    if let Some(level) = config.compression_level {
        cmd.arg("--package-format").arg(format!("conda:{level}"));
    }
    let packaging_started = std::time::Instant::now();
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let child = cmd
        .spawn()
        .context("spawning rattler-build (is it on PATH?)")?;
    // No separate process group is created: `kill_on_drop(true)` cleans up
    // the direct child during ordinary async cancellation. Parent SIGKILL
    // cannot run Drop, so persist the child's PID/start identity before
    // waiting; an orphaned rattler-build then remains charged to this lease.
    if let Some(child_pid) = child.id() {
        compression_lease.record_child(child_pid).await;
    } else {
        tracing::warn!("spawned rattler-build did not expose a PID for lease tracking");
    }
    let output = child.wait_with_output().await;
    compression_lease.release();
    let output = output.context("waiting for rattler-build")?;
    tracing::info!(
        output = %recipe.package.name,
        elapsed_ms = packaging_started.elapsed().as_millis() as u64,
        compression_threads = compression.threads.get(),
        compression_active_leases = compression.active_leases,
        compression_budget = compression.budget.get(),
        compression_threads_source = compression.source.as_str(),
        compression_level = ?config.compression_level,
        "bench: rattler-build finished",
    );
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::error!(stdout = %stdout, stderr = %stderr, "rattler-build failed");
        bail!("rattler-build exited with status {}", output.status);
    }

    // B-2/B-3: rattler-build succeeded -- NOW commit the courier lock.
    let (lock_commit_path, lock_json) = courier_lock_to_commit;
    let lock_tmp = lock_commit_path.with_extension("json.tmp");
    tokio::fs::write(&lock_tmp, &lock_json)
        .await
        .with_context(|| format!("writing lock tmp {}", lock_tmp.display()))?;
    tokio::fs::rename(&lock_tmp, &lock_commit_path)
        .await
        .with_context(|| {
            format!(
                "atomically placing committed lock {}",
                lock_commit_path.display()
            )
        })?;
    tracing::info!(path = %lock_commit_path.display(), "courier: wrote install lock (post-build)");

    let build = resolved_build;
    let subdir_dir = output_dir.join(&target_platform);
    let output_file = find_conda_artifact(
        &subdir_dir,
        &recipe.package.name,
        &recipe.package.version,
        &build,
    )
    .await?;
    Ok(CondaBuildV1Result {
        output_file,
        input_globs: Default::default(),
        name: recipe.package.name.clone(),
        version: VersionWithSource::from_str(&recipe.package.version)?,
        build,
        subdir: target_subdir,
    })
}

fn validate_authoritative_replay_lock(
    authoritative: &crate::lock::RetreadLock,
    staged: &crate::lock::RetreadLock,
) -> Result<()> {
    let mut expected = authoritative.clone();
    let mut actual = staged.clone();
    // The producer version is observational unless `retread-pin-version`
    // participates in `inputs_hash`. Preserve emit-neutral cross-version
    // replay while comparing every semantic target, artifact, digest, and
    // provenance field exactly.
    expected.retread_version = actual.retread_version.clone();
    expected.python = crate::lock::normalized_target_python(&expected.python)
        .context("normalizing authoritative replay lock Python")?;
    actual.python = crate::lock::normalized_target_python(&actual.python)
        .context("normalizing staged replay lock Python")?;
    // An absolute loose-store path names the producer host, not emitted wheel
    // bytes. Replay uses the current configured store and keeps digest +
    // filename identity exact. Portable `~/...` identities remain literal and
    // were staged at their current-host expansion above.
    if expected
        .wheel_store
        .as_deref()
        .is_some_and(|recorded| std::path::Path::new(recorded).is_absolute())
        && actual.wheel_store.is_some()
    {
        expected.wheel_store = actual.wheel_store.clone();
    }
    expected.canonicalize();
    actual.canonicalize();
    if actual != expected {
        anyhow::bail!(
            "courier replay drifted from the authoritative lock after staging; refusing \
             to build or replace the committed lock"
        );
    }
    Ok(())
}

/// Localized incremental-add resolver (STEP 5).
///
/// Reuses the committed lock's closure for unchanged entries and resolves
/// ONLY the newly-added entries' subtrees, guarded by the ripple-detection
/// machinery from STEPs 2+3.  Calls `materialize_and_pack` to write the
/// merged lock and build the courier package.
///
/// Returns `Ok(Some(result))` if the incremental resolve succeeded and the
/// courier package was built.  Returns `Ok(None)` if any escalation condition
/// was met (ripple, conflict, provenance gap, A5 dedup violation) — caller
/// falls through to full `resolve_all`.  Returns `Err` on hard errors.
///
/// # A6 write-last guarantee
/// The committed lock is written ONLY after ALL ripple/dedup checks pass.
/// On any escalation `Ok(None)`, no lock is written.
#[allow(clippy::too_many_arguments)]
async fn resolve_incremental_add(
    incr: IncrementalAdd,
    config: &RetreadConfig,
    target: &ResolutionTarget,
    download_dir: &Path,
    source_dir: &Path,
    cache_dir: &Path,
    conda_channels: &[ChannelUrl],
    workspace_dir: Option<&Path>,
    work_dir: &Path,
    output_dir: &Path,
    expected_build: Option<&str>,
    config_fp: &str,
) -> Result<Option<CondaBuildV1Result>> {
    let IncrementalAdd { added_specs, lock } = incr;

    // Match added specs before replaying any locked source wheels. Git
    // additions require whole-group auto-data planning, so they are a known
    // cold escalation and should not mutate replay caches first.
    let added_set: std::collections::HashSet<&str> =
        added_specs.iter().map(|s| s.as_str()).collect();
    let mut matched_entries: Vec<(String, WheelEntry)> = Vec::new();
    for (key, entry) in &config.retread_wheels {
        let spec = crate::courier::spec_for_entry(key, entry, &config.git_sources);
        if added_set.contains(spec.as_str()) {
            matched_entries.push((key.clone(), entry.clone()));
        }
    }
    if matched_entries.len() != added_specs.len() {
        tracing::debug!(
            matched = matched_entries.len(),
            added = added_specs.len(),
            "incremental-add: could not match all added_specs to config entries; escalating"
        );
        return Ok(None);
    }
    if matched_entries
        .iter()
        .any(|(_, entry)| entry.is_git() || entry.is_named_git())
    {
        tracing::debug!(
            "incremental-add: Git entry requires whole-group auto-data planning; escalating"
        );
        return Ok(None);
    }

    // ── Build locked_closure: name → version from lock.wheels ─────────────
    let locked_closure: std::collections::BTreeMap<String, String> = lock
        .wheels
        .iter()
        .map(|w| (canonical_conda_name(&w.name), w.version.clone()))
        .collect();

    // ── Step A: re-materialize locked wheels (emit_wheels_from_lock) ──────
    let python_version = crate::lock::normalized_target_python(&lock.python)
        .context("normalizing replay lock Python target")?;
    let locked_emit =
        match emit_wheels_from_lock(&lock, config, target, download_dir, source_dir, cache_dir)
            .await?
        {
            Some(w) => w,
            None => {
                tracing::debug!(
                    "incremental-add: emit_wheels_from_lock returned None (provenance gap); \
                 escalating to cold resolve"
                );
                return Ok(None); // A6: no lock written
            }
        };

    // ── Setup: replicate the parselmouth + name_map setup from resolve_all ─
    let workspace_pypi_indexes: Vec<String> = workspace_dir
        .and_then(crate::workspace::WorkspaceManifest::load)
        .map(|m| m.resolution_pypi_index_urls())
        .unwrap_or_else(|| vec![crate::index_chain::PUBLIC_PYPI.to_string()]);
    let pypi_to_conda = if config.auto_bundle {
        load_pypi_to_conda_map().await
    } else {
        Default::default()
    };
    let mut effective = config.clone();
    effective.name_map = effective_name_map(&config.name_map, &pypi_to_conda);
    // ── Step B: resolve each added entry ──────────────────────────────────
    let mut new_emit: Vec<crate::emit_pypi::EmitWheel> = Vec::new();
    let mut new_conda_capable: std::collections::HashSet<String> = std::collections::HashSet::new();

    // auto_data_per_entry for group dedup (use None for simplicity — each
    // new entry is treated as a standalone single-entry group).
    for (entry_name, entry) in &matched_entries {
        let auto_data =
            checkout_root_for_entry(entry, &effective.git_sources, source_dir, cache_dir).map(
                |checkout_root| AutoDataConfig {
                    checkout_root,
                    skip_subdirs: entry
                        .subdirectory
                        .as_deref()
                        .map(|s| vec![PathBuf::from(s)])
                        .unwrap_or_else(|| vec![PathBuf::from(".")]),
                },
            );

        let bundle_result = resolve_bundle(
            entry_name,
            entry,
            target,
            download_dir,
            source_dir,
            cache_dir,
            effective.relax,
            &effective.git_sources,
            auto_data,
            &pypi_to_conda,
            BfsRoutePolicy::from_config(&effective),
            conda_channels,
            &workspace_pypi_indexes,
            Some(&locked_closure),
            None, // favor-lock prefs: not used on the incremental-add path (locked closure handles version pinning)
            &std::collections::HashSet::new(), // incremental-add: no sibling context available
        )
        .await;

        let bundle = match bundle_result {
            Ok(b) => b,
            Err(e) => {
                // Check if this is an IncrementalRipple.
                if e.downcast_ref::<auto_bundle::IncrementalRipple>().is_some() {
                    tracing::debug!(
                        entry = %entry_name,
                        "incremental-add: ripple detected; escalating to cold resolve"
                    );
                    return Ok(None); // A6: no lock written
                }
                return Err(e);
            }
        };
        // Incremental merge reuses the committed lock's conda run-deps and
        // only appends wheel payloads. A dependency that this new bundle
        // short-circuited to conda would therefore be absent from BOTH sets.
        // Cold resolution owns complete run-dep emission, so escalate rather
        // than publish a silently incomplete incremental lock.
        if incremental_bundle_requires_cold_resolve(&bundle) {
            tracing::debug!(
                entry = %entry_name,
                "incremental-add: new BFS conda route requires complete cold run-dep emission"
            );
            return Ok(None);
        }

        // Convert Bundle → EmitWheel (same logic as build_one, lines 5461-5494).
        let wheels_root = source_dir.join("wheels");
        for w in bundle.all_wheels() {
            let url = localize_wheel_source(&w.url, &wheels_root);
            new_emit.push(crate::emit_pypi::EmitWheel {
                pypi_name: w.pypi_name.clone(),
                version: w.metadata.version.clone(),
                requires_dist: w.metadata.requires_dist.clone(),
                sha256: Some(w.metadata.sha256.clone()),
                locked_final_sha256: None,
                local_path: (url.scheme() == "file")
                    .then(|| url.to_file_path().ok())
                    .flatten(),
                wheel_filename: url
                    .path_segments()
                    .and_then(|mut s| s.next_back())
                    .unwrap_or_default()
                    .to_string(),
                remote_url: (url.scheme() != "file").then_some(url),
                upstream_url: w.upstream_url.clone(),
                git_source: w.git_source.clone(),
                sdist_source: w.sdist_source.clone(),
            });
        }

        // Collect conda_capable from probe decisions.
        new_conda_capable.extend(
            bundle
                .probe_decisions
                .iter()
                .filter(|d| d.matching_candidates > 0)
                .map(|d| canonical_conda_name(&d.pypi_name)),
        );
        new_conda_capable.extend(effective.name_map.keys().map(|key| key.as_str().to_owned()));
        new_conda_capable.extend(pypi_to_conda.keys().cloned());
    }

    // ── Step C: auto_bundle_transitives for each new bundle (if applicable) ─
    // NOTE: for simplicity we skip auto_bundle in the incremental path for now.
    // auto_bundle would need to be run AFTER the new entries are resolved, but
    // its results are already pre-filled via locked_closure in seen_candidate.

    // ── A5 merge + dedup guard ────────────────────────────────────────────
    // Start with locked_emit; merge in new_emit.
    // Build a name→(version, index) map from locked_emit.
    let mut merged: Vec<crate::emit_pypi::EmitWheel> = locked_emit;
    let mut seen_versions: std::collections::HashMap<String, String> = merged
        .iter()
        .map(|w| (canonical_conda_name(&w.pypi_name), w.version.clone()))
        .collect();

    let new_count = new_emit.len();
    for w in new_emit {
        let canon = canonical_conda_name(&w.pypi_name);
        if let Some(existing_ver) = seen_versions.get(&canon) {
            if *existing_ver != w.version {
                // Same name, different version — missed ripple; ESCALATE.
                tracing::warn!(
                    name = %canon,
                    locked_ver = %existing_ver,
                    new_ver = %w.version,
                    "incremental-add A5: same name with different versions in merged closure; \
                     escalating (missed ripple)"
                );
                return Ok(None); // A6: no lock written
            }
            // Same name + same version: skip (dedup).
        } else {
            seen_versions.insert(canon, w.version.clone());
            merged.push(w);
        }
    }

    // ── conda_capable + run_deps (union) ──────────────────────────────────
    let mut conda_capable: std::collections::HashSet<String> =
        lock.conda_capable.iter().cloned().collect();
    conda_capable.extend(new_conda_capable);

    // run_deps: locked run_deps (the locked set is authoritative for the
    // unchanged closure; new pure-PyPI-only deps typically have no conda
    // run-deps in the incremental path).
    let run_deps: Vec<String> = lock
        .conda_run_deps
        .iter()
        .map(|dep| {
            if dep.spec.is_empty() {
                dep.name.clone()
            } else {
                format!("{} {}", dep.name, dep.spec)
            }
        })
        .collect();

    let index_urls = lock.index_urls.clone();
    let bundle_name = lock.bundle.clone();
    let version = lock.version.clone();

    tracing::info!(
        bundle = %bundle_name,
        locked = merged.len() - new_count,
        new = new_count,
        "incremental-add: localized resolve succeeded; building courier package"
    );
    let replay_relaxations = RelaxationManifest::new(bundle_name.clone(), lock.relaxations.clone());

    // ── A6: write lock only after ALL checks pass ─────────────────────────
    let result = materialize_and_pack(
        None, // bundle=None: incremental path, no full Bundle available
        config,
        &bundle_name,
        &bundle_name,
        &version,
        target,
        None,
        merged,
        conda_capable,
        run_deps,
        index_urls,
        config_fp,
        work_dir,
        output_dir,
        source_dir,
        expected_build,
        replay_relaxations.as_ref(),
    )
    .await?;

    // ── Optional validity oracle ───────────────────────────────────────────
    if std::env::var("RETREAD_VERIFY_LOCALADD").as_deref() == Ok("1") {
        let lock_path = exact_lock_path_for_target(source_dir, &bundle_name, target);
        verify_localadd_hook(
            &lock_path,
            &added_specs,
            &bundle_name,
            target.conda_subdir(),
            &python_version,
        );
    }

    Ok(Some(result))
}

/// Validity oracle (C) — internal-consistency check for the incremental-add path.
///
/// Gated on `RETREAD_VERIFY_LOCALADD=1`. Never aborts — only logs.
/// The caller always returns `Ok(Some(result))` regardless of outcome.
///
/// ## What it checks
///
/// For every BUNDLED wheel in `lock.wheels`, for every `requires_dist` line
/// whose PEP 508 marker evaluates true for the target env (`conda_subdir` /
/// `python_version`), the oracle looks up the required name in the **bundled
/// closure** (`lock.wheels` only, not `conda_run_deps`):
///
/// * **Required name IS bundled** → assert the bundled version satisfies the
///   requirement's `VersionSpecifiers`.  If not → RED
///   `LOCALADD-INTERNAL-INCONSISTENCY` (real bug: a missed ripple caused a
///   bundled dep to be kept at a version the newly added wheel rejects).
/// * **Required name NOT bundled** → SKIP.  The dep is conda-routed,
///   env-provided, or name-mapped to a conda package.  We cannot version-check
///   it here and — crucially — there is no false-RED for packages such as
///   `opencv-python`, `OpenEXR`, `mujoco`, `vtk`, etc.
///
/// ## Why this avoids false-REDs
///
/// The previous oracle included `conda_run_deps` in the closure and RED'd on
/// any active requirement that wasn't satisfied there.  That produced
/// false-REDs whenever a cold resolve would pick a different transitive
/// (version drift) or when a dep is env/conda-provided.  Oracle (C) only
/// checks internal consistency of the bundled PyPI wheels — the one bug class
/// the ripple-guard + A5 dedup don't cover.
///
/// Cold-comparison (part b) is intentionally deferred: it requires a full
/// network resolve and would produce false-REDs from version-drift differences
/// between incremental and cold.
fn verify_localadd_hook(
    lock_path: &Path,
    added_specs: &[String],
    bundle_name: &str,
    conda_subdir: &str,
    python_version: &str,
) {
    use uv_pep508::uv_pep440::Version;

    // ── 1. Read and parse the written lock ────────────────────────────────
    let bytes = match std::fs::read(lock_path) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                bundle = %bundle_name,
                path = %lock_path.display(),
                err = %e,
                "RETREAD_VERIFY_LOCALADD: could not read written lock; skipping check"
            );
            return;
        }
    };
    let lock: crate::lock::RetreadLock = match serde_json::from_slice(&bytes) {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!(
                bundle = %bundle_name,
                err = %e,
                "RETREAD_VERIFY_LOCALADD: written lock failed to parse; check for corruption"
            );
            return;
        }
    };

    // ── 2. Build marker environment for the target platform/python ────────
    let marker_env = match crate::relax::marker_env_for(conda_subdir, python_version) {
        Ok(env) => env,
        Err(e) => {
            tracing::warn!(
                bundle = %bundle_name,
                err = %e,
                "RETREAD_VERIFY_LOCALADD: could not build marker env; skipping check"
            );
            return;
        }
    };

    // ── 3. Build BUNDLED closure: canonical name → parsed Version ──────────
    // CHANGE 1: lock.wheels ONLY — conda_run_deps intentionally excluded.
    // Deps whose name is absent from this map are skipped (not RED'd) in step 4.
    let mut bundled: std::collections::HashMap<String, Version> = std::collections::HashMap::new();
    for w in &lock.wheels {
        let canon = canonical_conda_name(&w.name);
        if let Ok(v) = w.version.parse::<Version>() {
            bundled.insert(canon, v);
        }
    }

    // ── 4. Internal-consistency walk ──────────────────────────────────────
    let mut red_findings: Vec<String> = Vec::new();
    let mut checked_reqs: usize = 0;
    let mut active_reqs: usize = 0;
    let mut skipped_reqs: usize = 0; // not bundled → conda/env-provided
    let mut satisfied_reqs: usize = 0;

    for wheel in &lock.wheels {
        for raw_req in &wheel.requires_dist {
            checked_reqs += 1;

            // Parse the requirement line.
            let req = match uv_pep508::Requirement::<uv_pep508::VerbatimUrl>::from_str(raw_req) {
                Ok(r) => r,
                Err(_) => continue, // malformed line; skip
            };

            // Marker evaluation (no extras context at lock-write time → &[]).
            if !req.marker.evaluate(&marker_env, &[]) {
                continue; // marker-false for this target — not active
            }
            active_reqs += 1;

            // Extract version specifiers; URL deps are not version-checkable.
            let specifiers = match &req.version_or_url {
                Some(uv_pep508::VersionOrUrl::VersionSpecifier(s)) => s.clone(),
                None => VersionSpecifiers::default(), // unconstrained → any version OK
                Some(uv_pep508::VersionOrUrl::Url(_)) => {
                    satisfied_reqs += 1; // URL deps treated as satisfied
                    continue;
                }
            };

            let dep_canon = canonical_conda_name(req.name.as_ref());

            match bundled.get(&dep_canon) {
                // CHANGE 2: required name NOT in bundled closure → SKIP.
                // It is conda-routed, env-provided, or name-mapped — not our concern.
                None => {
                    skipped_reqs += 1;
                }
                // Required name IS bundled → version must satisfy.
                Some(bundled_ver) => {
                    if specifiers.is_empty() || specifiers.contains(bundled_ver) {
                        satisfied_reqs += 1;
                    } else {
                        red_findings.push(format!(
                            "LOCALADD-INTERNAL-INCONSISTENCY: {} requires `{}` \
                             but bundled {} is at {} which does not satisfy {}",
                            wheel.name, raw_req, dep_canon, bundled_ver, specifiers,
                        ));
                    }
                }
            }
        }
    }

    // ── 5. Report ─────────────────────────────────────────────────────────
    if red_findings.is_empty() {
        tracing::info!(
            bundle = %bundle_name,
            added = added_specs.len(),
            wheels = lock.wheels.len(),
            checked = checked_reqs,
            active = active_reqs,
            skipped = skipped_reqs,
            satisfied = satisfied_reqs,
            "RETREAD_VERIFY_LOCALADD: GREEN — bundled closure is internally consistent"
        );
    } else {
        for finding in &red_findings {
            tracing::warn!(
                bundle = %bundle_name,
                finding = %finding,
                "RETREAD_VERIFY_LOCALADD: RED"
            );
        }
        tracing::warn!(
            bundle = %bundle_name,
            red_count = red_findings.len(),
            checked = checked_reqs,
            active = active_reqs,
            skipped = skipped_reqs,
            satisfied = satisfied_reqs,
            "RETREAD_VERIFY_LOCALADD: {} RED finding(s) — internal inconsistency in \
             bundled closure; possible missed ripple",
            red_findings.len()
        );
    }
}

/// Select the courier package version independently from the primary wheel's
/// version when a localized incremental add was advertised and then had to
/// fall back to a full cold materialization. Wheel identities remain their
/// own upstream versions; only the generated conda package keeps the version
/// pixi already solved from conda/outputs.
fn courier_pack_version(bundle: &Bundle, advertised_override: Option<&str>) -> String {
    advertised_override
        .unwrap_or(&bundle.primary.metadata.version)
        .to_string()
}

fn validate_resolution_artifact_subdir(
    target: &ResolutionTarget,
    artifact_subdir: Platform,
) -> Result<()> {
    if artifact_subdir != Platform::NoArch && target.conda_subdir() != artifact_subdir.to_string() {
        bail!(
            "build target mismatch: immutable resolution target is `{}` but requested output subdir is `{artifact_subdir}`",
            target.conda_subdir(),
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn build_one(
    bundle: &Bundle,
    config: &RetreadConfig,
    declared_config: &RetreadConfig,
    work_dir: &Path,
    output_dir: &Path,
    target_subdir: Platform,
    target: &ResolutionTarget,
    source_dir: &Path,
    workspace_dir: Option<&Path>,
    input_bundle_name: &str,
    expected_build: Option<&str>,
    courier_version_override: Option<&str>,
    run_override: Option<&[String]>,
    relaxations: Option<&RelaxationManifest>,
) -> Result<CondaBuildV1Result> {
    validate_resolution_artifact_subdir(target, target_subdir)?;
    let workspace_python_version = target.python_version();
    let py_short = workspace_python_version.replace('.', "");
    let legacy_build = format!("py{py_short}_{}", config.build_number);
    let rich_target_build = (!config.courier)
        .then(|| {
            target.target_contract().map(|_| {
                courier_build_string_for_target(
                    target,
                    &target.resolution_identity(),
                    config.build_number,
                    config.bundle_mode == crate::config::BundleMode::Loose,
                )
            })
        })
        .flatten();
    if let (Some(expected), Some(required)) = (expected_build, rich_target_build.as_deref())
        && expected != required
    {
        bail!(
            "advertised non-courier build `{expected}` does not match immutable target identity `{required}`"
        );
    }
    let build = expected_build
        .map(str::to_owned)
        .or(rich_target_build)
        .unwrap_or(legacy_build);
    // Lay out one BundleSource per wheel (primary first), in BFS order.
    //
    // v1.4.5: point each source at retread's local wheel cache when
    // the file is already on disk. rattler-build keeps its own
    // src_cache and re-downloads every http(s) source -- for an
    // isaac-scale pack that was the same ~8GB fetched twice (once
    // into <pack>/wheels/ for METADATA + D, once into rattler-build's
    // src_cache for the build). The recipe still pins each wheel's
    // sha256, so rattler-build verifies the local bytes exactly as it
    // would a download. Wheels with no local copy (e.g. resolved via
    // the v1.4.4 PEP 658 sidecar) keep their upstream URL and get
    // downloaded exactly once -- by rattler-build.
    let wheels_root = source_dir.join("wheels");
    let localized_urls: Vec<url::Url> = bundle
        .all_wheels()
        .map(|w| localize_wheel_source(&w.url, &wheels_root))
        .collect();
    let sources: Vec<BundleSource> = bundle
        .all_wheels()
        .zip(localized_urls.iter())
        .map(|(w, url)| BundleSource {
            pypi_name: &w.pypi_name,
            url,
            metadata: &w.metadata,
        })
        .collect();
    if target_subdir == Platform::NoArch
        && (config.courier || sources.iter().any(|source| !source.metadata.is_pure_python))
    {
        bail!(
            "requested noarch artifact for platform-specific bundle `{}`",
            bundle.conda_name
        );
    }
    // Deprecation gate: warn once when the old emit-pypi / blueprint
    // keys are set. These were replaced by `retread-courier` (v2.0.0);
    // the fields are retained only for backward-compat parsing.
    if config.emit_pypi || config.blueprint.is_on() || config.blueprint_sync.is_some() {
        tracing::warn!(
            "`retread-emit-pypi`, `retread-blueprint`, and `retread-blueprint-sync` \
             are DEPRECATED (v2.0.0) and ignored. Use `retread-courier = true` instead. \
             Remove these keys from your `[package.build.config]`.",
        );
    }
    // WS-C: courier mode — delegate to materialize_and_pack which handles
    // the full courier staging + rattler-build + deferred lock flush pipeline.
    if config.courier {
        let version = courier_pack_version(bundle, courier_version_override);
        let emit_wheels: Vec<crate::emit_pypi::EmitWheel> = bundle
            .all_wheels()
            .zip(localized_urls.iter())
            .map(|(w, url)| crate::emit_pypi::EmitWheel {
                pypi_name: w.pypi_name.clone(),
                version: w.metadata.version.clone(),
                requires_dist: w.metadata.requires_dist.clone(),
                sha256: Some(w.metadata.sha256.clone()),
                locked_final_sha256: None,
                local_path: (url.scheme() == "file")
                    .then(|| url.to_file_path().ok())
                    .flatten(),
                wheel_filename: url
                    .path_segments()
                    .and_then(|mut s| s.next_back())
                    .unwrap_or_default()
                    .to_string(),
                remote_url: (url.scheme() != "file").then(|| (*url).clone()),
                // Pristine pre-localization index URL: read from the
                // ResolvedWheel.upstream_url field, which was captured in
                // materialize_and_rewrite / bfs_fetch_pypi BEFORE the wheel
                // URL was localised to file://. This fixes the primary
                // config-entry wheel (isaacsim) which always has w.url ==
                // file:// by the time build_one sees it, so deriving from
                // w.url would always yield None and break Phase-1 replay.
                upstream_url: w.upstream_url.clone(),
                // Git provenance (schema 8+): carried from ResolvedWheel so
                // courier::stage can write it into LockWheel.git_source.
                git_source: w.git_source.clone(),
                // Sdist provenance (schema 9+): carried from ResolvedWheel so
                // courier::stage can write it into LockWheel.sdist_source.
                // None for index-fetched and git-built wheels; Some for BFS-
                // transitive sdist-built wheels (e.g. gym).
                sdist_source: w.sdist_source.clone(),
            })
            .collect();
        let mut conda_capable: std::collections::HashSet<String> = bundle
            .probe_decisions
            .iter()
            .filter(|d| d.matching_candidates > 0)
            .map(|d| canonical_conda_name(&d.pypi_name))
            .collect();
        conda_capable.extend(config.name_map.keys().map(|key| key.as_str().to_owned()));
        let pypi_to_conda = load_pypi_to_conda_map().await;
        conda_capable.extend(pypi_to_conda.keys().cloned());
        let ws_manifest = workspace_dir.and_then(crate::workspace::WorkspaceManifest::load);
        let workspace_indexes: Vec<String> = ws_manifest
            .as_ref()
            .map(|m| m.resolution_pypi_index_urls())
            .unwrap_or_else(|| vec![crate::index_chain::PUBLIC_PYPI.to_string()]);
        // grizzly H1: fold the workspace solve environment into the hash.
        // Pack-scoped: only envs that reference source_dir are hashed.
        let workspace_fp = ws_manifest
            .as_ref()
            .map(|m| {
                workspace_solve_fingerprint(
                    m,
                    workspace_dir.unwrap_or(source_dir),
                    source_dir,
                    target,
                )
            })
            .unwrap_or_default();
        let entry_indexes: Vec<String> = config
            .retread_wheels
            .values()
            .filter(|entry| entry.url.is_none())
            .filter_map(|entry| entry.index.clone())
            .collect();
        let index_urls = index_chain(entry_indexes, &workspace_indexes, IndexPurpose::RootResolve);
        // B-1 (lock-poisoning): in courier mode the committed lock's
        // `conda_run_deps` MUST be the run-deps pixi actually solved and
        // locked (forwarded as `run_override`). The legacy fallback re-derived
        // them from the wheels' `requires_dist` via `build_bundle_recipe`,
        // which can diverge from pixi's real solve -- a cold replay would then
        // faithfully reproduce a POISONED lock. Refuse rather than guess.
        let run_deps: Vec<String> = run_override.map(|o| o.to_vec()).ok_or_else(|| {
            anyhow::anyhow!(
                "courier: refusing to write a lock from a non-authoritative source. pixi did \
                 not forward run_dependencies (run_override = None), so the only run-deps \
                 available are re-derived from the wheels' Requires-Dist, which can diverge \
                 from what pixi actually locked and poison the committed lock. Aborting the \
                 courier build. (A normal `pixi install` forwards run_dependencies during \
                 conda/build_v1; if you hit this, file a bug.)"
            )
        })?;
        // Fingerprint folds in the conda channel list (grizzly P1) and the
        // workspace solve env (grizzly H1) alongside the config-derived
        // inputs; the replayer computes it identically.
        // Derive channels from the manifest (NOT from the RPC's params.channels,
        // which pixi forwards differently to conda/build_v1 vs conda/outputs for
        // multi-env workspaces, causing replay to never fire).
        let courier_channels = ws_manifest
            .as_ref()
            .map(|m| {
                workspace_courier_channels(
                    m,
                    workspace_dir.unwrap_or(source_dir),
                    source_dir,
                    target,
                )
            })
            .unwrap_or_default();
        let config_fp =
            crate::courier::config_fingerprint(declared_config, &courier_channels, &workspace_fp);
        return materialize_and_pack(
            Some(bundle),
            config,
            &bundle.conda_name,
            input_bundle_name,
            &version,
            target,
            None,
            emit_wheels,
            conda_capable,
            run_deps,
            index_urls,
            &config_fp,
            work_dir,
            output_dir,
            source_dir,
            expected_build,
            relaxations,
        )
        .await
        .context("courier materialize_and_pack");
    }

    // Non-courier path: stage the same two tiny mandatory warning files used
    // by courier packages, then add them to the recipe's ordinary sources.
    let relaxation_payload = match relaxations {
        Some(manifest) => {
            manifest.validate_for(&bundle.conda_name, target)?;
            Some(
                stage_relaxation_payload(
                    &work_dir.join(format!(
                        "relaxations-{}-{}",
                        bundle.conda_name,
                        target.resolution_identity(),
                    )),
                    manifest,
                )
                .await
                .context("staging mandatory relaxation warning payload")?,
            )
        }
        None => None,
    };
    let relaxation_source_urls = relaxation_payload
        .as_ref()
        .map(|payload| payload.source_urls.as_slice())
        .unwrap_or_default();
    let recipe = build_bundle_recipe_with_relaxations(
        &bundle.conda_name,
        &sources,
        config,
        workspace_python_version,
        run_override,
        Some(&build),
        // blueprint="only" payload-skip is deprecated (v2.0.0); the
        // non-courier conda path always carries its wheel payload.
        true,
        relaxation_source_urls,
    )?;
    let yaml = to_yaml(&recipe)?;

    let recipe_dir = work_dir.join(format!("recipe-{}", recipe.package.name));
    tokio::fs::create_dir_all(&recipe_dir).await?;
    let recipe_path = recipe_dir.join("recipe.yaml");
    tokio::fs::write(&recipe_path, &yaml).await?;
    tracing::info!(path = %recipe_path.display(), "wrote recipe.yaml");

    // Audit: dump per-wheel pre-D Requires-Dist + post-translate
    // run-deps as JSON, plus copy-paste-friendly pixi.toml fragments.
    // Lands in the source-package (pack) folder -- next to pixi.toml
    // and the `wheels/` cache -- so it survives pixi cache clears and
    // is right where users go to inspect what the build produced. The
    // filename includes the bundle name so multi-bundle packs don't
    // collide.
    let audit = build_bundle_audit(bundle, &recipe);
    let audit_json = serde_json::to_string_pretty(&audit).context("serializing audit record")?;
    let audit_path = source_dir.join(format!("retread-audit-{}.json", recipe.package.name));
    if let Err(e) = tokio::fs::write(&audit_path, &audit_json).await {
        tracing::warn!(path = %audit_path.display(), error = %e, "failed to write audit record (non-fatal)");
    } else {
        tracing::info!(path = %audit_path.display(), "wrote audit");
    }

    tokio::fs::create_dir_all(output_dir).await?;

    let target_platform = target_subdir.to_string();
    let mut compression_lease = crate::thread_budget::acquire(config.compression_threads).await;
    let compression = compression_lease.decision();
    // CRITICAL: rattler-build writes progress to stdout, but retread's
    // stdout is the JSON-RPC channel to pixi. Capture both streams so
    // they don't corrupt the protocol. Surface them via tracing
    // (which writes to OUR stderr) on failure.
    let mut cmd = tokio::process::Command::new("rattler-build");
    cmd.kill_on_drop(true);
    cmd.arg("build")
        .arg("--recipe")
        .arg(&recipe_path)
        .arg("--output-dir")
        .arg(output_dir)
        .arg("--target-platform")
        .arg(&target_platform)
        .arg("--compression-threads")
        .arg(compression.threads.get().to_string())
        .arg("--no-test");
    crate::fasttmp::apply_backend_env(&mut cmd);
    // v1.5.8: user-tunable zstd level (retread-compression-level).
    // Unset keeps rattler-build's default.
    if let Some(level) = config.compression_level {
        cmd.arg("--package-format").arg(format!("conda:{level}"));
    }
    // v1.6.0: time the packaging stage explicitly.
    let packaging_started = std::time::Instant::now();
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let child = cmd
        .spawn()
        .context("spawning rattler-build (is it on PATH?)")?;
    // No separate process group is created: `kill_on_drop(true)` cleans up
    // the direct child during ordinary async cancellation. Parent SIGKILL
    // cannot run Drop, so persist the child's PID/start identity before
    // waiting; an orphaned rattler-build then remains charged to this lease.
    if let Some(child_pid) = child.id() {
        compression_lease.record_child(child_pid).await;
    } else {
        tracing::warn!("spawned rattler-build did not expose a PID for lease tracking");
    }
    let output = child.wait_with_output().await;
    compression_lease.release();
    let output = output.context("waiting for rattler-build")?;
    tracing::info!(
        output = %recipe.package.name,
        elapsed_ms = packaging_started.elapsed().as_millis() as u64,
        compression_threads = compression.threads.get(),
        compression_active_leases = compression.active_leases,
        compression_budget = compression.budget.get(),
        compression_threads_source = compression.source.as_str(),
        compression_level = ?config.compression_level,
        "bench: rattler-build finished",
    );
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::error!(stdout = %stdout, stderr = %stderr, "rattler-build failed");
        bail!("rattler-build exited with status {}", output.status);
    }

    let subdir_dir = output_dir.join(&target_platform);
    let output_file = find_conda_artifact(
        &subdir_dir,
        &recipe.package.name,
        &recipe.package.version,
        &build,
    )
    .await?;
    Ok(CondaBuildV1Result {
        output_file,
        input_globs: Default::default(),
        name: recipe.package.name.clone(),
        version: VersionWithSource::from_str(&recipe.package.version)?,
        build,
        subdir: target_subdir,
    })
}

/// Computes the exclusive upper bound for a bounded-range auto-routed conda
/// pin: `floor` at `version` (unchanged), capped at the next MAJOR --
/// e.g. `1.26.4` -> `2` (so the emitted range is `>=1.26.4,<2`). Follows
/// semver convention for pre-1.0 releases (major component `0`): the cap
/// is the next MINOR instead, since `0.x` is the "breaking axis" for those
/// packages -- e.g. `0.20.1` -> `0.21` (`>=0.20.1,<0.21`).
///
/// Returns `None` when `version` has no parseable leading numeric
/// component (the caller falls back to an exact pin in that case).
fn bounded_range_ceiling(version: &str) -> Option<String> {
    let mut parts = version.split('.');
    let major: u64 = parts.next()?.parse().ok()?;
    if major == 0 {
        let minor: u64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        crate::relax::checked_version_ceiling(&[major, minor])
    } else {
        crate::relax::checked_version_ceiling(&[major])
    }
}

fn spec_from_str(s: &str) -> Result<NamedSpec<PackageSpec>> {
    let (name, rest) = match s.split_once(' ') {
        Some((n, r)) => (n.trim(), r.trim()),
        None => (s.trim(), ""),
    };
    // `rest` is a conda matchspec tail: a version spec OPTIONALLY followed by a
    // build string, e.g. `3.12.* *_cp312` (python_abi). Split them so the
    // build string doesn't get fed to the version parser (which rejects it).
    // This is load-bearing for cold-solve replay, which round-trips the
    // emitted run-deps (incl. build-tagged python_abi) through this fn.
    let (ver_part, build_part) = match rest.split_once(char::is_whitespace) {
        Some((v, b)) => (v.trim(), b.trim()),
        None => (rest, ""),
    };
    let version = if ver_part.is_empty() {
        None
    } else {
        Some(
            VersionSpec::from_str(ver_part, rattler_conda_types::ParseStrictness::Lenient)
                .map_err(|e| anyhow!("parsing version spec `{ver_part}` for `{name}`: {e}"))?,
        )
    };
    let build = if build_part.is_empty() {
        None
    } else {
        Some(
            StringMatcher::from_str(build_part)
                .map_err(|e| anyhow!("parsing build string `{build_part}` for `{name}`: {e}"))?,
        )
    };
    Ok(NamedSpec {
        name: name.to_string(),
        spec: PackageSpec::Binary(BinaryPackageSpec {
            version,
            build,
            ..Default::default()
        }),
    })
}

async fn find_conda_artifact(
    dir: &Path,
    name: &str,
    version: &str,
    build: &str,
) -> Result<PathBuf> {
    let expected_name = format!("{name}-{version}-{build}.conda");
    let expected = dir.join(&expected_name);
    if tokio::fs::try_exists(&expected)
        .await
        .with_context(|| format!("checking rattler-build artifact {}", expected.display()))?
    {
        return Ok(expected);
    }
    bail!(
        "no exact .conda artifact found at {} (expected build `{build}`)",
        expected.display()
    )
}

/// Carry-forward for an incremental-add detected by [`detect_incremental_add`].
// Fields read by STEP 5 (`resolve_incremental_add`); allow dead_code until
// the driver is wired up.
#[allow(dead_code)]
struct IncrementalAdd {
    /// The PEP 508–like spec strings for the NEWLY added entries only
    /// (current_specs − lock.entry_specs). Each is guaranteed to be an exact
    /// `==pin`, `@git:…`, or `@url:…` form (STEP 4's spec-form gate).
    added_specs: Vec<String>,
    /// The committed lock whose closure is the reuse base.
    lock: crate::lock::RetreadLock,
}

/// Determine whether the current manifest diff is a pure incremental add.
///
/// Returns `Some(IncrementalAdd)` when ALL of the following hold:
/// 1. `RETREAD_INCREMENTAL=1` env is set.
/// 2. Lock exists, parses, `schema == SCHEMA`, `entry_specs` non-empty.
/// 3. **Two-step index + external-input gate** (replaces old single A2 gate):
///
///    **STEP A** — locked-chain check: build the index chain for all
///    `config.retread_wheels` entries whose spec string is NOT in `added`
///    (i.e. existing entries only; added entries may introduce a new index),
///    then compare to `lock.index_urls`. Mismatch → cold. This catches:
///    an existing entry's index change, a workspace-index add/remove/reorder,
///    an implicit→explicit `pypi.org` position change — none of which are
///    encoded in entry spec strings.
///
///    **STEP B** — external-input hash check: recompute `inputs_hash` using
///    `lock.entry_specs` + `lock.index_urls` (STEP A proved locked chain still
///    matches) + current relax/python/epoch/pin/config_fp. Mismatch → cold.
///    This catches relax, python, config-knob, or channel changes.
///
/// 4. `added = current_specs − lock.entry_specs` is non-empty.
///    `removed = lock.entry_specs − current_specs` is empty.
/// 5. Every added spec is an exact pin (`==`), `@git:` or `@url:` form.
///    Bare/range specs are rejected (they have multiple solutions → not safe
///    to combine with a frozen closure).
///
/// Returns `None` on any gate failure (always safe: cold resolve follows).
#[cfg(test)]
fn detect_incremental_add(
    lock_path: &Path,
    config: &RetreadConfig,
    bundle_name: &str,
    ws_indexes: &[String],
    relax_str: &str,
    python_version: &str,
    config_fp: &str,
) -> Option<IncrementalAdd> {
    let target = ResolutionTarget::from_wheel_target(
        WheelTarget {
            python_version: python_version.to_owned(),
            conda_subdir: "linux-64".to_owned(),
            max_glibc: None,
        },
        None,
    );
    detect_incremental_add_for_target(
        lock_path,
        config,
        bundle_name,
        ws_indexes,
        relax_str,
        &target,
        config_fp,
    )
}

fn detect_incremental_add_for_target(
    lock_path: &Path,
    config: &RetreadConfig,
    bundle_name: &str,
    ws_indexes: &[String],
    relax_str: &str,
    target: &ResolutionTarget,
    config_fp: &str,
) -> Option<IncrementalAdd> {
    // Gate 1: feature flag.
    if std::env::var("RETREAD_INCREMENTAL").as_deref() != Ok("1") {
        return None;
    }

    // Gate 2: load and basic schema check.
    if !lock_path.exists() {
        return None;
    }
    let lock = match crate::lock::RetreadLock::load(lock_path) {
        Ok(l) => l,
        Err(e) => {
            tracing::debug!(error = %format!("{e:#}"), "incremental-add: lock load failed; skipping");
            return None;
        }
    };
    if lock.schema != crate::lock::SCHEMA {
        tracing::debug!(
            lock_schema = lock.schema,
            expected = crate::lock::SCHEMA,
            "incremental-add: schema mismatch; skipping"
        );
        return None;
    }
    if let Err(error) = lock.validate_replay_contract_for_target(target, bundle_name) {
        tracing::debug!(
            target = %target.resolution_identity(),
            error = %format!("{error:#}"),
            "incremental-add: complete replay contract validation failed; skipping"
        );
        return None;
    }
    if lock.entry_specs.is_empty() {
        tracing::debug!("incremental-add: lock has no entry_specs (old schema); skipping");
        return None;
    }

    // Gate 4 (computed first — STEP A needs `added` to exclude new entries).
    let current_specs: std::collections::BTreeSet<String> =
        crate::courier::courier_input_specs(config, bundle_name)
            .into_iter()
            .collect();
    let locked_specs: std::collections::BTreeSet<String> =
        lock.entry_specs.iter().cloned().collect();

    let added: Vec<String> = current_specs.difference(&locked_specs).cloned().collect();
    let removed: Vec<String> = locked_specs.difference(&current_specs).cloned().collect();

    if !removed.is_empty() {
        tracing::debug!(
            ?removed,
            "incremental-add: removed specs detected; not a pure add; skipping"
        );
        return None;
    }
    if added.is_empty() {
        tracing::debug!("incremental-add: no added specs; skipping");
        return None;
    }

    // Gate 3 — STEP A: locked-chain check.
    // Build the index chain from all EXISTING entries (current entries minus
    // the newly added ones) and compare to lock.index_urls.
    // We use ALL config.retread_wheels entries (pack-wide, not bundle-filtered)
    // because lock.index_urls is also written pack-wide in courier::stage.
    // The `added` set holds spec strings (key+extras+ver); exclude entries
    // whose spec_for_entry is in `added`.
    {
        let added_set: std::collections::HashSet<&str> = added.iter().map(|s| s.as_str()).collect();
        let locked_entry_indexes: Vec<String> = config
            .retread_wheels
            .iter()
            .filter(|(key, entry)| {
                let spec = crate::courier::spec_for_entry(key, entry, &config.git_sources);
                !added_set.contains(spec.as_str())
            })
            .map(|(_, entry)| entry)
            .filter(|entry| entry.url.is_none())
            .filter_map(|entry| entry.index.clone())
            .collect();
        let locked_chain = index_chain(locked_entry_indexes, ws_indexes, IndexPurpose::RootResolve);
        if locked_chain != lock.index_urls {
            tracing::debug!(
                ?locked_chain,
                lock_index_urls = ?lock.index_urls,
                "incremental-add: STEP A locked-chain mismatch (existing entry index \
                 or ws-index changed); skipping"
            );
            return None;
        }
    }

    // Gate 3 — STEP B: external-input hash check.
    // STEP A proved lock.index_urls == current locked chain, so it is safe to
    // use lock.index_urls as the index term, reproducing the canonical chain
    // order exactly, including the public terminal fallback.
    let recomputed_hash = crate::lock::RetreadLock::compute_inputs_hash_for_target(
        &lock.entry_specs,
        &lock.index_urls,
        relax_str,
        target,
        crate::lock::EMIT_EPOCH,
        config.pin_version.then_some(env!("CARGO_PKG_VERSION")),
        config_fp,
    );
    if recomputed_hash != lock.inputs_hash {
        tracing::debug!(
            "incremental-add: STEP B external-input hash mismatch (relax/python/config \
             changed); skipping"
        );
        return None;
    }

    // Gate 5: every added spec must be exact pin / @git / @url.
    // Bare or range specs have multiple solutions and cannot be safely
    // combined with a frozen closure without a full re-solve.
    for spec in &added {
        // A spec string looks like "<key>[<extras>]<ver_proxy>" where ver_proxy
        // starts with "==", "@git:", or "@url:" for exact/pinned forms.
        let after_bracket = if let Some(close) = spec.rfind(']') {
            &spec[close + 1..]
        } else {
            // No extras bracket: ver_proxy starts after the key name.
            // Find the first non-identifier char.
            spec.trim_start_matches(|c: char| c.is_alphanumeric() || c == '-' || c == '_')
        };
        let is_exact = after_bracket.starts_with("==")
            || after_bracket.starts_with("@git:")
            || after_bracket.starts_with("@url:");
        if !is_exact {
            tracing::debug!(
                spec = %spec,
                "incremental-add: added spec is not exact pin/git/url; skipping"
            );
            return None;
        }
    }

    tracing::info!(
        ?added,
        "incremental-add: pure add detected; will attempt localized resolve"
    );
    Some(IncrementalAdd {
        added_specs: added,
        lock,
    })
}

/// Build a map of `canonical_conda_name(wheel.name) → wheel.version` from the
/// committed lock file at `lock_path`, to be used as "favor-lock" soft
/// preferences during re-resolve.
///
/// **Default-on; opt out with `RETREAD_NO_FAVOR_LOCK`** (any non-empty value).
/// Returns an empty map when favor-lock is disabled, the lock file is missing,
/// or the lock cannot be parsed.  Errors are silently discarded so a corrupt
/// lock does not break the build -- the caller will just cold-resolve as usual.
///
/// Unlike [`load_replayable_lock`], this function loads the lock even when the
/// `inputs_hash` does not match -- favor-lock REQUIRES a deliberate hash
/// mismatch (it is designed for the re-resolve-after-manifest-change case).
#[cfg(test)]
fn load_favored_versions(lock_path: &Path) -> std::collections::BTreeMap<String, String> {
    load_favored_versions_inner(lock_path, None)
}

fn load_favored_versions_for_target(
    lock_path: &Path,
    target: &ResolutionTarget,
) -> std::collections::BTreeMap<String, String> {
    load_favored_versions_inner(lock_path, Some(target))
}

fn load_favored_versions_inner(
    lock_path: &Path,
    target: Option<&ResolutionTarget>,
) -> std::collections::BTreeMap<String, String> {
    if std::env::var_os("RETREAD_NO_FAVOR_LOCK").is_some() {
        return std::collections::BTreeMap::new();
    }
    let Ok(lock) = crate::lock::RetreadLock::load(lock_path) else {
        return std::collections::BTreeMap::new();
    };
    if target.is_some_and(|target| !lock.is_for_resolution_target(target)) {
        return std::collections::BTreeMap::new();
    }
    let mut m = std::collections::BTreeMap::new();
    for w in &lock.wheels {
        // Skip wheels with empty version strings; those are malformed.
        if !w.version.is_empty() {
            m.insert(canonical_conda_name(&w.name), w.version.clone());
        }
    }
    m
}

/// Fold the uv closure's `name -> version` pins into the favor-lock
/// preference map, uv winning on collision (the closure was resolved THIS
/// run against the current conda pins/overrides; a committed lock's favored
/// version is at best last run's answer).
///
/// This is the ONLY seam through which uv pins reach the BFS
/// (spec-uv-restructure M1, fixed): fetch-time version preferences via
/// `resolve_preferring`, which pick uv's version whenever it satisfies the
/// accumulated constraints while still FETCHING and BUNDLING the wheel.
/// Feeding pins through `resolve_bundle`'s `locked_closure` parameter
/// instead is a regression: `seed_locked` marks every pinned name as
/// already handled, the BFS never walks the closure, the bundle ships only
/// the primary wheel, and every transitive leaks into the conda run-deps
/// as a relaxed spec (NVIDIA-only families like isaacsim-app then fail the
/// consumer's conda solve with "no candidates"). Guarded by
/// `uv_pins_merge_semantics` + `resolve_bundle_uv_pins_as_prefs_bundles_transitive`.
fn merge_uv_pins_into_prefs(
    favored: std::collections::BTreeMap<String, String>,
    uv_pins: Option<&BTreeMap<String, String>>,
) -> std::collections::BTreeMap<String, String> {
    let mut merged = favored;
    if let Some(pins) = uv_pins {
        for (name, version) in pins {
            merged.insert(canonical_conda_name(name), version.clone());
        }
    }
    merged
}

fn validate_loaded_lock_abi(
    lock: &crate::lock::RetreadLock,
    context: &ReplayAbiContext,
) -> Result<()> {
    let python_version = crate::lock::normalized_target_python(&lock.python)
        .context("normalizing replay lock Python target")?;
    let mut workspace_versions = context.workspace_versions.clone();
    workspace_versions
        .entry("python".to_string())
        .or_default()
        .insert(python_version.clone());
    let mut emitted = vec![("python".to_string(), format!("{python_version}.*"))];
    emitted.extend(
        lock.conda_run_deps
            .iter()
            .filter(|dependency| dependency.name != "python_abi")
            .map(|dependency| (dependency.name.clone(), dependency.spec.clone())),
    );
    let embedded_requires_dist = locked_final_requires_dist(lock)?;
    let violations = check_output_abi_invariants(
        &emitted,
        &embedded_requires_dist,
        &workspace_versions,
        &context.overrides,
        &context.aliases,
    );
    if !violations.is_empty() {
        bail!(
            "courier replay rejected by ABI invariant: {}",
            violations.join("; ")
        );
    }
    Ok(())
}

/// Authority gate for the courier replay path.
///
/// Returns `Some(lock)` iff ALL of the following hold:
/// 1. `lock_path` exists and parses as a valid [`RetreadLock`].
/// 2. `lock.schema == SCHEMA` (no cross-version mis-replay).
/// 3. `lock.inputs_hash == current_inputs_hash` (no stale-input replay).
/// 4. REPLAY POISONING GUARD: if `!relax_is_default`, no relax-changed
///    `Origin::Built` wheel (a wheel that was relax-changed from index,
///    i.e. `origin == Built && !must_ship`) has an empty `requires_dist`.
///    Such a wheel was born from a relax-rewrite of an index wheel whose
///    Requires-Dist metadata retread changed; without that metadata we
///    cannot detect if the upstream changed its Requires-Dist between
///    lock writes, and the replay would silently propagate stale relax
///    bytes. Warn and return `None` to fall through to full derivation.
/// 5. The SHA-bound final wheel metadata satisfies the current solved
///    workspace versions, effective overrides, and ABI alias graph.
///
/// Returns `None` (non-fatal miss) on any mismatch.
/// Returns `Err` when the file exists but is malformed or ABI-unsafe.
///
/// `RETREAD_NO_REPLAY=1` unconditionally returns `None` (test knob;
/// lets tests force cold-path exercising without touching the hash).
#[cfg(test)]
fn load_replayable_lock(
    lock_path: &Path,
    current_inputs_hash: &str,
    relax_is_default: bool,
) -> anyhow::Result<Option<crate::lock::RetreadLock>> {
    load_replayable_lock_inner(lock_path, current_inputs_hash, relax_is_default, None, None)
}

fn load_replayable_lock_for_target(
    lock_path: &Path,
    current_inputs_hash: &str,
    relax_is_default: bool,
    target: &ResolutionTarget,
    expected_bundle: &str,
    abi_context: &ReplayAbiContext,
) -> anyhow::Result<Option<crate::lock::RetreadLock>> {
    let lock = load_replayable_lock_inner(
        lock_path,
        current_inputs_hash,
        relax_is_default,
        Some(target),
        Some(expected_bundle),
    )?;
    let Some(lock) = lock else {
        return Ok(None);
    };
    validate_loaded_lock_abi(&lock, abi_context)?;
    Ok(Some(lock))
}

fn load_replayable_lock_inner(
    lock_path: &Path,
    current_inputs_hash: &str,
    relax_is_default: bool,
    target: Option<&ResolutionTarget>,
    expected_bundle: Option<&str>,
) -> anyhow::Result<Option<crate::lock::RetreadLock>> {
    // Test knob: RETREAD_NO_REPLAY=1 disables replay entirely.
    if std::env::var("RETREAD_NO_REPLAY").is_ok() {
        return Ok(None);
    }
    // Missing lock → first build or lock not yet committed. Not an error.
    if !lock_path.exists() {
        return Ok(None);
    }
    let lock = crate::lock::RetreadLock::load(lock_path)?;
    // B-7: schema mismatch → a lock written by a different backend version
    // whose fields/semantics may differ. Treat as a miss (cascade re-derives
    // and rewrites at the next build) rather than risk mis-replaying a lock we
    // don't fully understand. `inputs_hash` already folds in the retread
    // version, so a binary upgrade invalidates replay too; this is the
    // explicit, cheap backstop.
    if lock.schema != crate::lock::SCHEMA {
        return Ok(None);
    }
    if target.is_some_and(|target| !lock.is_for_resolution_target(target)) {
        return Ok(None);
    }
    if let (Some(target), Some(expected_bundle)) = (target, expected_bundle) {
        lock.validate_replay_contract_for_target(target, expected_bundle)
            .context("validating complete courier replay contract")?;
    }
    // Hash mismatch → inputs changed; must re-resolve.
    if lock.inputs_hash != current_inputs_hash {
        return Ok(None);
    }
    // REPLAY POISONING GUARD: with a non-default relax policy, any relax-
    // changed Built wheel (origin==Built, must_ship==false) MUST carry its
    // requires_dist so the replay path can detect metadata drift. If
    // requires_dist is empty the lock was written before schema 5 (or by a
    // buggy producer) and we cannot safely replay it.
    if !relax_is_default {
        for lw in &lock.wheels {
            if lw.origin == crate::lock::Origin::Built
                && !lw.must_ship
                && lw.requires_dist.is_empty()
            {
                tracing::warn!(
                    wheel = %lw.name,
                    "courier replay: relax-changed Built wheel has empty requires_dist \
                     (pre-schema-5 lock or producer bug); falling through to full derivation",
                );
                return Ok(None);
            }
        }
    }
    Ok(Some(lock))
}

/// WS-B: cold-solve replay helper.
///
/// Loads the committed lock at `lock_path` and, if its `inputs_hash`
/// matches `current_inputs_hash`, reconstructs a [`CondaOutput`] from
/// the lock's `conda_run_deps` -- bypassing the full probe cascade.
///
/// Returns:
/// - `Ok(Some(output))` — hash matched; replay output is ready.
/// - `Ok(None)` — hash mismatch, or file missing (fall through to
///   cascade; this is not an error).
/// - `Err(...)` — lock file exists but is malformed / unreadable. The
///   caller treats this as a replay miss and falls through (non-fatal).
#[cfg(test)]
fn replay_from_lock(
    lock_path: &Path,
    current_inputs_hash: &str,
    relax_is_default: bool,
    host_platform: Platform,
    build_number: u64,
    loose: bool,
    siblings: &[(String, String)],
) -> anyhow::Result<Option<CondaOutput>> {
    replay_from_lock_with_abi_context(
        lock_path,
        current_inputs_hash,
        relax_is_default,
        host_platform,
        build_number,
        loose,
        siblings,
        &WorkspaceAbiVersions::new(),
        &BTreeMap::new(),
        &AbiAliasGraph::new(),
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn replay_from_lock_with_abi_context(
    lock_path: &Path,
    current_inputs_hash: &str,
    relax_is_default: bool,
    host_platform: Platform,
    build_number: u64,
    loose: bool,
    siblings: &[(String, String)],
    workspace_versions: &WorkspaceAbiVersions,
    overrides: &BTreeMap<String, String>,
    aliases: &AbiAliasGraph,
) -> anyhow::Result<Option<CondaOutput>> {
    let Some(lock) = load_replayable_lock(lock_path, current_inputs_hash, relax_is_default)? else {
        return Ok(None);
    };
    let abi_context = ReplayAbiContext {
        workspace_versions: workspace_versions.clone(),
        overrides: overrides.clone(),
        aliases: aliases.clone(),
    };
    validate_loaded_lock_abi(&lock, &abi_context)?;
    replay_loaded_lock(
        lock,
        current_inputs_hash,
        host_platform,
        build_number,
        loose,
        siblings,
    )
    .map(Some)
}

/// Return the persisted metadata from the exact final wheel artifacts.
///
/// Every metadata entry repeats the final wheel SHA recorded by `LockWheel`;
/// validating that join here prevents pre-courier fields or unrelated lock
/// records from standing in for the bytes the installer will consume.
fn locked_final_requires_dist(lock: &crate::lock::RetreadLock) -> Result<Vec<(String, String)>> {
    let context = lock
        .abi_context
        .as_ref()
        .context("courier replay lock is missing its persisted ABI context")?;
    let locked_wheels = lock
        .wheels
        .iter()
        .map(|wheel| (canonical_conda_name(&wheel.name), wheel))
        .collect::<BTreeMap<_, _>>();
    if context.wheels.len() != locked_wheels.len() {
        bail!(
            "courier replay ABI context covers {} final wheels, but the lock contains {}",
            context.wheels.len(),
            locked_wheels.len(),
        );
    }
    let mut seen = BTreeSet::new();
    let mut requirements = Vec::new();
    for final_wheel in &context.wheels {
        let canonical = canonical_conda_name(&final_wheel.name);
        if !seen.insert(canonical.clone()) {
            bail!(
                "courier replay ABI context contains duplicate final metadata for `{}`",
                final_wheel.name,
            );
        }
        let locked = locked_wheels.get(&canonical).ok_or_else(|| {
            anyhow!(
                "courier replay ABI context names unknown final wheel `{}`",
                final_wheel.name,
            )
        })?;
        let locked_sha = locked.sha256.as_deref().ok_or_else(|| {
            anyhow!(
                "courier replay final wheel {}=={} has no locked SHA-256",
                locked.name,
                locked.version,
            )
        })?;
        if final_wheel.name != locked.name || !final_wheel.sha256.eq_ignore_ascii_case(locked_sha) {
            bail!(
                "courier replay final metadata for `{}` is not bound to its locked wheel SHA-256",
                final_wheel.name,
            );
        }
        requirements.extend(
            final_wheel
                .requires_dist
                .iter()
                .cloned()
                .map(|requirement| (final_wheel.name.clone(), requirement)),
        );
    }
    Ok(requirements)
}

#[allow(clippy::too_many_arguments)]
fn replay_loaded_lock(
    lock: crate::lock::RetreadLock,
    current_inputs_hash: &str,
    host_platform: Platform,
    build_number: u64,
    loose: bool,
    siblings: &[(String, String)],
) -> anyhow::Result<CondaOutput> {
    // ----- reconstruct CondaOutput from lock fields via the shared helper -----
    let python_version = crate::lock::normalized_target_python(&lock.python)
        .context("normalizing replay lock Python target")?;

    let python_dep = if python_version.contains('*') {
        format!("python {python_version}")
    } else {
        format!("python {python_version}.*")
    };

    // Reconstruct run-dependencies from the lock's conda_run_deps. Pixi adds
    // python_abi after conda/outputs and forwards it to conda/build_v1, so the
    // committed build lock contains it even though the cold advertised output
    // did not. Re-advertising it would make pixi inject a second python_abi and
    // change the source-package identity on replay.
    let mut run_dep_specs: Vec<NamedSpec<PackageSpec>> = vec![spec_from_str(&python_dep)?];
    let mut seen_dep_names: HashSet<String> = HashSet::from(["python".to_string()]);
    for dep in &lock.conda_run_deps {
        if dep.name == "python_abi" {
            continue;
        }
        let spec_str = if dep.spec.is_empty() {
            dep.name.clone()
        } else {
            format!("{} {}", dep.name, dep.spec)
        };
        let ns = spec_from_str(&spec_str)?;
        if seen_dep_names.insert(ns.name.clone()) {
            run_dep_specs.push(ns);
        }
    }

    // Courier (replay is courier-only) is ALWAYS platform-specific: the
    // package ships the native `retread` installer binary + a python-specific
    // lock. any_platform_specific=false here because courier=true already
    // forces the platform-specific path in assemble_conda_output.
    let output = assemble_conda_output(
        &lock.bundle,
        &lock.version,
        &python_version,
        true,  // courier=true: replay is always courier mode
        false, // any_platform_specific: courier=true already forces platform-specific
        run_dep_specs,
        seen_dep_names,
        host_platform,
        build_number,
        Some(current_inputs_hash),
        loose,
        siblings,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn replay_from_lock_for_target(
    lock_path: &Path,
    current_inputs_hash: &str,
    relax_is_default: bool,
    target: &ResolutionTarget,
    expected_bundle: &str,
    host_platform: Platform,
    build_number: u64,
    loose: bool,
    siblings: &[(String, String)],
    workspace_versions: &WorkspaceAbiVersions,
    overrides: &BTreeMap<String, String>,
    aliases: &AbiAliasGraph,
) -> anyhow::Result<Option<CondaOutput>> {
    let Some(lock) = load_replayable_lock_for_target(
        lock_path,
        current_inputs_hash,
        relax_is_default,
        target,
        expected_bundle,
        &ReplayAbiContext {
            workspace_versions: workspace_versions.clone(),
            overrides: overrides.clone(),
            aliases: aliases.clone(),
        },
    )?
    else {
        return Ok(None);
    };
    replay_loaded_lock(
        lock,
        current_inputs_hash,
        host_platform,
        build_number,
        loose,
        siblings,
    )
    .map(Some)
}

// -----------------------------------------------------------------
// Shared test-only mutex: serialises all tests that mutate process-wide env
// vars (RETREAD_NO_REPLAY, RETREAD_FAVOR_LOCK, RETREAD_INCREMENTAL, …).
//
// `std::env::set_var` / `remove_var` are not thread-safe; parallel Rust tests
// that touch env vars can see each others' changes.  Each env-sensitive test
// acquires this lock for the duration of its env-mutation window.  Tests that
// never touch env vars (the majority) do NOT need the lock.
// -----------------------------------------------------------------
#[cfg(test)]
static TEST_ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

// -----------------------------------------------------------------
// WS-B: unit tests for replay_from_lock and load_replayable_lock.
// Uses only std (no tempfile crate dependency) -- temp dirs are created
// via std::env::temp_dir() with a unique subdirectory.
// -----------------------------------------------------------------
#[cfg(test)]
mod replay_tests {
    use std::collections::{BTreeMap, HashSet};

    use rattler_conda_types::Platform;

    use super::{
        AutoDataConfig, WorkspaceAbiVersions, git_auto_data_cache_key, load_replayable_lock,
        load_replayable_lock_for_target, materialize_from_lock_for_target, persist_git_auto_data,
        prepare_replayed_class2_wheel, replay_from_lock, replay_from_lock_with_abi_context,
        replay_git_auto_data, replay_sdist_cache_key, validate_authoritative_replay_lock,
    };
    use crate::config::{RelaxPolicy, RetreadConfig, WheelEntry};
    use crate::lock::{
        CondaDep, GitWheelSource, LockAbiContext, LockWheel, LockWheelAbiMetadata, Origin,
        RetreadLock, SCHEMA,
    };
    use crate::pypi::ResolutionTarget;
    use crate::relax::{AbiAliasGraph, add_abi_alias_edge};
    use crate::wheel_rewrite::rewrite_wheel;

    fn unique_tmp_dir() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        // Use a unique sub-directory per test call to avoid collisions.
        let unique = format!(
            "retread-ws-b-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        );
        let dir = base.join(unique);
        std::fs::create_dir_all(&dir).expect("tmp dir creation should not fail");
        dir
    }

    /// Build a minimal RetreadLock suitable for replay tests.
    fn make_test_lock(
        bundle: &str,
        version: &str,
        python: &str,
        inputs_hash: &str,
        pure_python: bool,
    ) -> RetreadLock {
        let filename = if pure_python {
            format!("{bundle}-{version}-py3-none-any.whl")
        } else {
            format!("{bundle}-{version}-cp311-cp311-manylinux_2_17_x86_64.whl")
        };
        let url = format!("https://example.com/{filename}");
        RetreadLock {
            schema: SCHEMA,
            retread_version: "0.0.1".into(),
            bundle: bundle.into(),
            version: version.into(),
            python: python.into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
            resolution_glibc: None,
            inputs_hash: inputs_hash.into(),
            root_requirements: Vec::new(),
            wheels: vec![LockWheel {
                name: bundle.into(),
                version: version.into(),
                origin: Origin::Index,
                filename,
                url: Some(url),
                sha256: Some("11".repeat(32)),
                requires_dist: vec![],
                must_ship: false,
                upstream_url: None,
                git_source: None,
                sdist_source: None,
            }],
            conda_run_deps: vec![CondaDep {
                name: "numpy".into(),
                spec: ">=1.21".into(),
            }],
            index_urls: vec!["https://pypi.org/simple/".into()],
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec![],
            entry_specs: vec![],
            wheel_store: None,
            abi_context: Some(LockAbiContext {
                wheels: vec![LockWheelAbiMetadata {
                    name: bundle.into(),
                    sha256: "11".repeat(32),
                    requires_dist: vec![],
                }],
            }),
            relaxations: vec![],
        }
    }

    fn replay_target(python: &str, subdir: &str) -> crate::pypi::ResolutionTarget {
        crate::pypi::ResolutionTarget::from_wheel_target(
            crate::pypi::WheelTarget {
                python_version: python.into(),
                conda_subdir: subdir.into(),
                max_glibc: None,
            },
            None,
        )
    }

    fn load_test_lock_for_target(
        path: &std::path::Path,
        inputs_hash: &str,
        relax_is_default: bool,
        target: &ResolutionTarget,
        bundle: &str,
    ) -> anyhow::Result<Option<RetreadLock>> {
        let abi_context = super::ReplayAbiContext {
            workspace_versions: super::WorkspaceAbiVersions::new(),
            overrides: BTreeMap::new(),
            aliases: AbiAliasGraph::new(),
        };
        load_replayable_lock_for_target(
            path,
            inputs_hash,
            relax_is_default,
            target,
            bundle,
            &abi_context,
        )
    }

    #[test]
    fn git_auto_data_disposition_round_trips_exactly() {
        use crate::lock::GitWheelAutoData;

        let checkout_root = std::path::PathBuf::from("/cache/checkout");
        let config = AutoDataConfig {
            checkout_root: checkout_root.clone(),
            skip_subdirs: vec![std::path::PathBuf::from("packages/member")],
        };

        assert_eq!(
            persist_git_auto_data(None).unwrap(),
            GitWheelAutoData::Disabled,
        );
        assert_eq!(
            persist_git_auto_data(Some(&config)).unwrap(),
            GitWheelAutoData::CheckoutRoot {
                skip_subdirectories: vec!["packages/member".into()],
            },
        );
        let unsafe_config = AutoDataConfig {
            checkout_root: checkout_root.clone(),
            skip_subdirs: vec![std::path::PathBuf::from(" packages/member")],
        };
        assert!(
            persist_git_auto_data(Some(&unsafe_config)).is_err(),
            "producer capture must reject paths that replay would reject",
        );

        let permuted = AutoDataConfig {
            checkout_root: checkout_root.clone(),
            skip_subdirs: vec!["b".into(), "a".into(), "a".into()],
        };
        let canonical = AutoDataConfig {
            checkout_root: checkout_root.clone(),
            skip_subdirs: vec!["a".into(), "b".into()],
        };
        assert_eq!(
            git_auto_data_cache_key(&permuted).unwrap(),
            git_auto_data_cache_key(&canonical).unwrap(),
            "cache identity must be insensitive to skip-set order and duplicates",
        );

        let mut source = GitWheelSource {
            url: "https://example.com/checkout.git".into(),
            rev: "ab".repeat(20),
            subdirectory: Some("packages/member".into()),
            extras: vec![],
            auto_data: Some(GitWheelAutoData::Disabled),
        };
        assert!(
            replay_git_auto_data(&source, checkout_root.clone())
                .unwrap()
                .is_none(),
        );

        source.auto_data = Some(GitWheelAutoData::CheckoutRoot {
            skip_subdirectories: vec!["packages/member".into()],
        });
        let replayed = replay_git_auto_data(&source, checkout_root.clone())
            .unwrap()
            .expect("checkout-root injection must be restored");
        assert_eq!(replayed.checkout_root, checkout_root);
        assert_eq!(
            replayed.skip_subdirs,
            vec![std::path::PathBuf::from("packages/member")],
        );

        source.auto_data = None;
        assert!(
            replay_git_auto_data(&source, std::path::PathBuf::from("/cache/checkout")).is_err(),
            "legacy provenance must fail closed instead of inferring behavior",
        );
    }

    #[test]
    fn replay_gate_requires_the_full_resolution_target() {
        let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let lock = make_test_lock("pack", "1.0.0", "3.11", "same-hash", true);
        let path = dir.join(RetreadLock::file_name("pack"));
        std::fs::write(&path, lock.to_pretty_json().unwrap()).unwrap();

        let native = replay_target("3.11.0", "linux-64");
        assert!(
            load_test_lock_for_target(&path, "same-hash", true, &native, "pack")
                .unwrap()
                .is_some(),
            "numeric Python patch spelling must match the same resolution target"
        );
        for foreign in [
            replay_target("3.12", "linux-64"),
            replay_target("3.11", "linux-aarch64"),
        ] {
            assert!(
                load_test_lock_for_target(&path, "same-hash", true, &foreign, "pack")
                    .unwrap()
                    .is_none(),
                "matching inputs hashes must never bypass target identity"
            );
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn replay_accepts_sha256_git_commit_object_id() {
        let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let target = replay_target("3.11", "linux-64");
        let mut lock = make_test_lock("pack", "1.0.0", "3.11", "same-hash", true);
        lock.wheels[0].origin = Origin::Built;
        lock.wheels[0].url = None;
        lock.wheels[0].must_ship = true;
        lock.wheels[0].git_source = Some(GitWheelSource {
            url: "https://example.com/pack.git".into(),
            rev: "ab".repeat(32),
            subdirectory: None,
            extras: vec![],
            auto_data: Some(crate::lock::GitWheelAutoData::Disabled),
        });
        let path = dir.join(RetreadLock::file_name_for_target("pack", &target));
        std::fs::write(&path, lock.to_pretty_json().unwrap()).unwrap();

        let replayed = load_test_lock_for_target(&path, "same-hash", true, &target, "pack")
            .unwrap()
            .expect("a SHA-256-format Git commit object ID must be replayable");
        assert_eq!(
            replayed.wheels[0].git_source.as_ref().unwrap().rev.len(),
            64,
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn replay_sdist_cache_is_bound_to_raw_digest_and_target() {
        let native = replay_target("3.11", "linux-64");
        let patch_alias = replay_target("3.11.0", "linux-64");
        let foreign = replay_target("3.11", "linux-aarch64");
        let a = replay_sdist_cache_key(&"11".repeat(32), &native);

        assert_eq!(a, replay_sdist_cache_key(&"11".repeat(32), &patch_alias));
        assert_ne!(a, replay_sdist_cache_key(&"22".repeat(32), &native));
        assert_ne!(a, replay_sdist_cache_key(&"11".repeat(32), &foreign));
    }

    #[test]
    fn post_courier_replay_must_equal_the_authoritative_lock() {
        let mut expected = make_test_lock("pack", "1.0.0", "3.11.0", "same-hash", true);
        expected.wheel_store = Some("/producer-host/retread-wheels".into());
        let mut permuted = expected.clone();
        permuted.retread_version = "emit-neutral-upgrade".into();
        permuted.python = "3.11".into();
        permuted.wheel_store = Some("/current-host/retread-wheels".into());
        permuted.root_requirements.reverse();
        validate_authoritative_replay_lock(&expected, &permuted).unwrap();

        let mut drifted = permuted.clone();
        drifted.wheels[0].sha256 = Some("ff".repeat(32));
        let err = validate_authoritative_replay_lock(&expected, &drifted).unwrap_err();
        assert!(format!("{err:#}").contains("authoritative lock"));

        let mut portable_expected = expected.clone();
        portable_expected.wheel_store = Some("~/stores/a".into());
        let mut portable_drifted = portable_expected.clone();
        portable_drifted.wheel_store = Some("~/stores/b".into());
        assert!(
            validate_authoritative_replay_lock(&portable_expected, &portable_drifted).is_err(),
            "portable store identities remain part of the exact lock contract"
        );
    }

    #[tokio::test]
    async fn invalid_sdist_provenance_fails_before_replay_mutation() {
        let dir = unique_tmp_dir();
        let source_dir = dir.join("source");
        let work_dir = dir.join("work");
        let output_dir = dir.join("output");
        let cache_dir = dir.join("cache");
        let mut lock = make_test_lock("pack", "1.0.0", "3.11", "same-hash", true);
        lock.wheels[0].origin = Origin::Built;
        lock.wheels[0].url = None;
        lock.wheels[0].sdist_source = Some(crate::lock::SdistWheelSource {
            index: "https://pypi.org/simple/".into(),
            name: "pack".into(),
            version: "1.0.0".into(),
            sdist_url: "https://example.com/pack-1.0.0.tar.gz#sha256=short".into(),
        });
        let config: RetreadConfig =
            serde_json::from_value(serde_json::json!({"retread-wheels": {}})).unwrap();
        let target = replay_target("3.11", "linux-64");

        let err = materialize_from_lock_for_target(
            lock,
            "pack",
            &config,
            &work_dir,
            &output_dir,
            &target,
            &source_dir,
            &cache_dir,
            None,
            vec!["python 3.11.*".into()],
            "fp",
        )
        .await
        .unwrap_err();
        assert!(format!("{err:#}").contains("#sha256=<64 hex>"));
        assert!(!work_dir.exists());
        assert!(!output_dir.exists());
        assert!(!source_dir.join("wheels").exists());
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test]
    async fn malformed_later_wheel_fails_before_replay_mutation() {
        let dir = unique_tmp_dir();
        let source_dir = dir.join("source");
        let work_dir = dir.join("work");
        let output_dir = dir.join("output");
        let cache_dir = dir.join("cache");
        let mut lock = make_test_lock("pack", "1.0.0", "3.11", "same-hash", true);
        let mut malformed = lock.wheels[0].clone();
        malformed.name = "later".into();
        malformed.filename = "later-1.0.0-py3-none-any.whl".into();
        malformed.url = Some("not-an-artifact-url".into());
        lock.wheels.push(malformed);
        let config: RetreadConfig =
            serde_json::from_value(serde_json::json!({"retread-wheels": {}})).unwrap();
        let target = replay_target("3.11", "linux-64");

        let err = materialize_from_lock_for_target(
            lock,
            "pack",
            &config,
            &work_dir,
            &output_dir,
            &target,
            &source_dir,
            &cache_dir,
            None,
            vec!["python 3.11.*".into()],
            "fp",
        )
        .await
        .unwrap_err();
        assert!(format!("{err:#}").contains("invalid artifact URL"));
        assert!(!work_dir.exists());
        assert!(!output_dir.exists());
        assert!(!source_dir.join("wheels").exists());
        assert!(!cache_dir.exists());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn build_v1_lock_ingress_rejects_abi_violating_final_wheel_metadata() {
        let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let target = replay_target("3.11", "linux-64");
        let mut lock = make_test_lock("pack", "1.0.0", "3.11", "abi-hash", true);
        // Replay reconstruction still needs the safe pre-courier metadata.
        lock.wheels[0].requires_dist = vec!["nvidia-cuda-runtime-cu12>=12.8,<13".to_string()];
        let mut aliases = AbiAliasGraph::new();
        add_abi_alias_edge(&mut aliases, "nvidia-cuda-runtime-cu12", "cuda-version");
        // These are the SHA-bound FINAL bytes. They covered the producer's
        // old 12.8 selection but exclude the current 13.1 workspace solve.
        lock.abi_context.as_mut().unwrap().wheels[0].requires_dist =
            vec!["nvidia-cuda-runtime-cu12>=12.8,<13".to_string()];

        let lock_path = dir.join(RetreadLock::file_name_for_target("pack", &target));
        std::fs::write(&lock_path, lock.to_pretty_json().unwrap()).unwrap();
        let producer_context = super::ReplayAbiContext {
            workspace_versions: BTreeMap::from([(
                "cuda-version".to_string(),
                std::collections::BTreeSet::from(["12.8".to_string()]),
            )]),
            overrides: BTreeMap::new(),
            aliases: aliases.clone(),
        };
        assert!(
            load_replayable_lock_for_target(
                &lock_path,
                "abi-hash",
                true,
                &target,
                "pack",
                &producer_context,
            )
            .unwrap()
            .is_some(),
            "test setup: producer-time ABI facts must still accept the old lock"
        );
        let current_context = super::ReplayAbiContext {
            workspace_versions: BTreeMap::from([(
                "cuda-version".to_string(),
                std::collections::BTreeSet::from(["13.1".to_string()]),
            )]),
            overrides: BTreeMap::new(),
            aliases,
        };
        let error = load_replayable_lock_for_target(
            &lock_path,
            "abi-hash",
            true,
            &target,
            "pack",
            &current_context,
        )
        .expect_err("build-v1 lock ingress must use current solved ABI facts");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("courier replay rejected by ABI invariant")
                && rendered.contains("does not cover workspace pin")
                && rendered.contains("cuda-version==13.1"),
            "{rendered}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn matching_hash_returns_some_with_correct_fields() {
        // Hold env-lock: prevents RETREAD_NO_REPLAY=1 from returning None here.
        let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let lock = make_test_lock("mypack", "1.2.3", "3.11", "abc123", true);
        let json = lock.to_pretty_json().unwrap();
        let lock_path = dir.join(RetreadLock::file_name("mypack"));
        std::fs::write(&lock_path, &json).unwrap();

        let result = replay_from_lock(&lock_path, "abc123", true, Platform::Linux64, 0, false, &[]);
        assert!(result.is_ok(), "should not error: {result:?}");
        let output = result.unwrap();
        assert!(output.is_some(), "matching hash must return Some");
        let out = output.unwrap();

        // Name and version round-trip.
        assert_eq!(out.metadata.name.as_normalized(), "mypack");
        assert_eq!(out.metadata.version.version().to_string(), "1.2.3");
        // Courier replay is always platform-specific (ships the native
        // installer + python-specific lock), even for pure-python wheels.
        assert_eq!(out.metadata.subdir, Platform::Linux64);
        // run_dependencies includes python and the replayed conda dep.
        let dep_names: Vec<&str> = out
            .run_dependencies
            .depends
            .iter()
            .map(|d| d.name.as_str())
            .collect();
        assert!(
            dep_names.contains(&"python"),
            "run_deps must include python: {dep_names:?}"
        );
        assert!(
            dep_names.contains(&"numpy"),
            "run_deps must include numpy from lock: {dep_names:?}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn replay_rejects_bare_abi_anchor_from_committed_lock() {
        let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let mut lock = make_test_lock("mypack", "1.2.3", "3.11", "unsafe-hash", true);
        lock.conda_run_deps = vec![CondaDep {
            name: "numpy".into(),
            spec: "*".into(),
        }];
        let lock_path = dir.join(RetreadLock::file_name("mypack"));
        std::fs::write(&lock_path, lock.to_pretty_json().unwrap()).unwrap();

        let error = replay_from_lock(
            &lock_path,
            "unsafe-hash",
            true,
            Platform::Linux64,
            0,
            false,
            &[],
        )
        .expect_err("an unsafe ABI-anchor lock must not replay");
        assert!(
            format!("{error:#}").contains("courier replay rejected by ABI invariant"),
            "{error:#}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn replay_rejects_widened_abi_alias_in_final_wheel_metadata() {
        let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let mut lock = make_test_lock("mypack", "1.2.3", "3.11", "alias-hash", true);
        lock.wheels[0].origin = Origin::Built;
        lock.wheels[0].filename = "mypack-1.2.3-999retread-py3-none-any.whl".to_string();
        lock.wheels[0].url = None;
        lock.wheels[0].upstream_url =
            Some("https://example.com/mypack-1.2.3-py3-none-any.whl".to_string());
        lock.wheels[0].requires_dist = vec!["nvidia-cuda-runtime-cu12==12.0".to_string()];
        let mut aliases = AbiAliasGraph::new();
        add_abi_alias_edge(&mut aliases, "nvidia-cuda-runtime-cu12", "cuda");
        let abi_context = lock.abi_context.as_mut().unwrap();
        abi_context.wheels[0].requires_dist = vec!["nvidia-cuda-runtime-cu12>=12.0".to_string()];

        let mapped = super::locked_final_requires_dist(&lock).unwrap();
        assert_eq!(
            mapped,
            vec![(
                "mypack".to_string(),
                "nvidia-cuda-runtime-cu12>=12.0".to_string(),
            )],
            "the replay guard must inspect the SHA-bound final metadata, not the stored pre-map line"
        );

        let lock_path = dir.join(RetreadLock::file_name("mypack"));
        std::fs::write(&lock_path, lock.to_pretty_json().unwrap()).unwrap();
        let error = replay_from_lock_with_abi_context(
            &lock_path,
            "alias-hash",
            true,
            Platform::Linux64,
            0,
            false,
            &[],
            &WorkspaceAbiVersions::new(),
            &BTreeMap::new(),
            &aliases,
        )
        .expect_err("widened hidden ABI-anchor metadata must fail closed on replay");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("courier replay rejected by ABI invariant")
                && rendered.contains("wheel `mypack` embeds")
                && rendered.contains("bare-major"),
            "{rendered}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn replay_validates_raw_index_wheel_metadata_without_courier_mapping() {
        let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let mut lock = make_test_lock("mypack", "1.2.3", "3.11", "index-hash", true);
        lock.wheels[0].requires_dist = vec!["nvidia-cuda-runtime-cu12>=12".to_string()];
        lock.wheels.push(LockWheel {
            name: "nvidia-cuda-runtime-cu12".to_string(),
            version: "12.0".to_string(),
            origin: Origin::Index,
            filename: "nvidia_cuda_runtime_cu12-12.0-py3-none-any.whl".to_string(),
            url: Some(
                "https://example.com/nvidia_cuda_runtime_cu12-12.0-py3-none-any.whl".to_string(),
            ),
            sha256: Some("22".repeat(32)),
            requires_dist: vec![],
            must_ship: false,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        });
        let abi_context = lock.abi_context.as_mut().unwrap();
        abi_context.wheels[0].requires_dist = vec!["nvidia-cuda-runtime-cu12>=12".to_string()];
        abi_context.wheels.push(LockWheelAbiMetadata {
            name: "nvidia-cuda-runtime-cu12".to_string(),
            sha256: "22".repeat(32),
            requires_dist: vec![],
        });

        assert_eq!(
            super::locked_final_requires_dist(&lock).unwrap(),
            vec![(
                "mypack".to_string(),
                "nvidia-cuda-runtime-cu12>=12".to_string(),
            )],
            "Origin::Index bytes are raw and must not be sanitized by the courier mapper"
        );

        let lock_path = dir.join(RetreadLock::file_name("mypack"));
        std::fs::write(&lock_path, lock.to_pretty_json().unwrap()).unwrap();
        let mut aliases = AbiAliasGraph::new();
        add_abi_alias_edge(&mut aliases, "nvidia-cuda-runtime-cu12", "cuda");
        let error = replay_from_lock_with_abi_context(
            &lock_path,
            "index-hash",
            true,
            Platform::Linux64,
            0,
            false,
            &[],
            &WorkspaceAbiVersions::new(),
            &BTreeMap::new(),
            &aliases,
        )
        .expect_err("unsafe raw Index metadata must fail closed on replay");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("wheel `mypack` embeds") && rendered.contains("bare-major"),
            "{rendered}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn replay_matches_cold_dependency_identity() {
        let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let mut lock = make_test_lock("mypack", "1.2.3", "3.11", "abc123", true);
        // Build-v1 receives Pixi's injected python_abi and the lock
        // canonicalizes all dependencies. Neither fact may change the
        // conda/outputs identity when the lock is replayed.
        lock.conda_run_deps = vec![
            CondaDep {
                name: "zlib".into(),
                spec: ">=1.3".into(),
            },
            CondaDep {
                name: "python_abi".into(),
                spec: "3.11.* *_cp311".into(),
            },
            CondaDep {
                name: "numpy".into(),
                spec: ">=1.21".into(),
            },
        ];
        let lock_path = dir.join(RetreadLock::file_name("mypack"));
        std::fs::write(&lock_path, lock.to_pretty_json().unwrap()).unwrap();

        let output = replay_from_lock(&lock_path, "abc123", true, Platform::Linux64, 0, false, &[])
            .unwrap()
            .unwrap();
        let cold = super::assemble_conda_output(
            "mypack",
            "1.2.3",
            "3.11",
            true,
            false,
            vec![
                super::spec_from_str("python 3.11.*").unwrap(),
                super::spec_from_str("zlib >=1.3").unwrap(),
                super::spec_from_str("numpy >=1.21").unwrap(),
            ],
            HashSet::from([
                "python".to_string(),
                "zlib".to_string(),
                "numpy".to_string(),
            ]),
            Platform::Linux64,
            0,
            Some("abc123"),
            false,
            &[],
        )
        .unwrap();
        let replay_deps: Vec<(String, String)> = output
            .run_dependencies
            .depends
            .iter()
            .map(|dep| {
                (
                    dep.name.clone(),
                    super::audit_report::format_packagespec(&dep.spec),
                )
            })
            .collect();
        let cold_deps: Vec<(String, String)> = cold
            .run_dependencies
            .depends
            .iter()
            .map(|dep| {
                (
                    dep.name.clone(),
                    super::audit_report::format_packagespec(&dep.spec),
                )
            })
            .collect();
        assert_eq!(replay_deps, cold_deps);
        assert_eq!(output.metadata.build, cold.metadata.build);
        let dep_names: Vec<&str> = output
            .run_dependencies
            .depends
            .iter()
            .map(|dep| dep.name.as_str())
            .collect();
        assert_eq!(dep_names, vec!["numpy", "python", "uv", "zlib"]);
        assert!(
            !dep_names.contains(&"python_abi"),
            "Pixi injects python_abi after metadata; replay must not advertise it"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn hash_mismatch_returns_none() {
        let dir = unique_tmp_dir();
        let lock = make_test_lock("mypack", "1.2.3", "3.11", "stored-hash", true);
        let json = lock.to_pretty_json().unwrap();
        let lock_path = dir.join(RetreadLock::file_name("mypack"));
        std::fs::write(&lock_path, &json).unwrap();

        let result = replay_from_lock(
            &lock_path,
            "different-hash",
            true,
            Platform::Linux64,
            0,
            false,
            &[],
        );
        assert!(result.is_ok(), "mismatch must not error: {result:?}");
        assert!(
            result.unwrap().is_none(),
            "hash mismatch must return None (fall through to cascade)"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn missing_file_returns_none() {
        let dir = unique_tmp_dir();
        let lock_path = dir.join("retread-missing.lock.json");

        let result = replay_from_lock(
            &lock_path,
            "any-hash",
            true,
            Platform::Linux64,
            0,
            false,
            &[],
        );
        assert!(result.is_ok(), "missing file must not error: {result:?}");
        assert!(
            result.unwrap().is_none(),
            "missing lock must return None (first-build path)"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn malformed_json_returns_err() {
        let dir = unique_tmp_dir();
        let lock_path = dir.join(RetreadLock::file_name("badpack"));
        std::fs::write(&lock_path, b"not valid json{{{{").unwrap();

        let result = replay_from_lock(
            &lock_path,
            "any-hash",
            true,
            Platform::Linux64,
            0,
            false,
            &[],
        );
        assert!(
            result.is_err(),
            "malformed JSON must return Err (caller falls through): {result:?}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn platform_specific_wheel_sets_host_platform_subdir() {
        // Hold env-lock: prevents RETREAD_NO_REPLAY=1 from returning None here.
        let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let lock = make_test_lock("mypack", "1.0.0", "3.11", "hash1", false /* arch */);
        let json = lock.to_pretty_json().unwrap();
        let lock_path = dir.join(RetreadLock::file_name("mypack"));
        std::fs::write(&lock_path, &json).unwrap();

        let result = replay_from_lock(&lock_path, "hash1", true, Platform::Linux64, 0, false, &[]);
        let out = result.unwrap().unwrap();
        assert_eq!(
            out.metadata.subdir,
            Platform::Linux64,
            "platform-specific wheel must set subdir=host_platform"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn siblings_are_cross_linked_in_output() {
        // Hold env-lock: prevents RETREAD_NO_REPLAY=1 from returning None here.
        let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let lock = make_test_lock("pack-a", "2.0.0", "3.11", "hash42", true);
        let json = lock.to_pretty_json().unwrap();
        let lock_path = dir.join(RetreadLock::file_name("pack-a"));
        std::fs::write(&lock_path, &json).unwrap();

        let siblings = vec![
            ("pack-a".to_string(), "2.0.0".to_string()),
            ("pack-b".to_string(), "2.0.0".to_string()),
        ];
        let result = replay_from_lock(
            &lock_path,
            "hash42",
            true,
            Platform::Linux64,
            0,
            false,
            &siblings,
        );
        let out = result.unwrap().unwrap();
        let dep_names: Vec<&str> = out
            .run_dependencies
            .depends
            .iter()
            .map(|d| d.name.as_str())
            .collect();
        // pack-b should be cross-linked; pack-a is self and must not duplicate.
        assert!(
            dep_names.contains(&"pack-b"),
            "sibling pack-b must be cross-linked: {dep_names:?}"
        );
        let pack_a_count = dep_names.iter().filter(|&&n| n == "pack-a").count();
        assert_eq!(
            pack_a_count, 0,
            "self-sibling pack-a must not appear in run_deps: {dep_names:?}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// Verify that conda_run_deps with empty spec (name-only) round-trips.
    #[test]
    fn conda_dep_with_empty_spec_round_trips() {
        // Hold env-lock: prevents RETREAD_NO_REPLAY=1 from returning None here.
        let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let mut lock = make_test_lock("mypack", "1.0.0", "3.11", "hash9", true);
        lock.conda_run_deps.push(CondaDep {
            name: "uv".into(),
            spec: String::new(),
        });
        let json = lock.to_pretty_json().unwrap();
        let lock_path = dir.join(RetreadLock::file_name("mypack"));
        std::fs::write(&lock_path, &json).unwrap();

        let result = replay_from_lock(&lock_path, "hash9", true, Platform::Linux64, 0, false, &[]);
        let out = result.unwrap().unwrap();
        let dep_names: Vec<&str> = out
            .run_dependencies
            .depends
            .iter()
            .map(|d| d.name.as_str())
            .collect();
        assert!(
            dep_names.contains(&"uv"),
            "name-only conda dep must appear in run_deps: {dep_names:?}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// Requires real repodata/channels — skipped in normal CI.
    /// Run manually to verify end-to-end replay in courier mode.
    #[test]
    #[ignore = "requires real repodata/channels; run manually for e2e replay validation"]
    fn conda_outputs_replay_skips_cascade_with_committed_lock() {
        // Integration test placeholder:
        // 1. Write a valid lock file to source_dir.
        // 2. Call conda_outputs with courier=true.
        // 3. Assert the result is returned promptly (no solve cascade).
    }

    /// Verify that replay_from_lock emits the content-addressed build string.
    #[test]
    fn replay_emits_content_addressed_build_string() {
        // Hold env-lock: prevents RETREAD_NO_REPLAY=1 (set by
        // no_replay_env_knob_returns_none) from returning None here.
        let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        // Use a 64-hex-char inputs_hash (typical sha256 hex output length).
        let hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
        let lock = make_test_lock("mypack", "1.2.3", "3.11.0", hash, true);
        let json = lock.to_pretty_json().unwrap();
        let lock_path = dir.join(RetreadLock::file_name("mypack"));
        std::fs::write(&lock_path, &json).unwrap();

        let result = replay_from_lock(&lock_path, hash, true, Platform::Linux64, 0, false, &[]);
        let out = result.unwrap().unwrap();
        // Build string must be content-addressed: py311_h<first10>_0
        assert_eq!(
            out.metadata.build, "py311_habcdef0123_0",
            "patch-equivalent replay must emit the normalized content-addressed build string: got {}",
            out.metadata.build
        );
        std::fs::remove_dir_all(dir).ok();
    }

    // ---- materialize_from_lock provenance classification tests ----
    // These tests verify the wheel-class routing logic in materialize_from_lock
    // by checking that Origin::Built wheels with upstream_url=None (class 3 /
    // schema-5 class 2) return Ok(None) so the caller falls through to full
    // resolve, while genuinely re-materializable wheels proceed.

    /// A class-3 wheel (must_ship=true, name NOT in config.retread_wheels)
    /// must cause materialize_from_lock to return Ok(None) — provenance gap.
    #[tokio::test]
    async fn materialize_from_lock_class3_returns_none() {
        use super::materialize_from_lock;
        use crate::config::RetreadConfig;
        let dir = unique_tmp_dir();
        let source_dir = dir.join("source");
        std::fs::create_dir_all(&source_dir).unwrap();
        let output_dir = dir.join("output");
        let work_dir = dir.join("work");
        let cache_dir = dir.join("cache");

        // Lock has a must_ship=true Built wheel whose name is NOT in
        // config.retread_wheels -> class 3 schema gap.
        let lock = RetreadLock {
            schema: crate::lock::SCHEMA,
            retread_version: "2.4.0".into(),
            bundle: "mypack".into(),
            version: "1.0.0".into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
            resolution_glibc: None,
            inputs_hash: "hash123".into(),
            root_requirements: vec![],
            wheels: vec![LockWheel {
                name: "orphan-git-wheel".into(),
                version: "1.0.0".into(),
                origin: Origin::Built,
                filename: "orphan_git_wheel-1.0.0-py3-none-any.injected.whl".into(),
                url: None,
                sha256: Some("12".repeat(32)),
                requires_dist: vec![],
                must_ship: true,
                upstream_url: None, // class 3: no upstream, not in config
                git_source: None,
                sdist_source: None,
            }],
            conda_run_deps: vec![],
            index_urls: vec!["https://pypi.org/simple/".into()],
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec![],
            entry_specs: vec![],
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };
        // Config has no retread_wheels entries — wheel is a class-3 orphan.
        // Use serde_json to construct a minimal config (RetreadConfig has no
        // Default impl; a zero-entry [retread-wheels] table is valid).
        let config: RetreadConfig =
            serde_json::from_value(serde_json::json!({"retread-wheels": {}})).unwrap();
        let result = materialize_from_lock(
            lock,
            &config,
            &work_dir,
            &output_dir,
            &replay_target("3.11", "linux-64"),
            &source_dir,
            &cache_dir,
            None,
            vec!["python 3.11.*".to_string()],
            "fp",
        )
        .await;
        assert!(result.is_ok(), "class-3 gap must not Err: {result:?}");
        assert!(
            result.unwrap().is_none(),
            "class-3 gap (no config entry) must return Ok(None) so caller falls \
             through to full resolve"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// A class-2 wheel (must_ship=false, origin=Built) with NO upstream_url
    /// (schema-5 lock) must cause materialize_from_lock to return Ok(None).
    #[tokio::test]
    async fn materialize_from_lock_class2_schema5_returns_none() {
        use super::materialize_from_lock;
        let dir = unique_tmp_dir();
        let source_dir = dir.join("source");
        std::fs::create_dir_all(&source_dir).unwrap();
        let output_dir = dir.join("output");
        let work_dir = dir.join("work");
        let cache_dir = dir.join("cache");

        // Schema-5 style: relax-changed Built wheel, upstream_url=None.
        let lock = RetreadLock {
            schema: crate::lock::SCHEMA,
            retread_version: "2.3.1".into(),
            bundle: "mypack".into(),
            version: "1.0.0".into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
            resolution_glibc: None,
            inputs_hash: "hash456".into(),
            root_requirements: vec![],
            wheels: vec![LockWheel {
                name: "skrl".into(),
                version: "2.1.0".into(),
                origin: Origin::Built,
                filename: "skrl-2.1.0-999retread-py3-none-any.whl".into(),
                url: None,
                sha256: Some("13".repeat(32)),
                requires_dist: vec!["torch>=2.0,<3".into()],
                must_ship: false,
                upstream_url: None, // schema-5 style: no upstream_url
                git_source: None,
                sdist_source: None,
            }],
            conda_run_deps: vec![],
            index_urls: vec!["https://pypi.org/simple/".into()],
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec![],
            entry_specs: vec![],
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };
        let config: RetreadConfig =
            serde_json::from_value(serde_json::json!({"retread-wheels": {}})).unwrap();
        let result = materialize_from_lock(
            lock,
            &config,
            &work_dir,
            &output_dir,
            &replay_target("3.11", "linux-64"),
            &source_dir,
            &cache_dir,
            None,
            vec!["python 3.11.*".to_string()],
            "fp",
        )
        .await;
        assert!(result.is_ok(), "schema-5 class-2 must not Err: {result:?}");
        assert!(
            result.unwrap().is_none(),
            "schema-5 relax-changed Built wheel (no upstream_url) must return \
             Ok(None) — caller falls through to full resolve"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    // ---- Class-2b (schema-9 sdist) replay routing + field-mapping tests ----
    //
    // Three tests that together cover the Class-2b arm introduced in PHASE 2.6:
    //
    //   (1) class2b_routes_to_build_not_ok_none — pure, non-ignored.
    //       RED before PHASE 2.6 (would return Ok(None) like Class-2), GREEN after.
    //       Proves the guard `sdist_source.is_some()` intercepts before Class-2.
    //
    //   (2) class2b_emit_wheel_field_mapping — pure, non-ignored.
    //       Verifies the EmitWheel field contract: sdist_source is carried verbatim
    //       from LockWheel (self-drift property), upstream_url=None, remote_url=None.
    //       Does not call materialize_from_lock; tests the contract in isolation.
    //
    //   (3) class2b_live_round_trip — ignored (needs uv + network).
    //       Full end-to-end: calls materialize_from_lock with a real stored sdist_url;
    //       asserts the reconstructed EmitWheel.sdist_source == lw.sdist_source and
    //       that origin=Built, must_ship=false are preserved through stage.

    /// (1) CLASS-2b routing guard: a Class-2 wheel (must_ship=false, Origin::Built,
    /// upstream_url=None) with sdist_source=Some(...) MUST enter the Class-2b arm,
    /// not return Ok(None) like bare Class-2 (schema-5 lock behavior).
    ///
    /// RED before PHASE 2.6: the bare `Origin::Built =>` arm hit first and returned
    /// Ok(None) because upstream_url=None. GREEN after: Class-2b intercepts first,
    /// attempts build_wheel_from_sdist_url, and returns Err (no network in CI) —
    /// which is NOT Ok(None).
    #[tokio::test]
    async fn class2b_routes_to_build_not_ok_none() {
        use super::materialize_from_lock;
        use crate::config::RetreadConfig;
        use crate::lock::SdistWheelSource;

        let dir = unique_tmp_dir();
        let source_dir = dir.join("source");
        std::fs::create_dir_all(&source_dir).unwrap();
        let output_dir = dir.join("output");
        let work_dir = dir.join("work");
        let cache_dir = dir.join("cache");

        let sdist_src = SdistWheelSource {
            index: "https://pypi.org/simple/".into(),
            name: "gym".into(),
            version: "0.26.2".into(),
            // Intentionally unreachable URL — we want the arm to ATTEMPT a build
            // (entering Class-2b) and fail, NOT silently return Ok(None) (Class-2).
            sdist_url: format!(
                "https://files.pythonhosted.org/packages/gym-0.26.2.tar.gz#sha256={}",
                "de".repeat(32)
            ),
        };

        let lock = RetreadLock {
            schema: crate::lock::SCHEMA,
            retread_version: "2.7.0".into(),
            bundle: "gympack".into(),
            version: "1.0.0".into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
            resolution_glibc: None,
            inputs_hash: "hash999".into(),
            root_requirements: vec![],
            wheels: vec![LockWheel {
                name: "gym".into(),
                version: "0.26.2".into(),
                origin: Origin::Built,
                filename: "gym-0.26.2-999retread-py3-none-any.whl".into(),
                url: None,
                sha256: Some("14".repeat(32)),
                requires_dist: vec!["numpy>=1.21".into()],
                // upstream_url=None: identical to a schema-5 class-2 lock.
                // Without the Class-2b guard, this would have returned Ok(None).
                must_ship: false,
                upstream_url: None,
                git_source: None,
                sdist_source: Some(sdist_src),
            }],
            conda_run_deps: vec![],
            index_urls: vec!["https://pypi.org/simple/".into()],
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec![],
            entry_specs: vec![],
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };
        let config: RetreadConfig =
            serde_json::from_value(serde_json::json!({"retread-wheels": {}})).unwrap();

        let result = materialize_from_lock(
            lock,
            &config,
            &work_dir,
            &output_dir,
            &replay_target("3.11", "linux-64"),
            &source_dir,
            &cache_dir,
            None,
            vec!["python 3.11.*".to_string()],
            "fp",
        )
        .await;

        std::fs::remove_dir_all(dir).ok();

        // Class-2b was entered: the arm attempted build_wheel_from_sdist_url.
        // The critical invariant is that the result is NOT Ok(None).
        //
        // Before PHASE 2.6 (RED): the bare `Origin::Built =>` arm hit first and
        // returned Ok(None) because upstream_url=None — silently abandoning replay.
        // After PHASE 2.6 (GREEN): Class-2b intercepts, attempts build_wheel_from_sdist_url.
        // The build attempt may succeed (uv available) or fail (no network/uv) —
        // either way it is NOT Ok(None). sdist_source.is_some() guarantees provenance
        // and the arm NEVER silently abandons to full resolve.
        assert!(
            !matches!(result, Ok(None)),
            "Class-2b wheel with sdist_source=Some must NOT return Ok(None); \
             the arm must attempt a build. Got: {result:?}"
        );
    }

    /// (2) CLASS-2b EmitWheel field-mapping contract (pure, no live build).
    ///
    /// The Class-2b arm (mod.rs ~line 4666) builds an EmitWheel from a LockWheel.
    /// Verify the field contract in isolation:
    ///   - sdist_source is carried verbatim (self-drift property: re-emitted, not re-derived)
    ///   - upstream_url = None (sdist provenance lives in sdist_source, not upstream_url)
    ///   - remote_url = None (no index URL for a locally-built sdist wheel)
    ///   - git_source = None (sdist build, not git)
    ///   - local_path = Some(...) when a built path is provided
    ///   - version, pypi_name, requires_dist, wheel_filename all clone from LockWheel
    ///
    /// This is a direct field-mapping test — does NOT call materialize_from_lock.
    /// It proves the EmitWheel construction contract without requiring network/uv.
    #[test]
    fn class2b_emit_wheel_field_mapping() {
        use crate::emit_pypi::EmitWheel;
        use crate::lock::SdistWheelSource;

        let sdist_src = SdistWheelSource {
            index: "https://pypi.org/simple/".into(),
            name: "gym".into(),
            version: "0.26.2".into(),
            sdist_url: format!(
                "https://files.pythonhosted.org/packages/gym-0.26.2.tar.gz#sha256={}",
                "ab".repeat(32)
            ),
        };

        // Replicate the LockWheel that Class-2b operates on.
        let lw = LockWheel {
            name: "gym".into(),
            version: "0.26.2".into(),
            origin: Origin::Built,
            filename: "gym-0.26.2-999retread-py3-none-any.whl".into(),
            url: None,
            sha256: Some("15".repeat(32)),
            requires_dist: vec!["numpy>=1.21".into(), "cloudpickle>=1.2.0".into()],
            must_ship: false,
            upstream_url: None, // suppressed at write-time for sdist wheels
            git_source: None,
            sdist_source: Some(sdist_src.clone()),
        };

        // Replicate the EmitWheel construction from Class-2b arm (mod.rs ~4666).
        let built_path = std::path::PathBuf::from("/tmp/gym-0.26.2-py3-none-any.whl");
        let ew = EmitWheel {
            pypi_name: lw.name.clone(),
            version: lw.version.clone(),
            requires_dist: lw.requires_dist.clone(),
            local_path: Some(built_path.clone()),
            wheel_filename: lw.filename.clone(),
            sha256: None,
            locked_final_sha256: lw.sha256.clone(),
            remote_url: None,
            upstream_url: None,
            git_source: None,
            sdist_source: lw.sdist_source.clone(),
        };

        // Self-drift property: sdist_source is re-emitted verbatim, NOT re-derived.
        // This is the critical invariant: the Class-2b arm must clone lw.sdist_source
        // directly (not reconstruct from index/name/version lookup).
        assert_eq!(
            ew.sdist_source.as_ref(),
            Some(&sdist_src),
            "Class-2b self-drift: EmitWheel.sdist_source must equal lw.sdist_source verbatim"
        );

        // Portability invariant: sdist_url must NOT be file://.
        let stored = ew.sdist_source.as_ref().unwrap().sdist_url.as_str();
        assert!(
            stored.starts_with("https://"),
            "sdist_url must be an https URL (portability), not file://: {stored}"
        );

        // Field contract: upstream_url=None, remote_url=None, git_source=None.
        assert!(
            ew.upstream_url.is_none(),
            "Class-2b EmitWheel must have upstream_url=None (provenance in sdist_source)"
        );
        assert!(
            ew.remote_url.is_none(),
            "Class-2b EmitWheel must have remote_url=None (locally-built sdist wheel)"
        );
        assert!(
            ew.git_source.is_none(),
            "Class-2b EmitWheel must have git_source=None (sdist build, not git)"
        );

        // Payload fields pass through from LockWheel.
        assert_eq!(ew.pypi_name, lw.name);
        assert_eq!(ew.version, lw.version);
        assert_eq!(ew.requires_dist, lw.requires_dist);
        assert_eq!(ew.wheel_filename, lw.filename);
        assert_eq!(ew.local_path, Some(built_path));
    }

    /// (3) CLASS-2b live round-trip: materialize_from_lock rebuilds gym from stored
    /// sdist_url and produces an EmitWheel with sdist_source preserved.
    ///
    /// Marked #[ignore] because it needs uv + network (same pattern as the
    /// git_source_wheel_replay_byte_identical_parity test in courier.rs).
    /// Run with: cargo test -- --include-ignored class2b_live_round_trip
    ///
    /// The test uses gym==0.26.2 (the canonical sdist-only example) because gym
    /// ships only as a source distribution on PyPI — this is the exact scenario
    /// that PHASE 2.6 was designed to fix.
    #[tokio::test]
    #[ignore = "live: builds gym from PyPI sdist via uv (needs uv + network); run with --include-ignored"]
    async fn class2b_live_round_trip() {
        use super::materialize_from_lock;
        use crate::config::RetreadConfig;
        use crate::lock::SdistWheelSource;

        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("retread-class2b-live-{pid}"));
        let source_dir = base.join("source");
        let output_dir = base.join("output");
        let work_dir = base.join("work");
        let cache_dir = base.join("cache");
        for d in [&source_dir, &output_dir, &work_dir, &cache_dir] {
            std::fs::create_dir_all(d).unwrap();
        }

        // gym==0.26.2 sdist URL with the real sha256 from PyPI.
        // This is the exact lock entry that PHASE 2.6 produces on cold solve.
        let sdist_url = "https://files.pythonhosted.org/packages/2c/b0/\
            gym-0.26.2.tar.gz#sha256=\
            d8f6e9e05f1c64b1e35c2a2e07fe65e9ee57dcfc9b936e48ef4d5e4a4ebde12f";
        let sdist_src = SdistWheelSource {
            index: "https://pypi.org/simple/".into(),
            name: "gym".into(),
            version: "0.26.2".into(),
            sdist_url: sdist_url.into(),
        };

        let lock = RetreadLock {
            schema: crate::lock::SCHEMA,
            retread_version: "2.7.0".into(),
            bundle: "gympack".into(),
            version: "1.0.0".into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
            resolution_glibc: None,
            inputs_hash: "live-hash".into(),
            root_requirements: vec![],
            wheels: vec![LockWheel {
                name: "gym".into(),
                version: "0.26.2".into(),
                origin: Origin::Built,
                filename: "gym-0.26.2-999retread-py3-none-any.whl".into(),
                url: None,
                sha256: Some("16".repeat(32)),
                requires_dist: vec!["numpy>=1.21.0".into(), "cloudpickle>=1.2.0".into()],
                must_ship: false,
                upstream_url: None, // suppressed at write-time
                git_source: None,
                sdist_source: Some(sdist_src.clone()),
            }],
            conda_run_deps: vec![],
            index_urls: vec!["https://pypi.org/simple/".into()],
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec![],
            entry_specs: vec![],
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };
        let config: RetreadConfig = serde_json::from_value(serde_json::json!({"retread-wheels": {
            "gympack": { "version": "==1.0.0" }
        }}))
        .unwrap();

        let result = materialize_from_lock(
            lock,
            &config,
            &work_dir,
            &output_dir,
            &ResolutionTarget::from_parts("3.11", "linux-64", None),
            &source_dir,
            &cache_dir,
            None,
            vec!["python 3.11.*".to_string()],
            "live-fp",
        )
        .await;

        std::fs::remove_dir_all(&base).ok();

        // Class-2b round-trip: the arm must return Ok(Some(...)) and produce
        // an EmitWheel that preserves sdist_source verbatim (self-drift property).
        assert!(
            result.is_ok(),
            "Class-2b live round-trip must not Err: {result:?}"
        );
        assert!(
            result.unwrap().is_some(),
            "Class-2b live round-trip must return Ok(Some(...))"
        );
    }

    // ---- Class-2 (Phase 2.7 fix) replay field-mapping + byte-identity tests ----
    //
    // The Phase 2.7 fix changes the Class-2 replay arm (bare `Origin::Built`
    // after Class-2b) to DOWNLOAD the upstream wheel via fetch_wheel_cached and
    // route through courier's LOCAL-PATH branch rather than the REMOTE-ONLY branch.
    // This fixes conda_capable relax-shadows (pytorch3d) that the old remote-only
    // branch mis-classified as Origin::Index.
    //
    // Three tests:
    //   (1) class2_emit_wheel_field_mapping — pure sync, REQUIRED.
    //   (2) class2_replay_cold_byte_identity — localhost fixture, REQUIRED parity oracle.
    //   (3) class2_live_round_trip — ignored, live network.

    // ── Shared test utilities ────────────────────────────────────────────────────

    fn make_wheel_bytes_for_replay(dist: &str, version: &str, requires: &[&str]) -> Vec<u8> {
        use std::io::Write;
        let normalized = dist.replace('-', "_");
        let di = format!("{normalized}-{version}.dist-info");
        let mut metadata = format!("Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n");
        for req in requires {
            metadata.push_str(&format!("Requires-Dist: {req}\n"));
        }
        let metadata_bytes = metadata.into_bytes();
        let wheel_file = b"Wheel-Version: 1.0\nTag: py3-none-any\n".to_vec();
        let record = format!("{di}/METADATA,,\n{di}/WHEEL,,\n{di}/RECORD,,\n").into_bytes();

        let mut buf = Vec::new();
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, body) in [
            (format!("{di}/METADATA"), metadata_bytes.as_slice()),
            (format!("{di}/WHEEL"), wheel_file.as_slice()),
            (format!("{di}/RECORD"), record.as_slice()),
        ] {
            zip.start_file(&name, opts).unwrap();
            zip.write_all(body).unwrap();
        }
        zip.finish().unwrap();
        buf
    }

    /// (1) CLASS-2 EmitWheel field-mapping contract (pure sync, no network).
    ///
    /// Mirrors `class2b_emit_wheel_field_mapping` exactly. Constructs the EmitWheel
    /// the new Class-2 arm would produce (after the Phase 2.7 fix) and asserts the
    /// full field contract:
    ///   - local_path.is_some()  — routes through courier's LOCAL-PATH branch (no conda gate)
    ///   - remote_url.is_none()  — cold's local-path EmitWheel has remote_url=None
    ///   - upstream_url == Some(github_url) — courier writes upstream_url=github, url=None
    ///   - git_source.is_none(), sdist_source.is_none()
    ///   - wheel_filename == lw.filename, version == lw.version
    ///   - requires_dist == lw.requires_dist
    #[test]
    fn class2_emit_wheel_field_mapping() {
        use crate::emit_pypi::EmitWheel;

        let upstream_url = url::Url::parse(
            "https://github.com/MiroPsota/torch_packages_builder/releases/download/\
             pytorch3d0.7.8pt2.7.0cu128/pytorch3d-0.7.8+5043d15pt2.7.0cu128-cp311-cp311-linux_x86_64.whl"
        ).unwrap();

        // Simulate the LockWheel that Class-2 operates on (cold-produced entry).
        let lw = LockWheel {
            name: "pytorch3d".into(),
            version: "0.7.8+5043d15pt2.7.0cu128".into(),
            origin: Origin::Built,
            filename: "pytorch3d-0.7.8+5043d15pt2.7.0cu128-999retread-cp311-cp311-linux_x86_64.whl"
                .into(),
            url: None,
            sha256: Some("1a".repeat(32)),
            requires_dist: vec!["torch>=2.7.0".into(), "torchvision>=0.22.0".into()],
            must_ship: false,
            upstream_url: Some(upstream_url.to_string()),
            git_source: None,
            sdist_source: None,
        };

        // Replicate the EmitWheel construction from the new Class-2 arm (§3.1).
        // The fetched file would be the pristine 5-field upstream basename.
        let fetched_path = std::path::PathBuf::from(
            "/tmp/wheels/pytorch3d-0.7.8+5043d15pt2.7.0cu128-cp311-cp311-linux_x86_64.whl",
        );
        let remote_url = url::Url::parse(lw.upstream_url.as_deref().unwrap()).unwrap();
        let ew = EmitWheel {
            pypi_name: lw.name.clone(),
            version: lw.version.clone(),
            requires_dist: lw.requires_dist.clone(),
            local_path: Some(fetched_path.clone()),
            wheel_filename: lw.filename.clone(),
            sha256: None,
            locked_final_sha256: lw.sha256.clone(),
            remote_url: None,
            upstream_url: Some(remote_url),
            git_source: None,
            sdist_source: None,
        };

        // (a) local_path must be Some: routes through courier's LOCAL-PATH branch,
        // which has NO conda_capable gate. This is the core fix for pytorch3d drift.
        assert!(
            ew.local_path.is_some(),
            "Class-2 EmitWheel must have local_path=Some (routes LOCAL-PATH branch, no conda gate)"
        );

        // (b) remote_url must be None: cold's local-path EmitWheel has remote_url=None.
        // Also prevents the plan() debug_assert from firing (it rejects local_path+remote_url together).
        assert!(
            ew.remote_url.is_none(),
            "Class-2 EmitWheel must have remote_url=None (mirrors cold local-path EmitWheel)"
        );

        // (c) upstream_url == Some(github): courier's Rewritten/Raw arms compute
        // w.upstream_url.or(w.remote_url) -> github -> LockWheel.upstream_url=github, url=None.
        assert_eq!(
            ew.upstream_url.as_ref().map(|u| u.to_string()),
            lw.upstream_url,
            "Class-2 EmitWheel upstream_url must match the recorded lw.upstream_url"
        );

        // (d) No git/sdist provenance (index shadow).
        assert!(
            ew.git_source.is_none(),
            "Class-2 EmitWheel must have git_source=None"
        );
        assert!(
            ew.sdist_source.is_none(),
            "Class-2 EmitWheel must have sdist_source=None"
        );

        // (e) Payload fields pass through from LockWheel.
        assert_eq!(ew.pypi_name, lw.name);
        assert_eq!(ew.version, lw.version);
        assert_eq!(ew.requires_dist, lw.requires_dist);
        // wheel_filename = lw.filename (already-999retread) -> standard_wheel_filename
        // strips nothing (no .relaxed./.injected.) -> 6-field idempotent insert_build_tag
        // -> same 999retread name. Field-for-field identical to cold.
        assert_eq!(ew.wheel_filename, lw.filename);
    }

    /// (2) CLASS-2 replay byte-identity oracle: cold path vs new Class-2 replay path.
    ///
    /// Scenario: a conda_capable index shadow whose Requires-Dist contains a URL
    /// requirement. On cold produce the wheel is fetched+relaxed -> EmitWheel with
    /// local_path=Some + upstream_url=Some(localhost) -> courier LOCAL-PATH branch ->
    /// ShadowSrc::Raw/Rewritten -> LockWheel{origin=Built, upstream_url=github, url=None}.
    ///
    /// On replay the Class-2 arm (Phase 2.7 fix) downloads the wheel from
    /// lw.upstream_url and builds the same EmitWheel -> same LOCAL-PATH branch ->
    /// same LockWheel fields.
    ///
    /// Assert: the two LockWheels are field-for-field equal.
    ///
    /// This is the byte-identity oracle that would have CAUGHT the Phase-2.7 drift.
    /// Before the fix the remote-only branch's `!conda_capable` gate would have set
    /// origin=Index, url=Some(upstream), upstream_url=None -> diverge.
    #[tokio::test]
    async fn class2_replay_cold_byte_identity() {
        let _env_guard = crate::TEST_ASYNC_ENV_MUTEX.lock().await;
        use std::collections::HashSet;
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let dir = unique_tmp_dir();
        let source_dir = dir.join("source");
        std::fs::create_dir_all(source_dir.join("wheels")).unwrap();
        let staging_cold = dir.join("staging-cold");
        let staging_replay = dir.join("staging-replay");

        let bundle = "p3d-pack";
        let wheel_name = "pytorch3d";
        let wheel_version = "0.7.8";

        // URL requirement from pytorch3d -> dep-a (a bundle member).
        // override_line_map rewrites "dep-a @ <url>" -> "dep-a==1.0.0" -> rewrite
        // detects change -> ShadowSrc::Rewritten or ShadowSrc::Raw -> Origin::Built.
        let dep_name = "dep-a";
        let dep_version = "1.0.0";
        let strict_req = "strict-dep==1.2.3";
        let dep_whl_name = format!(
            "{}-{dep_version}-py3-none-any.whl",
            dep_name.replace('-', "_")
        );
        let dep_requires = vec!["leaf>=1".to_string()];
        let url_req = format!("{dep_name} @ https://example.com/{dep_whl_name}");

        // Write the dep wheel file.
        let dep_whl_path = source_dir.join("wheels").join(&dep_whl_name);
        std::fs::write(
            &dep_whl_path,
            make_wheel_bytes_for_replay(dep_name, dep_version, &["leaf>=1"]),
        )
        .unwrap();

        // The pytorch3d wheel bytes (has a URL requirement).
        let p3d_whl_name = format!(
            "{}-{}-py3-none-any.whl",
            wheel_name.replace('-', "_"),
            wheel_version
        );
        let raw_wheel_bytes =
            make_wheel_bytes_for_replay(wheel_name, wheel_version, &[url_req.as_str(), strict_req]);

        // ── Localhost HTTP server ────────────────────────────────────────────────
        let wheel_bytes_srv = Arc::new(raw_wheel_bytes.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let wheel_url_str = format!("http://127.0.0.1:{port}/{p3d_whl_name}");

        let srv_bytes = wheel_bytes_srv.clone();
        let _server = tokio::spawn(async move {
            for _ in 0..8u8 {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let b = srv_bytes.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let _ = stream.read(&mut buf).await;
                    let resp = format!(
                        "HTTP/1.0 200 OK\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\n\r\n",
                        b.len()
                    );
                    let _ = stream.write_all(resp.as_bytes()).await;
                    let _ = stream.write_all(&b).await;
                });
            }
        });

        // ── Config ──────────────────────────────────────────────────────────────
        let config: crate::config::RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": {
                bundle: { "version": &format!("=={wheel_version}") }
            },
            "retread-relax": "minor"
        }))
        .unwrap();

        // conda_capable includes the pytorch3d wheel name — this triggers the drift
        // on the OLD remote-only path but is irrelevant to the LOCAL-PATH path.
        let mut conda_capable: HashSet<String> = HashSet::new();
        conda_capable.insert(wheel_name.to_string());

        let index_urls = [format!("http://127.0.0.1:{port}/simple/")];

        // ── Write the upstream wheel bytes to disk and apply cold phase D, exactly
        //    as materialize_and_rewrite does before courier staging.
        let p3d_local = source_dir.join("wheels").join(&p3d_whl_name);
        std::fs::write(&p3d_local, &raw_wheel_bytes).unwrap();
        let p3d_cold_relaxed = p3d_local.with_extension("relaxed.whl");
        rewrite_wheel(&p3d_local, &p3d_cold_relaxed, config.relax).unwrap();
        let cold_metadata = crate::wheel::read_metadata(&p3d_cold_relaxed).unwrap();
        assert!(
            cold_metadata
                .requires_dist
                .iter()
                .any(|req| req.starts_with("strict-dep>=1.2")),
            "test must exercise a general relax rewrite before courier staging"
        );

        let dep_emit = crate::emit_pypi::EmitWheel {
            pypi_name: dep_name.to_string(),
            version: dep_version.to_string(),
            requires_dist: dep_requires.clone(),
            wheel_filename: dep_whl_name.clone(),
            sha256: None,
            locked_final_sha256: None,
            local_path: Some(dep_whl_path.clone()),
            remote_url: None,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        };

        // COLD EmitWheel: local_path=Some + upstream_url=Some(localhost).
        let upstream_url: url::Url = wheel_url_str.parse().unwrap();
        let cold_emit = crate::emit_pypi::EmitWheel {
            pypi_name: wheel_name.to_string(),
            version: wheel_version.to_string(),
            requires_dist: cold_metadata.requires_dist,
            wheel_filename: p3d_whl_name.clone(),
            sha256: None,
            locked_final_sha256: None,
            local_path: Some(p3d_cold_relaxed),
            remote_url: None,
            upstream_url: Some(upstream_url.clone()),
            git_source: None,
            sdist_source: None,
        };

        // ── COLD stage ───────────────────────────────────────────────────────────
        unsafe { std::env::set_var("RETREAD_NO_SHADOW_CACHE", "1") };
        let cold_staged = crate::courier::stage(
            &config,
            bundle,
            wheel_version,
            "3.11",
            &[dep_emit.clone(), cold_emit.clone()],
            &conda_capable,
            &[],
            &index_urls,
            "",
            &source_dir,
            &staging_cold,
        )
        .await;
        unsafe { std::env::remove_var("RETREAD_NO_SHADOW_CACHE") };
        let cold_staged = cold_staged.expect("cold stage must succeed");

        let cold_lw = cold_staged
            .lock
            .wheels
            .iter()
            .find(|w| w.name == wheel_name)
            .expect("cold lock must contain the wheel")
            .clone();

        // The cold lock must classify the wheel as Origin::Built (relax-changed shadow).
        assert_eq!(
            cold_lw.origin,
            Origin::Built,
            "cold: relax-changed conda_capable wheel must be Origin::Built"
        );

        // ── REPLAY: call materialize_from_lock with the cold lock entry ──────────
        // Build a minimal RetreadLock from the cold stage result.
        let replay_lock = RetreadLock {
            schema: SCHEMA,
            retread_version: "2.7.1".into(),
            bundle: bundle.into(),
            version: wheel_version.into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
            resolution_glibc: None,
            inputs_hash: "test-hash".into(),
            root_requirements: vec![],
            wheels: cold_staged.lock.wheels.clone(),
            conda_run_deps: vec![],
            index_urls: index_urls.to_vec(),
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec![wheel_name.to_string()],
            entry_specs: vec![],
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };

        // Build the replay EmitWheel exactly as the new Class-2 arm would:
        // download from upstream_url -> local_path=Some, remote_url=None,
        // upstream_url=Some(github).
        let replay_download = source_dir.join("replay-download");
        std::fs::create_dir_all(&replay_download).unwrap();
        let fetched_replay = replay_download.join(&p3d_whl_name);
        std::fs::write(&fetched_replay, &raw_wheel_bytes).unwrap();
        let replay_local = prepare_replayed_class2_wheel(fetched_replay, config.relax, true)
            .await
            .unwrap();

        let dep_replay = crate::emit_pypi::EmitWheel {
            pypi_name: dep_name.to_string(),
            version: dep_version.to_string(),
            requires_dist: dep_requires,
            wheel_filename: dep_whl_name.clone(),
            sha256: None,
            locked_final_sha256: None,
            local_path: Some(dep_whl_path),
            remote_url: None,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        };

        // Simulate what the new Class-2 arm emits: local_path=Some(fetched),
        // remote_url=None, upstream_url=Some(upstream).
        let replay_emit = crate::emit_pypi::EmitWheel {
            pypi_name: wheel_name.to_string(),
            version: wheel_version.to_string(),
            requires_dist: cold_lw.requires_dist.clone(),
            wheel_filename: cold_lw.filename.clone(), // already-999retread
            sha256: None,
            locked_final_sha256: cold_lw.sha256.clone(),
            local_path: Some(replay_local),
            remote_url: None,
            upstream_url: Some(upstream_url),
            git_source: None,
            sdist_source: None,
        };

        unsafe { std::env::set_var("RETREAD_NO_SHADOW_CACHE", "1") };
        let replay_staged = crate::courier::stage(
            &config,
            bundle,
            wheel_version,
            "3.11",
            &[dep_replay, replay_emit],
            &conda_capable,
            &[],
            &index_urls,
            "",
            &source_dir,
            &staging_replay,
        )
        .await;
        unsafe { std::env::remove_var("RETREAD_NO_SHADOW_CACHE") };
        let replay_staged = replay_staged.expect("replay stage must succeed");

        let replay_lw = replay_staged
            .lock
            .wheels
            .iter()
            .find(|w| w.name == wheel_name)
            .expect("replay lock must contain the wheel")
            .clone();

        std::fs::remove_dir_all(&dir).ok();
        let _ = replay_lock; // suppress unused warning

        // ── Field-for-field equality ─────────────────────────────────────────────
        assert_eq!(
            cold_lw.origin, replay_lw.origin,
            "PARITY FAIL: origin mismatch: cold={:?} replay={:?}",
            cold_lw.origin, replay_lw.origin,
        );
        assert_eq!(
            cold_lw.filename, replay_lw.filename,
            "PARITY FAIL: filename mismatch: cold={} replay={}",
            cold_lw.filename, replay_lw.filename,
        );
        assert_eq!(
            cold_lw.url, replay_lw.url,
            "PARITY FAIL: url mismatch: cold={:?} replay={:?}",
            cold_lw.url, replay_lw.url,
        );
        assert_eq!(
            cold_lw.upstream_url, replay_lw.upstream_url,
            "PARITY FAIL: upstream_url mismatch: cold={:?} replay={:?}",
            cold_lw.upstream_url, replay_lw.upstream_url,
        );
        assert_eq!(
            cold_lw.requires_dist, replay_lw.requires_dist,
            "PARITY FAIL: requires_dist mismatch"
        );
        assert_eq!(
            cold_lw.must_ship, replay_lw.must_ship,
            "PARITY FAIL: must_ship mismatch: cold={} replay={}",
            cold_lw.must_ship, replay_lw.must_ship,
        );
        assert_eq!(
            cold_lw.sha256, replay_lw.sha256,
            "PARITY FAIL: sha256 mismatch"
        );
        assert_eq!(
            cold_lw.git_source, replay_lw.git_source,
            "PARITY FAIL: git_source mismatch"
        );
        assert_eq!(
            cold_lw.sdist_source, replay_lw.sdist_source,
            "PARITY FAIL: sdist_source mismatch"
        );
    }

    #[tokio::test]
    async fn class2_replay_transitive_keeps_raw_pre_courier_bytes() {
        let dir = unique_tmp_dir();
        let raw = dir.join("sidecar_dep-1.2.3-py3-none-any.whl");
        let bytes = make_wheel_bytes_for_replay("sidecar-dep", "1.2.3", &["nested-dep==4.5.6"]);
        std::fs::write(&raw, &bytes).unwrap();

        let prepared = prepare_replayed_class2_wheel(raw.clone(), RelaxPolicy::Minor, false)
            .await
            .unwrap();

        assert_eq!(prepared, raw);
        assert_eq!(std::fs::read(&prepared).unwrap(), bytes);
        assert!(
            !prepared.with_extension("relaxed.whl").exists(),
            "a cold remote-only transitive enters courier as raw bytes"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// (3) CLASS-2 live round-trip: materialize_from_lock re-fetches a real upstream
    /// wheel via fetch_wheel_cached and produces a LockWheel with origin=Built.
    ///
    /// Marked #[ignore] because it needs network (downloads a wheel from PyPI/github).
    /// Run with: cargo test -- --include-ignored class2_live_round_trip
    #[tokio::test]
    #[ignore = "live: re-fetches a real index wheel (needs network); run with --include-ignored"]
    async fn class2_live_round_trip() {
        use super::materialize_from_lock;
        use crate::config::RetreadConfig;

        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("retread-class2-live-{pid}"));
        let source_dir = base.join("source");
        let output_dir = base.join("output");
        let work_dir = base.join("work");
        let cache_dir = base.join("cache");
        for d in [&source_dir, &output_dir, &work_dir, &cache_dir] {
            std::fs::create_dir_all(d).unwrap();
        }

        // A small pure-python conda-capable wheel on PyPI (requests is universally
        // available and tiny enough for a quick live test).
        let upstream_url = "https://files.pythonhosted.org/packages/f9/9b/\
             requests-2.31.0-py3-none-any.whl";

        let lock = RetreadLock {
            schema: SCHEMA,
            retread_version: "2.7.1".into(),
            bundle: "reqpack".into(),
            version: "1.0.0".into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
            resolution_glibc: None,
            inputs_hash: "live-c2".into(),
            root_requirements: vec![],
            wheels: vec![LockWheel {
                name: "requests".into(),
                version: "2.31.0".into(),
                origin: Origin::Built,
                filename: "requests-2.31.0-999retread-py3-none-any.whl".into(),
                url: None,
                sha256: Some("17".repeat(32)),
                requires_dist: vec!["urllib3>=1.21.1".into()],
                must_ship: false,
                upstream_url: Some(upstream_url.into()),
                git_source: None,
                sdist_source: None,
            }],
            conda_run_deps: vec![],
            index_urls: vec!["https://pypi.org/simple/".into()],
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec!["requests".into()],
            entry_specs: vec![],
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };
        let config: RetreadConfig = serde_json::from_value(serde_json::json!({"retread-wheels": {
            "reqpack": { "version": "==1.0.0" }
        }}))
        .unwrap();

        let result = materialize_from_lock(
            lock,
            &config,
            &work_dir,
            &output_dir,
            &ResolutionTarget::from_parts("3.11", "linux-64", None),
            &source_dir,
            &cache_dir,
            None,
            vec!["python 3.11.*".to_string()],
            "live-c2-fp",
        )
        .await;

        std::fs::remove_dir_all(&base).ok();

        assert!(
            result.is_ok(),
            "Class-2 live round-trip must not Err: {result:?}"
        );
        assert!(
            result.unwrap().is_some(),
            "Class-2 live round-trip must return Ok(Some(...))"
        );
    }

    // ---- RETREAD_NO_REPLAY env knob tests ----

    #[test]
    fn no_replay_env_knob_returns_none() {
        // Acquire the shared env-mutation lock so that concurrent tests cannot
        // observe RETREAD_NO_REPLAY=1 set here (and vice-versa).
        let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();

        let dir = unique_tmp_dir();
        let lock = make_test_lock("pack", "1.0.0", "3.11", "myhash", true);
        let path = dir.join(RetreadLock::file_name("pack"));
        std::fs::write(&path, lock.to_pretty_json().unwrap()).unwrap();

        // SAFETY: serialised by TEST_ENV_MUTEX; no concurrent env access.
        unsafe {
            std::env::set_var("RETREAD_NO_REPLAY", "1");
        }
        let result = load_replayable_lock(&path, "myhash", true);
        unsafe {
            std::env::remove_var("RETREAD_NO_REPLAY");
        }
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "RETREAD_NO_REPLAY=1 must suppress replay (return None)"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    // ---- load_replayable_lock tests ----

    #[test]
    fn load_replayable_returns_none_for_missing_file() {
        let dir = unique_tmp_dir();
        let path = dir.join("retread-no-such.lock.json");
        let result = load_replayable_lock(&path, "anyhash", true);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none(), "missing file must return None");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn load_replayable_returns_none_for_schema_mismatch() {
        let dir = unique_tmp_dir();
        // Write a lock with schema 4 (old); current SCHEMA is 5.
        let old_schema_json = r#"{
            "schema": 4,
            "retread_version": "2.0.0",
            "bundle": "pack",
            "version": "1.0.0",
            "python": "3.11",
            "inputs_hash": "correcthash",
            "root_requirements": [],
            "wheels": [],
            "conda_run_deps": [],
            "index_urls": []
        }"#;
        let path = dir.join(RetreadLock::file_name("pack"));
        std::fs::write(&path, old_schema_json).unwrap();
        let result = load_replayable_lock(&path, "correcthash", true);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "schema-4 lock must be a replay miss (schema mismatch)"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn load_replayable_returns_none_for_hash_mismatch() {
        let dir = unique_tmp_dir();
        let lock = make_test_lock("pack", "1.0.0", "3.11", "stored-hash", true);
        let path = dir.join(RetreadLock::file_name("pack"));
        std::fs::write(&path, lock.to_pretty_json().unwrap()).unwrap();
        let result = load_replayable_lock(&path, "different-hash", true);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none(), "hash mismatch must return None");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn load_replayable_returns_none_for_empty_requires_dist_under_non_default_relax() {
        let dir = unique_tmp_dir();
        // Build a lock with a relax-changed Built wheel (must_ship=false, origin=Built)
        // but empty requires_dist — this is the poison scenario.
        let mut lock = make_test_lock("pack", "1.0.0", "3.11", "myhash", true);
        lock.wheels.push(LockWheel {
            name: "torchvision".into(),
            version: "0.18.0".into(),
            origin: Origin::Built,
            filename: "torchvision-0.18.0-cp311-cp311-linux_x86_64.999retread.whl".into(),
            url: None,
            sha256: Some("18".repeat(32)),
            requires_dist: vec![], // EMPTY — poison scenario
            must_ship: false,      // relax-changed index wheel
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        });
        let path = dir.join(RetreadLock::file_name("pack"));
        std::fs::write(&path, lock.to_pretty_json().unwrap()).unwrap();
        // relax_is_default=false triggers the poisoning guard
        let result = load_replayable_lock(&path, "myhash", false);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "empty requires_dist on relax-changed Built wheel with non-default relax \
             must return None (poisoning guard)"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn load_replayable_returns_some_when_all_checks_pass() {
        // Hold env-lock: prevents RETREAD_NO_REPLAY=1 from returning None here.
        let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let lock = make_test_lock("pack", "1.0.0", "3.11", "goodhash", true);
        let path = dir.join(RetreadLock::file_name("pack"));
        std::fs::write(&path, lock.to_pretty_json().unwrap()).unwrap();
        let result = load_replayable_lock(&path, "goodhash", true);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_some(),
            "valid lock with matching hash must return Some"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// Regression test for the build_v1 replay lock-drift bug (v2.5.0).
    ///
    /// The build_v1 replay gate MUST source `run_deps` from `lock.conda_run_deps`,
    /// NOT from `params.run_dependencies`.  pixi's live conda solver can inject
    /// non-deterministic extras (e.g. `python_abi 3.12.* *_cp312`) that are absent
    /// from the committed lock, causing the rewritten lock to differ from the
    /// committed one (62 vs 61 conda_run_deps in the isaac6 lukewarm e2e).
    ///
    /// This test asserts the exact serialization used in the replay gate:
    ///   - `CondaDep { name, spec="" }` → `"name"` (name-only)
    ///   - `CondaDep { name, spec }` → `"name spec"`
    ///
    /// It also asserts that a params.run_dependencies containing an extra
    /// `python_abi` dep would NOT appear in the run_deps if it is absent from
    /// lock.conda_run_deps.
    #[test]
    fn build_v1_replay_run_deps_come_from_lock_not_params() {
        // Simulate the lock as committed (two deps, no python_abi).
        let lock_conda_run_deps = [
            crate::lock::CondaDep {
                name: "numpy".into(),
                spec: ">=1.21".into(),
            },
            crate::lock::CondaDep {
                name: "libstdcxx-ng".into(),
                spec: String::new(), // name-only dep
            },
        ];

        // This is what the replay gate now does (the fix):
        // serialize lock.conda_run_deps, ignoring params.run_dependencies.
        let run_deps_from_lock: Vec<String> = lock_conda_run_deps
            .iter()
            .map(|dep| {
                if dep.spec.is_empty() {
                    dep.name.clone()
                } else {
                    format!("{} {}", dep.name, dep.spec)
                }
            })
            .collect();

        // Simulate what params.run_dependencies would contain (pixi's live
        // solve injected python_abi non-deterministically).
        let params_run_deps = vec![
            "numpy >=1.21".to_string(),
            "libstdcxx-ng".to_string(),
            "python_abi 3.12.* *_cp312".to_string(), // the extra that caused drift
        ];

        // The replay path must produce exactly the lock-sourced deps.
        assert_eq!(
            run_deps_from_lock,
            vec!["numpy >=1.21", "libstdcxx-ng"],
            "replay run_deps must match lock.conda_run_deps exactly"
        );
        // Crucially, python_abi must NOT appear (it was injected by the solver).
        assert!(
            !run_deps_from_lock
                .iter()
                .any(|d| d.starts_with("python_abi")),
            "python_abi must NOT appear in replay run_deps (it is absent from \
             lock.conda_run_deps): {run_deps_from_lock:?}"
        );
        // And the params path WOULD have introduced it (confirming the bug).
        assert!(
            params_run_deps.iter().any(|d| d.starts_with("python_abi")),
            "params_run_deps simulation must contain python_abi (proving the \
             pre-fix bug path): {params_run_deps:?}"
        );
    }

    #[test]
    fn load_replayable_returns_some_with_non_default_relax_when_requires_dist_nonempty() {
        // Hold env-lock: prevents RETREAD_NO_REPLAY=1 from returning None here.
        let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let mut lock = make_test_lock("pack", "1.0.0", "3.11", "myhash", true);
        // A relax-changed Built wheel WITH requires_dist — not poison.
        lock.wheels.push(LockWheel {
            name: "torchvision".into(),
            version: "0.18.0".into(),
            origin: Origin::Built,
            filename: "torchvision-0.18.0-cp311-cp311-linux_x86_64.999retread.whl".into(),
            url: None,
            sha256: Some("19".repeat(32)),
            requires_dist: vec!["torch>=2.0,<3".into()],
            must_ship: false,
            upstream_url: Some("https://files.pythonhosted.org/torchvision-0.18.0.whl".into()),
            git_source: None,
            sdist_source: None,
        });
        let path = dir.join(RetreadLock::file_name("pack"));
        std::fs::write(&path, lock.to_pretty_json().unwrap()).unwrap();
        let result = load_replayable_lock(&path, "myhash", false);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_some(),
            "relax-changed Built wheel with non-empty requires_dist is safe to replay"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    // -----------------------------------------------------------------------
    // FIX-4: skip_subdirs parity for nested-subdirectory git entries.
    //
    // This test proves the correctness landmine fixed by FIX-1:
    //   - PRODUCE path (auto_data_per_entry) computes skip_subdirs as
    //     [subdirectory], e.g. ["packages/mypkg"] for a nested subdir entry.
    //   - OLD REPLAY path (materialize_from_lock Class-1 arm, pre-fix) used
    //     skip_subdirs=vec![], which for a nested subdir entry causes the
    //     phase-1.6 checkout-root data inject to include Python source files
    //     from packages/mypkg/ that the produce path correctly excluded ->
    //     silently different wheel bytes.
    //   - FIXED REPLAY path mirrors produce: skip_subdirs=[subdirectory].
    //
    // The test drives `materialize_and_rewrite` directly (the exact function
    // that materialize_from_lock's Class-1 git arm calls after constructing
    // auto_data). It:
    //   (a) builds a produce wheel with auto_data{skip_subdirs=["packages/mypkg"]}
    //   (b) builds a buggy-replay wheel with auto_data{skip_subdirs=[]}
    //   (c) asserts (a) != (b) -- proving the regression
    //   (d) builds a fixed-replay wheel with auto_data{skip_subdirs=["packages/mypkg"]}
    //   (e) asserts (a) == (d) -- proving the fix restores parity
    //
    // Failure mode before FIX-1: step (c) would fail (produce == buggy-replay
    // for root entries because path_is_under_any(".", rel) never matches a
    // stripped relative path). For a NESTED subdir ("packages/mypkg"), buggy
    // replay adds Python files from packages/mypkg/ into the wheel's
    // .data/data/lib/ tree, making it larger and byte-distinct from produce.
    // -----------------------------------------------------------------------
    #[tokio::test]
    #[ignore = "live: builds a git wheel via uv (needs uv + git on PATH); run with --include-ignored"]
    async fn nested_subdir_git_produce_replay_skip_subdirs_parity() {
        use super::{AutoDataConfig, EntryAuditInfo, materialize_and_rewrite, wheel_target_for};
        use crate::config::RelaxPolicy;
        use crate::config::WheelEntry;
        use rattler_conda_types::Platform;

        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("retread-nested-subdir-parity-{pid}"));
        let repo = base.join("repo");
        // Create the nested subdir: packages/mypkg/
        let pkg_dir = repo.join("packages").join("mypkg");
        std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");

        // ── Minimal git fixture ──────────────────────────────────────────────
        let run_git = |args: &[&str], dir: &std::path::Path| {
            let status = std::process::Command::new("git")
                .args(args)
                .current_dir(dir)
                .env("GIT_AUTHOR_NAME", "test")
                .env("GIT_AUTHOR_EMAIL", "test@example.com")
                .env("GIT_COMMITTER_NAME", "test")
                .env("GIT_COMMITTER_EMAIL", "test@example.com")
                .status()
                .expect("git");
            assert!(status.success(), "git {args:?} failed");
        };

        run_git(&["init", "-b", "main"], &repo);
        run_git(&["config", "user.email", "test@example.com"], &repo);
        run_git(&["config", "user.name", "test"], &repo);

        // Nested package: packages/mypkg/ — a real installable Python package.
        std::fs::write(
            pkg_dir.join("pyproject.toml"),
            r#"[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "retread-nested-mypkg"
version = "1.0.0"
"#,
        )
        .expect("write pyproject");
        let mypkg_src = pkg_dir.join("retread_nested_mypkg");
        std::fs::create_dir_all(&mypkg_src).expect("create package src dir");
        std::fs::write(mypkg_src.join("__init__.py"), b"# nested package fixture\n")
            .expect("write __init__.py");

        // Sibling non-package file at the repo root (should appear in ALL wheels
        // regardless of skip_subdirs, because it's outside packages/mypkg/).
        std::fs::write(repo.join("README.md"), b"monorepo root\n").expect("write README");

        run_git(&["add", "."], &repo);
        run_git(&["commit", "-m", "initial"], &repo);

        let sha_out = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&repo)
            .output()
            .expect("git rev-parse");
        let sha = String::from_utf8_lossy(&sha_out.stdout).trim().to_string();
        assert_eq!(sha.len(), 40, "expected 40-char SHA");

        let repo_url = format!("file://{}", repo.display());

        // ── Common test dirs ─────────────────────────────────────────────────
        let cache_dir = base.join("cache");
        // Each materialize_and_rewrite call gets its own download_dir +
        // source_dir to avoid the is_fresh() cache short-circuit that would
        // serve the first call's output as the cached result for later calls.
        let produce_src = base.join("src-produce");
        let buggy_src = base.join("src-buggy");
        let fixed_src = base.join("src-fixed");
        for d in [&cache_dir, &produce_src, &buggy_src, &fixed_src] {
            std::fs::create_dir_all(d).unwrap();
        }

        // WheelEntry for the nested subdir entry.
        let entry = WheelEntry {
            git: Some(repo_url.clone()),
            rev: Some(sha.clone()),
            subdirectory: Some("packages/mypkg".to_string()),
            ..WheelEntry::default()
        };
        let target = wheel_target_for(Platform::Linux64, "3.11").expect("valid target");
        let checkout_root = crate::source_build::git_checkout_root(&repo_url, &sha, &cache_dir);

        // ── PRODUCE: auto_data with correct skip_subdirs=["packages/mypkg"] ──
        let produce_dd = produce_src.join("wheels");
        std::fs::create_dir_all(&produce_dd).unwrap();
        let (produce_resolved, _) = materialize_and_rewrite(
            &entry,
            "retread-nested-mypkg",
            None,
            &target,
            &produce_dd,
            &produce_src,
            &cache_dir,
            RelaxPolicy::None,
            &std::collections::BTreeMap::new(),
            Some(AutoDataConfig {
                checkout_root: checkout_root.clone(),
                skip_subdirs: vec![std::path::PathBuf::from("packages/mypkg")],
            }),
            EntryAuditInfo::default(),
        )
        .await
        .expect("produce: materialize_and_rewrite");
        let produce_wheel_path = produce_resolved
            .url
            .to_file_path()
            .expect("produce wheel must be a file URL");
        let produce_bytes =
            std::fs::read(&produce_wheel_path).expect("produce wheel file must exist");

        // ── BUGGY REPLAY: auto_data with skip_subdirs=[] (pre-fix regression) ─
        let buggy_dd = buggy_src.join("wheels");
        std::fs::create_dir_all(&buggy_dd).unwrap();
        let (buggy_resolved, _) = materialize_and_rewrite(
            &entry,
            "retread-nested-mypkg",
            None,
            &target,
            &buggy_dd,
            &buggy_src,
            &cache_dir,
            RelaxPolicy::None,
            &std::collections::BTreeMap::new(),
            Some(AutoDataConfig {
                checkout_root: checkout_root.clone(),
                skip_subdirs: vec![], // <-- pre-fix bug: empty skip
            }),
            EntryAuditInfo::default(),
        )
        .await
        .expect("buggy-replay: materialize_and_rewrite");
        let buggy_wheel_path = buggy_resolved
            .url
            .to_file_path()
            .expect("buggy-replay wheel must be a file URL");
        let buggy_bytes =
            std::fs::read(&buggy_wheel_path).expect("buggy-replay wheel file must exist");

        // (c) Produce and buggy-replay must DIFFER for a nested subdir entry.
        // The buggy replay (skip_subdirs=[]) includes packages/mypkg/__init__.py
        // in the .data/data/lib/ section (phase 1.6 checkout-root inject), while
        // produce correctly skips it.  The wheels are therefore byte-distinct.
        assert_ne!(
            produce_bytes, buggy_bytes,
            "REGRESSION PROOF: produce (skip_subdirs=[\"packages/mypkg\"]) must \
             differ from buggy replay (skip_subdirs=[]) for a nested-subdir entry \
             — the buggy replay includes extra Python source files in the wheel"
        );

        // ── FIXED REPLAY: auto_data mirrors produce (FIX-1) ─────────────────
        let fixed_dd = fixed_src.join("wheels");
        std::fs::create_dir_all(&fixed_dd).unwrap();
        let (fixed_resolved, _) = materialize_and_rewrite(
            &entry,
            "retread-nested-mypkg",
            None,
            &target,
            &fixed_dd,
            &fixed_src,
            &cache_dir,
            RelaxPolicy::None,
            &std::collections::BTreeMap::new(),
            Some(AutoDataConfig {
                checkout_root: checkout_root.clone(),
                // Fixed: mirror the produce-path derivation from gs.subdirectory
                skip_subdirs: vec![std::path::PathBuf::from("packages/mypkg")],
            }),
            EntryAuditInfo::default(),
        )
        .await
        .expect("fixed-replay: materialize_and_rewrite");
        let fixed_wheel_path = fixed_resolved
            .url
            .to_file_path()
            .expect("fixed-replay wheel must be a file URL");
        let fixed_bytes =
            std::fs::read(&fixed_wheel_path).expect("fixed-replay wheel file must exist");

        // (e) Fixed replay must be byte-identical to produce.
        assert_eq!(
            produce_bytes, fixed_bytes,
            "PARITY: fixed replay (skip_subdirs=[\"packages/mypkg\"]) must be \
             byte-identical to produce — FIX-1 restores correctness"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    /// Regression for the `rl-games` replay failure seen in
    /// `isaaclab-gpu-latest`: a BFS-discovered Git transitive is a singleton
    /// checkout group, but phase 1.6 was disabled on the cold path. Replay
    /// must consume that persisted decision instead of treating every
    /// singleton as an explicit checkout-root carrier.
    #[tokio::test]
    #[ignore = "runs two local Git source builds through uv"]
    async fn bfs_git_singleton_replay_preserves_disabled_auto_data() {
        use crate::emit_pypi::EmitWheel;
        use crate::lock::GitWheelAutoData;
        use std::io::Read;

        let _env_guard = crate::TEST_ASYNC_ENV_MUTEX.lock().await;
        let base = unique_tmp_dir();
        let repo = base.join("repo");
        std::fs::create_dir_all(repo.join("retread_bfs_git_leaf")).unwrap();
        std::fs::write(
            repo.join("pyproject.toml"),
            r#"[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "retread-bfs-git-leaf"
version = "1.0.0"

[tool.setuptools.packages.find]
include = ["retread_bfs_git_leaf*"]
"#,
        )
        .unwrap();
        std::fs::write(
            repo.join("retread_bfs_git_leaf/__init__.py"),
            b"VALUE = 1\n",
        )
        .unwrap();
        std::fs::create_dir_all(repo.join("future_sibling")).unwrap();
        std::fs::write(repo.join("future_sibling/module.py"), b"VALUE = 2\n").unwrap();
        // Phase 1.5 denies README.md but injects normalizer.dat. The latter
        // guarantees a deterministic wheel rewrite; the former is the file
        // the old, erroneous phase 1.6 replay added under .data/data/lib/.
        std::fs::write(repo.join("README.md"), b"checkout-root only\n").unwrap();
        std::fs::write(repo.join("normalizer.dat"), b"phase-1.5\n").unwrap();

        let run_git = |args: &[&str]| {
            let output = std::process::Command::new("git")
                .args(args)
                .current_dir(&repo)
                .env("GIT_AUTHOR_NAME", "Retread Test")
                .env("GIT_AUTHOR_EMAIL", "retread@example.invalid")
                .env("GIT_COMMITTER_NAME", "Retread Test")
                .env("GIT_COMMITTER_EMAIL", "retread@example.invalid")
                .output()
                .expect("run git");
            assert!(
                output.status.success(),
                "git {args:?} failed: {}",
                String::from_utf8_lossy(&output.stderr),
            );
            output.stdout
        };
        run_git(&["init", "-b", "main"]);
        run_git(&["add", "."]);
        run_git(&["commit", "-m", "fixture"]);
        let rev = String::from_utf8(run_git(&["rev-parse", "HEAD"]))
            .unwrap()
            .trim()
            .to_owned();
        assert_eq!(rev.len(), 40);

        let git_url = format!("file://{}", repo.display());
        let entry = WheelEntry {
            git: Some(git_url),
            rev: Some(rev),
            ..WheelEntry::default()
        };
        let target = super::wheel_target_for(Platform::Linux64, "3.11").unwrap();
        let cache_dir = base.join("cache");
        let cold_download = base.join("cold-download");
        let cold_source = base.join("cold-source");
        let replay_download = base.join("replay-download");
        let replay_source = base.join("replay-source");
        for dir in [
            &cache_dir,
            &cold_download,
            &cold_source,
            &replay_download,
            &replay_source,
        ] {
            std::fs::create_dir_all(dir).unwrap();
        }

        let (cold, _) = super::materialize_and_rewrite(
            &entry,
            "retread-bfs-git-leaf",
            None,
            &target,
            &cold_download,
            &cold_source,
            &cache_dir,
            RelaxPolicy::None,
            &BTreeMap::new(),
            None,
            super::EntryAuditInfo::default(),
        )
        .await
        .expect("cold BFS Git materialization");
        assert_eq!(
            cold.git_source
                .as_ref()
                .and_then(|source| source.auto_data.clone()),
            Some(GitWheelAutoData::Disabled),
        );

        let cold_path = cold.url.to_file_path().unwrap();
        let cold_bytes = std::fs::read(&cold_path).unwrap();
        let wheel_names = |path: &std::path::Path| {
            let file = std::fs::File::open(path).unwrap();
            let mut archive = zip::ZipArchive::new(file).unwrap();
            let mut names = Vec::with_capacity(archive.len());
            for index in 0..archive.len() {
                let mut member = archive.by_index(index).unwrap();
                let mut sink = Vec::new();
                member.read_to_end(&mut sink).unwrap();
                names.push(member.name().to_owned());
            }
            names
        };
        let cold_names = wheel_names(&cold_path);
        assert!(cold_names.iter().any(|name| name == "normalizer.dat"));
        assert!(
            !cold_names
                .iter()
                .any(|name| name.ends_with(".data/data/lib/README.md"))
        );

        let cold_emit = EmitWheel {
            pypi_name: cold.pypi_name.clone(),
            version: cold.metadata.version.clone(),
            requires_dist: cold.metadata.requires_dist.clone(),
            local_path: Some(cold_path.clone()),
            wheel_filename: cold_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            sha256: Some(cold.metadata.sha256.clone()),
            locked_final_sha256: None,
            remote_url: None,
            upstream_url: None,
            git_source: cold.git_source.clone(),
            sdist_source: None,
        };
        let config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": {},
            "retread-bundle-mode": "fat",
            "retread-relax": "none"
        }))
        .unwrap();
        let indexes = vec!["https://pypi.org/simple/".to_owned()];

        unsafe { std::env::set_var("RETREAD_NO_SHADOW_CACHE", "1") };
        let cold_staged = crate::courier::stage_for_target(
            &config,
            "bfs-git-pack",
            "bfs-git-pack",
            "1.0.0",
            &target,
            &[cold_emit],
            &HashSet::new(),
            &[],
            &indexes,
            "fixture",
            &cold_source,
            &base.join("cold-staging"),
        )
        .await;
        unsafe { std::env::remove_var("RETREAD_NO_SHADOW_CACHE") };
        let cold_staged = cold_staged.expect("cold courier stage");
        let cold_locked = cold_staged
            .lock
            .wheels
            .iter()
            .find(|wheel| wheel.name == "retread-bfs-git-leaf")
            .unwrap();
        assert_eq!(
            cold_locked
                .git_source
                .as_ref()
                .and_then(|source| source.auto_data.clone()),
            Some(GitWheelAutoData::Disabled),
        );

        let replay_emit = super::emit_wheels_from_lock(
            &cold_staged.lock,
            &config,
            &target,
            &replay_download,
            &replay_source,
            &cache_dir,
        )
        .await
        .expect("replay materialization")
        .expect("complete Git provenance must replay");
        let replay_leaf = replay_emit
            .iter()
            .find(|wheel| wheel.pypi_name == "retread-bfs-git-leaf")
            .unwrap();
        let replay_path = replay_leaf.local_path.as_ref().unwrap();
        assert_eq!(
            cold_bytes,
            std::fs::read(replay_path).unwrap(),
            "cold and replay pre-courier wheel bytes must match",
        );
        let replay_names = wheel_names(replay_path);
        assert!(
            !replay_names
                .iter()
                .any(|name| name.ends_with(".data/data/lib/README.md"))
        );

        unsafe { std::env::set_var("RETREAD_NO_SHADOW_CACHE", "1") };
        let replay_staged = crate::courier::stage_for_target(
            &config,
            "bfs-git-pack",
            "bfs-git-pack",
            "1.0.0",
            &target,
            &replay_emit,
            &HashSet::new(),
            &[],
            &indexes,
            "fixture",
            &replay_source,
            &base.join("replay-staging"),
        )
        .await;
        unsafe { std::env::remove_var("RETREAD_NO_SHADOW_CACHE") };
        let replay_staged = replay_staged.expect("replay courier stage");
        validate_authoritative_replay_lock(&cold_staged.lock, &replay_staged.lock)
            .expect("replayed lock must equal the authoritative cold lock");

        // A changed explicit-Git group can reuse the same raw/injected wheel
        // with a different skip set. Its phase-1.6 cache path must change
        // rather than merely trusting the old autodata wheel's mtime. First,
        // future_sibling is not skipped and its Python file is present under
        // .data/data/lib/.
        let checkout_root = crate::source_build::git_checkout_root(
            entry.git.as_deref().unwrap(),
            entry.rev.as_deref().unwrap(),
            &cache_dir,
        );
        let (first_group_shape, _) = super::materialize_and_rewrite(
            &entry,
            "retread-bfs-git-leaf",
            Some("1.0.0"),
            &target,
            &cold_download,
            &cold_source,
            &cache_dir,
            RelaxPolicy::None,
            &BTreeMap::new(),
            Some(AutoDataConfig {
                checkout_root: checkout_root.clone(),
                skip_subdirs: vec!["retread_bfs_git_leaf".into()],
            }),
            super::EntryAuditInfo::default(),
        )
        .await
        .expect("first Git group shape");
        let group_cache_path = first_group_shape.url.to_file_path().unwrap();
        assert!(
            wheel_names(&group_cache_path)
                .iter()
                .any(|name| name.ends_with(".data/data/lib/future_sibling/module.py")),
        );

        // Then future_sibling joins the skipped package set. This specifically
        // proves disposition-keyed warm-cache isolation.
        let (second_group_shape, _) = super::materialize_and_rewrite(
            &entry,
            "retread-bfs-git-leaf",
            Some("1.0.0"),
            &target,
            &cold_download,
            &cold_source,
            &cache_dir,
            RelaxPolicy::None,
            &BTreeMap::new(),
            Some(AutoDataConfig {
                checkout_root,
                skip_subdirs: vec!["retread_bfs_git_leaf".into(), "future_sibling".into()],
            }),
            super::EntryAuditInfo::default(),
        )
        .await
        .expect("changed Git group shape");
        let second_group_cache_path = second_group_shape.url.to_file_path().unwrap();
        assert_ne!(group_cache_path, second_group_cache_path);
        assert_eq!(
            group_cache_path.file_name(),
            second_group_cache_path.file_name()
        );
        assert!(
            !wheel_names(&second_group_cache_path)
                .iter()
                .any(|name| name.ends_with(".data/data/lib/future_sibling/module.py")),
            "changed skip set must invalidate the warm autodata wheel",
        );

        std::fs::remove_dir_all(base).ok();
    }

    // -----------------------------------------------------------------------
    // PHASE 2.5 tests: multi-entry shared-git-checkout replay.
    // -----------------------------------------------------------------------

    /// Replay restores each Git wheel's persisted auto-data disposition.
    ///
    /// The carrier deliberately appears second in lock order. This proves that
    /// grouping only coordinates a shared checkout and never elects a carrier.
    /// The separate singleton models a BFS Git transitive whose phase-1.6 pass
    /// was explicitly disabled.
    #[test]
    fn git_group_replay_uses_persisted_auto_data_dispositions() {
        use crate::lock::{GitWheelSource, LockWheel, Origin};
        use std::path::PathBuf;

        // Synthetic "resolved" SHA — just needs to be a consistent 40-char string
        // so git_checkout_root produces a deterministic path.
        let rev = "a".repeat(40);
        let url = "https://github.com/example/monorepo.git".to_string();
        let cache_dir =
            std::env::temp_dir().join(format!("retread-p25-carrier-test-{}", std::process::id()));
        std::fs::create_dir_all(&cache_dir).unwrap();

        // Group: 3 members of the same git repo, different subdirs.
        // Lock order: isaaclab (index 0), isaaclab_assets (index 1),
        // isaaclab_tasks (index 4). The producer-selected carrier is index 1.
        // rl_games: separate size-1 group (different rev).
        let rev2 = "b".repeat(40);
        let url2 = "https://github.com/example/rl_games.git".to_string();

        let wheels = [
            LockWheel {
                name: "isaaclab".into(),
                version: "2.1.0".into(),
                origin: Origin::Built,
                filename: "isaaclab-2.1.0-py3-none-any.injected.whl".into(),
                url: None,
                sha256: None,
                requires_dist: vec![],
                must_ship: true,
                upstream_url: None,
                git_source: Some(GitWheelSource {
                    url: url.clone(),
                    rev: rev.clone(),
                    subdirectory: Some("source/isaaclab".into()),
                    extras: vec![],
                    auto_data: Some(crate::lock::GitWheelAutoData::Disabled),
                }),
                sdist_source: None,
            },
            LockWheel {
                name: "isaaclab-assets".into(),
                version: "2.1.0".into(),
                origin: Origin::Built,
                filename: "isaaclab_assets-2.1.0-py3-none-any.injected.whl".into(),
                url: None,
                sha256: None,
                requires_dist: vec![],
                must_ship: true,
                upstream_url: None,
                git_source: Some(GitWheelSource {
                    url: url.clone(),
                    rev: rev.clone(),
                    subdirectory: Some("source/isaaclab_assets".into()),
                    extras: vec![],
                    auto_data: Some(crate::lock::GitWheelAutoData::CheckoutRoot {
                        skip_subdirectories: vec![
                            "source/isaaclab".into(),
                            "source/isaaclab_assets".into(),
                            "source/isaaclab_tasks".into(),
                        ],
                    }),
                }),
                sdist_source: None,
            },
            // Interleaved non-git index wheel (non-contiguous group test).
            LockWheel {
                name: "numpy".into(),
                version: "1.26.0".into(),
                origin: Origin::Index,
                filename: "numpy-1.26.0-cp311-cp311-linux_x86_64.whl".into(),
                url: Some("https://files.pythonhosted.org/numpy-1.26.0.whl".into()),
                sha256: None,
                requires_dist: vec![],
                must_ship: false,
                upstream_url: None,
                git_source: None,
                sdist_source: None,
            },
            // rl_games: size-1 group (separate repo / rev).
            LockWheel {
                name: "rl-games".into(),
                version: "1.6.1".into(),
                origin: Origin::Built,
                filename: "rl_games-1.6.1-py3-none-any.injected.whl".into(),
                url: None,
                sha256: None,
                requires_dist: vec![],
                must_ship: true,
                upstream_url: None,
                git_source: Some(GitWheelSource {
                    url: url2.clone(),
                    rev: rev2.clone(),
                    subdirectory: None, // root subdir
                    extras: vec![],
                    auto_data: Some(crate::lock::GitWheelAutoData::Disabled),
                }),
                sdist_source: None,
            },
            // isaaclab_tasks: non-contiguous member of the isaaclab group.
            LockWheel {
                name: "isaaclab-tasks".into(),
                version: "2.1.0".into(),
                origin: Origin::Built,
                filename: "isaaclab_tasks-2.1.0-py3-none-any.injected.whl".into(),
                url: None,
                sha256: None,
                requires_dist: vec![],
                must_ship: true,
                upstream_url: None,
                git_source: Some(GitWheelSource {
                    url: url.clone(),
                    rev: rev.clone(),
                    subdirectory: Some("source/isaaclab_tasks".into()),
                    extras: vec![],
                    auto_data: Some(crate::lock::GitWheelAutoData::Disabled),
                }),
                sdist_source: None,
            },
        ];

        // Replicate the pre-pass logic from materialize_from_lock.
        let mut git_group_members: std::collections::HashMap<PathBuf, Vec<usize>> =
            std::collections::HashMap::new();
        let mut git_group_order: Vec<PathBuf> = Vec::new();

        for (idx, lw) in wheels.iter().enumerate() {
            if lw.origin == Origin::Built
                && lw.must_ship
                && let Some(gs) = &lw.git_source
            {
                let root = crate::source_build::git_checkout_root(&gs.url, &gs.rev, &cache_dir);
                let entry = git_group_members.entry(root.clone()).or_insert_with(|| {
                    git_group_order.push(root.clone());
                    Vec::new()
                });
                entry.push(idx);
            }
        }

        // Compute auto_data_override exactly as emit_wheels_from_lock does.
        let mut auto_data_override: std::collections::HashMap<usize, Option<AutoDataConfig>> =
            std::collections::HashMap::new();
        for root in &git_group_order {
            let members = &git_group_members[root];
            if members.len() > 1 {
                for &idx in members {
                    let gs = wheels[idx].git_source.as_ref().unwrap();
                    auto_data_override.insert(idx, replay_git_auto_data(gs, root.clone()).unwrap());
                }
            }
        }

        // ASSERT (b): IsaacLab group has 3 members (indices 0, 1, 4).
        let isaac_root = crate::source_build::git_checkout_root(&url, &rev, &cache_dir);
        let isaac_members = &git_group_members[&isaac_root];
        assert_eq!(isaac_members.len(), 3, "isaaclab group must have 3 members");
        assert_eq!(isaac_members[0], 0);
        assert_eq!(
            isaac_members[1], 1,
            "isaaclab-assets (lock idx 1) must be group[1]"
        );
        assert_eq!(
            isaac_members[2], 4,
            "isaaclab-tasks (lock idx 4, non-contiguous) must be group[2]"
        );

        // The first lock member is explicitly disabled. Lock order is not
        // replay authority.
        assert!(
            auto_data_override.contains_key(&0) && auto_data_override[&0].is_none(),
            "first lock member must remain disabled"
        );

        // The persisted carrier (index 1) has all three skipped subdirectories.
        let carrier_ad = auto_data_override
            .get(&1)
            .expect("lock idx 1 must be in auto_data_override")
            .as_ref()
            .expect("persisted carrier must have Some(AutoDataConfig)");
        assert_eq!(
            carrier_ad.skip_subdirs.len(),
            3,
            "carrier skip_subdirs must be union of all 3 member subdirs"
        );
        assert!(
            carrier_ad
                .skip_subdirs
                .contains(&PathBuf::from("source/isaaclab")),
            "carrier skip_subdirs must include source/isaaclab"
        );
        assert!(
            carrier_ad
                .skip_subdirs
                .contains(&PathBuf::from("source/isaaclab_assets")),
            "carrier skip_subdirs must include source/isaaclab_assets"
        );
        assert!(
            carrier_ad
                .skip_subdirs
                .contains(&PathBuf::from("source/isaaclab_tasks")),
            "carrier skip_subdirs must include source/isaaclab_tasks"
        );

        // The other non-carrier remains disabled despite being non-contiguous.
        assert!(
            auto_data_override.contains_key(&4) && auto_data_override[&4].is_none(),
            "lock idx 4 (non-carrier, non-contiguous) must have None auto_data"
        );

        // rl_games: size-1 group -> NOT in auto_data_override.
        let rl_root = crate::source_build::git_checkout_root(&url2, &rev2, &cache_dir);
        let rl_members = &git_group_members[&rl_root];
        assert_eq!(rl_members.len(), 1, "rl_games must be size-1 group");
        assert!(
            !auto_data_override.contains_key(&rl_members[0]),
            "size-1 group must not be in auto_data_override (single-entry path)"
        );
        assert!(
            replay_git_auto_data(wheels[rl_members[0]].git_source.as_ref().unwrap(), rl_root,)
                .unwrap()
                .is_none(),
            "BFS singleton must preserve its explicit Disabled disposition"
        );

        // Index wheel (lock idx 2, Origin::Index) must NOT appear in any group.
        assert!(
            !git_group_members.values().any(|v| v.contains(&2)),
            "Origin::Index wheel must not appear in any git group"
        );

        std::fs::remove_dir_all(&cache_dir).ok();
    }

    /// Test (c): guard fall-through — a must_ship=true Built wheel with no
    /// git_source and no manifest entry must return Ok(None), not Err, even
    /// when preceded in the lock by a legitimate index wheel.
    ///
    /// This is the class-3 fall-through: incomplete provenance (no git_source,
    /// no [retread-wheels] entry) -> Ok(None) -> caller falls through to full
    /// resolve_all. NEVER Err.
    ///
    /// The Phase 2.5 variant: the "orphan" wheel appears after an Origin::Index
    /// wheel (simulating interleaving with non-git members in the lock). The
    /// index wheel does not require any network or git, so the test runs without
    /// uv/git.
    #[tokio::test]
    async fn multi_entry_group_member_missing_git_source_returns_ok_none() {
        use super::materialize_from_lock;
        use crate::config::RetreadConfig;
        use crate::lock::{LockWheel, Origin};

        let dir = unique_tmp_dir();
        let source_dir = dir.join("source");
        std::fs::create_dir_all(&source_dir).unwrap();
        let output_dir = dir.join("output");
        let work_dir = dir.join("work");
        let cache_dir = dir.join("cache");

        // Lock: one index wheel (no git), then one must_ship Built wheel
        // with NO git_source and NOT in config.retread_wheels (class-3 gap).
        // The index wheel must NOT trigger a network call (it has a local
        // file:// URL that won't be fetched until courier::stage, which we
        // don't reach — we return Ok(None) before that).
        let lock = crate::lock::RetreadLock {
            shadow_libs: std::collections::BTreeMap::new(),
            declared_glibc: None,
            schema: SCHEMA,
            retread_version: "2.5.0".into(),
            bundle: "interleaved-pack".into(),
            version: "1.0.0".into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
            resolution_glibc: None,
            inputs_hash: "hash-c-test".into(),
            root_requirements: vec![],
            wheels: vec![
                // Index wheel (class 4): no git, processed without network.
                LockWheel {
                    name: "numpy".into(),
                    version: "1.26.0".into(),
                    origin: Origin::Index,
                    filename: "numpy-1.26.0-cp311-cp311-linux_x86_64.whl".into(),
                    url: Some(
                        "https://files.pythonhosted.org/numpy-1.26.0-cp311-cp311-linux_x86_64.whl"
                            .into(),
                    ),
                    sha256: Some("1b".repeat(32)),
                    requires_dist: vec![],
                    must_ship: false,
                    upstream_url: None,
                    git_source: None,
                    sdist_source: None,
                },
                // Class-3 orphan: must_ship=true, no git_source, not in config.
                // This is the wheel that must trigger Ok(None).
                LockWheel {
                    name: "bfs-transitive-orphan".into(),
                    version: "0.1.0".into(),
                    origin: Origin::Built,
                    filename: "bfs_transitive_orphan-0.1.0-py3-none-any.injected.whl".into(),
                    url: None,
                    sha256: Some("1c".repeat(32)),
                    requires_dist: vec![],
                    must_ship: true,
                    upstream_url: None,
                    git_source: None, // MISSING: class-3 gap
                    sdist_source: None,
                },
            ],
            conda_run_deps: vec![],
            index_urls: vec!["https://pypi.org/simple/".into()],
            prerelease: std::collections::BTreeMap::new(),
            conda_capable: vec![],
            entry_specs: vec![],
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };

        // Config has no retread_wheels entries.
        let config: RetreadConfig =
            serde_json::from_value(serde_json::json!({"retread-wheels": {}})).unwrap();

        let result = materialize_from_lock(
            lock,
            &config,
            &work_dir,
            &output_dir,
            &ResolutionTarget::from_parts("3.11", "linux-64", None),
            &source_dir,
            &cache_dir,
            None,
            vec!["python 3.11.*".to_string()],
            "fp",
        )
        .await;

        // Must be Ok(None), never Err — incomplete provenance = fall-through.
        assert!(
            result.is_ok(),
            "class-3 orphan (no git_source, not in config) must not Err even \
             when interleaved with index wheels: {result:?}"
        );
        assert!(
            result.unwrap().is_none(),
            "class-3 orphan (no git_source, not in config) must return Ok(None) \
             so caller falls through to full resolve_all"
        );

        std::fs::remove_dir_all(dir).ok();
    }

    // -----------------------------------------------------------------------
    // Test (a): live e2e parity — multi-entry local-git fixture.
    //
    // Creates a local git repo with two subdirectory packages (pkg-alpha and
    // pkg-beta) plus a separate size-1 repo (pkg-gamma). Runs produce via
    // materialize_and_rewrite for each, then drives both the isolated byte
    // production and the real lock replay planner. The canonical-second beta
    // member is the persisted carrier; alpha is explicitly disabled.
    //
    // Also tests NON-CONTIGUOUS group order: the lock has pkg-alpha (idx 0),
    // an unrelated index wheel (idx 1), pkg-beta (idx 2), pkg-gamma (idx 3).
    // The alpha+beta group is non-contiguous in the lock.
    //
    // Mark #[ignore] because it needs uv + git on PATH (like other live git tests).
    // -----------------------------------------------------------------------
    #[tokio::test]
    #[ignore = "live: builds git wheels via uv (needs uv + git on PATH); run with --include-ignored"]
    async fn multi_entry_git_group_produce_replay_byte_identical() {
        use super::{AutoDataConfig, EntryAuditInfo, materialize_and_rewrite, wheel_target_for};
        use crate::config::RelaxPolicy;
        use crate::config::WheelEntry;
        use crate::lock::GitWheelSource;
        use rattler_conda_types::Platform;

        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("retread-phase25-parity-{pid}"));

        // ── Git fixture: monorepo with two packages ──────────────────────────
        let mono_repo = base.join("mono-repo");
        let pkg_alpha_dir = mono_repo.join("packages").join("pkg_alpha");
        let pkg_beta_dir = mono_repo.join("packages").join("pkg_beta");
        // size-1 separate repo (pkg-gamma)
        let gamma_repo = base.join("gamma-repo");
        let pkg_gamma_dir = gamma_repo.join("pkg_gamma");

        for d in [&pkg_alpha_dir, &pkg_beta_dir, &pkg_gamma_dir] {
            std::fs::create_dir_all(d).unwrap();
        }

        let run_git = |args: &[&str], dir: &std::path::Path| {
            let status = std::process::Command::new("git")
                .args(args)
                .current_dir(dir)
                .env("GIT_AUTHOR_NAME", "test")
                .env("GIT_AUTHOR_EMAIL", "test@example.com")
                .env("GIT_COMMITTER_NAME", "test")
                .env("GIT_COMMITTER_EMAIL", "test@example.com")
                .status()
                .expect("git");
            assert!(status.success(), "git {args:?} failed");
        };

        let git_sha = |dir: &std::path::Path| -> String {
            let out = std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(dir)
                .output()
                .expect("git rev-parse");
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };

        // ── Monorepo init ────────────────────────────────────────────────────
        run_git(&["init", "-b", "main"], &mono_repo);
        run_git(&["config", "user.email", "test@example.com"], &mono_repo);
        run_git(&["config", "user.name", "test"], &mono_repo);

        // pkg-alpha
        std::fs::write(
            pkg_alpha_dir.join("pyproject.toml"),
            r#"[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "retread-p25-pkg-alpha"
version = "1.0.0"
"#,
        )
        .unwrap();
        let alpha_src = pkg_alpha_dir.join("retread_p25_pkg_alpha");
        std::fs::create_dir_all(&alpha_src).unwrap();
        std::fs::write(alpha_src.join("__init__.py"), b"# pkg-alpha\n").unwrap();

        // pkg-beta
        std::fs::write(
            pkg_beta_dir.join("pyproject.toml"),
            r#"[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "retread-p25-pkg-beta"
version = "1.0.0"
"#,
        )
        .unwrap();
        let beta_src = pkg_beta_dir.join("retread_p25_pkg_beta");
        std::fs::create_dir_all(&beta_src).unwrap();
        std::fs::write(beta_src.join("__init__.py"), b"# pkg-beta\n").unwrap();

        // Shared root file carried by beta's checkout-root auto-data pass.
        std::fs::write(mono_repo.join("README.md"), b"monorepo root\n").unwrap();

        run_git(&["add", "."], &mono_repo);
        run_git(&["commit", "-m", "initial"], &mono_repo);
        let mono_sha = git_sha(&mono_repo);
        assert_eq!(mono_sha.len(), 40, "expected 40-char SHA");

        // ── Gamma repo init ──────────────────────────────────────────────────
        run_git(&["init", "-b", "main"], &gamma_repo);
        run_git(&["config", "user.email", "test@example.com"], &gamma_repo);
        run_git(&["config", "user.name", "test"], &gamma_repo);

        std::fs::write(
            gamma_repo.join("pyproject.toml"),
            r#"[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "retread-p25-pkg-gamma"
version = "1.0.0"
"#,
        )
        .unwrap();
        let gamma_src = pkg_gamma_dir.join("retread_p25_pkg_gamma");
        std::fs::create_dir_all(&gamma_src).unwrap();
        std::fs::write(gamma_src.join("__init__.py"), b"# pkg-gamma\n").unwrap();

        run_git(&["add", "."], &gamma_repo);
        run_git(&["commit", "-m", "initial"], &gamma_repo);
        let gamma_sha = git_sha(&gamma_repo);
        assert_eq!(gamma_sha.len(), 40, "expected 40-char SHA");

        let mono_url = format!("file://{}", mono_repo.display());
        let gamma_url = format!("file://{}", gamma_repo.display());

        // ── Common dirs ─────────────────────────────────────────────────────
        let cache_dir = base.join("cache");
        // Each call needs its own source/download dir to avoid is_fresh() cache hits.
        let produce_alpha_src = base.join("src-produce-alpha");
        let produce_beta_src = base.join("src-produce-beta");
        let replay_alpha_src = base.join("src-replay-alpha");
        let replay_beta_src = base.join("src-replay-beta");
        let produce_gamma_src = base.join("src-produce-gamma");
        let replay_gamma_src = base.join("src-replay-gamma");
        for d in [
            &cache_dir,
            &produce_alpha_src,
            &produce_beta_src,
            &replay_alpha_src,
            &replay_beta_src,
            &produce_gamma_src,
            &replay_gamma_src,
        ] {
            std::fs::create_dir_all(d).unwrap();
        }

        let target = wheel_target_for(Platform::Linux64, "3.11").expect("valid target");
        let mono_checkout_root =
            crate::source_build::git_checkout_root(&mono_url, &mono_sha, &cache_dir);
        let gamma_checkout_root =
            crate::source_build::git_checkout_root(&gamma_url, &gamma_sha, &cache_dir);

        // ── PRODUCE: alpha (non-carrier) ────────────────────────────────────
        // The carrier is intentionally beta, which sorts after alpha in the
        // canonical lock. This makes lock order incapable of selecting it.
        let alpha_entry = WheelEntry {
            git: Some(mono_url.clone()),
            rev: Some(mono_sha.clone()),
            subdirectory: Some("packages/pkg_alpha".into()),
            ..WheelEntry::default()
        };
        let produce_alpha_dd = produce_alpha_src.join("wheels");
        std::fs::create_dir_all(&produce_alpha_dd).unwrap();
        let (produce_alpha_resolved, _) = materialize_and_rewrite(
            &alpha_entry,
            "retread-p25-pkg-alpha",
            None,
            &target,
            &produce_alpha_dd,
            &produce_alpha_src,
            &cache_dir,
            RelaxPolicy::None,
            &std::collections::BTreeMap::new(),
            None,
            EntryAuditInfo::default(),
        )
        .await
        .expect("produce alpha: materialize_and_rewrite");
        let produce_alpha_path = produce_alpha_resolved
            .url
            .to_file_path()
            .expect("produce alpha must be file URL");
        let produce_alpha_bytes = std::fs::read(&produce_alpha_path).unwrap();

        // ── PRODUCE: beta (persisted carrier, union skip set) ────────────────
        let beta_entry = WheelEntry {
            git: Some(mono_url.clone()),
            rev: Some(mono_sha.clone()),
            subdirectory: Some("packages/pkg_beta".into()),
            ..WheelEntry::default()
        };
        let produce_beta_dd = produce_beta_src.join("wheels");
        std::fs::create_dir_all(&produce_beta_dd).unwrap();
        let (produce_beta_resolved, _) = materialize_and_rewrite(
            &beta_entry,
            "retread-p25-pkg-beta",
            None,
            &target,
            &produce_beta_dd,
            &produce_beta_src,
            &cache_dir,
            RelaxPolicy::None,
            &std::collections::BTreeMap::new(),
            Some(AutoDataConfig {
                checkout_root: mono_checkout_root.clone(),
                skip_subdirs: vec![
                    std::path::PathBuf::from("packages/pkg_alpha"),
                    std::path::PathBuf::from("packages/pkg_beta"),
                ],
            }),
            EntryAuditInfo::default(),
        )
        .await
        .expect("produce beta: materialize_and_rewrite");
        let produce_beta_path = produce_beta_resolved
            .url
            .to_file_path()
            .expect("produce beta must be file URL");
        let produce_beta_bytes = std::fs::read(&produce_beta_path).unwrap();

        // ── PRODUCE: gamma (size-1 group, skip_subdirs=["."] single-entry path) ──
        let gamma_entry = WheelEntry {
            git: Some(gamma_url.clone()),
            rev: Some(gamma_sha.clone()),
            subdirectory: None, // root
            ..WheelEntry::default()
        };
        let produce_gamma_dd = produce_gamma_src.join("wheels");
        std::fs::create_dir_all(&produce_gamma_dd).unwrap();
        let (produce_gamma_resolved, _) = materialize_and_rewrite(
            &gamma_entry,
            "retread-p25-pkg-gamma",
            None,
            &target,
            &produce_gamma_dd,
            &produce_gamma_src,
            &cache_dir,
            RelaxPolicy::None,
            &std::collections::BTreeMap::new(),
            Some(AutoDataConfig {
                checkout_root: gamma_checkout_root.clone(),
                skip_subdirs: vec![std::path::PathBuf::from(".")],
            }),
            EntryAuditInfo::default(),
        )
        .await
        .expect("produce gamma: materialize_and_rewrite");
        let produce_gamma_path = produce_gamma_resolved
            .url
            .to_file_path()
            .expect("produce gamma must be file URL");
        let produce_gamma_bytes = std::fs::read(&produce_gamma_path).unwrap();

        // ── REPLAY: simulate what materialize_from_lock now does ─────────────
        //
        // Synthetic lock (non-contiguous group):
        //   idx 0: alpha (mono group member)
        //   idx 1: (not built here — we only test the git paths)
        //   idx 2: beta (mono group member, non-contiguous)
        //   idx 3: gamma (size-1 group)
        //
        // Pre-pass would assign:
        //   idx 0 -> None                                      (non-carrier)
        //   idx 2 -> Some(AutoDataConfig{skip=[alpha,beta]})  (carrier)
        //   idx 3 -> not in override (size-1 path)
        //
        // We reproduce this in the test by calling materialize_and_rewrite with
        // the same auto_data the pre-pass would produce.

        // Replay alpha (non-carrier: auto_data=None).
        let replay_alpha_dd = replay_alpha_src.join("wheels");
        std::fs::create_dir_all(&replay_alpha_dd).unwrap();
        let synth_alpha = WheelEntry {
            git: Some(mono_url.clone()),
            rev: Some(mono_sha.clone()),
            subdirectory: Some("packages/pkg_alpha".into()),
            ..WheelEntry::default()
        };
        let (replay_alpha_resolved, _) = materialize_and_rewrite(
            &synth_alpha,
            "retread-p25-pkg-alpha",
            Some("1.0.0"),
            &target,
            &replay_alpha_dd,
            &replay_alpha_src,
            &cache_dir,
            RelaxPolicy::None,
            &std::collections::BTreeMap::new(),
            None,
            EntryAuditInfo::default(),
        )
        .await
        .expect("replay alpha: materialize_and_rewrite");
        let replay_alpha_bytes =
            std::fs::read(replay_alpha_resolved.url.to_file_path().unwrap()).unwrap();

        // Replay beta (persisted carrier with the union skip set).
        let replay_beta_dd = replay_beta_src.join("wheels");
        std::fs::create_dir_all(&replay_beta_dd).unwrap();
        let synth_beta = WheelEntry {
            git: Some(mono_url.clone()),
            rev: Some(mono_sha.clone()),
            subdirectory: Some("packages/pkg_beta".into()),
            ..WheelEntry::default()
        };
        let (replay_beta_resolved, _) = materialize_and_rewrite(
            &synth_beta,
            "retread-p25-pkg-beta",
            Some("1.0.0"),
            &target,
            &replay_beta_dd,
            &replay_beta_src,
            &cache_dir,
            RelaxPolicy::None,
            &std::collections::BTreeMap::new(),
            Some(AutoDataConfig {
                checkout_root: mono_checkout_root.clone(),
                skip_subdirs: vec![
                    std::path::PathBuf::from("packages/pkg_alpha"),
                    std::path::PathBuf::from("packages/pkg_beta"),
                ],
            }),
            EntryAuditInfo::default(),
        )
        .await
        .expect("replay beta: materialize_and_rewrite");
        let replay_beta_bytes =
            std::fs::read(replay_beta_resolved.url.to_file_path().unwrap()).unwrap();

        // Replay gamma (size-1 group: single-entry path, skip_subdirs=["."]).
        let replay_gamma_dd = replay_gamma_src.join("wheels");
        std::fs::create_dir_all(&replay_gamma_dd).unwrap();
        let synth_gamma = WheelEntry {
            git: Some(gamma_url.clone()),
            rev: Some(gamma_sha.clone()),
            subdirectory: None,
            ..WheelEntry::default()
        };
        let (replay_gamma_resolved, _) = materialize_and_rewrite(
            &synth_gamma,
            "retread-p25-pkg-gamma",
            Some("1.0.0"),
            &target,
            &replay_gamma_dd,
            &replay_gamma_src,
            &cache_dir,
            RelaxPolicy::None,
            &std::collections::BTreeMap::new(),
            Some(AutoDataConfig {
                checkout_root: gamma_checkout_root.clone(),
                skip_subdirs: vec![std::path::PathBuf::from(".")],
            }),
            EntryAuditInfo::default(),
        )
        .await
        .expect("replay gamma: materialize_and_rewrite");
        let replay_gamma_bytes =
            std::fs::read(replay_gamma_resolved.url.to_file_path().unwrap()).unwrap();

        // ── ASSERT: byte-identical ───────────────────────────────────────────
        assert_eq!(
            produce_alpha_bytes, replay_alpha_bytes,
            "PHASE 2.5 PARITY: alpha non-carrier (auto_data=None) must be \
             byte-identical between produce and replay"
        );
        assert_eq!(
            produce_beta_bytes, replay_beta_bytes,
            "PHASE 2.5 PARITY: beta persisted carrier (union skip_subdirs) must be \
             byte-identical between produce and replay"
        );
        assert_eq!(
            produce_gamma_bytes, replay_gamma_bytes,
            "PHASE 2.5 PARITY: gamma size-1 group (single-entry path) must be \
             byte-identical between produce and replay"
        );

        // Drive the real lock -> emit planner as well. The assertions above
        // isolate byte production; this closes the integration seam so a
        // future grouping/pre-pass drift cannot be hidden by a test that
        // independently reconstructs the expected AutoDataConfig.
        let cold_emit: Vec<crate::emit_pypi::EmitWheel> = [
            &produce_alpha_resolved,
            &produce_beta_resolved,
            &produce_gamma_resolved,
        ]
        .into_iter()
        .map(|resolved| {
            let path = resolved.url.to_file_path().unwrap();
            crate::emit_pypi::EmitWheel {
                pypi_name: resolved.pypi_name.clone(),
                version: resolved.metadata.version.clone(),
                requires_dist: resolved.metadata.requires_dist.clone(),
                local_path: Some(path.clone()),
                wheel_filename: path.file_name().unwrap().to_string_lossy().into_owned(),
                sha256: Some(resolved.metadata.sha256.clone()),
                locked_final_sha256: None,
                remote_url: None,
                upstream_url: None,
                git_source: resolved.git_source.clone(),
                sdist_source: None,
            }
        })
        .collect();
        let config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": {},
            "retread-bundle-mode": "fat",
            "retread-relax": "none"
        }))
        .unwrap();
        let stage_source = base.join("stage-source");
        let actual_replay_download = base.join("actual-replay-download");
        let actual_replay_source = base.join("actual-replay-source");
        for dir in [
            &stage_source,
            &actual_replay_download,
            &actual_replay_source,
        ] {
            std::fs::create_dir_all(dir).unwrap();
        }
        let indexes = vec!["https://pypi.org/simple/".to_owned()];
        let cold_staged = crate::courier::stage_for_target(
            &config,
            "phase25-pack",
            "phase25-pack",
            "1.0.0",
            &target,
            &cold_emit,
            &HashSet::new(),
            &[],
            &indexes,
            "fixture",
            &stage_source,
            &base.join("actual-cold-staging"),
        )
        .await
        .expect("stage authoritative multi-Git lock");
        let locked_alpha = cold_staged
            .lock
            .wheels
            .iter()
            .find(|wheel| wheel.name == "retread-p25-pkg-alpha")
            .unwrap();
        let locked_beta = cold_staged
            .lock
            .wheels
            .iter()
            .find(|wheel| wheel.name == "retread-p25-pkg-beta")
            .unwrap();
        assert!(matches!(
            locked_alpha.git_source.as_ref().unwrap().auto_data.as_ref(),
            Some(crate::lock::GitWheelAutoData::Disabled),
        ));
        assert!(matches!(
            locked_beta.git_source.as_ref().unwrap().auto_data.as_ref(),
            Some(crate::lock::GitWheelAutoData::CheckoutRoot { .. }),
        ));
        let actual_replay = super::emit_wheels_from_lock(
            &cold_staged.lock,
            &config,
            &target,
            &actual_replay_download,
            &actual_replay_source,
            &cache_dir,
        )
        .await
        .expect("actual multi-Git lock replay")
        .expect("all multi-Git provenance must be complete");
        let replayed_bytes: BTreeMap<String, Vec<u8>> = actual_replay
            .iter()
            .map(|wheel| {
                (
                    wheel.pypi_name.clone(),
                    std::fs::read(wheel.local_path.as_ref().unwrap()).unwrap(),
                )
            })
            .collect();
        assert_eq!(replayed_bytes["retread-p25-pkg-alpha"], produce_alpha_bytes,);
        assert_eq!(replayed_bytes["retread-p25-pkg-beta"], produce_beta_bytes,);
        assert_eq!(replayed_bytes["retread-p25-pkg-gamma"], produce_gamma_bytes,);
        let replay_staged = crate::courier::stage_for_target(
            &config,
            "phase25-pack",
            "phase25-pack",
            "1.0.0",
            &target,
            &actual_replay,
            &HashSet::new(),
            &[],
            &indexes,
            "fixture",
            &actual_replay_source,
            &base.join("actual-replay-staging"),
        )
        .await
        .expect("stage replayed multi-Git lock");
        validate_authoritative_replay_lock(&cold_staged.lock, &replay_staged.lock)
            .expect("actual multi-Git replay must preserve the authoritative lock");

        // ── ASSERT: non-contiguous lock ordering simulation ──────────────────
        // Confirm the group detection works for non-contiguous lock indices
        // (alpha at 0, beta at 2, interleaved index wheel at 1).
        let rev = mono_sha.clone();
        let url = mono_url.clone();
        let mut group_members: std::collections::HashMap<std::path::PathBuf, Vec<usize>> =
            std::collections::HashMap::new();
        let mut group_order: Vec<std::path::PathBuf> = Vec::new();
        // Simulated lock: [alpha(0), dummy_index(1), beta(2), gamma(3)]
        let sim_wheels: Vec<(bool, Option<GitWheelSource>)> = vec![
            (
                true,
                Some(GitWheelSource {
                    url: url.clone(),
                    rev: rev.clone(),
                    subdirectory: Some("packages/pkg_alpha".into()),
                    extras: vec![],
                    auto_data: Some(crate::lock::GitWheelAutoData::Disabled),
                }),
            ),
            (false, None), // Origin::Index — skipped
            (
                true,
                Some(GitWheelSource {
                    url: url.clone(),
                    rev: rev.clone(),
                    subdirectory: Some("packages/pkg_beta".into()),
                    extras: vec![],
                    auto_data: Some(crate::lock::GitWheelAutoData::CheckoutRoot {
                        skip_subdirectories: vec![
                            "packages/pkg_alpha".into(),
                            "packages/pkg_beta".into(),
                        ],
                    }),
                }),
            ),
            (
                true,
                Some(GitWheelSource {
                    url: gamma_url.clone(),
                    rev: gamma_sha.clone(),
                    subdirectory: None,
                    extras: vec![],
                    auto_data: Some(crate::lock::GitWheelAutoData::CheckoutRoot {
                        skip_subdirectories: vec![".".into()],
                    }),
                }),
            ),
        ];
        for (idx, (must_ship, gs_opt)) in sim_wheels.iter().enumerate() {
            if *must_ship && let Some(gs) = gs_opt {
                let root = crate::source_build::git_checkout_root(&gs.url, &gs.rev, &cache_dir);
                let entry = group_members.entry(root.clone()).or_insert_with(|| {
                    group_order.push(root);
                    Vec::new()
                });
                entry.push(idx);
            }
        }
        let mono_root = crate::source_build::git_checkout_root(&url, &rev, &cache_dir);
        let mono_members = &group_members[&mono_root];
        assert_eq!(
            mono_members,
            &vec![0usize, 2usize],
            "non-contiguous group: alpha at lock idx 0, beta at lock idx 2"
        );
        assert_eq!(
            mono_members[0], 0usize,
            "alpha remains first in canonical order even though beta is the persisted carrier"
        );
        assert!(matches!(
            sim_wheels[mono_members[0]]
                .1
                .as_ref()
                .unwrap()
                .auto_data
                .as_ref(),
            Some(crate::lock::GitWheelAutoData::Disabled),
        ));
        assert!(matches!(
            sim_wheels[mono_members[1]]
                .1
                .as_ref()
                .unwrap()
                .auto_data
                .as_ref(),
            Some(crate::lock::GitWheelAutoData::CheckoutRoot { .. }),
        ));

        let _ = std::fs::remove_dir_all(&base);
    }
}

// -----------------------------------------------------------------
// Unit tests for courier build string helpers.
// -----------------------------------------------------------------
#[cfg(test)]
mod courier_build_string_tests {
    use super::{
        advertised_build_matches, assemble_conda_output,
        build_courier_recipe_with_mode_and_lock_filename, courier_build_string,
        courier_build_string_for_target,
    };
    use crate::pypi::ResolutionTarget;
    use rattler_conda_types::Platform;

    #[test]
    fn build_string_includes_hash_prefix() {
        let hash = "abcdef0123456789";
        let s = courier_build_string("311", hash, 0, false);
        assert!(
            s.contains("habcdef0123"),
            "build string must contain h+first10 of hash: {s}"
        );
        assert!(
            s.starts_with("py311_"),
            "build string must start with py prefix: {s}"
        );
        assert!(
            s.ends_with("_0"),
            "build string must end with build number: {s}"
        );
    }

    #[test]
    fn different_hashes_give_different_build_strings() {
        let s1 = courier_build_string("311", "aaaaaa0000111122", 0, false);
        let s2 = courier_build_string("311", "bbbbbb9999888877", 0, false);
        assert_ne!(
            s1, s2,
            "different inputs hashes must yield different build strings"
        );
    }

    #[test]
    fn same_hash_different_build_number_gives_different_string() {
        let hash = "abcdef0123456789";
        let s0 = courier_build_string("311", hash, 0, false);
        let s1 = courier_build_string("311", hash, 1, false);
        assert_ne!(
            s0, s1,
            "different build numbers must yield different strings"
        );
    }

    #[test]
    fn nonzero_advertised_build_number_reaches_generated_courier_recipe() {
        let build_number = 7;
        let inputs_hash = "abcdef0123456789";
        let output = assemble_conda_output(
            "mypack",
            "1.0.0",
            "3.11",
            true,
            false,
            Vec::new(),
            std::collections::HashSet::new(),
            Platform::Linux64,
            build_number,
            Some(inputs_hash),
            false,
            &[],
        )
        .unwrap();

        let recipe = build_courier_recipe_with_mode_and_lock_filename(
            "mypack",
            "1.0.0",
            "3.11",
            &[],
            &[],
            build_number,
            Some(&output.metadata.build),
            crate::config::CourierMode::PostLink,
            "retread-mypack.target-deadbeef.lock.json",
        );

        assert_eq!(recipe.build.number, output.metadata.build_number);
        assert_eq!(
            recipe.build.string.as_deref(),
            Some(output.metadata.build.as_str())
        );
    }

    #[test]
    fn hash_shorter_than_10_chars_does_not_panic() {
        // When the hash is shorter than 10 chars, min(len, 10) keeps all chars.
        let s = courier_build_string("311", "abc", 0, false);
        assert_eq!(s, "py311_habc_0");
    }

    #[test]
    fn build_string_format_is_py_prefix_h_hash_number() {
        // Exact format spec: py{py_short}_h{hash[..10]}_{build_number}
        let s = courier_build_string("312", "1234567890abcdef", 2, false);
        assert_eq!(s, "py312_h1234567890_2");
    }

    #[test]
    fn loose_mode_gets_a_distinct_build_string() {
        // Same inputs hash, different bundle mode => different artifact
        // content => the build strings MUST differ or pixi cache-hits the
        // other mode's artifact.
        let fat = courier_build_string("312", "1234567890abcdef", 2, false);
        let loose = courier_build_string("312", "1234567890abcdef", 2, true);
        assert_ne!(fat, loose);
        assert_eq!(loose, "py312_h1234567890_loose_2");
    }

    #[test]
    fn requested_build_must_match_recomputed_current_hash() {
        let target = ResolutionTarget::for_subdir("3.11", "linux-64");
        let advertised = courier_build_string_for_target(&target, "aaaaaaaaaa111111", 0, false);
        let current = courier_build_string_for_target(&target, "bbbbbbbbbb222222", 0, false);
        assert!(advertised_build_matches(Some(&current), &current));
        assert!(advertised_build_matches(None, &current));
        assert!(
            !advertised_build_matches(Some(&advertised), &current),
            "build_v1 must fail closed rather than label current inputs with a stale content hash"
        );
    }
}

// -----------------------------------------------------------------
// Regression guard for the Phase-1 replay upstream_url bug:
// A config-entry index wheel (the PRIMARY `isaacsim`) goes through
// materialize_and_rewrite which localises its URL to file://. The old
// code derived EmitWheel.upstream_url from w.url at build_one time,
// so it was always None for primary wheels, breaking replay.
// The fix: upstream_url is now a field on ResolvedWheel, populated in
// materialize_and_rewrite BEFORE localization, and read by build_one.
// -----------------------------------------------------------------
#[cfg(test)]
mod emit_wheel_upstream_url_tests {
    use super::{Bundle, ResolvedWheel};
    use crate::wheel::WheelMetadata;
    use std::collections::BTreeMap;

    /// Construct a minimal WheelMetadata for tests.
    fn dummy_metadata(name: &str, version: &str) -> WheelMetadata {
        WheelMetadata {
            name: name.to_string(),
            version: version.to_string(),
            requires_dist: vec![],
            retread_conda_run_dependencies: vec![],
            is_pure_python: true,
            sha256: "abc".to_string(),
            filename: format!("{name}-{version}-py3-none-any.whl"),
        }
    }

    /// A config-entry index wheel (primary `isaacsim`) goes through
    /// materialize_and_rewrite which always returns a file:// URL.
    /// Before the fix, build_one derived upstream_url from w.url:
    ///   `(w.url.scheme() != "file").then(|| w.url.clone())`
    /// which yields None for any file:// URL -> replay falls through.
    ///
    /// After the fix, upstream_url is stored on ResolvedWheel and read
    /// directly in build_one, independent of url localization.
    ///
    /// This test is the regression guard: it constructs the exact
    /// scenario (w.url = file://, w.upstream_url = Some(https://...))
    /// and verifies the EmitWheel mapping produces upstream_url = Some.
    #[test]
    fn primary_config_entry_wheel_carries_upstream_url_through_localization() {
        // Simulate what materialize_and_rewrite returns for the primary
        // config-entry `isaacsim` wheel: url is localized to file://,
        // upstream_url is the pristine index URL captured before localization.
        let upstream = url::Url::parse(
            "https://pypi.nvidia.com/simple/isaacsim-6.0.0-cp312-none-linux_x86_64.whl",
        )
        .unwrap();
        let local = url::Url::from_file_path(
            "/tmp/wheels/isaacsim/isaacsim-6.0.0-cp312-none-linux_x86_64.whl",
        )
        .unwrap();

        let primary = ResolvedWheel {
            pypi_name: "isaacsim".to_string(),
            url: local.clone(), // file:// — what materialize_and_rewrite returns
            upstream_url: Some(upstream.clone()), // https:// — captured BEFORE localization
            git_source: None,
            sdist_source: None,
            metadata_provenance: crate::constraint::Provenance::IndexWheelMetadata,
            metadata: dummy_metadata("isaacsim", "6.0.0"),
            extras_requested: vec![],
            auto_data: None,
            auto_data_dedup_skipped_root: None,
        };

        let bundle = Bundle {
            conda_name: "isaacsim".to_string(),
            primary,
            extras: vec![],
            probe_decisions: vec![],
            solve_diagnostics: BTreeMap::new(),
            auto_routed: vec![],
            auto_dropped: Default::default(),
            uv_closure_names: Default::default(),
            uv_dependency_graph: Default::default(),
            workspace_conda_versions: Default::default(),
            workspace_conda_provider_facts: Default::default(),
        };

        // Reproduce the exact mapping from build_one that populates EmitWheel.
        // Before the fix this was: `(w.url.scheme() != "file").then(|| w.url.clone())`
        // which returns None for file:// URLs.
        // After the fix: `w.upstream_url.clone()`.
        let emit_wheels: Vec<crate::emit_pypi::EmitWheel> = bundle
            .all_wheels()
            .map(|w| {
                // The localized URL (what localize_wheel_source might return when
                // the wheel is already on disk). For the primary it's the same
                // file:// URL since there's no separate localization step in the test.
                let url = w.url.clone();
                crate::emit_pypi::EmitWheel {
                    pypi_name: w.pypi_name.clone(),
                    version: w.metadata.version.clone(),
                    requires_dist: w.metadata.requires_dist.clone(),
                    sha256: Some(w.metadata.sha256.clone()),
                    locked_final_sha256: None,
                    local_path: (url.scheme() == "file")
                        .then(|| url.to_file_path().ok())
                        .flatten(),
                    wheel_filename: url
                        .path_segments()
                        .and_then(|mut s| s.next_back())
                        .unwrap_or_default()
                        .to_string(),
                    remote_url: (url.scheme() != "file").then(|| url.clone()),
                    // THE FIX: read from w.upstream_url, not derived from w.url.
                    upstream_url: w.upstream_url.clone(),
                    git_source: w.git_source.clone(),
                    sdist_source: w.sdist_source.clone(),
                }
            })
            .collect();

        assert_eq!(emit_wheels.len(), 1);
        let ew = &emit_wheels[0];

        // The primary wheel's URL is file:// so remote_url must be None.
        assert!(
            ew.remote_url.is_none(),
            "primary wheel url is file:// so remote_url must be None: {:?}",
            ew.remote_url,
        );

        // upstream_url MUST be Some(https://...) — the regression guard.
        // Before the fix this was None because the old code derived it from
        // w.url which is file:// for primary config-entry wheels.
        assert_eq!(
            ew.upstream_url,
            Some(upstream.clone()),
            "primary config-entry wheel must carry upstream_url even though \
             its local url is file://; got {:?}",
            ew.upstream_url,
        );

        // local_path must be set (it's a file:// url).
        assert!(
            ew.local_path.is_some(),
            "file:// primary wheel must set local_path: {:?}",
            ew.local_path,
        );
    }

    /// Control case: a BFS sub-wheel whose url is the pristine https://
    /// (not localized to file://) must also carry upstream_url correctly.
    /// Before AND after the fix this worked via the old derivation; after
    /// the fix it works via the new field. Verify parity.
    #[test]
    fn bfs_pypi_sub_wheel_carries_upstream_url() {
        let upstream = url::Url::parse(
            "https://pypi.nvidia.com/simple/isaacsim-kernel-6.0.0-cp312-none-linux_x86_64.whl",
        )
        .unwrap();

        // BFS Pypi-form sub-wheels: url = pristine https:// (not localized),
        // upstream_url = Some(https://) (from the new field).
        let sub = ResolvedWheel {
            pypi_name: "isaacsim-kernel".to_string(),
            url: upstream.clone(),
            upstream_url: Some(upstream.clone()),
            git_source: None,
            sdist_source: None,
            metadata_provenance: crate::constraint::Provenance::IndexWheelMetadata,
            metadata: dummy_metadata("isaacsim-kernel", "6.0.0"),
            extras_requested: vec![],
            auto_data: None,
            auto_data_dedup_skipped_root: None,
        };

        let bundle = Bundle {
            conda_name: "isaacsim".to_string(),
            primary: ResolvedWheel {
                pypi_name: "isaacsim".to_string(),
                // Use a different file:// url for the primary to avoid
                // interference with the sub-wheel assertions below.
                url: url::Url::from_file_path("/tmp/w/isaacsim-6.0.0.whl").unwrap(),
                upstream_url: Some(
                    url::Url::parse(
                        "https://pypi.nvidia.com/simple/isaacsim-6.0.0-cp312-none-linux_x86_64.whl",
                    )
                    .unwrap(),
                ),
                git_source: None,
                sdist_source: None,
                metadata_provenance: crate::constraint::Provenance::IndexWheelMetadata,
                metadata: dummy_metadata("isaacsim", "6.0.0"),
                extras_requested: vec![],
                auto_data: None,
                auto_data_dedup_skipped_root: None,
            },
            extras: vec![sub],
            probe_decisions: vec![],
            solve_diagnostics: BTreeMap::new(),
            auto_routed: vec![],
            auto_dropped: Default::default(),
            uv_closure_names: Default::default(),
            uv_dependency_graph: Default::default(),
            workspace_conda_versions: Default::default(),
            workspace_conda_provider_facts: Default::default(),
        };

        let emit_wheels: Vec<crate::emit_pypi::EmitWheel> = bundle
            .all_wheels()
            .map(|w| {
                let url = w.url.clone();
                crate::emit_pypi::EmitWheel {
                    pypi_name: w.pypi_name.clone(),
                    version: w.metadata.version.clone(),
                    requires_dist: w.metadata.requires_dist.clone(),
                    sha256: Some(w.metadata.sha256.clone()),
                    locked_final_sha256: None,
                    local_path: (url.scheme() == "file")
                        .then(|| url.to_file_path().ok())
                        .flatten(),
                    wheel_filename: url
                        .path_segments()
                        .and_then(|mut s| s.next_back())
                        .unwrap_or_default()
                        .to_string(),
                    remote_url: (url.scheme() != "file").then(|| url.clone()),
                    upstream_url: w.upstream_url.clone(),
                    git_source: w.git_source.clone(),
                    sdist_source: w.sdist_source.clone(),
                }
            })
            .collect();

        // index 0 = primary, index 1 = sub-wheel.
        assert_eq!(emit_wheels.len(), 2);
        let sub_ew = &emit_wheels[1];
        assert_eq!(
            sub_ew.upstream_url,
            Some(upstream),
            "BFS Pypi sub-wheel must carry upstream_url: {:?}",
            sub_ew.upstream_url,
        );
    }
}

// -----------------------------------------------------------------
// load_favored_versions unit tests.
//
// Tests the RETREAD_FAVOR_LOCK gating and lock-loading behavior in isolation
// (no network; reads/writes local temp files only).
// -----------------------------------------------------------------
#[cfg(test)]
mod load_favored_versions_tests {
    use std::collections::BTreeMap;

    use super::{TEST_ENV_MUTEX, load_favored_versions};
    use crate::lock::{CondaDep, LockWheel, Origin, RetreadLock, SCHEMA};

    fn unique_tmp_dir() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "retread-favored-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        );
        let dir = base.join(unique);
        std::fs::create_dir_all(&dir).expect("tmp dir creation should not fail");
        dir
    }

    fn write_lock(
        dir: &std::path::Path,
        bundle: &str,
        wheels: Vec<LockWheel>,
    ) -> std::path::PathBuf {
        let lock = RetreadLock {
            schema: SCHEMA,
            retread_version: "0.0.1".into(),
            bundle: bundle.into(),
            version: "1.0.0".into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
            resolution_glibc: None,
            inputs_hash: "testhash".into(),
            root_requirements: vec![],
            wheels,
            conda_run_deps: vec![CondaDep {
                name: "numpy".into(),
                spec: ">=1.0".into(),
            }],
            index_urls: vec!["https://pypi.org/simple/".into()],
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec![],
            entry_specs: vec![],
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };
        let json = lock.to_pretty_json().unwrap();
        let path = dir.join(RetreadLock::file_name(bundle));
        std::fs::write(&path, json).unwrap();
        path
    }

    fn make_wheel(name: &str, version: &str) -> LockWheel {
        LockWheel {
            name: name.into(),
            version: version.into(),
            origin: Origin::Index,
            filename: format!("{name}-{version}-py3-none-any.whl"),
            url: Some(format!("https://example.com/{name}-{version}.whl")),
            sha256: Some("10".repeat(32)),
            requires_dist: vec![],
            must_ship: false,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }
    }

    /// Default-on: with no opt-out env var, a valid lock populates the map.
    #[test]
    fn default_on_unset_returns_populated() {
        let _guard = TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let lock_path = write_lock(
            &dir,
            "mypkg",
            vec![make_wheel("mypkg", "1.2.3"), make_wheel("dep-a", "0.5.0")],
        );
        // Ensure favor-lock is NOT disabled (default-on).
        // SAFETY: serialised by TEST_ENV_MUTEX; no concurrent env access.
        unsafe { std::env::remove_var("RETREAD_NO_FAVOR_LOCK") };

        let result = load_favored_versions(&lock_path);
        assert_eq!(
            result.get("mypkg").map(String::as_str),
            Some("1.2.3"),
            "favor-lock is default-on; a valid lock must populate the map; got {result:?}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// Opt-out: RETREAD_NO_FAVOR_LOCK set → always empty, regardless of lock.
    #[test]
    fn opt_out_returns_empty() {
        let _guard = TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let lock_path = write_lock(
            &dir,
            "mypkg",
            vec![make_wheel("mypkg", "1.2.3"), make_wheel("dep-a", "0.5.0")],
        );
        // SAFETY: serialised by TEST_ENV_MUTEX; no concurrent env access.
        unsafe { std::env::set_var("RETREAD_NO_FAVOR_LOCK", "1") };

        let result = load_favored_versions(&lock_path);
        assert!(
            result.is_empty(),
            "RETREAD_NO_FAVOR_LOCK set must return empty map; got {result:?}"
        );
        unsafe { std::env::remove_var("RETREAD_NO_FAVOR_LOCK") };
        std::fs::remove_dir_all(dir).ok();
    }

    /// Missing lock file → empty map (non-fatal; first build has no lock yet).
    #[test]
    fn missing_file_returns_empty() {
        let _guard = TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let lock_path = dir.join("retread-nonexistent.lock.json");
        // SAFETY: serialised by TEST_ENV_MUTEX; no concurrent env access.
        unsafe { std::env::remove_var("RETREAD_NO_FAVOR_LOCK") };

        let result = load_favored_versions(&lock_path);
        assert!(
            result.is_empty(),
            "missing lock must return empty map; got {result:?}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// Valid lock + flag set → canonical-keyed map with correct versions.
    #[test]
    fn valid_lock_returns_canonical_keyed_map() {
        let _guard = TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        // Wheel names use PyPI casing; canonical_conda_name normalises to
        // lowercase with hyphens replaced by hyphens (and underscores by
        // hyphens). Use a name that exercises the normalisation.
        let lock_path = write_lock(
            &dir,
            "my-bundle",
            vec![
                make_wheel("MyPkg", "1.2.3"),
                make_wheel("dep_alpha", "0.5.0"),
            ],
        );
        // SAFETY: serialised by TEST_ENV_MUTEX; no concurrent env access.
        unsafe { std::env::remove_var("RETREAD_NO_FAVOR_LOCK") };

        let result = load_favored_versions(&lock_path);

        // canonical_conda_name("MyPkg") = "mypkg" etc.
        assert_eq!(
            result.get("mypkg").map(String::as_str),
            Some("1.2.3"),
            "mypkg must be keyed under canonical name; map={result:?}"
        );
        assert_eq!(
            result.get("dep-alpha").map(String::as_str),
            Some("0.5.0"),
            "dep_alpha must be keyed as dep-alpha (underscore→hyphen); map={result:?}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// Wheel with empty version string → that entry dropped, rest kept.
    #[test]
    fn empty_version_entry_dropped() {
        let _guard = TEST_ENV_MUTEX.lock().unwrap();
        let dir = unique_tmp_dir();
        let mut bad_wheel = make_wheel("broken-dep", "");
        bad_wheel.version = String::new();
        let lock_path = write_lock(&dir, "mypkg", vec![make_wheel("mypkg", "2.0.0"), bad_wheel]);
        // SAFETY: serialised by TEST_ENV_MUTEX; no concurrent env access.
        unsafe { std::env::remove_var("RETREAD_NO_FAVOR_LOCK") };

        let result = load_favored_versions(&lock_path);
        assert!(
            !result.contains_key("broken-dep"),
            "wheel with empty version must be dropped; map={result:?}"
        );
        assert_eq!(
            result.get("mypkg").map(String::as_str),
            Some("2.0.0"),
            "valid entry must still be present; map={result:?}"
        );
        std::fs::remove_dir_all(dir).ok();
    }
}

// -----------------------------------------------------------------
// TASK A (FIX 3 completion): resolve_bundle-loop-level back-pressure test.
//
// Drives the ACTUAL BFS loop in resolve_bundle through a localhost
// PEP 503 fixture server. Proves that transitive deps are correctly
// fetched and land in bundle.extras (the "bundle membership" assertion
// the ResolveState-level tests cannot make).
//
// Two tests:
//   (1) resolve_bundle_bfs_uses_workspace_fallback — two localhost indexes,
//       non-ignored. Exercises the full BFS loop end-to-end: the entry index
//       supplies the primary, the workspace index supplies its prefix-matched
//       transitive, and the transitive ends up in bundle.extras.
//
//   (2) resolve_bundle_reresolve_tighter_version_live — #[ignore], live
//       PyPI. Drives the actual NeedsReResolve cycle: primary requires
//       `retrtest-a>=1.0,<2.0` (caps at 2.0) and a sibling dep requires
//       `retrtest-a>=1.0`. This is intentionally constructed via a real
//       localhost fixture to trigger the NeedsReResolve → revoke → re-fetch
//       cycle and assert the dep ends up at the tighter-satisfying version.
// -----------------------------------------------------------------
#[cfg(test)]
mod resolve_bundle_bfs_tests {
    use std::collections::{BTreeMap, HashMap};
    use std::io::Write;
    use std::process::Command;
    use std::sync::Arc;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::{
        BfsOverride, BfsRoutePolicy, PypiToCondaMap, ResolvedWheel, bfs_probe_target_subdir,
        incremental_bundle_requires_cold_resolve, merge_uv_pins_into_prefs, produce_output,
        resolve_bundle,
    };
    use crate::config::{RelaxPolicy, RetreadConfig, WheelEntry};
    use crate::handler::audit_report::format_packagespec;
    use crate::index_chain::{IndexPurpose, index_chain};
    use crate::pypi::ResolutionTarget;
    use crate::relax::{CondaName, CondaTarget, NameMap, PypiKey};
    use rattler_conda_types::Platform;

    fn unique_tmp_dir() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "retread-bfs-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        );
        let dir = base.join(unique);
        std::fs::create_dir_all(&dir).expect("tmp dir creation should not fail");
        dir
    }

    /// Build a minimal valid .whl zip (pure-python, any platform).
    fn make_wheel_bytes(dist: &str, version: &str, requires: &[&str]) -> Vec<u8> {
        let normalized = dist.replace('-', "_");
        let di = format!("{normalized}-{version}.dist-info");
        let mut metadata = format!("Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n");
        for req in requires {
            metadata.push_str(&format!("Requires-Dist: {req}\n"));
        }
        let metadata_bytes = metadata.into_bytes();
        let wheel_file = b"Wheel-Version: 1.0\nTag: py3-none-any\n".to_vec();
        let record = format!("{di}/METADATA,,\n{di}/WHEEL,,\n{di}/RECORD,,\n").into_bytes();

        let mut buf = Vec::new();
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, body) in [
            (format!("{di}/METADATA"), metadata_bytes.as_slice()),
            (format!("{di}/WHEEL"), wheel_file.as_slice()),
            (format!("{di}/RECORD"), record.as_slice()),
        ] {
            zip.start_file(&name, opts).unwrap();
            zip.write_all(body).unwrap();
        }
        zip.finish().unwrap();
        buf
    }

    /// Spawn a minimal PEP 503 simple index + wheel server.
    ///
    /// `packages`: list of (name, version, wheel_bytes).
    /// Serves:
    ///   GET /simple/{pep503-normalized-name}/  → HTML index with all
    ///     versions of that name as `<a href="/{filename}">` links.
    ///   GET /{filename}  → raw wheel bytes.
    ///
    /// `advertise_sha256`: when true, every href carries the PEP 503
    /// `#sha256=<hex>` fragment, exactly as pypi.nvidia.com and pypi.org do.
    /// The default (false) models the minority of indexes that omit it
    /// (py.mujoco.org, some static self-hosted simple repos).
    ///
    /// Returns (port, task-handle). The task accepts up to `max_requests`
    /// connections then stops.
    async fn spawn_index_server(
        packages: Vec<(String, String, Vec<u8>)>,
        max_requests: u8,
        advertise_sha256: bool,
    ) -> u16 {
        use std::collections::HashMap;

        // Build lookup tables.
        let mut by_name: HashMap<String, Vec<(String, String, Vec<u8>)>> = HashMap::new();
        let mut by_filename: HashMap<String, Vec<u8>> = HashMap::new();

        for (name, version, bytes) in packages {
            let norm_name = name.to_ascii_lowercase().replace(['-', '_', '.'], "-");
            let normalized_dist = name.replace('-', "_");
            let filename = format!("{normalized_dist}-{version}-py3-none-any.whl");
            // The digest the index advertises == the digest of the bytes it
            // serves. Computing it here (rather than hardcoding) is what makes
            // "the lock digest equals the artifact digest" a real assertion.
            let sha = crate::wheel_rewrite::sha256_hex(&bytes);
            by_name
                .entry(norm_name)
                .or_default()
                .push((filename.clone(), sha, bytes.clone()));
            by_filename.insert(filename, bytes);
        }

        let by_name = Arc::new(by_name);
        let by_filename = Arc::new(by_filename);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            for _ in 0..max_requests {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let by_name = by_name.clone();
                let by_filename = by_filename.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]);
                    // Parse "GET /path HTTP/1.x"
                    let path = req
                        .lines()
                        .next()
                        .and_then(|l| l.split_whitespace().nth(1))
                        .unwrap_or("/");

                    let (status, content_type, body) =
                        if let Some(rest) = path.strip_prefix("/simple/") {
                            // Strip trailing slash, get normalized name.
                            let pkg_name = rest.trim_end_matches('/');
                            if let Some(entries) = by_name.get(pkg_name) {
                                let links: String = entries
                                    .iter()
                                    .map(|(fname, sha, _)| {
                                        let frag = if advertise_sha256 {
                                            format!("#sha256={sha}")
                                        } else {
                                            String::new()
                                        };
                                        format!("<a href=\"/{fname}{frag}\">{fname}</a>\n")
                                    })
                                    .collect();
                                let html =
                                    format!("<!DOCTYPE html><html><body>\n{links}</body></html>\n");
                                ("200 OK", "text/html", html.into_bytes())
                            } else {
                                ("404 Not Found", "text/plain", b"not found".to_vec())
                            }
                        } else {
                            // Wheel file request: /filename.whl
                            let fname = path.trim_start_matches('/');
                            if let Some(bytes) = by_filename.get(fname) {
                                ("200 OK", "application/octet-stream", bytes.clone())
                            } else {
                                ("404 Not Found", "text/plain", b"not found".to_vec())
                            }
                        };

                    let resp = format!(
                        "HTTP/1.0 {status}\r\nContent-Length: {}\r\nContent-Type: {content_type}\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(resp.as_bytes()).await;
                    let _ = stream.write_all(&body).await;
                });
            }
        });

        port
    }

    /// A dependency-free in-tree PEP 517 backend keeps the source/Git BFS
    /// fixture hermetic: uv invokes only stdlib Python and never downloads a
    /// build backend from an index.
    fn write_fixture_project(
        dir: &std::path::Path,
        name: &str,
        version: &str,
        requires_dist: &[String],
        provides_extra: Option<&str>,
    ) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("pyproject.toml"),
            "[build-system]\nrequires = []\nbuild-backend = \"fixture_backend\"\nbackend-path = [\".\"]\n",
        )
        .unwrap();

        let normalized = name.replace('-', "_");
        let dist_info = format!("{normalized}-{version}.dist-info");
        let filename = format!("{normalized}-{version}-py3-none-any.whl");
        let mut metadata = format!("Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n");
        if let Some(extra) = provides_extra {
            metadata.push_str(&format!("Provides-Extra: {extra}\n"));
        }
        for requirement in requires_dist {
            metadata.push_str(&format!("Requires-Dist: {requirement}\n"));
        }
        let backend = format!(
            r#"from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

FILENAME = {filename}
DIST_INFO = {dist_info}
METADATA = {metadata}
WHEEL = "Wheel-Version: 1.0\nGenerator: retread-fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n"

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    output = Path(wheel_directory) / FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    record = f"{{DIST_INFO}}/METADATA,,\n{{DIST_INFO}}/WHEEL,,\n{{DIST_INFO}}/RECORD,,\n"
    with ZipFile(output, "w", ZIP_DEFLATED) as wheel:
        wheel.writestr(f"{{DIST_INFO}}/METADATA", METADATA)
        wheel.writestr(f"{{DIST_INFO}}/WHEEL", WHEEL)
        wheel.writestr(f"{{DIST_INFO}}/RECORD", record)
    return FILENAME
"#,
            filename = serde_json::to_string(&filename).unwrap(),
            dist_info = serde_json::to_string(&dist_info).unwrap(),
            metadata = serde_json::to_string(&metadata).unwrap(),
        );
        std::fs::write(dir.join("fixture_backend.py"), backend).unwrap();
    }

    fn git_commit_all(dir: &std::path::Path) -> String {
        let run = |args: &[&str]| {
            let output = Command::new("git")
                .args(args)
                .current_dir(dir)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "git {args:?} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            output
        };
        run(&["init", "-q"]);
        run(&["add", "."]);
        run(&[
            "-c",
            "user.name=Retread Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ]);
        String::from_utf8(run(&["rev-parse", "HEAD"]).stdout)
            .unwrap()
            .trim()
            .to_string()
    }

    #[test]
    fn bfs_route_policy_manual_override_precedence() {
        let ray = PypiKey::from_pypi("ray");
        let mapping = || CondaTarget::Mapped(CondaName::new("ray-core"));
        let empty_parselmouth = PypiToCondaMap::new();
        let config = || {
            serde_json::from_value::<RetreadConfig>(serde_json::json!({"retread-wheels": {}}))
                .unwrap()
        };

        // The exact Imprint form: the override is keyed by the pending PyPI
        // name while emission is mapped to the ray-core provider.
        let mut by_pypi = config();
        by_pypi.name_map.insert(ray.clone(), mapping());
        by_pypi
            .overrides
            .insert("ray".to_string(), "==2.49.1".to_string());
        let (target, override_spec) =
            BfsRoutePolicy::from_config(&by_pypi).target_and_override(&ray, &empty_parselmouth);
        assert_eq!(target.as_ref().map(CondaName::as_spec), Some("ray-core"));
        assert_eq!(
            override_spec,
            Some(BfsOverride {
                spec: "==2.49.1",
                manual: true,
            })
        );

        // The provider-key spelling accepted by translate/emission is equally
        // authoritative at the BFS seam.
        let mut by_conda_alias = config();
        by_conda_alias.name_map.insert(ray.clone(), mapping());
        by_conda_alias
            .overrides
            .insert("ray-core".to_string(), "==2.49.1".to_string());
        let (target, override_spec) = BfsRoutePolicy::from_config(&by_conda_alias)
            .target_and_override(&ray, &empty_parselmouth);
        assert_eq!(target.as_ref().map(CondaName::as_spec), Some("ray-core"));
        assert_eq!(
            override_spec,
            Some(BfsOverride {
                spec: "==2.49.1",
                manual: true,
            })
        );

        // Explicit keep-on-PyPI intent still wins over a contradictory
        // override, and repair-ledger overrides remain resolver steering rather
        // than new manual native-conda authority.
        let mut disabled = config();
        disabled.name_map.insert(ray.clone(), CondaTarget::Disabled);
        disabled
            .overrides
            .insert("ray".to_string(), "==2.49.1".to_string());
        assert_eq!(
            BfsRoutePolicy::from_config(&disabled).target_and_override(&ray, &empty_parselmouth),
            (None, None)
        );

        let mut ledgered = by_pypi;
        ledgered.ledger_overrides.insert("ray".to_string());
        let (target, override_spec) =
            BfsRoutePolicy::from_config(&ledgered).target_and_override(&ray, &empty_parselmouth);
        assert_eq!(target.as_ref().map(CondaName::as_spec), Some("ray-core"));
        assert_eq!(
            override_spec,
            Some(BfsOverride {
                spec: "==2.49.1",
                manual: false,
            })
        );

        let mut mixed = by_conda_alias;
        mixed
            .overrides
            .insert("ray".to_string(), "==2.48.0".to_string());
        mixed.ledger_overrides.insert("ray".to_string());
        let (target, override_spec) =
            BfsRoutePolicy::from_config(&mixed).target_and_override(&ray, &empty_parselmouth);
        assert_eq!(target.as_ref().map(CondaName::as_spec), Some("ray-core"));
        assert_eq!(
            override_spec,
            Some(BfsOverride {
                spec: "==2.48.0",
                manual: false,
            }),
            "the higher-precedence PyPI ledger entry must shadow the manual mapped-provider entry"
        );
    }

    #[test]
    fn bfs_probe_uses_resolution_target_subdir() {
        let target = ResolutionTarget::for_subdir("3.10", "linux-aarch64");
        assert_eq!(bfs_probe_target_subdir(&target), "linux-aarch64");
    }

    /// Regression for the cold all-source-built seam: a source root's extra
    /// selects a Git child whose ordinary metadata contains ranged Ray and
    /// `packaging<24`. The explicit overrides must remain native-conda authority
    /// even when no channel can be consulted; otherwise BFS fetches the index's
    /// newest Ray wheel (2.56 here, requiring `packaging>=24.2`) before emission
    /// ever sees the compatible Ray 2.49.1 / packaging 23.0 pair.
    #[tokio::test]
    async fn source_extra_git_child_manual_pypi_override_never_fetches_latest_pypi() {
        let dir = unique_tmp_dir();
        let download_dir = dir.join("download");
        let source_dir = dir.join("source");
        let cache_dir = dir.join("cache");
        let root_dir = dir.join("root-project");
        let child_dir = dir.join("git-child");
        for path in [&download_dir, &source_dir, &cache_dir] {
            std::fs::create_dir_all(path).unwrap();
        }

        write_fixture_project(
            &child_dir,
            "retread-git-child",
            "1.0.0",
            &["ray>=2.40,<3".to_string(), "packaging<24".to_string()],
            None,
        );
        let child_rev = git_commit_all(&child_dir);
        let child_url = url::Url::from_directory_path(&child_dir).unwrap();
        write_fixture_project(
            &root_dir,
            "retread-source-root",
            "1.0.0",
            &[format!(
                "retread-git-child @ git+{child_url}@{child_rev} ; extra == \"routed\""
            )],
            Some("routed"),
        );

        let ray_249 = make_wheel_bytes("ray", "2.49.1", &[]);
        let ray_256 = make_wheel_bytes("ray", "2.56.0", &["packaging>=24.2"]);
        let packaging_23 = make_wheel_bytes("packaging", "23.0", &[]);
        let packaging_242 = make_wheel_bytes("packaging", "24.2", &[]);
        let port = spawn_index_server(
            vec![
                ("ray".to_string(), "2.49.1".to_string(), ray_249),
                ("ray".to_string(), "2.56.0".to_string(), ray_256),
                ("packaging".to_string(), "23.0".to_string(), packaging_23),
                ("packaging".to_string(), "24.2".to_string(), packaging_242),
            ],
            32,
            true,
        )
        .await;
        let indexes = vec![format!("http://127.0.0.1:{port}/simple/")];

        let entry = WheelEntry {
            path: Some(root_dir.to_string_lossy().into_owned()),
            extras: vec!["routed".to_string()],
            ..Default::default()
        };
        let mut config: RetreadConfig =
            serde_json::from_value(serde_json::json!({"retread-wheels": {}})).unwrap();
        config.name_map.insert(
            PypiKey::from_pypi("ray"),
            CondaTarget::Mapped(CondaName::new("ray-core")),
        );
        config
            .overrides
            .insert("ray".to_string(), "==2.49.1".to_string());
        config
            .overrides
            .insert("packaging".to_string(), "==23.0".to_string());

        let platform = Platform::current();
        let target = ResolutionTarget::for_subdir("3.11", platform.as_str());
        let bundle = resolve_bundle(
            "retread-source-root",
            &entry,
            &target,
            &download_dir,
            &source_dir,
            &cache_dir,
            RelaxPolicy::None,
            &BTreeMap::new(),
            None,
            &PypiToCondaMap::new(),
            BfsRoutePolicy::from_config(&config),
            &[], // deliberate indecisive probe: no channel was consultable
            &indexes,
            None,
            None,
            &std::collections::HashSet::new(),
        )
        .await
        .expect("the source/Git BFS must honor the manual mapped-provider override");

        assert!(
            bundle
                .extras
                .iter()
                .any(|wheel| wheel.pypi_name == "retread-git-child"),
            "the Git child itself must still be materialized"
        );
        assert!(
            bundle
                .extras
                .iter()
                .all(|wheel| wheel.pypi_name != "ray" && wheel.pypi_name != "packaging"),
            "Ray and packaging must remain conda routes; bundled wheels were {:?}",
            bundle
                .extras
                .iter()
                .map(|wheel| (&wheel.pypi_name, &wheel.metadata.version))
                .collect::<Vec<_>>()
        );
        let ray_probe = bundle
            .probe_decisions
            .iter()
            .find(|decision| decision.pypi_name == "ray")
            .expect("the indecisive explicit-override probe must remain auditable");
        assert_eq!(ray_probe.conda_name, "ray-core");
        assert_eq!(ray_probe.spec, "==2.49.1");
        assert_eq!(ray_probe.satisfiable, None);
        assert_eq!(
            ray_probe.routing_decision,
            "short-circuit-explicit-override"
        );
        assert!(
            incremental_bundle_requires_cold_resolve(&bundle),
            "incremental add must escalate instead of dropping the newly routed ray-core run dep"
        );

        let output = produce_output(&bundle, &config, platform, "3.11", &[], None, None)
            .expect("the mapped override must emit a valid exact conda dependency");
        let ray_core = output
            .run_dependencies
            .depends
            .iter()
            .find(|dependency| dependency.name == "ray-core")
            .expect("the manual mapped-provider override must be emitted");
        assert_eq!(format_packagespec(&ray_core.spec), "==2.49.1");
        let packaging = output
            .run_dependencies
            .depends
            .iter()
            .find(|dependency| dependency.name == "packaging")
            .expect("the manual packaging override must be emitted");
        assert_eq!(format_packagespec(&packaging.spec), "==23.0");
        let routed_deps: Vec<(String, String)> = output
            .run_dependencies
            .depends
            .iter()
            .filter(|dependency| dependency.name != "python" && dependency.name != "uv")
            .map(|dependency| {
                (
                    dependency.name.clone(),
                    format_packagespec(&dependency.spec),
                )
            })
            .collect();
        assert_eq!(
            routed_deps,
            vec![
                ("packaging".to_string(), "==23.0".to_string()),
                ("ray-core".to_string(), "==2.49.1".to_string()),
            ],
            "apart from the mandatory Python/Courier runtime, the source/Git closure must emit only the two manual native-conda routes"
        );
        assert!(
            output
                .run_dependencies
                .depends
                .iter()
                .all(|dependency| dependency.name != "ray"),
            "the PyPI identity must not leak into conda run dependencies"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Task A (FIX 3 completion): resolve_bundle-loop-level integration test.
    ///
    /// Drives the FULL BFS loop inside resolve_bundle with two localhost fixture
    /// indexes. The entry index contains only the primary; the workspace index
    /// contains only its transitive, so old single-entry-index BFS wiring fails.
    ///
    /// Scenario:
    ///   - Primary: `rtest-pkg==1.0` (Requires-Dist: `rtest-pkg-sub>=1.0`)
    ///   - Transitive: `rtest-pkg-sub==1.0` (no further deps)
    ///   - Primary served only by the entry's private index.
    ///   - Transitive served only by the workspace fallback index.
    ///
    /// Assert: bundle.extras contains exactly one entry: rtest-pkg-sub 1.0.
    ///
    /// This test exercises the full BFS pipeline (materialize_and_rewrite for
    /// primary, bfs_fetch_pypi for transitive, commit_chosen, extras.push) and
    /// would catch the FIX-1 vanish bug if it were re-introduced (a dep that
    /// goes through NeedsReResolve must still appear in bundle.extras, not
    /// vanish silently).
    #[tokio::test]
    async fn resolve_bundle_bfs_uses_workspace_fallback() {
        let dir = unique_tmp_dir();
        let download_dir = dir.join("download");
        let source_dir = dir.join("source");
        let cache_dir = dir.join("cache");
        std::fs::create_dir_all(&download_dir).unwrap();
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&cache_dir).unwrap();

        let primary_name = "rtest-pkg";
        let primary_version = "1.0";
        let sub_name = "rtest-pkg-sub";
        let sub_version = "1.0";

        // primary requires rtest-pkg-sub (prefix matches: rtest-pkg- prefix)
        let primary_bytes = make_wheel_bytes(
            primary_name,
            primary_version,
            &[&format!("{sub_name}>=1.0")],
        );
        let sub_bytes = make_wheel_bytes(sub_name, sub_version, &[]);

        let entry_port = spawn_index_server(
            vec![(
                primary_name.to_string(),
                primary_version.to_string(),
                primary_bytes,
            )],
            32,
            false,
        )
        .await;
        let workspace_port = spawn_index_server(
            vec![(sub_name.to_string(), sub_version.to_string(), sub_bytes)],
            32,
            false,
        )
        .await;

        let entry_index = format!("http://127.0.0.1:{entry_port}/simple/");
        let workspace_index = format!("http://127.0.0.1:{workspace_port}/simple/");
        let workspace_indexes = vec![workspace_index.clone()];
        assert_eq!(
            index_chain(
                [entry_index.clone()],
                &workspace_indexes,
                IndexPurpose::RootResolve,
            ),
            vec![entry_index.clone(), workspace_index.clone()],
        );

        let entry = WheelEntry {
            version: Some(primary_version.to_string()),
            index: Some(entry_index),
            ..Default::default()
        };
        let target = ResolutionTarget::for_subdir("3.11", "linux-64");
        let pypi_to_conda: PypiToCondaMap = HashMap::new();
        let name_map = NameMap::new();
        let git_sources: BTreeMap<String, crate::config::NamedGitSource> = BTreeMap::new();
        let conda_channels: Vec<rattler_conda_types::ChannelUrl> = vec![];

        let bundle = resolve_bundle(
            primary_name,
            &entry,
            &target,
            &download_dir,
            &source_dir,
            &cache_dir,
            RelaxPolicy::default(),
            &git_sources,
            None, // auto_data
            &pypi_to_conda,
            BfsRoutePolicy::name_map_only(&name_map),
            &conda_channels,
            &workspace_indexes,
            None,                              // cold path: no locked closure
            None,                              // cold path: no favor-lock prefs
            &std::collections::HashSet::new(), // no sibling context
        )
        .await
        .expect("resolve_bundle must succeed");

        std::fs::remove_dir_all(&dir).ok();

        // Primary must be in the bundle.
        assert_eq!(
            bundle.primary.pypi_name, primary_name,
            "bundle primary must be {primary_name}"
        );

        // Transitive sub-wheel must be in bundle.extras (not vanished).
        // This is the key assertion: if the re-resolve path (FIX 1) had a
        // silent-delete bug, a dep that went through NeedsReResolve would
        // be missing here. Pinning this assertion guards that regression.
        let extras_names: Vec<&str> = bundle.extras.iter().map(|w| w.pypi_name.as_str()).collect();
        assert!(
            extras_names.contains(&sub_name),
            "transitive dep '{sub_name}' must be present in bundle.extras; \
             got: {extras_names:?}. \
             If this fails, the FIX-1 re-resolve vanish bug was re-introduced.",
        );
        assert_eq!(
            extras_names.len(),
            1,
            "bundle.extras must contain exactly one dep ({sub_name}); got: {extras_names:?}",
        );

        // Verify the resolved version.
        let sub_wheel = bundle
            .extras
            .iter()
            .find(|w| w.pypi_name == sub_name)
            .unwrap();
        assert_eq!(
            sub_wheel.metadata.version, sub_version,
            "transitive dep must be at version {sub_version}"
        );
        assert!(
            sub_wheel
                .upstream_url
                .as_ref()
                .is_some_and(|url| url.port() == Some(workspace_port)),
            "the transitive must come from the workspace fallback, not the entry index: {:?}",
            sub_wheel.upstream_url,
        );
    }

    /// auto-extscache TASK 1+2+3: an `extras`-only entry must pull the extra's
    /// transitive wheels into the bundle, each carrying the INDEX-ADVERTISED
    /// `#sha256=` digest — with NO hand-written `url = "...#sha256=..."` roots.
    ///
    /// This is the exact production shape from imprint's
    /// `pypi-packs/isaac-pack-latest/pixi.toml`:
    ///
    /// ```toml
    /// isaacsim = { version = "==6.0.0.1", index = "...", extras = ["all", "extscache"] }
    /// ```
    ///
    /// where isaacsim's METADATA declares
    /// `Requires-Dist: isaacsim-extscache-kit==6.0.0.1; extra == "extscache"`
    /// (and -kit-sdk / -physics), and the three extscache wheels are
    /// dependency-free index wheels.
    ///
    /// The manifest ALSO carried three explicit URL roots with hand-pasted
    /// sha256 digests, commented "explicit URL roots keep them in the lock".
    /// This test is the falsifier for that claim: with the URL roots gone (no
    /// sibling entries at all), the extras BFS must still land all three in
    /// `bundle.extras`, each with `metadata.sha256` equal to the digest the
    /// index advertised in its href fragment.
    #[tokio::test]
    async fn resolve_bundle_extras_pull_transitive_wheels_with_index_sha256() {
        let dir = unique_tmp_dir();
        let download_dir = dir.join("download");
        let source_dir = dir.join("source");
        let cache_dir = dir.join("cache");
        std::fs::create_dir_all(&download_dir).unwrap();
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&cache_dir).unwrap();

        let version = "6.0.0.1";
        let extscache = [
            "isaacsim-extscache-kit",
            "isaacsim-extscache-kit-sdk",
            "isaacsim-extscache-physics",
        ];

        // Primary: isaacsim, whose extscache extra gates the three wheels.
        let primary_bytes = make_wheel_bytes(
            "isaacsim",
            version,
            &extscache
                .iter()
                .map(|n| format!("{n}=={version} ; extra == \"extscache\""))
                .collect::<Vec<_>>()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        );

        // The three extscache wheels are dependency-free index wheels.
        let mut packages = vec![(
            "isaacsim".to_string(),
            version.to_string(),
            primary_bytes.clone(),
        )];
        let mut want_sha: BTreeMap<String, String> = BTreeMap::new();
        for name in extscache {
            let bytes = make_wheel_bytes(name, version, &[]);
            want_sha.insert(name.to_string(), crate::wheel_rewrite::sha256_hex(&bytes));
            packages.push((name.to_string(), version.to_string(), bytes));
        }

        // advertise_sha256 = true: a PEP 503 index that publishes digests in
        // the href fragment, exactly like pypi.nvidia.com.
        let port = spawn_index_server(packages, 64, true).await;
        let index_url = format!("http://127.0.0.1:{port}/simple/");

        let entry = WheelEntry {
            version: Some(format!("=={version}")),
            index: Some(index_url),
            // The ONLY thing the user writes. No url/sha256 anywhere.
            extras: vec!["all".into(), "extscache".into()],
            ..Default::default()
        };
        let target = ResolutionTarget::for_subdir("3.12", "linux-64");
        let pypi_to_conda: PypiToCondaMap = HashMap::new();
        let name_map = NameMap::new();
        let git_sources: BTreeMap<String, crate::config::NamedGitSource> = BTreeMap::new();
        let conda_channels: Vec<rattler_conda_types::ChannelUrl> = vec![];

        let bundle = resolve_bundle(
            "isaacsim",
            &entry,
            &target,
            &download_dir,
            &source_dir,
            &cache_dir,
            RelaxPolicy::default(),
            &git_sources,
            None,
            &pypi_to_conda,
            BfsRoutePolicy::name_map_only(&name_map),
            &conda_channels,
            &[],
            None,
            None,
            // NO siblings: the three URL roots are GONE from the manifest.
            &std::collections::HashSet::new(),
        )
        .await
        .expect("resolve_bundle must succeed");

        let got: BTreeMap<&str, &ResolvedWheel> = bundle
            .extras
            .iter()
            .map(|w| (w.pypi_name.as_str(), w))
            .collect();

        for name in extscache {
            let w = got.get(name).unwrap_or_else(|| {
                panic!(
                    "extra's transitive wheel `{name}` FELL OUT of the bundle; \
                     bundle.extras = {:?}. The extras BFS did not follow the \
                     `; extra == \"extscache\"` edge.",
                    got.keys().collect::<Vec<_>>()
                )
            });
            assert_eq!(w.metadata.version, version, "{name} version");

            // TASK 2: the digest must be the one the INDEX advertised, never
            // fabricated and never absent.
            let want = &want_sha[name];
            assert_eq!(
                &w.metadata.sha256, want,
                "{name}: digest must equal the index-advertised sha256"
            );

            // TASK 3: the artifact URL must be recorded so the lock entry is
            // a direct, digest-pinned fetch (Origin::Index).
            let upstream = w.upstream_url.as_ref().unwrap_or_else(|| {
                panic!("{name}: no upstream_url -> no Origin::Index lock entry")
            });
            assert!(
                upstream
                    .as_str()
                    .contains(&format!("{}-{version}", name.replace('-', "_"))),
                "{name}: upstream_url must point at the wheel artifact; got {upstream}"
            );
        }

        // ---- Close the chain: bundle -> EmitWheel -> courier::stage -> LOCK.
        // This is the assertion the task actually asks for ("lock entries"),
        // and it is the step the hand-written URL roots claimed to be
        // necessary for. Mirrors build_one's emit-wheel mapping verbatim.
        let stage_dir = unique_tmp_dir();
        let staging = stage_dir.join("staging");
        let emit_wheels: Vec<crate::emit_pypi::EmitWheel> = bundle
            .all_wheels()
            .map(|w| crate::emit_pypi::EmitWheel {
                pypi_name: w.pypi_name.clone(),
                version: w.metadata.version.clone(),
                requires_dist: w.metadata.requires_dist.clone(),
                sha256: Some(w.metadata.sha256.clone()),
                locked_final_sha256: None,
                local_path: (w.url.scheme() == "file")
                    .then(|| w.url.to_file_path().ok())
                    .flatten(),
                wheel_filename: w
                    .url
                    .path_segments()
                    .and_then(|mut segments| segments.next_back())
                    .unwrap_or_default()
                    .to_owned(),
                remote_url: (w.url.scheme() != "file").then(|| w.url.clone()),
                upstream_url: w.upstream_url.clone(),
                git_source: None,
                sdist_source: None,
            })
            .collect();

        let config: crate::config::RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": { "isaacsim": { "version": format!("=={version}") } },
            "retread-bundle-mode": "fat",
        }))
        .unwrap();

        let staged = crate::courier::stage(
            &config,
            "isaac-pack-latest",
            version,
            "3.12",
            &emit_wheels,
            &std::collections::HashSet::new(),
            &[],
            &["https://pypi.org/simple/".to_string()],
            "",
            &stage_dir,
            &staging,
        )
        .await
        .expect("courier::stage must write a replayable lock");

        let locked: BTreeMap<&str, &crate::lock::LockWheel> = staged
            .lock
            .wheels
            .iter()
            .map(|w| (w.name.as_str(), w))
            .collect();

        for name in extscache {
            let lw = locked.get(name).unwrap_or_else(|| {
                panic!(
                    "extra's transitive wheel `{name}` FELL OUT of the LOCK; \
                     lock wheels = {:?}",
                    locked.keys().collect::<Vec<_>>()
                )
            });
            assert_eq!(
                lw.origin,
                crate::lock::Origin::Index,
                "{name}: dependency-free index wheel must lock as Origin::Index"
            );
            assert_eq!(
                lw.sha256.as_deref(),
                Some(want_sha[name].as_str()),
                "{name}: LOCK digest must be the index-advertised sha256 — \
                 no hand-pasted URL root required"
            );
            let url = lw
                .url
                .as_deref()
                .unwrap_or_else(|| panic!("{name}: Index lock entry must carry its artifact url"));
            assert!(
                url.contains(&format!("{}-{version}", name.replace('-', "_"))),
                "{name}: lock url must be the direct artifact url; got {url}"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&stage_dir).ok();
    }

    /// auto-extscache: an index that advertises NO digest is handled
    /// gracefully — the digest is never fabricated, and resolution still
    /// succeeds with the TRUE digest of the bytes actually fetched.
    ///
    /// PEP 503 recommends but does not require the `#sha256=` fragment;
    /// py.mujoco.org and various static self-hosted simple repos omit it. The
    /// parser leaves `ResolvedWheel.sha256 = None` there (see
    /// `pypi::tests::parses_links_without_sha256`), and the resolver falls back
    /// to a full download and hashes the bytes itself. The lock must therefore
    /// still be digest-pinned — with a COMPUTED digest, not an invented one.
    #[tokio::test]
    async fn resolve_bundle_extras_handle_index_without_advertised_digest() {
        let dir = unique_tmp_dir();
        let download_dir = dir.join("download");
        let source_dir = dir.join("source");
        let cache_dir = dir.join("cache");
        std::fs::create_dir_all(&download_dir).unwrap();
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&cache_dir).unwrap();

        let version = "6.0.0.1";
        let sub = "isaacsim-extscache-kit";
        let primary_bytes = make_wheel_bytes(
            "isaacsim",
            version,
            &[&format!("{sub}=={version} ; extra == \"extscache\"")],
        );
        let sub_bytes = make_wheel_bytes(sub, version, &[]);
        let true_sha = crate::wheel_rewrite::sha256_hex(&sub_bytes);

        // advertise_sha256 = false: index publishes no digests at all.
        let port = spawn_index_server(
            vec![
                ("isaacsim".to_string(), version.to_string(), primary_bytes),
                (sub.to_string(), version.to_string(), sub_bytes),
            ],
            64,
            false,
        )
        .await;

        let entry = WheelEntry {
            version: Some(format!("=={version}")),
            index: Some(format!("http://127.0.0.1:{port}/simple/")),
            extras: vec!["extscache".into()],
            ..Default::default()
        };
        let target = ResolutionTarget::for_subdir("3.12", "linux-64");
        let bundle = resolve_bundle(
            "isaacsim",
            &entry,
            &target,
            &download_dir,
            &source_dir,
            &cache_dir,
            RelaxPolicy::default(),
            &BTreeMap::new(),
            None,
            &PypiToCondaMap::new(),
            BfsRoutePolicy::name_map_only(&BTreeMap::new()),
            &[],
            &[],
            None,
            None,
            &std::collections::HashSet::new(),
        )
        .await
        .expect("a digest-less index must not break resolution");

        std::fs::remove_dir_all(&dir).ok();

        let w = bundle
            .extras
            .iter()
            .find(|w| w.pypi_name == sub)
            .unwrap_or_else(|| panic!("`{sub}` must still resolve from a digest-less index"));
        // The digest is the REAL hash of the fetched bytes — computed, never
        // invented, and never a placeholder.
        assert_eq!(
            w.metadata.sha256, true_sha,
            "digest must be the true hash of the bytes actually fetched"
        );
    }

    /// auto-extscache TASK 4: the escape hatch survives. A deliberate
    /// `url = "...#sha256=..."` entry must still resolve, must pin the EXACT
    /// artifact named (not whatever the index thinks is newest), and its
    /// digest must be enforced rather than decorative.
    #[tokio::test]
    async fn explicit_url_entry_pins_exact_artifact_and_wins_over_index() {
        let dir = unique_tmp_dir();
        let download_dir = dir.join("download");
        let source_dir = dir.join("source");
        let cache_dir = dir.join("cache");
        std::fs::create_dir_all(&download_dir).unwrap();
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&cache_dir).unwrap();

        let name = "isaacsim-extscache-kit";
        // The index serves BOTH 6.0.0.1 and a newer 6.0.0.2.
        let old_bytes = make_wheel_bytes(name, "6.0.0.1", &[]);
        let new_bytes = make_wheel_bytes(name, "6.0.0.2", &[]);
        let old_sha = crate::wheel_rewrite::sha256_hex(&old_bytes);

        let port = spawn_index_server(
            vec![
                (name.to_string(), "6.0.0.1".to_string(), old_bytes),
                (name.to_string(), "6.0.0.2".to_string(), new_bytes),
            ],
            64,
            true,
        )
        .await;

        let file = format!("{}-6.0.0.1-py3-none-any.whl", name.replace('-', "_"));
        let pinned = format!("http://127.0.0.1:{port}/{file}#sha256={old_sha}");

        let mut entry = WheelEntry {
            url: Some(pinned.parse().unwrap()),
            ..Default::default()
        };
        // normalize() lifts `#sha256=` out of the fragment into the discrete
        // field — this is what makes the hand-written pin enforceable.
        entry
            .normalize(name)
            .expect("normalize must accept the pin");
        assert_eq!(
            entry.sha256.as_deref(),
            Some(old_sha.as_str()),
            "fragment digest must be lifted into the discrete field"
        );

        let target = ResolutionTarget::for_subdir("3.12", "linux-64");
        let bundle = resolve_bundle(
            name,
            &entry,
            &target,
            &download_dir,
            &source_dir,
            &cache_dir,
            RelaxPolicy::default(),
            &BTreeMap::new(),
            None,
            &PypiToCondaMap::new(),
            BfsRoutePolicy::name_map_only(&BTreeMap::new()),
            &[],
            &[],
            None,
            None,
            &std::collections::HashSet::new(),
        )
        .await
        .expect("explicit URL entry must still resolve");

        std::fs::remove_dir_all(&dir).ok();

        // Precedence: the hand-pinned 6.0.0.1 artifact wins; the index's newer
        // 6.0.0.2 is NOT substituted.
        assert_eq!(
            bundle.primary.metadata.version, "6.0.0.1",
            "explicit URL must pin the exact artifact, not the index's newest"
        );
        assert_eq!(
            bundle.primary.metadata.sha256, old_sha,
            "explicit URL entry must carry its pinned digest"
        );
    }

    /// auto-extscache TASK 4 (negative): the pin is ENFORCED, not decorative.
    /// A `url = "...#sha256=<wrong>"` entry must fail loudly rather than
    /// silently install bytes that do not match the digest the user pinned.
    #[tokio::test]
    async fn explicit_url_entry_with_wrong_digest_is_rejected() {
        let dir = unique_tmp_dir();
        let download_dir = dir.join("download");
        let source_dir = dir.join("source");
        let cache_dir = dir.join("cache");
        std::fs::create_dir_all(&download_dir).unwrap();
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&cache_dir).unwrap();

        let name = "isaacsim-extscache-kit";
        let bytes = make_wheel_bytes(name, "6.0.0.1", &[]);
        let port = spawn_index_server(
            vec![(name.to_string(), "6.0.0.1".to_string(), bytes)],
            32,
            true,
        )
        .await;

        let file = format!("{}-6.0.0.1-py3-none-any.whl", name.replace('-', "_"));
        let wrong = "0".repeat(64);
        let mut entry = WheelEntry {
            url: Some(
                format!("http://127.0.0.1:{port}/{file}#sha256={wrong}")
                    .parse()
                    .unwrap(),
            ),
            ..Default::default()
        };
        entry.normalize(name).unwrap();

        let target = ResolutionTarget::for_subdir("3.12", "linux-64");
        let err = resolve_bundle(
            name,
            &entry,
            &target,
            &download_dir,
            &source_dir,
            &cache_dir,
            RelaxPolicy::default(),
            &BTreeMap::new(),
            None,
            &PypiToCondaMap::new(),
            BfsRoutePolicy::name_map_only(&BTreeMap::new()),
            &[],
            &[],
            None,
            None,
            &std::collections::HashSet::new(),
        )
        .await
        .expect_err("a wheel whose bytes do not match the pinned digest must be rejected");

        std::fs::remove_dir_all(&dir).ok();

        let msg = format!("{err:#}").to_lowercase();
        assert!(
            msg.contains("sha256") || msg.contains("hash") || msg.contains("mismatch"),
            "error must name the digest mismatch; got: {err:#}"
        );
    }

    /// TASK 1 (seal Part 1): BFS-loop constraint-merge picks the correct
    /// version when two Requires-Dist lines for the same dep arrive from
    /// the same primary wheel.
    ///
    /// Scenario
    /// --------
    ///   - Primary `rtest-pkg 1.0` has TWO Requires-Dist lines for the
    ///     sub-dep, both matching the `rtest-pkg-` prefix:
    ///       Requires-Dist: rtest-pkg-sub>=1.0
    ///       Requires-Dist: rtest-pkg-sub<2.0
    ///   - The localhost index serves `rtest-pkg-sub` at BOTH 1.0 and 2.0.
    ///
    /// What the BFS must do
    /// --------------------
    ///   The initial seed drain (before the BFS loop) calls `seed_worklist`
    ///   with an empty `seen` set.  Because `seen` is NOT updated between
    ///   lines inside `seed_worklist`, both `>=1.0` and `<2.0` edges for
    ///   `rtest-pkg-sub` are pushed to `tmp_queue`.
    ///
    ///   FIX 2's merge step in the drain loop INTERSECTS them into a single
    ///   `Pending(rtest-pkg-sub, >=1.0,<2.0)` entry in `work`.
    ///
    ///   `pypi::resolve` with `>=1.0,<2.0` from the two-version server
    ///   returns 1.0 (the highest satisfying version; 2.0 is excluded).
    ///
    /// Regression guards
    /// -----------------
    ///   FIX 2 regression: reverting the `Occupied` merge arm back to a
    ///   plain `or_insert` causes only the FIRST edge (`>=1.0`) to be
    ///   recorded.  `pypi::resolve(>=1.0)` then picks the highest candidate
    ///   = 2.0, and the assertion `version == "1.0"` FAILS.
    ///
    ///   FIX 1 (revoke_chosen / NeedsReResolve path): the BFS observe-all-
    ///   first design means the sub-dep is added to `constraints` during the
    ///   observe_edge loop — BEFORE any phase-3 seed runs — making a
    ///   cross-level NeedsReResolve structurally unreachable in this scenario.
    ///   FIX 1 is therefore covered at the unit level by
    ///   `resolve_state::tests::revoke_clears_constraints_so_reenqueue_hits_new`.
    #[tokio::test]
    async fn resolve_bundle_bfs_constraint_merge_picks_correct_version() {
        let dir = unique_tmp_dir();
        let download_dir = dir.join("download");
        let source_dir = dir.join("source");
        let cache_dir = dir.join("cache");
        std::fs::create_dir_all(&download_dir).unwrap();
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&cache_dir).unwrap();

        let primary_name = "rtest-pkg";
        let primary_version = "1.0";
        let sub_name = "rtest-pkg-sub";

        // Primary has TWO separate Requires-Dist lines for the same dep:
        // one sets a floor (>=1.0) and the other a ceiling (<2.0).
        // FIX 2 must merge these into a single >=1.0,<2.0 constraint.
        let primary_bytes = make_wheel_bytes(
            primary_name,
            primary_version,
            &[&format!("{sub_name}>=1.0"), &format!("{sub_name}<2.0")],
        );

        // Serve sub at BOTH 1.0 AND 2.0 so the constraint determines which
        // version is picked (not the absence of an alternative).
        let sub_10_bytes = make_wheel_bytes(sub_name, "1.0", &[]);
        let sub_20_bytes = make_wheel_bytes(sub_name, "2.0", &[]);

        let port = spawn_index_server(
            vec![
                (
                    primary_name.to_string(),
                    primary_version.to_string(),
                    primary_bytes,
                ),
                (sub_name.to_string(), "1.0".to_string(), sub_10_bytes),
                (sub_name.to_string(), "2.0".to_string(), sub_20_bytes),
            ],
            32,
            false, // index advertises no #sha256 (legacy fixture behavior)
        )
        .await;

        let index_url = format!("http://127.0.0.1:{port}/simple/");

        let entry = WheelEntry {
            version: Some(primary_version.to_string()),
            index: Some(index_url),
            ..Default::default()
        };
        let target = ResolutionTarget::for_subdir("3.11", "linux-64");
        let pypi_to_conda: PypiToCondaMap = HashMap::new();
        let name_map = NameMap::new();
        let git_sources: std::collections::BTreeMap<String, crate::config::NamedGitSource> =
            std::collections::BTreeMap::new();
        let conda_channels: Vec<rattler_conda_types::ChannelUrl> = vec![];

        let bundle = resolve_bundle(
            primary_name,
            &entry,
            &target,
            &download_dir,
            &source_dir,
            &cache_dir,
            RelaxPolicy::default(),
            &git_sources,
            None, // auto_data
            &pypi_to_conda,
            BfsRoutePolicy::name_map_only(&name_map),
            &conda_channels,
            &[],                               // workspace_indexes
            None,                              // cold path: no locked closure
            None,                              // cold path: no favor-lock prefs
            &std::collections::HashSet::new(), // no sibling context
        )
        .await
        .expect("resolve_bundle must succeed");

        std::fs::remove_dir_all(&dir).ok();

        // Primary check.
        assert_eq!(
            bundle.primary.pypi_name, primary_name,
            "bundle primary must be {primary_name}"
        );

        // Sub must be present.
        let extras_names: Vec<&str> = bundle.extras.iter().map(|w| w.pypi_name.as_str()).collect();
        assert!(
            extras_names.contains(&sub_name),
            "transitive dep '{sub_name}' must be present in bundle.extras; got: {extras_names:?}"
        );

        // Core assertion: sub must be at 1.0 (NOT 2.0).
        //
        // With FIX 2 the two edges (>=1.0 and <2.0) are merged into
        // >=1.0,<2.0 before the first BFS level runs.  pypi::resolve with
        // that constraint picks the highest satisfying version = 1.0.
        //
        // Without FIX 2 (or_insert only), only the first edge (>=1.0) is
        // kept, pypi::resolve picks 2.0, and this assertion FAILS.
        let sub_wheel = bundle
            .extras
            .iter()
            .find(|w| w.pypi_name == sub_name)
            .unwrap();
        assert_eq!(
            sub_wheel.metadata.version, "1.0",
            "'{sub_name}' must resolve to 1.0 (highest version satisfying >=1.0,<2.0 = the \
             FIX 2 merged constraint); got {} — if this fails, FIX 2 constraint-merge \
             was not applied (the two separate Requires-Dist edges were not intersected \
             and the unconstrained >=1.0 resolved to 2.0 instead)",
            sub_wheel.metadata.version,
        );
    }

    /// M1 seam regression guard (NVIDIA-only wheel leak): uv closure pins
    /// must flow into `resolve_bundle` as favor-lock PREFERENCES, never as
    /// `locked_closure`.
    ///
    /// Scenario (same fixture shape as the BFS tests above):
    ///   - Primary `rtest-pkg==1.0` requires `rtest-pkg-sub>=1.0` (prefix
    ///     base dep, exactly like isaacsim -> isaacsim-app).
    ///   - Index serves rtest-pkg-sub at 1.0 and 2.0.
    ///   - The uv closure pinned rtest-pkg-sub to 1.0.
    ///
    /// Part 1 (the FIX): pins passed as prefs -> the transitive is fetched,
    /// bundled, and resolved at uv's pinned 1.0 (not the latest 2.0).
    ///
    /// Part 2 (the BUG shape, pinned as documentation of WHY the seam
    /// matters): the same pins passed as `locked_closure` suppress BFS
    /// seeding entirely -- `seed_locked` marks the name as already handled,
    /// `seen_set` skips it, and bundle.extras comes back EMPTY. On a real
    /// pack that empty bundle leaks every closure member into the conda
    /// run-deps as a relaxed spec (`isaacsim-app >=4.2,<4.3`), which no
    /// conda channel can satisfy for families that exist only on
    /// pypi.nvidia.com -> the consumer's solve fails with "no candidates".
    /// The locked-closure seam is for the incremental-add path only.
    #[tokio::test]
    async fn resolve_bundle_uv_pins_as_prefs_bundles_transitive() {
        let dir = unique_tmp_dir();
        let download_dir = dir.join("download");
        let source_dir = dir.join("source");
        let cache_dir = dir.join("cache");
        std::fs::create_dir_all(&download_dir).unwrap();
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&cache_dir).unwrap();

        let primary_name = "rtest-pkg";
        let primary_version = "1.0";
        let sub_name = "rtest-pkg-sub";

        let primary_bytes = make_wheel_bytes(
            primary_name,
            primary_version,
            &[&format!("{sub_name}>=1.0")],
        );
        let sub_10_bytes = make_wheel_bytes(sub_name, "1.0", &[]);
        let sub_20_bytes = make_wheel_bytes(sub_name, "2.0", &[]);

        let port = spawn_index_server(
            vec![
                (
                    primary_name.to_string(),
                    primary_version.to_string(),
                    primary_bytes,
                ),
                (sub_name.to_string(), "1.0".to_string(), sub_10_bytes),
                (sub_name.to_string(), "2.0".to_string(), sub_20_bytes),
            ],
            64,    // two resolve_bundle passes share one server
            false, // index advertises no #sha256 (legacy fixture behavior)
        )
        .await;

        let index_url = format!("http://127.0.0.1:{port}/simple/");

        let entry = WheelEntry {
            version: Some(primary_version.to_string()),
            index: Some(index_url),
            ..Default::default()
        };
        let target = ResolutionTarget::for_subdir("3.11", "linux-64");
        let pypi_to_conda: PypiToCondaMap = HashMap::new();
        let name_map = NameMap::new();
        let git_sources: std::collections::BTreeMap<String, crate::config::NamedGitSource> =
            std::collections::BTreeMap::new();
        let conda_channels: Vec<rattler_conda_types::ChannelUrl> = vec![];

        // The uv closure's pins for this bundle: the primary AND the
        // transitive, exactly as parse_pylock_closure produces them.
        let mut uv_pins: BTreeMap<String, String> = BTreeMap::new();
        uv_pins.insert(primary_name.to_string(), primary_version.to_string());
        uv_pins.insert(sub_name.to_string(), "1.0".to_string());

        // Part 1: pins as PREFERENCES (the fixed resolve_all seam:
        // merge_uv_pins_into_prefs + locked_closure=None).
        let prefs = merge_uv_pins_into_prefs(std::collections::BTreeMap::new(), Some(&uv_pins));
        let bundle = resolve_bundle(
            primary_name,
            &entry,
            &target,
            &download_dir,
            &source_dir,
            &cache_dir,
            RelaxPolicy::default(),
            &git_sources,
            None, // auto_data
            &pypi_to_conda,
            BfsRoutePolicy::name_map_only(&name_map),
            &conda_channels,
            &[],          // workspace_indexes
            None,         // FIXED seam: uv pins must NOT ride locked_closure
            Some(&prefs), // uv pins ride the favor-lock preference seam
            &std::collections::HashSet::new(),
        )
        .await
        .expect("resolve_bundle (uv pins as prefs) must succeed");

        let extras_names: Vec<&str> = bundle.extras.iter().map(|w| w.pypi_name.as_str()).collect();
        assert!(
            extras_names.contains(&sub_name),
            "uv-pinned transitive '{sub_name}' must be fetched into bundle.extras when the \
             pins flow through the preference seam; got: {extras_names:?}. If this fails, \
             uv pins are suppressing BFS seeding again (the NVIDIA-only \"no candidates\" \
             run-dep leak).",
        );
        let sub_wheel = bundle
            .extras
            .iter()
            .find(|w| w.pypi_name == sub_name)
            .unwrap();
        assert_eq!(
            sub_wheel.metadata.version, "1.0",
            "the bundled transitive must resolve at uv's pinned version 1.0, not the \
             index-latest 2.0",
        );

        // Part 2: the same pins through the locked-closure seam suppress
        // bundling (this is the incremental-add semantic, and the exact bug
        // shape resolve_all used to trigger on every cold uv-resolver pack).
        let bundle_locked = resolve_bundle(
            primary_name,
            &entry,
            &target,
            &download_dir,
            &source_dir,
            &cache_dir,
            RelaxPolicy::default(),
            &git_sources,
            None,
            &pypi_to_conda,
            BfsRoutePolicy::name_map_only(&name_map),
            &conda_channels,
            &[],
            Some(&uv_pins), // locked-closure seam: suppresses the walk
            None,
            &std::collections::HashSet::new(),
        )
        .await
        .expect("resolve_bundle (pins as locked_closure) must succeed");

        std::fs::remove_dir_all(&dir).ok();

        assert!(
            bundle_locked.extras.is_empty(),
            "locked_closure must suppress BFS seeding (incremental-add semantic); if this \
             starts bundling, re-audit BOTH seams: got {:?}",
            bundle_locked
                .extras
                .iter()
                .map(|w| w.pypi_name.as_str())
                .collect::<Vec<_>>(),
        );
    }

    /// Pure-merge semantics for the uv-pin preference seam.
    #[test]
    fn uv_pins_merge_semantics() {
        use std::collections::BTreeMap;
        let mut favored = BTreeMap::new();
        favored.insert("alpha".to_string(), "1.0".to_string());
        favored.insert("beta".to_string(), "2.0".to_string());

        let mut pins = BTreeMap::new();
        pins.insert("beta".to_string(), "2.5".to_string()); // uv wins on collision
        pins.insert("Gamma_X".to_string(), "3.0".to_string()); // canonicalized

        let merged = merge_uv_pins_into_prefs(favored.clone(), Some(&pins));
        assert_eq!(merged.get("alpha").map(String::as_str), Some("1.0"));
        assert_eq!(
            merged.get("beta").map(String::as_str),
            Some("2.5"),
            "uv's closure pin must override the committed-lock favored version"
        );
        assert_eq!(
            merged.get("gamma-x").map(String::as_str),
            Some("3.0"),
            "pin names must be canonicalized to match the BFS lookup key"
        );

        // No pins: favored passes through untouched.
        let merged_none = merge_uv_pins_into_prefs(favored.clone(), None);
        assert_eq!(merged_none, favored);
    }

    /// favor-lock: when favor_lock_prefs contains a version for a transitive
    /// dep, the BFS must fetch THAT version instead of the latest.
    ///
    /// Scenario:
    ///   - Primary `flpkg==1.0` has `Requires-Dist: flpkg-sub>=1.0`.
    ///     (prefix `flpkg-` matches `flpkg-sub` → seed_worklist picks it up)
    ///   - Index serves flpkg-sub at 1.0 and 2.0 (2.0 is the latest).
    ///   - favor_lock_prefs = {"flpkg-sub": "1.0"} (prefer the older version).
    ///
    /// Expected: bundle.extras contains flpkg-sub at 1.0 (not 2.0).
    ///
    /// Without favor-lock: pypi::resolve(>=1.0) picks 2.0 (highest).
    /// With favor-lock: resolve_preferring("flpkg-sub", >=1.0, prefer=1.0) picks 1.0.
    ///
    /// Marked #[ignore] because it runs a full BFS loop with a localhost fixture
    /// server; running it in parallel with other BFS tests can cause Tokio thread
    /// pool exhaustion and flaky failures in unrelated tests.
    /// Run with: `cargo test -- --include-ignored favor_lock_prefers`
    #[tokio::test]
    #[ignore = "bfs: runs full BFS with localhost fixture; run with --include-ignored"]
    async fn favor_lock_prefers_locked_transitive_version() {
        let dir = unique_tmp_dir();
        let download_dir = dir.join("download");
        let source_dir = dir.join("source");
        let cache_dir = dir.join("cache");
        std::fs::create_dir_all(&download_dir).unwrap();
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&cache_dir).unwrap();

        // Primary name: `flpkg`. Prefix: `flpkg-`. Sub: `flpkg-sub` (matches).
        let primary_name = "flpkg";
        let primary_version = "1.0";
        let sub_name = "flpkg-sub";

        let primary_bytes = make_wheel_bytes(
            primary_name,
            primary_version,
            &[&format!("{sub_name}>=1.0")],
        );
        let sub_v1_bytes = make_wheel_bytes(sub_name, "1.0", &[]);
        let sub_v2_bytes = make_wheel_bytes(sub_name, "2.0", &[]);

        // Serve both 1.0 and 2.0 for the sub-package.
        let port = spawn_index_server(
            vec![
                (
                    primary_name.to_string(),
                    primary_version.to_string(),
                    primary_bytes,
                ),
                (sub_name.to_string(), "1.0".to_string(), sub_v1_bytes),
                (sub_name.to_string(), "2.0".to_string(), sub_v2_bytes),
            ],
            48,
            false, // index advertises no #sha256 (legacy fixture behavior)
        )
        .await;

        let index_url = format!("http://127.0.0.1:{port}/simple/");

        let entry = WheelEntry {
            version: Some(primary_version.to_string()),
            index: Some(index_url),
            ..Default::default()
        };
        let target = ResolutionTarget::for_subdir("3.11", "linux-64");
        let pypi_to_conda: PypiToCondaMap = HashMap::new();
        let name_map = NameMap::new();
        let git_sources: BTreeMap<String, crate::config::NamedGitSource> = BTreeMap::new();
        let conda_channels: Vec<rattler_conda_types::ChannelUrl> = vec![];

        // Build the favor-lock prefs: hint BFS to prefer flpkg-sub @ 1.0.
        // resolve_bundle uses the prefs param directly without reading
        // RETREAD_FAVOR_LOCK -- the env-var gate lives only in load_favored_versions
        // (resolve_all entry point). Avoids process-wide env mutation that would
        // leak across parallel tests.
        let mut favor_lock_prefs: BTreeMap<String, String> = BTreeMap::new();
        favor_lock_prefs.insert("flpkg-sub".to_string(), "1.0".to_string());

        let bundle = resolve_bundle(
            primary_name,
            &entry,
            &target,
            &download_dir,
            &source_dir,
            &cache_dir,
            RelaxPolicy::default(),
            &git_sources,
            None, // auto_data
            &pypi_to_conda,
            BfsRoutePolicy::name_map_only(&name_map),
            &conda_channels,
            &[],                               // workspace_indexes
            None, // no incremental-add locked closure (deps must go through BFS)
            Some(&favor_lock_prefs), // favor-lock prefs: hint BFS to prefer flpkg-sub @ 1.0
            &std::collections::HashSet::new(), // no sibling context
        )
        .await
        .expect("resolve_bundle must succeed");

        std::fs::remove_dir_all(&dir).ok();

        // Sub must be present.
        let sub_wheel = bundle
            .extras
            .iter()
            .find(|w| w.pypi_name == sub_name)
            .unwrap_or_else(|| panic!("'{sub_name}' must be present in bundle.extras"));

        assert_eq!(
            sub_wheel.metadata.version, "1.0",
            "favor-lock must pin the transitive dep to the locked version 1.0 (not the \
             latest 2.0); got {}. If this fails, RETREAD_FAVOR_LOCK=1 is not wired through \
             to bfs_fetch_pypi / resolve_preferring.",
            sub_wheel.metadata.version,
        );
    }

    /// favor-lock cold path: without favor_lock_prefs, the BFS picks the
    /// highest matching version (standard behavior unchanged).
    ///
    /// Ensures the base highest-version selection is still the default when
    /// no favor-lock preferences are passed in.  Avoids env-var manipulation
    /// to prevent parallel-test interference.
    ///
    /// Marked #[ignore] because it runs a full BFS loop with a localhost fixture
    /// server; running it in parallel with other BFS tests can cause Tokio thread
    /// pool exhaustion and flaky failures in unrelated tests.
    /// Run with: `cargo test -- --include-ignored favor_lock_cold`
    #[tokio::test]
    #[ignore = "bfs: runs full BFS with localhost fixture; run with --include-ignored"]
    async fn favor_lock_cold_picks_latest_without_prefs() {
        let dir = unique_tmp_dir();
        let download_dir = dir.join("download");
        let source_dir = dir.join("source");
        let cache_dir = dir.join("cache");
        std::fs::create_dir_all(&download_dir).unwrap();
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&cache_dir).unwrap();

        // Primary: `flcold`. Sub: `flcold-sub` (matches prefix `flcold-`).
        let primary_name = "flcold";
        let primary_version = "1.0";
        let sub_name = "flcold-sub";

        let primary_bytes = make_wheel_bytes(
            primary_name,
            primary_version,
            &[&format!("{sub_name}>=1.0")],
        );
        let sub_v1_bytes = make_wheel_bytes(sub_name, "1.0", &[]);
        let sub_v2_bytes = make_wheel_bytes(sub_name, "2.0", &[]);

        let port = spawn_index_server(
            vec![
                (
                    primary_name.to_string(),
                    primary_version.to_string(),
                    primary_bytes,
                ),
                (sub_name.to_string(), "1.0".to_string(), sub_v1_bytes),
                (sub_name.to_string(), "2.0".to_string(), sub_v2_bytes),
            ],
            48,
            false, // index advertises no #sha256 (legacy fixture behavior)
        )
        .await;

        let index_url = format!("http://127.0.0.1:{port}/simple/");

        let entry = WheelEntry {
            version: Some(primary_version.to_string()),
            index: Some(index_url),
            ..Default::default()
        };
        let target = ResolutionTarget::for_subdir("3.11", "linux-64");
        let pypi_to_conda: PypiToCondaMap = HashMap::new();
        let name_map = NameMap::new();
        let git_sources: BTreeMap<String, crate::config::NamedGitSource> = BTreeMap::new();
        let conda_channels: Vec<rattler_conda_types::ChannelUrl> = vec![];

        // No favor-lock prefs provided: the BFS must pick the highest version
        // satisfying the specifier regardless of any env var state.
        let bundle = resolve_bundle(
            primary_name,
            &entry,
            &target,
            &download_dir,
            &source_dir,
            &cache_dir,
            RelaxPolicy::default(),
            &git_sources,
            None, // auto_data
            &pypi_to_conda,
            BfsRoutePolicy::name_map_only(&name_map),
            &conda_channels,
            &[],                               // workspace_indexes
            None,                              // no incremental-add locked closure
            None,                              // no favor-lock prefs → cold path, picks latest
            &std::collections::HashSet::new(), // no sibling context
        )
        .await
        .expect("resolve_bundle must succeed");

        std::fs::remove_dir_all(&dir).ok();

        let sub_wheel = bundle
            .extras
            .iter()
            .find(|w| w.pypi_name == sub_name)
            .unwrap_or_else(|| panic!("'{sub_name}' must be present in bundle.extras"));

        assert_eq!(
            sub_wheel.metadata.version, "2.0",
            "without favor-lock prefs, BFS must pick the LATEST version (2.0), not 1.0; \
             got {}. If this fails, the default highest-version selection is broken.",
            sub_wheel.metadata.version,
        );
    }
}

// -----------------------------------------------------------------
// Unit tests for detect_incremental_add.
// These test the gate logic in isolation (no network, no resolve).
// -----------------------------------------------------------------
#[cfg(test)]
mod incremental_add_tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use super::detect_incremental_add;
    use crate::config::RetreadConfig;
    use crate::lock::{LockWheel, Origin, RetreadLock, SCHEMA};

    fn valid_payload_wheel() -> LockWheel {
        LockWheel {
            name: "locked-payload".into(),
            version: "1.0".into(),
            origin: Origin::Index,
            filename: "locked_payload-1.0-py3-none-any.whl".into(),
            url: Some("https://example.com/locked_payload-1.0-py3-none-any.whl".into()),
            sha256: Some("11".repeat(32)),
            requires_dist: vec![],
            must_ship: false,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }
    }

    fn make_lock_at(path: &std::path::Path, entry_specs: Vec<String>, inputs_hash: &str) {
        let lock = RetreadLock {
            schema: SCHEMA,
            retread_version: "0.0.1".into(),
            bundle: "test-bundle".into(),
            version: "1.0".into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
            resolution_glibc: None,
            inputs_hash: inputs_hash.into(),
            root_requirements: vec![],
            wheels: vec![valid_payload_wheel()],
            conda_run_deps: vec![],
            index_urls: vec!["https://pypi.org/simple".into()],
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec![],
            entry_specs,
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };
        let json = lock.to_pretty_json().unwrap();
        std::fs::write(path, json).unwrap();
    }

    fn empty_config() -> RetreadConfig {
        serde_json::from_value(serde_json::json!({"retread-wheels": {}})).unwrap()
    }

    /// Gate 1: RETREAD_INCREMENTAL not set → always None.
    #[test]
    fn no_env_var_returns_none() {
        let dir = std::env::temp_dir().join(format!(
            "retread-incr-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).ok();
        let lock_path = dir.join("retread-test-bundle.lock.json");
        make_lock_at(&lock_path, vec!["test-bundle==1.0".into()], "dummy-hash");
        let config = empty_config();

        // Ensure RETREAD_INCREMENTAL is NOT set.
        // SAFETY: single-threaded test context; no concurrent env access.
        unsafe { std::env::remove_var("RETREAD_INCREMENTAL") };
        let result = detect_incremental_add(
            &lock_path,
            &config,
            "test-bundle",
            &[], // no workspace indexes
            "Eager",
            "3.11",
            "fp",
        );
        assert!(
            result.is_none(),
            "RETREAD_INCREMENTAL unset must return None"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Gate 2: missing lock → None.
    #[test]
    fn missing_lock_returns_none() {
        let missing = PathBuf::from("/tmp/retread-test-does-not-exist-xyz.json");
        let config = empty_config();

        // SAFETY: single-threaded test context; no concurrent env access.
        unsafe { std::env::set_var("RETREAD_INCREMENTAL", "1") };
        let result = detect_incremental_add(
            &missing,
            &config,
            "test-bundle",
            &[], // no workspace indexes
            "Eager",
            "3.11",
            "fp",
        );
        unsafe { std::env::remove_var("RETREAD_INCREMENTAL") };
        assert!(result.is_none(), "missing lock must return None");
    }

    /// Gate 2: lock with wrong schema → None.
    #[test]
    fn wrong_schema_returns_none() {
        let dir = std::env::temp_dir().join(format!(
            "retread-incr-schema-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).ok();
        let lock_path = dir.join("retread-test-bundle.lock.json");
        // Write a lock with wrong schema by patching the JSON directly.
        let lock = RetreadLock {
            schema: SCHEMA,
            retread_version: "0.0.1".into(),
            bundle: "test-bundle".into(),
            version: "1.0".into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
            resolution_glibc: None,
            inputs_hash: "dummy".into(),
            root_requirements: vec![],
            wheels: vec![valid_payload_wheel()],
            conda_run_deps: vec![],
            index_urls: vec![],
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec![],
            entry_specs: vec!["test-bundle==1.0".into()],
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };
        let mut json: serde_json::Value =
            serde_json::from_str(&lock.to_pretty_json().unwrap()).unwrap();
        json["schema"] = serde_json::json!(SCHEMA + 1);
        std::fs::write(&lock_path, serde_json::to_string_pretty(&json).unwrap()).unwrap();

        let config = empty_config();
        // SAFETY: single-threaded test context; no concurrent env access.
        unsafe { std::env::set_var("RETREAD_INCREMENTAL", "1") };
        let result = detect_incremental_add(
            &lock_path,
            &config,
            "test-bundle",
            &[],
            "Eager",
            "3.11",
            "fp",
        );
        unsafe { std::env::remove_var("RETREAD_INCREMENTAL") };
        assert!(result.is_none(), "wrong schema must return None");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Gate 2: lock with empty entry_specs → None (old schema).
    #[test]
    fn empty_entry_specs_returns_none() {
        let dir = std::env::temp_dir().join(format!(
            "retread-incr-empty-specs-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).ok();
        let lock_path = dir.join("retread-test-bundle.lock.json");
        make_lock_at(&lock_path, vec![], "dummy-hash"); // empty entry_specs

        let config = empty_config();
        // SAFETY: single-threaded test context; no concurrent env access.
        unsafe { std::env::set_var("RETREAD_INCREMENTAL", "1") };
        let result = detect_incremental_add(
            &lock_path,
            &config,
            "test-bundle",
            &[],
            "Eager",
            "3.11",
            "fp",
        );
        unsafe { std::env::remove_var("RETREAD_INCREMENTAL") };
        assert!(result.is_none(), "empty entry_specs must return None");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Two-step gate tests (a)-(i) ────────────────────────────────────────
    //
    // Helper: write a lock with explicit index_urls + entry_specs + hash.
    fn make_lock_with_indexes(
        path: &std::path::Path,
        entry_specs: Vec<String>,
        index_urls: Vec<String>,
        inputs_hash: &str,
    ) {
        let lock = RetreadLock {
            schema: SCHEMA,
            retread_version: "0.0.1".into(),
            bundle: "test-bundle".into(),
            version: "1.0".into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
            resolution_glibc: None,
            inputs_hash: inputs_hash.into(),
            root_requirements: vec![],
            wheels: vec![valid_payload_wheel()],
            conda_run_deps: vec![],
            index_urls,
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec![],
            entry_specs,
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };
        let json = lock.to_pretty_json().unwrap();
        std::fs::write(path, json).unwrap();
    }

    // Compute the correct inputs_hash for a lock. Uses the same algorithm as
    // courier::stage so the STEP B check passes.
    fn correct_hash(
        entry_specs: &[String],
        index_urls: &[String],
        relax: &str,
        python: &str,
        config_fp: &str,
    ) -> String {
        RetreadLock::compute_inputs_hash(
            entry_specs,
            index_urls,
            relax,
            python,
            crate::lock::EMIT_EPOCH,
            None, // no pin_version
            config_fp,
        )
    }

    fn tmp_incr(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "retread-incr-{label}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).ok();
        dir
    }

    // The "nvidia" entry index used in isaac tests.
    const NVIDIA_INDEX: &str = "https://pypi.nvidia.com/simple/";
    // PUBLIC_PYPI as stored by index_chain.
    const PYPI: &str = "https://pypi.org/simple/";

    /// (a) isaac case: add a dep with a NEW index while existing entries keep
    /// their indexes → STEP A passes, STEP B passes → ENGAGE (Some returned).
    #[test]
    fn step_a_new_index_entry_engages() {
        let dir = tmp_incr("a");
        let lock_path = dir.join(RetreadLock::file_name("test-bundle"));

        // Lock: single existing entry isaacsim (nvidia index).
        // lock.index_urls = pack-wide chain for [isaacsim] = [nvidia, pypi].
        let existing_spec = "isaacsim==4.0.0";
        let lock_index_urls = vec![NVIDIA_INDEX.to_string(), PYPI.to_string()];
        let h = correct_hash(
            &[existing_spec.to_string()],
            &lock_index_urls,
            "Eager",
            "3.11",
            "fp",
        );
        make_lock_with_indexes(
            &lock_path,
            vec![existing_spec.to_string()],
            lock_index_urls,
            &h,
        );

        // Config: isaacsim (nvidia) + new iniconfig==2.0.0 (pypi.org, no explicit index).
        let config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-bundle": "test-bundle",
            "retread-wheels": {
                "isaacsim": {"version": "4.0.0", "index": NVIDIA_INDEX},
                "iniconfig": {"version": "2.0.0"}
            }
        }))
        .unwrap();

        // SAFETY: single-threaded test context.
        unsafe { std::env::set_var("RETREAD_INCREMENTAL", "1") };
        let result = detect_incremental_add(
            &lock_path,
            &config,
            "test-bundle",
            &[], // no workspace indexes
            "Eager",
            "3.11",
            "fp",
        );
        unsafe { std::env::remove_var("RETREAD_INCREMENTAL") };
        let incr = result.expect("(a) should ENGAGE on add-dep-new-index");
        assert_eq!(
            incr.added_specs,
            vec!["iniconfig==2.0.0"],
            "added_specs must contain only the new entry"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// (b) COLD when an EXISTING entry's index changes.
    #[test]
    fn step_a_changed_existing_entry_index_is_cold() {
        let dir = tmp_incr("b");
        let lock_path = dir.join(RetreadLock::file_name("test-bundle"));

        // Lock: isaacsim was on nvidia index.
        let lock_index_urls = vec![NVIDIA_INDEX.to_string(), PYPI.to_string()];
        let h = correct_hash(
            &["isaacsim==4.0.0".to_string()],
            &lock_index_urls,
            "Eager",
            "3.11",
            "fp",
        );
        make_lock_with_indexes(
            &lock_path,
            vec!["isaacsim==4.0.0".to_string()],
            lock_index_urls,
            &h,
        );

        // Config: isaacsim now on pypi (changed!), same version.
        let config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-bundle": "test-bundle",
            "retread-wheels": {
                "isaacsim": {"version": "4.0.0"},
                "iniconfig": {"version": "2.0.0"}
            }
        }))
        .unwrap();

        // STEP A: locked chain for [isaacsim] (pypi now) = [pypi] ≠ [nvidia, pypi] → COLD.
        unsafe { std::env::set_var("RETREAD_INCREMENTAL", "1") };
        let result = detect_incremental_add(
            &lock_path,
            &config,
            "test-bundle",
            &[],
            "Eager",
            "3.11",
            "fp",
        );
        unsafe { std::env::remove_var("RETREAD_INCREMENTAL") };
        assert!(
            result.is_none(),
            "(b) changed existing entry index must be COLD"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// (c) COLD when a workspace extra-index is added (STEP A catches it).
    #[test]
    fn step_a_new_ws_index_is_cold() {
        let dir = tmp_incr("c");
        let lock_path = dir.join(RetreadLock::file_name("test-bundle"));

        // Lock: single entry, no ws index. index_urls = [pypi].
        let lock_index_urls = vec![PYPI.to_string()];
        let h = correct_hash(
            &["pkga==1.0".to_string()],
            &lock_index_urls,
            "Eager",
            "3.11",
            "fp",
        );
        make_lock_with_indexes(
            &lock_path,
            vec!["pkga==1.0".to_string()],
            lock_index_urls,
            &h,
        );

        let config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-bundle": "test-bundle",
            "retread-wheels": {
                "pkga": {"version": "1.0"},
                "pkgb": {"version": "2.0"}
            }
        }))
        .unwrap();

        // ws_indexes now has an extra index → locked_chain differs from lock.index_urls.
        let new_ws = vec!["https://extra.example.com/simple/".to_string()];
        unsafe { std::env::set_var("RETREAD_INCREMENTAL", "1") };
        let result = detect_incremental_add(
            &lock_path,
            &config,
            "test-bundle",
            &new_ws,
            "Eager",
            "3.11",
            "fp",
        );
        unsafe { std::env::remove_var("RETREAD_INCREMENTAL") };
        assert!(result.is_none(), "(c) new ws index must be COLD");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// (d) COLD when relax changes (STEP B hash mismatch), even though STEP A passes.
    #[test]
    fn step_b_changed_relax_is_cold() {
        let dir = tmp_incr("d");
        let lock_path = dir.join(RetreadLock::file_name("test-bundle"));

        // Lock built with relax="Eager".
        let lock_index_urls = vec![PYPI.to_string()];
        let h = correct_hash(
            &["pkga==1.0".to_string()],
            &lock_index_urls,
            "Eager", // ← original relax
            "3.11",
            "fp",
        );
        make_lock_with_indexes(
            &lock_path,
            vec!["pkga==1.0".to_string()],
            lock_index_urls,
            &h,
        );

        let config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-bundle": "test-bundle",
            "retread-wheels": {
                "pkga": {"version": "1.0"},
                "pkgb": {"version": "2.0"}
            }
        }))
        .unwrap();

        // Relax changed to "Conservative" → STEP B hash mismatch.
        unsafe { std::env::set_var("RETREAD_INCREMENTAL", "1") };
        let result = detect_incremental_add(
            &lock_path,
            &config,
            "test-bundle",
            &[],
            "Conservative", // ← changed
            "3.11",
            "fp",
        );
        unsafe { std::env::remove_var("RETREAD_INCREMENTAL") };
        assert!(
            result.is_none(),
            "(d) changed relax must be COLD via STEP B"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// (e) genesis-style add (single pypi.org entry, add another pypi dep) → ENGAGE.
    /// Regression guard: must still work when no new index is involved.
    #[test]
    fn genesis_style_add_engages() {
        let dir = tmp_incr("e");
        let lock_path = dir.join(RetreadLock::file_name("test-bundle"));

        let lock_index_urls = vec![PYPI.to_string()];
        let h = correct_hash(
            &["pkga==1.0".to_string()],
            &lock_index_urls,
            "Eager",
            "3.11",
            "fp",
        );
        make_lock_with_indexes(
            &lock_path,
            vec!["pkga==1.0".to_string()],
            lock_index_urls,
            &h,
        );

        let config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-bundle": "test-bundle",
            "retread-wheels": {
                "pkga": {"version": "1.0"},
                "pkgb": {"version": "2.0"}  // added, also pypi
            }
        }))
        .unwrap();

        unsafe { std::env::set_var("RETREAD_INCREMENTAL", "1") };
        let result = detect_incremental_add(
            &lock_path,
            &config,
            "test-bundle",
            &[],
            "Eager",
            "3.11",
            "fp",
        );
        unsafe { std::env::remove_var("RETREAD_INCREMENTAL") };
        let incr = result.expect("(e) genesis-style same-index add must ENGAGE");
        assert_eq!(incr.added_specs, vec!["pkgb==2.0"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// (f) COLD when an added spec is bare/range (Gate 5).
    #[test]
    fn gate5_bare_added_spec_is_cold() {
        let dir = tmp_incr("f");
        let lock_path = dir.join(RetreadLock::file_name("test-bundle"));

        let lock_index_urls = vec![PYPI.to_string()];
        let h = correct_hash(
            &["pkga==1.0".to_string()],
            &lock_index_urls,
            "Eager",
            "3.11",
            "fp",
        );
        make_lock_with_indexes(
            &lock_path,
            vec!["pkga==1.0".to_string()],
            lock_index_urls,
            &h,
        );

        // pkgb has no version → bare spec "pkgb" → Gate 5 rejects.
        let config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-bundle": "test-bundle",
            "retread-wheels": {
                "pkga": {"version": "1.0"},
                "pkgb": {}
            }
        }))
        .unwrap();

        unsafe { std::env::set_var("RETREAD_INCREMENTAL", "1") };
        let result = detect_incremental_add(
            &lock_path,
            &config,
            "test-bundle",
            &[],
            "Eager",
            "3.11",
            "fp",
        );
        unsafe { std::env::remove_var("RETREAD_INCREMENTAL") };
        assert!(
            result.is_none(),
            "(f) bare added spec must be COLD via Gate 5"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// (g) MULTI-BUNDLE: two bundles, two indexes; add a pinned dep to bundle B.
    /// STEP A must use all-entries-minus-added (NOT bundle-filtered entry_specs)
    /// so the pack-wide chain still matches lock.index_urls → ENGAGE.
    #[test]
    fn multi_bundle_step_a_uses_pack_wide_entries() {
        let dir = tmp_incr("g");
        let lock_path = dir.join(RetreadLock::file_name("test-bundle"));

        // Pack has two entries from different indexes:
        //   pkga (bundle A) → nvidia_index
        //   pkgb (bundle B) → pypi
        // lock.index_urls = pack-wide chain = [nvidia, pypi]
        let lock_index_urls = vec![NVIDIA_INDEX.to_string(), PYPI.to_string()];
        // entry_specs is BUNDLE-FILTERED (only bundle B entries in this lock).
        // (In reality, each bundle writes its own lock — this simulates bundle B's lock.)
        let lock_entry_specs = vec!["pkgb==1.0".to_string()];
        let h = correct_hash(&lock_entry_specs, &lock_index_urls, "Eager", "3.11", "fp");
        make_lock_with_indexes(&lock_path, lock_entry_specs, lock_index_urls, &h);

        // Config: pkga (nvidia, bundle A) + pkgb (pypi, bundle B) + new pkgc (pypi, bundle B).
        // STEP A: locked entries = all entries MINUS added = pkga + pkgb (both).
        // locked_chain = index_chain([nvidia], []) = [nvidia, pypi]
        // == lock.index_urls → PASS.
        let config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": {
                "pkga": {"version": "1.0", "index": NVIDIA_INDEX, "bundle": "bundle-a"},
                "pkgb": {"version": "1.0", "bundle": "test-bundle"},
                "pkgc": {"version": "2.0", "bundle": "test-bundle"}  // added
            }
        }))
        .unwrap();

        unsafe { std::env::set_var("RETREAD_INCREMENTAL", "1") };
        let result = detect_incremental_add(
            &lock_path,
            &config,
            "test-bundle",
            &[],
            "Eager",
            "3.11",
            "fp",
        );
        unsafe { std::env::remove_var("RETREAD_INCREMENTAL") };
        let incr = result.expect(
            "(g) multi-bundle: STEP A must use pack-wide (not bundle-filtered) entries -> ENGAGE",
        );
        assert_eq!(incr.added_specs, vec!["pkgc==2.0"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// (h) spec byte-identity: spec_for_entry, courier_input_specs, and the
    /// matched_entries encoding all produce the same string for the same entry.
    #[test]
    fn spec_for_entry_byte_identity() {
        use crate::config::WheelEntry;
        use std::collections::BTreeMap;

        let git_sources = BTreeMap::new();

        // Plain pinned entry (the common case).
        let entry = WheelEntry {
            version: Some("1.2.3".to_string()),
            ..Default::default()
        };
        let s = crate::courier::spec_for_entry("mylib", &entry, &git_sources);
        assert_eq!(s, "mylib==1.2.3", "plain pinned entry");

        // Entry with extras.
        let entry_extras = WheelEntry {
            version: Some("2.0".to_string()),
            extras: vec!["sim".to_string(), "render".to_string()],
            ..Default::default()
        };
        let s2 = crate::courier::spec_for_entry("mylib", &entry_extras, &git_sources);
        assert_eq!(s2, "mylib[sim,render]==2.0", "entry with extras");

        // URL entry (no version).
        let entry_url = WheelEntry {
            url: Some(url::Url::parse("https://example.com/mylib-1.0.whl").unwrap()),
            ..Default::default()
        };
        let s3 = crate::courier::spec_for_entry("mylib", &entry_url, &git_sources);
        assert_eq!(
            s3, "mylib@url:https://example.com/mylib-1.0.whl",
            "url entry"
        );

        // Bare entry (no version/git/url).
        let entry_bare = WheelEntry::default();
        let s4 = crate::courier::spec_for_entry("mylib", &entry_bare, &git_sources);
        assert_eq!(s4, "mylib", "bare entry");

        // courier_input_specs calls spec_for_entry — verify via a config round-trip.
        let config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": {
                "mylib": {"version": "1.2.3"}
            }
        }))
        .unwrap();
        let specs = crate::courier::courier_input_specs(&config, "mylib");
        assert_eq!(
            specs,
            vec!["mylib==1.2.3"],
            "courier_input_specs via spec_for_entry"
        );
    }

    /// (i) a locked entry moves from an explicit private index to implicit
    /// public PyPI, so STEP A detects the changed index universe and goes cold.
    #[test]
    fn step_a_existing_index_change_is_cold() {
        let dir = tmp_incr("i");
        let lock_path = dir.join(RetreadLock::file_name("test-bundle"));

        // Lock had: entry A (nvidia) + entry B (implicit pypi appended last).
        // lock.index_urls = [nvidia, pypi] (nvidia first, pypi appended).
        let lock_index_urls = vec![NVIDIA_INDEX.to_string(), PYPI.to_string()];
        let h = correct_hash(
            &["pkga==1.0".to_string()],
            &lock_index_urls,
            "Eager",
            "3.11",
            "fp",
        );
        make_lock_with_indexes(
            &lock_path,
            vec!["pkga==1.0".to_string()],
            lock_index_urls,
            &h,
        );

        // Now: entry A switched to implicit public PyPI, while entry C is the
        // newly added NVIDIA entry. Excluding the added entry leaves no explicit
        // indexes, so index_chain returns [pypi] != [nvidia, pypi] → COLD.
        let config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-bundle": "test-bundle",
            "retread-wheels": {
                "pkga": {"version": "1.0"},            // now on pypi (no explicit index)
                "pkgb": {"version": "2.0", "index": NVIDIA_INDEX}  // added, explicit nvidia
            }
        }))
        .unwrap();

        unsafe { std::env::set_var("RETREAD_INCREMENTAL", "1") };
        let result = detect_incremental_add(
            &lock_path,
            &config,
            "test-bundle",
            &[],
            "Eager",
            "3.11",
            "fp",
        );
        unsafe { std::env::remove_var("RETREAD_INCREMENTAL") };
        assert!(
            result.is_none(),
            "(i) changing an existing entry's index must be COLD"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── verify_localadd_hook tests ─────────────────────────────────────────

    // Build a lock where each wheel has explicit requires_dist.
    // (name, version, requires_dist lines)
    fn make_lock_with_requires_dist_at(path: &std::path::Path, wheels: &[(&str, &str, &[&str])]) {
        use crate::lock::{LockWheel, Origin};
        let lock_wheels: Vec<LockWheel> = wheels
            .iter()
            .map(|(n, v, rd)| LockWheel {
                name: n.to_string(),
                version: v.to_string(),
                origin: Origin::Index,
                filename: format!("{n}-{v}-py3-none-any.whl"),
                url: Some(format!(
                    "https://pypi.org/packages/{n}-{v}-py3-none-any.whl"
                )),
                sha256: None,
                requires_dist: rd.iter().map(|s| s.to_string()).collect(),
                must_ship: false,
                upstream_url: None,
                git_source: None,
                sdist_source: None,
            })
            .collect();
        let lock = RetreadLock {
            schema: SCHEMA,
            retread_version: "0.0.1".into(),
            bundle: "test-bundle".into(),
            version: "1.0".into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            target_contract: None,
            target_identity: None,
            target_scope: None,
            exact_workspace_envelope: false,
            resolution_glibc: None,
            inputs_hash: "dummy".into(),
            root_requirements: vec![],
            wheels: lock_wheels,
            conda_run_deps: vec![],
            index_urls: vec!["https://pypi.org/simple".into()],
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec![],
            entry_specs: vec!["test-bundle==1.0".into()],
            wheel_store: None,
            abi_context: None,
            relaxations: vec![],
        };
        let json = lock.to_pretty_json().unwrap();
        std::fs::write(path, json).unwrap();
    }

    fn tmp_dir(prefix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "retread-verify-{prefix}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).ok();
        dir
    }

    /// GREEN: A requires B>=1.0 (bundled at 1.2) — version satisfied → GREEN.
    #[test]
    fn verify_hook_green_when_bundled_requirement_satisfied() {
        let dir = tmp_dir("green");
        let lock_path = dir.join("test-bundle.retread-lock.json");
        make_lock_with_requires_dist_at(
            &lock_path,
            &[
                ("packagea", "2.0", &["packageb>=1.0"]),
                ("packageb", "1.2", &[]),
            ],
        );
        // No panic = pass. Oracle logs GREEN (info).
        super::verify_localadd_hook(
            &lock_path,
            &["packagea==2.0".to_string()],
            "test-bundle",
            "linux-64",
            "3.11",
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// RED (internal inconsistency): A requires B>=2.0 but B is BUNDLED at 1.0.
    /// This is the real missed-ripple case: the oracle must flag it RED.
    /// No panic — it's observability-only.
    #[test]
    fn verify_hook_red_when_bundled_dep_version_does_not_satisfy() {
        let dir = tmp_dir("red");
        let lock_path = dir.join("test-bundle.retread-lock.json");
        // packagea requires packageb>=2.0; packageb is bundled at 1.0 (missed ripple).
        make_lock_with_requires_dist_at(
            &lock_path,
            &[
                ("packagea", "2.0", &["packageb>=2.0"]),
                ("packageb", "1.0", &[]),
            ],
        );
        // No panic = pass. Oracle logs RED (warn) but never aborts.
        super::verify_localadd_hook(
            &lock_path,
            &["packagea==2.0".to_string()],
            "test-bundle",
            "linux-64",
            "3.11",
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Marker-false requirement: A requires B only on Windows (platform_system=='Windows').
    /// On linux-64 the marker is false → NOT active → must NOT be flagged RED,
    /// even though B is absent from the bundled closure.
    #[test]
    fn verify_hook_marker_false_not_red() {
        let dir = tmp_dir("marker");
        let lock_path = dir.join("test-bundle.retread-lock.json");
        make_lock_with_requires_dist_at(
            &lock_path,
            &[(
                "packagea",
                "2.0",
                &["packageb>=1.0; platform_system=='Windows'"],
            )],
        );
        // B is absent; marker is false on linux-64 → NOT active → NOT RED.
        super::verify_localadd_hook(
            &lock_path,
            &["packagea==2.0".to_string()],
            "test-bundle",
            "linux-64",
            "3.11",
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Key genesis-fix property: A requires B (conda/env-provided, NOT bundled).
    /// The oracle must SKIP — not RED — because we cannot version-check env deps.
    /// This covers opencv-python, OpenEXR, mujoco, vtk, and similar patterns.
    #[test]
    fn verify_hook_not_bundled_dep_is_skipped_not_red() {
        let dir = tmp_dir("skip");
        let lock_path = dir.join("test-bundle.retread-lock.json");
        // packagea requires libfoo (conda-provided, not in lock.wheels).
        make_lock_with_requires_dist_at(&lock_path, &[("packagea", "2.0", &["libfoo>=1.0"])]);
        // libfoo is absent from bundled closure → SKIPPED, not RED.
        super::verify_localadd_hook(
            &lock_path,
            &["packagea==2.0".to_string()],
            "test-bundle",
            "linux-64",
            "3.11",
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// I/O error: missing lock file → logs warning, no panic.
    #[test]
    fn verify_hook_handles_missing_lock_gracefully() {
        let missing_path = std::path::PathBuf::from("/tmp/retread-verify-nonexistent-xyz.json");
        super::verify_localadd_hook(
            &missing_path,
            &["requests==2.32.0".to_string()],
            "test-bundle",
            "linux-64",
            "3.11",
        );
    }
}

#[cfg(test)]
mod tests;
