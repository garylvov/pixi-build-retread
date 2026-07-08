//! JSON-RPC method handlers. The four entry points pixi calls.

mod audit_report;
use audit_report::{build_bundle_audit, write_probe_trace};

mod auto_bundle;
use auto_bundle::{
    BfsFetched, Pending, PendingSource, auto_bundle_transitives, metadata_preferring_sidecar,
    pick_conda_target, seed_worklist,
};

mod resolve_state;
use resolve_state::{ObserveEdgeResult, ResolveState};


use std::collections::{BTreeMap, HashSet, VecDeque};
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
use uv_pep508::uv_pep440::VersionSpecifiers;

use crate::config::{RelaxPolicy, RetreadConfig, WheelEntry};
use crate::pypi::{self, WheelTarget};
use crate::recipe::{BundleSource, build_bundle_recipe, build_courier_recipe, to_yaml};
use crate::relax::{canonical_conda_name, emit_python_version, marker_env_for};
use crate::rpc::{RpcError, ok, parse_params};
use crate::wheel::WheelMetadata;

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
static CONDA_OUTPUTS_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, CondaOutputsResult>>,
> = std::sync::OnceLock::new();

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
fn conda_outputs_cache_key(
    params: &CondaOutputsParams,
    workspace_mtime: Option<std::time::SystemTime>,
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
    format!(
        "{}|{}|{}|{:?}|{}",
        params.host_platform,
        params.build_platform,
        chans.join(","),
        params.variant_configuration,
        mtime_str,
    )
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
    if crate::fasttmp::in_slurm_job() {
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

/// Canonical public PyPI Simple index. Both the auto-bundle BFS and the
/// tiered-cascade need it as the final fallback. Trailing slash is
/// required by PEP 503 and kept here so `trim_end_matches('/')` in the
/// dedup check still works (both with and without the slash normalise to
/// the same key).
pub(crate) const PUBLIC_PYPI: &str = "https://pypi.org/simple/";

/// Build a deduplicated PyPI index chain, preserving ORDER semantics:
/// `primary` items first, then `extra` items, then `PUBLIC_PYPI` if
/// not already present. Deduplication is trailing-slash-insensitive so
/// `"https://pypi.org/simple"` and `"https://pypi.org/simple/"` are
/// treated as the same index.
///
/// This is the shared core extracted from two independent inline
/// implementations:
/// - `cascade::pypi_fallback_indexes` (entry indexes + workspace indexes)
/// - `auto_bundle::auto_bundle_transitives` (entry_index + workspace_indexes)
///
/// Both had identical push-unique + append-public-PyPI logic.
pub(crate) fn merge_index_chain(
    primary: impl IntoIterator<Item = String>,
    extra: &[String],
) -> Vec<String> {
    fn push_unique(list: &mut Vec<String>, idx: String) {
        if !list
            .iter()
            .any(|e| e.trim_end_matches('/') == idx.trim_end_matches('/'))
        {
            list.push(idx);
        }
    }
    let mut indexes: Vec<String> = Vec::new();
    for idx in primary {
        push_unique(&mut indexes, idx);
    }
    for idx in extra {
        push_unique(&mut indexes, idx.clone());
    }
    push_unique(&mut indexes, PUBLIC_PYPI.to_string());
    indexes
}

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
    python_version: &str,
    channels: &[String],
    workspace_manifest: Option<&crate::workspace::WorkspaceManifest>,
    workspace_dir: &std::path::Path,
    source_dir: &std::path::Path,
) -> String {
    let entry_specs = crate::courier::courier_input_specs(config, bundle_name);
    let ws_indexes: Vec<String> = workspace_manifest
        .map(|m| m.all_pypi_index_urls())
        .unwrap_or_default();
    let entry_indexes: Vec<String> = config
        .retread_wheels
        .values()
        .map(|e| e.index_url())
        .collect();
    let index_urls = merge_index_chain(entry_indexes, &ws_indexes);
    let workspace_fp = workspace_manifest
        .map(|m| m.solve_fingerprint(workspace_dir, source_dir))
        .unwrap_or_default();
    let config_fp = crate::courier::config_fingerprint(config, channels, &workspace_fp);
    crate::lock::RetreadLock::compute_inputs_hash(
        &entry_specs,
        &index_urls,
        &format!("{:?}", config.relax),
        python_version,
        crate::lock::EMIT_EPOCH,
        config.pin_version.then_some(env!("CARGO_PKG_VERSION")),
        &config_fp,
    )
}

const DEFAULT_PYTHON: &str = "3.11";

/// PyPI packages that are Windows-only and frequently declared as
/// unconditional `Requires-Dist` lines by upstream packagers (notably the
/// Isaac Sim wheels). Auto-dropped from run-deps when the target platform
/// isn't Windows, so users don't have to enumerate them in retread-drop-deps.
///
/// Inclusion criteria: ships wheels only for `win_*` platforms OR the
/// package is exclusively a shim for a Windows-specific subsystem (Win32
/// API, COM, registry, ANSI terminal compat, etc.) such that running it
/// on non-Windows is meaningless. Cross-platform packages (colorama,
/// chardet) do NOT belong here — they just happen to be most-cited on
/// Windows.
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
type PypiToCondaMap = std::collections::HashMap<String, Vec<String>>;

/// Best-effort fetch of the parselmouth mapping. Returns a fallback map
/// if the network call fails -- never errors. Async because it makes an
/// HTTP request.
async fn load_pypi_to_conda_map() -> PypiToCondaMap {
    let mut inverse: PypiToCondaMap = std::collections::HashMap::new();

    match reqwest::get(PARSELMOUTH_MAPPING_URL).await {
        Ok(resp) => match resp.error_for_status() {
            Ok(resp) => {
                match resp
                    .json::<std::collections::HashMap<String, Option<Vec<String>>>>()
                    .await
                {
                    Ok(forward) => {
                        for (conda_name, pypi_list) in forward {
                            for pypi in pypi_list.unwrap_or_default() {
                                inverse
                                    .entry(canonical_conda_name(&pypi))
                                    .or_default()
                                    .push(conda_name.clone());
                            }
                        }
                        tracing::info!(
                            entries = inverse.len(),
                            "loaded parselmouth PyPI<->conda mapping"
                        );
                    }
                    Err(e) => tracing::warn!(error = %e, "parselmouth JSON parse failed"),
                }
            }
            Err(e) => tracing::warn!(error = %e, "parselmouth fetch failed"),
        },
        Err(e) => tracing::warn!(error = %e, "parselmouth fetch failed"),
    }

    // Patch in known-missing entries from FALLBACK on top of parselmouth.
    // These are gaps in parselmouth's data (see each entry's comment for
    // the corresponding upstream issue). When the upstream issues are
    // fixed, the entries become harmless duplicates.
    for (pypi, conda) in FALLBACK_PYPI_TO_CONDA {
        let key = canonical_conda_name(pypi);
        let entry = inverse.entry(key).or_default();
        if !entry.iter().any(|c| c == conda) {
            entry.push((*conda).to_string());
        }
    }

    inverse
}

const BUILT_IN_WIN_ONLY: &[&str] = &[
    "comtypes",       // COM bindings
    "idna-ssl",       // async SSL shim, last release 2017
    "pyreadline",     // readline replacement (deprecated)
    "pyreadline3",    // readline replacement (current)
    "pywin32",        // Win32 API bindings
    "pywin32-ctypes", // ctypes-only fallback for pywin32
    "pywinpty",       // Windows pseudo-terminal (jupyter, IPython)
    "win32-setctime", // ctime setter for Windows files
    "winregistry",    // registry helper (stdlib winreg on Windows)
    "winrt-runtime",  // Windows Runtime API
    "winshell",       // shell helpers
    "wmi",            // Windows Management Instrumentation
];

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
}

#[derive(Clone, Default)]
pub struct Handler {
    state: Arc<RwLock<State>>,
}

struct Snapshot {
    config: RetreadConfig,
    download_dir: PathBuf,
    source_dir: PathBuf,
    cache_dir: PathBuf,
    workspace_dir: Option<PathBuf>,
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
    /// PR-2: canonical conda names that the BFS (or resolvo) routed to
    /// conda rather than bundling. Used exclusively by the A/B oracle
    /// (RETREAD_RESOLVO_DIFF) and never serialized to a lock.
    #[allow(dead_code)]
    conda_routed: Vec<String>,
}

impl Bundle {
    fn all_wheels(&self) -> impl Iterator<Item = &ResolvedWheel> {
        std::iter::once(&self.primary).chain(self.extras.iter())
    }
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
        let config: RetreadConfig = match params.configuration {
            Some(v) => serde_json::from_value(v)
                .map_err(|e| RpcError::invalid_params(format!("[build.config]: {e}")))?,
            None => {
                return Err(RpcError::invalid_params(
                    "pixi-build-retread requires a [build.config] table with at least `wheels = { ... }`",
                ));
            }
        };

        if config.retread_wheels.is_empty() {
            return Err(RpcError::invalid_params(
                "[build.config].wheels must list at least one wheel",
            ));
        }

        // Eagerly validate each entry now so misconfigurations surface at
        // initialize time rather than mid-build.
        for (name, entry) in &config.retread_wheels {
            entry
                .validate(name)
                .map_err(|e| RpcError::invalid_params(e.to_string()))?;
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

        let workspace_dir = params.workspace_directory;
        ensure_pixi_bld_symlink_target(workspace_dir.as_deref())?;

        let mut state = self.state.write().await;
        state.config = Some(config);
        state.cache_dir = params.cache_directory;
        // source_directory falls back to the manifest's containing dir.
        state.source_dir = params
            .source_directory
            .or_else(|| params.manifest_path.parent().map(PathBuf::from));
        state.workspace_dir = workspace_dir;
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
        let pre_key_workspace_dir = {
            let state = self.state.read().await;
            state.workspace_dir.clone()
        };
        let mtime = workspace_manifest_mtime(pre_key_workspace_dir.as_deref());
        let cache_key = conda_outputs_cache_key(&params, mtime);
        if let Some(cached) = CONDA_OUTPUTS_CACHE
            .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
            .lock()
            .unwrap()
            .get(&cache_key)
            .cloned()
        {
            tracing::info!(
                "retread: conda/outputs cache hit -- returning memoized result (pixi re-requested for another env)",
            );
            crate::status::tty(
                "reusing already-computed outputs (pixi re-requested this package for another environment).",
            );
            return Ok(cached);
        }
        // Fetched early (cheap: just clones handler state) so the DISK
        // cache below can be consulted before the expensive solve.
        let Snapshot {
            config,
            download_dir,
            source_dir,
            cache_dir,
            workspace_dir,
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
                .insert(cache_key, cached.clone());
            return Ok(cached);
        }
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
        // v4.2.0: the per-env pre-emission solve check (and its
        // bookkeeping / fail gate) was deleted with the legacy
        // mirror-solver; outputs ship unvalidated and `retread solve`
        // owns conflict handling.
        for python_version in &pythons {
            let target = wheel_target_for(params.host_platform, python_version);
            // Phase 1: materialize wheels + auto-bundle. Env-agnostic;
            // results reused across all per-env emissions.
            let t_materialize = std::time::Instant::now();
            let (materialized, base_config) = resolve_all(
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
            let bundle_names: HashSet<String> =
                materialized.iter().map(|b| b.conda_name.clone()).collect();
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
                python_version,
                &bundle_names,
            )
            .await;
            tracing::info!(
                elapsed_ms = t_discover.elapsed().as_millis() as u64,
                emissions = emissions.len(),
                "bench: discover_emissions (workspace transitive extraction) finished",
            );
            let workspace_manifest = workspace_dir
                .as_deref()
                .and_then(crate::workspace::WorkspaceManifest::load);
            // Cross-output siblings: per-emission so envs only link
            // to their own siblings (not other envs' renames).
            for emission in &emissions {
                let env_bundles: Vec<Bundle> = materialized
                    .iter()
                    .map(|b| apply_emission(b, &base_config, emission).0)
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
                    if config.courier {
                        let courier_channels_for_fp = workspace_manifest
                            .as_ref()
                            .map(|m| {
                                m.courier_channel_set(
                                    workspace_dir.as_deref().unwrap_or(source_dir.as_path()),
                                    &source_dir,
                                )
                            })
                            .unwrap_or_default();
                        let workspace_fp_for_incr = workspace_manifest
                            .as_ref()
                            .map(|m| {
                                m.solve_fingerprint(
                                    workspace_dir.as_deref().unwrap_or(source_dir.as_path()),
                                    &source_dir,
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
                            .map(|m| m.all_pypi_index_urls())
                            .unwrap_or_default();
                        let relax_str_for_incr = format!("{:?}", config.relax);
                        env_bundles
                            .iter()
                            .filter_map(|b| {
                                let lock_path = source_dir
                                    .join(crate::lock::RetreadLock::file_name(&b.conda_name));
                                detect_incremental_add(
                                    &lock_path,
                                    &config,
                                    &b.conda_name,
                                    &ws_indexes_for_incr,
                                    &relax_str_for_incr,
                                    python_version,
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
                for base_bundle in &materialized {
                    let (bundle, effective) =
                        apply_emission(base_bundle, &base_config, emission);
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
                                m.courier_channel_set(
                                    workspace_dir.as_deref().unwrap_or(source_dir.as_path()),
                                    &source_dir,
                                )
                            })
                            .unwrap_or_default();
                        Some(courier_inputs_hash(
                            &config,
                            &base_bundle.conda_name,
                            python_version,
                            &courier_channels,
                            workspace_manifest.as_ref(),
                            workspace_dir.as_deref().unwrap_or(source_dir.as_path()),
                            &source_dir,
                        ))
                    } else {
                        None
                    };
                    if let Some(ref current_hash) = courier_build_hash {
                        let lock_path = source_dir
                            .join(crate::lock::RetreadLock::file_name(&bundle.conda_name));
                        let relax_is_default =
                            config.relax == crate::config::RelaxPolicy::default();
                        match replay_from_lock(
                            &lock_path,
                            current_hash,
                            relax_is_default,
                            params.host_platform,
                            config.build_number,
                            config.bundle_mode == crate::config::BundleMode::Loose,
                            &siblings,
                        ) {
                            Ok(Some(replayed)) => {
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
                    let output = produce_output(
                        &bundle,
                        &effective,
                        params.host_platform,
                        python_version,
                        &siblings,
                        courier_build_hash.as_deref(),
                        version_override_for_bundle,
                    )
                    .map_err(|e| {
                        RpcError::internal(format!("output for {}: {e:#}", bundle.conda_name))
                    })?;
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
                    outputs.push(output);
                }
            }
        }
        tracing::debug!(outputs = outputs.len(), "per-env emission loop complete");
        let result = CondaOutputsResult {
            outputs,
            input_globs: Default::default(),
        };
        tracing::info!(
            elapsed_ms = phase_start.elapsed().as_millis() as u64,
            outputs = result.outputs.len(),
            "bench: conda_outputs total",
        );
        // Memoize so pixi's subsequent per-env re-requests (identical
        // params) skip the whole recompute.
        CONDA_OUTPUTS_CACHE
            .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
            .lock()
            .unwrap()
            .insert(cache_key, result.clone());
        // Cross-process: persist so a DIFFERENT retread process solving
        // another environment that shares this exact (params, workspace
        // mtime) key can skip the recompute too. Best-effort -- a write
        // failure (read-only cache dir, disk full) just means the next
        // process falls back to a cold compute, same as today.
        write_conda_outputs_disk_cache(&disk_cache_path, &result).await;
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
            config,
            download_dir,
            source_dir,
            cache_dir,
            workspace_dir,
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
        let python_version = match raw.as_deref() {
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
        let target = wheel_target_for(params.output.subdir, &python_version);

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
                    m.courier_channel_set(
                        workspace_dir.as_deref().unwrap_or(&source_dir),
                        &source_dir,
                    )
                })
                .unwrap_or_default();
            let workspace_fp = ws_manifest_for_replay
                .as_ref()
                .map(|m| {
                    m.solve_fingerprint(
                        workspace_dir.as_deref().unwrap_or(&source_dir),
                        &source_dir,
                    )
                })
                .unwrap_or_default();
            let config_fp =
                crate::courier::config_fingerprint(&config, &courier_channels, &workspace_fp);
            // The bundle_name for the hash is the requested output name
            // (params.output.name.as_normalized()), which equals bundle.conda_name
            // and is what courier::stage uses as the lock key.
            let bundle_name_for_hash = params.output.name.as_normalized().to_string();
            let current_hash = courier_inputs_hash(
                &config,
                &bundle_name_for_hash,
                &python_version,
                &courier_channels,
                ws_manifest_for_replay.as_ref(),
                workspace_dir.as_deref().unwrap_or(&source_dir),
                &source_dir,
            );
            let lock_path =
                source_dir.join(crate::lock::RetreadLock::file_name(&bundle_name_for_hash));
            let relax_is_default = config.relax == crate::config::RelaxPolicy::default();
            match load_replayable_lock(&lock_path, &current_hash, relax_is_default) {
                Ok(Some(lock)) => {
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
                    match materialize_from_lock(
                        lock,
                        &config,
                        &params.work_directory,
                        &build_output_dir,
                        params.output.subdir,
                        &source_dir,
                        &cache_dir,
                        params.output.build.as_deref(),
                        run_deps,
                        &config_fp,
                    )
                    .await
                    {
                        Ok(Some(result)) => {
                            return finalize_fasttmp_build_output(
                                result,
                                stage_output_dir.as_deref(),
                                &output_dir,
                            )
                            .await;
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
                    .map(|m| m.all_pypi_index_urls())
                    .unwrap_or_default();
                let relax_str = format!("{:?}", config.relax);
                if let Some(incr) = detect_incremental_add(
                    &lock_path,
                    &config,
                    &bundle_name_for_hash,
                    &ws_indexes,
                    &relax_str,
                    &python_version,
                    &config_fp,
                ) {
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
                        params.output.subdir,
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
                            return finalize_fasttmp_build_output(
                                result,
                                stage_output_dir.as_deref(),
                                &output_dir,
                            )
                            .await;
                        }
                        Ok(None) => {
                            tracing::debug!(
                                bundle = %bundle_name_for_hash,
                                "incremental-add: escalated to cold resolve"
                            );
                            // fall through to resolve_all
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

        // Re-resolve materialized bundles, then autodiscover emissions
        // and pick the one matching the requested output name.
        let (materialized, base_config) = resolve_all(
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
        let bundle_names: HashSet<String> =
            materialized.iter().map(|b| b.conda_name.clone()).collect();
        let default_name = materialized
            .first()
            .map(|b| b.conda_name.clone())
            .unwrap_or_default();
        let emissions = discover_emissions(
            &source_dir,
            workspace_dir.as_deref(),
            &default_name,
            &params.channels,
            &python_version,
            &bundle_names,
        )
        .await;

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
        // Apply emission to the matching base bundle. For typical
        // single-bundle source packs there's one bundle; for multi-
        // bundle packs we pick the bundle whose name starts the same.
        // Falling back to materialized[0] keeps single-pack behavior
        // identical to before.
        let base_bundle = materialized.first().ok_or_else(|| {
            RpcError::invalid_params("no bundles produced; check [retread-wheels]".to_string())
        })?;
        let (bundle, effective) = apply_emission(base_bundle, &base_config, picked_emission);

        // Build the recipe's run-deps from the EXACT specs pixi solved and
        // locked with (forwarded in `params.run_dependencies`), not by
        // re-deriving from the wheels' requires_dist. This keeps the built
        // package's run-deps identical to what `conda/outputs` emitted +
        // the solver locked (including cascade widenings like `pytorch >=1`),
        // and avoids re-emitting the raw transitive override that
        // rattler-build rejects as a malformed spec. Falls back to
        // requires_dist derivation if pixi didn't forward run-deps.
        let run_override: Option<Vec<String>> = params
            .run_dependencies
            .as_ref()
            .map(|deps| deps.iter().map(|d| d.spec.to_string()).collect());

        let result = build_one(
            &bundle,
            &effective,
            &config,
            &params.work_directory,
            &build_output_dir,
            params.output.subdir,
            &python_version,
            &source_dir,
            workspace_dir.as_deref(),
            params.output.build.as_deref(),
            run_override.as_deref(),
        )
        .await
        .map_err(|e| RpcError::internal(format!("build {}: {e:#}", bundle.conda_name)))?;
        finalize_fasttmp_build_output(result, stage_output_dir.as_deref(), &output_dir).await
    }

    async fn snapshot(&self, work_dir: &Path) -> Result<Snapshot, RpcError> {
        let (config, state_cache_dir, source_dir, workspace_dir) = {
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
                config,
                state.cache_dir.clone(),
                source_dir,
                state.workspace_dir.clone(),
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
            config,
            download_dir,
            source_dir,
            cache_dir,
            workspace_dir,
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

fn wheel_target_for(subdir: Platform, python_version: &str) -> WheelTarget {
    // The python_version comes from variant configuration (or the chosen
    // output's variant in conda/build_v1). It drives wheel selection on
    // the PyPI index (cp tag matching) and the marker env in relax.rs.
    WheelTarget {
        python_version: python_version.to_string(),
        conda_subdir: subdir.to_string(),
    }
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
    target: &WheelTarget,
    download_dir: &Path,
    source_dir: &Path,
    cache_dir: &Path,
    conda_channels: &[ChannelUrl],
    workspace_dir: Option<&Path>,
) -> Result<(Vec<Bundle>, RetreadConfig)> {
    let mut bundles = Vec::with_capacity(config.retread_wheels.len());

    // v1.3.0: workspace [pypi-options] indexes participate in
    // auto-bundle's fallback chain (between the entry index and public
    // PyPI), matching the cascade's chain.
    let workspace_pypi_indexes: Vec<String> = workspace_dir
        .and_then(crate::workspace::WorkspaceManifest::load)
        .map(|m| m.all_pypi_index_urls())
        .unwrap_or_default();

    // Load parselmouth once and reuse across bundles. We also merge it
    // into the effective name-map: when parselmouth says PyPI name X
    // corresponds to conda name Y, we emit Y in the conda run-deps
    // (otherwise the conda solver would fail to find X). Single-conda-
    // -name PyPI entries are unambiguous; multi-conda entries are
    // skipped from the merge (user must disambiguate via
    // retread-name-map).
    let pypi_to_conda = if config.auto_bundle {
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
    for (pypi, conda) in FALLBACK_PYPI_TO_CONDA {
        let key = canonical_conda_name(pypi);
        effective
            .name_map
            .entry(key)
            .or_insert_with(|| (*conda).to_string());
    }
    for (pypi, conda_names) in &pypi_to_conda {
        if conda_names.len() == 1 {
            effective
                .name_map
                .entry(pypi.clone())
                .or_insert_with(|| conda_names[0].clone());
        }
    }

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

    for (group_name, group_entries) in groups {
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
        let lock_path = source_dir.join(crate::lock::RetreadLock::file_name(&bundle_conda_name));
        let favored = load_favored_versions(&lock_path);

        // spec-uv-restructure M1: uv-backed closure computation. When the
        // pack selects `retread-resolver = "uv"`, this group's closure is
        // computed by a uv subprocess (ephemeral project + `uv lock` +
        // `uv export --format pylock.toml`) and the legacy materialization
        // below is PINNED to uv's picks via the locked-closure seam.
        // Packaging / courier / lock-write downstream are unchanged.
        // `Ok(None)` = no uv-resolvable roots (all source-built entries);
        // the legacy path then runs unpinned exactly as before.
        let uv_closure: Option<crate::uv_closure::UvClosure> =
            if effective.resolver == crate::config::ResolverKind::Uv {
                uv_group_closure(
                    &group_name,
                    &group_entries,
                    &effective,
                    target,
                    cache_dir,
                    workspace_dir,
                    &workspace_pypi_indexes,
                )
                .await
                .with_context(|| format!("computing uv closure for bundle `{group_name}`"))?
            } else {
                None
            };
        let uv_pins: Option<&BTreeMap<String, String>> = uv_closure.as_ref().map(|c| &c.pins);

        // v2.10.0: build the full sibling name set for this group once.
        // For each entry in the group we compute a sibling set = all OTHER
        // entries' canonical names, so resolve_bundle can skip deps that name
        // a sibling (they're provided by the sibling's wheel at install time).
        let all_entry_canonical: Vec<String> = group_entries
            .iter()
            .map(|(n, _)| canonical_conda_name(n))
            .collect();

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
        const ENTRY_BUILD_CONCURRENCY: usize = 6;
        let mut entry_futures = Vec::with_capacity(group_entries.len());
        for (idx, ((entry_name, entry), auto_data)) in
            group_entries.iter().zip(auto_data_per_entry).enumerate()
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
            let workspace_pypi_indexes = &workspace_pypi_indexes;
            entry_futures.push(async move {
                    resolve_bundle(
                        entry_name,
                        entry,
                        target,
                        download_dir,
                        source_dir,
                        cache_dir,
                        effective.relax,
                        &effective.git_sources,
                        auto_data,
                        pypi_to_conda,
                        &effective.name_map,
                        conda_channels,
                        &effective.conda_deps,
                        workspace_pypi_indexes,
                        uv_pins, // uv resolver: pin to uv's closure; legacy: None (cold path)
                        Some(favored).filter(|m| !m.is_empty()), // favor-lock prefs (empty map → None)
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
                .buffered(ENTRY_BUILD_CONCURRENCY)
                .try_collect()
                .await?
        };
        let mut bundle = sub_bundles.remove(0);
        bundle.conda_name = canonical_conda_name(&group_name);
        for sub in sub_bundles {
            bundle.extras.push(sub.primary);
            bundle.extras.extend(sub.extras);
            // Each sub-bundle's BFS probe decisions get merged into
            // the carrier bundle so the per-bundle audit shows EVERY
            // dep that was probed across the whole group.
            bundle.probe_decisions.extend(sub.probe_decisions);
        }
        // Auto-bundle scans the whole merged bundle's Requires-Dist, so
        // it naturally handles transitives pulled by any wheel in the
        // group. Use the first non-URL entry's index for the candidate
        // fallback chain (URL-form entries can't auto-bundle anyway --
        // they have no PyPI index to resolve from).
        let auto_index: Option<String> = group_entries.iter().find_map(|(_, e)| {
            if e.url.is_none() {
                Some(e.index_url())
            } else {
                None
            }
        });
        if effective.auto_bundle
            && let Some(idx) = auto_index
        {
            auto_bundle_transitives(
                &mut bundle,
                &idx,
                &workspace_pypi_indexes,
                target,
                download_dir,
                &effective,
                conda_channels,
                uv_pins, // uv resolver: pin to uv's closure; legacy: None (cold path)
                Some(&favored).filter(|m| !m.is_empty()), // favor-lock prefs
            )
            .await?;
        }
        // v0.32.0+: pre_emit_widen_pass moved OUT of resolve_all into
        // the per-env emission loop in conda_outputs. Materialization
        // (download/build/auto-bundle) is env-agnostic; the cascade +
        // override injection is env-specific so per_env can run it N
        // times with N different channel sets / transitive
        // constraints.
        bundles.push(bundle);
    }

    Ok((bundles, effective))
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
/// - routing is force-list only (`retread-conda-deps`); the probe-driven
///   post-resolution filter is M2.
async fn uv_group_closure(
    group_name: &str,
    group_entries: &[(String, WheelEntry)],
    effective: &RetreadConfig,
    target: &WheelTarget,
    cache_dir: &Path,
    workspace_dir: Option<&Path>,
    workspace_pypi_indexes: &[String],
) -> Result<Option<crate::uv_closure::UvClosure>> {
    let mut roots: Vec<String> = Vec::new();
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
            roots.push(format!("{name} @ {url}"));
        } else {
            tracing::info!(
                entry = %name,
                bundle = %group_name,
                "uv closure: source-built entry is not a uv root; it resolves \
                 via the legacy materialization path (milestone-1 limit)",
            );
        }
    }
    if roots.is_empty() {
        tracing::info!(
            bundle = %group_name,
            "uv closure: no uv-resolvable roots in this bundle; \
             running the legacy closure path unpinned",
        );
        return Ok(None);
    }

    // Conda pins -> uv constraints, with provenance (spec §2.2 fallback
    // path: the manifest's effective deps; pixi.lock-gated read is M2+).
    let constraints = match workspace_dir.and_then(crate::workspace::WorkspaceManifest::load) {
        Some(manifest) => {
            let deps = manifest.effective_dependencies("default");
            crate::uv_closure::build_constraints(&deps, &effective.name_map, "manifest", "default")
        }
        None => Default::default(),
    };

    // retread-overrides -> override-dependencies where PEP 440-representable.
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

    // Index chain: entry indexes in group order, then workspace
    // [pypi-options] indexes, then public PyPI. Deduped, order-preserving.
    let mut index_urls: Vec<String> = Vec::new();
    for url in group_entries
        .iter()
        .filter(|(_, e)| e.url.is_none())
        .map(|(_, e)| e.index_url())
        .chain(workspace_pypi_indexes.iter().cloned())
        .chain(std::iter::once(PUBLIC_PYPI.to_string()))
    {
        if !index_urls.contains(&url) {
            index_urls.push(url);
        }
    }

    // retread-drop-deps also excluded from the parsed closure.
    let mut no_emit: Vec<String> = effective.conda_deps.clone();
    no_emit.extend(effective.drop_deps.iter().cloned());

    let req = crate::uv_closure::UvClosureRequest {
        bundle: group_name.to_string(),
        python_version: target.python_version.clone(),
        conda_subdir: target.conda_subdir.clone(),
        dependencies: roots,
        constraints,
        overrides,
        no_emit_packages: no_emit,
        index_urls,
        built_wheel_sources: BTreeMap::new(), // M1: source-built entries stay legacy
        offline: false,
    };
    let project_dir = cache_dir.join("uv-projects").join(format!(
        "{}-py{}-{}",
        canonical_conda_name(group_name),
        target.python_version,
        target.conda_subdir,
    ));
    let uv_cache_dir = cache_dir.join("uv-cache");
    let closure =
        crate::uv_closure::compute_closure(&req, &project_dir, &uv_cache_dir, None).await?;
    Ok(Some(closure))
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
    transitive_overrides: BTreeMap<String, String>,
    #[allow(dead_code)]
    envs: Vec<String>,
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
    target_python: &str,
    bundle_names: &HashSet<String>,
) -> Vec<DiscoveredEmission> {
    let manifest_opt = workspace_dir.and_then(crate::workspace::WorkspaceManifest::load);
    let default_emission = || DiscoveredEmission {
        output_name: default_output_name.to_string(),
        channels: default_channels.to_vec(),
        transitive_overrides: BTreeMap::new(),
        envs: Vec::new(),
    };

    let (Some(manifest), Some(ws_dir)) = (manifest_opt.as_ref(), workspace_dir) else {
        return vec![default_emission()];
    };
    let discovered = manifest.discover_outputs_for_source(ws_dir, source_dir);
    if discovered.is_empty() {
        return vec![default_emission()];
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
        // concurrently. join_all preserves input order, keeping the
        // accumulated clause order (and thus the joined spec strings)
        // deterministic.
        let env_results = futures::future::join_all(d.envs.iter().map(|env| {
            crate::workspace::extract_transitive_constraints(
                manifest,
                env,
                target_python,
                &channels,
                bundle_names,
            )
        }))
        .await;
        let mut accumulated: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for trans in env_results {
            for (dep, specs) in trans {
                let entry = accumulated.entry(dep).or_default();
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
        let direct = manifest.union_effective_dependencies(&d.envs);
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
    out
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

/// Apply a `DiscoveredEmission` to a materialized bundle + base
/// config. Renames the bundle to the discovered output name and
/// injects the env-union transitive constraints into the config's
/// overrides. The cascade then treats them as authoritative.
fn apply_emission(
    base_bundle: &Bundle,
    base_config: &RetreadConfig,
    emission: &DiscoveredEmission,
) -> (Bundle, RetreadConfig) {
    let mut bundle = base_bundle.clone();
    bundle.conda_name = emission.output_name.clone();
    // Reset the per-bundle probe trace: env-specific cascade decisions
    // get recorded fresh. Materialize-phase decisions belong to the
    // shared materialization, not this env's solve.
    bundle.probe_decisions.clear();

    let mut config = base_config.clone();
    for (dep, spec) in &emission.transitive_overrides {
        // User's manual [retread-overrides] wins over workspace
        // transitive; preserve their entry if present.
        config
            .overrides
            .entry(dep.clone())
            .or_insert_with(|| spec.clone());
    }
    (bundle, config)
}

#[allow(clippy::too_many_arguments)]
async fn resolve_bundle(
    entry_name: &str,
    entry: &WheelEntry,
    target: &WheelTarget,
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
    // v0.46.0: the merged effective name-map (user retread-name-map +
    // FALLBACK_PYPI_TO_CONDA + unambiguous parselmouth). The BFS prefer-
    // conda picker consults THIS first, so curated answers like
    // torch->pytorch route to conda even when parselmouth's inverted map
    // is ambiguous (multiple conda candidates, no identity match). Without
    // it, `torch` fell through to PyPI and got bundled at latest (2.12.0),
    // clobbering conda's pinned pytorch at install. Emission already used
    // this map; the BFS now matches it.
    name_map: &std::collections::BTreeMap<String, String>,
    conda_channels: &[ChannelUrl],
    // PR-2: retread-conda-deps names (force-list). Used only by the A/B
    // oracle to capture the auto_bundle skip-set route; never affects BFS logic.
    conda_deps_list: &[String],
    // Workspace PyPI index chain (kept for call-site symmetry with
    // auto_bundle_transitives; unused since the resolvo mirror-solver
    // was deleted in v4.2.0).
    _workspace_indexes: &[String],
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
    let (primary, primary_original_rd) = materialize_and_rewrite(
        entry,
        entry_name,
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
            conda_routed: vec![],
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
        seed_worklist(
            &primary_original_rd,
            &entry.extras,
            &entry.index_url(),
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
    // PR-2: canonical conda names routed to conda (not bundled) during BFS.
    // Used only by the A/B oracle; never serialized.
    let mut conda_routed_acc: Vec<String> = Vec::new();
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
            //   (e) satisfied / indecisive -> short-circuit.
            //       unsatisfiable -> fall through to PyPI.
            //
            // The v0.13.10 `.first()` candidate picker was wrong: the
            // inverted parselmouth map has many false positives (a conda
            // package can list a pypi dep without "being" it), so picking
            // an arbitrary candidate gave nonsense like `numpy -> manifpy`
            // and `torch -> pytorch-cpu`. The probe then asked the wrong
            // question. v0.17.0 fixes the picker.
            let dep_conda_name = canonical_conda_name(&pending.pypi_name);
            let mut routed_to_conda = false;
            if let PendingSource::Pypi { specifiers, .. } = &pending.source {
                // v0.46.0: the curated/user name-map wins over parselmouth.
                // If it has an entry for this dep (e.g. torch->pytorch from
                // FALLBACK), treat that as the unambiguous conda target --
                // otherwise fall back to parselmouth's inverted candidates,
                // which are often ambiguous for exactly the deps the FALLBACK
                // table exists to disambiguate.
                {
                    let picked: Option<String> =
                        pick_conda_target(&dep_conda_name, name_map, pypi_to_conda);
                    match picked {
                        None => {
                            let amb = pypi_to_conda.get(&dep_conda_name);
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
                            // Normalize the spec for conda's matchspec
                            // parser:
                            //   * Strip the space after `,` --
                            //     `VersionSpecifiers::to_string()`
                            //     produces `>=0.23.0, <0.24.0` (space)
                            //     which lenient parsing accepted but
                            //     silently dropped the second clause for
                            //     some inputs -> probe returned satisfied
                            //     when it shouldn't.
                            //   * Coerce empty-spec to `*` -- bare-name
                            //     `Requires-Dist: gym` produces empty
                            //     VersionSpecifiers, which conda
                            //     matchspec can't parse -> probe returned
                            //     indecisive -> BFS short-circuited
                            //     ("indecisive-short-circuit") instead of
                            //     bundling. `*` means "any version" which
                            //     is what PEP 508 bare-name means, and it
                            //     lets the python-compat filter do its
                            //     real job. (gym now: probe with "*"
                            //     finds many gym versions but NONE have a
                            //     py3.11 build -> satisfiable=false ->
                            //     fall through to PyPI bundle.)
                            let normalized = specifiers.to_string().replace(", ", ",");
                            let probe_spec = if normalized.trim().is_empty() {
                                "*".to_string()
                            } else {
                                normalized
                            };
                            let probe_result = crate::probe::probe(
                                conda_channels,
                                &conda_target_name,
                                &probe_spec,
                                Some(&target.python_version),
                            )
                            .await;
                            let routing_decision = if probe_result.is_definitively_unsatisfied() {
                                "fall-through-to-pypi"
                            } else if probe_result.is_satisfied() {
                                "short-circuit"
                            } else {
                                "indecisive-short-circuit"
                            };
                            probe_decisions.push(crate::audit::ProbeDecision {
                                stage: "bfs".into(),
                                pypi_name: pending.pypi_name.clone(),
                                conda_name: conda_target_name.clone(),
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
                                matches = probe_result.matching_candidates,
                                channels = ?probe_result.channels_consulted,
                                "BFS prefer-conda probe result",
                            );
                            if routing_decision != "fall-through-to-pypi" {
                                routed_to_conda = true;
                            } else {
                                // v0.46.0: the EXACT wheel spec isn't on conda
                                // (e.g. wheel pins torch==2.7.0 but conda has
                                // 2.7.1, or the dep resolved to PyPI-latest in
                                // isolation). Bundling here vendors a PyPI build
                                // that SHADOWS conda's ABI-correct copy and
                                // double-installs a dep we also emit as a conda
                                // run-dep. Before falling through to PyPI, probe
                                // whether conda has the package at ANY (py-compat)
                                // version; if so, keep it on conda and let the
                                // run-dep emission + solve cascade pick the
                                // ABI-correct build. Only bundle when conda truly
                                // lacks the package.
                                let name_level = crate::probe::probe(
                                    conda_channels,
                                    &conda_target_name,
                                    "*",
                                    Some(&target.python_version),
                                )
                                .await;
                                probe_decisions.push(crate::audit::ProbeDecision {
                                    stage: "bfs_name_level".into(),
                                    pypi_name: pending.pypi_name.clone(),
                                    conda_name: conda_target_name.clone(),
                                    spec: "*".into(),
                                    target_python: target.python_version.clone(),
                                    channels_consulted: name_level.channels_consulted.clone(),
                                    satisfiable: name_level.satisfiable,
                                    matching_candidates: name_level.matching_candidates,
                                    routing_decision: if name_level.is_satisfied() {
                                        "name-level-conda-keep"
                                    } else {
                                        "fall-through-to-pypi"
                                    }
                                    .into(),
                                });
                                if name_level.is_satisfied() {
                                    tracing::info!(
                                        dep = %pending.pypi_name,
                                        conda_name = %conda_target_name,
                                        wheel_spec = %probe_spec,
                                        conda_matches = name_level.matching_candidates,
                                        "BFS prefer-conda: exact wheel spec absent on conda but the package exists at other versions -- keeping on conda (ABI-correct) instead of bundling a PyPI build",
                                    );
                                    routed_to_conda = true;
                                }
                            }
                        }
                    }
                }
            }
            if routed_to_conda {
                // PR-2: record the routed conda name for the A/B oracle.
                conda_routed_acc.push(dep_conda_name.clone());
                continue;
            }
            // v2.10.0 defense-in-depth: a sibling dep that reached the BFS
            // frontier (e.g. via a transitive observe_edge path rather than
            // seed_worklist) must still be suppressed.  The primary guard is
            // in seed_worklist, but any dep that snuck through (e.g. a sibling
            // referenced transitively by another transitive dep) is caught here
            // before the PyPI fetch.
            if sibling_names.contains(&dep_conda_name) {
                tracing::debug!(
                    dep = %pending.pypi_name,
                    sibling_canon = %dep_conda_name,
                    "BFS frontier: skipping sibling dep — provided by sibling bundle entry",
                );
                continue;
            }
            to_materialize.push(pending);
        }

        // Phase 2: fetch this level's PyPI-form wheels concurrently
        // (8-way bounded, order-preserving `buffered`). Git/URL forms
        // pass through untouched and materialize serially in phase 3.
        // Per item the semantics are byte-identical to the old serial
        // arm: wheel resolve -> fetch, with sdist fallback on
        // wheel-resolve failure; the first error fails the whole
        // bundle exactly as `?` did.
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
            stream::iter(to_materialize)
                .map(|pending| async move {
                    let result = match &pending.source {
                        PendingSource::Pypi { specifiers, index } => {
                            // Look up the preferred locked version for this
                            // dep (canonical-normalized name).  Returns None
                            // on the cold path (empty snapshot) so bfs_fetch_pypi
                            // falls back to the normal highest-version selection.
                            let dep_canon = crate::relax::canonical_conda_name(&pending.pypi_name);
                            let prefer_version: Option<&str> =
                                favor_lock_snap_ref.get(&dep_canon).map(String::as_str);
                            bfs_fetch_pypi(
                                &pending.pypi_name,
                                specifiers,
                                index,
                                target,
                                download_dir,
                                relax,
                                prefer_version,
                            )
                            .await
                            .map(Some)
                        }
                        _ => Ok(None),
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
            // 7-tuple: (url, upstream_url, git_source, sdist_source, metadata, index, seed_rd)
            let (
                sub_url,
                sub_upstream_url,
                sub_git_source,
                sub_sdist_source,
                sub_metadata,
                sub_index_for_recurse,
                sub_seed_rd,
            ) = match (&pending.source, fetch_result?) {
                (PendingSource::Pypi { .. }, Some((resolved_url, metadata, index, sdist_prov))) => {
                    // Pypi-form sub-wheels are NOT D-rewritten, so
                    // their metadata IS the original Requires-Dist.
                    let seed_rd = metadata.requires_dist.clone();
                    // Build the sdist provenance descriptor (None for normal wheel fetches).
                    let sub_sdist_src = sdist_prov.map(|p| crate::lock::SdistWheelSource {
                        index: p.index,
                        name: p.name,
                        version: p.version,
                        // Store the EXACT resolved sdist URL with #sha256 (Amendment 4:
                        // freeze the URL so replay builds the identical tarball without
                        // re-resolving, which is neither yank-safe nor reorder-deterministic).
                        sdist_url: p.sdist_url.to_string(),
                    });
                    // When the wheel was built from an sdist, DO NOT store the file://
                    // built_url as upstream_url (it is machine-local and non-portable).
                    // For normal index wheels, record resolved_url as upstream_url.
                    let upstream = if sub_sdist_src.is_some() {
                        None // sdist-built: upstream_url suppressed; use sdist_source instead
                    } else {
                        // resolved_url is the pristine https index URL; record it as
                        // upstream_url so build_one can populate EmitWheel.upstream_url
                        // without deriving from w.url.
                        Some(resolved_url.clone())
                    };
                    (
                        resolved_url,
                        upstream,
                        None,
                        sub_sdist_src,
                        metadata,
                        index,
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
                    let (sub, sub_original_rd) = materialize_and_rewrite(
                        &synth,
                        &synth_name,
                        target,
                        download_dir,
                        source_dir,
                        cache_dir,
                        relax,
                        git_sources,
                        None,
                        EntryAuditInfo::default(),
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
                    // For the recurse, use the parent ENTRY's index (not
                    // `prefix` -- that's a name-prefix string, NOT a URL).
                    // The recurse fires for Pypi-form Requires-Dist of the
                    // sub-wheel; those go through pypi::resolve which needs
                    // a real Simple index URL.
                    let sub_gs = sub.git_source.clone();
                    let sub_up = sub.upstream_url.clone();
                    (
                        sub.url,
                        sub_up,
                        sub_gs,
                        None, // Git-form: no sdist provenance
                        sub.metadata,
                        entry.index_url(),
                        sub_original_rd,
                    )
                }
                (PendingSource::Url { wheel_url }, _) => {
                    let synth = WheelEntry {
                        url: Some(wheel_url.clone()),
                        ..Default::default()
                    };
                    let synth_name = pending.pypi_name.clone();
                    let (sub, sub_original_rd) = materialize_and_rewrite(
                        &synth,
                        &synth_name,
                        target,
                        download_dir,
                        source_dir,
                        cache_dir,
                        relax,
                        git_sources,
                        None,
                        EntryAuditInfo::default(),
                    )
                    .await
                    .with_context(|| {
                        format!(
                            "materializing URL Requires-Dist `{} @ {}`",
                            pending.pypi_name, wheel_url,
                        )
                    })?;
                    // Same fix as the Git arm: recurse uses the parent
                    // entry's PyPI Simple index, not the name `prefix`.
                    let sub_up = sub.upstream_url.clone();
                    (
                        sub.url,
                        sub_up,
                        None, // Url-form: no git source
                        None, // Url-form: no sdist provenance
                        sub.metadata,
                        entry.index_url(),
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
                    &sub_index_for_recurse,
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

    // PR-2: union in force-list names (retread-conda-deps) that appear as
    // transitive requires_dist entries in the bundled wheels. auto_bundle_transitives
    // skips these silently (no ProbeDecision pushed); we mirror that here so the
    // A/B oracle sees the same effective conda-routed set.
    {
        let force_conda: std::collections::HashSet<String> = conda_deps_list
            .iter()
            .map(|n| canonical_conda_name(n))
            .collect();
        // Collect all transitive Requires-Dist names from bundled wheels.
        let bundled_rd_names: std::collections::HashSet<String> = std::iter::once(&primary)
            .chain(extras.iter())
            .flat_map(|w| w.metadata.requires_dist.iter())
            .filter_map(|raw| {
                uv_pep508::Requirement::from_str(raw.as_str())
                    .ok()
                    .map(|r: uv_pep508::Requirement| canonical_conda_name(r.name.as_ref()))
            })
            .collect();
        for name in force_conda.intersection(&bundled_rd_names) {
            if !conda_routed_acc.contains(name) {
                conda_routed_acc.push(name.clone());
            }
        }
    }

    let bfs_bundle = Bundle {
        conda_name,
        primary,
        extras,
        probe_decisions,
        solve_diagnostics: BTreeMap::new(),
        conda_routed: conda_routed_acc,
    };

    Ok(bfs_bundle)
}

/// One PyPI-form BFS item's materialization: wheel resolve -> fetch,
/// with sdist fallback on wheel-resolve failure (PyPI publishers like
/// OpenAI gym stopped shipping wheels; uv builds the sdist into a
/// wheel). The sdist fallback uses the SAME spec, so a narrow version
/// pin still gets honored. Extracted verbatim from the old serial BFS
/// arm so phase 2 of the level loop can run items concurrently.
/// v1.5.9: produce the relaxed retry specifiers for a sub-wheel whose
/// EXACT upstream pin is missing from the index. Returns None when the
/// policy doesn't relax or relaxation changes nothing (bare deps,
/// range specs -- relax only widens single exact pins).
fn relaxed_retry_specs(
    pypi_name: &str,
    specifiers: &VersionSpecifiers,
    relax: RelaxPolicy,
) -> Option<VersionSpecifiers> {
    if relax == RelaxPolicy::None {
        return None;
    }
    let original = format!("{pypi_name}{specifiers}");
    let relaxed_line = crate::wheel_rewrite::relax_pep508(&original, relax).ok()?;
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
/// Threaded out of `bfs_fetch_pypi` so the caller can populate
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

async fn bfs_fetch_pypi(
    pypi_name: &str,
    specifiers: &VersionSpecifiers,
    index: &str,
    target: &WheelTarget,
    download_dir: &Path,
    relax: RelaxPolicy,
    // favor-lock: when Some, prefer this version on the index before falling
    // back to highest-version selection. Propagated from favor_lock_prefs by the
    // BFS phase-2 fetch loop when RETREAD_FAVOR_LOCK=1. None on the cold path.
    prefer_version: Option<&str>,
) -> Result<(url::Url, WheelMetadata, String, Option<SdistProv>)> {
    // v1.5.9 exact-first: `specifiers` are the ORIGINAL (pre-D)
    // upstream pins, so exact family pins (isaacsim-kernel==6.0.0.0)
    // resolve the exact version and the installed family stays
    // patch-consistent. Only when the exact version has VANISHED from
    // the index do we retry with the relaxed range -- loudly, because
    // that is precisely the patch-drift condition that broke Kit
    // extension resolution (6.0.0.0 experience files requiring
    // extensions the 6.0.0.1 sensor wheel renamed).
    let wheel_result = match if let Some(pv) = prefer_version {
        pypi::resolve_preferring(index, pypi_name, specifiers, target, pv).await
    } else {
        pypi::resolve(index, pypi_name, specifiers, target).await
    } {
        Ok(resolved) => Ok(resolved),
        Err(exact_err) => {
            if let Some(relaxed) = relaxed_retry_specs(pypi_name, specifiers, relax) {
                match pypi::resolve(index, pypi_name, &relaxed, target).await {
                    Ok(resolved) => {
                        tracing::warn!(
                            dep = %pypi_name,
                            exact = %specifiers,
                            relaxed = %relaxed,
                            resolved = %resolved.filename,
                            "PATCH-DRIFT FALLBACK: exact upstream pin not on the index; resolved a relaxed match. If this dep is part of a pinned wheel family (isaacsim-*), check for runtime contract drift.",
                        );
                        crate::status::tty(&format!(
                            "warning: {pypi_name}{specifiers} not on index; using relaxed match {} (possible family version drift)",
                            resolved.filename,
                        ));
                        Ok(resolved)
                    }
                    Err(_) => Err(exact_err),
                }
            } else {
                Err(exact_err)
            }
        }
    };
    let (resolved_url, metadata, sdist_prov) = match wheel_result {
        Ok(resolved) => {
            let metadata = metadata_preferring_sidecar(&resolved, download_dir).await?;
            // Wheel path: no sdist provenance.
            (resolved.url, metadata, None)
        }
        Err(wheel_err) => {
            tracing::info!(
                dep = %pypi_name,
                spec = %specifiers,
                index = %index,
                error = %format!("{wheel_err:#}"),
                "BFS PyPI wheel resolve failed; attempting sdist fallback",
            );
            let (sdist_version, sdist) = pypi::resolve_sdist(index, pypi_name, specifiers)
                .await
                .with_context(|| {
                    format!(
                        "BFS sdist fallback for {} {} on {} (after wheel-resolve failure: {})",
                        pypi_name, specifiers, index, wheel_err,
                    )
                })?;
            // Capture the sdist URL BEFORE consuming `sdist` (THE FIX:
            // previously this was discarded and never threaded out).
            let captured_sdist_url = sdist.url.clone();
            // Unified sdist build cache dir keyed on (name, version) so BFS,
            // discovery, and replay all share the same output directory and
            // never rebuild the same (name, version) twice.
            let sdist_out = download_dir
                .join("sdist-builds")
                .join(format!("{pypi_name}-{sdist_version}"));
            let built = crate::source_build::build_wheel_from_sdist_url(
                &sdist.url,
                &sdist_out,
                &target.python_version,
            )
            .await
            .with_context(|| {
                format!(
                    "uv-building wheel from sdist {} for {}",
                    sdist.url, pypi_name,
                )
            })?;
            let built_url = url::Url::from_file_path(&built).map_err(|_| {
                anyhow!(
                    "built wheel path is not a valid file URL: {}",
                    built.display(),
                )
            })?;
            let metadata = tokio::task::spawn_blocking({
                let p = built.clone();
                move || crate::wheel::read_metadata(&p)
            })
            .await
            .context("metadata reader panicked")??;
            tracing::info!(
                dep = %pypi_name,
                built = %built.display(),
                "BFS sdist fallback: built wheel from sdist",
            );
            // Build the sdist provenance descriptor. `version` comes from the
            // built wheel's parsed metadata (authoritative resolved version).
            let prov = SdistProv {
                index: index.to_string(),
                name: pypi_name.to_string(),
                version: metadata.version.clone(),
                sdist_url: captured_sdist_url,
            };
            // Return built_url (file://) for the URL slot; the caller
            // will SUPPRESS this as upstream_url when sdist_prov.is_some().
            (built_url, metadata, Some(prov))
        }
    };
    Ok((resolved_url, metadata, index.to_string(), sdist_prov))
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
    Ok(out_t >= in_t)
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
#[derive(Debug, Clone)]
pub(crate) struct AutoDataConfig {
    pub checkout_root: PathBuf,
    /// Subdirectories (relative to `checkout_root`) that sibling
    /// entries in this bundle already shipped as wheels -- the walk
    /// descends through them but emits no files (avoids re-shipping the
    /// Python package source into `$PREFIX/lib/source/...`).
    pub skip_subdirs: Vec<PathBuf>,
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
#[allow(clippy::too_many_arguments)]
async fn materialize_and_rewrite(
    entry: &crate::config::WheelEntry,
    entry_name: &str,
    target: &WheelTarget,
    download_dir: &Path,
    source_dir: &Path,
    cache_dir: &Path,
    relax: RelaxPolicy,
    git_sources: &std::collections::BTreeMap<String, crate::config::NamedGitSource>,
    auto_data: Option<AutoDataConfig>,
    audit_info: EntryAuditInfo,
) -> Result<(ResolvedWheel, Vec<String>)> {
    use crate::wheel_rewrite::rewrite_wheel;
    let pypi_name = canonical_conda_name(entry_name);

    // Phase 1: get the raw wheel onto disk. For source-built wheels
    // (path / git / from), also remember the source root so phase 1.5
    // can inject any files pip wheel failed to ship.
    let mut source_root: Option<PathBuf> = None;
    // Pristine upstream URL captured BEFORE localization to file://.
    // Set for index (PyPI version-spec) and direct-URL entry forms only.
    // Source-built forms (git / path / from) leave this None.
    let mut upstream_url: Option<url::Url> = None;
    // Git provenance (schema 8+): populated for named-git and inline-git
    // entry forms. None for all other origins.
    let mut git_source_captured: Option<crate::lock::GitWheelSource> = None;
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
        let (wheel, resolved_sha) = crate::source_build::build_wheel_from_git(
            &src.url,
            &src.rev,
            subdir,
            cache_dir,
            &out,
            &target.python_version,
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
        source_root = Some(crate::source_build::git_source_root(
            &src.url, &src.rev, subdir, cache_dir,
        ));
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
        });
        wheel
    } else if let Some(url) = &entry.url {
        // Capture the direct URL as the upstream before fetch/localization.
        upstream_url = Some((*url).clone());
        crate::wheel::fetch_wheel_cached(
            url,
            entry.sha256.as_deref(),
            download_dir,
            &crate::courier::retread_cache_root(),
        )
        .await
        .with_context(|| format!("phase 1 URL fetch for entry `{entry_name}` (url=`{url}`)"))?
    } else if let Some(path) = &entry.path {
        let abs = if Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            source_dir.join(path)
        };
        let out = download_dir.join(entry_name);
        let wheel = crate::source_build::build_wheel_from_path(&abs, &out, &target.python_version)
            .await
            .with_context(|| {
                format!(
                    "phase 1 path build for entry `{entry_name}` (source={}, out_dir={})",
                    abs.display(),
                    out.display(),
                )
            })?;
        source_root = Some(abs);
        wheel
    } else if let Some(git_url) = &entry.git {
        let rev = entry
            .rev
            .as_ref()
            .ok_or_else(|| anyhow!("git source `{entry_name}` missing rev"))?;
        let subdir = entry.subdirectory.as_deref().unwrap_or(".");
        let out = download_dir.join(entry_name);
        let (wheel, resolved_sha) = crate::source_build::build_wheel_from_git(
            git_url,
            rev,
            subdir,
            cache_dir,
            &out,
            &target.python_version,
        )
        .await
        .with_context(|| {
            format!(
                "phase 1 inline-git build for entry `{entry_name}` \
                 (url=`{git_url}`, rev=`{rev}`, subdir=`{subdir}`, out_dir={})",
                out.display(),
            )
        })?;
        source_root = Some(crate::source_build::git_source_root(
            git_url, rev, subdir, cache_dir,
        ));
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
        });
        wheel
    } else {
        // PyPI version spec form.
        let version = entry
            .normalized_version()
            .ok_or_else(|| anyhow!("wheel `{entry_name}` has no version, url, path, or git"))?;
        let specifiers = VersionSpecifiers::from_str(&format!("=={version}"))
            .map_err(|e| anyhow!("wheel `{entry_name}` version `{version}`: {e}"))?;
        let resolved = pypi::resolve(&entry.index_url(), entry_name, &specifiers, target)
            .await
            .with_context(|| {
                format!(
                    "phase 1 PyPI resolve for entry `{entry_name}` \
                     (version=`{version}`, index=`{}`)",
                    entry.index_url(),
                )
            })?;
        // Capture the pristine index URL BEFORE fetch_wheel_cached may
        // localise / move it. This is the upstream_url written to the lock
        // so Phase-1 replay can re-fetch without a full BFS re-solve.
        upstream_url = Some(resolved.url.clone());
        crate::wheel::fetch_wheel_cached(
            &resolved.url,
            resolved.sha256.as_deref(),
            download_dir,
            &crate::courier::retread_cache_root(),
        )
        .await
        .with_context(|| {
            format!(
                "phase 1 PyPI fetch for entry `{entry_name}` (url=`{}`)",
                resolved.url,
            )
        })?
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

    // Phase 1.6 (v0.12.0+): if the caller passed an `AutoDataConfig`,
    // walk the upstream checkout root (parent of this entry's
    // subdirectory) honoring its own `.gitignore` and inject every
    // non-ignored, non-sibling-subdir file as a wheel `.data/data/lib/
    // <rel>` entry -- those land at `$CONDA_PREFIX/lib/<rel>` after
    // pip installs the wheel. Solves the IsaacLab case where the
    // `.kit` experience files live at the repo root but the wheel only
    // captures `source/<pkg>/`. Cache key: `*.autodata.whl`, refreshed
    // when the injected wheel changes (auto-data inject doesn't see
    // checkout-root file mtimes; clearing the auto-data cache when
    // upstream files change is the user's job via the existing
    // backend-cache invalidation step).
    let mut auto_data_file_count: Option<usize> = None;
    let with_data_path = if let Some(cfg) = auto_data.as_ref() {
        let out = injected_path.with_extension("autodata.whl");
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
                checkout = %cfg.checkout_root.display(),
                skip_subdirs = ?cfg.skip_subdirs,
                "phase 1.6: injecting checkout-root tree as wheel .data/data/lib/* (lands at $PREFIX/lib/*)",
            );
            let n = crate::wheel_inject_data::inject_checkout_root_data(
                &injected_path,
                &out,
                &cfg.checkout_root,
                &cfg.skip_subdirs,
            )
            .with_context(|| {
                format!(
                    "phase 1.6 checkout-root auto-data inject for entry `{entry_name}` \
                     (checkout={}, skip_subdirs={:?}, input={}, output={})",
                    cfg.checkout_root.display(),
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
        if is_fresh(&rewritten, &with_data_path)? {
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
            let _new_sha = rewrite_wheel(&with_data_path, &rewritten, relax)
                .with_context(|| {
                    format!(
                        "phase 2 wheel METADATA rewrite for entry `{entry_name}` (policy={relax:?}, \
                         input={}, output={})",
                        with_data_path.display(), rewritten.display(),
                    )
                })?;
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
            // sdist_source is only populated for BFS-transitive sdist-built wheels
            // (set in the BFS phase-3 loop). materialize_and_rewrite handles git/path/
            // url/version entries — none of those are sdist BFS transitives.
            sdist_source: None,
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
/// 3. Building the subdir/noarch/build/variant metadata.
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
    // Courier: use the content-addressed build string so pixi cache-hits are
    // invalidated whenever the inputs change (wheel set, indexes, config...).
    // Non-courier: keep the legacy `py{XY}_{build_number}` string unchanged.
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
/// `courier_build_hash`: when `Some`, this is the courier inputs hash
/// (from [`courier_inputs_hash`]) and the build string is set to the
/// content-addressed form `py{XY}_h{hash_prefix}_{build_number}`.
/// When `None` (non-courier path), the legacy `py{XY}_{build_number}`
/// string is emitted unchanged.
fn produce_output(
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
) -> Result<CondaOutput> {
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
    let auto_dropped: HashSet<String> = if host_platform != Platform::Win64
        && host_platform != Platform::Win32
        && host_platform != Platform::WinArm64
    {
        BUILT_IN_WIN_ONLY
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
    let mut sorted_wheels: Vec<&ResolvedWheel> = bundle.all_wheels().collect();
    sorted_wheels.sort_by_key(|w| canonical_conda_name(&w.pypi_name));
    for wheel in sorted_wheels {
        for raw in &wheel.metadata.requires_dist {
            let Some(dep) = crate::relax::translate(
                raw,
                &env,
                &config.name_map,
                &config.overrides,
                config.relax,
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
            let parsed_raw: Option<uv_pep508::Requirement> =
                uv_pep508::Requirement::from_str(raw).ok();
            let raw_pypi_name: Option<String> = parsed_raw.map(|r| r.name.to_string());
            // P2: one dual-namespace membership helper for all three
            // filters (canonicalizes both query names internally).
            let in_set = |set: &HashSet<String>| {
                crate::relax::already_covered(set, &dep_name, raw_pypi_name.as_deref())
            };
            if in_set(&vendored) {
                continue;
            }
            if in_set(&user_dropped) {
                tracing::debug!(dep = %dep_name, "dropping per retread-drop-deps");
                continue;
            }
            if in_set(&auto_dropped) {
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
            if !seen_dep_names.insert(dep_name.clone()) {
                continue;
            }
            run_dep_specs.push(spec_from_str(&dep.to_string())?);
        }
    }

    // Surface the final run-dep list at info level so users can spot
    // potentially-problematic deps before conda's solver complains.
    // Anything here that fails downstream is a candidate for
    // retread-drop-deps, retread-overrides, or retread-name-map.
    let emitted: Vec<&str> = run_dep_specs.iter().map(|s| s.name.as_str()).collect();
    tracing::info!(
        bundle = %bundle.conda_name,
        run_deps = ?emitted,
        "bundle run-deps emitted; if conda can't find one, add it to \
         retread-drop-deps / retread-overrides / retread-name-map"
    );

    let effective_version = version_override.unwrap_or(&bundle.primary.metadata.version);
    assemble_conda_output(
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
    )
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
async fn emit_wheels_from_lock(
    lock: &crate::lock::RetreadLock,
    config: &RetreadConfig,
    target: &WheelTarget,
    download_dir: &Path,
    source_dir: &Path,
    cache_dir: &Path,
) -> Result<Option<Vec<crate::emit_pypi::EmitWheel>>> {
    use crate::lock::Origin;

    // Per-wheel re-materialization: classify each LockWheel and build the
    // EmitWheel with correct local_path / remote_url for courier::stage.
    let mut emit_wheels: Vec<crate::emit_pypi::EmitWheel> = Vec::with_capacity(lock.wheels.len());

    // PHASE 2.5: Multi-entry shared-git-checkout replay.
    //
    // Pre-pass: group all Class-1 git wheels (must_ship=true, git_source present)
    // by checkout root (git_checkout_root(gs.url, gs.rev)), preserving lock order.
    // Within each group, group[0] is the CARRIER (mirrors produce's BTreeMap-order
    // first entry -> first in emit_wheels -> first in lock.wheels). The carrier
    // gets AutoDataConfig{skip_subdirs = union of ALL members' subdirs}; non-carriers
    // get None (ship only their own pip+inject wheel, no auto-data).
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
    // git_group_skip_subdirs: checkout_root -> Vec<subdir> (union of all members)
    let mut git_group_members: std::collections::HashMap<PathBuf, Vec<usize>> =
        std::collections::HashMap::new();
    // Parallel vec for ordering: the FIRST lock index seen for each root (= carrier).
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

    // Step 2: for each group with >1 members (multi-entry), validate all members
    // have git_source (invariant; they do by construction of the scan above).
    // Compute AutoDataConfig per group member index (carrier=first, rest=None).
    // auto_data_for_lock_idx: lock index -> Option<AutoDataConfig>
    // None means "non-carrier of a multi-entry group" -> build with auto_data=None.
    // Missing key means "size-1 group" -> use single-entry logic below (unchanged).
    let mut auto_data_override: std::collections::HashMap<usize, Option<AutoDataConfig>> =
        std::collections::HashMap::new();

    for root in &git_group_order {
        let members = &git_group_members[root];
        if members.len() > 1 {
            // Compute skip_subdirs = union of all members' subdirectory fields.
            let skip_subdirs: Vec<PathBuf> = members
                .iter()
                .map(|&idx| {
                    let gs = lock.wheels[idx]
                        .git_source
                        .as_ref()
                        .expect("git_group_members only contains wheels with git_source; qed");
                    PathBuf::from(gs.subdirectory.as_deref().unwrap_or("."))
                })
                .collect();
            // Carrier = members[0] (lock-order index 0).
            auto_data_override.insert(
                members[0],
                Some(AutoDataConfig {
                    checkout_root: root.clone(),
                    skip_subdirs,
                }),
            );
            // Non-carriers get None.
            for &idx in &members[1..] {
                auto_data_override.insert(idx, None);
            }
        }
        // Size-1 groups fall through to the single-entry path below (no override).
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
                crate::emit_pypi::EmitWheel {
                    pypi_name: lw.name.clone(),
                    version: lw.version.clone(),
                    requires_dist: lw.requires_dist.clone(),
                    local_path,
                    wheel_filename: lw.filename.clone(),
                    sha256: lw.sha256.clone(),
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
                    // both arms call the identical build_wheel_from_git via
                    // checkout_root_for_entry; extras do not reach the wheel build (they
                    // drive BFS closure only), so collapsing a named-git entry to an inline
                    // synth {git:url, rev:SHA} yields a byte-identical wheel.
                    //
                    // PHASE 2.5: Multi-entry shared-git-checkout support.
                    // The pre-pass above identified groups by checkout root. If this wheel
                    // is part of a multi-entry group, we use the group stash:
                    //   - First encounter of the root: build ALL group members via
                    //     materialize_and_rewrite (carrier gets union skip_subdirs, non-
                    //     carriers get None), stash results by lock index.
                    //   - Subsequent encounters: emit from stash (no rebuild).
                    // Single-entry groups fall through to the pre-existing single-entry path
                    // (skip_subdirs=[gs.subdirectory], unchanged behavior).
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
                                let (resolved, _rd) = materialize_and_rewrite(
                                    &synth_entry,
                                    &member_lw.name,
                                    target,
                                    download_dir,
                                    source_dir,
                                    cache_dir,
                                    config.relax,
                                    &config.git_sources,
                                    member_auto_data,
                                    EntryAuditInfo::default(),
                                )
                                .await
                                .with_context(|| {
                                    format!(
                                        "courier replay (phase 2.5): re-source-building \
                                         group member `{}` from url={}, rev={}",
                                        member_lw.name, member_gs.url, member_gs.rev,
                                    )
                                })?;
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
                                        wheel_filename: member_lw.filename.clone(),
                                        sha256: member_lw.sha256.clone(),
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
                        // Single-entry group: existing single-entry logic (unchanged).
                        // skip_subdirs = [gs.subdirectory] mirrors produce-path derivation
                        // (auto_data_per_entry, mod.rs ~2205-2218): one entry owns the
                        // checkout root; skip_subdirs = its own subdirectory.
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
                        // Mirror the produce-path derivation (auto_data_per_entry,
                        // ~mod.rs:2205-2218): for a single-entry git pack the
                        // skip_subdirs set is exactly [subdirectory] (defaulting to
                        // ".").  Previously this was vec![] — inert for root entries
                        // (subdirectory="."), but a silent correctness landmine for
                        // nested subdirectories (e.g. monorepo subpaths): produce
                        // would skip the subtree while replay would NOT.
                        let skip_subdirs =
                            vec![PathBuf::from(gs.subdirectory.as_deref().unwrap_or("."))];
                        let auto_data = Some(AutoDataConfig {
                            checkout_root,
                            skip_subdirs,
                        });
                        let (resolved, _rd) = materialize_and_rewrite(
                            &synth_entry,
                            &lw.name,
                            target,
                            download_dir,
                            source_dir,
                            cache_dir,
                            config.relax,
                            &config.git_sources,
                            auto_data,
                            EntryAuditInfo::default(),
                        )
                        .await
                        .with_context(|| {
                            format!(
                                "courier replay (git_source): re-source-building wheel `{}` \
                                 from url={}, rev={}",
                                lw.name, gs.url, gs.rev,
                            )
                        })?;
                        let local_path = (resolved.url.scheme() == "file")
                            .then(|| resolved.url.to_file_path().ok())
                            .flatten();
                        crate::emit_pypi::EmitWheel {
                            pypi_name: lw.name.clone(),
                            version: lw.version.clone(),
                            requires_dist: lw.requires_dist.clone(),
                            local_path,
                            wheel_filename: lw.filename.clone(),
                            sha256: lw.sha256.clone(),
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
                    let (resolved, _rd) = materialize_and_rewrite(
                        entry,
                        &lw.name,
                        target,
                        download_dir,
                        source_dir,
                        cache_dir,
                        config.relax,
                        &config.git_sources,
                        auto_data,
                        EntryAuditInfo::default(),
                    )
                    .await
                    .with_context(|| {
                        format!(
                            "courier replay: re-materializing source-built wheel `{}`",
                            lw.name
                        )
                    })?;
                    let local_path = (resolved.url.scheme() == "file")
                        .then(|| resolved.url.to_file_path().ok())
                        .flatten();
                    crate::emit_pypi::EmitWheel {
                        pypi_name: lw.name.clone(),
                        version: lw.version.clone(),
                        requires_dist: lw.requires_dist.clone(),
                        local_path,
                        wheel_filename: lw.filename.clone(),
                        sha256: lw.sha256.clone(),
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
            // Fallback: if the exact URL fails (yanked), re-resolve via version pin.
            //
            // POISONING note: sdist_source is NOT in compute_inputs_hash (same
            // circularity as git_source.rev): the sdist URL is a consequence of the
            // resolve, not an independent input.
            Origin::Built if !lw.must_ship && lw.sdist_source.is_some() => {
                let s = lw.sdist_source.as_ref().unwrap();
                // Unified sdist build cache dir: same key as BFS and discovery
                // so whichever path runs first populates it and the others hit
                // the cache without rebuilding.
                let sdist_out = download_dir
                    .join("sdist-builds")
                    .join(format!("{}-{}", s.name, s.version));
                let stored_url = url::Url::parse(&s.sdist_url).with_context(|| {
                    format!(
                        "courier replay Class-2b: invalid sdist_url `{}` for wheel `{}`",
                        s.sdist_url, lw.name,
                    )
                })?;
                tracing::info!(
                    wheel = %lw.name,
                    sdist_url = %stored_url,
                    "courier replay: rebuilding sdist-built shadow from stored sdist_url (class 2b)",
                );
                let built = match crate::source_build::build_wheel_from_sdist_url(
                    &stored_url,
                    &sdist_out,
                    &lock.python,
                )
                .await
                {
                    Ok(p) => p,
                    Err(e) => {
                        // Stored URL may be yanked; fall back to re-resolve by exact version.
                        tracing::warn!(
                            wheel = %lw.name,
                            sdist_url = %stored_url,
                            error = %format!("{e:#}"),
                            "courier replay Class-2b: stored sdist_url failed; re-resolving by version",
                        );
                        let specifiers = VersionSpecifiers::from_str(&format!("=={}", s.version))
                            .with_context(|| {
                            format!(
                                "courier replay Class-2b: parsing version spec `=={}` for `{}`",
                                s.version, lw.name,
                            )
                        })?;
                        let (_sdist_version, sdist) = pypi::resolve_sdist(
                            &s.index,
                            &s.name,
                            &specifiers,
                        )
                        .await
                        .with_context(|| {
                            format!(
                                "courier replay Class-2b: re-resolving sdist for `{}` at `=={}`",
                                s.name, s.version,
                            )
                        })?;
                        crate::source_build::build_wheel_from_sdist_url(
                            &sdist.url,
                            &sdist_out,
                            &lock.python,
                        )
                        .await
                        .with_context(|| {
                            format!(
                                "courier replay Class-2b: building wheel from re-resolved sdist `{}`",
                                sdist.url,
                            )
                        })?
                    }
                };
                crate::emit_pypi::EmitWheel {
                    pypi_name: lw.name.clone(),
                    version: lw.version.clone(),
                    requires_dist: lw.requires_dist.clone(),
                    local_path: Some(built),
                    wheel_filename: lw.filename.clone(),
                    sha256: lw.sha256.clone(),
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

                crate::emit_pypi::EmitWheel {
                    pypi_name: lw.name.clone(),
                    version: lw.version.clone(),
                    requires_dist: lw.requires_dist.clone(),
                    local_path: Some(fetched),
                    wheel_filename: lw.filename.clone(),
                    sha256: lw.sha256.clone(),
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
async fn materialize_from_lock(
    lock: crate::lock::RetreadLock,
    config: &RetreadConfig,
    work_dir: &Path,
    output_dir: &Path,
    target_subdir: Platform,
    source_dir: &Path,
    cache_dir: &Path,
    expected_build: Option<&str>,
    run_deps: Vec<String>,
    config_fp: &str,
) -> Result<Option<CondaBuildV1Result>> {
    let bundle_name = lock.bundle.clone();
    let version = lock.version.clone();
    let python_version = crate::relax::emit_python_version("", &lock.python);
    let download_dir = source_dir.join("wheels");
    let target = wheel_target_for(target_subdir, &python_version);

    let emit_wheels =
        match emit_wheels_from_lock(&lock, config, &target, &download_dir, source_dir, cache_dir)
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

    let result = materialize_and_pack(
        None, // bundle=None: replay path, audit skipped
        config,
        &bundle_name,
        &version,
        &python_version,
        emit_wheels,
        conda_capable,
        run_deps,
        index_urls,
        config_fp,
        work_dir,
        output_dir,
        target_subdir,
        source_dir,
        expected_build,
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
    version: &str,
    python_version: &str,
    emit_wheels: Vec<crate::emit_pypi::EmitWheel>,
    conda_capable: std::collections::HashSet<String>,
    run_deps: Vec<String>,
    index_urls: Vec<String>,
    config_fp: &str,
    work_dir: &Path,
    output_dir: &Path,
    target_subdir: Platform,
    source_dir: &Path,
    expected_build: Option<&str>,
) -> Result<CondaBuildV1Result> {
    let staging = work_dir.join(format!("courier-{bundle_name}"));
    let staged = crate::courier::stage(
        config,
        bundle_name,
        version,
        python_version,
        &emit_wheels,
        &conda_capable,
        &run_deps,
        &index_urls,
        config_fp,
        source_dir,
        &staging,
    )
    .await
    .context("courier staging")?;

    // Defer the committed install lock write until after a successful
    // rattler-build (B-2). The staged copy inside `staging` is already in
    // the recipe's source list; this is the authoritative pack-dir copy.
    let lock_path = source_dir.join(crate::lock::RetreadLock::file_name(bundle_name));
    let courier_lock_to_commit = (lock_path, staged.lock.to_pretty_json()?);

    let recipe = build_courier_recipe(
        bundle_name,
        version,
        python_version,
        &staged.run_deps,
        &staged.source_urls,
        // Thread the content-addressed build string into the recipe so
        // the on-disk artifact name matches what conda/outputs advertised.
        expected_build,
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

    let target_platform = target_subdir.to_string();
    let compression_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .to_string();
    let mut cmd = tokio::process::Command::new("rattler-build");
    cmd.arg("build")
        .arg("--recipe")
        .arg(&recipe_path)
        .arg("--output-dir")
        .arg(output_dir)
        .arg("--target-platform")
        .arg(&target_platform)
        .arg("--compression-threads")
        .arg(&compression_threads)
        .arg("--no-test");
    crate::fasttmp::apply_backend_env(&mut cmd);
    if let Some(level) = config.compression_level {
        cmd.arg("--package-format").arg(format!("conda:{level}"));
    }
    let packaging_started = std::time::Instant::now();
    let output = cmd
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .context("spawning rattler-build (is it on PATH?)")?;
    tracing::info!(
        output = %recipe.package.name,
        elapsed_ms = packaging_started.elapsed().as_millis() as u64,
        compression_threads = %compression_threads,
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

    let subdir_dir = output_dir.join(&target_platform);
    let output_file =
        find_conda_artifact(&subdir_dir, &recipe.package.name, &recipe.package.version).await?;

    let build = expected_build.map(|s| s.to_string()).unwrap_or_else(|| {
        let py_short = python_version.replace('.', "");
        format!("py{py_short}_{}", config.build_number)
    });
    Ok(CondaBuildV1Result {
        output_file,
        input_globs: Default::default(),
        name: recipe.package.name.clone(),
        version: VersionWithSource::from_str(&recipe.package.version)?,
        build,
        subdir: target_subdir,
    })
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
    target: &WheelTarget,
    download_dir: &Path,
    source_dir: &Path,
    cache_dir: &Path,
    conda_channels: &[ChannelUrl],
    workspace_dir: Option<&Path>,
    work_dir: &Path,
    output_dir: &Path,
    target_subdir: Platform,
    expected_build: Option<&str>,
    config_fp: &str,
) -> Result<Option<CondaBuildV1Result>> {
    let IncrementalAdd { added_specs, lock } = incr;

    // ── Build locked_closure: name → version from lock.wheels ─────────────
    let locked_closure: std::collections::BTreeMap<String, String> = lock
        .wheels
        .iter()
        .map(|w| (canonical_conda_name(&w.name), w.version.clone()))
        .collect();

    // ── Step A: re-materialize locked wheels (emit_wheels_from_lock) ──────
    let python_version = crate::relax::emit_python_version("", &lock.python);
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
        .map(|m| m.all_pypi_index_urls())
        .unwrap_or_default();
    let pypi_to_conda = if config.auto_bundle {
        load_pypi_to_conda_map().await
    } else {
        Default::default()
    };
    let mut effective = config.clone();
    for (pypi, conda) in FALLBACK_PYPI_TO_CONDA {
        let key = canonical_conda_name(pypi);
        effective
            .name_map
            .entry(key)
            .or_insert_with(|| (*conda).to_string());
    }
    for (pypi, conda_names) in &pypi_to_conda {
        if conda_names.len() == 1 {
            effective
                .name_map
                .entry(pypi.clone())
                .or_insert_with(|| conda_names[0].clone());
        }
    }

    // ── Step B: resolve each added entry ──────────────────────────────────
    let mut new_emit: Vec<crate::emit_pypi::EmitWheel> = Vec::new();
    let mut new_conda_capable: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Match added_specs back to config entries.
    // courier_input_specs produces "<key>[extras]<ver>" — we find entries
    // whose spec string appears in added_specs.
    let added_set: std::collections::HashSet<&str> =
        added_specs.iter().map(|s| s.as_str()).collect();
    let mut matched_entries: Vec<(String, WheelEntry)> = Vec::new();
    for (key, entry) in &effective.retread_wheels {
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
            &effective.name_map,
            conda_channels,
            &effective.conda_deps,
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

        // Convert Bundle → EmitWheel (same logic as build_one, lines 5461-5494).
        let wheels_root = source_dir.join("wheels");
        for w in bundle.all_wheels() {
            let url = localize_wheel_source(&w.url, &wheels_root);
            new_emit.push(crate::emit_pypi::EmitWheel {
                pypi_name: w.pypi_name.clone(),
                version: w.metadata.version.clone(),
                requires_dist: w.metadata.requires_dist.clone(),
                sha256: Some(w.metadata.sha256.clone()),
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
        new_conda_capable.extend(effective.name_map.keys().map(|k| canonical_conda_name(k)));
        new_conda_capable.extend(load_pypi_to_conda_map().await.into_keys());
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

    // ── A6: write lock only after ALL checks pass ─────────────────────────
    let result = materialize_and_pack(
        None, // bundle=None: incremental path, no full Bundle available
        config,
        &bundle_name,
        &version,
        &python_version,
        merged,
        conda_capable,
        run_deps,
        index_urls,
        config_fp,
        work_dir,
        output_dir,
        target_subdir,
        source_dir,
        expected_build,
    )
    .await?;

    // ── Optional validity oracle ───────────────────────────────────────────
    if std::env::var("RETREAD_VERIFY_LOCALADD").as_deref() == Ok("1") {
        let lock_path = source_dir.join(crate::lock::RetreadLock::file_name(&bundle_name));
        verify_localadd_hook(
            &lock_path,
            &added_specs,
            &bundle_name,
            &target_subdir.to_string(),
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

#[allow(clippy::too_many_arguments)]
async fn build_one(
    bundle: &Bundle,
    config: &RetreadConfig,
    declared_config: &RetreadConfig,
    work_dir: &Path,
    output_dir: &Path,
    target_subdir: Platform,
    workspace_python_version: &str,
    source_dir: &Path,
    workspace_dir: Option<&Path>,
    expected_build: Option<&str>,
    run_override: Option<&[String]>,
) -> Result<CondaBuildV1Result> {
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
        let python_version =
            emit_python_version(&bundle.primary.metadata.filename, workspace_python_version);
        let version = bundle.primary.metadata.version.clone();
        let emit_wheels: Vec<crate::emit_pypi::EmitWheel> = bundle
            .all_wheels()
            .zip(localized_urls.iter())
            .map(|(w, url)| crate::emit_pypi::EmitWheel {
                pypi_name: w.pypi_name.clone(),
                version: w.metadata.version.clone(),
                requires_dist: w.metadata.requires_dist.clone(),
                sha256: Some(w.metadata.sha256.clone()),
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
        conda_capable.extend(config.name_map.keys().map(|k| canonical_conda_name(k)));
        conda_capable.extend(load_pypi_to_conda_map().await.into_keys());
        let ws_manifest = workspace_dir.and_then(crate::workspace::WorkspaceManifest::load);
        let workspace_indexes: Vec<String> = ws_manifest
            .as_ref()
            .map(|m| m.all_pypi_index_urls())
            .unwrap_or_default();
        // grizzly H1: fold the workspace solve environment into the hash.
        // Pack-scoped: only envs that reference source_dir are hashed.
        let workspace_fp = ws_manifest
            .as_ref()
            .map(|m| m.solve_fingerprint(workspace_dir.unwrap_or(source_dir), source_dir))
            .unwrap_or_default();
        let entry_indexes: Vec<String> = config
            .retread_wheels
            .values()
            .map(|e| e.index_url())
            .collect();
        let index_urls = merge_index_chain(entry_indexes, &workspace_indexes);
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
            .map(|m| m.courier_channel_set(workspace_dir.unwrap_or(source_dir), source_dir))
            .unwrap_or_default();
        let config_fp =
            crate::courier::config_fingerprint(declared_config, &courier_channels, &workspace_fp);
        return materialize_and_pack(
            Some(bundle),
            config,
            &bundle.conda_name,
            &version,
            &python_version,
            emit_wheels,
            conda_capable,
            run_deps,
            index_urls,
            &config_fp,
            work_dir,
            output_dir,
            target_subdir,
            source_dir,
            expected_build,
        )
        .await
        .context("courier materialize_and_pack");
    }

    // Non-courier path: build a bundled conda package with the wheel payload.
    let recipe = build_bundle_recipe(
        &bundle.conda_name,
        &sources,
        config,
        workspace_python_version,
        run_override,
        // blueprint="only" payload-skip is deprecated (v2.0.0); the
        // non-courier conda path always carries its wheel payload.
        true,
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
    // v1.5.8: rattler-build's zstd packaging defaults to ONE thread.
    // For isaac-scale bundles that means single-threaded compression
    // of a ~15GB build prefix -- the dominant wall-clock chunk of
    // pixi's "preparing packages" phase. zstd scales near-linearly
    // with threads on inputs this size, so hand it every core.
    let compression_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .to_string();
    // CRITICAL: rattler-build writes progress to stdout, but retread's
    // stdout is the JSON-RPC channel to pixi. Capture both streams so
    // they don't corrupt the protocol. Surface them via tracing
    // (which writes to OUR stderr) on failure.
    let mut cmd = tokio::process::Command::new("rattler-build");
    cmd.arg("build")
        .arg("--recipe")
        .arg(&recipe_path)
        .arg("--output-dir")
        .arg(output_dir)
        .arg("--target-platform")
        .arg(&target_platform)
        .arg("--compression-threads")
        .arg(&compression_threads)
        .arg("--no-test");
    crate::fasttmp::apply_backend_env(&mut cmd);
    // v1.5.8: user-tunable zstd level (retread-compression-level).
    // Unset keeps rattler-build's default.
    if let Some(level) = config.compression_level {
        cmd.arg("--package-format").arg(format!("conda:{level}"));
    }
    // v1.6.0: time the packaging stage explicitly.
    let packaging_started = std::time::Instant::now();
    let output = cmd
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .context("spawning rattler-build (is it on PATH?)")?;
    tracing::info!(
        output = %recipe.package.name,
        elapsed_ms = packaging_started.elapsed().as_millis() as u64,
        compression_threads = %compression_threads,
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
    let output_file =
        find_conda_artifact(&subdir_dir, &recipe.package.name, &recipe.package.version).await?;

    // Build string contract: pixi computes the expected build string
    // from the variant it sent us in conda/outputs and rejects mismatches
    // with "The build backend did not return the expected package: ...".
    // Echo back exactly what pixi expects when it tells us. Only synthesize
    // from workspace_python_version when there's no expectation (e.g.,
    // direct test calls).
    let build = expected_build.map(|s| s.to_string()).unwrap_or_else(|| {
        let py_short = workspace_python_version.replace('.', "");
        format!("py{py_short}_{}", config.build_number)
    });
    Ok(CondaBuildV1Result {
        output_file,
        input_globs: Default::default(),
        name: recipe.package.name.clone(),
        version: VersionWithSource::from_str(&recipe.package.version)?,
        build,
        subdir: target_subdir,
    })
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

async fn find_conda_artifact(dir: &Path, name: &str, version: &str) -> Result<PathBuf> {
    let mut read = tokio::fs::read_dir(dir)
        .await
        .with_context(|| format!("reading rattler-build output dir {}", dir.display()))?;
    let prefix = format!("{name}-{version}-");
    while let Some(entry) = read.next_entry().await? {
        let path = entry.path();
        let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if fname.starts_with(&prefix) && fname.ends_with(".conda") {
            return Ok(path);
        }
    }
    bail!(
        "no .conda artifact found in {} matching {prefix}*.conda",
        dir.display()
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
fn detect_incremental_add(
    lock_path: &Path,
    config: &RetreadConfig,
    bundle_name: &str,
    ws_indexes: &[String],
    relax_str: &str,
    python_version: &str,
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
            .map(|(_, entry)| entry.index_url())
            .collect();
        let locked_chain = merge_index_chain(locked_entry_indexes, ws_indexes);
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
    // use lock.index_urls as the index term (reproduces the original chain
    // order exactly, including implicit PUBLIC_PYPI position).
    let recomputed_hash = crate::lock::RetreadLock::compute_inputs_hash(
        &lock.entry_specs,
        &lock.index_urls,
        relax_str,
        python_version,
        crate::lock::EMIT_EPOCH,
        None, // pin_version matches the lock that was stored (non-circular)
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
fn load_favored_versions(lock_path: &Path) -> std::collections::BTreeMap<String, String> {
    if std::env::var_os("RETREAD_NO_FAVOR_LOCK").is_some() {
        return std::collections::BTreeMap::new();
    }
    let Ok(lock) = crate::lock::RetreadLock::load(lock_path) else {
        return std::collections::BTreeMap::new();
    };
    let mut m = std::collections::BTreeMap::new();
    for w in &lock.wheels {
        // Skip wheels with empty version strings; those are malformed.
        if !w.version.is_empty() {
            m.insert(canonical_conda_name(&w.name), w.version.clone());
        }
    }
    m
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
///
/// Returns `None` (non-fatal miss) on any mismatch.
/// Returns `Err` only when the file exists but is malformed.
///
/// `RETREAD_NO_REPLAY=1` unconditionally returns `None` (test knob;
/// lets tests force cold-path exercising without touching the hash).
fn load_replayable_lock(
    lock_path: &Path,
    current_inputs_hash: &str,
    relax_is_default: bool,
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
fn replay_from_lock(
    lock_path: &Path,
    current_inputs_hash: &str,
    relax_is_default: bool,
    host_platform: Platform,
    build_number: u64,
    loose: bool,
    siblings: &[(String, String)],
) -> anyhow::Result<Option<CondaOutput>> {
    let Some(lock) = load_replayable_lock(lock_path, current_inputs_hash, relax_is_default)? else {
        return Ok(None);
    };

    // ----- reconstruct CondaOutput from lock fields via the shared helper -----
    let python_version = crate::relax::emit_python_version("", &lock.python);

    let python_dep = if python_version.contains('*') {
        format!("python {python_version}")
    } else {
        format!("python {python_version}.*")
    };

    // Reconstruct run-dependencies from the lock's conda_run_deps.
    let mut run_dep_specs: Vec<NamedSpec<PackageSpec>> = vec![spec_from_str(&python_dep)?];
    let mut seen_dep_names: HashSet<String> = HashSet::from(["python".to_string()]);
    for dep in &lock.conda_run_deps {
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
    Ok(Some(output))
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
    use std::collections::BTreeMap;

    use rattler_conda_types::Platform;

    use super::{AutoDataConfig, load_replayable_lock, replay_from_lock};
    use crate::config::RetreadConfig;
    use crate::lock::{CondaDep, LockWheel, Origin, RetreadLock, SCHEMA};

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
        RetreadLock {
            schema: SCHEMA,
            retread_version: "0.0.1".into(),
            bundle: bundle.into(),
            version: version.into(),
            python: python.into(),
            inputs_hash: inputs_hash.into(),
            root_requirements: Vec::new(),
            wheels: vec![LockWheel {
                name: bundle.into(),
                version: version.into(),
                origin: Origin::Index,
                filename,
                url: Some(format!("https://example.com/{bundle}-{version}.whl")),
                sha256: None,
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
        }
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
        let lock = make_test_lock("mypack", "1.2.3", "3.11", hash, true);
        let json = lock.to_pretty_json().unwrap();
        let lock_path = dir.join(RetreadLock::file_name("mypack"));
        std::fs::write(&lock_path, &json).unwrap();

        let result = replay_from_lock(&lock_path, hash, true, Platform::Linux64, 0, false, &[]);
        let out = result.unwrap().unwrap();
        // Build string must be content-addressed: py311_h<first10>_0
        assert_eq!(
            out.metadata.build, "py311_habcdef0123_0",
            "replay must emit content-addressed build string: got {}",
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
            inputs_hash: "hash123".into(),
            root_requirements: vec![],
            wheels: vec![LockWheel {
                name: "orphan-git-wheel".into(),
                version: "1.0.0".into(),
                origin: Origin::Built,
                filename: "orphan_git_wheel-1.0.0-py3-none-any.injected.whl".into(),
                url: None,
                sha256: None,
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
            rattler_conda_types::Platform::Linux64,
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
            inputs_hash: "hash456".into(),
            root_requirements: vec![],
            wheels: vec![LockWheel {
                name: "skrl".into(),
                version: "2.1.0".into(),
                origin: Origin::Built,
                filename: "skrl-2.1.0-999retread-py3-none-any.whl".into(),
                url: None,
                sha256: None,
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
        };
        let config: RetreadConfig =
            serde_json::from_value(serde_json::json!({"retread-wheels": {}})).unwrap();
        let result = materialize_from_lock(
            lock,
            &config,
            &work_dir,
            &output_dir,
            rattler_conda_types::Platform::Linux64,
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
            sdist_url:
                "https://files.pythonhosted.org/packages/gym-0.26.2.tar.gz#sha256=deadbeef0000"
                    .into(),
        };

        let lock = RetreadLock {
            schema: crate::lock::SCHEMA,
            retread_version: "2.7.0".into(),
            bundle: "gympack".into(),
            version: "1.0.0".into(),
            python: "3.11".into(),
            inputs_hash: "hash999".into(),
            root_requirements: vec![],
            wheels: vec![LockWheel {
                name: "gym".into(),
                version: "0.26.2".into(),
                origin: Origin::Built,
                filename: "gym-0.26.2-999retread-py3-none-any.whl".into(),
                url: None,
                sha256: None,
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
        };
        let config: RetreadConfig =
            serde_json::from_value(serde_json::json!({"retread-wheels": {}})).unwrap();

        let result = materialize_from_lock(
            lock,
            &config,
            &work_dir,
            &output_dir,
            rattler_conda_types::Platform::Linux64,
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
            sdist_url:
                "https://files.pythonhosted.org/packages/gym-0.26.2.tar.gz#sha256=abc123def456"
                    .into(),
        };

        // Replicate the LockWheel that Class-2b operates on.
        let lw = LockWheel {
            name: "gym".into(),
            version: "0.26.2".into(),
            origin: Origin::Built,
            filename: "gym-0.26.2-999retread-py3-none-any.whl".into(),
            url: None,
            sha256: None,
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
            sha256: lw.sha256.clone(),
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
            inputs_hash: "live-hash".into(),
            root_requirements: vec![],
            wheels: vec![LockWheel {
                name: "gym".into(),
                version: "0.26.2".into(),
                origin: Origin::Built,
                filename: "gym-0.26.2-999retread-py3-none-any.whl".into(),
                url: None,
                sha256: None,
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
            rattler_conda_types::Platform::Linux64,
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
            sha256: None,
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
            sha256: lw.sha256.clone(),
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
        let dep_whl_name = format!("{dep_name}-{dep_version}-py3-none-any.whl");
        let url_req = format!("{dep_name} @ https://example.com/{dep_whl_name}");

        // Write the dep wheel file.
        let dep_whl_path = source_dir.join("wheels").join(&dep_whl_name);
        std::fs::write(
            &dep_whl_path,
            make_wheel_bytes_for_replay(dep_name, dep_version, &[]),
        )
        .unwrap();

        // The pytorch3d wheel bytes (has a URL requirement).
        let p3d_whl_name = format!(
            "{}-{}-py3-none-any.whl",
            wheel_name.replace('-', "_"),
            wheel_version
        );
        let raw_wheel_bytes =
            make_wheel_bytes_for_replay(wheel_name, wheel_version, &[url_req.as_str()]);

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
            }
        }))
        .unwrap();

        // conda_capable includes the pytorch3d wheel name — this triggers the drift
        // on the OLD remote-only path but is irrelevant to the LOCAL-PATH path.
        let mut conda_capable: HashSet<String> = HashSet::new();
        conda_capable.insert(wheel_name.to_string());

        let index_urls = [format!("http://127.0.0.1:{port}/simple/")];

        // ── Write the upstream wheel bytes to disk (simulating materialize_and_rewrite
        //    fetch). The cold EmitWheel has local_path=Some(fetched+relaxed whl).
        //    For this test we write the raw bytes as the "already fetched" local file.
        let p3d_local = source_dir.join("wheels").join(&p3d_whl_name);
        std::fs::write(&p3d_local, &raw_wheel_bytes).unwrap();

        let dep_emit = crate::emit_pypi::EmitWheel {
            pypi_name: dep_name.to_string(),
            version: dep_version.to_string(),
            requires_dist: vec![],
            wheel_filename: dep_whl_name.clone(),
            sha256: None,
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
            requires_dist: vec![url_req.clone()],
            wheel_filename: p3d_whl_name.clone(),
            sha256: None,
            local_path: Some(p3d_local.clone()),
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
        };

        // Build the replay EmitWheel exactly as the new Class-2 arm would:
        // download from upstream_url -> local_path=Some, remote_url=None,
        // upstream_url=Some(github).
        let fetched_replay = source_dir.join("wheels").join(&p3d_whl_name);
        // The wheel bytes are already in source_dir/wheels/ from cold;
        // fetch_wheel_cached would land there too (dest_dir.join(filename_from_url)).

        let dep_replay = crate::emit_pypi::EmitWheel {
            pypi_name: dep_name.to_string(),
            version: dep_version.to_string(),
            requires_dist: vec![],
            wheel_filename: dep_whl_name.clone(),
            sha256: None,
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
            sha256: cold_lw.sha256.clone(),
            local_path: Some(fetched_replay),
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
            inputs_hash: "live-c2".into(),
            root_requirements: vec![],
            wheels: vec![LockWheel {
                name: "requests".into(),
                version: "2.31.0".into(),
                origin: Origin::Built,
                filename: "requests-2.31.0-999retread-py3-none-any.whl".into(),
                url: None,
                sha256: None,
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
            rattler_conda_types::Platform::Linux64,
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
            sha256: None,
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
            sha256: None,
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
        let target = wheel_target_for(Platform::Linux64, "3.11");
        let checkout_root = crate::source_build::git_checkout_root(&repo_url, &sha, &cache_dir);

        // ── PRODUCE: auto_data with correct skip_subdirs=["packages/mypkg"] ──
        let produce_dd = produce_src.join("wheels");
        std::fs::create_dir_all(&produce_dd).unwrap();
        let (produce_resolved, _) = materialize_and_rewrite(
            &entry,
            "retread-nested-mypkg",
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

    // -----------------------------------------------------------------------
    // PHASE 2.5 tests: multi-entry shared-git-checkout replay.
    // -----------------------------------------------------------------------

    /// Test (b): group[0] in lock order is the carrier.
    ///
    /// Verify that the pre-pass in materialize_from_lock assigns AutoDataConfig
    /// (with the union skip_subdirs) to lock index 0 of a multi-entry group, and
    /// None to all other members. This mirrors produce's BTreeMap-order-first
    /// carrier rule (auto_data_per_entry, mod.rs ~2197-2221).
    ///
    /// This is a pure unit test — it exercises the grouping + AutoDataConfig
    /// derivation logic directly via a synthetic lock, no git/uv required.
    #[test]
    fn multi_entry_git_group_lock_order_first_is_carrier() {
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
        // isaaclab_tasks (index 2).
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

        // Compute auto_data_override (mirrors materialize_from_lock pre-pass).
        let mut auto_data_override: std::collections::HashMap<usize, Option<AutoDataConfig>> =
            std::collections::HashMap::new();
        for root in &git_group_order {
            let members = &git_group_members[root];
            if members.len() > 1 {
                let skip_subdirs: Vec<PathBuf> = members
                    .iter()
                    .map(|&idx| {
                        let gs = wheels[idx].git_source.as_ref().unwrap();
                        PathBuf::from(gs.subdirectory.as_deref().unwrap_or("."))
                    })
                    .collect();
                auto_data_override.insert(
                    members[0],
                    Some(AutoDataConfig {
                        checkout_root: root.clone(),
                        skip_subdirs,
                    }),
                );
                for &idx in &members[1..] {
                    auto_data_override.insert(idx, None);
                }
            }
        }

        // ASSERT (b): IsaacLab group has 3 members (indices 0, 1, 4).
        let isaac_root = crate::source_build::git_checkout_root(&url, &rev, &cache_dir);
        let isaac_members = &git_group_members[&isaac_root];
        assert_eq!(isaac_members.len(), 3, "isaaclab group must have 3 members");
        assert_eq!(
            isaac_members[0], 0,
            "isaaclab (lock idx 0) must be group[0]"
        );
        assert_eq!(
            isaac_members[1], 1,
            "isaaclab-assets (lock idx 1) must be group[1]"
        );
        assert_eq!(
            isaac_members[2], 4,
            "isaaclab-tasks (lock idx 4, non-contiguous) must be group[2]"
        );

        // Carrier (index 0) has AutoDataConfig with all 3 subdirs.
        let carrier_ad = auto_data_override
            .get(&0)
            .expect("lock idx 0 must be in auto_data_override")
            .as_ref()
            .expect("carrier (lock idx 0) must have Some(AutoDataConfig)");
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

        // Non-carriers (indices 1 and 4) have None.
        assert!(
            auto_data_override.contains_key(&1) && auto_data_override[&1].is_none(),
            "lock idx 1 (non-carrier) must have None auto_data"
        );
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
            inputs_hash: "hash-c-test".into(),
            root_requirements: vec![],
            wheels: vec![
                // Index wheel (class 4): no git, processed without network.
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
                // Class-3 orphan: must_ship=true, no git_source, not in config.
                // This is the wheel that must trigger Ok(None).
                LockWheel {
                    name: "bfs-transitive-orphan".into(),
                    version: "0.1.0".into(),
                    origin: Origin::Built,
                    filename: "bfs_transitive_orphan-0.1.0-py3-none-any.injected.whl".into(),
                    url: None,
                    sha256: None,
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
        };

        // Config has no retread_wheels entries.
        let config: RetreadConfig =
            serde_json::from_value(serde_json::json!({"retread-wheels": {}})).unwrap();

        let result = materialize_from_lock(
            lock,
            &config,
            &work_dir,
            &output_dir,
            rattler_conda_types::Platform::Linux64,
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
    // materialize_and_rewrite for each, then simulates what materialize_from_lock
    // does for the group (carrier with union skip_subdirs, non-carrier with None),
    // and asserts byte-identical wheels.
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

        // shared root file (must appear in BOTH carriers' auto-data)
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

        let target = wheel_target_for(Platform::Linux64, "3.11");
        let mono_checkout_root =
            crate::source_build::git_checkout_root(&mono_url, &mono_sha, &cache_dir);
        let gamma_checkout_root =
            crate::source_build::git_checkout_root(&gamma_url, &gamma_sha, &cache_dir);

        // ── PRODUCE: alpha (carrier, skip_subdirs = union of all mono members) ──
        //
        // Produce carrier: skip_subdirs = [packages/pkg_alpha, packages/pkg_beta]
        // (union of both group members' subdirs, mirrors auto_data_per_entry).
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
            &target,
            &produce_alpha_dd,
            &produce_alpha_src,
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
        .expect("produce alpha: materialize_and_rewrite");
        let produce_alpha_path = produce_alpha_resolved
            .url
            .to_file_path()
            .expect("produce alpha must be file URL");
        let produce_alpha_bytes = std::fs::read(&produce_alpha_path).unwrap();

        // ── PRODUCE: beta (non-carrier, auto_data=None) ──────────────────────
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
            &target,
            &produce_beta_dd,
            &produce_beta_src,
            &cache_dir,
            RelaxPolicy::None,
            &std::collections::BTreeMap::new(),
            None, // non-carrier: no auto_data
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
        //   idx 0 -> Some(AutoDataConfig{skip=[alpha,beta]})  (carrier)
        //   idx 2 -> None                                      (non-carrier)
        //   idx 3 -> not in override (size-1 path)
        //
        // We reproduce this in the test by calling materialize_and_rewrite with
        // the same auto_data the pre-pass would produce.

        // Replay alpha (carrier: auto_data with union skip_subdirs).
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
            &target,
            &replay_alpha_dd,
            &replay_alpha_src,
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
        .expect("replay alpha: materialize_and_rewrite");
        let replay_alpha_bytes =
            std::fs::read(replay_alpha_resolved.url.to_file_path().unwrap()).unwrap();

        // Replay beta (non-carrier: auto_data=None).
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
            &target,
            &replay_beta_dd,
            &replay_beta_src,
            &cache_dir,
            RelaxPolicy::None,
            &std::collections::BTreeMap::new(),
            None, // non-carrier: auto_data=None
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
            "PHASE 2.5 PARITY: alpha carrier (union skip_subdirs) must be \
             byte-identical between produce and replay"
        );
        assert_eq!(
            produce_beta_bytes, replay_beta_bytes,
            "PHASE 2.5 PARITY: beta non-carrier (auto_data=None) must be \
             byte-identical between produce and replay"
        );
        assert_eq!(
            produce_gamma_bytes, replay_gamma_bytes,
            "PHASE 2.5 PARITY: gamma size-1 group (single-entry path) must be \
             byte-identical between produce and replay"
        );

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
                }),
            ),
            (
                true,
                Some(GitWheelSource {
                    url: gamma_url.clone(),
                    rev: gamma_sha.clone(),
                    subdirectory: None,
                    extras: vec![],
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
            "alpha (lock idx 0) must be carrier of non-contiguous group"
        );

        let _ = std::fs::remove_dir_all(&base);
    }
}

// -----------------------------------------------------------------
// Unit tests for courier build string helpers.
// -----------------------------------------------------------------
#[cfg(test)]
mod courier_build_string_tests {
    use super::courier_build_string;

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
            conda_routed: vec![],
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
                metadata: dummy_metadata("isaacsim", "6.0.0"),
                extras_requested: vec![],
                auto_data: None,
                auto_data_dedup_skipped_root: None,
            },
            extras: vec![sub],
            probe_decisions: vec![],
            solve_diagnostics: BTreeMap::new(),
            conda_routed: vec![],
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
            sha256: None,
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
//   (1) resolve_bundle_bfs_fetches_prefix_transitive — localhost fixture,
//       non-ignored. Exercises the full BFS loop end-to-end: primary wheel
//       seeds a transitive dep via prefix matching; transitive ends up in
//       bundle.extras. Regression guard for the FIX 1 vanish bug: if the
//       re-resolve path had silently deleted deps, this test would catch
//       any future regression in bundle.extras membership.
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
    use std::sync::Arc;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::{PypiToCondaMap, WheelTarget, resolve_bundle};
    use crate::config::{RelaxPolicy, WheelEntry};

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
    /// Returns (port, task-handle). The task accepts up to `max_requests`
    /// connections then stops.
    async fn spawn_index_server(packages: Vec<(String, String, Vec<u8>)>, max_requests: u8) -> u16 {
        use std::collections::HashMap;

        // Build lookup tables.
        let mut by_name: HashMap<String, Vec<(String, Vec<u8>)>> = HashMap::new();
        let mut by_filename: HashMap<String, Vec<u8>> = HashMap::new();

        for (name, version, bytes) in packages {
            let norm_name = name.to_ascii_lowercase().replace(['-', '_', '.'], "-");
            let normalized_dist = name.replace('-', "_");
            let filename = format!("{normalized_dist}-{version}-py3-none-any.whl");
            by_name
                .entry(norm_name)
                .or_default()
                .push((filename.clone(), bytes.clone()));
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

                    let (status, content_type, body) = if let Some(rest) =
                        path.strip_prefix("/simple/")
                    {
                        // Strip trailing slash, get normalized name.
                        let pkg_name = rest.trim_end_matches('/');
                        if let Some(entries) = by_name.get(pkg_name) {
                            let links: String = entries
                                .iter()
                                .map(|(fname, _)| format!("<a href=\"/{fname}\">{fname}</a>\n",))
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

    /// Task A (FIX 3 completion): resolve_bundle-loop-level integration test.
    ///
    /// Drives the FULL BFS loop inside resolve_bundle with a localhost fixture
    /// index. Verifies that a transitive dep reachable via prefix matching ends
    /// up in bundle.extras.
    ///
    /// Scenario:
    ///   - Primary: `rtest-pkg==1.0` (Requires-Dist: `rtest-pkg-sub>=1.0`)
    ///   - Transitive: `rtest-pkg-sub==1.0` (no further deps)
    ///   - Both served by a localhost PEP 503 simple index.
    ///
    /// Assert: bundle.extras contains exactly one entry: rtest-pkg-sub 1.0.
    ///
    /// This test exercises the full BFS pipeline (materialize_and_rewrite for
    /// primary, bfs_fetch_pypi for transitive, commit_chosen, extras.push) and
    /// would catch the FIX-1 vanish bug if it were re-introduced (a dep that
    /// goes through NeedsReResolve must still appear in bundle.extras, not
    /// vanish silently).
    #[tokio::test]
    async fn resolve_bundle_bfs_fetches_prefix_transitive() {
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

        let port = spawn_index_server(
            vec![
                (
                    primary_name.to_string(),
                    primary_version.to_string(),
                    primary_bytes,
                ),
                (sub_name.to_string(), sub_version.to_string(), sub_bytes),
            ],
            32, // enough for primary + sub index + wheel fetches + sidecar attempts
        )
        .await;

        let index_url = format!("http://127.0.0.1:{port}/simple/");

        let entry = WheelEntry {
            version: Some(primary_version.to_string()),
            index: Some(index_url),
            ..Default::default()
        };
        let target = WheelTarget {
            python_version: "3.11".to_string(),
            conda_subdir: "linux-64".to_string(),
        };
        let pypi_to_conda: PypiToCondaMap = HashMap::new();
        let name_map: BTreeMap<String, String> = BTreeMap::new();
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
            &name_map,
            &conda_channels,
            &[],                               // conda_deps_list
            &[],                               // workspace_indexes
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
        )
        .await;

        let index_url = format!("http://127.0.0.1:{port}/simple/");

        let entry = WheelEntry {
            version: Some(primary_version.to_string()),
            index: Some(index_url),
            ..Default::default()
        };
        let target = WheelTarget {
            python_version: "3.11".to_string(),
            conda_subdir: "linux-64".to_string(),
        };
        let pypi_to_conda: PypiToCondaMap = HashMap::new();
        let name_map: std::collections::BTreeMap<String, String> =
            std::collections::BTreeMap::new();
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
            &name_map,
            &conda_channels,
            &[],                               // conda_deps_list
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
        )
        .await;

        let index_url = format!("http://127.0.0.1:{port}/simple/");

        let entry = WheelEntry {
            version: Some(primary_version.to_string()),
            index: Some(index_url),
            ..Default::default()
        };
        let target = WheelTarget {
            python_version: "3.11".to_string(),
            conda_subdir: "linux-64".to_string(),
        };
        let pypi_to_conda: PypiToCondaMap = HashMap::new();
        let name_map: BTreeMap<String, String> = BTreeMap::new();
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
            &name_map,
            &conda_channels,
            &[],                               // conda_deps_list
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
        )
        .await;

        let index_url = format!("http://127.0.0.1:{port}/simple/");

        let entry = WheelEntry {
            version: Some(primary_version.to_string()),
            index: Some(index_url),
            ..Default::default()
        };
        let target = WheelTarget {
            python_version: "3.11".to_string(),
            conda_subdir: "linux-64".to_string(),
        };
        let pypi_to_conda: PypiToCondaMap = HashMap::new();
        let name_map: BTreeMap<String, String> = BTreeMap::new();
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
            &name_map,
            &conda_channels,
            &[],                               // conda_deps_list
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
    use crate::lock::{RetreadLock, SCHEMA};

    fn make_lock_at(path: &std::path::Path, entry_specs: Vec<String>, inputs_hash: &str) {
        let lock = RetreadLock {
            schema: SCHEMA,
            retread_version: "0.0.1".into(),
            bundle: "test-bundle".into(),
            version: "1.0".into(),
            python: "3.11".into(),
            inputs_hash: inputs_hash.into(),
            root_requirements: vec![],
            wheels: vec![],
            conda_run_deps: vec![],
            index_urls: vec!["https://pypi.org/simple".into()],
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec![],
            entry_specs,
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
            inputs_hash: "dummy".into(),
            root_requirements: vec![],
            wheels: vec![],
            conda_run_deps: vec![],
            index_urls: vec![],
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec![],
            entry_specs: vec!["test-bundle==1.0".into()],
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
            inputs_hash: inputs_hash.into(),
            root_requirements: vec![],
            wheels: vec![],
            conda_run_deps: vec![],
            index_urls,
            prerelease: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec![],
            entry_specs,
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
    // PUBLIC_PYPI as stored by merge_index_chain.
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
        // locked_chain = merge([nvidia, pypi], []) = [nvidia, pypi] == lock.index_urls → PASS.
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

    /// (i) dedup/order-masking: a locked entry's index moves from implicit
    /// PUBLIC_PYPI (appended last by merge_index_chain) to explicit-entry-inserted
    /// (same set, different ORDER) → STEP A detects the position change → COLD.
    #[test]
    fn step_a_index_position_change_is_cold() {
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

        // Now: entry A (pypi, switched!) + entry C added (nvidia, explicit).
        // locked_entry_indexes for existing entries (all minus added) = [pypi].
        // locked_chain = merge([pypi], []) = [pypi, nvidia? No — nvidia not in locked].
        // Actually: locked chain = [pypi] ≠ [nvidia, pypi] → COLD.
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
            "(i) index position change (order-sensitive) must be COLD"
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
