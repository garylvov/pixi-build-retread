//! JSON-RPC method handlers. The four entry points pixi calls.

mod audit_report;
use audit_report::{
    build_bundle_audit, format_packagespec, post_emit_widen_pass, write_probe_trace,
    write_solve_failed_summary,
};

mod auto_bundle;
use auto_bundle::{
    BfsFetched, Pending, PendingSource, auto_bundle_transitives, metadata_preferring_sidecar,
    pick_conda_target, seed_worklist,
};

mod cascade;
use cascade::{
    bundle_group_for, iterative_solve_refinement, merge_looser_override, pre_emit_widen_pass,
    pypi_fallback_indexes,
};

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

/// Outcome classification after all environments have been solved.
///
/// Used to drive BOTH the stderr abstention banner AND the MD-deletion
/// guard in `write_solve_failed_summary` -- a single pure helper so
/// the two call sites cannot drift apart.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum RunTerminal {
    /// Every attempted env produced a real solve (not abstained) and
    /// none produced a workspace-block message.
    VerifiedAllSat,
    /// At least one env produced a real solve and at least one of those
    /// produced a workspace-block message (unsat / B/C class).
    VerifiedUnsat,
    /// Every env that was attempted was skipped (no repodata reached).
    /// The run is an abstention -- we can neither confirm nor deny.
    AllAbstained,
    /// No env was attempted at all (e.g. env list was empty after
    /// filtering). Treated like abstention for the MD guard.
    NothingAttempted,
}

/// Classify the terminal state of a conda/outputs run.
///
/// Arguments:
/// - `envs_attempted`: total env iterations entered (including skipped).
/// - `envs_skipped`: how many of those were abstentions (no repodata).
/// - `has_block_messages`: true when `workspace_block_messages` is non-empty
///   (i.e. at least one real unsat with a workspace-block class was found).
///
/// Returns `(RunTerminal, skipped_count)` where `skipped_count` mirrors
/// `envs_skipped` (returned for convenience so callers don't need to
/// re-compute it separately).
/// P3 (grizzly #4): which envs in a completed level earn the single
/// sibling-seeded re-run. Eligible only when the level has more than
/// one env AND at least one sibling converged (otherwise there are no
/// sibling widenings to seed with). Returns indices of capped envs.
pub(crate) fn capped_envs_eligible_for_rerun<'a>(
    classifications: impl Iterator<Item = Option<&'a str>>,
) -> Vec<usize> {
    let tags: Vec<Option<&str>> = classifications.collect();
    let capped: Vec<usize> = tags
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == Some("A-iteration-cap"))
        .map(|(i, _)| i)
        .collect();
    let any_converged = tags.iter().any(|t| *t != Some("A-iteration-cap"));
    if tags.len() > 1 && any_converged {
        capped
    } else {
        Vec::new()
    }
}

pub(crate) fn classify_run_terminal(
    envs_attempted: usize,
    envs_skipped: usize,
    has_block_messages: bool,
) -> (RunTerminal, usize) {
    let terminal = if envs_attempted == 0 {
        RunTerminal::NothingAttempted
    } else if envs_skipped == envs_attempted {
        RunTerminal::AllAbstained
    } else if has_block_messages {
        RunTerminal::VerifiedUnsat
    } else {
        RunTerminal::VerifiedAllSat
    };
    (terminal, envs_skipped)
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
fn courier_build_string(py_short: &str, inputs_hash: &str, build_number: u64) -> String {
    let prefix = &inputs_hash[..inputs_hash.len().min(10)];
    format!("py{py_short}_h{prefix}_{build_number}")
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

/// One wheel after full resolution: URL is concrete, metadata parsed.
#[derive(Debug, Clone)]
struct ResolvedWheel {
    /// PEP 503 normalized PyPI name of this wheel (e.g. "isaacsim-kernel").
    /// Used to filter vendored deps out of the bundle's run-deps and to
    /// build the recipe.yaml's `source` list comments.
    pypi_name: String,
    url: url::Url,
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
            CONDA_BUILD_V1 => self
                .conda_build_v1(parse_params(params)?)
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

        let mut state = self.state.write().await;
        state.config = Some(config);
        state.cache_dir = params.cache_directory;
        // source_directory falls back to the manifest's containing dir.
        state.source_dir = params
            .source_directory
            .or_else(|| params.manifest_path.parent().map(PathBuf::from));
        state.workspace_dir = params.workspace_directory;
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
        tracing::info!(
            retread_version = env!("CARGO_PKG_VERSION"),
            "retread: computing conda outputs (resolving wheels + probing channels; large wheels may download here)",
        );
        let phase_start = std::time::Instant::now();
        let (config, download_dir, source_dir, cache_dir, workspace_dir) =
            self.snapshot(&params.work_directory).await?;
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
        // v0.35.0+: track whether ANY (bundle, env) solve succeeded
        // and accumulate workspace-block messages from envs that
        // ended in Class B/C. If nothing solved AND retread has
        // actionable workspace suggestions, fail conda/outputs with
        // a clear error so pixi shows the diagnostic INSTEAD of its
        // misleading leaf.
        // v0.36.0: tracked but no longer used in the fail gate (see
        // comments at the gate ~line 700). Kept for diagnostics
        // (tracing emission after the per-env loop).
        let mut any_solve_passed: bool = false;
        let mut all_solve_attempted: bool = false;
        let mut workspace_block_messages: Vec<String> = Vec::new();
        // v0.36.1+: per-env solve-outcome counts so the failure
        // message can name which envs failed instead of falsely
        // claiming "every env failed" when in fact only some did.
        let mut envs_attempted: usize = 0;
        // P1: how many of those attempts were abstentions (outcome.skipped).
        // Accumulates OUTSIDE the parallel solve tasks (invariant #9: per-env
        // state isolates inside tasks; aggregate counters live in the
        // coordinator scope). Mirrors how accumulated_diagnostics works.
        let mut envs_skipped: usize = 0;
        let mut envs_failed_with_block: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        // v0.36.2+: track REAL suggestions separately from "see the
        // trace" placeholders. The previous count conflated them,
        // so the message claimed "4 workspace-edit suggestion(s)
        // available" even when 0 actionable suggestions were
        // generated (all chains filtered as installable).
        let mut real_suggestion_count: usize = 0;
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
            // PyPI index fallback chain for the cascade's bundling
            // steps: entry indexes first (pypi.nvidia.com siblings),
            // then workspace [pypi-options] indexes, then public PyPI.
            // Extends the chain auto_bundle_transitives uses; without
            // it the cascade could only bundle from pypi.org and
            // private-index-only deps fell through to a doomed conda
            // emission.
            let workspace_manifest = workspace_dir
                .as_deref()
                .and_then(crate::workspace::WorkspaceManifest::load);
            let cascade_pypi_indexes =
                pypi_fallback_indexes(&base_config, workspace_manifest.as_ref());
            // Cross-output siblings: per-emission so envs only link
            // to their own siblings (not other envs' renames).
            for emission in &emissions {
                let env_bundles: Vec<Bundle> = materialized
                    .iter()
                    .map(|b| apply_emission(b, &base_config, emission).0)
                    .collect();
                let siblings: Vec<(String, String)> = env_bundles
                    .iter()
                    .map(|b| (b.conda_name.clone(), b.primary.metadata.version.clone()))
                    .collect();
                for base_bundle in &materialized {
                    let (mut bundle, mut effective) =
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
                    // v0.30.0+ pre-emit widen pass: now scoped to this env's
                    // channels + transitive overrides. Probes recorded
                    // regardless of policy; mutation gated by
                    // RelaxPolicy::allows_widening_mutation().
                    pre_emit_widen_pass(
                        &mut bundle,
                        &mut effective,
                        &emission.channels,
                        &target,
                        &download_dir,
                        &cascade_pypi_indexes,
                    )
                    .await
                    .map_err(|e| {
                        RpcError::internal(format!(
                            "pre-emit widen for {}: {e:#}",
                            bundle.conda_name,
                        ))
                    })?;
                    let mut output = produce_output(
                        &bundle,
                        &effective,
                        params.host_platform,
                        python_version,
                        &siblings,
                        courier_build_hash.as_deref(),
                    )
                    .map_err(|e| {
                        RpcError::internal(format!("output for {}: {e:#}", bundle.conda_name))
                    })?;
                    post_emit_widen_pass(
                        &mut output,
                        &emission.channels,
                        python_version,
                        effective.relax,
                        &mut bundle.probe_decisions,
                    )
                    .await
                    .map_err(|e| {
                        RpcError::internal(format!(
                            "post-emit widen for {}: {e:#}",
                            bundle.conda_name,
                        ))
                    })?;
                    // v0.33.5+ per-env pre-emission solve check. For
                    // each env that references this discovered
                    // output, run a real conda solve scoped to THAT
                    // env's channels + effective deps + retread's
                    // emission. Each env gets its own diagnostic so
                    // users see exactly which env conflicts on what.
                    //
                    // Why per-env (not union): cross-env union
                    // over-constrains -- e.g. an env that doesn't use
                    // feature.ros2 would still get ros-humble-*
                    // transitive constraints folded in, producing
                    // false-positive `ros2-distro-mutex` collisions
                    // in the diagnostic that don't reflect what pixi
                    // will actually fail on.
                    let manifest_for_solve = workspace_dir
                        .as_deref()
                        .and_then(crate::workspace::WorkspaceManifest::load);
                    // Honor the workspace's channel-priority setting
                    // so retread's solve check matches what pixi will
                    // actually do. Default to Strict (pixi's own
                    // default) when the workspace doesn't specify.
                    //
                    // v0.36.3+: was Disabled in v0.34.5-0.36.2 as a
                    // misread "fix" -- I thought strict was over-
                    // rejecting conda-forge candidates, but actually
                    // strict was doing exactly the right thing
                    // (pytorch from pytorch channel; conda-forge's
                    // pytorch would shadow it under disabled).
                    // Disabled lets conda-forge's CPU torchaudio
                    // compete with pytorch channel's GPU torchaudio
                    // on raw version comparison and pick the wrong
                    // one. Strict respects channel ORDER -- which is
                    // why users list pytorch first in the first place.
                    let workspace_channel_priority = manifest_for_solve
                        .as_ref()
                        .and_then(|m| m.channel_priority.as_deref())
                        .map(|s| match s {
                            "disabled" => rattler_solve::ChannelPriority::Disabled,
                            _ => rattler_solve::ChannelPriority::Strict,
                        })
                        .unwrap_or(rattler_solve::ChannelPriority::Strict);
                    let emitted_run_deps_strs: Vec<String> = output
                        .run_dependencies
                        .depends
                        .iter()
                        .map(|n| {
                            let raw = format_packagespec(&n.spec);
                            if raw.is_empty() {
                                n.name.as_str().to_string()
                            } else {
                                format!("{} {raw}", n.name.as_str())
                            }
                        })
                        .collect();
                    let mut env_names: Vec<String> = if emission.envs.is_empty() {
                        // Fallback default emission: no workspace
                        // envs known -- do ONE solve check using the
                        // emission's channels and just the
                        // run-deps.
                        vec!["__default__".to_string()]
                    } else {
                        emission.envs.clone()
                    };
                    // P1: filter out env names that are not declared in
                    // the workspace manifest. A typo or a stale reference
                    // after an env was removed would reach the solve with
                    // zero effective_dependencies (empty env), which the
                    // solver trivially satisfies -- hiding the misconfiguration
                    // and inflating `all_solve_attempted`. The __default__
                    // sentinel is always kept because it represents the
                    // fallback-no-workspace path, not a manifest env.
                    if let Some(m) = manifest_for_solve.as_ref() {
                        env_names.retain(|n| {
                            if n == "__default__" {
                                return true;
                            }
                            let present = m.has_environment(n);
                            if !present {
                                tracing::warn!(
                                    env = %n,
                                    bundle = %bundle.conda_name,
                                    "dropping env from solve: not declared in workspace manifest (typo or removed env?)",
                                );
                            }
                            present
                        });
                    }
                    // v0.44.0: solve the SMALLEST (most base) env first.
                    // Workspace envs nest -- e.g. isaaclab-gpu ⊆ gsi ⊆
                    // gsi-ros2 (each child adds features on top). Solving
                    // the base env first lets its discovered widenings
                    // SEED the children's refinement (see the
                    // `accumulated_overrides` seed passed into
                    // iterative_solve_refinement below), so a child env
                    // doesn't re-discover the base's widening chain from
                    // scratch. Ordering by ascending effective-dependency
                    // count is a cheap proxy for the subset relation
                    // (a superset env has at least as many deps). Pure
                    // performance: seeding only cuts iteration count, the
                    // final emission still unions every env's widenings
                    // and every solve is re-verified.
                    if let Some(m) = manifest_for_solve.as_ref() {
                        env_names.sort_by_key(|n| {
                            if n == "__default__" {
                                0
                            } else {
                                m.effective_dependencies(n).len()
                            }
                        });
                    }
                    // v0.36.2+: snapshot the post-pre-emit-widen state
                    // so each env's iterative refinement starts from
                    // the SAME baseline. Without this, env A's
                    // widenings leak into env B's solve via the shared
                    // `&mut bundle` / `&mut effective` -- making
                    // sibling envs appear "sat" because A already
                    // widened pytorch/torchvision/etc. The user
                    // caught this: gsi-ros2 ⊇ gsi (it includes every
                    // gsi feature plus ros2); if gsi is unsat,
                    // gsi-ros2 cannot truly be sat, but state leakage
                    // was producing exactly that false positive.
                    //
                    // solve_diagnostics accumulates ACROSS envs
                    // (each env contributes its own entry) -- only
                    // the override + probe_decision state resets.
                    let bundle_snapshot = bundle.clone();
                    let effective_snapshot = effective.clone();
                    let mut accumulated_diagnostics: BTreeMap<
                        String,
                        crate::audit::SolveDiagnostics,
                    > = BTreeMap::new();
                    // v0.36.4+: per-env refinement widens a LOCAL
                    // copy of effective.overrides (via the
                    // snapshot/restore pattern below). Union those
                    // widenings here so we can rebuild `output` ONCE
                    // after the env loop with the loosest spec per
                    // dep — otherwise the iter widenings were inert
                    // (the trace claimed they happened but the
                    // CondaOutput pixi receives reflected only the
                    // pre-refinement state). Starts from the
                    // snapshot's overrides so non-widened entries
                    // (transitive constraints injected by
                    // discover_emissions, user overrides) carry
                    // through unchanged.
                    let mut accumulated_overrides: BTreeMap<String, String> =
                        effective_snapshot.overrides.clone();
                    // v1.4.3: group the (ascending-dep-count-sorted)
                    // envs into levels of equal count and solve each
                    // level's envs CONCURRENTLY. The v0.44 parent-first
                    // seed contract orders base -> superset envs, and a
                    // strict superset always has strictly more
                    // effective deps -- so two envs with EQUAL counts
                    // cannot be a base/superset pair (equal sets gain
                    // nothing from seeding anyway). Each env still runs
                    // on its own clone of the post-pre-emit snapshot
                    // (the v0.36.1/.2 isolation contract), every
                    // iteration still runs a real run_solve_check, and
                    // the merge below runs serially in env order so
                    // accumulated_overrides stays a deterministic
                    // monotonic union.
                    let env_levels: Vec<Vec<String>> = {
                        let count_of = |n: &str| -> usize {
                            match (&manifest_for_solve, n) {
                                (Some(m), n) if n != "__default__" => {
                                    m.effective_dependencies(n).len()
                                }
                                _ => 0,
                            }
                        };
                        let mut levels: Vec<Vec<String>> = Vec::new();
                        let mut last_count: Option<usize> = None;
                        for n in &env_names {
                            let c = count_of(n);
                            if last_count == Some(c) {
                                levels.last_mut().unwrap().push(n.clone());
                            } else {
                                levels.push(vec![n.clone()]);
                                last_count = Some(c);
                            }
                        }
                        levels
                    };
                    for level in &env_levels {
                        // Per-level seed snapshot: every env in this
                        // level sees the widenings of all COMPLETED
                        // levels, never a same-level sibling's.
                        let level_seed = accumulated_overrides.clone();
                        // One env's full seeded refinement, reusable by
                        // both the parallel level fan-out and the P3
                        // capped-env re-run. Inputs are cloned into the
                        // future (snapshot isolation per invariant #9).
                        let solve_env = |env_name: &String, seed: &BTreeMap<String, String>| {
                            let mut bundle = bundle_snapshot.clone();
                            let mut effective = effective_snapshot.clone();
                            let env_name = env_name.clone();
                            let seed = seed.clone();
                            // Reference bindings so `async move` moves
                            // only the &refs, not the outer values.
                            let manifest_for_solve = &manifest_for_solve;
                            let params = &params;
                            let emitted_run_deps_strs = &emitted_run_deps_strs;
                            let siblings = &siblings;
                            async move {
                                let env_name = &env_name;
                                let level_seed = &seed;
                                {
                                    let (
                                        env_channels,
                                        env_workspace_specs,
                                        env_system_requirements,
                                    ) = match (manifest_for_solve, env_name.as_str()) {
                                        (Some(m), n) if n != "__default__" => {
                                            let chans: Vec<ChannelUrl> = m
                                                .effective_channels(n)
                                                .iter()
                                                .filter_map(|s| {
                                                    url::Url::parse(s).ok().map(ChannelUrl::from)
                                                })
                                                .collect();
                                            let chans = if chans.is_empty() {
                                                params.channels.clone()
                                            } else {
                                                chans
                                            };
                                            let mut specs: Vec<String> = Vec::new();
                                            for (dep_name, spec) in m.effective_dependencies(n) {
                                                if spec.is_empty() || spec == "*" {
                                                    specs.push(dep_name);
                                                } else {
                                                    specs.push(format!("{dep_name} {spec}"));
                                                }
                                            }
                                            // v0.37.0 D1b: pull per-env
                                            // system-requirements so solve_check
                                            // sees workspace-declared __cuda /
                                            // __glibc / __osx instead of the
                                            // build host's detected values.
                                            let sysreqs = m.effective_system_requirements(n);
                                            (chans, specs, sysreqs)
                                        }
                                        _ => (
                                            params.channels.clone(),
                                            Vec::new(),
                                            std::collections::BTreeMap::new(),
                                        ),
                                    };
                                    // v0.34.0+: iterative refinement. The solve
                                    // check may say UNSAT because retread emitted
                                    // a too-narrow spec (e.g. `triton >=3.7.0,<3.8`
                                    // when conda's triton 3.7 needs cuda 13 but
                                    // the workspace pins cuda 12.8). The cascade
                                    // didn't widen because the per-dep probe saw
                                    // triton 3.7.0 satisfying in isolation. Now
                                    // we feed the solve-check failure BACK into
                                    // the cascade: parse the blocking deps from
                                    // the unsat explanation, widen any of them
                                    // that are retread-emitted to `*`, re-run
                                    // produce_output + solve check. Iterate up
                                    // to MAX_REFINEMENT iterations (cap so we
                                    // don't loop forever on external conflicts).
                                    let t_env = std::time::Instant::now();
                                    let outcome = iterative_solve_refinement(
                                        emitted_run_deps_strs,
                                        level_seed,
                                        &env_workspace_specs,
                                        &env_channels,
                                        python_version,
                                        &params.host_platform.to_string(),
                                        &mut bundle,
                                        &mut effective,
                                        params.host_platform,
                                        siblings,
                                        env_name,
                                        manifest_for_solve.as_ref(),
                                        workspace_channel_priority,
                                        &env_system_requirements,
                                    )
                                    .await
                                    .map_err(|e| {
                                        RpcError::internal(format!(
                                            "solve refinement for {} env {}: {e:#}",
                                            bundle.conda_name, env_name,
                                        ))
                                    })?;
                                    tracing::info!(
                                        env = %env_name,
                                        elapsed_ms = t_env.elapsed().as_millis() as u64,
                                        satisfiable = outcome.satisfiable,
                                        refinement_rounds = outcome.refinement_steps.len(),
                                        "bench: env refinement finished",
                                    );
                                    Ok((env_name.clone(), outcome, effective.overrides))
                                }
                            }
                        };
                        let level_results: Vec<Result<EnvSolveResult, RpcError>> =
                            futures::future::join_all(
                                level.iter().map(|e| solve_env(e, &level_seed)),
                            )
                            .await;

                        // Unwrap all results first (fail fast on RPC errors).
                        let mut level_outcomes: Vec<EnvSolveResult> = Vec::new();
                        for result in level_results {
                            level_outcomes.push(result?);
                        }

                        // P3 (grizzly #4): same-level siblings can't
                        // cross-seed during the parallel solve. When an
                        // env hit MAX_REFINEMENT while a sibling in the
                        // SAME level converged, give it exactly ONE
                        // re-run with a fresh full budget, seeded with
                        // every sibling's widenings (snapshot/restore +
                        // union accumulator; levels stay parallel). A
                        // second cap is final: warn loudly and classify.
                        let capped_idx = capped_envs_eligible_for_rerun(
                            level_outcomes
                                .iter()
                                .map(|(_, o, _)| o.terminal_classification.as_deref()),
                        );
                        if !capped_idx.is_empty() {
                            let mut rerun_seed = level_seed.clone();
                            for (_, _, ovr) in &level_outcomes {
                                for (dep, spec) in ovr {
                                    merge_looser_override(&mut rerun_seed, dep, spec);
                                }
                            }
                            for i in capped_idx {
                                let env_name = level_outcomes[i].0.clone();
                                tracing::info!(
                                    env = %env_name,
                                    "P3: re-running MAX_REFINEMENT-capped env once with sibling seeds",
                                );
                                let rerun = solve_env(&env_name, &rerun_seed).await?;
                                if rerun.1.terminal_classification.as_deref()
                                    == Some("A-iteration-cap")
                                {
                                    tracing::warn!(
                                        env = %env_name,
                                        "P3: env hit MAX_REFINEMENT again after the sibling-seeded re-run; classifying capped",
                                    );
                                }
                                level_outcomes[i] = rerun;
                            }
                        }

                        // Serial, in env order: union widenings +
                        // bookkeeping. Identical to the old per-env
                        // tail, just hoisted out of the parallel part.
                        for result in level_outcomes {
                            let (env_name, outcome, env_overrides) = result;
                            let env_name = &env_name;
                            // v0.36.4+: refinement may have widened
                            // entries in `effective.overrides` for this
                            // env. Union them into accumulated_overrides
                            // (loosest spec per dep wins) so the
                            // post-loop rebuild ships a CondaOutput
                            // whose run-deps satisfy every env.
                            for (dep, spec) in &env_overrides {
                                merge_looser_override(&mut accumulated_overrides, dep, spec);
                            }
                            if !outcome.satisfiable
                                && !outcome.skipped
                                && !outcome.unsat_explanations.is_empty()
                            {
                                // Print the FULL diagnostic to stderr in
                                // a banner so it survives pixi's log
                                // filtering even at default verbosity.
                                // pixi reports its own (often misleading)
                                // leaf error -- this gives the user the
                                // REAL upstream conflict chain alongside.
                                let banner_lines: Vec<String> = outcome
                                    .unsat_explanations
                                    .iter()
                                    .map(|r| format!("    {}", r.replace('\n', "\n    ")))
                                    .collect();
                                eprintln!(
                                    "\n\
                                 ┌──────────────────────────────────────────────────────────────────────\n\
                                 │ RETREAD SOLVE FAILURE -- output `{}` env `{}`\n\
                                 │ (pre-emission solve check; pixi's own error below may be misleading)\n\
                                 ├──────────────────────────────────────────────────────────────────────\n\
                                 {}\n\
                                 │\n\
                                 │ Full trace: retread-probe-trace-{}.json\n\
                                 └──────────────────────────────────────────────────────────────────────\n",
                                    bundle.conda_name,
                                    env_name,
                                    banner_lines.join("\n"),
                                    bundle.conda_name,
                                );
                                tracing::error!(
                                    bundle = %bundle.conda_name,
                                    env = %env_name,
                                    "pre-emission solve check UNSAT (see banner on stderr)",
                                );
                            }
                            // v1.4.0: a SKIPPED check (no repodata) is an
                            // abstention -- it must not arm the fail gate.
                            if !outcome.skipped {
                                all_solve_attempted = true;
                            } else {
                                // P1: count abstentions so classify_run_terminal
                                // and the abstention banner both see the same
                                // number. Accumulates outside the parallel tasks
                                // (invariant #9: coordinator scope only).
                                envs_skipped += 1;
                            }
                            envs_attempted += 1;
                            if outcome.satisfiable {
                                any_solve_passed = true;
                            }
                            // For workspace-blocked envs, accumulate a
                            // one-liner suggestion for the consolidated
                            // RPC failure message at the end.
                            let class_tag =
                                outcome.terminal_classification.clone().unwrap_or_default();
                            if !outcome.satisfiable
                                && (class_tag.starts_with("B-")
                                    || class_tag.starts_with("C-")
                                    || class_tag.starts_with("A-exhausted")
                                    || class_tag.starts_with("A-no-widening")
                                    || class_tag.starts_with("A-iteration-cap"))
                            {
                                envs_failed_with_block.insert(env_name.clone());
                                for sug in &outcome.workspace_edit_suggestions {
                                    real_suggestion_count += 1;
                                    let feat = sug
                                        .feature
                                        .as_deref()
                                        .map(|f| format!("[feature.{f}.dependencies]"))
                                        .unwrap_or_else(|| "[dependencies]".to_string());
                                    workspace_block_messages.push(format!(
                                        "  env `{}` / {}: change `{}` -> `{}`  ({})",
                                        sug.env,
                                        feat,
                                        sug.current_pin,
                                        sug.suggested_pin,
                                        sug.reason,
                                    ));
                                }
                                // If no suggestions but workspace-blocked,
                                // still include a generic line so the user
                                // knows where to look.
                                if outcome.workspace_edit_suggestions.is_empty() {
                                    workspace_block_messages.push(format!(
                                    "  env `{}` blocked ({}): see retread-probe-trace-{}.json.solve_diagnostics.{}",
                                    env_name, class_tag, bundle.conda_name, env_name,
                                ));
                                }
                            }
                            accumulated_diagnostics.insert(
                                env_name.clone(),
                                crate::audit::SolveDiagnostics {
                                    satisfiable: outcome.satisfiable,
                                    unsat_explanations: outcome.unsat_explanations,
                                    channels_consulted: outcome.channels_consulted,
                                    specs_count: outcome.specs_count,
                                    records_count: outcome.records_count,
                                    refinement_steps: outcome.refinement_steps,
                                    workspace_edit_suggestions: outcome.workspace_edit_suggestions,
                                    terminal_classification: outcome.terminal_classification,
                                    skipped: outcome.skipped,
                                },
                            );
                        }
                    }
                    // Transfer accumulated per-env diagnostics back
                    // onto the bundle so the probe trace + audit MD
                    // files include all envs (the snapshot-restore
                    // pattern above resets bundle.solve_diagnostics
                    // every iteration).
                    bundle.solve_diagnostics = accumulated_diagnostics.clone();
                    // v0.36.4+: rebuild `output` to ship the
                    // union'd refinement widenings. Snapshot-restore
                    // semantics inside the env loop discarded each
                    // env's mutations to bundle/effective, and the
                    // original `output` (created pre-loop) reflected
                    // only the pre-emit widen pass. Restore the
                    // base bundle for emission, apply the
                    // accumulated overrides, re-run produce_output +
                    // post_emit_widen_pass, and replace `output`
                    // with the result. Cheap: produce_output is
                    // pure rendering and post_emit_widen_pass hits
                    // the in-memory repodata cache.
                    let widening_changed = accumulated_overrides != effective_snapshot.overrides;
                    if widening_changed {
                        let mut rebuild_effective = effective_snapshot.clone();
                        rebuild_effective.overrides = accumulated_overrides.clone();
                        let rebuilt = produce_output(
                            &bundle_snapshot,
                            &rebuild_effective,
                            params.host_platform,
                            python_version,
                            &siblings,
                            courier_build_hash.as_deref(),
                        )
                        .map_err(|e| {
                            RpcError::internal(format!(
                                "post-refinement output rebuild for {}: {e:#}",
                                bundle_snapshot.conda_name,
                            ))
                        })?;
                        output = rebuilt;
                        // post_emit_widen_pass records its probes
                        // against bundle.probe_decisions. After the
                        // env loop's snapshot/restore + diagnostic
                        // assignment above, `bundle` still holds the
                        // last env's clone. That's fine for probe
                        // recording -- the post-emit pass mutates
                        // output's run-deps in place per the
                        // ground-truth repodata, and any probe
                        // decisions it adds become part of this
                        // bundle's audit.
                        post_emit_widen_pass(
                            &mut output,
                            &emission.channels,
                            python_version,
                            rebuild_effective.relax,
                            &mut bundle.probe_decisions,
                        )
                        .await
                        .map_err(|e| {
                            RpcError::internal(format!(
                                "post-refinement post-emit widen for {}: {e:#}",
                                bundle.conda_name,
                            ))
                        })?;
                        tracing::info!(
                            bundle = %bundle.conda_name,
                            widened_count = accumulated_overrides.len()
                                - effective_snapshot.overrides.len(),
                            "rebuilt output with refinement-widened overrides",
                        );
                        let rebuilt_run_deps_strs: Vec<String> = output
                            .run_dependencies
                            .depends
                            .iter()
                            .map(|n| {
                                let raw = format_packagespec(&n.spec);
                                if raw.is_empty() {
                                    n.name.as_str().to_string()
                                } else {
                                    format!("{} {raw}", n.name.as_str())
                                }
                            })
                            .collect();
                        for env_name in &env_names {
                            let should_reverify = accumulated_diagnostics
                                .get(env_name)
                                .map(|d| !d.satisfiable)
                                .unwrap_or(true);
                            if !should_reverify {
                                continue;
                            }
                            let (env_channels, env_workspace_specs, env_system_requirements) =
                                match (&manifest_for_solve, env_name.as_str()) {
                                    (Some(m), n) if n != "__default__" => {
                                        let chans: Vec<ChannelUrl> = m
                                            .effective_channels(n)
                                            .iter()
                                            .filter_map(|s| {
                                                url::Url::parse(s).ok().map(ChannelUrl::from)
                                            })
                                            .collect();
                                        let chans = if chans.is_empty() {
                                            params.channels.clone()
                                        } else {
                                            chans
                                        };
                                        let mut specs: Vec<String> = Vec::new();
                                        for (dep_name, spec) in m.effective_dependencies(n) {
                                            if spec.is_empty() || spec == "*" {
                                                specs.push(dep_name);
                                            } else {
                                                specs.push(format!("{dep_name} {spec}"));
                                            }
                                        }
                                        let sysreqs = m.effective_system_requirements(n);
                                        (chans, specs, sysreqs)
                                    }
                                    _ => (
                                        params.channels.clone(),
                                        Vec::new(),
                                        std::collections::BTreeMap::new(),
                                    ),
                                };
                            let mut combined = rebuilt_run_deps_strs.clone();
                            combined.extend(env_workspace_specs.iter().cloned());
                            let outcome = crate::solve_check::run_solve_check(
                                &env_channels,
                                &combined,
                                python_version,
                                &params.host_platform.to_string(),
                                workspace_channel_priority,
                                &env_system_requirements,
                            )
                            .await;
                            if outcome.satisfiable {
                                any_solve_passed = true;
                                envs_failed_with_block.remove(env_name);
                                accumulated_diagnostics.insert(
                                    env_name.clone(),
                                    crate::audit::SolveDiagnostics {
                                        satisfiable: true,
                                        unsat_explanations: Vec::new(),
                                        channels_consulted: outcome.channels_consulted,
                                        specs_count: outcome.specs_count,
                                        records_count: outcome.records_count,
                                        refinement_steps: Vec::new(),
                                        workspace_edit_suggestions: Vec::new(),
                                        terminal_classification: None,
                                        skipped: outcome.skipped,
                                    },
                                );
                                tracing::info!(
                                    bundle = %bundle.conda_name,
                                    env = %env_name,
                                    "rebuilt output re-verified SAT after sibling-env widenings",
                                );
                            }
                        }
                        bundle.solve_diagnostics = accumulated_diagnostics.clone();
                    }
                    if let Err(e) = write_probe_trace(&bundle, &source_dir).await {
                        tracing::warn!(
                            bundle = %bundle.conda_name,
                            error = %format!("{e:#}"),
                            "probe trace write failed (non-fatal)",
                        );
                    }
                    // v0.34.2+: write a sticky human-readable diagnostic
                    // markdown file next to the source package's
                    // pixi.toml whenever ANY env's solve is unsat.
                    // pixi's progress spinner clears stderr lines, so
                    // the banner doesn't survive. A file does. Path:
                    // <source_dir>/RETREAD-SOLVE-FAILED-<bundle>.md
                    if let Err(e) = write_solve_failed_summary(&bundle, &source_dir).await {
                        tracing::debug!(
                            bundle = %bundle.conda_name,
                            error = %format!("{e:#}"),
                            "solve-failed summary write failed (non-fatal)",
                        );
                    }
                    outputs.push(output);
                }
            }
        }
        tracing::debug!(
            outputs = outputs.len(),
            any_solve_passed,
            all_solve_attempted,
            envs_attempted,
            envs_skipped,
            workspace_block_messages = workspace_block_messages.len(),
            "per-env emission loop complete",
        );
        // P1: classify the run terminal state and print a banner when any
        // env abstained. Abstentions are NOT errors (offline-best-effort
        // contract from invariant #9 / v1.4.0 skipped semantics), but they
        // ARE visible -- silently shipping unverified outputs is worse than
        // a loud notice. The banner goes to stderr AND the tty status line.
        {
            let (terminal, skipped_count) = classify_run_terminal(
                envs_attempted,
                envs_skipped,
                !workspace_block_messages.is_empty(),
            );
            if skipped_count > 0 {
                let banner = format!(
                    "retread: solve check ABSTAINED for {skipped_count} of {envs_attempted} \
                     env(s) (no repodata reachable); those envs ship UNVERIFIED",
                );
                eprintln!("\nretread WARNING: {banner}");
                crate::status::tty(&format!("WARNING: {banner}"));
                tracing::warn!(
                    skipped = skipped_count,
                    attempted = envs_attempted,
                    terminal = ?terminal,
                    "solve check abstained for some envs -- outputs ship unverified",
                );
            }
        }
        // v0.36.0+: fail conda/outputs if ANY env produced an
        // actionable workspace conflict, even if other envs passed.
        //
        // Why this changed in v0.36.0: in v0.35.x the gate was
        // `!any_solve_passed`. That hid a worst-case bug -- the
        // ABI-anchor widening regression silently corrupted outputs
        // (`python *`) so 3 of 4 envs "passed" against the
        // pre-emission solve check; retread shipped those corrupt
        // outputs to pixi; pixi's downstream solve then exploded on
        // a misleading leaf (`gymnasium ... python_abi 3.11`). With
        // v0.36.0's ABI-anchor invariant + per-chain verdicts, the
        // cascade no longer corrupts outputs in this way -- so the
        // gate can safely be tightened to "any env actionable" and
        // the user gets the structured RPC error + the MD-file path
        // even when sibling envs happen to be solvable.
        if all_solve_attempted && !workspace_block_messages.is_empty() {
            // Build the file-path hint pointing at the source dir.
            // The MD file is named after the bundle's conda name.
            let md_paths: Vec<String> =
                bundle_md_paths(&source_dir, &outputs).into_iter().collect();
            let md_hint = if md_paths.is_empty() {
                format!("RETREAD-SOLVE-FAILED-*.md under {}", source_dir.display())
            } else {
                md_paths.join(" ")
            };
            // ONE short headline + accurate scope + path. Detail
            // lives in the MD file. Pixi will display this verbatim.
            //
            // Scope text reflects whether the failure is total or
            // partial -- "1 of 4 envs failed" reads very differently
            // from "every env failed" and the user shouldn't have to
            // open the MD file to know which.
            let n_failed = envs_failed_with_block.len();
            let failed_env_list: Vec<String> = envs_failed_with_block.iter().cloned().collect();
            let scope = if envs_attempted > 0 && n_failed == envs_attempted {
                format!("every env ({n_failed}/{envs_attempted})")
            } else if envs_attempted > 0 {
                format!(
                    "{n_failed} of {envs_attempted} envs: [{}]",
                    failed_env_list.join(", "),
                )
            } else {
                format!("{n_failed} env(s)")
            };
            // v0.36.2+: distinguish real suggestions from
            // "see-the-trace" fallback. Previous message
            // conflated them ("4 workspace-edit suggestion(s)
            // available" even when 0 actionable suggestions were
            // generated) which was confusing when the MD's
            // suggestions section was empty.
            let action_line = if real_suggestion_count > 0 {
                format!(
                    "{real_suggestion_count} actionable workspace-edit suggestion(s) at the top of the MD",
                )
            } else {
                "no auto-suggestion (cascade exhausted; conflict is upstream-wheel-vs-workspace-pin and \
                 requires manual judgment -- see the per-env refinement steps in the MD for what was tried)"
                    .to_string()
            };
            let msg = format!(
                "retread: pre-emission solve check failed for {scope} against workspace pins. \
                 {action_line}. Open: {md_hint}",
            );
            return Err(RpcError::invalid_params(msg));
        }
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
        let (config, download_dir, source_dir, cache_dir, workspace_dir) =
            self.snapshot(&params.work_directory).await?;
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
                    let run_deps: Vec<String> = params
                        .run_dependencies
                        .as_ref()
                        .map(|deps| deps.iter().map(|d| d.spec.to_string()).collect())
                        .ok_or_else(|| {
                            RpcError::internal(
                                "courier replay: run_override is None; cannot replay without \
                                 pixi-forwarded run_dependencies"
                                    .to_string(),
                            )
                        })?;
                    match materialize_from_lock(
                        lock,
                        &config,
                        &params.work_directory,
                        &output_dir,
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

        build_one(
            &bundle,
            &effective,
            &config,
            &params.work_directory,
            &output_dir,
            params.output.subdir,
            &python_version,
            &source_dir,
            workspace_dir.as_deref(),
            params.output.build.as_deref(),
            run_override.as_deref(),
        )
        .await
        .map_err(|e| RpcError::internal(format!("build {}: {e:#}", bundle.conda_name)))
    }

    async fn snapshot(
        &self,
        work_dir: &Path,
    ) -> Result<(RetreadConfig, PathBuf, PathBuf, PathBuf, Option<PathBuf>), RpcError> {
        let state = self.state.read().await;
        let config = state
            .config
            .clone()
            .ok_or_else(|| RpcError::internal("initialize was not called"))?;
        let cache_dir = state
            .cache_dir
            .clone()
            .unwrap_or_else(|| work_dir.join("cache"));
        let source_dir = state
            .source_dir
            .clone()
            .unwrap_or_else(|| work_dir.to_path_buf());
        // Materialized wheels (downloads, source-builds, and relaxed copies)
        // live inside the pack folder so they're visible alongside the
        // pack's pixi.toml instead of buried in pixi's opaque cache.
        // cache_dir remains the scratch root for git clones.
        let download_dir = source_dir.join("wheels");
        let workspace_dir = state.workspace_dir.clone();
        Ok((config, download_dir, source_dir, cache_dir, workspace_dir))
    }
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

        let mut sub_bundles: Vec<Bundle> = Vec::with_capacity(group_entries.len());
        for ((entry_name, entry), auto_data) in group_entries.iter().zip(auto_data_per_entry) {
            let sub = resolve_bundle(
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
            })?;
            sub_bundles.push(sub);
        }
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

/// One emission targeting a specific discovered output name. The
/// `output_name` is what pixi expects (e.g. "isaac-pack-physx"); the
/// bundle's conda_name gets renamed to it.
#[derive(Debug, Clone)]
struct DiscoveredEmission {
    output_name: String,
    channels: Vec<ChannelUrl>,
    transitive_overrides: BTreeMap<String, String>,
    /// v0.33.5+: env names this discovered output is referenced by.
    /// Used to drive per-env solve checks downstream.
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

/// Build the list of `RETREAD-SOLVE-FAILED-*.md` paths the user
/// should open after a solve failure. One per emitted output that
/// produced an MD file. Returned as absolute paths so the user can
/// click them directly out of the terminal.
fn bundle_md_paths(source_dir: &Path, outputs: &[CondaOutput]) -> Vec<String> {
    let mut paths: HashSet<String> = HashSet::new();
    for o in outputs {
        let name = o.metadata.name.as_normalized();
        let p = source_dir.join(format!("RETREAD-SOLVE-FAILED-{name}.md"));
        paths.insert(p.display().to_string());
    }
    let mut v: Vec<String> = paths.into_iter().collect();
    v.sort();
    v
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
) -> Result<Bundle> {
    let conda_name = canonical_conda_name(entry_name);
    let mut seen: HashSet<String> = HashSet::new();
    let mut work: VecDeque<Pending> = VecDeque::new();
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
    seen.insert(canonical_conda_name(&primary.pypi_name));

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
    seed_worklist(
        &primary_original_rd,
        &entry.extras,
        &entry.index_url(),
        &prefix,
        &seen,
        &mut work,
    )?;

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
    'levels: loop {
        // Drain the current frontier, deduping at drain time. (The old
        // loop deduped at pop time -- equivalent, since seed_worklist
        // also consults `seen` before enqueuing.)
        let mut frontier: Vec<Pending> = Vec::new();
        while let Some(pending) = work.pop_front() {
            let dep_conda_name = canonical_conda_name(&pending.pypi_name);
            if !seen.insert(dep_conda_name) {
                continue;
            }
            frontier.push(pending);
        }
        if frontier.is_empty() {
            break 'levels;
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
        let fetched: Vec<(Pending, Result<Option<BfsFetched>>)> = {
            use futures::stream::{self, StreamExt};
            stream::iter(to_materialize)
                .map(|pending| async move {
                    let result = match &pending.source {
                        PendingSource::Pypi { specifiers, index } => bfs_fetch_pypi(
                            &pending.pypi_name,
                            specifiers,
                            index,
                            target,
                            download_dir,
                            relax,
                        )
                        .await
                        .map(Some),
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
            let (sub_url, sub_metadata, sub_index_for_recurse, sub_seed_rd) =
                match (&pending.source, fetch_result?) {
                    (PendingSource::Pypi { .. }, Some((resolved_url, metadata, index))) => {
                        // Pypi-form sub-wheels are NOT D-rewritten, so
                        // their metadata IS the original Requires-Dist.
                        let seed_rd = metadata.requires_dist.clone();
                        (resolved_url, metadata, index, seed_rd)
                    }
                    (PendingSource::Pypi { .. }, None) => {
                        unreachable!("phase 2 always fetches Pypi-form items")
                    }
                    (PendingSource::Git { url, rev }, _) => {
                        let synth = WheelEntry {
                            git: Some(url.clone()),
                            rev: rev.clone().or_else(|| Some("HEAD".to_string())),
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
                        (sub.url, sub.metadata, entry.index_url(), sub_original_rd)
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
                        (sub.url, sub.metadata, entry.index_url(), sub_original_rd)
                    }
                };

            // Recurse: this sub-wheel's own extras and prefix-matching base
            // deps also get pulled in. URL/git sub-wheels reuse the parent
            // bundle's `prefix` (often empty for source-form parents) so
            // they don't pull random siblings.
            seed_worklist(
                &sub_seed_rd,
                &pending.extras,
                &sub_index_for_recurse,
                &prefix,
                &seen,
                &mut work,
            )?;

            extras.push(ResolvedWheel {
                pypi_name: dep_conda_name,
                url: sub_url,
                metadata: sub_metadata,
                extras_requested: vec![],
                auto_data: None,
                auto_data_dedup_skipped_root: None,
            });
        }
    }

    Ok(Bundle {
        conda_name,
        primary,
        extras,
        probe_decisions,
        solve_diagnostics: BTreeMap::new(),
    })
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

async fn bfs_fetch_pypi(
    pypi_name: &str,
    specifiers: &VersionSpecifiers,
    index: &str,
    target: &WheelTarget,
    download_dir: &Path,
    relax: RelaxPolicy,
) -> Result<(url::Url, WheelMetadata, String)> {
    // v1.5.9 exact-first: `specifiers` are the ORIGINAL (pre-D)
    // upstream pins, so exact family pins (isaacsim-kernel==6.0.0.0)
    // resolve the exact version and the installed family stays
    // patch-consistent. Only when the exact version has VANISHED from
    // the index do we retry with the relaxed range -- loudly, because
    // that is precisely the patch-drift condition that broke Kit
    // extension resolution (6.0.0.0 experience files requiring
    // extensions the 6.0.0.1 sensor wheel renamed).
    let wheel_result = match pypi::resolve(index, pypi_name, specifiers, target).await {
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
    let (resolved_url, metadata) = match wheel_result {
        Ok(resolved) => {
            let metadata = metadata_preferring_sidecar(&resolved, download_dir).await?;
            (resolved.url, metadata)
        }
        Err(wheel_err) => {
            tracing::info!(
                dep = %pypi_name,
                spec = %specifiers,
                index = %index,
                error = %format!("{wheel_err:#}"),
                "BFS PyPI wheel resolve failed; attempting sdist fallback",
            );
            let sdist = pypi::resolve_sdist(index, pypi_name, specifiers)
                .await
                .with_context(|| {
                    format!(
                        "BFS sdist fallback for {} {} on {} (after wheel-resolve failure: {})",
                        pypi_name, specifiers, index, wheel_err,
                    )
                })?;
            // Per-entry build dir under download_dir so repeats hit
            // the wheel cache.
            let sdist_out = download_dir.join(pypi_name);
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
            (built_url, metadata)
        }
    };
    Ok((resolved_url, metadata, index.to_string()))
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
        let wheel = crate::source_build::build_wheel_from_git(
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
        wheel
    } else if let Some(url) = &entry.url {
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
        let wheel = crate::source_build::build_wheel_from_git(
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
            extras_requested: audit_info.extras_requested,
            auto_data: auto_data_report,
            auto_data_dedup_skipped_root: audit_info.dedup_skipped_root,
            metadata,
        },
        original_requires_dist,
    ))
}

/// (env name, refinement outcome, the env's final overrides) from one
/// env's concurrent solve task in the per-level env loop.
type EnvSolveResult = (
    String,
    crate::solve_check::SolveOutcome,
    BTreeMap<String, String>,
);

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
        Some(hash) => courier_build_string(&py_short, hash, build_number),
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
    let env = marker_env_for(&host_platform.to_string(), &python_version)?;
    let mut run_dep_specs: Vec<NamedSpec<PackageSpec>> = vec![spec_from_str(&python_dep)?];
    let mut seen_dep_names: HashSet<String> = HashSet::from(["python".to_string()]);
    for wheel in bundle.all_wheels() {
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

    assemble_conda_output(
        &bundle.conda_name,
        &bundle.primary.metadata.version,
        &python_version,
        config.courier,
        any_platform_specific,
        run_dep_specs,
        seen_dep_names,
        host_platform,
        config.build_number,
        courier_build_hash,
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

/// Re-materialize wheel bytes from the committed lock and run the shared
/// courier pack tail, skipping derivation (BFS / auto_bundle / solve).
///
/// The replay path for `conda_build_v1` calls this when `config.courier` is
/// true and `load_replayable_lock` confirms the lock is valid (schema +
/// inputs_hash match, no poisoning). Skips DERIVATION only — the full
/// materialization pipeline (download / source-build / inject / relax-rewrite)
/// is re-run per wheel class:
///
/// **Class 1 — `must_ship=true`, name in `config.retread_wheels`**: the wheel
/// was built from a git / path / url / named-git source. Re-run
/// `materialize_and_rewrite` on the config entry to repopulate `wheels/<name>/`
/// on disk, then pass the resulting `local_path` to `courier::stage`.
///
/// **Class 2 — `must_ship=false`, `origin=Built` (relax-changed shadow)**: the
/// wheel was an index wheel whose Requires-Dist was rewritten by the relax
/// pipeline. The original URL is recorded in `lock.upstream_url` (schema 6+).
/// Pass `remote_url=upstream_url` to `courier::stage`; it will download the
/// original from the index and re-apply the relax rewrite.  For schema-5 locks
/// where `upstream_url` is absent, fall through to full re-resolve.
///
/// **Class 3 — `must_ship=true`, name NOT in `config.retread_wheels`**: the
/// wheel was a BFS transitive built from a `pkg @ git+<url>@<rev>` line in a
/// Requires-Dist. The lock carries insufficient provenance (no git url+rev) to
/// re-build it. Returns `Ok(None)` so the caller falls through to full
/// `resolve_all`.
///
/// **Class 4 — `origin=Index`**: unchanged index wheel. Pass `remote_url` from
/// `lw.url`. `courier::stage` will record it as `Origin::Index` unchanged.
///
/// # Returns
/// - `Ok(Some(result))` — all wheels re-materialized; pack completed.
/// - `Ok(None)` — lock provenance insufficient for replay (class 3 / schema-5
///   class-2 gap); caller should fall through to full `resolve_all`.
/// - `Err(...)` — replay was attempted but failed (download error,
///   rattler-build failure, etc.). Caller should propagate as a hard error.
///
/// # Correctness
/// `run_deps` MUST come from `params.run_dependencies` (the live pixi-forwarded
/// deps), not from `lock.conda_run_deps`, so the rebuilt package's run-deps
/// are authoritative even on replay.
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
    use crate::lock::Origin;

    let bundle_name = lock.bundle.clone();
    let version = lock.version.clone();
    let python_version = crate::relax::emit_python_version("", &lock.python);
    let download_dir = source_dir.join("wheels");
    let target = wheel_target_for(target_subdir, &python_version);

    // Per-wheel re-materialization: classify each LockWheel and build the
    // EmitWheel with correct local_path / remote_url for courier::stage.
    let mut emit_wheels: Vec<crate::emit_pypi::EmitWheel> = Vec::with_capacity(lock.wheels.len());

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
                    remote_url,
                    upstream_url: None,
                }
            }
            Origin::Built if lw.must_ship => {
                // Class 1 or Class 3: `.injected` source-built wheel.
                if let Some(entry) = config.retread_wheels.get(&lw.name) {
                    // Class 1: top-level [retread-wheels] entry — re-run the
                    // full materialize_and_rewrite pipeline (Phase 1 download/
                    // source-build + Phase 1.5 inject + Phase 1.6 auto-data +
                    // Phase 2 relax-rewrite). This repopulates wheels/<name>/
                    // on disk even if the directory is empty (lukewarm box).
                    tracing::info!(
                        wheel = %lw.name,
                        "courier replay: re-materializing source-built wheel (class 1)"
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
                        &target,
                        &download_dir,
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
                        remote_url: None,
                        upstream_url: None,
                    }
                } else {
                    // Class 3: BFS transitive built from a `pkg @ git+<url>`
                    // Requires-Dist line. The lock carries no git url+rev
                    // (schema gap); cannot re-materialize without re-running
                    // the full BFS. Return Ok(None) so the caller falls through
                    // to full resolve_all.
                    tracing::warn!(
                        wheel = %lw.name,
                        "courier replay: wheel has no [retread-wheels] entry (class 3 / \
                         BFS git transitive — schema gap); falling through to full resolve",
                    );
                    return Ok(None);
                }
            }
            Origin::Built => {
                // Class 2: relax-changed shadow (must_ship=false). The
                // original upstream URL is in lw.upstream_url (schema 6+).
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
                crate::emit_pypi::EmitWheel {
                    pypi_name: lw.name.clone(),
                    version: lw.version.clone(),
                    requires_dist: lw.requires_dist.clone(),
                    local_path: None,
                    wheel_filename: lw.filename.clone(),
                    remote_url: Some(remote_url),
                    upstream_url: None,
                }
            }
        };
        emit_wheels.push(emit);
    }

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
                local_path: (url.scheme() == "file")
                    .then(|| url.to_file_path().ok())
                    .flatten(),
                wheel_filename: url
                    .path_segments()
                    .and_then(|mut s| s.next_back())
                    .unwrap_or_default()
                    .to_string(),
                remote_url: (url.scheme() != "file").then(|| (*url).clone()),
                // Pristine pre-localization index URL: read from w.url (the
                // original unlocalized URL, BEFORE localize_wheel_source may
                // have collapsed it to file://). Carried through to the lock
                // so the replay path can re-fetch Class-2 shadows without a
                // full BFS/solve. None for source-built wheels (file://-only).
                upstream_url: (w.url.scheme() != "file").then(|| w.url.clone()),
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
        siblings,
    )?;
    Ok(Some(output))
}

// -----------------------------------------------------------------
// WS-B: unit tests for replay_from_lock and load_replayable_lock.
// Uses only std (no tempfile crate dependency) -- temp dirs are created
// via std::env::temp_dir() with a unique subdirectory.
// -----------------------------------------------------------------
#[cfg(test)]
mod replay_tests {
    use std::collections::BTreeMap;

    use rattler_conda_types::Platform;

    use super::{load_replayable_lock, replay_from_lock};
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
            }],
            conda_run_deps: vec![CondaDep {
                name: "numpy".into(),
                spec: ">=1.21".into(),
            }],
            index_urls: vec!["https://pypi.org/simple/".into()],
            prerelease: BTreeMap::new(),
            conda_capable: vec![],
        }
    }

    #[test]
    fn matching_hash_returns_some_with_correct_fields() {
        let dir = unique_tmp_dir();
        let lock = make_test_lock("mypack", "1.2.3", "3.11", "abc123", true);
        let json = lock.to_pretty_json().unwrap();
        let lock_path = dir.join(RetreadLock::file_name("mypack"));
        std::fs::write(&lock_path, &json).unwrap();

        let result = replay_from_lock(&lock_path, "abc123", true, Platform::Linux64, 0, &[]);
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

        let result = replay_from_lock(&lock_path, "any-hash", true, Platform::Linux64, 0, &[]);
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

        let result = replay_from_lock(&lock_path, "any-hash", true, Platform::Linux64, 0, &[]);
        assert!(
            result.is_err(),
            "malformed JSON must return Err (caller falls through): {result:?}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn platform_specific_wheel_sets_host_platform_subdir() {
        let dir = unique_tmp_dir();
        let lock = make_test_lock("mypack", "1.0.0", "3.11", "hash1", false /* arch */);
        let json = lock.to_pretty_json().unwrap();
        let lock_path = dir.join(RetreadLock::file_name("mypack"));
        std::fs::write(&lock_path, &json).unwrap();

        let result = replay_from_lock(&lock_path, "hash1", true, Platform::Linux64, 0, &[]);
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
        let dir = unique_tmp_dir();
        let lock = make_test_lock("pack-a", "2.0.0", "3.11", "hash42", true);
        let json = lock.to_pretty_json().unwrap();
        let lock_path = dir.join(RetreadLock::file_name("pack-a"));
        std::fs::write(&lock_path, &json).unwrap();

        let siblings = vec![
            ("pack-a".to_string(), "2.0.0".to_string()),
            ("pack-b".to_string(), "2.0.0".to_string()),
        ];
        let result = replay_from_lock(&lock_path, "hash42", true, Platform::Linux64, 0, &siblings);
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
        let dir = unique_tmp_dir();
        let mut lock = make_test_lock("mypack", "1.0.0", "3.11", "hash9", true);
        lock.conda_run_deps.push(CondaDep {
            name: "uv".into(),
            spec: String::new(),
        });
        let json = lock.to_pretty_json().unwrap();
        let lock_path = dir.join(RetreadLock::file_name("mypack"));
        std::fs::write(&lock_path, &json).unwrap();

        let result = replay_from_lock(&lock_path, "hash9", true, Platform::Linux64, 0, &[]);
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
        let dir = unique_tmp_dir();
        // Use a 64-hex-char inputs_hash (typical sha256 hex output length).
        let hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
        let lock = make_test_lock("mypack", "1.2.3", "3.11", hash, true);
        let json = lock.to_pretty_json().unwrap();
        let lock_path = dir.join(RetreadLock::file_name("mypack"));
        std::fs::write(&lock_path, &json).unwrap();

        let result = replay_from_lock(&lock_path, hash, true, Platform::Linux64, 0, &[]);
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
            }],
            conda_run_deps: vec![],
            index_urls: vec!["https://pypi.org/simple/".into()],
            prerelease: BTreeMap::new(),
            conda_capable: vec![],
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
            }],
            conda_run_deps: vec![],
            index_urls: vec!["https://pypi.org/simple/".into()],
            prerelease: BTreeMap::new(),
            conda_capable: vec![],
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

    // ---- RETREAD_NO_REPLAY env knob tests ----

    #[test]
    fn no_replay_env_knob_returns_none() {
        // SAFETY: single-threaded test; set/unset env var atomically.
        // Tests run in parallel so we use a unique env var name and
        // restore state to avoid interfering with other tests.
        // Note: std::env::set_var is thread-unsafe; this test must
        // run in isolation (use `-- --test-threads=1` if flaky).
        let dir = unique_tmp_dir();
        let lock = make_test_lock("pack", "1.0.0", "3.11", "myhash", true);
        let path = dir.join(RetreadLock::file_name("pack"));
        std::fs::write(&path, lock.to_pretty_json().unwrap()).unwrap();

        // Set the knob and verify replay is suppressed.
        // SAFETY: This test is inherently racy in a multi-threaded test harness.
        // It is marked as a best-effort check; if it fails intermittently,
        // investigate test parallelism.
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

    #[test]
    fn load_replayable_returns_some_with_non_default_relax_when_requires_dist_nonempty() {
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
        let s = courier_build_string("311", hash, 0);
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
        let s1 = courier_build_string("311", "aaaaaa0000111122", 0);
        let s2 = courier_build_string("311", "bbbbbb9999888877", 0);
        assert_ne!(
            s1, s2,
            "different inputs hashes must yield different build strings"
        );
    }

    #[test]
    fn same_hash_different_build_number_gives_different_string() {
        let hash = "abcdef0123456789";
        let s0 = courier_build_string("311", hash, 0);
        let s1 = courier_build_string("311", hash, 1);
        assert_ne!(
            s0, s1,
            "different build numbers must yield different strings"
        );
    }

    #[test]
    fn hash_shorter_than_10_chars_does_not_panic() {
        // When the hash is shorter than 10 chars, min(len, 10) keeps all chars.
        let s = courier_build_string("311", "abc", 0);
        assert_eq!(s, "py311_habc_0");
    }

    #[test]
    fn build_string_format_is_py_prefix_h_hash_number() {
        // Exact format spec: py{py_short}_h{hash[..10]}_{build_number}
        let s = courier_build_string("312", "1234567890abcdef", 2);
        assert_eq!(s, "py312_h1234567890_2");
    }
}

#[cfg(test)]
mod tests;
