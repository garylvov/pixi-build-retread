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
    ChannelUrl, NoArchType, PackageName, Platform, VersionSpec, VersionWithSource,
};
use serde_json::Value;
use tokio::sync::RwLock;
use uv_pep508::uv_pep440::VersionSpecifiers;

use crate::config::{RelaxPolicy, RetreadConfig, WheelEntry};
use crate::pypi::{self, WheelTarget};
use crate::recipe::{BundleSource, build_bundle_recipe, to_yaml};
use crate::relax::{canonical_conda_name, emit_python_version, marker_env_for};
use crate::rpc::{RpcError, ok, parse_params};
use crate::wheel::{WheelMetadata, fetch_wheel};

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
fn conda_outputs_cache_key(params: &CondaOutputsParams) -> String {
    let mut chans: Vec<String> = params
        .channels
        .iter()
        .map(|c| c.url().to_string())
        .collect();
    chans.sort();
    format!(
        "{}|{}|{}|{:?}",
        params.host_platform,
        params.build_platform,
        chans.join(","),
        params.variant_configuration,
    )
}

const NEGOTIATE: &str = "negotiateCapabilities";
const INITIALIZE: &str = "initialize";
const CONDA_OUTPUTS: &str = "conda/outputs";
const CONDA_BUILD_V1: &str = "conda/build_v1";

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
        let cache_key = conda_outputs_cache_key(&params);
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
                        let level_results: Vec<Result<EnvSolveResult, RpcError>> =
                            futures::future::join_all(level.iter().map(|env_name| {
                                let mut bundle = bundle_snapshot.clone();
                                let mut effective = effective_snapshot.clone();
                                // Reference bindings so `async move` moves
                                // only the &refs, not the outer values.
                                let level_seed = &level_seed;
                                let manifest_for_solve = &manifest_for_solve;
                                let params = &params;
                                let emitted_run_deps_strs = &emitted_run_deps_strs;
                                let siblings = &siblings;
                                async move {
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
                            }))
                            .await;

                        // Serial, in env order: union widenings +
                        // bookkeeping. Identical to the old per-env
                        // tail, just hoisted out of the parallel part.
                        for result in level_results {
                            let (env_name, outcome, env_overrides) = result?;
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
            workspace_block_messages = workspace_block_messages.len(),
            "per-env emission loop complete",
        );
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

        let output_dir = params
            .output_directory
            .clone()
            .unwrap_or_else(|| params.work_directory.join("output"));

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
            &params.work_directory,
            &output_dir,
            params.output.subdir,
            &python_version,
            &source_dir,
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
    let primary = materialize_and_rewrite(
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
    seen.insert(primary.pypi_name.clone());

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
    seed_worklist(
        &primary.metadata,
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
            let (sub_url, sub_metadata, sub_index_for_recurse) =
                match (&pending.source, fetch_result?) {
                    (PendingSource::Pypi { .. }, Some((resolved_url, metadata, index))) => {
                        (resolved_url, metadata, index)
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
                        let sub = materialize_and_rewrite(
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
                        (sub.url, sub.metadata, entry.index_url())
                    }
                    (PendingSource::Url { wheel_url }, _) => {
                        let synth = WheelEntry {
                            url: Some(wheel_url.clone()),
                            ..Default::default()
                        };
                        let synth_name = pending.pypi_name.clone();
                        let sub = materialize_and_rewrite(
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
                        (sub.url, sub.metadata, entry.index_url())
                    }
                };

            // Recurse: this sub-wheel's own extras and prefix-matching base
            // deps also get pulled in. URL/git sub-wheels reuse the parent
            // bundle's `prefix` (often empty for source-form parents) so
            // they don't pull random siblings.
            seed_worklist(
                &sub_metadata,
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
async fn bfs_fetch_pypi(
    pypi_name: &str,
    specifiers: &VersionSpecifiers,
    index: &str,
    target: &WheelTarget,
    download_dir: &Path,
) -> Result<(url::Url, WheelMetadata, String)> {
    let wheel_result = pypi::resolve(index, pypi_name, specifiers, target).await;
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
) -> Result<ResolvedWheel> {
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
        fetch_wheel(url, entry.sha256.as_deref(), download_dir)
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
        fetch_wheel(&resolved.url, resolved.sha256.as_deref(), download_dir)
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
    Ok(ResolvedWheel {
        pypi_name,
        url: final_url,
        extras_requested: audit_info.extras_requested,
        auto_data: auto_data_report,
        auto_data_dedup_skipped_root: audit_info.dedup_skipped_root,
        metadata,
    })
}

/// (env name, refinement outcome, the env's final overrides) from one
/// env's concurrent solve task in the per-level env loop.
type EnvSolveResult = (
    String,
    crate::solve_check::SolveOutcome,
    BTreeMap<String, String>,
);

/// `siblings`: every (conda_name, version) produced by the same
/// `conda/outputs` call (including this bundle's own pair, which is
/// skipped). Each non-self entry becomes a run-dep on this output so
/// declaring any one output in the workspace pulls the whole pack via
/// the conda solver.
fn produce_output(
    bundle: &Bundle,
    config: &RetreadConfig,
    host_platform: Platform,
    workspace_python_version: &str,
    siblings: &[(String, String)],
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
    let any_platform_specific = bundle.all_wheels().any(|w| !w.metadata.is_pure_python);
    let subdir = if any_platform_specific {
        host_platform
    } else {
        Platform::NoArch
    };

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

    // Vendored set: every wheel that's part of this bundle is installed
    // alongside its siblings, so any `Requires-Dist` line that names one of
    // them must be dropped from the conda run-deps (otherwise conda would
    // try to install a separate copy from a channel that doesn't have it).
    let vendored: HashSet<String> = bundle.all_wheels().map(|w| w.pypi_name.clone()).collect();

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
    let mut depends_specs: Vec<NamedSpec<PackageSpec>> = vec![spec_from_str(&python_dep)?];
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
            let dep_name = dep.0.split_whitespace().next().unwrap_or("").to_string();
            let parsed_raw: Option<uv_pep508::Requirement> =
                uv_pep508::Requirement::from_str(raw).ok();
            let raw_pypi_name: Option<String> =
                parsed_raw.map(|r| canonical_conda_name(r.name.as_ref()));
            let in_set = |set: &HashSet<String>| {
                set.contains(&dep_name) || raw_pypi_name.as_ref().is_some_and(|p| set.contains(p))
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
            depends_specs.push(spec_from_str(&dep.0)?);
        }
    }

    // Cross-output linking: pin each sibling output produced by the same
    // conda/outputs call as an exact-version run-dep on this output. The
    // workspace then only needs to declare ONE of the names from the
    // pack -- conda solves the rest transitively. Without this, each
    // [retread-wheels] entry needs its own line in the workspace
    // pixi.toml, which gets verbose for stacks like IsaacLab (8 names).
    for (sib_name, sib_version) in siblings {
        if sib_name == &bundle.conda_name {
            continue;
        }
        if !seen_dep_names.insert(sib_name.clone()) {
            continue;
        }
        depends_specs.push(spec_from_str(&format!("{sib_name} =={sib_version}"))?);
    }

    // Surface the final run-dep list at info level so users can spot
    // potentially-problematic deps before conda's solver complains.
    // Anything here that fails downstream is a candidate for
    // retread-drop-deps, retread-overrides, or retread-name-map.
    let emitted: Vec<&str> = depends_specs.iter().map(|s| s.name.as_str()).collect();
    tracing::info!(
        bundle = %bundle.conda_name,
        run_deps = ?emitted,
        "bundle run-deps emitted; if conda can't find one, add it to \
         retread-drop-deps / retread-overrides / retread-name-map"
    );

    let name = PackageName::new_unchecked(bundle.conda_name.clone());
    let version = VersionWithSource::from_str(&bundle.primary.metadata.version)
        .map_err(|e| anyhow!("parsing version `{}`: {e}", bundle.primary.metadata.version))?;
    let noarch = if any_platform_specific {
        NoArchType::none()
    } else {
        NoArchType::python()
    };
    let py_short = python_version.replace('.', "");
    let build = format!("py{py_short}_{}", config.build_number);

    let mut variant = std::collections::BTreeMap::new();
    variant.insert(
        "python".to_string(),
        VariantValue::String(python_version.clone()),
    );

    Ok(CondaOutput {
        metadata: CondaOutputMetadata {
            name,
            version,
            build,
            build_number: config.build_number,
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
            depends: depends_specs,
            constraints: Vec::new(),
        },
        ignore_run_exports: CondaOutputIgnoreRunExports::default(),
        run_exports: CondaOutputRunExports::default(),
        input_globs: None,
    })
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

#[allow(clippy::too_many_arguments)]
async fn build_one(
    bundle: &Bundle,
    config: &RetreadConfig,
    work_dir: &Path,
    output_dir: &Path,
    target_subdir: Platform,
    workspace_python_version: &str,
    source_dir: &Path,
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
    let recipe = build_bundle_recipe(
        &bundle.conda_name,
        &sources,
        config,
        workspace_python_version,
        run_override,
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
    // CRITICAL: rattler-build writes progress to stdout, but retread's
    // stdout is the JSON-RPC channel to pixi. Capture both streams so
    // they don't corrupt the protocol. Surface them via tracing
    // (which writes to OUR stderr) on failure.
    let output = tokio::process::Command::new("rattler-build")
        .arg("build")
        .arg("--recipe")
        .arg(&recipe_path)
        .arg("--output-dir")
        .arg(output_dir)
        .arg("--target-platform")
        .arg(&target_platform)
        .arg("--no-test")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .context("spawning rattler-build (is it on PATH?)")?;
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
    let version = if rest.is_empty() {
        None
    } else {
        Some(
            VersionSpec::from_str(rest, rattler_conda_types::ParseStrictness::Lenient)
                .map_err(|e| anyhow!("parsing version spec `{rest}` for `{name}`: {e}"))?,
        )
    };
    Ok(NamedSpec {
        name: name.to_string(),
        spec: PackageSpec::Binary(BinaryPackageSpec {
            version,
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

#[cfg(test)]
mod tests;
