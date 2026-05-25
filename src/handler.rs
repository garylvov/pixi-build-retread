//! JSON-RPC method handlers. The four entry points pixi calls.

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use pixi_build_types::procedures::{
    conda_build_v1::{CondaBuildV1Params, CondaBuildV1Result},
    conda_outputs::{
        CondaOutput, CondaOutputDependencies, CondaOutputIgnoreRunExports, CondaOutputMetadata,
        CondaOutputRunExports, CondaOutputsParams, CondaOutputsResult,
    },
    initialize::{InitializeParams, InitializeResult},
    negotiate_capabilities::{NegotiateCapabilitiesParams, NegotiateCapabilitiesResult},
};
use pixi_build_types::{BackendCapabilities, BinaryPackageSpec, NamedSpec, PackageSpec, VariantValue};
use rattler_conda_types::{ChannelUrl, NoArchType, PackageName, Platform, VersionSpec, VersionWithSource};
use serde_json::Value;
use tokio::sync::RwLock;
use uv_pep508::uv_pep440::VersionSpecifiers;

use crate::config::{RelaxPolicy, RetreadConfig, WheelEntry};
use crate::pypi::{self, WheelTarget};
use crate::recipe::{build_bundle_recipe, to_yaml, BundleSource};
use crate::relax::{default_marker_env, marker_env_for, python_version_from_wheel_tag};
use crate::rpc::{ok, parse_params, RpcError};
use crate::wheel::{fetch_wheel, read_metadata, WheelMetadata};

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
const PARSELMOUTH_MAPPING_URL: &str =
    "https://raw.githubusercontent.com/prefix-dev/parselmouth/main/files/v0/conda-forge/compressed_mapping.json";

/// Hard-coded PyPI->conda name mappings that PATCH the parselmouth data.
/// These entries are merged into the inverse map on top of parselmouth's
/// (i.e., they ADD candidates rather than replacing). Used for known
/// gaps in parselmouth's coverage -- when the upstream issue is fixed
/// these entries become harmless duplicates.
///
/// Each entry should be accompanied by a link to the relevant
/// parselmouth issue so it can be removed when fixed.
const FALLBACK_PYPI_TO_CONDA: &[(&str, &str)] = &[
    // prefix-dev/parselmouth#10: `py-opencv` (conda-forge) is incorrectly
    // mapped to `None` instead of providing `opencv-python` /
    // `opencv-python-headless`. Both are real conda-forge packages.
    ("opencv-python",          "py-opencv"),
    ("opencv-python-headless", "py-opencv"),
    // Already covered by parselmouth (`pytorch: [torch]`) but here as a
    // safety net in case the fetch fails entirely.
    ("torch",                  "pytorch"),
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
                                    .entry(conda_name_from(&pypi))
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
        let key = conda_name_from(pypi);
        let entry = inverse.entry(key).or_default();
        if !entry.iter().any(|c| c == conda) {
            entry.push((*conda).to_string());
        }
    }

    inverse
}

/// Returns the conda package names to check when looking up `pypi_name`
/// on a conda channel. Always includes the PEP 503 normalized PyPI name,
/// any parselmouth-derived aliases, and any user-supplied alias from
/// `retread-name-map`.
///
/// Currently unused (we no longer probe conda channels for transitives;
/// see auto_bundle_transitives for why). Kept available for future
/// per-channel availability checks.
#[allow(dead_code)]
fn conda_candidates_for(
    pypi_name: &str,
    name_map: &std::collections::BTreeMap<String, String>,
    pypi_to_conda: &PypiToCondaMap,
) -> Vec<String> {
    let normalized = conda_name_from(pypi_name);
    let mut out = vec![normalized.clone()];
    if let Some(conda_names) = pypi_to_conda.get(&normalized) {
        for cn in conda_names {
            if !out.iter().any(|n| n == cn) {
                out.push(cn.clone());
            }
        }
    }
    if let Some(user) = name_map.get(pypi_name) {
        if !out.iter().any(|n| n == user) {
            out.push(user.clone());
        }
    }
    out
}

const BUILT_IN_WIN_ONLY: &[&str] = &[
    "comtypes",         // COM bindings
    "idna-ssl",         // async SSL shim, last release 2017
    "pyreadline",       // readline replacement (deprecated)
    "pyreadline3",      // readline replacement (current)
    "pywin32",          // Win32 API bindings
    "pywin32-ctypes",   // ctypes-only fallback for pywin32
    "pywinpty",         // Windows pseudo-terminal (jupyter, IPython)
    "win32-setctime",   // ctime setter for Windows files
    "winregistry",      // registry helper (stdlib winreg on Windows)
    "winrt-runtime",    // Windows Runtime API
    "winshell",         // shell helpers
    "wmi",              // Windows Management Instrumentation
];

#[derive(Default)]
struct State {
    config: Option<RetreadConfig>,
    cache_dir: Option<PathBuf>,
    /// Directory containing the source package's `pixi.toml`. Used to
    /// resolve relative `path = "..."` entries in [retread-wheels].
    source_dir: Option<PathBuf>,
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
            CONDA_BUILD_V1 => self.conda_build_v1(parse_params(params)?).await.and_then(ok),
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

    async fn initialize(
        &self,
        params: InitializeParams,
    ) -> Result<InitializeResult, RpcError> {
        let config: RetreadConfig = match params.configuration {
            Some(v) => serde_json::from_value(v)
                .map_err(|e| RpcError::invalid_params(format!("[build.config]: {e}")))?,
            None => {
                return Err(RpcError::invalid_params(
                    "pixi-build-retread requires a [build.config] table with at least `wheels = { ... }`",
                ))
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
        state.source_dir = params.source_directory.or_else(|| {
            params.manifest_path.parent().map(PathBuf::from)
        });
        Ok(InitializeResult {})
    }

    async fn conda_outputs(
        &self,
        params: CondaOutputsParams,
    ) -> Result<CondaOutputsResult, RpcError> {
        let (config, download_dir, source_dir, cache_dir) =
            self.snapshot(&params.work_directory).await?;

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
        for python_version in &pythons {
            let target = wheel_target_for(params.host_platform, python_version);
            let (resolved, effective_config) = resolve_all(
                &config,
                &target,
                &download_dir,
                &source_dir,
                &cache_dir,
                &params.channels,
            )
            .await
            .map_err(|e| {
                RpcError::invalid_params(format!(
                    "resolving wheels for python {python_version}: {e:#}"
                ))
            })?;
            // Cross-output linking: collect each bundle's (conda_name,
            // version) so produce_output can emit the other bundles as
            // run-dependencies. Declaring any single output in the
            // workspace then pulls in the whole pack, instead of the
            // workspace needing one line per [retread-wheels] entry.
            let siblings: Vec<(String, String)> = resolved
                .iter()
                .map(|b| (b.conda_name.clone(), b.primary.metadata.version.clone()))
                .collect();
            for bundle in &resolved {
                // v0.14.1+: write a probe-only trace per bundle BEFORE
                // returning outputs to pixi. If the conda solve fails
                // downstream (which is when "no candidates were found
                // for gym" errors fire), conda/build_v1 is never
                // called and the full audit doesn't get written --
                // but this side-file does, so the user can `cat
                // <pack>/retread-probe-trace-<name>.json` to see
                // exactly what each probe decided. Best-effort: a
                // write error logs at warn and never fails the solve.
                if let Err(e) = write_probe_trace(bundle, &source_dir).await {
                    tracing::warn!(
                        bundle = %bundle.conda_name,
                        error = %format!("{e:#}"),
                        "probe trace write failed (non-fatal)",
                    );
                }
                outputs.push(
                    produce_output(
                        bundle,
                        &effective_config,
                        params.host_platform,
                        python_version,
                        &siblings,
                    )
                    .map_err(|e| {
                        RpcError::internal(format!("output for {}: {e:#}", bundle.conda_name))
                    })?,
                );
            }
        }
        Ok(CondaOutputsResult {
            outputs,
            input_globs: Default::default(),
        })
    }

    async fn conda_build_v1(
        &self,
        params: CondaBuildV1Params,
    ) -> Result<CondaBuildV1Result, RpcError> {
        let (config, download_dir, source_dir, cache_dir) =
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
        let config_python = config.python.as_ref().and_then(|s| s.as_versions().into_iter().next());
        let raw = params
            .output
            .variant
            .get("python")
            .map(|v| v.to_string());
        let python_version = match raw.as_deref() {
            Some(v) if v.contains('.') => v.to_string(),
            Some(other) => {
                let fallback = config_python.clone().unwrap_or_else(|| DEFAULT_PYTHON.to_string());
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

        // We re-resolve the full set (the lookup is cheap once cached) and
        // pick the entry matching the requested output.
        let (resolved, effective_config) = resolve_all(
            &config,
            &target,
            &download_dir,
            &source_dir,
            &cache_dir,
            &params.channels,
        )
        .await
        .map_err(|e| RpcError::internal(format!("resolving wheels: {e:#}")))?;

        let requested = params.output.name.as_normalized().to_string();
        let picked = resolved
            .iter()
            .find(|b| b.conda_name == requested)
            .ok_or_else(|| {
                RpcError::invalid_params(format!(
                    "no resolved bundle matches requested output `{requested}`; \
                     known: {:?}",
                    resolved.iter().map(|b| &b.conda_name).collect::<Vec<_>>()
                ))
            })?;

        let output_dir = params
            .output_directory
            .clone()
            .unwrap_or_else(|| params.work_directory.join("output"));

        build_one(
            picked,
            &effective_config,
            &params.work_directory,
            &output_dir,
            params.output.subdir,
            &python_version,
            &source_dir,
            params.output.build.as_deref(),
        )
        .await
        .map_err(|e| RpcError::internal(format!("build {}: {e:#}", picked.conda_name)))
    }

    async fn snapshot(
        &self,
        work_dir: &Path,
    ) -> Result<(RetreadConfig, PathBuf, PathBuf, PathBuf), RpcError> {
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
        Ok((config, download_dir, source_dir, cache_dir))
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
    if let Some(values) = variants.and_then(|v| v.get("python")) {
        if !values.is_empty() {
            return values.iter().map(|v| v.to_string()).collect();
        }
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
) -> Result<(Vec<Bundle>, RetreadConfig)> {
    let mut bundles = Vec::with_capacity(config.retread_wheels.len());

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
        let key = conda_name_from(pypi);
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
    let mut groups: std::collections::BTreeMap<String, Vec<(String, &WheelEntry)>> =
        std::collections::BTreeMap::new();
    for (entry_name, entry) in &effective.retread_wheels {
        let group_name = entry
            .bundle
            .clone()
            .unwrap_or_else(|| entry_name.clone());
        groups
            .entry(group_name)
            .or_default()
            .push((entry_name.clone(), entry));
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
                            Some(PathBuf::from(
                                e.subdirectory.as_deref().unwrap_or("."),
                            ))
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
        for ((entry_name, entry), auto_data) in
            group_entries.iter().zip(auto_data_per_entry.into_iter())
        {
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
        bundle.conda_name = conda_name_from(&group_name);
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
        let auto_index: Option<String> = group_entries
            .iter()
            .find_map(|(_, e)| if e.url.is_none() { Some(e.index_url()) } else { None });
        if effective.auto_bundle {
            if let Some(idx) = auto_index {
                auto_bundle_transitives(
                    &mut bundle,
                    &idx,
                    target,
                    download_dir,
                    &effective,
                    conda_channels,
                    &pypi_to_conda,
                )
                .await?;
            }
        }
        bundles.push(bundle);
    }
    Ok((bundles, effective))
}

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
    conda_channels: &[ChannelUrl],
) -> Result<Bundle> {
    let conda_name = conda_name_from(entry_name);
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
    while let Some(pending) = work.pop_front() {
        let dep_conda_name = conda_name_from(&pending.pypi_name);
        if !seen.insert(dep_conda_name.clone()) {
            continue;
        }
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
        let mut routed_to_conda = false;
        if let PendingSource::Pypi { specifiers, .. } = &pending.source {
            if let Some(candidates) = pypi_to_conda.get(&dep_conda_name) {
                let picked: Option<String> =
                    if candidates.iter().any(|c| c == &dep_conda_name) {
                        Some(dep_conda_name.clone())
                    } else if candidates.len() == 1 {
                        Some(candidates[0].clone())
                    } else {
                        None
                    };
                match picked {
                    None => {
                        tracing::info!(
                            dep = %pending.pypi_name,
                            candidates = ?candidates,
                            "BFS prefer-conda: ambiguous parselmouth mapping with no identity match; not short-circuiting (add retread-name-map to force conda routing)",
                        );
                        probe_decisions.push(crate::audit::ProbeDecision {
                            stage: "bfs".into(),
                            pypi_name: pending.pypi_name.clone(),
                            conda_name: format!("(ambiguous: {candidates:?})"),
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
                        }
                    }
                }
            }
        }
        if routed_to_conda {
            continue;
        }
        let (sub_url, sub_metadata, sub_index_for_recurse) = match &pending.source {
            PendingSource::Pypi { specifiers, index } => {
                // Try the wheel path first. On failure, retry with
                // sdist (PyPI publishers like OpenAI gym stopped
                // shipping wheels; uv builds the sdist into a wheel).
                // Sdist fallback uses the SAME spec, so a narrow
                // version pin still gets honored.
                let wheel_result = pypi::resolve(index, &pending.pypi_name, specifiers, target)
                    .await;
                let (resolved_url, metadata) = match wheel_result {
                    Ok(resolved) => {
                        let metadata = fetch_and_parse(
                            &resolved.url,
                            resolved.sha256.as_deref(),
                            download_dir,
                        )
                        .await?;
                        (resolved.url, metadata)
                    }
                    Err(wheel_err) => {
                        tracing::info!(
                            dep = %pending.pypi_name,
                            spec = %specifiers,
                            index = %index,
                            error = %format!("{wheel_err:#}"),
                            "BFS PyPI wheel resolve failed; attempting sdist fallback",
                        );
                        let sdist =
                            pypi::resolve_sdist(index, &pending.pypi_name, specifiers)
                                .await
                                .with_context(|| {
                                    format!(
                                        "BFS sdist fallback for {} {} on {} (after wheel-resolve failure: {})",
                                        pending.pypi_name, specifiers, index, wheel_err,
                                    )
                                })?;
                        // Per-entry build dir under download_dir so
                        // repeats hit the wheel cache.
                        let sdist_out = download_dir.join(&pending.pypi_name);
                        let built = crate::source_build::build_wheel_from_sdist_url(
                            &sdist.url,
                            &sdist_out,
                            &target.python_version,
                        )
                        .await
                        .with_context(|| {
                            format!(
                                "uv-building wheel from sdist {} for {}",
                                sdist.url, pending.pypi_name,
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
                            dep = %pending.pypi_name,
                            built = %built.display(),
                            "BFS sdist fallback: built wheel from sdist",
                        );
                        (built_url, metadata)
                    }
                };
                (resolved_url, metadata, index.clone())
            }
            PendingSource::Git { url, rev } => {
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
            PendingSource::Url { wheel_url } => {
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

    Ok(Bundle {
        conda_name,
        primary,
        extras,
        probe_decisions,
    })
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
    let pypi_name = conda_name_from(entry_name);

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
            &src.url, &src.rev, subdir, cache_dir, &out, &target.python_version,
        )
        .await
        .with_context(|| {
            format!(
                "phase 1 named-git build for entry `{entry_name}` \
                 (from=`{}`, url=`{}`, rev=`{}`, subdir=`{subdir}`, out_dir={})",
                entry.from.as_deref().unwrap_or(""),
                src.url, src.rev,
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
                    abs.display(), out.display(),
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
            git_url, rev, subdir, cache_dir, &out, &target.python_version,
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
            crate::wheel_inject::inject(&raw_path, &out, &root)
                .with_context(|| {
                    format!(
                        "phase 1.5 source-root inject for entry `{entry_name}` \
                         (source={}, raw_wheel={}, out_wheel={})",
                        root.display(), raw_path.display(), out.display(),
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

/// Returns `true` if `conda_normalized_pypi_name` has an unambiguous conda
/// equivalent in the effective name_map (parselmouth + FALLBACK + user
/// retread-name-map). When true, the prefer-conda policy in
/// [`auto_bundle_transitives`] skips bundling so the dep flows through to
/// emission as a conda run-dep.
fn prefer_conda_match(
    conda_normalized_pypi_name: &str,
    name_map: &std::collections::BTreeMap<String, String>,
) -> bool {
    name_map.contains_key(conda_normalized_pypi_name)
}

/// Build the conda match-spec string the channel probe should look
/// for, given a resolved PyPI version and the active relax policy.
/// Mirrors what `translate(==<version>)` would emit, since that's
/// the spec the conda solver will eventually face. Falls back to
/// `*` (any version) when the version can't be parsed -- a generous
/// default that lets the probe succeed if ANY build of the package
/// exists on the channel.
fn probe_spec_for(version_str: &str, policy: RelaxPolicy) -> String {
    use std::str::FromStr;
    match uv_pep508::uv_pep440::Version::from_str(version_str) {
        Ok(v) => crate::relax::widen_exact(&v, policy).unwrap_or_else(|| "*".to_string()),
        Err(_) => "*".to_string(),
    }
}

/// This is the "pip autoresolve" path: deps that exist on PyPI but might
/// not be on the workspace's conda channels (`aiodns`, `qdldl`, etc.) get
/// pip-installed into the conda package alongside the primary wheel.
///
/// Prefer-conda by default: anything parselmouth or the user's name_map
/// knows a conda equivalent for is skipped here and emitted as a conda
/// run-dep instead.
///
/// Best-effort: a resolve failure logs at debug and leaves the dep to be
/// emitted as a conda run-dep (current fallback behavior).
async fn auto_bundle_transitives(
    bundle: &mut Bundle,
    entry_index: &str,
    target: &WheelTarget,
    download_dir: &Path,
    config: &RetreadConfig,
    conda_channels: &[ChannelUrl],
    pypi_to_conda: &PypiToCondaMap,
) -> Result<()> {
    // Build the skip set: anything already in the bundle, plus the user's
    // `retread-conda-deps` allowlist (deps that should stay as conda
    // run-deps), plus drop-deps, plus packages with explicit overrides
    // (user is forcing conda emission via a spec).
    //
    // There is intentionally NO built-in "conda-preferred" list. ABI
    // collisions (e.g. between a bundled numpy 1.26 and the workspace's
    // conda numpy 2.x) are the user's call -- add the package name to
    // `retread-conda-deps` to keep it on the conda side.
    let mut skip: HashSet<String> = bundle
        .all_wheels()
        .map(|w| w.pypi_name.clone())
        .collect();
    skip.extend(config.conda_deps.iter().map(|n| conda_name_from(n)));
    skip.extend(config.drop_deps.iter().map(|n| conda_name_from(n)));
    skip.extend(config.overrides.keys().map(|n| conda_name_from(n)));

    // Fallback chain: entry's index first (for siblings on private
    // indexes like pypi.nvidia.com), then public PyPI (for the broader
    // ecosystem -- aiodns, qdldl, ...). Public PyPI is hardcoded rather
    // than configurable for now; if a user has air-gap requirements
    // they can disable retread-auto-bundle entirely.
    let mut indexes = vec![entry_index.to_string()];
    let public = "https://pypi.org/simple/".to_string();
    if entry_index != public {
        indexes.push(public);
    }


    // Fixed-point loop: each newly-bundled wheel has its own
    // Requires-Dist that may name more PyPI-only transitives, which
    // themselves should be auto-bundled (e.g. bundling torch pulls in
    // nvidia-cuda-nvrtc-cu12). Re-scan after every bundle until no new
    // wheels are added. Cycle-detected via seen_candidate, which
    // accumulates across iterations.
    let mut seen_candidate: HashSet<String> = skip.clone();
    let mut processed_wheel_count = 0;
    loop {
        // Collect new candidates from wheels we haven't scanned yet.
        let mut candidates: Vec<(String, String)> = Vec::new();
        for wheel in bundle.all_wheels().skip(processed_wheel_count) {
            for raw in &wheel.metadata.requires_dist {
                let Some((name, version)) = pep508_exact_base_dep(raw)? else {
                    continue;
                };
                let conda_name = conda_name_from(&name);
                if !seen_candidate.insert(conda_name) {
                    continue;
                }
                candidates.push((name, version));
            }
        }
        processed_wheel_count = bundle.all_wheels().count();
        if candidates.is_empty() {
            break;
        }

        // Policy: prefer conda. If parselmouth (or our FALLBACK or the
        // user's retread-name-map) knows an unambiguous conda equivalent
        // for the PyPI name, skip bundling -- the dep flows through to
        // emission as a conda run-dep via `translate`, which uses the
        // same effective name_map.
        //
        // Why prefer conda for a conda-based tool: bundling vendors the
        // upstream-pinned version, but the conda copy is what every
        // other native package in the env was built against (BLAS,
        // glibc, CUDA, ABI in general). Double-installing a wheel on
        // top of a conda equivalent at best wastes disk and download
        // time; at worst it shadows the ABI-correct copy with one that
        // wasn't built for this env.
        //
        // Bundling still happens for everything parselmouth doesn't
        // know about (niche PyPI-only helpers). The fallback path below
        // is the original behavior, just with a smaller candidate set.
        //
        // Escape hatches when prefer-conda picks wrong: drop the dep
        // via `retread-drop-deps`, force a specific spec via
        // `retread-overrides`, or remove the parselmouth-discovered
        // entry by overriding it in `retread-name-map` (set to "" to
        // disable). For pin-forwarding conflicts arising on the PyPI
        // side, relax the offending editable's pyproject pin directly
        // (it's your code).

        let mut added_any = false;
        'next_candidate: for (name, version) in candidates {
            let conda_name = conda_name_from(&name);
            if prefer_conda_match(&conda_name, &config.name_map) {
                // Probe the workspace's conda channels for whether the
                // spec retread would emit is actually satisfiable. If
                // ANY channel has a matching candidate, keep on conda.
                // If every channel was reachable and returned versions
                // but NONE matched, fall through to auto-bundle. An
                // indecisive probe (no prefix.dev channels, or all
                // probes errored) keeps the legacy prefer-conda
                // behavior so a prefix.dev outage doesn't silently
                // reshape routing.
                let conda_target_name = config.name_map[&conda_name].clone();
                let probe_spec = probe_spec_for(&version, config.relax);
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
                bundle
                    .probe_decisions
                    .push(crate::audit::ProbeDecision {
                        stage: "auto_bundle".into(),
                        pypi_name: name.clone(),
                        conda_name: conda_target_name.clone(),
                        spec: probe_spec.clone(),
                        target_python: target.python_version.clone(),
                        channels_consulted: probe_result.channels_consulted.clone(),
                        satisfiable: probe_result.satisfiable,
                        matching_candidates: probe_result.matching_candidates,
                        routing_decision: routing_decision.into(),
                    });
                if probe_result.is_definitively_unsatisfied() {
                    tracing::info!(
                        dep = %name,
                        conda = %conda_target_name,
                        spec = %probe_spec,
                        channels = ?probe_result.channels_consulted,
                        "prefer-conda: conda spec is UNSATISFIABLE on workspace channels; falling back to auto-bundle from PyPI",
                    );
                    // intentional fall-through: continue with bundle path
                } else {
                    tracing::info!(
                        dep = %name,
                        conda = %conda_target_name,
                        spec = %probe_spec,
                        matches = probe_result.matching_candidates,
                        decision = %routing_decision,
                        "prefer-conda: skipping auto-bundle; dep will be emitted as a conda run-dep",
                    );
                    continue;
                }
            }
            let specifiers = match VersionSpecifiers::from_str(&format!("=={version}")) {
                Ok(s) => s,
                Err(e) => {
                    tracing::debug!(
                        dep = %name, version = %version,
                        error = %e,
                        "auto-bundle: skipping unparseable version"
                    );
                    continue;
                }
            };
            for index in &indexes {
                match pypi::resolve(index, &name, &specifiers, target).await {
                    Ok(resolved) => {
                        let metadata = match fetch_and_parse(
                            &resolved.url,
                            resolved.sha256.as_deref(),
                            download_dir,
                        )
                        .await
                        {
                            Ok(m) => m,
                            Err(e) => {
                                tracing::debug!(
                                    dep = %name,
                                    error = %format!("{e:#}"),
                                    "auto-bundle fetch failed; leaving as conda dep"
                                );
                                continue 'next_candidate;
                            }
                        };
                        tracing::info!(
                            dep = %name,
                            version = %version,
                            index = %index,
                            "auto-bundled into {}",
                            bundle.conda_name,
                        );
                        bundle.extras.push(ResolvedWheel {
                            pypi_name: conda_name,
                            url: resolved.url,
                            metadata,
                            extras_requested: vec![],
                            auto_data: None,
                            auto_data_dedup_skipped_root: None,
                        });
                        added_any = true;
                        continue 'next_candidate;
                    }
                    Err(e) => {
                        tracing::debug!(
                            dep = %name,
                            version = %version,
                            index = %index,
                            error = %format!("{e:#}"),
                            "auto-bundle resolve failed on this index"
                        );
                    }
                }
            }
            tracing::debug!(
                dep = %name,
                version = %version,
                "auto-bundle exhausted all indexes; leaving as conda dep. \
                 If conda can't satisfy it, add to retread-drop-deps."
            );
        }

        // Loop again only if we added at least one wheel; the new
        // wheels' Requires-Dist may need further auto-bundling.
        if !added_any {
            break;
        }
    }
    Ok(())
}

/// True if `package_name` exists on at least one of the workspace's conda
/// channels. Hits prefix.dev's package-existence API; channels not hosted
/// on prefix.dev are skipped (conservative -- those deps will go through
/// the auto-bundle path).
#[allow(dead_code)]
async fn check_on_any_channel(package_name: &str, channels: &[ChannelUrl]) -> bool {
    for channel in channels {
        let url_str = channel.url().as_str().trim_end_matches('/');
        // Only prefix.dev channels are supported by the existence API.
        // For other hosts we'd need to fetch repodata; skip for now.
        let Some(channel_name) = url_str.strip_prefix("https://prefix.dev/") else {
            continue;
        };
        if channel_name.is_empty() {
            continue;
        }
        let api_url = format!(
            "https://prefix.dev/api/v1/channels/{channel_name}/packages/{package_name}"
        );
        match reqwest::Client::new().head(&api_url).send().await {
            Ok(resp) if resp.status().is_success() => {
                tracing::debug!(
                    package = %package_name,
                    channel = %channel_name,
                    "found on conda channel; skip auto-bundle"
                );
                return true;
            }
            Ok(_) => {}
            Err(e) => {
                tracing::debug!(
                    package = %package_name,
                    channel = %channel_name,
                    error = %e,
                    "conda-availability check failed; assuming not present"
                );
            }
        }
    }
    false
}

/// Returns Some((name, exact_version)) if `raw` is a base dep (no
/// extras marker) with a single `== X.Y.Z` specifier. Returns None for
/// extras-gated deps, ranges, ~=, or URL deps.
fn pep508_exact_base_dep(raw: &str) -> Result<Option<(String, String)>> {
    use std::str::FromStr;
    let req: uv_pep508::Requirement = uv_pep508::Requirement::from_str(raw)
        .map_err(|e| anyhow!("parsing requirement `{raw}`: {e}"))?;
    let env = default_marker_env(DEFAULT_PYTHON)?;
    if !req.marker.evaluate(&env, &[]) {
        return Ok(None);
    }
    let Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) = req.version_or_url.as_ref() else {
        return Ok(None);
    };
    let specs: Vec<_> = specs.iter().collect();
    if specs.len() != 1
        || *specs[0].operator() != uv_pep508::uv_pep440::Operator::Equal
    {
        return Ok(None);
    }
    Ok(Some((req.name.to_string(), specs[0].version().to_string())))
}

/// One unit of pending work in the resolver BFS.
#[derive(Debug, Clone)]
struct Pending {
    pypi_name: String,
    source: PendingSource,
    /// Extras to activate on this wheel. Drives further worklist additions
    /// for `Requires-Dist: name ; extra == "X"` lines.
    extras: Vec<String>,
}

/// v0.12.0+: a dep can be sourced from a PyPI Simple index (the
/// original behavior) or from a direct URL / git URL declared via PEP
/// 508 `<name> @ <url>` form. URL-form deps are common in
/// `[project.optional-dependencies]` and previously made retread bail.
#[derive(Debug, Clone)]
enum PendingSource {
    /// `Requires-Dist: <name> <specifiers>` -- resolve via PyPI Simple.
    Pypi {
        specifiers: VersionSpecifiers,
        index: String,
    },
    /// `Requires-Dist: <name> @ git+<scheme>://<host>/<path>@<rev>` --
    /// clone + `pip wheel --no-deps`.
    Git { url: String, rev: Option<String> },
    /// `Requires-Dist: <name> @ <scheme>://...` (direct wheel/sdist).
    Url { wheel_url: url::Url },
}

fn conda_name_from(pypi_name: &str) -> String {
    pypi_name.to_ascii_lowercase().replace('_', "-")
}

/// Add extras-gated and prefix-matched base deps from `metadata` to `work`.
/// Skips entries already in `seen` so the BFS terminates.
fn seed_worklist(
    metadata: &WheelMetadata,
    extras_requested: &[String],
    index: &str,
    bundle_prefix: &str,
    seen: &HashSet<String>,
    work: &mut VecDeque<Pending>,
) -> Result<()> {
    for raw in &metadata.requires_dist {
        // 1. Extras-gated lines for each requested extra.
        let mut added = false;
        for extra in extras_requested {
            if let Some(dep) = pep508_extra_dep(raw, extra)? {
                let dn = conda_name_from(&dep.name);
                if seen.contains(&dn) {
                    continue;
                }
                work.push_back(Pending {
                    pypi_name: dep.name,
                    source: extra_dep_source_to_pending(dep.source, index),
                    extras: dep.extras,
                });
                added = true;
            }
        }
        if added {
            continue;
        }
        // 2. Base deps (no marker) whose PyPI name matches the bundle prefix.
        if let Some(dep) = pep508_base_dep_in_prefix(raw, bundle_prefix)? {
            let dn = conda_name_from(&dep.name);
            if seen.contains(&dn) {
                continue;
            }
            work.push_back(Pending {
                pypi_name: dep.name,
                source: extra_dep_source_to_pending(dep.source, index),
                extras: dep.extras,
            });
        }
    }
    Ok(())
}

fn extra_dep_source_to_pending(src: ExtraDepSource, default_index: &str) -> PendingSource {
    match src {
        ExtraDepSource::Pypi(specifiers) => PendingSource::Pypi {
            specifiers,
            index: default_index.to_string(),
        },
        ExtraDepSource::Git { url, rev } => PendingSource::Git { url, rev },
        ExtraDepSource::Url(wheel_url) => PendingSource::Url { wheel_url },
    }
}

/// Returns Some(ExtraDep) if `raw` is a base dep (no extras marker, or a
/// marker that's satisfied with empty extras) whose PEP 503 normalized name
/// starts with `prefix`. Used to bundle sibling sub-packages like
/// `isaacsim-kernel` that the metapackage depends on unconditionally.
fn pep508_base_dep_in_prefix(raw: &str, prefix: &str) -> Result<Option<ExtraDep>> {
    use std::str::FromStr;
    let req: uv_pep508::Requirement = uv_pep508::Requirement::from_str(raw)
        .map_err(|e| anyhow!("parsing requirement `{raw}`: {e}"))?;

    // Base dep: marker (if any) satisfied with empty extras.
    let env = default_marker_env(DEFAULT_PYTHON)?;
    if !req.marker.evaluate(&env, &[]) {
        return Ok(None);
    }

    let conda_name = conda_name_from(req.name.as_ref());
    if !conda_name.starts_with(prefix) {
        return Ok(None);
    }

    // Same any-version handling as pep508_extra_dep: a bare-name base
    // dep is legal PEP 508 and resolves to latest at the PyPI index.
    let source = match req.version_or_url.as_ref() {
        Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) => ExtraDepSource::Pypi(specs.clone()),
        Some(uv_pep508::VersionOrUrl::Url(verbatim)) => extra_dep_source_from_url(verbatim.raw())?,
        None => ExtraDepSource::Pypi(uv_pep508::uv_pep440::VersionSpecifiers::empty()),
    };
    Ok(Some(ExtraDep {
        name: req.name.to_string(),
        source,
        extras: req.extras.iter().map(|e| e.to_string()).collect(),
    }))
}

/// Convert a PEP 508 URL Requires-Dist into one of our
/// [`ExtraDepSource`] variants. Splits `git+<scheme>://...@<rev>` into
/// `(base_url, Some(rev))`; plain `https://.../file.whl` becomes a
/// direct-URL fetch.
fn extra_dep_source_from_url(raw_url: &url::Url) -> Result<ExtraDepSource> {
    let s = raw_url.as_str();
    if let Some(stripped) = s.strip_prefix("git+") {
        // PEP 508 doesn't say where the @<rev> lives but pip-compatible
        // syntax is `git+<scheme>://<host>/<path>@<rev>`. Find the
        // rightmost `@` that comes after `://` (skipping any in user-
        // info, though those are rare for public git).
        let scheme_end = stripped.find("://").map(|i| i + 3).unwrap_or(0);
        let (base, rev) = match stripped[scheme_end..].rfind('@') {
            Some(rel) => {
                let abs = scheme_end + rel;
                (
                    stripped[..abs].to_string(),
                    Some(stripped[abs + 1..].to_string()),
                )
            }
            None => (stripped.to_string(), None),
        };
        Ok(ExtraDepSource::Git { url: base, rev })
    } else {
        Ok(ExtraDepSource::Url(raw_url.clone()))
    }
}

async fn fetch_and_parse(
    url: &url::Url,
    sha256_hint: Option<&str>,
    download_dir: &Path,
) -> Result<WheelMetadata> {
    let path = fetch_wheel(url, sha256_hint, download_dir).await?;
    tokio::task::spawn_blocking(move || read_metadata(&path))
        .await
        .context("metadata reader panicked")?
}

/// One extras-derived dependency. v0.12.0+: source can be PyPI Simple
/// OR a direct URL / git URL (`pkg @ git+https://...@<rev>` or `pkg @
/// https://.../file.whl`). PyPI is the common case; URL+git unlock
/// extras like IsaacLab's `rl_games` which pulls `rl-games @ git+...`.
#[derive(Debug, Clone)]
struct ExtraDep {
    name: String,
    source: ExtraDepSource,
    extras: Vec<String>,
}

#[derive(Debug, Clone)]
enum ExtraDepSource {
    Pypi(VersionSpecifiers),
    Git { url: String, rev: Option<String> },
    Url(url::Url),
}

/// Returns `Some(ExtraDep)` if `raw` is a `Requires-Dist` line that is
/// gated on the requested extra. Returns None if the requirement is gated
/// on a different extra (or has no marker, i.e. is a base dep we don't
/// repack at all). Any specifier set is accepted; range resolution
/// happens at the index-fetch layer in pypi::resolve.
fn pep508_extra_dep(raw: &str, extra: &str) -> Result<Option<ExtraDep>> {
    use std::str::FromStr;
    let req: uv_pep508::Requirement = uv_pep508::Requirement::from_str(raw)
        .map_err(|e| anyhow!("parsing extra requirement `{raw}`: {e}"))?;

    let extra_name = uv_normalize::ExtraName::from_owned(extra.to_string())
        .map_err(|e| anyhow!("invalid extra name `{extra}`: {e}"))?;

    // The marker must match when this extra is active AND must not match
    // with no extras active (otherwise it's a base dep, not an extra dep).
    let env = default_marker_env(DEFAULT_PYTHON)?;
    let matches_with_extra = req.marker.evaluate(&env, &[extra_name.clone()]);
    let matches_without = req.marker.evaluate(&env, &[]);
    if !matches_with_extra || matches_without {
        return Ok(None);
    }

    // Bare name with no specifier and no URL is legal PEP 508
    // (`Requires-Dist: tqdm; extra == "sb3"`) -- means "any version".
    // Treat as PyPI with an empty specifier set; pypi::resolve returns
    // the latest matching the target python. Without this, every
    // extras-gated bare name in upstream wheels (rich, tqdm, gym, ...)
    // made retread bail with "no version or URL".
    let source = match req.version_or_url.as_ref() {
        Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) => ExtraDepSource::Pypi(specs.clone()),
        Some(uv_pep508::VersionOrUrl::Url(verbatim)) => extra_dep_source_from_url(verbatim.raw())?,
        None => ExtraDepSource::Pypi(uv_pep508::uv_pep440::VersionSpecifiers::empty()),
    };
    let _ = extra; // extras name is only used for marker evaluation above
    Ok(Some(ExtraDep {
        name: req.name.to_string(),
        source,
        extras: req.extras.iter().map(|e| e.to_string()).collect(),
    }))
}

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
    // Python version: prefer the workspace's variant; fall back to parsing
    // the primary wheel's cp tag.
    let python_version = if bundle.primary.metadata.is_pure_python {
        workspace_python_version.to_string()
    } else {
        python_version_from_wheel_tag(&bundle.primary.metadata.filename)
            .unwrap_or_else(|| workspace_python_version.to_string())
    };
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
        .map(|n| conda_name_from(n))
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
            let dep_name = dep
                .0
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_string();
            if vendored.contains(&dep_name) {
                continue;
            }
            if user_dropped.contains(&dep_name) {
                tracing::debug!(dep = %dep_name, "dropping per retread-drop-deps");
                continue;
            }
            if auto_dropped.contains(&dep_name) {
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
    let version = VersionWithSource::from_str(&bundle.primary.metadata.version).map_err(|e| {
        anyhow!(
            "parsing version `{}`: {e}",
            bundle.primary.metadata.version
        )
    })?;
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

/// Construct a [`crate::audit::BundleAudit`] from the bundle we're
/// about to ship to rattler-build and the generated recipe. Strictly
/// informational; the resulting JSON lands next to recipe.yaml so the
/// user can see exactly which upstream `Requires-Dist:` lines became
/// which conda run-deps, and copy-paste the rendered TOML blocks into
/// their workspace if they want to mirror the bundle exactly.
fn build_bundle_audit(
    bundle: &Bundle,
    recipe: &crate::recipe::Recipe,
) -> crate::audit::BundleAudit {
    let wheels = bundle
        .all_wheels()
        .map(|w| crate::audit::WheelAudit {
            name: w.metadata.name.clone(),
            version: w.metadata.version.clone(),
            requires_dist: w.metadata.requires_dist.clone(),
            extras: w.extras_requested.clone(),
            auto_data: w.auto_data.clone(),
            auto_data_dedup_skipped_root: w.auto_data_dedup_skipped_root.clone(),
        })
        .collect();
    let emitted_run_deps = recipe
        .requirements
        .run
        .iter()
        .map(|spec| {
            // Recipe's run list is `Vec<String>` of "<name> <constraint>"
            // strings; split into (name, full-spec) so the audit consumer
            // can render the conda-deps TOML block from constraint side
            // alone.
            let name = spec.split_whitespace().next().unwrap_or(spec).to_string();
            crate::audit::EmittedDep {
                name,
                spec: spec.clone(),
            }
        })
        .collect();
    crate::audit::BundleAudit::new(
        bundle.conda_name.clone(),
        bundle.primary.metadata.version.clone(),
        wheels,
        emitted_run_deps,
        bundle.probe_decisions.clone(),
    )
}

/// v0.14.1+: dump just the bundle's probe decisions to a side file
/// next to the source-package pixi.toml. Always-on; survives a failed
/// conda solve. The full audit (a superset) still gets written at
/// conda/build_v1 time. Filename matches the audit convention:
/// retread-probe-trace-<conda_name>.json.
async fn write_probe_trace(bundle: &Bundle, source_dir: &Path) -> Result<()> {
    let path = source_dir.join(format!(
        "retread-probe-trace-{}.json",
        bundle.conda_name,
    ));
    #[derive(serde::Serialize)]
    struct Trace<'a> {
        conda_name: &'a str,
        retread_version: &'static str,
        probe_decisions: &'a [crate::audit::ProbeDecision],
    }
    let trace = Trace {
        conda_name: &bundle.conda_name,
        retread_version: env!("CARGO_PKG_VERSION"),
        probe_decisions: &bundle.probe_decisions,
    };
    let bytes = serde_json::to_vec_pretty(&trace)?;
    tokio::fs::write(&path, &bytes)
        .await
        .with_context(|| format!("writing probe trace to {}", path.display()))?;
    tracing::info!(
        bundle = %bundle.conda_name,
        decisions = bundle.probe_decisions.len(),
        path = %path.display(),
        "wrote probe trace",
    );
    Ok(())
}

async fn build_one(
    bundle: &Bundle,
    config: &RetreadConfig,
    work_dir: &Path,
    output_dir: &Path,
    target_subdir: Platform,
    workspace_python_version: &str,
    source_dir: &Path,
    expected_build: Option<&str>,
) -> Result<CondaBuildV1Result> {
    // Lay out one BundleSource per wheel (primary first), in BFS order.
    let sources: Vec<BundleSource> = bundle
        .all_wheels()
        .map(|w| BundleSource {
            pypi_name: &w.pypi_name,
            url: &w.url,
            metadata: &w.metadata,
        })
        .collect();
    let recipe = build_bundle_recipe(
        &bundle.conda_name,
        &sources,
        config,
        workspace_python_version,
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
    let audit_json = serde_json::to_string_pretty(&audit)
        .context("serializing audit record")?;
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
    bail!("no .conda artifact found in {} matching {prefix}*.conda", dir.display())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RelaxPolicy;
    use std::collections::BTreeMap;

    fn cfg() -> RetreadConfig {
        RetreadConfig {
            retread_wheels: BTreeMap::new(),
            relax: RelaxPolicy::Minor,
            overrides: BTreeMap::new(),
            name_map: BTreeMap::new(),
            build_number: 0,
            drop_deps: Vec::new(),
            auto_bundle: false,
            conda_deps: Vec::new(),
            git_sources: std::collections::BTreeMap::new(),
            python: None,
        }
    }

    fn meta(name: &str, version: &str, requires: Vec<&str>, platform_specific: bool) -> WheelMetadata {
        WheelMetadata {
            name: name.into(),
            version: version.into(),
            requires_dist: requires.into_iter().map(String::from).collect(),
            is_pure_python: !platform_specific,
            sha256: format!("sha-{name}"),
            filename: if platform_specific {
                format!("{}-{version}-cp311-none-manylinux_2_35_x86_64.whl", name.replace('-', "_"))
            } else {
                format!("{}-{version}-py3-none-any.whl", name.replace('-', "_"))
            },
        }
    }

    fn rw(pypi: &str, m: WheelMetadata) -> ResolvedWheel {
        ResolvedWheel {
            pypi_name: pypi.to_string(),
            url: format!("https://example.com/{pypi}.whl").parse().unwrap(),
            metadata: m,
            extras_requested: vec![],
            auto_data: None,
            auto_data_dedup_skipped_root: None,
        }
    }

    fn solo_bundle(name: &str, requires: Vec<&str>) -> Bundle {
        Bundle {
            conda_name: name.into(),
            primary: rw(name, meta(name, "1.0.0", requires, true)),
            extras: vec![],
            probe_decisions: vec![],
        }
    }

    #[test]
    fn built_in_win_only_dropped_on_linux() {
        // idna-ssl is in BUILT_IN_WIN_ONLY. Targeting linux-64, it must
        // not appear in run-deps even though it has no explicit
        // `sys_platform == "win32"` marker.
        let bundle = solo_bundle("isaacsim", vec!["idna-ssl==1.1.0", "numpy==1.26.0"]);
        let output = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[]).unwrap();
        let names: Vec<String> = output
            .run_dependencies
            .depends
            .iter()
            .map(|d| d.name.clone())
            .collect();
        assert!(!names.iter().any(|n| n == "idna-ssl"),
            "idna-ssl auto-drop on linux failed; got: {names:?}");
        assert!(names.iter().any(|n| n == "numpy"),
            "numpy must still be emitted; got: {names:?}");
    }

    #[test]
    fn built_in_win_only_kept_on_windows() {
        // Same input, win-64 target. The auto-drop is non-Windows-only,
        // so idna-ssl is expected to remain.
        let bundle = solo_bundle("isaacsim", vec!["idna-ssl==1.1.0"]);
        let output = produce_output(&bundle, &cfg(), Platform::Win64, "3.11", &[]).unwrap();
        let names: Vec<String> = output
            .run_dependencies
            .depends
            .iter()
            .map(|d| d.name.clone())
            .collect();
        assert!(names.iter().any(|n| n == "idna-ssl"),
            "idna-ssl should NOT be auto-dropped on win-64; got: {names:?}");
    }

    #[test]
    fn explicit_override_beats_built_in_win_only() {
        // If a user actually needs idna-ssl on linux, retread-overrides
        // is the documented escape hatch. Setting it to any spec must
        // cancel the built-in auto-drop.
        let mut config = cfg();
        config
            .overrides
            .insert("idna-ssl".to_string(), "*".to_string());
        let bundle = solo_bundle("isaacsim", vec!["idna-ssl==1.1.0"]);
        let output = produce_output(&bundle, &config, Platform::Linux64, "3.11", &[]).unwrap();
        let names: Vec<String> = output
            .run_dependencies
            .depends
            .iter()
            .map(|d| d.name.clone())
            .collect();
        assert!(names.iter().any(|n| n == "idna-ssl"),
            "retread-overrides should cancel the auto-drop; got: {names:?}");
    }

    #[test]
    fn user_drop_deps_silently_drops() {
        // User-specified drop happens at debug level (no warn), unlike
        // the built-in auto-drop which warns. Behavior parity: dep is
        // not emitted.
        let mut config = cfg();
        config.drop_deps.push("requests".to_string());
        let bundle = solo_bundle("foo", vec!["requests==2.32.0", "numpy==1.26.0"]);
        let output = produce_output(&bundle, &config, Platform::Linux64, "3.11", &[]).unwrap();
        let names: Vec<String> = output
            .run_dependencies
            .depends
            .iter()
            .map(|d| d.name.clone())
            .collect();
        assert!(!names.iter().any(|n| n == "requests"),
            "requests should be dropped per retread-drop-deps; got: {names:?}");
        assert!(names.iter().any(|n| n == "numpy"));
    }

    #[test]
    fn vendored_sub_packages_dropped_from_run_deps() {
        // Mirror the isaacsim bundle: primary depends on sub-packages,
        // sub-packages depend on each other, all are vendored together.
        let bundle = Bundle {
            conda_name: "isaacsim".into(),
            primary: rw(
                "isaacsim",
                meta(
                    "isaacsim",
                    "5.1.0.0",
                    vec!["isaacsim-kernel==5.1.0.0 ; extra == \"all\""],
                    true,
                ),
            ),
            extras: vec![
                rw(
                    "isaacsim-kernel",
                    meta(
                        "isaacsim-kernel",
                        "5.1.0.0",
                        vec!["numpy==1.26.0", "Pillow==11.3.0"],
                        true,
                    ),
                ),
                rw(
                    "isaacsim-core",
                    meta(
                        "isaacsim-core",
                        "5.1.0.0",
                        vec![
                            "isaacsim-kernel==5.1.0.0",
                            "numpy==1.26.0",
                            "scipy==1.15.3",
                        ],
                        true,
                    ),
                ),
            ],
            probe_decisions: vec![],
        };

        let output = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[]).unwrap();
        let dep_names: Vec<String> = output
            .run_dependencies
            .depends
            .iter()
            .map(|d| d.name.clone())
            .collect();
        assert!(!dep_names.iter().any(|n| n == "isaacsim-kernel"),
            "isaacsim-kernel is vendored and must NOT appear in run-deps; got: {dep_names:?}");
        assert!(!dep_names.iter().any(|n| n == "isaacsim-core"),
            "isaacsim-core is vendored and must NOT appear in run-deps; got: {dep_names:?}");
        assert!(dep_names.iter().any(|n| n == "numpy"),
            "numpy must appear (deduped from multiple wheels); got: {dep_names:?}");
        assert!(dep_names.iter().any(|n| n == "pillow"),
            "pillow must appear; got: {dep_names:?}");
        assert!(dep_names.iter().any(|n| n == "scipy"),
            "scipy must appear; got: {dep_names:?}");
    }

    #[tokio::test]
    #[ignore = "spawns pip wheel; needs PATH with pip + setuptools (same prerequisites as path_source_does_not_corrupt_stdout)"]
    async fn d_rewrites_metadata_on_the_wheel_the_recipe_will_source() {
        // Regression for the silent-D bug: ResolvedWheel.url used to
        // carry the pre-D URL (upstream for PyPI form, file:// of the
        // unrewritten built wheel for path / git). That meant the conda
        // emission saw the relaxed pins but the wheel rattler-build
        // actually copied into the conda package still had the strict
        // pins -- pixi forwarded conda's choice to uv as a hard pin, uv
        // then read site-packages METADATA and found a conflicting
        // strict pin, and the solve failed with the classic "PyPI
        // packages have been pinned by the conda solve" error.
        // Contract: after a non-None relax policy, ResolvedWheel.url
        // points at a file:// path whose METADATA matches what we
        // emitted into conda run-deps.
        let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/sample_with_buildtime_dep");
        let tmp = std::env::temp_dir().join(format!(
            "retread-d-on-disk-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp).unwrap();

        let entry = crate::config::WheelEntry {
            path: Some(fixture.to_string_lossy().to_string()),
            ..Default::default()
        };
        let target = WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "linux-64".into(),
        };

        let resolved = materialize_and_rewrite(
            &entry,
            "retread-sample",
            &target,
            &tmp,
            &fixture,
            &tmp,
            RelaxPolicy::Minor,
            &std::collections::BTreeMap::new(),
            None,
            EntryAuditInfo::default(),
        )
        .await
        .expect("materialize_and_rewrite");

        assert_eq!(
            resolved.url.scheme(),
            "file",
            "recipe source URL must be file:// of the rewritten wheel; got {}",
            resolved.url,
        );
        let on_disk = resolved.url.to_file_path().expect("file path from URL");
        let on_disk_meta = crate::wheel::read_metadata(&on_disk)
            .expect("read METADATA from wheel-on-disk");

        let starlette_lines: Vec<&String> = on_disk_meta
            .requires_dist
            .iter()
            .filter(|l| l.to_lowercase().contains("starlette"))
            .collect();
        assert!(
            !starlette_lines.is_empty(),
            "fixture must declare starlette; got Requires-Dist: {:?}",
            on_disk_meta.requires_dist,
        );
        assert!(
            starlette_lines.iter().all(|l| !l.contains("==0.49.1")),
            "wheel on disk must have starlette pin relaxed (minor relax => `>=0.49,<1`), \
             but at least one Requires-Dist still reads `==0.49.1`: {:?}",
            on_disk_meta.requires_dist,
        );

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn bundle_field_groups_entries_into_one_output() {
        // Contract: two [retread-wheels] entries with the same `bundle`
        // field collapse to ONE conda output containing both wheels.
        // Verified at the produce_output level by constructing a Bundle
        // whose conda_name is the bundle group's name and whose
        // primary+extras list covers wheels from both source entries.
        //
        // resolve_all's grouping is integration-shaped (it shells out
        // to PyPI/pip); the contract this test pins is the downstream
        // behavior produce_output exposes: the bundle's conda_name
        // drives the output name, and all wheels in the bundle
        // contribute their Requires-Dist to the merged run-deps.
        let bundle = Bundle {
            conda_name: "isaac-pack".into(),
            primary: rw(
                "isaacsim",
                meta(
                    "isaacsim",
                    "5.1.0.0",
                    vec!["numpy==1.26.0"],
                    true,
                ),
            ),
            extras: vec![
                rw(
                    "isaaclab",
                    meta("isaaclab", "0.51.1", vec!["scipy==1.15.0"], true),
                ),
                rw(
                    "pytorch3d",
                    meta(
                        "pytorch3d",
                        "0.7.8+5043d15pt2.7.0cu128",
                        vec!["pillow==11.0.0"],
                        true,
                    ),
                ),
            ],
            probe_decisions: vec![],
        };

        let output =
            produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[]).unwrap();

        // Output name is the bundle's conda_name, not any one entry name.
        assert_eq!(
            output.metadata.name.as_normalized(),
            "isaac-pack",
            "merged bundle's conda output should be named after the bundle group",
        );
        // All three wheels' Requires-Dist flow into the merged run-deps.
        let dep_names: Vec<String> = output
            .run_dependencies
            .depends
            .iter()
            .map(|d| d.name.clone())
            .collect();
        for sib_dep in ["numpy", "scipy", "pillow"] {
            assert!(
                dep_names.iter().any(|n| n == sib_dep),
                "merged bundle should emit {sib_dep} from one of its wheels; got: {dep_names:?}",
            );
        }
    }

    // Regression: handoff-2026-05-24. The merged-bundle primary (alphabetically
    // first entry in [retread-wheels]) was `isaaclab` -- a `py3-none-any`
    // wheel. After D rewrite the file on disk is named
    // `isaaclab-0.51.1-py3-none-any.relaxed.whl`. `WheelMetadata.is_pure_python`
    // was computed from `filename.contains("-none-any.whl")`, which is false
    // on the relaxed filename. `produce_output` then took the wheel-tag
    // fallback branch and `python_version_from_wheel_tag` returned bare
    // `"3"` (from the `py3` tag), so the emitted run-dep was `python 3.*`.
    // The conda solver, given `python 3.*`, picked python 3.14 and implied
    // `python_abi 3.14.* *_cp314`, which collided with the workspace's
    // `python==3.11` pin -- producing the user-facing error:
    //   isaac-pack 0.51.1 would require python_abi 3.14.* *_cp314
    // Fix: `is_pure_python` is now derived from the canonical PEP 425 tag
    // via `wheel::is_pure_python_wheel_filename`, which strips the
    // cosmetic `.relaxed.whl` suffix before checking the platform tag.
    // This test pins the *end-to-end* contract: a relaxed pure-Python
    // primary must yield `python {workspace_python_version}.*`, NOT
    // `python 3.*`. Previously only `wheel.rs` had unit coverage of the
    // helper; nothing asserted the downstream produce_output behavior, so
    // the bug shipped silently.
    #[test]
    fn relaxed_pure_python_primary_pins_python_to_workspace_variant() {
        use crate::wheel::is_pure_python_wheel_filename;

        // Construct a bundle whose primary mirrors the post-D state of
        // the `isaaclab` wheel that fooled the previous filename check.
        let filename = "isaaclab-0.51.1-py3-none-any.relaxed.whl".to_string();
        // Helper-driven is_pure_python -- this is exactly what wheel.rs's
        // read_metadata path produces for the on-disk relaxed wheel.
        let is_pure = is_pure_python_wheel_filename(&filename);
        assert!(
            is_pure,
            "regression guard: helper must report relaxed py3-none-any wheel as pure"
        );
        let primary = WheelMetadata {
            name: "isaaclab".into(),
            version: "0.51.1".into(),
            requires_dist: vec![],
            is_pure_python: is_pure,
            sha256: "sha".into(),
            filename,
        };
        let bundle = Bundle {
            conda_name: "isaac-pack".into(),
            primary: ResolvedWheel {
                pypi_name: "isaaclab".into(),
                url: "https://example.com/isaaclab-0.51.1-py3-none-any.relaxed.whl"
                    .parse()
                    .unwrap(),
                metadata: primary,
                extras_requested: vec![],
                auto_data: None,
                auto_data_dedup_skipped_root: None,
            },
            extras: vec![],
            probe_decisions: vec![],
        };

        let output =
            produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[]).unwrap();

        // The conda output's variant must be the workspace's 3.11, not the
        // bare-major "3" parsed from the py3 tag.
        let variant_python = output
            .metadata
            .variant
            .get("python")
            .map(|v| v.to_string())
            .expect("variant.python must be set");
        assert_eq!(
            variant_python, "3.11",
            "variant.python must inherit the workspace's variant, not the wheel tag's bare major"
        );

        // Run-dep must be `python 3.11.*`, NOT `python 3.*` (which lets the
        // solver pick 3.14 and triggers the python_abi 3.14 collision).
        // rattler's NamelessMatchSpec Debug format is structural:
        //   StrictRange(StartsWith, StrictVersion(Version { version: [[0], [3], [11]], local: [] }))
        // so we search for the segment list `[0], [3], [11]` -- bare-major
        // would render as `[0], [3]` with only TWO components. Anchor with
        // both `[3]` and `[11]` to catch the minor; absence of `[11]` is the
        // smoking gun for the regression we just fixed.
        let python_spec = output
            .run_dependencies
            .depends
            .iter()
            .find(|d| d.name == "python")
            .map(|d| format!("{:?}", d.spec))
            .expect("python in run_deps");
        assert!(
            python_spec.contains("[3]") && python_spec.contains("[11]"),
            "run-dep python must pin to 3.11 (look for [3] and [11] segments); got: {python_spec}"
        );
        assert!(
            python_spec.contains("StartsWith"),
            "run-dep python must use StartsWith range (== `3.11.*`); got: {python_spec}"
        );

        // build_v1 reads the python version from the output's variant when
        // constructing the recipe; so as long as variant.python is right,
        // the recipe is too. The variant assertion above already covers
        // this -- pinning the contract here in case the read path changes.
    }

    #[test]
    fn bare_major_python_emits_glob_not_strict_equals() {
        // Regression: when python_version is bare-major like "3" (e.g.
        // wheel tag parsing yields just the major, or a workspace
        // variant is "python = [\"3\"]"), the emitted host-dep was
        // `python 3` which rattler-conda-types Lenient-parses as
        // `==3` strict, causing rattler-build to fail the host solve
        // with "No candidates were found for python ==3". Always
        // append `.*` so the glob form is used.
        //
        // Construct a bundle whose primary wheel produces python_version
        // = "3" via the pure-Python fallback (workspace_python_version)
        // -- pass "3" as the workspace_python_version arg.
        let bundle = solo_bundle("foo", vec![]);
        let output = produce_output(&bundle, &cfg(), Platform::Linux64, "3", &[]).unwrap();

        // python must appear with a wildcard, NOT as strict equals.
        let python = output
            .host_dependencies
            .as_ref()
            .unwrap()
            .depends
            .iter()
            .find(|d| d.name == "python")
            .expect("python in host_deps");
        let rendered = format!("{:?}", python.spec);
        assert!(
            !rendered.contains("Equals") || rendered.contains("Glob") || rendered.contains("*"),
            "host python dep must be a glob, not strict ==; got: {rendered}",
        );
    }

    #[test]
    fn cross_output_siblings_appear_as_run_deps() {
        // Contract: when a pack emits multiple outputs (isaacsim,
        // isaaclab, isaaclab-arena, pytorch3d, ...), each output's
        // run_dependencies must include every sibling at exact version.
        // Then the workspace pixi.toml can declare just one name and
        // conda transitively pulls the rest; without this, every
        // [retread-wheels] entry needs its own line in the consumer's
        // pixi.toml or its conda package sits unused. Also pins that
        // PEP 440 local version identifiers (the +5043d15... in
        // pytorch3d's version) survive verbatim in the run-dep spec.
        let bundle = solo_bundle("isaacsim", vec![]);
        let siblings = vec![
            ("isaacsim".to_string(), "1.0.0".to_string()), // self -- must be skipped
            ("isaaclab".to_string(), "0.51.1".to_string()),
            ("isaaclab-arena".to_string(), "0.4.2".to_string()),
            ("pytorch3d".to_string(), "0.7.8+5043d15pt2.7.0cu128".to_string()),
        ];
        let output =
            produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &siblings).unwrap();

        let dep_names: Vec<String> = output
            .run_dependencies
            .depends
            .iter()
            .map(|d| d.name.clone())
            .collect();

        // Self must not appear in own run-deps.
        assert_eq!(
            dep_names.iter().filter(|n| *n == "isaacsim").count(),
            0,
            "self should not be a sibling run-dep; got {dep_names:?}",
        );
        for sib in ["isaaclab", "isaaclab-arena", "pytorch3d"] {
            assert!(
                dep_names.iter().any(|n| n == sib),
                "missing sibling run-dep {sib}; got: {dep_names:?}",
            );
        }

        // Spec for the pytorch3d sibling must preserve the +local
        // version identifier. conda's match-spec parser splits the
        // local segment into tokens (e.g. `[5043, 'd', 15, 'pt', 2]`),
        // so the original string isn't contiguous in Debug -- check
        // for the distinctive token components and the `local:` marker
        // proving the +local segment was stored at all.
        let pytorch3d_spec = output
            .run_dependencies
            .depends
            .iter()
            .find(|d| d.name == "pytorch3d")
            .map(|d| format!("{:?}", d.spec))
            .expect("pytorch3d in deps");
        assert!(
            pytorch3d_spec.contains("local:"),
            "pytorch3d sibling pin lost the +local segment entirely; got: {pytorch3d_spec}",
        );
        for token in ["5043", "pt", "cu", "128"] {
            assert!(
                pytorch3d_spec.contains(token),
                "pytorch3d sibling pin missing token `{token}` from +local segment; got: {pytorch3d_spec}",
            );
        }
    }

    #[test]
    fn prefer_conda_skips_parselmouth_known_deps() {
        // Contract: anything in the effective name_map (parselmouth +
        // FALLBACK + user retread-name-map) is NOT auto-bundled -- it
        // flows to emission as a conda run-dep via translate. This is
        // the prefer-conda default. Concretely: torch in the bundle's
        // candidates should be skipped because parselmouth maps it to
        // pytorch; a niche pure-PyPI helper with no conda equivalent
        // (e.g. qdldl) should not be skipped.
        let mut name_map: std::collections::BTreeMap<String, String> =
            std::collections::BTreeMap::new();
        name_map.insert("torch".to_string(), "pytorch".to_string());
        name_map.insert("numpy".to_string(), "numpy".to_string());
        name_map.insert(
            "opencv-python-headless".to_string(),
            "py-opencv".to_string(),
        );

        // Parselmouth-known => prefer conda, don't bundle.
        assert!(prefer_conda_match("torch", &name_map));
        assert!(prefer_conda_match("numpy", &name_map));
        // FALLBACK_PYPI_TO_CONDA entry survives the same way.
        assert!(prefer_conda_match("opencv-python-headless", &name_map));

        // Unknown to parselmouth => fall through to auto-bundle path.
        // (These are the long tail retread should still vendor.)
        assert!(!prefer_conda_match("qdldl", &name_map));
        assert!(!prefer_conda_match("asteval", &name_map));
        assert!(!prefer_conda_match("aiodns", &name_map));
    }

    #[test]
    fn pep508_extra_dep_accepts_range_specifier() {
        // Regression: isaacsim's METADATA has
        //   Requires-Dist: isaacsim-extscache-kit>=5 ; extra == "extscache"
        // We used to require an exact `==X.Y.Z` pin on extras-gated
        // requirements and bail otherwise. NVIDIA publishes plenty of
        // metapackages that use `>=` to gate to a major series, so range
        // resolution is a hard requirement -- we now resolve the highest
        // matching version off the index instead of refusing.
        let dep = pep508_extra_dep(
            "isaacsim-extscache-kit>=5 ; extra == \"extscache\"",
            "extscache",
        )
        .expect("pep508 parse")
        .expect("extras-gated dep, got None");
        assert_eq!(dep.name, "isaacsim-extscache-kit");
        let specs = match &dep.source {
            ExtraDepSource::Pypi(s) => s,
            other => panic!("expected PyPI source, got {other:?}"),
        };
        assert!(
            specs.to_string().contains(">=5"),
            "specifiers must preserve `>=5`, got: {specs}",
        );
    }

    /// v0.12.0+: URL Requires-Dist (PEP 508 `pkg @ <url>`) is parsed
    /// into `ExtraDepSource::Git` / `Url` instead of bailing. Mirrors
    /// IsaacLab's `rl_games` extra:
    ///   Requires-Dist: rl-games @ git+https://.../rl_games.git@python3.11 ; extra == "rl_games"
    #[test]
    fn pep508_extra_dep_handles_git_url() {
        let dep = pep508_extra_dep(
            "rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11 ; extra == \"rl_games\"",
            "rl_games",
        )
        .expect("pep508 parse")
        .expect("extras-gated dep, got None");
        assert_eq!(dep.name, "rl-games");
        match dep.source {
            ExtraDepSource::Git { url, rev } => {
                assert_eq!(url, "https://github.com/isaac-sim/rl_games.git");
                assert_eq!(rev.as_deref(), Some("python3.11"));
            }
            other => panic!("expected Git source, got {other:?}"),
        }
    }

    #[test]
    fn pep508_extra_dep_handles_direct_url() {
        let dep = pep508_extra_dep(
            "foo @ https://example.com/foo-1.2.3-py3-none-any.whl ; extra == \"foo\"",
            "foo",
        )
        .expect("pep508 parse")
        .expect("extras-gated dep, got None");
        assert_eq!(dep.name, "foo");
        match dep.source {
            ExtraDepSource::Url(u) => {
                assert_eq!(u.as_str(), "https://example.com/foo-1.2.3-py3-none-any.whl");
            }
            other => panic!("expected Url source, got {other:?}"),
        }
    }

    /// Regression: extras-gated bare names (no version, no URL) are
    /// legal PEP 508 -- e.g. IsaacLab's
    ///   Requires-Dist: tqdm; extra == "sb3"
    ///   Requires-Dist: gym; extra == "rl_games"
    ///   Requires-Dist: rich; extra == "sb3"
    /// Previously made retread bail with "no version or URL". Now
    /// resolves as PyPI with empty specifiers -> latest matching the
    /// target python.
    #[test]
    fn pep508_extra_dep_handles_bare_name() {
        let dep = pep508_extra_dep(
            "tqdm; extra == \"sb3\"",
            "sb3",
        )
        .expect("pep508 parse")
        .expect("extras-gated bare-name dep, got None");
        assert_eq!(dep.name, "tqdm");
        match dep.source {
            ExtraDepSource::Pypi(specs) => {
                assert_eq!(specs.to_string(), "", "bare name -> empty specifiers, got: {specs}");
            }
            other => panic!("expected PyPI source, got {other:?}"),
        }
    }

    /// git URL without `@<rev>` -> Some(url), None for rev. Pipeline
    /// will default to HEAD when synthesizing the WheelEntry.
    #[test]
    fn extra_dep_source_from_url_git_without_rev() {
        let url: url::Url = "git+https://github.com/foo/bar.git".parse().unwrap();
        match extra_dep_source_from_url(&url).expect("parse") {
            ExtraDepSource::Git { url, rev } => {
                assert_eq!(url, "https://github.com/foo/bar.git");
                assert_eq!(rev, None);
            }
            other => panic!("expected Git, got {other:?}"),
        }
    }
}
