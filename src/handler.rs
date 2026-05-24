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
use rattler_conda_types::{NoArchType, PackageName, Platform, VersionSpec, VersionWithSource};
use serde_json::Value;
use tokio::sync::RwLock;
use uv_pep508::uv_pep440::Operator;

use crate::config::{RetreadConfig, WheelEntry};
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
        Ok(InitializeResult {})
    }

    async fn conda_outputs(
        &self,
        params: CondaOutputsParams,
    ) -> Result<CondaOutputsResult, RpcError> {
        let (config, download_dir) = self.snapshot(&params.work_directory).await?;

        // Pick the target Python versions. Precedence:
        //   1. workspace.build-variants python = [...]
        //   2. [build.config] python = "3.11" / ["3.11", "3.12"]
        //   3. DEFAULT_PYTHON
        let pythons = pythons_for(&config, params.variant_configuration.as_ref());

        // Fan out: one output per (python, wheel). If a target python has no
        // matching wheel on the index (e.g. user asked for 3.12 but the
        // upstream wheel is cp311-only), log and skip the combination rather
        // than failing the whole call -- other Python versions might still
        // resolve cleanly.
        let mut outputs = Vec::new();
        for python_version in &pythons {
            let target = wheel_target_for(params.host_platform, python_version);
            let resolved = match resolve_all(&config, &target, &download_dir).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        python = %python_version,
                        error = %format!("{e:#}"),
                        "skipping python variant: no matching wheels"
                    );
                    continue;
                }
            };
            for bundle in &resolved {
                outputs.push(
                    produce_output(bundle, &config, params.host_platform, python_version)
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
        let (config, download_dir) = self.snapshot(&params.work_directory).await?;
        // conda/build_v1 doesn't carry the variant set; the chosen variant
        // is encoded in params.output.variant. Look up `python` there;
        // fall back to the default if absent.
        let python_version = params
            .output
            .variant
            .get("python")
            .map(|v| v.to_string())
            .unwrap_or_else(|| DEFAULT_PYTHON.to_string());
        let target = wheel_target_for(params.output.subdir, &python_version);

        // We re-resolve the full set (the lookup is cheap once cached) and
        // pick the entry matching the requested output.
        let resolved = resolve_all(&config, &target, &download_dir)
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
            &config,
            &params.work_directory,
            &output_dir,
            params.output.subdir,
        )
        .await
        .map_err(|e| RpcError::internal(format!("build {}: {e:#}", picked.conda_name)))
    }

    async fn snapshot(&self, work_dir: &Path) -> Result<(RetreadConfig, PathBuf), RpcError> {
        let state = self.state.read().await;
        let config = state
            .config
            .clone()
            .ok_or_else(|| RpcError::internal("initialize was not called"))?;
        let download_dir = state
            .cache_dir
            .clone()
            .map(|d| d.join("pixi-build-retread-wheels"))
            .unwrap_or_else(|| work_dir.join("wheels"));
        Ok((config, download_dir))
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
) -> Result<Vec<Bundle>> {
    let mut bundles = Vec::with_capacity(config.retread_wheels.len());
    for (entry_name, entry) in &config.retread_wheels {
        bundles.push(resolve_bundle(entry_name, entry, target, download_dir).await?);
    }
    Ok(bundles)
}

async fn resolve_bundle(
    entry_name: &str,
    entry: &WheelEntry,
    target: &WheelTarget,
    download_dir: &Path,
) -> Result<Bundle> {
    let conda_name = conda_name_from(entry_name);
    let mut seen: HashSet<String> = HashSet::new();
    let mut work: VecDeque<Pending> = VecDeque::new();

    // Resolve and fetch the primary wheel first.
    let primary = if let Some(url) = &entry.url {
        let metadata = fetch_and_parse(url, entry.sha256.as_deref(), download_dir).await?;
        ResolvedWheel {
            pypi_name: conda_name_from(entry_name),
            url: url.clone(),
            metadata,
        }
    } else {
        let version = entry
            .normalized_version()
            .ok_or_else(|| anyhow!("wheel `{entry_name}` has neither url nor version"))?;
        let resolved = pypi::resolve(&entry.index_url(), entry_name, &version, target).await?;
        let metadata =
            fetch_and_parse(&resolved.url, resolved.sha256.as_deref(), download_dir).await?;
        ResolvedWheel {
            pypi_name: conda_name_from(entry_name),
            url: resolved.url,
            metadata,
        }
    };
    seen.insert(primary.pypi_name.clone());

    // Seed BFS from the primary's deps. Two flavors:
    // 1. Extras-gated (`; extra == "X"`) for each requested extra.
    // 2. Sibling base deps -- requirements without an extras marker whose
    //    PyPI name shares the entry's namespace prefix (`<entry>-...`).
    //    Real-world example: the isaacsim metapackage lists
    //    `Requires-Dist: isaacsim-kernel==5.1.0.0` (no marker) because the
    //    kernel is essential to ANY install of isaacsim. We bundle these
    //    sub-packages so the conda solver doesn't try to find them
    //    separately.
    let prefix = format!("{}-", conda_name);
    seed_worklist(
        &primary.metadata,
        &entry.extras,
        &entry.index_url(),
        &prefix,
        &seen,
        &mut work,
    )?;

    // BFS, accumulating sub-wheels.
    let mut extras = Vec::new();
    while let Some(pending) = work.pop_front() {
        let dep_conda_name = conda_name_from(&pending.pypi_name);
        if !seen.insert(dep_conda_name.clone()) {
            continue;
        }
        let resolved = pypi::resolve(&pending.index, &pending.pypi_name, &pending.version, target)
            .await
            .with_context(|| {
                format!(
                    "resolving {}=={} on index {}",
                    pending.pypi_name, pending.version, pending.index,
                )
            })?;
        let metadata =
            fetch_and_parse(&resolved.url, resolved.sha256.as_deref(), download_dir).await?;

        // Recurse: this sub-wheel's own extras and prefix-matching base
        // deps also get pulled in.
        seed_worklist(
            &metadata,
            &pending.extras,
            &pending.index,
            &prefix,
            &seen,
            &mut work,
        )?;

        extras.push(ResolvedWheel {
            pypi_name: dep_conda_name,
            url: resolved.url,
            metadata,
        });
    }

    Ok(Bundle {
        conda_name,
        primary,
        extras,
    })
}

/// One unit of pending work in the resolver BFS.
#[derive(Debug, Clone)]
struct Pending {
    pypi_name: String,
    version: String,
    index: String,
    /// Extras to activate on this wheel. Drives further worklist additions
    /// for `Requires-Dist: name ; extra == "X"` lines.
    extras: Vec<String>,
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
                    version: dep.version,
                    index: index.to_string(),
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
                version: dep.version,
                index: index.to_string(),
                extras: dep.extras,
            });
        }
    }
    Ok(())
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

    let Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) = req.version_or_url.as_ref() else {
        return Ok(None);
    };
    let specs: Vec<_> = specs.iter().collect();
    if specs.len() != 1 || *specs[0].operator() != Operator::Equal {
        return Ok(None);
    }
    Ok(Some(ExtraDep {
        name: req.name.to_string(),
        version: specs[0].version().to_string(),
        extras: req.extras.iter().map(|e| e.to_string()).collect(),
    }))
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

/// One extras-derived dependency: PyPI name + exact version + any extras
/// the requirement itself declares (`pkg[foo,bar]==1.0` form).
#[derive(Debug, Clone)]
struct ExtraDep {
    name: String,
    version: String,
    extras: Vec<String>,
}

/// Returns `Some(ExtraDep)` if `raw` is a `Requires-Dist` line that is
/// gated on the requested extra, and is an exact `==` pin. Returns None if
/// the requirement is gated on a different extra (or has no marker, i.e.
/// is a base dep we don't repack at all).
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

    // Need an exact == specifier we can use to drive the resolver.
    let Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) = req.version_or_url.as_ref() else {
        bail!("extra `{extra}` requirement has no version specifier: {raw}");
    };
    let specs: Vec<_> = specs.iter().collect();
    if specs.len() != 1 || *specs[0].operator() != Operator::Equal {
        bail!(
            "extra `{extra}` requires an exact version pin, got `{raw}`. \
             Range resolution is on the TODO list."
        );
    }
    Ok(Some(ExtraDep {
        name: req.name.to_string(),
        version: specs[0].version().to_string(),
        extras: req.extras.iter().map(|e| e.to_string()).collect(),
    }))
}

fn produce_output(
    bundle: &Bundle,
    config: &RetreadConfig,
    host_platform: Platform,
    workspace_python_version: &str,
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

    let python_dep = if python_version.contains('.') {
        format!("python {python_version}.*")
    } else {
        format!("python {python_version}")
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

async fn build_one(
    bundle: &Bundle,
    config: &RetreadConfig,
    work_dir: &Path,
    output_dir: &Path,
    target_subdir: Platform,
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
    let recipe = build_bundle_recipe(&bundle.conda_name, &sources, config)?;
    let yaml = to_yaml(&recipe)?;

    let recipe_dir = work_dir.join(format!("recipe-{}", recipe.package.name));
    tokio::fs::create_dir_all(&recipe_dir).await?;
    let recipe_path = recipe_dir.join("recipe.yaml");
    tokio::fs::write(&recipe_path, &yaml).await?;
    tracing::info!(path = %recipe_path.display(), "wrote recipe.yaml");

    tokio::fs::create_dir_all(output_dir).await?;

    let target_platform = target_subdir.to_string();
    let status = tokio::process::Command::new("rattler-build")
        .arg("build")
        .arg("--recipe")
        .arg(&recipe_path)
        .arg("--output-dir")
        .arg(output_dir)
        .arg("--target-platform")
        .arg(&target_platform)
        .arg("--no-test")
        .stdin(std::process::Stdio::null())
        .status()
        .await
        .context("spawning rattler-build (is it on PATH?)")?;
    if !status.success() {
        bail!("rattler-build exited with status {status}");
    }

    let subdir_dir = output_dir.join(&target_platform);
    let output_file =
        find_conda_artifact(&subdir_dir, &recipe.package.name, &recipe.package.version).await?;

    let py_short = python_version_from_wheel_tag(&bundle.primary.metadata.filename)
        .unwrap_or_default()
        .replace('.', "");
    Ok(CondaBuildV1Result {
        output_file,
        input_globs: Default::default(),
        name: recipe.package.name.clone(),
        version: VersionWithSource::from_str(&recipe.package.version)?,
        build: format!("py{py_short}_{}", config.build_number),
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
        }
    }

    fn solo_bundle(name: &str, requires: Vec<&str>) -> Bundle {
        Bundle {
            conda_name: name.into(),
            primary: rw(name, meta(name, "1.0.0", requires, true)),
            extras: vec![],
        }
    }

    #[test]
    fn built_in_win_only_dropped_on_linux() {
        // idna-ssl is in BUILT_IN_WIN_ONLY. Targeting linux-64, it must
        // not appear in run-deps even though it has no explicit
        // `sys_platform == "win32"` marker.
        let bundle = solo_bundle("isaacsim", vec!["idna-ssl==1.1.0", "numpy==1.26.0"]);
        let output = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11").unwrap();
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
        let output = produce_output(&bundle, &cfg(), Platform::Win64, "3.11").unwrap();
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
        let output = produce_output(&bundle, &config, Platform::Linux64, "3.11").unwrap();
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
        let output = produce_output(&bundle, &config, Platform::Linux64, "3.11").unwrap();
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
        };

        let output = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11").unwrap();
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
}
