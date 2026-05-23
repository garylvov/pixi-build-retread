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

use crate::config::RetreadConfig;
use crate::pypi::{self, WheelTarget};
use crate::recipe::{build_recipe, to_yaml};
use crate::relax::{default_marker_env, marker_env_for, python_version_from_wheel_tag};
use crate::rpc::{ok, parse_params, RpcError};
use crate::wheel::{fetch_wheel, read_metadata, WheelMetadata};

const NEGOTIATE: &str = "negotiateCapabilities";
const INITIALIZE: &str = "initialize";
const CONDA_OUTPUTS: &str = "conda/outputs";
const CONDA_BUILD_V1: &str = "conda/build_v1";

const DEFAULT_PYTHON: &str = "3.11";

#[derive(Default)]
struct State {
    config: Option<RetreadConfig>,
    cache_dir: Option<PathBuf>,
}

#[derive(Clone, Default)]
pub struct Handler {
    state: Arc<RwLock<State>>,
}

/// One wheel after full resolution: URL is concrete, metadata parsed,
/// conda package name decided. Both URL-form and spec-form entries collapse
/// into this once we've fetched + parsed the wheel.
#[derive(Debug, Clone)]
struct Resolved {
    conda_name: String,
    url: url::Url,
    metadata: WheelMetadata,
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
        let python_version = python_from_variants(params.variant_configuration.as_ref());
        let target = wheel_target_for(params.host_platform, &python_version);

        let resolved = resolve_all(&config, &target, &download_dir)
            .await
            .map_err(|e| RpcError::internal(format!("resolving wheels: {e:#}")))?;

        let mut outputs = Vec::with_capacity(resolved.len());
        for r in &resolved {
            outputs.push(
                produce_output(r, &config, params.host_platform, &python_version).map_err(
                    |e| RpcError::internal(format!("output for {}: {e:#}", r.conda_name)),
                )?,
            );
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
            .find(|r| r.conda_name == requested)
            .ok_or_else(|| {
                RpcError::invalid_params(format!(
                    "no resolved wheel matches requested output `{requested}`; \
                     known: {:?}",
                    resolved.iter().map(|r| &r.conda_name).collect::<Vec<_>>()
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

fn python_from_variants(
    variants: Option<&std::collections::BTreeMap<String, Vec<VariantValue>>>,
) -> String {
    variants
        .and_then(|v| v.get("python"))
        .and_then(|values| values.first())
        .map(|val| val.to_string())
        .unwrap_or_else(|| DEFAULT_PYTHON.to_string())
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

/// Resolve every user-supplied entry into a flat list of concrete wheels,
/// recursively expanding `extras` — every newly-discovered wheel has its
/// own extras (if any were requested by the parent's Requires-Dist line)
/// followed, until no new wheels are discovered. Cycle-detected by
/// conda-normalized name so that diamond dependencies (A and B both
/// require C) and self-cycles are handled.
async fn resolve_all(
    config: &RetreadConfig,
    target: &WheelTarget,
    download_dir: &Path,
) -> Result<Vec<Resolved>> {
    let mut out = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut work: VecDeque<Pending> = VecDeque::new();

    // Seed worklist from user-supplied entries.
    for (entry_name, entry) in &config.retread_wheels {
        let map_key_name = conda_name_from(entry_name);
        if let Some(url) = &entry.url {
            // URL form: resolve inline (no index lookup needed).
            if !seen.insert(map_key_name.clone()) {
                continue;
            }
            let metadata = fetch_and_parse(url, entry.sha256.as_deref(), download_dir).await?;
            out.push(Resolved {
                conda_name: map_key_name,
                url: url.clone(),
                metadata,
            });
        } else {
            // Spec form: dispatch through the worklist.
            let version = entry
                .normalized_version()
                .ok_or_else(|| anyhow!("wheel `{entry_name}` has neither url nor version"))?;
            work.push_back(Pending {
                pypi_name: entry_name.clone(),
                version,
                index: entry.index_url(),
                extras: entry.extras.clone(),
                override_conda_name: Some(map_key_name),
            });
        }
    }

    while let Some(pending) = work.pop_front() {
        let conda_name = pending
            .override_conda_name
            .clone()
            .unwrap_or_else(|| conda_name_from(&pending.pypi_name));
        if !seen.insert(conda_name.clone()) {
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
        out.push(Resolved {
            conda_name: conda_name.clone(),
            url: resolved.url,
            metadata: metadata.clone(),
        });

        // Each requested extra contributes more entries to the worklist.
        // Sub-packages' own extras (encoded as `pkg[foo,bar]==1.0` in the
        // parent's Requires-Dist line) are also followed.
        for extra in &pending.extras {
            for raw in &metadata.requires_dist {
                let Some(dep) = pep508_extra_dep(raw, extra)? else {
                    continue;
                };
                let dep_conda_name = conda_name_from(&dep.name);
                if seen.contains(&dep_conda_name) {
                    continue;
                }
                work.push_back(Pending {
                    pypi_name: dep.name,
                    version: dep.version,
                    // Extras-derived deps inherit the parent's index. This
                    // matches how `isaacsim[all]` works: the `==X` pin in
                    // `Requires-Dist: isaacsim-core==5.1.0.0 ; extra == "all"`
                    // resolves on the same NVIDIA index.
                    index: pending.index.clone(),
                    extras: dep.extras,
                    override_conda_name: None,
                });
            }
        }
    }

    Ok(out)
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
    /// If `Some`, use this as the conda package name instead of deriving
    /// from `pypi_name`. Set for user-supplied entries so the map key wins
    /// over the wheel's METADATA Name.
    override_conda_name: Option<String>,
}

fn conda_name_from(pypi_name: &str) -> String {
    pypi_name.to_ascii_lowercase().replace('_', "-")
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
    r: &Resolved,
    config: &RetreadConfig,
    host_platform: Platform,
    workspace_python_version: &str,
) -> Result<CondaOutput> {
    // Prefer the workspace's requested Python version; only fall back to
    // parsing the wheel filename if the variant doesn't say anything (e.g.
    // for a noarch / pure-python wheel where the cp tag is `py3`).
    let python_version = if r.metadata.is_pure_python {
        workspace_python_version.to_string()
    } else {
        python_version_from_wheel_tag(&r.metadata.filename)
            .unwrap_or_else(|| workspace_python_version.to_string())
    };
    let subdir = if r.metadata.is_pure_python {
        Platform::NoArch
    } else {
        host_platform
    };

    let python_dep = if python_version.contains('.') {
        format!("python {python_version}.*")
    } else {
        format!("python {python_version}")
    };

    let mut depends = vec![spec_from_str(&python_dep)?];
    let env = marker_env_for(&host_platform.to_string(), &python_version)?;
    for raw in &r.metadata.requires_dist {
        if let Some(dep) = crate::relax::translate(
            raw,
            &env,
            &config.name_map,
            &config.overrides,
            config.relax,
        )? {
            depends.push(spec_from_str(&dep.0)?);
        }
    }

    let name = PackageName::new_unchecked(r.conda_name.clone());
    let version = VersionWithSource::from_str(&r.metadata.version)
        .map_err(|e| anyhow!("parsing version `{}`: {e}", r.metadata.version))?;
    let noarch = if r.metadata.is_pure_python {
        NoArchType::python()
    } else {
        NoArchType::none()
    };
    let py_short = python_version.replace('.', "");
    let build = format!("py{py_short}_{}", config.build_number);

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
            variant: Default::default(),
        },
        build_dependencies: None,
        host_dependencies: Some(CondaOutputDependencies {
            depends: vec![spec_from_str(&python_dep)?, spec_from_str("pip")?],
            constraints: Vec::new(),
        }),
        run_dependencies: CondaOutputDependencies {
            depends,
            constraints: Vec::new(),
        },
        ignore_run_exports: CondaOutputIgnoreRunExports::default(),
        run_exports: CondaOutputRunExports::default(),
        input_globs: None,
    })
}

async fn build_one(
    r: &Resolved,
    config: &RetreadConfig,
    work_dir: &Path,
    output_dir: &Path,
    target_subdir: Platform,
) -> Result<CondaBuildV1Result> {
    // Override the recipe's package name with the conda_name we decided
    // earlier (which may differ from the wheel's METADATA Name when the
    // user supplied a map-key override).
    let mut recipe = build_recipe(&r.metadata, &r.url, config)?;
    recipe.package.name = r.conda_name.clone();
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

    let py_short = python_version_from_wheel_tag(&r.metadata.filename)
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
