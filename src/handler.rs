//! JSON-RPC method handlers. The four entry points pixi calls.

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
use pixi_build_types::{BackendCapabilities, BinaryPackageSpec, NamedSpec, PackageSpec};
use rattler_conda_types::{NoArchType, PackageName, Platform, VersionSpec, VersionWithSource};
use serde_json::Value;
use tokio::sync::RwLock;

use crate::config::{RetreadConfig, WheelEntry};
use crate::recipe::{build_recipe, to_yaml};
use crate::relax::python_version_from_wheel_tag;
use crate::rpc::{ok, parse_params, RpcError};
use crate::wheel::{fetch_wheel, read_metadata, WheelMetadata};

const NEGOTIATE: &str = "negotiateCapabilities";
const INITIALIZE: &str = "initialize";
const CONDA_OUTPUTS: &str = "conda/outputs";
const CONDA_BUILD_V1: &str = "conda/build_v1";

#[derive(Default)]
struct State {
    config: Option<RetreadConfig>,
    cache_dir: Option<PathBuf>,
}

#[derive(Clone, Default)]
pub struct Handler {
    state: Arc<RwLock<State>>,
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
                    "pixi-build-retread requires a [build.config] table with at least `wheels = [...]`",
                ))
            }
        };

        if config.wheels.is_empty() {
            return Err(RpcError::invalid_params(
                "[build.config].wheels must list at least one wheel",
            ));
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

        let mut outputs = Vec::with_capacity(config.wheels.len());
        for wheel in &config.wheels {
            let output = produce_output(wheel, &config, &download_dir, params.host_platform)
                .await
                .map_err(|e| RpcError::internal(format!("processing {}: {e:#}", wheel.url)))?;
            outputs.push(output);
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

        // Match the requested output to one of our configured wheels by name.
        let requested_name = params.output.name.as_normalized().to_string();
        let wheel = config
            .wheels
            .iter()
            .find(|w| filename_to_pep503(&extract_filename(&w.url)) == requested_name)
            .ok_or_else(|| {
                RpcError::invalid_params(format!(
                    "no [build.config].wheels entry matches requested output `{requested_name}`"
                ))
            })?;

        let output_dir = params
            .output_directory
            .clone()
            .unwrap_or_else(|| params.work_directory.join("output"));
        build_one(
            wheel,
            &config,
            &download_dir,
            &params.work_directory,
            &output_dir,
            params.output.subdir,
        )
        .await
        .map_err(|e| RpcError::internal(format!("build {}: {e:#}", wheel.url)))
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

async fn produce_output(
    wheel: &WheelEntry,
    config: &RetreadConfig,
    download_dir: &Path,
    host_platform: Platform,
) -> Result<CondaOutput> {
    let metadata = fetch_and_read(wheel, download_dir).await?;
    let python_version = python_version_from_wheel_tag(&metadata.filename).unwrap_or_else(|| "3".into());
    let subdir = if metadata.is_pure_python {
        Platform::NoArch
    } else {
        host_platform
    };

    let mut depends = Vec::new();
    let python_dep = if python_version.contains('.') {
        format!("python {python_version}.*")
    } else {
        format!("python {python_version}")
    };
    depends.push(spec_from_str(&python_dep)?);

    // Reuse the same translation pipeline the recipe generator uses so the
    // conda/outputs metadata stays in sync with what conda/build_v1 will
    // produce.
    let env = crate::relax::default_marker_env(&python_version)?;
    for raw in &metadata.requires_dist {
        match crate::relax::translate(
            raw,
            &env,
            &config.name_map,
            &config.overrides,
            config.relax,
        )? {
            Some(dep) => depends.push(spec_from_str(&dep.0)?),
            None => {}
        }
    }

    let conda_name = metadata.name.to_ascii_lowercase().replace('_', "-");
    let name = PackageName::new_unchecked(conda_name);
    let version = VersionWithSource::from_str(&metadata.version)
        .map_err(|e| anyhow!("parsing version `{}`: {e}", metadata.version))?;

    let noarch = if metadata.is_pure_python {
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
            depends: vec![
                spec_from_str(&python_dep)?,
                spec_from_str("pip")?,
            ],
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
    wheel: &WheelEntry,
    config: &RetreadConfig,
    download_dir: &Path,
    work_dir: &Path,
    output_dir: &Path,
    target_subdir: Platform,
) -> Result<CondaBuildV1Result> {
    let metadata = fetch_and_read(wheel, download_dir).await?;
    let recipe = build_recipe(&metadata, &wheel.url, config)?;
    let yaml = to_yaml(&recipe)?;

    // Lay out a clean recipe directory under work_dir/<package>.
    let recipe_dir = work_dir.join(format!("recipe-{}", recipe.package.name));
    tokio::fs::create_dir_all(&recipe_dir).await?;
    let recipe_path = recipe_dir.join("recipe.yaml");
    tokio::fs::write(&recipe_path, &yaml).await?;
    tracing::info!(path = %recipe_path.display(), "wrote recipe.yaml");

    tokio::fs::create_dir_all(output_dir).await?;

    // Shell out to rattler-build. The conda recipe declares rattler-build as a
    // run dep so the binary is on PATH when pixi invokes us.
    let target_platform = platform_str(target_subdir);
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

    // Locate the produced .conda file. rattler-build writes
    // `<output_dir>/<subdir>/<name>-<version>-<build>.conda`.
    let subdir_dir = output_dir.join(&target_platform);
    let output_file = find_conda_artifact(&subdir_dir, &recipe.package.name, &recipe.package.version)
        .await?;

    let py_short = python_version_from_wheel_tag(&metadata.filename)
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

async fn fetch_and_read(wheel: &WheelEntry, download_dir: &Path) -> Result<WheelMetadata> {
    let path = fetch_wheel(&wheel.url, wheel.sha256.as_deref(), download_dir).await?;
    let metadata = tokio::task::spawn_blocking(move || read_metadata(&path))
        .await
        .context("metadata reader panicked")??;
    Ok(metadata)
}

fn extract_filename(url: &url::Url) -> String {
    url.path_segments()
        .and_then(|s| s.last())
        .map(|s| s.to_string())
        .unwrap_or_default()
}

fn filename_to_pep503(filename: &str) -> String {
    // `Foo_Bar-1.2.3-cp311-...whl` -> `foo-bar`
    let stem = filename.split('-').next().unwrap_or("");
    let mut out = String::new();
    let mut prev_dash = false;
    for c in stem.to_ascii_lowercase().chars() {
        if c == '_' || c == '.' || c == '-' {
            if !prev_dash {
                out.push('-');
                prev_dash = true;
            }
        } else {
            out.push(c);
            prev_dash = false;
        }
    }
    out.trim_matches('-').to_string()
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

fn platform_str(p: Platform) -> String {
    p.to_string()
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
