//! Cached conda compiler environments for hermetic native wheel builds.
//!
//! `rattler-build debug setup` is useful here because it reuses the same
//! conda solver, prefix installer, and compiler activation that build recipes
//! use. Its generated activation scripts contain absolute prefix paths, so an
//! environment is provisioned directly in its final tuple-keyed cache path;
//! staging it elsewhere and renaming it would make the activation invalid.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use tokio::process::Command;

const CACHE_SCHEMA: &str = "retread-hermetic-build-environment-v4";
const CACHE_NAMESPACE: &str = "hermetic-build-envs";
const CACHE_VERSION: &str = "v4";
const COMPLETION_MARKER: &str = "complete.json";
const MIN_RATTLER_BUILD_VERSION: (u64, u64, u64) = (0, 70, 0);

/// A validated, immutable compiler environment ready to activate around a
/// PEP 517 build. Clones are cheap path/value copies; the underlying prefix is
/// shared read-only after its completion marker is published.
#[derive(Debug, Clone)]
pub struct HermeticBuildEnvironment {
    activation_script: PathBuf,
    python_executable: PathBuf,
    sysroot_path: PathBuf,
    cuda_executable: Option<PathBuf>,
    selected_sysroot: (u32, u32),
    platform_tag: String,
}

impl HermeticBuildEnvironment {
    pub fn activation_script(&self) -> &Path {
        &self.activation_script
    }

    pub fn python_executable(&self) -> &Path {
        &self.python_executable
    }

    pub fn sysroot_path(&self) -> &Path {
        &self.sysroot_path
    }

    pub fn cuda_executable(&self) -> Option<&Path> {
        self.cuda_executable.as_deref()
    }

    pub fn selected_sysroot(&self) -> (u32, u32) {
        self.selected_sysroot
    }

    pub fn platform_tag(&self) -> &str {
        &self.platform_tag
    }
}

#[derive(Debug, Clone)]
struct ProvisionRequest {
    target_floor: (u32, u32),
    python_minor: String,
    cuda_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletionMarker {
    schema: String,
    target_floor: (u32, u32),
    python_minor: String,
    cuda_version: Option<String>,
    selected_sysroot_version: String,
    selected_sysroot: (u32, u32),
    root_versions: BTreeMap<String, String>,
    work_dir: PathBuf,
    activation_script: PathBuf,
    python_executable: PathBuf,
    python_header: PathBuf,
    sysroot_path: PathBuf,
    cuda_executable: Option<PathBuf>,
    compiler_specs: PathBuf,
    platform_tag: String,
}

#[derive(Serialize)]
struct DebugRecipe {
    schema_version: u8,
    package: DebugPackage,
    build: DebugBuild,
    requirements: DebugRequirements,
}

#[derive(Serialize)]
struct DebugPackage {
    name: String,
    version: String,
}

#[derive(Serialize)]
struct DebugBuild {
    script: Vec<String>,
}

#[derive(Serialize)]
struct DebugRequirements {
    build: Vec<String>,
    host: Vec<String>,
}

/// Provision or reuse a conda-forge compiler environment for one compatibility
/// tuple. `cuda_version = None` omits NVCC; `Some("")` requests an unpinned
/// NVCC, and numeric `Some("12.9")` requests that CUDA release line.
pub(crate) async fn provision(
    target_floor: (u32, u32),
    python: &str,
    cuda_version: Option<&str>,
) -> Result<HermeticBuildEnvironment> {
    let python_minor = crate::pypi::normalized_python_minor(python)?.version();
    let cuda_version = normalize_cuda_version(cuda_version)?;
    let request = ProvisionRequest {
        target_floor,
        python_minor,
        cuda_version,
    };
    let cache_dir = cache_directory(&request)?;
    let _lock = crate::source_build::acquire_artifact_cache_lock(&cache_dir).await?;
    let marker_path = cache_dir.join(COMPLETION_MARKER);

    match std::fs::symlink_metadata(&marker_path) {
        Ok(metadata) => {
            if !metadata.is_file() || metadata.file_type().is_symlink() {
                bail!(
                    "hermetic build environment completion marker is not a regular file: {}",
                    marker_path.display()
                );
            }
            let marker: CompletionMarker = serde_json::from_slice(
                &std::fs::read(&marker_path)
                    .with_context(|| format!("reading {}", marker_path.display()))?,
            )
            .with_context(|| format!("parsing {}", marker_path.display()))?;
            return validate_marker(&cache_dir, &request, &marker);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("stating {}", marker_path.display()));
        }
    }

    // Markerless entries are interrupted setup attempts. The exclusive tuple
    // lock proves no cooperating setup still owns this leaf, so it is safe to
    // remove before provisioning directly at the final absolute path.
    remove_incomplete_cache(&cache_dir).await?;
    tokio::fs::create_dir_all(&cache_dir)
        .await
        .with_context(|| format!("creating hermetic cache {}", cache_dir.display()))?;

    let result = provision_uncached(&cache_dir, &request).await;
    match result {
        Ok(marker) => {
            write_completion_marker(&marker_path, &marker).await?;
            validate_marker(&cache_dir, &request, &marker)
        }
        Err(error) => {
            if let Err(cleanup) = remove_incomplete_cache(&cache_dir).await {
                return Err(error.context(format!(
                    "also failed to remove incomplete hermetic cache {}: {cleanup:#}",
                    cache_dir.display()
                )));
            }
            Err(error)
        }
    }
}

/// Enforce the selected PEP 600 policy and repair external ELF dependencies
/// before Retread emits its exact sysroot-derived tag. Auditwheel performs the
/// symbol-version/DT_NEEDED policy check that a filename-only validator cannot
/// provide (notably for C++/libstdc++), while patchelf removes compiler-prefix
/// RPATHs and bundles repairable non-policy libraries.
pub(crate) async fn repair_native_wheel(
    environment: &HermeticBuildEnvironment,
    wheel: &Path,
    private_build_dir: &Path,
) -> Result<PathBuf> {
    let repair_dir = private_build_dir.join("auditwheel-repair");
    if std::fs::symlink_metadata(&repair_dir).is_ok() {
        crate::source_build::remove_owned_cache_entry(&repair_dir)?;
    }
    std::fs::create_dir(&repair_dir)
        .with_context(|| format!("creating auditwheel output {}", repair_dir.display()))?;

    let script = r#"
unset PYTHON CONDA_BUILD_SYSROOT CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH PKG_CONFIG_PATH CMAKE_PREFIX_PATH CUDACXX CUDA_PATH CUDA_HOME NVCC CUDAHOSTCXX CUDAFLAGS NVCCFLAGS
source "$1" >/dev/null 2>&1 || true
set -euo pipefail
test "$(readlink -f "${PYTHON:-/missing}")" = "$2"
test "$(readlink -f "${CONDA_BUILD_SYSROOT:-/missing}")" = "$3"
exec "$PYTHON" -m auditwheel repair --only-plat --no-update-tags --plat "$4" --wheel-dir "$5" "$6"
"#;
    let mut command = Command::new("/bin/bash");
    command
        .arg("-c")
        .arg(script)
        .arg("retread-auditwheel-repair")
        .arg(environment.activation_script())
        .arg(environment.python_executable())
        .arg(environment.sysroot_path())
        .arg(environment.platform_tag())
        .arg(&repair_dir)
        .arg(wheel);
    run_captured(&mut command, "auditwheel native policy repair").await?;

    let mut repaired = Vec::new();
    for entry in std::fs::read_dir(&repair_dir)
        .with_context(|| format!("reading auditwheel output {}", repair_dir.display()))?
    {
        let entry = entry?;
        let metadata = entry
            .file_type()
            .with_context(|| format!("stating auditwheel output {}", entry.path().display()))?;
        if metadata.is_file()
            && entry
                .path()
                .extension()
                .is_some_and(|suffix| suffix == "whl")
        {
            repaired.push(entry.path());
        }
    }
    if repaired.len() != 1 {
        bail!(
            "auditwheel produced {} wheel archives in {}, expected exactly one",
            repaired.len(),
            repair_dir.display()
        );
    }
    let repaired = repaired.pop().expect("length was checked");
    strip_unsafe_native_rpaths(environment, &repaired, private_build_dir).await?;
    Ok(repaired)
}

async fn strip_unsafe_native_rpaths(
    environment: &HermeticBuildEnvironment,
    wheel: &Path,
    private_build_dir: &Path,
) -> Result<()> {
    let wheel_for_read = wheel.to_path_buf();
    let elf_members = tokio::task::spawn_blocking(move || -> Result<Vec<(String, Vec<u8>)>> {
        let file = std::fs::File::open(&wheel_for_read)
            .with_context(|| format!("opening repaired wheel {}", wheel_for_read.display()))?;
        let mut archive = zip::ZipArchive::new(file)
            .with_context(|| format!("reading repaired wheel {}", wheel_for_read.display()))?;
        let mut members = Vec::new();
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index)?;
            if entry.is_dir() {
                continue;
            }
            let mut bytes = Vec::with_capacity(usize::try_from(entry.size()).unwrap_or(0));
            entry.read_to_end(&mut bytes)?;
            if elf_has_dynamic_program_header(&bytes) {
                members.push((entry.name().replace('\\', "/"), bytes));
            }
        }
        Ok(members)
    })
    .await
    .context("repaired wheel ELF discovery task panicked")??;
    if elf_members.is_empty() {
        // ET_REL objects prove compilation for classification purposes but
        // cannot contain DT_RPATH/DT_RUNPATH and patchelf correctly refuses
        // them. Auditwheel has already applied symbol/needed policy to every
        // loadable object it found, so there is nothing to scrub here.
        return Ok(());
    }

    let strip_dir = private_build_dir.join("rpath-strip");
    if std::fs::symlink_metadata(&strip_dir).is_ok() {
        crate::source_build::remove_owned_cache_entry(&strip_dir)?;
    }
    std::fs::create_dir(&strip_dir)
        .with_context(|| format!("creating RPATH strip directory {}", strip_dir.display()))?;
    let mut extracted = Vec::with_capacity(elf_members.len());
    let mut extracted_args = Vec::with_capacity(elf_members.len() * 2);
    for (index, (name, bytes)) in elf_members.iter().enumerate() {
        let path = strip_dir.join(format!("native-{index}.elf"));
        std::fs::write(&path, bytes)
            .with_context(|| format!("writing extracted ELF `{name}` to {}", path.display()))?;
        let parent_depth = Path::new(name)
            .parent()
            .map_or(0, |parent| parent.components().count());
        extracted_args.push(path.clone().into_os_string());
        extracted_args.push(parent_depth.to_string().into());
        extracted.push(path);
    }

    let script = r#"
unset PYTHON CONDA_BUILD_SYSROOT CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH PKG_CONFIG_PATH CMAKE_PREFIX_PATH CUDACXX CUDA_PATH CUDA_HOME NVCC CUDAHOSTCXX CUDAFLAGS NVCCFLAGS
source "$1" >/dev/null 2>&1 || true
set -euo pipefail
test "$(readlink -f "${PYTHON:-/missing}")" = "$2"
test "$(readlink -f "${CONDA_BUILD_SYSROOT:-/missing}")" = "$3"
shift 3
while test "$#" -gt 0; do
  binary=$1
  parent_depth=$2
  shift 2
  rpath=$(patchelf --print-rpath "$binary")
  safe=
  old_ifs=$IFS
  IFS=:
  for component in $rpath; do
    relative=
    case "$component" in
      '$ORIGIN') relative= ;;
      '$ORIGIN/'*) relative=${component#'$ORIGIN/'} ;;
      '${ORIGIN}') relative= ;;
      '${ORIGIN}/'*) relative=${component#'${ORIGIN}/'} ;;
      *) continue ;;
    esac
    level=$parent_depth
    valid=1
    component_ifs=$IFS
    IFS=/
    for segment in $relative; do
      case "$segment" in
        ''|.) ;;
        ..)
          level=$((level - 1))
          if test "$level" -lt 0; then valid=0; break; fi
          ;;
        *) level=$((level + 1)) ;;
      esac
    done
    IFS=$component_ifs
    if test "$valid" = 1; then
      safe="${safe}${safe:+:}${component}"
    fi
  done
  IFS=$old_ifs
  if test -n "$safe"; then
    patchelf --set-rpath "$safe" "$binary"
  else
    patchelf --remove-rpath "$binary"
  fi
  test "$(patchelf --print-rpath "$binary")" = "$safe"
done
"#;
    let mut command = Command::new("/bin/bash");
    command
        .arg("-c")
        .arg(script)
        .arg("retread-rpath-strip")
        .arg(environment.activation_script())
        .arg(environment.python_executable())
        .arg(environment.sysroot_path())
        .args(&extracted_args);
    run_captured(&mut command, "removing unsafe native-wheel RPATHs").await?;

    let replacements = elf_members
        .into_iter()
        .zip(extracted)
        .map(|((name, _), path)| {
            std::fs::read(&path)
                .with_context(|| format!("reading RPATH-scrubbed ELF {}", path.display()))
                .map(|bytes| (name, bytes))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let wheel = wheel.to_path_buf();
    tokio::task::spawn_blocking(move || {
        crate::wheel_rewrite::replace_wheel_payloads(&wheel, &replacements)
    })
    .await
    .context("RPATH-scrubbed wheel repack task panicked")??;
    Ok(())
}

fn elf_has_dynamic_program_header(bytes: &[u8]) -> bool {
    if bytes.len() < 16 || !bytes.starts_with(b"\x7fELF") {
        return false;
    }
    let little_endian = match bytes[5] {
        1 => true,
        2 => false,
        _ => return false,
    };
    let (program_offset, entry_size_offset, entry_count_offset) = match bytes[4] {
        1 => (read_elf_uint(bytes, 28, 4, little_endian), 42, 44),
        2 => (read_elf_uint(bytes, 32, 8, little_endian), 54, 56),
        _ => return false,
    };
    let (Some(program_offset), Some(entry_size), Some(entry_count)) = (
        program_offset,
        read_elf_uint(bytes, entry_size_offset, 2, little_endian),
        read_elf_uint(bytes, entry_count_offset, 2, little_endian),
    ) else {
        return false;
    };
    let Ok(program_offset) = usize::try_from(program_offset) else {
        return false;
    };
    let Ok(entry_size) = usize::try_from(entry_size) else {
        return false;
    };
    let Ok(entry_count) = usize::try_from(entry_count) else {
        return false;
    };
    if entry_size < 4 {
        return false;
    }
    (0..entry_count).any(|index| {
        let Some(offset) = index
            .checked_mul(entry_size)
            .and_then(|offset| program_offset.checked_add(offset))
        else {
            return false;
        };
        read_elf_uint(bytes, offset, 4, little_endian) == Some(2) // PT_DYNAMIC
    })
}

fn read_elf_uint(bytes: &[u8], offset: usize, width: usize, little_endian: bool) -> Option<u64> {
    let slice = bytes.get(offset..offset.checked_add(width)?)?;
    let mut value = 0u64;
    if little_endian {
        for (shift, byte) in slice.iter().enumerate() {
            value |= u64::from(*byte) << (shift * 8);
        }
    } else {
        for byte in slice {
            value = (value << 8) | u64::from(*byte);
        }
    }
    Some(value)
}

async fn provision_uncached(
    cache_dir: &Path,
    request: &ProvisionRequest,
) -> Result<CompletionMarker> {
    ensure_rattler_build_version().await?;
    let solved = crate::conda_solve::solve_hermetic_build_environment(
        request.target_floor,
        &request.python_minor,
        request.cuda_version.as_deref(),
    )
    .await
    .map_err(|reasons| anyhow!(reasons.join("; ")))?;
    let root_versions = root_versions(&solved, request.cuda_version.is_some())?;
    let root_builds = root_builds(&solved, &root_versions)?;
    let recipe = render_debug_recipe(&root_versions, &root_builds, request.cuda_version.is_some())?;

    let recipe_dir = cache_dir.join("recipe");
    let recipe_path = recipe_dir.join("recipe.yaml");
    let output_dir = cache_dir.join("rattler-output");
    tokio::fs::create_dir_all(&recipe_dir)
        .await
        .with_context(|| format!("creating recipe directory {}", recipe_dir.display()))?;
    tokio::fs::create_dir_all(&output_dir)
        .await
        .with_context(|| format!("creating rattler output {}", output_dir.display()))?;
    tokio::fs::write(&recipe_path, recipe)
        .await
        .with_context(|| format!("writing {}", recipe_path.display()))?;

    let _build_permit = crate::concurrency::acquire_build_permit().await;
    let mut setup = Command::new("rattler-build");
    setup
        .arg("debug")
        .arg("setup")
        .arg("--no-config")
        .arg("--log-style")
        .arg("plain")
        .arg("--color")
        .arg("never")
        .arg("--recipe")
        .arg(&recipe_path)
        .arg("--channel")
        .arg("https://prefix.dev/conda-forge")
        .arg("--target-platform")
        .arg("linux-64")
        .arg("--host-platform")
        .arg("linux-64")
        .arg("--build-platform")
        .arg("linux-64")
        .arg("--output-dir")
        .arg(&output_dir)
        // Keep rattler-build's debug log tuple-local. `debug workdir` reads
        // that log, so sharing the backend's process cwd would race two
        // concurrently provisioned floor/Python/CUDA tuples.
        .current_dir(cache_dir);
    run_captured(&mut setup, "rattler-build debug setup").await?;
    drop(_build_permit);

    let mut locate = Command::new("rattler-build");
    locate
        .arg("debug")
        .arg("workdir")
        .arg("--no-config")
        .arg("--log-style")
        .arg("plain")
        .arg("--color")
        .arg("never")
        .arg("--output-dir")
        .arg(&output_dir)
        .current_dir(cache_dir);
    let output = run_captured(&mut locate, "rattler-build debug workdir").await?;
    let stdout = String::from_utf8(output.stdout)
        .context("rattler-build debug workdir emitted non-UTF-8 output")?;
    let work_dir = stdout
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| Path::new(line).is_absolute())
        .map(PathBuf::from)
        .ok_or_else(|| {
            anyhow!(
                "rattler-build debug workdir did not report an absolute work directory: {stdout:?}"
            )
        })?;
    ensure_cache_descendant(cache_dir, &work_dir, "rattler work directory")?;
    if !work_dir.is_dir() {
        bail!(
            "rattler-build debug work directory does not exist: {}",
            work_dir.display()
        );
    }
    let work_dir = std::fs::canonicalize(&work_dir).with_context(|| {
        format!(
            "canonicalizing rattler work directory {}",
            work_dir.display()
        )
    })?;

    let activation_script = work_dir.join("build_env.sh");
    let activation_metadata = std::fs::symlink_metadata(&activation_script)
        .with_context(|| format!("stating {}", activation_script.display()))?;
    if !activation_metadata.is_file() || activation_metadata.file_type().is_symlink() {
        bail!(
            "rattler-build activation script is not a regular file: {}",
            activation_script.display()
        );
    }
    let (
        python_executable,
        python_header,
        sysroot_path,
        discovered_cuda_executable,
        compiler_specs,
    ) = discover_activated_python(&activation_script).await?;
    let cuda_executable = request
        .cuda_version
        .as_ref()
        .and(discovered_cuda_executable);
    ensure_cache_descendant(cache_dir, &python_executable, "environment Python")?;
    ensure_cache_descendant(cache_dir, &python_header, "environment Python header")?;
    ensure_cache_descendant(cache_dir, &sysroot_path, "compiler sysroot")?;
    if !python_executable.is_file() {
        bail!(
            "hermetic environment Python is missing: {}",
            python_executable.display()
        );
    }
    if !python_header.is_file() {
        bail!(
            "hermetic environment Python.h is missing: {}",
            python_header.display()
        );
    }
    if !sysroot_path.is_dir() {
        bail!(
            "hermetic compiler sysroot is missing: {}",
            sysroot_path.display()
        );
    }
    if request.cuda_version.is_some() && cuda_executable.is_none() {
        bail!("cuda-nvcc_linux-64 activation did not expose a tuple-local nvcc executable");
    }
    // Store resolved regular files in the marker. Conda commonly exposes
    // `bin/python` as an in-prefix symlink; resolving it once lets cache hits
    // reject all marker-path symlinks without rejecting a valid fresh prefix.
    let python_executable = std::fs::canonicalize(&python_executable).with_context(|| {
        format!(
            "canonicalizing hermetic environment Python {}",
            python_executable.display()
        )
    })?;
    let python_header = std::fs::canonicalize(&python_header).with_context(|| {
        format!(
            "canonicalizing hermetic environment Python.h {}",
            python_header.display()
        )
    })?;
    let sysroot_path = std::fs::canonicalize(&sysroot_path).with_context(|| {
        format!(
            "canonicalizing hermetic compiler sysroot {}",
            sysroot_path.display()
        )
    })?;
    let cuda_executable = cuda_executable
        .map(|path| {
            ensure_cache_descendant(cache_dir, &path, "CUDA compiler")?;
            std::fs::canonicalize(&path)
                .with_context(|| format!("canonicalizing CUDA compiler {}", path.display()))
        })
        .transpose()?;
    ensure_cache_descendant(cache_dir, &compiler_specs, "compiler specs")?;
    let compiler_specs = std::fs::canonicalize(&compiler_specs)
        .with_context(|| format!("canonicalizing compiler specs {}", compiler_specs.display()))?;
    sanitize_compiler_specs(&compiler_specs)?;

    let selected_sysroot = solved.sysroot.glibc_floor;
    let platform_tag = platform_tag(selected_sysroot);
    Ok(CompletionMarker {
        schema: CACHE_SCHEMA.to_string(),
        target_floor: request.target_floor,
        python_minor: request.python_minor.clone(),
        cuda_version: request.cuda_version.clone(),
        selected_sysroot_version: solved.sysroot.conda_version,
        selected_sysroot,
        root_versions,
        work_dir,
        activation_script,
        python_executable,
        python_header,
        sysroot_path,
        cuda_executable,
        compiler_specs,
        platform_tag,
    })
}

fn root_versions(
    solved: &crate::conda_solve::HermeticBuildSolve,
    cuda: bool,
) -> Result<BTreeMap<String, String>> {
    let mut names = vec![
        "gcc_linux-64",
        "gxx_linux-64",
        "sysroot_linux-64",
        "python",
        "auditwheel",
        "patchelf",
        "cmake",
        "make",
        "ninja",
    ];
    if cuda {
        names.push("cuda-nvcc_linux-64");
    }
    let mut versions = BTreeMap::new();
    for name in names {
        let record = solved
            .records
            .iter()
            .find(|record| record.package_record.name.as_normalized() == name)
            .ok_or_else(|| anyhow!("hermetic compiler solve omitted required root `{name}`"))?;
        versions.insert(
            name.to_string(),
            record.package_record.version.as_str().to_string(),
        );
    }
    if versions.get("sysroot_linux-64") != Some(&solved.sysroot.conda_version) {
        bail!(
            "hermetic compiler solve did not retain exact sysroot_linux-64 =={}",
            solved.sysroot.conda_version
        );
    }
    Ok(versions)
}

fn root_builds(
    solved: &crate::conda_solve::HermeticBuildSolve,
    root_versions: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>> {
    root_versions
        .keys()
        .map(|name| {
            let record = solved
                .records
                .iter()
                .find(|record| record.package_record.name.as_normalized() == name)
                .ok_or_else(|| anyhow!("hermetic compiler solve omitted required root `{name}`"))?;
            Ok((name.clone(), record.package_record.build.clone()))
        })
        .collect()
}

fn render_debug_recipe(
    root_versions: &BTreeMap<String, String>,
    root_builds: &BTreeMap<String, String>,
    cuda: bool,
) -> Result<Vec<u8>> {
    let exact = |name: &str| -> Result<String> {
        let version = root_versions
            .get(name)
            .ok_or_else(|| anyhow!("missing solved version for `{name}`"))?;
        let build = root_builds
            .get(name)
            .ok_or_else(|| anyhow!("missing solved build string for `{name}`"))?;
        Ok(format!("{name} =={version} {build}"))
    };
    let mut build = vec![
        exact("gcc_linux-64")?,
        exact("gxx_linux-64")?,
        exact("sysroot_linux-64")?,
        exact("patchelf")?,
        exact("cmake")?,
        exact("make")?,
        exact("ninja")?,
    ];
    if cuda {
        build.push(exact("cuda-nvcc_linux-64")?);
    }
    let recipe = DebugRecipe {
        schema_version: 1,
        package: DebugPackage {
            name: "retread-hermetic-build-environment".to_string(),
            version: "1.0.0".to_string(),
        },
        build: DebugBuild {
            // `debug setup` stops before executing this; a valid script is
            // still required by the recipe schema.
            script: vec!["true".to_string()],
        },
        requirements: DebugRequirements {
            build,
            // Python belongs in host, not build. rattler-build's generated
            // `$PYTHON` points to the host prefix and compiler activation adds
            // that prefix's include directory, giving PEP 517 the matching
            // interpreter and Python.h.
            host: vec![exact("python")?, exact("auditwheel")?],
        },
    };
    serde_yaml::to_string(&recipe)
        .map(String::into_bytes)
        .context("serializing hermetic rattler-build debug recipe")
}

fn validate_marker(
    cache_dir: &Path,
    request: &ProvisionRequest,
    marker: &CompletionMarker,
) -> Result<HermeticBuildEnvironment> {
    if marker.schema != CACHE_SCHEMA
        || marker.target_floor != request.target_floor
        || marker.python_minor != request.python_minor
        || marker.cuda_version != request.cuda_version
    {
        bail!(
            "hermetic build environment marker does not match cache tuple at {}",
            cache_dir.display()
        );
    }
    let parsed =
        crate::glibc::parse_glibc_version(&marker.selected_sysroot_version).ok_or_else(|| {
            anyhow!(
                "invalid cached sysroot_linux-64 version `{}`",
                marker.selected_sysroot_version
            )
        })?;
    if parsed != marker.selected_sysroot || parsed > request.target_floor {
        bail!(
            "cached sysroot_linux-64 {} is incompatible with target glibc floor {}",
            marker.selected_sysroot_version,
            crate::glibc::format_glibc(request.target_floor)
        );
    }
    let expected_tag = platform_tag(marker.selected_sysroot);
    if marker.platform_tag != expected_tag {
        bail!(
            "cached hermetic platform tag `{}` does not match sysroot_linux-64 {} (`{expected_tag}`)",
            marker.platform_tag,
            marker.selected_sysroot_version
        );
    }
    validate_cached_path(
        cache_dir,
        &marker.work_dir,
        "rattler work directory",
        CachedPathKind::Directory,
    )?;
    match (&request.cuda_version, &marker.cuda_executable) {
        (Some(_), Some(path)) => {
            validate_cached_path(cache_dir, path, "CUDA compiler", CachedPathKind::File)?;
        }
        (Some(_), None) => {
            bail!("completed CUDA hermetic marker omits its tuple-local compiler");
        }
        (None, Some(_)) => {
            bail!("non-CUDA hermetic marker unexpectedly records a CUDA compiler");
        }
        (None, None) => {}
    }
    validate_cached_path(
        cache_dir,
        &marker.compiler_specs,
        "compiler specs",
        CachedPathKind::File,
    )?;
    validate_sanitized_compiler_specs(&marker.compiler_specs)?;
    for (path, label) in [
        (&marker.activation_script, "compiler activation script"),
        (&marker.python_executable, "environment Python"),
        (&marker.python_header, "environment Python header"),
    ] {
        validate_cached_path(cache_dir, path, label, CachedPathKind::File)?;
    }
    validate_cached_path(
        cache_dir,
        &marker.sysroot_path,
        "compiler sysroot",
        CachedPathKind::Directory,
    )?;
    let mut required = vec![
        "gcc_linux-64",
        "gxx_linux-64",
        "sysroot_linux-64",
        "python",
        "auditwheel",
        "patchelf",
        "cmake",
        "make",
        "ninja",
    ];
    if request.cuda_version.is_some() {
        required.push("cuda-nvcc_linux-64");
    }
    if required
        .iter()
        .any(|name| !marker.root_versions.contains_key(*name))
    {
        bail!(
            "completed hermetic build environment marker omits a solved root at {}",
            cache_dir.display()
        );
    }
    if marker.root_versions.get("sysroot_linux-64") != Some(&marker.selected_sysroot_version) {
        bail!("completed hermetic marker's sysroot root version is inconsistent");
    }

    Ok(HermeticBuildEnvironment {
        activation_script: marker.activation_script.clone(),
        python_executable: marker.python_executable.clone(),
        sysroot_path: marker.sysroot_path.clone(),
        cuda_executable: marker.cuda_executable.clone(),
        selected_sysroot: marker.selected_sysroot,
        platform_tag: marker.platform_tag.clone(),
    })
}

fn cache_directory(request: &ProvisionRequest) -> Result<PathBuf> {
    let root = crate::courier::retread_cache_root();
    let root = if root.is_absolute() {
        root
    } else {
        std::env::current_dir()
            .context("resolving relative RETREAD_CACHE_DIR")?
            .join(root)
    };
    let cuda = match request.cuda_version.as_deref() {
        None => "none".to_string(),
        Some("") => "any".to_string(),
        Some(version) => version.replace('.', "-"),
    };
    Ok(root.join(CACHE_NAMESPACE).join(CACHE_VERSION).join(format!(
        "glibc-{}-{}__python-{}__cuda-{cuda}",
        request.target_floor.0,
        request.target_floor.1,
        request.python_minor.replace('.', "-")
    )))
}

fn normalize_cuda_version(value: Option<&str>) -> Result<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value.trim();
    if value.is_empty() || value == "*" {
        return Ok(Some(String::new()));
    }
    let components = value.split('.').collect::<Vec<_>>();
    if !(1..=3).contains(&components.len())
        || components
            .iter()
            .any(|component| component.is_empty() || !component.bytes().all(|b| b.is_ascii_digit()))
    {
        bail!(
            "invalid CUDA version `{value}` for hermetic build: expected numeric MAJOR[.MINOR[.PATCH]]"
        );
    }
    let normalized = components
        .iter()
        .map(|component| {
            component
                .parse::<u32>()
                .map(|value| value.to_string())
                .with_context(|| format!("invalid CUDA version component in `{value}`"))
        })
        .collect::<Result<Vec<_>>>()?
        .join(".");
    Ok(Some(normalized))
}

fn platform_tag(sysroot: (u32, u32)) -> String {
    // The compiler wrapper links against the selected sysroot, not the newer
    // host libc. Tagging with that exact floor is maximally informative: an
    // older selected sysroot produces a more-compatible wheel and must not be
    // mislabeled as requiring the target's newer declared floor.
    format!("manylinux_{}_{}_x86_64", sysroot.0, sysroot.1)
}

fn ensure_cache_descendant(cache_dir: &Path, path: &Path, label: &str) -> Result<()> {
    if !path.is_absolute() || !path.starts_with(cache_dir) {
        bail!(
            "{label} `{}` escapes hermetic cache `{}`",
            path.display(),
            cache_dir.display()
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum CachedPathKind {
    Directory,
    File,
}

/// Validate paths read from the completion marker before any of them are
/// executed or opened. Canonical containment prevents `..` and symlink
/// escapes; rejecting symlinks inside the tuple prevents an attacker from
/// swapping a marker-owned activation script between validation and `source`.
fn validate_cached_path(
    cache_dir: &Path,
    path: &Path,
    label: &str,
    kind: CachedPathKind,
) -> Result<()> {
    let cache_metadata = std::fs::symlink_metadata(cache_dir)
        .with_context(|| format!("stating hermetic cache {}", cache_dir.display()))?;
    if !cache_metadata.is_dir() || cache_metadata.file_type().is_symlink() {
        bail!(
            "hermetic cache tuple is not a regular directory: {}",
            cache_dir.display()
        );
    }
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        bail!(
            "{label} `{}` is not a normalized absolute path",
            path.display()
        );
    }

    let canonical_cache = std::fs::canonicalize(cache_dir)
        .with_context(|| format!("canonicalizing hermetic cache {}", cache_dir.display()))?;
    let canonical_path = std::fs::canonicalize(path)
        .with_context(|| format!("canonicalizing {label} {}", path.display()))?;
    if !canonical_path.starts_with(&canonical_cache) {
        bail!(
            "{label} `{}` escapes hermetic cache `{}`",
            path.display(),
            cache_dir.display()
        );
    }

    // Check the marker spelling too when it lies lexically below the tuple;
    // canonicalization alone would hide a symlink that still redirects to a
    // different in-cache file.
    if let Ok(relative) = path.strip_prefix(cache_dir) {
        reject_symlink_components(cache_dir, relative, label)?;
    }
    let canonical_relative = canonical_path
        .strip_prefix(&canonical_cache)
        .expect("canonical containment was checked above");
    reject_symlink_components(&canonical_cache, canonical_relative, label)?;

    let metadata = std::fs::symlink_metadata(&canonical_path)
        .with_context(|| format!("stating canonical {label} {}", canonical_path.display()))?;
    let expected_kind = match kind {
        CachedPathKind::Directory => metadata.is_dir(),
        CachedPathKind::File => metadata.is_file(),
    };
    if !expected_kind || metadata.file_type().is_symlink() {
        bail!("{label} is not a regular cached path: {}", path.display());
    }
    Ok(())
}

fn reject_symlink_components(root: &Path, relative: &Path, label: &str) -> Result<()> {
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let std::path::Component::Normal(component) = component else {
            bail!(
                "{label} contains a non-normal path component: {}",
                relative.display()
            );
        };
        current.push(component);
        let metadata = std::fs::symlink_metadata(&current)
            .with_context(|| format!("stating {label} component {}", current.display()))?;
        if metadata.file_type().is_symlink() {
            bail!(
                "{label} contains a symlink component: {}",
                current.display()
            );
        }
    }
    Ok(())
}

fn absolute_compiler_rpath_regex() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| {
        regex::Regex::new(r"-rpath(?:[=,]|\s)+(/[^\s}%]+)").expect("compiler RPATH regex is valid")
    })
}

fn sanitize_compiler_specs(path: &Path) -> Result<()> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading conda compiler specs {}", path.display()))?;
    let regex = absolute_compiler_rpath_regex();
    let mut removed = 0usize;
    let sanitized = regex.replace_all(&raw, |_captures: &regex::Captures<'_>| {
        removed += 1;
        String::new()
    });
    if removed == 0 {
        validate_sanitized_compiler_specs(path)?;
        return Ok(());
    }
    let temporary = path.with_extension("retread-sanitized.tmp");
    std::fs::write(&temporary, sanitized.as_bytes())
        .with_context(|| format!("writing sanitized compiler specs {}", temporary.display()))?;
    std::fs::rename(&temporary, path)
        .with_context(|| format!("publishing sanitized compiler specs {}", path.display()))?;
    validate_sanitized_compiler_specs(path)
}

fn validate_sanitized_compiler_specs(path: &Path) -> Result<()> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading conda compiler specs {}", path.display()))?;
    if let Some(found) = absolute_compiler_rpath_regex().find(&raw) {
        bail!(
            "conda compiler specs retain an absolute native-wheel RPATH directive `{}` in {}",
            found.as_str(),
            path.display()
        );
    }
    Ok(())
}

async fn discover_activated_python(
    activation_script: &Path,
) -> Result<(PathBuf, PathBuf, PathBuf, Option<PathBuf>, PathBuf)> {
    let script = r#"
unset PYTHON CONDA_BUILD_SYSROOT CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH PKG_CONFIG_PATH CMAKE_PREFIX_PATH CUDACXX CUDA_PATH CUDA_HOME NVCC CUDAHOSTCXX CUDAFLAGS NVCCFLAGS
source "$1" >/dev/null 2>&1 || true
set -euo pipefail
test -n "${PYTHON:-}"
test -n "${CC:-}"
test -n "${CXX:-}"
test -d "${CONDA_BUILD_SYSROOT:-}"
printf '%s\n' "$PYTHON"
"$PYTHON" -c 'import pathlib, sysconfig; header = pathlib.Path(sysconfig.get_path("include")) / "Python.h"; assert header.is_file(), header; print(header)'
printf '%s\n' "$CONDA_BUILD_SYSROOT"
cuda_executable=$(command -v "${CUDACXX:-${NVCC:-nvcc}}" || true)
printf '%s\n' "${cuda_executable:--}"
"$CC" -print-file-name=specs
"#;
    let mut command = Command::new("bash");
    command
        .arg("-c")
        .arg(script)
        .arg("retread-hermetic-python-check")
        .arg(activation_script);
    let output = run_captured(&mut command, "validating hermetic Python and headers").await?;
    let stdout = String::from_utf8(output.stdout)
        .context("hermetic Python validation emitted non-UTF-8 output")?;
    let mut lines = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let python = lines
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("compiler activation did not define `$PYTHON`"))?;
    let header = lines
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("environment Python did not report Python.h"))?;
    let sysroot = lines
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("compiler activation did not define `$CONDA_BUILD_SYSROOT`"))?;
    let cuda = lines.next().filter(|line| *line != "-").map(PathBuf::from);
    let compiler_specs = lines
        .next()
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .ok_or_else(|| anyhow!("conda compiler did not report an absolute GCC specs path"))?;
    Ok((python, header, sysroot, cuda, compiler_specs))
}

async fn ensure_rattler_build_version() -> Result<()> {
    let mut command = Command::new("rattler-build");
    command.arg("--version");
    let output = run_captured(&mut command, "checking rattler-build version").await?;
    let rendered = format!(
        "{} {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let version = parse_rattler_build_version(&rendered).ok_or_else(|| {
        anyhow!(
            "could not parse rattler-build version from `{}`",
            rendered.trim()
        )
    })?;
    if version < MIN_RATTLER_BUILD_VERSION {
        bail!(
            "hermetic builds require rattler-build >=0.70.0 (`debug setup`); found {}.{}.{}",
            version.0,
            version.1,
            version.2
        );
    }
    Ok(())
}

fn parse_rattler_build_version(rendered: &str) -> Option<(u64, u64, u64)> {
    rendered.split_whitespace().find_map(|token| {
        let numeric = token.split_once('-').map_or(token, |(head, _)| head);
        let mut components = numeric.split('.');
        let major = components.next()?.parse().ok()?;
        let minor = components.next()?.parse().ok()?;
        let patch = components.next()?.parse().ok()?;
        Some((major, minor, patch))
    })
}

async fn run_captured(command: &mut Command, label: &str) -> Result<std::process::Output> {
    crate::fasttmp::apply_backend_env(command);
    #[cfg(unix)]
    command.process_group(0);
    let child = command
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {label}"))?;
    #[cfg(unix)]
    let mut process_group = crate::source_build::UnixProcessGroupGuard::new(
        child
            .id()
            .context("spawned hermetic-build child has no operating-system pid")?,
        label,
    )?;
    let output = child
        .wait_with_output()
        .await
        .with_context(|| format!("waiting for {label}"))?;
    #[cfg(unix)]
    process_group.disarm();
    if !output.status.success() {
        let stdout = output_snippet(&output.stdout);
        let stderr = output_snippet(&output.stderr);
        bail!(
            "{label} failed with status {}: stderr: {stderr}; stdout: {stdout}",
            output.status
        );
    }
    Ok(output)
}

fn output_snippet(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .chars()
        .rev()
        .take(4000)
        .collect::<String>()
        .chars()
        .rev()
        .collect()
}

async fn remove_incomplete_cache(cache_dir: &Path) -> Result<()> {
    let cache_dir = cache_dir.to_path_buf();
    tokio::task::spawn_blocking(move || crate::source_build::remove_owned_cache_entry(&cache_dir))
        .await
        .context("incomplete hermetic cache cleanup task panicked")?
}

async fn write_completion_marker(path: &Path, marker: &CompletionMarker) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(marker)
        .context("serializing hermetic environment completion marker")?;
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let parent = path
            .parent()
            .ok_or_else(|| anyhow!("completion marker has no parent: {}", path.display()))?;
        let temporary = parent.join(".complete.json.tmp");
        let mut file = std::fs::File::create(&temporary)
            .with_context(|| format!("creating {}", temporary.display()))?;
        file.write_all(&bytes)
            .with_context(|| format!("writing {}", temporary.display()))?;
        file.sync_all()
            .with_context(|| format!("syncing {}", temporary.display()))?;
        drop(file);
        std::fs::rename(&temporary, &path)
            .with_context(|| format!("publishing completion marker {}", path.display()))?;
        std::fs::File::open(parent)
            .and_then(|directory| directory.sync_all())
            .with_context(|| format!("syncing completion marker directory {}", parent.display()))
    })
    .await
    .context("completion marker write task panicked")?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_cache_component_is_numeric_and_normalized() {
        assert_eq!(normalize_cuda_version(None).unwrap(), None);
        assert_eq!(
            normalize_cuda_version(Some("*")).unwrap(),
            Some(String::new())
        );
        assert_eq!(
            normalize_cuda_version(Some("012.09")).unwrap(),
            Some("12.9".into())
        );
        assert!(normalize_cuda_version(Some(">=12")).is_err());
    }

    #[test]
    fn debug_recipe_keeps_python_in_host_and_toolchain_in_build() {
        let roots = BTreeMap::from([
            ("auditwheel".to_string(), "6.7.0".to_string()),
            ("cmake".to_string(), "3.31.6".to_string()),
            ("gcc_linux-64".to_string(), "14.3.0".to_string()),
            ("gxx_linux-64".to_string(), "14.3.0".to_string()),
            ("make".to_string(), "4.4.1".to_string()),
            ("ninja".to_string(), "1.13.1".to_string()),
            ("patchelf".to_string(), "0.18.0".to_string()),
            ("python".to_string(), "3.11.15".to_string()),
            ("sysroot_linux-64".to_string(), "2.28".to_string()),
        ]);
        let builds = roots
            .keys()
            .map(|name| (name.clone(), "h123456_0".to_string()))
            .collect();
        let rendered =
            String::from_utf8(render_debug_recipe(&roots, &builds, false).unwrap()).unwrap();
        let yaml: serde_yaml::Value = serde_yaml::from_str(&rendered).unwrap();
        let build = yaml["requirements"]["build"].as_sequence().unwrap();
        let host = yaml["requirements"]["host"].as_sequence().unwrap();
        assert!(
            build
                .iter()
                .any(|value| value.as_str() == Some("gcc_linux-64 ==14.3.0 h123456_0"))
        );
        assert!(
            build
                .iter()
                .any(|value| value.as_str() == Some("sysroot_linux-64 ==2.28 h123456_0"))
        );
        assert_eq!(host[0].as_str(), Some("python ==3.11.15 h123456_0"));
        assert!(
            host.iter()
                .any(|value| value.as_str() == Some("auditwheel ==6.7.0 h123456_0"))
        );
        assert!(
            build
                .iter()
                .any(|value| value.as_str() == Some("cmake ==3.31.6 h123456_0"))
        );
        assert!(
            build
                .iter()
                .any(|value| value.as_str() == Some("ninja ==1.13.1 h123456_0"))
        );
        assert!(!build.iter().any(|value| {
            value
                .as_str()
                .is_some_and(|value| value.starts_with("python "))
        }));

        let mut cuda_roots = roots.clone();
        cuda_roots.insert("cuda-nvcc_linux-64".to_string(), "12.9.1".to_string());
        let mut cuda_builds = builds;
        cuda_builds.insert("cuda-nvcc_linux-64".to_string(), "h123456_0".to_string());
        let cuda_rendered =
            String::from_utf8(render_debug_recipe(&cuda_roots, &cuda_builds, true).unwrap())
                .unwrap();
        let cuda_yaml: serde_yaml::Value = serde_yaml::from_str(&cuda_rendered).unwrap();
        assert!(
            cuda_yaml["requirements"]["build"]
                .as_sequence()
                .unwrap()
                .iter()
                .any(|value| { value.as_str() == Some("cuda-nvcc_linux-64 ==12.9.1 h123456_0") })
        );
    }

    #[test]
    fn exact_sysroot_drives_manylinux_tag() {
        assert_eq!(platform_tag((2, 17)), "manylinux_2_17_x86_64");
        assert_eq!(platform_tag((2, 28)), "manylinux_2_28_x86_64");
    }

    #[test]
    fn rpath_scrubber_requires_a_dynamic_program_header() {
        let mut header = [0u8; 120];
        header[..4].copy_from_slice(b"\x7fELF");
        header[4] = 2; // ELF64
        header[5] = 1;
        header[32..40].copy_from_slice(&64u64.to_le_bytes()); // e_phoff
        header[54..56].copy_from_slice(&56u16.to_le_bytes()); // e_phentsize
        header[56..58].copy_from_slice(&1u16.to_le_bytes()); // e_phnum
        header[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
        assert!(!elf_has_dynamic_program_header(&header));
        header[64..68].copy_from_slice(&2u32.to_le_bytes()); // PT_DYNAMIC
        assert!(elf_has_dynamic_program_header(&header));
        header[16..18].copy_from_slice(&1u16.to_le_bytes()); // ET_REL
        header[64..68].copy_from_slice(&0u32.to_le_bytes());
        assert!(!elf_has_dynamic_program_header(&header));
    }

    #[test]
    fn rattler_build_version_parser_requires_semver_token() {
        assert_eq!(
            parse_rattler_build_version("rattler-build 0.70.0"),
            Some((0, 70, 0))
        );
        assert_eq!(parse_rattler_build_version("rattler-build unknown"), None);
    }

    #[test]
    fn cache_leaf_is_keyed_by_floor_python_and_cuda() {
        let request = |target_floor, python_minor: &str, cuda_version: Option<&str>| {
            cache_directory(&ProvisionRequest {
                target_floor,
                python_minor: python_minor.to_string(),
                cuda_version: cuda_version.map(str::to_string),
            })
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned()
        };
        let base = request((2, 28), "3.11", None);
        assert_ne!(base, request((2, 17), "3.11", None));
        assert_ne!(base, request((2, 28), "3.12", None));
        assert_ne!(base, request((2, 28), "3.11", Some("12.9")));
    }

    #[test]
    fn compiler_specs_drop_absolute_rpath_but_keep_rpath_link() {
        let root = std::env::temp_dir().join(format!(
            "retread-hermetic-specs-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        std::fs::create_dir_all(&root).unwrap();
        let specs = root.join("specs");
        std::fs::write(
            &specs,
            "*link:\n%{!static:-rpath /tmp/cache/build/lib} -rpath-link /tmp/cache/build/lib\n",
        )
        .unwrap();
        sanitize_compiler_specs(&specs).unwrap();
        let sanitized = std::fs::read_to_string(&specs).unwrap();
        assert!(!sanitized.contains("-rpath /tmp/cache"));
        assert!(sanitized.contains("-rpath-link /tmp/cache/build/lib"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn cached_marker_paths_reject_parent_and_symlink_escapes() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!(
            "retread-hermetic-path-validation-{}",
            std::process::id()
        ));
        let cache = root.join("cache");
        let nested = cache.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        let outside = root.join("outside.sh");
        std::fs::write(&outside, b"untrusted").unwrap();

        let parent_escape = nested.join("..").join("..").join("outside.sh");
        assert!(
            validate_cached_path(&cache, &parent_escape, "test", CachedPathKind::File).is_err()
        );
        let link = cache.join("activation.sh");
        symlink(&outside, &link).unwrap();
        assert!(validate_cached_path(&cache, &link, "test", CachedPathKind::File).is_err());

        let _ = std::fs::remove_dir_all(&root);
    }
}
