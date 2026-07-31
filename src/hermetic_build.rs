//! Cached conda compiler environments for hermetic native wheel builds.
//!
//! `rattler-build debug setup` is useful here because it reuses the same
//! conda solver, prefix installer, and compiler activation that build recipes
//! use. Its generated activation scripts contain absolute prefix paths, so an
//! environment is provisioned directly in its final tuple-keyed cache path;
//! staging it elsewhere and renaming it would make the activation invalid.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::process::Command;

const CACHE_SCHEMA: &str = "retread-hermetic-build-environment-v5";
const CACHE_NAMESPACE: &str = "hermetic-build-envs";
const CACHE_VERSION: &str = "v5";
const COMPLETION_MARKER: &str = "complete.json";
const MIN_RATTLER_BUILD_VERSION: (u64, u64, u64) = (0, 70, 0);

/// A validated, immutable compiler environment ready to activate around a
/// PEP 517 build. Clones are cheap path/value copies; the underlying prefix is
/// shared read-only after its completion marker is published.
#[derive(Debug, Clone)]
pub struct HermeticBuildEnvironment {
    activation_script: PathBuf,
    build_prefix: PathBuf,
    host_prefix: PathBuf,
    python_executable: PathBuf,
    c_compiler: PathBuf,
    cxx_compiler: PathBuf,
    sysroot_path: PathBuf,
    cuda_executable: Option<PathBuf>,
    selected_sysroot: (u32, u32),
    platform_tag: String,
}

impl HermeticBuildEnvironment {
    pub fn activation_script(&self) -> &Path {
        &self.activation_script
    }

    pub fn build_prefix(&self) -> &Path {
        &self.build_prefix
    }

    pub fn host_prefix(&self) -> &Path {
        &self.host_prefix
    }

    pub fn python_executable(&self) -> &Path {
        &self.python_executable
    }

    pub fn c_compiler(&self) -> &Path {
        &self.c_compiler
    }

    pub fn cxx_compiler(&self) -> &Path {
        &self.cxx_compiler
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
    activation_script_sha256: String,
    activation_hooks: Vec<ActivationHookMarker>,
    build_prefix: PathBuf,
    host_prefix: PathBuf,
    python_executable: PathBuf,
    c_compiler: PathBuf,
    cxx_compiler: PathBuf,
    python_header: PathBuf,
    sysroot_path: PathBuf,
    cuda_executable: Option<PathBuf>,
    compiler_specs: PathBuf,
    platform_tag: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActivationHookMarker {
    path: PathBuf,
    sha256: String,
}

struct ActivatedEnvironment {
    python_executable: PathBuf,
    python_header: PathBuf,
    sysroot_path: PathBuf,
    cuda_executable: Option<PathBuf>,
    compiler_specs: PathBuf,
    c_compiler: PathBuf,
    cxx_compiler: PathBuf,
    build_prefix: PathBuf,
    host_prefix: PathBuf,
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
    validate_shell_safe_cache_path(&cache_dir)?;
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

    let result = async {
        let marker = provision_uncached(&cache_dir, &request).await?;
        let environment = validate_marker(&cache_dir, &request, &marker)?;
        write_completion_marker(&marker_path, &marker).await?;
        Ok::<_, anyhow::Error>(environment)
    }
    .await;
    match result {
        Ok(environment) => Ok(environment),
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

fn prepare_private_tool_scratch(private_build_dir: &Path, label: &str) -> Result<PathBuf> {
    let root = private_build_dir.join(format!(".retread-{label}-scratch"));
    if std::fs::symlink_metadata(&root).is_ok() {
        crate::source_build::remove_owned_cache_entry(&root)?;
    }
    std::fs::create_dir(&root)
        .with_context(|| format!("creating private {label} scratch {}", root.display()))?;
    let names = [
        "activation-tmp",
        "home",
        "runtime",
        "tmp",
        "xdg-cache",
        "xdg-config",
        "xdg-data",
    ];
    for name in names {
        let path = root.join(name);
        std::fs::create_dir(&path)
            .with_context(|| format!("creating private {label} directory {}", path.display()))?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for path in
            std::iter::once(root.clone()).chain(names.into_iter().map(|name| root.join(name)))
        {
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700))
                .with_context(|| format!("securing private {label} path {}", path.display()))?;
        }
    }
    std::fs::canonicalize(&root)
        .with_context(|| format!("canonicalizing private {label} scratch {}", root.display()))
}

/// Auditwheel discovers wheel ELFs through RECORD, not by walking every ZIP
/// member. Treat the archive inventory as a security boundary before handing
/// it to that external tool: every file must occur exactly once in both ZIP
/// and RECORD, and every ELF must match the x86_64 tag we will eventually
/// publish. CUDA tuples may additionally contain attested EM_CUDA device
/// ELFs, which are deliberately excluded from host dependency/RPATH checks.
/// The same check is repeated after repair because auditwheel can otherwise
/// ignore a foreign-architecture member while still succeeding.
#[derive(Debug, Default)]
struct NativeWheelInventory {
    host_dynamic_elfs: usize,
    cuda_device_elfs: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeElfKind {
    NotElf,
    HostX86_64,
    CudaDevice,
}

fn validate_native_wheel_archive(wheel: &Path, allow_cuda: bool) -> Result<NativeWheelInventory> {
    const MAX_RECORD_SIZE: u64 = 64 * 1024 * 1024;

    let file = std::fs::File::open(wheel)
        .with_context(|| format!("opening native wheel {}", wheel.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("reading native wheel {}", wheel.display()))?;
    // zip stores parsed entries in a name-keyed map, so duplicate central
    // directory names are collapsed before indexed access can expose them.
    // Compare the raw entry count with the map length first.
    let central_entries =
        zip_central_directory_entry_count(wheel, archive.central_directory_start())?;
    if central_entries != archive.len() {
        bail!(
            "native wheel contains duplicate or name-colliding ZIP members ({} central-directory entries, {} unique names)",
            central_entries,
            archive.len()
        );
    }
    let mut archive_names = BTreeSet::new();
    let mut file_names = BTreeSet::new();
    let mut record_name = None;
    let mut record_bytes = None;
    let mut inventory = NativeWheelInventory::default();

    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let name = entry.name().to_string();
        validate_wheel_member_name(&name, entry.is_dir())?;
        if !archive_names.insert(name.clone()) {
            bail!("native wheel contains duplicate ZIP member `{name}`");
        }
        if entry.is_dir() {
            continue;
        }
        file_names.insert(name.clone());

        if name.ends_with(".dist-info/RECORD") && name.matches('/').count() == 1 {
            if record_name.replace(name.clone()).is_some() {
                bail!("native wheel must contain exactly one root RECORD file");
            }
            if entry.size() > MAX_RECORD_SIZE {
                bail!("native wheel RECORD exceeds the {MAX_RECORD_SIZE}-byte validation limit");
            }
            let mut bytes = Vec::with_capacity(usize::try_from(entry.size()).unwrap_or(0));
            entry.read_to_end(&mut bytes)?;
            record_bytes = Some(bytes);
            continue;
        }

        let mut header = Vec::with_capacity(20);
        entry.by_ref().take(20).read_to_end(&mut header)?;
        match validate_native_elf_header(&name, &header, allow_cuda)? {
            NativeElfKind::HostX86_64 => {
                entry.read_to_end(&mut header)?;
                validate_elf_dependency_paths(&name, &header)?;
                let elf_type = read_elf_uint(&header, 16, 2, true);
                if elf_type != Some(3) || !elf_has_dynamic_program_header(&header) {
                    bail!(
                        "native wheel contains host ELF member '{name}' that is not a loadable ET_DYN object; ET_REL/static objects cannot be validated against the manylinux symbol-version policy"
                    );
                }
                inventory.host_dynamic_elfs += 1;
            }
            NativeElfKind::CudaDevice => {
                entry.read_to_end(&mut header)?;
                if header.len() < 64 {
                    bail!("native wheel contains truncated CUDA ELF member '{name}'");
                }
                inventory.cuda_device_elfs += 1;
            }
            NativeElfKind::NotElf if header.starts_with(b"!<thin>\n") => {
                bail!("native wheel contains unsupported thin archive member '{name}'");
            }
            NativeElfKind::NotElf if header.starts_with(b"!<arch>\n") => {
                bail!(
                    "native wheel contains static archive member '{name}'; archive objects cannot be validated against the manylinux symbol-version policy"
                );
            }
            NativeElfKind::NotElf => {}
        }
    }

    let record_name = record_name
        .ok_or_else(|| anyhow!("native wheel must contain exactly one root RECORD file"))?;
    let rows = parse_record_csv(
        record_bytes
            .as_deref()
            .expect("a discovered RECORD was read in the same archive pass"),
    )?;
    let mut record_names = BTreeSet::new();
    let mut record_self = None;
    for row in rows {
        let [path, hash, size] = row;
        validate_wheel_member_name(&path, false)?;
        if !record_names.insert(path.clone()) {
            bail!("native wheel RECORD contains duplicate entry `{path}`");
        }
        if path == record_name {
            record_self = Some((hash, size));
        }
    }
    let Some((record_hash, record_size)) = record_self else {
        bail!("native wheel RECORD does not inventory itself as `{record_name}`");
    };
    if !record_hash.is_empty() || !record_size.is_empty() {
        bail!("native wheel RECORD self-entry must have empty hash and size fields");
    }

    if file_names != record_names {
        let unrecorded = file_names.difference(&record_names).next();
        let missing = record_names.difference(&file_names).next();
        bail!(
            "native wheel RECORD inventory is not exact (unrecorded ZIP member: {}; missing ZIP member: {})",
            unrecorded.map_or("none", String::as_str),
            missing.map_or("none", String::as_str)
        );
    }
    if inventory.host_dynamic_elfs == 0 && inventory.cuda_device_elfs == 0 {
        bail!("native wheel contains no loadable x86_64 ELF payload");
    }
    Ok(inventory)
}

fn zip_central_directory_entry_count(wheel: &Path, start: u64) -> Result<usize> {
    const CENTRAL_FILE_HEADER: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];
    let mut file = std::fs::File::open(wheel)
        .with_context(|| format!("opening native wheel directory {}", wheel.display()))?;
    file.seek(SeekFrom::Start(start))
        .with_context(|| format!("seeking native wheel directory {}", wheel.display()))?;
    let mut count = 0usize;
    loop {
        let mut signature = [0u8; 4];
        file.read_exact(&mut signature).with_context(|| {
            format!(
                "reading native wheel central-directory signature from {}",
                wheel.display()
            )
        })?;
        if signature != CENTRAL_FILE_HEADER {
            break;
        }
        let mut fixed = [0u8; 42];
        file.read_exact(&mut fixed).with_context(|| {
            format!(
                "reading native wheel central-directory entry from {}",
                wheel.display()
            )
        })?;
        let name_len = u16::from_le_bytes([fixed[24], fixed[25]]);
        let extra_len = u16::from_le_bytes([fixed[26], fixed[27]]);
        let comment_len = u16::from_le_bytes([fixed[28], fixed[29]]);
        let variable_len = i64::from(name_len) + i64::from(extra_len) + i64::from(comment_len);
        file.seek(SeekFrom::Current(variable_len))
            .with_context(|| {
                format!(
                    "skipping native wheel central-directory entry in {}",
                    wheel.display()
                )
            })?;
        count = count
            .checked_add(1)
            .ok_or_else(|| anyhow!("native wheel central-directory entry count overflow"))?;
    }
    Ok(count)
}

fn validate_wheel_member_name(name: &str, directory: bool) -> Result<()> {
    // Native RPATH rewriting updates RECORD with the repository's existing
    // line-oriented writer. Reject names that require RFC 4180 quoting rather
    // than accepting an archive whose post-repair attestation cannot be
    // updated unambiguously.
    let normalized = if directory {
        name.strip_suffix('/').unwrap_or(name)
    } else {
        name
    };
    if normalized.is_empty()
        || normalized.starts_with('/')
        || normalized.contains('\\')
        || normalized.contains('\0')
        || normalized.contains(',')
        || normalized.contains('"')
        || normalized.chars().any(char::is_control)
        || normalized
            .split('/')
            .any(|component| component.is_empty() || matches!(component, "." | ".."))
    {
        bail!("native wheel contains non-normal member name `{name}`");
    }
    Ok(())
}

fn validate_native_elf_header(
    member: &str,
    header: &[u8],
    allow_cuda: bool,
) -> Result<NativeElfKind> {
    if !header.starts_with(b"\x7fELF") {
        return Ok(NativeElfKind::NotElf);
    }
    if header.len() < 20 {
        bail!("native wheel contains truncated ELF member `{member}`");
    }
    let class = header[4];
    let endian = header[5];
    let elf_version = header[6];
    let os_abi = header[7];
    let abi_version = header[8];
    let elf_type = u16::from_le_bytes([header[16], header[17]]);
    let machine = match endian {
        1 => u16::from_le_bytes([header[18], header[19]]),
        2 => u16::from_be_bytes([header[18], header[19]]),
        _ => bail!("native wheel ELF member `{member}` has invalid byte order {endian}"),
    };
    if class != 2 || endian != 1 || elf_version != 1 {
        bail!(
            "native wheel ELF member `{member}` is not a current 64-bit little-endian payload (class {class}, byte order {endian}, ELF version {elf_version}, machine {machine})"
        );
    }
    match machine {
        62 if matches!(os_abi, 0 | 3) && abi_version == 0 && elf_type == 3 => {
            Ok(NativeElfKind::HostX86_64)
        }
        62 => bail!(
            "native wheel x86_64 ELF member `{member}` is not a Linux/System-V ET_DYN payload (OS ABI {os_abi}, ABI version {abi_version}, ELF type {elf_type})"
        ),
        190 if allow_cuda
            && Path::new(member)
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("cubin"))
            && matches!(elf_type, 1 | 2) =>
        {
            Ok(NativeElfKind::CudaDevice)
        }
        190 if allow_cuda => bail!(
            "native wheel CUDA ELF member `{member}` is not an attested .cubin ET_REL/ET_EXEC device payload (ELF type {elf_type})"
        ),
        190 => bail!(
            "native wheel contains CUDA ELF member `{member}`, but its hermetic tuple did not provision cuda-nvcc_linux-64"
        ),
        _ => bail!(
            "native wheel ELF member `{member}` is neither x86_64 nor an enabled CUDA device payload (machine {machine})"
        ),
    }
}

fn validate_elf_dependency_paths(member: &str, bytes: &[u8]) -> Result<()> {
    // x86_64 was validated immediately before this function, so the ELF64
    // little-endian offsets below are unambiguous.
    let program_offset = read_elf_usize(bytes, 32, 8, true, member, "program-header offset")?;
    let entry_size = read_elf_usize(bytes, 54, 2, true, member, "program-header entry size")?;
    let entry_count = read_elf_usize(bytes, 56, 2, true, member, "program-header entry count")?;
    if entry_count == 0 {
        return Ok(());
    }
    if entry_size < 56 {
        bail!("native wheel ELF member '{member}' has a truncated program-header entry");
    }

    let mut loads = Vec::new();
    let mut dynamics = Vec::new();
    for index in 0..entry_count {
        let offset = index
            .checked_mul(entry_size)
            .and_then(|offset| program_offset.checked_add(offset))
            .ok_or_else(|| anyhow!("native wheel ELF member '{member}' program table overflows"))?;
        let kind = read_elf_uint(bytes, offset, 4, true).ok_or_else(|| {
            anyhow!("native wheel ELF member '{member}' has an incomplete program table")
        })?;
        let file_offset = read_elf_usize(bytes, offset + 8, 8, true, member, "segment offset")?;
        let virtual_address = read_elf_uint(bytes, offset + 16, 8, true).ok_or_else(|| {
            anyhow!("native wheel ELF member '{member}' has an incomplete virtual address")
        })?;
        let file_size = read_elf_usize(bytes, offset + 32, 8, true, member, "segment file size")?;
        file_offset
            .checked_add(file_size)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| {
                anyhow!("native wheel ELF member '{member}' has a segment outside its payload")
            })?;
        match kind {
            1 => loads.push((file_offset, virtual_address, file_size)), // PT_LOAD
            2 => dynamics.push((file_offset, file_size)),               // PT_DYNAMIC
            3 => bail!(
                "native wheel ELF member '{member}' contains PT_INTERP; wheel executables cannot be resolved hermetically"
            ),
            _ => {}
        }
    }

    for (dynamic_offset, dynamic_size) in dynamics {
        if dynamic_size % 16 != 0 {
            bail!("native wheel ELF member '{member}' has a malformed dynamic table");
        }
        let mut string_address = None;
        let mut string_size = None;
        let mut needed = Vec::new();
        for index in 0..(dynamic_size / 16) {
            let offset = dynamic_offset + index * 16;
            let tag = read_elf_uint(bytes, offset, 8, true).ok_or_else(|| {
                anyhow!("native wheel ELF member '{member}' has an incomplete dynamic table")
            })?;
            let value = read_elf_uint(bytes, offset + 8, 8, true).ok_or_else(|| {
                anyhow!("native wheel ELF member '{member}' has an incomplete dynamic value")
            })?;
            match tag {
                0 => break,                        // DT_NULL
                1 => needed.push(value),           // DT_NEEDED
                5 => string_address = Some(value), // DT_STRTAB
                10 => string_size = Some(value),   // DT_STRSZ
                // auditwheel's dependency graph follows DT_NEEDED. These
                // less common loader directives can load additional objects
                // outside that graph, so reject them before and after repair
                // instead of publishing an incompletely attested wheel.
                0x6fff_fefa => {
                    bail!("native wheel ELF member '{member}' contains unsupported DT_CONFIG")
                }
                0x6fff_fefb => {
                    bail!("native wheel ELF member '{member}' contains unsupported DT_DEPAUDIT")
                }
                0x6fff_fefc => {
                    bail!("native wheel ELF member '{member}' contains unsupported DT_AUDIT")
                }
                0x7fff_fffd => {
                    bail!("native wheel ELF member '{member}' contains unsupported DT_AUXILIARY")
                }
                0x7fff_fffe => {
                    bail!("native wheel ELF member '{member}' contains unsupported DT_USED")
                }
                0x7fff_ffff => {
                    bail!("native wheel ELF member '{member}' contains unsupported DT_FILTER")
                }
                _ => {}
            }
        }
        if needed.is_empty() {
            continue;
        }
        let string_address = string_address.ok_or_else(|| {
            anyhow!("native wheel ELF member '{member}' has DT_NEEDED without DT_STRTAB")
        })?;
        let string_size = usize::try_from(string_size.ok_or_else(|| {
            anyhow!("native wheel ELF member '{member}' has DT_NEEDED without DT_STRSZ")
        })?)
        .map_err(|_| anyhow!("native wheel ELF member '{member}' string table is too large"))?;
        let string_offset = loads
            .iter()
            .find_map(|(file_offset, virtual_address, file_size)| {
                let delta = string_address.checked_sub(*virtual_address)?;
                let delta = usize::try_from(delta).ok()?;
                (delta < *file_size)
                    .then(|| file_offset.checked_add(delta))
                    .flatten()
            })
            .ok_or_else(|| {
                anyhow!(
                    "native wheel ELF member '{member}' dynamic string table is outside PT_LOAD"
                )
            })?;
        let string_end = string_offset
            .checked_add(string_size)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| {
                anyhow!("native wheel ELF member '{member}' dynamic string table is out of bounds")
            })?;
        let strings = &bytes[string_offset..string_end];
        for needed_offset in needed {
            let needed_offset = usize::try_from(needed_offset).map_err(|_| {
                anyhow!("native wheel ELF member '{member}' DT_NEEDED offset is too large")
            })?;
            let value = strings.get(needed_offset..).ok_or_else(|| {
                anyhow!("native wheel ELF member '{member}' DT_NEEDED is outside DT_STRTAB")
            })?;
            let end = value.iter().position(|byte| *byte == 0).ok_or_else(|| {
                anyhow!("native wheel ELF member '{member}' has unterminated DT_NEEDED")
            })?;
            let dependency = std::str::from_utf8(&value[..end]).with_context(|| {
                format!("native wheel ELF member '{member}' has non-UTF-8 DT_NEEDED")
            })?;
            if dependency.contains('/') {
                bail!(
                    "native wheel ELF member '{member}' has path-valued DT_NEEDED '{dependency}'"
                );
            }
        }
    }
    Ok(())
}

fn read_elf_usize(
    bytes: &[u8],
    offset: usize,
    width: usize,
    little_endian: bool,
    member: &str,
    field: &str,
) -> Result<usize> {
    usize::try_from(
        read_elf_uint(bytes, offset, width, little_endian).ok_or_else(|| {
            anyhow!("native wheel ELF member '{member}' has an incomplete {field}")
        })?,
    )
    .map_err(|_| anyhow!("native wheel ELF member '{member}' {field} is too large"))
}

/// Parse the RFC 4180 quoting used by PEP 376 RECORD. A split-on-comma parser
/// is insufficient because wheel paths may themselves contain commas, quotes,
/// or newlines.
fn parse_record_csv(bytes: &[u8]) -> Result<Vec<[String; 3]>> {
    let raw = std::str::from_utf8(bytes).context("native wheel RECORD is not UTF-8")?;
    let bytes = raw.as_bytes();
    let mut rows = Vec::new();
    let mut fields = Vec::new();
    let mut field = Vec::new();
    let mut quoted = false;
    let mut quote_closed = false;
    let mut index = 0usize;

    let finish_field = |field: &mut Vec<u8>, fields: &mut Vec<String>| -> Result<()> {
        fields.push(
            String::from_utf8(std::mem::take(field))
                .context("native wheel RECORD field is not UTF-8")?,
        );
        Ok(())
    };
    let finish_row = |fields: &mut Vec<String>, rows: &mut Vec<[String; 3]>| -> Result<()> {
        let row = std::mem::take(fields);
        rows.push(row.try_into().map_err(|row: Vec<String>| {
            anyhow!(
                "native wheel RECORD row has {} fields, expected exactly 3",
                row.len()
            )
        })?);
        Ok(())
    };

    while index < bytes.len() {
        let byte = bytes[index];
        if quoted {
            if byte == b'"' {
                if bytes.get(index + 1) == Some(&b'"') {
                    field.push(b'"');
                    index += 2;
                    continue;
                }
                quoted = false;
                quote_closed = true;
            } else {
                field.push(byte);
            }
            index += 1;
            continue;
        }
        if quote_closed && !matches!(byte, b',' | b'\r' | b'\n') {
            bail!("native wheel RECORD has characters after a closing quote");
        }
        match byte {
            b'"' if field.is_empty() && !quote_closed => quoted = true,
            b'"' => bail!("native wheel RECORD has an unexpected quote in an unquoted field"),
            b',' => {
                finish_field(&mut field, &mut fields)?;
                quote_closed = false;
            }
            b'\n' => {
                finish_field(&mut field, &mut fields)?;
                finish_row(&mut fields, &mut rows)?;
                quote_closed = false;
            }
            b'\r' => {
                if bytes.get(index + 1) != Some(&b'\n') {
                    bail!("native wheel RECORD has a bare carriage return");
                }
                finish_field(&mut field, &mut fields)?;
                finish_row(&mut fields, &mut rows)?;
                quote_closed = false;
                index += 1;
            }
            _ => field.push(byte),
        }
        index += 1;
    }
    if quoted {
        bail!("native wheel RECORD has an unterminated quoted field");
    }
    if quote_closed || !field.is_empty() || !fields.is_empty() {
        finish_field(&mut field, &mut fields)?;
        finish_row(&mut fields, &mut rows)?;
    }
    if rows.is_empty() {
        bail!("native wheel RECORD is empty");
    }
    Ok(rows)
}

fn auditwheel_ldpaths(environment: &HermeticBuildEnvironment) -> Result<std::ffi::OsString> {
    let roots = [
        environment.build_prefix(),
        environment.host_prefix(),
        environment.sysroot_path(),
    ];
    let candidates = [
        environment.build_prefix().join("lib"),
        environment.build_prefix().join("lib64"),
        environment
            .build_prefix()
            .join("x86_64-conda-linux-gnu/lib"),
        environment
            .build_prefix()
            .join("x86_64-conda-linux-gnu/lib64"),
        environment.host_prefix().join("lib"),
        environment.host_prefix().join("lib64"),
        environment.sysroot_path().join("lib"),
        environment.sysroot_path().join("lib64"),
        environment.sysroot_path().join("usr/lib"),
        environment.sysroot_path().join("usr/lib64"),
    ];
    let mut directories = BTreeSet::new();
    for candidate in candidates {
        let metadata = match std::fs::metadata(&candidate) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("stating auditwheel library path {}", candidate.display())
                });
            }
        };
        if !metadata.is_dir() {
            bail!(
                "auditwheel tuple library path is not a directory: {}",
                candidate.display()
            );
        }
        let canonical = std::fs::canonicalize(&candidate).with_context(|| {
            format!(
                "canonicalizing auditwheel library path {}",
                candidate.display()
            )
        })?;
        if !roots.iter().any(|root| canonical.starts_with(root)) {
            bail!(
                "auditwheel library path {} escapes the solved tuple and sysroot",
                canonical.display()
            );
        }
        directories.insert(canonical);
    }
    if directories.is_empty() {
        bail!("hermetic build tuple contains no library directories for auditwheel --ldpaths");
    }
    std::env::join_paths(directories)
        .context("hermetic auditwheel library paths cannot be represented as --ldpaths")
}

const DEPENDENCY_PREFLIGHT_PYTHON: &str = r#"
import os
import sys
from pathlib import Path

from auditwheel.lddtree import LIBPYTHON_RE, ldd
from elftools.elf.elffile import ELFFile

wheel_root = Path(sys.argv[1]).resolve(strict=True)
build_root = Path(sys.argv[2]).resolve(strict=True)
host_root = Path(sys.argv[3]).resolve(strict=True)
sysroot = Path(sys.argv[4]).resolve(strict=True)
search = [Path(item).resolve(strict=True) for item in sys.argv[5].split(os.pathsep) if item]
allowed = (wheel_root, build_root, host_root, sysroot)

def contained(path):
    resolved = Path(path).resolve(strict=True)
    return resolved, any(resolved == root or root in resolved.parents for root in allowed)

for path in sorted(candidate for candidate in wheel_root.rglob("*") if candidate.is_file()):
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            continue
        stream.seek(0)
        elf = ELFFile(stream)
        for segment in elf.iter_segments():
            if segment.header.p_type == "PT_INTERP":
                raise SystemExit(f"dependency preflight rejected PT_INTERP in {path}")
            if segment.header.p_type != "PT_DYNAMIC":
                continue
            for tag in segment.iter_tags():
                if tag.entry.d_tag == "DT_NEEDED" and "/" in tag.needed:
                    raise SystemExit(
                        f"dependency preflight rejected path-valued DT_NEEDED {tag.needed!r} in {path}"
                    )

    tree = ldd(
        path,
        ldpaths={"conf": [], "env": [str(item) for item in search],
                 "interp": [], "rpath": [], "runpath": []},
    )
    for soname, library in tree.libraries.items():
        if LIBPYTHON_RE.match(soname):
            continue
        if library.realpath is None:
            raise SystemExit(f"dependency preflight could not resolve {soname!r} from {path}")
        resolved, is_allowed = contained(library.realpath)
        if not is_allowed:
            raise SystemExit(
                f"dependency preflight resolved {soname!r} outside the hermetic tuple: {resolved}"
            )
"#;

async fn preflight_native_dependencies(
    environment: &HermeticBuildEnvironment,
    wheel: &Path,
    private_build_dir: &Path,
    ldpaths: &std::ffi::OsStr,
) -> Result<()> {
    let scratch = prepare_private_tool_scratch(private_build_dir, "dependency-preflight")?;
    let wheel_root = scratch.join("wheel");
    std::fs::create_dir(&wheel_root).with_context(|| {
        format!(
            "creating dependency preflight tree {}",
            wheel_root.display()
        )
    })?;
    let file = std::fs::File::open(wheel)
        .with_context(|| format!("opening native wheel {}", wheel.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("reading native wheel {}", wheel.display()))?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().to_string();
        validate_wheel_member_name(&name, false)?;
        let mut bytes = Vec::with_capacity(20);
        entry.by_ref().take(20).read_to_end(&mut bytes)?;
        match validate_native_elf_header(&name, &bytes, environment.cuda_executable().is_some())? {
            NativeElfKind::HostX86_64 => {
                entry.read_to_end(&mut bytes)?;
            }
            NativeElfKind::CudaDevice | NativeElfKind::NotElf => continue,
        }
        let destination = wheel_root.join(&name);
        let parent = destination.parent().ok_or_else(|| {
            anyhow!(
                "dependency preflight member has no parent: {}",
                destination.display()
            )
        })?;
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "creating dependency preflight directory {}",
                parent.display()
            )
        })?;
        std::fs::write(&destination, bytes).with_context(|| {
            format!(
                "extracting dependency preflight ELF {}",
                destination.display()
            )
        })?;
    }

    let command_path = std::env::join_paths([
        environment.build_prefix().join("bin"),
        environment.host_prefix().join("bin"),
        PathBuf::from("/usr/bin"),
        PathBuf::from("/bin"),
    ])
    .context("constructing dependency preflight PATH")?;
    let mut command = Command::new(environment.python_executable());
    command
        .arg("-I")
        .arg("-c")
        .arg(DEPENDENCY_PREFLIGHT_PYTHON)
        .arg(&wheel_root)
        .arg(environment.build_prefix())
        .arg(environment.host_prefix())
        .arg(environment.sysroot_path())
        .arg(ldpaths)
        .env_clear()
        .env("PATH", command_path)
        .env("HOME", scratch.join("home"))
        .env("TMPDIR", scratch.join("tmp"))
        .env("TMP", scratch.join("tmp"))
        .env("TEMP", scratch.join("tmp"))
        .env("XDG_CACHE_HOME", scratch.join("xdg-cache"))
        .env("XDG_CONFIG_HOME", scratch.join("xdg-config"))
        .env("XDG_DATA_HOME", scratch.join("xdg-data"))
        .env("XDG_RUNTIME_DIR", scratch.join("runtime"))
        .env("PYTHONNOUSERSITE", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .current_dir(&scratch);
    run_captured_sealed(
        &mut command,
        "preflighting native-wheel dependency provenance",
    )
    .await?;
    Ok(())
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
    let allow_cuda = environment.cuda_executable().is_some();
    let inventory = validate_native_wheel_archive(wheel, allow_cuda)?;
    if inventory.host_dynamic_elfs == 0 && inventory.cuda_device_elfs != 0 {
        // Standalone cubins have no host glibc/symbol/RPATH surface for
        // auditwheel to inspect. The exact sysroot-derived tag is therefore
        // conservative; mixed host+device wheels still take the full repair.
        return Ok(wheel.to_path_buf());
    }
    // ELF DT_RPATH is searched before auditwheel's explicit --ldpaths. Scrub
    // the built archive first so a build-system-injected host path cannot
    // select a host library for grafting, then attest the rewritten RECORD.
    strip_unsafe_native_rpaths(environment, wheel, private_build_dir).await?;
    validate_native_wheel_archive(wheel, allow_cuda)?;

    let ldpaths = auditwheel_ldpaths(environment)?;
    preflight_native_dependencies(environment, wheel, private_build_dir, &ldpaths).await?;
    let scratch = prepare_private_tool_scratch(private_build_dir, "auditwheel")?;
    let repair_dir = private_build_dir.join("auditwheel-repair");
    if std::fs::symlink_metadata(&repair_dir).is_ok() {
        crate::source_build::remove_owned_cache_entry(&repair_dir)?;
    }
    std::fs::create_dir(&repair_dir)
        .with_context(|| format!("creating auditwheel output {}", repair_dir.display()))?;

    let script = r#"
set -euo pipefail
umask 077
scratch=$9
test -d "$scratch/activation-tmp"
export HOME="$scratch/home"
export TMPDIR="$scratch/tmp"
export TMP="$scratch/tmp"
export TEMP="$scratch/tmp"
export XDG_CACHE_HOME="$scratch/xdg-cache"
export XDG_CONFIG_HOME="$scratch/xdg-config"
export XDG_DATA_HOME="$scratch/xdg-data"
export XDG_RUNTIME_DIR="$scratch/runtime"
export RETREAD_ACTIVATION_TMPDIR="$scratch/activation-tmp"
export PATH=/usr/bin:/bin
unset PYTHON PYTHONHOME PYTHONPATH CONDA_BUILD_SYSROOT CC CXX AR CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LDFLAGS_LD DEBUG_CPPFLAGS DEBUG_CFLAGS DEBUG_CXXFLAGS QEMU_LD_PREFIX COMPILER_PATH GCC_EXEC_PREFIX OBJC_INCLUDE_PATH DEPENDENCIES_OUTPUT SUNPRO_DEPENDENCIES CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH LD_PRELOAD PKG_CONFIG_PATH CMAKE_PREFIX_PATH CUDACXX CUDA_PATH CUDA_HOME NVCC CUDAHOSTCXX CUDAFLAGS NVCCFLAGS NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS AUDITWHEEL_LD_LIBRARY_PATH AUDITWHEEL_ZIP_COMPRESSION_LEVEL
activation_cwd=$PWD
cd "$RETREAD_ACTIVATION_TMPDIR"
set +o pipefail
source "$1" >/dev/null 2>&1
set -o pipefail
cd "$activation_cwd"
test "$(/usr/bin/readlink -f "${PYTHON:-/missing}")" = "$2"
test "$(/usr/bin/readlink -f "${CONDA_BUILD_SYSROOT:-/missing}")" = "$3"
test "$(/usr/bin/readlink -f "${BUILD_PREFIX:-/missing}")" = "$7"
test "$(/usr/bin/readlink -f "${PREFIX:-/missing}")" = "$8"
export PATH="$7/bin:$8/bin:/usr/bin:/bin"
export HOME="$scratch/home"
export TMPDIR="$scratch/tmp"
export TMP="$scratch/tmp"
export TEMP="$scratch/tmp"
export XDG_CACHE_HOME="$scratch/xdg-cache"
export XDG_CONFIG_HOME="$scratch/xdg-config"
export XDG_DATA_HOME="$scratch/xdg-data"
export XDG_RUNTIME_DIR="$scratch/runtime"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
retread_sysroot=$CONDA_BUILD_SYSROOT
for retread_name in ${!CONDA_@}; do unset "$retread_name"; done
export CONDA_BUILD_SYSROOT="$retread_sysroot"
for retread_name in ${!PKG_@} ${!CONDA_ENV_SHLVL_@} ${!RATTLER_BUILD_@}; do unset "$retread_name"; done
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD AUDITWHEEL_LD_LIBRARY_PATH AUDITWHEEL_ZIP_COMPRESSION_LEVEL BUILD_DIR SRC_DIR RECIPE_DIR RATTLER_BUILD_PACKAGE_FILES SOURCE_DATE_EPOCH CONDA_BUILD CONDA_BUILD_STATE CONDA_BUILD_CROSS_COMPILATION CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL _CE_CONDA _CE_M CMAKE_GENERATOR CMAKE_ARGS PIP_IGNORE_INSTALLED PIP_NO_BUILD_ISOLATION
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
unset LDFLAGS_LD DEBUG_CPPFLAGS DEBUG_CFLAGS DEBUG_CXXFLAGS QEMU_LD_PREFIX COMPILER_PATH GCC_EXEC_PREFIX OBJC_INCLUDE_PATH DEPENDENCIES_OUTPUT SUNPRO_DEPENDENCIES
export USER=$(/usr/bin/id -un)
export LOGNAME="$USER"
unset PREFIX BUILD_PREFIX
cd "$scratch"
exec "$PYTHON" -I -m auditwheel repair --only-plat --no-update-tags --ldpaths "${10}" --plat "$4" --wheel-dir "$5" "$6"
"#;
    let mut command = Command::new("/bin/bash");
    command
        .env_clear()
        .arg("-p")
        .arg("-c")
        .arg(script)
        .arg("retread-auditwheel-repair")
        .arg(environment.activation_script())
        .arg(environment.python_executable())
        .arg(environment.sysroot_path())
        .arg(environment.platform_tag())
        .arg(&repair_dir)
        .arg(wheel)
        .arg(environment.build_prefix())
        .arg(environment.host_prefix())
        .arg(&scratch)
        .arg(&ldpaths);
    command
        .env_remove("BASH_ENV")
        .env_remove("ENV")
        .env_remove("LD_LIBRARY_PATH")
        .env_remove("LD_PRELOAD");
    run_captured_sealed(&mut command, "auditwheel native policy repair").await?;

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
    validate_native_wheel_archive(&repaired, allow_cuda)?;
    strip_unsafe_native_rpaths(environment, &repaired, private_build_dir).await?;
    validate_native_wheel_archive(&repaired, allow_cuda)?;
    Ok(repaired)
}

async fn strip_unsafe_native_rpaths(
    environment: &HermeticBuildEnvironment,
    wheel: &Path,
    private_build_dir: &Path,
) -> Result<()> {
    let wheel_for_read = wheel.to_path_buf();
    let allow_cuda = environment.cuda_executable().is_some();
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
            if validate_native_elf_header(entry.name(), &bytes, allow_cuda)?
                == NativeElfKind::HostX86_64
                && elf_has_dynamic_program_header(&bytes)
            {
                members.push((entry.name().replace('\\', "/"), bytes));
            }
        }
        Ok(members)
    })
    .await
    .context("repaired wheel ELF discovery task panicked")??;
    if elf_members.is_empty() {
        // A CUDA device-only payload has no host RPATH. Host ET_REL/static
        // objects were rejected at the archive boundary above.
        return Ok(());
    }

    let scratch = prepare_private_tool_scratch(private_build_dir, "rpath")?;

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
        // PEP 427 moves `.data/platlib` and `.data/purelib` contents into
        // site-packages (and the other scheme keys to their own roots). Use
        // the installed-relative depth, not the archive depth, or an RPATH
        // could climb through the two stripped `.data` components and escape
        // the installation scheme after unpacking.
        let parent_depth = installed_wheel_parent_depth(name)?;
        extracted_args.push(path.clone().into_os_string());
        extracted_args.push(parent_depth.to_string().into());
        extracted.push(path);
    }

    let script = r#"
set -euo pipefail
umask 077
scratch=$6
test -d "$scratch/activation-tmp"
export HOME="$scratch/home"
export TMPDIR="$scratch/tmp"
export TMP="$scratch/tmp"
export TEMP="$scratch/tmp"
export XDG_CACHE_HOME="$scratch/xdg-cache"
export XDG_CONFIG_HOME="$scratch/xdg-config"
export XDG_DATA_HOME="$scratch/xdg-data"
export XDG_RUNTIME_DIR="$scratch/runtime"
export RETREAD_ACTIVATION_TMPDIR="$scratch/activation-tmp"
export PATH=/usr/bin:/bin
unset PYTHON PYTHONHOME PYTHONPATH CONDA_BUILD_SYSROOT CC CXX AR CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LDFLAGS_LD DEBUG_CPPFLAGS DEBUG_CFLAGS DEBUG_CXXFLAGS QEMU_LD_PREFIX COMPILER_PATH GCC_EXEC_PREFIX OBJC_INCLUDE_PATH DEPENDENCIES_OUTPUT SUNPRO_DEPENDENCIES CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH LD_PRELOAD PKG_CONFIG_PATH CMAKE_PREFIX_PATH CUDACXX CUDA_PATH CUDA_HOME NVCC CUDAHOSTCXX CUDAFLAGS NVCCFLAGS NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS AUDITWHEEL_LD_LIBRARY_PATH AUDITWHEEL_ZIP_COMPRESSION_LEVEL
activation_cwd=$PWD
cd "$RETREAD_ACTIVATION_TMPDIR"
set +o pipefail
source "$1" >/dev/null 2>&1
set -o pipefail
cd "$activation_cwd"
test "$(/usr/bin/readlink -f "${PYTHON:-/missing}")" = "$2"
test "$(/usr/bin/readlink -f "${CONDA_BUILD_SYSROOT:-/missing}")" = "$3"
test "$(/usr/bin/readlink -f "${BUILD_PREFIX:-/missing}")" = "$4"
test "$(/usr/bin/readlink -f "${PREFIX:-/missing}")" = "$5"
export PATH="$4/bin:$5/bin:/usr/bin:/bin"
retread_sysroot=$CONDA_BUILD_SYSROOT
for retread_name in ${!CONDA_@}; do unset "$retread_name"; done
export CONDA_BUILD_SYSROOT="$retread_sysroot"
for retread_name in ${!PKG_@} ${!CONDA_ENV_SHLVL_@} ${!RATTLER_BUILD_@}; do unset "$retread_name"; done
unset LD_LIBRARY_PATH LD_PRELOAD AUDITWHEEL_LD_LIBRARY_PATH AUDITWHEEL_ZIP_COMPRESSION_LEVEL BUILD_DIR SRC_DIR RECIPE_DIR RATTLER_BUILD_PACKAGE_FILES SOURCE_DATE_EPOCH CONDA_BUILD CONDA_BUILD_STATE CONDA_BUILD_CROSS_COMPILATION CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL _CE_CONDA _CE_M CMAKE_GENERATOR CMAKE_ARGS PIP_IGNORE_INSTALLED PIP_NO_BUILD_ISOLATION
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
unset LDFLAGS_LD DEBUG_CPPFLAGS DEBUG_CFLAGS DEBUG_CXXFLAGS QEMU_LD_PREFIX COMPILER_PATH GCC_EXEC_PREFIX OBJC_INCLUDE_PATH DEPENDENCIES_OUTPUT SUNPRO_DEPENDENCIES
export USER=$(/usr/bin/id -un)
export LOGNAME="$USER"
unset PREFIX BUILD_PREFIX
patchelf_tool="$4/bin/patchelf"
test -x "$patchelf_tool"
shift 6
while test "$#" -gt 0; do
  binary=$1
  parent_depth=$2
  shift 2
  rpath=$("$patchelf_tool" --print-rpath "$binary")
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
    "$patchelf_tool" --set-rpath "$safe" "$binary"
  else
    "$patchelf_tool" --remove-rpath "$binary"
  fi
  test "$("$patchelf_tool" --print-rpath "$binary")" = "$safe"
done
"#;
    let mut command = Command::new("/bin/bash");
    command
        .env_clear()
        .arg("-p")
        .arg("-c")
        .arg(script)
        .arg("retread-rpath-strip")
        .arg(environment.activation_script())
        .arg(environment.python_executable())
        .arg(environment.sysroot_path())
        .arg(environment.build_prefix())
        .arg(environment.host_prefix())
        .arg(&scratch)
        .args(&extracted_args);
    command
        .env_remove("BASH_ENV")
        .env_remove("ENV")
        .env_remove("LD_LIBRARY_PATH")
        .env_remove("LD_PRELOAD");
    run_captured_sealed(&mut command, "removing unsafe native-wheel RPATHs").await?;

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

fn installed_wheel_parent_depth(member: &str) -> Result<usize> {
    if member.starts_with('/') || member.contains('\\') {
        bail!("native wheel contains a non-relative payload member `{member}`");
    }
    let components = member.split('/').collect::<Vec<_>>();
    if components.is_empty()
        || components
            .iter()
            .any(|component| component.is_empty() || matches!(*component, "." | ".."))
    {
        bail!("native wheel contains a non-normal payload member `{member}`");
    }
    let installed = if components[0].ends_with(".data") {
        let scheme = components.get(1).copied().ok_or_else(|| {
            anyhow!("native wheel has an incomplete .data payload member `{member}`")
        })?;
        if !matches!(
            scheme,
            "data" | "headers" | "platlib" | "purelib" | "scripts"
        ) {
            bail!("native wheel has an unknown .data scheme `{scheme}` in `{member}`");
        }
        components
            .get(2..)
            .filter(|parts| !parts.is_empty())
            .ok_or_else(|| {
                anyhow!("native wheel has an incomplete .data payload member `{member}`")
            })?
    } else {
        components.as_slice()
    };
    Ok(installed.len() - 1)
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

fn file_sha256(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn sanitize_activation_script(cache_dir: &Path, path: &Path) -> Result<String> {
    validate_cached_path(
        cache_dir,
        path,
        "compiler activation script",
        CachedPathKind::File,
    )?;
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading compiler activation script {}", path.display()))?;
    let mut path_exports = 0usize;
    let mut rendered = String::with_capacity(raw.len());
    for line in raw.lines() {
        if line.trim_start().starts_with("export PATH=") {
            path_exports += 1;
            rendered.push_str("export PATH=/usr/bin:/bin\n");
        } else {
            rendered.push_str(line);
            rendered.push('\n');
        }
    }
    if path_exports == 0 {
        bail!(
            "rattler-build activation script has no PATH boundary to sanitize: {}",
            path.display()
        );
    }
    if rendered.as_bytes() != raw.as_bytes() {
        let metadata = std::fs::symlink_metadata(path)
            .with_context(|| format!("stating compiler activation script {}", path.display()))?;
        let temporary = path.with_extension("retread-sanitized.tmp");
        std::fs::write(&temporary, rendered.as_bytes()).with_context(|| {
            format!(
                "writing sanitized compiler activation script {}",
                temporary.display()
            )
        })?;
        std::fs::set_permissions(&temporary, metadata.permissions()).with_context(|| {
            format!(
                "preserving compiler activation script mode on {}",
                temporary.display()
            )
        })?;
        std::fs::rename(&temporary, path).with_context(|| {
            format!(
                "publishing sanitized compiler activation script {}",
                path.display()
            )
        })?;
    }
    validate_sanitized_activation_script(path)?;
    file_sha256(path)
}

fn validate_sanitized_activation_script(path: &Path) -> Result<()> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading compiler activation script {}", path.display()))?;
    let mut path_exports = 0usize;
    for line in raw.lines() {
        if line.trim_start().starts_with("export PATH=") {
            path_exports += 1;
            if line.trim() != "export PATH=/usr/bin:/bin" {
                bail!(
                    "compiler activation script retains a provisioning-host PATH: {}",
                    path.display()
                );
            }
        }
    }
    if path_exports == 0 {
        bail!(
            "compiler activation script has no attested PATH boundary: {}",
            path.display()
        );
    }
    Ok(())
}

fn activation_hook_paths(cache_dir: &Path, activation_script: &Path) -> Result<Vec<PathBuf>> {
    let raw = std::fs::read_to_string(activation_script)
        .with_context(|| format!("reading {}", activation_script.display()))?;
    if raw.contains("/tmp/old-env-$$.txt") || raw.contains("/tmp/new-env-$$.txt") {
        bail!(
            "compiler activation script itself contains unsafe predictable /tmp environment dumps: {}",
            activation_script.display()
        );
    }
    let mut hooks = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if !line.contains("/etc/conda/activate.d/") {
            continue;
        }
        let candidate = line
            .strip_prefix(". ")
            .ok_or_else(|| anyhow!("unsupported compiler activation hook statement `{line}`"))?
            .trim();
        if candidate.is_empty()
            || !candidate.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'.' | b'_' | b'-' | b'+')
            })
        {
            bail!("unsupported compiler activation hook path `{candidate}`");
        }
        let path = PathBuf::from(candidate);
        validate_cached_path(
            cache_dir,
            &path,
            "compiler activation hook",
            CachedPathKind::File,
        )?;
        hooks.push(path);
    }
    hooks.sort();
    hooks.dedup();
    if hooks.is_empty() {
        bail!(
            "compiler activation script contains no tuple-local conda activation hooks: {}",
            activation_script.display()
        );
    }
    Ok(hooks)
}

fn validate_sanitized_activation_hook(path: &Path) -> Result<()> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading compiler activation hook {}", path.display()))?;
    if raw.contains("/tmp/") || raw.contains("/var/tmp/") {
        bail!(
            "compiler activation hook retains an unconfined temporary path: {}",
            path.display()
        );
    }
    if raw.lines().any(is_environment_dump_line) {
        bail!(
            "compiler activation hook still serializes the inherited environment: {}",
            path.display()
        );
    }
    Ok(())
}

fn is_environment_dump_line(line: &str) -> bool {
    let line = line.trim_start();
    if !line.contains('>') {
        return false;
    }
    line.starts_with("env ")
        || line.starts_with("printenv ")
        || line.starts_with("export -p ")
        || line.starts_with("declare -x ")
        || line.starts_with("set >")
}

fn suppress_environment_snapshot_writes(raw: &str) -> Result<String> {
    let mut sanitized = String::with_capacity(raw.len());
    for line in raw.lines() {
        if is_environment_dump_line(line) {
            let (prefix, target) = line.rsplit_once('>').ok_or_else(|| {
                anyhow!("unsupported compiler environment snapshot statement `{line}`")
            })?;
            if !prefix.trim_start().starts_with("env ")
                && !prefix.trim_start().starts_with("printenv ")
            {
                bail!("unsupported compiler environment snapshot statement `{line}`");
            }
            let target = target.trim();
            if !target.contains("RETREAD_ACTIVATION_TMPDIR") {
                bail!("compiler hook writes an environment snapshot outside private tmp: `{line}`");
            }
            let indentation = &line[..line.len() - line.trim_start().len()];
            sanitized.push_str(indentation);
            sanitized.push_str(": > ");
            sanitized.push_str(target);
            sanitized.push('\n');
        } else {
            sanitized.push_str(line);
            sanitized.push('\n');
        }
    }
    Ok(sanitized)
}

fn sanitize_activation_hooks(
    cache_dir: &Path,
    activation_script: &Path,
) -> Result<Vec<ActivationHookMarker>> {
    activation_hook_paths(cache_dir, activation_script)?
        .into_iter()
        .map(|path| {
            let raw = std::fs::read_to_string(&path)
                .with_context(|| format!("reading compiler activation hook {}", path.display()))?;
            let relocated = raw
                .replace(
                    "/tmp/old-env-$$.txt",
                    "\"${RETREAD_ACTIVATION_TMPDIR:?}/old-env-$$.txt\"",
                )
                .replace(
                    "/tmp/new-env-$$.txt",
                    "\"${RETREAD_ACTIVATION_TMPDIR:?}/new-env-$$.txt\"",
                );
            let sanitized = suppress_environment_snapshot_writes(&relocated)?;
            if sanitized != raw {
                let metadata = std::fs::symlink_metadata(&path).with_context(|| {
                    format!("stating compiler activation hook {}", path.display())
                })?;
                let temporary = path.with_extension("retread-sanitized.tmp");
                std::fs::write(&temporary, sanitized.as_bytes()).with_context(|| {
                    format!("writing sanitized activation hook {}", temporary.display())
                })?;
                std::fs::set_permissions(&temporary, metadata.permissions()).with_context(
                    || format!("preserving activation hook mode on {}", temporary.display()),
                )?;
                std::fs::rename(&temporary, &path).with_context(|| {
                    format!("publishing sanitized activation hook {}", path.display())
                })?;
            }
            validate_sanitized_activation_hook(&path)?;
            Ok(ActivationHookMarker {
                sha256: file_sha256(&path)?,
                path,
            })
        })
        .collect()
}

fn prepare_activation_tmp(path: &Path) -> Result<()> {
    if std::fs::symlink_metadata(path).is_ok() {
        crate::source_build::remove_owned_cache_entry(path)?;
    }
    std::fs::create_dir(path)
        .with_context(|| format!("creating private activation temporary {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
            .with_context(|| format!("securing private activation temporary {}", path.display()))?;
    }
    Ok(())
}

async fn provision_uncached(
    cache_dir: &Path,
    request: &ProvisionRequest,
) -> Result<CompletionMarker> {
    let rattler_scratch = prepare_private_tool_scratch(cache_dir, "rattler-setup")?;
    let rattler_build = rattler_build_executable()?;
    ensure_rattler_build_version(&rattler_build, &rattler_scratch).await?;
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
    let mut setup = Command::new(&rattler_build);
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
    configure_rattler_command_environment(&mut setup, &rattler_scratch);
    run_captured_sealed(&mut setup, "rattler-build debug setup").await?;
    drop(_build_permit);

    let mut locate = Command::new(&rattler_build);
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
    configure_rattler_command_environment(&mut locate, &rattler_scratch);
    let output = run_captured_sealed(&mut locate, "rattler-build debug workdir").await?;
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
    validate_cached_path(
        cache_dir,
        &work_dir,
        "rattler work directory",
        CachedPathKind::Directory,
    )?;

    let activation_script = work_dir.join("build_env.sh");
    // Conda-forge compiler hooks dump the complete inherited environment to
    // predictable, normally world-readable `/tmp/*-env-$$.txt` paths when
    // CONDA_BUILD=1. Retarget those diagnostics into a caller-supplied 0700
    // directory before the activation script is ever sourced, and attest the
    // sanitized hook bytes in the completion marker.
    let activation_script_sha256 = sanitize_activation_script(cache_dir, &activation_script)?;
    let activation_hooks = sanitize_activation_hooks(cache_dir, &activation_script)?;
    let activation_tmp = cache_dir.join(".activation-check-tmp");
    prepare_activation_tmp(&activation_tmp)?;
    let activated = discover_activated_python(&activation_script, &activation_tmp).await;
    crate::source_build::remove_owned_cache_entry(&activation_tmp)?;
    let ActivatedEnvironment {
        python_executable,
        python_header,
        sysroot_path,
        cuda_executable: discovered_cuda_executable,
        compiler_specs,
        c_compiler,
        cxx_compiler,
        build_prefix,
        host_prefix,
    } = activated?;
    let cuda_executable = request
        .cuda_version
        .as_ref()
        .and(discovered_cuda_executable);
    ensure_cache_descendant(cache_dir, &python_executable, "environment Python")?;
    ensure_cache_descendant(cache_dir, &python_header, "environment Python header")?;
    ensure_cache_descendant(cache_dir, &sysroot_path, "compiler sysroot")?;
    ensure_cache_descendant(cache_dir, &c_compiler, "C compiler")?;
    ensure_cache_descendant(cache_dir, &cxx_compiler, "C++ compiler")?;
    ensure_cache_descendant(cache_dir, &build_prefix, "compiler build prefix")?;
    ensure_cache_descendant(cache_dir, &host_prefix, "Python host prefix")?;
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
    let c_compiler = std::fs::canonicalize(&c_compiler)
        .with_context(|| format!("canonicalizing C compiler {}", c_compiler.display()))?;
    let cxx_compiler = std::fs::canonicalize(&cxx_compiler)
        .with_context(|| format!("canonicalizing C++ compiler {}", cxx_compiler.display()))?;
    let build_prefix = std::fs::canonicalize(&build_prefix).with_context(|| {
        format!(
            "canonicalizing compiler build prefix {}",
            build_prefix.display()
        )
    })?;
    let host_prefix = std::fs::canonicalize(&host_prefix).with_context(|| {
        format!(
            "canonicalizing Python host prefix {}",
            host_prefix.display()
        )
    })?;
    for (compiler, label) in [(&c_compiler, "C compiler"), (&cxx_compiler, "C++ compiler")] {
        if compiler.parent().and_then(Path::parent) != Some(build_prefix.as_path()) {
            bail!(
                "{label} {} is not in solved build prefix {}",
                compiler.display(),
                build_prefix.display()
            );
        }
    }
    if python_executable.parent().and_then(Path::parent) != Some(host_prefix.as_path()) {
        bail!(
            "environment Python {} is not in solved host prefix {}",
            python_executable.display(),
            host_prefix.display()
        );
    }
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
    validate_cached_path(
        cache_dir,
        &compiler_specs,
        "compiler specs",
        CachedPathKind::File,
    )?;
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
        activation_script_sha256,
        activation_hooks,
        build_prefix,
        host_prefix,
        python_executable,
        c_compiler,
        cxx_compiler,
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
        (&marker.c_compiler, "C compiler"),
        (&marker.cxx_compiler, "C++ compiler"),
        (&marker.python_executable, "environment Python"),
        (&marker.python_header, "environment Python header"),
    ] {
        validate_cached_path(cache_dir, path, label, CachedPathKind::File)?;
    }
    for (path, label) in [
        (&marker.build_prefix, "compiler build prefix"),
        (&marker.host_prefix, "Python host prefix"),
    ] {
        validate_cached_path(cache_dir, path, label, CachedPathKind::Directory)?;
    }
    validate_cached_path(
        cache_dir,
        &marker.sysroot_path,
        "compiler sysroot",
        CachedPathKind::Directory,
    )?;
    if file_sha256(&marker.activation_script)? != marker.activation_script_sha256 {
        bail!(
            "cached compiler activation script changed after hermetic environment completion: {}",
            marker.activation_script.display()
        );
    }
    validate_sanitized_activation_script(&marker.activation_script)?;
    let discovered_hooks = activation_hook_paths(cache_dir, &marker.activation_script)?;
    let recorded_hooks = marker
        .activation_hooks
        .iter()
        .map(|hook| hook.path.clone())
        .collect::<Vec<_>>();
    if discovered_hooks != recorded_hooks {
        bail!(
            "cached compiler activation hooks changed after hermetic environment completion at {}",
            cache_dir.display()
        );
    }
    for hook in &marker.activation_hooks {
        validate_cached_path(
            cache_dir,
            &hook.path,
            "compiler activation hook",
            CachedPathKind::File,
        )?;
        validate_sanitized_activation_hook(&hook.path)?;
        if file_sha256(&hook.path)? != hook.sha256 {
            bail!(
                "cached compiler activation hook changed after hermetic environment completion: {}",
                hook.path.display()
            );
        }
    }
    for (compiler, label) in [
        (&marker.c_compiler, "C compiler"),
        (&marker.cxx_compiler, "C++ compiler"),
    ] {
        if compiler.parent().and_then(Path::parent) != Some(marker.build_prefix.as_path()) {
            bail!(
                "cached {label} {} is not in solved build prefix {}",
                compiler.display(),
                marker.build_prefix.display()
            );
        }
    }
    if marker.python_executable.parent().and_then(Path::parent)
        != Some(marker.host_prefix.as_path())
    {
        bail!(
            "cached environment Python {} is not in solved host prefix {}",
            marker.python_executable.display(),
            marker.host_prefix.display()
        );
    }
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
        build_prefix: marker.build_prefix.clone(),
        host_prefix: marker.host_prefix.clone(),
        python_executable: marker.python_executable.clone(),
        c_compiler: marker.c_compiler.clone(),
        cxx_compiler: marker.cxx_compiler.clone(),
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

fn validate_shell_safe_cache_path(path: &Path) -> Result<()> {
    let rendered = path.to_str().ok_or_else(|| {
        anyhow!(
            "hermetic cache path is not UTF-8 and cannot be represented safely in rattler-build activation: {}",
            path.display()
        )
    })?;
    if !path.is_absolute()
        || rendered.is_empty()
        || !rendered.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'.' | b'_' | b'-' | b'+')
        })
    {
        bail!(
            "hermetic cache path `{}` contains whitespace or shell metacharacters that \
             rattler-build cannot quote safely; set RETREAD_CACHE_DIR to an absolute path \
             containing only letters, digits, `/`, `.`, `_`, `-`, or `+`",
            path.display()
        );
    }
    Ok(())
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
    activation_tmp: &Path,
) -> Result<ActivatedEnvironment> {
    let script = r#"
set -euo pipefail
test -d "$2" || exit 85
umask 077
export HOME=$2
export TMPDIR=$2
export TMP=$2
export TEMP=$2
export XDG_CACHE_HOME=$2
export XDG_CONFIG_HOME=$2
export XDG_DATA_HOME=$2
export RETREAD_ACTIVATION_TMPDIR=$2
export PATH=/usr/bin:/bin
unset PYTHON PYTHONHOME PYTHONPATH CONDA_BUILD_SYSROOT CC CXX AR CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LDFLAGS_LD DEBUG_CPPFLAGS DEBUG_CFLAGS DEBUG_CXXFLAGS QEMU_LD_PREFIX COMPILER_PATH GCC_EXEC_PREFIX OBJC_INCLUDE_PATH DEPENDENCIES_OUTPUT SUNPRO_DEPENDENCIES CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH LD_PRELOAD PKG_CONFIG_PATH CMAKE_PREFIX_PATH CUDACXX CUDA_PATH CUDA_HOME NVCC CUDAHOSTCXX CUDAFLAGS NVCCFLAGS NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS AUDITWHEEL_LD_LIBRARY_PATH AUDITWHEEL_ZIP_COMPRESSION_LEVEL
activation_cwd=$PWD
cd "$RETREAD_ACTIVATION_TMPDIR"
set +o pipefail
source "$1" >/dev/null 2>&1
set -o pipefail
cd "$activation_cwd"
test -n "${PYTHON:-}"
test -n "${CC:-}"
test -n "${CXX:-}"
test -n "${AR:-}"
test -n "${CFLAGS:-}"
test -n "${CXXFLAGS:-}"
test -n "${LDFLAGS:-}"
test -d "${CONDA_BUILD_SYSROOT:-}"
test -d "${BUILD_PREFIX:-}"
test -d "${PREFIX:-}"
export PATH="$BUILD_PREFIX/bin:$PREFIX/bin:/usr/bin:/bin"
c_compiler=$(command -v "$CC")
cxx_compiler=$(command -v "$CXX")
ar=$(command -v "$AR")
test "$(/usr/bin/readlink -f "$("$c_compiler" -print-sysroot)")" = "$(/usr/bin/readlink -f "$CONDA_BUILD_SYSROOT")"
test "$(/usr/bin/readlink -f "$("$cxx_compiler" -print-sysroot)")" = "$(/usr/bin/readlink -f "$CONDA_BUILD_SYSROOT")"
case "$(/usr/bin/readlink -f "$ar")" in "$BUILD_PREFIX"/*) ;; *) exit 86 ;; esac
printf '%s\n' "$PYTHON"
"$PYTHON" -I -c 'import pathlib, sysconfig; header = pathlib.Path(sysconfig.get_path("include")) / "Python.h"; assert header.is_file(), header; print(header)'
printf '%s\n' "$CONDA_BUILD_SYSROOT"
cuda_executable=$(command -v "${CUDACXX:-${NVCC:-nvcc}}" || true)
printf '%s\n' "${cuda_executable:--}"
"$c_compiler" -print-file-name=specs
printf '%s\n' "$c_compiler"
printf '%s\n' "$cxx_compiler"
printf '%s\n' "$BUILD_PREFIX"
printf '%s\n' "$PREFIX"
"#;
    let mut command = Command::new("/bin/bash");
    command
        .env_clear()
        .arg("-p")
        .arg("-c")
        .arg(script)
        .arg("retread-hermetic-python-check")
        .arg(activation_script)
        .arg(activation_tmp)
        .current_dir(activation_tmp);
    command
        .env_remove("BASH_ENV")
        .env_remove("ENV")
        .env_remove("LD_LIBRARY_PATH")
        .env_remove("LD_PRELOAD");
    let output =
        run_captured_sealed(&mut command, "validating hermetic Python and headers").await?;
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
    let c_compiler = lines
        .next()
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .ok_or_else(|| anyhow!("conda activation did not report an absolute C compiler"))?;
    let cxx_compiler = lines
        .next()
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .ok_or_else(|| anyhow!("conda activation did not report an absolute C++ compiler"))?;
    let build_prefix = lines
        .next()
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .ok_or_else(|| anyhow!("conda activation did not report an absolute build prefix"))?;
    let host_prefix = lines
        .next()
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .ok_or_else(|| anyhow!("conda activation did not report an absolute host prefix"))?;
    Ok(ActivatedEnvironment {
        python_executable: python,
        python_header: header,
        sysroot_path: sysroot,
        cuda_executable: cuda,
        compiler_specs,
        c_compiler,
        cxx_compiler,
        build_prefix,
        host_prefix,
    })
}

fn rattler_build_executable() -> Result<PathBuf> {
    let path = std::env::var_os("PATH")
        .ok_or_else(|| anyhow!("PATH is unset while resolving `rattler-build`"))?;
    for directory in std::env::split_paths(&path) {
        let candidate = directory.join("rattler-build");
        if candidate.is_file() {
            return std::fs::canonicalize(&candidate).with_context(|| {
                format!(
                    "canonicalizing rattler-build executable {}",
                    candidate.display()
                )
            });
        }
    }
    bail!("rattler-build was not found on PATH; hermetic builds require rattler-build >=0.70.0")
}

fn configure_rattler_command_environment(command: &mut Command, scratch: &Path) {
    // `debug setup` copies its child PATH into build_env.sh. Invoke the
    // already-resolved executable with a fixed base PATH so a caller-provided
    // newline or shell token can never become generated activation code.
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", scratch.join("home"))
        .env("TMPDIR", scratch.join("tmp"))
        .env("TMP", scratch.join("tmp"))
        .env("TEMP", scratch.join("tmp"))
        .env("XDG_CACHE_HOME", scratch.join("xdg-cache"))
        .env("XDG_CONFIG_HOME", scratch.join("xdg-config"))
        .env("XDG_DATA_HOME", scratch.join("xdg-data"))
        .env("XDG_RUNTIME_DIR", scratch.join("runtime"))
        .env("USER", "retread")
        .env("LOGNAME", "retread")
        .env("LANG", "C.UTF-8")
        .env("LC_ALL", "C.UTF-8")
        .env("TERM", "dumb");
    // Package downloads still need the caller's narrowly scoped transport
    // configuration. Do not inherit unrelated process state into generated
    // activation scripts, but preserve corporate proxy and CA settings.
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    if std::env::var_os("SSL_CERT_DIR").is_none() {
        command.env("SSL_CERT_DIR", "/etc/ssl/certs");
    }
}

async fn ensure_rattler_build_version(rattler_build: &Path, scratch: &Path) -> Result<()> {
    let mut command = Command::new(rattler_build);
    command.arg("--version");
    configure_rattler_command_environment(&mut command, scratch);
    let output = run_captured_sealed(&mut command, "checking rattler-build version").await?;
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

async fn run_captured_sealed(command: &mut Command, label: &str) -> Result<std::process::Output> {
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

    fn test_elf(machine: u16) -> Vec<u8> {
        let mut bytes = vec![0u8; 64];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2; // ELF64
        bytes[5] = 1; // little-endian
        bytes[6] = 1; // ELF header version
        bytes[18..20].copy_from_slice(&machine.to_le_bytes());
        bytes
    }

    fn test_elf_with_interp() -> Vec<u8> {
        let mut bytes = test_elf(62);
        bytes.resize(120, 0);
        bytes[16..18].copy_from_slice(&3u16.to_le_bytes()); // ET_DYN
        bytes[32..40].copy_from_slice(&64u64.to_le_bytes());
        bytes[54..56].copy_from_slice(&56u16.to_le_bytes());
        bytes[56..58].copy_from_slice(&1u16.to_le_bytes());
        bytes[64..68].copy_from_slice(&3u32.to_le_bytes()); // PT_INTERP
        bytes
    }

    fn test_elf_with_needed(needed: &str) -> Vec<u8> {
        let mut bytes = test_elf(62);
        bytes.resize(512, 0);
        bytes[16..18].copy_from_slice(&3u16.to_le_bytes()); // ET_DYN
        bytes[32..40].copy_from_slice(&64u64.to_le_bytes());
        bytes[54..56].copy_from_slice(&56u16.to_le_bytes());
        bytes[56..58].copy_from_slice(&2u16.to_le_bytes());

        // One PT_LOAD maps file offset zero at virtual address 0x400000.
        bytes[64..68].copy_from_slice(&1u32.to_le_bytes());
        bytes[80..88].copy_from_slice(&0x400000u64.to_le_bytes());
        bytes[96..104].copy_from_slice(&512u64.to_le_bytes());
        // PT_DYNAMIC occupies four ELF64 dynamic entries at file offset 256.
        bytes[120..124].copy_from_slice(&2u32.to_le_bytes());
        bytes[128..136].copy_from_slice(&256u64.to_le_bytes());
        bytes[136..144].copy_from_slice(&0x400100u64.to_le_bytes());
        bytes[152..160].copy_from_slice(&64u64.to_le_bytes());
        bytes[256..264].copy_from_slice(&5u64.to_le_bytes()); // DT_STRTAB
        bytes[264..272].copy_from_slice(&0x400180u64.to_le_bytes());
        bytes[272..280].copy_from_slice(&10u64.to_le_bytes()); // DT_STRSZ
        bytes[280..288].copy_from_slice(&64u64.to_le_bytes());
        bytes[288..296].copy_from_slice(&1u64.to_le_bytes()); // DT_NEEDED
        bytes[384..384 + needed.len()].copy_from_slice(needed.as_bytes());
        bytes[384 + needed.len()] = 0;
        bytes
    }

    fn test_regular_ar(payload: &[u8]) -> Vec<u8> {
        let mut archive = b"!<arch>\n".to_vec();
        let mut header = [b' '; 60];
        header[..6].copy_from_slice(b"obj.o/");
        let size = format!("{:<10}", payload.len());
        header[48..58].copy_from_slice(size.as_bytes());
        header[58] = 0x60;
        header[59] = b'\n';
        archive.extend_from_slice(&header);
        archive.extend_from_slice(payload);
        if !payload.len().is_multiple_of(2) {
            archive.push(b'\n');
        }
        archive
    }

    fn write_boundary_test_wheel(label: &str, entries: &[(&str, &[u8])]) -> (PathBuf, PathBuf) {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "retread-native-boundary-{label}-{}-{sequence}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir(&root).unwrap();
        let wheel = root.join("demo-1.0-cp311-cp311-linux_x86_64.whl");
        let file = std::fs::File::create(&wheel).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        for (name, bytes) in entries {
            writer
                .start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(bytes).unwrap();
        }
        writer.finish().unwrap();
        (root, wheel)
    }

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
    fn record_csv_supports_quoted_native_member_names() {
        let rows = parse_record_csv(
            b"\"pkg/native,\"\"part\"\".so\",sha256=abc,64\r\npkg-1.0.dist-info/RECORD,,\r\n",
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], "pkg/native,\"part\".so");
        assert_eq!(rows[0][1], "sha256=abc");
        assert_eq!(rows[0][2], "64");
    }

    #[test]
    fn native_wheel_archive_requires_exact_record_inventory() {
        let elf = test_elf_with_needed("libgood.so");
        let record = b"pkg/ext.so,,64\ndemo-1.0.dist-info/RECORD,,\n";
        let (root, wheel) = write_boundary_test_wheel(
            "exact",
            &[("pkg/ext.so", &elf), ("demo-1.0.dist-info/RECORD", record)],
        );
        validate_native_wheel_archive(&wheel, false).unwrap();
        let _ = std::fs::remove_dir_all(root);

        let (root, wheel) = write_boundary_test_wheel(
            "unrecorded",
            &[
                ("pkg/ext.so", &elf),
                ("pkg/unrecorded.so", &elf),
                ("demo-1.0.dist-info/RECORD", record),
            ],
        );
        let error = validate_native_wheel_archive(&wheel, false).unwrap_err();
        assert!(format!("{error:#}").contains("RECORD inventory is not exact"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_wheel_archive_rejects_duplicate_zip_and_record_entries() {
        let elf = test_elf_with_needed("libgood.so");
        let record = b"pkg/ext.so,,64\ndemo-1.0.dist-info/RECORD,,\n";
        let (root, wheel) = write_boundary_test_wheel(
            "duplicate-zip",
            &[
                ("pkg/ext.so", &elf),
                ("pkg/dup.so", &elf),
                ("demo-1.0.dist-info/RECORD", record),
            ],
        );
        // zip 6 refuses duplicate names when writing. Give the same-length
        // second name the first name in both local and central directory
        // records; filenames are outside the compressed data/CRC boundary.
        let mut duplicate_zip = std::fs::read(&wheel).unwrap();
        for offset in 0..=duplicate_zip.len() - b"pkg/dup.so".len() {
            if duplicate_zip[offset..].starts_with(b"pkg/dup.so") {
                duplicate_zip[offset..offset + b"pkg/ext.so".len()].copy_from_slice(b"pkg/ext.so");
            }
        }
        std::fs::write(&wheel, duplicate_zip).unwrap();
        let error = validate_native_wheel_archive(&wheel, false).unwrap_err();
        assert!(format!("{error:#}").contains("duplicate or name-colliding ZIP"));
        let _ = std::fs::remove_dir_all(root);

        let duplicate_record = b"pkg/ext.so,,64\npkg/ext.so,,64\ndemo-1.0.dist-info/RECORD,,\n";
        let (root, wheel) = write_boundary_test_wheel(
            "duplicate-record",
            &[
                ("pkg/ext.so", &elf),
                ("demo-1.0.dist-info/RECORD", duplicate_record),
            ],
        );
        let error = validate_native_wheel_archive(&wheel, false).unwrap_err();
        assert!(format!("{error:#}").contains("RECORD contains duplicate entry"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_wheel_archive_rejects_non_x86_64_elf() {
        let aarch64 = test_elf(183);
        let record = b"pkg/foreign.so,,64\ndemo-1.0.dist-info/RECORD,,\n";
        let (root, wheel) = write_boundary_test_wheel(
            "foreign-elf",
            &[
                ("pkg/foreign.so", &aarch64),
                ("demo-1.0.dist-info/RECORD", record),
            ],
        );
        let error = validate_native_wheel_archive(&wheel, false).unwrap_err();
        assert!(format!("{error:#}").contains("neither x86_64"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_wheel_archive_accepts_cuda_device_elf_only_for_cuda_tuple() {
        let host = test_elf_with_needed("libgood.so");
        let mut cuda = test_elf(190);
        cuda[16..18].copy_from_slice(&2u16.to_le_bytes()); // CUDA ET_EXEC cubin
        let error = validate_native_elf_header("pkg/fake-extension.so", &cuda, true).unwrap_err();
        assert!(format!("{error:#}").contains("attested .cubin"));
        let record = b"pkg/ext.so,,512\npkg/kernel.cubin,,64\ndemo-1.0.dist-info/RECORD,,\n";
        let (root, wheel) = write_boundary_test_wheel(
            "cuda-device",
            &[
                ("pkg/ext.so", &host),
                ("pkg/kernel.cubin", &cuda),
                ("demo-1.0.dist-info/RECORD", record),
            ],
        );
        let inventory = validate_native_wheel_archive(&wheel, true).unwrap();
        assert_eq!(inventory.host_dynamic_elfs, 1);
        assert_eq!(inventory.cuda_device_elfs, 1);
        let error = validate_native_wheel_archive(&wheel, false).unwrap_err();
        assert!(format!("{error:#}").contains("did not provision cuda-nvcc_linux-64"));
        let _ = std::fs::remove_dir_all(root);

        let record = b"pkg/kernel.cubin,,64\ndemo-1.0.dist-info/RECORD,,\n";
        let (root, wheel) = write_boundary_test_wheel(
            "cuda-device-only",
            &[
                ("pkg/kernel.cubin", &cuda),
                ("demo-1.0.dist-info/RECORD", record),
            ],
        );
        let inventory = validate_native_wheel_archive(&wheel, true).unwrap();
        assert_eq!(inventory.host_dynamic_elfs, 0);
        assert_eq!(inventory.cuda_device_elfs, 1);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_wheel_archive_rejects_non_linux_host_elf() {
        let mut host = test_elf_with_needed("libgood.so");
        host[7] = 9; // FreeBSD ELFOSABI
        let error = validate_native_elf_header("pkg/ext.so", &host, false).unwrap_err();
        assert!(format!("{error:#}").contains("not a Linux/System-V ET_DYN"));
    }

    #[test]
    fn native_elf_dependencies_reject_interp_and_path_needed() {
        let interp = test_elf_with_interp();
        let error = validate_elf_dependency_paths("pkg/tool", &interp).unwrap_err();
        assert!(format!("{error:#}").contains("PT_INTERP"));

        let path_needed = test_elf_with_needed("/host/libbad.so");
        let error = validate_elf_dependency_paths("pkg/ext.so", &path_needed).unwrap_err();
        assert!(format!("{error:#}").contains("path-valued DT_NEEDED"));

        let soname_needed = test_elf_with_needed("libgood.so");
        validate_elf_dependency_paths("pkg/ext.so", &soname_needed).unwrap();
    }

    #[test]
    fn native_elf_dependencies_reject_untracked_loader_directives() {
        for (tag, name) in [
            (0x6fff_fefa_u64, "DT_CONFIG"),
            (0x6fff_fefb_u64, "DT_DEPAUDIT"),
            (0x6fff_fefc, "DT_AUDIT"),
            (0x7fff_fffd, "DT_AUXILIARY"),
            (0x7fff_fffe, "DT_USED"),
            (0x7fff_ffff, "DT_FILTER"),
        ] {
            let mut elf = test_elf_with_needed("libgood.so");
            elf[288..296].copy_from_slice(&tag.to_le_bytes());
            let error = validate_elf_dependency_paths("pkg/ext.so", &elf).unwrap_err();
            assert!(format!("{error:#}").contains(name));
        }
    }

    #[test]
    fn native_wheel_archive_rejects_unverifiable_static_objects() {
        let mut object = test_elf(62);
        object[16..18].copy_from_slice(&1u16.to_le_bytes()); // ET_REL
        let archive = test_regular_ar(&object);
        let record = b"pkg/direct.o,,64\ndemo-1.0.dist-info/RECORD,,\n";
        let (root, wheel) = write_boundary_test_wheel(
            "relocatable-object",
            &[
                ("pkg/direct.o", &object),
                ("demo-1.0.dist-info/RECORD", record),
            ],
        );
        let error = validate_native_wheel_archive(&wheel, false).unwrap_err();
        assert!(format!("{error:#}").contains("not a Linux/System-V ET_DYN"));
        let _ = std::fs::remove_dir_all(root);

        let mut object_with_dynamic_table = test_elf_with_needed("libgood.so");
        object_with_dynamic_table[16..18].copy_from_slice(&1u16.to_le_bytes()); // ET_REL
        let record = b"pkg/tricky.o,,512\ndemo-1.0.dist-info/RECORD,,\n";
        let (root, wheel) = write_boundary_test_wheel(
            "relocatable-object-with-dynamic-table",
            &[
                ("pkg/tricky.o", &object_with_dynamic_table),
                ("demo-1.0.dist-info/RECORD", record),
            ],
        );
        let error = validate_native_wheel_archive(&wheel, false).unwrap_err();
        assert!(format!("{error:#}").contains("not a Linux/System-V ET_DYN"));
        let _ = std::fs::remove_dir_all(root);

        let record = b"pkg/libdemo.a,,132\ndemo-1.0.dist-info/RECORD,,\n";
        let (root, wheel) = write_boundary_test_wheel(
            "static-archive",
            &[
                ("pkg/libdemo.a", &archive),
                ("demo-1.0.dist-info/RECORD", record),
            ],
        );
        let error = validate_native_wheel_archive(&wheel, false).unwrap_err();
        assert!(format!("{error:#}").contains("static archive"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_wheel_archive_rejects_thin_ar() {
        let thin = b"!<thin>\n";
        let record = b"pkg/libbad.a,,8\ndemo-1.0.dist-info/RECORD,,\n";
        let (root, wheel) = write_boundary_test_wheel(
            "thin-ar",
            &[
                ("pkg/libbad.a", thin),
                ("demo-1.0.dist-info/RECORD", record),
            ],
        );
        let error = validate_native_wheel_archive(&wheel, false).unwrap_err();
        assert!(format!("{error:#}").contains("thin archive"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn auditwheel_ldpaths_are_tuple_local() {
        let root =
            std::env::temp_dir().join(format!("retread-auditwheel-ldpaths-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let build = root.join("build");
        let host = root.join("host");
        let sysroot = root.join("sysroot");
        for path in [
            build.join("lib"),
            host.join("lib"),
            sysroot.join("usr/lib64"),
        ] {
            std::fs::create_dir_all(path).unwrap();
        }
        let environment = HermeticBuildEnvironment {
            activation_script: root.join("activation.sh"),
            build_prefix: build.clone(),
            host_prefix: host.clone(),
            python_executable: host.join("bin/python"),
            c_compiler: build.join("bin/x86_64-conda-linux-gnu-cc"),
            cxx_compiler: build.join("bin/x86_64-conda-linux-gnu-c++"),
            sysroot_path: sysroot.clone(),
            cuda_executable: None,
            selected_sysroot: (2, 28),
            platform_tag: "manylinux_2_28_x86_64".to_string(),
        };
        let paths = std::env::split_paths(&auditwheel_ldpaths(&environment).unwrap())
            .collect::<BTreeSet<_>>();
        let expected = [
            build.join("lib"),
            host.join("lib"),
            sysroot.join("usr/lib64"),
        ]
        .into_iter()
        .map(|path| std::fs::canonicalize(path).unwrap())
        .collect::<BTreeSet<_>>();
        assert_eq!(paths, expected);
        assert!(paths.iter().all(|path| path.starts_with(&root)));
        let _ = std::fs::remove_dir_all(root);
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
    fn wheel_rpath_depth_uses_installed_pep427_location() {
        assert_eq!(installed_wheel_parent_depth("pkg/ext.so").unwrap(), 1);
        assert_eq!(
            installed_wheel_parent_depth("pkg-1.0.data/platlib/pkg/ext.so").unwrap(),
            1
        );
        assert_eq!(
            installed_wheel_parent_depth("pkg-1.0.data/scripts/tool").unwrap(),
            0
        );
        assert!(installed_wheel_parent_depth("pkg/../outside.so").is_err());
        assert!(installed_wheel_parent_depth("pkg-1.0.data/unknown/ext.so").is_err());
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

    #[test]
    fn activation_hooks_redirect_environment_dumps_to_private_tmp() {
        let root = std::env::temp_dir().join(format!(
            "retread-hermetic-hook-sanitization-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let hook_dir = root.join("prefix/etc/conda/activate.d");
        std::fs::create_dir_all(&hook_dir).unwrap();
        let hook = hook_dir.join("compiler.sh");
        std::fs::write(
            &hook,
            b"env > /tmp/old-env-$$.txt\nenv > /tmp/new-env-$$.txt\n",
        )
        .unwrap();
        let activation = root.join("build_env.sh");
        std::fs::write(&activation, format!(". {}\n", hook.display())).unwrap();

        let markers = sanitize_activation_hooks(&root, &activation).unwrap();
        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0].path, hook);
        assert_eq!(markers[0].sha256, file_sha256(&hook).unwrap());
        let sanitized = std::fs::read_to_string(&hook).unwrap();
        assert!(!sanitized.contains("/tmp/old-env-$$.txt"));
        assert!(!sanitized.contains("/tmp/new-env-$$.txt"));
        assert!(sanitized.contains("${RETREAD_ACTIVATION_TMPDIR:?}/old-env-$$.txt"));
        assert!(sanitized.contains("${RETREAD_ACTIVATION_TMPDIR:?}/new-env-$$.txt"));
        assert!(!sanitized.lines().any(is_environment_dump_line));
        assert!(sanitized.lines().all(|line| !line.starts_with("env >")));
        assert!(sanitized.lines().any(|line| line.starts_with(": >")));
        validate_sanitized_activation_hook(&hook).unwrap();

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn activation_script_drops_provisioning_host_path() {
        let root = std::env::temp_dir().join(format!(
            "retread-hermetic-path-sanitization-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let activation = root.join("build_env.sh");
        std::fs::write(
            &activation,
            "export PATH=/spack/bin:/cuda/bin:/usr/bin\nexport VALUE=safe\n",
        )
        .unwrap();

        let digest = sanitize_activation_script(&root, &activation).unwrap();
        assert_eq!(digest, file_sha256(&activation).unwrap());
        let sanitized = std::fs::read_to_string(&activation).unwrap();
        assert_eq!(sanitized, "export PATH=/usr/bin:/bin\nexport VALUE=safe\n");
        validate_sanitized_activation_script(&activation).unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn shell_safe_cache_path_rejects_metacharacters_and_relative_paths() {
        validate_shell_safe_cache_path(Path::new("/safe/cache-v5_1.0+local")).unwrap();
        for path in [
            "relative/cache",
            "/unsafe/with space",
            "/unsafe/cache;touch",
            "/unsafe/$(touch-pwn)",
            "/unsafe/line\nbreak",
        ] {
            assert!(
                validate_shell_safe_cache_path(Path::new(path)).is_err(),
                "unsafe cache path was accepted: {path:?}"
            );
        }
    }

    #[test]
    fn activation_hook_rejects_unrecognized_public_tmp_and_env_dumps() {
        let root = std::env::temp_dir().join(format!(
            "retread-hermetic-hook-rejection-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let hook = root.join("compiler.sh");
        std::fs::write(&hook, "env > /tmp/environment-${PPID}\n").unwrap();
        assert!(validate_sanitized_activation_hook(&hook).is_err());
        assert!(suppress_environment_snapshot_writes("env > relative-env.txt\n").is_err());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn activation_hook_parser_rejects_shell_metacharacters() {
        let root = std::env::temp_dir().join(format!(
            "retread-hermetic-hook-path-rejection-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let hook_dir = root.join("prefix/etc/conda/activate.d");
        std::fs::create_dir_all(&hook_dir).unwrap();
        let hook = hook_dir.join("compiler.sh");
        std::fs::write(&hook, "true\n").unwrap();
        let activation = root.join("build_env.sh");
        std::fs::write(
            &activation,
            format!(". {};touch /tmp/pwn\n", hook.display()),
        )
        .unwrap();
        assert!(activation_hook_paths(&root, &activation).is_err());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn activation_source_survives_conda_diff_under_pipefail() {
        let root =
            std::env::temp_dir().join(format!("retread-hermetic-pipefail-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let activation = root.join("build_env.sh");
        std::fs::write(
            &activation,
            "printf 'old\\n' > old.txt\nprintf 'new\\n' > new.txt\ndiff old.txt new.txt | sort >/dev/null\n",
        )
        .unwrap();
        let status = std::process::Command::new("/bin/bash")
            .arg("-c")
            .arg("set -euo pipefail; cd \"$2\"; set +o pipefail; source \"$1\"; set -o pipefail")
            .arg("retread-pipefail-test")
            .arg(&activation)
            .arg(&root)
            .status()
            .unwrap();
        assert!(status.success());
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
