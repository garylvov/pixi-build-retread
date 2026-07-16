//! Build a `.whl` from a local path or git checkout via `pip wheel`.
//!
//! Used by `[retread-wheels]` entries that take `path = "..."` or
//! `git = "..."` instead of the PyPI `version + index` form. The
//! produced wheel goes through the same auto-bundle + METADATA-rewrite
//! pipeline as any PyPI-resolved wheel.

use std::collections::HashMap;
use std::ffi::OsString;
#[cfg(unix)]
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::{Read, Seek, Write};
#[cfg(unix)]
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::pypi::{ResolutionTarget, normalized_python_minor};

const BUILT_WHEEL_CACHE_SCHEMA: &str = "retread-built-wheel-v3";
const BUILT_WHEEL_CACHE_ROOT: &str = "built-wheels";
const CHECKOUT_CACHE_VERSION: &str = "v3";
const LOCAL_SOURCE_SNAPSHOT_VERSION: &str = "v5";
const CANONICAL_GIT_SOURCE_SCHEMA: &str = "retread-canonical-git-source-v1";
static BUILD_TMP_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Optional caller knowledge used to bind a source-built artifact to the
/// package identity that requested it.  Even without this hint, every wheel is
/// checked for agreement between its PEP 427 filename and root METADATA.
#[derive(Debug, Clone)]
pub(crate) struct ExpectedWheel {
    pub(crate) name: String,
    pub(crate) version: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct SdistWheelBuild {
    pub(crate) wheel_path: PathBuf,
    pub(crate) sdist_sha256: String,
}

impl ExpectedWheel {
    pub(crate) fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: None,
        }
    }

    pub(crate) fn exact(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: Some(version.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuiltWheelMarker {
    schema: String,
    artifact_target: String,
    source_identity: String,
    filename: String,
    sha256: String,
    name: String,
    version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalGitSourceMarker {
    schema: String,
    repository_identity: String,
    resolved_sha: String,
    ref_state: String,
}

#[derive(Debug, Clone)]
struct CanonicalGitTagRef {
    name: Vec<u8>,
    object_id: String,
}

#[derive(Debug, Clone)]
struct CanonicalGitTagState {
    identity: String,
    refs: Vec<CanonicalGitTagRef>,
}

#[cfg(unix)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WheelFileFingerprint {
    device: u64,
    inode: u64,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[cfg(unix)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictWheelAttestation {
    schema: String,
    artifact_target: String,
    source_identity: String,
    filename: String,
    sha256: String,
    name: String,
    version: String,
    fingerprint: WheelFileFingerprint,
}

#[derive(Debug, Clone)]
struct CanonicalGitSnapshot {
    root: PathBuf,
    repository_identity: String,
    resolved_sha: String,
    ref_state: String,
}

#[derive(Debug)]
struct ValidatedWheel {
    path: PathBuf,
    marker: BuiltWheelMarker,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct ExpectedWheelMismatch(String);

fn is_expected_wheel_mismatch(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.is::<ExpectedWheelMismatch>())
}

/// A pinned ingress artifact whose bytes disagree with the caller's
/// authoritative digest. Callers may use this narrow classification to heal
/// a corrupt download/store entry without treating a correct-hash semantic
/// mismatch (name, version, tags, or archive structure) as disposable.
#[derive(Debug, thiserror::Error)]
#[error("pinned wheel hash mismatch: expected {expected}, found {actual}")]
pub(crate) struct AuthoritativeWheelHashMismatch {
    expected: String,
    actual: String,
}

pub(crate) fn is_authoritative_wheel_hash_mismatch(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.is::<AuthoritativeWheelHashMismatch>())
}

struct ArtifactCacheLock(File);

impl Drop for ArtifactCacheLock {
    fn drop(&mut self) {
        if let Err(error) = fs4::fs_std::FileExt::unlock(&self.0) {
            tracing::warn!(error = %error, "failed to unlock built-wheel cache entry");
        }
    }
}

struct StagingDir(PathBuf);

struct TemporaryFile {
    path: PathBuf,
    armed: bool,
}

impl TemporaryFile {
    fn armed(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn make_staging_tree_removable(path: &Path) {
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return;
    };
    if metadata.file_type().is_symlink() {
        return;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.is_dir() {
            let _ = std::fs::set_permissions(
                path,
                std::fs::Permissions::from_mode(metadata.permissions().mode() | 0o700),
            );
        }
    }
    #[cfg(not(unix))]
    {
        let mut permissions = metadata.permissions();
        permissions.set_readonly(false);
        let _ = std::fs::set_permissions(path, permissions);
    }
    if metadata.is_dir()
        && let Ok(entries) = std::fs::read_dir(path)
    {
        for entry in entries.flatten() {
            make_staging_tree_removable(&entry.path());
        }
    }
}

impl Drop for StagingDir {
    fn drop(&mut self) {
        make_staging_tree_removable(&self.0);
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn hash_fields(domain: &[u8], fields: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for field in fields {
        hasher.update((field.len() as u64).to_be_bytes());
        hasher.update(field);
    }
    format!("{:x}", hasher.finalize())
}

fn same_filesystem_inode(left: &Path, right: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let (Ok(left), Ok(right)) = (left.metadata(), right.metadata()) else {
            return false;
        };
        left.dev() == right.dev() && left.ino() == right.ino()
    }
    #[cfg(not(unix))]
    {
        let _ = (left, right);
        false
    }
}

fn validate_sha256(value: &str, label: &str) -> Result<String> {
    let normalized = value.to_ascii_lowercase();
    if normalized.len() != 64 || !normalized.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} must be exactly 64 hexadecimal SHA-256 characters");
    }
    Ok(normalized)
}

fn built_wheel_cache_dir(kind: &str, source_identity: &str, target: &ResolutionTarget) -> PathBuf {
    crate::courier::retread_cache_root()
        .join(BUILT_WHEEL_CACHE_ROOT)
        .join(kind)
        .join("v3")
        .join(target.artifact_cache_identity())
        .join(source_identity)
}

fn materialized_wheel_output_dir(
    out_dir: &Path,
    source_identity: &str,
    target: &ResolutionTarget,
) -> PathBuf {
    out_dir
        .join(".retread-source-wheels")
        .join("v3")
        .join(target.artifact_cache_identity())
        .join(source_identity)
}

fn native_build_allowed(target: &ResolutionTarget) -> bool {
    target.is_native_build_target()
}

fn source_build_refusal_error(target: &ResolutionTarget) -> anyhow::Error {
    if target.conda_subdir() == crate::glibc::current_pixi_platform()
        && target.conda_subdir() == "linux-aarch64"
        && let (Some(declared), Some(host)) = (target.declared_glibc(), crate::glibc::host_glibc())
        && host > declared
    {
        return anyhow!(
            "refusing to source-build for native target `linux-aarch64` with declared glibc {} on newer host glibc {}; use a compatible sysroot/container or a validated artifact-cache hit",
            crate::glibc::format_glibc(declared),
            crate::glibc::format_glibc(host),
        );
    }
    anyhow!(
        "refusing to build a wheel natively for foreign target `{}` on host `{}` after an exact validated artifact-cache miss",
        target.conda_subdir(),
        crate::glibc::current_pixi_platform(),
    )
}

async fn acquire_artifact_cache_lock(cache_dir: &Path) -> Result<ArtifactCacheLock> {
    let parent = cache_dir.parent().ok_or_else(|| {
        anyhow!(
            "built-wheel cache path has no parent: {}",
            cache_dir.display()
        )
    })?;
    tokio::fs::create_dir_all(parent)
        .await
        .with_context(|| format!("creating built-wheel cache parent {}", parent.display()))?;
    let file_name = cache_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("built-wheel cache path has no UTF-8 filename"))?;
    let lock_path = parent.join(format!(".{file_name}.lock"));
    tokio::task::spawn_blocking(move || {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .with_context(|| format!("opening built-wheel lock {}", lock_path.display()))?;
        fs4::fs_std::FileExt::lock_exclusive(&file)
            .with_context(|| format!("locking built-wheel cache {}", lock_path.display()))?;
        Ok(ArtifactCacheLock(file))
    })
    .await
    .context("built-wheel cache lock task panicked")?
}

fn remove_owned_cache_entry(path: &Path) -> Result<()> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_dir() && !metadata.file_type().is_symlink() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
    .with_context(|| format!("removing invalid owned cache entry {}", path.display()))
}

fn validate_wheel_file(
    path: &Path,
    target: &ResolutionTarget,
    expected: Option<&ExpectedWheel>,
) -> Result<BuiltWheelMarker> {
    validate_wheel_file_with(path, target, expected, true)
}

fn validate_wheel_file_with(
    path: &Path,
    target: &ResolutionTarget,
    expected: Option<&ExpectedWheel>,
    strict_archive: bool,
) -> Result<BuiltWheelMarker> {
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("wheel path has no UTF-8 filename: {}", path.display()))?;
    let standard_filename = crate::emit_pypi::standard_wheel_filename(filename);
    if crate::pypi::score_wheel(&standard_filename, target.wheel_target()) < 0 {
        bail!(
            "source-built wheel `{filename}` is incompatible with python {} on {}",
            target.python_version(),
            target.conda_subdir(),
        );
    }
    let (filename_name, filename_version) =
        crate::pypi::wheel_filename_identity(&standard_filename).ok_or_else(|| {
            anyhow!("source build produced invalid PEP 427 filename `{filename}`")
        })?;
    let metadata = if strict_archive {
        crate::wheel::read_metadata_strict(path)
    } else {
        let file_type = std::fs::symlink_metadata(path)
            .with_context(|| format!("stating cached wheel {}", path.display()))?
            .file_type();
        if !file_type.is_file() || file_type.is_symlink() {
            bail!("cached wheel is not a regular file: {}", path.display());
        }
        crate::wheel::read_metadata(path)
    }
    .with_context(|| format!("validating source-built wheel {}", path.display()))?;
    let metadata_name = crate::relax::canonical_conda_name(&metadata.name);
    let filename_name = crate::relax::canonical_conda_name(&filename_name);
    if metadata_name != filename_name {
        bail!(
            "source-built wheel identity mismatch: filename names `{filename_name}` but METADATA names `{metadata_name}`"
        );
    }
    let metadata_version = uv_pep508::uv_pep440::Version::from_str(&metadata.version)
        .with_context(|| format!("invalid METADATA version `{}`", metadata.version))?;
    if metadata_version != filename_version {
        bail!(
            "source-built wheel identity mismatch: filename version `{filename_version}` but METADATA version `{metadata_version}`"
        );
    }
    validate_expected_wheel(&metadata_name, &metadata_version.to_string(), expected)?;
    Ok(BuiltWheelMarker {
        schema: BUILT_WHEEL_CACHE_SCHEMA.to_string(),
        artifact_target: target.artifact_cache_identity(),
        source_identity: String::new(),
        filename: filename.to_string(),
        sha256: validate_sha256(&metadata.sha256, "built wheel hash")?,
        name: metadata_name,
        version: metadata_version.to_string(),
    })
}

fn validate_expected_wheel(
    actual_name: &str,
    actual_version: &str,
    expected: Option<&ExpectedWheel>,
) -> Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let expected_name = crate::relax::canonical_conda_name(&expected.name);
    if expected_name != actual_name {
        return Err(anyhow::Error::new(ExpectedWheelMismatch(format!(
            "source-built wheel identity mismatch: requested `{expected_name}` but artifact is `{actual_name}`"
        ))));
    }
    if let Some(expected_version) = &expected.version {
        let expected_version =
            uv_pep508::uv_pep440::Version::from_str(expected_version).map_err(|error| {
                anyhow::Error::new(ExpectedWheelMismatch(format!(
                    "invalid expected wheel version `{expected_version}`: {error}"
                )))
            })?;
        let actual_version = uv_pep508::uv_pep440::Version::from_str(actual_version)
            .with_context(|| format!("invalid cached wheel version `{actual_version}`"))?;
        if expected_version != actual_version {
            return Err(anyhow::Error::new(ExpectedWheelMismatch(format!(
                "source-built wheel version mismatch for `{expected_name}`: requested `{expected_version}` but artifact is `{actual_version}`"
            ))));
        }
    }
    Ok(())
}

pub(crate) fn validate_existing_wheel_for_target(
    path: &Path,
    target: &ResolutionTarget,
    expected: Option<&ExpectedWheel>,
) -> Result<String> {
    validate_wheel_file(path, target, expected).map(|marker| marker.sha256)
}

pub(crate) async fn validate_wheel_for_target_async(
    path: &Path,
    target: &ResolutionTarget,
    expected: Option<&ExpectedWheel>,
) -> Result<String> {
    let path = path.to_path_buf();
    let target = target.clone();
    let expected = expected.cloned();
    tokio::task::spawn_blocking(move || {
        validate_wheel_file(&path, &target, expected.as_ref()).map(|marker| marker.sha256)
    })
    .await
    .context("strict wheel validation task panicked")?
}

#[cfg(unix)]
fn wheel_file_fingerprint(path: &Path) -> Result<WheelFileFingerprint> {
    use std::os::unix::fs::MetadataExt;

    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("stating wheel for strict attestation {}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        bail!(
            "wheel artifact must be a regular file for strict attestation: {}",
            path.display(),
        );
    }
    Ok(WheelFileFingerprint {
        device: metadata.dev(),
        inode: metadata.ino(),
        size: metadata.size(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    })
}

fn raw_file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| {
        format!(
            "opening wheel for authoritative hash check {}",
            path.display()
        )
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hashing wheel {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Strictly admit a pinned ingress wheel once, then reuse the attestation only
/// while the exact Unix inode/stat tuple remains unchanged. This avoids
/// repeatedly inflating multi-gigabyte direct wheels while retaining the
/// authoritative SHA/name/version/target checks. Any replacement or in-place
/// mutation changes inode/ctime and forces a fresh strict scan.
pub(crate) async fn validate_pinned_wheel_for_target_async(
    path: &Path,
    target: &ResolutionTarget,
    expected: &ExpectedWheel,
    authoritative_sha256: &str,
    source: &str,
) -> Result<String> {
    let authoritative_sha256 = validate_sha256(authoritative_sha256, "pinned wheel hash")?;
    #[cfg(not(unix))]
    {
        let path_for_hash = path.to_path_buf();
        let actual_hash = tokio::task::spawn_blocking(move || raw_file_sha256(&path_for_hash))
            .await
            .context("pinned wheel hash task panicked")??;
        if actual_hash != authoritative_sha256 {
            return Err(anyhow::Error::new(AuthoritativeWheelHashMismatch {
                expected: authoritative_sha256,
                actual: actual_hash,
            }));
        }
        let actual = validate_wheel_for_target_async(path, target, Some(expected)).await?;
        return Ok(actual);
    }
    #[cfg(unix)]
    {
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow!("pinned wheel path has no UTF-8 filename"))?
            .to_string();
        let expected_name = crate::relax::canonical_conda_name(&expected.name);
        let expected_version = expected.version.as_deref().unwrap_or("");
        let artifact_target = target.artifact_cache_identity();
        let source_identity =
            hash_fields(b"retread-pinned-wheel-source-v1\0", &[source.as_bytes()]);
        let attestation_identity = hash_fields(
            b"retread-strict-wheel-attestation-v1\0",
            &[
                authoritative_sha256.as_bytes(),
                filename.as_bytes(),
                artifact_target.as_bytes(),
                source_identity.as_bytes(),
                expected_name.as_bytes(),
                expected_version.as_bytes(),
            ],
        );
        let attestation_dir = crate::courier::retread_cache_root()
            .join("strict-wheel-attestations")
            .join("v1")
            .join(attestation_identity);
        let _lock = acquire_artifact_cache_lock(&attestation_dir).await?;
        if attestation_dir.try_exists()? {
            let file_type = std::fs::symlink_metadata(&attestation_dir)
                .with_context(|| {
                    format!(
                        "stating strict wheel attestation {}",
                        attestation_dir.display()
                    )
                })?
                .file_type();
            if !file_type.is_dir() || file_type.is_symlink() {
                bail!(
                    "strict wheel attestation path is not a real directory: {}",
                    attestation_dir.display(),
                );
            }
        }
        let path_for_fingerprint = path.to_path_buf();
        let fingerprint =
            tokio::task::spawn_blocking(move || wheel_file_fingerprint(&path_for_fingerprint))
                .await
                .context("wheel attestation fingerprint task panicked")??;
        let marker_path = attestation_dir.join("attestation.json");
        if marker_path.try_exists()? {
            let file_type = std::fs::symlink_metadata(&marker_path)
                .with_context(|| format!("stating strict wheel marker {}", marker_path.display()))?
                .file_type();
            if !file_type.is_file() || file_type.is_symlink() {
                bail!(
                    "strict wheel marker is not a regular file: {}",
                    marker_path.display(),
                );
            }
        }
        if let Ok(marker_bytes) = std::fs::read(&marker_path)
            && let Ok(marker) = serde_json::from_slice::<StrictWheelAttestation>(&marker_bytes)
            && marker.schema == "retread-strict-wheel-attestation-v1"
            && marker.artifact_target == artifact_target
            && marker.source_identity == source_identity
            && marker.filename == filename
            && marker.sha256 == authoritative_sha256
            && marker.name == expected_name
            && expected
                .version
                .as_ref()
                .is_none_or(|version| marker.version == *version)
            && marker.fingerprint == fingerprint
        {
            return Ok(authoritative_sha256);
        }

        // Hash raw bytes before opening the ZIP. A truncated/malformed file
        // with the wrong authoritative digest is healable ingress corruption;
        // only a correct-hash artifact proceeds to terminal archive/semantic
        // validation below.
        let path_for_hash = path.to_path_buf();
        let actual_hash = tokio::task::spawn_blocking(move || raw_file_sha256(&path_for_hash))
            .await
            .context("pinned wheel hash task panicked")??;
        let path_for_post_hash_fingerprint = path.to_path_buf();
        let post_hash_fingerprint = tokio::task::spawn_blocking(move || {
            wheel_file_fingerprint(&path_for_post_hash_fingerprint)
        })
        .await
        .context("post-hash wheel fingerprint task panicked")??;
        if post_hash_fingerprint != fingerprint {
            bail!(
                "pinned wheel changed while its authoritative hash was checked: {}",
                path.display(),
            );
        }
        if actual_hash != authoritative_sha256 {
            return Err(anyhow::Error::new(AuthoritativeWheelHashMismatch {
                expected: authoritative_sha256,
                actual: actual_hash,
            }));
        }

        let path_for_validation = path.to_path_buf();
        let target_for_validation = target.clone();
        let expected_for_validation = expected.clone();
        let marker = tokio::task::spawn_blocking(move || {
            validate_wheel_file(
                &path_for_validation,
                &target_for_validation,
                Some(&expected_for_validation),
            )
        })
        .await
        .context("strict pinned-wheel validation task panicked")??;
        debug_assert_eq!(marker.sha256, authoritative_sha256);
        let path_for_fingerprint = path.to_path_buf();
        let final_fingerprint =
            tokio::task::spawn_blocking(move || wheel_file_fingerprint(&path_for_fingerprint))
                .await
                .context("post-validation wheel fingerprint task panicked")??;
        if final_fingerprint != fingerprint {
            bail!(
                "pinned wheel changed while it was strictly validated: {}",
                path.display(),
            );
        }
        std::fs::create_dir_all(&attestation_dir).with_context(|| {
            format!(
                "creating strict wheel attestation {}",
                attestation_dir.display()
            )
        })?;
        let attestation = StrictWheelAttestation {
            schema: "retread-strict-wheel-attestation-v1".to_string(),
            artifact_target,
            source_identity,
            filename,
            sha256: marker.sha256.clone(),
            name: marker.name,
            version: marker.version,
            fingerprint: final_fingerprint,
        };
        let temporary = attestation_dir.join(format!(
            ".attestation.{}.{}.tmp",
            std::process::id(),
            BUILD_TMP_SEQUENCE.fetch_add(1, Ordering::Relaxed),
        ));
        let temporary_guard = TemporaryFile::armed(temporary.clone());
        std::fs::write(
            &temporary,
            serde_json::to_vec_pretty(&attestation)
                .context("serializing strict wheel attestation")?,
        )
        .with_context(|| format!("writing strict wheel attestation {}", temporary.display()))?;
        std::fs::rename(&temporary, &marker_path).with_context(|| {
            format!(
                "publishing strict wheel attestation {}",
                marker_path.display()
            )
        })?;
        temporary_guard.disarm();
        Ok(marker.sha256)
    }
}

fn validate_cache_entry(
    cache_dir: &Path,
    source_identity: &str,
    target: &ResolutionTarget,
    expected: Option<&ExpectedWheel>,
) -> Result<Option<ValidatedWheel>> {
    if !cache_dir.try_exists()? {
        return Ok(None);
    }
    let cache_type = std::fs::symlink_metadata(cache_dir)
        .with_context(|| format!("stating built-wheel cache {}", cache_dir.display()))?
        .file_type();
    if !cache_type.is_dir() || cache_type.is_symlink() {
        bail!("built-wheel cache entry is not an owned regular directory");
    }
    let marker_path = cache_dir.join("artifact.json");
    let marker_type = std::fs::symlink_metadata(&marker_path)
        .with_context(|| format!("stating built-wheel marker {}", marker_path.display()))?
        .file_type();
    if !marker_type.is_file() || marker_type.is_symlink() {
        bail!("built-wheel cache marker is not a regular file");
    }
    let marker: BuiltWheelMarker = serde_json::from_slice(
        &std::fs::read(&marker_path)
            .with_context(|| format!("reading built-wheel marker {}", marker_path.display()))?,
    )
    .with_context(|| format!("parsing built-wheel marker {}", marker_path.display()))?;
    if marker.schema != BUILT_WHEEL_CACHE_SCHEMA
        || marker.source_identity != source_identity
        || marker.artifact_target != target.artifact_cache_identity()
        || Path::new(&marker.filename)
            .file_name()
            .and_then(|v| v.to_str())
            != Some(marker.filename.as_str())
    {
        bail!("built-wheel cache marker does not match its v3 namespace");
    }
    let path = cache_dir.join(&marker.filename);
    // Integrity and the caller's semantic expectation are distinct. A cache
    // entry can be perfectly valid for its source identity yet be the wrong
    // package/version for a replay request. Never delete such an entry: doing
    // so would turn a deterministic contract failure into a rebuild race.
    let actual = validate_wheel_file_with(&path, target, None, false)?;
    if actual.sha256 != marker.sha256
        || actual.name != marker.name
        || actual.version != marker.version
    {
        bail!("built-wheel cache marker does not match artifact bytes");
    }
    validate_expected_wheel(&marker.name, &marker.version, expected)?;
    Ok(Some(ValidatedWheel { path, marker }))
}

fn unique_output_temporary(out_dir: &Path) -> Result<(PathBuf, File)> {
    for _ in 0..100 {
        let sequence = BUILD_TMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        // Keep this basename bounded: the canonical wheel destination may
        // already be close to NAME_MAX.
        let path = out_dir.join(format!(
            ".retread-wheel-{}-{sequence}.tmp",
            std::process::id(),
        ));
        match std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
        {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("creating wheel output temp {}", path.display()));
            }
        }
    }
    bail!("could not allocate a unique wheel output temporary file")
}

async fn materialize_validated_wheel(wheel: &ValidatedWheel, out_dir: &Path) -> Result<PathBuf> {
    // Open the validated cache inode before the first await, while the caller
    // still owns the artifact-cache lock. If this future is later cancelled,
    // an already-spawned blocking publisher owns the fd and therefore cannot
    // observe a concurrently deleted/replaced cache pathname.
    let source_file = File::open(&wheel.path)
        .with_context(|| format!("opening validated wheel {}", wheel.path.display()))?;
    tokio::fs::create_dir_all(out_dir)
        .await
        .with_context(|| format!("creating wheel output dir {}", out_dir.display()))?;
    let destination = out_dir.join(&wheel.marker.filename);
    // Different source identities can legitimately produce the same wheel
    // basename and therefore hold different artifact-cache locks. Serialize
    // publication by the full caller destination in a managed lock namespace.
    let absolute_destination = std::fs::canonicalize(out_dir)
        .with_context(|| format!("canonicalizing wheel output dir {}", out_dir.display()))?
        .join(&wheel.marker.filename);
    let output_identity = hash_fields(
        b"retread-built-wheel-output-v1\0",
        &[absolute_destination.to_string_lossy().as_bytes()],
    );
    let output_lock_namespace = crate::courier::retread_cache_root()
        .join("built-wheel-output-locks")
        .join("v1")
        .join(output_identity);
    let output_lock = acquire_artifact_cache_lock(&output_lock_namespace).await?;
    if destination.is_file() && !same_filesystem_inode(&destination, &wheel.path) {
        let actual = tokio::task::spawn_blocking({
            let destination = destination.clone();
            move || crate::wheel::read_metadata(&destination)
        })
        .await
        .context("existing output wheel validation task panicked")?;
        if let Ok(metadata) = actual
            && metadata.sha256.eq_ignore_ascii_case(&wheel.marker.sha256)
        {
            return Ok(destination);
        }
    }
    // Always copy to a fresh inode. A hardlink would let caller-side rewrite
    // or corruption mutate the persistent content-addressed cache, and its old
    // mtime could make a derived wheel from another source identity look fresh.
    let source = wheel.path.clone();
    let out_dir = out_dir.to_path_buf();
    let destination_for_publish = destination.clone();
    tokio::task::spawn_blocking(move || {
        // The detached blocking job owns both cleanup and the publication
        // lock, so cancelling the async caller cannot strand a temp file or
        // let a competing publisher race a still-running copy.
        let (temporary, mut temporary_file) = unique_output_temporary(&out_dir)?;
        let temporary_guard = TemporaryFile::armed(temporary.clone());
        let mut source_file = source_file;
        std::io::copy(&mut source_file, &mut temporary_file).with_context(|| {
            format!(
                "materializing validated wheel {} -> {}",
                source.display(),
                temporary.display(),
            )
        })?;
        drop(temporary_file);
        std::fs::rename(&temporary, &destination_for_publish).with_context(|| {
            format!(
                "atomically publishing wheel into caller output {} (the output directory was not removed or scanned)",
                destination_for_publish.display(),
            )
        })?;
        temporary_guard.disarm();
        drop(output_lock);
        Ok::<_, anyhow::Error>(())
    })
    .await
    .context("wheel output publication task panicked")??;
    Ok(destination)
}

fn unique_staging_dir(cache_dir: &Path) -> Result<StagingDir> {
    let parent = cache_dir
        .parent()
        .ok_or_else(|| anyhow!("cache directory has no parent: {}", cache_dir.display()))?;
    let cache_name = cache_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("cache directory has no UTF-8 filename"))?;
    for _ in 0..100 {
        let sequence = BUILD_TMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".{cache_name}.{}.{}.tmp",
            std::process::id(),
            sequence,
        ));
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok(StagingDir(path)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("creating build staging dir {}", path.display()));
            }
        }
    }
    bail!("could not allocate a unique built-wheel staging directory")
}

async fn cached_build<F, Fut>(
    kind: &str,
    source_identity: &str,
    target: &ResolutionTarget,
    out_dir: &Path,
    expected: Option<&ExpectedWheel>,
    build: F,
) -> Result<PathBuf>
where
    F: FnOnce(PathBuf) -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    let cache_dir = built_wheel_cache_dir(kind, source_identity, target);
    let materialized_out = materialized_wheel_output_dir(out_dir, source_identity, target);
    let _lock = acquire_artifact_cache_lock(&cache_dir).await?;
    let cached = tokio::task::spawn_blocking({
        let cache_dir = cache_dir.clone();
        let source_identity = source_identity.to_string();
        let target = target.clone();
        let expected = expected.cloned();
        move || validate_cache_entry(&cache_dir, &source_identity, &target, expected.as_ref())
    })
    .await
    .context("built-wheel cache validation task panicked")?;
    match cached {
        Ok(Some(wheel)) => return materialize_validated_wheel(&wheel, &materialized_out).await,
        Ok(None) => {}
        Err(error) if is_expected_wheel_mismatch(&error) => return Err(error),
        Err(error) => {
            tracing::warn!(
                cache = %cache_dir.display(),
                error = %format!("{error:#}"),
                "invalid owned built-wheel cache entry; rebuilding",
            );
            remove_owned_cache_entry(&cache_dir)?;
        }
    }
    if !native_build_allowed(target) {
        return Err(source_build_refusal_error(target));
    }

    let staging = unique_staging_dir(&cache_dir)?;
    let build_dir = staging.0.join("build");
    std::fs::create_dir(&build_dir)
        .with_context(|| format!("creating empty build output {}", build_dir.display()))?;
    build(build_dir.clone()).await?;
    let built = find_built_wheel(&build_dir).await?;
    let mut marker = tokio::task::spawn_blocking({
        let built = built.clone();
        let target = target.clone();
        let expected = expected.cloned();
        move || validate_wheel_file(&built, &target, expected.as_ref())
    })
    .await
    .context("newly built wheel validation task panicked")??;
    marker.source_identity = source_identity.to_string();
    let cached_wheel = staging.0.join(&marker.filename);
    tokio::fs::rename(&built, &cached_wheel)
        .await
        .with_context(|| format!("staging built wheel {}", cached_wheel.display()))?;
    tokio::fs::remove_dir_all(&build_dir)
        .await
        .with_context(|| format!("cleaning private build dir {}", build_dir.display()))?;
    std::fs::write(
        staging.0.join("artifact.json"),
        serde_json::to_vec_pretty(&marker).context("serializing built-wheel marker")?,
    )
    .context("writing built-wheel marker")?;
    remove_owned_cache_entry(&cache_dir)?;
    std::fs::rename(&staging.0, &cache_dir).with_context(|| {
        format!(
            "atomically publishing built-wheel cache {}",
            cache_dir.display()
        )
    })?;
    let published = ValidatedWheel {
        path: cache_dir.join(&marker.filename),
        marker,
    };
    materialize_validated_wheel(&published, &materialized_out).await
}

async fn lookup_cached_build(
    kind: &str,
    source_identity: &str,
    target: &ResolutionTarget,
    out_dir: &Path,
    expected: Option<&ExpectedWheel>,
) -> Result<Option<PathBuf>> {
    let cache_dir = built_wheel_cache_dir(kind, source_identity, target);
    let materialized_out = materialized_wheel_output_dir(out_dir, source_identity, target);
    let _lock = acquire_artifact_cache_lock(&cache_dir).await?;
    let cached = tokio::task::spawn_blocking({
        let cache_dir = cache_dir.clone();
        let source_identity = source_identity.to_string();
        let target = target.clone();
        let expected = expected.cloned();
        move || validate_cache_entry(&cache_dir, &source_identity, &target, expected.as_ref())
    })
    .await
    .context("built-wheel cache validation task panicked")?;
    match cached {
        Ok(Some(wheel)) => Ok(Some(
            materialize_validated_wheel(&wheel, &materialized_out).await?,
        )),
        Ok(None) => Ok(None),
        Err(error) if is_expected_wheel_mismatch(&error) => Err(error),
        Err(error) => {
            tracing::warn!(
                cache = %cache_dir.display(),
                error = %format!("{error:#}"),
                "invalid owned built-wheel cache entry; treating as an exact miss",
            );
            remove_owned_cache_entry(&cache_dir)?;
            Ok(None)
        }
    }
}

/// Validate an exact built-wheel cache leaf without publishing it into a
/// caller output directory. This is only an authorization probe: callers must
/// perform a second exact lookup after resolving all mutable source state.
async fn probe_cached_build(
    kind: &str,
    source_identity: &str,
    target: &ResolutionTarget,
    expected: Option<&ExpectedWheel>,
) -> Result<bool> {
    let cache_dir = built_wheel_cache_dir(kind, source_identity, target);
    let _lock = acquire_artifact_cache_lock(&cache_dir).await?;
    let cached = tokio::task::spawn_blocking({
        let cache_dir = cache_dir.clone();
        let source_identity = source_identity.to_string();
        let target = target.clone();
        let expected = expected.cloned();
        move || validate_cache_entry(&cache_dir, &source_identity, &target, expected.as_ref())
    })
    .await
    .context("built-wheel cache probe task panicked")?;
    match cached {
        Ok(Some(_)) => Ok(true),
        Ok(None) => Ok(false),
        Err(error) if is_expected_wheel_mismatch(&error) => Ok(false),
        Err(error) => {
            tracing::warn!(
                cache = %cache_dir.display(),
                error = %format!("{error:#}"),
                "invalid owned built-wheel cache entry; treating as an exact miss",
            );
            remove_owned_cache_entry(&cache_dir)?;
            Ok(false)
        }
    }
}

/// Probe every historical ref-state leaf under an exact
/// URL/SHA/subdirectory family without materializing any artifact. A later
/// exact checkout determines which single state is current; old tag-state
/// leaves may coexist indefinitely without making replay ambiguous.
async fn probe_cached_git_family_states(
    family_identity: &str,
    target: &ResolutionTarget,
    expected: Option<&ExpectedWheel>,
) -> Result<Vec<String>> {
    let family_dir = built_wheel_cache_dir("git", family_identity, target);
    let metadata = match std::fs::symlink_metadata(&family_dir) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        bail!(
            "git built-wheel cache family is not a real directory: {}",
            family_dir.display(),
        );
    }
    let mut ref_states = std::fs::read_dir(&family_dir)
        .with_context(|| format!("reading git cache family {}", family_dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    ref_states.sort_by_key(std::fs::DirEntry::file_name);
    let mut hits = Vec::new();
    for entry in ref_states {
        let Some(ref_state) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if ref_state.len() != 64
            || !ref_state.bytes().all(|byte| byte.is_ascii_hexdigit())
            || ref_state != ref_state.to_ascii_lowercase()
            || !entry
                .file_type()
                .with_context(|| format!("stating git cache leaf {}", entry.path().display()))?
                .is_dir()
        {
            continue;
        }
        let source_identity = git_wheel_source_identity(family_identity, &ref_state);
        if probe_cached_build("git", &source_identity, target, expected).await? {
            hits.push(ref_state);
        }
    }
    Ok(hits)
}

struct PreparedSourceSnapshot {
    workspace: Arc<PreparedSourceWorkspace>,
    identity: String,
}

struct PreparedSourceWorkspace {
    directory: StagingDir,
    // When present, serializes use of the deterministic source-workspace path
    // for the complete uv build/injection lease.
    _workspace_lock: Option<ArtifactCacheLock>,
}

struct TemporaryWritableDirectory {
    path: PathBuf,
    original: std::fs::Permissions,
    restored: bool,
}

impl TemporaryWritableDirectory {
    fn new(path: &Path) -> Result<Self> {
        let original = std::fs::symlink_metadata(path)
            .with_context(|| format!("stating snapshot root {}", path.display()))?
            .permissions();
        let mut writable = original.clone();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            writable.set_mode(writable.mode() | 0o700);
        }
        #[cfg(not(unix))]
        writable.set_readonly(false);
        std::fs::set_permissions(path, writable).with_context(|| {
            format!(
                "temporarily making snapshot root writable {}",
                path.display()
            )
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            original,
            restored: false,
        })
    }

    fn restore(mut self) -> Result<()> {
        std::fs::set_permissions(&self.path, self.original.clone()).with_context(|| {
            format!(
                "restoring snapshot root permissions {}",
                self.path.display()
            )
        })?;
        self.restored = true;
        Ok(())
    }
}

impl Drop for TemporaryWritableDirectory {
    fn drop(&mut self) {
        if !self.restored
            && let Err(error) = std::fs::set_permissions(&self.path, self.original.clone())
        {
            tracing::warn!(
                path = %self.path.display(),
                error = %error,
                "failed to restore snapshot root permissions after metadata attachment",
            );
        }
    }
}

#[derive(Debug, Clone)]
struct ExternalPathGitState {
    resolved_sha: String,
    ref_state: String,
}

/// A path-built wheel plus the immutable source snapshot it was built from.
/// The handler retains this value through source-file injection so both phases
/// observe exactly the bytes bound to the cache identity.
pub(crate) struct PathWheelBuild {
    wheel_path: PathBuf,
    _source_snapshot: PreparedSourceSnapshot,
    project_root: PathBuf,
}

impl PathWheelBuild {
    pub(crate) fn wheel_path(&self) -> &Path {
        &self.wheel_path
    }

    pub(crate) fn source_root(&self) -> &Path {
        &self.project_root
    }
}

impl PreparedSourceSnapshot {
    fn root(&self) -> &Path {
        &self.workspace.directory.0
    }
}

async fn stabilize_source_snapshot_workspace(
    mut snapshot: PreparedSourceSnapshot,
    kind: &str,
    source_identity: &str,
) -> Result<PreparedSourceSnapshot> {
    let workspace = crate::courier::retread_cache_root()
        .join("source-workspaces")
        .join("v1")
        .join(kind)
        .join(source_identity);
    let lock = acquire_artifact_cache_lock(&workspace).await?;
    remove_owned_cache_entry(&workspace)?;
    let source_workspace = Arc::get_mut(&mut snapshot.workspace)
        .expect("a source snapshot cannot be shared before workspace publication");
    std::fs::rename(&source_workspace.directory.0, &workspace).with_context(|| {
        format!(
            "publishing deterministic source workspace {}",
            workspace.display(),
        )
    })?;
    source_workspace.directory.0 = workspace;
    source_workspace._workspace_lock = Some(lock);
    Ok(snapshot)
}

fn normalize_snapshot_times(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        // Use 2000-01-01 UTC: ordinary ZIP cannot represent pre-1980 dates,
        // and exactly 1980-01-01 UTC becomes 1979 in negative-offset zones.
        let zip_safe_epoch = std::time::UNIX_EPOCH + Duration::from_secs(946_684_800);
        let times = std::fs::FileTimes::new()
            .set_accessed(zip_safe_epoch)
            .set_modified(zip_safe_epoch);
        File::open(path)
            .with_context(|| {
                format!(
                    "opening snapshot path for timestamp normalization {}",
                    path.display()
                )
            })?
            .set_times(times)
            .with_context(|| format!("normalizing snapshot timestamp {}", path.display()))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn relative_source_path<'a>(root: &Path, path: &'a Path) -> Result<&'a str> {
    path.strip_prefix(root)
        .expect("source snapshot walk stays below its root")
        .to_str()
        .ok_or_else(|| anyhow!("source-build paths must be UTF-8: {}", path.display()))
}

fn hash_snapshot_record(hasher: &mut Sha256, kind: u8, relative: &str, mode: u32) {
    hasher.update([kind]);
    hasher.update((relative.len() as u64).to_be_bytes());
    hasher.update(relative.as_bytes());
    hasher.update(mode.to_be_bytes());
}

#[cfg(target_os = "linux")]
fn validate_snapshot_symlink(
    root: &Path,
    excluded_roots: &[PathBuf],
    link: &Path,
    target: &Path,
) -> Result<Vec<OsString>> {
    if target.is_absolute() {
        bail!(
            "source-build symlink {} escapes the source tree via absolute target {}",
            link.display(),
            target.display(),
        );
    }
    let parent = link
        .parent()
        .expect("source-build symlink below canonical source root");
    let mut normalized = parent
        .strip_prefix(root)
        .expect("source-build symlink stays below root")
        .components()
        .map(|component| component.as_os_str().to_owned())
        .collect::<Vec<_>>();
    for component in target.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::Normal(value) => normalized.push(value.to_owned()),
            std::path::Component::ParentDir => {
                if normalized.pop().is_none() {
                    bail!(
                        "source-build symlink {} escapes the source tree via target {}",
                        link.display(),
                        target.display(),
                    );
                }
            }
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                bail!(
                    "source-build symlink {} has unsupported target {}",
                    link.display(),
                    target.display(),
                );
            }
        }
    }
    let mut lexical_target = root.to_path_buf();
    lexical_target.extend(&normalized);
    if excluded_roots
        .iter()
        .any(|excluded| lexical_target.starts_with(excluded))
    {
        bail!(
            "source-build symlink {} targets the managed output subtree {}",
            link.display(),
            target.display(),
        );
    }
    Ok(normalized)
}

#[cfg(unix)]
fn create_snapshot_symlink(target: &Path, destination: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, destination)
}

#[cfg(windows)]
fn create_snapshot_symlink(target: &Path, destination: &Path) -> std::io::Result<()> {
    if target.is_dir() {
        std::os::windows::fs::symlink_dir(target, destination)
    } else {
        std::os::windows::fs::symlink_file(target, destination)
    }
}

fn canonicalize_future_path(path: &Path) -> Result<PathBuf> {
    let absolute = std::path::absolute(path)
        .with_context(|| format!("making path absolute: {}", path.display()))?;
    let mut ancestor = absolute.as_path();
    let mut missing = Vec::new();
    while !ancestor.exists() {
        let name = ancestor.file_name().ok_or_else(|| {
            anyhow!(
                "could not find an existing ancestor for path {}",
                absolute.display(),
            )
        })?;
        missing.push(name.to_owned());
        ancestor = ancestor
            .parent()
            .expect("a path with a file name has a parent");
    }
    let mut canonical = std::fs::canonicalize(ancestor)
        .with_context(|| format!("canonicalizing path ancestor {}", ancestor.display()))?;
    for component in missing.into_iter().rev() {
        canonical.push(component);
    }
    Ok(canonical)
}

fn is_managed_snapshot_output(name: &std::ffi::OsStr) -> bool {
    let Some(name) = name.to_str() else {
        return false;
    };
    let name = name.strip_prefix('.').unwrap_or(name);
    let strip_exact_suffix = |suffix: &str| {
        name.strip_suffix(suffix).or_else(|| {
            let temporary = name.strip_suffix(".tmp")?;
            if let Some(stem) = temporary.strip_suffix(suffix) {
                return Some(stem);
            }
            let (with_pid, sequence) = temporary.rsplit_once('.')?;
            let (stem, pid) = with_pid.rsplit_once('.')?;
            (pid.bytes().all(|byte| byte.is_ascii_digit())
                && !pid.is_empty()
                && sequence.bytes().all(|byte| byte.is_ascii_digit())
                && !sequence.is_empty())
            .then(|| stem.strip_suffix(suffix))
            .flatten()
        })
    };
    if let Some(stem) = strip_exact_suffix(".log") {
        return stem
            .strip_prefix("retread-progress-")
            .is_some_and(|bundle| !bundle.is_empty());
    }
    let Some(stem) = strip_exact_suffix(".json") else {
        return false;
    };
    stem == "retread-audit"
        || stem
            .strip_prefix("retread-audit-")
            .is_some_and(|bundle| !bundle.is_empty())
        || stem
            .strip_prefix("retread-probe-trace-")
            .is_some_and(|bundle| !bundle.is_empty())
        || stem.strip_prefix("retread-").is_some_and(|bundle| {
            bundle
                .strip_suffix(".lock")
                .or_else(|| bundle.strip_suffix(".retread-lock"))
                .is_some_and(|bundle| !bundle.is_empty())
        })
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SnapshotStat {
    device: u64,
    inode: u64,
    mode: u32,
    size: i128,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[cfg(unix)]
impl SnapshotStat {
    fn from_rustix(stat: &rustix::fs::Stat) -> Self {
        Self {
            device: stat.st_dev as u64,
            inode: stat.st_ino as u64,
            mode: stat.st_mode as u32,
            size: stat.st_size as i128,
            modified_seconds: stat.st_mtime as i64,
            modified_nanoseconds: stat.st_mtime_nsec as i64,
            changed_seconds: stat.st_ctime as i64,
            changed_nanoseconds: stat.st_ctime_nsec as i64,
        }
    }

    fn file_type(self) -> rustix::fs::FileType {
        rustix::fs::FileType::from_raw_mode(self.mode as _)
    }

    fn snapshot_mode(self) -> u32 {
        self.mode & 0o7777
    }

    fn file_size(self, path: &Path) -> Result<u64> {
        u64::try_from(self.size).with_context(|| {
            format!(
                "source file {} reported an invalid negative size",
                path.display()
            )
        })
    }

    fn permissions(self) -> std::fs::Permissions {
        use std::os::unix::fs::PermissionsExt;
        std::fs::Permissions::from_mode(self.snapshot_mode())
    }
}

#[cfg(unix)]
fn snapshot_stat_at<Fd: std::os::fd::AsFd>(directory: Fd, name: &CStr) -> Result<SnapshotStat> {
    rustix::fs::statat(directory, name, rustix::fs::AtFlags::SYMLINK_NOFOLLOW)
        .map(|stat| SnapshotStat::from_rustix(&stat))
        .map_err(Into::into)
}

#[cfg(unix)]
fn snapshot_fd_stat<Fd: std::os::fd::AsFd>(fd: Fd) -> Result<SnapshotStat> {
    rustix::fs::fstat(fd)
        .map(|stat| SnapshotStat::from_rustix(&stat))
        .map_err(Into::into)
}

#[cfg(unix)]
fn open_snapshot_at<Fd: std::os::fd::AsFd>(
    directory: Fd,
    name: &CStr,
    expected: SnapshotStat,
    expected_type: rustix::fs::FileType,
    display: &Path,
) -> Result<File> {
    let mut flags =
        rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::CLOEXEC | rustix::fs::OFlags::NOFOLLOW;
    if expected_type == rustix::fs::FileType::Directory {
        flags |= rustix::fs::OFlags::DIRECTORY;
    }
    let fd = rustix::fs::openat(directory, name, flags, rustix::fs::Mode::empty()).with_context(
        || {
            format!(
                "opening source entry relative to parent FD {}",
                display.display()
            )
        },
    )?;
    let opened = snapshot_fd_stat(&fd)
        .with_context(|| format!("stating opened source entry {}", display.display()))?;
    if opened != expected || opened.file_type() != expected_type {
        bail!(
            "source entry {} changed inode, type, or metadata before traversal",
            display.display(),
        );
    }
    Ok(File::from(fd))
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
enum PinnedSymlinkPhase<'a> {
    AfterOpen,
    AfterRead(&'a CStr),
}

#[cfg(target_os = "linux")]
fn read_snapshot_symlink_at<Fd: std::os::fd::AsFd>(
    directory: Fd,
    name: &CStr,
    expected: SnapshotStat,
    display: &Path,
) -> Result<CString> {
    let mut hook = |_: &Path, _: PinnedSymlinkPhase<'_>| Ok(());
    read_snapshot_symlink_at_with_hook(directory, name, expected, display, &mut hook)
}

/// Pin a symlink inode before reading its target. Linux supports reading an
/// `O_PATH|O_NOFOLLOW` symlink descriptor through an empty `readlinkat` path;
/// failure of that facility is terminal rather than falling back to a second
/// parent/name lookup.
#[cfg(target_os = "linux")]
fn read_snapshot_symlink_at_with_hook<Fd: std::os::fd::AsFd>(
    directory: Fd,
    name: &CStr,
    expected: SnapshotStat,
    display: &Path,
    hook: &mut dyn FnMut(&Path, PinnedSymlinkPhase<'_>) -> Result<()>,
) -> Result<CString> {
    if expected.file_type() != rustix::fs::FileType::Symlink {
        bail!(
            "source entry {} changed type before symlink pinning",
            display.display()
        );
    }
    let symlink_fd = rustix::fs::openat(
        &directory,
        name,
        rustix::fs::OFlags::PATH | rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .with_context(|| format!("pinning source symlink inode {}", display.display()))?;
    if snapshot_fd_stat(&symlink_fd)? != expected {
        bail!(
            "source symlink {} changed before its target was pinned",
            display.display(),
        );
    }
    hook(display, PinnedSymlinkPhase::AfterOpen)?;
    let target = rustix::fs::readlinkat(&symlink_fd, c"", Vec::new()).with_context(|| {
        format!(
            "reading pinned source symlink target through its descriptor {}",
            display.display()
        )
    })?;
    hook(display, PinnedSymlinkPhase::AfterRead(&target))?;
    let final_fd = snapshot_fd_stat(&symlink_fd)
        .with_context(|| format!("restating pinned source symlink {}", display.display()))?;
    let final_parent = snapshot_stat_at(directory, name).with_context(|| {
        format!(
            "restating pinned source symlink parent binding {}",
            display.display()
        )
    })?;
    if final_fd != expected || final_parent != expected {
        bail!(
            "source symlink {} changed while its target was read",
            display.display(),
        );
    }
    Ok(target)
}

#[cfg(unix)]
fn apply_descriptor_link_target(
    base: &[OsString],
    target: &Path,
    remainder: impl IntoIterator<Item = OsString>,
) -> Option<Vec<OsString>> {
    if target.is_absolute() {
        return None;
    }
    let mut result = base.to_vec();
    for component in target.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::Normal(value) => result.push(value.to_owned()),
            std::path::Component::ParentDir => {
                result.pop()?;
            }
            std::path::Component::RootDir | std::path::Component::Prefix(_) => return None,
        }
    }
    result.extend(remainder);
    Some(result)
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DescriptorResolution {
    Existing(rustix::fs::FileType),
    Missing,
    Escapes,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SnapshotVisitPhase {
    BeforeEnumeration,
    BeforeFinalValidation,
}

/// Resolve an already lexically in-tree path without reopening any source
/// pathname. Every component is looked up relative to a verified directory
/// descriptor, and every encountered symlink is expanded from that same FD.
#[cfg(target_os = "linux")]
fn resolve_snapshot_components(
    root_fd: &File,
    root: &Path,
    excluded_roots: &[PathBuf],
    initial: Vec<OsString>,
) -> Result<DescriptorResolution> {
    let mut symlink_hook = |_: &Path, _: PinnedSymlinkPhase<'_>| Ok(());
    resolve_snapshot_components_with_hook(root_fd, root, excluded_roots, initial, &mut symlink_hook)
}

#[cfg(target_os = "linux")]
fn resolve_snapshot_components_with_hook(
    root_fd: &File,
    root: &Path,
    excluded_roots: &[PathBuf],
    initial: Vec<OsString>,
    symlink_hook: &mut dyn FnMut(&Path, PinnedSymlinkPhase<'_>) -> Result<()>,
) -> Result<DescriptorResolution> {
    let mut components = initial;
    let mut followed_links = 0_u8;
    loop {
        if components.is_empty() {
            return Ok(DescriptorResolution::Existing(
                rustix::fs::FileType::Directory,
            ));
        }
        let logical = components
            .iter()
            .fold(root.to_path_buf(), |mut path, component| {
                path.push(component);
                path
            });
        if excluded_roots
            .iter()
            .any(|excluded| logical.starts_with(excluded))
        {
            return Ok(DescriptorResolution::Escapes);
        }
        let root_copy = rustix::io::dup(root_fd).context("duplicating source root descriptor")?;
        let mut directory = File::from(root_copy);
        let mut resolved = Vec::<OsString>::new();
        let mut index = 0_usize;
        while index < components.len() {
            let component = &components[index];
            let component_display = resolved.iter().fold(root.to_path_buf(), |mut path, item| {
                path.push(item);
                path
            });
            let component_display = component_display.join(component);
            let component_bytes = component.as_bytes();
            if component_bytes.is_empty()
                || component_bytes == b"."
                || component_bytes == b".."
                || component_bytes.contains(&b'/')
                || component_bytes.contains(&0)
            {
                bail!("invalid source path component during descriptor-relative resolution");
            }
            let component_c = CString::new(component_bytes)
                .context("source path component unexpectedly contained NUL")?;
            let stat = match snapshot_stat_at(&directory, &component_c) {
                Ok(stat) => stat,
                Err(error)
                    if error
                        .downcast_ref::<rustix::io::Errno>()
                        .is_some_and(|errno| {
                            *errno == rustix::io::Errno::NOENT
                                || *errno == rustix::io::Errno::NOTDIR
                        }) =>
                {
                    return Ok(DescriptorResolution::Missing);
                }
                Err(error) => return Err(error),
            };
            match stat.file_type() {
                rustix::fs::FileType::Symlink => {
                    followed_links = followed_links.saturating_add(1);
                    if followed_links > 40 {
                        return Ok(DescriptorResolution::Missing);
                    }
                    let target = read_snapshot_symlink_at_with_hook(
                        &directory,
                        &component_c,
                        stat,
                        &component_display,
                        symlink_hook,
                    )?;
                    let target = PathBuf::from(OsString::from_vec(target.as_bytes().to_vec()));
                    let Some(expanded) = apply_descriptor_link_target(
                        &resolved,
                        &target,
                        components[index + 1..].iter().cloned(),
                    ) else {
                        return Ok(DescriptorResolution::Escapes);
                    };
                    components = expanded;
                    break;
                }
                rustix::fs::FileType::Directory => {
                    if index + 1 == components.len() {
                        return Ok(DescriptorResolution::Existing(
                            rustix::fs::FileType::Directory,
                        ));
                    }
                    directory = open_snapshot_at(
                        &directory,
                        &component_c,
                        stat,
                        rustix::fs::FileType::Directory,
                        &logical,
                    )?;
                    resolved.push(component.clone());
                    index += 1;
                }
                file_type if index + 1 == components.len() => {
                    return Ok(DescriptorResolution::Existing(file_type));
                }
                _ => return Ok(DescriptorResolution::Missing),
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn should_copy_snapshot_gitdir_pointer(
    root_fd: &File,
    root: &Path,
    excluded_roots: &[PathBuf],
    pointer: &Path,
    text: &str,
) -> Result<bool> {
    let Some(target) = text
        .lines()
        .next()
        .map(str::trim)
        .and_then(|line| line.strip_prefix("gitdir:"))
        .map(str::trim)
    else {
        return Ok(true);
    };
    let target = Path::new(target);
    let parent = pointer
        .parent()
        .expect("a .git pointer always has a parent")
        .strip_prefix(root)
        .expect("a .git pointer stays below the source root")
        .components()
        .map(|component| component.as_os_str().to_owned())
        .collect::<Vec<_>>();
    let Some(components) = apply_descriptor_link_target(&parent, target, std::iter::empty()) else {
        tracing::warn!(
            pointer = %pointer.display(),
            gitdir = %target.display(),
            "omitting out-of-context .git indirection from immutable source snapshot",
        );
        return Ok(false);
    };
    if resolve_snapshot_components(root_fd, root, excluded_roots, components)?
        != DescriptorResolution::Existing(rustix::fs::FileType::Directory)
    {
        tracing::warn!(
            pointer = %pointer.display(),
            gitdir = %target.display(),
            "omitting unresolved or out-of-context .git indirection from immutable source snapshot",
        );
        return Ok(false);
    }
    Ok(true)
}

#[cfg(target_os = "linux")]
fn prepare_source_snapshot(
    source: &Path,
    out_dir: &Path,
    additional_excluded_roots: &[PathBuf],
) -> Result<PreparedSourceSnapshot> {
    let mut visit_hook = |_: &Path, _: SnapshotVisitPhase| Ok(());
    prepare_source_snapshot_with_hook(source, out_dir, additional_excluded_roots, &mut visit_hook)
}

#[cfg(target_os = "linux")]
fn prepare_source_snapshot_with_hook(
    source: &Path,
    out_dir: &Path,
    additional_excluded_roots: &[PathBuf],
    visit_hook: &mut dyn FnMut(&Path, SnapshotVisitPhase) -> Result<()>,
) -> Result<PreparedSourceSnapshot> {
    fn visit(
        root: &Path,
        root_fd: &File,
        excluded_roots: &[PathBuf],
        snapshot: &Path,
        path: &Path,
        directory: &File,
        expected_directory: SnapshotStat,
        parent_binding: Option<(&File, &CStr)>,
        hasher: &mut Sha256,
        visit_hook: &mut dyn FnMut(&Path, SnapshotVisitPhase) -> Result<()>,
    ) -> Result<()> {
        visit_hook(path, SnapshotVisitPhase::BeforeEnumeration)?;
        let mut reader = rustix::fs::Dir::read_from(directory)
            .with_context(|| format!("reading source tree from directory FD {}", path.display()))?;
        let mut entries = Vec::<CString>::new();
        for entry in &mut reader {
            let entry = entry.with_context(|| {
                format!(
                    "enumerating source tree from directory FD {}",
                    path.display()
                )
            })?;
            let bytes = entry.file_name().to_bytes();
            if bytes == b"." || bytes == b".." {
                continue;
            }
            if bytes.is_empty() || bytes.contains(&b'/') || bytes.contains(&0) {
                bail!(
                    "source directory {} returned an invalid child name",
                    path.display(),
                );
            }
            entries.push(entry.file_name().to_owned());
        }
        entries.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
        for entry_name_c in entries {
            let entry_name = OsString::from_vec(entry_name_c.as_bytes().to_vec());
            if path == root && is_managed_snapshot_output(&entry_name) {
                continue;
            }
            let entry_path = path.join(&entry_name);
            if excluded_roots
                .iter()
                .any(|excluded| entry_path.starts_with(excluded))
            {
                continue;
            }
            let metadata = snapshot_stat_at(directory, &entry_name_c)
                .with_context(|| format!("stating source entry {}", entry_path.display()))?;
            if metadata.file_type() == rustix::fs::FileType::Directory {
                let child = open_snapshot_at(
                    directory,
                    &entry_name_c,
                    metadata,
                    rustix::fs::FileType::Directory,
                    &entry_path,
                )?;
                let relative = relative_source_path(root, &entry_path)?;
                hash_snapshot_record(hasher, b'd', relative, metadata.snapshot_mode());
                let destination = snapshot.join(relative);
                std::fs::create_dir(&destination).with_context(|| {
                    format!(
                        "creating source snapshot directory {}",
                        destination.display()
                    )
                })?;
                visit(
                    root,
                    root_fd,
                    excluded_roots,
                    snapshot,
                    &entry_path,
                    &child,
                    metadata,
                    Some((directory, &entry_name_c)),
                    hasher,
                    visit_hook,
                )?;
                std::fs::set_permissions(&destination, metadata.permissions()).with_context(
                    || format!("preserving source directory mode {}", destination.display()),
                )?;
                normalize_snapshot_times(&destination)?;
            } else if metadata.file_type() == rustix::fs::FileType::Symlink {
                let target =
                    read_snapshot_symlink_at(directory, &entry_name_c, metadata, &entry_path)?;
                let target = PathBuf::from(OsString::from_vec(target.as_bytes().to_vec()));
                let normalized =
                    validate_snapshot_symlink(root, excluded_roots, &entry_path, &target)?;
                if resolve_snapshot_components(root_fd, root, excluded_roots, normalized)?
                    == DescriptorResolution::Escapes
                {
                    bail!(
                        "source-build symlink {} resolves outside the source tree via target {}",
                        entry_path.display(),
                        target.display(),
                    );
                }
                let relative = relative_source_path(root, &entry_path)?;
                let target_text = target.to_str().ok_or_else(|| {
                    anyhow!(
                        "source-build symlink targets must be UTF-8: {} -> {}",
                        entry_path.display(),
                        target.display(),
                    )
                })?;
                hash_snapshot_record(hasher, b'l', relative, metadata.snapshot_mode());
                hasher.update((target_text.len() as u64).to_be_bytes());
                hasher.update(target_text.as_bytes());
                let destination = snapshot.join(relative);
                create_snapshot_symlink(&target, &destination).with_context(|| {
                    format!(
                        "copying source symlink {} -> {}",
                        destination.display(),
                        target.display(),
                    )
                })?;
                let final_link = snapshot_stat_at(directory, &entry_name_c).with_context(|| {
                    format!("restating source symlink {}", entry_path.display())
                })?;
                if final_link != metadata {
                    bail!(
                        "source symlink {} changed while its build snapshot was prepared",
                        entry_path.display(),
                    );
                }
            } else if metadata.file_type() == rustix::fs::FileType::RegularFile {
                let relative = relative_source_path(root, &entry_path)?;
                let mut input = open_snapshot_at(
                    directory,
                    &entry_name_c,
                    metadata,
                    rustix::fs::FileType::RegularFile,
                    &entry_path,
                )?;
                let omit_git_pointer = if entry_name.as_bytes() == b".git" {
                    let mut text = String::new();
                    input.read_to_string(&mut text).with_context(|| {
                        format!("reading Git indirection file {}", entry_path.display())
                    })?;
                    input.rewind().with_context(|| {
                        format!("rewinding Git indirection file {}", entry_path.display())
                    })?;
                    !should_copy_snapshot_gitdir_pointer(
                        root_fd,
                        root,
                        excluded_roots,
                        &entry_path,
                        &text,
                    )?
                } else {
                    false
                };
                if omit_git_pointer {
                    let final_metadata = snapshot_fd_stat(&input).with_context(|| {
                        format!("restating omitted Git indirection {}", entry_path.display())
                    })?;
                    let final_parent_metadata = snapshot_stat_at(directory, &entry_name_c)
                        .with_context(|| {
                            format!(
                                "restating omitted Git indirection binding {}",
                                entry_path.display()
                            )
                        })?;
                    if final_metadata != metadata || final_parent_metadata != metadata {
                        bail!(
                            "source file {} changed while its build snapshot was prepared",
                            entry_path.display(),
                        );
                    }
                    // Bind the deliberate omission itself into the tree hash.
                    // Static-version projects continue to build; SCM-dependent
                    // projects fail closed instead of following mutable host
                    // metadata outside the captured context.
                    hash_snapshot_record(hasher, b'o', relative, 0);
                    hasher.update(b"retread-external-gitdir-omitted-v1\0");
                    continue;
                }
                let expected_size = metadata.file_size(&entry_path)?;
                hash_snapshot_record(hasher, b'f', relative, metadata.snapshot_mode());
                hasher.update(expected_size.to_be_bytes());
                let destination = snapshot.join(relative);
                let mut output = File::create(&destination).with_context(|| {
                    format!("creating source snapshot file {}", destination.display())
                })?;
                let mut copied = 0_u64;
                let mut buffer = [0_u8; 64 * 1024];
                loop {
                    let read = input
                        .read(&mut buffer)
                        .with_context(|| format!("reading source file {}", entry_path.display()))?;
                    if read == 0 {
                        break;
                    }
                    hasher.update(&buffer[..read]);
                    output.write_all(&buffer[..read]).with_context(|| {
                        format!("writing source snapshot {}", destination.display())
                    })?;
                    copied += read as u64;
                }
                if copied != expected_size {
                    bail!(
                        "source file {} changed size while its build snapshot was prepared",
                        entry_path.display(),
                    );
                }
                let final_metadata = snapshot_fd_stat(&input).with_context(|| {
                    format!("restating copied source file {}", entry_path.display())
                })?;
                let final_parent_metadata = snapshot_stat_at(directory, &entry_name_c)
                    .with_context(|| {
                        format!("restating source file binding {}", entry_path.display())
                    })?;
                if final_metadata != metadata || final_parent_metadata != metadata {
                    bail!(
                        "source file {} changed while its build snapshot was prepared",
                        entry_path.display(),
                    );
                }
                output.flush()?;
                std::fs::set_permissions(&destination, metadata.permissions()).with_context(
                    || format!("preserving source file mode {}", destination.display()),
                )?;
                normalize_snapshot_times(&destination)?;
            } else {
                bail!(
                    "source-build tree contains unsupported special file {}",
                    entry_path.display(),
                );
            }
        }
        visit_hook(path, SnapshotVisitPhase::BeforeFinalValidation)?;
        let final_opened = snapshot_fd_stat(directory)
            .with_context(|| format!("restating traversed source directory {}", path.display()))?;
        let final_parent = match parent_binding {
            Some((parent, name)) => Some(snapshot_stat_at(parent, name).with_context(|| {
                format!("restating source directory binding {}", path.display())
            })?),
            None => None,
        };
        if final_opened != expected_directory
            || final_parent.is_some_and(|metadata| metadata != expected_directory)
        {
            bail!(
                "source directory {} changed while its build snapshot was prepared",
                path.display(),
            );
        }
        Ok(())
    }

    let source = std::fs::canonicalize(source)
        .with_context(|| format!("canonicalizing source tree {}", source.display()))?;
    if !source.is_dir() {
        bail!("source-build path is not a directory: {}", source.display());
    }
    let snapshot_parent = crate::courier::retread_cache_root()
        .join("source-snapshots")
        .join(LOCAL_SOURCE_SNAPSHOT_VERSION);
    std::fs::create_dir_all(&snapshot_parent).with_context(|| {
        format!(
            "creating local-source snapshot parent {}",
            snapshot_parent.display(),
        )
    })?;
    let output = canonicalize_future_path(out_dir)?;
    let cache_root = canonicalize_future_path(&crate::courier::retread_cache_root())?;
    let mut candidates = vec![output, cache_root];
    for excluded in additional_excluded_roots {
        candidates.push(canonicalize_future_path(excluded)?);
    }
    let mut excluded_roots = candidates
        .into_iter()
        .filter(|path| path.starts_with(&source))
        .collect::<Vec<_>>();
    excluded_roots.sort();
    excluded_roots.dedup();
    let directory = unique_staging_dir(&snapshot_parent.join("source"))?;
    let root_metadata = rustix::fs::statat(
        rustix::fs::CWD,
        &source,
        rustix::fs::AtFlags::SYMLINK_NOFOLLOW,
    )
    .map(|stat| SnapshotStat::from_rustix(&stat))
    .with_context(|| format!("stating source root {}", source.display()))?;
    if root_metadata.file_type() != rustix::fs::FileType::Directory {
        bail!("source-build path is not a directory: {}", source.display());
    }
    let root_fd = rustix::fs::openat(
        rustix::fs::CWD,
        &source,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::CLOEXEC
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::DIRECTORY,
        rustix::fs::Mode::empty(),
    )
    .map(File::from)
    .with_context(|| format!("opening source root directory {}", source.display()))?;
    if snapshot_fd_stat(&root_fd)? != root_metadata {
        bail!(
            "source root {} changed before descriptor-relative traversal",
            source.display(),
        );
    }
    let mut hasher = Sha256::new();
    hasher.update(b"retread-local-source-snapshot-v5\0");
    hash_snapshot_record(&mut hasher, b'r', "", root_metadata.snapshot_mode());
    visit(
        &source,
        &root_fd,
        &excluded_roots,
        directory.0.as_path(),
        &source,
        &root_fd,
        root_metadata,
        None,
        &mut hasher,
        visit_hook,
    )?;
    let final_root = rustix::fs::statat(
        rustix::fs::CWD,
        &source,
        rustix::fs::AtFlags::SYMLINK_NOFOLLOW,
    )
    .map(|stat| SnapshotStat::from_rustix(&stat))
    .with_context(|| format!("restating source root binding {}", source.display()))?;
    if snapshot_fd_stat(&root_fd)? != root_metadata || final_root != root_metadata {
        bail!(
            "source root {} changed while its build snapshot was prepared",
            source.display(),
        );
    }
    std::fs::set_permissions(&directory.0, root_metadata.permissions()).with_context(|| {
        format!(
            "preserving source root mode on snapshot {}",
            directory.0.display(),
        )
    })?;
    normalize_snapshot_times(&directory.0)?;
    Ok(PreparedSourceSnapshot {
        workspace: Arc::new(PreparedSourceWorkspace {
            directory,
            _workspace_lock: None,
        }),
        identity: format!("{:x}", hasher.finalize()),
    })
}

#[cfg(not(target_os = "linux"))]
fn prepare_source_snapshot(
    source: &Path,
    out_dir: &Path,
    additional_excluded_roots: &[PathBuf],
) -> Result<PreparedSourceSnapshot> {
    let _ = (source, out_dir, additional_excluded_roots);
    bail!(
        "secure source snapshots require descriptor-relative no-follow traversal, which is unsupported on this platform"
    )
}

fn has_external_gitdir_pointer(source_root: &Path) -> Result<bool> {
    let pointer = source_root.join(".git");
    let metadata = match std::fs::symlink_metadata(&pointer) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Ok(false);
    }
    let text = std::fs::read_to_string(&pointer)
        .with_context(|| format!("reading Git indirection file {}", pointer.display()))?;
    let Some(target) = text
        .lines()
        .next()
        .map(str::trim)
        .and_then(|line| line.strip_prefix("gitdir:"))
        .map(str::trim)
    else {
        return Ok(false);
    };
    let target = Path::new(target);
    let resolved = if target.is_absolute() {
        std::fs::canonicalize(target)
    } else {
        std::fs::canonicalize(source_root.join(target))
    }
    .with_context(|| {
        format!(
            "resolving external Git metadata for path source {}",
            source_root.display()
        )
    })?;
    Ok(!resolved.starts_with(source_root))
}

async fn path_git_top_level(source_root: &Path) -> Result<Option<PathBuf>> {
    let has_git_marker = source_root
        .ancestors()
        .any(|ancestor| std::fs::symlink_metadata(ancestor.join(".git")).is_ok());
    if !has_git_marker {
        return Ok(None);
    }
    let top_level = run_output(
        Command::new("git")
            .args(["rev-parse", "--show-toplevel"])
            .env("GIT_OPTIONAL_LOCKS", "0")
            .current_dir(source_root),
        "git locate path-source root",
    )
    .await?;
    Ok(Some(std::fs::canonicalize(top_level.trim()).with_context(
        || {
            format!(
                "canonicalizing Git top-level `{}` for path source {}",
                top_level.trim(),
                source_root.display(),
            )
        },
    )?))
}

async fn run_readonly_git_status(cmd: &mut Command, label: &str) -> Result<std::process::Output> {
    cmd.env("GIT_OPTIONAL_LOCKS", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .with_context(|| format!("spawning {label}"))
}

async fn external_path_git_state(source_root: &Path) -> Result<Option<ExternalPathGitState>> {
    let external_pointer = has_external_gitdir_pointer(source_root)?;
    let Some(top_level) = path_git_top_level(source_root).await? else {
        return Ok(None);
    };
    if top_level != source_root {
        bail!(
            "path source {} has Git top-level {} outside its immutable snapshot context; retread cannot preserve sibling/ancestor SCM metadata",
            source_root.display(),
            top_level.display(),
        );
    }
    if !external_pointer {
        return Ok(None);
    }
    let symbolic = run_readonly_git_status(
        Command::new("git")
            .args(["symbolic-ref", "-q", "HEAD"])
            .current_dir(source_root),
        "git inspect external path-source HEAD",
    )
    .await?;
    match symbolic.status.code() {
        Some(0) => bail!(
            "external path-source Git HEAD is attached to `{}`; use an exact detached worktree so branch movement is not hidden from the snapshot identity",
            String::from_utf8_lossy(&symbolic.stdout).trim(),
        ),
        Some(1) => {}
        _ => bail!(
            "git inspect external path-source HEAD failed (status {}): {}",
            symbolic.status,
            String::from_utf8_lossy(&symbolic.stderr).trim(),
        ),
    }
    let staged = run_readonly_git_status(
        Command::new("git")
            .args(["diff", "--cached", "--quiet", "--exit-code", "HEAD", "--"])
            .current_dir(source_root),
        "git inspect external path-source index",
    )
    .await?;
    match staged.status.code() {
        Some(0) => {}
        Some(1) => bail!(
            "external path-source Git index contains staged changes that cannot be represented by a detached metadata snapshot"
        ),
        _ => bail!(
            "git inspect external path-source index failed (status {}): {}",
            staged.status,
            String::from_utf8_lossy(&staged.stderr).trim(),
        ),
    }
    let shallow = run_output(
        Command::new("git")
            .args(["rev-parse", "--is-shallow-repository"])
            .env("GIT_OPTIONAL_LOCKS", "0")
            .current_dir(source_root),
        "git inspect external path-source shallow state",
    )
    .await?;
    if shallow.trim() != "false" {
        bail!(
            "external path-source Git repository is shallow; its incomplete history cannot provide a stable SCM view"
        );
    }
    let resolved_sha = run_output(
        Command::new("git")
            .args(["rev-parse", "--verify", "HEAD^{commit}"])
            .env("GIT_OPTIONAL_LOCKS", "0")
            .current_dir(source_root),
        "git resolve external path-source HEAD",
    )
    .await?
    .trim()
    .to_ascii_lowercase();
    if !matches!(resolved_sha.len(), 40 | 64)
        || !resolved_sha.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("external path-source Git HEAD is not an exact commit: {resolved_sha}");
    }
    let ref_state = canonical_git_ref_state(source_root).await?;
    ensure_no_canonical_gitlinks(source_root, &resolved_sha).await?;
    Ok(Some(ExternalPathGitState {
        resolved_sha,
        ref_state,
    }))
}

async fn attach_external_path_git_metadata(
    source_root: &Path,
    snapshot: &PreparedSourceSnapshot,
    state: &ExternalPathGitState,
) -> Result<()> {
    if snapshot.root().join(".git").exists() {
        bail!(
            "external path-source snapshot unexpectedly retained a live .git indirection: {}",
            snapshot.root().display(),
        );
    }
    let staging_parent = crate::courier::retread_cache_root()
        .join("path-git-metadata")
        .join("v1");
    std::fs::create_dir_all(&staging_parent).with_context(|| {
        format!(
            "creating path-source Git metadata staging parent {}",
            staging_parent.display()
        )
    })?;
    let staging = unique_staging_dir(&staging_parent.join("metadata"))?;
    let repo = staging.0.join("repo");
    run_silent(
        Command::new("git")
            .args(["clone", "--no-local", "--no-checkout", "--"])
            .arg(source_root)
            .arg(&repo),
        "git clone path-source metadata",
    )
    .await?;
    run_silent(
        Command::new("git")
            .args(["fetch", "--force", "--tags", "origin"])
            .current_dir(&repo),
        "git fetch path-source tags",
    )
    .await?;
    run_silent(
        Command::new("git")
            .args(["fetch", "--force", "origin", &state.resolved_sha])
            .current_dir(&repo),
        "git fetch path-source commit",
    )
    .await?;
    run_silent(
        Command::new("git")
            .args(["checkout", "--detach", "--force", &state.resolved_sha])
            .current_dir(&repo),
        "git checkout path-source metadata",
    )
    .await?;
    delete_canonical_git_refs(&repo, "refs/heads").await?;
    delete_canonical_git_refs(&repo, "refs/remotes").await?;
    run_silent(
        Command::new("git")
            .args(["remote", "remove", "origin"])
            .current_dir(&repo),
        "git remove live path-source origin",
    )
    .await?;
    let actual_ref_state = canonical_git_ref_state(&repo).await?;
    if actual_ref_state != state.ref_state {
        bail!(
            "path-source Git tag/ref state changed during snapshot: expected {}, found {}",
            state.ref_state,
            actual_ref_state,
        );
    }
    ensure_no_canonical_gitlinks(&repo, &state.resolved_sha).await?;
    let git_dir = repo.join(".git");
    sanitize_canonical_git_metadata(&repo)?;
    let snapshot_git_dir = snapshot.root().join(".git");
    let writable_root = TemporaryWritableDirectory::new(snapshot.root())?;
    std::fs::rename(&git_dir, &snapshot_git_dir).with_context(|| {
        format!(
            "attaching self-contained Git metadata {}",
            snapshot_git_dir.display()
        )
    })?;
    normalize_source_tree_times(&snapshot_git_dir)?;
    // Attaching a child changes the parent directory mtime. Restore the same
    // deterministic epoch used by the original source snapshot before
    // restoring its exact permissions.
    normalize_snapshot_times(snapshot.root())?;
    let snapshot_head = run_output(
        Command::new("git")
            .args(["rev-parse", "--verify", "HEAD^{commit}"])
            .env("GIT_OPTIONAL_LOCKS", "0")
            .current_dir(snapshot.root()),
        "git validate attached path-source metadata",
    )
    .await?;
    if snapshot_head.trim().to_ascii_lowercase() != state.resolved_sha {
        bail!("attached path-source Git metadata resolved the wrong HEAD");
    }
    writable_root.restore()?;
    Ok(())
}

/// Build a wheel from a local source tree using `uv pip wheel --no-deps`.
///
/// `--no-deps` is the critical flag: it stops the build from fetching the
/// project's transitive runtime dependencies (which for things like
/// isaaclab is GBs of torch + CUDA wheels). retread already handles the
/// dependency story via the bundle + conda emission; we only need the
/// raw wheel for METADATA inspection and packaging.
///
/// `python_version` (e.g. "3.11", "3.13") tells uv which interpreter to
/// use for the build. uv downloads python-build-standalone on demand
/// (cached under `~/.cache/uv/python/`) so retread itself doesn't need
/// to ship any python -- ANY python version the user asks for works
/// without rebuilding retread.
///
/// Returns the path to the produced `.whl` (inside `out_dir`).
pub async fn build_wheel_from_path(
    source: &Path,
    out_dir: &Path,
    python_version: &str,
) -> Result<PathBuf> {
    let target = ResolutionTarget::for_subdir(
        &normalized_python_minor(python_version)?.version(),
        crate::glibc::current_pixi_platform(),
    );
    Ok(
        build_wheel_from_path_for_target(source, out_dir, &target, None, None, None)
            .await?
            .wheel_path,
    )
}

pub(crate) async fn build_wheel_from_path_for_target(
    source: &Path,
    out_dir: &Path,
    target: &ResolutionTarget,
    expected: Option<&ExpectedWheel>,
    managed_output_root: Option<&Path>,
    context_root: Option<&Path>,
) -> Result<PathWheelBuild> {
    let python = normalized_python_minor(target.python_version())?;
    let canonical_source = std::fs::canonicalize(source)
        .with_context(|| format!("canonicalizing source project {}", source.display()))?;
    let candidate_context = match context_root {
        Some(context) => std::fs::canonicalize(context)
            .with_context(|| format!("canonicalizing source context {}", context.display()))?,
        None => canonical_source.clone(),
    };
    // Relative entries may intentionally reach a sibling Git submodule via
    // `../..`. Select the nearest containing standalone Git root when one
    // contains both the project and declared source context; that keeps a
    // submodule's relative `.git/modules/...` pointer self-contained. Outside
    // a shared repository, retain the declared parent only when it contains
    // the project, otherwise snapshot the external project itself.
    let canonical_context = select_path_source_context(&canonical_source, &candidate_context);
    let project_relative = canonical_source
        .strip_prefix(&canonical_context)
        .with_context(|| {
            format!(
                "source project {} is outside declared context {}",
                canonical_source.display(),
                canonical_context.display(),
            )
        })?
        .to_path_buf();
    let external_git_state = external_path_git_state(&canonical_context).await?;
    let context_for_git_metadata = canonical_context.clone();
    let prepared = tokio::task::spawn_blocking({
        let source = canonical_context;
        let out_dir = out_dir.to_path_buf();
        let excluded = managed_output_root
            .map(Path::to_path_buf)
            .into_iter()
            .collect::<Vec<_>>();
        move || prepare_source_snapshot(&source, &out_dir, &excluded)
    })
    .await
    .context("source-tree snapshot task panicked")??;
    if let Some(state) = &external_git_state {
        // Replace the deliberately omitted live `.git` pointer with a private,
        // self-contained detached metadata store. SCM-aware backends therefore
        // observe exact HEAD/tags without following mutable superproject state.
        attach_external_path_git_metadata(&context_for_git_metadata, &prepared, state).await?;
    }
    let project_relative_text = project_relative
        .to_str()
        .ok_or_else(|| anyhow!("source project path relative to its context is not UTF-8"))?;
    let source_identity = hash_fields(
        b"retread-path-wheel-source-v6\0",
        &[
            prepared.identity.as_bytes(),
            project_relative_text.as_bytes(),
            external_git_state
                .as_ref()
                .map(|state| state.resolved_sha.as_bytes())
                .unwrap_or_default(),
            external_git_state
                .as_ref()
                .map(|state| state.ref_state.as_bytes())
                .unwrap_or_default(),
        ],
    );
    let prepared = stabilize_source_snapshot_workspace(prepared, "path", &source_identity).await?;
    let pristine_source = prepared.root().join(&project_relative);
    if !pristine_source.is_dir() {
        bail!(
            "source snapshot lost project subdirectory `{}`",
            project_relative.display(),
        );
    }
    tracing::info!(
        source = %source.display(),
        python = %python.version(),
        target = %target.conda_subdir(),
        "building wheel via uv build --wheel (this can take a minute; uv downloads python if missing)",
    );
    // `uv build --wheel`: build the project at `source` into a wheel.
    // (uv doesn't expose `uv pip wheel` -- the PEP 517 build pipeline
    // lives under the top-level `uv build` command.) `--python <ver>`
    // tells uv which interpreter to use; `UV_PYTHON_DOWNLOADS=automatic`
    // enables auto-fetching of python-build-standalone binaries when
    // the requested version isn't installed locally. uv build only
    // builds the project's own wheel -- it doesn't fetch runtime deps
    // -- so no equivalent of pip's `--no-deps` is needed.
    let pristine_workspace = Arc::clone(&prepared.workspace);
    let project_relative_for_build = project_relative.clone();
    let wheel_path = cached_build(
        "path",
        &source_identity,
        target,
        out_dir,
        expected,
        move |private_out| async move {
            // A PEP 517 backend may create egg-info/build/generated files in
            // its source directory. Give it a disposable copy and retain the
            // pristine hashed workspace for phase-1.5 injection, so cache miss
            // and cache hit derive the same final wheel.
            let disposable = tokio::task::spawn_blocking({
                let pristine_workspace = Arc::clone(&pristine_workspace);
                let out_dir = private_out.clone();
                move || prepare_source_snapshot(&pristine_workspace.directory.0, &out_dir, &[])
            })
            .await
            .context("disposable path-source copy task panicked")??;
            let build_source = disposable.root().join(&project_relative_for_build);
            if !build_source.is_dir() {
                bail!(
                    "disposable source build lost project subdirectory `{}`",
                    project_relative_for_build.display(),
                );
            }
            let py_arg = format!("--python={}", python.identity());
            let out_arg = format!("--out-dir={}", private_out.display());
            run_capturing_uv(&[
                "build",
                "--wheel",
                &py_arg,
                &out_arg,
                &build_source.display().to_string(),
            ])
            .await
        },
    )
    .await?;
    Ok(PathWheelBuild {
        wheel_path,
        project_root: prepared.root().join(project_relative),
        _source_snapshot: prepared,
    })
}

fn select_path_source_context(source: &Path, declared_context: &Path) -> PathBuf {
    if source.starts_with(declared_context) {
        declared_context.to_path_buf()
    } else {
        source.to_path_buf()
    }
}

/// v0.18.0+: download a PyPI sdist (`.tar.gz` / `.zip`) and run
/// `uv build --wheel` on it. Used as the BFS fallback when a dep is
/// sdist-only on PyPI (gym, classic-control, ...). Output is a normal
/// wheel that re-enters the bundle pipeline.
///
/// `out_dir` should be per-entry (e.g. `<pack>/wheels/<entry>/`) so
/// cache reuse and cleanup match what other materialize paths do.
/// Returns the path to the built wheel.
pub async fn build_wheel_from_sdist_url(
    sdist_url: &url::Url,
    out_dir: &Path,
    python_version: &str,
    expected_sha256: Option<&str>,
) -> Result<PathBuf> {
    let target =
        ResolutionTarget::try_for_subdir(python_version, crate::glibc::current_pixi_platform())?;
    Ok(
        build_wheel_from_sdist_url_for_target(sdist_url, out_dir, &target, expected_sha256, None)
            .await?
            .wheel_path,
    )
}

fn sdist_filename(url: &url::Url) -> Result<String> {
    let encoded = url
        .path_segments()
        .and_then(|mut segments| segments.next_back())
        .filter(|filename| !filename.is_empty())
        .ok_or_else(|| anyhow!("sdist URL {url} has no filename component"))?;
    let decoded = percent_encoding::percent_decode_str(encoded)
        .decode_utf8()
        .context("sdist filename is not UTF-8")?
        .into_owned();
    if Path::new(&decoded)
        .file_name()
        .and_then(|name| name.to_str())
        != Some(decoded.as_str())
    {
        bail!("sdist URL {url} has an unsafe filename component");
    }
    Ok(decoded)
}

fn sdist_source_identity(content_sha256: &str, filename: &str) -> String {
    hash_fields(
        b"retread-sdist-source-v4\0",
        &[content_sha256.as_bytes(), filename.as_bytes()],
    )
}

pub(crate) fn sdist_advertised_sha256(
    url: &url::Url,
    advertised: Option<&str>,
) -> Result<Option<String>> {
    let explicit = advertised
        .map(|value| validate_sha256(value, "advertised sdist hash"))
        .transpose()?;
    let mut fragment_hash = None;
    if let Some(fragment) = url.fragment() {
        for (key, value) in url::form_urlencoded::parse(fragment.as_bytes()) {
            if key.eq_ignore_ascii_case("sha256") {
                let value = validate_sha256(&value, "sdist URL fragment hash")?;
                if fragment_hash.replace(value.clone()).is_some() {
                    bail!("sdist URL contains more than one sha256 fragment");
                }
            }
        }
    }
    if let (Some(explicit), Some(fragment)) = (&explicit, &fragment_hash)
        && explicit != fragment
    {
        bail!(
            "sdist hash disagreement: index supplied `{explicit}` but URL fragment supplied `{fragment}`"
        );
    }
    Ok(explicit.or(fragment_hash))
}

async fn download_sdist(url: &url::Url, expected_sha256: Option<&str>) -> Result<Vec<u8>> {
    tracing::info!(url = %url, "downloading sdist for source build");
    let bytes = reqwest::get(url.clone())
        .await
        .with_context(|| format!("downloading sdist {url}"))?
        .error_for_status()
        .with_context(|| format!("sdist HTTP error for {url}"))?
        .bytes()
        .await
        .with_context(|| format!("reading sdist body from {url}"))?
        .to_vec();
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if let Some(expected) = expected_sha256
        && actual != expected
    {
        bail!("sdist sha256 mismatch for {url}: expected {expected}, got {actual}");
    }
    Ok(bytes)
}

pub(crate) async fn build_wheel_from_sdist_url_for_target(
    sdist_url: &url::Url,
    out_dir: &Path,
    target: &ResolutionTarget,
    advertised_sha256: Option<&str>,
    expected: Option<&ExpectedWheel>,
) -> Result<SdistWheelBuild> {
    let python = normalized_python_minor(target.python_version())?;
    let filename = sdist_filename(sdist_url)?;
    let advertised_sha256 = sdist_advertised_sha256(sdist_url, advertised_sha256)?;
    // A hash-bearing source has an exact cache identity before any network
    // access.  An unhashed source must be fetched to discover its content key;
    // foreign targets are rejected before that fetch because they cannot be
    // built natively on this host.
    let prefetched = if advertised_sha256.is_none() {
        if !native_build_allowed(target) {
            return Err(source_build_refusal_error(target));
        }
        Some(download_sdist(sdist_url, None).await?)
    } else {
        None
    };
    let content_sha = match (&advertised_sha256, &prefetched) {
        (Some(hash), _) => hash.clone(),
        (None, Some(bytes)) => format!("{:x}", Sha256::digest(bytes)),
        (None, None) => unreachable!("unhashed sdist was prefetched"),
    };
    // The archive basename is part of the build input: backends/uv use its
    // extension to select unpacking behavior. Identical bytes presented as a
    // `.zip` and `.tar.gz` must not share a warm artifact entry.
    let source_identity = sdist_source_identity(&content_sha, &filename);
    let url = sdist_url.clone();
    let filename_for_build = filename.clone();
    let expected_content_sha = content_sha.clone();
    let wheel_path = cached_build(
        "sdist",
        &source_identity,
        target,
        out_dir,
        expected,
        move |private_out| async move {
            let bytes = match prefetched {
                Some(bytes) => bytes,
                None => download_sdist(&url, Some(&expected_content_sha)).await?,
            };
            let sdist_path = private_out.join(&filename_for_build);
            tokio::fs::write(&sdist_path, &bytes)
                .await
                .with_context(|| format!("writing private sdist {}", sdist_path.display()))?;
            let py_arg = format!("--python={}", python.identity());
            let out_arg = format!("--out-dir={}", private_out.display());
            run_capturing_uv(&[
                "build",
                "--wheel",
                &py_arg,
                &out_arg,
                &sdist_path.display().to_string(),
            ])
            .await
        },
    )
    .await?;

    // DETERMINISM GUARD (Amendment 3): detect non-reproducible setuptools_scm
    // versions, mirroring the identical guard in build_wheel_from_git.
    // A wheel whose filename contains .devN, .dYYYYMMDD, or +g<sha> was built
    // without a release tag — its version/filename will DRIFT across calendar
    // days even when the sdist URL is pinned, causing lock drift on replay.
    // For static released versions (e.g. gym 0.26.2) this is a silent no-op.
    if wheel_path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(is_nondeterministic_version)
    {
        tracing::warn!(
            sdist_url = %sdist_url,
            filename = %wheel_path.display(),
            "sdist-built wheel has a non-reproducible setuptools_scm version \
             (contains .devN, .dYYYYMMDD, or +g<sha>). The wheel filename \
             will DRIFT across calendar days even when the sdist URL is \
             pinned, causing lock drift on replay. Fix: ensure the sdist's \
             build backend emits a static release version, or set \
             SETUPTOOLS_SCM_PRETEND_VERSION=<version> in the build env.",
        );
    }

    Ok(SdistWheelBuild {
        wheel_path,
        sdist_sha256: content_sha,
    })
}

/// Shared cross-pack cache family for built git wheels, keyed by
/// (repo url, resolved commit sha, subdirectory, python version).
///
/// Layout: `<retread cache root>/built-wheels/git/v3/<artifact-target-sha256>/
/// <family-sha256>/<ref-state-sha256>/{artifact.json,<raw>.whl}`. The leaf
/// additionally binds the canonical tag/ref state visible to SCM-aware build
/// backends. All identity components use the complete 64 hexadecimal
/// characters; earlier cache layouts are never consulted.
/// Every pack that pins the same (url, rev, subdir) reuses ONE build --
/// previously each pack rebuilt identical wheels into its own
/// `pypi-packs/<pack>/wheels/<entry>/` dir (isaac-pack and
/// isaac-pack-latest both building IsaacLab's 15-member group at the same
/// rev was the motivating case; ~30-60 s of `uv build` per member).
pub fn git_wheel_cache_dir(
    url: &str,
    sha: &str,
    subdirectory: &str,
    python_version: &str,
) -> PathBuf {
    let normalized = normalized_python_minor(python_version)
        .expect("git_wheel_cache_dir requires numeric MAJOR.MINOR[.PATCH]")
        .version();
    let target = ResolutionTarget::for_subdir(&normalized, crate::glibc::current_pixi_platform());
    let family_identity = git_wheel_family_identity(url, sha, subdirectory);
    built_wheel_cache_dir("git", &family_identity, &target)
}

fn git_wheel_family_identity(url: &str, sha: &str, subdirectory: &str) -> String {
    hash_fields(
        b"retread-git-wheel-family-v4\0",
        &[url.as_bytes(), sha.as_bytes(), subdirectory.as_bytes()],
    )
}

fn git_wheel_source_identity(family_identity: &str, ref_state: &str) -> String {
    format!("{family_identity}/{ref_state}")
}

fn canonical_git_repository_identity(url: &str, sha: &str) -> String {
    hash_fields(
        b"retread-canonical-git-repository-v1\0",
        &[url.as_bytes(), sha.as_bytes()],
    )
}

fn normalize_git_subdirectory(subdirectory: &str) -> Result<PathBuf> {
    use std::path::Component;

    let mut normalized = PathBuf::new();
    for component in Path::new(subdirectory).components() {
        match component {
            Component::CurDir => {}
            Component::Normal(component) => normalized.push(component),
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!(
                    "git wheel subdirectory `{subdirectory}` must be a relative path without `..`"
                );
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        normalized.push(".");
    }
    Ok(normalized)
}

fn confined_git_source_dir(clone_dir: &Path, subdirectory: &Path) -> Result<PathBuf> {
    let clone_root = std::fs::canonicalize(clone_dir)
        .with_context(|| format!("canonicalizing git checkout {}", clone_dir.display()))?;
    let candidate = clone_root.join(subdirectory);
    let resolved = std::fs::canonicalize(&candidate).with_context(|| {
        format!(
            "git wheel subdirectory `{}` not found in clone at {}",
            subdirectory.display(),
            clone_root.display(),
        )
    })?;
    if !resolved.starts_with(&clone_root) {
        bail!(
            "git wheel subdirectory `{}` resolves outside checkout {}",
            subdirectory.display(),
            clone_root.display(),
        );
    }
    if !resolved.is_dir() {
        bail!(
            "git wheel subdirectory `{}` is not a directory",
            subdirectory.display(),
        );
    }
    Ok(resolved)
}

/// Compute a git wheel entry's source directory without accessing it.
///
/// This is path derivation only; callers that read the returned tree must hold
/// the checkout lease obtained by this module's build/checkout boundary.
pub fn git_source_root(url: &str, rev: &str, subdirectory: &str, cache_dir: &Path) -> PathBuf {
    git_checkout_root(url, rev, cache_dir).join(subdirectory)
}

/// Compute the on-disk *checkout* directory for a (url, rev) pair --
/// the parent of each wheel entry's source subdirectory. Used as a pure
/// identity/grouping key by auto-data planning so the WHOLE upstream repo (minus
/// `.gitignore`'d paths and minus subdirectories already shipped as
/// wheels by sibling entries in the same bundle) can ride along into
/// the conda env at `$PREFIX/lib/<rel>`.
///
/// This function performs no synchronization and confers no permission to read
/// the returned path. Checkout consumers must retain the [`GitCheckout`] lease
/// returned by the internal checkout/build boundary.
///
/// Layout: cache_dir / retread-git-clones / v3 / <slug> /
/// <full-sha256> / ... -- a HIERARCHY rather than a single flat dirname.
/// This is what pip/uv do (the wheel itself stays a normal PEP 427
/// filename; disambiguation rides in parent directories). Each path
/// component is independently bounded:
///   - `<slug>`: repo-name slug, truncated to 24 chars
///   - `<full-sha256>`: all 64 hex chars of sha256(url + "\0" + rev)
///
/// Previously the (slug + raw 40-char git SHA) was flattened into one
/// 60+ char component; combined with the rattler cache prefix and
/// deep upstream repo internals (IsaacLab's nested test/snapshot
/// trees), full pathnames pushed against PATH_MAX and triggered
/// ENAMETOOLONG on git checkout. Splitting into a hierarchy also lets
/// multiple revs of the same repo share a parent dir, which is nicer
/// for inspection. Hashing the rev also kills any chance of `/`,
/// `@`, or `:` from a branch-name rev leaking into the on-disk path.
pub fn git_checkout_root(url: &str, rev: &str, cache_dir: &Path) -> PathBuf {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    hasher.update(b"\0");
    hasher.update(rev.as_bytes());
    let digest = hasher.finalize();
    let full_sha: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    let mut slug = git_slug(url);
    // The slug strips `https___github.com_`; cap whatever's left so
    // big-org/long-name repos don't blow the slug component.
    slug.truncate(24);
    cache_dir
        .join("retread-git-clones")
        .join(CHECKOUT_CACHE_VERSION)
        .join(slug)
        .join(full_sha)
}

const CHECKOUT_READY_MARKER: &str = ".retread-checkout-ready-v1";
const CHECKOUT_LOCK_POLL: Duration = Duration::from_millis(10);

#[cfg(test)]
static CHECKOUT_REPAIRS: OnceLock<Mutex<HashMap<PathBuf, usize>>> = OnceLock::new();

#[cfg(test)]
fn record_checkout_repair(clone_dir: &Path) {
    let mut repairs = CHECKOUT_REPAIRS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("checkout-repair test counter poisoned");
    *repairs.entry(clone_dir.to_path_buf()).or_default() += 1;
}

#[cfg(test)]
fn checkout_repair_count(clone_dir: &Path) -> usize {
    CHECKOUT_REPAIRS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("checkout-repair test counter poisoned")
        .get(clone_dir)
        .copied()
        .unwrap_or_default()
}

#[cfg(test)]
fn signal_checkout_test_path(variable: &str) -> Result<()> {
    let Some(path) = std::env::var_os(variable) else {
        return Ok(());
    };
    std::fs::write(PathBuf::from(&path), b"ready\n").with_context(|| {
        format!(
            "writing checkout test signal {} from {variable}",
            PathBuf::from(path).display()
        )
    })
}

#[cfg(test)]
async fn pause_initializer_after_exclusive_for_test() -> Result<()> {
    let Some(release) = std::env::var_os("RETREAD_TEST_EXCLUSIVE_RELEASE") else {
        return Ok(());
    };
    signal_checkout_test_path("RETREAD_TEST_EXCLUSIVE_HELD")?;
    let release = PathBuf::from(release);
    loop {
        if release.try_exists().with_context(|| {
            format!(
                "checking checkout test release signal {}",
                release.display()
            )
        })? {
            return Ok(());
        }
        tokio::time::sleep(CHECKOUT_LOCK_POLL).await;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CloneIdentity {
    digest: [u8; 32],
    marker_contents: Vec<u8>,
}

impl CloneIdentity {
    fn new(url: &str, rev: &str) -> Self {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(url.as_bytes());
        hasher.update(b"\0");
        hasher.update(rev.as_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        let digest_hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        let marker_contents =
            format!("pixi-build-retread-checkout-v1\n{digest_hex}\n").into_bytes();
        Self {
            digest,
            marker_contents,
        }
    }
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct ProcessLockKey {
    device: u64,
    inode: u64,
}

#[cfg(not(unix))]
type ProcessLockKey = PathBuf;

struct ProcessCloneLock {
    file: File,
    lock_path: PathBuf,
    identity: CloneIdentity,
    local: Arc<tokio::sync::RwLock<()>>,
    shared: AtomicBool,
    poisoned: AtomicBool,
    #[cfg(test)]
    os_acquisitions: std::sync::atomic::AtomicUsize,
}

fn process_clone_locks() -> &'static Mutex<HashMap<ProcessLockKey, Arc<ProcessCloneLock>>> {
    // Strong entries intentionally retain one fd per checkout touched by this
    // backend process. That bounded process-lifetime cost is what guarantees a
    // later worker can never reopen and flock the same inode a second time.
    static LOCKS: OnceLock<Mutex<HashMap<ProcessLockKey, Arc<ProcessCloneLock>>>> = OnceLock::new();
    LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(unix)]
fn process_lock_key(file: &File, _lock_path: &Path) -> std::io::Result<ProcessLockKey> {
    use std::os::unix::fs::MetadataExt;

    let metadata = file.metadata()?;
    Ok(ProcessLockKey {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

#[cfg(not(unix))]
fn process_lock_key(_file: &File, lock_path: &Path) -> std::io::Result<ProcessLockKey> {
    std::fs::canonicalize(lock_path)
}

/// Open the clone lock and merge it into the process registry before any
/// `flock` call. On Unix the key is the opened file's `(st_dev, st_ino)`, so
/// symlink/path aliases cannot make this process lock the same inode twice.
fn registered_clone_lock(
    lock_path: &Path,
    identity: &CloneIdentity,
) -> Result<Arc<ProcessCloneLock>> {
    let candidate = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .with_context(|| format!("opening git-clone lock file {}", lock_path.display()))?;
    let key = process_lock_key(&candidate, lock_path)
        .with_context(|| format!("identifying git-clone lock file {}", lock_path.display()))?;
    let mut locks = process_clone_locks()
        .lock()
        .map_err(|_| anyhow!("git-clone process lock registry is poisoned"))?;
    if let Some(existing) = locks.get(&key) {
        if existing.identity.digest != identity.digest {
            bail!(
                "git-clone cache-key collision: lock inode {} is already bound to a different full (url, rev) digest",
                existing.lock_path.display(),
            );
        }
        #[cfg(test)]
        signal_checkout_test_path("RETREAD_TEST_SECOND_REGISTRATION")?;
        // `candidate` is dropped here without ever being flocked. Every worker
        // in this process therefore flocks only the registry's one open
        // description, even though identifying an alias requires opening it.
        return Ok(Arc::clone(existing));
    }

    let lock = Arc::new(ProcessCloneLock {
        file: candidate,
        lock_path: lock_path.to_path_buf(),
        identity: identity.clone(),
        local: Arc::new(tokio::sync::RwLock::new(())),
        shared: AtomicBool::new(false),
        poisoned: AtomicBool::new(false),
        #[cfg(test)]
        os_acquisitions: std::sync::atomic::AtomicUsize::new(0),
    });
    locks.insert(key, Arc::clone(&lock));
    Ok(lock)
}

struct CloneReadLease {
    _local: tokio::sync::OwnedRwLockReadGuard<()>,
    lock: Arc<ProcessCloneLock>,
}

/// A checked-out source tree plus its process-local reader lease. The process
/// registry retains one cross-process SH flock for the life of the process;
/// this lease multiplexes that one OS lock across worker tasks without any
/// worker reopening or re-flocking the lock inode.
#[derive(Clone)]
pub(crate) struct GitCheckout {
    root: PathBuf,
    lease: Arc<CloneReadLease>,
}

impl GitCheckout {
    pub(crate) fn root(&self) -> &Path {
        &self.root
    }
}

impl std::fmt::Debug for GitCheckout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GitCheckout")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

impl PartialEq for GitCheckout {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root && self.lease.lock.identity == other.lease.lock.identity
    }
}

impl Eq for GitCheckout {}

#[derive(Debug)]
enum CheckoutMarkerState {
    Missing,
    Matching,
    Invalid,
}

fn checkout_ready_marker(clone_dir: &Path) -> PathBuf {
    clone_dir.join(".git").join(CHECKOUT_READY_MARKER)
}

fn checkout_marker_state(
    clone_dir: &Path,
    identity: &CloneIdentity,
) -> Result<CheckoutMarkerState> {
    let marker = checkout_ready_marker(clone_dir);
    match std::fs::read(&marker) {
        Ok(contents) if contents == identity.marker_contents => Ok(CheckoutMarkerState::Matching),
        Ok(_) => Ok(CheckoutMarkerState::Invalid),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(CheckoutMarkerState::Missing)
        }
        Err(error) => {
            Err(error).with_context(|| format!("reading git-checkout marker {}", marker.display()))
        }
    }
}

fn invalid_marker_error(clone_dir: &Path) -> anyhow::Error {
    anyhow!(
        "git-checkout marker {} is malformed or belongs to a different full (url, rev) identity; refusing to mutate a published checkout",
        checkout_ready_marker(clone_dir).display(),
    )
}

struct PendingOsLock {
    lock: Arc<ProcessCloneLock>,
    armed: bool,
}

impl PendingOsLock {
    fn acquired(lock: Arc<ProcessCloneLock>) -> Self {
        #[cfg(test)]
        lock.os_acquisitions.fetch_add(1, Ordering::Relaxed);
        Self { lock, armed: true }
    }

    fn commit_shared(mut self) {
        self.lock.shared.store(true, Ordering::Release);
        self.armed = false;
    }

    fn downgrade_and_commit(mut self) -> Result<()> {
        #[cfg(unix)]
        fs4::fs_std::FileExt::lock_shared(&self.lock.file).with_context(|| {
            format!(
                "downgrading git-clone lock to shared {}",
                self.lock.lock_path.display()
            )
        })?;

        // Windows LockFileEx layers a SH lock over an EX lock rather than
        // replacing it. Keep the same File/open description, but transition
        // through an unlock after the marker is published. Any interposing EX
        // owner rechecks that marker and therefore performs no mutation.
        #[cfg(not(unix))]
        {
            fs4::fs_std::FileExt::unlock(&self.lock.file).with_context(|| {
                format!(
                    "unlocking exclusive git-clone lock before shared transition {}",
                    self.lock.lock_path.display()
                )
            })?;
            self.armed = false;
            fs4::fs_std::FileExt::lock_shared(&self.lock.file).with_context(|| {
                format!(
                    "shared-locking git-clone lock after transition {}",
                    self.lock.lock_path.display()
                )
            })?;
            self.armed = true;
        }

        self.lock.shared.store(true, Ordering::Release);
        self.armed = false;
        Ok(())
    }
}

impl Drop for PendingOsLock {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let Err(error) = fs4::fs_std::FileExt::unlock(&self.lock.file) {
            self.lock.poisoned.store(true, Ordering::Release);
            tracing::error!(
                lock = %self.lock.lock_path.display(),
                error = %error,
                "failed to release uncommitted git-clone lock; process entry poisoned",
            );
        }
    }
}

enum InitialOsAccess {
    Ready,
    Initialize(PendingOsLock),
}

/// Establish this process's one OS lock for a clone. All probes are
/// nonblocking: cold losers never queue an EX request that the winner's
/// process-lifetime SH lock could strand forever.
fn acquire_initial_os_access(
    lock: Arc<ProcessCloneLock>,
    clone_dir: &Path,
) -> Result<InitialOsAccess> {
    loop {
        if lock.poisoned.load(Ordering::Acquire) {
            bail!(
                "git-clone lock {} is poisoned after an unlock failure",
                lock.lock_path.display()
            );
        }

        match checkout_marker_state(clone_dir, &lock.identity)? {
            CheckoutMarkerState::Invalid => return Err(invalid_marker_error(clone_dir)),
            CheckoutMarkerState::Matching => {
                let acquired =
                    fs4::fs_std::FileExt::try_lock_shared(&lock.file).with_context(|| {
                        format!(
                            "try-shared-locking git-clone lock file {}",
                            lock.lock_path.display()
                        )
                    })?;
                if acquired {
                    let pending = PendingOsLock::acquired(Arc::clone(&lock));
                    match checkout_marker_state(clone_dir, &lock.identity)? {
                        CheckoutMarkerState::Matching => {
                            pending.commit_shared();
                            return Ok(InitialOsAccess::Ready);
                        }
                        CheckoutMarkerState::Missing => {
                            drop(pending);
                        }
                        CheckoutMarkerState::Invalid => {
                            drop(pending);
                            return Err(invalid_marker_error(clone_dir));
                        }
                    }
                } else {
                    #[cfg(test)]
                    signal_checkout_test_path("RETREAD_TEST_SHARED_BLOCKED")?;
                }
            }
            CheckoutMarkerState::Missing => {
                if fs4::fs_std::FileExt::try_lock_exclusive(&lock.file).with_context(|| {
                    format!(
                        "try-exclusive-locking git-clone lock file {}",
                        lock.lock_path.display()
                    )
                })? {
                    let pending = PendingOsLock::acquired(Arc::clone(&lock));
                    // EX->SH replacement is not atomic on every flock
                    // implementation. An interposing process may have published
                    // readiness since our unlocked observation, so recheck under EX.
                    match checkout_marker_state(clone_dir, &lock.identity)? {
                        CheckoutMarkerState::Missing => {
                            return Ok(InitialOsAccess::Initialize(pending));
                        }
                        CheckoutMarkerState::Matching => {
                            pending.downgrade_and_commit()?;
                            return Ok(InitialOsAccess::Ready);
                        }
                        CheckoutMarkerState::Invalid => {
                            drop(pending);
                            return Err(invalid_marker_error(clone_dir));
                        }
                    }
                }
            }
        }
        std::thread::sleep(CHECKOUT_LOCK_POLL);
    }
}

fn publish_checkout_ready(clone_dir: &Path, identity: &CloneIdentity) -> Result<()> {
    let marker = checkout_ready_marker(clone_dir);
    let marker_parent = marker
        .parent()
        .ok_or_else(|| anyhow!("git-checkout marker has no parent: {}", marker.display()))?;
    let temporary = marker_parent.join(format!("{CHECKOUT_READY_MARKER}.tmp"));
    let publish = || -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&temporary)
            .with_context(|| format!("creating git-checkout marker {}", temporary.display()))?;
        file.write_all(&identity.marker_contents)
            .with_context(|| format!("writing git-checkout marker {}", temporary.display()))?;
        file.sync_all()
            .with_context(|| format!("syncing git-checkout marker {}", temporary.display()))?;
        std::fs::rename(&temporary, &marker).with_context(|| {
            format!(
                "publishing git-checkout marker {} -> {}",
                temporary.display(),
                marker.display()
            )
        })?;
        #[cfg(unix)]
        File::open(marker_parent)
            .and_then(|directory| directory.sync_all())
            .with_context(|| {
                format!(
                    "syncing git-checkout marker dir {}",
                    marker_parent.display()
                )
            })?;
        Ok(())
    };
    if let Err(error) = publish() {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(())
}

/// Populate or repair an UNPUBLISHED checkout while its one-time initializer
/// holds the clone's EX lock. This is never called after the ready marker has
/// been published; published warm checkouts are read-only cache entries.
async fn clone_and_checkout(clone_dir: &Path, url: &str, rev: &str) -> Result<()> {
    if !clone_dir.exists() {
        tracing::info!(url = %url, rev = %rev, "cloning git source");
        // Clone shallow without checkout. Use a two-step fetch so we can
        // target arbitrary commits (not just branch/tag tips).
        run_silent(
            Command::new("git")
                .arg("clone")
                .arg("--filter=blob:none")
                .arg("--no-checkout")
                .arg(url)
                .arg(clone_dir),
            "git clone",
        )
        .await?;
    } else {
        tracing::debug!(path = %clone_dir.display(), "git source already cached");
    }

    if checkout_rev_robust(clone_dir, rev).await? {
        return Ok(());
    }
    // Fetch the specific rev AND its reachable tags so that
    // setuptools_scm can find a release tag and emit a static version
    // (e.g. "1.1.1") rather than a drifting dev/date suffix (e.g.
    // "1.1.1.dev4+g1234567.d20250101").
    run_silent(
        Command::new("git")
            .args(["fetch", "--tags", "origin", rev])
            .current_dir(clone_dir),
        "git fetch --tags",
    )
    .await?;
    if !checkout_rev_robust(clone_dir, "FETCH_HEAD").await? {
        bail!(
            "git checkout FETCH_HEAD failed even after cleaning the working \
             tree, in clone at {}",
            clone_dir.display()
        );
    }
    Ok(())
}

/// Perform the legacy working-tree repair and full-reclone fallback exactly
/// once, before publishing this `(url, rev)` checkout. Existing v4.8.1 cache
/// directories have no marker, so their first access under the new protocol
/// receives this one migration repair. `pending` is threaded through every
/// unabortable filesystem job so none can continue after EX is released.
async fn initialize_checkout_once(
    clone_dir: &Path,
    url: &str,
    rev: &str,
    pending: PendingOsLock,
) -> Result<PendingOsLock> {
    if let Err(error) = clone_and_checkout(clone_dir, url, rev).await {
        tracing::warn!(
            url = %url, rev = %rev, error = %format!("{error:#}"),
            path = %clone_dir.display(),
            "unpublished git clone/checkout failed after working-tree repair; \
             wiping the clone dir and re-cloning once before publication",
        );
        let remove_dir = clone_dir.to_path_buf();
        let pending = tokio::task::spawn_blocking(move || -> Result<PendingOsLock> {
            std::fs::remove_dir_all(&remove_dir)
                .with_context(|| format!("wiping corrupted clone dir {}", remove_dir.display()))?;
            Ok(pending)
        })
        .await
        .context("git-checkout wipe task panicked")??;
        clone_and_checkout(clone_dir, url, rev)
            .await
            .with_context(|| {
                format!("re-clone after wiping corrupted dir still failed for {url}@{rev}")
            })?;
        return Ok(pending);
    }
    Ok(pending)
}

async fn initialize_process_clone_lock(
    lock: Arc<ProcessCloneLock>,
    clone_dir: PathBuf,
    url: String,
    rev: String,
    _writer: tokio::sync::OwnedRwLockWriteGuard<()>,
) -> Result<()> {
    if lock.shared.load(Ordering::Acquire) {
        return Ok(());
    }
    if lock.poisoned.load(Ordering::Acquire) {
        bail!(
            "git-clone lock {} is poisoned after an unlock failure",
            lock.lock_path.display()
        );
    }

    let access = {
        let lock = Arc::clone(&lock);
        let clone_dir = clone_dir.clone();
        tokio::task::spawn_blocking(move || acquire_initial_os_access(lock, &clone_dir))
            .await
            .context("git-clone OS-lock task panicked")??
    };
    let InitialOsAccess::Initialize(pending) = access else {
        return Ok(());
    };

    #[cfg(test)]
    pause_initializer_after_exclusive_for_test().await?;

    // `pending` is cancellation-safe and the outer caller awaits this work via
    // a detached Tokio task. The EX lock therefore spans clone/repair, marker
    // publication, and the same-fd EX->SH transition as one transaction.
    let pending = initialize_checkout_once(&clone_dir, &url, &rev, pending).await?;

    let identity = lock.identity.clone();
    let marker_clone_dir = clone_dir.clone();
    let pending = tokio::task::spawn_blocking(move || -> Result<PendingOsLock> {
        publish_checkout_ready(&marker_clone_dir, &identity)?;
        Ok(pending)
    })
    .await
    .context("git-checkout marker task panicked")??;

    // Downgrade and publish the in-process Shared state inside one blocking
    // closure: no async cancellation point may split those two state changes.
    tokio::task::spawn_blocking(move || pending.downgrade_and_commit())
        .await
        .context("git-clone downgrade task panicked")??;
    Ok(())
}

/// Return a reader lease for an immutable, published `(url, rev)` checkout.
///
/// The first process to observe a missing ready marker takes EX, performs the
/// one-time clone/repair, publishes the marker, and downgrades its ONE registry
/// fd to SH. Other processes poll nonblocking until they can take SH. Within a
/// process, all tasks share that one fd through an async RwLock; no task ever
/// flocks the inode a second time.
pub(crate) async fn ensure_git_checkout(
    url: &str,
    rev: &str,
    cache_dir: &Path,
) -> Result<GitCheckout> {
    // Delegate to git_checkout_root so the layout stays in sync. (Was
    // duplicated here before v0.13.3 -- update both or the resolver
    // half stops finding the cached clone the cloner half just made.)
    let clone_dir = git_checkout_root(url, rev, cache_dir);
    let parent = clone_dir.parent().unwrap();
    tokio::fs::create_dir_all(parent).await.with_context(|| {
        format!(
            "creating git-clone parent dir {} (for url={url}, rev={rev}, target={})",
            parent.display(),
            clone_dir.display(),
        )
    })?;

    let lock_path = clone_dir.with_extension("lock");
    let identity = CloneIdentity::new(url, rev);
    let lock = {
        let lock_path = lock_path.clone();
        let identity = identity.clone();
        tokio::task::spawn_blocking(move || registered_clone_lock(&lock_path, &identity))
            .await
            .context("git-clone registry task panicked")??
    };

    if !lock.shared.load(Ordering::Acquire) {
        let writer = Arc::clone(&lock.local).write_owned().await;
        if !lock.shared.load(Ordering::Acquire) {
            // The spawned task owns `writer`. Dropping this caller's JoinHandle
            // detaches rather than cancels an in-flight mutating transaction.
            let initializer = tokio::spawn(initialize_process_clone_lock(
                Arc::clone(&lock),
                clone_dir.clone(),
                url.to_string(),
                rev.to_string(),
                writer,
            ));
            initializer
                .await
                .context("git-clone initializer task panicked")??;
        }
    }

    if lock.poisoned.load(Ordering::Acquire) {
        bail!(
            "git-clone lock {} is poisoned after an unlock failure",
            lock.lock_path.display()
        );
    }
    let local = Arc::clone(&lock.local).read_owned().await;
    if !lock.shared.load(Ordering::Acquire) {
        bail!(
            "git-clone lock {} reached reader path before SH initialization",
            lock.lock_path.display()
        );
    }

    // Validate the full identity even on the process-local Shared fast path.
    // This catches a truncated checkout-path hash collision or marker
    // deletion/corruption without ever re-enabling a destructive writer.
    let marker = checkout_ready_marker(&clone_dir);
    let marker_contents = tokio::fs::read(&marker)
        .await
        .with_context(|| format!("reading published git-checkout marker {}", marker.display()))?;
    if marker_contents != identity.marker_contents {
        return Err(invalid_marker_error(&clone_dir));
    }

    Ok(GitCheckout {
        root: clone_dir,
        lease: Arc::new(CloneReadLease {
            _local: local,
            lock,
        }),
    })
}

async fn canonical_git_tag_state(repo: &Path) -> Result<CanonicalGitTagState> {
    let refs = run_output_bytes(
        Command::new("git")
            .args([
                "for-each-ref",
                "--sort=refname",
                "--format=%(refname)%00%(objectname)%00%(*objectname)",
                "refs/tags",
            ])
            .env("GIT_OPTIONAL_LOCKS", "0")
            .current_dir(repo),
        "git canonical tag-state query",
    )
    .await?;
    let mut parsed = Vec::new();
    for record in refs
        .split(|byte| *byte == b'\n')
        .filter(|record| !record.is_empty())
    {
        let mut fields = record.split(|byte| *byte == 0);
        let name = fields
            .next()
            .ok_or_else(|| anyhow!("canonical Git tag record has no ref name"))?;
        let object_id = fields
            .next()
            .ok_or_else(|| anyhow!("canonical Git tag record has no object ID"))?;
        let peeled_id = fields
            .next()
            .ok_or_else(|| anyhow!("canonical Git tag record has no peeled object field"))?;
        if fields.next().is_some() || !name.starts_with(b"refs/tags/") {
            bail!("git returned a malformed canonical tag record");
        }
        let parse_object_id = |value: &[u8], label: &str| -> Result<String> {
            let value = std::str::from_utf8(value)
                .with_context(|| format!("canonical Git {label} is not ASCII"))?;
            if !matches!(value.len(), 40 | 64)
                || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                bail!("canonical Git {label} is not an exact object ID: {value}");
            }
            Ok(value.to_ascii_lowercase())
        };
        let object_id = parse_object_id(object_id, "tag object ID")?;
        if !peeled_id.is_empty() {
            parse_object_id(peeled_id, "peeled tag object ID")?;
        }
        parsed.push(CanonicalGitTagRef {
            name: name.to_vec(),
            object_id,
        });
    }
    Ok(CanonicalGitTagState {
        identity: hash_fields(b"retread-git-tag-state-v1\0", &[&refs]),
        refs: parsed,
    })
}

async fn canonical_git_ref_state(repo: &Path) -> Result<String> {
    Ok(canonical_git_tag_state(repo).await?.identity)
}

async fn recreate_canonical_git_tags(repo: &Path, tags: &[CanonicalGitTagRef]) -> Result<()> {
    if tags.is_empty() {
        return Ok(());
    }
    let mut commands = Vec::new();
    for tag in tags {
        commands.extend_from_slice(b"create ");
        commands.extend_from_slice(&tag.name);
        commands.push(0);
        commands.extend_from_slice(tag.object_id.as_bytes());
        commands.push(0);
    }
    run_silent_with_input(
        Command::new("git")
            .args(["update-ref", "--stdin", "-z"])
            .current_dir(repo),
        "git recreate canonical tags",
        &commands,
    )
    .await
}

async fn git_checkout_has_promisor_remote(repo: &Path) -> Result<bool> {
    let output = run_readonly_git_status(
        Command::new("git")
            .args([
                "config",
                "--local",
                "--type=bool",
                "--get-regexp",
                r"^remote\..*\.promisor$",
            ])
            .current_dir(repo),
        "git inspect promisor remotes",
    )
    .await?;
    match output.status.code() {
        Some(0) => {
            for line in output
                .stdout
                .split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty())
            {
                if line.ends_with(b" true") {
                    return Ok(true);
                }
                if !line.ends_with(b" false") {
                    bail!(
                        "git returned malformed canonical promisor state: {}",
                        String::from_utf8_lossy(line),
                    );
                }
            }
            Ok(false)
        }
        Some(1) => Ok(false),
        _ => bail!(
            "git inspect promisor remotes failed (status {}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim(),
        ),
    }
}

async fn delete_canonical_git_refs(repo: &Path, namespace: &str) -> Result<()> {
    let refs = run_output_bytes(
        Command::new("git")
            .args([
                "for-each-ref",
                "--sort=refname",
                "--format=%(refname)",
                namespace,
            ])
            .env("GIT_OPTIONAL_LOCKS", "0")
            .current_dir(repo),
        "git canonical ref query",
    )
    .await?;
    // Delete symbolic refs themselves rather than dereferencing them. Git's
    // stdin `option no-deref` applies only to the next command, so repeat it
    // for every ref. Otherwise an ordinary ref sorted before `origin/HEAD`
    // consumes the option and deleting that symref plus its target is rejected
    // as two updates to the target ref.
    let mut commands = Vec::new();
    let mut ref_count = 0_usize;
    for reference in refs
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
    {
        let reference =
            std::str::from_utf8(reference).context("canonical Git ref name is not UTF-8")?;
        commands.extend_from_slice(b"option no-deref\n");
        commands.extend_from_slice(b"delete ");
        commands.extend_from_slice(reference.as_bytes());
        commands.push(b'\n');
        ref_count += 1;
    }
    if ref_count != 0 {
        run_silent_with_input(
            Command::new("git")
                .args(["update-ref", "--stdin"])
                .current_dir(repo),
            "git batch-delete non-canonical refs",
            &commands,
        )
        .await?;
    }
    Ok(())
}

async fn ensure_no_canonical_gitlinks(repo: &Path, resolved_sha: &str) -> Result<()> {
    let tree = run_output_bytes(
        Command::new("git")
            .args(["ls-tree", "-r", "-z", "--full-tree", resolved_sha])
            .env("GIT_OPTIONAL_LOCKS", "0")
            .current_dir(repo),
        "git gitlink query",
    )
    .await?;
    let mut gitlinks = Vec::new();
    for record in tree
        .split(|byte| *byte == 0)
        .filter(|record| !record.is_empty())
    {
        if record.starts_with(b"160000 ") {
            let path = record
                .iter()
                .position(|byte| *byte == b'\t')
                .map(|tab| String::from_utf8_lossy(&record[tab + 1..]).into_owned())
                .unwrap_or_else(|| "<malformed gitlink>".to_string());
            gitlinks.push(path);
        }
    }
    if !gitlinks.is_empty() {
        gitlinks.sort();
        bail!(
            "git source at commit {resolved_sha} contains unsupported submodule gitlinks; retread will not silently build an uninitialized tree: {}",
            gitlinks.join(", "),
        );
    }
    Ok(())
}

fn sanitize_canonical_git_metadata(repo: &Path) -> Result<()> {
    let git_dir = repo.join(".git");
    for directory in ["logs"] {
        let path = git_dir.join(directory);
        match std::fs::remove_dir_all(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("removing volatile Git metadata {}", path.display()));
            }
        }
    }
    for file in [
        "FETCH_HEAD",
        "ORIG_HEAD",
        "MERGE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "BISECT_LOG",
        "index.lock",
        "packed-refs.lock",
    ] {
        let path = git_dir.join(file);
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("removing volatile Git metadata {}", path.display()));
            }
        }
    }
    Ok(())
}

fn require_real_canonical_git_directory(path: &Path, label: &str) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("stating {label} {}", path.display()))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        bail!("{label} is not a real directory: {}", path.display());
    }
    Ok(())
}

fn reset_canonical_git_lfs_tmp(repo: &Path) -> Result<()> {
    let git_dir = repo.join(".git");
    require_real_canonical_git_directory(&git_dir, "canonical Git metadata")?;
    let lfs = git_dir.join("lfs");
    match std::fs::symlink_metadata(&lfs) {
        Ok(_) => require_real_canonical_git_directory(&lfs, "canonical Git LFS storage")?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir(&lfs)
                .with_context(|| format!("creating canonical Git LFS storage {}", lfs.display()))?;
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("stating canonical Git LFS storage {}", lfs.display()));
        }
    }
    let tmp = lfs.join("tmp");
    match std::fs::symlink_metadata(&tmp) {
        Ok(_) => require_real_canonical_git_directory(&tmp, "canonical Git LFS scratch")?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("stating canonical Git LFS scratch {}", tmp.display()));
        }
    }
    match std::fs::remove_dir_all(&tmp) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("clearing canonical Git LFS scratch {}", tmp.display()));
        }
    }
    std::fs::create_dir_all(&tmp)
        .with_context(|| format!("creating canonical Git LFS scratch {}", tmp.display()))
}

fn clear_canonical_git_lfs_tmp(repo: &Path) -> Result<()> {
    let git_dir = repo.join(".git");
    let lfs = git_dir.join("lfs");
    let tmp = lfs.join("tmp");
    require_real_canonical_git_directory(&git_dir, "canonical Git metadata")?;
    require_real_canonical_git_directory(&lfs, "canonical Git LFS storage")?;
    require_real_canonical_git_directory(&tmp, "canonical Git LFS scratch")?;
    for entry in std::fs::read_dir(&tmp)
        .with_context(|| format!("reading canonical Git LFS scratch {}", tmp.display()))?
    {
        let entry = entry
            .with_context(|| format!("enumerating canonical Git LFS scratch {}", tmp.display()))?;
        let path = entry.path();
        let metadata = std::fs::symlink_metadata(&path).with_context(|| {
            format!("stating canonical Git LFS scratch entry {}", path.display())
        })?;
        if metadata.is_dir() && !metadata.file_type().is_symlink() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        }
        .with_context(|| {
            format!(
                "clearing canonical Git LFS scratch entry {}",
                path.display()
            )
        })?;
    }
    Ok(())
}

fn make_canonical_git_lfs_tmp_writable(repo: &Path) -> Result<()> {
    let tmp = repo.join(".git/lfs/tmp");
    require_real_canonical_git_directory(&repo.join(".git"), "canonical Git metadata")?;
    require_real_canonical_git_directory(&repo.join(".git/lfs"), "canonical Git LFS storage")?;
    require_real_canonical_git_directory(&tmp, "canonical Git LFS scratch")?;
    let metadata = std::fs::symlink_metadata(&tmp)
        .with_context(|| format!("stating canonical Git LFS scratch {}", tmp.display()))?;
    let mut permissions = metadata.permissions();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        permissions.set_mode((permissions.mode() & !0o022) | 0o700);
    }
    #[cfg(not(unix))]
    permissions.set_readonly(false);
    std::fs::set_permissions(&tmp, permissions).with_context(|| {
        format!(
            "making canonical Git LFS filter scratch writable {}",
            tmp.display(),
        )
    })
}

async fn validate_canonical_git_worktree(cache_dir: &Path, repo: &Path) -> Result<Vec<u8>> {
    // Validation can run concurrently after multiple packs share a canonical
    // source. Serialize the one writable Git LFS directory, and keep the lock
    // inside a blocking owner: cancelling the async request then detaches the
    // join handle, but the status process still finishes and clears scratch
    // before releasing the lock.
    let validation_namespace = cache_dir.join("canonical-git-status");
    let lock = acquire_artifact_cache_lock(&validation_namespace).await?;
    let repo = repo.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let _lock = lock;
        clear_canonical_git_lfs_tmp(&repo)?;
        // `lfs.storage` is configurable at repository, user, and system
        // scope. Pin it so a clean filter cannot redirect temporary writes
        // outside the sealed canonical repository or to another read-only
        // directory within `.git`.
        let output = std::process::Command::new("git")
            .args([
                "-c",
                "lfs.storage=lfs",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ])
            .env("GIT_OPTIONAL_LOCKS", "0")
            .current_dir(&repo)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("spawning git validate canonical worktree")?;
        let cleanup = clear_canonical_git_lfs_tmp(&repo);
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            cleanup.context("cleaning Git LFS scratch after failed canonical validation")?;
            bail!(
                "git validate canonical worktree failed (status {}): {}",
                output.status,
                stderr.trim(),
            );
        }
        cleanup.context("cleaning Git LFS scratch after canonical validation")?;
        Ok(output.stdout)
    })
    .await
    .context("canonical Git worktree validation task panicked")?
}

fn normalize_source_tree_times(path: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("stating canonical source path {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Ok(());
    }
    if metadata.is_dir() {
        let mut entries = std::fs::read_dir(path)
            .with_context(|| format!("reading canonical source path {}", path.display()))?
            .collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            normalize_source_tree_times(&entry.path())?;
        }
    }
    normalize_snapshot_times(path)
}

fn make_source_tree_read_only(path: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("stating canonical source path {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Ok(());
    }
    if metadata.is_dir() {
        let mut entries = std::fs::read_dir(path)
            .with_context(|| format!("reading canonical source path {}", path.display()))?
            .collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            make_source_tree_read_only(&entry.path())?;
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = metadata.permissions().mode() & !0o222;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).with_context(
            || format!("making canonical source path read-only {}", path.display()),
        )?;
    }
    #[cfg(not(unix))]
    {
        let mut permissions = metadata.permissions();
        permissions.set_readonly(true);
        std::fs::set_permissions(path, permissions).with_context(|| {
            format!("making canonical source path read-only {}", path.display())
        })?;
    }
    Ok(())
}

async fn validate_canonical_git_snapshot(
    cache_dir: &Path,
    repository_identity: &str,
    resolved_sha: &str,
    ref_state: &str,
    require_clean: bool,
) -> Result<CanonicalGitSnapshot> {
    let cache_metadata = std::fs::symlink_metadata(cache_dir)
        .with_context(|| format!("stating canonical Git source cache {}", cache_dir.display()))?;
    if !cache_metadata.file_type().is_dir() || cache_metadata.file_type().is_symlink() {
        bail!(
            "canonical Git source cache is not a real directory: {}",
            cache_dir.display(),
        );
    }
    let marker_path = cache_dir.join("source.json");
    let marker_metadata = std::fs::symlink_metadata(&marker_path)
        .with_context(|| format!("stating canonical Git marker {}", marker_path.display()))?;
    if !marker_metadata.file_type().is_file() || marker_metadata.file_type().is_symlink() {
        bail!(
            "canonical Git source marker is not a regular file: {}",
            marker_path.display(),
        );
    }
    let marker: CanonicalGitSourceMarker = serde_json::from_slice(
        &std::fs::read(&marker_path)
            .with_context(|| format!("reading canonical Git marker {}", marker_path.display()))?,
    )
    .with_context(|| format!("parsing canonical Git marker {}", marker_path.display()))?;
    if marker.schema != CANONICAL_GIT_SOURCE_SCHEMA
        || marker.repository_identity != repository_identity
        || marker.resolved_sha != resolved_sha
        || marker.ref_state != ref_state
    {
        bail!(
            "canonical Git source marker does not match its full repository/commit/ref identity: {}",
            marker_path.display(),
        );
    }
    let repo = cache_dir.join("repo");
    let repo_metadata = std::fs::symlink_metadata(&repo)
        .with_context(|| format!("stating canonical Git repository {}", repo.display()))?;
    if !repo_metadata.file_type().is_dir() || repo_metadata.file_type().is_symlink() {
        bail!(
            "canonical Git repository is not a real directory: {}",
            repo.display(),
        );
    }
    let head = run_output(
        Command::new("git")
            .args(["rev-parse", "--verify", "HEAD^{commit}"])
            .env("GIT_OPTIONAL_LOCKS", "0")
            .current_dir(&repo),
        "git validate canonical HEAD",
    )
    .await?;
    if head.trim().to_ascii_lowercase() != resolved_sha {
        bail!(
            "canonical Git repository HEAD `{}` does not match `{resolved_sha}`",
            head.trim(),
        );
    }
    let actual_ref_state = canonical_git_ref_state(&repo).await?;
    if actual_ref_state != ref_state {
        bail!(
            "canonical Git repository tag/ref state changed: expected {ref_state}, found {actual_ref_state}"
        );
    }
    let noncanonical_refs = run_output_bytes(
        Command::new("git")
            .args([
                "for-each-ref",
                "--format=%(refname)",
                "refs/heads",
                "refs/remotes",
            ])
            .env("GIT_OPTIONAL_LOCKS", "0")
            .current_dir(&repo),
        "git validate canonical refs",
    )
    .await?;
    if !noncanonical_refs.is_empty() {
        bail!("canonical Git repository regained branch or remote-tracking refs");
    }
    ensure_no_canonical_gitlinks(&repo, resolved_sha).await?;
    if require_clean {
        let status = validate_canonical_git_worktree(cache_dir, &repo).await?;
        if !status.is_empty() {
            bail!(
                "canonical Git source was mutated while a build used it: {}",
                String::from_utf8_lossy(&status).trim(),
            );
        }
    }
    Ok(CanonicalGitSnapshot {
        root: repo,
        repository_identity: repository_identity.to_string(),
        resolved_sha: resolved_sha.to_string(),
        ref_state: ref_state.to_string(),
    })
}

async fn ensure_canonical_git_snapshot(
    shared_checkout: &Path,
    upstream_url: &str,
    resolved_sha: &str,
    ref_state: &str,
) -> Result<CanonicalGitSnapshot> {
    let repository_identity = canonical_git_repository_identity(upstream_url, resolved_sha);
    let cache_dir = crate::courier::retread_cache_root()
        .join("canonical-git-sources")
        .join("v1")
        .join(&repository_identity)
        .join(ref_state);
    let _lock = acquire_artifact_cache_lock(&cache_dir).await?;
    if cache_dir.try_exists().with_context(|| {
        format!(
            "checking canonical Git source cache {}",
            cache_dir.display()
        )
    })? {
        // Published canonical trees are never self-healed or replaced while a
        // reader could be using them. Corruption is therefore a fail-closed
        // error rather than a delete/rebuild race.
        return validate_canonical_git_snapshot(
            &cache_dir,
            &repository_identity,
            resolved_sha,
            ref_state,
            true,
        )
        .await;
    }

    let staging = unique_staging_dir(&cache_dir)?;
    let repo = staging.0.join("repo");
    tracing::debug!(
        source = %shared_checkout.display(),
        commit = %resolved_sha,
        refs = %ref_state,
        "preparing shared canonical Git source",
    );
    let promisor_checkout = git_checkout_has_promisor_remote(shared_checkout).await?;
    let shared_tags = if promisor_checkout {
        let tags = canonical_git_tag_state(shared_checkout).await?;
        if tags.identity != ref_state {
            bail!(
                "shared Git checkout tag/ref state changed before canonicalization: expected {ref_state}, found {}",
                tags.identity,
            );
        }
        Some(tags)
    } else {
        None
    };
    let mut clone = Command::new("git");
    clone.arg("clone");
    if promisor_checkout {
        // Serving a partial clone through a local upload-pack forbids the
        // lazy fetches needed to fill missing objects. Re-clone its already
        // bound upstream while retaining blob filtering instead of mutating
        // the published shared checkout.
        clone.args(["--filter=blob:none", "--no-tags"]);
    }
    clone.args(["--no-local", "--no-checkout", "--"]);
    if promisor_checkout {
        clone.arg(upstream_url);
    } else {
        clone.arg(shared_checkout);
    }
    clone.arg(&repo);
    run_silent(&mut clone, "git clone canonical source").await?;
    if let Some(shared_tags) = &shared_tags {
        let mut required_objects = vec![resolved_sha.to_string()];
        required_objects.extend(shared_tags.refs.iter().map(|tag| tag.object_id.clone()));
        required_objects.sort();
        required_objects.dedup();
        let mut fetch = Command::new("git");
        fetch
            .args(["fetch", "--force", "--no-tags", "origin"])
            .args(&required_objects)
            .current_dir(&repo);
        run_silent(&mut fetch, "git fetch canonical objects").await?;
        delete_canonical_git_refs(&repo, "refs/tags").await?;
        recreate_canonical_git_tags(&repo, &shared_tags.refs).await?;
    } else {
        run_silent(
            Command::new("git")
                .args(["fetch", "--force", "--tags", "origin"])
                .current_dir(&repo),
            "git fetch canonical tags",
        )
        .await?;
    }
    run_silent(
        Command::new("git")
            .args([
                "-c",
                "lfs.storage=lfs",
                "checkout",
                "--detach",
                "--force",
                resolved_sha,
            ])
            .current_dir(&repo),
        "git checkout canonical commit",
    )
    .await?;
    run_silent(
        Command::new("git")
            .args(["clean", "-ffdx"])
            .current_dir(&repo),
        "git clean canonical source",
    )
    .await?;
    delete_canonical_git_refs(&repo, "refs/heads").await?;
    delete_canonical_git_refs(&repo, "refs/remotes").await?;
    run_silent(
        Command::new("git")
            .args(["remote", "set-url", "origin", upstream_url])
            .current_dir(&repo),
        "git normalize canonical origin",
    )
    .await?;
    let actual_ref_state = canonical_git_ref_state(&repo).await?;
    if actual_ref_state != ref_state {
        bail!(
            "canonical Git clone tag/ref state differs from the state bound to its cache identity: expected {ref_state}, found {actual_ref_state}"
        );
    }
    let final_shared_ref_state = canonical_git_ref_state(shared_checkout).await?;
    if final_shared_ref_state != ref_state {
        bail!(
            "shared Git checkout tag/ref state changed during canonicalization: expected {ref_state}, found {final_shared_ref_state}"
        );
    }
    ensure_no_canonical_gitlinks(&repo, resolved_sha).await?;
    let status = run_output_bytes(
        Command::new("git")
            .args([
                "-c",
                "lfs.storage=lfs",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ])
            .env("GIT_OPTIONAL_LOCKS", "0")
            .current_dir(&repo),
        "git verify canonical source",
    )
    .await?;
    if !status.is_empty() {
        bail!(
            "canonical Git source is not clean after exact checkout: {}",
            String::from_utf8_lossy(&status).trim(),
        );
    }
    let repo_for_normalize = repo.clone();
    tokio::task::spawn_blocking(move || {
        sanitize_canonical_git_metadata(&repo_for_normalize)?;
        // `git status` runs clean filters. Git LFS writes a transient file to
        // `.git/lfs/tmp` even for an unchanged worktree, so retain one empty,
        // non-source scratch directory as the sole writable exception inside
        // the otherwise immutable canonical snapshot.
        reset_canonical_git_lfs_tmp(&repo_for_normalize)?;
        normalize_source_tree_times(&repo_for_normalize)?;
        make_source_tree_read_only(&repo_for_normalize)?;
        make_canonical_git_lfs_tmp_writable(&repo_for_normalize)
    })
    .await
    .context("canonical Git source normalization task panicked")??;
    let marker = CanonicalGitSourceMarker {
        schema: CANONICAL_GIT_SOURCE_SCHEMA.to_string(),
        repository_identity: repository_identity.clone(),
        resolved_sha: resolved_sha.to_string(),
        ref_state: ref_state.to_string(),
    };
    std::fs::write(
        staging.0.join("source.json"),
        serde_json::to_vec_pretty(&marker).context("serializing canonical Git marker")?,
    )
    .with_context(|| {
        format!(
            "writing canonical Git marker {}",
            staging.0.join("source.json").display()
        )
    })?;
    std::fs::rename(&staging.0, &cache_dir)
        .with_context(|| format!("publishing canonical Git source {}", cache_dir.display()))?;
    validate_canonical_git_snapshot(
        &cache_dir,
        &repository_identity,
        resolved_sha,
        ref_state,
        true,
    )
    .await
}

/// Derive a disposable, writable SCM checkout from an immutable canonical Git
/// snapshot. Build backends are allowed to create egg-info, SCM caches, and
/// other temporary files here; the canonical tree remains the pristine source
/// used later for wheel injection.
async fn prepare_private_git_build_tree(
    canonical: &CanonicalGitSnapshot,
    upstream_url: &str,
    subdirectory: &Path,
    private_out: &Path,
) -> Result<PathBuf> {
    let private_repo = private_out
        .parent()
        .ok_or_else(|| anyhow!("private wheel output has no staging parent"))?
        .join("git-build-source");
    run_silent(
        Command::new("git")
            .args(["clone", "--shared", "--no-checkout", "--"])
            .arg(&canonical.root)
            .arg(&private_repo),
        "git clone private build source",
    )
    .await?;
    run_silent(
        Command::new("git")
            .args([
                "-c",
                "lfs.storage=lfs",
                "checkout",
                "--detach",
                "--force",
                &canonical.resolved_sha,
            ])
            .current_dir(&private_repo),
        "git checkout private build source",
    )
    .await?;
    run_silent(
        Command::new("git")
            .args(["clean", "-ffdx"])
            .current_dir(&private_repo),
        "git clean private build source",
    )
    .await?;
    run_silent(
        Command::new("git")
            .args(["remote", "set-url", "origin"])
            .arg(upstream_url)
            .current_dir(&private_repo),
        "git normalize private build origin",
    )
    .await?;
    let head = run_output(
        Command::new("git")
            .args(["rev-parse", "--verify", "HEAD^{commit}"])
            .env("GIT_OPTIONAL_LOCKS", "0")
            .current_dir(&private_repo),
        "git verify private build HEAD",
    )
    .await?;
    if head.trim().to_ascii_lowercase() != canonical.resolved_sha {
        bail!(
            "private Git build checkout resolved `{}` instead of canonical commit `{}`",
            head.trim(),
            canonical.resolved_sha,
        );
    }
    let ref_state = canonical_git_ref_state(&private_repo).await?;
    if ref_state != canonical.ref_state {
        bail!(
            "private Git build checkout tag/ref state differs from its canonical snapshot: expected {}, found {ref_state}",
            canonical.ref_state,
        );
    }
    confined_git_source_dir(&private_repo, subdirectory)
}

/// Result of a git wheel build. Keeping this value alive retains both the
/// checkout's process-local reader lease (used only for its planned identity)
/// and the canonical clean source tree consumed by build and injection.
#[derive(Debug)]
pub(crate) struct GitWheelBuild {
    wheel_path: PathBuf,
    resolved_sha: String,
    checkout: GitCheckout,
    canonical: CanonicalGitSnapshot,
    project_root: PathBuf,
}

impl GitWheelBuild {
    pub(crate) fn wheel_path(&self) -> &Path {
        &self.wheel_path
    }

    pub(crate) fn resolved_sha(&self) -> &str {
        &self.resolved_sha
    }

    pub(crate) fn checkout_root(&self) -> &Path {
        self.checkout.root()
    }

    pub(crate) fn canonical_root(&self) -> &Path {
        &self.canonical.root
    }

    pub(crate) fn source_root(&self) -> &Path {
        &self.project_root
    }
}

pub(crate) async fn build_wheel_from_git_leased_for_target(
    url: &str,
    rev: &str,
    subdirectory: &str,
    cache_dir: &Path,
    out_dir: &Path,
    target: &ResolutionTarget,
    expected: Option<&ExpectedWheel>,
) -> Result<GitWheelBuild> {
    build_wheel_from_git_inner(url, rev, subdirectory, cache_dir, out_dir, target, expected).await
}

/// Build a wheel from a clone-once git checkout and return its path plus the
/// resolved commit SHA. The checkout stays under the process's retained SH
/// lock for this complete operation. `rev` may be a commit, tag, or branch.
///
/// The emitted wheel filename is also checked for non-reproducible
/// `setuptools_scm` markers (`.devN`, `.dYYYYMMDD`, and `+g<sha>`).
pub async fn build_wheel_from_git(
    url: &str,
    rev: &str,
    subdirectory: &str,
    cache_dir: &Path,
    out_dir: &Path,
    python_version: &str,
) -> Result<(PathBuf, String)> {
    let target = ResolutionTarget::for_subdir(
        &normalized_python_minor(python_version)?.version(),
        crate::glibc::current_pixi_platform(),
    );
    let build =
        build_wheel_from_git_inner(url, rev, subdirectory, cache_dir, out_dir, &target, None)
            .await?;
    Ok((
        build.wheel_path().to_path_buf(),
        build.resolved_sha().to_string(),
    ))
}

async fn build_wheel_from_git_inner(
    url: &str,
    rev: &str,
    subdirectory: &str,
    cache_dir: &Path,
    out_dir: &Path,
    target: &ResolutionTarget,
    expected: Option<&ExpectedWheel>,
) -> Result<GitWheelBuild> {
    let python = normalized_python_minor(target.python_version())?;
    let subdirectory = normalize_git_subdirectory(subdirectory)?;
    let subdirectory_identity = subdirectory
        .to_str()
        .ok_or_else(|| anyhow!("git wheel subdirectory is not UTF-8"))?
        .to_string();
    let exact_rev =
        matches!(rev.len(), 40 | 64) && rev.bytes().all(|byte| byte.is_ascii_hexdigit());

    // A foreign build may consume a previously validated compatible artifact,
    // but it must not clone/download merely to discover a moving ref or build
    // natively. Exact commit pins can probe the v3 artifact cache first.
    if !native_build_allowed(target) {
        if !exact_rev {
            return Err(source_build_refusal_error(target)).with_context(|| {
                format!("git source `{url}` uses moving/non-exact revision `{rev}`")
            });
        }
        let resolved_sha = rev.to_ascii_lowercase();
        let family_identity = git_wheel_family_identity(url, &resolved_sha, &subdirectory_identity);
        let candidate_ref_states =
            probe_cached_git_family_states(&family_identity, target, expected).await?;
        if candidate_ref_states.is_empty() {
            return Err(source_build_refusal_error(target));
        }
        // The validated artifact hit authorizes source-tree materialization for
        // downstream auto-data injection; it does not authorize a native build.
        // Historical tag-state leaves are deliberately not materialized yet:
        // the exact checkout below selects the one state visible now.
        let checkout = ensure_git_checkout(url, rev, cache_dir).await?;
        confined_git_source_dir(checkout.root(), &subdirectory)?;
        let checkout_sha = run_output(
            Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(checkout.root()),
            "git rev-parse HEAD",
        )
        .await?;
        if checkout_sha.trim().to_ascii_lowercase() != resolved_sha {
            bail!(
                "foreign git artifact cache hit was bound to commit `{resolved_sha}` but checkout resolved `{}`",
                checkout_sha.trim(),
            );
        }
        let checkout_ref_state = canonical_git_ref_state(checkout.root()).await?;
        let matching_states = candidate_ref_states
            .iter()
            .filter(|state| state.as_str() == checkout_ref_state)
            .count();
        if matching_states == 0 {
            bail!(
                "foreign git artifact cache family has no validated artifact for the checkout's current tag/ref state `{checkout_ref_state}`"
            );
        }
        if matching_states > 1 {
            bail!(
                "foreign git artifact cache family has {matching_states} validated leaves for current tag/ref state `{checkout_ref_state}`"
            );
        }
        let source_identity = git_wheel_source_identity(&family_identity, &checkout_ref_state);
        let wheel_path = lookup_cached_build(
            "git",
            &source_identity,
            target,
            out_dir,
            expected,
        )
        .await?
        .ok_or_else(|| {
            anyhow!(
                "validated foreign git artifact disappeared before exact ref-state materialization"
            )
        })?;
        let canonical =
            ensure_canonical_git_snapshot(checkout.root(), url, &resolved_sha, &checkout_ref_state)
                .await?;
        let project_root = confined_git_source_dir(&canonical.root, &subdirectory)?;
        return Ok(GitWheelBuild {
            wheel_path,
            resolved_sha,
            checkout,
            canonical,
            project_root,
        });
    }

    // NOTE on the shared cross-pack wheel cache: the lookup deliberately
    // happens AFTER clone+checkout (below), not here. Callers derive
    // `source_root` from the checkout for the auto-data inject phase, so the
    // clone must exist even on a cache hit. The clone is machine-shared per
    // (url, rev) and a no-op when warm; the cache only needs to skip the
    // expensive per-pack `uv build`.
    let checkout = ensure_git_checkout(url, rev, cache_dir).await?;
    let clone_dir = checkout.root();

    confined_git_source_dir(clone_dir, &subdirectory)?;

    // Resolve the actual commit object ID after checkout. This converts
    // branch names, tags, and "HEAD" to a full 40- or 64-hex identity that
    // the lock can store. Keying on the resolved identity (rather than the
    // original `rev` string) ensures a lukewarm replay clones the exact same
    // commit even when the original rev was a moving ref like a branch name.
    let resolved_sha = run_output(
        Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&clone_dir),
        "git rev-parse HEAD",
    )
    .await?;
    let resolved_sha = resolved_sha.trim().to_string();

    // Cross-pack shared-cache lookup, now that a moving rev (branch/tag)
    // has been resolved to an exact sha. A hit skips the `uv build` (the
    // expensive part; the clone above was needed anyway to resolve the sha).
    if !matches!(resolved_sha.len(), 40 | 64)
        || !resolved_sha.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("git rev-parse returned a non-commit identity `{resolved_sha}`");
    }
    let resolved_sha = resolved_sha.to_ascii_lowercase();
    let ref_state = canonical_git_ref_state(clone_dir).await?;
    let family_identity = git_wheel_family_identity(url, &resolved_sha, &subdirectory_identity);
    let source_identity = git_wheel_source_identity(&family_identity, &ref_state);
    let canonical =
        ensure_canonical_git_snapshot(clone_dir, url, &resolved_sha, &ref_state).await?;
    let project_root = confined_git_source_dir(&canonical.root, &subdirectory)?;

    // DETERMINISM GUARD: detect non-reproducible setuptools_scm versions.
    // A wheel whose version contains .devN, .dYYYYMMDD, or +g<sha> segments
    // was built without a reachable tag at the pinned SHA. Its filename (and
    // therefore the lock entry's `version` + `filename` fields) will DRIFT
    // across calendar days even when the commit SHA is unchanged, producing
    // a lock that is not byte-identical on replay. The `git fetch --tags`
    // above is cheap insurance; this warn fires when it was not enough.
    let canonical_for_build = canonical.clone();
    let subdirectory_for_build = subdirectory.clone();
    let upstream_url_for_build = url.to_string();
    let wheel_path = cached_build(
        "git",
        &source_identity,
        target,
        out_dir,
        expected,
        move |private_out| async move {
            let private_project_root = prepare_private_git_build_tree(
                &canonical_for_build,
                &upstream_url_for_build,
                &subdirectory_for_build,
                &private_out,
            )
            .await?;
            let py_arg = format!("--python={}", python.identity());
            let out_arg = format!("--out-dir={}", private_out.display());
            run_capturing_uv(&[
                "build",
                "--wheel",
                &py_arg,
                &out_arg,
                &private_project_root.display().to_string(),
            ])
            .await?;
            let cache_dir = canonical_for_build
                .root
                .parent()
                .expect("canonical Git repo has a cache parent");
            validate_canonical_git_snapshot(
                cache_dir,
                &canonical_for_build.repository_identity,
                &canonical_for_build.resolved_sha,
                &canonical_for_build.ref_state,
                true,
            )
            .await?;
            Ok(())
        },
    )
    .await?;
    if wheel_path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(is_nondeterministic_version)
    {
        tracing::warn!(
            url = %url,
            rev = %rev,
            resolved_sha = %resolved_sha,
            filename = %wheel_path.display(),
            "git-source wheel has a non-reproducible setuptools_scm version \
             (contains .devN, .dYYYYMMDD, or +g<sha>). The wheel filename \
             will DRIFT across calendar days even when the commit SHA is \
             pinned, causing lock drift on replay. Fix: ensure the upstream \
             repo has a reachable tag at the pinned commit, or set \
             SETUPTOOLS_SCM_PRETEND_VERSION=<version> in the build env.",
        );
    }

    Ok(GitWheelBuild {
        wheel_path,
        resolved_sha,
        checkout,
        canonical,
        project_root,
    })
}

/// Returns `true` when a wheel filename contains markers of a
/// non-reproducible `setuptools_scm`-style version:
/// - `.devN` — development distance (e.g. `1.1.1.dev4`)
/// - `.dYYYYMMDD` — local date segment (e.g. `+g1234.d20250101`)
/// - `+g<hex>` — local git-hash segment
///
/// These cause the version/filename to change daily even for a pinned
/// commit SHA, breaking byte-identical lock replay.
pub fn is_nondeterministic_version(filename: &str) -> bool {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        // Matches any of:
        //   .devN    (development distance)
        //   .dYYYYMMDD  (local date in setuptools_scm local segment)
        //   +g<hexchars>  (local git-hash segment)
        regex::Regex::new(r"(?:\.dev\d+|\.d\d{8}|\+g[0-9a-f]+)").unwrap()
    });
    re.is_match(filename)
}

/// Run a command silently and return its trimmed stdout as a `String`.
/// Fails if the command exits non-zero.
async fn run_output(cmd: &mut Command, label: &str) -> Result<String> {
    let output = run_output_bytes(cmd, label).await?;
    Ok(String::from_utf8_lossy(&output).into_owned())
}

/// Byte-preserving counterpart to [`run_output`]. Git ref names are byte
/// strings on Unix, so cache identities must not collapse distinct names via
/// lossy UTF-8 replacement.
async fn run_output_bytes(cmd: &mut Command, label: &str) -> Result<Vec<u8>> {
    let output = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .with_context(|| format!("spawning {label}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "{label} failed (status {}): {}",
            output.status,
            stderr.trim()
        );
    }
    Ok(output.stdout)
}

#[cfg(unix)]
struct UnixProcessGroupGuard {
    pgid: nix::unistd::Pid,
    armed: bool,
    label: String,
}

#[cfg(unix)]
impl UnixProcessGroupGuard {
    fn new(pgid: u32, label: impl Into<String>) -> Result<Self> {
        let pgid = i32::try_from(pgid).context("child process id exceeds Unix pid_t range")?;
        Ok(Self {
            pgid: nix::unistd::Pid::from_raw(pgid),
            armed: true,
            label: label.into(),
        })
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

#[cfg(unix)]
impl Drop for UnixProcessGroupGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        match nix::sys::signal::killpg(self.pgid, nix::sys::signal::Signal::SIGKILL) {
            Ok(()) | Err(nix::errno::Errno::ESRCH) => {}
            Err(error) => tracing::warn!(
                pgid = self.pgid.as_raw(),
                error = %error,
                label = %self.label,
                "failed to kill cancelled child process group",
            ),
        }
    }
}

/// Invoke `uv` with the given args, capturing stdout + stderr so neither
/// leaks to retread's stdout (which is the JSON-RPC channel to pixi).
/// Sets `UV_PYTHON_DOWNLOADS=automatic` so missing pythons are fetched
/// on demand without user intervention.
async fn run_capturing_uv(args: &[&str]) -> Result<()> {
    // The callers above have already exhausted their wheel-cache paths. Hold
    // the process-wide permit only for the real build subprocess so nested
    // handler concurrency cannot multiply expensive `uv build` work.
    let _build_permit = crate::concurrency::acquire_build_permit().await;
    let mut cmd = Command::new("uv");
    for arg in args {
        cmd.arg(arg);
    }
    crate::fasttmp::apply_backend_env(&mut cmd);
    #[cfg(unix)]
    cmd.process_group(0);
    let child = cmd
        .env("UV_PYTHON_DOWNLOADS", "automatic")
        // Canonical Git sources intentionally share one read-only metadata
        // store across subdirectory builds. SCM probes may read it, but Git
        // must not refresh its index or take optional locks there.
        .env("GIT_OPTIONAL_LOCKS", "0")
        // If pixi cancels a backend request, dropping this future must also
        // terminate the direct uv child. The Unix process-group guard below
        // additionally terminates Python/compiler descendants.
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawning uv (is it on PATH? expected via retread's runtime dep)")?;
    // Keep this guard declared after `_build_permit`: Rust drops locals in
    // reverse declaration order, so cancellation kills the complete process
    // group before releasing capacity to a replacement build.
    #[cfg(unix)]
    let mut process_group = UnixProcessGroupGuard::new(
        child
            .id()
            .context("spawned uv process has no operating-system pid")?,
        "uv build",
    )?;
    let output = child
        .wait_with_output()
        .await
        .context("waiting for uv build subprocess")?;
    #[cfg(unix)]
    process_group.disarm();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        tracing::error!(stdout = %stdout, stderr = %stderr, args = ?args, "uv failed");
        // Include stderr in the bail message so pixi surfaces the
        // actual uv error (usage / network / build failure) instead
        // of a bare "status 2".
        let snippet = stderr.trim();
        let snippet = if snippet.len() > 2000 {
            format!("{}...(truncated)", &snippet[..2000])
        } else {
            snippet.to_string()
        };
        bail!("uv {:?} failed (status {}): {snippet}", args, output.status,);
    }
    if !stdout.trim().is_empty() {
        tracing::debug!(stdout = %stdout, "uv output");
    }
    Ok(())
}

/// Run a child process, capturing stdout + stderr so neither leaks to
/// retread's stdout (which is the JSON-RPC channel). Fail with the
/// captured output attached if the child exits non-zero. v0.13.4+:
/// stderr is included in the bail message so the underlying tool's
/// real error (e.g. git's "Cannot create file '<some-200-char-name>':
/// File name too long") surfaces in the pixi JSON-RPC error instead
/// of getting buried in trace logs that nobody reads. The single
/// "status N" we used to emit was useless for diagnosing upstream
/// issues like ENAMETOOLONG on git checkout.
async fn run_silent(cmd: &mut Command, label: &str) -> Result<()> {
    let output = run_mutating_command(cmd, label).await?;
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::error!(label = %label, stdout = %stdout, stderr = %stderr, "{label} failed");
        // Inline the stderr snippet (and stdout if non-empty -- some
        // tools shovel errors to stdout). Cap at 4KB so a runaway
        // child doesn't drown the JSON-RPC error.
        let snippet_for = |s: &str| -> String {
            let s = s.trim();
            if s.len() > 4096 {
                format!("{}...(truncated)", &s[..4096])
            } else {
                s.to_string()
            }
        };
        let stderr_snip = snippet_for(&stderr);
        let stdout_snip = snippet_for(&stdout);
        let detail = match (stderr_snip.is_empty(), stdout_snip.is_empty()) {
            (true, true) => String::new(),
            (false, true) => format!(": {stderr_snip}"),
            (true, false) => format!(": (stdout) {stdout_snip}"),
            (false, false) => format!(": {stderr_snip} | (stdout) {stdout_snip}"),
        };
        bail!("{label} failed (status {}){detail}", output.status);
    }
    Ok(())
}

/// Run one mutating command with bounded caller-provided standard input. This
/// retains the same cancellation/process-group contract as [`run_silent`],
/// while allowing Git's transactional `update-ref --stdin` protocol instead
/// of spawning one process per ref.
async fn run_silent_with_input(cmd: &mut Command, label: &str, input: &[u8]) -> Result<()> {
    #[cfg(unix)]
    cmd.process_group(0);
    let mut child = cmd
        .kill_on_drop(true)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {label} (is the tool on PATH?)"))?;
    #[cfg(unix)]
    let mut process_group = UnixProcessGroupGuard::new(
        child
            .id()
            .context("spawned mutating child has no operating-system pid")?,
        label,
    )?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("{label} did not expose piped stdin"))?;
    stdin
        .write_all(input)
        .await
        .with_context(|| format!("writing {label} stdin"))?;
    drop(stdin);
    let output = child
        .wait_with_output()
        .await
        .with_context(|| format!("waiting for {label}"))?;
    #[cfg(unix)]
    process_group.disarm();
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "{label} failed (status {}): {}{}",
            output.status,
            stderr.trim(),
            if stdout.is_empty() {
                String::new()
            } else {
                format!(" | (stdout) {}", stdout.trim())
            },
        );
    }
    Ok(())
}

/// Run a checkout-mutating child in its own Unix process group. The group
/// guard is declared inside the caller that still owns the checkout's EX
/// transaction, so cancellation kills ordinary git helpers/transports before
/// that outer transaction can drop and unlock. Windows retains Tokio's
/// direct-child `kill_on_drop` behavior; process-tree parity there requires a
/// Job Object and is intentionally not claimed here.
async fn run_mutating_command(cmd: &mut Command, label: &str) -> Result<std::process::Output> {
    #[cfg(unix)]
    cmd.process_group(0);
    let child = cmd
        // Clone/checkout/clean/fetch run while the one-time EX transaction is
        // armed. If that task is aborted during runtime shutdown, do not let a
        // detached direct child continue mutating after the RAII guard unlocks.
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {label} (is the tool on PATH?)"))?;
    #[cfg(unix)]
    let mut process_group = UnixProcessGroupGuard::new(
        child
            .id()
            .context("spawned mutating child has no operating-system pid")?,
        label,
    )?;
    let output = child
        .wait_with_output()
        .await
        .with_context(|| format!("waiting for {label}"))?;
    #[cfg(unix)]
    process_group.disarm();
    Ok(output)
}

/// Like [`run_silent`] but returns `Ok(false)` instead of failing when
/// the child exits non-zero. Used by paths that have a fallback (e.g.,
/// `git checkout` -> `git fetch` -> `git checkout`).
async fn try_run_silent(cmd: &mut Command) -> Result<bool> {
    let output = run_mutating_command(cmd, "git subprocess").await?;
    Ok(output.status.success())
}

/// Check out `target` (a rev, tag, branch, or `FETCH_HEAD`) in `clone_dir`,
/// self-healing the two corruption modes a pre-v3.0.0 concurrent-resolve
/// race could leave behind (#8): stray untracked files blocking the
/// checkout ("untracked working tree files would be overwritten"), or a
/// previous checkout simply parked on the wrong commit. This helper is only
/// used by the one-time EX initializer before it publishes readiness; it must
/// never run against a warm checkout that readers can access.
///
/// Returns `Ok(false)` (not an error) when `target` isn't resolvable at
/// all in this clone_dir -- the caller's fallback is to `git fetch` first.
async fn checkout_rev_robust(clone_dir: &Path, target: &str) -> Result<bool> {
    if try_run_silent(
        Command::new("git")
            .args(["checkout", target])
            .current_dir(clone_dir),
    )
    .await?
    {
        return Ok(true);
    }
    // First attempt failed -- most likely stray untracked files left by a
    // prior corrupted run. `git clean -fdx` clears untracked AND
    // gitignored files (safe here: clone_dir only ever holds the
    // checkout itself, wheel output lands in a separate out_dir), then
    // retry once. If `target` still isn't resolvable locally, this
    // second attempt fails the same way a doomed checkout always would.
    #[cfg(test)]
    record_checkout_repair(clone_dir);
    run_silent(
        Command::new("git")
            .args(["clean", "-fdx"])
            .current_dir(clone_dir),
        "git clean -fdx (repairing a corrupted checkout)",
    )
    .await?;
    try_run_silent(
        Command::new("git")
            .args(["checkout", target])
            .current_dir(clone_dir),
    )
    .await
}

async fn find_built_wheel(dir: &Path) -> Result<PathBuf> {
    let mut read = tokio::fs::read_dir(dir)
        .await
        .with_context(|| format!("opening wheel-build dir {}", dir.display()))?;
    let mut wheels = Vec::new();
    while let Some(entry) = read
        .next_entry()
        .await
        .with_context(|| format!("reading wheel-build dir {}", dir.display()))?
    {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.ends_with(".whl") {
            wheels.push(path);
        }
    }
    match wheels.as_slice() {
        [wheel] => Ok(wheel.clone()),
        [] => bail!("source build produced no wheel in {}", dir.display()),
        _ => {
            wheels.sort();
            bail!(
                "source build produced {} wheels in {}; expected exactly one: {}",
                wheels.len(),
                dir.display(),
                wheels
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            )
        }
    }
}

/// Sanitize a git URL into a filesystem-safe slug for cache key.
fn git_slug(url: &str) -> String {
    url.replace(['/', ':', '@'], "_")
        .replace("https___github.com_", "")
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHECKOUT_SUBPROCESS_HELPER: &str = "source_build::tests::git_checkout_subprocess_helper";
    const UV_BUILD_LIMIT_SUBPROCESS_HELPER: &str =
        "source_build::tests::uv_build_limit_subprocess_helper";

    fn unique_test_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "retread-{label}-{}-{}",
            std::process::id(),
            BUILD_TMP_SEQUENCE.fetch_add(1, Ordering::Relaxed),
        ))
    }

    struct CheckoutTestChild {
        child: Option<std::process::Child>,
        label: String,
    }

    impl CheckoutTestChild {
        fn spawn(label: &str, mode: &str, environment: &[(&str, String)]) -> Self {
            Self::spawn_exact(label, CHECKOUT_SUBPROCESS_HELPER, mode, environment)
        }

        fn spawn_exact(
            label: &str,
            test_name: &str,
            mode: &str,
            environment: &[(&str, String)],
        ) -> Self {
            let mut command = std::process::Command::new(
                std::env::current_exe().expect("locate current Rust test executable"),
            );
            command
                .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
                .env("RETREAD_TEST_CHILD_MODE", mode)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            for (name, value) in environment {
                command.env(name, value);
            }
            let child = command.spawn().expect("spawn checkout test subprocess");
            Self {
                child: Some(child),
                label: label.to_string(),
            }
        }

        fn wait_for_signal(&mut self, path: &Path, timeout: Duration) {
            let deadline = std::time::Instant::now() + timeout;
            loop {
                if path.exists() {
                    return;
                }
                if let Some(status) = self
                    .child
                    .as_mut()
                    .expect("checkout test child already consumed")
                    .try_wait()
                    .expect("poll checkout test subprocess")
                {
                    let output = self
                        .child
                        .take()
                        .expect("checkout test child already consumed")
                        .wait_with_output()
                        .expect("collect checkout test subprocess output");
                    panic!(
                        "{} exited before signal {} ({status}):\nstdout:\n{}\nstderr:\n{}",
                        self.label,
                        path.display(),
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr),
                    );
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "{} timed out waiting for signal {}",
                    self.label,
                    path.display(),
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        }

        fn finish(mut self, timeout: Duration) {
            let deadline = std::time::Instant::now() + timeout;
            loop {
                let status = self
                    .child
                    .as_mut()
                    .expect("checkout test child already consumed")
                    .try_wait()
                    .expect("poll checkout test subprocess");
                if let Some(status) = status {
                    let output = self
                        .child
                        .take()
                        .expect("checkout test child already consumed")
                        .wait_with_output()
                        .expect("collect checkout test subprocess output");
                    assert!(
                        status.success(),
                        "{} failed ({status}):\nstdout:\n{}\nstderr:\n{}",
                        self.label,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr),
                    );
                    return;
                }
                if std::time::Instant::now() >= deadline {
                    let mut child = self
                        .child
                        .take()
                        .expect("checkout test child already consumed");
                    let _ = child.kill();
                    let output = child
                        .wait_with_output()
                        .expect("collect timed-out checkout test subprocess output");
                    panic!(
                        "{} timed out and was killed:\nstdout:\n{}\nstderr:\n{}",
                        self.label,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr),
                    );
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn recorded_uv_processes(
        state_dir: &Path,
        prefix: &str,
    ) -> std::collections::BTreeMap<String, u32> {
        let mut recorded = std::collections::BTreeMap::new();
        for entry in std::fs::read_dir(state_dir).expect("read fake-uv state directory") {
            let entry = entry.expect("read fake-uv state entry");
            let name = entry.file_name();
            let Some(id) = name.to_str().and_then(|name| name.strip_prefix(prefix)) else {
                continue;
            };
            let Ok(pid) = std::fs::read_to_string(entry.path()) else {
                continue;
            };
            let Ok(pid) = pid.trim().parse() else {
                continue;
            };
            recorded.insert(id.to_string(), pid);
        }
        recorded
    }

    #[cfg(target_os = "linux")]
    fn started_uv_processes(state_dir: &Path) -> std::collections::BTreeMap<String, u32> {
        recorded_uv_processes(state_dir, "started-")
    }

    #[cfg(target_os = "linux")]
    fn uv_grandchildren(state_dir: &Path) -> std::collections::BTreeMap<String, u32> {
        recorded_uv_processes(state_dir, "grandchild-")
    }

    #[cfg(target_os = "linux")]
    fn process_is_running(pid: u32) -> bool {
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
            return false;
        };
        // A killed child may briefly remain as a zombie until Tokio reaps it;
        // that is terminated for this regression's purposes.
        stat.rsplit_once(") ")
            .and_then(|(_, rest)| rest.chars().next())
            .is_some_and(|state| state != 'Z')
    }

    #[cfg(target_os = "linux")]
    async fn wait_for_test_condition(label: &str, mut condition: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !condition() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {label}"));
    }

    /// Subprocess-isolated because the production semaphore and parsed
    /// environment setting are intentionally initialized once per process.
    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn uv_build_limit_subprocess_helper() {
        if std::env::var("RETREAD_TEST_CHILD_MODE").as_deref() != Ok("uv-build-limit") {
            return;
        }
        let state_dir = PathBuf::from(
            std::env::var_os("RETREAD_TEST_UV_STATE").expect("missing fake-uv state directory"),
        );
        let ids = ["one", "two", "three"];
        let mut tasks = Vec::new();
        for id in ids {
            tasks.push(Some(tokio::spawn(async move {
                run_capturing_uv(&["build", id]).await
            })));
        }

        wait_for_test_condition("two capped uv children", || {
            started_uv_processes(&state_dir).len() >= 2
        })
        .await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        let first_wave = started_uv_processes(&state_dir);
        assert_eq!(
            first_wave.len(),
            2,
            "cap=2 allowed a third uv build to start: {first_wave:?}",
        );

        let cancelled_id = first_wave.keys().next().expect("one started uv id").clone();
        let cancelled_pid = first_wave[&cancelled_id];
        let cancelled_grandchild_pid = uv_grandchildren(&state_dir)
            .get(&cancelled_id)
            .copied()
            .expect("started fake uv recorded its grandchild pid");
        let cancelled_index = ids
            .iter()
            .position(|id| *id == cancelled_id)
            .expect("started uv id belongs to a task");
        let cancelled = tasks[cancelled_index]
            .take()
            .expect("cancelled task still present");
        cancelled.abort();
        assert!(
            cancelled
                .await
                .expect_err("aborted build task completed")
                .is_cancelled(),
            "build task did not report cancellation",
        );
        wait_for_test_condition("cancelled uv child termination", || {
            !process_is_running(cancelled_pid)
        })
        .await;
        wait_for_test_condition("cancelled uv grandchild termination", || {
            !process_is_running(cancelled_grandchild_pid)
        })
        .await;
        wait_for_test_condition("third uv child after permit release", || {
            started_uv_processes(&state_dir).len() == 3
        })
        .await;

        for task in tasks.into_iter().flatten() {
            task.abort();
            assert!(
                task.await
                    .expect_err("aborted build task completed")
                    .is_cancelled()
            );
        }
        for pid in started_uv_processes(&state_dir).into_values() {
            wait_for_test_condition("fake uv cleanup", || !process_is_running(pid)).await;
        }
        for pid in uv_grandchildren(&state_dir).into_values() {
            wait_for_test_condition("fake uv grandchild cleanup", || !process_is_running(pid))
                .await;
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn uv_builds_are_capped_and_killed_on_cancellation() {
        use std::os::unix::fs::PermissionsExt;

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "retread-uv-build-limit-{}-{unique}",
            std::process::id()
        ));
        let bin_dir = base.join("bin");
        let state_dir = base.join("state");
        std::fs::create_dir_all(&bin_dir).unwrap();
        std::fs::create_dir_all(&state_dir).unwrap();
        let fake_uv = bin_dir.join("uv");
        std::fs::write(
            &fake_uv,
            b"#!/bin/sh\nsleep 30 &\ngrandchild=$!\ngrandchild_tmp=\"${RETREAD_TEST_UV_STATE}/.grandchild-$2-$$\"\nprintf '%s\\n' \"$grandchild\" > \"$grandchild_tmp\"\nmv \"$grandchild_tmp\" \"${RETREAD_TEST_UV_STATE}/grandchild-$2\"\nstarted_tmp=\"${RETREAD_TEST_UV_STATE}/.started-$2-$$\"\nprintf '%s\\n' \"$$\" > \"$started_tmp\"\nmv \"$started_tmp\" \"${RETREAD_TEST_UV_STATE}/started-$2\"\nwait \"$grandchild\"\n",
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&fake_uv).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&fake_uv, permissions).unwrap();
        let path = format!(
            "{}:{}",
            bin_dir.display(),
            std::env::var_os("PATH")
                .unwrap_or_default()
                .to_string_lossy()
        );

        CheckoutTestChild::spawn_exact(
            "uv build concurrency/cancellation probe",
            UV_BUILD_LIMIT_SUBPROCESS_HELPER,
            "uv-build-limit",
            &[
                ("RETREAD_MAX_CONCURRENT_BUILDS", "2".to_string()),
                ("RETREAD_TEST_UV_STATE", state_dir.display().to_string()),
                ("PATH", path),
            ],
        )
        .finish(Duration::from_secs(15));
        let _ = std::fs::remove_dir_all(&base);
    }

    impl Drop for CheckoutTestChild {
        fn drop(&mut self) {
            let Some(child) = self.child.as_mut() else {
                return;
            };
            if child.try_wait().ok().flatten().is_none() {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
    }

    async fn wait_for_checkout_signal(variable: &str) {
        let path = PathBuf::from(
            std::env::var_os(variable)
                .unwrap_or_else(|| panic!("checkout test subprocess missing {variable}")),
        );
        tokio::time::timeout(Duration::from_secs(5), async {
            while !path.exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {variable} at {}", path.display()));
    }

    #[test]
    fn git_slug_strips_github_prefix() {
        assert_eq!(
            git_slug("https://github.com/isaac-sim/IsaacLab.git"),
            "isaac-sim_IsaacLab.git"
        );
    }

    /// v0.13.3+ regression: every on-disk path component in the
    /// checkout-root path is independently bounded. Layout is
    /// cache/retread-git-clones/v3/<slug<=24>/<full-sha256>. Previously the
    /// (slug + 40-char raw SHA) flattened into one 60+ char
    /// component; combined with the rattler cache prefix and deep
    /// IsaacLab internals, pathnames tripped ENAMETOOLONG on git
    /// checkout.
    #[test]
    fn checkout_root_components_are_short() {
        let cache = std::path::Path::new("/tmp/cache");
        let p = git_checkout_root(
            "https://github.com/isaac-sim/IsaacLab.git",
            "867cbf9b7b4edbb03f32e1209c585a38cb3d8edf",
            cache,
        );
        let comps: Vec<String> = p
            .components()
            .filter_map(|c| c.as_os_str().to_str().map(String::from))
            .collect();
        // Last component is the full 64-hex identity; second-to-last is the
        // slug (<=24 chars). Neither is near NAME_MAX / 255.
        let last = comps.last().expect("at least one component");
        let parent = &comps[comps.len() - 2];
        assert_eq!(
            last.len(),
            64,
            "full SHA-256 identity must be exactly 64 chars; got: {last}"
        );
        assert!(parent.len() <= 24, "slug must be <=24 chars; got {parent}");
    }

    /// The process registry must unify repeated opens before flocking, and its
    /// actual production RwLock must exclude the one-time writer from readers
    /// in both directions.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn process_clone_reader_and_writer_guards_never_overlap() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "retread-clone-rwlock-{}-{unique}",
            std::process::id()
        ));
        let clone_dir = base
            .join("retread-git-clones")
            .join("slug")
            .join("abcdef012345");
        std::fs::create_dir_all(clone_dir.parent().unwrap()).unwrap();
        let lock_path = clone_dir.with_extension("lock");
        let identity = CloneIdentity::new("https://example.com/repo.git", "rev");
        let first = registered_clone_lock(&lock_path, &identity).unwrap();
        let second = registered_clone_lock(&lock_path, &identity).unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "same inode must map to one process lock entry"
        );

        let writer = Arc::clone(&first.local).write_owned().await;
        let (reader_entered_tx, mut reader_entered_rx) = tokio::sync::oneshot::channel();
        let (reader_release_tx, reader_release_rx) = tokio::sync::oneshot::channel();
        let reader_lock = Arc::clone(&second.local);
        let reader_task = tokio::spawn(async move {
            let _reader = reader_lock.read_owned().await;
            let _ = reader_entered_tx.send(());
            let _ = reader_release_rx.await;
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut reader_entered_rx)
                .await
                .is_err(),
            "reader entered while the one-time writer was held"
        );
        drop(writer);
        tokio::time::timeout(Duration::from_secs(2), &mut reader_entered_rx)
            .await
            .expect("reader did not enter after writer release")
            .expect("reader entry sender dropped");

        let (writer_entered_tx, mut writer_entered_rx) = tokio::sync::oneshot::channel();
        let writer_lock = Arc::clone(&first.local);
        let writer_task = tokio::spawn(async move {
            let _writer = writer_lock.write_owned().await;
            let _ = writer_entered_tx.send(());
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut writer_entered_rx)
                .await
                .is_err(),
            "writer entered while a reader was held"
        );
        let _ = reader_release_tx.send(());
        reader_task.await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), &mut writer_entered_rx)
            .await
            .expect("writer did not enter after reader release")
            .expect("writer entry sender dropped");
        writer_task.await.unwrap();

        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn process_clone_registry_unifies_path_aliases_by_inode() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "retread-clone-alias-{}-{unique}",
            std::process::id()
        ));
        let real_cache = base.join("real-cache");
        let alias_cache = base.join("alias-cache");
        let real_parent = real_cache.join("retread-git-clones").join("slug");
        std::fs::create_dir_all(&real_parent).unwrap();
        std::os::unix::fs::symlink(&real_cache, &alias_cache).unwrap();
        let real_lock = real_parent.join("abcdef012345.lock");
        let alias_lock = alias_cache
            .join("retread-git-clones")
            .join("slug")
            .join("abcdef012345.lock");
        let identity = CloneIdentity::new("https://example.com/alias.git", "rev");
        let real = registered_clone_lock(&real_lock, &identity).unwrap();
        let alias = registered_clone_lock(&alias_lock, &identity).unwrap();
        assert!(
            Arc::ptr_eq(&real, &alias),
            "two paths to one lock inode created separate process entries"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Different (url, rev) pairs must NOT collide on disk -- the
    /// rev is the only thing distinguishing two checkouts of the same
    /// repo at different revisions.
    #[test]
    fn checkout_root_distinct_revs_do_not_collide() {
        let cache = std::path::Path::new("/tmp/cache");
        let a = git_checkout_root("https://example.com/r.git", "rev-a", cache);
        let b = git_checkout_root("https://example.com/r.git", "rev-b", cache);
        assert_ne!(a, b);
    }

    /// And two DIFFERENT repos at the same revision name must also
    /// differ (the url is hashed into the key alongside the rev).
    #[test]
    fn checkout_root_distinct_urls_do_not_collide() {
        let cache = std::path::Path::new("/tmp/cache");
        let a = git_checkout_root("https://example.com/r1.git", "main", cache);
        let b = git_checkout_root("https://example.com/r2.git", "main", cache);
        assert_ne!(a, b);
    }

    // ---------------------------------------------------------------------------
    // Determinism guard: is_nondeterministic_version
    // ---------------------------------------------------------------------------

    #[test]
    fn deterministic_version_not_flagged() {
        // Static release versions must NOT trigger the guard.
        assert!(!is_nondeterministic_version("mylib-1.1.1-py3-none-any.whl"));
        assert!(!is_nondeterministic_version(
            "newton-1.3.0-py3-none-any.whl"
        ));
        assert!(!is_nondeterministic_version(
            "genesis_world-1.1.1-py3-none-any.whl"
        ));
        assert!(!is_nondeterministic_version(
            "foo-2.0.0rc1-py3-none-any.whl"
        ));
        assert!(!is_nondeterministic_version(
            "bar-0.1.0.post1-py3-none-any.whl"
        ));
    }

    #[test]
    fn dev_version_is_flagged() {
        // .devN suffix (development distance without local segment).
        assert!(is_nondeterministic_version(
            "mylib-1.1.1.dev4-py3-none-any.whl"
        ));
        assert!(is_nondeterministic_version(
            "mylib-0.1.dev123-py3-none-any.whl"
        ));
    }

    #[test]
    fn date_segment_is_flagged() {
        // .dYYYYMMDD local date segment produced by setuptools_scm.
        assert!(is_nondeterministic_version(
            "mylib-1.0.dev4+g1234567.d20250101-py3-none-any.whl"
        ));
        assert!(is_nondeterministic_version(
            "mylib-1.0.dev0+g0000000.d20991231-py3-none-any.whl"
        ));
    }

    #[test]
    fn local_git_sha_segment_is_flagged() {
        // +g<hexchars> local git-hash segment.
        assert!(is_nondeterministic_version(
            "mylib-1.0+gabcdef0-py3-none-any.whl"
        ));
        assert!(is_nondeterministic_version(
            "mylib-2.0.post0+g1234abc-py3-none-any.whl"
        ));
    }

    /// Determinism guard (Amendment 3): build_wheel_from_sdist_url must warn
    /// on a non-reproducible version and be silent on a static one.
    /// Tests the guard logic that was added to mirror build_wheel_from_git.
    #[test]
    fn sdist_determinism_guard_matches_git_guard() {
        // Static released version (e.g. gym 0.26.2) — NO warn.
        assert!(
            !is_nondeterministic_version("gym-0.26.2-py3-none-any.whl"),
            "gym 0.26.2 is a static version; determinism guard must NOT fire"
        );
        // .dYYYYMMDD date segment — MUST warn.
        assert!(
            is_nondeterministic_version("mypkg-1.0.dev4+g1234567.d20250101-py3-none-any.whl"),
            "setuptools_scm date suffix must trigger determinism guard"
        );
        // .devN without date — MUST warn.
        assert!(
            is_nondeterministic_version("mypkg-0.1.dev5-py3-none-any.whl"),
            ".devN suffix must trigger determinism guard"
        );
        // +g<sha> local version — MUST warn.
        assert!(
            is_nondeterministic_version("mypkg-1.0+gabcdef0-py3-none-any.whl"),
            "+g<sha> local version must trigger determinism guard"
        );
    }

    #[tokio::test]
    async fn sdist_hash_mismatch_fails_before_cache_write_or_build() {
        use std::io::{Read as _, Write as _};

        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 2048];
            let _ = stream.read(&mut request).unwrap();
            let body = b"untrusted-sdist-bytes";
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len(),
            )
            .unwrap();
            stream.write_all(body).unwrap();
        });
        let url = url::Url::parse(&format!("http://{address}/demo-1.0.tar.gz")).unwrap();
        let out_dir = std::env::temp_dir().join(format!(
            "retread-sdist-hash-mismatch-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));

        let err = build_wheel_from_sdist_url(&url, &out_dir, "3.11", Some(&"00".repeat(32)))
            .await
            .unwrap_err();
        server.join().unwrap();

        assert!(format!("{err:#}").contains("sdist sha256 mismatch"));
        assert!(
            !out_dir.exists(),
            "unverified sdist bytes must not publish a cache directory"
        );
    }

    // ---------------------------------------------------------------------------
    // Local git fixture: build_wheel_from_git resolves and retains checkout
    // ---------------------------------------------------------------------------

    /// Verifies that `build_wheel_from_git` returns a 40-character resolved
    /// SHA and that the SHA is stable (calling again with the same rev returns
    /// the same SHA). Uses a minimal local git repo so no network access is
    /// required and CI stays fast.
    #[tokio::test]
    #[ignore = "live: builds a git wheel via uv (needs uv + git on PATH); run with --include-ignored"]
    async fn build_wheel_from_git_returns_resolved_sha() {
        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("retread-gitfixture-{pid}"));
        let repo = base.join("repo");
        std::fs::create_dir_all(&repo).expect("create repo dir");

        // Init git repo.
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

        // Write a minimal but buildable Python package.
        std::fs::write(
            repo.join("pyproject.toml"),
            r#"[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "retread-test-fixture"
version = "0.1.0"
"#,
        )
        .expect("write pyproject");
        std::fs::write(repo.join("README.md"), "test fixture").expect("write README");

        run_git(&["add", "."], &repo);
        run_git(&["commit", "-m", "initial"], &repo);

        // Get the commit SHA directly.
        let sha_output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&repo)
            .output()
            .expect("git rev-parse");
        let expected_sha = String::from_utf8_lossy(&sha_output.stdout)
            .trim()
            .to_string();
        assert_eq!(
            expected_sha.len(),
            40,
            "git rev-parse HEAD must be 40 chars"
        );

        let cache_dir = base.join("cache");
        let out_dir = base.join("out");
        std::fs::create_dir_all(&cache_dir).expect("cache dir");
        std::fs::create_dir_all(&out_dir).expect("out dir");
        let repo_url = format!("file://{}", repo.display());

        let (wheel_path, resolved_sha) =
            build_wheel_from_git(&repo_url, &expected_sha, ".", &cache_dir, &out_dir, "3.11")
                .await
                .expect("build_wheel_from_git");

        // The returned SHA must match what git reports.
        assert_eq!(
            resolved_sha, expected_sha,
            "resolved_sha must equal the commit SHA"
        );
        assert_eq!(resolved_sha.len(), 40, "resolved_sha must be 40 hex chars");
        // A wheel must have been produced.
        assert!(
            wheel_path.extension().is_some_and(|e| e == "whl"),
            "built file must be a .whl"
        );
        // The static version "0.1.0" must NOT be flagged as non-deterministic.
        let filename = wheel_path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("filename");
        assert!(
            !is_nondeterministic_version(filename),
            "a static version should not be flagged: {filename}"
        );

        // Cleanup.
        let _ = std::fs::remove_dir_all(&base);
    }

    struct GitCheckoutFixture {
        base: PathBuf,
        cache: PathBuf,
        url: String,
        rev1: String,
        rev2: String,
    }

    impl Drop for GitCheckoutFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.base);
        }
    }

    fn run_fixture_git(args: &[&str], directory: &Path) -> String {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(directory)
            .env("GIT_AUTHOR_NAME", "retread-test")
            .env("GIT_AUTHOR_EMAIL", "retread-test@example.com")
            .env("GIT_COMMITTER_NAME", "retread-test")
            .env("GIT_COMMITTER_EMAIL", "retread-test@example.com")
            .env("GIT_OPTIONAL_LOCKS", "0")
            .output()
            .expect("spawn fixture git");
        assert!(
            output.status.success(),
            "git {args:?} failed in {}: {}",
            directory.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn git_checkout_fixture(label: &str) -> GitCheckoutFixture {
        static NEXT_FIXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let base =
            std::env::temp_dir().join(format!("retread-{label}-{}-{sequence}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let repo = base.join("repo");
        let cache = base.join("cache");
        std::fs::create_dir_all(&repo).expect("create fixture repo");
        std::fs::create_dir_all(&cache).expect("create fixture cache");
        run_fixture_git(&["init", "-b", "main"], &repo);
        std::fs::write(repo.join("base.txt"), "base\n").expect("write base file");
        run_fixture_git(&["add", "."], &repo);
        run_fixture_git(&["commit", "-m", "base"], &repo);
        let rev1 = run_fixture_git(&["rev-parse", "HEAD"], &repo);
        std::fs::write(repo.join("extra.txt"), "tracked-content\n")
            .expect("write second-revision file");
        run_fixture_git(&["add", "."], &repo);
        run_fixture_git(&["commit", "-m", "add extra"], &repo);
        let rev2 = run_fixture_git(&["rev-parse", "HEAD"], &repo);
        let url = format!("file://{}", repo.display());
        GitCheckoutFixture {
            base,
            cache,
            url,
            rev1,
            rev2,
        }
    }

    fn seed_unmarked_checkout(fixture: &GitCheckoutFixture, checkout_rev: &str) -> PathBuf {
        let clone_dir = git_checkout_root(&fixture.url, &fixture.rev2, &fixture.cache);
        std::fs::create_dir_all(clone_dir.parent().unwrap()).expect("create clone parent");
        let clone_arg = clone_dir.to_str().expect("utf8 clone path");
        run_fixture_git(
            &["clone", "--no-checkout", &fixture.url, clone_arg],
            &fixture.base,
        );
        run_fixture_git(&["checkout", "--force", checkout_rev], &clone_dir);
        assert!(
            !checkout_ready_marker(&clone_dir).exists(),
            "legacy fixture must begin unpublished"
        );
        clone_dir
    }

    #[tokio::test]
    async fn canonical_ref_deletion_handles_symref_after_ordinary_ref() {
        let fixture = git_checkout_fixture("canonical-ref-symref-order");
        let repo = fixture.base.join("repo");
        run_fixture_git(
            &["update-ref", "refs/remotes/origin/3.5-hotfix", "HEAD"],
            &repo,
        );
        run_fixture_git(&["update-ref", "refs/remotes/origin/main", "HEAD"], &repo);
        run_fixture_git(
            &[
                "symbolic-ref",
                "refs/remotes/origin/HEAD",
                "refs/remotes/origin/main",
            ],
            &repo,
        );

        delete_canonical_git_refs(&repo, "refs/remotes")
            .await
            .expect("delete remote refs without dereferencing origin/HEAD");

        assert!(
            run_fixture_git(&["for-each-ref", "refs/remotes"], &repo).is_empty(),
            "canonicalization must remove both symbolic and direct remote refs",
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn readonly_canonical_git_status_allows_lfs_filter_scratch() {
        let fixture = git_checkout_fixture("canonical-lfs-status");
        let repo = fixture.base.join("repo");
        let filter = fixture.base.join("lfs-clean-filter.sh");
        let marker = fixture.base.join("lfs-clean-ran");
        let escaped_lfs = fixture.base.join("escaped-lfs");
        std::fs::write(
            &filter,
            format!(
                "#!/bin/sh\nset -eu\nstorage=$(git config --get lfs.storage)\nprintf '%s\\n' \"$storage\" > '{}'\nif [ \"$storage\" != lfs ]; then mkdir -p '{}'; : > '{}/escaped'; fi\ntmp=.git/lfs/tmp/retread-filter-$$\n: > \"$tmp\"\ncat\n",
                marker.display(),
                escaped_lfs.display(),
                escaped_lfs.display(),
            ),
        )
        .expect("write clean-filter fixture");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&filter, std::fs::Permissions::from_mode(0o755))
                .expect("make clean-filter fixture executable");
        }
        let clean_command = format!("{} %f", filter.display());
        run_fixture_git(
            &["config", "filter.retread-lfs.clean", &clean_command],
            &repo,
        );
        run_fixture_git(&["config", "filter.retread-lfs.smudge", "cat"], &repo);
        run_fixture_git(&["config", "filter.retread-lfs.required", "true"], &repo);
        run_fixture_git(&["config", "lfs.storage", "lfs"], &repo);
        std::fs::write(repo.join(".gitattributes"), "*.lfs filter=retread-lfs\n")
            .expect("write filter attributes");
        std::fs::write(repo.join("fixture.lfs"), "tracked lfs-style bytes\n")
            .expect("write filtered fixture");
        reset_canonical_git_lfs_tmp(&repo).expect("prepare filter scratch before git add");
        run_fixture_git(&["add", ".gitattributes", "fixture.lfs"], &repo);
        run_fixture_git(&["commit", "-m", "add filtered fixture"], &repo);
        std::fs::remove_file(&marker).expect("clear pre-seal filter marker");
        run_fixture_git(
            &["config", "lfs.storage", escaped_lfs.to_str().unwrap()],
            &repo,
        );

        reset_canonical_git_lfs_tmp(&repo).expect("reset canonical filter scratch");
        normalize_source_tree_times(&repo).expect("normalize canonical fixture times");
        make_source_tree_read_only(&repo).expect("seal canonical fixture");
        make_canonical_git_lfs_tmp_writable(&repo).expect("open only filter scratch");

        let status = validate_canonical_git_worktree(&fixture.base, &repo)
            .await
            .expect("status must run its clean filter against read-only source bytes");
        assert!(status.is_empty());
        assert_eq!(
            std::fs::read_to_string(&marker).expect("read post-seal filter marker"),
            "lfs\n",
            "canonical validation must override repository/user LFS storage",
        );
        assert!(
            !escaped_lfs.exists(),
            "canonical validation must not write to configured external LFS storage",
        );
        assert!(
            std::fs::read_dir(repo.join(".git/lfs/tmp"))
                .expect("read filter scratch")
                .next()
                .is_none(),
            "clean-filter scratch must remain empty after validation",
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for sealed_directory in [repo.clone(), repo.join(".git"), repo.join(".git/lfs")] {
                assert_eq!(
                    std::fs::metadata(&sealed_directory)
                        .unwrap()
                        .permissions()
                        .mode()
                        & 0o222,
                    0,
                    "{} must remain read-only",
                    sealed_directory.display(),
                );
            }
            assert_eq!(
                std::fs::metadata(repo.join("fixture.lfs"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o222,
                0,
            );
            assert_ne!(
                std::fs::metadata(repo.join(".git/lfs/tmp"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o200,
                0,
            );
        }
        make_staging_tree_removable(&repo);
    }

    #[tokio::test]
    async fn canonical_git_source_ignores_warm_dirt_and_binds_tag_state() {
        let fixture = git_checkout_fixture("canonical-source");
        let origin = fixture.base.join("repo");
        run_fixture_git(&["tag", "release-one", &fixture.rev2], &origin);
        run_fixture_git(
            &[
                "tag",
                "--annotate",
                "--message",
                "historical release",
                "history-note",
                &fixture.rev1,
            ],
            &origin,
        );
        // A remote-tracking ref sorting before origin/HEAD exercises the
        // per-command no-deref behavior in canonical ref deletion.
        run_fixture_git(&["branch", "3.5-hotfix", &fixture.rev1], &origin);
        let checkout = ensure_git_checkout(&fixture.url, &fixture.rev2, &fixture.cache)
            .await
            .expect("publish warm checkout");
        let warm = checkout.root();
        std::fs::write(warm.join("base.txt"), "dirty-warm-bytes\n").unwrap();
        std::fs::write(warm.join("warm-sentinel.txt"), "must not enter source\n").unwrap();
        std::fs::write(warm.join(".git/index.lock"), "external warm lock\n").unwrap();

        let first_ref_state = canonical_git_ref_state(warm).await.unwrap();
        let first =
            ensure_canonical_git_snapshot(warm, &fixture.url, &fixture.rev2, &first_ref_state)
                .await
                .expect("prepare first canonical source");
        assert_eq!(
            std::fs::read_to_string(first.root.join("base.txt")).unwrap(),
            "base\n"
        );
        assert!(!first.root.join("warm-sentinel.txt").exists());
        assert!(!first.root.join(".git/index.lock").exists());
        assert_eq!(
            run_fixture_git(&["rev-parse", "HEAD"], &first.root),
            fixture.rev2
        );
        assert_eq!(
            run_fixture_git(&["describe", "--tags", "--exact-match"], &first.root),
            "release-one"
        );
        assert_eq!(
            run_fixture_git(&["remote", "get-url", "origin"], &first.root),
            fixture.url
        );
        assert!(
            run_fixture_git(
                &["status", "--porcelain=v1", "--untracked-files=all"],
                &first.root,
            )
            .is_empty()
        );

        // The published checkout's exact tag snapshot remains authoritative
        // when the live upstream later adds and retargets tags.
        run_fixture_git(&["tag", "--force", "release-one", &fixture.rev1], &origin);
        run_fixture_git(&["tag", "upstream-only", &fixture.rev1], &origin);
        run_fixture_git(&["tag", "release-two", &fixture.rev2], warm);
        let second_ref_state = canonical_git_ref_state(warm).await.unwrap();
        assert_ne!(first_ref_state, second_ref_state);
        assert_ne!(
            canonical_git_ref_state(&origin).await.unwrap(),
            second_ref_state,
        );
        let family = git_wheel_family_identity(&fixture.url, &fixture.rev2, ".");
        assert_ne!(
            git_wheel_source_identity(&family, &first_ref_state),
            git_wheel_source_identity(&family, &second_ref_state),
        );
        let second =
            ensure_canonical_git_snapshot(warm, &fixture.url, &fixture.rev2, &second_ref_state)
                .await
                .expect("prepare second canonical source");
        assert_ne!(first.root.parent(), second.root.parent());
        assert_eq!(
            canonical_git_ref_state(&second.root).await.unwrap(),
            second_ref_state
        );
        assert_eq!(
            run_fixture_git(&["rev-parse", "refs/tags/release-one"], &second.root),
            fixture.rev2,
        );
        assert_eq!(
            run_fixture_git(&["tag", "--list"], &second.root),
            "history-note\nrelease-one\nrelease-two",
        );
        assert_eq!(
            run_fixture_git(&["cat-file", "-t", "refs/tags/history-note"], &second.root),
            "tag",
        );
        assert_eq!(
            run_fixture_git(&["rev-parse", "refs/tags/history-note^{}"], &second.root),
            fixture.rev1,
        );
        assert_eq!(
            run_fixture_git(&["tag", "--list"], warm),
            "history-note\nrelease-one\nrelease-two",
            "canonicalization must not import or rewrite shared tags",
        );
        assert_eq!(
            run_fixture_git(&["rev-parse", "refs/tags/release-one"], &origin),
            fixture.rev1,
        );

        // Canonicalization never repairs or cleans the published warm tree.
        assert_eq!(
            std::fs::read_to_string(warm.join("base.txt")).unwrap(),
            "dirty-warm-bytes\n"
        );
        assert!(warm.join("warm-sentinel.txt").exists());
        assert!(warm.join(".git/index.lock").exists());

        for snapshot in [first, second] {
            let cache_dir = snapshot.root.parent().unwrap().to_path_buf();
            make_staging_tree_removable(&cache_dir);
            let _ = std::fs::remove_dir_all(cache_dir);
        }
    }

    #[tokio::test]
    async fn canonical_git_source_reclones_promisor_from_bound_upstream() {
        let fixture = git_checkout_fixture("canonical-promisor");
        let upstream = fixture.base.join("repo");
        run_fixture_git(&["config", "uploadpack.allowFilter", "true"], &upstream);
        run_fixture_git(&["tag", "promisor-release", &fixture.rev2], &upstream);
        std::fs::write(upstream.join("future.txt"), "future-only blob\n").unwrap();
        run_fixture_git(&["add", "future.txt"], &upstream);
        run_fixture_git(&["commit", "-m", "future revision"], &upstream);
        let future_blob = run_fixture_git(&["rev-parse", "HEAD:future.txt"], &upstream);

        let shared = fixture.base.join("shared-promisor");
        run_fixture_git(
            &[
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                &fixture.url,
                shared.to_str().unwrap(),
            ],
            &fixture.base,
        );
        run_fixture_git(&["checkout", "--force", &fixture.rev2], &shared);
        run_fixture_git(&["config", "remote.origin.promisor", "true"], &shared);
        let ref_state = canonical_git_ref_state(&shared).await.unwrap();

        // Checkout fetched the blobs for rev2, but not the future revision's
        // blob. Poison its promisor URL so local upload-pack cannot repair the
        // deliberately incomplete object store as a side effect of cloning.
        let missing_upstream = format!("file://{}", fixture.base.join("missing").display());
        run_fixture_git(&["config", "remote.origin.url", &missing_upstream], &shared);
        let missing_blob = std::process::Command::new("git")
            .args(["cat-file", "-e", &future_blob])
            .current_dir(&shared)
            .env("GIT_OPTIONAL_LOCKS", "0")
            .output()
            .expect("probe filtered future blob");
        assert!(
            !missing_blob.status.success(),
            "fixture must retain a missing blob after exact checkout"
        );

        let unusable_clone = fixture.base.join("unusable-local-clone");
        let output = std::process::Command::new("git")
            .args(["clone", "--no-local", "--no-checkout", "--"])
            .arg(&shared)
            .arg(&unusable_clone)
            .env("GIT_OPTIONAL_LOCKS", "0")
            .output()
            .expect("probe unusable shared checkout");
        assert!(
            !output.status.success(),
            "fixture must fail if canonicalization clones the shared object store"
        );
        assert!(git_checkout_has_promisor_remote(&shared).await.unwrap());
        let pack_dir = shared.join(".git/objects/pack");
        let pack_files = || {
            let mut files = std::fs::read_dir(&pack_dir)
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect::<Vec<_>>();
            files.sort();
            files
        };
        let shared_packs_before = pack_files();

        let canonical =
            ensure_canonical_git_snapshot(&shared, &fixture.url, &fixture.rev2, &ref_state)
                .await
                .expect("canonicalize promisor checkout from its bound upstream");
        assert_eq!(
            run_fixture_git(&["rev-parse", "HEAD"], &canonical.root),
            fixture.rev2
        );
        assert_eq!(
            run_fixture_git(&["describe", "--tags", "--exact-match"], &canonical.root,),
            "promisor-release"
        );
        assert_eq!(
            run_fixture_git(&["remote", "get-url", "origin"], &canonical.root),
            fixture.url
        );
        assert_eq!(
            canonical_git_ref_state(&canonical.root).await.unwrap(),
            ref_state
        );
        assert_eq!(
            run_fixture_git(&["config", "remote.origin.url"], &shared),
            missing_upstream,
            "canonicalization must not rewrite the published checkout"
        );
        assert_eq!(
            pack_files(),
            shared_packs_before,
            "canonicalization must not lazy-fetch into the published checkout"
        );

        let cache_dir = canonical.root.parent().unwrap().to_path_buf();
        make_staging_tree_removable(&cache_dir);
        let _ = std::fs::remove_dir_all(cache_dir);
    }

    #[tokio::test]
    async fn canonical_git_source_keeps_full_checkout_offline() {
        let fixture = git_checkout_fixture("canonical-full-offline");
        let upstream = fixture.base.join("repo");
        run_fixture_git(&["tag", "offline-release", &fixture.rev2], &upstream);
        let shared = fixture.base.join("shared-full");
        run_fixture_git(
            &[
                "clone",
                "--no-checkout",
                &fixture.url,
                shared.to_str().unwrap(),
            ],
            &fixture.base,
        );
        run_fixture_git(&["checkout", "--force", &fixture.rev2], &shared);
        assert!(!git_checkout_has_promisor_remote(&shared).await.unwrap());
        let ref_state = canonical_git_ref_state(&shared).await.unwrap();

        let parked_upstream = fixture.base.join("parked-upstream");
        std::fs::rename(&upstream, &parked_upstream).unwrap();
        let canonical =
            ensure_canonical_git_snapshot(&shared, &fixture.url, &fixture.rev2, &ref_state)
                .await
                .expect("canonicalize full checkout without its upstream");
        assert_eq!(
            run_fixture_git(&["rev-parse", "HEAD"], &canonical.root),
            fixture.rev2
        );
        assert_eq!(
            run_fixture_git(&["describe", "--tags", "--exact-match"], &canonical.root,),
            "offline-release"
        );
        assert_eq!(
            run_fixture_git(&["remote", "get-url", "origin"], &canonical.root),
            fixture.url
        );
        assert_eq!(
            canonical_git_ref_state(&canonical.root).await.unwrap(),
            ref_state
        );

        let cache_dir = canonical.root.parent().unwrap().to_path_buf();
        make_staging_tree_removable(&cache_dir);
        let _ = std::fs::remove_dir_all(cache_dir);
    }

    #[tokio::test]
    async fn private_git_build_tree_is_writable_and_removed_on_cancellation() {
        let fixture = git_checkout_fixture("private-git-build");
        let checkout = ensure_git_checkout(&fixture.url, &fixture.rev2, &fixture.cache)
            .await
            .expect("publish fixture checkout");
        let ref_state = canonical_git_ref_state(checkout.root()).await.unwrap();
        let canonical =
            ensure_canonical_git_snapshot(checkout.root(), &fixture.url, &fixture.rev2, &ref_state)
                .await
                .expect("prepare canonical fixture");
        let staging_parent = fixture.base.join("private-build-staging");
        std::fs::create_dir_all(&staging_parent).unwrap();
        let cache_leaf = staging_parent.join("cache-leaf");
        let canonical_for_task = canonical.clone();
        let url_for_task = fixture.url.clone();
        let (published, receive_published) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            let staging = unique_staging_dir(&cache_leaf).unwrap();
            let private_out = staging.0.join("build");
            std::fs::create_dir(&private_out).unwrap();
            let private_project = prepare_private_git_build_tree(
                &canonical_for_task,
                &url_for_task,
                Path::new("."),
                &private_out,
            )
            .await
            .unwrap();
            let backend_scratch = private_project.join("retread_fixture.egg-info");
            std::fs::create_dir(&backend_scratch).expect("private Git build tree must be writable");
            published
                .send((staging.0.clone(), backend_scratch))
                .expect("test receiver remains alive");
            std::future::pending::<()>().await;
            drop(staging);
        });
        let (staging_path, backend_scratch) = receive_published
            .await
            .expect("private build task reached cancellation point");
        assert!(backend_scratch.is_dir());
        assert!(!canonical.root.join("retread_fixture.egg-info").exists());
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(
            !staging_path.exists(),
            "cancelling a build must remove its writable private tree",
        );
        validate_canonical_git_snapshot(
            canonical.root.parent().unwrap(),
            &canonical.repository_identity,
            &canonical.resolved_sha,
            &canonical.ref_state,
            true,
        )
        .await
        .expect("private build must not mutate canonical injection source");
        let canonical_cache = canonical.root.parent().unwrap().to_path_buf();
        make_staging_tree_removable(&canonical_cache);
        let _ = std::fs::remove_dir_all(canonical_cache);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_mutating_git_runner_kills_descendant_process_group() {
        let base = unique_test_dir("git-runner-cancel");
        std::fs::create_dir_all(&base).unwrap();
        let started = base.join("started");
        let finished = base.join("finished");
        let started_for_task = started.clone();
        let finished_for_task = finished.clone();
        let task = tokio::spawn(async move {
            run_silent(
                Command::new("/bin/sh").args([
                    "-c",
                    "touch \"$1\"; (sleep 1; touch \"$2\") & wait",
                    "retread-git-cancel-test",
                    &started_for_task.display().to_string(),
                    &finished_for_task.display().to_string(),
                ]),
                "git cancellation test",
            )
            .await
        });
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while !started.exists() && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(started.exists(), "mutating test command never started");
        task.abort();
        let _ = task.await;
        tokio::time::sleep(Duration::from_millis(1200)).await;
        assert!(
            !finished.exists(),
            "a mutating descendant survived cancellation"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Entry point used only by parent tests that need a killable process
    /// boundary. Running it without a mode is an intentional no-op so the
    /// normal libtest pass can include this helper safely.
    #[test]
    fn git_checkout_subprocess_helper() {
        let Ok(mode) = std::env::var("RETREAD_TEST_CHILD_MODE") else {
            return;
        };
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("build checkout test runtime");

        match mode.as_str() {
            "same-process-readers" => runtime.block_on(async {
                let fixture = git_checkout_fixture("child-two-readers");
                let barrier = Arc::new(tokio::sync::Barrier::new(3));

                let first_barrier = Arc::clone(&barrier);
                let first_url = fixture.url.clone();
                let first_rev = fixture.rev2.clone();
                let first_cache = fixture.cache.clone();
                let first = tokio::spawn(async move {
                    first_barrier.wait().await;
                    ensure_git_checkout(&first_url, &first_rev, &first_cache).await
                });

                let second_barrier = Arc::clone(&barrier);
                let second_url = fixture.url.clone();
                let second_rev = fixture.rev2.clone();
                let second_cache = fixture.cache.clone();
                let second = tokio::spawn(async move {
                    second_barrier.wait().await;
                    ensure_git_checkout(&second_url, &second_rev, &second_cache).await
                });

                barrier.wait().await;
                wait_for_checkout_signal("RETREAD_TEST_EXCLUSIVE_HELD").await;
                wait_for_checkout_signal("RETREAD_TEST_SECOND_REGISTRATION").await;
                assert!(
                    !first.is_finished() && !second.is_finished(),
                    "a checkout reader completed while initialization was paused under EX"
                );
                let release = PathBuf::from(
                    std::env::var_os("RETREAD_TEST_EXCLUSIVE_RELEASE")
                        .expect("missing initializer release path"),
                );
                std::fs::write(&release, b"release\n").expect("release checkout initializer");

                let (first, second) = tokio::time::timeout(Duration::from_secs(10), async {
                    (first.await, second.await)
                })
                .await
                .expect("same-process checkout readers deadlocked");
                let first = first
                    .expect("first reader task panicked")
                    .expect("first reader failed");
                let second = second
                    .expect("second reader task panicked")
                    .expect("second reader failed");
                assert!(Arc::ptr_eq(&first.lease.lock, &second.lease.lock));
                assert_eq!(
                    first.lease.lock.os_acquisitions.load(Ordering::Relaxed),
                    1,
                    "same-process callers took more than one OS flock"
                );
            }),
            "warm-reader" => runtime.block_on(async {
                let url = std::env::var("RETREAD_TEST_GIT_URL").expect("missing git URL");
                let rev = std::env::var("RETREAD_TEST_GIT_REV").expect("missing git rev");
                let cache = PathBuf::from(
                    std::env::var_os("RETREAD_TEST_GIT_CACHE").expect("missing git cache"),
                );
                let _checkout = tokio::time::timeout(
                    Duration::from_secs(10),
                    ensure_git_checkout(&url, &rev, &cache),
                )
                .await
                .expect("warm reader timed out")
                .expect("warm reader failed");
                signal_checkout_test_path("RETREAD_TEST_READER_ENTERED")
                    .expect("signal warm reader entry");
            }),
            other => panic!("unknown checkout test child mode `{other}`"),
        }
    }

    /// The first access repairs an unmarked legacy tree and publishes it. Any
    /// later warm access is read-only: a sentinel that `git clean -fdx` would
    /// remove and an index lock that would force wipe/reclone both survive.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn git_checkout_repairs_once_then_warm_tree_is_immutable() {
        let fixture = git_checkout_fixture("clone-once");
        let clone_dir = seed_unmarked_checkout(&fixture, &fixture.rev1);
        std::fs::write(clone_dir.join("extra.txt"), "stray-untracked\n")
            .expect("write checkout conflict");

        let first = tokio::time::timeout(
            Duration::from_secs(10),
            ensure_git_checkout(&fixture.url, &fixture.rev2, &fixture.cache),
        )
        .await
        .expect("initial checkout repair timed out")
        .expect("initial checkout repair failed");
        assert_eq!(
            std::fs::read_to_string(clone_dir.join("extra.txt")).unwrap(),
            "tracked-content\n"
        );
        assert!(matches!(
            checkout_marker_state(&clone_dir, &CloneIdentity::new(&fixture.url, &fixture.rev2))
                .unwrap(),
            CheckoutMarkerState::Matching
        ));
        assert_eq!(
            first.lease.lock.os_acquisitions.load(Ordering::Relaxed),
            1,
            "initialization must take the OS flock exactly once"
        );
        assert_eq!(
            checkout_repair_count(&clone_dir),
            1,
            "the unpublished checkout must run destructive repair exactly once"
        );

        let sentinel = clone_dir.join("warm-sentinel.txt");
        let index_lock = clone_dir.join(".git").join("index.lock");
        std::fs::write(&sentinel, "preserve me\n").unwrap();
        std::fs::write(&index_lock, "force any checkout to fail\n").unwrap();
        let marker_before = std::fs::read(checkout_ready_marker(&clone_dir)).unwrap();
        let head_log = clone_dir.join(".git").join("logs").join("HEAD");
        let head_log_before = std::fs::read(&head_log).unwrap();

        let url = fixture.url.clone();
        let rev = fixture.rev2.clone();
        let cache = fixture.cache.clone();
        let second_task =
            tokio::spawn(async move { ensure_git_checkout(&url, &rev, &cache).await });
        let second = tokio::time::timeout(Duration::from_secs(2), second_task)
            .await
            .expect("same-process warm reader deadlocked")
            .expect("warm reader task panicked")
            .expect("warm reader failed");
        assert!(Arc::ptr_eq(&first.lease.lock, &second.lease.lock));
        assert_eq!(
            second.lease.lock.os_acquisitions.load(Ordering::Relaxed),
            1,
            "warm reader must reuse the process's one OS flock"
        );
        let fresh_reader_entered = fixture.base.join("fresh-reader-entered");
        CheckoutTestChild::spawn(
            "fresh-process warm checkout reader",
            "warm-reader",
            &[
                ("RETREAD_TEST_GIT_URL", fixture.url.clone()),
                ("RETREAD_TEST_GIT_REV", fixture.rev2.clone()),
                (
                    "RETREAD_TEST_GIT_CACHE",
                    fixture.cache.display().to_string(),
                ),
                (
                    "RETREAD_TEST_READER_ENTERED",
                    fresh_reader_entered.display().to_string(),
                ),
            ],
        )
        .finish(Duration::from_secs(15));
        assert!(
            fresh_reader_entered.exists(),
            "fresh process never acquired the warm shared-reader path"
        );
        assert_eq!(std::fs::read_to_string(&sentinel).unwrap(), "preserve me\n");
        assert!(index_lock.exists(), "warm access ran checkout or recloned");
        assert_eq!(
            std::fs::read(checkout_ready_marker(&clone_dir)).unwrap(),
            marker_before
        );
        assert_eq!(std::fs::read(&head_log).unwrap(), head_log_before);
        assert_eq!(
            checkout_repair_count(&clone_dir),
            1,
            "warm access must not run destructive repair again"
        );
    }

    /// The subprocess boundary makes this regression killable even if a future
    /// implementation blocks a runtime thread in `flock`. Test hooks pause the
    /// winner after EX and prove the second worker registered before release.
    #[test]
    fn same_process_two_thread_checkout_reads_do_not_deadlock() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let signals = std::env::temp_dir().join(format!(
            "retread-two-reader-signals-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&signals).unwrap();
        let exclusive_held = signals.join("exclusive-held");
        let exclusive_release = signals.join("exclusive-release");
        let second_registration = signals.join("second-registration");
        CheckoutTestChild::spawn(
            "same-process checkout reader probe",
            "same-process-readers",
            &[
                (
                    "RETREAD_TEST_EXCLUSIVE_HELD",
                    exclusive_held.display().to_string(),
                ),
                (
                    "RETREAD_TEST_EXCLUSIVE_RELEASE",
                    exclusive_release.display().to_string(),
                ),
                (
                    "RETREAD_TEST_SECOND_REGISTRATION",
                    second_registration.display().to_string(),
                ),
            ],
        )
        .finish(Duration::from_secs(15));
        assert!(exclusive_held.exists());
        assert!(second_registration.exists());
        assert!(exclusive_release.exists());
        let _ = std::fs::remove_dir_all(&signals);
    }

    /// A production reader must observe the same cross-process writer lock as
    /// the one-time initializer. The child signals only after an SH attempt is
    /// denied, so the exclusion assertion does not depend on scheduler timing.
    #[cfg(unix)]
    #[test]
    fn published_checkout_reader_waits_for_cross_process_writer() {
        let fixture = git_checkout_fixture("cross-process-rw");
        let clone_dir = seed_unmarked_checkout(&fixture, &fixture.rev2);
        publish_checkout_ready(&clone_dir, &CloneIdentity::new(&fixture.url, &fixture.rev2))
            .expect("publish cross-process reader fixture");

        let lock_path = clone_dir.with_extension("lock");
        let writer = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .expect("open cross-process writer lock");
        fs4::fs_std::FileExt::lock_exclusive(&writer).expect("acquire cross-process writer lock");

        let shared_blocked = fixture.base.join("shared-blocked");
        let reader_entered = fixture.base.join("reader-entered");
        let mut child = CheckoutTestChild::spawn(
            "cross-process shared-reader probe",
            "warm-reader",
            &[
                ("RETREAD_TEST_GIT_URL", fixture.url.clone()),
                ("RETREAD_TEST_GIT_REV", fixture.rev2.clone()),
                (
                    "RETREAD_TEST_GIT_CACHE",
                    fixture.cache.display().to_string(),
                ),
                (
                    "RETREAD_TEST_SHARED_BLOCKED",
                    shared_blocked.display().to_string(),
                ),
                (
                    "RETREAD_TEST_READER_ENTERED",
                    reader_entered.display().to_string(),
                ),
            ],
        );
        child.wait_for_signal(&shared_blocked, Duration::from_secs(5));
        assert!(
            !reader_entered.exists(),
            "shared reader entered while the cross-process writer held EX"
        );

        fs4::fs_std::FileExt::unlock(&writer).expect("release cross-process writer lock");
        child.finish(Duration::from_secs(10));
        assert!(
            reader_entered.exists(),
            "shared reader did not enter after the writer released EX"
        );
    }

    /// Preserve v3.0.3's deep-corruption recovery only at the migration
    /// boundary: an unmarked legacy clone may be wiped once, before readers.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unmarked_git_dir_corruption_is_recloned_before_publication() {
        let fixture = git_checkout_fixture("legacy-reclone");
        let clone_dir = seed_unmarked_checkout(&fixture, &fixture.rev1);
        std::fs::write(clone_dir.join(".git").join("index.lock"), "stale\n").unwrap();

        let checkout = tokio::time::timeout(
            Duration::from_secs(10),
            ensure_git_checkout(&fixture.url, &fixture.rev2, &fixture.cache),
        )
        .await
        .expect("legacy re-clone timed out")
        .expect("legacy re-clone failed");
        assert_eq!(checkout.root(), clone_dir);
        assert!(!clone_dir.join(".git").join("index.lock").exists());
        assert!(matches!(
            checkout_marker_state(&clone_dir, &CloneIdentity::new(&fixture.url, &fixture.rev2))
                .unwrap(),
            CheckoutMarkerState::Matching
        ));
    }

    /// A failed initializer must release EX and leave the process registry
    /// retryable on the same fd. Otherwise one bad clone attempt permanently
    /// wedges every later worker in the same backend process.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_checkout_initialization_releases_lock_for_retry() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "retread-retry-init-{}-{unique}",
            std::process::id()
        ));
        let repo = base.join("repo-created-after-failure");
        let cache = base.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        let url = format!("file://{}", repo.display());

        let first = tokio::time::timeout(
            Duration::from_secs(10),
            ensure_git_checkout(&url, "main", &cache),
        )
        .await
        .expect("failing checkout attempt deadlocked");
        assert!(first.is_err(), "missing source unexpectedly initialized");
        let clone_dir = git_checkout_root(&url, "main", &cache);
        assert!(
            !checkout_ready_marker(&clone_dir).exists(),
            "failed initialization published readiness"
        );

        std::fs::create_dir_all(&repo).unwrap();
        run_fixture_git(&["init", "-b", "main"], &repo);
        std::fs::write(repo.join("after-retry.txt"), "ready\n").unwrap();
        run_fixture_git(&["add", "."], &repo);
        run_fixture_git(&["commit", "-m", "available"], &repo);

        let checkout = tokio::time::timeout(
            Duration::from_secs(10),
            ensure_git_checkout(&url, "main", &cache),
        )
        .await
        .expect("retry after failed initialization deadlocked")
        .expect("retry after failed initialization failed");
        assert_eq!(checkout.root(), clone_dir);
        assert_eq!(
            std::fs::read_to_string(clone_dir.join("after-retry.txt")).unwrap(),
            "ready\n"
        );
        assert_eq!(
            checkout.lease.lock.os_acquisitions.load(Ordering::Relaxed),
            2,
            "retry must reuse the registry fd after releasing the failed EX"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn git_wheel_cache_dir_is_keyed_by_all_inputs() {
        let a = git_wheel_cache_dir(
            "https://github.com/x/y.git",
            "a".repeat(40).as_str(),
            "sub",
            "3.11",
        );
        // Same inputs => same dir (deterministic).
        let a2 = git_wheel_cache_dir(
            "https://github.com/x/y.git",
            "a".repeat(40).as_str(),
            "sub",
            "3.11",
        );
        assert_eq!(a, a2);
        // Any input change => different dir.
        let b = git_wheel_cache_dir(
            "https://github.com/x/y.git",
            "b".repeat(40).as_str(),
            "sub",
            "3.11",
        );
        let c = git_wheel_cache_dir(
            "https://github.com/x/y.git",
            "a".repeat(40).as_str(),
            "other",
            "3.11",
        );
        let d = git_wheel_cache_dir(
            "https://github.com/x/y.git",
            "a".repeat(40).as_str(),
            "sub",
            "3.12",
        );
        let e = git_wheel_cache_dir(
            "https://github.com/x/z.git",
            "a".repeat(40).as_str(),
            "sub",
            "3.11",
        );
        for other in [&b, &c, &d, &e] {
            assert_ne!(&a, other);
        }
        assert!(
            a.components()
                .any(|component| component.as_os_str() == "v3")
        );
        assert_eq!(
            a.file_name().and_then(|name| name.to_str()).unwrap().len(),
            64,
            "git source identity must use the complete SHA-256 namespace",
        );
    }

    fn write_test_wheel_with_payload(
        path: &Path,
        metadata_name: &str,
        metadata_version: &str,
        payload: &[u8],
    ) {
        let file = File::create(path).unwrap();
        let mut archive = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        archive
            .start_file(
                format!(
                    "{}-{}.dist-info/METADATA",
                    metadata_name.replace('-', "_"),
                    metadata_version
                ),
                options,
            )
            .unwrap();
        archive
            .write_all(
                format!(
                    "Metadata-Version: 2.4\nName: {metadata_name}\nVersion: {metadata_version}\n\n"
                )
                .as_bytes(),
            )
            .unwrap();
        if !payload.is_empty() {
            archive.start_file("payload.bin", options).unwrap();
            archive.write_all(payload).unwrap();
        }
        archive.finish().unwrap();
    }

    fn write_test_wheel(path: &Path, metadata_name: &str, metadata_version: &str) {
        write_test_wheel_with_payload(path, metadata_name, metadata_version, &[]);
    }

    #[tokio::test]
    async fn v3_cache_marker_round_trip_preserves_user_outdir_sentinel() {
        let base =
            std::env::temp_dir().join(format!("retread-gitwheel-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let cache_dir = base.join("managed-cache");
        let out_dir = base.join("out");
        std::fs::create_dir_all(&cache_dir).unwrap();
        std::fs::create_dir_all(&out_dir).unwrap();
        std::fs::write(out_dir.join("user-sentinel"), b"keep").unwrap();
        let wheel = cache_dir.join("pkg-1.0.0-py3-none-any.whl");
        write_test_wheel(&wheel, "pkg", "1.0.0");
        let target = ResolutionTarget::from_parts("3.11", "linux-64", None);
        let source_identity = "a".repeat(64);
        let mut marker =
            validate_wheel_file(&wheel, &target, Some(&ExpectedWheel::exact("pkg", "1.0.0")))
                .unwrap();
        marker.source_identity = source_identity.clone();
        std::fs::write(
            cache_dir.join("artifact.json"),
            serde_json::to_vec_pretty(&marker).unwrap(),
        )
        .unwrap();
        let validated = validate_cache_entry(
            &cache_dir,
            &source_identity,
            &target,
            Some(&ExpectedWheel::exact("pkg", "1.0.0")),
        )
        .unwrap()
        .unwrap();
        let hit = materialize_validated_wheel(&validated, &out_dir)
            .await
            .unwrap();
        assert!(hit.is_file());
        assert_eq!(
            std::fs::read(out_dir.join("user-sentinel")).unwrap(),
            b"keep"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn build_output_requires_exactly_one_wheel() {
        let base = std::env::temp_dir().join(format!(
            "retread-multiwheel-{}-{}",
            std::process::id(),
            BUILD_TMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&base).unwrap();
        write_test_wheel(&base.join("one-1.0-py3-none-any.whl"), "one", "1.0");
        write_test_wheel(&base.join("two-1.0-py3-none-any.whl"), "two", "1.0");
        let error = find_built_wheel(&base).await.unwrap_err();
        assert!(format!("{error:#}").contains("expected exactly one"));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn wheel_validation_rejects_filename_metadata_identity_mismatch() {
        let base = std::env::temp_dir().join(format!(
            "retread-wrong-metadata-{}-{}",
            std::process::id(),
            BUILD_TMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&base).unwrap();
        let wheel = base.join("good-1.0-py3-none-any.whl");
        write_test_wheel(&wheel, "evil", "9.0");
        let target = ResolutionTarget::from_parts("3.11", "linux-64", None);
        let error =
            validate_wheel_file(&wheel, &target, Some(&ExpectedWheel::named("good"))).unwrap_err();
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("identity mismatch") || rendered.contains("does not match"),
            "unexpected strict identity error: {rendered}",
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn pinned_wheel_attestation_rechecks_changed_inode_bytes() {
        let base = unique_test_dir("pinned-attestation");
        std::fs::create_dir_all(&base).unwrap();
        let wheel = base.join("pkg-1.0.0-py3-none-any.whl");
        let unique_payload = format!(
            "first-{}-{}",
            std::process::id(),
            BUILD_TMP_SEQUENCE.fetch_add(1, Ordering::Relaxed),
        );
        write_test_wheel_with_payload(&wheel, "pkg", "1.0.0", unique_payload.as_bytes());
        let target =
            ResolutionTarget::from_parts("3.11", crate::glibc::current_pixi_platform(), None);
        let expected = ExpectedWheel::exact("pkg", "1.0.0");
        let authoritative = crate::wheel::read_metadata(&wheel).unwrap().sha256;
        assert_eq!(
            validate_pinned_wheel_for_target_async(
                &wheel,
                &target,
                &expected,
                &authoritative,
                "https://example.invalid/pkg-1.0.0.whl",
            )
            .await
            .unwrap(),
            authoritative,
        );
        // The unchanged inode/stat tuple takes the persisted fast path.
        assert_eq!(
            validate_pinned_wheel_for_target_async(
                &wheel,
                &target,
                &expected,
                &authoritative,
                "https://example.invalid/pkg-1.0.0.whl",
            )
            .await
            .unwrap(),
            authoritative,
        );

        write_test_wheel_with_payload(&wheel, "pkg", "1.0.0", b"coherent-wrong-bytes");
        let error = validate_pinned_wheel_for_target_async(
            &wheel,
            &target,
            &expected,
            &authoritative,
            "https://example.invalid/pkg-1.0.0.whl",
        )
        .await
        .unwrap_err();
        assert!(is_authoritative_wheel_hash_mismatch(&error));
        assert!(
            wheel.is_file(),
            "semantic hash mismatch must preserve ingress bytes"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn v3_artifact_namespaces_separate_architectures_and_use_full_hashes() {
        let x86 = ResolutionTarget::from_parts("3.11", "linux-64", Some((2, 35)));
        let arm = ResolutionTarget::from_parts("3.11", "linux-aarch64", Some((2, 35)));
        let source = "f".repeat(64);
        let x86_path = built_wheel_cache_dir("git", &source, &x86);
        let arm_path = built_wheel_cache_dir("git", &source, &arm);
        assert_ne!(x86_path, arm_path);
        assert!(x86_path.components().any(|part| part.as_os_str() == "v3"));
        assert_eq!(x86_path.file_name().unwrap().to_string_lossy().len(), 64);
        assert_eq!(
            x86_path
                .parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_string_lossy()
                .len(),
            64,
            "artifact target identity must be full SHA-256",
        );
    }

    #[test]
    fn partial_cache_is_invalid_without_touching_caller_output() {
        let base = std::env::temp_dir().join(format!(
            "retread-partial-cache-{}-{}",
            std::process::id(),
            BUILD_TMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let cache = base.join("cache");
        let output = base.join("output");
        std::fs::create_dir_all(&cache).unwrap();
        std::fs::create_dir_all(&output).unwrap();
        std::fs::write(output.join("sentinel"), b"owned by caller").unwrap();
        let target = ResolutionTarget::from_parts("3.11", "linux-64", None);
        assert!(validate_cache_entry(&cache, &"a".repeat(64), &target, None).is_err());
        assert_eq!(
            std::fs::read(output.join("sentinel")).unwrap(),
            b"owned by caller"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn sdist_pep691_and_fragment_hashes_must_agree() {
        let fragment = "a".repeat(64);
        let advertised = "b".repeat(64);
        let url = url::Url::parse(&format!(
            "https://example.invalid/pkg-1.0.tar.gz#sha256={fragment}"
        ))
        .unwrap();
        let error = sdist_advertised_sha256(&url, Some(&advertised)).unwrap_err();
        assert!(format!("{error:#}").contains("hash disagreement"));
    }

    #[tokio::test]
    async fn foreign_exact_miss_refuses_before_build_callback() {
        let foreign_subdir = match crate::glibc::current_pixi_platform() {
            "linux-aarch64" => "linux-64",
            _ => "linux-aarch64",
        };
        let target = ResolutionTarget::from_parts("3.11", foreign_subdir, Some((2, 35)));
        let source_identity = hash_fields(
            b"foreign-miss-test\0",
            &[format!(
                "{}-{}",
                std::process::id(),
                BUILD_TMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            )
            .as_bytes()],
        );
        let cache = built_wheel_cache_dir("git", &source_identity, &target);
        let _ = remove_owned_cache_entry(&cache);
        let callback_ran = Arc::new(AtomicBool::new(false));
        let callback_flag = Arc::clone(&callback_ran);
        let output = std::env::temp_dir().join(format!(
            "retread-foreign-miss-output-{}",
            std::process::id()
        ));
        let error = cached_build(
            "git",
            &source_identity,
            &target,
            &output,
            None,
            move |_private_out| async move {
                callback_flag.store(true, Ordering::Relaxed);
                Ok(())
            },
        )
        .await
        .unwrap_err();
        assert!(format!("{error:#}").contains("foreign target"));
        assert!(!callback_ran.load(Ordering::Relaxed));
        let _ = remove_owned_cache_entry(&cache);
        let _ = std::fs::remove_dir_all(&output);
    }

    #[tokio::test]
    async fn valid_cache_semantic_mismatch_is_preserved_without_rebuild() {
        let target =
            ResolutionTarget::from_parts("3.11", crate::glibc::current_pixi_platform(), None);
        let source_identity = hash_fields(
            b"semantic-mismatch-test\0",
            &[unique_test_dir("semantic-key").to_string_lossy().as_bytes()],
        );
        let cache = built_wheel_cache_dir("path", &source_identity, &target);
        remove_owned_cache_entry(&cache).unwrap();
        std::fs::create_dir_all(&cache).unwrap();
        let wheel = cache.join("pkg-1.0.0-py3-none-any.whl");
        write_test_wheel(&wheel, "pkg", "1.0.0");
        let mut marker = validate_wheel_file(&wheel, &target, None).unwrap();
        marker.source_identity = source_identity.clone();
        std::fs::write(
            cache.join("artifact.json"),
            serde_json::to_vec_pretty(&marker).unwrap(),
        )
        .unwrap();

        let callback_ran = Arc::new(AtomicBool::new(false));
        let callback_flag = Arc::clone(&callback_ran);
        let output = unique_test_dir("semantic-output");
        let error = cached_build(
            "path",
            &source_identity,
            &target,
            &output,
            Some(&ExpectedWheel::exact("pkg", "2.0.0")),
            move |_private_out| async move {
                callback_flag.store(true, Ordering::Relaxed);
                Ok(())
            },
        )
        .await
        .unwrap_err();
        assert!(is_expected_wheel_mismatch(&error));
        assert!(!callback_ran.load(Ordering::Relaxed));
        assert!(cache.join("artifact.json").is_file());
        assert!(wheel.is_file());

        remove_owned_cache_entry(&cache).unwrap();
        let _ = std::fs::remove_dir_all(&output);
    }

    #[tokio::test]
    async fn foreign_target_can_use_exact_validated_cache_hit() {
        let foreign_subdir = match crate::glibc::current_pixi_platform() {
            "linux-aarch64" => "linux-64",
            _ => "linux-aarch64",
        };
        let target = ResolutionTarget::from_parts("3.11", foreign_subdir, Some((2, 35)));
        let source_identity = hash_fields(
            b"foreign-hit-test\0",
            &[unique_test_dir("foreign-hit-key")
                .to_string_lossy()
                .as_bytes()],
        );
        let cache = built_wheel_cache_dir("git", &source_identity, &target);
        remove_owned_cache_entry(&cache).unwrap();
        std::fs::create_dir_all(&cache).unwrap();
        let wheel = cache.join("pkg-1.0.0-py3-none-any.whl");
        write_test_wheel(&wheel, "pkg", "1.0.0");
        let mut marker = validate_wheel_file(&wheel, &target, None).unwrap();
        marker.source_identity = source_identity.clone();
        std::fs::write(
            cache.join("artifact.json"),
            serde_json::to_vec_pretty(&marker).unwrap(),
        )
        .unwrap();

        let callback_ran = Arc::new(AtomicBool::new(false));
        let callback_flag = Arc::clone(&callback_ran);
        let output = unique_test_dir("foreign-hit-output");
        let hit = cached_build(
            "git",
            &source_identity,
            &target,
            &output,
            Some(&ExpectedWheel::exact("pkg", "1.0.0")),
            move |_private_out| async move {
                callback_flag.store(true, Ordering::Relaxed);
                Ok(())
            },
        )
        .await
        .unwrap();
        assert!(hit.is_file());
        assert!(!callback_ran.load(Ordering::Relaxed));

        remove_owned_cache_entry(&cache).unwrap();
        let _ = std::fs::remove_dir_all(&output);
    }

    #[tokio::test]
    async fn output_publication_replaces_a_b_a_without_cache_aliasing() {
        let base = unique_test_dir("output-aba");
        let cache_a = base.join("cache-a");
        let cache_b = base.join("cache-b");
        let output = base.join("output");
        std::fs::create_dir_all(&cache_a).unwrap();
        std::fs::create_dir_all(&cache_b).unwrap();
        let filename = "pkg-1.0.0-py3-none-any.whl";
        let path_a = cache_a.join(filename);
        let path_b = cache_b.join(filename);
        write_test_wheel_with_payload(&path_a, "pkg", "1.0.0", b"artifact-a");
        write_test_wheel_with_payload(&path_b, "pkg", "1.0.0", b"artifact-b");
        let target = ResolutionTarget::from_parts("3.11", "linux-64", None);
        let mut marker_a = validate_wheel_file(&path_a, &target, None).unwrap();
        marker_a.source_identity = "a".repeat(64);
        let mut marker_b = validate_wheel_file(&path_b, &target, None).unwrap();
        marker_b.source_identity = "b".repeat(64);
        assert_ne!(marker_a.sha256, marker_b.sha256);
        let validated_a = ValidatedWheel {
            path: path_a.clone(),
            marker: marker_a.clone(),
        };
        let validated_b = ValidatedWheel {
            path: path_b.clone(),
            marker: marker_b.clone(),
        };

        let published = materialize_validated_wheel(&validated_a, &output)
            .await
            .unwrap();
        assert_eq!(
            crate::wheel::read_metadata(&published).unwrap().sha256,
            marker_a.sha256
        );
        materialize_validated_wheel(&validated_b, &output)
            .await
            .unwrap();
        assert_eq!(
            crate::wheel::read_metadata(&published).unwrap().sha256,
            marker_b.sha256
        );
        materialize_validated_wheel(&validated_a, &output)
            .await
            .unwrap();
        assert_eq!(
            crate::wheel::read_metadata(&published).unwrap().sha256,
            marker_a.sha256
        );
        assert!(!same_filesystem_inode(&published, &path_a));
        assert!(!same_filesystem_inode(&published, &path_b));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn source_snapshot_captures_context_and_excludes_only_managed_roots() {
        let base = unique_test_dir("snapshot-context");
        let context = base.join("workspace");
        let project = context.join("packages/pkg");
        let managed = context.join("generated-wheels");
        std::fs::create_dir_all(project.join("src")).unwrap();
        std::fs::create_dir_all(context.join(".git")).unwrap();
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(project.join("pyproject.toml"), b"[build-system]\n").unwrap();
        std::fs::write(project.join("src/lib.py"), b"VALUE = 1\n").unwrap();
        std::fs::write(context.join(".git/HEAD"), b"ref: refs/heads/main\n").unwrap();
        std::fs::write(
            context.join("workspace.toml"),
            b"members = ['packages/pkg']\n",
        )
        .unwrap();
        for managed_file in [
            "retread-pack.lock.json",
            "retread-audit-pack.json",
            "retread-probe-trace-pack.json",
            "retread-progress-pack.log",
            ".retread-pack.lock.json.tmp",
        ] {
            std::fs::write(context.join(managed_file), b"managed generation one").unwrap();
        }
        std::fs::write(context.join("retread-real.py"), b"REAL = True\n").unwrap();
        std::fs::write(managed.join("old.whl"), b"old output").unwrap();

        let first =
            prepare_source_snapshot(&context, &managed, std::slice::from_ref(&managed)).unwrap();
        assert!(first.root().join("packages/pkg/pyproject.toml").is_file());
        assert!(first.root().join(".git/HEAD").is_file());
        assert!(first.root().join("workspace.toml").is_file());
        assert!(first.root().join("retread-real.py").is_file());
        assert!(!first.root().join("retread-pack.lock.json").exists());
        assert!(!first.root().join("retread-progress-pack.log").exists());
        assert!(!first.root().join("generated-wheels").exists());
        std::fs::write(managed.join("new.whl"), b"new output").unwrap();
        std::fs::write(
            context.join("retread-progress-pack.log"),
            b"managed generation two",
        )
        .unwrap();
        let second =
            prepare_source_snapshot(&context, &managed, std::slice::from_ref(&managed)).unwrap();
        assert_eq!(first.identity, second.identity);
        std::fs::write(
            context.join("workspace.toml"),
            b"members = ['packages/pkg', 'other']\n",
        )
        .unwrap();
        let third =
            prepare_source_snapshot(&context, &managed, std::slice::from_ref(&managed)).unwrap();
        assert_ne!(first.identity, third.identity);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn source_snapshot_workspace_path_is_stable_and_lease_scoped() {
        let base = unique_test_dir("stable-snapshot");
        let source = base.join("source");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("input.txt"), b"stable bytes").unwrap();
        let snapshot = prepare_source_snapshot(&source, &base.join("out"), &[]).unwrap();
        let source_identity = hash_fields(
            b"stable-workspace-test\0",
            &[unique_test_dir("stable-workspace-key")
                .to_string_lossy()
                .as_bytes()],
        );
        let stable = stabilize_source_snapshot_workspace(snapshot, "test", &source_identity)
            .await
            .unwrap();
        let expected = crate::courier::retread_cache_root()
            .join("source-workspaces/v1/test")
            .join(&source_identity);
        assert_eq!(stable.root(), expected);
        assert_eq!(
            std::fs::read(stable.root().join("input.txt")).unwrap(),
            b"stable bytes"
        );
        #[cfg(unix)]
        assert_eq!(
            std::fs::metadata(stable.root().join("input.txt"))
                .unwrap()
                .modified()
                .unwrap(),
            std::time::UNIX_EPOCH + Duration::from_secs(946_684_800),
        );
        drop(stable);
        assert!(
            !expected.exists(),
            "workspace must be removed before its lease unlocks"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn disposable_build_mutation_cannot_change_retained_injection_source() {
        let base = unique_test_dir("disposable-build-source");
        let source = base.join("source");
        std::fs::create_dir_all(source.join("pkg")).unwrap();
        std::fs::write(source.join("pkg/module.py"), b"committed = True\n").unwrap();
        let snapshot = prepare_source_snapshot(&source, &base.join("out"), &[]).unwrap();
        let identity = hash_fields(
            b"disposable-build-source-test\0",
            &[base.to_string_lossy().as_bytes()],
        );
        let pristine = stabilize_source_snapshot_workspace(snapshot, "test", &identity)
            .await
            .unwrap();
        let retained_workspace = Arc::clone(&pristine.workspace);
        let disposable = tokio::task::spawn_blocking({
            let out = base.join("private-output");
            move || prepare_source_snapshot(&retained_workspace.directory.0, &out, &[])
        })
        .await
        .unwrap()
        .unwrap();
        std::fs::write(disposable.root().join("pkg/module.py"), b"mutated = True\n").unwrap();
        std::fs::write(
            disposable.root().join("pkg/generated.egg-info"),
            b"generated\n",
        )
        .unwrap();

        assert_eq!(
            std::fs::read(pristine.root().join("pkg/module.py")).unwrap(),
            b"committed = True\n"
        );
        assert!(!pristine.root().join("pkg/generated.egg-info").exists());
        drop(disposable);
        drop(pristine);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn source_snapshot_rejects_transient_directory_rename_swap() {
        let base = unique_test_dir("snapshot-rename-swap");
        let source = base.join("source");
        let package = source.join("package");
        let parked = base.join("parked-original");
        let replacement = base.join("replacement");
        std::fs::create_dir_all(&package).unwrap();
        std::fs::create_dir_all(&replacement).unwrap();
        std::fs::write(package.join("module.py"), b"ORIGINAL = True\n").unwrap();
        std::fs::write(replacement.join("module.py"), b"REPLACEMENT = True\n").unwrap();

        let mut phase = 0_u8;
        let result = prepare_source_snapshot_with_hook(
            &source,
            &base.join("out"),
            &[],
            &mut |path, visit_phase| {
                if path != package {
                    return Ok(());
                }
                match (phase, visit_phase) {
                    (0, SnapshotVisitPhase::BeforeEnumeration) => {
                        std::fs::rename(&package, &parked)?;
                        std::fs::rename(&replacement, &package)?;
                        phase = 1;
                    }
                    (1, SnapshotVisitPhase::BeforeFinalValidation) => {
                        std::fs::rename(&package, &replacement)?;
                        std::fs::rename(&parked, &package)?;
                        phase = 2;
                    }
                    _ => {}
                }
                Ok(())
            },
        );

        assert_eq!(phase, 2, "test must execute and restore the A→B→A swap");
        let error = result
            .err()
            .expect("a transient source-directory rename swap must be rejected");
        assert!(
            format!("{error:#}").contains("changed while its build snapshot was prepared"),
            "unexpected rename-swap error: {error:#}",
        );
        assert_eq!(
            std::fs::read(package.join("module.py")).unwrap(),
            b"ORIGINAL = True\n"
        );
        assert_eq!(
            std::fs::read(replacement.join("module.py")).unwrap(),
            b"REPLACEMENT = True\n"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pinned_symlink_read_ignores_transient_direct_escape_swap() {
        use std::os::unix::fs::symlink;

        let base = unique_test_dir("snapshot-direct-symlink-swap");
        let source = base.join("source");
        let link = source.join("link");
        let parked = base.join("parked-link");
        let replacement = base.join("replacement-link");
        std::fs::create_dir_all(source.join("safe")).unwrap();
        std::fs::create_dir_all(base.join("outside")).unwrap();
        symlink("safe", &link).unwrap();
        symlink("../outside", &replacement).unwrap();
        let root_fd = rustix::fs::openat(
            rustix::fs::CWD,
            &source,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::CLOEXEC
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::DIRECTORY,
            rustix::fs::Mode::empty(),
        )
        .map(File::from)
        .unwrap();
        let name = CString::new("link").unwrap();
        let expected = snapshot_stat_at(&root_fd, &name).unwrap();
        let mut phase = 0_u8;
        let mut observed_target = None;
        let result = read_snapshot_symlink_at_with_hook(
            &root_fd,
            &name,
            expected,
            &link,
            &mut |path, symlink_phase| {
                assert_eq!(path, link);
                match (phase, symlink_phase) {
                    (0, PinnedSymlinkPhase::AfterOpen) => {
                        std::fs::rename(&link, &parked)?;
                        std::fs::rename(&replacement, &link)?;
                        phase = 1;
                    }
                    (1, PinnedSymlinkPhase::AfterRead(target)) => {
                        observed_target = Some(target.to_bytes().to_vec());
                        std::fs::rename(&link, &replacement)?;
                        std::fs::rename(&parked, &link)?;
                        phase = 2;
                    }
                    _ => {}
                }
                Ok(())
            },
        );

        assert_eq!(phase, 2, "test must execute and restore the A→B→A swap");
        assert_eq!(observed_target.as_deref(), Some(b"safe".as_slice()));
        match result {
            Ok(target) => assert_eq!(target.as_bytes(), b"safe"),
            Err(error) => assert!(
                format!("{error:#}").contains("changed while its target was read"),
                "unexpected direct symlink-swap error: {error:#}",
            ),
        }
        assert_eq!(std::fs::read_link(&link).unwrap(), Path::new("safe"));
        assert_eq!(
            std::fs::read_link(&replacement).unwrap(),
            Path::new("../outside")
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn descriptor_resolver_ignores_transient_intermediate_escape_swap() {
        use std::os::unix::fs::symlink;

        let base = unique_test_dir("snapshot-chain-symlink-swap");
        let source = base.join("source");
        let middle = source.join("middle");
        let parked = base.join("parked-middle");
        let replacement = base.join("replacement-middle");
        std::fs::create_dir_all(source.join("safe")).unwrap();
        std::fs::create_dir_all(base.join("outside")).unwrap();
        symlink("middle", source.join("chain")).unwrap();
        symlink("safe", &middle).unwrap();
        symlink("../outside", &replacement).unwrap();
        let root_fd = rustix::fs::openat(
            rustix::fs::CWD,
            &source,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::CLOEXEC
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::DIRECTORY,
            rustix::fs::Mode::empty(),
        )
        .map(File::from)
        .unwrap();
        let mut phase = 0_u8;
        let mut observed_target = None;
        let result = resolve_snapshot_components_with_hook(
            &root_fd,
            &source,
            &[],
            vec![OsString::from("chain")],
            &mut |path, symlink_phase| {
                if path != middle {
                    return Ok(());
                }
                match (phase, symlink_phase) {
                    (0, PinnedSymlinkPhase::AfterOpen) => {
                        std::fs::rename(&middle, &parked)?;
                        std::fs::rename(&replacement, &middle)?;
                        phase = 1;
                    }
                    (1, PinnedSymlinkPhase::AfterRead(target)) => {
                        observed_target = Some(target.to_bytes().to_vec());
                        std::fs::rename(&middle, &replacement)?;
                        std::fs::rename(&parked, &middle)?;
                        phase = 2;
                    }
                    _ => {}
                }
                Ok(())
            },
        );

        assert_eq!(phase, 2, "test must execute and restore the A→B→A swap");
        assert_eq!(observed_target.as_deref(), Some(b"safe".as_slice()));
        match result {
            Ok(resolution) => assert_eq!(
                resolution,
                DescriptorResolution::Existing(rustix::fs::FileType::Directory)
            ),
            Err(error) => assert!(
                format!("{error:#}").contains("changed while its target was read"),
                "unexpected intermediate symlink-swap error: {error:#}",
            ),
        }
        assert_eq!(std::fs::read_link(&middle).unwrap(), Path::new("safe"));
        assert_eq!(
            std::fs::read_link(&replacement).unwrap(),
            Path::new("../outside")
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn source_snapshot_identity_binds_modes_and_symlink_targets() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let base = unique_test_dir("snapshot-mode-link");
        let source = base.join("source");
        std::fs::create_dir_all(&source).unwrap();
        let script = source.join("tool.sh");
        std::fs::write(&script, b"#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o644)).unwrap();
        symlink("tool.sh", source.join("tool-link")).unwrap();
        let first = prepare_source_snapshot(&source, &base.join("out"), &[]).unwrap();

        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let second = prepare_source_snapshot(&source, &base.join("out"), &[]).unwrap();
        assert_ne!(first.identity, second.identity);
        std::fs::remove_file(source.join("tool-link")).unwrap();
        symlink("missing-tool.sh", source.join("tool-link")).unwrap();
        let third = prepare_source_snapshot(&source, &base.join("out"), &[]).unwrap();
        assert_ne!(second.identity, third.identity);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn source_snapshot_omits_external_gitdir_indirection() {
        let base = unique_test_dir("snapshot-gitdir");
        let source = base.join("source");
        let external_gitdir = base.join("external-gitdir");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::create_dir_all(&external_gitdir).unwrap();
        std::fs::write(
            source.join(".git"),
            format!("gitdir: {}\n", external_gitdir.display()),
        )
        .unwrap();
        let external = prepare_source_snapshot(&source, &base.join("out"), &[]).unwrap();
        assert!(!external.root().join(".git").exists());

        std::fs::write(source.join(".git"), "gitdir: .git-data\n").unwrap();
        std::fs::create_dir_all(source.join(".git-data")).unwrap();
        std::fs::write(source.join(".git-data/HEAD"), b"ref: refs/heads/main\n").unwrap();
        let snapshot = prepare_source_snapshot(&source, &base.join("out"), &[]).unwrap();
        assert!(snapshot.root().join(".git").is_file());
        assert!(snapshot.root().join(".git-data/HEAD").is_file());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn external_gitdir_path_source_gets_self_contained_scm_metadata() {
        let fixture = git_checkout_fixture("path-external-gitdir");
        let origin = fixture.base.join("repo");
        run_fixture_git(&["tag", "path-release", &fixture.rev2], &origin);
        let worktree = fixture.base.join("external-worktree");
        run_fixture_git(
            &[
                "worktree",
                "add",
                "--detach",
                worktree.to_str().unwrap(),
                &fixture.rev2,
            ],
            &origin,
        );
        assert!(worktree.join(".git").is_file());
        std::fs::write(worktree.join("base.txt"), b"dirty path bytes\n").unwrap();

        let state = external_path_git_state(&worktree)
            .await
            .unwrap()
            .expect("worktree uses external Git metadata");
        let snapshot = prepare_source_snapshot(&worktree, &fixture.base.join("out"), &[]).unwrap();
        assert!(!snapshot.root().join(".git").exists());
        attach_external_path_git_metadata(&worktree, &snapshot, &state)
            .await
            .unwrap();

        assert!(snapshot.root().join(".git").is_dir());
        assert_eq!(
            run_fixture_git(&["rev-parse", "HEAD"], snapshot.root()),
            fixture.rev2
        );
        assert_eq!(
            run_fixture_git(&["describe", "--tags", "--exact-match"], snapshot.root()),
            "path-release"
        );
        let status = run_fixture_git(
            &["status", "--porcelain=v1", "--untracked-files=all"],
            snapshot.root(),
        );
        assert!(
            status.contains("base.txt"),
            "dirty state must remain visible: {status}"
        );
        assert_eq!(
            std::fs::read(snapshot.root().join("base.txt")).unwrap(),
            b"dirty path bytes\n"
        );
    }

    #[test]
    fn external_relative_project_falls_back_to_its_own_context() {
        let base = unique_test_dir("external-relative-context");
        let declared_context = base.join("packs/one");
        let external_project = base.join("third_party/project");
        let submodule_gitdir = base.join(".git/modules/third_party/project");
        std::fs::create_dir_all(&declared_context).unwrap();
        std::fs::create_dir_all(&external_project).unwrap();
        std::fs::create_dir_all(&submodule_gitdir).unwrap();
        std::fs::write(external_project.join("pyproject.toml"), b"[build-system]\n").unwrap();
        std::fs::write(
            external_project.join(".git"),
            "gitdir: ../../.git/modules/third_party/project\n",
        )
        .unwrap();
        std::fs::write(submodule_gitdir.join("HEAD"), b"ref: refs/heads/main\n").unwrap();

        let canonical_source = std::fs::canonicalize(&external_project).unwrap();
        let candidate_context = std::fs::canonicalize(&declared_context).unwrap();
        let selected = select_path_source_context(&canonical_source, &candidate_context);
        assert_eq!(selected, canonical_source);
        let snapshot = prepare_source_snapshot(&selected, &base.join("out"), &[]).unwrap();
        assert!(snapshot.root().join("pyproject.toml").is_file());
        assert!(!snapshot.root().join(".git").exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn git_subdirectory_is_lexically_and_physically_confined() {
        use std::os::unix::fs::symlink;

        assert!(normalize_git_subdirectory("pkg/sub").is_ok());
        assert!(normalize_git_subdirectory("./pkg").is_ok());
        assert!(normalize_git_subdirectory("../outside").is_err());
        assert!(normalize_git_subdirectory("/outside").is_err());

        let base = unique_test_dir("git-confinement");
        let checkout = base.join("checkout");
        let outside = base.join("outside");
        std::fs::create_dir_all(checkout.join("pkg")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        symlink(&outside, checkout.join("escape")).unwrap();
        assert!(confined_git_source_dir(&checkout, Path::new("pkg")).is_ok());
        let error = confined_git_source_dir(&checkout, Path::new("escape")).unwrap_err();
        assert!(format!("{error:#}").contains("outside checkout"));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn sdist_identity_binds_url_derived_archive_filename() {
        let sha = "a".repeat(64);
        assert_ne!(
            sdist_source_identity(&sha, "pkg-1.0.tar.gz"),
            sdist_source_identity(&sha, "pkg-1.0.zip"),
        );
    }

    #[tokio::test]
    async fn invalid_python_target_fails_before_output_mutation() {
        let base = unique_test_dir("invalid-python");
        let source = base.join("source");
        let output = base.join("output");
        std::fs::create_dir_all(&source).unwrap();
        let error = build_wheel_from_path(&source, &output, "3")
            .await
            .unwrap_err();
        assert!(format!("{error:#}").contains("MAJOR.MINOR"));
        assert!(!output.exists());
        let _ = std::fs::remove_dir_all(&base);
    }
}
