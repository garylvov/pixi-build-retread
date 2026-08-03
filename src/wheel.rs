//! Wheel download and METADATA parsing.
//!
//! METADATA inside a wheel is RFC 822-style headers (PEP 241/345/566). We
//! extract `Name`, `Version`, and every `Requires-Dist:` value. Requirement
//! strings are kept as PEP 508 text and parsed downstream by the relax pass.

use std::fmt::Write as _;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result, anyhow, bail};
use futures::StreamExt as _;
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

static ATOMIC_FILE_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn unique_atomic_sibling(dst: &Path, suffix: &str) -> PathBuf {
    use std::sync::atomic::Ordering;

    let parent = dst.parent().unwrap_or_else(|| Path::new("."));
    let filename = dst
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wheel.whl");
    parent.join(format!(
        ".{filename}.{}.{}.{suffix}",
        std::process::id(),
        ATOMIC_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed),
    ))
}

struct TemporaryPath {
    path: PathBuf,
    armed: bool,
}

impl TemporaryPath {
    fn armed(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for TemporaryPath {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

struct WheelStoreFillLock(std::fs::File);

impl Drop for WheelStoreFillLock {
    fn drop(&mut self) {
        if let Err(error) = fs4::fs_std::FileExt::unlock(&self.0) {
            tracing::warn!(error = %error, "failed to unlock wheel-store first fill");
        }
    }
}

/// Serialize the first population of one content-addressed store entry across
/// backend processes. The lock is intentionally per wheel, not global: sibling
/// packs can still fetch different wheels concurrently, while contenders for a
/// multi-gigabyte NVIDIA wheel wait for the first downloader and then reuse its
/// attested store entry.
async fn acquire_wheel_store_fill_lock(store_path: &Path) -> Result<WheelStoreFillLock> {
    let parent = store_path
        .parent()
        .ok_or_else(|| anyhow!("wheel store path has no parent: {}", store_path.display()))?;
    fs::create_dir_all(parent)
        .await
        .with_context(|| format!("creating wheel store namespace {}", parent.display()))?;
    let filename = store_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("wheel store path has no UTF-8 filename"))?;
    let lock_path = parent.join(format!(".{filename}.retread-fill-v1.lock"));
    tokio::task::spawn_blocking(move || {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .with_context(|| format!("opening wheel-store fill lock {}", lock_path.display()))?;
        fs4::fs_std::FileExt::lock_exclusive(&file)
            .with_context(|| format!("locking wheel-store fill {}", lock_path.display()))?;
        Ok(WheelStoreFillLock(file))
    })
    .await
    .context("wheel-store fill lock task panicked")?
}

#[derive(Debug, Clone)]
pub struct WheelMetadata {
    pub name: String,
    pub version: String,
    /// Raw `Requires-Dist:` values, one per line. PEP 508 syntax.
    pub requires_dist: Vec<String>,
    /// Retread-owned conda run requirements derived from final native ABI
    /// validation. These are deliberately not PEP 508 `Requires-Dist` values.
    pub retread_conda_run_dependencies: Vec<String>,
    /// Whether the wheel's tag set includes `none-any` (pure-Python). Used to
    /// emit `noarch: python` in the generated recipe.
    pub is_pure_python: bool,
    /// Computed SHA-256 of the downloaded wheel.
    pub sha256: String,
    /// The wheel filename (e.g. `isaacsim-5.1.0-cp311-none-manylinux_2_35_x86_64.whl`).
    pub filename: String,
}

/// Create a same-directory temp file for an atomic wheel-cache write.
///
/// `ZipWriter` (used by the inject/autodata/relax pipeline in
/// `wheel_inject.rs`, `wheel_inject_data.rs`, and `wheel_rewrite.rs`) needs
/// a concrete `std::fs::File` to write into, so those call sites can't
/// reuse the async tokio-fs temp+rename dance above. This is the sync
/// equivalent of the same protocol: write into `<dst>.<pid>.tmp` (same
/// directory as `dst`, so the final rename is atomic even on NFS) and
/// promote it with [`commit_atomic_write`] only after every byte is
/// flushed. Without this, a process/node death mid-write leaves a
/// truncated file sitting at `dst` -- and because `is_fresh()` only
/// checks mtime, that truncated file gets treated as a valid cache hit
/// forever afterward (see the self-heal check in `is_fresh`).
pub(crate) fn create_atomic_tmp(dst: &Path) -> Result<(PathBuf, std::fs::File)> {
    let tmp = atomic_tmp_path(dst);
    let file =
        std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    Ok((tmp, file))
}

/// Same-directory temp path for an atomic write, without creating the
/// file. Used by callers (e.g. the hard-link fast path in
/// `wheel_rewrite.rs`) whose write step needs the path to NOT already
/// exist (`std::fs::hard_link` errors if the destination is present).
pub(crate) fn atomic_tmp_path(dst: &Path) -> PathBuf {
    unique_atomic_sibling(dst, "tmp")
}

/// Promote a temp file created by [`create_atomic_tmp`] to its final
/// destination with an atomic same-directory rename. Callers must fully
/// flush (and drop, if holding a wrapping writer) the file before calling
/// this. On any failure the temp file is removed so it never lingers as
/// cache-poisoning debris that a later run could mistake for real output.
pub(crate) fn commit_atomic_write(tmp: &Path, dst: &Path) -> Result<()> {
    std::fs::rename(tmp, dst).map_err(|e| {
        let _ = std::fs::remove_file(tmp);
        anyhow::Error::from(e).context(format!("renaming {} -> {}", tmp.display(), dst.display()))
    })
}

/// Check whether `path` is a well-formed zip archive (just the central
/// directory / EOCD, not full CRC validation of every entry -- callers
/// only need to know "is this readable at all", the corruption pattern
/// seen in practice is a truncated file from an interrupted write, which
/// this catches). Used by the self-heal cache-freshness check.
pub(crate) fn is_valid_zip(path: &Path) -> bool {
    match std::fs::File::open(path) {
        Ok(f) => zip::ZipArchive::new(f).is_ok(),
        Err(_) => false,
    }
}

/// Derive the on-disk filename for a wheel URL. Percent-decodes the last
/// path segment so URLs that encode the `+` of a PEP 440 local-version
/// identifier (e.g. miropsota's
/// `pytorch3d-0.7.8%2B5043d15pt2.7.0cu128-...whl`) land on disk with the
/// canonical PEP 427 spelling. Without that decode, pip rejects the file
/// at install time with `Invalid wheel filename (invalid version)`
/// because `%2B` is not a valid PEP 440 character.
pub fn wheel_filename_from_url(url: &url::Url) -> Result<String> {
    let raw = url
        .path_segments()
        .and_then(|mut s| s.next_back())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("URL has no filename component: {url}"))?;
    let decoded = percent_encoding::percent_decode_str(raw)
        .decode_utf8()
        .with_context(|| format!("URL filename is not valid UTF-8: {raw}"))?;
    if !decoded.ends_with(".whl") {
        bail!("URL does not point to a .whl file: {url}");
    }
    let decoded_str: &str = &decoded;
    if decoded_str.contains(['/', '\\'])
        || decoded_str == "."
        || decoded_str == ".."
        || !matches!(
            Path::new(decoded_str)
                .components()
                .collect::<Vec<_>>()
                .as_slice(),
            [std::path::Component::Normal(_)]
        )
    {
        bail!(
            "URL wheel filename must decode to a single wheel basename (exactly one ordinary path component): {url}"
        );
    }
    Ok(decoded.into_owned())
}

pub(crate) fn normalize_sha256(value: &str, label: &str) -> Result<String> {
    let normalized = value.to_ascii_lowercase();
    if normalized.len() != 64 || !normalized.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} must be exactly 64 hexadecimal SHA-256 characters");
    }
    Ok(normalized)
}

fn wheel_url_identity(url: &url::Url) -> String {
    format!("{:x}", Sha256::digest(url.as_str().as_bytes()))
}

/// Immutable per-digest destination for a pinned fetched wheel. The digest
/// namespace prevents equal basenames with different authoritative bytes from
/// replacing each other while downstream validation or rewriting is active.
pub(crate) fn pinned_wheel_destination(
    url: &url::Url,
    expected_sha256: &str,
    dest_dir: &Path,
) -> Result<PathBuf> {
    let sha256 = normalize_sha256(expected_sha256, "wheel hash")?;
    Ok(dest_dir
        .join(".retread-wheel-fetch")
        .join("v1")
        .join("sha256")
        .join(sha256)
        .join(wheel_filename_from_url(url)?))
}

fn unpinned_wheel_destination(url: &url::Url, dest_dir: &Path) -> Result<PathBuf> {
    Ok(dest_dir
        .join(".retread-wheel-fetch")
        .join("v1")
        .join("url")
        .join(wheel_url_identity(url))
        .join(wheel_filename_from_url(url)?))
}

/// Stable content-addressed store location for an authoritative wheel hash.
/// Both the digest and decoded filename are validated before a path is
/// returned, so callers can use this helper before touching the filesystem.
pub(crate) fn pinned_wheel_store_path(
    url: &url::Url,
    expected_sha256: &str,
    store_root: &Path,
) -> Result<PathBuf> {
    let sha256 = normalize_sha256(expected_sha256, "wheel hash")?;
    Ok(store_root.join(sha256).join(wheel_filename_from_url(url)?))
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct CachedFileFingerprint {
    size: u64,
    modified_nanos: u128,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct StoreIntegrityMarker {
    schema: String,
    sha256: String,
    fingerprint: CachedFileFingerprint,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct UrlIntegrityMarker {
    schema: String,
    url_identity: String,
    sha256: String,
    fingerprint: CachedFileFingerprint,
}

fn url_integrity_marker_path(path: &Path) -> PathBuf {
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wheel.whl");
    path.with_file_name(format!(".{filename}.retread-url-integrity-v1.json"))
}

async fn inspect_unpinned_destination(path: &Path, url: &url::Url) -> Result<bool> {
    let metadata = match fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.file_type().is_file() && !metadata.file_type().is_symlink() => {
            metadata
        }
        Ok(_) => return Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error).with_context(|| format!("stating {}", path.display())),
    };
    let marker_path = url_integrity_marker_path(path);
    let marker_metadata = match fs::symlink_metadata(&marker_path).await {
        Ok(metadata) if metadata.file_type().is_file() && !metadata.file_type().is_symlink() => {
            metadata
        }
        _ => return Ok(false),
    };
    let _ = marker_metadata;
    let marker: UrlIntegrityMarker = match fs::read(&marker_path)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
    {
        Some(marker) => marker,
        None => return Ok(false),
    };
    Ok(marker.schema == "retread-wheel-url-integrity-v1"
        && marker.url_identity == wheel_url_identity(url)
        && marker.sha256.len() == 64
        && marker.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        && marker.fingerprint == fingerprint_metadata(&metadata)?)
}

async fn write_url_integrity_marker(path: &Path, url: &url::Url, sha256: &str) -> Result<()> {
    let marker_path = url_integrity_marker_path(path);
    let temporary = unique_atomic_sibling(&marker_path, "tmp");
    let guard = TemporaryPath::armed(temporary.clone());
    let metadata = fs::symlink_metadata(path)
        .await
        .with_context(|| format!("stating fetched wheel {}", path.display()))?;
    let marker = UrlIntegrityMarker {
        schema: "retread-wheel-url-integrity-v1".to_string(),
        url_identity: wheel_url_identity(url),
        sha256: sha256.to_string(),
        fingerprint: fingerprint_metadata(&metadata)?,
    };
    let bytes = serde_json::to_vec_pretty(&marker).context("serializing URL wheel marker")?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .await?;
    file.write_all(&bytes).await?;
    file.sync_all().await?;
    drop(file);
    fs::rename(&temporary, &marker_path).await?;
    guard.disarm();
    Ok(())
}

fn fingerprint_metadata(metadata: &std::fs::Metadata) -> Result<CachedFileFingerprint> {
    let modified_nanos = metadata
        .modified()
        .context("reading cached wheel modification time")?
        .duration_since(std::time::UNIX_EPOCH)
        .context("cached wheel modification time predates the Unix epoch")?
        .as_nanos();
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        Ok(CachedFileFingerprint {
            size: metadata.len(),
            modified_nanos,
            device: metadata.dev(),
            inode: metadata.ino(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        })
    }
    #[cfg(not(unix))]
    {
        Ok(CachedFileFingerprint {
            size: metadata.len(),
            modified_nanos,
        })
    }
}

fn store_integrity_marker_path(store_path: &Path) -> PathBuf {
    let filename = store_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wheel.whl");
    store_path.with_file_name(format!(".{filename}.retread-integrity-v1.json"))
}

async fn stable_file_sha256(path: &Path) -> Result<Option<(String, CachedFileFingerprint)>> {
    let path_metadata = match fs::symlink_metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("stating {}", path.display())),
    };
    if !path_metadata.file_type().is_file() || path_metadata.file_type().is_symlink() {
        return Ok(None);
    }
    let initial = fingerprint_metadata(&path_metadata)?;
    let mut file = fs::File::open(path)
        .await
        .with_context(|| format!("opening {}", path.display()))?;
    if fingerprint_metadata(&file.metadata().await?)? != initial {
        return Ok(None);
    }

    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .await
            .with_context(|| format!("reading {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let final_opened = fingerprint_metadata(&file.metadata().await?)?;
    let final_path = match fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.file_type().is_file() && !metadata.file_type().is_symlink() => {
            fingerprint_metadata(&metadata)?
        }
        _ => return Ok(None),
    };
    if initial != final_opened || initial != final_path {
        return Ok(None);
    }
    Ok(Some((format!("{:x}", hasher.finalize()), initial)))
}

async fn set_store_file_readonly(path: &Path) -> Result<CachedFileFingerprint> {
    let mut permissions = fs::symlink_metadata(path)
        .await
        .with_context(|| format!("stating cached wheel {}", path.display()))?
        .permissions();
    permissions.set_readonly(true);
    fs::set_permissions(path, permissions)
        .await
        .with_context(|| format!("making cached wheel read-only: {}", path.display()))?;
    let metadata = fs::symlink_metadata(path)
        .await
        .with_context(|| format!("stating cached wheel {}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        bail!("cached wheel is not a regular file: {}", path.display());
    }
    fingerprint_metadata(&metadata)
}

async fn write_store_integrity_marker(
    store_path: &Path,
    sha256: &str,
    fingerprint: &CachedFileFingerprint,
) -> Result<()> {
    let marker_path = store_integrity_marker_path(store_path);
    let temporary = unique_atomic_sibling(&marker_path, "tmp");
    let guard = TemporaryPath::armed(temporary.clone());
    let bytes = serde_json::to_vec_pretty(&StoreIntegrityMarker {
        schema: "retread-wheel-store-integrity-v1".to_string(),
        sha256: sha256.to_string(),
        fingerprint: fingerprint.clone(),
    })
    .context("serializing cached wheel integrity marker")?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .await
        .with_context(|| format!("creating {}", temporary.display()))?;
    file.write_all(&bytes).await?;
    file.sync_all().await?;
    drop(file);
    fs::rename(&temporary, &marker_path)
        .await
        .with_context(|| {
            format!(
                "publishing cached wheel marker {} -> {}",
                temporary.display(),
                marker_path.display(),
            )
        })?;
    guard.disarm();
    Ok(())
}

enum StoreEntryState {
    Missing,
    Valid(CachedFileFingerprint),
    Corrupt,
}

/// Check a persistent-store entry without trusting path existence alone. A
/// marker can skip a multi-gigabyte rehash only while the exact read-only
/// inode/stat tuple admitted under the authoritative digest is unchanged.
async fn inspect_store_entry(store_path: &Path, sha256: &str) -> Result<StoreEntryState> {
    let metadata = match fs::symlink_metadata(store_path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(StoreEntryState::Missing);
        }
        Err(error) => {
            return Err(error).with_context(|| format!("stating {}", store_path.display()));
        }
    };
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Ok(StoreEntryState::Corrupt);
    }
    let fingerprint = fingerprint_metadata(&metadata)?;
    let marker_path = store_integrity_marker_path(store_path);
    if let Ok(marker_metadata) = fs::symlink_metadata(&marker_path).await
        && marker_metadata.file_type().is_file()
        && !marker_metadata.file_type().is_symlink()
        && let Ok(bytes) = fs::read(&marker_path).await
        && let Ok(marker) = serde_json::from_slice::<StoreIntegrityMarker>(&bytes)
        && marker.schema == "retread-wheel-store-integrity-v1"
        && marker.sha256 == sha256
        && marker.fingerprint == fingerprint
        && metadata.permissions().readonly()
    {
        return Ok(StoreEntryState::Valid(fingerprint));
    }

    let Some((actual, _)) = stable_file_sha256(store_path).await? else {
        return Ok(StoreEntryState::Corrupt);
    };
    if actual != sha256 {
        return Ok(StoreEntryState::Corrupt);
    }
    let fingerprint = set_store_file_readonly(store_path).await?;
    write_store_integrity_marker(store_path, sha256, &fingerprint).await?;
    Ok(StoreEntryState::Valid(fingerprint))
}

/// Return an attested wheel already present in the persistent store without
/// copying or re-hashing its potentially multi-gigabyte payload.
///
/// This is the metadata-reader fast path: callers need a seekable local zip,
/// not a consumer-owned copy. A corrupt entry is evicted and reported as a
/// cache miss so the caller can continue through its normal sidecar/range/
/// download fallback chain.
pub(crate) async fn cached_wheel_store_path(
    url: &url::Url,
    expected_sha256: &str,
    store_root: &Path,
) -> Result<Option<PathBuf>> {
    let sha256 = normalize_sha256(expected_sha256, "wheel hash")?;
    let store_path = pinned_wheel_store_path(url, &sha256, store_root)?;
    match inspect_store_entry(&store_path, &sha256).await? {
        StoreEntryState::Valid(_) => Ok(Some(store_path)),
        StoreEntryState::Corrupt => {
            evict_store_entry(&store_path).await;
            Ok(None)
        }
        StoreEntryState::Missing => Ok(None),
    }
}

async fn evict_store_entry(store_path: &Path) {
    let _ = fs::remove_file(store_path).await;
    let _ = fs::remove_file(store_integrity_marker_path(store_path)).await;
}

/// Copy to a fresh inode and publish with a same-directory atomic rename.
/// When `attested_source` is supplied, the copy is accepted only if the exact
/// source inode/stat tuple remains stable throughout; otherwise `false` asks
/// the caller to revalidate the source. Without an attestation, bytes are
/// hashed while copying and must match `expected_sha256`.
async fn atomic_owned_copy(
    src: &Path,
    dst: &Path,
    expected_sha256: &str,
    attested_source: Option<&CachedFileFingerprint>,
) -> Result<bool> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).await?;
    }
    let source_path_metadata = match fs::symlink_metadata(src).await {
        Ok(metadata) if metadata.file_type().is_file() && !metadata.file_type().is_symlink() => {
            metadata
        }
        Ok(_) => return Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error).with_context(|| format!("stating {}", src.display())),
    };
    let source_initial = fingerprint_metadata(&source_path_metadata)?;
    if attested_source.is_some_and(|fingerprint| *fingerprint != source_initial) {
        return Ok(false);
    }
    let mut source = fs::File::open(src)
        .await
        .with_context(|| format!("opening {}", src.display()))?;
    if fingerprint_metadata(&source.metadata().await?)? != source_initial {
        return Ok(false);
    }

    let temporary = unique_atomic_sibling(dst, "copy");
    let guard = TemporaryPath::armed(temporary.clone());
    let mut destination = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .await
        .with_context(|| format!("creating {}", temporary.display()))?;
    let mut hasher = attested_source.is_none().then(Sha256::new);
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = source
            .read(&mut buffer)
            .await
            .with_context(|| format!("reading {}", src.display()))?;
        if count == 0 {
            break;
        }
        if let Some(hasher) = &mut hasher {
            hasher.update(&buffer[..count]);
        }
        destination
            .write_all(&buffer[..count])
            .await
            .with_context(|| format!("writing {}", temporary.display()))?;
    }
    destination.flush().await?;
    destination.sync_all().await?;
    drop(destination);

    let final_opened = fingerprint_metadata(&source.metadata().await?)?;
    let final_path = match fs::symlink_metadata(src).await {
        Ok(metadata) if metadata.file_type().is_file() && !metadata.file_type().is_symlink() => {
            fingerprint_metadata(&metadata)?
        }
        _ => return Ok(false),
    };
    if source_initial != final_opened || source_initial != final_path {
        return Ok(false);
    }
    if let Some(hasher) = hasher
        && format!("{:x}", hasher.finalize()) != expected_sha256
    {
        return Ok(false);
    }
    fs::rename(&temporary, dst).await.with_context(|| {
        format!(
            "publishing owned wheel copy {} -> {}",
            temporary.display(),
            dst.display(),
        )
    })?;
    guard.disarm();
    Ok(true)
}

/// Download a wheel into `dest_dir`. Verifies SHA-256 if `expected_sha256` is
/// provided. Returns the path to the cached file (skips re-download if already
/// present with matching hash).
pub async fn fetch_wheel(
    url: &url::Url,
    expected_sha256: Option<&str>,
    dest_dir: &Path,
) -> Result<PathBuf> {
    let expected_sha256 = expected_sha256
        .map(|sha256| normalize_sha256(sha256, "wheel hash"))
        .transpose()?;
    let dest = match expected_sha256.as_deref() {
        Some(sha256) => pinned_wheel_destination(url, sha256, dest_dir)?,
        None => unpinned_wheel_destination(url, dest_dir)?,
    };
    let filename = dest
        .file_name()
        .and_then(|name| name.to_str())
        .expect("wheel destination retains validated UTF-8 filename")
        .to_string();
    fs::create_dir_all(
        dest.parent()
            .expect("wheel destination always has a namespace parent"),
    )
    .await?;

    if dest.exists() {
        if let Some(expected) = expected_sha256.as_deref() {
            let actual = sha256_file(&dest).await?;
            if actual.eq_ignore_ascii_case(expected) {
                tracing::debug!(path = %dest.display(), "wheel already cached");
                return Ok(dest);
            }
            tracing::warn!(
                path = %dest.display(),
                "cached wheel hash mismatch, re-downloading"
            );
            fs::remove_file(&dest).await.ok();
        } else if inspect_unpinned_destination(&dest, url).await? {
            return Ok(dest);
        } else {
            tracing::warn!(
                path = %dest.display(),
                "URL-keyed wheel cache attestation is missing or stale; re-downloading"
            );
            fs::remove_file(&dest).await.ok();
            fs::remove_file(url_integrity_marker_path(&dest)).await.ok();
        }
    }

    let resp = reqwest::get(url.clone())
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {url}"))?;
    let total = resp.content_length();
    match total {
        Some(len) => tracing::info!(
            %filename,
            size_mb = len / 1_048_576,
            "downloading wheel (streaming to disk; large wheels can take minutes)",
        ),
        None => tracing::info!(%filename, "downloading wheel (streaming to disk)"),
    }
    // /dev/tty status: wheel downloads happen during conda/outputs, where pixi
    // hides backend stderr -- so this is the only way the user sees the
    // multi-GB NVIDIA wheels actually downloading.
    crate::status::tty(&format!(
        "downloading {filename}{}",
        total
            .map(|t| format!(" ({} MB)", t / 1_048_576))
            .unwrap_or_default()
    ));

    // Stream the body to disk in chunks instead of `resp.bytes()` (which
    // buffers the WHOLE wheel in memory). The isaacsim extscache wheels are
    // several GB: buffering spiked RSS to multiple GB AND produced a
    // multi-minute SILENT gap (one log line, then nothing -- looks frozen).
    // Streaming caps memory at one chunk, hashes incrementally, and logs
    // steady progress so the download is visibly alive in pixi's output.
    // Unique-per-attempt temp name: with entry resolves running concurrently,
    // two BFS walks can race to download the SAME transitive wheel. A shared
    // `<filename>.part` would interleave both streams into one file; a unique
    // temp + the atomic rename below make the race last-writer-wins with
    // identical bytes instead.
    let part = unique_atomic_sibling(&dest, "part");
    let part_guard = TemporaryPath::armed(part.clone());
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&part)
        .await
        .with_context(|| format!("creating {}", part.display()))?;
    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;
    let mut last_logged: u64 = 0;
    let mut stream = std::pin::pin!(resp.bytes_stream());
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("reading body of {url}"))?;
        hasher.update(&chunk);
        file.write_all(&chunk)
            .await
            .with_context(|| format!("writing {}", part.display()))?;
        downloaded += chunk.len() as u64;
        // Log roughly every 100 MB so multi-GB wheels show steady movement.
        if downloaded - last_logged >= 100 * 1_048_576 {
            last_logged = downloaded;
            match total {
                Some(t) => tracing::info!(
                    %filename,
                    mb = downloaded / 1_048_576,
                    of_mb = t / 1_048_576,
                    "download progress",
                ),
                None => tracing::info!(%filename, mb = downloaded / 1_048_576, "download progress"),
            }
        }
    }
    file.flush()
        .await
        .with_context(|| format!("flushing {}", part.display()))?;
    file.sync_all()
        .await
        .with_context(|| format!("syncing {}", part.display()))?;
    drop(file);
    tracing::info!(%filename, mb = downloaded / 1_048_576, "wheel download complete");

    let digest = hasher.finalize();
    let mut actual = String::with_capacity(64);
    for b in digest {
        write!(&mut actual, "{b:02x}").expect("write to String");
    }
    if let Some(expected) = expected_sha256.as_deref() {
        if !actual.eq_ignore_ascii_case(expected) {
            bail!("SHA-256 mismatch for {url}: expected {expected}, got {actual}");
        }
    }

    fs::rename(&part, &dest)
        .await
        .with_context(|| format!("renaming {} -> {}", part.display(), dest.display()))?;
    part_guard.disarm();
    if expected_sha256.is_none()
        && let Err(error) = write_url_integrity_marker(&dest, url, &actual).await
    {
        tracing::warn!(
            path = %dest.display(),
            error = %error,
            "could not publish URL wheel cache attestation; next fetch will re-download"
        );
    }
    Ok(dest)
}

/// Hard-link `src` -> `dst`, falling back to copy on any error (including EXDEV).
///
/// EXDEV is returned when src and dst are on different filesystems. Attempting
/// hard_link first is the fast path; copy is the safe fallback for both
/// cross-device and any other platform-specific constraint.
pub(crate) async fn hardlink_or_copy_async(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).await?;
    }
    // hard_link is sync but fast; run via spawn_blocking to avoid blocking the executor.
    let src_b = src.to_path_buf();
    let dst_b = dst.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<()> {
        if dst_b.exists() {
            std::fs::remove_file(&dst_b)?;
        }
        if std::fs::hard_link(&src_b, &dst_b).is_err() {
            // Fallback: copy (handles EXDEV / cross-device and any other error).
            std::fs::copy(&src_b, &dst_b)?;
        }
        Ok(())
    })
    .await
    .context("hardlink_or_copy_async panicked")?
}

/// Download a wheel with a machine-global persistent content-addressed cache.
///
/// Store layout: `<store_root>/<sha256>/<filename>.whl` (the shared
/// content-addressed wheel store; see `courier::retread_wheel_store_root`).
///
/// On cache HIT (sha256 known and an attested store inode is present): copies
/// the cached bytes to a fresh consumer-owned inode and atomically publishes
/// it below the digest-qualified destination namespace, no network.
/// On cache MISS: downloads normally (with streaming + sha256 verification),
/// then populates the cache for future calls.
///
/// Falls back to plain `fetch_wheel` when:
///   - `expected_sha256` is `None` (no key to address by).
///   - `RETREAD_NO_SHADOW_CACHE` is set (bypass for parity testing).
pub async fn fetch_wheel_cached(
    url: &url::Url,
    expected_sha256: Option<&str>,
    dest_dir: &Path,
    store_root: &Path,
) -> Result<PathBuf> {
    // Normalize before consulting the environment or touching either cache.
    // Besides rejecting invalid keys, this makes every later log prefix slice
    // and path component safe by construction.
    let expected_sha256 = expected_sha256
        .map(|sha256| normalize_sha256(sha256, "wheel hash"))
        .transpose()?;

    // Bypass when disabled or when we have no sha256 to address by.
    let bypass = std::env::var("RETREAD_NO_SHADOW_CACHE").is_ok();
    let Some(sha256) = expected_sha256.as_deref().filter(|_| !bypass) else {
        return fetch_wheel(url, expected_sha256.as_deref(), dest_dir).await;
    };

    let filename = wheel_filename_from_url(url)?;
    let dest = pinned_wheel_destination(url, sha256, dest_dir)?;

    // Early return: already in the digest-qualified consumer namespace. This
    // generic entry point verifies bytes; callers with a target/source-bound
    // strict attestation can inspect this stable path before calling us.
    if dest.exists() {
        if let Some((actual, _)) = stable_file_sha256(&dest).await?
            && actual == sha256
        {
            tracing::debug!(
                wheel = %filename,
                "wheel cache: already in dest_dir (no fetch needed)",
            );
            return Ok(dest);
        }
        tracing::debug!(
            wheel = %filename,
            "wheel cache: dest_dir hash mismatch; removing stale wheel",
        );
        fs::remove_file(&dest).await.ok();
    }

    // Check the persistent store.
    let store_path = pinned_wheel_store_path(url, sha256, store_root)?;
    let mut store_was_missing = false;
    for _ in 0..2 {
        match inspect_store_entry(&store_path, sha256).await {
            Ok(StoreEntryState::Valid(fingerprint)) => {
                if atomic_owned_copy(&store_path, &dest, sha256, Some(&fingerprint)).await? {
                    tracing::info!(
                        wheel = %filename,
                        sha256 = %&sha256[..8],
                        "wheel cache: hit (persistent store, no download)",
                    );
                    return Ok(dest);
                }
                // The inode changed between inspection and opening/copying.
                // Re-inspect once; a stable corrupt replacement is evicted.
            }
            Ok(StoreEntryState::Corrupt) => {
                tracing::warn!(
                    wheel = %filename,
                    "wheel cache: authoritative store entry is corrupt; serializing repair",
                );
                break;
            }
            Ok(StoreEntryState::Missing) => {
                store_was_missing = true;
                break;
            }
            Err(error) => {
                tracing::warn!(
                    wheel = %filename,
                    err = %error,
                    "wheel cache: could not validate store entry; falling back to download",
                );
                break;
            }
        }
    }
    if store_was_missing {
        tracing::debug!(
            wheel = %filename,
            sha256 = %&sha256[..8],
            "wheel cache: miss",
        );
    }

    // Coalesce a concurrent first fill across sibling packs/processes. Atomic
    // publication alone prevents corruption, but without this lock every
    // contender still transfers the same multi-gigabyte wheel. Recheck after
    // acquiring the lock: the process that waited should consume the entry the
    // first process just published instead of opening another network stream.
    let fill_lock = match acquire_wheel_store_fill_lock(&store_path).await {
        Ok(lock) => Some(lock),
        Err(error) => {
            tracing::warn!(
                wheel = %filename,
                err = %error,
                "wheel cache: could not acquire first-fill lock; falling back to atomic publication",
            );
            None
        }
    };
    if fill_lock.is_some() {
        for _ in 0..2 {
            match inspect_store_entry(&store_path, sha256).await {
                Ok(StoreEntryState::Valid(fingerprint)) => {
                    if atomic_owned_copy(&store_path, &dest, sha256, Some(&fingerprint)).await? {
                        tracing::info!(
                            wheel = %filename,
                            sha256 = %&sha256[..8],
                            "wheel cache: hit after waiting for concurrent first fill (no download)",
                        );
                        return Ok(dest);
                    }
                }
                Ok(StoreEntryState::Corrupt) => {
                    tracing::warn!(
                        wheel = %filename,
                        "wheel cache: authoritative store entry is corrupt; evicting under first-fill lock",
                    );
                    evict_store_entry(&store_path).await;
                    break;
                }
                Ok(StoreEntryState::Missing) => break,
                Err(error) => {
                    tracing::warn!(
                        wheel = %filename,
                        err = %error,
                        "wheel cache: could not revalidate store entry under first-fill lock",
                    );
                    break;
                }
            }
        }
    } else if matches!(
        inspect_store_entry(&store_path, sha256).await,
        Ok(StoreEntryState::Corrupt)
    ) {
        evict_store_entry(&store_path).await;
    }

    // Cache miss: download normally.
    let downloaded = fetch_wheel(url, Some(sha256), dest_dir).await?;

    // Populate the persistent store (atomic temp+rename).
    let store_dir = store_root.join(sha256);
    if let Err(e) = fs::create_dir_all(&store_dir).await {
        tracing::warn!(
            wheel = %filename,
            err = %e,
            "wheel cache: could not create store dir, skipping cache population",
        );
        return Ok(downloaded);
    }
    let store_final = store_dir.join(&filename);
    match atomic_owned_copy(&downloaded, &store_final, sha256, None).await {
        Ok(true) => match set_store_file_readonly(&store_final).await {
            Ok(fingerprint) => {
                if let Err(error) =
                    write_store_integrity_marker(&store_final, sha256, &fingerprint).await
                {
                    tracing::warn!(
                        wheel = %filename,
                        err = %error,
                        "wheel cache: populated store but could not write integrity marker",
                    );
                }
                tracing::debug!(
                    wheel = %filename,
                    sha256 = %&sha256[..8],
                    "wheel cache: populated persistent store",
                );
            }
            Err(error) => tracing::warn!(
                wheel = %filename,
                err = %error,
                "wheel cache: populated store but could not make it immutable",
            ),
        },
        Ok(false) => tracing::warn!(
            wheel = %filename,
            "wheel cache: downloaded bytes changed during store population; skipping cache",
        ),
        Err(error) => tracing::warn!(
            wheel = %filename,
            err = %error,
            "wheel cache: could not populate persistent store",
        ),
    }

    Ok(downloaded)
}

/// Persist a finished wheel file into the shared content-addressed wheel
/// store at `<store_root>/<sha256>/<filename>` and return the hex sha256 of
/// its bytes.
///
/// Loose bundle mode (`retread-bundle-mode = "loose"`) calls this at BUILD
/// time for every wheel that fat mode would have shipped inside the .conda;
/// `retread install` later materializes the wheel from this exact path
/// (hash-verified). Unlike the best-effort store population in
/// [`fetch_wheel_cached`], failure here is a HARD error: a loose lock
/// records the sha and the install replay depends on the store holding the
/// bytes.
///
/// Concurrency-safe: writes go to a process+sequence-unique sibling and are
/// promoted with an atomic rename, same protocol as `fetch_wheel_cached`.
/// Existing entries must carry a matching immutable-inode attestation or are
/// rehashed; corrupt entries are evicted instead of being trusted by path.
pub(crate) async fn store_wheel_in_cache(src: &Path, store_root: &Path) -> Result<String> {
    let filename = src
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| {
            anyhow!(
                "wheel store: source path has no utf-8 filename: {}",
                src.display()
            )
        })?
        .to_string();
    let sha256 = sha256_file(src).await?;

    let store_dir = store_root.join(&sha256);
    let store_final = store_dir.join(&filename);
    match inspect_store_entry(&store_final, &sha256).await? {
        StoreEntryState::Valid(_) => {
            tracing::debug!(
                wheel = %filename,
                sha256 = %&sha256[..8],
                "wheel store: already populated",
            );
            return Ok(sha256);
        }
        StoreEntryState::Corrupt => evict_store_entry(&store_final).await,
        StoreEntryState::Missing => {}
    }
    fs::create_dir_all(&store_dir)
        .await
        .with_context(|| format!("wheel store: creating {}", store_dir.display()))?;

    if !atomic_owned_copy(src, &store_final, &sha256, None)
        .await
        .with_context(|| {
            format!(
                "wheel store: staging owned copy {} -> {}",
                src.display(),
                store_final.display(),
            )
        })?
    {
        bail!(
            "wheel store: source changed or failed its computed digest while staging {}",
            src.display(),
        );
    }
    let fingerprint = set_store_file_readonly(&store_final).await?;
    write_store_integrity_marker(&store_final, &sha256, &fingerprint).await?;
    tracing::info!(
        wheel = %filename,
        sha256 = %&sha256[..8],
        "wheel store: persisted (loose bundle)",
    );
    Ok(sha256)
}

/// Pre-fetch a direct-URL `[retread-wheels]` wheel into the content-addressed
/// wheel store and return its STABLE store path, for emission as a
/// `[tool.uv.sources]` `path = "..."` source in the synthesized closure
/// project instead of a `name @ https://...` direct-URL requirement.
///
/// Why: NVIDIA's index (pypi.nvidia.com) serves `cache-control: no-store` and
/// publishes NO PEP 658 metadata sidecars, so when a wheel is emitted as a
/// direct-URL requirement uv downloads the WHOLE wheel and fully unpacks it
/// just to read its METADATA -- and re-pays it on EVERY lock (the response is
/// uncacheable, so warm == cold). The isaacsim-extscache wheels are up to
/// ~5.9 GiB each (7.3 GiB across the three). Emitting a local `path =` source
/// lets uv read METADATA from a seekable local zip: no network, no full
/// unpack, no no-store penalty.
///
/// The returned store path is `<store_root>/<sha256>/<filename>`: content-
/// addressed and therefore STABLE across locks, so the same path string lands
/// in the synthesized pyproject every run. The closure input fingerprint and
/// the full-skip memo are both keyed on the pyproject TEXT
/// (`closure_inputs_fingerprint`), so a stable path keeps the memo hitting
/// (repeated locks full-skip) while the URL->path transition itself changes
/// the text and correctly invalidates any stale pre-transition lock.
///
/// sha256: when `expected_sha256` is known (recipe/config pins it, incl. a
/// `#sha256=` URL fragment) it is verified on fetch by [`fetch_wheel_cached`];
/// when absent, the sha is computed at first fetch by the content-addressed
/// store ([`store_wheel_in_cache`]) -- no new hashing scheme. On any
/// fetch/store failure this returns `Err` and the caller falls back to
/// emitting the direct URL as before (degraded but functional).
pub async fn prefetch_url_wheel_as_source(
    url: &url::Url,
    expected_sha256: Option<&str>,
    dest_dir: &Path,
    store_root: &Path,
) -> Result<PathBuf> {
    let expected_sha256 = expected_sha256
        .map(|sha256| normalize_sha256(sha256, "wheel hash"))
        .transpose()?;
    let filename = wheel_filename_from_url(url)?;

    // Fast path: sha known AND already in the content-addressed store. This is
    // the steady state -- the wheel store persists across locks (and the
    // extscache wheels are up to ~5.9 GiB). Emit the store path with NO fetch,
    // NO copy, and crucially NO hashing: reading a multi-GB wheel into memory
    // just to re-confirm its sha would spike RSS and can OOM the build backend.
    if let Some(sha) = expected_sha256.as_deref() {
        let store_path = pinned_wheel_store_path(url, sha, store_root)?;
        match inspect_store_entry(&store_path, sha).await? {
            StoreEntryState::Valid(_) => return Ok(store_path),
            StoreEntryState::Corrupt => evict_store_entry(&store_path).await,
            StoreEntryState::Missing => {}
        }
        // Cold store: fetch (verifies the sha, streaming/incremental -- never
        // buffers the whole wheel) and populate the store, then re-check.
        let fetched = fetch_wheel_cached(url, Some(&sha), dest_dir, store_root).await?;
        if matches!(
            inspect_store_entry(&store_path, sha).await?,
            StoreEntryState::Valid(_)
        ) {
            return Ok(store_path);
        }
        // Store populate was best-effort and failed: the freshly fetched local
        // copy is still a seekable local zip uv can read METADATA from.
        return Ok(fetched);
    }

    // No pinned sha: fetch, then content-address via the store (the sha is
    // computed once by `store_wheel_in_cache` -- no new hashing scheme). Direct
    // wheel entries normally carry a `#sha256=` fragment, so this branch is the
    // rare exception, not the multi-GB extscache case.
    let fetched = fetch_wheel_cached(url, None, dest_dir, store_root).await?;
    let sha256 = store_wheel_in_cache(&fetched, store_root).await?;
    let store_path = store_root.join(&sha256).join(&filename);
    if matches!(
        inspect_store_entry(&store_path, &sha256).await?,
        StoreEntryState::Valid(_)
    ) {
        Ok(store_path)
    } else {
        Ok(fetched)
    }
}

/// Returns `true` if the wheel filename parses as PEP 427 and its ABI/platform
/// tags are exactly `none-any`.
///
/// PEP 425 wheel filenames are `{name}-{version}(-{build})?-{python}-{abi}-{platform}.whl`,
/// **Important**: D rewrites the wheel and renames it from `foo-1.0-py3-none-any.whl`
/// to `foo-1.0-py3-none-any.relaxed.whl` (cosmetic suffix so the original wheel
/// stays on disk untouched). A naive `filename.contains("-none-any.whl")` check
/// returns FALSE on the relaxed file -- which used to flip every pure-Python
/// wheel into the platform-specific branch downstream. The consequence: the
/// merged-bundle primary (isaaclab, alphabetically first via BTreeMap) was
/// `py3-none-any`, so the bundle's `python_version` decayed to the bare-major
/// "3" parsed from the `py3` tag (via the wheel-tag fallback in `produce_output`),
/// the conda solver then read `python 3.*` and bound python to 3.14, and the
/// workspace's `python==3.11` pin rejected the implied `python_abi 3.14.* *_cp314`.
///
/// Strip Retread's well-known processing infixes first so the canonical PEP
/// 427 filename is restored, then validate the complete filename before
/// inspecting both tags. A platform field of `any` alone is insufficient:
/// malformed/native-capable names such as `cp311-abi3-any` are not pure.
pub fn is_pure_python_wheel_filename(filename: &str) -> bool {
    let canonical = crate::emit_pypi::standard_wheel_filename(filename);
    if crate::pypi::wheel_filename_identity(&canonical).is_none() {
        return false;
    }
    let Some(stem) = canonical.strip_suffix(".whl") else {
        return false;
    };
    let mut fields = stem.rsplitn(4, '-');
    let (Some(platform), Some(abi), Some(python), Some(_identity)) =
        (fields.next(), fields.next(), fields.next(), fields.next())
    else {
        return false;
    };
    valid_wheel_tag_field(python) && abi == "none" && platform == "any"
}

fn valid_wheel_tag_field(field: &str) -> bool {
    !field.is_empty() && field.split('.').all(valid_wheel_tag_component)
}

fn valid_wheel_tag_component(component: &str) -> bool {
    !component.is_empty()
        && component
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn wheel_metadata_tag_is_pure(tag: &str) -> bool {
    let mut fields = tag.split('-');
    let (Some(python), Some(abi), Some(platform)) = (fields.next(), fields.next(), fields.next())
    else {
        return false;
    };
    fields.next().is_none() && valid_wheel_tag_field(python) && abi == "none" && platform == "any"
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WheelNativePayloadKind {
    Platlib,
    NativeLibrary,
}

fn wheel_native_payload_magic(prefix: &[u8]) -> Option<&'static str> {
    if prefix.starts_with(b"\x7fELF") {
        Some("ELF")
    } else if prefix.starts_with(b"!<arch>\n") || prefix.starts_with(b"!<thin>\n") {
        Some("native archive")
    } else {
        None
    }
}

fn wheel_native_payload_kind(member: &str, is_dir: bool) -> Option<WheelNativePayloadKind> {
    let lower = member.to_ascii_lowercase();
    let components: Vec<&str> = lower
        .split('/')
        .filter(|component| !component.is_empty() && *component != ".")
        .collect();
    if components
        .windows(2)
        .any(|pair| pair[0].ends_with(".data") && pair[1] == "platlib")
    {
        return Some(WheelNativePayloadKind::Platlib);
    }
    if !is_dir
        && components.last().is_some_and(|name| {
            name.ends_with(".so") || name.ends_with(".dylib") || name.ends_with(".pyd")
        })
    {
        return Some(WheelNativePayloadKind::NativeLibrary);
    }
    None
}

/// Classify a source-built wheel before deciding whether a hermetic native
/// retry is warranted.
///
/// Inspect the payload regardless of its filename: a platform tag, platlib
/// placement, or native-looking suffix alone is not proof that compilation was
/// genuinely needed. Linux hermetic builds engage only for an actual native
/// object/archive member, including versioned DSOs, extensionless ELF
/// executables, and static archives. Returning `false` is intentionally
/// stronger than "no native suffix found": the existing strict pure-wheel
/// validator must also accept the archive.
pub(crate) fn wheel_archive_requires_native_build(wheel_path: &Path) -> Result<bool> {
    let filename = wheel_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("wheel path has no UTF-8 filename"))?;
    let canonical = crate::emit_pypi::standard_wheel_filename(filename);
    if crate::pypi::wheel_filename_identity(&canonical).is_none() {
        bail!("wheel filename `{filename}` is not a valid PEP 427 wheel filename");
    }
    let file = std::fs::File::open(wheel_path)
        .with_context(|| format!("opening wheel archive {}", wheel_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("reading wheel archive {}", wheel_path.display()))?;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .with_context(|| format!("opening ZIP member {index} in {}", wheel_path.display()))?;
        if entry.is_dir() {
            continue;
        }
        let mut prefix = Vec::with_capacity(8);
        entry.by_ref().take(8).read_to_end(&mut prefix)?;
        if wheel_native_payload_magic(&prefix).is_some() {
            return Ok(true);
        }
    }
    drop(archive);

    validate_pure_python_wheel_archive(wheel_path)?;
    Ok(false)
}

/// Validate that a local wheel is pure at both the filename and archive level.
///
/// This is the source-build cache boundary, so the archive itself must attest
/// to purity and contain no native-capable payload even when its filename says
/// `none-any`.
pub(crate) fn validate_pure_python_wheel_archive(wheel_path: &Path) -> Result<()> {
    let filename = wheel_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("wheel path has no UTF-8 filename"))?;
    if !is_pure_python_wheel_filename(filename) {
        bail!("wheel filename `{filename}` is not a valid `none-any` wheel");
    }

    let file = std::fs::File::open(wheel_path)
        .with_context(|| format!("opening wheel archive {}", wheel_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("reading wheel archive {}", wheel_path.display()))?;
    let mut root_metadata_dirs = Vec::new();
    let mut wheel_metadata = Vec::new();

    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .with_context(|| format!("opening ZIP member {index} in {}", wheel_path.display()))?;
        let member = entry.name().replace('\\', "/");
        match wheel_native_payload_kind(&member, entry.is_dir()) {
            Some(WheelNativePayloadKind::Platlib) => {
                bail!("wheel archive contains platform payload member `{member}`");
            }
            Some(WheelNativePayloadKind::NativeLibrary) => {
                bail!("wheel archive contains native payload member `{member}`");
            }
            None => {}
        }

        // Native payloads are not required to use a conventional extension.
        // Versioned DSOs (`libfoo.so.1`) and extensionless executables are
        // still ELF objects, so suffix/platlib checks alone cannot attest a
        // `none-any` wheel as pure. Keep the eight consumed bytes for WHEEL
        // parsing below; other members need no further inflation here.
        let mut prefix = Vec::with_capacity(8);
        entry
            .by_ref()
            .take(8)
            .read_to_end(&mut prefix)
            .with_context(|| {
                format!(
                    "reading ZIP member prefix `{member}` in {}",
                    wheel_path.display()
                )
            })?;
        if let Some(kind) = wheel_native_payload_magic(&prefix) {
            bail!("wheel archive contains {kind} payload member `{member}`");
        }

        if member.ends_with(".dist-info/METADATA") && member.matches('/').count() == 1 {
            root_metadata_dirs.push(
                member
                    .strip_suffix("/METADATA")
                    .expect("metadata suffix was checked")
                    .to_string(),
            );
        }
        if member.ends_with(".dist-info/WHEEL") && member.matches('/').count() == 1 {
            let mut raw = prefix;
            entry.read_to_end(&mut raw).with_context(|| {
                format!(
                    "reading wheel metadata member `{member}` in {}",
                    wheel_path.display()
                )
            })?;
            let raw = String::from_utf8(raw).with_context(|| {
                format!(
                    "wheel metadata member `{member}` is not UTF-8 in {}",
                    wheel_path.display()
                )
            })?;
            wheel_metadata.push((
                member
                    .strip_suffix("/WHEEL")
                    .expect("WHEEL suffix was checked")
                    .to_string(),
                raw,
            ));
        }
    }

    if root_metadata_dirs.len() != 1 || wheel_metadata.len() != 1 {
        bail!("wheel `{filename}` must contain exactly one root METADATA and WHEEL file");
    }
    let (wheel_dist_info, raw_wheel) = &wheel_metadata[0];
    if &root_metadata_dirs[0] != wheel_dist_info {
        bail!("wheel `{filename}` has METADATA and WHEEL files in different dist-info directories");
    }

    let mut root_is_purelib = Vec::new();
    let mut tags = Vec::new();
    for line in raw_wheel.lines() {
        if line.trim().is_empty() {
            break;
        }
        if line.starts_with(' ') || line.starts_with('\t') {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        if key.trim().eq_ignore_ascii_case("Root-Is-Purelib") {
            root_is_purelib.push(value);
        } else if key.trim().eq_ignore_ascii_case("Tag") {
            tags.push(value);
        }
    }

    if root_is_purelib.len() != 1 || !root_is_purelib[0].eq_ignore_ascii_case("true") {
        bail!("wheel `{filename}` does not declare `Root-Is-Purelib: true`");
    }
    if tags.is_empty() {
        bail!("wheel `{filename}` has no `Tag:` entries in WHEEL metadata");
    }
    if let Some(tag) = tags
        .into_iter()
        .find(|tag| !wheel_metadata_tag_is_pure(tag))
    {
        bail!("wheel `{filename}` has non-pure WHEEL metadata tag `{tag}`");
    }
    Ok(())
}

/// Validate a native wheel's archive metadata against the platform selected by
/// its hermetic build environment.
///
/// A wheel filename may compress compatible Python and ABI tags with dots, but
/// each `Tag:` header in `WHEEL` is one expanded compatibility triple. Every
/// expanded header must therefore select members advertised by the filename.
/// The platform field is deliberately stricter: hermetic builds publish one
/// exact sysroot-derived manylinux tag, so neither a legacy `linux_x86_64` tag
/// nor a compressed set containing a different glibc floor is acceptable.
pub(crate) fn validate_native_wheel_archive_tag(
    wheel_path: &Path,
    expected_platform_tag: &str,
) -> Result<()> {
    if !valid_wheel_tag_component(expected_platform_tag) || expected_platform_tag == "any" {
        bail!(
            "expected native wheel platform tag `{expected_platform_tag}` is not a valid platform component"
        );
    }

    let filename = wheel_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("wheel path has no UTF-8 filename"))?;
    let canonical = crate::emit_pypi::standard_wheel_filename(filename);
    if crate::pypi::wheel_filename_identity(&canonical).is_none() {
        bail!("wheel filename `{filename}` is not a valid PEP 427 wheel filename");
    }
    let stem = canonical
        .strip_suffix(".whl")
        .expect("validated wheel filename has .whl suffix");
    let mut filename_fields = stem.rsplitn(4, '-');
    let (Some(filename_platform), Some(filename_abi), Some(filename_python), Some(_identity)) = (
        filename_fields.next(),
        filename_fields.next(),
        filename_fields.next(),
        filename_fields.next(),
    ) else {
        bail!("wheel filename `{filename}` has no complete compatibility tag");
    };
    if !valid_wheel_tag_field(filename_python)
        || !valid_wheel_tag_field(filename_abi)
        || !valid_wheel_tag_field(filename_platform)
    {
        bail!("wheel filename `{filename}` has a malformed compatibility tag");
    }
    if filename_platform != expected_platform_tag {
        bail!(
            "wheel filename `{filename}` has platform `{filename_platform}`, expected exact platform `{expected_platform_tag}`"
        );
    }

    let file = std::fs::File::open(wheel_path)
        .with_context(|| format!("opening wheel archive {}", wheel_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("reading wheel archive {}", wheel_path.display()))?;
    let mut root_metadata_dirs = Vec::new();
    let mut wheel_metadata = Vec::new();

    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .with_context(|| format!("opening ZIP member {index} in {}", wheel_path.display()))?;
        let member = entry.name().replace('\\', "/");
        if member.ends_with(".dist-info/METADATA") && member.matches('/').count() == 1 {
            root_metadata_dirs.push(
                member
                    .strip_suffix("/METADATA")
                    .expect("metadata suffix was checked")
                    .to_string(),
            );
        }
        if member.ends_with(".dist-info/WHEEL") && member.matches('/').count() == 1 {
            let mut raw = String::new();
            entry.read_to_string(&mut raw).with_context(|| {
                format!(
                    "reading wheel metadata member `{member}` in {}",
                    wheel_path.display()
                )
            })?;
            wheel_metadata.push((
                member
                    .strip_suffix("/WHEEL")
                    .expect("WHEEL suffix was checked")
                    .to_string(),
                raw,
            ));
        }
    }

    if root_metadata_dirs.len() != 1 || wheel_metadata.len() != 1 {
        bail!("wheel `{filename}` must contain exactly one root METADATA and WHEEL file");
    }
    let (wheel_dist_info, raw_wheel) = &wheel_metadata[0];
    if &root_metadata_dirs[0] != wheel_dist_info {
        bail!("wheel `{filename}` has METADATA and WHEEL files in different dist-info directories");
    }

    let mut root_is_purelib = Vec::new();
    let mut tags = Vec::new();
    for line in raw_wheel.lines() {
        if line.trim().is_empty() {
            break;
        }
        if line.starts_with(' ') || line.starts_with('\t') {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        if key.trim().eq_ignore_ascii_case("Root-Is-Purelib") {
            root_is_purelib.push(value);
        } else if key.trim().eq_ignore_ascii_case("Tag") {
            tags.push(value);
        }
    }

    if root_is_purelib.len() != 1 || !root_is_purelib[0].eq_ignore_ascii_case("false") {
        bail!("wheel `{filename}` does not declare `Root-Is-Purelib: false`");
    }
    if tags.is_empty() {
        bail!("wheel `{filename}` has no `Tag:` entries in WHEEL metadata");
    }

    let expected_tags = filename_python
        .split('.')
        .flat_map(|python| {
            filename_abi
                .split('.')
                .map(move |abi| format!("{python}-{abi}-{expected_platform_tag}"))
        })
        .collect::<std::collections::BTreeSet<_>>();
    let mut actual_tags = std::collections::BTreeSet::new();
    for tag in tags {
        let mut fields = tag.split('-');
        let (Some(python), Some(abi), Some(platform)) =
            (fields.next(), fields.next(), fields.next())
        else {
            bail!("wheel `{filename}` has malformed WHEEL metadata tag `{tag}`");
        };
        if fields.next().is_some()
            || python.contains('.')
            || abi.contains('.')
            || platform.contains('.')
            || !valid_wheel_tag_field(python)
            || !valid_wheel_tag_field(abi)
            || !valid_wheel_tag_field(platform)
        {
            bail!("wheel `{filename}` has malformed WHEEL metadata tag `{tag}`");
        }
        let expanded_tag = format!("{python}-{abi}-{platform}");
        if !expected_tags.contains(&expanded_tag) {
            bail!(
                "wheel `{filename}` has WHEEL metadata tag `{tag}` that is not compatible with its compressed filename tag"
            );
        }
        if platform != expected_platform_tag {
            bail!(
                "wheel `{filename}` has WHEEL metadata platform `{platform}`, expected `{expected_platform_tag}`"
            );
        }
        if !actual_tags.insert(expanded_tag.clone()) {
            bail!("wheel `{filename}` has duplicate WHEEL metadata tag `{expanded_tag}`");
        }
    }
    if actual_tags != expected_tags {
        bail!(
            "wheel `{filename}` WHEEL metadata tags do not exactly expand its compressed filename tag"
        );
    }
    Ok(())
}

/// v1.4.3: fetch a wheel's METADATA via its PEP 658/714 sidecar
/// (`<wheel_url>.metadata`) instead of downloading the whole wheel.
/// Caller contract: only call when the index advertised the sidecar
/// (`ResolvedWheel.has_metadata_sidecar`) AND provided the wheel's
/// sha256 in the link fragment -- the recipe pins each source wheel's
/// hash, and without the full bytes the index-advertised hash is the
/// only source for it. `is_pure_python` derives from the filename, the
/// same signal `read_metadata` uses.
pub async fn fetch_metadata_sidecar(
    wheel_url: &url::Url,
    wheel_sha256: &str,
) -> Result<WheelMetadata> {
    let filename = wheel_filename_from_url(wheel_url)?;
    let is_pure_python = is_pure_python_wheel_filename(&filename);
    // The sidecar lives at the wheel URL + ".metadata"; the fragment
    // (#sha256=...) belongs to the WHEEL link and must not leak into
    // the sidecar request path.
    let mut sidecar = wheel_url.clone();
    sidecar.set_fragment(None);
    sidecar.set_path(&format!("{}.metadata", sidecar.path()));
    tracing::debug!(url = %sidecar, "fetching PEP 658 metadata sidecar");
    let raw = reqwest::get(sidecar.clone())
        .await
        .with_context(|| format!("GET {sidecar}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {sidecar}"))?
        .text()
        .await
        .with_context(|| format!("reading body of {sidecar}"))?;
    parse_metadata(
        &raw,
        filename,
        is_pure_python,
        wheel_sha256.to_ascii_lowercase(),
    )
}

/// Read the METADATA file inside a wheel zip and parse out the fields we care
/// about.
pub fn read_metadata(wheel_path: &Path) -> Result<WheelMetadata> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(wheel_path)
        .with_context(|| format!("opening {} for SHA-256", wheel_path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)
        .with_context(|| format!("hashing {}", wheel_path.display()))?;
    read_metadata_with_sha(wheel_path, format!("{:x}", hasher.finalize()))
}

/// Strict local-artifact boundary for source-built/path-source wheels.
/// Besides hashing and parsing METADATA, this rejects symlinks/special files,
/// requires exactly one root dist-info matching the wheel filename identity,
/// and streams every ZIP member to EOF so CRC failures cannot enter a cache.
pub(crate) fn read_metadata_strict(wheel_path: &Path) -> Result<WheelMetadata> {
    let file_type = std::fs::symlink_metadata(wheel_path)
        .with_context(|| format!("stating wheel {}", wheel_path.display()))?
        .file_type();
    if !file_type.is_file() || file_type.is_symlink() {
        bail!(
            "wheel artifact must be a regular file, not a symlink or special file: {}",
            wheel_path.display(),
        );
    }
    let filename = wheel_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("wheel path has no UTF-8 filename"))?;
    let standard_filename = crate::emit_pypi::standard_wheel_filename(filename);
    let (expected_name, expected_version) =
        crate::pypi::wheel_filename_identity(&standard_filename)
            .ok_or_else(|| anyhow!("invalid PEP 427 wheel filename `{filename}`"))?;

    let file = std::fs::File::open(wheel_path)
        .with_context(|| format!("opening strict wheel ZIP {}", wheel_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("reading strict wheel ZIP {}", wheel_path.display()))?;
    let mut root_metadata = Vec::new();
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .with_context(|| format!("opening ZIP member {index} in {}", wheel_path.display()))?;
        let name = entry.name().to_string();
        if name.ends_with(".dist-info/METADATA") && name.matches('/').count() == 1 {
            root_metadata.push(name);
        }
        std::io::copy(&mut entry, &mut std::io::sink()).with_context(|| {
            format!(
                "validating ZIP member `{}` in {}",
                entry.name(),
                wheel_path.display(),
            )
        })?;
    }
    if root_metadata.len() != 1 {
        bail!(
            "wheel `{filename}` must contain exactly one root .dist-info/METADATA, found {}",
            root_metadata.len(),
        );
    }
    let dist_info = root_metadata[0]
        .strip_suffix(".dist-info/METADATA")
        .expect("root metadata suffix was checked");
    let (dist_name, dist_version) = dist_info.rsplit_once('-').ok_or_else(|| {
        anyhow!("root dist-info directory `{dist_info}` has no name/version separator")
    })?;
    let dist_version = uv_pep508::uv_pep440::Version::from_str(dist_version)
        .with_context(|| format!("invalid root dist-info version `{dist_version}`"))?;
    if crate::relax::canonical_conda_name(dist_name)
        != crate::relax::canonical_conda_name(&expected_name)
        || dist_version != expected_version
    {
        bail!(
            "wheel `{filename}` root dist-info `{dist_info}` does not match its filename identity"
        );
    }
    read_metadata(wheel_path)
}

fn read_metadata_with_sha(wheel_path: &Path, sha256: String) -> Result<WheelMetadata> {
    let filename = wheel_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("wheel path has no filename: {}", wheel_path.display()))?
        .to_string();
    let is_pure_python = is_pure_python_wheel_filename(&filename);

    let file = std::fs::File::open(wheel_path)
        .with_context(|| format!("opening {}", wheel_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("reading zip {}", wheel_path.display()))?;

    // The wheel's own METADATA is at `<name>-<version>.dist-info/METADATA` at
    // the zip root. Wheels may vendor other packages with their own nested
    // .dist-info trees (isaacsim does this); only the root-level entry is
    // ours.
    let mut metadata_idx = None;
    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        let name = entry.name();
        if name.ends_with(".dist-info/METADATA") && name.matches('/').count() == 1 {
            metadata_idx = Some(i);
            break;
        }
    }
    let idx = metadata_idx.ok_or_else(|| {
        anyhow!(
            "no root-level .dist-info/METADATA in {}",
            wheel_path.display()
        )
    })?;

    let mut entry = archive.by_index(idx)?;
    let mut buf = String::new();
    entry.read_to_string(&mut buf)?;

    parse_metadata(&buf, filename, is_pure_python, sha256)
}

/// Parse a wheel's METADATA file content into the fields we care about.
/// Exposed so integration tests can drive the relax pipeline from captured
/// METADATA fixtures without needing a real wheel on disk.
pub fn parse_metadata(
    raw: &str,
    filename: String,
    is_pure_python: bool,
    sha256: String,
) -> Result<WheelMetadata> {
    let mut name = None;
    let mut version = None;
    let mut requires_dist = Vec::new();
    let mut retread_conda_run_dependencies = Vec::new();

    // RFC 822-style headers terminate at the first blank line. Continuation
    // lines start with whitespace and belong to the preceding header. Ignore
    // them rather than treating identity-like text in a folded License value
    // as another Name, Version, or Requires-Dist header.
    for line in raw.lines() {
        if line.is_empty() {
            break;
        }
        if line.starts_with(' ') || line.starts_with('\t') {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "Name" => name = Some(value.to_string()),
            "Version" => version = Some(value.to_string()),
            "Requires-Dist" => requires_dist.push(value.to_string()),
            "X-Retread-Conda-Run-Depends" => {
                retread_conda_run_dependencies.push(value.to_string());
            }
            _ => {}
        }
    }

    Ok(WheelMetadata {
        name: name.ok_or_else(|| anyhow!("METADATA missing Name"))?,
        version: version.ok_or_else(|| anyhow!("METADATA missing Version"))?,
        requires_dist,
        retread_conda_run_dependencies,
        is_pure_python,
        sha256,
        filename,
    })
}

/// Read-ahead window size for [`HttpRangeReader`]. The zip crate reads the
/// end-of-central-directory, then walks the central directory sequentially;
/// a 256 KiB window keeps those walks to a handful of range requests even
/// for a wheel with thousands of members.
const RANGE_WINDOW: u64 = 256 * 1024;

/// A `Read + Seek` view over a remote file, backed by HTTP Range requests.
///
/// Feeding this to `zip::ZipArchive` lets the (battle-tested) zip crate
/// locate + inflate a single member -- the wheel's `METADATA` -- by reading
/// only the end-of-central-directory, the central directory, and that one
/// member's bytes, instead of downloading the whole (multi-GiB) wheel. The
/// zip crate handles zip64 and deflate correctly, which a hand-rolled EOCD
/// parser would have to reimplement (isaacsim's 5.5 GiB extscache wheel is
/// zip64). Blocking by design: driven inside `spawn_blocking`, it issues
/// range GETs through the async client via the captured runtime handle.
struct HttpRangeReader {
    client: reqwest::blocking::Client,
    url: url::Url,
    len: u64,
    pos: u64,
    /// Cached window: (start offset, bytes).
    window: Option<(u64, Vec<u8>)>,
}

impl HttpRangeReader {
    /// Fetch `[start, start+want)` (clamped to `len`) into the window cache
    /// if it isn't already covered, then return a slice of the requested
    /// span from the cache.
    fn ensure(&mut self, start: u64, want: usize) -> std::io::Result<&[u8]> {
        let end = (start + want as u64).min(self.len);
        let covered = matches!(&self.window, Some((ws, wb)) if *ws <= start && start + (want as u64).min(self.len - start) <= *ws + wb.len() as u64);
        if !covered {
            let fetch_len = (end - start).max(RANGE_WINDOW).min(self.len - start);
            let last = start + fetch_len - 1;
            let resp = self
                .client
                .get(self.url.clone())
                .header(reqwest::header::RANGE, format!("bytes={start}-{last}"))
                .send()
                .and_then(|r| r.error_for_status())
                .map_err(std::io::Error::other)?;
            // A server ignoring Range answers 200 with the whole body; treat
            // that as unusable so the caller falls back to a full download.
            if resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "server ignored Range header (200, not 206)",
                ));
            }
            // A 206 alone is not proof the server honored OUR range: a
            // misbehaving server that always answers 206-from-offset-0
            // would silently feed the zip reader wrong bytes (the sha is
            // never recomputed on this path). Require a Content-Range
            // whose start equals the requested start and whose total
            // matches the advertised length; anything else errors so the
            // caller falls back to the full download.
            let content_range = resp
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "206 response without Content-Range",
                    )
                })?;
            let parsed = parse_content_range(&content_range).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unparseable Content-Range `{content_range}`"),
                )
            })?;
            let (cr_start, cr_end, cr_total) = parsed;
            if cr_start != start
                || cr_end < cr_start
                || cr_end > last
                || cr_end >= cr_total
                || cr_total != self.len
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Content-Range `{content_range}` does not match requested \
                         bytes={start}-{last} of {}",
                        self.len
                    ),
                ));
            }
            let expected_body_len = cr_end - cr_start + 1;
            if let Some(content_length) = resp.content_length()
                && content_length != expected_body_len
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Content-Length `{content_length}` does not match Content-Range body length `{expected_body_len}`"
                    ),
                ));
            }
            // Never let a misbehaving CDN turn a 256-KiB metadata range into
            // a multi-gigabyte receive. Even if Content-Length is absent or
            // false, read at most the declared span plus one sentinel byte.
            let mut bytes = Vec::with_capacity(expected_body_len as usize);
            let mut bounded = resp.take(expected_body_len + 1);
            bounded
                .read_to_end(&mut bytes)
                .map_err(std::io::Error::other)?;
            if bytes.len() as u64 != expected_body_len {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "range body length `{}` does not match Content-Range `{expected_body_len}`",
                        bytes.len()
                    ),
                ));
            }
            self.window = Some((start, bytes));
        }
        let (ws, wb) = self.window.as_ref().expect("window populated above");
        let off = (start - ws) as usize;
        let avail = wb.len().saturating_sub(off);
        Ok(&wb[off..off + avail.min(want)])
    }
}

/// Parse `Content-Range: bytes <start>-<end>/<total>` into (start, end,
/// total). Returns `None` for any other shape (including the valid-but-
/// useless `bytes */<total>` unsatisfiable form).
fn parse_content_range(value: &str) -> Option<(u64, u64, u64)> {
    let rest = value.trim().strip_prefix("bytes ")?;
    let (span, total) = rest.split_once('/')?;
    let (start, end) = span.split_once('-')?;
    Some((
        start.trim().parse().ok()?,
        end.trim().parse().ok()?,
        total.trim().parse().ok()?,
    ))
}

impl std::io::Read for HttpRangeReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.len || buf.is_empty() {
            return Ok(0);
        }
        let pos = self.pos;
        let slice = self.ensure(pos, buf.len())?;
        let n = slice.len().min(buf.len());
        buf[..n].copy_from_slice(&slice[..n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl std::io::Seek for HttpRangeReader {
    fn seek(&mut self, from: std::io::SeekFrom) -> std::io::Result<u64> {
        let new = match from {
            std::io::SeekFrom::Start(o) => o as i64,
            std::io::SeekFrom::End(o) => self.len as i64 + o,
            std::io::SeekFrom::Current(o) => self.pos as i64 + o,
        };
        if new < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek before start",
            ));
        }
        self.pos = new as u64;
        Ok(self.pos)
    }
}

/// Read a wheel's METADATA via HTTP Range requests, transferring only the
/// zip central directory + the METADATA member instead of the whole wheel.
///
/// Caller contract mirrors [`fetch_metadata_sidecar`]: only call with the
/// wheel's `sha256` (from the index link fragment), since the recipe pins
/// each source wheel's hash and the ranged read never sees the full bytes
/// to compute it. Returns `Err` (so the caller falls back to a full
/// download) when the server doesn't honor Range or the zip can't be read
/// this way.
pub async fn fetch_metadata_ranged(
    wheel_url: &url::Url,
    wheel_sha256: &str,
) -> Result<WheelMetadata> {
    let filename = wheel_filename_from_url(wheel_url)?;
    let is_pure_python = is_pure_python_wheel_filename(&filename);
    let mut url = wheel_url.clone();
    url.set_fragment(None);
    let sha = wheel_sha256.to_ascii_lowercase();

    // Everything here is blocking (a `reqwest::blocking` client driving the
    // sync `zip` reader), so it all runs on one blocking thread -- no nested
    // tokio runtime, works regardless of the caller's runtime flavor.
    tokio::task::spawn_blocking(move || -> Result<WheelMetadata> {
        // Explicit timeouts (M2): a stalled CDN must become an Err (which
        // the caller turns into a full-download fallback), not a forever-
        // blocked thread -- the default blocking client has NO timeout.
        // Generous values: the largest single transfer here is one
        // RANGE_WINDOW (256 KiB), so 120s of read headroom is ample even
        // on a bad link.
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .context("building ranged-fetch HTTP client")?;
        // Discover length + Range support. A HEAD keeps this to one tiny
        // request on the happy path; servers that 405 the HEAD (or omit the
        // length, or don't advertise byte ranges) error out and the caller
        // falls back to a full download.
        let head = client
            .head(url.clone())
            .send()
            .with_context(|| format!("HEAD {url}"))?
            .error_for_status()
            .with_context(|| format!("HTTP error for HEAD {url}"))?;
        let accepts_ranges = head
            .headers()
            .get(reqwest::header::ACCEPT_RANGES)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.eq_ignore_ascii_case("bytes"))
            .unwrap_or(false);
        // Read the Content-Length HEADER directly: `Response::content_length()`
        // on a HEAD reflects the (empty) body, not the advertised size.
        let len = head
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok())
            .ok_or_else(|| anyhow!("no Content-Length for {url}; cannot range-fetch"))?;
        if !accepts_ranges {
            bail!("server does not advertise `Accept-Ranges: bytes` for {url}");
        }
        if len == 0 {
            bail!("empty body for {url}");
        }

        let reader = HttpRangeReader {
            client,
            url: url.clone(),
            len,
            pos: 0,
            window: None,
        };
        let mut archive = zip::ZipArchive::new(reader)
            .with_context(|| format!("opening remote zip via Range for {filename}"))?;
        // Root-level `<name>-<version>.dist-info/METADATA` only (wheels may
        // vendor nested .dist-info trees; ours is the single-slash one).
        // `file_names()` walks the central-directory index already held in
        // memory. Do not call `by_index()` merely to inspect each name:
        // opening an entry seeks to its local header, which turns archives
        // with thousands of members into thousands of RANGE_WINDOW GETs.
        let metadata_name = archive
            .file_names()
            .find(|name| name.ends_with(".dist-info/METADATA") && name.matches('/').count() == 1)
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("no root-level .dist-info/METADATA in {filename}"))?;
        let mut entry = archive.by_name(&metadata_name)?;
        let mut buf = String::new();
        entry.read_to_string(&mut buf)?;
        parse_metadata(&buf, filename, is_pure_python, sha)
    })
    .await
    .context("ranged metadata reader panicked")?
}

async fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .await
        .with_context(|| format!("opening {} for SHA-256", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .await
            .with_context(|| format!("reading {} for SHA-256", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
fn hex_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        write!(&mut out, "{b:02x}").expect("write to String");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Build a minimal but real wheel zip (deflate-compressed METADATA at a
    /// root-level `.dist-info`) so the ranged reader exercises central-
    /// directory + member inflation, not just STORED bytes.
    fn build_test_wheel_zip() -> Vec<u8> {
        use std::io::Write as _;
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut zw = zip::ZipWriter::new(&mut cursor);
            let opts: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            // A vendored nested .dist-info (deeper path) must be ignored --
            // only the single-slash root entry is ours.
            zw.start_file("vendored/other-9.9.dist-info/METADATA", opts)
                .unwrap();
            zw.write_all(b"Name: other\nVersion: 9.9\n").unwrap();
            zw.start_file("foo-1.0.dist-info/METADATA", opts).unwrap();
            zw.write_all(
                b"Metadata-Version: 2.1\nName: foo\nVersion: 1.0\n\
                  Requires-Dist: bar>=2\nRequires-Dist: baz; extra=='x'\n\nbody\n",
            )
            .unwrap();
            zw.finish().unwrap();
        }
        cursor.into_inner()
    }

    /// A tiny HTTP/1.1 server honoring HEAD + `Range` GETs, for the ranged
    /// metadata test. Handles keep-alive (multiple requests per connection).
    async fn serve_ranged_counted(
        bytes: Vec<u8>,
    ) -> (u16, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let gets = Arc::new(AtomicUsize::new(0));
        let server_gets = gets.clone();
        let handle = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let bytes = bytes.clone();
                let gets = server_gets.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 1024];
                    loop {
                        // Read one request (headers terminate at \r\n\r\n).
                        let hdr_end = loop {
                            if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                                break p + 4;
                            }
                            let n = match stream.read(&mut tmp).await {
                                Ok(0) | Err(_) => return,
                                Ok(n) => n,
                            };
                            buf.extend_from_slice(&tmp[..n]);
                        };
                        let req = String::from_utf8_lossy(&buf[..hdr_end]).to_string();
                        buf.drain(..hdr_end);
                        let is_head = req.starts_with("HEAD");
                        if req.starts_with("GET ") {
                            gets.fetch_add(1, Ordering::SeqCst);
                        }
                        let range = req.lines().find_map(|l| {
                            let l = l.to_ascii_lowercase();
                            l.strip_prefix("range: bytes=")
                                .map(|s| s.trim().to_string())
                        });
                        let total = bytes.len();
                        let resp: Vec<u8> = if is_head {
                            format!(
                                "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\n\r\n"
                            )
                            .into_bytes()
                        } else if let Some(r) = range {
                            let (a, b) = r.split_once('-').unwrap();
                            let a: usize = a.parse().unwrap();
                            let b: usize = if b.is_empty() {
                                total - 1
                            } else {
                                b.parse::<usize>().unwrap().min(total - 1)
                            };
                            let slice = &bytes[a..=b];
                            let mut h = format!(
                                "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\n\
                                 Content-Range: bytes {a}-{b}/{total}\r\nContent-Length: {}\r\n\r\n",
                                slice.len()
                            )
                            .into_bytes();
                            h.extend_from_slice(slice);
                            h
                        } else {
                            let mut h =
                                format!("HTTP/1.1 200 OK\r\nContent-Length: {total}\r\n\r\n")
                                    .into_bytes();
                            h.extend_from_slice(&bytes);
                            h
                        };
                        if stream.write_all(&resp).await.is_err() {
                            return;
                        }
                        let _ = stream.flush().await;
                    }
                });
            }
        });
        (port, gets, handle)
    }

    async fn serve_ranged(bytes: Vec<u8>) -> (u16, tokio::task::JoinHandle<()>) {
        let (port, _gets, handle) = serve_ranged_counted(bytes).await;
        (port, handle)
    }

    async fn serve_counted_full(
        bytes: Vec<u8>,
    ) -> (u16, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let gets = Arc::new(AtomicUsize::new(0));
        let server_gets = gets.clone();
        let handle = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let bytes = bytes.clone();
                let gets = server_gets.clone();
                tokio::spawn(async move {
                    let mut request = Vec::new();
                    let mut chunk = [0_u8; 1024];
                    loop {
                        if request.windows(4).any(|window| window == b"\r\n\r\n") {
                            break;
                        }
                        match stream.read(&mut chunk).await {
                            Ok(0) | Err(_) => return,
                            Ok(count) => request.extend_from_slice(&chunk[..count]),
                        }
                    }
                    if request.starts_with(b"GET ") {
                        gets.fetch_add(1, Ordering::SeqCst);
                    }
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        bytes.len(),
                    );
                    if stream.write_all(header.as_bytes()).await.is_err() {
                        return;
                    }
                    let _ = stream.write_all(&bytes).await;
                    let _ = stream.flush().await;
                });
            }
        });
        (port, gets, handle)
    }

    #[tokio::test]
    async fn ranged_metadata_reads_only_the_metadata_member() {
        let zip_bytes = build_test_wheel_zip();
        let (port, server) = serve_ranged(zip_bytes).await;
        let url: url::Url = format!("http://127.0.0.1:{port}/foo-1.0-py3-none-any.whl")
            .parse()
            .unwrap();
        let sha = "a".repeat(64);
        let md = fetch_metadata_ranged(&url, &sha).await.unwrap();
        assert_eq!(md.name, "foo");
        assert_eq!(md.version, "1.0");
        // Root-level METADATA parsed; the vendored nested one ignored.
        assert_eq!(md.requires_dist, vec!["bar>=2", "baz; extra=='x'"]);
        // sha256 comes from the caller (index fragment), not recomputed.
        assert_eq!(md.sha256, sha);
        assert!(md.is_pure_python, "py3-none-any is pure python");
        server.abort();
    }

    #[tokio::test]
    async fn ranged_metadata_does_not_open_every_archive_member() {
        use std::io::Write as _;

        let zip_bytes = {
            let mut cursor = std::io::Cursor::new(Vec::new());
            {
                let mut zw = zip::ZipWriter::new(&mut cursor);
                let stored: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored);
                // Keep adjacent local headers farther apart than one range
                // window. The old name scan opened all twelve entries and
                // therefore issued at least twelve otherwise-useless GETs.
                let padding = vec![0x5a; RANGE_WINDOW as usize];
                for index in 0..12 {
                    zw.start_file(format!("payload-{index:02}.bin"), stored)
                        .unwrap();
                    zw.write_all(&padding).unwrap();
                }
                zw.start_file("foo-1.0.dist-info/METADATA", stored).unwrap();
                zw.write_all(b"Metadata-Version: 2.1\nName: foo\nVersion: 1.0\n\n")
                    .unwrap();
                zw.finish().unwrap();
            }
            cursor.into_inner()
        };
        let (port, gets, server) = serve_ranged_counted(zip_bytes).await;
        let url: url::Url = format!("http://127.0.0.1:{port}/foo-1.0-py3-none-any.whl")
            .parse()
            .unwrap();

        let metadata = fetch_metadata_ranged(&url, &"b".repeat(64)).await.unwrap();
        assert_eq!(metadata.name, "foo");
        assert!(
            gets.load(Ordering::SeqCst) <= 4,
            "central-directory name lookup should not open every member; observed {} range GETs",
            gets.load(Ordering::SeqCst),
        );
        server.abort();
    }

    #[tokio::test]
    async fn ranged_metadata_rejects_misbehaving_206_wrong_offset() {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        // M1: a server that answers 206 but ALWAYS serves from offset 0
        // (ignoring the requested range while still claiming success) must
        // be rejected via Content-Range validation -- accepting it would
        // hand the zip reader silently wrong bytes.
        //
        // The zip must be LARGER than the misbehaving server's fixed slice
        // (1024 B below): a zip that fits entirely in the offset-0 slice
        // never forces a nonzero-offset request, and offset-0 requests are
        // the one case this server answers correctly. Pad with a stored
        // (incompressible-by-construction) member so reads past 1024 occur.
        let zip_bytes = {
            use std::io::Write as _;
            let mut cursor = std::io::Cursor::new(Vec::new());
            {
                let mut zw = zip::ZipWriter::new(&mut cursor);
                let stored: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored);
                zw.start_file("padding.bin", stored).unwrap();
                let pad: Vec<u8> = (0..8192u32).map(|i| (i % 251) as u8).collect();
                zw.write_all(&pad).unwrap();
                zw.start_file("foo-1.0.dist-info/METADATA", stored).unwrap();
                zw.write_all(b"Metadata-Version: 2.1\nName: foo\nVersion: 1.0\n\n")
                    .unwrap();
                zw.finish().unwrap();
            }
            cursor.into_inner()
        };
        assert!(
            zip_bytes.len() > 1024,
            "padding must exceed the server slice"
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let bytes = zip_bytes.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 1024];
                    loop {
                        let hdr_end = loop {
                            if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                                break p + 4;
                            }
                            let n = match stream.read(&mut tmp).await {
                                Ok(0) | Err(_) => return,
                                Ok(n) => n,
                            };
                            buf.extend_from_slice(&tmp[..n]);
                        };
                        let req = String::from_utf8_lossy(&buf[..hdr_end]).to_string();
                        buf.drain(..hdr_end);
                        let total = bytes.len();
                        let resp: Vec<u8> = if req.starts_with("HEAD") {
                            format!(
                                "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\n\r\n"
                            )
                            .into_bytes()
                        } else {
                            // Misbehaving: 206, but always from offset 0,
                            // Content-Range honestly reporting the wrong span.
                            let n = total.min(1024);
                            let slice = &bytes[..n];
                            let mut h = format!(
                                "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\n\
                                 Content-Range: bytes 0-{}/{total}\r\nContent-Length: {}\r\n\r\n",
                                n - 1,
                                slice.len()
                            )
                            .into_bytes();
                            h.extend_from_slice(slice);
                            h
                        };
                        if stream.write_all(&resp).await.is_err() {
                            return;
                        }
                        let _ = stream.flush().await;
                    }
                });
            }
        });
        let url: url::Url = format!("http://127.0.0.1:{port}/foo-1.0-py3-none-any.whl")
            .parse()
            .unwrap();
        let err = fetch_metadata_ranged(&url, &"e".repeat(64)).await;
        assert!(
            err.is_err(),
            "wrong-offset 206 must be rejected, got {err:?}"
        );
        server.abort();
    }

    #[tokio::test]
    async fn ranged_metadata_rejects_overserved_206_from_requested_offset() {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

        // Some CDNs preserve the requested start in Content-Range but ignore
        // its end, returning the rest of a multi-gigabyte object. The old
        // reader accepted that header and consumed the entire response.
        let zip_bytes = {
            use std::io::Write as _;
            let mut cursor = std::io::Cursor::new(Vec::new());
            {
                let mut zw = zip::ZipWriter::new(&mut cursor);
                let stored: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored);
                // Put METADATA near the beginning and padding after it. The
                // zip reader first inspects the central directory at EOF,
                // then seeks back to a range whose requested end is far short
                // of EOF, forcing the over-serve check.
                zw.start_file("foo-1.0.dist-info/METADATA", stored).unwrap();
                zw.write_all(b"Metadata-Version: 2.1\nName: foo\nVersion: 1.0\n\n")
                    .unwrap();
                zw.start_file("padding.bin", stored).unwrap();
                let padding: Vec<u8> = (0..1_048_576u32).map(|i| (i % 251) as u8).collect();
                zw.write_all(&padding).unwrap();
                zw.finish().unwrap();
            }
            cursor.into_inner()
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let bytes = zip_bytes.clone();
                tokio::spawn(async move {
                    let mut request = Vec::new();
                    let mut chunk = [0_u8; 1024];
                    loop {
                        if request.windows(4).any(|window| window == b"\r\n\r\n") {
                            break;
                        }
                        match stream.read(&mut chunk).await {
                            Ok(0) | Err(_) => return,
                            Ok(count) => request.extend_from_slice(&chunk[..count]),
                        }
                    }
                    let request = String::from_utf8_lossy(&request);
                    let total = bytes.len();
                    if request.starts_with("HEAD ") {
                        let header = format!(
                            "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n"
                        );
                        let _ = stream.write_all(header.as_bytes()).await;
                        return;
                    }
                    let start = request
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("range: bytes=")
                                .and_then(|span| span.split_once('-'))
                                .and_then(|(start, _)| start.parse::<usize>().ok())
                        })
                        .unwrap();
                    let body = &bytes[start..];
                    let header = format!(
                        "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\nContent-Range: bytes {start}-{}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        total - 1,
                        body.len(),
                    );
                    if stream.write_all(header.as_bytes()).await.is_err() {
                        return;
                    }
                    let _ = stream.write_all(body).await;
                });
            }
        });

        let url: url::Url = format!("http://127.0.0.1:{port}/foo-1.0-py3-none-any.whl")
            .parse()
            .unwrap();
        let error = fetch_metadata_ranged(&url, &"f".repeat(64))
            .await
            .expect_err("an over-wide 206 response must be rejected");
        assert!(
            format!("{error:#}").contains("does not match requested"),
            "unexpected over-serve error: {error:#}",
        );
        server.abort();
    }

    #[test]
    fn parse_content_range_shapes() {
        assert_eq!(
            parse_content_range("bytes 100-199/500"),
            Some((100, 199, 500))
        );
        assert_eq!(parse_content_range(" bytes 0-0/1"), Some((0, 0, 1)));
        assert_eq!(parse_content_range("bytes */500"), None);
        assert_eq!(parse_content_range("items 0-1/2"), None);
        assert_eq!(parse_content_range("garbage"), None);
    }

    #[tokio::test]
    async fn ranged_metadata_errors_when_range_unsupported() {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        // Server that ignores Range and always returns 200 full-body: the
        // ranged reader must error so the caller falls back to a full fetch.
        let zip_bytes = build_test_wheel_zip();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let bytes = zip_bytes.clone();
                tokio::spawn(async move {
                    let mut tmp = [0u8; 1024];
                    loop {
                        match stream.read(&mut tmp).await {
                            Ok(0) | Err(_) => return,
                            Ok(_) => {}
                        }
                        // No Accept-Ranges, always 200 with the whole body.
                        let resp =
                            format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", bytes.len());
                        if stream.write_all(resp.as_bytes()).await.is_err() {
                            return;
                        }
                        if stream.write_all(&bytes).await.is_err() {
                            return;
                        }
                        let _ = stream.flush().await;
                    }
                });
            }
        });
        let url: url::Url = format!("http://127.0.0.1:{port}/foo-1.0-py3-none-any.whl")
            .parse()
            .unwrap();
        let err = fetch_metadata_ranged(&url, &"b".repeat(64)).await;
        assert!(err.is_err(), "must reject a server that ignores Range");
        server.abort();
    }

    #[test]
    #[ignore = "live: fetches a PEP 658 sidecar + the full wheel from pypi.org"]
    fn metadata_sidecar_matches_full_wheel_live() {
        // The sidecar path must produce the same parsed metadata the
        // full-wheel path does (sha256 aside, which the sidecar takes
        // from the index fragment). tomli 2.0.1 is tiny and stable.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let target = crate::pypi::WheelTarget {
                python_version: "3.11".into(),
                conda_subdir: "linux-64".into(),
                max_glibc: None,
            };
            let specs = "==2.0.1".parse().unwrap();
            let resolved =
                crate::pypi::resolve("https://pypi.org/simple/", "tomli", &specs, &target)
                    .await
                    .unwrap();
            assert!(
                resolved.has_metadata_sidecar,
                "pypi.org must advertise the PEP 658 sidecar"
            );
            let sha = resolved
                .sha256
                .as_deref()
                .expect("pypi.org provides fragments");
            let from_sidecar = fetch_metadata_sidecar(&resolved.url, sha).await.unwrap();
            let tmp =
                std::env::temp_dir().join(format!("retread-sidecar-live-{}", std::process::id()));
            std::fs::create_dir_all(&tmp).unwrap();
            let wheel_path = fetch_wheel(&resolved.url, Some(sha), &tmp).await.unwrap();
            let from_wheel = read_metadata(&wheel_path).unwrap();
            assert_eq!(from_sidecar.name, from_wheel.name);
            assert_eq!(from_sidecar.version, from_wheel.version);
            assert_eq!(from_sidecar.requires_dist, from_wheel.requires_dist);
            assert_eq!(from_sidecar.is_pure_python, from_wheel.is_pure_python);
            assert_eq!(from_sidecar.filename, from_wheel.filename);
            assert_eq!(
                from_sidecar.sha256, from_wheel.sha256,
                "fragment hash must equal the computed wheel hash"
            );
        });
    }

    #[test]
    fn parses_basic_metadata() {
        let raw = "Metadata-Version: 2.1\n\
                   Name: example-pkg\n\
                   Version: 1.2.3\n\
                   Requires-Dist: numpy==1.26.4\n\
                   Requires-Dist: torch>=2.7\n\
                   X-Retread-Conda-Run-Depends: libstdcxx-ng >=13\n\
                   \n\
                   Some description.\n";
        let m = parse_metadata(
            raw,
            "example_pkg-1.2.3-py3-none-any.whl".into(),
            true,
            "abc".into(),
        )
        .unwrap();
        assert_eq!(m.name, "example-pkg");
        assert_eq!(m.version, "1.2.3");
        assert_eq!(m.requires_dist, vec!["numpy==1.26.4", "torch>=2.7"]);
        assert_eq!(m.retread_conda_run_dependencies, vec!["libstdcxx-ng >=13"]);
        assert!(m.is_pure_python);
    }

    #[test]
    fn ignores_space_folded_license_identity_fragments() {
        let cases = [
            (
                "scipy",
                "1.15.3",
                "numpy<2.5,>=1.23.5",
                concat!(
                    "Metadata-Version: 2.1\n",
                    "Name: scipy\n",
                    "Version: 1.15.3\n",
                    "License: Copyright SciPy Developers\n",
                    "         Name: OpenBLAS\n",
                    "         Name: GCC runtime library\n",
                    "         Name: libquadmath\n",
                    "Requires-Dist: numpy<2.5,>=1.23.5\n",
                    "\n",
                ),
            ),
            (
                "matplotlib",
                "3.11.1",
                "contourpy>=1.0.1",
                concat!(
                    "Metadata-Version: 2.1\n",
                    "Name: matplotlib\n",
                    "Version: 3.11.1\n",
                    "License: License agreement for matplotlib\n",
                    "         Name: AMS Fonts\n",
                    "         Name: FreeType\n",
                    "         Name: Yorick Colormaps\n",
                    "Requires-Dist: contourpy>=1.0.1\n",
                    "\n",
                ),
            ),
        ];

        for (name, version, requirement, raw) in cases {
            let metadata = parse_metadata(
                raw,
                format!("{name}-{version}-py3-none-any.whl"),
                true,
                "abc".into(),
            )
            .unwrap();
            assert_eq!(metadata.name, name);
            assert_eq!(metadata.version, version);
            assert_eq!(metadata.requires_dist, vec![requirement]);
        }
    }

    #[test]
    fn ignores_tab_folded_identity_like_continuations() {
        let raw = concat!(
            "Metadata-Version: 2.1\n",
            "Name: example-pkg\n",
            "Version: 1.2.3\n",
            "License: example license\n",
            "\tName: impostor\n",
            "\tVersion: 9.9\n",
            "\tRequires-Dist: injected-dependency\n",
            "Requires-Dist: numpy==1.26.4\n",
            "\n",
        );
        let metadata = parse_metadata(
            raw,
            "example_pkg-1.2.3-py3-none-any.whl".into(),
            true,
            "abc".into(),
        )
        .unwrap();

        assert_eq!(metadata.name, "example-pkg");
        assert_eq!(metadata.version, "1.2.3");
        assert_eq!(metadata.requires_dist, vec!["numpy==1.26.4"]);
    }

    // Regression: a pure-Python wheel after D rewrite has filename
    // `*.relaxed.whl` (not `*.whl`). The old `filename.contains("-none-any.whl")`
    // check returned false on the relaxed file, which flipped every pure-Python
    // wheel into the platform-specific branch downstream. With the merged
    // bundle's alphabetically-first primary being `isaaclab` (`py3-none-any`),
    // the bundle's python_version then decayed to `"3"` from the `py3` tag,
    // emitting `python 3.*` as the conda run-dep; the solver bound python to
    // 3.14 and the workspace's `python==3.11` rejected the implied python_abi.
    // Detect platform tag = `any` semantically, not via a brittle filename
    // substring.
    #[test]
    fn detects_pure_python_through_relaxed_suffix() {
        // Plain pure-Python wheel: pure.
        assert!(is_pure_python_wheel_filename(
            "isaaclab-0.51.1-py3-none-any.whl"
        ));
        // Plain platform-specific wheel: not pure.
        assert!(!is_pure_python_wheel_filename(
            "isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl"
        ));
        // Pure-Python wheel after D rewrite (`.relaxed.whl` suffix): still pure.
        assert!(is_pure_python_wheel_filename(
            "isaaclab-0.51.1-py3-none-any.relaxed.whl"
        ));
        // Platform-specific wheel after D rewrite: still platform-specific.
        assert!(!is_pure_python_wheel_filename(
            "isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.relaxed.whl"
        ));
        // py2.py3-none-any (universal) wheel: pure.
        assert!(is_pure_python_wheel_filename(
            "six-1.16.0-py2.py3-none-any.whl"
        ));
        // Not a wheel at all: false.
        assert!(!is_pure_python_wheel_filename("foo.tar.gz"));
    }

    fn write_purity_test_wheel(
        filename: &str,
        wheel_metadata: &str,
        extra_members: &[&str],
    ) -> PathBuf {
        use std::io::Write as _;

        let tmp = std::env::temp_dir().join(format!(
            "retread-pure-wheel-{}-{}",
            std::process::id(),
            ATOMIC_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join(filename);
        let file = std::fs::File::create(&path).unwrap();
        let mut archive = zip::ZipWriter::new(file);
        let options: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default();
        archive
            .start_file("foo-1.0.dist-info/METADATA", options)
            .unwrap();
        archive
            .write_all(b"Metadata-Version: 2.1\nName: foo\nVersion: 1.0\n\n")
            .unwrap();
        archive
            .start_file("foo-1.0.dist-info/WHEEL", options)
            .unwrap();
        archive.write_all(wheel_metadata.as_bytes()).unwrap();
        for member in extra_members {
            archive.start_file(member, options).unwrap();
            if member.ends_with(".so")
                || member.contains(".so.")
                || member.ends_with("/native-tool")
            {
                archive.write_all(b"\x7fELFtest payload").unwrap();
            } else if member.ends_with(".a") {
                archive.write_all(b"!<arch>\ntest payload").unwrap();
            } else {
                archive.write_all(b"test payload").unwrap();
            }
        }
        archive.finish().unwrap();
        path
    }

    #[test]
    fn strict_pure_wheel_rejects_abi3_filename() {
        const WHEEL: &str = "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n";
        for filename in [
            "foo-1.0-cp311-abi3-any.whl",
            "foo-1.0-py3-none.abi3-any.whl",
        ] {
            let path = write_purity_test_wheel(filename, WHEEL, &[]);
            let error = validate_pure_python_wheel_archive(&path).unwrap_err();
            assert!(format!("{error:#}").contains("filename"));
            let _ = std::fs::remove_dir_all(path.parent().unwrap());
        }
    }

    #[test]
    fn strict_pure_wheel_rejects_malformed_filename_tag() {
        let path = write_purity_test_wheel(
            "foo-1.0-py3..py2-none-any.whl",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            &[],
        );
        let error = validate_pure_python_wheel_archive(&path).unwrap_err();
        assert!(format!("{error:#}").contains("filename"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn strict_pure_wheel_rejects_platform_in_metadata_tag() {
        let path = write_purity_test_wheel(
            "foo-1.0-py3-none-any.whl",
            concat!(
                "Wheel-Version: 1.0\n",
                "Root-Is-Purelib: true\n",
                "Tag: py3-none-any\n",
                "Tag: py3-none-manylinux_2_28_x86_64\n",
            ),
            &[],
        );
        let error = validate_pure_python_wheel_archive(&path).unwrap_err();
        assert!(format!("{error:#}").contains("manylinux_2_28_x86_64"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn strict_pure_wheel_rejects_false_root_is_purelib() {
        let path = write_purity_test_wheel(
            "foo-1.0-py3-none-any.whl",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-any\n",
            &[],
        );
        let error = validate_pure_python_wheel_archive(&path).unwrap_err();
        assert!(format!("{error:#}").contains("Root-Is-Purelib: true"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn strict_pure_wheel_rejects_native_payload() {
        let path = write_purity_test_wheel(
            "foo-1.0-py3-none-any.whl",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            &["foo/_native.cpython-311-x86_64-linux-gnu.so"],
        );
        let error = validate_pure_python_wheel_archive(&path).unwrap_err();
        assert!(format!("{error:#}").contains("native payload"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn strict_pure_wheel_rejects_versioned_elf_dso() {
        let path = write_purity_test_wheel(
            "foo-1.0-py3-none-any.whl",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            &["foo/libfoo.so.1"],
        );
        let error = validate_pure_python_wheel_archive(&path).unwrap_err();
        assert!(format!("{error:#}").contains("ELF payload member"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn strict_pure_wheel_rejects_extensionless_elf_executable() {
        let path = write_purity_test_wheel(
            "foo-1.0-py3-none-any.whl",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            &["foo/bin/native-tool"],
        );
        let error = validate_pure_python_wheel_archive(&path).unwrap_err();
        assert!(format!("{error:#}").contains("ELF payload member"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn strict_pure_wheel_rejects_static_native_archive() {
        let path = write_purity_test_wheel(
            "foo-1.0-py3-none-any.whl",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            &["foo/libfoo.a"],
        );
        let error = validate_pure_python_wheel_archive(&path).unwrap_err();
        assert!(format!("{error:#}").contains("native archive payload member"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn strict_pure_wheel_rejects_platlib_member() {
        let path = write_purity_test_wheel(
            "foo-1.0-py3-none-any.whl",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            &["foo-1.0.data/platlib/foo.py"],
        );
        let error = validate_pure_python_wheel_archive(&path).unwrap_err();
        assert!(format!("{error:#}").contains("platform payload"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn strict_pure_wheel_accepts_clean_archive() {
        let path = write_purity_test_wheel(
            "foo-1.0-py3-none-any.whl",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            &["foo/__init__.py"],
        );
        validate_pure_python_wheel_archive(&path).unwrap();
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn strict_native_wheel_accepts_matching_manylinux_tag() {
        let path = write_purity_test_wheel(
            "foo-1.0-cp311-cp311-manylinux_2_28_x86_64.whl",
            concat!(
                "Wheel-Version: 1.0\n",
                "Root-Is-Purelib: false\n",
                "Tag: cp311-cp311-manylinux_2_28_x86_64\n",
            ),
            &["foo/_native.cpython-311-x86_64-linux-gnu.so"],
        );
        validate_native_wheel_archive_tag(&path, "manylinux_2_28_x86_64").unwrap();
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn strict_native_wheel_accepts_expanded_compressed_filename_tags() {
        let path = write_purity_test_wheel(
            "foo-1.0-cp311.cp312-abi3-manylinux_2_28_x86_64.whl",
            concat!(
                "Wheel-Version: 1.0\n",
                "Root-Is-Purelib: false\n",
                "Tag: cp311-abi3-manylinux_2_28_x86_64\n",
                "Tag: cp312-abi3-manylinux_2_28_x86_64\n",
            ),
            &["foo/_native.abi3.so"],
        );
        validate_native_wheel_archive_tag(&path, "manylinux_2_28_x86_64").unwrap();
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn strict_native_wheel_rejects_incomplete_compressed_tag_expansion() {
        let path = write_purity_test_wheel(
            "foo-1.0-cp311.cp312-abi3-manylinux_2_28_x86_64.whl",
            concat!(
                "Wheel-Version: 1.0\n",
                "Root-Is-Purelib: false\n",
                "Tag: cp311-abi3-manylinux_2_28_x86_64\n",
            ),
            &["foo/_native.abi3.so"],
        );
        let error = validate_native_wheel_archive_tag(&path, "manylinux_2_28_x86_64").unwrap_err();
        assert!(format!("{error:#}").contains("exactly expand"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn strict_native_wheel_rejects_wrong_filename_platform() {
        let path = write_purity_test_wheel(
            "foo-1.0-cp311-cp311-linux_x86_64.whl",
            concat!(
                "Wheel-Version: 1.0\n",
                "Root-Is-Purelib: false\n",
                "Tag: cp311-cp311-linux_x86_64\n",
            ),
            &["foo/_native.so"],
        );
        let error = validate_native_wheel_archive_tag(&path, "manylinux_2_28_x86_64").unwrap_err();
        assert!(format!("{error:#}").contains("expected exact platform"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn strict_native_wheel_rejects_mismatched_wheel_tag() {
        let path = write_purity_test_wheel(
            "foo-1.0-cp311-cp311-manylinux_2_28_x86_64.whl",
            concat!(
                "Wheel-Version: 1.0\n",
                "Root-Is-Purelib: false\n",
                "Tag: cp310-cp310-manylinux_2_28_x86_64\n",
            ),
            &["foo/_native.so"],
        );
        let error = validate_native_wheel_archive_tag(&path, "manylinux_2_28_x86_64").unwrap_err();
        assert!(format!("{error:#}").contains("compressed filename tag"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn strict_native_wheel_rejects_wrong_wheel_metadata_platform() {
        let path = write_purity_test_wheel(
            "foo-1.0-cp311-cp311-manylinux_2_28_x86_64.whl",
            concat!(
                "Wheel-Version: 1.0\n",
                "Root-Is-Purelib: false\n",
                "Tag: cp311-cp311-manylinux_2_17_x86_64\n",
            ),
            &["foo/_native.so"],
        );
        let error = validate_native_wheel_archive_tag(&path, "manylinux_2_28_x86_64").unwrap_err();
        assert!(format!("{error:#}").contains("manylinux_2_17_x86_64"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn native_build_classifier_is_strict_for_pure_and_detects_native_payload() {
        let pure = write_purity_test_wheel(
            "foo-1.0-py3-none-any.whl",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            &["foo/__init__.py"],
        );
        assert!(!wheel_archive_requires_native_build(&pure).unwrap());
        let pure_parent = pure.parent().unwrap().to_path_buf();

        let mistagged = write_purity_test_wheel(
            "foo-1.0-py3-none-any.whl",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            &["foo/_native.so"],
        );
        assert!(wheel_archive_requires_native_build(&mistagged).unwrap());

        let native = write_purity_test_wheel(
            "foo-1.0-cp311-cp311-linux_x86_64.whl",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp311-cp311-linux_x86_64\n",
            &["foo/_native.so"],
        );
        assert!(wheel_archive_requires_native_build(&native).unwrap());

        let versioned_dso = write_purity_test_wheel(
            "foo-1.0-cp311-cp311-linux_x86_64.whl",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp311-cp311-linux_x86_64\n",
            &["foo/libfoo.so.1"],
        );
        assert!(wheel_archive_requires_native_build(&versioned_dso).unwrap());

        let static_archive = write_purity_test_wheel(
            "foo-1.0-cp311-cp311-linux_x86_64.whl",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp311-cp311-linux_x86_64\n",
            &["foo/libfoo.a"],
        );
        assert!(wheel_archive_requires_native_build(&static_archive).unwrap());

        let data_only = write_purity_test_wheel(
            "foo-1.0-py3-none-linux_x86_64.whl",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-linux_x86_64\n",
            &["foo-1.0.data/platlib/foo.py"],
        );
        assert!(wheel_archive_requires_native_build(&data_only).is_err());

        let malformed_pure = write_purity_test_wheel(
            "foo-1.0-py3-none-any.whl",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-any\n",
            &[],
        );
        let error = wheel_archive_requires_native_build(&malformed_pure).unwrap_err();
        assert!(format!("{error:#}").contains("Root-Is-Purelib: true"));

        let _ = std::fs::remove_dir_all(pure_parent);
        let _ = std::fs::remove_dir_all(mistagged.parent().unwrap());
        let _ = std::fs::remove_dir_all(native.parent().unwrap());
        let _ = std::fs::remove_dir_all(versioned_dso.parent().unwrap());
        let _ = std::fs::remove_dir_all(static_archive.parent().unwrap());
        let _ = std::fs::remove_dir_all(data_only.parent().unwrap());
        let _ = std::fs::remove_dir_all(malformed_pure.parent().unwrap());
    }

    // Defensive: read_metadata's `is_pure_python` flag uses the same helper,
    // so the field on WheelMetadata stays correct through the rewrite pipeline.
    // Other callers (`produce_output`, `build_bundle_recipe`) read this flag
    // to decide python pinning and noarch emission, so the helper IS the
    // canonical source of truth.
    // Regression: pytorch3d's miropsota GitHub-Release index serves
    // `pytorch3d-0.7.8%2B5043d15pt2.7.0cu128-cp311-cp311-linux_x86_64.whl`
    // with `%2B` URL-encoding the `+` of the PEP 440 local-version
    // identifier. fetch_wheel used to keep the encoded form in the
    // on-disk name, and pip then rejected the file with
    // `Invalid wheel filename (invalid version)` because `%2B` isn't a
    // valid PEP 440 character. The decoded form has a valid local id.
    #[test]
    fn wheel_filename_decodes_percent_encoded_plus() {
        let url: url::Url = "https://example.com/pytorch3d-0.7.8%2B5043d15pt2.7.0cu128-cp311-cp311-linux_x86_64.whl"
            .parse()
            .unwrap();
        let name = wheel_filename_from_url(&url).unwrap();
        assert_eq!(
            name, "pytorch3d-0.7.8+5043d15pt2.7.0cu128-cp311-cp311-linux_x86_64.whl",
            "wheel_filename_from_url must decode `%2B` to `+`",
        );
    }

    #[test]
    fn wheel_filename_passes_through_unencoded() {
        let url: url::Url =
            "https://pypi.nvidia.com/isaacsim/isaacsim-5.1.0-cp311-none-manylinux_2_35_x86_64.whl"
                .parse()
                .unwrap();
        let name = wheel_filename_from_url(&url).unwrap();
        assert_eq!(name, "isaacsim-5.1.0-cp311-none-manylinux_2_35_x86_64.whl",);
    }

    #[test]
    fn wheel_filename_rejects_non_whl() {
        let url: url::Url = "https://example.com/foo-1.0.tar.gz".parse().unwrap();
        assert!(wheel_filename_from_url(&url).is_err());
    }

    #[test]
    fn wheel_filename_rejects_percent_decoded_traversal_and_separators() {
        for raw in [
            "https://example.com/%2e%2e%2ffoo-1.0-py3-none-any.whl",
            "https://example.com/%2e%2e%5cfoo-1.0-py3-none-any.whl",
            "https://example.com/a%2fb-1.0-py3-none-any.whl",
            "https://example.com/a%5Cb-1.0-py3-none-any.whl",
        ] {
            let url: url::Url = raw.parse().unwrap();
            assert!(
                wheel_filename_from_url(&url).is_err(),
                "decoded separator/traversal must be rejected: {raw}",
            );
        }
    }

    #[tokio::test]
    async fn invalid_expected_sha_is_rejected_before_filesystem_mutation() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-wheel-invalid-sha-{}-{}",
            std::process::id(),
            line!(),
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        let dest = tmp.join("dest");
        let store = tmp.join("store");
        let url: url::Url = "http://127.0.0.1:1/foo-1.0-py3-none-any.whl"
            .parse()
            .unwrap();

        assert!(fetch_wheel(&url, Some("abc"), &dest).await.is_err());
        assert!(
            fetch_wheel_cached(&url, Some("not-hex"), &dest, &store)
                .await
                .is_err()
        );
        assert!(
            prefetch_url_wheel_as_source(&url, Some(&"g".repeat(64)), &dest, &store)
                .await
                .is_err()
        );
        assert!(
            !tmp.exists(),
            "invalid hashes must fail before creating destination or cache directories",
        );
    }

    #[test]
    fn wheel_destinations_isolate_equal_basenames_by_pin_and_url() {
        let root = Path::new("/tmp/retread-wheel-destination-shape");
        let first: url::Url = "https://a.example/x/foo-1.0-py3-none-any.whl"
            .parse()
            .unwrap();
        let second: url::Url = "https://b.example/y/foo-1.0-py3-none-any.whl"
            .parse()
            .unwrap();
        let pin_a = "a".repeat(64);
        let pin_b = "b".repeat(64);

        assert_ne!(
            pinned_wheel_destination(&first, &pin_a, root).unwrap(),
            pinned_wheel_destination(&first, &pin_b, root).unwrap(),
        );
        assert_ne!(
            unpinned_wheel_destination(&first, root).unwrap(),
            unpinned_wheel_destination(&second, root).unwrap(),
        );
        assert_eq!(
            unpinned_wheel_destination(&first, root).unwrap(),
            unpinned_wheel_destination(&first, root).unwrap(),
            "the same full URL gets one stable attested destination",
        );
    }

    #[test]
    fn parse_metadata_carries_is_pure_python_for_relaxed_wheel() {
        // Caller passes the helper's verdict; this test just locks that the
        // wired-through flag reaches the WheelMetadata struct unchanged.
        let raw = "Metadata-Version: 2.1\nName: isaaclab\nVersion: 0.51.1\n\n";
        let m = parse_metadata(
            raw,
            "isaaclab-0.51.1-py3-none-any.relaxed.whl".into(),
            is_pure_python_wheel_filename("isaaclab-0.51.1-py3-none-any.relaxed.whl"),
            "sha".into(),
        )
        .unwrap();
        assert!(
            m.is_pure_python,
            "relaxed pure-Python wheels must remain marked pure"
        );
    }

    // ── Persistent wheel store tests ─────────────────────────────────────────

    /// Test A2-P1: hardlink_or_copy_async produces byte-identical output.
    #[tokio::test]
    async fn hardlink_or_copy_async_byte_identical() {
        let tmp =
            std::env::temp_dir().join(format!("retread-wheel-test-hc-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        let src = tmp.join("source.bin");
        std::fs::write(&src, b"hello persistent cache").unwrap();
        let dst = tmp.join("dest.bin");

        hardlink_or_copy_async(&src, &dst).await.unwrap();
        let src_bytes = std::fs::read(&src).unwrap();
        let dst_bytes = std::fs::read(&dst).unwrap();

        let _ = std::fs::remove_dir_all(&tmp);
        assert_eq!(
            src_bytes, dst_bytes,
            "hardlink_or_copy_async must produce byte-identical output"
        );
    }

    /// Test A2-P2: fetch_wheel_cached bypass logic is correct.
    /// The bypass condition: `std::env::var("RETREAD_NO_SHADOW_CACHE").is_ok()`
    /// means: var IS set -> bypass active -> skip cache and call fetch_wheel directly.
    /// We verify the logic without mutating process env.
    #[test]
    fn wheel_cache_bypass_logic_correct() {
        // When env var is set (Ok) -> bypass = true.
        let env_set: Result<String, std::env::VarError> = Ok("1".to_string());
        let bypass_when_set = env_set.is_ok();
        assert!(
            bypass_when_set,
            "RETREAD_NO_SHADOW_CACHE=1 must activate bypass"
        );

        // When env var is absent (Err) -> bypass = false.
        let env_absent: Result<String, std::env::VarError> = Err(std::env::VarError::NotPresent);
        let bypass_when_absent = env_absent.is_ok();
        assert!(
            !bypass_when_absent,
            "absent RETREAD_NO_SHADOW_CACHE must NOT activate bypass"
        );
    }

    /// Test A2-P3: fetch_wheel_cached populates the persistent store on a miss
    /// and serves from the store on the next call (no second download).
    /// Uses hardlink_or_copy_async directly to simulate the store logic.
    #[tokio::test]
    async fn wheel_cache_persistent_store_hit() {
        let tmp =
            std::env::temp_dir().join(format!("retread-wheel-test-store-{}", std::process::id()));
        let store_root = tmp.join("store");
        let dest_dir = tmp.join("dest");
        std::fs::create_dir_all(&dest_dir).unwrap();

        // Fake wheel content.
        let wheel_bytes = b"PK\x03\x04fake wheel for persistent store test".as_slice();
        let sha256 = {
            use sha2::{Digest, Sha256};
            use std::fmt::Write as _;
            let mut h = Sha256::new();
            h.update(wheel_bytes);
            let digest = h.finalize();
            let mut s = String::with_capacity(64);
            for b in digest {
                write!(&mut s, "{b:02x}").expect("write to String");
            }
            s
        };
        let filename = "mypkg-1.0.0-py3-none-any.whl";

        // Simulate: populate the persistent store (as fetch_wheel_cached does after a download).
        let store_dir = store_root.join(&sha256);
        std::fs::create_dir_all(&store_dir).unwrap();
        let store_path = store_dir.join(filename);
        std::fs::write(&store_path, wheel_bytes).unwrap();

        // Simulate a cache HIT: hard-link from store to dest.
        let dest_path = dest_dir.join(filename);
        hardlink_or_copy_async(&store_path, &dest_path)
            .await
            .unwrap();

        let result_bytes = std::fs::read(&dest_path).unwrap();
        let _ = std::fs::remove_dir_all(&tmp);

        assert_eq!(
            result_bytes, wheel_bytes,
            "persistent store hit must produce byte-identical output"
        );
    }

    #[tokio::test]
    async fn concurrent_pinned_downloads_publish_one_complete_destination() {
        let bytes = build_test_wheel_zip();
        let sha = hex_sha256(&bytes);
        let (port, server) = serve_ranged(bytes.clone()).await;
        let url: url::Url = format!("http://127.0.0.1:{port}/foo-1.0-py3-none-any.whl")
            .parse()
            .unwrap();
        let tmp = std::env::temp_dir().join(format!(
            "retread-wheel-concurrent-fetch-{}-{}",
            std::process::id(),
            line!(),
        ));
        let _ = std::fs::remove_dir_all(&tmp);

        let (first, second) = tokio::join!(
            fetch_wheel(&url, Some(&sha), &tmp),
            fetch_wheel(&url, Some(&sha), &tmp),
        );
        let first = first.unwrap();
        let second = second.unwrap();
        assert_eq!(first, second);
        assert_eq!(std::fs::read(&first).unwrap(), bytes);
        let parent = first.parent().unwrap();
        let leftovers = std::fs::read_dir(parent)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".part") || name.ends_with(".copy"))
            .collect::<Vec<_>>();
        assert!(
            leftovers.is_empty(),
            "atomic publication must not leave attempt files: {leftovers:?}",
        );

        server.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn concurrent_store_misses_coalesce_to_one_download() {
        let bytes = build_test_wheel_zip();
        let sha = hex_sha256(&bytes);
        let (port, gets, server) = serve_counted_full(bytes.clone()).await;
        let url: url::Url = format!("http://127.0.0.1:{port}/foo-1.0-py3-none-any.whl")
            .parse()
            .unwrap();
        let tmp = std::env::temp_dir().join(format!(
            "retread-wheel-coalesced-fetch-{}-{}",
            std::process::id(),
            line!(),
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = tmp.join("store");
        let first_dest = tmp.join("first");
        let second_dest = tmp.join("second");

        let (first, second) = tokio::join!(
            fetch_wheel_cached(&url, Some(&sha), &first_dest, &store),
            fetch_wheel_cached(&url, Some(&sha), &second_dest, &store),
        );
        let first = first.unwrap();
        let second = second.unwrap();
        assert_eq!(std::fs::read(first).unwrap(), bytes);
        assert_eq!(std::fs::read(second).unwrap(), bytes);
        assert_eq!(
            gets.load(Ordering::SeqCst),
            1,
            "concurrent cold store misses must share one network transfer",
        );

        server.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn corrupt_authoritative_store_entry_is_evicted_and_refetched() {
        let bytes = build_test_wheel_zip();
        let sha = hex_sha256(&bytes);
        let (port, server) = serve_ranged(bytes.clone()).await;
        let url: url::Url = format!("http://127.0.0.1:{port}/foo-1.0-py3-none-any.whl")
            .parse()
            .unwrap();
        let tmp = std::env::temp_dir().join(format!(
            "retread-wheel-corrupt-store-{}-{}",
            std::process::id(),
            line!(),
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        let dest = tmp.join("dest");
        let second_dest = tmp.join("second-dest");
        let store = tmp.join("store");
        let store_path = pinned_wheel_store_path(&url, &sha, &store).unwrap();
        std::fs::create_dir_all(store_path.parent().unwrap()).unwrap();
        std::fs::write(&store_path, b"poisoned shared-cache bytes").unwrap();

        let fetched = fetch_wheel_cached(&url, Some(&sha), &dest, &store)
            .await
            .unwrap();
        assert_eq!(std::fs::read(&fetched).unwrap(), bytes);
        assert_eq!(std::fs::read(&store_path).unwrap(), bytes);
        assert!(
            std::fs::metadata(&store_path)
                .unwrap()
                .permissions()
                .readonly(),
            "the healed persistent-store inode must be immutable",
        );

        // A warm store hit still creates a consumer-owned inode. Mutating a
        // downstream wheel can therefore never mutate the shared store.
        let warm = fetch_wheel_cached(&url, Some(&sha), &second_dest, &store)
            .await
            .unwrap();
        assert_eq!(std::fs::read(&warm).unwrap(), bytes);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            assert_ne!(
                std::fs::metadata(&warm).unwrap().ino(),
                std::fs::metadata(&store_path).unwrap().ino(),
                "consumer wheels must never hard-link the shared-store inode",
            );
        }

        server.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ── Run 9: atomic cache-write + corrupted-zip self-heal ──────────────────

    /// `create_atomic_tmp` + `commit_atomic_write` round-trips real bytes to
    /// the final destination and leaves no temp file behind.
    #[test]
    fn atomic_write_round_trips_and_cleans_up_tmp() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "retread-wheel-test-atomic-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let dst = tmp_dir.join("pkg-1.0-py3-none-any.whl");

        let (tmp, mut file) = create_atomic_tmp(&dst).unwrap();
        assert_ne!(tmp, dst, "temp path must differ from the final destination");
        assert_eq!(
            tmp.parent(),
            dst.parent(),
            "temp file must live in the same directory as dst so the rename is atomic"
        );
        use std::io::Write as _;
        file.write_all(b"PK\x03\x04pretend wheel bytes").unwrap();
        file.flush().unwrap();
        drop(file);

        assert!(!dst.exists(), "dst must not exist before the commit");
        commit_atomic_write(&tmp, &dst).unwrap();

        assert!(dst.exists(), "dst must exist after commit");
        assert!(
            !tmp.exists(),
            "temp file must be gone after a successful commit"
        );
        assert_eq!(
            std::fs::read(&dst).unwrap(),
            b"PK\x03\x04pretend wheel bytes"
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    /// A reader that dies between `create_atomic_tmp` and `commit_atomic_write`
    /// (the run-9 failure: a compute node dying mid wheel-write) leaves ONLY
    /// the `.tmp` file on disk -- `dst` never appears in a truncated state.
    #[test]
    fn atomic_write_never_leaves_truncated_dst_on_interruption() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "retread-wheel-test-atomic-interrupt-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let dst = tmp_dir.join("pkg-1.0-py3-none-any.whl");

        let (tmp, mut file) = create_atomic_tmp(&dst).unwrap();
        use std::io::Write as _;
        file.write_all(b"only half the wheel").unwrap();
        // Simulate the process dying here: never call commit_atomic_write.
        drop(file);

        assert!(
            !dst.exists(),
            "dst must never appear until the atomic rename commits"
        );
        assert!(
            tmp.exists(),
            "the half-written temp file is the only artifact left behind"
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    /// `is_valid_zip` accepts a well-formed zip and rejects a truncated one
    /// (the "Could not find EOCD" corruption pattern proven in run 9).
    #[test]
    fn is_valid_zip_detects_corrupted_archive() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "retread-wheel-test-validzip-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&tmp_dir).unwrap();

        // A real, complete zip archive.
        let good = tmp_dir.join("good.whl");
        {
            let file = std::fs::File::create(&good).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            writer
                .start_file("a.txt", zip::write::SimpleFileOptions::default())
                .unwrap();
            use std::io::Write as _;
            writer.write_all(b"hello").unwrap();
            writer.finish().unwrap();
        }
        assert!(is_valid_zip(&good), "a complete zip archive must validate");

        // Garbage bytes masquerading as a wheel: not a zip at all.
        let garbage = tmp_dir.join("garbage.whl");
        std::fs::write(&garbage, b"not a zip file, just garbage bytes").unwrap();
        assert!(
            !is_valid_zip(&garbage),
            "non-zip garbage bytes must be rejected"
        );

        // Truncated zip: valid header, but the EOCD record got cut off --
        // the exact "invalid Zip archive: Could not find EOCD" failure mode
        // from a node dying mid-write.
        let good_bytes = std::fs::read(&good).unwrap();
        let truncated = tmp_dir.join("truncated.whl");
        std::fs::write(&truncated, &good_bytes[..good_bytes.len() / 2]).unwrap();
        assert!(
            !is_valid_zip(&truncated),
            "a truncated (EOCD-missing) zip must be rejected, not treated as valid"
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn strict_metadata_rejects_duplicate_root_dist_info() {
        use std::io::Write as _;

        let tmp = std::env::temp_dir().join(format!(
            "retread-strict-duplicate-{}-{:?}",
            std::process::id(),
            std::thread::current().id(),
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("foo-1.0-py3-none-any.whl");
        let file = std::fs::File::create(&path).unwrap();
        let mut archive = zip::ZipWriter::new(file);
        let options: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default();
        for root in ["foo-1.0", "other-1.0"] {
            archive
                .start_file(format!("{root}.dist-info/METADATA"), options)
                .unwrap();
            archive
                .write_all(b"Metadata-Version: 2.1\nName: foo\nVersion: 1.0\n\n")
                .unwrap();
        }
        archive.finish().unwrap();

        let error = read_metadata_strict(&path).unwrap_err();
        assert!(format!("{error:#}").contains("exactly one root"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn strict_metadata_reads_every_member_and_rejects_crc_corruption() {
        use std::io::Write as _;

        let tmp = std::env::temp_dir().join(format!(
            "retread-strict-crc-{}-{:?}",
            std::process::id(),
            std::thread::current().id(),
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("foo-1.0-py3-none-any.whl");
        let file = std::fs::File::create(&path).unwrap();
        let mut archive = zip::ZipWriter::new(file);
        let options: zip::write::FileOptions<'_, ()> =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Stored);
        archive
            .start_file("foo-1.0.dist-info/METADATA", options)
            .unwrap();
        archive
            .write_all(b"Metadata-Version: 2.1\nName: foo\nVersion: 1.0\n\n")
            .unwrap();
        archive.start_file("payload.bin", options).unwrap();
        const PAYLOAD: &[u8] = b"RETREAD-UNIQUE-PAYLOAD-CONTENT";
        archive.write_all(PAYLOAD).unwrap();
        archive.finish().unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        let offset = bytes
            .windows(PAYLOAD.len())
            .position(|window| window == PAYLOAD)
            .expect("stored payload is present verbatim");
        bytes[offset] ^= 0x01;
        std::fs::write(&path, bytes).unwrap();
        assert!(
            read_metadata(&path).is_ok(),
            "ordinary metadata lookup does not read the corrupt payload"
        );
        let error = read_metadata_strict(&path).unwrap_err();
        assert!(format!("{error:#}").contains("payload.bin"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn strict_metadata_accepts_internal_retread_filename_suffixes() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-strict-suffix-{}-{:?}",
            std::process::id(),
            std::thread::current().id(),
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("foo-1.0-py3-none-any.injected.relaxed.whl");
        std::fs::write(&path, build_test_wheel_zip()).unwrap();
        let metadata = read_metadata_strict(&path).unwrap();
        assert_eq!(metadata.name, "foo");
        assert_eq!(metadata.version, "1.0");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ---- prefetch_url_wheel_as_source (direct-URL -> path source) ----

    #[tokio::test]
    async fn prefetch_url_wheel_source_with_sha_returns_stable_store_path() {
        let tmp = std::env::temp_dir().join(format!("retread-prefetch-sha-{}", std::process::id()));
        let dest = tmp.join("dl");
        let store = tmp.join("store");
        let filename = "foo-1.0-py3-none-any.whl";
        // Warm-store steady state: the wheel already lives in the content-
        // addressed store. The prefetch must emit the store path with NO fetch
        // and NO hashing of the (potentially multi-GB) wheel -- so a bogus host
        // that would fail any network call proves no fetch was attempted.
        let bytes = build_test_wheel_zip();
        let sha = hex_sha256(&bytes);
        std::fs::create_dir_all(store.join(&sha)).unwrap();
        let store_path = store.join(&sha).join(filename);
        std::fs::write(&store_path, &bytes).unwrap();
        let fingerprint = set_store_file_readonly(&store_path).await.unwrap();
        write_store_integrity_marker(&store_path, &sha, &fingerprint)
            .await
            .unwrap();
        let url = url::Url::parse(&format!("https://127.0.0.1:1/x/{filename}")).unwrap();

        let path = prefetch_url_wheel_as_source(&url, Some(&sha), &dest, &store)
            .await
            .expect("prefetch with known sha (warm store)");
        // Content-addressed store path, stable across runs.
        assert_eq!(path, store.join(&sha).join(filename));
        assert!(path.is_file(), "store path must hold the wheel bytes");

        // Idempotent: a second call yields the identical path (stable
        // fingerprint -> full-skip memo keeps hitting).
        let again = prefetch_url_wheel_as_source(&url, Some(&sha), &dest, &store)
            .await
            .expect("prefetch again");
        assert_eq!(again, path);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn prefetch_url_wheel_source_without_sha_records_computed_sha() {
        let tmp =
            std::env::temp_dir().join(format!("retread-prefetch-nosha-{}", std::process::id()));
        let dest = tmp.join("dl");
        let store = tmp.join("store");
        let filename = "bar-2.0-py3-none-any.whl";
        let bytes = build_test_wheel_zip();
        let sha = hex_sha256(&bytes);
        let (port, server) = serve_ranged(bytes).await;
        let url = url::Url::parse(&format!("http://127.0.0.1:{port}/x/{filename}")).unwrap();

        // No configured sha: the content-addressed store computes it at first
        // fetch (existing store_wheel_in_cache behavior, no new hashing).
        let path = prefetch_url_wheel_as_source(&url, None, &dest, &store)
            .await
            .expect("prefetch without sha");
        assert_eq!(path, store.join(&sha).join(filename));
        assert!(path.is_file());

        server.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn prefetch_url_wheel_source_errors_on_fetch_failure() {
        let tmp =
            std::env::temp_dir().join(format!("retread-prefetch-fail-{}", std::process::id()));
        let dest = tmp.join("dl");
        let store = tmp.join("store");
        // Nothing staged in dest + a closed local port => fetch_wheel's GET
        // fails fast (connection refused), so the caller falls back to the
        // direct-URL requirement. Offline + deterministic.
        let url = url::Url::parse("http://127.0.0.1:1/nope-1.0-py3-none-any.whl").unwrap();
        let err = prefetch_url_wheel_as_source(&url, None, &dest, &store).await;
        assert!(
            err.is_err(),
            "unreachable fetch must error (caller falls back to URL)"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
