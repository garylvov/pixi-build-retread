//! C10: read a wheel's METADATA without re-inflating multi-gigabyte payloads
//! whose bytes are already attested.
//!
//! `wheel::read_metadata_strict` streams every ZIP member to a sink so its
//! SHA-256 comes out of the same pass that proves every member inflates. That
//! is the right shape for an artifact nobody has vouched for. It is the wrong
//! shape for the isaacsim extscache wheels, which arrive in
//! content-addressed storage: `<wheel store>/<sha256>/<file>.whl` beside a
//! `retread-wheel-store-integrity-v1` marker written after a verified fetch,
//! and `.retread-wheel-fetch/v1/sha256/<sha256>/<file>.whl` whose directory
//! name IS the authoritative digest. Measured on
//! `isaacsim_extscache_kit-6.0.0.1-cp312-none-manylinux_2_35_x86_64.whl`
//! (5 918 377 270 B, 75 594 members): the strict read costs 82.8 s in a live
//! relock and 50.9 s of that is a `sha256sum` recomputing the very digit
//! string that names the directory the file sits in; walking the central
//! directory and reading the single root `.dist-info/METADATA` member costs
//! 0.499 s.
//!
//! ## What is trusted, and what still proves it
//!
//! A [`WheelContentRecord`] is written only by the slow path here, only after
//! a full [`wheel::read_metadata_strict`] whose stat tuple was identical
//! before and after the read. It records the authoritative sha256, the
//! name/version that read parsed, the size, and a *structure digest*.
//!
//! The structure digest is a SHA-256 over the ZIP central directory as this
//! process parses it — for every member, in order: name, CRC-32, compressed
//! size, uncompressed size and local-header offset — plus the bytes of the
//! root `.dist-info/METADATA` member. It is not a stat field. Changing any
//! member's content changes that member's CRC-32, which is stored in the
//! central directory, so the digest moves even when size and mtime are held
//! fixed by an in-place rewrite.
//!
//! The fast path therefore requires all of:
//!
//! * a sha256 for the bytes that did not come from these bytes — the caller's
//!   authoritative digest, the store integrity marker, or the
//!   content-addressed directory name;
//! * a record filed under that sha256;
//! * the record's size equal to the file's size;
//! * the record's structure digest equal to one recomputed from the file NOW.
//!
//! Anything else falls through to the full strict read. A record is never a
//! reason to *accept*: it is only ever a reason to skip work that would have
//! recomputed a value the record already carries, and the recomputation
//! happens the moment the cheap probe disagrees.
//!
//! ## Threat model
//!
//! *A wheel replaced in place with the same size and the same mtime.* The
//! stat fingerprint (`dev`+`inode`+`size`+`mtime`) does **not** catch this and
//! is not asked to. The structure digest does: the replacement's members carry
//! different CRC-32 values, the probe disagrees with the record, and the read
//! falls back to the full strict pass — which then computes the real digest
//! and, on a content-addressed path, refuses outright because the bytes
//! disagree with the directory that names them
//! ([`ContentAddressedShaMismatch`]). What survives that is a substitution
//! that preserves every member's name, offset, compressed size, uncompressed
//! size *and* CRC-32 while changing payload bytes — a CRC-32 collision at
//! fixed length, per member, plus an unchanged METADATA. That residue is
//! stated, not defended against here.
//!
//! *`ctime` is deliberately absent from [`ContentFingerprint`].* `ctime` moves
//! when any other process links the inode, which staging does by construction
//! (`cp -al` out of a shared mirror); folding it into artifact identity is
//! what produced the ME1 false failure. This seam does not repeat that: what
//! replaces `ctime` here is not "one fewer check" but a check on the bytes
//! themselves. Note also that a fingerprint mismatch here is never a failure —
//! it is a rehash.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::wheel::WheelMetadata;

pub(crate) const RECORD_SCHEMA: &str = "retread-wheel-content-record-v1";

/// Stat identity of the exact bytes a record was measured against. `ctime` is
/// deliberately not a field; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ContentFingerprint {
    device: u64,
    inode: u64,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WheelContentRecord {
    pub(crate) schema: String,
    pub(crate) sha256: String,
    pub(crate) name: String,
    pub(crate) version: String,
    pub(crate) size: u64,
    pub(crate) structure_digest: String,
}

/// A content-addressed path whose bytes hash to something else. Terminal: the
/// store's whole contract is that the directory name is the digest.
#[derive(Debug)]
pub(crate) struct ContentAddressedShaMismatch {
    pub(crate) path: PathBuf,
    pub(crate) path_sha256: String,
    pub(crate) actual_sha256: String,
}

impl std::fmt::Display for ContentAddressedShaMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "content-addressed wheel {} disagrees with the directory that names it: path says {} but the bytes hash to {}",
            self.path.display(),
            self.path_sha256,
            self.actual_sha256,
        )
    }
}

impl std::error::Error for ContentAddressedShaMismatch {}

#[cfg(unix)]
pub(crate) fn content_fingerprint(path: &Path) -> Result<ContentFingerprint> {
    use std::os::unix::fs::MetadataExt;

    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("stating wheel for content record {}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        bail!(
            "wheel artifact must be a regular file for a content record: {}",
            path.display(),
        );
    }
    Ok(ContentFingerprint {
        device: metadata.dev(),
        inode: metadata.ino(),
        size: metadata.size(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
    })
}

#[cfg(not(unix))]
pub(crate) fn content_fingerprint(path: &Path) -> Result<ContentFingerprint> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("stating wheel for content record {}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        bail!(
            "wheel artifact must be a regular file for a content record: {}",
            path.display(),
        );
    }
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok());
    Ok(ContentFingerprint {
        device: 0,
        inode: 0,
        size: metadata.len(),
        modified_seconds: modified.map(|d| d.as_secs() as i64).unwrap_or(0),
        modified_nanoseconds: modified.map(|d| d.subsec_nanos() as i64).unwrap_or(0),
    })
}

/// Two spellings of the same inode, or one wheel read twice in one closure,
/// must not each pay a full pass. Keyed on the exact stat tuple this process
/// hashed, so a file that moved under us is a miss.
static INODE_SHA_MEMO: OnceLock<Mutex<HashMap<ContentFingerprint, String>>> = OnceLock::new();
/// Parsed records for `(sha256, fingerprint)` already proven in this process.
#[allow(clippy::type_complexity)]
static VERIFIED_MEMO: OnceLock<Mutex<HashMap<(String, ContentFingerprint), WheelMetadata>>> =
    OnceLock::new();

fn inode_sha_memo() -> &'static Mutex<HashMap<ContentFingerprint, String>> {
    INODE_SHA_MEMO.get_or_init(Default::default)
}

#[allow(clippy::type_complexity)]
fn verified_memo() -> &'static Mutex<HashMap<(String, ContentFingerprint), WheelMetadata>> {
    VERIFIED_MEMO.get_or_init(Default::default)
}

/// Drop both in-process memos. Test-only: a guard that wants to prove the
/// ON-DISK record carried the day must not be able to pass on a warm map.
#[cfg(test)]
pub(crate) fn reset_memos_for_test() {
    if let Ok(mut memo) = inode_sha_memo().lock() {
        memo.clear();
    }
    if let Ok(mut memo) = verified_memo().lock() {
        memo.clear();
    }
}

/// Test-only redirect for the record root. Guards must be able to start from
/// an EMPTY record store and to prove the on-disk record (not a warm map) is
/// what did the work, without mutating `RETREAD_CACHE_DIR` for every other
/// test sharing the process.
#[cfg(test)]
pub(crate) static RECORD_ROOT_OVERRIDE: Mutex<Option<PathBuf>> = Mutex::new(None);

fn record_dir(sha256: &str) -> PathBuf {
    #[cfg(test)]
    if let Some(root) = RECORD_ROOT_OVERRIDE.lock().ok().and_then(|root| root.clone()) {
        return root.join(sha256);
    }
    crate::courier::retread_cache_root()
        .join("wheel-content")
        .join("v1")
        .join(sha256)
}

fn record_path(sha256: &str) -> PathBuf {
    record_dir(sha256).join("record.json")
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

/// The sha256 a content-addressed location claims for these bytes, or `None`
/// when the path is not one of the two content-addressed layouts.
///
/// Layout 1 is the per-workspace pinned fetch directory
/// (`.retread-wheel-fetch/v1/sha256/<sha>/<file>`), built by
/// `wheel::pinned_wheel_destination` from an authoritative digest.
///
/// Layout 2 is the persistent wheel store (`<root>/<sha>/<file>`), recognised
/// not by guessing the root but by the `retread-wheel-store-integrity-v1`
/// marker `wheel::write_store_integrity_marker` drops beside the wheel after a
/// verified fetch; the marker's own sha must agree with the directory name.
/// This is what keeps `.retread-wheel-fetch/v1/url/<url hash>/<file>` — whose
/// parent is also 64 hex characters, of a URL and not of the bytes — out.
pub(crate) fn content_addressed_sha256(path: &Path) -> Option<String> {
    if !is_plain_wheel_filename(path.file_name()?.to_str()?) {
        return None;
    }
    if let Some(sha) = crate::wheel_rewrite::sha256_from_store_path(path) {
        return Some(sha);
    }
    let sha = path.parent()?.file_name()?.to_str()?;
    if !is_lowercase_sha256(sha) {
        return None;
    }
    let marker_sha = crate::wheel::store_integrity_marker_sha256(path)?;
    (marker_sha == sha).then(|| sha.to_string())
}

/// A wheel filename as a store or fetch entry is NAMED, as opposed to one
/// retread derived from it in the same directory.
///
/// This matters because a digest read off the directory is used to REFUSE a
/// wheel whose bytes disagree with it, and `handler`'s phase-2 relax writes
/// `with_data_path.with_extension("relaxed.whl")` -- a sibling of the entry,
/// inside the entry's own `<sha256>` directory, whose bytes are deliberately
/// NOT the ones that directory names. Reading the parent's digest for that file
/// would turn every relaxed wheel into a false refusal.
///
/// `with_extension` replaces the last component, so every derived name puts a
/// `.` inside what would be the platform tag (`…-py3-none-any.relaxed.whl`).
/// Requiring a PEP 427 name whose last field carries no `.` excludes them.
/// It also excludes the rare genuine multi-platform tag
/// (`…-macosx_10_9_x86_64.macosx_11_0_arm64.whl`), which merely gives up the
/// fast path for that wheel: conservative in the direction that costs seconds
/// rather than correctness.
fn is_plain_wheel_filename(filename: &str) -> bool {
    let Some(stem) = filename.strip_suffix(".whl") else {
        return false;
    };
    if crate::pypi::wheel_filename_identity(&crate::emit_pypi::standard_wheel_filename(filename))
        .is_none()
    {
        return false;
    }
    stem.rsplit('-').next().is_some_and(|tag| !tag.contains('.'))
}

/// SHA-256 over the ZIP central directory as parsed, plus the root
/// `.dist-info/METADATA` bytes. Cheap (one seek to the central directory and
/// one small member inflate) and sensitive to any payload edit, because the
/// central directory carries every member's CRC-32.
fn structure_digest_and_metadata(path: &Path) -> Result<(String, Vec<u8>, String)> {
    use std::io::Read;

    let file = std::fs::File::open(path)
        .with_context(|| format!("opening wheel for its central directory {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file))
        .with_context(|| format!("reading wheel central directory {}", path.display()))?;

    let mut hasher = Sha256::new();
    hasher.update(b"retread-wheel-structure-v1\0");
    hasher.update((archive.len() as u64).to_be_bytes());
    let mut metadata_member = None;
    for index in 0..archive.len() {
        let entry = archive
            .by_index_raw(index)
            .with_context(|| format!("reading ZIP entry {index} in {}", path.display()))?;
        let name = entry.name().to_string();
        hasher.update((name.len() as u64).to_be_bytes());
        hasher.update(name.as_bytes());
        hasher.update(entry.crc32().to_be_bytes());
        hasher.update(entry.compressed_size().to_be_bytes());
        hasher.update(entry.size().to_be_bytes());
        hasher.update(entry.header_start().to_be_bytes());
        if name.ends_with(".dist-info/METADATA") && name.matches('/').count() == 1 {
            if metadata_member.is_some() {
                bail!(
                    "wheel {} has more than one root .dist-info/METADATA",
                    path.display(),
                );
            }
            metadata_member = Some((index, name));
        }
    }
    let (metadata_index, metadata_name) = metadata_member.ok_or_else(|| {
        anyhow!(
            "wheel {} has no root .dist-info/METADATA",
            path.display()
        )
    })?;
    let mut metadata_bytes = Vec::new();
    archive
        .by_index(metadata_index)
        .with_context(|| format!("opening `{metadata_name}` in {}", path.display()))?
        .read_to_end(&mut metadata_bytes)
        .with_context(|| format!("reading `{metadata_name}` in {}", path.display()))?;
    hasher.update((metadata_bytes.len() as u64).to_be_bytes());
    hasher.update(&metadata_bytes);
    Ok((format!("{:x}", hasher.finalize()), metadata_bytes, metadata_name))
}

fn load_record(sha256: &str) -> Option<WheelContentRecord> {
    let path = record_path(sha256);
    let file_type = std::fs::symlink_metadata(&path).ok()?.file_type();
    if !file_type.is_file() || file_type.is_symlink() {
        return None;
    }
    let bytes = std::fs::read(&path).ok()?;
    let record: WheelContentRecord = serde_json::from_slice(&bytes).ok()?;
    (record.schema == RECORD_SCHEMA && record.sha256 == sha256).then_some(record)
}

fn write_record(record: &WheelContentRecord) -> Result<()> {
    let dir = record_dir(&record.sha256);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating wheel content record dir {}", dir.display()))?;
    let temporary = dir.join(format!(
        ".record.{}.{}.tmp",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default(),
    ));
    let bytes = serde_json::to_vec_pretty(record).context("serializing wheel content record")?;
    std::fs::write(&temporary, &bytes)
        .with_context(|| format!("writing wheel content record {}", temporary.display()))?;
    match std::fs::rename(&temporary, record_path(&record.sha256)) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = std::fs::remove_file(&temporary);
            Err(error).with_context(|| {
                format!(
                    "publishing wheel content record {}",
                    record_path(&record.sha256).display()
                )
            })
        }
    }
}

/// Assemble the metadata a verified record stands for. The parse runs on the
/// METADATA bytes the probe already pulled, so a hit costs one central
/// directory walk and no payload inflation.
fn metadata_from_record(
    path: &Path,
    record: &WheelContentRecord,
    metadata_bytes: &[u8],
) -> Result<WheelMetadata> {
    let metadata = crate::wheel::parse_metadata_bytes(path, metadata_bytes, record.sha256.clone())?;
    if crate::relax::canonical_conda_name(&metadata.name)
        != crate::relax::canonical_conda_name(&record.name)
        || metadata.version != record.version
    {
        bail!(
            "wheel content record for {} names `{}` `{}` but its METADATA now names `{}` `{}`",
            path.display(),
            record.name,
            record.version,
            metadata.name,
            metadata.version,
        );
    }
    Ok(metadata)
}

/// The strict local-artifact read, served from an attested record whenever the
/// bytes can be identified without hashing them.
///
/// `authoritative_sha256` is the caller's own digest for these bytes when it
/// has one (a pinned wheel's lock entry). When it is `None` the sha is taken
/// from the content-addressed path, and when the path is not content-addressed
/// there is nothing to look a record up by and the full strict read runs.
///
/// Semantics are those of [`crate::wheel::read_metadata_strict`]: same
/// `WheelMetadata`, same authoritative sha, same refusals — plus one refusal
/// the strict read never made, that a content-addressed wheel whose bytes
/// disagree with its own directory name is terminal.
pub(crate) fn read_metadata_verified(
    path: &Path,
    authoritative_sha256: Option<&str>,
) -> Result<WheelMetadata> {
    let started = std::time::Instant::now();
    let fingerprint = content_fingerprint(path)?;
    let path_sha256 = content_addressed_sha256(path);
    let claimed_sha256 = authoritative_sha256
        .filter(|sha| is_lowercase_sha256(sha))
        .map(str::to_string)
        .or_else(|| path_sha256.clone())
        .or_else(|| {
            inode_sha_memo()
                .lock()
                .ok()
                .and_then(|memo| memo.get(&fingerprint).cloned())
        });

    if let Some(sha256) = claimed_sha256.as_deref() {
        if let Some(metadata) = verified_memo()
            .lock()
            .ok()
            .and_then(|memo| memo.get(&(sha256.to_string(), fingerprint)).cloned())
        {
            tracing::info!(
                wheel = %path.display(),
                sha256 = %&sha256[..8],
                source = "process-memo",
                elapsed_ms = started.elapsed().as_millis() as u64,
                "bench: wheel_content_record hit",
            );
            return Ok(metadata);
        }
        if let Some(record) = load_record(sha256) {
            if record.size == fingerprint.size {
                match structure_digest_and_metadata(path) {
                    Ok((structure_digest, metadata_bytes, _)) => {
                        if structure_digest == record.structure_digest {
                            let metadata = metadata_from_record(path, &record, &metadata_bytes)?;
                            if let Ok(mut memo) = verified_memo().lock() {
                                memo.insert((sha256.to_string(), fingerprint), metadata.clone());
                            }
                            if let Ok(mut memo) = inode_sha_memo().lock() {
                                memo.insert(fingerprint, sha256.to_string());
                            }
                            tracing::info!(
                                wheel = %path.display(),
                                bytes = fingerprint.size,
                                sha256 = %&sha256[..8],
                                source = "record",
                                elapsed_ms = started.elapsed().as_millis() as u64,
                                "bench: wheel_content_record hit",
                            );
                            return Ok(metadata);
                        }
                        tracing::warn!(
                            wheel = %path.display(),
                            sha256 = %&sha256[..8],
                            reason = "structure-digest",
                            "bench: wheel_content_record miss -- re-reading strictly",
                        );
                    }
                    Err(error) => {
                        tracing::warn!(
                            wheel = %path.display(),
                            error = %error,
                            reason = "central-directory",
                            "bench: wheel_content_record miss -- re-reading strictly",
                        );
                    }
                }
            } else {
                tracing::warn!(
                    wheel = %path.display(),
                    sha256 = %&sha256[..8],
                    reason = "size",
                    "bench: wheel_content_record miss -- re-reading strictly",
                );
            }
        }
    }

    let metadata = crate::wheel::read_metadata_strict(path)?;
    let after = content_fingerprint(path)?;

    if let Some(path_sha256) = path_sha256.as_deref()
        && path_sha256 != metadata.sha256
    {
        return Err(anyhow::Error::new(ContentAddressedShaMismatch {
            path: path.to_path_buf(),
            path_sha256: path_sha256.to_string(),
            actual_sha256: metadata.sha256.clone(),
        }));
    }

    // A record is only ever filed for bytes whose stat tuple did not move
    // across the read that measured them: that window is what makes the sha,
    // the size and the structure digest one consistent statement.
    if after == fingerprint {
        match structure_digest_and_metadata(path) {
            Ok((structure_digest, _, _)) => {
                let record = WheelContentRecord {
                    schema: RECORD_SCHEMA.to_string(),
                    sha256: metadata.sha256.clone(),
                    name: metadata.name.clone(),
                    version: metadata.version.clone(),
                    size: fingerprint.size,
                    structure_digest,
                };
                if let Err(error) = write_record(&record) {
                    tracing::debug!(
                        wheel = %path.display(),
                        error = %error,
                        "wheel content record not published",
                    );
                }
                if let Ok(mut memo) = verified_memo().lock() {
                    memo.insert((metadata.sha256.clone(), fingerprint), metadata.clone());
                }
            }
            Err(error) => {
                tracing::debug!(
                    wheel = %path.display(),
                    error = %error,
                    "wheel content record not measured",
                );
            }
        }
        if let Ok(mut memo) = inode_sha_memo().lock() {
            memo.insert(fingerprint, metadata.sha256.clone());
        }
    }
    Ok(metadata)
}

/// Fast path only: `Some(metadata)` when an attested record covers these exact
/// bytes, `None` when the caller must run its own full validation. Never
/// hashes, never writes.
pub(crate) fn record_hit(path: &Path, authoritative_sha256: &str) -> Option<WheelMetadata> {
    if !is_lowercase_sha256(authoritative_sha256) {
        return None;
    }
    let fingerprint = content_fingerprint(path).ok()?;
    if let Some(metadata) = verified_memo()
        .lock()
        .ok()
        .and_then(|memo| memo.get(&(authoritative_sha256.to_string(), fingerprint)).cloned())
    {
        return Some(metadata);
    }
    let record = load_record(authoritative_sha256)?;
    if record.size != fingerprint.size {
        return None;
    }
    let (structure_digest, metadata_bytes, _) = structure_digest_and_metadata(path).ok()?;
    if structure_digest != record.structure_digest {
        return None;
    }
    let metadata = metadata_from_record(path, &record, &metadata_bytes).ok()?;
    if let Ok(mut memo) = verified_memo().lock() {
        memo.insert(
            (authoritative_sha256.to_string(), fingerprint),
            metadata.clone(),
        );
    }
    if let Ok(mut memo) = inode_sha_memo().lock() {
        memo.insert(fingerprint, authoritative_sha256.to_string());
    }
    tracing::info!(
        wheel = %path.display(),
        bytes = fingerprint.size,
        sha256 = %&authoritative_sha256[..8],
        source = "record",
        "bench: wheel_content_record hit",
    );
    Some(metadata)
}
