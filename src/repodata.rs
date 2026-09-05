//! v1.5.0: the ONE place repodata comes from.
//!
//! Both the probe layer (per-dep satisfiability) and the solve check
//! (full resolvo solves) consume conda repodata. They used to keep
//! separate full-document parses of the same JSON in separate
//! in-memory caches (plus duplicated disk-cache code): conda-forge's
//! linux-64 repodata is ~600MB of JSON, so each full parse cost
//! seconds, and the probe cache's check-then-insert pattern stampeded
//! under v1.4's 16-way batched probes -- a cold start parsed the same
//! 845MB document up to 32 times (~125s measured on the isaac6
//! benchmark).
//!
//! This module replaces all of that with memory-mapped
//! [`SparseRepoData`] handles: the JSON is mmap'd and only the byte
//! ranges for requested package names are ever parsed. One handle per
//! (channel, subdir), built once behind a per-key async OnceCell (a
//! concurrent miss AWAITS the in-flight build -- stampede is
//! impossible by construction), shared by every consumer.
//!
//! Disk cache: decompressed `repodata.json` under
//! `<cache root>/retread-repodata/` with a 30-minute TTL, where the cache
//! root is `$RATTLER_CACHE_DIR` when set (rattler's own semantics) and
//! `~/.cache/rattler/cache` otherwise -- the historical path, so existing
//! caches stay warm for default-configured hosts.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use rattler_conda_types::{Channel, ChannelConfig, ChannelUrl};
use rattler_repodata_gateway::sparse::SparseRepoData;
use sha2::{Digest, Sha256};

const REPODATA_TTL: Duration = Duration::from_secs(30 * 60);
pub(crate) const HTTP_USER_AGENT: &str = concat!("pixi-build-retread/", env!("CARGO_PKG_VERSION"));

/// One (channel, subdir)'s lazily-built sparse handle. `None` inside
/// the cell means the build was attempted and failed (channel
/// unreachable AND no disk cache) -- callers treat it as not-consulted.
/// The `None` IS cached in the `OnceCell` for the process lifetime:
/// `OnceCell::get_or_init` stores the result of the initializer (whether
/// `Some` or `None`) permanently and returns it on subsequent calls
/// without re-running the initializer. This is intentional -- retread is
/// a short-lived process (one pixi invocation), so a channel that failed
/// to fetch at startup will almost certainly fail again moments later, and
/// retrying on every probe call would add latency with no benefit.
type SparseCell = Arc<tokio::sync::OnceCell<Option<Arc<SparseRepoData>>>>;

static SPARSE_CACHE: OnceLock<Mutex<HashMap<(String, String), SparseCell>>> = OnceLock::new();

/// Get (building at most once per process) the sparse repodata handle
/// for `(channel_url, subdir)`. Returns `None` when no repodata could
/// be obtained (unreachable channel, no disk cache).
pub async fn sparse(channel_url: &str, subdir: &str) -> Option<Arc<SparseRepoData>> {
    let key = (channel_url.to_string(), subdir.to_string());
    let cell = {
        let mut map = SPARSE_CACHE
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap();
        Arc::clone(
            map.entry(key)
                .or_insert_with(|| Arc::new(tokio::sync::OnceCell::new())),
        )
    };
    let handle = cell
        .get_or_init(|| build_sparse(channel_url.to_string(), subdir.to_string()))
        .await
        .clone();
    // Never memoize a failure: a transient fetch/open error at one call site
    // must not poison every later solve in this process. Evict the cell so
    // the next caller rebuilds; a racing duplicate build is harmless.
    if handle.is_none() {
        let mut map = SPARSE_CACHE
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap();
        map.remove(&(channel_url.to_string(), subdir.to_string()));
    }
    handle
}

/// Build the ordered (channel_url, subdir) work list for a given
/// `target_subdir`. Every channel contributes two pairs: one for
/// `target_subdir` itself and one for `"noarch"` (which lives alongside
/// it on every channel). Both `sparse_pairs` and the probe layer use
/// exactly this fan-out; centralising it here ensures they can never
/// diverge.
///
/// Returns bare URL strings (no trailing slash) so they can be used
/// directly as cache keys and passed to `sparse()`.
pub fn channel_subdir_pairs(channels: &[ChannelUrl], target_subdir: &str) -> Vec<(String, String)> {
    let mut work: Vec<(String, String)> = Vec::with_capacity(channels.len() * 2);
    for channel in channels {
        let url = channel.url().as_str().trim_end_matches('/').to_string();
        work.push((url.clone(), target_subdir.to_string()));
        work.push((url, "noarch".to_string()));
    }
    work
}

/// The full (channel x [target_subdir, noarch]) fan-out every consumer
/// wants, built concurrently, returned in channel-priority order as
/// `("<channel>/<subdir>" label, handle)` pairs. Unreachable pairs are
/// skipped.
pub async fn sparse_pairs(
    channels: &[ChannelUrl],
    target_subdir: &str,
) -> Vec<(String, Arc<SparseRepoData>)> {
    let work = channel_subdir_pairs(channels, target_subdir);
    let handles = futures::future::join_all(
        work.iter()
            .map(|(channel_url, subdir)| sparse(channel_url, subdir)),
    )
    .await;
    let pairs: Vec<(String, Arc<SparseRepoData>)> = work
        .into_iter()
        .zip(handles)
        .filter_map(|((channel_url, subdir), handle)| {
            handle.map(|h| (format!("{channel_url}/{subdir}"), h))
        })
        .collect();
    // p6ad: the ONE provenance row per process, emitted the first time a
    // fan-out actually yields a universe. Fired here rather than at process
    // exit because a backend that dies mid-solve still has to have said what it
    // resolved against.
    if !pairs.is_empty() {
        emit_universe_summary();
    }
    pairs
}

async fn build_sparse(channel_url: String, subdir: String) -> Option<Arc<SparseRepoData>> {
    let path = disk_cache_path(&channel_url, &subdir);
    let cfg = ChannelConfig::default_with_root_dir(std::env::temp_dir());
    let channel = match Channel::from_str(&channel_url, &cfg) {
        Ok(channel) => channel,
        Err(error) => {
            tracing::warn!(
                channel = %channel_url, error = %error,
                "repodata: channel URL failed to parse; pair not consulted",
            );
            return None;
        }
    };
    // Two attempts: attempt 0 may discover the on-disk document is CORRUPT
    // (a truncated write published by a pre-4.10.77 release's shared `.part`
    // rename race, or an out-of-quota filesystem short write). It evicts the
    // corrupt file so attempt 1 refetches from the network. Without eviction
    // a corrupt cache file poisons this pair FOREVER: every refresh failure
    // falls back to it, every open fails, and consumers silently solve
    // against a partial channel view ("No candidates were found for
    // gcc_linux-64 13.*" while gcc_linux-64 plainly exists).
    for attempt in 0..2u8 {
        let t = std::time::Instant::now();
        // p6ad FREEZE. A frozen universe consults the document already on disk
        // whatever its age and NEVER refreshes -- that is the whole point: a
        // 30-minute TTL expiring mid-relock is exactly how MH-1's shared cache
        // moved a lock's universe under it. A missing document under a freeze
        // is a hard not-consulted with a loud row, never a quiet fetch: a
        // partial channel view that nobody was told about is the failure this
        // module already refuses on the hermetic path.
        if frozen() {
            if !path.exists() {
                tracing::error!(
                    channel = %channel_url, subdir = %subdir, path = %path.display(),
                    "repodata: FROZEN and this pair is not in the snapshot; pair not consulted",
                );
                return None;
            }
        } else if !disk_cache_is_fresh(&path).await {
            match refresh_disk_cache(&channel_url, &subdir, &path).await {
                Ok(()) => {}
                Err(e) => {
                    if !path.exists() {
                        tracing::warn!(
                            channel = %channel_url, subdir = %subdir, error = %format!("{e:#}"),
                            "repodata: unreachable and no disk cache; pair not consulted",
                        );
                        return None;
                    }
                    tracing::warn!(
                        channel = %channel_url, subdir = %subdir, error = %format!("{e:#}"),
                        "repodata: refresh failed; using stale disk cache",
                    );
                }
            }
        }
        match open_sparse_file(&channel_url, &subdir, channel.clone(), path.clone()).await {
            Some(handle) => {
                // p6ad: fingerprint the document we actually parsed, on the
                // success path only. A pair that failed to open is not part of
                // the universe and must not be named as if it were.
                record_document(&channel_url, &subdir, path.clone()).await;
                tracing::info!(
                    channel = %channel_url,
                    subdir = %subdir,
                    elapsed_ms = t.elapsed().as_millis() as u64,
                    repodata_universe = %universe_digest(),
                    "bench: sparse repodata handle built",
                );
                return Some(handle);
            }
            None if attempt == 0 && path.exists() && !frozen() => {
                tracing::warn!(
                    channel = %channel_url, subdir = %subdir, path = %path.display(),
                    "repodata: evicting corrupt disk cache and refetching",
                );
                if let Err(error) = tokio::fs::remove_file(&path).await {
                    tracing::warn!(
                        path = %path.display(), error = %error,
                        "repodata: corrupt cache eviction failed; pair not consulted",
                    );
                    return None;
                }
            }
            None => return None,
        }
    }
    None
}

/// mmap + sparse header parse on the blocking pool (the LazyRepoData
/// deserialize walks the whole document's key structure once). `None` means
/// the file is unreadable or not a well-formed repodata document.
async fn open_sparse_file(
    channel_url: &str,
    subdir: &str,
    channel: Channel,
    path: PathBuf,
) -> Option<Arc<SparseRepoData>> {
    let subdir_clone = subdir.to_string();
    let built = match tokio::task::spawn_blocking(move || {
        SparseRepoData::from_file(channel, subdir_clone, path, None)
    })
    .await
    {
        Ok(built) => built,
        Err(error) => {
            tracing::error!(
                channel = %channel_url,
                subdir = %subdir,
                error = %error,
                "repodata: sparse-open blocking task failed",
            );
            eprintln!(
                "retread: fatal worker failure while opening {channel_url}/{subdir}: {error}"
            );
            return None;
        }
    };
    match built {
        Ok(s) => Some(Arc::new(s)),
        Err(e) => {
            tracing::warn!(
                channel = %channel_url, subdir = %subdir, error = %e,
                "repodata: sparse open failed",
            );
            None
        }
    }
}

/// Fetch-and-cache one (channel, subdir), serialized ACROSS PROCESSES by an
/// exclusive flock on a sibling lock file. `pixi install --all` runs one
/// retread backend process per environment; on a cold or expired cache every
/// one of them used to fetch the same multi-hundred-MB document concurrently.
/// The flock elects one fetcher; waiters re-check freshness after acquiring
/// and reuse the winner's file.
async fn refresh_disk_cache(channel_url: &str, subdir: &str, path: &PathBuf) -> Result<()> {
    let _lock = acquire_repodata_fetch_lock(path).await?;
    if disk_cache_is_fresh(path).await {
        return Ok(());
    }
    let bytes = fetch_repodata_bytes(channel_url, subdir).await?;
    write_atomic(path, &bytes).await
}

struct RepodataFetchLock(std::fs::File);

impl Drop for RepodataFetchLock {
    fn drop(&mut self) {
        if let Err(error) = fs4::fs_std::FileExt::unlock(&self.0) {
            tracing::warn!(error = %error, "failed to unlock repodata fetch lock");
        }
    }
}

async fn acquire_repodata_fetch_lock(path: &std::path::Path) -> Result<RepodataFetchLock> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("repodata cache path has no parent: {}", path.display()))?;
    tokio::fs::create_dir_all(parent)
        .await
        .with_context(|| format!("creating repodata cache dir {}", parent.display()))?;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("repodata cache path has no UTF-8 filename"))?;
    let lock_path = parent.join(format!(".{filename}.retread-fetch-v1.lock"));
    tokio::task::spawn_blocking(move || {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .with_context(|| format!("opening repodata fetch lock {}", lock_path.display()))?;
        fs4::fs_std::FileExt::lock_exclusive(&file)
            .with_context(|| format!("locking repodata fetch {}", lock_path.display()))?;
        Ok(RepodataFetchLock(file))
    })
    .await
    .context("repodata fetch lock task panicked")?
}

async fn disk_cache_is_fresh(path: &PathBuf) -> bool {
    let Ok(meta) = tokio::fs::metadata(path).await else {
        return false;
    };
    let Ok(mtime) = meta.modified() else {
        return false;
    };
    std::time::SystemTime::now()
        .duration_since(mtime)
        .map(|age| age <= REPODATA_TTL)
        .unwrap_or(false)
}

/// Write `bytes` to `path` atomically: write to a UNIQUE temp sibling first,
/// then rename into place. The rename is load-bearing for mmap safety:
/// any concurrent reader that has already mmap'd the OLD file holds an
/// open file descriptor to the old inode -- the rename creates a NEW
/// inode at `path` without disturbing the old mapping. An in-place
/// `write` (truncate-then-overwrite) would corrupt the active mmap by
/// changing the bytes under it while the reader still holds a reference.
///
/// The temp name MUST be unique per process/call. Pre-4.10.77 this used a
/// shared `<path>.part`, and concurrent backend processes refreshing the same
/// pair interleaved their writes: one process renamed the shared temp into
/// place while a sibling was still writing it, publishing a TRUNCATED
/// document at the final path (observed: a 369MB prefix.dev/conda-forge
/// linux-64 cache cut mid-record at a page boundary, which then made every
/// hermetic toolchain solve report "No candidates were found for
/// gcc_linux-64 13.*"). `sync_all` + a length check additionally surface
/// short writes that NFS only reports at close/commit time (quota, ENOSPC)
/// BEFORE the rename can publish them.
async fn write_atomic(path: &PathBuf, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }
    let tmp = crate::wheel::unique_atomic_sibling(path, "part");
    let result = write_verified_then_rename(&tmp, path, bytes).await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&tmp).await;
    }
    result
}

async fn write_verified_then_rename(
    tmp: &std::path::Path,
    path: &std::path::Path,
    bytes: &[u8],
) -> Result<()> {
    use tokio::io::AsyncWriteExt as _;
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(tmp)
        .await
        .with_context(|| format!("creating {}", tmp.display()))?;
    file.write_all(bytes)
        .await
        .with_context(|| format!("writing {}", tmp.display()))?;
    file.sync_all()
        .await
        .with_context(|| format!("syncing {}", tmp.display()))?;
    let len = file
        .metadata()
        .await
        .with_context(|| format!("stat {}", tmp.display()))?
        .len();
    if len != bytes.len() as u64 {
        return Err(anyhow!(
            "short write to {}: {len} of {} bytes reached disk",
            tmp.display(),
            bytes.len(),
        ));
    }
    drop(file);
    tokio::fs::rename(tmp, path)
        .await
        .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Fetch `<channel>/<subdir>/repodata.json.zst` (falling back to the
/// uncompressed form) and return DECOMPRESSED JSON bytes.
async fn fetch_repodata_bytes(channel_url: &str, subdir: &str) -> Result<Vec<u8>> {
    let client = reqwest::Client::builder()
        .user_agent(HTTP_USER_AGENT)
        .build()
        .context("building HTTP client")?;
    let zst_url = format!("{channel_url}/{subdir}/repodata.json.zst");
    match client.get(&zst_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let bytes = resp
                .bytes()
                .await
                .with_context(|| format!("reading body of {zst_url}"))?;
            let decoded = tokio::task::spawn_blocking(move || zstd::decode_all(bytes.as_ref()))
                .await
                .context("zstd decoder panicked")?
                .with_context(|| format!("zstd-decoding {zst_url}"))?;
            return Ok(decoded);
        }
        Ok(resp) => {
            tracing::debug!(url = %zst_url, status = %resp.status(), "zst repodata unavailable; trying plain json");
        }
        Err(e) => {
            tracing::debug!(url = %zst_url, error = %e, "zst repodata fetch failed; trying plain json");
        }
    }
    let json_url = format!("{channel_url}/{subdir}/repodata.json");
    let resp = client
        .get(&json_url)
        .send()
        .await
        .with_context(|| format!("GET {json_url}"))?;
    if !resp.status().is_success() {
        return Err(anyhow!("HTTP {} for {json_url}", resp.status()));
    }
    Ok(resp
        .bytes()
        .await
        .with_context(|| format!("reading body of {json_url}"))?
        .to_vec())
}

/// Same path scheme both probe.rs and solve_check.rs used before the
/// unification, so caches written by older versions stay warm.
fn disk_cache_path(channel_url: &str, subdir: &str) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(channel_url.as_bytes());
    hasher.update(b"|");
    hasher.update(subdir.as_bytes());
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(16);
    for b in &digest[..8] {
        hex.push_str(&format!("{b:02x}"));
    }
    let dir = dirs_cache_root().join("retread-repodata");
    let slug = channel_url
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or("channel")
        .replace(|c: char| !c.is_ascii_alphanumeric(), "_");
    dir.join(format!("{slug}--{subdir}--{hex}.json"))
}

/// Cache root, honoring `RATTLER_CACHE_DIR` exactly like rattler itself
/// does (the variable names the cache ROOT, i.e. the equivalent of
/// `~/.cache/rattler/cache`). Hardcoding `$HOME` here stranded the repodata
/// cache on a shared NFS home even when the whole pipeline had been pointed
/// at node-local storage: an over-quota home then made the cache unwritable
/// on EVERY node at once (the fetch lock could not even be created), and the
/// hermetic toolchain solve degraded to "conda-forge repodata incomplete".
fn dirs_cache_root() -> PathBuf {
    cache_root_from(
        std::env::var_os("RATTLER_CACHE_DIR").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )
}

fn cache_root_from(
    rattler_cache_dir: Option<&std::ffi::OsStr>,
    home: Option<&std::ffi::OsStr>,
) -> PathBuf {
    if let Some(dir) = rattler_cache_dir.filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    if let Some(home) = home {
        PathBuf::from(home)
            .join(".cache")
            .join("rattler")
            .join("cache")
    } else {
        std::env::temp_dir().join("retread-cache")
    }
}

// ---------------------------------------------------------------------------
// p6ad: THE CONDA CANDIDATE UNIVERSE A LOCK WAS RESOLVED AGAINST
// ---------------------------------------------------------------------------
//
// p6ac named the mechanism and stopped one step short of the artefact. The
// vendored set of a bundle is the closure MINUS whatever the joint conda route
// validation left on the conda side, and that validation answers to the conda
// candidate universe of the moment: `conda_forge--linux-64.json` and
// `conda_forge--noarch.json` in the shared cache below are refreshed by any
// lane at any time (MH-1), and two canonical relocks hours apart legitimately
// resolve against different documents. p6ac-1 boarded the consequence: NOTHING
// PINS THAT UNIVERSE INTO THE LOCK, so no artefact can explain a delta after
// the fact.
//
// This module records it. Every document this process actually opened is
// fingerprinted by CONTENT, kept in a process registry, folded into one
// `repodata_universe` digest, printed on the rows that decide a lock, and
// stamped into the advertised-identity record (which refuses a mismatch, C11's
// discipline).
//
// WHY CONTENT AND NOT ETAG OR MTIME, stated because the campaign has already
// paid for the wrong answer twice:
//   * MTIME+LEN was the pre-v2 `repodata_identity`, and it is unstable across
//     identical bytes: `write_atomic` publishes a refetch as a NEW inode with a
//     NEW mtime whether or not one byte changed, so an unchanged universe read
//     as a moved one and discarded 13 of 14 verdict files (job 5611846).
//   * An ETAG never reaches this layer at all. `fetch_repodata_bytes` fetches
//     `repodata.json.zst`, DECOMPRESSES it, and `write_atomic`s the plaintext:
//     the upstream response headers are dropped on the floor, and the bytes on
//     disk are not the bytes the etag describes.
//   * A content hash of the document on disk is equal exactly when the
//     universe this process can reach is equal, which is the only property the
//     provenance is for. It is the one of the three that is STABLE ACROSS
//     IDENTICAL BYTES.
//
// The hash is not free (the conda-forge linux-64 document is ~640 MB), so it is
// memoized in a sidecar keyed on the stat tuple. The sidecar NEVER decides
// content: a stat mismatch costs a rehash, never a refusal and never an adopted
// value (p6y -- an unsettled stat tuple is not a verdict).

/// Wire tag of the universe fingerprint. Bumped when the folding rule changes
/// meaning; a row or a record carrying a different tag describes a different
/// rule and must not be compared with this one.
pub const UNIVERSE_SCHEMA: &str = "retread-conda-universe-v1";

/// Sidecar wire tag. Separate from [`UNIVERSE_SCHEMA`] because the sidecar is a
/// pure memo of one file's content hash and can change shape without the
/// fingerprint rule changing.
const UNIVERSE_SIDECAR_SCHEMA: &str = "retread-repodata-sha-v1";

/// The FREEZE switch. When set, [`build_sparse`] never refreshes: it consults
/// the document already on disk whatever its age, and a MISSING document is a
/// hard not-consulted with a loud row rather than a silent network fetch.
///
/// This is an environment variable and not an argv flag for one reason: pixi
/// spawns the backend, so the harness owns no argv here. It follows the shape
/// this codebase already uses for exactly that situation --
/// `RETREAD_BUILT_OUTPUT_STORE` / `retread-built-output-store` -- an env var the
/// harness sets, with the pack config key as the supported control.
pub const FROZEN_ENV: &str = "RETREAD_REPODATA_FROZEN";

/// True when the repodata universe is frozen for this process.
pub fn frozen() -> bool {
    std::env::var_os(FROZEN_ENV)
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

/// One repodata document, identified by the bytes that were actually parsed.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RepodataDocument {
    /// Channel URL with no trailing slash, exactly as `sparse()` was called.
    pub channel: String,
    pub subdir: String,
    /// Hex sha256 of the on-disk document.
    pub sha256: String,
    pub bytes: u64,
}

impl RepodataDocument {
    /// `"<channel>/<subdir>"`, the same label `sparse_pairs` returns.
    pub fn label(&self) -> String {
        format!("{}/{}", self.channel, self.subdir)
    }

    /// First 8 hex of the content hash -- what the per-pair list in the summary
    /// row prints.
    pub fn short(&self) -> String {
        self.sha256.chars().take(8).collect()
    }
}

static UNIVERSE: OnceLock<Mutex<std::collections::BTreeMap<(String, String), RepodataDocument>>> =
    OnceLock::new();
static UNIVERSE_SUMMARY: OnceLock<()> = OnceLock::new();

fn universe_registry()
-> &'static Mutex<std::collections::BTreeMap<(String, String), RepodataDocument>> {
    UNIVERSE.get_or_init(|| Mutex::new(std::collections::BTreeMap::new()))
}

/// Fold a set of documents into ONE digest. Pure, so a guard can drive it
/// without a filesystem, and so the `repodata-universe` verb and the backend
/// can never compute two different numbers from one set.
///
/// Order-independent (the input is sorted here, not by the caller) and
/// subdir-separated: `(conda-forge, linux-64, X)` + `(conda-forge, noarch, Y)`
/// must NOT fold to the same value as the same two hashes swapped between
/// subdirs, or a guard cannot tell one moved subdir from another.
pub fn universe_digest_of(documents: &[RepodataDocument]) -> String {
    let mut sorted: Vec<&RepodataDocument> = documents.iter().collect();
    sorted.sort();
    sorted.dedup();
    let mut hasher = Sha256::new();
    hasher.update(UNIVERSE_SCHEMA.as_bytes());
    hasher.update([0u8]);
    for document in sorted {
        hasher.update(document.channel.as_bytes());
        hasher.update([0x1fu8]);
        hasher.update(document.subdir.as_bytes());
        hasher.update([0x1fu8]);
        hasher.update(document.sha256.as_bytes());
        hasher.update([0x1fu8]);
        hasher.update(document.bytes.to_string().as_bytes());
        hasher.update([0u8]);
    }
    let digest = hasher.finalize();
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// Every document this process has consulted, in `(channel, subdir)` order.
pub fn universe_documents() -> Vec<RepodataDocument> {
    universe_registry()
        .lock()
        .unwrap()
        .values()
        .cloned()
        .collect()
}

/// The digest of what this process has consulted so far. An EMPTY registry
/// folds to the digest of the empty set rather than to `""`: "no repodata was
/// consulted" is itself a universe, and a row that prints nothing is a row a
/// grep cannot find.
pub fn universe_digest() -> String {
    universe_digest_of(&universe_documents())
}

/// Sidecar path for one document: `.<filename>.retread-universe-v1.json`,
/// a dotfile so it can never be mistaken for a repodata document by the
/// directory walk in the freeze harness.
fn universe_sidecar_path(document: &std::path::Path) -> Option<PathBuf> {
    let parent = document.parent()?;
    let filename = document.file_name()?.to_str()?;
    Some(parent.join(format!(".{filename}.retread-universe-v1.json")))
}

/// The stat tuple the sidecar memo is keyed on. NOT an identity -- only a
/// "has this file certainly not been rewritten" check that decides whether a
/// rehash can be skipped.
///
/// CTIME IS LOAD-BEARING AND WAS ADDED AFTER THE GUARD CAUGHT ITS ABSENCE.
/// Without it the memo can hand back a hash for bytes that are no longer there:
/// an in-place rewrite of the SAME LENGTH keeps len and ino, and mtime can land
/// in the same recorded tick, so `(len, ino, mtime)` matched across a changed
/// document and the guard
/// `p6ad_one_changed_byte_in_one_subdir_moves_the_universe_digest` went RED with
/// the two hashes equal. `ctime` cannot be set from userspace and moves on every
/// metadata or data write, so it closes that hole. Production's own refresh path
/// (`write_atomic`) publishes a NEW INODE and never needed it -- which is
/// precisely why only a guard, and not a relock, could ever have found this.
///
/// The p6y rule applies and applies the right way round: `ctime` is UNSTABLE
/// (a concurrent `cp -al` into a shared store bumps it under a reader), and here
/// an unstable ctime costs a REHASH and nothing else. It can never produce a
/// verdict, a refusal, or an eviction.
fn stat_key(meta: &std::fs::Metadata) -> String {
    use std::os::unix::fs::MetadataExt as _;
    format!(
        "{}:{}:{}:{}:{}:{}",
        meta.len(),
        meta.ino(),
        meta.mtime(),
        meta.mtime_nsec(),
        meta.ctime(),
        meta.ctime_nsec()
    )
}

/// Stream the file and return `(hex sha256, len)`. Blocking; call it on the
/// blocking pool.
fn hash_file_blocking(path: &std::path::Path) -> Result<(String, u64)> {
    use std::io::Read as _;
    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut total: u64 = 0;
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("reading {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        total += read as u64;
    }
    let digest = hasher.finalize();
    Ok((format!("{digest:x}"), total))
}

/// Content hash of one document, memoized in its sidecar UNDER A FREEZE ONLY.
///
/// MEASURED, node2341, the live shared cache: the conda-forge `linux-64`
/// document is 638 500 357 B and hashes in 4.01 s, `noarch` 253 762 009 B in
/// 1.52 s -- 5.5 s of NFS-bound read per backend process for the two documents
/// that matter, and a relock spawns many processes. That cost is what a memo is
/// for, and it is also why the memo has to be sound.
///
/// IT IS NOT SOUND OUTSIDE A FREEZE, and the guards proved it rather than the
/// docs asserting it. A stat-tuple memo cannot see an in-place rewrite of the
/// SAME LENGTH on every filesystem this campaign runs on: with `(len, ino,
/// mtime)` the guard went RED immediately, and even with `ctime` folded in it
/// went RED intermittently under a loaded parallel run -- a cached attribute is
/// not a fresh one. So the memo is consulted ONLY when [`frozen`] is set, where
/// the snapshot is written once by `retread_freeze_repodata` before the job and
/// nothing -- retread least of all, since a freeze disables every refresh --
/// writes into it again. With no freeze, every call hashes the bytes.
///
/// The p6y rule applies and applies the right way round: the stat tuple can
/// only ever cost a REHASH. It never produces a verdict, a refusal or an
/// eviction, and it can never supply a hash for bytes it does not describe.
fn document_identity_blocking(path: &std::path::Path) -> Result<(String, u64)> {
    let meta = std::fs::metadata(path)
        .with_context(|| format!("stat {} for its content hash", path.display()))?;
    let key = stat_key(&meta);
    let sidecar = universe_sidecar_path(path);
    if frozen()
        && let Some(sidecar) = sidecar.as_ref()
        && let Ok(text) = std::fs::read_to_string(sidecar)
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(&text)
        && value.get("schema").and_then(|v| v.as_str()) == Some(UNIVERSE_SIDECAR_SCHEMA)
        && value.get("stat").and_then(|v| v.as_str()) == Some(key.as_str())
        && let Some(sha) = value.get("sha256").and_then(|v| v.as_str())
        && sha.len() == 64
        && sha.chars().all(|c| c.is_ascii_hexdigit())
    {
        return Ok((sha.to_string(), meta.len()));
    }
    let (sha256, bytes) = hash_file_blocking(path)?;
    // The memo is WRITTEN whether or not this process is frozen: the process
    // that populates a snapshot is usually not the one that later reads it
    // frozen, and a memo nobody may write is a memo nobody can use.
    if let Some(sidecar) = sidecar {
        let body = serde_json::json!({
            "schema": UNIVERSE_SIDECAR_SCHEMA,
            "stat": key,
            "sha256": sha256,
            "bytes": bytes,
        })
        .to_string();
        // Temp + rename in the same directory, so a concurrent reader never
        // sees a half-written memo. Best effort: a memo that cannot be written
        // only costs the next process a rehash.
        let tmp = crate::wheel::unique_atomic_sibling(&sidecar, "part");
        if std::fs::write(&tmp, body.as_bytes()).is_ok() {
            if std::fs::rename(&tmp, &sidecar).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        } else {
            let _ = std::fs::remove_file(&tmp);
        }
    }
    Ok((sha256, bytes))
}

/// Record the document behind one consulted `(channel, subdir)` pair.
///
/// Called from [`build_sparse`] on the success path only: a pair that could not
/// be opened is NOT part of the universe, and printing it as if it were would
/// state a fact the solve never had.
async fn record_document(channel_url: &str, subdir: &str, path: PathBuf) {
    let shown = path.display().to_string();
    let identity = tokio::task::spawn_blocking(move || document_identity_blocking(&path)).await;
    let (sha256, bytes) = match identity {
        Ok(Ok(pair)) => pair,
        Ok(Err(error)) => {
            tracing::warn!(
                channel = %channel_url, subdir = %subdir, path = %shown,
                error = %format!("{error:#}"),
                "repodata: could not fingerprint the document; the universe row will not name it",
            );
            return;
        }
        Err(error) => {
            tracing::warn!(
                channel = %channel_url, subdir = %subdir, path = %shown, error = %error,
                "repodata: fingerprint task failed; the universe row will not name it",
            );
            return;
        }
    };
    let document = RepodataDocument {
        channel: channel_url.to_string(),
        subdir: subdir.to_string(),
        sha256,
        bytes,
    };
    tracing::info!(
        channel = %document.channel,
        subdir = %document.subdir,
        repodata_sha256 = %document.sha256,
        bytes = document.bytes,
        frozen = frozen(),
        "repodata: document fingerprinted",
    );
    universe_registry()
        .lock()
        .unwrap()
        .insert((document.channel.clone(), document.subdir.clone()), document);
}

/// The ONE greppable provenance row for a lock: which channels were consulted
/// and which bytes each one contributed.
///
/// Emitted once per backend process, at the end of the first [`sparse_pairs`]
/// fan-out that yields at least one handle. A relock spawns MANY backend
/// processes, so "once per lock" is only ever achievable as "once per process,
/// and a grep over the lock's backend log yields the set" -- and that is the
/// honest claim. Two rows in one log that disagree is the finding.
pub fn emit_universe_summary() {
    if UNIVERSE_SUMMARY.set(()).is_err() {
        return;
    }
    let documents = universe_documents();
    let channels: Vec<String> = documents.iter().map(RepodataDocument::label).collect();
    let digests: Vec<String> = documents.iter().map(RepodataDocument::short).collect();
    tracing::info!(
        schema = UNIVERSE_SCHEMA,
        digest = %universe_digest_of(&documents),
        pairs = documents.len(),
        frozen = frozen(),
        channels = %format!("[{}]", channels.join(",")),
        digests = %format!("[{}]", digests.join(",")),
        "conda_universe",
    );
}

/// Human-readable form of the summary, for the `repodata-universe` verb. The
/// verb and the backend fold the SAME [`universe_digest_of`], so the harness
/// can print a job header that is comparable with the backend's own rows
/// instead of a second implementation that drifts from it.
pub fn universe_summary_line(documents: &[RepodataDocument]) -> String {
    let channels: Vec<String> = documents.iter().map(RepodataDocument::label).collect();
    let digests: Vec<String> = documents.iter().map(RepodataDocument::short).collect();
    format!(
        "### conda_universe schema={UNIVERSE_SCHEMA} digest={} pairs={} channels=[{}] digests=[{}]",
        universe_digest_of(documents),
        documents.len(),
        channels.join(","),
        digests.join(","),
    )
}

/// Fingerprint every repodata document under `<cache root>/retread-repodata/`
/// WITHOUT solving, fetching or refreshing anything. This is what the
/// `repodata-universe` verb runs, and what a frozen snapshot's job header
/// prints.
pub fn universe_from_cache_root(cache_root: &std::path::Path) -> Result<Vec<RepodataDocument>> {
    let dir = cache_root.join("retread-repodata");
    let mut documents = Vec::new();
    let entries = std::fs::read_dir(&dir)
        .with_context(|| format!("reading repodata cache dir {}", dir.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("walking {}", dir.display()))?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // Dotfiles are the fetch locks and the sha sidecars, never documents.
        if name.starts_with('.') || !name.ends_with(".json") {
            continue;
        }
        // `<slug>--<subdir>--<hex16>.json`, the scheme `disk_cache_path` writes.
        let stem = name.trim_end_matches(".json");
        let parts: Vec<&str> = stem.split("--").collect();
        if parts.len() != 3 {
            continue;
        }
        let (sha256, bytes) = document_identity_blocking(&path)?;
        documents.push(RepodataDocument {
            // MEASURED on the live shared cache: the slug alone COLLIDES --
            // two distinct pytorch channel URLs both render `pytorch` and the
            // verb printed `pytorch/linux-64` twice with different hashes. The
            // filename's hex suffix IS `sha256(channel_url|subdir)[..8]`
            // (`disk_cache_path`), so appending it makes the label injective
            // again without inventing a URL the filename does not carry.
            //
            // The slug is still a lossy rendering of the channel URL: the
            // backend's own rows carry the full URL, the verb carries this.
            // Both fold the SAME rule, so a verb digest is comparable with
            // another VERB digest and a backend digest with another BACKEND
            // digest. Cross-comparing the two is not a defect the digest can
            // catch, and this comment is the warning.
            channel: format!("{}#{}", parts[0], parts[2]),
            subdir: parts[1].to_string(),
            sha256,
            bytes,
        });
    }
    documents.sort();
    Ok(documents)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "retread-repodata-test-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Guard for the 4.10.77 root fix: the published cache file must always
    /// be COMPLETE, and no shared `.part` sibling may linger.
    #[tokio::test]
    async fn write_atomic_publishes_complete_bytes_and_no_shared_part() {
        let dir = unique_tmp_dir("complete");
        let dest = dir.join("repodata.json");
        let payload = vec![b'x'; 1_048_576];
        write_atomic(&dest, &payload).await.unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), payload);
        // The pre-fix shared temp name must never exist: its presence would
        // mean two writers can interleave into one file again.
        assert!(!dest.with_extension("part").exists());
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path() != dest)
            .collect();
        assert!(leftovers.is_empty(), "temp siblings leaked: {leftovers:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Concurrent writers to the SAME destination must each publish a
    /// complete document (last-writer-wins); the final file must be byte-for-
    /// byte equal to ONE of the payloads, never an interleaving. This is the
    /// exact race that truncated the prefix.dev/conda-forge linux-64 cache
    /// and produced "No candidates were found for gcc_linux-64 13.*".
    #[tokio::test]
    async fn concurrent_write_atomic_never_publishes_interleaved_bytes() {
        let dir = unique_tmp_dir("race");
        let dest = dir.join("repodata.json");
        let payloads: Vec<Vec<u8>> = (0..8u8)
            .map(|i| vec![b'a' + i; 512 * 1024 + usize::from(i) * 4096])
            .collect();
        let mut tasks = Vec::new();
        for payload in payloads.clone() {
            let dest = dest.clone();
            tasks.push(tokio::spawn(
                async move { write_atomic(&dest, &payload).await },
            ));
        }
        for task in tasks {
            task.await.unwrap().unwrap();
        }
        let published = std::fs::read(&dest).unwrap();
        assert!(
            payloads.contains(&published),
            "published file matches no single writer's payload (interleaved or truncated)",
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `RATTLER_CACHE_DIR` names the cache ROOT (rattler's own semantics);
    /// empty/unset falls back to `$HOME/.cache/rattler/cache`, then temp.
    #[test]
    fn cache_root_honors_rattler_cache_dir_then_home() {
        use std::ffi::OsStr;
        assert_eq!(
            cache_root_from(
                Some(OsStr::new("/tmp/rcache")),
                Some(OsStr::new("/users/x"))
            ),
            PathBuf::from("/tmp/rcache"),
        );
        assert_eq!(
            cache_root_from(Some(OsStr::new("")), Some(OsStr::new("/users/x"))),
            PathBuf::from("/users/x/.cache/rattler/cache"),
        );
        assert_eq!(
            cache_root_from(None, Some(OsStr::new("/users/x"))),
            PathBuf::from("/users/x/.cache/rattler/cache"),
        );
    }

    /// A corrupt (truncated) cached document must be EVICTED, not silently
    /// dropped from the consulted set: pre-4.10.77 a corrupt file poisoned
    /// its (channel, subdir) pair forever because every refresh failure fell
    /// back to it and every sparse open failed.
    #[tokio::test]
    async fn build_sparse_evicts_corrupt_disk_cache() {
        let channel_url = format!(
            "https://retread-test.invalid/corrupt-evict-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        );
        let path = disk_cache_path(&channel_url, "linux-64");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // Fresh mtime + truncated JSON: exactly the poisoned state observed.
        std::fs::write(&path, br#"{"packages": {"gcc_linux-64-13.4.0-h"#).unwrap();
        let handle = build_sparse(channel_url, "linux-64".to_string()).await;
        // The channel is unreachable (.invalid), so no handle can be built --
        // but the corrupt file must be GONE so the next run can heal.
        assert!(handle.is_none());
        assert!(
            !path.exists(),
            "corrupt cache file survived at {}",
            path.display(),
        );
        if let Some(parent) = path.parent() {
            let filename = path.file_name().unwrap().to_str().unwrap();
            std::fs::remove_file(parent.join(format!(".{filename}.retread-fetch-v1.lock"))).ok();
        }
    }

    // -----------------------------------------------------------------------
    // p6ad GUARDS. Every one of these is mutation-tested; the mutations and
    // which guard each one turns RED are recorded in `p6ad_negctl.sh`.
    // -----------------------------------------------------------------------

    /// `FROZEN_ENV` is process-global; the one test that flips it holds this so
    /// it can never switch another test's production path underneath it.
    static FROZEN_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Run `body` with the freeze on, and put the variable back however `body`
    /// ended.
    fn with_frozen(body: impl FnOnce()) {
        // SAFETY: single-threaded within this test, serialized by
        // FROZEN_TEST_LOCK against the only other writer of this variable.
        unsafe { std::env::set_var(FROZEN_ENV, "1") };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
        unsafe { std::env::remove_var(FROZEN_ENV) };
        if let Err(payload) = result {
            std::panic::resume_unwind(payload);
        }
    }

    fn doc(channel: &str, subdir: &str, sha: &str, bytes: u64) -> RepodataDocument {
        RepodataDocument {
            channel: channel.to_string(),
            subdir: subdir.to_string(),
            sha256: sha.to_string(),
            bytes,
        }
    }

    fn write_doc(dir: &std::path::Path, name: &str, body: &[u8]) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        path
    }

    /// The property the whole design rests on: the fingerprint is CONTENT, so
    /// two copies of the same document at different paths, with different
    /// mtimes and different inodes, are the SAME universe.
    ///
    /// This is the guard that fails the pre-v2 `repodata_identity` rule (length
    /// + mtime), which is exactly what discarded 13 of 14 verdict files in job
    /// 5611846 when nothing upstream had moved.
    #[test]
    fn p6ad_identical_repodata_bytes_are_the_same_universe_whatever_the_mtime_or_path() {
        let a = unique_tmp_dir("p6ad-same-a");
        let b = unique_tmp_dir("p6ad-same-b");
        let body = br#"{"packages":{"zlib-1.3-h1.conda":{"name":"zlib"}}}"#;
        let path_a = write_doc(&a, "conda_forge--linux-64--aaaaaaaaaaaaaaaa.json", body);
        let path_b = write_doc(&b, "conda_forge--linux-64--aaaaaaaaaaaaaaaa.json", body);
        // Two different inodes, and mtimes a second apart on any filesystem
        // that records them -- the two inputs the old rule keyed on.
        std::fs::File::options()
            .write(true)
            .open(&path_b)
            .unwrap()
            .set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000))
            .unwrap();

        let (sha_a, len_a) = document_identity_blocking(&path_a).unwrap();
        let (sha_b, len_b) = document_identity_blocking(&path_b).unwrap();
        assert_eq!(sha_a, sha_b, "identical bytes must fingerprint identically");
        assert_eq!(len_a, len_b);
        assert_eq!(
            universe_digest_of(&[doc("https://c/conda-forge", "linux-64", &sha_a, len_a)]),
            universe_digest_of(&[doc("https://c/conda-forge", "linux-64", &sha_b, len_b)]),
        );
        std::fs::remove_dir_all(&a).ok();
        std::fs::remove_dir_all(&b).ok();
    }

    /// One byte in ONE subdir moves the aggregate digest, and does NOT move the
    /// untouched subdir's own digest. Both halves matter: the first is the
    /// detection, the second is what makes the row diagnostic instead of a
    /// single opaque bit.
    #[test]
    fn p6ad_one_changed_byte_in_one_subdir_moves_the_universe_digest() {
        let dir = unique_tmp_dir("p6ad-onebyte");
        let linux = write_doc(&dir, "conda_forge--linux-64--a.json", br#"{"packages":{"a":1}}"#);
        let noarch_before = write_doc(&dir, "conda_forge--noarch--b.json", br#"{"packages":{"b":1}}"#);

        let (linux_sha, linux_len) = document_identity_blocking(&linux).unwrap();
        let (noarch_sha, noarch_len) = document_identity_blocking(&noarch_before).unwrap();
        let before = universe_digest_of(&[
            doc("https://c/conda-forge", "linux-64", &linux_sha, linux_len),
            doc("https://c/conda-forge", "noarch", &noarch_sha, noarch_len),
        ]);

        // One byte, in noarch only. The sidecar must NOT be believed here --
        // it was written moments ago for the old bytes.
        std::fs::write(&noarch_before, br#"{"packages":{"b":2}}"#).unwrap();
        let (noarch_sha2, noarch_len2) = document_identity_blocking(&noarch_before).unwrap();
        assert_ne!(noarch_sha, noarch_sha2, "the changed document must rehash");
        let (linux_sha2, linux_len2) = document_identity_blocking(&linux).unwrap();
        assert_eq!(linux_sha, linux_sha2, "the untouched subdir must not move");

        let after = universe_digest_of(&[
            doc("https://c/conda-forge", "linux-64", &linux_sha2, linux_len2),
            doc("https://c/conda-forge", "noarch", &noarch_sha2, noarch_len2),
        ]);
        assert_ne!(before, after, "one changed subdir must move the universe");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The digest separates SUBDIRS. Swapping two documents between linux-64
    /// and noarch is a different universe, and a rule that concatenated the
    /// hashes without their pair labels would call it the same one.
    #[test]
    fn p6ad_the_universe_digest_is_keyed_per_channel_and_subdir_not_on_a_bag_of_hashes() {
        let straight = [
            doc("https://c/conda-forge", "linux-64", &"a".repeat(64), 10),
            doc("https://c/conda-forge", "noarch", &"b".repeat(64), 20),
        ];
        let swapped = [
            doc("https://c/conda-forge", "linux-64", &"b".repeat(64), 20),
            doc("https://c/conda-forge", "noarch", &"a".repeat(64), 10),
        ];
        assert_ne!(
            universe_digest_of(&straight),
            universe_digest_of(&swapped),
            "two subdirs swapping documents is a DIFFERENT universe"
        );
        // A second channel contributing the same subdir is also distinct.
        let other_channel = [
            doc("https://c/conda-forge", "linux-64", &"a".repeat(64), 10),
            doc("https://c/nvidia", "noarch", &"b".repeat(64), 20),
        ];
        assert_ne!(universe_digest_of(&straight), universe_digest_of(&other_channel));
        // And the fold is order-independent: the caller's ordering is not part
        // of the answer.
        let mut reversed = straight.to_vec();
        reversed.reverse();
        assert_eq!(universe_digest_of(&straight), universe_digest_of(&reversed));
    }

    /// The memo is consulted ONLY under a freeze, and even then it is a memo and
    /// never an authority.
    ///
    /// Both halves are load-bearing and both were written because a guard went
    /// RED, not because the design predicted it: with no freeze a stat-tuple
    /// memo cannot see an in-place same-length rewrite (it read the old hash for
    /// the new bytes, twice, once with `ctime` folded in), so with no freeze
    /// there is no memo path at all. Under a freeze a sidecar whose stat tuple
    /// does not describe the file is still ignored and the content re-hashed.
    #[test]
    fn p6ad_the_memo_is_consulted_only_under_a_freeze_and_never_as_an_authority() {
        let _guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = unique_tmp_dir("p6ad-sidecar");
        let path = write_doc(&dir, "conda_forge--noarch--c.json", br#"{"packages":{"c":1}}"#);
        let truth = document_identity_blocking(&path).unwrap().0;

        let sidecar = universe_sidecar_path(&path).unwrap();

        // NON-VACUITY FIRST: under a freeze the memo is genuinely consulted. A
        // sidecar carrying a sentinel hash for the file's OWN stat tuple comes
        // back verbatim -- so the "ignored" assertions below are about a live
        // path, not a dead one.
        let sentinel = "a".repeat(64);
        let stamp = |sha: &str| {
            serde_json::json!({
                "schema": UNIVERSE_SIDECAR_SCHEMA,
                "stat": stat_key(&std::fs::metadata(&path).unwrap()),
                "sha256": sha,
                "bytes": 0,
            })
            .to_string()
        };
        std::fs::write(&sidecar, stamp(&sentinel)).unwrap();
        with_frozen(|| {
            assert_eq!(
                document_identity_blocking(&path).unwrap().0,
                sentinel,
                "under a freeze the memo must actually be consulted"
            );
        });
        // And with NO freeze the same sidecar is inert: the bytes decide.
        assert_eq!(
            document_identity_blocking(&path).unwrap().0,
            truth,
            "with no freeze the memo must not be consulted at all"
        );

        // A sidecar claiming a different hash for a stat tuple that does not
        // match: the classic poisoned memo, refused even under a freeze.
        std::fs::write(
            &sidecar,
            serde_json::json!({
                "schema": UNIVERSE_SIDECAR_SCHEMA,
                "stat": "0:0:0:0:0:0",
                "sha256": "f".repeat(64),
                "bytes": 0,
            })
            .to_string(),
        )
        .unwrap();
        with_frozen(|| {
            assert_eq!(
                document_identity_blocking(&path).unwrap().0,
                truth,
                "a sidecar whose stat tuple does not match must not supply the hash"
            );
        });

        // A sidecar from another schema is equally inert under a freeze.
        std::fs::write(
            &sidecar,
            serde_json::json!({
                "schema": "retread-repodata-sha-v0",
                "stat": stat_key(&std::fs::metadata(&path).unwrap()),
                "sha256": "e".repeat(64),
            })
            .to_string(),
        )
        .unwrap();
        with_frozen(|| assert_eq!(document_identity_blocking(&path).unwrap().0, truth));

        // Garbage is inert too, and never fatal.
        std::fs::write(&sidecar, b"not json at all").unwrap();
        with_frozen(|| assert_eq!(document_identity_blocking(&path).unwrap().0, truth));
        assert_eq!(document_identity_blocking(&path).unwrap().0, truth);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The verb reads the snapshot on disk and nothing else: dotfiles (the
    /// fetch locks and the sha sidecars) are not documents, and the summary
    /// line names every pair it found with the digest it folded.
    #[test]
    fn p6ad_the_summary_names_every_document_the_snapshot_actually_holds() {
        let root = unique_tmp_dir("p6ad-verb");
        let dir = root.join("retread-repodata");
        std::fs::create_dir_all(&dir).unwrap();
        write_doc(&dir, "conda_forge--linux-64--aa.json", br#"{"packages":{"a":1}}"#);
        write_doc(&dir, "conda_forge--noarch--bb.json", br#"{"packages":{"b":1}}"#);
        // Neither of these is a document.
        write_doc(&dir, ".conda_forge--linux-64--aa.json.retread-fetch-v1.lock", b"");
        write_doc(&dir, "not-a-repodata-name.json", b"{}");

        let documents = universe_from_cache_root(&root).unwrap();
        assert_eq!(
            documents.iter().map(RepodataDocument::label).collect::<Vec<_>>(),
            vec!["conda_forge#aa/linux-64", "conda_forge#bb/noarch"],
            "the summary must name exactly the documents on disk"
        );
        let line = universe_summary_line(&documents);
        assert!(line.contains(&format!("digest={}", universe_digest_of(&documents))), "{line}");
        assert!(line.contains("pairs=2"), "{line}");
        assert!(line.contains("channels=[conda_forge#aa/linux-64,conda_forge#bb/noarch]"), "{line}");
        // The suffix is what keeps two channels that slug to one name apart --
        // measured on the live cache, where two pytorch URLs both render
        // `pytorch` and collided into one label before this.
        write_doc(&dir, "pytorch--linux-64--11.json", br#"{"packages":{"p":1}}"#);
        write_doc(&dir, "pytorch--linux-64--22.json", br#"{"packages":{"p":2}}"#);
        let with_collision = universe_from_cache_root(&root).unwrap();
        let labels: Vec<String> = with_collision.iter().map(RepodataDocument::label).collect();
        assert_eq!(
            labels.iter().collect::<std::collections::HashSet<_>>().len(),
            labels.len(),
            "two channels that slug to one name must still have distinct labels: {labels:?}"
        );
        assert!(line.contains(UNIVERSE_SCHEMA), "{line}");
        for document in &documents {
            assert!(line.contains(&document.short()), "{line} is missing {}", document.short());
        }

        // And the digest tracks the snapshot: change one document, the line moves.
        let before = universe_digest_of(&documents);
        std::fs::write(dir.join("conda_forge--noarch--bb.json"), br#"{"packages":{"b":2}}"#).unwrap();
        let after = universe_digest_of(&universe_from_cache_root(&root).unwrap());
        assert_ne!(before, after);
        std::fs::remove_dir_all(&root).ok();
    }

    /// An EMPTY consulted set is still a universe with a name. A row that
    /// printed nothing here would be a row no grep can find, and "no repodata
    /// was consulted" is precisely the state a reader most needs named.
    #[test]
    fn p6ad_an_empty_universe_still_has_a_digest() {
        let empty = universe_digest_of(&[]);
        assert_eq!(empty.len(), 16, "{empty}");
        assert_ne!(
            empty,
            universe_digest_of(&[doc("https://c/conda-forge", "linux-64", &"a".repeat(64), 1)]),
        );
    }
}

/// Drop every cached view of `(channel_url, subdir)` so the next
/// [`sparse`] call refetches the document from the network.
///
/// Two layers hold repodata: the per-process `SPARSE_CACHE` memo (built at
/// most once per process, by design) and the on-disk document under
/// `<cache root>/retread-repodata/`, fresh for [`REPODATA_TTL`]. Both must
/// go, or a re-solve inside the same process returns exactly the snapshot
/// that was just proved stale.
///
/// This exists because retread and `rattler-build` resolve the SAME hermetic
/// recipe against TWO different snapshots: retread pins exact
/// `name ==version build` triples out of this cache (up to `REPODATA_TTL`
/// old), and `rattler-build` re-resolves those pins against a channel it
/// fetches into a cold private scratch on every single provision, i.e.
/// always current. When upstream supersedes a build inside that window the
/// recipe is unsatisfiable and rattler-build reports only "No candidates
/// were found for <our own pin>". Forcing a refetch here makes the two
/// snapshots converge instead of dead-ending the build.
///
/// Returns `true` when a disk document was actually removed.
pub(crate) async fn invalidate(channel_url: &str, subdir: &str) -> bool {
    if let Some(map) = SPARSE_CACHE.get() {
        map.lock()
            .unwrap()
            .remove(&(channel_url.to_string(), subdir.to_string()));
    }
    let path = disk_cache_path(channel_url, subdir);
    match tokio::fs::remove_file(&path).await {
        Ok(()) => {
            tracing::warn!(
                channel = %channel_url,
                subdir = %subdir,
                cache = %path.display(),
                "repodata: dropped the cached snapshot; the next solve refetches",
            );
            true
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => {
            tracing::warn!(
                cache = %path.display(), error = %error,
                "repodata: could not drop the cached snapshot",
            );
            false
        }
    }
}

/// Drop the cached snapshot of every `(channel, subdir)` pair in `pairs`.
pub(crate) async fn invalidate_pairs(pairs: &[(String, String)]) -> bool {
    let mut dropped = false;
    for (channel_url, subdir) in pairs {
        dropped |= invalidate(channel_url, subdir).await;
    }
    dropped

}
