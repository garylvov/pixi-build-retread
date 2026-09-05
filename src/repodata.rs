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
        // p6ad-6: prime off the reactor before the summary formats the digest.
        prime_universe_digest().await;
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
                // p6ad-6: the snapshot fold is seconds of NFS read on a cold memo
                // and this is a tokio worker thread. Prime it on the blocking
                // pool, so `universe_digest()` below is a memo hit and no row's
                // formatting can park the reactor.
                let universe = prime_universe_digest().await;
                tracing::info!(
                    channel = %channel_url,
                    subdir = %subdir,
                    elapsed_ms = t.elapsed().as_millis() as u64,
                    repodata_universe = %universe,
                    consulted_digest = %consulted_digest(),
                    "bench: sparse repodata handle built",
                );
                return Some(handle);
            }
            None if attempt == 0 && path.exists() && !frozen() => {
                tracing::warn!(
                    channel = %channel_url, subdir = %subdir, path = %path.display(),
                    "repodata: evicting corrupt disk cache and refetching",
                );
                // p6ad-6: an eviction removes a document from the snapshot, so
                // the memoized universe digest no longer names what is on disk.
                // p6ad-4-3: the eviction touches THIS path and no other, so the
                // other nine documents keep their memoized hashes.
                invalidate_snapshot_digest(&path);
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
    write_atomic(path, &bytes).await?;
    // p6ad-6: this process just changed its own snapshot. The memoized universe
    // digest names the bytes that were there BEFORE this write, so it is dropped
    // by the writer and never left to age out.
    // p6ad-4-3: `write_atomic` replaced exactly `path` and published a NEW
    // inode for it, so only `path`'s memo entries are dropped -- the other nine
    // documents in the snapshot were not touched by this write and their
    // hashes, which carry 99.2 % of the bytes, survive the refresh.
    invalidate_snapshot_digest(path);
    Ok(())
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
///
/// v1 -> v2 (p6ad-6): the fold is unchanged, the INPUT SET is not. v1 folded
/// the documents one process had consulted so far; v2 folds every document in
/// the cache-root snapshot. A v1 row and a v2 row are not comparable and the
/// tag is what says so.
pub const UNIVERSE_SCHEMA: &str = "retread-conda-universe-v2";

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

/// The digest of what THIS PROCESS has consulted so far. An EMPTY registry
/// folds to the digest of the empty set rather than to `""`: "no repodata was
/// consulted" is itself a universe, and a row that prints nothing is a row a
/// grep cannot find.
///
/// p6ad-6: THIS IS NOT THE LOCK'S UNIVERSE AND MUST NOT BE PRINTED AS IF IT
/// WERE. It is the universe INTERSECTED with one process's channel needs, at
/// one instant in one fan-out. See [`universe_digest`].
pub fn consulted_digest() -> String {
    universe_digest_of(&universe_documents())
}

/// The token a row carries when the snapshot could not be read at all. It is
/// deliberately not a hex digest: a grep for a digest can never match it, and
/// nobody can mistake it for "the empty universe", which is a real state with
/// a real digest.
pub const UNIVERSE_UNAVAILABLE: &str = "unavailable";

/// Memo of the snapshot digest, KEYED ON THE CACHE ROOT it was computed from,
/// so a guard driving a temp root can never poison production's value and vice
/// versa. `None` = never computed, or invalidated by a write to the snapshot.
static SNAPSHOT_DIGEST: OnceLock<Mutex<Option<(PathBuf, String)>>> = OnceLock::new();

fn snapshot_digest_memo() -> &'static Mutex<Option<(PathBuf, String)>> {
    SNAPSHOT_DIGEST.get_or_init(|| Mutex::new(None))
}

/// Drop the memoized snapshot digest.
///
/// Called on every successful `refresh_disk_cache` and on the corrupt-cache
/// eviction, which are the only two ways a document under this process's cache
/// root changes while this process is alive. A memo that outlived the bytes it
/// names would be the p6y failure in a new costume -- a stale key producing a
/// VERDICT -- so it is cleared by the writer, not trusted to age out.
///
/// p6ad-4: it drops the per-document content memo too. The two memos name the
/// same bytes at two granularities -- `DOCUMENT_IDENTITY` holds each document's
/// sha256, `SNAPSHOT_DIGEST` holds the fold of all of them -- so a writer that
/// cleared only the fold would leave the very hash the next fold reads for the
/// bytes it just replaced. One writer, one invalidation, both memos.
///
/// p6ad-4-3: `changed` IS THE DOCUMENT THE CALLER JUST REPLACED, and only that
/// document's entries leave the per-document memo. p6ad-4 `clear()`ed the whole
/// map because that made the invalidation obviously correct, and MEASURED the
/// price on its own arm: 374 memo hits against 222 misses is 62.8 % by COUNT
/// and **12.05 %** by BYTES (7.145 GB served of 59.286 GB attempted), because
/// the documents a refresh replaces are precisely conda-forge's `linux-64`
/// (638 520 052 B) and `noarch` (253 824 887 B) -- 892 of 899 MB, 99.2 % of the
/// bytes -- so every whole-map clear threw away exactly the entries worth
/// keeping.
///
/// WHY SCOPING IT IS STILL SOUND, and it is soundER, not weaker:
///
///   * The two production writers each replace exactly ONE document and know
///     its path (`refresh_disk_cache`'s `write_atomic` target, and the
///     corrupt-cache eviction's `path`). Nothing else under this root changes
///     because of them.
///   * A document changed by ANOTHER process was never covered by the
///     `clear()` either -- no writer in this process runs -- and it is covered
///     the way it always was, by `stat_key`'s ctime term (p6k-b/p6y): a
///     same-length in-place rewrite moves `ctime`, which cannot be set from
///     userspace, so the key misses and the bytes are read.
///   * Removal is BY PATH AND EVERY STAT TUPLE, not by the key the caller could
///     compute: `write_atomic` publishes a new inode, so at the moment the
///     writer calls this the entry it must kill is the one under the OLD stat
///     tuple, which is no longer derivable from the file on disk.
///
/// `SNAPSHOT_DIGEST` -- the fold of all ten documents -- is still dropped
/// WHOLESALE, because one changed document changes the fold. Only which
/// documents have to be re-hashed to rebuild that fold changes here; the fold
/// itself is untouched, so the universe digest is byte-identical to what the
/// whole-map clear produced. That is
/// `p6ad43_only_the_replaced_document_is_rehashed`, which asserts the tally AND
/// the equality with a full re-hash in one test. The other two guards are
/// `p6ad43_invalidation_is_scoped_to_the_replaced_document` (the scoping, with
/// no byte changing on disk) and
/// `p6ad43_a_same_length_rewrite_with_preserved_mtime_still_invalidates` (the
/// ctime term, with its own mutation control).
pub fn invalidate_snapshot_digest(changed: &std::path::Path) {
    *snapshot_digest_memo().lock().unwrap() = None;
    document_identity_memo()
        .lock()
        .unwrap()
        .retain(|(path, _stat_tuple), _| path.as_path() != changed);
}

/// Test-only: drop BOTH memos in their entirety.
///
/// p6ad-4-3 removed the whole-map clear from the production invalidator, and
/// two guards need it back for a reason production never has: they drive
/// several temp roots in ONE process and have to start from a memo that carries
/// nothing from a previous test. It is `#[cfg(test)]` on purpose -- a
/// production caller that wants this wants the scoped invalidator instead, and
/// a whole-map clear with a production call site would put p6ad-4-3's 99.2 %
/// straight back on the floor.
#[cfg(test)]
pub fn reset_snapshot_memos() {
    *snapshot_digest_memo().lock().unwrap() = None;
    document_identity_memo().lock().unwrap().clear();
}

/// **The conda repodata universe this lock was resolved against.**
///
/// p6ad-6 ROOT FIX. Until this commit the field printed on every lock-deciding
/// row was [`consulted_digest`] -- the fold of the process-local registry --
/// and MEASURED on the first canonical relock that ever carried it
/// (`mBH-relock` 5851226, 269 rows) that produced **20 distinct digests inside
/// ONE relock off ONE unmoved snapshot**. Two inputs folded in, neither of them
/// the channel:
///
///   1. THE PER-PROCESS CHANNEL SUBSET. 14 backend processes consulted three
///      different nested channel sets (4, 8 and 10 pairs), so the three digests
///      on the lock-deciding rows were three subsets of one snapshot. Every
///      one of the 10 `(channel, subdir)` pairs had exactly ONE sha256 across
///      all 76 `document fingerprinted` rows -- the universe never moved.
///   2. THE MID-FAN-OUT PREFIX. `build_sparse` logged the digest immediately
///      after `record_document`, i.e. from inside the fan-out, so it folded
///      whatever prefix of that process's own pairs had landed at that instant.
///      Two rows 48 ms apart, same process, no document changed, different
///      digest. That input alone invented 17 of the 20.
///
/// A number that moves when nothing moved cannot adjudicate a lock delta, which
/// is the one job this field has. So the digest is now a function of the
/// SNAPSHOT ON DISK under this process's cache root and of nothing else: the
/// same rule [`universe_from_cache_root`] and the `repodata-universe` verb
/// already fold, which additionally makes a backend row and a verb header
/// COMPARABLE for the first time (the warning in `universe_from_cache_root`
/// about cross-comparing the two is retired by this commit).
///
/// COST, and why this is not the regression it looks like: the snapshot fold
/// hashes every document in the cache root, not just the ones this process
/// opened. On the measured production shape that is 10 documents / 899 MB
/// against the 892 MB (`conda-forge` linux-64 + noarch) that EVERY process
/// already hashed -- 14 x 899 MB vs 14 x 892 MB, a 0.8 % change. Under a freeze
/// the sidecar memo makes each document a stat, and the root is job-scoped and
/// holds exactly the job's documents.
///
/// THE PRICE, BOARDED AS p6ad-6-1 AND NOT HIDDEN: on the SHARED cache root the
/// digest now also moves when an unrelated lane drops an unrelated document in.
/// That is a false positive this rule cannot avoid and a freeze
/// (`RETREAD_REPODATA_FROZEN` + a job-scoped root, p6ad / p6af) removes.
pub fn universe_digest() -> String {
    universe_digest_at(&dirs_cache_root())
}

/// Compute [`universe_digest`] ON THE BLOCKING POOL and return it.
///
/// The snapshot fold is an NFS read of every document in the cache root -- 899
/// MB and ~5.5 s on the measured production shape when the memo is cold and the
/// sidecars are absent. `universe_digest()` is called from inside `tracing`
/// macro arguments on tokio worker threads, and doing that read there parks the
/// reactor for the whole fan-out. Every async caller goes through this instead;
/// the synchronous entry point stays for the verb, the tests and the identity
/// record, which are not on the reactor.
pub async fn prime_universe_digest() -> String {
    let root = dirs_cache_root();
    match tokio::task::spawn_blocking(move || universe_digest_at(&root)).await {
        Ok(digest) => digest,
        Err(error) => {
            tracing::warn!(
                error = %error,
                "repodata: universe snapshot task failed; rows will say unavailable",
            );
            UNIVERSE_UNAVAILABLE.to_string()
        }
    }
}

/// [`universe_digest`] against an explicit cache root. Production always passes
/// [`dirs_cache_root`]; the guards pass a temp root, so no test has to reach
/// for `set_var` on a process-global (the idiom `cache_root_from` already sets
/// in this module).
///
/// Two threads racing a cold memo both compute, and both compute the same
/// value from the same bytes -- the lock is deliberately NOT held across the
/// hashing, because a re-entrant `lock()` on a `std::sync::Mutex` is a deadlock
/// and seconds of NFS read under a held global lock is worse than a duplicated
/// hash.
pub fn universe_digest_at(cache_root: &std::path::Path) -> String {
    {
        // Scoped so the guard is DEFINITELY dropped before the recompute
        // below: `universe_from_cache_root` is seconds of NFS read, and a
        // global lock held across it would serialise every backend task.
        let memo = snapshot_digest_memo().lock().unwrap();
        if let Some((root, digest)) = memo.as_ref()
            && root.as_path() == cache_root
        {
            return digest.clone();
        }
    }
    // p6ad-4: the PRODUCTION recompute folds through the per-document memo, so
    // a document `record_document` already hashed in this process is not
    // streamed through `Sha256` a second time here. `snapshot_digest_at` below
    // stays memo-free on purpose -- see its doc comment.
    let digest = snapshot_digest_memoized_at(cache_root);
    *snapshot_digest_memo().lock().unwrap() = Some((cache_root.to_path_buf(), digest.clone()));
    digest
}

/// The fold itself, with NO memo of any kind: read the snapshot under
/// `cache_root`, HASH EVERY DOCUMENT, and return its digest, or
/// [`UNIVERSE_UNAVAILABLE`] if it cannot be read at all.
///
/// Separate from [`universe_digest_at`] so the guards can assert the RULE
/// without touching a process-global memo -- a global that two parallel tests
/// would race, and a flaky guard is a guard that cannot fail for the right
/// reason.
///
/// p6ad-4 EXTENDS THAT CONTRACT RATHER THAN BREAKING IT. This function now also
/// bypasses the per-document memo, because several guards rewrite a document
/// IN PLACE and re-fold with no invalidation call -- exactly the shape a stat
/// tuple is not guaranteed to see. Production goes through
/// [`snapshot_digest_memoized_at`]; the rule is asserted here, on the bytes.
pub fn snapshot_digest_at(cache_root: &std::path::Path) -> String {
    match universe_from_cache_root(cache_root) {
        Ok(documents) => universe_digest_of(&documents),
        Err(error) => {
            tracing::warn!(
                cache_root = %cache_root.display(),
                error = %format!("{error:#}"),
                "repodata: the universe snapshot could not be read; rows will say unavailable",
            );
            UNIVERSE_UNAVAILABLE.to_string()
        }
    }
}

/// p6ad-4: [`snapshot_digest_at`]'s rule, folded through the per-document memo.
///
/// The ONE production entry to the fold. It returns the same value
/// [`snapshot_digest_at`] returns for the same bytes -- that is
/// `p6ad4_the_memo_does_not_move_the_digest`, asserted on a fixture -- while
/// hashing each document at most once per process instead of once per caller.
pub fn snapshot_digest_memoized_at(cache_root: &std::path::Path) -> String {
    match universe_from_cache_root_inner(cache_root, document_identity_blocking) {
        Ok(documents) => universe_digest_of(&documents),
        Err(error) => {
            tracing::warn!(
                cache_root = %cache_root.display(),
                error = %format!("{error:#}"),
                "repodata: the universe snapshot could not be read; rows will say unavailable",
            );
            UNIVERSE_UNAVAILABLE.to_string()
        }
    }
}

/// Sidecar path for one document: `.<filename>.retread-universe-v1.json`,
/// a dotfile so it can never be mistaken for a repodata document by the
/// directory walk in the freeze harness.
fn universe_sidecar_path(document: &std::path::Path) -> Option<PathBuf> {
    let parent = document.parent()?;
    let filename = document.file_name()?.to_str()?;
    Some(parent.join(format!(".{filename}.retread-universe-v1.json")))
}


/// p6ad-4: the per-process content-hash memo, keyed on `(path, stat tuple)`.
///
/// MEASURED, `P6AF2G-5872517-N.backend.log` (a cold canonical relock, arm N,
/// unfrozen by construction): a repodata document is hashed TWICE by every
/// backend process that consults it, because two callers reach
/// [`document_identity_blocking`] with the same path --
/// [`record_document`] (once per consulted pair, the 76 `document
/// fingerprinted` rows, 12.52 GB in aggregate) and
/// [`universe_from_cache_root`] under the snapshot fold (all 10 documents,
/// 899 198 366 B per process, 12.59 GB in aggregate, and it emits no row at
/// all). 25.11 GB per relock, exactly half of it a re-read of bytes this same
/// process has already hashed. The whole repodata phase is 45.9 s of that
/// run's 1646 s wall, and the decisive row pair is a 52 669 B document whose
/// `bench: sparse repodata handle built` row lands 34.4 s after its own
/// fingerprint row with nothing between them but `prime_universe_digest()`.
///
/// WHY THIS IS NOT THE SIDECAR, AND WHY p6ad's REFUSAL STANDS. The sidecar
/// (`.<file>.retread-universe-v1.json`) is a CROSS-PROCESS, CROSS-JOB store on
/// a shared NFS root, and p6ad's guards MEASURED it unsound outside a freeze:
/// RED with `(len, ino, mtime)`, RED intermittently under a loaded parallel
/// run even with `ctime`, because a cached NFS attribute is not a fresh one.
/// That refusal is untouched -- the sidecar is still consulted only under
/// [`frozen`]. This memo is strictly narrower in every direction:
///
///   * it lives in ONE process's address space for that process's lifetime,
///     never on disk, so no other job's writer can populate it;
///   * it is keyed on the SAME [`stat_key`] tuple, so a document rewritten
///     under this process misses and is rehashed;
///   * it is cleared by [`invalidate_snapshot_digest`], i.e. by exactly the
///     two writers -- a successful `refresh_disk_cache` and the corrupt-cache
///     eviction -- that already clear [`SNAPSHOT_DIGEST`];
///   * and its exposure is therefore STRICTLY SMALLER than the memo already in
///     production: `SNAPSHOT_DIGEST` caches the FOLD of all ten documents for
///     the whole process lifetime keyed on the cache root ALONE, with no stat
///     check of any kind. Anything this memo could get wrong, that one already
///     gets wrong first and more coarsely.
///
/// The p6y rule applies the right way round, as it did for [`stat_key`]: an
/// unstable stat costs a REHASH and nothing else. It can never produce a
/// verdict, a refusal or an eviction, and it can never supply a hash for bytes
/// whose stat tuple it does not describe.
///
/// THE DIGEST DOES NOT MOVE. [`universe_digest_of`] folds
/// `(channel, subdir, sha256, bytes)`, and a memo hit returns the same
/// `(sha256, bytes)` the rehash would have computed from the same bytes. The
/// value is byte-identical by construction, and
/// `p6ad4_the_memo_does_not_move_the_digest` asserts it on a fixture.
static DOCUMENT_IDENTITY: OnceLock<Mutex<HashMap<(PathBuf, String), (String, u64)>>> =
    OnceLock::new();

fn document_identity_memo() -> &'static Mutex<HashMap<(PathBuf, String), (String, u64)>> {
    DOCUMENT_IDENTITY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Bytes this process actually streamed through `Sha256`, and bytes it served
/// from the memo instead. The READER for the fix: without these two counters
/// the saving is invisible in a production log, and p6ad-4 was boarded for a
/// year precisely because the silent half of the hashing emitted no row.
static HASHED_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MEMOIZED_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HASH_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MEMO_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SIDECAR_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(hashes, memo hits, sidecar hits, bytes hashed, bytes served from the
/// memo)` for this process. Public so a guard can assert the count rather than
/// grep a log for it.
pub fn document_identity_counters() -> (u64, u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        HASH_CALLS.load(Relaxed),
        MEMO_HITS.load(Relaxed),
        SIDECAR_HITS.load(Relaxed),
        HASHED_BYTES.load(Relaxed),
        MEMOIZED_BYTES.load(Relaxed),
    )
}

/// Test-only: what the memo holds for `path` AT ITS CURRENT STAT TUPLE.
#[cfg(test)]
pub fn document_identity_memo_get(path: &std::path::Path) -> Option<(String, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    document_identity_memo()
        .lock()
        .unwrap()
        .get(&(path.to_path_buf(), stat_key(&meta)))
        .cloned()
}

/// Test-only: how many entries the memo holds for `path` ACROSS ALL STAT
/// TUPLES.
///
/// p6ad-4-3's invalidator removes by PATH, not by the key the caller could
/// compute, and the reason is visible only through this reader: after an
/// in-place rewrite the entry that must die is filed under the OLD stat tuple,
/// which no longer exists on disk, so `document_identity_memo_get` (which stats
/// the file) cannot see it and cannot prove it was removed.
#[cfg(test)]
pub fn document_identity_memo_entries_for(path: &std::path::Path) -> usize {
    document_identity_memo()
        .lock()
        .unwrap()
        .keys()
        .filter(|(p, _)| p.as_path() == path)
        .count()
}

/// Test-only: put a SENTINEL hash in the memo for `path`'s current stat tuple.
///
/// This is how the p6ad-4 guards prove a caller consulted the memo instead of
/// re-reading the file, and it is the same idiom
/// `p6ad_the_memo_is_consulted_only_under_a_freeze_and_never_as_an_authority`
/// already uses for the sidecar. It is deliberately NOT a counter assertion:
/// the counters are process-global, `cargo test` runs the suite in one process,
/// and a guard whose value another test can move is a guard that cannot fail
/// for the right reason.
#[cfg(test)]
pub fn document_identity_memo_poison(path: &std::path::Path, sha256: &str, bytes: u64) {
    let meta = std::fs::metadata(path).unwrap();
    document_identity_memo()
        .lock()
        .unwrap()
        .insert((path.to_path_buf(), stat_key(&meta)), (sha256.to_string(), bytes));
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
/// p6ad-4: THE PRODUCTION ENTRY POINT — [`document_identity_uncached`] behind
/// the per-process memo described on [`DOCUMENT_IDENTITY`].
///
/// The split follows the idiom this module already uses for
/// [`snapshot_digest_at`] versus [`universe_digest_at`]: the RULE lives in a
/// function no process-global can shadow, so the guards that assert the rule
/// (the sidecar's freeze gating, the one-changed-byte detection) drive the
/// uncached function and can never go intermittently RED because of a memo they
/// are not testing. The memo's own three guards drive THIS function.
fn document_identity_blocking(path: &std::path::Path) -> Result<(String, u64)> {
    let meta = std::fs::metadata(path)
        .with_context(|| format!("stat {} for its content hash", path.display()))?;
    let memo_key = (path.to_path_buf(), stat_key(&meta));
    if let Some((sha256, bytes)) = document_identity_memo()
        .lock()
        .unwrap()
        .get(&memo_key)
        .cloned()
    {
        MEMO_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        MEMOIZED_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        return Ok((sha256, bytes));
    }
    let (sha256, bytes) = document_identity_uncached(path)?;
    // Insert under the key computed from the stat taken BEFORE the read. If the
    // file was rewritten between the two, the next caller's stat differs from
    // this key and misses -- the entry is dead weight, never a wrong answer.
    document_identity_memo()
        .lock()
        .unwrap()
        .insert(memo_key, (sha256.clone(), bytes));
    Ok((sha256, bytes))
}

/// The rule itself, with NO per-process memo: stat, consult the sidecar under a
/// freeze only, otherwise hash the bytes, and write the sidecar either way.
fn document_identity_uncached(path: &std::path::Path) -> Result<(String, u64)> {
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
        SIDECAR_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok((sha.to_string(), meta.len()));
    }
    let (sha256, bytes) = hash_file_blocking(path)?;
    HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    HASHED_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
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
        digest = %universe_digest(),
        consulted_digest = %universe_digest_of(&documents),
        pairs = documents.len(),
        frozen = frozen(),
        channels = %format!("[{}]", channels.join(",")),
        digests = %format!("[{}]", digests.join(",")),
        "conda_universe",
    );
    // p6ad-4 THE READER. Without this row the cost of fingerprinting is
    // invisible in a production log: the snapshot fold emits nothing at all, so
    // half of a cold relock's 25 GB of hashing had no writer and no reader for
    // the whole campaign. `bytes_hashed` is what this process actually streamed
    // through Sha256; `bytes_memoized` is what the p6ad-4 memo served instead;
    // `sidecar_hits` is the frozen path. Summing `bytes_hashed` over a relock's
    // backend log is the aggregate p6ad-4 was boarded to measure.
    let (hashes, memo_hits, sidecar_hits, bytes_hashed, bytes_memoized) =
        document_identity_counters();
    tracing::info!(
        hashes,
        memo_hits,
        sidecar_hits,
        bytes_hashed,
        bytes_memoized,
        frozen = frozen(),
        "bench: repodata document identity",
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
    universe_from_cache_root_inner(cache_root, document_identity_uncached)
}

/// p6ad-4: the walk, parameterised by which identity function reads each
/// document, so the RULE has exactly one implementation and the memoized and
/// unmemoized folds can never diverge in anything but the hashing.
///
/// p6ad-4-3 widens the parameter from a bare `fn` pointer to `impl Fn` for ONE
/// reason, and it is a guard reason: `p6ad43_only_the_replaced_document_is_
/// rehashed` has to count WHICH documents this fold re-read, and a counting
/// closure captures its own tally. The alternative -- asserting on the process-
/// global `HASH_CALLS`/`MEMO_HITS` counters -- is ruled out by the note on
/// `document_identity_memo_poison`: `cargo test` runs the suite in one process,
/// so a counter another test can move is a guard that cannot fail for the right
/// reason. Both production callers still pass a plain `fn` item.
fn universe_from_cache_root_inner(
    cache_root: &std::path::Path,
    identity: impl Fn(&std::path::Path) -> Result<(String, u64)>,
) -> Result<Vec<RepodataDocument>> {
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
        let (sha256, bytes) = identity(&path)?;
        documents.push(RepodataDocument {
            // MEASURED on the live shared cache: the slug alone COLLIDES --
            // two distinct pytorch channel URLs both render `pytorch` and the
            // verb printed `pytorch/linux-64` twice with different hashes. The
            // filename's hex suffix IS `sha256(channel_url|subdir)[..8]`
            // (`disk_cache_path`), so appending it makes the label injective
            // again without inventing a URL the filename does not carry.
            //
            // The slug is still a lossy rendering of the channel URL, and the
            // backend's `conda_universe` row carries full URLs in its
            // `channels=[..]` list. p6ad-6 RETIRES the warning that used to
            // stand here ("a verb digest is only comparable with another VERB
            // digest"): the backend's `repodata_universe` is now folded from
            // THIS function's output too, so a verb header and a backend row
            // print the same number for the same snapshot, and a disagreement
            // between them is now a real finding rather than a category error.
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
    ///
    /// p6ad-4-3 ROOT FIX, AND IT IS A FLAKE THIS CAMPAIGN ALREADY PAID FOR.
    /// Holding it in the WRITER alone is not enough: `with_frozen` decides
    /// whether `document_identity_uncached` consults the sidecar AT ALL, so any
    /// test running in parallel with it reads a different production path than
    /// the one it was written against. MEASURED: gate 5918015 went RED with
    /// `p6ad_one_changed_byte_in_one_subdir_moves_the_universe_digest` failing
    /// on "the changed document must rehash" -- a sidecar written moments
    /// earlier for the OLD bytes, believed because the freeze was on and the
    /// stat tuple had not moved -- and the gate's own isolation re-run printed
    /// `ISOLATED_GREEN` for it, which is the signature of exactly this race and
    /// not of a real defect. The same gate on the same tree had been GREEN an
    /// hour earlier (5917002), so it is a coin flip, and a guard that fails for
    /// the wrong reason half the time is a guard nobody can read.
    ///
    /// So EVERY test that reads a document through the identity functions --
    /// `document_identity_uncached`, `document_identity_blocking`, and the four
    /// folds that call them -- takes this lock now, not only the one that flips
    /// the variable. It costs the suite the serialisation of fifteen tests that
    /// each run in milliseconds.
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
        // p6ad-4-3 ROOT FIX (see the note on FROZEN_TEST_LOCK): every test that
        // reads a document through the identity functions takes this lock, because
        // `with_frozen` flips a PROCESS-GLOBAL env var that decides whether the
        // sidecar is consulted at all.
        let _freeze_guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

        let (sha_a, len_a) = document_identity_uncached(&path_a).unwrap();
        let (sha_b, len_b) = document_identity_uncached(&path_b).unwrap();
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
        // p6ad-4-3 ROOT FIX (see the note on FROZEN_TEST_LOCK): every test that
        // reads a document through the identity functions takes this lock, because
        // `with_frozen` flips a PROCESS-GLOBAL env var that decides whether the
        // sidecar is consulted at all.
        let _freeze_guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = unique_tmp_dir("p6ad-onebyte");
        let linux = write_doc(&dir, "conda_forge--linux-64--a.json", br#"{"packages":{"a":1}}"#);
        let noarch_before = write_doc(&dir, "conda_forge--noarch--b.json", br#"{"packages":{"b":1}}"#);

        let (linux_sha, linux_len) = document_identity_uncached(&linux).unwrap();
        let (noarch_sha, noarch_len) = document_identity_uncached(&noarch_before).unwrap();
        let before = universe_digest_of(&[
            doc("https://c/conda-forge", "linux-64", &linux_sha, linux_len),
            doc("https://c/conda-forge", "noarch", &noarch_sha, noarch_len),
        ]);

        // One byte, in noarch only. The sidecar must NOT be believed here --
        // it was written moments ago for the old bytes.
        std::fs::write(&noarch_before, br#"{"packages":{"b":2}}"#).unwrap();
        let (noarch_sha2, noarch_len2) = document_identity_uncached(&noarch_before).unwrap();
        assert_ne!(noarch_sha, noarch_sha2, "the changed document must rehash");
        let (linux_sha2, linux_len2) = document_identity_uncached(&linux).unwrap();
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
        let truth = document_identity_uncached(&path).unwrap().0;

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
                document_identity_uncached(&path).unwrap().0,
                sentinel,
                "under a freeze the memo must actually be consulted"
            );
        });
        // And with NO freeze the same sidecar is inert: the bytes decide.
        assert_eq!(
            document_identity_uncached(&path).unwrap().0,
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
                document_identity_uncached(&path).unwrap().0,
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
        with_frozen(|| assert_eq!(document_identity_uncached(&path).unwrap().0, truth));

        // Garbage is inert too, and never fatal.
        std::fs::write(&sidecar, b"not json at all").unwrap();
        with_frozen(|| assert_eq!(document_identity_uncached(&path).unwrap().0, truth));
        assert_eq!(document_identity_uncached(&path).unwrap().0, truth);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// p6ad-4 GUARD 1 — TWO CONSUMERS OF ONE DOCUMENT HASH IT ONCE.
    ///
    /// The fix, stated as the thing that used to be false. On `54b5852` the two
    /// consumers of a repodata document's identity in one backend process --
    /// `record_document` for each consulted pair, and the snapshot fold over
    /// every document in the cache root -- each stream the file through
    /// `Sha256`. MEASURED on a cold canonical relock (`P6AF2G-5872517-N`, arm N,
    /// unfrozen): 12.52 GB on the rowed path plus 12.59 GB silently under the
    /// fold = 25.11 GB, of which exactly half is this duplicate.
    ///
    /// PROVED BY POISON, NOT BY A COUNTER. A sentinel hash is planted in the
    /// memo for the document's current stat tuple; the fold must return the
    /// SENTINEL, which it can only do by consulting the memo instead of reading
    /// the file. The unmemoized fold over the same root returns the truth in the
    /// same breath, so the assertion is non-vacuous.
    ///
    /// RED ON THE OLD CODE: with no memo the fold rehashes and returns the true
    /// sha, not the sentinel.
    #[test]
    fn p6ad4_two_consumers_of_one_document_hash_it_once() {
        // p6ad-4-3 ROOT FIX (see the note on FROZEN_TEST_LOCK): every test that
        // reads a document through the identity functions takes this lock, because
        // `with_frozen` flips a PROCESS-GLOBAL env var that decides whether the
        // sidecar is consulted at all.
        let _freeze_guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let root = unique_tmp_dir("p6ad4-once");
        let dir = root.join("retread-repodata");
        std::fs::create_dir_all(&dir).unwrap();
        let body = br#"{"packages":{"a":1,"b":2,"c":3}}"#;
        let path = write_doc(&dir, "conda_forge--linux-64--aa.json", body);

        // CONSUMER 1: the `record_document` path, through the production entry
        // point. It must both answer correctly AND populate the memo.
        let (truth, bytes) = document_identity_blocking(&path).unwrap();
        assert_eq!(bytes, body.len() as u64);
        assert_eq!(
            document_identity_memo_get(&path),
            Some((truth.clone(), bytes)),
            "the first consumer must leave the hash in the memo"
        );

        // Plant the sentinel for the file's OWN stat tuple.
        let sentinel = "a".repeat(64);
        document_identity_memo_poison(&path, &sentinel, bytes);

        // CONSUMER 2: the snapshot fold, over the SAME document, in the SAME
        // process, driven through the production function.
        let memoized = universe_from_cache_root_inner(&root, document_identity_blocking).unwrap();
        assert_eq!(memoized.len(), 1, "one document in the fixture root");
        assert_eq!(
            memoized[0].sha256, sentinel,
            "TWO consumers, ONE hash: the fold must take the memo, not the bytes"
        );

        // NON-VACUITY: the unmemoized fold reads the file and gets the truth,
        // so the assertion above is about a live path and a real difference.
        let plain = universe_from_cache_root(&root).unwrap();
        assert_eq!(plain[0].sha256, truth, "the unmemoized fold reads the bytes");
        assert_ne!(truth, sentinel);

        // And the reader row's counters moved in the right direction. Monotonic
        // only: other tests share this process and may add to them.
        let (_, memo_hits, _, _, bytes_memoized) = document_identity_counters();
        assert!(memo_hits >= 1, "the memo hit must be counted for the row");
        assert!(bytes_memoized >= bytes, "and the bytes it saved recorded");
        std::fs::remove_dir_all(&root).ok();
    }

    /// p6ad-4 GUARD 2 — A MUTATED DOCUMENT INVALIDATES THE MEMO, BOTH WAYS.
    ///
    /// The memo has two invalidators and a guard that exercised one would leave
    /// the other free to hand back a stale hash:
    ///
    ///   (a) THE STAT KEY. A document rewritten at a different length is a
    ///       different key, so the planted sentinel is unreachable and the bytes
    ///       are read, with no invalidation call at all.
    ///   (b) THE WRITER'S INVALIDATION. `invalidate_snapshot_digest` is what
    ///       `refresh_disk_cache` calls after `write_atomic` and what the
    ///       corrupt-cache eviction calls, and it must drop the per-document
    ///       memo as well as the fold -- otherwise the next fold reads this memo
    ///       for the bytes the writer just replaced. This is the half that
    ///       covers the one shape a stat tuple is NOT guaranteed to see, the
    ///       same-length in-place rewrite p6ad's own guards went RED on.
    ///
    /// RED ON THE OLD CODE: half (b) fails, because `invalidate_snapshot_digest`
    /// on `54b5852` clears only `SNAPSHOT_DIGEST`.
    #[test]
    fn p6ad4_a_mutated_document_invalidates_the_memo() {
        // p6ad-4-3 ROOT FIX (see the note on FROZEN_TEST_LOCK): every test that
        // reads a document through the identity functions takes this lock, because
        // `with_frozen` flips a PROCESS-GLOBAL env var that decides whether the
        // sidecar is consulted at all.
        let _freeze_guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = unique_tmp_dir("p6ad4-mutate");
        let path = write_doc(&dir, "conda_forge--noarch--bb.json", br#"{"packages":{"b":1}}"#);
        let truth = document_identity_blocking(&path).unwrap().0;
        let sentinel = "c".repeat(64);

        // NON-VACUITY: with the sentinel planted the production entry point
        // returns it, so both misses below are real misses.
        document_identity_memo_poison(&path, &sentinel, 20);
        assert_eq!(
            document_identity_blocking(&path).unwrap().0,
            sentinel,
            "the memo must genuinely be consulted"
        );

        // (a) A DIFFERENT LENGTH: the stat key alone must miss.
        std::fs::write(&path, br#"{"packages":{"b":1,"c":2}}"#).unwrap();
        let grown = document_identity_blocking(&path).unwrap();
        assert_ne!(grown.0, sentinel, "a rewritten document must not take the memo");
        assert_ne!(grown.0, truth, "and must report its own bytes");
        assert_eq!(grown.1, 26);

        // (b) THE WRITER'S INVALIDATION, with the file untouched so that ONLY
        // the invalidation can be what clears it.
        document_identity_memo_poison(&path, &sentinel, 26);
        assert_eq!(document_identity_blocking(&path).unwrap().0, sentinel);
        invalidate_snapshot_digest(&path);
        assert_eq!(
            document_identity_memo_get(&path),
            None,
            "invalidate_snapshot_digest must drop the per-document memo too"
        );
        assert_eq!(
            document_identity_blocking(&path).unwrap().0,
            grown.0,
            "and the next read must name the bytes on disk"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// p6ad-4 GUARD 3 — THE MEMO DOES NOT MOVE THE DIGEST.
    ///
    /// The one thing a speed fix on this path is not allowed to do. The same
    /// fixture root is folded twice -- `snapshot_digest_at`, which hashes every
    /// document, and `snapshot_digest_memoized_at`, which is what production
    /// calls -- and the two must be byte-identical. The memo is warmed first and
    /// asserted warm, so the equality is not the equality of two cold folds.
    #[test]
    fn p6ad4_the_memo_does_not_move_the_digest() {
        // p6ad-4-3 ROOT FIX (see the note on FROZEN_TEST_LOCK): every test that
        // reads a document through the identity functions takes this lock, because
        // `with_frozen` flips a PROCESS-GLOBAL env var that decides whether the
        // sidecar is consulted at all.
        let _freeze_guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let root = unique_tmp_dir("p6ad4-digest");
        let dir = root.join("retread-repodata");
        std::fs::create_dir_all(&dir).unwrap();
        let a = write_doc(&dir, "conda_forge--linux-64--aa.json", br#"{"packages":{"a":1}}"#);
        let b = write_doc(&dir, "conda_forge--noarch--bb.json", br#"{"packages":{"b":1}}"#);
        let c = write_doc(&dir, "nvidia--linux-64--cc.json", br#"{"packages":{"c":1}}"#);

        let unmemoized = snapshot_digest_at(&root);
        assert_ne!(unmemoized, UNIVERSE_UNAVAILABLE, "{unmemoized}");

        // Warm the memo through the production entry point, and assert it IS
        // warm -- otherwise the equality below is two cold folds agreeing.
        let memoized_cold = snapshot_digest_memoized_at(&root);
        for path in [&a, &b, &c] {
            assert!(
                document_identity_memo_get(path).is_some(),
                "the production fold must populate the memo for {}",
                path.display()
            );
        }
        let memoized_warm = snapshot_digest_memoized_at(&root);

        assert_eq!(
            unmemoized, memoized_cold,
            "the memoized fold must be byte-identical to the unmemoized one"
        );
        assert_eq!(memoized_cold, memoized_warm, "and stable across a warm memo");
        std::fs::remove_dir_all(&root).ok();
    }

    /// p6ad-4-3 GUARD 1 — A ONE-DOCUMENT REFRESH RE-HASHES EXACTLY ONE
    /// DOCUMENT, AND THE UNIVERSE DIGEST DOES NOT MOVE.
    ///
    /// This is the whole of p6ad-4-3 in one test. p6ad-4 measured the price of
    /// the whole-map `clear()` on its own arm: 374 memo hits against 222 misses
    /// is 62.8 % by COUNT but 12.05 % by BYTES, because the two documents a
    /// refresh replaces (conda-forge `linux-64` 638 520 052 B + `noarch`
    /// 253 824 887 B) carry 99.2 % of the snapshot's bytes and the clear threw
    /// away every other document's hash along with them.
    ///
    /// RED ON THE OLD CODE: with `invalidate_snapshot_digest` clearing the
    /// whole map, the tally below is all FOUR documents, not one. (The
    /// mutation control for this lane is exactly that: put `clear()` back in
    /// the invalidator's body, keep the signature, and this assert fires.)
    ///
    /// THE TALLY IS A CAPTURED CLOSURE, NOT THE PROCESS-GLOBAL COUNTERS, for
    /// the reason `document_identity_memo_poison` already states: `cargo test`
    /// runs the suite in one process and a counter another test can move is a
    /// guard that cannot fail for the right reason.
    #[test]
    fn p6ad43_only_the_replaced_document_is_rehashed() {
        // p6ad-4-3 ROOT FIX (see the note on FROZEN_TEST_LOCK): every test that
        // reads a document through the identity functions takes this lock, because
        // `with_frozen` flips a PROCESS-GLOBAL env var that decides whether the
        // sidecar is consulted at all.
        let _freeze_guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let root = unique_tmp_dir("p6ad43-scope");
        let dir = root.join("retread-repodata");
        std::fs::create_dir_all(&dir).unwrap();
        let a = write_doc(&dir, "conda_forge--linux-64--aa.json", br#"{"packages":{"a":1}}"#);
        let b = write_doc(&dir, "conda_forge--noarch--bb.json", br#"{"packages":{"b":1}}"#);
        let c = write_doc(&dir, "nvidia--linux-64--cc.json", br#"{"packages":{"c":1}}"#);
        let d = write_doc(&dir, "nvidia--noarch--dd.json", br#"{"packages":{"d":1}}"#);

        // Warm every entry through the PRODUCTION fold, and assert it warm --
        // otherwise "only one was re-hashed" is four cold misses in disguise.
        let warm = snapshot_digest_memoized_at(&root);
        assert_ne!(warm, UNIVERSE_UNAVAILABLE, "{warm}");
        for path in [&a, &b, &c, &d] {
            assert!(
                document_identity_memo_get(path).is_some(),
                "the production fold must populate the memo for {}",
                path.display()
            );
        }

        // ONE writer replaces ONE document, which is what `refresh_disk_cache`
        // and the corrupt-cache eviction each do, and it names what it changed.
        std::fs::write(&a, br#"{"packages":{"a":2,"a2":3}}"#).unwrap();
        invalidate_snapshot_digest(&a);

        // The three it did not touch keep their memoized hashes.
        for path in [&b, &c, &d] {
            assert!(
                document_identity_memo_get(path).is_some(),
                "p6ad-4-3: a one-document refresh must not evict {}",
                path.display()
            );
        }

        // And the next fold reads exactly one document off disk.
        let rehashed: std::sync::Mutex<Vec<PathBuf>> = std::sync::Mutex::new(Vec::new());
        let documents = universe_from_cache_root_inner(&root, |p| {
            if document_identity_memo_get(p).is_none() {
                rehashed.lock().unwrap().push(p.to_path_buf());
            }
            document_identity_blocking(p)
        })
        .unwrap();
        let rehashed = rehashed.into_inner().unwrap();
        assert_eq!(
            rehashed,
            vec![a.clone()],
            "exactly ONE document may be re-hashed after a one-document refresh; re-hashed {rehashed:?}"
        );
        assert_eq!(documents.len(), 4, "all four documents are still in the fold");

        // THE VALUE IS THE SAME VALUE. The fold through the scoped memo is
        // byte-identical to a full re-hash of the same bytes -- which is the
        // one thing this change is not allowed to move.
        let full_rehash = snapshot_digest_at(&root);
        assert_ne!(full_rehash, UNIVERSE_UNAVAILABLE, "{full_rehash}");
        assert_eq!(
            universe_digest_of(&documents),
            full_rehash,
            "the scoped-memo fold must equal a full re-hash"
        );
        assert_eq!(
            snapshot_digest_memoized_at(&root),
            full_rehash,
            "and so must the production entry point"
        );
        assert_ne!(full_rehash, warm, "NON-VACUITY: the replacement did move the digest");
        std::fs::remove_dir_all(&root).ok();
    }

    /// p6ad-4-3 GUARD 2 — THE INVALIDATION IS SCOPED, WITH NOT ONE BYTE
    /// CHANGING ON DISK.
    ///
    /// Two documents, two sentinels, one invalidation naming ONE of them. No
    /// file is written between the sentinels and the assertions, so the
    /// invalidation is the only thing that can move either entry: this is the
    /// half that isolates the fix from the stat key entirely.
    ///
    /// RED ON THE OLD CODE: `clear()` takes the kept document's sentinel with
    /// it and the second assertion fires.
    #[test]
    fn p6ad43_invalidation_is_scoped_to_the_replaced_document() {
        // p6ad-4-3 ROOT FIX (see the note on FROZEN_TEST_LOCK): every test that
        // reads a document through the identity functions takes this lock, because
        // `with_frozen` flips a PROCESS-GLOBAL env var that decides whether the
        // sidecar is consulted at all.
        let _freeze_guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = unique_tmp_dir("p6ad43-scoped");
        let replaced = write_doc(&dir, "conda_forge--linux-64--aa.json", br#"{"packages":{"a":1}}"#);
        let kept = write_doc(&dir, "conda_forge--noarch--bb.json", br#"{"packages":{"b":1}}"#);
        let sentinel_replaced = "d".repeat(64);
        let sentinel_kept = "e".repeat(64);

        // NON-VACUITY FIRST: both sentinels are genuinely reachable through the
        // production entry point, so both assertions below are real.
        document_identity_memo_poison(&replaced, &sentinel_replaced, 20);
        document_identity_memo_poison(&kept, &sentinel_kept, 20);
        assert_eq!(
            document_identity_blocking(&replaced).unwrap().0,
            sentinel_replaced,
            "the memo must genuinely be consulted for the replaced document"
        );
        assert_eq!(
            document_identity_blocking(&kept).unwrap().0,
            sentinel_kept,
            "the memo must genuinely be consulted for the kept document"
        );

        invalidate_snapshot_digest(&replaced);

        assert_eq!(
            document_identity_memo_get(&replaced),
            None,
            "the replaced document's entry must go"
        );
        assert_eq!(
            document_identity_memo_get(&kept).map(|(sha256, _)| sha256),
            Some(sentinel_kept),
            "p6ad-4-3: a writer that replaced ONE document must not throw away another \
             document's hash -- on the production shape that is 99.2 % of the bytes"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// p6ad-4-3 GUARD 3 — A SAME-LENGTH IN-PLACE REWRITE WITH mtime RESTORED
    /// MUST STILL INVALIDATE, WHETHER OR NOT THE STAT TUPLE MOVED.
    ///
    /// The document is rewritten to DIFFERENT BYTES at the SAME LENGTH, in
    /// place (same inode), and its mtime/atime are put back where they were, so
    /// `len`, `ino` and `mtime` all match the memo's key. Whether `ctime` moves
    /// is a property of the FILESYSTEM AND THE CLOCK, not of this code, and
    /// MEASURED: gate 5917002 (node2341) saw it move, gate 5918015 (node2338)
    /// and mutation arm BASE (node2341) saw it NOT move -- the whole sequence
    /// lands inside one ctime tick when the box is quiet, and
    /// `stat_key(before) == stat_key(after)` byte for byte
    /// (`20:78490866:1788642585:910882335:1788642585:910882335`, both sides).
    /// A guard that asserts "ctime moved" is therefore a guard that fails for
    /// the wrong reason on a quiet node, and the first version of this one did
    /// exactly that.
    ///
    /// So it asserts the thing that is TRUE IN BOTH WORLDS and is p6ad-4-3's
    /// own claim: the writer names the document it replaced, and after that
    /// call the memo holds NOTHING for that path under ANY stat tuple. Which
    /// branch the filesystem took is printed, and each branch carries its own
    /// assertion about what the memo held BEFORE the invalidation -- so the
    /// non-vacuity is real either way:
    ///
    ///   * ctime moved   -> the stat key missed on its own and the memo was
    ///     already cold for the current tuple, but the STALE entry is still in
    ///     the map under the old one (`entries_for` = 2 after a re-read).
    ///   * ctime did not -> the stat key MATCHES and the memo is genuinely
    ///     STALE: it hands back the hash of bytes that are no longer on disk.
    ///     This is the exact shape the brief names, and the only thing that can
    ///     save it is the writer's invalidation.
    ///
    /// The MUTATION CONTROL on the ctime term stays, in the form that does not
    /// depend on the clock: the key built WITHOUT the ctime term is asserted
    /// UNCHANGED across the rewrite, so `stat_key` minus ctime is demonstrably
    /// blind here. (The ctime term's own RED-on-removal arm is p6ad-4's M3 in
    /// `p6ad4-work/mutations.sh`, against
    /// `p6ad4_a_mutated_document_invalidates_the_memo` half (a); this lane does
    /// not duplicate it.)
    #[test]
    fn p6ad43_a_same_length_rewrite_with_preserved_mtime_still_invalidates() {
        // p6ad-4-3 ROOT FIX (see the note on FROZEN_TEST_LOCK): every test that
        // reads a document through the identity functions takes this lock, because
        // `with_frozen` flips a PROCESS-GLOBAL env var that decides whether the
        // sidecar is consulted at all.
        let _freeze_guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::os::unix::fs::MetadataExt as _;
        let dir = unique_tmp_dir("p6ad43-ctime");
        let path = write_doc(&dir, "conda_forge--linux-64--aa.json", br#"{"packages":{"a":1}}"#);
        let before = std::fs::metadata(&path).unwrap();
        let (truth, bytes) = document_identity_blocking(&path).unwrap();
        assert!(
            document_identity_memo_get(&path).is_some(),
            "warm the memo before rewriting under it"
        );
        assert_eq!(document_identity_memo_entries_for(&path), 1);

        std::fs::write(&path, br#"{"packages":{"a":9}}"#).unwrap();
        let handle = std::fs::File::options().write(true).open(&path).unwrap();
        handle
            .set_times(
                std::fs::FileTimes::new()
                    .set_accessed(before.accessed().unwrap())
                    .set_modified(before.modified().unwrap()),
            )
            .unwrap();
        drop(handle);
        let after = std::fs::metadata(&path).unwrap();

        // MUTATION CONTROL ON THE CTIME TERM, clock-independent: every other
        // field of the key is unchanged, so a `stat_key` without ctime is blind
        // to this rewrite by construction.
        let key_without_ctime = |m: &std::fs::Metadata| {
            format!("{}:{}:{}:{}", m.len(), m.ino(), m.mtime(), m.mtime_nsec())
        };
        assert_eq!(
            key_without_ctime(&before),
            key_without_ctime(&after),
            "the rewrite must be invisible to len/ino/mtime -- otherwise this guard proves nothing"
        );

        let ctime_moved = stat_key(&before) != stat_key(&after);
        eprintln!(
            "p6ad43 guard 3: ctime_moved={ctime_moved} before={} after={}",
            stat_key(&before),
            stat_key(&after)
        );
        if ctime_moved {
            // The ctime term saw it with no invalidation call at all, and the
            // re-read files a SECOND entry under the new tuple.
            assert_eq!(
                document_identity_memo_get(&path),
                None,
                "the ctime term must miss on a same-length in-place rewrite"
            );
            let seen = document_identity_blocking(&path).unwrap();
            assert_ne!(seen.0, truth, "the re-hash must name the new bytes");
            assert_eq!(seen.1, bytes, "which are the same length as the old ones");
            assert_eq!(
                document_identity_memo_entries_for(&path),
                2,
                "the OLD stat tuple's entry is still in the map, unreachable by key"
            );
        } else {
            // THE STAT TUPLE IS UNCHANGED AND THE BYTES DIFFER: the memo is
            // stale and there is nothing in the key that can tell.
            assert_eq!(
                document_identity_memo_get(&path).map(|(sha256, _)| sha256),
                Some(truth.clone()),
                "with the whole stat tuple unchanged the memo must still be answering \
                 -- that staleness is the reason the writer has to invalidate"
            );
            assert_eq!(document_identity_memo_entries_for(&path), 1);
        }

        // THE CLAIM UNDER TEST, TRUE IN BOTH BRANCHES: the writer names the
        // document it replaced, and the memo then holds NOTHING for that path
        // under ANY stat tuple -- including the stale one.
        invalidate_snapshot_digest(&path);
        assert_eq!(
            document_identity_memo_entries_for(&path),
            0,
            "p6ad-4-3 removes by path across every stat tuple, including the stale one"
        );
        let fresh = document_identity_blocking(&path).unwrap();
        assert_ne!(fresh.0, truth, "and the next read names the bytes on disk");
        assert_eq!(fresh.1, bytes, "at the same length as the bytes it replaced");
        std::fs::remove_dir_all(&dir).ok();
    }


    /// The verb reads the snapshot on disk and nothing else: dotfiles (the
    /// fetch locks and the sha sidecars) are not documents, and the summary
    /// line names every pair it found with the digest it folded.
    #[test]
    fn p6ad_the_summary_names_every_document_the_snapshot_actually_holds() {
        // p6ad-4-3 ROOT FIX (see the note on FROZEN_TEST_LOCK): every test that
        // reads a document through the identity functions takes this lock, because
        // `with_frozen` flips a PROCESS-GLOBAL env var that decides whether the
        // sidecar is consulted at all.
        let _freeze_guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        // p6ad-4-3 ROOT FIX (see the note on FROZEN_TEST_LOCK): every test that
        // reads a document through the identity functions takes this lock, because
        // `with_frozen` flips a PROCESS-GLOBAL env var that decides whether the
        // sidecar is consulted at all.
        let _freeze_guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let empty = universe_digest_of(&[]);
        assert_eq!(empty.len(), 16, "{empty}");
        assert_ne!(
            empty,
            universe_digest_of(&[doc("https://c/conda-forge", "linux-64", &"a".repeat(64), 1)]),
        );
    }

    // -----------------------------------------------------------------
    // p6ad-6 GUARDS. The defect these pin: `repodata_universe` was the fold
    // of the PROCESS-LOCAL CONSULTED REGISTRY, so it moved when a process's
    // channel subset moved and when a fan-out was only half done. MEASURED on
    // mBH-relock 5851226: 20 distinct digests across 269 rows inside ONE
    // relock, off a snapshot where every one of the 10 (channel, subdir)
    // pairs had exactly ONE sha256. Every guard below is RED on that rule.
    //
    // All but one drive `snapshot_digest_at`, the fold with no memo, so they
    // are safe under `cargo test`'s parallelism. The single guard that has to
    // exercise the process-global memo is the ONLY test that touches it.
    // -----------------------------------------------------------------

    /// Build a cache root holding the named documents.
    fn snapshot_root(tag: &str, documents: &[(&str, &str)]) -> PathBuf {
        let root = unique_tmp_dir(tag);
        let dir = root.join("retread-repodata");
        std::fs::create_dir_all(&dir).unwrap();
        for (name, body) in documents {
            write_doc(&dir, name, body.as_bytes());
        }
        root
    }

    /// THE FIXTURE. Two bundles resolved against ONE snapshot must carry ONE
    /// digest, whether or not they consulted the same channels.
    ///
    /// RED ON THE OLD RULE, and the fixture proves itself DISCRIMINATING
    /// rather than merely passing: the two consulted sets are asserted to fold
    /// to two different values under the old rule before the new rule is
    /// asserted to fold them to one.
    #[test]
    fn p6ad6_two_bundles_sharing_one_snapshot_yield_one_universe_digest() {
        // p6ad-4-3 ROOT FIX (see the note on FROZEN_TEST_LOCK): every test that
        // reads a document through the identity functions takes this lock, because
        // `with_frozen` flips a PROCESS-GLOBAL env var that decides whether the
        // sidecar is consulted at all.
        let _freeze_guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let root = snapshot_root(
            "p6ad6-share",
            &[
                ("conda_forge--linux-64--aa.json", r#"{"packages":{"a":1}}"#),
                ("conda_forge--noarch--bb.json", r#"{"packages":{"b":1}}"#),
                ("nvidia--linux-64--cc.json", r#"{"packages":{"c":1}}"#),
                ("nvidia--noarch--dd.json", r#"{"packages":{"d":1}}"#),
            ],
        );
        let all = universe_from_cache_root(&root).unwrap();
        assert_eq!(all.len(), 4, "{all:?}");

        // Bundle A consulted 2 pairs, bundle B all 4 -- the production shape
        // exactly (pairs=4 / pairs=8 / pairs=10 in one relock).
        let bundle_a = all[..2].to_vec();
        let bundle_b = all.clone();
        assert_ne!(
            universe_digest_of(&bundle_a),
            universe_digest_of(&bundle_b),
            "the OLD rule must disagree here, or this fixture proves nothing",
        );

        let digest_a = snapshot_digest_at(&root);
        let digest_b = snapshot_digest_at(&root);
        assert_eq!(digest_a, digest_b, "the snapshot digest is not a function of the root");
        // ...and it is the number the `repodata-universe` verb prints, which is
        // what makes a backend row and a job header comparable at all.
        assert_eq!(digest_a, universe_digest_of(&all), "verb and backend fold different sets");
        std::fs::remove_dir_all(&root).ok();
    }

    /// The digest must not move while a fan-out accumulates. On the old rule
    /// every `record_document` moved it, which is what invented 17 of the 20.
    #[test]
    fn p6ad6_the_universe_digest_does_not_move_while_the_fan_out_accumulates() {
        // p6ad-4-3 ROOT FIX (see the note on FROZEN_TEST_LOCK): every test that
        // reads a document through the identity functions takes this lock, because
        // `with_frozen` flips a PROCESS-GLOBAL env var that decides whether the
        // sidecar is consulted at all.
        let _freeze_guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let root = snapshot_root(
            "p6ad6-accum",
            &[
                ("conda_forge--linux-64--aa.json", r#"{"packages":{"a":1}}"#),
                ("conda_forge--noarch--bb.json", r#"{"packages":{"b":1}}"#),
                ("pytorch--linux-64--ee.json", r#"{"packages":{"e":1}}"#),
            ],
        );
        let all = universe_from_cache_root(&root).unwrap();
        let stable = snapshot_digest_at(&root);
        let mut consulted = Vec::new();
        let mut old_rule = Vec::new();
        for document in &all {
            consulted.push(document.clone());
            old_rule.push(universe_digest_of(&consulted));
            assert_eq!(
                snapshot_digest_at(&root),
                stable,
                "the snapshot digest moved while only the consulted set grew",
            );
        }
        old_rule.sort();
        old_rule.dedup();
        assert_eq!(old_rule.len(), 3, "the OLD rule must move on every accumulation step");
        std::fs::remove_dir_all(&root).ok();
    }

    /// One changed byte anywhere in the snapshot still moves the digest -- the
    /// p6ad property the new input set must not have cost us.
    #[test]
    fn p6ad6_one_changed_byte_in_the_snapshot_still_moves_the_snapshot_digest() {
        // p6ad-4-3 ROOT FIX (see the note on FROZEN_TEST_LOCK): every test that
        // reads a document through the identity functions takes this lock, because
        // `with_frozen` flips a PROCESS-GLOBAL env var that decides whether the
        // sidecar is consulted at all.
        let _freeze_guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let root = snapshot_root(
            "p6ad6-byte",
            &[
                ("conda_forge--linux-64--aa.json", r#"{"packages":{"a":1}}"#),
                ("conda_forge--noarch--bb.json", r#"{"packages":{"b":1}}"#),
            ],
        );
        let before = snapshot_digest_at(&root);
        write_doc(
            &root.join("retread-repodata"),
            "conda_forge--noarch--bb.json",
            br#"{"packages":{"b":2}}"#,
        );
        assert_ne!(before, snapshot_digest_at(&root), "a changed document did not move the digest");
        std::fs::remove_dir_all(&root).ok();
    }

    /// A cache root with no snapshot at all is NOT the empty universe: it gets
    /// a token no grep for a digest can match. "Could not be read" and "is
    /// empty" are different facts and a row must not conflate them.
    #[test]
    fn p6ad6_an_unreadable_snapshot_is_not_the_empty_universe() {
        // p6ad-4-3 ROOT FIX (see the note on FROZEN_TEST_LOCK): every test that
        // reads a document through the identity functions takes this lock, because
        // `with_frozen` flips a PROCESS-GLOBAL env var that decides whether the
        // sidecar is consulted at all.
        let _freeze_guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let root = unique_tmp_dir("p6ad6-missing");
        let digest = snapshot_digest_at(&root);
        assert_eq!(digest, UNIVERSE_UNAVAILABLE, "{digest}");
        assert_ne!(digest, universe_digest_of(&[]), "{digest}");
        std::fs::remove_dir_all(&root).ok();
    }

    /// The MEMO, both directions, and the only test that touches the
    /// process-global one. A memo that outlives the bytes it names would be
    /// the p6y failure in a new costume, so the writer drops it; and it is
    /// keyed on the root, so a guard's temp root can never answer for
    /// production's cache.
    #[test]
    fn p6ad6_the_snapshot_memo_is_keyed_on_the_root_and_dropped_by_the_writer() {
        // p6ad-4-3 ROOT FIX (see the note on FROZEN_TEST_LOCK): every test that
        // reads a document through the identity functions takes this lock, because
        // `with_frozen` flips a PROCESS-GLOBAL env var that decides whether the
        // sidecar is consulted at all.
        let _freeze_guard = FROZEN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let a = snapshot_root(
            "p6ad6-memo-a",
            &[("conda_forge--linux-64--aa.json", r#"{"packages":{"a":1}}"#)],
        );
        let b = snapshot_root(
            "p6ad6-memo-b",
            &[("conda_forge--linux-64--aa.json", r#"{"packages":{"z":9}}"#)],
        );
        reset_snapshot_memos();

        let digest_a = universe_digest_at(&a);
        assert_eq!(digest_a, snapshot_digest_at(&a), "the memo returned something else");
        // A different root must never be answered from another root's entry.
        let digest_b = universe_digest_at(&b);
        assert_eq!(digest_b, snapshot_digest_at(&b), "root b was answered from root a's memo");
        assert_ne!(digest_a, digest_b, "two roots with different bytes share one digest");

        // Rewrite root b under the memo. Without an invalidation the memo is
        // supposed to still answer -- that is what makes it a memo...
        write_doc(
            &b.join("retread-repodata"),
            "conda_forge--linux-64--aa.json",
            br#"{"packages":{"z":10}}"#,
        );
        assert_eq!(universe_digest_at(&b), digest_b, "the memo is supposed to be a memo");
        // ...and after the writer drops it, the digest names the bytes on disk.
        // p6ad-4-3: the writer names the document it replaced, which is exactly
        // the one `write_doc` just rewrote under root b.
        invalidate_snapshot_digest(&b.join("retread-repodata").join("conda_forge--linux-64--aa.json"));
        assert_ne!(universe_digest_at(&b), digest_b, "an invalidated memo still answered");

        reset_snapshot_memos();
        std::fs::remove_dir_all(&a).ok();
        std::fs::remove_dir_all(&b).ok();
    }

    /// The wire tag must say the rule changed: a v1 row folded a consulted
    /// subset, a v2 row folds a snapshot, and comparing them is a category
    /// error a reader has to be stopped from making.
    #[test]
    fn p6ad6_the_schema_tag_records_that_the_input_set_changed() {
        assert_eq!(UNIVERSE_SCHEMA, "retread-conda-universe-v2");
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
