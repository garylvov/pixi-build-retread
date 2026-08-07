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
    work.into_iter()
        .zip(handles)
        .filter_map(|((channel_url, subdir), handle)| {
            handle.map(|h| (format!("{channel_url}/{subdir}"), h))
        })
        .collect()
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
        if !disk_cache_is_fresh(&path).await {
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
                tracing::info!(
                    channel = %channel_url,
                    subdir = %subdir,
                    elapsed_ms = t.elapsed().as_millis() as u64,
                    "bench: sparse repodata handle built",
                );
                return Some(handle);
            }
            None if attempt == 0 && path.exists() => {
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
}
