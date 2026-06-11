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
//! `~/.cache/rattler/cache/retread-repodata/` with a 30-minute TTL,
//! exactly the scheme both modules used before (same paths, so
//! existing caches stay warm across this upgrade).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use rattler_conda_types::{Channel, ChannelConfig, ChannelUrl};
use rattler_repodata_gateway::sparse::SparseRepoData;
use sha2::{Digest, Sha256};

const REPODATA_TTL: Duration = Duration::from_secs(30 * 60);
const HTTP_USER_AGENT: &str = concat!("pixi-build-retread/", env!("CARGO_PKG_VERSION"));

/// One (channel, subdir)'s lazily-built sparse handle. `None` inside
/// the cell means the build was attempted and failed (channel
/// unreachable AND no disk cache) -- callers treat it as
/// not-consulted, and the negative result is NOT cached so a later
/// call may retry.
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
    cell.get_or_init(|| build_sparse(channel_url.to_string(), subdir.to_string()))
        .await
        .clone()
}

/// The full (channel x [target_subdir, noarch]) fan-out every consumer
/// wants, built concurrently, returned in channel-priority order as
/// `("<channel>/<subdir>" label, handle)` pairs. Unreachable pairs are
/// skipped.
pub async fn sparse_pairs(
    channels: &[ChannelUrl],
    target_subdir: &str,
) -> Vec<(String, Arc<SparseRepoData>)> {
    let mut work: Vec<(String, String)> = Vec::new();
    for channel in channels {
        let url = channel.url().as_str().trim_end_matches('/').to_string();
        work.push((url.clone(), target_subdir.to_string()));
        work.push((url, "noarch".to_string()));
    }
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
    let t = std::time::Instant::now();
    let path = disk_cache_path(&channel_url, &subdir);
    let fresh = disk_cache_is_fresh(&path).await;
    if !fresh {
        match fetch_repodata_bytes(&channel_url, &subdir).await {
            Ok(bytes) => {
                if let Err(e) = write_atomic(&path, &bytes).await {
                    tracing::debug!(
                        channel = %channel_url, subdir = %subdir, error = %format!("{e:#}"),
                        "repodata: disk-cache write failed",
                    );
                    // Stale disk cache (if any) is still better than nothing.
                }
            }
            Err(e) => {
                if !path.exists() {
                    tracing::debug!(
                        channel = %channel_url, subdir = %subdir, error = %format!("{e:#}"),
                        "repodata: unreachable and no disk cache; pair not consulted",
                    );
                    return None;
                }
                tracing::debug!(
                    channel = %channel_url, subdir = %subdir, error = %format!("{e:#}"),
                    "repodata: refresh failed; using stale disk cache",
                );
            }
        }
    }
    // mmap + sparse header parse on the blocking pool (the LazyRepoData
    // deserialize walks the whole document's key structure once).
    let cfg = ChannelConfig::default_with_root_dir(std::env::temp_dir());
    let channel = Channel::from_str(&channel_url, &cfg).ok()?;
    let subdir_clone = subdir.clone();
    let path_clone = path.clone();
    let built = tokio::task::spawn_blocking(move || {
        SparseRepoData::from_file(channel, subdir_clone, path_clone, None)
    })
    .await
    .ok()?;
    match built {
        Ok(s) => {
            tracing::info!(
                channel = %channel_url,
                subdir = %subdir,
                elapsed_ms = t.elapsed().as_millis() as u64,
                "bench: sparse repodata handle built",
            );
            Some(Arc::new(s))
        }
        Err(e) => {
            tracing::debug!(
                channel = %channel_url, subdir = %subdir, error = %e,
                "repodata: sparse open failed",
            );
            None
        }
    }
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

async fn write_atomic(path: &PathBuf, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }
    let tmp = path.with_extension("part");
    tokio::fs::write(&tmp, bytes)
        .await
        .with_context(|| format!("writing {}", tmp.display()))?;
    tokio::fs::rename(&tmp, path)
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

fn dirs_cache_root() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home)
            .join(".cache")
            .join("rattler")
            .join("cache")
    } else {
        std::env::temp_dir().join("retread-cache")
    }
}
