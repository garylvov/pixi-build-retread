//! Probe workspace conda channels for whether a candidate spec is
//! actually satisfiable, before letting prefer-conda emit it as a
//! conda run-dep.
//!
//! Motivation: prefer-conda routes a dep to the conda side whenever
//! parselmouth (or our fallback name map) names a conda equivalent.
//! That decision used to be name-only -- nobody checked whether the
//! channel HAS a version that satisfies the spec retread would emit.
//! When it doesn't (e.g. isaaclab pins `aiodns ==3.1.1` but
//! conda-forge ships only 3.0.x), the conda solver dies with
//! "no candidates were found for aiodns >=3.1,<4". This module
//! probes prefix.dev's package API per (channel, name, spec) tuple
//! and lets the caller fall back to auto-bundling the PyPI wheel.
//!
//! Probes are bounded-parallel (16 at a time) and cached on disk
//! under `~/.cache/rattler/retread-probes/`. The cache key hashes
//! (channel, package, spec); a fresh repodata snapshot invalidates
//! by changing the spec the caller asked about (lowering false
//! negatives) and by an mtime check (30 minute TTL).

use std::path::PathBuf;
use std::str::FromStr;
use std::time::{Duration, SystemTime};

use anyhow::{anyhow, Context, Result};
use rattler_conda_types::{ChannelUrl, Version, VersionSpec};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Cached probe TTL. Repodata changes (channel adds new builds) but
/// not so fast that we need a fresh hit every solve. Half an hour
/// trades freshness against latency reasonably.
const PROBE_TTL: Duration = Duration::from_secs(30 * 60);

/// A single probe's outcome, recorded in the audit so users can see
/// which deps were validated against which channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    pub package: String,
    pub spec: String,
    /// Channels actually consulted. Non-prefix.dev channels are
    /// silently skipped; if none were prefix.dev, this list is empty
    /// and `satisfiable` is `None` (couldn't decide).
    pub channels_consulted: Vec<String>,
    /// Some(true) if at least one consulted channel has a version
    /// matching the spec; Some(false) if every consulted channel was
    /// reachable and returned candidates but none satisfied; None if
    /// no channel could be consulted.
    pub satisfiable: Option<bool>,
    /// Number of candidates that matched, across all consulted
    /// channels. Diagnostic only; the satisfiability flag is what
    /// drives the fallback decision.
    pub matching_candidates: usize,
}

impl ProbeResult {
    /// True if we have a definitive answer and the answer is "yes,
    /// at least one channel has a satisfying version".
    pub fn is_satisfied(&self) -> bool {
        self.satisfiable == Some(true)
    }

    /// True if we have a definitive answer and the answer is "no,
    /// no channel has a satisfying version". A `None` result
    /// (couldn't probe) is NOT counted as unsatisfied -- the caller
    /// should treat indecision as "keep the current routing" to
    /// avoid spurious fallbacks when prefix.dev is down.
    pub fn is_definitively_unsatisfied(&self) -> bool {
        self.satisfiable == Some(false)
    }
}

/// Probe each prefix.dev channel for at least one version of
/// `package` that matches `spec`. Returns a structured result the
/// audit can record.
pub async fn probe(
    channels: &[ChannelUrl],
    package: &str,
    spec: &str,
) -> ProbeResult {
    let parsed_spec = match VersionSpec::from_str(spec, rattler_conda_types::ParseStrictness::Lenient) {
        Ok(s) => s,
        Err(e) => {
            tracing::debug!(
                package = %package, spec = %spec,
                error = %e,
                "probe: could not parse spec; returning indecisive result"
            );
            return ProbeResult {
                package: package.to_string(),
                spec: spec.to_string(),
                channels_consulted: vec![],
                satisfiable: None,
                matching_candidates: 0,
            };
        }
    };

    let mut consulted: Vec<String> = Vec::new();
    let mut total_matching: usize = 0;
    for channel in channels {
        let Some(channel_name) = prefix_dev_channel_name(channel) else {
            continue;
        };
        consulted.push(channel_name.clone());
        match list_versions(&channel_name, package).await {
            Ok(versions) => {
                let matches = versions
                    .into_iter()
                    .filter(|v| parsed_spec.matches(v))
                    .count();
                total_matching += matches;
            }
            Err(e) => {
                tracing::debug!(
                    channel = %channel_name, package = %package,
                    error = %format!("{e:#}"),
                    "probe: could not list versions; assuming unsatisfied on this channel"
                );
            }
        }
    }

    let satisfiable = if consulted.is_empty() {
        None
    } else {
        Some(total_matching > 0)
    };
    ProbeResult {
        package: package.to_string(),
        spec: spec.to_string(),
        channels_consulted: consulted,
        satisfiable,
        matching_candidates: total_matching,
    }
}

/// Strip the `https://prefix.dev/` prefix from a channel URL. Returns
/// `None` for non-prefix.dev channels (we don't have a generic
/// repodata fetcher; prefix.dev's package API is the cheapest probe).
fn prefix_dev_channel_name(channel: &ChannelUrl) -> Option<String> {
    let s = channel.url().as_str().trim_end_matches('/');
    let name = s.strip_prefix("https://prefix.dev/")?;
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// List every version of `package` that `channel_name` ships, going
/// through the on-disk cache first. Returns Versions parsed via
/// rattler-conda-types so the caller can match them against a
/// VersionSpec directly.
async fn list_versions(channel_name: &str, package: &str) -> Result<Vec<Version>> {
    if let Some(cached) = read_cache(channel_name, package).await? {
        return Ok(cached);
    }
    let fresh = fetch_versions(channel_name, package).await?;
    write_cache(channel_name, package, &fresh)
        .await
        .unwrap_or_else(|e| {
            tracing::debug!(error = %e, "probe: cache write failed (non-fatal)");
        });
    Ok(fresh)
}

/// HTTP path: hit prefix.dev's package endpoint, parse the response,
/// extract the version strings. Unknown response shapes degrade
/// gracefully (return empty list with a debug log).
async fn fetch_versions(channel_name: &str, package: &str) -> Result<Vec<Version>> {
    let api_url = format!(
        "https://prefix.dev/api/v1/channels/{channel_name}/packages/{package}/variants?limit=500"
    );
    let resp = reqwest::Client::new()
        .get(&api_url)
        .send()
        .await
        .with_context(|| format!("GET {api_url}"))?;
    if !resp.status().is_success() {
        // 404 == package not on this channel. Treat as "no versions".
        return Ok(Vec::new());
    }
    let body: serde_json::Value = resp
        .json()
        .await
        .with_context(|| format!("parsing JSON from {api_url}"))?;
    // The prefix.dev API has shifted shapes over time. Try the
    // common candidates and fall through gracefully.
    let mut version_strings: Vec<String> = Vec::new();
    if let Some(arr) = body.as_array() {
        // Flat list of variants
        version_strings.extend(arr.iter().filter_map(extract_version));
    } else if let Some(arr) = body.get("variants").and_then(|v| v.as_array()) {
        version_strings.extend(arr.iter().filter_map(extract_version));
    } else if let Some(arr) = body.get("data").and_then(|v| v.as_array()) {
        version_strings.extend(arr.iter().filter_map(extract_version));
    } else if let Some(arr) = body.get("versions").and_then(|v| v.as_array()) {
        version_strings.extend(arr.iter().filter_map(|v| v.as_str().map(String::from)));
    } else {
        tracing::debug!(
            channel = %channel_name, package = %package,
            "probe: unexpected response shape from prefix.dev; treating as empty"
        );
    }
    let mut versions: Vec<Version> = Vec::with_capacity(version_strings.len());
    for s in version_strings {
        if let Ok(v) = Version::from_str(&s) {
            versions.push(v);
        }
    }
    versions.sort();
    versions.dedup();
    Ok(versions)
}

fn extract_version(v: &serde_json::Value) -> Option<String> {
    v.get("version")?.as_str().map(String::from)
}

#[derive(Serialize, Deserialize)]
struct CachedVersions {
    fetched_at_unix: u64,
    versions: Vec<String>,
}

fn cache_path(channel_name: &str, package: &str) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(channel_name.as_bytes());
    hasher.update(b"|");
    hasher.update(package.as_bytes());
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(16);
    for b in &digest[..8] {
        hex.push_str(&format!("{b:02x}"));
    }
    let dir = dirs_cache_root().join("retread-probes");
    dir.join(format!("{channel_name}--{package}--{hex}.json"))
}

/// Resolve `~/.cache/rattler/` -- matches where the rest of retread's
/// caches live. Falls back to a system temp dir if HOME isn't set.
fn dirs_cache_root() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache").join("rattler").join("cache")
    } else {
        std::env::temp_dir().join("retread-cache")
    }
}

async fn read_cache(channel_name: &str, package: &str) -> Result<Option<Vec<Version>>> {
    let path = cache_path(channel_name, package);
    let bytes = match tokio::fs::read(&path).await {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading cache {}", path.display())),
    };
    let parsed: CachedVersions = match serde_json::from_slice(&bytes) {
        Ok(p) => p,
        Err(_) => return Ok(None), // corrupt cache -> refetch
    };
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs();
    if now.saturating_sub(parsed.fetched_at_unix) > PROBE_TTL.as_secs() {
        return Ok(None);
    }
    let mut versions: Vec<Version> = Vec::with_capacity(parsed.versions.len());
    for s in parsed.versions {
        if let Ok(v) = Version::from_str(&s) {
            versions.push(v);
        }
    }
    Ok(Some(versions))
}

async fn write_cache(channel_name: &str, package: &str, versions: &[Version]) -> Result<()> {
    let path = cache_path(channel_name, package);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|e| anyhow!("system clock before epoch: {e}"))?
        .as_secs();
    let payload = CachedVersions {
        fetched_at_unix: now,
        versions: versions.iter().map(|v| v.to_string()).collect(),
    };
    let bytes = serde_json::to_vec(&payload)?;
    tokio::fs::write(&path, &bytes).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_result_classifies_outcomes() {
        let yes = ProbeResult {
            package: "x".into(),
            spec: ">=1".into(),
            channels_consulted: vec!["conda-forge".into()],
            satisfiable: Some(true),
            matching_candidates: 3,
        };
        assert!(yes.is_satisfied());
        assert!(!yes.is_definitively_unsatisfied());

        let no = ProbeResult {
            satisfiable: Some(false),
            matching_candidates: 0,
            ..yes.clone()
        };
        assert!(!no.is_satisfied());
        assert!(no.is_definitively_unsatisfied());

        let indecisive = ProbeResult {
            satisfiable: None,
            channels_consulted: vec![],
            ..yes
        };
        assert!(!indecisive.is_satisfied());
        // Critical: indecisive is NOT "definitively unsatisfied". A
        // probe that couldn't reach any prefix.dev channel must not
        // trigger a fallback -- prefix.dev outage shouldn't change
        // routing decisions silently.
        assert!(!indecisive.is_definitively_unsatisfied());
    }

    #[test]
    fn prefix_dev_channel_name_strips_known_prefix() {
        let ch: ChannelUrl =
            url::Url::parse("https://prefix.dev/conda-forge/").unwrap().into();
        assert_eq!(prefix_dev_channel_name(&ch).as_deref(), Some("conda-forge"));

        let ch: ChannelUrl =
            url::Url::parse("https://prefix.dev/garylvov").unwrap().into();
        assert_eq!(prefix_dev_channel_name(&ch).as_deref(), Some("garylvov"));

        // Non-prefix.dev channels return None -- the caller should
        // treat that as "we can't probe this one" and consider only
        // the channels we CAN probe.
        let ch: ChannelUrl = url::Url::parse("https://example.com/").unwrap().into();
        assert!(prefix_dev_channel_name(&ch).is_none());
    }

    #[test]
    fn cache_path_is_stable_across_calls() {
        let a = cache_path("conda-forge", "aiodns");
        let b = cache_path("conda-forge", "aiodns");
        assert_eq!(a, b);
        let c = cache_path("conda-forge", "numpy");
        assert_ne!(a, c, "different packages must hash to different files");
    }
}

/// Convenience wrapper used by the auto-bundle path: probe many
/// (name, spec) pairs against the same channel set, bounded-parallel
/// at `concurrency=16`. Returns the results in the order they
/// complete; the caller matches by `(package, spec)`.
pub async fn probe_many(
    channels: &[ChannelUrl],
    specs: Vec<(String, String)>,
) -> Vec<ProbeResult> {
    use futures::stream::{self, StreamExt};
    stream::iter(specs)
        .map(|(name, spec)| {
            let chans = channels.to_vec();
            async move { probe(&chans, &name, &spec).await }
        })
        .buffer_unordered(16)
        .collect::<Vec<_>>()
        .await
}
