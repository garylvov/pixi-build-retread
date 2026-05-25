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
use rattler_conda_types::{ChannelUrl, ParseStrictness, Version, VersionSpec};
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

/// Probe each prefix.dev channel for at least one BUILD of `package`
/// that matches `spec` AND is compatible with `target_python` (when
/// supplied). v0.13.13+: python-aware. Without the python filter, the
/// probe would say "satisfied" for e.g. `gym 0.23.1` (which only
/// ships for py<=3.10 on conda-forge) when the workspace is on py3.11,
/// and the conda solver would then fail with "no candidates found"
/// downstream. `target_python = None` preserves the legacy
/// version-only check.
pub async fn probe(
    channels: &[ChannelUrl],
    package: &str,
    spec: &str,
    target_python: Option<&str>,
) -> ProbeResult {
    let parsed_spec = match VersionSpec::from_str(spec, ParseStrictness::Lenient) {
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

    let target_py_version: Option<Version> = target_python.and_then(|tp| {
        match Version::from_str(tp) {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::debug!(
                    target_python = %tp, error = %e,
                    "probe: could not parse target python; falling back to version-only check",
                );
                None
            }
        }
    });

    let mut consulted: Vec<String> = Vec::new();
    let mut total_matching: usize = 0;
    for channel in channels {
        let Some(channel_name) = prefix_dev_channel_name(channel) else {
            continue;
        };
        consulted.push(channel_name.clone());
        match list_variants(&channel_name, package).await {
            Ok(variants) => {
                let matches = variants
                    .into_iter()
                    .filter(|v| parsed_spec.matches(&v.version))
                    .filter(|v| match (&target_py_version, &v.python_constraint) {
                        // No target python -> legacy behavior: any
                        // version matching the spec counts.
                        (None, _) => true,
                        // Variant declares no python dep (true noarch
                        // or non-python package) -> compatible with
                        // anything.
                        (Some(_), PythonConstraint::Any) => true,
                        (Some(tp), PythonConstraint::Spec(s)) => s.matches(tp),
                    })
                    .count();
                total_matching += matches;
            }
            Err(e) => {
                tracing::debug!(
                    channel = %channel_name, package = %package,
                    error = %format!("{e:#}"),
                    "probe: could not list variants; assuming unsatisfied on this channel"
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

/// One conda variant of a package, with enough metadata to answer
/// "is this build python-compatible with the workspace's target?".
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VariantInfo {
    version: Version,
    python_constraint: PythonConstraint,
}

/// What this variant says about which python it supports.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum PythonConstraint {
    /// No `python` / `python_abi` dep listed. Either a noarch non-
    /// python package, or a build whose python metadata we couldn't
    /// recover from the API. Either way: don't reject on python.
    Any,
    /// Parsed `python >=3.9` / `python_abi 3.11.*` / etc. The probe
    /// uses `spec.matches(target_python_version)`.
    Spec(VersionSpec),
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

/// List every (version, python-constraint) pair for `package` on
/// `channel_name`, going through the on-disk cache first.
async fn list_variants(channel_name: &str, package: &str) -> Result<Vec<VariantInfo>> {
    if let Some(cached) = read_cache(channel_name, package).await? {
        return Ok(cached);
    }
    let fresh = fetch_variants(channel_name, package).await?;
    write_cache(channel_name, package, &fresh)
        .await
        .unwrap_or_else(|e| {
            tracing::debug!(error = %e, "probe: cache write failed (non-fatal)");
        });
    Ok(fresh)
}

/// HTTP path: hit prefix.dev's package endpoint and pull (version,
/// depends[]) per variant. v0.13.13+ also parses the `depends` array
/// so we can recover each build's python constraint. Unknown response
/// shapes degrade gracefully (empty list + debug log).
async fn fetch_variants(channel_name: &str, package: &str) -> Result<Vec<VariantInfo>> {
    let api_url = format!(
        "https://prefix.dev/api/v1/channels/{channel_name}/packages/{package}/variants?limit=500"
    );
    let resp = reqwest::Client::new()
        .get(&api_url)
        .send()
        .await
        .with_context(|| format!("GET {api_url}"))?;
    if !resp.status().is_success() {
        // 404 == package not on this channel. Treat as "no variants".
        return Ok(Vec::new());
    }
    let body: serde_json::Value = resp
        .json()
        .await
        .with_context(|| format!("parsing JSON from {api_url}"))?;
    // The prefix.dev API has shifted shapes over time. Try the common
    // candidates and fall through gracefully.
    let mut variants: Vec<VariantInfo> = Vec::new();
    let push_from_array = |arr: &[serde_json::Value], out: &mut Vec<VariantInfo>| {
        for v in arr {
            if let Some(vi) = extract_variant_info(v) {
                out.push(vi);
            }
        }
    };
    if let Some(arr) = body.as_array() {
        push_from_array(arr, &mut variants);
    } else if let Some(arr) = body.get("variants").and_then(|v| v.as_array()) {
        push_from_array(arr, &mut variants);
    } else if let Some(arr) = body.get("data").and_then(|v| v.as_array()) {
        push_from_array(arr, &mut variants);
    } else if let Some(arr) = body.get("versions").and_then(|v| v.as_array()) {
        // Legacy shape -- version strings only, no depends. Promote
        // to PythonConstraint::Any so the python filter is a no-op
        // (matches legacy behavior on stale API responses).
        for v in arr {
            if let Some(s) = v.as_str() {
                if let Ok(version) = Version::from_str(s) {
                    variants.push(VariantInfo {
                        version,
                        python_constraint: PythonConstraint::Any,
                    });
                }
            }
        }
    } else {
        tracing::debug!(
            channel = %channel_name, package = %package,
            "probe: unexpected response shape from prefix.dev; treating as empty"
        );
    }
    Ok(variants)
}

fn extract_variant_info(v: &serde_json::Value) -> Option<VariantInfo> {
    let version_str = v.get("version")?.as_str()?;
    let version = Version::from_str(version_str).ok()?;
    // `depends` may be an array of strings ("python >=3.9", ...) on
    // some prefix.dev shapes. If absent or non-array, fall through to
    // PythonConstraint::Any (don't reject the variant on python).
    let python_constraint = v
        .get("depends")
        .and_then(|d| d.as_array())
        .map(|deps| python_constraint_from_depends(deps))
        .unwrap_or(PythonConstraint::Any);
    Some(VariantInfo {
        version,
        python_constraint,
    })
}

/// Walk the variant's `depends` array, find any entry naming
/// `python` or `python_abi`, parse its version constraint. Returns
/// `PythonConstraint::Any` if no such entry is present (treat as
/// noarch / compatible-with-anything). Multiple python dep lines
/// (rare) take the first parseable one.
fn python_constraint_from_depends(deps: &[serde_json::Value]) -> PythonConstraint {
    for dep in deps {
        let Some(s) = dep.as_str() else { continue };
        let trimmed = s.trim();
        // matchspec shape: "<name> <version-spec> [<build-spec>]"
        // We only care about the first two tokens.
        let mut parts = trimmed.split_whitespace();
        let name = match parts.next() {
            Some(n) => n,
            None => continue,
        };
        if name != "python" && name != "python_abi" {
            continue;
        }
        let spec_str = parts.next().unwrap_or("*");
        if spec_str == "*" {
            // `python *` (or `python` alone) is any version -- still
            // semantically a python constraint though, so it's not
            // PythonConstraint::Any (which means "no python dep at
            // all"). Use a wide-open VersionSpec instead.
            if let Ok(spec) = VersionSpec::from_str("*", ParseStrictness::Lenient) {
                return PythonConstraint::Spec(spec);
            }
            continue;
        }
        if let Ok(spec) = VersionSpec::from_str(spec_str, ParseStrictness::Lenient) {
            return PythonConstraint::Spec(spec);
        }
    }
    PythonConstraint::Any
}

#[derive(Serialize, Deserialize)]
struct CachedVersions {
    fetched_at_unix: u64,
    /// v0.13.13+: cache the full VariantInfo (version + python
    /// constraint) per build, not just version strings. The old
    /// `versions: Vec<String>` shape is no longer written; old caches
    /// fail to deserialize and refetch (handled by read_cache's
    /// "corrupt cache -> refetch" path).
    variants: Vec<CachedVariant>,
}

#[derive(Serialize, Deserialize)]
struct CachedVariant {
    version: String,
    /// Stored as the conda match-spec string ("python >=3.9",
    /// "python_abi 3.11.*"), reparsed on load. None = no python dep
    /// (PythonConstraint::Any).
    python_constraint: Option<String>,
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

async fn read_cache(channel_name: &str, package: &str) -> Result<Option<Vec<VariantInfo>>> {
    let path = cache_path(channel_name, package);
    let bytes = match tokio::fs::read(&path).await {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading cache {}", path.display())),
    };
    let parsed: CachedVersions = match serde_json::from_slice(&bytes) {
        Ok(p) => p,
        Err(_) => return Ok(None), // corrupt/legacy cache -> refetch
    };
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs();
    if now.saturating_sub(parsed.fetched_at_unix) > PROBE_TTL.as_secs() {
        return Ok(None);
    }
    let mut variants: Vec<VariantInfo> = Vec::with_capacity(parsed.variants.len());
    for cv in parsed.variants {
        let Ok(version) = Version::from_str(&cv.version) else {
            continue;
        };
        let python_constraint = match cv.python_constraint {
            None => PythonConstraint::Any,
            Some(spec_str) => match VersionSpec::from_str(&spec_str, ParseStrictness::Lenient) {
                Ok(s) => PythonConstraint::Spec(s),
                Err(_) => PythonConstraint::Any,
            },
        };
        variants.push(VariantInfo {
            version,
            python_constraint,
        });
    }
    Ok(Some(variants))
}

async fn write_cache(channel_name: &str, package: &str, variants: &[VariantInfo]) -> Result<()> {
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
        variants: variants
            .iter()
            .map(|v| CachedVariant {
                version: v.version.to_string(),
                python_constraint: match &v.python_constraint {
                    PythonConstraint::Any => None,
                    PythonConstraint::Spec(s) => Some(s.to_string()),
                },
            })
            .collect(),
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

    /// v0.13.13+ regression: when prefer-conda's BFS probe asks "does
    /// conda have a version of gym matching `>=0.23,<0.24`", the
    /// version filter alone says YES (conda-forge has gym 0.23.1).
    /// But the only build of that version is `python_abi 3.9.*` /
    /// `python_abi 3.10.*` -- target python 3.11 isn't covered, and
    /// the conda solver then dies with "no candidates found". Parsing
    /// `depends` per variant lets us reject those builds at probe
    /// time and trigger the PyPI fallback.
    #[test]
    fn python_constraint_extracted_from_depends() {
        use serde_json::json;
        // python_abi pin -- explicit minor
        let c = python_constraint_from_depends(&[
            json!("python_abi 3.11.* *_cp311"),
            json!("six >=1.16"),
        ]);
        match c {
            PythonConstraint::Spec(s) => {
                assert!(s.matches(&Version::from_str("3.11.0").unwrap()));
                assert!(!s.matches(&Version::from_str("3.10.0").unwrap()));
            }
            _ => panic!("expected Spec(3.11.*), got Any"),
        }

        // python range constraint
        let c = python_constraint_from_depends(&[json!("python >=3.9")]);
        match c {
            PythonConstraint::Spec(s) => {
                assert!(s.matches(&Version::from_str("3.11").unwrap()));
                assert!(s.matches(&Version::from_str("3.9").unwrap()));
                assert!(!s.matches(&Version::from_str("3.8").unwrap()));
            }
            _ => panic!("expected Spec(>=3.9), got Any"),
        }

        // No python dep at all -> Any (don't reject on python)
        let c = python_constraint_from_depends(&[json!("zlib >=1.2")]);
        assert!(matches!(c, PythonConstraint::Any));

        // Empty depends -> Any
        let c = python_constraint_from_depends(&[]);
        assert!(matches!(c, PythonConstraint::Any));
    }
}

/// Convenience wrapper used by the auto-bundle path: probe many
/// (name, spec) pairs against the same channel set, bounded-parallel
/// at `concurrency=16`. Returns the results in the order they
/// complete; the caller matches by `(package, spec)`.
/// `target_python` is forwarded to each `probe` call so the python-
/// awareness filter applies uniformly.
pub async fn probe_many(
    channels: &[ChannelUrl],
    specs: Vec<(String, String)>,
    target_python: Option<&str>,
) -> Vec<ProbeResult> {
    use futures::stream::{self, StreamExt};
    let tp_owned: Option<String> = target_python.map(|s| s.to_string());
    stream::iter(specs)
        .map(|(name, spec)| {
            let chans = channels.to_vec();
            let tp = tp_owned.clone();
            async move { probe(&chans, &name, &spec, tp.as_deref()).await }
        })
        .buffer_unordered(16)
        .collect::<Vec<_>>()
        .await
}
