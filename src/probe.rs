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
//! answers that question per (channel, name, spec, target_python)
//! tuple and lets the caller fall back to auto-bundling the PyPI
//! wheel.
//!
//! v0.22.0: backed by `repodata.json[.zst]` directly. The prior
//! implementation hit `prefix.dev/api/v1/channels/<c>/packages/<p>/
//! variants?limit=500` which a) returned 404 on the current API
//! revision (silently false-unsat on every probe!) and b) pagination
//! cutoff at 500 missed builds for popular packages even when the
//! endpoint worked. Repodata is the canonical, conda-native source of
//! truth -- same data the conda solver itself sees, no rate limits,
//! ~10MB per (channel, subdir) compressed.
//!
//! Parallelization:
//!   * Per probe call, channel x subdir fetches go through
//!     `futures::join_all` so multi-channel workspaces don't pay the
//!     latency serially.
//!   * Per-process in-memory cache (Arc<RepodataIndex>) means the
//!     first probe per (channel, subdir) pays the fetch+parse cost
//!     and every later probe in the same retread invocation is
//!     in-memory hash lookups.
//!   * Per-disk cache (`~/.cache/rattler/cache/retread-repodata/`)
//!     survives across retread invocations with a 30-minute TTL.

use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime};

use anyhow::{anyhow, Context, Result};
use rattler_conda_types::{ChannelUrl, ParseStrictness, Version, VersionSpec};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex as AsyncMutex;

const REPODATA_TTL: Duration = Duration::from_secs(30 * 60);
const HTTP_USER_AGENT: &str = concat!("pixi-build-retread/", env!("CARGO_PKG_VERSION"));

/// A single probe's outcome, recorded in the audit so users can see
/// which deps were validated against which channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    pub package: String,
    pub spec: String,
    /// Channels actually consulted (after `(channel, subdir)` fan-out;
    /// shown as the bare channel URL strings). If empty, NO repodata
    /// could be fetched -- treat as indecisive, not unsat.
    pub channels_consulted: Vec<String>,
    /// Some(true) iff at least one consulted (channel, subdir) had a
    /// version+python-compatible match; Some(false) iff every
    /// consulted channel was reachable AND has the package AND none
    /// of its candidates satisfied; None iff no channel could be
    /// consulted.
    pub satisfiable: Option<bool>,
    /// Diagnostic match count across all consulted (channel, subdir)s.
    pub matching_candidates: usize,
}

impl ProbeResult {
    pub fn is_satisfied(&self) -> bool {
        self.satisfiable == Some(true)
    }
    pub fn is_definitively_unsatisfied(&self) -> bool {
        self.satisfiable == Some(false)
    }
}

/// v0.22.0+ probe: against the workspace's conda channels' repodata.
/// Fetches `<channel>/<target_subdir>/repodata.json.zst` AND
/// `<channel>/noarch/repodata.json.zst` per channel in parallel,
/// caches the parsed index, and answers in memory.
///
/// `target_python = None` skips the python filter (legacy behavior).
/// `target_subdir = None` defaults to `"linux-64"` (only subdir
/// retread targets today). Pass explicitly if/when retread learns
/// other platforms.
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
                "probe: could not parse spec; returning indecisive",
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
                    "probe: could not parse target python; skipping python filter",
                );
                None
            }
        }
    });

    // Build the full (channel, subdir) work list. linux-64 is the
    // only retread target today; noarch lives alongside it on every
    // channel. We fan out both per channel.
    let target_subdir = "linux-64";
    let mut work: Vec<(String, String)> = Vec::new();
    for channel in channels {
        let url = channel.url().as_str().trim_end_matches('/').to_string();
        work.push((url.clone(), target_subdir.to_string()));
        work.push((url, "noarch".to_string()));
    }

    // Parallel fetch all (channel, subdir) repodatas. Each
    // get_repodata is in-memory-cached after first call, so repeated
    // probes within a single retread process pay zero network cost.
    use futures::stream::{FuturesUnordered, StreamExt};
    let mut futs: FuturesUnordered<_> = work
        .into_iter()
        .map(|(channel_url, subdir)| async move {
            let res = get_repodata(&channel_url, &subdir).await;
            (channel_url, subdir, res)
        })
        .collect();

    let mut consulted: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();
    let mut total_matching: usize = 0;
    let mut any_reachable: bool = false;
    while let Some((channel_url, subdir, res)) = futs.next().await {
        match res {
            Ok(idx) => {
                any_reachable = true;
                consulted.insert(format!("{channel_url}/{subdir}"));
                if let Some(records) = idx.by_name.get(package) {
                    let matches = records
                        .iter()
                        .filter(|v| parsed_spec.matches(&v.version))
                        .filter(|v| match (&target_py_version, &v.python_constraint) {
                            (None, _) => true,
                            (Some(_), PythonConstraint::Any) => true,
                            (Some(tp), PythonConstraint::Spec(s)) => s.matches(tp),
                        })
                        .count();
                    total_matching += matches;
                }
            }
            Err(e) => {
                tracing::debug!(
                    channel = %channel_url, subdir = %subdir, package = %package,
                    error = %format!("{e:#}"),
                    "probe: repodata fetch failed; treating channel as not-consulted",
                );
            }
        }
    }

    let satisfiable = if any_reachable {
        Some(total_matching > 0)
    } else {
        None
    };
    ProbeResult {
        package: package.to_string(),
        spec: spec.to_string(),
        channels_consulted: consulted.into_iter().collect(),
        satisfiable,
        matching_candidates: total_matching,
    }
}

/// What this conda package build says about which python it supports.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum PythonConstraint {
    /// No `python` / `python_abi` dep listed. Either a noarch non-
    /// python package, or a build whose python metadata we couldn't
    /// recover. Either way: don't reject on python.
    Any,
    /// Parsed `python >=3.9` / `python_abi 3.11.*` / etc. The probe
    /// uses `spec.matches(target_python_version)`.
    Spec(VersionSpec),
}

/// One conda build (`packages` or `packages.conda` entry) reduced to
/// what the probe needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VariantInfo {
    version: Version,
    python_constraint: PythonConstraint,
    /// v0.32.0+: the raw `depends` array from the conda record. Kept
    /// so transitive-constraint extraction can walk each workspace
    /// conda dep's latest build and accumulate what IT requires.
    /// Strings look like `"numpy >=1.26,<2"`, `"python_abi 3.11.* *_cp311"`.
    depends: Vec<String>,
}

/// Parsed repodata for one (channel, subdir), indexed by package
/// name. Wrapped in Arc + reused across probe calls in the same
/// process.
struct RepodataIndex {
    by_name: HashMap<String, Vec<VariantInfo>>,
}

/// In-memory cache keyed by `(channel_url, subdir)`. Survives across
/// every probe call in a single retread process. Concurrent probe
/// calls coordinate via the AsyncMutex so multiple callers don't
/// double-fetch the same repodata.
fn cache() -> &'static AsyncMutex<HashMap<(String, String), Arc<RepodataIndex>>> {
    static CACHE: OnceLock<AsyncMutex<HashMap<(String, String), Arc<RepodataIndex>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| AsyncMutex::new(HashMap::new()))
}

/// Get the repodata index for `(channel_url, subdir)`. In-memory
/// hit -> Arc clone. Else disk-cache hit -> parse, install, Arc.
/// Else HTTP fetch + decompress + parse + write disk + install.
async fn get_repodata(channel_url: &str, subdir: &str) -> Result<Arc<RepodataIndex>> {
    let key = (channel_url.to_string(), subdir.to_string());
    {
        let guard = cache().lock().await;
        if let Some(idx) = guard.get(&key) {
            return Ok(idx.clone());
        }
    }

    let raw_data = if let Some(bytes) = read_disk_cache(channel_url, subdir).await? {
        bytes
    } else {
        let fetched = fetch_repodata_bytes(channel_url, subdir).await?;
        write_disk_cache(channel_url, subdir, &fetched).await.ok();
        fetched
    };

    let parsed: RepodataFile = serde_json::from_slice(&raw_data)
        .with_context(|| format!("parsing repodata for {channel_url}/{subdir}"))?;
    let idx = Arc::new(build_index(parsed));

    let mut guard = cache().lock().await;
    guard.insert(key, idx.clone());
    Ok(idx)
}

#[derive(Deserialize)]
struct RepodataFile {
    #[serde(default)]
    packages: HashMap<String, PackageRecord>,
    #[serde(default, rename = "packages.conda")]
    packages_conda: HashMap<String, PackageRecord>,
}

#[derive(Deserialize)]
struct PackageRecord {
    name: String,
    version: String,
    #[serde(default)]
    depends: Vec<String>,
}

fn build_index(file: RepodataFile) -> RepodataIndex {
    let mut by_name: HashMap<String, Vec<VariantInfo>> = HashMap::new();
    let all = file.packages.into_values().chain(file.packages_conda.into_values());
    for rec in all {
        let Ok(version) = Version::from_str(&rec.version) else {
            continue;
        };
        let python_constraint = python_constraint_from_depends(&rec.depends);
        by_name
            .entry(rec.name)
            .or_default()
            .push(VariantInfo {
                version,
                python_constraint,
                depends: rec.depends,
            });
    }
    RepodataIndex { by_name }
}

/// Walk the build's `depends` list, find any entry naming `python` or
/// `python_abi`, parse its version constraint. Returns
/// `PythonConstraint::Any` if absent.
fn python_constraint_from_depends(deps: &[String]) -> PythonConstraint {
    for dep in deps {
        let trimmed = dep.trim();
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

/// HTTP fetch of `<channel_url>/<subdir>/repodata.json.zst`,
/// decompressed to the raw uncompressed JSON bytes. Falls back to
/// `repodata.json` if the .zst variant isn't served. Errors propagate
/// to the caller, which logs + treats the channel as unreachable.
async fn fetch_repodata_bytes(channel_url: &str, subdir: &str) -> Result<Vec<u8>> {
    let client = reqwest::Client::builder()
        .user_agent(HTTP_USER_AGENT)
        .build()
        .context("building reqwest client")?;
    let zst_url = format!("{channel_url}/{subdir}/repodata.json.zst");
    tracing::info!(url = %zst_url, "probe: fetching repodata");
    let resp = client
        .get(&zst_url)
        .send()
        .await
        .with_context(|| format!("GET {zst_url}"))?;
    if resp.status().is_success() {
        let bytes = resp
            .bytes()
            .await
            .with_context(|| format!("reading body of {zst_url}"))?;
        let decoded = zstd::decode_all(bytes.as_ref())
            .with_context(|| format!("zstd-decoding {zst_url}"))?;
        return Ok(decoded);
    }
    // Fall back to uncompressed.
    let plain_url = format!("{channel_url}/{subdir}/repodata.json");
    tracing::info!(url = %plain_url, "probe: .zst not available, falling back to plain repodata");
    let resp = client
        .get(&plain_url)
        .send()
        .await
        .with_context(|| format!("GET {plain_url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {plain_url}"))?;
    let bytes = resp
        .bytes()
        .await
        .with_context(|| format!("reading body of {plain_url}"))?;
    Ok(bytes.to_vec())
}

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
        PathBuf::from(home).join(".cache").join("rattler").join("cache")
    } else {
        std::env::temp_dir().join("retread-cache")
    }
}

async fn read_disk_cache(channel_url: &str, subdir: &str) -> Result<Option<Vec<u8>>> {
    let path = disk_cache_path(channel_url, subdir);
    let meta = match tokio::fs::metadata(&path).await {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("stat {}", path.display())),
    };
    let mtime = meta
        .modified()
        .with_context(|| format!("mtime {}", path.display()))?;
    let age = SystemTime::now().duration_since(mtime).unwrap_or(Duration::ZERO);
    if age > REPODATA_TTL {
        return Ok(None);
    }
    let bytes = tokio::fs::read(&path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;
    Ok(Some(bytes))
}

async fn write_disk_cache(channel_url: &str, subdir: &str, bytes: &[u8]) -> Result<()> {
    let path = disk_cache_path(channel_url, subdir);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }
    tokio::fs::write(&path, bytes)
        .await
        .with_context(|| format!("writing cache {}", path.display()))?;
    Ok(())
}

/// v0.32.0+: fetch the `depends` array of the LATEST target-python-
/// compatible build of `(package, spec)` across the given channels.
/// Used by the workspace-transitive-constraint extractor to learn
/// what each workspace-declared conda dep requires of OTHER deps
/// (e.g. `ros-humble-joint-state-publisher` declares `numpy >=1.26,<2`).
///
/// Returns the raw `depends` strings (`["numpy >=1.26,<2", "python_abi
/// 3.11.* *_cp311", ...]`) of the highest-version matching build that
/// also satisfies the target python. Returns an empty Vec if no
/// channel had a matching build, or if all candidates failed the
/// python filter.
pub async fn fetch_latest_build_depends(
    channels: &[ChannelUrl],
    package: &str,
    spec: &str,
    target_python: Option<&str>,
) -> Vec<String> {
    let parsed_spec = match VersionSpec::from_str(spec, ParseStrictness::Lenient) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let target_py_version: Option<Version> = target_python.and_then(|tp| Version::from_str(tp).ok());

    let target_subdir = "linux-64";
    let mut work: Vec<(String, String)> = Vec::new();
    for channel in channels {
        let url = channel.url().as_str().trim_end_matches('/').to_string();
        work.push((url.clone(), target_subdir.to_string()));
        work.push((url, "noarch".to_string()));
    }

    use futures::stream::{FuturesUnordered, StreamExt};
    let mut futs: FuturesUnordered<_> = work
        .into_iter()
        .map(|(channel_url, subdir)| async move {
            get_repodata(&channel_url, &subdir).await.ok()
        })
        .collect();

    // Walk every (channel, subdir), keep the candidate with the
    // highest version that satisfies spec + python filter.
    let mut best: Option<(Version, Vec<String>)> = None;
    while let Some(maybe_idx) = futs.next().await {
        let Some(idx) = maybe_idx else { continue };
        let Some(records) = idx.by_name.get(package) else { continue };
        for v in records {
            if !parsed_spec.matches(&v.version) {
                continue;
            }
            let py_ok = match (&target_py_version, &v.python_constraint) {
                (None, _) => true,
                (Some(_), PythonConstraint::Any) => true,
                (Some(tp), PythonConstraint::Spec(s)) => s.matches(tp),
            };
            if !py_ok {
                continue;
            }
            match &best {
                Some((best_v, _)) if best_v >= &v.version => {}
                _ => best = Some((v.version.clone(), v.depends.clone())),
            }
        }
    }
    best.map(|(_, deps)| deps).unwrap_or_default()
}

/// Convenience wrapper used by the auto-bundle path: probe many
/// (name, spec) pairs against the same channel set. Each probe call
/// already parallelizes per-channel fetches; this layer adds
/// bounded-parallel probing across (name, spec) pairs at
/// `concurrency=16`. With the in-memory cache, only the FIRST probe
/// pays the per-(channel, subdir) fetch cost; later probes are
/// in-memory hashmap lookups.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_result_classifies_outcomes() {
        let yes = ProbeResult {
            package: "x".into(),
            spec: ">=1".into(),
            channels_consulted: vec!["c/linux-64".into()],
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
        // Indecisive is NOT "definitively unsatisfied". Callers must
        // not trigger a fallback on indecisive results -- a network
        // outage shouldn't reshape routing.
        assert!(!indecisive.is_definitively_unsatisfied());
    }

    #[test]
    fn python_constraint_from_depends_basic() {
        let c = python_constraint_from_depends(&["python_abi 3.11.* *_cp311".into()]);
        match c {
            PythonConstraint::Spec(s) => {
                assert!(s.matches(&Version::from_str("3.11.0").unwrap()));
                assert!(!s.matches(&Version::from_str("3.10.0").unwrap()));
            }
            _ => panic!("expected Spec"),
        }

        let c = python_constraint_from_depends(&["python >=3.9".into()]);
        match c {
            PythonConstraint::Spec(s) => {
                assert!(s.matches(&Version::from_str("3.11").unwrap()));
                assert!(!s.matches(&Version::from_str("3.8").unwrap()));
            }
            _ => panic!("expected Spec"),
        }

        let c = python_constraint_from_depends(&["zlib >=1.2".into()]);
        assert!(matches!(c, PythonConstraint::Any));
    }

    #[test]
    fn build_index_groups_by_name() {
        use serde_json::json;
        let raw = json!({
            "packages": {
                "matplotlib-3.7.0-py311h_0.tar.bz2": {
                    "name": "matplotlib",
                    "version": "3.7.0",
                    "depends": ["python_abi 3.11.* *_cp311"],
                }
            },
            "packages.conda": {
                "matplotlib-3.8.0-py312h_0.conda": {
                    "name": "matplotlib",
                    "version": "3.8.0",
                    "depends": ["python_abi 3.12.* *_cp312"],
                },
                "numpy-1.26.0-py311h_0.conda": {
                    "name": "numpy",
                    "version": "1.26.0",
                    "depends": ["python_abi 3.11.* *_cp311"],
                }
            }
        });
        let file: RepodataFile = serde_json::from_value(raw).unwrap();
        let idx = build_index(file);
        assert_eq!(idx.by_name.get("matplotlib").map(|v| v.len()), Some(2));
        assert_eq!(idx.by_name.get("numpy").map(|v| v.len()), Some(1));
        // Verify python filter applied to the matplotlib variants:
        // 3.7.0 (cp311) matches py3.11, 3.8.0 (cp312) does not.
        let mpl = idx.by_name.get("matplotlib").unwrap();
        let py311 = Version::from_str("3.11.0").unwrap();
        let matches: usize = mpl
            .iter()
            .filter(|v| match &v.python_constraint {
                PythonConstraint::Spec(s) => s.matches(&py311),
                PythonConstraint::Any => true,
            })
            .count();
        assert_eq!(matches, 1, "only matplotlib 3.7.0 should match py3.11");
    }
}
