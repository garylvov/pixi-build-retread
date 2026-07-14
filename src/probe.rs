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

use std::str::FromStr;

use rattler_conda_types::{ChannelUrl, ParseStrictness, Version, VersionSpec};
use serde::{Deserialize, Serialize};

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

    let target_py_version: Option<Version> =
        target_python.and_then(|tp| match Version::from_str(tp) {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::debug!(
                    target_python = %tp, error = %e,
                    "probe: could not parse target python; skipping python filter",
                );
                None
            }
        });

    // Build the full (channel, subdir) work list. linux-64 is the
    // only retread target today; noarch lives alongside it on every
    // channel. Both subdirs are fanned out via the shared helper so
    // this list is always consistent with sparse_pairs.
    let target_subdir = "linux-64";
    let work = crate::repodata::channel_subdir_pairs(channels, target_subdir);

    // Parallel fetch all (channel, subdir) repodatas. Each
    // get_repodata is in-memory-cached after first call, so repeated
    // probes within a single retread process pay zero network cost.
    use futures::stream::{FuturesUnordered, StreamExt};
    let mut futs: FuturesUnordered<_> = work
        .into_iter()
        .map(|(channel_url, subdir)| async move {
            let res = candidates_for(&channel_url, &subdir, package).await;
            (channel_url, subdir, res)
        })
        .collect();

    let mut consulted: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut total_matching: usize = 0;
    let mut any_reachable: bool = false;
    while let Some((channel_url, subdir, res)) = futs.next().await {
        match res {
            Some(candidates) => {
                any_reachable = true;
                consulted.insert(format!("{channel_url}/{subdir}"));
                total_matching += candidates
                    .iter()
                    .filter(|v| parsed_spec.matches(&v.version))
                    .filter(|v| match (&target_py_version, &v.python_constraint) {
                        (None, _) => true,
                        (Some(_), PythonConstraint::Any) => true,
                        (Some(tp), PythonConstraint::Spec(s)) => s.matches(tp),
                    })
                    .count();
            }
            None => {
                tracing::debug!(
                    channel = %channel_url, subdir = %subdir, package = %package,
                    "probe: repodata unavailable; treating channel as not-consulted",
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

/// v1.5.0: candidates for ONE package name from one (channel,
/// subdir), via the shared memory-mapped sparse store in
/// `crate::repodata`. The old path parsed the ENTIRE repodata document
/// into a by-name index (seconds per 600MB channel file, and its
/// check-then-insert cache stampeded under 16-way batched probes:
/// a cold isaac6 run parsed the same document 32 times, ~125s).
/// The sparse handle binary-searches the name and parses only its
/// entries -- microseconds-to-milliseconds per lookup.
///
/// `None` = pair unreachable (caller treats as not-consulted);
/// `Some(vec)` = consulted, possibly with zero candidates.
async fn candidates_for(
    channel_url: &str,
    subdir: &str,
    package: &str,
) -> Option<Vec<VariantInfo>> {
    use rattler_repodata_gateway::sparse::PackageFormatSelection;
    let sparse = crate::repodata::sparse(channel_url, subdir).await?;
    let Ok(name) = rattler_conda_types::PackageName::try_from(package) else {
        // Unparseable conda name can't exist on the channel.
        return Some(Vec::new());
    };
    let records = match sparse.load_records(&name, PackageFormatSelection::default()) {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(
                channel = %channel_url, subdir = %subdir, package = %package, error = %e,
                "probe: sparse record load failed; treating pair as not-consulted",
            );
            return None;
        }
    };
    Some(
        records
            .into_iter()
            .map(|rec| VariantInfo {
                version: rec.package_record.version.version().clone(),
                python_constraint: python_constraint_from_depends(&rec.package_record.depends),
                depends: rec.package_record.depends.clone(),
            })
            .collect(),
    )
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
    let target_py_version: Option<Version> =
        target_python.and_then(|tp| Version::from_str(tp).ok());

    let target_subdir = "linux-64";
    // Reuse the shared fan-out helper so the work list here is
    // always identical to what probe() and sparse_pairs use.
    let work = crate::repodata::channel_subdir_pairs(channels, target_subdir);

    use futures::stream::{FuturesUnordered, StreamExt};
    let mut futs: FuturesUnordered<_> =
        work.into_iter()
            .map(|(channel_url, subdir)| async move {
                candidates_for(&channel_url, &subdir, package).await
            })
            .collect();

    // Walk every (channel, subdir), keep the candidate with the
    // highest version that satisfies spec + python filter.
    let mut best: Option<(Version, Vec<String>)> = None;
    while let Some(maybe_candidates) = futs.next().await {
        let Some(records) = maybe_candidates else {
            continue;
        };
        for v in &records {
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

/// A successful conda-route lookup: the best (highest) conda version
/// satisfying the queried spec, plus the `<channel>/<subdir>` label it
/// was found on. Consumed by the uv-closure auto-route loop
/// (spec-uv-restructure M2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteHit {
    /// Conda version string of the best matching build.
    pub version: String,
    /// `<channel_url>/<subdir>` the match came from (channel-priority
    /// order breaks version ties toward the earlier channel).
    pub channel: String,
    /// Route-time metadata-consistency check: the raw `depends` array
    /// of the matched build (same strings `VariantInfo::depends`
    /// carries, e.g. `"pillow <11.0,>=9.2.0"`). Lets the auto-route
    /// planner reject a route whose conda-forge repackage metadata
    /// contradicts the locked uv closure one level deep, without a
    /// second repodata fetch.
    pub depends: Vec<String>,
}

/// spec-uv-restructure M2: query the workspace channels' repodata for a
/// build of `package` satisfying `spec` (+ `target_python` compat) and
/// report WHICH channel had it at WHICH version. Same repodata source /
/// caches as [`probe`]; unlike `probe` this keeps per-channel
/// attribution so the auto-route log line can name the channel.
///
/// Returns `None` when no consulted channel has a satisfying build
/// (including the nothing-reachable case: auto-routing must never
/// trigger off a network outage).
pub async fn find_route(
    channels: &[ChannelUrl],
    package: &str,
    spec: &str,
    target_python: Option<&str>,
) -> Option<RouteHit> {
    let parsed_spec = VersionSpec::from_str(spec, ParseStrictness::Lenient).ok()?;
    let target_py_version: Option<Version> =
        target_python.and_then(|tp| Version::from_str(tp).ok());

    let target_subdir = "linux-64";
    let work = crate::repodata::channel_subdir_pairs(channels, target_subdir);

    // Sequential in channel-priority order is fine here: repodata is
    // in-memory-cached after the first probe of a build, so these are
    // hashmap lookups in the steady state.
    let mut best: Option<(Version, RouteHit)> = None;
    for (channel_url, subdir) in work {
        let Some(candidates) = candidates_for(&channel_url, &subdir, package).await else {
            continue;
        };
        for v in &candidates {
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
            let better = match &best {
                Some((best_v, _)) => &v.version > best_v,
                None => true,
            };
            if better {
                best = Some((
                    v.version.clone(),
                    RouteHit {
                        version: v.version.to_string(),
                        channel: format!("{channel_url}/{subdir}"),
                        depends: v.depends.clone(),
                    },
                ));
            }
        }
    }
    best.map(|(_, hit)| hit)
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
        // Indecisive is distinct from a proven miss. Routing callers still
        // fail closed unless `is_satisfied()` is backed by a real candidate;
        // other callers can preserve the outage-vs-unsat distinction.
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
    fn variant_python_filter_logic() {
        // The python-compat filter the probe applies to candidates,
        // exercised on VariantInfo directly (the sparse store now
        // supplies these; the old full-parse RepodataFile/build_index
        // path is gone).
        let mpl_311 = VariantInfo {
            version: Version::from_str("3.7.0").unwrap(),
            python_constraint: python_constraint_from_depends(
                &["python_abi 3.11.* *_cp311".into()],
            ),
            depends: vec![],
        };
        let mpl_312 = VariantInfo {
            version: Version::from_str("3.8.0").unwrap(),
            python_constraint: python_constraint_from_depends(
                &["python_abi 3.12.* *_cp312".into()],
            ),
            depends: vec![],
        };
        let py311 = Version::from_str("3.11.0").unwrap();
        let matches: usize = [&mpl_311, &mpl_312]
            .iter()
            .filter(|v| match &v.python_constraint {
                PythonConstraint::Spec(s) => s.matches(&py311),
                PythonConstraint::Any => true,
            })
            .count();
        assert_eq!(matches, 1, "only the cp311 build should match py3.11");
    }
}
