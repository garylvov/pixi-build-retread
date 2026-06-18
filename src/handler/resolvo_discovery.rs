// PR-1b is a standalone module not yet wired into production code.
// Dead-code warnings are expected until PR-1c wires up the DependencyProvider.
#![allow(dead_code)]

//! PR-1b: Discovery fixpoint + candidate Pool for the resolvo solver.
//!
//! resolvo 0.10.3 is pseudo-async: its `DependencyProvider` callbacks are
//! polled exactly once and panic if they yield.  This means ALL PyPI
//! metadata, wheel artifacts, sdist builds, and conda-route decisions must be
//! gathered into an in-memory [`DiscoveryPool`] BEFORE the synchronous solve.
//! This module implements that gathering pass.
//!
//! ## Design
//!
//! ### Discovery fixpoint
//!
//! Starting from the primary wheel's `Requires-Dist`, the pass:
//! 1. Maintains a `frontier: BTreeSet<String>` of canonical PyPI names not yet
//!    visited, and a `visited: HashSet<String>` of names already processed.
//! 2. For each name in the frontier:
//!    - Calls [`pypi::list_all_versions`] (PR-1a) to enumerate every version
//!      that has a compatible wheel or sdist.
//!    - For **wheel** candidates: fetches METADATA (PEP 658 sidecar preferred,
//!      full download fallback) and parses `Requires-Dist`.
//!    - For **sdist-only** candidates: builds the sdist into a wheel via
//!      [`source_build::build_wheel_from_sdist_url`] and reads the resulting
//!      wheel's METADATA.  This is necessary because resolvo may select an
//!      sdist-only version; its metadata (deps) must be known before solving.
//!    - Evaluates PEP 508 markers against the fixed [`MarkerEnvironment`] for
//!      the target platform + python version.
//!    - Adds any dep whose marker is satisfied and whose canonical name has not
//!      yet been visited to the frontier.
//! 3. Records every `(dep_conda_name, spec_string)` edge into the
//!    **probe-needs set** for the conda-route memo.
//! 4. Repeats until the frontier is empty (fixpoint).
//!
//! ### Conda-route memo
//!
//! After discovery, the pass batches all probe-need `(conda_name, spec)` pairs
//! through [`crate::probe::probe_many`] (16-way concurrent) and stores the
//! routing decisions in a `HashMap<(String,String), RouteDecision>`.
//!
//! **Two spec shapes per edge** are probed (grizzly amendment 2):
//! - The **raw edge spec** from `Requires-Dist` (e.g. `>=1.2,<2`), normalised
//!   the same way the BFS does: spaces after commas stripped, empty → `"*"`.
//! - The **version-widened spec** for every candidate version of the parent
//!   (`widen_exact(candidate_version, relax)`, i.e. the `probe_spec_for` shape
//!   used by `auto_bundle_transitives`). This covers the case where the solver
//!   picks a specific version and the run-dep emission probes at that version's
//!   widened form.
//!
//! ### Pool record
//!
//! Each candidate becomes a [`PoolRecord`] carrying:
//! - Version + parsed metadata (requires_dist).
//! - Wheel URL + sha256 + filename + index.
//! - Provenance mirror of [`super::ResolvedWheel`]: `upstream_url` (suppressed
//!   for sdist-built), `sdist_source` (set for sdist-built), `git_source`
//!   (always `None` here — git/URL deps are primary-entry-only).
//!
//! ### What is NOT yet wired
//!
//! PR-1b is additive and zero-default-risk.  Nothing in `resolve_bundle` is
//! changed.  The `DiscoveryPool` type is defined and tested here; PR-1c wires
//! it into a `DependencyProvider` + sync resolvo solve.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, anyhow, bail};
use rattler_conda_types::ChannelUrl;
use uv_pep508::uv_pep440::{Version, VersionSpecifiers};

use crate::config::RelaxPolicy;
use crate::lock::SdistWheelSource;
use crate::pypi::{self, WheelTarget};
use crate::relax::{canonical_conda_name, marker_env_for, widen_exact};
use crate::wheel::WheelMetadata;

use super::PypiToCondaMap;
use super::auto_bundle::{metadata_preferring_sidecar, pick_conda_target};

// ── Route decision ────────────────────────────────────────────────────────────

/// The routing decision for a single `(dep_conda_name, spec)` edge.
///
/// Mirrors the BFS `routed_to_conda` flag: anything that is NOT
/// `RouteDecision::Bundle` stays on the conda side and is recorded as a
/// conda run-dep; `Bundle` deps are resolved by the PyPI solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteDecision {
    /// Conda probe was satisfied or indecisive (short-circuit): keep on conda.
    KeepConda,
    /// Conda probe was definitively unsatisfied AND name-level probe was also
    /// unsatisfied (or name is not in conda at all): bundle via PyPI.
    Bundle,
}

// ── Pool record ───────────────────────────────────────────────────────────────

/// One `(name, version)` candidate in the pre-solve Pool.
///
/// Carries everything the downstream Bundle construction needs so the sync
/// resolvo solve never performs I/O (grizzly amendment 3).
///
/// Provenance rules mirror `resolve_bundle`'s `ResolvedWheel`:
/// - `upstream_url`: the pristine HTTPS index URL of the wheel.  **Suppressed**
///   (`None`) for sdist-built candidates; callers use `sdist_source` instead.
/// - `sdist_source`: `Some` only when the wheel was built from a PyPI sdist.
/// - `git_source`: always `None` here (git/URL entries are primary-only and
///   materialise via `materialize_and_rewrite`, not this discovery pass).
#[derive(Debug, Clone)]
pub struct PoolRecord {
    /// PEP 503-normalised PyPI project name (e.g. `"torch"`).
    pub pypi_name: String,
    /// Parsed PEP 440 version of this candidate.
    pub version: Version,
    /// Resolved wheel URL.  For sdist-built candidates this is the `file://`
    /// path of the built wheel; for index candidates it is the HTTPS URL.
    pub wheel_url: url::Url,
    /// SHA-256 of the wheel file.  `None` when not yet known (e.g. sdist-built
    /// wheels whose hash is computed later by `fetch_wheel`).
    pub sha256: Option<String>,
    /// Whether the index advertised a PEP 658/714 metadata sidecar.
    pub has_metadata_sidecar: bool,
    /// Wheel filename (e.g. `"torch-2.7.0-cp311-cp311-manylinux_2_28_x86_64.whl"`).
    pub filename: String,
    /// PEP 503 simple-index base URL this wheel was resolved from.
    pub index: String,
    /// Parsed wheel METADATA: name, version, requires_dist, etc.
    pub metadata: WheelMetadata,
    /// Pristine upstream HTTPS index URL (NOT the local `file://` path).
    /// `None` for sdist-built candidates (use `sdist_source` instead).
    pub upstream_url: Option<url::Url>,
    /// Sdist provenance when this wheel was built from a source distribution.
    /// `None` for normal index-wheel candidates.
    pub sdist_source: Option<SdistWheelSource>,
}

// ── Conda-route memo ──────────────────────────────────────────────────────────

/// The conda-route memo populated after the discovery fixpoint.
///
/// Keyed by `(canonical_conda_name, normalised_spec_string)` — both the raw
/// edge-spec form AND the version-widened form for every discovered candidate
/// version.  The solver's `get_dependencies` must look up every emitted edge
/// here; a miss is fail-closed (see [`DiscoveryPool::route`]).
pub type CondaRouteMemo = HashMap<(String, String), RouteDecision>;

// ── Discovery Pool ─────────────────────────────────────────────────────────────

/// The complete pre-computed candidate set produced by the discovery fixpoint.
///
/// Passed to the resolvo `DependencyProvider` (PR-1c), which uses only O(1)
/// in-memory lookups during the sync solve.
/// A candidate version that was excluded from the pool due to a build failure
/// or because no versions were found on the index. Used by the A/B oracle to
/// distinguish resolvo Unsolvable caused by build-failure exclusions vs.
/// genuine version conflicts.
#[derive(Debug, Clone)]
pub struct ExcludedCandidate {
    /// PEP 503-normalised PyPI project name.
    pub pypi_name: String,
    /// Version string of the excluded candidate.
    pub version: String,
    /// Human-readable reason for exclusion.
    pub reason: String,
}

#[derive(Debug, Default, Clone)]
pub struct DiscoveryPool {
    /// All discovered `(name, version)` candidates, ordered by
    /// `(pypi_name, version desc)` for deterministic iteration.
    ///
    /// Outer key: canonical PyPI name.  Inner vec: all versions, sorted
    /// highest-first.
    pub candidates: BTreeMap<String, Vec<PoolRecord>>,

    /// Pre-computed conda-route decisions.
    pub conda_route_memo: CondaRouteMemo,

    /// Canonical names that were identified as conda-routed AND not recursed
    /// into.  The solver records these as conda run-deps.
    pub conda_routed_names: HashSet<String>,

    /// PR-2: candidates excluded from the pool due to sdist build failures or
    /// no available versions. The A/B oracle uses this to demote resolvo
    /// Unsolvable errors that are caused by these exclusions rather than
    /// genuine version conflicts.
    pub excluded_sdist_builds: Vec<ExcludedCandidate>,
}

impl DiscoveryPool {
    /// Look up the route decision for `(conda_name, spec)`.
    ///
    /// Returns `None` on a memo miss.  The caller (PR-1c `get_dependencies`)
    /// should treat a miss as an error (fail-closed) rather than guessing.
    pub fn route(&self, conda_name: &str, spec: &str) -> Option<&RouteDecision> {
        self.conda_route_memo
            .get(&(conda_name.to_string(), spec.to_string()))
    }

    /// All candidate versions for `pypi_name`, highest-first.
    pub fn versions_for(&self, pypi_name: &str) -> &[PoolRecord] {
        self.candidates
            .get(pypi_name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

// ── Discovery parameters ──────────────────────────────────────────────────────

/// Everything the discovery pass needs that isn't the primary's
/// `requires_dist`.
///
/// This bundles the same parameters `resolve_bundle` already has at call time,
/// so the eventual wiring (PR-1c) is a thin adapter layer.
pub struct DiscoveryParams<'a> {
    /// The PEP 503 simple-index URL for the primary entry (and default for
    /// all transitive deps unless a dep-level index is negotiated).
    pub index: &'a str,
    /// Wheel target platform + python.
    pub target: &'a WheelTarget,
    /// Working directory for metadata fetches and sdist builds.
    pub download_dir: &'a Path,
    /// Relax policy used to compute the version-widened probe spec for each
    /// discovered candidate (grizzly amendment 2, spec-shape B).
    pub relax: RelaxPolicy,
    /// Workspace conda channels — passed to the conda-route probe.
    pub conda_channels: &'a [ChannelUrl],
    /// User name-map (overrides parselmouth; see `pick_conda_target`).
    pub name_map: &'a BTreeMap<String, String>,
    /// Parselmouth inverted map (PyPI name → conda candidate names).
    pub pypi_to_conda: &'a PypiToCondaMap,
    /// Hard cap on BFS depth to prevent runaway fixpoints on pathological
    /// indexes.  Typical well-behaved packages resolve in < 10 levels.
    pub max_iterations: usize,
    /// PR-2: retread-conda-deps force-list. Names canonicalized and intersected
    /// with discovered requires_dist names to populate `conda_routed_names`
    /// symmetrically with the BFS force-list union (mod.rs:3280-3300).
    /// Pass `&[]` when not using the A/B oracle.
    pub conda_deps: &'a [String],
}

impl<'a> DiscoveryParams<'a> {
    /// Sensible default iteration cap.
    pub const DEFAULT_MAX_ITERATIONS: usize = 200;
}

// ── Edge record (internal) ────────────────────────────────────────────────────

/// A single `Requires-Dist` edge that will need a conda-route probe.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ProbeNeed {
    /// Conda name of the child dep (already `canonical_conda_name`-normalized).
    conda_name: String,
    /// Conda target name (from `pick_conda_target`). May differ from
    /// `conda_name` for aliased packages (e.g. `torch` → `pytorch`).
    conda_target: String,
    /// Raw edge spec normalised to conda matchspec format (spaces after commas
    /// stripped; empty → `"*"`).  Spec-shape A in grizzly amendment 2.
    raw_spec: String,
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// Run the discovery fixpoint and return the complete [`DiscoveryPool`].
///
/// This is the async pre-pass that produces everything resolvo needs before
/// the sync solve.  It is called from `resolve_bundle` (PR-1c) inside a normal
/// async context; the sync solve follows inside `spawn_blocking`.
pub async fn run_discovery(
    primary_requires_dist: &[String],
    params: &DiscoveryParams<'_>,
) -> Result<DiscoveryPool> {
    let marker_env = marker_env_for(&params.target.conda_subdir, &params.target.python_version)
        .context("building marker environment for discovery pass")?;

    let mut pool = DiscoveryPool::default();

    // Names whose candidates + metadata have been fully fetched.
    let mut visited: HashSet<String> = HashSet::new();

    // Names queued for the next iteration.
    let mut frontier: BTreeSet<String> = BTreeSet::new();

    // All probe needs collected during the fixpoint.  Keyed by (conda_target,
    // spec) to deduplicate.  The value is a ProbeNeed carrying the canonical
    // pypi/conda name for the route-decision insert step.
    let mut probe_needs: HashMap<(String, String), ProbeNeed> = HashMap::new();

    // ── Seed: parse primary's requires_dist ───────────────────────────────────
    //
    // The primary wheel itself is NOT added to the pool here (it's already
    // materialised upstream and carried as `Bundle.primary`); we only need its
    // dependency edges to seed the frontier.
    seed_frontier_from_requires_dist(
        primary_requires_dist,
        &[],
        &marker_env,
        &visited,
        &mut frontier,
        &mut probe_needs,
        params,
    );

    // ── Fixpoint loop ─────────────────────────────────────────────────────────
    let mut iterations = 0usize;
    while !frontier.is_empty() {
        iterations += 1;
        if iterations > params.max_iterations {
            bail!(
                "resolvo discovery: iteration cap ({}) exceeded — possible dependency cycle or \
                 runaway transitive closure; frontier at cap: {:?}",
                params.max_iterations,
                frontier.iter().take(8).collect::<Vec<_>>(),
            );
        }

        // Drain the current frontier into a sorted vec for deterministic order.
        let current: Vec<String> = std::mem::take(&mut frontier).into_iter().collect();

        for pypi_name in &current {
            if visited.contains(pypi_name.as_str()) {
                continue;
            }
            visited.insert(pypi_name.clone());

            // Fetch all versions for this name.
            let all_versions = pypi::list_all_versions(params.index, pypi_name, params.target)
                .await
                .with_context(|| format!("discovery: list_all_versions for `{pypi_name}`"))?;

            if all_versions.is_empty() {
                tracing::debug!(name = %pypi_name, "discovery: no versions found on index; skipping");
                pool.excluded_sdist_builds.push(ExcludedCandidate {
                    pypi_name: pypi_name.clone(),
                    version: String::new(),
                    reason: "no versions found on index".to_string(),
                });
                continue;
            }

            let mut records_for_name: Vec<PoolRecord> = Vec::new();

            for candidate in all_versions {
                let version = candidate.version.clone();

                // ── Collect version-widened probe spec (grizzly amendment 2, shape B) ──
                // For each candidate version of this dep, compute the widened spec that
                // auto_bundle_transitives would emit (probe_spec_for(version, relax)),
                // and queue it as a probe need for every potential conda target.
                // We do this lazily: add to probe_needs keyed by (conda_target, widened_spec).
                // This is recorded here — the actual probe fires after the fixpoint.
                let widened_spec = {
                    let vs = version.to_string();
                    match uv_pep508::uv_pep440::Version::from_str(&vs) {
                        Ok(v) => widen_exact(&v, params.relax).unwrap_or_else(|| "*".to_string()),
                        Err(_) => "*".to_string(),
                    }
                };
                let conda_dep_name = canonical_conda_name(pypi_name);
                if let Some(conda_target) =
                    pick_conda_target(&conda_dep_name, params.name_map, params.pypi_to_conda)
                {
                    probe_needs
                        .entry((conda_target.clone(), widened_spec.clone()))
                        .or_insert(ProbeNeed {
                            conda_name: conda_dep_name.clone(),
                            conda_target,
                            raw_spec: widened_spec,
                        });
                }

                // ── Fetch METADATA for this candidate ─────────────────────────────────
                let (pool_record_opt, child_requires_dist) = match candidate.wheel {
                    Some(ref resolved_wheel) => {
                        // Normal wheel: fetch metadata (sidecar preferred).
                        let metadata =
                            metadata_preferring_sidecar(resolved_wheel, params.download_dir)
                                .await
                                .with_context(|| {
                                    format!(
                                        "discovery: fetching metadata for `{pypi_name}=={version}`"
                                    )
                                })?;
                        let requires_dist = metadata.requires_dist.clone();
                        let record = PoolRecord {
                            pypi_name: pypi_name.clone(),
                            version: version.clone(),
                            wheel_url: resolved_wheel.url.clone(),
                            sha256: resolved_wheel.sha256.clone(),
                            has_metadata_sidecar: resolved_wheel.has_metadata_sidecar,
                            filename: resolved_wheel.filename.clone(),
                            index: params.index.to_string(),
                            metadata,
                            upstream_url: Some(resolved_wheel.url.clone()),
                            sdist_source: None,
                        };
                        (Some(record), requires_dist)
                    }
                    None => {
                        // Sdist-only candidate.  Build the sdist so its METADATA
                        // (and the built wheel) are available before the sync solve.
                        match candidate.sdist_url {
                            Some(ref sdist_url) => {
                                let sdist_build_dir = params
                                    .download_dir
                                    .join("sdist-builds")
                                    .join(format!("{pypi_name}-{version}"));
                                match crate::source_build::build_wheel_from_sdist_url(
                                    sdist_url,
                                    &sdist_build_dir,
                                    &params.target.python_version,
                                )
                                .await
                                {
                                    Ok(wheel_path) => {
                                        let wheel_url = url::Url::from_file_path(&wheel_path)
                                            .map_err(|_| {
                                                anyhow!(
                                                    "sdist build produced path that cannot be \
                                                     converted to file:// URL: {}",
                                                    wheel_path.display()
                                                )
                                            })?;
                                        let metadata = tokio::task::spawn_blocking({
                                            let p = wheel_path.clone();
                                            move || crate::wheel::read_metadata(&p)
                                        })
                                        .await
                                        .context("sdist-built wheel metadata reader panicked")?
                                        .with_context(|| {
                                            format!(
                                                "reading metadata from sdist-built wheel: {}",
                                                wheel_path.display()
                                            )
                                        })?;
                                        let requires_dist = metadata.requires_dist.clone();
                                        let filename = wheel_path
                                            .file_name()
                                            .and_then(|n| n.to_str())
                                            .unwrap_or("unknown.whl")
                                            .to_string();
                                        let sdist_source = SdistWheelSource {
                                            index: params.index.to_string(),
                                            name: pypi_name.clone(),
                                            version: version.to_string(),
                                            sdist_url: sdist_url.to_string(),
                                        };
                                        let record = PoolRecord {
                                            pypi_name: pypi_name.clone(),
                                            version: version.clone(),
                                            wheel_url,
                                            sha256: None,
                                            has_metadata_sidecar: false,
                                            filename,
                                            index: params.index.to_string(),
                                            metadata,
                                            upstream_url: None,
                                            sdist_source: Some(sdist_source),
                                        };
                                        (Some(record), requires_dist)
                                    }
                                    Err(e) => {
                                        // Sdist build failed: log and skip this candidate.
                                        // It won't appear in the pool; resolvo won't select it.
                                        tracing::warn!(
                                            name = %pypi_name,
                                            version = %version,
                                            sdist_url = %sdist_url,
                                            error = %format!("{e:#}"),
                                            "discovery: sdist build failed; candidate excluded from pool",
                                        );
                                        pool.excluded_sdist_builds.push(ExcludedCandidate {
                                            pypi_name: pypi_name.clone(),
                                            version: version.to_string(),
                                            reason: format!("sdist build failed: {e:#}"),
                                        });
                                        (None, vec![])
                                    }
                                }
                            }
                            None => {
                                // Unreachable: list_all_versions only returns candidates
                                // with wheel OR sdist_url.
                                tracing::warn!(
                                    name = %pypi_name,
                                    version = %version,
                                    "discovery: candidate has neither wheel nor sdist; skipping",
                                );
                                (None, vec![])
                            }
                        }
                    }
                };

                if let Some(record) = pool_record_opt {
                    records_for_name.push(record);
                }

                // ── Expand child edges from this version's requires_dist ───────────────
                seed_frontier_from_requires_dist(
                    &child_requires_dist,
                    &[],
                    &marker_env,
                    &visited,
                    &mut frontier,
                    &mut probe_needs,
                    params,
                );
            }

            // Store all candidates for this name, highest-version first.
            // list_all_versions already returns highest-first; preserve that order.
            if !records_for_name.is_empty() {
                pool.candidates
                    .entry(pypi_name.clone())
                    .or_default()
                    .extend(records_for_name);
            }
        }
    }

    // ── Conda-route memo: batch all probe needs ───────────────────────────────
    //
    // After the fixpoint, we have the complete probe-needs set.  Fire all probes
    // in one batched call (16-way concurrent via probe_many) and populate the
    // memo.
    if !probe_needs.is_empty() {
        let specs_to_probe: Vec<(String, String)> = probe_needs
            .keys()
            .map(|(target, spec)| (target.clone(), spec.clone()))
            .collect();

        let results = crate::probe::probe_many(
            params.conda_channels,
            specs_to_probe,
            Some(&params.target.python_version),
        )
        .await;

        for result in &results {
            let key = (result.package.clone(), result.spec.clone());
            // Mirror the BFS two-probe routing logic:
            //   - definitively_unsatisfied (exact spec) → second probe at "*"
            //     is handled in a second pass below.
            //   - satisfied or indecisive → KeepConda.
            let decision = if result.is_definitively_unsatisfied() {
                // Will be resolved by the name-level second probe below.
                // For now mark as Bundle; overwrite if name-level is satisfied.
                RouteDecision::Bundle
            } else {
                RouteDecision::KeepConda
            };
            pool.conda_route_memo.insert(key, decision);
        }

        // Second pass: for all keys that are Bundle after the first probe,
        // fire a name-level probe at spec="*" and upgrade to KeepConda if
        // it is satisfied.
        let bundle_keys: Vec<(String, String)> = pool
            .conda_route_memo
            .iter()
            .filter(|(_, v)| **v == RouteDecision::Bundle)
            .map(|(k, _)| k.clone())
            .collect();

        if !bundle_keys.is_empty() {
            // Collect the unique conda_targets that need a name-level probe.
            let name_level_targets: BTreeSet<String> = bundle_keys
                .iter()
                .map(|(target, _)| target.clone())
                .collect();

            let name_level_specs: Vec<(String, String)> = name_level_targets
                .iter()
                .map(|target| (target.clone(), "*".to_string()))
                .collect();

            let name_level_results = crate::probe::probe_many(
                params.conda_channels,
                name_level_specs,
                Some(&params.target.python_version),
            )
            .await;

            // Build a quick lookup: conda_target → name-level satisfied?
            let name_level_sat: HashMap<&str, bool> = name_level_results
                .iter()
                .map(|r| (r.package.as_str(), r.is_satisfied()))
                .collect();

            // Upgrade to KeepConda for any bundle_key whose target has a
            // satisfied name-level probe.
            for (target, spec) in &bundle_keys {
                if name_level_sat
                    .get(target.as_str())
                    .copied()
                    .unwrap_or(false)
                {
                    pool.conda_route_memo
                        .insert((target.clone(), spec.clone()), RouteDecision::KeepConda);
                }
            }
        }
    }

    // ── Populate conda_routed_names ───────────────────────────────────────────
    //
    // A name is conda-routed if ALL its (target, spec) entries in the memo
    // resolve to KeepConda.  We use a simpler heuristic: if a name has ANY
    // KeepConda decision (regardless of spec), record it.  The solver will
    // consult the memo per-edge for the precise decision.
    for pn in &visited {
        let conda_name = canonical_conda_name(pn);
        if let Some(target) = pick_conda_target(&conda_name, params.name_map, params.pypi_to_conda)
        {
            let any_keep = probe_needs
                .keys()
                .filter(|(t, _)| t == &target)
                .any(|(t, s)| {
                    pool.conda_route_memo
                        .get(&(t.clone(), s.clone()))
                        .map(|d| *d == RouteDecision::KeepConda)
                        .unwrap_or(false)
                });
            if any_keep {
                pool.conda_routed_names.insert(conda_name);
            }
        }
    }

    // ── Force-list union (PR-2 symmetry) ─────────────────────────────────────
    //
    // The BFS force-list union (mod.rs) adds retread-conda-deps names to
    // conda_routed_acc when they appear in any bundled wheel's transitive
    // requires_dist.  Mirror that here: intersect the force-list with the set
    // of names discovered in this fixpoint and add them to conda_routed_names.
    if !params.conda_deps.is_empty() {
        let force_conda: std::collections::HashSet<String> = params
            .conda_deps
            .iter()
            .map(|n| canonical_conda_name(n))
            .collect();
        // All discovered pypi names (visited set), canonicalized.
        for pn in &visited {
            let canon = canonical_conda_name(pn);
            if force_conda.contains(&canon) {
                pool.conda_routed_names.insert(canon);
            }
        }
    }

    tracing::debug!(
        candidates = pool.candidates.len(),
        total_records = pool.candidates.values().map(|v| v.len()).sum::<usize>(),
        memo_entries = pool.conda_route_memo.len(),
        conda_routed = pool.conda_routed_names.len(),
        iterations,
        "resolvo discovery: fixpoint complete",
    );

    Ok(pool)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Parse `requires_dist` lines, evaluate markers, and add unseen dep names to
/// `frontier`.  Also registers raw-edge probe needs (grizzly amendment 2,
/// spec-shape A).
///
/// `extras_active` is the set of extras active on the *parent* (empty for base
/// deps; non-empty for extras-gated dep expansion — not yet used in PR-1b but
/// wired for PR-1c extras support).
fn seed_frontier_from_requires_dist(
    requires_dist: &[String],
    extras_active: &[String],
    marker_env: &uv_pep508::MarkerEnvironment,
    visited: &HashSet<String>,
    frontier: &mut BTreeSet<String>,
    probe_needs: &mut HashMap<(String, String), ProbeNeed>,
    params: &DiscoveryParams<'_>,
) {
    let extras_as_extra_names: Vec<uv_normalize::ExtraName> = extras_active
        .iter()
        .filter_map(|e| uv_normalize::ExtraName::from_owned(e.clone()).ok())
        .collect();

    for raw in requires_dist {
        let req: uv_pep508::Requirement = match uv_pep508::Requirement::from_str(raw) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(raw = %raw, error = %e, "discovery: skipping unparseable Requires-Dist line");
                continue;
            }
        };

        // Evaluate marker.  Skip if the marker is false for both the active
        // extras and the base (empty-extras) evaluation.
        let marker_ok = req
            .marker
            .evaluate(marker_env, extras_as_extra_names.as_slice())
            || req.marker.evaluate(marker_env, &[]);
        if !marker_ok {
            continue;
        }

        // Skip non-PyPI (URL/git) deps in the discovery frontier — they are
        // primary-entry-only and handled via materialize_and_rewrite.
        let specifiers = match &req.version_or_url {
            Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) => specs.clone(),
            None => VersionSpecifiers::default(),
            Some(uv_pep508::VersionOrUrl::Url(_)) => {
                tracing::debug!(
                    dep = %req.name,
                    "discovery: skipping URL-form Requires-Dist (not followed in discovery pass)",
                );
                continue;
            }
        };

        let pypi_name = req.name.to_string();
        let conda_dep_name = canonical_conda_name(&pypi_name);

        // Add to frontier if not yet visited.
        if !visited.contains(&conda_dep_name) {
            frontier.insert(pypi_name.clone());
        }

        // Register raw edge spec as a probe need (grizzly amendment 2, shape A).
        if let Some(conda_target) =
            pick_conda_target(&conda_dep_name, params.name_map, params.pypi_to_conda)
        {
            let raw_spec = {
                let s = specifiers.to_string().replace(", ", ",");
                if s.trim().is_empty() {
                    "*".to_string()
                } else {
                    s
                }
            };
            probe_needs
                .entry((conda_target.clone(), raw_spec.clone()))
                .or_insert(ProbeNeed {
                    conda_name: conda_dep_name,
                    conda_target,
                    raw_spec,
                });
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use std::sync::Arc;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Spawn a minimal PEP 503 simple-index + wheel file server.
    ///
    /// `entries`: `(filename, body_bytes)` pairs.
    ///
    /// Serves:
    ///   `GET /simple/<pep503-name>/`  → HTML listing all matching filenames,
    ///     each with a fake `#sha256=aaa…aaa` fragment.
    ///   `GET /<filename>`             → raw bytes.
    ///
    /// `max_requests`: connection limit before the server stops.
    async fn spawn_fixture_server(entries: Vec<(String, Vec<u8>)>, max_requests: usize) -> u16 {
        let by_filename: Arc<HashMap<String, Vec<u8>>> = Arc::new(
            entries
                .iter()
                .map(|(n, b)| (n.clone(), b.clone()))
                .collect(),
        );
        let all_filenames: Arc<Vec<String>> =
            Arc::new(entries.into_iter().map(|(n, _)| n).collect());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            for _ in 0..max_requests {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let by_filename = by_filename.clone();
                let all_filenames = all_filenames.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    let req_str = String::from_utf8_lossy(&buf[..n]);
                    let path = req_str
                        .lines()
                        .next()
                        .and_then(|l| l.split_whitespace().nth(1))
                        .unwrap_or("/");

                    let (status, ct, body): (&str, &str, Vec<u8>) = if let Some(rest) =
                        path.strip_prefix("/simple/")
                    {
                        let pkg_name = rest.trim_end_matches('/');
                        // PEP 503 normalize: lowercase, runs of -/_.  → -
                        let pkg_norm: String = {
                            let mut out = String::new();
                            let mut prev = false;
                            for c in pkg_name.chars().flat_map(|c| c.to_lowercase()) {
                                if matches!(c, '-' | '_' | '.') {
                                    if !prev {
                                        out.push('-');
                                        prev = true;
                                    }
                                } else {
                                    out.push(c);
                                    prev = false;
                                }
                            }
                            out
                        };
                        let prefix = format!("{pkg_norm}-");
                        let links: String = all_filenames
                            .iter()
                            .filter(|fname| {
                                fname
                                    .to_ascii_lowercase()
                                    .replace('_', "-")
                                    .starts_with(&prefix)
                            })
                            .map(|fname| {
                                // Omit the sha256 fragment — the download
                                // path accepts None sha256 and skips
                                // verification, which is fine for tests.
                                format!("<a href=\"/{fname}\">{fname}</a>\n")
                            })
                            .collect();
                        let html = format!("<!DOCTYPE html><html><body>\n{links}</body></html>\n");
                        ("200 OK", "text/html", html.into_bytes())
                    } else {
                        let fname = path.trim_start_matches('/');
                        let fname = fname.split('#').next().unwrap_or(fname);
                        match by_filename.get(fname) {
                            Some(b) => ("200 OK", "application/octet-stream", b.clone()),
                            None => ("404 Not Found", "text/plain", b"not found".to_vec()),
                        }
                    };

                    let resp = format!(
                        "HTTP/1.0 {status}\r\nContent-Length: {}\r\nContent-Type: \
                         {ct}\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(resp.as_bytes()).await;
                    let _ = stream.write_all(&body).await;
                });
            }
        });

        port
    }

    /// Build a minimal valid `.whl` zip in memory.
    fn make_wheel_bytes(dist: &str, version: &str, requires: &[&str]) -> Vec<u8> {
        let normalized = dist.replace('-', "_");
        let di = format!("{normalized}-{version}.dist-info");
        let mut metadata = format!("Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n");
        for req in requires {
            metadata.push_str(&format!("Requires-Dist: {req}\n"));
        }
        let metadata_bytes = metadata.into_bytes();
        let wheel_file = b"Wheel-Version: 1.0\nTag: py3-none-any\n".to_vec();
        let record = format!("{di}/METADATA,,\n{di}/WHEEL,,\n{di}/RECORD,,\n").into_bytes();

        let mut buf = Vec::new();
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, body) in [
            (format!("{di}/METADATA"), metadata_bytes.as_slice()),
            (format!("{di}/WHEEL"), wheel_file.as_slice()),
            (format!("{di}/RECORD"), record.as_slice()),
        ] {
            zip.start_file(&name, opts).unwrap();
            zip.write_all(body).unwrap();
        }
        zip.finish().unwrap();
        buf
    }

    fn linux64_target() -> WheelTarget {
        WheelTarget {
            python_version: "3.11".to_string(),
            conda_subdir: "linux-64".to_string(),
        }
    }

    fn unique_tmp_dir() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "retread-disc-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        );
        let dir = base.join(unique);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    // ── Test A: diamond graph ─────────────────────────────────────────────────
    //
    // primary → A, B
    // A        → C >=1
    // B        → C <2
    // C served at 1.0 and 2.0
    //
    // Discovery must:
    //  (a) collect all 4 names (A, B, C at both versions) in the pool
    //  (b) record edge specs C>=1 and C<2 in the probe-needs (as conda
    //      route memo entries when a name_map entry exists for C)
    //  (c) not recurse infinitely

    #[tokio::test]
    async fn discovery_collects_diamond_graph_all_versions() {
        let dir = unique_tmp_dir();

        // primary requires A and B (both share the "disc-" prefix)
        let primary_rd = vec!["disc-a>=1.0".to_string(), "disc-b>=1.0".to_string()];
        // A requires C >=1
        let a_bytes = make_wheel_bytes("disc-a", "1.0", &["disc-c>=1"]);
        // B requires C <2
        let b_bytes = make_wheel_bytes("disc-b", "1.0", &["disc-c<2"]);
        // C at two versions; no further deps
        let c10_bytes = make_wheel_bytes("disc-c", "1.0", &[]);
        let c20_bytes = make_wheel_bytes("disc-c", "2.0", &[]);

        let port = spawn_fixture_server(
            vec![
                ("disc_a-1.0-py3-none-any.whl".to_string(), a_bytes),
                ("disc_b-1.0-py3-none-any.whl".to_string(), b_bytes),
                ("disc_c-1.0-py3-none-any.whl".to_string(), c10_bytes),
                ("disc_c-2.0-py3-none-any.whl".to_string(), c20_bytes),
            ],
            128,
        )
        .await;

        let index = format!("http://127.0.0.1:{port}/simple/");
        let target = linux64_target();
        let name_map: BTreeMap<String, String> = BTreeMap::new();
        let pypi_to_conda: PypiToCondaMap = HashMap::new();

        let params = DiscoveryParams {
            index: &index,
            target: &target,
            download_dir: &dir,
            relax: RelaxPolicy::default(),
            conda_channels: &[],
            name_map: &name_map,
            pypi_to_conda: &pypi_to_conda,
            max_iterations: DiscoveryParams::DEFAULT_MAX_ITERATIONS,
            conda_deps: &[],
        };

        let pool = run_discovery(&primary_rd, &params)
            .await
            .expect("diamond discovery must succeed");

        std::fs::remove_dir_all(&dir).ok();

        // All three transitive names must be in the pool.
        assert!(
            pool.candidates.contains_key("disc-a"),
            "disc-a must be in pool; got {:?}",
            pool.candidates.keys().collect::<Vec<_>>()
        );
        assert!(
            pool.candidates.contains_key("disc-b"),
            "disc-b must be in pool"
        );
        assert!(
            pool.candidates.contains_key("disc-c"),
            "disc-c must be in pool"
        );

        // C must have BOTH versions (1.0 and 2.0).
        let c_versions: Vec<String> = pool.candidates["disc-c"]
            .iter()
            .map(|r| r.version.to_string())
            .collect();
        assert!(
            c_versions.contains(&"2.0".to_string()),
            "disc-c 2.0 must be in pool; got {c_versions:?}"
        );
        assert!(
            c_versions.contains(&"1.0".to_string()),
            "disc-c 1.0 must be in pool; got {c_versions:?}"
        );

        // Primary (the entry point) should NOT be in the pool — it is handled
        // upstream as Bundle.primary.
        assert!(
            !pool.candidates.contains_key("primary"),
            "primary itself must not be in pool"
        );
    }

    // ── Test B: lower version has an edge a higher version lacks ─────────────
    //
    // dep-x served at 1.0 (requires dep-y) and 2.0 (no requires).
    // Discovery from dep-x must collect dep-y (from version 1.0's metadata)
    // even though 2.0 lacks it.  This proves the all-versions recursion.

    #[tokio::test]
    async fn discovery_follows_edges_from_lower_versions() {
        let dir = unique_tmp_dir();

        let primary_rd = vec!["dep-x".to_string()];

        // dep-x 1.0 requires dep-y; dep-x 2.0 does not.
        let x10 = make_wheel_bytes("dep-x", "1.0", &["dep-y>=0.1"]);
        let x20 = make_wheel_bytes("dep-x", "2.0", &[]);
        let y_bytes = make_wheel_bytes("dep-y", "0.1", &[]);

        let port = spawn_fixture_server(
            vec![
                ("dep_x-1.0-py3-none-any.whl".to_string(), x10),
                ("dep_x-2.0-py3-none-any.whl".to_string(), x20),
                ("dep_y-0.1-py3-none-any.whl".to_string(), y_bytes),
            ],
            64,
        )
        .await;

        let index = format!("http://127.0.0.1:{port}/simple/");
        let target = linux64_target();
        let name_map: BTreeMap<String, String> = BTreeMap::new();
        let pypi_to_conda: PypiToCondaMap = HashMap::new();

        let params = DiscoveryParams {
            index: &index,
            target: &target,
            download_dir: &dir,
            relax: RelaxPolicy::default(),
            conda_channels: &[],
            name_map: &name_map,
            pypi_to_conda: &pypi_to_conda,
            max_iterations: DiscoveryParams::DEFAULT_MAX_ITERATIONS,
            conda_deps: &[],
        };

        let pool = run_discovery(&primary_rd, &params)
            .await
            .expect("lower-version-edge discovery must succeed");

        std::fs::remove_dir_all(&dir).ok();

        // dep-x must have both versions.
        let x_versions: Vec<String> = pool
            .candidates
            .get("dep-x")
            .map(|v| v.iter().map(|r| r.version.to_string()).collect())
            .unwrap_or_default();
        assert_eq!(
            {
                let mut sv = x_versions.clone();
                sv.sort();
                sv
            },
            vec!["1.0", "2.0"],
            "dep-x must have both versions; got {x_versions:?}"
        );

        // dep-y must be discovered (came from dep-x 1.0's requires_dist).
        assert!(
            pool.candidates.contains_key("dep-y"),
            "dep-y must be in pool (found via dep-x 1.0); pool keys: {:?}",
            pool.candidates.keys().collect::<Vec<_>>()
        );
    }

    // ── Test C: conda-route memo ──────────────────────────────────────────────
    //
    // conda-dep is in name_map → conda_target = "conda-conda-dep".
    // With empty conda_channels the probe is indecisive → KeepConda.
    // The dep must NOT be recursed into (its wheel is not in the pool),
    // and the route memo must record KeepConda for at least one (target, spec).

    #[tokio::test]
    async fn discovery_conda_route_memo_keeps_conda_dep() {
        let dir = unique_tmp_dir();

        // primary requires conda-dep AND pypi-only-dep.
        let primary_rd = vec![
            "conda-dep>=1.0".to_string(),
            "pypi-only-dep>=1.0".to_string(),
        ];

        // Only pypi-only-dep is served on the index.  conda-dep is NOT
        // served (no wheel exists); if discovery tried to recurse into it,
        // it would get 0 versions and produce an empty slot.
        let pypi_dep = make_wheel_bytes("pypi-only-dep", "1.0", &[]);

        let port = spawn_fixture_server(
            vec![("pypi_only_dep-1.0-py3-none-any.whl".to_string(), pypi_dep)],
            32,
        )
        .await;

        let index = format!("http://127.0.0.1:{port}/simple/");
        let target = linux64_target();

        // conda-dep is in the name_map → always has a conda target.
        let mut name_map: BTreeMap<String, String> = BTreeMap::new();
        name_map.insert("conda-dep".to_string(), "conda-conda-dep".to_string());
        let pypi_to_conda: PypiToCondaMap = HashMap::new();

        let params = DiscoveryParams {
            index: &index,
            target: &target,
            download_dir: &dir,
            relax: RelaxPolicy::default(),
            conda_channels: &[], // empty → probe is indecisive → KeepConda
            name_map: &name_map,
            pypi_to_conda: &pypi_to_conda,
            max_iterations: DiscoveryParams::DEFAULT_MAX_ITERATIONS,
            conda_deps: &[],
        };

        let pool = run_discovery(&primary_rd, &params)
            .await
            .expect("conda-route discovery must succeed");

        std::fs::remove_dir_all(&dir).ok();

        // pypi-only-dep must be in the pool.
        assert!(
            pool.candidates.contains_key("pypi-only-dep"),
            "pypi-only-dep must be in pool; got {:?}",
            pool.candidates.keys().collect::<Vec<_>>()
        );

        // The conda-route memo must have at least one KeepConda entry for
        // ("conda-conda-dep", some-spec).
        let has_keep_conda = pool.conda_route_memo.iter().any(|((target, _), decision)| {
            target == "conda-conda-dep" && *decision == RouteDecision::KeepConda
        });
        assert!(
            has_keep_conda,
            "conda_route_memo must have a KeepConda entry for conda-conda-dep; memo: {:?}",
            pool.conda_route_memo
                .iter()
                .filter(|((t, _), _)| t == "conda-conda-dep")
                .collect::<Vec<_>>()
        );
    }

    // ── Test D: both spec shapes in probe-needs ───────────────────────────────
    //
    // Verify that both the raw edge spec (shape A: `>=1.0`) AND the version-
    // widened spec (shape B: e.g. `>=1,<2` under the default relax policy)
    // are recorded in the conda-route memo for a conda-routed dep.

    #[tokio::test]
    async fn discovery_records_both_probe_spec_shapes() {
        let dir = unique_tmp_dir();

        // primary requires spec-dep >=1.0
        let primary_rd = vec!["spec-dep>=1.0".to_string()];

        // spec-dep served at 1.5 with no further deps.
        let spec_dep = make_wheel_bytes("spec-dep", "1.5", &[]);

        let port = spawn_fixture_server(
            vec![("spec_dep-1.5-py3-none-any.whl".to_string(), spec_dep)],
            16,
        )
        .await;

        let index = format!("http://127.0.0.1:{port}/simple/");
        let target = linux64_target();

        // spec-dep routes to conda.
        let mut name_map: BTreeMap<String, String> = BTreeMap::new();
        name_map.insert("spec-dep".to_string(), "conda-spec-dep".to_string());
        let pypi_to_conda: PypiToCondaMap = HashMap::new();

        let params = DiscoveryParams {
            index: &index,
            target: &target,
            download_dir: &dir,
            relax: RelaxPolicy::default(),
            conda_channels: &[],
            name_map: &name_map,
            pypi_to_conda: &pypi_to_conda,
            max_iterations: DiscoveryParams::DEFAULT_MAX_ITERATIONS,
            conda_deps: &[],
        };

        let pool = run_discovery(&primary_rd, &params)
            .await
            .expect("both-spec-shapes discovery must succeed");

        std::fs::remove_dir_all(&dir).ok();

        // Raw edge spec `>=1.0` → normalized to `>=1.0` (no space, non-empty).
        let has_raw_spec = pool
            .conda_route_memo
            .keys()
            .any(|(target, spec)| target == "conda-spec-dep" && spec == ">=1.0");
        assert!(
            has_raw_spec,
            "memo must contain raw edge spec >=1.0 for conda-spec-dep; memo keys: {:?}",
            pool.conda_route_memo.keys().collect::<Vec<_>>()
        );

        // Version-widened spec for 1.5: widen_exact("1.5", default) should
        // produce a range like `>=1.5,<2` or similar.  We just assert that
        // at least one additional (conda-spec-dep, *) key exists beyond the
        // raw spec — confirming shape-B was also registered.
        let all_specs_for_target: Vec<&str> = pool
            .conda_route_memo
            .keys()
            .filter(|(t, _)| t == "conda-spec-dep")
            .map(|(_, s)| s.as_str())
            .collect();
        assert!(
            all_specs_for_target.len() >= 2,
            "must have at least 2 spec shapes for conda-spec-dep (raw edge + widened version); \
             got: {all_specs_for_target:?}"
        );
    }
}
