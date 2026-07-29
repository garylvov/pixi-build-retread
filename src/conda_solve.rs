//! Conda solving helpers that survived the v4.2.0 removal of the
//! legacy mirror-solver's pre-emission solve check.
//!
//! `solve_selected_records` solves a spec set coherently against the
//! workspace's channels (sparse, reachable-subset repodata) and returns
//! the concrete records the solver selected. Its one production consumer
//! is `workspace::extract_transitive_constraints`, which derives the
//! constraints conda's actual picks impose on other packages -- part of
//! the conda-first preference plumbing (conda pins flow into emissions
//! and into the uv resolver's constraint set).

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::str::FromStr;
use std::sync::Arc;

use rattler_conda_types::{
    ChannelUrl, GenericVirtualPackage, MatchSpec, PackageName, ParseStrictness, RepoDataRecord,
    Version,
};
use rattler_solve::{ChannelPriority, SolveStrategy, SolverImpl, SolverTask, resolvo};

use crate::relax::CondaMatchSpec;

/// Failure classification retained at the bundle-scoped probe boundary.
///
/// Only `Unsolvable` is a monotone proof suitable for block deletion.
/// Operational/configuration failures and unavailable repodata must never be
/// promoted into such a proof by the conflict-localization reducer.
#[derive(Debug)]
pub(crate) enum SharedSolveFailure {
    Unsolvable(Vec<String>),
    Unproven(Vec<String>),
    Unavailable(Vec<String>),
}

impl SharedSolveFailure {
    fn into_reasons(self) -> Vec<String> {
        match self {
            Self::Unsolvable(reasons) | Self::Unproven(reasons) | Self::Unavailable(reasons) => {
                reasons
            }
        }
    }
}

/// Solve a spec set against already-loaded records and return the
/// concrete records the solver selected. Shared by the pre-emission
/// solve check and workspace transitive extraction so both reason
/// about the SAME coherent solution instead of each inventing its own
/// approximation.
///
/// `preferred` seeds rattler's `locked_packages` field (soft preference
/// — the solver will pick the preferred variant when available but WILL
/// deviate if constraints require it). Pass `&[]` for a cold solve.
/// This is a pure speed/stability optimisation: the solver's SAT/UNSAT
/// verdict and the diagnostic strings it emits are identical regardless
/// of what is in `preferred` because `locked_packages` never hard-
/// constrains a version.
#[cfg(test)]
fn solve_selected_records_from_records(
    parsed_specs: Vec<MatchSpec>,
    all_records: &[RepoDataRecord],
    target_python: &str,
    channel_priority: ChannelPriority,
    system_requirements: &BTreeMap<String, String>,
    strategy: SolveStrategy,
    preferred: Vec<RepoDataRecord>,
) -> std::result::Result<Vec<RepoDataRecord>, Vec<String>> {
    solve_selected_records_from_records_for_target(
        parsed_specs,
        all_records,
        target_python,
        channel_priority,
        system_requirements,
        None,
        strategy,
        preferred,
    )
    .map_err(SharedSolveFailure::into_reasons)
}

#[allow(clippy::too_many_arguments)]
fn solve_selected_records_from_records_for_target(
    parsed_specs: Vec<MatchSpec>,
    all_records: &[RepoDataRecord],
    target_python: &str,
    channel_priority: ChannelPriority,
    system_requirements: &BTreeMap<String, String>,
    detected_virtual_packages: Option<&BTreeMap<String, String>>,
    strategy: SolveStrategy,
    preferred: Vec<RepoDataRecord>,
) -> std::result::Result<Vec<RepoDataRecord>, SharedSolveFailure> {
    let virtual_packages = build_virtual_packages_for_target(
        target_python,
        system_requirements,
        detected_virtual_packages,
    )
    .map_err(|error| SharedSolveFailure::Unproven(vec![error]))?;
    let task = SolverTask {
        available_packages: vec![all_records],
        // `locked_packages` = soft preference in rattler_solve: the solver
        // prefers these versions but will deviate when forced by other
        // constraints.  `pinned_packages` (NOT used here) is the hard field
        // that forces a specific version even at the cost of downgrading
        // other packages — too strong for a co-installability check that
        // must never veto a build spuriously.
        locked_packages: preferred,
        pinned_packages: Vec::new(),
        virtual_packages,
        specs: parsed_specs,
        constraints: Vec::new(),
        timeout: Some(std::time::Duration::from_secs(60)),
        channel_priority,
        exclude_newer: None,
        min_age: None,
        strategy,
    };
    let mut solver = resolvo::Solver;
    match solver.solve(task) {
        Ok(solution) => Ok(solution.records),
        Err(rattler_solve::SolveError::Unsolvable(reasons)) => {
            Err(SharedSolveFailure::Unsolvable(reasons))
        }
        Err(other) => Err(SharedSolveFailure::Unproven(vec![format!(
            "solver error: {other}"
        )])),
    }
}

/// v1.4.0: run the resolvo solve on the blocking thread pool. The
/// solve is pure CPU (seconds on ~1M records); calling it directly on
/// the async executor pinned a runtime worker for the duration of
/// every solve (up to 40 per bundle across the refinement loop),
/// starving concurrent probes and downloads.
///
/// `preferred` is forwarded directly to
/// `solve_selected_records_from_records` as the warm-start seed (soft
/// preference via `locked_packages`). Pass `Vec::new()` for a cold solve.
#[cfg(test)]
async fn solve_on_blocking_pool(
    parsed_specs: Vec<MatchSpec>,
    records: Vec<RepoDataRecord>,
    target_python: String,
    channel_priority: ChannelPriority,
    system_requirements: BTreeMap<String, String>,
    strategy: SolveStrategy,
    preferred: Vec<RepoDataRecord>,
) -> std::result::Result<Vec<RepoDataRecord>, Vec<String>> {
    solve_on_blocking_pool_for_target(
        parsed_specs,
        records.into(),
        None,
        target_python,
        channel_priority,
        system_requirements,
        None,
        strategy,
        preferred,
    )
    .await
    .map_err(SharedSolveFailure::into_reasons)
}

#[allow(clippy::too_many_arguments)]
async fn solve_on_blocking_pool_for_target(
    parsed_specs: Vec<MatchSpec>,
    records: Arc<[RepoDataRecord]>,
    probe_grant: Option<crate::thread_budget::ProbePoolGrant>,
    target_python: String,
    channel_priority: ChannelPriority,
    system_requirements: BTreeMap<String, String>,
    detected_virtual_packages: Option<BTreeMap<String, String>>,
    strategy: SolveStrategy,
    preferred: Vec<RepoDataRecord>,
) -> std::result::Result<Vec<RepoDataRecord>, SharedSolveFailure> {
    let probe_task = match &probe_grant {
        Some(grant) => Some(grant.acquire_task().await),
        None => None,
    };
    let t_solve = std::time::Instant::now();
    let specs_count = parsed_specs.len();
    let records_count = records.len();
    let result = tokio::task::spawn_blocking(move || {
        // Keep the bundle's one coordinated grant alive until this CPU task
        // actually exits, even if its async waiter is cancelled.
        let _probe_grant = probe_grant;
        let _probe_task = probe_task;
        solve_selected_records_from_records_for_target(
            parsed_specs,
            &records,
            &target_python,
            channel_priority,
            &system_requirements,
            detected_virtual_packages.as_ref(),
            strategy,
            preferred,
        )
    })
    .await
    .unwrap_or_else(|e| {
        Err(SharedSolveFailure::Unproven(vec![format!(
            "solver task panicked: {e}"
        )]))
    });
    tracing::info!(
        elapsed_ms = t_solve.elapsed().as_millis() as u64,
        satisfiable = result.is_ok(),
        specs = specs_count,
        records = records_count,
        "bench: resolvo solve finished",
    );
    result
}

fn parse_match_specs(specs: &[String]) -> Vec<MatchSpec> {
    let mut parsed_specs: Vec<MatchSpec> = Vec::with_capacity(specs.len());
    for raw in specs {
        match MatchSpec::from_str(raw, ParseStrictness::Lenient) {
            Ok(s) => parsed_specs.push(s),
            Err(e) => {
                tracing::debug!(spec = %raw, error = %e, "solve-check: skipping unparseable spec");
            }
        }
    }
    parsed_specs
}

type SparsePair = (
    String,
    Arc<rattler_repodata_gateway::sparse::SparseRepoData>,
);
type SparseLoadResult = std::result::Result<(Vec<RepoDataRecord>, Vec<String>), Vec<String>>;

/// Bundle-scoped sparse-repodata and reachable-record cache.
///
/// The same [`CondaCoSolveContext`](crate::handler::CondaCoSolveContext) runs
/// many counterfactual solves whose root sets are subsets of the first full
/// route set. Sparse handles are shared once, and reachable records grow only
/// when a later phase introduces a root that was not covered previously.
/// Returning `Arc` snapshots lets concurrent solves share the exact record
/// allocation while a later root expansion safely replaces the cached union.
#[derive(Clone)]
pub(crate) struct SharedSparseSolveData {
    channels: Arc<[ChannelUrl]>,
    target_subdir: Arc<str>,
    pairs: Arc<tokio::sync::OnceCell<Arc<[SparsePair]>>>,
    reachable: Arc<tokio::sync::Mutex<ReachableState>>,
}

impl std::fmt::Debug for SharedSparseSolveData {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SharedSparseSolveData")
            .field("channels", &self.channels)
            .field("target_subdir", &self.target_subdir)
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
struct ReachableState {
    root_order: Vec<PackageName>,
    covered_roots: BTreeSet<PackageName>,
    record_names: BTreeSet<PackageName>,
    records: Option<Arc<[RepoDataRecord]>>,
    consulted: Arc<[String]>,
}

impl SharedSparseSolveData {
    pub(crate) fn new(channels: &[ChannelUrl], target_subdir: &str) -> Self {
        Self {
            channels: channels.to_vec().into(),
            target_subdir: Arc::from(target_subdir),
            pairs: Arc::new(tokio::sync::OnceCell::new()),
            reachable: Arc::new(tokio::sync::Mutex::new(ReachableState::default())),
        }
    }

    async fn pairs(&self) -> Arc<[SparsePair]> {
        self.pairs
            .get_or_init(|| async {
                crate::repodata::sparse_pairs(&self.channels, &self.target_subdir)
                    .await
                    .into()
            })
            .await
            .clone()
    }

    async fn records_for(
        &self,
        parsed_specs: &[MatchSpec],
    ) -> (Arc<[RepoDataRecord]>, Arc<[String]>) {
        self.records_for_with_loader(parsed_specs, load_selected_records_sparse_from_pairs)
            .await
    }

    async fn records_for_with_loader<Load, LoadFuture>(
        &self,
        parsed_specs: &[MatchSpec],
        load: Load,
    ) -> (Arc<[RepoDataRecord]>, Arc<[String]>)
    where
        Load: FnOnce(Arc<[SparsePair]>, Vec<PackageName>) -> LoadFuture,
        LoadFuture: std::future::Future<Output = SparseLoadResult>,
    {
        let requested_roots = exact_root_names(parsed_specs);
        let mut state = self.reachable.lock().await;
        if requested_roots
            .iter()
            .all(|root| state.covered_roots.contains(root) || state.record_names.contains(root))
            && let Some(records) = &state.records
        {
            tracing::debug!(
                records = records.len(),
                covered_roots = state.covered_roots.len(),
                requested_roots = requested_roots.len(),
                "bench: reusing bundle-scoped sparse reachable records",
            );
            return (Arc::clone(records), Arc::clone(&state.consulted));
        }

        let mut root_order = state.root_order.clone();
        let mut covered_roots = state.covered_roots.clone();
        for root in requested_roots {
            if !covered_roots.contains(&root) && !state.record_names.contains(&root) {
                covered_roots.insert(root.clone());
                root_order.push(root);
            }
        }
        let pairs = self.pairs().await;
        let (records, consulted) = match load(pairs, root_order.clone()).await {
            Ok(loaded) => loaded,
            // Gateway errors and spawn-blocking panics both take this path.
            // Keep the last good snapshot intact so a later probe can retry.
            Err(consulted) => return (Arc::from([]), consulted.into()),
        };
        let record_names = records
            .iter()
            .map(|record| record.package_record.name.clone())
            .collect();
        let records: Arc<[RepoDataRecord]> = records.into();
        let consulted: Arc<[String]> = consulted.into();
        state.root_order = root_order;
        state.covered_roots = covered_roots;
        state.record_names = record_names;
        state.records = Some(Arc::clone(&records));
        state.consulted = Arc::clone(&consulted);
        (records, consulted)
    }
}

fn exact_root_names(parsed_specs: &[MatchSpec]) -> Vec<PackageName> {
    let mut seen = BTreeSet::new();
    parsed_specs
        .iter()
        .filter_map(|spec| match spec.name.as_ref() {
            Some(rattler_conda_types::PackageNameMatcher::Exact(name))
                if seen.insert(name.clone()) =>
            {
                Some(name.clone())
            }
            _ => None,
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn solve_records_for_test(
    records: &[RepoDataRecord],
    specs: &[String],
    target_python: &str,
) -> std::result::Result<Vec<RepoDataRecord>, Vec<String>> {
    let parsed_specs = specs
        .iter()
        .map(|raw| {
            MatchSpec::from_str(raw, ParseStrictness::Lenient)
                .map_err(|error| format!("invalid test match spec `{raw}`: {error}"))
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| vec![error])?;
    solve_selected_records_from_records(
        parsed_specs,
        records,
        target_python,
        ChannelPriority::Strict,
        &BTreeMap::new(),
        SolveStrategy::Highest,
        Vec::new(),
    )
}

/// Run a conda solve over the combined spec set. `specs` must include
/// both retread's emitted run-deps for this output AND the workspace's
/// effective conda deps for the env this output is built for. Target
/// python is encoded as a virtual package (`__cuda` not included; if
/// you need to model GPU/CUDA-only constraints, add them via the
/// caller).
///
/// `target_subdir` is the linux-64/osx-64/etc. selector. retread today
/// targets linux-64 only.
/// v0.37.0+: build the legacy rattler virtual-package set the solver should
/// see. Host detection supplies defaults and workspace system requirements
/// override them. Exact rich targets use [`build_virtual_packages_for_target`]
/// instead: a nonempty Pixi-detected map is a complete set, so absence (most
/// importantly the absence of `__cuda`) must be preserved rather than filled
/// from the build host.
///
/// Without the workspace override on the legacy path, retread's solve check
/// sees the build host's virtual packages while Pixi's actual solve sees the
/// workspace declarations. That asymmetry produced "retread sat, pixi unsat"
/// for the gsn gymnasium failure that motivated v0.37.0.
pub fn build_virtual_packages(
    target_python: &str,
    system_requirements: &std::collections::BTreeMap<String, String>,
) -> Vec<GenericVirtualPackage> {
    build_virtual_packages_for_target(target_python, system_requirements, None)
        .expect("legacy virtual-package mapping uses valid built-in package names")
}

/// Build virtual packages for either a legacy inferred target or an exact
/// Pixi target contract. `Some` means the request is contract-qualified, even
/// when its detected map is empty. A nonempty map is the authoritative
/// complete set reported by Pixi. We deliberately do not run host detection
/// on either contract-qualified path: merging a host baseline would turn an
/// exact CPU contract into a CUDA contract, or retain host OS markers that do
/// not belong to the requested target.
pub(crate) fn build_virtual_packages_for_target(
    target_python: &str,
    system_requirements: &BTreeMap<String, String>,
    detected_virtual_packages: Option<&BTreeMap<String, String>>,
) -> Result<Vec<GenericVirtualPackage>, String> {
    let host_baseline = if detected_virtual_packages.is_some() {
        Vec::new()
    } else {
        match rattler_virtual_packages::VirtualPackage::detect(
            &rattler_virtual_packages::VirtualPackageOverrides::default(),
        ) {
            Ok(vps) => vps.into_iter().map(GenericVirtualPackage::from).collect(),
            Err(e) => {
                tracing::debug!(
                    error = %e,
                    "solve-check: host virtual package detection failed; falling back to minimal set",
                );
                Vec::new()
            }
        }
    };
    build_virtual_packages_from_baseline(
        target_python,
        system_requirements,
        detected_virtual_packages,
        host_baseline,
    )
}

/// Pure assembly boundary used by the exact-target regression. The injected
/// baseline is intentionally discarded when `detected_virtual_packages` is
/// `Some`, making both exact and contract-qualified-empty absence semantics
/// directly testable on any host.
fn build_virtual_packages_from_baseline(
    target_python: &str,
    system_requirements: &BTreeMap<String, String>,
    detected_virtual_packages: Option<&BTreeMap<String, String>>,
    host_baseline: Vec<GenericVirtualPackage>,
) -> Result<Vec<GenericVirtualPackage>, String> {
    let contract_qualified = detected_virtual_packages.is_some();
    let exact_detected = detected_virtual_packages.filter(|packages| !packages.is_empty());
    let mut virtual_packages = if contract_qualified {
        Vec::new()
    } else {
        host_baseline
    };

    if let Some(packages) = exact_detected {
        for (req_key, req_value) in packages {
            insert_virtual_package(&mut virtual_packages, req_key, req_value, true)?;
        }
        // Pixi 0.73 does not include a declared CUDA compute capability in
        // its detected VP list. Supply only that known declaration gap; all
        // other absences in the exact detected set remain authoritative.
        if !packages.contains_key("cuda_arch")
            && let Some(cuda_arch) = system_requirements.get("cuda_arch")
        {
            insert_virtual_package(&mut virtual_packages, "cuda_arch", cuda_arch, false)?;
        }
    } else {
        for (req_key, req_value) in system_requirements {
            insert_virtual_package(&mut virtual_packages, req_key, req_value, false)?;
        }
    }

    if let Ok(v) = Version::from_str(target_python)
        && let Ok(name) = PackageName::from_str("__cpython")
    {
        virtual_packages.retain(|vp| vp.name.as_normalized() != "__cpython");
        virtual_packages.push(GenericVirtualPackage {
            name,
            version: v,
            build_string: String::new(),
        });
    }
    Ok(virtual_packages)
}

fn insert_virtual_package(
    virtual_packages: &mut Vec<GenericVirtualPackage>,
    req_key: &str,
    req_value: &str,
    allow_arbitrary_detected_package: bool,
) -> Result<(), String> {
    let normalized_key = req_key.trim().trim_start_matches("__");
    let (vp_name, is_archspec) = match normalized_key {
        "cuda" => ("__cuda".to_string(), false),
        "cuda_arch" => ("__cuda_arch".to_string(), false),
        "libc" | "glibc" => ("__glibc".to_string(), false),
        "macos" | "osx" => ("__osx".to_string(), false),
        "windows" | "win" => ("__win".to_string(), false),
        "archspec" => ("__archspec".to_string(), true),
        "linux" => ("__linux".to_string(), false),
        "unix" => ("__unix".to_string(), false),
        other if allow_arbitrary_detected_package && !other.is_empty() => {
            (format!("__{other}"), false)
        }
        other => {
            tracing::trace!(
                key = %other,
                "solve-check: ignoring unrecognized system-requirement key (not in pixi schema)",
            );
            return Ok(());
        }
    };
    let Ok(name) = PackageName::from_str(&vp_name) else {
        if allow_arbitrary_detected_package {
            return Err(format!(
                "invalid exact target virtual-package name `{req_key}`"
            ));
        }
        tracing::trace!(
            key = %req_key,
            "solve-check: ignoring invalid detected virtual-package name",
        );
        return Ok(());
    };

    // Pixi renders generic virtual packages as `name=version=build`. Preserve
    // that representation for exact contracts. Legacy archspec declarations
    // often contain only the architecture string and retain the historical
    // synthetic version 1.0 encoding.
    let (version, build_string) = if let Some((version, build_string)) = req_value.split_once('=')
        && let Ok(version) = Version::from_str(version)
    {
        (version, build_string.to_string())
    } else if normalized_key == "unix" && req_value.is_empty() {
        (Version::major(0), "0".to_string())
    } else if is_archspec {
        let Ok(version) = Version::from_str("1.0") else {
            return Ok(());
        };
        (version, req_value.to_string())
    } else {
        match Version::from_str(req_value) {
            Ok(version) => (version, String::new()),
            Err(_) => {
                let Ok(version) = Version::from_str("1.0") else {
                    return Ok(());
                };
                (version, req_value.to_string())
            }
        }
    };
    virtual_packages.retain(|package| package.name.as_normalized() != vp_name);
    virtual_packages.push(GenericVirtualPackage {
        name,
        version,
        build_string,
    });
    Ok(())
}

/// v1.5.0: load only the records REACHABLE from the spec set's
/// package names (transitive closure over `depends`), via the shared
/// memory-mapped sparse repodata store in `crate::repodata`. The old
/// path fully parsed and materialized every channel record (~1M
/// records, seconds of CPU, GBs of RSS, once per channel-set); the
/// reachable subset for a typical bundle is a few thousand records
/// loaded in milliseconds. Returns `(records, consulted)` -- empty
/// `consulted` means NO repodata could be obtained (callers abstain).
/// Dedup is first-seen filename in channel-priority order, because
/// rattler_solve aborts on DuplicateRecords.
async fn load_selected_records_sparse(
    channels: &[ChannelUrl],
    target_subdir: &str,
    parsed_specs: &[MatchSpec],
) -> (Vec<RepoDataRecord>, Vec<String>) {
    let pairs: Arc<[SparsePair]> = crate::repodata::sparse_pairs(channels, target_subdir)
        .await
        .into();
    if pairs.is_empty() {
        return (Vec::new(), Vec::new());
    }
    // Exact-named specs seed the reachable-set walk. Glob/regex
    // matchers (rare; retread never emits them) can't seed a name walk
    // -- they're skipped here and simply unmatchable in the subset,
    // identical to how an absent package behaved in the full set.
    let root_names = exact_root_names(parsed_specs);
    match load_selected_records_sparse_from_pairs(pairs, root_names).await {
        Ok(loaded) => loaded,
        Err(consulted) => (Vec::new(), consulted),
    }
}

async fn load_selected_records_sparse_from_pairs(
    pairs: Arc<[SparsePair]>,
    root_names: Vec<PackageName>,
) -> SparseLoadResult {
    use rattler_repodata_gateway::sparse::{PackageFormatSelection, SparseRepoData};
    let consulted: Vec<String> = pairs.iter().map(|(label, _)| label.clone()).collect();
    let roots = root_names.len();
    let t = std::time::Instant::now();
    let per_repo = match tokio::task::spawn_blocking(move || {
        SparseRepoData::load_records_recursive(
            pairs.iter().map(|(_, handle)| handle.as_ref()),
            root_names,
            None,
            PackageFormatSelection::default(),
        )
    })
    .await
    {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "solve-check: sparse record load failed");
            return Err(consulted);
        }
        Err(e) => {
            tracing::warn!(error = %e, "solve-check: sparse record load panicked");
            return Err(consulted);
        }
    };
    let mut records: Vec<RepoDataRecord> = Vec::new();
    let mut seen_filenames: HashSet<String> = HashSet::new();
    for repo_records in per_repo {
        for rec in repo_records {
            if seen_filenames.insert(rec.file_name.clone()) {
                records.push(rec);
            }
        }
    }
    tracing::info!(
        elapsed_ms = t.elapsed().as_millis() as u64,
        records = records.len(),
        roots,
        pairs = consulted.len(),
        "bench: sparse reachable records loaded",
    );
    Ok((records, consulted))
}

/// Solve a spec set against cached repodata and return the concrete
/// records the solver selected. Callers that need to reason about
/// transitives must use this instead of "latest matching build"
/// guesses, otherwise open-ended pins like `torchvision >=0.22` and
/// `pytorch-gpu >=2.7` get their transitives sourced from DIFFERENT
/// package generations and create impossible merged constraints.
pub async fn solve_selected_records(
    channels: &[ChannelUrl],
    specs: &[CondaMatchSpec],
    target_python: &str,
    target_subdir: &str,
    channel_priority: ChannelPriority,
    system_requirements: &BTreeMap<String, String>,
    strategy: SolveStrategy,
) -> std::result::Result<Vec<RepoDataRecord>, Vec<String>> {
    solve_selected_records_for_target(
        channels,
        specs,
        target_python,
        target_subdir,
        channel_priority,
        system_requirements,
        None,
        strategy,
    )
    .await
}

/// Target-contract-aware form of [`solve_selected_records`]. A nonempty
/// detected map is Pixi's complete virtual-package set for the selected rich
/// target. `Some(empty)` remains contract-qualified and suppresses host
/// detection; only `None` selects legacy host inference.
#[allow(clippy::too_many_arguments)]
pub async fn solve_selected_records_for_target(
    channels: &[ChannelUrl],
    specs: &[CondaMatchSpec],
    target_python: &str,
    target_subdir: &str,
    channel_priority: ChannelPriority,
    system_requirements: &BTreeMap<String, String>,
    detected_virtual_packages: Option<&BTreeMap<String, String>>,
    strategy: SolveStrategy,
) -> std::result::Result<Vec<RepoDataRecord>, Vec<String>> {
    // The only production entrance to conda's MatchSpec parser is the typed
    // raw-name boundary. Test helpers retain String input so solver fixtures
    // can express arbitrary malformed/diagnostic cases without weakening the
    // production API.
    let rendered_specs: Vec<String> = specs.iter().map(ToString::to_string).collect();
    let parsed_specs = parse_match_specs(&rendered_specs);
    let (records, _consulted) =
        load_selected_records_sparse(channels, target_subdir, &parsed_specs).await;
    if records.is_empty() {
        return Err(vec![
            "solve-check skipped: no repodata available from disk cache".into(),
        ]);
    }
    solve_on_blocking_pool_for_target(
        parsed_specs,
        records.into(),
        None,
        target_python.to_string(),
        channel_priority,
        system_requirements.clone(),
        detected_virtual_packages.cloned(),
        strategy,
        Vec::new(),
    )
    .await
    .map_err(SharedSolveFailure::into_reasons)
}

/// Bundle-scoped form of [`solve_selected_records_for_target`] that shares
/// sparse handles and the grow-only reachable record union across probes.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn solve_selected_records_for_target_shared(
    shared: &SharedSparseSolveData,
    specs: &[CondaMatchSpec],
    probe_grant: Option<crate::thread_budget::ProbePoolGrant>,
    target_python: &str,
    channel_priority: ChannelPriority,
    system_requirements: &BTreeMap<String, String>,
    detected_virtual_packages: Option<&BTreeMap<String, String>>,
    strategy: SolveStrategy,
) -> std::result::Result<Vec<RepoDataRecord>, SharedSolveFailure> {
    let rendered_specs: Vec<String> = specs.iter().map(ToString::to_string).collect();
    let parsed_specs = parse_match_specs(&rendered_specs);
    let (records, _consulted) = shared.records_for(&parsed_specs).await;
    if records.is_empty() {
        return Err(SharedSolveFailure::Unavailable(vec![
            "solve-check skipped: no repodata available from disk cache".into(),
        ]));
    }
    solve_on_blocking_pool_for_target(
        parsed_specs,
        records,
        probe_grant,
        target_python.to_string(),
        channel_priority,
        system_requirements.clone(),
        detected_virtual_packages.cloned(),
        strategy,
        Vec::new(),
    )
    .await
}

/// Populate the bundle-scoped reachable-record union without running resolvo.
///
/// Standalone provider probes can then share one complete immutable snapshot
/// instead of introducing roots one at a time and rebuilding a growing union.
pub(crate) async fn prewarm_selected_records_for_target_shared(
    shared: &SharedSparseSolveData,
    specs: &[CondaMatchSpec],
) {
    let rendered_specs: Vec<String> = specs.iter().map(ToString::to_string).collect();
    let parsed_specs = parse_match_specs(&rendered_specs);
    let _ = shared.records_for(&parsed_specs).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rattler_conda_types::{PackageRecord, RepoDataRecord, VersionWithSource};
    use std::str::FromStr;
    use url::Url;

    fn repo_record(name: &str, version: &str, depends: &[&str]) -> RepoDataRecord {
        let mut package_record = PackageRecord::new(
            name.parse().unwrap(),
            VersionWithSource::from_str(version).unwrap(),
            "h123456_0".to_string(),
        );
        package_record.subdir = "linux-64".to_string();
        package_record.depends = depends.iter().map(|s| (*s).to_string()).collect();
        RepoDataRecord {
            package_record,
            file_name: format!("{name}-{version}-h123456_0.conda"),
            url: Url::parse(&format!(
                "https://example.invalid/linux-64/{name}-{version}-h123456_0.conda"
            ))
            .unwrap(),
            channel: Some("https://example.invalid".into()),
        }
    }

    fn vp_lookup<'a>(
        vps: &'a [GenericVirtualPackage],
        name: &str,
    ) -> Option<&'a GenericVirtualPackage> {
        vps.iter().find(|vp| vp.name.as_normalized() == name)
    }

    fn virtual_package(name: &str, version: &str, build_string: &str) -> GenericVirtualPackage {
        GenericVirtualPackage {
            name: PackageName::from_str(name).unwrap(),
            version: Version::from_str(version).unwrap(),
            build_string: build_string.to_string(),
        }
    }

    fn shared_sparse_fixture(state: ReachableState) -> SharedSparseSolveData {
        let pairs = tokio::sync::OnceCell::new();
        assert!(pairs.set(Vec::<SparsePair>::new().into()).is_ok());
        SharedSparseSolveData {
            channels: Arc::from([]),
            target_subdir: Arc::from("linux-64"),
            pairs: Arc::new(pairs),
            reachable: Arc::new(tokio::sync::Mutex::new(state)),
        }
    }

    #[test]
    fn exact_root_names_preserve_first_seen_order() {
        let specs = parse_match_specs(&[
            "zlib >=1.2".to_string(),
            "numpy >=2".to_string(),
            "zlib <2".to_string(),
            "python 3.11.*".to_string(),
        ]);

        let roots = exact_root_names(&specs)
            .into_iter()
            .map(|name| name.as_normalized().to_string())
            .collect::<Vec<_>>();

        assert_eq!(roots, ["zlib", "numpy", "python"]);
    }

    #[tokio::test]
    async fn shared_sparse_solve_data_reuses_covered_record_arc() {
        let records: Arc<[RepoDataRecord]> = vec![
            repo_record("python", "3.11.5", &[]),
            repo_record("numpy", "2.1.0", &["python >=3.11,<3.12"]),
        ]
        .into();
        let shared = SharedSparseSolveData {
            channels: Arc::from([]),
            target_subdir: Arc::from("linux-64"),
            pairs: Arc::new(tokio::sync::OnceCell::new()),
            reachable: Arc::new(tokio::sync::Mutex::new(ReachableState {
                root_order: vec![
                    PackageName::from_str("python").unwrap(),
                    PackageName::from_str("numpy").unwrap(),
                ],
                covered_roots: BTreeSet::from([
                    PackageName::from_str("numpy").unwrap(),
                    PackageName::from_str("python").unwrap(),
                ]),
                record_names: BTreeSet::from([
                    PackageName::from_str("numpy").unwrap(),
                    PackageName::from_str("python").unwrap(),
                ]),
                records: Some(Arc::clone(&records)),
                consulted: Arc::from(["fixture/linux-64".to_string()]),
            })),
        };

        let full = parse_match_specs(&["python 3.11.*".into(), "numpy >=2".into()]);
        let subset = parse_match_specs(&["python 3.11.*".into()]);
        let (full_records, full_consulted) = shared.records_for(&full).await;
        let (subset_records, subset_consulted) = shared.records_for(&subset).await;

        assert!(Arc::ptr_eq(&records, &full_records));
        assert!(Arc::ptr_eq(&full_records, &subset_records));
        assert_eq!(&*full_consulted, &*subset_consulted);
    }

    #[tokio::test]
    async fn shared_sparse_solve_data_reuses_transitively_loaded_root() {
        let parent = PackageName::from_str("parent").unwrap();
        let child = PackageName::from_str("child").unwrap();
        let records: Arc<[RepoDataRecord]> = vec![
            repo_record("parent", "1.0", &["child >=1"]),
            repo_record("child", "1.0", &[]),
        ]
        .into();
        let shared = shared_sparse_fixture(ReachableState {
            root_order: vec![parent.clone()],
            covered_roots: BTreeSet::from([parent.clone()]),
            record_names: BTreeSet::from([parent, child]),
            records: Some(Arc::clone(&records)),
            consulted: Arc::from(["fixture/linux-64".to_string()]),
        });
        let loader_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let loader_calls_for_probe = Arc::clone(&loader_calls);
        let specs = parse_match_specs(&["child >=1".to_string()]);

        let (reused, consulted) = shared
            .records_for_with_loader(&specs, move |_pairs, _roots| async move {
                loader_calls_for_probe.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok((Vec::new(), Vec::new()))
            })
            .await;

        assert_eq!(loader_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(Arc::ptr_eq(&records, &reused));
        assert_eq!(&*consulted, ["fixture/linux-64"]);
    }

    #[tokio::test]
    async fn failed_sparse_load_does_not_poison_reachable_cache() {
        let parent = PackageName::from_str("parent").unwrap();
        let child = PackageName::from_str("child").unwrap();
        let missing = PackageName::from_str("missing").unwrap();
        let original_records: Arc<[RepoDataRecord]> = vec![
            repo_record("parent", "1.0", &["child >=1"]),
            repo_record("child", "1.0", &[]),
        ]
        .into();
        let shared = shared_sparse_fixture(ReachableState {
            root_order: vec![parent.clone()],
            covered_roots: BTreeSet::from([parent.clone()]),
            record_names: BTreeSet::from([parent.clone(), child.clone()]),
            records: Some(Arc::clone(&original_records)),
            consulted: Arc::from(["fixture/linux-64".to_string()]),
        });
        let specs = parse_match_specs(&["missing >=1".to_string()]);
        let failed_roots = Arc::new(std::sync::Mutex::new(Vec::new()));
        let failed_roots_for_probe = Arc::clone(&failed_roots);

        let (failed_records, failed_consulted) = shared
            .records_for_with_loader(&specs, move |_pairs, roots| async move {
                failed_roots_for_probe.lock().unwrap().push(roots);
                Err(vec!["fixture/linux-64".to_string()])
            })
            .await;

        assert!(failed_records.is_empty());
        assert_eq!(&*failed_consulted, ["fixture/linux-64"]);
        {
            let state = shared.reachable.lock().await;
            assert_eq!(state.root_order, [parent.clone()]);
            assert_eq!(state.covered_roots, BTreeSet::from([parent.clone()]));
            assert_eq!(
                state.record_names,
                BTreeSet::from([parent.clone(), child.clone()])
            );
            assert!(Arc::ptr_eq(
                state.records.as_ref().unwrap(),
                &original_records
            ));
        }

        let expanded_records = vec![
            repo_record("parent", "1.0", &["child >=1"]),
            repo_record("child", "1.0", &[]),
            repo_record("missing", "1.0", &[]),
        ];
        let retried_roots = Arc::new(std::sync::Mutex::new(Vec::new()));
        let retried_roots_for_probe = Arc::clone(&retried_roots);
        let expected_records = expanded_records.clone();
        let (loaded_records, loaded_consulted) = shared
            .records_for_with_loader(&specs, move |_pairs, roots| async move {
                retried_roots_for_probe.lock().unwrap().push(roots);
                Ok((expected_records, vec!["fixture/linux-64".to_string()]))
            })
            .await;

        assert_eq!(loaded_records.len(), 3);
        assert_eq!(&*loaded_consulted, ["fixture/linux-64"]);
        let expected_root_order = vec![parent.clone(), missing.clone()];
        assert_eq!(*failed_roots.lock().unwrap(), [expected_root_order.clone()]);
        assert_eq!(*retried_roots.lock().unwrap(), [expected_root_order]);
        let state = shared.reachable.lock().await;
        assert_eq!(state.root_order, [parent, missing.clone()]);
        assert_eq!(
            state.covered_roots,
            BTreeSet::from([PackageName::from_str("parent").unwrap(), missing])
        );
        assert_eq!(
            state.record_names,
            BTreeSet::from([
                PackageName::from_str("parent").unwrap(),
                child,
                PackageName::from_str("missing").unwrap(),
            ])
        );
    }

    #[test]
    fn exact_detected_virtual_packages_discard_host_cuda_and_other_os_markers() {
        let host_baseline = vec![
            virtual_package("__cuda", "99", ""),
            virtual_package("__osx", "15", "0"),
            virtual_package("__win", "0", "0"),
            virtual_package("__linux", "99", ""),
            virtual_package("__cpython", "3.10", ""),
        ];
        let detected = BTreeMap::from([
            ("archspec".to_string(), "1=x86_64".to_string()),
            ("glibc".to_string(), "2.28".to_string()),
            ("linux".to_string(), "4.18".to_string()),
            ("unix".to_string(), "0=0".to_string()),
        ]);

        let vps = build_virtual_packages_from_baseline(
            "3.11",
            &BTreeMap::new(),
            Some(&detected),
            host_baseline,
        )
        .unwrap();

        assert!(vp_lookup(&vps, "__cuda").is_none());
        assert!(vp_lookup(&vps, "__osx").is_none());
        assert!(vp_lookup(&vps, "__win").is_none());
        assert_eq!(
            vp_lookup(&vps, "__linux").unwrap().version.to_string(),
            "4.18"
        );
        assert_eq!(
            vp_lookup(&vps, "__glibc").unwrap().version.to_string(),
            "2.28"
        );
        let archspec = vp_lookup(&vps, "__archspec").unwrap();
        assert_eq!(archspec.version.to_string(), "1");
        assert_eq!(archspec.build_string, "x86_64");
        let unix = vp_lookup(&vps, "__unix").unwrap();
        assert_eq!(unix.version.to_string(), "0");
        assert_eq!(unix.build_string, "0");
        assert!(
            vp_lookup(&vps, "__cpython")
                .unwrap()
                .version
                .to_string()
                .starts_with("3.11")
        );
    }

    #[test]
    fn exact_detected_virtual_packages_retain_generic_entries() {
        let detected = BTreeMap::from([("cuda_arch".to_string(), "8.6".to_string())]);
        let vps = build_virtual_packages_from_baseline(
            "3.11",
            &BTreeMap::new(),
            Some(&detected),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            vp_lookup(&vps, "__cuda_arch").unwrap().version.to_string(),
            "8.6"
        );
    }

    #[test]
    fn exact_detected_virtual_packages_fill_only_declared_cuda_arch_gap() {
        let host_baseline = vec![
            virtual_package("__cuda", "99", ""),
            virtual_package("__linux", "99", ""),
        ];
        let detected = BTreeMap::from([
            ("archspec".to_string(), "1=x86_64".to_string()),
            ("glibc".to_string(), "2.35".to_string()),
            ("linux".to_string(), "4.18".to_string()),
            ("unix".to_string(), "0=0".to_string()),
        ]);
        let system_requirements = BTreeMap::from([
            ("cuda".to_string(), "12".to_string()),
            ("cuda_arch".to_string(), "8.6".to_string()),
            ("linux".to_string(), "5.15".to_string()),
        ]);

        let vps = build_virtual_packages_from_baseline(
            "3.11",
            &system_requirements,
            Some(&detected),
            host_baseline,
        )
        .unwrap();

        assert!(vp_lookup(&vps, "__cuda").is_none());
        assert_eq!(
            vp_lookup(&vps, "__cuda_arch").unwrap().version.to_string(),
            "8.6"
        );
        assert_eq!(
            vp_lookup(&vps, "__linux").unwrap().version.to_string(),
            "4.18"
        );
    }

    #[test]
    fn contract_qualified_empty_detected_map_does_not_inherit_host_baseline() {
        let host_baseline = vec![
            virtual_package("__archspec", "1", "x86_64"),
            virtual_package("__cuda", "99", ""),
            virtual_package("__osx", "15", "0"),
        ];
        let system_requirements = BTreeMap::from([
            ("archspec".to_string(), "aarch64".to_string()),
            ("libc".to_string(), "2.35".to_string()),
            ("linux".to_string(), "5.15".to_string()),
        ]);
        let detected = BTreeMap::new();

        let legacy = build_virtual_packages_from_baseline(
            "3.10",
            &system_requirements,
            None,
            host_baseline.clone(),
        )
        .unwrap();
        assert!(vp_lookup(&legacy, "__cuda").is_some());
        assert!(vp_lookup(&legacy, "__osx").is_some());

        let vps = build_virtual_packages_from_baseline(
            "3.10",
            &system_requirements,
            Some(&detected),
            host_baseline,
        )
        .unwrap();

        assert!(vp_lookup(&vps, "__cuda").is_none());
        assert!(vp_lookup(&vps, "__osx").is_none());
        assert_eq!(
            vp_lookup(&vps, "__archspec").unwrap().build_string,
            "aarch64"
        );
        assert_eq!(
            vp_lookup(&vps, "__glibc").unwrap().version.to_string(),
            "2.35"
        );
        assert_eq!(
            vp_lookup(&vps, "__linux").unwrap().version.to_string(),
            "5.15"
        );
    }

    #[test]
    fn exact_detected_virtual_packages_reject_invalid_names() {
        let detected = BTreeMap::from([("bad/name".to_string(), "1".to_string())]);
        let error = build_virtual_packages_from_baseline(
            "3.11",
            &BTreeMap::new(),
            Some(&detected),
            Vec::new(),
        )
        .expect_err("an exact contract must not silently omit an invalid virtual package");
        assert!(error.contains("bad/name"), "unexpected error: {error}");
    }

    #[test]
    fn build_virtual_packages_injects_cpython_from_target_python() {
        let sysreqs = std::collections::BTreeMap::new();
        let vps = build_virtual_packages("3.11", &sysreqs);
        let cp = vp_lookup(&vps, "__cpython").expect("__cpython must be present");
        // Version parses as `3.11.0` (zero-extended).
        assert!(cp.version.to_string().starts_with("3.11"));
    }

    #[test]
    fn build_virtual_packages_maps_workspace_cuda_to_double_underscore() {
        // The load-bearing case for v0.37.0 D1: workspace declares
        // `cuda = "12"` in `[feature.gpu.system-requirements]`. Without
        // this mapping, retread sees the build host's `__cuda` (which
        // may be absent or a different version), pixi sees __cuda 12,
        // they disagree on whether a `__cuda >=12` dep is installable.
        let mut sysreqs = std::collections::BTreeMap::new();
        sysreqs.insert("cuda".into(), "12".into());
        let vps = build_virtual_packages("3.11", &sysreqs);
        let cuda = vp_lookup(&vps, "__cuda").expect("__cuda must be injected");
        assert!(
            cuda.version.to_string().starts_with("12"),
            "expected __cuda 12.x, got {}",
            cuda.version,
        );
    }

    #[test]
    fn build_virtual_packages_maps_libc_to_glibc() {
        // pixi's schema names the glibc requirement `libc` (or `glibc`).
        // The matching virtual package is `__glibc` (linux-specific
        // family). Without this mapping, packages that declare
        // `__glibc >=2.35` would see the host's __glibc instead of
        // the workspace's declaration.
        let mut sysreqs = std::collections::BTreeMap::new();
        sysreqs.insert("libc".into(), "2.35".into());
        let vps = build_virtual_packages("3.11", &sysreqs);
        let glibc = vp_lookup(&vps, "__glibc").expect("__glibc must be injected");
        assert_eq!(glibc.version.to_string(), "2.35");
    }

    #[test]
    fn build_virtual_packages_overrides_host_detection() {
        // Workspace value MUST win over host detection. The build host
        // probably has a __glibc version different from the workspace's
        // declared one; the workspace's value is authoritative because
        // that's what pixi uses.
        let mut sysreqs = std::collections::BTreeMap::new();
        sysreqs.insert("libc".into(), "9.99".into());
        let vps = build_virtual_packages("3.11", &sysreqs);
        // Exactly ONE __glibc entry (overriding any host-detected one).
        let count = vps
            .iter()
            .filter(|vp| vp.name.as_normalized() == "__glibc")
            .count();
        assert_eq!(count, 1, "must replace, not append, the host's __glibc");
        let glibc = vp_lookup(&vps, "__glibc").unwrap();
        assert_eq!(glibc.version.to_string(), "9.99");
    }

    #[test]
    fn build_virtual_packages_archspec_encodes_as_build_string() {
        // Archspec values are NOT versions (`x86_64`, `aarch64`); they
        // belong in the build-string slot. Pin that encoding.
        let mut sysreqs = std::collections::BTreeMap::new();
        sysreqs.insert("archspec".into(), "x86_64".into());
        let vps = build_virtual_packages("3.11", &sysreqs);
        let arch = vp_lookup(&vps, "__archspec").expect("__archspec must be injected");
        assert_eq!(arch.build_string, "x86_64");
    }

    #[test]
    fn build_virtual_packages_skips_unknown_keys() {
        // Forward-compat: future pixi schema additions shouldn't kill
        // the solve check. Unknown keys log + skip.
        let mut sysreqs = std::collections::BTreeMap::new();
        sysreqs.insert("something-new-in-pixi".into(), "12".into());
        let vps_baseline = build_virtual_packages("3.11", &std::collections::BTreeMap::new());
        let vps = build_virtual_packages("3.11", &sysreqs);
        // No __something-new-in-pixi (or any other) was added.
        assert_eq!(
            vps.len(),
            vps_baseline.len(),
            "unknown system-requirement keys must not add virtual packages"
        );
    }

    #[test]
    fn solve_selected_records_can_pick_lowest_direct_torch_line() {
        let all_records = vec![
            repo_record("python", "3.11.5", &[]),
            repo_record("pytorch", "2.7.0", &["python >=3.11,<3.12.0a0"]),
            repo_record("pytorch", "2.10.0", &["python >=3.11,<3.12.0a0"]),
            repo_record("pytorch", "2.11.0", &["python >=3.11,<3.12.0a0"]),
            repo_record("pytorch-gpu", "2.7.1", &["pytorch 2.7.0"]),
            repo_record("pytorch-gpu", "2.11.0", &["pytorch 2.11.0"]),
            repo_record("torchvision", "0.22.0", &["pytorch >=2.7.0,<2.8.0a0"]),
            repo_record("torchvision", "0.25.0", &["pytorch >=2.10.0,<2.11.0a0"]),
            repo_record("torchvision", "0.26.0", &["pytorch >=2.11.0,<2.12.0a0"]),
        ];
        let specs = parse_match_specs(&[
            "pytorch-gpu >=2.7.1,<3".to_string(),
            "torchvision >=0.22.0".to_string(),
        ]);
        let solved = solve_selected_records_from_records(
            specs,
            &all_records,
            "3.11",
            ChannelPriority::Strict,
            &BTreeMap::new(),
            SolveStrategy::LowestVersionDirect,
            Vec::new(),
        )
        .expect("torch family should solve coherently");

        let pytorch_gpu = solved
            .iter()
            .find(|r| r.package_record.name.as_normalized() == "pytorch-gpu")
            .expect("pytorch-gpu must be selected");
        let torchvision = solved
            .iter()
            .find(|r| r.package_record.name.as_normalized() == "torchvision")
            .expect("torchvision must be selected");
        let pytorch = solved
            .iter()
            .find(|r| r.package_record.name.as_normalized() == "pytorch")
            .expect("pytorch must be selected");

        assert_eq!(pytorch_gpu.package_record.version.as_str(), "2.7.1");
        assert_eq!(torchvision.package_record.version.as_str(), "0.22.0");
        assert_eq!(pytorch.package_record.version.as_str(), "2.7.0");
    }

    #[test]
    fn selected_package_run_constraint_conflicts_with_installed_package() {
        let mut pandas = repo_record("pandas", "3.0.3", &["python >=3.11,<3.12.0a0"]);
        pandas.package_record.constrains = vec!["numba >=0.60.0".to_string()];
        let all_records = vec![
            repo_record("python", "3.11.5", &[]),
            repo_record("numba", "0.59.1", &["python >=3.11,<3.12.0a0"]),
            repo_record("numba", "0.60.0", &["python >=3.11,<3.12.0a0"]),
            pandas,
        ];
        let specs = parse_match_specs(&[
            "python 3.11.*".to_string(),
            "pandas ==3.0.3".to_string(),
            "numba >=0.59.1,<0.60".to_string(),
        ]);
        let reasons = solve_selected_records_from_records(
            specs,
            &all_records,
            "3.11",
            ChannelPriority::Strict,
            &BTreeMap::new(),
            SolveStrategy::Highest,
            Vec::new(),
        )
        .expect_err("pandas' run constraint must reject numba 0.59");
        assert!(
            reasons.iter().any(|reason| unsat_mentions(reason, "numba")),
            "unsat reasons should name numba: {reasons:?}",
        );
    }

    #[test]
    fn resolvo_unsat_is_typed_as_monotone_probe_proof() {
        let records = vec![
            repo_record("python", "3.11.5", &[]),
            repo_record("fixture-dep", "1.0.0", &[]),
        ];
        let specs = parse_match_specs(&[
            "python 3.11.*".to_string(),
            "fixture-dep ==2.0.0".to_string(),
        ]);

        let failure = solve_selected_records_from_records_for_target(
            specs,
            &records,
            "3.11",
            ChannelPriority::Strict,
            &BTreeMap::new(),
            None,
            SolveStrategy::Highest,
            Vec::new(),
        )
        .expect_err("the unavailable exact version must be unsatisfiable");

        assert!(
            matches!(failure, SharedSolveFailure::Unsolvable(_)),
            "only resolvo's genuine Unsolvable result may authorize block deletion: {failure:?}",
        );
    }

    #[tokio::test]
    #[ignore = "requires conda-forge repodata"]
    async fn sparse_conda_forge_pandas_keeps_numba_run_constraint() {
        let channels = vec![ChannelUrl::from(
            url::Url::parse("https://prefix.dev/conda-forge").unwrap(),
        )];
        let specs = parse_match_specs(&[
            "python 3.11.*".to_string(),
            "pandas >=3.0.3,<4".to_string(),
            "numba >=0.59.1,<0.60".to_string(),
        ]);
        let (records, consulted) =
            load_selected_records_sparse(&channels, "linux-64", &specs).await;
        assert!(
            !consulted.is_empty(),
            "conda-forge repodata was not available"
        );
        let pandas = records
            .iter()
            .filter(|record| record.package_record.name.as_normalized() == "pandas")
            .find(|record| record.package_record.version.as_str() == "3.0.3")
            .expect("pandas 3.0.3 must be present in sparse records");
        assert!(
            pandas.package_record.constrains.iter().any(|constraint| {
                constraint.starts_with("numba ") && constraint.contains(">=0.60")
            }),
            "pandas constraints were {:?}",
            pandas.package_record.constrains,
        );
        let reasons = solve_on_blocking_pool(
            specs,
            records,
            "3.11".to_string(),
            ChannelPriority::Strict,
            BTreeMap::new(),
            SolveStrategy::Highest,
            Vec::new(),
        )
        .await
        .expect_err("the real pandas/numba closure must be unsatisfiable");
        assert!(reasons.iter().any(|reason| unsat_mentions(reason, "numba")));
    }

    /// The cuda-bindings incident, reproduced at the record level: an
    /// auto-routed exact pin (`cuda-bindings ==13.3.1`) whose conda
    /// variant requires `cuda-version >=13,<14` is fed into the SAME
    /// solve as the consuming env's explicit `cuda-version ==12.9`
    /// spec (as `consuming_env_dependencies` would surface it). The
    /// two together must be genuinely UNSAT — proving the transitive
    /// conflict is caught by a real solve, not a by-name range check,
    /// even though `cuda-bindings` never appears in any env's declared
    /// dependency table (only `cuda-version` does).
    #[test]
    fn cuda_bindings_transitive_conflict_is_unsat_against_pinned_cuda_version() {
        let all_records = vec![
            repo_record("python", "3.12.0", &[]),
            repo_record("cuda-version", "12.9", &[]),
            repo_record("cuda-version", "13.0", &[]),
            repo_record(
                "cuda-bindings",
                "13.3.1",
                &["cuda-version >=13,<14.0a0", "python"],
            ),
        ];
        let specs = parse_match_specs(&[
            "cuda-bindings ==13.3.1".to_string(),
            "cuda-version ==12.9".to_string(),
            "python 3.12.*".to_string(),
        ]);
        let result = solve_selected_records_from_records(
            specs,
            &all_records,
            "3.12",
            ChannelPriority::Strict,
            &BTreeMap::new(),
            SolveStrategy::Highest,
            Vec::new(),
        );
        let reasons =
            result.expect_err("cuda-bindings 13.3.1 must conflict with cuda-version ==12.9");
        assert!(
            reasons.iter().any(|r| unsat_mentions(r, "cuda-bindings")),
            "unsat reasons should name cuda-bindings: {reasons:?}"
        );
    }

    /// Sat counterpart: dropping the auto-routed cuda-bindings pin (the
    /// un-route outcome) lets the same consuming-env cuda-version spec
    /// solve cleanly — confirms the conflict is specific to the exact
    /// pin, not the fixture itself.
    #[test]
    fn cuda_version_alone_solves_once_cuda_bindings_is_unrouted() {
        let all_records = vec![
            repo_record("python", "3.12.0", &[]),
            repo_record("cuda-version", "12.9", &[]),
            repo_record("cuda-version", "13.0", &[]),
        ];
        let specs = parse_match_specs(&[
            "cuda-version ==12.9".to_string(),
            "python 3.12.*".to_string(),
        ]);
        let solved = solve_selected_records_from_records(
            specs,
            &all_records,
            "3.12",
            ChannelPriority::Strict,
            &BTreeMap::new(),
            SolveStrategy::Highest,
            Vec::new(),
        )
        .expect("cuda-version alone must solve");
        assert!(
            solved
                .iter()
                .any(|r| r.package_record.name.as_normalized() == "cuda-version"
                    && r.package_record.version.as_str() == "12.9")
        );
    }

    /// Simple substring check mirroring `uv_closure::unsat_reason_names_package`'s
    /// core intent, kept local so this test module doesn't need to
    /// depend on `uv_closure`.
    fn unsat_mentions(reason: &str, name: &str) -> bool {
        reason.to_ascii_lowercase().contains(name)
    }

    /// Warm-start seeding (locked_packages = soft preference) must be
    /// emit-neutral: passing a previously-resolved set as `preferred`
    /// must produce the SAME satisfiable/unsat result and the SAME
    /// selected versions as a cold solve with `preferred = Vec::new()`.
    ///
    /// This is the load-bearing correctness invariant for the warm-start
    /// optimisation: preferences bias version SELECTION (speed/stability)
    /// but never change whether a solution EXISTS or which versions satisfy
    /// the hard constraints the caller emits.
    #[test]
    fn warm_start_locked_packages_is_emit_neutral() {
        let all_records = vec![
            repo_record("python", "3.11.5", &[]),
            repo_record("numpy", "1.24.0", &["python >=3.11,<3.12.0a0"]),
            repo_record("numpy", "1.26.0", &["python >=3.11,<3.12.0a0"]),
            repo_record(
                "scipy",
                "1.10.0",
                &["numpy >=1.24,<2", "python >=3.11,<3.12.0a0"],
            ),
            repo_record(
                "scipy",
                "1.12.0",
                &["numpy >=1.26,<2", "python >=3.11,<3.12.0a0"],
            ),
        ];
        let specs =
            parse_match_specs(&["numpy >=1.24.0".to_string(), "scipy >=1.10.0".to_string()]);

        // Cold solve (no preferred packages).
        let cold = solve_selected_records_from_records(
            specs.clone(),
            &all_records,
            "3.11",
            ChannelPriority::Strict,
            &BTreeMap::new(),
            SolveStrategy::Highest,
            Vec::new(),
        )
        .expect("cold solve should succeed");

        // Warm-start: seed with the cold result as preferred versions.
        // The solver should produce the same SAT result.
        let warm = solve_selected_records_from_records(
            specs,
            &all_records,
            "3.11",
            ChannelPriority::Strict,
            &BTreeMap::new(),
            SolveStrategy::Highest,
            cold.clone(),
        )
        .expect("warm-start solve should succeed with same result");

        // Both solves must agree on which packages are selected.
        let mut cold_names: Vec<&str> = cold
            .iter()
            .map(|r| r.package_record.name.as_normalized())
            .collect();
        cold_names.sort_unstable();
        let mut warm_names: Vec<&str> = warm
            .iter()
            .map(|r| r.package_record.name.as_normalized())
            .collect();
        warm_names.sort_unstable();
        assert_eq!(
            cold_names, warm_names,
            "warm-start must select the same package set as cold solve"
        );

        // Versions must match too.
        for cold_rec in &cold {
            let warm_rec = warm
                .iter()
                .find(|r| r.package_record.name == cold_rec.package_record.name)
                .expect("warm solve must include every package the cold solve selected");
            assert_eq!(
                cold_rec.package_record.version,
                warm_rec.package_record.version,
                "warm-start must select the same version for {}",
                cold_rec.package_record.name.as_normalized(),
            );
        }
    }

    /// Warm-start must also be emit-neutral when the solve is UNSAT:
    /// passing stale/irrelevant preferred records must not change the
    /// UNSAT verdict (locked_packages cannot rescue an impossible solve).
    #[test]
    fn warm_start_does_not_rescue_unsat() {
        let all_records = vec![
            repo_record("python", "3.11.5", &[]),
            // pkg-a 1.0 needs pkg-b ==1.0; forcing pkg-a ==1.0 and
            // pkg-b ==2.0 simultaneously is irreconcilable.
            repo_record("pkg-a", "1.0", &["pkg-b ==1.0"]),
            repo_record("pkg-b", "1.0", &[]),
            repo_record("pkg-a", "2.0", &["pkg-b ==2.0"]),
            repo_record("pkg-b", "2.0", &[]),
        ];
        let conflicting_specs =
            parse_match_specs(&["pkg-a ==1.0".to_string(), "pkg-b ==2.0".to_string()]);

        // With pkg-a ==1.0 and pkg-b ==2.0: pkg-a 1.0 needs pkg-b ==1.0
        // but we pin pkg-b ==2.0 — genuine conflict.
        let cold_err = solve_selected_records_from_records(
            conflicting_specs.clone(),
            &all_records,
            "3.11",
            ChannelPriority::Strict,
            &BTreeMap::new(),
            SolveStrategy::Highest,
            Vec::new(),
        );
        assert!(
            cold_err.is_err(),
            "cold solve of conflicting specs must be UNSAT"
        );

        // Warm-start with the records that DO exist — seeding preferences
        // must NOT change the UNSAT verdict.
        let stale_preferred = vec![
            repo_record("pkg-a", "1.0", &["pkg-b ==1.0"]),
            repo_record("pkg-b", "1.0", &[]),
        ];
        let warm_err = solve_selected_records_from_records(
            conflicting_specs,
            &all_records,
            "3.11",
            ChannelPriority::Strict,
            &BTreeMap::new(),
            SolveStrategy::Highest,
            stale_preferred,
        );
        assert!(
            warm_err.is_err(),
            "warm-start must not rescue a genuinely-UNSAT solve"
        );
    }
}
