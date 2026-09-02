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
        &SolveTarget {
            python: target_python,
            channel_priority,
            system_requirements,
            detected_virtual_packages: None,
            strategy,
            subdir: None,
        },
        preferred,
    )
    .map_err(SharedSolveFailure::into_reasons)
}

pub(crate) struct SolveTarget<'a> {
    python: &'a str,
    channel_priority: ChannelPriority,
    system_requirements: &'a BTreeMap<String, String>,
    detected_virtual_packages: Option<&'a BTreeMap<String, String>>,
    strategy: SolveStrategy,
    subdir: Option<&'a str>,
}

impl<'a> SolveTarget<'a> {
    pub(crate) fn new(
        python: &'a str,
        channel_priority: ChannelPriority,
        system_requirements: &'a BTreeMap<String, String>,
        detected_virtual_packages: Option<&'a BTreeMap<String, String>>,
        strategy: SolveStrategy,
    ) -> Self {
        Self {
            python,
            channel_priority,
            system_requirements,
            detected_virtual_packages,
            strategy,
            subdir: None,
        }
    }

    /// Declare the conda subdir this solve targets, so the platform virtual
    /// packages every record on that subdir depends on (`__linux`/`__unix`,
    /// `__win`, `__osx`) can be supplied without host detection. Callers that
    /// omit it keep the previous behavior.
    pub(crate) fn with_subdir(mut self, subdir: &'a str) -> Self {
        self.subdir = Some(subdir);
        self
    }
}

fn solve_selected_records_from_records_for_target(
    parsed_specs: Vec<MatchSpec>,
    all_records: &[RepoDataRecord],
    target: &SolveTarget<'_>,
    preferred: Vec<RepoDataRecord>,
) -> std::result::Result<Vec<RepoDataRecord>, SharedSolveFailure> {
    let virtual_packages = build_virtual_packages_for_subdir_target(
        target.python,
        target.system_requirements,
        target.detected_virtual_packages,
        target.subdir,
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
        channel_priority: target.channel_priority,
        exclude_newer: None,
        min_age: None,
        strategy: target.strategy,
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
        SolveTarget {
            python: &target_python,
            channel_priority,
            system_requirements: &system_requirements,
            detected_virtual_packages: None,
            strategy,
            subdir: None,
        },
        preferred,
    )
    .await
    .map_err(SharedSolveFailure::into_reasons)
}

async fn solve_on_blocking_pool_for_target(
    parsed_specs: Vec<MatchSpec>,
    records: Arc<[RepoDataRecord]>,
    probe_grant: Option<crate::thread_budget::ProbePoolGrant>,
    target: SolveTarget<'_>,
    preferred: Vec<RepoDataRecord>,
) -> std::result::Result<Vec<RepoDataRecord>, SharedSolveFailure> {
    let probe_task = match &probe_grant {
        Some(grant) => Some(grant.acquire_task().await),
        None => None,
    };
    let t_solve = std::time::Instant::now();
    let specs_count = parsed_specs.len();
    let records_count = records.len();
    let target_python = target.python.to_string();
    let system_requirements = target.system_requirements.clone();
    let detected_virtual_packages = target.detected_virtual_packages.cloned();
    let channel_priority = target.channel_priority;
    let strategy = target.strategy;
    let subdir = target.subdir.map(str::to_string);
    let result = tokio::task::spawn_blocking(move || {
        // Keep the bundle's one coordinated grant alive until this CPU task
        // actually exits, even if its async waiter is cancelled.
        let _probe_grant = probe_grant;
        let _probe_task = probe_task;
        solve_selected_records_from_records_for_target(
            parsed_specs,
            &records,
            &SolveTarget {
                python: &target_python,
                channel_priority,
                system_requirements: &system_requirements,
                detected_virtual_packages: detected_virtual_packages.as_ref(),
                strategy,
                subdir: subdir.as_deref(),
            },
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
/// many counterfactual solves. Sparse handles are shared once, while each
/// exact root-name set gets its own immutable reachable-record snapshot.
/// A larger route question must never change the candidate universe of an
/// earlier, narrower question: doing so can change its resolvo verdict.
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
    snapshots: BTreeMap<Vec<PackageName>, ReachableSnapshot>,
}

#[derive(Clone)]
struct ReachableSnapshot {
    records: Arc<[RepoDataRecord]>,
    consulted: Arc<[String]>,
    /// Content fingerprint of `records` -- the candidate universe this
    /// question can actually see. Computed once per root-name set and
    /// reused, so keying a memoized verdict on it costs one BTreeMap
    /// lookup, not a rehash.
    universe: Arc<str>,
}

/// Hex sha256 over the REACHABLE candidate universe: every field of every
/// record that resolvo can act on, in a canonical order.
///
/// This is the sound, fine-grained replacement for
/// `crate::repodata::repodata_identity` as a route-probe verdict-cache key.
/// That function hashed the on-disk repodata cache file's LENGTH and MTIME,
/// so an identical document re-fetched after the 30-minute TTL produced a
/// different key and discarded every memoized verdict -- measured directly:
/// jobs 5598763 arm A and arm B ran on ONE node in ONE job against ONE
/// manifest and differed only in which directory held the repodata cache,
/// and every one of their 14 bundle validity keys differed. Job 5611846
/// (fresh workspace, warm shared caches) then discarded 13 of 14 verdict
/// files and re-executed all 315 probes.
///
/// Hashing the whole repodata document instead would not have helped: the
/// conda-forge linux-64 document measurably changes within the hour
/// (637,578,869 bytes at 06:54 EDT vs 637,595,538 at 08:02 EDT on
/// 2026-09-02, different sha256). What a verdict actually depends on is the
/// transitive closure `load_records_recursive` walks from this question's
/// exact root names -- an upload of an unrelated package cannot reach it,
/// and a new version of a reachable package DOES change these bytes because
/// the walk loads it. So this is both sound and stable across jobs.
fn reachable_universe_digest(records: &[RepoDataRecord]) -> Arc<str> {
    use sha2::{Digest, Sha256};
    let mut lines: Vec<String> = records
        .iter()
        .map(|record| {
            let package = &record.package_record;
            format!(
                "{}\u{1}{}\u{1}{}\u{1}{}\u{1}{}\u{1}{}\u{1}{}\u{1}{}\u{1}{}",
                record.file_name,
                record.channel.as_deref().unwrap_or(""),
                package.name.as_normalized(),
                package.version.as_str(),
                package.build,
                package.build_number,
                package.subdir,
                package.depends.join("\u{2}"),
                package.constrains.join("\u{2}"),
            )
        })
        .collect();
    lines.sort();
    lines.dedup();
    let mut hasher = Sha256::new();
    hasher.update(b"retread-reachable-universe-v1\0");
    for line in &lines {
        hasher.update(line.as_bytes());
        hasher.update([0u8]);
    }
    Arc::from(format!("{:x}", hasher.finalize()).as_str())
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
        let snapshot = self
            .snapshot_for_with_loader(parsed_specs, load_selected_records_sparse_from_pairs)
            .await;
        (snapshot.records, snapshot.consulted)
    }

    /// The reachable universe fingerprint for one question. `None` when no
    /// repodata could be loaded at all -- the verdict would be `Skipped`,
    /// which is never memoized anyway.
    async fn universe_for(&self, parsed_specs: &[MatchSpec]) -> Option<Arc<str>> {
        let snapshot = self
            .snapshot_for_with_loader(parsed_specs, load_selected_records_sparse_from_pairs)
            .await;
        (!snapshot.records.is_empty()).then_some(snapshot.universe)
    }

    async fn snapshot_for_with_loader<Load, LoadFuture>(
        &self,
        parsed_specs: &[MatchSpec],
        load: Load,
    ) -> ReachableSnapshot
    where
        Load: FnOnce(Arc<[SparsePair]>, Vec<PackageName>) -> LoadFuture,
        LoadFuture: std::future::Future<Output = SparseLoadResult>,
    {
        let mut requested_roots = exact_root_names(parsed_specs);
        requested_roots.sort();
        if let Some(snapshot) = self
            .reachable
            .lock()
            .await
            .snapshots
            .get(&requested_roots)
            .cloned()
        {
            tracing::debug!(
                records = snapshot.records.len(),
                requested_roots = requested_roots.len(),
                "bench: reusing question-scoped sparse reachable records",
            );
            return snapshot;
        }

        let pairs = self.pairs().await;
        let (records, consulted) = match load(pairs, requested_roots.clone()).await {
            Ok(loaded) => loaded,
            // Gateway errors and spawn-blocking panics both take this path.
            // Keep the last good snapshot intact so a later probe can retry.
            Err(consulted) => {
                return ReachableSnapshot {
                    records: Arc::from([]),
                    consulted: consulted.into(),
                    universe: Arc::from(""),
                };
            }
        };
        let universe = reachable_universe_digest(&records);
        let records: Arc<[RepoDataRecord]> = records.into();
        let consulted: Arc<[String]> = consulted.into();
        let snapshot = ReachableSnapshot {
            records,
            consulted,
            universe,
        };
        self.reachable
            .lock()
            .await
            .snapshots
            .insert(requested_roots, snapshot.clone());
        snapshot
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
    build_virtual_packages_for_subdir_target(
        target_python,
        system_requirements,
        detected_virtual_packages,
        None,
    )
}

/// [`build_virtual_packages_for_target`] plus the conda subdir the solve
/// targets.
///
/// A contract-qualified solve whose detected set is empty previously derived
/// its entire virtual-package set from `[system-requirements]`. Nobody writes
/// `linux = ...` there — the platform is implied by the subdir — so the set
/// came out with no `__linux`/`__unix`, and every record on a linux subdir
/// depends on `__linux`. The solve was therefore unsatisfiable for a reason
/// that had nothing to do with the request (observed as `cuda-toolkit 12.9.*
/// would require __linux *, for which no candidates were found`). Supplying
/// the platform packages from the subdir is target-derived, not host-derived,
/// so it does not reintroduce host leakage into an exact contract.
pub(crate) fn build_virtual_packages_for_subdir_target(
    target_python: &str,
    system_requirements: &BTreeMap<String, String>,
    detected_virtual_packages: Option<&BTreeMap<String, String>>,
    subdir: Option<&str>,
) -> Result<Vec<GenericVirtualPackage>, String> {
    let host_baseline = if detected_virtual_packages.is_some() {
        Vec::new()
    } else {
        // F14: detect with the workspace's declared floors substituted for
        // anything this host cannot satisfy, and warn once with the exports
        // pixi itself needs. `VirtualPackageOverrides::default()` is `None`
        // for every field, so the pre-F14 call also ignored an explicit
        // `CONDA_OVERRIDE_*`; `solve_overrides` honours it.
        match rattler_virtual_packages::VirtualPackage::detect(
            &crate::virtual_override::solve_overrides(system_requirements),
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
    let mut virtual_packages = build_virtual_packages_from_baseline(
        target_python,
        system_requirements,
        detected_virtual_packages,
        host_baseline,
    )?;
    if let Some(subdir) = subdir {
        add_platform_virtual_packages(&mut virtual_packages, subdir, system_requirements)?;
    }
    Ok(virtual_packages)
}

/// Supply the OS virtual packages implied by `subdir` when the assembled set
/// does not already carry them. Never overrides an entry an authoritative
/// detected set or an explicit system-requirement already provided.
fn add_platform_virtual_packages(
    virtual_packages: &mut Vec<GenericVirtualPackage>,
    subdir: &str,
    system_requirements: &BTreeMap<String, String>,
) -> Result<(), String> {
    let present = |packages: &Vec<GenericVirtualPackage>, name: &str| {
        packages.iter().any(|vp| vp.name.as_normalized() == name)
    };
    // Pixi's own default floor for `linux`, used when the workspace does not
    // declare one. Conda-forge's `__linux` dependencies are overwhelmingly
    // unversioned, so this only has to clear the rare versioned floor.
    const DEFAULT_LINUX: &str = "4.18";
    let (os_key, os_value): (&str, &str) = if subdir.starts_with("linux") {
        (
            "linux",
            system_requirements
                .get("linux")
                .map_or(DEFAULT_LINUX, String::as_str),
        )
    } else if subdir.starts_with("osx") {
        (
            "macos",
            system_requirements
                .get("macos")
                .map_or("11.0", String::as_str),
        )
    } else if subdir.starts_with("win") {
        ("windows", "0")
    } else {
        // `noarch` and unknown subdirs carry no OS contract of their own.
        return Ok(());
    };
    if !present(
        virtual_packages,
        &format!(
            "__{}",
            match os_key {
                "macos" => "osx",
                "windows" => "win",
                other => other,
            }
        ),
    ) {
        insert_virtual_package(virtual_packages, os_key, os_value, false)?;
    }
    if (subdir.starts_with("linux") || subdir.starts_with("osx"))
        && !present(virtual_packages, "__unix")
    {
        insert_virtual_package(virtual_packages, "unix", "", false)?;
    }
    Ok(())
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

/// The sysroot chosen independently of the compiler solve.
///
/// Keep both representations: the parsed glibc pair drives wheel tags and
/// cache identities, while `conda_version` is the exact channel version that
/// must be pinned in the coherent toolchain solve. Reconstructing the latter
/// from `(major, minor)` would be wrong if conda-forge publishes `2.34.0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SelectedSysroot {
    pub(crate) conda_version: String,
    pub(crate) glibc_floor: (u32, u32),
}

/// Concrete coherent solution used to render the rattler-build debug recipe.
#[derive(Debug, Clone)]
pub(crate) struct HermeticBuildSolve {
    pub(crate) sysroot: SelectedSysroot,
    pub(crate) records: Vec<RepoDataRecord>,
}

/// Select the newest conda-forge sysroot whose glibc version does not exceed
/// the requested target floor. This happens *before* the compiler solve: an
/// all-tools solve is otherwise free to silently downgrade the sysroot to
/// satisfy another root, weakening the requested "newest <= floor" rule.
fn select_newest_compatible_sysroot(
    records: &[RepoDataRecord],
    target_floor: (u32, u32),
) -> std::result::Result<SelectedSysroot, String> {
    let mut candidates = records
        .iter()
        .filter(|record| record.package_record.name.as_normalized() == "sysroot_linux-64")
        .filter_map(|record| {
            let rendered = record.package_record.version.as_str();
            let glibc_floor = crate::glibc::parse_glibc_version(&rendered)?;
            if glibc_floor > target_floor {
                return None;
            }
            let version = match Version::from_str(&rendered) {
                Ok(version) => version,
                Err(error) => {
                    tracing::debug!(
                        version = %rendered,
                        error = %error,
                        "ignoring unparseable sysroot_linux-64 version",
                    );
                    return None;
                }
            };
            Some((version, glibc_floor, rendered.to_string()))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.0.cmp(&right.0));
    let Some((_version, glibc_floor, conda_version)) = candidates.pop() else {
        return Err(format!(
            "missing compatible sysroot_linux-64: conda-forge has no sysroot at or below target glibc floor {}",
            crate::glibc::format_glibc(target_floor),
        ));
    };
    Ok(SelectedSysroot {
        conda_version,
        glibc_floor,
    })
}

fn parse_match_specs_strict(specs: &[String]) -> std::result::Result<Vec<MatchSpec>, Vec<String>> {
    specs
        .iter()
        .map(|raw| {
            MatchSpec::from_str(raw, ParseStrictness::Lenient)
                .map_err(|error| format!("invalid hermetic-build match spec `{raw}`: {error}"))
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| vec![error])
}

fn hermetic_toolchain_specs(
    python_minor: &str,
    sysroot_version: &str,
    cuda_version: Option<&str>,
) -> Vec<String> {
    let mut specs = vec![
        // Keep the native ABI line stable.  Exact builds are selected below
        // and included in the environment digest, while these root bounds
        // prevent a newly-published compiler major from silently changing the
        // C++ ABI/runtime policy for an otherwise identical request.
        "gcc_linux-64 13.*".to_string(),
        "gxx_linux-64 13.*".to_string(),
        format!("sysroot_linux-64 =={sysroot_version}"),
        format!("python {python_minor}.*"),
        "auditwheel 6.*".to_string(),
        // conda-forge removed its patchelf 0.18 builds as broken; the channel
        // currently tops out at 0.17.2. Accept 0.17.2+ and allow a future
        // fixed 0.18 without silently jumping a major.
        "patchelf >=0.17.2,<0.19".to_string(),
        "cmake >=3.20,<4".to_string(),
        "make 4.*".to_string(),
        "ninja 1.*".to_string(),
    ];
    if let Some(version) = cuda_version {
        specs.push(if version.is_empty() {
            "cuda-nvcc_linux-64 12.*".to_string()
        } else {
            format!("cuda-nvcc_linux-64 {version}.*")
        });
    }
    specs
}

fn solve_hermetic_build_environment_from_records(
    records: &[RepoDataRecord],
    target_floor: (u32, u32),
    python_minor: &str,
    cuda_version: Option<&str>,
) -> std::result::Result<HermeticBuildSolve, Vec<String>> {
    let sysroot =
        select_newest_compatible_sysroot(records, target_floor).map_err(|error| vec![error])?;
    let specs = hermetic_toolchain_specs(python_minor, &sysroot.conda_version, cuda_version);
    let parsed_specs = parse_match_specs_strict(&specs)?;
    let system_requirements = BTreeMap::new();
    // The hermetic toolchain is always solved for linux-64. Declare that
    // subdir so the platform virtual packages (`__linux`/`__unix`) every
    // record on it depends on come from the target rather than from host
    // detection: with an empty system-requirements map and no detected set,
    // a host that reports no virtual packages leaves the solve unable to
    // satisfy `__linux`, and resolvo reports that as the toolchain roots
    // having no viable candidates ("No candidates were found for
    // gcc_linux-64 13.*") rather than as a missing platform package.
    let solved = solve_selected_records_from_records_for_target(
        parsed_specs,
        records,
        &SolveTarget::new(
            python_minor,
            ChannelPriority::Strict,
            &system_requirements,
            None,
            SolveStrategy::Highest,
        )
        .with_subdir("linux-64"),
        Vec::new(),
    )
    .map_err(|failure| {
        vec![format!(
            "hermetic compiler environment solve failed with sysroot_linux-64 =={}: {}",
            sysroot.conda_version,
            failure.into_reasons().join("; "),
        )]
    })?;

    let mut required = vec![
        "gcc_linux-64",
        "gxx_linux-64",
        "sysroot_linux-64",
        "python",
        "auditwheel",
        "patchelf",
        "cmake",
        "make",
        "ninja",
    ];
    if cuda_version.is_some() {
        required.push("cuda-nvcc_linux-64");
    }
    for name in required {
        if !solved
            .iter()
            .any(|record| record.package_record.name.as_normalized() == name)
        {
            return Err(vec![format!(
                "hermetic compiler environment solve omitted required root `{name}`"
            )]);
        }
    }

    Ok(HermeticBuildSolve {
        sysroot,
        records: solved,
    })
}

/// The (channel, subdir) pairs the hermetic solve REQUIRES, minus the pairs
/// actually consulted. Labels mirror `sparse_pairs`'s `"<channel>/<subdir>"`
/// format. Nonempty means the loaded record set is a partial channel view and
/// must not be solved against.
fn missing_hermetic_pairs(channels: &[ChannelUrl], consulted: &[String]) -> Vec<String> {
    crate::repodata::channel_subdir_pairs(channels, "linux-64")
        .into_iter()
        .map(|(channel_url, subdir)| format!("{channel_url}/{subdir}"))
        .filter(|label| !consulted.iter().any(|seen| seen == label))
        .collect()
}

/// Solve the compiler environment against conda-forge's sparse repodata.
///
/// Stage one loads all candidate roots and selects the newest compatible
/// `sysroot_linux-64` independently. Stage two exact-pins that version in one
/// coherent gcc/gxx/Python/(optional CUDA) solve.
pub(crate) async fn solve_hermetic_build_environment(
    target_floor: (u32, u32),
    python_minor: &str,
    cuda_version: Option<&str>,
) -> std::result::Result<HermeticBuildSolve, Vec<String>> {
    let channels = vec![ChannelUrl::from(
        url::Url::parse("https://prefix.dev/conda-forge")
            .expect("the built-in conda-forge channel URL is valid"),
    )];
    let mut roots = vec![
        "gcc_linux-64".to_string(),
        "gxx_linux-64".to_string(),
        "sysroot_linux-64".to_string(),
        format!("python {python_minor}.*"),
        "auditwheel".to_string(),
        "patchelf".to_string(),
        "cmake <4".to_string(),
        "make".to_string(),
        "ninja".to_string(),
    ];
    if let Some(version) = cuda_version {
        roots.push(if version.is_empty() {
            "cuda-nvcc_linux-64".to_string()
        } else {
            format!("cuda-nvcc_linux-64 {version}.*")
        });
    }
    let parsed_roots = parse_match_specs_strict(&roots)?;
    // Repodata can be transiently unavailable (index hiccup, cold cache under
    // memory pressure). Since failures are no longer memoized, retry with
    // backoff before declaring the toolchain unsolvable.
    let mut records = Vec::new();
    let mut consulted = Vec::new();
    for (attempt, delay_s) in [0u64, 5, 15].into_iter().enumerate() {
        if delay_s > 0 {
            tracing::warn!(
                attempt = attempt + 1,
                delay_s,
                "hermetic-build repodata unavailable; retrying after backoff"
            );
            tokio::time::sleep(std::time::Duration::from_secs(delay_s)).await;
        }
        (records, consulted) =
            load_selected_records_sparse(&channels, "linux-64", &parsed_roots).await;
        if !records.is_empty() && missing_hermetic_pairs(&channels, &consulted).is_empty() {
            break;
        }
    }
    if records.is_empty() {
        let detail = if consulted.is_empty() {
            "no conda-forge sparse repodata was available".to_string()
        } else {
            format!("consulted {}", consulted.join(", "))
        };
        return Err(vec![format!(
            "missing sysroot_linux-64: unable to load hermetic-build records ({detail})"
        )]);
    }
    // A PARTIAL channel view must be a hard error, not a solver mystery.
    // `sysroot_linux-64` lives on noarch while the compilers live on
    // linux-64: with only the noarch pair consulted (the linux-64 fetch
    // failed, or its disk cache was corrupt), stage one still selects a
    // sysroot and stage two then reports "No candidates were found for
    // gcc_linux-64 13.*" -- a misdirection that cost a full night. Name the
    // real defect instead.
    let missing = missing_hermetic_pairs(&channels, &consulted);
    if !missing.is_empty() {
        return Err(vec![format!(
            "hermetic compiler solve aborted: conda-forge repodata incomplete; \
             unconsulted pair(s): {} (fetch failed or corrupt disk cache); \
             refusing to solve against a partial channel view",
            missing.join(", "),
        )]);
    }

    let python_minor = python_minor.to_string();
    let cuda_version = cuda_version.map(str::to_string);
    tokio::task::spawn_blocking(move || {
        solve_hermetic_build_environment_from_records(
            &records,
            target_floor,
            &python_minor,
            cuda_version.as_deref(),
        )
    })
    .await
    .map_err(|error| vec![format!("hermetic compiler solve task panicked: {error}")])?
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
        SolveTarget {
            python: target_python,
            channel_priority,
            system_requirements,
            detected_virtual_packages,
            strategy,
            subdir: Some(target_subdir),
        },
        Vec::new(),
    )
    .await
    .map_err(SharedSolveFailure::into_reasons)
}

/// Bundle-scoped form of [`solve_selected_records_for_target`] that shares
/// sparse handles and immutable reachable-record snapshots across probes.
pub(crate) async fn solve_selected_records_for_target_shared(
    shared: &SharedSparseSolveData,
    specs: &[CondaMatchSpec],
    probe_grant: Option<crate::thread_budget::ProbePoolGrant>,
    target: &SolveTarget<'_>,
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
        SolveTarget {
            python: target.python,
            channel_priority: target.channel_priority,
            system_requirements: target.system_requirements,
            detected_virtual_packages: target.detected_virtual_packages,
            strategy: target.strategy,
            subdir: target.subdir,
        },
        Vec::new(),
    )
    .await
}

/// Content fingerprint of the candidate universe this exact question can
/// reach, for use as a route-probe verdict-cache key. Reuses (and populates)
/// the same question-scoped snapshot the solve itself consumes, so it adds no
/// repodata work. `None` means no repodata was available.
pub(crate) async fn reachable_universe_digest_shared(
    shared: &SharedSparseSolveData,
    specs: &[CondaMatchSpec],
) -> Option<Arc<str>> {
    let rendered_specs: Vec<String> = specs.iter().map(ToString::to_string).collect();
    let parsed_specs = parse_match_specs(&rendered_specs);
    shared.universe_for(&parsed_specs).await
}

/// Populate the exact-question reachable-record snapshot without running resolvo.
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

    /// Guard for the 4.10.77 root fix: a hermetic solve where the linux-64
    /// pair dropped out (fetch failure / corrupt disk cache) must be named a
    /// repodata problem, never allowed through to a "no candidates" solver
    /// misdirection. `sysroot_linux-64` is noarch, the compilers are
    /// linux-64, so a noarch-only view passes stage one and lies in stage two.
    #[test]
    fn missing_hermetic_pairs_flags_partial_channel_views() {
        let channels = vec![ChannelUrl::from(
            Url::parse("https://prefix.dev/conda-forge").unwrap(),
        )];
        let complete = vec![
            "https://prefix.dev/conda-forge/linux-64".to_string(),
            "https://prefix.dev/conda-forge/noarch".to_string(),
        ];
        assert!(missing_hermetic_pairs(&channels, &complete).is_empty());

        let noarch_only = vec!["https://prefix.dev/conda-forge/noarch".to_string()];
        assert_eq!(
            missing_hermetic_pairs(&channels, &noarch_only),
            vec!["https://prefix.dev/conda-forge/linux-64".to_string()],
        );

        assert_eq!(missing_hermetic_pairs(&channels, &[]).len(), 2);
    }

    #[test]
    fn newest_sysroot_at_or_below_floor_is_selected_numerically() {
        let records = vec![
            repo_record("sysroot_linux-64", "2.9", &[]),
            repo_record("sysroot_linux-64", "2.17", &[]),
            repo_record("sysroot_linux-64", "2.28", &[]),
            repo_record("sysroot_linux-64", "2.34", &[]),
            repo_record("sysroot_linux-64", "2.39", &[]),
        ];

        let selected = select_newest_compatible_sysroot(&records, (2, 34)).unwrap();
        assert_eq!(selected.conda_version, "2.34");
        assert_eq!(selected.glibc_floor, (2, 34));
    }

    #[test]
    fn missing_compatible_sysroot_names_sysroot_linux_64_and_floor() {
        let records = vec![repo_record("sysroot_linux-64", "2.39", &[])];
        let error = select_newest_compatible_sysroot(&records, (2, 34)).unwrap_err();

        assert!(
            error.contains("sysroot_linux-64"),
            "unexpected error: {error}"
        );
        assert!(error.contains("2.34"), "unexpected error: {error}");
    }

    #[test]
    fn hermetic_solve_floor_2_34_python_3_11_exact_pins_newest_compatible_sysroot() {
        let records = vec![
            repo_record("sysroot_linux-64", "2.17", &[]),
            repo_record("sysroot_linux-64", "2.28", &[]),
            repo_record("sysroot_linux-64", "2.39", &[]),
            repo_record("python", "3.10.15", &[]),
            repo_record("python", "3.11.15", &[]),
            repo_record("gcc_linux-64", "13.2.0", &["sysroot_linux-64"]),
            repo_record(
                "gxx_linux-64",
                "13.2.0",
                &["gcc_linux-64 ==13.2.0", "sysroot_linux-64"],
            ),
            repo_record("auditwheel", "6.7.0", &["python >=3.10"]),
            repo_record("patchelf", "0.18.0", &[]),
            repo_record("cmake", "3.31.6", &[]),
            repo_record("make", "4.4.1", &[]),
            repo_record("ninja", "1.13.1", &[]),
        ];

        let solved = solve_hermetic_build_environment_from_records(&records, (2, 34), "3.11", None)
            .expect("synthetic compiler environment should solve");
        assert_eq!(solved.sysroot.conda_version, "2.28");
        assert_eq!(solved.sysroot.glibc_floor, (2, 28));
        assert!(solved.records.iter().any(|record| {
            record.package_record.name.as_normalized() == "python"
                && record.package_record.version.as_str() == "3.11.15"
        }));
        assert!(solved.records.iter().any(|record| {
            record.package_record.name.as_normalized() == "sysroot_linux-64"
                && record.package_record.version.as_str() == "2.28"
        }));
        assert!(solved.records.iter().any(|record| {
            record.package_record.name.as_normalized() == "ninja"
                && record.package_record.version.as_str() == "1.13.1"
        }));
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

    /// GUARD for the measured p5u defect. The route-probe verdict key must be
    /// a function of the candidate universe's CONTENT and of nothing on the
    /// filesystem: not the workspace path, not the repodata cache directory,
    /// not an mtime. Job 5611846 (fresh workspace, WARM shared caches) locked
    /// in 2366s against arm C's 69s because the old key fed
    /// `repodata_identity` -- the repodata cache file's length and mtime --
    /// into the file-level validity key, so 13 of 14 verdict files were
    /// discarded and all 315 probes re-ran against 116 hits.
    ///
    /// This test CAN fail: it fails the moment any path- or time-derived
    /// input is folded back into the fingerprint.
    #[test]
    fn universe_digest_is_content_keyed_not_path_or_time_keyed() {
        let records = vec![
            repo_record("python", "3.12.1", &[]),
            repo_record("numpy", "2.1.0", &["python >=3.12,<3.13.0a0"]),
        ];
        let baseline = reachable_universe_digest(&records);

        // (1) Same content, different order in memory -> same digest.
        let mut shuffled = records.clone();
        shuffled.reverse();
        assert_eq!(
            baseline,
            reachable_universe_digest(&shuffled),
            "record order must not change the universe fingerprint",
        );

        // (2) Two different "workspaces"/cache roots -> same digest. The
        // digest must not consult the filesystem at all.
        let mut env_dirs = Vec::new();
        for tag in ["ws-a-11111", "ws-b-22222"] {
            let dir = std::env::temp_dir().join(format!(
                "retread-p5u-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            std::fs::create_dir_all(&dir).expect("temp workspace");
            // Byte-identical pack source, written at two different paths and
            // at two different times.
            std::fs::write(dir.join("pack-source.txt"), b"pack source bytes\n")
                .expect("pack source");
            env_dirs.push(dir);
        }
        let digests: Vec<Arc<str>> = env_dirs
            .iter()
            .map(|dir| {
                let _ = std::fs::metadata(dir.join("pack-source.txt"));
                reachable_universe_digest(&records)
            })
            .collect();
        assert_eq!(
            digests[0], digests[1],
            "two workspaces at different paths with identical sources must \
             produce the identical universe fingerprint",
        );
        assert_eq!(baseline, digests[0]);
        for dir in &env_dirs {
            std::fs::remove_dir_all(dir).ok();
        }

        // (3) A one-field change in the reachable universe DOES change it.
        let mut bumped = records.clone();
        bumped[1].package_record.depends = vec!["python >=3.12,<3.14.0a0".to_string()];
        assert_ne!(
            baseline,
            reachable_universe_digest(&bumped),
            "a changed dependency must change the universe fingerprint",
        );
        let mut added = records.clone();
        added.push(repo_record("scipy", "1.14.0", &["numpy >=2"]));
        assert_ne!(
            baseline,
            reachable_universe_digest(&added),
            "a new reachable candidate must change the universe fingerprint",
        );
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
    async fn shared_sparse_solve_data_reuses_exact_question_snapshot() {
        let records: Arc<[RepoDataRecord]> = vec![
            repo_record("python", "3.11.5", &[]),
            repo_record("numpy", "2.1.0", &["python >=3.11,<3.12"]),
        ]
        .into();
        let roots = vec![
            PackageName::from_str("numpy").unwrap(),
            PackageName::from_str("python").unwrap(),
        ];
        let shared = shared_sparse_fixture(ReachableState {
            snapshots: BTreeMap::from([(
                roots,
                ReachableSnapshot {
                    records: Arc::clone(&records),
                    consulted: Arc::from(["fixture/linux-64".to_string()]),
                    universe: Arc::from("fixture-universe"),
                },
            )]),
        });

        let full = parse_match_specs(&["python 3.11.*".into(), "numpy >=2".into()]);
        let reordered = parse_match_specs(&["numpy >=2".into(), "python 3.11.*".into()]);
        let (full_records, full_consulted) = shared.records_for(&full).await;
        let (reordered_records, reordered_consulted) = shared.records_for(&reordered).await;

        assert!(Arc::ptr_eq(&records, &full_records));
        assert!(Arc::ptr_eq(&full_records, &reordered_records));
        assert_eq!(&*full_consulted, &*reordered_consulted);
    }

    #[tokio::test]
    async fn larger_question_cannot_replace_baseline_candidate_snapshot() {
        let shared = shared_sparse_fixture(ReachableState::default());
        let baseline = parse_match_specs(&["baseline >=1".to_string()]);
        let expanded = parse_match_specs(&["baseline >=1".to_string(), "route >=1".to_string()]);
        let baseline_records = vec![repo_record("baseline", "1.0", &[])];
        let expanded_records = vec![
            repo_record("baseline", "1.0", &[]),
            repo_record("route", "1.0", &[]),
        ];

        let before = shared
            .snapshot_for_with_loader(&baseline, move |_pairs, roots| async move {
                assert_eq!(roots, [PackageName::from_str("baseline").unwrap()]);
                Ok((baseline_records, vec!["fixture/linux-64".to_string()]))
            })
            .await;
        let grown = shared
            .snapshot_for_with_loader(&expanded, move |_pairs, roots| async move {
                assert_eq!(
                    roots,
                    [
                        PackageName::from_str("baseline").unwrap(),
                        PackageName::from_str("route").unwrap(),
                    ]
                );
                Ok((expanded_records, vec!["fixture/linux-64".to_string()]))
            })
            .await;
        let after = shared
            .snapshot_for_with_loader(&baseline, |_pairs, _roots| async move {
                panic!("the original question must use its immutable snapshot")
            })
            .await;

        assert_eq!(before.records.len(), 1);
        assert_eq!(grown.records.len(), 2);
        assert!(Arc::ptr_eq(&before.records, &after.records));
        assert_eq!(
            after.records.len(),
            1,
            "candidate growth must not change baseline"
        );
        assert_eq!(
            before.universe, after.universe,
            "an unchanged question keeps its universe fingerprint"
        );
        assert_ne!(
            before.universe, grown.universe,
            "a grown candidate universe must change the fingerprint"
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
    fn hermetic_toolchain_solve_declares_its_linux_subdir() {
        // The hermetic toolchain solve passes an empty system-requirements
        // map and no detected virtual packages, so without a declared subdir
        // it depends on host detection to supply `__linux`. Every conda-forge
        // record on linux-64 depends on `__linux`, and when it is absent
        // resolvo eliminates every candidate and reports the roots as
        // unsatisfiable rather than naming the missing platform package.
        let sysreqs = BTreeMap::new();
        let vps =
            build_virtual_packages_for_subdir_target("3.11", &sysreqs, None, Some("linux-64"))
                .expect("assembly must succeed");
        assert!(
            vp_lookup(&vps, "__linux").is_some(),
            "the toolchain solve must carry __linux: {vps:?}"
        );
        assert!(
            vp_lookup(&vps, "__unix").is_some(),
            "the toolchain solve must carry __unix: {vps:?}"
        );
    }

    #[test]
    fn contract_qualified_empty_detection_still_supplies_platform_packages() {
        // The genesis regression: a contract-qualified target whose detected
        // set is empty derived its whole virtual-package set from
        // [system-requirements], which never declares `linux`. Every record on
        // a linux subdir depends on `__linux`, so the solve failed for a
        // reason unrelated to the request and its constraints were dropped.
        let sysreqs = BTreeMap::from([
            ("cuda".to_string(), "12".to_string()),
            ("libc".to_string(), "2.35".to_string()),
        ]);
        let detected = BTreeMap::new();
        let without_subdir =
            build_virtual_packages_for_subdir_target("3.12", &sysreqs, Some(&detected), None)
                .expect("assembly must succeed");
        assert!(
            vp_lookup(&without_subdir, "__linux").is_none(),
            "no subdir means no platform contract to derive from"
        );

        let vps = build_virtual_packages_for_subdir_target(
            "3.12",
            &sysreqs,
            Some(&detected),
            Some("linux-64"),
        )
        .expect("assembly must succeed");
        assert!(
            vp_lookup(&vps, "__linux").is_some(),
            "a linux subdir must supply __linux: {vps:?}"
        );
        assert!(
            vp_lookup(&vps, "__unix").is_some(),
            "a linux subdir must supply __unix: {vps:?}"
        );
        // The declared contract is untouched.
        assert!(vp_lookup(&vps, "__cuda").is_some(), "{vps:?}");
        assert!(vp_lookup(&vps, "__glibc").is_some(), "{vps:?}");
    }

    #[test]
    fn declared_linux_floor_wins_over_the_default() {
        let sysreqs = BTreeMap::from([("linux".to_string(), "5.15".to_string())]);
        let detected = BTreeMap::new();
        let vps = build_virtual_packages_for_subdir_target(
            "3.12",
            &sysreqs,
            Some(&detected),
            Some("linux-64"),
        )
        .expect("assembly must succeed");
        assert_eq!(
            vp_lookup(&vps, "__linux")
                .expect("__linux must be present")
                .version
                .to_string(),
            "5.15"
        );
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
            &SolveTarget::new(
                "3.11",
                ChannelPriority::Strict,
                &BTreeMap::new(),
                None,
                SolveStrategy::Highest,
            ),
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
