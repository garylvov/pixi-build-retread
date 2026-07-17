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

use std::collections::{BTreeMap, HashSet};
use std::str::FromStr;

use rattler_conda_types::{
    ChannelUrl, GenericVirtualPackage, MatchSpec, PackageName, ParseStrictness, RepoDataRecord,
    Version,
};
use rattler_solve::{ChannelPriority, SolveStrategy, SolverImpl, SolverTask, resolvo};

use crate::relax::CondaMatchSpec;

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
fn solve_selected_records_from_records(
    parsed_specs: Vec<MatchSpec>,
    all_records: &[RepoDataRecord],
    target_python: &str,
    channel_priority: ChannelPriority,
    system_requirements: &BTreeMap<String, String>,
    strategy: SolveStrategy,
    preferred: Vec<RepoDataRecord>,
) -> std::result::Result<Vec<RepoDataRecord>, Vec<String>> {
    let virtual_packages = build_virtual_packages(target_python, system_requirements);
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
        Err(rattler_solve::SolveError::Unsolvable(reasons)) => Err(reasons),
        Err(other) => Err(vec![format!("solver error: {other}")]),
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
async fn solve_on_blocking_pool(
    parsed_specs: Vec<MatchSpec>,
    records: Vec<RepoDataRecord>,
    target_python: String,
    channel_priority: ChannelPriority,
    system_requirements: BTreeMap<String, String>,
    strategy: SolveStrategy,
    preferred: Vec<RepoDataRecord>,
) -> std::result::Result<Vec<RepoDataRecord>, Vec<String>> {
    let t_solve = std::time::Instant::now();
    let specs_count = parsed_specs.len();
    let records_count = records.len();
    let result = tokio::task::spawn_blocking(move || {
        solve_selected_records_from_records(
            parsed_specs,
            &records,
            &target_python,
            channel_priority,
            &system_requirements,
            strategy,
            preferred,
        )
    })
    .await
    .unwrap_or_else(|e| Err(vec![format!("solver task panicked: {e}")]));
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
/// v0.37.0+: build the rattler virtual-package set the solver
/// should see. Order of operations:
///   1. Host detection (`__archspec`, `__linux`, `__glibc`, `__cuda`,
///      `__osx`, ...) — provides defaults for keys the workspace
///      doesn't constrain explicitly.
///   2. `__cpython` override from the variant-derived `target_python`
///      so transitive `python_abi` constraints resolve consistently
///      with what pixi will install.
///   3. Workspace `[feature.X.system-requirements]` overrides — pixi
///      treats these as authoritative; retread must too. Keys map:
///      `cuda -> __cuda`, `libc -> __glibc` (linux; value is glibc
///      version), `macos -> __osx`, `archspec -> __archspec`
///      (build-string encoded), `linux -> __linux`. Unrecognized keys
///      are trace-logged + skipped (forward-compat).
///
/// Without (3), retread's solve_check sees the BUILD HOST's virtual
/// packages while pixi's actual solve sees the WORKSPACE-declared
/// ones — the asymmetry produced "retread sat, pixi unsat" for the
/// gsn gymnasium failure that motivated v0.37.0. Extracted into a
/// pure function so the mapping logic is unit-testable without
/// running a full solve.
pub fn build_virtual_packages(
    target_python: &str,
    system_requirements: &std::collections::BTreeMap<String, String>,
) -> Vec<GenericVirtualPackage> {
    let mut virtual_packages: Vec<GenericVirtualPackage> =
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
        };
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
    for (req_key, req_value) in system_requirements {
        let (vp_name, is_build_string) = match req_key.as_str() {
            "cuda" => ("__cuda", false),
            "libc" | "glibc" => ("__glibc", false),
            "macos" | "osx" => ("__osx", false),
            // Archspec is unconditionally a build-string virtual
            // package -- even values like "1" or "10" that LOOK
            // like versions are arch identifiers; treat as such.
            "archspec" => ("__archspec", true),
            "linux" => ("__linux", false),
            other => {
                tracing::trace!(
                    key = %other,
                    "solve-check: ignoring unrecognized system-requirement key (not in pixi schema)",
                );
                continue;
            }
        };
        let Ok(name) = PackageName::from_str(vp_name) else {
            continue;
        };
        let (version, build_string) = if is_build_string {
            let Ok(v1) = Version::from_str("1.0") else {
                continue;
            };
            (v1, req_value.clone())
        } else {
            match Version::from_str(req_value) {
                Ok(v) => (v, String::new()),
                Err(_) => {
                    let Ok(v1) = Version::from_str("1.0") else {
                        continue;
                    };
                    (v1, req_value.clone())
                }
            }
        };
        virtual_packages.retain(|vp| vp.name.as_normalized() != vp_name);
        virtual_packages.push(GenericVirtualPackage {
            name,
            version,
            build_string,
        });
    }
    virtual_packages
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
    use rattler_repodata_gateway::sparse::{PackageFormatSelection, SparseRepoData};
    let pairs = crate::repodata::sparse_pairs(channels, target_subdir).await;
    if pairs.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let consulted: Vec<String> = pairs.iter().map(|(label, _)| label.clone()).collect();
    // Exact-named specs seed the reachable-set walk. Glob/regex
    // matchers (rare; retread never emits them) can't seed a name walk
    // -- they're skipped here and simply unmatchable in the subset,
    // identical to how an absent package behaved in the full set.
    let root_names: Vec<PackageName> = parsed_specs
        .iter()
        .filter_map(|spec| match spec.name.as_ref() {
            Some(rattler_conda_types::PackageNameMatcher::Exact(name)) => Some(name.clone()),
            _ => None,
        })
        .collect();
    let roots = root_names.len();
    let t = std::time::Instant::now();
    let handles: Vec<_> = pairs.into_iter().map(|(_, h)| h).collect();
    let per_repo = match tokio::task::spawn_blocking(move || {
        SparseRepoData::load_records_recursive(
            handles.iter().map(|h| h.as_ref()),
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
            return (Vec::new(), consulted);
        }
        Err(e) => {
            tracing::warn!(error = %e, "solve-check: sparse record load panicked");
            return (Vec::new(), consulted);
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
    (records, consulted)
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
    solve_on_blocking_pool(
        parsed_specs,
        records,
        target_python.to_string(),
        channel_priority,
        system_requirements.clone(),
        strategy,
        Vec::new(),
    )
    .await
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
