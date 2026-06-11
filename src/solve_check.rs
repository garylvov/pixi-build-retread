//! v0.33.0+: pre-emission solve check.
//!
//! After retread builds a per-env CondaOutput, run a real conda solve
//! over (workspace effective deps + retread's emitted run-deps) against
//! the workspace's channels. If unsat, capture the explanation strings
//! and persist them to the audit. This catches cross-package conflicts
//! that the per-dep probe layer can't see:
//!
//!   - retread emits `cuda-bindings >=13.0.3,<14` (from a wheel's
//!     Requires-Dist)
//!   - workspace pins `cuda-toolkit 12.8.*`
//!   - cuda-bindings 13.x's `depends` says it needs `cuda 13.x`
//!   - cuda-toolkit 12.8.*'s `depends` says cuda 12.8.*
//!   - same dep (`cuda`) with incompatible ranges across two
//!     workspace-/retread-pinned packages -> solver explanation
//!
//! The per-dep probe never catches this because it only asks "does the
//! spec I'd emit have ANY candidate on the channel" -- not "does that
//! candidate compose with the rest of the workspace."
//!
//! Cost: the iterative refinement loop calls this ~8x per env across
//! ~4 envs (~32 solves). v0.43.0: the ~1M repodata records are now
//! parsed from the disk cache ONCE per (subdir, channel-set) and shared
//! across every solve via `RECORDS_CACHE` (an `Arc<RecordSet>`). Before
//! this, `run_solve_check` re-read + re-parsed + re-deduped ~1.08M
//! RepoDataRecords on EVERY call -- ~32 redundant parses of hundreds of
//! MB of JSON, which was the dominant CPU cost and the ~4 GB RSS thrash
//! (NOT the resolvo solve itself). The older "repodata is cached by
//! probe.rs" claim here was false: probe.rs caches a different parsed
//! form (`RepodataIndex`) that the solver can't consume.

use std::collections::{BTreeMap, HashSet};
use std::str::FromStr;

use rattler_conda_types::{
    ChannelUrl, GenericVirtualPackage, MatchSpec, PackageName, ParseStrictness, RepoDataRecord,
    Version,
};
use rattler_solve::{ChannelPriority, SolveStrategy, SolverImpl, SolverTask, resolvo};
use serde::Serialize;

/// Result of running the solve check.
#[derive(Debug, Clone, Serialize)]
pub struct SolveOutcome {
    /// True if the solver found a valid solution.
    pub satisfiable: bool,
    /// Per-leaf explanation strings from the rattler solver when
    /// `satisfiable == false`. Each string names a constraint chain
    /// the solver couldn't reconcile.
    pub unsat_explanations: Vec<String>,
    /// Channels actually consulted (after disk-cache reads).
    pub channels_consulted: Vec<String>,
    /// Total specs the solver was asked to satisfy.
    pub specs_count: usize,
    /// Total available records across all channels.
    pub records_count: usize,
    /// v0.34.0+: refinement passes the cascade made before reaching
    /// this outcome (each pass widened one or more retread-emitted
    /// deps to `*`). Empty when the first solve succeeded.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refinement_steps: Vec<crate::audit::RefinementStep>,
    /// v0.35.0+: workspace-edit suggestions from the classifier.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub workspace_edit_suggestions: Vec<crate::conflict_classifier::WorkspaceEditSuggestion>,
    /// v0.35.0+: terminal classification name (A/AExhausted/B/C/None).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_classification: Option<String>,
    /// v1.4.0: true when the check COULD NOT RUN (no channels given,
    /// or no repodata loadable from any of them). A skipped check is
    /// an abstention, not an unsat verdict: callers must not refine
    /// against it, print failure banners for it, or fail conda/outputs
    /// because of it. Before this flag, an empty/unreachable channel
    /// set surfaced as `satisfiable: false`, the refinement loop
    /// classified the "no repodata" text as cascade-exhausted, and
    /// the fail gate hard-errored the build -- a diagnostic that
    /// never ran was vetoing the output.
    #[serde(default)]
    pub skipped: bool,
}

impl SolveOutcome {
    pub fn unreachable() -> Self {
        Self {
            satisfiable: false,
            unsat_explanations: vec!["solve-check skipped: no repodata could be loaded".into()],
            channels_consulted: Vec::new(),
            specs_count: 0,
            records_count: 0,
            refinement_steps: Vec::new(),
            workspace_edit_suggestions: Vec::new(),
            terminal_classification: None,
            skipped: true,
        }
    }
}

/// Solve a spec set against already-loaded records and return the
/// concrete records the solver selected. Shared by the pre-emission
/// solve check and workspace transitive extraction so both reason
/// about the SAME coherent solution instead of each inventing its own
/// approximation.
fn solve_selected_records_from_records(
    parsed_specs: Vec<MatchSpec>,
    all_records: &[RepoDataRecord],
    target_python: &str,
    channel_priority: ChannelPriority,
    system_requirements: &BTreeMap<String, String>,
    strategy: SolveStrategy,
) -> std::result::Result<Vec<RepoDataRecord>, Vec<String>> {
    let virtual_packages = build_virtual_packages(target_python, system_requirements);
    let task = SolverTask {
        available_packages: vec![all_records],
        locked_packages: Vec::new(),
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
async fn solve_on_blocking_pool(
    parsed_specs: Vec<MatchSpec>,
    records: Vec<RepoDataRecord>,
    target_python: String,
    channel_priority: ChannelPriority,
    system_requirements: BTreeMap<String, String>,
    strategy: SolveStrategy,
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

pub async fn run_solve_check(
    channels: &[ChannelUrl],
    specs: &[String],
    target_python: &str,
    target_subdir: &str,
    channel_priority: ChannelPriority,
    system_requirements: &std::collections::BTreeMap<String, String>,
) -> SolveOutcome {
    // Parse the specs first; bad input shouldn't be hidden behind
    // network IO. Skip specs that don't parse (rare; logged at debug).
    let parsed_specs = parse_match_specs(specs);

    let (records, consulted) =
        load_selected_records_sparse(channels, target_subdir, &parsed_specs).await;

    if records.is_empty() {
        return SolveOutcome {
            satisfiable: false,
            unsat_explanations: vec![
                "solve-check skipped: no repodata available from disk cache".into(),
            ],
            channels_consulted: consulted,
            specs_count: parsed_specs.len(),
            records_count: 0,
            refinement_steps: Vec::new(),
            workspace_edit_suggestions: Vec::new(),
            terminal_classification: None,
            skipped: true,
        };
    }

    let records_count = records.len();
    let specs_count = parsed_specs.len();
    match solve_on_blocking_pool(
        parsed_specs,
        records,
        target_python.to_string(),
        channel_priority,
        system_requirements.clone(),
        SolveStrategy::Highest,
    )
    .await
    {
        Ok(_records) => SolveOutcome {
            satisfiable: true,
            unsat_explanations: Vec::new(),
            channels_consulted: consulted,
            specs_count,
            records_count,
            refinement_steps: Vec::new(),
            workspace_edit_suggestions: Vec::new(),
            terminal_classification: None,
            skipped: false,
        },
        Err(reasons) => SolveOutcome {
            satisfiable: false,
            unsat_explanations: reasons,
            channels_consulted: consulted,
            specs_count,
            records_count,
            refinement_steps: Vec::new(),
            workspace_edit_suggestions: Vec::new(),
            terminal_classification: None,
            skipped: false,
        },
    }
}

/// Solve a spec set against cached repodata and return the concrete
/// records the solver selected. Callers that need to reason about
/// transitives must use this instead of "latest matching build"
/// guesses, otherwise open-ended pins like `torchvision >=0.22` and
/// `pytorch-gpu >=2.7` get their transitives sourced from DIFFERENT
/// package generations and create impossible merged constraints.
pub async fn solve_selected_records(
    channels: &[ChannelUrl],
    specs: &[String],
    target_python: &str,
    target_subdir: &str,
    channel_priority: ChannelPriority,
    system_requirements: &BTreeMap<String, String>,
    strategy: SolveStrategy,
) -> std::result::Result<Vec<RepoDataRecord>, Vec<String>> {
    let parsed_specs = parse_match_specs(specs);
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
    )
    .await
}

/// Compute the disk-cache path for repodata. Must match probe.rs's
/// layout exactly -- we read what that module wrote.
/// v0.34.0+: parse rattler_solve's tree-formatted unsat explanation
/// strings to find the package names that are the ENTRY POINTS of the
/// conflict graph. These are the deps the solver couldn't satisfy at
/// the top level -- widening any of them in retread's emission gives
/// the solver more freedom to backtrack.
///
/// Returns `["triton"]` for the typical triton-cuda chain. Multiple
/// sibling top-level conflicts return multiple names.
pub fn extract_blocking_dep_names(unsat_strs: &[String]) -> Vec<String> {
    extract_blocking_chains(unsat_strs)
        .into_iter()
        .map(|c| c.name)
        .collect()
}

/// v0.35.0+: richer version of `extract_blocking_dep_names`. Preserves
/// the version data we used to throw away after extracting the name:
/// the spec the blocked package was looking for, the versions the
/// solver tried + rejected, and the transitive requirement that caused
/// each rejection.
///
/// The classifier (`conflict_classifier::classify_unsat`) consumes
/// these chains to produce a structured `ConflictClassification` --
/// distinguishing "retread-side widenable" from "workspace-pin-
/// dominated" failures.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct BlockingChain {
    /// Top-level package name (e.g. `"torchvision"`).
    pub name: String,
    /// Spec that was looked up at the top level (e.g. `">=0.22.0"`).
    /// Empty when the parser couldn't extract one.
    pub current_spec: String,
    /// Concrete versions the solver tried and rejected
    /// (e.g. `["0.25.0", "0.26.0"]`). Deduped, sorted in encountered
    /// order. May be empty when the rattler explanation didn't list
    /// individual versions.
    pub rejected_versions: Vec<String>,
    /// Transitive requirement that caused the rejection
    /// (e.g. `"pytorch >=2.10.0,<2.11.0a0"`). Empty when the chain
    /// doesn't dive deeper or the parse missed it.
    pub transitive_requirement: String,
    /// v0.36.1+: true when rattler said "can be installed with any
    /// of the following options" (the dep is satisfiable in
    /// isolation but listed as part of a multi-dep incompatibility
    /// group). false when "cannot be installed" (genuinely
    /// blocking). The classifier uses this to skip suggestion
    /// derivation for installable chains -- they're context, not
    /// the root cause.
    #[serde(default)]
    pub installable: bool,
}

pub fn extract_blocking_chains(unsat_strs: &[String]) -> Vec<BlockingChain> {
    let mut out: Vec<BlockingChain> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for raw in unsat_strs {
        // Walk lines; whenever a top-level marker is found, parse the
        // chain rooted at that line + its descendants (denoted by
        // deeper indentation).
        let lines: Vec<&str> = raw.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            let trimmed_start = line
                .strip_prefix("├─ ")
                .or_else(|| line.strip_prefix("└─ "));
            let Some(rest) = trimmed_start else { continue };
            let token = rest.split_whitespace().next().unwrap_or("");
            if token.is_empty() {
                continue;
            }
            let name = token.split('[').next().unwrap_or(token);
            if name
                .chars()
                .any(|c| matches!(c, '<' | '>' | '=' | ',' | '!' | '~' | '*'))
            {
                continue;
            }
            if !seen.insert(name.to_string()) {
                continue;
            }
            // Extract the spec: everything after the name up to
            // " cannot be installed" / " can be installed" / EOL.
            // Also track whether this chain is genuinely blocking
            // ("cannot be installed") or just listed as context
            // ("can be installed with any of the following options").
            let after_name = rest[token.len()..].trim_start();
            let (current_spec, installable) =
                if let Some(end) = after_name.find(" cannot be installed") {
                    (after_name[..end].trim().to_string(), false)
                } else if let Some(end) = after_name.find(" can be installed") {
                    (after_name[..end].trim().to_string(), true)
                } else {
                    (String::new(), false)
                };

            // Walk descendant lines (deeper indentation than the
            // current line) to find a `would require` block and the
            // versions it enumerates.
            let mut rejected_versions: Vec<String> = Vec::new();
            let mut rejected_seen: HashSet<String> = HashSet::new();
            let mut transitive_requirement: String = String::new();
            for tail in lines.iter().skip(idx + 1) {
                if !tail.starts_with('│')
                    && !tail.starts_with(' ')
                    && !tail.starts_with('├')
                    && !tail.starts_with('└')
                    && !tail.starts_with('\t')
                {
                    // Back to top-level, this chain is over.
                    if !tail.is_empty() {
                        break;
                    }
                }
                // The "X 1.0 | 1.0 | 1.0 ... would require" line lists
                // candidate versions. Extract the part after the name
                // and before "would require".
                if let Some(would_idx) = tail.find(" would require") {
                    let before = &tail[..would_idx];
                    // Strip leading tree decoration: any chars in
                    // `│├└─ \t`.
                    let payload = before.trim_start_matches(|c: char| {
                        matches!(c, '│' | '├' | '└' | '─' | ' ' | '\t')
                    });
                    // payload looks like `<name> 0.25.0 | 0.25.0 | ...`
                    // Drop the name (first whitespace-separated token).
                    let mut it = payload.splitn(2, char::is_whitespace);
                    let _ = it.next();
                    if let Some(versions_part) = it.next() {
                        for v in versions_part.split('|') {
                            let v = v.trim();
                            if v.is_empty() {
                                continue;
                            }
                            if rejected_seen.insert(v.to_string()) {
                                rejected_versions.push(v.to_string());
                            }
                        }
                    }
                    continue;
                }
                // Lines like `└─ pytorch >=2.10.0,<2.11.0a0, which
                // cannot be installed` are the transitive requirement
                // chain. Capture the FIRST such line per top-level
                // chain (the closest cause).
                if transitive_requirement.is_empty() {
                    let strip = tail.trim_start_matches(|c: char| {
                        matches!(c, '│' | '├' | '└' | '─' | ' ' | '\t')
                    });
                    // Strip trailing ", which cannot be installed..." /
                    // ", for which no candidates...".
                    let end = strip
                        .find(", which")
                        .or_else(|| strip.find(", for which"))
                        .unwrap_or(strip.len());
                    let candidate = strip[..end].trim().to_string();
                    // Only keep if it looks like `<name> <spec>` not
                    // a versions-list line. Version-enumeration
                    // lines like `pytorch-gpu 2.7.1 | 2.7.1` contain
                    // `|` (the rattler separator); those are
                    // candidates, not transitive requirements.
                    let looks_like_version_list = candidate.contains(" | ");
                    if !candidate.contains(" would require")
                        && !looks_like_version_list
                        && !candidate.is_empty()
                        && candidate
                            .chars()
                            .next()
                            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                    {
                        transitive_requirement = candidate;
                    }
                }
            }

            out.push(BlockingChain {
                name: name.to_string(),
                current_spec,
                rejected_versions,
                transitive_requirement,
                installable,
            });
        }
    }
    out
}

#[cfg(test)]
mod parse_tests {
    use super::*;

    #[test]
    fn extract_blocking_dep_names_pulls_top_level_only() {
        let unsat = vec![
            "The following packages are incompatible\n\
             ├─ triton >=3.7.0,<3.8 cannot be installed because there are no viable options:\n\
             │  ├─ triton 3.7.0 | 3.7.0 would require\n\
             │  │  └─ cuda-version >=13.0,<14, which cannot be installed because there are no viable options:\n\
             │  │     ├─ cuda-version 13.2, which conflicts with the versions reported above.\n\
             └─ another-pkg >=1.0 cannot be installed because there are no viable options:\n\
             │  └─ stuff"
                .to_string(),
        ];
        let blocking = extract_blocking_dep_names(&unsat);
        // Both `triton` and `another-pkg` are top-level conflict
        // entry points. Nested deps (cuda-version) are NOT.
        assert!(
            blocking.contains(&"triton".to_string()),
            "expected triton in {blocking:?}"
        );
        assert!(
            blocking.contains(&"another-pkg".to_string()),
            "expected another-pkg in {blocking:?}"
        );
        assert!(
            !blocking.contains(&"cuda-version".to_string()),
            "cuda-version is nested, not top-level: {blocking:?}"
        );
    }

    #[test]
    fn extract_blocking_dep_names_dedups() {
        let unsat = vec!["├─ pkg cannot be installed\n├─ pkg also cannot be installed".to_string()];
        let blocking = extract_blocking_dep_names(&unsat);
        assert_eq!(blocking, vec!["pkg".to_string()]);
    }

    #[test]
    fn extract_blocking_chains_captures_versions_and_transitive() {
        let unsat = vec![
            "The following packages are incompatible\n\
             └─ torchvision >=0.22.0 cannot be installed because there are no viable options:\n   \
                ├─ torchvision 0.25.0 | 0.25.0 | 0.26.0 would require\n   \
                │  └─ pytorch >=2.10.0,<2.11.0a0, which cannot be installed because there are no viable options:\n   \
                │     └─ pytorch 2.10.0 would require\n   \
                │        └─ cuda-version >=12.9,<13, for which no candidates were found."
                .to_string(),
        ];
        let chains = extract_blocking_chains(&unsat);
        assert_eq!(chains.len(), 1);
        let c = &chains[0];
        assert_eq!(c.name, "torchvision");
        assert_eq!(c.current_spec, ">=0.22.0");
        // Versions deduped, in encountered order.
        assert!(
            c.rejected_versions.contains(&"0.25.0".to_string())
                && c.rejected_versions.contains(&"0.26.0".to_string()),
            "rejected_versions = {:?}",
            c.rejected_versions,
        );
        // Transitive picks the closest cause (pytorch), not the
        // deepest (cuda-version).
        assert!(
            c.transitive_requirement.starts_with("pytorch "),
            "transitive_requirement = {:?}",
            c.transitive_requirement,
        );
    }

    #[test]
    fn extract_blocking_chains_handles_multiple_top_levels() {
        let unsat = vec![
            "The following packages are incompatible\n\
             ├─ pkga >=1 cannot be installed because there are no viable options:\n   \
                └─ stuff\n\
             └─ pkgb cannot be installed because there are no viable options:\n   \
                └─ more stuff"
                .to_string(),
        ];
        let chains = extract_blocking_chains(&unsat);
        let names: Vec<&str> = chains.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"pkga"), "names = {names:?}");
        assert!(names.contains(&"pkgb"), "names = {names:?}");
    }

    #[test]
    fn extract_blocking_chains_empty_fields_when_unparseable() {
        // No "would require" or transitive; chain still emits with
        // empty rejected_versions / transitive_requirement.
        let unsat = vec!["├─ foo cannot be installed because reasons".to_string()];
        let chains = extract_blocking_chains(&unsat);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].name, "foo");
        assert!(chains[0].rejected_versions.is_empty());
        assert!(chains[0].transitive_requirement.is_empty());
    }

    #[test]
    fn extract_blocking_dep_names_skips_specifier_tokens() {
        // Lines whose first token after the marker is a version
        // operator (rare; defensive) shouldn't accidentally count.
        let unsat = vec!["├─ >=1.0 noisy".to_string()];
        assert!(extract_blocking_dep_names(&unsat).is_empty());
    }
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

    #[test]
    fn solve_outcome_unreachable_default() {
        let o = SolveOutcome::unreachable();
        assert!(!o.satisfiable);
        assert_eq!(o.unsat_explanations.len(), 1);
    }

    // -------------------------------------------------------------
    // v0.37.0 T1: build_virtual_packages exposes the workspace -> rattler
    // virtual-package mapping for unit testing. The full end-to-end
    // run_solve_check parity test against synthetic RepoData is left
    // as future work (it would require committing repodata fixtures
    // under tests/fixtures/); these unit tests pin the contract that
    // the mapping is correct, which is the load-bearing piece. If
    // these pass and the cascade is wired right, retread's solver
    // sees the same virtual packages pixi's will.
    // -------------------------------------------------------------

    fn vp_lookup<'a>(
        vps: &'a [GenericVirtualPackage],
        name: &str,
    ) -> Option<&'a GenericVirtualPackage> {
        vps.iter().find(|vp| vp.name.as_normalized() == name)
    }

    #[test]
    fn run_solve_check_with_no_channels_abstains() {
        // v1.4.0 regression (caught by tests/jsonrpc_protocol.rs):
        // with no channels (or none loadable), the check must ABSTAIN
        // (skipped=true, no terminal classification), not report a
        // pseudo-unsat that the refinement loop classifies as
        // cascade-exhausted and the fail gate turns into a hard
        // conda/outputs error. A diagnostic that never ran must not
        // veto the build.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let outcome = rt.block_on(run_solve_check(
            &[],
            &["python ==3.11".to_string()],
            "3.11",
            "linux-64",
            ChannelPriority::Strict,
            &std::collections::BTreeMap::new(),
        ));
        assert!(outcome.skipped, "no channels -> the check must abstain");
        assert!(!outcome.satisfiable, "skipped is not a sat verdict either");
        assert!(outcome.terminal_classification.is_none());
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
}
