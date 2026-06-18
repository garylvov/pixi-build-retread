// PR-2: in-process resolvo-vs-BFS A/B oracle.
//
// WHAT THIS MODULE DOES
// =====================
// When `RETREAD_RESOLVO_DIFF=<path>` is set, `resolve_bundle` calls
// `ab_diff_hook` right before returning the authoritative BFS bundle. The
// hook independently runs the resolvo pipeline on the same primary/entry,
// diffs the two results, classifies the difference, and appends one JSON line
// to a per-pid JSONL file (`<path>.<pid>.jsonl`).
//
// BFS is never touched — it remains authoritative. The resolvo result is
// measured and dropped. Default-unset = hook never entered.
//
// PHASE SCOPE
// ===========
// The phase is "pre-auto-bundle". The diff covers the resolve_bundle result
// BEFORE auto_bundle_transitives runs (which happens in the caller after
// resolve_bundle returns). Both the BFS and resolvo bundles reflect this
// same pre-auto-bundle scope.
//
// VERDICTS (precedence, first match wins)
// ========================================
// SKIPPED       – source-form entry (no extras); resolvo never runs.
// RED           – any of: Unsolvable AND !demote_to_excluded;
//                 provenance field non-empty on shared name+version;
//                 routing_mismatch non-empty;
//                 only_in_bfs or only_in_resolvo non-empty;
//                 bundled_set_matches=false.
// UNSOLVABLE-EXCLUDED – Unsolvable AND demote_to_excluded (narrow safe case).
// VERSION-DIFF  – Solved, version_changed non-empty, bundled_set_matches,
//                 provenance clean.
// GREEN         – Solved, only shared_same_version, bundled_set_matches,
//                 provenance clean.
//
// ERROR-SWALLOWING
// ================
// ab_diff_hook and ab_skip_hook NEVER propagate errors. On internal failure
// they write a RED record with Unsolvable{message:"ab-hook error: .."} or
// emit a tracing::warn!. The resolve_bundle caller is NEVER interrupted.

use std::io::Write as _;

use anyhow::Result;
use serde::Serialize;

use crate::config::RelaxPolicy;
use crate::handler::resolvo_provider::{
    SolveOutcome, pool_record_to_resolved_wheel, resolvo_solve_pool,
};
use crate::handler::{
    Bundle, PypiToCondaMap, ResolvedWheel, WheelEntry, WheelTarget, canonical_conda_name,
};
use crate::lock::SdistWheelSource;
use rattler_conda_types::ChannelUrl;

pub(crate) const AB_SCHEMA_VERSION: u32 = 1;

// ── Serialisable report types ─────────────────────────────────────────────────

/// One per entry per resolve_bundle call, appended to the per-pid JSONL file.
#[derive(Debug, Serialize)]
pub(crate) struct AbReport {
    pub schema_version: u32,
    pub entry_name: String,
    pub target: String,
    /// "pre-auto-bundle" or "skipped"
    pub phase: String,
    pub conda_name: String,
    pub verdict: String,
    pub resolvo_outcome: ResolvoOutcome,
    pub version_set: Option<VersionSetDiff>,
    pub conda_routing: Option<CondaRoutingDiff>,
    pub provenance: Option<Vec<ProvenanceDiff>>,
    pub skip_reason: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub(crate) enum ResolvoOutcome {
    Solved,
    Unsolvable { message: String },
}

#[derive(Debug, Serialize)]
pub(crate) struct VersionSetDiff {
    pub shared_same_version: Vec<String>,
    pub version_changed: Vec<VersionChange>,
    pub only_in_bfs: Vec<String>,
    pub only_in_resolvo: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct VersionChange {
    pub name: String,
    pub bfs: String,
    pub resolvo: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct CondaRoutingDiff {
    pub bfs_conda_routed: Vec<String>,
    pub resolvo_conda_routed: Vec<String>,
    pub bundled_set_matches: bool,
    pub routing_mismatch: Vec<RoutingMismatch>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RoutingMismatch {
    pub name: String,
    pub bfs: Side,
    pub resolvo: Side,
}

#[derive(Debug, Serialize, PartialEq, Eq, Clone, Copy)]
pub(crate) enum Side {
    Bundled,
    Conda,
    Absent,
}

/// Tier-1 provenance diff for wheels present in both solves at the same
/// version. Tier-2 fields (sha256/index/has_metadata_sidecar) are resolvo-only
/// (informational, not included in verdict classification).
#[derive(Debug, Serialize)]
pub(crate) struct ProvenanceDiff {
    pub name: String,
    pub version: String,
    pub fields: Vec<ProvenanceFieldDiff>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ProvenanceFieldDiff {
    pub field: String,
    pub bfs: Option<String>,
    pub resolvo: Option<String>,
}

// ── Bundled-set helpers ───────────────────────────────────────────────────────

/// Extract the set of canonical pypi names bundled in a Bundle (primary + extras).
pub(crate) fn derive_bundled_set(bundle: &Bundle) -> std::collections::HashSet<String> {
    std::iter::once(&bundle.primary)
        .chain(bundle.extras.iter())
        .map(|w| w.pypi_name.clone())
        .collect()
}

// ── Diff helpers ──────────────────────────────────────────────────────────────

/// Build the `VersionSetDiff` by comparing BFS and resolvo bundled wheels.
fn build_version_set_diff(bfs_bundle: &Bundle, resolvo_wheels: &[ResolvedWheel]) -> VersionSetDiff {
    use std::collections::HashMap;

    // BFS: name -> version (primary + extras)
    let bfs_map: HashMap<String, String> = std::iter::once(&bfs_bundle.primary)
        .chain(bfs_bundle.extras.iter())
        .map(|w| (w.pypi_name.clone(), w.metadata.version.clone()))
        .collect();

    // Resolvo: name -> version
    let resolvo_map: HashMap<String, String> = resolvo_wheels
        .iter()
        .map(|w| (w.pypi_name.clone(), w.metadata.version.clone()))
        .collect();

    let mut shared_same_version: Vec<String> = Vec::new();
    let mut version_changed: Vec<VersionChange> = Vec::new();
    let mut only_in_bfs: Vec<String> = Vec::new();
    let mut only_in_resolvo: Vec<String> = Vec::new();

    // Names in BFS
    let mut all_names: std::collections::HashSet<String> = bfs_map.keys().cloned().collect();
    all_names.extend(resolvo_map.keys().cloned());

    let mut all_names_sorted: Vec<String> = all_names.into_iter().collect();
    all_names_sorted.sort();

    for name in all_names_sorted {
        match (bfs_map.get(&name), resolvo_map.get(&name)) {
            (Some(bv), Some(rv)) if bv == rv => shared_same_version.push(name),
            (Some(bv), Some(rv)) => version_changed.push(VersionChange {
                name,
                bfs: bv.clone(),
                resolvo: rv.clone(),
            }),
            (Some(_), None) => only_in_bfs.push(name),
            (None, Some(_)) => only_in_resolvo.push(name),
            (None, None) => unreachable!(),
        }
    }

    VersionSetDiff {
        shared_same_version,
        version_changed,
        only_in_bfs,
        only_in_resolvo,
    }
}

/// Build the `CondaRoutingDiff` from the two resolvers' conda-routed sets
/// and the bundled sets.
///
/// All comparisons are done in canonical-conda-name space so raw PyPI name
/// vs canonical conda name skew (Pillow/pillow, nvidia_x/nvidia-x,
/// tinyobjloader/tinyobjloader-python) does not produce spurious RED.
fn build_conda_routing_diff(
    bfs_bundle: &Bundle,
    bfs_bundled_set: &std::collections::HashSet<String>,
    resolvo_conda_routed: &[String],
    resolvo_bundled_set: &std::collections::HashSet<String>,
) -> CondaRoutingDiff {
    // Canonicalize all sets to canonical_conda_name for apples-to-apples comparison.
    let canon_bfs_bundled: std::collections::HashSet<String> = bfs_bundled_set
        .iter()
        .map(|n| canonical_conda_name(n))
        .collect();
    let canon_resolvo_bundled: std::collections::HashSet<String> = resolvo_bundled_set
        .iter()
        .map(|n| canonical_conda_name(n))
        .collect();

    let bfs_conda_routed: Vec<String> = {
        let mut v: Vec<String> = bfs_bundle
            .conda_routed
            .iter()
            .map(|n| canonical_conda_name(n))
            .collect();
        v.sort();
        v.dedup();
        v
    };

    let resolvo_conda_routed_sorted: Vec<String> = {
        let mut v: Vec<String> = resolvo_conda_routed
            .iter()
            .map(|n| canonical_conda_name(n))
            .collect();
        v.sort();
        v.dedup();
        v
    };

    // bundled_set_matches: canonical bundled sets agree.
    let bundled_set_matches = canon_bfs_bundled == canon_resolvo_bundled;

    // Routing mismatch: names that land on different sides across the two solvers.
    let mut all_names: std::collections::HashSet<String> = canon_bfs_bundled.clone();
    all_names.extend(canon_resolvo_bundled.iter().cloned());
    all_names.extend(bfs_conda_routed.iter().cloned());
    all_names.extend(resolvo_conda_routed_sorted.iter().cloned());

    let bfs_conda_set: std::collections::HashSet<&String> = bfs_conda_routed.iter().collect();
    let resolvo_conda_set: std::collections::HashSet<&String> =
        resolvo_conda_routed_sorted.iter().collect();

    let mut routing_mismatch: Vec<RoutingMismatch> = Vec::new();
    let mut names_sorted: Vec<String> = all_names.into_iter().collect();
    names_sorted.sort();

    for name in &names_sorted {
        let bfs_side = if canon_bfs_bundled.contains(name) {
            Side::Bundled
        } else if bfs_conda_set.contains(name) {
            Side::Conda
        } else {
            Side::Absent
        };
        let resolvo_side = if canon_resolvo_bundled.contains(name) {
            Side::Bundled
        } else if resolvo_conda_set.contains(name) {
            Side::Conda
        } else {
            Side::Absent
        };
        if bfs_side != resolvo_side {
            routing_mismatch.push(RoutingMismatch {
                name: name.clone(),
                bfs: bfs_side,
                resolvo: resolvo_side,
            });
        }
    }

    CondaRoutingDiff {
        bfs_conda_routed,
        resolvo_conda_routed: resolvo_conda_routed_sorted,
        bundled_set_matches,
        routing_mismatch,
    }
}

/// Tier-1 provenance diff for wheels present in both solves at the same version.
fn build_provenance_diffs(
    bfs_bundle: &Bundle,
    resolvo_wheels: &[ResolvedWheel],
) -> Vec<ProvenanceDiff> {
    use std::collections::HashMap;

    let bfs_map: HashMap<(&str, &str), &ResolvedWheel> = std::iter::once(&bfs_bundle.primary)
        .chain(bfs_bundle.extras.iter())
        .map(|w| ((w.pypi_name.as_str(), w.metadata.version.as_str()), w))
        .collect();

    let mut diffs: Vec<ProvenanceDiff> = Vec::new();

    for rw in resolvo_wheels {
        let key = (rw.pypi_name.as_str(), rw.metadata.version.as_str());
        let Some(bfs_wheel) = bfs_map.get(&key) else {
            continue;
        };

        // Tier-1: filename (from wheel metadata), upstream_url, sdist_source, git_source.
        let mut fields: Vec<ProvenanceFieldDiff> = Vec::new();

        // filename — catches tag differences (cp311 vs cp312, abi3 vs cp-specific)
        // on wheels at the same (name, version).
        let bfs_filename = Some(bfs_wheel.metadata.filename.clone());
        let resolvo_filename = Some(rw.metadata.filename.clone());
        if bfs_filename != resolvo_filename {
            fields.push(ProvenanceFieldDiff {
                field: "filename".to_string(),
                bfs: bfs_filename,
                resolvo: resolvo_filename,
            });
        }

        // upstream_url
        let bfs_upstream = bfs_wheel.upstream_url.as_ref().map(|u| u.to_string());
        let resolvo_upstream = rw.upstream_url.as_ref().map(|u| u.to_string());
        if bfs_upstream != resolvo_upstream {
            fields.push(ProvenanceFieldDiff {
                field: "upstream_url".to_string(),
                bfs: bfs_upstream,
                resolvo: resolvo_upstream,
            });
        }

        // sdist_source (compare presence/absence and sdist url)
        let bfs_sdist = bfs_wheel
            .sdist_source
            .as_ref()
            .map(|s: &SdistWheelSource| s.sdist_url.to_string());
        let resolvo_sdist = rw
            .sdist_source
            .as_ref()
            .map(|s: &SdistWheelSource| s.sdist_url.to_string());
        if bfs_sdist != resolvo_sdist {
            fields.push(ProvenanceFieldDiff {
                field: "sdist_source".to_string(),
                bfs: bfs_sdist,
                resolvo: resolvo_sdist,
            });
        }

        // git_source
        let bfs_git = bfs_wheel
            .git_source
            .as_ref()
            .map(|g| format!("{}@{}", g.url, g.rev));
        let resolvo_git = rw
            .git_source
            .as_ref()
            .map(|g| format!("{}@{}", g.url, g.rev));
        if bfs_git != resolvo_git {
            fields.push(ProvenanceFieldDiff {
                field: "git_source".to_string(),
                bfs: bfs_git,
                resolvo: resolvo_git,
            });
        }

        if !fields.is_empty() {
            diffs.push(ProvenanceDiff {
                name: rw.pypi_name.clone(),
                version: rw.metadata.version.clone(),
                fields,
            });
        }
    }

    diffs
}

// ── Demote-to-excluded heuristic ─────────────────────────────────────────────

/// Returns `true` ONLY in the narrow safe case: the Unsolvable message text
/// references at least one excluded (name, version) pair AND no other
/// identifier-shaped token appears outside the excluded set.
/// Ambiguous cases return `false` (fail-to-RED).
///
/// This is best-effort-strengthen-only: resolvo's `Conflict` type has no
/// structured name API, so we work from the display string.
///
/// Every non-digit-leading alphabetic/identifier token is treated as a
/// potential name candidate. English connectives (is/because/cannot/…) in
/// resolvo's display_user_friendly output will each force RED — that is the
/// INTENDED fail-to-RED-harder posture: over-report RED, never under-report.
/// Single-word PyPI names (numpy, torch, scipy, gymnasium, pillow, …) that
/// appear in a conflict message alongside an excluded hyphenated package MUST
/// also force RED; the old hyphen/underscore short-circuit silently ignored
/// them, allowing a genuine numpy conflict to be demoted to UNSOLVABLE-EXCLUDED.
pub(crate) fn demote_to_excluded(
    msg: &str,
    excluded: &[crate::handler::resolvo_discovery::ExcludedCandidate],
) -> bool {
    if excluded.is_empty() {
        return false;
    }

    // Check that at least one excluded (name, version) is referenced in the message.
    let any_excluded_referenced = excluded
        .iter()
        .any(|e| !e.version.is_empty() && msg.contains(&e.pypi_name) && msg.contains(&e.version));

    if !any_excluded_referenced {
        return false;
    }

    // Fail-to-RED: if ANY identifier-shaped token appears that is not in the
    // excluded set, return false (ambiguous => RED).
    // Digit-starting tokens are skipped (version strings: "1.26.4", "3.11").
    let excluded_names: std::collections::HashSet<&str> =
        excluded.iter().map(|e| e.pypi_name.as_str()).collect();

    for word in msg.split_whitespace() {
        let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_');
        if cleaned.is_empty() {
            continue;
        }
        // Skip version-like tokens (start with a digit).
        if cleaned.starts_with(|c: char| c.is_ascii_digit()) {
            continue;
        }
        // Every other identifier token that is NOT in excluded_names is a
        // potential conflict participant -> fail-to-RED.
        if !excluded_names.contains(cleaned) {
            return false;
        }
    }

    true
}

// ── Verdict classification ────────────────────────────────────────────────────

/// Classify the diff into a verdict string. Precedence (first match wins):
/// SKIPPED > RED > UNSOLVABLE-EXCLUDED > VERSION-DIFF > GREEN.
pub(crate) fn classify(
    resolvo_outcome: &ResolvoOutcome,
    version_set: Option<&VersionSetDiff>,
    conda_routing: Option<&CondaRoutingDiff>,
    provenance: Option<&Vec<ProvenanceDiff>>,
    excluded: &[crate::handler::resolvo_discovery::ExcludedCandidate],
) -> String {
    match resolvo_outcome {
        ResolvoOutcome::Unsolvable { message } => {
            if demote_to_excluded(message, excluded) {
                "UNSOLVABLE-EXCLUDED".to_string()
            } else {
                "RED".to_string()
            }
        }
        ResolvoOutcome::Solved => {
            // Check routing mismatch (includes bundled_set_matches=false).
            if let Some(cr) = conda_routing
                && (!cr.bundled_set_matches || !cr.routing_mismatch.is_empty())
            {
                return "RED".to_string();
            }

            // Check only_in_bfs / only_in_resolvo.
            if let Some(vs) = version_set
                && (!vs.only_in_bfs.is_empty() || !vs.only_in_resolvo.is_empty())
            {
                return "RED".to_string();
            }

            // Check provenance diffs on shared name+version wheels.
            if let Some(prov) = provenance
                && !prov.is_empty()
            {
                return "RED".to_string();
            }

            // Check for version changes.
            if let Some(vs) = version_set
                && !vs.version_changed.is_empty()
            {
                return "VERSION-DIFF".to_string();
            }

            "GREEN".to_string()
        }
    }
}

// ── JSONL writer ──────────────────────────────────────────────────────────────

/// Write one `AbReport` as a JSON line to `<base_path>.<pid>.jsonl`.
/// Uses a single `write_all` call (whole-record single syscall) so concurrent
/// calls from multiple `resolve_bundle` invocations don't interleave records.
fn write_report(base_path: &str, report: &AbReport) -> Result<()> {
    let pid = std::process::id();
    let path = format!("{base_path}.{pid}.jsonl");
    let mut bytes = serde_json::to_vec(report)?;
    bytes.push(b'\n');
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    file.write_all(&bytes)?;
    Ok(())
}

// ── Public hooks ──────────────────────────────────────────────────────────────

/// Called at the source-form early return in `resolve_bundle`. Writes a
/// SKIPPED record so the driver knows this entry was out of scope.
pub(crate) async fn ab_skip_hook(
    report_path: &str,
    entry_name: &str,
    target: &WheelTarget,
    skip_reason: &str,
) {
    let conda_name = canonical_conda_name(entry_name);
    let report = AbReport {
        schema_version: AB_SCHEMA_VERSION,
        entry_name: entry_name.to_string(),
        target: format!("{}/{}", target.conda_subdir, target.python_version),
        phase: "skipped".to_string(),
        conda_name,
        verdict: "SKIPPED".to_string(),
        resolvo_outcome: ResolvoOutcome::Unsolvable {
            message: format!("skipped: {skip_reason}"),
        },
        version_set: None,
        conda_routing: None,
        provenance: None,
        skip_reason: Some(skip_reason.to_string()),
    };
    if let Err(e) = write_report(report_path, &report) {
        tracing::warn!(
            entry = %entry_name,
            error = %format!("{e:#}"),
            "ab_skip_hook: failed to write JSONL record"
        );
    }
}

/// The main A/B diff hook. Called right before `resolve_bundle` returns its
/// BFS bundle. Runs resolvo independently, diffs, classifies, and writes a
/// JSONL record. NEVER propagates errors.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn ab_diff_hook(
    report_path: &str,
    entry_name: &str,
    target: &WheelTarget,
    bfs_bundle: &Bundle,
    entry: &WheelEntry,
    download_dir: &std::path::Path,
    relax: RelaxPolicy,
    name_map: &std::collections::BTreeMap<String, String>,
    conda_channels: &[ChannelUrl],
    pypi_to_conda: &PypiToCondaMap,
    conda_deps: &[String],
    workspace_indexes: &[String],
) {
    let target_str = format!("{}/{}", target.conda_subdir, target.python_version);
    let conda_name = bfs_bundle.conda_name.clone();

    // Run the resolvo pipeline.
    let primary_rd = bfs_bundle.primary.metadata.requires_dist.clone();
    let pool_and_outcome = resolvo_solve_pool(
        &primary_rd,
        entry,
        target,
        download_dir,
        relax,
        name_map,
        conda_channels,
        pypi_to_conda,
        conda_deps,
        workspace_indexes,
    )
    .await;

    let (pool, outcome) = match pool_and_outcome {
        Ok(pair) => pair,
        Err(e) => {
            // Discovery I/O error — write RED record.
            let report = AbReport {
                schema_version: AB_SCHEMA_VERSION,
                entry_name: entry_name.to_string(),
                target: target_str,
                phase: "pre-auto-bundle".to_string(),
                conda_name,
                verdict: "RED".to_string(),
                resolvo_outcome: ResolvoOutcome::Unsolvable {
                    message: format!("ab-hook error: {e:#}"),
                },
                version_set: None,
                conda_routing: None,
                provenance: None,
                skip_reason: None,
            };
            if let Err(write_err) = write_report(report_path, &report) {
                tracing::warn!(
                    entry = %entry_name,
                    error = %format!("{write_err:#}"),
                    "ab_diff_hook: failed to write RED record after discovery error"
                );
            }
            return;
        }
    };

    // Map SolveOutcome -> ResolvoOutcome + resolved wheels.
    let (resolvo_outcome, resolvo_wheels): (ResolvoOutcome, Vec<ResolvedWheel>) = match outcome {
        SolveOutcome::Solved(solved_wheels) => {
            let mut rws: Vec<ResolvedWheel> = Vec::new();
            for sw in &solved_wheels {
                match pool_record_to_resolved_wheel(sw, &pool) {
                    Ok(rw) => rws.push(rw),
                    Err(e) => {
                        tracing::warn!(
                            name = %sw.pypi_name,
                            error = %format!("{e:#}"),
                            "ab_diff_hook: failed to map solved wheel; writing RED"
                        );
                        let report = AbReport {
                            schema_version: AB_SCHEMA_VERSION,
                            entry_name: entry_name.to_string(),
                            target: target_str,
                            phase: "pre-auto-bundle".to_string(),
                            conda_name,
                            verdict: "RED".to_string(),
                            resolvo_outcome: ResolvoOutcome::Unsolvable {
                                message: format!("ab-hook error: pool map failed: {e:#}"),
                            },
                            version_set: None,
                            conda_routing: None,
                            provenance: None,
                            skip_reason: None,
                        };
                        if let Err(write_err) = write_report(report_path, &report) {
                            tracing::warn!(error = %format!("{write_err:#}"), "ab_diff_hook: write failed");
                        }
                        return;
                    }
                }
            }
            (ResolvoOutcome::Solved, rws)
        }
        SolveOutcome::Unsolvable(msg) => (ResolvoOutcome::Unsolvable { message: msg }, vec![]),
    };

    // Build diff components.
    //
    // The resolvo side omits the primary: resolvo_solve_pool discovers
    // *transitive* deps only (the primary is already materialised by the
    // caller and stored in bfs_bundle.primary unchanged).  Add it back so
    // both sides are compared on an equal footing: primary + extras.
    let resolvo_with_primary: Vec<ResolvedWheel> = std::iter::once(bfs_bundle.primary.clone())
        .chain(resolvo_wheels.iter().cloned())
        .collect();

    let bfs_bundled_set = derive_bundled_set(bfs_bundle);
    let resolvo_bundled_set: std::collections::HashSet<String> = resolvo_with_primary
        .iter()
        .map(|w| w.pypi_name.clone())
        .collect();

    let resolvo_conda_routed: Vec<String> = pool.conda_routed_names.iter().cloned().collect();

    let (version_set, conda_routing, provenance) = match &resolvo_outcome {
        ResolvoOutcome::Solved => {
            let vs = build_version_set_diff(bfs_bundle, &resolvo_with_primary);
            let cr = build_conda_routing_diff(
                bfs_bundle,
                &bfs_bundled_set,
                &resolvo_conda_routed,
                &resolvo_bundled_set,
            );
            let prov = build_provenance_diffs(bfs_bundle, &resolvo_with_primary);
            (Some(vs), Some(cr), Some(prov))
        }
        ResolvoOutcome::Unsolvable { .. } => (None, None, None),
    };

    let verdict = classify(
        &resolvo_outcome,
        version_set.as_ref(),
        conda_routing.as_ref(),
        provenance.as_ref(),
        &pool.excluded_sdist_builds,
    );

    let report = AbReport {
        schema_version: AB_SCHEMA_VERSION,
        entry_name: entry_name.to_string(),
        target: target_str,
        phase: "pre-auto-bundle".to_string(),
        conda_name,
        verdict,
        resolvo_outcome,
        version_set,
        conda_routing,
        provenance,
        skip_reason: None,
    };

    if let Err(e) = write_report(report_path, &report) {
        tracing::warn!(
            entry = %entry_name,
            error = %format!("{e:#}"),
            "ab_diff_hook: failed to write JSONL record"
        );
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handler::resolvo_discovery::ExcludedCandidate;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn solved() -> ResolvoOutcome {
        ResolvoOutcome::Solved
    }

    fn unsolvable(msg: &str) -> ResolvoOutcome {
        ResolvoOutcome::Unsolvable {
            message: msg.to_string(),
        }
    }

    fn vs(
        shared: &[&str],
        changed: &[(&str, &str, &str)],
        only_bfs: &[&str],
        only_resolvo: &[&str],
    ) -> VersionSetDiff {
        VersionSetDiff {
            shared_same_version: shared.iter().map(|s| s.to_string()).collect(),
            version_changed: changed
                .iter()
                .map(|(n, b, r)| VersionChange {
                    name: n.to_string(),
                    bfs: b.to_string(),
                    resolvo: r.to_string(),
                })
                .collect(),
            only_in_bfs: only_bfs.iter().map(|s| s.to_string()).collect(),
            only_in_resolvo: only_resolvo.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn cr(matches: bool, mismatches: &[(&str, Side, Side)]) -> CondaRoutingDiff {
        CondaRoutingDiff {
            bfs_conda_routed: vec![],
            resolvo_conda_routed: vec![],
            bundled_set_matches: matches,
            routing_mismatch: mismatches
                .iter()
                .map(|(n, b, r)| RoutingMismatch {
                    name: n.to_string(),
                    bfs: *b,
                    resolvo: *r,
                })
                .collect(),
        }
    }

    fn clean_prov() -> Vec<ProvenanceDiff> {
        vec![]
    }

    fn dirty_prov(name: &str) -> Vec<ProvenanceDiff> {
        vec![ProvenanceDiff {
            name: name.to_string(),
            version: "1.0".to_string(),
            fields: vec![ProvenanceFieldDiff {
                field: "upstream_url".to_string(),
                bfs: Some("https://bfs.example.com/a.whl".to_string()),
                resolvo: Some("https://resolvo.example.com/a.whl".to_string()),
            }],
        }]
    }

    fn no_excluded() -> Vec<ExcludedCandidate> {
        vec![]
    }

    fn excluded(name: &str, version: &str) -> Vec<ExcludedCandidate> {
        vec![ExcludedCandidate {
            pypi_name: name.to_string(),
            version: version.to_string(),
            reason: "sdist build failed".to_string(),
        }]
    }

    // ── classify truth table ──────────────────────────────────────────────────

    #[test]
    fn classify_green_when_all_match() {
        let verdict = classify(
            &solved(),
            Some(&vs(&["pkg-a", "pkg-b"], &[], &[], &[])),
            Some(&cr(true, &[])),
            Some(&clean_prov()),
            &no_excluded(),
        );
        assert_eq!(verdict, "GREEN");
    }

    #[test]
    fn classify_version_diff_when_version_changed_and_bundled_set_matches() {
        let verdict = classify(
            &solved(),
            Some(&vs(&["pkg-b"], &[("pkg-a", "1.0", "1.1")], &[], &[])),
            Some(&cr(true, &[])),
            Some(&clean_prov()),
            &no_excluded(),
        );
        assert_eq!(verdict, "VERSION-DIFF");
    }

    #[test]
    fn classify_red_on_provenance_diff_in_shared_version() {
        let verdict = classify(
            &solved(),
            Some(&vs(&["pkg-a"], &[], &[], &[])),
            Some(&cr(true, &[])),
            Some(&dirty_prov("pkg-a")),
            &no_excluded(),
        );
        assert_eq!(verdict, "RED");
    }

    #[test]
    fn classify_red_on_routing_flip() {
        // pkg-a is bundled in BFS but conda-routed in resolvo.
        let verdict = classify(
            &solved(),
            Some(&vs(&[], &[], &["pkg-a"], &[])),
            Some(&cr(false, &[("pkg-a", Side::Bundled, Side::Conda)])),
            Some(&clean_prov()),
            &no_excluded(),
        );
        assert_eq!(verdict, "RED");
    }

    #[test]
    fn classify_red_on_only_in_bfs() {
        let verdict = classify(
            &solved(),
            Some(&vs(&["pkg-b"], &[], &["pkg-a"], &[])),
            Some(&cr(false, &[])),
            Some(&clean_prov()),
            &no_excluded(),
        );
        assert_eq!(verdict, "RED");
    }

    #[test]
    fn classify_red_on_only_in_resolvo() {
        let verdict = classify(
            &solved(),
            Some(&vs(&["pkg-b"], &[], &[], &["pkg-c"])),
            Some(&cr(false, &[])),
            Some(&clean_prov()),
            &no_excluded(),
        );
        assert_eq!(verdict, "RED");
    }

    #[test]
    fn classify_unsolvable_excluded_when_demote_succeeds() {
        // Narrow safe case: message contains ONLY the excluded name+version.
        // Any prose tokens (is/required/excluded/…) would force RED.
        let msg = "pkg-x 2.0";
        let verdict = classify(
            &unsolvable(msg),
            None,
            None,
            None,
            &excluded("pkg-x", "2.0"),
        );
        assert_eq!(verdict, "UNSOLVABLE-EXCLUDED");
    }

    #[test]
    fn classify_red_on_unsolvable_conflict_fail_to_red() {
        // Unsolvable message references a non-excluded package — ambiguous -> RED.
        let msg = "pkg-a 1.0 requires pkg-b >=2.0 but pkg-b 1.5 is installed";
        let verdict = classify(
            &unsolvable(msg),
            None,
            None,
            None,
            &excluded("pkg-x", "3.0"), // excluded is different package
        );
        assert_eq!(verdict, "RED");
    }

    #[test]
    fn classify_red_on_unsolvable_with_no_excluded() {
        let msg = "dependency conflict";
        let verdict = classify(&unsolvable(msg), None, None, None, &no_excluded());
        assert_eq!(verdict, "RED");
    }

    // ── demote_to_excluded tests ──────────────────────────────────────────────

    #[test]
    fn demote_empty_excluded_returns_false() {
        assert!(!demote_to_excluded("pkg-x 2.0 conflict", &no_excluded()));
    }

    #[test]
    fn demote_ambiguous_returns_false() {
        // Message references pkg-a which is NOT in excluded.
        let msg = "pkg-a 1.0 conflicts with pkg-b 2.0";
        let exc = excluded("pkg-x", "3.0");
        assert!(!demote_to_excluded(msg, &exc));
    }

    #[test]
    fn demote_narrow_safe_returns_true() {
        // Message contains ONLY the excluded name and a version token.
        // Any English prose would force RED (fail-to-RED-harder posture).
        let msg = "pkg-x 2.0";
        let exc = excluded("pkg-x", "2.0");
        assert!(demote_to_excluded(msg, &exc));
    }

    #[test]
    fn demote_single_word_conflict_name_forces_red() {
        // Regression: a genuine conflict over single-word "numpy" co-occurring
        // with excluded "tinyobjloader-python" must NOT be demoted to
        // UNSOLVABLE-EXCLUDED. The old hyphen/underscore short-circuit silently
        // ignored "numpy" and returned true (false green). The fixed heuristic
        // returns false -> RED.
        let msg = "numpy 1.26.4, which conflicts with the versions reported above. \
                   tinyobjloader-python 2.0 is excluded because sdist build failed";
        let exc = excluded("tinyobjloader-python", "2.0");
        assert!(!demote_to_excluded(msg, &exc));
        // Verify the full classify path also returns RED.
        let verdict = classify(&unsolvable(msg), None, None, None, &exc);
        assert_eq!(verdict, "RED");
    }

    // ── primary-in-resolvo-with-primary regression ────────────────────────────
    //
    // Before PR-2-fix2, ab_diff_hook built resolvo_bundled_set from
    // resolvo_wheels ONLY (extras), omitting the primary.  Every entry
    // false-REDed on its own primary being "only_in_bfs".  This test
    // verifies that an entry with no extras (resolvo_wheels=vec![]) and a
    // matching primary classifies as GREEN.

    #[test]
    fn classify_green_when_resolvo_extras_empty_primary_matches() {
        let mk_wheel = |name: &str, version: &str| -> ResolvedWheel {
            let normalized = name.replace('-', "_");
            ResolvedWheel {
                pypi_name: name.to_string(),
                url: url::Url::parse("file:///tmp/fake.whl").unwrap(),
                upstream_url: None,
                git_source: None,
                sdist_source: None,
                metadata: crate::wheel::WheelMetadata {
                    name: name.to_string(),
                    version: version.to_string(),
                    requires_dist: vec![],
                    is_pure_python: true,
                    sha256: String::new(),
                    filename: format!("{normalized}-{version}-py3-none-any.whl"),
                },
                extras_requested: vec![],
                auto_data: None,
                auto_data_dedup_skipped_root: None,
            }
        };

        // BFS bundle: primary only, no extras.
        let bundle = Bundle {
            conda_name: "my-pkg".to_string(),
            primary: mk_wheel("my-pkg", "1.0"),
            extras: vec![],
            probe_decisions: vec![],
            solve_diagnostics: std::collections::BTreeMap::new(),
            conda_routed: vec![],
        };

        // resolvo_wheels = vec![] (no extras; primary already in bfs_bundle.primary).
        // resolvo_with_primary = [primary].  Both sides have {"my-pkg" -> "1.0"}.
        let resolvo_wheels: Vec<ResolvedWheel> = vec![];
        let resolvo_with_primary: Vec<ResolvedWheel> = std::iter::once(bundle.primary.clone())
            .chain(resolvo_wheels.iter().cloned())
            .collect();

        let bfs_bundled_set = derive_bundled_set(&bundle);
        let resolvo_bundled_set: std::collections::HashSet<String> = resolvo_with_primary
            .iter()
            .map(|w| w.pypi_name.clone())
            .collect();

        let version_set = build_version_set_diff(&bundle, &resolvo_with_primary);
        let conda_routing =
            build_conda_routing_diff(&bundle, &bfs_bundled_set, &[], &resolvo_bundled_set);
        let provenance = build_provenance_diffs(&bundle, &resolvo_with_primary);

        let verdict = classify(
            &solved(),
            Some(&version_set),
            Some(&conda_routing),
            Some(&provenance),
            &no_excluded(),
        );
        assert_eq!(
            verdict, "GREEN",
            "primary-only bundle with matching versions must be GREEN"
        );
    }

    // ── derive_bundled_set symmetry ───────────────────────────────────────────

    #[test]
    fn derive_bundled_set_includes_primary_and_extras() {
        // Build a minimal Bundle via helper.
        let mk_wheel = |name: &str| -> ResolvedWheel {
            let normalized = name.replace('-', "_");
            ResolvedWheel {
                pypi_name: name.to_string(),
                url: url::Url::parse("file:///tmp/fake.whl").unwrap(),
                upstream_url: None,
                git_source: None,
                sdist_source: None,
                metadata: crate::wheel::WheelMetadata {
                    name: name.to_string(),
                    version: "1.0".to_string(),
                    requires_dist: vec![],
                    is_pure_python: true,
                    sha256: String::new(),
                    filename: format!("{normalized}-1.0-py3-none-any.whl"),
                },
                extras_requested: vec![],
                auto_data: None,
                auto_data_dedup_skipped_root: None,
            }
        };

        let bundle = Bundle {
            conda_name: "my-pkg".to_string(),
            primary: mk_wheel("my-pkg"),
            extras: vec![mk_wheel("my-pkg-sub")],
            probe_decisions: vec![],
            solve_diagnostics: std::collections::BTreeMap::new(),
            conda_routed: vec![],
        };

        let set = derive_bundled_set(&bundle);
        assert!(set.contains("my-pkg"), "primary must be in bundled set");
        assert!(set.contains("my-pkg-sub"), "extra must be in bundled set");
        assert_eq!(set.len(), 2);
    }
}
