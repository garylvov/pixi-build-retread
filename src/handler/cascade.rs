//! Widen-cascade cluster: iterative solve refinement, per-dep tiered
//! cascade (conda -> PyPI fallback), and bundle/index helpers.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::str::FromStr;

use anyhow::Result;
use rattler_conda_types::ChannelUrl;
use uv_pep508::uv_pep440::VersionSpecifiers;

use crate::config::{RelaxPolicy, RetreadConfig, WheelEntry};
use crate::pypi;
use crate::relax::canonical_conda_name;

use super::auto_bundle::metadata_preferring_sidecar;
use super::{Bundle, ResolvedWheel};

/// Cap on how many widening rounds `iterative_solve_refinement` may
/// spend before giving up.
///
/// > v0.36.2+: bumped from 5 to 10. The original 5-iter cap was set when
/// > the cascade widened multiple deps per round opportunistically. With
/// > the per-chain verdict model, each round may surface a NEW set of
/// > blockers as earlier widenings unlock candidates -- so longer chains
/// > of progressive widening need more headroom. 10 is still bounded
/// > enough that pathological cases terminate quickly.
const MAX_REFINEMENT: usize = 10;

pub(crate) fn widening_level(spec: &str) -> u8 {
    let trimmed = spec.trim();
    if trimmed == "*" || trimmed.is_empty() {
        return 4;
    }
    // Exact pin (`==X.Y.Z`) is the tightest possible spec. It has
    // no `<` upper bound so the major/minor heuristic below would
    // misclassify it as level 2 (major-widened) — guard before
    // calling the heuristic. Also catches comma-chained exact pins
    // like `>=1.4,==2.10.0`.
    if trimmed.contains("==") {
        return 0;
    }
    let Some(version_str) = extract_anchor_version(trimmed) else {
        // Pure upper-bound / exclusion / unrecognized shape (e.g.
        // `<2`, `!=1.0`). `widen_one_level` jumps straight to `*`
        // for these, so anything wider than 0 indicates we widened.
        return 0;
    };
    let parts: Vec<&str> = version_str.split('.').collect();
    let major: u64 = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
    let minor: u64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let next_major = major + 1;
    let next_minor = minor + 1;
    let has_minor_upper = trimmed.contains(&format!("<{major}.{next_minor}"));
    let has_major_upper = trimmed.contains(&format!("<{next_major}"));
    let lower_has_minor = trimmed.contains(&format!(">={major}."));
    if has_minor_upper {
        0
    } else if has_major_upper && lower_has_minor {
        1 // minor range `>=A.B,<A+1`
    } else if has_major_upper {
        2 // major floor, still bounded `>=A,<A+1`
    } else {
        3 // major-open `>=A`, no upper
    }
}

/// v0.36.4+: merge a candidate (dep, spec) into `accum`, keeping
/// whichever is LOOSER per `widening_level`. Used after each env's
/// `iterative_solve_refinement` to accumulate the per-dep widenings
/// across envs — the emitted output ships ONE set of run-deps that
/// every env must accept, so we always carry forward the widest.
///
/// Ties go to the existing spec to keep behavior stable: if env A
/// and env B both landed on the same level for `dep`, A's spec
/// stays. Levels are monotone with respect to `widen_one_level`'s
/// steps so this comparison is the natural lattice join.
pub(crate) fn merge_looser_override(
    accum: &mut std::collections::BTreeMap<String, String>,
    dep: &str,
    candidate: &str,
) {
    let new_level = widening_level(candidate);
    let existing_level = accum.get(dep).map(|s| widening_level(s)).unwrap_or(0);
    if new_level > existing_level {
        // 0c boundary contract (debug-only): widened specs entering
        // the accumulation chokepoint must parse as MatchSpecs.
        crate::relax::assert_spec_roundtrips(dep, candidate);
        accum.insert(dep.to_string(), candidate.to_string());
    }
}

/// Progressively widen a conda match-spec by ONE level. Detects the
/// current widening level from the spec's upper bound shape and
/// returns the next-wider spec:
///
///   - Patch (`>=A.B.C,<A.B+1`)        -> Minor (`>=A.B,<A+1`)
///   - Minor (`>=A.B,<A+1`)            -> Major (`>=A`)
///   - Major (`>=A`)                   -> Star  (`*`)
///   - `*`                             -> None (already maximally wide)
///   - exact (`==A.B.C`) or unknown    -> Minor (treat as Patch's next)
///
/// Returns `None` only when the spec is already `*`.
pub(crate) fn widen_one_level(current_spec: &str) -> Option<String> {
    let trimmed = current_spec.trim();
    if trimmed == "*" || trimmed.is_empty() {
        return None;
    }
    // Extract the lower-bound version (the `>=A.B.C` part, or the
    // exact pin in `==A.B.C`). Used as the anchor for widening.
    let Some(version_str) = extract_anchor_version(trimmed) else {
        // No anchor version found -- the spec is pure upper-bound
        // (`<X`, `<=X`) or pure exclusion (`!=X`). Can't escalate
        // through patch/minor/major levels because there's no
        // lower-bound anchor to widen FROM. The only meaningful
        // widening is to drop the constraint entirely. Skip
        // intermediate steps and jump to `*`. The user explicitly
        // chose this dep, so widening to `*` is consistent with
        // last-resort behavior elsewhere in the cascade.
        return Some("*".to_string());
    };
    let parts: Vec<&str> = version_str.split('.').collect();
    let major: u64 = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
    let minor: u64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

    // Detect current level by inspecting the upper bound:
    //   contains `<A.B+1`  -> Patch (next: Minor)
    //   contains `<A+1`    -> Minor (next: Major)
    //   no upper bound     -> Major (next: Star)
    let next_major = major + 1;
    let next_minor = minor + 1;
    let has_minor_upper = trimmed.contains(&format!("<{major}.{next_minor}"));
    let has_major_upper = trimmed.contains(&format!("<{next_major}"));
    // Does the lower bound carry a minor component (`>=A.B...`) or is it
    // a bare major floor (`>=A`)? Distinguishes `>=2.10,<3` (minor lower)
    // from `>=2,<3` (major-floor lower) so the major step makes progress
    // WITHOUT dropping the upper bound prematurely.
    let lower_has_minor = trimmed.contains(&format!(">={major}."));
    if has_minor_upper {
        // patch/minor -> minor: `>=3.7.0,<3.8` -> `>=3.7,<4`.
        Some(format!(">={major}.{minor},<{next_major}"))
    } else if has_major_upper && lower_has_minor {
        // minor range -> major FLOOR, KEEPING the `<A+1` upper:
        // `>=2.10,<3` -> `>=2,<3`. (v0.46.0: preserve the upper so the
        // emitted pack stays bounded to one major even without a
        // workspace pin -- previously this dropped to `>=2`, unbounded.)
        Some(format!(">={major},<{next_major}"))
    } else {
        // Either a bare-major floor with an upper (`>=2,<3`) or no upper
        // bound at all (`>=2`) -- both have nowhere tighter to go but
        // `*`. (Dropping the major upper is the last widening step.)
        Some("*".to_string())
    }
}

/// Extract the major.minor (.patch) version string from a conda
/// match-spec's lower bound or exact pin. Returns None for specs
/// that have no parseable anchor version (e.g. `>=A,!=B`).
pub(crate) fn extract_anchor_version(spec: &str) -> Option<String> {
    // Among all `>=X.Y.Z` / `==X.Y.Z` / `>X` clauses, pick the HIGHEST
    // (tightest) anchor version.
    //
    // A merged spec can carry several lower bounds from different wheels:
    // e.g. `>=1.4,==2.10.0,>=2.10.0,<2.11.0a0` (one wheel pins
    // `torch>=1.4`, isaacsim pins `torch==2.10.0`). The tightest lower
    // bound (2.10.0) is the one that actually constrains, so widen FROM
    // it. The previous "first clause wins" logic anchored on the stray
    // `>=1.4`, and `has_major_upper`'s `<2` substring test then matched
    // `<2.11.0a0`, collapsing the spec to `>=1` in a single step --
    // discarding the real 2.10 anchor AND the upper bound. Comparing
    // candidates as PEP 440 versions and keeping the max fixes that:
    // 2.10.0 widens to `>=2.10,<3` (the `<2.11` upper is recognized),
    // then `>=2`, never `>=1`.
    use std::str::FromStr;
    let mut best: Option<(uv_pep508::uv_pep440::Version, String)> = None;
    let mut fallback: Option<String> = None;
    for clause in spec.split(',') {
        let c = clause.trim();
        let Some(payload) = c
            .strip_prefix(">=")
            .or_else(|| c.strip_prefix("=="))
            .or_else(|| c.strip_prefix(">"))
        else {
            continue;
        };
        let payload = payload.trim();
        if payload.is_empty() {
            continue;
        }
        // Take the leading run of digits-and-dots.
        let end = payload
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(payload.len());
        let head = payload[..end].trim_end_matches('.');
        if head.is_empty() || !head.chars().any(|c| c.is_ascii_digit()) {
            continue;
        }
        // First parseable clause is the fallback if NONE compare cleanly.
        if fallback.is_none() {
            fallback = Some(head.to_string());
        }
        if let Ok(v) = uv_pep508::uv_pep440::Version::from_str(head)
            && best.as_ref().is_none_or(|(bv, _)| &v > bv)
        {
            best = Some((v, head.to_string()));
        }
    }
    best.map(|(_, s)| s).or(fallback)
}

/// v0.36.0+: post-condition invariant. After `produce_output` runs in
/// the refinement loop, verify the emitted output respects the ABI
/// contract. Returns one human-readable message per violation, in a
/// stable order. The caller logs them loudly + threads them into the
/// audit; the cascade does NOT fail on violations because (a) the
/// invariant is new and may have false-positive shapes we haven't seen
/// yet, (b) silently corrupting an output is strictly worse than
/// loudly continuing with a flag in the audit.
///
/// Three checks per dep emitted in `run_dependencies.depends`:
///   1. If the dep is an ABI anchor (`is_abi_anchor`), its emitted
///      spec must NOT be empty / `*`. Empty means retread widened it
///      to "any version" -- the exact corruption we're guarding
///      against (gsi round 4 emitting `python *`).
///   2. If the workspace pins the same ABI-anchor dep, retread's
///      emitted spec must NOT be looser. The cheap "looser" detector:
///      if the workspace spec is `==X.Y` and retread's contains no
///      `==X` / `>=X` / `<=X` clause anchored at the same major or
///      tighter, flag it.
///   3. The `effective.overrides` map must not contain an ABI-anchor
///      entry mapped to `*` (the inverse direction -- catches code
///      paths where overrides are mutated independently of the
///      output's run_deps).
pub(crate) fn check_output_abi_invariants(
    output_run_deps: &[(String, String)],
    workspace_specs: &[String],
    overrides: &std::collections::BTreeMap<String, String>,
) -> Vec<String> {
    use crate::conflict_classifier::is_abi_anchor;
    let mut violations: Vec<String> = Vec::new();

    // (1) + (2): walk retread's emitted run_deps.
    for (name, spec) in output_run_deps {
        if !is_abi_anchor(name) {
            continue;
        }
        let trimmed = spec.trim();
        // (1) Empty or `*` on an ABI anchor is always a corruption.
        if trimmed.is_empty() || trimmed == "*" {
            violations.push(format!(
                "ABI invariant: retread emitted `{name} {trimmed}` (empty/*) -- ABI anchors must always carry a concrete spec",
            ));
            continue;
        }
        // v0.37.0+: bare-major glob (`3.*`, `12.*`) is the second
        // corruption shape, distinct from `*`/empty. Surfaced by the
        // pixi-forwards-bare-major-python bug: retread emitted `python
        // 3.*` in host/run deps because the variant arrived as `"3"`.
        // `pythons_for` now rejects bare-major at the input boundary
        // (v0.37.0 D2) so this should never fire in practice, but the
        // invariant is defense-in-depth — a future code path that
        // synthesizes specs from a major-only template would otherwise
        // recreate the gsn corruption silently.
        //
        // Conservative shape detection: `^[0-9]+\.\*$` (digits + dot
        // + asterisk, nothing else). Anything tighter (`3.11.*`,
        // `>=3.11`, `==3.11.5`) is fine.
        let trimmed_no_ws = trimmed.replace(' ', "");
        let is_bare_major_glob = {
            let parts: Vec<&str> = trimmed_no_ws.split('.').collect();
            parts.len() == 2
                && parts[0].chars().all(|c| c.is_ascii_digit())
                && !parts[0].is_empty()
                && parts[1] == "*"
        };
        if is_bare_major_glob {
            violations.push(format!(
                "ABI invariant: retread emitted `{name} {trimmed}` (bare-major glob) -- ABI anchors must carry minor or stricter (e.g. `3.11.*`)",
            ));
            continue;
        }
        // (2) Compare with workspace pin if present. The check is
        // intentionally conservative: only flag if the workspace
        // pin's lower-bound major doesn't appear in retread's spec
        // at all (e.g. workspace `python ==3.11`, retread `python >=3`
        // -> retread covers 3.x AND 4.x which is broader than ==3.11
        // but at least contains it; a spec like `python ==3.11.5`
        // would also be OK because the workspace pin still allows it).
        // The corruption shape we MUST catch is retread emitting a
        // spec with NO version anchor at all (handled by (1)).
        // Future refinement: parse both as conda VersionSpec and
        // check spec.intersects(workspace_spec) == workspace_spec.
        // Out of scope for v0.36.0; the (1) check covers the gsi bug.
        let ws_spec: Option<&String> = workspace_specs.iter().find_map(|w| {
            let mut parts = w.splitn(2, char::is_whitespace);
            let n = parts.next()?;
            if n == name {
                parts.next().map(|_| w)
            } else {
                None
            }
        });
        if let Some(_ws) = ws_spec {
            // Workspace pins this dep too; for now we just record
            // the fact in trace-level logging (no violation flag).
            tracing::trace!(
                dep = %name,
                retread_spec = %trimmed,
                "ABI invariant: both retread and workspace emit ABI anchor; relying on conda solver to reconcile",
            );
        }
    }

    // (3) Walk overrides for ABI anchors mapped to `*`.
    for (k, v) in overrides {
        if !is_abi_anchor(k) {
            continue;
        }
        let trimmed = v.trim();
        if trimmed.is_empty() || trimmed == "*" {
            violations.push(format!(
                "ABI invariant: `effective.overrides[{k}]` is `{trimmed}` -- ABI anchors must never be widened to `*`",
            ));
        }
    }

    violations
}

/// Compact a list of "dep -> spec" change strings into one short line
/// for the /dev/tty status. Shows the first two verbatim and collapses
/// the rest into "(+N more)" so the status never wraps the console even
/// when a round widens many deps at once.
fn summarize_changes(changes: &[String]) -> String {
    match changes.len() {
        0 => "(no change)".to_string(),
        1 | 2 => changes.join(", "),
        n => format!("{}, {} (+{} more)", changes[0], changes[1], n - 2),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn iterative_solve_refinement(
    base_run_deps: &[String],
    seed_overrides: &BTreeMap<String, String>,
    workspace_specs: &[String],
    channels: &[ChannelUrl],
    python_version: &str,
    target_subdir: &str,
    bundle: &mut Bundle,
    effective: &mut RetreadConfig,
    host_platform: rattler_conda_types::Platform,
    siblings: &[(String, String)],
    env_name: &str,
    workspace_manifest: Option<&crate::workspace::WorkspaceManifest>,
    channel_priority: rattler_solve::ChannelPriority,
    system_requirements: &std::collections::BTreeMap<String, String>,
) -> Result<crate::solve_check::SolveOutcome> {
    use crate::conflict_classifier::{
        PerChainVerdict, WorkspaceEditSuggestion, classify_chains, is_abi_anchor,
        summarize_verdicts,
    };
    let mut current_run_deps = base_run_deps.to_vec();
    let mut refinement_steps: Vec<crate::audit::RefinementStep> = Vec::new();
    let mut widened_to_star: HashSet<String> = HashSet::new();

    // v0.44.0: seed from sibling envs' already-discovered widenings.
    // env_names is solved base-first (see conda_outputs), so by the
    // time a superset env reaches here, `seed_overrides`
    // (= accumulated_overrides) carries the looser specs the base env
    // converged on. Apply any that are STRICTLY looser than this env's
    // current override for the same dep, then re-emit once so iter 0
    // solves against the shared widening frontier instead of
    // re-discovering it one level per round.
    //
    // SAFE re: the v0.36.2 false-SAT bug. That bug was leaked &mut
    // state letting an env be reported sat off another env's widenings
    // WITHOUT its own verification. Here every iteration -- including
    // iter 0 with the seed applied -- runs a real `run_solve_check`
    // resolvo solve, so a SAT verdict always reflects a genuine
    // solution for THIS env's full spec set. Seeding only changes how
    // many iterations we need; never whether we verify. ABI anchors are
    // never seeded (defense-in-depth: classify_chains/widen already
    // refuse them, and the post-widen invariant check still runs).
    let seed_to_apply: Vec<(String, String)> = seed_overrides
        .iter()
        .filter(|&(dep, spec)| {
            !is_abi_anchor(dep)
                && widening_level(spec)
                    > effective
                        .overrides
                        .get(dep)
                        .map(|s| widening_level(s))
                        .unwrap_or(0)
        })
        .map(|(d, s)| (d.clone(), s.clone()))
        .collect();
    if !seed_to_apply.is_empty() {
        for (dep, spec) in &seed_to_apply {
            if *spec == "*" {
                widened_to_star.insert(dep.clone());
            }
            effective.overrides.insert(dep.clone(), spec.clone());
        }
        let seeded =
            super::produce_output(bundle, effective, host_platform, python_version, siblings)?;
        current_run_deps = seeded
            .run_dependencies
            .depends
            .iter()
            .map(|n| {
                let raw = super::audit_report::format_packagespec(&n.spec);
                if raw.is_empty() {
                    n.name.as_str().to_string()
                } else {
                    format!("{} {raw}", n.name.as_str())
                }
            })
            .collect();
        tracing::info!(
            env = %env_name,
            seeded = ?seed_to_apply.iter().map(|(d, s)| format!("{d} -> {s}")).collect::<Vec<_>>(),
            "iterative refinement: seeded from sibling envs' widenings (base-first ordering)",
        );
    }
    let workspace_dep_names: HashSet<String> = workspace_specs
        .iter()
        .filter_map(|s| s.split_whitespace().next().map(String::from))
        .collect();

    // One-line summary of what the PREVIOUS attempt changed (the dep it
    // widened, or the sibling-env seed), shown as a suffix on the next
    // attempt's status line so the user can see WHY we're re-solving --
    // without a second line per attempt. None on attempt 1 (nothing
    // widened yet).
    let mut last_action: Option<String> = if seed_to_apply.is_empty() {
        None
    } else {
        Some(format!(
            "seeded from base env: {}",
            summarize_changes(
                &seed_to_apply
                    .iter()
                    .map(|(d, s)| format!("{d} -> {s}"))
                    .collect::<Vec<_>>()
            ),
        ))
    };

    for iter in 0..=MAX_REFINEMENT {
        // Heartbeat to /dev/tty: each solve over ~1M records takes a while and
        // pixi shows nothing during conda/outputs, so without this the user
        // sees a multi-minute silent freeze. (stderr equivalent is hidden.)
        let suffix = match &last_action {
            Some(a) => format!(" -- {a}"),
            None => String::new(),
        };
        crate::status::tty(&format!(
            "solve-check: env '{env_name}' (attempt {}){suffix}",
            iter + 1
        ));
        let mut combined = current_run_deps.clone();
        combined.extend(workspace_specs.iter().cloned());
        let mut outcome = crate::solve_check::run_solve_check(
            channels,
            &combined,
            python_version,
            target_subdir,
            channel_priority,
            system_requirements,
        )
        .await;
        if outcome.satisfiable {
            outcome.refinement_steps = refinement_steps;
            return Ok(outcome);
        }
        if outcome.skipped {
            // v1.4.0: the check COULD NOT RUN (no channels / no
            // repodata). Abstain: don't classify the "no repodata"
            // text as an unsat chain, don't widen anything, don't
            // burn MAX_REFINEMENT rounds against a checker with zero
            // information. terminal_classification stays None so the
            // fail gate downstream doesn't treat this as a
            // workspace-blocked env.
            tracing::warn!(
                env = %env_name,
                "solve check skipped (no repodata available); emitting without pre-emission verification",
            );
            outcome.refinement_steps = refinement_steps;
            return Ok(outcome);
        }

        // v0.36.0: per-chain verdicts (no aggregate-class collapse).
        let chains = crate::solve_check::extract_blocking_chains(&outcome.unsat_explanations);
        let emitted_names: HashSet<String> = current_run_deps
            .iter()
            .filter_map(|s| s.split_whitespace().next().map(String::from))
            .collect();
        let verdicts = classify_chains(
            &chains,
            &emitted_names,
            &workspace_dep_names,
            &widened_to_star,
            env_name,
        );
        let blocking_summary = summarize_verdicts(&verdicts);

        // Collect suggestions FROM the verdicts (per-chain, in order).
        let mut suggestions: Vec<WorkspaceEditSuggestion> = Vec::new();
        for v in &verdicts {
            if let PerChainVerdict::WorkspacePinDominates {
                suggestion: Some(s),
                ..
            } = v
            {
                let mut s = s.clone();
                if s.feature.is_none() {
                    let name_only = s
                        .current_pin
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .to_string();
                    s.feature = workspace_manifest
                        .and_then(|m| m.find_declaring_feature(env_name, &name_only));
                }
                suggestions.push(s);
            }
        }

        // v0.36.0 policy: widen ONLY `WidenRetread` verdicts. If ANY
        // non-widenable verdict is present (AbiAnchor /
        // WorkspacePinDominates / TransitiveOnly / AlreadyExhausted)
        // AND no widenable verdict offers progress, stop the loop and
        // surface the verdict-derived diagnostics.
        //
        // Why "AND no widenable verdict": if pytorch (widenable) +
        // python (ABI anchor) both block, the cascade still gets to
        // widen pytorch on this iteration -- maybe widening it
        // resolves the conflict on the next round without ever
        // touching python. Stopping the loop the moment an ABI anchor
        // shows up would mean retread never tries the widen-pytorch
        // step, and the user gets a workspace-edit suggestion for a
        // conflict the cascade COULD have resolved. So: try to widen
        // every widenable; only stop if there's nothing to widen.
        let widenable: Vec<&PerChainVerdict> =
            verdicts.iter().filter(|v| v.is_widenable()).collect();

        if widenable.is_empty() {
            // Nothing to widen. Stop with the structured diagnostic
            // derived from the verdict mix.
            let blocking_deps_list: Vec<String> = verdicts.iter().map(|v| v.dep().into()).collect();
            let class_tag = derive_class_tag(&verdicts);
            refinement_steps.push(crate::audit::RefinementStep {
                iteration: iter,
                blocking_deps: blocking_deps_list,
                widened_deps: Vec::new(),
                classification: Some(class_tag.clone()),
                blocking_summary: blocking_summary.clone(),
                verdicts: verdicts.clone(),
                invariant_violations: Vec::new(),
            });
            outcome.refinement_steps = refinement_steps;
            outcome.workspace_edit_suggestions = suggestions;
            outcome.terminal_classification = Some(class_tag);
            return Ok(outcome);
        }

        if iter == MAX_REFINEMENT {
            // We HAD widenable verdicts but ran out of iterations.
            refinement_steps.push(crate::audit::RefinementStep {
                iteration: iter,
                blocking_deps: verdicts.iter().map(|v| v.dep().into()).collect(),
                widened_deps: Vec::new(),
                classification: Some("A-iteration-cap".into()),
                blocking_summary: blocking_summary.clone(),
                verdicts: verdicts.clone(),
                invariant_violations: Vec::new(),
            });
            outcome.refinement_steps = refinement_steps;
            outcome.workspace_edit_suggestions = suggestions;
            outcome.terminal_classification = Some("A-iteration-cap".into());
            return Ok(outcome);
        }

        // Widen each `WidenRetread` verdict by one level. Skip if the
        // dep is an ABI anchor (defense-in-depth -- classify_chains
        // already filtered them, but the predicate is the load-bearing
        // check and we re-assert here so any future bug that allows
        // an ABI anchor into `WidenRetread` still gets blocked).
        let mut widened_this_round: Vec<String> = Vec::new();
        for v in widenable {
            let PerChainVerdict::WidenRetread { dep, current_spec } = v else {
                continue;
            };
            // Belt and suspenders: NEVER widen an ABI anchor, even if
            // classify_chains let one through somehow.
            if is_abi_anchor(dep) {
                tracing::error!(
                    dep = %dep,
                    "iterative_solve_refinement: ABI anchor leaked into WidenRetread verdict -- refusing to widen (classify_chains bug)",
                );
                continue;
            }
            let Some(next_spec) = widen_one_level(current_spec) else {
                continue;
            };
            if next_spec == "*" {
                widened_to_star.insert(dep.clone());
            }
            effective.overrides.insert(dep.clone(), next_spec.clone());
            widened_this_round.push(format!("{dep} -> {next_spec}"));
        }
        let blocking_deps_list: Vec<String> = verdicts.iter().map(|v| v.dep().into()).collect();

        if widened_this_round.is_empty() {
            // We HAD widenable verdicts but every one of them was
            // either already at `*` or shape-unrecognized. Treat as
            // exhausted.
            refinement_steps.push(crate::audit::RefinementStep {
                iteration: iter,
                blocking_deps: blocking_deps_list,
                widened_deps: Vec::new(),
                classification: Some("A-no-widening-possible".into()),
                blocking_summary: blocking_summary.clone(),
                verdicts: verdicts.clone(),
                invariant_violations: Vec::new(),
            });
            outcome.refinement_steps = refinement_steps;
            outcome.workspace_edit_suggestions = suggestions;
            outcome.terminal_classification = Some("A-no-widening-possible".into());
            return Ok(outcome);
        }

        tracing::info!(
            iteration = iter,
            widened = ?widened_this_round,
            "iterative refinement: widening retread-emitted deps and re-solving",
        );
        // Record what this round changed so the NEXT attempt's status
        // line can show it (one line, no console blow-up).
        last_action = Some(format!(
            "widened {}",
            summarize_changes(&widened_this_round)
        ));
        // Re-emit produce_output with the updated overrides.
        let new_output =
            super::produce_output(bundle, effective, host_platform, python_version, siblings)?;
        current_run_deps = new_output
            .run_dependencies
            .depends
            .iter()
            .map(|n| {
                let raw = super::audit_report::format_packagespec(&n.spec);
                if raw.is_empty() {
                    n.name.as_str().to_string()
                } else {
                    format!("{} {raw}", n.name.as_str())
                }
            })
            .collect();

        // v0.36.0+: post-condition invariant. Build a (name, spec)
        // view of the freshly-emitted output and assert ABI anchors
        // weren't corrupted by the widening we just did.
        let emitted_pairs: Vec<(String, String)> = new_output
            .run_dependencies
            .depends
            .iter()
            .map(|n| {
                (
                    n.name.as_str().to_string(),
                    super::audit_report::format_packagespec(&n.spec),
                )
            })
            .collect();
        let invariant_violations =
            check_output_abi_invariants(&emitted_pairs, workspace_specs, &effective.overrides);
        if !invariant_violations.is_empty() {
            // Loud log + audit entry. Don't fail the cascade -- the
            // invariant is a safety-net, not a precondition. If it
            // fires we've still produced an output but the caller can
            // observe the corruption flag.
            for msg in &invariant_violations {
                tracing::error!(
                    bundle = %bundle.conda_name,
                    env = %env_name,
                    iteration = iter,
                    violation = %msg,
                    "ABI invariant violated by iterative_solve_refinement (output may be corrupt)",
                );
                // debug_assert so test runs fail-fast on regression
                // while release builds keep limping forward.
                debug_assert!(false, "ABI invariant violation: {msg}");
            }
        }

        refinement_steps.push(crate::audit::RefinementStep {
            iteration: iter,
            blocking_deps: blocking_deps_list,
            widened_deps: widened_this_round.clone(),
            classification: Some("A-retread-widenable".into()),
            blocking_summary: blocking_summary.clone(),
            verdicts: verdicts.clone(),
            invariant_violations,
        });
    }
    // Unreachable (loop returns).
    Ok(crate::solve_check::SolveOutcome::unreachable())
}

/// Map a verdict mix to a terminal-classification tag string. Mirrors
/// `class_label`'s output for back-compat with the audit + RPC error
/// pipeline; called only from `iterative_solve_refinement`'s stop-early
/// path where the verdicts are already computed.
fn derive_class_tag(verdicts: &[crate::conflict_classifier::PerChainVerdict]) -> String {
    use crate::conflict_classifier::PerChainVerdict;
    let any_workspace = verdicts
        .iter()
        .any(|v| matches!(v, PerChainVerdict::WorkspacePinDominates { .. }));
    let any_abi = verdicts
        .iter()
        .any(|v| matches!(v, PerChainVerdict::AbiAnchor { .. }));
    let any_exhausted = verdicts
        .iter()
        .any(|v| matches!(v, PerChainVerdict::AlreadyExhausted { .. }));
    let any_transitive = verdicts
        .iter()
        .any(|v| matches!(v, PerChainVerdict::TransitiveOnly { .. }));
    // Order picks the most-actionable label first.
    if any_workspace {
        "B-workspace-pin-dominates".into()
    } else if any_abi {
        // ABI-anchor-only failure: workspace probably pins the anchor
        // and retread's cascade can't help. Tag it as B so the audit
        // surfaces it under "user must edit".
        "B-workspace-pin-dominates".into()
    } else if any_exhausted {
        "A-exhausted".into()
    } else if any_transitive {
        "C-workspace-only".into()
    } else {
        "A-exhausted".into()
    }
}

/// v0.30.0 pre-emit widening pass. ALWAYS runs (regardless of relax
/// policy) so the audit captures probe outcomes for every dep this
/// bundle would emit. Mutation is policy-gated:
///
/// * `*-with-last-resort`: widen unsat specs to `*` via override
///   injection (v0.19.0 behavior, renamed from `last_resort_widen_pass`).
/// * `patch-then-minor-then-major-then-last-resort`: per-dep escalate
///   patch -> PyPI -> minor -> PyPI -> major -> PyPI -> `*`, stopping at
///   the first level that satisfies. PyPI hits add the wheel to the
///   bundle and drop the dep from conda emission.
/// * Everything else: probe + record only, no mutation.
///
/// Why we re-translate per dep rather than re-using the BFS probe
/// results: pyglet (and similar conda-routed deps that AREN'T in the
/// BFS's extras/prefix set) never get probed by the BFS. They land
/// directly at translate-time with a strict spec and the conda solver
/// then fails. This pass catches them.
pub(crate) async fn pre_emit_widen_pass(
    bundle: &mut Bundle,
    effective: &mut RetreadConfig,
    conda_channels: &[ChannelUrl],
    target: &crate::pypi::WheelTarget,
    download_dir: &Path,
    pypi_indexes: &[String],
) -> Result<()> {
    // v0.32.0+: workspace pins flow in via effective.overrides
    // (injected by apply_emission from the per-env EnvEmission's
    // transitive_overrides). The cascade still uses them at step 7;
    // keep the alias name so the existing logic doesn't need to know
    // about the rename.
    let workspace_pins: BTreeMap<String, String> = effective.overrides.clone();
    use crate::relax::{default_marker_env, translate};
    let env = default_marker_env(&target.python_version)?;
    // Names of wheels in the bundle itself. translate emissions for
    // these (e.g. an isaaclab sub-package referencing isaaclab-tasks)
    // would be self-references that the bundle satisfies internally;
    // probing them would mislead the widening.
    let bundled_names: std::collections::HashSet<String> = bundle
        .all_wheels()
        .map(|w| canonical_conda_name(&w.pypi_name))
        .collect();
    // Dedup across multiple wheels declaring the same dep -- we only
    // need to probe each (conda_name, spec) once per bundle.
    let mut seen_probes: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();
    // Walk every wheel's Requires-Dist; for each line that translates
    // to a conda spec WITH an upper bound or strict pin, probe.
    let raw_lines: Vec<String> = bundle
        .all_wheels()
        .flat_map(|w| w.metadata.requires_dist.iter().cloned())
        .collect();
    let tiered = effective.relax.has_tiered_cascade();
    let allows_mut = effective.relax.allows_widening_mutation();
    // Phase A (pure, no awaits): collect probe candidates. The skip
    // filters mirror the previous serial loop exactly.
    struct PreEmitCandidate {
        raw: String,
        pypi_name: Option<String>,
        pypi_specs: Option<VersionSpecifiers>,
        conda_name: String,
        spec: String,
    }
    let mut candidates: Vec<PreEmitCandidate> = Vec::new();
    for raw in raw_lines {
        // Capture the original PyPI name + specifiers from the raw
        // requires_dist line. Needed for tiered-cascade PyPI fallback.
        let parsed: Option<uv_pep508::Requirement> = uv_pep508::Requirement::from_str(&raw).ok();
        let pypi_name = parsed.as_ref().map(|r| r.name.to_string());
        let pypi_specs: Option<VersionSpecifiers> =
            parsed.as_ref().and_then(|r| match &r.version_or_url {
                Some(uv_pep508::VersionOrUrl::VersionSpecifier(s)) => Some(s.clone()),
                _ => None,
            });

        // Predict what translate WOULD emit at the effective policy.
        // If translate returns None (marker false / vendored / dropped),
        // skip.
        let translated = match translate(
            &raw,
            &env,
            &effective.name_map,
            &effective.overrides,
            effective.relax,
        ) {
            Ok(Some(d)) => d,
            _ => continue,
        };
        let conda_name = translated.name.clone();
        let mut spec = translated.spec.clone();
        if conda_name.is_empty() {
            continue;
        }
        // v1.5.6: BARE deps (`Requires-Dist: DracoPy`, no version
        // spec) are probed too, at the name level (`*`). The old skip
        // predates cascade step 8: back then a spec-less dep could
        // only be WIDENED (pointless -- it is already maximally wide),
        // so probing was wasted work. Step 8 changed that: a bare dep
        // with ZERO conda candidates can now be auto-bundled from PyPI
        // (latest compatible wheel), but only if it reaches the
        // cascade at all. Skipping here shipped `dracopy *` as a
        // doomed conda run-dep (genesis-world's DracoPy).
        if spec.is_empty() {
            spec = "*".to_string();
        }
        // Match the bundle's vendored wheels on EITHER namespace via
        // the shared dual-namespace helper (P2): the wheels are
        // recorded under PyPI names, but name_map may translate this
        // dep's emission name to a different conda name
        // (tinyobjloader -> tinyobjloader-python).
        if crate::relax::already_covered(&bundled_names, &conda_name, pypi_name.as_deref()) {
            continue;
        }
        if effective.overrides.contains_key(&conda_name) {
            continue;
        }
        if !seen_probes.insert((conda_name.clone(), spec.clone())) {
            continue;
        }
        candidates.push(PreEmitCandidate {
            raw,
            pypi_name,
            pypi_specs,
            conda_name,
            spec,
        });
    }

    // Phase B (v1.4.0): batch every step-1 probe through probe_many
    // (16-way bounded) instead of one serial await per dep. ~80 deps
    // per bundle previously meant ~80 sequential awaits here.
    // buffer_unordered yields in completion order, so results are
    // re-keyed by (package, spec) -- unique per candidate thanks to
    // the seen_probes dedup above.
    let pairs: Vec<(String, String)> = candidates
        .iter()
        .map(|c| (c.conda_name.clone(), c.spec.clone()))
        .collect();
    let mut probes_by_key: std::collections::HashMap<(String, String), crate::probe::ProbeResult> =
        crate::probe::probe_many(conda_channels, pairs, Some(&target.python_version))
            .await
            .into_iter()
            .map(|r| ((r.package.clone(), r.spec.clone()), r))
            .collect();

    // Phase C: record decisions + run cascades serially in candidate
    // order. Cascades mutate bundle/effective, so they stay serial;
    // the fresh overrides re-check below preserves the old serial
    // semantics where an earlier candidate's injected override skips
    // later candidates of the same dep.
    for c in candidates {
        let PreEmitCandidate {
            raw,
            pypi_name,
            pypi_specs,
            conda_name,
            spec,
        } = c;
        if effective.overrides.contains_key(&conda_name) {
            continue;
        }
        let Some(strict_probe) = probes_by_key.remove(&(conda_name.clone(), spec.clone())) else {
            tracing::debug!(
                dep = %conda_name,
                spec = %spec,
                "pre-emit widen: probe result missing from batch; skipping",
            );
            continue;
        };
        let stage1 = if tiered {
            "tiered-cascade-step1-conda"
        } else {
            "pre-emit-widen-strict"
        };
        let initial_routing = if strict_probe.is_satisfied() {
            "satisfied"
        } else if strict_probe.is_definitively_unsatisfied() {
            "unsat"
        } else {
            "indecisive"
        };
        bundle
            .probe_decisions
            .push(crate::audit::ProbeDecision::from_probe(
                stage1,
                &pypi_name.clone().unwrap_or_else(|| conda_name.clone()),
                &conda_name,
                &spec,
                &target.python_version,
                &strict_probe,
                initial_routing,
            ));
        if strict_probe.is_satisfied() || !strict_probe.is_definitively_unsatisfied() {
            continue;
        }
        // Strict spec is definitively unsat. Mutation requires policy opt-in.
        if !allows_mut {
            continue;
        }

        if tiered {
            tiered_cascade_for_dep(
                bundle,
                effective,
                conda_channels,
                target,
                download_dir,
                pypi_indexes,
                &raw,
                &env,
                pypi_name.as_deref().unwrap_or(&conda_name),
                pypi_specs.as_ref(),
                &conda_name,
                &workspace_pins,
            )
            .await?;
        } else {
            // `*-with-last-resort`: prefer the workspace's `[dependencies]`
            // pin if one exists (audit clarity + workspace wins anyway);
            // else probe `*` and inject if conda has any py-compatible
            // build of the package.
            let (target_spec, source_tag) = match workspace_pins.get(&conda_name) {
                Some(pin) => (pin.clone(), "from-workspace-pin"),
                None => ("*".to_string(), "any-version"),
            };
            let probe_result = crate::probe::probe(
                conda_channels,
                &conda_name,
                &target_spec,
                Some(&target.python_version),
            )
            .await;
            let widened = probe_result.is_satisfied();
            let routing_decision = if widened {
                if source_tag == "from-workspace-pin" {
                    "widened-to-workspace-pin"
                } else {
                    "widened-to-any-version"
                }
            } else {
                "no-py-compat-version-on-conda"
            };
            bundle
                .probe_decisions
                .push(crate::audit::ProbeDecision::from_probe(
                    "last-resort-widen",
                    &conda_name,
                    &conda_name,
                    &target_spec,
                    &target.python_version,
                    &probe_result,
                    routing_decision,
                ));
            if widened {
                tracing::info!(
                    dep = %conda_name,
                    strict_spec = %spec,
                    widened_to = %target_spec,
                    source = %source_tag,
                    "last-resort-widen: conda satisfies widened spec; injecting override",
                );
                effective.overrides.insert(conda_name, target_spec);
            } else {
                tracing::warn!(
                    dep = %conda_name,
                    strict_spec = %spec,
                    "last-resort-widen: conda has ZERO py-compat builds; consider retread-drop-deps + post-install pip, or use the patch-then-minor-then-major-then-last-resort policy for automatic PyPI fallback.",
                );
            }
        }
    }
    Ok(())
}

/// Bundle group for a `[retread-wheels]` entry: the per-entry `bundle`
/// field wins, then the pack-wide `retread-bundle` default (v1.4.0),
/// then standalone (the entry's own name -- one conda output per
/// entry, the historical behavior).
pub(crate) fn bundle_group_for(
    entry_name: &str,
    entry: &WheelEntry,
    default_bundle: Option<&str>,
) -> String {
    entry
        .bundle
        .clone()
        .or_else(|| default_bundle.map(String::from))
        .unwrap_or_else(|| entry_name.to_string())
}

/// Index fallback chain for the cascade's PyPI bundling steps, tried
/// in order: every distinct non-URL `[retread-wheels]` entry index
/// (private indexes like pypi.nvidia.com first, in declaration order),
/// then every index the consumer workspace declares under
/// `[pypi-options]` / `[feature.X.pypi-options]` (`index-url` +
/// `extra-index-urls`), then public PyPI. Nothing here is
/// vendor-specific -- any manifest-declared index participates.
/// Mirrors (and extends) the chain `auto_bundle_transitives` builds.
pub(crate) fn pypi_fallback_indexes(
    config: &RetreadConfig,
    workspace: Option<&crate::workspace::WorkspaceManifest>,
) -> Vec<String> {
    // Primary: every entry index (non-URL entries only, since URL
    // entries don't have an associated PyPI index).
    let primary = config
        .retread_wheels
        .values()
        .filter(|e| e.url.is_none())
        .map(|e| e.index_url());
    // Extra: workspace [pypi-options] indexes (if any).
    let extra: Vec<String> = workspace
        .map(|ws| ws.all_pypi_index_urls())
        .unwrap_or_default();
    super::merge_index_chain(primary, &extra)
}

/// Resolve + fetch `pypi_name` from each index in turn; on the first
/// success, vendor the wheel into the bundle and drop the dep from
/// conda emission (push to `effective.drop_deps`, which `translate`
/// consults). Records one ProbeDecision per resolve attempt. Returns
/// true when a wheel was bundled.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn try_pypi_bundle(
    bundle: &mut Bundle,
    effective: &mut RetreadConfig,
    target: &crate::pypi::WheelTarget,
    download_dir: &Path,
    pypi_indexes: &[String],
    pypi_name: &str,
    conda_name: &str,
    specs: &VersionSpecifiers,
    stage: &str,
    success_routing: &str,
) -> bool {
    for index in pypi_indexes {
        match pypi::resolve(index, pypi_name, specs, target).await {
            Ok(resolved) => match metadata_preferring_sidecar(&resolved, download_dir).await {
                Ok(metadata) => {
                    bundle.probe_decisions.push(crate::audit::ProbeDecision {
                        stage: stage.into(),
                        pypi_name: pypi_name.to_string(),
                        conda_name: conda_name.to_string(),
                        spec: specs.to_string(),
                        target_python: target.python_version.clone(),
                        channels_consulted: vec![index.clone()],
                        satisfiable: Some(true),
                        matching_candidates: 1,
                        routing_decision: success_routing.into(),
                    });
                    tracing::info!(
                        dep = %pypi_name,
                        index = %index,
                        url = %resolved.url,
                        "tiered-cascade: PyPI fallback bundled wheel; dropping conda emit",
                    );
                    bundle.extras.push(ResolvedWheel {
                        pypi_name: canonical_conda_name(pypi_name),
                        url: resolved.url,
                        metadata,
                        extras_requested: vec![],
                        auto_data: None,
                        auto_data_dedup_skipped_root: None,
                    });
                    effective.drop_deps.push(pypi_name.to_string());
                    return true;
                }
                Err(e) => {
                    tracing::debug!(
                        dep = %pypi_name,
                        index = %index,
                        error = %format!("{e:#}"),
                        "tiered-cascade: PyPI fetch failed; trying next index",
                    );
                }
            },
            Err(e) => {
                bundle.probe_decisions.push(crate::audit::ProbeDecision {
                    stage: stage.into(),
                    pypi_name: pypi_name.to_string(),
                    conda_name: conda_name.to_string(),
                    spec: specs.to_string(),
                    target_python: target.python_version.clone(),
                    channels_consulted: vec![index.clone()],
                    satisfiable: Some(false),
                    matching_candidates: 0,
                    routing_decision: "pypi-resolve-failed".into(),
                });
                tracing::debug!(
                    dep = %pypi_name,
                    index = %index,
                    error = %format!("{e:#}"),
                    "tiered-cascade: PyPI resolve failed on this index; trying next",
                );
            }
        }
    }
    false
}

/// v0.30.0 tiered cascade: per-dep escalate widening levels with PyPI
/// fallback between them. Called only when the policy is
/// `patch-then-minor-then-major-then-last-resort` AND step 1 (base
/// patch probe) already came back definitively unsat.
///
/// Cascade steps (step 1 already attempted by caller):
///   2. PyPI at patch range -> bundle + drop conda emit
///   3. conda at minor widening
///   4. PyPI at minor range -> bundle + drop conda emit
///   5. conda at major widening
///   6. PyPI at major range -> bundle + drop conda emit
///   7. widen conda emit to `*`
///   8. (v1.3.0) conda DEFINITIVELY has zero py-compat builds at any
///      version: bundle from PyPI at any version -> bundle + drop
///      conda emit. Leaving the dep on the conda side is guaranteed
///      to fail the solve, so auto-rerouting cannot make things worse
///      and is never silent (warn + ProbeDecision). Skipped when the
///      user forced the conda side via `retread-conda-deps` or when
///      the probe was indecisive (channel fetch failure must not
///      reroute deps).
///
/// "Drop conda emit" is done by pushing the PyPI name into
/// `effective.drop_deps`, which `translate` consults to skip emission.
/// The added wheel lives in `bundle.extras` and gets pip-installed at
/// build time.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn tiered_cascade_for_dep(
    bundle: &mut Bundle,
    effective: &mut RetreadConfig,
    conda_channels: &[ChannelUrl],
    target: &crate::pypi::WheelTarget,
    download_dir: &Path,
    pypi_indexes: &[String],
    raw: &str,
    env: &uv_pep508::MarkerEnvironment,
    pypi_name: &str,
    pypi_specs: Option<&VersionSpecifiers>,
    conda_name: &str,
    workspace_pins: &std::collections::BTreeMap<String, String>,
) -> Result<()> {
    use crate::relax::translate;
    use std::collections::BTreeMap;

    let empty_overrides: BTreeMap<String, String> = BTreeMap::new();

    // Steps 2/4/6: each level's PyPI fallback resolves the original
    // PyPI requirement and bundles the wheel. The original `pypi_specs`
    // already encodes the upstream version range; we don't widen the
    // PyPI search range at each conda-widening level because PyPI's
    // job is just "give me a compatible wheel for the upstream pin",
    // and the conda widening is independent of that.
    for (level_idx, level_policy) in [
        (0usize, RelaxPolicy::Patch),
        (1, RelaxPolicy::Minor),
        (2, RelaxPolicy::Major),
    ]
    .into_iter()
    {
        // PyPI fallback happens AFTER the conda probe at each level
        // EXCEPT level 0 (Patch), where the caller already did the
        // conda probe. So the sequence becomes:
        //   level 0: caller did conda (failed) -> step 2 PyPI here
        //   level 1: step 3 conda, step 4 PyPI
        //   level 2: step 5 conda, step 6 PyPI

        // For levels 1 and 2: re-translate at this widening level and
        // probe conda.
        if level_idx > 0 {
            let widened_spec = match translate(
                raw,
                env,
                &effective.name_map,
                &empty_overrides,
                level_policy,
            ) {
                Ok(Some(d)) => d.spec,
                _ => continue,
            };
            if widened_spec.is_empty() {
                continue;
            }
            let conda_stage = match level_idx {
                1 => "tiered-cascade-step3-conda",
                2 => "tiered-cascade-step5-conda",
                _ => unreachable!(),
            };
            let probe = crate::probe::probe(
                conda_channels,
                conda_name,
                &widened_spec,
                Some(&target.python_version),
            )
            .await;
            let routing = if probe.is_satisfied() {
                "satisfied-widened"
            } else if probe.is_definitively_unsatisfied() {
                "unsat"
            } else {
                "indecisive"
            };
            bundle
                .probe_decisions
                .push(crate::audit::ProbeDecision::from_probe(
                    conda_stage,
                    pypi_name,
                    conda_name,
                    &widened_spec,
                    &target.python_version,
                    &probe,
                    routing,
                ));
            if probe.is_satisfied() {
                tracing::info!(
                    dep = %conda_name,
                    widened_spec = %widened_spec,
                    level = ?level_policy,
                    "tiered-cascade: conda satisfied at widened level; injecting override",
                );
                effective
                    .overrides
                    .insert(conda_name.to_string(), widened_spec);
                return Ok(());
            }
            if !probe.is_definitively_unsatisfied() {
                // Indecisive at this level -- don't escalate further,
                // don't widen optimistically. Bail out of the cascade.
                return Ok(());
            }
        }

        // PyPI fallback at this level.
        let pypi_stage = match level_idx {
            0 => "tiered-cascade-step2-pypi",
            1 => "tiered-cascade-step4-pypi",
            2 => "tiered-cascade-step6-pypi",
            _ => unreachable!(),
        };
        if let Some(specs) = pypi_specs {
            if try_pypi_bundle(
                bundle,
                effective,
                target,
                download_dir,
                pypi_indexes,
                pypi_name,
                conda_name,
                specs,
                pypi_stage,
                "pypi-bundled-dropping-conda-emit",
            )
            .await
            {
                return Ok(());
            }
        } else {
            // No version specifiers (URL/git/bare dep). PyPI fallback
            // doesn't apply; just record + advance.
            bundle.probe_decisions.push(crate::audit::ProbeDecision {
                stage: pypi_stage.into(),
                pypi_name: pypi_name.to_string(),
                conda_name: conda_name.to_string(),
                spec: String::new(),
                target_python: target.python_version.clone(),
                channels_consulted: vec![],
                satisfiable: None,
                matching_candidates: 0,
                routing_decision: "no-pypi-specs-skipped".into(),
            });
        }
    }

    // Step 7: last resort. Prefer the workspace's `[dependencies]`
    // pin (if any) over `*` so retread's emission mirrors the
    // workspace's source of truth -- the audit shows the real spec
    // instead of an opaque wildcard, and the conda solver picks the
    // same version it would have anyway. Fall through to `*` if the
    // workspace doesn't pin this dep or its pin doesn't probe-satisfy.
    let (target_spec, source_tag) = match workspace_pins.get(conda_name) {
        Some(pin) => (pin.clone(), "from-workspace-pin"),
        None => ("*".to_string(), "any-version"),
    };
    let probe_result = crate::probe::probe(
        conda_channels,
        conda_name,
        &target_spec,
        Some(&target.python_version),
    )
    .await;
    let widened = probe_result.is_satisfied();
    bundle
        .probe_decisions
        .push(crate::audit::ProbeDecision::from_probe(
            "tiered-cascade-step7-last-resort",
            pypi_name,
            conda_name,
            &target_spec,
            &target.python_version,
            &probe_result,
            if widened {
                if source_tag == "from-workspace-pin" {
                    "widened-to-workspace-pin"
                } else {
                    "widened-to-any-version"
                }
            } else {
                "no-py-compat-version-on-conda"
            },
        ));
    if widened {
        tracing::info!(
            dep = %conda_name,
            widened_to = %target_spec,
            source = %source_tag,
            "tiered-cascade: last-resort widening; injecting override",
        );
        effective
            .overrides
            .insert(conda_name.to_string(), target_spec);
        return Ok(());
    }

    // Establish the name-level result: can conda provide ANY py-compat
    // version of this package? When there was no workspace pin,
    // `probe_result` above already probed `*`; with a pin that didn't
    // probe-satisfy, fall back to `*` (and inject it if satisfied) the
    // way pre-v1.3.0 did.
    let name_level_probe = if source_tag == "from-workspace-pin" {
        let any_probe = crate::probe::probe(
            conda_channels,
            conda_name,
            "*",
            Some(&target.python_version),
        )
        .await;
        if any_probe.is_satisfied() {
            tracing::info!(
                dep = %conda_name,
                workspace_pin = %target_spec,
                "tiered-cascade: workspace pin didn't probe-satisfy; falling through to `*`",
            );
            bundle
                .probe_decisions
                .push(crate::audit::ProbeDecision::from_probe(
                    "tiered-cascade-step7-last-resort",
                    pypi_name,
                    conda_name,
                    "*",
                    &target.python_version,
                    &any_probe,
                    "widened-to-any-version-after-workspace-pin-miss",
                ));
            effective
                .overrides
                .insert(conda_name.to_string(), "*".into());
            return Ok(());
        }
        any_probe
    } else {
        probe_result
    };

    // Step 8: conda definitively has zero py-compat builds of this
    // package at any version. A conda run-dep is guaranteed to fail the
    // solve, so bundle the wheel from PyPI instead -- unless the user
    // explicitly forced the conda side via retread-conda-deps, or the
    // probe was indecisive (a channel fetch failure must not silently
    // reroute deps that conda may well have).
    // retread-conda-deps entries are PyPI names by documented
    // convention, but accept either namespace: the dep's emission name
    // may differ from its PyPI name via name_map.
    let user_forced_conda = effective.conda_deps.iter().any(|n| {
        let norm = canonical_conda_name(n);
        norm == conda_name || norm == canonical_conda_name(pypi_name)
    });
    if name_level_probe.is_definitively_unsatisfied() && !user_forced_conda {
        let any_specs = pypi_specs.cloned().unwrap_or_else(VersionSpecifiers::empty);
        if try_pypi_bundle(
            bundle,
            effective,
            target,
            download_dir,
            pypi_indexes,
            pypi_name,
            conda_name,
            &any_specs,
            "tiered-cascade-step8-pypi-last-resort",
            "auto-pypi-no-conda-candidates",
        )
        .await
        {
            tracing::warn!(
                dep = %conda_name,
                pypi = %pypi_name,
                "tiered-cascade: conda has no py-compat build at any version; auto-bundled the wheel from PyPI and dropped the conda emission",
            );
            return Ok(());
        }
    }
    tracing::warn!(
        dep = %conda_name,
        "tiered-cascade: every step exhausted; conda has no py-compat build at any version and PyPI bundling did not produce a wheel. Solve will fail. Consider retread-drop-deps.",
    );
    Ok(())
}
