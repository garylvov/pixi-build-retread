//! Audit construction, probe-trace writing, and solve-failure reporting.
//!
//! Extracted from handler.rs (Phase 0b.2). All functions are behavior-
//! identical whole-function moves; no logic changes.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use pixi_build_types::procedures::conda_outputs::CondaOutput;
use pixi_build_types::{BinaryPackageSpec, PackageSpec};
use rattler_conda_types::ChannelUrl;

use crate::config::RelaxPolicy;

use super::Bundle;

/// Construct a [`crate::audit::BundleAudit`] from the bundle we're
/// about to ship to rattler-build and the generated recipe. Strictly
/// informational; the resulting JSON lands next to recipe.yaml so the
/// user can see exactly which upstream `Requires-Dist:` lines became
/// which conda run-deps, and copy-paste the rendered TOML blocks into
/// their workspace if they want to mirror the bundle exactly.
pub(crate) fn build_bundle_audit(
    bundle: &Bundle,
    recipe: &crate::recipe::Recipe,
) -> crate::audit::BundleAudit {
    let wheels = bundle
        .all_wheels()
        .map(|w| crate::audit::WheelAudit {
            name: w.metadata.name.clone(),
            version: w.metadata.version.clone(),
            requires_dist: w.metadata.requires_dist.clone(),
            extras: w.extras_requested.clone(),
            auto_data: w.auto_data.clone(),
            auto_data_dedup_skipped_root: w.auto_data_dedup_skipped_root.clone(),
        })
        .collect();
    let emitted_run_deps = recipe
        .requirements
        .run
        .iter()
        .map(|spec| {
            // Recipe's run list is `Vec<String>` of "<name> <constraint>"
            // strings; split into (name, full-spec) so the audit consumer
            // can render the conda-deps TOML block from constraint side
            // alone.
            let name = spec.split_whitespace().next().unwrap_or(spec).to_string();
            crate::audit::EmittedDep {
                name,
                spec: spec.clone(),
            }
        })
        .collect();
    crate::audit::BundleAudit::new(
        bundle.conda_name.clone(),
        bundle.primary.metadata.version.clone(),
        wheels,
        emitted_run_deps,
        bundle.probe_decisions.clone(),
        bundle.solve_diagnostics.clone(),
    )
}

/// v0.14.1+: dump just the bundle's probe decisions to a side file
/// v0.23.0+ POST-EMIT widening: probe every run-dep that produce_output
/// emitted, widen any that the workspace's conda channels can't satisfy.
///
/// Runs in `conda_outputs` AFTER produce_output produces the
/// CondaOutput. Walks `output.run_dependencies.depends`, calls
/// `probe::probe(name, spec, python)` for each non-python spec. For
/// any spec that's definitively unsatisfied, mutates the spec in
/// place to `*` (the "widen to any version" fallback) AND records a
/// ProbeDecision under stage `"post-emit-widen"` so the audit shows
/// what got changed.
///
/// Why this is in addition to (and arguably instead of) the pre-emit
/// `last_resort_widen_pass`: the pre-emit cascade re-translates each
/// wheel's `requires_dist` and probes the result, but produce_output's
/// emit path also applies the vendored/dropped/cross-output filters
/// and dedups by first-spec-wins -- so what cascade predicts can
/// diverge from what produce_output actually emits. This pass probes
/// the ACTUAL output, eliminating that drift.
pub(crate) async fn post_emit_widen_pass(
    output: &mut CondaOutput,
    conda_channels: &[ChannelUrl],
    target_python: &str,
    policy: RelaxPolicy,
    decisions: &mut Vec<crate::audit::ProbeDecision>,
) -> Result<()> {
    let allows_mut = policy.allows_widening_mutation();
    // v1.4.0: batch the per-dep probes (16-way bounded) instead of one
    // serial await per emitted run-dep (~80 per bundle). Pairs are
    // deduped because probe_many yields in completion order and
    // results are re-keyed by (package, spec).
    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut seen_pairs: HashSet<(String, String)> = HashSet::new();
    for spec in output.run_dependencies.depends.iter() {
        let name_str = spec.name.as_str().to_string();
        if name_str == "python" {
            continue;
        }
        let spec_str = format_packagespec(&spec.spec);
        if spec_str.is_empty() || spec_str == "*" {
            continue;
        }
        if seen_pairs.insert((name_str.clone(), spec_str.clone())) {
            pairs.push((name_str, spec_str));
        }
    }
    let probes_by_key: std::collections::HashMap<(String, String), crate::probe::ProbeResult> =
        crate::probe::probe_many(conda_channels, pairs, Some(target_python))
            .await
            .into_iter()
            .map(|r| ((r.package.clone(), r.spec.clone()), r))
            .collect();

    for spec in output.run_dependencies.depends.iter_mut() {
        let name_str = spec.name.as_str().to_string();
        if name_str == "python" {
            continue;
        }
        let spec_str = format_packagespec(&spec.spec);
        if spec_str.is_empty() || spec_str == "*" {
            continue;
        }
        let Some(probe_result) = probes_by_key.get(&(name_str.clone(), spec_str.clone())) else {
            continue;
        };
        let routing_decision = if probe_result.is_definitively_unsatisfied() {
            if allows_mut {
                "widened-to-any-version"
            } else {
                "unsat-no-mutation"
            }
        } else if probe_result.is_satisfied() {
            "no-widening-needed"
        } else {
            "skipped-indecisive"
        };
        decisions.push(crate::audit::ProbeDecision {
            stage: "post-emit-widen".into(),
            pypi_name: name_str.clone(),
            conda_name: name_str.clone(),
            spec: spec_str.clone(),
            target_python: target_python.to_string(),
            channels_consulted: probe_result.channels_consulted.clone(),
            satisfiable: probe_result.satisfiable,
            matching_candidates: probe_result.matching_candidates,
            routing_decision: routing_decision.into(),
        });
        if probe_result.is_definitively_unsatisfied() && allows_mut {
            tracing::info!(
                dep = %name_str,
                strict_spec = %spec_str,
                "post-emit-widen: emitted spec is unsat on conda channels; rewriting to `*`",
            );
            spec.spec = wildcard_packagespec();
        }
    }
    Ok(())
}

/// Render a PackageSpec to its conda match-spec string form (just the
/// version-constraint half, name not included).
pub(crate) fn format_packagespec(spec: &PackageSpec) -> String {
    match spec {
        PackageSpec::Binary(b) => b
            .version
            .as_ref()
            .map(|v| v.to_string())
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// Build a wildcard ("any version") PackageSpec for the post-emit
/// widening fallback.
pub(crate) fn wildcard_packagespec() -> PackageSpec {
    use rattler_conda_types::{ParseStrictness, VersionSpec};
    PackageSpec::Binary(BinaryPackageSpec {
        version: Some(
            VersionSpec::from_str("*", ParseStrictness::Lenient)
                .expect("'*' is always a valid VersionSpec"),
        ),
        ..Default::default()
    })
}

/// next to the source-package pixi.toml. Always-on; survives a failed
/// conda solve. The full audit (a superset) still gets written at
/// conda/build_v1 time. Filename matches the audit convention:
/// retread-probe-trace-<conda_name>.json.
/// v0.34.2+: write a sticky human-readable summary of any UNSAT
/// solve-check outcomes to `RETREAD-SOLVE-FAILED-<bundle>.md` in the
/// source package's dir. pixi's progress spinner overwrites stderr
/// lines so the in-process banner doesn't survive to the user's
/// terminal -- a file does. Skipped silently when every env is SAT
/// (no failure = no summary to write).
pub(crate) async fn write_solve_failed_summary(bundle: &Bundle, source_dir: &Path) -> Result<()> {
    // Skipped checks are abstentions ("unknown"), not failures.
    let any_unsat = bundle
        .solve_diagnostics
        .values()
        .any(|d| !d.satisfiable && !d.skipped);
    let path = source_dir.join(format!("RETREAD-SOLVE-FAILED-{}.md", bundle.conda_name));
    if !any_unsat {
        // Remove a stale file from a previous failed run; clean state.
        let _ = tokio::fs::remove_file(&path).await;
        return Ok(());
    }
    let mut out = String::new();
    out.push_str(&format!(
        "# retread solve check: UNSAT for output `{}`\n\n",
        bundle.conda_name,
    ));

    // v0.35.0+: surface the classifier's actionable suggestions FIRST
    // so the user sees what to do without scrolling past 200 lines of
    // solver enumeration. Collect across all unsat envs and dedup.
    let mut all_suggestions: Vec<&crate::conflict_classifier::WorkspaceEditSuggestion> = Vec::new();
    let mut seen_sugs: HashSet<(String, String)> = HashSet::new();
    for diag in bundle.solve_diagnostics.values() {
        for sug in &diag.workspace_edit_suggestions {
            let key = (sug.env.clone(), sug.current_pin.clone());
            if seen_sugs.insert(key) {
                all_suggestions.push(sug);
            }
        }
    }
    if !all_suggestions.is_empty() {
        out.push_str("## Suggested workspace edits\n\n");
        out.push_str(
            "The conflict is in YOUR workspace pixi.toml -- retread can't \
             widen these for you. Pick one of the suggestions below \
             (or relax differently if you have a preference).\n\n",
        );
        for sug in &all_suggestions {
            let feature_label = sug
                .feature
                .as_deref()
                .map(|f| format!("[feature.{f}.dependencies]"))
                .unwrap_or_else(|| "[dependencies]".to_string());
            out.push_str(&format!(
                "- **env `{}` / `{}`**: change `{}` to `{}`\n  reason: {}\n",
                sug.env, feature_label, sug.current_pin, sug.suggested_pin, sug.reason,
            ));
        }
        out.push('\n');
    } else {
        // v0.36.2+: when no actionable suggestion exists, synthesize
        // a "cascade exhausted" headline that names WHICH retread
        // emission ran out of widening room against WHICH workspace
        // pin. Otherwise the file's top section is empty and the
        // user has to scroll past 200 lines of solver enumeration
        // to find the (still-present) per-env classification.
        out.push_str("## Cascade exhausted — no auto-suggestion\n\n");
        out.push_str(
            "Every failing env hit the iteration cap or ran out of \
             widenable deps. retread can't suggest a single workspace \
             edit because the conflict involves multiple workspace \
             pins co-blocking each other (typical pattern: workspace \
             pins `pytorch-gpu` AND wheel-emitted `torchaudio` needs \
             a different pytorch).\n\n\
             ### What to look at\n\n\
             1. **Per-env classification** below: which envs are \
                stuck and at what verdict.\n\
             2. **Refinement steps per env**: the cascade's trace \
                shows which retread-emitted dep was being widened \
                in each round + what workspace pins co-blocked.\n\
             3. **Final unsat chain (verbatim from rattler solver)**: \
                the actual conflict graph -- the FIRST `cannot be \
                installed` entry per env is the genuine blocker; \
                everything labeled `can be installed with any of the \
                following options` is context, not the cause.\n\n\
             ### Common fixes\n\n\
             - Bump the workspace pin that's blocking (`pytorch-gpu` \
                in [feature.gpu.dependencies] is the usual suspect \
                for isaacsim envs).\n\
             - Move conflicting deps out of `[feature.X.dependencies]` \
                and into `[feature.X.pypi-dependencies]` so uv \
                handles them instead of conda.\n\
             - Add the offending dep to `retread-drop-deps` in the \
                source package's pixi.toml so retread stops emitting \
                it as a conda dep.\n\n",
        );
    }

    // Class-by-class summary so the user knows whether the cascade is
    // still useful or completely stuck.
    out.push_str("## Per-env classification\n\n");
    let mut envs: Vec<&String> = bundle
        .solve_diagnostics
        .iter()
        .filter(|(_, d)| !d.satisfiable)
        .map(|(k, _)| k)
        .collect();
    envs.sort();
    for env in &envs {
        let diag = &bundle.solve_diagnostics[*env];
        let class = diag
            .terminal_classification
            .as_deref()
            .unwrap_or("unclassified");
        out.push_str(&format!("- `{env}`: **{class}**\n"));
    }
    out.push('\n');

    out.push_str(
        "Class meanings:\n\
         - `A-retread-widenable`: retread cascade can widen its emission; should self-resolve next iteration.\n\
         - `A-exhausted`: cascade widened blockers to `*`; conflict is via a transitive retread can't touch.\n\
         - `A-iteration-cap` / `A-no-widening-possible`: cascade gave up; usually means a workspace pin is the floor.\n\
         - `B-workspace-pin-dominates`: workspace pins the conflicting dep -- edit the workspace (see suggestions above).\n\
         - `C-workspace-only`: blocking dep isn't declared by retread or workspace; likely a transitive bubbled up.\n\n",
    );

    for env in &envs {
        let diag = &bundle.solve_diagnostics[*env];
        out.push_str(&format!("## env `{env}` — full detail\n\n"));
        if !diag.refinement_steps.is_empty() {
            out.push_str("### refinement attempted\n\n");
            for s in &diag.refinement_steps {
                out.push_str(&format!(
                    "- round {}: class={:?} blocking={:?} widened={:?}\n  summary: {}\n",
                    s.iteration,
                    s.classification.as_deref().unwrap_or(""),
                    s.blocking_deps,
                    s.widened_deps,
                    s.blocking_summary,
                ));
            }
            out.push('\n');
        }
        out.push_str("### final unsat chain (verbatim from rattler solver)\n\n```\n");
        for r in &diag.unsat_explanations {
            out.push_str(r);
            out.push('\n');
        }
        out.push_str("```\n\n");
    }
    out.push_str(
        "## Reading this file\n\n\
         - **Top section** is the actionable answer -- if it's present, edit the workspace as shown.\n\
         - **Per-env classification** tells you which envs are retread's responsibility vs yours.\n\
         - The verbatim unsat chain is the rattler solver's raw output; pixi's terminal error often picks a misleading leaf from this.\n\
         - The machine-readable form lives at `retread-probe-trace-<bundle>.json.solve_diagnostics`.\n",
    );
    tokio::fs::write(&path, out)
        .await
        .with_context(|| format!("writing solve-failed summary to {}", path.display()))?;
    tracing::error!(
        bundle = %bundle.conda_name,
        path = %path.display(),
        "WROTE RETREAD-SOLVE-FAILED summary -- see this file for the real conflict chain",
    );
    Ok(())
}

pub(crate) async fn write_probe_trace(bundle: &Bundle, source_dir: &Path) -> Result<()> {
    let path = source_dir.join(format!("retread-probe-trace-{}.json", bundle.conda_name,));
    #[derive(serde::Serialize)]
    struct Trace<'a> {
        conda_name: &'a str,
        retread_version: &'static str,
        probe_decisions: &'a [crate::audit::ProbeDecision],
        /// v0.33.5+: per-env solve diagnostics. Map keyed by env
        /// name; each entry is one env's view of (its channels +
        /// its deps + retread's emission). Surfaces here on the
        /// probe trace (which always lands) instead of only on the
        /// audit (only lands at conda/build_v1).
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        solve_diagnostics: &'a BTreeMap<String, crate::audit::SolveDiagnostics>,
    }
    let trace = Trace {
        conda_name: &bundle.conda_name,
        retread_version: env!("CARGO_PKG_VERSION"),
        probe_decisions: &bundle.probe_decisions,
        solve_diagnostics: &bundle.solve_diagnostics,
    };
    let bytes = serde_json::to_vec_pretty(&trace)?;
    tokio::fs::write(&path, &bytes)
        .await
        .with_context(|| format!("writing probe trace to {}", path.display()))?;
    tracing::info!(
        bundle = %bundle.conda_name,
        decisions = bundle.probe_decisions.len(),
        path = %path.display(),
        "wrote probe trace",
    );
    Ok(())
}
