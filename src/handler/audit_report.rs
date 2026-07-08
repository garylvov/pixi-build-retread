//! Audit construction and probe-trace writing.
//!
//! Extracted from handler.rs (Phase 0b.2). All functions are behavior-
//! identical whole-function moves; no logic changes.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
#[cfg(test)]
use pixi_build_types::PackageSpec;

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
    // P1: derive envs_skipped from solve_diagnostics so the audit field
    // is populated even when build_bundle_audit is called from
    // conda_build_v1 (which doesn't have the coordinator counter in
    // scope). This is a pure derivation from already-stored state, so
    // it cannot diverge from the coordinator's counter.
    let envs_skipped = bundle
        .solve_diagnostics
        .values()
        .filter(|d| d.skipped)
        .count();
    crate::audit::BundleAudit::new(
        bundle.conda_name.clone(),
        bundle.primary.metadata.version.clone(),
        wheels,
        emitted_run_deps,
        bundle.probe_decisions.clone(),
        bundle.solve_diagnostics.clone(),
        envs_skipped,
    )
}


/// Render a PackageSpec to its conda match-spec string form (just the
/// version-constraint half, name not included).
#[cfg(test)]
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

/// v0.14.1+: dump the bundle's probe decisions to a side file next to
/// the source-package pixi.toml. Always-on; survives a failed build.
/// The full audit (a superset) still gets written at conda/build_v1
/// time. Filename matches the audit convention:
/// retread-probe-trace-<conda_name>.json.
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
