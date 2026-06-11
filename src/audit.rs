//! Audit record of what retread saw and what it emitted, per bundle.
//!
//! Serialized as `retread-audit.json` next to the generated
//! `recipe.yaml`. Two purposes:
//!
//! 1. Debug: when the conda solve fails on a transitive pin you didn't
//!    write, grep the audit for the offending package name and you'll
//!    see exactly which bundled wheel pulled it in, what the original
//!    `Requires-Dist:` line said, and what retread emitted to conda.
//!
//! 2. Manual mirroring: the `pixi_toml_blocks` field renders ready-to-
//!    paste TOML for the workspace's `[dependencies]` and
//!    `[feature.X.pypi-options.dependency-overrides]` tables. Useful
//!    when you want to force the workspace's PyPI side (uv) to match
//!    retread's bundled versions exactly -- the same pattern your
//!    `[feature.isaaclab.pypi-options.dependency-overrides]` block
//!    uses today, but auto-derived from the bundle instead of
//!    hand-maintained.
//!
//! The audit is purely informational; nothing else in retread reads
//! it back. Removing the file or skipping its write has no effect on
//! the build.

use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// `serde` skip-serializing predicate: omit the field when it is zero.
/// Used by `BundleAudit::envs_skipped` so old audit JSON round-trips
/// cleanly (absent field defaults to 0 via `#[serde(default)]`).
fn is_zero(n: &usize) -> bool {
    *n == 0
}

/// v0.12.0+: per-wheel record of the auto-data-files inject phase
/// (phase 1.6). Surfaces which wheel carried the upstream checkout-
/// root tree as `.data/data/lib/<rel>` entries (lands at
/// `$CONDA_PREFIX/lib/<rel>` after install), and which sibling wheels
/// were dedup'd out because they share the same checkout. Purely
/// informational -- grep this to find the .kit files / repo-root data
/// without unpacking the .conda.
#[derive(Debug, Clone, Serialize)]
pub struct AutoDataReport {
    /// Repo on disk that was walked (parent of the entry's subdirectory
    /// for git/named-git entries; absent for path/url/spec entries).
    pub checkout_root: PathBuf,
    /// Number of files emitted as `<dist>-<ver>.data/data/lib/<rel>`.
    pub file_count: usize,
    /// Subdirectories of the checkout that were skipped because
    /// sibling entries in this bundle already shipped them as wheels.
    pub skip_subdirs: Vec<PathBuf>,
}

/// One wheel's pre-D view: the literal `Requires-Dist:` lines as they
/// appear in upstream's wheel METADATA (before any D rewriting or
/// translate).
#[derive(Debug, Serialize)]
pub struct WheelAudit {
    pub name: String,
    pub version: String,
    pub requires_dist: Vec<String>,
    /// v0.12.0+: extras the user requested on this entry (any form --
    /// PyPI, git, path, named-git). Empty when none requested.
    /// Surfaces what got fed to the extras BFS so debugging "did
    /// extras=[all] actually expand to the right sub-wheels?" is one
    /// grep away. `#[serde(default)]` keeps old audit JSON parseable.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extras: Vec<String>,
    /// v0.12.0+: present when this wheel carried the auto-data-files
    /// inject for its checkout root. `None` for: PyPI/URL/path entries
    /// (no checkout root), and for git/named-git entries 2..N within a
    /// bundle that share a checkout with an earlier entry (dedup'd).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_data: Option<AutoDataReport>,
    /// v0.12.0+: present when this wheel was the 2..N entry from a
    /// checkout root already served by an earlier wheel. Records the
    /// shared checkout so debugging "why didn't this wheel carry the
    /// apps/?" is obvious.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_data_dedup_skipped_root: Option<PathBuf>,
}

/// One emitted conda run-dep: the spec retread sends to pixi, plus the
/// PyPI name it came from (recovered by reversing conda_name_from) so a
/// reader can correlate emit ↔ upstream pin.
#[derive(Debug, Serialize)]
pub struct EmittedDep {
    pub name: String,
    pub spec: String,
}

/// TOML fragments the user can copy-paste into their workspace to
/// mirror the bundle exactly. Each is a string ready to drop under
/// the corresponding header.
#[derive(Debug, Serialize)]
pub struct PixiTomlBlocks {
    /// Body of a `[dependencies]` or `[feature.X.dependencies]` table.
    /// One line per emitted conda run-dep:
    ///   `numpy = ">=1"`
    ///   `pyglet = "*"`
    /// Drop into the workspace to constrain its conda solve to the
    /// same versions retread chose.
    pub dependencies: String,

    /// Body of a `[pypi-options.dependency-overrides]` or
    /// `[feature.X.pypi-options.dependency-overrides]` table. One
    /// line per wheel in the bundle, pinned to that wheel's exact
    /// version:
    ///   `isaacsim = "==5.1.0.0"`
    ///   `pytorch3d = "==0.7.8+5043d15pt2.7.0cu128"`
    /// Drop into the workspace to force uv (PyPI side) to pick the
    /// same versions retread bundled. This is the block whose
    /// strict-pinning fights with conda picks -- mirroring it
    /// exactly is the surest fix.
    pub pypi_options_dependency_overrides: String,
}

/// v0.14.1+: one routing decision the BFS / auto-bundle path made
/// against the conda channel probe. Records WHAT was probed, WHAT
/// conda returned, and WHAT retread did with the answer. Written to
/// the audit even when the conda solve fails downstream (audit is
/// flushed at conda/outputs, not just conda/build_v1). This is the
/// disk-side answer to "why did retread emit `gym >=0.23,<0.24` as a
/// conda run-dep when conda-forge has no py3.11 build?"
#[derive(Debug, Clone, Serialize)]
pub struct ProbeDecision {
    /// Where in retread the decision happened.
    /// `"bfs"` = `resolve_bundle` extras/base-dep BFS.
    /// `"auto_bundle"` = `auto_bundle_transitives` (post-merge).
    pub stage: String,
    /// PyPI name as it appeared in the wheel's Requires-Dist.
    pub pypi_name: String,
    /// Conda name retread looked up (parselmouth alias or identity).
    pub conda_name: String,
    /// Spec retread asked the probe about (PyPI specifiers, post-
    /// normalization).
    pub spec: String,
    /// Target python version retread asked the probe to filter for.
    pub target_python: String,
    /// Channels the probe actually consulted (non-prefix.dev channels
    /// silently skipped; empty list means "indecisive").
    pub channels_consulted: Vec<String>,
    /// Some(true) at least one channel had a python-compatible
    /// satisfying version; Some(false) all consulted channels had
    /// candidates but none satisfied python+spec; None = indecisive
    /// (no channels consulted OR all probes errored).
    pub satisfiable: Option<bool>,
    /// Diagnostic count of matching variants across all channels.
    pub matching_candidates: usize,
    /// What retread did with the answer: `"short-circuit"` (emit as
    /// conda run-dep), `"fall-through-to-pypi"` (resolve+bundle from
    /// PyPI), `"indecisive-short-circuit"` (kept legacy optimistic
    /// behavior).
    pub routing_decision: String,
}

/// All info about one conda output retread produced.
#[derive(Debug, Serialize)]
pub struct BundleAudit {
    pub conda_name: String,
    pub version: String,
    pub wheels: Vec<WheelAudit>,
    pub emitted_run_deps: Vec<EmittedDep>,
    pub pixi_toml_blocks: PixiTomlBlocks,
    /// v0.14.1+: per-dep routing decisions made via the conda channel
    /// probe. Populated incrementally during resolve_all + auto-bundle;
    /// always present in the on-disk audit so debugging "why did this
    /// dep go to conda?" doesn't require any --trace flags.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub probe_decisions: Vec<ProbeDecision>,
    /// v0.33.5+: per-env results of running a real conda solve over
    /// (this env's effective deps + retread's emitted run-deps for
    /// this output). Keyed by env name. Each entry shows what THAT
    /// env's conda solver will fail on -- much more actionable than
    /// a cross-env union which would over-constrain.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub solve_diagnostics: BTreeMap<String, SolveDiagnostics>,
    /// P1 (cleanup): how many of the attempted env solve checks were
    /// abstentions (skipped because no repodata was reachable). When
    /// `envs_skipped == solve_diagnostics.len()` the entire run was an
    /// abstention and outputs shipped UNVERIFIED. Purely informational;
    /// never used for flow control. `#[serde(default)]` keeps older audit
    /// JSON parseable (absent field reads as 0).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub envs_skipped: usize,
}

/// Persisted form of `solve_check::SolveOutcome`. Mirrors the in-memory
/// type to keep the audit format stable even if rattler_solve's error
/// API changes. Stored in `retread-audit-<name>.json` under
/// `solve_diagnostics`.
#[derive(Debug, Clone, Serialize)]
pub struct SolveDiagnostics {
    pub satisfiable: bool,
    pub unsat_explanations: Vec<String>,
    pub channels_consulted: Vec<String>,
    pub specs_count: usize,
    pub records_count: usize,
    /// v0.34.0+: iterative refinement steps applied to make the
    /// emission solvable. Each entry records one widening pass; later
    /// passes saw the conflict shift to a new dep. Empty when the
    /// first solve passed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refinement_steps: Vec<RefinementStep>,
    /// v0.35.0+: actionable workspace-edit suggestions when the
    /// classifier determined the conflict is workspace-side
    /// (Class B/C). Empty for Class A (cascade resolved or in
    /// progress) and for sat outcomes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub workspace_edit_suggestions: Vec<crate::conflict_classifier::WorkspaceEditSuggestion>,
    /// v0.35.0+: terminal classification. Tells the user at a glance
    /// whether the failure is retread's responsibility ("A"/"A-exhausted")
    /// or the workspace's ("B"/"C"). Empty for sat outcomes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_classification: Option<String>,
    /// v1.4.0: true when the check could not run (no channels / no
    /// repodata) and abstained. `satisfiable: false` + `skipped: true`
    /// means "unknown", not "unsat" -- the SOLVE-FAILED summary and
    /// the conda/outputs fail gate both ignore skipped entries.
    #[serde(default)]
    pub skipped: bool,
}

/// One iteration of the solve-driven cascade refinement.
#[derive(Debug, Clone, Serialize)]
pub struct RefinementStep {
    /// 0-based iteration counter.
    pub iteration: usize,
    /// Top-level dep names rattler_solve flagged as the conflict
    /// entry points (before this widening).
    pub blocking_deps: Vec<String>,
    /// Deps retread widened to `*` in this iteration. Subset of
    /// blocking_deps (only deps retread is emitting can be widened
    /// from retread's side -- workspace-pinned deps go unwidened).
    pub widened_deps: Vec<String>,
    /// v0.35.0+: classifier verdict for THIS iteration's unsat.
    /// Tells the user whether retread can help or whether the
    /// workspace needs editing. `None` for legacy/empty rounds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classification: Option<String>,
    /// v0.35.0+: human-readable summary of what blocked, per
    /// classifier.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub blocking_summary: String,
    /// v0.36.0+: per-chain verdicts (one entry per blocking chain).
    /// Replaces v0.35.0's aggregate `classification` as the
    /// audit-readable source of truth for the loop's decisions.
    /// Each verdict tells the user exactly what retread did with
    /// THAT chain (widened, refused-as-ABI-anchor, suggested
    /// workspace edit, ...).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub verdicts: Vec<crate::conflict_classifier::PerChainVerdict>,
    /// v0.36.0+: invariant-violation messages emitted by the
    /// post-condition assertion in `iterative_solve_refinement`.
    /// Non-empty means retread caught itself trying to corrupt the
    /// output's ABI contract. Each entry names the offending dep +
    /// the expected vs actual spec. The cascade does NOT fail on
    /// these (they're logged loudly) but the audit makes them
    /// observable so a downstream test can pin them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub invariant_violations: Vec<String>,
}

/// Top-level audit: keyed by the (conda_name, python_version) pair
/// since the same source package may emit different outputs per
/// python variant.
#[derive(Debug, Serialize, Default)]
pub struct Audit {
    pub bundles: BTreeMap<String, BundleAudit>,
}

/// Build the `[pypi-options.dependency-overrides]` TOML body. The
/// PyPI override format is `pkg = "constraint"`; PyPI names keep
/// their original casing/separators per PEP 503, but pixi normalizes
/// to lowercase-dash on read so emitting lowercase-dash is safe and
/// matches what the user's hand-maintained overrides look like.
fn render_pypi_overrides(wheels: &[WheelAudit]) -> String {
    let mut lines = Vec::with_capacity(wheels.len());
    for w in wheels {
        let pypi_name = w.name.to_ascii_lowercase().replace('_', "-");
        // Quote-escape the version (PEP 440 local-version `+` is legal
        // in TOML bare strings but be safe).
        let escaped = w.version.replace('"', "\\\"");
        lines.push(format!("{pypi_name} = \"=={escaped}\""));
    }
    lines.join("\n")
}

/// Build the `[dependencies]` TOML body from the conda run-deps
/// retread emitted. Each line is `<conda_name> = "<spec>"` where
/// `<spec>` is the conda match-spec emitted (e.g. ">=1", "<2",
/// "==5.1.0.0", "*"). The leading `<name> ` prefix from the emitted
/// spec is stripped if present so the right-hand side is JUST the
/// version constraint.
fn render_conda_deps(emitted: &[EmittedDep]) -> String {
    let mut lines = Vec::with_capacity(emitted.len());
    for d in emitted {
        // Spec may be "<name> <version-constraint>" or just "<name>".
        // Strip the leading name to get the constraint part.
        let constraint = d
            .spec
            .strip_prefix(&d.name)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .unwrap_or("*");
        let escaped = constraint.replace('"', "\\\"");
        lines.push(format!("{} = \"{}\"", d.name, escaped));
    }
    lines.join("\n")
}

impl BundleAudit {
    /// Construct a BundleAudit including the rendered TOML blocks.
    /// `probe_decisions` is the routing-trace collected during the
    /// BFS + auto-bundle passes (v0.14.1+). Empty for legacy callers.
    /// `envs_skipped` counts how many env solve checks abstained (P1).
    pub fn new(
        conda_name: String,
        version: String,
        wheels: Vec<WheelAudit>,
        emitted_run_deps: Vec<EmittedDep>,
        probe_decisions: Vec<ProbeDecision>,
        solve_diagnostics: BTreeMap<String, SolveDiagnostics>,
        envs_skipped: usize,
    ) -> Self {
        let pypi_options_dependency_overrides = render_pypi_overrides(&wheels);
        let dependencies = render_conda_deps(&emitted_run_deps);
        Self {
            conda_name,
            version,
            wheels,
            emitted_run_deps,
            pixi_toml_blocks: PixiTomlBlocks {
                dependencies,
                pypi_options_dependency_overrides,
            },
            probe_decisions,
            solve_diagnostics,
            envs_skipped,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wheel(name: &str, version: &str, requires: &[&str]) -> WheelAudit {
        WheelAudit {
            name: name.into(),
            version: version.into(),
            requires_dist: requires.iter().map(|s| (*s).to_string()).collect(),
            extras: vec![],
            auto_data: None,
            auto_data_dedup_skipped_root: None,
        }
    }

    #[test]
    fn pypi_overrides_block_pins_every_wheel_to_exact_version() {
        // The pypi-options block is the one whose strictness matters
        // most -- it's what the user pastes into the workspace to
        // force uv (PyPI side) to match retread's bundled versions
        // exactly. Each line must be valid pixi.toml syntax:
        //     `pkg = "==X.Y.Z"`
        let wheels = vec![
            wheel("isaacsim", "5.1.0.0", &[]),
            wheel("isaaclab", "0.51.1", &[]),
            wheel("pytorch3d", "0.7.8+5043d15pt2.7.0cu128", &[]),
            // Underscore in PyPI name -> dash in pixi.toml override
            // (PEP 503 normalization, pixi reads either).
            wheel("typing_extensions", "4.12.2", &[]),
        ];
        let block = render_pypi_overrides(&wheels);

        // Each wheel produces one line, ==<exact-version>.
        assert!(
            block.contains("isaacsim = \"==5.1.0.0\""),
            "block:\n{block}"
        );
        assert!(block.contains("isaaclab = \"==0.51.1\""));
        // PEP 440 local-version identifier must survive verbatim --
        // this is the case (pytorch3d) that fails on the PyPI side
        // when the version doesn't exactly match.
        assert!(
            block.contains("pytorch3d = \"==0.7.8+5043d15pt2.7.0cu128\""),
            "+local identifier dropped; block:\n{block}",
        );
        // Underscore normalized to dash.
        assert!(block.contains("typing-extensions = \"==4.12.2\""));

        // Each line must match the TOML key-equals-quoted-value shape
        // (no special chars in the constraint side that would need
        // escaping for our supported version syntax).
        for line in block.lines() {
            assert!(
                line.contains(" = \"") && line.ends_with('"'),
                "line must be a valid `key = \"value\"` pair; got: {line}",
            );
        }
    }

    #[test]
    fn conda_deps_block_renders_spec_constraints() {
        let emitted = vec![
            EmittedDep {
                name: "numpy".into(),
                spec: "numpy >=1".into(),
            },
            EmittedDep {
                name: "pyglet".into(),
                spec: "pyglet".into(),
            }, // no constraint
            EmittedDep {
                name: "pytorch".into(),
                spec: "pytorch >=2.7,<3".into(),
            },
        ];
        let block = render_conda_deps(&emitted);
        assert!(block.contains("numpy = \">=1\""), "block:\n{block}");
        // `pyglet` (no constraint) renders as `pyglet = "*"`.
        assert!(block.contains("pyglet = \"*\""));
        assert!(block.contains("pytorch = \">=2.7,<3\""));

        for line in block.lines() {
            assert!(
                line.contains(" = \"") && line.ends_with('"'),
                "line must be a valid `key = \"value\"` pair; got: {line}",
            );
        }
    }

    #[test]
    fn bundle_audit_serializes_as_well_formed_json() {
        let audit = BundleAudit::new(
            "isaac-pack".into(),
            "5.1.0.0".into(),
            vec![wheel("isaacsim", "5.1.0.0", &["numpy==1.26.0", "pyglet<2"])],
            vec![
                EmittedDep {
                    name: "numpy".into(),
                    spec: "numpy >=1".into(),
                },
                EmittedDep {
                    name: "pyglet".into(),
                    spec: "pyglet".into(),
                },
            ],
            vec![],
            BTreeMap::new(),
            0,
        );
        let json = serde_json::to_string_pretty(&audit).unwrap();
        // Spot-check shape rather than exact bytes (field order is
        // stable but values may evolve).
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["conda_name"], "isaac-pack");
        assert_eq!(parsed["wheels"][0]["requires_dist"][1], "pyglet<2");
        assert!(
            parsed["pixi_toml_blocks"]["pypi_options_dependency_overrides"]
                .as_str()
                .unwrap()
                .contains("isaacsim = \"==5.1.0.0\""),
            "json:\n{json}",
        );
    }
}
