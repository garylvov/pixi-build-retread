//! Carry a bundled wheel's `Requires-Dist` bounds onto the conda side.
//!
//! ## The defect this closes
//!
//! `crate::audit::WheelAudit::requires_dist` records, per bundled wheel,
//! the literal `Requires-Dist:` lines the shipped wheel carries. Until
//! this module existed that record had NO reader (`src/audit.rs`: *"The
//! audit is purely informational; nothing else in retread reads it
//! back"*), so a bound a bundled wheel states about a **conda-routed**
//! name never reached the conda solver. Concretely: `transformers`
//! requires `huggingface-hub<1.0,>=0.34.0`, `huggingface_hub` is served
//! from conda (routed in as a transitive of `datasets`), and nothing in
//! the emitted output stopped the conda solver from picking
//! `huggingface_hub 1.28.0` — an install that satisfies conda and breaks
//! at import.
//!
//! ## What "routed to conda" means here
//!
//! A PyPI distribution is conda-routed for this output exactly when its
//! canonical name (`crate::relax::canonical_conda_name`, which folds
//! `-`/`_`/`.`/case per PEP 503 — so `huggingface-hub` and
//! `huggingface_hub` are the same key) already appears as a run
//! dependency of the emitted output. That set IS the live routing
//! decision this build made; reading it back needs no second whitelist
//! that could drift from it. A name that is not routed gets nothing —
//! the bundled wheel's own METADATA is the authority there, and the
//! conda solver has no opinion to correct.
//!
//! ## STATUS — producer half only (boarded debt, turn 2 of fix_d1_metadata)
//!
//! This module is the translator + intersection primitive. Its READER is NOT
//! yet wired: a post-hoc merge onto the emitted `depends` was implemented,
//! measured, and REVERTED because it overrides `relax_decision::decide`,
//! which `src/handler/mod.rs` documents as *"the sole policy-aware conflict
//! relaxation boundary"*. Six lib tests encode that policy and failed under
//! the merge: `handler::tests::{final_emission_opts_into_minimal_stale_cap_relaxation,
//! within_major_relaxation_becomes_the_final_structured_bundle_record,
//! final_emission_bands_numpy_exact_pin_but_preserves_cuda_cap,
//! produce_output_reflects_overrides_for_refinement_widening,
//! auto_routed_underscored_conda_name_emits_raw,
//! authoritative_numpy_pin_uses_conflicting_workspace_patch_fact_only_for_validation}`.
//!
//! The correct reader is therefore NOT after emission. The emission path
//! already intersects every bundled wheel's `Requires-Dist` into the conda
//! run-deps (`add_emission_constraint`, `src/handler/mod.rs`); the reason the
//! cap still goes missing is UPSTREAM of it — phase 2 rewrites each wheel's
//! METADATA per the relax policy and the bundle then carries only the
//! POST-rewrite lines (`src/handler/mod.rs:11215` computes
//! `original_requires_dist`, but `ResolvedWheel` never stores it), so
//! `strip_upper_bounds` has already deleted `<1.0` before emission reads it.
//! Wiring = keep the pre-D lines on `ResolvedWheel` and feed them to the
//! emission-constraint collection, or advertise them as conda `constrains`
//! (`CondaOutputDependencies::constraints`, currently always empty) with the
//! matching lock field so cold and replay advertise byte-identical metadata.
//!
//! ## Failing loudly
//!
//! [`conda_matchspec`] translates the WHOLE specifier set or returns an
//! error. Supported: `>=`, `>`, `<`, `<=`, `==`, `==X.*`, `!=`,
//! `!=X.*`, `~=`. Unsupported: `===` (PEP 440 arbitrary equality has no
//! conda rendering) and any `~=X` with a single release segment. A bound
//! is never silently dropped: the caller gets an `Err` naming the line.

use std::collections::BTreeSet;
use std::str::FromStr;

use anyhow::{Context, Result, anyhow};
use uv_pep508::uv_pep440::{self, Operator, VersionSpecifiers};
use uv_pep508::{MarkerEnvironment, Requirement, VersionOrUrl};

use crate::relax::canonical_conda_name;

/// One bound carried from a bundled wheel onto a conda-routed name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CarriedConstraint {
    /// Canonical (PEP 503-folded) name of the conda-routed dependency.
    pub canonical_name: String,
    /// Conda match-spec constraint half, e.g. `">=0.34.0,<1.0"`.
    pub spec: String,
    /// Bundled wheel whose `Requires-Dist` stated the bound.
    pub requirer: String,
}

/// Render one PEP 440 specifier into its conda form, or `Err` when it has
/// no faithful conda rendering. Never returns `Ok` having dropped a
/// clause — that is the whole point of this function existing beside
/// `relax::convert_specifiers`, which drops `===` with a warning.
fn conda_one(spec: &uv_pep440::VersionSpecifier) -> Result<String> {
    let op = match spec.operator() {
        Operator::Equal => "==",
        Operator::NotEqual => "!=",
        Operator::LessThan => "<",
        Operator::LessThanEqual => "<=",
        Operator::GreaterThan => ">",
        Operator::GreaterThanEqual => ">=",
        Operator::EqualStar => {
            // Conda spells a PEP 440 wildcard the same way.
            return Ok(format!("=={}.*", spec.version()));
        }
        Operator::NotEqualStar => return Ok(format!("!={}.*", spec.version())),
        Operator::TildeEqual => {
            // `~=X.Y[.Z]` == `>=X.Y[.Z],<X[.Y+1]`. A single-segment `~=X`
            // is not valid PEP 440 compatible-release syntax; refuse it
            // rather than invent a ceiling.
            let release = spec.version().release();
            if release.len() < 2 {
                return Err(anyhow!(
                    "`~={}` has no compatible-release ceiling (needs at least two \
                     release segments)",
                    spec.version()
                ));
            }
            let mut ceiling: Vec<u64> = release[..release.len() - 1].to_vec();
            let last = ceiling
                .last_mut()
                .expect("release[..len-1] is non-empty for len >= 2");
            *last = last
                .checked_add(1)
                .ok_or_else(|| anyhow!("`~={}` release segment overflows", spec.version()))?;
            let upper = ceiling
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(".");
            let upper = if spec.version().epoch() == 0 {
                upper
            } else {
                format!("{}!{upper}", spec.version().epoch())
            };
            return Ok(format!(">={},<{upper}", spec.version()));
        }
        Operator::ExactEqual => {
            return Err(anyhow!(
                "`==={}` (PEP 440 arbitrary equality) has no conda match-spec \
                 rendering",
                spec.version()
            ));
        }
    };
    Ok(format!("{op}{}", spec.version()))
}

/// Translate a full PEP 440 specifier set into a conda match-spec
/// constraint (the half after the name). Every clause must translate.
pub fn conda_matchspec(specifiers: &VersionSpecifiers) -> Result<String> {
    let mut parts = Vec::with_capacity(specifiers.len());
    for spec in specifiers.iter() {
        parts.push(conda_one(spec)?);
    }
    Ok(parts.join(","))
}

/// Split a conda constraint (`">=1,<2"`) into its clauses.
fn clauses(spec: &str) -> Vec<String> {
    spec.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty() && *part != "*")
        .map(str::to_string)
        .collect()
}

/// Split one conda clause into `(operator, version)`.
fn split_clause(clause: &str) -> (&str, &str) {
    for op in ["===", "==", "!=", ">=", "<=", ">", "<", "="] {
        if let Some(rest) = clause.strip_prefix(op) {
            return (op, rest.trim());
        }
    }
    ("", clause)
}

/// Compare two bounds in the SAME direction and answer whether `candidate`
/// is strictly tighter than `incumbent`. `upper` selects the direction.
///
/// Returns `None` when either version is not PEP 440-parseable — the caller
/// must then keep BOTH clauses rather than guess, so no bound is ever lost.
fn tighter(incumbent: (&str, &str), candidate: (&str, &str), upper: bool) -> Option<bool> {
    let left = uv_pep440::Version::from_str(incumbent.1).ok()?;
    let right = uv_pep440::Version::from_str(candidate.1).ok()?;
    if left != right {
        return Some(if upper { right < left } else { right > left });
    }
    // Same version: the EXCLUSIVE form is the tighter one.
    let incl = |op: &str| op == "<=" || op == ">=";
    Some(incl(incumbent.0) && !incl(candidate.0))
}

/// Merge a carried constraint into a spec the output already emits.
///
/// Returns `None` when nothing should change. The merge INTERSECTS: a
/// carried bound that is tighter than the emitted one replaces it, and a
/// bound in a direction the emitted spec left open is added. That is the
/// point — a bundled wheel's `<1.0` has to beat an emitted `<2` band, or
/// the solver still floats past the shipped wheel's cap.
///
/// Two things it will not do:
///
/// * touch an existing exact pin (`==1.2.3`): that is a deliberate
///   selection (a bundled family member, an override), and a foreign
///   clause could only empty its version set;
/// * guess when a version string will not parse as PEP 440 — both clauses
///   are then kept, so an genuinely irreconcilable pair surfaces as a
///   solver refusal instead of a silently dropped bound.
pub fn merge_into_existing(existing: &str, carried: &str) -> Option<String> {
    let existing_clauses = clauses(existing);
    if existing_clauses.iter().any(|clause| {
        let (op, version) = split_clause(clause);
        (op == "==" || op == "=" || op == "===") && !version.ends_with('*')
    }) {
        return None;
    }

    let mut lower: Option<String> = None;
    let mut upper: Option<String> = None;
    let mut other: Vec<String> = Vec::new();
    let mut unmergeable: Vec<String> = Vec::new();

    let mut absorb = |clause: &str| {
        let (op, _) = split_clause(clause);
        let slot = match op {
            ">" | ">=" => &mut lower,
            "<" | "<=" => &mut upper,
            _ => {
                if !other.iter().any(|kept| kept == clause) {
                    other.push(clause.to_string());
                }
                return;
            }
        };
        let is_upper = matches!(op, "<" | "<=");
        match slot.as_deref() {
            None => *slot = Some(clause.to_string()),
            Some(incumbent) if incumbent == clause => {}
            Some(incumbent) => {
                match tighter(split_clause(incumbent), split_clause(clause), is_upper) {
                    Some(true) => *slot = Some(clause.to_string()),
                    Some(false) => {}
                    // Unorderable pair: keep both, fail closed at solve time.
                    None => {
                        if !unmergeable.iter().any(|kept| kept == clause) {
                            unmergeable.push(clause.to_string());
                        }
                    }
                }
            }
        }
    };

    for clause in &existing_clauses {
        absorb(clause);
    }
    for clause in clauses(carried) {
        absorb(&clause);
    }
    drop(absorb);

    let merged: Vec<String> = lower
        .into_iter()
        .chain(upper)
        .chain(other)
        .chain(unmergeable)
        .collect();
    if merged == existing_clauses {
        return None;
    }
    Some(merged.join(","))
}

/// Compute the constraints a bundle's wheels impose on conda-routed
/// names.
///
/// * `wheel_requires_dist` — `(requirer wheel name, raw Requires-Dist
///   line)` pairs; this is exactly the record
///   `crate::audit::WheelAudit::requires_dist` carries, per wheel.
/// * `routed` — canonical names the output already routes to conda.
/// * `env` — marker environment; a requirement whose marker is false for
///   this target (including every `extra ==` line, evaluated with no
///   active extras) states nothing about the shipped install and is
///   skipped, exactly as `relax::translate` does.
///
/// Errors when a routed name's bound cannot be rendered for conda, so an
/// unsupported operator surfaces as a refusal instead of a lost bound.
pub fn carried_constraints(
    wheel_requires_dist: &[(String, String)],
    routed: &BTreeSet<String>,
    env: &MarkerEnvironment,
) -> Result<Vec<CarriedConstraint>> {
    let mut out: Vec<CarriedConstraint> = Vec::new();
    for (requirer, raw) in wheel_requires_dist {
        let Ok(req): std::result::Result<Requirement, _> = Requirement::from_str(raw) else {
            // Unparseable lines are already reported by the emission path;
            // this reader must not turn a warning into a build failure.
            continue;
        };
        if !req.marker.evaluate(env, &[]) {
            continue;
        }
        let canonical = canonical_conda_name(req.name.as_ref());
        if !routed.contains(&canonical) {
            continue;
        }
        let Some(VersionOrUrl::VersionSpecifier(specifiers)) = req.version_or_url.as_ref() else {
            // No bound (or a direct URL, which is an artifact selection and
            // not expressible as a conda range) — nothing to carry.
            continue;
        };
        if specifiers.is_empty() {
            continue;
        }
        let spec = conda_matchspec(specifiers).with_context(|| {
            format!(
                "carrying bundled wheel `{requirer}`'s requirement `{raw}` onto the \
                 conda-routed dependency `{canonical}`"
            )
        })?;
        out.push(CarriedConstraint {
            canonical_name: canonical,
            spec,
            requirer: requirer.clone(),
        });
    }
    out.sort();
    out.dedup();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> MarkerEnvironment {
        crate::relax::default_marker_env("3.11").unwrap()
    }

    fn routed(names: &[&str]) -> BTreeSet<String> {
        names
            .iter()
            .map(|name| canonical_conda_name(name))
            .collect()
    }

    fn lines(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(wheel, raw)| ((*wheel).to_string(), (*raw).to_string()))
            .collect()
    }

    /// THE live pair. `transformers` states `huggingface-hub<1.0,>=0.34.0`;
    /// `huggingface_hub` is routed to conda (underscore spelling, as
    /// conda-forge names it). Both halves of the bound must survive.
    #[test]
    fn conda_routed_requirement_carries_both_bounds_across_name_spellings() {
        let carried = carried_constraints(
            &lines(&[("transformers", "huggingface-hub<1.0,>=0.34.0")]),
            &routed(&["huggingface_hub"]),
            &env(),
        )
        .unwrap();
        assert_eq!(carried.len(), 1, "{carried:?}");
        assert_eq!(carried[0].canonical_name, "huggingface-hub");
        assert_eq!(carried[0].spec, ">=0.34.0,<1.0");
        assert_eq!(carried[0].requirer, "transformers");
    }

    /// Negative arm: the same line, but the name is NOT conda-routed.
    /// Nothing is carried — the bundled wheel's own METADATA is the
    /// authority for a PyPI-side dependency.
    #[test]
    fn unrouted_requirement_carries_nothing() {
        let carried = carried_constraints(
            &lines(&[("transformers", "huggingface-hub<1.0,>=0.34.0")]),
            &routed(&["numpy"]),
            &env(),
        )
        .unwrap();
        assert!(carried.is_empty(), "{carried:?}");
    }

    #[test]
    fn every_supported_operator_renders() {
        for (raw, want) in [
            ("pkg>=1.2", ">=1.2"),
            ("pkg>1.2", ">1.2"),
            ("pkg<2", "<2"),
            ("pkg<=2", "<=2"),
            ("pkg==1.2.3", "==1.2.3"),
            ("pkg!=1.4", "!=1.4"),
            ("pkg==1.2.*", "==1.2.*"),
            ("pkg!=1.2.*", "!=1.2.*"),
            ("pkg~=1.4.2", ">=1.4.2,<1.5"),
            ("pkg~=2.1", ">=2.1,<3"),
        ] {
            let carried =
                carried_constraints(&lines(&[("w", raw)]), &routed(&["pkg"]), &env()).unwrap();
            assert_eq!(carried.len(), 1, "{raw} -> {carried:?}");
            assert_eq!(carried[0].spec, want, "{raw}");
        }
    }

    /// A bound with no conda rendering must FAIL, never be dropped: a
    /// silently dropped cap is the exact defect this module closes.
    #[test]
    fn unsupported_operator_fails_loudly_instead_of_dropping_the_bound() {
        let error = carried_constraints(
            &lines(&[("w", "pkg===1.0+local")]),
            &routed(&["pkg"]),
            &env(),
        )
        .expect_err("`===` has no conda rendering and must refuse");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("arbitrary equality"), "{rendered}");
        assert!(rendered.contains("pkg"), "{rendered}");
    }

    /// An unsupported operator on an UNROUTED name is not this reader's
    /// business and must not fail the build.
    #[test]
    fn unsupported_operator_on_unrouted_name_is_ignored() {
        carried_constraints(
            &lines(&[("w", "pkg===1.0+local")]),
            &routed(&["other"]),
            &env(),
        )
        .expect("an unrouted name states nothing about the conda side");
    }

    #[test]
    fn false_markers_and_extras_state_nothing() {
        let carried = carried_constraints(
            &lines(&[
                ("w", "pkg<1.0; python_version < \"3.9\""),
                ("w", "pkg<2.0; extra == \"torchhub\""),
            ]),
            &routed(&["pkg"]),
            &env(),
        )
        .unwrap();
        assert!(carried.is_empty(), "{carried:?}");
    }

    #[test]
    fn bare_and_url_requirements_carry_nothing() {
        let carried = carried_constraints(
            &lines(&[
                ("w", "pkg"),
                (
                    "w",
                    "pkg @ https://example.invalid/pkg-1.0-py3-none-any.whl",
                ),
            ]),
            &routed(&["pkg"]),
            &env(),
        )
        .unwrap();
        assert!(carried.is_empty(), "{carried:?}");
    }

    #[test]
    fn merge_adds_the_missing_cap_and_is_idempotent() {
        assert_eq!(
            merge_into_existing(">=0.34.0", ">=0.34.0,<1.0").as_deref(),
            Some(">=0.34.0,<1.0"),
        );
        // Second application changes nothing.
        assert_eq!(merge_into_existing(">=0.34.0,<1.0", ">=0.34.0,<1.0"), None);
    }

    /// The whole point: an emitted band that ADMITS the forbidden version
    /// (`huggingface_hub >=1.28,<2` vs the bundled wheel's `<1.0`) must be
    /// intersected down, not left alone.
    #[test]
    fn merge_intersects_a_looser_emitted_cap() {
        assert_eq!(
            merge_into_existing(">=1.28,<2", ">=0.34.0,<1.0").as_deref(),
            Some(">=1.28,<1.0"),
        );
        assert_eq!(merge_into_existing(">=1,<2", "<3").as_deref(), None);
        // `<X` is tighter than `<=X` at the same version.
        assert_eq!(
            merge_into_existing("<=1.0", "<1.0").as_deref(),
            Some("<1.0")
        );
    }

    #[test]
    fn merge_never_touches_an_exact_pin() {
        assert_eq!(merge_into_existing("==1.28.0", "<1.0"), None);
        // A wildcard equality is a range, not a pin, so it is kept as a
        // clause alongside the carried bound.
        assert_eq!(
            merge_into_existing("==1.*", "<1.9").as_deref(),
            Some("<1.9,==1.*"),
        );
    }

    /// A version neither side can parse must not silently lose a bound.
    #[test]
    fn merge_keeps_both_clauses_when_a_version_is_unorderable() {
        let merged = merge_into_existing("<snapshot-x", "<1.0").expect("must change");
        assert!(merged.contains("<snapshot-x"), "{merged}");
        assert!(merged.contains("<1.0"), "{merged}");
    }

    #[test]
    fn merge_treats_star_as_unconstrained() {
        assert_eq!(merge_into_existing("*", "<1.0").as_deref(), Some("<1.0"));
        assert_eq!(merge_into_existing("", ">=1").as_deref(), Some(">=1"));
    }
}
