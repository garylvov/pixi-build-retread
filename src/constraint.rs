//! Typed constraint provenance and the single constraint finalization path.
//!
//! Provenance is retained until an emitted or restored requirement is fully
//! assembled. Authority is deliberately derived only by [`authority`], so a
//! new source of constraints cannot acquire independent hard/soft semantics.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use uv_pep508::uv_pep440::{
    Operator, Version, VersionSpecifier, VersionSpecifiers, release_specifiers_to_ranges,
};

use crate::relax::PypiKey;

/// Where a constraint entered the composed dependency graph.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Provenance {
    /// `Requires-Dist` metadata from an index-fetched wheel.
    #[default]
    IndexWheelMetadata,
    /// Relaxed metadata from a git, sdist, or path-built wheel.
    SourceBuiltRelaxed,
    /// A relaxed floor inherited through `retread-deps-from`.
    DepsFromRelaxed,
    /// A precise fact solved for a consuming workspace conda environment.
    WorkspaceCondaFact(String),
    /// A root requirement given to uv.
    UvRoot,
    /// An additive uv constraint.
    UvConstraint,
    /// A uv override, which replaces ordinary dependency requirements.
    UvOverride,
    /// A version selected by a prior solve, retained only as a preference.
    PriorSelection,
}

/// The authority a constraint has while the composed graph is finalized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Authority {
    Authoritative,
    Advisory,
    Preference,
}

/// The single provenance-to-authority derivation.
pub fn authority(provenance: &Provenance) -> Authority {
    match provenance {
        Provenance::IndexWheelMetadata
        | Provenance::WorkspaceCondaFact(_)
        | Provenance::UvRoot
        | Provenance::UvConstraint
        | Provenance::UvOverride => Authority::Authoritative,
        Provenance::SourceBuiltRelaxed | Provenance::DepsFromRelaxed => Authority::Advisory,
        Provenance::PriorSelection => Authority::Preference,
    }
}

/// Stable identity for the structured origin of one constraint.
///
/// The encoded value is assembled from semantic fields supplied by the
/// constraint producer (for example wheel name/version and normalized
/// requirement), never from the free-form, user-facing [`Constraint::source`].
/// Every field is byte-length-prefixed so distinct component boundaries cannot
/// collide (`["ab", "c"]` and `["a", "bc"]` remain different identities).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstraintOriginId(String);

impl ConstraintOriginId {
    /// Construct an origin identity from a stable kind and ordered semantic
    /// components.
    ///
    /// Callers must pass canonical structured values, not diagnostic prose or
    /// [`Constraint::source`]. No `From<String>` or unstructured constructor is
    /// provided so origin creation remains explicit at each source boundary.
    pub fn from_parts<I, S>(kind: &str, parts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        fn push_part(encoded: &mut String, part: &str) {
            encoded.push_str(&part.len().to_string());
            encoded.push(':');
            encoded.push_str(part);
        }

        let mut encoded = String::new();
        push_part(&mut encoded, kind);
        for part in parts {
            push_part(&mut encoded, part.as_ref());
        }
        Self(encoded)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ConstraintOriginId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One version constraint with enough provenance to determine its authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    pub specifiers: VersionSpecifiers,
    pub provenance: Provenance,
    pub source: String,
    pub origin_id: ConstraintOriginId,
}

impl Constraint {
    pub fn authority(&self) -> Authority {
        authority(&self.provenance)
    }
}

/// A finalized requirement whose active constraints have an empty
/// intersection.
///
/// The package identity and complete source list remain structurally available
/// to callers. Joint-route callers attach the concrete environment/target
/// scope before surfacing the conflict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conflict {
    pub package: PypiKey,
    pub sources: String,
    scope: String,
    requirements: Vec<Constraint>,
    suggestion: Option<Box<ConflictSuggestion>>,
}

/// A paste-ready manifest edit attached to one fail-closed conflict.
///
/// The suggestion is diagnostic only: constructing or rendering it never
/// mutates the named pack manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConflictSuggestion {
    pub pack_manifest: String,
    pub toml: String,
}

impl Conflict {
    /// Add a user-facing solve scope such as an environment, target profile,
    /// platform, Python version, and bundle identity.
    pub(crate) fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = format!(" {}", scope.into());
        self
    }

    pub(crate) fn with_suggestion(
        mut self,
        pack_manifest: impl Into<String>,
        toml: impl Into<String>,
    ) -> Self {
        self.suggestion = Some(Box::new(ConflictSuggestion {
            pack_manifest: pack_manifest.into(),
            toml: toml.into(),
        }));
        self
    }

    pub(crate) fn requirements(&self) -> &[Constraint] {
        &self.requirements
    }

    pub fn suggestion(&self) -> Option<&ConflictSuggestion> {
        self.suggestion.as_deref()
    }
}

impl fmt::Display for Conflict {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "dependency conflict{}: `{}` requirements are mutually unsatisfiable: {}. \
             Resolve by pinning one side, or use `retread-relax`, `retread-overrides`, or \
             `retread-drop-deps` in the pack manifest (see README).",
            self.scope, self.package, self.sources
        )?;
        if let Some(suggestion) = &self.suggestion {
            write!(
                formatter,
                "\n\nSuggested fix in {}:\n{}",
                suggestion.pack_manifest,
                suggestion.toml.trim_end()
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for Conflict {}

/// A successful policy-neutral finalization, retaining whether the legacy
/// conda-as-truth rule had to discard advisory floor/equality clauses.
///
/// The detailed form lets ABI-sensitive callers fail closed without changing
/// the public [`finalize`] contract for ordinary packages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FinalizeSuccess {
    Unchanged(VersionSpecifiers),
    AdvisoryFloorSoftened {
        specifiers: VersionSpecifiers,
        unsoftened_conflict: Conflict,
    },
}

impl FinalizeSuccess {
    fn into_specifiers(self) -> VersionSpecifiers {
        match self {
            Self::Unchanged(specifiers) | Self::AdvisoryFloorSoftened { specifiers, .. } => {
                specifiers
            }
        }
    }
}

/// Every structural dependency conflict found while validating one solve
/// request.
///
/// Individual entries retain the ordinary [`Conflict`] rendering so package,
/// provenance, solve scope, and remediation stay actionable. The report only
/// adds deterministic numbering and separation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConflictReport {
    pub conflicts: Vec<Conflict>,
}

impl ConflictReport {
    pub(crate) fn with_scope(mut self, scope: impl Into<String>) -> Self {
        let scope = scope.into();
        self.conflicts = self
            .conflicts
            .into_iter()
            .map(|conflict| conflict.with_scope(scope.clone()))
            .collect();
        self
    }
}

impl fmt::Display for ConflictReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} dependency conflicts found:", self.conflicts.len())?;
        for (index, conflict) in self.conflicts.iter().enumerate() {
            if index > 0 {
                write!(f, "\n\n")?;
            }
            write!(f, "{}. {conflict}", index + 1)?;
        }
        Ok(())
    }
}

impl std::error::Error for ConflictReport {}

/// Preserve the existing typed single-conflict error while representing two
/// or more conflicts as one report.
pub(crate) fn aggregate_conflicts(mut conflicts: Vec<Conflict>) -> anyhow::Error {
    assert!(!conflicts.is_empty(), "cannot aggregate zero conflicts");
    conflicts.sort_by(|left, right| {
        (
            &left.package,
            &left.sources,
            &left.scope,
            left.suggestion
                .as_ref()
                .map(|suggestion| (&suggestion.pack_manifest, &suggestion.toml)),
        )
            .cmp(&(
                &right.package,
                &right.sources,
                &right.scope,
                right
                    .suggestion
                    .as_ref()
                    .map(|suggestion| (&suggestion.pack_manifest, &suggestion.toml)),
            ))
    });
    if conflicts.len() == 1 {
        anyhow::Error::new(conflicts.pop().expect("one conflict"))
    } else {
        anyhow::Error::new(ConflictReport { conflicts })
    }
}

/// Append a typed conflict error to a request-level collection.
///
/// `anyhow` context may wrap either a singleton [`Conflict`] or an already
/// aggregated [`ConflictReport`]. Inspecting the chain by reference retains
/// that typing without discarding the original error when it is unrelated.
pub(crate) fn collect_conflicts(
    error: anyhow::Error,
    conflicts: &mut Vec<Conflict>,
) -> Result<(), anyhow::Error> {
    if let Some(report) = error.downcast_ref::<ConflictReport>() {
        conflicts.extend(report.conflicts.iter().cloned());
        Ok(())
    } else if let Some(conflict) = error.downcast_ref::<Conflict>() {
        conflicts.push(conflict.clone());
        Ok(())
    } else {
        Err(error)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum DedupeVersionKey {
    /// Ordinary PEP 440 comparison ignores trailing release zeroes.
    Pep440(Version),
    /// Compatible-release and wildcard operators depend on written length.
    ReleaseLength(Version, usize),
    /// Arbitrary equality compares the retained version spelling.
    Arbitrary(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SpecifierDedupeKey {
    operator: Operator,
    version: DedupeVersionKey,
}

fn specifier_dedupe_key(specifier: &VersionSpecifier) -> SpecifierDedupeKey {
    let operator = *specifier.operator();
    let version = match operator {
        Operator::TildeEqual | Operator::EqualStar | Operator::NotEqualStar => {
            DedupeVersionKey::ReleaseLength(
                specifier.version().clone(),
                specifier.version().release().len(),
            )
        }
        Operator::ExactEqual => DedupeVersionKey::Arbitrary(specifier.to_string()),
        Operator::Equal
        | Operator::NotEqual
        | Operator::LessThan
        | Operator::LessThanEqual
        | Operator::GreaterThan
        | Operator::GreaterThanEqual => DedupeVersionKey::Pep440(specifier.version().clone()),
    };
    SpecifierDedupeKey { operator, version }
}

/// Deduplicate without collapsing clauses whose written release length or
/// spelling changes their semantics.
pub(crate) fn dedup_specifier_clauses(
    clauses: impl Iterator<Item = VersionSpecifier>,
) -> VersionSpecifiers {
    let mut unique: BTreeMap<SpecifierDedupeKey, (String, VersionSpecifier)> = BTreeMap::new();
    for specifier in clauses {
        let rendered = specifier.to_string();
        let key = specifier_dedupe_key(&specifier);
        match unique.entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert((rendered, specifier));
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                let current = &entry.get().0;
                if (rendered.len(), rendered.as_str()) < (current.len(), current.as_str()) {
                    entry.insert((rendered, specifier));
                }
            }
        }
    }
    unique
        .into_values()
        .map(|(_, specifier)| specifier)
        .collect()
}

/// Remove same-operator bounds already implied by a stricter bound. This
/// keeps the finalized form stable when an emitted compatibility envelope is
/// stricter than upstream metadata, without granting that envelope override
/// semantics or collapsing operator-sensitive clauses such as `~=`/`==*`.
fn remove_redundant_specifier_clauses(specifiers: VersionSpecifiers) -> VersionSpecifiers {
    let target = release_specifiers_to_ranges(specifiers.clone());
    let mut clauses: Vec<VersionSpecifier> = specifiers.into_iter().collect();
    let mut index = 0;
    while index < clauses.len() {
        let operator = *clauses[index].operator();
        if !matches!(
            operator,
            Operator::GreaterThan
                | Operator::GreaterThanEqual
                | Operator::LessThan
                | Operator::LessThanEqual
        ) || !clauses
            .iter()
            .enumerate()
            .any(|(other, clause)| other != index && *clause.operator() == operator)
        {
            index += 1;
            continue;
        }
        let trial: VersionSpecifiers = clauses
            .iter()
            .enumerate()
            .filter(|(candidate, _)| *candidate != index)
            .map(|(_, specifier)| specifier.clone())
            .collect();
        if release_specifiers_to_ranges(trial) == target {
            clauses.remove(index);
        } else {
            index += 1;
        }
    }
    clauses.into_iter().collect()
}

/// True when the full PEP 440 intersection is empty, including arbitrary
/// equality contradictions that ordered-version equality cannot distinguish.
pub(crate) fn specifiers_unsatisfiable(specifiers: &VersionSpecifiers) -> bool {
    if specifiers.iter().any(|specifier| {
        let version = specifier.version();
        match *specifier.operator() {
            Operator::TildeEqual => {
                let release = version.release();
                release.len() >= 2 && release[release.len() - 2] == u64::MAX
            }
            Operator::GreaterThan => match version.dev() {
                Some(dev) => dev == u64::MAX,
                None => version.post() == Some(u64::MAX),
            },
            Operator::EqualStar | Operator::NotEqualStar => {
                version.post() == Some(u64::MAX)
                    || version.pre().is_some_and(|pre| pre.number == u64::MAX)
                    || version
                        .release()
                        .last()
                        .is_some_and(|segment| *segment == u64::MAX)
            }
            _ => false,
        }
    }) {
        // uv's range conversion constructs exclusive ceilings with unchecked
        // `u64 + 1`. A ceiling that cannot be represented cannot prove a
        // nonempty ABI constraint, so reject it before either conversion.
        return true;
    }

    let full = release_specifiers_to_ranges(VersionSpecifiers::empty());
    let range_is_empty = full.intersection(&specifiers.clone().into()).is_empty();
    let arbitrary_exact_conflict = specifiers.iter().any(|specifier| {
        *specifier.operator() == Operator::ExactEqual && !specifiers.contains(specifier.version())
    });
    range_is_empty || arbitrary_exact_conflict
}

fn intersect(constraints: &[&Constraint]) -> VersionSpecifiers {
    dedup_specifier_clauses(
        constraints
            .iter()
            .flat_map(|constraint| constraint.specifiers.iter().cloned()),
    )
}

fn operator_forces_floor(operator: Operator) -> bool {
    matches!(
        operator,
        Operator::GreaterThan
            | Operator::GreaterThanEqual
            | Operator::Equal
            | Operator::ExactEqual
            | Operator::TildeEqual
            | Operator::EqualStar
    )
}

fn conflict_from_active(package: &PypiKey, active: &[&Constraint]) -> Conflict {
    let sources = active
        .iter()
        .map(|constraint| {
            let rendered = if constraint.specifiers.is_empty() {
                "*".to_string()
            } else {
                constraint.specifiers.to_string()
            };
            format!("`{rendered}` required by {}", constraint.source)
        })
        .collect::<Vec<_>>()
        .join("; ");
    Conflict {
        package: package.clone(),
        sources,
        scope: String::new(),
        requirements: active
            .iter()
            .map(|constraint| (*constraint).clone())
            .collect(),
        suggestion: None,
    }
}

/// Apply override replacement, exclude preferences, deduplicate, and prove
/// the active constraint intersection satisfiable.
///
/// If (and only if) separately satisfiable authoritative and advisory sets
/// conflict solely because of advisory floor/equality clauses, those clauses
/// yield. Advisory caps and exclusions are retained. Conflicts internal to
/// either authority class always fail loudly.
pub fn finalize(
    package: &PypiKey,
    constraints: &[Constraint],
) -> Result<VersionSpecifiers, Conflict> {
    finalize_impl(package, constraints, true).map(FinalizeSuccess::into_specifiers)
}

/// The exact strict finalization semantics without diagnostic side effects.
///
/// Candidate search uses this oracle for speculative subsets. The public
/// [`finalize`] wrapper retains its existing committed-path warning behavior.
pub(crate) fn finalize_quiet(
    package: &PypiKey,
    constraints: &[Constraint],
) -> Result<VersionSpecifiers, Conflict> {
    finalize_impl(package, constraints, false).map(FinalizeSuccess::into_specifiers)
}

/// Quiet policy-neutral finalization with enough detail for ABI callers to
/// reject a success that depended on discarding an advisory clause.
pub(crate) fn finalize_quiet_detailed(
    package: &PypiKey,
    constraints: &[Constraint],
) -> Result<FinalizeSuccess, Conflict> {
    finalize_impl(package, constraints, false)
}

fn finalize_impl(
    package: &PypiKey,
    constraints: &[Constraint],
    emit_diagnostics: bool,
) -> Result<FinalizeSuccess, Conflict> {
    let has_override = constraints
        .iter()
        .any(|constraint| matches!(&constraint.provenance, Provenance::UvOverride));
    let active: Vec<&Constraint> = constraints
        .iter()
        .filter(|constraint| constraint.authority() != Authority::Preference)
        .filter(|constraint| {
            !has_override
                || matches!(
                    &constraint.provenance,
                    Provenance::UvOverride
                        | Provenance::UvConstraint
                        | Provenance::WorkspaceCondaFact(_)
                        | Provenance::DepsFromRelaxed
                )
        })
        .collect();

    let combined = intersect(&active);
    if !specifiers_unsatisfiable(&combined) {
        return Ok(FinalizeSuccess::Unchanged(
            remove_redundant_specifier_clauses(combined),
        ));
    }

    let authoritative: Vec<&Constraint> = active
        .iter()
        .copied()
        .filter(|constraint| constraint.authority() == Authority::Authoritative)
        .collect();
    let advisory: Vec<&Constraint> = active
        .iter()
        .copied()
        .filter(|constraint| constraint.authority() == Authority::Advisory)
        .collect();

    // Prove both classes independently. Without this guard, an unrelated
    // authoritative `*` could mask a real advisory-vs-advisory conflict.
    if !authoritative.is_empty()
        && !advisory.is_empty()
        && !specifiers_unsatisfiable(&intersect(&authoritative))
        && !specifiers_unsatisfiable(&intersect(&advisory))
    {
        let mut dropped_floor = false;
        let mut kept = Vec::new();
        for constraint in &active {
            for specifier in constraint.specifiers.iter() {
                if constraint.authority() == Authority::Advisory
                    && operator_forces_floor(*specifier.operator())
                {
                    dropped_floor = true;
                } else {
                    kept.push(specifier.clone());
                }
            }
        }
        let softened = dedup_specifier_clauses(kept.into_iter());
        if dropped_floor && !specifiers_unsatisfiable(&softened) {
            let unsoftened_conflict = conflict_from_active(package, &active);
            if emit_diagnostics {
                tracing::warn!(
                    package = %package,
                    softened = %softened,
                    "an advisory lower bound conflicted with an authoritative constraint; \
                     dropped the advisory floor (conda-as-truth)",
                );
            }
            return Ok(FinalizeSuccess::AdvisoryFloorSoftened {
                specifiers: remove_redundant_specifier_clauses(softened),
                unsoftened_conflict,
            });
        }
    }

    Err(conflict_from_active(package, &active))
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    fn constraint(specifiers: &str, provenance: Provenance, source: &str) -> Constraint {
        let provenance_tag = match &provenance {
            Provenance::IndexWheelMetadata => "index-wheel-metadata",
            Provenance::SourceBuiltRelaxed => "source-built-relaxed",
            Provenance::DepsFromRelaxed => "deps-from-relaxed",
            Provenance::WorkspaceCondaFact(_) => "workspace-conda-fact",
            Provenance::UvRoot => "uv-root",
            Provenance::UvConstraint => "uv-constraint",
            Provenance::UvOverride => "uv-override",
            Provenance::PriorSelection => "prior-selection",
        };
        Constraint {
            specifiers: if specifiers.is_empty() {
                VersionSpecifiers::empty()
            } else {
                VersionSpecifiers::from_str(specifiers).unwrap()
            },
            provenance,
            source: source.to_string(),
            origin_id: ConstraintOriginId::from_parts(
                "constraint-unit-test",
                [provenance_tag, specifiers],
            ),
        }
    }

    fn package() -> PypiKey {
        PypiKey::from_pypi("example_package")
    }

    #[test]
    fn structured_origin_ids_preserve_component_boundaries_and_order() {
        let first = ConstraintOriginId::from_parts("wheel", ["a", "bc"]);
        let second = ConstraintOriginId::from_parts("wheel", ["ab", "c"]);
        let repeated = ConstraintOriginId::from_parts("wheel", ["a", "bc"]);
        let other_kind = ConstraintOriginId::from_parts("route", ["a", "bc"]);

        assert_eq!(first, repeated);
        assert_ne!(first, second);
        assert_ne!(first, other_kind);

        let lower = ConstraintOriginId::from_parts("wheel", ["a"]);
        let upper = ConstraintOriginId::from_parts("wheel", ["b"]);
        assert!(lower < upper);
    }

    #[test]
    fn authority_maps_every_provenance_variant() {
        assert_eq!(
            authority(&Provenance::IndexWheelMetadata),
            Authority::Authoritative
        );
        assert_eq!(
            authority(&Provenance::SourceBuiltRelaxed),
            Authority::Advisory
        );
        assert_eq!(authority(&Provenance::DepsFromRelaxed), Authority::Advisory);
        assert_eq!(
            authority(&Provenance::WorkspaceCondaFact("env".to_string())),
            Authority::Authoritative
        );
        assert_eq!(authority(&Provenance::UvRoot), Authority::Authoritative);
        assert_eq!(
            authority(&Provenance::UvConstraint),
            Authority::Authoritative
        );
        assert_eq!(authority(&Provenance::UvOverride), Authority::Authoritative);
        assert_eq!(
            authority(&Provenance::PriorSelection),
            Authority::Preference
        );
    }

    #[test]
    fn advisory_source_built_floor_yields_to_authoritative_cap() {
        let constraints = vec![
            constraint(
                ">=0.49.1,<0.50",
                Provenance::SourceBuiltRelaxed,
                "wheel `isaaclab==0.54.2` Requires-Dist `starlette>=0.49.1,<0.50`",
            ),
            constraint(
                ">=0.40.0,<0.46.0",
                Provenance::IndexWheelMetadata,
                "wheel `fastapi==0.115.7` Requires-Dist `starlette<0.46.0,>=0.40.0`",
            ),
        ];
        let softened = finalize(&PypiKey::from_pypi("starlette"), &constraints)
            .expect("advisory floor must soften");
        assert!(softened.contains(&Version::from_str("0.45.3").unwrap()));
        assert!(!softened.contains(&Version::from_str("0.49.1").unwrap()));
    }

    #[test]
    fn authoritative_only_conflict_still_errors() {
        let constraints = vec![
            constraint(
                ">=0.49.1,<0.50",
                Provenance::IndexWheelMetadata,
                "wheel `a==1` Requires-Dist `x>=0.49.1`",
            ),
            constraint(
                ">=0.40.0,<0.46.0",
                Provenance::UvConstraint,
                "wheel `b==1` Requires-Dist `x<0.46`",
            ),
        ];
        let error = finalize(&package(), &constraints).expect_err("must stay unsatisfiable");
        assert!(error.to_string().contains("mutually unsatisfiable"));
        assert!(error.sources.contains("wheel `a==1`"));
        assert!(error.sources.contains("wheel `b==1`"));
    }

    #[test]
    fn advisory_only_conflict_still_errors() {
        let constraints = vec![
            constraint(
                ">=2.3.5,<2.4",
                Provenance::SourceBuiltRelaxed,
                "wheel `pkg-a==1` Requires-Dist `x>=2.3.5,<2.4`",
            ),
            constraint(
                "<2,>=1.23.5",
                Provenance::DepsFromRelaxed,
                "wheel `pkg-b==1` Requires-Dist `x<2,>=1.23.5`",
            ),
        ];
        let error = finalize(&package(), &constraints)
            .expect_err("advisory-only conflict must stay unsatisfiable");
        assert!(error.to_string().contains("mutually unsatisfiable"));
    }

    #[test]
    fn override_replaces_ordinary_requirements_but_keeps_additive_constraints() {
        let constraints = vec![
            constraint("==1", Provenance::IndexWheelMetadata, "index metadata"),
            constraint(">=3", Provenance::SourceBuiltRelaxed, "source metadata"),
            constraint("<2", Provenance::UvRoot, "uv root"),
            constraint("==2", Provenance::UvOverride, "uv override"),
            constraint("<3", Provenance::UvConstraint, "uv constraint"),
            constraint(
                "==2.0",
                Provenance::WorkspaceCondaFact("test".to_string()),
                "workspace fact",
            ),
            constraint(">=2", Provenance::DepsFromRelaxed, "deps-from floor"),
            constraint("==99", Provenance::PriorSelection, "prior solve"),
        ];
        let finalized = finalize(&package(), &constraints).unwrap();
        assert!(finalized.contains(&Version::from_str("2").unwrap()));
        assert!(!finalized.contains(&Version::from_str("1").unwrap()));
        let rendered = finalized.to_string();
        assert!(!rendered.contains("==1"));
        assert!(!rendered.contains(">=3"));
        assert!(!rendered.contains("<2"));
        assert!(!rendered.contains("==99"));
    }

    #[test]
    fn prior_selection_is_never_promoted_to_a_hard_equality() {
        let finalized = finalize(
            &PypiKey::from_pypi("pillow"),
            &[constraint(
                "==12.3.0",
                Provenance::PriorSelection,
                "prior uv selection",
            )],
        )
        .unwrap();
        assert!(finalized.is_empty());
    }

    #[test]
    fn dedup_is_operator_sensitive() {
        let finalized = finalize(
            &package(),
            &[constraint(
                ">=11,>=11.0,<11.1,<11.1,~=11.0,~=11.0.0",
                Provenance::UvConstraint,
                "combined closure",
            )],
        )
        .unwrap();
        let clauses: Vec<String> = finalized.iter().map(ToString::to_string).collect();
        assert_eq!(clauses.iter().filter(|clause| *clause == ">=11").count(), 1);
        assert_eq!(
            clauses.iter().filter(|clause| *clause == ">=11.0").count(),
            0
        );
        assert_eq!(
            clauses.iter().filter(|clause| *clause == "<11.1").count(),
            1
        );
        assert!(clauses.iter().any(|clause| clause == "~=11.0"));
        assert!(clauses.iter().any(|clause| clause == "~=11.0.0"));
    }

    #[test]
    fn finalized_constraints_drop_semantically_redundant_bounds() {
        let constraints = vec![
            constraint(">=1.21", Provenance::IndexWheelMetadata, "wheel metadata"),
            constraint(">=2.1,<3", Provenance::UvConstraint, "route envelope"),
        ];
        assert_eq!(
            finalize(&package(), &constraints).unwrap(),
            VersionSpecifiers::from_str(">=2.1,<3").unwrap()
        );
    }

    #[test]
    fn mixed_authority_does_not_mask_an_internal_advisory_conflict() {
        let constraints = vec![
            constraint("", Provenance::UvConstraint, "unconstrained authority"),
            constraint(">=2,<2", Provenance::DepsFromRelaxed, "broken advisory"),
        ];
        finalize(&package(), &constraints).expect_err("advisory conflict must remain fatal");
    }

    #[test]
    fn advisory_caps_do_not_yield() {
        let constraints = vec![
            constraint(">=3", Provenance::UvConstraint, "authoritative floor"),
            constraint("<2", Provenance::SourceBuiltRelaxed, "advisory cap"),
        ];
        finalize(&package(), &constraints).expect_err("advisory cap must remain active");
    }

    #[test]
    fn request_collection_flattens_contextual_singletons_and_reports() {
        let first = finalize(
            &PypiKey::from_pypi("alpha"),
            &[
                constraint("<2", Provenance::UvRoot, "alpha root"),
                constraint(">=3", Provenance::UvConstraint, "alpha constraint"),
            ],
        )
        .unwrap_err();
        let second = finalize(
            &PypiKey::from_pypi("beta"),
            &[
                constraint("<4", Provenance::UvRoot, "beta root"),
                constraint(">=5", Provenance::UvConstraint, "beta constraint"),
            ],
        )
        .unwrap_err();
        let third = finalize(
            &PypiKey::from_pypi("gamma"),
            &[
                constraint("<6", Provenance::UvRoot, "gamma root"),
                constraint(">=7", Provenance::UvConstraint, "gamma constraint"),
            ],
        )
        .unwrap_err();

        let mut collected = Vec::new();
        collect_conflicts(
            anyhow::Error::new(first).context("resolving bundle alpha"),
            &mut collected,
        )
        .unwrap();
        collect_conflicts(
            aggregate_conflicts(vec![second, third]).context("resolving the remaining bundles"),
            &mut collected,
        )
        .unwrap();

        assert_eq!(
            collected
                .iter()
                .map(|conflict| conflict.package.as_str())
                .collect::<Vec<_>>(),
            ["alpha", "beta", "gamma"]
        );

        let unrelated = anyhow::anyhow!("network unavailable").context("fetching metadata");
        let unrelated = collect_conflicts(unrelated, &mut collected)
            .expect_err("non-conflict errors must propagate");
        assert_eq!(
            format!("{unrelated:#}"),
            "fetching metadata: network unavailable"
        );
    }
}
