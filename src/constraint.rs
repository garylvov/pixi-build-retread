//! Typed constraint provenance and the single constraint finalization path.
//!
//! Provenance is retained until an emitted or restored requirement is fully
//! assembled. Authority is deliberately derived only by [`authority`], so a
//! new source of constraints cannot acquire independent hard/soft semantics.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use uv_pep508::uv_pep440::{
    Operator, Version, VersionSpecifier, VersionSpecifiers, release_specifiers_to_ranges,
};

use crate::relax::PypiKey;

/// Where a constraint entered the composed dependency graph.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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

/// One version constraint with enough provenance to determine its authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    pub specifiers: VersionSpecifiers,
    pub provenance: Provenance,
    pub source: String,
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
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error(
    "dependency conflict{scope}: `{package}` requirements are mutually unsatisfiable: {sources}. \
     Resolve by pinning one side, or use `retread-relax`, `retread-overrides`, or \
     `retread-drop-deps` in the pack manifest (see README)."
)]
pub struct Conflict {
    pub package: PypiKey,
    pub sources: String,
    scope: String,
}

impl Conflict {
    /// Add a user-facing solve scope such as an environment, target profile,
    /// platform, Python version, and bundle identity.
    pub(crate) fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = format!(" {}", scope.into());
        self
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
        (&left.package, &left.sources, &left.scope).cmp(&(
            &right.package,
            &right.sources,
            &right.scope,
        ))
    });
    if conflicts.len() == 1 {
        anyhow::Error::new(conflicts.pop().expect("one conflict"))
    } else {
        anyhow::Error::new(ConflictReport { conflicts })
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
fn dedup_specifier_clauses(clauses: impl Iterator<Item = VersionSpecifier>) -> VersionSpecifiers {
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
fn specifiers_unsatisfiable(specifiers: &VersionSpecifiers) -> bool {
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
        return Ok(remove_redundant_specifier_clauses(combined));
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
            tracing::warn!(
                package = %package,
                softened = %softened,
                "an advisory lower bound conflicted with an authoritative constraint; \
                 dropped the advisory floor (conda-as-truth)",
            );
            return Ok(remove_redundant_specifier_clauses(softened));
        }
    }

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
    Err(Conflict {
        package: package.clone(),
        sources,
        scope: String::new(),
    })
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    fn constraint(specifiers: &str, provenance: Provenance, source: &str) -> Constraint {
        Constraint {
            specifiers: if specifiers.is_empty() {
                VersionSpecifiers::empty()
            } else {
                VersionSpecifiers::from_str(specifiers).unwrap()
            },
            provenance,
            source: source.to_string(),
        }
    }

    fn package() -> PypiKey {
        PypiKey::from_pypi("example_package")
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
}
