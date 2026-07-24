//! Pure, strict-first decisions for resolving cross-origin version conflicts.
//!
//! [`crate::constraint::finalize`] remains the policy-neutral authority. This
//! module calls it first and, only for a strict conflict, searches explicit
//! clause-level relaxations permitted by the caller's policy. It never logs,
//! mutates caller-owned constraints, or drops a whole constraint edge.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::str::FromStr;

use uv_pep508::uv_pep440::{Operator, Version, VersionSpecifier, VersionSpecifiers};

use crate::config::RelaxPolicy;
use crate::constraint::{
    Conflict, Constraint, ConstraintOriginId, FinalizeSuccess, Provenance, dedup_specifier_clauses,
    finalize_quiet, finalize_quiet_detailed,
};
use crate::relax::PypiKey;

const MAX_SEARCH_STATES: usize = 65_536;

/// Additional structured identity needed for safety checks.
///
/// PyPI keys are PEP-503-normalized, which can erase the underscore spelling
/// used by conda compiler packages. Emission therefore supplies its raw conda
/// name as well; restore-only callers may leave it absent.
#[derive(Clone, Copy, Debug, Default)]
pub struct SafetyContext<'a> {
    pub conda_name: Option<&'a str>,
    pub abi_anchor_alias: bool,
}

impl<'a> SafetyContext<'a> {
    pub const fn new(conda_name: Option<&'a str>) -> Self {
        Self {
            conda_name,
            abi_anchor_alias: false,
        }
    }

    pub const fn with_abi_anchor_alias(mut self, abi_anchor_alias: bool) -> Self {
        self.abi_anchor_alias = abi_anchor_alias;
        self
    }

    fn protects(self, package: &PypiKey) -> bool {
        crate::solve::is_abi_anchor(package.as_str())
            || self.conda_name.is_some_and(crate::solve::is_abi_anchor)
            || self.abi_anchor_alias
    }
}

/// The semantic operation selected for one original clause.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RelaxationKind {
    ExactPinWidened,
    UpperCapStripped,
    AdvisoryFloorDropped,
}

impl RelaxationKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::ExactPinWidened => "exact pin widened",
            Self::UpperCapStripped => "upper cap stripped",
            Self::AdvisoryFloorDropped => "advisory floor dropped",
        }
    }
}

/// One selected, source-rich clause transformation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelaxationDecision {
    pub origin_id: ConstraintOriginId,
    pub kind: RelaxationKind,
    pub original: String,
    pub relaxed: String,
    pub original_clause: String,
    pub relaxed_clause: Option<String>,
    pub source: String,
    pub tier: RelaxPolicy,
}

/// Result of one strict-first decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Strict {
        specifiers: VersionSpecifiers,
        diagnostics: Vec<RelaxationDecision>,
    },
    Relaxed {
        specifiers: VersionSpecifiers,
        decisions: Vec<RelaxationDecision>,
    },
    Conflict(Conflict),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct AtomId {
    origin_id: ConstraintOriginId,
    clause: String,
}

#[derive(Clone, Debug)]
struct Atom {
    id: AtomId,
    constraint_index: usize,
    original: VersionSpecifier,
    choices: Vec<Mutation>,
}

#[derive(Clone, Debug)]
struct Mutation {
    replacement: Vec<VersionSpecifier>,
    kind: RelaxationKind,
    tier: RelaxPolicy,
    caps_or_exclusions_removed: u32,
    widening_distance: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CandidateScore {
    whole_edges_lost: u32,
    constraints_changed: u32,
    caps_or_exclusions_removed: u32,
    widening_distance: u32,
    provenance_rank: Vec<u8>,
    stable_actions: Vec<StableActionKey>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct StableActionKey {
    origin_id: ConstraintOriginId,
    clause: String,
    kind: RelaxationKind,
    replacement: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct QueueEntry {
    score: CandidateScore,
    state: Vec<u8>,
}

/// Strictly finalize the input, then select the least-lossy safe relaxation
/// when the active policy allows one.
///
/// A returned [`Decision::Conflict`] is always the original strict conflict,
/// not a later search-state failure. Track 1 deliberately carries no
/// suggestion payload.
pub fn decide(
    package: &PypiKey,
    constraints: &[Constraint],
    policy: RelaxPolicy,
    safety: &SafetyContext<'_>,
) -> Decision {
    let canonical = canonical_constraints(constraints);
    let is_anchor = safety.protects(package);
    let strict = match finalize_quiet_detailed(package, &canonical) {
        Ok(FinalizeSuccess::Unchanged(specifiers)) => {
            return Decision::Strict {
                specifiers,
                diagnostics: Vec::new(),
            };
        }
        Ok(FinalizeSuccess::AdvisoryFloorSoftened {
            specifiers,
            unsoftened_conflict,
        }) => {
            if is_anchor {
                return Decision::Conflict(unsoftened_conflict);
            }
            let diagnostics = advisory_floor_diagnostics(&canonical, &specifiers);
            return Decision::Strict {
                specifiers,
                diagnostics,
            };
        }
        Err(conflict) => conflict,
    };

    if policy == RelaxPolicy::None || is_anchor {
        return Decision::Conflict(strict);
    }

    let atoms = build_atoms(&canonical, policy);
    if atoms.iter().all(|atom| atom.choices.is_empty()) {
        return Decision::Conflict(strict);
    }

    let initial = vec![0; atoms.len()];
    let initial_core = minimal_unsatisfiable_core(package, &canonical, &atoms, &initial);
    let mut heap = BinaryHeap::new();
    let mut seen = BTreeSet::from([initial.clone()]);
    enqueue_core_mutations(
        &canonical,
        &atoms,
        &initial,
        &initial_core,
        &mut seen,
        &mut heap,
    );

    let mut searched = 0;
    while let Some(Reverse(entry)) = heap.pop() {
        searched += 1;
        if searched > MAX_SEARCH_STATES {
            break;
        }
        let candidate = apply_state(&canonical, &atoms, &entry.state);
        match finalize_quiet_detailed(package, &candidate) {
            Ok(FinalizeSuccess::Unchanged(specifiers)) => {
                let decisions = selected_decisions(&canonical, &candidate, &atoms, &entry.state);
                return Decision::Relaxed {
                    specifiers,
                    decisions,
                };
            }
            Ok(FinalizeSuccess::AdvisoryFloorSoftened { specifiers, .. }) => {
                let mut decisions =
                    selected_decisions(&canonical, &candidate, &atoms, &entry.state);
                decisions.extend(advisory_floor_diagnostics(&candidate, &specifiers));
                decisions.sort_by(|left, right| {
                    (&left.origin_id, &left.original_clause, left.kind).cmp(&(
                        &right.origin_id,
                        &right.original_clause,
                        right.kind,
                    ))
                });
                return Decision::Relaxed {
                    specifiers,
                    decisions,
                };
            }
            Err(_) => {
                let core = minimal_unsatisfiable_core(package, &canonical, &atoms, &entry.state);
                enqueue_core_mutations(
                    &canonical,
                    &atoms,
                    &entry.state,
                    &core,
                    &mut seen,
                    &mut heap,
                );
            }
        }
    }

    Decision::Conflict(strict)
}

fn provenance_rank(provenance: &Provenance) -> u8 {
    match provenance {
        Provenance::IndexWheelMetadata => 0,
        Provenance::SourceBuiltRelaxed => 1,
        Provenance::DepsFromRelaxed => 2,
        Provenance::WorkspaceCondaFact(_) => 3,
        Provenance::UvRoot => 4,
        Provenance::UvConstraint => 5,
        Provenance::UvOverride => 6,
        Provenance::PriorSelection => 7,
    }
}

fn advisory_operator_forces_floor(operator: Operator) -> bool {
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

fn advisory_floor_diagnostics(
    constraints: &[Constraint],
    softened: &VersionSpecifiers,
) -> Vec<RelaxationDecision> {
    let relaxed = render_specifiers(softened);
    let mut diagnostics = constraints
        .iter()
        .filter(|constraint| {
            matches!(
                constraint.provenance,
                Provenance::SourceBuiltRelaxed | Provenance::DepsFromRelaxed
            )
        })
        .flat_map(|constraint| {
            let relaxed = relaxed.clone();
            constraint
                .specifiers
                .iter()
                .filter(|clause| advisory_operator_forces_floor(*clause.operator()))
                .map(move |clause| RelaxationDecision {
                    origin_id: constraint.origin_id.clone(),
                    kind: RelaxationKind::AdvisoryFloorDropped,
                    original: render_specifiers(&constraint.specifiers),
                    relaxed: relaxed.clone(),
                    original_clause: clause.to_string(),
                    relaxed_clause: None,
                    source: constraint.source.clone(),
                    tier: RelaxPolicy::None,
                })
        })
        .collect::<Vec<_>>();
    diagnostics.sort_by(|left, right| {
        (&left.origin_id, &left.original_clause).cmp(&(&right.origin_id, &right.original_clause))
    });
    diagnostics
}

fn render_specifiers(specifiers: &VersionSpecifiers) -> String {
    if specifiers.is_empty() {
        "*".to_string()
    } else {
        specifiers.to_string().replace(", ", ",")
    }
}

fn canonical_specifiers(clauses: impl IntoIterator<Item = VersionSpecifier>) -> VersionSpecifiers {
    dedup_specifier_clauses(clauses.into_iter())
}

fn canonical_constraints(constraints: &[Constraint]) -> Vec<Constraint> {
    let mut grouped =
        BTreeMap::<(ConstraintOriginId, Provenance, String), (Constraint, BTreeSet<String>)>::new();
    for mut constraint in constraints.iter().cloned() {
        constraint.specifiers = canonical_specifiers(constraint.specifiers.iter().cloned());
        let key = (
            constraint.origin_id.clone(),
            constraint.provenance.clone(),
            render_specifiers(&constraint.specifiers),
        );
        let source = constraint.source.clone();
        grouped
            .entry(key)
            .and_modify(|(_, sources)| {
                sources.insert(source.clone());
            })
            .or_insert_with(|| (constraint, BTreeSet::from([source])));
    }
    grouped
        .into_values()
        .map(|(mut constraint, sources)| {
            // Source prose is retained only as sorted diagnostics. It never
            // participates in candidate identity or selection order.
            constraint.source = sources.into_iter().collect::<Vec<_>>().join("; ");
            constraint
        })
        .collect()
}

#[derive(Clone, Copy)]
struct AllowedMutations {
    patch: bool,
    minor: bool,
    major: bool,
    strip_caps: bool,
}

fn allowed_mutations(policy: RelaxPolicy) -> AllowedMutations {
    use RelaxPolicy::{
        CondaAware, Major, MajorWithLastResort, Minor, MinorWithLastResort, None, Patch,
        PatchThenMinorThenMajorThenLastResort, PatchWithLastResort, StrongMajor,
    };

    match policy {
        None => AllowedMutations {
            patch: false,
            minor: false,
            major: false,
            strip_caps: false,
        },
        Patch => AllowedMutations {
            patch: true,
            minor: false,
            major: false,
            strip_caps: false,
        },
        Minor => AllowedMutations {
            patch: true,
            minor: true,
            major: false,
            strip_caps: false,
        },
        Major => AllowedMutations {
            patch: true,
            minor: true,
            major: true,
            strip_caps: false,
        },
        StrongMajor | CondaAware => AllowedMutations {
            patch: true,
            minor: true,
            major: true,
            strip_caps: true,
        },
        PatchWithLastResort
        | MinorWithLastResort
        | MajorWithLastResort
        | PatchThenMinorThenMajorThenLastResort => AllowedMutations {
            patch: true,
            minor: true,
            major: true,
            strip_caps: true,
        },
    }
}

fn build_atoms(constraints: &[Constraint], policy: RelaxPolicy) -> Vec<Atom> {
    let allowed = allowed_mutations(policy);
    let mut atoms = Vec::new();
    for (constraint_index, constraint) in constraints.iter().enumerate() {
        for original in constraint.specifiers.iter().cloned() {
            let choices = if matches!(
                &constraint.provenance,
                Provenance::IndexWheelMetadata
                    | Provenance::SourceBuiltRelaxed
                    | Provenance::DepsFromRelaxed
            ) {
                mutations_for(&original, allowed)
            } else {
                Vec::new()
            };
            atoms.push(Atom {
                id: AtomId {
                    origin_id: constraint.origin_id.clone(),
                    clause: original.to_string(),
                },
                constraint_index,
                original,
                choices,
            });
        }
    }
    atoms.sort_by(|left, right| left.id.cmp(&right.id));
    atoms
}

fn parse_replacement(raw: &str) -> Option<Vec<VersionSpecifier>> {
    VersionSpecifiers::from_str(raw)
        .ok()
        .map(|specifiers| specifiers.into_iter().collect())
}

fn exact_widening(version: &Version, tier: RelaxPolicy) -> Option<Vec<VersionSpecifier>> {
    crate::relax::widen_exact(version, tier)
        .as_deref()
        .and_then(parse_replacement)
}

fn mutations_for(original: &VersionSpecifier, allowed: AllowedMutations) -> Vec<Mutation> {
    let mut choices = Vec::new();
    match *original.operator() {
        Operator::Equal => {
            if allowed.patch
                && let Some(replacement) = exact_widening(original.version(), RelaxPolicy::Patch)
            {
                choices.push(Mutation {
                    replacement,
                    kind: RelaxationKind::ExactPinWidened,
                    tier: RelaxPolicy::Patch,
                    caps_or_exclusions_removed: 0,
                    widening_distance: 1,
                });
            }
            if allowed.minor
                && let Some(replacement) = exact_widening(original.version(), RelaxPolicy::Minor)
            {
                choices.push(Mutation {
                    replacement,
                    kind: RelaxationKind::ExactPinWidened,
                    tier: RelaxPolicy::Minor,
                    caps_or_exclusions_removed: 0,
                    widening_distance: 2,
                });
            }
            if allowed.major
                && let Some(replacement) = exact_widening(original.version(), RelaxPolicy::Major)
            {
                choices.push(Mutation {
                    replacement,
                    kind: RelaxationKind::ExactPinWidened,
                    tier: RelaxPolicy::Major,
                    caps_or_exclusions_removed: 0,
                    widening_distance: 3,
                });
            }
        }
        Operator::LessThan | Operator::LessThanEqual if allowed.strip_caps => {
            choices.push(Mutation {
                replacement: Vec::new(),
                kind: RelaxationKind::UpperCapStripped,
                tier: RelaxPolicy::StrongMajor,
                caps_or_exclusions_removed: 1,
                widening_distance: 0,
            });
        }
        Operator::TildeEqual if allowed.strip_caps => {
            if let Some(replacement) = parse_replacement(&format!(">={}", original.version())) {
                choices.push(Mutation {
                    replacement,
                    kind: RelaxationKind::UpperCapStripped,
                    tier: RelaxPolicy::StrongMajor,
                    caps_or_exclusions_removed: 1,
                    widening_distance: 0,
                });
            }
        }
        Operator::ExactEqual
        | Operator::EqualStar
        | Operator::NotEqual
        | Operator::NotEqualStar
        | Operator::GreaterThan
        | Operator::GreaterThanEqual
        | Operator::LessThan
        | Operator::LessThanEqual
        | Operator::TildeEqual => {}
    }
    choices
}

fn apply_state(constraints: &[Constraint], atoms: &[Atom], state: &[u8]) -> Vec<Constraint> {
    let mut clauses = constraints
        .iter()
        .map(|_| Vec::<VersionSpecifier>::new())
        .collect::<Vec<_>>();
    for (atom_index, atom) in atoms.iter().enumerate() {
        let choice = usize::from(state[atom_index]);
        if choice == 0 {
            clauses[atom.constraint_index].push(atom.original.clone());
        } else {
            clauses[atom.constraint_index]
                .extend(atom.choices[choice - 1].replacement.iter().cloned());
        }
    }
    constraints
        .iter()
        .cloned()
        .zip(clauses)
        .map(|(mut constraint, clauses)| {
            constraint.specifiers = canonical_specifiers(clauses);
            constraint
        })
        .collect()
}

fn state_loses_whole_edge(constraints: &[Constraint], atoms: &[Atom], state: &[u8]) -> bool {
    apply_state(constraints, atoms, state)
        .iter()
        .zip(constraints)
        .any(|(candidate, original)| {
            !original.specifiers.is_empty() && candidate.specifiers.is_empty()
        })
}

fn score_state(constraints: &[Constraint], atoms: &[Atom], state: &[u8]) -> CandidateScore {
    let mut origins = BTreeSet::new();
    let mut provenance = Vec::new();
    let mut stable_actions = Vec::new();
    let mut caps_or_exclusions_removed = 0;
    let mut widening_distance = 0;
    for (atom_index, &choice_index) in state.iter().enumerate() {
        if choice_index == 0 {
            continue;
        }
        let atom = &atoms[atom_index];
        let choice = &atom.choices[usize::from(choice_index) - 1];
        origins.insert(atom.id.origin_id.clone());
        provenance.push(provenance_rank(
            &constraints[atom.constraint_index].provenance,
        ));
        caps_or_exclusions_removed += choice.caps_or_exclusions_removed;
        widening_distance += choice.widening_distance;
        stable_actions.push(StableActionKey {
            origin_id: atom.id.origin_id.clone(),
            clause: atom.id.clause.clone(),
            kind: choice.kind,
            replacement: choice.replacement.iter().map(ToString::to_string).collect(),
        });
    }
    provenance.sort_unstable();
    stable_actions.sort();
    CandidateScore {
        whole_edges_lost: 0,
        constraints_changed: origins.len() as u32,
        caps_or_exclusions_removed,
        widening_distance,
        provenance_rank: provenance,
        stable_actions,
    }
}

fn minimal_unsatisfiable_core(
    package: &PypiKey,
    constraints: &[Constraint],
    atoms: &[Atom],
    state: &[u8],
) -> BTreeSet<usize> {
    let mut active = (0..atoms.len()).collect::<BTreeSet<_>>();
    for atom_index in 0..atoms.len() {
        let mut trial = active.clone();
        trial.remove(&atom_index);
        let candidate = apply_active_state(constraints, atoms, state, &trial);
        if finalize_quiet(package, &candidate).is_err() {
            active = trial;
        }
    }
    active
}

fn apply_active_state(
    constraints: &[Constraint],
    atoms: &[Atom],
    state: &[u8],
    active: &BTreeSet<usize>,
) -> Vec<Constraint> {
    let mut clauses = constraints
        .iter()
        .map(|_| Vec::<VersionSpecifier>::new())
        .collect::<Vec<_>>();
    for (atom_index, atom) in atoms.iter().enumerate() {
        if !active.contains(&atom_index) {
            continue;
        }
        let choice = usize::from(state[atom_index]);
        if choice == 0 {
            clauses[atom.constraint_index].push(atom.original.clone());
        } else {
            clauses[atom.constraint_index]
                .extend(atom.choices[choice - 1].replacement.iter().cloned());
        }
    }
    constraints
        .iter()
        .cloned()
        .zip(clauses)
        .map(|(mut constraint, clauses)| {
            constraint.specifiers = canonical_specifiers(clauses);
            constraint
        })
        .collect()
}

fn enqueue_core_mutations(
    constraints: &[Constraint],
    atoms: &[Atom],
    state: &[u8],
    core: &BTreeSet<usize>,
    seen: &mut BTreeSet<Vec<u8>>,
    heap: &mut BinaryHeap<Reverse<QueueEntry>>,
) {
    for &atom_index in core {
        let next = usize::from(state[atom_index]) + 1;
        if next > atoms[atom_index].choices.len() {
            continue;
        }
        let mut candidate = state.to_vec();
        candidate[atom_index] = next as u8;
        if state_loses_whole_edge(constraints, atoms, &candidate) || !seen.insert(candidate.clone())
        {
            continue;
        }
        heap.push(Reverse(QueueEntry {
            score: score_state(constraints, atoms, &candidate),
            state: candidate,
        }));
    }
}

fn selected_decisions(
    original: &[Constraint],
    relaxed: &[Constraint],
    atoms: &[Atom],
    state: &[u8],
) -> Vec<RelaxationDecision> {
    let mut decisions = state
        .iter()
        .enumerate()
        .filter_map(|(atom_index, &choice_index)| {
            if choice_index == 0 {
                return None;
            }
            let atom = &atoms[atom_index];
            let choice = &atom.choices[usize::from(choice_index) - 1];
            let constraint = &original[atom.constraint_index];
            Some(RelaxationDecision {
                origin_id: atom.id.origin_id.clone(),
                kind: choice.kind,
                original: render_specifiers(&constraint.specifiers),
                relaxed: render_specifiers(&relaxed[atom.constraint_index].specifiers),
                original_clause: atom.original.to_string(),
                relaxed_clause: (!choice.replacement.is_empty()).then(|| {
                    choice
                        .replacement
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                }),
                source: constraint.source.clone(),
                tier: choice.tier,
            })
        })
        .collect::<Vec<_>>();
    decisions.sort_by(|left, right| {
        (
            &left.origin_id,
            &left.original_clause,
            left.kind,
            left.relaxed_clause.as_deref(),
        )
            .cmp(&(
                &right.origin_id,
                &right.original_clause,
                right.kind,
                right.relaxed_clause.as_deref(),
            ))
    });
    decisions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constraint(
        origin: &str,
        specifiers: &str,
        provenance: Provenance,
        source: &str,
    ) -> Constraint {
        let specifiers = VersionSpecifiers::from_str(specifiers).unwrap();
        let canonical_specifiers = render_specifiers(&specifiers);
        Constraint {
            specifiers,
            provenance,
            source: source.to_string(),
            origin_id: ConstraintOriginId::from_parts(
                "relax-decision-test",
                [origin, canonical_specifiers.as_str()],
            ),
        }
    }

    fn package(name: &str) -> PypiKey {
        PypiKey::from_pypi(name)
    }

    fn relaxed(
        package_name: &str,
        constraints: &[Constraint],
        policy: RelaxPolicy,
    ) -> (VersionSpecifiers, Vec<RelaxationDecision>) {
        match decide(
            &package(package_name),
            constraints,
            policy,
            &SafetyContext::default(),
        ) {
            Decision::Relaxed {
                specifiers,
                decisions,
            } => (specifiers, decisions),
            other => panic!("expected a relaxed decision, got {other:?}"),
        }
    }

    #[test]
    fn strict_success_never_relaxes_a_satisfiable_pin() {
        let constraints = vec![
            constraint(
                "pin",
                "==1.2.3",
                Provenance::IndexWheelMetadata,
                "wheel pin",
            ),
            constraint(
                "floor",
                ">=1",
                Provenance::IndexWheelMetadata,
                "wheel floor",
            ),
        ];
        assert!(matches!(
            decide(
                &package("demo"),
                &constraints,
                RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
                &SafetyContext::default(),
            ),
            Decision::Strict { .. }
        ));
    }

    #[test]
    fn patch_conflict_relaxes_only_the_exact_clause() {
        let constraints = vec![
            constraint(
                "old-pin",
                "==1.2.3",
                Provenance::IndexWheelMetadata,
                "wheel old",
            ),
            constraint(
                "new-floor",
                ">=1.2.4,<1.3",
                Provenance::UvConstraint,
                "uv floor",
            ),
        ];
        let (specifiers, decisions) = relaxed(
            "demo",
            &constraints,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
        );
        assert!(specifiers.contains(&Version::from_str("1.2.4").unwrap()));
        assert!(!specifiers.contains(&Version::from_str("1.3").unwrap()));
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].kind, RelaxationKind::ExactPinWidened);
        assert_eq!(decisions[0].tier, RelaxPolicy::Patch);
        assert_eq!(decisions[0].original_clause, "==1.2.3");
        assert_eq!(decisions[0].relaxed_clause.as_deref(), Some(">=1.2.3,<1.3"));
    }

    #[test]
    fn semantic_aliases_share_one_clause_and_one_decision() {
        let constraints = vec![
            constraint(
                "old-pin",
                "==1,==1.0",
                Provenance::IndexWheelMetadata,
                "wheel old",
            ),
            constraint(
                "new-floor",
                ">=1.1,<2",
                Provenance::UvConstraint,
                "uv floor",
            ),
        ];
        let (_, decisions) = relaxed(
            "demo",
            &constraints,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
        );
        assert_eq!(decisions.len(), 1, "{decisions:?}");
        assert_eq!(decisions[0].kind, RelaxationKind::ExactPinWidened);
    }

    #[test]
    fn minor_crossing_within_the_same_major_auto_resolves() {
        let constraints = vec![
            constraint(
                "old-pin",
                "==1.2.3",
                Provenance::IndexWheelMetadata,
                "wheel old",
            ),
            constraint(
                "new-floor",
                ">=1.3,<2",
                Provenance::UvConstraint,
                "uv floor",
            ),
        ];
        let (specifiers, decisions) = relaxed(
            "demo",
            &constraints,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
        );
        assert!(specifiers.contains(&Version::from_str("1.3").unwrap()));
        assert!(!specifiers.contains(&Version::from_str("2").unwrap()));
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].kind, RelaxationKind::ExactPinWidened);
        assert_eq!(decisions[0].tier, RelaxPolicy::Minor);
        assert_eq!(decisions[0].relaxed_clause.as_deref(), Some(">=1.2,<2"));
    }

    #[test]
    fn exact_widening_candidates_match_landed_tier_lower_bounds() {
        let exact = VersionSpecifier::from_str("==1.26.4").unwrap();
        let choices = mutations_for(
            &exact,
            AllowedMutations {
                patch: true,
                minor: true,
                major: true,
                strip_caps: false,
            },
        );
        let rendered = choices
            .iter()
            .map(|choice| {
                (
                    choice.tier,
                    render_specifiers(&canonical_specifiers(choice.replacement.iter().cloned())),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            rendered,
            vec![
                (RelaxPolicy::Patch, ">=1.26.4,<1.27".to_string()),
                (RelaxPolicy::Minor, ">=1.26,<2".to_string()),
                (RelaxPolicy::Major, ">=1".to_string()),
            ]
        );
    }

    #[test]
    fn downward_patch_conflict_auto_resolves_at_minor_tier() {
        let constraints = vec![
            constraint(
                "newer-pin",
                "==1.26.4",
                Provenance::IndexWheelMetadata,
                "newer wheel",
            ),
            constraint(
                "older-patch-window",
                ">=1.26,<1.26.4",
                Provenance::UvConstraint,
                "uv range",
            ),
        ];
        let (specifiers, decisions) = relaxed(
            "demo",
            &constraints,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
        );
        assert!(specifiers.contains(&Version::from_str("1.26.3").unwrap()));
        assert!(!specifiers.contains(&Version::from_str("1.26.4").unwrap()));
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].tier, RelaxPolicy::Minor);
        assert_eq!(decisions[0].relaxed_clause.as_deref(), Some(">=1.26,<2"));
    }

    #[test]
    fn downward_minor_conflict_auto_resolves_within_major() {
        let constraints = vec![
            constraint(
                "newer-pin",
                "==1.26.4",
                Provenance::IndexWheelMetadata,
                "newer wheel",
            ),
            constraint(
                "older-minor-window",
                ">=1.24,<1.25",
                Provenance::UvConstraint,
                "uv range",
            ),
        ];
        let (specifiers, decisions) = relaxed(
            "demo",
            &constraints,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
        );
        assert!(specifiers.contains(&Version::from_str("1.24.5").unwrap()));
        assert!(!specifiers.contains(&Version::from_str("1.25").unwrap()));
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].tier, RelaxPolicy::Major);
        assert_eq!(decisions[0].relaxed_clause.as_deref(), Some(">=1"));
    }

    #[test]
    fn stale_cap_strips_only_the_conflicting_clause() {
        let constraints = vec![
            constraint(
                "stale-range",
                ">=1,<2,<99",
                Provenance::IndexWheelMetadata,
                "wheel stale",
            ),
            constraint(
                "current-range",
                ">=2.1,<3",
                Provenance::UvConstraint,
                "uv current",
            ),
        ];
        let (specifiers, decisions) = relaxed(
            "demo",
            &constraints,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
        );
        assert!(specifiers.contains(&Version::from_str("2.1").unwrap()));
        assert!(!specifiers.contains(&Version::from_str("3").unwrap()));
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].kind, RelaxationKind::UpperCapStripped);
        assert_eq!(decisions[0].original_clause, "<2");
        assert_eq!(decisions[0].relaxed_clause, None);
        assert!(
            decisions[0].relaxed.contains("<99"),
            "the non-conflicting cap must remain on its origin: {:?}",
            decisions[0]
        );
    }

    #[test]
    fn load_bearing_advisory_cap_is_not_dropped() {
        let constraints = vec![
            constraint("hard-floor", ">=3", Provenance::UvConstraint, "uv floor"),
            constraint(
                "advisory-cap",
                "<2",
                Provenance::SourceBuiltRelaxed,
                "source wheel cap",
            ),
        ];
        let strict =
            finalize_quiet(&package("demo"), &canonical_constraints(&constraints)).unwrap_err();
        assert_eq!(
            decide(
                &package("demo"),
                &constraints,
                RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
                &SafetyContext::default(),
            ),
            Decision::Conflict(strict)
        );
    }

    #[test]
    fn relax_none_returns_the_original_strict_conflict() {
        let constraints = vec![
            constraint("old", "==1", Provenance::IndexWheelMetadata, "wheel old"),
            constraint("new", ">=2", Provenance::IndexWheelMetadata, "wheel new"),
        ];
        let strict =
            finalize_quiet(&package("demo"), &canonical_constraints(&constraints)).unwrap_err();
        assert_eq!(
            decide(
                &package("demo"),
                &constraints,
                RelaxPolicy::None,
                &SafetyContext::default(),
            ),
            Decision::Conflict(strict)
        );
    }

    #[test]
    fn numpy_and_cuda_anchors_are_protected_from_every_mutation() {
        for (name, old, new) in [
            ("numpy", "==1.26.4", ">=2"),
            ("numpy", ">=1,<2", ">=2,<3"),
            ("cuda", "==12.8", ">=13"),
            ("cuda", ">=12,<13", ">=13,<14"),
        ] {
            let constraints = vec![
                constraint("old", old, Provenance::IndexWheelMetadata, "old wheel"),
                constraint("new", new, Provenance::IndexWheelMetadata, "new wheel"),
            ];
            assert!(crate::solve::is_abi_anchor(name));
            let strict =
                finalize_quiet(&package(name), &canonical_constraints(&constraints)).unwrap_err();
            assert_eq!(
                decide(
                    &package(name),
                    &constraints,
                    RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
                    &SafetyContext::default(),
                ),
                Decision::Conflict(strict),
                "{name} must retain its strict conflict for {old} versus {new}"
            );
        }
    }

    #[test]
    fn abi_anchor_rejects_legacy_advisory_floor_softening() {
        for (package_name, safety_name) in [("numpy", None), ("array-provider", Some("cuda"))] {
            let constraints = vec![
                constraint(
                    "advisory-floor",
                    ">=2",
                    Provenance::SourceBuiltRelaxed,
                    "source-built ABI floor",
                ),
                constraint(
                    "workspace-cap",
                    "<2",
                    Provenance::UvConstraint,
                    "workspace ABI cap",
                ),
            ];
            let legacy = crate::constraint::finalize(&package(package_name), &constraints).unwrap();
            assert!(legacy.contains(&Version::from_str("1.26.4").unwrap()));
            assert!(!legacy.contains(&Version::from_str("2").unwrap()));

            let decision = decide(
                &package(package_name),
                &constraints,
                RelaxPolicy::StrongMajor,
                &SafetyContext::new(safety_name),
            );
            let Decision::Conflict(conflict) = decision else {
                panic!(
                    "{package_name}/{safety_name:?} must fail closed when strict success drops an anchor clause"
                )
            };
            assert!(
                conflict.sources.contains("source-built ABI floor"),
                "{conflict}"
            );
            assert!(conflict.sources.contains("workspace ABI cap"), "{conflict}");
        }
    }

    #[test]
    fn non_anchor_advisory_floor_softening_remains_a_structured_warning() {
        let constraints = vec![
            constraint(
                "advisory-floor",
                ">=2",
                Provenance::SourceBuiltRelaxed,
                "source-built floor",
            ),
            constraint(
                "authoritative-cap",
                "<2",
                Provenance::UvConstraint,
                "workspace cap",
            ),
        ];
        let Decision::Strict {
            specifiers,
            diagnostics,
        } = decide(
            &package("ordinary-package"),
            &constraints,
            RelaxPolicy::None,
            &SafetyContext::default(),
        )
        else {
            panic!("legacy non-anchor authority softening must remain policy-neutral")
        };
        assert!(specifiers.contains(&Version::from_str("1.9").unwrap()));
        assert!(!specifiers.contains(&Version::from_str("2").unwrap()));
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].kind, RelaxationKind::AdvisoryFloorDropped);
        assert_eq!(diagnostics[0].original_clause, ">=2");
        assert_eq!(diagnostics[0].tier, RelaxPolicy::None);
    }

    #[test]
    fn non_anchor_advisory_stale_cap_can_relax_without_losing_its_edge() {
        let constraints = vec![
            constraint(
                "advisory-range",
                ">=1,<2",
                Provenance::SourceBuiltRelaxed,
                "source-built range",
            ),
            constraint(
                "current-range",
                ">=2.1,<3",
                Provenance::UvConstraint,
                "uv current",
            ),
        ];
        let (specifiers, decisions) = relaxed(
            "demo",
            &constraints,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
        );
        assert!(specifiers.contains(&Version::from_str("2.1").unwrap()));
        assert!(!specifiers.contains(&Version::from_str("3").unwrap()));
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].kind, RelaxationKind::UpperCapStripped);
        assert_eq!(decisions[0].original_clause, "<2");
        assert!(decisions[0].relaxed.contains(">=1"));
    }

    #[test]
    fn equal_authority_drop_only_conflict_fails_closed_in_every_order() {
        let constraints = vec![
            constraint("old-cap", "<4", Provenance::IndexWheelMetadata, "wheel z"),
            constraint(
                "new-range",
                ">=5,<6",
                Provenance::IndexWheelMetadata,
                "wheel a",
            ),
        ];
        let strict = finalize_quiet(
            &package("typing-extensions"),
            &canonical_constraints(&constraints),
        )
        .unwrap_err();
        let forward = decide(
            &package("typing-extensions"),
            &constraints,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
            &SafetyContext::default(),
        );
        let mut reversed = constraints;
        reversed.reverse();
        let backward = decide(
            &package("typing-extensions"),
            &reversed,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
            &SafetyContext::default(),
        );
        assert_eq!(forward, Decision::Conflict(strict));
        assert_eq!(forward, backward);
    }

    #[test]
    fn candidate_score_uses_the_agreed_semantic_loss_order() {
        let origin = ConstraintOriginId::from_parts("score-test", ["origin"]);
        let action = StableActionKey {
            origin_id: origin,
            clause: "<2".to_string(),
            kind: RelaxationKind::UpperCapStripped,
            replacement: Vec::new(),
        };
        let score = |whole_edges_lost,
                     constraints_changed,
                     caps_or_exclusions_removed,
                     widening_distance,
                     provenance_rank,
                     stable_actions| CandidateScore {
            whole_edges_lost,
            constraints_changed,
            caps_or_exclusions_removed,
            widening_distance,
            provenance_rank,
            stable_actions,
        };

        assert!(
            score(0, 99, 99, 99, vec![99], vec![action.clone()])
                < score(1, 0, 0, 0, vec![0], Vec::new())
        );
        assert!(
            score(0, 0, 99, 99, vec![99], vec![action.clone()])
                < score(0, 1, 0, 0, vec![0], Vec::new())
        );
        assert!(
            score(0, 0, 0, 99, vec![99], vec![action.clone()])
                < score(0, 0, 1, 0, vec![0], Vec::new())
        );
        assert!(
            score(0, 0, 0, 0, vec![99], vec![action.clone()])
                < score(0, 0, 0, 1, vec![0], Vec::new())
        );
        assert!(
            score(0, 0, 0, 0, vec![0], vec![action.clone()])
                < score(0, 0, 0, 0, vec![1], Vec::new())
        );
        assert!(score(0, 0, 0, 0, vec![0], Vec::new()) < score(0, 0, 0, 0, vec![0], vec![action]));
    }

    #[test]
    fn wheel_and_clause_permutations_select_the_same_total_order() {
        let forward = vec![
            constraint(
                "z-origin",
                "<2,>=1,<99",
                Provenance::IndexWheelMetadata,
                "human source a",
            ),
            constraint(
                "a-origin",
                "<3,>=2.1",
                Provenance::UvConstraint,
                "human source z",
            ),
        ];
        let mut reversed = vec![
            constraint(
                "z-origin",
                "<99,>=1,<2",
                Provenance::IndexWheelMetadata,
                "human source a",
            ),
            constraint(
                "a-origin",
                ">=2.1,<3",
                Provenance::UvConstraint,
                "human source z",
            ),
        ];
        reversed.reverse();

        let first = decide(
            &package("demo"),
            &forward,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
            &SafetyContext::default(),
        );
        let second = decide(
            &package("demo"),
            &reversed,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
            &SafetyContext::default(),
        );
        assert_eq!(first, second);
        let Decision::Relaxed { decisions, .. } = first else {
            panic!("permutation fixture must relax")
        };
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].original_clause, "<2");
        assert!(
            decisions[0].origin_id.as_str().contains("z-origin"),
            "stable origin, not human source order, identifies the change"
        );
    }
}
