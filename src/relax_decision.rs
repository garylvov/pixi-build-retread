//! Pure, strict-first decisions for resolving cross-origin version conflicts.
//!
//! [`crate::constraint::finalize`] remains the policy-neutral authority. This
//! module calls it first and, only for a strict conflict, searches explicit
//! clause-level relaxations permitted by the caller's policy. It never logs,
//! mutates caller-owned constraints, or drops a whole constraint edge.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fmt;
use std::ops::Bound;
use std::str::FromStr;

use uv_pep508::uv_pep440::{
    Operator, Version, VersionSpecifier, VersionSpecifiers, release_specifiers_to_ranges,
};

use crate::config::RelaxPolicy;
use crate::constraint::{
    Conflict, Constraint, ConstraintOriginId, FinalizeSuccess, Provenance, dedup_specifier_clauses,
    finalize_quiet, finalize_quiet_detailed,
};
use crate::relax::PypiKey;

const MAX_SEARCH_STATES: usize = 65_536;
const ADVISORY_FLOOR_DROP_DISTANCE: u32 = 4;

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
    AbiAnchorCapCompleted,
}

impl RelaxationKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::ExactPinWidened => "exact pin widened",
            Self::UpperCapStripped => "upper cap stripped",
            Self::AdvisoryFloorDropped => "advisory floor dropped",
            Self::AbiAnchorCapCompleted => "ABI anchor cap completed",
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
    SearchExhausted(SearchExhausted),
}

/// The bounded candidate search stopped with unexplored states remaining.
///
/// This is deliberately distinct from [`Decision::Conflict`]: the original
/// strict conflict remains actionable, but exhaustion does not prove that no
/// safe relaxation exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchExhausted {
    pub original_conflict: Conflict,
    pub searched_states: usize,
    pub limit: usize,
    pub pending_states: usize,
}

impl SearchExhausted {
    pub(crate) fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.original_conflict = self.original_conflict.with_scope(scope);
        self
    }
}

impl fmt::Display for SearchExhausted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "auto-relax search exhausted after {} candidate states (limit {}, {} pending); \
             this does not prove that no safe relaxation exists: {}",
            self.searched_states, self.limit, self.pending_states, self.original_conflict
        )
    }
}

impl std::error::Error for SearchExhausted {}

#[derive(Clone, Debug)]
enum CandidateEvaluation {
    Finalized(FinalizeSuccess),
    Conflict,
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

/// Monotonic explicit-loss prefix used as an admissible frontier bound.
///
/// Implicit advisory drops are deliberately excluded: descendants can remove
/// the need for those drops, so charging them in the frontier order would not
/// be monotonic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CandidateLowerBound {
    whole_edges_lost: u32,
    constraints_changed: u32,
    caps_or_exclusions_removed: u32,
    widening_distance: u32,
}

impl CandidateScore {
    const fn numeric(&self) -> CandidateLowerBound {
        CandidateLowerBound {
            whole_edges_lost: self.whole_edges_lost,
            constraints_changed: self.constraints_changed,
            caps_or_exclusions_removed: self.caps_or_exclusions_removed,
            widening_distance: self.widening_distance,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct StableActionKey {
    origin_id: ConstraintOriginId,
    clause: String,
    kind: RelaxationKind,
    replacement: Vec<String>,
}

#[derive(Clone, Debug)]
struct QueueEntry {
    lower_bound: CandidateLowerBound,
    score: CandidateScore,
    state: Vec<u8>,
    evaluation: CandidateEvaluation,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        (&self.lower_bound, &self.state) == (&other.lower_bound, &other.state)
    }
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        (&self.lower_bound, &self.state).cmp(&(&other.lower_bound, &other.state))
    }
}

#[derive(Clone, Debug)]
struct FinalizedCandidate {
    score: CandidateScore,
    state: Vec<u8>,
    success: FinalizeSuccess,
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
    decide_with_search_limit(package, constraints, policy, safety, MAX_SEARCH_STATES)
}

/// Decide against every constraint while omitting validation-only origins from
/// the successful emitted projection.
///
/// The omitted constraints remain authoritative during strict finalization,
/// candidate search, and ABI safety checks. This is used for precise workspace
/// facts that validate a portable emitted range without becoming that range.
pub(crate) fn decide_for_emission(
    package: &PypiKey,
    constraints: &[Constraint],
    validation_only_origins: &BTreeSet<ConstraintOriginId>,
    policy: RelaxPolicy,
    safety: &SafetyContext<'_>,
) -> Decision {
    decide_with_search_limit_and_projection(
        package,
        constraints,
        validation_only_origins,
        policy,
        safety,
        MAX_SEARCH_STATES,
    )
}

fn decide_with_search_limit(
    package: &PypiKey,
    constraints: &[Constraint],
    policy: RelaxPolicy,
    safety: &SafetyContext<'_>,
    search_limit: usize,
) -> Decision {
    decide_with_search_limit_and_projection(
        package,
        constraints,
        &BTreeSet::new(),
        policy,
        safety,
        search_limit,
    )
}

fn decide_with_search_limit_and_projection(
    package: &PypiKey,
    constraints: &[Constraint],
    validation_only_origins: &BTreeSet<ConstraintOriginId>,
    policy: RelaxPolicy,
    safety: &SafetyContext<'_>,
    search_limit: usize,
) -> Decision {
    let canonical = canonical_constraints(constraints);
    let is_anchor = safety.protects(package);
    let strict = match finalize_quiet_detailed(package, &canonical) {
        Ok(FinalizeSuccess::Unchanged(specifiers)) => {
            return Decision::Strict {
                specifiers: projected_specifiers(
                    package,
                    &canonical,
                    validation_only_origins,
                    specifiers,
                ),
                diagnostics: Vec::new(),
            };
        }
        Ok(FinalizeSuccess::AdvisoryFloorSoftened {
            specifiers,
            unsoftened_conflict,
        }) => {
            if is_anchor {
                // The legacy finalizer's advisory-floor fallback is not safe
                // for an ABI anchor, but another explicit candidate may be.
                // Preserve the unsoftened conflict as the fail-closed result
                // and let the same candidate-level major check decide.
                unsoftened_conflict
            } else {
                let diagnostics = advisory_floor_diagnostics(&canonical, &specifiers);
                return Decision::Strict {
                    specifiers: projected_specifiers(
                        package,
                        &canonical,
                        validation_only_origins,
                        specifiers,
                    ),
                    diagnostics,
                };
            }
        }
        Err(conflict) => conflict,
    };

    if policy == RelaxPolicy::None {
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
        package,
        &canonical,
        &atoms,
        &initial,
        &initial_core,
        &mut seen,
        &mut heap,
    );

    let mut searched = 0;
    let mut best = None;
    loop {
        if let Some(finalized) = best.as_ref()
            && frontier_cannot_beat(finalized, &heap)
        {
            return finalized_candidate_decision(
                package,
                &canonical,
                &atoms,
                finalized.clone(),
                validation_only_origins,
            );
        }
        if heap.is_empty() {
            return match best {
                Some(finalized) => finalized_candidate_decision(
                    package,
                    &canonical,
                    &atoms,
                    finalized,
                    validation_only_origins,
                ),
                None => Decision::Conflict(strict),
            };
        }
        if searched == search_limit {
            return Decision::SearchExhausted(SearchExhausted {
                original_conflict: strict,
                searched_states: searched,
                limit: search_limit,
                pending_states: heap.len(),
            });
        }

        let entry = match heap.pop() {
            Some(Reverse(entry)) => entry,
            None => {
                return match best {
                    Some(finalized) => finalized_candidate_decision(
                        package,
                        &canonical,
                        &atoms,
                        finalized,
                        validation_only_origins,
                    ),
                    None => Decision::Conflict(strict),
                };
            }
        };
        searched += 1;
        match entry.evaluation {
            CandidateEvaluation::Finalized(success) => {
                let validation_safe = workspace_validation_mutations_are_patch_exact(
                    &canonical,
                    &atoms,
                    &entry.state,
                    validation_only_origins,
                );
                let validation_result_safe = validation_only_origins.is_empty()
                    || matches!(&success, FinalizeSuccess::Unchanged(_));
                let safe = validation_safe
                    && validation_result_safe
                    && match &success {
                        FinalizeSuccess::Unchanged(specifiers) => {
                            !is_anchor
                                || anchor_candidate_stays_within_original_compatibility_band(
                                    &canonical,
                                    &atoms,
                                    &entry.state,
                                    specifiers,
                                )
                        }
                        FinalizeSuccess::AdvisoryFloorSoftened { .. } => !is_anchor,
                    };
                let softened = matches!(&success, FinalizeSuccess::AdvisoryFloorSoftened { .. });

                // A softened success may hide a lower-loss descendant whose
                // additional explicit widening makes the advisory floor
                // satisfiable. Unsafe anchor successes likewise need their
                // remaining successors explored before failing closed.
                if softened || !safe {
                    enqueue_all_mutations(
                        package,
                        &canonical,
                        &atoms,
                        &entry.state,
                        &mut seen,
                        &mut heap,
                    );
                }
                if safe {
                    retain_better_candidate(
                        &mut best,
                        FinalizedCandidate {
                            score: entry.score,
                            state: entry.state,
                            success,
                        },
                    );
                }
            }
            CandidateEvaluation::Conflict => {
                let core = minimal_unsatisfiable_core(package, &canonical, &atoms, &entry.state);
                enqueue_core_mutations(
                    package,
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
}

fn frontier_cannot_beat(best: &FinalizedCandidate, heap: &BinaryHeap<Reverse<QueueEntry>>) -> bool {
    match heap.peek() {
        None => true,
        // The explicit numeric score is an admissible, monotonic lower
        // bound for every descendant. Equal numeric scores must still be
        // visited so provenance and stable-action tie-breakers remain exact.
        Some(Reverse(next)) => next.lower_bound > best.score.numeric(),
    }
}

fn retain_better_candidate(best: &mut Option<FinalizedCandidate>, candidate: FinalizedCandidate) {
    let replaces = match best {
        Some(current) => (&candidate.score, &candidate.state) < (&current.score, &current.state),
        None => true,
    };
    if replaces {
        *best = Some(candidate);
    }
}

fn finalized_candidate_decision(
    package: &PypiKey,
    original: &[Constraint],
    atoms: &[Atom],
    finalized: FinalizedCandidate,
    validation_only_origins: &BTreeSet<ConstraintOriginId>,
) -> Decision {
    let relaxed = apply_state(original, atoms, &finalized.state);
    let mut decisions = selected_decisions(original, &relaxed, atoms, &finalized.state);
    let success = if validation_only_origins.is_empty() {
        finalized.success
    } else {
        let emitted = relaxed
            .iter()
            .filter(|constraint| !validation_only_origins.contains(&constraint.origin_id))
            .cloned()
            .collect::<Vec<_>>();
        finalize_quiet_detailed(package, &emitted)
            .expect("removing validation-only constraints cannot create an emission conflict")
    };
    match success {
        FinalizeSuccess::Unchanged(specifiers) => Decision::Relaxed {
            specifiers,
            decisions,
        },
        FinalizeSuccess::AdvisoryFloorSoftened { specifiers, .. } => {
            decisions.extend(advisory_floor_diagnostics(&relaxed, &specifiers));
            decisions.sort_by(|left, right| {
                (&left.origin_id, &left.original_clause, left.kind).cmp(&(
                    &right.origin_id,
                    &right.original_clause,
                    right.kind,
                ))
            });
            Decision::Relaxed {
                specifiers,
                decisions,
            }
        }
    }
}

fn projected_specifiers(
    package: &PypiKey,
    constraints: &[Constraint],
    validation_only_origins: &BTreeSet<ConstraintOriginId>,
    fallback: VersionSpecifiers,
) -> VersionSpecifiers {
    if validation_only_origins.is_empty() {
        return fallback;
    }
    let emitted = constraints
        .iter()
        .filter(|constraint| !validation_only_origins.contains(&constraint.origin_id))
        .cloned()
        .collect::<Vec<_>>();
    match finalize_quiet_detailed(package, &emitted)
        .expect("removing validation-only constraints cannot create an emission conflict")
    {
        FinalizeSuccess::Unchanged(specifiers)
        | FinalizeSuccess::AdvisoryFloorSoftened { specifiers, .. } => specifiers,
    }
}

/// A precise workspace ABI fact may justify only a patch-tier widening of an
/// authoritative wheel exact pin in the same epoch/major/minor line. Cap
/// stripping and broader exact tiers fail closed; the general anchor rule
/// intentionally remains same-major for other origin pairs.
fn workspace_validation_mutations_are_patch_exact(
    original: &[Constraint],
    atoms: &[Atom],
    state: &[u8],
    validation_only_origins: &BTreeSet<ConstraintOriginId>,
) -> bool {
    if validation_only_origins.is_empty() {
        return true;
    }
    let workspace_fact_versions = original
        .iter()
        .filter(|constraint| {
            validation_only_origins.contains(&constraint.origin_id)
                && matches!(&constraint.provenance, Provenance::WorkspaceCondaFact(_))
        })
        .flat_map(|constraint| constraint.specifiers.iter())
        .filter_map(|specifier| {
            (*specifier.operator() == Operator::Equal).then_some(specifier.version())
        })
        .collect::<Vec<_>>();
    let has_workspace_fact = original.iter().any(|constraint| {
        validation_only_origins.contains(&constraint.origin_id)
            && matches!(&constraint.provenance, Provenance::WorkspaceCondaFact(_))
    });
    if !has_workspace_fact {
        return false;
    }

    let selected = atoms
        .iter()
        .zip(state)
        .filter(|(_, choice_index)| **choice_index != 0)
        .map(|(atom, choice_index)| {
            (
                atom,
                &atom.choices[usize::from(*choice_index) - 1],
                &original[atom.constraint_index],
            )
        })
        .collect::<Vec<_>>();
    !selected.is_empty()
        && selected.into_iter().all(|(atom, choice, constraint)| {
            matches!(&constraint.provenance, Provenance::IndexWheelMetadata)
                && choice.kind == RelaxationKind::ExactPinWidened
                && choice.tier == RelaxPolicy::Patch
                && *atom.original.operator() == Operator::Equal
                && !workspace_fact_versions.is_empty()
                && workspace_fact_versions
                    .iter()
                    .all(|fact| same_epoch_major_minor(atom.original.version(), fact))
        })
}

fn same_epoch_major_minor(left: &Version, right: &Version) -> bool {
    let left_release = left.release();
    let right_release = right.release();
    left.epoch() == right.epoch()
        && left_release.first().copied().unwrap_or(0) == right_release.first().copied().unwrap_or(0)
        && left_release.get(1).copied().unwrap_or(0) == right_release.get(1).copied().unwrap_or(0)
}

/// ABI anchors may relax within one compatibility band, but a selected
/// candidate must neither admit another band nor move an original edge into a
/// band it never admitted. For ordinary releases the band is one major; for
/// `0.x` it is one `0.minor`, where SemVer permits breaking changes. Compare
/// effective PEP 440 ranges rather than the chosen policy tier: a `Major`
/// primitive can still be safe when another constraint confines the result.
fn anchor_candidate_stays_within_original_compatibility_band(
    original: &[Constraint],
    atoms: &[Atom],
    state: &[u8],
    result: &VersionSpecifiers,
) -> bool {
    let result_range = release_specifiers_to_ranges(result.clone());
    let Some((lower, _)) = result_range.bounding_range() else {
        return false;
    };
    let (compatibility_lower, compatibility_upper) = match lower {
        Bound::Included(version) | Bound::Excluded(version) => {
            let release = version.release();
            let Some(major) = release.first().copied() else {
                return false;
            };
            if major == 0 {
                let minor = release.get(1).copied().unwrap_or(0);
                let Some(next_minor) = minor.checked_add(1) else {
                    return false;
                };
                (Version::new([0, minor]), Version::new([0, next_minor]))
            } else {
                let Some(next_major) = major.checked_add(1) else {
                    return false;
                };
                (Version::new([major]), Version::new([next_major]))
            }
        }
        Bound::Unbounded => return false,
    };
    let compatibility_band: VersionSpecifiers = [
        VersionSpecifier::greater_than_equal_version(compatibility_lower),
        VersionSpecifier::less_than_version(compatibility_upper),
    ]
    .into_iter()
    .collect();
    let compatibility_range = release_specifiers_to_ranges(compatibility_band);
    if !result_range
        .intersection(&compatibility_range.complement())
        .is_empty()
    {
        return false;
    }

    let changed_constraints = atoms
        .iter()
        .zip(state)
        .filter_map(|(atom, &choice)| (choice != 0).then_some(atom.constraint_index))
        .collect::<BTreeSet<_>>();
    !changed_constraints.is_empty()
        && changed_constraints.into_iter().all(|constraint_index| {
            let original_range =
                release_specifiers_to_ranges(original[constraint_index].specifiers.clone());
            !original_range.intersection(&compatibility_range).is_empty()
        })
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
    let mut diagnostics = softened_advisory_constraints(constraints)
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

fn softened_advisory_constraints(constraints: &[Constraint]) -> impl Iterator<Item = &Constraint> {
    let has_override = constraints
        .iter()
        .any(|constraint| matches!(constraint.provenance, Provenance::UvOverride));
    constraints.iter().filter(move |constraint| {
        matches!(constraint.provenance, Provenance::DepsFromRelaxed)
            || (!has_override && matches!(constraint.provenance, Provenance::SourceBuiltRelaxed))
    })
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

fn score_state(
    constraints: &[Constraint],
    candidate: &[Constraint],
    atoms: &[Atom],
    state: &[u8],
    evaluation: &CandidateEvaluation,
) -> (CandidateLowerBound, CandidateScore) {
    let mut origins = BTreeSet::new();
    let mut provenance = Vec::new();
    let mut stable_actions = Vec::new();
    let mut whole_edges_lost = 0;
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

    let lower_bound = CandidateLowerBound {
        whole_edges_lost: 0,
        constraints_changed: origins.len() as u32,
        caps_or_exclusions_removed,
        widening_distance,
    };

    if matches!(
        evaluation,
        CandidateEvaluation::Finalized(FinalizeSuccess::AdvisoryFloorSoftened { .. })
    ) {
        for constraint in softened_advisory_constraints(candidate) {
            let dropped = constraint
                .specifiers
                .iter()
                .filter(|clause| advisory_operator_forces_floor(*clause.operator()))
                .collect::<Vec<_>>();
            if dropped.is_empty() {
                continue;
            }
            if dropped.len() == constraint.specifiers.len() {
                whole_edges_lost += 1;
            }
            origins.insert(constraint.origin_id.clone());
            for clause in dropped {
                provenance.push(provenance_rank(&constraint.provenance));
                // Dropping a lower bound admits an unbounded range beneath it,
                // which is strictly broader than any bounded exact-pin tier.
                widening_distance += ADVISORY_FLOOR_DROP_DISTANCE;
                stable_actions.push(StableActionKey {
                    origin_id: constraint.origin_id.clone(),
                    clause: clause.to_string(),
                    kind: RelaxationKind::AdvisoryFloorDropped,
                    replacement: Vec::new(),
                });
            }
        }
    }

    provenance.sort_unstable();
    stable_actions.sort();
    (
        lower_bound,
        CandidateScore {
            whole_edges_lost,
            constraints_changed: origins.len() as u32,
            caps_or_exclusions_removed,
            widening_distance,
            provenance_rank: provenance,
            stable_actions,
        },
    )
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
    package: &PypiKey,
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
        let relaxed = apply_state(constraints, atoms, &candidate);
        let evaluation = match finalize_quiet_detailed(package, &relaxed) {
            Ok(success) => CandidateEvaluation::Finalized(success),
            Err(_) => CandidateEvaluation::Conflict,
        };
        let (lower_bound, score) =
            score_state(constraints, &relaxed, atoms, &candidate, &evaluation);
        heap.push(Reverse(QueueEntry {
            lower_bound,
            score,
            state: candidate,
            evaluation,
        }));
    }
}

fn enqueue_all_mutations(
    package: &PypiKey,
    constraints: &[Constraint],
    atoms: &[Atom],
    state: &[u8],
    seen: &mut BTreeSet<Vec<u8>>,
    heap: &mut BinaryHeap<Reverse<QueueEntry>>,
) {
    let all_atoms = (0..atoms.len()).collect();
    enqueue_core_mutations(package, constraints, atoms, state, &all_atoms, seen, heap);
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
    fn epoch_exact_pin_conflict_keeps_strict_result() {
        let constraints = vec![
            constraint(
                "epoch-pin",
                "==1!3.1",
                Provenance::IndexWheelMetadata,
                "wheel epoch pin",
            ),
            constraint(
                "stable-range",
                ">=3,<4",
                Provenance::UvConstraint,
                "uv stable range",
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
    fn prerelease_exact_pin_conflict_keeps_strict_result() {
        let constraints = vec![
            constraint(
                "prerelease-pin",
                "==2.0.0a1",
                Provenance::IndexWheelMetadata,
                "wheel prerelease pin",
            ),
            constraint(
                "stable-range",
                ">=2,<3",
                Provenance::UvConstraint,
                "uv stable range",
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
    fn numpy_and_cuda_same_major_minor_conflicts_auto_resolve() {
        for (name, old, new, admitted, next_major) in [
            ("numpy", "==1.24", ">=1.25,<2", "1.25", "2"),
            ("cuda", "==12.4", ">=12.5,<13", "12.5", "13"),
        ] {
            let constraints = vec![
                constraint("old", old, Provenance::IndexWheelMetadata, "old wheel"),
                constraint("new", new, Provenance::UvConstraint, "uv range"),
            ];
            assert!(crate::solve::is_abi_anchor(name));
            let (specifiers, decisions) = relaxed(
                name,
                &constraints,
                RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
            );
            assert!(specifiers.contains(&Version::from_str(admitted).unwrap()));
            assert!(!specifiers.contains(&Version::from_str(next_major).unwrap()));
            assert_eq!(decisions.len(), 1);
            assert_eq!(decisions[0].tier, RelaxPolicy::Minor);
        }
    }

    #[test]
    fn numpy_workspace_patch_fact_emits_portable_patch_band() {
        let constraints = vec![
            constraint(
                "wheel-pin",
                "==1.26.0",
                Provenance::IndexWheelMetadata,
                "authoritative wheel pin",
            ),
            constraint(
                "workspace-fact",
                "==1.26.4",
                Provenance::WorkspaceCondaFact("precise-consuming-envs".to_string()),
                "precise workspace fact",
            ),
        ];
        let validation_only_origins = BTreeSet::from([constraints[1].origin_id.clone()]);
        let Decision::Relaxed {
            specifiers,
            decisions,
        } = decide_for_emission(
            &package("numpy"),
            &constraints,
            &validation_only_origins,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
            &SafetyContext::default(),
        )
        else {
            panic!("a same-minor workspace patch fact must relax the wheel pin")
        };

        assert_eq!(render_specifiers(&specifiers), ">=1.26.0,<1.27");
        assert!(specifiers.contains(&Version::from_str("1.26.4").unwrap()));
        assert!(!specifiers.contains(&Version::from_str("1.27").unwrap()));
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].tier, RelaxPolicy::Patch);
        assert_eq!(
            decisions[0].relaxed_clause.as_deref(),
            Some(">=1.26.0,<1.27")
        );
    }

    #[test]
    fn numpy_workspace_cross_minor_facts_fail_closed() {
        for package_name in ["numpy", "ordinary-package"] {
            for wheel_spec in ["==1.26.0", ">=1.26,<1.27"] {
                let constraints = vec![
                    constraint(
                        "wheel-constraint",
                        wheel_spec,
                        Provenance::IndexWheelMetadata,
                        "authoritative wheel constraint",
                    ),
                    constraint(
                        "workspace-fact",
                        "==1.27.0",
                        Provenance::WorkspaceCondaFact("precise-consuming-envs".to_string()),
                        "precise workspace fact",
                    ),
                ];
                let strict =
                    finalize_quiet(&package(package_name), &canonical_constraints(&constraints))
                        .unwrap_err();
                let validation_only_origins = BTreeSet::from([constraints[1].origin_id.clone()]);

                assert_eq!(
                    decide_for_emission(
                        &package(package_name),
                        &constraints,
                        &validation_only_origins,
                        RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
                        &SafetyContext::default(),
                    ),
                    Decision::Conflict(strict),
                    "a precise workspace fact must not reconcile `{wheel_spec}` across minors \
                     for `{package_name}`"
                );
            }
        }
    }

    #[test]
    fn workspace_validation_does_not_hide_advisory_floor_softening() {
        let constraints = vec![
            constraint(
                "wheel-pin",
                "==1.26.0",
                Provenance::IndexWheelMetadata,
                "authoritative wheel pin",
            ),
            constraint(
                "advisory-floor",
                ">=1.27",
                Provenance::SourceBuiltRelaxed,
                "source-built advisory floor",
            ),
            constraint(
                "workspace-fact",
                "==1.26.4",
                Provenance::WorkspaceCondaFact("precise-consuming-envs".to_string()),
                "precise workspace fact",
            ),
        ];
        let strict = finalize_quiet(
            &package("ordinary-package"),
            &canonical_constraints(&constraints),
        )
        .unwrap_err();
        let validation_only_origins = BTreeSet::from([constraints[2].origin_id.clone()]);

        assert_eq!(
            decide_for_emission(
                &package("ordinary-package"),
                &constraints,
                &validation_only_origins,
                RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
                &SafetyContext::default(),
            ),
            Decision::Conflict(strict),
            "workspace validation must not hide implicit advisory-floor loss"
        );
    }

    #[test]
    fn validation_projection_does_not_narrow_an_active_override() {
        let constraints = vec![
            constraint(
                "ignored-wheel-pin",
                "==1.25.0",
                Provenance::IndexWheelMetadata,
                "wheel pin ignored by override",
            ),
            constraint(
                "active-override",
                ">=1.26,<1.27",
                Provenance::UvOverride,
                "active UV override",
            ),
            constraint(
                "workspace-fact",
                "==1.26.4",
                Provenance::WorkspaceCondaFact("precise-consuming-envs".to_string()),
                "precise workspace fact",
            ),
        ];
        let validation_only_origins = BTreeSet::from([constraints[2].origin_id.clone()]);
        let Decision::Strict { specifiers, .. } = decide_for_emission(
            &package("numpy"),
            &constraints,
            &validation_only_origins,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
            &SafetyContext::default(),
        ) else {
            panic!("the active override and workspace fact are compatible")
        };
        assert_eq!(render_specifiers(&specifiers), ">=1.26,<1.27");
        assert!(specifiers.contains(&Version::from_str("1.26.5").unwrap()));
    }

    #[test]
    fn zero_major_abi_anchor_patch_conflict_relaxes_within_minor() {
        let constraints = vec![
            constraint(
                "old",
                "==0.2.1",
                Provenance::IndexWheelMetadata,
                "old wheel",
            ),
            constraint("new", ">=0.2.2,<0.3", Provenance::UvConstraint, "uv range"),
        ];
        let (specifiers, decisions) = relaxed(
            "numpy",
            &constraints,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
        );
        assert!(specifiers.contains(&Version::from_str("0.2.2").unwrap()));
        assert!(!specifiers.contains(&Version::from_str("0.3").unwrap()));
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].tier, RelaxPolicy::Patch);
    }

    #[test]
    fn zero_major_abi_anchor_cross_minor_conflicts_fail_closed() {
        for new in [">=0.3,<0.4", ">=0.3"] {
            let constraints = vec![
                constraint(
                    "old",
                    "==0.2.1",
                    Provenance::IndexWheelMetadata,
                    "old wheel",
                ),
                constraint("new", new, Provenance::UvConstraint, "uv range"),
            ];
            let strict = finalize_quiet(&package("numpy"), &canonical_constraints(&constraints))
                .unwrap_err();
            assert_eq!(
                decide(
                    &package("numpy"),
                    &constraints,
                    RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
                    &SafetyContext::default(),
                ),
                Decision::Conflict(strict),
                "0.x ABI anchors must not relax across a 0.minor boundary for `{new}`"
            );
        }
    }

    #[test]
    fn numpy_and_cuda_downward_minor_conflicts_allow_safe_major_primitive() {
        for (name, old, new, admitted, upper) in [
            ("numpy", "==1.26.4", ">=1.24,<1.25", "1.24.5", "1.25"),
            ("cuda", "==12.8", ">=12.4,<12.5", "12.4.5", "12.5"),
        ] {
            let constraints = vec![
                constraint(
                    "newer-pin",
                    old,
                    Provenance::IndexWheelMetadata,
                    "newer wheel",
                ),
                constraint("older-window", new, Provenance::UvConstraint, "uv range"),
            ];
            let (specifiers, decisions) = relaxed(
                name,
                &constraints,
                RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
            );
            assert!(specifiers.contains(&Version::from_str(admitted).unwrap()));
            assert!(!specifiers.contains(&Version::from_str(upper).unwrap()));
            assert_eq!(decisions.len(), 1);
            assert_eq!(decisions[0].tier, RelaxPolicy::Major);
        }
    }

    #[test]
    fn numpy_and_cuda_major_crossings_fail_closed() {
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
                "{name} must reject the major crossing for {old} versus {new}"
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
    fn implicit_advisory_floor_loss_is_scored_before_candidate_selection() {
        let constraints = vec![
            constraint(
                "exact-pin",
                "==1.26.4",
                Provenance::IndexWheelMetadata,
                "wheel exact pin",
            ),
            constraint(
                "stale-cap",
                ">=1,<1.26.4",
                Provenance::IndexWheelMetadata,
                "wheel stale cap",
            ),
            constraint(
                "advisory-floor",
                ">=1.26.4",
                Provenance::SourceBuiltRelaxed,
                "source-built floor",
            ),
        ];

        let forward = decide(
            &package("demo"),
            &constraints,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
            &SafetyContext::default(),
        );
        let mut reversed = constraints;
        reversed.reverse();
        let backward = decide(
            &package("demo"),
            &reversed,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
            &SafetyContext::default(),
        );
        assert_eq!(forward, backward);
        let Decision::Relaxed {
            specifiers,
            decisions,
        } = forward
        else {
            panic!("the lower-loss cap-strip candidate must resolve the conflict")
        };

        assert!(specifiers.contains(&Version::from_str("1.26.4").unwrap()));
        assert_eq!(decisions.len(), 1, "{decisions:?}");
        assert_eq!(decisions[0].kind, RelaxationKind::UpperCapStripped);
        assert_eq!(decisions[0].original_clause, "<1.26.4");
        assert!(
            decisions
                .iter()
                .all(|decision| decision.kind != RelaxationKind::AdvisoryFloorDropped),
            "the lower-loss cap strip must beat exact widening plus an implicit floor drop"
        );
    }

    #[test]
    fn softened_success_expands_to_a_lower_loss_unsoftened_descendant() {
        let constraints = vec![
            constraint(
                "mutable-pin",
                "==1.2.3",
                Provenance::IndexWheelMetadata,
                "wheel exact pin",
            ),
            constraint(
                "authoritative-range",
                ">=1.5,<3",
                Provenance::UvConstraint,
                "uv range",
            ),
            constraint(
                "advisory-range",
                ">=2,<3",
                Provenance::SourceBuiltRelaxed,
                "source-built range",
            ),
        ];

        let limited = decide_with_search_limit(
            &package("demo"),
            &constraints,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
            &SafetyContext::default(),
            2,
        );
        let Decision::SearchExhausted(exhausted) = limited else {
            panic!("a pending Major successor prevents accepting the provisional Minor result")
        };
        assert_eq!(exhausted.searched_states, 2);
        assert!(exhausted.pending_states > 0);

        let (specifiers, decisions) = relaxed(
            "demo",
            &constraints,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
        );

        assert!(specifiers.contains(&Version::from_str("2").unwrap()));
        assert!(!specifiers.contains(&Version::from_str("3").unwrap()));
        assert_eq!(decisions.len(), 1, "{decisions:?}");
        assert_eq!(decisions[0].kind, RelaxationKind::ExactPinWidened);
        assert_eq!(decisions[0].tier, RelaxPolicy::Major);
        assert_eq!(decisions[0].relaxed_clause.as_deref(), Some(">=1"));
        assert!(
            decisions
                .iter()
                .all(|decision| decision.kind != RelaxationKind::AdvisoryFloorDropped),
            "the Major successor must beat its softened Minor parent"
        );
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
    fn search_exhaustion_is_distinct_from_a_drained_conflict() {
        let constraints = (0..20)
            .map(|minor| {
                constraint(
                    &format!("pin-{minor}"),
                    &format!("==1.{minor}.0"),
                    Provenance::IndexWheelMetadata,
                    &format!("wheel pin {minor}"),
                )
            })
            .collect::<Vec<_>>();
        let canonical = canonical_constraints(&constraints);
        let atoms = build_atoms(
            &canonical,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
        );
        let all_minor = atoms
            .iter()
            .map(|atom| {
                atom.choices
                    .iter()
                    .position(|choice| choice.tier == RelaxPolicy::Minor)
                    .map(|index| (index + 1) as u8)
                    .expect("every exact pin has a minor candidate")
            })
            .collect::<Vec<_>>();
        let safe_candidate = apply_state(&canonical, &atoms, &all_minor);
        assert!(
            finalize_quiet(&package("many-pins"), &safe_candidate).is_ok(),
            "the high-cardinality fixture has a safe solution"
        );

        let decision = decide_with_search_limit(
            &package("many-pins"),
            &constraints,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
            &SafetyContext::default(),
            8,
        );
        let Decision::SearchExhausted(exhausted) = decision else {
            panic!("the bounded high-cardinality search must report exhaustion")
        };
        assert_eq!(exhausted.searched_states, 8);
        assert_eq!(exhausted.limit, 8);
        assert!(exhausted.pending_states > 0);
        assert!(
            exhausted
                .to_string()
                .contains("does not prove that no safe relaxation exists")
        );

        let hard_conflict = vec![
            constraint("hard-one", "==1", Provenance::UvRoot, "root pin"),
            constraint(
                "hard-two",
                "==2",
                Provenance::UvConstraint,
                "constraint pin",
            ),
        ];
        assert!(matches!(
            decide_with_search_limit(
                &package("hard-pins"),
                &hard_conflict,
                RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
                &SafetyContext::default(),
                8,
            ),
            Decision::Conflict(_)
        ));
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
