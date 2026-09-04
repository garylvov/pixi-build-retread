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

/// Whether a constraint participates in finalization when an override exists.
///
/// Keep callers that decide whether to add validation inputs aligned with the
/// active projection used by [`finalize_impl`].
pub(crate) fn active_for_finalization(constraint: &Constraint, has_override: bool) -> bool {
    !has_override
        || matches!(
            &constraint.provenance,
            Provenance::UvOverride
                | Provenance::UvConstraint
                | Provenance::WorkspaceCondaFact(_)
                | Provenance::DepsFromRelaxed
        )
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

    /// The kind and semantic components this origin was built from, recovered
    /// from the length-prefixed encoding.
    ///
    /// Emission policy has to tell "this bound came from the pack's own uv
    /// closure" apart from "this bound came from a wheel's `Requires-Dist`",
    /// and [`Constraint::source`] is diagnostic prose that must never be
    /// matched on. The parts are the structured half of the identity, so they
    /// are the half a policy decision may read.
    pub fn parts(&self) -> Vec<&str> {
        let mut parts = Vec::new();
        let mut rest = self.0.as_str();
        while !rest.is_empty() {
            let Some((len, tail)) = rest.split_once(':') else {
                break;
            };
            let Ok(len) = len.parse::<usize>() else {
                break;
            };
            let Some(part) = tail.get(..len) else {
                break;
            };
            parts.push(part);
            rest = &tail[len..];
        }
        parts
    }

    /// The stable KIND this origin was constructed with.
    pub fn kind(&self) -> Option<&str> {
        self.parts().into_iter().next()
    }

    /// Whether one of this origin's semantic components is exactly `part`.
    pub fn has_part(&self, part: &str) -> bool {
        self.parts().contains(&part)
    }
}

#[cfg(test)]
mod origin_id_parts_tests {
    use super::ConstraintOriginId;

    #[test]
    fn parts_round_trip_the_length_prefixed_encoding() {
        let id = ConstraintOriginId::from_parts("auto-route", ["fsspec", "2024.6.1", ":weird:5"]);
        assert_eq!(id.kind(), Some("auto-route"));
        assert_eq!(
            id.parts(),
            vec!["auto-route", "fsspec", "2024.6.1", ":weird:5"],
        );
        assert!(id.has_part("2024.6.1"));
        // A component that merely APPEARS inside another component's text is
        // not a part: this is why policy reads `has_part`, never `contains`.
        assert!(!id.has_part("weird"));
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

/// A merge-oriented manifest edit attached to one fail-closed conflict.
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
        kind: SoftenKind,
        unsoftened_conflict: Conflict,
    },
}

/// Which policy let a finalization succeed after discarding a clause.
///
/// Both are already-decided policy; the distinction is what the WARN row says
/// and what a post-hoc audit can separate. `AdvisoryFloor` is the long-standing
/// conda-as-truth rule. `LearnedWorkspaceFact` is p6z's reader for a policy
/// that already had two writers and no reader here -- see
/// [`finalize_impl`]'s learned-fact step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SoftenKind {
    AdvisoryFloor,
    LearnedWorkspaceFact,
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

// ---- p6z-1: THE YIELD'S APPLIED-ROW ---------------------------------------
//
// THE DEFECT (boarded p6z-1, measured on arms 5787087 / 5787088). Both p6z
// pair-2 arms resolved `flashsac-pack` with all thirteen detected roots, and
// NEITHER printed a single `learned fact yields` row. The policy that
// rescued the pack left no trace of having run: `finalize_impl`'s yield fired
// inside `finalize_quiet` / `finalize_quiet_detailed` (`emit_diagnostics =
// false`) during candidate evaluation, and the committed path then never met
// the conflict at all, so the loud entry point had nothing to say.
//
// A policy that silently changes a pack's outcome is exactly the shape the
// reader/writer law exists to catch. The prose WARN below is a DIAGNOSTIC and
// stays behind `emit_diagnostics` -- it belongs to the committed path. The
// APPLIED-ROW is an audit record and is emitted wherever the yield is
// applied, quiet or loud.
//
// TWO THINGS THE ROW NEEDS THAT THIS FUNCTION DID NOT HAVE.
//
// (a) The bundle. `finalize_impl` decides one package's specifier set and has
//     never known which pack it is deciding for, so a row it wrote could not
//     have named `flashsac-pack` even if it had written one.
//     `handler::produce_output_with_conflicts` -- the ONE production entry
//     into the emission reconciler, and the only caller of
//     `relax_decision::decide_for_emission` -- enters `ActiveBundleScope` for
//     the bundle it is producing. The scope is thread-local and restored on
//     drop, so work that fans out to another thread degrades to `<unknown>`
//     instead of reporting the wrong pack.
//
// (b) A bound on volume. Candidate search calls the quiet oracle once per
//     speculative subset, and a yield that holds for the committed set holds
//     for most of them; an unconditional row would write the same sentence
//     hundreds of times per package. 842 MB of repeated warnings to /oscar is
//     a measured way to take a node out, so the row is emitted ONCE per
//     (package, yielded clause, kept) WITHIN ONE BUNDLE SCOPE and says so.
//
//     The seen-set lives IN the scope and dies with it, and that placement is
//     load-bearing rather than tidy. A process-wide set makes an audit row's
//     presence depend on what some earlier, unrelated resolve happened to do:
//     the second pack to yield the same clause prints nothing, and the row
//     that is supposed to prove a policy ran becomes evidence of nothing. A
//     caller with no scope entered -- a unit guard, any future direct caller
//     -- gets every row, because suppressing an audit record by default is
//     the wrong failure direction.
/// The bundle this thread is finalizing for, and the applied-rows it has
/// already written for that bundle. Both die with the scope.
struct BundleFrame {
    bundle: String,
    rows_written: std::collections::HashSet<String>,
}

thread_local! {
    static ACTIVE_BUNDLE: std::cell::RefCell<Option<BundleFrame>> =
        const { std::cell::RefCell::new(None) };
}

/// Names the bundle whose closure the reconciler is finalizing, for the
/// duration of the guard, and carries that bundle's seen-set. Restores the
/// previous frame on drop, so nesting is safe and a panic cannot leave a stale
/// name -- or a stale seen-set -- behind.
pub(crate) struct ActiveBundleScope(Option<BundleFrame>);

impl ActiveBundleScope {
    pub(crate) fn enter(bundle: &str) -> Self {
        Self(ACTIVE_BUNDLE.with(|slot| {
            slot.replace(Some(BundleFrame {
                bundle: bundle.to_string(),
                rows_written: std::collections::HashSet::new(),
            }))
        }))
    }
}

impl Drop for ActiveBundleScope {
    fn drop(&mut self) {
        ACTIVE_BUNDLE.with(|slot| *slot.borrow_mut() = self.0.take());
    }
}

/// The bundle the current thread is finalizing for, or `<unknown>`.
pub(crate) fn active_bundle() -> String {
    ACTIVE_BUNDLE.with(|slot| match slot.borrow().as_ref() {
        Some(frame) => frame.bundle.clone(),
        None => "<unknown>".to_string(),
    })
}

/// Record `key` against the current bundle scope; `true` if it is the first
/// time this scope has seen it. With no scope entered every row is written.
fn yield_row_is_new(key: &str) -> bool {
    ACTIVE_BUNDLE.with(|slot| match slot.borrow_mut().as_mut() {
        Some(frame) => frame.rows_written.insert(key.to_string()),
        None => true,
    })
}

/// The row the yield writes wherever it is APPLIED, quiet or loud.
///
/// `learned_fact_yielded bundle=… package=… clause=… kept=… reason=…`
fn record_learned_fact_yield(package: &PypiKey, yielded: &str, kept: &VersionSpecifiers) {
    let bundle = active_bundle();
    let kept = if kept.is_empty() {
        "*".to_string()
    } else {
        kept.to_string()
    };
    let key = format!("{package}\u{1f}{yielded}\u{1f}{kept}");
    if !yield_row_is_new(&key) {
        return;
    }
    tracing::warn!(
        "learned_fact_yielded bundle={} package={} clause={} kept={} \
         reason=satisfiable-only-without-it (once per distinct yield in this bundle)",
        bundle,
        package,
        yielded,
        kept,
    );
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
        .filter(|constraint| active_for_finalization(constraint, has_override))
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
                kind: SoftenKind::AdvisoryFloor,
                unsoftened_conflict,
            });
        }
    }

    // p6z / boarded p6w-2. THE LEARNED WORKSPACE FACT YIELDS HERE TOO.
    //
    // The policy is not new and it is not this lane's invention. It is stated
    // twice already: `uv_closure::is_yieldable_advisory_source` says a LEARNED
    // workspace conda fact "yields to a hard requirement inside the closure,
    // rather than failing the pack", and `learned_fact_yield_needed` /
    // `apply_learned_fact_yields` enforce exactly that -- but ONLY for a
    // failure uv reports, because they read uv's conflict prose. The emission
    // side enforces it too
    // (`a_learned_workspace_conda_fact_cannot_veto_a_bundled_wheels_cap`).
    // This reconciler runs BEFORE uv, so a conflict it raises never reached
    // either reader: a writer with no reader on this path.
    //
    // MEASURED, jobs 5764452/5776669, `flashsac-pack`: `setuptools`
    // `==84.0.0` "required by uv constraint from workspace conda fact
    // (learned: selected by every consuming env's conda solve)" against
    // `<=65` required by wheel `FlashRL==0.1.0` Requires-Dist
    // `setuptools<=65` -- a REAL runtime dependency declared in
    // `third_party/FlashSAC/pyproject.toml`. This finalization failed closed,
    // the resolve-time back-off could attribute nothing (neither side is a
    // detected root), and THIRTEEN detected roots were dropped to route
    // around a fact that was false for the pack's own consuming env: the lock
    // the back-off then produced gives `flashsac-gpu` setuptools 59.8.0, not
    // 84.0.0. The learned fact was wrong and the wheel was right, and the
    // policy already said so.
    //
    // A DECLARED fact is never in this set -- it is operator intent and keeps
    // its own recovery. Nothing is dropped unless dropping the learned clauses
    // ALONE makes the request satisfiable, so this can never mask a conflict
    // between two hard requirements.
    let (learned, hard): (Vec<&Constraint>, Vec<&Constraint>) = active
        .iter()
        .copied()
        .partition(|constraint| crate::uv_closure::is_learned_advisory_sentence(&constraint.source));
    if !learned.is_empty() && !hard.is_empty() {
        let without_learned = intersect(&hard);
        if !specifiers_unsatisfiable(&without_learned) {
            let unsoftened_conflict = conflict_from_active(package, &active);
            let yielded = learned
                .iter()
                .map(|constraint| {
                    format!(
                        "`{}` from {}",
                        if constraint.specifiers.is_empty() {
                            "*".to_string()
                        } else {
                            constraint.specifiers.to_string()
                        },
                        constraint.source
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            // p6z-1. The APPLIED-ROW, regardless of `emit_diagnostics`: this
            // is the point at which the yield changes the answer, and the
            // quiet oracle is where it changed `flashsac-pack`'s.
            record_learned_fact_yield(package, &yielded, &without_learned);
            if emit_diagnostics {
                tracing::warn!(
                    package = %package,
                    yielded = %yielded,
                    kept = %without_learned,
                    "a LEARNED workspace fact contradicted a hard requirement stated inside \
                     this closure; the learned fact yields (it is what a previous solve \
                     happened to pick, not operator intent)",
                );
            }
            return Ok(FinalizeSuccess::AdvisoryFloorSoftened {
                specifiers: remove_redundant_specifier_clauses(without_learned),
                kind: SoftenKind::LearnedWorkspaceFact,
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

    /// The `flashsac-pack` `setuptools` clause set, job 5776669 line 250906,
    /// verbatim. Shared by p6z guard (b) and p6aa guard (a) so the fixture the
    /// yield is proved on and the fixture the ROW is proved on cannot drift.
    ///
    /// The LEARNED clause carries the RENDERED sentence, which CONTAINS the
    /// constant and is not equal to it -- p6z pair 1 built this on the bare
    /// constant, the yield fired in the test and never in the arm, and
    /// flashsac-pack dropped its 13 roots again.
    fn flashsac_learned_setuptools_fact() -> Constraint {
        let mut constraint = constraint(
            "==84.0.0",
            Provenance::UvConstraint,
            "uv constraint `setuptools==84.0.0` from workspace conda fact",
        );
        constraint.source = format!(
            "uv constraint `setuptools==84.0.0` from {} `precise-consuming-envs` \
             (conda `setuptools==84.0.0`)",
            crate::uv_closure::LEARNED_WORKSPACE_FACT_SOURCE,
        );
        assert_ne!(
            constraint.source,
            crate::uv_closure::LEARNED_WORKSPACE_FACT_SOURCE,
            "non-vacuity: the rendered sentence must NOT equal the constant, or the \
             guards built on it cannot catch the equality-vs-substring defect",
        );
        constraint
    }

    /// `FlashRL==0.1.0`'s declared `setuptools<=65` -- a real runtime
    /// dependency (`third_party/FlashSAC/pyproject.toml`), and the fifth
    /// clause whose absence made §24's transcription unable to tell "no root
    /// is the culprit" from "there is no conflict".
    fn flashsac_flashrl_cap() -> Constraint {
        constraint(
            "<=65",
            Provenance::IndexWheelMetadata,
            "wheel `FlashRL==0.1.0` Requires-Dist `setuptools<=65`",
        )
    }

    fn flashsac_measured_clauses() -> Vec<Constraint> {
        vec![
            flashsac_learned_setuptools_fact(),
            constraint(
                "!=50.0.0",
                Provenance::IndexWheelMetadata,
                "wheel `dm_control==1.0.45` Requires-Dist `setuptools!=50.0.0`",
            ),
            constraint(
                ">=41.0.0",
                Provenance::IndexWheelMetadata,
                "wheel `tensorboard==2.21.0` Requires-Dist `setuptools>=41.0.0`",
            ),
            constraint(
                "",
                Provenance::IndexWheelMetadata,
                "wheel `sapien==3.0.3` Requires-Dist `setuptools`",
            ),
            flashsac_flashrl_cap(),
        ]
    }

    /// p6z guard (b), the reconciler half. Boarded p6w-2.
    ///
    /// THE MEASURED CLAUSES, job 5776669 line 250906, verbatim:
    ///   `==84.0.0`  uv constraint from workspace conda fact (LEARNED)
    ///   `!=50.0.0`  wheel `dm_control==1.0.45`
    ///   `>=41.0.0`  wheel `tensorboard==2.21.0`
    ///   `*`         wheel `sapien==3.0.3`
    ///   `<=65`      wheel `FlashRL==0.1.0`
    ///
    /// Before p6z this finalization failed closed and THIRTEEN detected roots
    /// were dropped by the request-wide back-off -- none of which carries the
    /// contradiction. `FlashRL`'s `setuptools<=65` is a real runtime
    /// dependency (`third_party/FlashSAC/pyproject.toml`, `[project]
    /// dependencies`), so it stays. The learned fact is what an earlier solve
    /// happened to pick, and the lock the back-off produced proves it wrong
    /// for this pack's own env: `flashsac-gpu` locks setuptools 59.8.0.
    #[test]
    fn p6z_b_a_learned_workspace_fact_yields_to_a_wheels_declared_cap() {
        // THE SOURCE AS A CONSTRAINT ACTUALLY CARRIES IT, measured off arm
        // 5784994: the RENDERED sentence, which CONTAINS the constant and is
        // not equal to it. p6z pair 1 built this guard on the bare constant,
        // the yield fired in the test and never fired in the arm, and
        // flashsac-pack dropped its 13 roots again. The fixture is now the
        // real string.
        let learned = flashsac_learned_setuptools_fact;
        let flashrl = flashsac_flashrl_cap;
        let constraints = flashsac_measured_clauses();

        let specifiers = finalize(&PypiKey::from_pypi("setuptools"), &constraints)
            .expect("the LEARNED fact must yield rather than fail the pack");
        let fifty_nine = uv_pep508::uv_pep440::Version::from_str("59.8.0").unwrap();
        let eighty_four = uv_pep508::uv_pep440::Version::from_str("84.0.0").unwrap();
        assert!(
            specifiers.contains(&fifty_nine),
            "the version the workspace's own solve picks for flashsac-gpu must be \
             admissible: {specifiers}",
        );
        assert!(
            !specifiers.contains(&eighty_four),
            "the wheel's declared cap is a HARD requirement and must survive: {specifiers}",
        );

        // NON-VACUITY 1: a DECLARED workspace pin is operator intent and does
        // NOT yield -- the same split `a_declared_workspace_pin_still_decides_
        // the_constrains_carry` makes at emission.
        let declared = constraint(
            "==84.0.0",
            Provenance::WorkspaceCondaFact("flashsac-gpu".to_string()),
            "workspace conda fact `flashsac-gpu`",
        );
        finalize(
            &PypiKey::from_pypi("setuptools"),
            &[declared, flashrl()],
        )
        .expect_err("a DECLARED pin must still fail closed against the wheel's cap");

        // NON-VACUITY 2: dropping the learned clause must be what rescues it.
        // Two HARD requirements that contradict each other still fail, even
        // with a learned clause beside them.
        finalize(
            &PypiKey::from_pypi("setuptools"),
            &[
                learned(),
                flashrl(),
                constraint(
                    ">=70",
                    Provenance::IndexWheelMetadata,
                    "wheel `other==1.0` Requires-Dist `setuptools>=70`",
                ),
            ],
        )
        .expect_err("a contradiction between two hard requirements must stay fatal");

        // NON-VACUITY 3: without the yield the measured set really is fatal,
        // so the guard passes because of the yield and not because the clauses
        // happen to intersect.
        let hard_only: Vec<Constraint> = constraints
            .iter()
            .filter(|constraint| {
                !crate::uv_closure::is_learned_advisory_sentence(&constraint.source)
            })
            .cloned()
            .chain(std::iter::once(constraint(
                "==84.0.0",
                Provenance::UvConstraint,
                "uv constraint `setuptools==84.0.0` from a DECLARED pin",
            )))
            .collect();
        finalize(&PypiKey::from_pypi("setuptools"), &hard_only)
            .expect_err("the measured clause set is unsatisfiable unless the learned fact yields");
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

    /// Capture WARN-level rows written by `body`, exactly as an arm's backend
    /// log records them.
    fn captured_rows<T>(body: impl FnOnce() -> T) -> (T, String) {
        #[derive(Clone)]
        struct Sink(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Sink {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_max_level(tracing::Level::WARN)
            .with_writer({
                let buffer = std::sync::Arc::clone(&buffer);
                move || Sink(std::sync::Arc::clone(&buffer))
            })
            .finish();
        let value = tracing::subscriber::with_default(subscriber, body);
        let text = String::from_utf8(buffer.lock().unwrap().clone()).unwrap();
        (value, text)
    }

    /// p6aa guard, boarded p6z-1. THE YIELD MUST LEAVE A ROW WHERE IT IS
    /// APPLIED, AND THE QUIET ORACLE IS WHERE IT WAS APPLIED.
    ///
    /// RED on `f59aa35`: both p6z pair-2 arms (5787087 / 5787088) resolved
    /// `flashsac-pack` with all thirteen detected roots and printed NO yield
    /// row at all, because the yield fired inside `finalize_quiet` during
    /// candidate evaluation and the committed path never met the conflict.
    /// `finalize_quiet` is the entry point this guard drives, and before this
    /// commit it wrote nothing.
    ///
    /// The clause set is §25.5's measured one, reused through the same
    /// fixture builders so the two guards cannot drift apart.
    #[test]
    fn p6aa_a_the_learned_fact_yield_writes_its_applied_row_from_the_quiet_oracle() {
        let constraints = flashsac_measured_clauses();
        let package = PypiKey::from_pypi("setuptools");

        // The bundle name is not decoration: the operator reading this row
        // needs to know WHICH pack's outcome the policy changed, and
        // `finalize_impl` learns that only from the scope
        // `produce_output_with_conflicts` enters.
        let (result, rows) = captured_rows(|| {
            let _scope = ActiveBundleScope::enter("flashsac-pack");
            finalize_quiet(&package, &constraints)
        });
        result.expect("the LEARNED fact must yield rather than fail the pack");

        assert!(
            rows.contains("learned_fact_yielded"),
            "the quiet oracle applied the yield and must say so; rows were:\n{rows}",
        );
        assert!(
            rows.contains("bundle=flashsac-pack"),
            "the row must name the pack whose outcome changed; rows were:\n{rows}",
        );
        assert!(
            rows.contains("package=setuptools"),
            "the row must name the package it decided; rows were:\n{rows}",
        );
        assert!(
            rows.contains("reason=satisfiable-only-without-it"),
            "the row must state WHY the clause yielded -- dropping it alone is \
             what makes the request satisfiable; rows were:\n{rows}",
        );
        assert!(
            rows.contains("setuptools==84.0.0"),
            "the row must quote the clause that yielded, not just its package; \
             rows were:\n{rows}",
        );

        // NON-VACUITY 1: the row is not written for a finalization that never
        // yields. A satisfiable set takes the early return and says nothing.
        let (ok, quiet_rows) = captured_rows(|| {
            let _scope = ActiveBundleScope::enter("flashsac-pack");
            finalize_quiet(
                &PypiKey::from_pypi("setuptools"),
                &[constraint(
                    ">=41.0.0",
                    Provenance::IndexWheelMetadata,
                    "wheel `tensorboard==2.21.0` Requires-Dist `setuptools>=41.0.0`",
                )],
            )
        });
        ok.expect("a satisfiable set finalizes");
        assert!(
            !quiet_rows.contains("learned_fact_yielded"),
            "a finalization that applied no yield must write no row: {quiet_rows}",
        );

        // NON-VACUITY 2: the scope really is what supplies the name. Outside
        // one the row is honest about not knowing rather than blaming a pack.
        let (_, unscoped) = captured_rows(|| {
            finalize_quiet(&PypiKey::from_pypi("setuptools-unscoped-probe"), &{
                let mut probe = flashsac_measured_clauses();
                probe.push(constraint(
                    "",
                    Provenance::IndexWheelMetadata,
                    "wheel `probe==1.0` Requires-Dist `setuptools`",
                ));
                probe
            })
        });
        assert!(
            unscoped.contains("bundle=<unknown>"),
            "with no scope entered the row must say `<unknown>`, never a stale \
             or invented pack: {unscoped}",
        );

        // NON-VACUITY 3: THE ROW IS ONCE PER SCOPE, NOT ONCE PER PROCESS, AND
        // THIS IS THE ASSERTION THAT CAUGHT THE FIRST DESIGN. With the
        // seen-set process-wide, whether this guard saw its own row depended
        // on whether some earlier, unrelated resolve had already yielded the
        // same clause -- and it went red the moment a sibling lane's merge
        // changed which tests share the process. An audit record that a
        // previous unrelated call can suppress proves nothing.
        //
        // Inside ONE scope the second identical yield is quiet (that is the
        // volume bound doing its job); a NEW scope prints again.
        let (_, repeat_same_scope) = captured_rows(|| {
            let _scope = ActiveBundleScope::enter("flashsac-pack");
            let first = finalize_quiet(&package, &constraints);
            let second = finalize_quiet(&package, &constraints);
            (first.is_ok(), second.is_ok())
        });
        assert_eq!(
            repeat_same_scope.matches("learned_fact_yielded").count(),
            1,
            "two identical yields inside one bundle scope are ONE row: {repeat_same_scope}",
        );
        let (_, fresh_scope) = captured_rows(|| {
            let _scope = ActiveBundleScope::enter("holosoma-pack");
            finalize_quiet(&package, &constraints)
        });
        assert!(
            fresh_scope.contains("bundle=holosoma-pack"),
            "a DIFFERENT pack yielding the same clause must get its own row -- \
             the whole point of naming the pack: {fresh_scope}",
        );

        // NON-VACUITY 4: the LOUD entry point keeps its prose diagnostic. The
        // applied-row is an addition, not a replacement -- deleting the
        // committed-path WARN would still be a regression.
        let (loud, loud_rows) = captured_rows(|| {
            let _scope = ActiveBundleScope::enter("flashsac-pack-loud");
            finalize(&PypiKey::from_pypi("setuptools-loud-probe"), &constraints)
        });
        loud.expect("the loud path yields too");
        assert!(
            loud_rows.contains("a LEARNED workspace fact contradicted a hard requirement"),
            "the committed path's prose diagnostic must survive: {loud_rows}",
        );
        assert!(
            loud_rows.contains("learned_fact_yielded"),
            "and the applied-row is written on the loud path as well: {loud_rows}",
        );
    }
}
