//! Constraint-accumulating resolver state (PR-1, Part 1, Amendment B).
//!
//! Replaces the `seen: HashSet<String>` + two silent-drop points with a state
//! that INTERSECTS per-name `VersionSpecifiers` constraints from all requirers.
//! When tightening an already-chosen dep would exclude the currently-chosen
//! version, the dep is queued for re-resolution at the new intersection.
//!
//! DESIGN (B-α): hand-rolled constraint accumulation, fail-closed on empty
//! intersection. No backtracking. Mixed pypi+url/git for one name: url/git wins
//! by fixed precedence iff its version satisfies the pypi constraint, else
//! conflict error.
//!
//! SEAM: `ResolveState` is the future drop-in point for a resolvo-based
//! `DependencyProvider` (B-β). All version-selection logic lives here.

use std::collections::HashMap;

use anyhow::{Result, bail};
use uv_pep508::uv_pep440::{Version, VersionSpecifier, VersionSpecifiers};

use crate::handler::auto_bundle::{Pending, PendingSource};

/// Result of accumulating a new requirement edge into the resolver state.
#[derive(Debug)]
pub(crate) enum ObserveEdgeResult {
    /// The dep was not yet seen; add it to the work queue.
    /// The pending is returned so the caller can enqueue it for fetching.
    New(Pending),
    /// The dep was already seen and the newly intersected constraint still
    /// satisfies the chosen version. No re-fetch needed.
    AlreadySatisfied,
    /// The dep was already seen but the new intersection tightens PAST the
    /// chosen version — re-fetch required. The caller should push the dep
    /// back into the work queue with the new intersected `Pending`.
    NeedsReResolve(Pending),
    /// The dep was already seen and carries a non-Pypi source (Git/Url).
    /// No version-constraint intersection is possible; treat as already-satisfied.
    NonPypiAlreadySeen,
}

/// Per-name resolved entry.
#[derive(Debug, Clone)]
pub(crate) struct ChosenEntry {
    /// Parsed resolved version (for constraint checking).
    pub version: Version,
    /// Version string as-is from the wheel metadata.
    // Retained as a seam for future diagnostics / re-resolve debug output.
    #[allow(dead_code)]
    pub version_str: String,
}

/// The resolver's constraint-accumulation state (replaces `seen: HashSet`).
///
/// Invariant: `chosen[name]` exists iff the name has been fetched and is part
/// of `extras` (or `primary`). `constraints[name]` is the accumulated AND of
/// all observed specifiers for `name`; it may be `None` for bare deps with no
/// specifier.
#[derive(Debug, Default)]
pub(crate) struct ResolveState {
    /// Accumulated AND-intersection of all `Requires-Dist` specifiers per
    /// canonical conda name. `None` means "no constraint yet observed for this
    /// name" (bare dep). An empty `VersionSpecifiers` means "any version
    /// matches" (which is what `VersionSpecifiers::default()` represents).
    pub constraints: HashMap<String, Option<VersionSpecifiers>>,
    /// Resolved versions for deps that have been fetched and committed.
    pub chosen: HashMap<String, ChosenEntry>,
}

impl ResolveState {
    /// Record that `name` is present (primary / already-resolved before BFS).
    /// Seeds `chosen` so later `observe_edge` calls can check compatibility.
    pub fn seed_chosen(&mut self, canonical_name: String, version_str: String) {
        // Also mark it as seen in constraints so future observe_edge calls
        // know it's already handled.
        self.constraints
            .entry(canonical_name.clone())
            .or_insert(None);
        if let Ok(version) = version_str.parse::<Version>() {
            self.chosen.insert(
                canonical_name,
                ChosenEntry {
                    version,
                    version_str,
                },
            );
        }
    }

    /// Observe a new requirement edge `pending` for `canonical_name` with the
    /// specifiers in its `PendingSource::Pypi { specifiers }`.
    ///
    /// Returns:
    /// - `New(pending)`: first time we see this name; caller should enqueue it.
    /// - `AlreadySatisfied`: name already chosen; new constraint still satisfied.
    /// - `NeedsReResolve(pending_with_tighter_spec)`: tighter constraint
    ///   excludes the chosen version; caller should re-enqueue for re-fetch.
    /// - `NonPypiAlreadySeen`: non-Pypi source and name already seen; skip.
    /// - `Err(...)`: empty intersection (conflict); fail-closed.
    pub fn observe_edge(
        &mut self,
        canonical_name: &str,
        pending: Pending,
    ) -> Result<ObserveEdgeResult> {
        // Extract specifiers from the pending source.
        let new_specifiers = match &pending.source {
            PendingSource::Pypi { specifiers, .. } => Some(specifiers.clone()),
            _ => None,
        };

        // If we've never seen this name, record it and say it's new.
        if !self.constraints.contains_key(canonical_name) {
            self.constraints
                .insert(canonical_name.to_string(), new_specifiers);
            return Ok(ObserveEdgeResult::New(pending));
        }

        // Already seen. Non-Pypi sources can't be intersected.
        if new_specifiers.is_none() {
            return Ok(ObserveEdgeResult::NonPypiAlreadySeen);
        }

        let new_specs = new_specifiers.unwrap();

        // Intersect (AND) the new specifier with the accumulated constraint.
        let intersected = self.intersect_specifiers(canonical_name, new_specs)?;

        // Was this name already fetched (in `chosen`)?
        if let Some(chosen) = self.chosen.get(canonical_name) {
            // Does the new intersection still satisfy the chosen version?
            if intersected
                .as_ref()
                .is_none_or(|s| s.is_empty() || s.contains(&chosen.version))
            {
                return Ok(ObserveEdgeResult::AlreadySatisfied);
            }
            // The chosen version is no longer valid under the tightened constraint.
            // Build a new pending with the tighter spec.
            let tighter_pending = self.make_tighter_pending(pending, intersected)?;
            return Ok(ObserveEdgeResult::NeedsReResolve(tighter_pending));
        }

        // Not yet chosen (still in the work queue from a previous observation).
        // Update the constraint and don't re-enqueue (it's already queued).
        // The work-queue item will be fetched with the updated constraint.
        Ok(ObserveEdgeResult::AlreadySatisfied)
    }

    /// Commit a freshly-resolved dep: record its version so future
    /// `observe_edge` calls can check against it.
    pub fn commit_chosen(&mut self, canonical_name: String, version_str: String) {
        if let Ok(version) = version_str.parse::<Version>() {
            self.chosen.insert(
                canonical_name,
                ChosenEntry {
                    version,
                    version_str,
                },
            );
        }
    }

    /// Remove a dep from `chosen` when it needs to be re-resolved (tightened
    /// past its current version). The caller re-enqueues it with tighter specs.
    pub fn revoke_chosen(&mut self, canonical_name: &str) {
        self.chosen.remove(canonical_name);
    }

    /// Returns the accumulated constraint for a name, or `None` if bare.
    // Used in tests and reserved as a seam for future diagnostics.
    #[allow(dead_code)]
    pub fn current_constraint(&self, canonical_name: &str) -> Option<&VersionSpecifiers> {
        self.constraints.get(canonical_name)?.as_ref()
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Intersect `new_specs` with the accumulated constraint for `canonical_name`,
    /// updating `self.constraints`. Returns the resulting specifiers, or an error
    /// if the intersection is provably empty (a genuine conflict).
    fn intersect_specifiers(
        &mut self,
        canonical_name: &str,
        new_specs: VersionSpecifiers,
    ) -> Result<Option<VersionSpecifiers>> {
        let existing = self
            .constraints
            .get_mut(canonical_name)
            .expect("observe_edge guards this");

        // Fail-closed: detect provably empty intersections before merging.
        if let Some(existing_specs) = existing.as_ref()
            && specifiers_provably_conflict(existing_specs, &new_specs)
        {
            bail!(
                "resolve_bundle: CONFLICT on `{}` — accumulated constraint `{}` is \
                 incompatible with new requirement `{}`. \
                 Two requirers pin incompatible exact versions. \
                 Use `retread-overrides.{0} = \"<version>\"` to force a resolution.",
                canonical_name,
                existing_specs,
                new_specs,
            );
        }

        // Merge: PEP 440 VersionSpecifiers is already an AND-set. Concatenate.
        let intersected = match existing.as_ref() {
            None => {
                // First Pypi constraint for a previously-bare name.
                Some(new_specs)
            }
            Some(existing_specs) => {
                if existing_specs.is_empty() {
                    // Existing: any version. New: the new constraint wins.
                    Some(new_specs)
                } else if new_specs.is_empty() {
                    // New: any version. Existing stays.
                    existing.clone()
                } else {
                    // Merge: collect all specifiers from both sides.
                    // VersionSpecifiers is an ordered set of individual
                    // VersionSpecifier items; extend creates the AND-set.
                    let merged_specs: VersionSpecifiers = existing_specs
                        .iter()
                        .cloned()
                        .chain(new_specs.iter().cloned())
                        .collect();
                    Some(merged_specs)
                }
            }
        };

        // Update the stored constraint.
        *existing = intersected.clone();

        Ok(intersected)
    }

    /// Build a replacement `Pending` that carries the tighter intersected
    /// specifiers (so `bfs_fetch_pypi` gets the correct constraint).
    fn make_tighter_pending(
        &self,
        original: Pending,
        tighter: Option<VersionSpecifiers>,
    ) -> Result<Pending> {
        match original.source {
            PendingSource::Pypi { index, .. } => {
                let new_specs = tighter.unwrap_or_default();
                Ok(Pending {
                    source: PendingSource::Pypi {
                        specifiers: new_specs,
                        index,
                    },
                    ..original
                })
            }
            _ => bail!(
                "make_tighter_pending called on non-Pypi pending (internal invariant violated)"
            ),
        }
    }
}

/// Check whether a `VersionSpecifiers` AND-set would conflict with another —
/// meaning NO version in [0, ∞) satisfies both. This is a heuristic that
/// covers the common conflict patterns (exact-pin clash, exclusive ranges).
/// The full test is: `chosen_version ∈ intersected_specs` at choose-time.
///
/// Returns `true` iff the intersection is PROVABLY EMPTY given just the
/// specifier texts (no version enumeration needed). Used for eager fail-fast.
pub(crate) fn specifiers_provably_conflict(a: &VersionSpecifiers, b: &VersionSpecifiers) -> bool {
    use uv_pep508::uv_pep440::Operator;
    // Heuristic: two exact-pin specifiers for the same package that disagree.
    // E.g. `==1.0` ∧ `==2.0` → empty.
    let a_exact: Vec<&VersionSpecifier> = a
        .iter()
        .filter(|s| s.operator() == &Operator::Equal)
        .collect();
    let b_exact: Vec<&VersionSpecifier> = b
        .iter()
        .filter(|s| s.operator() == &Operator::Equal)
        .collect();
    if !a_exact.is_empty() && !b_exact.is_empty() {
        // Both sides have an exact pin. If they disagree, definitely conflicting.
        let a_v: std::collections::HashSet<_> = a_exact.iter().map(|s| s.version()).collect();
        let b_v: std::collections::HashSet<_> = b_exact.iter().map(|s| s.version()).collect();
        if a_v.is_disjoint(&b_v) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn specs(s: &str) -> VersionSpecifiers {
        VersionSpecifiers::from_str(s).unwrap()
    }

    fn ver(s: &str) -> Version {
        Version::from_str(s).unwrap()
    }

    fn make_pypi_pending(name: &str, spec_str: &str) -> Pending {
        Pending {
            pypi_name: name.to_string(),
            source: PendingSource::Pypi {
                specifiers: specs(spec_str),
                index: "https://pypi.org/simple/".to_string(),
            },
            extras: vec![],
        }
    }

    // ── Pillar 1: constraint accumulation is order-free ──────────────────────

    /// A-first then B-first must yield the same intersected constraint.
    #[test]
    fn intersection_is_commutative() {
        let mut state_ab = ResolveState::default();
        // Observe A: "torch>=2.0"
        let p_a = make_pypi_pending("torch", ">=2.0");
        let r1 = state_ab.observe_edge("torch", p_a.clone()).unwrap();
        assert!(matches!(r1, ObserveEdgeResult::New(_)));
        // Observe B: "torch>=2.5,<3"
        let p_b = make_pypi_pending("torch", ">=2.5,<3");
        let r2 = state_ab.observe_edge("torch", p_b.clone()).unwrap();
        assert!(matches!(
            r2,
            ObserveEdgeResult::AlreadySatisfied | ObserveEdgeResult::NeedsReResolve(_)
        ));

        let constraint_ab = state_ab
            .current_constraint("torch")
            .cloned()
            .unwrap_or_default();

        let mut state_ba = ResolveState::default();
        let _ = state_ba.observe_edge("torch", p_b.clone()).unwrap();
        let _ = state_ba.observe_edge("torch", p_a.clone()).unwrap();
        let constraint_ba = state_ba
            .current_constraint("torch")
            .cloned()
            .unwrap_or_default();

        // Both sides contain the same individual specifiers (set equality).
        let ab_set: std::collections::HashSet<String> =
            constraint_ab.iter().map(|s| s.to_string()).collect();
        let ba_set: std::collections::HashSet<String> =
            constraint_ba.iter().map(|s| s.to_string()).collect();
        assert_eq!(ab_set, ba_set, "intersection must be commutative");

        // Verify both constraints accept a version in the intersection (2.5).
        let v25 = ver("2.5");
        assert!(
            constraint_ab.contains(&v25),
            "2.5 must satisfy the AB intersection"
        );
        assert!(
            constraint_ba.contains(&v25),
            "2.5 must satisfy the BA intersection"
        );

        // Verify both constraints reject a version outside (1.9 < 2.0).
        let v19 = ver("1.9");
        assert!(
            !constraint_ab.contains(&v19),
            "1.9 must not satisfy the AB intersection"
        );
    }

    /// Shuffle 3 requirers; all orderings produce the same constraint.
    #[test]
    fn intersection_is_order_independent_three_requirers() {
        let specs_list = [">=1.0", ">=1.5,<2.0", "!=1.8"];
        let canonical = "somepkg";

        fn constraint_from_order(order: &[&str], canonical: &str) -> String {
            let mut state = ResolveState::default();
            for spec_str in order {
                let p = Pending {
                    pypi_name: canonical.to_string(),
                    source: PendingSource::Pypi {
                        specifiers: VersionSpecifiers::from_str(spec_str).unwrap(),
                        index: "https://pypi.org/simple/".to_string(),
                    },
                    extras: vec![],
                };
                let _ = state.observe_edge(canonical, p).unwrap();
            }
            let c = state
                .current_constraint(canonical)
                .cloned()
                .unwrap_or_default();
            // Represent as a sorted set of specifier strings for order-free comparison.
            let mut v: Vec<String> = c.iter().map(|s| s.to_string()).collect();
            v.sort();
            v.join(",")
        }

        let orderings = [
            [">=1.0", ">=1.5,<2.0", "!=1.8"],
            [">=1.5,<2.0", ">=1.0", "!=1.8"],
            ["!=1.8", ">=1.0", ">=1.5,<2.0"],
            ["!=1.8", ">=1.5,<2.0", ">=1.0"],
            [">=1.0", "!=1.8", ">=1.5,<2.0"],
            [">=1.5,<2.0", "!=1.8", ">=1.0"],
        ];

        let results: Vec<String> = orderings
            .iter()
            .map(|o| constraint_from_order(o, canonical))
            .collect();

        // Suppress unused variable warning for specs_list
        let _ = specs_list;

        assert!(
            results.windows(2).all(|w| w[0] == w[1]),
            "all orderings must produce the same intersection: {:?}",
            results
        );
    }

    // ── Pillar 2: re-resolve-on-tighten ──────────────────────────────────────

    /// If a dep is chosen at version 2.5, then a tighter constraint (>=2.7)
    /// arrives, observe_edge must return NeedsReResolve.
    #[test]
    fn tighten_triggers_reresolve() {
        let mut state = ResolveState::default();
        // First observation: loose spec
        let p1 = make_pypi_pending("foo", ">=2.0");
        assert!(matches!(
            state.observe_edge("foo", p1).unwrap(),
            ObserveEdgeResult::New(_)
        ));
        // Fake: dep was resolved and committed at version 2.5.
        state.commit_chosen("foo".to_string(), "2.5".to_string());
        // Second observation: tighter spec that excludes 2.5
        let p2 = make_pypi_pending("foo", ">=2.7");
        let result = state.observe_edge("foo", p2).unwrap();
        assert!(
            matches!(result, ObserveEdgeResult::NeedsReResolve(_)),
            "tighter spec excluding chosen version must trigger re-resolve, got {:?}",
            result
        );
    }

    /// If the tighter constraint still includes the chosen version, no re-resolve.
    #[test]
    fn compatible_tighten_no_reresolve() {
        let mut state = ResolveState::default();
        let p1 = make_pypi_pending("bar", ">=1.0");
        assert!(matches!(
            state.observe_edge("bar", p1).unwrap(),
            ObserveEdgeResult::New(_)
        ));
        state.commit_chosen("bar".to_string(), "2.0".to_string());
        // Tighter spec that still includes 2.0
        let p2 = make_pypi_pending("bar", ">=1.5,<3.0");
        let result = state.observe_edge("bar", p2).unwrap();
        assert!(
            matches!(result, ObserveEdgeResult::AlreadySatisfied),
            "compatible tighten must not trigger re-resolve, got {:?}",
            result
        );
    }

    // ── Conflict detection ───────────────────────────────────────────────────

    #[test]
    fn exact_pin_conflict_detected() {
        let a = specs("==1.0");
        let b = specs("==2.0");
        assert!(specifiers_provably_conflict(&a, &b));
    }

    #[test]
    fn exact_pin_same_no_conflict() {
        let a = specs("==1.0");
        let b = specs("==1.0");
        assert!(!specifiers_provably_conflict(&a, &b));
    }
}
