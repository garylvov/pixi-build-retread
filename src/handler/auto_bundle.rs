//! Auto-bundle BFS: prefer-conda routing and PyPI-only transitive
//! packaging into the bundle.
//!
//! Extracted from handler.rs (Phase 0b.3). All functions are behavior-
//! identical whole-function moves; no logic changes.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::future::Future;
use std::ops::Bound;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, anyhow};
use rattler_conda_types::ChannelUrl;
use uv_pep508::MarkerEnvironment;
use uv_pep508::uv_pep440::{Version, VersionSpecifiers, release_specifiers_to_ranges};

use crate::config::{RelaxPolicy, RetreadConfig, WheelEntry};
#[cfg(test)]
use crate::constraint::finalize;
use crate::constraint::{
    Conflict, ConflictReport, Constraint, ConstraintOriginId, Provenance, aggregate_conflicts,
    collect_conflicts, dedup_specifier_clauses, specifiers_unsatisfiable,
};
use crate::pypi;
use crate::relax::{
    CondaName, CondaTarget, NameMap, PypiKey, canonical_conda_name, default_marker_env,
    marker_env_for,
};
use crate::relax_decision::{
    Decision as RelaxDecision, RelaxationDecision, RelaxationKind as WheelMetadataRelaxationKind,
    SafetyContext, decide as decide_relaxation,
};
use crate::relaxation_record::{RelaxationRecord, RelaxationScope};
use crate::wheel::WheelMetadata;

use super::resolve_state::ResolveState;
use super::{Bundle, DEFAULT_PYTHON, PypiToCondaMap, ResolvedWheel};

/// Sentinel error returned when the incremental-add BFS detects that a new
/// dep's transitive subtree would force a version change on a dep already
/// committed in the lock closure (a "ripple").  The caller must escalate to a
/// full cold `resolve_all` rather than writing a partial lock.
///
/// Constructed via [`anyhow::Error::new`] so callers can detect it with
/// `e.downcast_ref::<IncrementalRipple>().is_some()`.
#[derive(Debug)]
pub(crate) struct IncrementalRipple {
    /// Human-readable description of which locked dep triggered the ripple.
    pub reason: String,
}

impl std::fmt::Display for IncrementalRipple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "incremental-add ripple: {}", self.reason)
    }
}

impl std::error::Error for IncrementalRipple {}

/// Whether a route the FINAL joint conda solve rejects is re-resolved under
/// the closure's current constraints, or re-injected at its pre-routing
/// version.
///
/// N27-RETREAD-142 INVERTED THIS DEFAULT AND DELETED ITS ENV VAR. It used to
/// read `#[default] Disabled` with a `from_env_value` that returned `Enabled`
/// only for the literal `"1"`, and its one production reader was
/// `std::env::var_os("RETREAD_UV_RERESOLVE")` -- a variable that occurs ZERO
/// times in the landing relock's sbatch, in the backend shim it invokes, and
/// in the 33.8 MB backend log the run produced. So the capability was built,
/// tested, and never once reached from production, which is the reader/writer
/// law's own example of a defect; and the door it was behind was an ambient
/// environment variable, which is the args-over-env rule's. The switch is now
/// [`crate::config::ROUTE_RESTORE_RERESOLVE_KEY`], a declared input on the
/// request, and its ABSENT state is the re-resolving one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum UvReresolveMode {
    Disabled,
    #[default]
    Enabled,
}

impl UvReresolveMode {
    /// The manifest key decides this, and nothing else does.
    ///
    /// `None` (the key absent) and `Some(true)` are both [`Self::Enabled`].
    /// `Some(false)` is the ONE remaining reader of [`Self::Disabled`], and it
    /// is a DELIBERATE OPT-OUT that says so on stderr: a mode that restores
    /// the behaviour which emitted an unsatisfiable cap must not be reachable
    /// silently.
    pub(crate) fn from_manifest_flag(configured: Option<bool>) -> Self {
        match configured {
            Some(false) => {
                eprintln!("{}", route_restore_reresolve_opt_out_row());
                Self::Disabled
            }
            None | Some(true) => Self::Enabled,
        }
    }

    fn is_enabled(self) -> bool {
        self == Self::Enabled
    }
}

/// The row the deliberate opt-out prints, so a lock produced without the
/// re-resolve says so in its own log rather than being indistinguishable from
/// one produced with it.
pub(crate) fn route_restore_reresolve_opt_out_row() -> String {
    format!(
        "### PYPI ROUTE RERESOLVE DISABLED key={} reason=deliberate-opt-out \
         effect=a rejected route's wheel is re-injected at its pre-routing version and its \
         Requires-Dist is never tested against the workspace conda facts (N27-RETREAD-142)",
        crate::config::ROUTE_RESTORE_RERESOLVE_KEY,
    )
}

/// How many times one restored route may backtrack before the bundle refuses.
///
/// A bound, not a policy: the measured case needs ONE step
/// (`googleapis-common-protos 1.75.3` -> `1.75.0`), and a route that has
/// rejected sixteen consecutive releases is not converging on a compatible one
/// -- it is telling us the fact and the requirement are irreconcilable, which
/// is the refusal below.
pub(crate) const ROUTE_RESTORE_RERESOLVE_MAX_BACKTRACKS: usize = 16;

/// One workspace conda fact that a candidate wheel's requirement EXCLUDES.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FactCrossing {
    /// The canonical conda name both the fact and the requirement carry.
    pub(crate) fact_name: String,
    /// The version every precise consuming environment selected for it.
    ///
    /// When the consumers hold MORE than one version and the requirement
    /// excludes all of them, this is the lowest of them: a reader needs one
    /// number in the row, and `holders_total` below says how many there were.
    pub(crate) fact_version: String,
    /// The wheel's own `Requires-Dist` line, verbatim, because the refusal has
    /// to be readable without the wheel in hand.
    pub(crate) requirement: String,
    /// How many precise consumers hold this name at all. A crossing is only a
    /// crossing when ZERO of them are admitted, so the row prints
    /// `consumers_holding=0/<holders_total>` and the denominator is the whole
    /// evidence the decision was taken on.
    pub(crate) holders_total: usize,
}

/// A name whose precise consumers DISAGREE and whose requirement admits some of
/// them but not all.
///
/// This is not a crossing -- one consumer holding an admitted version is the
/// existence proof the door asks for -- but it is not nothing either, and
/// before N27-RETREAD-141 it was invisible twice over: the pre-141 single-fact reader
/// dropped every multi-version fact, so neither the admission nor a row ever
/// happened. A decision nobody can see is the defect this series is about, so
/// the split gets its own row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FactConsumerSplit {
    pub(crate) fact_name: String,
    pub(crate) requirement: String,
    pub(crate) holders_admitting: usize,
    pub(crate) holders_total: usize,
    /// `env=version` for every holder, in holder order.
    pub(crate) held: Vec<String>,
}

/// What one door's fact check did, for the row the call site prints.
///
/// `from == to` is returned, not swallowed: a row that appears only when the
/// answer changed cannot tell a reader "this fetch was checked and needed
/// nothing" apart from "this fetch was never checked", and the second is the
/// defect. `holders_total` is how many precise consumers held the name the
/// decision turned on, so `consumers_holding=0/<n>` can be printed without the
/// call site re-deriving the evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FactCheck {
    pub(crate) from: String,
    pub(crate) to: String,
    pub(crate) holders_total: usize,
}

/// What one candidate wheel's `Requires-Dist` does to the consumers' holdings.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FactVerdict {
    /// Names NO consumer's held version satisfies. These decide.
    pub(crate) crossings: Vec<FactCrossing>,
    /// Names some but not all consumers satisfy. These are reported, never
    /// backtracked on.
    pub(crate) splits: Vec<FactConsumerSplit>,
}

/// The workspace conda facts a candidate wheel's `Requires-Dist` EXCLUDES.
///
/// This is the ACTUATOR's predicate and it is deliberately NOT
/// [`route_restore_crossings`], which matches names only. A name match is the
/// right shape for a reader's row -- `crosses=packaging,numpy,...` points a
/// human at the boundary -- and the wrong shape for a decision, because most
/// of those rows are benign: the requirement names a fact and ADMITS its
/// version. Only a requirement whose specifier excludes the fact's version can
/// become the cap that no consuming environment satisfies, so only that is
/// backtracked on.
///
/// Markers are evaluated exactly as the emission evaluates them
/// (`relax::translate`: `marker.evaluate(env, &[])`, no active extras), so a
/// requirement this pack will never actually impose cannot force a backtrack.
/// A requirement with no version clause, a URL requirement, and an
/// unparseable line are all skipped: none of them can EXCLUDE a version.
///
/// A name is considered only where [`admission_fact_holdings`] produced at
/// least one holder whose version survives the PEP 440 round trip, so a name
/// with no usable evidence still fails closed. It is NO LONGER restricted to
/// names with exactly one selected version (N27-RETREAD-141): a name whose
/// consumers disagree is a real question with a real answer, and the answer is
/// "crossing only if every one of them is excluded".
/// KEPT AS A TEST-ONLY VIEW, DELIBERATELY. It is one line of delegation to
/// [`fact_verdict_for_requires_dist`], so it cannot drift from the production
/// predicate, and it lets every guard written before N27-RETREAD-141 keep its
/// call shape instead of being rewritten in the same commit that changes what
/// it is testing.
#[cfg(test)]
fn fact_versions_excluded_by_requires_dist(
    requires_dist: &[String],
    holdings: &BTreeMap<String, BTreeMap<String, Version>>,
    env: &MarkerEnvironment,
) -> Vec<FactCrossing> {
    fact_verdict_for_requires_dist(requires_dist, holdings, env).crossings
}

/// The full per-consumer verdict: see [`fact_versions_excluded_by_requires_dist`]
/// for the shared rules and [`admission_fact_holdings`] for the evidence.
///
/// THE PREDICATE IS AN EXISTENCE TEST OVER THE CONSUMERS AND THAT IS THE WHOLE
/// CHANGE (N27-RETREAD-141). A requirement is a crossing only when it excludes
/// EVERY holder's version. One holder it admits is the proof that the pack can
/// ship this wheel and still have an environment that imports, so the door has
/// no business backtracking -- it reports the split instead.
pub(crate) fn fact_verdict_for_requires_dist(
    requires_dist: &[String],
    holdings: &BTreeMap<String, BTreeMap<String, Version>>,
    env: &MarkerEnvironment,
) -> FactVerdict {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut verdict = FactVerdict::default();
    for raw in requires_dist {
        let Ok(requirement) = crate::pep508_lenient::parse_requirement_lenient(raw) else {
            continue;
        };
        if !requirement.marker.evaluate(env, &[]) {
            continue;
        }
        let key = canonical_conda_name(requirement.name.as_ref());
        let Some(held) = holdings.get(&key) else {
            continue;
        };
        if held.is_empty() {
            continue;
        }
        let Some(uv_pep508::VersionOrUrl::VersionSpecifier(specifiers)) =
            requirement.version_or_url.as_ref()
        else {
            continue;
        };
        let admitting = held
            .values()
            .filter(|version| specifiers.contains(version))
            .count();
        if admitting == held.len() {
            continue;
        }
        if !seen.insert(key.clone()) {
            continue;
        }
        if admitting > 0 {
            verdict.splits.push(FactConsumerSplit {
                fact_name: key,
                requirement: raw.clone(),
                holders_admitting: admitting,
                holders_total: held.len(),
                held: held
                    .iter()
                    .map(|(holder, version)| format!("{holder}={version}"))
                    .collect(),
            });
            continue;
        }
        // Every holder is excluded. The row needs ONE number and the lowest is
        // the least arbitrary choice: it is the version a reader checking "is
        // there really nothing here the requirement admits?" reaches first.
        let fact_version = held
            .values()
            .min()
            .map(Version::to_string)
            .unwrap_or_default();
        verdict.crossings.push(FactCrossing {
            fact_name: key,
            fact_version,
            requirement: raw.clone(),
            holders_total: held.len(),
        });
    }
    verdict
}


/// The ONE version-scheme filter every fact reader applies, extracted so they
/// cannot drift.
///
/// "Unparseable as PEP 440" is NOT the filter that excludes conda calendar
/// spellings: `Version::from_str("2026c")` SUCCEEDS, because PEP 440 reads the
/// trailing `c` as a release-candidate marker and normalizes it to `2026rc0` --
/// a version tzdata never published, which as a door constraint could backtrack
/// or refuse against a fact that does not exist. The filter is therefore the one
/// `uv_closure::learned_fact_constraints` already applies to this same evidence,
/// inherited rather than re-invented: only a spelling that survives PEP 440
/// normalization UNCHANGED is a fact about the PyPI side. A workspace that
/// really pinned a pre-release spells it the PyPI way (`2.1.0rc1`) and is kept.
fn fact_pep440(name: &str, raw: &str) -> Option<Version> {
    let parsed = Version::from_str(raw).ok()?;
    if parsed.to_string() != *raw {
        tracing::debug!(
            conda_package = %name,
            conda_version = %raw,
            pep440 = %parsed,
            "route-restore fact skipped: its PEP 440 translation is a different \
             version than conda spelling (version-scheme mismatch), so it cannot \
             be asserted about a PyPI requirement",
        );
        return None;
    }
    Some(parsed)
}

/// The label a holding carries when it came from the workspace fact rather
/// than from a named environment's locked set.
pub(crate) const WORKSPACE_FACT_HOLDER: &str = "workspace-fact";

/// What each precise consuming environment HOLDS for a conda name:
/// name -> holder -> version.
///
/// WHY THIS REPLACES THE SINGLE FACT AT THE DOOR (N27-RETREAD-141). The door's
/// question is "can this wheel's requirement be satisfied by what the consuming
/// environments actually have?", and that is an EXISTENCE question over the
/// consumers -- not a question about one number. The pre-141 single-fact reader
/// answers it with one number and fails two ways at once. (a) When the
/// consumers DISAGREE it returns nothing at all, so a requirement that excludes
/// EVERY one of their versions is silently admitted. (b) When the base lock is
/// present the fact and the held version are the same object -- N27-RETREAD-130
/// already seeds `per_env_versions` from the lock -- but when it is ABSENT the
/// fact is the day's float, produced by a solve that filters this very pack
/// out, so it is the version the environment would hold IF THE PACK DID NOT
/// EXIST. Measured on relock `6115467` (`--base-lock drop`): the fact was
/// `numpy==2.4.6` while every consuming environment's emitted lock holds
/// `numpy 1.26.4` under the pack's own `constrains`.
///
/// THE PRECEDENCE IS EVIDENCE-FIRST AND IT IS NOT A MERGE. When the base lock
/// carries a name for at least one consumer, the LOCKED versions are the whole
/// answer for that name and the float is dropped: the lock is what the
/// environment holds WITH the pack installed, the float is a counterfactual,
/// and unioning a counterfactual into an existence proof would only ever
/// weaken the predicate. When the lock carries nothing for a name -- the
/// `--base-lock drop` shape, and a cold first pass -- the float stands as the
/// only evidence there is, and the door decides exactly as it did before, which
/// is why this change moves no row of a drop-mode relock.
///
/// Holders are keyed so a reader can see WHICH consumer holds what: an
/// environment name when the lock supplied it, `workspace-fact` otherwise.
pub(crate) fn admission_fact_holdings(
    bundle: &Bundle,
) -> BTreeMap<String, BTreeMap<String, Version>> {
    let mut holdings: BTreeMap<String, BTreeMap<String, Version>> = BTreeMap::new();
    for (env, versions) in &bundle.workspace_locked_conda {
        for (name, raw) in versions {
            let Some(parsed) = fact_pep440(name, raw) else {
                continue;
            };
            holdings
                .entry(name.clone())
                .or_default()
                .insert(env.clone(), parsed);
        }
    }
    for (name, fact) in &bundle.workspace_conda_provider_facts {
        if holdings.contains_key(name) {
            // The lock already answered for this name; the float is a
            // counterfactual and does not get a vote beside it.
            continue;
        }
        let mut entry: BTreeMap<String, Version> = BTreeMap::new();
        for raw in &fact.selected_versions {
            let Some(parsed) = fact_pep440(name, raw) else {
                continue;
            };
            entry.insert(format!("{WORKSPACE_FACT_HOLDER}:{raw}"), parsed);
        }
        if !entry.is_empty() {
            holdings.insert(name.clone(), entry);
        }
    }
    holdings
}

/// The row a re-resolved route restore writes.
///
/// `from` is the version the UNCONSTRAINED restore would have re-injected --
/// the pre-routing selection, which is the number -142 is about -- and `to` is
/// the version the workspace facts admit. `from == to` is printed too, and
/// deliberately: a row that appears only when the answer changed cannot tell a
/// reader "this restore was checked and needed nothing" apart from "this
/// restore was never checked", and the second is the defect.
///
/// `constraints` is how many workspace conda facts were in force for this
/// re-resolve, i.e. the number of names [`admission_fact_holdings`] carries. A row reading
/// `constraints=0` is the honest statement that the restore had nothing to
/// check against.
pub(crate) fn pypi_route_reresolved_row(
    dep: &str,
    from: &str,
    to: &str,
    constraints: usize,
) -> String {
    format!("### PYPI ROUTE RERESOLVED dep={dep} from={from} to={to} constraints={constraints}")
}

/// What the bundle says when NO release admitted by a door is compatible with
/// the workspace conda facts.
///
/// It names the dep, the cap and the fact, because those three are what the
/// reader needs to decide which side is wrong. It NO LONGER says the bundle
/// refuses: relock `6113841` proved that a pinned candidate set of one turns
/// this into a hard failure on a `constrains` edge the emission policy handles
/// without halting, so the door admits and defers and this text is the prose
/// half of [`pypi_admission_fact_crossing_unresolved_row`].
pub(crate) fn fact_crossing_unresolved_message(
    dep: &str,
    bundle: &str,
    base: &VersionSpecifiers,
    excluded: &[Version],
    crossing: &FactCrossing,
    site: FactConstrainedSite,
) -> String {
    let tried = excluded
        .iter()
        .map(Version::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let base = base.to_string();
    let base = if base.is_empty() {
        "(unconstrained)".to_string()
    } else {
        base
    };
    format!(
        "{site} for `{dep}` in bundle `{bundle}` has no release compatible \
         with the workspace conda fact `{}=={}`: within `{base}` every candidate this admission \
         reached requires `{}`, which excludes that version. Versions tried and refused: {tried}. \
         There is nothing left to backtrack to, so this door admits the pinned wheel and defers \
         to the pack's `constrains` emission policy, which decides the bound and does not halt \
         the lock ({EMISSION_CROSSING_POLICY}) (N27-RETREAD-142, N27-RETREAD-145, \
         N27-RETREAD-146).",
        crossing.fact_name,
        crossing.fact_version,
        crossing.requirement,
        site = site.label(),
    )
}

/// `base`, narrowed by a `!=` clause for every version this re-resolve has
/// already refused.
///
/// This is the whole of the backtrack: `bfs_fetch_pypi_from_chain` selects the
/// highest release satisfying the specifiers it is handed, so excluding the
/// refused version and re-issuing IS asking for the next one down.
pub(crate) fn specifiers_excluding(
    base: &VersionSpecifiers,
    excluded: &[Version],
    dep: &str,
) -> Result<VersionSpecifiers> {
    if excluded.is_empty() {
        return Ok(base.clone());
    }
    let mut clauses: Vec<String> = base.iter().map(ToString::to_string).collect();
    clauses.extend(excluded.iter().map(|version| format!("!={version}")));
    let joined = clauses.join(",");
    VersionSpecifiers::from_str(&joined).with_context(|| {
        format!("building re-resolve specifiers `{joined}` while restoring `{dep}`")
    })
}

/// WHICH ADMISSION PATH a wheel took into the pack's closure.
///
/// N27-RETREAD-142 fixed ONE of the two, and the production RED then came back
/// through the other. `MERGE-B44prime-4`'s relock `6112256` is the measurement:
/// in `6106911`'s backend log `PYPI ROUTE RESTORED dep=googleapis-common-protos`
/// occurs 2 times and `auto-bundled into isaaclab-2-3x-pack …
/// googleapis-common-protos` 0 times; in `6112256`'s it is 0 and 2. The SAME
/// dep, the SAME emitted cap `protobuf>=6.33.5`, the SAME unsatisfiable
/// `constrains` against the SAME workspace fact `protobuf==5.29.3` -- by a
/// different door, chosen by the store/route-cache state of the hour. A fix
/// that guards one door is therefore not a fix; both admissions are now
/// resolved under the same workspace conda facts, and this enum exists so the
/// refusal can still say WHICH door refused (N27-RETREAD-145).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FactConstrainedSite {
    /// `jointly_unroute_unsolvable`: a rejected conda route restored to PyPI.
    JointRouteRestore,
    /// `auto_bundle_transitives`: a transitive dep fetched into the bundle
    /// because no validated conda candidate satisfies it.
    AutoBundleAdmission,
}

impl FactConstrainedSite {
    /// The phrase the refusal opens with, so a reader knows which admission
    /// path to look at without diffing two log regions.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::JointRouteRestore => "joint-solve route restore",
            Self::AutoBundleAdmission => "auto-bundle PyPI admission",
        }
    }

    /// The `door=` field of [`pypi_admission_fact_crossing_unresolved_row`].
    ///
    /// One token, not the prose [`Self::label`] renders, because this half is
    /// what a gate criterion greps.
    pub(crate) fn door(self) -> &'static str {
        match self {
            Self::JointRouteRestore => "restore",
            Self::AutoBundleAdmission => "auto-bundle",
        }
    }
}

/// What the `constrains` emission does with a crossing this door could not
/// resolve -- one token, because the row it rides in is machine-read.
///
/// It is not a promise this module makes; it is a NAME for the policy that
/// already lives in `handler::mod`'s `RelaxDecision::Conflict` arm, whose
/// `constrains_only` branch has exactly two exits and neither halts: a LEARNED
/// workspace conda fact yields and the wheel's cap is emitted (the WARN says
/// "the bundled wheel's cap wins"), a DECLARED manifest pin keeps today's
/// behaviour, and an entry that stays undecidable is dropped under the WARN
/// "omitting the bound. The workspace conda provider is unbounded for this
/// name." That last branch's own comment states the rule this door broke:
/// "An undecidable one must NOT turn a pack that built before into a hard
/// failure -- that would be a regression introduced by the carry itself."
pub(crate) const EMISSION_CROSSING_POLICY: &str =
    "constrains-only:learned-fact-yields-to-cap|declared-pin-kept|undecidable-omitted";

/// The row a door writes when it admitted a wheel it could NOT resolve under
/// the workspace conda facts.
///
/// N27-RETREAD-145 gave both doors a backtracking actuator and, when the
/// backtrack ran out, a refusal that halted the lock. Relock `6113841`
/// measured what that costs: `torch` is pinned `==2.5.1+cu124`, a candidate
/// set of ONE whose `Requires-Dist` carries `sympy (==1.13.1)` against the
/// workspace conda fact `sympy==1.14.0`, so there is nothing to backtrack TO
/// and the whole lock died at 216 s on a `constrains` edge that the emission
/// policy had been handling for weeks -- B42's `sage-isaac-pack` record
/// (`6bf4ea3cfb45bc81c48a8e4a09cc3f99`) emits `sympy` as a BARE name and the
/// environments built and ran.
///
/// A door that cannot decide therefore does NOT decide: it admits the pinned
/// wheel and defers to emission, which is total for `constrains` entries. What
/// it must not do is go quiet -- a crossing nobody can see is the defect this
/// whole series is about -- so the crossing is printed here, on stderr, in the
/// grammar the other `###` rows use, and reaches the backend log, the gate
/// criteria and the operator.
pub(crate) const PYPI_ADMISSION_FACT_CROSSING_UNRESOLVED_PREFIX: &str =
    "### PYPI ADMISSION FACT-CROSSING UNRESOLVED";

pub(crate) fn pypi_admission_fact_crossing_unresolved_row(
    dep: &str,
    crossing: &FactCrossing,
    site: FactConstrainedSite,
) -> String {
    format!(
        "{PYPI_ADMISSION_FACT_CROSSING_UNRESOLVED_PREFIX} dep={dep} requirement={} fact={}=={} \
         consumers_holding=0/{} door={} policy={EMISSION_CROSSING_POLICY}",
        crossing.requirement,
        crossing.fact_name,
        crossing.fact_version,
        crossing.holders_total,
        site.door(),
    )
}

/// The prefix of the row a CONSUMER-SPLIT writes.
pub(crate) const PYPI_ADMISSION_CONSUMER_SPLIT_PREFIX: &str =
    "### PYPI ADMISSION CONSUMER-SPLIT";

/// The row a name whose precise consumers DISAGREE writes when the candidate
/// wheel's requirement admits some of them and not others.
///
/// It is deliberately NOT a crossing row: nothing backtracks, nothing refuses,
/// and the wheel is admitted. It exists because before N27-RETREAD-141 this
/// case produced no decision AND no row -- the multi-version fact was dropped
/// before the predicate ever saw it -- and a split that no operator can see is
/// how an environment that cannot import gets shipped quietly.
pub(crate) fn pypi_admission_consumer_split_row(
    dep: &str,
    split: &FactConsumerSplit,
    site: FactConstrainedSite,
) -> String {
    format!(
        "{PYPI_ADMISSION_CONSUMER_SPLIT_PREFIX} dep={dep} fact={} requirement={} \
         holders={}/{} held={} door={}",
        split.fact_name,
        split.requirement,
        split.holders_admitting,
        split.holders_total,
        split.held.join(","),
        site.door(),
    )
}

/// The per-pass row the AUTO-BUNDLE admission writes.
///
/// One row per fixed-point pass that admitted at least one wheel, on stderr
/// (`rpc.rs` owns stdout). `names` is how many workspace conda facts were in
/// force for the pass -- `names=0` is the honest statement that the pass had
/// nothing to check against -- and `backtracked` lists every dep whose
/// selection MOVED, as `<dep> <from> <to>` triples separated by `, `.
///
/// `backtracked=none` prints too, and deliberately, for the reason
/// [`pypi_route_reresolved_row`] gives: a row that appears only when something
/// moved cannot distinguish "this pass was checked and needed nothing" from
/// "this pass was never checked", and the second is the defect this whole
/// series is about.
/// Each backtrack entry also carries `consumers_holding=0/<m>`: `m` is how many
/// precise consumers held the name that FORCED the backtrack, and the numerator
/// is zero by construction, because under N27-RETREAD-141 a backtrack happens
/// only when NO consumer's held version is admitted. A reader who sees
/// `consumers_holding=0/1` on a `--base-lock drop` run knows the denominator is
/// one workspace float and not one environment's locked set.
pub(crate) fn pypi_closure_fact_constrained_row(
    names: usize,
    backtracked: &[(String, String, String, usize)],
) -> String {
    let moved = if backtracked.is_empty() {
        "none".to_string()
    } else {
        backtracked
            .iter()
            .map(|(dep, from, to, holders)| {
                format!("{dep} {from} {to} consumers_holding=0/{holders}")
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!("### PYPI CLOSURE FACT-CONSTRAINED names={names} backtracked={moved}")
}

/// Fetch one PyPI candidate UNDER the workspace conda facts, backtracking
/// until the selection's `Requires-Dist` admits every fact -- or, when NO
/// candidate does, admitting the pinned wheel and printing the crossing.
///
/// THE TWO OUTCOMES ARE NOT SYMMETRIC AND THAT IS THE POINT (N27-RETREAD-146).
/// When an admissible release EXISTS the door decides: it backtracks to it,
/// and the protobuf case (1.75.3 -> 1.75.0) is unchanged. When none exists the
/// door has no decision to make -- `torch==2.5.1+cu124` is a candidate set of
/// ONE -- so it defers to the `constrains` emission policy that governed this
/// case before either door existed, and says so loudly rather than halting a
/// lock that the emission would have completed.
///
/// THIS IS THE ONE ACTUATOR BOTH ADMISSION PATHS SHARE. It is a whole-body
/// extraction of the loop CAPWINS-5 wrote at the restore site -- the backtrack
/// is still "exclude the refused version and re-issue", because
/// `bfs_fetch_pypi_from_chain` returns the highest release satisfying the
/// specifiers it is handed, so a `!=` clause IS asking for the next one down.
///
/// Returns the wheel and, when the facts were actually consulted, the
/// `(from, to)` pair: `from` is the version the UNCONSTRAINED fetch would have
/// taken and `to` is the version the facts admit. `from == to` is returned,
/// not swallowed, so each call site can print "checked, needed nothing".
/// `None` means the re-resolve was deliberately opted out of.
#[allow(clippy::too_many_arguments)]
async fn fetch_under_workspace_facts<X, XF>(
    request: PypiFetchRequest,
    indexes: Vec<String>,
    failure_context: String,
    holdings: &BTreeMap<String, BTreeMap<String, Version>>,
    marker_env: &MarkerEnvironment,
    bundle_label: &str,
    site: FactConstrainedSite,
    enabled: bool,
    fetch_pypi: &X,
) -> Result<(ResolvedWheel, Option<FactCheck>)>
where
    X: Fn(PypiFetchRequest, Vec<String>, String) -> XF,
    XF: Future<Output = Result<ResolvedWheel>>,
{
    let dep = request.pypi_name.clone();
    let base = request.specifiers.clone();
    let mut excluded: Vec<Version> = Vec::new();
    let mut from_version: Option<String> = None;
    let mut first_wheel: Option<ResolvedWheel> = None;
    let mut refused_by: Option<FactCrossing> = None;
    // A door that has run out of candidates ADMITS and DEFERS (N27-RETREAD-146).
    // It hands back the wheel the UNCONSTRAINED fetch took -- the pinned
    // selection, the one every pre-145 lock admitted silently -- and prints the
    // crossing so it reaches the log, the gate criteria and the operator.
    // Refusing here instead would be a NEW hard failure on a `constrains` edge
    // whose emission policy is total: see [`EMISSION_CROSSING_POLICY`].
    macro_rules! admit_and_defer {
        ($crossing:expr) => {{
            let crossing: FactCrossing = $crossing;
            let wheel = first_wheel
                .clone()
                .expect("a crossing is only recorded after the first fetch succeeded");
            let version = from_version
                .clone()
                .unwrap_or_else(|| wheel.metadata.version.clone());
            // STDERR, NEVER STDOUT: `rpc.rs` owns stdout as the JSON-RPC
            // channel, exactly as every other `###` row in this module.
            eprintln!(
                "{}",
                pypi_admission_fact_crossing_unresolved_row(&dep, &crossing, site)
            );
            tracing::warn!(
                dep = %dep,
                bundle = %bundle_label,
                door = %site.door(),
                "{}",
                fact_crossing_unresolved_message(&dep, bundle_label, &base, &excluded, &crossing, site),
            );
            return Ok((
                wheel,
                Some(FactCheck {
                    from: version.clone(),
                    to: version,
                    holders_total: crossing.holders_total,
                }),
            ));
        }};
    }
    loop {
        let mut attempt = request.clone();
        attempt.specifiers = specifiers_excluding(&base, &excluded, &dep)?;
        if !excluded.is_empty() {
            // A backtrack must not be steered by the pre-routing preference:
            // that preference IS the selection just refused, and a soft
            // `preferred_version` pointing at it would make every retry
            // re-propose it.
            attempt.preferred_version = None;
        }
        let wheel = match fetch_pypi(attempt, indexes.clone(), failure_context.clone()).await {
            Ok(wheel) => wheel,
            Err(error) => {
                // An index that cannot serve the ORIGINAL request is the
                // pre-existing failure and keeps its own message. An index
                // that runs out only after this re-resolve excluded versions
                // is the -142/-145 refusal, and it must name what it refused
                // for.
                let Some(crossing) = refused_by else {
                    return Err(error);
                };
                admit_and_defer!(crossing);
            }
        };
        if from_version.is_none() {
            from_version = Some(wheel.metadata.version.clone());
            first_wheel = Some(wheel.clone());
        }
        if !enabled {
            return Ok((wheel, None));
        }
        let verdict = fact_verdict_for_requires_dist(
            &wheel.metadata.requires_dist,
            holdings,
            marker_env,
        );
        let Some(crossing) = verdict.crossings.into_iter().next() else {
            // A SPLIT IS NOT A CROSSING AND IT IS NOT SILENCE EITHER. The wheel
            // is admitted -- some consumer holds a version this requirement
            // admits -- and the disagreement is printed so the operator can see
            // which environments are on which side. STDERR, NEVER STDOUT.
            for split in &verdict.splits {
                eprintln!("{}", pypi_admission_consumer_split_row(&dep, split, site));
            }
            let from = from_version
                .clone()
                .unwrap_or_else(|| wheel.metadata.version.clone());
            let to = wheel.metadata.version.clone();
            // The denominator a backtrack reports is the evidence that FORCED
            // it -- the refused crossing's holders -- not the accepted wheel's,
            // which by definition crosses nothing.
            let holders_total = refused_by
                .as_ref()
                .map(|crossing| crossing.holders_total)
                .or_else(|| verdict.splits.first().map(|split| split.holders_total))
                .unwrap_or(0);
            return Ok((
                wheel,
                Some(FactCheck {
                    from,
                    to,
                    holders_total,
                }),
            ));
        };
        let refused = Version::from_str(&wheel.metadata.version).with_context(|| {
            format!(
                "parsing candidate wheel version `{}` while re-resolving `{dep}`",
                wheel.metadata.version
            )
        })?;
        if excluded.contains(&refused) {
            // The index answered with a version this request had already
            // excluded. That is not a backtrack that can terminate, so it
            // stops here rather than spinning.
            admit_and_defer!(crossing);
        }
        excluded.push(refused);
        refused_by = Some(crossing.clone());
        if excluded.len() > ROUTE_RESTORE_RERESOLVE_MAX_BACKTRACKS {
            admit_and_defer!(crossing);
        }
    }
}

/// Per-group state needed to hand a rejected conda route back to the outer uv
/// auto-route fixpoint. `keep_pypi` is monotonic across group retries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct UvReresolveContext {
    pub(crate) mode: UvReresolveMode,
    pub(crate) uv_backed: bool,
    pub(crate) keep_pypi: BTreeSet<PypiKey>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AutoBundleOutcome {
    Complete {
        relaxations: Vec<WheelMetadataRelaxation>,
    },
    RetryKeepPypi {
        keep_pypi: BTreeSet<PypiKey>,
    },
}

/// Returns `true` if `pypi_key` has an enabled, unambiguous conda equivalent
/// in the effective name map (parselmouth + FALLBACK + user
/// retread-name-map). This identifies a conda target; the requirement-specific
/// probe must still pass [`validated_conda_route`] before auto-bundling is
/// skipped.
pub(crate) fn prefer_conda_match(pypi_key: &PypiKey, name_map: &NameMap) -> bool {
    name_map
        .get(pypi_key)
        .is_some_and(|target| target.mapped_name().is_some())
}

/// A conda route is valid only when the requirement-specific probe found at
/// least one matching candidate. Indecisive probes and internally
/// inconsistent `Some(true)`/zero-candidate results stay on the PyPI path.
pub(crate) fn validated_conda_route(probe: &crate::probe::ProbeResult) -> bool {
    probe.satisfiable == Some(true) && probe.matching_candidates > 0
}

fn indecisive_probe(package: String, spec: String) -> crate::probe::ProbeResult {
    crate::probe::ProbeResult {
        package,
        spec,
        channels_consulted: vec![],
        satisfiable: None,
        matching_candidates: 0,
    }
}

/// Render the original PEP 440 requirement for conda's match-spec parser.
/// `VersionSpecifiers` inserts a space after commas; rattler's lenient parser
/// can silently lose the following clause unless that space is removed.
/// Empty specifiers are the genuinely unconstrained `*` case.
pub(crate) fn conda_probe_spec(specifiers: &VersionSpecifiers) -> String {
    let normalized = specifiers.to_string().replace(", ", ",");
    if normalized.trim().is_empty() {
        "*".to_string()
    } else {
        normalized
    }
}

/// v0.46.0: pick the conda target name for a PyPI dep in the BFS
/// prefer-conda decision.
///
/// Precedence (matches what emission's `translate` uses, so the BFS and
/// emission agree on routing):
///   1. The merged `name_map` (user retread-name-map, FALLBACK_PYPI_TO_CONDA,
///      and unambiguous parselmouth) -- a curated, unambiguous answer wins
///      outright. This is what makes `torch -> pytorch` route to conda even
///      though parselmouth's inverted map lists multiple ambiguous conda
///      candidates for `torch` with no identity match.
///   2. Else parselmouth's inverted candidates: an identity match
///      (`numpy -> numpy`) wins; else a single candidate; else `None`
///      (ambiguous -> caller leaves it on the PyPI/bundle path).
///
/// Returns the conda package name to probe/route to, or `None` to keep the
/// dep on the PyPI side.
pub(crate) fn pick_conda_target(
    dep_pypi_key: &PypiKey,
    name_map: &NameMap,
    pypi_to_conda: &PypiToCondaMap,
) -> Option<CondaName> {
    if let Some(target) = name_map.get(dep_pypi_key) {
        return target.mapped_name().cloned();
    }
    let candidates = pypi_to_conda.get(dep_pypi_key.as_str())?;
    if let Some(identity) = candidates
        .iter()
        .find(|candidate| PypiKey::from_pypi(candidate) == *dep_pypi_key)
    {
        Some(CondaName::new(identity.clone()))
    } else if candidates.len() == 1 {
        Some(CondaName::new(candidates[0].clone()))
    } else {
        None
    }
}

/// One PyPI wheel request routed through the canonical auto-bundle index
/// chain. `bundle_name` is the identity stored in the vendored set; it is
/// normally the PyPI name, but the ordinary name-mapped auto-bundle path
/// preserves its historical conda-name identity.
#[derive(Clone, Debug)]
struct PypiFetchRequest {
    pypi_name: String,
    bundle_name: String,
    specifiers: VersionSpecifiers,
    preferred_version: Option<String>,
    relaxations: Vec<WheelMetadataRelaxation>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct WheelMetadataRelaxation {
    bundle: String,
    package: PypiKey,
    kind: WheelMetadataRelaxationKind,
    original: String,
    relaxed: String,
    source: String,
    involved_sources: Vec<String>,
    scope: String,
    tier: RelaxPolicy,
}

const ABI_ANCHOR_EMISSION_SOURCE: &str = "retread ABI-anchor emission invariant";

impl std::fmt::Display for WheelMetadataRelaxation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.kind == WheelMetadataRelaxationKind::AbiAnchorCapCompleted {
            return write!(
                f,
                "RETREAD AUTO-COMPLETED ABI anchor in bundle `{}`: package `{}` `{}` -> `{}`; \
                 sources involved: {}",
                self.bundle,
                self.package,
                self.original,
                self.relaxed,
                self.involved_sources.join("; "),
            );
        }
        if self.kind == WheelMetadataRelaxationKind::ExactPinWidened
            && self.source == ABI_ANCHOR_EMISSION_SOURCE
        {
            return write!(
                f,
                "RETREAD AUTO-WIDENED ABI anchor exact pin in bundle `{}`: package `{}` \
                 `{}` -> `{}`; sources involved: {}",
                self.bundle,
                self.package,
                self.original,
                self.relaxed,
                self.involved_sources.join("; "),
            );
        }
        write!(
            f,
            "RETREAD AUTO-RELAXED unsatisfiable wheel metadata{}: `{}` {} `{}` -> `{}` \
             from {}; wheels involved: {}",
            self.scope,
            self.package,
            self.kind.label(),
            self.original,
            self.relaxed,
            self.source,
            self.involved_sources.join("; "),
        )
    }
}

pub(super) fn abi_anchor_cap_completion(
    bundle: &str,
    package: &PypiKey,
    original: impl Into<String>,
    normalized: impl Into<String>,
    involved_sources: Vec<String>,
) -> WheelMetadataRelaxation {
    WheelMetadataRelaxation {
        bundle: bundle.to_string(),
        package: package.clone(),
        kind: WheelMetadataRelaxationKind::AbiAnchorCapCompleted,
        original: original.into(),
        relaxed: normalized.into(),
        source: ABI_ANCHOR_EMISSION_SOURCE.to_string(),
        involved_sources,
        scope: String::new(),
        tier: RelaxPolicy::Minor,
    }
}

pub(super) fn abi_anchor_exact_pin_widening(
    bundle: &str,
    package: &PypiKey,
    original: impl Into<String>,
    normalized: impl Into<String>,
    involved_sources: Vec<String>,
) -> WheelMetadataRelaxation {
    WheelMetadataRelaxation {
        bundle: bundle.to_string(),
        package: package.clone(),
        kind: WheelMetadataRelaxationKind::ExactPinWidened,
        original: original.into(),
        relaxed: normalized.into(),
        source: ABI_ANCHOR_EMISSION_SOURCE.to_string(),
        involved_sources,
        scope: String::new(),
        tier: RelaxPolicy::Minor,
    }
}

impl WheelMetadataRelaxation {
    pub(super) fn bundle(&self) -> &str {
        &self.bundle
    }

    pub(super) fn to_record(&self, scope: &RelaxationScope) -> RelaxationRecord {
        RelaxationRecord {
            package: self.package.as_str().to_string(),
            original_spec: self.original.clone(),
            resulting_spec: self.relaxed.clone(),
            tier: self.tier,
            kind: self.kind.into(),
            source: self.source.clone(),
            involved_wheels: self.involved_sources.clone(),
            scope: scope.clone(),
        }
    }

    pub(super) fn emit(&self) {
        let message = self.to_string();
        tracing::warn!(
            package = %self.package,
            original = %self.original,
            relaxed = %self.relaxed,
            source = %self.source,
            tier = ?self.tier,
            "{message}",
        );
        // Pixi drains successful backend stderr and the protocol has no
        // non-fatal diagnostic field. The controlling terminal is the only
        // channel that reaches an ordinary interactive `pixi install`.
        crate::status::tty(&format!("WARNING: {message}"));
    }
}

/// Active, target-marker-matched dependency declarations keyed in the PyPI
/// identity domain. Provenance remains attached until the shared finalizer has
/// resolved authority and produced a source-rich conflict, if any.
type ObservedRequirements = BTreeMap<PypiKey, Vec<Constraint>>;

fn wheel_requirement_origin_id(
    wheel_name: &str,
    wheel_version: &str,
    dependency_name: &str,
    specifiers: &VersionSpecifiers,
    raw_requirement: &str,
) -> ConstraintOriginId {
    ConstraintOriginId::from_parts(
        "wheel-requirement",
        [
            PypiKey::from_pypi(wheel_name).into_string(),
            wheel_version.to_string(),
            PypiKey::from_pypi(dependency_name).into_string(),
            render_specifiers(specifiers),
            raw_requirement.to_string(),
        ],
    )
}

fn provenance_origin_label(provenance: &Provenance) -> String {
    match provenance {
        Provenance::IndexWheelMetadata => "index-wheel-metadata".to_string(),
        Provenance::SourceBuiltRelaxed => "source-built-relaxed".to_string(),
        Provenance::DepsFromRelaxed => "deps-from-relaxed".to_string(),
        Provenance::WorkspaceCondaFact(scope) => format!("workspace-conda-fact:{scope}"),
        Provenance::UvRoot => "uv-root".to_string(),
        Provenance::UvConstraint => "uv-constraint".to_string(),
        Provenance::UvOverride => "uv-override".to_string(),
        Provenance::PriorSelection => "prior-selection".to_string(),
    }
}

fn observe_requirement(
    observed: &mut ObservedRequirements,
    pypi_name: &str,
    specifiers: &VersionSpecifiers,
    wheel_name: &str,
    wheel_version: &str,
    raw_requirement: &str,
    provenance: Provenance,
) {
    let observations = observed.entry(PypiKey::from_pypi(pypi_name)).or_default();
    let observation = Constraint {
        specifiers: specifiers.clone(),
        source: format!("wheel `{wheel_name}=={wheel_version}` Requires-Dist `{raw_requirement}`"),
        provenance,
        origin_id: wheel_requirement_origin_id(
            wheel_name,
            wheel_version,
            pypi_name,
            specifiers,
            raw_requirement,
        ),
    };
    if !observations.contains(&observation) {
        observations.push(observation);
    }
}

const JOINT_ROUTE_CONFIGURATION_REMEDIATION: &str = "Review the named dependencies and \
    PyPI indexes, then adjust `retread-overrides` or `retread-drop-deps` in the pack \
    manifest if needed (see README).";

/// User-visible identity retained at the joint-route boundary. A rich
/// resolution target can name the consuming environment and target profile;
/// legacy/test paths fall back to the bundle, concrete subdir, and Python.
#[derive(Clone, Debug, PartialEq, Eq)]
struct JointRouteDiagnosticContext {
    bundle: String,
    environments: Vec<String>,
    profiles: Vec<String>,
    platform: String,
    python: String,
}

impl JointRouteDiagnosticContext {
    fn new(
        bundle: &Bundle,
        target: &crate::pypi::WheelTarget,
        workspace_scope: Option<&crate::workspace::ResolvedWorkspaceTarget>,
    ) -> Self {
        Self {
            bundle: bundle.conda_name.clone(),
            environments: workspace_scope
                .map(|scope| scope.environments.clone())
                .unwrap_or_default(),
            profiles: workspace_scope
                .map(|scope| scope.profiles.clone())
                .unwrap_or_default(),
            platform: target.conda_subdir.clone(),
            python: target.python_version.clone(),
        }
    }

    fn quoted(values: &[String]) -> String {
        values
            .iter()
            .map(|value| format!("'{value}'"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn scope(&self) -> String {
        let owner = match self.environments.as_slice() {
            [] => format!("in bundle '{}'", self.bundle),
            [environment] => format!(
                "in environment '{environment}' for bundle '{}'",
                self.bundle
            ),
            environments => format!(
                "in environments {} for bundle '{}'",
                Self::quoted(environments),
                self.bundle
            ),
        };
        let mut details = Vec::new();
        match self.profiles.as_slice() {
            [] => {}
            [profile] => details.push(format!("target profile '{profile}'")),
            profiles => details.push(format!("target profiles {}", Self::quoted(profiles))),
        }
        details.push(format!("platform {}", self.platform));
        details.push(format!("python {}", self.python));
        format!("{owner} ({})", details.join(", "))
    }

    fn routing_error(&self, problem: impl std::fmt::Display) -> anyhow::Error {
        anyhow!(
            "dependency routing failed {}: {problem}. {JOINT_ROUTE_CONFIGURATION_REMEDIATION}",
            self.scope()
        )
    }

    fn scope_error(&self, error: anyhow::Error) -> anyhow::Error {
        match error.downcast::<crate::constraint::Conflict>() {
            Ok(conflict) => anyhow::Error::new(conflict.with_scope(self.scope())),
            Err(error) => match error.downcast::<ConflictReport>() {
                Ok(report) => anyhow::Error::new(report.with_scope(self.scope())),
                Err(error) => {
                    error.context(format!("validating dependency routes {}", self.scope()))
                }
            },
        }
    }
}

pub(super) fn scope_conflicts_for_target(
    conflicts: Vec<Conflict>,
    bundle: &Bundle,
    target: &crate::pypi::ResolutionTarget,
) -> Vec<Conflict> {
    let scope =
        JointRouteDiagnosticContext::new(bundle, target.wheel_target(), target.workspace_scope())
            .scope();
    conflicts
        .into_iter()
        .map(|conflict| conflict.with_scope(scope.clone()))
        .collect()
}

fn render_specifiers(specifiers: &VersionSpecifiers) -> String {
    if specifiers.is_empty() {
        "*".to_string()
    } else {
        specifiers.to_string().replace(", ", ",")
    }
}

fn constraints_are_unsatisfiable(constraints: &[&Constraint]) -> bool {
    let combined = dedup_specifier_clauses(
        constraints
            .iter()
            .flat_map(|constraint| constraint.specifiers.iter().cloned()),
    );
    specifiers_unsatisfiable(&combined)
}

/// Reduce a fail-closed conflict to an inclusion-minimal set of requirements.
///
/// The strict finalizer deliberately retains every active requirement for the
/// source-rich diagnostic. The override menu is easier to act on when it
/// names only the sides needed to prove the contradiction.
fn minimal_unsat_requirements(requirements: &[Constraint]) -> Vec<&Constraint> {
    let mut core = requirements.iter().collect::<Vec<_>>();
    if !constraints_are_unsatisfiable(&core) {
        return core;
    }

    let mut index = 0;
    while index < core.len() {
        let mut trial = core.clone();
        trial.remove(index);
        if constraints_are_unsatisfiable(&trial) {
            core = trial;
        } else {
            index += 1;
        }
    }
    core
}

fn fallback_pack_manifest(bundle: &str) -> String {
    format!("pypi-packs/{bundle}/pixi.toml")
}

/// Attach the ordinary leaf-level remediation used when no unique
/// transitive root can be proven. This helper is pure: the rendered TOML is
/// carried by the typed error and is never applied to the manifest.
pub(super) fn attach_leaf_conflict_suggestion(
    conflict: Conflict,
    pack_manifest: Option<&str>,
    bundle: &str,
) -> Conflict {
    let mut alternatives = minimal_unsat_requirements(conflict.requirements())
        .into_iter()
        .map(|constraint| {
            (
                render_specifiers(&constraint.specifiers),
                constraint.source.clone(),
            )
        })
        .collect::<Vec<_>>();
    alternatives.sort();
    alternatives.dedup();

    let package = conflict.package.as_str().to_string();
    let toml = crate::pack_overrides::render_drop_deps_with_override_menu(&package, &alternatives);
    conflict.with_suggestion(
        pack_manifest
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| fallback_pack_manifest(bundle)),
        toml,
    )
}

#[derive(Clone, Debug)]
struct ClosureDependencyEdge {
    parent: PypiKey,
    parent_wheel_name: String,
    parent_wheel_version: String,
    child: PypiKey,
    specifiers: VersionSpecifiers,
    raw_requirement: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum RootPinOrigin {
    DepsFromRoot {
        requirement: String,
    },
    RequiresDist {
        parent: PypiKey,
        requirement: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RootPinCandidate {
    package: PypiKey,
    specifier: String,
    edit: RootPinEdit,
    origin: RootPinOrigin,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum RootPinEdit {
    Override,
    RetreadWheel { bundle_group: String },
}

#[derive(Clone, Debug)]
struct BundleRoot {
    constraint: ConfiguredRootConstraint,
    bundle_group: String,
    deps_from: bool,
}

fn configured_primary_bundle_group(bundle: &Bundle, config: &RetreadConfig) -> String {
    for primary_name in [
        PypiKey::from_pypi(&bundle.primary.pypi_name),
        PypiKey::from_pypi(&bundle.primary.metadata.name),
    ] {
        if let Some((name, entry)) = config
            .retread_wheels
            .iter()
            .find(|(name, _)| PypiKey::from_pypi(name) == primary_name)
        {
            return super::bundle_group_for(name, entry, config.default_bundle.as_deref());
        }
    }

    let bundle_key = canonical_conda_name(&bundle.conda_name);
    let groups = config
        .retread_wheels
        .iter()
        .map(|(name, entry)| super::bundle_group_for(name, entry, config.default_bundle.as_deref()))
        .filter(|group| canonical_conda_name(group) == bundle_key)
        .collect::<BTreeSet<_>>();
    if groups.len() == 1 {
        return groups.into_iter().next().expect("one bundle group exists");
    }
    bundle.conda_name.clone()
}

fn configured_bundle_roots(
    bundle: &Bundle,
    config: &RetreadConfig,
) -> BTreeMap<PypiKey, BundleRoot> {
    let bundle_group = configured_primary_bundle_group(bundle, config);
    let bundle_key = canonical_conda_name(&bundle_group);
    let declared = config
        .retread_wheels
        .iter()
        .filter(|(name, entry)| {
            canonical_conda_name(&super::bundle_group_for(
                name,
                entry,
                config.default_bundle.as_deref(),
            )) == bundle_key
        })
        .collect::<Vec<_>>();

    let mut roots = declared
        .into_iter()
        .map(|(name, entry)| {
            let package = PypiKey::from_pypi(name);
            (
                package.clone(),
                BundleRoot {
                    constraint: configured_root_constraint(&package, entry),
                    bundle_group: super::bundle_group_for(
                        name,
                        entry,
                        config.default_bundle.as_deref(),
                    ),
                    deps_from: false,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    for wheel in bundle.all_wheels() {
        let pypi_name = PypiKey::from_pypi(&wheel.pypi_name);
        let metadata_name = PypiKey::from_pypi(&wheel.metadata.name);
        let root = roots
            .get(&pypi_name)
            .or_else(|| roots.get(&metadata_name))
            .cloned();
        if let Some(root) = root {
            roots.insert(pypi_name, root.clone());
            roots.insert(metadata_name, root);
        }
    }

    for (name, requirements) in &bundle.uv_dependency_graph.deps_from_root_requirements {
        let package = PypiKey::from_pypi(name);
        roots.insert(
            package.clone(),
            BundleRoot {
                constraint: root_constraint_from_requirements(&package, requirements),
                bundle_group: bundle_group.clone(),
                deps_from: true,
            },
        );
    }
    roots
}

fn closure_dependency_edges(
    bundle: &Bundle,
    platform: &str,
    python: &str,
) -> Option<Vec<ClosureDependencyEdge>> {
    let marker_env = marker_env_for(platform, python).ok()?;
    let mut edges = Vec::new();
    for wheel in bundle.all_wheels() {
        for raw_requirement in &wheel.metadata.requires_dist {
            let requirement: uv_pep508::Requirement =
                crate::pep508_lenient::parse_requirement_lenient(raw_requirement).ok()?;
            if !requirement.marker.evaluate(&marker_env, &[]) {
                continue;
            }
            let specifiers = match requirement.version_or_url.as_ref() {
                Some(uv_pep508::VersionOrUrl::VersionSpecifier(specifiers)) => specifiers.clone(),
                Some(uv_pep508::VersionOrUrl::Url(_)) | None => VersionSpecifiers::empty(),
            };
            edges.push(ClosureDependencyEdge {
                parent: PypiKey::from_pypi(&wheel.metadata.name),
                parent_wheel_name: wheel.metadata.name.clone(),
                parent_wheel_version: wheel.metadata.version.clone(),
                child: PypiKey::from_pypi(requirement.name.as_ref()),
                specifiers,
                raw_requirement: raw_requirement.clone(),
            });
        }
    }
    edges.sort_by(|left, right| {
        (
            &left.parent,
            &left.child,
            render_specifiers(&left.specifiers),
            &left.raw_requirement,
            &left.parent_wheel_version,
        )
            .cmp(&(
                &right.parent,
                &right.child,
                render_specifiers(&right.specifiers),
                &right.raw_requirement,
                &right.parent_wheel_version,
            ))
    });
    edges.dedup_by(|left, right| {
        left.parent == right.parent
            && left.child == right.child
            && left.specifiers == right.specifiers
            && left.raw_requirement == right.raw_requirement
            && left.parent_wheel_version == right.parent_wheel_version
    });
    Some(edges)
}

fn concrete_open_floor(specifiers: &VersionSpecifiers) -> Option<String> {
    let range = release_specifiers_to_ranges(specifiers.clone());
    let (lower, upper) = range.bounding_range()?;
    if !matches!(upper, Bound::Unbounded) {
        return None;
    }
    let Bound::Included(version) = lower else {
        return None;
    };
    specifiers.contains(version).then(|| format!("=={version}"))
}

#[derive(Clone, Debug)]
enum ConfiguredRootConstraint {
    OpenFloor {
        specifier: String,
        requirement: String,
    },
    Fixed,
    Unactionable,
}

fn root_constraint_from_specifiers(
    package: &PypiKey,
    specifiers: &VersionSpecifiers,
) -> ConfiguredRootConstraint {
    if specifiers.is_empty() {
        return ConfiguredRootConstraint::Unactionable;
    }
    if let Some(specifier) = concrete_open_floor(specifiers) {
        return ConfiguredRootConstraint::OpenFloor {
            specifier,
            requirement: format!("{package}{}", render_specifiers(specifiers)),
        };
    }
    let rendered = render_specifiers(specifiers);
    let exact = rendered
        .strip_prefix("===")
        .or_else(|| rendered.strip_prefix("=="))
        .filter(|version| {
            !version.is_empty()
                && !version.contains([',', '*'])
                && Version::from_str(version).is_ok()
        })
        .is_some();
    if exact {
        ConfiguredRootConstraint::Fixed
    } else {
        ConfiguredRootConstraint::Unactionable
    }
}

fn root_constraint_from_requirements(
    package: &PypiKey,
    requirements: &[String],
) -> ConfiguredRootConstraint {
    let mut clauses = Vec::new();
    for raw in requirements {
        let Ok(specifiers) = VersionSpecifiers::from_str(raw) else {
            return ConfiguredRootConstraint::Unactionable;
        };
        clauses.extend(specifiers);
    }
    let combined = dedup_specifier_clauses(clauses.into_iter());
    root_constraint_from_specifiers(package, &combined)
}

fn configured_root_constraint(package: &PypiKey, entry: &WheelEntry) -> ConfiguredRootConstraint {
    let Some(raw_version) = entry.version.as_deref() else {
        return ConfiguredRootConstraint::Fixed;
    };
    let raw_version = raw_version.trim();
    if raw_version.is_empty() || raw_version == "*" {
        return ConfiguredRootConstraint::Unactionable;
    }
    let normalized = if raw_version.starts_with(['<', '>', '=', '!', '~']) {
        raw_version.to_string()
    } else {
        format!("=={raw_version}")
    };
    let Ok(specifiers) = VersionSpecifiers::from_str(&normalized) else {
        return ConfiguredRootConstraint::Unactionable;
    };
    root_constraint_from_specifiers(package, &specifiers)
}

fn dependency_topology(bundle: &Bundle) -> BTreeMap<PypiKey, Vec<PypiKey>> {
    let edges = bundle
        .uv_dependency_graph
        .edges
        .iter()
        .map(|edge| {
            (
                PypiKey::from_pypi(&edge.parent),
                PypiKey::from_pypi(&edge.child),
            )
        })
        .collect::<BTreeSet<_>>();
    let mut parents: BTreeMap<PypiKey, Vec<PypiKey>> = BTreeMap::new();
    for (parent, child) in edges {
        parents.entry(child).or_default().push(parent);
    }
    parents
}

fn root_boundary_candidate(
    parent: &PypiKey,
    child: &PypiKey,
    root: &BundleRoot,
    metadata_edges: &[ClosureDependencyEdge],
) -> Option<RootPinCandidate> {
    match &root.constraint {
        ConfiguredRootConstraint::OpenFloor {
            specifier,
            requirement,
        } if root.deps_from => {
            return Some(RootPinCandidate {
                package: parent.clone(),
                specifier: specifier.clone(),
                edit: RootPinEdit::Override,
                origin: RootPinOrigin::DepsFromRoot {
                    requirement: requirement.clone(),
                },
            });
        }
        ConfiguredRootConstraint::OpenFloor { .. } => return None,
        ConfiguredRootConstraint::Fixed if !root.deps_from => {}
        ConfiguredRootConstraint::Fixed => return None,
        ConfiguredRootConstraint::Unactionable => return None,
    }

    let boundary = metadata_edges
        .iter()
        .filter(|edge| &edge.parent == parent && &edge.child == child)
        .collect::<Vec<_>>();
    if boundary.is_empty() {
        return None;
    }
    let combined = dedup_specifier_clauses(
        boundary
            .iter()
            .flat_map(|edge| edge.specifiers.iter().cloned()),
    );
    let raw_requirement = boundary
        .iter()
        .map(|edge| edge.raw_requirement.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(" and ");
    Some(RootPinCandidate {
        package: child.clone(),
        specifier: concrete_open_floor(&combined)?,
        edit: RootPinEdit::RetreadWheel {
            bundle_group: root.bundle_group.clone(),
        },
        origin: RootPinOrigin::RequiresDist {
            parent: parent.clone(),
            requirement: raw_requirement,
        },
    })
}

/// Reverse-walk every parent path until it reaches a configured pack root.
///
/// An open-ended deps-from root is the outermost package the user can pin.
/// For an already-fixed configured wheel, its first open-ended Requires-Dist
/// edge is the outermost remaining choice. Missing parents, cycles, bounded
/// constraints, or divergent paths return `None` and preserve the ordinary
/// leaf suggestion.
fn walk_to_root_pin(
    package: &PypiKey,
    roots: &BTreeMap<PypiKey, BundleRoot>,
    parents: &BTreeMap<PypiKey, Vec<PypiKey>>,
    metadata_edges: &[ClosureDependencyEdge],
    visiting: &mut BTreeSet<PypiKey>,
) -> Option<Vec<RootPinCandidate>> {
    if !visiting.insert(package.clone()) {
        return None;
    }
    let incoming = parents.get(package)?;
    let mut candidates = Vec::new();
    for parent in incoming {
        if let Some(root) = roots.get(parent) {
            candidates.push(root_boundary_candidate(
                parent,
                package,
                root,
                metadata_edges,
            )?);
        } else {
            candidates.extend(walk_to_root_pin(
                parent,
                roots,
                parents,
                metadata_edges,
                visiting,
            )?);
        }
    }
    visiting.remove(package);
    (!candidates.is_empty()).then_some(candidates)
}

/// Identify one transitive root responsible for a leaf conflict.
///
/// Conflict requirements are first reduced to their minimal unsatisfiable
/// core and matched back to exact wheel edges by structured origin IDs. Direct
/// root requirements form the baseline side; each transitive side is walked
/// back through the materialized closure graph. A root pin is returned only
/// when every traversed path agrees on one package/floor and that floor moves
/// the currently selected transitive root to an older version.
fn transitive_root_pin_candidate(
    conflict: &Conflict,
    bundle: &Bundle,
    config: &RetreadConfig,
    platform: &str,
    python: &str,
) -> Option<RootPinCandidate> {
    if bundle.uv_dependency_graph.edges.is_empty() {
        return None;
    }
    let roots = configured_bundle_roots(bundle, config);
    if roots.is_empty() {
        return None;
    }
    let edges = closure_dependency_edges(bundle, platform, python)?;
    let parents = dependency_topology(bundle);

    let mut candidates = BTreeSet::new();
    let mut saw_transitive_side = false;
    for constraint in minimal_unsat_requirements(conflict.requirements()) {
        if !matches!(
            constraint.provenance,
            Provenance::IndexWheelMetadata
                | Provenance::SourceBuiltRelaxed
                | Provenance::DepsFromRelaxed
        ) {
            continue;
        }
        let matching = edges
            .iter()
            .filter(|edge| edge.child == conflict.package)
            .filter(|edge| {
                let observed_origin = wheel_requirement_origin_id(
                    &edge.parent_wheel_name,
                    &edge.parent_wheel_version,
                    edge.child.as_str(),
                    &constraint.specifiers,
                    &edge.raw_requirement,
                );
                let emission_origin = ConstraintOriginId::from_parts(
                    "wheel-requires-dist",
                    [
                        edge.parent_wheel_name.as_str(),
                        edge.parent_wheel_version.as_str(),
                        edge.raw_requirement.as_str(),
                    ],
                );
                constraint.origin_id == observed_origin || constraint.origin_id == emission_origin
            })
            .collect::<Vec<_>>();
        if matching.is_empty() {
            return None;
        }
        for edge in matching {
            if roots.contains_key(&edge.parent) {
                continue;
            }
            saw_transitive_side = true;
            let path_candidates =
                walk_to_root_pin(&edge.parent, &roots, &parents, &edges, &mut BTreeSet::new())?;
            for candidate in path_candidates {
                candidates.insert(candidate);
            }
        }
    }
    if !saw_transitive_side || candidates.len() != 1 {
        return None;
    }
    let candidate = candidates.into_iter().next()?;
    let selected = bundle
        .uv_dependency_graph
        .selected_versions
        .get(candidate.package.as_str())
        .map(String::as_str)
        .or_else(|| {
            bundle
                .all_wheels()
                .find(|wheel| {
                    PypiKey::from_pypi(&wheel.pypi_name) == candidate.package
                        || PypiKey::from_pypi(&wheel.metadata.name) == candidate.package
                })
                .map(|wheel| wheel.metadata.version.as_str())
        })?;
    let pinned = Version::from_str(candidate.specifier.strip_prefix("==")?).ok()?;
    let selected = Version::from_str(selected).ok()?;
    (pinned < selected).then_some(candidate)
}

fn render_root_pin_suggestion(candidate: &RootPinCandidate, leaf: &PypiKey) -> String {
    let origin = match &candidate.origin {
        RootPinOrigin::DepsFromRoot { requirement } => {
            format!("# Deps-from root requirement `{requirement}` starts that transitive path.\n")
        }
        RootPinOrigin::RequiresDist {
            parent,
            requirement,
        } => format!("# `{parent}` Requires-Dist `{requirement}`.\n"),
    };
    let mut rendered = format!(
        "# Constrain the transitive root that introduced the `{leaf}` conflict.\n\
         {origin}"
    );
    match &candidate.edit {
        RootPinEdit::Override => {
            rendered.push_str(
                "# Under [package.build.config.retread-overrides], add or update this key\n\
                 # (create that table once if absent).\n\
                 # This override must replace the deps-from floor for the pin to take effect.\n\
                 # Confirm the version exists on the configured index\n\
                 # and resolves the conflict:\n",
            );
            rendered.push_str(&crate::pack_overrides::render_override_toml(
                candidate.package.as_str(),
                &candidate.specifier,
            ));
        }
        RootPinEdit::RetreadWheel { bundle_group } => {
            rendered.push_str(
                "# Edit the existing [package.build.config.retread-wheels] table.\n\
         # Add or update this entry, then confirm the version exists on the configured index\n\
                 # and resolves the conflict:\n",
            );
            rendered.push_str(&crate::pack_overrides::render_root_pin_toml(
                candidate.package.as_str(),
                &candidate.specifier,
                bundle_group,
            ));
        }
    }
    rendered
}

pub(super) fn attach_conflict_suggestion(
    conflict: Conflict,
    bundle: &Bundle,
    config: &RetreadConfig,
    platform: &str,
    python: &str,
) -> Conflict {
    if let Some(candidate) =
        transitive_root_pin_candidate(&conflict, bundle, config, platform, python)
    {
        let toml = render_root_pin_suggestion(&candidate, &conflict.package);
        return conflict.with_suggestion(
            config
                .pack_manifest_path
                .clone()
                .unwrap_or_else(|| fallback_pack_manifest(&bundle.conda_name)),
            toml,
        );
    }
    attach_leaf_conflict_suggestion(
        conflict,
        config.pack_manifest_path.as_deref(),
        &bundle.conda_name,
    )
}

pub(super) fn wheel_metadata_relaxations(
    package: &PypiKey,
    constraints: &[Constraint],
    decisions: Vec<RelaxationDecision>,
    bundle: &str,
    scope: String,
) -> Vec<WheelMetadataRelaxation> {
    let involved_sources: Vec<String> = constraints
        .iter()
        .filter(|constraint| {
            matches!(
                &constraint.provenance,
                Provenance::IndexWheelMetadata
                    | Provenance::SourceBuiltRelaxed
                    | Provenance::DepsFromRelaxed
            )
        })
        .map(|constraint| constraint.source.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    decisions
        .into_iter()
        .map(|decision| WheelMetadataRelaxation {
            bundle: bundle.to_string(),
            package: package.clone(),
            kind: decision.kind,
            original: decision.original,
            relaxed: decision.relaxed,
            source: decision.source,
            involved_sources: involved_sources.clone(),
            scope: scope.clone(),
            tier: decision.tier,
        })
        .collect()
}

#[derive(Clone, Debug)]
struct RestoreRequestBuilder {
    pypi_name: String,
    bundle_name: String,
    safety_conda_names: BTreeSet<String>,
    abi_anchor_alias: bool,
    constraints: Vec<Constraint>,
    route_preferences: BTreeMap<String, BTreeSet<String>>,
    lock_preferences: BTreeMap<String, BTreeSet<String>>,
    relax: RelaxPolicy,
}

#[derive(Clone, Copy)]
enum ConflictSuggestionMode<'a> {
    #[cfg(test)]
    Leaf { pack_manifest: Option<&'a str> },
    Graph {
        bundle: &'a Bundle,
        config: &'a RetreadConfig,
        target: &'a crate::pypi::WheelTarget,
    },
}

impl RestoreRequestBuilder {
    fn new(pypi_name: &str, relax: RelaxPolicy) -> Self {
        Self {
            pypi_name: pypi_name.to_string(),
            bundle_name: canonical_conda_name(pypi_name),
            safety_conda_names: BTreeSet::new(),
            abi_anchor_alias: false,
            constraints: Vec::new(),
            route_preferences: BTreeMap::new(),
            lock_preferences: BTreeMap::new(),
            relax,
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) {
        if !self.constraints.contains(&constraint) {
            self.constraints.push(constraint);
        }
    }

    fn add_safety_conda_name(&mut self, conda_name: &str) {
        self.safety_conda_names.insert(conda_name.to_string());
    }

    fn add_abi_anchor_alias(&mut self, abi_anchor_alias: bool) {
        self.abi_anchor_alias |= abi_anchor_alias;
    }

    fn add_preference(
        preferences: &mut BTreeMap<String, BTreeSet<String>>,
        version: String,
        source: String,
    ) {
        preferences.entry(version).or_default().insert(source);
    }

    fn unique_preference(preferences: &BTreeMap<String, BTreeSet<String>>) -> Option<String> {
        if preferences.len() > 1 {
            return None;
        }
        preferences.keys().next().cloned()
    }

    #[cfg(test)]
    fn finish(self) -> Result<PypiFetchRequest> {
        self.finish_with_context(None)
    }

    #[cfg(test)]
    fn finish_with_context(
        self,
        diagnostic_context: Option<&JointRouteDiagnosticContext>,
    ) -> Result<PypiFetchRequest> {
        self.finish_with_context_and_pack(diagnostic_context, None)
    }

    #[cfg(test)]
    fn finish_with_context_and_pack(
        self,
        diagnostic_context: Option<&JointRouteDiagnosticContext>,
        pack_manifest: Option<&str>,
    ) -> Result<PypiFetchRequest> {
        self.finish_with_suggestion(
            diagnostic_context,
            ConflictSuggestionMode::Leaf { pack_manifest },
        )
    }

    fn finish_with_graph_context(
        self,
        diagnostic_context: Option<&JointRouteDiagnosticContext>,
        bundle: &Bundle,
        config: &RetreadConfig,
        target: &crate::pypi::WheelTarget,
    ) -> Result<PypiFetchRequest> {
        self.finish_with_suggestion(
            diagnostic_context,
            ConflictSuggestionMode::Graph {
                bundle,
                config,
                target,
            },
        )
    }

    fn finish_with_suggestion(
        mut self,
        diagnostic_context: Option<&JointRouteDiagnosticContext>,
        suggestion_mode: ConflictSuggestionMode<'_>,
    ) -> Result<PypiFetchRequest> {
        let prior_constraints = self
            .route_preferences
            .iter()
            .map(|preference| ("route-prior-selection", preference))
            .chain(
                self.lock_preferences
                    .iter()
                    .map(|preference| ("lock-prior-selection", preference)),
            )
            .map(|(origin_kind, (version, sources))| {
                let specifiers = VersionSpecifiers::from_str(&format!("=={version}"))
                    .with_context(|| {
                        format!(
                            "parsing prior selection `{version}` for `{}`",
                            self.pypi_name
                        )
                    })?;
                Ok(Constraint {
                    specifiers,
                    provenance: Provenance::PriorSelection,
                    source: sources.iter().cloned().collect::<Vec<_>>().join(", "),
                    origin_id: ConstraintOriginId::from_parts(
                        origin_kind,
                        [
                            PypiKey::from_pypi(&self.pypi_name).into_string(),
                            version.clone(),
                            format!("=={version}"),
                        ],
                    ),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for constraint in prior_constraints {
            self.add_constraint(constraint);
        }
        if self.constraints.is_empty() {
            let problem = format!(
                "no active PyPI requirement or prior version is available for `{}`",
                self.pypi_name
            );
            return Err(match diagnostic_context {
                Some(context) => context.routing_error(problem),
                None => anyhow!("{problem}"),
            });
        }
        let route_preference = Self::unique_preference(&self.route_preferences);
        let lock_preference = Self::unique_preference(&self.lock_preferences);
        let package = PypiKey::from_pypi(&self.pypi_name);
        let mapped_anchor = self
            .safety_conda_names
            .iter()
            .find(|name| crate::solve::is_abi_anchor(name))
            .map(String::as_str);
        let record_bundle = diagnostic_context
            .map(|context| context.bundle.as_str())
            .unwrap_or(&self.bundle_name)
            .to_string();
        // C30-1. NAME THE PACK FOR THE RESTORE PATH'S ROWS.
        //
        // This is the SECOND production entry into the constraint reconciler
        // (`relax_decision::decide` -> `constraint::finalize_quiet_detailed`
        // -> `finalize_impl`), and until now the only one without a scope.
        // p6z-1 scoped `handler::produce_output_with_conflicts` on the belief
        // that emission was the only entry; it is the only caller of
        // `decide_for_emission`, which is not the same claim. Every route the
        // joint conda solve rejects comes back through here, BEFORE emission,
        // and that is where the learned-fact yield actually fired in job
        // 5834618: `setuptools` and `prettytable` restoring into
        // `flashsac-pack`, both rows unattributed.
        //
        // The name comes from the joint-route diagnostic context, which is
        // the bundle the restore is being performed FOR -- not
        // `self.bundle_name`, which is `canonical_conda_name(pypi_name)`, the
        // PACKAGE's own sanitized name. Printing that would have been worse
        // than printing nothing: a plausible-looking pack that no manifest
        // declares. Where there is no context the mode still carries the real
        // `Bundle`; the leaf mode that carries neither is test-only, and it
        // gets no scope, so the emitter's refusal covers it.
        let _yield_scope = match (diagnostic_context, suggestion_mode) {
            (Some(context), _) => Some(crate::constraint::ActiveBundleScope::enter_for_environments(
                &context.bundle,
                &context.environments,
            )),
            (None, ConflictSuggestionMode::Graph { bundle, .. }) => Some(
                crate::constraint::ActiveBundleScope::enter(&bundle.conda_name),
            ),
            #[cfg(test)]
            (None, ConflictSuggestionMode::Leaf { .. }) => None,
        };
        let (specifiers, relaxations) = match decide_relaxation(
            &package,
            &self.constraints,
            self.relax,
            &SafetyContext::new(mapped_anchor).with_abi_anchor_alias(self.abi_anchor_alias),
        ) {
            RelaxDecision::Strict {
                specifiers,
                diagnostics,
            } => (
                specifiers,
                wheel_metadata_relaxations(
                    &package,
                    &self.constraints,
                    diagnostics,
                    &record_bundle,
                    diagnostic_context
                        .map(|context| format!(" {}", context.scope()))
                        .unwrap_or_default(),
                ),
            ),
            RelaxDecision::Relaxed {
                specifiers,
                decisions,
            } => (
                specifiers,
                wheel_metadata_relaxations(
                    &package,
                    &self.constraints,
                    decisions,
                    &record_bundle,
                    diagnostic_context
                        .map(|context| format!(" {}", context.scope()))
                        .unwrap_or_default(),
                ),
            ),
            RelaxDecision::Conflict(original_conflict) => {
                let original_conflict = match suggestion_mode {
                    #[cfg(test)]
                    ConflictSuggestionMode::Leaf { pack_manifest } => {
                        attach_leaf_conflict_suggestion(
                            original_conflict,
                            pack_manifest,
                            diagnostic_context
                                .map(|context| context.bundle.as_str())
                                .unwrap_or(&self.bundle_name),
                        )
                    }
                    ConflictSuggestionMode::Graph {
                        bundle,
                        config,
                        target,
                    } => attach_conflict_suggestion(
                        original_conflict,
                        bundle,
                        config,
                        &target.conda_subdir,
                        &target.python_version,
                    ),
                };
                return Err(anyhow::Error::new(match diagnostic_context {
                    Some(context) => original_conflict.with_scope(context.scope()),
                    None => original_conflict,
                }));
            }
            RelaxDecision::SearchExhausted(exhausted) => {
                return Err(anyhow::Error::new(match diagnostic_context {
                    Some(context) => exhausted.with_scope(context.scope()),
                    None => exhausted,
                }));
            }
        };

        Ok(PypiFetchRequest {
            pypi_name: self.pypi_name,
            bundle_name: self.bundle_name,
            specifiers,
            // The uv selection reflects the current closure and therefore
            // outranks an older favor-lock hint. Both remain soft.
            preferred_version: route_preference.or(lock_preference),
            relaxations,
        })
    }
}

/// A Requires-Dist dependency provisionally left on conda after its
/// requirement-specific probe succeeded. Unlike uv auto-routes, these routes
/// are emitted naturally from wheel metadata, so they live only for the
/// duration of auto-bundle's final Rule-2 validation.
#[derive(Clone, Debug)]
struct ProvisionalMetadataRoute {
    pypi_name: String,
    conda_name: String,
    preferred_versions: BTreeSet<String>,
}

type ProvisionalMetadataRoutes = BTreeMap<String, Vec<ProvisionalMetadataRoute>>;

#[derive(Debug, PartialEq, Eq)]
enum JointRouteOutcome {
    Unchanged,
    Mutated {
        relaxations: Vec<WheelMetadataRelaxation>,
    },
    RetryKeepPypi {
        keep_pypi: BTreeSet<PypiKey>,
    },
}

#[cfg(test)]
async fn fetch_from_index_chain<X, XF>(
    indexes: &[String],
    request: PypiFetchRequest,
    fetch_pypi: &X,
    failure_context: String,
) -> Result<ResolvedWheel>
where
    X: Fn(PypiFetchRequest, String) -> XF,
    XF: Future<Output = Result<ResolvedWheel>>,
{
    super::fetch_from_pypi_index_chain(
        indexes,
        |index| fetch_pypi(request.clone(), index),
        failure_context,
    )
    .await
}

fn record_metadata_route(
    routes: &mut ProvisionalMetadataRoutes,
    pypi_name: String,
    conda_name: String,
    preferred_version: Option<String>,
) {
    let conda_key = canonical_conda_name(&conda_name);
    let pypi_key = canonical_conda_name(&pypi_name);
    let origins = routes.entry(conda_key).or_default();
    if let Some(existing) = origins
        .iter_mut()
        .find(|origin| canonical_conda_name(&origin.pypi_name) == pypi_key)
    {
        if let Some(version) = preferred_version {
            existing.preferred_versions.insert(version);
        }
        return;
    }
    let preferred_versions = preferred_version.into_iter().collect();
    origins.push(ProvisionalMetadataRoute {
        pypi_name,
        conda_name,
        preferred_versions,
    });
}

fn expand_name_map_groups(names: &mut HashSet<String>, name_map: &NameMap) {
    loop {
        let mut changed = false;
        for (pypi_name, target) in name_map {
            let Some(conda_name) = target.mapped_name() else {
                continue;
            };
            let pypi_name = pypi_name.as_str().to_string();
            let conda_name = conda_name.key().as_str().to_string();
            if names.contains(&pypi_name) || names.contains(&conda_name) {
                changed |= names.insert(pypi_name);
                changed |= names.insert(conda_name);
            }
        }
        if !changed {
            break;
        }
    }
}

/// This is the "pip autoresolve" path: deps that exist on PyPI but might
/// not be on the workspace's conda channels (`aiodns`, `qdldl`, etc.) get
/// pip-installed into the conda package alongside the primary wheel.
///
/// Prefer-conda by default: anything parselmouth or the user's name_map
/// knows a conda equivalent for is skipped here and emitted as a conda
/// run-dep instead.
///
/// Once validated routing chooses PyPI, resolution failure is fatal.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn auto_bundle_transitives(
    bundle: &mut Bundle,
    indexes: &[String],
    target: &crate::pypi::ResolutionTarget,
    download_dir: &Path,
    config: &RetreadConfig,
    conda_channels: &[ChannelUrl],
    // incremental-add path: pre-fill seen_candidate with locked names so
    // auto_bundle_transitives does not try to re-bundle them. Cold path: None.
    locked_closure: Option<&std::collections::BTreeMap<String, String>>,
    // favor-lock: preferred versions for PyPI auto-bundle resolution.
    // When RETREAD_FAVOR_LOCK=1 and a dep has a committed lock version,
    // use resolve_preferring so the re-resolve prefers that version.
    // Cold path (RETREAD_FAVOR_LOCK unset or first build): None.
    favor_lock_prefs: Option<&std::collections::BTreeMap<String, String>>,
    // uv resolver: canonical name -> resolved version of the uv
    // closure's exported wheels (auto-routed members already excluded
    // via --no-emit-package). The closure is AUTHORITATIVE: uv resolved
    // these from PyPI under no-build, the auto-route probe already
    // declined to move them to conda at the resolved version, so they
    // MUST ship in the bundle. Candidates in this set bypass the
    // prefer-conda probes below because uv already ran the validated route
    // decision for their exact resolved versions.
    //
    // v4.6 Part A follow-up (run-44 smoke failure): the map's entries
    // are ALSO seeded as first-round candidates. Requires-Dist scans
    // alone cannot discover closure members no bundled wheel's metadata
    // names — retread-deps-from roots (tensordict, lightning, ...) ride
    // only the uv closure, and under the minimal routing policy they are
    // refused conda routes AND blocked from conda run-dep emission
    // (Bundle::uv_closure_names), so without this seed they silently
    // vanished from the pack (shipped wheels == [protomotions], smoke
    // ModuleNotFoundError). Under aggressive routing the gap was
    // invisible: the sweep moved them all to conda.
    // Legacy resolver path: None (probes decide, unchanged).
    uv_closure_wheels: Option<&std::collections::BTreeMap<String, String>>,
    // Shared Rule-1/Rule-2 workspace-aware conda co-solve oracle. Rule 2
    // finalizes mutable uv routes only after the merged bundle's actual
    // emitted sibling constraints are known.
    conda_co_solve: &super::CondaCoSolveContext,
    uv_reresolve: &UvReresolveContext,
) -> Result<AutoBundleOutcome> {
    let probe_channels = conda_channels.to_vec();
    let probe_python = target.python_version.clone();
    let probe_many = move |pairs: Vec<(String, String)>| {
        let channels = probe_channels.clone();
        let python = probe_python.clone();
        async move { crate::probe::probe_many(&channels, pairs, Some(&python)).await }
    };
    let solve_context = conda_co_solve.clone();
    let co_solve = move |routes: Vec<crate::uv_closure::CondaRouteSpec>| {
        let context = solve_context.clone();
        async move { context.solve(routes).await }
    };
    let prewarm_context = conda_co_solve.clone();
    let prewarm_routes = move |routes: Vec<crate::uv_closure::CondaRouteSpec>| {
        let context = prewarm_context.clone();
        async move { context.prewarm(routes).await }
    };
    let provider_context = conda_co_solve.clone();
    let validate_standalone_provider_route = move |route: crate::uv_closure::CondaRouteSpec| {
        let context = provider_context.clone();
        async move { context.validate_standalone_provider_route(route).await }
    };
    let target = target.clone();
    let fetch_target = target.clone();
    let download_dir = download_dir.to_path_buf();
    let relax = config.relax;
    let abi_aliases = super::output_abi_aliases(bundle, config);
    // Production resolution owns the complete chain here. The generic harness
    // passes the whole ordered chain and caller-specific exhaustion context to
    // this callback once, avoiding the old chain-inside-chain retry shape.
    let fetch_pypi =
        move |request: PypiFetchRequest, indexes: Vec<String>, failure_context: String| {
            let target = fetch_target.clone();
            let download_dir = download_dir.clone();
            let abi_aliases = abi_aliases.clone();
            async move {
                let (resolved_url, metadata, sdist_prov) = super::bfs_fetch_pypi_from_chain(
                    &request.pypi_name,
                    &request.specifiers,
                    &indexes,
                    &target,
                    &download_dir,
                    relax,
                    &abi_aliases,
                    request.preferred_version.as_deref(),
                    failure_context,
                )
                .await?;
                let (upstream_url, sdist_source, metadata_provenance) =
                    super::bfs_fetch_provenance(&resolved_url, sdist_prov);
                Ok(ResolvedWheel {
                    pypi_name: request.bundle_name,
                    url: resolved_url,
                    upstream_url,
                    git_source: None,
                    sdist_source,
                    metadata_provenance,
                    // This path fetches the wheel as-is (no phase-D rewrite of
                    // its own), so the read metadata IS the pre-D view.
                    original_requires_dist: metadata.requires_dist.clone(),
                    metadata,
                    extras_requested: vec![],
                    auto_data: None,
                    auto_data_dedup_skipped_root: None,
                })
            }
        };
    let channels_consulted = conda_co_solve.channels_consulted();
    auto_bundle_transitives_with_route_precheck_and_prewarm(
        AutoBundleInputs {
            bundle,
            indexes,
            target: target.wheel_target(),
            workspace_scope: target.workspace_scope(),
            config,
            locked_closure,
            favor_lock_prefs,
            uv_closure_wheels,
            channels_consulted: &channels_consulted,
            probe_parallelism: conda_co_solve.probe_parallelism(),
            uv_reresolve,
        },
        &probe_many,
        &co_solve,
        &prewarm_routes,
        &validate_standalone_provider_route,
        &fetch_pypi,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn auto_bundle_transitives_with<P, PF, C, CF, X, XF>(
    bundle: &mut Bundle,
    indexes: &[String],
    target: &crate::pypi::WheelTarget,
    config: &RetreadConfig,
    locked_closure: Option<&BTreeMap<String, String>>,
    favor_lock_prefs: Option<&BTreeMap<String, String>>,
    uv_closure_wheels: Option<&BTreeMap<String, String>>,
    probe_many: &P,
    co_solve: &C,
    fetch_pypi: &X,
    channels_consulted: &[String],
    uv_reresolve: &UvReresolveContext,
) -> Result<AutoBundleOutcome>
where
    P: Fn(Vec<(String, String)>) -> PF,
    PF: Future<Output = Vec<crate::probe::ProbeResult>>,
    C: Fn(Vec<crate::uv_closure::CondaRouteSpec>) -> CF,
    CF: Future<Output = crate::uv_closure::CoInstallVerdict>,
    X: Fn(PypiFetchRequest, Vec<String>, String) -> XF,
    XF: Future<Output = Result<ResolvedWheel>>,
{
    let allow_source_route = |_route: crate::uv_closure::CondaRouteSpec| async {
        crate::uv_closure::CoInstallVerdict::Sat
    };
    let no_prewarm = |_routes: Vec<crate::uv_closure::CondaRouteSpec>| async {};
    auto_bundle_transitives_with_route_precheck_and_prewarm(
        AutoBundleInputs {
            bundle,
            indexes,
            target,
            workspace_scope: None,
            config,
            locked_closure,
            favor_lock_prefs,
            uv_closure_wheels,
            channels_consulted,
            probe_parallelism: 1,
            uv_reresolve,
        },
        probe_many,
        co_solve,
        &no_prewarm,
        &allow_source_route,
        fetch_pypi,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
async fn auto_bundle_transitives_with_route_precheck<P, PF, C, CF, V, VF, X, XF>(
    bundle: &mut Bundle,
    indexes: &[String],
    target: &crate::pypi::WheelTarget,
    workspace_scope: Option<&crate::workspace::ResolvedWorkspaceTarget>,
    config: &RetreadConfig,
    locked_closure: Option<&BTreeMap<String, String>>,
    favor_lock_prefs: Option<&BTreeMap<String, String>>,
    uv_closure_wheels: Option<&BTreeMap<String, String>>,
    probe_many: &P,
    co_solve: &C,
    validate_standalone_provider_route: &V,
    fetch_pypi: &X,
    channels_consulted: &[String],
    probe_parallelism: usize,
    uv_reresolve: &UvReresolveContext,
) -> Result<AutoBundleOutcome>
where
    P: Fn(Vec<(String, String)>) -> PF,
    PF: Future<Output = Vec<crate::probe::ProbeResult>>,
    C: Fn(Vec<crate::uv_closure::CondaRouteSpec>) -> CF,
    CF: Future<Output = crate::uv_closure::CoInstallVerdict>,
    V: Fn(crate::uv_closure::CondaRouteSpec) -> VF,
    VF: Future<Output = crate::uv_closure::CoInstallVerdict>,
    X: Fn(PypiFetchRequest, Vec<String>, String) -> XF,
    XF: Future<Output = Result<ResolvedWheel>>,
{
    let no_prewarm = |_routes: Vec<crate::uv_closure::CondaRouteSpec>| async {};
    auto_bundle_transitives_with_route_precheck_and_prewarm(
        AutoBundleInputs {
            bundle,
            indexes,
            target,
            workspace_scope,
            config,
            locked_closure,
            favor_lock_prefs,
            uv_closure_wheels,
            channels_consulted,
            probe_parallelism,
            uv_reresolve,
        },
        probe_many,
        co_solve,
        &no_prewarm,
        validate_standalone_provider_route,
        fetch_pypi,
    )
    .await
}

struct AutoBundleInputs<'a> {
    bundle: &'a mut Bundle,
    indexes: &'a [String],
    target: &'a crate::pypi::WheelTarget,
    workspace_scope: Option<&'a crate::workspace::ResolvedWorkspaceTarget>,
    config: &'a RetreadConfig,
    locked_closure: Option<&'a BTreeMap<String, String>>,
    favor_lock_prefs: Option<&'a BTreeMap<String, String>>,
    uv_closure_wheels: Option<&'a BTreeMap<String, String>>,
    channels_consulted: &'a [String],
    probe_parallelism: usize,
    uv_reresolve: &'a UvReresolveContext,
}

async fn auto_bundle_transitives_with_route_precheck_and_prewarm<
    P,
    PF,
    C,
    CF,
    W,
    WF,
    V,
    VF,
    X,
    XF,
>(
    inputs: AutoBundleInputs<'_>,
    probe_many: &P,
    co_solve: &C,
    prewarm_routes: &W,
    validate_standalone_provider_route: &V,
    fetch_pypi: &X,
) -> Result<AutoBundleOutcome>
where
    P: Fn(Vec<(String, String)>) -> PF,
    PF: Future<Output = Vec<crate::probe::ProbeResult>>,
    C: Fn(Vec<crate::uv_closure::CondaRouteSpec>) -> CF,
    CF: Future<Output = crate::uv_closure::CoInstallVerdict>,
    W: Fn(Vec<crate::uv_closure::CondaRouteSpec>) -> WF,
    WF: Future<Output = ()>,
    V: Fn(crate::uv_closure::CondaRouteSpec) -> VF,
    VF: Future<Output = crate::uv_closure::CoInstallVerdict>,
    X: Fn(PypiFetchRequest, Vec<String>, String) -> XF,
    XF: Future<Output = Result<ResolvedWheel>>,
{
    let AutoBundleInputs {
        bundle,
        indexes,
        target,
        workspace_scope,
        config,
        locked_closure,
        favor_lock_prefs,
        uv_closure_wheels,
        channels_consulted,
        probe_parallelism,
        uv_reresolve,
    } = inputs;
    let diagnostic_context = JointRouteDiagnosticContext::new(bundle, target, workspace_scope);

    // Build the skip set: anything already in the bundle, plus the user's
    // `retread-conda-deps` allowlist (deps that should stay as conda
    // run-deps), plus drop-deps, plus packages with explicit overrides
    // (user is forcing conda emission via a spec).
    //
    // There is intentionally NO built-in "conda-preferred" list. ABI
    // collisions (e.g. between a bundled numpy 1.26 and the workspace's
    // conda numpy 2.x) are the user's call -- add the package name to
    // `retread-conda-deps` to keep it on the conda side.
    // P2 (grizzly #2): seed CANONICAL names. Wheels record raw pypi
    // names (underscores, case, dots); candidates are checked in
    // canonical form -- a raw-seeded set missed them and re-bundled an
    // already-present wheel (double-installing ABI-sensitive deps).
    let mut skip: HashSet<String> = bundle
        .all_wheels()
        .map(|w| canonical_conda_name(&w.pypi_name))
        .collect();
    let mut explicitly_fixed: HashSet<String> = config
        .conda_deps
        .iter()
        .map(|name| canonical_conda_name(name))
        .chain(
            config
                .overrides
                .keys()
                .map(|name| canonical_conda_name(name)),
        )
        .collect();
    expand_name_map_groups(&mut explicitly_fixed, &config.name_map);
    skip.extend(explicitly_fixed);
    skip.extend(config.drop_deps.iter().map(|n| canonical_conda_name(n)));
    skip.extend(bundle.auto_dropped.iter().cloned());

    // Fixed-point loop: each newly-bundled wheel has its own
    // Requires-Dist that may name more PyPI-only transitives, which
    // themselves should be auto-bundled (e.g. bundling torch pulls in
    // nvidia-cuda-nvrtc-cu12). Re-scan after every bundle until no new
    // wheels are added. Cycle-detected via seen_candidate, which
    // accumulates across iterations.
    let mut seen_candidate: HashSet<String> = skip.clone();
    // A uv auto-route is already a complete routing decision. Do not send its
    // PyPI identity through the independent metadata-probe/fetch path again:
    // an identity-fallback route need not have a name-map entry, so that path
    // can otherwise fetch the route's old exact wheel before Rule 2 sees the
    // complete set of wheel constraints. Requirements are observed before the
    // seen check below, preserving every declaring wheel for the final joint
    // route precheck.
    seen_candidate.extend(
        bundle
            .auto_routed
            .iter()
            .map(|route| canonical_conda_name(&route.route.pypi_name)),
    );
    let mut committed_relaxations = Vec::new();
    // incremental-add: pre-fill with locked names so we don't re-bundle them.
    if let Some(closure) = locked_closure {
        seen_candidate.extend(closure.keys().map(|n| canonical_conda_name(n)));
    }
    // Preserve every metadata requirement before `seen_candidate` collapses
    // repeated names. A rejected conda route must restore a PyPI version that
    // satisfies the intersection of all observed requirements, not whichever
    // wheel happened to be scanned first.
    let mut observed_requirements = ObservedRequirements::new();
    let mut provisional_metadata_routes: ProvisionalMetadataRoutes = BTreeMap::new();
    let mut standalone_provider_safe: BTreeSet<(String, String)> = BTreeSet::new();
    let marker_env = marker_env_for(&target.conda_subdir, &target.python_version)?;
    // v4.6: seed the first round with every exported closure wheel not
    // already in the bundle/skip set (see the `uv_closure_wheels` doc
    // above — deps-from roots are reachable ONLY through the closure).
    // Each seed hits the `is_closure_member` authoritative-bundle arm
    // below and is fetched at uv's exact resolved version.
    let mut closure_seed: Vec<(String, String)> = Vec::new();
    if let Some(pins) = uv_closure_wheels {
        for (name, version) in pins {
            if seen_candidate.insert(canonical_conda_name(name)) {
                closure_seed.push((name.clone(), version.clone()));
            }
        }
    }
    let mut processed_wheel_count = 0;
    loop {
        // Collect new candidates from wheels we haven't scanned yet.
        let mut candidates: Vec<(String, String)> = Vec::new();
        // v1.7.0: ALSO consider bare/ranged base deps. They use the same
        // requirement-specific validated-routing gate as exact deps.
        let mut loose_candidates: Vec<(String, VersionSpecifiers)> = Vec::new();
        for wheel in bundle.all_wheels().skip(processed_wheel_count) {
            for raw in &wheel.metadata.requires_dist {
                // p6w-1: NAME THE DISTRIBUTION. The holosoma failure that cost
                // 26 detected roots said only ``parsing requirement
                // `PyYAML (>=5.1.*)` `` -- with no owner, so the back-off could
                // not tell the one culprit from its 25 bystanders, and neither
                // could an operator reading the row. Every metadata parse
                // failure in this walk now carries the wheel that published the
                // clause.
                let owner = |e: anyhow::Error| {
                    e.context(format!(
                        "reading `Requires-Dist` of wheel `{}=={}`",
                        wheel.metadata.name, wheel.metadata.version
                    ))
                };
                if let Some((name, version)) =
                    pep508_exact_base_dep(raw, &marker_env).map_err(owner)?
                {
                    let specifiers = VersionSpecifiers::from_str(&format!("=={version}"))
                        .with_context(|| {
                            format!("parsing exact auto-bundle requirement `{name}=={version}`")
                        })?;
                    observe_requirement(
                        &mut observed_requirements,
                        &name,
                        &specifiers,
                        &wheel.metadata.name,
                        &wheel.metadata.version,
                        raw,
                        wheel.metadata_provenance.clone(),
                    );
                    let conda_name = canonical_conda_name(&name);
                    if !seen_candidate.insert(conda_name) {
                        continue;
                    }
                    candidates.push((name, version));
                } else if let Some((name, specs)) =
                    pep508_loose_base_dep(raw, &marker_env).map_err(owner)?
                {
                    observe_requirement(
                        &mut observed_requirements,
                        &name,
                        &specs,
                        &wheel.metadata.name,
                        &wheel.metadata.version,
                        raw,
                        wheel.metadata_provenance.clone(),
                    );
                    let conda_name = canonical_conda_name(&name);
                    if !seen_candidate.insert(conda_name) {
                        continue;
                    }
                    loose_candidates.push((name, specs));
                }
            }
        }
        processed_wheel_count = bundle.all_wheels().count();
        // v4.6: drain the one-time closure seed into this round's exact
        // candidates (first iteration; empty afterwards).
        candidates.append(&mut closure_seed);
        if candidates.is_empty() && loose_candidates.is_empty() {
            match jointly_unroute_unsolvable_with_route_precheck(
                bundle,
                &mut provisional_metadata_routes,
                &observed_requirements,
                indexes,
                target,
                &diagnostic_context,
                config,
                co_solve,
                prewarm_routes,
                validate_standalone_provider_route,
                &mut standalone_provider_safe,
                fetch_pypi,
                channels_consulted,
                probe_parallelism,
                uv_reresolve,
            )
            .await?
            {
                JointRouteOutcome::Unchanged => break,
                JointRouteOutcome::Mutated { relaxations } => {
                    committed_relaxations.extend(relaxations);
                    // The legacy path restored wheels. Scan restored metadata
                    // before accepting the remaining conda routes.
                    continue;
                }
                JointRouteOutcome::RetryKeepPypi { keep_pypi } => {
                    return Ok(AutoBundleOutcome::RetryKeepPypi { keep_pypi });
                }
            }
        }

        // PR-1 (Site 2): sort candidates by canonical name so routing is
        // confluent (processing order doesn't affect which spec wins when
        // the same dep appears in multiple wheels).
        candidates.sort_by_key(|(a, _)| canonical_conda_name(a));
        loose_candidates.sort_by_key(|(a, _)| canonical_conda_name(a));

        // Policy: prefer conda. If parselmouth (or our FALLBACK or the
        // user's retread-name-map) knows an unambiguous conda equivalent
        // for the PyPI name, skip bundling -- the dep flows through to
        // emission as a conda run-dep via `translate`, which uses the
        // same effective name_map.
        //
        // Why prefer conda for a conda-based tool: bundling vendors the
        // upstream-pinned version, but the conda copy is what every
        // other native package in the env was built against (BLAS,
        // glibc, CUDA, ABI in general). Double-installing a wheel on
        // top of a conda equivalent at best wastes disk and download
        // time; at worst it shadows the ABI-correct copy with one that
        // wasn't built for this env.
        //
        // Bundling still happens for everything parselmouth doesn't
        // know about (niche PyPI-only helpers). The fallback path below
        // is the original behavior, just with a smaller candidate set.
        //
        // Escape hatches when prefer-conda picks wrong: drop the dep
        // via `retread-drop-deps`, force a specific spec via
        // `retread-overrides`, or remove the parselmouth-discovered
        // entry by overriding it in `retread-name-map` (set to "" to
        // disable). For pin-forwarding conflicts arising on the PyPI
        // side, relax the offending editable's pyproject pin directly
        // (it's your code).

        // v1.4.0: batch this round's prefer-conda probes (16-way
        // bounded) instead of one serial await per candidate.
        let is_closure_member = |name: &str| {
            uv_closure_wheels.is_some_and(|s| s.contains_key(&canonical_conda_name(name)))
        };
        let held_by_uv_reresolve =
            |name: &str| uv_reresolve.keep_pypi.contains(&PypiKey::from_pypi(name));
        let prefer_pairs: Vec<(String, String)> = candidates
            .iter()
            .filter(|(name, _)| !is_closure_member(name))
            .filter(|(name, _)| !held_by_uv_reresolve(name))
            .filter_map(|(name, version)| {
                let pypi_key = PypiKey::from_pypi(name);
                config
                    .name_map
                    .get(&pypi_key)
                    .and_then(CondaTarget::mapped_name)
                    .map(|target| (target.as_spec().to_string(), format!("=={version}")))
            })
            .collect();
        let prefer_probes: std::collections::HashMap<(String, String), crate::probe::ProbeResult> =
            probe_many(prefer_pairs)
                .await
                .into_iter()
                .map(|r| ((r.package.clone(), r.spec.clone()), r))
                .collect();

        let mut added_any = false;
        // Candidates routed to PyPI this round; fetched concurrently
        // after the (serial, mutating) routing decisions below.
        // The 5th field is the favor-lock preferred version (if any) for that dep.
        let mut to_fetch: Vec<(String, String, String, VersionSpecifiers, Option<String>)> =
            Vec::new();
        for (name, version) in candidates {
            let conda_name = canonical_conda_name(&name);
            let pypi_key = PypiKey::from_pypi(&name);
            let specifiers = match VersionSpecifiers::from_str(&format!("=={version}")) {
                Ok(specifiers) => specifiers,
                Err(error) => {
                    tracing::debug!(
                        dep = %name, version = %version,
                        error = %error,
                        "auto-bundle: skipping unparseable version"
                    );
                    continue;
                }
            };
            let preferred_ver = favor_lock_prefs
                .and_then(|preferences| preferences.get(&conda_name))
                .cloned();
            if is_closure_member(&name) {
                // uv closure member not auto-routed: ships in the bundle,
                // no conda probes (see uv_closure_wheels doc above).
                bundle.probe_decisions.push(crate::audit::ProbeDecision {
                    stage: "auto_bundle".into(),
                    pypi_name: name.clone(),
                    conda_name: conda_name.clone(),
                    spec: format!("=={version}"),
                    target_python: target.python_version.clone(),
                    channels_consulted: vec![],
                    satisfiable: None,
                    matching_candidates: 0,
                    routing_decision: "uv-closure-authoritative-bundle".into(),
                });
                tracing::info!(
                    dep = %name,
                    version = %version,
                    "auto-bundle: uv closure member not moved to conda by the \
                     auto-route; bundling from PyPI (closure is authoritative)",
                );
            } else if held_by_uv_reresolve(&name) {
                bundle.probe_decisions.push(crate::audit::ProbeDecision {
                    stage: "auto_bundle".into(),
                    pypi_name: name.clone(),
                    conda_name: conda_name.clone(),
                    spec: format!("=={version}"),
                    target_python: target.python_version.clone(),
                    channels_consulted: vec![],
                    satisfiable: None,
                    matching_candidates: 0,
                    routing_decision: "uv-reresolve-keep-pypi".into(),
                });
            } else if prefer_conda_match(&pypi_key, &config.name_map) {
                // Probe the workspace's conda channels for whether the
                // spec retread would emit is actually satisfiable. If
                // ANY channel has a matching candidate, keep on conda.
                // Only a concrete match may route to conda. Empty,
                // unsatisfied, or indecisive results stay on PyPI.
                let conda_target_name = config.name_map[&pypi_key]
                    .mapped_name()
                    .expect("prefer_conda_match accepted only a mapped target")
                    .as_spec()
                    .to_string();
                let probe_spec = format!("=={version}");
                let probe_result =
                    match prefer_probes.get(&(conda_target_name.clone(), probe_spec.clone())) {
                        Some(r) => r.clone(),
                        // Defensive: shouldn't happen (pairs built from the
                        // same predicate), but fall back to a direct probe
                        // rather than mis-routing.
                        None => probe_many(vec![(conda_target_name.clone(), probe_spec.clone())])
                            .await
                            .into_iter()
                            .next()
                            .unwrap_or_else(|| {
                                indecisive_probe(conda_target_name.clone(), probe_spec.clone())
                            }),
                    };
                let route_to_conda = validated_conda_route(&probe_result);
                let routing_decision = if route_to_conda {
                    "short-circuit"
                } else {
                    "fall-through-to-pypi"
                };
                bundle
                    .probe_decisions
                    .push(crate::audit::ProbeDecision::from_probe(
                        "auto_bundle",
                        &name,
                        &conda_target_name,
                        &probe_spec,
                        &target.python_version,
                        &probe_result,
                        routing_decision,
                    ));
                if !route_to_conda {
                    tracing::info!(
                        dep = %name,
                        conda = %conda_target_name,
                        spec = %probe_spec,
                        channels = ?probe_result.channels_consulted,
                        "prefer-conda: no validated conda candidate satisfies the dependency; auto-bundling from PyPI",
                    );
                } else {
                    record_metadata_route(
                        &mut provisional_metadata_routes,
                        name.clone(),
                        conda_target_name.clone(),
                        preferred_ver.clone(),
                    );
                    tracing::info!(
                        dep = %name,
                        conda = %conda_target_name,
                        spec = %probe_spec,
                        matches = probe_result.matching_candidates,
                        decision = %routing_decision,
                        "prefer-conda: skipping auto-bundle; dep will be emitted as a conda run-dep",
                    );
                    continue;
                }
            }
            to_fetch.push((name, version, conda_name, specifiers, preferred_ver));
        }

        // Loose (bare/ranged) candidates use the same requirement-specific,
        // fail-closed routing rule as exact candidates.
        let loose_pairs: Vec<(String, String)> = loose_candidates
            .iter()
            .filter(|(name, _)| !is_closure_member(name))
            .filter(|(name, _)| !held_by_uv_reresolve(name))
            .filter_map(|(name, specs)| {
                let pypi_key = PypiKey::from_pypi(name);
                let target_name = match config.name_map.get(&pypi_key) {
                    Some(target) => target
                        .mapped_name()
                        .map(|target| target.as_spec().to_string()),
                    None => Some(pypi_key.as_str().to_string()),
                }?;
                Some((target_name, conda_probe_spec(specs)))
            })
            .collect();
        let loose_probes: std::collections::HashMap<(String, String), crate::probe::ProbeResult> =
            probe_many(loose_pairs)
                .await
                .into_iter()
                .map(|r| ((r.package.clone(), r.spec.clone()), r))
                .collect();
        for (name, specs) in loose_candidates {
            let conda_name = canonical_conda_name(&name);
            let pypi_key = PypiKey::from_pypi(&name);
            let preferred_ver = favor_lock_prefs
                .and_then(|preferences| preferences.get(&conda_name))
                .cloned();
            if is_closure_member(&name) {
                // Same authoritative-closure rule as the exact arm above.
                bundle.probe_decisions.push(crate::audit::ProbeDecision {
                    stage: "auto_bundle_loose".into(),
                    pypi_name: name.clone(),
                    conda_name: conda_name.clone(),
                    spec: specs.to_string(),
                    target_python: target.python_version.clone(),
                    channels_consulted: vec![],
                    satisfiable: None,
                    matching_candidates: 0,
                    routing_decision: "uv-closure-authoritative-bundle".into(),
                });
                tracing::info!(
                    dep = %name,
                    specs = %specs,
                    "auto-bundle: uv closure member (loose spec) not moved to conda \
                     by the auto-route; bundling from PyPI (closure is authoritative)",
                );
                to_fetch.push((name, specs.to_string(), conda_name, specs, preferred_ver));
                continue;
            }
            if held_by_uv_reresolve(&name) {
                bundle.probe_decisions.push(crate::audit::ProbeDecision {
                    stage: "auto_bundle_loose".into(),
                    pypi_name: name.clone(),
                    conda_name: conda_name.clone(),
                    spec: specs.to_string(),
                    target_python: target.python_version.clone(),
                    channels_consulted: vec![],
                    satisfiable: None,
                    matching_candidates: 0,
                    routing_decision: "uv-reresolve-keep-pypi".into(),
                });
                to_fetch.push((name, specs.to_string(), conda_name, specs, preferred_ver));
                continue;
            }
            let target_name = match config.name_map.get(&pypi_key) {
                Some(target) => match target.mapped_name() {
                    Some(target) => target.as_spec().to_string(),
                    None => {
                        to_fetch.push((name, specs.to_string(), conda_name, specs, preferred_ver));
                        continue;
                    }
                },
                None => conda_name.clone(),
            };
            let probe_spec = conda_probe_spec(&specs);
            let probe_result = match loose_probes.get(&(target_name.clone(), probe_spec.clone())) {
                Some(r) => r.clone(),
                None => probe_many(vec![(target_name.clone(), probe_spec.clone())])
                    .await
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| indecisive_probe(target_name.clone(), probe_spec.clone())),
            };
            let route_to_conda = validated_conda_route(&probe_result);
            bundle
                .probe_decisions
                .push(crate::audit::ProbeDecision::from_probe(
                    "auto_bundle_loose",
                    &name,
                    &target_name,
                    &probe_spec,
                    &target.python_version,
                    &probe_result,
                    if route_to_conda {
                        "short-circuit"
                    } else {
                        "fall-through-to-pypi"
                    },
                ));
            if route_to_conda {
                record_metadata_route(
                    &mut provisional_metadata_routes,
                    name,
                    target_name,
                    preferred_ver,
                );
                continue;
            }
            tracing::info!(
                dep = %name,
                specs = %specs,
                "auto-bundle: no validated conda candidate satisfies bare/ranged dep; bundling from PyPI",
            );
            to_fetch.push((name, specs.to_string(), conda_name, specs, preferred_ver));
        }

        // v1.4.0: fetch this round's PyPI-bound wheels concurrently
        // (8-way bounded). `buffered` (not buffer_unordered) preserves
        // candidate order, so extras order -- and therefore the next
        // round's Requires-Dist scan order -- stays deterministic.
        // Per item, the index fallback chain is walked serially. The shared
        // BFS fetcher supplies the wheel -> sdist-build fallback. Exhausting
        // the chain is an error because conda routing was already refused.
        // EVERY WHEEL ADMITTED INTO THE PACK CLOSURE IS RESOLVED UNDER THE
        // WORKSPACE'S CONDA FACTS (N27-RETREAD-145), not only the ones the
        // joint-solve restore admits.
        //
        // WHY THIS SITE AND NOT A UV CLOSURE CONSTRAINT. The obvious reading of
        // the production RED is that `protobuf` sits in `auto_dropped` -- it is
        // workspace-owned, the conda provider holds `5.29.3` -- so the pack's uv
        // closure resolve never sees it as a requirement to satisfy and freely
        // picks a `googleapis-common-protos` that needs `>=6.33.5`. MEASURED in
        // relock `6112256`'s 88 MB backend log, that is NOT what happened: no
        // closure resolve admitted this dep at all. The two rows, adjacent, are
        // `pypi: fetching simple index url=https://pypi.org/simple/
        // googleapis-common-protos/` and `auto_bundle: auto-bundled into
        // isaaclab-2-3x-pack dep=googleapis-common-protos version=~=1.52` -- the
        // LOOSE arm of this very loop, fetching under a bare `~=1.52` from a
        // parent wheel's `Requires-Dist`, and `bfs_fetch_pypi_from_chain`
        // returning the highest release that satisfies it (1.75.3 today).
        // Constraining a uv closure would therefore have fixed nothing. The
        // fact has to reach THIS fetch, which is where the selection is made.
        //
        // The actuator is the same one the restore uses, extracted whole
        // (`fetch_under_workspace_facts`): backtrack by exclusion until the
        // selection's `Requires-Dist` admits every fact, and when the index
        // runs out REFUSE loudly naming dep, requirement and fact rather than
        // emitting a `constrains` bound no consuming environment can satisfy.
        let fact_versions = admission_fact_holdings(bundle);
        let admission_reresolve = uv_reresolve.mode.is_enabled();
        let bundle_label = bundle.conda_name.clone();
        let fetched: Vec<Result<(String, String, ResolvedWheel, Option<FactCheck>)>> = {
            use futures::stream::{self, StreamExt};
            let indexes_ref = indexes;
            let fact_versions = &fact_versions;
            let marker_env_ref = &marker_env;
            let bundle_label = bundle_label.as_str();
            stream::iter(to_fetch)
                .map(
                    |(name, version, conda_name, specifiers, preferred_ver)| async move {
                        let request = PypiFetchRequest {
                            pypi_name: name.clone(),
                            bundle_name: conda_name,
                            specifiers: specifiers.clone(),
                            preferred_version: preferred_ver,
                            relaxations: Vec::new(),
                        };
                        let failure_context = format!(
                            "auto-bundle: no PyPI index could resolve `{name}{specifiers}` after conda routing was refused"
                        );
                        fetch_under_workspace_facts(
                            request,
                            indexes_ref.to_vec(),
                            failure_context,
                            fact_versions,
                            marker_env_ref,
                            bundle_label,
                            FactConstrainedSite::AutoBundleAdmission,
                            admission_reresolve,
                            fetch_pypi,
                        )
                        .await
                        .map(|(wheel, moved)| (name, version, wheel, moved))
                    },
                )
                .buffered(8)
                .collect()
                .await
        };
        let mut admitted_any = false;
        let mut backtracked: Vec<(String, String, String, usize)> = Vec::new();
        for result in fetched {
            let (name, version, wheel, moved) = result?;
            if let Some(check) = moved {
                if check.from != check.to {
                    backtracked.push((
                        name.clone(),
                        check.from,
                        check.to,
                        check.holders_total,
                    ));
                }
            }
            tracing::info!(
                dep = %name,
                version = %version,
                selected = %wheel.metadata.version,
                "auto-bundled into {}",
                bundle.conda_name,
            );
            bundle.extras.push(wheel);
            admitted_any = true;
            added_any = true;
        }
        if admitted_any {
            // STDERR, NEVER STDOUT: `rpc.rs` owns stdout as the JSON-RPC
            // channel, exactly as the RERESOLVED row above.
            eprintln!(
                "{}",
                pypi_closure_fact_constrained_row(fact_versions.len(), &backtracked)
            );
        }

        // Scan metadata from every wheel fetched this round before finalizing
        // routes. Otherwise a later wheel can tighten a requirement after its
        // conda route has already been rejected and restored from PyPI.
        if !added_any {
            match jointly_unroute_unsolvable_with_route_precheck(
                bundle,
                &mut provisional_metadata_routes,
                &observed_requirements,
                indexes,
                target,
                &diagnostic_context,
                config,
                co_solve,
                prewarm_routes,
                validate_standalone_provider_route,
                &mut standalone_provider_safe,
                fetch_pypi,
                channels_consulted,
                probe_parallelism,
                uv_reresolve,
            )
            .await?
            {
                JointRouteOutcome::Unchanged => {}
                JointRouteOutcome::Mutated { relaxations } => {
                    committed_relaxations.extend(relaxations);
                    added_any = true;
                }
                JointRouteOutcome::RetryKeepPypi { keep_pypi } => {
                    return Ok(AutoBundleOutcome::RetryKeepPypi { keep_pypi });
                }
            }
        }

        // Loop again only if we added at least one wheel; the new
        // wheels' Requires-Dist may need further auto-bundling.
        if !added_any {
            break;
        }
    }
    Ok(AutoBundleOutcome::Complete {
        relaxations: committed_relaxations,
    })
}

fn restored_route_pypi_names(
    bundle: &Bundle,
    metadata_routes: &ProvisionalMetadataRoutes,
    conda_name: &str,
) -> BTreeSet<String> {
    bundle
        .auto_routed
        .iter()
        .filter(|route| canonical_conda_name(&route.route.conda_name) == conda_name)
        .map(|route| route.route.pypi_name.clone())
        .chain(
            metadata_routes
                .get(conda_name)
                .into_iter()
                .flatten()
                .map(|origin| origin.pypi_name.clone()),
        )
        .collect()
}

type IndexedEmissionSupport = (String, super::EmissionSupport);

struct EmissionSupportIndex {
    support_counts: BTreeMap<String, usize>,
    auto_by_owner: BTreeMap<String, BTreeSet<IndexedEmissionSupport>>,
    wheel_by_identity: BTreeMap<String, BTreeSet<IndexedEmissionSupport>>,
}

impl EmissionSupportIndex {
    fn new(supports_by_conda: BTreeMap<String, BTreeSet<super::EmissionSupport>>) -> Self {
        let mut support_counts = BTreeMap::new();
        let mut auto_by_owner: BTreeMap<_, BTreeSet<_>> = BTreeMap::new();
        let mut wheel_by_identity: BTreeMap<_, BTreeSet<_>> = BTreeMap::new();
        for (group, supports) in supports_by_conda {
            support_counts.insert(group.clone(), supports.len());
            for support in supports {
                let indexed = (group.clone(), support.clone());
                match &support {
                    super::EmissionSupport::AutoRoute { owner_conda_name } => {
                        auto_by_owner
                            .entry(owner_conda_name.clone())
                            .or_default()
                            .insert(indexed);
                    }
                    super::EmissionSupport::WheelRequirement {
                        translated_conda_name,
                        raw_pypi_name,
                    } => {
                        for identity in [translated_conda_name, raw_pypi_name] {
                            wheel_by_identity
                                .entry(canonical_conda_name(identity))
                                .or_default()
                                .insert(indexed.clone());
                        }
                    }
                }
            }
        }
        Self {
            support_counts,
            auto_by_owner,
            wheel_by_identity,
        }
    }
}

/// Prove from the already-finalized support index that removing this route
/// deletes its complete emission group and cannot perturb any other group.
///
/// Ambiguous alias/cross-group cases retain the exact counterfactual renderer
/// below. The ordinary one-owner route path is O(1) after the single base
/// emission instead of rebuilding every dependency group.
fn support_index_proves_isolated_route_removal(
    support_index: &EmissionSupportIndex,
    conda_name: &str,
    restored_pypi_names: &BTreeSet<String>,
) -> bool {
    let Some(&target_support_count) = support_index.support_counts.get(conda_name) else {
        return false;
    };
    if target_support_count == 0 {
        return false;
    }
    let mut removed = support_index
        .auto_by_owner
        .get(conda_name)
        .cloned()
        .unwrap_or_default();
    for identity in restored_pypi_names {
        if let Some(wheel_supports) = support_index
            .wheel_by_identity
            .get(&canonical_conda_name(identity))
        {
            removed.extend(wheel_supports.iter().cloned());
        }
    }
    removed.len() == target_support_count && removed.iter().all(|(group, _)| group == conda_name)
}

/// Does this canonical conda dependency disappear if every provisional
/// origin in its group is restored to PyPI? A remaining dependency has a
/// fixed/user-owned origin and must not be offered to Rule 2 as mutable.
fn route_group_is_fully_mutable(
    bundle: &Bundle,
    metadata_routes: &ProvisionalMetadataRoutes,
    support_index: &EmissionSupportIndex,
    conda_name: &str,
    config: &RetreadConfig,
    target: &crate::pypi::WheelTarget,
) -> Result<bool> {
    let restored_pypi_names = restored_route_pypi_names(bundle, metadata_routes, conda_name);
    if support_index_proves_isolated_route_removal(support_index, conda_name, &restored_pypi_names)
    {
        return Ok(true);
    }
    route_group_is_fully_mutable_counterfactual(
        bundle,
        conda_name,
        config,
        target,
        restored_pypi_names,
    )
}

fn route_group_is_fully_mutable_counterfactual(
    bundle: &Bundle,
    conda_name: &str,
    config: &RetreadConfig,
    target: &crate::pypi::WheelTarget,
    restored_pypi_names: BTreeSet<String>,
) -> Result<bool> {
    let mut trial = bundle.clone();
    trial
        .auto_routed
        .retain(|route| canonical_conda_name(&route.route.conda_name) != conda_name);

    // Only the vendored identity matters for this counterfactual. Empty
    // metadata prevents the placeholder itself from adding dependencies.
    for pypi_name in restored_pypi_names {
        if trial
            .all_wheels()
            .any(|wheel| canonical_conda_name(&wheel.pypi_name) == canonical_conda_name(&pypi_name))
        {
            continue;
        }
        let mut placeholder = trial.primary.clone();
        placeholder.pypi_name = pypi_name.clone();
        placeholder.metadata.name = pypi_name;
        placeholder.metadata.requires_dist.clear();
        trial.extras.push(placeholder);
    }

    let assembly = super::emitted_bundle_route_assembly(&trial, config, target)?;
    Ok(assembly
        .routes
        .iter()
        .all(|route| route.conda_name.key().as_str() != conda_name)
        && assembly
            .conflicts
            .iter()
            .all(|conflict| conflict.conda_name.key().as_str() != conda_name))
}

/// Does this canonical conda dependency have metadata supplied by a
/// source-built wheel?
///
/// Source-built routes still participate in Rule 2: the dependency wheel can
/// be bundled even though the declaring root wheel already ships in the pack.
/// We retain the provenance distinction so an independently-unsatisfiable
/// fixed baseline does not spuriously move every source-root dependency to
/// PyPI. A source route is restored only after a satisfiable baseline proves
/// that route (or a route group containing it) is the conflict.
fn metadata_route_group_has_source_built_origin(
    metadata_routes: &ProvisionalMetadataRoutes,
    observed_requirements: &ObservedRequirements,
    conda_name: &str,
) -> Result<bool> {
    let Some(origins) = metadata_routes.get(conda_name) else {
        return Ok(false);
    };
    let mut has_source_built_origin = false;
    for origin in origins {
        let pypi_key = PypiKey::from_pypi(&origin.pypi_name);
        let requirements = observed_requirements.get(&pypi_key).ok_or_else(|| {
            anyhow!(
                "cannot verify whether PyPI dependency `{}` should use conda package `{}` because \
                 its source requirement is unavailable",
                origin.pypi_name,
                origin.conda_name
            )
        })?;
        if requirements.is_empty() {
            return Err(anyhow!(
                "cannot verify whether PyPI dependency `{}` should use conda package `{}` because \
                 it has no active source requirements",
                origin.pypi_name,
                origin.conda_name
            ));
        }
        has_source_built_origin |= requirements
            .iter()
            .any(|requirement| matches!(requirement.provenance, Provenance::SourceBuiltRelaxed));
    }
    Ok(has_source_built_origin)
}

/// Finalize every provisional conda route against the exact dependency set
/// this bundle would emit. Uv routes and metadata-probe routes, including
/// requirements declared by source-built roots, are mutable until this check
/// succeeds. Rejected routes are restored through the same ordered PyPI
/// fallback chain before the bundle is changed.
async fn jointly_unroute_unsolvable<C, CF, X, XF>(
    bundle: &mut Bundle,
    metadata_routes: &mut ProvisionalMetadataRoutes,
    observed_requirements: &ObservedRequirements,
    indexes: &[String],
    target: &crate::pypi::WheelTarget,
    config: &RetreadConfig,
    co_solve: &C,
    fetch_pypi: &X,
    channels_consulted: &[String],
    uv_reresolve: &UvReresolveContext,
) -> Result<JointRouteOutcome>
where
    C: Fn(Vec<crate::uv_closure::CondaRouteSpec>) -> CF,
    CF: Future<Output = crate::uv_closure::CoInstallVerdict>,
    X: Fn(PypiFetchRequest, Vec<String>, String) -> XF,
    XF: Future<Output = Result<ResolvedWheel>>,
{
    let allow_source_route = |_route: crate::uv_closure::CondaRouteSpec| async {
        crate::uv_closure::CoInstallVerdict::Sat
    };
    let no_prewarm = |_routes: Vec<crate::uv_closure::CondaRouteSpec>| async {};
    let mut standalone_provider_safe = BTreeSet::new();
    let diagnostic_context = JointRouteDiagnosticContext::new(bundle, target, None);
    jointly_unroute_unsolvable_with_route_precheck(
        bundle,
        metadata_routes,
        observed_requirements,
        indexes,
        target,
        &diagnostic_context,
        config,
        co_solve,
        &no_prewarm,
        &allow_source_route,
        &mut standalone_provider_safe,
        fetch_pypi,
        channels_consulted,
        1,
        uv_reresolve,
    )
    .await
}

fn should_eagerly_prewarm_probe_routes(
    probe_parallelism: usize,
    has_mutable_candidates: bool,
) -> bool {
    probe_parallelism > 1 && has_mutable_candidates
}

/// Bounded concurrency for the post-rejection PyPI restore fetches (fix
/// f17). Matches the 8-way bound already used for the per-round fetch.
pub(crate) const RESTORE_FETCH_CONCURRENCY: usize = 8;

/// Run `make` over `items` with bounded concurrency, preserving INPUT
/// order in the output.
///
/// `buffered` (not `buffer_unordered`) is deliberate: the results become
/// bundle extras, and extras order decides the next round's
/// `Requires-Dist` scan order, so the output must stay deterministic even
/// though the requests overlap on the wire. The first `Err` short-circuits,
/// preserving the old `await?` failure semantics.
pub(crate) async fn fetch_bounded_concurrent<I, T, F, Fut, R>(
    items: I,
    concurrency: usize,
    make: F,
) -> Result<Vec<R>>
where
    I: IntoIterator<Item = T>,
    F: Fn(T) -> Fut,
    Fut: std::future::Future<Output = Result<R>>,
{
    use futures::stream::{self, StreamExt, TryStreamExt};
    stream::iter(items.into_iter().map(make))
        .buffered(concurrency.max(1))
        .try_collect()
        .await
}


/// The row a joint-solve route restore writes so the decision has a reader.
///
/// TRIGGER-1 measured that this decision — the one that turns an upstream
/// dependency add into a pack that bundles a wheel instead of routing it to
/// conda, and therefore into a `Requires-Dist` that can cross a learned
/// workspace conda fact — has exactly two sinks today and BOTH are unreachable
/// after the fact: a `tracing::warn!` into the backend log, which
/// N27-RETREAD-133 established the arm gzips and `rm -f`s, and a probe-trace
/// JSON under the job's ephemeral fast root (`grep -c "probe trace"` = 0 in
/// MDP-6097871's backend log; 12 in C37 a1, whose destination no longer
/// exists). Under law 2 that is a writer with no reader.
///
/// `crosses` names the workspace conda providers this restored wheel's own
/// requirements mention, which is the precursor of the cap-vs-fact collision
/// that surfaces two layers later. An empty `crosses` is still printed: a
/// route row filtered by the very condition it exists to give early warning of
/// is not early warning.
///
/// SIZING, MEASURED, AND TRIGGER-1'S PROPOSED FILTER IS A NO-OP HERE. It asked
/// that the row print "only for a restored wheel that ends up `owner=pack`".
/// Every wheel restored at this site is pushed into `trial.extras`, i.e. is
/// shipped by this pack, so that filter selects all of them and changes no
/// volume. The true volume is therefore the restore count itself: 12 for the
/// pack in MDP-6097871 and 442 across C37 arm a1.
pub(crate) fn pypi_route_restored_row(
    pypi_name: &str,
    conda_name: &str,
    bundle_name: &str,
    crosses: &[String],
) -> String {
    format!(
        "### PYPI ROUTE RESTORED dep={pypi_name} conda={conda_name} bundle={bundle_name} \
         reason=joint-co-solve-rejected crosses={}",
        if crosses.is_empty() {
            "none".to_string()
        } else {
            crosses.join(",")
        }
    )
}

/// The workspace conda provider names a restored wheel's `Requires-Dist`
/// mentions. Deliberately a plain name match against the fact boundary: the
/// row is a pointer for a reader, not a solver. A version-aware answer here
/// would have to re-decide the cap-versus-fact question on the real
/// constraint set, which is a different capability and is not on this base --
/// the row exists to say WHICH names are about to meet that boundary, early
/// enough to be read, and nothing more.
pub(crate) fn route_restore_crossings(
    requires_dist: &[String],
    fact_names: &BTreeSet<String>,
) -> Vec<String> {
    let mut crossings: BTreeSet<String> = BTreeSet::new();
    for requirement in requires_dist {
        let name: String = requirement
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '-' || *c == '_' || *c == '.')
            .collect();
        if name.is_empty() {
            continue;
        }
        let key = super::canonical_conda_name(&name);
        if fact_names.contains(&key) {
            crossings.insert(key);
        }
    }
    crossings.into_iter().collect()
}
async fn jointly_unroute_unsolvable_with_route_precheck<C, CF, W, WF, V, VF, X, XF>(
    bundle: &mut Bundle,
    metadata_routes: &mut ProvisionalMetadataRoutes,
    observed_requirements: &ObservedRequirements,
    indexes: &[String],
    target: &crate::pypi::WheelTarget,
    diagnostic_context: &JointRouteDiagnosticContext,
    config: &RetreadConfig,
    co_solve: &C,
    prewarm_routes: &W,
    validate_standalone_provider_route: &V,
    standalone_provider_safe: &mut BTreeSet<(String, String)>,
    fetch_pypi: &X,
    channels_consulted: &[String],
    probe_parallelism: usize,
    uv_reresolve: &UvReresolveContext,
) -> Result<JointRouteOutcome>
where
    C: Fn(Vec<crate::uv_closure::CondaRouteSpec>) -> CF,
    CF: Future<Output = crate::uv_closure::CoInstallVerdict>,
    W: Fn(Vec<crate::uv_closure::CondaRouteSpec>) -> WF,
    WF: Future<Output = ()>,
    V: Fn(crate::uv_closure::CondaRouteSpec) -> VF,
    VF: Future<Output = crate::uv_closure::CoInstallVerdict>,
    X: Fn(PypiFetchRequest, Vec<String>, String) -> XF,
    XF: Future<Output = Result<ResolvedWheel>>,
{
    if bundle.auto_routed.is_empty() && metadata_routes.is_empty() {
        return Ok(JointRouteOutcome::Unchanged);
    }

    let assembly = super::emitted_bundle_route_assembly(bundle, config, target)?;
    let emitted = assembly.routes;
    let assembly_conflicts = assembly.conflicts;
    let support_index = EmissionSupportIndex::new(assembly.supports_by_conda);
    let emitted_by_conda: BTreeMap<String, crate::uv_closure::CondaRouteSpec> = emitted
        .iter()
        .cloned()
        .map(|route| (route.conda_name.key().into_string(), route))
        .collect();
    let mut fixed_by_config: HashSet<String> = config
        .force_conda
        .iter()
        .map(|name| canonical_conda_name(name))
        .chain(
            config
                .conda_deps
                .iter()
                .map(|name| canonical_conda_name(name)),
        )
        .chain(
            config
                .overrides
                .keys()
                .map(|name| canonical_conda_name(name)),
        )
        .collect();
    expand_name_map_groups(&mut fixed_by_config, &config.name_map);
    let abi_aliases = super::output_abi_aliases(bundle, config);

    // Route ownership is grouped by the canonical conda dependency because
    // emission deduplicates there. If any alias in a group is force-conda,
    // the whole emitted dependency is fixed rather than partially mutable.
    let route_conda_names: BTreeSet<String> = bundle
        .auto_routed
        .iter()
        .map(|route| canonical_conda_name(&route.route.conda_name))
        .chain(metadata_routes.keys().cloned())
        .collect();
    let mut mutable_conda_names = BTreeSet::new();
    let mut source_metadata_conda_names = BTreeSet::new();
    for conda_name in route_conda_names {
        if fixed_by_config.contains(&conda_name) {
            continue;
        }
        let uv_forced = bundle.auto_routed.iter().any(|route| {
            canonical_conda_name(&route.route.conda_name) == conda_name
                && fixed_by_config.contains(&canonical_conda_name(&route.route.pypi_name))
        });
        let metadata_forced = metadata_routes.get(&conda_name).is_some_and(|origins| {
            origins
                .iter()
                .any(|origin| fixed_by_config.contains(&canonical_conda_name(&origin.pypi_name)))
        });
        if uv_forced || metadata_forced {
            continue;
        }
        let route_group_is_abi_anchor = super::is_semantic_abi_anchor(&conda_name, &abi_aliases)
            || bundle.auto_routed.iter().any(|route| {
                canonical_conda_name(&route.route.conda_name) == conda_name
                    && (super::is_semantic_abi_anchor(&route.route.pypi_name, &abi_aliases)
                        || super::is_semantic_abi_anchor(&route.route.conda_name, &abi_aliases))
            })
            || metadata_routes.get(&conda_name).is_some_and(|origins| {
                origins.iter().any(|origin| {
                    super::is_semantic_abi_anchor(&origin.pypi_name, &abi_aliases)
                        || super::is_semantic_abi_anchor(&origin.conda_name, &abi_aliases)
                })
            });
        // Route rejection changes the provider ecosystem. ABI anchors keep
        // their validated conda provider even when the corresponding wheel
        // constraints could be reconciled after removing that route.
        if route_group_is_abi_anchor {
            continue;
        }
        if metadata_route_group_has_source_built_origin(
            metadata_routes,
            observed_requirements,
            &conda_name,
        )
        .map_err(|error| diagnostic_context.routing_error(format!("{error:#}")))?
        {
            source_metadata_conda_names.insert(conda_name.clone());
        }
        if route_group_is_fully_mutable(
            bundle,
            metadata_routes,
            &support_index,
            &conda_name,
            config,
            target,
        )? {
            mutable_conda_names.insert(conda_name);
        }
    }
    let fixed_conflicts: Vec<_> = assembly_conflicts
        .iter()
        .filter(|conflict| !mutable_conda_names.contains(conflict.conda_name.key().as_str()))
        .map(|conflict| {
            conflict
                .conflict
                .clone()
                .with_scope(diagnostic_context.scope())
        })
        .collect();
    let conflicted_keys: BTreeSet<String> = assembly_conflicts
        .iter()
        .map(|conflict| conflict.conda_name.key().into_string())
        .collect();
    let mut pre_rejected: Vec<_> = conflicted_keys
        .iter()
        .filter(|name| mutable_conda_names.contains(*name))
        .filter_map(|name| emitted_by_conda.get(name).cloned())
        .collect();
    let mut mutable_candidates: Vec<_> = mutable_conda_names
        .iter()
        .filter(|name| !conflicted_keys.contains(*name))
        .filter_map(|name| emitted_by_conda.get(name).cloned())
        .collect();
    if mutable_candidates.is_empty() && pre_rejected.is_empty() {
        return if fixed_conflicts.is_empty() {
            Ok(JointRouteOutcome::Unchanged)
        } else {
            Err(aggregate_conflicts(fixed_conflicts))
        };
    }

    let mutable_keys: HashSet<String> = mutable_conda_names.iter().cloned().collect();
    let fixed: Vec<_> = emitted
        .iter()
        .filter(|route| !mutable_keys.contains(route.conda_name.key().as_str()))
        .cloned()
        .collect();
    // Load the complete reachable record union before a parallel standalone
    // provider batch introduces roots one at a time. Serial mode keeps the
    // shared cache but grows it lazily; the eager prewarm was introduced only
    // to make independent concurrent solves see the same complete snapshot.
    if should_eagerly_prewarm_probe_routes(probe_parallelism, !mutable_candidates.is_empty()) {
        let full_routes = fixed
            .iter()
            .chain(&mutable_candidates)
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        prewarm_routes(full_routes).await;
    }

    // A globally unsatisfiable fixed baseline cannot identify ordinary route
    // conflicts, so source-built metadata routes normally remain on conda.
    // Direct workspace PyPI provider shadows are route-local: solve each
    // mutable source route by itself and reject only a positively conflicting
    // route. These small trials run concurrently and avoid both the unsafe
    // fail-open (`tensordict -> pytorch` against PyPI torch) and an exhaustive
    // whole-workspace core reduction.
    let provider_trials = {
        use futures::stream::{self, StreamExt};
        // Own the selected routes before entering the buffered async stream.
        // Keeping the slice iterator in the stream made Handler::dispatch's
        // opaque future fail the RPC server's higher-ranked Send requirement.
        let routes_to_validate: Vec<_> = mutable_candidates
            .iter()
            .filter(|route| {
                source_metadata_conda_names.contains(route.conda_name.key().as_str())
                    && !standalone_provider_safe
                        .contains(&(route.conda_name.key().into_string(), route.spec.clone()))
            })
            .cloned()
            .collect();
        stream::iter(routes_to_validate)
            .map(|route| async move {
                let verdict = validate_standalone_provider_route(route.clone()).await;
                (route, verdict)
            })
            .buffered(probe_parallelism.max(1))
            .collect::<Vec<_>>()
            .await
    };
    let mut provider_rejected_keys = BTreeSet::new();
    for (route, verdict) in provider_trials {
        match verdict {
            crate::uv_closure::CoInstallVerdict::Unsat(reasons)
            | crate::uv_closure::CoInstallVerdict::ExactUnsat(reasons) => {
                tracing::info!(
                    route = %route.conda_name,
                    reasons = ?reasons,
                    "joint route solve: standalone source route rejected before fixed-baseline validation",
                );
                provider_rejected_keys.insert(route.conda_name.key().into_string());
                pre_rejected.push(route);
            }
            crate::uv_closure::CoInstallVerdict::Sat => {
                standalone_provider_safe
                    .insert((route.conda_name.key().into_string(), route.spec.clone()));
            }
            crate::uv_closure::CoInstallVerdict::Skipped(reason) => {
                tracing::debug!(
                    route = %route.conda_name,
                    reason = %reason,
                    "joint route solve: standalone source-route validation unavailable; preserving baseline fail-open",
                );
            }
        }
    }
    mutable_candidates
        .retain(|route| !provider_rejected_keys.contains(route.conda_name.key().as_str()));
    let selection = if mutable_candidates.is_empty() {
        Some(crate::uv_closure::JointRouteSelection {
            accepted: Vec::new(),
            rejected: Vec::new(),
        })
    } else {
        crate::uv_closure::select_jointly_solvable_routes_for_bundle(
            &diagnostic_context.bundle,
            fixed,
            mutable_candidates.clone(),
            co_solve,
            probe_parallelism,
        )
        .await
    };
    // Rule 2 is fail-closed for ordinary mutable routes: an unsatisfiable or
    // indeterminate baseline cannot authorize them. Source-root metadata
    // routes retain their existing conda placement until a satisfiable
    // baseline positively identifies a conflicting route; otherwise an
    // unrelated broken baseline would vendor every source-root dependency.
    // Typed assembly conflicts are always pre-rejected, then pass through the
    // same restore gate below.
    let mut rejected = match selection {
        Some(selection) => selection.rejected,
        None => mutable_candidates
            .into_iter()
            .filter(|route| !source_metadata_conda_names.contains(route.conda_name.key().as_str()))
            .collect(),
    };
    rejected.extend(pre_rejected);
    let mut seen_rejected = BTreeSet::new();
    rejected.retain(|route| seen_rejected.insert(route.conda_name.key().into_string()));
    if rejected.is_empty() {
        return if fixed_conflicts.is_empty() {
            Ok(JointRouteOutcome::Unchanged)
        } else {
            Err(aggregate_conflicts(fixed_conflicts))
        };
    }
    let rejected_keys: BTreeSet<String> = rejected
        .iter()
        .map(|route| route.conda_name.key().into_string())
        .collect();

    if uv_reresolve.mode.is_enabled() && uv_reresolve.uv_backed && fixed_conflicts.is_empty() {
        // A keep-PyPI retry is useful only when every structural conflict is
        // mutable. If a fixed route already conflicts, continue through the
        // local restore preflight so its mutable peers can be reported with
        // it in this request.
        // Preserve every PyPI origin, not just CondaRouteSpec::pypi_name:
        // emission groups aliases by conda identity, while uv's keep-pypi
        // policy is keyed by the original PyPI identity.
        let mut rejected_pypi_origins = BTreeSet::new();
        for route in &bundle.auto_routed {
            if rejected_keys.contains(&canonical_conda_name(&route.route.conda_name)) {
                rejected_pypi_origins.insert(PypiKey::from_pypi(&route.route.pypi_name));
            }
        }
        for conda_name in &rejected_keys {
            if let Some(origins) = metadata_routes.get(conda_name) {
                rejected_pypi_origins.extend(
                    origins
                        .iter()
                        .map(|origin| PypiKey::from_pypi(&origin.pypi_name)),
                );
            }
        }
        if rejected_pypi_origins.is_empty() {
            return Err(diagnostic_context.routing_error(format!(
                "cannot move conda dependencies {} back to PyPI because their corresponding \
                 PyPI package names are unavailable",
                rejected_keys.iter().cloned().collect::<Vec<_>>().join(", ")
            )));
        }
        let mut keep_pypi = uv_reresolve.keep_pypi.clone();
        keep_pypi.extend(rejected_pypi_origins);
        return Ok(JointRouteOutcome::RetryKeepPypi { keep_pypi });
    }

    let rejected_specs: BTreeMap<String, String> = rejected
        .iter()
        .map(|route| (route.conda_name.key().into_string(), route.spec.clone()))
        .collect();
    let mut restore_requests: BTreeMap<String, RestoreRequestBuilder> = BTreeMap::new();
    let mut audit_origins: BTreeSet<(String, String)> = BTreeSet::new();
    for route in &bundle.auto_routed {
        let conda_name = canonical_conda_name(&route.route.conda_name);
        if !rejected_keys.contains(&conda_name) {
            continue;
        }
        let key = canonical_conda_name(&route.route.pypi_name);
        let request = restore_requests
            .entry(key)
            .or_insert_with(|| RestoreRequestBuilder::new(&route.route.pypi_name, config.relax));
        request.add_safety_conda_name(&route.route.conda_name);
        request.add_abi_anchor_alias(
            super::is_semantic_abi_anchor(&route.route.pypi_name, &abi_aliases)
                || super::is_semantic_abi_anchor(&route.route.conda_name, &abi_aliases),
        );
        for input in &route.route.input_requirements {
            let specifiers = if input.specifiers.trim().is_empty() {
                VersionSpecifiers::empty()
            } else {
                VersionSpecifiers::from_str(&input.specifiers).with_context(|| {
                    format!(
                        "parsing authoritative PyPI input `{}` for rejected route `{}`",
                        input.specifiers, route.route.pypi_name
                    )
                })?
            };
            let provenance = input.effective_provenance();
            let role = match input.role {
                crate::uv_closure::AutoRouteInputRole::Requirement => "requirement",
                crate::uv_closure::AutoRouteInputRole::Constraint => "constraint",
                crate::uv_closure::AutoRouteInputRole::Override => "override",
            };
            let origin_id = ConstraintOriginId::from_parts(
                "auto-route-input",
                [
                    PypiKey::from_pypi(&route.route.pypi_name).into_string(),
                    route.route.conda_name.clone(),
                    route.route.pypi_version.clone(),
                    route.route.conda_version.clone(),
                    route.route.channel.clone(),
                    role.to_string(),
                    provenance_origin_label(&provenance),
                    render_specifiers(&specifiers),
                ],
            );
            let requirement = Constraint {
                specifiers,
                source: input.source.clone(),
                origin_id,
                provenance,
            };
            request.add_constraint(requirement);
        }
        if let Some(requirements) =
            observed_requirements.get(&PypiKey::from_pypi(&route.route.pypi_name))
        {
            for requirement in requirements {
                request.add_constraint(requirement.clone());
            }
        }
        RestoreRequestBuilder::add_preference(
            &mut request.route_preferences,
            route.route.pypi_version.clone(),
            format!(
                "uv route `{}=={}` to conda `{}=={}` on `{}`",
                route.route.pypi_name,
                route.route.pypi_version,
                route.route.conda_name,
                route.route.conda_version,
                route.route.channel
            ),
        );
        audit_origins.insert((
            route.route.pypi_name.clone(),
            route.route.conda_name.clone(),
        ));
    }
    for conda_name in &rejected_keys {
        if let Some(origins) = metadata_routes.get(conda_name) {
            for origin in origins {
                let key = canonical_conda_name(&origin.pypi_name);
                let request = restore_requests
                    .entry(key.clone())
                    .or_insert_with(|| RestoreRequestBuilder::new(&origin.pypi_name, config.relax));
                request.add_safety_conda_name(&origin.conda_name);
                request.add_abi_anchor_alias(
                    super::is_semantic_abi_anchor(&origin.pypi_name, &abi_aliases)
                        || super::is_semantic_abi_anchor(&origin.conda_name, &abi_aliases),
                );
                let requirements = observed_requirements
                    .get(&PypiKey::from_pypi(&key))
                    .ok_or_else(|| {
                        diagnostic_context.routing_error(format!(
                            "cannot move PyPI dependency `{}` from conda package `{}` back to PyPI \
                             because its source requirement is unavailable",
                            origin.pypi_name, origin.conda_name
                        ))
                    })?;
                for requirement in requirements {
                    request.add_constraint(requirement.clone());
                }
                for preferred_version in &origin.preferred_versions {
                    RestoreRequestBuilder::add_preference(
                        &mut request.lock_preferences,
                        preferred_version.clone(),
                        format!(
                            "favor-lock for metadata route `{} -> {}`",
                            origin.pypi_name, origin.conda_name
                        ),
                    );
                }
                audit_origins.insert((origin.pypi_name.clone(), origin.conda_name.clone()));
            }
        }
    }
    if restore_requests.is_empty() {
        return Err(diagnostic_context.routing_error(format!(
            "cannot move conda dependencies {} back to PyPI because their corresponding \
             PyPI package names are unavailable",
            rejected_keys.iter().cloned().collect::<Vec<_>>().join(", ")
        )));
    }
    // Finalize every requirement (dedupe + semantic satisfiability) before
    // the first index request, so one genuine conflict cannot be obscured by
    // a network error for another rejected route. Conflicts retain the P3
    // typed diagnostic and fail before any fetch.
    let mut finalized_restore_requests = Vec::new();
    let mut restore_conflicts = Vec::new();
    for request in restore_requests.into_values() {
        match request.finish_with_graph_context(Some(diagnostic_context), bundle, config, target) {
            Ok(request) => finalized_restore_requests.push(request),
            Err(error) => collect_conflicts(error, &mut restore_conflicts)?,
        }
    }
    restore_conflicts.extend(fixed_conflicts);
    if !restore_conflicts.is_empty() {
        return Err(aggregate_conflicts(restore_conflicts));
    }

    // Retain successful changes until the trial bundle commits. Fetch or
    // post-fetch validation can still fail, and an AUTO-RELAXED warning must
    // never describe a transaction that was rolled back.
    let pending_relaxations = finalized_restore_requests
        .iter()
        .flat_map(|request| request.relaxations.iter().cloned())
        .collect::<Vec<_>>();

    // Fetch every wheel before changing routing. A missing index candidate
    // therefore fails without leaving the bundle partially un-routed.
    //
    // fix f17: the fetches are issued with bounded concurrency. They used to
    // be a serial `for request { ... fetch_pypi(..).await? }`, so a bundle
    // with 100 rejected routes paid 100 index round-trips end to end --
    // inside pixi's single conda-solve permit and with no log output
    // (`isaac-pack-latest`: ~30 minutes of total silence). The wheel-store
    // lock discipline is unchanged: `fetch_pypi` is the same entry point the
    // already-concurrent round-fetch above uses.
    // N27-RETREAD-142: A RESTORED ROUTE IS RE-RESOLVED, NOT RE-INJECTED.
    //
    // The legacy shape of the block below fetched each rejected route's wheel
    // under the closure's ORIGINAL requirement specifiers and pushed it into
    // `trial.extras` unexamined. That is how `googleapis-common-protos
    // 1.75.3` -- whose `Requires-Dist` is `protobuf<8.0.0,>=6.33.5` -- entered
    // `isaaclab-2.3x-pack` in the landing relock 6106911, whose own uv lock had
    // just resolved `protobuf` at `5.29.3`: the two are mutually exclusive, and
    // the wheel got in the other way round, through a restore that never
    // re-asked the question. The emission then read that wheel's
    // `Requires-Dist` and carried the bound into the pack's conda `constrains`
    // as a cap NO consuming environment could satisfy.
    //
    // So the restore now carries the workspace conda facts as constraints of
    // its own. A candidate whose requirement EXCLUDES a fact version is
    // refused, its version is added to the request's exclusions, and the fetch
    // is re-issued -- which is uv's own backtrack, expressed in the one
    // vocabulary this call site has (`bfs_fetch_pypi_from_chain` selects the
    // highest release satisfying the specifiers it is given).
    // `googleapis-common-protos 1.75.0` requires `protobuf<8.0.0,>=4.25.8` and
    // is one release below the one that was taken, so the backtrack is real
    // and short. When NO release is compatible the bundle REFUSES, naming both
    // the requirement and the fact, rather than emitting an unsatisfiable cap:
    // a pack that ships code needing `protobuf>=6.33.5` into environments
    // holding `5.29.3` is broken whether or not the cap that says so is
    // printed, and every cap-omission form to date hid that rather than
    // resolving it.
    //
    // THE CROSSING TEST IS SPECIFIER-AWARE, WHICH `route_restore_crossings` IS
    // NOT. That function matches NAMES, by design, because it feeds a reader's
    // row; using it as the actuator's predicate would backtrack on the benign
    // `crosses=packaging,numpy,...` rows too. The predicate here is
    // `fact_versions_excluded_by_requires_dist`, which parses the requirement
    // and asks whether its specifier admits the fact's version.
    //
    // STATED LIMIT, SO IT IS NOT MISTAKEN FOR COVERAGE. The `already_bundled`
    // arm below is UNCHANGED. A wheel that is already in the bundle was not
    // put there by this restore, so re-resolving it here would be re-deciding
    // another path's selection from inside this one; -142's proof is that the
    // production wheel arrived through the FETCH.
    let marker_env = marker_env_for(&target.conda_subdir, &target.python_version)?;
    let fact_versions = admission_fact_holdings(bundle);
    let reresolve = uv_reresolve.mode.is_enabled();
    let bundle_label = bundle.conda_name.to_string();

    let mut to_fetch = Vec::with_capacity(finalized_restore_requests.len());
    for request in finalized_restore_requests {
        let request_key = PypiKey::from_pypi(&request.pypi_name);
        let already_bundled = bundle
            .all_wheels()
            .find(|wheel| {
                PypiKey::from_pypi(&wheel.metadata.name) == request_key
                    || PypiKey::from_pypi(&wheel.pypi_name) == request_key
            })
            .map(|wheel| {
                (
                    wheel.pypi_name.clone(),
                    wheel.metadata.name.clone(),
                    wheel.metadata.version.clone(),
                )
            });
        if let Some((bundle_name, metadata_name, version_text)) = already_bundled {
            let version = Version::from_str(&version_text).with_context(|| {
                format!(
                    "parsing bundled wheel version `{version_text}` while restoring `{}`",
                    request.pypi_name
                )
            })?;
            if !request.specifiers.contains(&version) {
                return Err(diagnostic_context.routing_error(format!(
                    "PyPI dependency `{}` requires `{}`, but the bundle already contains \
                     `{bundle_name}` (metadata `{metadata_name}`) at incompatible version \
                     `{version_text}`",
                    request.pypi_name, request.specifiers
                )));
            }
            tracing::info!(
                pypi = %request.pypi_name,
                version = %version_text,
                "joint route validation reused compatible wheel already present in bundle",
            );
            continue;
        }
        let requirement = request.specifiers.to_string();
        let failure_context = diagnostic_context
            .routing_error(format!(
                "no configured PyPI index can provide `{}{requirement}`",
                request.pypi_name
            ))
            .to_string();
        to_fetch.push((request, failure_context));
    }
    let restored: Vec<(ResolvedWheel, Option<String>)> = fetch_bounded_concurrent(
        to_fetch,
        RESTORE_FETCH_CONCURRENCY,
        |(request, failure_context): (PypiFetchRequest, String)| {
            let fact_versions = &fact_versions;
            let marker_env = &marker_env;
            let bundle_label = bundle_label.as_str();
            async move {
                // ONE ACTUATOR, TWO DOORS (N27-RETREAD-145). The backtracking
                // loop that used to live inline here is now
                // `fetch_under_workspace_facts`, called identically by the
                // AUTO-BUNDLE admission. Nothing about this path's decision
                // moved -- the extraction is a whole-body move -- but the
                // other door is no longer unguarded.
                let dep = request.pypi_name.clone();
                let (wheel, moved) = fetch_under_workspace_facts(
                    request,
                    indexes.to_vec(),
                    failure_context,
                    fact_versions,
                    marker_env,
                    bundle_label,
                    FactConstrainedSite::JointRouteRestore,
                    reresolve,
                    fetch_pypi,
                )
                .await?;
                let row = moved.map(|check| {
                    pypi_route_reresolved_row(
                        &dep,
                        &check.from,
                        &check.to,
                        fact_versions.len(),
                    )
                });
                Ok((wheel, row))
            }
        },
    )
    .await?;
    let mut restored_wheels = Vec::with_capacity(restored.len());
    let mut reresolved_rows = Vec::with_capacity(restored.len());
    for (wheel, row) in restored {
        restored_wheels.push(wheel);
        if let Some(row) = row {
            reresolved_rows.push(row);
        }
    }

    let mut trial = bundle.clone();
    trial
        .auto_routed
        .retain(|route| !rejected_keys.contains(&canonical_conda_name(&route.route.conda_name)));
    trial.extras.extend(restored_wheels);
    // STDERR, NEVER STDOUT, for the same reason the RESTORED row below is on
    // stderr: `rpc.rs` owns stdout as the JSON-RPC channel.
    for row in &reresolved_rows {
        eprintln!("{row}");
    }
    let fact_names: BTreeSet<String> = trial.workspace_conda_provider_facts.keys().cloned().collect();
    for (pypi_name, conda_name) in audit_origins {
        // THE DECISION GETS A READER (TRIGGER-1, law 2). The WARN below is
        // deleted with the arm's backend log and the probe trace lands under a
        // fast root the job removes, so this restore -- the step that turns an
        // upstream dependency add into a bundled wheel whose Requires-Dist can
        // cross a learned conda fact -- was, until now, unrecoverable after the
        // fact. STDERR, NEVER STDOUT: `rpc.rs` owns stdout as the JSON-RPC
        // channel and a `###` row on it corrupts the very lock it reports on.
        let crosses = trial
            .all_wheels()
            .find(|wheel| {
                PypiKey::from_pypi(&wheel.pypi_name) == PypiKey::from_pypi(&pypi_name)
                    || PypiKey::from_pypi(&wheel.metadata.name) == PypiKey::from_pypi(&pypi_name)
            })
            .map(|wheel| {
                route_restore_crossings(&wheel.metadata.requires_dist, &fact_names)
            })
            .unwrap_or_default();
        eprintln!(
            "{}",
            pypi_route_restored_row(&pypi_name, &conda_name, &trial.conda_name.to_string(), &crosses)
        );
        let conda_key = canonical_conda_name(&conda_name);
        let spec = rejected_specs.get(&conda_key).cloned().unwrap_or_default();
        trial.probe_decisions.push(crate::audit::ProbeDecision {
            stage: "auto_route_joint_solve".into(),
            pypi_name: pypi_name.clone(),
            conda_name: conda_name.clone(),
            spec,
            target_python: target.python_version.clone(),
            channels_consulted: channels_consulted.to_vec(),
            satisfiable: Some(false),
            matching_candidates: 0,
            routing_decision: "joint-co-solve-rejected-to-pypi".into(),
        });
        tracing::warn!(
            pypi = %pypi_name,
            conda = %conda_name,
            "individually valid conda route rejected by final joint solve; restoring PyPI wheel",
        );
    }

    let still_emitted: Vec<String> = super::emitted_bundle_route_specs(&trial, config, target)
        .map_err(|error| diagnostic_context.scope_error(error))?
        .into_iter()
        .map(|route| route.conda_name.key().into_string())
        .filter(|name| rejected_keys.contains(name))
        .collect();
    if !still_emitted.is_empty() {
        return Err(diagnostic_context.routing_error(format!(
            "could not move conda dependencies {} back to PyPI",
            still_emitted.join(", ")
        )));
    }

    *bundle = trial;
    metadata_routes.retain(|conda_name, _| !rejected_keys.contains(conda_name));
    Ok(JointRouteOutcome::Mutated {
        relaxations: pending_relaxations,
    })
}

/// Returns Some((name, exact_version)) if `raw` is a base dep (no
/// extras marker) with a single `== X.Y.Z` specifier. Returns None for
/// extras-gated deps, ranges, ~=, or URL deps.
fn pep508_exact_base_dep(
    raw: &str,
    marker_env: &MarkerEnvironment,
) -> Result<Option<(String, String)>> {
    let req: uv_pep508::Requirement =
        crate::pep508_lenient::parse_requirement_lenient(raw).map_err(|e| anyhow!("{e}"))?;
    if !req.marker.evaluate(marker_env, &[]) {
        return Ok(None);
    }
    let Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) = req.version_or_url.as_ref() else {
        return Ok(None);
    };
    let specs: Vec<_> = specs.iter().collect();
    if specs.len() != 1 || *specs[0].operator() != uv_pep508::uv_pep440::Operator::Equal {
        return Ok(None);
    }
    Ok(Some((req.name.to_string(), specs[0].version().to_string())))
}

/// Base (unmarked, non-URL) deps that are NOT a single exact pin: bare
/// names (`nvidia-srl-usd-to-urdf`) and ranges. Returns the name plus
/// the line's specifiers (empty for bare). v1.7.0: the v1.5.6 bare-dep
/// fix covered the conda_outputs cascade only; auto_bundle (which also
/// runs at build time and is what the conda recipe + emit-pypi
/// actually see) silently skipped these, so a PyPI-only bare
/// transitive like isaaclab-mimic's nvidia-srl-usd-to-urdf never made
/// it into the built pack.
fn pep508_loose_base_dep(
    raw: &str,
    marker_env: &MarkerEnvironment,
) -> Result<Option<(String, VersionSpecifiers)>> {
    let req: uv_pep508::Requirement =
        crate::pep508_lenient::parse_requirement_lenient(raw).map_err(|e| anyhow!("{e}"))?;
    if !req.marker.evaluate(marker_env, &[]) {
        return Ok(None);
    }
    match req.version_or_url.as_ref() {
        None => Ok(Some((req.name.to_string(), VersionSpecifiers::empty()))),
        Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) => {
            let v: Vec<_> = specs.iter().collect();
            let is_exact =
                v.len() == 1 && *v[0].operator() == uv_pep508::uv_pep440::Operator::Equal;
            if is_exact {
                // pep508_exact_base_dep's territory.
                Ok(None)
            } else {
                Ok(Some((req.name.to_string(), specs.clone())))
            }
        }
        Some(uv_pep508::VersionOrUrl::Url(_)) => Ok(None),
    }
}

/// (wheel URL, parsed METADATA, optional sdist provenance) for one PyPI-form
/// BFS item fetched in the level loop's phase 2. The 3rd element is `Some`
/// only when the item was built from
/// a PyPI sdist (no compatible wheel on the index); `None` for normal
/// index-wheel fetches.
pub(crate) type BfsFetched = (url::Url, WheelMetadata, Option<super::SdistProv>);

/// One unit of pending work in the resolver BFS.
#[derive(Debug, Clone)]
pub(crate) struct Pending {
    pub(crate) pypi_name: String,
    pub(crate) source: PendingSource,
    /// Extras to activate on this wheel. Drives further worklist additions
    /// for `Requires-Dist: name ; extra == "X"` lines.
    pub(crate) extras: Vec<String>,
}

/// v0.12.0+: a dep can be sourced from a PyPI Simple index (the
/// original behavior) or from a direct URL / git URL declared via PEP
/// 508 `<name> @ <url>` form. URL-form deps are common in
/// `[project.optional-dependencies]` and previously made retread bail.
#[derive(Debug, Clone)]
pub(crate) enum PendingSource {
    /// `Requires-Dist: <name> <specifiers>` -- resolve via PyPI Simple.
    Pypi {
        specifiers: VersionSpecifiers,
        indexes: Vec<String>,
    },
    /// `Requires-Dist: <name> @ git+<scheme>://<host>/<path>@<rev>[#subdirectory=<sub>]` --
    /// clone + `pip wheel --no-deps`.
    ///
    /// `subdirectory` is parsed from the URL fragment `#subdirectory=<sub>`
    /// (A-0 fix: previously the fragment was appended to `rev`, corrupting it
    /// to `"rev#subdirectory=..."` which made the checkout key wrong and the
    /// wheel build fail or produce a stale clone).
    Git {
        url: String,
        rev: Option<String>,
        /// Subdirectory within the repo to build the wheel from (default: root ".").
        subdirectory: Option<String>,
    },
    /// `Requires-Dist: <name> @ <scheme>://...` (direct wheel/sdist).
    Url { wheel_url: url::Url },
}

/// Add extras-gated and prefix-matched base deps from `metadata` to `work`.
/// Skips entries already in `seen` so the BFS terminates.
///
/// `sibling_names`: canonical conda names of OTHER entries in the same bundle
/// group.  A dep whose canonical name is in this set is a "sibling" — it is
/// provided at install time by the sibling's wheel and must NOT be resolved
/// from PyPI or conda.  Such deps are silently dropped without being enqueued.
///
/// `state`: when `Some`, locked deps that appear in `seen` are NOT silently
/// skipped — they are routed through `state.observe_edge` to detect ripples.
/// `AlreadySatisfied` → continue; `NeedsReResolve` or conflict `Err` →
/// returns [`IncrementalRipple`].  Non-locked deps in `seen` are silently
/// skipped as before.  When `state` is `None` (cold path), behavior is
/// identical to the previous unconditional `continue`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn seed_worklist(
    requires_dist: &[String],
    extras_requested: &[String],
    indexes: &[String],
    bundle_prefix: &str,
    seen: &HashSet<String>,
    work: &mut VecDeque<Pending>,
    state: Option<&mut ResolveState>,
    sibling_names: &HashSet<String>,
) -> Result<()> {
    // Helper: check whether a dep that's already in `seen` should trigger a
    // ripple check (incremental path, dep is locked).
    //
    // We take `state` by value here (moved in), then return it so the loop
    // can reuse it across iterations.  Using a closure would require a mutable
    // borrow of `state` that conflicts with the `work.push_back` borrow below,
    // so we do the logic inline via a separate inner function.
    //
    // Note: `state` is `Option<&mut ResolveState>` — reborrow it as needed.

    macro_rules! check_locked_seen {
        ($dn:expr, $pending:expr, $state:expr) => {{
            if let Some(ref mut st) = $state {
                if st.is_locked($dn) {
                    match st.observe_edge($dn, $pending) {
                        Ok(
                            super::resolve_state::ObserveEdgeResult::AlreadySatisfied
                            | super::resolve_state::ObserveEdgeResult::NonPypiAlreadySeen,
                        ) => {
                            continue;
                        }
                        Ok(super::resolve_state::ObserveEdgeResult::NeedsReResolve(_)) => {
                            return Err(anyhow::Error::new(IncrementalRipple {
                                reason: format!("locked dep `{}` would need re-resolution", $dn),
                            }));
                        }
                        Ok(super::resolve_state::ObserveEdgeResult::New(_)) => {
                            // Locked dep appeared as New — shouldn't happen if
                            // seed_locked was called; treat as AlreadySatisfied.
                            continue;
                        }
                        Err(e) => {
                            return Err(anyhow::Error::new(IncrementalRipple {
                                reason: format!("locked dep `{}` conflicts: {e}", $dn),
                            }));
                        }
                    }
                }
            }
            // Non-locked or cold path: fall through to the existing `continue`.
            continue;
        }};
    }

    let mut state = state;
    for raw in requires_dist {
        // 1. Extras-gated lines for each requested extra.
        let mut added = false;
        for extra in extras_requested {
            if let Some(dep) = pep508_extra_dep(raw, extra)? {
                let dn = canonical_conda_name(&dep.name);
                // Sibling check: a dep naming another entry in the same bundle
                // group is provided by that sibling's wheel at install time.
                // Do NOT resolve it from PyPI — drop it silently.
                if sibling_names.contains(&dn) {
                    tracing::debug!(
                        dep = %dep.name,
                        sibling_canon = %dn,
                        "seed_worklist: skipping sibling dep (extras-gated) — provided by sibling bundle entry",
                    );
                    added = true;
                    continue;
                }
                if seen.contains(&dn) {
                    let pending = Pending {
                        pypi_name: dep.name.clone(),
                        source: extra_dep_source_to_pending(dep.source.clone(), indexes),
                        extras: dep.extras.clone(),
                    };
                    check_locked_seen!(&dn, pending, state);
                }
                work.push_back(Pending {
                    pypi_name: dep.name,
                    source: extra_dep_source_to_pending(dep.source, indexes),
                    extras: dep.extras,
                });
                added = true;
            }
        }
        if added {
            continue;
        }
        // 2. Base deps (no marker) whose PyPI name matches the bundle prefix.
        if let Some(dep) = pep508_base_dep_in_prefix(raw, bundle_prefix)? {
            let dn = canonical_conda_name(&dep.name);
            // Sibling check: same as extras path above — a base dep that names
            // another bundle entry must not be fetched from PyPI.
            if sibling_names.contains(&dn) {
                tracing::debug!(
                    dep = %dep.name,
                    sibling_canon = %dn,
                    "seed_worklist: skipping sibling dep (base) — provided by sibling bundle entry",
                );
                continue;
            }
            if seen.contains(&dn) {
                let pending = Pending {
                    pypi_name: dep.name.clone(),
                    source: extra_dep_source_to_pending(dep.source.clone(), indexes),
                    extras: dep.extras.clone(),
                };
                check_locked_seen!(&dn, pending, state);
            }
            work.push_back(Pending {
                pypi_name: dep.name,
                source: extra_dep_source_to_pending(dep.source, indexes),
                extras: dep.extras,
            });
        }
    }
    Ok(())
}

fn extra_dep_source_to_pending(src: ExtraDepSource, indexes: &[String]) -> PendingSource {
    match src {
        ExtraDepSource::Pypi(specifiers) => PendingSource::Pypi {
            specifiers,
            indexes: indexes.to_vec(),
        },
        ExtraDepSource::Git {
            url,
            rev,
            subdirectory,
        } => PendingSource::Git {
            url,
            rev,
            subdirectory,
        },
        ExtraDepSource::Url(wheel_url) => PendingSource::Url { wheel_url },
    }
}

/// Returns Some(ExtraDep) if `raw` is a base dep (no extras marker, or a
/// marker that's satisfied with empty extras) whose PEP 503 normalized name
/// starts with `prefix`. Used to bundle sibling sub-packages like
/// `isaacsim-kernel` that the metapackage depends on unconditionally.
fn pep508_base_dep_in_prefix(raw: &str, prefix: &str) -> Result<Option<ExtraDep>> {
    let req: uv_pep508::Requirement =
        crate::pep508_lenient::parse_requirement_lenient(raw).map_err(|e| anyhow!("{e}"))?;

    // Base dep: marker (if any) satisfied with empty extras.
    let env = default_marker_env(DEFAULT_PYTHON)?;
    if !req.marker.evaluate(&env, &[]) {
        return Ok(None);
    }

    let conda_name = canonical_conda_name(req.name.as_ref());
    if !conda_name.starts_with(prefix) {
        return Ok(None);
    }

    // Same any-version handling as pep508_extra_dep: a bare-name base
    // dep is legal PEP 508 and resolves to latest at the PyPI index.
    let source = match req.version_or_url.as_ref() {
        Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) => {
            ExtraDepSource::Pypi(specs.clone())
        }
        Some(uv_pep508::VersionOrUrl::Url(verbatim)) => extra_dep_source_from_url(verbatim.raw())?,
        None => ExtraDepSource::Pypi(uv_pep508::uv_pep440::VersionSpecifiers::empty()),
    };
    Ok(Some(ExtraDep {
        name: req.name.to_string(),
        source,
        extras: req.extras.iter().map(|e| e.to_string()).collect(),
    }))
}

/// Convert a PEP 508 URL Requires-Dist into one of our
/// [`ExtraDepSource`] variants. Splits `git+<scheme>://...@<rev>[#subdirectory=<sub>]`
/// into `(base_url, Some(rev), subdirectory)`; plain `https://.../file.whl`
/// becomes a direct-URL fetch.
///
/// # A-0 fix: `#subdirectory=` stripping
///
/// PEP 508 / pip allow a URL fragment of the form `#subdirectory=<path>` to
/// indicate which subdirectory of the repo contains the Python package. Without
/// this fix, `rfind('@')` finds the `@` before `<rev>` but includes the
/// `#subdirectory=<path>` suffix as part of `rev` (e.g. `rev` becomes
/// `"ce11136#subdirectory=src/newton"`), corrupting the checkout cache key
/// (which sha256-hashes url+rev) and the git checkout itself.
pub(crate) fn extra_dep_source_from_url(raw_url: &url::Url) -> Result<ExtraDepSource> {
    let s = raw_url.as_str();
    if let Some(stripped) = s.strip_prefix("git+") {
        // PEP 508 doesn't say where the @<rev> lives but pip-compatible
        // syntax is `git+<scheme>://<host>/<path>@<rev>[#subdirectory=<sub>]`.
        // Find the rightmost `@` that comes after `://` (skipping any in user-
        // info, though those are rare for public git).
        let scheme_end = stripped.find("://").map(|i| i + 3).unwrap_or(0);
        let (base, rev_with_fragment) = match stripped[scheme_end..].rfind('@') {
            Some(rel) => {
                let abs = scheme_end + rel;
                (
                    stripped[..abs].to_string(),
                    Some(stripped[abs + 1..].to_string()),
                )
            }
            None => (stripped.to_string(), None),
        };

        // A-0 fix: split the fragment `#subdirectory=<sub>` out of the rev
        // string. Without this, any `git+https://host/repo@<rev>#subdirectory=<sub>`
        // URL corrupts `rev` to `"<rev>#subdirectory=<sub>"`, which:
        //   (a) keys the git checkout cache on a hash that includes the junk suffix,
        //   (b) passes the junk rev to `git checkout`, which fails or produces a
        //       stale clone at a wrong cache path,
        //   (c) stores the junk rev in the lock's GitWheelSource.rev (once that
        //       field exists), making replay impossible.
        let (rev, subdirectory) = match rev_with_fragment {
            None => (None, None),
            Some(rv) => {
                if let Some(frag_pos) = rv.find('#') {
                    let clean_rev = rv[..frag_pos].to_string();
                    let fragment = &rv[frag_pos + 1..];
                    // Parse `subdirectory=<path>` from the fragment.
                    let subdir = fragment.strip_prefix("subdirectory=").map(str::to_string);
                    (Some(clean_rev), subdir)
                } else {
                    (Some(rv), None)
                }
            }
        };

        Ok(ExtraDepSource::Git {
            url: base,
            rev,
            subdirectory,
        })
    } else {
        Ok(ExtraDepSource::Url(raw_url.clone()))
    }
}

pub(crate) async fn fetch_and_parse(
    url: &url::Url,
    sha256_hint: Option<&str>,
    download_dir: &Path,
) -> Result<WheelMetadata> {
    let path = crate::wheel::fetch_wheel(url, sha256_hint, download_dir).await?;
    tokio::task::spawn_blocking(move || crate::wheel_content::read_metadata_recorded(&path))
        .await
        .context("metadata reader panicked")?
}

async fn fetch_and_parse_cached(
    url: &url::Url,
    sha256_hint: Option<&str>,
    download_dir: &Path,
    store_root: &Path,
) -> Result<WheelMetadata> {
    let path = crate::wheel::fetch_wheel_cached(url, sha256_hint, download_dir, store_root).await?;
    tokio::task::spawn_blocking(move || crate::wheel_content::read_metadata_recorded(&path))
        .await
        .context("metadata reader panicked")?
}

/// v1.4.3: metadata-only acquisition for wheels whose BYTES aren't
/// needed in this phase. BFS extras, auto-bundled, and cascade-bundled
/// wheels enter the recipe by their UPSTREAM url -- rattler-build
/// fetches the bytes at build time, so the local download here only
/// ever served the METADATA read. Preference order:
///   1. wheel already in the disk cache -> read it (no network, and
///      warm re-runs stay as fast as before);
///   2. wheel already in the attested machine-global store -> read it in
///      place (no copy, re-hash, range request, or download);
///   3. index advertised a PEP 658/714 sidecar AND a sha256 fragment
///      -> fetch the KB-sized `.metadata` sidecar instead of the
///      potentially-GB wheel (the fragment hash stands in for the
///      computed one the recipe pins);
///   4. seek the zip metadata through HTTP ranges when supported;
///   5. full download through the shared content-addressed wheel store.
///
/// pypi.org serves sidecars; pypi.nvidia.com and static GitHub-Pages
/// indexes do not (measured 2026-06-10), so NVIDIA-index-only wheels
/// still take path 3 on a cold cache.
pub(crate) async fn metadata_preferring_sidecar(
    resolved: &pypi::ResolvedWheel,
    download_dir: &Path,
) -> Result<WheelMetadata> {
    metadata_preferring_sidecar_with_store(
        resolved,
        download_dir,
        &crate::courier::retread_wheel_store_root(),
    )
    .await
}

async fn metadata_preferring_sidecar_with_store(
    resolved: &pypi::ResolvedWheel,
    download_dir: &Path,
    store_root: &Path,
) -> Result<WheelMetadata> {
    if let Ok(filename) = crate::wheel::wheel_filename_from_url(&resolved.url)
        && download_dir.join(&filename).exists()
    {
        return fetch_and_parse(&resolved.url, resolved.sha256.as_deref(), download_dir).await;
    }
    // Warm retries must consult the machine-global content-addressed store
    // before touching the network. NVIDIA serves no metadata sidecars and its
    // giant Isaac wheels are marked no-store; without this lookup, a failed
    // solve repeatedly downloaded payload bytes that Retread already had.
    if let Some(sha256) = resolved.sha256.as_deref() {
        match crate::wheel::cached_wheel_store_path(&resolved.url, sha256, store_root).await {
            Ok(Some(path)) => {
                tracing::info!(
                    wheel = %resolved.filename,
                    "wheel metadata cache: hit (persistent store, no download)",
                );
                // C10-b: this is the persistent content-addressed store, whose
                // directory name IS the digest -- 991 of the C10 proof relock's
                // 1121 full `hash+parse` reads (258.6 s, 29.57 GB) came in here.
                return tokio::task::spawn_blocking(move || {
                    crate::wheel_content::read_metadata_recorded(&path)
                })
                    .await
                    .context("metadata reader panicked")?;
            }
            Ok(None) => {}
            Err(error) => tracing::debug!(
                wheel = %resolved.filename,
                error = %format!("{error:#}"),
                "persistent wheel metadata lookup failed; continuing through network fallbacks",
            ),
        }
    }
    if resolved.has_metadata_sidecar
        && let Some(sha) = resolved.sha256.as_deref()
    {
        match crate::wheel::fetch_metadata_sidecar(&resolved.url, sha).await {
            Ok(m) => return Ok(m),
            Err(e) => {
                tracing::debug!(
                    url = %resolved.url,
                    error = %format!("{e:#}"),
                    "metadata sidecar fetch failed; falling back to ranged/full fetch",
                );
            }
        }
    }
    // No sidecar (pypi.nvidia.com, static GitHub-Pages indexes): try a
    // ranged read of the zip central directory + METADATA member before
    // resorting to downloading the whole (potentially multi-GiB) wheel just
    // to read a few KiB of metadata. Index-agnostic: any server honoring
    // HTTP Range works; anything else errors and we fall through to the full
    // download. Requires the index-advertised sha256 (the recipe pins it and
    // the ranged read never computes the full-wheel hash).
    if let Some(sha) = resolved.sha256.as_deref() {
        match crate::wheel::fetch_metadata_ranged(&resolved.url, sha).await {
            Ok(m) => return Ok(m),
            Err(e) => {
                tracing::debug!(
                    url = %resolved.url,
                    error = %format!("{e:#}"),
                    "ranged metadata fetch failed; falling back to full wheel download",
                );
            }
        }
    }
    fetch_and_parse_cached(
        &resolved.url,
        resolved.sha256.as_deref(),
        download_dir,
        store_root,
    )
    .await
}

/// One extras-derived dependency. v0.12.0+: source can be PyPI Simple
/// OR a direct URL / git URL (`pkg @ git+https://...@<rev>` or `pkg @
/// https://.../file.whl`). PyPI is the common case; URL+git unlock
/// extras like IsaacLab's `rl_games` which pulls `rl-games @ git+...`.
#[derive(Debug, Clone)]
pub(crate) struct ExtraDep {
    pub(crate) name: String,
    pub(crate) source: ExtraDepSource,
    pub(crate) extras: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) enum ExtraDepSource {
    Pypi(VersionSpecifiers),
    /// A `git+<url>@<rev>[#subdirectory=<sub>]` dependency.
    ///
    /// `subdirectory` carries the `#subdirectory=<sub>` fragment so it is
    /// NOT corrupted into `rev` (A-0 fix).
    Git {
        url: String,
        rev: Option<String>,
        subdirectory: Option<String>,
    },
    Url(url::Url),
}

/// Returns `Some(ExtraDep)` if `raw` is a `Requires-Dist` line that is
/// gated on the requested extra. Returns None if the requirement is gated
/// on a different extra (or has no marker, i.e. is a base dep we don't
/// repack at all). Any specifier set is accepted; range resolution
/// happens at the index-fetch layer in pypi::resolve.
pub(crate) fn pep508_extra_dep(raw: &str, extra: &str) -> Result<Option<ExtraDep>> {
    let req: uv_pep508::Requirement = crate::pep508_lenient::parse_requirement_lenient(raw)
        .map_err(|e| anyhow!("parsing extra requirement: {e}"))?;

    let extra_name = uv_normalize::ExtraName::from_owned(extra.to_string())
        .map_err(|e| anyhow!("invalid extra name `{extra}`: {e}"))?;

    // The marker must match when this extra is active AND must not match
    // with no extras active (otherwise it's a base dep, not an extra dep).
    let env = default_marker_env(DEFAULT_PYTHON)?;
    let matches_with_extra = req.marker.evaluate(&env, std::slice::from_ref(&extra_name));
    let matches_without = req.marker.evaluate(&env, &[]);
    if !matches_with_extra || matches_without {
        return Ok(None);
    }

    // Bare name with no specifier and no URL is legal PEP 508
    // (`Requires-Dist: tqdm; extra == "sb3"`) -- means "any version".
    // Treat as PyPI with an empty specifier set; pypi::resolve returns
    // the latest matching the target python. Without this, every
    // extras-gated bare name in upstream wheels (rich, tqdm, gym, ...)
    // made retread bail with "no version or URL".
    let source = match req.version_or_url.as_ref() {
        Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) => {
            ExtraDepSource::Pypi(specs.clone())
        }
        Some(uv_pep508::VersionOrUrl::Url(verbatim)) => extra_dep_source_from_url(verbatim.raw())?,
        None => ExtraDepSource::Pypi(uv_pep508::uv_pep440::VersionSpecifiers::empty()),
    };
    let _ = extra; // extras name is only used for marker evaluation above
    Ok(Some(ExtraDep {
        name: req.name.to_string(),
        source,
        extras: req.extras.iter().map(|e| e.to_string()).collect(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rattler_conda_types::{PackageRecord, RepoDataRecord, VersionWithSource};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use url::Url;

    #[test]
    fn serial_probe_mode_skips_eager_prewarm() {
        assert!(!should_eagerly_prewarm_probe_routes(0, true));
        assert!(!should_eagerly_prewarm_probe_routes(1, true));
        assert!(!should_eagerly_prewarm_probe_routes(4, false));
        assert!(should_eagerly_prewarm_probe_routes(2, true));
        assert!(should_eagerly_prewarm_probe_routes(4, true));
    }

    #[test]
    fn abi_anchor_cap_completion_has_loud_warning_and_durable_record() {
        let relaxation = abi_anchor_cap_completion(
            "flashsac-pack",
            &PypiKey::from_pypi("numpy"),
            ">=2",
            ">=2.0,<3",
            vec!["wheel `consumer==1` Requires-Dist `numpy>=2`".to_string()],
        );

        let warning = relaxation.to_string();
        assert!(warning.contains("RETREAD AUTO-COMPLETED ABI anchor"));
        assert!(warning.contains("bundle `flashsac-pack`"));
        assert!(warning.contains("package `numpy`"));
        assert!(warning.contains("`>=2` -> `>=2.0,<3`"));
        assert!(warning.contains("wheel `consumer==1` Requires-Dist `numpy>=2`"));

        let scope = RelaxationScope {
            environments: vec!["flashsac-gpu".to_string()],
            targets: vec!["linux-64".to_string()],
            platform: "linux-64".to_string(),
            python: "3.11".to_string(),
        };
        let record = relaxation.to_record(&scope);
        assert_eq!(
            record.kind,
            crate::relaxation_record::RelaxationRecordKind::AbiAnchorCapCompleted
        );
        assert_eq!(record.tier, RelaxPolicy::Minor);
        assert_eq!(record.original_spec, ">=2");
        assert_eq!(record.resulting_spec, ">=2.0,<3");
        assert_eq!(record.involved_wheels, relaxation.involved_sources);
        assert_eq!(record.scope, scope);
    }

    fn test_origin(label: &str, specifiers: &str) -> ConstraintOriginId {
        ConstraintOriginId::from_parts(
            "auto-bundle-test",
            [label.to_string(), specifiers.to_string()],
        )
    }

    fn metadata_test_wheel() -> Vec<u8> {
        use std::io::Write as _;
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut archive = zip::ZipWriter::new(&mut cursor);
            let options: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default();
            archive
                .start_file("cached_pkg-1.2.3.dist-info/METADATA", options)
                .unwrap();
            archive
                .write_all(
                    b"Metadata-Version: 2.1\nName: cached-pkg\nVersion: 1.2.3\nRequires-Dist: child>=4\n\n",
                )
                .unwrap();
            archive.finish().unwrap();
        }
        cursor.into_inner()
    }

    #[tokio::test]
    async fn metadata_reader_reuses_attested_persistent_wheel_without_network() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-metadata-store-test-{}",
            std::process::id()
        ));
        let store = tmp.join("store");
        let downloads = tmp.join("downloads");
        std::fs::create_dir_all(&downloads).unwrap();
        let filename = "cached_pkg-1.2.3-py3-none-any.whl";
        let source = tmp.join(filename);
        std::fs::write(&source, metadata_test_wheel()).unwrap();
        let sha256 = crate::wheel::store_wheel_in_cache(&source, &store)
            .await
            .unwrap();
        let resolved = pypi::ResolvedWheel {
            // The test must fail if this path touches the network.
            url: Url::parse(&format!("http://127.0.0.1:1/{filename}")).unwrap(),
            sha256: Some(sha256),
            filename: filename.to_string(),
            has_metadata_sidecar: false,
        };

        let metadata = metadata_preferring_sidecar_with_store(&resolved, &downloads, &store)
            .await
            .unwrap();

        assert_eq!(metadata.name, "cached-pkg");
        assert_eq!(metadata.version, "1.2.3");
        assert_eq!(metadata.requires_dist, ["child>=4"]);
        assert!(std::fs::read_dir(&downloads).unwrap().next().is_none());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn full_metadata_fallback_coalesces_into_shared_store() {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

        // fetch_wheel_cached consults RETREAD_NO_SHADOW_CACHE. Other async
        // tests deliberately toggle that process-global variable, so hold the
        // shared guard while asserting the cache-enabled path.
        let _env_guard = crate::TEST_ASYNC_ENV_MUTEX.lock().await;
        let bytes = metadata_test_wheel();
        let sha256 = crate::wheel_rewrite::sha256_hex(&bytes);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let gets = Arc::new(AtomicUsize::new(0));
        let server_gets = gets.clone();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let bytes = bytes.clone();
                let gets = server_gets.clone();
                tokio::spawn(async move {
                    let mut request = Vec::new();
                    let mut chunk = [0_u8; 1024];
                    loop {
                        if request.windows(4).any(|window| window == b"\r\n\r\n") {
                            break;
                        }
                        match stream.read(&mut chunk).await {
                            Ok(0) | Err(_) => return,
                            Ok(count) => request.extend_from_slice(&chunk[..count]),
                        }
                    }
                    if request.starts_with(b"GET ") {
                        gets.fetch_add(1, Ordering::SeqCst);
                    }
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        bytes.len(),
                    );
                    if stream.write_all(header.as_bytes()).await.is_err() {
                        return;
                    }
                    let _ = stream.write_all(&bytes).await;
                });
            }
        });

        let tmp = std::env::temp_dir().join(format!(
            "retread-metadata-coalesced-fallback-{}-{}",
            std::process::id(),
            line!(),
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        let first_downloads = tmp.join("first");
        let second_downloads = tmp.join("second");
        let store = tmp.join("store");
        let url = Url::parse(&format!(
            "http://127.0.0.1:{port}/cached_pkg-1.2.3-py3-none-any.whl"
        ))
        .unwrap();

        let (first, second) = tokio::join!(
            fetch_and_parse_cached(&url, Some(&sha256), &first_downloads, &store),
            fetch_and_parse_cached(&url, Some(&sha256), &second_downloads, &store),
        );
        assert_eq!(first.unwrap().name, "cached-pkg");
        assert_eq!(second.unwrap().name, "cached-pkg");
        assert_eq!(
            gets.load(Ordering::SeqCst),
            1,
            "concurrent metadata fallbacks must share one full-wheel GET",
        );
        assert!(
            crate::wheel::cached_wheel_store_path(&url, &sha256, &store)
                .await
                .unwrap()
                .is_some(),
            "metadata fallback must durably populate the shared wheel store",
        );
        server.abort();
        let _ = std::fs::remove_dir_all(tmp);
    }

    fn test_config() -> RetreadConfig {
        serde_json::from_value(serde_json::json!({
            "retread-wheels": {},
            "retread-relax": "none",
            "retread-conda-deps": ["libtorch"]
        }))
        .unwrap()
    }

    fn tiered_mapped_config(package: &str) -> RetreadConfig {
        let mut config = test_config();
        config.relax = RelaxPolicy::PatchThenMinorThenMajorThenLastResort;
        config.name_map.insert(
            PypiKey::from_pypi(package),
            CondaTarget::Mapped(CondaName::new(package)),
        );
        config
    }

    fn test_wheel(
        bundle_name: &str,
        metadata_name: &str,
        version: &str,
        requires_dist: &[&str],
    ) -> ResolvedWheel {
        let filename = format!(
            "{}-{version}-py3-none-any.whl",
            metadata_name.replace('-', "_")
        );
        let url = Url::parse(&format!("https://example.invalid/{filename}")).unwrap();
        ResolvedWheel {
            pypi_name: bundle_name.to_string(),
            url: url.clone(),
            upstream_url: Some(url),
            git_source: None,
            sdist_source: None,
            metadata_provenance: Provenance::IndexWheelMetadata,
            original_requires_dist: requires_dist
                .iter()
                .map(|requirement| (*requirement).to_string())
                .collect(),
            metadata: WheelMetadata {
                name: metadata_name.to_string(),
                version: version.to_string(),
                requires_dist: requires_dist
                    .iter()
                    .map(|requirement| (*requirement).to_string())
                    .collect(),
                retread_conda_run_dependencies: vec![],
                is_pure_python: true,
                sha256: format!("sha256-{metadata_name}-{version}"),
                filename,
            },
            extras_requested: vec![],
            auto_data: None,
            auto_data_dedup_skipped_root: None,
        }
    }

    fn test_bundle(requires_dist: &[&str]) -> Bundle {
        Bundle {
            conda_name: "regression-pack".to_string(),
            primary: test_wheel("regression-root", "regression-root", "1.0.0", requires_dist),
            extras: vec![],
            probe_decisions: vec![],
            solve_diagnostics: BTreeMap::new(),
            auto_routed: vec![],
            auto_dropped: HashSet::new(),
            uv_closure_names: HashSet::new(),
            auto_imports_injected: HashSet::new(),
            workspace_declared_pypi_specs: BTreeMap::new(),
            uv_dependency_graph: Default::default(),
            workspace_conda_versions: BTreeMap::new(),
            workspace_conda_provider_facts: BTreeMap::new(),
            constrains_basis: Default::default(),
            workspace_selected_conda_packages: BTreeMap::new(),
            workspace_declared_pypi: BTreeSet::new(),
            workspace_locked_pypi: BTreeMap::new(),
            workspace_locked_conda: BTreeMap::new(),
        }
    }

    #[test]
    fn namemap_noncanonical_key_matches() {
        let mut configured: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": {},
            "retread-relax": "none",
            "retread-name-map": {
                "opencv_python_headless": "py-opencv"
            }
        }))
        .unwrap();
        let key = PypiKey::from_pypi("opencv-python-headless");
        assert_eq!(
            configured
                .name_map
                .get(&key)
                .and_then(CondaTarget::mapped_name)
                .map(CondaName::as_spec),
            Some("py-opencv")
        );

        let mut global = PypiToCondaMap::new();
        global.insert(
            key.as_str().to_string(),
            vec!["opencv-python-headless-fallback".to_string()],
        );
        let effective = super::super::effective_name_map(&configured.name_map, &global);
        assert_eq!(
            effective
                .get(&key)
                .and_then(CondaTarget::mapped_name)
                .map(CondaName::as_spec),
            Some("py-opencv"),
            "the canonicalized user entry must beat the global fallback"
        );
        assert_eq!(
            pick_conda_target(&key, &effective, &global)
                .expect("the configured route should be selected")
                .as_spec(),
            "py-opencv",
            "the BFS must preserve the mapped conda spelling"
        );

        configured.name_map = effective;
        let bundle = test_bundle(&["opencv-python-headless==4.11.0"]);
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let emitted =
            super::super::emitted_bundle_route_specs(&bundle, &configured, &target).unwrap();
        assert!(
            emitted
                .iter()
                .any(|route| route.conda_name.as_spec() == "py-opencv"),
            "actual dependency emission must preserve the raw configured target: {emitted:?}"
        );
    }

    #[test]
    fn disabled_namemap_vetoes_global_route() {
        let configured: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": {},
            "retread-name-map": {
                "torch": ""
            }
        }))
        .unwrap();
        let key = PypiKey::from_pypi("torch");
        let mut global = PypiToCondaMap::new();
        global.insert(key.as_str().to_string(), vec!["pytorch".to_string()]);

        let effective = super::super::effective_name_map(&configured.name_map, &global);
        assert!(matches!(effective.get(&key), Some(CondaTarget::Disabled)));
        assert!(!prefer_conda_match(&key, &effective));
        assert_eq!(pick_conda_target(&key, &effective, &global), None);
    }

    /// GUARD (d), N27-RETREAD-142. THE ENVIRONMENT CANNOT REACH THIS DECISION
    /// ANY MORE.
    ///
    /// This test replaces `uv_reresolve_mode_requires_exact_literal_one`,
    /// which asserted the opposite polarity: that only the literal `"1"`
    /// turned the re-resolve ON and everything else "preserved the default
    /// legacy path". That default is the defect. The mode is now decided by
    /// the manifest key alone, so the guard sets `RETREAD_UV_RERESOLVE` to
    /// every value the old reader distinguished -- including the `"1"` that
    /// used to be the magic one, and the `"0"` that used to force the legacy
    /// path -- and asserts the mode is unmoved in BOTH directions. A guard
    /// that only checked `"0"` would pass on a tree that still read the
    /// variable and happened to default ON.
    ///
    /// The env var is set and removed inside the test rather than assumed
    /// absent, because "the reader is gone" is exactly what an ambient-absent
    /// environment cannot demonstrate.
    #[test]
    fn capwins5_the_env_var_cannot_move_the_reresolve_mode() {
        // SAFETY: single-threaded within this test, and every value is
        // removed again before it returns.
        for value in ["1", "0", "", "true", "false"] {
            unsafe { std::env::set_var("RETREAD_UV_RERESOLVE", value) };
            assert_eq!(
                UvReresolveMode::from_manifest_flag(None),
                UvReresolveMode::Enabled,
                "an absent manifest key must re-resolve whatever `{value}` says",
            );
            assert_eq!(
                UvReresolveMode::from_manifest_flag(Some(true)),
                UvReresolveMode::Enabled,
                "an explicit `true` must re-resolve whatever `{value}` says",
            );
            assert_eq!(
                UvReresolveMode::from_manifest_flag(Some(false)),
                UvReresolveMode::Disabled,
                "the deliberate opt-out must hold whatever `{value}` says",
            );
        }
        unsafe { std::env::remove_var("RETREAD_UV_RERESOLVE") };
        assert_eq!(
            UvReresolveMode::default(),
            UvReresolveMode::Enabled,
            "the derived default is the re-resolving mode",
        );
        // The opt-out is the ONE remaining reader of `Disabled` and it prints.
        let row = route_restore_reresolve_opt_out_row();
        assert!(
            row.starts_with("### PYPI ROUTE RERESOLVE DISABLED key=") &&
                row.contains(crate::config::ROUTE_RESTORE_RERESOLVE_KEY) &&
                row.contains("deliberate-opt-out"),
            "the opt-out must name its key and say it was deliberate; got {row}",
        );
    }


    /// The fixture index: releases newest-first with the `Requires-Dist` each
    /// one really carries, MEASURED from PyPI's own release metadata rather
    /// than invented (`1.75.3 -> protobuf<8.0.0,>=6.33.5`,
    /// `1.75.0 -> protobuf<8.0.0,>=4.25.8`).
    fn googleapis_releases(versions: &[(&str, &str)]) -> Vec<(String, Vec<String>)> {
        versions
            .iter()
            .map(|(version, requires)| ((*version).to_string(), vec![(*requires).to_string()]))
            .collect()
    }

    /// A rejected route for `googleapis-common-protos`, the shape the landing
    /// relock 6106911 printed as
    /// `### PYPI ROUTE RESTORED dep=googleapis-common-protos … crosses=protobuf`.
    fn googleapis_auto_route(version: &str) -> super::super::BundleAutoRoute {
        let mut route = pillow_auto_route(version);
        route.route.pypi_name = "googleapis-common-protos".to_string();
        route.route.conda_name = "googleapis-common-protos".to_string();
        route
    }

    /// A bundle carrying ONE workspace conda provider fact, which is what
    /// `workspace_owned_drops_from_lock` proved the pack's own uv lock held.
    fn bundle_with_protobuf_fact(fact_version: Option<&str>) -> Bundle {
        let mut bundle = test_bundle(&[]);
        bundle.auto_routed.push(googleapis_auto_route("1.75.3"));
        if let Some(fact_version) = fact_version {
            bundle.workspace_conda_provider_facts.insert(
                "protobuf".to_string(),
                super::super::WorkspaceCondaProviderFact {
                    selected_versions: [fact_version.to_string()].into_iter().collect(),
                    declared_specs: BTreeSet::new(),
                    present_in_all_consumers: true,
                },
            );
        }
        bundle
    }

    /// Drive the joint-solve restore against a fixture index that behaves the
    /// way `bfs_fetch_pypi_from_chain` does: it returns the HIGHEST release
    /// satisfying the specifiers it is handed, and errors when none does.
    async fn restore_against_index(
        bundle: &mut Bundle,
        releases: Vec<(String, Vec<String>)>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<JointRouteOutcome> {
        let fetch = move |request: PypiFetchRequest,
                          _indexes: Vec<String>,
                          failure_context: String| {
            let releases = releases.clone();
            let calls = Arc::clone(&calls);
            async move {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let mut best: Option<(Version, Vec<String>)> = None;
                for (raw, requires) in &releases {
                    let version = Version::from_str(raw).expect("fixture version parses");
                    if !request.specifiers.contains(&version) {
                        continue;
                    }
                    let better = match &best {
                        None => true,
                        Some((chosen, _)) => &version > chosen,
                    };
                    if better {
                        best = Some((version, requires.clone()));
                    }
                }
                let Some((version, requires_dist)) = best else {
                    return Err(anyhow!("{failure_context}"));
                };
                let requires: Vec<&str> = requires_dist.iter().map(String::as_str).collect();
                Ok(test_wheel(
                    &request.bundle_name,
                    &request.pypi_name,
                    &version.to_string(),
                    &requires,
                ))
            }
        };
        let mut metadata_routes = ProvisionalMetadataRoutes::new();
        let observed_requirements = ObservedRequirements::new();
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        // `uv_backed: false` is what puts this on the RESTORE path rather than
        // the keep-PyPI handoff: a bundle auto-routed without a uv closure has
        // no fixpoint to hand the name back to, so the restore is the only
        // actuator it has. `mode` is the new default in both arms.
        let context = UvReresolveContext {
            mode: UvReresolveMode::default(),
            uv_backed: false,
            keep_pypi: BTreeSet::new(),
        };
        jointly_unroute_unsolvable(
            bundle,
            &mut metadata_routes,
            &observed_requirements,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &test_config(),
            &reject_every_mutable_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &context,
        )
        .await
    }

    fn restored_version(bundle: &Bundle, pypi_name: &str) -> Option<String> {
        bundle
            .all_wheels()
            .find(|wheel| {
                PypiKey::from_pypi(&wheel.pypi_name) == PypiKey::from_pypi(pypi_name)
                    || PypiKey::from_pypi(&wheel.metadata.name) == PypiKey::from_pypi(pypi_name)
            })
            .map(|wheel| wheel.metadata.version.clone())
    }

    /// GUARD (a), N27-RETREAD-142. THE RESTORE BACKTRACKS INSTEAD OF
    /// RE-INJECTING.
    ///
    /// This is the production case, reduced to its two facts: the pack's own
    /// uv lock resolved `protobuf` at `5.29.3`, and the index offers
    /// `googleapis-common-protos 1.75.3` (`protobuf<8.0.0,>=6.33.5`) above
    /// `1.75.0` (`protobuf<8.0.0,>=4.25.8`). At 695f108 the restore takes
    /// 1.75.3 -- the CONTROL arm below shows the fixture really does offer it
    /// first, so the assertion is not an artefact of the fixture's ordering --
    /// and the emission then carries `>=6.33.5` into the pack's `constrains`.
    ///
    /// "NO CAP-OMITTED ROW" IS ASSERTED HERE AS THE CONDITION THAT PRODUCES
    /// ONE, not as the row itself. The omission fires two layers later, in
    /// `assemble_conda_output`, on a cap that excludes the shared fact; the
    /// exact predicate for "this restored wheel can produce that cap" is
    /// `fact_versions_excluded_by_requires_dist` returning non-empty for the
    /// wheel that survives. Asserting it empty here is asserting there is
    /// nothing for the omission to fire on -- and it is checkable at this
    /// layer, which a string search of another module's output is not.
    #[tokio::test]
    async fn capwins5_a_restored_route_backtracks_to_a_release_the_fact_admits() {
        let releases = googleapis_releases(&[
            ("1.75.3", "protobuf<8.0.0,>=6.33.5"),
            ("1.75.0", "protobuf<8.0.0,>=4.25.8"),
        ]);

        // CONTROL: no workspace conda fact, so nothing constrains the restore
        // and the index's newest release is taken. This is 695f108's answer.
        let mut control = bundle_with_protobuf_fact(None);
        let control_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        restore_against_index(&mut control, releases.clone(), Arc::clone(&control_calls))
            .await
            .expect("the control restore succeeds");
        assert_eq!(
            restored_version(&control, "googleapis-common-protos").as_deref(),
            Some("1.75.3"),
            "the fixture must really offer the crossing release first",
        );
        assert_eq!(control_calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        // THE GUARD: the same index, with the fact the lock actually held.
        let mut bundle = bundle_with_protobuf_fact(Some("5.29.3"));
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let outcome = restore_against_index(&mut bundle, releases, Arc::clone(&calls))
            .await
            .expect("a compatible release exists one below, so the restore succeeds");
        assert!(matches!(outcome, JointRouteOutcome::Mutated { .. }));
        assert_eq!(
            restored_version(&bundle, "googleapis-common-protos").as_deref(),
            Some("1.75.0"),
            "the restore must re-resolve under the workspace conda fact, not re-inject",
        );
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "exactly one backtrack: the crossing release, then the one below it",
        );

        let facts = admission_fact_holdings(&bundle);
        assert_eq!(facts.len(), 1, "one fact is in force");
        let env = crate::relax::marker_env_for("linux-64", "3.11").unwrap();
        let surviving = bundle
            .all_wheels()
            .find(|wheel| {
                PypiKey::from_pypi(&wheel.metadata.name)
                    == PypiKey::from_pypi("googleapis-common-protos")
            })
            .expect("the restored wheel is in the bundle");
        assert!(
            fact_versions_excluded_by_requires_dist(
                &surviving.metadata.requires_dist,
                &facts,
                &env,
            )
            .is_empty(),
            "the surviving wheel must give the cap-omission decision nothing to fire on; \
             requires_dist = {:?}",
            surviving.metadata.requires_dist,
        );
        assert_eq!(
            pypi_route_reresolved_row("googleapis-common-protos", "1.75.3", "1.75.0", facts.len()),
            "### PYPI ROUTE RERESOLVED dep=googleapis-common-protos from=1.75.3 to=1.75.0 \
             constraints=1",
        );
    }

    /// GUARD (b), N27-RETREAD-142, AS AMENDED BY N27-RETREAD-146. NO
    /// COMPATIBLE RELEASE IS A LOUD DEFERRAL, NOT A HALT.
    ///
    /// The index offers only the crossing release. -142 made this a refusal
    /// that killed the lock; relock `6113841` then died on `torch`, a pinned
    /// candidate set of ONE, on a `constrains` edge whose emission policy had
    /// been handling exactly this for weeks. So the restore now ADMITS the
    /// pinned wheel -- the pre-142 selection, byte for byte -- and prints the
    /// crossing. The `-142` needles stay in the prose, because the message is
    /// still the thing a reader needs to decide which side is wrong.
    #[tokio::test]
    async fn capwins5_a_restore_with_no_admissible_release_admits_and_defers_loudly() {
        let releases = googleapis_releases(&[("1.75.3", "protobuf<8.0.0,>=6.33.5")]);
        let mut bundle = bundle_with_protobuf_fact(Some("5.29.3"));
        let before = bundle.all_wheels().count();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        restore_against_index(&mut bundle, releases, Arc::clone(&calls))
            .await
            .expect("an unresolvable crossing must NOT halt the lock");
        assert_eq!(
            bundled_version(&bundle, "googleapis-common-protos").as_deref(),
            Some("1.75.3"),
            "the pinned wheel is admitted and emission decides the bound",
        );
        assert_eq!(
            bundle.all_wheels().count(),
            before + 1,
            "the restore admits exactly the one wheel it could not re-resolve",
        );

        // THE CROSSING REACHES THE LOG. Pinned by bytes, because a merge lane
        // and a gate criterion both grep for this row by string.
        let crossing = FactCrossing {
            fact_name: "protobuf".to_string(),
            fact_version: "5.29.3".to_string(),
            requirement: "protobuf<8.0.0,>=6.33.5".to_string(),
            holders_total: 4,
        };
        assert_eq!(
            pypi_admission_fact_crossing_unresolved_row(
                "googleapis-common-protos",
                &crossing,
                FactConstrainedSite::JointRouteRestore,
            ),
            "### PYPI ADMISSION FACT-CROSSING UNRESOLVED dep=googleapis-common-protos \
             requirement=protobuf<8.0.0,>=6.33.5 fact=protobuf==5.29.3 consumers_holding=0/4 door=restore \
             policy=constrains-only:learned-fact-yields-to-cap|declared-pin-kept|\
             undecidable-omitted",
        );
        let message = fact_crossing_unresolved_message(
            "googleapis-common-protos",
            "isaaclab-2-3x-pack",
            &VersionSpecifiers::from_str("~=1.52").unwrap(),
            &[Version::from_str("1.75.3").unwrap()],
            &crossing,
            FactConstrainedSite::JointRouteRestore,
        );
        for needle in [
            "googleapis-common-protos",
            "protobuf==5.29.3",
            "protobuf<8.0.0,>=6.33.5",
            "1.75.3",
            "N27-RETREAD-142",
            "N27-RETREAD-146",
        ] {
            assert!(
                message.contains(needle),
                "the crossing message must name `{needle}`; got {message}",
            );
        }
    }

    /// GUARD (c), N27-RETREAD-142. AN ALREADY-COMPATIBLE RESTORE IS UNCHANGED
    /// AND STILL SAYS SO.
    ///
    /// The ordinary case, which is nearly every restore: the pre-routing
    /// selection already admits every fact. The version must not move -- this
    /// is the arm that pins "the fix changes emitted bytes only where a route
    /// actually crossed a fact" -- and exactly ONE fetch must happen, so no
    /// silent extra index round-trip is bought for the ordinary path. The row
    /// still prints, with `from == to`: a row that appeared only on a change
    /// could not distinguish "checked, needed nothing" from "never checked",
    /// and the second is the defect.
    #[tokio::test]
    async fn capwins5_a_restore_that_already_admits_the_fact_keeps_its_version() {
        let releases = googleapis_releases(&[("1.75.0", "protobuf<8.0.0,>=4.25.8")]);
        let mut bundle = bundle_with_protobuf_fact(Some("5.29.3"));
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        restore_against_index(&mut bundle, releases, Arc::clone(&calls))
            .await
            .expect("the only release admits the fact");
        assert_eq!(
            restored_version(&bundle, "googleapis-common-protos").as_deref(),
            Some("1.75.0"),
        );
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the ordinary path costs exactly one fetch, as before the fix",
        );
        assert_eq!(
            pypi_route_reresolved_row("googleapis-common-protos", "1.75.0", "1.75.0", 1),
            "### PYPI ROUTE RERESOLVED dep=googleapis-common-protos from=1.75.0 to=1.75.0 \
             constraints=1",
        );
    }

    /// The specifier-aware crossing test is what makes the actuator fire on
    /// the ONE row that matters instead of on the benign
    /// `crosses=packaging,numpy,…` neighbours -142 records. Held separately
    /// from the guards above because it is the predicate they all rest on.
    #[test]
    fn capwins5_only_a_specifier_that_excludes_the_fact_is_a_crossing() {
        let env = crate::relax::marker_env_for("linux-64", "3.11").unwrap();
        // ONE holder per name: the single-consumer shape, which is what a
        // `--base-lock drop` relock has for every name and therefore what this
        // guard must keep asserting unchanged (N27-RETREAD-141).
        let facts: BTreeMap<String, BTreeMap<String, Version>> = [
            (
                "protobuf".to_string(),
                BTreeMap::from([(
                    "pace".to_string(),
                    Version::from_str("5.29.3").unwrap(),
                )]),
            ),
            (
                "packaging".to_string(),
                BTreeMap::from([(
                    "pace".to_string(),
                    Version::from_str("26.3").unwrap(),
                )]),
            ),
        ]
        .into_iter()
        .collect();

        // NAMES a fact and EXCLUDES it: the one that matters.
        let crossings = fact_versions_excluded_by_requires_dist(
            &["protobuf<8.0.0,>=6.33.5".to_string()],
            &facts,
            &env,
        );
        assert_eq!(crossings.len(), 1);
        assert_eq!(crossings[0].fact_name, "protobuf");
        assert_eq!(crossings[0].fact_version, "5.29.3");

        // NAMES a fact and ADMITS it: `route_restore_crossings` reports this
        // one and the actuator must not.
        assert!(
            fact_versions_excluded_by_requires_dist(
                &["packaging>=20".to_string(), "protobuf<8.0.0,>=4.25.8".to_string()],
                &facts,
                &env,
            )
            .is_empty(),
            "an admitting requirement is not a crossing",
        );
        assert_eq!(
            route_restore_crossings(
                &["packaging>=20".to_string(), "protobuf<8.0.0,>=4.25.8".to_string()],
                &["packaging".to_string(), "protobuf".to_string()]
                    .into_iter()
                    .collect(),
            ),
            vec!["packaging".to_string(), "protobuf".to_string()],
            "the READER's row still matches on names, which is why it is not the predicate",
        );

        // No version clause, an unnamed package, and a marker this target
        // does not satisfy all yield nothing.
        assert!(
            fact_versions_excluded_by_requires_dist(&["protobuf".to_string()], &facts, &env)
                .is_empty()
        );
        assert!(
            fact_versions_excluded_by_requires_dist(&["numpy<1".to_string()], &facts, &env)
                .is_empty()
        );
        assert!(
            fact_versions_excluded_by_requires_dist(
                &["protobuf>=6.33.5 ; sys_platform == \"win32\"".to_string()],
                &facts,
                &env,
            )
            .is_empty(),
            "a requirement this target never imposes cannot force a backtrack",
        );
    }

    /// A conda spelling PEP 440 reads as a DIFFERENT version is "no answer
    /// about PyPI" and must leave the door unconstrained rather than guessing.
    ///
    /// N27-RETREAD-141 SPLIT THIS GUARD'S TWO LEGS APART AND THE ASSERTION
    /// BELOW SAYS SO. The version-scheme leg (tzdata) is unchanged: it is not a
    /// fact about PyPI at all. The two-selections leg (protobuf) MOVED and had
    /// to: "the consumers disagree" is not "there is nothing to check", it is a
    /// question with a per-consumer answer, and dropping the name meant a
    /// requirement excluding EVERY one of their versions was admitted in
    /// silence. protobuf is therefore now carried with both holders, and
    /// whether it crosses is decided by
    /// [`fact_verdict_for_requires_dist`], not here.
    ///
    /// THIS GUARD CAUGHT A REAL DEFECT IN MY FIRST WRITING OF
    /// the fact readers, which is why the tzdata leg is not decorative.
    /// I had filtered on "parses as PEP 440", believing conda `2026c` would
    /// fail to parse. It PARSES -- PEP 440 reads the trailing `c` as a
    /// release-candidate marker and normalizes it to `2026rc0` -- so the
    /// restore would have been constrained by a version tzdata never published,
    /// and could have backtracked or REFUSED against a fact that does not
    /// exist. Gate 6108932 printed
    /// `left: ["packaging", "tzdata"]  right: ["packaging"]`, and the fix is
    /// `uv_closure::learned_fact_constraints` own round-trip test, inherited
    /// rather than re-invented.
    #[test]
    fn fact1_a_version_scheme_mismatch_constrains_nothing_but_a_split_is_carried() {
        let mut bundle = test_bundle(&[]);
        bundle.workspace_conda_provider_facts.insert(
            "protobuf".to_string(),
            super::super::WorkspaceCondaProviderFact {
                selected_versions: ["5.29.3".to_string(), "6.33.5".to_string()]
                    .into_iter()
                    .collect(),
                declared_specs: BTreeSet::new(),
                present_in_all_consumers: false,
            },
        );
        bundle.workspace_conda_provider_facts.insert(
            "tzdata".to_string(),
            super::super::WorkspaceCondaProviderFact {
                selected_versions: ["2026c".to_string()].into_iter().collect(),
                declared_specs: BTreeSet::new(),
                present_in_all_consumers: true,
            },
        );
        bundle.workspace_conda_provider_facts.insert(
            "packaging".to_string(),
            super::super::WorkspaceCondaProviderFact {
                selected_versions: ["26.3".to_string()].into_iter().collect(),
                declared_specs: BTreeSet::new(),
                present_in_all_consumers: true,
            },
        );
        let facts = admission_fact_holdings(&bundle);
        assert_eq!(
            facts.keys().cloned().collect::<Vec<_>>(),
            vec!["packaging".to_string(), "protobuf".to_string()],
            "a version-scheme mismatch is not a fact about PyPI; a split IS",
        );
        assert_eq!(
            facts["protobuf"].len(),
            2,
            "both holders are carried so the predicate can ask about each",
        );
        assert_eq!(facts["packaging"].len(), 1);
    }

    /// The backtrack is the whole mechanism, so its one primitive is pinned.
    #[test]
    fn capwins5_a_backtrack_narrows_the_request_by_exclusion() {
        let base = VersionSpecifiers::from_str(">=1.70").unwrap();
        let narrowed = specifiers_excluding(
            &base,
            &[Version::from_str("1.75.3").unwrap()],
            "googleapis-common-protos",
        )
        .unwrap();
        assert!(!narrowed.contains(&Version::from_str("1.75.3").unwrap()));
        assert!(narrowed.contains(&Version::from_str("1.75.0").unwrap()));
        assert_eq!(
            specifiers_excluding(&base, &[], "googleapis-common-protos").unwrap(),
            base,
            "with nothing excluded the request is byte-identical to the old one",
        );
    }


    // ---- CAPWINS-6 / N27-RETREAD-145: the AUTO-BUNDLE door.

    /// A pack whose parent wheel declares the dep the way production does.
    ///
    /// MEASURED, not invented: relock `6112256`'s backend log prints
    /// `auto_bundle: auto-bundled into isaaclab-2-3x-pack
    /// dep=googleapis-common-protos version=~=1.52` -- a LOOSE requirement from
    /// a parent wheel's `Requires-Dist`, which is why this fixture uses
    /// `~=1.52` and not an exact pin.
    ///
    /// `auto_dropped` carries `protobuf` for the same reason production does:
    /// the name is workspace-owned (the conda provider holds it), so it is in
    /// the auto-bundle skip set and can never be fetched. That is precisely
    /// what makes the fact the ONLY thing standing between the closure and an
    /// unsatisfiable cap -- and, until this commit, nothing applied it here.
    fn bundle_with_loose_googleapis_dep(fact_version: Option<&str>) -> Bundle {
        let mut bundle = test_bundle(&["googleapis-common-protos~=1.52"]);
        bundle.auto_dropped.insert("protobuf".to_string());
        if let Some(fact_version) = fact_version {
            bundle.workspace_conda_provider_facts.insert(
                "protobuf".to_string(),
                super::super::WorkspaceCondaProviderFact {
                    selected_versions: [fact_version.to_string()].into_iter().collect(),
                    declared_specs: BTreeSet::new(),
                    present_in_all_consumers: true,
                },
            );
        }
        bundle
    }

    /// Drive the AUTO-BUNDLE admission against the same fixture index shape
    /// `restore_against_index` uses: the index returns the HIGHEST release
    /// satisfying the specifiers it is handed, and errors when none does,
    /// which is what `bfs_fetch_pypi_from_chain` does in production.
    ///
    /// The probe refuses every conda route (`satisfiable: Some(false)`), which
    /// is the state that sends a loose candidate down the PyPI path at all.
    async fn auto_bundle_against_index(
        bundle: &mut Bundle,
        releases: Vec<(String, Vec<String>)>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<AutoBundleOutcome> {
        let fetch = move |request: PypiFetchRequest,
                          _indexes: Vec<String>,
                          failure_context: String| {
            let releases = releases.clone();
            let calls = Arc::clone(&calls);
            async move {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let mut best: Option<(Version, Vec<String>)> = None;
                for (raw, requires) in &releases {
                    let version = Version::from_str(raw).expect("fixture version parses");
                    if !request.specifiers.contains(&version) {
                        continue;
                    }
                    let better = match &best {
                        None => true,
                        Some((chosen, _)) => &version > chosen,
                    };
                    if better {
                        best = Some((version, requires.clone()));
                    }
                }
                let Some((version, requires_dist)) = best else {
                    return Err(anyhow!("{failure_context}"));
                };
                let requires: Vec<&str> = requires_dist.iter().map(String::as_str).collect();
                Ok(test_wheel(
                    &request.bundle_name,
                    &request.pypi_name,
                    &version.to_string(),
                    &requires,
                ))
            }
        };
        let refuse_every_conda_route = |pairs: Vec<(String, String)>| async move {
            pairs
                .into_iter()
                .map(|(package, spec)| crate::probe::ProbeResult {
                    package,
                    spec,
                    channels_consulted: vec!["conda-forge/linux-64".to_string()],
                    satisfiable: Some(false),
                    matching_candidates: 0,
                })
                .collect()
        };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        auto_bundle_transitives_with(
            bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &test_config(),
            None,
            None,
            None,
            &refuse_every_conda_route,
            &reject_every_mutable_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext {
                mode: UvReresolveMode::default(),
                uv_backed: false,
                keep_pypi: BTreeSet::new(),
            },
        )
        .await
    }

    fn bundled_version(bundle: &Bundle, pypi_name: &str) -> Option<String> {
        bundle
            .all_wheels()
            .find(|wheel| {
                PypiKey::from_pypi(&wheel.pypi_name) == PypiKey::from_pypi(pypi_name)
                    || PypiKey::from_pypi(&wheel.metadata.name) == PypiKey::from_pypi(pypi_name)
            })
            .map(|wheel| wheel.metadata.version.clone())
    }

    /// GUARD (a), N27-RETREAD-145. THE AUTO-BUNDLE ADMISSION BACKTRACKS.
    ///
    /// This is relock `6112256` reduced to its two facts. That run took the
    /// AUTO-BUNDLE door -- `auto-bundled … googleapis-common-protos` twice,
    /// `PYPI ROUTE RESTORED … googleapis-common-protos` zero times, the exact
    /// reverse of `6106911` -- so N27-RETREAD-142's re-resolve was never
    /// consulted and 1.75.3 walked into the pack's `constrains` as
    /// `protobuf>=6.33.5` against the workspace fact `protobuf==5.29.3`.
    ///
    /// The CONTROL arm is not decoration: it proves the fixture really offers
    /// the crossing release first, so the guard's assertion is not an artefact
    /// of fixture ordering.
    #[tokio::test]
    async fn capwins6_an_auto_bundled_wheel_backtracks_to_a_release_the_fact_admits() {
        let releases = googleapis_releases(&[
            ("1.75.3", "protobuf<8.0.0,>=6.33.5"),
            ("1.75.0", "protobuf<8.0.0,>=4.25.8"),
        ]);

        // CONTROL: no workspace conda fact, so nothing constrains the
        // admission and the index's newest release is taken. This is 96ff3dd's
        // answer, and it is the production RED.
        let mut control = bundle_with_loose_googleapis_dep(None);
        let control_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        auto_bundle_against_index(&mut control, releases.clone(), Arc::clone(&control_calls))
            .await
            .expect("the control admission succeeds");
        assert_eq!(
            bundled_version(&control, "googleapis-common-protos").as_deref(),
            Some("1.75.3"),
            "the fixture must really offer the crossing release first",
        );
        assert_eq!(control_calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        // THE GUARD: the same index, with the fact the workspace really holds.
        let mut bundle = bundle_with_loose_googleapis_dep(Some("5.29.3"));
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        auto_bundle_against_index(&mut bundle, releases, Arc::clone(&calls))
            .await
            .expect("a compatible release exists one below, so the admission succeeds");
        assert_eq!(
            bundled_version(&bundle, "googleapis-common-protos").as_deref(),
            Some("1.75.0"),
            "the auto-bundle admission must resolve under the workspace conda facts",
        );
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "exactly one backtrack: the crossing release, then the one below it",
        );

        // NO UNSATISFIABLE `constrains` CAN BE EMITTED, asserted as the
        // CONDITION that produces one rather than as a string search of
        // another module's output -- the same shape CAPWINS-5's guard (a) uses.
        let facts = admission_fact_holdings(&bundle);
        assert_eq!(facts.len(), 1, "one fact is in force");
        let env = crate::relax::marker_env_for("linux-64", "3.11").unwrap();
        let surviving = bundle
            .all_wheels()
            .find(|wheel| {
                PypiKey::from_pypi(&wheel.metadata.name)
                    == PypiKey::from_pypi("googleapis-common-protos")
            })
            .expect("the admitted wheel is in the bundle");
        assert!(
            fact_versions_excluded_by_requires_dist(
                &surviving.metadata.requires_dist,
                &facts,
                &env,
            )
            .is_empty(),
            "the surviving wheel must give the cap-omission decision nothing to fire on; \
             requires_dist = {:?}",
            surviving.metadata.requires_dist,
        );
        assert_eq!(
            pypi_closure_fact_constrained_row(
                facts.len(),
                &[(
                    "googleapis-common-protos".to_string(),
                    "1.75.3".to_string(),
                    "1.75.0".to_string(),
                    4,
                )],
            ),
            "### PYPI CLOSURE FACT-CONSTRAINED names=1 backtracked=googleapis-common-protos \
              1.75.3 1.75.0 consumers_holding=0/4",
        );
    }

    /// GUARD (c), N27-RETREAD-145, AS AMENDED BY N27-RETREAD-146. NO
    /// ADMISSIBLE RELEASE IS A LOUD DEFERRAL AT THIS DOOR TOO.
    ///
    /// This is relock `6113841` in miniature, with the dep the fixtures
    /// already carry. `torch` is pinned `==2.5.1+cu124`: ONE candidate, whose
    /// `Requires-Dist` says `sympy (==1.13.1)` against the workspace fact
    /// `sympy==1.14.0`. -145 refused and the whole lock died at 216 s. B42's
    /// `sage-isaac-pack` record, the last green, emits `sympy` as a BARE name
    /// and those environments ran. So the door admits and prints; emission
    /// decides the bound.
    #[tokio::test]
    async fn capwins6_an_auto_bundle_with_no_admissible_release_admits_and_defers_loudly() {
        let releases = googleapis_releases(&[("1.75.3", "protobuf<8.0.0,>=6.33.5")]);
        let mut bundle = bundle_with_loose_googleapis_dep(Some("5.29.3"));
        let before = bundle.all_wheels().count();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        auto_bundle_against_index(&mut bundle, releases, Arc::clone(&calls))
            .await
            .expect("an unresolvable crossing must NOT halt the lock");
        assert_eq!(
            bundled_version(&bundle, "googleapis-common-protos").as_deref(),
            Some("1.75.3"),
            "the pinned wheel is admitted and emission decides the bound",
        );
        assert_eq!(
            bundle.all_wheels().count(),
            before + 1,
            "the admission admits exactly the one wheel it could not re-resolve",
        );

        // THE IDENTITY THAT MAKES THE DEFERRAL SAFE, ASSERTED AND NOT ARGUED.
        // Emission is a function of the admitted wheels and the workspace
        // facts. The facts do not move here, so "the emitted `constrains` are
        // what 9ca588f emitted" reduces to "the admitted wheel is what
        // 9ca588f admitted" -- and 9ca588f's behaviour on this path is
        // exactly the fact-free arm, because before N27-RETREAD-145 no fact
        // reached this fetch at all.
        let mut unconstrained = bundle_with_loose_googleapis_dep(None);
        let unconstrained_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        auto_bundle_against_index(
            &mut unconstrained,
            googleapis_releases(&[("1.75.3", "protobuf<8.0.0,>=6.33.5")]),
            Arc::clone(&unconstrained_calls),
        )
        .await
        .expect("the pre-145 admission succeeds");
        let admitted = |b: &Bundle| {
            b.all_wheels()
                .find(|wheel| {
                    PypiKey::from_pypi(&wheel.metadata.name)
                        == PypiKey::from_pypi("googleapis-common-protos")
                })
                .map(|wheel| (wheel.metadata.version.clone(), wheel.metadata.requires_dist.clone()))
                .expect("the admitted wheel is in the bundle")
        };
        assert_eq!(
            admitted(&bundle),
            admitted(&unconstrained),
            "a crossing this door cannot resolve must admit BYTE-FOR-BYTE what the pre-145 \
             door admitted, so emission sees exactly the inputs it saw in the last green",
        );
        // The deferral costs exactly ONE extra probe -- the backtrack that
        // found the index empty -- and nothing else. Pinned, because a door
        // that quietly re-walked the index would be a different fix.
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert_eq!(
            unconstrained_calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
        );

        let crossing = FactCrossing {
            fact_name: "protobuf".to_string(),
            fact_version: "5.29.3".to_string(),
            requirement: "protobuf<8.0.0,>=6.33.5".to_string(),
            holders_total: 4,
        };
        assert_eq!(
            pypi_admission_fact_crossing_unresolved_row(
                "googleapis-common-protos",
                &crossing,
                FactConstrainedSite::AutoBundleAdmission,
            ),
            "### PYPI ADMISSION FACT-CROSSING UNRESOLVED dep=googleapis-common-protos \
             requirement=protobuf<8.0.0,>=6.33.5 fact=protobuf==5.29.3 consumers_holding=0/4 door=auto-bundle \
             policy=constrains-only:learned-fact-yields-to-cap|declared-pin-kept|\
             undecidable-omitted",
        );
        let message = fact_crossing_unresolved_message(
            "googleapis-common-protos",
            "isaaclab-2-3x-pack",
            &VersionSpecifiers::from_str("~=1.52").unwrap(),
            &[Version::from_str("1.75.3").unwrap()],
            &crossing,
            FactConstrainedSite::AutoBundleAdmission,
        );
        for needle in [
            "auto-bundle PyPI admission",
            "googleapis-common-protos",
            "protobuf==5.29.3",
            "protobuf<8.0.0,>=6.33.5",
            "1.75.3",
            "N27-RETREAD-145",
            "N27-RETREAD-146",
        ] {
            assert!(
                message.contains(needle),
                "the crossing message must name `{needle}`; got {message}",
            );
        }
    }

    /// N27-RETREAD-146. THE TWO DOORS DIFFER ONLY IN `door=`.
    ///
    /// The row is a gate criterion's grep target, so its `door=` field must be
    /// a stable single token and the two doors must be distinguishable by it
    /// alone -- the failure -145 was built for is a fix that guards one door
    /// while the production RED comes back through the other.
    #[test]
    fn capwins7_the_unresolved_row_names_the_door_and_the_emission_policy() {
        let crossing = FactCrossing {
            fact_name: "sympy".to_string(),
            fact_version: "1.14.0".to_string(),
            requirement: "sympy (==1.13.1) ; python_version >= \"3.9\"".to_string(),
            holders_total: 1,
        };
        let admission = pypi_admission_fact_crossing_unresolved_row(
            "torch",
            &crossing,
            FactConstrainedSite::AutoBundleAdmission,
        );
        let restore = pypi_admission_fact_crossing_unresolved_row(
            "torch",
            &crossing,
            FactConstrainedSite::JointRouteRestore,
        );
        assert_eq!(
            admission,
            "### PYPI ADMISSION FACT-CROSSING UNRESOLVED dep=torch \
             requirement=sympy (==1.13.1) ; python_version >= \"3.9\" fact=sympy==1.14.0 \
             consumers_holding=0/1 door=auto-bundle policy=constrains-only:learned-fact-yields-to-cap|\
             declared-pin-kept|undecidable-omitted",
        );
        assert_eq!(
            restore,
            admission.replace("door=auto-bundle", "door=restore"),
            "the two rows must differ in `door=` and nothing else",
        );
        assert!(
            admission.contains(EMISSION_CROSSING_POLICY),
            "the row must state what emission will do; got {admission}",
        );
    }

    /// GUARD (d), N27-RETREAD-145. AN ORDINARY PACK IS BYTE-FOR-BYTE
    /// UNCHANGED, AND COSTS NO EXTRA FETCH.
    ///
    /// Nearly every auto-bundled wheel already admits every fact. This is the
    /// non-regression arm that pins "the fix moves emitted bytes only where a
    /// requirement really crossed a fact": the selection must not move, and
    /// exactly ONE index round-trip may happen, so the whole daily lock does
    /// not silently buy a second fetch per dep. Both arms -- fact present and
    /// fact absent -- must agree, which is what "no crossing" means.
    #[tokio::test]
    async fn capwins6_an_ordinary_admission_keeps_its_version_and_its_one_fetch() {
        let releases = googleapis_releases(&[("1.75.0", "protobuf<8.0.0,>=4.25.8")]);

        let mut constrained = bundle_with_loose_googleapis_dep(Some("5.29.3"));
        let constrained_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        auto_bundle_against_index(
            &mut constrained,
            releases.clone(),
            Arc::clone(&constrained_calls),
        )
        .await
        .expect("the only release admits the fact");

        let mut unconstrained = bundle_with_loose_googleapis_dep(None);
        let unconstrained_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        auto_bundle_against_index(
            &mut unconstrained,
            releases,
            Arc::clone(&unconstrained_calls),
        )
        .await
        .expect("the control admission succeeds too");

        assert_eq!(
            bundled_version(&constrained, "googleapis-common-protos"),
            bundled_version(&unconstrained, "googleapis-common-protos"),
            "with no crossing the fact-constrained admission and the old one must agree",
        );
        assert_eq!(
            bundled_version(&constrained, "googleapis-common-protos").as_deref(),
            Some("1.75.0"),
        );
        assert_eq!(
            constrained_calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the ordinary path costs exactly one fetch, as before the fix",
        );
        assert_eq!(
            unconstrained_calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
        );
        // The pass still SAYS it was checked. A row that appeared only on a
        // change could not tell "checked, needed nothing" from "never
        // checked", and the second is the whole defect.
        assert_eq!(
            pypi_closure_fact_constrained_row(1, &[]),
            "### PYPI CLOSURE FACT-CONSTRAINED names=1 backtracked=none",
        );
    }

    /// The crossing message must say WHICH door hit it, because the two doors are why
    /// N27-RETREAD-142 read as fixed and came back: `6106911` took the restore
    /// (2 rows) and `6112256` took the auto-bundle (2 rows), same dep, same
    /// cap, same RED.
    #[test]
    fn capwins6_the_crossing_message_names_the_admission_path_it_fired_on() {
        let crossing = FactCrossing {
            fact_name: "protobuf".to_string(),
            fact_version: "5.29.3".to_string(),
            requirement: "protobuf<8.0.0,>=6.33.5".to_string(),
            holders_total: 4,
        };
        let base = VersionSpecifiers::from_str("~=1.52").unwrap();
        let excluded = [Version::from_str("1.75.3").unwrap()];
        let restore = fact_crossing_unresolved_message(
            "googleapis-common-protos",
            "isaaclab-2-3x-pack",
            &base,
            &excluded,
            &crossing,
            FactConstrainedSite::JointRouteRestore,
        );
        let admission = fact_crossing_unresolved_message(
            "googleapis-common-protos",
            "isaaclab-2-3x-pack",
            &base,
            &excluded,
            &crossing,
            FactConstrainedSite::AutoBundleAdmission,
        );
        assert!(restore.starts_with("joint-solve route restore for"), "{restore}");
        assert!(
            admission.starts_with("auto-bundle PyPI admission for"),
            "{admission}",
        );
        assert_ne!(restore, admission, "the two doors must be distinguishable");
        // N27-RETREAD-142's own needles survive the generalisation, which is
        // what keeps CAPWINS-5's refusal guard a real guard and not a rewrite.
        for needle in ["protobuf==5.29.3", "protobuf<8.0.0,>=6.33.5", "1.75.3", "N27-RETREAD-142"] {
            assert!(restore.contains(needle), "{restore}");
            assert!(admission.contains(needle), "{admission}");
        }
    }

    /// The per-pass row is the reader for a decision that otherwise leaves no
    /// trace (law 2). Its two shapes are pinned because a merge lane greps for
    /// them by string.
    #[test]
    fn capwins6_the_per_pass_row_reports_both_shapes() {
        assert_eq!(
            pypi_closure_fact_constrained_row(0, &[]),
            "### PYPI CLOSURE FACT-CONSTRAINED names=0 backtracked=none",
        );
        assert_eq!(
            pypi_closure_fact_constrained_row(
                3,
                &[
                    ("googleapis-common-protos".to_string(), "1.75.3".to_string(), "1.75.0".to_string(), 4),
                    ("wandb".to_string(), "0.30.0".to_string(), "0.29.0".to_string(), 2),
                ],
            ),
            "### PYPI CLOSURE FACT-CONSTRAINED names=3 backtracked=googleapis-common-protos \
              1.75.3 1.75.0 consumers_holding=0/4, wandb 0.30.0 0.29.0 consumers_holding=0/2",
        );
    }
    #[test]
    fn dependency_conflict_names_workspace_scope_and_remediation() {
        let context = JointRouteDiagnosticContext {
            bundle: "isaaclab-pack".to_string(),
            environments: vec!["uwlab-gpu".to_string()],
            profiles: vec!["linux-64-cuda-12".to_string()],
            platform: "linux-64".to_string(),
            python: "3.11".to_string(),
        };
        let constraints = vec![
            Constraint {
                specifiers: VersionSpecifiers::from_str("<2").unwrap(),
                provenance: Provenance::IndexWheelMetadata,
                source: "isaaclab==0.54.2, isaaclab_rl==0.4.7, \
                         isaaclab_tasks==0.11.12"
                    .to_string(),
                origin_id: test_origin("isaaclab-family-numpy-cap", "<2"),
            },
            Constraint {
                specifiers: VersionSpecifiers::from_str(">=2.0").unwrap(),
                provenance: Provenance::UvConstraint,
                source: "cmeel-boost==1.90.0".to_string(),
                origin_id: test_origin("cmeel-boost-numpy-floor", ">=2.0"),
            },
        ];

        let message = finalize(&PypiKey::from_pypi("numpy"), &constraints)
            .unwrap_err()
            .with_scope(context.scope())
            .to_string();

        assert_eq!(
            message,
            "dependency conflict in environment 'uwlab-gpu' for bundle 'isaaclab-pack' \
             (target profile 'linux-64-cuda-12', platform linux-64, python 3.11): \
             `numpy` requirements are mutually unsatisfiable: `<2` required by \
             isaaclab==0.54.2, isaaclab_rl==0.4.7, isaaclab_tasks==0.11.12; \
             `>=2.0` required by cmeel-boost==1.90.0. Resolve by pinning one side, or \
             use `retread-relax`, `retread-overrides`, or `retread-drop-deps` in the \
             pack manifest (see README)."
        );
    }

    #[test]
    fn source_built_metadata_alias_fixes_entire_conda_route_group() {
        let specifiers = VersionSpecifiers::from_str(">=4.0.1,<4.1").unwrap();
        let mut routes = ProvisionalMetadataRoutes::new();
        let mut observed = ObservedRequirements::new();

        record_metadata_route(
            &mut routes,
            "index-alias".to_string(),
            "shared-conda-dep".to_string(),
            None,
        );
        observe_requirement(
            &mut observed,
            "index-alias",
            &specifiers,
            "index-origin",
            "1.0.0",
            "index-alias>=4.0.1,<4.1",
            Provenance::IndexWheelMetadata,
        );
        assert!(
            !metadata_route_group_has_source_built_origin(&routes, &observed, "shared-conda-dep")
                .unwrap(),
            "an index-only metadata route must remain mutable"
        );

        record_metadata_route(
            &mut routes,
            "source-alias".to_string(),
            "shared-conda-dep".to_string(),
            None,
        );
        observe_requirement(
            &mut observed,
            "source-alias",
            &specifiers,
            "source-origin",
            "1.0.0",
            "source-alias>=4.0.1,<4.1",
            Provenance::SourceBuiltRelaxed,
        );
        assert!(
            metadata_route_group_has_source_built_origin(&routes, &observed, "shared-conda-dep")
                .unwrap(),
            "one source-built alias must fix the shared conda identity"
        );
    }

    #[test]
    fn metadata_route_source_ownership_requires_recorded_provenance() {
        let mut routes = ProvisionalMetadataRoutes::new();
        record_metadata_route(
            &mut routes,
            "orphan-origin".to_string(),
            "orphan-conda-dep".to_string(),
            None,
        );

        let missing = metadata_route_group_has_source_built_origin(
            &routes,
            &ObservedRequirements::new(),
            "orphan-conda-dep",
        )
        .unwrap_err()
        .to_string();
        assert!(
            missing.contains("source requirement is unavailable"),
            "{missing}"
        );

        let mut empty = ObservedRequirements::new();
        empty.insert(PypiKey::from_pypi("orphan-origin"), Vec::new());
        let empty =
            metadata_route_group_has_source_built_origin(&routes, &empty, "orphan-conda-dep")
                .unwrap_err()
                .to_string();
        assert!(empty.contains("no active source requirements"), "{empty}");

        let mut mixed_routes = ProvisionalMetadataRoutes::new();
        record_metadata_route(
            &mut mixed_routes,
            "source-first".to_string(),
            "mixed-conda-dep".to_string(),
            None,
        );
        record_metadata_route(
            &mut mixed_routes,
            "orphan-second".to_string(),
            "mixed-conda-dep".to_string(),
            None,
        );
        let mut source_only = ObservedRequirements::new();
        observe_requirement(
            &mut source_only,
            "source-first",
            &VersionSpecifiers::empty(),
            "source-first-origin",
            "1.0.0",
            "source-first",
            Provenance::SourceBuiltRelaxed,
        );
        let later_missing = metadata_route_group_has_source_built_origin(
            &mixed_routes,
            &source_only,
            "mixed-conda-dep",
        )
        .unwrap_err()
        .to_string();
        assert!(later_missing.contains("orphan-second"), "{later_missing}");
        assert!(
            later_missing.contains("source requirement is unavailable"),
            "{later_missing}"
        );
    }

    fn repo_record(name: &str, version: &str, depends: &[&str]) -> RepoDataRecord {
        let mut package_record = PackageRecord::new(
            name.parse().unwrap(),
            VersionWithSource::from_str(version).unwrap(),
            "h123456_0".to_string(),
        );
        package_record.subdir = "linux-64".to_string();
        package_record.depends = depends.iter().map(|dep| (*dep).to_string()).collect();
        RepoDataRecord {
            package_record,
            file_name: format!("{name}-{version}-h123456_0.conda"),
            url: Url::parse(&format!(
                "https://example.invalid/linux-64/{name}-{version}-h123456_0.conda"
            ))
            .unwrap(),
            channel: Some("https://example.invalid".into()),
        }
    }

    /// Build provider evidence through the production fact derivation path.
    /// Each tuple is `(environment, selected version, direct conda spec)`;
    /// absent versions model a successful consumer solve that did not select
    /// this provider, while absent specs model transitive conda provision.
    fn single_provider_facts(
        provider: &str,
        consumers: &[(&str, Option<&str>, Option<&str>)],
    ) -> super::super::WorkspaceCondaFacts {
        let env_records = consumers
            .iter()
            .map(|(env, version, _)| {
                (
                    (*env).to_string(),
                    version
                        .map(|version| vec![repo_record(provider, version, &[])])
                        .unwrap_or_default(),
                )
            })
            .collect();
        let env_conda_deps = consumers
            .iter()
            .map(|(env, _, spec)| {
                let deps = spec
                    .map(|spec| BTreeMap::from([(provider.to_string(), spec.to_string())]))
                    .unwrap_or_default();
                ((*env).to_string(), deps)
            })
            .collect();
        super::super::facts_from_solved_records(
            env_records,
            env_conda_deps,
            BTreeSet::new(),
            &NameMap::new(),
            "regression-pack",
            &BTreeMap::new(),
        )
    }

    async fn validated_probe(pairs: Vec<(String, String)>) -> Vec<crate::probe::ProbeResult> {
        pairs
            .into_iter()
            .map(|(package, spec)| crate::probe::ProbeResult {
                package,
                spec,
                channels_consulted: vec!["conda-forge/linux-64".to_string()],
                satisfiable: Some(true),
                matching_candidates: 1,
            })
            .collect()
    }

    async fn reject_every_mutable_route(
        routes: Vec<crate::uv_closure::CondaRouteSpec>,
    ) -> crate::uv_closure::CoInstallVerdict {
        if routes.is_empty() {
            crate::uv_closure::CoInstallVerdict::Sat
        } else {
            crate::uv_closure::CoInstallVerdict::Unsat(vec![
                "test fixture rejects every mutable route".to_string(),
            ])
        }
    }

    async fn reject_numpy_route(
        routes: Vec<crate::uv_closure::CondaRouteSpec>,
    ) -> crate::uv_closure::CoInstallVerdict {
        if routes
            .iter()
            .any(|route| route.conda_name.key().as_str() == "numpy")
        {
            crate::uv_closure::CoInstallVerdict::Unsat(vec![
                "test fixture rejects the generated NumPy route".to_string(),
            ])
        } else {
            crate::uv_closure::CoInstallVerdict::Sat
        }
    }

    fn pillow_auto_route(version: &str) -> super::super::BundleAutoRoute {
        super::super::BundleAutoRoute {
            route: crate::uv_closure::AutoRoutedPackage {
                pypi_name: "pillow".to_string(),
                conda_name: "pillow".to_string(),
                pypi_version: version.to_string(),
                conda_version: version.to_string(),
                channel: "https://conda.anaconda.org/conda-forge/linux-64".to_string(),
                input_requirements: Vec::new(),
                origin: crate::uv_closure::RouteOrigin::Fixpoint,
            },
            provenance: Provenance::PriorSelection,
            workspace_provider: None,
        }
    }

    #[tokio::test]
    async fn rejected_uv_route_returns_all_keep_pypi_origins_without_shadow_fetch() {
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Err(anyhow!("uv re-resolve must return before shadow fetch")) }
            }
        };
        let mut bundle = test_bundle(&[]);
        bundle.auto_routed.push(pillow_auto_route("10.4.0"));
        let mut alias = pillow_auto_route("10.4.0");
        alias.route.pypi_name = "Pillow_SIMD".to_string();
        bundle.auto_routed.push(alias);
        let mut metadata_routes = ProvisionalMetadataRoutes::new();
        record_metadata_route(
            &mut metadata_routes,
            "pillow-metadata-alias".to_string(),
            "pillow".to_string(),
            None,
        );
        let mut observed_requirements = ObservedRequirements::new();
        observe_requirement(
            &mut observed_requirements,
            "pillow-metadata-alias",
            &VersionSpecifiers::empty(),
            "pillow-metadata-origin",
            "1.0.0",
            "pillow-metadata-alias",
            Provenance::IndexWheelMetadata,
        );
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let context = UvReresolveContext {
            mode: UvReresolveMode::Enabled,
            uv_backed: true,
            keep_pypi: [PypiKey::from_pypi("already-kept")].into(),
        };

        let outcome = jointly_unroute_unsolvable(
            &mut bundle,
            &mut metadata_routes,
            &observed_requirements,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &test_config(),
            &reject_every_mutable_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &context,
        )
        .await
        .unwrap();

        assert_eq!(
            outcome,
            JointRouteOutcome::RetryKeepPypi {
                keep_pypi: [
                    PypiKey::from_pypi("already-kept"),
                    PypiKey::from_pypi("pillow"),
                    PypiKey::from_pypi("Pillow_SIMD"),
                    PypiKey::from_pypi("pillow-metadata-alias"),
                ]
                .into(),
            }
        );
        assert_eq!(fetch_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(bundle.auto_routed.len(), 2, "retry must not mutate bundle");
        assert!(metadata_routes.contains_key("pillow"));
    }

    /// The production auto-bundle outcome is consumed as the next uv
    /// fixpoint's dynamic keep set. The retry starts from the clean base
    /// request, so the rejected route's prior selection cannot become a hard
    /// equality; its lock must match a fresh direct keep-PyPI solve.
    #[tokio::test]
    async fn rejected_route_reenters_fixpoint_as_keep_pypi() {
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Err(anyhow!("uv re-resolve must bypass shadow fetch")) }
            }
        };
        let mut prior_route = pillow_auto_route("3.0.0");
        prior_route.route.pypi_name = "flatdict".to_string();
        prior_route.route.conda_name = "flatdict".to_string();
        prior_route.route.pypi_version = "3.0.0".to_string();
        prior_route.route.conda_version = "3.0.0".to_string();
        let mut bundle = test_bundle(&[]);
        bundle.auto_routed.push(prior_route);
        let target = crate::pypi::WheelTarget::for_subdir("3.12", "linux-64");
        let closure_members = BTreeMap::new();

        let outcome = auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &test_config(),
            None,
            None,
            Some(&closure_members),
            &validated_probe,
            &reject_every_mutable_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext {
                mode: UvReresolveMode::Enabled,
                uv_backed: true,
                keep_pypi: BTreeSet::new(),
            },
        )
        .await
        .unwrap();
        let AutoBundleOutcome::RetryKeepPypi { keep_pypi } = outcome else {
            panic!("rejected uv route must schedule a fixpoint retry")
        };
        assert_eq!(keep_pypi, BTreeSet::from([PypiKey::from_pypi("flatdict")]));
        assert_eq!(fetch_calls.load(std::sync::atomic::Ordering::SeqCst), 0);

        let request = crate::uv_closure::UvClosureRequest {
            bundle: "flatdict-pack".to_string(),
            python_version: "3.12".to_string(),
            conda_subdir: "linux-64".to_string(),
            // `root` stands in for the wheel whose hidden metadata pulls
            // flatdict; the mock solve returns that transitive selection.
            dependencies: vec!["root==1.0".to_string()],
            dependency_provenance: BTreeMap::new(),
            constraints: crate::uv_closure::ConstraintSet::default(),
            overrides: Vec::new(),
            no_emit_packages: Vec::new(),
            index_urls: vec![crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            built_wheel_sources: BTreeMap::new(),
            explicit_pins: BTreeMap::new(),
            workspace_owned: crate::uv_closure::WorkspaceOwnedPlan::default(),
            offline: false,
        };
        let retry_options = crate::uv_closure::AutoRouteOptions {
            enabled: true,
            keep_pypi: keep_pypi
                .iter()
                .map(|name| name.as_str().to_string())
                .collect(),
            ..Default::default()
        };
        let make_solve = |calls: Arc<Mutex<Vec<crate::uv_closure::UvClosureRequest>>>| {
            move |request: crate::uv_closure::UvClosureRequest| {
                let calls = Arc::clone(&calls);
                Box::pin(async move {
                    calls.lock().unwrap().push(request.clone());
                    assert!(
                        !request
                            .constraints
                            .constraints
                            .iter()
                            .chain(request.overrides.iter())
                            .chain(request.dependencies.iter())
                            .any(|line| line == "flatdict==3.0.0"),
                        "the rejected prior selection must not enter the clean retry as a hard pin"
                    );
                    let version = "4.0.1";
                    Ok(crate::uv_closure::UvClosure {
                        wheels: vec![crate::lock::LockWheel {
                            name: "flatdict".to_string(),
                            version: version.to_string(),
                            origin: crate::lock::Origin::Index,
                            filename: format!("flatdict-{version}-py3-none-any.whl"),
                            url: Some(format!("https://example.invalid/flatdict-{version}.whl")),
                            sha256: Some("00".repeat(32)),
                            requires_dist: Vec::new(),
                            must_ship: false,
                            upstream_url: None,
                            git_source: None,
                            sdist_source: None,
                        }],
                        pins: BTreeMap::from([("flatdict".to_string(), version.to_string())]),
                        uv_version: "0.11.26".to_string(),
                        auto_routed: Vec::new(),
                        auto_dropped: BTreeSet::new(),
                        dependency_graph: Default::default(),
                        effective_input_requirements: None,
                    })
                })
                    as futures::future::BoxFuture<'static, Result<crate::uv_closure::UvClosure>>
            }
        };
        let probe = |_name: String, _spec: String| {
            Box::pin(async {
                panic!("the dynamic keep set must veto flatdict before probing");
                #[allow(unreachable_code)]
                None
            })
                as futures::future::BoxFuture<'static, Option<crate::uv_closure::RouteProbeHit>>
        };
        let retry_calls = Arc::new(Mutex::new(Vec::new()));
        let retry = crate::uv_closure::auto_route_fixpoint_checked(
            &request,
            &retry_options,
            make_solve(Arc::clone(&retry_calls)),
            probe,
            |_| Box::pin(async { crate::uv_closure::CoInstallVerdict::Sat }),
        )
        .await
        .unwrap();
        let direct_calls = Arc::new(Mutex::new(Vec::new()));
        let direct = crate::uv_closure::auto_route_fixpoint_checked(
            &request,
            &retry_options,
            make_solve(Arc::clone(&direct_calls)),
            |_name, _spec| Box::pin(async { None }),
            |_| Box::pin(async { crate::uv_closure::CoInstallVerdict::Sat }),
        )
        .await
        .unwrap();

        assert_eq!(retry.pins, direct.pins);
        assert_eq!(retry.pins["flatdict"], "4.0.1");
        assert_ne!(retry.pins["flatdict"], "3.0.0");
        assert_eq!(retry_calls.lock().unwrap().len(), 1);
        assert_eq!(direct_calls.lock().unwrap().len(), 1);
    }

    fn holosoma_numpy_conflict_bundle() -> Bundle {
        let mut bundle = test_bundle(&[]);
        bundle.conda_name = "holosoma-pack".to_string();
        bundle.primary = test_wheel("holosoma", "holosoma", "0.0.1", &["numpy<2,>=1.23.5"]);
        bundle.extras = vec![
            test_wheel(
                "holosoma-retargeting",
                "holosoma-retargeting",
                "0.1.0",
                &["numpy>=2.3.5,<2.4"],
            ),
            test_wheel("mujoco-warp", "mujoco-warp", "0.0.1", &["numpy"]),
            test_wheel("smplx", "smplx", "0.1.28", &["numpy (>=1.16.2)"]),
        ];
        bundle.primary.metadata_provenance = Provenance::SourceBuiltRelaxed;
        bundle.extras[0].metadata_provenance = Provenance::SourceBuiltRelaxed;
        bundle
    }

    fn pace_packaging_conflict_bundle() -> Bundle {
        let mut bundle = test_bundle(&[]);
        bundle.conda_name = "isaac-pack".to_string();
        bundle.primary = test_wheel("isaaclab", "isaaclab", "0.54.2", &["packaging"]);
        bundle.primary.metadata_provenance = Provenance::SourceBuiltRelaxed;
        bundle.extras = vec![
            test_wheel("isaaclab-rl", "isaaclab_rl", "0.4.7", &["packaging<24"]),
            test_wheel(
                "isaacsim-core",
                "isaacsim-core",
                "5.1.0.0",
                &["packaging==23.0"],
            ),
            test_wheel("matplotlib", "matplotlib", "3.10.3", &["packaging>=20.0"]),
            test_wheel("skrl", "skrl", "2.1.0", &["packaging"]),
        ];
        bundle.extras[0].metadata_provenance = Provenance::SourceBuiltRelaxed;
        bundle
    }

    fn isaaclab_psutil_conflict_bundle() -> Bundle {
        let mut bundle = test_bundle(&[]);
        bundle.conda_name = "isaaclab-2.3x-pack".to_string();
        bundle.primary = test_wheel(
            "isaacsim-kernel",
            "isaacsim-kernel",
            "5.1.0.0",
            &["psutil==5.9.8"],
        );
        bundle.extras = vec![
            test_wheel("ipython", "ipython", "9.15.0", &["psutil>=7"]),
            test_wheel(
                "rl-games",
                "rl_games",
                "1.6.1",
                &["psutil (>=5.9.0,<6.0.0)"],
            ),
        ];
        bundle
    }

    fn isaacsim_typing_extensions_conflict_bundle() -> Bundle {
        let mut bundle = test_bundle(&[]);
        bundle.conda_name = "isaacsim-pack".to_string();
        bundle.primary = test_wheel(
            "isaacsim",
            "isaacsim",
            "5.1.0.0",
            &["typing-extensions==4.12.2"],
        );
        bundle.extras = vec![test_wheel(
            "onnx",
            "onnx",
            "1.22.0",
            &["typing-extensions>=4.15.0"],
        )];
        bundle
    }

    fn prior_selection_route(name: &str, version: &str) -> super::super::BundleAutoRoute {
        super::super::BundleAutoRoute {
            route: crate::uv_closure::AutoRoutedPackage {
                pypi_name: name.to_string(),
                conda_name: name.to_string(),
                pypi_version: version.to_string(),
                conda_version: version.to_string(),
                channel: "https://conda.example.invalid/linux-64".to_string(),
                input_requirements: Vec::new(),
                origin: crate::uv_closure::RouteOrigin::Fixpoint,
            },
            provenance: Provenance::PriorSelection,
            workspace_provider: None,
        }
    }

    fn assert_workspace_fact_conflict_before_ownership(
        bundle: &Bundle,
        config: &RetreadConfig,
        target: &crate::pypi::WheelTarget,
        package: &str,
        version: &str,
    ) {
        let error = super::super::emitted_bundle_route_specs(bundle, config, target)
            .expect_err("the unowned workspace fact must reproduce the typed route conflict");
        assert!(
            error
                .downcast_ref::<crate::constraint::Conflict>()
                .is_some(),
            "pre-ownership assembly must return the typed conflict: {error:#}"
        );
        let message = format!("{error:#}");
        assert!(
            message.contains(&format!(
                "`{package}` requirements are mutually unsatisfiable"
            )),
            "{message}"
        );
        assert!(
            message.contains(&format!("workspace conda fact `{package}=={version}`")),
            "the fixture must include the workspace fact in the conflict:\n{message}"
        );
    }

    #[tokio::test]
    async fn workspace_packaging_fact_drop_owns_pace_conflict() {
        let probe_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let probe = {
            let probe_calls = Arc::clone(&probe_calls);
            move |pairs: Vec<(String, String)>| {
                probe_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async move { validated_probe(pairs).await }
            }
        };
        let solve_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let co_solve = {
            let solve_calls = Arc::clone(&solve_calls);
            move |_routes: Vec<crate::uv_closure::CondaRouteSpec>| {
                solve_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async {
                    crate::uv_closure::CoInstallVerdict::Unsat(vec![
                        "the workspace packaging pin conflicts with wheel metadata".to_string(),
                    ])
                }
            }
        };
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Err(anyhow!("workspace-owned packaging must never fetch PyPI")) }
            }
        };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let config = test_config();
        let mut bundle = pace_packaging_conflict_bundle();
        let facts = single_provider_facts("packaging", &[("pace", Some("26.2"), Some("==26.2"))]);
        bundle.workspace_conda_versions = facts.common_selected_versions;
        bundle.workspace_conda_provider_facts = facts.provider_facts;
        bundle
            .auto_routed
            .push(prior_selection_route("packaging", "23.0"));
        assert_workspace_fact_conflict_before_ownership(
            &bundle,
            &config,
            &target,
            "packaging",
            "26.2",
        );

        bundle.apply_workspace_conda_fact_ownership(
            &config,
            &config.name_map,
            &BTreeSet::new(),
            &BTreeSet::new(),
        );
        assert_eq!(
            bundle.auto_dropped,
            HashSet::from(["packaging".to_string()])
        );
        assert!(
            bundle.auto_routed.is_empty(),
            "fact ownership must remove the stale uv route before emission"
        );

        let outcome = auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &config,
            None,
            None,
            None,
            &probe,
            &co_solve,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .unwrap();

        assert!(matches!(outcome, AutoBundleOutcome::Complete { .. }));
        assert_eq!(probe_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(solve_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(fetch_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(
            bundle.all_wheels().count(),
            5,
            "the parent wheels remain installed"
        );
        let emitted = super::super::emitted_bundle_route_specs(&bundle, &config, &target).unwrap();
        assert!(
            emitted
                .iter()
                .all(|route| route.conda_name.key().as_str() != "packaging"),
            "the generated pack must not re-emit workspace-owned packaging: {emitted:?}"
        );
    }

    #[tokio::test]
    async fn workspace_numpy_fact_drop_owns_holosoma_conflict() {
        let probe_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let probe = {
            let probe_calls = Arc::clone(&probe_calls);
            move |pairs: Vec<(String, String)>| {
                probe_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async move { validated_probe(pairs).await }
            }
        };
        let solve_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let co_solve = {
            let solve_calls = Arc::clone(&solve_calls);
            move |_routes: Vec<crate::uv_closure::CondaRouteSpec>| {
                solve_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async {
                    crate::uv_closure::CoInstallVerdict::Unsat(vec![
                        "the workspace NumPy pin conflicts with wheel metadata".to_string(),
                    ])
                }
            }
        };
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Err(anyhow!("workspace-owned numpy must never fetch PyPI")) }
            }
        };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let config = test_config();
        let mut bundle = holosoma_numpy_conflict_bundle();
        let facts =
            single_provider_facts("numpy", &[("holosoma", Some("1.26.4"), Some("==1.26.4"))]);
        bundle.workspace_conda_versions = facts.common_selected_versions;
        bundle.workspace_conda_provider_facts = facts.provider_facts;
        assert_workspace_fact_conflict_before_ownership(
            &bundle, &config, &target, "numpy", "1.26.4",
        );

        bundle.apply_workspace_conda_fact_ownership(
            &config,
            &config.name_map,
            &BTreeSet::new(),
            &BTreeSet::new(),
        );
        assert_eq!(bundle.auto_dropped, HashSet::from(["numpy".to_string()]));

        let outcome = auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &config,
            None,
            None,
            None,
            &probe,
            &co_solve,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .unwrap();

        assert!(matches!(outcome, AutoBundleOutcome::Complete { .. }));
        assert_eq!(probe_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(solve_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(fetch_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(
            bundle.all_wheels().count(),
            4,
            "the parent wheels remain installed"
        );
        let emitted = super::super::emitted_bundle_route_specs(&bundle, &config, &target).unwrap();
        assert!(
            emitted
                .iter()
                .all(|route| route.conda_name.key().as_str() != "numpy"),
            "the generated pack must not re-emit workspace-owned numpy: {emitted:?}"
        );
    }

    #[tokio::test]
    async fn partial_workspace_psutil_provider_replaces_conflicting_wheel_pins() {
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let config = tiered_mapped_config("psutil");
        let mut bundle = isaaclab_psutil_conflict_bundle();
        let requests = Arc::new(Mutex::new(Vec::<PypiFetchRequest>::new()));
        let fetch = {
            let requests = Arc::clone(&requests);
            move |request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                requests.lock().unwrap().push(request);
                async { Ok(test_wheel("psutil", "psutil", "7.2.2", &[])) }
            }
        };

        let outcome = auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &config,
            None,
            None,
            None,
            &validated_probe,
            &reject_every_mutable_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .unwrap();

        assert!(matches!(outcome, AutoBundleOutcome::Complete { .. }));
        let requests = requests.lock().unwrap();
        assert_eq!(
            requests.len(),
            1,
            "the conflicting conda route must restore exactly one PyPI wheel"
        );
        let request = &requests[0];
        assert_eq!(request.pypi_name, "psutil");
        assert_eq!(
            render_specifiers(&request.specifiers),
            ">=7",
            "the narrowest satisfiable final intersection must let IPython's floor win"
        );
        assert!(
            request
                .specifiers
                .contains(&Version::from_str("7.2.2").unwrap())
        );
        assert!(
            !request
                .specifiers
                .contains(&Version::from_str("5.9.8").unwrap())
        );
        assert_eq!(request.relaxations.len(), 2);
        assert_eq!(
            request
                .relaxations
                .iter()
                .map(|relaxation| (
                    relaxation.kind,
                    relaxation.original.as_str(),
                    relaxation.relaxed.as_str(),
                    relaxation.source.as_str(),
                    relaxation.tier,
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    WheelMetadataRelaxationKind::ExactPinWidened,
                    "==5.9.8",
                    ">=5",
                    "wheel `isaacsim-kernel==5.1.0.0` Requires-Dist `psutil==5.9.8`",
                    RelaxPolicy::Major,
                ),
                (
                    WheelMetadataRelaxationKind::UpperCapStripped,
                    ">=5.9.0,<6.0.0",
                    ">=5.9.0",
                    "wheel `rl_games==1.6.1` Requires-Dist `psutil (>=5.9.0,<6.0.0)`",
                    RelaxPolicy::StrongMajor,
                ),
            ]
        );
        for relaxation in &request.relaxations {
            let warning = relaxation.to_string();
            assert!(warning.contains("RETREAD AUTO-RELAXED"), "{warning}");
            assert!(warning.contains(&relaxation.original), "{warning}");
            assert!(warning.contains(&relaxation.relaxed), "{warning}");
            for source in [
                "wheel `isaacsim-kernel==5.1.0.0` Requires-Dist `psutil==5.9.8`",
                "wheel `ipython==9.15.0` Requires-Dist `psutil>=7`",
                "wheel `rl_games==1.6.1` Requires-Dist `psutil (>=5.9.0,<6.0.0)`",
            ] {
                assert!(
                    warning.contains(source),
                    "warning must name every involved wheel:\n{warning}"
                );
            }
            assert!(
                warning.contains("in bundle 'isaaclab-2.3x-pack' (platform linux-64, python 3.11)"),
                "{warning}"
            );
        }
        drop(requests);

        assert_eq!(
            bundle.all_wheels().count(),
            4,
            "the three declaring wheels and restored psutil wheel must remain"
        );
        assert!(bundle.all_wheels().any(|wheel| {
            PypiKey::from_pypi(&wheel.pypi_name) == PypiKey::from_pypi("psutil")
                && wheel.metadata.version == "7.2.2"
        }));
        let emitted = super::super::emitted_bundle_route_specs(&bundle, &config, &target).unwrap();
        assert!(
            emitted
                .iter()
                .all(|route| route.conda_name.key().as_str() != "psutil"),
            "the restored PyPI wheel must replace the conflicting conda run dep: {emitted:?}"
        );
    }

    #[tokio::test]
    async fn immutable_psutil_auto_route_unroutes_against_full_wheel_closure() {
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let mut config = test_config();
        config.relax = RelaxPolicy::PatchThenMinorThenMajorThenLastResort;
        let mut bundle = isaaclab_psutil_conflict_bundle();
        bundle
            .auto_routed
            .push(prior_selection_route("psutil", "5.9.8"));
        let requests = Arc::new(Mutex::new(Vec::<PypiFetchRequest>::new()));
        let fetch = {
            let requests = Arc::clone(&requests);
            move |request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                requests.lock().unwrap().push(request);
                async { Ok(test_wheel("psutil", "psutil", "7.2.2", &[])) }
            }
        };
        let accept_route_probe = |_routes: Vec<crate::uv_closure::CondaRouteSpec>| async {
            crate::uv_closure::CoInstallVerdict::Sat
        };

        let outcome = auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &config,
            None,
            None,
            None,
            &validated_probe,
            &accept_route_probe,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .unwrap();

        assert!(matches!(outcome, AutoBundleOutcome::Complete { .. }));
        let requests = requests.lock().unwrap();
        assert_eq!(
            requests.len(),
            1,
            "the full wheel closure must reject the otherwise-solvable conda route"
        );
        assert_eq!(requests[0].pypi_name, "psutil");
        assert_eq!(
            render_specifiers(&requests[0].specifiers),
            ">=7",
            "after unroute, ordinary wheel auto-relax must let IPython's floor win"
        );
        assert!(
            requests[0]
                .specifiers
                .contains(&Version::from_str("7.2.2").unwrap())
        );
        assert!(
            !requests[0]
                .specifiers
                .contains(&Version::from_str("5.9.8").unwrap())
        );
        assert_eq!(
            requests[0]
                .relaxations
                .iter()
                .map(|relaxation| relaxation.kind)
                .collect::<Vec<_>>(),
            vec![
                WheelMetadataRelaxationKind::ExactPinWidened,
                WheelMetadataRelaxationKind::UpperCapStripped,
            ],
            "the restored wheel closure must widen Isaac Sim's exact pin and strip rl_games' cap"
        );
        drop(requests);

        assert!(
            bundle
                .auto_routed
                .iter()
                .all(|route| canonical_conda_name(&route.route.conda_name) != "psutil"),
            "the immutable route envelope must disappear with the rejected route"
        );
        assert!(bundle.all_wheels().any(|wheel| {
            PypiKey::from_pypi(&wheel.pypi_name) == PypiKey::from_pypi("psutil")
                && wheel.metadata.version == "7.2.2"
        }));
        let emitted = super::super::emitted_bundle_route_specs(&bundle, &config, &target).unwrap();
        assert!(
            emitted
                .iter()
                .all(|route| route.conda_name.key().as_str() != "psutil"),
            "the restored PyPI wheel must replace the conflicting auto route: {emitted:?}"
        );
    }

    #[tokio::test]
    async fn immutable_abi_anchor_auto_route_remains_fail_closed() {
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let mut config = test_config();
        config.relax = RelaxPolicy::PatchThenMinorThenMajorThenLastResort;
        let mut bundle = test_bundle(&["numpy==1.26.0"]);
        bundle.extras.push(test_wheel(
            "newer-numpy-floor",
            "newer-numpy-floor",
            "1.0.0",
            &["numpy>=1.26.4"],
        ));
        bundle
            .auto_routed
            .push(prior_selection_route("numpy", "1.26.0"));
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Ok(test_wheel("numpy", "numpy", "1.26.4", &[])) }
            }
        };
        let accept_route_probe = |_routes: Vec<crate::uv_closure::CondaRouteSpec>| async {
            crate::uv_closure::CoInstallVerdict::Sat
        };

        let error = auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &config,
            None,
            None,
            None,
            &validated_probe,
            &accept_route_probe,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .expect_err("an ABI-anchor route conflict must remain fail-closed");

        assert!(
            error
                .downcast_ref::<crate::constraint::Conflict>()
                .is_some(),
            "the anchor failure must retain its typed conflict: {error:#}"
        );
        assert_eq!(
            fetch_calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "ABI anchors must be rejected before any PyPI restore"
        );
        assert!(
            bundle
                .auto_routed
                .iter()
                .any(|route| canonical_conda_name(&route.route.conda_name) == "numpy"),
            "the ABI-anchor conda route must remain intact"
        );
        assert!(
            bundle
                .all_wheels()
                .all(|wheel| canonical_conda_name(&wheel.pypi_name) != "numpy"),
            "the ABI anchor must not be restored as a PyPI wheel"
        );
    }

    #[tokio::test]
    async fn typing_extensions_exact_pin_widens_only_to_satisfy_newer_floor() {
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let workspace_scope = crate::workspace::ResolvedWorkspaceTarget {
            contract: crate::workspace::WorkspaceTargetContract {
                subdir: "linux-64".to_string(),
                declared_virtual_packages: BTreeMap::new(),
                detected_virtual_packages: BTreeMap::new(),
            },
            profiles: vec!["linux-64-cuda-12".to_string()],
            environments: vec!["isaacsim".to_string()],
        };
        let config = tiered_mapped_config("typing-extensions");
        let mut bundle = isaacsim_typing_extensions_conflict_bundle();
        let requests = Arc::new(Mutex::new(Vec::<PypiFetchRequest>::new()));
        let fetch = {
            let requests = Arc::clone(&requests);
            move |request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                requests.lock().unwrap().push(request);
                async {
                    Ok(test_wheel(
                        "typing-extensions",
                        "typing-extensions",
                        "4.15.0",
                        &[],
                    ))
                }
            }
        };
        let allow_source_route = |_route: crate::uv_closure::CondaRouteSpec| async {
            crate::uv_closure::CoInstallVerdict::Sat
        };

        let outcome = auto_bundle_transitives_with_route_precheck(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            Some(&workspace_scope),
            &config,
            None,
            None,
            None,
            &validated_probe,
            &reject_every_mutable_route,
            &allow_source_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            1,
            &UvReresolveContext::default(),
        )
        .await
        .unwrap();

        let AutoBundleOutcome::Complete { relaxations } = outcome else {
            panic!("typing-extensions restoration must complete")
        };
        assert_eq!(
            relaxations.len(),
            1,
            "the warning must remain pending for the outer transaction"
        );
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(request.pypi_name, "typing-extensions");
        assert_eq!(render_specifiers(&request.specifiers), ">=4.15.0,<5");
        assert!(
            request
                .specifiers
                .contains(&Version::from_str("4.15.0").unwrap())
        );
        assert!(
            !request
                .specifiers
                .contains(&Version::from_str("4.14.9").unwrap())
        );
        assert!(
            !request
                .specifiers
                .contains(&Version::from_str("5.0.0").unwrap())
        );
        assert_eq!(request.relaxations.len(), 1);
        let relaxation = &request.relaxations[0];
        assert_eq!(relaxation.bundle(), "isaacsim-pack");
        assert_eq!(
            (
                relaxation.kind,
                relaxation.original.as_str(),
                relaxation.relaxed.as_str(),
                relaxation.source.as_str(),
                relaxation.tier,
            ),
            (
                WheelMetadataRelaxationKind::ExactPinWidened,
                "==4.12.2",
                ">=4.12,<5",
                "wheel `isaacsim==5.1.0.0` Requires-Dist `typing-extensions==4.12.2`",
                RelaxPolicy::Minor,
            )
        );
        let warning = relaxation.to_string();
        assert!(warning.contains("`typing-extensions`"), "{warning}");
        assert!(warning.contains("isaacsim==5.1.0.0"), "{warning}");
        assert!(warning.contains("onnx==1.22.0"), "{warning}");
        assert!(
            warning.contains(
                "in environment 'isaacsim' for bundle 'isaacsim-pack' \
                 (target profile 'linux-64-cuda-12', platform linux-64, python 3.11)"
            ),
            "{warning}"
        );
        drop(requests);

        assert_eq!(bundle.all_wheels().count(), 3);
        assert!(bundle.all_wheels().any(|wheel| {
            PypiKey::from_pypi(&wheel.pypi_name) == PypiKey::from_pypi("typing-extensions")
                && wheel.metadata.version == "4.15.0"
        }));
        let emitted = super::super::emitted_bundle_route_specs(&bundle, &config, &target).unwrap();
        assert!(
            emitted
                .iter()
                .all(|route| route.conda_name.key().as_str() != "typing-extensions"),
            "the restored PyPI wheel must replace the conflicting conda run dep: {emitted:?}"
        );
    }

    #[tokio::test]
    async fn outer_fixed_point_failure_discards_pending_relaxation_warning() {
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let config = tiered_mapped_config("typing-extensions");
        let mut bundle = isaacsim_typing_extensions_conflict_bundle();
        let requests = Arc::new(Mutex::new(Vec::<PypiFetchRequest>::new()));
        let fetch = {
            let requests = Arc::clone(&requests);
            move |request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                let requested = request.pypi_name.clone();
                requests.lock().unwrap().push(request);
                async move {
                    if PypiKey::from_pypi(&requested) == PypiKey::from_pypi("typing-extensions") {
                        Ok(test_wheel(
                            "typing-extensions",
                            "typing-extensions",
                            "4.15.0",
                            &["late-missing==1"],
                        ))
                    } else {
                        Err(anyhow!(
                            "deliberate late fixed-point fetch failure for {requested}"
                        ))
                    }
                }
            }
        };
        let allow_source_route = |_route: crate::uv_closure::CondaRouteSpec| async {
            crate::uv_closure::CoInstallVerdict::Sat
        };

        let error = auto_bundle_transitives_with_route_precheck(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            None,
            &config,
            None,
            None,
            None,
            &validated_probe,
            &reject_every_mutable_route,
            &allow_source_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            1,
            &UvReresolveContext::default(),
        )
        .await
        .expect_err("a later fixed-point fetch must roll back the whole warning transaction");
        assert!(
            format!("{error:#}").contains("deliberate late fixed-point fetch failure"),
            "{error:#}"
        );

        let requests = requests.lock().unwrap();
        assert!(requests.len() >= 2, "{requests:?}");
        assert_eq!(
            requests[0].relaxations.len(),
            1,
            "the first inner restore selected a warning that must remain uncommitted"
        );
        assert!(
            requests[1..]
                .iter()
                .all(|request| request.relaxations.is_empty()),
            "no later request may inherit the rolled-back warning: {requests:?}"
        );
    }

    #[test]
    fn satisfiable_wheel_metadata_is_never_relaxed() {
        let mut builder = RestoreRequestBuilder::new(
            "typing-extensions",
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
        );
        builder.add_constraint(Constraint {
            specifiers: VersionSpecifiers::from_str("==4.12.2").unwrap(),
            provenance: Provenance::IndexWheelMetadata,
            source: "wheel `isaacsim==5.1.0.0` Requires-Dist \
                     `typing-extensions==4.12.2`"
                .to_string(),
            origin_id: test_origin("isaacsim-typing-extensions-pin", "==4.12.2"),
        });
        builder.add_constraint(Constraint {
            specifiers: VersionSpecifiers::from_str(">=4.10").unwrap(),
            provenance: Provenance::IndexWheelMetadata,
            source: "wheel `compatible==1.0.0` Requires-Dist `typing-extensions>=4.10`".to_string(),
            origin_id: test_origin("compatible-typing-extensions-floor", ">=4.10"),
        });

        let request = builder.finish().unwrap();
        assert!(
            render_specifiers(&request.specifiers).contains("==4.12.2"),
            "the original exact pin must remain intact: {}",
            request.specifiers
        );
        assert!(
            !request
                .specifiers
                .contains(&Version::from_str("4.12.3").unwrap()),
            "a satisfiable exact pin must not be widened"
        );
        assert!(
            request.relaxations.is_empty(),
            "a satisfiable raw intersection must not emit relaxation warnings"
        );
    }

    #[test]
    fn relax_none_keeps_unsatisfiable_index_metadata_strict() {
        let mut builder = RestoreRequestBuilder::new("strict-wheel-dep", RelaxPolicy::None);
        builder.add_constraint(Constraint {
            specifiers: VersionSpecifiers::from_str("==1.0.0").unwrap(),
            provenance: Provenance::IndexWheelMetadata,
            source: "wheel `strict-a==1.0.0` Requires-Dist `strict-wheel-dep==1.0.0`".to_string(),
            origin_id: test_origin("strict-a-wheel-pin", "==1.0.0"),
        });
        builder.add_constraint(Constraint {
            specifiers: VersionSpecifiers::from_str(">=2.0.0").unwrap(),
            provenance: Provenance::IndexWheelMetadata,
            source: "wheel `strict-b==1.0.0` Requires-Dist `strict-wheel-dep>=2.0.0`".to_string(),
            origin_id: test_origin("strict-b-wheel-floor", ">=2.0.0"),
        });

        let error = builder
            .finish()
            .expect_err("retread-relax=none must preserve the original conflict");
        assert!(
            error
                .downcast_ref::<crate::constraint::Conflict>()
                .is_some(),
            "{error:#}"
        );
        let message = format!("{error:#}");
        assert!(message.contains("strict-a==1.0.0"), "{message}");
        assert!(message.contains("strict-b==1.0.0"), "{message}");
    }

    #[test]
    fn numpy_and_cuda_abi_anchors_are_never_relaxed() {
        for (package, pinned, conflicting) in
            [("numpy", "==1.26.4", ">=2"), ("cuda", "==12.8", ">=13")]
        {
            assert!(
                crate::solve::is_abi_anchor(package),
                "fixture must exercise the shared ABI-anchor veto: {package}"
            );
            let constraints = vec![
                Constraint {
                    specifiers: VersionSpecifiers::from_str(pinned).unwrap(),
                    provenance: Provenance::IndexWheelMetadata,
                    source: format!("wheel `pinned==1` Requires-Dist `{package}{pinned}`"),
                    origin_id: test_origin(&format!("{package}-pinned-wheel"), pinned),
                },
                Constraint {
                    specifiers: VersionSpecifiers::from_str(conflicting).unwrap(),
                    provenance: Provenance::IndexWheelMetadata,
                    source: format!("wheel `newer==1` Requires-Dist `{package}{conflicting}`"),
                    origin_id: test_origin(&format!("{package}-newer-wheel"), conflicting),
                },
            ];

            assert!(
                matches!(
                    decide_relaxation(
                        &PypiKey::from_pypi(package),
                        &constraints,
                        RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
                        &SafetyContext::default(),
                    ),
                    RelaxDecision::Conflict(_)
                ),
                "{package} must not be widened or stripped"
            );

            let mut builder = RestoreRequestBuilder::new(
                package,
                RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
            );
            for constraint in constraints {
                builder.add_constraint(constraint);
            }
            let error = builder
                .finish()
                .expect_err("the original ABI-anchor conflict must fail closed");
            assert!(
                error
                    .downcast_ref::<crate::constraint::Conflict>()
                    .is_some(),
                "{package} lost the strict typed conflict: {error:#}"
            );
        }
    }

    #[test]
    fn equal_authority_conflict_suggests_pack_drop_and_override_menu() {
        let context = JointRouteDiagnosticContext {
            bundle: "robotics-output".to_string(),
            environments: vec!["robotics".to_string()],
            profiles: vec!["linux-64".to_string()],
            platform: "linux-64".to_string(),
            python: "3.11".to_string(),
        };
        let mut builder =
            RestoreRequestBuilder::new("numpy", RelaxPolicy::PatchThenMinorThenMajorThenLastResort);
        for (wheel, version, specifier) in [
            ("old-extension", "1.0.0", "==1.26.4"),
            ("new-extension", "2.0.0", ">=2,<3"),
        ] {
            let requirement = format!("numpy{specifier}");
            let specifiers = VersionSpecifiers::from_str(specifier).unwrap();
            builder.add_constraint(Constraint {
                specifiers: specifiers.clone(),
                provenance: Provenance::IndexWheelMetadata,
                source: format!("wheel `{wheel}=={version}` Requires-Dist `{requirement}`"),
                origin_id: wheel_requirement_origin_id(
                    wheel,
                    version,
                    "numpy",
                    &specifiers,
                    &requirement,
                ),
            });
        }

        let error = builder
            .finish_with_context_and_pack(
                Some(&context),
                Some("pypi-packs/robotics-pack/pixi.toml"),
            )
            .expect_err("cross-major NumPy conflict must fail closed");
        let conflict = error
            .downcast_ref::<Conflict>()
            .expect("the fail-closed decision must retain its typed conflict");
        let suggestion = conflict
            .suggestion()
            .expect("the typed conflict must carry a paste-ready suggestion");
        let message = conflict.to_string();

        for expected in [
            "in environment 'robotics' for bundle 'robotics-output' \
             (target profile 'linux-64', platform linux-64, python 3.11)",
            "wheel `old-extension==1.0.0` Requires-Dist `numpy==1.26.4`",
            "wheel `new-extension==2.0.0` Requires-Dist `numpy>=2,<3`",
        ] {
            assert!(
                message.contains(expected),
                "missing `{expected}`:\n{message}"
            );
        }
        assert!(message.contains("Suggested fix in pypi-packs/robotics-pack/pixi.toml:"));
        assert!(!message.contains("pypi-packs/robotics-output/pixi.toml"));
        assert!(suggestion.toml.contains("retread-drop-deps = [\"numpy\"]"));
        assert_eq!(suggestion.toml.matches("# Alternative ").count(), 2);
        assert!(
            suggestion.toml.contains("# numpy = \"==1.26.4\""),
            "{}",
            suggestion.toml
        );
        assert!(
            suggestion.toml.contains("# numpy = \">=2,<3\""),
            "{}",
            suggestion.toml
        );

        let parsed: toml::Value =
            toml::from_str(&suggestion.toml).expect("suggested table body must be valid TOML");
        assert_eq!(parsed["retread-drop-deps"][0].as_str(), Some("numpy"));
    }

    fn transitive_numpy_conflict_fixture() -> (
        JointRouteDiagnosticContext,
        Bundle,
        RetreadConfig,
        Vec<Constraint>,
    ) {
        let context = JointRouteDiagnosticContext {
            bundle: "robotics-output".to_string(),
            environments: vec!["robotics".to_string()],
            profiles: vec!["linux-64".to_string()],
            platform: "linux-64".to_string(),
            python: "3.11".to_string(),
        };
        let mut bundle = test_bundle(&[]);
        bundle.conda_name = context.bundle.clone();
        bundle.primary = test_wheel(
            "dex-retargeting",
            "dex-retargeting",
            "1.0.0",
            &["numpy<2", "pin>=2.6.20"],
        );
        // `pin` is deliberately absent from the materialized wheel set, as
        // it can be routed/no-emit. The retained uv graph must bridge that
        // missing node between the root and the conflicting cmeel wheel.
        bundle.extras = vec![test_wheel(
            "cmeel-boost",
            "cmeel-boost",
            "1.89.0",
            &["numpy>=2"],
        )];
        bundle.uv_dependency_graph = crate::uv_closure::UvDependencyGraph {
            edges: BTreeSet::from([
                crate::uv_closure::UvDependencyEdge {
                    parent: "dex-retargeting".to_string(),
                    child: "pin".to_string(),
                },
                crate::uv_closure::UvDependencyEdge {
                    parent: "pin".to_string(),
                    child: "cmeel-boost".to_string(),
                },
                crate::uv_closure::UvDependencyEdge {
                    parent: "cmeel-boost".to_string(),
                    child: "numpy".to_string(),
                },
            ]),
            selected_versions: BTreeMap::from([("pin".to_string(), "4.0.0".to_string())]),
            deps_from_root_requirements: BTreeMap::new(),
        };
        let mut config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": {
                "dex-retargeting": {
                    "version": "1.0.0",
                    "bundle": "robotics-output"
                }
            }
        }))
        .unwrap();
        config.pack_manifest_path = Some("pypi-packs/robotics-pack/pixi.toml".to_string());

        let mut observed = ObservedRequirements::new();
        for (specifiers, wheel, version, raw) in [
            ("<2", "dex-retargeting", "1.0.0", "numpy<2"),
            (">=2", "cmeel-boost", "1.89.0", "numpy>=2"),
        ] {
            observe_requirement(
                &mut observed,
                "numpy",
                &VersionSpecifiers::from_str(specifiers).unwrap(),
                wheel,
                version,
                raw,
                Provenance::IndexWheelMetadata,
            );
        }
        let constraints = observed
            .remove(&PypiKey::from_pypi("numpy"))
            .expect("the fixture must record both NumPy sides");
        (context, bundle, config, constraints)
    }

    fn deps_from_pin_root_conflict_fixture() -> (
        JointRouteDiagnosticContext,
        Bundle,
        RetreadConfig,
        Vec<Constraint>,
    ) {
        let context = JointRouteDiagnosticContext {
            bundle: "robotics-output".to_string(),
            environments: vec!["robotics".to_string()],
            profiles: vec!["linux-64".to_string()],
            platform: "linux-64".to_string(),
            python: "3.11".to_string(),
        };
        let mut bundle = test_bundle(&[]);
        bundle.conda_name = context.bundle.clone();
        bundle.primary = test_wheel("robotics-root", "robotics-root", "1.0.0", &[]);
        bundle.extras = vec![
            test_wheel("pin", "pin", "4.0.0", &["cmeel-boost>=1.89.0"]),
            test_wheel("cmeel-boost", "cmeel-boost", "1.89.0", &["numpy>=2"]),
        ];
        bundle.uv_dependency_graph = crate::uv_closure::UvDependencyGraph {
            edges: BTreeSet::from([
                crate::uv_closure::UvDependencyEdge {
                    parent: "pin".to_string(),
                    child: "cmeel-boost".to_string(),
                },
                crate::uv_closure::UvDependencyEdge {
                    parent: "cmeel-boost".to_string(),
                    child: "numpy".to_string(),
                },
            ]),
            selected_versions: BTreeMap::from([
                ("pin".to_string(), "4.0.0".to_string()),
                ("cmeel-boost".to_string(), "1.89.0".to_string()),
                ("numpy".to_string(), "2.0.0".to_string()),
            ]),
            deps_from_root_requirements: BTreeMap::from([(
                "pin".to_string(),
                vec![">=2.6.20".to_string()],
            )]),
        };
        let mut config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-deps-from": "requirements.txt",
            "retread-wheels": {
                "robotics-root": {
                    "version": "1.0.0",
                    "bundle": "robotics-output"
                }
            }
        }))
        .unwrap();
        config.pack_manifest_path = Some("pypi-packs/robotics-pack/pixi.toml".to_string());

        let baseline = Constraint {
            specifiers: VersionSpecifiers::from_str("<2").unwrap(),
            provenance: Provenance::UvRoot,
            source: "uv root requirement `numpy<2`".to_string(),
            origin_id: test_origin("numpy-root-cap", "<2"),
        };
        let mut observed = ObservedRequirements::new();
        observe_requirement(
            &mut observed,
            "numpy",
            &VersionSpecifiers::from_str(">=2").unwrap(),
            "cmeel-boost",
            "1.89.0",
            "numpy>=2",
            Provenance::IndexWheelMetadata,
        );
        let mut constraints = vec![baseline];
        constraints.extend(
            observed
                .remove(&PypiKey::from_pypi("numpy"))
                .expect("the fixture must record the cmeel NumPy floor"),
        );
        (context, bundle, config, constraints)
    }

    fn graph_conflict_suggestion(
        context: &JointRouteDiagnosticContext,
        bundle: &Bundle,
        config: &RetreadConfig,
        constraints: Vec<Constraint>,
    ) -> crate::constraint::ConflictSuggestion {
        let mut builder =
            RestoreRequestBuilder::new("numpy", RelaxPolicy::PatchThenMinorThenMajorThenLastResort);
        for constraint in constraints {
            builder.add_constraint(constraint);
        }
        let target = crate::pypi::WheelTarget::for_subdir(&context.python, &context.platform);
        let error = builder
            .finish_with_graph_context(Some(context), bundle, config, &target)
            .expect_err("the cross-major NumPy conflict must fail closed");
        error
            .downcast_ref::<Conflict>()
            .and_then(Conflict::suggestion)
            .cloned()
            .expect("the fail-closed conflict must retain a suggestion")
    }

    #[test]
    fn transitive_numpy_conflict_suggests_pin_root_not_leaf() {
        let (context, bundle, config, constraints) = transitive_numpy_conflict_fixture();
        let mut builder =
            RestoreRequestBuilder::new("numpy", RelaxPolicy::PatchThenMinorThenMajorThenLastResort);
        for constraint in constraints {
            builder.add_constraint(constraint);
        }
        let target = crate::pypi::WheelTarget::for_subdir(&context.python, &context.platform);
        let error = builder
            .finish_with_graph_context(Some(&context), &bundle, &config, &target)
            .expect_err("the cross-major NumPy conflict must fail closed");
        let conflict = error
            .downcast_ref::<Conflict>()
            .expect("the graph suggestion must retain the typed conflict");
        let suggestion = conflict
            .suggestion()
            .expect("the transitive conflict must carry a root suggestion");
        let message = conflict.to_string();

        for expected in [
            "in environment 'robotics' for bundle 'robotics-output' \
             (target profile 'linux-64', platform linux-64, python 3.11)",
            "wheel `dex-retargeting==1.0.0` Requires-Dist `numpy<2`",
            "wheel `cmeel-boost==1.89.0` Requires-Dist `numpy>=2`",
            "Suggested fix in pypi-packs/robotics-pack/pixi.toml:",
            "Edit the existing [package.build.config.retread-wheels] table.",
            "pin = { version = \"==2.6.20\", bundle = \"robotics-output\" }",
            "confirm the version exists on the configured index",
        ] {
            assert!(
                message.contains(expected),
                "missing `{expected}`:\n{message}"
            );
        }
        assert!(!suggestion.toml.contains("retread-drop-deps"));
        assert!(!suggestion.toml.contains("retread-overrides"));
        assert!(!suggestion.toml.contains("[\"numpy\"]"));

        let parsed: toml::Value = toml::from_str(&suggestion.toml)
            .expect("root table-body suggestion must be valid TOML");
        assert_eq!(parsed["pin"]["version"].as_str(), Some("==2.6.20"));
        assert_eq!(parsed["pin"]["bundle"].as_str(), Some("robotics-output"));
    }

    #[test]
    fn deps_from_pin_root_is_suggested_without_masking_parent_edge() {
        let (context, bundle, config, constraints) = deps_from_pin_root_conflict_fixture();
        assert!(
            !config.retread_wheels.contains_key("pin"),
            "`pin` must enter as a uv/deps-from root, not an invalid ranged wheel entry"
        );
        assert!(
            !config.deps_from.is_empty(),
            "the direct-root regression must model a production deps-from root"
        );
        assert_eq!(
            bundle
                .uv_dependency_graph
                .deps_from_root_requirements
                .get("pin")
                .and_then(|requirements| requirements.first())
                .map(String::as_str),
            Some(">=2.6.20")
        );
        assert_eq!(
            bundle.uv_dependency_graph.edges,
            BTreeSet::from([
                crate::uv_closure::UvDependencyEdge {
                    parent: "pin".to_string(),
                    child: "cmeel-boost".to_string(),
                },
                crate::uv_closure::UvDependencyEdge {
                    parent: "cmeel-boost".to_string(),
                    child: "numpy".to_string(),
                },
            ]),
            "the regression topology must not contain a masking parent of `pin`"
        );

        let suggestion = graph_conflict_suggestion(&context, &bundle, &config, constraints);
        let parsed: toml::Value = toml::from_str(&suggestion.toml).unwrap();

        assert_eq!(parsed["pin"].as_str(), Some("==2.6.20"));
        assert!(parsed.get("cmeel-boost").is_none(), "{}", suggestion.toml);
        assert!(
            suggestion
                .toml
                .contains("Deps-from root requirement `pin>=2.6.20`"),
            "{}",
            suggestion.toml
        );
        assert!(
            suggestion
                .toml
                .contains("Under [package.build.config.retread-overrides], add or update this key"),
            "{}",
            suggestion.toml
        );
        assert!(
            suggestion
                .toml
                .contains("override must replace the deps-from floor"),
            "{}",
            suggestion.toml
        );

        let merged_manifest = format!(
            r#"[package]
name = "robotics-pack"
version = "1.0.0"

[package.build.config]
retread-deps-from = "requirements.txt"

[package.build.config.retread-wheels]
robotics-root = {{ version = "1.0.0", bundle = "robotics-output" }}

[package.build.config.retread-overrides]
{}"#,
            suggestion.toml
        );
        let merged: toml::Value = toml::from_str(&merged_manifest)
            .expect("creating the override table once and merging the suggestion must parse");
        assert_eq!(
            merged["package"]["build"]["config"]["retread-overrides"]["pin"].as_str(),
            Some("==2.6.20")
        );
    }

    #[test]
    fn deps_from_root_pin_combines_effective_constraints() {
        let (context, mut bundle, config, constraints) = deps_from_pin_root_conflict_fixture();
        bundle
            .uv_dependency_graph
            .deps_from_root_requirements
            .insert(
                "pin".to_string(),
                vec![">=2.6.20".to_string(), ">=3.0".to_string()],
            );

        let suggestion = graph_conflict_suggestion(&context, &bundle, &config, constraints);
        let parsed: toml::Value = toml::from_str(&suggestion.toml).unwrap();

        assert_eq!(
            parsed["pin"].as_str(),
            Some("==3.0"),
            "additive effective constraints must raise the suggested floor"
        );
    }

    #[test]
    fn root_pin_carries_exact_failing_bundle_group() {
        let (context, mut bundle, mut config, constraints) = transitive_numpy_conflict_fixture();
        config
            .retread_wheels
            .get_mut("dex-retargeting")
            .expect("the configured root must exist")
            .bundle = Some("robotics_output".to_string());
        bundle.conda_name = "workspace-alias".to_string();
        assert!(!config.retread_wheels.contains_key("pin"));

        let suggestion = graph_conflict_suggestion(&context, &bundle, &config, constraints);
        let parsed: toml::Value = toml::from_str(&suggestion.toml).unwrap();

        assert_eq!(
            parsed["pin"]["bundle"].as_str(),
            Some("robotics_output"),
            "the edit must preserve the raw group spelling that produced the failing output"
        );
    }

    #[test]
    fn configured_roots_ignore_other_bundle_wheels_after_output_alias() {
        let (_context, mut bundle, mut config, _constraints) = transitive_numpy_conflict_fixture();
        config.retread_wheels.insert(
            "cmeel-boost".to_string(),
            serde_json::from_value(serde_json::json!({
                "version": "1.89.0",
                "bundle": "another-output"
            }))
            .unwrap(),
        );
        bundle.conda_name = "workspace-alias".to_string();

        let roots = configured_bundle_roots(&bundle, &config);

        assert!(roots.contains_key(&PypiKey::from_pypi("dex-retargeting")));
        assert!(
            !roots.contains_key(&PypiKey::from_pypi("cmeel-boost")),
            "a configured root from another output must not truncate this bundle's graph walk"
        );
    }

    #[test]
    fn final_emission_transitive_conflict_uses_root_suggestion() {
        let (context, bundle, config, _constraints) = transitive_numpy_conflict_fixture();
        let error = super::super::produce_output(
            &bundle,
            &config,
            rattler_conda_types::Platform::Linux64,
            &context.python,
            &[],
            None,
            None,
        )
        .expect_err("final emission must surface the structural NumPy conflict");
        let message = format!("{error:#}");

        assert!(
            message.contains("Suggested fix in pypi-packs/robotics-pack/pixi.toml:"),
            "{message}"
        );
        assert!(
            message.contains("pin = { version = \"==2.6.20\", bundle = \"robotics-output\" }"),
            "{message}"
        );
        assert!(!message.contains("retread-drop-deps = [\"numpy\"]"));
    }

    #[test]
    fn ambiguous_transitive_roots_fall_back_to_leaf_suggestion() {
        let (context, mut bundle, config, constraints) = transitive_numpy_conflict_fixture();
        bundle
            .primary
            .metadata
            .requires_dist
            .push("other-pin>=1".to_string());
        bundle.uv_dependency_graph.edges.extend([
            crate::uv_closure::UvDependencyEdge {
                parent: "dex-retargeting".to_string(),
                child: "other-pin".to_string(),
            },
            crate::uv_closure::UvDependencyEdge {
                parent: "other-pin".to_string(),
                child: "cmeel-boost".to_string(),
            },
        ]);
        bundle
            .uv_dependency_graph
            .selected_versions
            .insert("other-pin".to_string(), "2.0.0".to_string());

        let mut builder =
            RestoreRequestBuilder::new("numpy", RelaxPolicy::PatchThenMinorThenMajorThenLastResort);
        for constraint in constraints {
            builder.add_constraint(constraint);
        }
        let target = crate::pypi::WheelTarget::for_subdir(&context.python, &context.platform);
        let error = builder
            .finish_with_graph_context(Some(&context), &bundle, &config, &target)
            .expect_err("ambiguous graph must keep the fail-closed conflict");
        let suggestion = error
            .downcast_ref::<Conflict>()
            .and_then(Conflict::suggestion)
            .expect("fallback must remain actionable");

        assert!(
            suggestion.toml.contains("retread-drop-deps = [\"numpy\"]"),
            "{}",
            suggestion.toml
        );
        assert!(suggestion.toml.contains("retread-overrides"));
        assert!(!suggestion.toml.contains("retread-wheels"));
        let parsed = toml::from_str::<toml::Value>(&suggestion.toml)
            .expect("fallback table-body suggestion must remain valid TOML");
        assert_eq!(parsed["retread-drop-deps"][0].as_str(), Some("numpy"));
    }

    #[test]
    fn empty_dependency_graph_falls_back_to_leaf_suggestion() {
        let (context, mut bundle, config, constraints) = deps_from_pin_root_conflict_fixture();
        bundle.uv_dependency_graph.edges.clear();

        let suggestion = graph_conflict_suggestion(&context, &bundle, &config, constraints);

        assert!(
            suggestion.toml.contains("retread-drop-deps = [\"numpy\"]"),
            "{}",
            suggestion.toml
        );
        assert!(!suggestion.toml.contains("retread-wheels"));
    }

    #[test]
    fn cyclic_dependency_graph_falls_back_to_leaf_suggestion() {
        let (context, mut bundle, config, constraints) = transitive_numpy_conflict_fixture();
        bundle
            .uv_dependency_graph
            .edges
            .insert(crate::uv_closure::UvDependencyEdge {
                parent: "cmeel-boost".to_string(),
                child: "pin".to_string(),
            });

        let suggestion = graph_conflict_suggestion(&context, &bundle, &config, constraints);

        assert!(
            suggestion.toml.contains("retread-drop-deps = [\"numpy\"]"),
            "{}",
            suggestion.toml
        );
        assert!(!suggestion.toml.contains("retread-wheels"));
    }

    #[test]
    fn mapped_conda_abi_anchor_vetoes_non_anchor_pypi_restore() {
        let mut builder = RestoreRequestBuilder::new(
            "array-provider",
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
        );
        builder.add_safety_conda_name("numpy");
        for (label, specifiers) in [("old", "==1.26.4"), ("new", ">=2")] {
            builder.add_constraint(Constraint {
                specifiers: VersionSpecifiers::from_str(specifiers).unwrap(),
                provenance: Provenance::IndexWheelMetadata,
                source: format!("wheel `{label}==1` Requires-Dist `array-provider{specifiers}`"),
                origin_id: test_origin(&format!("mapped-anchor-{label}"), specifiers),
            });
        }

        let error = builder
            .finish()
            .expect_err("the mapped NumPy ABI anchor must retain the strict conflict");
        assert!(
            error
                .downcast_ref::<crate::constraint::Conflict>()
                .is_some(),
            "mapped anchor lost its typed strict conflict: {error:#}"
        );
    }

    #[test]
    fn two_non_metadata_hard_exact_pins_remain_a_scoped_conflict() {
        let context = JointRouteDiagnosticContext {
            bundle: "hard-pins-pack".to_string(),
            environments: vec!["hard-pins-env".to_string()],
            profiles: vec![],
            platform: "linux-64".to_string(),
            python: "3.11".to_string(),
        };
        let constraints = vec![
            Constraint {
                specifiers: VersionSpecifiers::from_str("==1.0.0").unwrap(),
                provenance: Provenance::UvRoot,
                source: "uv root requirement `hard-pins==1.0.0`".to_string(),
                origin_id: test_origin("hard-pins-uv-root", "==1.0.0"),
            },
            Constraint {
                specifiers: VersionSpecifiers::from_str("==2.0.0").unwrap(),
                provenance: Provenance::UvConstraint,
                source: "uv constraint `hard-pins==2.0.0`".to_string(),
                origin_id: test_origin("hard-pins-uv-constraint", "==2.0.0"),
            },
        ];
        assert!(
            matches!(
                decide_relaxation(
                    &PypiKey::from_pypi("hard-pins"),
                    &constraints,
                    RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
                    &SafetyContext::default(),
                ),
                RelaxDecision::Conflict(_)
            ),
            "hard non-metadata pins must produce no relaxation record"
        );

        let mut builder = RestoreRequestBuilder::new(
            "hard-pins",
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
        );
        for constraint in constraints {
            builder.add_constraint(constraint);
        }
        let error = builder
            .finish_with_context(Some(&context))
            .expect_err("two untouched hard exact pins must remain irreconcilable");
        assert!(
            error
                .downcast_ref::<crate::constraint::Conflict>()
                .is_some(),
            "the original typed conflict must survive: {error:#}"
        );
        let message = format!("{error:#}");
        for expected in [
            "dependency conflict in environment 'hard-pins-env' for bundle 'hard-pins-pack'",
            "`hard-pins` requirements are mutually unsatisfiable",
            "uv root requirement `hard-pins==1.0.0`",
            "uv constraint `hard-pins==2.0.0`",
        ] {
            assert!(
                message.contains(expected),
                "missing `{expected}`:\n{message}"
            );
        }
    }

    #[tokio::test]
    async fn no_workspace_fact_holosoma_conflict_still_fail_closes() {
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Err(anyhow!("semantic conflict must fail before fetch")) }
            }
        };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let mut config = test_config();
        config.relax = RelaxPolicy::PatchThenMinorThenMajorThenLastResort;
        let mut bundle = holosoma_numpy_conflict_bundle();
        let unrelated =
            single_provider_facts("packaging", &[("holosoma", Some("26.2"), Some(">=24"))]);
        bundle.workspace_conda_versions = unrelated.common_selected_versions;
        bundle.workspace_conda_provider_facts = unrelated.provider_facts;
        let favor_lock_prefs = BTreeMap::from([("numpy".to_string(), "1.26.4".to_string())]);
        bundle.apply_workspace_conda_fact_ownership(
            &config,
            &config.name_map,
            &BTreeSet::new(),
            &BTreeSet::new(),
        );
        assert!(
            !bundle.auto_dropped.contains("numpy"),
            "an unrelated conda provider cannot own numpy"
        );

        let error = auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &config,
            None,
            Some(&favor_lock_prefs),
            None,
            &validated_probe,
            &reject_numpy_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .unwrap_err();

        assert!(
            error
                .downcast_ref::<crate::constraint::Conflict>()
                .is_some(),
            "legacy reconstruction must preserve the typed P3 conflict: {error:#}"
        );
        let message = format!("{error:#}");
        assert!(
            message.contains("`numpy` requirements are mutually unsatisfiable"),
            "{message}"
        );
        assert!(message.contains("mutually unsatisfiable"), "{message}");
        for source in [
            "wheel `holosoma==0.0.1` Requires-Dist `numpy<2,>=1.23.5`",
            "wheel `holosoma-retargeting==0.1.0` Requires-Dist `numpy>=2.3.5,<2.4`",
            "wheel `mujoco-warp==0.0.1` Requires-Dist `numpy`",
            "wheel `smplx==0.1.28` Requires-Dist `numpy (>=1.16.2)`",
        ] {
            assert!(
                message.contains(source),
                "missing `{source}` in:\n{message}"
            );
        }
        assert!(
            !bundle.auto_dropped.contains("numpy"),
            "the genuine wheel conflict must remain unowned"
        );
        assert_eq!(
            fetch_calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "semantic conflicts must fail before any index request"
        );
    }

    #[tokio::test]
    async fn joint_validation_keeps_cosolvable_metadata_route_on_conda() {
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Err(anyhow!("accepted conda route must not fetch PyPI")) }
            }
        };
        let cosolvable = |_| async { crate::uv_closure::CoInstallVerdict::Sat };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let mut bundle = test_bundle(&["numpy>=1.26,<3"]);

        auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &test_config(),
            None,
            None,
            None,
            &validated_probe,
            &cosolvable,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .unwrap();

        assert!(bundle.auto_dropped.is_empty());
        assert_eq!(fetch_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        let emitted =
            super::super::emitted_bundle_route_specs(&bundle, &test_config(), &target).unwrap();
        assert!(
            emitted
                .iter()
                .any(|route| route.conda_name.key().as_str() == "numpy"),
            "a jointly solvable metadata route must remain on conda: {emitted:?}"
        );
    }

    #[tokio::test]
    async fn source_built_route_survives_unsatisfiable_fixed_baseline() {
        const NAME: &str = "serveable-transitive";
        const RANGE: &str = ">=4.0.1,<4.1";

        let probe_inputs = Arc::new(Mutex::new(Vec::<Vec<(String, String)>>::new()));
        let probe = {
            let probe_inputs = Arc::clone(&probe_inputs);
            move |pairs: Vec<(String, String)>| {
                probe_inputs.lock().unwrap().push(pairs.clone());
                async move { validated_probe(pairs).await }
            }
        };
        let solve_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let co_solve = {
            let solve_calls = Arc::clone(&solve_calls);
            move |_routes: Vec<crate::uv_closure::CondaRouteSpec>| {
                solve_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async {
                    crate::uv_closure::CoInstallVerdict::Unsat(vec![
                        "unrelated fixed baseline is unsatisfiable".to_string(),
                    ])
                }
            }
        };
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Ok(test_wheel(NAME, NAME, "4.0.1", &[])) }
            }
        };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let config = test_config();
        let mut bundle = test_bundle(&[&format!("{NAME}{RANGE}")]);
        bundle.primary.metadata.name = "source-built-pack".to_string();
        bundle.primary.metadata_provenance = Provenance::SourceBuiltRelaxed;

        auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &config,
            None,
            None,
            None,
            &probe,
            &co_solve,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .unwrap();

        let probed = probe_inputs
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(probed, vec![(NAME.to_string(), RANGE.to_string())]);
        assert_eq!(
            solve_calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "Rule 2 must test the complete route set and the fixed baseline"
        );
        assert_eq!(
            fetch_calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "a conda-serveable source-built transitive must never reach the wheel fetch path"
        );
        assert!(bundle.extras.is_empty());
        assert!(bundle.probe_decisions.iter().any(|decision| {
            decision.stage == "auto_bundle_loose"
                && decision.pypi_name == NAME
                && decision.routing_decision == "short-circuit"
        }));

        let emitted = super::super::emitted_bundle_route_specs(&bundle, &config, &target).unwrap();
        let emitted_spec = emitted
            .iter()
            .find(|route| route.conda_name.key().as_str() == NAME)
            .map(|route| route.spec.as_str())
            .expect("the serveable transitive must be emitted on conda");
        assert_eq!(emitted_spec, RANGE);

        let output = super::super::produce_output(
            &bundle,
            &config,
            rattler_conda_types::Platform::Linux64,
            &target.python_version,
            &[],
            None,
            None,
        )
        .unwrap();
        let output_spec = output
            .run_dependencies
            .depends
            .iter()
            .find(|dependency| dependency.name.as_str() == NAME)
            .map(|dependency| super::super::audit_report::format_packagespec(&dependency.spec))
            .expect("recipe output must contain the serveable transitive");
        assert_eq!(output_spec, RANGE);
    }

    #[tokio::test]
    async fn source_provider_shadow_restores_when_fixed_baseline_is_unsatisfiable() {
        const NAME: &str = "tensordict";
        const RANGE: &str = ">=0.8,<0.9";

        let co_solve_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let co_solve = {
            let calls = Arc::clone(&co_solve_calls);
            move |_routes: Vec<crate::uv_closure::CondaRouteSpec>| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async {
                    crate::uv_closure::CoInstallVerdict::Unsat(vec![
                        "unrelated fixed baseline is unsatisfiable".to_string(),
                    ])
                }
            }
        };
        let provider_checks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let probe_order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let prewarm_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let prewarm = {
            let calls = Arc::clone(&prewarm_calls);
            let order = Arc::clone(&probe_order);
            move |routes: Vec<crate::uv_closure::CondaRouteSpec>| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                order.lock().unwrap().push("prewarm");
                assert!(
                    routes
                        .iter()
                        .any(|route| route.conda_name.key().as_str() == NAME),
                    "the extraction prewarm must include every mutable route root",
                );
                async {}
            }
        };
        let provider_check = {
            let calls = Arc::clone(&provider_checks);
            let order = Arc::clone(&probe_order);
            move |route: crate::uv_closure::CondaRouteSpec| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                order.lock().unwrap().push("provider");
                async move {
                    assert_eq!(route.conda_name.key().as_str(), NAME);
                    crate::uv_closure::CoInstallVerdict::Unsat(vec![
                        "standalone tensordict route selects workspace-owned pytorch".to_string(),
                    ])
                }
            }
        };
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let calls = Arc::clone(&fetch_calls);
            move |request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                assert_eq!(request.pypi_name, NAME);
                async { Ok(test_wheel(NAME, NAME, "0.8.3", &[])) }
            }
        };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let config = test_config();
        let mut bundle = test_bundle(&[&format!("{NAME}{RANGE}")]);
        bundle.primary.metadata.name = "gr00t".to_string();
        bundle.primary.metadata_provenance = Provenance::SourceBuiltRelaxed;

        auto_bundle_transitives_with_route_precheck_and_prewarm(
            AutoBundleInputs {
                bundle: &mut bundle,
                indexes: &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
                target: &target,
                workspace_scope: None,
                config: &config,
                locked_closure: None,
                favor_lock_prefs: None,
                uv_closure_wheels: None,
                channels_consulted: &["conda-forge/linux-64".to_string()],
                probe_parallelism: 3,
                uv_reresolve: &UvReresolveContext::default(),
            },
            &validated_probe,
            &co_solve,
            &prewarm,
            &provider_check,
            &fetch,
        )
        .await
        .unwrap();

        assert_eq!(provider_checks.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(prewarm_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(
            probe_order.lock().unwrap().as_slice(),
            ["prewarm", "provider"],
            "record extraction must finish before standalone provider probes begin",
        );
        assert_eq!(
            co_solve_calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "record prewarming must not add a discarded resolvo solve before provider rejection",
        );
        assert_eq!(fetch_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(bundle.extras.iter().any(|wheel| wheel.pypi_name == NAME));
        assert!(
            super::super::emitted_bundle_route_specs(&bundle, &config, &target)
                .unwrap()
                .iter()
                .all(|route| route.conda_name.key().as_str() != NAME),
            "positive provider-shadow evidence must override only the source-route baseline fail-open",
        );
    }

    #[tokio::test]
    async fn source_built_unsolvable_transitive_restores_pypi_wheel() {
        const NAME: &str = "source-conflicting-transitive";
        const RANGE: &str = ">=3,<4";

        let co_solve = |routes: Vec<crate::uv_closure::CondaRouteSpec>| async move {
            if routes
                .iter()
                .any(|route| route.conda_name.key().as_str() == NAME)
            {
                crate::uv_closure::CoInstallVerdict::Unsat(vec![format!(
                    "{NAME} conflicts with a co-activated sibling constraint"
                )])
            } else {
                crate::uv_closure::CoInstallVerdict::Sat
            }
        };
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                assert_eq!(request.pypi_name, NAME);
                assert_eq!(
                    request.specifiers,
                    VersionSpecifiers::from_str(RANGE).unwrap()
                );
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Ok(test_wheel(NAME, NAME, "3.2.1", &[])) }
            }
        };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let config = test_config();
        let mut bundle = test_bundle(&[&format!("{NAME}{RANGE}")]);
        bundle.primary.metadata.name = "source-built-pack".to_string();
        bundle.primary.metadata_provenance = Provenance::SourceBuiltRelaxed;

        auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &config,
            None,
            None,
            None,
            &validated_probe,
            &co_solve,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .unwrap();

        assert_eq!(fetch_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(bundle.extras.iter().any(|wheel| wheel.pypi_name == NAME));
        let emitted = super::super::emitted_bundle_route_specs(&bundle, &config, &target).unwrap();
        assert!(
            emitted
                .iter()
                .all(|route| route.conda_name.key().as_str() != NAME),
            "the rejected source-root dependency must be provided by its bundled wheel: {emitted:?}"
        );
    }

    #[test]
    fn deps_from_source_route_group_is_fully_mutable() {
        const NAME: &str = "tensordict";

        let mut bundle = test_bundle(&[NAME]);
        bundle.primary.metadata.name = "gr00t".to_string();
        bundle.primary.metadata_provenance = Provenance::SourceBuiltRelaxed;
        let mut route = pillow_auto_route("0.8.3");
        route.route.pypi_name = NAME.to_string();
        route.route.conda_name = NAME.to_string();
        route.route.pypi_version = "0.8.3".to_string();
        route.route.conda_version = "0.8.3".to_string();
        route.provenance = Provenance::DepsFromRelaxed;
        route.route.input_requirements = vec![crate::uv_closure::AutoRouteInputRequirement {
            specifiers: String::new(),
            source: "retread-deps-from root `tensordict`".to_string(),
            provenance: Provenance::DepsFromRelaxed,
            role: crate::uv_closure::AutoRouteInputRole::Requirement,
        }];
        bundle.auto_routed.push(route);

        let mut metadata_routes = ProvisionalMetadataRoutes::new();
        record_metadata_route(
            &mut metadata_routes,
            NAME.to_string(),
            NAME.to_string(),
            Some("0.8.3".to_string()),
        );
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let config = test_config();
        let assembly =
            super::super::emitted_bundle_route_assembly(&bundle, &config, &target).unwrap();
        let support_index = EmissionSupportIndex::new(assembly.supports_by_conda);

        assert!(
            route_group_is_fully_mutable(
                &bundle,
                &metadata_routes,
                &support_index,
                NAME,
                &config,
                &target,
            )
            .unwrap(),
            "a deps-from route duplicated by source-wheel metadata must disappear when its PyPI wheel is restored",
        );
    }

    #[test]
    fn route_group_mutability_index_matches_counterfactual_rerender() {
        const ROUTE_COUNT: usize = 64;

        let mut bundle = test_bundle(&[]);
        bundle.primary.metadata.requires_dist = (0..ROUTE_COUNT)
            .map(|index| format!("probe-dep-{index:02}>=1"))
            .collect();
        for index in 0..ROUTE_COUNT {
            let name = format!("probe-dep-{index:02}");
            let mut route = pillow_auto_route("1.2.3");
            route.route.pypi_name = name.clone();
            route.route.conda_name = name;
            route.route.pypi_version = "1.2.3".to_string();
            route.route.conda_version = "1.2.3".to_string();
            bundle.auto_routed.push(route);
        }

        let metadata_routes = ProvisionalMetadataRoutes::new();
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let config = test_config();
        // The production path assembles this support index once for the whole
        // bundle. Every candidate lookup below reuses that immutable result.
        let assembly =
            super::super::emitted_bundle_route_assembly(&bundle, &config, &target).unwrap();
        let support_index = EmissionSupportIndex::new(assembly.supports_by_conda);

        for index in 0..ROUTE_COUNT {
            let conda_name = format!("probe-dep-{index:02}");
            let restored = restored_route_pypi_names(&bundle, &metadata_routes, &conda_name);
            assert!(
                support_index_proves_isolated_route_removal(&support_index, &conda_name, &restored,),
                "ordinary route `{conda_name}` should use the shared support index",
            );
            let indexed = route_group_is_fully_mutable(
                &bundle,
                &metadata_routes,
                &support_index,
                &conda_name,
                &config,
                &target,
            )
            .unwrap();
            let counterfactual = route_group_is_fully_mutable_counterfactual(
                &bundle,
                &conda_name,
                &config,
                &target,
                restored,
            )
            .unwrap();
            assert_eq!(
                indexed, counterfactual,
                "support-index classification changed `{conda_name}`",
            );
        }

        // A route sharing its conda target with an unrelated wheel edge is
        // deliberately ambiguous. The index declines the shortcut and the
        // exact renderer preserves the fixed edge.
        let mut mixed = test_bundle(&["shared-fixed>=1"]);
        let mut route = pillow_auto_route("1.2.3");
        route.route.pypi_name = "provisional-owner".to_string();
        route.route.conda_name = "shared-fixed".to_string();
        route.route.pypi_version = "1.2.3".to_string();
        route.route.conda_version = "1.2.3".to_string();
        mixed.auto_routed.push(route);
        let mixed_assembly =
            super::super::emitted_bundle_route_assembly(&mixed, &config, &target).unwrap();
        let mixed_support_index = EmissionSupportIndex::new(mixed_assembly.supports_by_conda);
        let restored = restored_route_pypi_names(&mixed, &metadata_routes, "shared-fixed");
        assert!(
            !support_index_proves_isolated_route_removal(
                &mixed_support_index,
                "shared-fixed",
                &restored,
            ),
            "a fixed wheel edge must force the exact fallback",
        );
        let indexed = route_group_is_fully_mutable(
            &mixed,
            &metadata_routes,
            &mixed_support_index,
            "shared-fixed",
            &config,
            &target,
        )
        .unwrap();
        let counterfactual = route_group_is_fully_mutable_counterfactual(
            &mixed,
            "shared-fixed",
            &config,
            &target,
            restored,
        )
        .unwrap();
        assert_eq!(indexed, counterfactual);
        assert!(!indexed, "the unrelated fixed wheel edge must remain");
    }

    #[tokio::test]
    async fn rule2_linear_and_bisect_emit_byte_identical_outputs_and_warnings() {
        const ROUTE_COUNT: usize = 32;

        let names = std::iter::once("numpy".to_string())
            .chain((1..ROUTE_COUNT).map(|index| format!("emission-dep-{index:02}")))
            .collect::<Vec<_>>();
        let mut base = test_bundle(&[]);
        base.primary.metadata.requires_dist = names
            .iter()
            .map(|name| {
                if name == "numpy" {
                    "numpy>=2.1,<2.2".to_string()
                } else {
                    format!("{name}>=1")
                }
            })
            .collect();
        for name in &names {
            let version = if name == "numpy" { "2.1.0" } else { "1.2.3" };
            let mut route = pillow_auto_route(version);
            route.route.pypi_name = name.clone();
            route.route.conda_name = name.clone();
            route.route.pypi_version = version.to_string();
            route.route.conda_version = version.to_string();
            base.auto_routed.push(route);
        }

        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let config = test_config();
        let candidate_names = names.iter().cloned().collect::<BTreeSet<_>>();
        let candidates = super::super::emitted_bundle_route_specs(&base, &config, &target)
            .unwrap()
            .into_iter()
            .filter(|route| candidate_names.contains(route.conda_name.key().as_str()))
            .collect::<Vec<_>>();
        assert_eq!(candidates.len(), ROUTE_COUNT);
        let conflict_names =
            BTreeSet::from(["emission-dep-07".to_string(), "emission-dep-25".to_string()]);
        let oracle = |routes: Vec<crate::uv_closure::CondaRouteSpec>| {
            let conflict_names = conflict_names.clone();
            async move {
                if routes
                    .iter()
                    .any(|route| conflict_names.contains(route.conda_name.key().as_str()))
                {
                    crate::uv_closure::CoInstallVerdict::Unsat(vec![
                        "opaque emission fixture conflict".to_string(),
                    ])
                } else {
                    crate::uv_closure::CoInstallVerdict::Sat
                }
            }
        };
        let linear = crate::uv_closure::select_jointly_solvable_routes_linear_reference(
            Vec::new(),
            candidates.clone(),
            &oracle,
        )
        .await
        .expect("fixture baseline is satisfiable");
        let bisect = crate::uv_closure::select_jointly_solvable_routes_bisect_for_test(
            Vec::new(),
            candidates,
            &oracle,
            4,
        )
        .await
        .expect("fixture baseline is satisfiable");
        assert_eq!(bisect, linear);

        let emit_bytes = |selection: &crate::uv_closure::JointRouteSelection| {
            let rejected = selection
                .rejected
                .iter()
                .map(|route| route.conda_name.key().into_string())
                .collect::<BTreeSet<_>>();
            let mut finalized = base.clone();
            let restored = finalized
                .auto_routed
                .iter()
                .filter(|route| rejected.contains(&canonical_conda_name(&route.route.conda_name)))
                .map(|route| {
                    (
                        route.route.pypi_name.clone(),
                        route.route.pypi_version.clone(),
                    )
                })
                .collect::<Vec<_>>();
            finalized
                .auto_routed
                .retain(|route| !rejected.contains(&canonical_conda_name(&route.route.conda_name)));
            finalized.extras.extend(
                restored
                    .iter()
                    .map(|(name, version)| test_wheel(name, name, version, &[])),
            );
            let (output, warnings) = super::super::produce_output_pending_relaxations(
                &finalized,
                &config,
                rattler_conda_types::Platform::Linux64,
                &target.python_version,
                &[],
                None,
                None,
            )
            .unwrap();
            let warnings = warnings.iter().map(ToString::to_string).collect::<Vec<_>>();
            assert!(
                !warnings.is_empty(),
                "the accepted NumPy ABI anchor must exercise warning-byte parity",
            );
            serde_json::to_vec(&(output, warnings)).unwrap()
        };

        assert_eq!(
            emit_bytes(&bisect),
            emit_bytes(&linear),
            "linear and bisect decisions must produce byte-identical output and warnings",
        );
    }

    #[tokio::test]
    async fn workspace_fact_owns_source_built_advisory_dependency() {
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Err(anyhow!("a softened conda route must not fetch PyPI")) }
            }
        };
        let solve_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let co_solve = {
            let solve_calls = Arc::clone(&solve_calls);
            move |_routes: Vec<crate::uv_closure::CondaRouteSpec>| {
                solve_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { crate::uv_closure::CoInstallVerdict::Sat }
            }
        };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let config = test_config();
        let mut bundle = test_bundle(&["starlette>=0.49.1,<0.50"]);
        bundle.primary.metadata.name = "isaaclab".to_string();
        bundle.primary.metadata_provenance = Provenance::SourceBuiltRelaxed;
        let facts = single_provider_facts(
            "starlette",
            &[("isaaclab", Some("0.45.3"), Some("==0.45.3"))],
        );
        bundle.workspace_conda_versions = facts.common_selected_versions;
        bundle.workspace_conda_provider_facts = facts.provider_facts;
        bundle.apply_workspace_conda_fact_ownership(
            &config,
            &config.name_map,
            &BTreeSet::new(),
            &BTreeSet::new(),
        );

        assert_eq!(
            bundle.auto_dropped,
            HashSet::from(["starlette".to_string()]),
            "the shared workspace conda fact must enter Rule-1 drop ownership"
        );

        auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &config,
            None,
            None,
            None,
            &validated_probe,
            &co_solve,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .unwrap();

        assert_eq!(fetch_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(
            solve_calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "source-built metadata routes must remain outside Rule 2"
        );

        let emitted = super::super::emitted_bundle_route_specs(&bundle, &config, &target).unwrap();
        assert!(
            emitted
                .iter()
                .all(|route| route.conda_name.key().as_str() != "starlette"),
            "the generated pack must not re-emit a workspace-owned dependency: {emitted:?}"
        );

        let output = super::super::produce_output(
            &bundle,
            &config,
            rattler_conda_types::Platform::Linux64,
            &target.python_version,
            &[],
            None,
            None,
        )
        .unwrap();
        assert!(
            output
                .run_dependencies
                .depends
                .iter()
                .all(|dependency| dependency.name.as_str() != "starlette"),
            "recipe output must leave the workspace-owned conda fact to the workspace"
        );
    }

    #[tokio::test]
    async fn joint_unroute_deduplicates_repeated_requires_dist_clauses() {
        let ordinary_aliases = [">=11.0,>=11,<11.1.0,<11.1"]
            .into_iter()
            .map(|raw| Constraint {
                specifiers: VersionSpecifiers::from_str(raw).unwrap(),
                source: format!("test source `{raw}`"),
                provenance: Provenance::IndexWheelMetadata,
                origin_id: test_origin("ordinary-aliases", raw),
            })
            .collect::<Vec<_>>();
        let ordinary =
            finalize(&PypiKey::from_pypi("ordinary-aliases"), &ordinary_aliases).unwrap();
        assert_eq!(
            ordinary.len(),
            2,
            "ordinary trailing-zero aliases must collapse: {ordinary}"
        );
        let length_sensitive = ["~=1.0", "~=1.0.0"]
            .into_iter()
            .map(|raw| Constraint {
                specifiers: VersionSpecifiers::from_str(raw).unwrap(),
                source: format!("test source `{raw}`"),
                provenance: Provenance::IndexWheelMetadata,
                origin_id: test_origin("length-sensitive-aliases", raw),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            finalize(&PypiKey::from_pypi("length-sensitive"), &length_sensitive)
                .unwrap()
                .len(),
            2,
            "semantic length-sensitive clauses must not be deduplicated"
        );
        let arbitrary_exact = ["===1", "===1.0"]
            .into_iter()
            .map(|raw| Constraint {
                specifiers: VersionSpecifiers::from_str(raw).unwrap(),
                source: format!("test source `{raw}`"),
                provenance: Provenance::IndexWheelMetadata,
                origin_id: test_origin("arbitrary-exact-aliases", raw),
            })
            .collect::<Vec<_>>();
        assert!(
            finalize(&PypiKey::from_pypi("arbitrary-exact"), &arbitrary_exact)
                .unwrap_err()
                .to_string()
                .contains("mutually unsatisfiable"),
            "arbitrary equality must preserve trailing-zero spelling"
        );

        let requests = Arc::new(Mutex::new(Vec::<PypiFetchRequest>::new()));
        let fetch = {
            let requests = Arc::clone(&requests);
            move |request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                let requests = Arc::clone(&requests);
                async move {
                    requests.lock().unwrap().push(request.clone());
                    Ok(test_wheel(
                        &request.bundle_name,
                        &request.pypi_name,
                        "11.0.0",
                        &[],
                    ))
                }
            }
        };
        let mut bundle = test_bundle(&["pillow>=11,<11.1"]);
        bundle.extras.push(test_wheel(
            "second-root",
            "second-root",
            "2.0.0",
            &["pillow>=11,<11.1"],
        ));
        let target = crate::pypi::WheelTarget::for_subdir("3.10", "linux-64");

        let outcome = auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &test_config(),
            None,
            None,
            None,
            &validated_probe,
            &reject_every_mutable_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext {
                // N27-RETREAD-142. This fixture is about the RESTORE's request
                // construction (one deduped fetch for `pillow`), so it has to
                // stay on the restore path. It used to reach that path through
                // `from_env_value(None)` -- i.e. by the mode being default-OFF.
                // Now that the mode defaults ON, what keeps a bundle on this
                // path is having no uv closure to hand the name back to. Every
                // assertion below is unchanged.
                mode: UvReresolveMode::default(),
                uv_backed: false,
                keep_pypi: BTreeSet::new(),
            },
        )
        .await
        .unwrap();

        assert!(matches!(outcome, AutoBundleOutcome::Complete { .. }));

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].pypi_name, "pillow");
        assert_eq!(
            requests[0].specifiers,
            VersionSpecifiers::from_str(">=11,<11.1").unwrap()
        );
        let rendered = requests[0].specifiers.to_string();
        assert_eq!(rendered.matches(">=11").count(), 1, "{rendered}");
        assert_eq!(rendered.matches("<11.1").count(), 1, "{rendered}");
    }

    fn observed(specifiers: &str, source: &str, provenance: Provenance) -> Constraint {
        Constraint {
            specifiers: VersionSpecifiers::from_str(specifiers).unwrap(),
            source: source.to_string(),
            origin_id: test_origin(
                &format!("observed-{}", provenance_origin_label(&provenance)),
                specifiers,
            ),
            provenance,
        }
    }

    /// The IsaacLab regression (integration relock relocks against the imprint
    /// workspace): a source-built wheel (`isaaclab`, git `[retread-wheels]`
    /// entry) declares `starlette==0.49.1`, which retread relaxes to
    /// `>=0.49.1,<0.50`. A workspace-authoritative `fastapi==0.115.7` caps
    /// `starlette<0.46`. The intersection is empty, but the advisory floor is
    /// the ONLY reason: conda-as-truth drops it so `starlette` resolves to a
    /// `<0.46` version (0.45.3 on the real relock) instead of failing the pack.
    #[test]
    fn advisory_source_built_floor_yields_to_authoritative_cap() {
        let observations = vec![
            // Source-built IsaacLab wheel -> advisory (retread-relaxed floor).
            observed(
                ">=0.49.1,<0.50",
                "wheel `isaaclab==0.54.2` Requires-Dist `starlette>=0.49.1,<0.50`",
                Provenance::SourceBuiltRelaxed,
            ),
            // Index wheel `fastapi==0.115.7` -> authoritative.
            observed(
                ">=0.40.0,<0.46.0",
                "wheel `fastapi==0.115.7` Requires-Dist `starlette<0.46.0,>=0.40.0`",
                Provenance::IndexWheelMetadata,
            ),
        ];
        let softened =
            finalize(&PypiKey::from_pypi("starlette"), &observations).expect("floor must soften");
        // The authoritative cap is honored; a `<0.46` version resolves.
        assert!(
            softened.contains(&Version::from_str("0.45.3").unwrap()),
            "0.45.3 must satisfy the softened spec: {softened}"
        );
        // The advisory floor is gone: no `>=0.49` version can satisfy it.
        assert!(
            !softened.contains(&Version::from_str("0.49.1").unwrap()),
            "advisory floor must not survive: {softened}"
        );
    }

    /// Softening is scoped: a conflict among AUTHORITATIVE constraints (no
    /// source-built advisory involved) still fails loudly -- the user must
    /// resolve a genuine incompatibility, not have it silently softened. This
    /// is the `holosoma`/`numpy` behavior the legacy restore finalizer protects.
    #[test]
    fn authoritative_only_conflict_still_errors() {
        let observations = vec![
            observed(
                ">=0.49.1,<0.50",
                "wheel `a==1` Requires-Dist `x>=0.49.1`",
                Provenance::IndexWheelMetadata,
            ),
            observed(
                ">=0.40.0,<0.46.0",
                "wheel `b==1` Requires-Dist `x<0.46`",
                Provenance::UvConstraint,
            ),
        ];
        let error =
            finalize(&PypiKey::from_pypi("x"), &observations).expect_err("must stay unsatisfiable");
        assert!(error.to_string().contains("mutually unsatisfiable"));
    }

    /// Softening needs an authoritative anchor. Two advisory (both source-built)
    /// pins that mutually conflict are NOT silently reconciled: without an
    /// authoritative constraint to defer to there is no principled winner.
    #[test]
    fn advisory_only_conflict_still_errors() {
        let observations = vec![
            observed(
                ">=2.3.5,<2.4",
                "wheel `pkg-a==1` Requires-Dist `x>=2.3.5,<2.4`",
                Provenance::SourceBuiltRelaxed,
            ),
            observed(
                "<2,>=1.23.5",
                "wheel `pkg-b==1` Requires-Dist `x<2,>=1.23.5`",
                Provenance::DepsFromRelaxed,
            ),
        ];
        let error = finalize(&PypiKey::from_pypi("x"), &observations)
            .expect_err("advisory-only conflict must stay unsatisfiable");
        assert!(error.to_string().contains("mutually unsatisfiable"));
    }

    #[tokio::test]
    async fn joint_unroute_scopes_pillow_to_precise_pack_consumer() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-unroute-scope-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(tmp.join("hover-pack")).unwrap();
        std::fs::write(
            tmp.join("pixi.toml"),
            r#"
[environments]
legacy = { features = ["legacy"], no-default-feature = true }
wide-a = { features = ["wide-a"], no-default-feature = true }
hover = { features = ["hover"], no-default-feature = true }
wide-b = { features = ["wide-b"], no-default-feature = true }

[feature.legacy.dependencies]
pillow = ">=9.4,<10"

[feature.wide-a.dependencies]
pillow = ">=10,<13"

[feature.hover.dependencies]
hover-pack = { path = "./hover-pack" }
pillow = ">=11.0.0,<12"

[feature.wide-b.dependencies]
pillow = ">=10,<13"
"#,
        )
        .unwrap();
        let manifest = crate::workspace::WorkspaceManifest::load(&tmp).unwrap();
        let target = crate::pypi::WheelTarget::for_subdir("3.10", "linux-64");
        let resolution_target =
            crate::pypi::ResolutionTarget::from_wheel_target(target.clone(), None);
        let context = super::super::CondaCoSolveContext::new(
            Some(&manifest),
            Some(&tmp),
            &tmp.join("hover-pack"),
            &resolution_target,
            &[],
            "hover-pack",
            &BTreeSet::new(),
            &NameMap::new(),
        );
        assert_eq!(
            context.workspace_deps.get(&CondaName::new("pillow")),
            Some(&vec![">=11.0.0,<12".to_string()]),
            "only the environment that consumes hover-pack may constrain its route validation"
        );
        let solve_calls = Arc::new(Mutex::new(
            Vec::<Vec<crate::uv_closure::CondaRouteSpec>>::new(),
        ));
        let recorded_solve_calls = Arc::clone(&solve_calls);
        let scoped_reject = move |routes: Vec<crate::uv_closure::CondaRouteSpec>| {
            let solve_calls = Arc::clone(&recorded_solve_calls);
            async move {
                solve_calls.lock().unwrap().push(routes.clone());
                if routes.is_empty() {
                    crate::uv_closure::CoInstallVerdict::Sat
                } else {
                    crate::uv_closure::CoInstallVerdict::Unsat(vec![
                        "scoped fixture rejects the Pillow route".to_string(),
                    ])
                }
            }
        };
        let requests = Arc::new(Mutex::new(Vec::<PypiFetchRequest>::new()));
        let fetch = {
            let requests = Arc::clone(&requests);
            move |request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                let requests = Arc::clone(&requests);
                async move {
                    requests.lock().unwrap().push(request.clone());
                    Ok(test_wheel(
                        &request.bundle_name,
                        &request.pypi_name,
                        "11.0.0",
                        &[],
                    ))
                }
            }
        };
        let mut bundle = test_bundle(&[
            "pillow>=11,<11.1 ; python_version == '3.10' and sys_platform == 'linux'",
            "pillow>=12,<13 ; python_version == '3.11'",
        ]);
        auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &test_config(),
            None,
            None,
            None,
            &validated_probe,
            &scoped_reject,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(
            solve_calls.lock().unwrap().iter().any(|routes| routes
                .iter()
                .any(|route| route.pypi_name.as_str() == "pillow")),
            "the scoped co-solve must validate the provisional Pillow route"
        );
        assert_eq!(requests[0].pypi_name, "pillow");
        assert_eq!(
            requests[0].specifiers,
            VersionSpecifiers::from_str(">=11,<11.1").unwrap(),
            "unrelated environments and inactive target markers must not enter the restore request"
        );
        std::fs::remove_dir_all(tmp).unwrap();
    }

    #[tokio::test]
    async fn joint_unroute_rejects_unsatisfiable_requirement_with_sources() {
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async {
                    Err(anyhow!(
                        "fetch must not run for an unsatisfiable requirement"
                    ))
                }
            }
        };
        let mut bundle = test_bundle(&["pillow>=11,<11.1"]);
        bundle.extras.push(test_wheel(
            "conflicting-root",
            "conflicting-root",
            "2.0.0",
            &["pillow>=12,<13"],
        ));
        let mut deps_from_route = pillow_auto_route("12.3.0");
        deps_from_route.provenance = Provenance::DepsFromRelaxed;
        deps_from_route.route.input_requirements.push(
            crate::uv_closure::AutoRouteInputRequirement {
                specifiers: "==12.3.0".to_string(),
                source: "uv root requirement `pillow==12.3.0`".to_string(),
                provenance: Provenance::UvRoot,
                role: crate::uv_closure::AutoRouteInputRole::Requirement,
            },
        );
        bundle.auto_routed.push(deps_from_route);
        let target = crate::pypi::WheelTarget::for_subdir("3.10", "linux-64");
        let workspace_scope = crate::workspace::ResolvedWorkspaceTarget {
            contract: crate::workspace::WorkspaceTargetContract {
                subdir: "linux-64".to_string(),
                declared_virtual_packages: BTreeMap::new(),
                detected_virtual_packages: BTreeMap::new(),
            },
            profiles: vec!["linux-64-cuda-12".to_string()],
            environments: vec!["uwlab-gpu".to_string()],
        };
        let allow_source_route = |_route: crate::uv_closure::CondaRouteSpec| async {
            crate::uv_closure::CoInstallVerdict::Sat
        };
        let mut config = test_config();
        config.conda_deps.push("pillow".to_string());

        let error = auto_bundle_transitives_with_route_precheck(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            Some(&workspace_scope),
            &config,
            None,
            None,
            None,
            &validated_probe,
            &reject_every_mutable_route,
            &allow_source_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            1,
            &UvReresolveContext::default(),
        )
        .await
        .unwrap_err();

        let message = format!("{error:#}");
        assert!(
            error
                .downcast_ref::<crate::constraint::Conflict>()
                .is_some(),
            "the scoped diagnostic must retain its typed Conflict source: {message}"
        );
        assert!(message.contains("pillow"), "{message}");
        assert!(message.contains("mutually unsatisfiable"), "{message}");
        assert!(
            message.contains(
                "in environment 'uwlab-gpu' for bundle 'regression-pack' \
                 (target profile 'linux-64-cuda-12', platform linux-64, python 3.10)"
            ),
            "{message}"
        );
        for remediation in [
            "retread-relax",
            "retread-overrides",
            "retread-drop-deps",
            "(see README)",
        ] {
            assert!(message.contains(remediation), "{message}");
        }
        assert!(message.contains("regression-root==1.0.0"), "{message}");
        assert!(message.contains("pillow>=11,<11.1"), "{message}");
        assert!(message.contains("conflicting-root==2.0.0"), "{message}");
        assert!(message.contains("pillow>=12,<13"), "{message}");
        assert!(
            message.contains("uv root requirement `pillow==12.3.0`"),
            "{message}"
        );
        assert_eq!(
            fetch_calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "semantic conflicts must fail before any index request"
        );
    }

    #[tokio::test]
    async fn joint_unroute_typing_extensions_drop_only_conflict_fails_closed_before_fetch() {
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Err(anyhow!("conflicts must fail before any fetch")) }
            }
        };
        let mut bundle = test_bundle(&["pillow>=11,<11.1", "typing-extensions<4"]);
        bundle.extras.push(test_wheel(
            "conflicting-root",
            "conflicting-root",
            "2.0.0",
            &["pillow>=12,<13", "typing-extensions>=5,<6"],
        ));

        let mut pillow_route = pillow_auto_route("12.3.0");
        pillow_route.provenance = Provenance::DepsFromRelaxed;
        pillow_route
            .route
            .input_requirements
            .push(crate::uv_closure::AutoRouteInputRequirement {
                specifiers: "==12.3.0".to_string(),
                source: "uv root requirement `pillow==12.3.0`".to_string(),
                provenance: Provenance::UvRoot,
                role: crate::uv_closure::AutoRouteInputRole::Requirement,
            });
        bundle.auto_routed.push(pillow_route);

        let mut typing_route = pillow_auto_route("5.0.0");
        typing_route.route.pypi_name = "typing-extensions".to_string();
        typing_route.route.conda_name = "typing-extensions".to_string();
        typing_route.provenance = Provenance::DepsFromRelaxed;
        typing_route
            .route
            .input_requirements
            .push(crate::uv_closure::AutoRouteInputRequirement {
                specifiers: "==5.0.0".to_string(),
                source: "uv root requirement `typing-extensions==5.0.0`".to_string(),
                provenance: Provenance::UvRoot,
                role: crate::uv_closure::AutoRouteInputRole::Requirement,
            });
        bundle.auto_routed.push(typing_route);

        let target = crate::pypi::WheelTarget::for_subdir("3.10", "linux-64");
        let workspace_scope = crate::workspace::ResolvedWorkspaceTarget {
            contract: crate::workspace::WorkspaceTargetContract {
                subdir: "linux-64".to_string(),
                declared_virtual_packages: BTreeMap::new(),
                detected_virtual_packages: BTreeMap::new(),
            },
            profiles: vec!["linux-64-cuda-12".to_string()],
            environments: vec!["uwlab-gpu".to_string()],
        };
        let allow_source_route = |_route: crate::uv_closure::CondaRouteSpec| async {
            crate::uv_closure::CoInstallVerdict::Sat
        };
        let mut config = test_config();
        config.relax = RelaxPolicy::PatchThenMinorThenMajorThenLastResort;
        // Pillow is safely cap-relaxable, while typing-extensions can be
        // reconciled only by erasing the whole `<4` wheel edge. The latter
        // must remain a hard typed conflict under the active policy.
        config.conda_deps.push("pillow".to_string());

        let error = auto_bundle_transitives_with_route_precheck(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            Some(&workspace_scope),
            &config,
            None,
            None,
            None,
            &validated_probe,
            &reject_every_mutable_route,
            &allow_source_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            1,
            &UvReresolveContext::default(),
        )
        .await
        .unwrap_err();

        let message = format!("{error:#}");
        assert!(
            error
                .downcast_ref::<crate::constraint::Conflict>()
                .is_some(),
            "drop-only failure must retain the strict typed conflict: {message}"
        );
        assert!(
            message.contains(
                "dependency conflict in environment 'uwlab-gpu' for bundle \
                 'regression-pack' (target profile 'linux-64-cuda-12', platform \
                 linux-64, python 3.10): `typing-extensions` requirements are mutually \
                 unsatisfiable"
            ),
            "{message}"
        );
        assert!(message.contains("typing-extensions<4"), "{message}");
        assert!(message.contains("typing-extensions>=5,<6"), "{message}");
        assert_eq!(
            fetch_calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "all semantic conflicts must be collected before any index request"
        );
    }

    #[tokio::test]
    async fn joint_unroute_restores_compatible_pypi_requirement() {
        let mut overridden = RestoreRequestBuilder::new("overridden", RelaxPolicy::None);
        overridden.add_constraint(Constraint {
            specifiers: VersionSpecifiers::from_str("<2").unwrap(),
            source: "wheel metadata requires overridden<2".to_string(),
            provenance: Provenance::IndexWheelMetadata,
            origin_id: test_origin("overridden-wheel-cap", "<2"),
        });
        overridden.add_constraint(Constraint {
            specifiers: VersionSpecifiers::from_str("==3").unwrap(),
            source: "uv override requirement `overridden==3`".to_string(),
            provenance: Provenance::UvOverride,
            origin_id: test_origin("overridden-uv-override", "==3"),
        });
        overridden.add_constraint(Constraint {
            specifiers: VersionSpecifiers::from_str("<4").unwrap(),
            source: "uv constraint `overridden<4`".to_string(),
            provenance: Provenance::UvConstraint,
            origin_id: test_origin("overridden-uv-constraint", "<4"),
        });
        let overridden = overridden.finish().unwrap();
        assert!(
            overridden
                .specifiers
                .contains(&Version::from_str("3").unwrap()),
            "override plus additive constraint must remain satisfiable"
        );
        assert!(
            !overridden.specifiers.to_string().contains("<2"),
            "uv overrides must replace wheel/root requirements"
        );

        let mut soft_hints = RestoreRequestBuilder::new("soft-hints", RelaxPolicy::None);
        soft_hints.add_constraint(Constraint {
            specifiers: VersionSpecifiers::from_str(">=1").unwrap(),
            source: "wheel `root==1` Requires-Dist `soft-hints>=1`".to_string(),
            provenance: Provenance::IndexWheelMetadata,
            origin_id: test_origin("soft-hints-root-wheel", ">=1"),
        });
        RestoreRequestBuilder::add_preference(
            &mut soft_hints.route_preferences,
            "1.0".to_string(),
            "first soft hint".to_string(),
        );
        RestoreRequestBuilder::add_preference(
            &mut soft_hints.route_preferences,
            "2.0".to_string(),
            "second soft hint".to_string(),
        );
        assert_eq!(
            soft_hints.finish().unwrap().preferred_version,
            None,
            "conflicting soft hints must not make satisfiable hard requirements fail"
        );

        let mut prior_only = RestoreRequestBuilder::new("s3transfer", RelaxPolicy::None);
        RestoreRequestBuilder::add_preference(
            &mut prior_only.route_preferences,
            "0.13.1".to_string(),
            "prior uv route selected s3transfer==0.13.1".to_string(),
        );
        let prior_only = prior_only.finish().unwrap();
        assert!(
            prior_only.specifiers.is_empty(),
            "PriorSelection must never become a hard ==: {}",
            prior_only.specifiers
        );
        assert_eq!(prior_only.preferred_version.as_deref(), Some("0.13.1"));

        let mut index_pin = RestoreRequestBuilder::new("s3transfer", RelaxPolicy::None);
        index_pin.add_constraint(Constraint {
            specifiers: VersionSpecifiers::from_str("==0.13.1").unwrap(),
            provenance: Provenance::IndexWheelMetadata,
            source: "index wheel Requires-Dist `s3transfer==0.13.1`".to_string(),
            origin_id: test_origin("s3transfer-index-wheel", "==0.13.1"),
        });
        assert_eq!(
            index_pin.finish().unwrap().specifiers,
            VersionSpecifiers::from_str("==0.13.1").unwrap(),
            "an index-wheel exact requirement remains authoritative"
        );

        let requests = Arc::new(Mutex::new(Vec::<PypiFetchRequest>::new()));
        let fetch = {
            let requests = Arc::clone(&requests);
            move |request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                let requests = Arc::clone(&requests);
                async move {
                    requests.lock().unwrap().push(request.clone());
                    Ok(test_wheel(
                        &request.bundle_name,
                        &request.pypi_name,
                        "11.0.0",
                        &[],
                    ))
                }
            }
        };
        let mut bundle = test_bundle(&["pillow>=11,<11.1"]);
        bundle.auto_routed.push(pillow_auto_route("12.3.0"));
        let target = crate::pypi::WheelTarget::for_subdir("3.10", "linux-64");

        auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &test_config(),
            None,
            None,
            None,
            &validated_probe,
            &reject_every_mutable_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].specifiers,
            VersionSpecifiers::from_str(">=11,<11.1").unwrap()
        );
        assert_eq!(requests[0].preferred_version.as_deref(), Some("12.3.0"));
        assert!(
            requests[0]
                .specifiers
                .to_string()
                .find("==12.3.0")
                .is_none(),
            "a previous solver selection must stay soft"
        );
        drop(requests);
        assert!(bundle.auto_routed.is_empty());
        assert!(
            bundle
                .extras
                .iter()
                .any(|wheel| wheel.pypi_name == "pillow" && wheel.metadata.version == "11.0.0")
        );
    }

    #[tokio::test]
    async fn joint_unroute_reuses_compatible_wheel_already_in_bundle() {
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _indexes: Vec<String>, _failure_context: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async {
                    Err(anyhow!(
                        "compatible bundled wheel must prevent a duplicate fetch"
                    ))
                }
            }
        };
        let mut bundle = test_bundle(&["pillow>=11,<11.1"]);
        bundle
            .extras
            .push(test_wheel("pillow", "pillow", "11.0.0", &[]));
        bundle.auto_routed.push(pillow_auto_route("12.3.0"));
        let target = crate::pypi::WheelTarget::for_subdir("3.10", "linux-64");

        auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &test_config(),
            None,
            None,
            None,
            &validated_probe,
            &reject_every_mutable_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .unwrap();

        assert!(bundle.auto_routed.is_empty());
        assert_eq!(fetch_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(
            bundle
                .all_wheels()
                .filter(|wheel| PypiKey::from_pypi(&wheel.metadata.name)
                    == PypiKey::from_pypi("pillow"))
                .count(),
            1,
            "joint restoration must retain exactly one copy of the distribution",
        );
    }

    #[tokio::test]
    async fn joint_validation_rejects_ancient_metadata_route_with_default_index() {
        let records = vec![
            repo_record("python", "3.5.10", &[]),
            repo_record("python", "3.11.9", &[]),
            repo_record("zlib", "1.2.13", &[]),
            repo_record("zlib", "1.3.2", &[]),
            repo_record("openh264", "2.3.1", &["zlib <1.3"]),
            repo_record("ffmpeg", "4.4.4", &["openh264 <2.4", "zlib <1.3"]),
            repo_record("pyglet", "1.5.27", &["ffmpeg <6", "python >=3.11,<3.12"]),
            repo_record("pyglet", "1.2.4", &["python 3.5.*"]),
            repo_record("pyglet", "2.1.15", &["zlib >=1.3"]),
            repo_record(
                "libtorch",
                "2.10.0",
                &["python >=3.11,<3.12", "zlib >=1.3.1"],
            ),
        ];
        let fixed_specs = vec!["python 3.11.*".to_string(), "libtorch >=2.10".to_string()];
        assert!(
            crate::conda_solve::solve_records_for_test(&records, &fixed_specs, "3.11").is_ok(),
            "the fixed conda baseline must be satisfiable"
        );
        let ancient_specs = vec!["python 3.11.*".to_string(), "pyglet <2".to_string()];
        let ancient_solution =
            crate::conda_solve::solve_records_for_test(&records, &ancient_specs, "3.11")
                .expect("the individually valid ancient pyglet branch must solve");
        for (name, expected_version) in [
            ("pyglet", "1.5.27"),
            ("ffmpeg", "4.4.4"),
            ("zlib", "1.2.13"),
        ] {
            assert!(
                ancient_solution.iter().any(|record| {
                    record.package_record.name.as_normalized() == name
                        && record.package_record.version.to_string() == expected_version
                }),
                "the individual route must select ancient {name} {expected_version}: {ancient_solution:?}"
            );
        }
        let mut conflicting_specs = fixed_specs;
        conflicting_specs.push("pyglet <2".to_string());
        assert!(
            crate::conda_solve::solve_records_for_test(&records, &conflicting_specs, "3.11")
                .is_err(),
            "every pyglet<2 record must conflict through zlib or Python"
        );

        let probe = |pairs: Vec<(String, String)>| async move {
            pairs
                .into_iter()
                .map(|(package, spec)| crate::probe::ProbeResult {
                    package,
                    spec,
                    channels_consulted: vec!["conda-forge/linux-64".to_string()],
                    satisfiable: Some(true),
                    matching_candidates: 2,
                })
                .collect()
        };
        let records = Arc::new(records);
        let solve_inputs = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
        let solve = {
            let records = Arc::clone(&records);
            let solve_inputs = Arc::clone(&solve_inputs);
            move |routes: Vec<crate::uv_closure::CondaRouteSpec>| {
                let records = Arc::clone(&records);
                let solve_inputs = Arc::clone(&solve_inputs);
                async move {
                    let specs: Vec<String> = routes
                        .iter()
                        .map(|route| route.match_spec().to_string())
                        .collect();
                    solve_inputs.lock().unwrap().push(specs.clone());
                    match crate::conda_solve::solve_records_for_test(&records, &specs, "3.11") {
                        Ok(_) => crate::uv_closure::CoInstallVerdict::Sat,
                        Err(reasons) => crate::uv_closure::CoInstallVerdict::Unsat(reasons),
                    }
                }
            }
        };
        let fetch_calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |request: PypiFetchRequest, indexes: Vec<String>, failure_context: String| {
                let fetch_calls = Arc::clone(&fetch_calls);
                async move {
                    assert_eq!(request.pypi_name, "pyglet");
                    assert_eq!(
                        request.specifiers,
                        VersionSpecifiers::from_str("<2").unwrap()
                    );
                    let fetch_one = move |request: PypiFetchRequest, index: String| {
                        let fetch_calls = Arc::clone(&fetch_calls);
                        async move {
                            fetch_calls.lock().unwrap().push(index.clone());
                            if index.trim_end_matches('/')
                                != crate::workspace::DEFAULT_PYPI_INDEX.trim_end_matches('/')
                            {
                                return Err(crate::pypi::pypi_index_miss(format!(
                                    "package absent from {index}"
                                )));
                            }
                            Ok(test_wheel(
                                &request.bundle_name,
                                &request.pypi_name,
                                "1.5.27",
                                &[],
                            ))
                        }
                    };
                    fetch_from_index_chain(&indexes, request, &fetch_one, failure_context).await
                }
            }
        };
        let mut bundle = test_bundle(&["libtorch>=2.10", "pyglet<2"]);
        let indexes = vec![
            "https://pypi.nvidia.com".to_string(),
            crate::workspace::DEFAULT_PYPI_INDEX.to_string(),
        ];
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");

        auto_bundle_transitives_with(
            &mut bundle,
            &indexes,
            &target,
            &test_config(),
            None,
            None,
            None,
            &probe,
            &solve,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .unwrap();

        assert!(
            bundle
                .extras
                .iter()
                .any(|wheel| wheel.pypi_name == "pyglet")
        );
        let emitted =
            super::super::emitted_bundle_route_specs(&bundle, &test_config(), &target).unwrap();
        assert!(
            emitted
                .iter()
                .all(|route| route.conda_name.key().as_str() != "pyglet"),
            "the incompatible conda pyglet route must be absent after restoration: {emitted:?}"
        );
        assert!(bundle.probe_decisions.iter().any(|decision| {
            decision.stage == "auto_bundle_loose"
                && decision.pypi_name == "pyglet"
                && decision.routing_decision == "short-circuit"
        }));
        assert!(bundle.probe_decisions.iter().any(|decision| {
            decision.stage == "auto_route_joint_solve"
                && decision.pypi_name == "pyglet"
                && decision.routing_decision == "joint-co-solve-rejected-to-pypi"
        }));
        assert!(solve_inputs.lock().unwrap().iter().any(|specs| {
            specs.iter().any(|spec| spec.starts_with("pyglet "))
                && specs.iter().any(|spec| spec.starts_with("libtorch "))
        }));
        assert_eq!(
            *fetch_calls.lock().unwrap(),
            vec![
                "https://pypi.nvidia.com".to_string(),
                crate::workspace::DEFAULT_PYPI_INDEX.to_string(),
            ],
            "joint rejection must retain the complete default-index fallback chain"
        );
    }

    #[tokio::test]
    async fn default_index_fallback_resolves_flatdict_after_private_index_miss() {
        let source_entry = crate::config::WheelEntry {
            path: Some("owned-source".to_string()),
            ..Default::default()
        };
        let private_entry = crate::config::WheelEntry {
            version: Some("1.0.0".to_string()),
            index: Some("https://pypi.nvidia.com".to_string()),
            ..Default::default()
        };
        let explicit_private_chain = crate::index_chain::index_chain(
            [&source_entry, &private_entry]
                .into_iter()
                .filter(|entry| !entry.is_url())
                .filter_map(|entry| entry.index.clone()),
            &[],
            crate::index_chain::IndexPurpose::RootResolve,
        );
        assert_eq!(
            explicit_private_chain,
            vec![
                "https://pypi.nvidia.com".to_string(),
                crate::workspace::DEFAULT_PYPI_INDEX.to_string(),
            ]
        );
        let calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let fetch = {
            let calls = Arc::clone(&calls);
            move |request: PypiFetchRequest, indexes: Vec<String>, failure_context: String| {
                let calls = Arc::clone(&calls);
                async move {
                    assert_eq!(request.pypi_name, "flatdict");
                    assert_eq!(
                        request.specifiers,
                        VersionSpecifiers::from_str(">=4.0.1,<4.1").unwrap()
                    );
                    let fetch_one = move |request: PypiFetchRequest, index: String| {
                        let calls = Arc::clone(&calls);
                        async move {
                            calls.lock().unwrap().push(index.clone());
                            if index.trim_end_matches('/')
                                != crate::workspace::DEFAULT_PYPI_INDEX.trim_end_matches('/')
                            {
                                return Err(crate::pypi::pypi_index_miss(format!(
                                    "flatdict absent from {index}"
                                )));
                            }
                            Ok(test_wheel(
                                &request.bundle_name,
                                &request.pypi_name,
                                "4.0.1",
                                &[],
                            ))
                        }
                    };
                    fetch_from_index_chain(&indexes, request, &fetch_one, failure_context).await
                }
            }
        };
        let probe = |pairs: Vec<(String, String)>| async move {
            pairs
                .into_iter()
                .map(|(package, spec)| crate::probe::ProbeResult {
                    package,
                    spec,
                    channels_consulted: vec!["conda-forge/linux-64".to_string()],
                    satisfiable: Some(false),
                    matching_candidates: 0,
                })
                .collect()
        };
        let solve = |_| async { crate::uv_closure::CoInstallVerdict::Sat };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let mut bundle = test_bundle(&["flatdict>=4.0.1,<4.1"]);
        auto_bundle_transitives_with(
            &mut bundle,
            &explicit_private_chain,
            &target,
            &test_config(),
            None,
            None,
            None,
            &probe,
            &solve,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .unwrap();

        let wheel = bundle
            .extras
            .iter()
            .find(|wheel| wheel.pypi_name == "flatdict")
            .expect("flatdict must be bundled after conda routing is refused");
        assert_eq!(wheel.metadata.name, "flatdict");
        assert_eq!(wheel.metadata.version, "4.0.1");
        assert_eq!(
            *calls.lock().unwrap(),
            vec![
                "https://pypi.nvidia.com".to_string(),
                crate::workspace::DEFAULT_PYPI_INDEX.to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn chain_fetch_runs_once_and_preserves_failure_context() {
        let indexes = vec!["private".to_string(), "public".to_string()];
        let calls = Arc::new(Mutex::new(Vec::<(Vec<String>, String)>::new()));
        let fetch = {
            let calls = Arc::clone(&calls);
            move |request: PypiFetchRequest, indexes: Vec<String>, failure_context: String| {
                let calls = Arc::clone(&calls);
                async move {
                    assert_eq!(request.pypi_name, "flatdict");
                    calls
                        .lock()
                        .unwrap()
                        .push((indexes.clone(), failure_context.clone()));
                    super::super::fetch_artifact_from_pypi_index_chain(
                        &indexes,
                        true,
                        request.preferred_version,
                        |_index, _phase, _prefer_version| async {
                            Err::<ResolvedWheel, _>(crate::pypi::pypi_index_miss("absent"))
                        },
                        failure_context,
                    )
                    .await
                }
            }
        };
        let probe = |pairs: Vec<(String, String)>| async move {
            pairs
                .into_iter()
                .map(|(package, spec)| crate::probe::ProbeResult {
                    package,
                    spec,
                    channels_consulted: vec!["conda-forge/linux-64".to_string()],
                    satisfiable: Some(false),
                    matching_candidates: 0,
                })
                .collect()
        };
        let solve = |_| async { crate::uv_closure::CoInstallVerdict::Sat };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let mut bundle = test_bundle(&["flatdict>=4.0.1,<4.1"]);

        let error = auto_bundle_transitives_with(
            &mut bundle,
            &indexes,
            &target,
            &test_config(),
            None,
            None,
            None,
            &probe,
            &solve,
            &fetch,
            &["conda-forge/linux-64".to_string()],
            &UvReresolveContext::default(),
        )
        .await
        .unwrap_err();

        assert!(crate::pypi::is_pypi_index_miss(&error));
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1, "the chain-aware callback must run once");
        assert_eq!(calls[0].0, indexes);
        assert!(
            calls[0]
                .1
                .starts_with("auto-bundle: no PyPI index could resolve `flatdict")
        );
        assert!(calls[0].1.ends_with("after conda routing was refused"));
        let rendered = format!("{error:#}");
        assert!(rendered.contains(&calls[0].1));
        assert!(rendered.contains("exact wheel private: absent"));
        assert!(rendered.contains("sdist public: absent"));
    }

    /// v2.10.0: seed_worklist must NOT enqueue a dep whose canonical name is
    /// in `sibling_names`, but MUST enqueue a dep that is NOT in the set.
    ///
    /// Scenario: "isaaclab-visualizers" (extras=["all"]) has two Requires-Dist:
    ///   - `isaaclab`         → sibling (same bundle group) → must NOT be enqueued
    ///   - `matplotlib`       → normal dep                  → MUST be enqueued
    ///
    /// The prefix is empty (source-form entry), so pep508_base_dep_in_prefix
    /// would normally pick up ANY base dep.
    #[test]
    fn seed_worklist_skips_sibling_base_dep() {
        let requires_dist = vec!["isaaclab".to_string(), "matplotlib".to_string()];
        // extras_requested is empty → only base-dep path runs.
        let mut siblings = HashSet::new();
        siblings.insert("isaaclab".to_string()); // canonical_conda_name("isaaclab") = "isaaclab"

        let seen: HashSet<String> = HashSet::new();
        let mut work: VecDeque<Pending> = VecDeque::new();

        seed_worklist(
            &requires_dist,
            &[], // no extras requested
            &["https://pypi.org/simple/".to_string()],
            "", // empty prefix: all base deps match
            &seen,
            &mut work,
            None, // no state
            &siblings,
        )
        .expect("seed_worklist must not error");

        let enqueued: Vec<&str> = work.iter().map(|p| p.pypi_name.as_str()).collect();

        assert!(
            !enqueued.contains(&"isaaclab"),
            "sibling 'isaaclab' must NOT be enqueued; enqueued={enqueued:?}"
        );
        assert!(
            enqueued.contains(&"matplotlib"),
            "non-sibling 'matplotlib' MUST be enqueued; enqueued={enqueued:?}"
        );
        assert_eq!(
            enqueued.len(),
            1,
            "exactly one dep (matplotlib) should be enqueued; got {enqueued:?}"
        );
    }

    /// v2.10.0: seed_worklist must NOT enqueue an extras-gated dep whose
    /// canonical name is in `sibling_names`.
    ///
    /// Scenario: "isaaclab-visualizers" requests extra "all", which gate-deps:
    ///   - `isaaclab; extra == "all"`   → sibling → must NOT be enqueued
    ///   - `numpy; extra == "all"`      → normal  → MUST be enqueued
    #[test]
    fn seed_worklist_skips_sibling_extra_dep() {
        let requires_dist = vec![
            "isaaclab; extra == \"all\"".to_string(),
            "numpy; extra == \"all\"".to_string(),
        ];
        let extras_requested = vec!["all".to_string()];

        let mut siblings = HashSet::new();
        siblings.insert("isaaclab".to_string());

        let seen: HashSet<String> = HashSet::new();
        let mut work: VecDeque<Pending> = VecDeque::new();

        seed_worklist(
            &requires_dist,
            &extras_requested,
            &["https://pypi.org/simple/".to_string()],
            "", // empty prefix
            &seen,
            &mut work,
            None,
            &siblings,
        )
        .expect("seed_worklist must not error");

        let enqueued: Vec<&str> = work.iter().map(|p| p.pypi_name.as_str()).collect();

        assert!(
            !enqueued.contains(&"isaaclab"),
            "extras-gated sibling 'isaaclab' must NOT be enqueued; enqueued={enqueued:?}"
        );
        assert!(
            enqueued.contains(&"numpy"),
            "non-sibling extras dep 'numpy' MUST be enqueued; enqueued={enqueued:?}"
        );
        assert_eq!(
            enqueued.len(),
            1,
            "exactly one dep (numpy) should be enqueued; got {enqueued:?}"
        );
    }

    /// v2.10.0: when `sibling_names` is empty, seed_worklist behaves exactly
    /// as before — all matching deps are enqueued.
    #[test]
    fn seed_worklist_empty_siblings_enqueues_all() {
        let requires_dist = vec!["isaaclab".to_string(), "matplotlib".to_string()];
        let seen: HashSet<String> = HashSet::new();
        let mut work: VecDeque<Pending> = VecDeque::new();

        seed_worklist(
            &requires_dist,
            &[],
            &["https://pypi.org/simple/".to_string()],
            "",
            &seen,
            &mut work,
            None,
            &HashSet::new(), // empty siblings → no-op
        )
        .expect("seed_worklist must not error");

        let enqueued: Vec<&str> = work.iter().map(|p| p.pypi_name.as_str()).collect();

        assert!(
            enqueued.contains(&"isaaclab"),
            "isaaclab must be enqueued when siblings is empty; enqueued={enqueued:?}"
        );
        assert!(
            enqueued.contains(&"matplotlib"),
            "matplotlib must be enqueued when siblings is empty; enqueued={enqueued:?}"
        );
        assert_eq!(
            enqueued.len(),
            2,
            "both deps should be enqueued; got {enqueued:?}"
        );
    }

    // ---- C30-1 guards: the restore path's learned-fact yield names its pack ----
    //
    // Both are RED on `c0f1549`. That binary printed the campaign's first two
    // `learned_fact_yielded` rows (job 5834618) and both said the placeholder
    // instead of a pack name, because the only scope in the tree was entered
    // by `produce_output_with_conflicts` and the yield happened HERE, on the
    // route-restore path, before any emission ran.
    //
    // These guards drive the same function production drives --
    // `finish_with_graph_context` -> `finish_with_suggestion` ->
    // `relax_decision::decide` -> `constraint::finalize_quiet_detailed` --
    // and read the row the process actually writes. A guard that entered the
    // scope itself and then called `finalize_quiet` would have been green on
    // the broken binary, which is precisely how p6aa guard (a) missed this.

    /// Capture the WARN/ERROR rows a body writes, so a guard can assert on
    /// the row the operator will read rather than on a return value.
    fn c30_captured_rows<T>(body: impl FnOnce() -> T) -> (T, String) {
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

    /// A LEARNED workspace conda fact, in the rendered shape the backend
    /// emits. The sentence CONTAINS `LEARNED_WORKSPACE_FACT_SOURCE` and is
    /// not equal to it -- `is_learned_advisory_sentence` tests containment,
    /// and a fixture built on the bare constant proved a yield that never
    /// fired in an arm once already (p6z pair 1).
    fn c30_learned_conda_fact(pypi_name: &str, version: &str) -> Constraint {
        let source = format!(
            "uv constraint `{pypi_name}=={version}` from {} `precise-consuming-envs` \
             (conda `{pypi_name}=={version}`)",
            crate::uv_closure::LEARNED_WORKSPACE_FACT_SOURCE,
        );
        assert!(
            crate::uv_closure::is_learned_advisory_sentence(&source),
            "non-vacuity: the fixture's sentence must be classified LEARNED, or the \
             yield under test cannot fire: {source}",
        );
        Constraint {
            specifiers: VersionSpecifiers::from_str(&format!("=={version}")).unwrap(),
            provenance: Provenance::UvConstraint,
            source,
            origin_id: test_origin(
                &format!("{pypi_name}-learned-conda-fact"),
                &format!("=={version}"),
            ),
        }
    }

    fn c30_wheel_requirement(pypi_name: &str, specifiers: &str, wheel: &str) -> Constraint {
        Constraint {
            specifiers: VersionSpecifiers::from_str(specifiers).unwrap(),
            provenance: Provenance::IndexWheelMetadata,
            source: format!("wheel `{wheel}` Requires-Dist `{pypi_name}{specifiers}`"),
            origin_id: test_origin(&format!("{pypi_name}-{wheel}"), specifiers),
        }
    }

    /// Run one restore through the production entry and hand back its rows.
    fn c30_restore_rows(
        pypi_name: &str,
        bundle_name: &str,
        environments: &[&str],
        constraints: Vec<Constraint>,
    ) -> String {
        let context = JointRouteDiagnosticContext {
            bundle: bundle_name.to_string(),
            environments: environments.iter().map(|env| (*env).to_string()).collect(),
            profiles: vec!["linux-64".to_string()],
            platform: "linux-64".to_string(),
            python: "3.10".to_string(),
        };
        let mut bundle = test_bundle(&[]);
        bundle.conda_name = context.bundle.clone();
        let config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-deps-from": "requirements.txt",
            "retread-wheels": {}
        }))
        .unwrap();
        let target = crate::pypi::WheelTarget::for_subdir(&context.python, &context.platform);
        let mut builder = RestoreRequestBuilder::new(
            pypi_name,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
        );
        for constraint in constraints {
            builder.add_constraint(constraint);
        }
        let (request, rows) = c30_captured_rows(|| {
            builder.finish_with_graph_context(Some(&context), &bundle, &config, &target)
        });
        request.expect(
            "the LEARNED fact must yield and the restore must resolve; a conflict here \
             means the fixture stopped exercising the yield",
        );
        rows
    }

    /// C30-1 guard (a). The `flashsac-pack` clause set -- the case p6z-1 was
    /// written for -- must name `flashsac-pack` when it is reached through
    /// the restore path, not through a scope the guard entered itself.
    #[test]
    fn c30_b_flashsac_restore_yield_row_names_the_pack() {
        let rows = c30_restore_rows(
            "setuptools",
            "flashsac-pack",
            &["flashsac-gpu"],
            vec![
                c30_learned_conda_fact("setuptools", "84.0.0"),
                c30_wheel_requirement("setuptools", ">=41.0.0", "tensorboard==2.21.0"),
                c30_wheel_requirement("setuptools", "!=50.0.0", "dm_control==1.0.45"),
                c30_wheel_requirement("setuptools", "<=65", "FlashRL==0.1.0"),
            ],
        );

        assert!(
            rows.contains("learned_fact_yielded"),
            "the restore path applied the yield and must leave its audit row: {rows}",
        );
        assert!(
            rows.contains("bundle=flashsac-pack"),
            "the row must name the pack whose outcome the policy changed: {rows}",
        );
        assert!(
            rows.contains("envs=flashsac-gpu"),
            "and the consuming environment the restore was performed for, which the \
             joint-route context carries and the emission path does not: {rows}",
        );
        assert!(
            rows.contains("package=setuptools"),
            "the row must name the package it decided: {rows}",
        );
        assert!(
            !rows.contains("learned_fact_yield_unattributable"),
            "a scoped caller must never reach the emitter's refusal arm: {rows}",
        );
    }

    /// C30-1 guard (b). THE TWO ROWS THE ARM ACTUALLY PRINTED.
    ///
    /// Transcribed from job 5834618's backend log (ANSI stripped), the only
    /// arm in which this row has ever appeared: `setuptools` `==84.0.0` kept
    /// `>=41.0.0, !=50.0.0, <=65`, and `prettytable` `==3.18.0` kept
    /// `>=3.3, <3.4`, both from `precise-consuming-envs`, both immediately
    /// before `restoring PyPI wheel` and `computed vendored set
    /// bundle=flashsac-pack`. Both printed the placeholder. Both must now
    /// print the pack, and the SECOND one matters on its own: it proves the
    /// scope is per-restore and not a name some earlier call left behind.
    #[test]
    fn c30_c_the_two_rows_the_c30_arm_printed_name_their_bundle() {
        let setuptools = c30_restore_rows(
            "setuptools",
            "flashsac-pack",
            &["flashsac-gpu"],
            vec![
                c30_learned_conda_fact("setuptools", "84.0.0"),
                c30_wheel_requirement("setuptools", ">=41.0.0", "tensorboard==2.21.0"),
                c30_wheel_requirement("setuptools", "!=50.0.0", "dm_control==1.0.45"),
                c30_wheel_requirement("setuptools", "<=65", "FlashRL==0.1.0"),
            ],
        );
        let prettytable = c30_restore_rows(
            "prettytable",
            "flashsac-pack",
            &["flashsac-gpu"],
            vec![
                c30_learned_conda_fact("prettytable", "3.18.0"),
                c30_wheel_requirement("prettytable", ">=3.3", "sage-agent==0.4.0"),
                c30_wheel_requirement("prettytable", "<3.4", "sage-agent==0.4.0"),
            ],
        );

        for (rows, package, clause, kept) in [
            (
                &setuptools,
                "setuptools",
                "`==84.0.0`",
                "kept=>=41.0.0, !=50.0.0, <=65",
            ),
            (&prettytable, "prettytable", "`==3.18.0`", "kept=>=3.3, <3.4"),
        ] {
            assert!(
                rows.contains(&format!("package={package}")),
                "the {package} row must be present: {rows}",
            );
            assert!(
                rows.contains("bundle=flashsac-pack"),
                "the {package} row printed an unattributed bundle in job 5834618 and \
                 must now name the pack: {rows}",
            );
            assert!(
                rows.contains(clause),
                "the {package} row must quote the clause that yielded: {rows}",
            );
            assert!(
                rows.contains(kept),
                "the {package} row must state what was kept, which is what makes it \
                 auditable against the lock: {rows}",
            );
        }
    }

    /// The evidence a door decides on, built for one name across named
    /// consuming environments. `locked` is what each environment HOLDS.
    fn fact1_bundle(
        fact: &[(&str, &[&str])],
        locked: &[(&str, &[(&str, &str)])],
    ) -> Bundle {
        let mut bundle = test_bundle(&[]);
        for (name, versions) in fact {
            bundle.workspace_conda_provider_facts.insert(
                (*name).to_string(),
                super::super::WorkspaceCondaProviderFact {
                    selected_versions: versions.iter().map(|v| (*v).to_string()).collect(),
                    declared_specs: BTreeSet::new(),
                    present_in_all_consumers: true,
                },
            );
        }
        for (env, versions) in locked {
            bundle.workspace_locked_conda.insert(
                (*env).to_string(),
                versions
                    .iter()
                    .map(|(name, version)| ((*name).to_string(), (*version).to_string()))
                    .collect(),
            );
        }
        bundle
    }

    /// GUARD (a), N27-RETREAD-141. THE MEASURED dex-retargeting SHAPE.
    ///
    /// Relock `6115467` printed, six times,
    /// `dep=dex-retargeting requirement=numpy <2.0.0,>=1.21.0 fact=numpy==2.4.6`
    /// -- and its own emitted lock holds `numpy 1.26.4` in BOTH consuming
    /// environments (`groot-sonic-gpu`, `viral-gpu`), because the pack emitted
    /// `constrains: numpy >=1.26.0,<1.27` under the cap. The fact was the
    /// version those environments would hold IF THE PACK DID NOT EXIST: the
    /// solve that produces it filters the pack out. A requirement every
    /// consumer's HELD version satisfies is not a crossing, so no backtrack and
    /// no row.
    #[test]
    fn fact1_a_requirement_every_consumer_holds_a_version_for_is_not_a_crossing() {
        let bundle = fact1_bundle(
            &[("numpy", &["2.4.6"])],
            &[
                ("groot-sonic-gpu", &[("numpy", "1.26.4")]),
                ("viral-gpu", &[("numpy", "1.26.4")]),
            ],
        );
        let holdings = admission_fact_holdings(&bundle);
        assert_eq!(
            holdings["numpy"].keys().cloned().collect::<Vec<_>>(),
            vec!["groot-sonic-gpu".to_string(), "viral-gpu".to_string()],
            "the lock answers for this name, so the float does not get a vote",
        );
        let env = crate::relax::marker_env_for("linux-64", "3.11").unwrap();
        let verdict = fact_verdict_for_requires_dist(
            &["numpy<2.0.0,>=1.21.0".to_string()],
            &holdings,
            &env,
        );
        assert!(
            verdict.crossings.is_empty(),
            "every consumer holds 1.26.4, which the requirement admits: {:?}",
            verdict.crossings,
        );
        assert!(
            verdict.splits.is_empty(),
            "the consumers agree, so there is no split to report: {:?}",
            verdict.splits,
        );
    }

    /// GUARD (b), N27-RETREAD-141. THE MEASURED googleapis-common-protos SHAPE,
    /// WHICH MUST NOT MOVE.
    ///
    /// All four precise consumers of `isaaclab-2.3x-pack` (`pace`,
    /// `pm-isaaclab`, `unitree-rl-lab-gpu`, `uwlab-gpu`) hold `protobuf 5.29.3`
    /// in relock `6115467`'s emitted lock, and `protobuf<8.0.0,>=6.33.5`
    /// admits none of them. That is a TRUE crossing and the backtrack
    /// `1.75.3 -> 1.75.0` this run printed four times stays.
    #[test]
    fn fact1_a_requirement_no_consumer_holds_a_version_for_is_still_a_crossing() {
        let held: &[(&str, &str)] = &[("protobuf", "5.29.3")];
        let bundle = fact1_bundle(
            &[("protobuf", &["5.29.3"])],
            &[
                ("pace", held),
                ("pm-isaaclab", held),
                ("unitree-rl-lab-gpu", held),
                ("uwlab-gpu", held),
            ],
        );
        let holdings = admission_fact_holdings(&bundle);
        let env = crate::relax::marker_env_for("linux-64", "3.11").unwrap();
        let verdict = fact_verdict_for_requires_dist(
            &["protobuf<8.0.0,>=6.33.5".to_string()],
            &holdings,
            &env,
        );
        assert_eq!(verdict.crossings.len(), 1, "the crossing must survive");
        let crossing = &verdict.crossings[0];
        assert_eq!(crossing.fact_name, "protobuf");
        assert_eq!(crossing.holders_total, 4, "four consumers, none admitted");
        assert!(verdict.splits.is_empty());
        assert_eq!(
            pypi_admission_fact_crossing_unresolved_row(
                "googleapis-common-protos",
                crossing,
                FactConstrainedSite::AutoBundleAdmission,
            ),
            "### PYPI ADMISSION FACT-CROSSING UNRESOLVED dep=googleapis-common-protos \
             requirement=protobuf<8.0.0,>=6.33.5 fact=protobuf==5.29.3 consumers_holding=0/4 \
             door=auto-bundle policy=constrains-only:learned-fact-yields-to-cap|\
             declared-pin-kept|undecidable-omitted",
        );
    }

    /// GUARD (c), N27-RETREAD-141. A SPLIT IS NOT A CROSSING, AND IT IS NOT
    /// SILENCE.
    ///
    /// One consumer holding a version the requirement admits is the existence
    /// proof the door asks for, so nothing backtracks. Before this commit the
    /// name was dropped by `selected_versions.len() != 1` before the predicate
    /// ever saw it, so the split produced no decision AND no row.
    #[test]
    fn fact1_a_split_admits_the_wheel_and_prints_its_own_row() {
        let bundle = fact1_bundle(
            &[("numpy", &["2.4.6"])],
            &[
                ("viral-gpu", &[("numpy", "1.26.4")]),
                ("newton-gpu", &[("numpy", "2.4.6")]),
            ],
        );
        let holdings = admission_fact_holdings(&bundle);
        let env = crate::relax::marker_env_for("linux-64", "3.11").unwrap();
        let verdict = fact_verdict_for_requires_dist(
            &["numpy<2.0.0,>=1.21.0".to_string()],
            &holdings,
            &env,
        );
        assert!(
            verdict.crossings.is_empty(),
            "one consumer holds an admitted version, so this is not a crossing: {:?}",
            verdict.crossings,
        );
        assert_eq!(verdict.splits.len(), 1);
        let split = &verdict.splits[0];
        assert_eq!((split.holders_admitting, split.holders_total), (1, 2));
        assert_eq!(
            pypi_admission_consumer_split_row(
                "dex-retargeting",
                split,
                FactConstrainedSite::AutoBundleAdmission,
            ),
            "### PYPI ADMISSION CONSUMER-SPLIT dep=dex-retargeting fact=numpy \
             requirement=numpy<2.0.0,>=1.21.0 holders=1/2 \
             held=newton-gpu=2.4.6,viral-gpu=1.26.4 door=auto-bundle",
        );
    }

    /// GUARD (d), N27-RETREAD-141. THE ORDINARY CASE DOES NOT MOVE.
    ///
    /// With no base lock -- the `--base-lock drop` shape relock `6115467` ran,
    /// where `workspace_locked_conda` is EMPTY -- the door falls back to the
    /// workspace fact and decides exactly as it did before this commit: one
    /// holder, admitted, no crossing, no split, and `backtracked=none`
    /// byte-identical. `EMIT_EPOCH` therefore does not move: no emitted byte of
    /// an unchanged case changes.
    #[test]
    fn fact1_with_no_lock_the_float_still_decides_and_the_ordinary_case_is_unchanged() {
        let bundle = fact1_bundle(&[("packaging", &["23.0"])], &[]);
        let holdings = admission_fact_holdings(&bundle);
        assert_eq!(
            holdings["packaging"].keys().cloned().collect::<Vec<_>>(),
            vec!["workspace-fact:23.0".to_string()],
            "with no lock the float is the only evidence there is",
        );
        let env = crate::relax::marker_env_for("linux-64", "3.11").unwrap();
        let admits = fact_verdict_for_requires_dist(
            &["packaging>=20".to_string()],
            &holdings,
            &env,
        );
        assert!(admits.crossings.is_empty() && admits.splits.is_empty());
        let excludes = fact_verdict_for_requires_dist(
            &["packaging>=24.0".to_string()],
            &holdings,
            &env,
        );
        assert_eq!(excludes.crossings.len(), 1, "the wheel/packaging row of 6115467");
        assert_eq!(excludes.crossings[0].holders_total, 1);
        assert_eq!(
            pypi_closure_fact_constrained_row(holdings.len(), &[]),
            "### PYPI CLOSURE FACT-CONSTRAINED names=1 backtracked=none",
        );
        assert_eq!(
            crate::lock::EMIT_EPOCH,
            59,
            "no emitted byte of an unchanged case moved, so the epoch does not",
        );
    }
}
