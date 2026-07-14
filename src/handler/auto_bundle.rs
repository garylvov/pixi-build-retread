//! Auto-bundle BFS: prefer-conda routing and PyPI-only transitive
//! packaging into the bundle.
//!
//! Extracted from handler.rs (Phase 0b.3). All functions are behavior-
//! identical whole-function moves; no logic changes.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::future::Future;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, anyhow};
use rattler_conda_types::ChannelUrl;
use uv_pep508::MarkerEnvironment;
use uv_pep508::uv_pep440::{
    Operator, Version, VersionSpecifier, VersionSpecifiers, release_specifiers_to_ranges,
};

use crate::config::RetreadConfig;
use crate::pypi;
use crate::relax::{canonical_conda_name, default_marker_env, marker_env_for};
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

/// Returns `true` if `conda_normalized_pypi_name` has an unambiguous conda
/// equivalent in the effective name_map (parselmouth + FALLBACK + user
/// retread-name-map). This identifies a conda target; the requirement-specific
/// probe must still pass [`validated_conda_route`] before auto-bundling is
/// skipped.
pub(crate) fn prefer_conda_match(
    conda_normalized_pypi_name: &str,
    name_map: &BTreeMap<String, String>,
) -> bool {
    name_map.contains_key(conda_normalized_pypi_name)
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
    dep_conda_name: &str,
    name_map: &BTreeMap<String, String>,
    pypi_to_conda: &PypiToCondaMap,
) -> Option<String> {
    if let Some(target) = name_map.get(dep_conda_name) {
        return Some(target.clone());
    }
    let candidates = pypi_to_conda.get(dep_conda_name)?;
    if candidates.iter().any(|c| c == dep_conda_name) {
        Some(dep_conda_name.to_string())
    } else if candidates.len() == 1 {
        Some(candidates[0].clone())
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
}

/// One active, target-marker-matched dependency declaration. The source text
/// stays attached until the complete restore requirement has been proven
/// satisfiable, so an empty intersection can name the wheels and raw
/// `Requires-Dist` lines that created it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ObservedRequirement {
    specifiers: VersionSpecifiers,
    source: String,
}

type ObservedRequirements = BTreeMap<String, Vec<ObservedRequirement>>;

/// Structural identity for the one restore failure that a precise workspace
/// conda owner can safely supersede. The display text intentionally preserves
/// the v4.6.6 diagnostic verbatim.
#[derive(Debug)]
struct UnsatisfiableRestoreRequirements {
    pypi_name: String,
    sources: String,
}

impl std::fmt::Display for UnsatisfiableRestoreRequirements {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "joint route validation cannot restore `{}` to PyPI: active requirements are mutually unsatisfiable:\n{}",
            self.pypi_name, self.sources
        )
    }
}

impl std::error::Error for UnsatisfiableRestoreRequirements {}

/// Why an impossible PyPI restore was NOT superseded by a workspace-owned
/// drop, in the exact terms of the guards that blocked it.
///
/// Silence here is what let a no-op ship: v4.6.7 added the drop but fell
/// through to the v4.6.6 error verbatim whenever a guard declined, so "the
/// guard never fired" and "the guard does not exist" were byte-identical from
/// the outside -- through a full publish and a 30-minute relock. Every decline
/// must now name the dependency and the specific guard.
fn drop_decline_reasons(
    pypi_key: &str,
    workspace_ownership: &super::WorkspaceRouteOwnership,
    rejected_conda_routes: &BTreeSet<String>,
    workspace_drop_authorized: bool,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if !workspace_drop_authorized {
        reasons.push(
            "the fixed workspace conda baseline did not solve, so rule 1 abstains \
             (ownership is indeterminate, not disproven)"
                .to_string(),
        );
    }
    if workspace_ownership.excluded_pypi_names.contains(pypi_key) {
        reasons.push(format!(
            "`{pypi_key}` is held on PyPI by an explicit retread-override, a keep-pypi \
             entry, a wheel entry of this pack, or a direct-URL wheel source"
        ));
    }
    let owns_pypi_name = workspace_ownership.pypi_names.contains(pypi_key);
    let unowned_routes: Vec<&String> = rejected_conda_routes
        .iter()
        .filter(|name| !workspace_ownership.conda_names.contains(*name))
        .collect();
    if !owns_pypi_name && (rejected_conda_routes.is_empty() || !unowned_routes.is_empty()) {
        if workspace_ownership.conda_names.is_empty() && workspace_ownership.pypi_names.is_empty() {
            reasons.push(format!(
                "no precise consuming workspace environment conda-owns `{pypi_key}`: this \
                 pack's workspace ownership set is EMPTY (the workspace declares no \
                 consuming environment for this pack, or that environment's conda baseline \
                 could not be solved)"
            ));
        } else if rejected_conda_routes.is_empty() {
            reasons.push(format!(
                "no rejected conda route recorded a concrete conda target for `{pypi_key}`, \
                 so its identity cannot be proven against the workspace"
            ));
        } else {
            reasons.push(format!(
                "the consuming workspace does not conda-own the rejected conda route(s) {:?} \
                 behind `{pypi_key}` (workspace conda-owns: {:?})",
                unowned_routes, workspace_ownership.conda_names,
            ));
        }
    }
    if reasons.is_empty() {
        // Defensive: the guard declined for a reason not enumerated above.
        reasons.push(
            "an unenumerated workspace-ownership guard declined the drop (this is a bug \
             in `drop_decline_reasons`)"
                .to_string(),
        );
    }
    reasons
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum DedupeVersionKey {
    /// Ordinary PEP 440 comparison ignores trailing release zeroes.
    Pep440(Version),
    /// Compatible-release and wildcard operators also depend on how many
    /// release segments were written.
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

fn observe_requirement(
    observed: &mut ObservedRequirements,
    pypi_name: &str,
    specifiers: &VersionSpecifiers,
    source: String,
) {
    let observations = observed.entry(canonical_conda_name(pypi_name)).or_default();
    let observation = ObservedRequirement {
        specifiers: specifiers.clone(),
        source,
    };
    if !observations.contains(&observation) {
        observations.push(observation);
    }
}

/// Canonicalize an intersection without losing its source declarations.
/// `VersionSpecifiers` sorts but does not deduplicate, and parsing a combined
/// string accepts semantically empty intersections. Deduplicate parsed clauses,
/// then use uv-pep440's full PEP 440 range conversion to reject emptiness before
/// any index request is attempted.
fn finalize_observed_requirement(
    pypi_name: &str,
    observations: &[ObservedRequirement],
) -> std::result::Result<VersionSpecifiers, UnsatisfiableRestoreRequirements> {
    // Ordinary bounds are semantically equal across trailing release zeroes
    // (`>=11` == `>=11.0`), but compatible-release/wildcard semantics depend
    // on the written release length and `===` depends on spelling. Use an
    // operator-sensitive key and retain the shortest deterministic spelling
    // for ordinary aliases.
    let mut clauses: BTreeMap<SpecifierDedupeKey, (String, VersionSpecifier)> = BTreeMap::new();
    for specifier in observations
        .iter()
        .flat_map(|observation| observation.specifiers.iter().cloned())
    {
        let rendered = specifier.to_string();
        let key = specifier_dedupe_key(&specifier);
        match clauses.entry(key) {
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
    let specifiers: VersionSpecifiers = clauses
        .into_values()
        .map(|(_, specifier)| specifier)
        .collect();

    // `Ranges` is intentionally not named/re-exported by uv-pep440. An empty
    // release-only conversion yields a full range of the inferred type; the
    // following `Into` uses the full PEP 440 conversion (including pre/post/
    // local and wildcard semantics), not release-only semantics.
    let full = release_specifiers_to_ranges(VersionSpecifiers::empty());
    let range_is_empty = full.intersection(&specifiers.clone().into()).is_empty();
    // uv's ordered Version equality also ignores trailing release zeroes,
    // while arbitrary equality (`===`) compares the preserved spelling. A
    // singleton range therefore cannot distinguish `===1` from `===1.0`;
    // validate every arbitrary-exact candidate against the original clauses.
    let arbitrary_exact_conflict = specifiers.iter().any(|specifier| {
        *specifier.operator() == Operator::ExactEqual && !specifiers.contains(specifier.version())
    });
    if range_is_empty || arbitrary_exact_conflict {
        let sources = observations
            .iter()
            .map(|observation| {
                let spec = if observation.specifiers.is_empty() {
                    "*".to_string()
                } else {
                    observation.specifiers.to_string()
                };
                format!("  - `{spec}` from {}", observation.source)
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Err(UnsatisfiableRestoreRequirements {
            pypi_name: pypi_name.to_string(),
            sources,
        });
    }
    Ok(specifiers)
}

#[derive(Clone, Debug)]
struct RestoreRequestBuilder {
    pypi_name: String,
    bundle_name: String,
    requirements: Vec<ObservedRequirement>,
    constraints: Vec<ObservedRequirement>,
    overrides: Vec<ObservedRequirement>,
    route_preferences: BTreeMap<String, BTreeSet<String>>,
    lock_preferences: BTreeMap<String, BTreeSet<String>>,
}

impl RestoreRequestBuilder {
    fn new(pypi_name: &str) -> Self {
        Self {
            pypi_name: pypi_name.to_string(),
            bundle_name: canonical_conda_name(pypi_name),
            requirements: Vec::new(),
            constraints: Vec::new(),
            overrides: Vec::new(),
            route_preferences: BTreeMap::new(),
            lock_preferences: BTreeMap::new(),
        }
    }

    fn add_input(
        &mut self,
        role: crate::uv_closure::AutoRouteInputRole,
        requirement: ObservedRequirement,
    ) {
        let destination = match role {
            crate::uv_closure::AutoRouteInputRole::Requirement => &mut self.requirements,
            crate::uv_closure::AutoRouteInputRole::Constraint => &mut self.constraints,
            crate::uv_closure::AutoRouteInputRole::Override => &mut self.overrides,
        };
        if !destination.contains(&requirement) {
            destination.push(requirement);
        }
    }

    fn add_preference(
        preferences: &mut BTreeMap<String, BTreeSet<String>>,
        version: String,
        source: String,
    ) {
        preferences.entry(version).or_default().insert(source);
    }

    fn unique_preference(
        pypi_name: &str,
        kind: &str,
        preferences: BTreeMap<String, BTreeSet<String>>,
        conflict_is_error: bool,
    ) -> Result<Option<String>> {
        if preferences.len() > 1 {
            if !conflict_is_error {
                return Ok(None);
            }
            let details = preferences
                .iter()
                .map(|(version, sources)| {
                    format!(
                        "{version} from {}",
                        sources.iter().cloned().collect::<Vec<_>>().join(", ")
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            return Err(anyhow!(
                "joint route validation found conflicting {kind} versions for \
                 PyPI package `{pypi_name}`: {details}"
            ));
        }
        Ok(preferences.into_keys().next())
    }

    fn finish(mut self) -> Result<PypiFetchRequest> {
        // uv overrides replace ordinary dependency requirements, including
        // `Requires-Dist` lines materialized only after the closure solve.
        // Constraints remain additive in either case.
        let mut hard_requirements = if self.overrides.is_empty() {
            std::mem::take(&mut self.requirements)
        } else {
            std::mem::take(&mut self.overrides)
        };
        hard_requirements.extend(std::mem::take(&mut self.constraints));
        let has_requirements = !hard_requirements.is_empty();
        let route_preference = Self::unique_preference(
            &self.pypi_name,
            "prior uv-route",
            std::mem::take(&mut self.route_preferences),
            !has_requirements,
        )?;
        let lock_preference = Self::unique_preference(
            &self.pypi_name,
            "favor-lock",
            std::mem::take(&mut self.lock_preferences),
            false,
        )?;

        // A previously selected uv version is a solver output, not an input
        // requirement. When active wheel metadata declares this dependency,
        // keep the old version only as a soft preference; the index resolver
        // ignores it when it no longer satisfies the newly complete graph.
        // With no declaration available, retain the exact version as the only
        // safe reconstruction of the earlier route.
        let specifiers = if !has_requirements {
            let version = route_preference.as_ref().ok_or_else(|| {
                anyhow!(
                    "joint route validation rejected `{}`, but no active PyPI \
                     requirement or prior uv-route version was available",
                    self.pypi_name
                )
            })?;
            VersionSpecifiers::from_str(&format!("=={version}")).with_context(|| {
                format!(
                    "parsing exact PyPI restore spec for rejected route `{} =={version}`",
                    self.pypi_name
                )
            })?
        } else {
            finalize_observed_requirement(&self.pypi_name, &hard_requirements)?
        };

        Ok(PypiFetchRequest {
            pypi_name: self.pypi_name,
            bundle_name: self.bundle_name,
            specifiers,
            // The uv selection reflects the current closure and therefore
            // outranks an older favor-lock hint. Both remain soft.
            preferred_version: route_preference.or(lock_preference),
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
    let mut last_error = None;
    for index in indexes {
        match fetch_pypi(request.clone(), index.clone()).await {
            Ok(wheel) => return Ok(wheel),
            Err(error) => {
                tracing::debug!(
                    dep = %request.pypi_name,
                    index = %index,
                    error = %format!("{error:#}"),
                    "auto-bundle PyPI fetch failed on this index"
                );
                last_error = Some(error);
            }
        }
    }
    Err(last_error
        .unwrap_or_else(|| anyhow!("no PyPI index configured for `{}`", request.pypi_name))
        .context(failure_context))
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

fn expand_name_map_groups(names: &mut HashSet<String>, name_map: &BTreeMap<String, String>) {
    loop {
        let mut changed = false;
        for (pypi_name, conda_name) in name_map {
            let pypi_name = canonical_conda_name(pypi_name);
            let conda_name = canonical_conda_name(conda_name);
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
/// Once validated routing chooses PyPI, resolution failure is fatal. The sole
/// exception is a semantically impossible restore with a precise workspace
/// conda owner and a positively solved fixed baseline; that dependency is
/// dropped rather than returned to a rejected generated conda route.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn auto_bundle_transitives(
    bundle: &mut Bundle,
    indexes: &[String],
    target: &crate::pypi::WheelTarget,
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
    // Exact Rule-1 ownership authority, already policy-filtered by the caller.
    workspace_ownership: &super::WorkspaceRouteOwnership,
    // Shared Rule-1/Rule-2 workspace-aware conda co-solve oracle. Rule 2
    // finalizes mutable uv routes only after the merged bundle's actual
    // emitted sibling constraints are known.
    conda_co_solve: &super::CondaCoSolveContext,
) -> Result<()> {
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
    let target = target.clone();
    let fetch_target = target.clone();
    let download_dir = download_dir.to_path_buf();
    let relax = config.relax;
    let fetch_pypi = move |request: PypiFetchRequest, index: String| {
        let target = fetch_target.clone();
        let download_dir = download_dir.clone();
        async move {
            let (resolved_url, metadata, _resolved_index, sdist_prov) = super::bfs_fetch_pypi(
                &request.pypi_name,
                &request.specifiers,
                &index,
                &target,
                &download_dir,
                relax,
                request.preferred_version.as_deref(),
            )
            .await?;
            let (upstream_url, sdist_source) =
                super::bfs_fetch_provenance(&resolved_url, sdist_prov);
            Ok(ResolvedWheel {
                pypi_name: request.bundle_name,
                url: resolved_url,
                upstream_url,
                git_source: None,
                sdist_source,
                metadata,
                extras_requested: vec![],
                auto_data: None,
                auto_data_dedup_skipped_root: None,
            })
        }
    };
    let channels_consulted = conda_co_solve.channels_consulted();
    auto_bundle_transitives_with(
        bundle,
        indexes,
        &target,
        config,
        locked_closure,
        favor_lock_prefs,
        uv_closure_wheels,
        workspace_ownership,
        &probe_many,
        &co_solve,
        &fetch_pypi,
        &channels_consulted,
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
    workspace_ownership: &super::WorkspaceRouteOwnership,
    probe_many: &P,
    co_solve: &C,
    fetch_pypi: &X,
    channels_consulted: &[String],
) -> Result<()>
where
    P: Fn(Vec<(String, String)>) -> PF,
    PF: Future<Output = Vec<crate::probe::ProbeResult>>,
    C: Fn(Vec<crate::uv_closure::CondaRouteSpec>) -> CF,
    CF: Future<Output = crate::uv_closure::CoInstallVerdict>,
    X: Fn(PypiFetchRequest, String) -> XF,
    XF: Future<Output = Result<ResolvedWheel>>,
{
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
                if let Some((name, version)) = pep508_exact_base_dep(raw, &marker_env)? {
                    let specifiers = VersionSpecifiers::from_str(&format!("=={version}"))
                        .with_context(|| {
                            format!("parsing exact auto-bundle requirement `{name}=={version}`")
                        })?;
                    observe_requirement(
                        &mut observed_requirements,
                        &name,
                        &specifiers,
                        format!(
                            "wheel `{}=={}` Requires-Dist `{raw}`",
                            wheel.metadata.name, wheel.metadata.version
                        ),
                    );
                    let conda_name = canonical_conda_name(&name);
                    if !seen_candidate.insert(conda_name) {
                        continue;
                    }
                    candidates.push((name, version));
                } else if let Some((name, specs)) = pep508_loose_base_dep(raw, &marker_env)? {
                    observe_requirement(
                        &mut observed_requirements,
                        &name,
                        &specs,
                        format!(
                            "wheel `{}=={}` Requires-Dist `{raw}`",
                            wheel.metadata.name, wheel.metadata.version
                        ),
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
            if jointly_unroute_unsolvable(
                bundle,
                &mut provisional_metadata_routes,
                &observed_requirements,
                indexes,
                target,
                config,
                workspace_ownership,
                co_solve,
                fetch_pypi,
                channels_consulted,
            )
            .await?
            {
                // Rejected routes were restored as wheels or safely dropped
                // to a precise workspace owner. Scan any restored metadata
                // before accepting the remaining conda routes.
                continue;
            }
            break;
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
        let prefer_pairs: Vec<(String, String)> = candidates
            .iter()
            .filter(|(name, _)| !is_closure_member(name))
            .filter(|(name, _)| prefer_conda_match(&canonical_conda_name(name), &config.name_map))
            .map(|(name, version)| {
                let conda_name = canonical_conda_name(name);
                (config.name_map[&conda_name].clone(), format!("=={version}"))
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
            } else if prefer_conda_match(&conda_name, &config.name_map) {
                // Probe the workspace's conda channels for whether the
                // spec retread would emit is actually satisfiable. If
                // ANY channel has a matching candidate, keep on conda.
                // Only a concrete match may route to conda. Empty,
                // unsatisfied, or indecisive results stay on PyPI.
                let conda_target_name = config.name_map[&conda_name].clone();
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
            .map(|(name, specs)| {
                let conda_name = canonical_conda_name(name);
                let target_name = config
                    .name_map
                    .get(&conda_name)
                    .cloned()
                    .unwrap_or(conda_name);
                (target_name, conda_probe_spec(specs))
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
            let target_name = config
                .name_map
                .get(&conda_name)
                .cloned()
                .unwrap_or_else(|| conda_name.clone());
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
        let fetched: Vec<Result<(String, String, ResolvedWheel)>> = {
            use futures::stream::{self, StreamExt};
            let indexes_ref = indexes;
            stream::iter(to_fetch)
                .map(
                    |(name, version, conda_name, specifiers, preferred_ver)| async move {
                        let request = PypiFetchRequest {
                            pypi_name: name.clone(),
                            bundle_name: conda_name,
                            specifiers: specifiers.clone(),
                            preferred_version: preferred_ver,
                        };
                        let failure_context = format!(
                            "auto-bundle: no PyPI index could resolve `{name}{specifiers}` after conda routing was refused"
                        );
                        fetch_from_index_chain(
                            indexes_ref,
                            request,
                            fetch_pypi,
                            failure_context,
                        )
                        .await
                        .map(|wheel| (name, version, wheel))
                    },
                )
                .buffered(8)
                .collect()
                .await
        };
        for result in fetched {
            let (name, version, wheel) = result?;
            tracing::info!(
                dep = %name,
                version = %version,
                "auto-bundled into {}",
                bundle.conda_name,
            );
            bundle.extras.push(wheel);
            added_any = true;
        }

        // Scan metadata from every wheel fetched this round before finalizing
        // routes. Otherwise a later wheel can tighten a requirement after its
        // conda route has already been rejected and restored from PyPI.
        if !added_any
            && jointly_unroute_unsolvable(
                bundle,
                &mut provisional_metadata_routes,
                &observed_requirements,
                indexes,
                target,
                config,
                workspace_ownership,
                co_solve,
                fetch_pypi,
                channels_consulted,
            )
            .await?
        {
            added_any = true;
        }

        // Loop again only if we added at least one wheel; the new
        // wheels' Requires-Dist may need further auto-bundling.
        if !added_any {
            break;
        }
    }
    Ok(())
}

/// Does this canonical conda dependency disappear if every provisional
/// origin in its group is restored to PyPI? A remaining dependency has a
/// fixed/user-owned origin and must not be offered to Rule 2 as mutable.
fn route_group_is_fully_mutable(
    bundle: &Bundle,
    metadata_routes: &ProvisionalMetadataRoutes,
    conda_name: &str,
    config: &RetreadConfig,
    target: &crate::pypi::WheelTarget,
) -> Result<bool> {
    let mut trial = bundle.clone();
    let mut restored_pypi_names: BTreeSet<String> = trial
        .auto_routed
        .iter()
        .filter(|route| canonical_conda_name(&route.route.conda_name) == conda_name)
        .map(|route| route.route.pypi_name.clone())
        .collect();
    trial
        .auto_routed
        .retain(|route| canonical_conda_name(&route.route.conda_name) != conda_name);
    if let Some(origins) = metadata_routes.get(conda_name) {
        restored_pypi_names.extend(origins.iter().map(|origin| origin.pypi_name.clone()));
    }

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

    Ok(super::emitted_bundle_route_specs(&trial, config, target)?
        .into_iter()
        .all(|route| canonical_conda_name(&route.conda_name) != conda_name))
}

/// Finalize every provisional conda route against the exact dependency set
/// this bundle would emit. Both uv routes and metadata-probe routes are
/// mutable until this check succeeds. Rejected routes are restored through
/// the same ordered PyPI fallback chain before the bundle is changed.
async fn jointly_unroute_unsolvable<C, CF, X, XF>(
    bundle: &mut Bundle,
    metadata_routes: &mut ProvisionalMetadataRoutes,
    observed_requirements: &ObservedRequirements,
    indexes: &[String],
    target: &crate::pypi::WheelTarget,
    config: &RetreadConfig,
    workspace_ownership: &super::WorkspaceRouteOwnership,
    co_solve: &C,
    fetch_pypi: &X,
    channels_consulted: &[String],
) -> Result<bool>
where
    C: Fn(Vec<crate::uv_closure::CondaRouteSpec>) -> CF,
    CF: Future<Output = crate::uv_closure::CoInstallVerdict>,
    X: Fn(PypiFetchRequest, String) -> XF,
    XF: Future<Output = Result<ResolvedWheel>>,
{
    if bundle.auto_routed.is_empty() && metadata_routes.is_empty() {
        return Ok(false);
    }

    let emitted = super::emitted_bundle_route_specs(bundle, config, target)?;
    let emitted_by_conda: BTreeMap<String, crate::uv_closure::CondaRouteSpec> = emitted
        .iter()
        .cloned()
        .map(|route| (canonical_conda_name(&route.conda_name), route))
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
        if route_group_is_fully_mutable(bundle, metadata_routes, &conda_name, config, target)? {
            mutable_conda_names.insert(conda_name);
        }
    }
    let mutable_candidates: Vec<_> = mutable_conda_names
        .iter()
        .filter_map(|name| emitted_by_conda.get(name).cloned())
        .collect();
    if mutable_candidates.is_empty() {
        return Ok(false);
    }

    let mutable_keys: HashSet<String> = mutable_candidates
        .iter()
        .map(|route| canonical_conda_name(&route.conda_name))
        .collect();
    let fixed: Vec<_> = emitted
        .into_iter()
        .filter(|route| !mutable_keys.contains(&canonical_conda_name(&route.conda_name)))
        .collect();
    let selection = crate::uv_closure::select_jointly_solvable_routes(
        fixed,
        mutable_candidates.clone(),
        co_solve,
    )
    .await;
    // `Some` proves the fixed workspace baseline solved positively. A `None`
    // result is deliberately not enough authority to drop even a precisely
    // owned dependency: Rule 1 abstains when that baseline is indeterminate.
    let workspace_drop_authorized = selection.is_some();
    // Rule 2 is fail-closed: an unsatisfiable/indeterminate baseline cannot
    // authorize any mutable conda route.
    let rejected = selection
        .map(|selection| selection.rejected)
        .unwrap_or(mutable_candidates);
    if rejected.is_empty() {
        return Ok(false);
    }
    let rejected_keys: BTreeSet<String> = rejected
        .iter()
        .map(|route| canonical_conda_name(&route.conda_name))
        .collect();

    let rejected_specs: BTreeMap<String, String> = rejected
        .iter()
        .map(|route| (canonical_conda_name(&route.conda_name), route.spec.clone()))
        .collect();
    let mut restore_requests: BTreeMap<String, RestoreRequestBuilder> = BTreeMap::new();
    // Route-specific identity proof for direct conda ownership. Global
    // same-name inference remains forbidden; only the concrete rejected route
    // edges that produced each restore request participate.
    let mut restore_conda_routes: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut audit_origins: BTreeSet<(String, String)> = BTreeSet::new();
    for route in &bundle.auto_routed {
        let conda_name = canonical_conda_name(&route.route.conda_name);
        if !rejected_keys.contains(&conda_name) {
            continue;
        }
        let key = canonical_conda_name(&route.route.pypi_name);
        restore_conda_routes
            .entry(key.clone())
            .or_default()
            .insert(conda_name.clone());
        let request = restore_requests
            .entry(key)
            .or_insert_with(|| RestoreRequestBuilder::new(&route.route.pypi_name));
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
            let requirement = ObservedRequirement {
                specifiers,
                source: input.source.clone(),
            };
            request.add_input(input.role, requirement);
        }
        if let Some(requirements) =
            observed_requirements.get(&canonical_conda_name(&route.route.pypi_name))
        {
            for requirement in requirements {
                if !request.requirements.contains(requirement) {
                    request.requirements.push(requirement.clone());
                }
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
                restore_conda_routes
                    .entry(key.clone())
                    .or_default()
                    .insert(canonical_conda_name(&origin.conda_name));
                let request = restore_requests
                    .entry(key.clone())
                    .or_insert_with(|| RestoreRequestBuilder::new(&origin.pypi_name));
                let requirements = observed_requirements.get(&key).ok_or_else(|| {
                    anyhow!(
                        "joint route validation rejected metadata route `{} -> {}`, \
                         but no active Requires-Dist provenance was recorded",
                        origin.pypi_name,
                        origin.conda_name
                    )
                })?;
                for requirement in requirements {
                    if !request.requirements.contains(requirement) {
                        request.requirements.push(requirement.clone());
                    }
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
        return Err(anyhow!(
            "joint route validation rejected {:?}, but no PyPI route provenance was available",
            rejected_keys
        ));
    }
    // Finalize every requirement (dedupe + semantic satisfiability) before
    // the first index request, so one genuine conflict cannot be obscured by
    // a network error for another rejected route. Only the exact semantic
    // impossibility proven here can fall back to a precise workspace owner;
    // preference/provenance/parse failures remain fatal.
    let mut finalized_restore_requests = Vec::new();
    let mut workspace_drops = BTreeSet::new();
    for (pypi_key, request) in restore_requests {
        let owns_every_concrete_conda_route =
            restore_conda_routes
                .get(&pypi_key)
                .is_some_and(|conda_names| {
                    !conda_names.is_empty()
                        && conda_names
                            .iter()
                            .all(|name| workspace_ownership.conda_names.contains(name))
                });
        let workspace_owns_request = !workspace_ownership.excluded_pypi_names.contains(&pypi_key)
            && (workspace_ownership.pypi_names.contains(&pypi_key)
                || owns_every_concrete_conda_route);
        match request.finish() {
            Ok(request) => finalized_restore_requests.push(request),
            Err(error)
                if workspace_drop_authorized
                    && workspace_owns_request
                    && error
                        .downcast_ref::<UnsatisfiableRestoreRequirements>()
                        .is_some() =>
            {
                workspace_drops.insert(pypi_key);
            }
            Err(error) => {
                // An impossible restore that we decline to drop is fatal, but it
                // must never be silent: name the dep and the guard that blocked
                // the drop, appended to the v4.6.6 diagnostic verbatim.
                let Some(unsatisfiable) = error.downcast_ref::<UnsatisfiableRestoreRequirements>()
                else {
                    return Err(error);
                };
                let empty = BTreeSet::new();
                let reasons = drop_decline_reasons(
                    &pypi_key,
                    workspace_ownership,
                    restore_conda_routes.get(&pypi_key).unwrap_or(&empty),
                    workspace_drop_authorized,
                );
                tracing::warn!(
                    pypi = %pypi_key,
                    reasons = ?reasons,
                    "PyPI restore is impossible and the dependency was NOT dropped; \
                     the bundle cannot be resolved",
                );
                return Err(anyhow!(
                    "{unsatisfiable}\n\nretread did NOT drop `{pypi_key}` (dropping it would \
                     let the consuming workspace's own conda copy satisfy every wheel above) \
                     because:\n{}",
                    reasons
                        .iter()
                        .map(|reason| format!("  - {reason}"))
                        .collect::<Vec<_>>()
                        .join("\n"),
                ));
            }
        }
    }

    // Fetch every wheel before changing routing. A missing index candidate
    // therefore fails without leaving the bundle partially un-routed.
    let mut restored_wheels = Vec::with_capacity(finalized_restore_requests.len());
    for request in finalized_restore_requests {
        let requirement = request.specifiers.to_string();
        let failure_context = format!(
            "joint route validation kept `{}` on PyPI, but no configured index could fetch `{}`",
            request.pypi_name, requirement
        );
        restored_wheels
            .push(fetch_from_index_chain(indexes, request, fetch_pypi, failure_context).await?);
    }

    let mut trial = bundle.clone();
    trial
        .auto_routed
        .retain(|route| !rejected_keys.contains(&canonical_conda_name(&route.route.conda_name)));
    trial.auto_dropped.extend(workspace_drops.iter().cloned());
    trial.extras.extend(restored_wheels);
    for (pypi_name, conda_name) in audit_origins {
        let conda_key = canonical_conda_name(&conda_name);
        let dropped = workspace_drops.contains(&canonical_conda_name(&pypi_name));
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
            routing_decision: if dropped {
                "joint-co-solve-rejected-workspace-owned-drop".into()
            } else {
                "joint-co-solve-rejected-to-pypi".into()
            },
        });
        if dropped {
            tracing::warn!(
                pypi = %pypi_name,
                conda = %conda_name,
                "individually valid conda route rejected by final joint solve; \
                 PyPI restore is impossible, so dropping the dependency owned \
                 by every precise consuming workspace environment",
            );
        } else {
            tracing::warn!(
                pypi = %pypi_name,
                conda = %conda_name,
                "individually valid conda route rejected by final joint solve; restoring PyPI wheel",
            );
        }
    }

    let still_emitted: Vec<String> = super::emitted_bundle_route_specs(&trial, config, target)?
        .into_iter()
        .map(|route| canonical_conda_name(&route.conda_name))
        .filter(|name| rejected_keys.contains(name))
        .collect();
    if !still_emitted.is_empty() {
        return Err(anyhow!(
            "joint route validation restored PyPI wheels, but rejected conda routes remain emitted: {}",
            still_emitted.join(", ")
        ));
    }

    *bundle = trial;
    metadata_routes.retain(|conda_name, _| !rejected_keys.contains(conda_name));
    Ok(true)
}

/// Returns Some((name, exact_version)) if `raw` is a base dep (no
/// extras marker) with a single `== X.Y.Z` specifier. Returns None for
/// extras-gated deps, ranges, ~=, or URL deps.
fn pep508_exact_base_dep(
    raw: &str,
    marker_env: &MarkerEnvironment,
) -> Result<Option<(String, String)>> {
    let req: uv_pep508::Requirement = uv_pep508::Requirement::from_str(raw)
        .map_err(|e| anyhow!("parsing requirement `{raw}`: {e}"))?;
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
    let req: uv_pep508::Requirement = uv_pep508::Requirement::from_str(raw)
        .map_err(|e| anyhow!("parsing requirement `{raw}`: {e}"))?;
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

/// (wheel URL, parsed METADATA, index to recurse with, optional sdist
/// provenance) for one PyPI-form BFS item fetched in the level loop's
/// phase 2. The 4th element is `Some` only when the item was built from
/// a PyPI sdist (no compatible wheel on the index); `None` for normal
/// index-wheel fetches.
pub(crate) type BfsFetched = (url::Url, WheelMetadata, String, Option<super::SdistProv>);

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
        index: String,
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
    index: &str,
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
                        source: extra_dep_source_to_pending(dep.source.clone(), index),
                        extras: dep.extras.clone(),
                    };
                    check_locked_seen!(&dn, pending, state);
                }
                work.push_back(Pending {
                    pypi_name: dep.name,
                    source: extra_dep_source_to_pending(dep.source, index),
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
                    source: extra_dep_source_to_pending(dep.source.clone(), index),
                    extras: dep.extras.clone(),
                };
                check_locked_seen!(&dn, pending, state);
            }
            work.push_back(Pending {
                pypi_name: dep.name,
                source: extra_dep_source_to_pending(dep.source, index),
                extras: dep.extras,
            });
        }
    }
    Ok(())
}

fn extra_dep_source_to_pending(src: ExtraDepSource, default_index: &str) -> PendingSource {
    match src {
        ExtraDepSource::Pypi(specifiers) => PendingSource::Pypi {
            specifiers,
            index: default_index.to_string(),
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
    let req: uv_pep508::Requirement = uv_pep508::Requirement::from_str(raw)
        .map_err(|e| anyhow!("parsing requirement `{raw}`: {e}"))?;

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
    tokio::task::spawn_blocking(move || crate::wheel::read_metadata(&path))
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
///   2. index advertised a PEP 658/714 sidecar AND a sha256 fragment
///      -> fetch the KB-sized `.metadata` sidecar instead of the
///      potentially-GB wheel (the fragment hash stands in for the
///      computed one the recipe pins);
///   3. full download via fetch_and_parse (unchanged behavior).
///
/// pypi.org serves sidecars; pypi.nvidia.com and static GitHub-Pages
/// indexes do not (measured 2026-06-10), so NVIDIA-index-only wheels
/// still take path 3 on a cold cache.
pub(crate) async fn metadata_preferring_sidecar(
    resolved: &pypi::ResolvedWheel,
    download_dir: &Path,
) -> Result<WheelMetadata> {
    if let Ok(filename) = crate::wheel::wheel_filename_from_url(&resolved.url)
        && download_dir.join(&filename).exists()
    {
        return fetch_and_parse(&resolved.url, resolved.sha256.as_deref(), download_dir).await;
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
    fetch_and_parse(&resolved.url, resolved.sha256.as_deref(), download_dir).await
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
    let req: uv_pep508::Requirement = uv_pep508::Requirement::from_str(raw)
        .map_err(|e| anyhow!("parsing extra requirement `{raw}`: {e}"))?;

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
    use std::sync::{Arc, Mutex};
    use url::Url;

    fn test_config() -> RetreadConfig {
        serde_json::from_value(serde_json::json!({
            "retread-wheels": {},
            "retread-relax": "none",
            "retread-conda-deps": ["libtorch"]
        }))
        .unwrap()
    }

    fn workspace_ownership(direct_conda_names: &[&str]) -> super::super::WorkspaceRouteOwnership {
        super::super::WorkspaceRouteOwnership {
            conda_names: direct_conda_names
                .iter()
                .map(|name| canonical_conda_name(name))
                .collect(),
            ..Default::default()
        }
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
            metadata: WheelMetadata {
                name: metadata_name.to_string(),
                version: version.to_string(),
                requires_dist: requires_dist
                    .iter()
                    .map(|requirement| (*requirement).to_string())
                    .collect(),
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
        }
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
            .any(|route| canonical_conda_name(&route.conda_name) == "numpy")
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
            },
            deps_from_floor: false,
        }
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
        bundle
    }

    #[tokio::test]
    async fn joint_unroute_drops_workspace_owned_unsatisfiable_numpy_restore() {
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _index: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Err(anyhow!("owned impossible restore must not fetch PyPI")) }
            }
        };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let mut bundle = holosoma_numpy_conflict_bundle();

        auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &test_config(),
            None,
            None,
            None,
            &workspace_ownership(&["numpy"]),
            &validated_probe,
            &reject_numpy_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
        )
        .await
        .unwrap();

        assert_eq!(
            fetch_calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "a semantically impossible owned restore must never reach an index"
        );
        assert_eq!(bundle.auto_dropped, HashSet::from(["numpy".to_string()]));
        assert!(
            bundle
                .all_wheels()
                .all(|wheel| canonical_conda_name(&wheel.pypi_name) != "numpy"),
            "no NumPy wheel may be restored over the workspace's conda owner"
        );
        let emitted =
            super::super::emitted_bundle_route_specs(&bundle, &test_config(), &target).unwrap();
        assert!(
            emitted
                .iter()
                .all(|route| canonical_conda_name(&route.conda_name) != "numpy"),
            "the generated pack must not re-emit the workspace-owned NumPy route: {emitted:?}"
        );
        assert!(bundle.probe_decisions.iter().any(|decision| {
            decision.pypi_name == "numpy"
                && decision.routing_decision == "joint-co-solve-rejected-workspace-owned-drop"
        }));
    }

    #[tokio::test]
    async fn joint_unroute_errors_for_unowned_unsatisfiable_numpy_with_all_sources() {
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _index: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Err(anyhow!("unowned semantic conflict must fail before fetch")) }
            }
        };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let mut bundle = holosoma_numpy_conflict_bundle();

        let error = auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &test_config(),
            None,
            None,
            None,
            &workspace_ownership(&[]),
            &validated_probe,
            &reject_numpy_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
        )
        .await
        .unwrap_err();

        let message = format!("{error:#}");
        assert!(
            message.contains("cannot restore `numpy` to PyPI"),
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
        assert!(bundle.auto_dropped.is_empty());
        assert_eq!(
            fetch_calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "semantic conflicts must fail before any index request"
        );

        // The decline must NAME the dep and the guard. A silent fall-through to
        // the v4.6.6 text is exactly how a no-op fix survived a publish.
        assert!(
            message.contains("retread did NOT drop `numpy`"),
            "the error must say the drop was declined, and for which dep:\n{message}"
        );
        assert!(
            message.contains("conda-own") && message.contains("EMPTY"),
            "the error must name the ownership guard that blocked the drop:\n{message}"
        );
    }

    /// The ownership guard is evaluated PER DEPENDENCY, not all-or-nothing over
    /// the pack. A second rejected conda route that the workspace does NOT own
    /// (here `pillow`, which restores to PyPI perfectly well) must not block the
    /// owned, impossible-to-restore `numpy` from being dropped.
    #[tokio::test]
    async fn joint_unroute_drops_owned_dep_alongside_unowned_rejected_route() {
        let fetched: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let fetch = {
            let fetched = Arc::clone(&fetched);
            move |request: PypiFetchRequest, _index: String| {
                let fetched = Arc::clone(&fetched);
                async move {
                    fetched.lock().unwrap().push(request.pypi_name.clone());
                    Ok(test_wheel("pillow", "pillow", "10.4.0", &[]))
                }
            }
        };
        // Reject BOTH routes jointly; only `numpy` is workspace-owned.
        let reject_both = |routes: Vec<crate::uv_closure::CondaRouteSpec>| async move {
            if routes.iter().any(|route| {
                matches!(
                    canonical_conda_name(&route.conda_name).as_str(),
                    "numpy" | "pillow"
                )
            }) {
                crate::uv_closure::CoInstallVerdict::Unsat(vec![
                    "test fixture rejects both generated routes".to_string(),
                ])
            } else {
                crate::uv_closure::CoInstallVerdict::Sat
            }
        };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let mut bundle = holosoma_numpy_conflict_bundle();
        // `pillow` arrives as a uv auto-route with a perfectly restorable spec.
        bundle.auto_routed = vec![pillow_auto_route("10.4.0")];

        auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &test_config(),
            None,
            None,
            None,
            // NumPy owned; pillow deliberately NOT owned.
            &workspace_ownership(&["numpy"]),
            &validated_probe,
            &reject_both,
            &fetch,
            &["conda-forge/linux-64".to_string()],
        )
        .await
        .unwrap();

        assert_eq!(
            bundle.auto_dropped,
            HashSet::from(["numpy".to_string()]),
            "the owned dep must drop even though a sibling rejected route is unowned"
        );
        assert_eq!(
            *fetched.lock().unwrap(),
            vec!["pillow".to_string()],
            "the unowned rejected route must still restore to PyPI, and the owned \
             impossible one must never reach an index"
        );
    }

    #[tokio::test]
    async fn joint_unroute_abstains_from_owned_drop_when_fixed_baseline_is_indeterminate() {
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _index: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Err(anyhow!("indeterminate baseline must fail before fetch")) }
            }
        };
        let indeterminate = |_| async {
            crate::uv_closure::CoInstallVerdict::Skipped(
                "test fixture has no authoritative baseline solve".to_string(),
            )
        };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let mut bundle = holosoma_numpy_conflict_bundle();

        let error = auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &test_config(),
            None,
            None,
            None,
            &workspace_ownership(&["numpy"]),
            &validated_probe,
            &indeterminate,
            &fetch,
            &["conda-forge/linux-64".to_string()],
        )
        .await
        .unwrap_err();

        let message = format!("{error:#}");
        assert!(message.contains("mutually unsatisfiable"), "{message}");
        assert!(
            bundle.auto_dropped.is_empty(),
            "ownership without a positively solved fixed baseline must abstain"
        );
        assert_eq!(fetch_calls.load(std::sync::atomic::Ordering::SeqCst), 0);

        // Rule-1 abstain-on-ambiguity is preserved -- but it must SAY so, and
        // name the baseline as the guard rather than blaming ownership.
        assert!(
            message.contains("retread did NOT drop `numpy`"),
            "{message}"
        );
        assert!(
            message.contains("workspace conda baseline did not solve"),
            "the abstention must name the baseline guard:\n{message}"
        );
    }

    #[tokio::test]
    async fn joint_validation_keeps_cosolvable_metadata_route_on_conda() {
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _index: String| {
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
            &workspace_ownership(&[]),
            &validated_probe,
            &cosolvable,
            &fetch,
            &["conda-forge/linux-64".to_string()],
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
                .any(|route| canonical_conda_name(&route.conda_name) == "numpy"),
            "a jointly solvable metadata route must remain on conda: {emitted:?}"
        );
    }

    #[tokio::test]
    async fn joint_unroute_deduplicates_repeated_requires_dist_clauses() {
        let ordinary_aliases = [">=11.0,>=11,<11.1.0,<11.1"]
            .into_iter()
            .map(|raw| ObservedRequirement {
                specifiers: VersionSpecifiers::from_str(raw).unwrap(),
                source: format!("test source `{raw}`"),
            })
            .collect::<Vec<_>>();
        let ordinary =
            finalize_observed_requirement("ordinary-aliases", &ordinary_aliases).unwrap();
        assert_eq!(
            ordinary.len(),
            2,
            "ordinary trailing-zero aliases must collapse: {ordinary}"
        );
        let length_sensitive = ["~=1.0", "~=1.0.0"]
            .into_iter()
            .map(|raw| ObservedRequirement {
                specifiers: VersionSpecifiers::from_str(raw).unwrap(),
                source: format!("test source `{raw}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            finalize_observed_requirement("length-sensitive", &length_sensitive)
                .unwrap()
                .len(),
            2,
            "semantic length-sensitive clauses must not be deduplicated"
        );
        let arbitrary_exact = ["===1", "===1.0"]
            .into_iter()
            .map(|raw| ObservedRequirement {
                specifiers: VersionSpecifiers::from_str(raw).unwrap(),
                source: format!("test source `{raw}`"),
            })
            .collect::<Vec<_>>();
        assert!(
            finalize_observed_requirement("arbitrary-exact", &arbitrary_exact)
                .unwrap_err()
                .to_string()
                .contains("mutually unsatisfiable"),
            "arbitrary equality must preserve trailing-zero spelling"
        );

        let requests = Arc::new(Mutex::new(Vec::<PypiFetchRequest>::new()));
        let fetch = {
            let requests = Arc::clone(&requests);
            move |request: PypiFetchRequest, _index: String| {
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

        auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &test_config(),
            None,
            None,
            None,
            &workspace_ownership(&[]),
            &validated_probe,
            &reject_every_mutable_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
        )
        .await
        .unwrap();

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
        let context = super::super::CondaCoSolveContext::new(
            Some(&manifest),
            Some(&tmp),
            &tmp.join("hover-pack"),
            &target,
            &[],
            "hover-pack",
        );
        assert_eq!(
            context.workspace_deps.get("pillow"),
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
            move |request: PypiFetchRequest, _index: String| {
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
            &workspace_ownership(&[]),
            &validated_probe,
            &scoped_reject,
            &fetch,
            &["conda-forge/linux-64".to_string()],
        )
        .await
        .unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(
            solve_calls
                .lock()
                .unwrap()
                .iter()
                .any(|routes| routes.iter().any(|route| route.pypi_name == "pillow")),
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
            move |_request: PypiFetchRequest, _index: String| {
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
        deps_from_route.deps_from_floor = true;
        deps_from_route.route.input_requirements.push(
            crate::uv_closure::AutoRouteInputRequirement {
                specifiers: "==12.3.0".to_string(),
                source: "uv root requirement `pillow==12.3.0`".to_string(),
                role: crate::uv_closure::AutoRouteInputRole::Requirement,
            },
        );
        bundle.auto_routed.push(deps_from_route);
        let target = crate::pypi::WheelTarget::for_subdir("3.10", "linux-64");

        let error = auto_bundle_transitives_with(
            &mut bundle,
            &[crate::workspace::DEFAULT_PYPI_INDEX.to_string()],
            &target,
            &test_config(),
            None,
            None,
            None,
            &workspace_ownership(&[]),
            &validated_probe,
            &reject_every_mutable_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
        )
        .await
        .unwrap_err();

        let message = format!("{error:#}");
        assert!(message.contains("pillow"), "{message}");
        assert!(message.contains("mutually unsatisfiable"), "{message}");
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
    async fn joint_unroute_restores_compatible_pypi_requirement() {
        let mut overridden = RestoreRequestBuilder::new("overridden");
        overridden.requirements.push(ObservedRequirement {
            specifiers: VersionSpecifiers::from_str("<2").unwrap(),
            source: "wheel metadata requires overridden<2".to_string(),
        });
        overridden.add_input(
            crate::uv_closure::AutoRouteInputRole::Override,
            ObservedRequirement {
                specifiers: VersionSpecifiers::from_str("==3").unwrap(),
                source: "uv override requirement `overridden==3`".to_string(),
            },
        );
        overridden.add_input(
            crate::uv_closure::AutoRouteInputRole::Constraint,
            ObservedRequirement {
                specifiers: VersionSpecifiers::from_str("<4").unwrap(),
                source: "uv constraint `overridden<4`".to_string(),
            },
        );
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

        let mut soft_hints = RestoreRequestBuilder::new("soft-hints");
        soft_hints.requirements.push(ObservedRequirement {
            specifiers: VersionSpecifiers::from_str(">=1").unwrap(),
            source: "wheel `root==1` Requires-Dist `soft-hints>=1`".to_string(),
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

        let requests = Arc::new(Mutex::new(Vec::<PypiFetchRequest>::new()));
        let fetch = {
            let requests = Arc::clone(&requests);
            move |request: PypiFetchRequest, _index: String| {
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
            &workspace_ownership(&[]),
            &validated_probe,
            &reject_every_mutable_route,
            &fetch,
            &["conda-forge/linux-64".to_string()],
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
                    let specs: Vec<String> =
                        routes.iter().map(|route| route.match_spec()).collect();
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
            move |request: PypiFetchRequest, index: String| {
                let fetch_calls = Arc::clone(&fetch_calls);
                async move {
                    assert_eq!(request.pypi_name, "pyglet");
                    assert_eq!(
                        request.specifiers,
                        VersionSpecifiers::from_str("<2").unwrap()
                    );
                    fetch_calls.lock().unwrap().push(index.clone());
                    if index.trim_end_matches('/')
                        != crate::workspace::DEFAULT_PYPI_INDEX.trim_end_matches('/')
                    {
                        return Err(anyhow!("package absent from {index}"));
                    }
                    Ok(test_wheel(
                        &request.bundle_name,
                        &request.pypi_name,
                        "1.5.27",
                        &[],
                    ))
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
            &workspace_ownership(&[]),
            &probe,
            &solve,
            &fetch,
            &["conda-forge/linux-64".to_string()],
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
                .all(|route| canonical_conda_name(&route.conda_name) != "pyglet"),
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
        let workspace_indexes = vec![
            crate::workspace::DEFAULT_PYPI_INDEX.to_string(),
            "https://pypi.nvidia.com".to_string(),
        ];
        assert_eq!(
            super::super::auto_bundle_group_index_chain(
                [&source_entry, &private_entry],
                &workspace_indexes,
            ),
            workspace_indexes,
            "an implicit-default first entry must not be skipped for a later private index"
        );

        let explicit_private_chain =
            super::super::auto_bundle_group_index_chain([&private_entry], &workspace_indexes);
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
            move |request: PypiFetchRequest, index: String| {
                let calls = Arc::clone(&calls);
                async move {
                    assert_eq!(request.pypi_name, "flatdict");
                    assert_eq!(
                        request.specifiers,
                        VersionSpecifiers::from_str(">=4.0.1,<4.1").unwrap()
                    );
                    calls.lock().unwrap().push(index.clone());
                    if index.trim_end_matches('/')
                        != crate::workspace::DEFAULT_PYPI_INDEX.trim_end_matches('/')
                    {
                        return Err(anyhow!("flatdict absent from {index}"));
                    }
                    Ok(test_wheel(
                        &request.bundle_name,
                        &request.pypi_name,
                        "4.0.1",
                        &[],
                    ))
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
            &workspace_ownership(&[]),
            &probe,
            &solve,
            &fetch,
            &["conda-forge/linux-64".to_string()],
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
            "https://pypi.org/simple/",
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
            "https://pypi.org/simple/",
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
            "https://pypi.org/simple/",
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
}
