//! Auto-bundle BFS: prefer-conda routing and PyPI-only transitive
//! packaging into the bundle.
//!
//! Extracted from handler.rs (Phase 0b.3). All functions are behavior-
//! identical whole-function moves; no logic changes.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::ffi::OsStr;
use std::future::Future;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, anyhow};
use rattler_conda_types::ChannelUrl;
use uv_pep508::MarkerEnvironment;
#[cfg(test)]
use uv_pep508::uv_pep440::Version;
use uv_pep508::uv_pep440::VersionSpecifiers;

use crate::config::RetreadConfig;
use crate::constraint::{Constraint, Provenance, finalize};
use crate::pypi;
use crate::relax::{
    CondaName, CondaTarget, NameMap, PypiKey, canonical_conda_name, default_marker_env,
    marker_env_for,
};
use crate::wheel::WheelMetadata;

use super::resolve_state::ResolveState;
use super::{Bundle, DEFAULT_PYTHON, PypiToCondaMap, ResolvedWheel, expand_name_map_groups};

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

/// Exact opt-in for the P4 uv re-resolve path. Every value except the literal
/// `1` preserves the legacy reconstruct-and-fetch behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum UvReresolveMode {
    #[default]
    Disabled,
    Enabled,
}

impl UvReresolveMode {
    pub(crate) fn from_env_value(value: Option<&OsStr>) -> Self {
        if value == Some(OsStr::new("1")) {
            Self::Enabled
        } else {
            Self::Disabled
        }
    }

    fn is_enabled(self) -> bool {
        self == Self::Enabled
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
    Complete,
    RetryKeepPypi { keep_pypi: BTreeSet<PypiKey> },
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
}

/// Active, target-marker-matched dependency declarations keyed in the PyPI
/// identity domain. Provenance remains attached until the shared finalizer has
/// resolved authority and produced a source-rich conflict, if any.
type ObservedRequirements = BTreeMap<PypiKey, Vec<Constraint>>;

fn observe_requirement(
    observed: &mut ObservedRequirements,
    pypi_name: &str,
    specifiers: &VersionSpecifiers,
    source: String,
    provenance: Provenance,
) {
    let observations = observed.entry(PypiKey::from_pypi(pypi_name)).or_default();
    let observation = Constraint {
        specifiers: specifiers.clone(),
        source,
        provenance,
    };
    if !observations.contains(&observation) {
        observations.push(observation);
    }
}

#[derive(Clone, Debug)]
struct RestoreRequestBuilder {
    pypi_name: String,
    bundle_name: String,
    constraints: Vec<Constraint>,
    route_preferences: BTreeMap<String, BTreeSet<String>>,
    lock_preferences: BTreeMap<String, BTreeSet<String>>,
}

impl RestoreRequestBuilder {
    fn new(pypi_name: &str) -> Self {
        Self {
            pypi_name: pypi_name.to_string(),
            bundle_name: canonical_conda_name(pypi_name),
            constraints: Vec::new(),
            route_preferences: BTreeMap::new(),
            lock_preferences: BTreeMap::new(),
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) {
        if !self.constraints.contains(&constraint) {
            self.constraints.push(constraint);
        }
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

    fn finish(mut self) -> Result<PypiFetchRequest> {
        let prior_constraints = self
            .route_preferences
            .iter()
            .chain(self.lock_preferences.iter())
            .map(|(version, sources)| {
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
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for constraint in prior_constraints {
            self.add_constraint(constraint);
        }
        if self.constraints.is_empty() {
            return Err(anyhow!(
                "joint route validation rejected `{}`, but no active PyPI \
                 requirement or prior selection was available",
                self.pypi_name
            ));
        }
        let route_preference = Self::unique_preference(&self.route_preferences);
        let lock_preference = Self::unique_preference(&self.lock_preferences);
        let specifiers = finalize(&PypiKey::from_pypi(&self.pypi_name), &self.constraints)?;

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

#[derive(Debug, PartialEq, Eq)]
enum JointRouteOutcome {
    Unchanged,
    Mutated,
    RetryKeepPypi { keep_pypi: BTreeSet<PypiKey> },
}

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
    let target = target.clone();
    let fetch_target = target.clone();
    let download_dir = download_dir.to_path_buf();
    let relax = config.relax;
    let fetch_pypi = move |request: PypiFetchRequest, index: String| {
        let target = fetch_target.clone();
        let download_dir = download_dir.clone();
        async move {
            let (resolved_url, metadata, sdist_prov) = super::bfs_fetch_pypi(
                &request.pypi_name,
                &request.specifiers,
                &index,
                &target,
                &download_dir,
                relax,
                request.preferred_version.as_deref(),
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
        &probe_many,
        &co_solve,
        &fetch_pypi,
        &channels_consulted,
        uv_reresolve,
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
                        wheel.metadata_provenance.clone(),
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
            match jointly_unroute_unsolvable(
                bundle,
                &mut provisional_metadata_routes,
                &observed_requirements,
                indexes,
                target,
                config,
                co_solve,
                fetch_pypi,
                channels_consulted,
                uv_reresolve,
            )
            .await?
            {
                JointRouteOutcome::Unchanged => break,
                JointRouteOutcome::Mutated => {
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
        if !added_any {
            match jointly_unroute_unsolvable(
                bundle,
                &mut provisional_metadata_routes,
                &observed_requirements,
                indexes,
                target,
                config,
                co_solve,
                fetch_pypi,
                channels_consulted,
                uv_reresolve,
            )
            .await?
            {
                JointRouteOutcome::Unchanged => {}
                JointRouteOutcome::Mutated => added_any = true,
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
    Ok(AutoBundleOutcome::Complete)
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
/// source-built wheel? Those requirements belong to the source package's
/// natural run-dependency emission: the source wheel already ships in the
/// bundle and is not a uv root that can be restored through Rule 2.
///
/// Route ownership is grouped by canonical conda identity because emission
/// deduplicates at that boundary. One source-built PyPI alias therefore fixes
/// the whole group; partially restoring another alias would still remove the
/// shared conda dependency.
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
                "provisional metadata route `{} -> {}` has no recorded Requires-Dist provenance",
                origin.pypi_name,
                origin.conda_name
            )
        })?;
        if requirements.is_empty() {
            return Err(anyhow!(
                "provisional metadata route `{} -> {}` has an empty Requires-Dist provenance set",
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
/// this bundle would emit. Uv routes and index-backed metadata routes are
/// mutable until this check succeeds; source-built metadata routes remain
/// owned by natural conda run-dependency emission. Rejected mutable routes are
/// restored through the same ordered PyPI fallback chain before the bundle is
/// changed.
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
    X: Fn(PypiFetchRequest, String) -> XF,
    XF: Future<Output = Result<ResolvedWheel>>,
{
    if bundle.auto_routed.is_empty() && metadata_routes.is_empty() {
        return Ok(JointRouteOutcome::Unchanged);
    }

    let assembly = super::emitted_bundle_route_assembly(bundle, config, target)?;
    let emitted = assembly.routes;
    let assembly_conflicts = assembly.conflicts;
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
        if metadata_route_group_has_source_built_origin(
            metadata_routes,
            observed_requirements,
            &conda_name,
        )? {
            continue;
        }
        if route_group_is_fully_mutable(bundle, metadata_routes, &conda_name, config, target)? {
            mutable_conda_names.insert(conda_name);
        }
    }
    for conflict in &assembly_conflicts {
        let conda_key = conflict.conda_name.key().into_string();
        if !mutable_conda_names.contains(&conda_key) {
            return Err(anyhow::Error::new(conflict.conflict.clone()));
        }
    }
    let conflicted_keys: BTreeSet<String> = assembly_conflicts
        .iter()
        .map(|conflict| conflict.conda_name.key().into_string())
        .collect();
    let pre_rejected: Vec<_> = conflicted_keys
        .iter()
        .filter_map(|name| emitted_by_conda.get(name).cloned())
        .collect();
    let mutable_candidates: Vec<_> = mutable_conda_names
        .iter()
        .filter(|name| !conflicted_keys.contains(*name))
        .filter_map(|name| emitted_by_conda.get(name).cloned())
        .collect();
    if mutable_candidates.is_empty() && pre_rejected.is_empty() {
        return Ok(JointRouteOutcome::Unchanged);
    }

    let mutable_keys: HashSet<String> = mutable_conda_names.iter().cloned().collect();
    let fixed: Vec<_> = emitted
        .into_iter()
        .filter(|route| !mutable_keys.contains(route.conda_name.key().as_str()))
        .collect();
    let selection = crate::uv_closure::select_jointly_solvable_routes(
        fixed,
        mutable_candidates.clone(),
        co_solve,
    )
    .await;
    // Rule 2 is fail-closed: an unsatisfiable/indeterminate baseline cannot
    // authorize any mutable conda route. Typed assembly conflicts are always
    // pre-rejected, then pass through the same restore gate below.
    let mut rejected = selection
        .map(|selection| selection.rejected)
        .unwrap_or(mutable_candidates);
    rejected.extend(pre_rejected);
    let mut seen_rejected = BTreeSet::new();
    rejected.retain(|route| seen_rejected.insert(route.conda_name.key().into_string()));
    if rejected.is_empty() {
        return Ok(JointRouteOutcome::Unchanged);
    }
    let rejected_keys: BTreeSet<String> = rejected
        .iter()
        .map(|route| route.conda_name.key().into_string())
        .collect();

    if uv_reresolve.mode.is_enabled() && uv_reresolve.uv_backed {
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
            return Err(anyhow!(
                "joint route validation rejected {:?}, but no PyPI route provenance was available for uv re-resolve",
                rejected_keys
            ));
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
            let requirement = Constraint {
                specifiers,
                source: input.source.clone(),
                provenance: input.effective_provenance(),
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
                    .or_insert_with(|| RestoreRequestBuilder::new(&origin.pypi_name));
                let requirements = observed_requirements
                    .get(&PypiKey::from_pypi(&key))
                    .ok_or_else(|| {
                        anyhow!(
                            "joint route validation rejected metadata route `{} -> {}`, \
                             but no active Requires-Dist provenance was recorded",
                            origin.pypi_name,
                            origin.conda_name
                        )
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
        return Err(anyhow!(
            "joint route validation rejected {:?}, but no PyPI route provenance was available",
            rejected_keys
        ));
    }
    // Finalize every requirement (dedupe + semantic satisfiability) before
    // the first index request, so one genuine conflict cannot be obscured by
    // a network error for another rejected route. Conflicts retain the P3
    // typed diagnostic and fail before any fetch.
    let mut finalized_restore_requests = Vec::new();
    for request in restore_requests.into_values() {
        finalized_restore_requests.push(request.finish()?);
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
    trial.extras.extend(restored_wheels);
    for (pypi_name, conda_name) in audit_origins {
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

    let still_emitted: Vec<String> = super::emitted_bundle_route_specs(&trial, config, target)?
        .into_iter()
        .map(|route| route.conda_name.key().into_string())
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
    Ok(JointRouteOutcome::Mutated)
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
            workspace_conda_versions: BTreeMap::new(),
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

    #[test]
    fn uv_reresolve_mode_requires_exact_literal_one() {
        assert_eq!(
            UvReresolveMode::from_env_value(Some(OsStr::new("1"))),
            UvReresolveMode::Enabled
        );
        for value in [None, Some(""), Some("0"), Some("01"), Some("true")] {
            assert_eq!(
                UvReresolveMode::from_env_value(value.map(OsStr::new)),
                UvReresolveMode::Disabled,
                "{value:?} must preserve the default legacy path"
            );
        }
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
            "index wheel metadata".to_string(),
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
            "source-built wheel metadata".to_string(),
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
            missing.contains("no recorded Requires-Dist provenance"),
            "{missing}"
        );

        let mut empty = ObservedRequirements::new();
        empty.insert(PypiKey::from_pypi("orphan-origin"), Vec::new());
        let empty =
            metadata_route_group_has_source_built_origin(&routes, &empty, "orphan-conda-dep")
                .unwrap_err()
                .to_string();
        assert!(
            empty.contains("empty Requires-Dist provenance set"),
            "{empty}"
        );

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
            "source-built wheel metadata".to_string(),
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
            later_missing.contains("no recorded Requires-Dist provenance"),
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
            },
            provenance: Provenance::PriorSelection,
        }
    }

    #[tokio::test]
    async fn rejected_uv_route_returns_all_keep_pypi_origins_without_shadow_fetch() {
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _index: String| {
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
            "test index metadata route origin".to_string(),
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
            move |_request: PypiFetchRequest, _index: String| {
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
            },
            provenance: Provenance::PriorSelection,
        }
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
            move |_request: PypiFetchRequest, _index: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Err(anyhow!("workspace-owned packaging must never fetch PyPI")) }
            }
        };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let config = test_config();
        let mut bundle = pace_packaging_conflict_bundle();
        bundle
            .workspace_conda_versions
            .insert("packaging".to_string(), "26.2".to_string());
        bundle
            .auto_routed
            .push(prior_selection_route("packaging", "23.0"));

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

        assert_eq!(outcome, AutoBundleOutcome::Complete);
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
            move |_request: PypiFetchRequest, _index: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Err(anyhow!("workspace-owned numpy must never fetch PyPI")) }
            }
        };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let config = test_config();
        let mut bundle = holosoma_numpy_conflict_bundle();
        bundle
            .workspace_conda_versions
            .insert("numpy".to_string(), "1.26.4".to_string());

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

        assert_eq!(outcome, AutoBundleOutcome::Complete);
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
    async fn no_workspace_fact_holosoma_conflict_still_fail_closes() {
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _index: String| {
                fetch_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Err(anyhow!("semantic conflict must fail before fetch")) }
            }
        };
        let target = crate::pypi::WheelTarget::for_subdir("3.11", "linux-64");
        let config = test_config();
        let mut bundle = holosoma_numpy_conflict_bundle();
        let favor_lock_prefs = BTreeMap::from([("numpy".to_string(), "1.26.4".to_string())]);
        bundle.apply_workspace_conda_fact_ownership(
            &config,
            &config.name_map,
            &BTreeSet::new(),
            &BTreeSet::new(),
        );
        assert!(
            bundle.auto_dropped.is_empty(),
            "a prior uv/favor-lock selection is not workspace conda ownership"
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
    async fn source_built_serveable_transitive_emits_conda_run_dependency() {
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
            move |_request: PypiFetchRequest, _index: String| {
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
            0,
            "source-built metadata routes must not enter Rule-2 candidate selection"
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
    async fn workspace_fact_owns_source_built_advisory_dependency() {
        let fetch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fetch = {
            let fetch_calls = Arc::clone(&fetch_calls);
            move |_request: PypiFetchRequest, _index: String| {
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
        bundle
            .workspace_conda_versions
            .insert("starlette".to_string(), "0.45.3".to_string());
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
                mode: UvReresolveMode::from_env_value(None),
                uv_backed: true,
                keep_pypi: BTreeSet::new(),
            },
        )
        .await
        .unwrap();

        assert_eq!(outcome, AutoBundleOutcome::Complete);

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
        let context = super::super::CondaCoSolveContext::new(
            Some(&manifest),
            Some(&tmp),
            &tmp.join("hover-pack"),
            &target,
            &[],
            "hover-pack",
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

        let error = auto_bundle_transitives_with(
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
        overridden.add_constraint(Constraint {
            specifiers: VersionSpecifiers::from_str("<2").unwrap(),
            source: "wheel metadata requires overridden<2".to_string(),
            provenance: Provenance::IndexWheelMetadata,
        });
        overridden.add_constraint(Constraint {
            specifiers: VersionSpecifiers::from_str("==3").unwrap(),
            source: "uv override requirement `overridden==3`".to_string(),
            provenance: Provenance::UvOverride,
        });
        overridden.add_constraint(Constraint {
            specifiers: VersionSpecifiers::from_str("<4").unwrap(),
            source: "uv constraint `overridden<4`".to_string(),
            provenance: Provenance::UvConstraint,
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

        let mut soft_hints = RestoreRequestBuilder::new("soft-hints");
        soft_hints.add_constraint(Constraint {
            specifiers: VersionSpecifiers::from_str(">=1").unwrap(),
            source: "wheel `root==1` Requires-Dist `soft-hints>=1`".to_string(),
            provenance: Provenance::IndexWheelMetadata,
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

        let mut prior_only = RestoreRequestBuilder::new("s3transfer");
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

        let mut index_pin = RestoreRequestBuilder::new("s3transfer");
        index_pin.add_constraint(Constraint {
            specifiers: VersionSpecifiers::from_str("==0.13.1").unwrap(),
            provenance: Provenance::IndexWheelMetadata,
            source: "index wheel Requires-Dist `s3transfer==0.13.1`".to_string(),
        });
        assert_eq!(
            index_pin.finish().unwrap().specifiers,
            VersionSpecifiers::from_str("==0.13.1").unwrap(),
            "an index-wheel exact requirement remains authoritative"
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
            crate::index_chain::IndexPurpose::Resolve,
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
}
