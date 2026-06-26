//! Auto-bundle BFS: prefer-conda routing and PyPI-only transitive
//! packaging into the bundle.
//!
//! Extracted from handler.rs (Phase 0b.3). All functions are behavior-
//! identical whole-function moves; no logic changes.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, anyhow};
use rattler_conda_types::ChannelUrl;
use uv_pep508::uv_pep440::VersionSpecifiers;

use crate::config::{RelaxPolicy, RetreadConfig};
use crate::pypi;
use crate::relax::{canonical_conda_name, default_marker_env};
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
/// retread-name-map). When true, the prefer-conda policy in
/// [`auto_bundle_transitives`] skips bundling so the dep flows through to
/// emission as a conda run-dep.
pub(crate) fn prefer_conda_match(
    conda_normalized_pypi_name: &str,
    name_map: &BTreeMap<String, String>,
) -> bool {
    name_map.contains_key(conda_normalized_pypi_name)
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

/// Build the conda match-spec string the channel probe should look
/// for, given a resolved PyPI version and the active relax policy.
/// Mirrors what `translate(==<version>)` would emit, since that's
/// the spec the conda solver will eventually face. Falls back to
/// `*` (any version) when the version can't be parsed -- a generous
/// default that lets the probe succeed if ANY build of the package
/// exists on the channel.
fn probe_spec_for(version_str: &str, policy: RelaxPolicy) -> String {
    match uv_pep508::uv_pep440::Version::from_str(version_str) {
        Ok(v) => crate::relax::widen_exact(&v, policy).unwrap_or_else(|| "*".to_string()),
        Err(_) => "*".to_string(),
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
/// Best-effort: a resolve failure logs at debug and leaves the dep to be
/// emitted as a conda run-dep (current fallback behavior).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn auto_bundle_transitives(
    bundle: &mut Bundle,
    entry_index: &str,
    workspace_indexes: &[String],
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
) -> Result<()> {
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
    skip.extend(config.conda_deps.iter().map(|n| canonical_conda_name(n)));
    skip.extend(config.drop_deps.iter().map(|n| canonical_conda_name(n)));
    skip.extend(config.overrides.keys().map(|n| canonical_conda_name(n)));

    // Fallback chain: entry's index first (for siblings on private
    // indexes like pypi.nvidia.com), then workspace [pypi-options]
    // indexes, then public PyPI (for the broader ecosystem -- aiodns,
    // qdldl, ...). Public PyPI is appended by merge_index_chain when
    // not already present; ordering and trailing-slash dedup are both
    // handled there.
    let indexes =
        super::merge_index_chain(std::iter::once(entry_index.to_string()), workspace_indexes);

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
    let mut processed_wheel_count = 0;
    loop {
        // Collect new candidates from wheels we haven't scanned yet.
        let mut candidates: Vec<(String, String)> = Vec::new();
        // v1.7.0: ALSO consider bare/ranged base deps. They get the
        // step-8 gate below: bundle only when a definitive name-level
        // probe says conda has ZERO candidates (a true PyPI-only dep);
        // otherwise they stay conda-side exactly as before.
        let mut loose_candidates: Vec<(String, VersionSpecifiers)> = Vec::new();
        for wheel in bundle.all_wheels().skip(processed_wheel_count) {
            for raw in &wheel.metadata.requires_dist {
                if let Some((name, version)) = pep508_exact_base_dep(raw)? {
                    let conda_name = canonical_conda_name(&name);
                    if !seen_candidate.insert(conda_name) {
                        continue;
                    }
                    candidates.push((name, version));
                } else if let Some((name, specs)) = pep508_loose_base_dep(raw)? {
                    let conda_name = canonical_conda_name(&name);
                    if !seen_candidate.insert(conda_name) {
                        continue;
                    }
                    loose_candidates.push((name, specs));
                }
            }
        }
        processed_wheel_count = bundle.all_wheels().count();
        if candidates.is_empty() && loose_candidates.is_empty() {
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
        // bounded) instead of one serial await per candidate. The
        // name-level + PyPI fallback steps below stay serial -- they
        // only run for the few definitively-unsat candidates, against
        // the already-warm in-memory repodata cache.
        let prefer_pairs: Vec<(String, String)> = candidates
            .iter()
            .filter(|(name, _)| prefer_conda_match(&canonical_conda_name(name), &config.name_map))
            .map(|(name, version)| {
                let conda_name = canonical_conda_name(name);
                (
                    config.name_map[&conda_name].clone(),
                    probe_spec_for(version, config.relax),
                )
            })
            .collect();
        let prefer_probes: std::collections::HashMap<(String, String), crate::probe::ProbeResult> =
            crate::probe::probe_many(conda_channels, prefer_pairs, Some(&target.python_version))
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
            if prefer_conda_match(&conda_name, &config.name_map) {
                // Probe the workspace's conda channels for whether the
                // spec retread would emit is actually satisfiable. If
                // ANY channel has a matching candidate, keep on conda.
                // If every channel was reachable and returned versions
                // but NONE matched, fall through to auto-bundle. An
                // indecisive probe (no prefix.dev channels, or all
                // probes errored) keeps the legacy prefer-conda
                // behavior so a prefix.dev outage doesn't silently
                // reshape routing.
                let conda_target_name = config.name_map[&conda_name].clone();
                let probe_spec = probe_spec_for(&version, config.relax);
                let probe_result =
                    match prefer_probes.get(&(conda_target_name.clone(), probe_spec.clone())) {
                        Some(r) => r.clone(),
                        // Defensive: shouldn't happen (pairs built from the
                        // same predicate), but fall back to a direct probe
                        // rather than mis-routing.
                        None => {
                            crate::probe::probe(
                                conda_channels,
                                &conda_target_name,
                                &probe_spec,
                                Some(&target.python_version),
                            )
                            .await
                        }
                    };
                let routing_decision = if probe_result.is_definitively_unsatisfied() {
                    "fall-through-to-pypi"
                } else if probe_result.is_satisfied() {
                    "short-circuit"
                } else {
                    "indecisive-short-circuit"
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
                if probe_result.is_definitively_unsatisfied() {
                    // v0.46.0: the EXACT resolved version isn't on conda --
                    // but that's usually because the transitive was resolved
                    // to PyPI-latest in isolation (e.g. a bundled `pytorch3d`
                    // pulls `torch`, which resolves to 2.12.0 while conda
                    // tops out at pytorch 2.7.x). Bundling that too-new wheel
                    // SHADOWS conda's ABI-correct copy (a cu130 torch wheel
                    // over the env's cu128 pytorch) AND double-installs a dep
                    // retread also emits as a conda run-dep -- the bundled
                    // wheel then clobbers the conda build at install time.
                    //
                    // So before bundling, probe whether conda has the package
                    // at ANY (python-compatible) version. If it does, keep it
                    // on the conda side: the run-dep emission + solve cascade
                    // pick the ABI-correct conda build, derived from the
                    // wheel's ORIGINAL requirement (not PyPI-latest). Only
                    // bundle when conda genuinely lacks the package -- a true
                    // PyPI-only dependency. (Escape hatch if this picks wrong:
                    // retread-overrides to force a spec, or retread-drop-deps.)
                    let name_level = crate::probe::probe(
                        conda_channels,
                        &conda_target_name,
                        "*",
                        Some(&target.python_version),
                    )
                    .await;
                    bundle
                        .probe_decisions
                        .push(crate::audit::ProbeDecision::from_probe(
                            "auto_bundle_name_level",
                            &name,
                            &conda_target_name,
                            "*",
                            &target.python_version,
                            &name_level,
                            if name_level.is_satisfied() {
                                "name-level-conda-keep"
                            } else {
                                "fall-through-to-pypi"
                            },
                        ));
                    if name_level.is_satisfied() {
                        tracing::info!(
                            dep = %name,
                            conda = %conda_target_name,
                            resolved_version = %version,
                            conda_matches = name_level.matching_candidates,
                            "prefer-conda: exact resolved version absent on conda, but the package exists at other versions -- keeping on conda (ABI-correct) instead of bundling a too-new PyPI build",
                        );
                        continue;
                    }
                    tracing::info!(
                        dep = %name,
                        conda = %conda_target_name,
                        spec = %probe_spec,
                        channels = ?probe_result.channels_consulted,
                        "prefer-conda: conda has no build of this package at any version; falling back to auto-bundle from PyPI",
                    );
                    // intentional fall-through: continue with bundle path
                } else {
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
            let specifiers = match VersionSpecifiers::from_str(&format!("=={version}")) {
                Ok(s) => s,
                Err(e) => {
                    tracing::debug!(
                        dep = %name, version = %version,
                        error = %e,
                        "auto-bundle: skipping unparseable version"
                    );
                    continue;
                }
            };
            // favor-lock: look up preferred version for this dep by canonical name.
            let preferred_ver = favor_lock_prefs
                .and_then(|m| m.get(&canonical_conda_name(&conda_name)))
                .cloned();
            to_fetch.push((name, version, conda_name, specifiers, preferred_ver));
        }

        // Loose (bare/ranged) candidates: the step-8 gate. A name-level
        // probe with definitive ZERO conda candidates means conda can
        // never satisfy the dep -- bundle it from PyPI (newest version
        // matching the line's specifiers). Anything conda CAN ship
        // stays conda-side (the cascade translates the line).
        let loose_pairs: Vec<(String, String)> = loose_candidates
            .iter()
            .map(|(name, _)| {
                let conda_name = canonical_conda_name(name);
                let target_name = config
                    .name_map
                    .get(&conda_name)
                    .cloned()
                    .unwrap_or(conda_name);
                (target_name, "*".to_string())
            })
            .collect();
        let loose_probes: std::collections::HashMap<(String, String), crate::probe::ProbeResult> =
            crate::probe::probe_many(conda_channels, loose_pairs, Some(&target.python_version))
                .await
                .into_iter()
                .map(|r| ((r.package.clone(), r.spec.clone()), r))
                .collect();
        for (name, specs) in loose_candidates {
            let conda_name = canonical_conda_name(&name);
            let target_name = config
                .name_map
                .get(&conda_name)
                .cloned()
                .unwrap_or_else(|| conda_name.clone());
            let probe_result = match loose_probes.get(&(target_name.clone(), "*".to_string())) {
                Some(r) => r.clone(),
                None => {
                    crate::probe::probe(
                        conda_channels,
                        &target_name,
                        "*",
                        Some(&target.python_version),
                    )
                    .await
                }
            };
            let bundle_it = probe_result.is_definitively_unsatisfied();
            bundle
                .probe_decisions
                .push(crate::audit::ProbeDecision::from_probe(
                    "auto_bundle_loose",
                    &name,
                    &target_name,
                    "*",
                    &target.python_version,
                    &probe_result,
                    if bundle_it {
                        "auto-pypi-no-conda-candidates"
                    } else {
                        "conda-keep"
                    },
                ));
            if !bundle_it {
                continue;
            }
            tracing::info!(
                dep = %name,
                specs = %specs,
                "auto-bundle: bare/ranged dep has zero conda candidates; bundling from PyPI",
            );
            // favor-lock: look up preferred version for this dep by canonical name.
            let preferred_ver = favor_lock_prefs
                .and_then(|m| m.get(&canonical_conda_name(&conda_name)))
                .cloned();
            to_fetch.push((name, specs.to_string(), conda_name, specs, preferred_ver));
        }

        // v1.4.0: fetch this round's PyPI-bound wheels concurrently
        // (8-way bounded). `buffered` (not buffer_unordered) preserves
        // candidate order, so extras order -- and therefore the next
        // round's Requires-Dist scan order -- stays deterministic.
        // Per item, the index fallback chain is still walked serially
        // exactly as before: a resolve failure tries the next index, a
        // FETCH failure gives up on the candidate (leaves it as a
        // conda dep), exhaustion logs the retread-drop-deps hint.
        let fetched: Vec<Option<(String, String, ResolvedWheel)>> = {
            use futures::stream::{self, StreamExt};
            let indexes_ref = &indexes;
            stream::iter(to_fetch)
                .map(
                    |(name, version, conda_name, specifiers, preferred_ver)| async move {
                        // favor-lock: if a committed lock version exists for this dep,
                        // prefer it over the highest satisfying version on PyPI.
                        // `preferred_ver` is pre-looked-up before the async block to
                        // avoid capturing a reference to the prefs map.
                        let preferred: Option<&str> = preferred_ver.as_deref();
                        for index in indexes_ref {
                            let resolved_result = if let Some(pv) = preferred {
                                pypi::resolve_preferring(index, &name, &specifiers, target, pv)
                                    .await
                            } else {
                                pypi::resolve(index, &name, &specifiers, target).await
                            };
                            match resolved_result {
                                Ok(resolved) => {
                                    match metadata_preferring_sidecar(&resolved, download_dir).await
                                    {
                                        Ok(metadata) => {
                                            // resolved.url is the pristine index
                                            // URL; clone it for upstream_url
                                            // before moving into the struct.
                                            let upstream = Some(resolved.url.clone());
                                            return Some((
                                                name,
                                                version,
                                                ResolvedWheel {
                                                    pypi_name: conda_name,
                                                    url: resolved.url,
                                                    upstream_url: upstream,
                                                    git_source: None,
                                                    sdist_source: None,
                                                    metadata,
                                                    extras_requested: vec![],
                                                    auto_data: None,
                                                    auto_data_dedup_skipped_root: None,
                                                },
                                            ));
                                        }
                                        Err(e) => {
                                            tracing::debug!(
                                                dep = %name,
                                                error = %format!("{e:#}"),
                                                "auto-bundle fetch failed; leaving as conda dep"
                                            );
                                            return None;
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::debug!(
                                        dep = %name,
                                        version = %version,
                                        index = %index,
                                        error = %format!("{e:#}"),
                                        "auto-bundle resolve failed on this index"
                                    );
                                }
                            }
                        }
                        tracing::debug!(
                            dep = %name,
                            version = %version,
                            "auto-bundle exhausted all indexes; leaving as conda dep. \
                             If conda can't satisfy it, add to retread-drop-deps."
                        );
                        None
                    },
                )
                .buffered(8)
                .collect()
                .await
        };
        for (name, version, wheel) in fetched.into_iter().flatten() {
            tracing::info!(
                dep = %name,
                version = %version,
                "auto-bundled into {}",
                bundle.conda_name,
            );
            bundle.extras.push(wheel);
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

/// Returns Some((name, exact_version)) if `raw` is a base dep (no
/// extras marker) with a single `== X.Y.Z` specifier. Returns None for
/// extras-gated deps, ranges, ~=, or URL deps.
fn pep508_exact_base_dep(raw: &str) -> Result<Option<(String, String)>> {
    let req: uv_pep508::Requirement = uv_pep508::Requirement::from_str(raw)
        .map_err(|e| anyhow!("parsing requirement `{raw}`: {e}"))?;
    let env = default_marker_env(DEFAULT_PYTHON)?;
    if !req.marker.evaluate(&env, &[]) {
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
fn pep508_loose_base_dep(raw: &str) -> Result<Option<(String, VersionSpecifiers)>> {
    let req: uv_pep508::Requirement = uv_pep508::Requirement::from_str(raw)
        .map_err(|e| anyhow!("parsing requirement `{raw}`: {e}"))?;
    let env = default_marker_env(DEFAULT_PYTHON)?;
    if !req.marker.evaluate(&env, &[]) {
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
                    "metadata sidecar fetch failed; falling back to full wheel download",
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
