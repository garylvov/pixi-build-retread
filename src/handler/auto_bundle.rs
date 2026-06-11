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

use super::{Bundle, DEFAULT_PYTHON, PypiToCondaMap, ResolvedWheel};

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
pub(crate) async fn auto_bundle_transitives(
    bundle: &mut Bundle,
    entry_index: &str,
    workspace_indexes: &[String],
    target: &crate::pypi::WheelTarget,
    download_dir: &Path,
    config: &RetreadConfig,
    conda_channels: &[ChannelUrl],
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
    let mut skip: HashSet<String> = bundle.all_wheels().map(|w| w.pypi_name.clone()).collect();
    skip.extend(config.conda_deps.iter().map(|n| canonical_conda_name(n)));
    skip.extend(config.drop_deps.iter().map(|n| canonical_conda_name(n)));
    skip.extend(config.overrides.keys().map(|n| canonical_conda_name(n)));

    // Fallback chain: entry's index first (for siblings on private
    // indexes like pypi.nvidia.com), then workspace [pypi-options]
    // indexes, then public PyPI (for the broader ecosystem -- aiodns,
    // qdldl, ...). Public PyPI is hardcoded rather than configurable
    // for now; if a user has air-gap requirements they can disable
    // retread-auto-bundle entirely.
    let mut indexes = vec![entry_index.to_string()];
    for url in workspace_indexes {
        if !indexes
            .iter()
            .any(|e| e.trim_end_matches('/') == url.trim_end_matches('/'))
        {
            indexes.push(url.clone());
        }
    }
    let public = "https://pypi.org/simple/".to_string();
    if !indexes
        .iter()
        .any(|e| e.trim_end_matches('/') == public.trim_end_matches('/'))
    {
        indexes.push(public);
    }

    // Fixed-point loop: each newly-bundled wheel has its own
    // Requires-Dist that may name more PyPI-only transitives, which
    // themselves should be auto-bundled (e.g. bundling torch pulls in
    // nvidia-cuda-nvrtc-cu12). Re-scan after every bundle until no new
    // wheels are added. Cycle-detected via seen_candidate, which
    // accumulates across iterations.
    let mut seen_candidate: HashSet<String> = skip.clone();
    let mut processed_wheel_count = 0;
    loop {
        // Collect new candidates from wheels we haven't scanned yet.
        let mut candidates: Vec<(String, String)> = Vec::new();
        for wheel in bundle.all_wheels().skip(processed_wheel_count) {
            for raw in &wheel.metadata.requires_dist {
                let Some((name, version)) = pep508_exact_base_dep(raw)? else {
                    continue;
                };
                let conda_name = canonical_conda_name(&name);
                if !seen_candidate.insert(conda_name) {
                    continue;
                }
                candidates.push((name, version));
            }
        }
        processed_wheel_count = bundle.all_wheels().count();
        if candidates.is_empty() {
            break;
        }

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
        let mut to_fetch: Vec<(String, String, String, VersionSpecifiers)> = Vec::new();
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
                bundle.probe_decisions.push(crate::audit::ProbeDecision {
                    stage: "auto_bundle".into(),
                    pypi_name: name.clone(),
                    conda_name: conda_target_name.clone(),
                    spec: probe_spec.clone(),
                    target_python: target.python_version.clone(),
                    channels_consulted: probe_result.channels_consulted.clone(),
                    satisfiable: probe_result.satisfiable,
                    matching_candidates: probe_result.matching_candidates,
                    routing_decision: routing_decision.into(),
                });
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
                    bundle.probe_decisions.push(crate::audit::ProbeDecision {
                        stage: "auto_bundle_name_level".into(),
                        pypi_name: name.clone(),
                        conda_name: conda_target_name.clone(),
                        spec: "*".into(),
                        target_python: target.python_version.clone(),
                        channels_consulted: name_level.channels_consulted.clone(),
                        satisfiable: name_level.satisfiable,
                        matching_candidates: name_level.matching_candidates,
                        routing_decision: if name_level.is_satisfied() {
                            "name-level-conda-keep"
                        } else {
                            "fall-through-to-pypi"
                        }
                        .into(),
                    });
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
            to_fetch.push((name, version, conda_name, specifiers));
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
                .map(|(name, version, conda_name, specifiers)| async move {
                    for index in indexes_ref {
                        match pypi::resolve(index, &name, &specifiers, target).await {
                            Ok(resolved) => {
                                match metadata_preferring_sidecar(&resolved, download_dir).await {
                                    Ok(metadata) => {
                                        return Some((
                                            name,
                                            version,
                                            ResolvedWheel {
                                                pypi_name: conda_name,
                                                url: resolved.url,
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
                })
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

/// (wheel URL, parsed METADATA, index to recurse with) for one
/// PyPI-form BFS item fetched in the level loop's phase 2.
pub(crate) type BfsFetched = (url::Url, WheelMetadata, String);

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
    /// `Requires-Dist: <name> @ git+<scheme>://<host>/<path>@<rev>` --
    /// clone + `pip wheel --no-deps`.
    Git { url: String, rev: Option<String> },
    /// `Requires-Dist: <name> @ <scheme>://...` (direct wheel/sdist).
    Url { wheel_url: url::Url },
}

/// Add extras-gated and prefix-matched base deps from `metadata` to `work`.
/// Skips entries already in `seen` so the BFS terminates.
pub(crate) fn seed_worklist(
    metadata: &WheelMetadata,
    extras_requested: &[String],
    index: &str,
    bundle_prefix: &str,
    seen: &HashSet<String>,
    work: &mut VecDeque<Pending>,
) -> Result<()> {
    for raw in &metadata.requires_dist {
        // 1. Extras-gated lines for each requested extra.
        let mut added = false;
        for extra in extras_requested {
            if let Some(dep) = pep508_extra_dep(raw, extra)? {
                let dn = canonical_conda_name(&dep.name);
                if seen.contains(&dn) {
                    continue;
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
            if seen.contains(&dn) {
                continue;
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
        ExtraDepSource::Git { url, rev } => PendingSource::Git { url, rev },
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
/// [`ExtraDepSource`] variants. Splits `git+<scheme>://...@<rev>` into
/// `(base_url, Some(rev))`; plain `https://.../file.whl` becomes a
/// direct-URL fetch.
pub(crate) fn extra_dep_source_from_url(raw_url: &url::Url) -> Result<ExtraDepSource> {
    let s = raw_url.as_str();
    if let Some(stripped) = s.strip_prefix("git+") {
        // PEP 508 doesn't say where the @<rev> lives but pip-compatible
        // syntax is `git+<scheme>://<host>/<path>@<rev>`. Find the
        // rightmost `@` that comes after `://` (skipping any in user-
        // info, though those are rare for public git).
        let scheme_end = stripped.find("://").map(|i| i + 3).unwrap_or(0);
        let (base, rev) = match stripped[scheme_end..].rfind('@') {
            Some(rel) => {
                let abs = scheme_end + rel;
                (
                    stripped[..abs].to_string(),
                    Some(stripped[abs + 1..].to_string()),
                )
            }
            None => (stripped.to_string(), None),
        };
        Ok(ExtraDepSource::Git { url: base, rev })
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
    Git { url: String, rev: Option<String> },
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
