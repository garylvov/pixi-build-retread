//! uv-backed wheel-closure computation (spec-uv-restructure.md, milestone 1).
//!
//! Replaces the cascade/resolvo mirror-solver's *closure-computation* role
//! with `uv` run as a subprocess:
//!
//! 1. Synthesize an ephemeral uv project (`pyproject.toml`) whose
//!    `[project.dependencies]` are the bundle's root requirements and whose
//!    `[tool.uv] constraint-dependencies` mirror the workspace's conda pins
//!    (name-mapped pypi<-conda), so uv resolves the PyPI side compatibly
//!    with conda — exactly pixi's conda-first handoff.
//! 2. `uv lock` the project, then `uv export --format pylock.toml` with
//!    `--no-emit-package <name>` per conda-routed package.
//! 3. Parse the PEP 751 pylock into the closure/lock shapes the
//!    materialization path consumes (`crate::lock::LockWheel`), selecting ONE
//!    wheel per package by tag priority (`crate::pypi::score_wheel`).
//!
//! Constraint *provenance* is load-bearing: every generated constraint line
//! carries a record of the conda source package it came from
//! (`constraints.provenance.json`), so a `uv lock` conflict can be
//! attributed to the offending conda pin. A proven transitive exact pin that
//! contradicts a precise workspace-solved conda fact is retried with that
//! fact as a graph-wide uv override; every other conflict is reported for
//! `retread solve` unchanged.
//!
//! uv is the only resolver: this closure computation runs unconditionally
//! for every bundle with uv-resolvable roots (v4.4.0; the `retread-resolver`
//! knob and the historical cascade/resolvo mirror-solver are both gone).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use uv_pep508::Requirement;

use crate::lock::{LockWheel, Origin};
use crate::pypi::WheelTarget;
use crate::relax::canonical_conda_name;

/// Env var overriding the uv binary path (spec §2.5).
pub const UV_BIN_ENV: &str = "RETREAD_UV";

/// Marker appended to `retread-drop-deps` override entries so uv removes
/// the name from the resolution graph entirely (spec AMENDMENT A3: the
/// documented uv idiom for dependency removal — an override with an
/// unmatchable environment marker).
pub const DROP_MARKER: &str = "python_version < '0'";

// ---------------------------------------------------------------------------
// Request / provenance types
// ---------------------------------------------------------------------------

/// Provenance of one generated constraint line: which conda package (and
/// which manifest/lock source) produced it. Keyed by PyPI name in
/// [`ConstraintSet::provenance`]. Serialized shape matches spec §2.2.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstraintProvenance {
    /// The emitted PEP 508 constraint line, e.g. `torch==2.10.0`.
    pub constraint: String,
    /// Conda package name the constraint was derived from (pre name-map).
    pub conda_name: String,
    /// Conda version/spec string as declared by the source.
    pub conda_version: String,
    /// Where the pin was read from: `"manifest"` or `"pixi.lock"`.
    pub source: String,
    /// Environment the pin belongs to (e.g. `"default"`).
    pub env: String,
    /// Non-authoritative hint that may constrain an existing root but must not
    /// authorize repair mutations against a workspace or pack manifest.
    #[serde(default)]
    pub advisory: bool,
}

/// Generated constraint lines + their provenance, keyed by PyPI name.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstraintSet {
    /// PEP 508 constraint lines for `[tool.uv] constraint-dependencies`.
    pub constraints: Vec<String>,
    /// PyPI name -> provenance for every line in `constraints`.
    pub provenance: BTreeMap<String, ConstraintProvenance>,
    /// Constraint lines synthesized only to keep an already-routed package's
    /// PyPI selection stable during the next uv round. These are solver
    /// bookkeeping, not authoritative user/workspace requirements, and must
    /// not become hard requirements if joint validation later un-routes the
    /// package.
    #[serde(skip)]
    pub auto_route_constraint_indices: BTreeSet<usize>,
}

/// Everything needed to synthesize + solve one (bundle, python, platform)
/// closure.
#[derive(Debug, Clone)]
pub struct UvClosureRequest {
    /// Bundle (conda output) name; used for the project name and messages.
    pub bundle: String,
    /// Target python `X.Y` (e.g. `"3.12"`).
    pub python_version: String,
    /// Target conda subdir (e.g. `"linux-64"`).
    pub conda_subdir: String,
    /// PEP 508 root requirements (the bundle's `[retread-wheels]` entries).
    pub dependencies: Vec<String>,
    /// Conda pins as uv constraints, with provenance.
    pub constraints: ConstraintSet,
    /// PEP 508 `override-dependencies` lines (user `retread-overrides`
    /// translated, plus `retread-drop-deps` unmatchable markers).
    pub overrides: Vec<String>,
    /// Names excluded from the exported closure (conda-routed;
    /// `--no-emit-package` + authoritative post-parse filter).
    pub no_emit_packages: Vec<String>,
    /// Simple-index chain, in priority order. Public PyPI last.
    pub index_urls: Vec<String>,
    /// retread-built wheels satisfying in-project names:
    /// entry name -> path (relative to the project dir or absolute),
    /// emitted as `[tool.uv.sources]` path sources.
    pub built_wheel_sources: BTreeMap<String, PathBuf>,
    /// Self-heal repairs injected as EXPLICIT first-party `name==version`
    /// entries in the synthesized project's `dependencies` (see
    /// [`with_sdist_heal`]): canonical pypi name -> exact resolved
    /// version. Two heal classes need first-party status:
    /// * transitive PRERELEASE pins -- uv's `if-necessary-or-explicit`
    ///   policy only honors a pre-release specifier declared first-party;
    /// * heal-BUILT sdist wheels -- a `[tool.uv.sources]` path entry only
    ///   applies to the project's OWN requirements, so a wheel built for
    ///   a TRANSITIVE dependency is invisible to the resolver ("not found
    ///   in the package registry") unless the name is also a direct
    ///   dependency (isaac-pack repro: `idna-ssl==1.1.0`, required only
    ///   by isaacsim-kernel, re-failed the healed relock identically).
    ///
    /// Empty on the green path; kept out of the user's manifest (the
    /// synthesized project is ephemeral).
    pub explicit_pins: BTreeMap<String, String>,
    /// Append `--offline` to uv invocations (replay mode).
    pub offline: bool,
}

/// A computed closure: index wheels in lock shape + the name->version pin
/// map (the seam consumed by the legacy materialization path as a locked
/// closure until the M3 seam swap).
#[derive(Debug, Clone)]
pub struct UvClosure {
    /// One selected index wheel per resolved package, lock-shaped.
    pub wheels: Vec<LockWheel>,
    /// PEP 503-normalized name -> resolved version for every package in
    /// the exported closure (including packages whose artifact selection
    /// belongs to retread's own built wheels).
    pub pins: BTreeMap<String, String>,
    /// uv version that produced this closure.
    pub uv_version: String,
    /// spec-uv-restructure M2: packages the auto-route loop moved out of
    /// the wheel closure onto the conda side. Empty when auto-route is
    /// off (or the plain [`compute_closure`] driver was used).
    pub auto_routed: Vec<AutoRoutedPackage>,
    /// Workspace-owned packages removed from the wheel side after a
    /// validated conda route. Unlike `auto_routed`, these are not emitted as
    /// run dependencies of the generated pack: the consuming workspace
    /// already declares and supplies them. This is ephemeral build evidence,
    /// never persisted as part of an [`AutoRoutedPackage`].
    pub auto_dropped: BTreeSet<String>,
    /// Effective authoritative uv inputs for the exact request that produced
    /// this closure. `None` is reserved for parser-only/test closures that did
    /// not cross the request-aware solve boundary; route planning then derives
    /// the same map from its visible request.
    pub effective_input_requirements: Option<BTreeMap<String, Vec<AutoRouteInputRequirement>>>,
}

// ---------------------------------------------------------------------------
// Auto-route (spec-uv-restructure M2): probe-driven conda routing filter
// ---------------------------------------------------------------------------

/// One auto-routed package: excluded from the wheel closure, provided by
/// a conda channel instead. Recorded so the backend emits it as a conda
/// run-dependency of the stub package.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AutoRouteInputRole {
    /// A direct/root requirement.
    #[default]
    Requirement,
    /// An additive uv constraint.
    Constraint,
    /// A uv override, which replaces ordinary dependency requirements.
    Override,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoRouteInputRequirement {
    /// PEP 440 specifiers from an authoritative uv input (empty means any
    /// version). This is distinct from the version uv happened to select.
    pub specifiers: String,
    /// Human-readable origin retained for source-rich un-route conflicts.
    pub source: String,
    /// uv semantic role. Overrides replace ordinary requirements; constraints
    /// remain additive in either case.
    #[serde(default)]
    pub role: AutoRouteInputRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoRoutedPackage {
    /// PEP 503-canonical PyPI name (closure-side identity).
    pub pypi_name: String,
    /// Conda package name it routes to (name-mapped; identity fallback).
    pub conda_name: String,
    /// The closure's resolved PyPI version the conda side must satisfy.
    pub pypi_version: String,
    /// Conda version of the matching build on the channel.
    pub conda_version: String,
    /// `<channel_url>/<subdir>` the match was found on.
    pub channel: String,
    /// Active root/constraint/override/self-heal inputs that governed uv's
    /// selection for this package. Empty means the selected PyPI version was
    /// only solver output and may be used as a soft restoration preference.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_requirements: Vec<AutoRouteInputRequirement>,
}

/// Configuration for the auto-route loop.
#[derive(Debug, Clone)]
pub struct AutoRouteOptions {
    /// Master switch (`auto-route` in `[package.build.config]`; default on).
    pub enabled: bool,
    /// Which candidates the routing sweep may move to conda.
    /// `PreferCondaValidated` considers every candidate but accepts only
    /// fact-validated routes; `Minimal` caps eligibility at the ABI/binary
    /// whitelist (see [`route_policy_admits`]); `Aggressive` preserves the
    /// legacy route-anything-conda-has behavior.
    ///
    /// NOTE: the struct `Default` here is `Aggressive` (so the extensive
    /// pre-v4.6 unit-test matrix keeps exercising the legacy sweep it was
    /// written against); the PRODUCTION default comes from the config
    /// layer (`crate::config::RoutePolicy::default()` =
    /// `PreferCondaValidated`), which the handler wires in explicitly.
    pub route_policy: crate::config::RoutePolicy,
    /// v4.6 Part A: extra canonical PyPI names admitted to routing under
    /// `Minimal` (`retread-route-include`), beyond the built-in whitelist.
    pub route_include: BTreeSet<String>,
    /// Canonical PyPI names the user opted OUT of auto-routing
    /// (`keep-pypi`). Never routed, no probe issued.
    pub keep_pypi: BTreeSet<String>,
    /// Canonical PyPI names that must stay in the closure regardless of
    /// conda availability: the bundle's own root entries and retread-built
    /// wheels. Routing a root to conda would hollow out the pack.
    pub protected: BTreeSet<String>,
    /// Effective pypi -> conda name map (user retread-name-map + fallback
    /// table + parselmouth merge). Missing names use the identity mapping.
    pub name_map: BTreeMap<String, String>,
    /// Canonical PyPI names the self-healing un-route step must NEVER
    /// move back to the wheel closure (`force-conda`), even when the
    /// co-installability solve names them in an unsat core. The user
    /// asserts these must ship as conda run-deps; a persisting conflict
    /// then fails the workspace lock loudly instead of silently
    /// reverting to the wheel.
    pub force_conda: BTreeSet<String>,
    /// Workspace/system ABI-anchor conda pins (`cuda-version`,
    /// `python_abi`, ...), keyed by canonical conda name -> the raw
    /// consuming-env spec (`"==12.8"`), surfaced from
    /// `WorkspaceManifest::consuming_env_dependencies` -- the same source
    /// the proactive cuda-major capping already reads. Checked at route
    /// time by [`route_metadata_consistent`] against a candidate build's
    /// own conda `depends`, so a routed build requiring an anchor version
    /// the env can't provide (conda-only, hence invisible to the pypi
    /// closure) is refused up front instead of surfacing as a ladder
    /// conflict one iteration later. Empty in tests/probes that don't
    /// supply it -- the check then simply has no anchor to contradict.
    pub abi_anchor_pins: BTreeMap<String, String>,
    /// Exact conda name -> version selected identically by every precise
    /// consuming environment, restricted to packages directly declared by
    /// all of them. When the co-install check is unsat and names a routed
    /// package present here at a different version, the route is harmonized
    /// to this agreed fact. Ambiguous ownership, transitive-only names,
    /// failed solves, or cross-environment disagreement leave the name out;
    /// the un-route fallback then behaves exactly as before.
    pub workspace_conda_versions: BTreeMap<String, String>,
    /// Canonical PyPI names directly owned by every precisely identified
    /// consuming workspace environment. Populated only for the validated
    /// routing policy. A successful route for one of these names is an
    /// automatic drop rather than a generated-pack run dependency.
    pub workspace_owned: BTreeSet<String>,
    /// Stable digest of the concrete per-environment conda facts used to
    /// validate routing. It participates in the persisted heal-facts stamp so
    /// a changed workspace solution cannot replay stale routes.
    pub workspace_fact_fingerprint: String,
}

impl Default for AutoRouteOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            // Legacy sweep semantics for bare test/probe construction; the
            // production path wires the config-layer default (`Minimal`)
            // explicitly. See the `route_policy` field doc.
            route_policy: crate::config::RoutePolicy::Aggressive,
            route_include: BTreeSet::new(),
            keep_pypi: BTreeSet::new(),
            protected: BTreeSet::new(),
            name_map: BTreeMap::new(),
            force_conda: BTreeSet::new(),
            abi_anchor_pins: BTreeMap::new(),
            workspace_conda_versions: BTreeMap::new(),
            workspace_owned: BTreeSet::new(),
            workspace_fact_fingerprint: String::new(),
        }
    }
}

/// v4.6 Part A routing-policy gate: may this (pypi name, mapped conda
/// name) candidate be auto-routed to conda?
///
/// `PreferCondaValidated` and `Aggressive` admit every name; the former is
/// subsequently required to pass the fact-validation gates. `Minimal`
/// admits only:
/// - python / python_abi (the interpreter ABI itself),
/// - the torch family (torch, pytorch, pytorch-gpu, pytorch-cpu,
///   torchvision, torchaudio -- checked on BOTH the pypi and the mapped
///   conda spelling, so a `torch -> pytorch-gpu` name-map edge is
///   covered from either side),
/// - the cuda-* family (cuda-version, cuda-toolkit, cudatoolkit, ...),
/// - the pack's explicit `retread-route-include` entries, and
/// - `force-conda` entries (the user asserts these must be conda).
///
/// Everything else ships as a PyPI wheel via the courier -- per-env
/// versions, no conda arbitration (the 43-run imprint campaign's repair
/// ladder empirically un-routed wrapt/moviepy/dm-tree/grpcio/cycler/
/// fsspec/huggingface-hub/...; the fixes converged on this whitelist).
pub fn route_policy_admits(pypi_name: &str, conda_name: &str, opts: &AutoRouteOptions) -> bool {
    if matches!(
        opts.route_policy,
        crate::config::RoutePolicy::PreferCondaValidated | crate::config::RoutePolicy::Aggressive
    ) {
        return true;
    }
    const WHITELIST: &[&str] = &[
        "python",
        "python_abi",
        "python-abi",
        "torch",
        "pytorch",
        "pytorch-gpu",
        "pytorch-cpu",
        "torchvision",
        "torchaudio",
    ];
    let admits_name = |n: &str| {
        WHITELIST.contains(&n) || n.starts_with("cuda-") || n == "cudatoolkit" || n == "cuda"
    };
    admits_name(pypi_name)
        || admits_name(conda_name)
        || opts.route_include.contains(pypi_name)
        || opts.route_include.contains(conda_name)
        || opts.force_conda.contains(pypi_name)
        || opts.force_conda.contains(conda_name)
}

/// Hard cap on auto-route discovery rounds (rounds that GROW the
/// routing set). Growth rounds are capped here; self-healing un-route
/// iterations (see [`auto_route_fixpoint_checked`]) are bounded
/// separately by the closure size, since each one strictly grows the
/// blocked set.
pub const AUTO_ROUTE_MAX_ROUNDS: usize = 5;

/// Per-round probe fan-out bound (matches `probe::probe_many`).
pub const AUTO_ROUTE_PROBE_CONCURRENCY: usize = 16;

/// A successful conda-route probe for the loop: mirrors
/// [`crate::probe::RouteHit`] (redeclared here so this module's loop is
/// testable without touching repodata).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteProbeHit {
    pub conda_version: String,
    pub channel: String,
    /// Route-time metadata-consistency check (prevention for the
    /// PyPI-vs-conda-forge metadata-skew class, see
    /// `route_metadata_consistent`): the raw conda `depends` array of
    /// the routed build, when the caller has it available. Empty when
    /// unavailable (older probes, most tests) -- the check then simply
    /// has nothing to contradict and the route proceeds unchecked,
    /// identical to pre-this-fix behavior.
    pub depends: Vec<String>,
}

fn active_input_requirement(
    raw: &str,
    target: &uv_pep508::MarkerEnvironment,
) -> Result<Option<(String, String)>> {
    let requirement: Requirement = Requirement::from_str(raw)
        .with_context(|| format!("parsing authoritative uv input requirement `{raw}`"))?;
    if !requirement.marker.evaluate(target, &[]) {
        return Ok(None);
    }
    let specifiers = match requirement.version_or_url.as_ref() {
        Some(uv_pep508::VersionOrUrl::VersionSpecifier(specifiers)) => specifiers.to_string(),
        None => String::new(),
        // Direct URL/path artifacts are protected roots or pin-only archive
        // records, never index-wheel auto-route candidates.
        Some(uv_pep508::VersionOrUrl::Url(_)) => return Ok(None),
    };
    Ok(Some((
        canonical_conda_name(requirement.name.as_ref()),
        specifiers,
    )))
}

/// Preserve the effective authoritative uv inputs that produced a closure.
/// The closure wheel versions are deliberately absent: they are solver output
/// and become restoration preferences, not hard requirements.
///
/// uv override semantics are load-bearing here: an active override replaces
/// ordinary root/transitive requirements for that package, while constraints
/// remain additive. Keeping the semantic role lets the later metadata scan
/// apply the same precedence to newly materialized `Requires-Dist` lines.
fn effective_auto_route_input_requirements(
    req: &UvClosureRequest,
) -> Result<BTreeMap<String, Vec<AutoRouteInputRequirement>>> {
    let target = crate::relax::marker_env_for(&req.conda_subdir, &req.python_version)?;
    let mut requirements: BTreeMap<String, Vec<AutoRouteInputRequirement>> = BTreeMap::new();
    let mut constraints: BTreeMap<String, Vec<AutoRouteInputRequirement>> = BTreeMap::new();
    let mut overrides: BTreeMap<String, Vec<AutoRouteInputRequirement>> = BTreeMap::new();
    let push = |destination: &mut BTreeMap<String, Vec<AutoRouteInputRequirement>>,
                raw: &str,
                source: String,
                role: AutoRouteInputRole|
     -> Result<()> {
        let Some((name, specifiers)) = active_input_requirement(raw, &target)? else {
            return Ok(());
        };
        let requirement = AutoRouteInputRequirement {
            specifiers,
            source,
            role,
        };
        let entries = destination.entry(name).or_default();
        if !entries.contains(&requirement) {
            entries.push(requirement);
        }
        Ok(())
    };

    for raw in &req.dependencies {
        push(
            &mut requirements,
            raw,
            format!("uv root requirement `{raw}`"),
            AutoRouteInputRole::Requirement,
        )?;
    }
    for (index, raw) in req.constraints.constraints.iter().enumerate() {
        if req
            .constraints
            .auto_route_constraint_indices
            .contains(&index)
        {
            continue;
        }
        let provenance = req
            .constraints
            .provenance
            .values()
            .find(|provenance| provenance.constraint == *raw);
        let source = match provenance {
            Some(provenance) => format!(
                "uv constraint `{raw}` from {} `{}` (conda `{}{}`)",
                provenance.source, provenance.env, provenance.conda_name, provenance.conda_version
            ),
            None => format!("uv constraint `{raw}`"),
        };
        push(
            &mut constraints,
            raw,
            source,
            AutoRouteInputRole::Constraint,
        )?;
    }
    for raw in &req.overrides {
        push(
            &mut overrides,
            raw,
            format!("uv override requirement `{raw}`"),
            AutoRouteInputRole::Override,
        )?;
    }
    let names: BTreeSet<String> = requirements
        .keys()
        .chain(constraints.keys())
        .chain(overrides.keys())
        .cloned()
        .collect();
    let mut effective = BTreeMap::new();
    for name in names {
        let mut entries = if overrides.contains_key(&name) {
            overrides.remove(&name).unwrap_or_default()
        } else {
            requirements.remove(&name).unwrap_or_default()
        };
        entries.extend(constraints.remove(&name).unwrap_or_default());
        effective.insert(name, entries);
    }
    Ok(effective)
}

fn attach_effective_input_requirements(
    closure: &mut UvClosure,
    req: &UvClosureRequest,
) -> Result<()> {
    if closure.effective_input_requirements.is_none() {
        closure.effective_input_requirements = Some(effective_auto_route_input_requirements(req)?);
    }
    Ok(())
}

fn hydrate_route_input_requirements(routes: &mut [AutoRoutedPackage], closure: &UvClosure) {
    let Some(inputs) = &closure.effective_input_requirements else {
        return;
    };
    for route in routes {
        route.input_requirements = inputs
            .get(&canonical_conda_name(&route.pypi_name))
            .cloned()
            .unwrap_or_default();
    }
}

/// Pure routing sweep over one solved closure: which index wheels can
/// move to the conda side this round?
///
/// A wheel is a CANDIDATE unless it is already excluded
/// (`no_emit_packages`), user-kept (`keep-pypi`), protected (root /
/// built wheel), or routed in a previous round. For each candidate the
/// caller-supplied `probe_hits` map (conda name -> hit, from repodata)
/// decides: present = route it.
pub fn plan_auto_route_round(
    closure: &UvClosure,
    req: &UvClosureRequest,
    opts: &AutoRouteOptions,
    already_routed: &[AutoRoutedPackage],
    probe_hits: &BTreeMap<String, RouteProbeHit>,
) -> Result<Vec<AutoRoutedPackage>> {
    let effective_inputs = match &closure.effective_input_requirements {
        Some(inputs) => inputs.clone(),
        None => effective_auto_route_input_requirements(req)?,
    };
    let excluded: BTreeSet<String> = req
        .no_emit_packages
        .iter()
        .map(|n| canonical_conda_name(n))
        .collect();
    let mut out = Vec::new();
    for wheel in &closure.wheels {
        let name = &wheel.name; // canonical (parse_pylock_closure output)
        if excluded.contains(name)
            || opts.keep_pypi.contains(name)
            || opts.protected.contains(name)
            || already_routed.iter().any(|r| &r.pypi_name == name)
        {
            continue;
        }
        let conda_name = opts
            .name_map
            .get(name)
            .map(|c| canonical_conda_name(c))
            .unwrap_or_else(|| name.clone());
        // v4.6 Part A: the routing-policy gate. Under `Minimal`, a
        // non-whitelisted candidate never routes regardless of conda
        // availability -- it ships as a wheel.
        if !route_policy_admits(name, &conda_name, opts) {
            continue;
        }
        if let Some(hit) = probe_hits.get(&conda_name) {
            // Route-time metadata-consistency check (prevention for the
            // PyPI-vs-conda-forge metadata-skew class -- run-15's
            // moviepy==2.2.1/pillow==11.3.0 shape): the routed build's
            // OWN conda `depends` might name a version of something the
            // closure ALREADY locked that its own recipe can't actually
            // accept (conda-forge's repackage metadata lagging the
            // upstream PyPI release). Refuse the route rather than ship
            // a conda run-dep the workspace solve is guaranteed to choke
            // on one iteration later; the package stays on PyPI instead.
            if let Err(reason) = route_metadata_consistent(
                name,
                &wheel.version,
                &hit.depends,
                &closure.pins,
                &opts.abi_anchor_pins,
            ) {
                eprintln!(
                    "retread: route refused for {name}=={} — {reason}",
                    wheel.version
                );
                continue;
            }
            out.push(AutoRoutedPackage {
                pypi_name: name.clone(),
                conda_name,
                pypi_version: wheel.version.clone(),
                conda_version: hit.conda_version.clone(),
                channel: hit.channel.clone(),
                input_requirements: effective_inputs.get(name).cloned().unwrap_or_default(),
            });
        }
    }
    Ok(out)
}

/// One-level-deep route-time metadata-consistency check (prevention
/// counterpart to the CLI fallback engine's UN-ROUTE cure): before
/// routing `candidate`==`candidate_version` to conda, walk the conda
/// build's own `depends` array and check each entry naming a package the
/// uv closure ALREADY locked (`locked_versions`, PEP 503-normalized name
/// -> resolved version -- `UvClosure::pins`) against that dep's version
/// spec. A mismatch means the routed build's own recipe metadata
/// contradicts the closure the rest of the workspace already committed
/// to -- almost always a conda-forge repackage whose pinned dependency
/// range lags the upstream PyPI release the wheel closure resolved
/// against (the run-15 shape: conda `moviepy-2.2.1`'s own recipe still
/// says `pillow <11.0`, but PyPI's `moviepy 2.2.1` legally allows
/// `pillow <12`, and the closure already locked `pillow==11.3.0`).
///
/// Deliberately shallow (one level): only `depends`'s own entries are
/// checked, never their OWN transitive dependencies -- a contradiction
/// found only several levels down still reaches the ordinary solve
/// attempt as a ladder conflict, for the CLI's generic-repair UN-ROUTE
/// candidate to cure after the fact (`retread lock`'s fallback engine).
/// Unrecognized dep entries (bad name, unparseable spec, `*`/empty
/// range, or a name the closure never locked at all) are silently
/// skipped rather than treated as a contradiction -- this check only
/// ever REFUSES on a provable mismatch, never on missing information.
pub fn route_metadata_consistent(
    candidate: &str,
    candidate_version: &str,
    depends: &[String],
    locked_versions: &BTreeMap<String, String>,
    abi_anchor_pins: &BTreeMap<String, String>,
) -> std::result::Result<(), String> {
    use rattler_conda_types::{ParseStrictness, Version, VersionSpec};
    use std::str::FromStr;
    let _ = candidate_version;

    for dep in depends {
        let trimmed = dep.trim();
        // Conda dep lines are `name version build` (3 whitespace-separated
        // fields, build optional) -- e.g. `python_abi 3.13.* *_cp313`. A
        // 2-way split leaves the build string glued onto the version
        // (`"3.13.* *_cp313"`), which `VersionSpec::from_str` cannot parse,
        // silently falling through the `let Ok(spec) = ... else { continue }`
        // guard below and disarming this entire check for every anchor that
        // carries a build tag (python_abi always does). Match
        // `split_conda_dep_line`'s (`workspace.rs`) 3-way split so the build
        // field is discarded here too.
        let mut parts = trimmed.splitn(3, char::is_whitespace);
        let Some(raw_name) = parts.next() else {
            continue;
        };
        if raw_name.is_empty() {
            continue;
        }
        let dep_name = canonical_conda_name(raw_name);
        if dep_name == canonical_conda_name(candidate) {
            continue;
        }
        let spec_str = parts.next().unwrap_or("").trim();
        if spec_str.is_empty() || spec_str == "*" {
            continue;
        }
        let Ok(spec) = VersionSpec::from_str(spec_str, ParseStrictness::Lenient) else {
            continue;
        };
        if let Some(locked_version) = locked_versions.get(&dep_name) {
            // The pypi-closure path: the routed build's own recipe range
            // vs. the version the rest of the workspace already locked
            // (the run-15 conda-forge repackage metadata skew).
            let Ok(locked) = Version::from_str(locked_version) else {
                continue;
            };
            if !spec.matches(&locked) {
                return Err(format!(
                    "conda repackage metadata ({dep_name} {spec_str}) contradicts locked closure ({dep_name} {locked_version})"
                ));
            }
            continue;
        }
        // Conda-only ABI anchor (`cuda-version`, `python_abi`, ...) the
        // pypi closure never locks -- previously invisible to this check,
        // so a routed build requiring an anchor version the env can't
        // provide (run 16c: `triton ==3.6.0` needs `cuda-version
        // >=12.9,<13`, workspace pins 12.8) sailed through route-time and
        // only surfaced as an unparseable ladder conflict later. Check the
        // build's requirement against the workspace's concretely pinned
        // anchor instead. Only an EXACT workspace pin arms the check, so it
        // never over-refuses a range the build might legally fall inside.
        //
        // Deliberately keyed by `raw_name`, NOT the PEP503-canonicalized
        // `dep_name`: `ABI_ANCHOR_NAMES` and `abi_anchor_pins` are both
        // populated from raw conda dependency identifiers, which use
        // underscores conda-side names never do (`python_abi`, the
        // `gcc_`/`sysroot_`-prefixed compiler families) -- canonicalizing
        // to `dep_name` turns `python_abi` into `python-abi`, which
        // matches neither `ABI_ANCHOR_NAMES`'s literal entry nor the
        // anchor map's key, silently disarming the check for every
        // underscore-named anchor (run 17's root cause for `python_abi`).
        if crate::solve::is_abi_anchor(raw_name)
            && let Some(anchor_spec) = abi_anchor_pins.get(raw_name)
            && let Some(pinned) = exact_anchor_version(anchor_spec)
            && !spec.matches(&pinned)
        {
            return Err(format!(
                "routed build's conda depend ({dep_name} {spec_str}) contradicts the workspace ABI anchor ({dep_name} {anchor_spec})"
            ));
        }
    }
    Ok(())
}

/// The single exact version an ABI-anchor workspace pin names, if it is
/// an exact pin (`==X`, `=X`, or a bare `X`) -- `None` for ranges,
/// globs, or any inequality operator. Keeps [`route_metadata_consistent`]'s
/// ABI-anchor branch to a PROVABLE contradiction: an exact env pin the
/// routed build's requirement demonstrably excludes, never a range the
/// build might still satisfy.
fn exact_anchor_version(spec: &str) -> Option<rattler_conda_types::Version> {
    use rattler_conda_types::Version;
    use std::str::FromStr;
    let s = spec.trim();
    if s.is_empty() || s == "*" || s.contains(',') {
        return None;
    }
    let v = s
        .strip_prefix("==")
        .or_else(|| s.strip_prefix('='))
        .unwrap_or(s)
        .trim();
    if v.is_empty() || v.contains('*') || v.starts_with(['>', '<', '!', '~', '=']) {
        return None;
    }
    Version::from_str(v).ok()
}

/// The conda names + `==version` specs to probe for one round's
/// candidates (everything [`plan_auto_route_round`] would consider).
/// Returned as (conda_name, spec) pairs, deduped.
pub fn auto_route_probe_specs(
    closure: &UvClosure,
    req: &UvClosureRequest,
    opts: &AutoRouteOptions,
    already_routed: &[AutoRoutedPackage],
) -> Vec<(String, String)> {
    let excluded: BTreeSet<String> = req
        .no_emit_packages
        .iter()
        .map(|n| canonical_conda_name(n))
        .collect();
    let mut out: Vec<(String, String)> = Vec::new();
    for wheel in &closure.wheels {
        let name = &wheel.name;
        if excluded.contains(name)
            || opts.keep_pypi.contains(name)
            || opts.protected.contains(name)
            || already_routed.iter().any(|r| &r.pypi_name == name)
        {
            continue;
        }
        let conda_name = opts
            .name_map
            .get(name)
            .map(|c| canonical_conda_name(c))
            .unwrap_or_else(|| name.clone());
        // v4.6 Part A: don't even probe repodata for candidates the
        // routing policy refuses -- they ship as wheels.
        if !route_policy_admits(name, &conda_name, opts) {
            continue;
        }
        let pair = (conda_name, format!("=={}", wheel.version));
        if !out.contains(&pair) {
            out.push(pair);
        }
    }
    out
}

/// Apply one round's routing decisions to the request: each routed
/// package is excluded from the exported closure (`--no-emit-package`)
/// and its resolved version becomes a `constraint-dependencies` pin (so
/// the re-lock cannot drift the rest of the graph away from what conda
/// will provide). Provenance records `source = "auto-route"`.
pub fn apply_auto_route(req: &mut UvClosureRequest, hits: &[AutoRoutedPackage]) {
    for h in hits {
        req.no_emit_packages.push(h.pypi_name.clone());
        let line = format!("{}=={}", h.pypi_name, h.pypi_version);
        let index = req.constraints.constraints.len();
        req.constraints.constraints.push(line.clone());
        req.constraints.auto_route_constraint_indices.insert(index);
        req.constraints
            .provenance
            .entry(h.pypi_name.clone())
            .or_insert_with(|| ConstraintProvenance {
                constraint: line,
                conda_name: h.conda_name.clone(),
                conda_version: format!("=={}", h.conda_version),
                source: "auto-route".to_string(),
                env: "default".to_string(),
                advisory: false,
            });
    }
}

/// spec-uv-restructure M2 outer loop: solve, sweep the closure for
/// wheels a workspace channel can provide at the resolved version, move
/// them to the conda side, re-solve, repeat to fixpoint (max
/// [`AUTO_ROUTE_MAX_ROUNDS`] discovery rounds — the routing set only
/// grows, so this terminates).
///
/// * `solve`: computes a closure for the (progressively updated)
///   request — production wires [`compute_closure`]; tests can can it.
/// * `probe`: `(conda_name, "==version")` -> conda availability —
///   production wires `crate::probe::find_route`; tests can can it.
///
/// The returned closure's `auto_routed` lists every moved package;
/// each move is logged as `auto-routed X==v to conda (channel C)`.
///
/// Compatibility wrapper over [`auto_route_fixpoint_checked`] with the
/// co-installability check disabled (every routing round is accepted).
pub async fn auto_route_fixpoint<S, P>(
    req: &UvClosureRequest,
    opts: &AutoRouteOptions,
    solve: S,
    probe: P,
) -> Result<UvClosure>
where
    S: FnMut(UvClosureRequest) -> futures::future::BoxFuture<'static, Result<UvClosure>>,
    P: Fn(String, String) -> futures::future::BoxFuture<'static, Option<RouteProbeHit>>,
{
    auto_route_fixpoint_checked(req, opts, solve, probe, |_| {
        Box::pin(async { CoInstallVerdict::Skipped("co-solve not wired".into()) })
    })
    .await
}

/// Verdict of the conda co-installability check over one candidate
/// auto-routed set (exact `conda_name ==conda_version` pins solved
/// together against the workspace channels).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoInstallVerdict {
    /// The pins co-install: accept the round.
    Sat,
    /// The conda solver reported unsat; the reason strings are scanned
    /// for auto-routed names to un-route.
    Unsat(Vec<String>),
    /// The check could not run (no repodata on disk, offline, ...).
    /// Routing proceeds UNCHECKED — identical to pre-check behavior —
    /// so a missing cache can never veto a build. Deterministic given
    /// the same cache state.
    Skipped(String),
}

/// One conda route considered for generated run-dependency emission.
///
/// Unlike [`AutoRoutedPackage`], this type also represents dependencies
/// which were already destined for conda before Rule 2 considered moving a
/// PyPI dependency there. `spec` is the version portion of the conda match
/// spec; an empty string and `*` both mean unconstrained.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CondaRouteSpec {
    pub pypi_name: String,
    pub conda_name: String,
    pub spec: String,
}

impl CondaRouteSpec {
    /// Render this route as a conda match spec suitable for the shared
    /// co-installability oracle.
    pub fn match_spec(&self) -> String {
        let spec = self.spec.trim();
        if spec.is_empty() || spec == "*" {
            self.conda_name.clone()
        } else {
            format!("{} {spec}", self.conda_name)
        }
    }
}

/// Strict Rule 2 routing outcome. Accepted dependencies may be emitted as
/// conda routes; rejected dependencies must remain on PyPI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JointRouteSelection {
    pub accepted: Vec<CondaRouteSpec>,
    pub rejected: Vec<CondaRouteSpec>,
}

/// Select Rule 2 conda routes only when they co-solve with the dependencies
/// already fixed on the conda side.
///
/// The common case costs one solve over the complete set. If that set is not
/// satisfiable, the fixed baseline must first prove satisfiable; otherwise
/// there is no sound candidate-level decision and `None` is returned. From a
/// satisfiable baseline, deletion-minimal unsatisfiable candidate groups are
/// isolated and rejected together until the remaining set co-solves. This
/// preserves unrelated routes while handling conflicts that cannot be healed
/// by removing only one member. Only a positive [`CoInstallVerdict::Sat`]
/// may move a dependency to conda; skipped probes conservatively keep every
/// unresolved candidate on PyPI.
pub async fn select_jointly_solvable_routes<C, F>(
    fixed: Vec<CondaRouteSpec>,
    candidates: Vec<CondaRouteSpec>,
    co_solve: &C,
) -> Option<JointRouteSelection>
where
    C: Fn(Vec<CondaRouteSpec>) -> F,
    F: std::future::Future<Output = CoInstallVerdict>,
{
    let fixed: Vec<CondaRouteSpec> = fixed
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let candidates: Vec<CondaRouteSpec> = candidates
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let combined = |accepted: &[CondaRouteSpec]| {
        fixed
            .iter()
            .chain(accepted)
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    };

    if matches!(
        co_solve(combined(&candidates)).await,
        CoInstallVerdict::Sat
    ) {
        return Some(JointRouteSelection {
            accepted: candidates,
            rejected: Vec::new(),
        });
    }

    if !matches!(co_solve(fixed.clone()).await, CoInstallVerdict::Sat) {
        return None;
    }

    let mut rejected = Vec::new();
    let mut remaining = candidates;
    loop {
        match co_solve(combined(&remaining)).await {
            CoInstallVerdict::Sat => {
                return Some(JointRouteSelection {
                    accepted: remaining,
                    rejected,
                });
            }
            CoInstallVerdict::Skipped(_) => {
                rejected.extend(remaining);
                return Some(JointRouteSelection {
                    accepted: Vec::new(),
                    rejected,
                });
            }
            CoInstallVerdict::Unsat(_) => {}
        }

        // Reduce the failing set to a deletion-minimal unsatisfiable core:
        // if removing one route leaves the trial unsatisfiable, that route
        // is unrelated to this core and stays eligible for a later round.
        let mut core = remaining.clone();
        for candidate in remaining.clone() {
            let Some(pos) = core.iter().position(|route| route == &candidate) else {
                continue;
            };
            let mut trial = core.clone();
            trial.remove(pos);
            if matches!(
                co_solve(combined(&trial)).await,
                CoInstallVerdict::Unsat(_)
            ) {
                core = trial;
            }
        }

        // A satisfiable fixed baseline makes an empty unsat core impossible,
        // but fail closed if an inconsistent oracle reports one.
        if core.is_empty() {
            rejected.extend(remaining);
            return Some(JointRouteSelection {
                accepted: Vec::new(),
                rejected,
            });
        }
        let core_set: BTreeSet<_> = core.iter().cloned().collect();
        rejected.extend(core);
        remaining.retain(|route| !core_set.contains(route));
    }
}

/// Does one resolvo unsat-reason line name this conda package?
///
/// Reasons embed names either as `name ==ver` / `name >=ver` match
/// specs or as `name-version-build` filenames. Plain substring search
/// over-matches (`ray` in `ray-core`, `numpy` in `numpydoc`), so an
/// occurrence only counts when:
///   * the char before it is not part of a conda name
///     (`[a-z0-9._-]`), and
///   * the char after it is either not a name char, or is `-`
///     immediately followed by a digit (the `name-1.2.3-build`
///     filename form; `ray-core` stays rejected).
pub fn unsat_reason_names_package(reason: &str, conda_name: &str) -> bool {
    // `-`/`_`/`.` are equivalent under conda name normalization
    // (typing_extensions vs typing-extensions): fold both sides to `-`
    // before matching. Versions in the reason get mangled by the fold,
    // but only name-occurrence boundaries matter here.
    let fold = |s: &str| s.to_ascii_lowercase().replace(['_', '.'], "-");
    let hay = fold(reason);
    let needle = fold(conda_name);
    if needle.is_empty() {
        return false;
    }
    let is_name_char = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-');
    let bytes = hay.as_bytes();
    let mut from = 0usize;
    while let Some(pos) = hay[from..].find(&needle) {
        let start = from + pos;
        let end = start + needle.len();
        let before_ok = start == 0 || !is_name_char(hay[..start].chars().next_back().unwrap());
        let after_ok = match hay[end..].chars().next() {
            None => true,
            Some(c) if !is_name_char(c) => true,
            Some('-') => hay[end + 1..]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit()),
            Some(_) => false,
        };
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
        if from >= bytes.len() {
            break;
        }
    }
    false
}

/// spec-uv-restructure M2 loop with the self-healing un-route step
/// (fix for the exact-pin co-installability blocker): after each round
/// that would GROW the routing set, `co_solve` checks that the whole
/// candidate routed set (exact conda pins) still co-installs on the
/// workspace channels.
///
/// * `Sat` / `Skipped`: the round is applied as before.
/// * `Unsat(reasons)`: every candidate whose conda name appears in the
///   unsat report is UN-ROUTED — moved back into the wheel closure and
///   blocked from all future rounds — then the request is rebuilt from
///   the surviving routes and the closure re-locked. `force-conda`
///   names are exempt (warned, kept routed). If no routed name is
///   named, the report is about something else entirely (channel-side
///   breakage the un-route cannot heal); the round is applied with a
///   warning so the workspace solve surfaces the real error.
///
/// Determinism: the verdict is a pure function of the candidate pin
/// set + the channel repodata snapshot (and `Skipped` degrades to the
/// old unchecked behavior), so identical inputs give identical routing.
/// Termination: growth rounds are capped at [`AUTO_ROUTE_MAX_ROUNDS`];
/// every un-route iteration strictly grows the blocked set, which is
/// bounded by the closure size.
/// Rebuild the closure request from scratch: base request + surviving
/// routes + workspace-harmonized pypi pins (run-34; see
/// [`AutoRouteOptions::workspace_conda_versions`]). Every rebuild site
/// in the fixpoint goes through this so a harmonized pin survives
/// un-route rebuilds and vice versa.
fn rebuild_routed_request(
    base: &UvClosureRequest,
    routed: &[AutoRoutedPackage],
    harmonize_pins: &BTreeMap<String, (String, String)>,
) -> UvClosureRequest {
    let mut req = base.clone();
    apply_auto_route(&mut req, routed);
    for (pypi_name, (version, conda_name)) in harmonize_pins {
        let line = format!("{pypi_name}=={version}");
        req.constraints.constraints.push(line.clone());
        req.constraints.provenance.insert(
            pypi_name.clone(),
            ConstraintProvenance {
                constraint: line,
                conda_name: conda_name.clone(),
                conda_version: format!("=={version}"),
                source: "workspace-harmonize".to_string(),
                env: "default".to_string(),
                advisory: false,
            },
        );
    }
    req
}

pub async fn auto_route_fixpoint_checked<S, P, C>(
    req: &UvClosureRequest,
    opts: &AutoRouteOptions,
    mut solve: S,
    probe: P,
    co_solve: C,
) -> Result<UvClosure>
where
    S: FnMut(UvClosureRequest) -> futures::future::BoxFuture<'static, Result<UvClosure>>,
    P: Fn(String, String) -> futures::future::BoxFuture<'static, Option<RouteProbeHit>>,
    C: Fn(Vec<AutoRoutedPackage>) -> futures::future::BoxFuture<'static, CoInstallVerdict>,
{
    let base_req = req.clone();
    let mut req = req.clone();
    let mut closure = solve(req.clone()).await?;
    if !opts.enabled {
        return Ok(closure);
    }
    let mut routed: Vec<AutoRoutedPackage> = Vec::new();
    // PyPI names the un-route step moved back to the wheel closure;
    // excluded from every later round exactly like `keep-pypi`.
    let mut blocked: BTreeSet<String> = BTreeSet::new();
    // Workspace-harmonized pins (run-34): pypi_name -> (version, conda
    // name). Applied as uv `constraint-dependencies` on every rebuild so
    // the closure re-locks the package at the version the consuming
    // envs' workspace solve provides (hand-written pins win). Each name
    // is harmonized at most once (`harmonize_tried`); if it is named in
    // an unsat core AGAIN afterwards, the un-route fallback takes over.
    let mut harmonize_pins: BTreeMap<String, (String, String)> = BTreeMap::new();
    let mut harmonize_tried: BTreeSet<String> = BTreeSet::new();
    let mut growth_rounds = 0usize;
    // Belt-and-braces bound: growth rounds + one iteration per possible
    // blocked name. The loop provably progresses (each iteration grows
    // `routed`, grows `blocked`, or breaks), so this never fires.
    let hard_cap = AUTO_ROUTE_MAX_ROUNDS + closure.wheels.len() + 8;
    for _iteration in 0..hard_cap {
        // Effective opt-out set for this iteration: user keep-pypi plus
        // everything the self-heal has blocked so far.
        let mut eff_opts = opts.clone();
        eff_opts.keep_pypi.extend(blocked.iter().cloned());
        // Probe every still-unrouted closure wheel's conda equivalent at
        // the resolved version — the WHOLE round's candidate batch runs
        // concurrently (same 16-way bound as `probe_many`; repodata is
        // in-memory-cached after the first hit per (channel, subdir), so
        // this collapses a cold 100-package sweep from serial fetch
        // latency to a handful of parallel fetches). The round loop
        // itself stays serial: each re-lock depends on the previous
        // round's routing set.
        use futures::stream::{self, StreamExt};
        let batch: Vec<(String, Option<RouteProbeHit>)> =
            stream::iter(auto_route_probe_specs(&closure, &req, &eff_opts, &routed))
                .map(|(conda_name, spec)| {
                    let fut = probe(conda_name.clone(), spec);
                    async move { (conda_name, fut.await) }
                })
                .buffer_unordered(AUTO_ROUTE_PROBE_CONCURRENCY)
                .collect()
                .await;
        let mut hits: BTreeMap<String, RouteProbeHit> = BTreeMap::new();
        for (conda_name, hit) in batch {
            if let Some(hit) = hit {
                hits.insert(conda_name, hit);
            }
        }
        let new_routes = plan_auto_route_round(&closure, &req, &eff_opts, &routed, &hits)?;
        if new_routes.is_empty() {
            break; // fixpoint
        }
        // Candidate set = surviving routes + this round's additions;
        // the co-install check always sees the CUMULATIVE pin set (a
        // new pin can conflict with a pin accepted rounds ago).
        let mut candidate = routed.clone();
        candidate.extend(new_routes.iter().cloned());
        let verdict = co_solve(candidate.clone()).await;
        if let CoInstallVerdict::Unsat(reasons) = &verdict {
            let mut named: Vec<AutoRoutedPackage> = Vec::new();
            for pkg in &candidate {
                let in_report = reasons
                    .iter()
                    .any(|r| unsat_reason_names_package(r, &pkg.conda_name));
                if !in_report {
                    continue;
                }
                if opts.force_conda.contains(&pkg.pypi_name) {
                    tracing::warn!(
                        bundle = %req.bundle,
                        package = %pkg.pypi_name,
                        "auto-route: {} =={} is named in the conda unsat \
                         report but is force-conda; keeping it routed — \
                         the workspace solve may fail",
                        pkg.conda_name,
                        pkg.conda_version,
                    );
                    continue;
                }
                named.push(pkg.clone());
            }
            if !named.is_empty() {
                // Run-34 doctrine split: a named candidate whose conda
                // name the WORKSPACE solve pins at a different version
                // is HARMONIZED (pypi side re-pinned to the workspace
                // version, closure re-locked, route re-planned at the
                // right version next round) rather than un-routed --
                // un-routing would ship a wheel at the conflicting
                // version that clobbers the conda package the workspace
                // installs (torch 2.10.0 wheel over conda pytorch
                // 2.7.0, pulled by the hand-written pytorch-gpu
                // ==2.7.0). Each name gets ONE harmonization attempt;
                // named again afterwards -> un-route fallback.
                let mut round_harmonized: Vec<String> = Vec::new();
                for pkg in &named {
                    let ws_version = opts.workspace_conda_versions.get(&pkg.conda_name);
                    if let Some(ws_version) = ws_version
                        && ws_version != &pkg.conda_version
                        && !harmonize_tried.contains(&pkg.pypi_name)
                    {
                        harmonize_tried.insert(pkg.pypi_name.clone());
                        harmonize_pins.insert(
                            pkg.pypi_name.clone(),
                            (ws_version.clone(), pkg.conda_name.clone()),
                        );
                        round_harmonized.push(pkg.pypi_name.clone());
                        tracing::info!(
                            bundle = %req.bundle,
                            "auto-route: workspace-harmonized {} {} -> {} \
                             (consuming envs' workspace solve provides conda \
                             {} =={}; hand-written pins win over the uv pick)",
                            pkg.pypi_name,
                            pkg.pypi_version,
                            ws_version,
                            pkg.conda_name,
                            ws_version,
                        );
                        continue;
                    }
                    tracing::info!(
                        bundle = %req.bundle,
                        "auto-route: un-routed {}=={} (conda {} =={}) — \
                         exact conda pins do not co-install; back to the \
                         wheel closure",
                        pkg.pypi_name,
                        pkg.pypi_version,
                        pkg.conda_name,
                        pkg.conda_version,
                    );
                    blocked.insert(pkg.pypi_name.clone());
                }
                let previously_routed = routed.len();
                routed.retain(|r| !blocked.contains(&r.pypi_name));
                if routed.len() != previously_routed || !round_harmonized.is_empty() {
                    // A previously ACCEPTED route was un-routed and/or a
                    // harmonized pin was added: rebuild the request from
                    // scratch (an un-routed exclusion + constraint must
                    // disappear; a harmonized constraint must appear)
                    // and re-lock so wheels rejoin/re-pick accordingly.
                    req = rebuild_routed_request(&base_req, &routed, &harmonize_pins);
                    match solve(req.clone()).await {
                        Ok(c) => closure = c,
                        Err(e) if !round_harmonized.is_empty() => {
                            // The harmonized version has no satisfiable
                            // pypi-side pick (no wheel / metadata refuses
                            // it). Drop THIS round's harmonized pins and
                            // un-route those packages instead -- the
                            // pre-harmonization fallback behavior.
                            tracing::warn!(
                                bundle = %req.bundle,
                                error = %e,
                                "auto-route: workspace-harmonized re-lock \
                                 failed; falling back to un-route for {:?}",
                                round_harmonized,
                            );
                            for name in &round_harmonized {
                                harmonize_pins.remove(name);
                                blocked.insert(name.clone());
                            }
                            routed.retain(|r| !blocked.contains(&r.pypi_name));
                            req = rebuild_routed_request(&base_req, &routed, &harmonize_pins);
                            closure = solve(req.clone()).await?;
                        }
                        Err(e) => return Err(e),
                    }
                }
                // This round's additions were never applied to `req`;
                // re-plan next iteration with the blocked set in force.
                continue;
            }
            // The unsat report names none of the routed candidates
            // directly — the cuda-bindings/cuda-version shape: the
            // solver's phrasing can point at a shared TRANSITIVE
            // anchor (`cuda-version >=13,<14, for which no candidates
            // were found`) rather than the routed package that pulled
            // it in, so text-matching alone finds nothing to blame.
            // Fall back to a greedy retry-solve: drop routed
            // candidates one at a time and re-check; the first
            // single-candidate removal that turns the verdict
            // Sat/Skipped is blamed and un-routed, repeated until
            // fixpoint or the candidate set is exhausted. This is the
            // same co_solve the name-match path already uses — just
            // driven by outcome instead of by string search — so it
            // catches conflicts whose unsat text never mentions an
            // auto-routed name at all.
            let mut greedy: Vec<AutoRoutedPackage> = Vec::new();
            let mut probe_set = candidate.clone();
            loop {
                // Re-check the CURRENT probe_set before trying another
                // removal — once it's already Sat/Skipped, stop; a
                // single-candidate subset of an already-satisfiable set
                // is not evidence that subset was the offender.
                if !matches!(
                    co_solve(probe_set.clone()).await,
                    CoInstallVerdict::Unsat(_)
                ) {
                    break;
                }
                let mut healed_idx: Option<usize> = None;
                for (i, pkg) in probe_set.iter().enumerate() {
                    if opts.force_conda.contains(&pkg.pypi_name) {
                        continue;
                    }
                    let mut trial = probe_set.clone();
                    trial.remove(i);
                    if !matches!(co_solve(trial).await, CoInstallVerdict::Unsat(_)) {
                        healed_idx = Some(i);
                        break;
                    }
                }
                let Some(i) = healed_idx else { break };
                let removed = probe_set.remove(i);
                tracing::info!(
                    bundle = %req.bundle,
                    "auto-route: greedy retry-solve isolated {}=={} as the \
                     transitive-conflict offender (unsat report named no \
                     auto-routed package directly)",
                    removed.pypi_name,
                    removed.pypi_version,
                );
                greedy.push(removed);
            }
            if !greedy.is_empty() {
                for pkg in &greedy {
                    blocked.insert(pkg.pypi_name.clone());
                }
                let previously_routed = routed.len();
                routed.retain(|r| !blocked.contains(&r.pypi_name));
                if routed.len() != previously_routed {
                    req = rebuild_routed_request(&base_req, &routed, &harmonize_pins);
                    closure = solve(req.clone()).await?;
                }
                continue;
            }
            tracing::warn!(
                bundle = %req.bundle,
                reasons = ?reasons,
                "auto-route: conda co-install check is unsat but names no \
                 auto-routed package and greedy retry-solve found no \
                 single-candidate fix; un-routing cannot heal this — \
                 applying the round unchanged",
            );
        } else if let CoInstallVerdict::Skipped(why) = &verdict {
            tracing::debug!(
                bundle = %req.bundle,
                why = %why,
                "auto-route: co-install check skipped; routing unchecked",
            );
        }
        growth_rounds += 1;
        for h in &new_routes {
            tracing::info!(
                bundle = %req.bundle,
                round = growth_rounds,
                "auto-routed {}=={} to conda (channel {})",
                h.pypi_name,
                h.pypi_version,
                h.channel,
            );
        }
        apply_auto_route(&mut req, &new_routes);
        routed = candidate;
        // Re-lock with the updated exclusions + constraints so transitive
        // deps only the routed wheels pulled in fall out of (or shift
        // within) the closure.
        closure = solve(req.clone()).await?;
        if growth_rounds == AUTO_ROUTE_MAX_ROUNDS {
            tracing::warn!(
                bundle = %req.bundle,
                rounds = AUTO_ROUTE_MAX_ROUNDS,
                "auto-route: reached the round cap before fixpoint; \
                 remaining closure wheels stay on PyPI",
            );
            break;
        }
    }
    closure.auto_dropped.extend(
        routed
            .iter()
            .filter(|route| opts.workspace_owned.contains(&route.pypi_name))
            .map(|route| route.pypi_name.clone()),
    );
    closure.auto_routed = routed;
    Ok(closure)
}

// ---------------------------------------------------------------------------
// Self-heal detection: sdist-only + transitive-prerelease closure failures
// ---------------------------------------------------------------------------
//
// Two classes of `uv lock` failure are self-healed by re-solving with a
// pin/route/build rung (see `with_sdist_heal`) instead of surfacing to the
// user:
//
//   * SDIST-ONLY: a transitive dependency publishes NO usable wheel (pure
//     sdist, e.g. `pyperclip`, or a wheel-less version band pinned by a
//     requirer's metadata). Under `--no-build` (spec §8.2 -- retread's
//     source_build path owns builds) uv fails the WHOLE lock loudly, even
//     when the sdist-only package is a deep transitive nobody asked for.
//   * TRANSITIVE PRERELEASE: an exact pre-release pin declared by a
//     TRANSITIVE dependency (e.g. isaacsim-core pinning
//     `tinyobjloader==2.0.0rc13`). uv's `--prerelease
//     if-necessary-or-explicit` only honors a pre-release specifier in a
//     FIRST-PARTY (project) requirement, and "if-necessary" only fires
//     when a package has zero stable releases -- so a transitive exact-rc
//     pin still fails even with the flag set (issue #10, second half).
//
// DETECTION IS STRUCTURED, NOT PROSE-PARSED (issue #10 root cause). The
// old approach regex-matched uv's human-readable stderr for a handful of
// phrasings; any message uv worded differently fell through and failed the
// whole lock. Instead, `compute_closure` runs a SECOND `uv lock` (Pass B)
// with the offending restrictions relaxed (drop `--no-build`, set
// `--prerelease allow`); if Pass B succeeds it EXPORTS the resolved lock
// and reads, from the pylock document itself, exactly which packages
// resolved sdist-only and which resolved to a pre-release version. That
// structured offender set is carried to `with_sdist_heal` as a
// [`HealNeeded`] error, replacing all stderr regex parsing. The manylinux
// platform-tag ceiling stays a DIFFERENT recovery layer's job (glibc
// relaxation, `installer::is_platform_tag_conflict`) and never enters the
// two-pass at all.

/// The two offender classes [`classify_pylock_offenders`] extracts from a
/// Pass-B export.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClosureOffenders {
    /// Packages that resolved WITHOUT a usable wheel (sdist-only):
    /// `(canonical pypi name, resolved version)`.
    pub sdist_only: Vec<(String, String)>,
    /// Packages that resolved to a PEP 440 pre-release version:
    /// `(canonical pypi name, resolved version)`.
    pub prerelease: Vec<(String, String)>,
}

/// Structured outcome of the two-pass detection in [`compute_closure`]:
/// Pass A (`uv lock` with retread's normal restrictions) FAILED, but Pass
/// B (the same lock with the offending restrictions relaxed) SUCCEEDED,
/// and its exported lock was inspected to name the offenders exactly.
/// Carried as an `anyhow` error so [`with_sdist_heal`] can `downcast` it
/// and drive the heal rungs; any OTHER error class (a genuine resolution
/// conflict, a platform-tag ceiling, Pass B also failing) surfaces as an
/// ordinary error and passes straight through the heal wrapper unchanged.
#[derive(Debug, Clone)]
pub struct HealNeeded {
    /// sdist-only offenders. Empty under `sdist-build = "never"` (Pass B
    /// keeps `--no-build`, so sdist-only packages never resolve there --
    /// only the prerelease relaxation applies).
    pub sdist_only: Vec<(String, String)>,
    /// Prerelease offenders. Healed by injecting an explicit first-party
    /// `name==version` pin so uv's `explicit` prerelease policy honors it
    /// on the next Pass A.
    pub prerelease: Vec<(String, String)>,
    /// Pass A's formatted failure, surfaced verbatim if the heal cannot
    /// make progress (never silently drops a dependency).
    pub original_error: String,
}

impl std::fmt::Display for HealNeeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.original_error)
    }
}

impl std::error::Error for HealNeeded {}

/// Classify the offenders in a Pass-B pylock export (structured, no prose
/// parsing). A package is SDIST-ONLY when it carries a source
/// distribution but no wheels in the resolved lock (uv had to build it
/// from source because it publishes no usable wheel); and PRERELEASE when
/// its resolved version parses as a PEP 440 pre-release
/// ([`uv_pep440::Version::any_prerelease`]). Local sources
/// (path/vcs/archive -- retread's own built wheels / editable checkouts)
/// are skipped: they are never index offenders. A package may fall in
/// BOTH lists (a wheel-less pre-release); each list is then handled by its
/// own rung.
pub fn classify_pylock_offenders(pylock_text: &str) -> Result<ClosureOffenders> {
    let doc: toml::Value =
        toml::from_str(pylock_text).context("parsing Pass-B pylock.toml for offender detection")?;
    let packages = doc
        .get("packages")
        .and_then(|p| p.as_array())
        .ok_or_else(|| anyhow!("Pass-B pylock.toml: missing [[packages]] array"))?;
    let mut sdist_only = Vec::new();
    let mut prerelease = Vec::new();
    for pkg in packages {
        let Some(name) = pkg.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        // Local sources are never index offenders.
        if pkg.get("directory").is_some()
            || pkg.get("vcs").is_some()
            || pkg.get("archive").is_some()
        {
            continue;
        }
        let Some(version) = pkg.get("version").and_then(|v| v.as_str()) else {
            continue;
        };
        let canon = canonical_conda_name(name);
        // sdist-only: has a source distribution but no wheels resolved.
        let has_sdist = pkg.get("sdist").is_some();
        let has_wheel = pkg
            .get("wheels")
            .and_then(|w| w.as_array())
            .is_some_and(|a| !a.is_empty());
        if has_sdist && !has_wheel {
            sdist_only.push((canon.clone(), version.to_string()));
        }
        // prerelease: resolved version is a PEP 440 pre-release.
        if uv_pep508::uv_pep440::Version::from_str(version)
            .map(|v| v.any_prerelease())
            .unwrap_or(false)
        {
            prerelease.push((canon, version.to_string()));
        }
    }
    Ok(ClosureOffenders {
        sdist_only,
        prerelease,
    })
}

/// A wheel AUTO-BUILT from a PyPI sdist by the sdist-only self-heal's
/// THIRD rung (ladder: wheel -> conda-route -> sdist auto-build ->
/// error), reached when a package has no wheels at all AND no conda
/// channel carries any version of it. Built through the SAME machinery
/// git-sourced `[retread-wheels]` entries use
/// ([`crate::source_build::build_wheel_from_sdist_url`]), stored
/// content-addressed in the shared wheel store
/// ([`crate::wheel::store_wheel_in_cache`]), and carrying the same
/// provenance shape a legacy gym-class sdist-built wheel gets
/// ([`crate::lock::SdistWheelSource`]) so replay never re-resolves
/// (no-resolve edict).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltSdistWheel {
    /// PEP 503-canonical PyPI name.
    pub pypi_name: String,
    /// Resolved version (from the built wheel's METADATA).
    pub version: String,
    /// Standardized wheel filename (basename of `wheel_path`).
    pub filename: String,
    /// On-disk path to the built wheel (already persisted into the
    /// shared wheel store; this is the store-relative materialized
    /// copy, hard-linked/copied wherever the caller needs it).
    pub wheel_path: PathBuf,
    /// sha256 of the built wheel's bytes (from
    /// [`crate::wheel::store_wheel_in_cache`]).
    pub sha256: String,
    /// Sdist provenance: index, name, version, and the exact resolved
    /// sdist URL (+ `#sha256` when the index advertised one).
    pub sdist_source: crate::lock::SdistWheelSource,
}

/// A transitive PRERELEASE pin the heal injected: the offending package
/// resolved to a pre-release version only once uv's prerelease policy was
/// relaxed (Pass B), so the heal re-pins it as an EXPLICIT first-party
/// `name==version` requirement (`req.explicit_pins`) to make uv's
/// `explicit` policy honor it on the next Pass A. Unlike a sdist build,
/// the package keeps its own index wheel; this record exists so the caller
/// can log/audit the repair with the same provenance conventions the
/// route/build rungs use (the resolved version also lands in the closure's
/// `pins` naturally, keeping the repair out of the user's manifest).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrereleasePin {
    /// PEP 503-canonical PyPI name.
    pub pypi_name: String,
    /// Resolved pre-release version pinned explicitly.
    pub version: String,
}

/// Guidance appended to the ORIGINAL uv error when the sdist auto-build
/// rung was attempted for one or more sdist-only names and FAILED for
/// at least one of them. Surfaces the tail of each build failure
/// (already includes the underlying `uv build --wheel` stderr snippet
/// via [`crate::source_build::build_wheel_from_sdist_url`]'s error
/// chain) so the user can act without re-running with more verbosity.
pub fn sdist_build_failed_message(failures: &[(String, String)]) -> String {
    let mut out = String::from("\nsdist auto-build failed for:\n");
    for (name, err) in failures {
        let lines: Vec<&str> = err.lines().collect();
        let tail_start = lines.len().saturating_sub(20);
        out.push_str(&format!("  {name}:\n"));
        for line in &lines[tail_start..] {
            out.push_str("    ");
            out.push_str(line);
            out.push('\n');
        }
    }
    out.push_str(
        "options: fix the sdist build (missing build backend / native \
         toolchain), allow a manual build (no-build = false), drop-dep, \
         or vendor a wheel.\n",
    );
    out
}

/// Wrap a `solve` closure with the self-heal ladder. On a
/// [`HealNeeded`] error (Pass A failed, Pass B succeeded, offenders named
/// structurally by [`classify_pylock_offenders`]), each offender is
/// repaired before the caller ever sees an error:
///
/// SDIST-ONLY offenders `(name, resolved_version)` climb the pre-existing
/// rungs, now keyed on the EXACT version Pass B resolved (`==version`)
/// rather than a regex-guessed range:
/// 1. **conda-route**: probed against the workspace conda channels at the
///    resolved version (`sdist_probe`). A hit routes the package to conda
///    (exclude from the closure at export + pin). This alone can never
///    satisfy `uv lock` (`--no-emit-package` is export-only), so it runs
///    ALONGSIDE rung 2, never instead of it.
/// 2. **sdist auto-build**: builds the package from its PyPI sdist at the
///    resolved version via `sdist_build` (config `sdist-build = "auto"`,
///    the default). Registers a `tool.uv.sources` path source so the
///    re-solve is satisfied like a real index wheel, and records a
///    [`BuiltSdistWheel`] for the caller to splice into the closure.
/// 3. **error**: build rung disabled (`sdist_build` is `None`) or the
///    build itself fails -> Pass A's original error surfaces (with the
///    build failure's log tail on a build failure). Never silently drops a
///    dependency. (Under `sdist-build = "never"`, Pass B keeps
///    `--no-build` so sdist-only offenders never appear here in the first
///    place -- this branch is defensive.)
///
/// PRERELEASE offenders `(name, resolved_version)` are re-pinned as an
/// EXPLICIT first-party `name==version` requirement (`req.explicit_pins`
/// -> the synthesized project's `dependencies`), so uv's
/// `if-necessary-or-explicit` policy honors the transitive pre-release on
/// the next Pass A. This is orthogonal to the build policy and works even
/// under `sdist-build = "never"`.
///
/// Bounded by [`AUTO_ROUTE_MAX_ROUNDS`] heal attempts; every attempt that
/// doesn't abort or succeed strictly grows the accumulated
/// routed+built+prereleased set. The routes/builds/prerelease pins
/// discovered are appended to `routed` / `built` / `prereleased` (shared
/// with the caller via `Arc<Mutex<_>>`) so a wrapping caller can splice
/// them into the final closure and log them.
pub fn with_sdist_heal<S, SP, SB>(
    bundle: String,
    solve: S,
    sdist_probe: SP,
    sdist_build: Option<SB>,
    routed: std::sync::Arc<std::sync::Mutex<Vec<AutoRoutedPackage>>>,
    built: std::sync::Arc<std::sync::Mutex<Vec<BuiltSdistWheel>>>,
    prereleased: std::sync::Arc<std::sync::Mutex<Vec<PrereleasePin>>>,
) -> impl FnMut(UvClosureRequest) -> futures::future::BoxFuture<'static, Result<UvClosure>>
where
    S: FnMut(UvClosureRequest) -> futures::future::BoxFuture<'static, Result<UvClosure>>
        + Send
        + 'static,
    SP: Fn(String, String) -> futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        + Send
        + Sync
        + 'static,
    SB: Fn(String, Option<String>) -> futures::future::BoxFuture<'static, Result<BuiltSdistWheel>>
        + Send
        + Sync
        + 'static,
{
    // `solve` is FnMut: shared across retries WITHIN one call and across
    // the outer fixpoint's repeated calls to this wrapper via a mutex
    // rather than by-value capture (moving it into the `async move` block
    // below would strand the next outer invocation with nothing to call).
    let solve = std::sync::Arc::new(std::sync::Mutex::new(solve));
    let sdist_probe = std::sync::Arc::new(sdist_probe);
    let sdist_build = std::sync::Arc::new(sdist_build);
    move |mut req: UvClosureRequest| {
        let bundle = bundle.clone();
        let solve = std::sync::Arc::clone(&solve);
        let sdist_probe = std::sync::Arc::clone(&sdist_probe);
        let sdist_build = std::sync::Arc::clone(&sdist_build);
        let routed = std::sync::Arc::clone(&routed);
        let built = std::sync::Arc::clone(&built);
        let prereleased = std::sync::Arc::clone(&prereleased);
        // Already-healed exclusions/pins/path-sources/prerelease-pins
        // accumulated across earlier outer-loop rounds must apply to THIS
        // round's request too — the outer fixpoint owns `req` and knows
        // nothing about them.
        {
            let already = routed.lock().unwrap();
            apply_auto_route(&mut req, &already);
        }
        {
            let already = built.lock().unwrap();
            for w in already.iter() {
                req.built_wheel_sources
                    .insert(w.pypi_name.clone(), w.wheel_path.clone());
                // A path source only applies to FIRST-PARTY requirements
                // (see `UvClosureRequest::explicit_pins`): without this
                // pin a wheel built for a transitive dep is invisible to
                // the resolver and the relock re-fails identically.
                req.explicit_pins
                    .insert(w.pypi_name.clone(), w.version.clone());
            }
        }
        {
            let already = prereleased.lock().unwrap();
            for p in already.iter() {
                req.explicit_pins
                    .insert(p.pypi_name.clone(), p.version.clone());
            }
        }
        let fut = (*solve.lock().unwrap())(req.clone());
        Box::pin(async move {
            let mut req = req;
            let mut attempt = fut;
            for _ in 0..AUTO_ROUTE_MAX_ROUNDS {
                match attempt.await {
                    Ok(mut closure) => {
                        attach_effective_input_requirements(&mut closure, &req)?;
                        let mut route_facts = routed.lock().unwrap();
                        hydrate_route_input_requirements(&mut route_facts, &closure);
                        return Ok(closure);
                    }
                    Err(e) => {
                        // ONLY the structured two-pass verdict drives the
                        // heal; every other error class (genuine conflict,
                        // platform-tag ceiling, Pass B also failing) passes
                        // straight through unchanged.
                        let heal = match e.downcast::<HealNeeded>() {
                            Ok(h) => h,
                            Err(other) => return Err(other),
                        };
                        // Name -> version maps (not name sets): whether a
                        // ledger entry EXHAUSTS an offender depends on the
                        // version. A ledger fact at the SAME version the
                        // solve just failed with is genuinely exhausted
                        // (rebuilding the identical wheel can't help). A
                        // ledger fact at a DIFFERENT version is STALE state
                        // -- e.g. persisted facts pin bar==1.5 while the
                        // graph now needs bar==2.5 -- and must be dropped
                        // and re-healed fresh, never used to suppress the
                        // repair (B1 wedge: the old name-set filter emptied
                        // the rebuild set, surfaced Pass A's error forever,
                        // and the failed solve never rewrote the poisoned
                        // facts file).
                        let already_built: std::collections::BTreeMap<String, String> = {
                            let b = built.lock().unwrap();
                            b.iter()
                                .map(|w| (w.pypi_name.clone(), w.version.clone()))
                                .collect()
                        };
                        let already_routed: std::collections::BTreeSet<String> = {
                            let r = routed.lock().unwrap();
                            r.iter().map(|r| r.pypi_name.clone()).collect()
                        };
                        let already_pre: std::collections::BTreeMap<String, String> = {
                            let p = prereleased.lock().unwrap();
                            p.iter()
                                .map(|p| (p.pypi_name.clone(), p.version.clone()))
                                .collect()
                        };
                        // Prerelease offenders not yet pinned at THIS version.
                        let new_pre: Vec<PrereleasePin> = heal
                            .prerelease
                            .iter()
                            .filter(|(n, v)| already_pre.get(n) != Some(v))
                            .map(|(n, v)| PrereleasePin {
                                pypi_name: n.clone(),
                                version: v.clone(),
                            })
                            .collect();
                        // Sdist-only offenders that still need a wheel at
                        // the version Pass B resolved. A name is fully
                        // exhausted only once rung 2 has actually BUILT a
                        // wheel for it AT THAT VERSION (a rung-1 route
                        // alone can never satisfy `uv lock` -- deps-from
                        // proof run 6).
                        let sdist_names: Vec<(String, String)> = heal
                            .sdist_only
                            .iter()
                            .filter(|(n, v)| already_built.get(n) != Some(v))
                            .cloned()
                            .collect();
                        if new_pre.is_empty() && sdist_names.is_empty() {
                            // Nothing new to try -> surface Pass A's error.
                            return Err(anyhow!("{}", heal.original_error));
                        }
                        // Drop stale ledger/request state for anything being
                        // re-healed at a new version, so the fresh fact
                        // replaces (never coexists with) the old one.
                        {
                            let stale_built: Vec<&String> = sdist_names
                                .iter()
                                .map(|(n, _)| n)
                                .filter(|n| already_built.contains_key(*n))
                                .collect();
                            if !stale_built.is_empty() {
                                let mut b = built.lock().unwrap();
                                b.retain(|w| !stale_built.contains(&&w.pypi_name));
                                for n in &stale_built {
                                    req.built_wheel_sources.remove(*n);
                                    req.explicit_pins.remove(*n);
                                    tracing::info!(
                                        bundle = %bundle,
                                        pkg = %n,
                                        "stale persisted built-wheel fact dropped; re-healing fresh",
                                    );
                                }
                            }
                            let stale_pre: Vec<&String> = new_pre
                                .iter()
                                .map(|p| &p.pypi_name)
                                .filter(|n| already_pre.contains_key(*n))
                                .collect();
                            if !stale_pre.is_empty() {
                                let mut p = prereleased.lock().unwrap();
                                p.retain(|e| !stale_pre.contains(&&e.pypi_name));
                                for n in &stale_pre {
                                    req.explicit_pins.remove(*n);
                                }
                            }
                        }

                        // -- sdist rungs (only when a wheel is needed) --
                        let mut new_routes = Vec::new();
                        let mut new_built = Vec::new();
                        if !sdist_names.is_empty() {
                            let Some(build) = sdist_build.as_ref() else {
                                // `sdist-build = "never"`: no wheel can be
                                // produced (should be unreachable -- Pass B
                                // keeps `--no-build` under Never so sdist-
                                // only offenders don't resolve there).
                                return Err(anyhow!("{}", heal.original_error));
                            };
                            let mut build_failures: Vec<(String, String)> = Vec::new();
                            for (name, version) in &sdist_names {
                                let spec = format!("=={version}");
                                // Rung 1: conda-route (skip already-routed).
                                if !already_routed.contains(name)
                                    && let Some(hit) = sdist_probe(name.clone(), spec.clone()).await
                                {
                                    new_routes.push(AutoRoutedPackage {
                                        pypi_name: name.clone(),
                                        conda_name: name.clone(),
                                        pypi_version: version.clone(),
                                        conda_version: hit.conda_version.clone(),
                                        channel: hit.channel.clone(),
                                        input_requirements: Vec::new(),
                                    });
                                }
                                // Rung 2: build from the sdist at the exact
                                // resolved version.
                                match build(name.clone(), Some(spec)).await {
                                    Ok(w) => new_built.push(w),
                                    Err(be) => {
                                        build_failures.push((name.clone(), format!("{be:#}")))
                                    }
                                }
                            }
                            // Rung 3: any build failure aborts loudly (a
                            // partial success still fails rather than
                            // silently dropping the failed name).
                            if !build_failures.is_empty() {
                                bail!(
                                    "{}{}",
                                    heal.original_error,
                                    sdist_build_failed_message(&build_failures)
                                );
                            }
                        }

                        // -- apply discoveries to the request + ledgers --
                        for h in &new_routes {
                            tracing::info!(
                                bundle = %bundle,
                                "sdist-only {} routed to conda (channel {})",
                                h.pypi_name,
                                h.channel,
                            );
                        }
                        apply_auto_route(&mut req, &new_routes);
                        routed.lock().unwrap().extend(new_routes.clone());
                        for w in &new_built {
                            req.built_wheel_sources
                                .insert(w.pypi_name.clone(), w.wheel_path.clone());
                            // First-party pin so the path source applies to
                            // a TRANSITIVE dependency (see the reapply
                            // block above / `explicit_pins` docs).
                            req.explicit_pins
                                .insert(w.pypi_name.clone(), w.version.clone());
                        }
                        built.lock().unwrap().extend(new_built.clone());
                        for p in &new_pre {
                            tracing::info!(
                                bundle = %bundle,
                                "transitive prerelease {}=={} pinned as an explicit \
                                 first-party requirement so uv honors it",
                                p.pypi_name,
                                p.version,
                            );
                            req.explicit_pins
                                .insert(p.pypi_name.clone(), p.version.clone());
                        }
                        prereleased.lock().unwrap().extend(new_pre.clone());

                        attempt = (*solve.lock().unwrap())(req.clone());
                    }
                }
            }
            bail!(
                "self-heal exceeded {} rounds for bundle `{}`",
                AUTO_ROUTE_MAX_ROUNDS,
                bundle,
            );
        })
    }
}

/// Compatibility entry point: [`auto_route_fixpoint`] with the
/// sdist-only self-heal ladder layered underneath (see
/// [`with_sdist_heal`]). `sdist_probe` answers "does ANY workspace
/// conda channel carry `conda_name` at a version matching the given
/// spec?" — production wires [`crate::probe::find_route`] with the
/// spec [`with_sdist_heal`] derives from the originating pypi
/// requirement (falling back to `"*"`, any version, when none can be
/// extracted). `sdist_build` is `Some`
/// when the pack's `sdist-build` config is `"auto"` (default); `None`
/// disables the build rung (`"never"`), reproducing the original
/// conda-route-or-error behavior exactly.
///
/// The routes/built wheels the heal discovers are appended to the
/// returned closure's `auto_routed` list and `wheels` (as `Origin::Built`
/// entries carrying `sdist_source` provenance) respectively — same
/// shape/logging convention as ordinary auto-route hits — even though
/// the outer fixpoint never saw them directly.
pub async fn auto_route_fixpoint_with_sdist_heal<S, P, SP, SB>(
    req: &UvClosureRequest,
    opts: &AutoRouteOptions,
    solve: S,
    probe: P,
    sdist_probe: SP,
    sdist_build: Option<SB>,
) -> Result<UvClosure>
where
    S: FnMut(UvClosureRequest) -> futures::future::BoxFuture<'static, Result<UvClosure>>
        + Send
        + 'static,
    P: Fn(String, String) -> futures::future::BoxFuture<'static, Option<RouteProbeHit>>,
    SP: Fn(String, String) -> futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        + Send
        + Sync
        + 'static,
    SB: Fn(String, Option<String>) -> futures::future::BoxFuture<'static, Result<BuiltSdistWheel>>
        + Send
        + Sync
        + 'static,
{
    let sdist_routed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sdist_built = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    // Prerelease pins the heal injects surface naturally in the closure's
    // `pins`/`wheels` (the offending package keeps its own index wheel); we
    // collect them here only to log/audit the repair.
    let sdist_prereleased = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let healed_solve = with_sdist_heal(
        req.bundle.clone(),
        solve,
        sdist_probe,
        sdist_build,
        std::sync::Arc::clone(&sdist_routed),
        std::sync::Arc::clone(&sdist_built),
        std::sync::Arc::clone(&sdist_prereleased),
    );
    let mut closure = auto_route_fixpoint(req, opts, healed_solve, probe).await?;
    let extra = sdist_routed.lock().unwrap();
    closure.auto_routed.extend(extra.iter().cloned());
    drop(extra);
    let built = sdist_built.lock().unwrap();
    for w in built.iter() {
        closure
            .pins
            .entry(w.pypi_name.clone())
            .or_insert_with(|| w.version.clone());
        closure.wheels.push(LockWheel {
            name: w.pypi_name.clone(),
            version: w.version.clone(),
            origin: Origin::Built,
            filename: w.filename.clone(),
            url: None,
            sha256: Some(w.sha256.clone()),
            requires_dist: Vec::new(),
            // No fetchable index URL: the wheel exists only in the
            // shared wheel store, exactly like a git-source-built wheel
            // (see `crate::lock::LockWheel::must_ship` docs).
            must_ship: true,
            upstream_url: None,
            git_source: None,
            sdist_source: Some(w.sdist_source.clone()),
        });
    }
    Ok(closure)
}

// ---------------------------------------------------------------------------
// Constraint generation (conda pins -> PEP 440), with provenance
// ---------------------------------------------------------------------------

/// Translate a conda version spec into a PEP 440 specifier where
/// representable. Returns `None` for specs that must be skipped (same
/// spirit as `installer::conda_deps_to_constraints` skip rules): `*`,
/// build-string / space-bearing specs, `|` alternations.
pub fn conda_spec_to_pep440(spec: &str) -> Option<String> {
    let s = spec.trim();
    if s.is_empty() || s == "*" || s == "==*" {
        return None;
    }
    // Build strings ("2.1.0 py312_0"), alternations ("1.2|1.3") and
    // anything with characters outside a conservative PEP 440 alphabet
    // are conda-only: skip.
    if s.contains(' ') || s.contains('|') {
        return None;
    }
    let ok = |c: char| c.is_ascii_alphanumeric() || ".*,<>=!~+-".contains(c);
    if !s.chars().all(ok) {
        return None;
    }
    // Operator-prefixed conda specs are PEP 440-compatible as-is
    // (">=1.2,<2", "==1.2.3", "~=1.2", "!=1.3").
    if s.starts_with("==")
        || s.starts_with(">=")
        || s.starts_with("<=")
        || s.starts_with("!=")
        || s.starts_with("~=")
        || s.starts_with('>')
        || s.starts_with('<')
    {
        return Some(s.to_string());
    }
    // conda `=1.2` means 1.2.* (fuzzy).
    if let Some(rest) = s.strip_prefix('=') {
        if rest.is_empty() {
            return None;
        }
        return Some(if rest.ends_with('*') {
            format!("=={rest}")
        } else {
            format!("=={rest}.*")
        });
    }
    // Bare versions: conda treats "1.2" as fuzzy (startswith) and
    // "1.2.*" explicitly so. Both map to `==X.Y.*`.
    if s.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return Some(if s.ends_with('*') {
            format!("=={s}")
        } else {
            format!("=={s}.*")
        });
    }
    None
}

/// Conda names never emitted as PyPI constraints (conda-only surface).
fn is_conda_only_name(name: &str) -> bool {
    name.is_empty() || name == "python" || name == "python_abi" || name.starts_with("__")
}

/// Build the `constraint-dependencies` set from conda pins.
///
/// * `conda_deps`: conda package name -> conda version spec (from the
///   workspace manifest's effective deps or from a `pixi.lock` read).
/// * `name_map`: the *effective* pypi -> conda name map (user
///   `retread-name-map` + fallback table + parselmouth merge). It is
///   inverted here to recover the PyPI name for each conda pin; conda
///   names without a mapping use their PEP 503-canonical form as the
///   PyPI name (identity mapping).
/// * `source` / `env`: recorded verbatim into provenance.
pub fn build_constraints(
    conda_deps: &BTreeMap<String, String>,
    name_map: &BTreeMap<String, String>,
    global_name_map: &crate::handler::PypiToCondaMap,
    source: &str,
    env: &str,
) -> ConstraintSet {
    // Invert pypi->conda. BTreeMap iteration is ordered, so on conda-name
    // collisions the alphabetically-first PyPI name wins deterministically.
    // The pack's own `retread-name-map` is inverted FIRST (its edges win);
    // the parselmouth-backed global map (run-38 fix) fills the rest so a
    // hand-written accelerator meta-package pin (`pytorch-gpu ==2.7.0`)
    // maps to the pypi name uv actually resolves (`torch`) even when the
    // pack declares no name map of its own -- previously such a pin
    // produced an inert constraint on a nonexistent pypi name
    // (`pytorch-gpu==2.7.0`) and uv free-picked torch 2.10.
    let mut conda_to_pypi: BTreeMap<String, String> = BTreeMap::new();
    for (pypi, conda) in name_map {
        conda_to_pypi
            .entry(canonical_conda_name(conda))
            .or_insert_with(|| canonical_conda_name(pypi));
    }
    // Deterministic order: sort the HashMap's pypi keys before merging.
    let mut global_sorted: Vec<(&String, &Vec<String>)> = global_name_map.iter().collect();
    global_sorted.sort_by_key(|(pypi, _)| (*pypi).clone());
    for (pypi, condas) in global_sorted {
        for conda in condas {
            conda_to_pypi
                .entry(canonical_conda_name(conda))
                .or_insert_with(|| canonical_conda_name(pypi));
        }
    }

    let mut set = ConstraintSet::default();
    for (conda_name, conda_spec) in conda_deps {
        if is_conda_only_name(conda_name) {
            continue;
        }
        let Some(pep) = conda_spec_to_pep440(conda_spec) else {
            continue;
        };
        let canon = canonical_conda_name(conda_name);
        let pypi_name = conda_to_pypi.get(&canon).cloned().unwrap_or(canon);
        let line = format!("{pypi_name}{pep}");
        set.constraints.push(line.clone());
        set.provenance.insert(
            pypi_name,
            ConstraintProvenance {
                constraint: line,
                conda_name: conda_name.clone(),
                conda_version: conda_spec.clone(),
                source: source.to_string(),
                env: env.to_string(),
                advisory: false,
            },
        );
    }
    set
}

/// PyPI families whose published release line tracks the CUDA toolkit
/// MAJOR version 1:1, the same way conda-forge's `cuda-version`
/// metapackage anchors the conda side (`cuda-bindings` releases 12.x
/// against CUDA 12, 13.x against CUDA 13, with no narrower
/// compatibility range published on either side). When a workspace's
/// consuming env(s) pin `cuda-version` to some major X, uv must be
/// capped to the same line up front — otherwise its resolver is free
/// to pick e.g. `cuda-bindings==13.3.1` purely from PyPI metadata,
/// which the auto-route co-install check then has to catch AFTER the
/// fact and un-route (the incident this table exists to pre-empt:
/// `cuda-bindings ==13.3.1`'s conda variant requires `cuda-version
/// >=13,<14` against a workspace pinned to `cuda-version 12.9`).
///
/// Deliberately NOT included: `nvidia-*-cuXX` wheels (e.g.
/// `nvidia-cublas-cu12`) and `cupy-cuda11x`/`cupy-cuda12x` — those
/// already encode the CUDA major IN THE PACKAGE NAME, so "capping a
/// version range" doesn't apply; picking the wrong one is a
/// name-selection problem handled by the existing pypi<->conda name
/// map, not a version constraint.
pub const CUDA_MAJOR_TRACKED_PYPI_FAMILIES: &[&str] = &["cuda-bindings", "cuda-python"];

/// Extract a CUDA major version (e.g. `12`) from the first parseable
/// `cuda-version` conda spec string (`"==12.9"`, `">=12.8,<13"`,
/// `"12"`, ...). Only the leading digit run is read — sufficient to
/// recover the major line these specs always anchor on. `None` when no
/// spec yields a leading integer.
pub fn cuda_major_from_specs(specs: &[String]) -> Option<u32> {
    specs.iter().find_map(|spec| {
        let s = spec.trim();
        let start = s.find(|c: char| c.is_ascii_digit())?;
        s[start..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .ok()
    })
}

/// Constraint lines capping [`CUDA_MAJOR_TRACKED_PYPI_FAMILIES`] to the
/// given CUDA major (`>=X,<X+1`). Harmless to add unconditionally: a
/// uv constraints file entry for a package absent from the resolved
/// graph is simply never applied (constraints never pull in a package
/// on their own), so these are only load-bearing for bundles that
/// actually depend on one of the tracked families.
pub fn cuda_family_constraints(cuda_major: u32) -> Vec<(&'static str, String)> {
    CUDA_MAJOR_TRACKED_PYPI_FAMILIES
        .iter()
        .map(|name| (*name, format!(">={cuda_major},<{}", cuda_major + 1)))
        .collect()
}

// ---------------------------------------------------------------------------
// Ephemeral project synthesis
// ---------------------------------------------------------------------------

/// `(lower, upper)` bounds spanning the target python minor: `"3.12"`
/// (or `"3.12.4"`) -> `("3.12", "3.13")`. `None` when the string does
/// not start with a parseable `major.minor`.
pub fn python_minor_bounds(python_version: &str) -> Option<(String, String)> {
    let mut parts = python_version.split('.');
    let major: u64 = parts.next()?.trim().parse().ok()?;
    let minor_digits: String = parts
        .next()?
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let minor: u64 = minor_digits.parse().ok()?;
    Some((format!("{major}.{minor}"), format!("{major}.{}", minor + 1)))
}

/// `tool.uv.environments` marker restricting universal resolution to the
/// single (python minor, platform) environment retread materializes for.
///
/// The python clause prunes marker-dead branches (e.g. aiohttp's
/// `idna-ssl ; python_version < '3.7'`) from the resolve so they never
/// need per-pack `retread-drop-deps`; the platform clause does the same
/// for foreign platforms. `None` only when BOTH halves are unknown
/// (unparseable python AND noarch/unknown subdir); tag selection in
/// [`parse_pylock_closure`] still enforces the platform either way.
pub fn environment_marker(python_version: &str, conda_subdir: &str) -> Option<String> {
    let python_clause = python_minor_bounds(python_version)
        .map(|(lo, hi)| format!("python_version >= '{lo}' and python_version < '{hi}'"));
    let platform_clause = match conda_subdir {
        "linux-64" => Some(("linux", "x86_64")),
        "linux-aarch64" => Some(("linux", "aarch64")),
        "linux-ppc64le" => Some(("linux", "ppc64le")),
        "osx-64" => Some(("darwin", "x86_64")),
        "osx-arm64" => Some(("darwin", "arm64")),
        "win-64" => Some(("win32", "AMD64")),
        _ => None,
    }
    .map(|(platform, machine)| {
        format!("sys_platform == '{platform}' and platform_machine == '{machine}'")
    });
    match (python_clause, platform_clause) {
        (Some(py), Some(plat)) => Some(format!("{py} and {plat}")),
        (Some(py), None) => Some(py),
        (None, Some(plat)) => Some(plat),
        (None, None) => None,
    }
}

/// Decide which built-in Windows-only shim names
/// ([`crate::config::BUILT_IN_WIN_ONLY`]) to inject into a target's
/// override-dependencies as unmatchable-marker drops.
///
/// NVIDIA's index strips the `sys_platform == "win32"` marker from these
/// Requires-Dist lines, so [`environment_marker`] can't prune them; injecting
/// the same `name ; python_version < '0'` override the `retread-drop-deps`
/// path uses removes them from uv's resolution graph without packs having to
/// enumerate them. Semantics mirror the conda run-dep auto-drop:
///
/// * Windows targets (`win-64`, `win-32`, `win-arm64`) inject NONE — the
///   shims are legitimate there.
/// * A name the user explicitly overrode (`retread-overrides`) is skipped so
///   the override wins (re-enable on Linux).
/// * A name already in the pack's `retread-drop-deps` is skipped so there is
///   no duplicate override line.
/// * A name the user declared as an explicit TOP-LEVEL (first-party) root
///   requirement is skipped -- explicit user intent at the top level means
///   "resolve this deliberately", and a graph-wide uv override marker would
///   otherwise suppress it. Normal resolution (incl. sdist heal) handles it.
///
/// Returns the built-in names to inject, in `BUILT_IN_WIN_ONLY` order.
pub fn built_in_win_only_to_inject<'a>(
    conda_subdir: &str,
    is_overridden: impl Fn(&str) -> bool,
    pack_drop_deps: &[String],
    first_party: &[String],
) -> Vec<&'a str> {
    if matches!(conda_subdir, "win-64" | "win-32" | "win-arm64") {
        return Vec::new();
    }
    let already_dropped: BTreeSet<String> = pack_drop_deps
        .iter()
        .map(|n| canonical_conda_name(n))
        .collect();
    let first_party_names: BTreeSet<String> = first_party
        .iter()
        .map(|n| canonical_conda_name(n))
        .collect();
    crate::config::BUILT_IN_WIN_ONLY
        .iter()
        .copied()
        .filter(|name| !is_overridden(name))
        .filter(|name| !already_dropped.contains(&canonical_conda_name(name)))
        .filter(|name| !first_party_names.contains(&canonical_conda_name(name)))
        .collect()
}

/// Escape a string for a TOML basic (double-quoted) string.
fn toml_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04X}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn toml_string_array(indent: &str, items: &[String]) -> String {
    if items.is_empty() {
        return "[]".to_string();
    }
    let mut out = String::from("[\n");
    for item in items {
        out.push_str(indent);
        out.push_str("    ");
        out.push_str(&toml_str(item));
        out.push_str(",\n");
    }
    out.push_str(indent);
    out.push(']');
    out
}

/// PEP 503-ish project-name sanitization for the synthetic project.
fn project_name(bundle: &str) -> String {
    let canon = canonical_conda_name(bundle);
    format!("retread-closure-{canon}")
}

/// Render the ephemeral project's `pyproject.toml` (spec §2.1). Pure and
/// deterministic — golden-snapshot tested.
pub fn synthesize_pyproject(req: &UvClosureRequest) -> String {
    let mut out = String::new();
    out.push_str("# Generated by pixi-build-retread (retread-resolver = \"uv\"). Do not edit.\n");
    out.push_str("[project]\n");
    out.push_str(&format!(
        "name = {}\n",
        toml_str(&project_name(&req.bundle))
    ));
    out.push_str("version = \"0\"\n");
    // Bounded range (not `==X.Y.*`): the explicit upper AND lower bound
    // restricts uv's universal resolution to the one python minor retread
    // materializes for, so marker-dead branches on other pythons are never
    // resolved. Falls back to the exact-match form when the version string
    // has no parseable minor.
    let requires_python = match python_minor_bounds(&req.python_version) {
        Some((lo, hi)) => format!(">={lo},<{hi}"),
        None => format!("=={}.*", req.python_version),
    };
    out.push_str(&format!(
        "requires-python = {}\n",
        toml_str(&requires_python)
    ));
    // Self-heal repairs are appended to the project's DIRECT dependencies
    // as explicit `name==version` pins -- transitive prerelease pins (uv's
    // `explicit` prerelease policy is first-party-only) AND heal-built
    // sdist wheels (a `[tool.uv.sources]` path entry only applies to
    // first-party requirements; see `UvClosureRequest::explicit_pins`).
    // Ephemeral -- never touches the user's manifest.
    let mut deps = req.dependencies.clone();
    for (name, version) in &req.explicit_pins {
        deps.push(format!("{}=={}", canonical_conda_name(name), version));
    }
    out.push_str(&format!(
        "dependencies = {}\n",
        toml_string_array("", &deps)
    ));

    out.push_str("\n[tool.uv]\n");
    if let Some(marker) = environment_marker(&req.python_version, &req.conda_subdir) {
        out.push_str(&format!("environments = [{}]\n", toml_str(&marker)));
    }
    // sdists are never built by uv (spec §8.2): retread's source_build
    // path owns builds. An sdist-only transitive fails the lock loudly.
    out.push_str("no-build = true\n");
    // Matches the installer's index semantics (installer.rs build_uv_args).
    out.push_str("index-strategy = \"unsafe-best-match\"\n");
    // Packages that publish ONLY pre-releases (e.g. tinyobjloader
    // 2.0.0rc13) resolve without a per-pack override; stable releases
    // still win everywhere else. Config-file-class key: also passed as
    // a CLI flag on `uv lock` (see compute_closure) because UV_NO_CONFIG
    // strips it from this table on uv 0.11.x.
    out.push_str("prerelease = \"if-necessary-or-explicit\"\n");
    if !req.constraints.constraints.is_empty() {
        out.push_str(&format!(
            "constraint-dependencies = {}\n",
            toml_string_array("", &req.constraints.constraints)
        ));
    }
    // User overrides first, then drop-dep unmatchable markers (A3).
    if !req.overrides.is_empty() {
        out.push_str(&format!(
            "override-dependencies = {}\n",
            toml_string_array("", &req.overrides)
        ));
    }

    for url in &req.index_urls {
        out.push_str("\n[[tool.uv.index]]\n");
        out.push_str(&format!("url = {}\n", toml_str(url)));
    }

    if !req.built_wheel_sources.is_empty() {
        out.push_str("\n[tool.uv.sources]\n");
        for (name, path) in &req.built_wheel_sources {
            out.push_str(&format!(
                "{} = {{ path = {} }}\n",
                canonical_conda_name(name),
                toml_str(&path.to_string_lossy())
            ));
        }
    }
    out
}

/// Serialize the provenance table to the JSON shape of spec §2.2
/// (`constraints.provenance.json`).
pub fn provenance_json(set: &ConstraintSet) -> Result<String> {
    serde_json::to_string_pretty(&set.provenance).context("serializing constraint provenance")
}

// ---------------------------------------------------------------------------
// pylock.toml (PEP 751) parsing -> lock shapes
// ---------------------------------------------------------------------------

/// Parse a PEP 751 `pylock.toml` into lock-shaped wheels + pins.
///
/// * One wheel is selected per package by tag priority
///   (`crate::pypi::score_wheel`) for `target`.
/// * `exclude`: PEP 503-canonical names filtered from the closure. This
///   post-parse filter is the *authoritative* routing mechanism; the
///   `--no-emit-package` export flags are an optimization (AMENDMENT A1).
/// * Packages sourced from a local directory / vcs / archive (retread's
///   own built wheels via `tool.uv.sources`) contribute a pin but no
///   index wheel — retread merges its built wheels separately.
/// * Index wheels missing a sha256 are a hard error (spec §8.4); an
///   index package with no tag-compatible wheel (e.g. sdist-only under
///   `no-build`) is a hard error naming the package.
pub fn parse_pylock_closure(
    text: &str,
    target: &WheelTarget,
    exclude: &BTreeSet<String>,
    uv_version: &str,
) -> Result<UvClosure> {
    let doc: toml::Value = toml::from_str(text).context("parsing pylock.toml")?;
    let packages = doc
        .get("packages")
        .and_then(|p| p.as_array())
        .ok_or_else(|| anyhow!("pylock.toml: missing [[packages]] array"))?;

    let mut wheels: Vec<LockWheel> = Vec::with_capacity(packages.len());
    let mut pins: BTreeMap<String, String> = BTreeMap::new();

    for pkg in packages {
        let name = pkg
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("pylock.toml: package missing `name`"))?;
        let canon = canonical_conda_name(name);
        if exclude.contains(&canon) {
            continue;
        }
        let version = pkg.get("version").and_then(|v| v.as_str()).unwrap_or("");

        // Local sources (our own built wheels / editable checkouts):
        // pin only, no index wheel.
        let is_local = pkg.get("directory").is_some()
            || pkg.get("vcs").is_some()
            || pkg.get("archive").is_some();
        if is_local {
            if !version.is_empty() {
                pins.insert(canon, version.to_string());
            }
            continue;
        }

        if version.is_empty() {
            bail!("pylock.toml: index package `{name}` missing `version`");
        }

        let wheel_entries = pkg
            .get("wheels")
            .and_then(|w| w.as_array())
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        if wheel_entries.is_empty() {
            bail!(
                "package `{name}=={version}` has no wheels in the exported closure \
                 (sdist-only under `no-build = true`?). Route it to conda via \
                 `retread-conda-deps`, add a git/path source entry, or drop it."
            );
        }

        // Select ONE wheel by tag priority for (python, platform).
        let mut best: Option<(i64, &toml::Value, String)> = None;
        for w in wheel_entries {
            let filename = w
                .get("name")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .or_else(|| {
                    w.get("url")
                        .and_then(|v| v.as_str())
                        .and_then(|u| u.rsplit('/').next().map(str::to_string))
                });
            let Some(filename) = filename else { continue };
            let score = crate::pypi::score_wheel(&filename, target);
            if score >= 0 && best.as_ref().is_none_or(|(s, _, _)| score > *s) {
                best = Some((score, w, filename));
            }
        }
        let Some((_, wheel, filename)) = best else {
            let glibc_hint = match target.max_glibc {
                Some((maj, min)) => format!(
                    " manylinux ceiling in effect: glibc {maj}.{min} (max of \
                     declared system-requirements libc and host glibc). If the \
                     runtime env provides a newer glibc, declare it: \
                     `[system-requirements] libc = \"X.Y\"` (pixi < 0.71) or the \
                     rich `platforms = [{{ platform = ..., glibc = \"X.Y\" }}]` \
                     form (pixi >= 0.71)."
                ),
                None => String::new(),
            };
            bail!(
                "package `{name}=={version}`: none of its {} wheel(s) is compatible \
                 with python {} on {} (tag selection). If only an sdist fits, route \
                 it via `retread-conda-deps` or a source entry.{glibc_hint}",
                wheel_entries.len(),
                target.python_version,
                target.conda_subdir,
            );
        };
        let url = wheel
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("pylock.toml: wheel `{filename}` missing `url`"))?;
        let sha256 = wheel
            .get("hashes")
            .and_then(|h| h.get("sha256"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                anyhow!(
                    "pylock.toml: wheel `{filename}` has no sha256 hash; refusing to \
                     ship an unhashed index wheel"
                )
            })?;

        pins.insert(canon.clone(), version.to_string());
        wheels.push(LockWheel {
            name: canon,
            version: version.to_string(),
            origin: Origin::Index,
            filename,
            url: Some(url.to_string()),
            sha256: Some(sha256.to_string()),
            requires_dist: Vec::new(),
            must_ship: false,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        });
    }

    Ok(UvClosure {
        wheels,
        pins,
        uv_version: uv_version.to_string(),
        auto_routed: Vec::new(),
        auto_dropped: BTreeSet::new(),
        effective_input_requirements: None,
    })
}

// ---------------------------------------------------------------------------
// Conflict attribution (uv stderr x constraint provenance)
// ---------------------------------------------------------------------------

/// One attributed conflict: a constrained package named in uv's error
/// text, joined back to its conda source via the provenance table.
#[derive(Debug, Clone, Serialize)]
pub struct ConflictAttribution {
    /// PyPI package name (provenance key).
    pub package: String,
    /// Requirement range uv reported for the package, when parseable
    /// from the message (`None` otherwise — never block on parse
    /// quality, spec §4a).
    pub required: Option<String>,
    /// The conda-derived constraint the requirement collided with.
    pub conflicting_constraint: String,
    /// Provenance of that constraint.
    pub conda_source: ConstraintProvenance,
}

/// One graph-wide uv override learned from a precise workspace-solved conda
/// fact. Two shapes: a Rule-3 version override (an upstream exact pin proven to
/// contradict the fact is replaced with the conda-selected version), or a
/// Rule-1 drop (the workspace conda-OWNS the dep and its PyPI requirement is
/// unsatisfiable, so the wheel-side requirement is removed graph-wide and the
/// conda owner provides it at runtime).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceFactOverride {
    /// Canonical PyPI name uv resolves.
    pub pypi_name: String,
    /// PEP 440 version selected by every precise consuming environment's
    /// conda solve. For a `drop`, this is retained only as the owning conda
    /// version for provenance/logging -- the emitted requirement carries no
    /// version.
    pub version: String,
    /// When true this is a Rule-1 DROP, not a Rule-3 version pin: the workspace
    /// conda-owns `pypi_name` (through a retread-name-map edge, possibly the
    /// identity edge) and its PyPI requirement could not be satisfied, so the
    /// requirement is removed from uv's graph entirely (unmatchable marker) and
    /// the conda package owns the import at runtime. Defaulted for back-compat
    /// with heal-fact files written before drops existed.
    #[serde(default)]
    pub drop: bool,
}

impl WorkspaceFactOverride {
    fn requirement(&self) -> String {
        if self.drop {
            // Graph-wide unmatchable-marker override: same mechanism as
            // `retread-drop-deps` and the workspace-owned `[pypi-dependencies]`
            // drop, so the removal covers a transitive `Requires-Dist` exact
            // pin, not just a direct root.
            format!("{} ; {DROP_MARKER}", self.pypi_name)
        } else {
            format!("{}=={}", self.pypi_name, self.version)
        }
    }
}

/// Structured signal from [`compute_closure`] to
/// [`with_workspace_fact_overrides`]. Ordinary resolution errors remain
/// ordinary `anyhow` errors and never enter this recovery path.
#[derive(Debug, Clone)]
struct WorkspaceFactOverrideNeeded {
    fact: WorkspaceFactOverride,
    upstream_pin: String,
    original_error: String,
}

impl std::fmt::Display for WorkspaceFactOverrideNeeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.original_error)
    }
}

impl std::error::Error for WorkspaceFactOverrideNeeded {}

/// Parse one atomic, non-wildcard PEP 440 `==`/`===` pin. Returning the full
/// specifier set lets callers use PEP 440 containment (rather than textual
/// version inequality) for local versions, epochs, and normalized spellings.
fn exact_pep440_pin(
    raw: &str,
) -> Option<(
    uv_pep508::uv_pep440::VersionSpecifiers,
    uv_pep508::uv_pep440::Version,
)> {
    use uv_pep508::uv_pep440::{Operator, Version, VersionSpecifiers};

    let specs = VersionSpecifiers::from_str(raw.trim()).ok()?;
    let one: Vec<_> = specs.iter().collect();
    if one.len() != 1 || !matches!(one[0].operator(), Operator::Equal | Operator::ExactEqual) {
        return None;
    }
    let raw_version = raw
        .trim()
        .strip_prefix("===")
        .or_else(|| raw.trim().strip_prefix("=="))?
        .trim();
    if raw_version.is_empty() || raw_version.contains('*') {
        return None;
    }
    let version = Version::from_str(raw_version).ok()?;
    Some((specs, version))
}

fn exact_requirement_pin(
    raw: &str,
) -> Option<(
    String,
    uv_pep508::uv_pep440::VersionSpecifiers,
    uv_pep508::uv_pep440::Version,
)> {
    let req: Requirement = Requirement::from_str(raw).ok()?;
    let uv_pep508::VersionOrUrl::VersionSpecifier(specs) = req.version_or_url.as_ref()? else {
        return None;
    };
    let rendered = specs.to_string();
    let (exact, version) = exact_pep440_pin(&rendered)?;
    Some((canonical_conda_name(req.name.as_ref()), exact, version))
}

fn request_has_direct_root(req: &UvClosureRequest, name: &str) -> bool {
    req.dependencies.iter().any(|raw| {
        let root: Result<Requirement, _> = Requirement::from_str(raw);
        root.is_ok_and(|root| {
            canonical_conda_name(root.name.as_ref()) == canonical_conda_name(name)
        })
    })
}

fn override_name(raw: &str) -> Option<String> {
    let req: Requirement = Requirement::from_str(raw).ok()?;
    Some(canonical_conda_name(req.name.as_ref()))
}

/// Select the first fail-closed Rule-3 repair from attributed uv prose.
/// Attribution is already joined to constraint provenance; this layer arms
/// only for precise Rule-1 facts and only when the opposing upstream pin is
/// provably exact and excludes the conda-selected version.
fn workspace_fact_override_needed(
    req: &UvClosureRequest,
    attributions: &[ConflictAttribution],
    original_error: &str,
) -> Option<WorkspaceFactOverrideNeeded> {
    for attribution in attributions {
        if attribution.conda_source.source != "workspace-solved"
            || request_has_direct_root(req, &attribution.package)
        {
            continue;
        }
        let Some(required) = attribution.required.as_deref() else {
            continue;
        };
        let Some((upstream_specs, _)) = exact_pep440_pin(required) else {
            continue;
        };
        let Some((constraint_name, fact_specs, fact_version)) =
            exact_requirement_pin(&attribution.conflicting_constraint)
        else {
            continue;
        };
        if constraint_name != canonical_conda_name(&attribution.package) {
            continue;
        }
        let Some((conda_specs, conda_version)) =
            exact_pep440_pin(&attribution.conda_source.conda_version)
        else {
            continue;
        };
        // Provenance must agree with the emitted pypi constraint. If either
        // exact spec rejects the other's normalized version, the record is
        // inconsistent and cannot authorize an automatic graph rewrite.
        if !fact_specs.contains(&conda_version) || !conda_specs.contains(&fact_version) {
            continue;
        }
        // Equal/equivalent exact pins are not contradictions. This also keeps
        // `==2.10.0` compatible with a local `2.10.0+cu129` fact per PEP 440.
        if upstream_specs.contains(&fact_version) {
            continue;
        }
        return Some(WorkspaceFactOverrideNeeded {
            fact: WorkspaceFactOverride {
                pypi_name: constraint_name,
                version: fact_version.to_string(),
                drop: false,
            },
            upstream_pin: required.to_string(),
            original_error: original_error.to_string(),
        });
    }
    None
}

/// Select the first Rule-1 DROP the failed lock authorizes: a transitive
/// dependency the workspace conda-OWNS (through a precise workspace-solved
/// fact, i.e. a retread-name-map edge whose conda target every consuming
/// environment declares) whose PyPI requirement neither Pass A nor Pass B
/// could satisfy. The conda owner provides the import at runtime, so the
/// wheel-side requirement is removed graph-wide rather than erroring.
///
/// This is the Rule-1 complement of [`workspace_fact_override_needed`]: Rule 3
/// fires when an upstream EXACT pin CONTRADICTS the fact (replace it with the
/// conda version, which exists on PyPI); this fires when the conda version
/// SATISFIES the upstream requirement but PyPI cannot supply it (e.g. conda
/// `py-opencv==4.11.0` maps to PyPI `opencv-python`, whose versions are
/// `4.11.0.86`-style -- `opencv-python==4.11.0` simply does not exist). It is
/// invoked only after Rule 3 declines and after Pass B also fails, so an
/// ordinary conflict, an sdist-only heal, or a version override always wins
/// first.
///
/// Guards preserved: only a `workspace-solved` provenance (never a routing
/// alias) is ownership authority; a first-party direct root is user intent and
/// is never dropped; and a DIFFERING upstream requirement must actually be
/// SATISFIED by the conda version, or this is a genuine version conflict that
/// keeps erroring rather than a conda-owned drop.
fn workspace_owned_drop_needed(
    req: &UvClosureRequest,
    attributions: &[ConflictAttribution],
    original_error: &str,
) -> Option<WorkspaceFactOverrideNeeded> {
    for attribution in attributions {
        // Only a precise workspace-solved conda fact is ownership authority.
        if attribution.conda_source.source != "workspace-solved" {
            continue;
        }
        // A first-party direct root is the user's explicit intent, not an
        // upstream requirement -- never silently drop it.
        if request_has_direct_root(req, &attribution.package) {
            continue;
        }
        // The owning conda version (an exact `==` from the workspace solve).
        let Some((_, conda_version)) = exact_pep440_pin(&attribution.conda_source.conda_version)
        else {
            continue;
        };
        // Relying on the conda package at runtime is only correct when the
        // conda-selected version SATISFIES what the wheel required. `required
        // == None` means uv's upstream pin equalled our conda-derived
        // constraint (attribute_conflict elides the echo of our own line), so
        // the conda version trivially satisfies it -- exactly the
        // `opencv-python==4.11.0` shape. A differing range/pin must actually
        // contain the conda version.
        if let Some(required) = attribution.required.as_deref() {
            let Ok(required_specs) =
                uv_pep508::uv_pep440::VersionSpecifiers::from_str(required)
            else {
                continue;
            };
            if !required_specs.contains(&conda_version) {
                continue;
            }
        }
        return Some(WorkspaceFactOverrideNeeded {
            fact: WorkspaceFactOverride {
                pypi_name: canonical_conda_name(&attribution.package),
                version: conda_version.to_string(),
                drop: true,
            },
            upstream_pin: attribution
                .required
                .clone()
                .unwrap_or_else(|| attribution.conflicting_constraint.clone()),
            original_error: original_error.to_string(),
        });
    }
    None
}

/// Best-effort join of uv's conflict prose to the constraint provenance
/// table: any constrained name appearing in the error text is attributed
/// to its conda source package. Degrades gracefully — an unparseable
/// message still yields records for every constrained name it mentions.
pub fn attribute_conflict(
    stderr: &str,
    provenance: &BTreeMap<String, ConstraintProvenance>,
) -> Vec<ConflictAttribution> {
    let mut out = Vec::new();
    for (pypi_name, prov) in provenance {
        if prov.advisory {
            continue;
        }
        // Word-boundary match on the normalized name.
        let re = regex::Regex::new(&format!(
            r"(?i)\b{}(?:\[[^\]]*\])?((?:===|==|>=|<=|~=|!=|>|<)[0-9][^\s,)`']*)?",
            regex::escape(pypi_name)
        ))
        .expect("static conflict regex");
        let mut mentioned = false;
        let mut required: Option<String> = None;
        for cap in re.captures_iter(stderr) {
            let whole = cap.get(0).expect("a regex capture always has group 0");
            if stderr[whole.end()..]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
            {
                // Project-name end boundary: `torch` must not attribute a
                // `torchvision` conflict.
                continue;
            }
            mentioned = true;
            if let Some(spec) = cap.get(1) {
                let spec = spec.as_str().trim_end_matches(['.', ',']);
                // Skip the echo of our own constraint; we want the
                // *other* side of the conflict when visible.
                if spec != prov.constraint.trim_start_matches(pypi_name.as_str()) {
                    required = Some(spec.to_string());
                    break;
                }
            }
        }
        if mentioned {
            out.push(ConflictAttribution {
                package: pypi_name.clone(),
                required,
                conflicting_constraint: prov.constraint.clone(),
                conda_source: prov.clone(),
            });
        }
    }
    out
}

fn apply_workspace_fact_overrides(req: &mut UvClosureRequest, facts: &[WorkspaceFactOverride]) {
    for fact in facts {
        if req
            .overrides
            .iter()
            .filter_map(|line| override_name(line))
            .any(|name| name == fact.pypi_name)
        {
            continue;
        }
        req.overrides.push(fact.requirement());
    }
}

/// Retry uv closure resolution when a structured Rule-3 signal proves that a
/// transitive upstream exact pin contradicts a precise workspace conda fact.
/// Learned facts are shared across calls so auto-route and sdist-heal relocks
/// always receive the same graph-wide overrides. Progress is monotonic and
/// finite: a retry occurs only after appending one previously unseen fact.
pub fn with_workspace_fact_overrides<S>(
    solve: S,
    learned: std::sync::Arc<std::sync::Mutex<Vec<WorkspaceFactOverride>>>,
) -> impl FnMut(UvClosureRequest) -> futures::future::BoxFuture<'static, Result<UvClosure>>
where
    S: FnMut(UvClosureRequest) -> futures::future::BoxFuture<'static, Result<UvClosure>>
        + Send
        + 'static,
{
    let solve = std::sync::Arc::new(std::sync::Mutex::new(solve));
    move |mut req: UvClosureRequest| {
        let solve = std::sync::Arc::clone(&solve);
        let learned = std::sync::Arc::clone(&learned);
        {
            let facts = learned.lock().unwrap();
            apply_workspace_fact_overrides(&mut req, &facts);
        }
        let first = {
            let mut locked = solve.lock().unwrap();
            (*locked)(req.clone())
        };
        Box::pin(async move {
            let mut attempt = first;
            loop {
                match attempt.await {
                    Ok(mut closure) => {
                        attach_effective_input_requirements(&mut closure, &req)?;
                        return Ok(closure);
                    }
                    Err(error) => {
                        let needed = match error.downcast::<WorkspaceFactOverrideNeeded>() {
                            Ok(needed) => needed,
                            Err(other) => return Err(other),
                        };
                        let original_error = needed.original_error.clone();
                        let name = needed.fact.pypi_name.clone();
                        // An existing manual/drop override is explicit intent;
                        // never stack a second uv override for the same name.
                        if req
                            .overrides
                            .iter()
                            .filter_map(|line| override_name(line))
                            .any(|existing| existing == name)
                        {
                            return Err(anyhow!(original_error));
                        }
                        let inserted = {
                            let mut facts = learned.lock().unwrap();
                            if facts.iter().any(|fact| fact.pypi_name == name) {
                                false
                            } else {
                                facts.push(needed.fact.clone());
                                true
                            }
                        };
                        if !inserted {
                            return Err(anyhow!(original_error));
                        }
                        if needed.fact.drop {
                            tracing::info!(
                                bundle = %req.bundle,
                                package = %name,
                                upstream_requirement = %needed.upstream_pin,
                                conda_version = %needed.fact.version,
                                "uv closure: dropping workspace-conda-owned dependency whose PyPI \
                                 requirement is unsatisfiable; the conda owner provides it",
                            );
                        } else {
                            tracing::info!(
                                bundle = %req.bundle,
                                package = %name,
                                upstream_pin = %needed.upstream_pin,
                                conda_version = %needed.fact.version,
                                "uv closure: replacing upstream exact pin with precise workspace conda fact",
                            );
                        }
                        req.overrides.push(needed.fact.requirement());
                        attempt = {
                            let mut locked = solve.lock().unwrap();
                            (*locked)(req.clone())
                        };
                    }
                }
            }
        })
    }
}

/// Render the human-facing failure message: verbatim uv stderr (its
/// conflict prose is good and must not be paraphrased), then the
/// provenance attribution, then the `retread solve` hint.
pub fn format_lock_failure(
    req: &UvClosureRequest,
    stderr: &str,
    attributions: &[ConflictAttribution],
) -> String {
    let mut msg = format!(
        "uv lock failed for bundle `{}` (python {}, {}):\n\n{}\n",
        req.bundle,
        req.python_version,
        req.conda_subdir,
        stderr.trim_end(),
    );
    if attributions.is_empty() {
        msg.push_str(
            "\nno generated conda constraint was named in uv's message; the conflict \
             may be intrinsic to the PyPI requirements.\n",
        );
    } else {
        msg.push_str("\nconflict attribution (conda constraint provenance):\n");
        for a in attributions {
            let required = a
                .required
                .as_deref()
                .map(|r| format!("requires `{}{}`", a.package, r))
                .unwrap_or_else(|| "is named in the conflict".to_string());
            msg.push_str(&format!(
                "  - package `{}` {} but conda pins `{}` (conda package `{}` {}, from {}, env `{}`)\n",
                a.package,
                required,
                a.conflicting_constraint,
                a.conda_source.conda_name,
                a.conda_source.conda_version,
                a.conda_source.source,
                a.conda_source.env,
            ));
        }
    }
    msg.push_str("\nhint: run `retread solve` to widen the offending conda pin.\n");
    msg
}

// ---------------------------------------------------------------------------
// uv subprocess driver
// ---------------------------------------------------------------------------

/// Resolve the uv binary (from `RETREAD_UV` or PATH) and report its
/// version string (e.g. `"0.11.15"`).
pub async fn detect_uv() -> Result<(PathBuf, String)> {
    let bin = std::env::var_os(UV_BIN_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("uv"));
    let out = tokio::process::Command::new(&bin)
        .arg("--version")
        .output()
        .await
        .with_context(|| {
            format!(
                "running `{} --version` — is uv on PATH? (override with ${UV_BIN_ENV})",
                bin.display()
            )
        })?;
    if !out.status.success() {
        bail!("`{} --version` exited with {}", bin.display(), out.status);
    }
    // "uv 0.11.15 (hash date)" -> "0.11.15"
    let stdout = String::from_utf8_lossy(&out.stdout);
    let version = stdout
        .split_whitespace()
        .nth(1)
        .unwrap_or("unknown")
        .to_string();
    Ok((bin, version))
}

/// Warn (do NOT error) when the uv on PATH differs from a previously
/// recorded version (spec §2.5's hard pin is deferred; milestone 1 warns).
pub fn warn_on_uv_version_skew(current: &str, recorded: Option<&str>) {
    match recorded {
        Some(rec) if rec != current => tracing::warn!(
            current = %current,
            recorded = %rec,
            "uv version differs from the lock-recorded version; the closure may \
             not reproduce byte-identically. Consider aligning uv with the \
             version pixi embeds.",
        ),
        _ => {}
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ClosureMeta {
    uv_version: String,
    /// Fingerprint of every resolution input uv itself cannot see in the
    /// project files: the CLI flag vector (indexes, --no-build,
    /// index-strategy) plus the synthesized pyproject and the uv version.
    /// A pre-existing `uv.lock` is only reused when this matches -- uv's
    /// own lock-freshness check validates against the pyproject TEXT, so
    /// a lock produced by an older backend (different flags, or a run
    /// where `[tool.uv]` was ignored under UV_NO_CONFIG) passes uv's
    /// check while pinning packages to the wrong index (e.g. isaacsim to
    /// pypi.org's stub sdist instead of pypi.nvidia.com's wheels).
    #[serde(default)]
    inputs_fingerprint: String,
}

/// Hex sha256 over the resolution inputs recorded in [`ClosureMeta`].
///
/// `export_args` joined (issue #10 perf): the fingerprint now also gates the
/// full-skip reuse path (fingerprint match -> re-parse the previously
/// exported pylock, NO uv invocation at all), and the pylock's content
/// depends on the `--no-emit-package` export set -- a shrunk exclusion list
/// must invalidate, or the reused pylock would be missing a package that
/// should now be emitted (the parse-time exclude filter can only REMOVE
/// packages from a superset, never restore absent ones).
fn closure_inputs_fingerprint(
    pyproject: &str,
    lock_args: &[String],
    export_args: &[String],
    uv_version: &str,
) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(pyproject.as_bytes());
    for arg in lock_args {
        h.update([0u8]);
        h.update(arg.as_bytes());
    }
    h.update([1u8]);
    for arg in export_args {
        h.update([0u8]);
        h.update(arg.as_bytes());
    }
    h.update([0u8]);
    h.update(uv_version.as_bytes());
    format!("{:x}", h.finalize())
}

const META_FILE: &str = "retread-closure.meta.json";
// uv requires the export filename to match `pylock.*.toml`.
const PYLOCK_FILE: &str = "pylock.retread.toml";
const PROVENANCE_FILE: &str = "constraints.provenance.json";
const CONFLICT_FILE: &str = "retread-conflict.json";

/// The self-heal facts learned during a heal cycle -- workspace-fact
/// overrides, routed sdist-only packages, sdist-built wheels, and transitive
/// prerelease pins. Persisting these next to the uv project and re-injecting
/// them on the next run's FIRST Pass A is what makes a cold rerun converge in
/// a single lock: the synthesized pyproject already carries the
/// overrides/pins/path-sources that made the previous run's heal succeed, so
/// Pass A resolves immediately AND the resulting pyproject text matches the
/// fingerprint recorded in
/// [`ClosureMeta`] -- so uv's own lock-freshness check reuses the healed
/// `uv.lock` instead of re-resolving the whole closure from scratch (issue
/// #10 perf: the fingerprint could never match before, because the meta was
/// written for the pinned pyproject while the next run's Pass A started
/// pinless).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HealFacts {
    /// Hash of the resolution-relevant inputs the facts were learned under
    /// (see [`heal_facts_stamp`]). Facts are ONLY replayed when the current
    /// run's stamp matches: they are injected pre-solve as first-party
    /// explicit pins / path sources / routing constraints, so replaying them
    /// under a changed manifest or routing config would force stale pins
    /// into a graph they no longer belong to (phantom pins for removed
    /// deps, wedged version bumps, ignored keep-pypi/route-policy edits).
    /// Files written by older binaries deserialize with an empty stamp and
    /// are discarded (one extra heal, never a stale replay).
    #[serde(default)]
    pub stamp: String,
    /// Graph-wide overrides learned from exact upstream pins that contradicted
    /// precise workspace-solved conda facts. Ordered by discovery so warm
    /// replay emits byte-identical pyproject text.
    #[serde(default)]
    pub workspace_overrides: Vec<WorkspaceFactOverride>,
    #[serde(default)]
    pub routed: Vec<AutoRoutedPackage>,
    #[serde(default)]
    pub built: Vec<BuiltSdistWheel>,
    #[serde(default)]
    pub prereleased: Vec<PrereleasePin>,
}

impl HealFacts {
    pub fn is_empty(&self) -> bool {
        self.workspace_overrides.is_empty()
            && self.routed.is_empty()
            && self.built.is_empty()
            && self.prereleased.is_empty()
    }
}

const HEAL_FACTS_STAMP_SCHEMA: &str = "v2-sdist-route-version-domains";

/// Hex sha256 over every input that decides whether persisted heal facts
/// are still VALID to replay: the request's roots/constraints/overrides/
/// exclusions/indexes (manifest-derived resolution inputs), the routing
/// policy knobs (`route-policy`, `route-include`, `keep-pypi`,
/// `force-conda`, `name-map`, protected roots), AND the `sdist-build`
/// policy. Any change to any of these must discard the facts (cold
/// re-heal) rather than replay them.
///
/// `sdist_build_policy` is in the stamp because it selects the Pass-B
/// relaxation ([`LockRelaxations::pass_b_for`]): under `auto` the heal may
/// build (and persist) sdist wheels that `never` would refuse, so a facts
/// file learned under one policy can carry built-wheel / route facts that
/// are invalid to replay under the other. Without it, flipping the policy
/// between runs could replay a stale facts file as if still valid.
///
/// Deliberately EXCLUDES per-round mutable state (`explicit_pins`,
/// `built_wheel_sources`, learned workspace overrides, auto-route constraints
/// applied by [`apply_auto_route`]) -- the stamp must be computed from the
/// BASE request, before any facts/heal state is injected, and stay stable
/// across fixpoint rounds. Also excludes `workspace_conda_versions` (populated
/// later in the phase and not a fact-validity input).
pub fn heal_facts_stamp(
    req: &UvClosureRequest,
    opts: &AutoRouteOptions,
    sdist_build_policy: crate::config::SdistBuildPolicy,
) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    let mut field = |tag: &str, vals: &mut dyn Iterator<Item = &str>| {
        h.update(tag.as_bytes());
        h.update([0xffu8]);
        for v in vals {
            h.update(v.as_bytes());
            h.update([0u8]);
        }
    };
    field("schema", &mut std::iter::once(HEAL_FACTS_STAMP_SCHEMA));
    field("python", &mut std::iter::once(req.python_version.as_str()));
    field("subdir", &mut std::iter::once(req.conda_subdir.as_str()));
    field("deps", &mut req.dependencies.iter().map(String::as_str));
    field(
        "constraints",
        &mut req.constraints.constraints.iter().map(String::as_str),
    );
    field("overrides", &mut req.overrides.iter().map(String::as_str));
    field(
        "no-emit",
        &mut req.no_emit_packages.iter().map(String::as_str),
    );
    field("indexes", &mut req.index_urls.iter().map(String::as_str));
    let policy = format!("{:?}|enabled={}", opts.route_policy, opts.enabled);
    field("route-policy", &mut std::iter::once(policy.as_str()));
    field(
        "route-include",
        &mut opts.route_include.iter().map(String::as_str),
    );
    field("keep-pypi", &mut opts.keep_pypi.iter().map(String::as_str));
    field(
        "force-conda",
        &mut opts.force_conda.iter().map(String::as_str),
    );
    let name_map: Vec<String> = opts
        .name_map
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    field("name-map", &mut name_map.iter().map(String::as_str));
    field("protected", &mut opts.protected.iter().map(String::as_str));
    field(
        "workspace-owned",
        &mut opts.workspace_owned.iter().map(String::as_str),
    );
    field(
        "workspace-facts",
        &mut std::iter::once(opts.workspace_fact_fingerprint.as_str()),
    );
    let anchors: Vec<String> = opts
        .abi_anchor_pins
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    field("abi-anchors", &mut anchors.iter().map(String::as_str));
    let sdist_build = format!("{sdist_build_policy:?}");
    field("sdist-build", &mut std::iter::once(sdist_build.as_str()));
    format!("{:x}", h.finalize())
}

/// Absolute path of the persisted heal-facts file for a target.
///
/// Stored under `<cache_dir>/retread-heal-facts/`, a SIBLING of
/// `<cache_dir>/uv-projects/` -- deliberately OUTSIDE the uv project so it
/// survives "delete the uv-projects state" (the operation that forces a
/// cold re-resolve). That survival is the whole point: after wiping the
/// uv.lock, the persisted facts still seed the first Pass A so it converges
/// in a single lock (issue #10 perf, item 3b). Keyed identically to the uv
/// project dir (bundle + python minor + subdir) so each target has its own.
pub fn heal_facts_path(
    cache_dir: &Path,
    bundle: &str,
    python_version: &str,
    subdir: &str,
) -> PathBuf {
    cache_dir.join("retread-heal-facts").join(format!(
        "{}-py{}-{}.json",
        canonical_conda_name(bundle),
        python_version,
        subdir,
    ))
}

/// Load persisted heal facts from `path`, dropping any built-wheel entry
/// whose store path no longer exists (the content-addressed wheel store is
/// durable, but a pruned cache must fall back to a rebuild rather than feed
/// uv a `[tool.uv.sources]` path that 404s). Missing/corrupt file -> empty
/// facts (cold start).
///
/// `expected_stamp` (from [`heal_facts_stamp`] over the CURRENT run's base
/// request + routing options) gates replay: a mismatch means the manifest
/// or routing config changed since the facts were learned, and the whole
/// file is discarded -- replaying pins/routes learned under a different
/// manifest state can phantom-pin removed deps or wedge version bumps.
pub fn load_heal_facts(path: &Path, expected_stamp: &str) -> HealFacts {
    let Ok(text) = std::fs::read_to_string(path) else {
        return HealFacts::default();
    };
    let mut facts: HealFacts = match serde_json::from_str(&text) {
        Ok(f) => f,
        Err(e) => {
            tracing::debug!(path = %path.display(), error = %e, "heal facts: unreadable, ignoring");
            return HealFacts::default();
        }
    };
    if facts.stamp != expected_stamp {
        tracing::info!(
            path = %path.display(),
            "heal facts: resolution inputs (manifest/routing config) changed \
             since the facts were learned; discarding for a fresh heal",
        );
        return HealFacts::default();
    }
    facts.built.retain(|w| {
        let present = w.wheel_path.exists();
        if !present {
            tracing::info!(
                pkg = %w.pypi_name,
                path = %w.wheel_path.display(),
                "heal facts: built wheel missing from store; dropping (will rebuild if needed)",
            );
        }
        present
    });
    facts
}

/// Persist heal facts atomically (temp + rename) to `path` (see
/// [`heal_facts_path`]). Best-effort: a write failure only costs the reuse
/// optimization on the next run, never correctness. Writing empty facts
/// removes any stale file so a pack that stopped needing a heal doesn't keep
/// injecting dead pins.
pub fn save_heal_facts(path: &Path, facts: &HealFacts) {
    if facts.is_empty() {
        let _ = std::fs::remove_file(path);
        return;
    }
    let Ok(json) = serde_json::to_string_pretty(facts) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // Tmp name keeps the full destination filename (bundle + python +
    // subdir) and adds pid + per-process sequence, so concurrent writers --
    // even for DIFFERENT targets in the same directory -- can never promote
    // each other's torn temp file.
    static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = path.with_file_name(format!(
        "{}.{}.{}.tmp",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("heal-facts.json"),
        std::process::id(),
        TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    ));
    if std::fs::write(&tmp, json.as_bytes()).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// The relaxations Pass B may apply over Pass A when `uv lock` fails.
/// SINGLE-SOURCED so the Pass A and Pass B argument vectors can only
/// differ by these known bits: any new restrictive flag added to
/// [`build_lock_args`] appears in BOTH passes automatically (guarded by
/// `pass_a_and_pass_b_differ_only_by_known_relaxations`), closing the
/// stderr-regex whack-a-mole that issue #10 kept re-opening.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LockRelaxations {
    /// Allow uv to build sdists (drop `--no-build`) so sdist-only
    /// transitive deps can resolve. Gated by `sdist-build` policy.
    allow_sdist_build: bool,
    /// `--prerelease` policy value.
    prerelease: &'static str,
}

impl LockRelaxations {
    /// Pass A: retread's normal restrictions -- no sdist builds (retread's
    /// source_build path owns builds), pre-releases only when strictly
    /// necessary or explicitly first-party.
    const PASS_A: Self = Self {
        allow_sdist_build: false,
        prerelease: "if-necessary-or-explicit",
    };
    /// Pass B under `sdist-build = "auto"`: relax BOTH the build gate and
    /// the prerelease policy in one retry.
    const PASS_B_AUTO: Self = Self {
        allow_sdist_build: true,
        prerelease: "allow",
    };
    /// Pass B under `sdist-build = "never"`: prerelease relaxation ONLY
    /// (build policy is orthogonal; `--no-build` stays, so sdist-only
    /// packages still can't resolve -- only transitive-prerelease healing
    /// works under Never).
    const PASS_B_NEVER: Self = Self {
        allow_sdist_build: false,
        prerelease: "allow",
    };

    /// Pass B relaxations appropriate for the pack's build policy.
    fn pass_b_for(policy: crate::config::SdistBuildPolicy) -> Self {
        match policy {
            crate::config::SdistBuildPolicy::Auto => Self::PASS_B_AUTO,
            crate::config::SdistBuildPolicy::Never => Self::PASS_B_NEVER,
        }
    }
}

/// Assemble the `uv lock` argument vector. COMMON args are single-sourced
/// here; only the [`LockRelaxations`]-controlled bits (`--no-build`
/// presence and the `--prerelease` value) vary between Pass A and Pass B.
///
/// CRITICAL: uv runs with `UV_NO_CONFIG=1` to isolate the resolve from
/// user-level uv.toml, which on uv 0.11.x ALSO strips config-file-class
/// keys (`no-build`, `index-strategy`, `prerelease`, `[[tool.uv.index]]`)
/// from the synthesized pyproject's `[tool.uv]` table -- so every such
/// setting is passed here as a CLI flag, which `UV_NO_CONFIG` never
/// touches. Without the index flags uv falls back to pypi.org alone (where
/// e.g. `isaacsim` is only a stub sdist) and the closure exports zero
/// wheels.
fn build_lock_args(
    project_dir: &Path,
    python_version: &str,
    index_urls: &[String],
    offline: bool,
    relax: LockRelaxations,
) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "lock".into(),
        "--project".into(),
        project_dir.to_string_lossy().into_owned(),
        "--python".into(),
        python_version.to_string(),
        "--no-progress".into(),
        "--color".into(),
        "never".into(),
    ];
    // Relaxation-controlled: `--no-build` is a DENY flag, present only when
    // sdist builds are NOT allowed.
    if !relax.allow_sdist_build {
        args.push("--no-build".into());
    }
    args.push("--index-strategy".into());
    args.push("unsafe-best-match".into());
    // Relaxation-controlled: the `--prerelease` VALUE differs.
    args.push("--prerelease".into());
    args.push(relax.prerelease.into());
    for url in index_urls {
        args.push("--index".into());
        args.push(url.clone());
    }
    if offline {
        args.push("--offline".into());
    }
    args
}

/// Assemble the `uv export` argument vector (single-sourced so the green
/// path and the Pass-B offender-detection export stay identical).
fn build_export_args(
    project_dir: &Path,
    no_emit_packages: &[String],
    offline: bool,
) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "export".into(),
        "--project".into(),
        project_dir.to_string_lossy().into_owned(),
        "--format".into(),
        "pylock.toml".into(),
        "--frozen".into(),
        "--no-emit-project".into(),
        "--no-annotate".into(),
        "--no-progress".into(),
        "--color".into(),
        "never".into(),
        "--output-file".into(),
        PYLOCK_FILE.into(),
    ];
    for name in no_emit_packages {
        args.push("--no-emit-package".into());
        args.push(canonical_conda_name(name));
    }
    if offline {
        args.push("--offline".into());
    }
    args
}

/// Compute the closure for `req` under `project_dir` (created if absent):
/// write the synthesized project, run `uv lock` + `uv export`, parse the
/// pylock. `recorded_uv_version` (when Some, e.g. from a committed lock)
/// triggers the skew warning; the version used is also persisted next to
/// the project so back-to-back runs self-check.
pub async fn compute_closure(
    req: &UvClosureRequest,
    project_dir: &Path,
    uv_cache_dir: &Path,
    recorded_uv_version: Option<&str>,
    sdist_build_policy: crate::config::SdistBuildPolicy,
) -> Result<UvClosure> {
    let (uv_bin, uv_version) = detect_uv().await?;
    tracing::info!(
        uv = %uv_bin.display(),
        version = %uv_version,
        bundle = %req.bundle,
        python = %req.python_version,
        subdir = %req.conda_subdir,
        "uv closure: resolving via uv",
    );
    warn_on_uv_version_skew(&uv_version, recorded_uv_version);
    let meta_path = project_dir.join(META_FILE);
    if recorded_uv_version.is_none()
        && let Ok(prev) = std::fs::read_to_string(&meta_path)
        && let Ok(meta) = serde_json::from_str::<ClosureMeta>(&prev)
    {
        warn_on_uv_version_skew(&uv_version, Some(&meta.uv_version));
    }

    tokio::fs::create_dir_all(project_dir)
        .await
        .with_context(|| format!("creating uv project dir {}", project_dir.display()))?;
    tokio::fs::create_dir_all(uv_cache_dir)
        .await
        .with_context(|| format!("creating uv cache dir {}", uv_cache_dir.display()))?;
    let pyproject_text = synthesize_pyproject(req);
    tokio::fs::write(project_dir.join("pyproject.toml"), &pyproject_text)
        .await
        .context("writing synthesized pyproject.toml")?;
    tokio::fs::write(
        project_dir.join(PROVENANCE_FILE),
        provenance_json(&req.constraints)?,
    )
    .await
    .context("writing constraints.provenance.json")?;

    let run = |args: Vec<String>| {
        let uv_bin = uv_bin.clone();
        let project_dir = project_dir.to_path_buf();
        let uv_cache_dir = uv_cache_dir.to_path_buf();
        async move {
            tokio::process::Command::new(&uv_bin)
                .args(&args)
                .current_dir(&project_dir)
                .env("UV_CACHE_DIR", &uv_cache_dir)
                .env("UV_NO_CONFIG", "1")
                .output()
                .await
                .with_context(|| format!("spawning `{} {}`", uv_bin.display(), args.join(" ")))
        }
    };

    // -- uv lock (Pass A) --------------------------------------------------
    // Pass A uses retread's normal restrictions. On failure, `compute_
    // closure` runs a STRUCTURED two-pass detection (Pass B, relaxed) to
    // name the offenders exactly -- see the `HealNeeded` module doc.
    let lock_args = build_lock_args(
        project_dir,
        &req.python_version,
        &req.index_urls,
        req.offline,
        LockRelaxations::PASS_A,
    );

    // Stale-lock guard: `uv lock` reuses a pre-existing uv.lock whenever
    // it still satisfies the pyproject TEXT -- it cannot see the CLI
    // flags above, so a lock written by a different backend version /
    // flag set survives validation while pinning packages to the wrong
    // index. Drop the lock (forcing a fresh resolve) whenever the
    // recorded input fingerprint is absent or different.
    //
    // ISSUE #10 PERF ROOT CAUSE (why cross-run reuse used to always miss):
    // the fingerprint is over the SYNTHESIZED pyproject. A HEALED run writes
    // the meta for the pinned pyproject (explicit_pins + built-wheel
    // path-sources appended). The next run's FIRST Pass A started PINLESS,
    // so its pyproject -- and therefore its fingerprint -- never matched the
    // recorded one; the healed uv.lock was discarded and the whole 9-min
    // closure re-resolved from scratch every time. The persisted heal facts
    // (see `HealFacts` / `load_heal_facts`, re-seeded into the ledgers before
    // the first solve) make Pass A carry the same pins the meta was written
    // for, so the fingerprint matches and this guard KEEPS the lock -> uv
    // fast-relocks instead of re-resolving.
    let export_args = build_export_args(project_dir, &req.no_emit_packages, req.offline);
    let fingerprint =
        closure_inputs_fingerprint(&pyproject_text, &lock_args, &export_args, &uv_version);
    let lock_file = project_dir.join("uv.lock");
    let pylock_file = project_dir.join(PYLOCK_FILE);
    let recorded_fingerprint = std::fs::read_to_string(&meta_path)
        .ok()
        .and_then(|s| serde_json::from_str::<ClosureMeta>(&s).ok())
        .map(|m| m.inputs_fingerprint);
    let fingerprint_matches = recorded_fingerprint.as_deref() == Some(fingerprint.as_str());

    // FULL-SKIP reuse (issue #10 perf, the warm path): identical resolution
    // inputs => identical exported pylock, so re-parse the one on disk and
    // return WITHOUT invoking uv at all. Measured motivation: even with a
    // valid uv.lock kept by the guard below, a bare `uv lock` re-validated
    // every package's metadata against pypi.nvidia.com (no PEP 658 sidecars
    // -> per-package range reads, ~9.5 min for the isaac closure) -- as
    // expensive as a cold resolve. The fingerprint covers the pyproject,
    // the lock flag vector, the export flag vector (`--no-emit-package`
    // set), and the uv version, so any drift takes the normal path. A parse
    // failure (corrupt/truncated pylock) also falls through.
    if fingerprint_matches && lock_file.exists() && pylock_file.exists() {
        match tokio::fs::read_to_string(&pylock_file).await {
            Ok(pylock) => {
                let exclude: BTreeSet<String> = req
                    .no_emit_packages
                    .iter()
                    .map(|n| canonical_conda_name(n))
                    .collect();
                let target = WheelTarget::for_subdir(&req.python_version, &req.conda_subdir);
                match parse_pylock_closure(&pylock, &target, &exclude, &uv_version) {
                    Ok(mut closure) => {
                        attach_effective_input_requirements(&mut closure, req)?;
                        tracing::info!(
                            bundle = %req.bundle,
                            wheels = closure.wheels.len(),
                            pins = closure.pins.len(),
                            "uv closure: inputs unchanged; reusing exported pylock \
                             (no uv invocation)",
                        );
                        return Ok(closure);
                    }
                    Err(e) => {
                        tracing::warn!(
                            bundle = %req.bundle,
                            error = %format!("{e:#}"),
                            "uv closure: cached pylock unusable; re-resolving",
                        );
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    bundle = %req.bundle,
                    error = %e,
                    "uv closure: cached pylock unreadable; re-resolving",
                );
            }
        }
    }

    if lock_file.exists() && !fingerprint_matches {
        tracing::info!(
            bundle = %req.bundle,
            "uv closure: resolution inputs changed since the cached uv.lock \
             was written; discarding it for a fresh resolve",
        );
        let _ = std::fs::remove_file(&lock_file);
    }

    let lock_out = run(lock_args).await?;
    if !lock_out.status.success() {
        let stderr = String::from_utf8_lossy(&lock_out.stderr).into_owned();
        let attributions = attribute_conflict(&stderr, &req.constraints.provenance);
        // Machine-readable record next to the project (spec §4a).
        let record = serde_json::json!({
            "bundle": req.bundle,
            "python": req.python_version,
            "platform": req.conda_subdir,
            "uv_stderr": stderr,
            "attributions": attributions,
        });
        let _ = std::fs::write(
            project_dir.join(CONFLICT_FILE),
            serde_json::to_string_pretty(&record).unwrap_or_default(),
        );
        let original_error = format_lock_failure(req, &stderr, &attributions);

        // The manylinux platform-tag ceiling is a DIFFERENT recovery
        // layer's job (glibc relaxation, `installer::is_platform_tag_
        // conflict`) -- never enter the sdist/prerelease two-pass for it
        // (Pass B could "heal" it by building an sdist for a package that
        // publishes wheels for other platforms, stealing the glibc-relax
        // path's ownership).
        if crate::installer::is_platform_tag_conflict(&stderr) {
            bail!("{original_error}");
        }

        if let Some(needed) = workspace_fact_override_needed(req, &attributions, &original_error) {
            return Err(anyhow::Error::new(needed));
        }

        // -- uv lock (Pass B): relax the offending restrictions ----------
        // Same invocation as Pass A, single-sourced through
        // `build_lock_args`, differing ONLY by the known relaxations
        // (drop `--no-build` under `sdist-build = "auto"`, and set
        // `--prerelease allow` always). Pass B may build sdists to extract
        // metadata; those artifacts are discarded (we only inspect the
        // resolved lock). Pass B inherits Pass A's `--offline` flag: an
        // offline replay stays offline (deterministic/sandbox-safe) and
        // simply fails Pass B if it can't fetch sdist metadata, which then
        // surfaces Pass A's error below -- exactly the pre-two-pass
        // behavior.
        let pass_b_args = build_lock_args(
            project_dir,
            &req.python_version,
            &req.index_urls,
            req.offline,
            LockRelaxations::pass_b_for(sdist_build_policy),
        );
        let pass_b_out = run(pass_b_args).await?;
        if !pass_b_out.status.success() {
            // Pass B can uncover an exact-pin contradiction that Pass A's
            // no-build/prerelease error masked. Classify that conflict before
            // falling back to Pass A's error; every non-workspace or non-exact
            // conflict keeps the historical behavior.
            let pass_b_stderr = String::from_utf8_lossy(&pass_b_out.stderr).into_owned();
            let pass_b_attributions =
                attribute_conflict(&pass_b_stderr, &req.constraints.provenance);
            let pass_b_error = format_lock_failure(req, &pass_b_stderr, &pass_b_attributions);
            if let Some(needed) =
                workspace_fact_override_needed(req, &pass_b_attributions, &pass_b_error)
            {
                return Err(anyhow::Error::new(needed));
            }
            // Rule 1 (last resort, both passes failed): a transitive dependency
            // the workspace conda-OWNS whose PyPI requirement neither pass could
            // satisfy (e.g. an exact `Requires-Dist` pin on a version that only
            // exists in conda's numbering, `opencv-python==4.11.0` -> conda
            // `py-opencv`). Drop it graph-wide so the closure succeeds and the
            // conda owner supplies the import. Uses Pass A's attributions/error
            // -- the surfaced diagnostic's provenance. Declines to the ordinary
            // error for any non-owned or genuinely-conflicting name.
            if let Some(needed) = workspace_owned_drop_needed(req, &attributions, &original_error) {
                return Err(anyhow::Error::new(needed));
            }
            bail!("{original_error}");
        }

        // Pass B resolved. Export its lock and read the offenders
        // STRUCTURALLY from the pylock document (no stderr prose parsing).
        let pass_b_export = run(build_export_args(
            project_dir,
            &req.no_emit_packages,
            req.offline,
        ))
        .await?;
        if !pass_b_export.status.success() {
            // Can't inspect the Pass B lock -> fall back to Pass A's error.
            bail!("{original_error}");
        }
        let pass_b_pylock = tokio::fs::read_to_string(project_dir.join(PYLOCK_FILE))
            .await
            .context("reading Pass-B pylock for offender detection")?;
        let offenders = classify_pylock_offenders(&pass_b_pylock)?;
        if offenders.sdist_only.is_empty() && offenders.prerelease.is_empty() {
            // Pass B succeeded but named no healable offender (whatever the
            // relaxation flipped, it isn't a class we repair) -> surface
            // Pass A's error rather than loop.
            bail!("{original_error}");
        }
        return Err(anyhow::Error::new(HealNeeded {
            sdist_only: offenders.sdist_only,
            prerelease: offenders.prerelease,
            original_error,
        }));
    }

    // -- uv export ---------------------------------------------------------
    // Same vector the fingerprint above was computed over.
    let export_out = run(export_args).await?;
    if !export_out.status.success() {
        bail!(
            "uv export failed for bundle `{}`:\n{}",
            req.bundle,
            String::from_utf8_lossy(&export_out.stderr),
        );
    }

    let pylock = tokio::fs::read_to_string(project_dir.join(PYLOCK_FILE))
        .await
        .context("reading exported pylock.retread.toml")?;
    // Belt-and-braces authoritative post-filter (AMENDMENT A1).
    let exclude: BTreeSet<String> = req
        .no_emit_packages
        .iter()
        .map(|n| canonical_conda_name(n))
        .collect();
    let target = WheelTarget::for_subdir(&req.python_version, &req.conda_subdir);
    let mut closure = parse_pylock_closure(&pylock, &target, &exclude, &uv_version)?;
    attach_effective_input_requirements(&mut closure, req)?;

    let _ = std::fs::write(
        &meta_path,
        serde_json::to_string_pretty(&ClosureMeta {
            uv_version: uv_version.clone(),
            inputs_fingerprint: fingerprint,
        })
        .unwrap_or_default(),
    );

    tracing::info!(
        bundle = %req.bundle,
        wheels = closure.wheels.len(),
        pins = closure.pins.len(),
        "uv closure: resolved",
    );
    Ok(closure)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn target(py: &str, subdir: &str) -> WheelTarget {
        WheelTarget {
            python_version: py.to_string(),
            conda_subdir: subdir.to_string(),
            max_glibc: None,
        }
    }

    fn sample_request() -> UvClosureRequest {
        let mut conda_deps = BTreeMap::new();
        conda_deps.insert("pytorch-gpu".to_string(), "==2.10.0".to_string());
        conda_deps.insert("numpy".to_string(), ">=1.26,<3".to_string());
        conda_deps.insert("python".to_string(), "3.12.*".to_string());
        let mut name_map = BTreeMap::new();
        name_map.insert("torch".to_string(), "pytorch-gpu".to_string());
        let constraints = build_constraints(
            &conda_deps,
            &name_map,
            &Default::default(),
            "manifest",
            "default",
        );
        let mut built = BTreeMap::new();
        built.insert(
            "isaaclab".to_string(),
            PathBuf::from("wheels/isaaclab/isaaclab-2.0.0-py3-none-any.whl"),
        );
        UvClosureRequest {
            bundle: "isaac-pack-latest".to_string(),
            python_version: "3.12".to_string(),
            conda_subdir: "linux-64".to_string(),
            dependencies: vec![
                "isaacsim[all,extscache]==5.1.0".to_string(),
                "mujoco==3.5.0".to_string(),
            ],
            constraints,
            overrides: vec![
                "protobuf>=4".to_string(),
                format!("pywin32 ; {DROP_MARKER}"),
            ],
            no_emit_packages: vec!["numpy".to_string(), "torch".to_string()],
            index_urls: vec![
                "https://pypi.nvidia.com".to_string(),
                "https://pypi.org/simple/".to_string(),
            ],
            built_wheel_sources: built,
            explicit_pins: BTreeMap::new(),
            offline: false,
        }
    }

    // ---- ephemeral project synthesis ------------------------------------

    #[test]
    fn synthesize_pyproject_golden() {
        let req = sample_request();
        let got = synthesize_pyproject(&req);
        let want = r#"# Generated by pixi-build-retread (retread-resolver = "uv"). Do not edit.
[project]
name = "retread-closure-isaac-pack-latest"
version = "0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "isaacsim[all,extscache]==5.1.0",
    "mujoco==3.5.0",
]

[tool.uv]
environments = ["python_version >= '3.12' and python_version < '3.13' and sys_platform == 'linux' and platform_machine == 'x86_64'"]
no-build = true
index-strategy = "unsafe-best-match"
prerelease = "if-necessary-or-explicit"
constraint-dependencies = [
    "numpy>=1.26,<3",
    "torch==2.10.0",
]
override-dependencies = [
    "protobuf>=4",
    "pywin32 ; python_version < '0'",
]

[[tool.uv.index]]
url = "https://pypi.nvidia.com"

[[tool.uv.index]]
url = "https://pypi.org/simple/"

[tool.uv.sources]
isaaclab = { path = "wheels/isaaclab/isaaclab-2.0.0-py3-none-any.whl" }
"#;
        assert_eq!(got, want);
        // And it must be valid TOML.
        toml::from_str::<toml::Value>(&got).expect("synthesized pyproject parses as TOML");
    }

    /// A direct-URL wheel (isaacsim-extscache-kit) that the handler pre-fetched
    /// into the content-addressed store is emitted as a BARE first-party
    /// requirement plus a `[tool.uv.sources]` `path =` source -- NOT as a
    /// `name @ https://...` direct-URL requirement. This is what makes uv read
    /// METADATA from a local seekable zip instead of downloading the whole
    /// no-store 5.9 GiB wheel on every lock.
    #[test]
    fn synthesize_pyproject_url_wheel_renders_path_source_not_direct_url() {
        let mut built = BTreeMap::new();
        built.insert(
            "isaacsim-extscache-kit".to_string(),
            PathBuf::from(
                "/store/deadbeef/isaacsim_extscache_kit-5.1.0-cp312-cp312-manylinux_2_34_x86_64.whl",
            ),
        );
        let req = UvClosureRequest {
            bundle: "isaac-pack-latest".to_string(),
            python_version: "3.12".to_string(),
            conda_subdir: "linux-64".to_string(),
            // Bare first-party requirement (the path source binds by name).
            dependencies: vec!["isaacsim-extscache-kit".to_string()],
            constraints: ConstraintSet::default(),
            overrides: vec![],
            no_emit_packages: vec![],
            index_urls: vec!["https://pypi.nvidia.com".to_string()],
            built_wheel_sources: built,
            explicit_pins: BTreeMap::new(),
            offline: false,
        };
        let got = synthesize_pyproject(&req);
        // No direct-URL requirement leaked into the closure project.
        assert!(
            !got.contains(" @ https://"),
            "url wheel must not be emitted as a direct-URL requirement:\n{got}"
        );
        // Bare requirement + local path source.
        assert!(got.contains("\"isaacsim-extscache-kit\","), "{got}");
        assert!(
            got.contains(
                "isaacsim-extscache-kit = { path = \"/store/deadbeef/\
                 isaacsim_extscache_kit-5.1.0-cp312-cp312-manylinux_2_34_x86_64.whl\" }"
            ),
            "{got}"
        );
        toml::from_str::<toml::Value>(&got).expect("synthesized pyproject parses as TOML");
    }

    #[test]
    fn synthesize_pyproject_noarch_keeps_python_environments() {
        // Unknown subdir: the platform clause is dropped but the python
        // clause still restricts universal resolution.
        let mut req = sample_request();
        req.conda_subdir = "noarch".to_string();
        let got = synthesize_pyproject(&req);
        assert!(
            got.contains(
                "environments = [\"python_version >= '3.12' and python_version < '3.13'\"]"
            )
        );
        assert!(got.contains("no-build = true"));
    }

    #[test]
    fn synthesize_pyproject_unparseable_python_falls_back() {
        let mut req = sample_request();
        req.python_version = "weird".to_string();
        let got = synthesize_pyproject(&req);
        assert!(got.contains("requires-python = \"==weird.*\""));
        // Platform-only marker; no python clause.
        assert!(got.contains(
            "environments = [\"sys_platform == 'linux' and platform_machine == 'x86_64'\"]"
        ));
    }

    #[test]
    fn synthesize_pyproject_sets_prerelease_policy() {
        let got = synthesize_pyproject(&sample_request());
        assert!(got.contains("prerelease = \"if-necessary-or-explicit\""));
    }

    #[test]
    fn python_minor_bounds_matrix() {
        assert_eq!(
            python_minor_bounds("3.12"),
            Some(("3.12".to_string(), "3.13".to_string()))
        );
        // Patch component is ignored; only the minor is bounded.
        assert_eq!(
            python_minor_bounds("3.10.14"),
            Some(("3.10".to_string(), "3.11".to_string()))
        );
        assert_eq!(
            python_minor_bounds("3.9"),
            Some(("3.9".to_string(), "3.10".to_string()))
        );
        assert_eq!(python_minor_bounds("3"), None);
        assert_eq!(python_minor_bounds("weird"), None);
        assert_eq!(python_minor_bounds(""), None);
    }

    #[test]
    fn environment_marker_matrix() {
        assert_eq!(
            environment_marker("3.12", "linux-64").as_deref(),
            Some(
                "python_version >= '3.12' and python_version < '3.13' \
                 and sys_platform == 'linux' and platform_machine == 'x86_64'"
            )
        );
        assert_eq!(
            environment_marker("3.11", "osx-arm64").as_deref(),
            Some(
                "python_version >= '3.11' and python_version < '3.12' \
                 and sys_platform == 'darwin' and platform_machine == 'arm64'"
            )
        );
        assert_eq!(
            environment_marker("3.12", "win-64").as_deref(),
            Some(
                "python_version >= '3.12' and python_version < '3.13' \
                 and sys_platform == 'win32' and platform_machine == 'AMD64'"
            )
        );
        // noarch keeps the python clause.
        assert_eq!(
            environment_marker("3.12", "noarch").as_deref(),
            Some("python_version >= '3.12' and python_version < '3.13'")
        );
        // Unknown python keeps the platform clause.
        assert_eq!(
            environment_marker("weird", "linux-64").as_deref(),
            Some("sys_platform == 'linux' and platform_machine == 'x86_64'")
        );
        // Neither half known: omitted.
        assert_eq!(environment_marker("weird", "noarch"), None);
    }

    // ---- built-in Windows-only drop injection ----------------------------

    #[test]
    fn built_in_win_only_injected_on_linux_without_pack_drop_deps() {
        // (a) A linux-64 closure with NO pack drop-deps still injects the
        // built-in shims. Assert the marker override is present for both
        // idna-ssl and pywin32, and that it renders into the pyproject.
        let names = built_in_win_only_to_inject("linux-64", |_| false, &[], &[]);
        assert!(names.contains(&"idna-ssl"), "idna-ssl injected: {names:?}");
        assert!(names.contains(&"pywin32"), "pywin32 injected: {names:?}");
        assert_eq!(names, crate::config::BUILT_IN_WIN_ONLY.to_vec());

        // End-to-end: the injected override lines render as unmatchable
        // markers in the synthesized project.
        let mut req = sample_request();
        req.overrides = names
            .iter()
            .map(|n| format!("{n} ; {DROP_MARKER}"))
            .collect();
        let got = synthesize_pyproject(&req);
        assert!(got.contains("\"idna-ssl ; python_version < '0'\""), "{got}");
        assert!(got.contains("\"pywin32 ; python_version < '0'\""), "{got}");
    }

    #[test]
    fn built_in_win_only_injects_none_on_windows() {
        // (b) win-64 (and the other win subdirs) inject nothing — the
        // shims are legitimate there.
        assert!(built_in_win_only_to_inject("win-64", |_| false, &[], &[]).is_empty());
        assert!(built_in_win_only_to_inject("win-32", |_| false, &[], &[]).is_empty());
        assert!(built_in_win_only_to_inject("win-arm64", |_| false, &[], &[]).is_empty());
    }

    #[test]
    fn built_in_win_only_user_override_suppresses_injection() {
        // (c) A user override for one of the names suppresses just that
        // name; the others still inject.
        let overridden = |name: &str| name == "pywin32";
        let names = built_in_win_only_to_inject("linux-64", overridden, &[], &[]);
        assert!(
            !names.contains(&"pywin32"),
            "override must suppress pywin32: {names:?}"
        );
        assert!(
            names.contains(&"idna-ssl"),
            "un-overridden names still inject: {names:?}"
        );
    }

    #[test]
    fn built_in_win_only_coexists_with_pack_drop_deps() {
        // (d) A pack that ALREADY drops one built-in (in any name form,
        // e.g. underscore) does not get a duplicate; a non-built-in
        // drop-dep is untouched and the remaining built-ins still inject.
        let drops = vec!["pywin32".to_string(), "some_other_pkg".to_string()];
        let names = built_in_win_only_to_inject("linux-64", |_| false, &drops, &[]);
        assert!(
            !names.contains(&"pywin32"),
            "already-dropped built-in must not duplicate: {names:?}"
        );
        assert!(names.contains(&"idna-ssl"), "{names:?}");

        // Non-canonical form of an already-dropped built-in is deduped too.
        let drops2 = vec!["idna_ssl".to_string()];
        let names2 = built_in_win_only_to_inject("linux-64", |_| false, &drops2, &[]);
        assert!(
            !names2.contains(&"idna-ssl"),
            "canonicalized dedupe against idna_ssl failed: {names2:?}"
        );
    }

    #[test]
    fn built_in_win_only_first_party_root_suppresses_injection() {
        // (e) A user who declares one of the built-in names as an explicit
        // TOP-LEVEL (first-party) root requirement wants it resolved
        // deliberately -- the injected graph-wide marker must NOT suppress
        // it. Just that name is skipped; the other built-ins still inject.
        let first_party = vec!["idna-ssl".to_string()];
        let names = built_in_win_only_to_inject("linux-64", |_| false, &[], &first_party);
        assert!(
            !names.contains(&"idna-ssl"),
            "first-party root must suppress idna-ssl injection: {names:?}"
        );
        assert!(
            names.contains(&"pywin32"),
            "un-declared built-ins still inject: {names:?}"
        );

        // Non-canonical form of the first-party name is deduped too.
        let first_party2 = vec!["idna_ssl".to_string()];
        let names2 = built_in_win_only_to_inject("linux-64", |_| false, &[], &first_party2);
        assert!(
            !names2.contains(&"idna-ssl"),
            "canonicalized dedupe against first-party idna_ssl failed: {names2:?}"
        );
    }

    // ---- constraint generation + provenance ------------------------------

    #[test]
    fn conda_spec_to_pep440_matrix() {
        assert_eq!(conda_spec_to_pep440("==1.2.3").as_deref(), Some("==1.2.3"));
        assert_eq!(
            conda_spec_to_pep440(">=1.2,<2").as_deref(),
            Some(">=1.2,<2")
        );
        assert_eq!(conda_spec_to_pep440("~=2.1").as_deref(), Some("~=2.1"));
        assert_eq!(conda_spec_to_pep440("1.2.*").as_deref(), Some("==1.2.*"));
        assert_eq!(conda_spec_to_pep440("1.2").as_deref(), Some("==1.2.*"));
        assert_eq!(conda_spec_to_pep440("=1.2").as_deref(), Some("==1.2.*"));
        assert_eq!(conda_spec_to_pep440("*"), None);
        assert_eq!(conda_spec_to_pep440(""), None);
        // build strings / alternations are conda-only: skipped
        assert_eq!(conda_spec_to_pep440("2.1.0 py312_0"), None);
        assert_eq!(conda_spec_to_pep440("1.2|1.3"), None);
    }

    #[test]
    fn build_constraints_maps_names_and_records_provenance() {
        let mut conda_deps = BTreeMap::new();
        conda_deps.insert("pytorch-gpu".into(), "==2.10.0".into());
        conda_deps.insert("Py-OpenCV".into(), "4.10.*".into());
        conda_deps.insert("python".into(), "3.12.*".into()); // skipped
        conda_deps.insert("python_abi".into(), "3.12".into()); // skipped
        conda_deps.insert("__glibc".into(), ">=2.28".into()); // skipped
        conda_deps.insert("scipy".into(), "*".into()); // unrepresentable spec

        let mut name_map = BTreeMap::new();
        name_map.insert("torch".into(), "pytorch-gpu".into());
        name_map.insert("opencv-python-headless".into(), "py-opencv".into());

        let set = build_constraints(
            &conda_deps,
            &name_map,
            &Default::default(),
            "manifest",
            "default",
        );
        assert_eq!(
            set.constraints,
            vec![
                "opencv-python-headless==4.10.*".to_string(),
                "torch==2.10.0".to_string(),
            ]
        );
        let torch = &set.provenance["torch"];
        assert_eq!(torch.constraint, "torch==2.10.0");
        assert_eq!(torch.conda_name, "pytorch-gpu");
        assert_eq!(torch.conda_version, "==2.10.0");
        assert_eq!(torch.source, "manifest");
        assert_eq!(torch.env, "default");
        // conda name with no mapping would fall back to identity; the
        // skipped ones must not appear at all.
        assert!(!set.provenance.contains_key("python"));
        assert!(!set.provenance.contains_key("scipy"));

        // provenance JSON round-trips with the spec's field names
        let json = provenance_json(&set).unwrap();
        assert!(json.contains("\"conda_name\": \"pytorch-gpu\""));
        assert!(json.contains("\"conda_version\": \"==2.10.0\""));
    }

    /// Run-38 fix: a pack with NO `retread-name-map` of its own must
    /// still map a hand-written accelerator meta-package pin
    /// (`pytorch-gpu ==2.7.0`) to the pypi name uv resolves (`torch`)
    /// via the parselmouth-backed global map -- previously the identity
    /// fallback emitted an inert `pytorch-gpu==2.7.0` constraint and uv
    /// free-picked torch 2.10, whose auto-routed `pytorch >=2.10.0,<3`
    /// emission excluded every cp311 torchaudio-2.7.0 conda build.
    #[test]
    fn build_constraints_global_map_maps_meta_package_to_pypi_name() {
        let mut conda_deps = BTreeMap::new();
        conda_deps.insert("pytorch-gpu".to_string(), "==2.7.0".to_string());
        let mut global: crate::handler::PypiToCondaMap = Default::default();
        global.insert(
            "torch".to_string(),
            vec!["pytorch".to_string(), "pytorch-gpu".to_string()],
        );
        let set = build_constraints(
            &conda_deps,
            &BTreeMap::new(),
            &global,
            "manifest",
            "consuming-envs",
        );
        assert_eq!(set.constraints, vec!["torch==2.7.0".to_string()]);
        assert_eq!(set.provenance["torch"].conda_name, "pytorch-gpu");

        // The pack's own name map still wins over the global one.
        let mut pack_map = BTreeMap::new();
        pack_map.insert("my-torch".to_string(), "pytorch-gpu".to_string());
        let set2 = build_constraints(
            &conda_deps,
            &pack_map,
            &global,
            "manifest",
            "consuming-envs",
        );
        assert_eq!(set2.constraints, vec!["my-torch==2.7.0".to_string()]);
    }

    /// The cuda-bindings incident, at the unit level: a workspace whose
    /// consuming env pins `cuda-version ==12.9` must synthesize a
    /// `cuda-bindings>=12,<13` cap (and `cuda-python`, the other tracked
    /// family) so uv never independently resolves a cuda-13-line release
    /// whose conda variant the auto-route co-install check would later
    /// have to un-route.
    #[test]
    fn cuda_family_constraints_caps_tracked_families_to_env_major() {
        let major = cuda_major_from_specs(&["==12.9".to_string()]).expect("major parses");
        assert_eq!(major, 12);
        let caps = cuda_family_constraints(major);
        assert!(
            caps.contains(&("cuda-bindings", ">=12,<13".to_string())),
            "caps: {caps:?}"
        );
        assert!(
            caps.contains(&("cuda-python", ">=12,<13".to_string())),
            "caps: {caps:?}"
        );
    }

    #[test]
    fn cuda_major_from_specs_handles_range_and_bare_forms() {
        assert_eq!(cuda_major_from_specs(&[">=12.8,<13".to_string()]), Some(12));
        assert_eq!(cuda_major_from_specs(&["12".to_string()]), Some(12));
        assert_eq!(cuda_major_from_specs(&["13.0".to_string()]), Some(13));
        assert_eq!(cuda_major_from_specs(&[]), None);
        assert_eq!(cuda_major_from_specs(&["*".to_string()]), None);
    }

    /// v4.2.0 (post mirror-solver deletion): the "prefer conda deps"
    /// guarantee, end to end at the unit level. A conda-pinned package
    /// (torch, name-mapped to conda's pytorch-gpu) must (a) participate
    /// in uv resolution as a constraint carrying conda provenance, and
    /// (b) be EXCLUDED from the emitted wheel closure so the conda
    /// package -- not a PyPI wheel -- provides it at install time.
    #[test]
    fn conda_pinned_torch_constrains_uv_and_is_excluded_from_closure() {
        // (a) conda pin -> uv constraint with provenance.
        let mut conda_deps = BTreeMap::new();
        conda_deps.insert("pytorch-gpu".into(), "==2.10.0".into());
        let mut name_map = BTreeMap::new();
        name_map.insert("torch".into(), "pytorch-gpu".into());
        let set = build_constraints(
            &conda_deps,
            &name_map,
            &Default::default(),
            "manifest",
            "default",
        );
        assert_eq!(set.constraints, vec!["torch==2.10.0".to_string()]);
        assert_eq!(set.provenance["torch"].conda_name, "pytorch-gpu");

        // (b) conda-routed name -> post-parse filter drops it from the
        // closure (wheels AND pins), the authoritative routing step.
        let pylock = r#"
lock-version = "1.0"
created-by = "uv"

[[packages]]
name = "torch"
version = "2.10.0"

[[packages.wheels]]
name = "torch-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl"
url = "https://files.pythonhosted.org/torch-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl"

[packages.wheels.hashes]
sha256 = "1111111111111111111111111111111111111111111111111111111111111111"

[[packages]]
name = "typing-extensions"
version = "4.12.0"

[[packages.wheels]]
name = "typing_extensions-4.12.0-py3-none-any.whl"
url = "https://files.pythonhosted.org/typing_extensions-4.12.0-py3-none-any.whl"

[packages.wheels.hashes]
sha256 = "3333333333333333333333333333333333333333333333333333333333333333"
"#;
        let mut exclude = BTreeSet::new();
        exclude.insert(canonical_conda_name("torch"));
        let closure =
            parse_pylock_closure(pylock, &target("3.12", "linux-64"), &exclude, "0.11.15").unwrap();
        let names: Vec<&str> = closure.wheels.iter().map(|w| w.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["typing-extensions"],
            "conda-routed torch must not ship as a PyPI wheel"
        );
        assert!(!closure.pins.contains_key("torch"));
    }

    // ---- pylock parsing ---------------------------------------------------

    const PYLOCK_FIXTURE: &str = r#"
lock-version = "1.0"
created-by = "uv"
requires-python = "==3.12.*"

[[packages]]
name = "numpy"
version = "2.1.0"

[[packages.wheels]]
name = "numpy-2.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
url = "https://files.pythonhosted.org/packages/aa/numpy-2.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
[packages.wheels.hashes]
sha256 = "1111111111111111111111111111111111111111111111111111111111111111"

[[packages.wheels]]
name = "numpy-2.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
url = "https://files.pythonhosted.org/packages/bb/numpy-2.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
[packages.wheels.hashes]
sha256 = "2222222222222222222222222222222222222222222222222222222222222222"

[[packages.wheels]]
name = "numpy-2.1.0-cp312-cp312-macosx_11_0_arm64.whl"
url = "https://files.pythonhosted.org/packages/cc/numpy-2.1.0-cp312-cp312-macosx_11_0_arm64.whl"
[packages.wheels.hashes]
sha256 = "3333333333333333333333333333333333333333333333333333333333333333"

[[packages]]
name = "typing-extensions"
version = "4.12.2"

[[packages.wheels]]
name = "typing_extensions-4.12.2-py3-none-any.whl"
url = "https://files.pythonhosted.org/packages/dd/typing_extensions-4.12.2-py3-none-any.whl"
[packages.wheels.hashes]
sha256 = "4444444444444444444444444444444444444444444444444444444444444444"

[[packages]]
name = "mujoco"
version = "3.5.0"

[[packages.wheels]]
name = "mujoco-3.5.0-cp312-cp312-manylinux_2_28_x86_64.whl"
url = "https://py.mujoco.org/mujoco-3.5.0-cp312-cp312-manylinux_2_28_x86_64.whl"
[packages.wheels.hashes]
sha256 = "5555555555555555555555555555555555555555555555555555555555555555"

[[packages]]
name = "isaaclab"
version = "2.0.0"
directory = { path = "wheels/isaaclab" }
"#;

    #[test]
    fn parse_pylock_selects_by_tag_and_filters() {
        let mut exclude = BTreeSet::new();
        exclude.insert("mujoco".to_string());
        let closure = parse_pylock_closure(
            PYLOCK_FIXTURE,
            &target("3.12", "linux-64"),
            &exclude,
            "0.11.15",
        )
        .unwrap();

        // mujoco excluded (conda-routed); isaaclab is a local source (pin
        // only); numpy + typing-extensions selected.
        let names: Vec<&str> = closure.wheels.iter().map(|w| w.name.as_str()).collect();
        assert_eq!(names, vec!["numpy", "typing-extensions"]);

        let numpy = &closure.wheels[0];
        assert_eq!(numpy.version, "2.1.0");
        assert!(matches!(numpy.origin, Origin::Index));
        // cp312 linux wheel chosen over cp311 and macosx
        assert_eq!(
            numpy.filename,
            "numpy-2.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        );
        assert_eq!(
            numpy.sha256.as_deref(),
            Some("2222222222222222222222222222222222222222222222222222222222222222")
        );
        assert!(numpy.url.as_deref().unwrap().starts_with("https://"));
        assert!(!numpy.must_ship);

        // pins include local-source packages and exclude the conda-routed one
        assert_eq!(
            closure.pins.get("isaaclab").map(String::as_str),
            Some("2.0.0")
        );
        assert_eq!(closure.pins.get("numpy").map(String::as_str), Some("2.1.0"));
        assert!(!closure.pins.contains_key("mujoco"));
        assert_eq!(closure.uv_version, "0.11.15");
    }

    /// A direct-URL wheel is recorded by `uv export` as a PEP 751
    /// `[packages.archive]`. Whether the archive carries a remote `url` (the
    /// pre-fix `name @ https://...` shape) OR a local `path` (the pre-fetched
    /// path-source shape), `parse_pylock_closure` classifies it as a local
    /// source: pin-only, NO index wheel. This equivalence is what makes the
    /// URL->path rewrite provenance-preserving for downstream consumers (the
    /// artifact + upstream URL are threaded through the materialize path, not
    /// the closure). Guards against a regression in the `is_local` predicate.
    #[test]
    fn parse_pylock_archive_url_and_path_are_both_pin_only() {
        let archive_url = r#"
[[packages]]
name = "isaacsim-extscache-kit"
version = "5.1.0"
[packages.archive]
url = "https://pypi.nvidia.com/x/isaacsim_extscache_kit-5.1.0-cp312-cp312-manylinux_2_34_x86_64.whl"
[packages.archive.hashes]
sha256 = "1111111111111111111111111111111111111111111111111111111111111111"
"#;
        let archive_path = r#"
[[packages]]
name = "isaacsim-extscache-kit"
version = "5.1.0"
[packages.archive]
path = "/store/deadbeef/isaacsim_extscache_kit-5.1.0-cp312-cp312-manylinux_2_34_x86_64.whl"
[packages.archive.hashes]
sha256 = "1111111111111111111111111111111111111111111111111111111111111111"
"#;
        for text in [archive_url, archive_path] {
            let closure =
                parse_pylock_closure(text, &target("3.12", "linux-64"), &BTreeSet::new(), "x")
                    .unwrap();
            // Pin recorded, but no index wheel emitted from the closure.
            assert_eq!(
                closure
                    .pins
                    .get("isaacsim-extscache-kit")
                    .map(String::as_str),
                Some("5.1.0"),
                "archive package must contribute a pin:\n{text}"
            );
            assert!(
                closure.wheels.is_empty(),
                "archive package must be pin-only (no index wheel):\n{text}"
            );
        }
    }

    #[test]
    fn parse_pylock_errors_on_missing_hash() {
        let text = r#"
[[packages]]
name = "foo"
version = "1.0"
[[packages.wheels]]
name = "foo-1.0-py3-none-any.whl"
url = "https://example.com/foo-1.0-py3-none-any.whl"
"#;
        let err = parse_pylock_closure(text, &target("3.12", "linux-64"), &BTreeSet::new(), "x")
            .unwrap_err();
        assert!(err.to_string().contains("no sha256"), "{err}");
    }

    #[test]
    fn parse_pylock_errors_on_sdist_only_package() {
        let text = r#"
[[packages]]
name = "gym"
version = "0.21.0"
[packages.sdist]
url = "https://files.pythonhosted.org/packages/ee/gym-0.21.0.tar.gz"
"#;
        let err = parse_pylock_closure(text, &target("3.12", "linux-64"), &BTreeSet::new(), "x")
            .unwrap_err();
        assert!(err.to_string().contains("retread-conda-deps"), "{err}");
    }

    #[test]
    fn parse_pylock_errors_when_no_compatible_wheel() {
        let text = r#"
[[packages]]
name = "foo"
version = "1.0"
[[packages.wheels]]
name = "foo-1.0-cp312-cp312-win_amd64.whl"
url = "https://example.com/foo-1.0-cp312-cp312-win_amd64.whl"
[packages.wheels.hashes]
sha256 = "6666666666666666666666666666666666666666666666666666666666666666"
"#;
        let err = parse_pylock_closure(text, &target("3.12", "linux-64"), &BTreeSet::new(), "x")
            .unwrap_err();
        assert!(err.to_string().contains("compatible"), "{err}");
    }

    // ---- conflict attribution --------------------------------------------

    #[test]
    fn attribute_conflict_names_conda_source() {
        let mut conda_deps = BTreeMap::new();
        conda_deps.insert("mujoco".to_string(), "==3.5.0".to_string());
        conda_deps.insert("numpy".to_string(), "==1.26.4".to_string());
        let set = build_constraints(
            &conda_deps,
            &BTreeMap::new(),
            &Default::default(),
            "manifest",
            "default",
        );

        let stderr = "  x No solution found when resolving dependencies:\n  \
             `-> Because dm-control depends on mujoco>=3.7 and you require mujoco==3.5.0,\n  \
                 we can conclude that your requirements are unsatisfiable.";
        let attributions = attribute_conflict(stderr, &set.provenance);
        assert_eq!(attributions.len(), 1, "{attributions:?}");
        let a = &attributions[0];
        assert_eq!(a.package, "mujoco");
        assert_eq!(a.required.as_deref(), Some(">=3.7"));
        assert_eq!(a.conflicting_constraint, "mujoco==3.5.0");
        assert_eq!(a.conda_source.conda_name, "mujoco");

        // And the rendered message carries the verbatim stderr + hint.
        let req = sample_request();
        let msg = format_lock_failure(&req, stderr, &attributions);
        assert!(msg.contains("No solution found"));
        assert!(msg.contains("retread solve"));
        assert!(msg.contains("conda package `mujoco`"));
    }

    #[test]
    fn attribute_conflict_degrades_gracefully_on_unparseable_text() {
        let mut conda_deps = BTreeMap::new();
        conda_deps.insert("torch".to_string(), "==2.10.0".to_string());
        let set = build_constraints(
            &conda_deps,
            &BTreeMap::new(),
            &Default::default(),
            "manifest",
            "default",
        );
        // Name mentioned without a parseable range: record with required=None.
        let attributions = attribute_conflict("something about torch went wrong", &set.provenance);
        assert_eq!(attributions.len(), 1);
        assert_eq!(attributions[0].required, None);
        // Name not mentioned at all: no record.
        let none = attribute_conflict("unrelated failure", &set.provenance);
        assert!(none.is_empty());

        // Project-name boundary: `torch` must not attribute `torchvision`.
        let none = attribute_conflict(
            "isaacsim-core depends on torchvision==0.26.0",
            &set.provenance,
        );
        assert!(
            none.is_empty(),
            "torch prefix-matched torchvision: {none:?}"
        );
    }

    fn workspace_attribution(
        package: &str,
        required: Option<&str>,
        fact_version: &str,
        source: &str,
    ) -> ConflictAttribution {
        ConflictAttribution {
            package: package.to_string(),
            required: required.map(str::to_string),
            conflicting_constraint: format!("{package}=={fact_version}"),
            conda_source: ConstraintProvenance {
                constraint: format!("{package}=={fact_version}"),
                conda_name: package.to_string(),
                conda_version: format!("=={fact_version}"),
                source: source.to_string(),
                env: "precise-consuming-envs".to_string(),
                advisory: false,
            },
        }
    }

    #[test]
    fn workspace_fact_exact_conflict_requests_graph_wide_override() {
        let mut req = sample_request();
        req.dependencies = vec!["isaacsim-core==6.0.0.1".to_string()];
        let attribution =
            workspace_attribution("torch", Some("==2.11.0"), "2.10.0", "workspace-solved");
        req.constraints = ConstraintSet {
            constraints: vec![attribution.conflicting_constraint.clone()],
            provenance: BTreeMap::from([("torch".to_string(), attribution.conda_source.clone())]),
            auto_route_constraint_indices: BTreeSet::new(),
        };
        let stderr = "  x No solution found when resolving dependencies:\n  \
             `-> Because isaacsim-core==6.0.0.1 depends on torch==2.11.0 and \
                 you require torch==2.10.0, your requirements are unsatisfiable.";
        let attributions = attribute_conflict(stderr, &req.constraints.provenance);
        let needed = workspace_fact_override_needed(&req, &attributions, "original uv error")
            .expect("a differing upstream exact pin must request a workspace-fact override");
        assert_eq!(
            needed.fact,
            WorkspaceFactOverride {
                pypi_name: "torch".to_string(),
                version: "2.10.0".to_string(),
                drop: false,
            }
        );
        assert_eq!(needed.upstream_pin, "==2.11.0");
        assert_eq!(needed.original_error, "original uv error");
    }

    #[test]
    fn workspace_fact_override_classifier_abstains_without_proof_or_authority() {
        let mut req = sample_request();
        req.dependencies = vec!["isaacsim-core==6.0.0.1".to_string()];
        let classify = |req: &UvClosureRequest, attribution: ConflictAttribution| {
            workspace_fact_override_needed(req, &[attribution], "error")
        };

        for required in [
            Some("==2.10.0"), // identical
            Some("==2.10"),   // PEP 440-equivalent
            Some(">=2.11.0"), // range
            Some("==2.11.*"), // wildcard equality
            Some("===not-a-version"),
            None,
        ] {
            assert!(
                classify(
                    &req,
                    workspace_attribution("torch", required, "2.10.0", "workspace-solved")
                )
                .is_none(),
                "classifier must abstain for {required:?}"
            );
        }

        // PEP 440 equality without a local segment accepts a local fact on
        // the same public version, so this is not a contradiction.
        assert!(
            classify(
                &req,
                workspace_attribution(
                    "torch",
                    Some("==2.10.0"),
                    "2.10.0+cu129",
                    "workspace-solved"
                )
            )
            .is_none()
        );
        for source in ["manifest", "cuda-major-table", "auto-route"] {
            assert!(
                classify(
                    &req,
                    workspace_attribution("torch", Some("==2.11.0"), "2.10.0", source)
                )
                .is_none(),
                "non-Rule-1 source {source} must not authorize an override"
            );
        }

        let mut direct_root = req;
        direct_root.dependencies.push("torch==2.11.0".to_string());
        assert!(
            classify(
                &direct_root,
                workspace_attribution("torch", Some("==2.11.0"), "2.10.0", "workspace-solved")
            )
            .is_none(),
            "an explicit direct root pin is user intent, not an upstream pin"
        );
    }

    // ---- Rule 1: conda-owned drop on an unsatisfiable PyPI requirement -----

    /// A `workspace-solved` attribution whose conda owner (`conda_name`) may
    /// differ from the PyPI name -- the retread-name-map case.
    fn conda_owned_attribution(
        package: &str,
        conda_name: &str,
        required: Option<&str>,
        fact_version: &str,
    ) -> ConflictAttribution {
        ConflictAttribution {
            package: package.to_string(),
            required: required.map(str::to_string),
            conflicting_constraint: format!("{package}=={fact_version}"),
            conda_source: ConstraintProvenance {
                constraint: format!("{package}=={fact_version}"),
                conda_name: conda_name.to_string(),
                conda_version: format!("=={fact_version}"),
                source: "workspace-solved".to_string(),
                env: "precise-consuming-envs".to_string(),
                advisory: false,
            },
        }
    }

    /// The exact `sage` regression: a TRANSITIVE wheel (`isaacsim-core`) pins
    /// `opencv-python==4.11.0`, a version that does not exist on PyPI (its
    /// releases are `4.11.0.86`-style). The workspace conda-OWNS it through the
    /// retread-name-map edge `opencv-python -> py-opencv`. Rule 3 cannot fire
    /// (the upstream pin equals the conda-derived constraint, so `required` is
    /// None), so Rule 1 must drop it -- graph-wide, reaching the transitive
    /// `Requires-Dist` pin, not just a direct root.
    #[test]
    fn workspace_owned_impossible_transitive_pin_requests_drop() {
        let mut req = sample_request();
        req.dependencies = vec!["isaacsim-core==4.2.0.2".to_string()];
        let prov = ConstraintProvenance {
            constraint: "opencv-python==4.11.0".to_string(),
            conda_name: "py-opencv".to_string(),
            conda_version: "==4.11.0".to_string(),
            source: "workspace-solved".to_string(),
            env: "precise-consuming-envs".to_string(),
            advisory: false,
        };
        req.constraints = ConstraintSet {
            constraints: vec!["opencv-python==4.11.0".to_string()],
            provenance: BTreeMap::from([("opencv-python".to_string(), prov)]),
            auto_route_constraint_indices: BTreeSet::new(),
        };
        let stderr = "  x No solution found when resolving dependencies:\n  \
             `-> Because there is no version of opencv-python==4.11.0 and \
                 isaacsim-core==4.2.0.2 depends on opencv-python==4.11.0, we can \
                 conclude that isaacsim-core==4.2.0.2 cannot be used.";
        let attributions = attribute_conflict(stderr, &req.constraints.provenance);
        // Faithful to the surfaced diagnostic ("is named in the conflict"):
        // no differing upstream spec, so `required` is None.
        assert_eq!(attributions.len(), 1, "{attributions:?}");
        assert_eq!(attributions[0].required, None);
        // Rule 3 correctly declines: there is no exact-pin CONTRADICTION.
        assert!(
            workspace_fact_override_needed(&req, &attributions, "err").is_none(),
            "no contradiction, so Rule 3 must not fire"
        );

        let needed = workspace_owned_drop_needed(&req, &attributions, "uv lock failed")
            .expect("conda-owned opencv-python with an impossible PyPI pin must request a drop");
        assert_eq!(
            needed.fact,
            WorkspaceFactOverride {
                pypi_name: "opencv-python".to_string(),
                version: "4.11.0".to_string(),
                drop: true,
            }
        );
        // The emitted uv requirement removes it from the graph entirely.
        assert_eq!(
            needed.fact.requirement(),
            format!("opencv-python ; {DROP_MARKER}")
        );
    }

    /// The drop bridges the name-map for the DIRECT-name shape too, but a
    /// first-party direct ROOT is the user's explicit intent and is never
    /// dropped out from under them.
    #[test]
    fn workspace_owned_drop_declines_for_first_party_direct_root() {
        let mut req = sample_request();
        req.dependencies = vec!["opencv-python==4.11.0".to_string()];
        let attribution = conda_owned_attribution("opencv-python", "py-opencv", None, "4.11.0");
        assert!(
            workspace_owned_drop_needed(&req, &[attribution], "err").is_none(),
            "a first-party direct root is user intent, never a conda-owned drop"
        );
    }

    /// A dep that is NOT conda-owned (no workspace-solved fact) with an
    /// impossible pin still errors clearly -- the drop path must not fire and
    /// the diagnostic is preserved.
    #[test]
    fn workspace_owned_drop_declines_without_ownership_authority() {
        let mut req = sample_request();
        req.dependencies = vec!["isaacsim-core==4.2.0.2".to_string()];
        // A non-`workspace-solved` provenance is a routing alias, not fact
        // authority, and cannot authorize a drop.
        for source in ["manifest", "cuda-major-table", "auto-route", "deps-from-conda-advisory"] {
            let attribution = workspace_attribution("opencv-python", None, "4.11.0", source);
            assert!(
                workspace_owned_drop_needed(&req, &[attribution], "err").is_none(),
                "source `{source}` is not conda-ownership authority"
            );
        }
        // A name uv mentions that is in NO provenance table yields no
        // attribution at all -> the clear PyPI-unsatisfiable error survives.
        let stderr = "  x No solution found: there is no version of totally-unowned==9.9.9";
        let attributions = attribute_conflict(stderr, &req.constraints.provenance);
        assert!(attributions.is_empty(), "{attributions:?}");
        assert!(workspace_owned_drop_needed(&req, &attributions, "err").is_none());
    }

    /// The co-solvability guard is preserved: a DIFFERING upstream requirement
    /// the conda owner's version does NOT satisfy is a genuine version conflict
    /// (installing the conda version would violate the wheel), so Rule 1
    /// abstains. A differing requirement the conda version DOES satisfy is a
    /// legitimate conda-owned drop (PyPI can't supply it, conda can).
    #[test]
    fn workspace_owned_drop_respects_upstream_satisfiability() {
        let mut req = sample_request();
        req.dependencies = vec!["isaacsim-core==4.2.0.2".to_string()];

        // conda `pytorch-gpu==2.10.0` (-> PyPI `torch`) cannot satisfy an
        // upstream `>=2.11` requirement: dropping would install a forbidden
        // version, so abstain (leave it to error / a real reconciliation).
        let conflicting = conda_owned_attribution("torch", "pytorch-gpu", Some(">=2.11"), "2.10.0");
        assert!(
            workspace_owned_drop_needed(&req, &[conflicting], "err").is_none(),
            "a conda version that violates the upstream range is a real conflict"
        );

        // A `>=2.0` requirement IS satisfied by the conda version 2.10.0: PyPI
        // can't supply it but conda can, so this is a legitimate drop.
        let satisfiable = conda_owned_attribution("torch", "pytorch-gpu", Some(">=2.0"), "2.10.0");
        let needed = workspace_owned_drop_needed(&req, &[satisfiable], "err")
            .expect("a satisfied-by-conda requirement is a legitimate conda-owned drop");
        assert!(needed.fact.drop);
        assert_eq!(needed.fact.pypi_name, "torch");
    }

    // ---- auto-route (spec-uv-restructure M2) -------------------------------

    use std::sync::{Arc, Mutex};

    /// The drop signal flows through the SAME retry wrapper as Rule 3: the
    /// first solve fails asking for the drop, the retry injects the
    /// unmatchable-marker override, the second solve succeeds, and the drop is
    /// recorded as a learned (persistable) fact.
    #[tokio::test]
    async fn workspace_owned_drop_retry_injects_unmatchable_override() {
        let seen = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
        let drop_line = format!("opencv-python ; {DROP_MARKER}");
        let solve = {
            let seen = Arc::clone(&seen);
            let drop_line = drop_line.clone();
            move |req: UvClosureRequest| {
                let seen = Arc::clone(&seen);
                let drop_line = drop_line.clone();
                Box::pin(async move {
                    seen.lock().unwrap().push(req.overrides.clone());
                    if !req.overrides.iter().any(|o| o == &drop_line) {
                        return Err(anyhow::Error::new(WorkspaceFactOverrideNeeded {
                            fact: WorkspaceFactOverride {
                                pypi_name: "opencv-python".to_string(),
                                version: "4.11.0".to_string(),
                                drop: true,
                            },
                            upstream_pin: "opencv-python==4.11.0".to_string(),
                            original_error: "opencv-python unsatisfiable on PyPI".to_string(),
                        }));
                    }
                    Ok(UvClosure {
                        wheels: vec![],
                        pins: BTreeMap::new(),
                        uv_version: "test".to_string(),
                        auto_routed: vec![],
                        auto_dropped: BTreeSet::new(),
                        effective_input_requirements: None,
                    })
                }) as futures::future::BoxFuture<'static, Result<UvClosure>>
            }
        };
        let learned = Arc::new(Mutex::new(Vec::new()));
        let mut solve = with_workspace_fact_overrides(solve, Arc::clone(&learned));
        let _ = solve(sample_request()).await.unwrap();

        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 2, "one failed attempt, one healed retry");
        assert!(
            !calls[0].iter().any(|o| o == &drop_line),
            "first attempt is pre-drop"
        );
        assert!(
            calls[1].iter().any(|o| o == &drop_line),
            "retry carries the unmatchable-marker drop override"
        );
        let facts = learned.lock().unwrap();
        assert_eq!(facts.len(), 1);
        assert!(facts[0].drop, "the drop is recorded as a learned/persistable fact");
        assert_eq!(facts[0].pypi_name, "opencv-python");
    }

    fn workspace_fact_needed_error(name: &str, version: &str, upstream_pin: &str) -> anyhow::Error {
        anyhow::Error::new(WorkspaceFactOverrideNeeded {
            fact: WorkspaceFactOverride {
                pypi_name: name.to_string(),
                version: version.to_string(),
                drop: false,
            },
            upstream_pin: upstream_pin.to_string(),
            original_error: format!("{name} exact-pin conflict"),
        })
    }

    #[tokio::test]
    async fn workspace_fact_override_retry_accumulates_torch_family() {
        let calls = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
        let projects = Arc::new(Mutex::new(Vec::<String>::new()));
        let solve = {
            let calls = Arc::clone(&calls);
            let projects = Arc::clone(&projects);
            move |req: UvClosureRequest| {
                let calls = Arc::clone(&calls);
                let projects = Arc::clone(&projects);
                Box::pin(async move {
                    calls.lock().unwrap().push(req.overrides.clone());
                    for (name, fact, upstream) in [
                        ("torch", "2.10.0", "==2.11.0"),
                        ("torchvision", "0.25.0", "==0.26.0"),
                        ("torchaudio", "2.10.0", "==2.11.0"),
                    ] {
                        let line = format!("{name}=={fact}");
                        if !req.overrides.iter().any(|candidate| candidate == &line) {
                            return Err(workspace_fact_needed_error(name, fact, upstream));
                        }
                    }
                    projects.lock().unwrap().push(synthesize_pyproject(&req));
                    Ok(UvClosure {
                        wheels: vec![],
                        pins: BTreeMap::from([
                            ("torch".to_string(), "2.10.0".to_string()),
                            ("torchvision".to_string(), "0.25.0".to_string()),
                            ("torchaudio".to_string(), "2.10.0".to_string()),
                        ]),
                        uv_version: "test".to_string(),
                        auto_routed: vec![],
                        auto_dropped: BTreeSet::new(),
                        effective_input_requirements: None,
                    })
                }) as futures::future::BoxFuture<'static, Result<UvClosure>>
            }
        };
        let learned = Arc::new(Mutex::new(Vec::new()));
        let mut solve = with_workspace_fact_overrides(solve, Arc::clone(&learned));
        let mut req = sample_request();
        req.dependencies = vec!["isaacsim-core==6.0.0.1".to_string()];

        let closure = solve(req.clone()).await.unwrap();
        assert_eq!(closure.pins["torch"], "2.10.0");
        assert_eq!(closure.pins["torchvision"], "0.25.0");
        assert_eq!(closure.pins["torchaudio"], "2.10.0");
        let effective_inputs = closure.effective_input_requirements.as_ref().unwrap();
        assert!(
            effective_inputs["torch"].iter().any(|input| {
                input.specifiers == "==2.10.0"
                    && input.source == "uv override requirement `torch==2.10.0`"
                    && input.role == AutoRouteInputRole::Override
            }),
            "the closure must retain the learned override from the actual successful request"
        );
        assert!(
            effective_inputs["torch"]
                .iter()
                .all(|input| input.role != AutoRouteInputRole::Requirement),
            "the learned override must replace ordinary requirements while retaining constraints"
        );

        let trio = |lines: &[String]| -> BTreeSet<String> {
            lines
                .iter()
                .filter(|line| {
                    matches!(
                        override_name(line).as_deref(),
                        Some("torch" | "torchvision" | "torchaudio")
                    )
                })
                .cloned()
                .collect()
        };
        let calls_after_first = calls.lock().unwrap().clone();
        assert_eq!(calls_after_first.len(), 4);
        assert_eq!(trio(&calls_after_first[0]), BTreeSet::new());
        assert_eq!(
            trio(&calls_after_first[1]),
            BTreeSet::from(["torch==2.10.0".to_string()])
        );
        assert_eq!(
            trio(&calls_after_first[2]),
            BTreeSet::from([
                "torch==2.10.0".to_string(),
                "torchvision==0.25.0".to_string(),
            ])
        );
        let all_three = BTreeSet::from([
            "torch==2.10.0".to_string(),
            "torchvision==0.25.0".to_string(),
            "torchaudio==2.10.0".to_string(),
        ]);
        assert_eq!(trio(&calls_after_first[3]), all_three);

        let project = projects.lock().unwrap()[0].clone();
        let parsed: toml::Value = toml::from_str(&project).unwrap();
        let rendered = parsed["tool"]["uv"]["override-dependencies"]
            .as_array()
            .unwrap();
        for line in ["torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0"] {
            assert_eq!(
                rendered
                    .iter()
                    .filter(|value| value.as_str() == Some(line))
                    .count(),
                1,
                "{line} must be emitted exactly once under override-dependencies"
            );
        }

        // A later auto-route/sdist outer relock starts from a fresh base
        // request; the shared ledger must seed all facts before its first raw
        // solve, with no rediscovery calls.
        solve(req).await.unwrap();
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 5);
        assert_eq!(trio(&calls[4]), all_three);
        assert_eq!(learned.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn workspace_fact_override_retry_respects_existing_override() {
        for existing in ["torch>=2.9".to_string(), format!("torch ; {DROP_MARKER}")] {
            let calls = Arc::new(Mutex::new(0usize));
            let raw = {
                let calls = Arc::clone(&calls);
                move |_req: UvClosureRequest| {
                    *calls.lock().unwrap() += 1;
                    Box::pin(async {
                        Err(workspace_fact_needed_error("torch", "2.10.0", "==2.11.0"))
                    }) as futures::future::BoxFuture<'static, Result<UvClosure>>
                }
            };
            let learned = Arc::new(Mutex::new(Vec::new()));
            let mut solve = with_workspace_fact_overrides(raw, Arc::clone(&learned));
            let mut req = sample_request();
            req.overrides.push(existing);
            let error = solve(req).await.unwrap_err().to_string();
            assert!(error.contains("torch exact-pin conflict"), "{error}");
            assert_eq!(*calls.lock().unwrap(), 1, "must not retry over user intent");
            assert!(learned.lock().unwrap().is_empty());
        }
    }

    /// Canned repodata: conda name -> hit. Mirrors how a real channel
    /// would answer `find_route(name, ==version)`.
    fn canned_probe(
        hits: BTreeMap<String, RouteProbeHit>,
    ) -> impl Fn(String, String) -> futures::future::BoxFuture<'static, Option<RouteProbeHit>> {
        move |name: String, _spec: String| {
            let hit = hits.get(&name).cloned();
            Box::pin(async move { hit })
        }
    }

    /// Canned solve: parse the shared pylock fixture with the request's
    /// CURRENT `no_emit_packages` as the exclude set (exactly what the
    /// real compute_closure does post-export), recording each request.
    fn canned_solve(
        calls: Arc<Mutex<Vec<UvClosureRequest>>>,
    ) -> impl FnMut(UvClosureRequest) -> futures::future::BoxFuture<'static, Result<UvClosure>>
    {
        move |r: UvClosureRequest| {
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                calls.lock().unwrap().push(r.clone());
                let exclude: BTreeSet<String> = r
                    .no_emit_packages
                    .iter()
                    .map(|n| canonical_conda_name(n))
                    .collect();
                parse_pylock_closure(
                    PYLOCK_FIXTURE,
                    &WheelTarget {
                        python_version: "3.12".into(),
                        conda_subdir: "linux-64".into(),
                        max_glibc: None,
                    },
                    &exclude,
                    "0.11.15",
                )
            })
        }
    }

    fn auto_route_req() -> UvClosureRequest {
        UvClosureRequest {
            bundle: "tiny-pack".into(),
            python_version: "3.12".into(),
            conda_subdir: "linux-64".into(),
            dependencies: vec!["mujoco==3.5.0".into()],
            constraints: ConstraintSet::default(),
            overrides: vec![],
            no_emit_packages: vec![],
            index_urls: vec!["https://pypi.org/simple/".into()],
            built_wheel_sources: BTreeMap::new(),
            explicit_pins: BTreeMap::new(),
            offline: false,
        }
    }

    fn auto_route_opts() -> AutoRouteOptions {
        AutoRouteOptions {
            enabled: true,
            keep_pypi: BTreeSet::new(),
            force_conda: BTreeSet::new(),
            // mujoco is the bundle's own root entry: never routed.
            protected: BTreeSet::from(["mujoco".to_string()]),
            name_map: BTreeMap::new(),
            abi_anchor_pins: BTreeMap::new(),
            workspace_conda_versions: BTreeMap::new(),
            ..Default::default()
        }
    }

    /// The core loop: numpy exists on a channel at the resolved version
    /// -> excluded from the closure, pinned as a constraint, recorded as
    /// auto-routed; the loop re-locks once and reaches fixpoint.
    #[tokio::test]
    async fn auto_route_routes_hit_and_reaches_fixpoint() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut req = auto_route_req();
        req.dependencies.push("numpy>=2,<3".to_string());
        let mut hits = BTreeMap::new();
        hits.insert(
            "numpy".to_string(),
            RouteProbeHit {
                conda_version: "2.1.0".into(),
                channel: "https://conda.anaconda.org/conda-forge/linux-64".into(),
                depends: Vec::new(),
            },
        );
        let closure = auto_route_fixpoint(
            &req,
            &auto_route_opts(),
            canned_solve(Arc::clone(&calls)),
            canned_probe(hits),
        )
        .await
        .unwrap();

        // numpy moved to conda; typing-extensions (no conda hit) and the
        // protected root (mujoco) stay.
        let names: Vec<&str> = closure.wheels.iter().map(|w| w.name.as_str()).collect();
        assert_eq!(names, vec!["typing-extensions", "mujoco"]);
        assert_eq!(closure.auto_routed.len(), 1);
        let r = &closure.auto_routed[0];
        assert_eq!(r.pypi_name, "numpy");
        assert_eq!(r.conda_name, "numpy");
        assert_eq!(r.pypi_version, "2.1.0");
        assert_eq!(r.conda_version, "2.1.0");
        assert!(r.channel.contains("conda-forge"));
        assert_eq!(
            r.input_requirements,
            vec![AutoRouteInputRequirement {
                specifiers: uv_pep508::uv_pep440::VersionSpecifiers::from_str(">=2,<3")
                    .unwrap()
                    .to_string(),
                source: "uv root requirement `numpy>=2,<3`".to_string(),
                role: AutoRouteInputRole::Requirement,
            }]
        );

        // Initial lock + one re-lock after routing; round 2 found
        // nothing new (no third solve).
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        // The re-lock saw the exclusion AND the exact-version constraint
        // with auto-route provenance.
        let relock = &calls[1];
        assert!(relock.no_emit_packages.contains(&"numpy".to_string()));
        assert!(
            relock
                .constraints
                .constraints
                .contains(&"numpy==2.1.0".to_string())
        );
        let prov = &relock.constraints.provenance["numpy"];
        assert_eq!(prov.source, "auto-route");
        assert_eq!(prov.conda_name, "numpy");
        assert_eq!(prov.conda_version, "==2.1.0");
    }

    #[tokio::test]
    async fn workspace_owned_validated_route_becomes_auto_drop() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let hits = BTreeMap::from([(
            "numpy".to_string(),
            RouteProbeHit {
                conda_version: "2.1.0".into(),
                channel: "https://conda.anaconda.org/conda-forge/linux-64".into(),
                depends: Vec::new(),
            },
        )]);
        let mut opts = auto_route_opts();
        opts.route_policy = crate::config::RoutePolicy::PreferCondaValidated;
        opts.workspace_owned.insert("numpy".to_string());

        let closure = auto_route_fixpoint_checked(
            &auto_route_req(),
            &opts,
            canned_solve(calls),
            canned_probe(hits),
            canned_co_solve(Vec::new()),
        )
        .await
        .unwrap();

        assert_eq!(closure.auto_routed.len(), 1);
        assert_eq!(
            closure.auto_dropped,
            BTreeSet::from(["numpy".to_string()]),
            "only a successfully probed and co-solvable owned route may be dropped"
        );
    }

    /// `keep-pypi` names are never routed (and never probed).
    #[tokio::test]
    async fn auto_route_respects_keep_pypi() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut hits = BTreeMap::new();
        hits.insert(
            "numpy".to_string(),
            RouteProbeHit {
                conda_version: "2.1.0".into(),
                channel: "c/linux-64".into(),
                depends: Vec::new(),
            },
        );
        let mut opts = auto_route_opts();
        opts.keep_pypi.insert("numpy".to_string());
        let closure = auto_route_fixpoint(
            &auto_route_req(),
            &opts,
            canned_solve(Arc::clone(&calls)),
            canned_probe(hits),
        )
        .await
        .unwrap();
        assert!(closure.auto_routed.is_empty());
        assert!(closure.wheels.iter().any(|w| w.name == "numpy"));
        // Fixpoint on round 1 with zero routes: no re-lock.
        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    /// Root entries (protected) are never routed even on a conda hit.
    #[tokio::test]
    async fn auto_route_never_routes_protected_roots() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut hits = BTreeMap::new();
        hits.insert(
            "mujoco".to_string(),
            RouteProbeHit {
                conda_version: "3.5.0".into(),
                channel: "c/linux-64".into(),
                depends: Vec::new(),
            },
        );
        let closure = auto_route_fixpoint(
            &auto_route_req(),
            &auto_route_opts(),
            canned_solve(Arc::clone(&calls)),
            canned_probe(hits),
        )
        .await
        .unwrap();
        assert!(closure.auto_routed.is_empty());
        assert!(closure.wheels.iter().any(|w| w.name == "mujoco"));
        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    /// `auto-route = false`: one solve, no probing effects, empty record.
    #[tokio::test]
    async fn auto_route_disabled_is_single_solve() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut hits = BTreeMap::new();
        hits.insert(
            "numpy".to_string(),
            RouteProbeHit {
                conda_version: "2.1.0".into(),
                channel: "c/linux-64".into(),
                depends: Vec::new(),
            },
        );
        let mut opts = auto_route_opts();
        opts.enabled = false;
        let closure = auto_route_fixpoint(
            &auto_route_req(),
            &opts,
            canned_solve(Arc::clone(&calls)),
            canned_probe(hits),
        )
        .await
        .unwrap();
        assert!(closure.auto_routed.is_empty());
        assert!(closure.wheels.iter().any(|w| w.name == "numpy"));
        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    /// The name-map is honored: `torch` routes via conda's `pytorch`.
    #[tokio::test]
    async fn auto_route_maps_pypi_to_conda_names() {
        let pylock = r#"
[[packages]]
name = "torch"
version = "2.10.0"
[[packages.wheels]]
name = "torch-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl"
url = "https://files.pythonhosted.org/torch-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl"
[packages.wheels.hashes]
sha256 = "1111111111111111111111111111111111111111111111111111111111111111"
"#;
        let calls = Arc::new(Mutex::new(Vec::<UvClosureRequest>::new()));
        let solve = {
            let calls = Arc::clone(&calls);
            move |r: UvClosureRequest| {
                let calls = Arc::clone(&calls);
                Box::pin(async move {
                    calls.lock().unwrap().push(r.clone());
                    let exclude: BTreeSet<String> = r
                        .no_emit_packages
                        .iter()
                        .map(|n| canonical_conda_name(n))
                        .collect();
                    parse_pylock_closure(pylock, &target("3.12", "linux-64"), &exclude, "0.11.15")
                }) as futures::future::BoxFuture<'static, Result<UvClosure>>
            }
        };
        let mut hits = BTreeMap::new();
        hits.insert(
            "pytorch".to_string(),
            RouteProbeHit {
                conda_version: "2.10.0".into(),
                channel: "c/linux-64".into(),
                depends: Vec::new(),
            },
        );
        let mut opts = auto_route_opts();
        opts.name_map
            .insert("torch".to_string(), "pytorch".to_string());
        let closure = auto_route_fixpoint(&auto_route_req(), &opts, solve, canned_probe(hits))
            .await
            .unwrap();
        assert_eq!(closure.auto_routed.len(), 1);
        assert_eq!(closure.auto_routed[0].pypi_name, "torch");
        assert_eq!(closure.auto_routed[0].conda_name, "pytorch");
        assert!(closure.wheels.is_empty(), "torch left the closure");
        // Constraint pins the PYPI name (uv-side), not the conda name.
        assert!(
            calls.lock().unwrap()[1]
                .constraints
                .constraints
                .contains(&"torch==2.10.0".to_string())
        );
    }

    /// Termination guard: a pathological solve that grows a NEW routable
    /// wheel every round must stop at AUTO_ROUTE_MAX_ROUNDS re-locks.
    #[tokio::test]
    async fn auto_route_terminates_at_round_cap() {
        let n_solves = Arc::new(Mutex::new(0usize));
        let solve = {
            let n_solves = Arc::clone(&n_solves);
            move |_r: UvClosureRequest| {
                let n_solves = Arc::clone(&n_solves);
                Box::pin(async move {
                    let mut n = n_solves.lock().unwrap();
                    *n += 1;
                    let round_id = *n;
                    // Every solve invents a fresh package, so every sweep
                    // finds something new to route.
                    Ok(UvClosure {
                        wheels: vec![LockWheel {
                            name: format!("gen{round_id}"),
                            version: "1.0".into(),
                            origin: Origin::Index,
                            filename: format!("gen{round_id}-1.0-py3-none-any.whl"),
                            url: Some("https://example.com/x.whl".into()),
                            sha256: Some("00".repeat(32)),
                            requires_dist: vec![],
                            must_ship: false,
                            upstream_url: None,
                            git_source: None,
                            sdist_source: None,
                        }],
                        pins: BTreeMap::new(),
                        uv_version: "0.11.15".into(),
                        auto_routed: vec![],
                        auto_dropped: BTreeSet::new(),
                        effective_input_requirements: None,
                    })
                }) as futures::future::BoxFuture<'static, Result<UvClosure>>
            }
        };
        // Probe hits EVERYTHING.
        let probe = |name: String, _spec: String| {
            Box::pin(async move {
                Some(RouteProbeHit {
                    conda_version: "1.0".into(),
                    channel: format!("c/{name}"),
                    depends: Vec::new(),
                })
            }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let closure = auto_route_fixpoint(&auto_route_req(), &auto_route_opts(), solve, probe)
            .await
            .unwrap();
        // Initial solve + exactly AUTO_ROUTE_MAX_ROUNDS re-locks.
        assert_eq!(*n_solves.lock().unwrap(), 1 + AUTO_ROUTE_MAX_ROUNDS);
        assert_eq!(closure.auto_routed.len(), AUTO_ROUTE_MAX_ROUNDS);
    }

    /// Within one round, every candidate's repodata lookup is issued
    /// CONCURRENTLY (16-way `buffer_unordered`), and the whole batch is
    /// collected before any re-lock. Six candidates whose probes each
    /// park on a timer must overlap; a serial sweep would show
    /// max-in-flight == 1.
    #[tokio::test]
    async fn auto_route_probes_run_concurrently_within_a_round() {
        let n_solves = Arc::new(Mutex::new(0usize));
        let solve = {
            let n_solves = Arc::clone(&n_solves);
            move |_r: UvClosureRequest| {
                let n_solves = Arc::clone(&n_solves);
                Box::pin(async move {
                    *n_solves.lock().unwrap() += 1;
                    let wheels = (0..6)
                        .map(|i| LockWheel {
                            name: format!("p{i}"),
                            version: "1.0".into(),
                            origin: Origin::Index,
                            filename: format!("p{i}-1.0-py3-none-any.whl"),
                            url: Some("https://example.com/x.whl".into()),
                            sha256: Some("00".repeat(32)),
                            requires_dist: vec![],
                            must_ship: false,
                            upstream_url: None,
                            git_source: None,
                            sdist_source: None,
                        })
                        .collect();
                    Ok(UvClosure {
                        wheels,
                        pins: BTreeMap::new(),
                        uv_version: "0.11.15".into(),
                        auto_routed: vec![],
                        auto_dropped: BTreeSet::new(),
                        effective_input_requirements: None,
                    })
                }) as futures::future::BoxFuture<'static, Result<UvClosure>>
            }
        };
        // (in_flight, max_in_flight, total_calls)
        let gauge = Arc::new(Mutex::new((0usize, 0usize, 0usize)));
        let probe = {
            let gauge = Arc::clone(&gauge);
            move |_name: String, _spec: String| {
                let gauge = Arc::clone(&gauge);
                Box::pin(async move {
                    {
                        let mut g = gauge.lock().unwrap();
                        g.0 += 1;
                        g.1 = g.1.max(g.0);
                        g.2 += 1;
                    }
                    // Park so overlapping probes are observable.
                    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                    gauge.lock().unwrap().0 -= 1;
                    None // no hits -> single round, no re-lock
                }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
            }
        };
        let closure = auto_route_fixpoint(&auto_route_req(), &auto_route_opts(), solve, probe)
            .await
            .unwrap();
        assert!(closure.auto_routed.is_empty());
        let (_, max_in_flight, total) = *gauge.lock().unwrap();
        assert_eq!(total, 6, "every candidate probed exactly once");
        assert!(
            max_in_flight >= 2,
            "probes must overlap within a round (max in-flight {max_in_flight})"
        );
        // The whole batch completed with NO intervening re-lock.
        assert_eq!(*n_solves.lock().unwrap(), 1);
    }

    /// The pure planner: an already-excluded (`retread-conda-deps`) name
    /// is not re-routed, and identity fallback applies without a map.
    #[test]
    fn plan_auto_route_round_skips_already_excluded() {
        let mut req = auto_route_req();
        req.no_emit_packages.push("numpy".into());
        let exclude: BTreeSet<String> = req
            .no_emit_packages
            .iter()
            .map(|n| canonical_conda_name(n))
            .collect();
        let closure =
            parse_pylock_closure(PYLOCK_FIXTURE, &target("3.12", "linux-64"), &exclude, "x")
                .unwrap();
        let mut hits = BTreeMap::new();
        for n in ["numpy", "typing-extensions"] {
            hits.insert(
                n.to_string(),
                RouteProbeHit {
                    conda_version: "1".into(),
                    channel: "c/linux-64".into(),
                    depends: Vec::new(),
                },
            );
        }
        let routes = plan_auto_route_round(&closure, &req, &auto_route_opts(), &[], &hits).unwrap();
        let names: Vec<&str> = routes.iter().map(|r| r.pypi_name.as_str()).collect();
        assert_eq!(names, vec!["typing-extensions"]);
    }

    /// v4.6 Part A: the routing-policy decision function. Minimal admits
    /// only the ABI/binary whitelist (either name side), route-include
    /// entries, and force-conda entries; aggressive is legacy admit-all.
    #[test]
    fn route_policy_minimal_admits_only_whitelist_includes_and_force_conda() {
        let mut opts = AutoRouteOptions {
            route_policy: crate::config::RoutePolicy::Minimal,
            ..Default::default()
        };
        for (pypi, conda) in [
            ("python", "python"),
            ("torch", "pytorch-gpu"),
            ("pytorch", "pytorch"),
            ("torchvision", "torchvision"),
            ("torchaudio", "torchaudio"),
            ("cuda-version", "cuda-version"),
            ("cudatoolkit", "cudatoolkit"),
            // conda-side spelling alone suffices (name-map edge).
            ("some-pypi-alias", "cuda-toolkit"),
        ] {
            assert!(
                route_policy_admits(pypi, conda, &opts),
                "whitelist must admit ({pypi}, {conda})"
            );
        }
        for n in [
            "wrapt",
            "fsspec",
            "huggingface-hub",
            "grpcio",
            "cycler",
            "dm-tree",
            "moviepy",
            "matplotlib",
            "tensordict",
        ] {
            assert!(!route_policy_admits(n, n, &opts), "minimal must refuse {n}");
        }
        opts.route_include.insert("grpcio".to_string());
        assert!(route_policy_admits("grpcio", "grpcio", &opts));
        opts.force_conda.insert("mujoco".to_string());
        assert!(route_policy_admits("mujoco", "mujoco", &opts));
    }

    #[test]
    fn route_policy_aggressive_is_legacy_admit_all() {
        let opts = AutoRouteOptions::default();
        assert_eq!(
            opts.route_policy,
            crate::config::RoutePolicy::Aggressive,
            "bare test/probe construction keeps legacy sweep semantics"
        );
        assert!(route_policy_admits("wrapt", "wrapt", &opts));
        assert!(route_policy_admits(
            "anything-at-all",
            "anything-at-all",
            &opts
        ));
    }

    /// v4.6 Part A at the sweep level: under Minimal, a conda-available
    /// non-whitelisted wheel is NOT routed (ships as a wheel) and is not
    /// even probed; whitelisted/include-listed candidates still route.
    #[test]
    fn plan_auto_route_round_minimal_policy_refuses_non_whitelisted() {
        let req = auto_route_req();
        let closure = parse_pylock_closure(
            PYLOCK_FIXTURE,
            &target("3.12", "linux-64"),
            &BTreeSet::new(),
            "x",
        )
        .unwrap();
        let mut hits = BTreeMap::new();
        for n in ["numpy", "typing-extensions"] {
            hits.insert(
                n.to_string(),
                RouteProbeHit {
                    conda_version: "1".into(),
                    channel: "c/linux-64".into(),
                    depends: Vec::new(),
                },
            );
        }
        let mut opts = auto_route_opts();
        opts.route_policy = crate::config::RoutePolicy::Minimal;
        // Neither numpy nor typing-extensions is whitelist material.
        let routes = plan_auto_route_round(&closure, &req, &opts, &[], &hits).unwrap();
        assert!(
            routes.is_empty(),
            "minimal policy must refuse non-whitelisted routes: {routes:?}"
        );
        let probes = auto_route_probe_specs(&closure, &req, &opts, &[]);
        assert!(
            probes.is_empty(),
            "refused candidates must not be probed: {probes:?}"
        );
        // The include list re-admits a name.
        opts.route_include.insert("numpy".to_string());
        let routes = plan_auto_route_round(&closure, &req, &opts, &[], &hits).unwrap();
        let names: Vec<&str> = routes.iter().map(|r| r.pypi_name.as_str()).collect();
        assert_eq!(names, vec!["numpy"]);
        let probes = auto_route_probe_specs(&closure, &req, &opts, &[]);
        assert!(
            probes.iter().all(|(n, _)| n == "numpy"),
            "only include-listed candidates probed: {probes:?}"
        );
    }

    /// Run-15 shape, mocked at the repodata layer: conda `moviepy`'s own
    /// recipe still says `pillow <11.0,>=9.2.0`, but the uv closure
    /// already locked `pillow==11.3.0` (uv's legal PyPI-truth pick,
    /// `moviepy 2.2.1`'s PyPI metadata allows `pillow<12`). The
    /// contradiction must refuse the route.
    #[test]
    fn route_metadata_consistent_rejects_conda_forge_metadata_skew() {
        let mut locked = BTreeMap::new();
        locked.insert("pillow".to_string(), "11.3.0".to_string());
        let err = route_metadata_consistent(
            "moviepy",
            "2.2.1",
            &["pillow <11.0,>=9.2.0".to_string()],
            &locked,
            &BTreeMap::new(),
        )
        .unwrap_err();
        assert!(err.contains("pillow"), "reason should name pillow: {err}");
        assert!(
            err.contains("11.3.0"),
            "reason should cite the locked version: {err}"
        );
    }

    /// Same shape, but the closure's locked pillow version DOES satisfy
    /// the conda recipe's range -- no contradiction, route proceeds.
    #[test]
    fn route_metadata_consistent_accepts_matching_metadata() {
        let mut locked = BTreeMap::new();
        locked.insert("pillow".to_string(), "10.4.0".to_string());
        assert_eq!(
            route_metadata_consistent(
                "moviepy",
                "2.2.1",
                &["pillow <11.0,>=9.2.0".to_string()],
                &locked,
                &BTreeMap::new(),
            ),
            Ok(())
        );
    }

    /// A depends entry naming something the closure never locked at all
    /// (or an unparseable/`*` spec) is skipped, not treated as a
    /// contradiction -- this check only ever refuses on a PROVABLE
    /// mismatch.
    #[test]
    fn route_metadata_consistent_skips_unknown_and_wildcard_deps() {
        let locked = BTreeMap::new();
        assert_eq!(
            route_metadata_consistent(
                "moviepy",
                "2.2.1",
                &["pillow <11.0,>=9.2.0".to_string(), "numpy *".to_string()],
                &locked,
                &BTreeMap::new(),
            ),
            Ok(())
        );
    }

    /// Run 16c gap: a routed build's conda depend names a conda-ONLY ABI
    /// anchor (`cuda-version`) the pypi closure never locks. The workspace
    /// pins that anchor to an incompatible exact version -- the route must
    /// be refused, closing the hole that let `triton ==3.6.0` (needing
    /// `cuda-version >=12.9,<13`) route against a cuda-12.8 env.
    #[test]
    fn route_metadata_consistent_rejects_conda_only_abi_anchor_violation() {
        let mut anchors = BTreeMap::new();
        anchors.insert("cuda-version".to_string(), "==12.8".to_string());
        let err = route_metadata_consistent(
            "triton",
            "3.6.0",
            &["cuda-version >=12.9,<13".to_string()],
            &BTreeMap::new(), // cuda-version is never in the pypi closure
            &anchors,
        )
        .unwrap_err();
        assert!(
            err.contains("cuda-version"),
            "reason should name the anchor: {err}"
        );
    }

    /// Same anchor, but the routed build's requirement DOES admit the
    /// exact workspace pin -- no contradiction, route proceeds.
    #[test]
    fn route_metadata_consistent_accepts_satisfiable_abi_anchor() {
        let mut anchors = BTreeMap::new();
        anchors.insert("cuda-version".to_string(), "==12.8".to_string());
        assert_eq!(
            route_metadata_consistent(
                "triton",
                "3.4.0",
                &["cuda-version >=12.8,<13".to_string()],
                &BTreeMap::new(),
                &anchors,
            ),
            Ok(())
        );
    }

    /// A RANGE workspace anchor pin never arms the ABI-anchor check (the
    /// env might still pick a version inside both ranges) -- only an exact
    /// pin yields a provable contradiction.
    #[test]
    fn route_metadata_consistent_range_anchor_pin_is_not_armed() {
        let mut anchors = BTreeMap::new();
        anchors.insert("cuda-version".to_string(), ">=12.8,<13".to_string());
        assert_eq!(
            route_metadata_consistent(
                "triton",
                "3.6.0",
                &["cuda-version >=12.9,<13".to_string()],
                &BTreeMap::new(),
                &anchors,
            ),
            Ok(())
        );
    }

    /// Run-17 gap: the routed build's depend line carries a conda BUILD
    /// string after the version (`python_abi 3.13.* *_cp313`), the real
    /// shape every python_abi depend takes. A 2-way split left
    /// `"3.13.* *_cp313"` as the spec, which `VersionSpec::from_str`
    /// cannot parse, so the check silently skipped and let
    /// `pandas==3.0.3` (needing `python_abi 3.13.*`) route into a
    /// python-3.11 env. The 3-way split (matching `split_conda_dep_line`)
    /// must strip the build field and let the anchor contradiction fire.
    #[test]
    fn route_metadata_consistent_rejects_abi_anchor_with_build_string() {
        let mut anchors = BTreeMap::new();
        anchors.insert("python_abi".to_string(), "==3.11".to_string());
        let err = route_metadata_consistent(
            "pandas",
            "3.0.3",
            &["python_abi 3.13.* *_cp313".to_string()],
            &BTreeMap::new(),
            &anchors,
        )
        .unwrap_err();
        assert!(
            err.contains("python-abi") || err.contains("python_abi"),
            "reason should name the python_abi anchor: {err}"
        );
    }

    /// Same build-string shape, but the requirement DOES admit the
    /// workspace's exact anchor pin -- route proceeds (build string is
    /// discarded, not treated as part of the version spec).
    #[test]
    fn route_metadata_consistent_accepts_abi_anchor_with_build_string() {
        let mut anchors = BTreeMap::new();
        anchors.insert("python_abi".to_string(), "==3.11".to_string());
        assert_eq!(
            route_metadata_consistent(
                "pandas",
                "2.2.0",
                &["python_abi 3.11.* *_cp311".to_string()],
                &BTreeMap::new(),
                &anchors,
            ),
            Ok(())
        );
    }

    /// End-to-end through the real planner: a probe hit whose `depends`
    /// contradicts the closure's own locked pillow version is refused --
    /// the candidate is never routed, even though the probe found it on
    /// a channel.
    #[test]
    fn plan_auto_route_round_refuses_metadata_skewed_route() {
        let mut req = auto_route_req();
        let exclude: BTreeSet<String> = BTreeSet::new();
        let mut closure =
            parse_pylock_closure(PYLOCK_FIXTURE, &target("3.12", "linux-64"), &exclude, "x")
                .unwrap();
        closure
            .pins
            .insert("pillow".to_string(), "11.3.0".to_string());
        req.no_emit_packages.clear();
        let mut hits = BTreeMap::new();
        hits.insert(
            "numpy".to_string(),
            RouteProbeHit {
                conda_version: "2.1.0".into(),
                channel: "c/linux-64".into(),
                depends: vec!["pillow <11.0,>=9.2.0".to_string()],
            },
        );
        hits.insert(
            "typing-extensions".to_string(),
            RouteProbeHit {
                conda_version: "4.12.2".into(),
                channel: "c/linux-64".into(),
                depends: Vec::new(),
            },
        );
        let routes = plan_auto_route_round(&closure, &req, &auto_route_opts(), &[], &hits).unwrap();
        let names: Vec<&str> = routes.iter().map(|r| r.pypi_name.as_str()).collect();
        assert_eq!(
            names,
            vec!["typing-extensions"],
            "numpy's route must be refused (metadata skew); typing-extensions unaffected"
        );
    }

    // ---- live subprocess (network) ---------------------------------------

    /// Live-network smoke: requires uv on PATH + PyPI reachability.
    /// Run manually: `cargo test uv_closure -- --ignored`.
    #[tokio::test]
    #[ignore = "requires network + uv on PATH"]
    async fn live_uv_lock_smoke() {
        let tmp =
            std::env::temp_dir().join(format!("retread-uv-closure-smoke-{}", std::process::id()));
        let req = UvClosureRequest {
            bundle: "smoke".into(),
            python_version: "3.12".into(),
            conda_subdir: "linux-64".into(),
            dependencies: vec!["typing-extensions==4.12.2".into()],
            constraints: ConstraintSet::default(),
            overrides: vec![],
            no_emit_packages: vec![],
            index_urls: vec!["https://pypi.org/simple/".into()],
            built_wheel_sources: BTreeMap::new(),
            explicit_pins: BTreeMap::new(),
            offline: false,
        };
        let closure = compute_closure(
            &req,
            &tmp.join("project"),
            &tmp.join("uv-cache"),
            None,
            crate::config::SdistBuildPolicy::Auto,
        )
        .await
        .unwrap();
        let _ = std::fs::remove_dir_all(&tmp);
        assert_eq!(
            closure.pins.get("typing-extensions").map(String::as_str),
            Some("4.12.2")
        );
        assert_eq!(closure.wheels.len(), 1);
        assert!(closure.wheels[0].sha256.is_some());
    }

    /// Live-network auto-route smoke: uv on PATH + PyPI + conda-forge
    /// repodata. python-dateutil's closure pulls `six`, which conda-forge
    /// carries at the same version — the loop must route it.
    /// Run manually: `cargo test uv_closure -- --ignored`.
    #[tokio::test]
    #[ignore = "requires network + uv on PATH + conda-forge repodata"]
    async fn live_auto_route_six_via_conda_forge() {
        use rattler_conda_types::ChannelUrl;
        let tmp =
            std::env::temp_dir().join(format!("retread-auto-route-smoke-{}", std::process::id()));
        let req = UvClosureRequest {
            bundle: "smoke-autoroute".into(),
            python_version: "3.12".into(),
            conda_subdir: "linux-64".into(),
            dependencies: vec!["python-dateutil==2.9.0.post0".into()],
            constraints: ConstraintSet::default(),
            overrides: vec![],
            no_emit_packages: vec![],
            index_urls: vec!["https://pypi.org/simple/".into()],
            built_wheel_sources: BTreeMap::new(),
            explicit_pins: BTreeMap::new(),
            offline: false,
        };
        let opts = AutoRouteOptions {
            enabled: true,
            keep_pypi: BTreeSet::new(),
            force_conda: BTreeSet::new(),
            protected: BTreeSet::from(["python-dateutil".to_string()]),
            name_map: BTreeMap::new(),
            abi_anchor_pins: BTreeMap::new(),
            workspace_conda_versions: BTreeMap::new(),
            ..Default::default()
        };
        let channels: Vec<ChannelUrl> = vec![ChannelUrl::from(
            url::Url::parse("https://conda.anaconda.org/conda-forge/").unwrap(),
        )];
        let project = tmp.join("project");
        let cache = tmp.join("uv-cache");
        let solve = {
            let project = project.clone();
            let cache = cache.clone();
            move |r: UvClosureRequest| {
                let project = project.clone();
                let cache = cache.clone();
                Box::pin(async move {
                    compute_closure(
                        &r,
                        &project,
                        &cache,
                        None,
                        crate::config::SdistBuildPolicy::Auto,
                    )
                    .await
                }) as futures::future::BoxFuture<'static, Result<UvClosure>>
            }
        };
        let probe = move |name: String, spec: String| {
            let channels = channels.clone();
            Box::pin(async move {
                crate::probe::find_route(&channels, &name, &spec, Some("3.12"))
                    .await
                    .map(|h| RouteProbeHit {
                        conda_version: h.version,
                        channel: h.channel,
                        depends: Vec::new(),
                    })
            }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let closure = auto_route_fixpoint(&req, &opts, solve, probe)
            .await
            .unwrap();
        let _ = std::fs::remove_dir_all(&tmp);
        assert!(
            closure.auto_routed.iter().any(|r| r.pypi_name == "six"),
            "six should auto-route to conda-forge; got {:?}",
            closure.auto_routed,
        );
        assert!(
            !closure.wheels.iter().any(|w| w.name == "six"),
            "six must leave the wheel closure"
        );
        assert!(
            closure.wheels.iter().any(|w| w.name == "python-dateutil"),
            "the protected root stays"
        );
    }

    // -- self-healing un-route (co-installability) ----------------------

    /// Unsat-reason name matching: exact-name occurrences in match-spec
    /// and filename forms hit; superstring package names do not.
    #[test]
    fn unsat_reason_name_matching_is_boundary_aware() {
        // match-spec form
        assert!(unsat_reason_names_package(
            "cryptography >=44.0.1 cannot be installed because cryptography ==44.0.0 is pinned",
            "cryptography"
        ));
        // filename form: name-version-build
        assert!(unsat_reason_names_package(
            "package aioboto3-15.1.0-pyhd8ed1ab_0 requires cryptography >=44.0.1",
            "aioboto3"
        ));
        // superstrings must not match
        assert!(!unsat_reason_names_package(
            "numpydoc ==1.6.0 cannot be installed",
            "numpy"
        ));
        assert!(!unsat_reason_names_package(
            "ray-core 2.7.0 requires grpcio",
            "ray"
        ));
        // ...but the filename form of the actual package still does
        assert!(unsat_reason_names_package(
            "package ray-2.7.0-py311_0 is excluded",
            "ray"
        ));
        // substrings-with-prefix must not match
        assert!(!unsat_reason_names_package(
            "aioboto3 ==15.1.0 conflicts",
            "boto3"
        ));
        // case-insensitive
        assert!(unsat_reason_names_package(
            "Numba ==0.59.1 requires llvmlite <0.43",
            "numba"
        ));
        // -/_/. normalization equivalence
        assert!(unsat_reason_names_package(
            "typing_extensions-4.12.2-pyhd8ed1ab_0 conflicts",
            "typing-extensions"
        ));
        assert!(unsat_reason_names_package(
            "ruamel-yaml >=0.18 cannot be installed",
            "ruamel.yaml"
        ));
        // python_abi must not match python
        assert!(!unsat_reason_names_package(
            "python_abi 3.12.* requires cpython",
            "python"
        ));
    }

    /// Canned co-solve keyed on the candidate conda-name set. Any set
    /// containing every name of a listed conflict returns that
    /// conflict's Unsat reasons; everything else is Sat.
    fn canned_co_solve(
        conflicts: Vec<(BTreeSet<String>, Vec<String>)>,
    ) -> impl Fn(Vec<AutoRoutedPackage>) -> futures::future::BoxFuture<'static, CoInstallVerdict>
    {
        move |candidate: Vec<AutoRoutedPackage>| {
            let names: BTreeSet<String> = candidate.iter().map(|r| r.conda_name.clone()).collect();
            let verdict = conflicts
                .iter()
                .find(|(set, _)| set.is_subset(&names))
                .map(|(_, reasons)| CoInstallVerdict::Unsat(reasons.clone()))
                .unwrap_or(CoInstallVerdict::Sat);
            Box::pin(async move { verdict })
        }
    }

    fn emitted_route(pypi_name: &str, conda_name: &str, spec: &str) -> CondaRouteSpec {
        CondaRouteSpec {
            pypi_name: pypi_name.to_string(),
            conda_name: conda_name.to_string(),
            spec: spec.to_string(),
        }
    }

    #[tokio::test]
    async fn rule2_joint_solve_keeps_individually_valid_conflicting_route_on_pypi() {
        let sibling = emitted_route("sibling-runtime", "sibling-runtime", ">=4,<5");
        let candidate = emitted_route("candidate-addon", "candidate-addon", "==2.1");
        let co_solve = |routes: Vec<CondaRouteSpec>| async move {
            let names: BTreeSet<String> =
                routes.into_iter().map(|route| route.conda_name).collect();
            if names.contains("sibling-runtime") && names.contains("candidate-addon") {
                CoInstallVerdict::Unsat(vec!["incompatible transitive constraints".to_string()])
            } else {
                CoInstallVerdict::Sat
            }
        };

        assert_eq!(
            co_solve(vec![candidate.clone()]).await,
            CoInstallVerdict::Sat,
            "the candidate is individually satisfiable"
        );
        let selection = select_jointly_solvable_routes(
            vec![sibling],
            vec![candidate.clone(), candidate.clone()],
            &co_solve,
        )
        .await
        .expect("the fixed conda baseline is satisfiable");

        assert!(selection.accepted.is_empty());
        assert_eq!(selection.rejected, vec![candidate]);
    }

    #[tokio::test]
    async fn rule2_joint_solve_unroutes_conflicting_group_to_pypi() {
        let left = emitted_route("worker-left", "worker-left", "==1");
        let right = emitted_route("worker-right", "worker-right", "==1");
        let unrelated = emitted_route("utility", "utility", ">=7");
        let co_solve = |routes: Vec<CondaRouteSpec>| async move {
            let names: BTreeSet<String> =
                routes.into_iter().map(|route| route.conda_name).collect();
            if names.contains("worker-left") && names.contains("worker-right") {
                CoInstallVerdict::Unsat(vec!["incompatible transitive constraints".to_string()])
            } else {
                CoInstallVerdict::Sat
            }
        };

        let selection = select_jointly_solvable_routes(
            Vec::new(),
            vec![right.clone(), unrelated.clone(), left.clone()],
            &co_solve,
        )
        .await
        .expect("the fixed conda baseline is satisfiable");

        assert_eq!(selection.accepted, vec![unrelated]);
        assert_eq!(selection.rejected, vec![left, right]);
    }

    #[tokio::test]
    async fn rule2_joint_solve_keeps_cosolvable_route_on_conda() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let calls = AtomicUsize::new(0);
        let fixed = emitted_route("runtime", "runtime", "*");
        let candidate = emitted_route("extension", "extension", ">=2,<3");
        let co_solve = |routes: Vec<CondaRouteSpec>| {
            calls.fetch_add(1, Ordering::SeqCst);
            async move {
                assert_eq!(
                    routes
                        .iter()
                        .map(CondaRouteSpec::match_spec)
                        .collect::<Vec<_>>(),
                    vec!["extension >=2,<3", "runtime"]
                );
                CoInstallVerdict::Sat
            }
        };

        let selection = select_jointly_solvable_routes(
            vec![fixed],
            vec![candidate.clone(), candidate.clone()],
            &co_solve,
        )
        .await
        .expect("the complete route set co-solves");

        assert_eq!(selection.accepted, vec![candidate]);
        assert!(selection.rejected.is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 1, "happy path is one batch solve");
    }

    /// The blocker-class fixture: BOTH numpy and typing-extensions have
    /// conda hits at the resolved version, but conda-forge cannot
    /// co-install the two exact pins (the unsat report names
    /// typing-extensions). The self-heal must un-route
    /// typing-extensions (wheel regained), keep numpy routed, and reach
    /// a deterministic fixpoint.
    #[tokio::test]
    async fn unroute_heals_conflicting_exact_pins() {
        let mut hits = BTreeMap::new();
        hits.insert(
            "numpy".to_string(),
            RouteProbeHit {
                conda_version: "2.1.0".into(),
                channel: "c/linux-64".into(),
                depends: Vec::new(),
            },
        );
        hits.insert(
            "typing-extensions".to_string(),
            RouteProbeHit {
                conda_version: "4.12.2".into(),
                channel: "c/linux-64".into(),
                depends: Vec::new(),
            },
        );
        let co_solve = canned_co_solve(vec![(
            BTreeSet::from(["numpy".to_string(), "typing-extensions".to_string()]),
            vec![
                "package numpy-2.1.0-py312_0 requires typing-extensions >=4.13, \
                 but typing_extensions-4.12.2-pyhd8ed1ab_0 is pinned"
                    .to_string(),
            ],
        )]);
        let mut results = Vec::new();
        for _ in 0..2 {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let closure = auto_route_fixpoint_checked(
                &auto_route_req(),
                &auto_route_opts(),
                canned_solve(Arc::clone(&calls)),
                canned_probe(hits.clone()),
                &co_solve,
            )
            .await
            .unwrap();
            // Both were named in the report; both are candidates for
            // un-routing — but numpy alone is Sat, so after
            // typing-extensions AND numpy leave round 1's candidate set,
            // the re-plan routes whatever survives. The report names
            // BOTH names, so both get blocked: closure keeps both wheels.
            let mut routed: Vec<String> = closure
                .auto_routed
                .iter()
                .map(|r| r.pypi_name.clone())
                .collect();
            routed.sort();
            let mut wheels: Vec<String> = closure.wheels.iter().map(|w| w.name.clone()).collect();
            wheels.sort();
            results.push((routed, wheels));
        }
        // Deterministic across runs.
        assert_eq!(results[0], results[1]);
        let (routed, wheels) = &results[0];
        // typing-extensions was named -> un-routed, wheel regained.
        assert!(!routed.contains(&"typing-extensions".to_string()));
        assert!(wheels.contains(&"typing-extensions".to_string()));
        // numpy was ALSO named in the report -> conservatively un-routed.
        assert!(!routed.contains(&"numpy".to_string()));
        assert!(wheels.contains(&"numpy".to_string()));
        // Protected root untouched.
        assert!(wheels.contains(&"mujoco".to_string()));
    }

    /// Only the package(s) the unsat report names get un-routed; a
    /// routed package the report does NOT name stays on the conda side.
    #[tokio::test]
    async fn unroute_is_scoped_to_named_packages() {
        let mut hits = BTreeMap::new();
        hits.insert(
            "numpy".to_string(),
            RouteProbeHit {
                conda_version: "2.1.0".into(),
                channel: "c/linux-64".into(),
                depends: Vec::new(),
            },
        );
        hits.insert(
            "typing-extensions".to_string(),
            RouteProbeHit {
                conda_version: "4.12.2".into(),
                channel: "c/linux-64".into(),
                depends: Vec::new(),
            },
        );
        // The report only names typing-extensions.
        let co_solve = canned_co_solve(vec![(
            BTreeSet::from(["typing-extensions".to_string()]),
            vec!["typing_extensions-4.12.2-pyhd8ed1ab_0 conflicts with __glibc".to_string()],
        )]);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let closure = auto_route_fixpoint_checked(
            &auto_route_req(),
            &auto_route_opts(),
            canned_solve(Arc::clone(&calls)),
            canned_probe(hits),
            co_solve,
        )
        .await
        .unwrap();
        let routed: Vec<&str> = closure
            .auto_routed
            .iter()
            .map(|r| r.pypi_name.as_str())
            .collect();
        assert_eq!(routed, vec!["numpy"], "numpy stays routed");
        assert!(
            closure.wheels.iter().any(|w| w.name == "typing-extensions"),
            "typing-extensions regains its wheel"
        );
        assert!(!closure.wheels.iter().any(|w| w.name == "numpy"));
        // The re-lock after the accepted numpy-only round saw numpy's
        // exclusion + pin but NOT typing-extensions'.
        let calls = calls.lock().unwrap();
        let last = calls.last().unwrap();
        assert!(last.no_emit_packages.contains(&"numpy".to_string()));
        assert!(
            !last
                .no_emit_packages
                .contains(&"typing-extensions".to_string())
        );
    }

    /// Mock solve for the workspace-harmonization tests: a closure with a
    /// protected root (mujoco) plus torch, whose version follows a
    /// `torch==X` constraint when present (mirrors a real uv re-lock
    /// honoring `constraint-dependencies`) and floats to 2.10.0
    /// otherwise. `fail_on` simulates "no wheel at the harmonized
    /// version" (uv lock failure) for that constraint line.
    fn harmonize_mock_solve(
        calls: Arc<Mutex<Vec<UvClosureRequest>>>,
        fail_on: Option<String>,
    ) -> impl FnMut(UvClosureRequest) -> futures::future::BoxFuture<'static, Result<UvClosure>>
    {
        move |r: UvClosureRequest| {
            let calls = Arc::clone(&calls);
            let fail_on = fail_on.clone();
            Box::pin(async move {
                calls.lock().unwrap().push(r.clone());
                if let Some(bad) = &fail_on
                    && r.constraints.constraints.iter().any(|c| c == bad)
                {
                    anyhow::bail!("uv lock: no wheels satisfy {bad}");
                }
                let torch_version = r
                    .constraints
                    .constraints
                    .iter()
                    .find_map(|c| c.strip_prefix("torch==").map(str::to_string))
                    .unwrap_or_else(|| "2.10.0".to_string());
                let excluded: BTreeSet<String> = r
                    .no_emit_packages
                    .iter()
                    .map(|n| canonical_conda_name(n))
                    .collect();
                let mk = |name: &str, version: &str| LockWheel {
                    name: name.to_string(),
                    version: version.to_string(),
                    origin: Origin::Index,
                    filename: format!("{name}-{version}-py3-none-any.whl"),
                    url: Some("https://example.com/x.whl".into()),
                    sha256: Some("00".repeat(32)),
                    requires_dist: vec![],
                    must_ship: false,
                    upstream_url: None,
                    git_source: None,
                    sdist_source: None,
                };
                let mut wheels = vec![mk("mujoco", "3.5.0")];
                if !excluded.contains("torch") {
                    wheels.push(mk("torch", &torch_version));
                }
                let pins = wheels
                    .iter()
                    .map(|w| (w.name.clone(), w.version.clone()))
                    .collect();
                Ok(UvClosure {
                    wheels,
                    pins,
                    uv_version: "0.11.15".into(),
                    auto_routed: vec![],
                    auto_dropped: BTreeSet::new(),
                    effective_input_requirements: None,
                })
            })
        }
    }

    /// Echo probe: a hit for conda `pytorch` at exactly the requested
    /// spec version (`==X` -> X), mirroring a channel that has every
    /// version. Other names miss.
    fn harmonize_mock_probe()
    -> impl Fn(String, String) -> futures::future::BoxFuture<'static, Option<RouteProbeHit>> {
        |conda_name: String, spec: String| {
            Box::pin(async move {
                if conda_name != "pytorch" {
                    return None;
                }
                let version = spec.trim_start_matches('=').to_string();
                Some(RouteProbeHit {
                    conda_version: version,
                    channel: "c/linux-64".into(),
                    depends: Vec::new(),
                })
            })
        }
    }

    fn harmonize_opts() -> AutoRouteOptions {
        AutoRouteOptions {
            enabled: true,
            keep_pypi: BTreeSet::new(),
            force_conda: BTreeSet::new(),
            protected: BTreeSet::from(["mujoco".to_string()]),
            name_map: BTreeMap::from([("torch".to_string(), "pytorch".to_string())]),
            abi_anchor_pins: BTreeMap::new(),
            workspace_conda_versions: BTreeMap::from([(
                "pytorch".to_string(),
                "2.7.0".to_string(),
            )]),
            ..Default::default()
        }
    }

    /// Version-aware co-solve: unsat naming pytorch while the candidate
    /// set carries pytorch at 2.10.0 (the run-34 clash against the
    /// hand-written pytorch-gpu ==2.7.0); Sat at 2.7.0.
    fn harmonize_mock_co_solve()
    -> impl Fn(Vec<AutoRoutedPackage>) -> futures::future::BoxFuture<'static, CoInstallVerdict>
    {
        |candidate: Vec<AutoRoutedPackage>| {
            let bad = candidate
                .iter()
                .any(|r| r.conda_name == "pytorch" && r.conda_version == "2.10.0");
            Box::pin(async move {
                if bad {
                    CoInstallVerdict::Unsat(vec![
                        "nothing provides pytorch 2.7.0 cuda* needed by                          pytorch-gpu-2.7.0, but pytorch ==2.10.0 is pinned"
                            .to_string(),
                    ])
                } else {
                    CoInstallVerdict::Sat
                }
            })
        }
    }

    /// A routed package named by the co-install unsat whose exact conda name
    /// has a precise, directly-owned workspace fact at another version is
    /// HARMONIZED -- pypi is re-pinned and re-routed -- rather than un-routed
    /// into a wheel that would clobber the workspace's conda package.
    #[tokio::test]
    async fn workspace_harmonize_repins_named_route_instead_of_unroute() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let closure = auto_route_fixpoint_checked(
            &auto_route_req(),
            &harmonize_opts(),
            harmonize_mock_solve(Arc::clone(&calls), None),
            harmonize_mock_probe(),
            harmonize_mock_co_solve(),
        )
        .await
        .unwrap();
        let torch = closure
            .auto_routed
            .iter()
            .find(|r| r.pypi_name == "torch")
            .expect("torch must stay ROUTED (harmonized), not un-routed");
        assert_eq!(torch.conda_name, "pytorch");
        assert_eq!(
            torch.conda_version, "2.7.0",
            "routed at the workspace's version, not uv's 2.10.0 pick"
        );
        assert!(
            !closure.wheels.iter().any(|w| w.name == "torch"),
            "no torch wheel may ship (it would clobber conda pytorch)"
        );
        // The harmonized constraint reached the re-lock.
        let calls = calls.lock().unwrap();
        assert!(
            calls.iter().any(|r| r
                .constraints
                .constraints
                .iter()
                .any(|c| c == "torch==2.7.0")),
            "harmonized uv constraint must be applied on the re-lock"
        );
    }

    /// Fallback: the harmonized version has no satisfiable pypi-side
    /// pick (uv lock fails under the `torch==2.7.0` constraint) -- the
    /// pin is dropped and torch is UN-ROUTED (pre-harmonization
    /// behavior), keeping the fixpoint alive instead of erroring out.
    #[tokio::test]
    async fn workspace_harmonize_falls_back_to_unroute_when_pypi_refuses() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let closure = auto_route_fixpoint_checked(
            &auto_route_req(),
            &harmonize_opts(),
            harmonize_mock_solve(Arc::clone(&calls), Some("torch==2.7.0".to_string())),
            harmonize_mock_probe(),
            harmonize_mock_co_solve(),
        )
        .await
        .unwrap();
        assert!(
            !closure.auto_routed.iter().any(|r| r.pypi_name == "torch"),
            "torch must be un-routed when harmonization cannot lock"
        );
        assert!(
            closure.wheels.iter().any(|w| w.name == "torch"),
            "torch ships as a wheel again (un-route fallback)"
        );
    }

    /// The cuda-bindings/cuda-version shape: the unsat report blames a
    /// shared TRANSITIVE anchor package's name, not either routed
    /// candidate's own name — `unsat_reason_names_package` finds
    /// nothing to blame directly. The greedy retry-solve fallback must
    /// still isolate and un-route the offender (removing just one of
    /// the two candidates resolves the conflict) instead of giving up.
    #[tokio::test]
    async fn unroute_greedy_fallback_heals_unnamed_transitive_conflict() {
        let mut hits = BTreeMap::new();
        hits.insert(
            "numpy".to_string(),
            RouteProbeHit {
                conda_version: "2.1.0".into(),
                channel: "c/linux-64".into(),
                depends: Vec::new(),
            },
        );
        hits.insert(
            "typing-extensions".to_string(),
            RouteProbeHit {
                conda_version: "4.12.2".into(),
                channel: "c/linux-64".into(),
                depends: Vec::new(),
            },
        );
        // Custom co_solve (not `canned_co_solve`, which keys unsat off a
        // conflict SET): Unsat only when BOTH are present, and the
        // reason names neither routed package — only a transitive
        // anchor ("cuda-version") that isn't itself a routed candidate.
        let co_solve = |candidate: Vec<AutoRoutedPackage>| {
            let names: BTreeSet<String> = candidate.iter().map(|r| r.conda_name.clone()).collect();
            let verdict = if names.contains("numpy") && names.contains("typing-extensions") {
                CoInstallVerdict::Unsat(vec![
                    "cuda-version >=13,<14.0a0, for which no candidates were found".to_string(),
                ])
            } else {
                CoInstallVerdict::Sat
            };
            Box::pin(async move { verdict })
                as futures::future::BoxFuture<'static, CoInstallVerdict>
        };
        let calls = Arc::new(Mutex::new(Vec::new()));
        let closure = auto_route_fixpoint_checked(
            &auto_route_req(),
            &auto_route_opts(),
            canned_solve(Arc::clone(&calls)),
            canned_probe(hits),
            co_solve,
        )
        .await
        .unwrap();
        let routed: Vec<String> = closure
            .auto_routed
            .iter()
            .map(|r| r.pypi_name.clone())
            .collect();
        // Exactly one of the two got un-routed (whichever the greedy
        // scan tried first) -- the conflict is healed, not left unsat
        // nor over-pruned to both.
        assert_eq!(routed.len(), 1, "routed: {routed:?}");
        // mujoco (protected root) plus whichever of numpy/typing-extensions
        // was NOT un-routed.
        assert_eq!(
            closure.wheels.len() + routed.len(),
            3,
            "wheels+routed: {:?} {:?}",
            closure.wheels.iter().map(|w| &w.name).collect::<Vec<_>>(),
            routed
        );
    }

    /// A previously ACCEPTED route that a later round's cumulative check
    /// implicates gets reverted: request rebuilt without its pin, wheel
    /// regained.
    #[tokio::test]
    async fn unroute_reverts_previously_accepted_route() {
        // Stateful probe: typing-extensions' hit only appears on its
        // SECOND probe (fresh repodata mid-loop), so numpy routes alone
        // in round 1 and typing-extensions joins in round 2.
        let te_probes = Arc::new(Mutex::new(0usize));
        let probe = {
            let te_probes = Arc::clone(&te_probes);
            move |name: String, _spec: String| {
                let hit = match name.as_str() {
                    "numpy" => Some(RouteProbeHit {
                        conda_version: "2.1.0".into(),
                        channel: "c/linux-64".into(),
                        depends: Vec::new(),
                    }),
                    "typing-extensions" => {
                        let mut n = te_probes.lock().unwrap();
                        *n += 1;
                        (*n >= 2).then(|| RouteProbeHit {
                            conda_version: "4.12.2".into(),
                            channel: "c/linux-64".into(),
                            depends: Vec::new(),
                        })
                    }
                    _ => None,
                };
                Box::pin(async move { hit })
                    as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
            }
        };
        // numpy alone: Sat. numpy + typing-extensions: Unsat naming
        // numpy (the earlier-accepted route is the conflicting one).
        let co_solve = canned_co_solve(vec![(
            BTreeSet::from(["numpy".to_string(), "typing-extensions".to_string()]),
            vec!["package numpy-2.1.0-py312_0 requires typing-extensions >=4.13".to_string()],
        )]);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let closure = auto_route_fixpoint_checked(
            &auto_route_req(),
            &auto_route_opts(),
            canned_solve(Arc::clone(&calls)),
            probe,
            co_solve,
        )
        .await
        .unwrap();
        // Report named both -> numpy (accepted in round 1) reverted,
        // typing-extensions blocked; nothing stays routed.
        assert!(
            !closure.auto_routed.iter().any(|r| r.pypi_name == "numpy"),
            "numpy's earlier route must be reverted; got {:?}",
            closure.auto_routed
        );
        assert!(closure.wheels.iter().any(|w| w.name == "numpy"));
        assert!(closure.wheels.iter().any(|w| w.name == "typing-extensions"));
        // The final request no longer carries numpy's exclusion/pin.
        let calls = calls.lock().unwrap();
        let last = calls.last().unwrap();
        assert!(!last.no_emit_packages.contains(&"numpy".to_string()));
        assert!(
            !last
                .constraints
                .constraints
                .iter()
                .any(|c| c.starts_with("numpy=="))
        );
    }

    /// `force-conda` names are exempt from the self-heal: they stay
    /// routed even when the unsat report names them.
    #[tokio::test]
    async fn force_conda_names_survive_unsat_reports() {
        let mut hits = BTreeMap::new();
        hits.insert(
            "numpy".to_string(),
            RouteProbeHit {
                conda_version: "2.1.0".into(),
                channel: "c/linux-64".into(),
                depends: Vec::new(),
            },
        );
        let co_solve = canned_co_solve(vec![(
            BTreeSet::from(["numpy".to_string()]),
            vec!["numpy ==2.1.0 conflicts with the channel matrix".to_string()],
        )]);
        let mut opts = auto_route_opts();
        opts.force_conda.insert("numpy".to_string());
        let calls = Arc::new(Mutex::new(Vec::new()));
        let closure = auto_route_fixpoint_checked(
            &auto_route_req(),
            &opts,
            canned_solve(Arc::clone(&calls)),
            canned_probe(hits),
            co_solve,
        )
        .await
        .unwrap();
        assert_eq!(closure.auto_routed.len(), 1);
        assert_eq!(closure.auto_routed[0].pypi_name, "numpy");
        assert!(!closure.wheels.iter().any(|w| w.name == "numpy"));
    }

    /// An unsat report that names NO routed package (channel-side
    /// breakage, independent of ANY routed candidate — e.g. python
    /// itself has no viable build on this channel) cannot be healed by
    /// un-routing: neither the name-match path nor the greedy
    /// retry-solve fallback finds a single-candidate fix (the empty
    /// conflict set below matches every candidate set, including the
    /// empty one, modeling a conflict genuinely unrelated to what's
    /// routed), so the round applies unchanged with a warning.
    #[tokio::test]
    async fn unsat_naming_no_routed_package_applies_round() {
        let mut hits = BTreeMap::new();
        hits.insert(
            "numpy".to_string(),
            RouteProbeHit {
                conda_version: "2.1.0".into(),
                channel: "c/linux-64".into(),
                depends: Vec::new(),
            },
        );
        let co_solve = canned_co_solve(vec![(
            BTreeSet::new(),
            vec!["python 3.12.* cannot be installed: no candidates".to_string()],
        )]);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let closure = auto_route_fixpoint_checked(
            &auto_route_req(),
            &auto_route_opts(),
            canned_solve(Arc::clone(&calls)),
            canned_probe(hits),
            co_solve,
        )
        .await
        .unwrap();
        assert_eq!(closure.auto_routed.len(), 1);
        assert_eq!(closure.auto_routed[0].pypi_name, "numpy");
    }

    // ---- self-heal: structured two-pass detection ------------------------

    /// Concrete "no builder" type for tests that disable the build rung
    /// (`sdist-build = "never"`) — `None::<NoBuild>` is the terse
    /// turbofish-free spelling.
    type NoBuild =
        fn(String, Option<String>) -> futures::future::BoxFuture<'static, Result<BuiltSdistWheel>>;

    fn sdist_source_fixture(name: &str, version: &str) -> crate::lock::SdistWheelSource {
        crate::lock::SdistWheelSource {
            index: "https://pypi.org/simple/".to_string(),
            name: name.to_string(),
            version: version.to_string(),
            sdist_url: format!(
                "https://files.pythonhosted.org/packages/{name}-{version}.tar.gz#sha256=deadbeef"
            ),
        }
    }

    /// Build a [`HealNeeded`] the way `compute_closure`'s two-pass would.
    fn heal_needed(
        sdist_only: &[(&str, &str)],
        prerelease: &[(&str, &str)],
        original_error: &str,
    ) -> anyhow::Error {
        anyhow::Error::new(HealNeeded {
            sdist_only: sdist_only
                .iter()
                .map(|(n, v)| (n.to_string(), v.to_string()))
                .collect(),
            prerelease: prerelease
                .iter()
                .map(|(n, v)| (n.to_string(), v.to_string()))
                .collect(),
            original_error: original_error.to_string(),
        })
    }

    // ---- offender classification (structured, no prose parsing) ----------

    #[test]
    fn classify_pylock_offenders_splits_sdist_only_and_prerelease() {
        // A Pass-B export with four packages: a plain index wheel (not an
        // offender), an sdist-only package (source dist + no wheels), a
        // pre-release-versioned wheel, and a local/directory source (never
        // an offender).
        let pylock = r#"
lock-version = "1.0"
created-by = "uv"

[[packages]]
name = "requests"
version = "2.32.0"
[[packages.wheels]]
name = "requests-2.32.0-py3-none-any.whl"
url = "https://files.pythonhosted.org/requests-2.32.0-py3-none-any.whl"
[packages.wheels.hashes]
sha256 = "1111111111111111111111111111111111111111111111111111111111111111"

[[packages]]
name = "pyperclip"
version = "1.8.2"
[packages.sdist]
name = "pyperclip-1.8.2.tar.gz"
url = "https://files.pythonhosted.org/pyperclip-1.8.2.tar.gz"
[packages.sdist.hashes]
sha256 = "2222222222222222222222222222222222222222222222222222222222222222"

[[packages]]
name = "tinyobjloader"
version = "2.0.0rc13"
[[packages.wheels]]
name = "tinyobjloader-2.0.0rc13-cp312-cp312-manylinux_2_28_x86_64.whl"
url = "https://files.pythonhosted.org/tinyobjloader-2.0.0rc13.whl"
[packages.wheels.hashes]
sha256 = "3333333333333333333333333333333333333333333333333333333333333333"

[[packages]]
name = "my-editable"
version = "0.1.0"
[packages.directory]
path = "../my-editable"
"#;
        let offenders = classify_pylock_offenders(pylock).unwrap();
        assert_eq!(
            offenders.sdist_only,
            vec![("pyperclip".to_string(), "1.8.2".to_string())],
            "only the source-dist-without-wheels package is sdist-only"
        );
        assert_eq!(
            offenders.prerelease,
            vec![("tinyobjloader".to_string(), "2.0.0rc13".to_string())],
            "only the PEP 440 pre-release version is a prerelease offender"
        );
    }

    #[test]
    fn classify_pylock_offenders_flags_wheelless_prerelease_in_both() {
        // A package that is BOTH sdist-only AND a pre-release lands in
        // both lists (each rung handles its own concern).
        let pylock = r#"
lock-version = "1.0"
created-by = "uv"

[[packages]]
name = "weird"
version = "0.9.0a1"
[packages.sdist]
name = "weird-0.9.0a1.tar.gz"
url = "https://example/weird-0.9.0a1.tar.gz"
[packages.sdist.hashes]
sha256 = "4444444444444444444444444444444444444444444444444444444444444444"
"#;
        let offenders = classify_pylock_offenders(pylock).unwrap();
        assert_eq!(
            offenders.sdist_only,
            vec![("weird".to_string(), "0.9.0a1".to_string())]
        );
        assert_eq!(
            offenders.prerelease,
            vec![("weird".to_string(), "0.9.0a1".to_string())]
        );
    }

    // ---- Pass A / Pass B lock-arg invariant ------------------------------

    /// Multiset of the tokens in `a` not balanced by `b` and vice versa.
    fn arg_symmetric_difference(a: &[String], b: &[String]) -> BTreeSet<String> {
        let mut counts: BTreeMap<String, i64> = BTreeMap::new();
        for t in a {
            *counts.entry(t.clone()).or_default() += 1;
        }
        for t in b {
            *counts.entry(t.clone()).or_default() -= 1;
        }
        counts
            .into_iter()
            .filter(|(_, c)| *c != 0)
            .map(|(t, _)| t)
            .collect()
    }

    /// Drift guard (the coordinator's whack-a-mole close-out): Pass A and
    /// Pass B lock-arg vectors are single-sourced through `build_lock_args`
    /// and may differ ONLY by the known relaxations. Any NEW restrictive
    /// flag added to the shared builder appears in BOTH passes
    /// automatically; a flag added to only one pass (via a new relaxation
    /// field) shows up in the symmetric difference and fails this test
    /// unless deliberately classified here.
    #[test]
    fn pass_a_and_pass_b_differ_only_by_known_relaxations() {
        let dir = Path::new("/tmp/proj");
        let indexes = vec![
            "https://pypi.nvidia.com".to_string(),
            "https://pypi.org/simple/".to_string(),
        ];
        let a = build_lock_args(dir, "3.12", &indexes, false, LockRelaxations::PASS_A);
        let b_auto = build_lock_args(dir, "3.12", &indexes, false, LockRelaxations::PASS_B_AUTO);
        let b_never = build_lock_args(dir, "3.12", &indexes, false, LockRelaxations::PASS_B_NEVER);

        // Pass A vs Pass B (auto): drop `--no-build`, flip prerelease value.
        assert_eq!(
            arg_symmetric_difference(&a, &b_auto),
            BTreeSet::from([
                "--no-build".to_string(),
                "if-necessary-or-explicit".to_string(),
                "allow".to_string(),
            ]),
            "Pass A/B(auto) may differ ONLY by the no-build gate + prerelease value; \
             a new deny-style flag added to only one pass would surface here"
        );
        // Pass A vs Pass B (never): prerelease value only (`--no-build`
        // stays -- build policy is orthogonal to prerelease healing).
        assert_eq!(
            arg_symmetric_difference(&a, &b_never),
            BTreeSet::from(["if-necessary-or-explicit".to_string(), "allow".to_string(),]),
            "under `sdist-build = never`, Pass B keeps --no-build and relaxes only prerelease"
        );
        // `--no-build` really is retained under Never and dropped under Auto.
        assert!(b_never.iter().any(|t| t == "--no-build"));
        assert!(!b_auto.iter().any(|t| t == "--no-build"));
    }

    #[test]
    fn pass_a_lock_args_carry_prerelease_policy() {
        // Regression: the CLI args vector must carry the prerelease policy
        // (UV_NO_CONFIG=1 strips it from the synthesized pyproject table).
        let dir = Path::new("/tmp/proj");
        let a = build_lock_args(dir, "3.12", &[], false, LockRelaxations::PASS_A);
        let idx = a
            .iter()
            .position(|t| t == "--prerelease")
            .expect("--prerelease flag present");
        assert_eq!(
            a.get(idx + 1).map(String::as_str),
            Some("if-necessary-or-explicit")
        );
    }

    // ---- heal ladder driven by the structured verdict --------------------

    /// A non-[`HealNeeded`] error (e.g. a genuine conflict or the
    /// platform-tag ceiling `compute_closure` refuses to two-pass) passes
    /// straight through the heal wrapper: no retry, no probe, no build.
    #[tokio::test]
    async fn heal_passes_through_non_healneeded_errors() {
        let calls = Arc::new(Mutex::new(0usize));
        let solve = {
            let calls = Arc::clone(&calls);
            move |_r: UvClosureRequest| {
                let calls = Arc::clone(&calls);
                Box::pin(async move {
                    *calls.lock().unwrap() += 1;
                    bail!("uv lock failed ... has no wheels with a matching platform tag ...")
                }) as futures::future::BoxFuture<'static, Result<UvClosure>>
            }
        };
        let probe = |_n: String, _s: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let probe_calls = Arc::new(Mutex::new(0usize));
        let sdist_probe = {
            let probe_calls = Arc::clone(&probe_calls);
            move |_n: String, _s: String| {
                let probe_calls = Arc::clone(&probe_calls);
                Box::pin(async move {
                    *probe_calls.lock().unwrap() += 1;
                    None
                }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
            }
        };
        let err = auto_route_fixpoint_with_sdist_heal(
            &auto_route_req(),
            &auto_route_opts(),
            solve,
            probe,
            sdist_probe,
            None::<NoBuild>,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("platform tag"), "{err}");
        assert_eq!(
            *calls.lock().unwrap(),
            1,
            "no retry on a non-HealNeeded error"
        );
        assert_eq!(
            *probe_calls.lock().unwrap(),
            0,
            "no probe on a non-HealNeeded error"
        );
    }

    /// Sdist-only offender: the mock `compute_closure` reports it as a
    /// structured [`HealNeeded`] until a wheel source exists, at which
    /// point the re-solve succeeds. The conda-route rung hits AND the
    /// build rung runs in the SAME round (routing alone can never satisfy
    /// `uv lock`).
    #[tokio::test]
    async fn heal_routes_and_builds_sdist_only_offender() {
        let attempts = Arc::new(Mutex::new(0usize));
        let solve = {
            let attempts = Arc::clone(&attempts);
            move |r: UvClosureRequest| {
                let attempts = Arc::clone(&attempts);
                Box::pin(async move {
                    *attempts.lock().unwrap() += 1;
                    // Model the real resolver: a path source alone is NOT
                    // enough for a transitive dep -- the name must ALSO be
                    // pinned first-party (isaac-pack idna-ssl regression).
                    if !r.built_wheel_sources.contains_key("pyperclip")
                        || r.explicit_pins.get("pyperclip").map(String::as_str) != Some("1.8.2")
                    {
                        return Err(heal_needed(
                            &[("pyperclip", "1.8.2")],
                            &[],
                            "package `pyperclip` has no usable wheels",
                        ));
                    }
                    parse_pylock_closure(
                        PYLOCK_FIXTURE,
                        &target("3.12", "linux-64"),
                        &BTreeSet::new(),
                        "0.11.15",
                    )
                }) as futures::future::BoxFuture<'static, Result<UvClosure>>
            }
        };
        let probe = |_n: String, _s: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let seen_specs = Arc::new(Mutex::new(Vec::new()));
        let sdist_probe = {
            let seen_specs = Arc::clone(&seen_specs);
            move |name: String, spec: String| {
                let seen_specs = Arc::clone(&seen_specs);
                Box::pin(async move {
                    seen_specs.lock().unwrap().push(spec.clone());
                    if name == "pyperclip" {
                        Some(RouteProbeHit {
                            // Deliberately differs from the structured PyPI
                            // offender version: the two version domains must
                            // remain distinct in the persisted route fact.
                            conda_version: "1.8.3".into(),
                            channel: "https://conda.anaconda.org/conda-forge/linux-64".into(),
                            depends: Vec::new(),
                        })
                    } else {
                        None
                    }
                }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
            }
        };
        let seen_build_reqs = Arc::new(Mutex::new(Vec::new()));
        let sdist_build = {
            let seen_build_reqs = Arc::clone(&seen_build_reqs);
            move |name: String, req: Option<String>| {
                let seen_build_reqs = Arc::clone(&seen_build_reqs);
                Box::pin(async move {
                    seen_build_reqs.lock().unwrap().push(req);
                    Ok(BuiltSdistWheel {
                        pypi_name: name.clone(),
                        version: "1.8.2".to_string(),
                        filename: "pyperclip-1.8.2-py3-none-any.whl".to_string(),
                        wheel_path: PathBuf::from("/tmp/wheels/pyperclip-1.8.2-py3-none-any.whl"),
                        sha256: "a".repeat(64),
                        sdist_source: sdist_source_fixture("pyperclip", "1.8.2"),
                    })
                }) as futures::future::BoxFuture<'static, Result<BuiltSdistWheel>>
            }
        };
        let closure = auto_route_fixpoint_with_sdist_heal(
            &auto_route_req(),
            &auto_route_opts(),
            solve,
            probe,
            sdist_probe,
            Some(sdist_build),
        )
        .await
        .unwrap();
        assert_eq!(
            *attempts.lock().unwrap(),
            2,
            "one failure + one healed retry"
        );
        // Both rungs keyed on the EXACT resolved version from Pass B.
        assert_eq!(
            seen_specs.lock().unwrap().as_slice(),
            &["==1.8.2".to_string()]
        );
        assert_eq!(
            seen_build_reqs.lock().unwrap().as_slice(),
            &[Some("==1.8.2".to_string())]
        );
        assert_eq!(closure.auto_routed.len(), 1);
        assert_eq!(closure.auto_routed[0].pypi_name, "pyperclip");
        assert_eq!(closure.auto_routed[0].pypi_version, "1.8.2");
        assert_eq!(closure.auto_routed[0].conda_version, "1.8.3");
        let built = closure
            .wheels
            .iter()
            .find(|w| w.name == "pyperclip")
            .expect("built wheel spliced");
        assert!(matches!(built.origin, Origin::Built));
        assert_eq!(built.version, "1.8.2");
    }

    /// A build FAILURE surfaces the original error plus the build log tail
    /// and never silently drops the dependency.
    #[tokio::test]
    async fn heal_surfaces_build_failure_log_tail() {
        let solve = |_r: UvClosureRequest| {
            Box::pin(async {
                Err(heal_needed(
                    &[("pyperclip", "1.8.2")],
                    &[],
                    "package `pyperclip` has no usable wheels",
                ))
            }) as futures::future::BoxFuture<'static, Result<UvClosure>>
        };
        let probe = |_n: String, _s: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let sdist_probe = |_n: String, _s: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let sdist_build = |_n: String, _r: Option<String>| {
            Box::pin(async {
                bail!("uv [\"build\", \"--wheel\"] failed (status 1): error: missing gcc")
            }) as futures::future::BoxFuture<'static, Result<BuiltSdistWheel>>
        };
        let err = auto_route_fixpoint_with_sdist_heal(
            &auto_route_req(),
            &auto_route_opts(),
            solve,
            probe,
            sdist_probe,
            Some(sdist_build),
        )
        .await
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("has no usable wheels"), "{msg}");
        assert!(msg.contains("sdist auto-build failed"), "{msg}");
        assert!(msg.contains("missing gcc"), "{msg}");
    }

    /// `sdist-build = "never"` with a (defensive) sdist-only offender and
    /// no builder: the original error surfaces, no build attempted.
    #[tokio::test]
    async fn heal_never_policy_surfaces_original_error_for_sdist_only() {
        let solve = |_r: UvClosureRequest| {
            Box::pin(async {
                Err(heal_needed(
                    &[("pyperclip", "1.8.2")],
                    &[],
                    "package `pyperclip` has no usable wheels",
                ))
            }) as futures::future::BoxFuture<'static, Result<UvClosure>>
        };
        let probe = |_n: String, _s: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let sdist_probe = |_n: String, _s: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let err = auto_route_fixpoint_with_sdist_heal(
            &auto_route_req(),
            &auto_route_opts(),
            solve,
            probe,
            sdist_probe,
            None::<NoBuild>,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("has no usable wheels"), "{err}");
    }

    /// Transitive-prerelease offender: the mock reports a [`HealNeeded`]
    /// with a prerelease entry until the request carries an explicit
    /// first-party pin, at which point the re-solve succeeds. Heals even
    /// with the build rung DISABLED (prerelease is orthogonal to
    /// `sdist-build`).
    #[tokio::test]
    async fn heal_pins_transitive_prerelease_explicitly() {
        let seen_pins = Arc::new(Mutex::new(Vec::new()));
        let solve = {
            let seen_pins = Arc::clone(&seen_pins);
            move |r: UvClosureRequest| {
                let seen_pins = Arc::clone(&seen_pins);
                Box::pin(async move {
                    seen_pins.lock().unwrap().push(r.explicit_pins.clone());
                    if r.explicit_pins.get("tinyobjloader").map(String::as_str) == Some("2.0.0rc13")
                    {
                        parse_pylock_closure(
                            PYLOCK_FIXTURE,
                            &target("3.12", "linux-64"),
                            &BTreeSet::new(),
                            "0.11.15",
                        )
                    } else {
                        Err(heal_needed(
                            &[],
                            &[("tinyobjloader", "2.0.0rc13")],
                            "tinyobjloader was requested with a pre-release marker, but pre-releases weren't enabled",
                        ))
                    }
                }) as futures::future::BoxFuture<'static, Result<UvClosure>>
            }
        };
        let probe = |_n: String, _s: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let sdist_probe = |_n: String, _s: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let closure = auto_route_fixpoint_with_sdist_heal(
            &auto_route_req(),
            &auto_route_opts(),
            solve,
            probe,
            sdist_probe,
            None::<NoBuild>,
        )
        .await
        .unwrap();
        // The re-solve saw the explicit first-party pin.
        let pins = seen_pins.lock().unwrap();
        assert!(
            pins.iter()
                .any(|p| p.get("tinyobjloader").map(String::as_str) == Some("2.0.0rc13"))
        );
        // Closure resolved (mock returns the standard fixture on success).
        assert!(!closure.wheels.is_empty());
    }

    /// The prerelease pin renders into the synthesized project's DIRECT
    /// dependencies as an explicit `name==version` requirement (uv's
    /// `explicit` policy only honors first-party pre-release specifiers).
    #[test]
    fn synthesize_pyproject_renders_prerelease_pin_as_first_party_dep() {
        let mut req = sample_request();
        req.explicit_pins
            .insert("tinyobjloader".to_string(), "2.0.0rc13".to_string());
        let got = synthesize_pyproject(&req);
        assert!(
            got.contains("\"tinyobjloader==2.0.0rc13\""),
            "prerelease pin must appear as a direct dependency:\n{got}"
        );
    }

    // ---- live subprocess: two-pass transitive-prerelease heal ------------

    /// Build a minimal, valid py3-none-any wheel (a zip with
    /// METADATA/WHEEL/RECORD -- no build backend needed) into `dir`, and
    /// return the PEP 503 index href for it (`<filename>#sha256=<hash>`) so
    /// uv records a hash in the exported lock (uv leaves `hashes = {}` for
    /// a bare `file://` wheel, which `parse_pylock_closure` rejects).
    fn write_test_wheel(dir: &Path, name: &str, version: &str, requires: &[&str]) -> String {
        use sha2::{Digest, Sha256};
        use std::io::Write as _;
        std::fs::create_dir_all(dir).unwrap();
        let filename = format!("{name}-{version}-py3-none-any.whl");
        let di = format!("{name}-{version}.dist-info");
        let mut metadata = format!("Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n");
        for r in requires {
            metadata.push_str(&format!("Requires-Dist: {r}\n"));
        }
        let path = dir.join(&filename);
        {
            let f = std::fs::File::create(&path).unwrap();
            let mut zip = zip::ZipWriter::new(f);
            let opts = zip::write::SimpleFileOptions::default();
            for (entry, body) in [
                (format!("{di}/METADATA"), metadata.clone()),
                (
                    format!("{di}/WHEEL"),
                    "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n".to_string(),
                ),
                (
                    format!("{di}/RECORD"),
                    format!("{di}/METADATA,,\n{di}/WHEEL,,\n{di}/RECORD,,\n"),
                ),
            ] {
                zip.start_file(&entry, opts).unwrap();
                zip.write_all(body.as_bytes()).unwrap();
            }
            zip.finish().unwrap();
        }
        let hash = format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap()));
        format!("{filename}#sha256={hash}")
    }

    /// End-to-end reproduction of issue #10's transitive-prerelease half,
    /// fully offline (local wheels + uv-managed python). `astub==1.0`
    /// depends on `bstub==1.0rc1` (a TRANSITIVE exact pre-release pin);
    /// `bstub` also publishes a stable `0.9`. Under `--prerelease
    /// if-necessary-or-explicit` Pass A FAILS ("if-necessary" doesn't fire
    /// -- bstub has a stable release -- and "explicit" only honors
    /// first-party pre-release specifiers). The structured two-pass names
    /// `bstub==1.0rc1` from Pass B's export, re-pins it explicitly
    /// first-party, and the heal converges with `bstub` locked at
    /// `1.0rc1`.
    #[tokio::test]
    async fn transitive_prerelease_two_pass_heal_converges_offline() {
        if detect_uv().await.is_err() {
            eprintln!("skipping: uv not found on PATH");
            return;
        }
        let tmp =
            std::env::temp_dir().join(format!("retread-prerelease-heal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let idx = tmp.join("index");
        let bstub_dir = idx.join("bstub");
        let astub_dir = idx.join("astub");
        let b_stable = write_test_wheel(&bstub_dir, "bstub", "0.9", &[]);
        let b_pre = write_test_wheel(&bstub_dir, "bstub", "1.0rc1", &[]);
        let a = write_test_wheel(&astub_dir, "astub", "1.0", &["bstub==1.0rc1"]);
        std::fs::write(
            bstub_dir.join("index.html"),
            format!("<a href=\"{b_stable}\">s</a><a href=\"{b_pre}\">p</a>"),
        )
        .unwrap();
        std::fs::write(
            astub_dir.join("index.html"),
            format!("<a href=\"{a}\">a</a>"),
        )
        .unwrap();
        let index_url = format!("file://{}/", idx.display());

        let mk_req = || UvClosureRequest {
            bundle: "prerelease-smoke".into(),
            python_version: "3.12".into(),
            conda_subdir: "linux-64".into(),
            dependencies: vec!["astub".into()],
            constraints: ConstraintSet::default(),
            overrides: vec![],
            no_emit_packages: vec![],
            index_urls: vec![index_url.clone()],
            built_wheel_sources: BTreeMap::new(),
            explicit_pins: BTreeMap::new(),
            offline: false,
        };

        // Pass A alone reproduces the bug and the two-pass names the
        // offender structurally.
        let err = compute_closure(
            &mk_req(),
            &tmp.join("projA"),
            &tmp.join("cache"),
            None,
            crate::config::SdistBuildPolicy::Auto,
        )
        .await
        .expect_err(
            "Pass A must fail: transitive prerelease not honored under if-necessary-or-explicit",
        );
        let heal = err
            .downcast_ref::<HealNeeded>()
            .unwrap_or_else(|| panic!("expected structured HealNeeded, got: {err:#}"));
        assert!(heal.sdist_only.is_empty(), "no sdist-only offender here");
        assert_eq!(
            heal.prerelease,
            vec![("bstub".to_string(), "1.0rc1".to_string())],
            "Pass B's export must name the transitive prerelease offender"
        );

        // Full heal via the production wrapper converges and pins the
        // prerelease.
        let project = tmp.join("projB");
        let cache = tmp.join("cache");
        let solve = {
            let project = project.clone();
            let cache = cache.clone();
            move |r: UvClosureRequest| {
                let project = project.clone();
                let cache = cache.clone();
                Box::pin(async move {
                    compute_closure(
                        &r,
                        &project,
                        &cache,
                        None,
                        crate::config::SdistBuildPolicy::Auto,
                    )
                    .await
                }) as futures::future::BoxFuture<'static, Result<UvClosure>>
            }
        };
        let probe = |_n: String, _s: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let sdist_probe = |_n: String, _s: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let opts = AutoRouteOptions {
            enabled: true,
            protected: BTreeSet::from(["astub".to_string()]),
            ..Default::default()
        };
        let closure = auto_route_fixpoint_with_sdist_heal(
            &mk_req(),
            &opts,
            solve,
            probe,
            sdist_probe,
            None::<NoBuild>,
        )
        .await
        .expect("two-pass heal must converge");
        assert_eq!(
            closure.pins.get("bstub").map(String::as_str),
            Some("1.0rc1"),
            "final lock must pin bstub at the transitive prerelease"
        );
        assert!(
            closure
                .wheels
                .iter()
                .any(|w| w.name == "bstub" && w.version == "1.0rc1"),
            "bstub's 1.0rc1 wheel must be in the closure"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Write a minimal legacy zip sdist (`<name>-<version>.zip`) carrying
    /// STATIC PEP 643 metadata (`PKG-INFO`, Metadata-Version 2.2) so uv's
    /// Pass B can resolve it WITHOUT invoking a build backend
    /// (offline-safe; zip rather than tar.gz to reuse the existing `zip`
    /// dependency). Returns the PEP 503 index href
    /// (`<filename>#sha256=<hash>`).
    fn write_test_sdist(dir: &Path, name: &str, version: &str) -> String {
        use sha2::{Digest, Sha256};
        use std::io::Write as _;
        std::fs::create_dir_all(dir).unwrap();
        let filename = format!("{name}-{version}.zip");
        let path = dir.join(&filename);
        {
            let f = std::fs::File::create(&path).unwrap();
            let mut zip = zip::ZipWriter::new(f);
            zip.start_file(
                format!("{name}-{version}/PKG-INFO"),
                zip::write::SimpleFileOptions::default(),
            )
            .unwrap();
            zip.write_all(
                format!("Metadata-Version: 2.2\nName: {name}\nVersion: {version}\n").as_bytes(),
            )
            .unwrap();
            zip.finish().unwrap();
        }
        let hash = format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap()));
        format!("{filename}#sha256={hash}")
    }

    /// End-to-end reproduction of issue #10's sdist-only half AND the
    /// isaac-pack return-path regression, fully offline, at the REAL
    /// `compute_closure` seam (the layer that answered the failing RPC).
    /// `astub==1.0` (wheel) depends on `tstub==1.0`, which the index
    /// carries ONLY as an sdist -- Pass A fails with uv's "building from
    /// source is disabled" hint (the exact issue-#10 phrasing the old
    /// regexes missed), Pass B resolves via the sdist's static metadata
    /// and names `tstub` structurally, rung 2 "builds" a wheel (mock
    /// builder writing a REAL wheel file), and the healed relock must
    /// succeed -- proving the FUNCTION RETURN is the healed closure, not
    /// just on-disk artifacts. Regression guard for the isaac-pack repro
    /// where the built wheel's `[tool.uv.sources]` entry alone was
    /// invisible to the resolver for a TRANSITIVE dependency (no
    /// first-party pin), the healed relock re-failed identically, and the
    /// heal surfaced Pass A's error to the RPC despite healed artifacts
    /// sitting on disk.
    #[tokio::test]
    async fn sdist_only_transitive_two_pass_heal_returns_healed_closure_offline() {
        if detect_uv().await.is_err() {
            eprintln!("skipping: uv not found on PATH");
            return;
        }
        let tmp = std::env::temp_dir().join(format!("retread-sdist-heal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let idx = tmp.join("index");
        let astub_dir = idx.join("astub");
        let tstub_dir = idx.join("tstub");
        let a = write_test_wheel(&astub_dir, "astub", "1.0", &["tstub==1.0"]);
        let t = write_test_sdist(&tstub_dir, "tstub", "1.0");
        std::fs::write(
            astub_dir.join("index.html"),
            format!("<a href=\"{a}\">a</a>"),
        )
        .unwrap();
        std::fs::write(
            tstub_dir.join("index.html"),
            format!("<a href=\"{t}\">t</a>"),
        )
        .unwrap();
        let index_url = format!("file://{}/", idx.display());

        let req = UvClosureRequest {
            bundle: "sdist-smoke".into(),
            python_version: "3.12".into(),
            conda_subdir: "linux-64".into(),
            dependencies: vec!["astub".into()],
            constraints: ConstraintSet::default(),
            overrides: vec![],
            no_emit_packages: vec![],
            index_urls: vec![index_url],
            built_wheel_sources: BTreeMap::new(),
            explicit_pins: BTreeMap::new(),
            offline: false,
        };
        let project = tmp.join("proj");
        let cache = tmp.join("cache");
        let solve = {
            let project = project.clone();
            let cache = cache.clone();
            move |r: UvClosureRequest| {
                let project = project.clone();
                let cache = cache.clone();
                Box::pin(async move {
                    compute_closure(
                        &r,
                        &project,
                        &cache,
                        None,
                        crate::config::SdistBuildPolicy::Auto,
                    )
                    .await
                }) as futures::future::BoxFuture<'static, Result<UvClosure>>
            }
        };
        let probe = |_n: String, _s: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let sdist_probe = |_n: String, _s: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        // Rung-2 builder: writes a REAL wheel file (what
        // `handler::build_sdist_wheel` would produce from the sdist), so
        // the relock's `[tool.uv.sources]` path + first-party pin resolve
        // against an actual artifact.
        let wheels_dir = tmp.join("built-wheels");
        let sdist_build = {
            let wheels_dir = wheels_dir.clone();
            move |name: String, req: Option<String>| {
                let wheels_dir = wheels_dir.clone();
                Box::pin(async move {
                    assert_eq!(name, "tstub");
                    assert_eq!(
                        req.as_deref(),
                        Some("==1.0"),
                        "build rung must receive the exact Pass-B-resolved version"
                    );
                    let href = write_test_wheel(&wheels_dir, "tstub", "1.0", &[]);
                    let filename = href.split('#').next().unwrap().to_string();
                    Ok(BuiltSdistWheel {
                        pypi_name: "tstub".to_string(),
                        version: "1.0".to_string(),
                        wheel_path: wheels_dir.join(&filename),
                        sha256: href.split("sha256=").nth(1).unwrap().to_string(),
                        filename,
                        sdist_source: sdist_source_fixture("tstub", "1.0"),
                    })
                }) as futures::future::BoxFuture<'static, Result<BuiltSdistWheel>>
            }
        };
        let opts = AutoRouteOptions {
            enabled: true,
            protected: BTreeSet::from(["astub".to_string()]),
            ..Default::default()
        };
        let closure = auto_route_fixpoint_with_sdist_heal(
            &req,
            &opts,
            solve,
            probe,
            sdist_probe,
            Some(sdist_build),
        )
        .await
        .expect("the RETURNED result must be the healed closure, not Pass A's error");
        assert_eq!(
            closure.pins.get("tstub").map(String::as_str),
            Some("1.0"),
            "healed closure must pin the sdist-only transitive dep"
        );
        let built = closure
            .wheels
            .iter()
            .find(|w| w.name == "tstub")
            .expect("built wheel spliced into the returned closure");
        assert!(matches!(built.origin, Origin::Built));
        assert!(built.must_ship);
        assert!(
            closure.wheels.iter().any(|w| w.name == "astub"),
            "the requirer's own index wheel stays in the closure"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // -- issue #10 perf: closure-reuse fingerprint stability -------------

    #[test]
    fn fingerprint_is_stable_for_identical_inputs() {
        // The whole cross-run reuse guard hinges on this: identical
        // resolution inputs must hash to the same fingerprint, or the
        // healed uv.lock is discarded and re-resolved every run.
        let args = vec!["lock".to_string(), "--no-build".to_string()];
        let export = vec!["export".to_string()];
        let a = closure_inputs_fingerprint("[project]\nname='x'\n", &args, &export, "0.11.0");
        let b = closure_inputs_fingerprint("[project]\nname='x'\n", &args, &export, "0.11.0");
        assert_eq!(a, b);
    }

    #[test]
    fn fingerprint_changes_when_manifest_changes() {
        let args = vec!["lock".to_string()];
        let export = vec!["export".to_string()];
        let base = closure_inputs_fingerprint("deps=['a==1']", &args, &export, "0.11.0");
        // A changed synthesized pyproject (e.g. a new explicit pin) must
        // invalidate: otherwise a pinned lock would be reused for a
        // different (pinless) request.
        assert_ne!(
            base,
            closure_inputs_fingerprint("deps=['a==2']", &args, &export, "0.11.0")
        );
        // A changed flag vector (index set, prerelease policy) invalidates.
        assert_ne!(
            base,
            closure_inputs_fingerprint(
                "deps=['a==1']",
                &["lock".into(), "--x".into()],
                &export,
                "0.11.0"
            )
        );
        // A changed EXPORT vector (--no-emit-package set) invalidates: the
        // full-skip path reuses the exported pylock, whose contents depend
        // on the exclusion list.
        assert_ne!(
            base,
            closure_inputs_fingerprint(
                "deps=['a==1']",
                &args,
                &["export".into(), "--no-emit-package".into(), "x".into()],
                "0.11.0"
            )
        );
        // A uv upgrade invalidates.
        assert_ne!(
            base,
            closure_inputs_fingerprint("deps=['a==1']", &args, &export, "0.12.0")
        );
    }

    // -- issue #10 perf: persisted heal-facts round-trip -----------------

    #[test]
    fn heal_facts_round_trip_and_stale_wheel_dropped() {
        let tmp = std::env::temp_dir().join(format!("retread-healfacts-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // Facts path is a SIBLING of uv-projects under the cache dir; keyed
        // per (bundle, python, subdir). Verify the layout survives being
        // outside any uv project dir.
        let facts_file = heal_facts_path(&tmp, "isaac-pack", "3.12", "linux-64");
        assert!(facts_file.starts_with(tmp.join("retread-heal-facts")));

        // A real wheel on disk (kept) and a phantom path (dropped on load).
        let live_wheel = tmp.join("live-1.0-py3-none-any.whl");
        std::fs::write(&live_wheel, b"pk").unwrap();
        let facts = HealFacts {
            stamp: "stamp-under-state-1".to_string(),
            workspace_overrides: vec![],
            routed: vec![AutoRoutedPackage {
                pypi_name: "routed-pkg".into(),
                conda_name: "routed-pkg".into(),
                pypi_version: "1.0".into(),
                conda_version: "1.0".into(),
                channel: "conda-forge".into(),
                input_requirements: vec![AutoRouteInputRequirement {
                    specifiers: ">=0.9,<2".into(),
                    source: "uv constraint `routed-pkg>=0.9,<2`".into(),
                    role: AutoRouteInputRole::Constraint,
                }],
            }],
            built: vec![
                BuiltSdistWheel {
                    pypi_name: "live".into(),
                    version: "1.0".into(),
                    filename: "live-1.0-py3-none-any.whl".into(),
                    wheel_path: live_wheel.clone(),
                    sha256: "a".repeat(64),
                    sdist_source: sdist_source_fixture("live", "1.0"),
                },
                BuiltSdistWheel {
                    pypi_name: "gone".into(),
                    version: "2.0".into(),
                    filename: "gone-2.0-py3-none-any.whl".into(),
                    wheel_path: tmp.join("does-not-exist.whl"),
                    sha256: "b".repeat(64),
                    sdist_source: sdist_source_fixture("gone", "2.0"),
                },
            ],
            prereleased: vec![PrereleasePin {
                pypi_name: "pre".into(),
                version: "3.0rc1".into(),
            }],
        };

        save_heal_facts(&facts_file, &facts);
        let loaded = load_heal_facts(&facts_file, "stamp-under-state-1");
        assert_eq!(loaded.routed.len(), 1);
        assert_eq!(
            loaded.routed[0].input_requirements, facts.routed[0].input_requirements,
            "typed restore provenance must survive persisted heal-facts"
        );
        assert_eq!(loaded.prereleased.len(), 1);
        // Stale built-wheel (missing from store) is dropped; live one kept.
        assert_eq!(loaded.built.len(), 1);
        assert_eq!(loaded.built[0].pypi_name, "live");

        // B1 (a)/(c): facts stamped under manifest/routing state 1 are
        // DISCARDED WHOLESALE when loaded under state 2 -- no partial
        // replay of pins learned under a different manifest.
        let stale = load_heal_facts(&facts_file, "stamp-under-state-2");
        assert!(
            stale.is_empty(),
            "stamp mismatch must discard all facts, got {stale:?}"
        );

        // Saving empty facts removes the file (a pack that stopped needing
        // a heal must not keep injecting dead pins).
        save_heal_facts(&facts_file, &HealFacts::default());
        assert!(load_heal_facts(&facts_file, "stamp-under-state-1").is_empty());
        assert!(!facts_file.exists());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn workspace_fact_override_only_heal_facts_round_trip() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-workspace-overrides-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        let facts_file = heal_facts_path(&tmp, "isaac-pack", "3.12", "linux-64");
        let facts = HealFacts {
            stamp: "workspace-state-1".to_string(),
            workspace_overrides: vec![
                WorkspaceFactOverride {
                    pypi_name: "torch".to_string(),
                    version: "2.10.0".to_string(),
                    drop: false,
                },
                WorkspaceFactOverride {
                    pypi_name: "torchvision".to_string(),
                    version: "0.25.0".to_string(),
                    drop: false,
                },
                WorkspaceFactOverride {
                    pypi_name: "torchaudio".to_string(),
                    version: "2.10.0".to_string(),
                    drop: false,
                },
            ],
            ..Default::default()
        };
        assert!(!facts.is_empty(), "override-only facts must be persisted");
        save_heal_facts(&facts_file, &facts);

        let loaded = load_heal_facts(&facts_file, "workspace-state-1");
        assert_eq!(loaded.workspace_overrides, facts.workspace_overrides);
        assert!(loaded.routed.is_empty());
        assert!(loaded.built.is_empty());
        assert!(loaded.prereleased.is_empty());

        let stale = load_heal_facts(&facts_file, "workspace-state-2");
        assert!(
            stale.is_empty(),
            "workspace solution changes must discard learned overrides"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// B1: the stamp must move with every fact-validity input class --
    /// manifest deps, constraints, overrides, and each routing knob.
    #[test]
    fn heal_facts_stamp_tracks_resolution_inputs() {
        let base_req = || UvClosureRequest {
            bundle: "b".into(),
            python_version: "3.12".into(),
            conda_subdir: "linux-64".into(),
            dependencies: vec!["foo==1.0".into()],
            constraints: ConstraintSet {
                constraints: vec!["bar<2".into()],
                provenance: BTreeMap::new(),
                auto_route_constraint_indices: BTreeSet::new(),
            },
            overrides: vec!["baz>=1".into()],
            no_emit_packages: vec!["python".into()],
            index_urls: vec!["https://pypi.org/simple/".into()],
            built_wheel_sources: BTreeMap::new(),
            explicit_pins: BTreeMap::new(),
            offline: false,
        };
        let base_opts = || AutoRouteOptions {
            enabled: true,
            route_policy: crate::config::RoutePolicy::Minimal,
            route_include: ["inc".to_string()].into(),
            keep_pypi: ["kp".to_string()].into(),
            protected: ["root".to_string()].into(),
            name_map: BTreeMap::new(),
            force_conda: BTreeSet::new(),
            abi_anchor_pins: BTreeMap::new(),
            workspace_conda_versions: BTreeMap::new(),
            workspace_owned: BTreeSet::new(),
            workspace_fact_fingerprint: String::new(),
        };
        // Holds the `sdist-build` policy fixed at the default while probing
        // the manifest/routing input classes; the policy gets its own
        // invariant below.
        let stamp = |req: &UvClosureRequest, opts: &AutoRouteOptions| {
            heal_facts_stamp(req, opts, crate::config::SdistBuildPolicy::Auto)
        };
        let base = stamp(&base_req(), &base_opts());
        // Identical inputs -> identical stamp.
        assert_eq!(base, stamp(&base_req(), &base_opts()));

        // (a) manifest dep change (add/remove/bump) invalidates.
        let mut r = base_req();
        r.dependencies = vec!["foo==2.0".into()];
        assert_ne!(base, stamp(&r, &base_opts()));
        let mut r = base_req();
        r.dependencies.clear();
        assert_ne!(base, stamp(&r, &base_opts()));
        // Constraint / override change invalidates.
        let mut r = base_req();
        r.constraints.constraints = vec!["bar<3".into()];
        assert_ne!(base, stamp(&r, &base_opts()));
        let mut r = base_req();
        r.overrides.clear();
        assert_ne!(base, stamp(&r, &base_opts()));

        // (c) each routing knob invalidates.
        let mut o = base_opts();
        o.route_policy = crate::config::RoutePolicy::Aggressive;
        assert_ne!(base, stamp(&base_req(), &o));
        let mut o = base_opts();
        o.keep_pypi.insert("torch".into());
        assert_ne!(base, stamp(&base_req(), &o));
        let mut o = base_opts();
        o.force_conda.insert("numpy".into());
        assert_ne!(base, stamp(&base_req(), &o));
        let mut o = base_opts();
        o.route_include.insert("scipy".into());
        assert_ne!(base, stamp(&base_req(), &o));
        let mut o = base_opts();
        o.name_map.insert("torch".into(), "pytorch".into());
        assert_ne!(base, stamp(&base_req(), &o));
        // Python bump invalidates.
        let mut r = base_req();
        r.python_version = "3.13".into();
        assert_ne!(base, stamp(&r, &base_opts()));

        // (d) the `sdist-build` policy is a fact-validity input: it selects
        // the Pass-B relaxation, so facts learned under `auto` (which may
        // build/persist sdist wheels) must not replay under `never`, and
        // vice versa. Two policies over IDENTICAL request+routing must yield
        // DIFFERENT stamps.
        let auto = heal_facts_stamp(
            &base_req(),
            &base_opts(),
            crate::config::SdistBuildPolicy::Auto,
        );
        let never = heal_facts_stamp(
            &base_req(),
            &base_opts(),
            crate::config::SdistBuildPolicy::Never,
        );
        assert_eq!(base, auto, "default policy stamp must equal the Auto stamp");
        assert_ne!(
            auto, never,
            "sdist-build policy change must move the heal-facts stamp",
        );
    }

    /// B1 (b) wedge self-recovery: a SEEDED built-wheel fact at a stale
    /// version (bar 1.5, e.g. loaded from a facts file written before a
    /// dependency bump) must not suppress the heal when the solve fails
    /// needing bar 2.5 -- the stale ledger entry is dropped, bar is rebuilt
    /// at 2.5, and the retry succeeds with the fresh fact in the ledger.
    #[tokio::test]
    async fn stale_seeded_built_fact_is_dropped_and_rehealed() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let attempts = Arc::new(AtomicUsize::new(0));
        let solve = {
            let attempts = Arc::clone(&attempts);
            move |req: UvClosureRequest| {
                let n = attempts.fetch_add(1, Ordering::SeqCst);
                let fut = async move {
                    if n == 0 {
                        // First attempt: the stale bar==1.5 pin is in the
                        // request (seeded), and the solve fails naming bar
                        // 2.5 as the sdist-only offender.
                        assert_eq!(
                            req.explicit_pins.get("bar").map(String::as_str),
                            Some("1.5"),
                            "seeded stale fact must be injected on attempt 0"
                        );
                        Err(anyhow::Error::new(HealNeeded {
                            sdist_only: vec![("bar".to_string(), "2.5".to_string())],
                            prerelease: vec![],
                            original_error: "bar needs 2.5".to_string(),
                        }))
                    } else {
                        // Retry: the stale pin must be REPLACED by 2.5.
                        assert_eq!(
                            req.explicit_pins.get("bar").map(String::as_str),
                            Some("2.5"),
                            "retry must carry the freshly-healed pin"
                        );
                        Ok(UvClosure {
                            wheels: vec![],
                            pins: BTreeMap::new(),
                            auto_routed: vec![],
                            auto_dropped: BTreeSet::new(),
                            uv_version: "test".into(),
                            effective_input_requirements: None,
                        })
                    }
                };
                Box::pin(fut) as futures::future::BoxFuture<'static, Result<UvClosure>>
            }
        };
        let sdist_probe = |_n: String, _s: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let sdist_build = Some(|name: String, req: Option<String>| {
            Box::pin(async move {
                assert_eq!(req.as_deref(), Some("==2.5"));
                Ok(BuiltSdistWheel {
                    pypi_name: name,
                    version: "2.5".to_string(),
                    filename: "bar-2.5-py3-none-any.whl".to_string(),
                    wheel_path: PathBuf::from("/nonexistent/bar-2.5-py3-none-any.whl"),
                    sha256: "c".repeat(64),
                    sdist_source: sdist_source_fixture("bar", "2.5"),
                })
            }) as futures::future::BoxFuture<'static, Result<BuiltSdistWheel>>
        });

        // Ledger seeded with the STALE persisted fact: bar 1.5.
        let built = Arc::new(std::sync::Mutex::new(vec![BuiltSdistWheel {
            pypi_name: "bar".to_string(),
            version: "1.5".to_string(),
            filename: "bar-1.5-py3-none-any.whl".to_string(),
            wheel_path: PathBuf::from("/nonexistent/bar-1.5-py3-none-any.whl"),
            sha256: "d".repeat(64),
            sdist_source: sdist_source_fixture("bar", "1.5"),
        }]));
        let routed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let prereleased = Arc::new(std::sync::Mutex::new(Vec::new()));

        let mut healing = with_sdist_heal(
            "wedge-test".to_string(),
            solve,
            sdist_probe,
            sdist_build,
            Arc::clone(&routed),
            Arc::clone(&built),
            Arc::clone(&prereleased),
        );
        let req = UvClosureRequest {
            bundle: "wedge-test".into(),
            python_version: "3.12".into(),
            conda_subdir: "linux-64".into(),
            dependencies: vec!["foo==2.0".into()],
            constraints: ConstraintSet::default(),
            overrides: vec![],
            no_emit_packages: vec![],
            index_urls: vec![],
            built_wheel_sources: BTreeMap::new(),
            explicit_pins: BTreeMap::new(),
            offline: false,
        };
        healing(req).await.expect("wedge must self-recover");
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            2,
            "fail once, heal, succeed"
        );
        // Ledger now holds ONLY the fresh fact.
        let b = built.lock().unwrap();
        assert_eq!(b.len(), 1);
        assert_eq!(
            (b[0].pypi_name.as_str(), b[0].version.as_str()),
            ("bar", "2.5")
        );
    }
}
