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
//! attributed to the offending conda pin and `retread solve` knows which
//! pin to widen. This layer is policy-free: it never widens, never retries
//! with altered inputs (spec §4c) — on conflict it reports and points at
//! `retread solve`.
//!
//! uv is the only resolver: this closure computation runs unconditionally
//! for every bundle with uv-resolvable roots (v4.4.0; the `retread-resolver`
//! knob and the historical cascade/resolvo mirror-solver are both gone).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

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
}

/// Generated constraint lines + their provenance, keyed by PyPI name.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstraintSet {
    /// PEP 508 constraint lines for `[tool.uv] constraint-dependencies`.
    pub constraints: Vec<String>,
    /// PyPI name -> provenance for every line in `constraints`.
    pub provenance: BTreeMap<String, ConstraintProvenance>,
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
}

// ---------------------------------------------------------------------------
// Auto-route (spec-uv-restructure M2): probe-driven conda routing filter
// ---------------------------------------------------------------------------

/// One auto-routed package: excluded from the wheel closure, provided by
/// a conda channel instead. Recorded so the backend emits it as a conda
/// run-dependency of the stub package.
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
}

/// Configuration for the auto-route loop.
#[derive(Debug, Clone, Default)]
pub struct AutoRouteOptions {
    /// Master switch (`auto-route` in `[package.build.config]`; default on).
    pub enabled: bool,
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
) -> Vec<AutoRoutedPackage> {
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
            });
        }
    }
    out
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
        let mut parts = trimmed.splitn(2, char::is_whitespace);
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
        if crate::solve::is_abi_anchor(&dep_name)
            && let Some(anchor_spec) = abi_anchor_pins.get(&dep_name)
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
        req.constraints.constraints.push(line.clone());
        req.constraints.provenance.insert(
            h.pypi_name.clone(),
            ConstraintProvenance {
                constraint: line,
                conda_name: h.conda_name.clone(),
                conda_version: format!("=={}", h.conda_version),
                source: "auto-route".to_string(),
                env: "default".to_string(),
            },
        );
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
        let new_routes = plan_auto_route_round(&closure, &req, &eff_opts, &routed, &hits);
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
                for pkg in &named {
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
                if routed.len() != previously_routed {
                    // A previously ACCEPTED route was un-routed: rebuild
                    // the request from scratch (its exclusion +
                    // constraint must disappear) and re-lock so the
                    // package's wheel — and any transitives it drags in —
                    // rejoin the closure.
                    req = base_req.clone();
                    apply_auto_route(&mut req, &routed);
                    closure = solve(req.clone()).await?;
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
                    req = base_req.clone();
                    apply_auto_route(&mut req, &routed);
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
    closure.auto_routed = routed;
    Ok(closure)
}

// ---------------------------------------------------------------------------
// Self-heal: sdist-only packages (no wheels at all under `no-build`)
// ---------------------------------------------------------------------------
//
// A THIRD class of `uv lock` failure, alongside the ordinary conflict
// (`format_lock_failure`) and the manylinux platform-tag ceiling
// (`installer::is_platform_tag_conflict`, glibc relaxation): a package
// that publishes NO wheels at all (pure sdist, e.g. pyperclip). Under
// `no-build = true` (spec §8.2 — retread's source_build path owns
// builds, uv never builds an sdist) this fails the WHOLE lock loudly,
// even when the sdist-only package is a deep transitive nobody asked
// for by name.
//
// uv phrases this "<pkg> has no wheels" (no tag qualifier) or "<pkg>
// has no wheels with a matching <foo> tag" for some other unsatisfied
// tag axis (python/abi). This is DISTINCT from the manylinux ceiling,
// which uv always phrases as "...matching platform tag..." — that
// case has its own recovery path (glibc relax) and must never be
// mistaken for "no conda route exists at all".

/// True when `text` (typically `uv lock` stderr) names a package with
/// literally zero published wheels as the resolver failure — the
/// sdist-only class this module self-heals. Returns `false` for the
/// manylinux platform-tag ceiling (`installer::is_platform_tag_conflict`
/// owns that recovery), even though both mention "no wheels": the tag
/// case always says "platform tag" and is excluded here first.
/// Whitespace-collapsed before matching (uv line-wraps its resolver
/// prose across output lines).
pub fn is_sdist_only_uv_error(text: &str) -> bool {
    let t = text
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if t.contains("platform tag") {
        return false;
    }
    // "has no usable wheels" is uv's phrasing when a NAMED VERSION RANGE
    // (not the bare package) is the subject -- e.g. a transitive exact
    // pin with a `.*` wildcard range collapses to "pkg>=X,<=Y has no
    // usable wheels" rather than the bare-package "pkg has no wheels"
    // shape below (deps-from proof run 3: `hydra-core==1.3.2` pins
    // `antlr4-python3-runtime==4.9.*`, which resolves to a range with
    // zero wheel-bearing builds under `--no-build`). Same sdist-only
    // class, different uv prose shape -- must self-heal identically.
    if t.contains("has no wheels")
        || t.contains("no wheels with a matching")
        || t.contains("has no usable wheels")
    {
        return true;
    }
    // "only the following versions ... are available" band-complement
    // shape (deps-from proof run 7, step4-lock-run7.log): once a
    // wheel-less version band is filtered OUT of uv's candidate set
    // entirely (e.g. `--no-build` hides every 4.9.x sdist-only release
    // while a conda pin still demands one), uv stops saying "has no
    // (usable) wheels" at all and instead lists the AVAILABLE set as
    // the band's complement -- "only the following versions of
    // antlr4-python3-runtime are available: antlr4-python3-
    // runtime<4.9.dev0, antlr4-python3-runtime>=4.10.dev0". The
    // `.dev0` range boundaries are the tell that an interior band of
    // versions EXISTS upstream but was filtered (a genuinely-missing
    // version is listed as plain released versions/ranges, never as a
    // dev-bounded complement), so this is the same sdist-only class in
    // a third prose shape. A rare misfire is benign: rung 2's sdist
    // resolve fails for a version that truly doesn't exist and rung 3
    // surfaces the ORIGINAL error verbatim (plus guidance).
    t.contains("only the following versions of")
        && t.contains("are available")
        && t.contains(".dev0")
}

/// Extract the PEP 503-canonical package name(s) uv's error names as
/// sdist-only, e.g. from `"Because pyperclip has no wheels ... we can
/// conclude..."` -> `["pyperclip"]`. Empty when [`is_sdist_only_uv_error`]
/// is false (including the platform-tag class). Best-effort: an
/// unparseable message that still matches the class returns an empty
/// list rather than guessing, so the caller falls through to surfacing
/// the original error (never silently drops a dependency it couldn't
/// name).
pub fn extract_sdist_only_packages(text: &str) -> Vec<String> {
    if !is_sdist_only_uv_error(text) {
        return Vec::new();
    }
    let t = text.split_whitespace().collect::<Vec<_>>().join(" ");
    // uv's resolver prose: "Because <name>[extra]==version has no wheels".
    // `-`/`_`/`.` all appear in real PyPI names; extras and version
    // pins are optional and dropped.
    let re = regex::Regex::new(
        r"(?i)because\s+([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?(?:[=><!~][=~<>0-9.a-zA-Z*, ]*)?\s+has no wheels",
    )
    .expect("static sdist-only extraction regex");
    let mut names: BTreeSet<String> = BTreeSet::new();
    for cap in re.captures_iter(&t) {
        names.insert(canonical_conda_name(&cap[1]));
    }
    // "has no usable wheels" shape: the subject is a RANGE, not the bare
    // "Because <name>" clause above (e.g. "...and antlr4-python3-
    // runtime>=4.9,<=4.9.3 has no usable wheels, we can conclude..." --
    // the package name sits directly before the version range, not right
    // after "Because"). Anchor on the range/operator immediately
    // preceding the phrase instead.
    let re_ranged = regex::Regex::new(
        r"(?i)([A-Za-z0-9][A-Za-z0-9._-]*)(?:>=|<=|==|>|<)\S*\s+has no usable wheels",
    )
    .expect("static ranged sdist-only extraction regex");
    for cap in re_ranged.captures_iter(&t) {
        names.insert(canonical_conda_name(&cap[1]));
    }
    // Band-complement shape (run 7): the failing package is the subject
    // of "only the following versions of <name> are available" -- no
    // "has no (usable) wheels" clause exists anywhere in the message,
    // so neither regex above can name it.
    let re_avail = regex::Regex::new(
        r"(?i)only the following versions of\s+([A-Za-z0-9][A-Za-z0-9._-]*)\s+are\s+available",
    )
    .expect("static available-band sdist-only extraction regex");
    for cap in re_avail.captures_iter(&t) {
        names.insert(canonical_conda_name(&cap[1]));
    }
    names.into_iter().collect()
}

/// Find the ORIGINATING pypi requirement's raw version specifier for
/// `name` in a sdist-only `uv lock` error, e.g. from `"...hydra-core==1.3.2
/// depends on antlr4-python3-runtime==4.9.* and your project depends on
/// hydra-core==1.3.2..."` with `name = "antlr4-python3-runtime"` ->
/// `Some("==4.9.*")`. `name` is matched separator-insensitively (`-`/`_`
/// both fold to a shared class) since uv's prose preserves the
/// requirer's original spelling, which may differ from the
/// PEP 503-canonical form [`extract_sdist_only_packages`] returns.
/// `None` when no "depends on <name><specifier>" clause is found (either
/// no specifier at all -- an unpinned dependency -- or the shape isn't
/// present), which callers treat as "unpinned" (probe any version).
pub fn extract_sdist_only_requirement(text: &str, name: &str) -> Option<String> {
    let t = text.split_whitespace().collect::<Vec<_>>().join(" ");
    // Escape each separator-delimited segment BEFORE inserting the
    // `[-_]` character class -- escaping the whole name first (e.g.
    // `regex::escape("antlr4-python3-runtime")` -> `antlr4\-python3\-
    // runtime`) and then substituting `-` would land the class right
    // after a stray backslash (`\[-_]`), which the regex engine reads
    // as an escaped literal `[`, not a character class.
    let name_pat = name
        .split(['-', '_'])
        .map(regex::escape)
        .collect::<Vec<_>>()
        .join("[-_]");
    let re = regex::Regex::new(&format!(
        r"(?i)depends on\s+{name_pat}((?:[=><!~]=?|==)\S*)"
    ))
    .ok()?;
    re.captures(&t)
        .and_then(|c| c.get(1))
        // uv's prose may run the specifier straight into sentence
        // punctuation ("...depends on antlr4-python3-runtime==4.9.*, we
        // can conclude..." -- run 7's band-complement shape), which the
        // greedy `\S*` capture swallows. A PEP 440 specifier never ENDS
        // with `,` (comma is the AND separator between clauses) or `.`
        // (a version segment always follows a dot), so trailing sentence
        // punctuation is safe to strip.
        .map(|m| m.as_str().trim_end_matches([',', '.', ';']).to_string())
}

/// Translate a raw PyPI/PEP 440 version specifier (as captured by
/// [`extract_sdist_only_requirement`]) into the equivalent conda
/// match-spec version fragment, so a conda-route probe for a
/// wheel-less transitive dependency is constrained to the SAME range
/// the originating pypi requirement demanded (rather than resolving to
/// conda's unconstrained latest, which can clash with the pypi
/// requirement's own metadata -- deps-from proof run 5:
/// `antlr4-python3-runtime` routed to conda `==4.13.2` while
/// hydra-core's metadata still required `==4.9.*`).
///
/// Translation rules (conda match-spec version syntax is a near-superset
/// of PEP 440 for the forms uv's error prose actually emits):
/// * `None` (no specifier captured -- unpinned dependency) -> `"*"`
///   (any version, the pre-existing probe behavior).
/// * `==X.Y.*` (wildcard/prefix match) -> `X.Y.*` (conda's own prefix
///   wildcard syntax; the `==` prefix is conda-invalid and dropped).
/// * `==X.Y.Z` (exact pin, no wildcard) -> `==X.Y.Z` (conda accepts `==`
///   for an exact match; passed through unchanged).
/// * Comma-joined range clauses (`>=A,<B`, `>=A,<=B`, etc.) -> passed
///   through unchanged; conda match-spec ANDs comma-separated clauses
///   exactly like PEP 440 does.
/// * Anything else unrecognized -> passed through unchanged (best
///   effort: an over-constrained but syntactically-invalid spec fails
///   the probe closed, i.e. routes nothing, rather than silently
///   widening to conda's latest).
pub fn conda_spec_from_pypi_specifier(spec: Option<&str>) -> String {
    let Some(spec) = spec else {
        return "*".to_string();
    };
    let spec = spec.trim();
    if spec.is_empty() {
        return "*".to_string();
    }
    if let Some(rest) = spec.strip_prefix("==")
        && rest.ends_with(".*")
    {
        return rest.to_string();
    }
    spec.to_string()
}

/// Guidance appended to the ORIGINAL uv error (verbatim, never
/// paraphrased — spec convention shared with [`format_lock_failure`])
/// when one or more sdist-only packages have no conda candidate
/// either AND the sdist auto-build rung is disabled (`sdist-build =
/// "never"`) or was never reached. Never silently drops the
/// dependency: the caller must choose.
pub fn sdist_only_no_route_message(names: &[String]) -> String {
    let list = names.join(", ");
    format!(
        "\npackage {list} has no wheels and no conda candidate; options: allow \
         build (no-build = false), drop-dep, or vendor a wheel.\n"
    )
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
#[derive(Debug, Clone)]
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

/// Wrap a `solve` closure with the sdist-only self-heal ladder: on a
/// `uv lock` failure naming the sdist-only class
/// ([`is_sdist_only_uv_error`]), each named package climbs a THREE-rung
/// ladder before the caller ever sees an error:
///
/// 1. **conda-route** (pre-existing): probed against the workspace conda
///    channels for ANY compatible version (`sdist_probe`; unlike the
///    exact-version auto-route round, there is no resolved pypi version
///    to probe AT — uv never produced a closure). A hit routes the
///    package to conda via the same mechanism as an ordinary auto-route
///    hit (exclude from the closure at `uv export` time + pin/range on
///    the conda-resolved version). **This alone can never satisfy `uv
///    lock`**: `--no-emit-package` is an `export`-only flag
///    (`compute_closure` never passes `req.no_emit_packages` to the
///    `lock` subcommand), so a package with zero usable PyPI wheels in
///    its required range fails `uv lock` identically whether or not it
///    is conda-routed. Rung 1 exists purely to keep the routed package
///    out of the final exported closure (conda already provides it) —
///    it is NOT an alternative to rung 2, it runs alongside it.
/// 2. **sdist auto-build**: attempted for EVERY named package this
///    round — including ones rung 1 just routed or routed in an earlier
///    round — when `sdist_build` is `Some` (config `sdist-build =
///    "auto"`, the default). The package is built from its PyPI sdist
///    via `sdist_build`, which receives the ORIGINATING pypi
///    requirement's raw version specifier (the same one rung 1's conda
///    probe uses, e.g. `==4.9.*`; `None` = unpinned = newest) so the
///    built wheel actually SATISFIES the requirer's metadata — building
///    the newest sdist (4.13.x) for a `==4.9.*` requirement just re-fails
///    the re-solve (deps-from proof run 7). Same machinery git-sourced
///    `[retread-wheels]` entries use
///    (`crate::source_build::build_wheel_from_sdist_url`), cached/stored
///    content-addressed. A success registers a `tool.uv.sources` path
///    source (`req.built_wheel_sources`) so the re-solve is satisfied
///    exactly like a real index wheel — this is what actually clears
///    `uv lock`, whether or not the package was also conda-routed — and
///    records a [`BuiltSdistWheel`] for the caller to splice into the
///    final closure's `wheels` with full provenance (harmless if the
///    package is ALSO conda-routed: `no_emit_packages` still drops it
///    from the exported closure, so conda's copy is what ships).
/// 3. **error**: when the build rung is disabled (`sdist_build` is
///    `None` / `sdist-build = "never"`) or the build itself fails, the
///    original uv error surfaces verbatim, with guidance appended
///    ([`sdist_only_no_route_message`] for names with no conda route /
///    a routed-but-build-disabled note for names that do /
///    [`sdist_build_failed_message`] on a build failure). It never
///    silently drops a dependency.
///
/// A name is considered fully "healed" (dropped from later rounds'
/// `names`) only once rung 2 has actually BUILT a wheel for it — a
/// rung-1-only route is never sufficient (see point 1 above), so a name
/// that keeps recurring in uv's error after being routed keeps climbing
/// to rung 2 rather than being treated as already handled (deps-from
/// proof run 6: an earlier revision filtered on "routed OR built" and
/// let a rung-1 hit permanently mask the still-failing `uv lock`).
///
/// Bounded by [`AUTO_ROUTE_MAX_ROUNDS`] heal attempts, mirroring the
/// auto-route round cap; every attempt that doesn't abort or succeed
/// strictly grows the accumulated routed+built set, which is bounded by
/// the number of distinct sdist-only names uv can name.
///
/// The routes/builds discovered here are appended to `routed` / `built`
/// (shared with the caller via `Arc<Mutex<_>>`) so a wrapping
/// [`auto_route_fixpoint`] caller can splice them into the final
/// closure and log them with the same provenance conventions.
pub fn with_sdist_heal<S, SP, SB>(
    bundle: String,
    solve: S,
    sdist_probe: SP,
    sdist_build: Option<SB>,
    routed: std::sync::Arc<std::sync::Mutex<Vec<AutoRoutedPackage>>>,
    built: std::sync::Arc<std::sync::Mutex<Vec<BuiltSdistWheel>>>,
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
        // Already-healed exclusions/pins/path-sources accumulated across
        // earlier outer-loop rounds must apply to THIS round's request
        // too — the outer fixpoint owns `req` and knows nothing about
        // them.
        {
            let already = routed.lock().unwrap();
            apply_auto_route(&mut req, &already);
        }
        {
            let already = built.lock().unwrap();
            for w in already.iter() {
                req.built_wheel_sources
                    .insert(w.pypi_name.clone(), w.wheel_path.clone());
            }
        }
        let fut = (*solve.lock().unwrap())(req.clone());
        Box::pin(async move {
            let mut req = req;
            let mut attempt = fut;
            for _ in 0..AUTO_ROUTE_MAX_ROUNDS {
                match attempt.await {
                    Ok(closure) => return Ok(closure),
                    Err(e) => {
                        let msg = format!("{e:#}");
                        // A name is fully exhausted only once a wheel has
                        // actually been BUILT for it (rung 2 success).
                        // Conda-routing (rung 1) alone can NEVER satisfy
                        // `uv lock` for this error class: `--no-emit-
                        // package` only reaches the later `uv export`
                        // step (see the module doc + `compute_closure`),
                        // so a package pinned to a conda-resolved version
                        // that itself has zero usable PyPI wheels fails
                        // `uv lock` identically, whether or not it is
                        // routed. Filtering `names` on `routed` too (as
                        // an earlier revision did) let a rung-1 hit mask
                        // the fact that rung 2 was never reached, then
                        // treated the IDENTICAL failure recurring next
                        // round as "already handled" and surfaced it
                        // verbatim (deps-from proof run 6: antlr4-python3-
                        // runtime==4.9.3, conda-routed, `uv lock` still
                        // failed on it every round after).
                        let already_built: std::collections::BTreeSet<String> = {
                            let b = built.lock().unwrap();
                            b.iter().map(|w| w.pypi_name.clone()).collect()
                        };
                        let already_routed: std::collections::BTreeSet<String> = {
                            let r = routed.lock().unwrap();
                            r.iter().map(|r| r.pypi_name.clone()).collect()
                        };
                        let names: Vec<String> = extract_sdist_only_packages(&msg)
                            .into_iter()
                            .filter(|n| !already_built.contains(n))
                            .collect();
                        if names.is_empty() {
                            return Err(e);
                        }
                        // Rung 1: conda-route. Skip names already routed
                        // in an earlier round (no need to re-probe or
                        // re-log an already-known route) -- but note a
                        // route, new or old, does NOT exempt the name
                        // from rung 2 below.
                        // Carry the ORIGINATING pypi requirement's
                        // version range into BOTH rungs -- a bare
                        // "any version" let a wheel-less transitive dep
                        // like `antlr4-python3-runtime` route to conda's
                        // unconstrained latest (`==4.13.2`) and, run 7,
                        // let the build rung download/build the newest
                        // sdist (4.13.x) instead of one satisfying the
                        // requirer's own metadata (`==4.9.*`), so the
                        // re-solve failed identically.
                        let requirements: std::collections::BTreeMap<&str, Option<String>> = names
                            .iter()
                            .map(|n| (n.as_str(), extract_sdist_only_requirement(&msg, n)))
                            .collect();
                        let mut new_routes = Vec::new();
                        for name in names.iter().filter(|n| !already_routed.contains(*n)) {
                            let raw_requirement =
                                requirements.get(name.as_str()).cloned().unwrap_or_default();
                            let spec = conda_spec_from_pypi_specifier(raw_requirement.as_deref());
                            if let Some(hit) = sdist_probe(name.clone(), spec).await {
                                new_routes.push(AutoRoutedPackage {
                                    pypi_name: name.clone(),
                                    conda_name: name.clone(),
                                    pypi_version: hit.conda_version.clone(),
                                    conda_version: hit.conda_version.clone(),
                                    channel: hit.channel.clone(),
                                });
                            }
                        }
                        let routed_now: std::collections::BTreeSet<&str> =
                            new_routes.iter().map(|r| r.pypi_name.as_str()).collect();
                        // Rung 2: sdist auto-build. Attempted for EVERY
                        // name still unresolved this round -- not only
                        // rung-1 misses -- since rung 1 can never satisfy
                        // `uv lock` by itself for this error class (see
                        // comment above `names`). Only attempted when the
                        // caller enabled it (`sdist-build = "auto"`, the
                        // default); a `None` builder means `"never"` and
                        // rung 3 fires immediately, matching the pre-
                        // build-rung behavior for names with no route,
                        // plus a distinct message for names that DID
                        // route but still can't clear `uv lock` without a
                        // build.
                        let mut new_built = Vec::new();
                        let mut build_failures: Vec<(String, String)> = Vec::new();
                        match sdist_build.as_ref() {
                            Some(build) => {
                                for name in &names {
                                    let raw_requirement = requirements
                                        .get(name.as_str())
                                        .cloned()
                                        .unwrap_or_default();
                                    match build(name.clone(), raw_requirement).await {
                                        Ok(w) => new_built.push(w),
                                        Err(e) => {
                                            build_failures.push((name.clone(), format!("{e:#}")));
                                        }
                                    }
                                }
                            }
                            None => {
                                let unrouted: Vec<String> = names
                                    .iter()
                                    .filter(|n| {
                                        !routed_now.contains(n.as_str())
                                            && !already_routed.contains(*n)
                                    })
                                    .cloned()
                                    .collect();
                                let mut guidance = String::new();
                                if !unrouted.is_empty() {
                                    guidance.push_str(&sdist_only_no_route_message(&unrouted));
                                }
                                let routed_unbuildable: Vec<String> = names
                                    .iter()
                                    .filter(|n| !unrouted.contains(n))
                                    .cloned()
                                    .collect();
                                if !routed_unbuildable.is_empty() {
                                    guidance.push_str(&format!(
                                        "\npackage {} has a conda candidate but `sdist-build = \
                                         \"never\"` -- `uv lock` still requires a real PyPI \
                                         wheel for it (conda-routing only exempts a package at \
                                         the later `uv export` step); set `sdist-build = \
                                         \"auto\"` or vendor a wheel.\n",
                                        routed_unbuildable.join(", "),
                                    ));
                                }
                                bail!("{msg}{guidance}");
                            }
                        }
                        // Rung 3: error. Only the names the build rung
                        // actually failed for abort — a partial success
                        // (some names built, one didn't) still fails
                        // loudly rather than silently dropping the
                        // failed name.
                        if !build_failures.is_empty() {
                            bail!("{msg}{}", sdist_build_failed_message(&build_failures));
                        }
                        for h in &new_routes {
                            tracing::info!(
                                bundle = %bundle,
                                "sdist-only {} routed to conda (channel {})",
                                h.pypi_name,
                                h.channel,
                            );
                        }
                        apply_auto_route(&mut req, &new_routes);
                        {
                            let mut g = routed.lock().unwrap();
                            g.extend(new_routes.clone());
                        }
                        for w in &new_built {
                            req.built_wheel_sources
                                .insert(w.pypi_name.clone(), w.wheel_path.clone());
                        }
                        {
                            let mut g = built.lock().unwrap();
                            g.extend(new_built.clone());
                        }
                        attempt = (*solve.lock().unwrap())(req.clone());
                    }
                }
            }
            bail!(
                "sdist-only self-heal exceeded {} rounds for bundle `{}`",
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
    let healed_solve = with_sdist_heal(
        req.bundle.clone(),
        solve,
        sdist_probe,
        sdist_build,
        std::sync::Arc::clone(&sdist_routed),
        std::sync::Arc::clone(&sdist_built),
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
    source: &str,
    env: &str,
) -> ConstraintSet {
    // Invert pypi->conda. BTreeMap iteration is ordered, so on conda-name
    // collisions the alphabetically-first PyPI name wins deterministically.
    let mut conda_to_pypi: BTreeMap<String, String> = BTreeMap::new();
    for (pypi, conda) in name_map {
        conda_to_pypi
            .entry(canonical_conda_name(conda))
            .or_insert_with(|| canonical_conda_name(pypi));
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
    out.push_str(&format!(
        "dependencies = {}\n",
        toml_string_array("", &req.dependencies)
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
        // Word-boundary match on the normalized name.
        let re = regex::Regex::new(&format!(
            r"(?i)\b{}(?:\[[^\]]*\])?((?:==|>=|<=|~=|!=|>|<)[0-9][^\s,)`']*)?",
            regex::escape(pypi_name)
        ))
        .expect("static conflict regex");
        let mut mentioned = false;
        let mut required: Option<String> = None;
        for cap in re.captures_iter(stderr) {
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
fn closure_inputs_fingerprint(pyproject: &str, lock_args: &[String], uv_version: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(pyproject.as_bytes());
    for arg in lock_args {
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

    // -- uv lock -----------------------------------------------------------
    let mut lock_args: Vec<String> = vec![
        "lock".into(),
        "--project".into(),
        project_dir.to_string_lossy().into_owned(),
        "--python".into(),
        req.python_version.clone(),
        "--no-progress".into(),
        "--color".into(),
        "never".into(),
    ];
    // CRITICAL: we run uv with UV_NO_CONFIG=1 to isolate the resolve
    // from user-level uv.toml. As of uv 0.11.x that ALSO strips the
    // configuration-file-class keys from the synthesized pyproject's
    // `[tool.uv]` table -- `[[tool.uv.index]]`, `no-build`,
    // `index-strategy`, and `prerelease` are silently ignored
    // (project-only keys like `environments`, `override-dependencies`,
    // `constraint-dependencies` and `[tool.uv.sources]` still apply).
    // Without the index flags uv falls back to pypi.org alone, where
    // e.g. `isaacsim` exists only as a stub sdist (the real
    // manylinux_2_35 wheels live on pypi.nvidia.com) -- the closure then
    // exports zero wheels and the whole workspace lock fails. Pass every
    // config-file-class setting as a CLI flag, which UV_NO_CONFIG never
    // touches. The `[tool.uv]` copies stay in the synthesized pyproject
    // for uv versions where --no-config leaves project tables alone
    // (the flags and the table agree).
    lock_args.push("--no-build".into());
    lock_args.push("--index-strategy".into());
    lock_args.push("unsafe-best-match".into());
    lock_args.push("--prerelease".into());
    lock_args.push("if-necessary-or-explicit".into());
    for url in &req.index_urls {
        lock_args.push("--index".into());
        lock_args.push(url.clone());
    }
    if req.offline {
        lock_args.push("--offline".into());
    }

    // Stale-lock guard: `uv lock` reuses a pre-existing uv.lock whenever
    // it still satisfies the pyproject TEXT -- it cannot see the CLI
    // flags above, so a lock written by a different backend version /
    // flag set survives validation while pinning packages to the wrong
    // index. Drop the lock (forcing a fresh resolve) whenever the
    // recorded input fingerprint is absent or different.
    let fingerprint = closure_inputs_fingerprint(&pyproject_text, &lock_args, &uv_version);
    let lock_file = project_dir.join("uv.lock");
    let recorded_fingerprint = std::fs::read_to_string(&meta_path)
        .ok()
        .and_then(|s| serde_json::from_str::<ClosureMeta>(&s).ok())
        .map(|m| m.inputs_fingerprint);
    if lock_file.exists() && recorded_fingerprint.as_deref() != Some(fingerprint.as_str()) {
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
        bail!("{}", format_lock_failure(req, &stderr, &attributions));
    }

    // -- uv export ---------------------------------------------------------
    let mut export_args: Vec<String> = vec![
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
    for name in &req.no_emit_packages {
        export_args.push("--no-emit-package".into());
        export_args.push(canonical_conda_name(name));
    }
    if req.offline {
        export_args.push("--offline".into());
    }
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
    let closure = parse_pylock_closure(&pylock, &target, &exclude, &uv_version)?;

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
        let constraints = build_constraints(&conda_deps, &name_map, "manifest", "default");
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

        let set = build_constraints(&conda_deps, &name_map, "manifest", "default");
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
        let set = build_constraints(&conda_deps, &name_map, "manifest", "default");
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
        let set = build_constraints(&conda_deps, &BTreeMap::new(), "manifest", "default");

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
        let set = build_constraints(&conda_deps, &BTreeMap::new(), "manifest", "default");
        // Name mentioned without a parseable range: record with required=None.
        let attributions = attribute_conflict("something about torch went wrong", &set.provenance);
        assert_eq!(attributions.len(), 1);
        assert_eq!(attributions[0].required, None);
        // Name not mentioned at all: no record.
        let none = attribute_conflict("unrelated failure", &set.provenance);
        assert!(none.is_empty());
    }

    // ---- auto-route (spec-uv-restructure M2) -------------------------------

    use std::sync::{Arc, Mutex};

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
        }
    }

    /// The core loop: numpy exists on a channel at the resolved version
    /// -> excluded from the closure, pinned as a constraint, recorded as
    /// auto-routed; the loop re-locks once and reaches fixpoint.
    #[tokio::test]
    async fn auto_route_routes_hit_and_reaches_fixpoint() {
        let calls = Arc::new(Mutex::new(Vec::new()));
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
            &auto_route_req(),
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
        let routes = plan_auto_route_round(&closure, &req, &auto_route_opts(), &[], &hits);
        let names: Vec<&str> = routes.iter().map(|r| r.pypi_name.as_str()).collect();
        assert_eq!(names, vec!["typing-extensions"]);
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
        let routes = plan_auto_route_round(&closure, &req, &auto_route_opts(), &[], &hits);
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
            offline: false,
        };
        let closure = compute_closure(&req, &tmp.join("project"), &tmp.join("uv-cache"), None)
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
            offline: false,
        };
        let opts = AutoRouteOptions {
            enabled: true,
            keep_pypi: BTreeSet::new(),
            force_conda: BTreeSet::new(),
            protected: BTreeSet::from(["python-dateutil".to_string()]),
            name_map: BTreeMap::new(),
            abi_anchor_pins: BTreeMap::new(),
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
                Box::pin(async move { compute_closure(&r, &project, &cache, None).await })
                    as futures::future::BoxFuture<'static, Result<UvClosure>>
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

    // ---- sdist-only self-heal --------------------------------------------

    const SDIST_ONLY_UV_ERR: &str = "  x No solution found when resolving dependencies:\n  \
         `-> Because pyperclip has no wheels and you require pyperclip,\n  \
             we can conclude that your requirements are unsatisfiable.";

    const PLATFORM_TAG_UV_ERR: &str = "× No solution found when resolving dependencies:\n  \
        ╰─▶ Because isaacsim[all]==6.0.0.1 has no wheels with a matching platform \
        tag (e.g., `manylinux_2_34_x86_64`) and you require isaacsim[all]==6.0.0.1, \
        we can conclude that your requirements are unsatisfiable.";

    #[test]
    fn is_sdist_only_uv_error_matches_no_wheels_class() {
        assert!(is_sdist_only_uv_error(SDIST_ONLY_UV_ERR));
        // Platform-tag ceiling: distinct class, must NOT trigger this path
        // (installer::is_platform_tag_conflict owns glibc relaxation there).
        assert!(!is_sdist_only_uv_error(PLATFORM_TAG_UV_ERR));
        assert!(!is_sdist_only_uv_error("unrelated network error"));
    }

    #[test]
    fn extract_sdist_only_packages_names_the_package() {
        assert_eq!(
            extract_sdist_only_packages(SDIST_ONLY_UV_ERR),
            vec!["pyperclip".to_string()]
        );
        // Platform-tag class extracts nothing -- it is not this class.
        assert!(extract_sdist_only_packages(PLATFORM_TAG_UV_ERR).is_empty());
        assert!(extract_sdist_only_packages("unrelated failure").is_empty());
    }

    #[test]
    fn extract_sdist_only_packages_handles_extras_and_version() {
        let text = "Because foo-bar[extra]==1.2.3 has no wheels and you require \
                     foo-bar, we can conclude that your requirements are unsatisfiable.";
        assert_eq!(
            extract_sdist_only_packages(text),
            vec!["foo-bar".to_string()]
        );
    }

    // deps-from proof run 3 (step4-lock-run3.log): a wildcard-pinned
    // transitive dep (`hydra-core==1.3.2` depends on
    // `antlr4-python3-runtime==4.9.*`) collapses to a RANGE subject in
    // uv's prose ("...antlr4-python3-runtime>=4.9,<=4.9.3 has no usable
    // wheels...") rather than the bare "Because <name> has no wheels"
    // shape -- distinct phrasing ("usable"), distinct clause shape (the
    // package sits right before the range/operator, not right after
    // "Because"). Must still be recognized as the same sdist-only class
    // so the pre-existing conda-route/auto-build ladder self-heals it.
    const NO_USABLE_WHEELS_RANGE_UV_ERR: &str = "× No solution found when resolving dependencies for split: \
         Because only the following versions of antlr4-python3-runtime are \
         available: antlr4-python3-runtime<4.9.dev0, antlr4-python3-runtime==4.9, \
         antlr4-python3-runtime==4.9.1, antlr4-python3-runtime==4.9.2, \
         antlr4-python3-runtime==4.9.3, antlr4-python3-runtime>4.10.dev0 and \
         antlr4-python3-runtime>=4.9,<=4.9.3 has no usable wheels, we can \
         conclude that antlr4-python3-runtime>=4.9,<=4.9.3 cannot be used. \
         And because hydra-core==1.3.2 depends on antlr4-python3-runtime==4.9.* \
         and your project depends on hydra-core==1.3.2, we can conclude that \
         your project's requirements are unsatisfiable.";

    #[test]
    fn is_sdist_only_uv_error_matches_has_no_usable_wheels_range_class() {
        assert!(is_sdist_only_uv_error(NO_USABLE_WHEELS_RANGE_UV_ERR));
    }

    #[test]
    fn extract_sdist_only_packages_names_wildcard_pinned_transitive_dep() {
        assert_eq!(
            extract_sdist_only_packages(NO_USABLE_WHEELS_RANGE_UV_ERR),
            vec!["antlr4-python3-runtime".to_string()]
        );
    }

    // ---- run-5 fix: carry the pypi requirement's range into the conda
    // probe (rather than routing to conda's unconstrained latest) -----

    #[test]
    fn extract_sdist_only_requirement_finds_wildcard_pin() {
        assert_eq!(
            extract_sdist_only_requirement(NO_USABLE_WHEELS_RANGE_UV_ERR, "antlr4-python3-runtime"),
            Some("==4.9.*".to_string())
        );
    }

    #[test]
    fn extract_sdist_only_requirement_none_when_unpinned() {
        // The bare "has no wheels" class never names a requirer's own
        // specifier -- no "depends on <name><spec>" clause exists.
        assert_eq!(
            extract_sdist_only_requirement(SDIST_ONLY_UV_ERR, "pyperclip"),
            None
        );
    }

    #[test]
    fn extract_sdist_only_requirement_is_separator_insensitive() {
        // uv may spell the requirer's dependency with underscores where
        // the canonical (PEP 503) name uses hyphens, or vice versa.
        let text = "Because foo_bar has no wheels, we can conclude that \
                     foo_bar cannot be used. And because baz depends on \
                     foo-bar==1.2.* and your project depends on baz, we \
                     can conclude your requirements are unsatisfiable.";
        assert_eq!(
            extract_sdist_only_requirement(text, "foo_bar"),
            Some("==1.2.*".to_string())
        );
    }

    // deps-from proof run 7 (step4-lock-run7.log, ErrorObject message
    // decoded: pixi display-wrapping undone, `\n` escapes restored to
    // real newlines -- the shape the backend-side error chain actually
    // carries): once the wheel-less 4.9.x band is filtered out of uv's
    // candidate set entirely, uv stops saying "has no (usable) wheels"
    // and lists the AVAILABLE set as the band's dev-bounded complement
    // ("antlr4-python3-runtime<4.9.dev0", ">=4.10.dev0"). Third prose
    // shape of the same sdist-only class.
    const ONLY_VERSIONS_AVAILABLE_BAND_UV_ERR: &str =
        include_str!("../tests/fixtures/solve_errors/uv_closure_only_versions_available_band.txt");

    #[test]
    fn is_sdist_only_uv_error_matches_available_band_class() {
        assert!(is_sdist_only_uv_error(ONLY_VERSIONS_AVAILABLE_BAND_UV_ERR));
    }

    #[test]
    fn extract_sdist_only_packages_names_available_band_subject() {
        assert_eq!(
            extract_sdist_only_packages(ONLY_VERSIONS_AVAILABLE_BAND_UV_ERR),
            vec!["antlr4-python3-runtime".to_string()]
        );
    }

    #[test]
    fn extract_sdist_only_requirement_trims_trailing_sentence_punctuation() {
        // Run 7's prose runs the specifier straight into a comma
        // ("...depends on antlr4-python3-runtime==4.9.*, we can...") --
        // the greedy capture must not keep it.
        assert_eq!(
            extract_sdist_only_requirement(
                ONLY_VERSIONS_AVAILABLE_BAND_UV_ERR,
                "antlr4-python3-runtime"
            ),
            Some("==4.9.*".to_string())
        );
    }

    #[test]
    fn conda_spec_from_pypi_specifier_translates_wildcard() {
        assert_eq!(conda_spec_from_pypi_specifier(Some("==4.9.*")), "4.9.*");
    }

    #[test]
    fn conda_spec_from_pypi_specifier_passes_through_exact_pin() {
        assert_eq!(conda_spec_from_pypi_specifier(Some("==4.9.3")), "==4.9.3");
    }

    #[test]
    fn conda_spec_from_pypi_specifier_passes_through_range() {
        assert_eq!(
            conda_spec_from_pypi_specifier(Some(">=4.9,<=4.9.3")),
            ">=4.9,<=4.9.3"
        );
    }

    #[test]
    fn conda_spec_from_pypi_specifier_none_is_unpinned_wildcard() {
        assert_eq!(conda_spec_from_pypi_specifier(None), "*");
    }

    /// Fixture-driven fixpoint reproducing deps-from proof run 5: the
    /// sdist-route probe must receive the requirement's `==4.9.*` range
    /// (translated to conda's `4.9.*`) rather than an unconstrained
    /// `"*"`, so a conda-forge channel carrying BOTH a 4.9.x build and a
    /// newer 4.13.x build routes to the 4.9.x version instead of
    /// clashing with hydra-core's own pypi metadata. Run 6 additionally
    /// proved routing alone can never satisfy `uv lock` for this error
    /// class, so this fixture's mock `solve` (like real
    /// `compute_closure`) also requires the rung-2 built wheel source
    /// before it succeeds.
    #[tokio::test]
    async fn sdist_heal_probe_carries_pypi_requirement_range() {
        let attempts = Arc::new(Mutex::new(0usize));
        let solve = {
            let attempts = Arc::clone(&attempts);
            move |r: UvClosureRequest| {
                let attempts = Arc::clone(&attempts);
                Box::pin(async move {
                    let mut n = attempts.lock().unwrap();
                    *n += 1;
                    if !r
                        .no_emit_packages
                        .contains(&"antlr4-python3-runtime".to_string())
                        || !r.built_wheel_sources.contains_key("antlr4-python3-runtime")
                    {
                        bail!("{NO_USABLE_WHEELS_RANGE_UV_ERR}");
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
        let probe = |_name: String, _spec: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let seen_specs = Arc::new(Mutex::new(Vec::new()));
        let sdist_probe = {
            let seen_specs = Arc::clone(&seen_specs);
            move |name: String, spec: String| {
                let seen_specs = Arc::clone(&seen_specs);
                Box::pin(async move {
                    seen_specs.lock().unwrap().push(spec.clone());
                    if name == "antlr4-python3-runtime" && spec == "4.9.*" {
                        Some(RouteProbeHit {
                            conda_version: "4.9.3".into(),
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
                        version: "4.9.3".to_string(),
                        filename: format!("{name}-4.9.3-py3-none-any.whl"),
                        wheel_path: PathBuf::from(format!("/tmp/wheels/{name}-4.9.3.whl")),
                        sha256: "d".repeat(64),
                        sdist_source: sdist_source_fixture(&name, "4.9.3"),
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
            seen_specs.lock().unwrap().as_slice(),
            &["4.9.*".to_string()],
            "probe must be queried with the translated pypi requirement range, not a bare wildcard"
        );
        assert_eq!(
            seen_build_reqs.lock().unwrap().as_slice(),
            &[Some("==4.9.*".to_string())],
            "the build rung must receive the RAW pypi requirement range (run-7 fix: a \
             match-any selection built the newest sdist instead of one satisfying the requirer)"
        );
        assert_eq!(closure.auto_routed.len(), 1);
        let r = &closure.auto_routed[0];
        assert_eq!(r.pypi_name, "antlr4-python3-runtime");
        assert_eq!(r.conda_version, "4.9.3");
    }

    /// Run-7 end-to-end: the band-complement template (which contains NO
    /// "has no (usable) wheels" prose at all) must drive the full heal
    /// ladder, and the build rung must receive the extracted requirement
    /// range (`==4.9.*`) so the sdist selection builds 4.9.3 rather than
    /// the newest release.
    #[tokio::test]
    async fn sdist_heal_available_band_template_builds_at_required_range() {
        let solve = move |r: UvClosureRequest| {
            Box::pin(async move {
                if !r.built_wheel_sources.contains_key("antlr4-python3-runtime") {
                    bail!("{ONLY_VERSIONS_AVAILABLE_BAND_UV_ERR}");
                }
                parse_pylock_closure(
                    PYLOCK_FIXTURE,
                    &target("3.12", "linux-64"),
                    &BTreeSet::new(),
                    "0.11.15",
                )
            }) as futures::future::BoxFuture<'static, Result<UvClosure>>
        };
        let probe = |_name: String, _spec: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let sdist_probe = |_name: String, _spec: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let seen_build_reqs = Arc::new(Mutex::new(Vec::new()));
        let sdist_build = {
            let seen_build_reqs = Arc::clone(&seen_build_reqs);
            move |name: String, req: Option<String>| {
                let seen_build_reqs = Arc::clone(&seen_build_reqs);
                Box::pin(async move {
                    seen_build_reqs.lock().unwrap().push((name.clone(), req));
                    Ok(BuiltSdistWheel {
                        pypi_name: name.clone(),
                        version: "4.9.3".to_string(),
                        filename: format!("{name}-4.9.3-py3-none-any.whl"),
                        wheel_path: PathBuf::from(format!("/tmp/wheels/{name}-4.9.3.whl")),
                        sha256: "e".repeat(64),
                        sdist_source: sdist_source_fixture(&name, "4.9.3"),
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
            seen_build_reqs.lock().unwrap().as_slice(),
            &[(
                "antlr4-python3-runtime".to_string(),
                Some("==4.9.*".to_string())
            )],
            "build rung must be reached from the band-complement template and receive \
             the requirer's raw range"
        );
        let built = closure
            .wheels
            .iter()
            .find(|w| w.name == "antlr4-python3-runtime")
            .expect("built sdist wheel recorded in the closure");
        assert_eq!(built.version, "4.9.3");
        assert!(matches!(built.origin, Origin::Built));
    }

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

    /// Fixture-driven fixpoint: `uv lock` fails once with the sdist-only
    /// class naming `pyperclip`; the mock repodata probe has a hit -> the
    /// package is ALSO routed to conda (excluded from export + pinned).
    /// Routing alone can never satisfy `uv lock` (`--no-emit-package` is
    /// an `export`-only flag), so the mock `solve` requires BOTH the
    /// route AND a built wheel before it succeeds — modeling real
    /// `compute_closure` behavior — and the build rung MUST still run
    /// even though rung 1 hit (deps-from proof run 6 regression test:
    /// an earlier revision skipped rung 2 whenever rung 1 hit and got
    /// stuck re-surfacing the identical `uv lock` failure every round).
    #[tokio::test]
    async fn sdist_heal_builds_even_after_conda_route_hit() {
        let attempts = Arc::new(Mutex::new(0usize));
        let solve = {
            let attempts = Arc::clone(&attempts);
            move |r: UvClosureRequest| {
                let attempts = Arc::clone(&attempts);
                Box::pin(async move {
                    let mut n = attempts.lock().unwrap();
                    *n += 1;
                    if !r.no_emit_packages.contains(&"pyperclip".to_string())
                        || !r.built_wheel_sources.contains_key("pyperclip")
                    {
                        bail!("{SDIST_ONLY_UV_ERR}");
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
        let probe = |_name: String, _spec: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let sdist_probe = |name: String, _spec: String| {
            Box::pin(async move {
                if name == "pyperclip" {
                    Some(RouteProbeHit {
                        conda_version: "1.8.2".into(),
                        channel: "https://conda.anaconda.org/conda-forge/linux-64".into(),
                        depends: Vec::new(),
                    })
                } else {
                    None
                }
            }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let build_calls = Arc::new(Mutex::new(0usize));
        let sdist_build = {
            let build_calls = Arc::clone(&build_calls);
            move |name: String, _req: Option<String>| {
                let build_calls = Arc::clone(&build_calls);
                Box::pin(async move {
                    *build_calls.lock().unwrap() += 1;
                    assert_eq!(name, "pyperclip");
                    Ok(BuiltSdistWheel {
                        pypi_name: name.clone(),
                        version: "1.8.2".to_string(),
                        filename: "pyperclip-1.8.2-py3-none-any.whl".to_string(),
                        wheel_path: PathBuf::from("/tmp/wheels/pyperclip-1.8.2-py3-none-any.whl"),
                        sha256: "b".repeat(64),
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
        assert_eq!(closure.auto_routed.len(), 1);
        let r = &closure.auto_routed[0];
        assert_eq!(r.pypi_name, "pyperclip");
        assert_eq!(r.conda_name, "pyperclip");
        assert_eq!(r.conda_version, "1.8.2");
        assert!(r.channel.contains("conda-forge"));
        assert_eq!(
            *build_calls.lock().unwrap(),
            1,
            "rung 2 (build) must run in the SAME round as a rung-1 conda-route hit -- \
             routing alone can never satisfy `uv lock`",
        );
    }

    /// Regression test for deps-from proof run 6: the SAME package name
    /// recurs in uv's error across TWO rounds even after being
    /// conda-routed in round 1 (mirroring a real `uv lock` that still
    /// needs a wheel despite the conda pin). The old ladder treated
    /// "routed" as fully healed and, on round 2's identical failure,
    /// filtered the name out of `names` and returned the original error
    /// verbatim instead of climbing to rung 2. The fixed ladder must
    /// still reach the build rung and succeed.
    #[tokio::test]
    async fn sdist_heal_recurring_failure_after_route_still_reaches_build_rung() {
        let attempts = Arc::new(Mutex::new(0usize));
        let solve = {
            let attempts = Arc::clone(&attempts);
            move |r: UvClosureRequest| {
                let attempts = Arc::clone(&attempts);
                Box::pin(async move {
                    let mut n = attempts.lock().unwrap();
                    *n += 1;
                    if !r.built_wheel_sources.contains_key("antlr4-python3-runtime") {
                        bail!("{NO_USABLE_WHEELS_RANGE_UV_ERR}");
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
        let probe = |_name: String, _spec: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let route_calls = Arc::new(Mutex::new(0usize));
        let sdist_probe = {
            let route_calls = Arc::clone(&route_calls);
            move |name: String, _spec: String| {
                let route_calls = Arc::clone(&route_calls);
                Box::pin(async move {
                    *route_calls.lock().unwrap() += 1;
                    if name == "antlr4-python3-runtime" {
                        Some(RouteProbeHit {
                            conda_version: "4.9.3".into(),
                            channel: "https://conda.anaconda.org/conda-forge/linux-64".into(),
                            depends: Vec::new(),
                        })
                    } else {
                        None
                    }
                }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
            }
        };
        let build_calls = Arc::new(Mutex::new(0usize));
        let sdist_build = {
            let build_calls = Arc::clone(&build_calls);
            move |name: String, _req: Option<String>| {
                let build_calls = Arc::clone(&build_calls);
                Box::pin(async move {
                    *build_calls.lock().unwrap() += 1;
                    Ok(BuiltSdistWheel {
                        pypi_name: name.clone(),
                        version: "4.9.3".to_string(),
                        filename: format!("{name}-4.9.3-py3-none-any.whl"),
                        wheel_path: PathBuf::from(format!("/tmp/wheels/{name}-4.9.3.whl")),
                        sha256: "c".repeat(64),
                        sdist_source: sdist_source_fixture(&name, "4.9.3"),
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

        assert!(
            *attempts.lock().unwrap() >= 2,
            "must retry at least once after the conda-route-only round still fails"
        );
        assert_eq!(
            *build_calls.lock().unwrap(),
            1,
            "the recurring failure must still climb to rung 2 (build) instead of being \
             surfaced verbatim as already-handled",
        );
        assert_eq!(closure.auto_routed.len(), 1);
        assert_eq!(closure.auto_routed[0].pypi_name, "antlr4-python3-runtime");
    }

    /// Double-miss (no conda candidate) with the build rung ENABLED: the
    /// sdist auto-build is invoked and, on success, the closure gains an
    /// `Origin::Built` wheel carrying `sdist_source` provenance instead
    /// of erroring.
    #[tokio::test]
    async fn sdist_heal_builds_on_double_miss() {
        let attempts = Arc::new(Mutex::new(0usize));
        let solve = {
            let attempts = Arc::clone(&attempts);
            move |r: UvClosureRequest| {
                let attempts = Arc::clone(&attempts);
                Box::pin(async move {
                    let mut n = attempts.lock().unwrap();
                    *n += 1;
                    if !r.built_wheel_sources.contains_key("pyperclip") {
                        bail!("{SDIST_ONLY_UV_ERR}");
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
        let probe = |_name: String, _spec: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let sdist_probe = |_name: String, _spec: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let build_calls = Arc::new(Mutex::new(0usize));
        let sdist_build = {
            let build_calls = Arc::clone(&build_calls);
            move |name: String, _req: Option<String>| {
                let build_calls = Arc::clone(&build_calls);
                Box::pin(async move {
                    *build_calls.lock().unwrap() += 1;
                    assert_eq!(name, "pyperclip");
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
        assert_eq!(*build_calls.lock().unwrap(), 1);
        assert!(closure.auto_routed.is_empty(), "no conda route this time");
        let built = closure
            .wheels
            .iter()
            .find(|w| w.name == "pyperclip")
            .expect("built sdist wheel recorded in the closure");
        assert_eq!(built.version, "1.8.2");
        assert!(matches!(built.origin, Origin::Built));
        assert!(built.must_ship);
        assert_eq!(built.sha256.as_deref(), Some("a".repeat(64).as_str()));
        let prov = built
            .sdist_source
            .as_ref()
            .expect("built sdist wheel must carry sdist_source provenance");
        assert_eq!(prov.name, "pyperclip");
        assert_eq!(prov.version, "1.8.2");
        assert_eq!(
            closure.pins.get("pyperclip").map(String::as_str),
            Some("1.8.2")
        );
    }

    /// Build rung enabled but the build ITSELF fails: rung 3 (error)
    /// fires, surfacing the original uv error plus the build failure's
    /// log tail and guidance -- never silently drops the dependency.
    #[tokio::test]
    async fn sdist_heal_surfaces_build_failure_log_tail() {
        let solve = |_r: UvClosureRequest| {
            Box::pin(async { bail!("{SDIST_ONLY_UV_ERR}") })
                as futures::future::BoxFuture<'static, Result<UvClosure>>
        };
        let probe = |_name: String, _spec: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let sdist_probe = |_name: String, _spec: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let sdist_build = |_name: String, _req: Option<String>| {
            Box::pin(async {
                bail!(
                    "uv [\"build\", \"--wheel\"] failed (status 1): error: \
                     failed to build `pyperclip` (missing gcc)"
                )
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
        assert!(msg.contains("pyperclip has no wheels"), "{msg}");
        assert!(msg.contains("sdist auto-build failed"), "{msg}");
        assert!(msg.contains("missing gcc"), "{msg}");
    }

    /// `sdist-build = "never"` (build rung disabled, `sdist_build: None`):
    /// no conda candidate either -> the ORIGINAL uv error surfaces
    /// verbatim, with the guidance text appended, exactly as before the
    /// build rung existed. The build closure type itself is never
    /// constructed (`None::<NoBuild>`), so there is no way for this path
    /// to attempt a build.
    #[tokio::test]
    async fn sdist_heal_never_policy_surfaces_original_error_with_guidance_on_miss() {
        let solve = |_r: UvClosureRequest| {
            Box::pin(async { bail!("{SDIST_ONLY_UV_ERR}") })
                as futures::future::BoxFuture<'static, Result<UvClosure>>
        };
        let probe = |_name: String, _spec: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let sdist_probe = |_name: String, _spec: String| {
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
        let msg = err.to_string();
        assert!(msg.contains("pyperclip has no wheels"), "{msg}");
        assert!(msg.contains("no conda candidate"), "{msg}");
        assert!(msg.contains("no-build = false"), "{msg}");
    }

    /// A platform-tag conflict (the manylinux ceiling, already handled
    /// by `installer::is_platform_tag_conflict` via glibc relaxation)
    /// must pass straight through unchanged -- the sdist-only heal must
    /// never misfire on it, and neither the conda-route probe nor the
    /// build rung is ever consulted.
    #[tokio::test]
    async fn sdist_heal_ignores_platform_tag_class() {
        let calls = Arc::new(Mutex::new(0usize));
        let solve = {
            let calls = Arc::clone(&calls);
            move |_r: UvClosureRequest| {
                let calls = Arc::clone(&calls);
                Box::pin(async move {
                    *calls.lock().unwrap() += 1;
                    bail!("{PLATFORM_TAG_UV_ERR}")
                }) as futures::future::BoxFuture<'static, Result<UvClosure>>
            }
        };
        let probe = |_name: String, _spec: String| {
            Box::pin(async { None }) as futures::future::BoxFuture<'static, Option<RouteProbeHit>>
        };
        let sdist_probe_calls = Arc::new(Mutex::new(0usize));
        let sdist_probe = {
            let sdist_probe_calls = Arc::clone(&sdist_probe_calls);
            move |_name: String, _spec: String| {
                let sdist_probe_calls = Arc::clone(&sdist_probe_calls);
                Box::pin(async move {
                    *sdist_probe_calls.lock().unwrap() += 1;
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
        assert!(err.to_string().contains("platform"), "{err}");
        // The original error propagated straight through: no retry, and
        // the sdist-heal probe was never issued (wrong error class).
        assert_eq!(*calls.lock().unwrap(), 1);
        assert_eq!(*sdist_probe_calls.lock().unwrap(), 0);
    }
}
