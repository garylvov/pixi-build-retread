//! v1.6.0: emit-pypi side-channel (experimental).
//!
//! When `retread-emit-pypi = true`, every conda/build_v1 additionally
//! writes, next to the pack manifest:
//!
//! ```text
//! retread-pypi/<bundle>/wheels/        retread-fixed wheels, standard names
//! retread-pypi/<bundle>/pixi-snippet.toml   paste-ready feature blocks
//! ```
//!
//! Motivation (2026-06-11 benchmark): the conda path round-trips the
//! whole bundle through zstd -- rattler-build compresses ~25GB into a
//! ~6GB .conda, pixi decompresses it back out, and every env relink
//! re-validates the extracted cache by content. pixi's embedded uv
//! installs from an UNCOMPRESSED wheel cache by hardlink: 78s vs 3s
//! for the same Isaac env recreate. This side-channel lets a workspace
//! consume retread's brains (source builds, data injection, pin
//! relaxation) through pixi's native `[pypi-dependencies]` path:
//!
//! - Wheels retread BUILT (git/path sources, recognizable by the
//!   `.injected` infix -- they exist on no index), plus any local
//!   wheel that is the target of a direct-URL requirement, are copied
//!   into `wheels/` under standard filenames for a `find-links`
//!   source. Without the injection these wheels are broken at import
//!   (IsaacLab ships its `config/extension.toml` tree via phase 1.6
//!   auto-data).
//! - Index-origin wheels are NOT copied (an isaac-scale pack would
//!   duplicate gigabytes); their exact upstream pins are neutralized
//!   by a generated `[pypi-options.dependency-overrides]` table
//!   instead, mirroring what D-rewriting does for the conda path.
//! - The user's `[retread-wheels]` entries become `[pypi-dependencies]`
//!   lines: built wheels pin `==<resolved version>` (find-links serves
//!   them), index entries pass through version/extras/index.
//!
//! Override values are FLOOR-ONLY envelopes (`>=<lowest pin seen>`,
//! deliberately uncapped) rather than the D-rewrite's tighter ranges:
//! on the conda path the cascade iteratively widens on solve failure,
//! but here the solve happens later inside pixi where retread cannot
//! iterate, so the table must be tolerant up front (see
//! [`floor_envelope`] for the two capped shapes that failed live). A
//! pin whose lower bound is a prerelease keeps it verbatim, which is
//! also what tells uv to allow prereleases for that package (the
//! tinyobjloader==2.0.0rc13 case).
//!
//! Everything here is derived from the same in-memory bundle the conda
//! recipe is built from -- zero hand edits, regenerated every build.
//! The snippet is advisory output (like the audit JSON); nothing in
//! the conda pipeline reads it back.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Result;
use uv_pep508::uv_pep440::{Operator, Version};

use crate::config::WheelEntry;

/// Everything emit needs to know about one wheel of the bundle,
/// decoupled from handler::ResolvedWheel so the logic is unit-testable.
#[derive(Clone)]
pub struct EmitWheel {
    /// PEP 503 normalized PyPI name.
    pub pypi_name: String,
    /// Resolved version (from wheel METADATA).
    pub version: String,
    /// POST-relax Requires-Dist lines (what the final wheel carries).
    pub requires_dist: Vec<String>,
    /// Local path of the final post-D wheel, when the source URL is
    /// file:// (retread modified or built it). `None` for wheels whose
    /// recipe source stayed a remote URL.
    pub local_path: Option<PathBuf>,
    /// The wheel's filename (basename of its source URL), used to read
    /// platform tags. Index wheels keep their upstream name; local
    /// wheels carry retread's infixes (harmless for tag parsing).
    pub wheel_filename: String,
    /// SHA-256 of the exact wheel bytes when known. Index wheels must carry
    /// this into the courier lock so install-time replay can fetch the locked
    /// artifact URL directly without consulting index metadata.
    pub sha256: Option<String>,
    /// Upstream URL when the wheel was never materialized locally
    /// (PEP 658 sidecar metadata). Blueprint mode fetches these on
    /// demand when their Requires-Dist needs rewriting.
    pub remote_url: Option<url::Url>,
    /// Pristine pre-localization index URL for this wheel, populated at
    /// cold-produce time from the unlocalized `w.url` (before
    /// `localize_wheel_source` collapses it to `file://`). Used by the
    /// courier's Class-2 shadow path to record the upstream URL in the
    /// lock so the replay path can re-fetch and re-relax the shadow
    /// without running the full BFS/solve.
    ///
    /// `None` for source-built `.injected` wheels (no upstream index URL)
    /// and for wheels that were only ever seen as remote (those use
    /// `remote_url` directly).
    pub upstream_url: Option<url::Url>,
    /// Git provenance for source-built wheels (schema 8+). Populated by
    /// `materialize_and_rewrite` for both named-git and inline-git entry
    /// forms. Written into `LockWheel.git_source` by `courier::stage` so
    /// the replay path can re-source-build manifest-independently.
    pub git_source: Option<crate::lock::GitWheelSource>,
    /// Sdist provenance for BFS-transitive wheels built from a PyPI sdist
    /// (schema 9+). Populated by the BFS phase-3 loop when `bfs_fetch_pypi`
    /// returns a `SdistProv`. Written into `LockWheel.sdist_source` by
    /// `courier::stage` so the Class-2b replay path can re-build from the
    /// stored sdist_url manifest-independently.
    pub sdist_source: Option<crate::lock::SdistWheelSource>,
}

impl EmitWheel {
    /// Built-by-retread wheels carry the phase-1.5 `.injected` infix;
    /// they exist on no index and MUST ship via find-links. Index
    /// wheels (even relaxed `.relaxed.whl` copies) are reachable
    /// upstream and are handled by the overrides table instead.
    pub fn must_ship(&self) -> bool {
        self.local_path
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.contains(".injected"))
    }
}

/// Strip retread's processing infixes from a cached wheel filename,
/// recovering the standard PEP 427 name find-links requires:
/// `x-1.0-py3-none-any.injected.autodata.relaxed.whl` ->
/// `x-1.0-py3-none-any.whl`. The infixes only ever appear in this
/// order (phase 1.5, 1.6, 2).
pub fn standard_wheel_filename(cached: &str) -> String {
    cached
        .replace(".injected.", ".")
        .replace(".autodata.", ".")
        .replace(".relaxed.", ".")
}

/// Lower bound of one PEP 440 specifier set: the smallest version
/// mentioned by an operator that bounds from below. `None` when the
/// set has no lower bound (a bare or `<x`-only requirement needs no
/// override).
fn lower_bound(specs: &uv_pep508::uv_pep440::VersionSpecifiers) -> Option<Version> {
    specs
        .iter()
        .filter(|s| {
            matches!(
                s.operator(),
                Operator::Equal
                    | Operator::ExactEqual
                    | Operator::TildeEqual
                    | Operator::GreaterThan
                    | Operator::GreaterThanEqual
            )
        })
        .map(|s| s.version().clone())
        .min()
}

/// `">=<floor>"` -- floor-only, deliberately uncapped. Two capped
/// shapes were tried and both failed live on isaac-pack:
/// min-pin-major caps forced ancient majors when wheels disagree
/// (attrs 17.x vs 25.x), and max-pin-major caps still excluded
/// whatever the CONDA side pinned (wheels pin psutil 5.9, conda env
/// pins 7.2.2). On the conda path the cascade widens iteratively on
/// solve failure; here the solve happens later inside pixi where
/// retread cannot iterate, so the only safe cap is none: the override
/// neutralizes upstream pins, and the actual version discipline comes
/// from the workspace's conda pins (via the conda-pypi mapping) and
/// uv's resolution. Epoch-carrying versions don't fit; callers fall
/// back to `"*"`.
fn floor_envelope(floor: &Version) -> Option<String> {
    if floor.epoch() != 0 {
        return None;
    }
    // Strip any local segment from the printed floor: PEP 440 forbids
    // local versions with `>=`-style comparisons in some resolvers'
    // strict modes, and the public release is what matters.
    let floor_clean = floor.clone().without_local();
    // Prerelease floors must be EXACT: pixi's uv does not treat a
    // `>=X.YrcN` floor as a per-package prerelease opt-in (found live:
    // tinyobjloader>=2.0.0rc13 fails with "pre-releases weren't
    // enabled"), but an exact `==X.YrcN` pin is explicit and resolves.
    if floor_clean.any_prerelease() {
        return Some(format!("=={floor_clean}"));
    }
    Some(format!(">={floor_clean}"))
}

/// Build the `[pypi-options.dependency-overrides]` map from every
/// wheel's (post-relax) Requires-Dist, and decide which wheels must
/// ship in find-links. Returned as one plan because the two decisions
/// are coupled through direct-URL requirements (below).
///
/// - Version-bounded lines get a floor-only envelope of the LOWEST
///   bound seen; names of shipped wheels are skipped (they get exact
///   pins in `[pypi-dependencies]`).
/// - Direct-URL lines (`pkg @ git+...` / `pkg @ https://...whl`):
///   uv forbids these transitively, but retread's BFS already resolved
///   every followed URL requirement into a wheel IN THE BUNDLE. The
///   general reroute: exact-pin override to the bundled version, and
///   force-ship the target wheel when it's a local artifact (uv can't
///   fetch a git-built wheel by name from any index). A URL line whose
///   name is NOT in the bundle was extras-gated and never followed --
///   nothing to reroute to; logged in case the consumer requests that
///   extra.
pub struct EmitPlan {
    /// Names (PEP 503) of wheels to copy into find-links: retread-built
    /// wheels plus local targets of direct-URL requirements.
    pub ship: std::collections::HashSet<String>,
    pub overrides: BTreeMap<String, String>,
    /// PEP 503 names of DEAD/ORPHAN direct-URL Requires-Dist targets:
    /// direct-URL (git/url) requirements whose target name is ABSENT from
    /// the resolved bundle closure (`bundle_versions`). retread did not
    /// follow them into the bundle — whether because the gating extra was
    /// inactive (e.g. isaaclab_mimic 1.2.x marked form) OR because they are
    /// unconditional deps retread chose not to bundle (e.g. isaaclab_mimic
    /// 1.3.2, which carries NO `; extra==` marker and NO `Provides-Extra`).
    /// The decision is MARKER-INDEPENDENT: bundle-membership is the sole
    /// predicate. Their Requires-Dist lines are STRIPPED from emitted wheel
    /// METADATA so uv does not see an orphan URL dependency and abort.
    ///
    /// This is bundle-MEMBERSHIP-based and does NOT consult `config.drop_deps`
    /// (which feeds the auto_bundle_transitives skip set + emit conda-run-dep
    /// filter, NOT the extras BFS `seen` set). A drop_deps name that is also
    /// an active-extra URL target stays in the bundle and is pinned via the
    /// Some-arm, not stripped (pre-existing behavior, out of scope). Phase 2.8.
    pub drop_url: std::collections::HashSet<String>,
}

/// `conda_capable`: PyPI names (PEP 503) for which the cascade's probe
/// found conda candidates. Bundle members in this set never get exact
/// pins -- the consuming env's conda side may pin ANY version of them
/// (requests 2.34.2 vs the bundled 2.32.3), and conda wins in pixi's
/// model. Exactness is reserved for names conda definitively lacks
/// (the isaacsim family), where it reproduces the BFS's exact-first
/// resolution. Same gate as the step-8 auto-bundle reroute.
pub fn plan(wheels: &[EmitWheel], conda_capable: &std::collections::HashSet<String>) -> EmitPlan {
    let bundle_versions: std::collections::HashMap<&str, &EmitWheel> =
        wheels.iter().map(|w| (w.pypi_name.as_str(), w)).collect();
    // Pass 1: every direct-URL requirement's target must resolve by
    // name -- exact pin to the bundled version, shipping the wheel
    // when it only exists locally.
    let mut ship: std::collections::HashSet<String> = wheels
        .iter()
        .filter(|w| w.must_ship())
        .map(|w| w.pypi_name.clone())
        .collect();
    let mut exact: BTreeMap<String, String> = BTreeMap::new();
    let mut drop_url: std::collections::HashSet<String> = std::collections::HashSet::new();
    for w in wheels {
        for line in &w.requires_dist {
            let req: uv_pep508::Requirement = match uv_pep508::Requirement::from_str(line) {
                Ok(r) => r,
                Err(_) => continue,
            };
            if !matches!(req.version_or_url, Some(uv_pep508::VersionOrUrl::Url(_))) {
                continue;
            }
            let name = req.name.to_string();
            match bundle_versions.get(name.as_str()) {
                Some(target) => {
                    exact.insert(name.clone(), format!("=={}", target.version));
                    if target.local_path.is_some() {
                        // Invariant: a wheel that enters the ship set via the
                        // local_path gate must NOT simultaneously carry a
                        // remote_url. Index-origin wheels and relax-changed
                        // index shadows (Origin::Built && !must_ship) are
                        // reconstructed with local_path:None and
                        // remote_url:Some(...) in the Class-2 arm of
                        // materialize_from_lock (handler/mod.rs), so they
                        // never reach this branch. Wheels that ARE here have
                        // either been retread-built (.injected, remote_url:None)
                        // or locally materialized from a direct url= source
                        // (also remote_url:None). If remote_url.is_some() here
                        // it means a future code path set local_path on an
                        // index-origin wheel and forgot to clear remote_url,
                        // which would corrupt the manifest by bundling a wheel
                        // the index should serve.
                        debug_assert!(
                            target.remote_url.is_none(),
                            "emit-pypi invariant violated: wheel `{}` is a direct-URL \
                             Requires-Dist target with local_path set but also has \
                             remote_url set -- index-origin / relax-changed index shadows \
                             must never be direct-URL targets (they carry remote_url \
                             instead of local_path)",
                            target.pypi_name
                        );
                        ship.insert(name);
                    }
                }
                None => {
                    // Orphan direct-URL dep: target absent from bundle closure.
                    // Strip its Requires-Dist line so uv does not see an
                    // unresolvable URL edge. Decision is MARKER-INDEPENDENT
                    // (bundle-membership only; see EmitPlan.drop_url doc).
                    tracing::info!(
                        requirer = %w.pypi_name,
                        requirement = %line,
                        "emit-pypi: stripping dead/orphan direct-URL requirement \
                         (target absent from bundle closure)",
                    );
                    drop_url.insert(name);
                }
            }
        }
    }
    // Pass 2: floor envelopes for version-bounded lines, skipping
    // anything already exact-pinned or shipped.
    let mut lowest: BTreeMap<String, Version> = BTreeMap::new();
    for w in wheels {
        for line in &w.requires_dist {
            let req: uv_pep508::Requirement = match uv_pep508::Requirement::from_str(line) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let name = req.name.to_string();
            if name == "python" || ship.contains(&name) || exact.contains_key(&name) {
                continue;
            }
            let Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) = req.version_or_url else {
                continue;
            };
            let Some(lower) = lower_bound(&specs) else {
                continue;
            };
            match lowest.entry(name) {
                std::collections::btree_map::Entry::Vacant(e) => {
                    e.insert(lower);
                }
                std::collections::btree_map::Entry::Occupied(mut e) => {
                    if lower < *e.get() {
                        e.insert(lower);
                    }
                }
            }
        }
    }
    let mut overrides: BTreeMap<String, String> = lowest
        .into_iter()
        .map(|(name, lower)| {
            // v1.5.9's exact-first principle, reconstructed: when the
            // requirement floor EQUALS the version the BFS resolved
            // into the bundle, the upstream pin was exact (relaxation
            // keeps the pinned version as the lower bound), so the
            // emitted env must reproduce it exactly -- a floor here
            // lets sibling families float to newer patches (isaacsim
            // 6.0.0.0 metapackage + 6.0.0.1 sub-wheels = Kit extension
            // breakage, the patch-drift class). Range floors that
            // don't match the bundled version stay floor-only.
            let bundled_exact = bundle_versions
                .get(name.as_str())
                .filter(|_| !conda_capable.contains(&name))
                .and_then(|w| {
                    let bundled = Version::from_str(&w.version).ok()?;
                    (bundled == lower).then(|| format!("=={}", w.version))
                });
            let value = bundled_exact
                .or_else(|| floor_envelope(&lower))
                .unwrap_or_else(|| "*".to_string());
            (name, value)
        })
        .collect();
    // Exact reroutes win over floor envelopes for the same name.
    overrides.extend(exact);
    EmitPlan {
        ship,
        overrides,
        drop_url,
    }
}

/// v1.7.0 blueprint mode: the override table's semantics as a
/// Requires-Dist line mapper, baked into shipped wheel METADATA via
/// [`crate::wheel_rewrite::rewrite_wheel_with`].
///
/// - Name in `drop_url` (Phase 2.8): return `Drop` — strip the orphan
///   direct-URL line. Checked FIRST so a dropped name is never also pinned.
/// - Name in `overrides`: rebuild the line with the override spec
///   (`"*"` means drop the specifier entirely). Returns `Keep` when the
///   rebuilt line equals the original (exact family pins stay
///   byte-identical -- no rewrite, no shadow wheel needed).
/// - Direct-URL lines whose name has an exact override: rebuilt as a
///   version pin (uv rejects transitive URL requirements; find-links
///   serves the pinned wheel).
/// - CAP-ONLY lines (`foo<2`, no lower bound) for conda-capable names:
///   the v1.6 table was structurally blind to these (floor envelopes
///   need a floor), but a cap can exclude whatever the consumer's
///   conda side pinned just like an exact pin can. Strip `<`/`<=`
///   bounds, keep `!=` exclusions.
pub fn override_line_map<'a>(
    overrides: &'a BTreeMap<String, String>,
    conda_capable: &'a std::collections::HashSet<String>,
    drop_url: &'a std::collections::HashSet<String>,
) -> impl Fn(&str) -> crate::wheel_rewrite::LineAction + 'a {
    use crate::wheel_rewrite::LineAction;
    move |line: &str| {
        let Ok(req) = uv_pep508::Requirement::from_str(line) else {
            return LineAction::Keep;
        };
        let name = req.name.to_string();
        if name == "python" {
            return LineAction::Keep;
        }
        // Drop check BEFORE overrides so a dropped name is never double-handled.
        if drop_url.contains(&name) {
            return LineAction::Drop;
        }
        if let Some(value) = overrides.get(&name) {
            let spec = if value == "*" {
                String::new()
            } else {
                value.clone()
            };
            let rebuilt = crate::wheel_rewrite::rebuild_requirement(&req, &spec);
            if rebuilt != line {
                return LineAction::Replace(rebuilt);
            }
            return LineAction::Keep;
        }
        // Cap-only handling for names without a table entry.
        if let Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) = req.version_or_url.as_ref()
        {
            let has_lower = lower_bound(specs).is_some();
            let has_cap = specs
                .iter()
                .any(|s| matches!(s.operator(), Operator::LessThan | Operator::LessThanEqual));
            if !has_lower && has_cap && conda_capable.contains(&name) {
                let kept: Vec<String> = specs
                    .iter()
                    .filter(|s| {
                        !matches!(s.operator(), Operator::LessThan | Operator::LessThanEqual)
                    })
                    .map(|s| s.to_string())
                    .collect();
                let rebuilt = crate::wheel_rewrite::rebuild_requirement(&req, &kept.join(","));
                if rebuilt != line {
                    return LineAction::Replace(rebuilt);
                }
                return LineAction::Keep;
            }
        }
        LineAction::Keep
    }
}

/// Insert a PEP 427 build tag into a standard wheel filename:
/// `dist-version-py-abi-plat.whl` -> `dist-version-TAG-py-abi-plat.whl`.
/// Wheel filenames escape `-` in dist names, so splitting on `-` is
/// well-defined (5 fields without a build tag, 6 with).
pub fn insert_build_tag(std_name: &str, tag: &str) -> Result<String> {
    let stem = std_name
        .strip_suffix(".whl")
        .ok_or_else(|| anyhow::anyhow!("not a wheel filename: {std_name}"))?;
    let parts: Vec<&str> = stem.split('-').collect();
    match parts.len() {
        5 => Ok(format!(
            "{}-{}-{tag}-{}-{}-{}.whl",
            parts[0], parts[1], parts[2], parts[3], parts[4]
        )),
        6 => Ok(format!(
            "{}-{}-{tag}-{}-{}-{}.whl",
            parts[0], parts[1], parts[3], parts[4], parts[5]
        )),
        n => Err(anyhow::anyhow!(
            "unexpected wheel filename shape ({n} fields): {std_name}"
        )),
    }
}

/// Generate the `<bundle>-pypi` meta-wheel: a pure dist-info wheel
/// whose Requires-Dist lines carry the pack's `[retread-wheels]` entry
/// pins (with extras), so the workspace block needs exactly one
/// `[pypi-dependencies]` line. Returns `(filename, bytes)`.
pub fn build_meta_wheel(
    bundle_name: &str,
    version: &str,
    entries: &[(String, WheelEntry, Option<String>)],
) -> (String, Vec<u8>) {
    use std::io::Write as _;
    let dist = format!("{}_pypi", bundle_name.replace('-', "_"));
    let di = format!("{dist}-{version}.dist-info");
    let mut metadata = format!(
        "Metadata-Version: 2.1\nName: {bundle_name}-pypi\nVersion: {version}\n\
         Summary: pixi-build-retread blueprint meta-package for {bundle_name}\n"
    );
    for (entry_name, entry, resolved) in entries {
        let extras = if entry.extras.is_empty() {
            String::new()
        } else {
            format!("[{}]", entry.extras.join(","))
        };
        let pin = entry
            .normalized_version()
            .map(|v| format!("=={v}"))
            .or_else(|| resolved.as_ref().map(|v| format!("=={v}")))
            .unwrap_or_default();
        metadata.push_str(&format!("Requires-Dist: {entry_name}{extras}{pin}\n"));
    }
    let metadata = metadata.into_bytes();
    let wheel_file = format!(
        "Wheel-Version: 1.0\nGenerator: pixi-build-retread {}\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        env!("CARGO_PKG_VERSION")
    )
    .into_bytes();
    let b64 = crate::wheel_inject::sha256_base64_urlsafe_nopad;
    let record = format!(
        "{di}/METADATA,sha256={},{}\n{di}/WHEEL,sha256={},{}\n{di}/RECORD,,\n",
        b64(&metadata),
        metadata.len(),
        b64(&wheel_file),
        wheel_file.len(),
    );
    let mut buf = Vec::new();
    {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            // Pin the timestamp (1980 DOS epoch) for byte-deterministic output
            // regardless of the zip `time` feature -- see wheel_rewrite.rs.
            .last_modified_time(zip::DateTime::default());
        for (name, body) in [
            (format!("{di}/METADATA"), metadata.as_slice()),
            (format!("{di}/WHEEL"), wheel_file.as_slice()),
            (format!("{di}/RECORD"), record.as_bytes()),
        ] {
            zip.start_file(&name, opts).expect("in-memory zip");
            zip.write_all(body).expect("in-memory zip");
        }
        zip.finish().expect("in-memory zip");
    }
    (format!("{dist}-{version}-py3-none-any.whl"), buf)
}

/// The prerelease micro-table for blueprint fence mode. uv builds its
/// explicit-prerelease set ONLY from direct requirements + overrides; a
/// meta-wheel's Requires-Dist line does NOT opt a package into prerelease
/// resolution. So every prerelease the blueprint depends on needs a
/// `"name" = "==<prerelease>"` override row or `pixi lock` fails
/// "pre-releases not enabled". Three disjoint sources, each invisible
/// to the others:
///   1. `overrides` `==<prerelease>` rows -- transitive prerelease deps
///      (the tinyobjloader case) that survived `plan()` Pass 2.
///   2. shipped/injected wheels whose own version is a prerelease --
///      Pass 2 skips `ship` names (emit_pypi `name == ... || ship.contains`),
///      so they never reach `overrides`.
///   3. `[retread-wheels]` entries pinned to a prerelease -- entry names
///      are stripped from `overrides` to protect their own pin.
///
/// The exact pin added equals the meta-wheel pin, so the pack never floats.
pub fn collect_prerelease_pins(
    overrides: &BTreeMap<String, String>,
    wheels: &[EmitWheel],
    ship: &std::collections::HashSet<String>,
    entries: &[(String, WheelEntry, Option<String>)],
) -> BTreeMap<String, String> {
    let is_prerelease = |v: &str| Version::from_str(v).is_ok_and(|ver| ver.any_prerelease());
    let mut pins: BTreeMap<String, String> = overrides
        .iter()
        .filter(|(_, v)| v.strip_prefix("==").is_some_and(is_prerelease))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for w in wheels {
        if ship.contains(&w.pypi_name) && is_prerelease(&w.version) {
            pins.insert(w.pypi_name.clone(), format!("=={}", w.version));
        }
    }
    for (key, entry, resolved) in entries {
        // Match build_meta_wheel's pin precedence EXACTLY (normalized
        // entry version first, then resolved). If these diverge, a
        // prerelease-pinned entry whose wheel lookup missed (resolved ==
        // None) or resolved to a different version would get a meta-wheel
        // `==<prerelease>` Requires-Dist line with no prerelease opt-in,
        // failing `pixi lock` -- the exact case source 3 exists to cover.
        if let Some(ver) = entry.normalized_version().or_else(|| resolved.clone())
            && is_prerelease(&ver)
        {
            pins.insert(crate::relax::canonical_conda_name(key), format!("=={ver}"));
        }
    }
    pins
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wheel(name: &str, version: &str, requires: &[&str], local: Option<&str>) -> EmitWheel {
        EmitWheel {
            pypi_name: name.into(),
            version: version.into(),
            requires_dist: requires.iter().map(|s| (*s).to_string()).collect(),
            wheel_filename: local
                .map(|p| {
                    PathBuf::from(p)
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_else(|| format!("{name}-{version}-py3-none-any.whl")),
            sha256: None,
            local_path: local.map(PathBuf::from),
            remote_url: None,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }
    }

    #[test]
    fn line_map_applies_table_semantics() {
        use crate::wheel_rewrite::LineAction;
        let mut overrides = BTreeMap::new();
        overrides.insert("pillow".into(), ">=8".into());
        overrides.insert("isaacsim-core".into(), "==6.0.0.0".into());
        overrides.insert("rl-games".into(), "==1.6.1".into());
        overrides.insert("loose".into(), "*".into());
        let capable: std::collections::HashSet<String> =
            ["psutil".to_string(), "pillow".to_string()].into();
        let drop: std::collections::HashSet<String> = std::collections::HashSet::new();
        let map = override_line_map(&overrides, &capable, &drop);
        // Floor override rewrites the pin.
        assert_eq!(
            map("pillow==12.1.1"),
            LineAction::Replace("pillow>=8".to_string())
        );
        // Exact override equal to the existing pin: no change -> no
        // shadow wheel (family pins stay byte-identical).
        assert_eq!(map("isaacsim-core==6.0.0.0"), LineAction::Keep);
        // URL requirement rerouted to the version pin.
        assert_eq!(
            map("rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11"),
            LineAction::Replace("rl-games==1.6.1".to_string())
        );
        // "*" drops the specifier; extras and markers survive.
        assert_eq!(
            map("loose[fast]==1.0 ; sys_platform == \"linux\""),
            LineAction::Replace("loose[fast] ; sys_platform == 'linux'".to_string())
        );
        // Cap-only line on a conda-capable name: cap stripped (the v1.6
        // table was structurally blind to these).
        assert_eq!(map("psutil<6"), LineAction::Replace("psutil".to_string()));
        // Cap-only on a non-capable name: left alone (cap is harmless,
        // conda can't pin above it).
        assert_eq!(map("notconda<2"), LineAction::Keep);
        // python is never touched.
        assert_eq!(map("python>=3.10"), LineAction::Keep);
    }

    /// Helper: build a minimal EmitWheel with requires_dist and no local path.
    fn remote_wheel(name: &str, version: &str, requires: &[&str]) -> EmitWheel {
        EmitWheel {
            pypi_name: name.into(),
            version: version.into(),
            requires_dist: requires.iter().map(|s| (*s).to_string()).collect(),
            wheel_filename: format!("{name}-{version}-py3-none-any.whl"),
            sha256: None,
            local_path: None,
            remote_url: None,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }
    }

    /// Test 1 (Phase 2.8 / Amendment 1): unconditional orphan URL dep is stripped.
    ///
    /// Mirrors the REAL isaaclab_mimic 1.3.2 bug: a `Requires-Dist` that is a
    /// direct-URL git line with NO `; extra ==` marker, whose target (`robomimic`)
    /// is NOT in the bundle. Confirms `drop_url` contains `robomimic`; the
    /// `overrides`/`ship` sets do NOT contain it.
    #[test]
    fn drop_predicate_strips_unconditional_orphan_url() {
        let mimic = remote_wheel(
            "isaaclab-mimic",
            "1.3.2",
            &["robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0"],
        );
        // robomimic is NOT in the bundle.
        let emit_plan = plan(&[mimic], &std::collections::HashSet::new());
        assert!(
            emit_plan.drop_url.contains("robomimic"),
            "unconditional orphan URL dep must be in drop_url; got: {:?}",
            emit_plan.drop_url
        );
        assert!(
            !emit_plan.ship.contains("robomimic"),
            "orphan must not be in ship"
        );
        assert!(
            !emit_plan.overrides.contains_key("robomimic"),
            "orphan must not be in overrides"
        );
    }

    /// Test 2 (Phase 2.8): marked orphan URL dep is also stripped.
    ///
    /// The 1.2.x isaaclab_mimic form: `robomimic @ git+…@v0.4.0 ; extra == "robomimic"`.
    /// Target absent from bundle → stripped. Confirms marker-independence: both
    /// marked and unmarked orphans use the same bundle-membership predicate.
    #[test]
    fn drop_predicate_strips_marked_orphan_url() {
        let mimic = remote_wheel(
            "isaaclab-mimic",
            "1.2.3",
            &[
                r#"robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0 ; extra == "robomimic""#,
            ],
        );
        let emit_plan = plan(&[mimic], &std::collections::HashSet::new());
        assert!(
            emit_plan.drop_url.contains("robomimic"),
            "marked orphan URL dep must be in drop_url; got: {:?}",
            emit_plan.drop_url
        );
    }

    /// Test 3 (Phase 2.8): active URL dep (target IN bundle) is NOT dropped.
    ///
    /// When the target `robomimic` IS in the bundle, it goes through the Some-arm
    /// → exact pin in overrides; it is NOT in drop_url.
    #[test]
    fn drop_predicate_keeps_active_url() {
        let robomimic = remote_wheel("robomimic", "0.4.0", &[]);
        let mimic = remote_wheel(
            "isaaclab-mimic",
            "1.3.2",
            &["robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0"],
        );
        let emit_plan = plan(&[robomimic, mimic], &std::collections::HashSet::new());
        assert!(
            !emit_plan.drop_url.contains("robomimic"),
            "active URL dep must NOT be in drop_url; overrides: {:?}",
            emit_plan.overrides
        );
        // It SHOULD be in overrides (exact pin from Some-arm).
        assert!(
            emit_plan.overrides.contains_key("robomimic"),
            "active URL dep must be in overrides as an exact pin; got: {:?}",
            emit_plan.overrides
        );
    }

    /// Test 4 (Phase 2.8): non-URL extras dep is never dropped.
    ///
    /// `foo>=1 ; extra == "x"` is NOT a URL requirement → continues before the
    /// bundle lookup → never enters drop_url.
    #[test]
    fn drop_predicate_ignores_non_url_extra_dep() {
        let w = remote_wheel("pkg", "1.0.0", &[r#"foo>=1 ; extra == "x""#]);
        // `foo` is NOT in the bundle.
        let emit_plan = plan(&[w], &std::collections::HashSet::new());
        assert!(
            !emit_plan.drop_url.contains("foo"),
            "non-URL dep must NOT be in drop_url; drop_url: {:?}",
            emit_plan.drop_url
        );
    }

    /// Test 5 (Phase 2.8): URL dep whose target IS a bundle entry is not dropped.
    ///
    /// A URL-named target that is also a top-level bundle entry hits the Some-arm
    /// (bundle_versions hit) → NOT in drop_url; and is exact-pinned.
    #[test]
    fn drop_predicate_ignores_url_config_entry() {
        // `special` is in the bundle as a top-level entry.
        let special = remote_wheel("special", "2.0.0", &[]);
        let requirer = remote_wheel(
            "requirer",
            "1.0.0",
            &["special @ git+https://github.com/example/special.git@main"],
        );
        let emit_plan = plan(&[special, requirer], &std::collections::HashSet::new());
        assert!(
            !emit_plan.drop_url.contains("special"),
            "URL dep whose target is a bundle entry must NOT be in drop_url; \
             drop_url: {:?}",
            emit_plan.drop_url
        );
        assert!(
            emit_plan.overrides.contains_key("special"),
            "URL dep whose target is a bundle entry must be exact-pinned; \
             overrides: {:?}",
            emit_plan.overrides
        );
    }

    /// Test 8 (Phase 2.8): drop takes precedence over overrides.
    ///
    /// A name that is in BOTH `drop_url` and `overrides` resolves to
    /// `LineAction::Drop` (drop wins; no double-handling).
    #[test]
    fn override_line_map_drop_precedence() {
        use crate::wheel_rewrite::LineAction;
        let mut overrides = BTreeMap::new();
        overrides.insert("robomimic".into(), "==0.4.0".into());
        let conda_capable: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut drop: std::collections::HashSet<String> = std::collections::HashSet::new();
        drop.insert("robomimic".into());

        let map = override_line_map(&overrides, &conda_capable, &drop);
        assert_eq!(
            map("robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0"),
            LineAction::Drop,
            "drop must win over overrides"
        );
    }

    #[test]
    fn build_tag_insertion() {
        assert_eq!(
            insert_build_tag(
                "isaacsim_kernel-6.0.0.0-cp312-none-manylinux_2_35_x86_64.whl",
                "999retread"
            )
            .unwrap(),
            "isaacsim_kernel-6.0.0.0-999retread-cp312-none-manylinux_2_35_x86_64.whl"
        );
        assert_eq!(
            insert_build_tag("isaaclab-4.5.22-py3-none-any.whl", "999retread").unwrap(),
            "isaaclab-4.5.22-999retread-py3-none-any.whl"
        );
        // Existing build tag gets replaced, not doubled.
        assert_eq!(
            insert_build_tag("foo-1.0-1-py3-none-any.whl", "999retread").unwrap(),
            "foo-1.0-999retread-py3-none-any.whl"
        );
        assert!(insert_build_tag("not-a-wheel.tar.gz", "x").is_err());
    }

    #[test]
    fn meta_wheel_is_valid_and_carries_entry_pins() {
        let entries = vec![
            (
                "isaacsim".to_string(),
                WheelEntry {
                    version: Some("==6.0.0.0".into()),
                    index: Some("https://pypi.nvidia.com".into()),
                    extras: vec!["all".into(), "extscache".into()],
                    ..Default::default()
                },
                None,
            ),
            (
                "isaaclab".to_string(),
                WheelEntry {
                    from: Some("isaaclab".into()),
                    ..Default::default()
                },
                Some("4.5.22".to_string()),
            ),
        ];
        let (name, bytes) = build_meta_wheel("isaac-pack", "6.0.0", &entries);
        assert_eq!(name, "isaac_pack_pypi-6.0.0-py3-none-any.whl");
        let mut z = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
        let mut meta = String::new();
        std::io::Read::read_to_string(
            &mut z
                .by_name("isaac_pack_pypi-6.0.0.dist-info/METADATA")
                .unwrap(),
            &mut meta,
        )
        .unwrap();
        assert!(meta.contains("Name: isaac-pack-pypi"));
        assert!(meta.contains("Requires-Dist: isaacsim[all,extscache]==6.0.0.0"));
        assert!(meta.contains("Requires-Dist: isaaclab==4.5.22"));
        // RECORD references all three members with b64 hashes.
        let mut rec = String::new();
        std::io::Read::read_to_string(
            &mut z.by_name("isaac_pack_pypi-6.0.0.dist-info/RECORD").unwrap(),
            &mut rec,
        )
        .unwrap();
        assert_eq!(rec.lines().count(), 3);
        assert!(rec.contains("METADATA,sha256="));
    }

    #[test]
    fn prerelease_pins_cover_all_three_sources() {
        use crate::config::WheelEntry;
        // 1. transitive prerelease surviving in overrides.
        let mut overrides = BTreeMap::new();
        overrides.insert("tinyobjloader".to_string(), "==2.0.0rc13".to_string());
        overrides.insert("requests".to_string(), ">=2.32.3".to_string()); // stable floor: excluded
        // 2. a SHIPPED wheel whose own version is a prerelease.
        let wheels = vec![
            wheel(
                "isaaclab",
                "0.3.0.dev0",
                &[],
                Some("/x/isaaclab.injected.whl"),
            ),
            wheel("stablebuilt", "1.0.0", &[], Some("/x/stablebuilt.whl")),
        ];
        let mut ship = std::collections::HashSet::new();
        ship.insert("isaaclab".to_string());
        ship.insert("stablebuilt".to_string());
        // 3. ENTRY prerelease cases. The pin precedence MUST match
        // build_meta_wheel: normalized entry version first, then resolved.
        let pre_entry = |ver: &str| WheelEntry {
            version: Some(ver.to_string()),
            ..WheelEntry::default()
        };
        let entries = vec![
            // 3a: pin == resolved (the common path).
            (
                "tinyobjloader-entry".to_string(),
                pre_entry("==2.0.0rc13"),
                Some("2.0.0rc13".to_string()),
            ),
            // 3b: prerelease PIN but resolved is a stable/different version.
            // The meta-wheel pins ==1.0.0rc1 from normalized_version, so the
            // override row MUST too -- reading `resolved` alone (stable)
            // would emit no row and break `pixi lock`.
            (
                "skewed-entry".to_string(),
                pre_entry("==1.0.0rc1"),
                Some("1.0.0".to_string()),
            ),
            // 3c: prerelease pin, wheel lookup missed entirely (resolved None).
            ("missed-entry".to_string(), pre_entry("==3.0.0b2"), None),
            // 3d: stable entry -- no row.
            (
                "stable-entry".to_string(),
                pre_entry("==1.2.3"),
                Some("1.2.3".to_string()),
            ),
        ];
        let pins = collect_prerelease_pins(&overrides, &wheels, &ship, &entries);
        // source 1: transitive prerelease override kept.
        assert_eq!(
            pins.get("tinyobjloader").map(String::as_str),
            Some("==2.0.0rc13")
        );
        // source 2: shipped prerelease wheel got a row; stable shipped did not.
        assert_eq!(
            pins.get("isaaclab").map(String::as_str),
            Some("==0.3.0.dev0")
        );
        assert!(!pins.contains_key("stablebuilt"));
        // source 3a: pin == resolved.
        assert_eq!(
            pins.get("tinyobjloader-entry").map(String::as_str),
            Some("==2.0.0rc13")
        );
        // source 3b: normalized prerelease pin wins over a stable resolved.
        assert_eq!(
            pins.get("skewed-entry").map(String::as_str),
            Some("==1.0.0rc1")
        );
        // source 3c: prerelease pin honored even when resolved is None.
        assert_eq!(
            pins.get("missed-entry").map(String::as_str),
            Some("==3.0.0b2")
        );
        // source 3d: stable entry got no row.
        assert!(!pins.contains_key("stable-entry"));
        // stable floor override is not a prerelease row.
        assert!(!pins.contains_key("requests"));
    }

    #[test]
    fn standard_filename_strips_all_infixes() {
        for (cached, want) in [
            (
                "isaaclab-4.5.22-py3-none-any.injected.autodata.relaxed.whl",
                "isaaclab-4.5.22-py3-none-any.whl",
            ),
            (
                "isaaclab_rl-0.5.0-py3-none-any.injected.relaxed.whl",
                "isaaclab_rl-0.5.0-py3-none-any.whl",
            ),
            ("x-1.0-py3-none-any.injected.whl", "x-1.0-py3-none-any.whl"),
            ("gym-0.26.2-py3-none-any.whl", "gym-0.26.2-py3-none-any.whl"),
        ] {
            assert_eq!(standard_wheel_filename(cached), want, "from {cached}");
        }
    }

    #[test]
    fn must_ship_only_injected() {
        assert!(
            wheel(
                "a",
                "1",
                &[],
                Some("/w/a-1-py3-none-any.injected.relaxed.whl")
            )
            .must_ship()
        );
        // Relaxed copy of an index wheel: reachable upstream, don't ship.
        assert!(
            !wheel(
                "isaacsim",
                "6.0.0.0",
                &[],
                Some("/w/isaacsim-6.0.0.0-cp312-none-manylinux_2_35_x86_64.relaxed.whl"),
            )
            .must_ship()
        );
        // Remote wheel: nothing local to ship.
        assert!(!wheel("b", "1", &[], None).must_ship());
    }

    #[test]
    fn override_table_envelopes_and_merges() {
        let wheels = [
            wheel(
                "isaacsim-core",
                "6.0.0.0",
                &[
                    "pillow==12.1.1",
                    "kiwisolver>=1.4.9,<1.5",
                    "tinyobjloader==2.0.0rc13",
                    "toml", // no lower bound: no override
                ],
                None,
            ),
            wheel(
                "isaaclab",
                "4.5.22",
                &[
                    "pillow==12.0.0",   // lower than core's pin: wins
                    "isaaclab-rl>=0.5", // shipped: skipped
                ],
                Some("/w/isaaclab-4.5.22-py3-none-any.injected.autodata.relaxed.whl"),
            ),
            wheel(
                "isaaclab-rl",
                "0.5.0",
                &[],
                Some("/w/isaaclab_rl-0.5.0-py3-none-any.injected.relaxed.whl"),
            ),
        ];
        let table = plan(&wheels, &Default::default()).overrides;
        assert_eq!(table.get("pillow").unwrap(), ">=12.0.0");
        assert_eq!(table.get("kiwisolver").unwrap(), ">=1.4.9");
        // Prerelease floors become EXACT pins: pixi's uv rejects a
        // prerelease `>=` floor as non-explicit (found live).
        assert_eq!(table.get("tinyobjloader").unwrap(), "==2.0.0rc13");
        assert!(!table.contains_key("toml"), "unbounded line: no override");
        assert!(
            !table.contains_key("isaaclab-rl"),
            "shipped wheels are exact-pinned in pypi-dependencies, not overridden"
        );
    }

    #[test]
    fn override_envelope_admits_disagreeing_majors_and_conda_pins() {
        // Both found live on isaac-pack. (1) one wheel pins ancient
        // attrs (>=17.3.0), siblings pin CalVer attrs 25.x -- a cap at
        // the lowest pin's major forced the ancient major. (2) wheels
        // pin psutil 5.9.x but the CONSUMER's conda env pins psutil
        // 7.2.2 -- ANY cap derived from wheel pins can exclude the
        // conda choice, and unlike the conda cascade there is no
        // iterate-on-unsat here. Floor-only is the invariant.
        let wheels = [
            wheel("old", "1", &["attrs>=17.3.0", "psutil>=5.9.0,<6"], None),
            wheel("new", "1", &["attrs==25.1.0"], None),
        ];
        let table = plan(&wheels, &Default::default()).overrides;
        assert_eq!(table.get("attrs").unwrap(), ">=17.3.0");
        assert_eq!(table.get("psutil").unwrap(), ">=5.9.0");
    }

    #[test]
    fn bundle_members_with_matching_floor_pin_exact() {
        // The patch-drift class, caught live: isaacsim's family wheels
        // exact-pin each other at 6.0.0.0; relaxation turns those into
        // floors, and a floor-only override let uv float the whole
        // family to 6.0.0.1 (Kit extension breakage -- the exact bug
        // v1.5.9 fixed on the conda path). floor == bundled version
        // means the upstream pin was exact: reproduce it exactly.
        // scipy-style genuine ranges (floor != bundled version) stay
        // floor-only so conda-pinned versions can satisfy them.
        let wheels = [
            wheel(
                "isaacsim",
                "6.0.0.0",
                &["isaacsim-core==6.0.0.0", "scipy>=1.14"],
                None,
            ),
            wheel("isaacsim-core", "6.0.0.0", &[], None),
            wheel("scipy", "1.17.0", &[], None),
        ];
        let table = plan(&wheels, &Default::default()).overrides;
        assert_eq!(table.get("isaacsim-core").unwrap(), "==6.0.0.0");
        assert_eq!(table.get("scipy").unwrap(), ">=1.14");
    }

    #[test]
    fn url_requirements_reroute_to_bundle_generally() {
        // uv rejects transitive direct-URL requirements outright, but
        // retread's BFS resolved every followed URL requirement into a
        // wheel IN THE BUNDLE. The reroute is fully general: exact pin
        // to the bundled version, force-shipping local targets that no
        // index serves. Found live via isaaclab-rl's
        // `rl-games @ git+...` but deliberately not specific to it.
        let wheels = [
            wheel(
                "isaaclab-rl",
                "0.5.0",
                &[
                    "rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11",
                    // URL target that is local but NOT injected (e.g. an
                    // sdist-built or url= wheel): must be force-shipped.
                    "somepkg @ https://example.com/somepkg-2.0-py3-none-any.whl",
                ],
                Some("/w/isaaclab_rl-0.5.0-py3-none-any.injected.relaxed.whl"),
            ),
            wheel(
                "rl-games",
                "1.6.1",
                &[],
                Some("/w/rl_games-1.6.1-py3-none-any.injected.relaxed.whl"),
            ),
            wheel(
                "somepkg",
                "2.0",
                &[],
                Some("/w/somepkg-2.0-py3-none-any.whl"),
            ),
        ];
        let p = plan(&wheels, &Default::default());
        assert_eq!(p.overrides.get("rl-games").unwrap(), "==1.6.1");
        assert_eq!(p.overrides.get("somepkg").unwrap(), "==2.0");
        assert!(p.ship.contains("rl-games"), "injected target ships");
        assert!(
            p.ship.contains("somepkg"),
            "plain local URL target must be force-shipped: no index serves it by name"
        );
        // URL requirement never followed into the bundle (extras-gated):
        // nothing to reroute to; left out (warned at emit time).
        let wheels = [wheel(
            "isaaclab-mimic",
            "1.2.3",
            &["robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@abc"],
            Some("/w/isaaclab_mimic-1.2.3-py3-none-any.injected.relaxed.whl"),
        )];
        assert!(
            !plan(&wheels, &Default::default())
                .overrides
                .contains_key("robomimic")
        );
    }

    #[test]
    fn override_envelope_skips_epoch_and_local() {
        // NOTE: local versions are only PEP 440-legal with `==`/`!=`
        // comparisons, so the exact-pin form is the realistic shape
        // (pytorch3d's `==0.7.9+d9839a9pt2.10.0cu128` style).
        let wheels = [wheel(
            "a",
            "1",
            &["weird==1!2.0", "pytorch3d==0.7.9+d9839a9pt2.10.0cu128"],
            None,
        )];
        let table = plan(&wheels, &Default::default()).overrides;
        assert_eq!(table.get("weird").unwrap(), "*", "epoch falls back to *");
        // Local segment stripped from the envelope's lower bound.
        assert_eq!(table.get("pytorch3d").unwrap(), ">=0.7.9");
    }

    /// Step-3 parity assertion: plan() is a pure function of its inputs.
    /// Identical (requires_dist, version, must_ship/filename, local_path.is_some(),
    /// conda_capable) must produce identical EmitPlan (overrides BTreeMap eq,
    /// ship as sorted Vec).
    ///
    /// This guards against the #4 parity bug regression: if requires_dist is
    /// empty for index wheels, the override table on replay is a SUBSET of the
    /// cold-produce table, which can flip a relax-shadow back to Origin::Index
    /// and poison the lock.
    #[test]
    fn plan_purity_identical_inputs_identical_output() {
        use std::collections::HashSet;

        // A bundle with an index wheel (isaacsim-style) that has Requires-Dist
        // that the relax policy would change, PLUS a must_ship wheel that
        // requires it via URL (so plan()'s Pass-1 fires).
        let injected_path = "/tmp/rl-games-1.6.1.injected.whl";
        let wheels_a = vec![
            wheel(
                "isaacsim-core",
                "6.0.0.0",
                &["numpy>=1.24", "pillow==12.1.1"],
                None,
            ),
            wheel(
                "rl-games",
                "1.6.1",
                &["isaacsim-core @ https://pypi.nvidia.com/isaacsim_core-6.0.0.0-py3-none-any.whl"],
                Some(injected_path),
            ),
        ];
        // Identical inputs (clone).
        let wheels_b = wheels_a.clone();

        let mut conda_capable: HashSet<String> = HashSet::new();
        conda_capable.insert("pillow".to_string());

        let plan_a = plan(&wheels_a, &conda_capable);
        let plan_b = plan(&wheels_b, &conda_capable);

        // Overrides must be byte-identical (BTreeMap, deterministic).
        assert_eq!(
            plan_a.overrides, plan_b.overrides,
            "plan() overrides must be deterministic for identical inputs"
        );

        // Ship set: compare as sorted Vec for stable comparison.
        let mut ship_a: Vec<_> = plan_a.ship.iter().cloned().collect();
        let mut ship_b: Vec<_> = plan_b.ship.iter().cloned().collect();
        ship_a.sort();
        ship_b.sort();
        assert_eq!(
            ship_a, ship_b,
            "plan() ship set must be deterministic for identical inputs"
        );

        // Verify the expected semantics:
        // rl-games is in ship (must_ship=true, .injected filename).
        assert!(
            plan_a.ship.contains("rl-games"),
            "rl-games (.injected) must be in ship set"
        );
        // isaacsim-core is NOT in ship: must_ship=false (no .injected) AND
        // local_path=None -> Pass-1 URL-target check skips the ship.insert.
        // But it DOES get an exact-pin override (==6.0.0.0) because it's a
        // URL-requirement target.
        assert!(
            !plan_a.ship.contains("isaacsim-core"),
            "isaacsim-core (index wheel, no local_path) must NOT be in ship set"
        );
        assert_eq!(
            plan_a.overrides.get("isaacsim-core").map(|s| s.as_str()),
            Some("==6.0.0.0"),
            "isaacsim-core must get exact pin from Pass-1 URL-target override"
        );
    }
}
