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
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result};
use uv_pep508::uv_pep440::{Operator, Version};

use crate::config::{RetreadConfig, WheelEntry};

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
    /// Upstream URL when the wheel was never materialized locally
    /// (PEP 658 sidecar metadata). Blueprint mode fetches these on
    /// demand when their Requires-Dist needs rewriting.
    pub remote_url: Option<url::Url>,
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

/// Highest glibc floor among the bundle's `manylinux_X_Y` wheel tags,
/// as a `"X.Y"` string. pixi's uv assumes a conservative baseline
/// glibc unless the feature declares `system-requirements.libc`;
/// without this block, manylinux_2_35 wheels (isaacsim,
/// omniverseclient) are "no matching platform tag" at solve time.
pub fn manylinux_floor(wheels: &[EmitWheel]) -> Option<String> {
    let tag = regex::Regex::new(r"manylinux_(\d+)_(\d+)").expect("static regex");
    let mut max: Option<(u32, u32)> = None;
    for w in wheels {
        for cap in tag.captures_iter(&w.wheel_filename) {
            let pair = (
                cap[1].parse::<u32>().unwrap_or(0),
                cap[2].parse::<u32>().unwrap_or(0),
            );
            if max.is_none_or(|m| pair > m) {
                max = Some(pair);
            }
        }
    }
    max.map(|(major, minor)| format!("{major}.{minor}"))
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
                        ship.insert(name);
                    }
                }
                None => {
                    tracing::warn!(
                        requirer = %w.pypi_name,
                        requirement = %line,
                        "emit-pypi: direct-URL requirement was not followed into the \
                         bundle (extras-gated); uv rejects these transitively -- if a \
                         requested extra pulls it, add it as a top-level \
                         pypi-dependency by hand",
                    );
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
    EmitPlan { ship, overrides }
}

/// Render one `[pypi-dependencies]` line for a `[retread-wheels]`
/// entry. Built entries pin the resolved version exactly (find-links
/// serves the wheel); index entries pass the user's pin + index
/// through so uv fetches upstream directly.
fn render_dependency_line(
    entry_name: &str,
    entry: &WheelEntry,
    resolved_version: Option<&str>,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(url) = &entry.url {
        parts.push(format!("url = \"{url}\""));
    } else if entry.from.is_some() || entry.git.is_some() || entry.path.is_some() {
        let v = resolved_version.unwrap_or("*");
        let pin = if v == "*" {
            "*".into()
        } else {
            format!("=={v}")
        };
        parts.push(format!("version = \"{pin}\""));
    } else {
        let v = entry
            .normalized_version()
            .map(|v| format!("=={v}"))
            .unwrap_or_else(|| "*".to_string());
        parts.push(format!("version = \"{v}\""));
        if let Some(index) = &entry.index {
            parts.push(format!("index = \"{index}\""));
        }
    }
    if !entry.extras.is_empty() {
        let extras = entry
            .extras
            .iter()
            .map(|e| format!("\"{e}\""))
            .collect::<Vec<_>>()
            .join(", ");
        parts.push(format!("extras = [{extras}]"));
    }
    format!("{entry_name} = {{ {} }}", parts.join(", "))
}

/// Render the paste-ready snippet. `pack_dir_name` is the pack
/// folder's name (e.g. "isaac-pack"); the find-links path is written
/// relative to a workspace manifest that has the pack as a direct
/// child folder -- the comment tells the user to adjust otherwise.
pub fn render_snippet(
    bundle_name: &str,
    pack_dir_name: &str,
    entries: &[(String, WheelEntry, Option<String>)],
    overrides: &BTreeMap<String, String>,
    libc_floor: Option<&str>,
    retread_version: &str,
) -> String {
    let feature = format!("{bundle_name}-pypi");
    let mut out = String::new();
    out.push_str(&format!(
        "# Generated by pixi-build-retread {retread_version} (retread-emit-pypi, experimental).\n\
         # Regenerated on every build of `{bundle_name}` -- do not hand-edit.\n\
         # Paste into the workspace pixi.toml and add an environment using the\n\
         # `{feature}` feature. The find-links path below is relative to the\n\
         # workspace manifest and assumes the pack folder `{pack_dir_name}/` is a\n\
         # direct child of the workspace; adjust if it lives elsewhere.\n\n"
    ));
    out.push_str(&format!("[feature.{feature}.pypi-options]\n"));
    out.push_str(&format!(
        "find-links = [{{ path = \"{pack_dir_name}/retread-pypi/{bundle_name}/wheels\" }}]\n\n"
    ));
    out.push_str(&format!(
        "[feature.{feature}.pypi-options.dependency-overrides]\n"
    ));
    out.push_str(
        "# Floor-only envelopes of every lower-bounded Requires-Dist line in the\n\
         # bundle. They neutralize upstream exact pins (which D-rewriting handles\n\
         # on the conda path) so conda-pinned and uv-chosen versions can satisfy\n\
         # them. A prerelease lower bound doubles as uv's per-package prerelease\n\
         # opt-in.\n",
    );
    for (name, value) in overrides {
        out.push_str(&format!("\"{name}\" = \"{value}\"\n"));
    }
    out.push_str(&format!("\n[feature.{feature}.pypi-dependencies]\n"));
    for (entry_name, entry, resolved_version) in entries {
        out.push_str(&render_dependency_line(
            entry_name,
            entry,
            resolved_version.as_deref(),
        ));
        out.push('\n');
    }
    if let Some(libc) = libc_floor {
        out.push_str(&format!(
            "\n# Highest manylinux glibc floor among the bundle's wheels. Without\n\
             # this, uv assumes a lower baseline and rejects manylinux_{}_* wheels\n\
             # as \"no matching platform tag\".\n\
             [feature.{feature}.system-requirements]\n\
             libc = \"{libc}\"\n",
            libc.replace('.', "_"),
        ));
    }
    out
}

/// v1.7.0 blueprint mode: the override table's semantics as a
/// Requires-Dist line mapper, baked into shipped wheel METADATA via
/// [`crate::wheel_rewrite::rewrite_wheel_with`].
///
/// - Name in `overrides`: rebuild the line with the override spec
///   (`"*"` means drop the specifier entirely). Returns `None` when the
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
) -> impl Fn(&str) -> Option<String> + 'a {
    move |line: &str| {
        let req: uv_pep508::Requirement = uv_pep508::Requirement::from_str(line).ok()?;
        let name = req.name.to_string();
        if name == "python" {
            return None;
        }
        if let Some(value) = overrides.get(&name) {
            let spec = if value == "*" {
                String::new()
            } else {
                value.clone()
            };
            let rebuilt = crate::wheel_rewrite::rebuild_requirement(&req, &spec);
            return (rebuilt != line).then_some(rebuilt);
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
                return (rebuilt != line).then_some(rebuilt);
            }
        }
        None
    }
}

/// Is a Requires-Dist line's rewrite LOAD-BEARING for the solve?
/// True when the original line constrains from ABOVE -- an exact pin,
/// a cap, or `~=` (implicit cap) -- on a name the map rewrites, or a
/// direct-URL requirement being rerouted. Floor-only loosening
/// (`sympy>=1.13` -> `sympy>=1.5`) is almost never what makes a solve
/// fail, and shipping every closure wheel with a floor hit dragged in
/// 512MB of torch that the conda side satisfies anyway.
pub fn load_bearing_rewrite<'a>(
    overrides: &'a BTreeMap<String, String>,
    conda_capable: &'a std::collections::HashSet<String>,
) -> impl Fn(&str) -> bool + 'a {
    move |line: &str| {
        let Ok(req) = uv_pep508::Requirement::from_str(line) else {
            return false;
        };
        let req: uv_pep508::Requirement = req;
        let name = req.name.to_string();
        if name == "python" {
            return false;
        }
        match req.version_or_url.as_ref() {
            Some(uv_pep508::VersionOrUrl::Url(_)) => overrides.contains_key(&name),
            Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) => {
                let has_upper = specs.iter().any(|s| {
                    matches!(
                        s.operator(),
                        Operator::Equal
                            | Operator::ExactEqual
                            | Operator::TildeEqual
                            | Operator::LessThan
                            | Operator::LessThanEqual
                    )
                });
                has_upper && (overrides.contains_key(&name) || conda_capable.contains(&name))
            }
            None => false,
        }
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
            .compression_method(zip::CompressionMethod::Deflated);
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

/// The blueprint-mode workspace block: static, tiny, identical across
/// rebuilds unless the bundle version or libc floor changes.
pub fn render_snippet_blueprint(
    bundle_name: &str,
    pack_rel: &str,
    version: &str,
    libc_floor: Option<&str>,
    prerelease_pins: &BTreeMap<String, String>,
    retread_version: &str,
) -> String {
    let feature = format!("{bundle_name}-pypi");
    let mut out = format!(
        "# Generated by pixi-build-retread {retread_version} (retread-blueprint, experimental).\n\
         # Static: override semantics live in the wheels under find-links, not here.\n\
         # Reference the `{feature}` feature from an environment to use it. After a\n\
         # pack change, re-lock (`pixi lock`): pixi's lock check does not inspect\n\
         # find-links directory contents.\n\n\
         [feature.{feature}.pypi-options]\n\
         find-links = [{{ path = \"{pack_rel}/retread-pypi/{bundle_name}/wheels\" }}]\n\n\
         [feature.{feature}.pypi-dependencies]\n\
         {feature} = \"=={version}\"\n"
    );
    if !prerelease_pins.is_empty() {
        // The ONE class overrides remain necessary for: pixi's uv only
        // honors prerelease pins in DIRECT requirements (workspace
        // overrides qualify; wheel METADATA does not), so the handful
        // of prerelease-only deps ride here.
        out.push_str(&format!(
            "\n[feature.{feature}.pypi-options.dependency-overrides]\n"
        ));
        for (name, value) in prerelease_pins {
            out.push_str(&format!("\"{name}\" = \"{value}\"\n"));
        }
    }
    if let Some(libc) = libc_floor {
        out.push_str(&format!(
            "\n[feature.{feature}.system-requirements]\nlibc = \"{libc}\"\n"
        ));
    }
    out
}

fn fence_open(bundle_name: &str) -> String {
    format!("# >>> pixi-build-retread emit-pypi: {bundle_name} (generated; do not edit) >>>")
}

fn fence_close(bundle_name: &str) -> String {
    format!("# <<< pixi-build-retread emit-pypi: {bundle_name} <<<")
}

/// Maintain this bundle's fenced block inside a workspace manifest.
/// Returns the new manifest text when a write is needed, `None` when
/// the manifest is already up to date. Markers absent: append the
/// fenced block at EOF. Both markers present: replace the interior.
/// Corrupted fencing (one marker, or close before open) returns an
/// error -- never guess inside someone's manifest.
pub fn sync_workspace_block(
    manifest_text: &str,
    bundle_name: &str,
    block: &str,
) -> Result<Option<String>> {
    let open = fence_open(bundle_name);
    let close = fence_close(bundle_name);
    let fenced = format!("{open}\n{}\n{close}\n", block.trim_end());
    match (manifest_text.find(&open), manifest_text.find(&close)) {
        (None, None) => {
            let mut out = manifest_text.to_string();
            if !out.ends_with('\n') {
                out.push('\n');
            }
            out.push('\n');
            out.push_str(&fenced);
            Ok(Some(out))
        }
        (Some(start), Some(close_at)) if close_at > start => {
            let end = close_at + close.len();
            // Swallow one trailing newline after the close marker so
            // repeated syncs don't accumulate blank lines.
            let end = if manifest_text[end..].starts_with('\n') {
                end + 1
            } else {
                end
            };
            if &manifest_text[start..end] == fenced.as_str() {
                return Ok(None);
            }
            let mut out = String::with_capacity(manifest_text.len() + fenced.len());
            out.push_str(&manifest_text[..start]);
            out.push_str(&fenced);
            out.push_str(&manifest_text[end..]);
            Ok(Some(out))
        }
        _ => Err(anyhow::anyhow!(
            "corrupted emit-pypi fence for `{bundle_name}` (markers missing or reversed); \
             fix or delete the fenced block by hand"
        )),
    }
}

/// Write the side-channel for one bundle: copy must-ship wheels under
/// standard filenames and write the snippet. Idempotent; the wheels
/// dir is recreated from scratch each build so renamed/removed entries
/// can't leave stale wheels behind.
///
/// When `workspace_dir` is known, the snippet's feature block is also
/// synced into the workspace manifest inside a fenced, machine-owned
/// region -- the user references the `<bundle>-pypi` feature from an
/// environment once and never touches generated content. The write is
/// atomic (tmp + rename) and skipped entirely when the block is
/// already current, so pixi's manifest mtime only moves on real
/// changes. NOTE: pixi reads the manifest once per invocation, so a
/// semantic change to the pack lands in the NEXT pixi run -- loudly
/// logged.
#[allow(clippy::too_many_arguments)]
pub async fn emit(
    bundle_name: &str,
    bundle_version: &str,
    source_dir: &Path,
    workspace_dir: Option<&Path>,
    wheels: &[EmitWheel],
    closure_wheels: &[EmitWheel],
    conda_capable: &std::collections::HashSet<String>,
    config: &RetreadConfig,
) -> Result<PathBuf> {
    let root = source_dir.join("retread-pypi").join(bundle_name);
    let wheels_dir = root.join("wheels");
    if wheels_dir.exists() {
        tokio::fs::remove_dir_all(&wheels_dir)
            .await
            .with_context(|| format!("clearing stale {}", wheels_dir.display()))?;
    }
    tokio::fs::create_dir_all(&wheels_dir)
        .await
        .with_context(|| format!("creating {}", wheels_dir.display()))?;

    let emit_plan = plan(wheels, conda_capable);

    // [retread-wheels] entries belonging to THIS bundle become the
    // [pypi-dependencies] block. Standalone entries (no bundle group)
    // match when the entry key IS the bundle name.
    let entries: Vec<(String, WheelEntry, Option<String>)> = config
        .retread_wheels
        .iter()
        .filter(|(key, entry)| {
            let group = entry.bundle.as_deref().or(config.default_bundle.as_deref());
            match group {
                Some(g) => g == bundle_name,
                None => key.as_str() == bundle_name,
            }
        })
        .map(|(key, entry)| {
            let resolved = wheels
                .iter()
                .find(|w| w.pypi_name == crate::relax::canonical_conda_name(key))
                .map(|w| w.version.clone());
            (key.clone(), entry.clone(), resolved)
        })
        .collect();

    // Entry pins are authoritative: uv overrides REPLACE direct
    // requirements too, so an override generated from some wheel's
    // range requirement on an entry name (`isaacsim >= 5.1`) would
    // clobber the entry's own `==6.0.0.0` pin and let the whole pack
    // float. Names with a [pypi-dependencies] entry never get an
    // override row.
    let mut emit_plan = emit_plan;
    for (key, _, _) in &entries {
        emit_plan
            .overrides
            .remove(&crate::relax::canonical_conda_name(key));
    }

    // find-links must be relative to the workspace manifest. With a
    // known workspace, compute the true relative path; otherwise fall
    // back to assuming the pack folder is a direct child.
    let pack_rel = workspace_dir
        .and_then(|w| source_dir.strip_prefix(w).ok())
        .and_then(|p| p.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            source_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(".")
                .to_string()
        });
    let libc_floor = manylinux_floor(wheels);

    let mut shipped = 0usize;
    let snippet = if config.blueprint.is_on() {
        // v1.7.0 blueprint: bake the override semantics into wheel
        // METADATA. Changed wheels ship build-tagged (uv prefers the
        // tagged wheel over the registry original at the same
        // version); unchanged wheels ship only when no index serves
        // them (built/.injected or URL targets); everything else stays
        // remote. Entry pins live in the generated meta-wheel.
        // Override map and floors derive from the BUNDLE wheels only;
        // closure wheels are merely CHECKED against it (their lines
        // joining the table would inflate it with the whole
        // ecosystem's floors -- found live: 398 shipped wheels).
        let mapper = override_line_map(&emit_plan.overrides, conda_capable);
        let load_bearing = load_bearing_rewrite(&emit_plan.overrides, conda_capable);
        let bundle_names: std::collections::HashSet<&str> =
            wheels.iter().map(|w| w.pypi_name.as_str()).collect();
        for w in wheels.iter().chain(closure_wheels.iter()) {
            // Closure wheels (not pack members) ship only when their
            // rewrite is load-bearing: caps/exact pins that could make
            // the consumer's solve fail. Floor-only hits are logged
            // and left on the index.
            if !bundle_names.contains(w.pypi_name.as_str()) {
                let bearing = w.requires_dist.iter().any(|l| load_bearing(l));
                if !bearing {
                    if w.requires_dist.iter().any(|l| mapper(l).is_some()) {
                        tracing::debug!(
                            wheel = %w.pypi_name,
                            "blueprint: closure wheel has floor-only rewrites; staying on index",
                        );
                    }
                    continue;
                }
            }
            let src: PathBuf = match w.local_path.as_ref() {
                Some(p) => p.clone(),
                None => {
                    // Remote wheel (sidecar metadata, bytes never
                    // downloaded). When its pins need rewriting, fetch
                    // it now -- into the pack's persistent wheels/
                    // cache so regens don't re-download. Found live:
                    // the auto-bundled nvidia-srl chain (numpy<2 caps)
                    // arrives exactly this way.
                    if !w.requires_dist.iter().any(|l| mapper(l).is_some()) {
                        continue;
                    }
                    let Some(url) = w.remote_url.as_ref() else {
                        tracing::warn!(
                            wheel = %w.pypi_name,
                            "blueprint: wheel needs rewriting but has neither local \
                             bytes nor a URL; skipping",
                        );
                        continue;
                    };
                    tracing::info!(
                        wheel = %w.pypi_name,
                        url = %url,
                        "blueprint: fetching remote wheel whose pins need rewriting",
                    );
                    crate::wheel::fetch_wheel(url, None, &source_dir.join("wheels"))
                        .await
                        .with_context(|| format!("blueprint fetch of {url}"))?
                }
            };
            let src = &src;
            let cached_name = src
                .file_name()
                .and_then(|n| n.to_str())
                .context("wheel path has no filename")?;
            // Map from the PRE-D wheel (original Requires-Dist), not
            // the relaxed final. Two reasons, both found live: (1) the
            // index serves the ORIGINAL bytes, so "mapper changed
            // nothing vs the relaxed copy" says nothing about whether
            // the index original needs shadowing (nvidia-srl's
            // numpy<2 cap survived exactly this way); (2) original
            // exact family pins are what blueprint mode wants to
            // preserve -- D's patch-relaxation is a conda-path
            // concern. The pre-D file always exists alongside the
            // final (`.relaxed` is the last suffix in the chain).
            let pre_d_name = cached_name
                .strip_suffix(".relaxed.whl")
                .map(|s| format!("{s}.whl"))
                .unwrap_or_else(|| cached_name.to_string());
            let base_src = src.with_file_name(&pre_d_name);
            let base_src = if base_src.is_file() {
                base_src
            } else {
                src.clone()
            };
            let std_name = standard_wheel_filename(cached_name);
            let tmp = wheels_dir.join(format!(".tmp-{std_name}"));
            // Per-wheel isolation: a single unrewritable wheel (e.g. a
            // RECORD that doesn't list its own METADATA -- tensordict
            // 0.13.0 in the wild) must not abort the whole blueprint.
            // It stays on the index, loudly.
            let (_sha, changed) =
                match crate::wheel_rewrite::rewrite_wheel_with(&base_src, &tmp, &mapper) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!(
                            wheel = %w.pypi_name,
                            error = %format!("{e:#}"),
                            "blueprint: wheel rewrite failed; leaving it on the index UNFIXED",
                        );
                        let _ = tokio::fs::remove_file(&tmp).await;
                        continue;
                    }
                };
            let final_name = if changed {
                // 999: robust against upstream build-tagged republishes
                // (uv compares build tags numerically first).
                Some(insert_build_tag(&std_name, "999retread")?)
            } else if emit_plan.ship.contains(&w.pypi_name) {
                Some(std_name)
            } else {
                None
            };
            match final_name {
                Some(name) => {
                    tokio::fs::rename(&tmp, wheels_dir.join(&name))
                        .await
                        .with_context(|| format!("renaming {name}"))?;
                    shipped += 1;
                }
                None => {
                    let _ = tokio::fs::remove_file(&tmp).await;
                }
            }
        }
        let (meta_name, meta_bytes) = build_meta_wheel(bundle_name, bundle_version, &entries);
        tokio::fs::write(wheels_dir.join(&meta_name), meta_bytes)
            .await
            .with_context(|| format!("writing {meta_name}"))?;
        shipped += 1;
        let prerelease_pins: BTreeMap<String, String> = emit_plan
            .overrides
            .iter()
            .filter(|(_, v)| {
                v.strip_prefix("==")
                    .and_then(|ver| Version::from_str(ver).ok())
                    .is_some_and(|ver| ver.any_prerelease())
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        render_snippet_blueprint(
            bundle_name,
            &pack_rel,
            bundle_version,
            libc_floor.as_deref(),
            &prerelease_pins,
            env!("CARGO_PKG_VERSION"),
        )
    } else {
        for w in wheels {
            if !emit_plan.ship.contains(&w.pypi_name) {
                continue;
            }
            let Some(src) = w.local_path.as_ref() else {
                continue;
            };
            let cached_name = src
                .file_name()
                .and_then(|n| n.to_str())
                .context("wheel path has no filename")?;
            let dst = wheels_dir.join(standard_wheel_filename(cached_name));
            tokio::fs::copy(src, &dst)
                .await
                .with_context(|| format!("copying {} -> {}", src.display(), dst.display()))?;
            shipped += 1;
        }
        render_snippet(
            bundle_name,
            &pack_rel,
            &entries,
            &emit_plan.overrides,
            libc_floor.as_deref(),
            env!("CARGO_PKG_VERSION"),
        )
    };
    let snippet_path = root.join("pixi-snippet.toml");
    tokio::fs::write(&snippet_path, &snippet)
        .await
        .with_context(|| format!("writing {}", snippet_path.display()))?;

    // v1.6.1: own the workspace manifest block. No pasting -- the
    // feature definition lives in a fenced region retread keeps in
    // sync; the user references `<bundle>-pypi` from an environment
    // once.
    if let Some(workspace) = workspace_dir {
        let manifest_path = workspace.join("pixi.toml");
        match tokio::fs::read_to_string(&manifest_path).await {
            Ok(text) => match sync_workspace_block(&text, bundle_name, &snippet) {
                Ok(Some(updated)) => {
                    let tmp = workspace.join("pixi.toml.retread-tmp");
                    tokio::fs::write(&tmp, updated)
                        .await
                        .with_context(|| format!("writing {}", tmp.display()))?;
                    tokio::fs::rename(&tmp, &manifest_path)
                        .await
                        .with_context(|| format!("renaming over {}", manifest_path.display()))?;
                    tracing::info!(
                        manifest = %manifest_path.display(),
                        feature = %format!("{bundle_name}-pypi"),
                        "emit-pypi: synced workspace manifest block (pixi reads the \
                         manifest once per run, so this lands on the NEXT pixi \
                         invocation; reference the feature from an environment to \
                         use it)",
                    );
                }
                Ok(None) => {}
                Err(e) => tracing::warn!(
                    manifest = %manifest_path.display(), error = %e,
                    "emit-pypi: workspace manifest sync skipped",
                ),
            },
            Err(e) => tracing::warn!(
                manifest = %manifest_path.display(), error = %e,
                "emit-pypi: workspace manifest unreadable; sync skipped",
            ),
        }
    }
    tracing::info!(
        bundle = %bundle_name,
        wheels_shipped = shipped,
        overrides = emit_plan.overrides.len(),
        snippet = %snippet_path.display(),
        "emit-pypi: side-channel written",
    );
    Ok(root)
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
            local_path: local.map(PathBuf::from),
            remote_url: None,
        }
    }

    #[test]
    fn line_map_applies_table_semantics() {
        let mut overrides = BTreeMap::new();
        overrides.insert("pillow".into(), ">=8".into());
        overrides.insert("isaacsim-core".into(), "==6.0.0.0".into());
        overrides.insert("rl-games".into(), "==1.6.1".into());
        overrides.insert("loose".into(), "*".into());
        let capable: std::collections::HashSet<String> =
            ["psutil".to_string(), "pillow".to_string()].into();
        let map = override_line_map(&overrides, &capable);
        // Floor override rewrites the pin.
        assert_eq!(map("pillow==12.1.1").as_deref(), Some("pillow>=8"));
        // Exact override equal to the existing pin: no change -> no
        // shadow wheel (family pins stay byte-identical).
        assert_eq!(map("isaacsim-core==6.0.0.0"), None);
        // URL requirement rerouted to the version pin.
        assert_eq!(
            map("rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11").as_deref(),
            Some("rl-games==1.6.1")
        );
        // "*" drops the specifier; extras and markers survive.
        assert_eq!(
            map("loose[fast]==1.0 ; sys_platform == \"linux\"").as_deref(),
            Some("loose[fast] ; sys_platform == 'linux'")
        );
        // Cap-only line on a conda-capable name: cap stripped (the v1.6
        // table was structurally blind to these).
        assert_eq!(map("psutil<6").as_deref(), Some("psutil"));
        // Cap-only on a non-capable name: left alone (cap is harmless,
        // conda can't pin above it).
        assert_eq!(map("notconda<2"), None);
        // python is never touched.
        assert_eq!(map("python>=3.10"), None);
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
    fn blueprint_snippet_is_static_and_tableless() {
        let s = render_snippet_blueprint(
            "isaac-pack",
            "isaac-pack",
            "6.0.0",
            Some("2.35"),
            &BTreeMap::new(),
            "1.7.0",
        );
        assert!(
            s.contains("find-links = [{ path = \"isaac-pack/retread-pypi/isaac-pack/wheels\" }]")
        );
        assert!(s.contains("isaac-pack-pypi = \"==6.0.0\""));
        assert!(s.contains("libc = \"2.35\""));
        assert!(!s.contains("dependency-overrides"), "no table, ever");
        assert!(
            s.lines()
                .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
                .count()
                <= 8
        );
    }

    #[test]
    fn workspace_block_sync_lifecycle() {
        let manifest = "[workspace]\nname = \"x\"\n";
        let block = "[feature.p-pypi.pypi-options]\nfind-links = []";
        // Insert: appended at EOF inside the fence.
        let v1 = sync_workspace_block(manifest, "p", block).unwrap().unwrap();
        assert!(v1.starts_with(manifest));
        assert!(v1.contains(">>> pixi-build-retread emit-pypi: p"));
        assert!(v1.contains(block));
        // Idempotent: same block -> no write.
        assert!(sync_workspace_block(&v1, "p", block).unwrap().is_none());
        // Replace: changed block swaps the interior, preserves the rest.
        let v2 = sync_workspace_block(&v1, "p", "[feature.p-pypi]\nnew = 1")
            .unwrap()
            .unwrap();
        assert!(v2.contains("new = 1"));
        assert!(!v2.contains("find-links"));
        assert!(v2.starts_with(manifest));
        assert_eq!(
            v2.matches(">>> pixi-build-retread emit-pypi: p").count(),
            1,
            "fences never accumulate"
        );
        // Corrupted fence: refuse to guess.
        let broken = v2.replace("# <<< pixi-build-retread emit-pypi: p <<<", "");
        assert!(sync_workspace_block(&broken, "p", block).is_err());
        // Different bundle gets its own independent fence.
        let v3 = sync_workspace_block(&v2, "q", block).unwrap().unwrap();
        assert!(v3.contains("emit-pypi: p"));
        assert!(v3.contains("emit-pypi: q"));
    }

    #[test]
    fn manylinux_floor_takes_highest_tag() {
        let mut w = wheel("isaacsim", "6.0.0.0", &[], None);
        w.wheel_filename = "isaacsim-6.0.0.0-cp312-none-manylinux_2_35_x86_64.whl".into();
        let mut older = wheel("torch", "2.10.0", &[], None);
        older.wheel_filename = "torch-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl".into();
        let pure = wheel("toml", "0.10.2", &[], None);
        assert_eq!(
            manylinux_floor(&[w, older, pure.clone()]).as_deref(),
            Some("2.35")
        );
        assert_eq!(manylinux_floor(&[pure]), None, "pure-python pack: no block");
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

    #[test]
    fn dependency_lines_per_entry_form() {
        let git_entry = WheelEntry {
            from: Some("isaaclab".into()),
            subdirectory: Some("source/isaaclab".into()),
            ..Default::default()
        };
        assert_eq!(
            render_dependency_line("isaaclab", &git_entry, Some("4.5.22")),
            "isaaclab = { version = \"==4.5.22\" }"
        );
        let index_entry = WheelEntry {
            version: Some("==6.0.0.0".into()),
            index: Some("https://pypi.nvidia.com".into()),
            extras: vec!["all".into(), "extscache".into()],
            ..Default::default()
        };
        assert_eq!(
            render_dependency_line("isaacsim", &index_entry, None),
            "isaacsim = { version = \"==6.0.0.0\", index = \"https://pypi.nvidia.com\", extras = [\"all\", \"extscache\"] }"
        );
    }

    #[test]
    fn snippet_renders_all_blocks() {
        let entries = vec![(
            "isaaclab".to_string(),
            WheelEntry {
                from: Some("isaaclab".into()),
                ..Default::default()
            },
            Some("4.5.22".to_string()),
        )];
        let mut overrides = BTreeMap::new();
        overrides.insert("pillow".to_string(), ">=12,<13".to_string());
        let s = render_snippet(
            "isaac-pack",
            "isaac-pack",
            &entries,
            &overrides,
            Some("2.35"),
            "1.6.0",
        );
        assert!(s.contains("[feature.isaac-pack-pypi.pypi-options]"));
        assert!(
            s.contains("find-links = [{ path = \"isaac-pack/retread-pypi/isaac-pack/wheels\" }]")
        );
        assert!(s.contains("[feature.isaac-pack-pypi.pypi-options.dependency-overrides]"));
        assert!(s.contains("\"pillow\" = \">=12,<13\""));
        assert!(s.contains("[feature.isaac-pack-pypi.pypi-dependencies]"));
        assert!(s.contains("isaaclab = { version = \"==4.5.22\" }"));
        assert!(s.contains("[feature.isaac-pack-pypi.system-requirements]"));
        assert!(s.contains("libc = \"2.35\""));
    }
}
