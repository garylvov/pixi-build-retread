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
}
