//! PEP 508 -> conda match-spec translation with version-pin widening.

use std::collections::BTreeMap;
use std::str::FromStr;

use anyhow::{anyhow, Result};
use uv_pep508::uv_pep440::{self, Operator, Version};
use uv_pep508::{MarkerEnvironment, MarkerEnvironmentBuilder, Requirement, VersionOrUrl};

use crate::config::RelaxPolicy;

/// A single conda dependency line ready to drop into recipe.yaml `run:` list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CondaDep(pub String);

/// Translate one PEP 508 requirement string into a conda match-spec.
///
/// Returns `Ok(None)` if the requirement is filtered out (marker evaluates to
/// false, or extras-only marker that doesn't match an active extra).
pub fn translate(
    raw: &str,
    env: &MarkerEnvironment,
    name_map: &BTreeMap<String, String>,
    overrides: &BTreeMap<String, String>,
    policy: RelaxPolicy,
) -> Result<Option<CondaDep>> {
    let req: Requirement = Requirement::from_str(raw)
        .map_err(|e| anyhow!("failed to parse `{raw}` as PEP 508: {e}"))?;

    // Marker evaluation: skip if the marker is unsatisfied in our target env.
    // No active extras — we only repack base deps in this milestone.
    if !req.marker.evaluate(env, &[]) {
        tracing::debug!(req = %raw, "skipped: marker false");
        return Ok(None);
    }

    let pypi_name = req.name.as_ref();
    let conda_name = map_name(pypi_name, name_map);

    // User override wins, full replacement.
    if let Some(spec) = overrides.get(pypi_name).or_else(|| overrides.get(&conda_name)) {
        return Ok(Some(CondaDep(format_dep(&conda_name, spec))));
    }

    // `python` is fully off-limits to relax: every widened form
    // (`python >=3,<4` from major, `python >=3` from strong-major,
    // and the bare `python` that strong-major produces from a single
    // `==X.Y.Z`) is either meaningless or rejected by the conda solver
    // (`python 3` => "missing range specifier"). Pass python through
    // untouched under every policy.
    let effective_policy = if conda_name == "python" {
        RelaxPolicy::None
    } else {
        policy
    };

    let spec = match &req.version_or_url {
        None => String::new(),
        Some(VersionOrUrl::VersionSpecifier(specifiers)) => {
            convert_specifiers(specifiers, effective_policy)
        }
        Some(VersionOrUrl::Url(_)) => {
            // URL deps aren't expressible as conda match specs.
            tracing::warn!(req = %raw, "skipped: URL-based dependency");
            return Ok(None);
        }
    };

    Ok(Some(CondaDep(format_dep(&conda_name, &spec))))
}

fn format_dep(name: &str, spec: &str) -> String {
    if spec.is_empty() {
        name.to_string()
    } else {
        format!("{name} {spec}")
    }
}

fn map_name(pypi: &str, overrides: &BTreeMap<String, String>) -> String {
    if let Some(mapped) = overrides.get(pypi) {
        return mapped.clone();
    }
    // PEP 503 normalization: lowercase, runs of _ . - collapsed to single -.
    // Conda canonical form is also lowercase with single-dash separators.
    let lower = pypi.to_ascii_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut prev_dash = false;
    for c in lower.chars() {
        if c == '_' || c == '.' || c == '-' {
            if !prev_dash {
                out.push('-');
                prev_dash = true;
            }
        } else {
            out.push(c);
            prev_dash = false;
        }
    }
    out.trim_matches('-').to_string()
}

fn convert_specifiers(
    specifiers: &uv_pep440::VersionSpecifiers,
    policy: RelaxPolicy,
) -> String {
    // Detect single `==X.Y.Z` pin -> apply relax policy.
    let specs: Vec<_> = specifiers.iter().collect();
    if specs.len() == 1 && *specs[0].operator() == Operator::Equal && policy != RelaxPolicy::None {
        if let Some(widened) = widen_exact(specs[0].version(), policy) {
            return widened;
        }
    }

    // TODO(conda-aware): CondaAware currently behaves IDENTICALLY to
    // StrongMajor here -- both unconditionally strip every upper bound
    // from range specs. The real conda-aware design probes the
    // workspace's conda channels per-spec and only strips when zero
    // candidates satisfy the bound; see RelaxPolicy::CondaAware doc.
    // Until that probe lands, conda-aware silently degrades to
    // strong-major.
    if matches!(policy, RelaxPolicy::StrongMajor | RelaxPolicy::CondaAware) {
        return strip_upper_bounds(&specs)
            .iter()
            .filter_map(|s| convert_one(s))
            .collect::<Vec<_>>()
            .join(",");
    }

    // Otherwise pass through, converting each specifier individually.
    specs
        .iter()
        .filter_map(|s| convert_one(s))
        .collect::<Vec<_>>()
        .join(",")
}

/// Drop specifiers that impose an upper bound, expand `~=` to its
/// lower-bound-only form. Used by StrongMajor / CondaAware to keep
/// upstream caps from blocking the conda solve.
///
/// Kept inputs: `>X`, `>=X`, `!=X` (point exclusion isn't an upper),
/// `==X.*`/`!=X.*` (these are conceptually upper-bounded but rare;
/// passed through to convert_one which expands them as ranges and
/// would still satisfy any candidate).
///
/// Stripped: `<X`, `<=X`, the `<` half of `>=X,<Y`, the implicit upper
/// of `~=X.Y`.
fn strip_upper_bounds(
    specs: &[&uv_pep440::VersionSpecifier],
) -> Vec<uv_pep440::VersionSpecifier> {
    use std::str::FromStr;
    let mut kept: Vec<uv_pep440::VersionSpecifier> = Vec::with_capacity(specs.len());
    for spec in specs {
        match spec.operator() {
            Operator::LessThan | Operator::LessThanEqual => {
                // Drop entirely -- this is a pure upper bound.
            }
            Operator::TildeEqual => {
                // `~=X.Y` means `>=X.Y,<X.(Y+1)`. Keep the lower half.
                let r = spec.version().release();
                if r.is_empty() {
                    continue;
                }
                let major = r[0];
                let minor = r.get(1).copied().unwrap_or(0);
                let lower = format!(">={major}.{minor}");
                if let Ok(parsed) = uv_pep440::VersionSpecifier::from_str(&lower) {
                    kept.push(parsed);
                }
            }
            _ => kept.push((*spec).clone()),
        }
    }
    kept
}

fn convert_one(spec: &uv_pep440::VersionSpecifier) -> Option<String> {
    let op = match spec.operator() {
        Operator::Equal => "==",
        Operator::NotEqual => "!=",
        Operator::LessThan => "<",
        Operator::LessThanEqual => "<=",
        Operator::GreaterThan => ">",
        Operator::GreaterThanEqual => ">=",
        // `~=X.Y` is "compatible release". Conda has no direct equivalent;
        // we expand to `>=X.Y,<X+1`. Caller composes the comma list.
        Operator::TildeEqual => {
            let release = spec.version().release();
            if release.len() < 2 {
                return None;
            }
            let major = release[0];
            let minor = release[1];
            return Some(format!(">={major}.{minor},<{}", major + 1));
        }
        // `===` (arbitrary equality, PEP 440) — drop the spec; let any version match.
        Operator::ExactEqual => {
            tracing::warn!(spec = %spec, "dropping `===` specifier (no conda equivalent)");
            return None;
        }
        // Wildcards (== 1.2.* / != 1.2.*) — convert to range.
        Operator::EqualStar => {
            return widen_star(spec.version(), false);
        }
        Operator::NotEqualStar => {
            return widen_star(spec.version(), true);
        }
    };
    Some(format!("{op}{}", spec.version()))
}

pub fn widen_exact(v: &Version, policy: RelaxPolicy) -> Option<String> {
    let r = v.release();
    if r.is_empty() {
        return None;
    }
    let major = r[0];
    let minor = r.get(1).copied().unwrap_or(0);
    let patch = r.get(2).copied().unwrap_or(0);

    // The `*WithLastResort` variants behave IDENTICALLY to their base
    // here; the cascade is a separate post-translate probe pass in
    // handler.rs::last_resort_widen_pass that only widens further for
    // unsatisfiable specs.
    match policy {
        RelaxPolicy::None => Some(format!("=={v}")),
        RelaxPolicy::Patch | RelaxPolicy::PatchWithLastResort => {
            Some(format!(">={major}.{minor}.{patch},<{major}.{}", minor + 1))
        }
        RelaxPolicy::Minor | RelaxPolicy::MinorWithLastResort => {
            Some(format!(">={major}.{minor},<{}", major + 1))
        }
        // Major / StrongMajor all widen exact pins to bare-major; the
        // difference is range handling (Major: passthrough;
        // StrongMajor: strip uppers). TODO(conda-aware): CondaAware is
        // grouped here as well but the probe-and-decide layer is not
        // implemented -- it currently degrades to StrongMajor's
        // unconditional upper-strip. See RelaxPolicy::CondaAware doc
        // for the intended design.
        RelaxPolicy::Major
        | RelaxPolicy::MajorWithLastResort
        | RelaxPolicy::StrongMajor
        | RelaxPolicy::CondaAware => Some(format!(">={major}")),
    }
}

fn widen_star(v: &Version, negate: bool) -> Option<String> {
    // `==1.2.*` -> `>=1.2,<1.3`
    let r = v.release();
    if r.len() < 2 {
        return None;
    }
    let lo = format!(
        "{}",
        r.iter().take(r.len()).map(|n| n.to_string()).collect::<Vec<_>>().join(".")
    );
    // Bump the last digit for the upper bound.
    let mut upper = r.to_vec();
    let last = upper.len() - 1;
    upper[last] += 1;
    let hi = upper.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(".");
    if negate {
        Some(format!("<{lo}|>={hi}"))
    } else {
        Some(format!(">={lo},<{hi}"))
    }
}

/// Build a marker env from a conda subdir + Python version. This is the
/// production path called from the JSON-RPC handler. `host_platform` comes
/// from `CondaOutputsParams::host_platform`; `python_version` from the
/// pixi workspace's variant configuration (typically `python = "3.11"`).
///
/// Falls back to sensible defaults for fields PEP 508 markers rarely touch
/// (platform_release, platform_version).
pub fn marker_env_for(
    conda_subdir: &str,
    python_version: &str,
) -> Result<MarkerEnvironment> {
    let plat = PlatformAttrs::for_subdir(conda_subdir);
    let normalized = if python_version.contains('.') {
        python_version.to_string()
    } else {
        format!("{python_version}.0")
    };
    let full = format!("{normalized}.0");
    let python_version_ref = normalized.as_str();

    MarkerEnvironment::try_from(MarkerEnvironmentBuilder {
        implementation_name: "cpython",
        implementation_version: &full,
        os_name: plat.os_name,
        platform_machine: plat.machine,
        platform_python_implementation: "CPython",
        platform_release: "",
        platform_system: plat.system,
        platform_version: "",
        python_full_version: &full,
        python_version: python_version_ref,
        sys_platform: plat.sys_platform,
    })
    .map_err(|e| anyhow!("building marker env: {e}"))
}

struct PlatformAttrs {
    os_name: &'static str,
    sys_platform: &'static str,
    machine: &'static str,
    system: &'static str,
}

impl PlatformAttrs {
    fn for_subdir(subdir: &str) -> Self {
        match subdir {
            "linux-64" => Self {
                os_name: "posix",
                sys_platform: "linux",
                machine: "x86_64",
                system: "Linux",
            },
            "linux-aarch64" => Self {
                os_name: "posix",
                sys_platform: "linux",
                machine: "aarch64",
                system: "Linux",
            },
            "osx-64" => Self {
                os_name: "posix",
                sys_platform: "darwin",
                machine: "x86_64",
                system: "Darwin",
            },
            "osx-arm64" => Self {
                os_name: "posix",
                sys_platform: "darwin",
                machine: "arm64",
                system: "Darwin",
            },
            "win-64" => Self {
                os_name: "nt",
                sys_platform: "win32",
                machine: "AMD64",
                system: "Windows",
            },
            // noarch and unknowns fall back to linux-x86_64; markers that
            // would care (sys_platform-specific deps) are rare in pure-python
            // wheels.
            _ => Self {
                os_name: "posix",
                sys_platform: "linux",
                machine: "x86_64",
                system: "Linux",
            },
        }
    }
}

/// Convenience wrapper that targets linux-x86_64 + the given Python version.
/// Used by code paths that don't have access to the conda subdir (tests,
/// recipe generator's fallback).
pub fn default_marker_env(python_version: &str) -> Result<MarkerEnvironment> {
    // Accept "3" (any minor) or "3.11". The PEP 508 `python_version` marker
    // must be MAJOR.MINOR; pad accordingly. `python_full_version` adds .0.
    let normalized = if python_version.contains('.') {
        python_version.to_string()
    } else {
        format!("{python_version}.0")
    };
    let full = format!("{normalized}.0");
    let python_version = normalized.as_str();
    MarkerEnvironment::try_from(MarkerEnvironmentBuilder {
        implementation_name: "cpython",
        implementation_version: &full,
        os_name: "posix",
        platform_machine: "x86_64",
        platform_python_implementation: "CPython",
        platform_release: "",
        platform_system: "Linux",
        platform_version: "",
        python_full_version: &full,
        python_version,
        sys_platform: "linux",
    })
    .map_err(|e| anyhow!("building marker env: {e}"))
}

/// Parse the Python tag from a wheel filename. `cp311` -> Some("3.11"),
/// `py3` -> Some("3"), `none-any` wheels can be `py2.py3` -> Some("3").
pub fn python_version_from_wheel_tag(filename: &str) -> Option<String> {
    // Filename: {name}-{version}(-{build})?-{python}-{abi}-{platform}.whl
    let stem = filename.strip_suffix(".whl")?;
    let parts: Vec<&str> = stem.rsplitn(4, '-').collect(); // [platform, abi, python, rest]
    let python_tag = parts.get(2)?;

    // Take the highest cpXX or pyX in dotted-tag list.
    let mut best: Option<(u32, u32)> = None;
    for tag in python_tag.split('.') {
        let digits: String = tag.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            continue;
        }
        let (major, minor) = if digits.len() >= 2 && tag.starts_with("cp") {
            // cp311 -> 3.11
            let major: u32 = digits[..1].parse().ok()?;
            let minor: u32 = digits[1..].parse().ok()?;
            (major, minor)
        } else if tag.starts_with("py") {
            // py3, py311
            if digits.len() == 1 {
                (digits.parse().ok()?, 0)
            } else {
                (digits[..1].parse().ok()?, digits[1..].parse().ok()?)
            }
        } else {
            continue;
        };
        if best.map_or(true, |b| (major, minor) > b) {
            best = Some((major, minor));
        }
    }
    let (major, minor) = best?;
    Some(if minor > 0 { format!("{major}.{minor}") } else { format!("{major}") })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> MarkerEnvironment {
        default_marker_env("3.11").unwrap()
    }

    fn t(req: &str, policy: RelaxPolicy) -> Option<String> {
        translate(req, &env(), &BTreeMap::new(), &BTreeMap::new(), policy)
            .unwrap()
            .map(|d| d.0)
    }

    #[test]
    fn exact_pin_minor_relax() {
        assert_eq!(t("numpy==1.26.4", RelaxPolicy::Minor).as_deref(), Some("numpy >=1.26,<2"));
    }

    #[test]
    fn exact_pin_patch_relax() {
        assert_eq!(t("pillow==12.0.0", RelaxPolicy::Patch).as_deref(), Some("pillow >=12.0.0,<12.1"));
    }

    #[test]
    fn exact_pin_no_relax() {
        assert_eq!(t("torch==2.7.1", RelaxPolicy::None).as_deref(), Some("torch ==2.7.1"));
    }

    #[test]
    fn range_pass_through() {
        assert_eq!(
            t("requests>=2.0,<3", RelaxPolicy::Minor).as_deref(),
            Some("requests >=2.0,<3"),
        );
    }

    #[test]
    fn bare_name_stays_bare() {
        assert_eq!(t("six", RelaxPolicy::Minor).as_deref(), Some("six"));
    }

    #[test]
    fn name_normalization() {
        assert_eq!(t("Typing_Extensions==4.12.2", RelaxPolicy::Minor).as_deref(), Some("typing-extensions >=4.12,<5"));
    }

    #[test]
    fn override_replaces() {
        let mut overrides = BTreeMap::new();
        overrides.insert("torch".to_string(), ">=2.7".to_string());
        let r = translate(
            "torch==2.7.1",
            &env(),
            &BTreeMap::new(),
            &overrides,
            RelaxPolicy::Minor,
        )
        .unwrap();
        assert_eq!(r.unwrap().0, "torch >=2.7");
    }

    #[test]
    fn marker_filters() {
        assert!(t(r#"pywin32; sys_platform == "win32""#, RelaxPolicy::Minor).is_none());
    }

    #[test]
    fn strong_major_strips_upper_bounds() {
        // pyglet<2 -> conda spec with no upper bound. This is the
        // exact case that prompted strong-major: a transitive
        // Requires-Dist cap that conda-forge can't satisfy under
        // python 3.11 (only old pyglet 1.x is available with that
        // constraint, and those need python 3.5).
        assert_eq!(t("pyglet<2", RelaxPolicy::StrongMajor).as_deref(), Some("pyglet"));
        // >=A,<B keeps the lower bound, drops the upper.
        assert_eq!(
            t("numpy>=1.26,<2", RelaxPolicy::StrongMajor).as_deref(),
            Some("numpy >=1.26"),
        );
        // ~=A.B becomes >=A.B (no upper).
        assert_eq!(
            t("requests~=2.0", RelaxPolicy::StrongMajor).as_deref(),
            Some("requests >=2.0"),
        );
        // Exact pins behave like Major.
        assert_eq!(
            t("numpy==1.26.4", RelaxPolicy::StrongMajor).as_deref(),
            Some("numpy >=1"),
        );
        // Pure lower bound passes through.
        assert_eq!(
            t("setuptools>=40.8.0", RelaxPolicy::StrongMajor).as_deref(),
            Some("setuptools >=40.8.0"),
        );
    }

    #[test]
    fn conda_aware_behaves_like_strong_major_at_translate_time() {
        // The probe step is in handler.rs / a future repodata
        // probe; the translate-time behavior is identical to
        // strong-major. This pins that contract so the future
        // probe layer can be a pure refinement on top.
        assert_eq!(t("pyglet<2", RelaxPolicy::CondaAware).as_deref(), Some("pyglet"));
        assert_eq!(
            t("numpy==1.26.4", RelaxPolicy::CondaAware).as_deref(),
            Some("numpy >=1"),
        );
    }

    #[test]
    fn major_still_passes_ranges_unchanged() {
        // Regression: don't accidentally regress Major's existing
        // semantics when adding StrongMajor.
        assert_eq!(
            t("pyglet<2", RelaxPolicy::Major).as_deref(),
            Some("pyglet <2"),
            "Major must NOT strip upper bounds -- only StrongMajor/CondaAware do",
        );
    }

    #[test]
    fn python_dep_is_never_relaxed() {
        // No relax policy may widen a python requirement. Major would
        // emit `python >=3,<4`; strong-major / conda-aware would strip
        // the upper to give `python >=3` (and `python` from a single
        // exact pin), all of which either lose ABI meaning or trip
        // rattler-build's "missing range specifier" error. Pass through
        // unchanged regardless of policy.
        for policy in [
            RelaxPolicy::Patch,
            RelaxPolicy::Minor,
            RelaxPolicy::Major,
            RelaxPolicy::StrongMajor,
            RelaxPolicy::CondaAware,
        ] {
            assert_eq!(
                t("python==3.11.0", policy).as_deref(),
                Some("python ==3.11.0"),
                "policy {policy:?} must not modify python",
            );
            assert_eq!(
                t("python>=3.9,<3.13", policy).as_deref(),
                Some("python >=3.9,<3.13"),
                "policy {policy:?} must not modify python range",
            );
        }
        // Sanity: non-python deps under Major still get bare-major widening.
        assert_eq!(
            t("numpy==1.26.4", RelaxPolicy::Major).as_deref(),
            Some("numpy >=1"),
        );
    }

    #[test]
    fn wheel_tag_cp311() {
        assert_eq!(
            python_version_from_wheel_tag("isaacsim-5.1.0-cp311-none-manylinux_2_35_x86_64.whl"),
            Some("3.11".to_string())
        );
    }

    #[test]
    fn wheel_tag_py3() {
        assert_eq!(
            python_version_from_wheel_tag("requests-2.32.5-py3-none-any.whl"),
            Some("3".to_string())
        );
    }
}
