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

    let spec = match &req.version_or_url {
        None => String::new(),
        Some(VersionOrUrl::VersionSpecifier(specifiers)) => {
            convert_specifiers(specifiers, policy)
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

    // Otherwise pass through, converting each specifier individually.
    specs
        .iter()
        .filter_map(|s| convert_one(s))
        .collect::<Vec<_>>()
        .join(",")
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

fn widen_exact(v: &Version, policy: RelaxPolicy) -> Option<String> {
    let r = v.release();
    if r.is_empty() {
        return None;
    }
    let major = r[0];
    let minor = r.get(1).copied().unwrap_or(0);
    let patch = r.get(2).copied().unwrap_or(0);

    match policy {
        RelaxPolicy::None => Some(format!("=={v}")),
        RelaxPolicy::Patch => Some(format!(">={major}.{minor}.{patch},<{major}.{}", minor + 1)),
        RelaxPolicy::Minor => Some(format!(">={major}.{minor},<{}", major + 1)),
        RelaxPolicy::Major => Some(format!(">={major}")),
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

/// Default marker env used when evaluating dependency markers. Targets
/// Python 3.11 on linux-x86_64. TODO: derive from host_platform parameter
/// in conda/outputs once we wire that through.
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
