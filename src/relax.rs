//! PEP 508 -> conda match-spec translation with version-pin widening.

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use uv_pep508::uv_pep440::{self, Operator, Version};
use uv_pep508::{MarkerEnvironment, MarkerEnvironmentBuilder, Requirement, VersionOrUrl};

use crate::config::RelaxPolicy;

/// A conda package name exactly as declared.
///
/// Conda names legitimately contain underscores (for example,
/// `cuda-nvcc_linux-64`). This raw form is the only name domain from which a
/// conda solver match spec can be constructed. Identity comparisons use
/// [`CondaName::key`] instead.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CondaName(String);

impl CondaName {
    /// Preserve a conda package name exactly as declared.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Return the raw conda name that is valid at the solver boundary.
    pub fn as_spec(&self) -> &str {
        &self.0
    }

    /// Return the canonical identity key used for comparisons and dedupe.
    pub fn key(&self) -> PypiKey {
        PypiKey::from_pypi(&self.0)
    }

    /// Build a conda match spec while preserving this raw conda name.
    ///
    /// This is intentionally the only constructor for [`CondaMatchSpec`]. A
    /// [`PypiKey`] therefore cannot be passed to the conda solver by mistake.
    pub fn match_spec(&self, spec: &str) -> CondaMatchSpec {
        let spec = spec.trim();
        if spec.is_empty() || spec == "*" {
            CondaMatchSpec(self.0.clone())
        } else {
            CondaMatchSpec(format!("{} {spec}", self.0))
        }
    }
}

impl fmt::Display for CondaName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A PEP 503-normalized PyPI identity key.
///
/// This type is for comparisons, ownership sets, dedupe, and map keys only.
/// It deliberately has no conversion into [`CondaName`] or
/// [`CondaMatchSpec`].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PypiKey(String);

impl PypiKey {
    /// Canonicalize a PyPI spelling into its PEP 503 identity key.
    pub fn from_pypi(name: &str) -> Self {
        Self(canonical_conda_name(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for PypiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for PypiKey {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for PypiKey {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::from_pypi(&raw))
    }
}

/// A rendered conda solver match spec.
///
/// The private field and lack of a public constructor make arbitrary strings
/// ineligible for production conda solves. Construct one only through
/// [`CondaName::match_spec`].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CondaMatchSpec(String);

impl CondaMatchSpec {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CondaMatchSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The configured conda target for a canonical PyPI key.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CondaTarget {
    Mapped(CondaName),
    Disabled,
}

impl CondaTarget {
    pub fn mapped_name(&self) -> Option<&CondaName> {
        match self {
            Self::Mapped(name) => Some(name),
            Self::Disabled => None,
        }
    }
}

impl fmt::Display for CondaTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mapped(name) => name.fmt(f),
            Self::Disabled => Ok(()),
        }
    }
}

impl Serialize for CondaTarget {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for CondaTarget {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        if raw.is_empty() {
            Ok(Self::Disabled)
        } else {
            Ok(Self::Mapped(CondaName::new(raw)))
        }
    }
}

/// Canonical PyPI identity to raw conda target mapping.
pub type NameMap = BTreeMap<PypiKey, CondaTarget>;

/// Constraint origin retained across the PyPI-to-conda syntax boundary.
///
/// PyPI requirements carry both their source specifiers and the effective
/// PEP 440 form after the configured relaxation policy. Explicit overrides
/// are the sole boundary where native conda syntax may enter instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CondaConstraintOrigin {
    Pypi {
        original_specifiers: String,
        effective_specifiers: String,
    },
    ExplicitOverride,
}

/// A single translated dependency with both PyPI and conda identities.
///
/// `name` and `spec` are ready to drop into recipe.yaml's `run:` list. The
/// `Display` impl renders `"<name> <spec>"` when constrained and a bare name
/// otherwise, while `constraint_origin` preserves the PyPI-side constraint
/// needed by shared finalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CondaDep {
    pub pypi_name: PypiKey,
    pub name: String,
    pub spec: String,
    pub constraint_origin: CondaConstraintOrigin,
}

impl CondaDep {
    fn from_pypi(
        pypi_name: PypiKey,
        name: String,
        spec: String,
        original_specifiers: String,
        effective_specifiers: String,
    ) -> Self {
        Self {
            pypi_name,
            name,
            spec,
            constraint_origin: CondaConstraintOrigin::Pypi {
                original_specifiers,
                effective_specifiers,
            },
        }
    }

    fn from_explicit_override(pypi_name: PypiKey, name: String, spec: String) -> Self {
        Self {
            pypi_name,
            name,
            spec,
            constraint_origin: CondaConstraintOrigin::ExplicitOverride,
        }
    }
}

impl std::fmt::Display for CondaDep {
    /// Produce the joined conda match-spec string: `"<name> <spec>"` when
    /// spec is non-empty, bare `"<name>"` when spec is empty. Matches the
    /// old tuple-struct join: every call site that previously read `dep.0`
    /// can now use `dep.to_string()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.spec.is_empty() {
            write!(f, "{}", self.name)
        } else {
            write!(f, "{} {}", self.name, self.spec)
        }
    }
}

/// Split a conda dependency string `"<name> <spec>"` (or bare `"<name>"`)
/// into `(name, spec)`. The first whitespace-delimited token is the name;
/// everything after the first whitespace run (trimmed) is the spec. A bare
/// name with no whitespace returns `(name, "")`.
///
/// This is the canonical place for the first-token-is-name semantic. Use it
/// when you need BOTH name and spec from a dep string. Do NOT use it in
/// place of `workspace::split_conda_dep_line`, which has its own
/// build-string-aware contract (`splitn(3)`) and separate tests.
#[cfg(test)]
pub(crate) fn parse_named_spec(line: &str) -> (String, String) {
    let trimmed = line.trim();
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("").to_string();
    let spec = parts
        .next()
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    (name, spec)
}

/// Translate one PEP 508 requirement string into a conda match-spec.
///
/// Returns `Ok(None)` if the requirement is filtered out (marker evaluates to
/// false, or extras-only marker that doesn't match an active extra).
pub fn translate(
    raw: &str,
    env: &MarkerEnvironment,
    name_map: &NameMap,
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
    let pypi_key = PypiKey::from_pypi(pypi_name);
    let conda_name = map_name(pypi_name, name_map);

    // User override wins, full replacement.
    if let Some(spec) = overrides
        .get(pypi_name)
        .or_else(|| overrides.get(conda_name.as_spec()))
    {
        return Ok(Some(CondaDep::from_explicit_override(
            pypi_key,
            conda_name.to_string(),
            spec.clone(),
        )));
    }

    // `python` is fully off-limits to relax: every widened form
    // (`python >=3,<4` from major, `python >=3` from strong-major,
    // and the bare `python` that strong-major produces from a single
    // `==X.Y.Z`) is either meaningless or rejected by the conda solver
    // (`python 3` => "missing range specifier"). Pass python through
    // untouched under every policy.
    let effective_policy = if conda_name.as_spec() == "python" {
        RelaxPolicy::None
    } else {
        policy
    };

    let (spec, original_specifiers, effective_specifiers) = match &req.version_or_url {
        None => (String::new(), String::new(), String::new()),
        Some(VersionOrUrl::VersionSpecifier(specifiers)) => {
            let translated = convert_specifiers(specifiers, effective_policy)?;
            (
                translated.conda,
                render_pep440(specifiers.iter()),
                translated.pep440,
            )
        }
        Some(VersionOrUrl::Url(_)) => {
            // URL deps aren't expressible as conda match specs.
            tracing::warn!(req = %raw, "skipped: URL-based dependency");
            return Ok(None);
        }
    };

    Ok(Some(CondaDep::from_pypi(
        pypi_key,
        conda_name.to_string(),
        spec,
        original_specifiers,
        effective_specifiers,
    )))
}

/// v1.5.x cleanup 0c: THE canonical conda-name normalizer -- full
/// PEP 503: lowercase, runs of `_ . -` collapsed to a single `-`,
/// leading/trailing separators trimmed. This replaced the simpler
/// underscore-only normalizer (the former `conda_name_from` /
/// `conda_normalize` twins, briefly unified as `conda_name_simple`):
/// the two normalizations disagreed on dotted names (`ruamel.yaml`),
/// so a user's `retread-name-map` entry could be keyed under one form
/// and looked up under the other and silently never match. One fn,
/// one canonical form, bug class closed. `map_name` now constructs a
/// [`PypiKey`] before lookup, so every equivalent user spelling matches the
/// same canonical config key.
pub(crate) fn canonical_conda_name(name: &str) -> String {
    // Rattler virtual packages (`__cuda`, `__glibc`, ...) are
    // conda-side specials, never PyPI names -- PEP 503's
    // separator-trim would corrupt them (`__cuda` -> `cuda`). Pass
    // them through untouched (caught by the 0c anchor-surface test).
    if name.starts_with("__") {
        return name.to_ascii_lowercase();
    }
    let lower = name.to_ascii_lowercase();
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

/// P2 (bloat M2 / grizzly #2): dual-namespace set membership. True
/// when the canonical form of EITHER the conda-side name OR the
/// pypi-side name is in `set`. Sets must be seeded with
/// `canonical_conda_name` output; canonicalizing both query names here
/// means no call site can reintroduce the raw-vs-canonical skew that
/// shipped doomed conda deps alongside bundled wheels.
pub(crate) fn already_covered(
    set: &std::collections::HashSet<String>,
    conda_name: &str,
    pypi_name: Option<&str>,
) -> bool {
    set.contains(&canonical_conda_name(conda_name))
        || pypi_name.is_some_and(|p| set.contains(&canonical_conda_name(p)))
}

fn map_name(pypi: &str, overrides: &NameMap) -> CondaName {
    let key = PypiKey::from_pypi(pypi);
    if let Some(mapped) = overrides.get(&key).and_then(CondaTarget::mapped_name) {
        return mapped.clone();
    }
    CondaName::new(key.into_string())
}

/// v1.5.x cleanup 0c (grizzly finding #7): debug-time contract that a
/// `(name, spec)` pair survives the string pipeline -- it must parse
/// as a conda MatchSpec exactly as the downstream layers will parse
/// it. Compiled out of release builds (the e2e gates guard the release
/// path); in test/dev builds a layer emitting a malformed pair fails
/// fast at the boundary that produced it instead of at a misleading
/// downstream leaf (the v0.37.1 build-string-leak class).
#[cfg(test)]
pub(crate) fn assert_spec_roundtrips(name: &str, spec: &str) {
    #[cfg(debug_assertions)]
    {
        // Validate the parts the way downstream actually parses them:
        // the spec portion as a VersionSpec (override-map values,
        // comma-joins -- this is where the v0.37.1 embedded-space
        // build-string leak broke), the name as a PackageName. NOTE:
        // MatchSpec::from_str on the joined string is too lenient for
        // this contract -- it happily reads an embedded space as a
        // build-string separator, which is exactly the corruption the
        // contract exists to catch.
        debug_assert!(
            spec.is_empty()
                || rattler_conda_types::VersionSpec::from_str(
                    spec,
                    rattler_conda_types::ParseStrictness::Lenient
                )
                .is_ok(),
            "spec pipeline contract violated: `{spec}` (for `{name}`) does not parse as a VersionSpec",
        );
        debug_assert!(
            rattler_conda_types::PackageName::try_from(name).is_ok(),
            "spec pipeline contract violated: `{name}` is not a valid conda package name",
        );
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (name, spec);
    }
}

struct ConvertedSpecifiers {
    conda: String,
    pep440: String,
}

fn render_pep440<'a>(
    specifiers: impl IntoIterator<Item = &'a uv_pep440::VersionSpecifier>,
) -> String {
    specifiers
        .into_iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn convert_specifiers(
    specifiers: &uv_pep440::VersionSpecifiers,
    policy: RelaxPolicy,
) -> Result<ConvertedSpecifiers> {
    // Apply policy while the constraint is still PEP 440. The resulting
    // representation is retained for shared finalization; conda lowering is
    // a separate rendering step and is never used to reconstruct it.
    let effective = relax_version_specifiers(specifiers, policy)?;

    let mut conda = Vec::with_capacity(effective.len());
    let mut pep440 = Vec::with_capacity(effective.len());
    for specifier in effective.iter() {
        pep440.push(specifier.to_string());
        if let Some(rendered) = convert_one(specifier) {
            conda.push(rendered);
        }
    }
    Ok(ConvertedSpecifiers {
        conda: conda.join(","),
        pep440: pep440.join(","),
    })
}

/// Apply one existing relax-policy tier while the requirement is still PEP
/// 440. Both final recipe translation and the inline cross-wheel conflict
/// recovery path use this function so exact-pin widening and cap stripping
/// cannot drift.
pub(crate) fn relax_version_specifiers(
    specifiers: &uv_pep440::VersionSpecifiers,
    policy: RelaxPolicy,
) -> Result<uv_pep440::VersionSpecifiers> {
    let specs: Vec<uv_pep440::VersionSpecifier> = specifiers.iter().cloned().collect();
    let effective: Vec<uv_pep440::VersionSpecifier> = if specs.len() == 1
        && *specs[0].operator() == Operator::Equal
        && policy != RelaxPolicy::None
        && let Some(widened) = widen_exact(specs[0].version(), policy)
    {
        uv_pep440::VersionSpecifiers::from_str(&widened)
            .map_err(|error| anyhow!("parsing generated relaxed constraint `{widened}`: {error}"))?
            .into_iter()
            .collect()
    } else if matches!(policy, RelaxPolicy::StrongMajor | RelaxPolicy::CondaAware) {
        let refs: Vec<_> = specs.iter().collect();
        strip_upper_bounds(&refs)
    } else {
        specs
    };

    // TODO(conda-aware): CondaAware currently behaves IDENTICALLY to
    // StrongMajor here -- both unconditionally strip every upper bound
    // from range specs. The real conda-aware design probes the
    // workspace's conda channels per-spec and only strips when zero
    // candidates satisfy the bound; see RelaxPolicy::CondaAware doc.
    // Until that probe lands, conda-aware silently degrades to
    // strong-major.
    Ok(effective.into_iter().collect())
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
/// of `~=X.Y[.Z...]`.
fn strip_upper_bounds(specs: &[&uv_pep440::VersionSpecifier]) -> Vec<uv_pep440::VersionSpecifier> {
    use std::str::FromStr;
    let mut kept: Vec<uv_pep440::VersionSpecifier> = Vec::with_capacity(specs.len());
    for spec in specs {
        match spec.operator() {
            Operator::LessThan | Operator::LessThanEqual => {
                // Drop entirely -- this is a pure upper bound.
            }
            Operator::TildeEqual => {
                // Keep the full written version as the lower bound; only the
                // compatible release's implicit cap is stripped.
                let r = spec.version().release();
                if r.is_empty() {
                    continue;
                }
                let lower = format!(">={}", spec.version());
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
            let lower = spec.version();
            let mut upper = release[..release.len() - 1].to_vec();
            let last = upper.len() - 1;
            upper[last] += 1;
            let upper_release = upper
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(".");
            let upper = if spec.version().epoch() == 0 {
                upper_release
            } else {
                format!("{}!{upper_release}", spec.version().epoch())
            };
            return Some(format!(">={lower},<{upper}"));
        }
        // `===` (arbitrary equality) has no conda rendering. It remains in
        // the preserved PEP 440 form so constraint-aware emission fails
        // closed; legacy Display-only callers retain the existing warning.
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

    // The `*WithLastResort` variants behave IDENTICALLY to their base at
    // translation time. The auto-bundle finalizer invokes this same primitive
    // at broader tiers only after a collected wheel-metadata intersection is
    // proven unsatisfiable.
    match policy {
        RelaxPolicy::None => Some(format!("=={v}")),
        // Tiered recovery starts at the narrowest (patch) widening.
        // Translate-time emission mirrors plain Patch; inline cross-wheel
        // finalization escalates only after the raw intersection is empty.
        RelaxPolicy::Patch
        | RelaxPolicy::PatchWithLastResort
        | RelaxPolicy::PatchThenMinorThenMajorThenLastResort => {
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
    if r.is_empty() {
        return None;
    }
    let lo = r
        .iter()
        .take(r.len())
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(".")
        .to_string();
    // Bump the last digit for the upper bound.
    let mut upper = r.to_vec();
    let last = upper.len() - 1;
    upper[last] += 1;
    let hi = upper
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(".");
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
pub fn marker_env_for(conda_subdir: &str, python_version: &str) -> Result<MarkerEnvironment> {
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
        if best.is_none_or(|b| (major, minor) > b) {
            best = Some((major, minor));
        }
    }
    let (major, minor) = best?;
    Some(if minor > 0 {
        format!("{major}.{minor}")
    } else {
        format!("{major}")
    })
}

/// The dotted `X.Y` python to stamp on the emitted conda output (its
/// `variant`, build string, and `python` dep) AND on the build recipe.
///
/// Prefer the primary wheel's cpXY tag when it is dotted (`cp311` -> `3.11`):
/// a compiled wheel pins a real ABI, so the conda package genuinely only
/// works on that minor. Fall back to the workspace build-variant python
/// whenever the wheel carries only a bare-major tag (`py3-none-*`, including
/// the platform-specific `py3-none-manylinux...` form whose `is_pure_python`
/// is false) or no parseable tag.
///
/// This function NEVER returns a bare-major string. A bare-major python is an
/// ABI-anchor corruption: it makes retread emit `variant {python: "3"}`,
/// build `py3_0`, and `python 3.*`. With that anchor the consumer's solver
/// floats `python_abi` (the source package's build-host env resolves
/// `python 3.*` unpinned and the `"3"` variant pins no cpXY), so every
/// transitive dep that requires a concrete `python_abi X.Y.* *_cpXY`
/// (gymnasium, ros-humble-*, ...) reports "no candidates were found" and the
/// whole environment goes unsat against a misleading leaf.
///
/// Single source of truth for both `handler::produce_output` (the
/// `conda/outputs` metadata the solver consumes) and `recipe::build_recipe`
/// (the recipe rattler-build builds), so the two paths can never drift.
pub fn emit_python_version(primary_wheel_filename: &str, workspace_python_version: &str) -> String {
    match python_version_from_wheel_tag(primary_wheel_filename) {
        Some(v) if v.contains('.') => v,
        _ => workspace_python_version.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // P4.4: parse_named_spec table test.
    // -----------------------------------------------------------------

    #[test]
    fn parse_named_spec_table() {
        // Typical conda dep with a version spec.
        assert_eq!(
            parse_named_spec("numpy >=1.26,<2"),
            ("numpy".to_string(), ">=1.26,<2".to_string()),
        );
        // Build-string-bearing spec: the spec portion is everything after
        // the first whitespace, including embedded spaces. This is
        // intentionally different from split_conda_dep_line (which
        // handles build strings via a separate 3-way split); here the
        // entire rest-of-line after the name is the spec.
        assert_eq!(
            parse_named_spec("python_abi 3.11.* *_cp311"),
            ("python_abi".to_string(), "3.11.* *_cp311".to_string()),
        );
        // Bare name (no spec).
        assert_eq!(
            parse_named_spec("pyglet"),
            ("pyglet".to_string(), "".to_string()),
        );
        // Multiple spaces between name and spec: spec is trimmed.
        assert_eq!(
            parse_named_spec("torch  >=2.7,<3"),
            ("torch".to_string(), ">=2.7,<3".to_string()),
        );
        // Empty string produces two empty strings.
        assert_eq!(parse_named_spec(""), ("".to_string(), "".to_string()),);
    }

    // -----------------------------------------------------------------
    // Cleanup 0c: the explicit PEP 503 semantic matrix. The 0a property
    // test (legacy twin equivalence) is superseded by these -- the
    // legacy underscore-only behavior is intentionally GONE for dotted
    // and doubled-separator names; that divergence was the latent
    // ruamel.yaml-class bug (a name-map entry keyed under one
    // normalization and looked up under the other never matched).
    // -----------------------------------------------------------------

    #[test]
    fn canonical_conda_name_pep503_matrix() {
        // Skip-set class: names normalize to one canonical form.
        assert_eq!(canonical_conda_name("ruamel.yaml"), "ruamel-yaml");
        assert_eq!(canonical_conda_name("a__b"), "a-b");
        assert_eq!(canonical_conda_name("a.-_b"), "a-b");
        assert_eq!(
            canonical_conda_name("_leading.trailing_"),
            "leading-trailing"
        );
        // Unchanged for the simple cases (the 99% path).
        assert_eq!(canonical_conda_name("numpy"), "numpy");
        assert_eq!(canonical_conda_name("opencv_python"), "opencv-python");
        assert_eq!(canonical_conda_name("Pillow"), "pillow");
        assert_eq!(
            canonical_conda_name("nvidia-cuda-nvrtc-cu12"),
            "nvidia-cuda-nvrtc-cu12"
        );
    }

    #[test]
    fn canonical_conda_name_abi_anchor_surface_unchanged_by_0c() {
        // ABI-compare class (invariant #8): anchor checks happen on
        // RAW conda-side names (depends arrays, emissions) that never
        // pass through the pypi-name normalizer -- pinned here first.
        // (v4.2.0: the in-backend classifier is gone; `retread solve`
        // owns the anchor list -- see src/solve/repair.rs.)
        use crate::solve::is_abi_anchor;
        let anchors = [
            "python",
            "python_abi",
            "cuda-version",
            "libstdcxx-ng",
            "__cuda",
        ];
        for anchor in anchors {
            assert!(is_abi_anchor(anchor), "raw anchor {anchor:?} must match");
        }
        // And for the paths that DO normalize pypi names: 0c changed
        // nothing for these names relative to the pre-0c simple
        // normalizer (underscore-to-dash happened under BOTH; no dots
        // or doubled separators are involved). Pins that 0c introduced
        // no NEW divergence on the anchor surface.
        let pre_0c = |s: &str| s.to_ascii_lowercase().replace('_', "-");
        for anchor in ["python", "python_abi", "cuda-version", "libstdcxx-ng"] {
            assert_eq!(
                canonical_conda_name(anchor),
                pre_0c(anchor),
                "0c must not change normalization of anchor {anchor:?}"
            );
        }
        // Virtual packages pass through untouched (the pre-0c simple
        // normalizer mangled them to "--cuda"; nothing ever fed them
        // through it, and the new fn makes the safe behavior explicit).
        assert_eq!(canonical_conda_name("__cuda"), "__cuda");
        assert_eq!(canonical_conda_name("__glibc"), "__glibc");
    }

    #[test]
    fn name_map_override_uses_canonical_pypi_key() {
        // Config load canonicalizes user spellings before translation, so all
        // equivalent PyPI spellings hit the same typed key.
        let mut overrides = BTreeMap::new();
        overrides.insert(
            PypiKey::from_pypi("ruamel.yaml"),
            CondaTarget::Mapped(CondaName::new("custom_target")),
        );
        assert_eq!(
            map_name("ruamel_yaml", &overrides).as_spec(),
            "custom_target"
        );
        // Without an override, the canonical form is produced.
        assert_eq!(
            map_name("ruamel.yaml", &BTreeMap::new()).as_spec(),
            "ruamel-yaml"
        );

        overrides.insert(PypiKey::from_pypi("wheel_only"), CondaTarget::Disabled);
        assert_eq!(
            map_name("wheel.only", &overrides).as_spec(),
            "wheel-only",
            "a disabled mapping keeps the key occupied but translation uses identity"
        );
    }

    #[test]
    fn underscored_conda_name_solves_raw() {
        let name = CondaName::new("cuda-nvcc_linux-64");

        assert_eq!(name.as_spec(), "cuda-nvcc_linux-64");
        assert_eq!(name.key().as_str(), "cuda-nvcc-linux-64");
        assert_eq!(
            name.match_spec(">=12.9,<13").as_str(),
            "cuda-nvcc_linux-64 >=12.9,<13"
        );
        assert_eq!(
            name.match_spec("*").as_str(),
            "cuda-nvcc_linux-64",
            "an unconstrained match spec must still retain the raw conda name"
        );
    }

    #[test]
    fn name_map_round_trip_through_translate() {
        // Round-trip through the real consumer: a dotted-name dep line
        // translates to its canonical conda name (pre-0c the lookup
        // and the emission disagreed on this class).
        let dep = t("ruamel.yaml==0.18.6", RelaxPolicy::Minor).unwrap();
        assert!(
            dep.starts_with("ruamel-yaml "),
            "dotted pypi name must emit canonical conda name, got: {dep}"
        );
    }

    #[test]
    fn spec_roundtrip_contract() {
        // Good pairs pass silently.
        assert_spec_roundtrips("numpy", ">=1.26,<2");
        assert_spec_roundtrips("pytorch", "");
    }

    #[test]
    #[should_panic(expected = "spec pipeline contract violated")]
    #[cfg(debug_assertions)]
    fn spec_roundtrip_contract_trips_on_malformed() {
        // The v0.37.1 class: a build string leaking into a comma-joined
        // version spec produces an embedded space that MatchSpec
        // rejects. The contract must fail fast at the boundary.
        assert_spec_roundtrips("pytorch", ">=1.4,2.10.0 cuda*_mkl*303,>=2.10.0");
    }

    fn env() -> MarkerEnvironment {
        default_marker_env("3.11").unwrap()
    }

    // -----------------------------------------------------------------
    // P4.5: CondaDep Display round-trip
    // -----------------------------------------------------------------

    #[test]
    fn conda_dep_display_roundtrip() {
        // Spec-bearing: `to_string()` must produce the joined form.
        let dep = CondaDep::from_pypi(
            PypiKey::from_pypi("numpy"),
            "numpy".to_string(),
            ">=1.26,<2".to_string(),
            ">=1.26,<2".to_string(),
            ">=1.26,<2".to_string(),
        );
        assert_eq!(dep.to_string(), "numpy >=1.26,<2");
        assert_eq!(dep.name, "numpy");
        assert_eq!(dep.spec, ">=1.26,<2");

        // Empty spec -> bare name, NO trailing space.
        let bare = CondaDep::from_pypi(
            PypiKey::from_pypi("pyglet"),
            "pyglet".to_string(),
            String::new(),
            String::new(),
            String::new(),
        );
        assert_eq!(bare.to_string(), "pyglet");
        assert!(
            !bare.to_string().ends_with(' '),
            "must not have trailing space"
        );

        // Translate round-trip: the joined form from Display must match what
        // the old CondaDep(String) tuple-struct produced.
        let from_translate = translate(
            "numpy==1.26.4",
            &env(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            crate::config::RelaxPolicy::Minor,
        )
        .unwrap()
        .unwrap();
        assert_eq!(from_translate.to_string(), "numpy >=1.26,<2");
        assert_eq!(from_translate.name, "numpy");
        assert_eq!(from_translate.pypi_name, PypiKey::from_pypi("numpy"));
        assert_eq!(
            from_translate.constraint_origin,
            CondaConstraintOrigin::Pypi {
                original_specifiers: "==1.26.4".to_string(),
                effective_specifiers: ">=1.26,<2".to_string(),
            }
        );
    }

    fn translated(req: &str, policy: RelaxPolicy) -> CondaDep {
        translate(req, &env(), &BTreeMap::new(), &BTreeMap::new(), policy)
            .unwrap()
            .unwrap()
    }

    fn t(req: &str, policy: RelaxPolicy) -> Option<String> {
        translate(req, &env(), &BTreeMap::new(), &BTreeMap::new(), policy)
            .unwrap()
            .map(|d| d.to_string())
    }

    #[test]
    fn exact_pin_minor_relax() {
        assert_eq!(
            t("numpy==1.26.4", RelaxPolicy::Minor).as_deref(),
            Some("numpy >=1.26,<2")
        );
    }

    #[test]
    fn pypi_wildcard_exclusion_retains_pep440_before_conda_lowering() {
        let dep = translated("jupyter-core!=5.0.*,>=4.12", RelaxPolicy::Minor);
        assert_eq!(dep.spec, ">=4.12,<5.0|>=5.1");
        let CondaConstraintOrigin::Pypi {
            original_specifiers,
            effective_specifiers,
        } = dep.constraint_origin
        else {
            panic!("ordinary PyPI metadata must retain PyPI constraint origin");
        };
        for preserved in [original_specifiers, effective_specifiers] {
            assert!(preserved.contains(">=4.12"), "{preserved}");
            assert!(preserved.contains("!=5.0.*"), "{preserved}");
            uv_pep440::VersionSpecifiers::from_str(&preserved)
                .expect("preserved constraint must remain valid PEP 440");
        }
    }

    #[test]
    fn one_component_wildcards_remain_represented_in_both_domains() {
        let excluded = translated("demo!=1.*", RelaxPolicy::None);
        assert_eq!(excluded.spec, "<1|>=2");
        assert_eq!(
            excluded.constraint_origin,
            CondaConstraintOrigin::Pypi {
                original_specifiers: "!=1.*".to_string(),
                effective_specifiers: "!=1.*".to_string(),
            }
        );

        let included = translated("demo==1.*", RelaxPolicy::None);
        assert_eq!(included.spec, ">=1,<2");
        assert_eq!(
            included.constraint_origin,
            CondaConstraintOrigin::Pypi {
                original_specifiers: "==1.*".to_string(),
                effective_specifiers: "==1.*".to_string(),
            }
        );
    }

    #[test]
    fn compatible_release_lowering_preserves_full_pep440_version() {
        let two_part = translated("demo~=2.1", RelaxPolicy::None);
        assert_eq!(two_part.spec, ">=2.1,<3");

        let three_part = translated("demo~=2.1.4", RelaxPolicy::None);
        assert_eq!(three_part.spec, ">=2.1.4,<2.2");
        assert_eq!(
            three_part.constraint_origin,
            CondaConstraintOrigin::Pypi {
                original_specifiers: "~=2.1.4".to_string(),
                effective_specifiers: "~=2.1.4".to_string(),
            }
        );

        let prerelease = translated("demo~=1.4.5a1", RelaxPolicy::None);
        assert_eq!(prerelease.spec, ">=1.4.5a1,<1.5");

        let epoch = translated("demo~=1!2.1.4", RelaxPolicy::None);
        assert_eq!(epoch.spec, ">=1!2.1.4,<1!2.2");

        let strong = translated("demo~=2.1.4", RelaxPolicy::StrongMajor);
        assert_eq!(strong.spec, ">=2.1.4");
        assert_eq!(
            strong.constraint_origin,
            CondaConstraintOrigin::Pypi {
                original_specifiers: "~=2.1.4".to_string(),
                effective_specifiers: ">=2.1.4".to_string(),
            }
        );
    }

    #[test]
    fn effective_pep440_tracks_strong_major_and_unsupported_exact_equal() {
        let strong = translated("numpy>=1.26,<2", RelaxPolicy::StrongMajor);
        assert_eq!(strong.spec, ">=1.26");
        assert_eq!(
            strong.constraint_origin,
            CondaConstraintOrigin::Pypi {
                original_specifiers: ">=1.26,<2".to_string(),
                effective_specifiers: ">=1.26".to_string(),
            }
        );

        let arbitrary = translated("tensordict===0.9.0", RelaxPolicy::None);
        assert!(arbitrary.spec.is_empty());
        assert_eq!(
            arbitrary.constraint_origin,
            CondaConstraintOrigin::Pypi {
                original_specifiers: "===0.9.0".to_string(),
                effective_specifiers: "===0.9.0".to_string(),
            },
            "conda-inexpressible PEP 440 must remain visible to the fail-closed emission guard"
        );
    }

    #[test]
    fn exact_pin_patch_relax() {
        assert_eq!(
            t("pillow==12.0.0", RelaxPolicy::Patch).as_deref(),
            Some("pillow >=12.0.0,<12.1")
        );
    }

    #[test]
    fn exact_pin_no_relax() {
        assert_eq!(
            t("torch==2.7.1", RelaxPolicy::None).as_deref(),
            Some("torch ==2.7.1")
        );
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
        assert_eq!(
            t("Typing_Extensions==4.12.2", RelaxPolicy::Minor).as_deref(),
            Some("typing-extensions >=4.12,<5")
        );
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
        let dep = r.unwrap();
        assert_eq!(dep.to_string(), "torch >=2.7");
        assert_eq!(
            dep.constraint_origin,
            CondaConstraintOrigin::ExplicitOverride
        );
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
        assert_eq!(
            t("pyglet<2", RelaxPolicy::StrongMajor).as_deref(),
            Some("pyglet")
        );
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
        assert_eq!(
            t("pyglet<2", RelaxPolicy::CondaAware).as_deref(),
            Some("pyglet")
        );
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
    fn tiered_policy_translates_at_patch_widening() {
        // The policy mirrors `Patch` at translation time; collected
        // cross-wheel conflicts escalate through the same primitives in the
        // auto-bundle finalizer.
        assert_eq!(
            t(
                "numpy==1.26.4",
                RelaxPolicy::PatchThenMinorThenMajorThenLastResort
            )
            .as_deref(),
            Some("numpy >=1.26.4,<1.27"),
        );
        // Ranges pass through unchanged at translation time. The inline
        // finalizer strips their caps only when the collected intersection is
        // unsatisfiable through the Major tier.
        assert_eq!(
            t(
                "pyglet<2",
                RelaxPolicy::PatchThenMinorThenMajorThenLastResort
            )
            .as_deref(),
            Some("pyglet <2"),
        );
        // python is never widened under any policy.
        assert_eq!(
            t(
                "python==3.11.0",
                RelaxPolicy::PatchThenMinorThenMajorThenLastResort
            )
            .as_deref(),
            Some("python ==3.11.0"),
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

    #[test]
    fn emit_python_prefers_dotted_cp_tag() {
        // A compiled cp311 wheel pins a real ABI -> use it verbatim.
        assert_eq!(
            emit_python_version(
                "isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl",
                "3.11"
            ),
            "3.11"
        );
    }

    #[test]
    fn emit_python_rejects_bare_major_platform_wheel() {
        // THE BUG: a platform-specific wheel with a bare `py3` python tag
        // (is_pure_python == false, so produce_output took the wheel-tag
        // branch) parses to "3". emit_python_version must NOT return that --
        // it must fall back to the workspace's dotted python. A bare-major
        // emission corrupts the ABI anchor (`variant {python: "3"}`,
        // `python 3.*`) and floats python_abi in the consumer's solve.
        assert_eq!(
            emit_python_version("someext-1.0-py3-none-manylinux_2_35_x86_64.whl", "3.11"),
            "3.11"
        );
    }

    #[test]
    fn emit_python_rejects_bare_major_pure_wheel() {
        // py3-none-any (pure python) also has a bare tag -> workspace fallback.
        assert_eq!(
            emit_python_version("isaaclab-0.1.0-py3-none-any.whl", "3.11"),
            "3.11"
        );
    }

    #[test]
    fn emit_python_falls_back_when_tag_unparseable() {
        assert_eq!(emit_python_version("not-a-wheel-filename", "3.12"), "3.12");
    }
}
