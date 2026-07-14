//! Dependency-file parser: turns requirements.txt, PEP 621 pyproject.toml, or
//! conda environment YAML into typed dependency inputs for a retread closure.
//!
//! The parser (`parse_dep_source`) does no package resolution. Exact versions
//! from conda environment exports are deliberately translated into advisory
//! lower bounds rather than reproducing machine-specific solved pins.
//! It does inspect `[tool.uv.sources]` paths against the fetched file's origin
//! so nonportable local/editable dependencies never fall back to a same-named
//! registry package.
//!
//! The fetcher (`fetch_dep_source`) below is the layer that gets a
//! `deps_from`-style source spec (local path / raw URL / git@rev) down to
//! file text, so a caller can pipe the result straight into
//! `parse_dep_source`. It does no wiring into config/handler/uv_closure --
//! that's a separate piece.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Parsed inputs contributed by one or more `retread-deps-from` sources.
///
/// This is deliberately typed rather than a bare `Vec<String>`: only
/// `pypi_roots` may enter uv's PEP 508 root set. Other source formats can add
/// separate advisory metadata without ever being mistaken for a PyPI package.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedDepsFrom {
    pub pypi_roots: Vec<String>,
    pub advisory_conda_floors: Vec<AdvisoryCondaFloor>,
    pub notices: Vec<DepsFromNotice>,
}

impl ParsedDepsFrom {
    fn extend(&mut self, mut other: Self) {
        self.pypi_roots.append(&mut other.pypi_roots);
        self.advisory_conda_floors
            .append(&mut other.advisory_conda_floors);
        self.notices.append(&mut other.notices);
    }
}

/// A lower-bound hint extracted from a conda environment export.
///
/// These are not roots or conda run dependencies. The handler may translate a
/// floor into a uv constraint only through an explicit, unambiguous pack
/// name-map edge to an already-active PyPI root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdvisoryCondaFloor {
    /// Native conda package spelling; underscores are intentionally retained.
    pub conda_name: String,
    /// A positive lower bound such as `>=3.9.16` or `>1.0`.
    pub floor_spec: String,
    /// Dependency-file origin retained for diagnostics/provenance.
    pub source: String,
}

impl AdvisoryCondaFloor {
    pub fn as_conda_requirement(&self) -> String {
        format!("{} {}", self.conda_name, self.floor_spec)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepsFromNoticeKind {
    LocalUvSource,
    CondaEnvironmentEntry,
}

/// A visible, structured explanation for an input deliberately omitted from
/// the closure. The resolver logs every notice; parser tests can assert the
/// same report without installing a process-global tracing subscriber.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepsFromNotice {
    pub kind: DepsFromNoticeKind,
    pub dependency: String,
    pub configured_path: String,
    pub resolved_path: Option<PathBuf>,
    pub source: String,
    pub reason: String,
}

impl DepsFromNotice {
    fn log(&self) {
        let resolved_path = self
            .resolved_path
            .as_deref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<unavailable>".to_string());
        match self.kind {
            DepsFromNoticeKind::LocalUvSource => tracing::warn!(
                dependency = %self.dependency,
                configured_path = %self.configured_path,
                resolved_path = %resolved_path,
                source = %self.source,
                reason = %self.reason,
                "retread-deps-from: skipping dependency backed by a local uv source",
            ),
            DepsFromNoticeKind::CondaEnvironmentEntry => tracing::warn!(
                dependency = %self.dependency,
                entry = %self.configured_path,
                source = %self.source,
                reason = %self.reason,
                "retread-deps-from: skipping conda environment dependency entry",
            ),
        }
    }
}

/// Fetched source text plus the origin needed to interpret relative paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchedDepSource {
    pub content: String,
    pub filename_hint: String,
    /// Parent directory of a local file or a file in a git checkout. Raw URL
    /// sources have no consumer-local filesystem base.
    pub filesystem_base: Option<PathBuf>,
    pub display_origin: String,
}

/// Parse a fetched dependency source into typed closure inputs.
///
/// `filename_hint` (e.g. `"requirements_isaaclab.txt"` or `"pyproject.toml"`)
/// selects the format. If the hint is ambiguous, the content is sniffed:
/// content with a TOML section header is treated as pyproject, while a
/// top-level YAML `dependencies` list is treated as a conda environment;
/// otherwise it falls back to requirements-line parsing.
pub fn parse_dep_source(source: &FetchedDepSource) -> Result<ParsedDepsFrom> {
    match detect_format(&source.filename_hint, &source.content)? {
        DepSourceFormat::PyProject => parse_pyproject(source),
        DepSourceFormat::CondaEnvironment => parse_conda_environment(source),
        DepSourceFormat::Requirements => Ok(ParsedDepsFrom {
            pypi_roots: parse_requirements(&source.content),
            advisory_conda_floors: Vec::new(),
            notices: Vec::new(),
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DepSourceFormat {
    Requirements,
    PyProject,
    CondaEnvironment,
}

fn detect_format(filename_hint: &str, content: &str) -> Result<DepSourceFormat> {
    let lower = filename_hint.to_ascii_lowercase();
    if lower.ends_with(".toml") || lower == "pyproject" {
        return Ok(DepSourceFormat::PyProject);
    }
    if lower.ends_with(".txt") {
        return Ok(DepSourceFormat::Requirements);
    }
    if lower.ends_with(".yaml") || lower.ends_with(".yml") {
        validate_conda_environment_shape(content)?;
        return Ok(DepSourceFormat::CondaEnvironment);
    }
    // Ambiguous hint: prefer the semantic YAML shape before the broad TOML
    // section heuristic. A flow-style YAML dependencies list may itself begin
    // with `[` on a continuation line.
    if has_conda_environment_shape(content) {
        return Ok(DepSourceFormat::CondaEnvironment);
    }
    if content
        .lines()
        .map(str::trim)
        .any(|l| l.starts_with('[') && l.contains(']') && !l.starts_with("[["))
    {
        return Ok(DepSourceFormat::PyProject);
    }
    Ok(DepSourceFormat::Requirements)
}

fn yaml_mapping_value<'a>(
    mapping: &'a serde_yaml::Mapping,
    key: &str,
) -> Option<&'a serde_yaml::Value> {
    mapping.get(&serde_yaml::Value::String(key.to_string()))
}

fn has_conda_environment_shape(content: &str) -> bool {
    let Ok(serde_yaml::Value::Mapping(mapping)) =
        serde_yaml::from_str::<serde_yaml::Value>(content)
    else {
        return false;
    };
    yaml_mapping_value(&mapping, "dependencies").is_some_and(serde_yaml::Value::is_sequence)
}

fn validate_conda_environment_shape(content: &str) -> Result<()> {
    let value: serde_yaml::Value =
        serde_yaml::from_str(content).context("parsing conda environment YAML")?;
    let mapping = value
        .as_mapping()
        .ok_or_else(|| anyhow::anyhow!("conda environment YAML must be a top-level mapping"))?;
    let dependencies = yaml_mapping_value(mapping, "dependencies").ok_or_else(|| {
        anyhow::anyhow!("conda environment YAML must contain a top-level `dependencies` list")
    })?;
    if !dependencies.is_sequence() {
        bail!("conda environment YAML top-level `dependencies` must be a list");
    }
    Ok(())
}

/// Parse pip-style `requirements.txt` content.
fn parse_requirements(content: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with("-r")
            || line.starts_with("-c")
            || line.starts_with("-e")
            || line.starts_with("--")
            || line.starts_with('.')
        {
            continue;
        }

        // VCS / URL lines: pass through unchanged, no inline-comment
        // stripping (URLs can legitimately contain '#' fragments, and a
        // later stage decides what to do with these).
        if is_vcs_or_url_line(line) {
            out.push(line.to_string());
            continue;
        }

        // Strip a trailing inline comment (` #...`), but only if preceded
        // by whitespace so we don't clobber markers or extras containing
        // '#' (which don't occur in valid PEP 508 anyway).
        let stripped = strip_inline_comment(line);
        let stripped = stripped.trim();
        if stripped.is_empty() {
            continue;
        }
        out.push(stripped.to_string());
    }
    out
}

/// Parse a conda `environment.yaml` export. Pip subsection entries remain
/// PEP 508 roots; conda entries become non-installing advisory lower bounds.
fn parse_conda_environment(source: &FetchedDepSource) -> Result<ParsedDepsFrom> {
    use rattler_conda_types::{EnvironmentYaml, MatchSpecOrSubSection};

    validate_conda_environment_shape(&source.content)?;
    let environment = EnvironmentYaml::from_yaml_str(&source.content)
        .context("parsing conda environment dependency entries")?;
    let mut out = ParsedDepsFrom::default();
    for dependency in &environment.dependencies {
        match dependency {
            MatchSpecOrSubSection::MatchSpec(spec) => {
                let rendered = spec.to_string();
                match advisory_floor_from_match_spec(spec, &source.display_origin) {
                    Ok(Some(floor)) => out.advisory_conda_floors.push(floor),
                    Ok(None) => {
                        let dependency = spec
                            .name
                            .as_ref()
                            .and_then(|name| name.as_exact())
                            .map(|name| name.as_normalized().to_string())
                            .unwrap_or_else(|| "<unrepresentable>".to_string());
                        out.notices.push(conda_environment_notice(
                            dependency,
                            rendered,
                            &source.display_origin,
                            "entry has no safe positive PEP 440 lower bound",
                        ));
                    }
                    Err(reason) => {
                        let dependency = spec
                            .name
                            .as_ref()
                            .and_then(|name| name.as_exact())
                            .map(|name| name.as_normalized().to_string())
                            .unwrap_or_else(|| "<unrepresentable>".to_string());
                        out.notices.push(conda_environment_notice(
                            dependency,
                            rendered,
                            &source.display_origin,
                            &reason,
                        ));
                    }
                }
            }
            MatchSpecOrSubSection::SubSection(name, specs) if name == "pip" => {
                out.pypi_roots.extend(parse_requirements(&specs.join("\n")));
            }
            MatchSpecOrSubSection::SubSection(name, _) => {
                bail!(
                    "unsupported `{name}` subsection in conda environment dependencies; only `pip` is supported"
                );
            }
        }
    }
    Ok(out)
}

fn advisory_floor_from_match_spec(
    spec: &rattler_conda_types::MatchSpec,
    source: &str,
) -> std::result::Result<Option<AdvisoryCondaFloor>, String> {
    let Some(name) = spec.name.as_ref().and_then(|name| name.as_exact()) else {
        return Err("entry does not have an exact conda package name".to_string());
    };
    let conda_name = name.as_normalized().to_string();
    if conda_name.starts_with('_') && conda_name.ends_with("_mutex") {
        return Err("solver mutex package is environment-export noise".to_string());
    }

    let Some(bound) = spec.version.as_ref().and_then(conda_positive_lower_bound) else {
        return Ok(None);
    };
    let version = bound.version.to_string();
    let operator = if bound.exclusive { ">" } else { ">=" };
    let floor_spec = format!("{operator}{version}");

    // Conda accepts version spellings PEP 440 does not. Retain only floors uv
    // can actually consume; unsupported forms remain visible as notices.
    let pep_name = crate::relax::canonical_conda_name(&conda_name);
    let requirement = format!("{pep_name}{floor_spec}");
    if uv_pep508::Requirement::<uv_pep508::VerbatimUrl>::from_str(&requirement).is_err() {
        return Err(format!(
            "conda version `{version}` is not representable as a PEP 440 lower bound"
        ));
    }

    Ok(Some(AdvisoryCondaFloor {
        conda_name,
        floor_spec,
        source: source.to_string(),
    }))
}

#[derive(Debug, Clone)]
struct CondaLowerBound {
    version: rattler_conda_types::Version,
    exclusive: bool,
}

/// Extract a lower bound that is true for every version accepted by `spec`.
/// Conjunctions retain their strongest positive bound; disjunctions contribute
/// none because neither branch is guaranteed to hold.
fn conda_positive_lower_bound(spec: &rattler_conda_types::VersionSpec) -> Option<CondaLowerBound> {
    use rattler_conda_types::VersionSpec;
    use rattler_conda_types::version_spec::{
        EqualityOperator, LogicalOperator, RangeOperator, StrictRangeOperator,
    };

    match spec {
        VersionSpec::Exact(EqualityOperator::Equals, version) => Some(CondaLowerBound {
            version: version.clone(),
            exclusive: false,
        }),
        VersionSpec::StrictRange(
            StrictRangeOperator::StartsWith | StrictRangeOperator::Compatible,
            version,
        ) => Some(CondaLowerBound {
            version: version.0.clone(),
            exclusive: false,
        }),
        VersionSpec::Range(RangeOperator::Greater, version) => Some(CondaLowerBound {
            version: version.clone(),
            exclusive: true,
        }),
        VersionSpec::Range(RangeOperator::GreaterEquals, version) => Some(CondaLowerBound {
            version: version.clone(),
            exclusive: false,
        }),
        VersionSpec::Group(LogicalOperator::And, members) => members
            .iter()
            .filter_map(conda_positive_lower_bound)
            .max_by(|left, right| {
                left.version
                    .cmp(&right.version)
                    .then_with(|| left.exclusive.cmp(&right.exclusive))
            }),
        VersionSpec::Group(LogicalOperator::Or, _) => None,
        _ => None,
    }
}

fn conda_environment_notice(
    dependency: String,
    entry: String,
    source: &str,
    reason: &str,
) -> DepsFromNotice {
    DepsFromNotice {
        kind: DepsFromNoticeKind::CondaEnvironmentEntry,
        dependency,
        configured_path: entry,
        resolved_path: None,
        source: source.to_string(),
        reason: reason.to_string(),
    }
}

fn is_vcs_or_url_line(line: &str) -> bool {
    line.contains(" @ ")
        || line.starts_with("git+")
        || line.starts_with("http://")
        || line.starts_with("https://")
}

fn strip_inline_comment(line: &str) -> &str {
    // Look for " #" (whitespace followed by '#') to avoid false positives.
    if let Some(idx) = line.find(" #") {
        &line[..idx]
    } else {
        line
    }
}

/// Parse a PEP 621 `pyproject.toml`'s `[project] dependencies` array and
/// suppress dependencies whose uv source is a local path.
fn parse_pyproject(source: &FetchedDepSource) -> Result<ParsedDepsFrom> {
    let value: toml::Value = source.content.parse()?;

    if let Some(project) = value.get("project") {
        let deps = project
            .get("dependencies")
            .and_then(|d| d.as_array())
            .cloned()
            .unwrap_or_default();
        let uv_sources = collect_uv_sources(&value)?;
        let mut out = ParsedDepsFrom {
            pypi_roots: Vec::with_capacity(deps.len()),
            advisory_conda_floors: Vec::new(),
            notices: Vec::new(),
        };
        for dep in deps {
            match dep.as_str() {
                Some(requirement) => {
                    let parsed: Option<uv_pep508::Requirement> =
                        uv_pep508::Requirement::from_str(requirement).ok();
                    let normalized_name = parsed
                        .as_ref()
                        .map(|req| crate::relax::canonical_conda_name(req.name.as_ref()));
                    let Some(source_variants) = normalized_name
                        .as_ref()
                        .and_then(|name| uv_sources.get(name))
                    else {
                        out.pypi_roots.push(requirement.to_string());
                        continue;
                    };

                    let local_paths: Vec<&str> = source_variants
                        .iter()
                        .filter_map(|variant| match variant {
                            UvSourceVariant::LocalPath(path) => Some(path.as_str()),
                            UvSourceVariant::NonLocal => None,
                        })
                        .collect();
                    if local_paths.is_empty() {
                        out.pypi_roots.push(requirement.to_string());
                        continue;
                    }
                    let dependency = parsed
                        .as_ref()
                        .expect("normalized name requires a parsed requirement")
                        .name
                        .to_string();
                    for configured_path in local_paths {
                        out.notices.push(inspect_local_uv_source(
                            &dependency,
                            configured_path,
                            source,
                        )?);
                    }
                    // Local/editable source trees are intentionally omitted
                    // even when present. Retread does not transport arbitrary
                    // source directories, and falling back to a registry
                    // project with the same name is incorrect.
                }
                None => bail!("non-string entry in [project.dependencies]: {dep:?}"),
            }
        }
        return Ok(out);
    }

    if value
        .get("tool")
        .and_then(|t| t.get("poetry"))
        .and_then(|p| p.get("dependencies"))
        .is_some()
    {
        bail!("poetry format not supported; use requirements.txt or PEP621");
    }

    // No [project] and no poetry deps: nothing to report, but also nothing
    // that looks like a recognized format. Treat as empty (e.g. a bare
    // pyproject.toml with only [build-system]).
    Ok(ParsedDepsFrom::default())
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum UvSourceVariant {
    LocalPath(String),
    NonLocal,
}

/// Collect uv source declarations by normalized distribution name. Values are
/// appended rather than overwritten so aliases such as `foo-bar` and
/// `foo_bar` cannot hide a local source through last-wins normalization.
fn collect_uv_sources(value: &toml::Value) -> Result<BTreeMap<String, Vec<UvSourceVariant>>> {
    let Some(sources) = value
        .get("tool")
        .and_then(|tool| tool.get("uv"))
        .and_then(|uv| uv.get("sources"))
    else {
        return Ok(BTreeMap::new());
    };
    let table = sources
        .as_table()
        .ok_or_else(|| anyhow::anyhow!("[tool.uv.sources] must be a table"))?;
    let mut out: BTreeMap<String, Vec<UvSourceVariant>> = BTreeMap::new();
    for (name, declaration) in table {
        let normalized = crate::relax::canonical_conda_name(name);
        let variants = out.entry(normalized).or_default();
        match declaration {
            toml::Value::Table(source) => variants.push(parse_uv_source_table(name, source)?),
            toml::Value::Array(items) => {
                if items.is_empty() {
                    bail!("[tool.uv.sources].{name} source array must not be empty");
                }
                for item in items {
                    let table = item.as_table().ok_or_else(|| {
                        anyhow::anyhow!("[tool.uv.sources].{name} array entries must be tables")
                    })?;
                    variants.push(parse_uv_source_table(name, table)?);
                }
            }
            _ => bail!("[tool.uv.sources].{name} must be a table or array of tables"),
        }
    }
    Ok(out)
}

fn parse_uv_source_table(
    name: &str,
    source: &toml::map::Map<String, toml::Value>,
) -> Result<UvSourceVariant> {
    match source.get("path") {
        Some(path) => path
            .as_str()
            .map(|path| UvSourceVariant::LocalPath(path.to_string()))
            .ok_or_else(|| anyhow::anyhow!("[tool.uv.sources].{name}.path must be a string")),
        None => Ok(UvSourceVariant::NonLocal),
    }
}

fn inspect_local_uv_source(
    dependency: &str,
    configured_path: &str,
    source: &FetchedDepSource,
) -> Result<DepsFromNotice> {
    let configured = Path::new(configured_path);
    let resolved = if configured.is_absolute() {
        Some(configured.to_path_buf())
    } else {
        source
            .filesystem_base
            .as_ref()
            .map(|base| base.join(configured))
    };

    let reason = match resolved.as_deref() {
        Some(path) => match path.try_exists() {
            Ok(false) => "configured local path does not exist on this consumer".to_string(),
            Ok(true) => "configured local path exists, but local/editable source trees are not portable retread closure roots".to_string(),
            Err(err) => {
                return Err(err).with_context(|| {
                    format!(
                        "inspecting [tool.uv.sources] path `{configured_path}` for `{dependency}` from {}",
                        source.display_origin
                    )
                });
            }
        },
        None => "relative local path cannot be resolved from a raw URL dependency source"
            .to_string(),
    };

    Ok(DepsFromNotice {
        kind: DepsFromNoticeKind::LocalUvSource,
        dependency: dependency.to_string(),
        configured_path: configured_path.to_string(),
        resolved_path: resolved,
        source: source.display_origin.clone(),
        reason,
    })
}

/// A resolved dependency-source spec, i.e. "where does the requirements.txt
/// / pyproject.toml text come from."
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DepSource {
    /// A file living inside the workspace, given as a path relative to
    /// `workspace_root` (or absolute).
    Local(PathBuf),
    /// A raw HTTP(S) URL serving the file directly (e.g. a
    /// raw.githubusercontent.com blob link).
    Url(String),
    /// A path inside a git repo, pinned to a specific rev (commit SHA, tag,
    /// or branch name -- see the "Determinism" note below for why a
    /// resolved SHA is preferred).
    Git {
        git: String,
        rev: String,
        path: String,
    },
}

/// Deserialize form for one `retread-deps-from` list element / the bare
/// (non-list) value. A bare string is scheme-sniffed: `http://` / `https://`
/// -> [`DepSource::Url`], anything else -> [`DepSource::Local`]. A table
/// with `git`/`rev`/`path` keys -> [`DepSource::Git`].
#[derive(Deserialize)]
#[serde(untagged)]
enum DepSourceRepr {
    Str(String),
    Git {
        git: String,
        rev: String,
        path: String,
    },
}

impl<'de> Deserialize<'de> for DepSource {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match DepSourceRepr::deserialize(deserializer)? {
            DepSourceRepr::Str(s) => {
                if s.starts_with("http://") || s.starts_with("https://") {
                    DepSource::Url(s)
                } else {
                    DepSource::Local(PathBuf::from(s))
                }
            }
            DepSourceRepr::Git { git, rev, path } => DepSource::Git { git, rev, path },
        })
    }
}

impl Serialize for DepSource {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            DepSource::Local(path) => serializer.serialize_str(&path.to_string_lossy()),
            DepSource::Url(u) => serializer.serialize_str(u),
            DepSource::Git { git, rev, path } => {
                use serde::ser::SerializeStruct;
                let mut s = serializer.serialize_struct("DepSource", 3)?;
                s.serialize_field("git", git)?;
                s.serialize_field("rev", rev)?;
                s.serialize_field("path", path)?;
                s.end()
            }
        }
    }
}

/// Resolve a `DepSource` to file text and the origin information needed by
/// `parse_dep_source`.
///
/// - `DepSource::Local(path)`: `path` is resolved relative to
///   `workspace_root` (an absolute `path` is used as-is), then read.
/// - `DepSource::Git { git, rev, path }`: reuses
///   `source_build::ensure_git_checkout`, the same clone + per-(url, rev)
///   flock dance `build_wheel_from_git` uses, so concurrent resolvers don't
///   race on the same on-disk clone and repeated calls for the same (git,
///   rev) are a cheap no-op. `rev` should be a pinned commit SHA for
///   reproducibility (a moving branch/tag ref means the fetched content can
///   change between calls); this function does not itself resolve a moving
///   ref to a SHA.
/// - `DepSource::Url(u)`: plain HTTP GET via `reqwest` (same client
///   `wheel::fetch_wheel` / `pypi` use), cached under `cache_dir` keyed by a
///   hash of the URL so repeated calls don't re-fetch.
pub async fn fetch_dep_source(
    src: &DepSource,
    workspace_root: &Path,
    cache_dir: &Path,
) -> Result<FetchedDepSource> {
    match src {
        DepSource::Local(path) => {
            let resolved = if path.is_absolute() {
                path.clone()
            } else {
                workspace_root.join(path)
            };
            let content = tokio::fs::read_to_string(&resolved)
                .await
                .with_context(|| format!("reading local dep source {}", resolved.display()))?;
            let hint = filename_of(&resolved)?;
            Ok(FetchedDepSource {
                content,
                filename_hint: hint,
                filesystem_base: resolved.parent().map(Path::to_path_buf),
                display_origin: resolved.display().to_string(),
            })
        }
        DepSource::Git { git, rev, path } => {
            let clone_dir = crate::source_build::ensure_git_checkout(git, rev, cache_dir)
                .await
                .with_context(|| format!("cloning {git}@{rev} for dep source {path}"))?;
            let file_path = clone_dir.join(path);
            let content = tokio::fs::read_to_string(&file_path)
                .await
                .with_context(|| {
                    format!(
                        "reading {path} from git clone of {git}@{rev} (at {})",
                        file_path.display()
                    )
                })?;
            let hint = filename_of(Path::new(path))?;
            Ok(FetchedDepSource {
                content,
                filename_hint: hint,
                filesystem_base: file_path.parent().map(Path::to_path_buf),
                display_origin: format!("{git}@{rev}:{path}"),
            })
        }
        DepSource::Url(u) => {
            let content = fetch_url_cached(u, cache_dir).await?;
            let parsed =
                url::Url::parse(u).with_context(|| format!("parsing dep source URL {u}"))?;
            let hint = parsed
                .path_segments()
                .and_then(|mut s| s.next_back())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .ok_or_else(|| anyhow::anyhow!("URL has no filename component: {u}"))?;
            Ok(FetchedDepSource {
                content,
                filename_hint: hint,
                filesystem_base: None,
                display_origin: u.clone(),
            })
        }
    }
}

/// Wiring layer: fetch + parse every `DepSource` in order, preserving typed
/// input classes and source diagnostics. PyPI roots retain source/list order,
/// which is relevant to the caller's last-wins-by-name dedupe.
pub async fn resolve_deps_from(
    sources: &[DepSource],
    workspace_root: &Path,
    cache_dir: &Path,
) -> Result<ParsedDepsFrom> {
    let mut out = ParsedDepsFrom::default();
    for src in sources {
        let fetched = fetch_dep_source(src, workspace_root, cache_dir)
            .await
            .with_context(|| format!("retread-deps-from: fetching {src:?}"))?;
        let parsed = parse_dep_source(&fetched).with_context(|| {
            format!(
                "retread-deps-from: parsing {} ({src:?})",
                fetched.filename_hint
            )
        })?;
        for notice in &parsed.notices {
            notice.log();
        }
        out.extend(parsed);
    }
    Ok(out)
}

/// GET `url`, caching the response body under `cache_dir` keyed by a
/// sha256 hash of the URL string (the same style of cache key
/// `source_build::git_checkout_root` uses for git clones). Cache hits skip
/// the network entirely.
async fn fetch_url_cached(url: &str, cache_dir: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    let digest = hasher.finalize();
    let hash: String = digest.iter().take(12).map(|b| format!("{b:02x}")).collect();

    let dir = cache_dir.join("retread-dep-source-urls");
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("creating dep-source URL cache dir {}", dir.display()))?;
    let cached_path = dir.join(&hash);

    if let Ok(cached) = tokio::fs::read_to_string(&cached_path).await {
        return Ok(cached);
    }

    let body = reqwest::get(url)
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {url}"))?
        .text()
        .await
        .with_context(|| format!("reading body of {url}"))?;

    tokio::fs::write(&cached_path, &body)
        .await
        .with_context(|| format!("caching dep source URL body at {}", cached_path.display()))?;

    Ok(body)
}

/// Extract a file's base name as a `String` for use as `parse_dep_source`'s
/// `filename_hint`.
fn filename_of(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("path has no valid filename component: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fetched(content: &str, filename_hint: &str) -> FetchedDepSource {
        FetchedDepSource {
            content: content.to_string(),
            filename_hint: filename_hint.to_string(),
            filesystem_base: None,
            display_origin: format!("test:{filename_hint}"),
        }
    }

    fn fetched_at(content: &str, filename_hint: &str, base: &Path) -> FetchedDepSource {
        FetchedDepSource {
            content: content.to_string(),
            filename_hint: filename_hint.to_string(),
            filesystem_base: Some(base.to_path_buf()),
            display_origin: base.join(filename_hint).display().to_string(),
        }
    }

    /// Modeled on ProtoMotions' requirements_isaaclab.txt.
    const PROTOMOTIONS_REQUIREMENTS: &str = "\
tensordict==0.9.0
lightning
rtree==1.2.0
typer>=0.6.1
pkg[cli]>=1.9.4
# a comment line

-r base.txt
";

    #[test]
    fn parses_protomotions_style_requirements() {
        let result = parse_dep_source(&fetched(
            PROTOMOTIONS_REQUIREMENTS,
            "requirements_isaaclab.txt",
        ))
        .expect("requirements parse should succeed");
        assert_eq!(
            result.pypi_roots,
            vec![
                "tensordict==0.9.0".to_string(),
                "lightning".to_string(),
                "rtree==1.2.0".to_string(),
                "typer>=0.6.1".to_string(),
                "pkg[cli]>=1.9.4".to_string(),
            ]
        );
    }

    #[test]
    fn requirements_keeps_env_markers_and_skips_flags() {
        let content = "\
pkg==1; python_version<'3.11'
another>=1,<2  # inline comment
-e .
--extra-index-url https://example.com
.
http://example.com/pkg.tar.gz
foo @ git+https://example.com/foo.git
";
        let result = parse_dep_source(&fetched(content, "requirements.txt")).unwrap();
        assert_eq!(
            result.pypi_roots,
            vec![
                "pkg==1; python_version<'3.11'".to_string(),
                "another>=1,<2".to_string(),
                "http://example.com/pkg.tar.gz".to_string(),
                "foo @ git+https://example.com/foo.git".to_string(),
            ]
        );
    }

    /// Modeled on sage's server/pyproject.toml.
    const SAGE_PYPROJECT: &str = r#"
[project]
name = "sage-server"
version = "0.1.0"
dependencies = [
    "httpx",
    "mcp[cli]>=1.9.4",
    "mujoco==3.3.4",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]
"#;

    #[test]
    fn parses_sage_style_pyproject() {
        let result = parse_dep_source(&fetched(SAGE_PYPROJECT, "pyproject.toml"))
            .expect("pyproject parse should succeed");
        assert_eq!(
            result.pypi_roots,
            vec![
                "httpx".to_string(),
                "mcp[cli]>=1.9.4".to_string(),
                "mujoco==3.3.4".to_string(),
            ]
        );
    }

    #[test]
    fn empty_content_yields_empty_vec() {
        assert_eq!(
            parse_dep_source(&fetched("", "requirements.txt"))
                .unwrap()
                .pypi_roots,
            Vec::<String>::new()
        );
        assert_eq!(
            parse_dep_source(&fetched("", "pyproject.toml"))
                .unwrap()
                .pypi_roots,
            Vec::<String>::new()
        );
    }

    #[test]
    fn poetry_only_pyproject_is_an_error() {
        let content = r#"
[tool.poetry]
name = "foo"

[tool.poetry.dependencies]
python = "^3.10"
requests = "^2.28"
"#;
        let err = parse_dep_source(&fetched(content, "pyproject.toml")).unwrap_err();
        assert!(err.to_string().contains("poetry format not supported"));
    }

    #[test]
    fn uv_sources_missing_local_path_is_skipped_and_reported() {
        let workspace = unique_tmp_dir("uv-source-missing");
        let content = r#"
[project]
name = "example"
version = "0.1.0"
dependencies = ["Local_Project", "requests>=2"]

[tool.uv.sources]
local-project = { path = "missing/local-project", editable = true }
"#;
        let parsed = parse_dep_source(&fetched_at(content, "pyproject.toml", &workspace))
            .expect("pyproject with a missing uv path should parse");

        assert_eq!(parsed.pypi_roots, vec!["requests>=2"]);
        assert_eq!(parsed.notices.len(), 1);
        let notice = &parsed.notices[0];
        assert_eq!(notice.dependency, "local-project");
        assert_eq!(notice.configured_path, "missing/local-project");
        assert_eq!(
            notice.resolved_path.as_deref(),
            Some(workspace.join("missing/local-project").as_path())
        );
        assert!(notice.reason.contains("does not exist"));
        assert!(notice.source.contains("pyproject.toml"));

        std::fs::remove_dir_all(workspace).ok();
    }

    #[test]
    fn uv_sources_existing_local_path_is_explicitly_nonportable() {
        let workspace = unique_tmp_dir("uv-source-existing");
        let local_project = workspace.join("local-project");
        std::fs::create_dir_all(&local_project).unwrap();
        let content = r#"
[project]
name = "example"
version = "0.1.0"
dependencies = ["local-project", "httpx"]

[tool.uv.sources]
local-project = { path = "local-project", editable = true }
"#;
        let parsed = parse_dep_source(&fetched_at(content, "pyproject.toml", &workspace))
            .expect("pyproject with an existing uv path should parse");

        assert_eq!(parsed.pypi_roots, vec!["httpx"]);
        assert_eq!(parsed.notices.len(), 1);
        assert!(parsed.notices[0].reason.contains("not portable"));

        std::fs::remove_dir_all(workspace).ok();
    }

    #[test]
    fn pyproject_dependencies_without_uv_sources_are_unchanged() {
        let content = r#"
[project]
name = "example"
version = "0.1.0"
dependencies = [
    "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git",
    "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@stable",
    "pillow==9.3.0",
]

[tool.uv.sources]
unrelated = { index = "private" }
"#;
        let parsed = parse_dep_source(&fetched(content, "pyproject.toml")).unwrap();
        assert_eq!(
            parsed.pypi_roots,
            vec![
                "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git",
                "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@stable",
                "pillow==9.3.0",
            ]
        );
        assert!(parsed.notices.is_empty());
    }

    #[test]
    fn uv_sources_mixed_local_and_nonlocal_variants_omit_only_the_dependency() {
        let content = r#"
[project]
name = "example"
version = "0.1.0"
dependencies = ["demo", "requests"]

[tool.uv.sources]
demo = [
    { path = "../demo", marker = "sys_platform == 'linux'" },
    { index = "private", marker = "sys_platform != 'linux'" },
]
"#;
        let parsed = parse_dep_source(&fetched(content, "pyproject.toml")).unwrap();
        assert_eq!(parsed.pypi_roots, vec!["requests"]);
        assert_eq!(parsed.notices.len(), 1);
        assert_eq!(parsed.notices[0].dependency, "demo");
        assert!(
            parsed.notices[0]
                .reason
                .contains("cannot be resolved from a raw URL")
        );
    }

    const ROBOGEN_ENVIRONMENT: &str = r#"
name: robogen
channels:
  - anaconda
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.9.16=h7a1cb2a_2
  - cuda-runtime=11.7.1=0
  - _libgcc_mutex=0.1=conda_forge
  - _openmp_mutex=4.5=2_gnu
  - pip:
      - absl-py==1.4.0
      - accelerate==0.21.0
"#;

    #[test]
    fn conda_environment_pip_list_becomes_pep508_roots() {
        let parsed = parse_dep_source(&fetched(ROBOGEN_ENVIRONMENT, "environment.yaml"))
            .expect("conda environment should parse");
        assert_eq!(
            parsed.pypi_roots,
            vec!["absl-py==1.4.0", "accelerate==0.21.0"]
        );
    }

    #[test]
    fn conda_environment_strips_build_strings_and_softens_pins() {
        let parsed = parse_dep_source(&fetched(ROBOGEN_ENVIRONMENT, "environment.yaml"))
            .expect("conda environment should parse");
        let requirements: Vec<String> = parsed
            .advisory_conda_floors
            .iter()
            .map(AdvisoryCondaFloor::as_conda_requirement)
            .collect();

        assert!(requirements.contains(&"python >=3.9.16".to_string()));
        assert!(requirements.contains(&"cuda-runtime >=11.7.1".to_string()));
        assert!(requirements.iter().all(|floor| !floor.contains("==")));
        assert!(
            requirements
                .iter()
                .all(|floor| !floor.contains("h7a1cb2a_2"))
        );
        assert!(requirements.iter().all(|floor| !floor.contains("2_gnu")));
    }

    #[test]
    fn conda_environment_preserves_safe_positive_lower_bounds() {
        let content = r#"
dependencies:
  - strict>1.0
  - compound>=1,<2
  - strongest>=1,>2
  - ambiguous>=1|<2
"#;
        let parsed = parse_dep_source(&fetched(content, "environment.yaml"))
            .expect("conda lower bounds should parse");
        let requirements: Vec<String> = parsed
            .advisory_conda_floors
            .iter()
            .map(AdvisoryCondaFloor::as_conda_requirement)
            .collect();

        assert!(requirements.contains(&"strict >1.0".to_string()));
        assert!(requirements.contains(&"compound >=1".to_string()));
        assert!(requirements.contains(&"strongest >2".to_string()));
        assert!(
            requirements
                .iter()
                .all(|floor| !floor.starts_with("ambiguous "))
        );
        assert!(parsed.notices.iter().any(|notice| {
            notice.dependency == "ambiguous"
                && notice
                    .reason
                    .contains("no safe positive PEP 440 lower bound")
        }));
    }

    #[test]
    fn conda_environment_skips_and_reports_mutex_noise() {
        let parsed = parse_dep_source(&fetched(ROBOGEN_ENVIRONMENT, "environment.yaml"))
            .expect("conda environment should parse");
        let skipped: Vec<&str> = parsed
            .notices
            .iter()
            .map(|notice| notice.dependency.as_str())
            .collect();

        assert_eq!(skipped, vec!["_libgcc_mutex", "_openmp_mutex"]);
        assert!(
            parsed
                .notices
                .iter()
                .all(|notice| notice.reason.contains("environment-export noise"))
        );
    }

    #[test]
    fn yaml_hint_never_falls_back_to_requirements_parsing() {
        let content = "name: broken\ndependencies: not-a-list\n  - bogus==1\n";
        let err = parse_dep_source(&fetched(content, "environment.yml")).unwrap_err();
        assert!(err.to_string().contains("dependencies"));
        assert!(err.to_string().contains("list"));
    }

    #[test]
    fn ambiguous_filename_sniffs_conda_environment_shape() {
        let parsed = parse_dep_source(&fetched(ROBOGEN_ENVIRONMENT, "external-dependencies"))
            .expect("content sniff should recognize conda environment YAML");
        assert_eq!(
            parsed.pypi_roots,
            vec!["absl-py==1.4.0", "accelerate==0.21.0"]
        );
        assert_eq!(parsed.advisory_conda_floors.len(), 2);
        assert!(
            parsed
                .pypi_roots
                .iter()
                .all(|root| !root.contains("dependencies:") && !root.contains("channels:"))
        );

        let flow_style = "dependencies:\n  [python=3.9.16=h7a1cb2a_2]\n";
        let parsed = parse_dep_source(&fetched(flow_style, "external-dependencies"))
            .expect("semantic YAML sniff should precede TOML section heuristics");
        assert_eq!(
            parsed.advisory_conda_floors[0].as_conda_requirement(),
            "python >=3.9.16"
        );
    }

    #[test]
    fn requirements_and_pyproject_dispatch_remain_distinct() {
        let requirements = parse_dep_source(&fetched("demo==1\n", "requirements.txt")).unwrap();
        let pyproject = parse_dep_source(&fetched(
            "[project]\nname='x'\nversion='0.1'\ndependencies=['demo==1']\n",
            "pyproject.toml",
        ))
        .unwrap();

        assert_eq!(requirements.pypi_roots, vec!["demo==1"]);
        assert_eq!(pyproject.pypi_roots, vec!["demo==1"]);
        assert!(requirements.advisory_conda_floors.is_empty());
        assert!(pyproject.advisory_conda_floors.is_empty());
    }

    // --- fetch_dep_source ---------------------------------------------

    /// Uses only std (no tempfile crate dependency), matching the
    /// convention in `handler::replay_tests` -- a unique subdir of
    /// `std::env::temp_dir()` per test call.
    fn unique_tmp_dir(tag: &str) -> PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "retread-deps-from-test-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let dir = base.join(unique);
        std::fs::create_dir_all(&dir).expect("tmp dir creation should not fail");
        dir
    }

    /// Local path + parse pipeline end-to-end: no network needed.
    #[tokio::test]
    async fn fetch_local_round_trips_through_parse() {
        let workspace = unique_tmp_dir("local-roundtrip");
        std::fs::write(
            workspace.join("requirements_isaaclab.txt"),
            PROTOMOTIONS_REQUIREMENTS,
        )
        .expect("write temp requirements.txt");

        let src = DepSource::Local(PathBuf::from("requirements_isaaclab.txt"));
        let cache_dir = workspace.join("cache"); // unused by Local, but must be a valid Path
        let fetched = fetch_dep_source(&src, &workspace, &cache_dir)
            .await
            .expect("fetch_dep_source(Local) should succeed");

        assert_eq!(fetched.filename_hint, "requirements_isaaclab.txt");
        assert_eq!(fetched.content, PROTOMOTIONS_REQUIREMENTS);
        assert_eq!(
            fetched.filesystem_base.as_deref(),
            Some(workspace.as_path())
        );

        let parsed = parse_dep_source(&fetched).expect("parse should succeed");
        assert_eq!(
            parsed.pypi_roots,
            vec![
                "tensordict==0.9.0".to_string(),
                "lightning".to_string(),
                "rtree==1.2.0".to_string(),
                "typer>=0.6.1".to_string(),
                "pkg[cli]>=1.9.4".to_string(),
            ]
        );

        std::fs::remove_dir_all(workspace).ok();
    }

    /// Local path given as absolute is used as-is (not joined onto
    /// workspace_root).
    #[tokio::test]
    async fn fetch_local_absolute_path_ignores_workspace_root() {
        let dir = unique_tmp_dir("local-absolute-file");
        let file = dir.join("pyproject.toml");
        std::fs::write(&file, SAGE_PYPROJECT).expect("write temp pyproject.toml");

        let other_workspace = unique_tmp_dir("local-absolute-workspace");
        let src = DepSource::Local(file.clone());
        let fetched = fetch_dep_source(&src, &other_workspace, &other_workspace)
            .await
            .expect("fetch_dep_source(Local, absolute) should succeed");

        assert_eq!(fetched.filename_hint, "pyproject.toml");
        assert_eq!(fetched.content, SAGE_PYPROJECT);

        std::fs::remove_dir_all(dir).ok();
        std::fs::remove_dir_all(other_workspace).ok();
    }

    /// Missing local file surfaces a readable error rather than panicking.
    #[tokio::test]
    async fn fetch_local_missing_file_errors() {
        let workspace = unique_tmp_dir("local-missing");
        let src = DepSource::Local(PathBuf::from("does_not_exist.txt"));
        let err = fetch_dep_source(&src, &workspace, &workspace)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("does_not_exist.txt"));

        std::fs::remove_dir_all(workspace).ok();
    }

    /// Root-assembly test: `resolve_deps_from` over a single Local
    /// source produces exactly the PEP 508 strings a caller (e.g.
    /// `uv_group_closure`) would extend its root requirement set with.
    #[tokio::test]
    async fn resolve_deps_from_local_source_yields_requirements() {
        let workspace = unique_tmp_dir("resolve-roots-local");
        std::fs::write(
            workspace.join("requirements_isaaclab.txt"),
            PROTOMOTIONS_REQUIREMENTS,
        )
        .expect("write temp requirements.txt");

        let sources = vec![DepSource::Local(PathBuf::from("requirements_isaaclab.txt"))];
        let cache_dir = workspace.join("cache");
        let parsed = resolve_deps_from(&sources, &workspace, &cache_dir)
            .await
            .expect("resolve_deps_from should succeed");

        assert_eq!(
            parsed.pypi_roots,
            vec![
                "tensordict==0.9.0".to_string(),
                "lightning".to_string(),
                "rtree==1.2.0".to_string(),
                "typer>=0.6.1".to_string(),
                "pkg[cli]>=1.9.4".to_string(),
            ]
        );

        std::fs::remove_dir_all(workspace).ok();
    }

    /// Multiple sources flatten in list order.
    #[tokio::test]
    async fn resolve_deps_from_multiple_sources_flatten_in_order() {
        let workspace = unique_tmp_dir("resolve-roots-multi");
        std::fs::write(workspace.join("a.txt"), "foo==1.0\n").expect("write a.txt");
        std::fs::write(workspace.join("b.txt"), "bar==2.0\n").expect("write b.txt");

        let sources = vec![
            DepSource::Local(PathBuf::from("a.txt")),
            DepSource::Local(PathBuf::from("b.txt")),
        ];
        let cache_dir = workspace.join("cache");
        let parsed = resolve_deps_from(&sources, &workspace, &cache_dir)
            .await
            .expect("resolve_deps_from should succeed");

        assert_eq!(
            parsed.pypi_roots,
            vec!["foo==1.0".to_string(), "bar==2.0".to_string()]
        );

        std::fs::remove_dir_all(workspace).ok();
    }

    // --- DepSource (de)serialization -----------------------------------

    #[test]
    fn depsource_bare_string_local_vs_url_sniff() {
        let local: DepSource = toml::from_str("v = \"requirements.txt\"")
            .map(|t: toml::Value| DepSource::deserialize(t.get("v").unwrap().clone()).unwrap())
            .unwrap();
        assert_eq!(local, DepSource::Local(PathBuf::from("requirements.txt")));

        let url: DepSource = toml::from_str("v = \"https://example.com/requirements.txt\"")
            .map(|t: toml::Value| DepSource::deserialize(t.get("v").unwrap().clone()).unwrap())
            .unwrap();
        assert_eq!(
            url,
            DepSource::Url("https://example.com/requirements.txt".to_string())
        );
    }

    #[test]
    fn depsource_table_form_is_git() {
        let toml_val: toml::Value = toml::from_str(
            r#"git = "https://github.com/foo/bar"
rev = "deadbeef"
path = "requirements.txt"
"#,
        )
        .unwrap();
        let src = DepSource::deserialize(toml_val).unwrap();
        assert_eq!(
            src,
            DepSource::Git {
                git: "https://github.com/foo/bar".to_string(),
                rev: "deadbeef".to_string(),
                path: "requirements.txt".to_string(),
            }
        );
    }

    /// Live network: fetch ProtoMotions' requirements_isaaclab.txt straight
    /// off raw.githubusercontent.com at a pinned commit. Ignored by default
    /// (`cargo test deps_from` does not need network); run explicitly with
    /// `cargo test deps_from -- --ignored`.
    #[tokio::test]
    #[ignore = "hits raw.githubusercontent.com; run with --ignored"]
    async fn fetch_url_live_protomotions_requirements() {
        let cache_dir = unique_tmp_dir("url-live-cache");
        let workspace = unique_tmp_dir("url-live-workspace");
        let src = DepSource::Url(
            "https://raw.githubusercontent.com/NVlabs/ProtoMotions/main/requirements_isaaclab.txt"
                .to_string(),
        );
        let fetched = fetch_dep_source(&src, &workspace, &cache_dir)
            .await
            .expect("fetch_dep_source(Url) should succeed");
        assert_eq!(fetched.filename_hint, "requirements_isaaclab.txt");
        assert!(!fetched.content.is_empty());
        parse_dep_source(&fetched).expect("parse should succeed");

        // Second call should hit the on-disk cache (same URL hash), not
        // the network -- exercised implicitly by not erroring/timing out.
        let fetched2 = fetch_dep_source(&src, &workspace, &cache_dir)
            .await
            .expect("cached fetch_dep_source(Url) should succeed");
        assert_eq!(fetched.content, fetched2.content);

        std::fs::remove_dir_all(cache_dir).ok();
        std::fs::remove_dir_all(workspace).ok();
    }

    /// Live network: clone ProtoMotions at a pinned rev and read
    /// requirements_isaaclab.txt out of the checkout via the reused
    /// `source_build::ensure_git_checkout` helper.
    #[tokio::test]
    #[ignore = "clones github.com/NVlabs/ProtoMotions; run with --ignored"]
    async fn fetch_git_live_protomotions_requirements() {
        let cache_dir = unique_tmp_dir("git-live-cache");
        let workspace = unique_tmp_dir("git-live-workspace");
        let src = DepSource::Git {
            git: "https://github.com/NVlabs/ProtoMotions".to_string(),
            rev: "main".to_string(),
            path: "requirements_isaaclab.txt".to_string(),
        };
        let fetched = fetch_dep_source(&src, &workspace, &cache_dir)
            .await
            .expect("fetch_dep_source(Git) should succeed");
        assert_eq!(fetched.filename_hint, "requirements_isaaclab.txt");
        assert!(!fetched.content.is_empty());
        parse_dep_source(&fetched).expect("parse should succeed");

        std::fs::remove_dir_all(cache_dir).ok();
        std::fs::remove_dir_all(workspace).ok();
    }
}
