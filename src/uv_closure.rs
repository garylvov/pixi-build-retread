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
//! 3. Parse the PEP 751 pylock into the SAME closure/lock shapes the legacy
//!    cascade produces (`crate::lock::LockWheel`), selecting ONE wheel per
//!    package by tag priority (`crate::pypi::score_wheel`).
//!
//! Constraint *provenance* is load-bearing: every generated constraint line
//! carries a record of the conda source package it came from
//! (`constraints.provenance.json`), so a `uv lock` conflict can be
//! attributed to the offending conda pin and `retread solve` knows which
//! pin to widen. This layer is policy-free: it never widens, never retries
//! with altered inputs (spec §4c) — on conflict it reports and points at
//! `retread solve`.
//!
//! Selected behind `[package.build.config] retread-resolver = "uv"`
//! (default: `"legacy"`, the cascade/resolvo path — untouched).

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

// ---------------------------------------------------------------------------
// Ephemeral project synthesis
// ---------------------------------------------------------------------------

/// `tool.uv.environments` marker for a conda subdir. `None` for noarch /
/// unknown subdirs (the `environments` key is then omitted; tag selection
/// in [`parse_pylock_closure`] still enforces the platform).
pub fn environment_marker(conda_subdir: &str) -> Option<String> {
    let (platform, machine) = match conda_subdir {
        "linux-64" => ("linux", "x86_64"),
        "linux-aarch64" => ("linux", "aarch64"),
        "linux-ppc64le" => ("linux", "ppc64le"),
        "osx-64" => ("darwin", "x86_64"),
        "osx-arm64" => ("darwin", "arm64"),
        "win-64" => ("win32", "AMD64"),
        _ => return None,
    };
    Some(format!(
        "sys_platform == '{platform}' and platform_machine == '{machine}'"
    ))
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
    out.push_str(&format!("name = {}\n", toml_str(&project_name(&req.bundle))));
    out.push_str("version = \"0\"\n");
    out.push_str(&format!(
        "requires-python = {}\n",
        toml_str(&format!("=={}.*", req.python_version))
    ));
    out.push_str(&format!(
        "dependencies = {}\n",
        toml_string_array("", &req.dependencies)
    ));

    out.push_str("\n[tool.uv]\n");
    if let Some(marker) = environment_marker(&req.conda_subdir) {
        out.push_str(&format!("environments = [{}]\n", toml_str(&marker)));
    }
    // sdists are never built by uv (spec §8.2): retread's source_build
    // path owns builds. An sdist-only transitive fails the lock loudly.
    out.push_str("no-build = true\n");
    // Matches the installer's index semantics (installer.rs build_uv_args).
    out.push_str("index-strategy = \"unsafe-best-match\"\n");
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
            bail!(
                "package `{name}=={version}`: none of its {} wheel(s) is compatible \
                 with python {} on {} (tag selection). If only an sdist fits, route \
                 it via `retread-conda-deps` or a source entry.",
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
    if recorded_uv_version.is_none() {
        if let Ok(prev) = std::fs::read_to_string(&meta_path) {
            if let Ok(meta) = serde_json::from_str::<ClosureMeta>(&prev) {
                warn_on_uv_version_skew(&uv_version, Some(&meta.uv_version));
            }
        }
    }

    tokio::fs::create_dir_all(project_dir)
        .await
        .with_context(|| format!("creating uv project dir {}", project_dir.display()))?;
    tokio::fs::create_dir_all(uv_cache_dir)
        .await
        .with_context(|| format!("creating uv cache dir {}", uv_cache_dir.display()))?;
    tokio::fs::write(project_dir.join("pyproject.toml"), synthesize_pyproject(req))
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
    if req.offline {
        lock_args.push("--offline".into());
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
    let target = WheelTarget {
        python_version: req.python_version.clone(),
        conda_subdir: req.conda_subdir.clone(),
    };
    let closure = parse_pylock_closure(&pylock, &target, &exclude, &uv_version)?;

    let _ = std::fs::write(
        &meta_path,
        serde_json::to_string_pretty(&ClosureMeta {
            uv_version: uv_version.clone(),
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
requires-python = "==3.12.*"
dependencies = [
    "isaacsim[all,extscache]==5.1.0",
    "mujoco==3.5.0",
]

[tool.uv]
environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
no-build = true
index-strategy = "unsafe-best-match"
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
    fn synthesize_pyproject_noarch_omits_environments() {
        let mut req = sample_request();
        req.conda_subdir = "noarch".to_string();
        let got = synthesize_pyproject(&req);
        assert!(!got.contains("environments ="));
        assert!(got.contains("no-build = true"));
    }

    #[test]
    fn environment_marker_matrix() {
        assert_eq!(
            environment_marker("linux-64").as_deref(),
            Some("sys_platform == 'linux' and platform_machine == 'x86_64'")
        );
        assert_eq!(
            environment_marker("osx-arm64").as_deref(),
            Some("sys_platform == 'darwin' and platform_machine == 'arm64'")
        );
        assert_eq!(
            environment_marker("win-64").as_deref(),
            Some("sys_platform == 'win32' and platform_machine == 'AMD64'")
        );
        assert_eq!(environment_marker("noarch"), None);
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
        assert_eq!(closure.pins.get("isaaclab").map(String::as_str), Some("2.0.0"));
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

    // ---- live subprocess (network) ---------------------------------------

    /// Live-network smoke: requires uv on PATH + PyPI reachability.
    /// Run manually: `cargo test uv_closure -- --ignored`.
    #[tokio::test]
    #[ignore = "requires network + uv on PATH"]
    async fn live_uv_lock_smoke() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-uv-closure-smoke-{}",
            std::process::id()
        ));
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
        assert_eq!(closure.pins.get("typing-extensions").map(String::as_str), Some("4.12.2"));
        assert_eq!(closure.wheels.len(), 1);
        assert!(closure.wheels[0].sha256.is_some());
    }
}
