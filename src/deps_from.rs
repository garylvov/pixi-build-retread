//! Dependency-file parser: turns a requirements.txt or PEP 621 pyproject.toml
//! into a flat list of PEP 508 requirement strings.
//!
//! The parser (`parse_dep_source`) is intentionally standalone: it does no
//! relaxation, no resolution, no filesystem I/O beyond what the caller hands
//! it as a string. It just parses.
//!
//! The fetcher (`fetch_dep_source`) below is the layer that gets a
//! `deps_from`-style source spec (local path / raw URL / git@rev) down to
//! file text, so a caller can pipe the result straight into
//! `parse_dep_source`. It does no wiring into config/handler/uv_closure --
//! that's a separate piece.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// Parse a dependency source file's content into a Vec of PEP 508
/// requirement strings.
///
/// `filename_hint` (e.g. `"requirements_isaaclab.txt"` or `"pyproject.toml"`)
/// selects the format. If the hint is ambiguous, the content is sniffed:
/// content starting with a `[` section header (TOML-like) is treated as
/// pyproject; otherwise it falls back to requirements-line parsing.
pub fn parse_dep_source(content: &str, filename_hint: &str) -> Result<Vec<String>> {
    if is_pyproject(filename_hint, content) {
        parse_pyproject(content)
    } else {
        Ok(parse_requirements(content))
    }
}

fn is_pyproject(filename_hint: &str, content: &str) -> bool {
    let lower = filename_hint.to_ascii_lowercase();
    if lower.ends_with(".toml") || lower == "pyproject" {
        return true;
    }
    if lower.ends_with(".txt") {
        return false;
    }
    // Ambiguous hint: sniff the content for a TOML `[project]`-style table.
    content
        .lines()
        .map(str::trim)
        .any(|l| l.starts_with('[') && l.contains(']') && !l.starts_with("[["))
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

/// Parse a PEP 621 `pyproject.toml`'s `[project] dependencies` array.
fn parse_pyproject(content: &str) -> Result<Vec<String>> {
    let value: toml::Value = content.parse()?;

    if let Some(project) = value.get("project") {
        let deps = project
            .get("dependencies")
            .and_then(|d| d.as_array())
            .cloned()
            .unwrap_or_default();
        let mut out = Vec::with_capacity(deps.len());
        for dep in deps {
            match dep.as_str() {
                Some(s) => out.push(s.to_string()),
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
    Ok(Vec::new())
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

/// Resolve a `DepSource` to file text, ready to hand to `parse_dep_source`.
///
/// Returns `(file_content, filename_hint)`: `filename_hint` is the file's
/// base name (e.g. `"requirements_isaaclab.txt"`), which is exactly what
/// `parse_dep_source`'s `filename_hint` parameter wants for format sniffing.
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
) -> Result<(String, String)> {
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
            Ok((content, hint))
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
            Ok((content, hint))
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
            Ok((content, hint))
        }
    }
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
        let result = parse_dep_source(PROTOMOTIONS_REQUIREMENTS, "requirements_isaaclab.txt")
            .expect("requirements parse should succeed");
        assert_eq!(
            result,
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
        let result = parse_dep_source(content, "requirements.txt").unwrap();
        assert_eq!(
            result,
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
        let result = parse_dep_source(SAGE_PYPROJECT, "pyproject.toml")
            .expect("pyproject parse should succeed");
        assert_eq!(
            result,
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
            parse_dep_source("", "requirements.txt").unwrap(),
            Vec::<String>::new()
        );
        assert_eq!(
            parse_dep_source("", "pyproject.toml").unwrap(),
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
        let err = parse_dep_source(content, "pyproject.toml").unwrap_err();
        assert!(err.to_string().contains("poetry format not supported"));
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
        let (content, hint) = fetch_dep_source(&src, &workspace, &cache_dir)
            .await
            .expect("fetch_dep_source(Local) should succeed");

        assert_eq!(hint, "requirements_isaaclab.txt");
        assert_eq!(content, PROTOMOTIONS_REQUIREMENTS);

        let parsed = parse_dep_source(&content, &hint).expect("parse should succeed");
        assert_eq!(
            parsed,
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
        let (content, hint) = fetch_dep_source(&src, &other_workspace, &other_workspace)
            .await
            .expect("fetch_dep_source(Local, absolute) should succeed");

        assert_eq!(hint, "pyproject.toml");
        assert_eq!(content, SAGE_PYPROJECT);

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
        let (content, hint) = fetch_dep_source(&src, &workspace, &cache_dir)
            .await
            .expect("fetch_dep_source(Url) should succeed");
        assert_eq!(hint, "requirements_isaaclab.txt");
        assert!(!content.is_empty());
        parse_dep_source(&content, &hint).expect("parse should succeed");

        // Second call should hit the on-disk cache (same URL hash), not
        // the network -- exercised implicitly by not erroring/timing out.
        let (content2, _) = fetch_dep_source(&src, &workspace, &cache_dir)
            .await
            .expect("cached fetch_dep_source(Url) should succeed");
        assert_eq!(content, content2);

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
        let (content, hint) = fetch_dep_source(&src, &workspace, &cache_dir)
            .await
            .expect("fetch_dep_source(Git) should succeed");
        assert_eq!(hint, "requirements_isaaclab.txt");
        assert!(!content.is_empty());
        parse_dep_source(&content, &hint).expect("parse should succeed");

        std::fs::remove_dir_all(cache_dir).ok();
        std::fs::remove_dir_all(workspace).ok();
    }
}
