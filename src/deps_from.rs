//! Dependency-file parser: turns a requirements.txt or PEP 621 pyproject.toml
//! into a flat list of PEP 508 requirement strings.
//!
//! This module is intentionally standalone: it does no relaxation, no
//! resolution, no filesystem I/O beyond what the caller hands it as a
//! string. It just parses.

use anyhow::{bail, Result};

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
        assert_eq!(parse_dep_source("", "requirements.txt").unwrap(), Vec::<String>::new());
        assert_eq!(parse_dep_source("", "pyproject.toml").unwrap(), Vec::<String>::new());
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
}
