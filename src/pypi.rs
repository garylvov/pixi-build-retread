//! PEP 503 simple-index resolver. Turns `(name, version, index)` into a
//! concrete `(url, sha256)` for the wheel matching our target platform +
//! python tag.

use anyhow::{anyhow, bail, Result};
use regex::Regex;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct ResolvedWheel {
    pub url: url::Url,
    /// SHA-256 hash, if the index advertised one in the URL fragment.
    /// PEP 503 recommends but does not require this; some indexes (e.g.
    /// py.mujoco.org) omit it. When absent, `fetch_wheel` computes it on
    /// download for caching / lock-file invalidation.
    pub sha256: Option<String>,
    pub filename: String,
}

/// What we need to know about the build target in order to pick the right
/// wheel from a set of candidates. Filled in from `CondaOutputsParams` and
/// the variant configuration.
#[derive(Debug, Clone)]
pub struct WheelTarget {
    /// e.g. "3.11"
    pub python_version: String,
    /// e.g. "linux-64", "osx-arm64", or "noarch" if the workspace is
    /// targeting a noarch package and we should prefer a `none-any` wheel.
    pub conda_subdir: String,
}

/// Fetch the simple index for `name` and pick the best wheel for
/// `(version, target)`. Returns the URL (absolute) and the sha256 carried in
/// the `#sha256=` URL fragment.
pub async fn resolve(
    index: &str,
    name: &str,
    version: &str,
    target: &WheelTarget,
) -> Result<ResolvedWheel> {
    let index_url = build_index_url(index, name)?;
    tracing::info!(url = %index_url, "fetching simple index");
    let html = reqwest::get(index_url.clone())
        .await?
        .error_for_status()?
        .text()
        .await?;

    let mut candidates = parse_index_links(&html, &index_url)?;
    candidates.retain(|c| c.filename.ends_with(".whl"));
    if candidates.is_empty() {
        bail!("no wheels listed at {index_url}");
    }

    let filename_pattern = wheel_filename_prefix(name, version);
    candidates.retain(|c| {
        c.filename.starts_with(&filename_pattern)
            // After the version comes either `-` (start of next tag) or `_` (some
            // version strings include trailing zeros that look like the prefix
            // boundary). Be strict: require `-` so we don't match `5.1.0.0a1`
            // when asked for `5.1.0`.
            && c.filename[filename_pattern.len()..].starts_with('-')
    });
    if candidates.is_empty() {
        bail!(
            "no wheels match {name} == {version} at {index_url}; \
             checked prefix `{filename_pattern}`"
        );
    }

    let picked = pick_best(candidates, target).ok_or_else(|| {
        anyhow!(
            "no wheel at {index_url} for {name}=={version} matches target \
             python={} subdir={}",
            target.python_version,
            target.conda_subdir,
        )
    })?;

    Ok(picked)
}

fn build_index_url(index: &str, name: &str) -> Result<url::Url> {
    // PEP 503: name normalized to lowercase, runs of `_.-` collapsed to a
    // single `-`. The index path ends with a `/`.
    let mut base = index.to_string();
    if !base.ends_with('/') {
        base.push('/');
    }
    let normalized = pep503_normalize(name);
    Ok(url::Url::parse(&format!("{base}{normalized}/"))?)
}

fn pep503_normalize(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_dash = false;
    for c in name.chars().flat_map(|c| c.to_lowercase()) {
        if c == '-' || c == '_' || c == '.' {
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

/// Wheel filename prefix per PEP 427: `{distribution}-{version}` where
/// distribution uses `_` in place of `-`. PEP 503 normalization to lowercase
/// applies too in modern wheels, but to be safe we don't enforce case here.
fn wheel_filename_prefix(name: &str, version: &str) -> String {
    format!("{}-{version}", name.replace('-', "_"))
}

fn parse_index_links(html: &str, base: &url::Url) -> Result<Vec<ResolvedWheel>> {
    static RE: OnceLock<Regex> = OnceLock::new();
    // PEP 503 advertises hashes in the URL fragment (`#sha256=<hex>`) but
    // doesn't require it. Match any `href="..."` and treat the hash as
    // optional so non-conforming indexes (py.mujoco.org, some self-hosted
    // simple repos) still work.
    let re = RE.get_or_init(|| Regex::new(r#"href="([^"]+)""#).unwrap());

    let mut out = Vec::new();
    for cap in re.captures_iter(html) {
        let href = &cap[1];
        let url = match base.join(href) {
            Ok(u) => u,
            Err(_) => continue,
        };
        let filename = match url.path_segments().and_then(|s| s.last()) {
            Some(f) if !f.is_empty() => f.to_string(),
            _ => continue,
        };
        if !filename.ends_with(".whl") {
            continue;
        }
        let sha256 = url.fragment().and_then(|f| {
            f.strip_prefix("sha256=")
                .filter(|h| h.len() == 64 && h.chars().all(|c| c.is_ascii_hexdigit()))
                .map(|h| h.to_ascii_lowercase())
        });
        out.push(ResolvedWheel { url, sha256, filename });
    }
    if out.is_empty() {
        bail!("no `<a href=...>` wheel links found at index");
    }
    Ok(out)
}

fn pick_best(
    mut candidates: Vec<ResolvedWheel>,
    target: &WheelTarget,
) -> Option<ResolvedWheel> {
    // Score each candidate; higher is better. Negative score = incompatible.
    candidates
        .iter()
        .map(|c| score_wheel(&c.filename, target))
        .zip(candidates.iter())
        .filter(|(s, _)| *s >= 0)
        .max_by_key(|(s, _)| *s)
        .map(|(_, c)| c.clone())
        .or_else(|| candidates.pop())
        .filter(|c| score_wheel(&c.filename, target) >= 0)
}

/// Returns a non-negative score for a compatible wheel, -1 otherwise.
/// Larger means "more preferred" for this target.
fn score_wheel(filename: &str, target: &WheelTarget) -> i64 {
    let parts = match parse_wheel_tags(filename) {
        Some(t) => t,
        None => return -1,
    };

    let py_score = match score_python_tag(&parts.python, &target.python_version) {
        Some(s) => s,
        None => return -1,
    };
    let plat_score = match score_platform_tag(&parts.platform, &target.conda_subdir) {
        Some(s) => s,
        None => return -1,
    };

    // Multiply python by a smaller factor so platform specificity dominates.
    py_score as i64 + plat_score
}

struct WheelTags<'a> {
    python: &'a str,
    #[allow(dead_code)]
    abi: &'a str,
    platform: &'a str,
}

fn parse_wheel_tags(filename: &str) -> Option<WheelTags<'_>> {
    let stem = filename.strip_suffix(".whl")?;
    // {distribution}-{version}(-{build})?-{python}-{abi}-{platform}
    let rev: Vec<&str> = stem.rsplitn(4, '-').collect();
    if rev.len() < 3 {
        return None;
    }
    Some(WheelTags {
        platform: rev[0],
        abi: rev[1],
        python: rev[2],
    })
}

fn score_python_tag(tag: &str, target: &str) -> Option<u32> {
    // Compatible if any dotted sub-tag matches.
    let (target_major, target_minor) = parse_python_version(target)?;
    for sub in tag.split('.') {
        if let Some(score) = score_one_python(sub, target_major, target_minor) {
            return Some(score);
        }
    }
    None
}

fn score_one_python(tag: &str, t_major: u32, t_minor: u32) -> Option<u32> {
    if tag.starts_with("cp") {
        // cp311 means CPython 3.11 (most preferred when matching exactly).
        let digits = &tag[2..];
        let (major, minor) = split_python_digits(digits)?;
        if major == t_major && minor == t_minor {
            return Some(100);
        }
    } else if tag.starts_with("py") {
        let digits = &tag[2..];
        if digits.is_empty() {
            return None;
        }
        let (major, minor) = split_python_digits(digits)?;
        if major == t_major && (minor == 0 || minor == t_minor) {
            // py3 -> matches any 3.x (lower score), py311 -> matches 3.11
            return Some(if minor == 0 { 30 } else { 60 });
        }
    }
    None
}

fn split_python_digits(digits: &str) -> Option<(u32, u32)> {
    if digits.is_empty() {
        return None;
    }
    if digits.len() == 1 {
        Some((digits.parse().ok()?, 0))
    } else {
        let major: u32 = digits[..1].parse().ok()?;
        let minor: u32 = digits[1..].parse().ok()?;
        Some((major, minor))
    }
}

fn parse_python_version(s: &str) -> Option<(u32, u32)> {
    let mut parts = s.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    Some((major, minor))
}

fn score_platform_tag(tag: &str, conda_subdir: &str) -> Option<i64> {
    // PEP 425 compressed tag sets: a single wheel can list multiple platform
    // tags joined by `.`. Score each and return the best match.
    tag.split('.')
        .filter_map(|t| score_one_platform_tag(t, conda_subdir))
        .max()
}

fn score_one_platform_tag(tag: &str, conda_subdir: &str) -> Option<i64> {
    if tag == "any" {
        // Universal wheel. Match `noarch` strongly; for native subdirs it's
        // still usable (we'll just emit a noarch conda package).
        return Some(if conda_subdir == "noarch" { 1000 } else { 50 });
    }

    let arch = match conda_subdir {
        "linux-64" => "x86_64",
        "linux-aarch64" => "aarch64",
        "osx-64" => "x86_64",
        "osx-arm64" => "arm64",
        "win-64" => "amd64",
        // Unknown subdir: only `any` wheels are compatible.
        _ => return None,
    };

    if conda_subdir.starts_with("linux") {
        if let Some(rest) = tag.strip_prefix("manylinux_") {
            // Format: X_Y_<arch>
            let suffix = format!("_{arch}");
            let ver = rest.strip_suffix(&suffix)?;
            // Score: prefer higher glibc (more specific).
            let mut parts = ver.split('_');
            let major: i64 = parts.next()?.parse().ok()?;
            let minor: i64 = parts.next()?.parse().ok()?;
            return Some(200 + major * 100 + minor);
        }
        if tag.starts_with("manylinux1_") && tag.ends_with(arch) {
            return Some(150);
        }
        if tag.starts_with("manylinux2010_") && tag.ends_with(arch) {
            return Some(160);
        }
        if tag.starts_with("manylinux2014_") && tag.ends_with(arch) {
            return Some(170);
        }
        if tag == format!("linux_{arch}") {
            return Some(100);
        }
        return None;
    }

    if tag.contains(arch) {
        Some(120)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t() -> WheelTarget {
        WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "linux-64".into(),
        }
    }

    #[test]
    fn pep503_normalization() {
        assert_eq!(pep503_normalize("Foo_Bar.Baz"), "foo-bar-baz");
        assert_eq!(pep503_normalize("isaacsim"), "isaacsim");
        assert_eq!(pep503_normalize("isaacsim-Core"), "isaacsim-core");
        assert_eq!(pep503_normalize("isaacsim_kernel"), "isaacsim-kernel");
    }

    #[test]
    fn parses_pep503_links_with_hash() {
        let html = r#"
            <a href="isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl#sha256=ad2c027831ed5d4a62552735bb799dea4e4604530d2ab9b526ddb6cd19a98c11">link</a>
            <a href="isaacsim-5.1.0.0-cp311-none-win_amd64.whl#sha256=f2f4cbc13594749deb5905aebdf76ac68c3e5caef5db88be941b18735a889751">link</a>
        "#;
        let base = url::Url::parse("https://pypi.nvidia.com/isaacsim/").unwrap();
        let links = parse_index_links(html, &base).unwrap();
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].filename, "isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl");
        assert_eq!(
            links[0].sha256.as_deref(),
            Some("ad2c027831ed5d4a62552735bb799dea4e4604530d2ab9b526ddb6cd19a98c11"),
        );
    }

    #[test]
    fn parses_links_without_sha256() {
        // py.mujoco.org and other indexes don't include sha256 in the URL.
        let html = r#"
            <a href="/mujoco/mujoco-3.5.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl">mujoco-3.5.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl</a><br>
        "#;
        let base = url::Url::parse("https://py.mujoco.org/mujoco/").unwrap();
        let links = parse_index_links(html, &base).unwrap();
        assert_eq!(links.len(), 1);
        assert!(links[0].sha256.is_none());
        assert_eq!(
            links[0].url.as_str(),
            "https://py.mujoco.org/mujoco/mujoco-3.5.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
        );
    }

    #[test]
    fn compressed_tag_set_matches() {
        // PEP 425 compressed tag sets: one wheel that lists multiple platform
        // tags joined by `.`. Common with mujoco wheels.
        let cands = vec![
            mk("mujoco-3.5.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"),
        ];
        let picked = pick_best(cands, &t()).unwrap();
        assert!(picked.filename.contains("manylinux_2_28_x86_64"));
    }

    #[test]
    fn picks_linux_x86_64_wheel_over_others() {
        let cands = vec![
            mk("isaacsim-5.1.0.0-cp311-none-win_amd64.whl"),
            mk("isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl"),
            mk("isaacsim-5.1.0.0-cp311-none-manylinux_2_35_aarch64.whl"),
        ];
        let picked = pick_best(cands, &t()).unwrap();
        assert_eq!(
            picked.filename,
            "isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl"
        );
    }

    #[test]
    fn prefers_cp311_over_py3() {
        let cands = vec![
            mk("foo-1.0-py3-none-any.whl"),
            mk("foo-1.0-cp311-cp311-manylinux_2_28_x86_64.whl"),
        ];
        let picked = pick_best(cands, &t()).unwrap();
        assert!(picked.filename.contains("cp311"));
    }

    #[test]
    fn rejects_wrong_python_version() {
        let cands = vec![mk("foo-1.0-cp310-cp310-manylinux_2_28_x86_64.whl")];
        let picked = pick_best(cands, &t());
        assert!(picked.is_none());
    }

    #[test]
    fn pure_python_falls_back_to_any() {
        let cands = vec![mk("requests-2.32.5-py3-none-any.whl")];
        let picked = pick_best(cands, &t()).unwrap();
        assert!(picked.filename.contains("none-any"));
    }

    fn mk(name: &str) -> ResolvedWheel {
        ResolvedWheel {
            url: format!("https://example.com/{name}").parse().unwrap(),
            sha256: Some("0".repeat(64)),
            filename: name.to_string(),
        }
    }
}
