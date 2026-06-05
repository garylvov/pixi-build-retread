//! PEP 503 simple-index resolver. Turns `(name, version, index)` into a
//! concrete `(url, sha256)` for the wheel matching our target platform +
//! python tag.

use anyhow::{Result, bail};
use regex::Regex;
use std::str::FromStr;
use std::sync::OnceLock;
use uv_pep508::uv_pep440::{self, VersionSpecifiers};

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

/// Fetch the simple index for `name` and pick the best wheel for the
/// `(specifiers, target)` pair. When `specifiers` matches multiple versions
/// on the index (e.g. `>=5`), the highest matching version that also has a
/// target-compatible wheel wins.
pub async fn resolve(
    index: &str,
    name: &str,
    specifiers: &VersionSpecifiers,
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

    // PEP 427 says wheel filenames use the project distribution name with
    // `-` replaced by `_`. Some publishers preserve case (`Pillow-...whl`),
    // some normalize to lowercase (`pillow-...whl`). PEP 503 normalizes
    // names case-insensitively, so we match the same way: lowercase both
    // the prefix and the filename before comparing.
    let name_prefix_lower = format!("{}-", name.replace('-', "_").to_ascii_lowercase());

    // Parse (version, candidate) from every wheel whose filename name-prefix
    // matches, then filter to those whose version satisfies `specifiers`.
    // PEP 440 normalization makes `5.1.0` and `5.1.0.0` compare equal, so
    // a user pin of `==5.1.0` still matches a four-component publisher tag.
    let mut versioned: Vec<(uv_pep440::Version, ResolvedWheel)> = candidates
        .into_iter()
        .filter_map(|c| {
            let filename_lower = c.filename.to_ascii_lowercase();
            let rest = filename_lower.strip_prefix(&name_prefix_lower)?;
            let version_str = rest.split('-').next()?;
            let version = uv_pep440::Version::from_str(version_str).ok()?;
            Some((version, c))
        })
        .filter(|(v, _)| specifiers.contains(v))
        .collect();
    if versioned.is_empty() {
        bail!(
            "no wheels match {name} {specifiers} at {index_url}; \
             checked PEP 440 normalized version against case-insensitive filename prefix `{name_prefix_lower}`"
        );
    }

    // Descending version order. For each version (highest first), see if any
    // of its wheels are target-compatible; return the first match. The
    // version sort runs before the tag pick so a higher-version wheel that
    // happens to be (say) cp310-only doesn't block a lower-version cp311.
    versioned.sort_by(|a, b| b.0.cmp(&a.0));
    let mut grouped: Vec<(uv_pep440::Version, Vec<ResolvedWheel>)> = Vec::new();
    for (v, w) in versioned {
        match grouped.last_mut() {
            Some((last_v, group)) if *last_v == v => group.push(w),
            _ => grouped.push((v, vec![w])),
        }
    }
    for (_v, group) in grouped {
        if let Some(picked) = pick_best(group, target) {
            return Ok(picked);
        }
    }
    bail!(
        "no wheel for {name} {specifiers} at {index_url} matches target \
         python={} subdir={}",
        target.python_version,
        target.conda_subdir,
    )
}

/// v0.18.0+: PyPI Simple resolution scoped to source distributions
/// (`.tar.gz`, `.zip`). Used as the BFS fallback when [`resolve`]
/// returns "no wheels match" for a dep that PyPI only ships as sdist
/// (gym, classic-control, packages where the maintainer never uploads
/// wheels). The caller pipes the returned sdist URL through
/// `source_build::build_wheel_from_sdist_url` -> uv build --wheel,
/// then the produced wheel reenters the normal bundle pipeline.
pub async fn resolve_sdist(
    index: &str,
    name: &str,
    specifiers: &VersionSpecifiers,
) -> Result<ResolvedWheel> {
    let index_url = build_index_url(index, name)?;
    tracing::info!(url = %index_url, "sdist fallback: fetching simple index");
    let html = reqwest::get(index_url.clone())
        .await?
        .error_for_status()?
        .text()
        .await?;

    let mut candidates = parse_index_links_any(&html, &index_url)?;
    // sdist suffixes per PEP 625 + the legacy ones still on PyPI.
    candidates.retain(|c| {
        let f = c.filename.to_ascii_lowercase();
        f.ends_with(".tar.gz") || f.ends_with(".zip") || f.ends_with(".tar.bz2")
    });
    if candidates.is_empty() {
        bail!("no sdists listed at {index_url}");
    }

    let name_norm_dash = name.replace('_', "-").to_ascii_lowercase();
    let name_norm_underscore = name.replace('-', "_").to_ascii_lowercase();
    let mut versioned: Vec<(uv_pep440::Version, ResolvedWheel)> = candidates
        .into_iter()
        .filter_map(|c| {
            let f_lower = c.filename.to_ascii_lowercase();
            // sdist filenames: `<name>-<version>.tar.gz`. Name uses
            // either dash or underscore depending on the maintainer.
            // Try both, longest-prefix wins (so `gym-0.23.1.tar.gz`
            // doesn't match for `gym-notices`).
            let stem = f_lower
                .strip_suffix(".tar.gz")
                .or_else(|| f_lower.strip_suffix(".zip"))
                .or_else(|| f_lower.strip_suffix(".tar.bz2"))?;
            let rest = stem
                .strip_prefix(&format!("{name_norm_dash}-"))
                .or_else(|| stem.strip_prefix(&format!("{name_norm_underscore}-")))?;
            let version = uv_pep440::Version::from_str(rest).ok()?;
            Some((version, c))
        })
        .filter(|(v, _)| specifiers.contains(v))
        .collect();
    if versioned.is_empty() {
        bail!("no sdist for {name} {specifiers} at {index_url}");
    }
    versioned.sort_by(|a, b| b.0.cmp(&a.0));
    Ok(versioned.into_iter().next().unwrap().1)
}

/// Parser variant that DOES NOT filter by `.whl` suffix. Used by
/// [`resolve_sdist`] which wants tarballs.
fn parse_index_links_any(html: &str, base: &url::Url) -> Result<Vec<ResolvedWheel>> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r#"href="([^"]+)""#).unwrap());
    let mut out = Vec::new();
    for cap in re.captures_iter(html) {
        let href = &cap[1];
        let url = match base.join(href) {
            Ok(u) => u,
            Err(_) => continue,
        };
        let filename = match url.path_segments().and_then(|mut s| s.next_back()) {
            Some(f) if !f.is_empty() => percent_encoding::percent_decode_str(f)
                .decode_utf8_lossy()
                .into_owned(),
            _ => continue,
        };
        let sha256 = url.fragment().and_then(|f| {
            f.strip_prefix("sha256=")
                .filter(|h| h.len() == 64 && h.chars().all(|c| c.is_ascii_hexdigit()))
                .map(|h| h.to_ascii_lowercase())
        });
        out.push(ResolvedWheel {
            url,
            sha256,
            filename,
        });
    }
    if out.is_empty() {
        bail!("no `<a href=...>` links found at index");
    }
    Ok(out)
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
        // `path_segments()` returns segments percent-encoded. miropsota's
        // torch_packages_builder index URL-encodes `+` (the PEP 440
        // local-version-identifier marker) as `%2B`, so the raw segment
        // for a pytorch3d wheel reads
        //   `pytorch3d-0.7.9%2Bd9839a9pt2.10.0cpu-cp311-cp311-linux_x86_64.whl`.
        // uv_pep440 rejects that on parse (% isn't legal in a version),
        // so we'd silently drop every candidate and report "no wheels
        // match" at the version-filter step. Decode here so the filename
        // we store and parse downstream is the literal wheel name.
        let filename = match url.path_segments().and_then(|mut s| s.next_back()) {
            Some(f) if !f.is_empty() => percent_encoding::percent_decode_str(f)
                .decode_utf8_lossy()
                .into_owned(),
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
        out.push(ResolvedWheel {
            url,
            sha256,
            filename,
        });
    }
    if out.is_empty() {
        bail!("no `<a href=...>` wheel links found at index");
    }
    Ok(out)
}

fn pick_best(mut candidates: Vec<ResolvedWheel>, target: &WheelTarget) -> Option<ResolvedWheel> {
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

    let py_score = match score_python_tag_with_abi(parts.python, parts.abi, &target.python_version)
    {
        Some(s) => s,
        None => return -1,
    };
    let plat_score = match score_platform_tag(parts.platform, &target.conda_subdir) {
        Some(s) => s,
        None => return -1,
    };

    // Multiply python by a smaller factor so platform specificity dominates.
    py_score as i64 + plat_score
}

/// Wraps `score_python_tag` with abi3 awareness. PEP 425: when the ABI
/// tag is `abi3`, the wheel uses the CPython stable ABI and the python
/// tag declares the MINIMUM supported python (e.g. `cp36-abi3` is
/// compatible with python 3.6+, including 3.11). Without this, retread
/// rejected every psutil 5.x wheel (all are cp36-abi3) when the target
/// was 3.11.
fn score_python_tag_with_abi(python_tag: &str, abi_tag: &str, target: &str) -> Option<u32> {
    if abi_tag == "abi3" {
        let (target_major, target_minor) = parse_python_version(target)?;
        for sub in python_tag.split('.') {
            if let Some(score) = score_abi3_python(sub, target_major, target_minor) {
                return Some(score);
            }
        }
        // Fall through to regular check for cases like cpXY-abi3-none
        // where the cpXY happens to match exactly (caller would still
        // accept it via the regular path).
    }
    score_python_tag(python_tag, target)
}

/// abi3-specific: `cpXY-abi3` is compatible with any target python
/// where (major, minor) >= (X, Y).
fn score_abi3_python(tag: &str, t_major: u32, t_minor: u32) -> Option<u32> {
    let digits = tag.strip_prefix("cp")?;
    let (major, minor) = split_python_digits(digits)?;
    if major != t_major {
        return None; // py2 abi3 doesn't satisfy py3 (and vice versa)
    }
    if minor <= t_minor {
        // Slightly lower score than the exact `cpXY-cpXY` match so
        // identical-version-built wheels still win when both exist.
        Some(80)
    } else {
        None
    }
}

struct WheelTags<'a> {
    python: &'a str,
    /// PEP 425 ABI tag. v0.13.10+: consumed by
    /// `score_python_tag_with_abi` to treat `abi3` as compatible with
    /// any python >= the python tag's declared minimum.
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
    if let Some(digits) = tag.strip_prefix("cp") {
        // cp311 means CPython 3.11 (most preferred when matching exactly).
        let (major, minor) = split_python_digits(digits)?;
        if major == t_major && minor == t_minor {
            return Some(100);
        }
    } else if let Some(digits) = tag.strip_prefix("py") {
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

    if tag.contains(arch) { Some(120) } else { None }
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
        assert_eq!(
            links[0].filename,
            "isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl"
        );
        assert_eq!(
            links[0].sha256.as_deref(),
            Some("ad2c027831ed5d4a62552735bb799dea4e4604530d2ab9b526ddb6cd19a98c11"),
        );
    }

    #[test]
    fn decodes_percent_encoded_plus_in_filename() {
        // miropsota's torch_packages_builder hosts pytorch3d wheels at
        // GitHub release URLs where the `+` between upstream version and
        // local-version identifier is URL-encoded as `%2B`. Without
        // decoding the path segment, the stored filename contains `%2B`
        // literally, uv_pep440 rejects the version on parse, and every
        // candidate is silently filtered out at the version step
        // (symptom: "no wheels match pytorch3d ==0.7.8+... at <index>").
        let html = r#"
            <a href="https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9%2Bd9839a9/pytorch3d-0.7.9%2Bd9839a9pt2.10.0cpu-cp311-cp311-linux_x86_64.whl">link</a>
        "#;
        let base = url::Url::parse("https://miropsota.github.io/torch_packages_builder/pytorch3d/")
            .unwrap();
        let links = parse_index_links(html, &base).unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(
            links[0].filename, "pytorch3d-0.7.9+d9839a9pt2.10.0cpu-cp311-cp311-linux_x86_64.whl",
            "filename must be percent-decoded so `+` is literal for uv_pep440",
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
        let cands = vec![mk(
            "mujoco-3.5.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        )];
        let picked = pick_best(cands, &t()).unwrap();
        assert!(picked.filename.contains("manylinux_2_28_x86_64"));
    }

    /// v0.13.10+ regression: abi3 wheels declare the MINIMUM python they
    /// support via the python tag; the matcher must accept them for any
    /// target python >= that minimum. psutil 5.9.x only ships
    /// `cp36-abi3` wheels; before this fix retread bailed with
    /// "no wheel matches target python=3.11" even though py3.11
    /// satisfies the stable-ABI compatibility contract.
    #[test]
    fn abi3_wheel_matches_newer_python() {
        let cands = vec![mk(
            "psutil-5.9.8-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        )];
        let picked = pick_best(cands, &t()).unwrap();
        assert!(picked.filename.starts_with("psutil-5.9.8-cp36-abi3-"));
    }

    /// Exact cpXY-cpXY match still beats cpXY-abi3 when both are present
    /// (closer-to-target version wins via a higher score).
    #[test]
    fn exact_python_tag_beats_abi3_when_both_present() {
        let cands = vec![
            mk("foo-1.0-cp36-abi3-manylinux_2_17_x86_64.whl"),
            mk("foo-1.0-cp311-cp311-manylinux_2_17_x86_64.whl"),
        ];
        let picked = pick_best(cands, &t()).unwrap();
        assert!(
            picked.filename.contains("cp311-cp311"),
            "exact cp311 should outscore cp36-abi3; got {}",
            picked.filename,
        );
    }

    /// abi3 with a python tag declaring a HIGHER min than target is
    /// rejected (e.g. cp310-abi3 doesn't satisfy target=3.9).
    #[test]
    fn abi3_python_min_higher_than_target_rejected() {
        let cands = vec![mk("foo-1.0-cp312-abi3-manylinux_2_17_x86_64.whl")];
        // target python is 3.11 (from t() helper); cp312-abi3 declares
        // min=3.12, so should be rejected.
        let picked = pick_best(cands, &t());
        assert!(picked.is_none(), "cp312-abi3 must not match target=3.11");
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

    #[test]
    fn local_version_identifier_round_trips_through_resolve_logic() {
        // pytorch3d ships wheels with PEP 440 local-version identifiers
        // ("+5043d15pt2.7.0cu128") that encode the torch/CUDA matrix.
        // This pins the resolve()-internal filename->version extraction
        // and the spec containment check for that case so future
        // refactors of the parsing block can't silently drop support.
        //
        // Mirrors the inline logic in resolve(): name-prefix strip,
        // split on '-' to get the version segment, parse via uv_pep440,
        // check VersionSpecifiers::contains.
        let spec = VersionSpecifiers::from_str("==0.7.8+5043d15pt2.7.0cu128").unwrap();
        let filename = "pytorch3d-0.7.8+5043d15pt2.7.0cu128-cp311-cp311-linux_x86_64.whl";
        let stem = filename.strip_suffix(".whl").unwrap();
        let rest = stem.strip_prefix("pytorch3d-").unwrap();
        let version_str = rest.split('-').next().unwrap();
        let version = uv_pep440::Version::from_str(version_str)
            .expect("uv_pep440 must accept PEP 440 local-version identifiers");
        assert!(
            spec.contains(&version),
            "spec `==0.7.8+5043d15pt2.7.0cu128` must match extracted version `{version}`",
        );

        // Negative: a wheel for the same upstream version but a
        // different local-version identifier MUST NOT match. Otherwise
        // we'd silently install a torch-2.6 build into a torch-2.7 env.
        let other = "pytorch3d-0.7.8+abcdefgpt2.6.0cu124-cp311-cp311-linux_x86_64.whl";
        let other_rest = other
            .strip_suffix(".whl")
            .unwrap()
            .strip_prefix("pytorch3d-")
            .unwrap();
        let other_v = uv_pep440::Version::from_str(other_rest.split('-').next().unwrap()).unwrap();
        assert!(
            !spec.contains(&other_v),
            "different +local identifier must not match the exact pin",
        );
    }

    fn mk(name: &str) -> ResolvedWheel {
        ResolvedWheel {
            url: format!("https://example.com/{name}").parse().unwrap(),
            sha256: Some("0".repeat(64)),
            filename: name.to_string(),
        }
    }
}
