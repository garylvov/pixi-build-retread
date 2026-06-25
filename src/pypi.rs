//! PEP 503 simple-index resolver. Turns `(name, version, index)` into a
//! concrete `(url, sha256)` for the wheel matching our target platform +
//! python tag.
//!
//! PR-1a (resolvo foundation): adds `list_all_versions` / `PyPiCandidate`
//! for whole-index discovery -- the resolvo `DependencyProvider`'s
//! `get_candidates` needs every compatible version, not just the highest one.

use anyhow::{Result, bail};
use regex::Regex;
use std::str::FromStr;
use std::sync::OnceLock;
use uv_pep508::uv_pep440::{self, Version, VersionSpecifiers};

#[derive(Debug, Clone)]
pub struct ResolvedWheel {
    pub url: url::Url,
    /// SHA-256 hash, if the index advertised one in the URL fragment.
    /// PEP 503 recommends but does not require this; some indexes (e.g.
    /// py.mujoco.org) omit it. When absent, `fetch_wheel` computes it on
    /// download for caching / lock-file invalidation.
    pub sha256: Option<String>,
    pub filename: String,
    /// v1.4.3: true when the index link carried a PEP 658
    /// `data-dist-info-metadata` (or its PEP 714 rename,
    /// `data-core-metadata`) attribute -- the wheel's METADATA is then
    /// served as a sidecar at `<wheel_url>.metadata`, so callers that
    /// only need Requires-Dist can skip the (potentially multi-GB)
    /// wheel download. pypi.org serves it; pypi.nvidia.com and most
    /// static (GitHub Pages) indexes do not, so this stays best-effort.
    pub has_metadata_sidecar: bool,
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
    resolve_inner(index, name, specifiers, target, None).await
}

/// Like [`resolve`] but prefers `prefer_version` when it satisfies `specifiers`
/// and a target-compatible wheel exists for it. Falls back to highest-version
/// selection when the preferred version has no compatible wheel or does not
/// satisfy `specifiers`.
///
/// Used by the favor-lock path (`RETREAD_FAVOR_LOCK=1`) so cold re-resolves
/// that have a committed lock prefer the locked version, keeping the closure
/// stable across routine re-resolves (e.g. a pack with an unlocked range spec
/// stays pinned to the version the last engineer verified).
pub async fn resolve_preferring(
    index: &str,
    name: &str,
    specifiers: &VersionSpecifiers,
    target: &WheelTarget,
    prefer_version: &str,
) -> Result<ResolvedWheel> {
    resolve_inner(index, name, specifiers, target, Some(prefer_version)).await
}

/// Shared implementation for [`resolve`] and [`resolve_preferring`].
///
/// `prefer_version`: when `Some(v)` and `v` satisfies `specifiers`, the
/// grouped pass tries the preferred-version group FIRST.  If it has a
/// target-compatible wheel that wheel is returned immediately.  Otherwise
/// selection falls back to the normal highest-version sweep (no error is
/// raised because the locked version may simply not be on this index, or may
/// lack a compatible wheel for the current target).
async fn resolve_inner(
    index: &str,
    name: &str,
    specifiers: &VersionSpecifiers,
    target: &WheelTarget,
    prefer_version: Option<&str>,
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
    //
    // Uses the shared `extract_version_from_wheel_filename` helper (also used
    // by `list_all_versions`) so both functions stay in sync on parsing logic.
    let mut versioned: Vec<(uv_pep440::Version, ResolvedWheel)> = candidates
        .into_iter()
        .filter_map(|c| {
            let filename_lower = c.filename.to_ascii_lowercase();
            let version = extract_version_from_wheel_filename(&filename_lower, &name_prefix_lower)?;
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

    // favor-lock: if a preferred version was supplied AND it satisfies
    // `specifiers` AND a target-compatible wheel exists for it, return that
    // wheel immediately (before the normal highest-version sweep).  This
    // keeps re-resolves stable when a range spec (e.g. `>=5`) could legally
    // pick a newer version -- we keep the one that was previously verified.
    if let Some(pref_str) = prefer_version
        && let Ok(pref_ver) = Version::from_str(pref_str)
        && specifiers.contains(&pref_ver)
    {
        if let Some((_, group)) = grouped.iter().find(|(v, _)| *v == pref_ver)
            && let Some(picked) = pick_best(group.clone(), target)
        {
            tracing::debug!(
                dep = %name,
                preferred = %pref_str,
                wheel = %picked.filename,
                "favor-lock: using preferred locked version instead of latest",
            );
            return Ok(picked);
        }
        // Preferred version not on index or no compatible wheel; fall through to
        // normal highest-version selection (logged as trace so it's auditable).
        tracing::trace!(
            dep = %name,
            preferred = %pref_str,
            "favor-lock: preferred version absent or no compatible wheel; using latest",
        );
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
/// Resolve the best sdist URL for `(name, specifiers)` on `index`.
///
/// Returns `(resolved_version, wheel)` so callers can key the sdist build
/// cache dir on the exact resolved version and avoid rebuilding the same
/// `(name, version)` from divergent directories.
pub async fn resolve_sdist(
    index: &str,
    name: &str,
    specifiers: &VersionSpecifiers,
) -> Result<(uv_pep440::Version, ResolvedWheel)> {
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
    let (version, wheel) = versioned.into_iter().next().unwrap();
    Ok((version, wheel))
}

// ── PR-1a: all-versions discovery for the resolvo DependencyProvider ─────────

/// A single version on a PyPI simple index, as seen from the target platform's
/// perspective.
///
/// Used by the resolvo `DependencyProvider`'s `get_candidates` callback, which
/// needs the FULL candidate set (not just the highest matching version).  Each
/// `PyPiCandidate` represents one PEP 440 version that is reachable on the index
/// and is not excluded by platform incompatibility.
///
/// A candidate is either:
/// - **wheel** (`wheel` is `Some`): a platform-compatible wheel was found.
///   The `wheel` field holds the best wheel for this version (same pick as
///   `pypi::resolve` would use).  `sdist_url` MAY also be `Some` if a source
///   distribution exists alongside the wheel.
/// - **sdist-only** (`wheel` is `None`, `sdist_url` is `Some`): no compatible
///   wheel exists for this version, but an sdist does.  The resolvo provider
///   may later build the sdist via `source_build::build_wheel_from_sdist_url`.
///   `is_sdist_only()` is a convenience test for this case.
///
/// Versions with neither a compatible wheel NOR an sdist are silently omitted
/// by `list_all_versions` (they are unreachable from this target).
#[derive(Debug, Clone)]
pub struct PyPiCandidate {
    /// Parsed PEP 440 version.
    pub version: Version,
    /// Best target-compatible wheel for this version, if any.
    /// `None` when the version is sdist-only.
    pub wheel: Option<ResolvedWheel>,
    /// Source-distribution URL for this version, if one was advertised on the
    /// index.  Present for both wheel+sdist and sdist-only candidates.
    pub sdist_url: Option<url::Url>,
}

impl PyPiCandidate {
    /// True when no compatible wheel is available and only an sdist exists.
    /// The caller is responsible for building the sdist when this is true.
    pub fn is_sdist_only(&self) -> bool {
        self.wheel.is_none() && self.sdist_url.is_some()
    }
}

/// Fetch the simple index for `name` and return **all** version candidates that
/// are reachable from `target` (i.e. have a compatible wheel or at least an
/// sdist), sorted highest-version first.
///
/// This is the all-versions counterpart to [`resolve`], which returns only the
/// single highest matching version.  The resolvo `DependencyProvider` needs the
/// full list so the solver can backtrack to a lower version when the highest one
/// conflicts.
///
/// # Reuse
///
/// Internally reuses the same helpers as [`resolve`] and [`resolve_sdist`]:
/// - `build_index_url` -- PEP 503 normalized URL construction.
/// - `parse_index_links_any` -- parse ALL `<a href=...>` links on the page.
/// - `extract_version_from_wheel_filename` -- name-prefix + version segment
///   extraction, shared with `resolve` to avoid drift.
/// - `pick_best` / `score_wheel` -- identical platform compat filter; if
///   `pick_best` returns `None` for a version, that version has no compatible
///   wheel for this target.
///
/// # Sdist-only versions
///
/// If a version has no compatible wheel but the index lists a source
/// distribution (`.tar.gz`, `.zip`, `.tar.bz2`), the version is included as
/// a `PyPiCandidate` with `wheel = None` and `sdist_url = Some(...)`.  The
/// caller decides whether to build the sdist (resolvo PR-1b/discovery pass).
/// Building is NOT done here.
///
/// # Errors
///
/// Returns an error only on HTTP failure or a completely empty/unparseable
/// index.  An index that lists files but none for `name` is OK -- returns an
/// empty `Vec`.
pub async fn list_all_versions(
    index: &str,
    name: &str,
    target: &WheelTarget,
) -> Result<Vec<PyPiCandidate>> {
    let index_url = build_index_url(index, name)?;
    tracing::info!(url = %index_url, "list_all_versions: fetching simple index");
    let html = reqwest::get(index_url.clone())
        .await?
        .error_for_status()?
        .text()
        .await?;

    // Parse all links on the page (wheels + sdists).
    let all_links = match parse_index_links_any(&html, &index_url) {
        Ok(links) => links,
        // Empty index: return no candidates rather than propagating the error.
        Err(_) => return Ok(vec![]),
    };

    // Shared name prefix for wheel filename matching (same as resolve()).
    // PEP 503: case-insensitive, `-` / `_` are equivalent in names.
    let name_prefix_lower = format!("{}-", name.replace('-', "_").to_ascii_lowercase());

    // ── Separate wheels from sdists ──────────────────────────────────────────

    // Wheel candidates: (parsed_version, ResolvedWheel).  We need the
    // compat-tagged ResolvedWheel to pass to pick_best, but parse_index_links_any
    // returns plain ResolvedWheel structs without the compat info.  Re-parse
    // the compat attributes via score_wheel later.
    //
    // We rebuild proper ResolvedWheel structs from parse_index_links_any output;
    // they already carry url, sha256, filename, has_metadata_sidecar.
    let mut wheels_by_version: std::collections::BTreeMap<Version, Vec<ResolvedWheel>> =
        std::collections::BTreeMap::new();

    // Sdist candidates: (parsed_version, url).
    let mut sdists_by_version: std::collections::BTreeMap<Version, url::Url> =
        std::collections::BTreeMap::new();

    let name_norm_dash = name.replace('_', "-").to_ascii_lowercase();
    let name_norm_underscore = name.replace('-', "_").to_ascii_lowercase();

    for link in all_links {
        let fname_lower = link.filename.to_ascii_lowercase();

        if fname_lower.ends_with(".whl") {
            // Wheel: extract version via the name-prefix strip.
            if let Some(version) =
                extract_version_from_wheel_filename(&fname_lower, &name_prefix_lower)
            {
                wheels_by_version.entry(version).or_default().push(link);
            }
        } else if fname_lower.ends_with(".tar.gz")
            || fname_lower.ends_with(".zip")
            || fname_lower.ends_with(".tar.bz2")
        {
            // Sdist: name-<version>.tar.gz (same logic as resolve_sdist).
            let stem = fname_lower
                .strip_suffix(".tar.gz")
                .or_else(|| fname_lower.strip_suffix(".zip"))
                .or_else(|| fname_lower.strip_suffix(".tar.bz2"));
            if let Some(stem) = stem {
                let rest = stem
                    .strip_prefix(&format!("{name_norm_dash}-"))
                    .or_else(|| stem.strip_prefix(&format!("{name_norm_underscore}-")));
                if let Some(ver_str) = rest
                    && let Ok(version) = Version::from_str(ver_str)
                {
                    // First sdist for this version wins (same as resolve_sdist).
                    sdists_by_version.entry(version).or_insert(link.url);
                }
            }
        }
    }

    // ── Assemble candidates per version ─────────────────────────────────────

    // Collect all versions mentioned by either wheels or sdists.
    let mut all_versions: std::collections::BTreeSet<Version> = std::collections::BTreeSet::new();
    all_versions.extend(wheels_by_version.keys().cloned());
    all_versions.extend(sdists_by_version.keys().cloned());

    let mut candidates: Vec<PyPiCandidate> = Vec::new();

    for version in all_versions {
        let sdist_url = sdists_by_version.get(&version).cloned();

        let best_wheel = wheels_by_version
            .remove(&version)
            .and_then(|group| pick_best(group, target));

        // Include this version only if it has a compatible wheel OR an sdist.
        // Versions whose wheels are all platform-incompatible AND have no sdist
        // are silently dropped (unreachable from this target).
        if best_wheel.is_some() || sdist_url.is_some() {
            candidates.push(PyPiCandidate {
                version,
                wheel: best_wheel,
                sdist_url,
            });
        }
    }

    // Sort highest version first (matches pypi::resolve sort order and the
    // resolvo provider's preferred-highest-first strategy).
    candidates.sort_by(|a, b| b.version.cmp(&a.version));

    Ok(candidates)
}

/// Extract the PEP 440 version from a wheel filename given the pre-computed
/// lowercase name prefix (e.g. `"requests-"`).
///
/// Returns `None` if the filename does not start with the prefix or the version
/// segment cannot be parsed.  This is factored out of `resolve()` so both
/// functions use identical extraction logic.
fn extract_version_from_wheel_filename(
    filename_lower: &str,
    name_prefix_lower: &str,
) -> Option<Version> {
    let rest = filename_lower.strip_prefix(name_prefix_lower)?;
    let version_str = rest.split('-').next()?;
    Version::from_str(version_str).ok()
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
            has_metadata_sidecar: false,
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
    // doesn't require it. Match the whole `<a ...>` tag so the PEP 658/714
    // metadata-sidecar attribute can be read alongside the href; treat the
    // hash as optional so non-conforming indexes (py.mujoco.org, some
    // self-hosted simple repos) still work.
    let re = RE.get_or_init(|| Regex::new(r#"<a\s+[^>]*href="([^"]+)"[^>]*>"#).unwrap());

    let mut out = Vec::new();
    for cap in re.captures_iter(html) {
        let full_tag = &cap[0];
        let href = &cap[1];
        let url = match base.join(href) {
            Ok(u) => u,
            Err(_) => continue,
        };
        // PEP 658 attribute, or its PEP 714 rename. Value is either a
        // hash or "true"; presence is what matters.
        let has_metadata_sidecar = full_tag.contains("data-core-metadata=")
            || full_tag.contains("data-dist-info-metadata=");
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
            has_metadata_sidecar,
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
    fn parses_pep658_metadata_sidecar_attribute() {
        // pypi.org serves both the PEP 658 attribute and its PEP 714
        // rename on the same tag; pypi.nvidia.com serves neither
        // (measured 2026-06-10). Detection keys on attribute presence.
        let base: url::Url = "https://example.com/simple/foo/".parse().unwrap();
        let html = r#"
            <a href="foo-1.0-py3-none-any.whl#sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" data-dist-info-metadata="sha256=bbbb" data-core-metadata="sha256=bbbb">old+new</a>
            <a href="foo-2.0-py3-none-any.whl#sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc" data-core-metadata="true">new only</a>
            <a href="foo-3.0-py3-none-any.whl#sha256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd">bare</a>
        "#;
        let links = parse_index_links(html, &base).unwrap();
        assert_eq!(links.len(), 3);
        assert!(links[0].has_metadata_sidecar, "PEP 658 spelling");
        assert!(links[1].has_metadata_sidecar, "PEP 714 spelling");
        assert!(!links[2].has_metadata_sidecar, "no attribute -> no sidecar");
        assert!(
            links[0].sha256.is_some(),
            "wheel hash still parsed from fragment"
        );
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
            has_metadata_sidecar: false,
        }
    }

    // ── PR-1a: list_all_versions tests (localhost PEP 503 fixture server) ────
    //
    // These tests mirror the fixture-server pattern from
    // `handler::resolve_bundle_bfs_tests` (mod.rs).  A minimal TCP listener
    // serves a PEP 503 simple index page and raw bytes on demand.

    /// Spawn a minimal PEP 503 simple-index server for testing.
    ///
    /// `entries`: list of `(filename, bytes)`.  The server serves:
    ///   GET /simple/<pep503-normalized-name>/  -> HTML listing all
    ///     entries whose filename matches that name prefix.
    ///   GET /<filename>  -> raw bytes for that file.
    ///
    /// Returns the bound port.  The server accepts `max_requests` connections
    /// then stops.  Each connection is handled in its own tokio task so
    /// concurrent fetches (index + wheel) are supported.
    async fn spawn_fixture_server(entries: Vec<(String, Vec<u8>)>, max_requests: u8) -> u16 {
        use std::collections::HashMap;
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let by_filename: Arc<HashMap<String, Vec<u8>>> = Arc::new(
            entries
                .iter()
                .map(|(name, bytes)| (name.clone(), bytes.clone()))
                .collect(),
        );
        let all_filenames: Arc<Vec<String>> =
            Arc::new(entries.into_iter().map(|(name, _)| name).collect());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            for _ in 0..max_requests {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let by_filename = by_filename.clone();
                let all_filenames = all_filenames.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let path = req
                        .lines()
                        .next()
                        .and_then(|l| l.split_whitespace().nth(1))
                        .unwrap_or("/");

                    let (status, content_type, body): (&str, &str, Vec<u8>) = if let Some(rest) =
                        path.strip_prefix("/simple/")
                    {
                        // Build a simple index page for the requested name.
                        let pkg_name = rest.trim_end_matches('/');
                        // PEP 503 normalize: lowercase, collapse -/_/. -> -
                        let pkg_norm: String = {
                            let mut out = String::new();
                            let mut prev = false;
                            for c in pkg_name.chars().flat_map(|c| c.to_lowercase()) {
                                if c == '-' || c == '_' || c == '.' {
                                    if !prev {
                                        out.push('-');
                                        prev = true;
                                    }
                                } else {
                                    out.push(c);
                                    prev = false;
                                }
                            }
                            out
                        };
                        // Match files whose name-prefix normalizes to pkg_norm.
                        let links: String = all_filenames
                            .iter()
                            .filter(|fname| {
                                let fname_lower = fname.to_ascii_lowercase();
                                // Wheel: <dist_norm>-<ver>-...whl
                                // Sdist: <name_norm>-<ver>.tar.gz etc.
                                // Match any file whose normalized name prefix matches.
                                let prefix = format!("{pkg_norm}-");
                                let fname_norm = fname_lower.replace('_', "-");
                                fname_norm.starts_with(&prefix)
                            })
                            .map(|fname| {
                                // Include a fake sha256 for wheel files so
                                // the sha256-parsing test can verify it.
                                let hash_frag = if fname.ends_with(".whl") {
                                    format!("#sha256={}", "a".repeat(64))
                                } else {
                                    String::new()
                                };
                                format!("<a href=\"/{fname}{hash_frag}\">{fname}</a>\n")
                            })
                            .collect();
                        let html = format!("<!DOCTYPE html><html><body>\n{links}</body></html>\n");
                        ("200 OK", "text/html", html.into_bytes())
                    } else {
                        // File request.
                        let fname = path.trim_start_matches('/');
                        // Strip hash fragment from the path if present.
                        let fname = fname.split('#').next().unwrap_or(fname);
                        if let Some(bytes) = by_filename.get(fname) {
                            ("200 OK", "application/octet-stream", bytes.clone())
                        } else {
                            ("404 Not Found", "text/plain", b"not found".to_vec())
                        }
                    };

                    let resp = format!(
                        "HTTP/1.0 {status}\r\nContent-Length: {}\r\nContent-Type: \
                         {content_type}\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(resp.as_bytes()).await;
                    let _ = stream.write_all(&body).await;
                });
            }
        });

        port
    }

    /// Three wheel versions (1.0, 2.0, 3.0), all compatible with linux-64 /
    /// py3.11.  `list_all_versions` must return all three, sorted 3.0 > 2.0 >
    /// 1.0.  Each candidate must carry the correct version and wheel URL.
    #[tokio::test]
    async fn list_all_versions_returns_all_three_wheel_versions() {
        let entries = vec![
            (
                "mylib-1.0-py3-none-any.whl".to_string(),
                b"wheel10".to_vec(),
            ),
            (
                "mylib-2.0-py3-none-any.whl".to_string(),
                b"wheel20".to_vec(),
            ),
            (
                "mylib-3.0-py3-none-any.whl".to_string(),
                b"wheel30".to_vec(),
            ),
        ];
        let port = spawn_fixture_server(entries, 8).await;
        let index = format!("http://127.0.0.1:{port}/simple/");
        let target = WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "linux-64".into(),
        };

        let candidates = list_all_versions(&index, "mylib", &target)
            .await
            .expect("list_all_versions must succeed");

        assert_eq!(
            candidates.len(),
            3,
            "must return all 3 versions; got {candidates:?}"
        );

        // Verify descending sort.
        let versions: Vec<String> = candidates.iter().map(|c| c.version.to_string()).collect();
        assert_eq!(
            versions,
            vec!["3.0", "2.0", "1.0"],
            "must be sorted highest first"
        );

        // All three must be wheel candidates (not sdist-only).
        for c in &candidates {
            assert!(
                c.wheel.is_some(),
                "version {} must have a compatible wheel",
                c.version
            );
            assert!(
                !c.is_sdist_only(),
                "version {} must not be sdist-only",
                c.version
            );
        }

        // Wheel URLs must reference the correct filenames.
        let filenames: Vec<String> = candidates
            .iter()
            .map(|c| c.wheel.as_ref().unwrap().filename.clone())
            .collect();
        assert_eq!(
            filenames,
            vec![
                "mylib-3.0-py3-none-any.whl",
                "mylib-2.0-py3-none-any.whl",
                "mylib-1.0-py3-none-any.whl",
            ]
        );
    }

    /// A version that has NO compatible wheel (wrong python tag: cp310 when
    /// target is 3.11) but an sdist.  That version must be included as
    /// sdist-only.  The compatible wheel version (2.0, py3-none-any) must be
    /// included as a wheel candidate.
    #[tokio::test]
    async fn list_all_versions_sdist_only_when_no_compatible_wheel() {
        let entries = vec![
            // Version 1.0: only a cp310 wheel (incompatible with py3.11) + sdist.
            (
                "mylib-1.0-cp310-cp310-manylinux_2_17_x86_64.whl".to_string(),
                b"incompatible_wheel".to_vec(),
            ),
            ("mylib-1.0.tar.gz".to_string(), b"sdist_bytes".to_vec()),
            // Version 2.0: compatible wheel.
            (
                "mylib-2.0-py3-none-any.whl".to_string(),
                b"wheel20".to_vec(),
            ),
        ];
        let port = spawn_fixture_server(entries, 12).await;
        let index = format!("http://127.0.0.1:{port}/simple/");
        let target = WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "linux-64".into(),
        };

        let candidates = list_all_versions(&index, "mylib", &target)
            .await
            .expect("list_all_versions must succeed");

        assert_eq!(
            candidates.len(),
            2,
            "must have 2 candidates; got {candidates:?}"
        );

        // 2.0 first (highest).
        let v2 = &candidates[0];
        assert_eq!(v2.version.to_string(), "2.0");
        assert!(v2.wheel.is_some(), "2.0 must have a compatible wheel");
        assert!(!v2.is_sdist_only());

        // 1.0 second: no compatible wheel, sdist-only.
        let v1 = &candidates[1];
        assert_eq!(v1.version.to_string(), "1.0");
        assert!(
            v1.wheel.is_none(),
            "1.0 must have no compatible wheel (cp310 != target 3.11)"
        );
        assert!(v1.sdist_url.is_some(), "1.0 must have an sdist url");
        assert!(v1.is_sdist_only(), "1.0 must be sdist-only");
    }

    /// A version whose wheel has an incompatible platform tag (arm64 when
    /// target is linux-64/x86_64) and no sdist must be EXCLUDED entirely.
    /// A compatible version (py3-none-any) must still be returned.
    #[tokio::test]
    async fn list_all_versions_excludes_incompatible_wheel_version_without_sdist() {
        let entries = vec![
            // Version 1.0: wrong arch wheel only, no sdist.
            (
                "mylib-1.0-cp311-cp311-manylinux_2_17_aarch64.whl".to_string(),
                b"arm_wheel".to_vec(),
            ),
            // Version 2.0: universally compatible.
            (
                "mylib-2.0-py3-none-any.whl".to_string(),
                b"wheel20".to_vec(),
            ),
        ];
        let port = spawn_fixture_server(entries, 8).await;
        let index = format!("http://127.0.0.1:{port}/simple/");
        let target = WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "linux-64".into(),
        };

        let candidates = list_all_versions(&index, "mylib", &target)
            .await
            .expect("list_all_versions must succeed");

        // Only version 2.0 must be returned; 1.0 is unreachable (wrong arch,
        // no sdist fallback).
        assert_eq!(
            candidates.len(),
            1,
            "only 1.0 with wrong arch + no sdist must be excluded; got {candidates:?}"
        );
        assert_eq!(candidates[0].version.to_string(), "2.0");
    }

    // ── favor-lock: resolve_preferring tests ─────────────────────────────────

    /// When the preferred version is on the index and a compatible wheel exists,
    /// `resolve_preferring` must return that version rather than the latest.
    #[tokio::test]
    async fn resolve_preferring_returns_preferred_not_latest() {
        let entries = vec![
            ("mylib-1.0-py3-none-any.whl".to_string(), b"v10".to_vec()),
            ("mylib-2.0-py3-none-any.whl".to_string(), b"v20".to_vec()),
            ("mylib-3.0-py3-none-any.whl".to_string(), b"v30".to_vec()),
        ];
        let port = spawn_fixture_server(entries, 4).await;
        let index = format!("http://127.0.0.1:{port}/simple/");
        let target = WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "linux-64".into(),
        };
        let specs: VersionSpecifiers = ">=1.0".parse().unwrap();

        let picked = resolve_preferring(&index, "mylib", &specs, &target, "2.0")
            .await
            .expect("resolve_preferring must succeed");

        assert_eq!(
            picked.filename, "mylib-2.0-py3-none-any.whl",
            "must return the preferred version 2.0, not the latest 3.0"
        );
    }

    /// When the preferred version is NOT on the index, `resolve_preferring`
    /// falls back to the highest matching version (normal behavior).
    #[tokio::test]
    async fn resolve_preferring_falls_back_when_preferred_absent() {
        let entries = vec![
            ("mylib-1.0-py3-none-any.whl".to_string(), b"v10".to_vec()),
            ("mylib-3.0-py3-none-any.whl".to_string(), b"v30".to_vec()),
        ];
        let port = spawn_fixture_server(entries, 4).await;
        let index = format!("http://127.0.0.1:{port}/simple/");
        let target = WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "linux-64".into(),
        };
        let specs: VersionSpecifiers = ">=1.0".parse().unwrap();

        // Prefer 2.0, which does not exist on this index.
        let picked = resolve_preferring(&index, "mylib", &specs, &target, "2.0")
            .await
            .expect("resolve_preferring must fall back without error");

        assert_eq!(
            picked.filename, "mylib-3.0-py3-none-any.whl",
            "must fall back to the highest matching version (3.0) when preferred 2.0 absent"
        );
    }

    /// When the preferred version does NOT satisfy `specifiers`,
    /// `resolve_preferring` ignores it and returns the highest matching version.
    #[tokio::test]
    async fn resolve_preferring_ignores_preferred_outside_specifiers() {
        let entries = vec![
            ("mylib-1.0-py3-none-any.whl".to_string(), b"v10".to_vec()),
            ("mylib-2.0-py3-none-any.whl".to_string(), b"v20".to_vec()),
        ];
        let port = spawn_fixture_server(entries, 4).await;
        let index = format!("http://127.0.0.1:{port}/simple/");
        let target = WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "linux-64".into(),
        };
        // Specifiers only allow >=2.0, but we prefer 1.0 (outside the range).
        let specs: VersionSpecifiers = ">=2.0".parse().unwrap();

        let picked = resolve_preferring(&index, "mylib", &specs, &target, "1.0")
            .await
            .expect("resolve_preferring must succeed");

        assert_eq!(
            picked.filename, "mylib-2.0-py3-none-any.whl",
            "must ignore preferred 1.0 (outside >=2.0) and return 2.0"
        );
    }

    /// `resolve` (no preferred version) still picks the highest matching version.
    /// This guards the default code path against regressions from the
    /// `resolve_inner` refactor.
    #[tokio::test]
    async fn resolve_without_preference_picks_highest() {
        let entries = vec![
            ("mylib-1.0-py3-none-any.whl".to_string(), b"v10".to_vec()),
            ("mylib-2.0-py3-none-any.whl".to_string(), b"v20".to_vec()),
            ("mylib-3.0-py3-none-any.whl".to_string(), b"v30".to_vec()),
        ];
        let port = spawn_fixture_server(entries, 4).await;
        let index = format!("http://127.0.0.1:{port}/simple/");
        let target = WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "linux-64".into(),
        };
        let specs: VersionSpecifiers = ">=1.0".parse().unwrap();

        let picked = resolve(&index, "mylib", &specs, &target)
            .await
            .expect("resolve must succeed");

        assert_eq!(
            picked.filename, "mylib-3.0-py3-none-any.whl",
            "plain resolve must pick the highest version (3.0)"
        );
    }

    /// The #sha256=<hex> fragment on wheel index links must be parsed into
    /// `wheel.sha256`.  The fixture server appends a fake 64-hex-char sha256
    /// to each wheel's link; verify it round-trips into the candidate.
    #[tokio::test]
    async fn list_all_versions_parses_sha256_fragment() {
        let entries = vec![(
            "mylib-1.0-py3-none-any.whl".to_string(),
            b"wheel_bytes".to_vec(),
        )];
        let port = spawn_fixture_server(entries, 4).await;
        let index = format!("http://127.0.0.1:{port}/simple/");
        let target = WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "linux-64".into(),
        };

        let candidates = list_all_versions(&index, "mylib", &target)
            .await
            .expect("list_all_versions must succeed");

        assert_eq!(candidates.len(), 1);
        let wheel = candidates[0].wheel.as_ref().expect("must have a wheel");
        assert_eq!(
            wheel.sha256.as_deref(),
            // The fixture server appends `#sha256=aaa...aaa` (64 'a' chars).
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            "sha256 from URL fragment must be parsed into the candidate"
        );
    }
}
