//! Wheel download and METADATA parsing.
//!
//! METADATA inside a wheel is RFC 822-style headers (PEP 241/345/566). We
//! extract `Name`, `Version`, and every `Requires-Dist:` value. Requirement
//! strings are kept as PEP 508 text and parsed downstream by the relax pass.

use std::fmt::Write as _;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use sha2::{Digest, Sha256};
use tokio::fs;

#[derive(Debug, Clone)]
pub struct WheelMetadata {
    pub name: String,
    pub version: String,
    /// Raw `Requires-Dist:` values, one per line. PEP 508 syntax.
    pub requires_dist: Vec<String>,
    /// Whether the wheel's tag set includes `none-any` (pure-Python). Used to
    /// emit `noarch: python` in the generated recipe.
    pub is_pure_python: bool,
    /// Computed SHA-256 of the downloaded wheel.
    pub sha256: String,
    /// The wheel filename (e.g. `isaacsim-5.1.0-cp311-none-manylinux_2_35_x86_64.whl`).
    pub filename: String,
}

/// Derive the on-disk filename for a wheel URL. Percent-decodes the last
/// path segment so URLs that encode the `+` of a PEP 440 local-version
/// identifier (e.g. miropsota's
/// `pytorch3d-0.7.8%2B5043d15pt2.7.0cu128-...whl`) land on disk with the
/// canonical PEP 427 spelling. Without that decode, pip rejects the file
/// at install time with `Invalid wheel filename (invalid version)`
/// because `%2B` is not a valid PEP 440 character.
pub fn wheel_filename_from_url(url: &url::Url) -> Result<String> {
    let raw = url
        .path_segments()
        .and_then(|s| s.last())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("URL has no filename component: {url}"))?;
    let decoded = percent_encoding::percent_decode_str(raw)
        .decode_utf8()
        .with_context(|| format!("URL filename is not valid UTF-8: {raw}"))?;
    if !decoded.ends_with(".whl") {
        bail!("URL does not point to a .whl file: {url}");
    }
    Ok(decoded.into_owned())
}

/// Download a wheel into `dest_dir`. Verifies SHA-256 if `expected_sha256` is
/// provided. Returns the path to the cached file (skips re-download if already
/// present with matching hash).
pub async fn fetch_wheel(
    url: &url::Url,
    expected_sha256: Option<&str>,
    dest_dir: &Path,
) -> Result<PathBuf> {
    fs::create_dir_all(dest_dir).await?;

    let filename = wheel_filename_from_url(url)?;
    let dest = dest_dir.join(&filename);

    if dest.exists() {
        if let Some(expected) = expected_sha256 {
            let actual = sha256_file(&dest).await?;
            if actual.eq_ignore_ascii_case(expected) {
                tracing::debug!(path = %dest.display(), "wheel already cached");
                return Ok(dest);
            }
            tracing::warn!(
                path = %dest.display(),
                "cached wheel hash mismatch, re-downloading"
            );
            fs::remove_file(&dest).await.ok();
        } else {
            return Ok(dest);
        }
    }

    tracing::info!(url = %url, "downloading wheel");
    let resp = reqwest::get(url.clone())
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {url}"))?;
    let bytes = resp
        .bytes()
        .await
        .with_context(|| format!("reading body of {url}"))?;

    if let Some(expected) = expected_sha256 {
        let actual = hex_sha256(&bytes);
        if !actual.eq_ignore_ascii_case(expected) {
            bail!(
                "SHA-256 mismatch for {url}: expected {expected}, got {actual}"
            );
        }
    }

    fs::write(&dest, &bytes)
        .await
        .with_context(|| format!("writing {}", dest.display()))?;
    Ok(dest)
}

/// Returns `true` if the wheel filename's PEP 425 platform tag is `any`
/// (i.e. the wheel is pure-Python and runs on every platform).
///
/// PEP 425 wheel filenames are `{name}-{version}(-{build})?-{python}-{abi}-{platform}.whl`,
/// so the platform tag is the LAST hyphen-separated segment of the stem.
/// **Important**: D rewrites the wheel and renames it from `foo-1.0-py3-none-any.whl`
/// to `foo-1.0-py3-none-any.relaxed.whl` (cosmetic suffix so the original wheel
/// stays on disk untouched). A naive `filename.contains("-none-any.whl")` check
/// returns FALSE on the relaxed file -- which used to flip every pure-Python
/// wheel into the platform-specific branch downstream. The consequence: the
/// merged-bundle primary (isaaclab, alphabetically first via BTreeMap) was
/// `py3-none-any`, so the bundle's `python_version` decayed to the bare-major
/// "3" parsed from the `py3` tag (via the wheel-tag fallback in `produce_output`),
/// the conda solver then read `python 3.*` and bound python to 3.14, and the
/// workspace's `python==3.11` pin rejected the implied `python_abi 3.14.* *_cp314`.
///
/// Strip the well-known `.relaxed.whl` suffix first so the canonical PEP 425
/// suffix `.whl` is restored, then inspect the platform tag. Any future
/// rewrite suffix needs to be added here in lock-step.
pub fn is_pure_python_wheel_filename(filename: &str) -> bool {
    let canonical = filename
        .strip_suffix(".relaxed.whl")
        .map(|stem| format!("{stem}.whl"))
        .unwrap_or_else(|| filename.to_string());
    let Some(stem) = canonical.strip_suffix(".whl") else {
        return false;
    };
    // Platform tag is the LAST hyphen segment. `any` => pure-Python.
    stem.rsplit('-').next() == Some("any")
}

/// Read the METADATA file inside a wheel zip and parse out the fields we care
/// about.
pub fn read_metadata(wheel_path: &Path) -> Result<WheelMetadata> {
    let filename = wheel_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("wheel path has no filename: {}", wheel_path.display()))?
        .to_string();
    let is_pure_python = is_pure_python_wheel_filename(&filename);

    let sha256 = hex_sha256(&std::fs::read(wheel_path)?);

    let file = std::fs::File::open(wheel_path)
        .with_context(|| format!("opening {}", wheel_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("reading zip {}", wheel_path.display()))?;

    // The wheel's own METADATA is at `<name>-<version>.dist-info/METADATA` at
    // the zip root. Wheels may vendor other packages with their own nested
    // .dist-info trees (isaacsim does this); only the root-level entry is
    // ours.
    let mut metadata_idx = None;
    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        let name = entry.name();
        if name.ends_with(".dist-info/METADATA") && name.matches('/').count() == 1 {
            metadata_idx = Some(i);
            break;
        }
    }
    let idx = metadata_idx.ok_or_else(|| {
        anyhow!("no root-level .dist-info/METADATA in {}", wheel_path.display())
    })?;

    let mut entry = archive.by_index(idx)?;
    let mut buf = String::new();
    entry.read_to_string(&mut buf)?;

    parse_metadata(&buf, filename, is_pure_python, sha256)
}

/// Parse a wheel's METADATA file content into the fields we care about.
/// Exposed so integration tests can drive the relax pipeline from captured
/// METADATA fixtures without needing a real wheel on disk.
pub fn parse_metadata(
    raw: &str,
    filename: String,
    is_pure_python: bool,
    sha256: String,
) -> Result<WheelMetadata> {
    let mut name = None;
    let mut version = None;
    let mut requires_dist = Vec::new();

    // RFC 822-style headers terminate at the first blank line.
    // Continuation lines start with whitespace; we don't currently need them
    // since Name/Version/Requires-Dist are single-line in every wheel I've seen.
    for line in raw.lines() {
        if line.is_empty() {
            break;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "Name" => name = Some(value.to_string()),
            "Version" => version = Some(value.to_string()),
            "Requires-Dist" => requires_dist.push(value.to_string()),
            _ => {}
        }
    }

    Ok(WheelMetadata {
        name: name.ok_or_else(|| anyhow!("METADATA missing Name"))?,
        version: version.ok_or_else(|| anyhow!("METADATA missing Version"))?,
        requires_dist,
        is_pure_python,
        sha256,
        filename,
    })
}

async fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).await?;
    Ok(hex_sha256(&bytes))
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        write!(&mut out, "{b:02x}").expect("write to String");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_metadata() {
        let raw = "Metadata-Version: 2.1\n\
                   Name: example-pkg\n\
                   Version: 1.2.3\n\
                   Requires-Dist: numpy==1.26.4\n\
                   Requires-Dist: torch>=2.7\n\
                   \n\
                   Some description.\n";
        let m = parse_metadata(raw, "example_pkg-1.2.3-py3-none-any.whl".into(), true, "abc".into()).unwrap();
        assert_eq!(m.name, "example-pkg");
        assert_eq!(m.version, "1.2.3");
        assert_eq!(m.requires_dist, vec!["numpy==1.26.4", "torch>=2.7"]);
        assert!(m.is_pure_python);
    }

    // Regression: a pure-Python wheel after D rewrite has filename
    // `*.relaxed.whl` (not `*.whl`). The old `filename.contains("-none-any.whl")`
    // check returned false on the relaxed file, which flipped every pure-Python
    // wheel into the platform-specific branch downstream. With the merged
    // bundle's alphabetically-first primary being `isaaclab` (`py3-none-any`),
    // the bundle's python_version then decayed to `"3"` from the `py3` tag,
    // emitting `python 3.*` as the conda run-dep; the solver bound python to
    // 3.14 and the workspace's `python==3.11` rejected the implied python_abi.
    // Detect platform tag = `any` semantically, not via a brittle filename
    // substring.
    #[test]
    fn detects_pure_python_through_relaxed_suffix() {
        // Plain pure-Python wheel: pure.
        assert!(is_pure_python_wheel_filename("isaaclab-0.51.1-py3-none-any.whl"));
        // Plain platform-specific wheel: not pure.
        assert!(!is_pure_python_wheel_filename(
            "isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl"
        ));
        // Pure-Python wheel after D rewrite (`.relaxed.whl` suffix): still pure.
        assert!(is_pure_python_wheel_filename(
            "isaaclab-0.51.1-py3-none-any.relaxed.whl"
        ));
        // Platform-specific wheel after D rewrite: still platform-specific.
        assert!(!is_pure_python_wheel_filename(
            "isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.relaxed.whl"
        ));
        // py2.py3-none-any (universal) wheel: pure.
        assert!(is_pure_python_wheel_filename("six-1.16.0-py2.py3-none-any.whl"));
        // Not a wheel at all: false.
        assert!(!is_pure_python_wheel_filename("foo.tar.gz"));
    }

    // Defensive: read_metadata's `is_pure_python` flag uses the same helper,
    // so the field on WheelMetadata stays correct through the rewrite pipeline.
    // Other callers (`produce_output`, `build_bundle_recipe`) read this flag
    // to decide python pinning and noarch emission, so the helper IS the
    // canonical source of truth.
    // Regression: pytorch3d's miropsota GitHub-Release index serves
    // `pytorch3d-0.7.8%2B5043d15pt2.7.0cu128-cp311-cp311-linux_x86_64.whl`
    // with `%2B` URL-encoding the `+` of the PEP 440 local-version
    // identifier. fetch_wheel used to keep the encoded form in the
    // on-disk name, and pip then rejected the file with
    // `Invalid wheel filename (invalid version)` because `%2B` isn't a
    // valid PEP 440 character. The decoded form has a valid local id.
    #[test]
    fn wheel_filename_decodes_percent_encoded_plus() {
        let url: url::Url = "https://example.com/pytorch3d-0.7.8%2B5043d15pt2.7.0cu128-cp311-cp311-linux_x86_64.whl"
            .parse()
            .unwrap();
        let name = wheel_filename_from_url(&url).unwrap();
        assert_eq!(
            name,
            "pytorch3d-0.7.8+5043d15pt2.7.0cu128-cp311-cp311-linux_x86_64.whl",
            "wheel_filename_from_url must decode `%2B` to `+`",
        );
    }

    #[test]
    fn wheel_filename_passes_through_unencoded() {
        let url: url::Url = "https://pypi.nvidia.com/isaacsim/isaacsim-5.1.0-cp311-none-manylinux_2_35_x86_64.whl"
            .parse()
            .unwrap();
        let name = wheel_filename_from_url(&url).unwrap();
        assert_eq!(
            name,
            "isaacsim-5.1.0-cp311-none-manylinux_2_35_x86_64.whl",
        );
    }

    #[test]
    fn wheel_filename_rejects_non_whl() {
        let url: url::Url = "https://example.com/foo-1.0.tar.gz".parse().unwrap();
        assert!(wheel_filename_from_url(&url).is_err());
    }

    #[test]
    fn parse_metadata_carries_is_pure_python_for_relaxed_wheel() {
        // Caller passes the helper's verdict; this test just locks that the
        // wired-through flag reaches the WheelMetadata struct unchanged.
        let raw = "Metadata-Version: 2.1\nName: isaaclab\nVersion: 0.51.1\n\n";
        let m = parse_metadata(
            raw,
            "isaaclab-0.51.1-py3-none-any.relaxed.whl".into(),
            is_pure_python_wheel_filename("isaaclab-0.51.1-py3-none-any.relaxed.whl"),
            "sha".into(),
        )
        .unwrap();
        assert!(m.is_pure_python, "relaxed pure-Python wheels must remain marked pure");
    }
}
