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

/// Download a wheel into `dest_dir`. Verifies SHA-256 if `expected_sha256` is
/// provided. Returns the path to the cached file (skips re-download if already
/// present with matching hash).
pub async fn fetch_wheel(
    url: &url::Url,
    expected_sha256: Option<&str>,
    dest_dir: &Path,
) -> Result<PathBuf> {
    fs::create_dir_all(dest_dir).await?;

    let filename = url
        .path_segments()
        .and_then(|s| s.last())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("URL has no filename component: {url}"))?;
    if !filename.ends_with(".whl") {
        bail!("URL does not point to a .whl file: {url}");
    }
    let dest = dest_dir.join(filename);

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

/// Read the METADATA file inside a wheel zip and parse out the fields we care
/// about.
pub fn read_metadata(wheel_path: &Path) -> Result<WheelMetadata> {
    let filename = wheel_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("wheel path has no filename: {}", wheel_path.display()))?
        .to_string();
    let is_pure_python = filename.contains("-none-any.whl");

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
}
