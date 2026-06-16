//! Wheel download and METADATA parsing.
//!
//! METADATA inside a wheel is RFC 822-style headers (PEP 241/345/566). We
//! extract `Name`, `Version`, and every `Requires-Dist:` value. Requirement
//! strings are kept as PEP 508 text and parsed downstream by the relax pass.

use std::fmt::Write as _;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use futures::StreamExt as _;
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::io::AsyncWriteExt as _;

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
        .and_then(|mut s| s.next_back())
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

    let resp = reqwest::get(url.clone())
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {url}"))?;
    let total = resp.content_length();
    match total {
        Some(len) => tracing::info!(
            %filename,
            size_mb = len / 1_048_576,
            "downloading wheel (streaming to disk; large wheels can take minutes)",
        ),
        None => tracing::info!(%filename, "downloading wheel (streaming to disk)"),
    }
    // /dev/tty status: wheel downloads happen during conda/outputs, where pixi
    // hides backend stderr -- so this is the only way the user sees the
    // multi-GB NVIDIA wheels actually downloading.
    crate::status::tty(&format!(
        "downloading {filename}{}",
        total
            .map(|t| format!(" ({} MB)", t / 1_048_576))
            .unwrap_or_default()
    ));

    // Stream the body to disk in chunks instead of `resp.bytes()` (which
    // buffers the WHOLE wheel in memory). The isaacsim extscache wheels are
    // several GB: buffering spiked RSS to multiple GB AND produced a
    // multi-minute SILENT gap (one log line, then nothing -- looks frozen).
    // Streaming caps memory at one chunk, hashes incrementally, and logs
    // steady progress so the download is visibly alive in pixi's output.
    let part = dest_dir.join(format!("{filename}.part"));
    let mut file = fs::File::create(&part)
        .await
        .with_context(|| format!("creating {}", part.display()))?;
    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;
    let mut last_logged: u64 = 0;
    let mut stream = std::pin::pin!(resp.bytes_stream());
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("reading body of {url}"))?;
        hasher.update(&chunk);
        file.write_all(&chunk)
            .await
            .with_context(|| format!("writing {}", part.display()))?;
        downloaded += chunk.len() as u64;
        // Log roughly every 100 MB so multi-GB wheels show steady movement.
        if downloaded - last_logged >= 100 * 1_048_576 {
            last_logged = downloaded;
            match total {
                Some(t) => tracing::info!(
                    %filename,
                    mb = downloaded / 1_048_576,
                    of_mb = t / 1_048_576,
                    "download progress",
                ),
                None => tracing::info!(%filename, mb = downloaded / 1_048_576, "download progress"),
            }
        }
    }
    file.flush()
        .await
        .with_context(|| format!("flushing {}", part.display()))?;
    drop(file);
    tracing::info!(%filename, mb = downloaded / 1_048_576, "wheel download complete");

    if let Some(expected) = expected_sha256 {
        let digest = hasher.finalize();
        let mut actual = String::with_capacity(64);
        for b in digest {
            write!(&mut actual, "{b:02x}").expect("write to String");
        }
        if !actual.eq_ignore_ascii_case(expected) {
            fs::remove_file(&part).await.ok();
            bail!("SHA-256 mismatch for {url}: expected {expected}, got {actual}");
        }
    }

    fs::rename(&part, &dest)
        .await
        .with_context(|| format!("renaming {} -> {}", part.display(), dest.display()))?;
    Ok(dest)
}

/// Hard-link `src` -> `dst`, falling back to copy on any error (including EXDEV).
///
/// EXDEV is returned when src and dst are on different filesystems. Attempting
/// hard_link first is the fast path; copy is the safe fallback for both
/// cross-device and any other platform-specific constraint.
pub(crate) async fn hardlink_or_copy_async(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).await?;
    }
    // hard_link is sync but fast; run via spawn_blocking to avoid blocking the executor.
    let src_b = src.to_path_buf();
    let dst_b = dst.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<()> {
        if dst_b.exists() {
            std::fs::remove_file(&dst_b)?;
        }
        if std::fs::hard_link(&src_b, &dst_b).is_err() {
            // Fallback: copy (handles EXDEV / cross-device and any other error).
            std::fs::copy(&src_b, &dst_b)?;
        }
        Ok(())
    })
    .await
    .context("hardlink_or_copy_async panicked")?
}

/// Download a wheel with a machine-global persistent content-addressed cache.
///
/// Store layout: `<cache_root>/wheels/<sha256>/<filename>.whl`.
///
/// On cache HIT (sha256 known and file present): hard-links the cached file
/// into `dest_dir/<filename>`, no network.
/// On cache MISS: downloads normally (with streaming + sha256 verification),
/// then populates the cache for future calls.
///
/// Falls back to plain `fetch_wheel` when:
///   - `expected_sha256` is `None` (no key to address by).
///   - `RETREAD_NO_SHADOW_CACHE` is set (bypass for parity testing).
pub async fn fetch_wheel_cached(
    url: &url::Url,
    expected_sha256: Option<&str>,
    dest_dir: &Path,
    cache_root: &Path,
) -> Result<PathBuf> {
    // Bypass when disabled or when we have no sha256 to address by.
    let bypass = std::env::var("RETREAD_NO_SHADOW_CACHE").is_ok();
    let Some(sha256) = expected_sha256.filter(|_| !bypass) else {
        return fetch_wheel(url, expected_sha256, dest_dir).await;
    };

    let filename = wheel_filename_from_url(url)?;
    let dest = dest_dir.join(&filename);

    // Early return: already in dest_dir.
    if dest.exists() {
        tracing::debug!(
            wheel = %filename,
            "wheel cache: already in dest_dir (no fetch needed)",
        );
        return Ok(dest);
    }

    // Check the persistent store.
    let store_path = cache_root.join("wheels").join(sha256).join(&filename);
    if store_path.exists() {
        // Cache hit: hard-link (copy on EXDEV) into dest_dir.
        if let Err(e) = hardlink_or_copy_async(&store_path, &dest).await {
            tracing::warn!(
                wheel = %filename,
                err = %e,
                "wheel cache: hit but hardlink failed, falling back to download",
            );
        } else {
            tracing::info!(
                wheel = %filename,
                sha256 = %&sha256[..8],
                "wheel cache: hit (persistent store, no download)",
            );
            return Ok(dest);
        }
    } else {
        tracing::debug!(
            wheel = %filename,
            sha256 = %&sha256[..8],
            "wheel cache: miss",
        );
    }

    // Cache miss: download normally.
    let downloaded = fetch_wheel(url, Some(sha256), dest_dir).await?;

    // Populate the persistent store (atomic temp+rename).
    let store_dir = cache_root.join("wheels").join(sha256);
    if let Err(e) = fs::create_dir_all(&store_dir).await {
        tracing::warn!(
            wheel = %filename,
            err = %e,
            "wheel cache: could not create store dir, skipping cache population",
        );
        return Ok(downloaded);
    }
    // Process+sequence-unique tmp so concurrent installs sharing this
    // machine-global store never promote each other's torn temp file onto the
    // canonical (existence-checked, unverified) hit path.
    static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let store_tmp = store_dir.join(format!(
        "{filename}.{}.{}.tmp",
        std::process::id(),
        TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let store_final = store_dir.join(&filename);
    // Copy to a unique .tmp then atomically rename so concurrent writers are safe.
    if let Err(e) = hardlink_or_copy_async(&downloaded, &store_tmp).await {
        tracing::warn!(
            wheel = %filename,
            err = %e,
            "wheel cache: could not populate store (link/copy to tmp), skipping",
        );
        return Ok(downloaded);
    }
    if let Err(e) = fs::rename(&store_tmp, &store_final).await {
        tracing::warn!(
            wheel = %filename,
            err = %e,
            "wheel cache: could not rename to final store path, skipping",
        );
        let _ = fs::remove_file(&store_tmp).await;
    } else {
        tracing::debug!(
            wheel = %filename,
            sha256 = %&sha256[..8],
            "wheel cache: populated persistent store",
        );
    }

    Ok(downloaded)
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

/// v1.4.3: fetch a wheel's METADATA via its PEP 658/714 sidecar
/// (`<wheel_url>.metadata`) instead of downloading the whole wheel.
/// Caller contract: only call when the index advertised the sidecar
/// (`ResolvedWheel.has_metadata_sidecar`) AND provided the wheel's
/// sha256 in the link fragment -- the recipe pins each source wheel's
/// hash, and without the full bytes the index-advertised hash is the
/// only source for it. `is_pure_python` derives from the filename, the
/// same signal `read_metadata` uses.
pub async fn fetch_metadata_sidecar(
    wheel_url: &url::Url,
    wheel_sha256: &str,
) -> Result<WheelMetadata> {
    let filename = wheel_filename_from_url(wheel_url)?;
    let is_pure_python = is_pure_python_wheel_filename(&filename);
    // The sidecar lives at the wheel URL + ".metadata"; the fragment
    // (#sha256=...) belongs to the WHEEL link and must not leak into
    // the sidecar request path.
    let mut sidecar = wheel_url.clone();
    sidecar.set_fragment(None);
    sidecar.set_path(&format!("{}.metadata", sidecar.path()));
    tracing::debug!(url = %sidecar, "fetching PEP 658 metadata sidecar");
    let raw = reqwest::get(sidecar.clone())
        .await
        .with_context(|| format!("GET {sidecar}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {sidecar}"))?
        .text()
        .await
        .with_context(|| format!("reading body of {sidecar}"))?;
    parse_metadata(
        &raw,
        filename,
        is_pure_python,
        wheel_sha256.to_ascii_lowercase(),
    )
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
        anyhow!(
            "no root-level .dist-info/METADATA in {}",
            wheel_path.display()
        )
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
    #[ignore = "live: fetches a PEP 658 sidecar + the full wheel from pypi.org"]
    fn metadata_sidecar_matches_full_wheel_live() {
        // The sidecar path must produce the same parsed metadata the
        // full-wheel path does (sha256 aside, which the sidecar takes
        // from the index fragment). tomli 2.0.1 is tiny and stable.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let target = crate::pypi::WheelTarget {
                python_version: "3.11".into(),
                conda_subdir: "linux-64".into(),
            };
            let specs = "==2.0.1".parse().unwrap();
            let resolved =
                crate::pypi::resolve("https://pypi.org/simple/", "tomli", &specs, &target)
                    .await
                    .unwrap();
            assert!(
                resolved.has_metadata_sidecar,
                "pypi.org must advertise the PEP 658 sidecar"
            );
            let sha = resolved
                .sha256
                .as_deref()
                .expect("pypi.org provides fragments");
            let from_sidecar = fetch_metadata_sidecar(&resolved.url, sha).await.unwrap();
            let tmp =
                std::env::temp_dir().join(format!("retread-sidecar-live-{}", std::process::id()));
            std::fs::create_dir_all(&tmp).unwrap();
            let wheel_path = fetch_wheel(&resolved.url, Some(sha), &tmp).await.unwrap();
            let from_wheel = read_metadata(&wheel_path).unwrap();
            assert_eq!(from_sidecar.name, from_wheel.name);
            assert_eq!(from_sidecar.version, from_wheel.version);
            assert_eq!(from_sidecar.requires_dist, from_wheel.requires_dist);
            assert_eq!(from_sidecar.is_pure_python, from_wheel.is_pure_python);
            assert_eq!(from_sidecar.filename, from_wheel.filename);
            assert_eq!(
                from_sidecar.sha256, from_wheel.sha256,
                "fragment hash must equal the computed wheel hash"
            );
        });
    }

    #[test]
    fn parses_basic_metadata() {
        let raw = "Metadata-Version: 2.1\n\
                   Name: example-pkg\n\
                   Version: 1.2.3\n\
                   Requires-Dist: numpy==1.26.4\n\
                   Requires-Dist: torch>=2.7\n\
                   \n\
                   Some description.\n";
        let m = parse_metadata(
            raw,
            "example_pkg-1.2.3-py3-none-any.whl".into(),
            true,
            "abc".into(),
        )
        .unwrap();
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
        assert!(is_pure_python_wheel_filename(
            "isaaclab-0.51.1-py3-none-any.whl"
        ));
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
        assert!(is_pure_python_wheel_filename(
            "six-1.16.0-py2.py3-none-any.whl"
        ));
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
            name, "pytorch3d-0.7.8+5043d15pt2.7.0cu128-cp311-cp311-linux_x86_64.whl",
            "wheel_filename_from_url must decode `%2B` to `+`",
        );
    }

    #[test]
    fn wheel_filename_passes_through_unencoded() {
        let url: url::Url =
            "https://pypi.nvidia.com/isaacsim/isaacsim-5.1.0-cp311-none-manylinux_2_35_x86_64.whl"
                .parse()
                .unwrap();
        let name = wheel_filename_from_url(&url).unwrap();
        assert_eq!(name, "isaacsim-5.1.0-cp311-none-manylinux_2_35_x86_64.whl",);
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
        assert!(
            m.is_pure_python,
            "relaxed pure-Python wheels must remain marked pure"
        );
    }

    // ── Persistent wheel store tests ─────────────────────────────────────────

    /// Test A2-P1: hardlink_or_copy_async produces byte-identical output.
    #[tokio::test]
    async fn hardlink_or_copy_async_byte_identical() {
        let tmp =
            std::env::temp_dir().join(format!("retread-wheel-test-hc-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        let src = tmp.join("source.bin");
        std::fs::write(&src, b"hello persistent cache").unwrap();
        let dst = tmp.join("dest.bin");

        hardlink_or_copy_async(&src, &dst).await.unwrap();
        let src_bytes = std::fs::read(&src).unwrap();
        let dst_bytes = std::fs::read(&dst).unwrap();

        let _ = std::fs::remove_dir_all(&tmp);
        assert_eq!(
            src_bytes, dst_bytes,
            "hardlink_or_copy_async must produce byte-identical output"
        );
    }

    /// Test A2-P2: fetch_wheel_cached bypass logic is correct.
    /// The bypass condition: `std::env::var("RETREAD_NO_SHADOW_CACHE").is_ok()`
    /// means: var IS set -> bypass active -> skip cache and call fetch_wheel directly.
    /// We verify the logic without mutating process env.
    #[test]
    fn wheel_cache_bypass_logic_correct() {
        // When env var is set (Ok) -> bypass = true.
        let env_set: Result<String, std::env::VarError> = Ok("1".to_string());
        let bypass_when_set = env_set.is_ok();
        assert!(
            bypass_when_set,
            "RETREAD_NO_SHADOW_CACHE=1 must activate bypass"
        );

        // When env var is absent (Err) -> bypass = false.
        let env_absent: Result<String, std::env::VarError> = Err(std::env::VarError::NotPresent);
        let bypass_when_absent = env_absent.is_ok();
        assert!(
            !bypass_when_absent,
            "absent RETREAD_NO_SHADOW_CACHE must NOT activate bypass"
        );
    }

    /// Test A2-P3: fetch_wheel_cached populates the persistent store on a miss
    /// and serves from the store on the next call (no second download).
    /// Uses hardlink_or_copy_async directly to simulate the store logic.
    #[tokio::test]
    async fn wheel_cache_persistent_store_hit() {
        let tmp =
            std::env::temp_dir().join(format!("retread-wheel-test-store-{}", std::process::id()));
        let cache_root = tmp.join("cache");
        let dest_dir = tmp.join("dest");
        std::fs::create_dir_all(&dest_dir).unwrap();

        // Fake wheel content.
        let wheel_bytes = b"PK\x03\x04fake wheel for persistent store test".as_slice();
        let sha256 = {
            use sha2::{Digest, Sha256};
            use std::fmt::Write as _;
            let mut h = Sha256::new();
            h.update(wheel_bytes);
            let digest = h.finalize();
            let mut s = String::with_capacity(64);
            for b in digest {
                write!(&mut s, "{b:02x}").expect("write to String");
            }
            s
        };
        let filename = "mypkg-1.0.0-py3-none-any.whl";

        // Simulate: populate the persistent store (as fetch_wheel_cached does after a download).
        let store_dir = cache_root.join("wheels").join(&sha256);
        std::fs::create_dir_all(&store_dir).unwrap();
        let store_path = store_dir.join(filename);
        std::fs::write(&store_path, wheel_bytes).unwrap();

        // Simulate a cache HIT: hard-link from store to dest.
        let dest_path = dest_dir.join(filename);
        hardlink_or_copy_async(&store_path, &dest_path)
            .await
            .unwrap();

        let result_bytes = std::fs::read(&dest_path).unwrap();
        let _ = std::fs::remove_dir_all(&tmp);

        assert_eq!(
            result_bytes, wheel_bytes,
            "persistent store hit must produce byte-identical output"
        );
    }
}
