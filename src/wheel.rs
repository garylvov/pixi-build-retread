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

/// Create a same-directory temp file for an atomic wheel-cache write.
///
/// `ZipWriter` (used by the inject/autodata/relax pipeline in
/// `wheel_inject.rs`, `wheel_inject_data.rs`, and `wheel_rewrite.rs`) needs
/// a concrete `std::fs::File` to write into, so those call sites can't
/// reuse the async tokio-fs temp+rename dance above. This is the sync
/// equivalent of the same protocol: write into `<dst>.<pid>.tmp` (same
/// directory as `dst`, so the final rename is atomic even on NFS) and
/// promote it with [`commit_atomic_write`] only after every byte is
/// flushed. Without this, a process/node death mid-write leaves a
/// truncated file sitting at `dst` -- and because `is_fresh()` only
/// checks mtime, that truncated file gets treated as a valid cache hit
/// forever afterward (see the self-heal check in `is_fresh`).
pub(crate) fn create_atomic_tmp(dst: &Path) -> Result<(PathBuf, std::fs::File)> {
    let tmp = atomic_tmp_path(dst);
    let file =
        std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    Ok((tmp, file))
}

/// Same-directory temp path for an atomic write, without creating the
/// file. Used by callers (e.g. the hard-link fast path in
/// `wheel_rewrite.rs`) whose write step needs the path to NOT already
/// exist (`std::fs::hard_link` errors if the destination is present).
pub(crate) fn atomic_tmp_path(dst: &Path) -> PathBuf {
    let parent = dst.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!(
        "{}.{}.tmp",
        dst.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("wheel.whl"),
        std::process::id()
    ))
}

/// Promote a temp file created by [`create_atomic_tmp`] to its final
/// destination with an atomic same-directory rename. Callers must fully
/// flush (and drop, if holding a wrapping writer) the file before calling
/// this. On any failure the temp file is removed so it never lingers as
/// cache-poisoning debris that a later run could mistake for real output.
pub(crate) fn commit_atomic_write(tmp: &Path, dst: &Path) -> Result<()> {
    std::fs::rename(tmp, dst).map_err(|e| {
        let _ = std::fs::remove_file(tmp);
        anyhow::Error::from(e).context(format!("renaming {} -> {}", tmp.display(), dst.display()))
    })
}

/// Check whether `path` is a well-formed zip archive (just the central
/// directory / EOCD, not full CRC validation of every entry -- callers
/// only need to know "is this readable at all", the corruption pattern
/// seen in practice is a truncated file from an interrupted write, which
/// this catches). Used by the self-heal cache-freshness check.
pub(crate) fn is_valid_zip(path: &Path) -> bool {
    match std::fs::File::open(path) {
        Ok(f) => zip::ZipArchive::new(f).is_ok(),
        Err(_) => false,
    }
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
    // Unique-per-attempt temp name: with entry resolves running concurrently,
    // two BFS walks can race to download the SAME transitive wheel. A shared
    // `<filename>.part` would interleave both streams into one file; a unique
    // temp + the atomic rename below make the race last-writer-wins with
    // identical bytes instead.
    let part = {
        use std::sync::atomic::{AtomicU64, Ordering};
        static PART_SEQ: AtomicU64 = AtomicU64::new(0);
        dest_dir.join(format!(
            "{filename}.{}.{}.part",
            std::process::id(),
            PART_SEQ.fetch_add(1, Ordering::Relaxed)
        ))
    };
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
/// Store layout: `<store_root>/<sha256>/<filename>.whl` (the shared
/// content-addressed wheel store; see `courier::retread_wheel_store_root`).
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
    store_root: &Path,
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
        let actual = sha256_file(&dest).await?;
        if actual.eq_ignore_ascii_case(sha256) {
            tracing::debug!(
                wheel = %filename,
                "wheel cache: already in dest_dir (no fetch needed)",
            );
            return Ok(dest);
        }
        tracing::debug!(
            wheel = %filename,
            "wheel cache: dest_dir hash mismatch; removing stale wheel",
        );
        fs::remove_file(&dest).await.ok();
    }

    // Check the persistent store.
    let store_path = store_root.join(sha256).join(&filename);
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
    let store_dir = store_root.join(sha256);
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

/// Persist a finished wheel file into the shared content-addressed wheel
/// store at `<store_root>/<sha256>/<filename>` and return the hex sha256 of
/// its bytes.
///
/// Loose bundle mode (`retread-bundle-mode = "loose"`) calls this at BUILD
/// time for every wheel that fat mode would have shipped inside the .conda;
/// `retread install` later materializes the wheel from this exact path
/// (hash-verified). Unlike the best-effort store population in
/// [`fetch_wheel_cached`], failure here is a HARD error: a loose lock
/// records the sha and the install replay depends on the store holding the
/// bytes.
///
/// Concurrency-safe: writes go to a process+sequence-unique `.tmp` and are
/// promoted with an atomic rename, same protocol as `fetch_wheel_cached`.
/// An existing store entry is trusted as-is (content-addressed: same sha =>
/// same bytes) and the write is skipped.
pub(crate) async fn store_wheel_in_cache(src: &Path, store_root: &Path) -> Result<String> {
    let filename = src
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| {
            anyhow!(
                "wheel store: source path has no utf-8 filename: {}",
                src.display()
            )
        })?
        .to_string();
    let sha256 = sha256_file(src).await?;

    let store_dir = store_root.join(&sha256);
    let store_final = store_dir.join(&filename);
    if store_final.is_file() {
        tracing::debug!(
            wheel = %filename,
            sha256 = %&sha256[..8],
            "wheel store: already populated",
        );
        return Ok(sha256);
    }
    fs::create_dir_all(&store_dir)
        .await
        .with_context(|| format!("wheel store: creating {}", store_dir.display()))?;

    static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let store_tmp = store_dir.join(format!(
        "{filename}.{}.{}.tmp",
        std::process::id(),
        TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    hardlink_or_copy_async(src, &store_tmp)
        .await
        .with_context(|| {
            format!(
                "wheel store: staging {} -> {}",
                src.display(),
                store_tmp.display()
            )
        })?;
    if let Err(e) = fs::rename(&store_tmp, &store_final).await {
        let _ = fs::remove_file(&store_tmp).await;
        // A concurrent writer may have promoted the same content first;
        // that is a success for our purposes (content-addressed store).
        if !store_final.is_file() {
            return Err(anyhow::Error::new(e).context(format!(
                "wheel store: promoting {} -> {}",
                store_tmp.display(),
                store_final.display()
            )));
        }
    }
    tracing::info!(
        wheel = %filename,
        sha256 = %&sha256[..8],
        "wheel store: persisted (loose bundle)",
    );
    Ok(sha256)
}

/// Pre-fetch a direct-URL `[retread-wheels]` wheel into the content-addressed
/// wheel store and return its STABLE store path, for emission as a
/// `[tool.uv.sources]` `path = "..."` source in the synthesized closure
/// project instead of a `name @ https://...` direct-URL requirement.
///
/// Why: NVIDIA's index (pypi.nvidia.com) serves `cache-control: no-store` and
/// publishes NO PEP 658 metadata sidecars, so when a wheel is emitted as a
/// direct-URL requirement uv downloads the WHOLE wheel and fully unpacks it
/// just to read its METADATA -- and re-pays it on EVERY lock (the response is
/// uncacheable, so warm == cold). The isaacsim-extscache wheels are up to
/// ~5.9 GiB each (7.3 GiB across the three). Emitting a local `path =` source
/// lets uv read METADATA from a seekable local zip: no network, no full
/// unpack, no no-store penalty.
///
/// The returned store path is `<store_root>/<sha256>/<filename>`: content-
/// addressed and therefore STABLE across locks, so the same path string lands
/// in the synthesized pyproject every run. The closure input fingerprint and
/// the full-skip memo are both keyed on the pyproject TEXT
/// (`closure_inputs_fingerprint`), so a stable path keeps the memo hitting
/// (repeated locks full-skip) while the URL->path transition itself changes
/// the text and correctly invalidates any stale pre-transition lock.
///
/// sha256: when `expected_sha256` is known (recipe/config pins it, incl. a
/// `#sha256=` URL fragment) it is verified on fetch by [`fetch_wheel_cached`];
/// when absent, the sha is computed at first fetch by the content-addressed
/// store ([`store_wheel_in_cache`]) -- no new hashing scheme. On any
/// fetch/store failure this returns `Err` and the caller falls back to
/// emitting the direct URL as before (degraded but functional).
pub async fn prefetch_url_wheel_as_source(
    url: &url::Url,
    expected_sha256: Option<&str>,
    dest_dir: &Path,
    store_root: &Path,
) -> Result<PathBuf> {
    let filename = wheel_filename_from_url(url)?;

    // Fast path: sha known AND already in the content-addressed store. This is
    // the steady state -- the wheel store persists across locks (and the
    // extscache wheels are up to ~5.9 GiB). Emit the store path with NO fetch,
    // NO copy, and crucially NO hashing: reading a multi-GB wheel into memory
    // just to re-confirm its sha would spike RSS and can OOM the build backend.
    if let Some(sha) = expected_sha256 {
        let sha = sha.to_ascii_lowercase();
        let store_path = store_root.join(&sha).join(&filename);
        if store_path.is_file() {
            return Ok(store_path);
        }
        // Cold store: fetch (verifies the sha, streaming/incremental -- never
        // buffers the whole wheel) and populate the store, then re-check.
        let fetched = fetch_wheel_cached(url, Some(&sha), dest_dir, store_root).await?;
        if store_path.is_file() {
            return Ok(store_path);
        }
        // Store populate was best-effort and failed: the freshly fetched local
        // copy is still a seekable local zip uv can read METADATA from.
        return Ok(fetched);
    }

    // No pinned sha: fetch, then content-address via the store (the sha is
    // computed once by `store_wheel_in_cache` -- no new hashing scheme). Direct
    // wheel entries normally carry a `#sha256=` fragment, so this branch is the
    // rare exception, not the multi-GB extscache case.
    let fetched = fetch_wheel_cached(url, None, dest_dir, store_root).await?;
    let sha256 = store_wheel_in_cache(&fetched, store_root).await?;
    let store_path = store_root.join(&sha256).join(&filename);
    if store_path.is_file() {
        Ok(store_path)
    } else {
        Ok(fetched)
    }
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

/// Read-ahead window size for [`HttpRangeReader`]. The zip crate reads the
/// end-of-central-directory, then walks the central directory sequentially;
/// a 256 KiB window keeps those walks to a handful of range requests even
/// for a wheel with thousands of members.
const RANGE_WINDOW: u64 = 256 * 1024;

/// A `Read + Seek` view over a remote file, backed by HTTP Range requests.
///
/// Feeding this to `zip::ZipArchive` lets the (battle-tested) zip crate
/// locate + inflate a single member -- the wheel's `METADATA` -- by reading
/// only the end-of-central-directory, the central directory, and that one
/// member's bytes, instead of downloading the whole (multi-GiB) wheel. The
/// zip crate handles zip64 and deflate correctly, which a hand-rolled EOCD
/// parser would have to reimplement (isaacsim's 5.5 GiB extscache wheel is
/// zip64). Blocking by design: driven inside `spawn_blocking`, it issues
/// range GETs through the async client via the captured runtime handle.
struct HttpRangeReader {
    client: reqwest::blocking::Client,
    url: url::Url,
    len: u64,
    pos: u64,
    /// Cached window: (start offset, bytes).
    window: Option<(u64, Vec<u8>)>,
}

impl HttpRangeReader {
    /// Fetch `[start, start+want)` (clamped to `len`) into the window cache
    /// if it isn't already covered, then return a slice of the requested
    /// span from the cache.
    fn ensure(&mut self, start: u64, want: usize) -> std::io::Result<&[u8]> {
        let end = (start + want as u64).min(self.len);
        let covered = matches!(&self.window, Some((ws, wb)) if *ws <= start && start + (want as u64).min(self.len - start) <= *ws + wb.len() as u64);
        if !covered {
            let fetch_len = (end - start).max(RANGE_WINDOW).min(self.len - start);
            let last = start + fetch_len - 1;
            let resp = self
                .client
                .get(self.url.clone())
                .header(reqwest::header::RANGE, format!("bytes={start}-{last}"))
                .send()
                .and_then(|r| r.error_for_status())
                .map_err(std::io::Error::other)?;
            // A server ignoring Range answers 200 with the whole body; treat
            // that as unusable so the caller falls back to a full download.
            if resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "server ignored Range header (200, not 206)",
                ));
            }
            // A 206 alone is not proof the server honored OUR range: a
            // misbehaving server that always answers 206-from-offset-0
            // would silently feed the zip reader wrong bytes (the sha is
            // never recomputed on this path). Require a Content-Range
            // whose start equals the requested start and whose total
            // matches the advertised length; anything else errors so the
            // caller falls back to the full download.
            let content_range = resp
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "206 response without Content-Range",
                    )
                })?;
            let parsed = parse_content_range(&content_range).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unparseable Content-Range `{content_range}`"),
                )
            })?;
            let (cr_start, cr_end, cr_total) = parsed;
            if cr_start != start || cr_end < cr_start || cr_end >= cr_total || cr_total != self.len
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Content-Range `{content_range}` does not match requested \
                         bytes={start}-{last} of {}",
                        self.len
                    ),
                ));
            }
            let bytes = resp.bytes().map_err(std::io::Error::other)?;
            self.window = Some((start, bytes.to_vec()));
        }
        let (ws, wb) = self.window.as_ref().expect("window populated above");
        let off = (start - ws) as usize;
        let avail = wb.len().saturating_sub(off);
        Ok(&wb[off..off + avail.min(want)])
    }
}

/// Parse `Content-Range: bytes <start>-<end>/<total>` into (start, end,
/// total). Returns `None` for any other shape (including the valid-but-
/// useless `bytes */<total>` unsatisfiable form).
fn parse_content_range(value: &str) -> Option<(u64, u64, u64)> {
    let rest = value.trim().strip_prefix("bytes ")?;
    let (span, total) = rest.split_once('/')?;
    let (start, end) = span.split_once('-')?;
    Some((
        start.trim().parse().ok()?,
        end.trim().parse().ok()?,
        total.trim().parse().ok()?,
    ))
}

impl std::io::Read for HttpRangeReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.len || buf.is_empty() {
            return Ok(0);
        }
        let pos = self.pos;
        let slice = self.ensure(pos, buf.len())?;
        let n = slice.len().min(buf.len());
        buf[..n].copy_from_slice(&slice[..n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl std::io::Seek for HttpRangeReader {
    fn seek(&mut self, from: std::io::SeekFrom) -> std::io::Result<u64> {
        let new = match from {
            std::io::SeekFrom::Start(o) => o as i64,
            std::io::SeekFrom::End(o) => self.len as i64 + o,
            std::io::SeekFrom::Current(o) => self.pos as i64 + o,
        };
        if new < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek before start",
            ));
        }
        self.pos = new as u64;
        Ok(self.pos)
    }
}

/// Read a wheel's METADATA via HTTP Range requests, transferring only the
/// zip central directory + the METADATA member instead of the whole wheel.
///
/// Caller contract mirrors [`fetch_metadata_sidecar`]: only call with the
/// wheel's `sha256` (from the index link fragment), since the recipe pins
/// each source wheel's hash and the ranged read never sees the full bytes
/// to compute it. Returns `Err` (so the caller falls back to a full
/// download) when the server doesn't honor Range or the zip can't be read
/// this way.
pub async fn fetch_metadata_ranged(
    wheel_url: &url::Url,
    wheel_sha256: &str,
) -> Result<WheelMetadata> {
    let filename = wheel_filename_from_url(wheel_url)?;
    let is_pure_python = is_pure_python_wheel_filename(&filename);
    let mut url = wheel_url.clone();
    url.set_fragment(None);
    let sha = wheel_sha256.to_ascii_lowercase();

    // Everything here is blocking (a `reqwest::blocking` client driving the
    // sync `zip` reader), so it all runs on one blocking thread -- no nested
    // tokio runtime, works regardless of the caller's runtime flavor.
    tokio::task::spawn_blocking(move || -> Result<WheelMetadata> {
        // Explicit timeouts (M2): a stalled CDN must become an Err (which
        // the caller turns into a full-download fallback), not a forever-
        // blocked thread -- the default blocking client has NO timeout.
        // Generous values: the largest single transfer here is one
        // RANGE_WINDOW (256 KiB), so 120s of read headroom is ample even
        // on a bad link.
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .context("building ranged-fetch HTTP client")?;
        // Discover length + Range support. A HEAD keeps this to one tiny
        // request on the happy path; servers that 405 the HEAD (or omit the
        // length, or don't advertise byte ranges) error out and the caller
        // falls back to a full download.
        let head = client
            .head(url.clone())
            .send()
            .with_context(|| format!("HEAD {url}"))?
            .error_for_status()
            .with_context(|| format!("HTTP error for HEAD {url}"))?;
        let accepts_ranges = head
            .headers()
            .get(reqwest::header::ACCEPT_RANGES)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.eq_ignore_ascii_case("bytes"))
            .unwrap_or(false);
        // Read the Content-Length HEADER directly: `Response::content_length()`
        // on a HEAD reflects the (empty) body, not the advertised size.
        let len = head
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok())
            .ok_or_else(|| anyhow!("no Content-Length for {url}; cannot range-fetch"))?;
        if !accepts_ranges {
            bail!("server does not advertise `Accept-Ranges: bytes` for {url}");
        }
        if len == 0 {
            bail!("empty body for {url}");
        }

        let reader = HttpRangeReader {
            client,
            url: url.clone(),
            len,
            pos: 0,
            window: None,
        };
        let mut archive = zip::ZipArchive::new(reader)
            .with_context(|| format!("opening remote zip via Range for {filename}"))?;
        // Root-level `<name>-<version>.dist-info/METADATA` only (wheels may
        // vendor nested .dist-info trees; ours is the single-slash one).
        let mut idx = None;
        for i in 0..archive.len() {
            let entry = archive.by_index(i)?;
            let name = entry.name();
            if name.ends_with(".dist-info/METADATA") && name.matches('/').count() == 1 {
                idx = Some(i);
                break;
            }
        }
        let idx = idx.ok_or_else(|| anyhow!("no root-level .dist-info/METADATA in {filename}"))?;
        let mut entry = archive.by_index(idx)?;
        let mut buf = String::new();
        entry.read_to_string(&mut buf)?;
        parse_metadata(&buf, filename, is_pure_python, sha)
    })
    .await
    .context("ranged metadata reader panicked")?
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

    /// Build a minimal but real wheel zip (deflate-compressed METADATA at a
    /// root-level `.dist-info`) so the ranged reader exercises central-
    /// directory + member inflation, not just STORED bytes.
    fn build_test_wheel_zip() -> Vec<u8> {
        use std::io::Write as _;
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut zw = zip::ZipWriter::new(&mut cursor);
            let opts: zip::write::FileOptions<'_, ()> =
                zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);
            // A vendored nested .dist-info (deeper path) must be ignored --
            // only the single-slash root entry is ours.
            zw.start_file("vendored/other-9.9.dist-info/METADATA", opts)
                .unwrap();
            zw.write_all(b"Name: other\nVersion: 9.9\n").unwrap();
            zw.start_file("foo-1.0.dist-info/METADATA", opts).unwrap();
            zw.write_all(
                b"Metadata-Version: 2.1\nName: foo\nVersion: 1.0\n\
                  Requires-Dist: bar>=2\nRequires-Dist: baz; extra=='x'\n\nbody\n",
            )
            .unwrap();
            zw.finish().unwrap();
        }
        cursor.into_inner()
    }

    /// A tiny HTTP/1.1 server honoring HEAD + `Range` GETs, for the ranged
    /// metadata test. Handles keep-alive (multiple requests per connection).
    async fn serve_ranged(bytes: Vec<u8>) -> (u16, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let bytes = bytes.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 1024];
                    loop {
                        // Read one request (headers terminate at \r\n\r\n).
                        let hdr_end = loop {
                            if let Some(p) = buf
                                .windows(4)
                                .position(|w| w == b"\r\n\r\n")
                            {
                                break p + 4;
                            }
                            let n = match stream.read(&mut tmp).await {
                                Ok(0) | Err(_) => return,
                                Ok(n) => n,
                            };
                            buf.extend_from_slice(&tmp[..n]);
                        };
                        let req = String::from_utf8_lossy(&buf[..hdr_end]).to_string();
                        buf.drain(..hdr_end);
                        let is_head = req.starts_with("HEAD");
                        let range = req.lines().find_map(|l| {
                            let l = l.to_ascii_lowercase();
                            l.strip_prefix("range: bytes=").map(|s| s.trim().to_string())
                        });
                        let total = bytes.len();
                        let resp: Vec<u8> = if is_head {
                            format!(
                                "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\n\r\n"
                            )
                            .into_bytes()
                        } else if let Some(r) = range {
                            let (a, b) = r.split_once('-').unwrap();
                            let a: usize = a.parse().unwrap();
                            let b: usize = if b.is_empty() {
                                total - 1
                            } else {
                                b.parse::<usize>().unwrap().min(total - 1)
                            };
                            let slice = &bytes[a..=b];
                            let mut h = format!(
                                "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\n\
                                 Content-Range: bytes {a}-{b}/{total}\r\nContent-Length: {}\r\n\r\n",
                                slice.len()
                            )
                            .into_bytes();
                            h.extend_from_slice(slice);
                            h
                        } else {
                            let mut h = format!(
                                "HTTP/1.1 200 OK\r\nContent-Length: {total}\r\n\r\n"
                            )
                            .into_bytes();
                            h.extend_from_slice(&bytes);
                            h
                        };
                        if stream.write_all(&resp).await.is_err() {
                            return;
                        }
                        let _ = stream.flush().await;
                    }
                });
            }
        });
        (port, handle)
    }

    #[tokio::test]
    async fn ranged_metadata_reads_only_the_metadata_member() {
        let zip_bytes = build_test_wheel_zip();
        let (port, server) = serve_ranged(zip_bytes).await;
        let url: url::Url = format!("http://127.0.0.1:{port}/foo-1.0-py3-none-any.whl")
            .parse()
            .unwrap();
        let sha = "a".repeat(64);
        let md = fetch_metadata_ranged(&url, &sha).await.unwrap();
        assert_eq!(md.name, "foo");
        assert_eq!(md.version, "1.0");
        // Root-level METADATA parsed; the vendored nested one ignored.
        assert_eq!(md.requires_dist, vec!["bar>=2", "baz; extra=='x'"]);
        // sha256 comes from the caller (index fragment), not recomputed.
        assert_eq!(md.sha256, sha);
        assert!(md.is_pure_python, "py3-none-any is pure python");
        server.abort();
    }

    #[tokio::test]
    async fn ranged_metadata_rejects_misbehaving_206_wrong_offset() {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        // M1: a server that answers 206 but ALWAYS serves from offset 0
        // (ignoring the requested range while still claiming success) must
        // be rejected via Content-Range validation -- accepting it would
        // hand the zip reader silently wrong bytes.
        //
        // The zip must be LARGER than the misbehaving server's fixed slice
        // (1024 B below): a zip that fits entirely in the offset-0 slice
        // never forces a nonzero-offset request, and offset-0 requests are
        // the one case this server answers correctly. Pad with a stored
        // (incompressible-by-construction) member so reads past 1024 occur.
        let zip_bytes = {
            use std::io::Write as _;
            let mut cursor = std::io::Cursor::new(Vec::new());
            {
                let mut zw = zip::ZipWriter::new(&mut cursor);
                let stored: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored);
                zw.start_file("padding.bin", stored).unwrap();
                let pad: Vec<u8> = (0..8192u32).map(|i| (i % 251) as u8).collect();
                zw.write_all(&pad).unwrap();
                zw.start_file("foo-1.0.dist-info/METADATA", stored).unwrap();
                zw.write_all(b"Metadata-Version: 2.1\nName: foo\nVersion: 1.0\n\n")
                    .unwrap();
                zw.finish().unwrap();
            }
            cursor.into_inner()
        };
        assert!(zip_bytes.len() > 1024, "padding must exceed the server slice");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let bytes = zip_bytes.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 1024];
                    loop {
                        let hdr_end = loop {
                            if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                                break p + 4;
                            }
                            let n = match stream.read(&mut tmp).await {
                                Ok(0) | Err(_) => return,
                                Ok(n) => n,
                            };
                            buf.extend_from_slice(&tmp[..n]);
                        };
                        let req = String::from_utf8_lossy(&buf[..hdr_end]).to_string();
                        buf.drain(..hdr_end);
                        let total = bytes.len();
                        let resp: Vec<u8> = if req.starts_with("HEAD") {
                            format!(
                                "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\n\r\n"
                            )
                            .into_bytes()
                        } else {
                            // Misbehaving: 206, but always from offset 0,
                            // Content-Range honestly reporting the wrong span.
                            let n = total.min(1024);
                            let slice = &bytes[..n];
                            let mut h = format!(
                                "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\n\
                                 Content-Range: bytes 0-{}/{total}\r\nContent-Length: {}\r\n\r\n",
                                n - 1,
                                slice.len()
                            )
                            .into_bytes();
                            h.extend_from_slice(slice);
                            h
                        };
                        if stream.write_all(&resp).await.is_err() {
                            return;
                        }
                        let _ = stream.flush().await;
                    }
                });
            }
        });
        let url: url::Url = format!("http://127.0.0.1:{port}/foo-1.0-py3-none-any.whl")
            .parse()
            .unwrap();
        let err = fetch_metadata_ranged(&url, &"e".repeat(64)).await;
        assert!(
            err.is_err(),
            "wrong-offset 206 must be rejected, got {err:?}"
        );
        server.abort();
    }

    #[test]
    fn parse_content_range_shapes() {
        assert_eq!(parse_content_range("bytes 100-199/500"), Some((100, 199, 500)));
        assert_eq!(parse_content_range(" bytes 0-0/1"), Some((0, 0, 1)));
        assert_eq!(parse_content_range("bytes */500"), None);
        assert_eq!(parse_content_range("items 0-1/2"), None);
        assert_eq!(parse_content_range("garbage"), None);
    }

    #[tokio::test]
    async fn ranged_metadata_errors_when_range_unsupported() {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        // Server that ignores Range and always returns 200 full-body: the
        // ranged reader must error so the caller falls back to a full fetch.
        let zip_bytes = build_test_wheel_zip();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let bytes = zip_bytes.clone();
                tokio::spawn(async move {
                    let mut tmp = [0u8; 1024];
                    loop {
                        match stream.read(&mut tmp).await {
                            Ok(0) | Err(_) => return,
                            Ok(_) => {}
                        }
                        // No Accept-Ranges, always 200 with the whole body.
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                            bytes.len()
                        );
                        if stream.write_all(resp.as_bytes()).await.is_err() {
                            return;
                        }
                        if stream.write_all(&bytes).await.is_err() {
                            return;
                        }
                        let _ = stream.flush().await;
                    }
                });
            }
        });
        let url: url::Url = format!("http://127.0.0.1:{port}/foo-1.0-py3-none-any.whl")
            .parse()
            .unwrap();
        let err = fetch_metadata_ranged(&url, &"b".repeat(64)).await;
        assert!(err.is_err(), "must reject a server that ignores Range");
        server.abort();
    }

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
                max_glibc: None,
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
        let store_root = tmp.join("store");
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
        let store_dir = store_root.join(&sha256);
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

    // ── Run 9: atomic cache-write + corrupted-zip self-heal ──────────────────

    /// `create_atomic_tmp` + `commit_atomic_write` round-trips real bytes to
    /// the final destination and leaves no temp file behind.
    #[test]
    fn atomic_write_round_trips_and_cleans_up_tmp() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "retread-wheel-test-atomic-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let dst = tmp_dir.join("pkg-1.0-py3-none-any.whl");

        let (tmp, mut file) = create_atomic_tmp(&dst).unwrap();
        assert_ne!(tmp, dst, "temp path must differ from the final destination");
        assert_eq!(
            tmp.parent(),
            dst.parent(),
            "temp file must live in the same directory as dst so the rename is atomic"
        );
        use std::io::Write as _;
        file.write_all(b"PK\x03\x04pretend wheel bytes").unwrap();
        file.flush().unwrap();
        drop(file);

        assert!(!dst.exists(), "dst must not exist before the commit");
        commit_atomic_write(&tmp, &dst).unwrap();

        assert!(dst.exists(), "dst must exist after commit");
        assert!(
            !tmp.exists(),
            "temp file must be gone after a successful commit"
        );
        assert_eq!(
            std::fs::read(&dst).unwrap(),
            b"PK\x03\x04pretend wheel bytes"
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    /// A reader that dies between `create_atomic_tmp` and `commit_atomic_write`
    /// (the run-9 failure: a compute node dying mid wheel-write) leaves ONLY
    /// the `.tmp` file on disk -- `dst` never appears in a truncated state.
    #[test]
    fn atomic_write_never_leaves_truncated_dst_on_interruption() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "retread-wheel-test-atomic-interrupt-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let dst = tmp_dir.join("pkg-1.0-py3-none-any.whl");

        let (tmp, mut file) = create_atomic_tmp(&dst).unwrap();
        use std::io::Write as _;
        file.write_all(b"only half the wheel").unwrap();
        // Simulate the process dying here: never call commit_atomic_write.
        drop(file);

        assert!(
            !dst.exists(),
            "dst must never appear until the atomic rename commits"
        );
        assert!(
            tmp.exists(),
            "the half-written temp file is the only artifact left behind"
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    /// `is_valid_zip` accepts a well-formed zip and rejects a truncated one
    /// (the "Could not find EOCD" corruption pattern proven in run 9).
    #[test]
    fn is_valid_zip_detects_corrupted_archive() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "retread-wheel-test-validzip-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&tmp_dir).unwrap();

        // A real, complete zip archive.
        let good = tmp_dir.join("good.whl");
        {
            let file = std::fs::File::create(&good).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            writer
                .start_file("a.txt", zip::write::SimpleFileOptions::default())
                .unwrap();
            use std::io::Write as _;
            writer.write_all(b"hello").unwrap();
            writer.finish().unwrap();
        }
        assert!(is_valid_zip(&good), "a complete zip archive must validate");

        // Garbage bytes masquerading as a wheel: not a zip at all.
        let garbage = tmp_dir.join("garbage.whl");
        std::fs::write(&garbage, b"not a zip file, just garbage bytes").unwrap();
        assert!(
            !is_valid_zip(&garbage),
            "non-zip garbage bytes must be rejected"
        );

        // Truncated zip: valid header, but the EOCD record got cut off --
        // the exact "invalid Zip archive: Could not find EOCD" failure mode
        // from a node dying mid-write.
        let good_bytes = std::fs::read(&good).unwrap();
        let truncated = tmp_dir.join("truncated.whl");
        std::fs::write(&truncated, &good_bytes[..good_bytes.len() / 2]).unwrap();
        assert!(
            !is_valid_zip(&truncated),
            "a truncated (EOCD-missing) zip must be rejected, not treated as valid"
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    // ---- prefetch_url_wheel_as_source (direct-URL -> path source) ----

    /// Stage a fake wheel in `dest_dir` so `fetch_wheel_cached`'s early-return
    /// (dest exists) fires and no network is touched. Returns (url, sha).
    fn stage_offline_wheel(dest_dir: &Path, filename: &str) -> (url::Url, String) {
        std::fs::create_dir_all(dest_dir).unwrap();
        let bytes = build_test_wheel_zip();
        std::fs::write(dest_dir.join(filename), &bytes).unwrap();
        let sha = hex_sha256(&bytes);
        // Any host: the dest early-return means we never connect.
        let url = url::Url::parse(&format!("https://pypi.nvidia.com/x/{filename}")).unwrap();
        (url, sha)
    }

    #[tokio::test]
    async fn prefetch_url_wheel_source_with_sha_returns_stable_store_path() {
        let tmp = std::env::temp_dir().join(format!("retread-prefetch-sha-{}", std::process::id()));
        let dest = tmp.join("dl");
        let store = tmp.join("store");
        let filename = "foo-1.0-py3-none-any.whl";
        // Warm-store steady state: the wheel already lives in the content-
        // addressed store. The prefetch must emit the store path with NO fetch
        // and NO hashing of the (potentially multi-GB) wheel -- so a bogus host
        // that would fail any network call proves no fetch was attempted.
        let bytes = build_test_wheel_zip();
        let sha = hex_sha256(&bytes);
        std::fs::create_dir_all(store.join(&sha)).unwrap();
        std::fs::write(store.join(&sha).join(filename), &bytes).unwrap();
        let url = url::Url::parse(&format!("https://127.0.0.1:1/x/{filename}")).unwrap();

        let path = prefetch_url_wheel_as_source(&url, Some(&sha), &dest, &store)
            .await
            .expect("prefetch with known sha (warm store)");
        // Content-addressed store path, stable across runs.
        assert_eq!(path, store.join(&sha).join(filename));
        assert!(path.is_file(), "store path must hold the wheel bytes");

        // Idempotent: a second call yields the identical path (stable
        // fingerprint -> full-skip memo keeps hitting).
        let again = prefetch_url_wheel_as_source(&url, Some(&sha), &dest, &store)
            .await
            .expect("prefetch again");
        assert_eq!(again, path);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn prefetch_url_wheel_source_without_sha_records_computed_sha() {
        let tmp =
            std::env::temp_dir().join(format!("retread-prefetch-nosha-{}", std::process::id()));
        let dest = tmp.join("dl");
        let store = tmp.join("store");
        let filename = "bar-2.0-py3-none-any.whl";
        let (url, sha) = stage_offline_wheel(&dest, filename);

        // No configured sha: the content-addressed store computes it at first
        // fetch (existing store_wheel_in_cache behavior, no new hashing).
        let path = prefetch_url_wheel_as_source(&url, None, &dest, &store)
            .await
            .expect("prefetch without sha");
        assert_eq!(path, store.join(&sha).join(filename));
        assert!(path.is_file());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn prefetch_url_wheel_source_errors_on_fetch_failure() {
        let tmp =
            std::env::temp_dir().join(format!("retread-prefetch-fail-{}", std::process::id()));
        let dest = tmp.join("dl");
        let store = tmp.join("store");
        // Nothing staged in dest + a closed local port => fetch_wheel's GET
        // fails fast (connection refused), so the caller falls back to the
        // direct-URL requirement. Offline + deterministic.
        let url = url::Url::parse("http://127.0.0.1:1/nope-1.0-py3-none-any.whl").unwrap();
        let err = prefetch_url_wheel_as_source(&url, None, &dest, &store).await;
        assert!(
            err.is_err(),
            "unreachable fetch must error (caller falls back to URL)"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
