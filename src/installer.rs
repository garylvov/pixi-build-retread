//! v2.0.0 courier: the `retread install` subcommand.
//!
//! Invoked from the courier conda package's post-link script at env link
//! time: `pixi-build-retread install --lock <lock> --prefix <prefix>`.
//! Reads the committed lock and installs the bundle's exact PyPI wheel files
//! into the active conda env via uv (fast hardlink), without dependency
//! resolution or index metadata access.
//! Idempotent: a content-hash marker makes a re-link a no-op.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr as _;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use crate::lock::{Origin, RetreadLock};

/// PEP 503 normalized distribution name (lowercase; runs of `-`, `_`, `.`
/// collapse to a single `-`). Used to compare installed metadata with the
/// lock's wheel names during verification.
pub(crate) fn normalize_dist_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_sep = false;
    for c in name.trim().chars() {
        if c == '-' || c == '_' || c == '.' {
            if !prev_sep && !out.is_empty() {
                out.push('-');
            }
            prev_sep = true;
        } else {
            out.push(c.to_ascii_lowercase());
            prev_sep = false;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

fn pep440_versions_equal(left: &str, right: &str) -> bool {
    match (
        uv_pep508::uv_pep440::Version::from_str(left),
        uv_pep508::uv_pep440::Version::from_str(right),
    ) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

fn installed_version_path<'a>(
    versions: &'a BTreeMap<String, PathBuf>,
    expected: &str,
) -> Option<&'a PathBuf> {
    versions
        .iter()
        .find_map(|(version, path)| pep440_versions_equal(version, expected).then_some(path))
}

/// True if uv's output shows a wheel rejected purely for an unsatisfied
/// manylinux platform tag -- the one failure mode the `--python-platform`
/// glibc relaxation is allowed to recover from. uv phrases it as
/// "has no wheels with a matching platform tag (e.g., `manylinux_2_34_x86_64`)".
/// Matched conservatively so unrelated solve/network/link failures are never
/// silently "relaxed" past.
pub(crate) fn is_platform_tag_conflict(text: &str) -> bool {
    // uv line-wraps its resolver errors ("...matching platform\n      tag..."),
    // so collapse all whitespace runs to single spaces before matching.
    let t = text
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    t.contains("matching platform tag") || (t.contains("platform tag") && t.contains("manylinux"))
}

/// Decide whether to relax the glibc/manylinux host gate after a uv install
/// failure. Relaxation is allowed only for a genuine platform-tag rejection
/// and only up to a glibc floor declared by the live workspace or lock.
fn relax_platform_on_conflict(
    uv: &OsString,
    base_args: &[OsString],
    lock: &RetreadLock,
) -> Result<Option<crate::glibc::RelaxOutcome>> {
    let mut probe: Vec<OsString> = base_args.to_vec();
    probe.push("--dry-run".into());
    // This output is parsed (is_platform_tag_conflict), not shown to the
    // user -- force plain text so a FORCE_COLOR/CLICOLOR_FORCE in the
    // caller's environment can't taint it with ANSI codes that break the
    // substring match.
    probe.push("--color".into());
    probe.push("never".into());
    let out = match Command::new(uv).args(&probe).output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("retread: relax probe failed to spawn uv: {e}");
            return Ok(None);
        }
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    if !is_platform_tag_conflict(&text) {
        eprintln!(
            "retread: relax probe: no platform-tag conflict detected (probe status {}, {} bytes): {}",
            out.status,
            text.len(),
            text.chars().take(2000).collect::<String>()
        );
        return Ok(None);
    }

    let host = crate::glibc::host_glibc();
    let declared = crate::glibc::resolve_declared_glibc(lock);
    match crate::glibc::relax_decision(declared.as_ref().map(|d| d.version), host) {
        crate::glibc::RelaxDecision::Relax { target } => {
            let declared = declared.expect("decision had declared version");
            let platform = format!(
                "{}-manylinux_{}_{}",
                std::env::consts::ARCH,
                target.0,
                target.1
            );
            crate::glibc::emit_glibc_relax_warning(
                host.expect("decision had host"),
                declared.version,
                declared.source,
                &platform,
            );
            Ok(Some(crate::glibc::RelaxOutcome {
                platform,
                declared: declared.version,
                declaration_source: declared.source,
            }))
        }
        crate::glibc::RelaxDecision::NotNeeded => {
            if let (Some(host), Some(declared)) = (host, declared) {
                eprintln!(
                    "retread: platform-tag conflict detected, but declared glibc {} <= host {}; not a glibc-floor problem",
                    crate::glibc::format_glibc(declared.version),
                    crate::glibc::format_glibc(host)
                );
            }
            Ok(None)
        }
        crate::glibc::RelaxDecision::Undeclared => {
            bail!(
                "{}",
                crate::glibc::undeclared_glibc_error(
                    host,
                    crate::glibc::extract_manylinux_floor(&text)
                )
            );
        }
        crate::glibc::RelaxDecision::HostUnknown => {
            tracing::warn!(
                "manylinux platform-tag conflict detected, but the host glibc \
                 could not be detected (getconf/ldd unavailable); cannot \
                 auto-relax the platform tag"
            );
            eprintln!(
                "retread: manylinux platform-tag conflict detected, but host \
                 glibc is undetectable; not relaxing"
            );
            Ok(None)
        }
    }
}

fn lock_digest(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}

#[cfg(test)]
fn sha256_file(path: &Path) -> Result<String> {
    use std::io::Read as _;

    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("hashing {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_safe_bundle_component(value: &str) -> bool {
    !value.trim().is_empty()
        && !matches!(value, "." | "..")
        && !value.contains('/')
        && !value.contains('\\')
        && Path::new(value)
            .file_name()
            .and_then(|component| component.to_str())
            == Some(value)
        && Path::new(value).components().count() == 1
}

fn validate_install_lock_for_platform(
    lock_path: &Path,
    lock: &RetreadLock,
    platform: &str,
) -> Result<()> {
    if !is_safe_bundle_component(&lock.bundle) {
        bail!(
            "retread install: lock {} records invalid bundle path component {:?}; \
             rebuild the courier pack",
            lock_path.display(),
            lock.bundle,
        );
    }
    if lock.schema != crate::lock::SCHEMA {
        bail!(
            "retread install: lock {} has schema {}, but this retread requires schema {} \
             for zero-resolution replay. Rebuild the courier pack so the lock records \
             direct wheel URLs and hashes.",
            lock_path.display(),
            lock.schema,
            crate::lock::SCHEMA
        );
    }
    lock.validate_replay_provenance().with_context(|| {
        format!(
            "retread install: lock {} has invalid replay provenance; rebuild the courier pack",
            lock_path.display()
        )
    })?;
    // Courier packages embed the native retread executable and a wheel set
    // selected for one exact conda subdir. They are never cross-installable.
    if !lock.is_for_target(platform) {
        bail!(
            "retread install: lock {} targets {}, but this host is {}. \
             Rebuild the courier pack for the host platform before installing it.",
            lock_path.display(),
            lock.target_subdir,
            platform,
        );
    }
    let target = lock.resolution_target().with_context(|| {
        format!(
            "retread install: lock {} has malformed target metadata; rebuild the courier pack",
            lock_path.display()
        )
    })?;
    lock.validate_replay_contract_for_target(&target, &lock.bundle)
        .with_context(|| {
            format!(
                "retread install: lock {} has invalid complete replay provenance; rebuild the courier pack",
                lock_path.display()
            )
        })?;
    let mut seen_distributions: BTreeMap<String, String> = BTreeMap::new();
    let mut seen_filenames: BTreeMap<String, String> = BTreeMap::new();
    for wheel in &lock.wheels {
        let mut source_provenance: Option<(String, String)> = None;
        if wheel.name.trim().is_empty()
            || wheel.version.trim().is_empty()
            || wheel.filename.trim().is_empty()
        {
            bail!(
                "retread install: lock {} contains an incomplete wheel entry; rebuild the courier pack",
                lock_path.display()
            );
        }
        let canonical_name = normalize_dist_name(&wheel.name);
        if let Some(prior) = seen_distributions.insert(canonical_name, wheel.name.clone()) {
            bail!(
                "retread install: lock {} contains duplicate distributions `{prior}` and `{}`; \
                 rebuild the courier pack",
                lock_path.display(),
                wheel.name,
            );
        }
        let standard_filename = crate::courier::validate_wheel_filename_for_target(
            &wheel.name,
            &wheel.version,
            &wheel.filename,
            target.wheel_target(),
            "retread install locked wheel filename",
        )
        .with_context(|| {
            format!(
                "retread install: lock {} records the wrong wheel identity for {}=={}",
                lock_path.display(),
                wheel.name,
                wheel.version,
            )
        })?;
        debug_assert_eq!(
            standard_filename,
            crate::emit_pypi::standard_wheel_filename(&wheel.filename)
        );
        if let Some(prior) = seen_filenames.insert(
            standard_filename.to_ascii_lowercase(),
            wheel.filename.clone(),
        ) {
            bail!(
                "retread install: lock {} contains duplicate wheel filenames `{prior}` and `{}`; \
                 rebuild the courier pack",
                lock_path.display(),
                wheel.filename,
            );
        }
        if wheel.origin == Origin::Index && (wheel.url.is_none() || wheel.sha256.is_none()) {
            bail!(
                "retread install: lock {} cannot replay {}=={} without resolution: \
                 Origin::Index entries require both url and sha256. Rebuild the courier pack.",
                lock_path.display(),
                wheel.name,
                wheel.version
            );
        }
        if let Some(sha256) = wheel.sha256.as_deref()
            && !is_sha256(sha256)
        {
            bail!(
                "retread install: lock {} records an invalid sha256 for {}=={}; \
                 rebuild the courier pack",
                lock_path.display(),
                wheel.name,
                wheel.version,
            );
        }
        for (provenance, url_text) in [
            ("locked URL", wheel.url.as_deref()),
            ("upstream URL", wheel.upstream_url.as_deref()),
        ] {
            let Some(url_text) = url_text else { continue };
            let url = url::Url::parse(url_text).with_context(|| {
                format!(
                    "retread install: invalid {provenance} for {}=={}: {url_text}",
                    wheel.name, wheel.version
                )
            })?;
            let filename = crate::wheel::wheel_filename_from_url(&url).with_context(|| {
                format!(
                    "retread install: invalid wheel filename in {provenance} for {}=={}",
                    wheel.name, wheel.version
                )
            })?;
            let standard = crate::courier::validate_wheel_filename_for_target(
                &wheel.name,
                &wheel.version,
                &filename,
                target.wheel_target(),
                &format!("retread install {provenance} filename"),
            )
            .with_context(|| {
                format!(
                    "retread install: {provenance} records the wrong wheel identity for {}=={}",
                    wheel.name, wheel.version,
                )
            })?;
            debug_assert_eq!(
                standard,
                crate::emit_pypi::standard_wheel_filename(&filename)
            );
            if wheel.origin == Origin::Index && filename != wheel.filename {
                bail!(
                    "retread install: locked URL filename for {}=={} is {}, but the lock \
                     records {}; rebuild the courier pack",
                    wheel.name,
                    wheel.version,
                    filename,
                    wheel.filename,
                );
            }
            if wheel.origin == Origin::Built
                && !crate::courier::wheel_filename_provenance_matches(&wheel.filename, &filename)
            {
                bail!(
                    "retread install: {provenance} filename for {}=={} is {}, but the lock \
                     records different wheel provenance {}; rebuild the courier pack",
                    wheel.name,
                    wheel.version,
                    filename,
                    wheel.filename,
                );
            }
            if wheel.origin == Origin::Built && !crate::courier::has_owned_shadow_build(&standard) {
                if let Some((prior_provenance, prior_filename)) = &source_provenance
                    && prior_filename != &standard
                {
                    bail!(
                        "retread install: source provenance mismatch for {}=={}: \
                         {prior_provenance} records {}, but {provenance} records {}; \
                         rebuild the courier pack",
                        wheel.name,
                        wheel.version,
                        prior_filename,
                        filename,
                    );
                }
                source_provenance = Some((provenance.to_owned(), standard));
            }
        }
    }
    Ok(())
}

fn validate_install_lock(lock_path: &Path, lock: &RetreadLock) -> Result<()> {
    validate_install_lock_for_platform(lock_path, lock, crate::glibc::current_pixi_platform())
}

fn lock_basename_matches_target(
    lock_path: &Path,
    bundle: &str,
    target: &crate::pypi::ResolutionTarget,
) -> bool {
    let Some(lock_basename) = lock_path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    lock_basename == RetreadLock::file_name_for_target(bundle, target)
        || (target.conda_subdir() == "linux-64" && lock_basename == RetreadLock::file_name(bundle))
}

fn read_validated_lock(lock_path: &Path) -> Result<(Vec<u8>, RetreadLock)> {
    let raw = std::fs::read(lock_path).with_context(|| {
        format!(
            "reading lock {}. Rebuild the courier pack so install replay has a valid lock.",
            lock_path.display()
        )
    })?;
    let mut lock: RetreadLock = serde_json::from_slice(&raw).with_context(|| {
        format!(
            "parsing lock {}. Rebuild the courier pack so install replay has a valid lock.",
            lock_path.display()
        )
    })?;
    validate_install_lock(lock_path, &lock)?;
    let target = lock.resolution_target()?;
    if !lock_basename_matches_target(lock_path, &lock.bundle, &target) {
        bail!(
            "retread install: lock {} does not match its recorded bundle `{}` and target; rebuild the courier pack",
            lock_path.display(),
            lock.bundle,
        );
    }
    // Runtime paths are minor-scoped (`lib/python3.11/site-packages`). Locks
    // may compatibly spell the same target as 3.11.0, so canonicalize only the
    // in-memory representation after validating and retaining the raw bytes
    // for the idempotency digest.
    lock.python = crate::lock::normalized_target_python(&lock.python)?;
    Ok((raw, lock))
}

/// Resolve the wheel-store root this install reads from (and populates).
/// Order: `RETREAD_WHEEL_STORE` env override > store recorded in the lock at
/// build time (`~` expands against THIS machine's home) > shared default.
fn resolve_wheel_store_root(lock: &RetreadLock) -> PathBuf {
    resolve_wheel_store_root_with(&|key| std::env::var(key).ok(), lock)
}

/// Testable core of [`resolve_wheel_store_root`]; `env` is the variable
/// lookup. With no override and no lock-recorded store this is EXACTLY the
/// build-side default (`courier::wheel_store_root_with`), so producer and
/// consumer agree even when fast-tmp has redirected `RETREAD_CACHE_DIR`.
fn resolve_wheel_store_root_with(
    env: &dyn Fn(&str) -> Option<String>,
    lock: &RetreadLock,
) -> PathBuf {
    if let Some(dir) = env("RETREAD_WHEEL_STORE").filter(|s| !s.trim().is_empty()) {
        return PathBuf::from(dir);
    }
    if let Some(recorded) = lock.wheel_store.as_deref() {
        return crate::courier::expand_wheel_store_path(recorded);
    }
    crate::courier::wheel_store_root_with(env)
}

/// Store roots to probe for a locked Built wheel, in precedence order:
/// the resolved primary store, the lock-recorded store, the shared default,
/// and the legacy `<retread cache root>/wheels` location (stores written by
/// older binaries under `RETREAD_CACHE_DIR`, including fast-tmp-redirected
/// ones still alive on this node). Deduplicated, order-preserving.
fn built_wheel_store_candidates(lock: &RetreadLock, primary: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = vec![primary.to_path_buf()];
    if let Some(recorded) = lock.wheel_store.as_deref() {
        out.push(crate::courier::expand_wheel_store_path(recorded));
    }
    out.push(crate::courier::retread_wheel_store_root());
    out.push(crate::courier::retread_cache_root().join("wheels"));
    let mut seen = std::collections::HashSet::new();
    out.retain(|p| seen.insert(p.clone()));
    out
}

fn validate_materialized_wheel(
    wheel: &crate::lock::LockWheel,
    path: &Path,
    provenance: &str,
) -> Result<()> {
    crate::courier::validate_wheel_file_identity(
        &wheel.name,
        &wheel.version,
        path,
        wheel.sha256.as_deref(),
        provenance,
    )
    .with_context(|| {
        format!(
            "retread install: {provenance} for {}=={} failed provenance validation",
            wheel.name, wheel.version,
        )
    })
}

async fn materialize_index_wheel(
    lock: &RetreadLock,
    wheel: &crate::lock::LockWheel,
    fetch_dir: &Path,
    store_root: &Path,
) -> Result<PathBuf> {
    let url_text = wheel.url.as_deref().expect("validated index url");
    let expected_sha = wheel.sha256.as_deref().expect("validated index sha256");
    let url = url::Url::parse(url_text).with_context(|| {
        format!(
            "retread install: invalid locked URL for {}=={}: {}",
            wheel.name, wheel.version, url_text
        )
    })?;
    let url_filename = crate::wheel::wheel_filename_from_url(&url)?;
    if url_filename != wheel.filename {
        bail!(
            "retread install: locked URL filename for {}=={} is {}, but lock records {}; \
             rebuild the courier pack",
            wheel.name,
            wheel.version,
            url_filename,
            wheel.filename
        );
    }

    if url.scheme() == "file" {
        let path = url.to_file_path().map_err(|_| {
            anyhow::anyhow!(
                "retread install: locked file URL for {}=={} is not a valid path: {}",
                wheel.name,
                wheel.version,
                url
            )
        })?;
        validate_materialized_wheel(wheel, &path, "locked file-URL wheel")?;
        return Ok(path);
    }

    let store_path = store_root.join(expected_sha).join(&wheel.filename);
    if store_path.is_file() {
        match validate_materialized_wheel(wheel, &store_path, "cached index wheel") {
            Ok(()) => return Ok(store_path),
            Err(err) => {
                tracing::warn!(
                    path = %store_path.display(),
                    error = %format!("{err:#}"),
                    "retread install: cached wheel hash mismatch; refetching locked URL"
                );
                let _ = std::fs::remove_file(&store_path);
            }
        }
    }

    let fetched = crate::wheel::fetch_wheel_cached(&url, Some(expected_sha), fetch_dir, store_root)
        .await
        .with_context(|| {
            format!(
                "retread install: fetching locked wheel {}=={} from {}",
                wheel.name, wheel.version, url
            )
        })?;
    if let Err(err) = validate_materialized_wheel(wheel, &fetched, "downloaded index wheel") {
        let _ = std::fs::remove_file(&fetched);
        let _ = std::fs::remove_file(&store_path);
        return Err(err);
    }
    tracing::info!(
        bundle = %lock.bundle,
        wheel = %wheel.filename,
        "retread install: fetched locked wheel bytes"
    );
    Ok(fetched)
}

async fn materialize_locked_wheels(
    lock: &RetreadLock,
    prefix: &Path,
    shipped_wheels_dir: &Path,
    store_root: &Path,
    conda_owned: &BTreeSet<(String, String)>,
    editable_owned: &BTreeSet<String>,
    drop_cuda_shadows: bool,
) -> Result<Vec<PathBuf>> {
    let fetch_dir = prefix
        .join("share")
        .join("retread")
        .join(&lock.bundle)
        .join("fetched");
    let store_candidates = built_wheel_store_candidates(lock, store_root);
    let mut files = Vec::with_capacity(lock.wheels.len());
    for wheel in &lock.wheels {
        // conda already owns this distribution at the exact locked version:
        // leave it to conda and keep it out of the uv replay entirely (not
        // even materialized), so uv never uninstalls the conda payload. See
        // `conda_owned_distributions` for why that uninstall is destructive.
        let normalized_name = normalize_dist_name(&wheel.name);
        if conda_owned.iter().any(|(name, version)| {
            name == &normalized_name && pep440_versions_equal(version, &wheel.version)
        }) {
            eprintln!(
                "retread install: {}=={} is conda-provided in the prefix; \
                 skipping wheel replay to avoid clobbering the conda payload",
                wheel.name, wheel.version
            );
            continue;
        }
        // The user has overlaid this distribution as an editable install
        // (`pip install -e`); replaying the bundled wheel would clobber their
        // working checkout. Skip on NAME match at ANY version -- the editable
        // overlay is the user's explicit local-development intent. See
        // `editable_owned_distributions`. To restore the bundled wheel:
        // `pip uninstall <name>` then relink the env.
        if editable_owned.contains(&normalize_dist_name(&wheel.name)) {
            tracing::info!(
                bundle = %lock.bundle,
                dist = %wheel.name,
                "retread install: {} is editable-installed (pip install -e) in the \
                 prefix; skipping bundled wheel replay to preserve the user's local \
                 checkout. To restore the bundled wheel: pip uninstall {} + relink the env",
                wheel.name,
                wheel.name
            );
            continue;
        }
        // conda owns the CUDA runtime in this prefix and this is a PyPI
        // `nvidia-*-cu<major>` lib-shim wheel: it duplicates a conda CUDA
        // library under a different distribution name (conda `nccl` vs PyPI
        // `nvidia-nccl-cu12`), so `conda_owned_distributions` -- which matches
        // by shared site-packages dist-info name -- cannot catch it. Left in
        // the env it SHADOWS the conda lib (conda torch's `_preload_cuda_deps`
        // globs `site-packages/nvidia/*/lib` and preloads the stale wheel),
        // breaking `import torch` with `undefined symbol: ncclAlltoAll`. Drop
        // it so conda's CUDA stack stays authoritative. See
        // `conda_owns_cuda_runtime` / `is_pypi_cuda_shadow_wheel`.
        if drop_cuda_shadows && is_pypi_cuda_shadow_wheel(&wheel.name) {
            eprintln!(
                "retread install: {}=={} is a PyPI CUDA lib-shim wheel shadowed by \
                 the conda CUDA stack in the prefix (conda-meta has cuda-version); \
                 skipping wheel replay so conda's CUDA libraries stay authoritative",
                wheel.name, wheel.version
            );
            continue;
        }
        let shipped = shipped_wheels_dir.join(&wheel.filename);
        if shipped.is_file() {
            validate_materialized_wheel(wheel, &shipped, "shipped wheel")?;
            files.push(shipped);
            continue;
        }

        match wheel.origin {
            Origin::Index => {
                files.push(materialize_index_wheel(lock, wheel, &fetch_dir, store_root).await?);
            }
            Origin::Built => {
                // Loose bundle mode: Built wheels are not shipped inside
                // the .conda -- their bytes were persisted to the shared
                // content-addressed wheel store at build time and the lock
                // records the sha256 lookup key. Verify before trusting.
                // Probe the primary store first, then self-heal from the
                // lock-recorded / default / legacy store locations (a hit
                // there means the paths diverged, not that bytes are lost).
                if let Some(expected) = wheel.sha256.as_deref() {
                    let mut found: Option<PathBuf> = None;
                    for (i, root) in store_candidates.iter().enumerate() {
                        let store_path = root.join(expected).join(&wheel.filename);
                        if !store_path.is_file() {
                            continue;
                        }
                        match validate_materialized_wheel(
                            wheel,
                            &store_path,
                            "built wheel-store entry",
                        ) {
                            Ok(()) => {
                                if i > 0 {
                                    tracing::warn!(
                                        wheel = %wheel.filename,
                                        primary = %store_root.display(),
                                        fallback = %root.display(),
                                        "retread install: built wheel missing from the \
                                         primary wheel store; self-healed from a fallback \
                                         store location (copying into the primary store)"
                                    );
                                    // Best-effort forward-heal so the next
                                    // install on this machine hits directly.
                                    if let Err(err) =
                                        crate::wheel::store_wheel_in_cache(&store_path, store_root)
                                            .await
                                    {
                                        tracing::warn!(
                                            error = %format!("{err:#}"),
                                            "retread install: could not copy built wheel \
                                             into the primary store; continuing from the \
                                             fallback location"
                                        );
                                    }
                                }
                                found = Some(store_path);
                                break;
                            }
                            Err(err) => {
                                tracing::warn!(
                                    path = %store_path.display(),
                                    error = %format!("{err:#}"),
                                    "retread install: wheel-store hash mismatch for built \
                                     wheel; removing corrupt entry"
                                );
                                let _ = std::fs::remove_file(&store_path);
                            }
                        }
                    }
                    if let Some(store_path) = found {
                        files.push(store_path);
                        continue;
                    }
                    bail!(
                        "retread install: locked built wheel {}=={} is neither shipped at {} \
                         nor present in the shared wheel store at {} (loose bundle mode; also \
                         probed: {}). The store entry was evicted or this machine does not \
                         share the build machine's wheel store; rebuild/reinstall the pack to \
                         re-populate it.",
                        wheel.name,
                        wheel.version,
                        shipped.display(),
                        store_candidates[0]
                            .join(expected)
                            .join(&wheel.filename)
                            .display(),
                        store_candidates[1..]
                            .iter()
                            .map(|p| p.display().to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
                bail!(
                    "retread install: locked wheel {}=={} is not present at {}. \
                     This wheel class is shipped inside the courier package (source-built, \
                     local-only, sdist-built, or relax-rewritten shadow) and cannot be \
                     recovered without pack rebuild/reinstall.",
                    wheel.name,
                    wheel.version,
                    shipped.display()
                );
            }
        }
    }
    Ok(files)
}

fn site_packages_dir(prefix: &Path, python: &str) -> PathBuf {
    prefix
        .join("lib")
        .join(format!("python{python}"))
        .join("site-packages")
}

fn parse_metadata_name_version(text: &str) -> Option<(String, String)> {
    let mut name: Option<String> = None;
    let mut version: Option<String> = None;

    for line in text.lines() {
        if line.trim().is_empty() {
            break;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        match key.trim().to_ascii_lowercase().as_str() {
            "name" => name = Some(normalize_dist_name(value)),
            "version" => version = Some(value.trim().to_string()),
            _ => {}
        }
        if name.is_some() && version.is_some() {
            break;
        }
    }

    Some((name?, version?))
}

fn read_metadata_file(path: &Path) -> Option<(String, String)> {
    let text = std::fs::read_to_string(path).ok()?;
    parse_metadata_name_version(&text)
}

fn installed_distributions(
    site_packages: &Path,
) -> Result<BTreeMap<String, BTreeMap<String, PathBuf>>> {
    let mut out: BTreeMap<String, BTreeMap<String, PathBuf>> = BTreeMap::new();
    if !site_packages.is_dir() {
        return Ok(out);
    }

    for entry in std::fs::read_dir(site_packages)
        .with_context(|| format!("reading site-packages {}", site_packages.display()))?
    {
        let entry =
            entry.with_context(|| format!("reading entry in {}", site_packages.display()))?;
        let path = entry.path();
        let file_name = entry.file_name().to_string_lossy().into_owned();
        let (metadata, dist_root) = if file_name.ends_with(".dist-info") {
            (path.join("METADATA"), path.clone())
        } else if file_name.ends_with(".egg-info") {
            if path.is_dir() {
                (path.join("PKG-INFO"), path.clone())
            } else {
                (path.clone(), path.clone())
            }
        } else {
            continue;
        };

        if let Some((name, version)) = read_metadata_file(&metadata) {
            out.entry(name).or_default().insert(version, dist_root);
        }
    }

    Ok(out)
}

/// Distributions in `site_packages` that conda already installed, keyed by
/// (PEP 503 name, version). Ownership is read from each dist-info's
/// `INSTALLER` marker (`conda` writes the literal string `conda`).
///
/// The courier must NOT hand a conda-owned distribution to `uv pip install`:
/// uv sees the bundle's PyPI wheel as a different distribution (direct-URL vs
/// the conda-recorded install) and uninstalls the conda one to replace it.
/// conda lays some payloads out differently than the wheel RECORD expects --
/// e.g. `pytorch` ships `torch/include/torch` as a *symlink* to the shared
/// `$PREFIX/include/torch`, so uv's uninstall aborts trying to `rmdir` it
/// ("Not a directory", ENOTDIR) after already deleting most of the tree,
/// gutting `torch` (no `__init__.py`, no dist-info) into an empty namespace
/// package. conda is authoritative for anything it already installed at the
/// locked version, so those wheels are dropped from the replay; `verify`
/// still passes because the conda dist-info + RECORD satisfy the payload
/// check.
fn conda_owned_distributions(site_packages: &Path) -> BTreeSet<(String, String)> {
    let mut out = BTreeSet::new();
    let Ok(installed) = installed_distributions(site_packages) else {
        return out;
    };
    for (name, versions) in &installed {
        for (version, dist_root) in versions {
            match std::fs::read_to_string(dist_root.join("INSTALLER")) {
                Ok(body) if body.trim() == "conda" => {
                    out.insert((name.clone(), version.clone()));
                }
                _ => {}
            }
        }
    }
    out
}

/// True if `name` is one of PyPI's `nvidia-<component>-cu<major>` CUDA-runtime
/// wheels (e.g. `nvidia-nccl-cu12`, `nvidia-cublas-cu12`,
/// `nvidia-cuda-runtime-cu12`). These wheels exist only to ship CUDA shared
/// libraries into `site-packages/nvidia/<component>/lib/` for a PyPI CUDA
/// framework (they are the transitive deps of the PyPI `torch`/`jax` wheels).
///
/// The name is anchored on a trailing `-cu<digits>` tag so genuine CUDA-adjacent
/// PyPI packages that are NOT lib-shim wheels are never matched:
/// `nvidia-ml-py`, `nvidia-cudnn-frontend`, etc. fall through.
fn is_pypi_cuda_shadow_wheel(name: &str) -> bool {
    let normalized = normalize_dist_name(name);
    let Some(rest) = normalized.strip_prefix("nvidia-") else {
        return false;
    };
    // rest = "<component>-cu<digits>", component itself may contain '-'
    // (e.g. "cuda-runtime"). Split on the LAST '-' to isolate the cuNN tag.
    match rest.rsplit_once('-') {
        Some((component, tag)) => {
            !component.is_empty()
                && tag
                    .strip_prefix("cu")
                    .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
        }
        None => false,
    }
}

/// True if conda owns the CUDA runtime in this prefix, signalled by the
/// conda-forge `cuda-version` metapackage (`conda-meta/cuda-version-*.json`).
/// Every conda CUDA install pins `cuda-version`, and conda `pytorch-gpu` /
/// `jaxlib` depend on it, so its presence means conda's CUDA shared libraries
/// (`$PREFIX/lib/libnccl.so`, `libcublas.so`, ...) are the authoritative stack.
///
/// When true, the PyPI `nvidia-*-cu<major>` shadow wheels
/// (`is_pypi_cuda_shadow_wheel`) must be kept out of the env: conda pytorch's
/// `_preload_cuda_deps` globs `site-packages/nvidia/*/lib/lib*.so*` and, if a
/// stale PyPI wheel is present (e.g. `nvidia-nccl-cu12==2.27.5` alongside conda
/// `nccl==2.30.7`), preloads the OLDER lib with `RTLD_GLOBAL` -- shadowing
/// conda's. `libtorch_cuda.so` then fails at load with
/// `undefined symbol: ncclAlltoAll` (a symbol only the newer conda nccl
/// exports). Dropping the shadow wheels lets the conda lib win.
fn conda_owns_cuda_runtime(prefix: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(prefix.join("conda-meta")) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry
            .file_name()
            .to_str()
            .is_some_and(|n| n.starts_with("cuda-version-") && n.ends_with(".json"))
    })
}

/// Distributions in `site_packages` the user has overlaid as an EDITABLE
/// install (`pip install -e <path>`), keyed by PEP 503 name only (version is
/// intentionally ignored -- see below).
///
/// A user who runs `pip install -e third_party/ProtoMotions --no-deps` is
/// expressing an explicit local-development intent: imports must resolve to
/// their working checkout, not the bundled wheel in site-packages. But the
/// courier replays the locked wheel set on EVERY activation self-heal, and
/// `uv pip install --reinstall <wheel>` would force the bundled wheel back
/// over the editable overlay -- the pack wheel would always win. So editable
/// overlays join the same replay skip-set as conda-owned distributions: the
/// courier never materializes or replays a wheel whose distribution the user
/// has made editable.
///
/// Detection (PEP 660 primary): the editable dist-info's `direct_url.json`
/// records `{"dir_info": {"editable": true}, ...}`. A legacy setuptools
/// `develop` install instead drops a `<name>.egg-link` file in site-packages;
/// that is trivially name-keyed, so it is accepted as a fallback signal.
///
/// Unlike conda ownership, this is keyed on NAME ONLY, not (name, version):
/// the user's checkout may declare any version (often a dev/local version that
/// differs from the locked wheel), and the overlay is authoritative regardless.
fn editable_owned_distributions(site_packages: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    if !site_packages.is_dir() {
        return out;
    }

    // PEP 660: dist-info/direct_url.json with dir_info.editable == true.
    if let Ok(installed) = installed_distributions(site_packages) {
        for (name, versions) in &installed {
            for dist_root in versions.values() {
                let Ok(body) = std::fs::read_to_string(dist_root.join("direct_url.json")) else {
                    continue;
                };
                let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) else {
                    continue;
                };
                if json
                    .get("dir_info")
                    .and_then(|d| d.get("editable"))
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
                {
                    out.insert(name.clone());
                }
            }
        }
    }

    // Legacy setuptools `develop`: a `<project>.egg-link` in site-packages.
    if let Ok(entries) = std::fs::read_dir(site_packages) {
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            if let Some(stem) = file_name.to_str().and_then(|n| n.strip_suffix(".egg-link")) {
                out.insert(normalize_dist_name(stem));
            }
        }
    }

    out
}

pub(crate) fn missing_locked_wheels_from_installed(
    lock: &RetreadLock,
    installed: &BTreeMap<String, BTreeMap<String, PathBuf>>,
) -> Vec<String> {
    let mut missing: Vec<String> = lock
        .wheels
        .iter()
        .filter_map(|wheel| {
            let name = normalize_dist_name(&wheel.name);
            let present = installed
                .get(&name)
                .and_then(|versions| installed_version_path(versions, &wheel.version))
                .is_some();
            (!present).then(|| format!("{}=={}", wheel.name, wheel.version))
        })
        .collect();
    missing.sort();
    missing.dedup();
    missing
}

fn record_path_token(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if trimmed.is_empty() {
        return None;
    }

    if let Some(rest) = trimmed.strip_prefix('"') {
        let mut out = String::new();
        let mut chars = rest.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    out.push('"');
                    continue;
                }
                break;
            }
            out.push(c);
        }
        (!out.is_empty()).then_some(out)
    } else {
        let raw = trimmed.split(',').next()?.trim();
        (!raw.is_empty()).then(|| raw.to_string())
    }
}

fn is_pycache_pyc_entry(token: &str) -> bool {
    token.ends_with(".pyc")
        && token
            .rsplit('/')
            .nth(1)
            .is_some_and(|dir| dir == "__pycache__")
}

fn verify_record_payload(site_packages: &Path, dist_root: &Path) -> Result<()> {
    if !dist_root
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|name| name.ends_with(".dist-info"))
    {
        bail!(
            "installed distribution metadata at {} has no wheel RECORD",
            dist_root.display()
        );
    }

    let record = dist_root.join("RECORD");
    let body = std::fs::read_to_string(&record)
        .with_context(|| format!("reading wheel RECORD {}", record.display()))?;
    let mut checked = 0usize;
    for line in body.lines() {
        let Some(token) = record_path_token(line) else {
            continue;
        };
        // Byte-compiled artifacts are interpreter-version-specific and
        // regenerated at runtime; a RECORD produced under a different Python
        // minor version references .pyc files with the wrong cpython tag.
        if is_pycache_pyc_entry(&token) {
            continue;
        }
        let path = site_packages.join(token);
        if !path.exists() {
            bail!(
                "wheel RECORD {} references missing installed file {}",
                record.display(),
                path.display()
            );
        }
        checked += 1;
    }
    if checked == 0 {
        bail!("wheel RECORD {} has no file entries", record.display());
    }
    Ok(())
}

fn verify_payload_installed(lock: &RetreadLock, prefix: &Path) -> Result<()> {
    let site_packages = site_packages_dir(prefix, &lock.python);
    let installed = installed_distributions(&site_packages)?;
    // An editable overlay (`pip install -e`) satisfies a locked distribution
    // at ANY version: the user has replaced the bundled wheel's dist-info with
    // the checkout's own (PEP 660 editables ship a real dist-info + RECORD, so
    // they would already pass the RECORD check -- but the recorded version may
    // differ from the lock, which would otherwise flag the locked version as
    // "missing" and trigger a clobbering reinstall). Names the user has made
    // editable are therefore counted present and skipped from the missing /
    // RECORD checks entirely.
    let editable_owned = editable_owned_distributions(&site_packages);
    let missing: Vec<String> = missing_locked_wheels_from_installed(lock, &installed)
        .into_iter()
        .filter(|item| {
            let name = item.split("==").next().map(normalize_dist_name);
            name.is_none_or(|n| !editable_owned.contains(&n))
        })
        .collect();
    if !missing.is_empty() {
        bail!(
            "retread verify: bundle {} is missing {} locked wheel(s) in {}: {}",
            lock.bundle,
            missing.len(),
            site_packages.display(),
            missing.join(", ")
        );
    }

    for wheel in &lock.wheels {
        let name = normalize_dist_name(&wheel.name);
        if editable_owned.contains(&name) {
            continue;
        }
        let dist_root = installed
            .get(&name)
            .and_then(|versions| installed_version_path(versions, &wheel.version))
            .expect("missing list already checked");
        verify_record_payload(&site_packages, dist_root).with_context(|| {
            format!(
                "retread verify: {}=={} payload check failed",
                wheel.name, wheel.version
            )
        })?;
    }
    Ok(())
}

fn marker_matches(marker: &Path, want: &str) -> bool {
    crate::glibc::marker_digest_matches(marker, want)
}

fn is_shared_library_record_path(path: &str) -> bool {
    Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| name.ends_with(".so") || name.contains(".so."))
}

fn installed_payload_libraries(
    lock: &RetreadLock,
    prefix: &Path,
) -> Result<(PathBuf, Vec<crate::glibc::PayloadLib>)> {
    let site_packages = site_packages_dir(prefix, &lock.python);
    let installed = installed_distributions(&site_packages)?;
    // Editable overlays own their own dist-info (possibly at a different
    // version than the lock) and their glibc/manylinux posture is the user's
    // to manage; the courier neither installed nor audits them. Skip so the
    // (name, version) lookup below does not error on the overlaid version.
    let editable_owned = editable_owned_distributions(&site_packages);
    let mut out: BTreeMap<String, crate::glibc::PayloadLib> = BTreeMap::new();
    for wheel in &lock.wheels {
        let name = normalize_dist_name(&wheel.name);
        if editable_owned.contains(&name) {
            continue;
        }
        let dist_root = installed
            .get(&name)
            .and_then(|versions| installed_version_path(versions, &wheel.version))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "retread audit: {}=={} is not installed in {}",
                    wheel.name,
                    wheel.version,
                    site_packages.display()
                )
            })?;
        let record = dist_root.join("RECORD");
        let body = std::fs::read_to_string(&record)
            .with_context(|| format!("reading wheel RECORD {}", record.display()))?;
        for line in body.lines() {
            let Some(token) = record_path_token(line) else {
                continue;
            };
            if !is_shared_library_record_path(&token) {
                continue;
            }
            let abs_path = site_packages.join(&token);
            out.entry(token.clone())
                .or_insert(crate::glibc::PayloadLib {
                    rel_path: token,
                    abs_path,
                });
        }
    }
    Ok((site_packages, out.into_values().collect()))
}

/// Verify that the marker belongs to this lock AND the target prefix still
/// contains the locked wheel payload. Used by activate.d and by `run()` before
/// trusting an existing marker.
pub fn verify(lock_path: &Path, prefix: &Path, full: bool) -> Result<()> {
    let (raw, lock) = read_validated_lock(lock_path)?;

    let share = prefix.join("share").join("retread");
    let marker = share.join(lock.marker_name());
    let want = lock_digest(&raw);
    let have = std::fs::read_to_string(&marker)
        .with_context(|| format!("reading marker {}", marker.display()))?;
    if have.lines().next().map(str::trim) != Some(want.as_str()) {
        bail!(
            "retread verify: marker {} does not match {}",
            marker.display(),
            lock_path.display()
        );
    }
    verify_payload_installed(&lock, prefix)?;
    let audit = crate::glibc::verify_marker_state(&lock, prefix, &have)?;
    if full {
        let (site_packages, libs) = installed_payload_libraries(&lock, prefix)?;
        let _ =
            crate::glibc::full_verify_audit(&lock, prefix, &site_packages, &libs, Some(&audit))?;
    }
    Ok(())
}

/// Build the `uv pip install` replay argument list (pure; no I/O, no spawn; no
/// argv[0]). The only install targets are explicit wheel files; `--no-deps` and
/// `--offline` make uv a wheel installer, not a resolver.
pub(crate) fn build_uv_replay_args(
    prefix: &Path,
    wheel_files: &[PathBuf],
    python_platform: Option<&str>,
    reinstall: bool,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        "pip".into(),
        "install".into(),
        "--python".into(),
        prefix.join("bin").join("python").into(),
    ];

    // Gated glibc / manylinux platform-tag relaxation.
    // uv derives manylinux compatibility from the host glibc. When the
    // workspace declares a higher glibc runtime contract, retread may retry
    // once with that exact declared `--python-platform`; None leaves uv's host
    // tag gate untouched.
    if let Some(plat) = python_platform {
        args.push("--python-platform".into());
        args.push(plat.into());
    }

    args.push("--no-deps".into());
    args.push("--offline".into());
    if reinstall {
        args.push("--reinstall".into());
    }

    for wheel in wheel_files {
        args.push(wheel.as_os_str().into());
    }

    args
}

/// Atomically replace the install marker (temp file + rename). The
/// lock-free fast path in `run()` reads the marker without holding the
/// install lock, so a plain `std::fs::write` (truncate-then-write) could
/// expose a torn marker to a concurrent reader; rename is atomic on POSIX.
fn write_marker_atomic(marker: &Path, body: &str) -> Result<()> {
    let tmp = marker.with_extension(format!("tmp.{}", std::process::id()));
    std::fs::write(&tmp, body).with_context(|| format!("writing marker temp {}", tmp.display()))?;
    std::fs::rename(&tmp, marker)
        .with_context(|| format!("renaming marker {} -> {}", tmp.display(), marker.display()))
}

/// Install (or no-op) the bundle described by `lock_path` into `prefix`.
pub async fn run(lock_path: &Path, prefix: &Path) -> Result<()> {
    let (raw, lock) = read_validated_lock(lock_path)?;

    let share = prefix.join("share").join("retread");
    std::fs::create_dir_all(&share).ok();

    let marker = share.join(lock.marker_name());
    let want = lock_digest(&raw);

    // ── Lock-free fast path ──────────────────────────────────────────────
    // The self-heal activate.d guard runs `retread install` on EVERY
    // activation, so parallel `pixi run`s into a healthy env used to
    // serialize on an exclusive install lock just to read the marker and
    // no-op. The check below is read-only (marker read + payload/audit
    // verification), and marker writes are atomic temp+rename, so a
    // lock-free reader sees either the old or the new marker, never a
    // torn one. Any mismatch (including a race with an in-flight writer)
    // just falls through to the locked slow path, which re-evaluates.
    if marker_matches(&marker, &want) && verify_payload_installed(&lock, prefix).is_ok() {
        let marker_text = std::fs::read_to_string(&marker).unwrap_or_default();
        if crate::glibc::verify_marker_state(&lock, prefix, &marker_text).is_ok() {
            let msg = format!("retread install: {} already current; skipping", lock.bundle);
            eprintln!("{msg}");
            crate::status::phase(
                lock_path.parent().unwrap_or(std::path::Path::new(".")),
                &lock.bundle,
                &msg,
            );
            return Ok(());
        }
    }

    // ── Locked slow path: repair or install is (probably) needed ────────
    // Concurrency semantics for two concurrent activations of the SAME
    // prefix + bundle: the second WAITS here on the per-bundle advisory
    // exclusive lock while the first installs, then re-evaluates the
    // marker below (double-checked locking) and NO-OPS via the "already
    // current" branch. The lock is per-bundle, so different bundles
    // installing into the same prefix don't serialize on it; their uv
    // invocations are serialized separately by the per-prefix uv guard
    // taken just before spawning uv. Best-effort: if the lock can't be
    // taken we proceed unlocked rather than block the repair.
    let _install_lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(share.join(format!(".{}.install.lock", lock.bundle)))
        .inspect(|f| {
            let _ = fs4::fs_std::FileExt::lock_exclusive(f);
        })
        .ok();
    let marker_current = marker_matches(&marker, &want);
    let mut force_reinstall = !marker_current;
    if marker_current {
        match verify_payload_installed(&lock, prefix) {
            Ok(()) => {
                let marker_text = std::fs::read_to_string(&marker).unwrap_or_default();
                match crate::glibc::verify_marker_state(&lock, prefix, &marker_text) {
                    Ok(_) => {
                        let msg =
                            format!("retread install: {} already current; skipping", lock.bundle);
                        eprintln!("{msg}");
                        crate::status::phase(
                            lock_path.parent().unwrap_or(std::path::Path::new(".")),
                            &lock.bundle,
                            &msg,
                        );
                        return Ok(());
                    }
                    Err(err) => {
                        eprintln!(
                            "retread install: {} marker exists but GLIBC audit verification failed; \
                             reapplying fixups/audit ({err:#})",
                            lock.bundle
                        );
                        let (site_packages, libs) = installed_payload_libraries(&lock, prefix)?;
                        let previous = crate::glibc::parse_marker_audit(&marker_text);
                        let previous_relaxed =
                            previous.as_ref().and_then(|p| p.relaxed_platform.clone());
                        let previous_declaration =
                            previous.as_ref().and_then(|p| p.declaration_source.clone());
                        let audit = crate::glibc::install_audit(
                            &lock,
                            prefix,
                            &site_packages,
                            &libs,
                            previous.as_ref(),
                            previous_relaxed,
                            previous_declaration,
                        )?;
                        write_marker_atomic(&marker, &crate::glibc::marker_body(&want, &audit)?)?;
                        let _ = std::fs::remove_file(share.join(format!("{}.broken", lock.bundle)));
                        let msg = format!("retread install: {} audit refreshed", lock.bundle);
                        eprintln!("{msg}");
                        crate::status::phase(
                            lock_path.parent().unwrap_or(std::path::Path::new(".")),
                            &lock.bundle,
                            &msg,
                        );
                        return Ok(());
                    }
                }
            }
            Err(err) => {
                // The digest marker is only trusted together with installed
                // wheel metadata. A payload miss invalidates it for replay, so
                // uv must replace the locked wheel set rather than skip
                // already-installed distributions.
                eprintln!(
                    "retread install: {} marker exists but payload verification failed; \
                     reinstalling ({err:#})",
                    lock.bundle
                );
                force_reinstall = true;
            }
        }
    }

    if lock.wheels.is_empty() {
        eprintln!(
            "retread install: {} has no locked wheels; nothing to do",
            lock.bundle
        );
        return Ok(());
    }

    let python = prefix.join("bin").join("python");
    if !python.exists() {
        bail!(
            "python not found at {} (post-link runs after the env python is linked)",
            python.display()
        );
    }
    // Shipped (built/changed) wheels live next to the lock, under the env.
    let wheels_dir = share.join(&lock.bundle).join("wheels");

    let uv: OsString = {
        let p = prefix.join("bin").join("uv");
        if p.exists() {
            p.into_os_string()
        } else {
            "uv".into()
        }
    };

    // conda-owned distributions are dropped from the replay: uv must never
    // uninstall a conda-managed distribution to swap in the bundle's PyPI
    // wheel (that clobbers the conda payload -- see
    // `conda_owned_distributions`). conda is authoritative at the locked
    // version and its dist-info still satisfies `verify`.
    let conda_owned = conda_owned_distributions(&site_packages_dir(prefix, &lock.python));
    // Editable overlays the user has installed on top of the bundle: the
    // courier must never replay the bundled wheel over a `pip install -e`
    // checkout. Dropped from the replay by name at any version.
    let editable_owned = editable_owned_distributions(&site_packages_dir(prefix, &lock.python));
    // When conda owns the CUDA runtime (conda-meta has `cuda-version`), the PyPI
    // `nvidia-*-cu<major>` lib-shim wheels are conda-shadowed duplicates that
    // break `import torch` if installed alongside conda's CUDA libs -- drop them
    // from the replay. See `conda_owns_cuda_runtime`.
    let drop_cuda_shadows = conda_owns_cuda_runtime(prefix);
    let wheel_files = materialize_locked_wheels(
        &lock,
        prefix,
        &wheels_dir,
        &resolve_wheel_store_root(&lock),
        &conda_owned,
        &editable_owned,
        drop_cuda_shadows,
    )
    .await?;

    let mut relaxed_platform: Option<String> = None;
    let mut declaration_source: Option<String> = None;

    if wheel_files.is_empty() {
        // Every locked wheel is already provided by conda in this prefix;
        // there is nothing for uv to replay. Fall through to verify/audit,
        // which the conda dist-infos satisfy.
        let msg = format!(
            "retread install: {} -> {} (0 uv wheels; all locked wheels are conda-provided)",
            lock.bundle,
            prefix.display()
        );
        eprintln!("{msg}");
        crate::status::phase(
            lock_path.parent().unwrap_or(std::path::Path::new(".")),
            &lock.bundle,
            &msg,
        );
    } else {
        let args = build_uv_replay_args(prefix, &wheel_files, None, force_reinstall);

        let install_msg = format!(
            "retread install: {} -> {} ({} wheels, zero-resolution replay)",
            lock.bundle,
            prefix.display(),
            wheel_files.len()
        );
        eprintln!("{install_msg}");
        crate::status::phase(
            lock_path.parent().unwrap_or(std::path::Path::new(".")),
            &lock.bundle,
            &install_msg,
        );

        // Per-PREFIX uv guard: the per-bundle install lock above only
        // serializes same-bundle installs, so two DIFFERENT bundles can reach
        // here concurrently for one prefix. uv mutates shared site-packages
        // state (RECORD files, console scripts, overlapping dist-infos)
        // non-transactionally, so serialize the uv invocation (and the
        // verify/audit that reads the freshly installed payload -- the guard
        // is held until this function returns). Best-effort, like the
        // install lock.
        let _uv_guard = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(share.join(".uv.install.lock"))
            .inspect(|f| {
                let _ = fs4::fs_std::FileExt::lock_exclusive(f);
            })
            .ok();

        let status = Command::new(&uv)
            .args(&args)
            .status()
            .with_context(|| format!("spawning uv ({uv:?})"))?;
        if !status.success() {
            // Relax on a manylinux platform-tag conflict only. Classify the failure
            // with a captured `--dry-run` resolve; if it is purely the platform tag
            // and a libc declaration authorizes a higher floor, retry once targeting
            // exactly that declaration. Any other failure surfaces unchanged.
            match relax_platform_on_conflict(&uv, &args, &lock)? {
                Some(outcome) => {
                    let relaxed = build_uv_replay_args(
                        prefix,
                        &wheel_files,
                        Some(&outcome.platform),
                        force_reinstall,
                    );
                    let status = Command::new(&uv)
                        .args(&relaxed)
                        .status()
                        .with_context(|| format!("spawning uv ({uv:?}) with relaxed platform"))?;
                    if !status.success() {
                        bail!(
                            "uv pip install failed for bundle {} even after relaxing the \
                             manylinux platform tag to {} (status {status})",
                            lock.bundle,
                            outcome.platform
                        );
                    }
                    relaxed_platform = Some(outcome.platform);
                    declaration_source = Some(outcome.declaration_source.to_string());
                }
                None => {
                    bail!(
                        "uv pip install failed for bundle {} (status {status})",
                        lock.bundle
                    );
                }
            }
        }
    }

    verify_payload_installed(&lock, prefix).with_context(|| {
        format!(
            "retread install: {} post-install verification failed",
            lock.bundle
        )
    })?;
    let (site_packages, libs) = installed_payload_libraries(&lock, prefix)?;
    let previous = crate::glibc::marker_audit(&marker);
    let audit = crate::glibc::install_audit(
        &lock,
        prefix,
        &site_packages,
        &libs,
        previous.as_ref(),
        relaxed_platform,
        declaration_source,
    )?;
    std::fs::create_dir_all(&share).ok();
    write_marker_atomic(&marker, &crate::glibc::marker_body(&want, &audit)?)?;
    let _ = std::fs::remove_file(share.join(format!("{}.broken", lock.bundle)));
    let done_msg = format!("retread install: {} installed", lock.bundle);
    eprintln!("{done_msg}");
    crate::status::phase(
        lock_path.parent().unwrap_or(std::path::Path::new(".")),
        &lock.bundle,
        &done_msg,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::lock::{CondaDep, LockWheel, Origin};

    #[test]
    fn cuda_shadow_wheel_matches_nvidia_lib_shims_only() {
        // The full PyPI CUDA-runtime wheel family torch pulls: all dropped.
        for name in [
            "nvidia-nccl-cu12",
            "nvidia-cublas-cu12",
            "nvidia-cuda-runtime-cu12",
            "nvidia-cuda-nvrtc-cu12",
            "nvidia-cuda-cupti-cu12",
            "nvidia-cudnn-cu12",
            "nvidia-cufft-cu12",
            "nvidia-cufile-cu12",
            "nvidia-curand-cu12",
            "nvidia-cusolver-cu12",
            "nvidia-cusparse-cu12",
            "nvidia-cusparselt-cu12",
            "nvidia-nvjitlink-cu12",
            "nvidia-nvshmem-cu12",
            "nvidia-nvtx-cu12",
            // major-version agnostic (cu11/cu13) and underscore-normalized forms
            "nvidia-nccl-cu11",
            "nvidia_nccl_cu13",
        ] {
            assert!(
                is_pypi_cuda_shadow_wheel(name),
                "expected {name} to be treated as a conda-shadowed CUDA lib-shim wheel"
            );
        }
        // NOT lib-shim wheels: must never be dropped.
        for name in [
            "nvidia-ml-py",          // pynvml bindings, pure python
            "nvidia-cudnn-frontend", // C++ header lib, no -cuNN tag
            "torch",
            "nccl", // the conda name, never appears as a wheel here
            "cupy-cuda12x",
            "nvidia",
        ] {
            assert!(
                !is_pypi_cuda_shadow_wheel(name),
                "expected {name} to be left installed"
            );
        }
    }

    fn lock_wheel(name: &str, version: &str) -> LockWheel {
        LockWheel {
            name: name.into(),
            version: version.into(),
            origin: Origin::Built,
            filename: format!("{}-{version}-py3-none-any.whl", name.replace('-', "_")),
            url: None,
            sha256: Some(hex_sha256(&test_wheel_bytes(name, version))),
            requires_dist: vec![],
            must_ship: true,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }
    }

    fn make_lock(
        conda_run_deps: Vec<CondaDep>,
        index_urls: Vec<String>,
        prerelease: BTreeMap<String, String>,
    ) -> RetreadLock {
        RetreadLock {
            schema: crate::lock::SCHEMA,
            retread_version: "2.0.0".into(),
            bundle: "test-bundle".into(),
            version: "1.0.0".into(),
            python: "3.11".into(),
            target_subdir: "linux-64".into(),
            inputs_hash: "abc".into(),
            root_requirements: vec!["mypackage==1.0.0".into()],
            wheels: vec![lock_wheel("mypackage", "1.0.0")],
            conda_run_deps,
            index_urls,
            prerelease,
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            resolution_glibc: None,
            conda_capable: vec![],
            entry_specs: vec![],
            wheel_store: None,
        }
    }

    #[test]
    fn write_marker_atomic_replaces_and_leaves_no_temp() {
        let dir =
            std::env::temp_dir().join(format!("retread-marker-atomic-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join("retread-test.installed");
        write_marker_atomic(&marker, "first\n").unwrap();
        assert_eq!(std::fs::read_to_string(&marker).unwrap(), "first\n");
        write_marker_atomic(&marker, "second\n").unwrap();
        assert_eq!(std::fs::read_to_string(&marker).unwrap(), "second\n");
        let leftovers: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains("tmp"))
            .collect();
        assert!(leftovers.is_empty(), "stray marker temps: {leftovers:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    fn marker_with_audit(digest: &str) -> String {
        let audit = crate::glibc::InstalledMarkerAudit {
            schema: 1,
            host_glibc: None,
            relaxed_platform: None,
            declaration_source: None,
            audit: crate::glibc::AuditStatus::Passed,
            fixups: Vec::new(),
            offenders: Vec::new(),
            file_cache: Vec::new(),
        };
        crate::glibc::marker_body(digest, &audit).unwrap()
    }

    fn argv_strings(args: &[OsString]) -> Vec<String> {
        args.iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    /// Collect all values for a given flag in a flat args vector.
    fn flag_values(args: &[String], flag: &str) -> Vec<String> {
        args.windows(2)
            .filter(|w| w[0] == flag)
            .map(|w| w[1].clone())
            .collect()
    }

    fn hex_sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn test_wheel_bytes(name: &str, version: &str) -> Vec<u8> {
        use std::io::{Cursor, Write as _};

        let dist = name.replace(['-', '.'], "_");
        let dist_info = format!("{dist}-{version}.dist-info");
        let mut cursor = Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(&mut cursor);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        zip.start_file(format!("{dist_info}/METADATA"), options)
            .unwrap();
        write!(
            zip,
            "Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n"
        )
        .unwrap();
        zip.start_file(format!("{dist_info}/WHEEL"), options)
            .unwrap();
        zip.write_all(b"Wheel-Version: 1.0\nTag: py3-none-any\n")
            .unwrap();
        zip.start_file(format!("{dist_info}/RECORD"), options)
            .unwrap();
        zip.finish().unwrap();
        cursor.into_inner()
    }

    fn index_lock_wheel(name: &str, version: &str, url: &str, sha256: &str) -> LockWheel {
        LockWheel {
            name: name.into(),
            version: version.into(),
            origin: Origin::Index,
            filename: crate::wheel::wheel_filename_from_url(&url::Url::parse(url).unwrap())
                .unwrap(),
            url: Some(url.into()),
            sha256: Some(sha256.into()),
            requires_dist: vec![],
            must_ship: false,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }
    }

    #[test]
    fn replay_args_are_explicit_wheel_files_only() {
        let prefix = PathBuf::from("/fake/prefix");
        let wheels = vec![
            PathBuf::from("/cache/a-1.0-py3-none-any.whl"),
            PathBuf::from("/prefix/share/retread/b/wheels/b-2.0-py3-none-any.whl"),
        ];
        let args = build_uv_replay_args(&prefix, &wheels, None, false);
        let strs = argv_strings(&args);

        assert_eq!(strs[0], "pip");
        assert_eq!(strs[1], "install");
        assert_eq!(strs[2], "--python");
        assert_eq!(strs[3], "/fake/prefix/bin/python");
        assert!(strs.contains(&"--no-deps".to_string()));
        assert!(strs.contains(&"--offline".to_string()));
        assert!(!strs.contains(&"--reinstall".to_string()));
        for forbidden in [
            "--index-url",
            "--extra-index-url",
            "--find-links",
            "--constraints",
            "--overrides",
            "--excludes",
            "--index-strategy",
        ] {
            assert!(
                !strs.contains(&forbidden.to_string()),
                "forbidden resolver flag {forbidden}"
            );
        }
        assert!(strs.contains(&"/cache/a-1.0-py3-none-any.whl".to_string()));
        assert!(
            strs.contains(&"/prefix/share/retread/b/wheels/b-2.0-py3-none-any.whl".to_string())
        );
        assert!(
            !strs.iter().any(|s| s == "mypackage==1.0.0"),
            "root requirements must not drive install replay"
        );
    }

    #[test]
    fn replay_args_preserve_python_platform_and_reinstall_only_when_requested() {
        let prefix = PathBuf::from("/fake/prefix");
        let wheels = vec![PathBuf::from("/cache/a-1.0-py3-none-any.whl")];
        let args = build_uv_replay_args(&prefix, &wheels, Some("x86_64-manylinux_2_35"), true);
        let strs = argv_strings(&args);

        assert_eq!(
            flag_values(&strs, "--python-platform"),
            vec!["x86_64-manylinux_2_35".to_string()]
        );
        assert!(strs.contains(&"--reinstall".to_string()));

        let args = build_uv_replay_args(&prefix, &wheels, None, false);
        let strs = argv_strings(&args);
        assert!(!strs.contains(&"--python-platform".to_string()));
        assert!(!strs.contains(&"--reinstall".to_string()));
    }

    #[test]
    fn validate_rejects_old_schema_and_index_without_hash() {
        let mut lock = make_lock(
            vec![],
            vec!["https://pypi.org/simple/".into()],
            BTreeMap::new(),
        );
        lock.schema = crate::lock::SCHEMA - 1;
        let err = validate_install_lock_for_platform(Path::new("/lock.json"), &lock, "linux-64")
            .unwrap_err();
        assert!(format!("{err:#}").contains("requires schema"));

        let mut lock = make_lock(
            vec![],
            vec!["https://pypi.org/simple/".into()],
            BTreeMap::new(),
        );
        lock.wheels = vec![LockWheel {
            name: "remote".into(),
            version: "1.0.0".into(),
            origin: Origin::Index,
            filename: "remote-1.0.0-py3-none-any.whl".into(),
            url: Some("https://example.com/remote-1.0.0-py3-none-any.whl".into()),
            sha256: None,
            requires_dist: vec![],
            must_ship: false,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }];
        let err = validate_install_lock_for_platform(Path::new("/lock.json"), &lock, "linux-64")
            .unwrap_err();
        assert!(format!("{err:#}").contains("has no final sha256"));
    }

    #[test]
    fn validate_install_lock_requires_native_target() {
        let path = Path::new("/lock.json");
        let mut lock = make_lock(vec![], vec![], BTreeMap::new());

        validate_install_lock_for_platform(path, &lock, "linux-64").unwrap();
        let err = format!(
            "{:#}",
            validate_install_lock_for_platform(path, &lock, "linux-aarch64").unwrap_err()
        );
        assert!(err.contains("targets linux-64"));
        assert!(err.contains("host is linux-aarch64"));

        lock.target_subdir = "linux-aarch64".into();
        lock.declared_glibc = Some("2.35".into());
        lock.resolution_glibc = Some("2.35".into());
        validate_install_lock_for_platform(path, &lock, "linux-aarch64").unwrap();
        assert!(validate_install_lock_for_platform(path, &lock, "linux-64").is_err());
    }

    #[test]
    fn validate_install_lock_rescores_every_arm_provenance() {
        let path = Path::new("/arm-lock.json");
        let mut lock = make_lock(vec![], vec![], BTreeMap::new());
        lock.target_subdir = "linux-aarch64".into();
        lock.declared_glibc = Some("2.35".into());
        lock.resolution_glibc = Some("2.35".into());

        lock.wheels[0].filename = "mypackage-1.0.0-cp311-cp311-manylinux_2_35_aarch64.whl".into();
        validate_install_lock_for_platform(path, &lock, "linux-aarch64").unwrap();

        lock.wheels[0].filename = "mypackage-1.0.0-cp311-cp311-manylinux_2_35_x86_64.whl".into();
        let err = format!(
            "{:#}",
            validate_install_lock_for_platform(path, &lock, "linux-aarch64").unwrap_err()
        );
        assert!(err.contains("incompatible"));

        lock.wheels[0].filename = "mypackage-1.0.0-not-a-wheel.whl".into();
        assert!(validate_install_lock_for_platform(path, &lock, "linux-aarch64").is_err());

        let filename = "remote-1.0.0-cp311-cp311-manylinux_2_35_x86_64.whl";
        lock.wheels = vec![index_lock_wheel(
            "remote",
            "1.0.0",
            &format!("https://example.com/{filename}"),
            &"0".repeat(64),
        )];
        let err = format!(
            "{:#}",
            validate_install_lock_for_platform(path, &lock, "linux-aarch64").unwrap_err()
        );
        assert!(err.contains("incompatible"));
    }

    #[test]
    fn validate_install_lock_rejects_malformed_target_and_hash_metadata() {
        let path = Path::new("/lock.json");
        let mut lock = make_lock(vec![], vec![], BTreeMap::new());
        lock.resolution_glibc = Some("2.35 trailing".into());
        let err = validate_install_lock_for_platform(path, &lock, "linux-64")
            .unwrap_err()
            .to_string();
        assert!(err.contains("malformed target metadata"));

        lock.resolution_glibc = None;
        lock.wheels[0].sha256 = Some("abc".into());
        let err = format!(
            "{:#}",
            validate_install_lock_for_platform(path, &lock, "linux-64").unwrap_err()
        );
        assert!(err.contains("invalid final sha256"));

        lock.wheels[0].sha256 = Some("0".repeat(64));
        lock.bundle = "../escape".into();
        let err = validate_install_lock_for_platform(path, &lock, "linux-64")
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid bundle path component"));
    }

    #[test]
    fn validate_install_lock_binds_filename_and_url_identity() {
        let path = Path::new("/lock.json");
        let mut lock = make_lock(vec![], vec![], BTreeMap::new());
        lock.wheels[0].filename = "evil-1.0.0-py3-none-any.whl".into();
        let err = format!(
            "{:#}",
            validate_install_lock_for_platform(path, &lock, "linux-64").unwrap_err()
        );
        assert!(err.contains("records distribution `evil`"));

        lock.wheels[0].filename = "mypackage-9.9-py3-none-any.whl".into();
        let err = format!(
            "{:#}",
            validate_install_lock_for_platform(path, &lock, "linux-64").unwrap_err()
        );
        assert!(err.contains("records version `9.9`"));

        lock.wheels = vec![index_lock_wheel(
            "remote",
            "1.0.0",
            "https://example.com/evil-9.9-py3-none-any.whl",
            &"0".repeat(64),
        )];
        lock.wheels[0].filename = "remote-1.0.0-py3-none-any.whl".into();
        let err = format!(
            "{:#}",
            validate_install_lock_for_platform(path, &lock, "linux-64").unwrap_err()
        );
        assert!(
            err.contains("artifact URL") && err.contains("distribution `evil`"),
            "unexpected identity error: {err}",
        );

        lock.wheels = vec![LockWheel {
            name: "demo".into(),
            version: "1.0".into(),
            origin: Origin::Built,
            filename: "demo-1.0-999retread-cp311-cp311-manylinux_2_17_x86_64.whl".into(),
            url: None,
            sha256: Some("12".repeat(32)),
            requires_dist: vec![],
            must_ship: false,
            upstream_url: Some(
                "https://example.com/demo-1.0-1-cp311-cp311-manylinux_2_28_x86_64.whl".into(),
            ),
            git_source: None,
            sdist_source: None,
        }];
        let err = validate_install_lock_for_platform(path, &lock, "linux-64").unwrap_err();
        assert!(format!("{err:#}").contains("different wheel provenance"));

        lock.wheels[0].filename =
            "demo-1.0-999retread-cp311-cp311-manylinux_2_17_x86_64.whl".into();
        lock.wheels[0].url =
            Some("https://example.com/demo-1.0-1-cp311-cp311-manylinux_2_17_x86_64.whl".into());
        lock.wheels[0].upstream_url =
            Some("https://example.com/demo-1.0-2-cp311-cp311-manylinux_2_17_x86_64.whl".into());
        let err = validate_install_lock_for_platform(path, &lock, "linux-64").unwrap_err();
        assert!(format!("{err:#}").contains("unexpectedly records an index URL"));

        lock.wheels[0].url = None;
        lock.wheels[0].upstream_url = Some(
            "https://example.com/demo-1.0-1%2F..%2F..%2Fevil-cp311-cp311-manylinux_2_17_x86_64.whl"
                .into(),
        );
        let err = validate_install_lock_for_platform(path, &lock, "linux-64").unwrap_err();
        assert!(format!("{err:#}").contains("single wheel basename"));

        lock.wheels[0] = lock_wheel("my-package", "1.0");
        lock.wheels[0].filename = "My_Package-1.0.0-py3-none-any.whl".into();
        validate_install_lock_for_platform(path, &lock, "linux-64").unwrap();

        let first = index_lock_wheel(
            "remote",
            "1.0.0",
            "https://example.com/remote-1.0.0-py3-none-any.whl",
            &"a".repeat(64),
        );
        let mut second = first.clone();
        second.sha256 = Some("b".repeat(64));
        lock.wheels = vec![first, second];
        let err = validate_install_lock_for_platform(path, &lock, "linux-64").unwrap_err();
        assert!(format!("{err:#}").contains("duplicate distributions"));
    }

    #[test]
    fn read_validated_lock_canonicalizes_python_patch_for_runtime_paths() {
        let root = tempdir("python-patch-lock");
        let mut lock = make_lock(vec![], vec![], BTreeMap::new());
        lock.python = "3.11.0".into();
        let target = lock.resolution_target().unwrap();
        let lock_path = root.join(RetreadLock::file_name_for_target(&lock.bundle, &target));
        std::fs::write(&lock_path, lock.to_pretty_json().unwrap()).unwrap();

        let (_, decoded) = read_validated_lock(&lock_path).unwrap();
        assert_eq!(decoded.python, "3.11");
        assert_eq!(
            site_packages_dir(Path::new("/prefix"), &decoded.python),
            PathBuf::from("/prefix/lib/python3.11/site-packages")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn legacy_lock_basename_is_linux_64_only() {
        let mut lock = make_lock(vec![], vec![], BTreeMap::new());
        let legacy = Path::new("retread-demo.lock.json");
        let linux = lock.resolution_target().unwrap();
        assert!(lock_basename_matches_target(legacy, "demo", &linux));

        lock.target_subdir = "linux-aarch64".into();
        let arm = lock.resolution_target().unwrap();
        assert!(!lock_basename_matches_target(legacy, "demo", &arm));
        let qualified = RetreadLock::file_name_for_target("demo", &arm);
        assert!(lock_basename_matches_target(
            Path::new(&qualified),
            "demo",
            &arm,
        ));
    }

    #[test]
    fn legacy_missing_target_remains_linux_64_only() {
        let path = Path::new("/legacy-lock.json");
        let lock = make_lock(vec![], vec![], BTreeMap::new());
        let mut json = serde_json::to_value(lock).unwrap();
        json.as_object_mut().unwrap().remove("target_subdir");
        let legacy: RetreadLock = serde_json::from_value(json).unwrap();

        assert_eq!(legacy.target_subdir, "linux-64");
        validate_install_lock_for_platform(path, &legacy, "linux-64").unwrap();
        assert!(validate_install_lock_for_platform(path, &legacy, "linux-aarch64").is_err());
    }

    #[test]
    fn read_validated_lock_reports_rebuild_for_missing_or_corrupt_lock() {
        let root = tempdir("invalid-lock");
        let missing = root.join("missing.lock.json");
        let err = read_validated_lock(&missing).unwrap_err();
        assert!(format!("{err:#}").contains("Rebuild the courier pack"));

        let corrupt = root.join("corrupt.lock.json");
        std::fs::write(&corrupt, b"not json").unwrap();
        let err = read_validated_lock(&corrupt).unwrap_err();
        assert!(format!("{err:#}").contains("Rebuild the courier pack"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn materialize_index_wheel_uses_sha_cache_without_network() {
        let root = tempdir("index-cache");
        let store_root = root.join("store");
        let fetch_dir = root.join("fetch");
        let bytes = test_wheel_bytes("remote", "1.0.0");
        let sha = hex_sha256(&bytes);
        let filename = "remote-1.0.0-py3-none-any.whl";
        let cached = store_root.join(&sha).join(filename);
        std::fs::create_dir_all(cached.parent().unwrap()).unwrap();
        std::fs::write(&cached, &bytes).unwrap();

        let lock = make_lock(vec![], vec![], BTreeMap::new());
        let wheel = index_lock_wheel(
            "remote",
            "1.0.0",
            &format!("http://127.0.0.1:9/{filename}"),
            &sha,
        );
        let path = materialize_index_wheel(&lock, &wheel, &fetch_dir, &store_root)
            .await
            .unwrap();
        assert_eq!(path, cached);
        assert!(
            !fetch_dir.exists(),
            "cache hit must not create fetch dir or touch network"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn materialize_locked_wheels_uses_shipped_file_offline() {
        let root = tempdir("offline-shipped");
        let prefix = root.join("prefix");
        let wheels_dir = root.join("wheels");
        let bytes = test_wheel_bytes("remote", "1.0.0");
        let sha = hex_sha256(&bytes);
        let filename = "remote-1.0.0-py3-none-any.whl";
        let shipped = wheels_dir.join(filename);
        std::fs::create_dir_all(&wheels_dir).unwrap();
        std::fs::write(&shipped, &bytes).unwrap();

        let mut lock = make_lock(vec![], vec![], BTreeMap::new());
        lock.wheels = vec![index_lock_wheel(
            "remote",
            "1.0.0",
            &format!("http://127.0.0.1:9/{filename}"),
            &sha,
        )];

        let files = materialize_locked_wheels(
            &lock,
            &prefix,
            &wheels_dir,
            &root.join("cache"),
            &BTreeSet::new(),
            &BTreeSet::new(),
            false,
        )
        .await
        .unwrap();
        assert_eq!(files, vec![shipped]);
        assert!(
            !prefix
                .join("share")
                .join("retread")
                .join(&lock.bundle)
                .join("fetched")
                .exists(),
            "shipped offline replay must not create fetch dir"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// Loose bundle mode: an Origin::Built wheel that is NOT shipped inside
    /// the package must be materialized from the shared wheel store by its
    /// lock sha256, hash-verified, without touching the network.
    #[tokio::test]
    async fn materialize_locked_wheels_serves_built_wheel_from_store() {
        let root = tempdir("loose-built-store");
        let prefix = root.join("prefix");
        let wheels_dir = root.join("wheels"); // shipped dir: intentionally empty
        let store_root = root.join("store");
        let bytes = test_wheel_bytes("builtpkg", "1.0.0");
        let sha = hex_sha256(&bytes);
        let filename = "builtpkg-1.0.0-py3-none-any.whl";
        let stored = store_root.join(&sha).join(filename);
        std::fs::create_dir_all(stored.parent().unwrap()).unwrap();
        std::fs::write(&stored, &bytes).unwrap();

        let mut lock = make_lock(vec![], vec![], BTreeMap::new());
        lock.wheels = vec![LockWheel {
            name: "builtpkg".into(),
            version: "1.0.0".into(),
            origin: Origin::Built,
            filename: filename.into(),
            url: None,
            sha256: Some(sha),
            requires_dist: vec![],
            must_ship: true,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }];

        let files = materialize_locked_wheels(
            &lock,
            &prefix,
            &wheels_dir,
            &store_root,
            &BTreeSet::new(),
            &BTreeSet::new(),
            false,
        )
        .await
        .unwrap();
        assert_eq!(files, vec![stored]);
    }

    #[tokio::test]
    async fn materialization_rejects_shipped_and_store_metadata_mismatch() {
        let root = tempdir("materialized-identity-mismatch");
        let prefix = root.join("prefix");
        let wheels_dir = root.join("wheels");
        std::fs::create_dir_all(&wheels_dir).unwrap();

        let shipped = wheels_dir.join("mypackage-1.0.0-py3-none-any.whl");
        std::fs::write(&shipped, test_wheel_bytes("evil", "9.9")).unwrap();
        let lock = make_lock(vec![], vec![], BTreeMap::new());
        let err = materialize_locked_wheels(
            &lock,
            &prefix,
            &wheels_dir,
            &root.join("store"),
            &BTreeSet::new(),
            &BTreeSet::new(),
            false,
        )
        .await
        .unwrap_err();
        assert!(format!("{err:#}").contains("records distribution `evil`"));

        std::fs::remove_file(&shipped).unwrap();
        let bytes = test_wheel_bytes("evil", "9.9");
        let sha = hex_sha256(&bytes);
        let filename = "builtpkg-1.0.0-py3-none-any.whl";
        let stored = root.join("store").join(&sha).join(filename);
        std::fs::create_dir_all(stored.parent().unwrap()).unwrap();
        std::fs::write(&stored, bytes).unwrap();
        let mut lock = make_lock(vec![], vec![], BTreeMap::new());
        lock.wheels = vec![LockWheel {
            name: "builtpkg".into(),
            version: "1.0.0".into(),
            origin: Origin::Built,
            filename: filename.into(),
            url: None,
            sha256: Some(sha),
            requires_dist: vec![],
            must_ship: true,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }];
        assert!(
            materialize_locked_wheels(
                &lock,
                &prefix,
                &wheels_dir,
                &root.join("store"),
                &BTreeSet::new(),
                &BTreeSet::new(),
                false,
            )
            .await
            .is_err()
        );
        assert!(!stored.exists(), "invalid store entry must be evicted");
        let _ = std::fs::remove_dir_all(root);
    }

    /// A dist-info whose INSTALLER marker reads `conda` is reported as
    /// conda-owned; anything else (uv, missing marker) is not.
    #[test]
    fn conda_owned_distributions_reads_installer_marker() {
        let root = tempdir("conda-owned");
        let sp = root.join("lib/python3.11/site-packages");
        let write_dist = |name: &str, ver: &str, installer: Option<&str>| {
            let di = sp.join(format!("{name}-{ver}.dist-info"));
            std::fs::create_dir_all(&di).unwrap();
            std::fs::write(
                di.join("METADATA"),
                format!("Name: {name}\nVersion: {ver}\n"),
            )
            .unwrap();
            if let Some(i) = installer {
                std::fs::write(di.join("INSTALLER"), i).unwrap();
            }
        };
        // conda-owned (note the trailing newline conda writes)
        write_dist("torch", "2.7.0", Some("conda\n"));
        // uv-owned
        write_dist("tensordict", "0.9.0", Some("uv\n"));
        // no INSTALLER at all
        write_dist("orphan", "1.0.0", None);

        let owned = conda_owned_distributions(&sp);
        assert!(owned.contains(&("torch".into(), "2.7.0".into())));
        assert!(!owned.contains(&("tensordict".into(), "0.9.0".into())));
        assert!(!owned.contains(&("orphan".into(), "1.0.0".into())));
        assert_eq!(owned.len(), 1);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A locked wheel whose (name, version) conda already owns in the prefix
    /// is dropped from the replay entirely -- never materialized, so uv is
    /// never asked to uninstall the conda payload. Non-owned wheels are
    /// still materialized.
    #[tokio::test]
    async fn materialize_locked_wheels_skips_conda_owned_wheel() {
        let root = tempdir("skip-conda-owned");
        let prefix = root.join("prefix");
        let wheels_dir = root.join("wheels");
        std::fs::create_dir_all(&wheels_dir).unwrap();

        // Ship both wheels so materialization is a pure offline file lookup.
        let torch_bytes = test_wheel_bytes("torch", "2.7.0");
        let td_bytes = test_wheel_bytes("tensordict", "0.9.0");
        std::fs::write(
            wheels_dir.join("torch-2.7.0-py3-none-any.whl"),
            &torch_bytes,
        )
        .unwrap();
        let td_shipped = wheels_dir.join("tensordict-0.9.0-py3-none-any.whl");
        std::fs::write(&td_shipped, &td_bytes).unwrap();

        let mut lock = make_lock(vec![], vec![], BTreeMap::new());
        lock.wheels = vec![
            // PEP 440-equivalent spelling must still preserve conda ownership.
            lock_wheel("torch", "2.7.0.0"),
            lock_wheel("tensordict", "0.9.0"),
        ];

        let mut conda_owned = BTreeSet::new();
        conda_owned.insert(("torch".to_string(), "2.7.0".to_string()));

        let files = materialize_locked_wheels(
            &lock,
            &prefix,
            &wheels_dir,
            &root.join("cache"),
            &conda_owned,
            &BTreeSet::new(),
            false,
        )
        .await
        .unwrap();

        // Only the non-owned wheel survives; torch is left to conda.
        assert_eq!(files, vec![td_shipped]);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Helper: write a dist-info with METADATA and an optional
    /// direct_url.json. `editable` controls the PEP 660 `dir_info.editable`
    /// flag; `None` writes no direct_url.json at all.
    fn write_dist_info(sp: &Path, name: &str, ver: &str, editable: Option<bool>) -> PathBuf {
        let di = sp.join(format!("{name}-{ver}.dist-info"));
        std::fs::create_dir_all(&di).unwrap();
        std::fs::write(
            di.join("METADATA"),
            format!("Name: {name}\nVersion: {ver}\n"),
        )
        .unwrap();
        if let Some(is_editable) = editable {
            let body = if is_editable {
                r#"{"url": "file:///home/u/checkout", "dir_info": {"editable": true}}"#
            } else {
                // A regular directory/url install: dir_info present but not editable.
                r#"{"url": "file:///home/u/checkout", "dir_info": {}}"#
            };
            std::fs::write(di.join("direct_url.json"), body).unwrap();
        }
        di
    }

    /// A dist-info whose direct_url.json carries `dir_info.editable == true`
    /// (PEP 660) is reported editable-owned; a non-editable url install is
    /// NOT; and a legacy setuptools `.egg-link` is accepted by name.
    #[test]
    fn editable_owned_distributions_reads_direct_url_and_egg_link() {
        let root = tempdir("editable-owned");
        let sp = root.join("lib/python3.11/site-packages");
        std::fs::create_dir_all(&sp).unwrap();

        // (a) PEP 660 editable overlay.
        write_dist_info(&sp, "protomotions", "0.1.0", Some(true));
        // (b) non-editable url/dir install -> must NOT be treated as editable.
        write_dist_info(&sp, "tensordict", "0.9.0", Some(false));
        // a plain wheel install with no direct_url.json at all.
        write_dist_info(&sp, "orphan", "1.0.0", None);
        // legacy setuptools develop overlay (case-folded name).
        std::fs::write(sp.join("Legacy_Pkg.egg-link"), "/home/u/legacy\n.").unwrap();

        let owned = editable_owned_distributions(&sp);
        assert!(owned.contains("protomotions"), "PEP 660 editable detected");
        assert!(
            !owned.contains("tensordict"),
            "non-editable url install must not be editable-owned"
        );
        assert!(!owned.contains("orphan"), "plain wheel is not editable");
        assert!(
            owned.contains(&normalize_dist_name("Legacy_Pkg")),
            "legacy .egg-link detected by name"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// A locked wheel whose distribution the user has overlaid as an editable
    /// install is dropped from the replay by NAME, even when the editable's
    /// version differs from the lock -- the overlay is authoritative at any
    /// version. Non-overlaid wheels are still materialized.
    #[tokio::test]
    async fn materialize_locked_wheels_skips_editable_owned_wheel() {
        let root = tempdir("skip-editable-owned");
        let prefix = root.join("prefix");
        let wheels_dir = root.join("wheels");
        std::fs::create_dir_all(&wheels_dir).unwrap();

        // Ship both wheels so materialization is a pure offline file lookup.
        std::fs::write(
            wheels_dir.join("protomotions-2.0.0-py3-none-any.whl"),
            test_wheel_bytes("protomotions", "2.0.0"),
        )
        .unwrap();
        let td_shipped = wheels_dir.join("tensordict-0.9.0-py3-none-any.whl");
        std::fs::write(&td_shipped, test_wheel_bytes("tensordict", "0.9.0")).unwrap();

        let mut lock = make_lock(vec![], vec![], BTreeMap::new());
        // Lock pins protomotions==2.0.0; the user's editable is a DIFFERENT
        // version (0.1.0.dev0) -- must still be skipped on name match.
        lock.wheels = vec![
            lock_wheel("protomotions", "2.0.0"),
            lock_wheel("tensordict", "0.9.0"),
        ];

        let mut editable_owned = BTreeSet::new();
        editable_owned.insert("protomotions".to_string());

        let files = materialize_locked_wheels(
            &lock,
            &prefix,
            &wheels_dir,
            &root.join("cache"),
            &BTreeSet::new(),
            &editable_owned,
            false,
        )
        .await
        .unwrap();

        // Only tensordict survives; the editable protomotions is left alone.
        assert_eq!(files, vec![td_shipped]);
        let _ = std::fs::remove_dir_all(root);
    }

    /// verify_payload_installed treats an editable overlay as satisfying the
    /// locked distribution even when the checkout's version differs from the
    /// lock -- no "missing wheel" bail, no RECORD requirement.
    #[test]
    fn verify_payload_installed_accepts_editable_overlay_at_any_version() {
        let root = tempdir("verify-editable");
        let prefix = root.join("prefix");
        let sp = site_packages_dir(&prefix, "3.11");
        std::fs::create_dir_all(&sp).unwrap();

        // The lock pins mypackage==1.0.0 (make_lock default). The user has
        // overlaid an editable checkout that declares 9.9.9.dev0 and ships no
        // wheel RECORD; verify must still pass.
        write_dist_info(&sp, "mypackage", "9.9.9.dev0", Some(true));

        let lock = make_lock(vec![], vec![], BTreeMap::new());
        assert_eq!(lock.wheels[0].version, "1.0.0");
        verify_payload_installed(&lock, &prefix)
            .expect("editable overlay at any version satisfies verify");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn verify_payload_installed_accepts_pep440_equivalent_version() {
        let root = tempdir("verify-equivalent-version");
        let prefix = root.join("prefix");
        let sp = site_packages_dir(&prefix, "3.11");
        std::fs::create_dir_all(&sp).unwrap();
        let dist_info = write_dist_info(&sp, "mypackage", "1.0", None);
        std::fs::write(sp.join("mypackage.py"), "value = 1\n").unwrap();
        std::fs::write(
            dist_info.join("RECORD"),
            "mypackage.py,,\nmypackage-1.0.dist-info/RECORD,,\n",
        )
        .unwrap();

        let lock = make_lock(vec![], vec![], BTreeMap::new());
        assert_eq!(lock.wheels[0].version, "1.0.0");
        assert!(
            missing_locked_wheels_from_installed(&lock, &installed_distributions(&sp).unwrap(),)
                .is_empty()
        );
        verify_payload_installed(&lock, &prefix)
            .expect("PEP 440-equivalent installed version satisfies verify");
        let _ = std::fs::remove_dir_all(root);
    }

    /// Loose mode failure story: a Built wheel with a sha but an EMPTY store
    /// must fail loudly with the rebuild hint (no silent fallthrough).
    #[tokio::test]
    async fn materialize_locked_wheels_built_wheel_missing_from_store_errors() {
        let root = tempdir("loose-built-miss");
        let prefix = root.join("prefix");
        let wheels_dir = root.join("wheels");
        let store_root = root.join("store");

        let mut lock = make_lock(vec![], vec![], BTreeMap::new());
        lock.wheels = vec![LockWheel {
            name: "builtpkg".into(),
            version: "1.0.0".into(),
            origin: Origin::Built,
            filename: "builtpkg-1.0.0-py3-none-any.whl".into(),
            url: None,
            sha256: Some("0".repeat(64)),
            requires_dist: vec![],
            must_ship: true,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }];

        let err = materialize_locked_wheels(
            &lock,
            &prefix,
            &wheels_dir,
            &store_root,
            &BTreeSet::new(),
            &BTreeSet::new(),
            false,
        )
        .await
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("shared wheel store") && msg.contains("rebuild"),
            "error must explain the loose-mode store miss: {msg}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// THE fast-tmp divergence regression: the build side resolves the wheel
    /// store via `courier::wheel_store_root_with` and the install side via
    /// `resolve_wheel_store_root_with`. Under fast-tmp env (RETREAD_CACHE_DIR
    /// redirected into a job-local tmp namespace) both must still resolve to
    /// the SAME shared store — the redirect must not move the store.
    #[test]
    fn build_and_install_store_resolution_identical_under_fast_tmp_env() {
        let fasttmp_env = |key: &str| -> Option<String> {
            match key {
                // What fasttmp::desired_env_pairs exports in a SLURM job.
                "RETREAD_CACHE_DIR" => {
                    Some("/tmp/retread-user/abc123/job-99/caches/retread".into())
                }
                "PIXI_CACHE_DIR" | "RATTLER_CACHE_DIR" | "UV_CACHE_DIR" => {
                    Some("/tmp/retread-user/abc123/job-99/caches/x".into())
                }
                "HOME" => Some("/users/tester".into()),
                _ => None,
            }
        };
        let build_side = crate::courier::wheel_store_root_with(&fasttmp_env);
        let lock = make_lock(vec![], vec![], BTreeMap::new());
        let install_side = resolve_wheel_store_root_with(&fasttmp_env, &lock);
        assert_eq!(
            build_side, install_side,
            "build-store and courier-store must resolve identically under fast-tmp env"
        );
        assert_eq!(
            build_side,
            PathBuf::from("/users/tester/.cache/retread/wheels"),
            "the store must stay in the shared per-user cache, not job-local tmp"
        );
        assert!(
            !build_side.starts_with("/tmp"),
            "fast-tmp RETREAD_CACHE_DIR redirect must never relocate the wheel store"
        );
    }

    /// Install-side precedence: RETREAD_WHEEL_STORE env > lock-recorded
    /// store (with `~` expanded against the local HOME) > shared default.
    #[test]
    fn install_store_resolution_precedence_env_lock_default() {
        let env_with_override = |key: &str| -> Option<String> {
            match key {
                "RETREAD_WHEEL_STORE" => Some("/shared/project/wheel-store".into()),
                "HOME" => Some("/users/tester".into()),
                _ => None,
            }
        };
        let env_plain = |key: &str| -> Option<String> {
            match key {
                "HOME" => Some("/users/tester".into()),
                _ => None,
            }
        };

        let mut lock = make_lock(vec![], vec![], BTreeMap::new());
        lock.wheel_store = Some("/nfs/build-host/wheel-store".into());

        assert_eq!(
            resolve_wheel_store_root_with(&env_with_override, &lock),
            PathBuf::from("/shared/project/wheel-store"),
            "env override must win over the lock-recorded store"
        );
        assert_eq!(
            resolve_wheel_store_root_with(&env_plain, &lock),
            PathBuf::from("/nfs/build-host/wheel-store"),
            "lock-recorded store must win over the default"
        );
        lock.wheel_store = None;
        assert_eq!(
            resolve_wheel_store_root_with(&env_plain, &lock),
            PathBuf::from("/users/tester/.cache/retread/wheels"),
            "no override + no lock record must fall back to the shared default"
        );
    }

    /// Store-path mismatch self-heal: a Built wheel absent from the primary
    /// store but present in the LOCK-RECORDED store must install (with the
    /// bytes forward-copied into the primary store), not hard-fail.
    #[tokio::test]
    async fn materialize_locked_wheels_built_wheel_heals_from_lock_recorded_store() {
        let root = tempdir("loose-built-heal");
        let prefix = root.join("prefix");
        let wheels_dir = root.join("wheels"); // shipped dir: intentionally empty
        let primary_store = root.join("primary-store"); // intentionally empty
        let recorded_store = root.join("recorded-store");
        let bytes = test_wheel_bytes("builtpkg", "1.0.0");
        let sha = hex_sha256(&bytes);
        let filename = "builtpkg-1.0.0-py3-none-any.whl";
        let recorded_path = recorded_store.join(&sha).join(filename);
        std::fs::create_dir_all(recorded_path.parent().unwrap()).unwrap();
        std::fs::write(&recorded_path, &bytes).unwrap();

        let mut lock = make_lock(vec![], vec![], BTreeMap::new());
        lock.wheel_store = Some(recorded_store.display().to_string());
        lock.wheels = vec![LockWheel {
            name: "builtpkg".into(),
            version: "1.0.0".into(),
            origin: Origin::Built,
            filename: filename.into(),
            url: None,
            sha256: Some(sha.clone()),
            requires_dist: vec![],
            must_ship: true,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }];

        let files = materialize_locked_wheels(
            &lock,
            &prefix,
            &wheels_dir,
            &primary_store,
            &BTreeSet::new(),
            &BTreeSet::new(),
            false,
        )
        .await
        .unwrap();
        assert_eq!(
            files,
            vec![recorded_path],
            "self-heal must serve the wheel from the lock-recorded store"
        );
        assert!(
            primary_store.join(&sha).join(filename).is_file(),
            "self-heal must forward-copy the wheel into the primary store"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn materialize_index_wheel_fetches_locked_url_and_verifies_hash() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let root = tempdir("index-fetch");
        let store_root = root.join("store");
        let fetch_dir = root.join("fetch");
        let filename = "remote-1.0.0-py3-none-any.whl";
        let bytes = test_wheel_bytes("remote", "1.0.0");
        let sha = hex_sha256(&bytes);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let serve = tokio::spawn({
            let bytes = bytes.clone();
            async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf).await;
                let resp = format!(
                    "HTTP/1.0 200 OK\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\n\r\n",
                    bytes.len()
                );
                stream.write_all(resp.as_bytes()).await.unwrap();
                stream.write_all(&bytes).await.unwrap();
            }
        });

        let lock = make_lock(vec![], vec![], BTreeMap::new());
        let wheel = index_lock_wheel(
            "remote",
            "1.0.0",
            &format!("http://127.0.0.1:{port}/{filename}"),
            &sha,
        );
        let path = materialize_index_wheel(&lock, &wheel, &fetch_dir, &store_root)
            .await
            .unwrap();
        serve.await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert_eq!(sha256_file(&path).unwrap(), sha);
        assert!(
            store_root.join(&sha).join(filename).exists(),
            "fetch must populate the sha-addressed wheel store"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn materialize_index_wheel_rejects_downloaded_metadata_mismatch() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let root = tempdir("index-fetch-identity-mismatch");
        let store_root = root.join("store");
        let fetch_dir = root.join("fetch");
        let filename = "remote-1.0.0-py3-none-any.whl";
        let bytes = test_wheel_bytes("evil", "9.9");
        let sha = hex_sha256(&bytes);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let serve = tokio::spawn({
            let bytes = bytes.clone();
            async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf).await;
                let resp = format!(
                    "HTTP/1.0 200 OK\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\n\r\n",
                    bytes.len()
                );
                stream.write_all(resp.as_bytes()).await.unwrap();
                stream.write_all(&bytes).await.unwrap();
            }
        });

        let lock = make_lock(vec![], vec![], BTreeMap::new());
        let wheel = index_lock_wheel(
            "remote",
            "1.0.0",
            &format!("http://127.0.0.1:{port}/{filename}"),
            &sha,
        );
        let err = materialize_index_wheel(&lock, &wheel, &fetch_dir, &store_root)
            .await
            .unwrap_err();
        serve.await.unwrap();
        assert!(format!("{err:#}").contains("records distribution `evil`"));
        assert!(!fetch_dir.join(filename).exists());
        assert!(!store_root.join(&sha).join(filename).exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn normalize_dist_name_pep503() {
        assert_eq!(normalize_dist_name("VTK"), "vtk");
        assert_eq!(
            normalize_dist_name("PyOpenGL_accelerate"),
            "pyopengl-accelerate"
        );
        assert_eq!(normalize_dist_name("ruamel.yaml"), "ruamel-yaml");
        assert_eq!(normalize_dist_name("foo__bar--baz"), "foo-bar-baz");
        assert_eq!(normalize_dist_name("genesis-world"), "genesis-world");
    }

    // glibc banner parsing: getconf, ldd, and a 3-part micro all yield the
    // (major, minor) pair; copyright noise on later lines is ignored.
    #[test]
    fn parses_glibc_version_banners() {
        assert_eq!(
            crate::glibc::parse_glibc_version("glibc 2.34\n"),
            Some((2, 34))
        );
        assert_eq!(
            crate::glibc::parse_glibc_version("ldd (GNU libc) 2.34\nCopyright (C) 2021 ...\n"),
            Some((2, 34))
        );
        assert_eq!(
            crate::glibc::parse_glibc_version("glibc 2.34.9000"),
            Some((2, 34))
        );
        assert_eq!(crate::glibc::parse_glibc_version("no version here"), None);
    }

    // Only a genuine manylinux platform-tag rejection triggers auto-relax;
    // unrelated solve/network failures must NOT be silently relaxed past.
    #[test]
    fn detects_platform_tag_conflict_only() {
        // uv's real rejection text (from the issue report).
        let uv_err = "× No solution found when resolving dependencies:\n  \
            Because isaacsim[all]==6.0.0.1 has no wheels with a matching platform \
            tag (e.g., `manylinux_2_34_x86_64`) and isaac-pack depends on \
            isaacsim[all]==6.0.0.1, we can conclude that the requirements are \
            unsatisfiable.";
        assert!(is_platform_tag_conflict(uv_err));
        // An unrelated version conflict must NOT count.
        let version_err = "× No solution found: because foo==1 depends on bar>=2 \
            and only bar==1 is available, resolution failed.";
        assert!(!is_platform_tag_conflict(version_err));
        // A network failure must NOT count.
        assert!(!is_platform_tag_conflict(
            "error: Failed to fetch: connection timed out"
        ));
    }

    fn tempdir(label: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!(
            "pixi-build-retread-installer-{label}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn verify_requires_marker_and_installed_wheel_metadata() {
        let root = tempdir("verify-marker");
        let prefix = root.join("prefix");
        let share = prefix.join("share").join("retread");
        std::fs::create_dir_all(&share).unwrap();

        let lock = make_lock(
            vec![],
            vec!["https://pypi.org/simple/".into()],
            BTreeMap::new(),
        );
        let raw = serde_json::to_vec(&lock).unwrap();
        let target = lock.resolution_target().unwrap();
        let lock_path = share.join(RetreadLock::file_name_for_target(&lock.bundle, &target));
        std::fs::write(&lock_path, &raw).unwrap();
        std::fs::write(
            share.join(lock.marker_name()),
            marker_with_audit(&lock_digest(&raw)),
        )
        .unwrap();

        let err = verify(&lock_path, &prefix, false).expect_err("marker alone must not verify");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("missing 1 locked wheel") && msg.contains("mypackage==1.0.0"),
            "unexpected verifier error: {msg}"
        );

        let site_packages = site_packages_dir(&prefix, &lock.python);
        let dist_info = site_packages.join("mypackage-1.0.0.dist-info");
        std::fs::create_dir_all(&dist_info).unwrap();
        std::fs::write(
            dist_info.join("METADATA"),
            "Metadata-Version: 2.1\nName: MyPackage\nVersion: 1.0.0\n",
        )
        .unwrap();

        let err = verify(&lock_path, &prefix, false)
            .expect_err("metadata without RECORD must not verify");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("RECORD") && msg.contains("payload check failed"),
            "unexpected verifier error: {msg}"
        );

        std::fs::create_dir_all(site_packages.join("mypackage")).unwrap();
        std::fs::write(site_packages.join("mypackage/__init__.py"), "").unwrap();
        std::fs::write(
            dist_info.join("RECORD"),
            "mypackage/__init__.py,,\nmypackage-1.0.0.dist-info/METADATA,,\nmypackage-1.0.0.dist-info/RECORD,,\n",
        )
        .unwrap();

        verify(&lock_path, &prefix, false).expect("matching marker plus metadata should verify");
        let _ = std::fs::remove_dir_all(root);
    }

    // A wheel built under Python 3.10 ships a RECORD referencing
    // __pycache__/*.cpython-310.pyc; under a 3.12 interpreter those files are
    // never materialized (3.12 writes cpython-312.pyc instead). Verification
    // must skip pyc RECORD entries rather than flag the install as broken.
    #[test]
    fn verify_ignores_stale_pycache_pyc_record_entries() {
        let root = tempdir("verify-pyc");
        let prefix = root.join("prefix");
        let share = prefix.join("share").join("retread");
        std::fs::create_dir_all(&share).unwrap();

        let mut lock = make_lock(
            vec![],
            vec!["https://pypi.org/simple/".into()],
            BTreeMap::new(),
        );
        lock.python = "3.12".into();
        let raw = serde_json::to_vec(&lock).unwrap();
        let target = lock.resolution_target().unwrap();
        let lock_path = share.join(RetreadLock::file_name_for_target(&lock.bundle, &target));
        std::fs::write(&lock_path, &raw).unwrap();
        std::fs::write(
            share.join(lock.marker_name()),
            marker_with_audit(&lock_digest(&raw)),
        )
        .unwrap();

        let site_packages = site_packages_dir(&prefix, &lock.python);
        let dist_info = site_packages.join("mypackage-1.0.0.dist-info");
        std::fs::create_dir_all(&dist_info).unwrap();
        std::fs::write(
            dist_info.join("METADATA"),
            "Metadata-Version: 2.1\nName: MyPackage\nVersion: 1.0.0\n",
        )
        .unwrap();
        std::fs::create_dir_all(site_packages.join("mypackage")).unwrap();
        std::fs::write(site_packages.join("mypackage/__init__.py"), "").unwrap();
        // RECORD lists a 3.10 pyc that does not (and will never) exist here.
        std::fs::write(
            dist_info.join("RECORD"),
            "mypackage/__init__.py,,\n\
             mypackage/__pycache__/__init__.cpython-310.pyc,,\n\
             mypackage-1.0.0.dist-info/METADATA,,\n\
             mypackage-1.0.0.dist-info/RECORD,,\n",
        )
        .unwrap();

        verify(&lock_path, &prefix, false)
            .expect("stale interpreter-specific pyc RECORD entries must not fail verify");

        // A missing real (non-pyc) file must still fail.
        std::fs::remove_file(site_packages.join("mypackage/__init__.py")).unwrap();
        let err = verify(&lock_path, &prefix, false)
            .expect_err("missing real payload file must still fail verify");
        assert!(format!("{err:#}").contains("__init__.py"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn pycache_pyc_entry_detection() {
        assert!(is_pycache_pyc_entry(
            "aiobotocore/__pycache__/__init__.cpython-310.pyc"
        ));
        assert!(is_pycache_pyc_entry("__pycache__/mod.cpython-312.pyc"));
        // Not in a __pycache__ dir, or not a .pyc: must still be checked.
        assert!(!is_pycache_pyc_entry("pkg/module.pyc"));
        assert!(!is_pycache_pyc_entry("pkg/__pycache__/data.txt"));
        assert!(!is_pycache_pyc_entry("pkg/__init__.py"));
    }

    #[test]
    fn verify_rejects_stale_marker_even_when_payload_exists() {
        let root = tempdir("verify-stale-marker");
        let prefix = root.join("prefix");
        let share = prefix.join("share").join("retread");
        std::fs::create_dir_all(&share).unwrap();

        let lock = make_lock(
            vec![],
            vec!["https://pypi.org/simple/".into()],
            BTreeMap::new(),
        );
        let raw = serde_json::to_vec(&lock).unwrap();
        let lock_path = share.join("retread-test-bundle.lock.json");
        std::fs::write(&lock_path, &raw).unwrap();
        std::fs::write(share.join(lock.marker_name()), "not-the-lock-hash").unwrap();

        let site_packages = site_packages_dir(&prefix, &lock.python);
        let dist_info = site_packages.join("mypackage-1.0.0.dist-info");
        std::fs::create_dir_all(&dist_info).unwrap();
        std::fs::write(
            dist_info.join("METADATA"),
            "Metadata-Version: 2.1\nName: mypackage\nVersion: 1.0.0\n",
        )
        .unwrap();
        std::fs::create_dir_all(site_packages.join("mypackage")).unwrap();
        std::fs::write(site_packages.join("mypackage/__init__.py"), "").unwrap();
        std::fs::write(
            dist_info.join("RECORD"),
            "mypackage/__init__.py,,\nmypackage-1.0.0.dist-info/METADATA,,\nmypackage-1.0.0.dist-info/RECORD,,\n",
        )
        .unwrap();

        let err = verify(&lock_path, &prefix, false).expect_err("stale marker must not verify");
        let msg = format!("{err:#}");
        assert!(msg.contains("does not match"), "unexpected error: {msg}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn verify_rejects_bare_digest_marker_even_when_payload_exists() {
        let root = tempdir("verify-bare-marker");
        let prefix = root.join("prefix");
        let share = prefix.join("share").join("retread");
        std::fs::create_dir_all(&share).unwrap();

        let lock = make_lock(
            vec![],
            vec!["https://pypi.org/simple/".into()],
            BTreeMap::new(),
        );
        let raw = serde_json::to_vec(&lock).unwrap();
        let lock_path = share.join("retread-test-bundle.lock.json");
        std::fs::write(&lock_path, &raw).unwrap();
        std::fs::write(share.join(lock.marker_name()), lock_digest(&raw)).unwrap();

        let site_packages = site_packages_dir(&prefix, &lock.python);
        let dist_info = site_packages.join("mypackage-1.0.0.dist-info");
        std::fs::create_dir_all(&dist_info).unwrap();
        std::fs::write(
            dist_info.join("METADATA"),
            "Metadata-Version: 2.1\nName: mypackage\nVersion: 1.0.0\n",
        )
        .unwrap();
        std::fs::create_dir_all(site_packages.join("mypackage")).unwrap();
        std::fs::write(site_packages.join("mypackage/__init__.py"), "").unwrap();
        std::fs::write(
            dist_info.join("RECORD"),
            "mypackage/__init__.py,,\nmypackage-1.0.0.dist-info/METADATA,,\nmypackage-1.0.0.dist-info/RECORD,,\n",
        )
        .unwrap();

        let err = verify(&lock_path, &prefix, false)
            .expect_err("bare digest marker must trigger audit refresh");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no glibc audit record"),
            "unexpected error: {msg}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn verify_rejects_metadata_when_record_payload_is_missing() {
        let root = tempdir("verify-missing-record-payload");
        let prefix = root.join("prefix");
        let share = prefix.join("share").join("retread");
        std::fs::create_dir_all(&share).unwrap();

        let lock = make_lock(
            vec![],
            vec!["https://pypi.org/simple/".into()],
            BTreeMap::new(),
        );
        let raw = serde_json::to_vec(&lock).unwrap();
        let lock_path = share.join("retread-test-bundle.lock.json");
        std::fs::write(&lock_path, &raw).unwrap();
        std::fs::write(
            share.join(lock.marker_name()),
            marker_with_audit(&lock_digest(&raw)),
        )
        .unwrap();

        let site_packages = site_packages_dir(&prefix, &lock.python);
        let dist_info = site_packages.join("mypackage-1.0.0.dist-info");
        std::fs::create_dir_all(&dist_info).unwrap();
        std::fs::write(
            dist_info.join("METADATA"),
            "Metadata-Version: 2.1\nName: mypackage\nVersion: 1.0.0\n",
        )
        .unwrap();
        std::fs::write(
            dist_info.join("RECORD"),
            "mypackage/__init__.py,,\nmypackage-1.0.0.dist-info/METADATA,,\nmypackage-1.0.0.dist-info/RECORD,,\n",
        )
        .unwrap();

        let err =
            verify(&lock_path, &prefix, false).expect_err("missing RECORD payload must not verify");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("missing installed file") && msg.contains("mypackage/__init__.py"),
            "unexpected error: {msg}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn record_path_token_handles_csv_quoted_commas_and_quotes() {
        assert_eq!(
            record_path_token("\"pkg/data,part.txt\",sha256=abc,123").as_deref(),
            Some("pkg/data,part.txt")
        );
        assert_eq!(
            record_path_token("\"pkg/data\"\"part.txt\",,").as_deref(),
            Some("pkg/data\"part.txt")
        );
        assert_eq!(
            record_path_token("pkg/__init__.py,,").as_deref(),
            Some("pkg/__init__.py")
        );
        assert_eq!(record_path_token(",,").as_deref(), None);
    }
}
