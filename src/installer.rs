//! v2.0.0 courier: the `retread install` subcommand.
//!
//! Invoked from the courier conda package's post-link script at env link
//! time: `pixi-build-retread install --lock <lock> --prefix <prefix>`.
//! Reads the committed lock and installs the bundle's PyPI wheels into the
//! active conda env via uv (fast hardlink), resolving the root
//! requirements against the shipped find-links wheels + the recorded index
//! chain, into the conda env so shared transitives stay conda-provided.
//! Idempotent: a content-hash marker makes a re-link a no-op.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use crate::lock::RetreadLock;

/// Fallback primary index when the lock carries no index_urls.
pub(crate) use crate::handler::PUBLIC_PYPI;

/// PEP 503 normalized distribution name (lowercase; runs of `-`, `_`, `.`
/// collapse to a single `-`). Used to compare names across uv's `pip list`
/// output and the lock's wheel names when building the uv exclude set.
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

/// S1: build a uv constraints-file body from the conda-provided transitives,
/// so uv cannot resolve a PyPI wheel OUTSIDE the version conda will install
/// (e.g. bound `torchaudio>=2.7,<3` so the closure can't jump to 2.11 and
/// skew against conda's torch). Only clean PEP 508 version specifiers are
/// emitted: conda-only names (`python`, `python_abi`) and conda specs that
/// carry a build string (a space, e.g. `3.11.* *_cp311`) or no comparison
/// operator are skipped -- they are not valid uv constraints.
pub(crate) fn conda_deps_to_constraints(deps: &[crate::lock::CondaDep]) -> String {
    let mut out = String::new();
    for d in deps {
        let name = d.name.trim();
        let spec = d.spec.trim();
        if name.is_empty() || name == "python" || name == "python_abi" {
            continue;
        }
        if spec.is_empty() || spec.contains(' ') {
            continue;
        }
        if !spec.starts_with(['<', '>', '=', '!', '~']) {
            continue;
        }
        out.push_str(name);
        out.push_str(spec);
        out.push('\n');
    }
    out
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
    // substring match. See the identical fix on the `uv pip list` call in
    // `install` for the concrete failure this class of bug causes.
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
                crate::glibc::undeclared_glibc_error(host, crate::glibc::extract_manylinux_floor(&text))
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
                .is_some_and(|versions| versions.contains_key(&wheel.version));
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
    let missing = missing_locked_wheels_from_installed(lock, &installed);
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
        let dist_root = installed
            .get(&name)
            .and_then(|versions| versions.get(&wheel.version))
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
    let mut out: BTreeMap<String, crate::glibc::PayloadLib> = BTreeMap::new();
    for wheel in &lock.wheels {
        let name = normalize_dist_name(&wheel.name);
        let dist_root = installed
            .get(&name)
            .and_then(|versions| versions.get(&wheel.version))
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
    let raw = std::fs::read(lock_path)
        .with_context(|| format!("reading lock {}", lock_path.display()))?;
    let lock: RetreadLock = serde_json::from_slice(&raw)
        .with_context(|| format!("parsing lock {}", lock_path.display()))?;

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
        let _ = crate::glibc::full_verify_audit(
            &lock,
            prefix,
            &site_packages,
            &libs,
            Some(&audit),
        )?;
    }
    Ok(())
}

/// Build the `uv pip install` argument list (pure; no I/O, no spawn; no
/// argv[0]). S1: a `--constraints` file bounds conda-provided transitives to
/// conda's range. S3: `lock.index_urls` is replayed verbatim (first = primary
/// `--index-url`, rest `--extra-index-url`; public PyPI only as the empty
/// fallback) so the backend's recorded resolution priority is preserved.
pub(crate) fn build_uv_args(
    lock: &RetreadLock,
    prefix: &Path,
    wheels_dir: Option<&Path>,
    overrides_file: Option<&Path>,
    constraints_file: Option<&Path>,
    excludes_file: Option<&Path>,
    python_platform: Option<&str>,
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

    // Optional find-links directory for locally-shipped wheels.
    if let Some(dir) = wheels_dir {
        args.push("--find-links".into());
        args.push(dir.into());
    }

    // Search ALL indexes for the best-compatible wheel instead of uv's default
    // `first-index` (stop at the first index that lists the name). Bundles like
    // Isaac Sim publish the real binary wheel ONLY on a secondary index
    // (pypi.nvidia.com) while a stub of the same name sits on public PyPI;
    // first-index then locks onto the stub and reports "no wheels with a
    // matching platform tag", which no amount of glibc relaxation can fix
    // because the usable wheel was never considered. This mirrors the index
    // strategy pixi/uv workspaces already set (and the documented manual
    // recovery command) so the post-link resolves the same wheel the solve did.
    args.push("--index-strategy".into());
    args.push("unsafe-best-match".into());

    // S3: replay index chain verbatim.
    // find-links suppresses uv's implicit default index, so always set one
    // explicitly. First lock entry = primary; subsequent = extras. If the lock
    // carries no indexes at all, fall back to public PyPI as the sole primary.
    if lock.index_urls.is_empty() {
        args.push("--index-url".into());
        args.push(PUBLIC_PYPI.into());
    } else {
        let mut it = lock.index_urls.iter();
        args.push("--index-url".into());
        args.push(it.next().expect("non-empty").into());
        for u in it {
            args.push("--extra-index-url".into());
            args.push(u.into());
        }
    }

    // S1: bound conda-provided transitives to conda's version range via a
    // constraints file, so uv's closure can't pick a PyPI wheel outside what
    // conda installs (the torchaudio>=2.7,<3 vs 2.11 skew). Written by run().
    if let Some(c) = constraints_file {
        args.push("--constraints".into());
        args.push(c.into());
    }

    // Exclude the conda-populated set from uv's resolution. The courier
    // contract is: conda provides the conda-capable transitives; uv only ADDS
    // the bundle's PyPI-only wheels (+ their PyPI-only tail). Without this, uv
    // re-resolves the FULL closure and tries to REPLACE conda dists -- which
    // fails hard on any conda package uv can't uninstall (legacy egg-info /
    // RECORD-less dist-info, e.g. vtk). The exclude set is uv's OWN view of the
    // prefix (`uv pip list`), so names match the resolver and every packaging
    // shape is covered -- no per-package logic, no conda/PyPI name-skew. Written
    // by run() (minus the bundle's own wheels, which uv must still install).
    if let Some(e) = excludes_file {
        args.push("--excludes".into());
        args.push(e.into());
    }

    // Prerelease overrides file (written before calling build_uv_args in run()).
    if let Some(ovr) = overrides_file {
        args.push("--overrides".into());
        args.push(ovr.into());
    }

    // Root requirements drive the closure.
    for r in &lock.root_requirements {
        args.push(r.as_str().into());
    }

    args
}

/// Install (or no-op) the bundle described by `lock_path` into `prefix`.
pub fn run(lock_path: &Path, prefix: &Path) -> Result<()> {
    let raw = std::fs::read(lock_path)
        .with_context(|| format!("reading lock {}", lock_path.display()))?;
    let lock: RetreadLock = serde_json::from_slice(&raw)
        .with_context(|| format!("parsing lock {}", lock_path.display()))?;

    let share = prefix.join("share").join("retread");

    // Serialize concurrent installs into the same prefix. The self-heal
    // activate.d guard runs `retread install` on activation, so parallel
    // `pixi run`s can race two uv installs into one env. Hold an advisory
    // exclusive lock for the whole run; a second waiter then sees the marker
    // already current and no-ops. Best-effort: if the lock can't be taken we
    // proceed unlocked rather than block the repair.
    std::fs::create_dir_all(&share).ok();
    let _install_lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(share.join(format!(".{}.install.lock", lock.bundle)))
        .inspect(|f| {
            let _ = fs4::fs_std::FileExt::lock_exclusive(f);
        })
        .ok();

    let marker = share.join(lock.marker_name());
    let want = lock_digest(&raw);
    if marker_matches(&marker, &want) {
        match verify_payload_installed(&lock, prefix) {
            Ok(()) => {
                let marker_text = std::fs::read_to_string(&marker).unwrap_or_default();
                match crate::glibc::verify_marker_state(&lock, prefix, &marker_text) {
                    Ok(_) => {
                        let msg = format!("retread install: {} already current; skipping", lock.bundle);
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
                        std::fs::write(&marker, crate::glibc::marker_body(&want, &audit)?)
                            .with_context(|| format!("writing marker {}", marker.display()))?;
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
                eprintln!(
                    "retread install: {} marker exists but payload verification failed; \
                     reinstalling ({err:#})",
                    lock.bundle
                );
            }
        }
    }

    if lock.root_requirements.is_empty() {
        eprintln!(
            "retread install: {} has no root requirements; nothing to do",
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
    let wheels_dir_opt = wheels_dir.is_dir().then_some(wheels_dir.as_path());

    let uv: OsString = {
        let p = prefix.join("bin").join("uv");
        if p.exists() {
            p.into_os_string()
        } else {
            "uv".into()
        }
    };

    // Prerelease: uv only honors prereleases from direct reqs + overrides,
    // so pass the micro-table as an overrides file.
    let ovr_path = share.join(format!("{}.overrides.txt", lock.bundle));
    let overrides_file = if !lock.prerelease.is_empty() {
        std::fs::create_dir_all(&share).ok();
        let mut body = String::new();
        for (name, spec) in &lock.prerelease {
            body.push_str(&format!("{name}{spec}\n"));
        }
        std::fs::write(&ovr_path, &body)
            .with_context(|| format!("writing {}", ovr_path.display()))?;
        Some(ovr_path.as_path())
    } else {
        None
    };

    // S1 constraints file: bound conda-provided transitives to conda's range.
    let con_path = share.join(format!("{}.constraints.txt", lock.bundle));
    let constraints_body = conda_deps_to_constraints(&lock.conda_run_deps);
    let constraints_file = if !constraints_body.is_empty() {
        std::fs::create_dir_all(&share).ok();
        std::fs::write(&con_path, &constraints_body)
            .with_context(|| format!("writing {}", con_path.display()))?;
        Some(con_path.as_path())
    } else {
        None
    };

    // Exclude the conda-populated set from uv's resolution (see build_uv_args).
    // Source = uv's OWN view of the prefix, so names match the resolver and
    // every packaging shape is covered. Subtract the bundle's own payload (the
    // meta-wheel + each shipped/fetched wheel) so uv still installs those even
    // if a conda package ever collided on the same normalized name.
    let protect: std::collections::HashSet<String> = lock
        .wheels
        .iter()
        .map(|w| normalize_dist_name(&w.name))
        .chain(
            lock.root_requirements
                .iter()
                .map(|r| normalize_dist_name(r.split("==").next().unwrap_or(r))),
        )
        .collect();
    let exc_path = share.join(format!("{}.excludes.txt", lock.bundle));
    let excludes_file = {
        // v3.0.4: this output is parsed line-by-line and the result is
        // written VERBATIM to excludes.txt, which a later `uv pip install
        // -r excludes.txt` call re-parses as a requirements file. uv
        // respects FORCE_COLOR/CLICOLOR_FORCE from the environment even
        // when stdout is piped (not a tty), so without `--color never` a
        // colorized caller environment corrupts every package name with
        // raw ANSI escapes (`\x1b[1mabsl-py\x1b[0m`), and uv's requirements
        // parser then fails with "Unexpected '', expected '-c', '-e', '-r'
        // or the start of a requirement" on the very first byte.
        let out = Command::new(&uv)
            .args(["pip", "list", "--color", "never", "--format", "freeze", "--python"])
            .arg(&python)
            .output()
            .with_context(|| "listing installed packages for the uv exclude set")?;
        if out.status.success() {
            let mut body = String::new();
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                let name = line.split("==").next().unwrap_or("").trim();
                if name.is_empty() || protect.contains(&normalize_dist_name(name)) {
                    continue;
                }
                body.push_str(name);
                body.push('\n');
            }
            std::fs::create_dir_all(&share).ok();
            std::fs::write(&exc_path, &body)
                .with_context(|| format!("writing {}", exc_path.display()))?;
            Some(exc_path.as_path())
        } else {
            tracing::warn!(
                status = %out.status,
                "uv pip list failed; proceeding without an exclude set (uv may try to \
                 replace conda-provided packages)",
            );
            None
        }
    };

    // First attempt: uv's manylinux host gate untouched. Streamed live so
    // multi-GB index-wheel download progress is visible.
    let args = build_uv_args(
        &lock,
        prefix,
        wheels_dir_opt,
        overrides_file,
        constraints_file,
        excludes_file,
        None,
    );

    let install_msg = format!(
        "retread install: {} -> {} ({} root reqs)",
        lock.bundle,
        prefix.display(),
        lock.root_requirements.len()
    );
    eprintln!("{install_msg}");
    crate::status::phase(
        lock_path.parent().unwrap_or(std::path::Path::new(".")),
        &lock.bundle,
        &install_msg,
    );

    let status = Command::new(&uv)
        .args(&args)
        .status()
        .with_context(|| format!("spawning uv ({uv:?})"))?;
    let mut relaxed_platform: Option<String> = None;
    let mut declaration_source: Option<String> = None;
    if !status.success() {
        // Relax on a manylinux platform-tag conflict only. Classify the failure
        // with a captured `--dry-run` resolve; if it is purely the platform tag
        // and a libc declaration authorizes a higher floor, retry once targeting
        // exactly that declaration. Any other failure surfaces unchanged.
        match relax_platform_on_conflict(&uv, &args, &lock)? {
            Some(outcome) => {
                let relaxed = build_uv_args(
                    &lock,
                    prefix,
                    wheels_dir_opt,
                    overrides_file,
                    constraints_file,
                    excludes_file,
                    Some(&outcome.platform),
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
    std::fs::write(&marker, crate::glibc::marker_body(&want, &audit)?)
        .with_context(|| format!("writing marker {}", marker.display()))?;
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
    use std::path::PathBuf;

    use super::*;
    use crate::lock::{CondaDep, LockWheel, Origin};

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
            inputs_hash: "abc".into(),
            root_requirements: vec!["mypackage==1.0.0".into()],
            wheels: vec![LockWheel {
                name: "mypackage".into(),
                version: "1.0.0".into(),
                origin: Origin::Built,
                filename: "mypackage-1.0.0-py3-none-any.whl".into(),
                url: None,
                sha256: None,
                requires_dist: vec![],
                must_ship: true,
                upstream_url: None,
                git_source: None,
                sdist_source: None,
            }],
            conda_run_deps,
            index_urls,
            prerelease,
            shadow_libs: BTreeMap::new(),
            declared_glibc: None,
            conda_capable: vec![],
            entry_specs: vec![],
        }
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

    // S1: conda-covered transitives become a uv constraints file bounding
    // them to conda's range (so the closure can't skew, e.g. torchaudio 2.11).
    #[test]
    fn s1_conda_run_deps_constraints() {
        let deps = vec![
            CondaDep {
                name: "torch".into(),
                spec: ">=2.7,<3".into(),
            },
            CondaDep {
                name: "torchaudio".into(),
                spec: ">=2.7,<3".into(),
            },
            // conda-only specs must be dropped (not valid uv constraints).
            CondaDep {
                name: "python_abi".into(),
                spec: "3.11.* *_cp311".into(),
            },
            CondaDep {
                name: "python".into(),
                spec: "3.11.*".into(),
            },
        ];
        let body = conda_deps_to_constraints(&deps);
        assert!(
            body.contains("torch>=2.7,<3"),
            "torch bounded; got {body:?}"
        );
        assert!(
            body.contains("torchaudio>=2.7,<3"),
            "torchaudio bounded; got {body:?}"
        );
        assert!(!body.contains("python_abi"), "build-string spec dropped");
        assert!(
            !body
                .lines()
                .any(|l| l.starts_with("python3") || l == "python3.11.*"),
            "conda-only python dropped; got {body:?}"
        );

        // and the flag is threaded into argv when a constraints file is given.
        let lock = make_lock(deps, vec![PUBLIC_PYPI.into()], BTreeMap::new());
        let prefix = PathBuf::from("/fake/prefix");
        let con = PathBuf::from("/fake/prefix/share/retread/test-bundle.constraints.txt");
        let args = build_uv_args(&lock, &prefix, None, None, Some(con.as_path()), None, None);
        let strs = argv_strings(&args);
        assert_eq!(
            flag_values(&strs, "--constraints"),
            vec![con.to_string_lossy().into_owned()]
        );
        // never the invalid flag.
        assert!(!strs.iter().any(|s| s == "--no-install-package"));
    }

    // S1: no constraints file -> no --constraints flag.
    #[test]
    fn s1_no_constraints_no_flag() {
        let lock = make_lock(vec![], vec![PUBLIC_PYPI.into()], BTreeMap::new());
        let prefix = PathBuf::from("/fake/prefix");
        let args = build_uv_args(&lock, &prefix, None, None, None, None, None);
        let strs = argv_strings(&args);
        assert!(!strs.contains(&"--constraints".to_string()));
    }

    // The exclude set (conda-populated packages) is threaded as --excludes when
    // present, and absent otherwise.
    #[test]
    fn excludes_file_appears_in_argv() {
        let lock = make_lock(vec![], vec![PUBLIC_PYPI.into()], BTreeMap::new());
        let prefix = PathBuf::from("/fake/prefix");
        let exc = PathBuf::from("/fake/prefix/share/retread/test-bundle.excludes.txt");
        let args = build_uv_args(&lock, &prefix, None, None, None, Some(exc.as_path()), None);
        let strs = argv_strings(&args);
        assert_eq!(
            flag_values(&strs, "--excludes"),
            vec![exc.to_string_lossy().into_owned()]
        );
        // absent when not provided.
        let args = build_uv_args(&lock, &prefix, None, None, None, None, None);
        assert!(!argv_strings(&args).contains(&"--excludes".to_string()));
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

    // S3: index chain order matches lock.index_urls exactly, primary = index_urls[0].
    #[test]
    fn s3_index_chain_matches_lock_order() {
        let lock = make_lock(
            vec![],
            vec![
                "https://pypi.nvidia.com".into(),
                "https://pypi.org/simple/".into(),
            ],
            BTreeMap::new(),
        );
        let prefix = PathBuf::from("/fake/prefix");
        let args = build_uv_args(&lock, &prefix, None, None, None, None, None);
        let strs = argv_strings(&args);

        let primary = flag_values(&strs, "--index-url");
        let extras = flag_values(&strs, "--extra-index-url");

        assert_eq!(primary, vec!["https://pypi.nvidia.com"]);
        assert_eq!(extras, vec!["https://pypi.org/simple/"]);
    }

    // S3: public PyPI is NOT forced as primary when lock names a different primary.
    #[test]
    fn s3_no_forced_public_pypi_primary() {
        let lock = make_lock(
            vec![],
            vec![
                "https://pypi.nvidia.com".into(),
                "https://pypi.org/simple/".into(),
            ],
            BTreeMap::new(),
        );
        let prefix = PathBuf::from("/fake/prefix");
        let args = build_uv_args(&lock, &prefix, None, None, None, None, None);
        let strs = argv_strings(&args);

        let primary = flag_values(&strs, "--index-url");
        assert!(
            !primary
                .iter()
                .any(|u| u.trim_end_matches('/') == PUBLIC_PYPI.trim_end_matches('/')),
            "public PyPI must not be the primary when lock specifies a different primary; got {primary:?}"
        );
    }

    // S3: empty index_urls falls back to PUBLIC_PYPI as sole primary.
    #[test]
    fn s3_empty_index_urls_fallback_to_public_pypi() {
        let lock = make_lock(vec![], vec![], BTreeMap::new());
        let prefix = PathBuf::from("/fake/prefix");
        let args = build_uv_args(&lock, &prefix, None, None, None, None, None);
        let strs = argv_strings(&args);

        let primary = flag_values(&strs, "--index-url");
        let extras = flag_values(&strs, "--extra-index-url");

        assert_eq!(
            primary,
            vec![PUBLIC_PYPI],
            "empty index_urls must fall back to public PyPI as primary"
        );
        assert!(
            extras.is_empty(),
            "no extra indexes when lock has none; got {extras:?}"
        );
    }

    // Overrides file is passed through when provided.
    #[test]
    fn overrides_file_appears_in_argv() {
        let lock = make_lock(vec![], vec![PUBLIC_PYPI.into()], BTreeMap::new());
        let prefix = PathBuf::from("/fake/prefix");
        let ovr = PathBuf::from("/fake/prefix/share/retread/test-bundle.overrides.txt");
        let args = build_uv_args(&lock, &prefix, None, Some(ovr.as_path()), None, None, None);
        let strs = argv_strings(&args);
        let ovr_vals = flag_values(&strs, "--overrides");
        assert_eq!(ovr_vals, vec![ovr.to_string_lossy().into_owned()]);
    }

    // Root requirements appear in argv.
    #[test]
    fn root_requirements_in_argv() {
        let lock = make_lock(vec![], vec![PUBLIC_PYPI.into()], BTreeMap::new());
        let prefix = PathBuf::from("/fake/prefix");
        let args = build_uv_args(&lock, &prefix, None, None, None, None, None);
        let strs = argv_strings(&args);
        assert!(
            strs.contains(&"mypackage==1.0.0".to_string()),
            "root requirements must appear in argv"
        );
    }

    // HOTFIX v2.10.1: --python-platform is threaded through when Some, and
    // absent (uv's host gate untouched) when None.
    #[test]
    fn python_platform_flag_appears_in_argv() {
        let lock = make_lock(vec![], vec![PUBLIC_PYPI.into()], BTreeMap::new());
        let prefix = PathBuf::from("/fake/prefix");
        let args = build_uv_args(
            &lock,
            &prefix,
            None,
            None,
            None,
            None,
            Some("x86_64-manylinux_2_35"),
        );
        let strs = argv_strings(&args);
        assert_eq!(
            flag_values(&strs, "--python-platform"),
            vec!["x86_64-manylinux_2_35".to_string()]
        );
        // absent when None -> uv's manylinux host gate is left untouched.
        let args = build_uv_args(&lock, &prefix, None, None, None, None, None);
        assert!(!argv_strings(&args).contains(&"--python-platform".to_string()));
    }

    // Multi-index bundles (Isaac Sim's real wheel is on pypi.nvidia.com, a
    // stub of the same name on public PyPI) require uv to weigh ALL indexes,
    // not stop at the first that lists the name. Without this the post-link
    // resolves the stub and dies with a platform-tag error the glibc relax
    // can't recover from.
    #[test]
    fn index_strategy_is_unsafe_best_match() {
        let lock = make_lock(
            vec![],
            vec![PUBLIC_PYPI.into(), "https://pypi.nvidia.com".into()],
            BTreeMap::new(),
        );
        let args = build_uv_args(&lock, &PathBuf::from("/fake/prefix"), None, None, None, None, None);
        let strs = argv_strings(&args);
        assert_eq!(
            flag_values(&strs, "--index-strategy"),
            vec!["unsafe-best-match".to_string()],
            "installer must search all indexes for the best-compatible wheel"
        );
    }

    // glibc banner parsing: getconf, ldd, and a 3-part micro all yield the
    // (major, minor) pair; copyright noise on later lines is ignored.
    #[test]
    fn parses_glibc_version_banners() {
        assert_eq!(crate::glibc::parse_glibc_version("glibc 2.34\n"), Some((2, 34)));
        assert_eq!(
            crate::glibc::parse_glibc_version("ldd (GNU libc) 2.34\nCopyright (C) 2021 ...\n"),
            Some((2, 34))
        );
        assert_eq!(crate::glibc::parse_glibc_version("glibc 2.34.9000"), Some((2, 34)));
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

        let lock = make_lock(vec![], vec![PUBLIC_PYPI.into()], BTreeMap::new());
        let raw = serde_json::to_vec(&lock).unwrap();
        let lock_path = share.join(lock.marker_name().replace(".installed", ".lock.json"));
        std::fs::write(&lock_path, &raw).unwrap();
        std::fs::write(share.join(lock.marker_name()), marker_with_audit(&lock_digest(&raw))).unwrap();

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

        let err = verify(&lock_path, &prefix, false).expect_err("metadata without RECORD must not verify");
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

    #[test]
    fn verify_rejects_stale_marker_even_when_payload_exists() {
        let root = tempdir("verify-stale-marker");
        let prefix = root.join("prefix");
        let share = prefix.join("share").join("retread");
        std::fs::create_dir_all(&share).unwrap();

        let lock = make_lock(vec![], vec![PUBLIC_PYPI.into()], BTreeMap::new());
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

        let lock = make_lock(vec![], vec![PUBLIC_PYPI.into()], BTreeMap::new());
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
        assert!(msg.contains("no glibc audit record"), "unexpected error: {msg}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn verify_rejects_metadata_when_record_payload_is_missing() {
        let root = tempdir("verify-missing-record-payload");
        let prefix = root.join("prefix");
        let share = prefix.join("share").join("retread");
        std::fs::create_dir_all(&share).unwrap();

        let lock = make_lock(vec![], vec![PUBLIC_PYPI.into()], BTreeMap::new());
        let raw = serde_json::to_vec(&lock).unwrap();
        let lock_path = share.join("retread-test-bundle.lock.json");
        std::fs::write(&lock_path, &raw).unwrap();
        std::fs::write(share.join(lock.marker_name()), marker_with_audit(&lock_digest(&raw))).unwrap();

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

        let err = verify(&lock_path, &prefix, false).expect_err("missing RECORD payload must not verify");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("missing installed file") && msg.contains("mypackage/__init__.py"),
            "unexpected error: {msg}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    // v3.0.4 regression: `uv pip list` output is parsed line-by-line and the
    // result is written VERBATIM to excludes.txt, which a later `uv pip
    // install -r excludes.txt` re-parses as a requirements file. uv respects
    // FORCE_COLOR/CLICOLOR_FORCE from the environment even when stdout is
    // piped (not a tty) -- without `--color never` a colorized caller
    // environment corrupts every package name with raw ANSI escapes
    // (`\x1b[1mabsl-py\x1b[0m`), and uv's requirements parser then fails
    // with "Unexpected '', expected '-c', '-e', '-r' or the start of a
    // requirement" on the very first byte. This is the exact subprocess
    // invocation `install`'s excludes.txt generation uses (see the `uv pip
    // list` call there) -- asserting no ESC bytes appear even under
    // FORCE_COLOR locks in the fix.
    #[test]
    #[ignore = "live: needs uv on PATH; run with --include-ignored"]
    fn uv_pip_list_color_never_survives_force_color() {
        let python = std::process::Command::new("which")
            .arg("python3")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "python3".to_string());

        let out = std::process::Command::new("uv")
            .args(["pip", "list", "--color", "never", "--format", "freeze", "--python"])
            .arg(&python)
            .env("FORCE_COLOR", "3")
            .env("CLICOLOR_FORCE", "1")
            .output()
            .expect("spawning uv");
        assert!(out.status.success(), "uv pip list failed: {out:?}");
        assert!(
            !out.stdout.contains(&0x1b),
            "uv pip list --color never must not emit ANSI escapes even with \
             FORCE_COLOR/CLICOLOR_FORCE set in the environment (found ESC byte \
             in output, which would corrupt excludes.txt exactly as in #8's \
             follow-up bug)"
        );
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
