//! v2.0.0 courier: the `retread install` subcommand.
//!
//! Invoked from the courier conda package's post-link script at env link
//! time: `pixi-build-retread install --lock <lock> --prefix <prefix>`.
//! Reads the committed lock and installs the bundle's PyPI wheels into the
//! active conda env via uv (fast hardlink), resolving the root
//! requirements against the shipped find-links wheels + the recorded index
//! chain, into the conda env so shared transitives stay conda-provided.
//! Idempotent: a content-hash marker makes a re-link a no-op.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use crate::lock::RetreadLock;

/// Install (or no-op) the bundle described by `lock_path` into `prefix`.
pub fn run(lock_path: &Path, prefix: &Path) -> Result<()> {
    let raw = std::fs::read(lock_path)
        .with_context(|| format!("reading lock {}", lock_path.display()))?;
    let lock: RetreadLock = serde_json::from_slice(&raw)
        .with_context(|| format!("parsing lock {}", lock_path.display()))?;

    let share = prefix.join("share").join("retread");
    let marker = share.join(lock.marker_name());
    let want = format!("{:x}", Sha256::digest(&raw));
    if let Ok(have) = std::fs::read_to_string(&marker)
        && have.trim() == want
    {
        eprintln!("retread install: {} already current; skipping", lock.bundle);
        return Ok(());
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
    let uv = {
        let p = prefix.join("bin").join("uv");
        if p.exists() {
            p.into_os_string()
        } else {
            "uv".into()
        }
    };

    let mut cmd = Command::new(&uv);
    cmd.arg("pip").arg("install").arg("--python").arg(&python);
    if wheels_dir.is_dir() {
        cmd.arg("--find-links").arg(&wheels_dir);
    }
    // Index chain. FINDING-C: find-links suppresses the implicit default
    // index, so set it explicitly; entry indexes (nvidia, ...) ride as
    // extras.
    let default = crate::handler::PUBLIC_PYPI.trim_end_matches('/');
    cmd.arg("--index-url").arg(crate::handler::PUBLIC_PYPI);
    for u in &lock.index_urls {
        if u.trim_end_matches('/') != default {
            cmd.arg("--extra-index-url").arg(u);
        }
    }
    // Prerelease: uv only honors prereleases from direct reqs + overrides,
    // so pass the micro-table as an overrides file.
    if !lock.prerelease.is_empty() {
        std::fs::create_dir_all(&share).ok();
        let ovr = share.join(format!("{}.overrides.txt", lock.bundle));
        let mut body = String::new();
        for (name, spec) in &lock.prerelease {
            body.push_str(&format!("{name}{spec}\n"));
        }
        std::fs::write(&ovr, body).with_context(|| format!("writing {}", ovr.display()))?;
        cmd.arg("--overrides").arg(&ovr);
    }
    for r in &lock.root_requirements {
        cmd.arg(r);
    }

    eprintln!(
        "retread install: {} -> {} ({} root reqs)",
        lock.bundle,
        prefix.display(),
        lock.root_requirements.len()
    );
    let status = cmd
        .status()
        .with_context(|| format!("spawning uv ({uv:?})"))?;
    if !status.success() {
        bail!(
            "uv pip install failed for bundle {} (status {status})",
            lock.bundle
        );
    }

    std::fs::create_dir_all(&share).ok();
    std::fs::write(&marker, want)
        .with_context(|| format!("writing marker {}", marker.display()))?;
    eprintln!("retread install: {} installed", lock.bundle);
    Ok(())
}
