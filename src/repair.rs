//! Atomic activation repair (defect D2).
//!
//! The activate.d self-heal guard runs `retread install`, which replays the
//! locked wheel set through a single `uv pip install --reinstall`. uv performs
//! that replay as uninstall-then-install and its uninstall is **not**
//! transactional: it deletes every path listed in a distribution's RECORD and
//! then `rmdir`s the emptied directories. A `.pyc` produced by any earlier
//! import is NOT in RECORD, so the `rmdir` fails with
//! `Directory not empty (os error 39)` and uv aborts *after* the uninstall
//! phase and *before* the install phase.
//!
//! Observed on `hover-gpu` (2026-08-18): torch + torchvision were uninstalled,
//! nothing was reinstalled, and `site-packages/torch/` survived as 459
//! `__pycache__` directories with no `__init__.py` and no `torch-*.dist-info`.
//! Python treats such a tree as a **PEP 420 namespace package**: `import torch`
//! and `importlib.util.find_spec("torch")` both SUCCEED and only attribute
//! access fails (`module 'torch' has no attribute '__version__'`). That is a
//! silent ghost, and it violates §1.9 (failure must be loud).
//!
//! This module makes the replay survivable at both ends:
//!
//! * [`pre_clean_for_uninstall`] deletes the unowned bytecode *before* uv
//!   touches the prefix, so uv's `rmdir` cannot fail on it (the trigger).
//! * [`inventory_transaction`] snapshots, before the transaction, which
//!   top-level directories each distribution owns, and
//!   [`rollback_half_removed`] deletes those directories on failure when the
//!   owning `.dist-info` is gone — turning a namespace ghost into a loud
//!   `ModuleNotFoundError`.
//! * [`RepairState`] makes the marker state machine explicit
//!   (installed / repairing / broken) instead of leaving the `.broken` file to
//!   whichever shell caller happened to notice the nonzero exit.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Explicit state of a bundle's install marker.
///
/// `repairing` is written before the destructive replay starts and is the
/// state a crashed/killed repair leaves behind, so a later activation can tell
/// "never finished" from "finished and failed".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairState {
    Installed,
    Repairing,
    Broken,
}

impl RepairState {
    pub fn as_str(self) -> &'static str {
        match self {
            RepairState::Installed => "installed",
            RepairState::Repairing => "repairing",
            RepairState::Broken => "broken",
        }
    }
}

/// Path of the explicit state file for `bundle` under `share/retread`.
pub fn state_path(share: &Path, bundle: &str) -> PathBuf {
    share.join(format!("{bundle}.state"))
}

/// Path of the legacy `.broken` sentinel the activate.d guard backs off on.
pub fn broken_path(share: &Path, bundle: &str) -> PathBuf {
    share.join(format!("{bundle}.broken"))
}

/// Record `state` for `bundle`. Best-effort: a repair must never fail because
/// its own bookkeeping file could not be written.
///
/// `Broken` also creates the legacy `<bundle>.broken` sentinel (the activate.d
/// guard's 300 s backoff reads it) if the caller has not already written one;
/// `Installed` removes both.
pub fn mark_state(share: &Path, bundle: &str, state: RepairState, detail: &str) {
    let _ = std::fs::create_dir_all(share);
    let state_file = state_path(share, bundle);
    match state {
        RepairState::Installed => {
            let _ = std::fs::remove_file(&state_file);
            let _ = std::fs::remove_file(broken_path(share, bundle));
        }
        RepairState::Repairing | RepairState::Broken => {
            let body = if detail.is_empty() {
                format!("{}\n", state.as_str())
            } else {
                format!("{}\n{detail}\n", state.as_str())
            };
            let _ = std::fs::write(&state_file, body);
            if state == RepairState::Broken {
                let broken = broken_path(share, bundle);
                if !broken.exists() {
                    let _ = std::fs::write(&broken, format!("{bundle}: {detail}\n"));
                }
            }
        }
    }
}

/// Read the recorded state for `bundle`, if any. `None` means no state file
/// (a prefix installed by an older retread, or never repaired).
pub fn read_state(share: &Path, bundle: &str) -> Option<RepairState> {
    let body = std::fs::read_to_string(state_path(share, bundle)).ok()?;
    match body.lines().next()?.trim() {
        "installed" => Some(RepairState::Installed),
        "repairing" => Some(RepairState::Repairing),
        "broken" => Some(RepairState::Broken),
        _ => None,
    }
}

/// One installed distribution the replay transaction is about to uninstall,
/// snapshotted BEFORE the transaction runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransactionDist {
    /// PEP 503 normalized distribution name.
    pub name: String,
    /// `<site-packages>/<name>-<version>.dist-info` (or `.egg-info`).
    pub dist_info: PathBuf,
    /// Top-level directories under site-packages this distribution owns,
    /// derived from its RECORD (falling back to `top_level.txt`).
    pub roots: Vec<PathBuf>,
}

/// PEP 503 normalization, duplicated here so this module stands alone.
fn normalize(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_sep = false;
    for c in name.trim().chars() {
        if c == '-' || c == '_' || c == '.' {
            if !prev_sep {
                out.push('-');
                prev_sep = true;
            }
        } else {
            out.push(c.to_ascii_lowercase());
            prev_sep = false;
        }
    }
    out
}

/// Distribution name a wheel FILE belongs to (`torch-2.5.1+cu124-cp310-…whl`
/// -> `torch`). Returns `None` for a name with no `-` separator.
pub fn wheel_file_dist_name(wheel: &Path) -> Option<String> {
    let stem = wheel.file_name()?.to_str()?;
    let stem = stem.strip_suffix(".whl").unwrap_or(stem);
    let (name, _) = stem.split_once('-')?;
    (!name.is_empty()).then(|| normalize(name))
}

/// The distribution names a wheel-file replay list will uninstall/reinstall.
pub fn replay_dist_names(wheel_files: &[PathBuf]) -> BTreeSet<String> {
    wheel_files
        .iter()
        .filter_map(|w| wheel_file_dist_name(w))
        .collect()
}

/// First path component of a RECORD token, when it is a plain relative path
/// inside site-packages (`torch/nn/__init__.py` -> `torch`). Escapes (`..`),
/// absolute paths, and bare top-level files are rejected.
fn record_top_level_dir(token: &str) -> Option<&str> {
    if token.starts_with('/') || token.starts_with("../") || token.starts_with("..\\") {
        return None;
    }
    let head = token.split(['/', '\\']).next()?;
    if head.is_empty() || head == "." || head == ".." {
        return None;
    }
    // A bare file at the top of site-packages owns no directory.
    token.contains(['/', '\\']).then_some(head)
}

/// Directories under `site_packages` that `dist_info`'s RECORD claims,
/// excluding the dist-info directory itself.
fn roots_from_metadata(site_packages: &Path, dist_info: &Path) -> Vec<PathBuf> {
    let self_name = dist_info
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();
    let mut heads: BTreeSet<String> = BTreeSet::new();

    if let Ok(body) = std::fs::read_to_string(dist_info.join("RECORD")) {
        for line in body.lines() {
            let Some(token) = crate::installer::record_path_token(line) else {
                continue;
            };
            if let Some(head) = record_top_level_dir(&token) {
                heads.insert(head.to_string());
            }
        }
    }
    if heads.is_empty() {
        // Wheels always ship RECORD; conda/legacy metadata may not.
        if let Ok(body) = std::fs::read_to_string(dist_info.join("top_level.txt")) {
            for line in body.lines() {
                let name = line.trim();
                if !name.is_empty() && site_packages.join(name).is_dir() {
                    heads.insert(name.to_string());
                }
            }
        }
    }

    heads
        .into_iter()
        .filter(|head| *head != self_name)
        .map(|head| site_packages.join(head))
        .filter(|p| p.is_dir())
        .collect()
}

/// Every `.dist-info` / `.egg-info` directory in `site_packages`, keyed by
/// PEP 503 name. Read straight from the directory listing (not from METADATA)
/// so a distribution whose METADATA uv already deleted is still recognised.
fn dist_info_dirs(site_packages: &Path) -> BTreeMap<String, PathBuf> {
    let mut out = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(site_packages) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(file_name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let stem = match file_name
            .strip_suffix(".dist-info")
            .or_else(|| file_name.strip_suffix(".egg-info"))
        {
            Some(stem) => stem,
            None => continue,
        };
        let name = stem.split('-').next().unwrap_or(stem);
        if name.is_empty() {
            continue;
        }
        out.insert(normalize(name), path);
    }
    out
}

/// Snapshot the distributions in `site_packages` that `names` will uninstall,
/// together with the directories they own. Call this BEFORE handing the prefix
/// to uv: after a failed transaction the RECORD is gone and ownership can no
/// longer be reconstructed.
pub fn inventory_transaction(
    site_packages: &Path,
    names: &BTreeSet<String>,
) -> Vec<TransactionDist> {
    if !site_packages.is_dir() {
        return Vec::new();
    }
    let installed = dist_info_dirs(site_packages);
    names
        .iter()
        .filter_map(|name| {
            let dist_info = installed.get(name)?;
            Some(TransactionDist {
                name: name.clone(),
                dist_info: dist_info.clone(),
                roots: roots_from_metadata(site_packages, dist_info),
            })
        })
        .collect()
}

/// Recursively delete every `__pycache__` directory under `dir`.
fn purge_pycache(dir: &Path, removed: &mut usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() || path.is_symlink() {
            continue;
        }
        if entry.file_name() == std::ffi::OsStr::new("__pycache__") {
            if std::fs::remove_dir_all(&path).is_ok() {
                *removed += 1;
            }
            continue;
        }
        purge_pycache(&path, removed);
    }
}

/// Delete the byte-compiled caches under every directory the transaction will
/// uninstall, so uv's post-uninstall `rmdir` cannot abort on bytecode that no
/// RECORD owns (`os error 39`). `.pyc` files are regenerated by the
/// interpreter on the next import and RECORD-listed ones are already exempt
/// from payload verification (`installer::is_pycache_pyc_entry`), so this is
/// non-destructive.
///
/// Returns the number of `__pycache__` directories removed.
pub fn pre_clean_for_uninstall(dists: &[TransactionDist]) -> usize {
    let mut removed = 0usize;
    for dist in dists {
        for root in &dist.roots {
            purge_pycache(root, &mut removed);
        }
        purge_pycache(&dist.dist_info, &mut removed);
    }
    removed
}

/// After a FAILED transaction: delete the directories of every distribution
/// whose `.dist-info` uv already removed. Those directories are orphans — no
/// metadata owns them and no `__init__.py` survives — and leaving them makes
/// the package importable as a PEP 420 namespace package.
///
/// Only directories snapshotted by [`inventory_transaction`] are touched, and
/// only when the owning metadata is gone, so a distribution uv left intact is
/// never removed.
///
/// Returns the paths removed.
pub fn rollback_half_removed(dists: &[TransactionDist]) -> Vec<PathBuf> {
    let mut removed = Vec::new();
    for dist in dists {
        if dist.dist_info.exists() {
            // uv did not get as far as removing this distribution's metadata;
            // the distribution is still owned, so leave it alone.
            continue;
        }
        for root in &dist.roots {
            if !root.exists() {
                continue;
            }
            if std::fs::remove_dir_all(root).is_ok() {
                removed.push(root.clone());
            }
        }
    }
    removed
}

/// Terminal handler for a failed replay: roll the half-removed distributions
/// back to a clean ABSENCE, record `broken`, and annotate the error with what
/// was removed.
pub fn fail_repair(
    share: &Path,
    bundle: &str,
    dists: &[TransactionDist],
    err: anyhow::Error,
) -> anyhow::Error {
    let removed = rollback_half_removed(dists);
    let detail = if removed.is_empty() {
        format!("{err:#}")
    } else {
        let names: Vec<String> = removed
            .iter()
            .map(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default()
                    .to_string()
            })
            .collect();
        format!(
            "{err:#}; removed half-uninstalled package dirs: {}",
            names.join(", ")
        )
    };
    mark_state(share, bundle, RepairState::Broken, &detail);
    if !removed.is_empty() {
        eprintln!(
            "retread install: {bundle} repair failed; removed {} half-uninstalled package \
             directory/ies so the modules are ABSENT rather than importable as empty \
             namespace packages",
            removed.len()
        );
    }
    err.context(detail)
}

/// Convenience wrapper used by the installer: snapshot + pre-clean in one call
/// against a live prefix's site-packages.
pub fn prepare_transaction(
    site_packages: &Path,
    names: &BTreeSet<String>,
) -> Result<(Vec<TransactionDist>, usize)> {
    let dists = inventory_transaction(site_packages, names);
    let pruned = pre_clean_for_uninstall(&dists);
    // Surface an unreadable site-packages rather than silently pre-cleaning
    // nothing; a repair that cannot see the prefix must not proceed blind.
    if site_packages.exists() {
        std::fs::read_dir(site_packages)
            .with_context(|| format!("reading site-packages {}", site_packages.display()))?;
    }
    Ok((dists, pruned))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "retread-repair-{label}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: &Path, body: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    /// A `torch` install: real sources plus a dist-info whose RECORD lists
    /// exactly those sources (and NOT the stale bytecode).
    fn fixture_torch(sp: &Path) {
        write(&sp.join("torch/__init__.py"), "__version__ = '2.5.1'\n");
        write(&sp.join("torch/nn/__init__.py"), "\n");
        write(&sp.join("torch/nn/modules/__init__.py"), "\n");
        write(
            &sp.join("torch-2.5.1.dist-info/METADATA"),
            "Name: torch\nVersion: 2.5.1\n\n",
        );
        write(
            &sp.join("torch-2.5.1.dist-info/RECORD"),
            "torch/__init__.py,sha256=x,10\n\
             torch/nn/__init__.py,sha256=x,1\n\
             torch/nn/modules/__init__.py,sha256=x,1\n\
             torch-2.5.1.dist-info/METADATA,sha256=x,30\n\
             torch-2.5.1.dist-info/RECORD,,\n",
        );
    }

    /// Stale bytecode from an earlier bare-python import: present on disk,
    /// absent from RECORD. This is the D2 trigger.
    fn stale_bytecode(sp: &Path) {
        write(
            &sp.join("torch/nn/modules/__pycache__/container.cpython-310.pyc"),
            "\0\0stale",
        );
        write(
            &sp.join("torch/__pycache__/__init__.cpython-310.pyc"),
            "\0\0stale",
        );
    }

    /// Uninstall the way uv does: delete exactly the RECORD paths, then
    /// `rmdir` the directories bottom-up. Fails with `os error 39` when an
    /// unowned file survives -- exactly the abort seen on hover-gpu.
    fn uv_style_uninstall(sp: &Path, dist_info: &Path) -> std::io::Result<()> {
        let record = std::fs::read_to_string(dist_info.join("RECORD")).unwrap();
        let mut dirs: BTreeSet<PathBuf> = BTreeSet::new();
        for line in record.lines() {
            let Some(token) = crate::installer::record_path_token(line) else {
                continue;
            };
            let path = sp.join(&token);
            let _ = std::fs::remove_file(&path);
            let mut cur = path.parent().map(Path::to_path_buf);
            while let Some(dir) = cur {
                if dir == sp {
                    break;
                }
                dirs.insert(dir.clone());
                cur = dir.parent().map(Path::to_path_buf);
            }
        }
        // Deepest first, like uv.
        for dir in dirs.iter().rev() {
            std::fs::remove_dir(dir)?;
        }
        Ok(())
    }

    /// GUARD (a) -- Fix 3: a distribution carrying stale, unowned `.pyc` must
    /// still uninstall cleanly, because the repair pre-cleans the bytecode
    /// before uv ever sees the prefix.
    ///
    /// Fails without the fix: with `pre_clean_for_uninstall` returning 0 (the
    /// v4.10.89 behaviour -- no pre-clean exists at all), the uv-style
    /// uninstall below aborts with `Directory not empty (os error 39)`.
    #[test]
    fn pre_clean_lets_a_dist_with_unowned_bytecode_uninstall() {
        let root = tempdir("preclean");
        let sp = root.join("site-packages");
        fixture_torch(&sp);
        stale_bytecode(&sp);

        // Without the pre-clean the uninstall aborts -- assert the fixture
        // really does reproduce the defect.
        let unpruned = uv_style_uninstall(&sp, &sp.join("torch-2.5.1.dist-info"));
        let err = unpruned.expect_err("stale bytecode must block the rmdir");
        assert_eq!(
            err.raw_os_error(),
            Some(39),
            "fixture must reproduce ENOTEMPTY, got {err}"
        );

        // Rebuild and run the real repair path.
        std::fs::remove_dir_all(&sp).unwrap();
        fixture_torch(&sp);
        stale_bytecode(&sp);

        let names: BTreeSet<String> = ["torch".to_string()].into_iter().collect();
        let (dists, pruned) = prepare_transaction(&sp, &names).unwrap();
        assert_eq!(dists.len(), 1, "torch must be inventoried: {dists:?}");
        assert_eq!(dists[0].roots, vec![sp.join("torch")]);
        assert!(
            pruned >= 2,
            "both __pycache__ dirs must be pruned, got {pruned}"
        );
        assert!(
            !sp.join("torch/nn/modules/__pycache__").exists(),
            "the blocking __pycache__ must be gone"
        );
        assert!(
            sp.join("torch/__init__.py").exists(),
            "pre-clean must not touch RECORD-owned sources"
        );

        uv_style_uninstall(&sp, &sp.join("torch-2.5.1.dist-info"))
            .expect("uninstall must succeed after the pre-clean");
        assert!(!sp.join("torch").exists(), "torch/ must be fully removed");

        std::fs::remove_dir_all(&root).ok();
    }

    /// GUARD (b) -- Fix 2: a transaction that dies after the uninstall phase
    /// must leave the package ABSENT, never as a PEP 420 namespace ghost.
    ///
    /// Fails without the fix: v4.10.89 has no rollback at all, so
    /// `site-packages/torch/` survives as `__pycache__` skeletons, `import
    /// torch` succeeds, and the marker is only written by the shell caller.
    #[test]
    fn failed_transaction_removes_the_orphan_tree_and_marks_broken() {
        let root = tempdir("rollback");
        let sp = root.join("site-packages");
        let share = root.join("share/retread");
        fixture_torch(&sp);

        let names: BTreeSet<String> = ["torch".to_string()].into_iter().collect();
        let (dists, _) = prepare_transaction(&sp, &names).unwrap();
        mark_state(
            &share,
            "hover-pack",
            RepairState::Repairing,
            "replaying 55 wheels",
        );
        assert_eq!(
            read_state(&share, "hover-pack"),
            Some(RepairState::Repairing)
        );

        // Simulate uv aborting mid-transaction: RECORD paths + dist-info are
        // gone, but unowned bytecode kept `torch/` alive.
        std::fs::remove_dir_all(sp.join("torch-2.5.1.dist-info")).unwrap();
        std::fs::remove_file(sp.join("torch/__init__.py")).unwrap();
        std::fs::remove_file(sp.join("torch/nn/__init__.py")).unwrap();
        std::fs::remove_file(sp.join("torch/nn/modules/__init__.py")).unwrap();
        write(
            &sp.join("torch/nn/modules/__pycache__/container.cpython-310.pyc"),
            "\0\0stale",
        );
        assert!(
            sp.join("torch").is_dir() && !sp.join("torch/__init__.py").exists(),
            "fixture must reproduce the namespace-package ghost"
        );

        let err = fail_repair(
            &share,
            "hover-pack",
            &dists,
            anyhow::anyhow!("uv pip install failed for bundle hover-pack (status exit status: 2)"),
        );

        assert!(
            !sp.join("torch").exists(),
            "the orphan namespace tree must be removed so import torch raises ModuleNotFoundError"
        );
        assert_eq!(read_state(&share, "hover-pack"), Some(RepairState::Broken));
        assert!(
            broken_path(&share, "hover-pack").exists(),
            "the activate.d backoff sentinel must exist"
        );
        assert!(
            format!("{err:#}").contains("removed half-uninstalled package dirs: torch"),
            "the error must name what was rolled back: {err:#}"
        );

        std::fs::remove_dir_all(&root).ok();
    }

    /// A distribution uv left INTACT must never be rolled back.
    #[test]
    fn rollback_spares_distributions_whose_metadata_survived() {
        let root = tempdir("spare");
        let sp = root.join("site-packages");
        fixture_torch(&sp);
        let names: BTreeSet<String> = ["torch".to_string()].into_iter().collect();
        let (dists, _) = prepare_transaction(&sp, &names).unwrap();

        assert!(rollback_half_removed(&dists).is_empty());
        assert!(sp.join("torch/__init__.py").exists());

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn replay_names_come_from_wheel_filenames() {
        let files = vec![
            PathBuf::from("/w/torch-2.5.1+cu124-999retread-cp310-cp310-linux_x86_64.whl"),
            PathBuf::from("/w/torch_vision-0.20.1-cp310-cp310-linux_x86_64.whl"),
            PathBuf::from("/w/notawheel"),
        ];
        let names = replay_dist_names(&files);
        assert!(names.contains("torch"));
        assert!(names.contains("torch-vision"));
        assert_eq!(names.len(), 2, "{names:?}");
    }

    #[test]
    fn record_top_level_dir_rejects_escapes_and_bare_files() {
        assert_eq!(record_top_level_dir("torch/nn/__init__.py"), Some("torch"));
        assert_eq!(record_top_level_dir("../../bin/uv"), None);
        assert_eq!(record_top_level_dir("/etc/passwd"), None);
        assert_eq!(record_top_level_dir("typing_extensions.py"), None);
    }

    #[test]
    fn inventory_skips_names_that_are_not_installed() {
        let root = tempdir("absent");
        let sp = root.join("site-packages");
        fixture_torch(&sp);
        let names: BTreeSet<String> = ["torchvision".to_string()].into_iter().collect();
        assert!(inventory_transaction(&sp, &names).is_empty());
        std::fs::remove_dir_all(&root).ok();
    }
}
