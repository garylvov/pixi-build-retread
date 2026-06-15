//! v2.0.0 courier: the `retread install` subcommand.
//!
//! Invoked from the courier conda package's post-link script at env link
//! time: `pixi-build-retread install --lock <lock> --prefix <prefix>`.
//! Reads the committed lock and installs the bundle's PyPI wheels into the
//! active conda env via uv (fast hardlink), resolving the root
//! requirements against the shipped find-links wheels + the recorded index
//! chain, into the conda env so shared transitives stay conda-provided.
//! Idempotent: a content-hash marker makes a re-link a no-op.

use std::ffi::OsString;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use crate::lock::RetreadLock;

/// Fallback primary index when the lock carries no index_urls.
pub(crate) use crate::handler::PUBLIC_PYPI;

/// Build the argument list for `uv pip install` as a pure function (no I/O,
/// no spawn). The returned vector does NOT include argv[0] (the uv binary
/// itself) -- callers pass it directly to `Command::new`.
///
/// # S1 – conda-covered transitive shielding
/// `lock.conda_run_deps` lists names whose wheels are provided by conda (e.g.
/// torch, torchaudio). We pass `--no-install-package <name>` for each one.
/// uv still *resolves* them (so it can compute the closure) but does NOT
/// fetch or install a PyPI wheel for them, ensuring conda stays the
/// authoritative source and no version skew is introduced.
/// This is simpler and more correct than a constraints file: we don't need
/// to express a version pin because conda is already authoritative -- we just
/// need uv to leave those dists alone.
///
/// # S3 – index chain replay
/// We replay `lock.index_urls` verbatim: first entry becomes `--index-url`,
/// the rest become `--extra-index-url`. We do NOT hard-code public PyPI as
/// the primary; it is used only when `index_urls` is empty. This preserves
/// the resolution priority the backend recorded (e.g. pypi.nvidia.com is
/// primary for isaac bundles).
pub(crate) fn build_uv_args(
    lock: &RetreadLock,
    prefix: &Path,
    wheels_dir: Option<&Path>,
    overrides_file: Option<&Path>,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        "pip".into(),
        "install".into(),
        "--python".into(),
        prefix.join("bin").join("python").into(),
    ];

    // Optional find-links directory for locally-shipped wheels.
    if let Some(dir) = wheels_dir {
        args.push("--find-links".into());
        args.push(dir.into());
    }

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

    // S1: shield conda-provided names from uv installation.
    // --no-install-package tells uv to resolve but skip fetching/installing
    // those packages. Conda is already the authoritative install for them.
    for dep in &lock.conda_run_deps {
        args.push("--no-install-package".into());
        args.push(dep.name.as_str().into());
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

    let args = build_uv_args(&lock, prefix, wheels_dir_opt, overrides_file);

    eprintln!(
        "retread install: {} -> {} ({} root reqs)",
        lock.bundle,
        prefix.display(),
        lock.root_requirements.len()
    );

    let status = Command::new(&uv)
        .args(&args)
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
            }],
            conda_run_deps,
            index_urls,
            prerelease,
        }
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

    // S1: conda-covered names are excluded via --no-install-package.
    #[test]
    fn s1_conda_run_deps_shielded() {
        let lock = make_lock(
            vec![
                CondaDep {
                    name: "torch".into(),
                    spec: ">=2.7,<3".into(),
                },
                CondaDep {
                    name: "torchaudio".into(),
                    spec: ">=2.7,<3".into(),
                },
            ],
            vec!["https://pypi.nvidia.com".into(), PUBLIC_PYPI.into()],
            BTreeMap::new(),
        );
        let prefix = PathBuf::from("/fake/prefix");
        let args = build_uv_args(&lock, &prefix, None, None);
        let strs = argv_strings(&args);

        let shielded = flag_values(&strs, "--no-install-package");
        assert!(
            shielded.contains(&"torch".to_string()),
            "torch must be shielded; got {shielded:?}"
        );
        assert!(
            shielded.contains(&"torchaudio".to_string()),
            "torchaudio must be shielded; got {shielded:?}"
        );

        // Verify no --constraint flag (we use --no-install-package instead).
        assert!(
            !strs.contains(&"--constraint".to_string()),
            "should not emit --constraint; S1 uses --no-install-package"
        );
    }

    // S1: when conda_run_deps is empty, no --no-install-package appears.
    #[test]
    fn s1_no_conda_deps_no_shield_flags() {
        let lock = make_lock(vec![], vec![PUBLIC_PYPI.into()], BTreeMap::new());
        let prefix = PathBuf::from("/fake/prefix");
        let args = build_uv_args(&lock, &prefix, None, None);
        let strs = argv_strings(&args);
        assert!(
            !strs.contains(&"--no-install-package".to_string()),
            "no --no-install-package when conda_run_deps is empty"
        );
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
        let args = build_uv_args(&lock, &prefix, None, None);
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
        let args = build_uv_args(&lock, &prefix, None, None);
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
        let args = build_uv_args(&lock, &prefix, None, None);
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
        let args = build_uv_args(&lock, &prefix, None, Some(ovr.as_path()));
        let strs = argv_strings(&args);
        let ovr_vals = flag_values(&strs, "--overrides");
        assert_eq!(ovr_vals, vec![ovr.to_string_lossy().into_owned()]);
    }

    // Root requirements appear in argv.
    #[test]
    fn root_requirements_in_argv() {
        let lock = make_lock(vec![], vec![PUBLIC_PYPI.into()], BTreeMap::new());
        let prefix = PathBuf::from("/fake/prefix");
        let args = build_uv_args(&lock, &prefix, None, None);
        let strs = argv_strings(&args);
        assert!(
            strs.contains(&"mypackage==1.0.0".to_string()),
            "root requirements must appear in argv"
        );
    }
}
