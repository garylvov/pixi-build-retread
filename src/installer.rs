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
    let marker = share.join(lock.marker_name());
    let want = format!("{:x}", Sha256::digest(&raw));
    if let Ok(have) = std::fs::read_to_string(&marker)
        && have.trim() == want
    {
        let msg = format!("retread install: {} already current; skipping", lock.bundle);
        eprintln!("{msg}");
        crate::status::phase(
            lock_path.parent().unwrap_or(std::path::Path::new(".")),
            &lock.bundle,
            &msg,
        );
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
        let out = Command::new(&uv)
            .args(["pip", "list", "--format", "freeze", "--python"])
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

    let args = build_uv_args(
        &lock,
        prefix,
        wheels_dir_opt,
        overrides_file,
        constraints_file,
        excludes_file,
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
    if !status.success() {
        bail!(
            "uv pip install failed for bundle {} (status {status})",
            lock.bundle
        );
    }

    std::fs::create_dir_all(&share).ok();
    std::fs::write(&marker, want)
        .with_context(|| format!("writing marker {}", marker.display()))?;
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
        let args = build_uv_args(&lock, &prefix, None, None, Some(con.as_path()), None);
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
        let args = build_uv_args(&lock, &prefix, None, None, None, None);
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
        let args = build_uv_args(&lock, &prefix, None, None, None, Some(exc.as_path()));
        let strs = argv_strings(&args);
        assert_eq!(
            flag_values(&strs, "--excludes"),
            vec![exc.to_string_lossy().into_owned()]
        );
        // absent when not provided.
        let args = build_uv_args(&lock, &prefix, None, None, None, None);
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
        let args = build_uv_args(&lock, &prefix, None, None, None, None);
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
        let args = build_uv_args(&lock, &prefix, None, None, None, None);
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
        let args = build_uv_args(&lock, &prefix, None, None, None, None);
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
        let args = build_uv_args(&lock, &prefix, None, Some(ovr.as_path()), None, None);
        let strs = argv_strings(&args);
        let ovr_vals = flag_values(&strs, "--overrides");
        assert_eq!(ovr_vals, vec![ovr.to_string_lossy().into_owned()]);
    }

    // Root requirements appear in argv.
    #[test]
    fn root_requirements_in_argv() {
        let lock = make_lock(vec![], vec![PUBLIC_PYPI.into()], BTreeMap::new());
        let prefix = PathBuf::from("/fake/prefix");
        let args = build_uv_args(&lock, &prefix, None, None, None, None);
        let strs = argv_strings(&args);
        assert!(
            strs.contains(&"mypackage==1.0.0".to_string()),
            "root requirements must appear in argv"
        );
    }
}
