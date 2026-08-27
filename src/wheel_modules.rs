//! Map a wheel to the top-level Python modules it provides.
//!
//! P3 of the auto-import-deps work. `import_scan` answers *what does this code
//! import*; this answers *which distribution provides that module*, so the two
//! can be joined without a network call and without name heuristics.
//!
//! `cv2 -> opencv-python` falls out for free: opencv-python's wheel literally
//! ships a `cv2/` tree and names it in `top_level.txt`. No alias table can go
//! stale because the wheel itself is the source of truth.
//!
//! MEASURED HAZARD (do not "simplify" this away): `top_level.txt` is optional.
//! It is absent from many modern wheels, and `importlib.metadata`'s
//! `packages_distributions()` returns None for conda-installed distributions
//! that ship none. RECORD is the fallback, and it is the reason this reads two
//! sources rather than one.

use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};

/// Top-level module names a wheel provides, and where the answer came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WheelModules {
    /// Top-level importable names, e.g. `{"cv2"}` for opencv-python.
    pub modules: BTreeSet<String>,
    /// Submodules of those top levels, first component retained, e.g.
    /// `etils.epath`. Kept because an extra is frequently named after a
    /// submodule (`etils[epath]`), which is only recoverable if the second
    /// dotted component survives the scan.
    pub submodules: BTreeSet<String>,
    /// True when `top_level.txt` supplied the answer; false when it was
    /// derived from RECORD paths.
    pub from_top_level_txt: bool,
}

/// Names that appear at a wheel's root but are never importable modules.
fn is_metadata_dir(name: &str) -> bool {
    name.ends_with(".dist-info")
        || name.ends_with(".data")
        || name.ends_with(".egg-info")
        || name == "__pycache__"
}

/// Strip a nested install prefix so the real module root is visible.
///
/// MEASURED (2026-08-27, wheel store): cmeel-packaged wheels -- `coal`,
/// `hpp_fcl`, `cmeel_boost`, and 18 others in a 400-wheel sample -- ship an
/// entire prefix tree and put site-packages UNDER it:
/// `cmeel.prefix/lib/python3.12/site-packages/coal/__init__.py`.
/// Taking the first component yields `cmeel.prefix`, which is not importable,
/// so those wheels read as providing nothing and their real modules (`coal`,
/// `hppfcl`) are missed entirely.
fn strip_install_prefix(path: &str) -> &str {
    match path.split_once("site-packages/") {
        Some((_, rest)) if !rest.is_empty() => rest,
        _ => path,
    }
}

/// A RECORD path's first component, when it names a real module.
fn record_top_level(path: &str) -> Option<String> {
    let path = strip_install_prefix(path);
    let first = path.split('/').next()?;
    if first.is_empty() || is_metadata_dir(first) {
        return None;
    }
    if let Some(stem) = first.strip_suffix(".py") {
        // A single-file module at the wheel root.
        return (!stem.is_empty() && stem != "__init__").then(|| stem.to_string());
    }
    // A compiled EXTENSION module at the wheel root. MEASURED (2026-08-27):
    // manifold3d and xatlas ship exactly one file each --
    // `manifold3d.cpython-310-x86_64-linux-gnu.so` -- and are importable as
    // `manifold3d` / `xatlas`. CPython takes the name up to the FIRST dot, so
    // the ABI tag must be cut there, not with a plain extension strip.
    if first.ends_with(".so") || first.ends_with(".pyd") {
        let stem = first.split('.').next().unwrap_or_default();
        return (!stem.is_empty()).then(|| stem.to_string());
    }
    // A package directory only counts if the path descends into it.
    if path.contains('/') && !first.contains('.') {
        return Some(first.to_string());
    }
    None
}

/// Read the top-level modules a wheel provides.
///
/// Prefers `*.dist-info/top_level.txt`; falls back to RECORD paths. Returns an
/// empty set (not an error) for a wheel that provides no importable top level,
/// which is a real shape -- namespace-only and data-only wheels exist.
pub fn wheel_modules(wheel: &Path) -> Result<WheelModules> {
    let file = std::fs::File::open(wheel)
        .with_context(|| format!("opening wheel {}", wheel.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("reading wheel zip {}", wheel.display()))?;

    let names: Vec<String> = archive.file_names().map(str::to_string).collect();

    // 1. top_level.txt, when the wheel ships one.
    let top_level_entry = names
        .iter()
        .find(|n| n.ends_with(".dist-info/top_level.txt"))
        .cloned();
    let mut modules: BTreeSet<String> = BTreeSet::new();
    let mut from_top_level_txt = false;
    if let Some(entry) = top_level_entry {
        let mut buf = String::new();
        archive
            .by_name(&entry)
            .with_context(|| format!("reading {entry} from {}", wheel.display()))?
            .read_to_string(&mut buf)
            .with_context(|| format!("decoding {entry} from {}", wheel.display()))?;
        modules.extend(
            buf.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(str::to_string),
        );
        from_top_level_txt = !modules.is_empty();
    }

    // 2. RECORD fallback -- and the submodule pass, which runs either way.
    let mut submodules: BTreeSet<String> = BTreeSet::new();
    let record_entry = names.iter().find(|n| n.ends_with(".dist-info/RECORD")).cloned();
    if let Some(entry) = record_entry {
        let mut buf = String::new();
        archive
            .by_name(&entry)
            .with_context(|| format!("reading {entry} from {}", wheel.display()))?
            .read_to_string(&mut buf)
            .with_context(|| format!("decoding {entry} from {}", wheel.display()))?;
        for line in buf.lines() {
            let path = match line.split(',').next() {
                Some(p) if !p.is_empty() => p,
                _ => continue,
            };
            if let Some(top) = record_top_level(path) {
                if !from_top_level_txt {
                    modules.insert(top.clone());
                }
                // Retain the SECOND component: `etils/epath/__init__.py` ->
                // `etils.epath`, which is what makes `etils[epath]` inferable
                // against the dist's Provides-Extra list.
                let mut parts = strip_install_prefix(path).split('/');
                let (Some(first), Some(second)) = (parts.next(), parts.next()) else {
                    continue;
                };
                let second = second.strip_suffix(".py").unwrap_or(second);
                if !second.is_empty() && second != "__init__" && !second.contains('.') {
                    submodules.insert(format!("{first}.{second}"));
                }
            }
        }
    }

    Ok(WheelModules { modules, submodules, from_top_level_txt })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_wheel(label: &str, entries: &[(&str, &str)]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "retread-wheelmods-{label}-{}-{}.whl",
            std::process::id(),
            line!()
        ));
        let f = std::fs::File::create(&path).unwrap();
        let mut z = zip::ZipWriter::new(f);
        let opts: zip::write::FileOptions<()> =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for (name, body) in entries {
            z.start_file(*name, opts).unwrap();
            z.write_all(body.as_bytes()).unwrap();
        }
        z.finish().unwrap();
        path
    }

    #[test]
    fn top_level_txt_wins_and_is_reported_as_the_source() {
        let w = make_wheel(
            "toplevel",
            &[
                ("opencv_python-4.10.0.84.dist-info/top_level.txt", "cv2\n"),
                ("opencv_python-4.10.0.84.dist-info/RECORD", "cv2/__init__.py,,\n"),
                ("cv2/__init__.py", ""),
            ],
        );
        let m = wheel_modules(&w).unwrap();
        assert!(m.from_top_level_txt, "top_level.txt must be preferred");
        assert_eq!(m.modules, ["cv2".to_string()].into_iter().collect());
        let _ = std::fs::remove_file(w);
    }

    /// top_level.txt is OPTIONAL and absent from many modern wheels. RECORD is
    /// the fallback; without it a wheel reads as providing nothing.
    #[test]
    fn record_is_the_fallback_when_top_level_txt_is_absent() {
        let w = make_wheel(
            "record",
            &[(
                "somepkg-1.0.dist-info/RECORD",
                "somepkg/__init__.py,,\nsomepkg/core.py,,\nsomepkg-1.0.dist-info/METADATA,,\nlone.py,,\n",
            )],
        );
        let m = wheel_modules(&w).unwrap();
        assert!(!m.from_top_level_txt);
        assert_eq!(
            m.modules,
            ["somepkg".to_string(), "lone".to_string()].into_iter().collect(),
            "package dirs and root single-file modules both count"
        );
        assert!(
            !m.modules.iter().any(|x| x.contains("dist-info")),
            "metadata dirs must never read as modules: {:?}",
            m.modules
        );
        let _ = std::fs::remove_file(w);
    }

    /// The whole point of keeping the second dotted component: `etils[epath]`
    /// is only inferable if `etils.epath` survives the scan.
    #[test]
    fn second_dotted_component_is_retained_for_extra_inference() {
        let w = make_wheel(
            "submod",
            &[
                ("etils-1.13.0.dist-info/top_level.txt", "etils\n"),
                (
                    "etils-1.13.0.dist-info/RECORD",
                    "etils/__init__.py,,\netils/epath/__init__.py,,\netils/enp/util.py,,\n",
                ),
            ],
        );
        let m = wheel_modules(&w).unwrap();
        assert_eq!(m.modules, ["etils".to_string()].into_iter().collect());
        assert!(m.submodules.contains("etils.epath"), "got {:?}", m.submodules);
        assert!(m.submodules.contains("etils.enp"), "got {:?}", m.submodules);
        assert!(
            !m.submodules.iter().any(|s| s.ends_with("__init__")),
            "__init__ is not a submodule: {:?}",
            m.submodules
        );
        let _ = std::fs::remove_file(w);
    }

    /// A data-only wheel provides no importable top level. That is a real
    /// shape, not an error -- returning Err here would fail legitimate builds.
    #[test]
    fn data_only_wheel_yields_empty_not_error() {
        let w = make_wheel(
            "dataonly",
            &[(
                "datapack-1.0.dist-info/RECORD",
                "datapack-1.0.dist-info/METADATA,,\ndatapack-1.0.data/scripts/run.sh,,\n",
            )],
        );
        let m = wheel_modules(&w).unwrap();
        assert!(m.modules.is_empty(), "got {:?}", m.modules);
        let _ = std::fs::remove_file(w);
    }
    /// Diagnostic against REAL wheels from the store, not synthetic zips.
    /// `#[ignore]` because it depends on a populated local wheel store; run
    /// with `cargo test --lib real_wheels -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn real_wheels_from_the_store() {
        let index = std::path::Path::new(
            "/users/glvov/.claude/jobs/78d7a173/tmp/wheel_index.txt",
        );
        let Ok(list) = std::fs::read_to_string(index) else {
            eprintln!("no wheel index at {}; skipping", index.display());
            return;
        };
        let mut with_txt = 0usize;
        let mut from_record = 0usize;
        let mut empty = 0usize;
        let mut errs = 0usize;
        for line in list.lines().take(400) {
            let p = std::path::Path::new(line);
            match wheel_modules(p) {
                Ok(m) => {
                    if m.modules.is_empty() {
                        empty += 1;
                    } else if m.from_top_level_txt {
                        with_txt += 1;
                    } else {
                        from_record += 1;
                    }
                    let base = p.file_name().unwrap().to_string_lossy();
                    if m.modules.is_empty() {
                        eprintln!("  EMPTY: {base}");
                    }
                    if base.starts_with("opencv") || base.starts_with("etils") {
                        eprintln!(
                            "  {base}\n    modules={:?}\n    from_top_level_txt={}\n    submodules(sample)={:?}",
                            m.modules,
                            m.from_top_level_txt,
                            m.submodules.iter().take(6).collect::<Vec<_>>()
                        );
                    }
                }
                Err(e) => {
                    errs += 1;
                    eprintln!("  ERROR {}: {e:#}", p.display());
                }
            }
        }
        eprintln!(
            "\nTOTALS  top_level.txt={with_txt}  RECORD-fallback={from_record}  empty={empty}  errors={errs}"
        );
        assert_eq!(errs, 0, "no real wheel may fail to parse");
    }

    /// cmeel-packaged wheels nest site-packages under an install prefix.
    /// Measured on coal-3.0.3 in the real store: without stripping, this
    /// wheel reports NO modules while actually providing `coal` and `hppfcl`.
    #[test]
    fn nested_site_packages_prefix_is_stripped() {
        let w = make_wheel(
            "cmeel",
            &[(
                "coal-3.0.3.dist-info/RECORD",
                "cmeel.prefix/lib/python3.12/site-packages/coal/__init__.py,,\n\
                 cmeel.prefix/lib/python3.12/site-packages/coal/viewer.py,,\n\
                 cmeel.prefix/lib/python3.12/site-packages/hppfcl/__init__.py,,\n\
                 coal-3.0.3.dist-info/METADATA,,\n",
            )],
        );
        let m = wheel_modules(&w).unwrap();
        assert_eq!(
            m.modules,
            ["coal".to_string(), "hppfcl".to_string()].into_iter().collect(),
            "cmeel prefix must not hide the real modules"
        );
        assert!(m.submodules.contains("coal.viewer"), "got {:?}", m.submodules);
        let _ = std::fs::remove_file(w);
    }

    /// Root-level compiled extension modules. Measured on manifold3d-3.5.2 and
    /// xatlas-0.0.11 in the real store: each wheel's only payload file is
    /// `<name>.cpython-310-x86_64-linux-gnu.so`, importable as `<name>`.
    /// Without this they report NO modules at all.
    #[test]
    fn root_level_extension_modules_are_found() {
        let w = make_wheel(
            "extmod",
            &[(
                "manifold3d-3.5.2.dist-info/RECORD",
                "manifold3d.cpython-310-x86_64-linux-gnu.so,,\n\
                 manifold3d-3.5.2.dist-info/METADATA,,\n",
            )],
        );
        let m = wheel_modules(&w).unwrap();
        assert_eq!(
            m.modules,
            ["manifold3d".to_string()].into_iter().collect(),
            "the ABI tag must be cut at the FIRST dot, not treated as an extension"
        );
        let _ = std::fs::remove_file(w);
    }

}
