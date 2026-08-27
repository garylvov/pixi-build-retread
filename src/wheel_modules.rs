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

/// A RECORD path's first component, when it names a real module.
fn record_top_level(path: &str) -> Option<String> {
    let first = path.split('/').next()?;
    if first.is_empty() || is_metadata_dir(first) {
        return None;
    }
    if let Some(stem) = first.strip_suffix(".py") {
        // A single-file module at the wheel root.
        return (!stem.is_empty() && stem != "__init__").then(|| stem.to_string());
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
                let mut parts = path.split('/');
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
}
