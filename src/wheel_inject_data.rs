//! Auto-inject the upstream repo's checkout-root tree into a wheel as
//! `.data/data/lib/<rel>` so the files land at `$CONDA_PREFIX/lib/<rel>`
//! when pip installs the wheel.
//!
//! Why this exists: many Python packages compute paths to non-Python
//! assets via `__file__` arithmetic relative to the source layout (the
//! IsaacLab pattern: `dirname(__file__) + *[".."] * 4 + "apps"`). When
//! `pip wheel` only packages the subdirectory's Python tree, those
//! sibling repo-root files (apps/*.kit, tools/, share/, etc.) never
//! land in the conda env, and `__file__` math from
//! `<env>/lib/python3.X/site-packages/<pkg>/...` lands at `<env>/lib/`
//! looking for paths that don't exist.
//!
//! Solution: at wheel build time, walk the WHOLE checkout root
//! (honoring its own `.gitignore`) and emit every non-ignored file as
//! a wheel `.data/data/lib/<rel>` entry. Pip extracts those to
//! `$CONDA_PREFIX/lib/<rel>` on install, putting `apps/foo.kit` at
//! `$CONDA_PREFIX/lib/apps/foo.kit` -- exactly where the `__file__ + 4
//! up + apps` arithmetic looks.
//!
//! Dedup: the bundle's entries each specify a `subdirectory` that's
//! already shipped via `pip wheel` into site-packages. Callers pass
//! those subdirectory paths in `skip_subdirs` so this walk doesn't
//! re-ship the Python package source tree (which would also blow up
//! disk for nothing). When several entries share a single checkout
//! root, the handler is responsible for calling this function exactly
//! once per (bundle, checkout_root) pair so duplicate `.data/` entries
//! don't appear across the bundle's wheels.

use std::collections::HashSet;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use ignore::WalkBuilder;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::wheel_inject::sha256_base64_urlsafe_nopad;

/// Hardcoded floor of names to skip at any depth. Belt-and-suspenders
/// for projects whose `.gitignore` forgets these (or for non-git path
/// entries where there is no `.gitignore` at all). Everything else
/// defers to the repo's own `.gitignore` via the `ignore` crate.
const ALWAYS_SKIP: &[&str] = &[
    "__pycache__",
    ".pixi",
    ".venv",
    "venv",
    "node_modules",
    "target", // rust build dir
];

/// One file pulled from the checkout root, ready to inject as a wheel
/// data entry.
struct DataEntry {
    /// Path INSIDE the wheel zip: `<dist>-<ver>.data/data/lib/<rel>`.
    /// Forward slashes, no leading slash.
    zip_path: String,
    bytes: Vec<u8>,
    sha_b64: String,
}

/// Inject the checkout-root tree into `src_wheel`, writing the result
/// to `dst_wheel`.
///
/// - `checkout_root`: the upstream repo's root (parent of all entry
///   subdirectories that get `pip wheel`'d into this bundle).
/// - `skip_subdirs`: subdirectories (relative to `checkout_root`) that
///   were already shipped as wheels by sibling entries in the same
///   bundle. The walk descends but emits nothing under these paths.
///
/// The wheel's distribution name + version are auto-discovered from
/// the `*.dist-info/` directory inside `src_wheel` (PEP 427 says
/// `<name>-<ver>.dist-info`, with `<name>` normalized the same way the
/// `<name>-<ver>.data/` directory expects).
///
/// No-op fast path: if the walk yields zero entries, `src_wheel` is
/// copied verbatim to `dst_wheel` so callers can treat the output path
/// uniformly.
///
/// Returns the number of files injected (for logging + audit).
pub fn inject_checkout_root_data(
    src_wheel: &Path,
    dst_wheel: &Path,
    checkout_root: &Path,
    skip_subdirs: &[PathBuf],
) -> Result<usize> {
    // Read the source wheel so we can snapshot its zip entries (we
    // never overwrite anything pip-wheel chose to ship), locate the
    // RECORD entry that needs rebuilding, and derive the
    // `<name>-<ver>.data/data/` prefix from the dist-info dir name.
    let bytes = fs::read(src_wheel).with_context(|| format!("reading {}", src_wheel.display()))?;
    let mut archive = ZipArchive::new(Cursor::new(&bytes))
        .with_context(|| format!("opening zip {}", src_wheel.display()))?;

    let mut existing: HashSet<String> = HashSet::new();
    let mut record_name: Option<String> = None;
    let mut dist_info_dir: Option<String> = None;
    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        existing.insert(name.clone());
        if record_name.is_none()
            && name.matches('/').count() == 1
            && name.ends_with(".dist-info/RECORD")
        {
            // <stem>.dist-info/RECORD -> <stem>.dist-info (without the
            // trailing /RECORD), then strip .dist-info to get
            // <name>-<ver> for the .data dir name.
            let dir = name.trim_end_matches("/RECORD").to_string();
            dist_info_dir = Some(dir);
            record_name = Some(name);
        }
    }
    let record_name = record_name
        .ok_or_else(|| anyhow!("no root-level .dist-info/RECORD in {}", src_wheel.display()))?;
    let dist_stem = dist_info_dir
        .as_ref()
        .and_then(|d| d.strip_suffix(".dist-info"))
        .ok_or_else(|| anyhow!("malformed dist-info dir in {}", src_wheel.display()))?;
    let data_prefix = format!("{}.data/data/lib/", dist_stem);

    // Normalize skip_subdirs once: relative-to-checkout-root, forward
    // slashes, trailing slash stripped, ignoring empty entries.
    let skip_set: HashSet<String> = skip_subdirs
        .iter()
        .filter_map(|p| {
            let s = path_to_forward_slash(p);
            if s.is_empty() { None } else { Some(s) }
        })
        .collect();

    // Walk the checkout root with the project's own .gitignore in
    // effect. standard_filters covers: hidden filter is OFF by default
    // (we want .github/ etc. visible -- it's the gitignore that should
    // decide), parent ignore files climb the tree, .git/info/exclude
    // is honored, and global excludes too.
    let mut data_entries: Vec<DataEntry> = Vec::new();
    let canon_root = fs::canonicalize(checkout_root)
        .with_context(|| format!("canonicalizing checkout root {}", checkout_root.display()))?;
    let walker = WalkBuilder::new(&canon_root)
        .standard_filters(true)
        .require_git(false) // honor .gitignore even outside a git working tree
        .hidden(false) // let .gitignore (not the leading-dot heuristic) decide
        .build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(error = %err, "skipping unreadable path under checkout root");
                continue;
            }
        };
        let path = entry.path();
        if path == canon_root {
            continue;
        }
        let rel = match path.strip_prefix(&canon_root) {
            Ok(r) => r,
            Err(_) => continue, // walker promised in-root paths; defensive skip
        };
        // ALWAYS_SKIP floor: applied at every depth.
        if rel
            .components()
            .any(|c| ALWAYS_SKIP.iter().any(|s| *s == c.as_os_str()))
        {
            continue;
        }
        // Sibling-wheel subdirectory skip (v0.21.0+ refinement):
        // when a file is under a subdir that's already shipped as a
        // wheel by another bundle entry, we only skip it if it's
        // PYTHON SOURCE (would be a duplicate of what pip wheel put
        // in site-packages). Non-Python files (Kit extension.toml,
        // data assets, config dirs, README) STILL get shipped under
        // `<env>/lib/<rel>` because Omniverse Kit (and similar tools
        // that resolve files via `__file__` arithmetic OR via an
        // extension-search path) need them at the on-disk layout the
        // upstream repo provides, NOT in site-packages.
        //
        // Concrete case: IsaacLab declares Kit extensions named
        // "isaaclab", "isaaclab_assets", etc. in its .kit experience
        // files. Kit scans `${app}/../source/<ext>/config/extension.toml`
        // to find them. Before v0.21, we skipped all of source/* (since
        // those subdirs are wheel'd) -- Kit found nothing and exited
        // with "untrusted extension". v0.21 ships the non-Python
        // sibling files (extension.toml, data/, etc.) so Kit's scan
        // succeeds without double-shipping the Python source.
        if path_is_under_any(rel, &skip_set) && is_python_artifact(rel) {
            continue;
        }
        let file_type = match entry.file_type() {
            Some(t) => t,
            None => continue, // root-of-walk sentinel; already filtered above
        };
        if !file_type.is_file() {
            continue; // dirs walked into automatically; symlinks dropped silently
        }
        let zip_path = format!("{}{}", data_prefix, path_to_forward_slash(rel));
        if existing.contains(&zip_path) {
            // Wheel already shipped something at this exact zip path
            // (would only happen via an `egg-info`-style trick); don't
            // overwrite.
            continue;
        }
        // Per-file errors (ENAMETOOLONG from a >255-char component,
        // EACCES on a permission glitch, a vanishing file mid-walk)
        // get logged + skipped rather than failing the whole bundle.
        // One bad upstream file shouldn't wedge every gigastrap solve.
        // The ALWAYS_SKIP floor + .gitignore filter cover the
        // common-clutter cases; this catches the long-tail.
        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    file = %path.display(),
                    "skipping unreadable checkout-root file during auto-data inject",
                );
                continue;
            }
        };
        let sha_b64 = sha256_base64_urlsafe_nopad(&bytes);
        data_entries.push(DataEntry { zip_path, bytes, sha_b64 });
    }

    // Stable ordering so on-disk wheel layouts and tests are
    // deterministic across filesystems.
    data_entries.sort_by(|a, b| a.zip_path.cmp(&b.zip_path));

    if data_entries.is_empty() {
        // Nothing to add. Copy through so the caller can uniformly
        // chain on the next pipeline phase.
        fs::copy(src_wheel, dst_wheel)?;
        return Ok(0);
    }

    tracing::info!(
        count = data_entries.len(),
        checkout = %canon_root.display(),
        dist_prefix = %data_prefix,
        "injecting checkout-root tree into wheel as .data/data/lib/* (lands at $PREFIX/lib/*)",
    );

    // Rebuild RECORD with one appended line per data entry. Same shape
    // as wheel_inject::extend_record: keep the wheel's own RECORD self-
    // line (empty hash/size) at the end of the file.
    let mut record_str = String::new();
    archive
        .by_name(&record_name)?
        .read_to_string(&mut record_str)?;
    let new_record = extend_record_with_data(&record_str, &record_name, &data_entries)?;

    // Two-pass writeback identical in shape to wheel_inject: copy
    // every original entry through (substituting RECORD for the
    // rebuilt one), then append new data entries.
    let dst_file = fs::File::create(dst_wheel)
        .with_context(|| format!("creating {}", dst_wheel.display()))?;
    let mut writer = ZipWriter::new(dst_file);
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        let compression = match entry.compression() {
            CompressionMethod::Stored => CompressionMethod::Stored,
            _ => CompressionMethod::Deflated,
        };
        let options = SimpleFileOptions::default().compression_method(compression);
        writer.start_file(&name, options)?;
        if name == record_name {
            writer.write_all(new_record.as_bytes())?;
        } else {
            std::io::copy(&mut entry, &mut writer)?;
        }
    }
    for data in &data_entries {
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        writer.start_file(&data.zip_path, options)?;
        writer.write_all(&data.bytes)?;
    }
    writer.finish()?.flush()?;
    Ok(data_entries.len())
}

fn path_to_forward_slash(p: &Path) -> String {
    let mut parts = Vec::new();
    for c in p.components() {
        parts.push(c.as_os_str().to_string_lossy().into_owned());
    }
    parts.join("/")
}

/// v0.21.0+: identify files that pip wheel would have shipped into
/// site-packages. Used by the `skip_subdirs` filter to drop ONLY
/// Python source/cache/build-meta files, while letting non-Python
/// siblings (Kit extension.toml, data/, config/, assets) through so
/// they land at `<env>/lib/<rel>` for tools that read them by path.
///
/// Conservative match: only files clearly produced by / consumed by
/// pip's build pipeline get classified as Python artifacts. Anything
/// else is treated as upstream-authored data and shipped.
fn is_python_artifact(rel: &Path) -> bool {
    // Any component in __pycache__ -> python bytecode cache.
    if rel
        .components()
        .any(|c| c.as_os_str() == "__pycache__")
    {
        return true;
    }
    let name = rel
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    // Cache + bytecode + stubs.
    if name.ends_with(".py")
        || name.ends_with(".pyi")
        || name.ends_with(".pyc")
        || name.ends_with(".pyo")
    {
        return true;
    }
    // Packaging metadata pip itself reads/writes.
    matches!(
        name,
        "setup.py"
            | "setup.cfg"
            | "pyproject.toml"
            | "MANIFEST.in"
            | "PKG-INFO"
            | "Pipfile"
            | "Pipfile.lock"
    )
}

/// True when `rel` (path inside the checkout root) is at or beneath any
/// of `skip` (also rooted at the checkout root). Both use forward
/// slashes already.
fn path_is_under_any(rel: &Path, skip: &HashSet<String>) -> bool {
    if skip.is_empty() {
        return false;
    }
    let rel_str = path_to_forward_slash(rel);
    if rel_str.is_empty() {
        return false;
    }
    for s in skip {
        if rel_str == *s || rel_str.starts_with(&format!("{s}/")) {
            return true;
        }
    }
    false
}

/// Rebuild RECORD: keep every existing line except the RECORD self-
/// line, append one PEP 376 line per data entry, then put the RECORD
/// self-line back at the end (empty hash/size by convention).
fn extend_record_with_data(
    record: &str,
    record_name: &str,
    extras: &[DataEntry],
) -> Result<String> {
    let mut out = String::with_capacity(record.len() + extras.len() * 96);
    let mut record_line: Option<String> = None;
    for line in record.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            out.push_str(line);
            continue;
        }
        let path = trimmed.splitn(2, ',').next().unwrap_or("");
        if path == record_name {
            record_line = Some(line.to_string());
            continue;
        }
        out.push_str(line);
    }
    let newline = if record.contains("\r\n") { "\r\n" } else { "\n" };
    for data in extras {
        out.push_str(&format!(
            "{},sha256={},{}{}",
            data.zip_path,
            data.sha_b64,
            data.bytes.len(),
            newline,
        ));
    }
    if let Some(line) = record_line {
        out.push_str(&line);
    } else {
        bail!("RECORD has no entry for itself ({record_name})");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny in-memory wheel that looks like what pip wheel
    /// produces: one python module + a dist-info with METADATA, WHEEL,
    /// and RECORD. Mirrors the fixture style in wheel_inject tests.
    fn build_stub_wheel(dist_name: &str, dist_version: &str) -> Vec<u8> {
        let dist_info = format!("{dist_name}-{dist_version}.dist-info");
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut w = ZipWriter::new(Cursor::new(&mut buf));
            let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
            w.start_file(format!("{dist_name}/__init__.py"), opts)
                .unwrap();
            w.write_all(b"").unwrap();
            w.start_file(format!("{dist_info}/METADATA"), opts).unwrap();
            w.write_all(format!("Metadata-Version: 2.1\nName: {dist_name}\nVersion: {dist_version}\n").as_bytes())
                .unwrap();
            w.start_file(format!("{dist_info}/WHEEL"), opts).unwrap();
            w.write_all(b"Wheel-Version: 1.0\nGenerator: stub\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
                .unwrap();
            w.start_file(format!("{dist_info}/RECORD"), opts).unwrap();
            // Real wheels list every shipped file in RECORD; for the
            // test we only need the self-line in the right shape so the
            // rebuilder can find and preserve it.
            w.write_all(format!("{dist_info}/RECORD,,\n").as_bytes())
                .unwrap();
            w.finish().unwrap();
        }
        buf
    }

    fn write_file(root: &Path, rel: &str, body: &[u8]) {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, body).unwrap();
    }

    /// End-to-end on a tmp checkout: a .gitignore'd file is skipped, a
    /// regular file lands at the right wheel zip path with a correct
    /// RECORD sha256 line, ALWAYS_SKIP enforcement works without any
    /// .gitignore at all. .gitignore itself rides along (projects
    /// commit it; the user explicitly opted into the maximalist
    /// auto-inject and asked us not to overfit).
    #[test]
    fn injects_respecting_gitignore_and_always_skip() -> Result<()> {
        let parent = tempdir_in_target("retread-data-inject-gitignore")?;
        let checkout = parent.join("repo");
        fs::create_dir_all(&checkout)?;
        // Create a fake checkout-root: an apps/ dir we want shipped, a
        // build/ dir that's gitignored, and a __pycache__ dir that
        // ALWAYS_SKIP should drop even though .gitignore says nothing.
        write_file(&checkout, ".gitignore", b"build/\n");
        write_file(&checkout, "apps/foo.kit", b"app-bytes\n");
        write_file(&checkout, "apps/sub/bar.kit", b"sub-bytes\n");
        write_file(&checkout, "build/junk.txt", b"shouldnt-ship\n");
        write_file(
            &checkout,
            "__pycache__/x.cpython-311.pyc",
            b"shouldnt-ship\n",
        );

        // Wheel files live OUTSIDE the checkout root so the walker
        // doesn't sweep them up.
        let src_wheel = parent.join("stub.whl");
        let dst_wheel = parent.join("stub.injected.whl");
        fs::write(&src_wheel, build_stub_wheel("foo", "1.0"))?;

        let injected = inject_checkout_root_data(
            &src_wheel,
            &dst_wheel,
            &checkout,
            &[],
        )?;
        // 2 apps/ files + 1 .gitignore (project content, intentionally
        // shipped).
        assert_eq!(injected, 3, "expected apps/foo.kit, apps/sub/bar.kit, .gitignore");

        let out_bytes = fs::read(&dst_wheel)?;
        let mut archive = ZipArchive::new(Cursor::new(&out_bytes))?;
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        assert!(names.contains(&"foo-1.0.data/data/lib/apps/foo.kit".to_string()));
        assert!(names.contains(&"foo-1.0.data/data/lib/apps/sub/bar.kit".to_string()));
        assert!(names.contains(&"foo-1.0.data/data/lib/.gitignore".to_string()));
        assert!(!names.iter().any(|n| n.contains("build/")));
        assert!(!names.iter().any(|n| n.contains("__pycache__")));

        // RECORD updated with PEP 376 lines for the new data entries.
        let mut record = String::new();
        archive
            .by_name("foo-1.0.dist-info/RECORD")?
            .read_to_string(&mut record)?;
        for path in [
            "foo-1.0.data/data/lib/apps/foo.kit,sha256=",
            "foo-1.0.data/data/lib/apps/sub/bar.kit,sha256=",
        ] {
            assert!(record.contains(path), "RECORD missing {path}:\n{record}");
        }
        Ok(())
    }

    /// v0.21.0+: under skip_subdirs, ONLY python source/cache/build
    /// files get skipped. Non-Python siblings (extension.toml, data
    /// files, config dirs, assets) still get shipped so tools that
    /// read them by path (Omniverse Kit extension scanner) find them.
    #[test]
    fn skip_subdirs_ships_non_python_files_under_skipped_dir() -> Result<()> {
        let parent = tempdir_in_target("retread-data-inject-non-python")?;
        let checkout = parent.join("repo");
        fs::create_dir_all(&checkout)?;
        // Mimic IsaacLab's source/isaaclab/ layout.
        write_file(&checkout, "source/isaaclab/isaaclab/__init__.py", b"# py\n");
        write_file(&checkout, "source/isaaclab/setup.py", b"# py\n");
        write_file(&checkout, "source/isaaclab/pyproject.toml", b"# build-meta\n");
        // Kit extension config + data that Kit needs to find at the
        // on-disk source layout.
        write_file(
            &checkout,
            "source/isaaclab/config/extension.toml",
            b"[package]\nversion=\"4.5.22\"\n",
        );
        write_file(&checkout, "source/isaaclab/data/robot.usd", b"USD-FAKE\n");
        write_file(&checkout, "source/isaaclab/docs/README.md", b"# docs\n");

        let src_wheel = parent.join("stub.whl");
        let dst_wheel = parent.join("stub.injected.whl");
        fs::write(&src_wheel, build_stub_wheel("baz", "1.0"))?;
        let injected = inject_checkout_root_data(
            &src_wheel,
            &dst_wheel,
            &checkout,
            &[PathBuf::from("source/isaaclab")],
        )?;
        // Expected: extension.toml + robot.usd + README.md (3 non-Python)
        // Skipped: __init__.py, setup.py, pyproject.toml (Python artifacts)
        assert_eq!(injected, 3, "expected 3 non-Python files shipped, py source skipped");
        let out_bytes = fs::read(&dst_wheel)?;
        let mut archive = ZipArchive::new(Cursor::new(&out_bytes))?;
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        // Non-Python files INSIDE the skipped subdir landed at .data/.
        assert!(names.contains(
            &"baz-1.0.data/data/lib/source/isaaclab/config/extension.toml".to_string()
        ), "extension.toml must be shipped so Kit finds the extension");
        assert!(names.contains(
            &"baz-1.0.data/data/lib/source/isaaclab/data/robot.usd".to_string()
        ));
        assert!(names.contains(
            &"baz-1.0.data/data/lib/source/isaaclab/docs/README.md".to_string()
        ));
        // Python files were correctly skipped (they're in the wheel
        // already via pip wheel). Only check INJECTED entries (under
        // `<dist>-<ver>.data/data/lib/`) -- the stub wheel itself
        // ships `baz/__init__.py` at the zip root, which is unrelated.
        let injected_names: Vec<&String> = names
            .iter()
            .filter(|n| n.starts_with("baz-1.0.data/data/lib/"))
            .collect();
        assert!(!injected_names.iter().any(|n| n.ends_with("/__init__.py")));
        assert!(!injected_names.iter().any(|n| n.ends_with("/setup.py")));
        assert!(!injected_names.iter().any(|n| n.ends_with("/pyproject.toml")));
        Ok(())
    }

    /// skip_subdirs == ["pkg_a", "pkg_b"] drops every file under either.
    #[test]
    fn skip_subdirs_drops_sibling_wheel_sources() -> Result<()> {
        let parent = tempdir_in_target("retread-data-inject-skipsubdirs")?;
        let checkout = parent.join("repo");
        fs::create_dir_all(&checkout)?;
        write_file(&checkout, "pkg_a/inside.py", b"x\n");
        write_file(&checkout, "pkg_a/nested/deep.py", b"x\n");
        write_file(&checkout, "pkg_b/inside.py", b"x\n");
        write_file(&checkout, "outside/kept.txt", b"keep-me\n");

        let src_wheel = parent.join("stub.whl");
        let dst_wheel = parent.join("stub.injected.whl");
        fs::write(&src_wheel, build_stub_wheel("bar", "2.0"))?;

        let injected = inject_checkout_root_data(
            &src_wheel,
            &dst_wheel,
            &checkout,
            &[PathBuf::from("pkg_a"), PathBuf::from("pkg_b")],
        )?;
        assert_eq!(injected, 1);

        let out_bytes = fs::read(&dst_wheel)?;
        let mut archive = ZipArchive::new(Cursor::new(&out_bytes))?;
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        assert!(names.contains(&"bar-2.0.data/data/lib/outside/kept.txt".to_string()));
        assert!(!names.iter().any(|n| n.contains("pkg_a")));
        assert!(!names.iter().any(|n| n.contains("pkg_b")));
        Ok(())
    }

    /// No injectable files -> output is a verbatim copy of the input
    /// wheel (no RECORD rewrite).
    #[test]
    fn empty_walk_falls_through_as_copy() -> Result<()> {
        let parent = tempdir_in_target("retread-data-inject-empty")?;
        let checkout = parent.join("repo");
        fs::create_dir_all(&checkout)?;
        let src_wheel = parent.join("stub.whl");
        let dst_wheel = parent.join("stub.injected.whl");
        fs::write(&src_wheel, build_stub_wheel("baz", "0.1"))?;
        let count = inject_checkout_root_data(
            &src_wheel,
            &dst_wheel,
            &checkout,
            &[],
        )?;
        assert_eq!(count, 0);
        assert_eq!(fs::read(&src_wheel)?, fs::read(&dst_wheel)?);
        Ok(())
    }

    /// Test-only tmpdir under target/ so we don't fight permissions on
    /// /tmp and so failed tests leave artifacts that are easy to find.
    fn tempdir_in_target(slug: &str) -> Result<PathBuf> {
        let base = std::env::temp_dir().join(format!(
            "{slug}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir_all(&base)?;
        Ok(base)
    }
}
