//! Auto-inject source-root files that pip wheel failed to include.
//!
//! Why this exists: a recurring upstream packaging pattern declares
//! `packages=["isaaclab"]` in setup.py (no `find_packages()`), with no
//! MANIFEST.in covering siblings like `config/extension.toml` or the
//! package's own subpackages. `pip wheel` produces a near-empty wheel
//! that imports cleanly only via an editable install (where Python
//! discovers the real files on disk by path). retread targets the
//! conda-installable case where the wheel must be self-sufficient, so
//! we top up the wheel with everything that lives in the source tree
//! but isn't already inside.
//!
//! The deny-list is conservative: VCS, build artifacts, tests, docs,
//! examples, and packaging files are skipped at every path component,
//! so a stray `tests/` inside a package dir is excluded even though
//! its sibling `__init__.py` is included.
//!
//! Two-pass writeback: append every new file to the wheel zip, then
//! rebuild RECORD with the original lines + one new entry per injected
//! file (PEP 427 base64-urlsafe-nopad sha256). The wheel's own RECORD
//! line (empty hash/size) is preserved at the end.

use std::collections::HashSet;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

/// Path-component names that NEVER ship in a wheel.
const DENY_COMPONENTS: &[&str] = &[
    // VCS
    ".git",
    ".github",
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
    ".hg",
    ".svn",
    // Python build artifacts
    "build",
    "dist",
    "__pycache__",
    // Test scaffolding
    "tests",
    "test",
    "conftest.py",
    // Docs
    "docs",
    "doc",
    "documentation",
    // Examples / demos / scripts (data-only repos should declare
    // those as `data/` or `assets/` instead).
    "examples",
    "example",
    "demo",
    "demos",
    "scripts",
    // Local envs / IDE / lock dirs.
    //
    // `env` is NOT here: unlike `.venv`/`venv` it is an ordinary Python
    // package name, and denying it at every level silently ate
    // `robojudo/config/g1/env/` -- a git-tracked subpackage -- which broke
    // `import robojudo` outright while its siblings survived. A virtualenv
    // named `env` lives at the source root, so it is denied there only
    // (DENY_TOP_LEVEL).
    ".pixi",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "__pypackages__",
    ".vscode",
    ".idea",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    // Packaging / install inputs (the wheel itself is the output of
    // these; shipping them inside the wheel would be incoherent).
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "MANIFEST.in",
    "Makefile",
    "tox.ini",
    "pixi.toml",
    "pixi.lock",
];

/// File-suffix patterns to skip at every level. Cheap substring match
/// because none of these legitimately appear inside a desired filename.
const DENY_SUFFIXES: &[&str] = &[".egg-info", ".dist-info", ".pyc", ".pyo", ".swp"];

/// Top-level filenames to skip (project-root noise that we don't want
/// in every package install).
const DENY_TOP_LEVEL: &[&str] = &[
    // A virtualenv conventionally named `env` sits at the source root; a
    // nested `env` is a package (see DENY_COMPONENTS).
    "env",
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "COPYING",
    "NOTICE",
    "README",
    "README.md",
    "README.rst",
    "README.txt",
    "CHANGELOG",
    "CHANGELOG.md",
    "CHANGELOG.rst",
    "CHANGES",
    "CHANGES.md",
    "HISTORY",
    "HISTORY.md",
    "AUTHORS",
    "CONTRIBUTORS",
    "requirements.txt",
    "requirements-dev.txt",
    "dev-requirements.txt",
];

/// One file pulled from the source tree, ready to append to the wheel.
struct Extra {
    /// Path INSIDE the wheel zip (forward slashes, no leading slash).
    zip_path: String,
    bytes: Vec<u8>,
    sha_b64: String,
}

/// Read `src` (a `.whl`), walk `source_root`, and write a new wheel at
/// `dst` containing every file from `source_root` that wasn't already
/// in the wheel and that the deny-list doesn't filter out. RECORD is
/// updated with one new entry per injected file (PEP 376 / PEP 427:
/// `<path>,sha256=<b64-urlsafe-nopad>,<size>`).
///
/// No-op fast path: if the walk produces zero extras, `src` is copied
/// verbatim to `dst`.
pub fn inject_source_extras(src: &Path, dst: &Path, source_root: &Path) -> Result<()> {
    let bytes = fs::read(src).with_context(|| format!("reading {}", src.display()))?;
    let mut archive = ZipArchive::new(Cursor::new(&bytes))
        .with_context(|| format!("opening zip {}", src.display()))?;

    // Snapshot every path already in the wheel; we won't overwrite
    // anything pip-wheel chose to ship. Also pick out the RECORD entry
    // name so we can rebuild it at the end.
    let mut existing: HashSet<String> = HashSet::new();
    let mut record_name: Option<String> = None;
    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        existing.insert(name.clone());
        if record_name.is_none()
            && name.matches('/').count() == 1
            && name.ends_with(".dist-info/RECORD")
        {
            record_name = Some(name);
        }
    }
    let record_name = record_name
        .ok_or_else(|| anyhow!("no root-level .dist-info/RECORD in {}", src.display()))?;

    // Walk the source root, collecting every (rel_path, bytes) the
    // deny-list permits and the wheel doesn't already carry.
    let extras = collect_extras(source_root, &existing)?;

    if extras.is_empty() {
        // Nothing to add. Copy through so the caller can treat the
        // output path uniformly. Atomic: copy to a same-directory temp
        // file then rename, so a process/node death mid-copy never
        // leaves a truncated file at `dst` for a later run's mtime-only
        // `is_fresh()` check to mistake for a valid cache hit.
        let (tmp, _) = crate::wheel::create_atomic_tmp(dst)?;
        fs::copy(src, &tmp)?;
        crate::wheel::commit_atomic_write(&tmp, dst)?;
        return Ok(());
    }

    tracing::info!(
        count = extras.len(),
        source = %source_root.display(),
        "injecting source-root files into wheel that pip wheel didn't ship",
    );

    // Read original RECORD; we'll append a line per extra and put
    // RECORD's own line (empty hash/size) back at the end.
    let mut record_str = String::new();
    archive
        .by_name(&record_name)?
        .read_to_string(&mut record_str)?;
    let new_record = extend_record(&record_str, &record_name, &extras)?;

    // Rewrite the wheel: copy every original entry through, swap
    // RECORD for the new one, then append the extras at the end.
    //
    // Atomic write: build the zip in a same-directory temp file, then
    // rename over `dst` only once every byte is flushed, so a process/
    // node death mid-write can never leave a truncated wheel at `dst`
    // for a later run's mtime-only `is_fresh()` check to mistake for a
    // valid cache hit (the exact failure mode proven in run 9).
    let (tmp, dst_file) = crate::wheel::create_atomic_tmp(dst)?;
    let mut writer = ZipWriter::new(dst_file);

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        let compression = match entry.compression() {
            CompressionMethod::Stored => CompressionMethod::Stored,
            _ => CompressionMethod::Deflated,
        };
        // Source-built wheels inherit the build clock in their ZIP entry
        // timestamps. Normalize every copied entry so identical source bytes
        // produce an identical injected wheel across worktrees and replay.
        let options = SimpleFileOptions::default()
            .compression_method(compression)
            .last_modified_time(zip::DateTime::default());
        writer.start_file(&name, options)?;
        if name == record_name {
            writer.write_all(new_record.as_bytes())?;
        } else {
            std::io::copy(&mut entry, &mut writer)?;
        }
    }
    for extra in &extras {
        let options = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .last_modified_time(zip::DateTime::default());
        writer.start_file(&extra.zip_path, options)?;
        writer.write_all(&extra.bytes)?;
    }
    let mut finished = writer.finish()?;
    finished.flush()?;
    drop(finished);
    crate::wheel::commit_atomic_write(&tmp, dst)?;
    Ok(())
}

/// Walk `source_root`, returning every file whose relative path the
/// deny-list permits and which isn't already in `existing`.
fn collect_extras(source_root: &Path, existing: &HashSet<String>) -> Result<Vec<Extra>> {
    let mut out = Vec::new();
    walk(source_root, source_root, existing, &mut out)?;
    // Stable order so test snapshots and on-disk layouts are
    // deterministic across runs.
    out.sort_by(|a, b| a.zip_path.cmp(&b.zip_path));
    Ok(out)
}

fn walk(root: &Path, dir: &Path, existing: &HashSet<String>, out: &mut Vec<Extra>) -> Result<()> {
    let read = match fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", dir.display())),
    };
    for entry in read {
        let entry = entry?;
        let path = entry.path();
        let rel = path
            .strip_prefix(root)
            .map_err(|_| anyhow!("path {} escaped root {}", path.display(), root.display()))?;
        if rel.as_os_str().is_empty() {
            continue;
        }
        let is_top = rel.components().count() == 1;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // A directory carrying `__init__.py` is an importable package, and
        // dropping it breaks the consuming environment at import time rather
        // than merely slimming the wheel. Real trees do ship packages named
        // `env`, `tests`, `examples` and `scripts` (rsl_rl_g/env and
        // protomotions/tests among them), so the clutter deny list never
        // applies to one. Non-package directories with those names are still
        // skipped, which is the case the list exists for.
        if is_denied(&name_str, is_top) && !path.join("__init__.py").is_file() {
            continue;
        }
        // Skip path components anywhere up the chain too (covers
        // symlinks into a denied subtree etc.).
        // Same rule for every ancestor: a denied component only disqualifies
        // this path when that ancestor is not itself an importable package.
        let mut denied_ancestor = root.to_path_buf();
        let mut skip = false;
        for component in rel.components() {
            denied_ancestor.push(component);
            if is_denied(&component.as_os_str().to_string_lossy(), false)
                && !denied_ancestor.join("__init__.py").is_file()
            {
                skip = true;
                break;
            }
        }
        if skip {
            continue;
        }
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            walk(root, &path, existing, out)?;
        } else if file_type.is_file() {
            let zip_path = to_zip_path(rel);
            if existing.contains(&zip_path) {
                continue;
            }
            let bytes = fs::read(&path)
                .with_context(|| format!("reading source file {}", path.display()))?;
            let sha_b64 = sha256_base64_urlsafe_nopad(&bytes);
            out.push(Extra {
                zip_path,
                bytes,
                sha_b64,
            });
        }
        // Skip symlinks and other oddities silently. A symlink pointing
        // outside the source tree could leak files; declining to follow
        // is the safe call.
    }
    Ok(())
}

fn to_zip_path(rel: &Path) -> String {
    let mut parts = Vec::new();
    for c in rel.components() {
        parts.push(c.as_os_str().to_string_lossy().into_owned());
    }
    parts.join("/")
}

/// True if `name` should be skipped. `top_level` toggles the extra
/// project-root noise filter (LICENSE, README, etc.).
fn is_denied(name: &str, top_level: bool) -> bool {
    if name.is_empty() {
        return true;
    }
    if DENY_COMPONENTS.contains(&name) {
        return true;
    }
    if DENY_SUFFIXES.iter().any(|s| name.ends_with(s)) {
        return true;
    }
    if top_level && DENY_TOP_LEVEL.contains(&name) {
        return true;
    }
    false
}

/// Rebuild RECORD with a line appended per injected extra. The wheel's
/// own RECORD entry (which has empty hash/size by convention) stays at
/// the end so RECORD-line ordering matches the by-convention layout.
fn extend_record(record: &str, record_name: &str, extras: &[Extra]) -> Result<String> {
    let mut out = String::with_capacity(record.len() + extras.len() * 96);
    let mut record_line: Option<String> = None;
    for line in record.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            out.push_str(line);
            continue;
        }
        let path = trimmed.split(',').next().unwrap_or("");
        if path == record_name {
            // Hold RECORD's self-line until the end.
            record_line = Some(line.to_string());
            continue;
        }
        out.push_str(line);
    }
    let newline = if record.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    };
    for extra in extras {
        out.push_str(&format!(
            "{},sha256={},{}{}",
            extra.zip_path,
            extra.sha_b64,
            extra.bytes.len(),
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

/// PEP 376 / PEP 427: sha256 digest encoded as base64 urlsafe with no
/// padding. Inline to avoid pulling a base64 crate for one call site.
pub(crate) fn sha256_base64_urlsafe_nopad(bytes: &[u8]) -> String {
    let digest = {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hasher.finalize()
    };
    base64_urlsafe_nopad(&digest)
}

fn base64_urlsafe_nopad(bytes: &[u8]) -> String {
    const ALPHA: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity((bytes.len() * 4).div_ceil(3));
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8) | (bytes[i + 2] as u32);
        out.push(ALPHA[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3f) as usize] as char);
        out.push(ALPHA[((n >> 6) & 0x3f) as usize] as char);
        out.push(ALPHA[(n & 0x3f) as usize] as char);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let n = (bytes[i] as u32) << 16;
        out.push(ALPHA[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3f) as usize] as char);
    } else if rem == 2 {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8);
        out.push(ALPHA[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3f) as usize] as char);
        out.push(ALPHA[((n >> 6) & 0x3f) as usize] as char);
    }
    out
}

/// Convenience wrapper used by `handler::materialize_and_rewrite`:
/// resolve `source_root` to its canonical absolute form (so the walk
/// doesn't trip on `..` segments) before invoking the inject pass.
pub fn inject(src: &Path, dst: &Path, source_root: &Path) -> Result<PathBuf> {
    let canon = source_root
        .canonicalize()
        .with_context(|| format!("canonicalizing source root {}", source_root.display()))?;
    inject_source_extras(src, dst, &canon)?;
    Ok(dst.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deny_list_blocks_common_clutter() {
        // Tests / docs / VCS / build artifacts are skipped at every
        // level, so a stray tests/ or .git/ inside a package dir is
        // still excluded.
        for name in [
            ".git",
            ".github",
            "tests",
            "test",
            "docs",
            "doc",
            "build",
            "dist",
            "__pycache__",
            "setup.py",
            "pyproject.toml",
            "node_modules",
            ".pixi",
        ] {
            assert!(is_denied(name, false), "{name} should be denied");
        }
        // Hidden caches via suffix.
        assert!(is_denied("foo.egg-info", false));
        assert!(is_denied("bar.dist-info", false));
        assert!(is_denied("module.pyc", false));
        // Top-level-only filters: README at root, allowed deeper.
        assert!(is_denied("README.md", true));
        assert!(!is_denied("README.md", false));
        // `env` is denied only at the source root (a virtualenv lives
        // there); nested it is an ordinary package name.
        assert!(is_denied("env", true), "a root-level env/ is a virtualenv");
        assert!(
            !is_denied("env", false),
            "a nested env/ is a package and must ship"
        );

        // Legitimate data dirs pass through.
        for name in [
            "config",
            "data",
            "assets",
            "resources",
            "templates",
            "isaaclab",
        ] {
            assert!(!is_denied(name, false), "{name} should pass");
            assert!(!is_denied(name, true), "{name} should pass at top");
        }
    }

    #[test]
    fn base64_matches_python_urlsafe_nopad() {
        // sha256(b"") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        // python: base64.urlsafe_b64encode(...).rstrip(b"=") ->
        // 47DEQpj8HBSa-_TImW-5JCeuQeRkm5NMpJWZG3hSuFU
        let s = sha256_base64_urlsafe_nopad(b"");
        assert_eq!(s, "47DEQpj8HBSa-_TImW-5JCeuQeRkm5NMpJWZG3hSuFU");
    }

    #[test]
    fn zip_path_uses_forward_slashes() {
        let p = Path::new("config").join("extension.toml");
        assert_eq!(to_zip_path(&p), "config/extension.toml");
        let p = Path::new("isaaclab").join("envs").join("manager.py");
        assert_eq!(to_zip_path(&p), "isaaclab/envs/manager.py");
    }

    #[test]
    fn injection_normalizes_source_wheel_timestamps() -> Result<()> {
        fn build(timestamp: zip::DateTime) -> Result<Vec<u8>> {
            let mut buf = Vec::new();
            let mut wheel = ZipWriter::new(Cursor::new(&mut buf));
            let options = SimpleFileOptions::default()
                .compression_method(CompressionMethod::Deflated)
                .last_modified_time(timestamp);
            wheel.start_file("foo/__init__.py", options)?;
            wheel.write_all(b"x = 1\n")?;
            wheel.start_file("foo-0.1.0.dist-info/RECORD", options)?;
            wheel.write_all(b"foo/__init__.py,sha256=dummy,6\nfoo-0.1.0.dist-info/RECORD,,\n")?;
            wheel.finish()?;
            Ok(buf)
        }

        let base =
            std::env::temp_dir().join(format!("retread-inject-timestamp-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let source_root = base.join("source");
        let wheels = base.join("wheels");
        fs::create_dir_all(source_root.join("foo"))?;
        fs::create_dir_all(&wheels)?;
        fs::write(source_root.join("foo/extra.py"), b"y = 2\n")?;

        let early = wheels.join("early.whl");
        let late = wheels.join("late.whl");
        fs::write(
            &early,
            build(zip::DateTime::from_date_and_time(2025, 1, 2, 3, 4, 6)?)?,
        )?;
        fs::write(
            &late,
            build(zip::DateTime::from_date_and_time(2026, 7, 8, 9, 10, 12)?)?,
        )?;
        let early_out = wheels.join("early.injected.whl");
        let late_out = wheels.join("late.injected.whl");
        inject_source_extras(&early, &early_out, &source_root)?;
        inject_source_extras(&late, &late_out, &source_root)?;
        assert_eq!(fs::read(early_out)?, fs::read(late_out)?);

        let _ = fs::remove_dir_all(&base);
        Ok(())
    }

    /// End-to-end: build a minimal wheel in memory, inject from a tmp
    /// source tree, verify the new entries appear and RECORD has
    /// well-formed PEP 376 lines for each.
    #[test]
    fn injects_missing_files_and_extends_record() -> Result<()> {
        // Build a tiny "as-if-pip-wheel-built" zip with just an
        // __init__.py and dist-info. Mirrors the IsaacLab case where
        // packages=["isaaclab"] gives a wheel with only the top-level
        // __init__.py.
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut w = ZipWriter::new(Cursor::new(&mut buf));
            let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
            w.start_file("foo/__init__.py", opts)?;
            w.write_all(b"# top-level only\n")?;
            w.start_file("foo-0.1.0.dist-info/METADATA", opts)?;
            w.write_all(b"Metadata-Version: 2.1\nName: foo\nVersion: 0.1.0\n")?;
            w.start_file("foo-0.1.0.dist-info/RECORD", opts)?;
            // Original RECORD: __init__ + METADATA + RECORD itself.
            w.write_all(
                b"foo/__init__.py,sha256=dummy,17\n\
                  foo-0.1.0.dist-info/METADATA,sha256=dummy,50\n\
                  foo-0.1.0.dist-info/RECORD,,\n",
            )?;
            w.finish()?;
        }

        // Stage a tmp source root containing both an existing path
        // (foo/__init__.py, which must NOT be re-injected) and three
        // new ones (foo/sub/mod.py, config/extension.toml, tests/* —
        // tests is denied), plus README at top (denied).
        let tmp = std::env::temp_dir().join(format!("retread-inject-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("foo/sub"))?;
        fs::create_dir_all(tmp.join("config"))?;
        fs::create_dir_all(tmp.join("tests"))?;
        fs::write(tmp.join("foo/__init__.py"), b"# top-level only\n")?;
        fs::write(tmp.join("foo/sub/mod.py"), b"x = 1\n")?;
        fs::write(
            tmp.join("config/extension.toml"),
            b"[package]\nversion = '0.1.0'\n",
        )?;
        fs::write(tmp.join("tests/test_x.py"), b"def test(): pass\n")?;
        fs::write(tmp.join("README.md"), b"# foo\n")?;

        let src_whl = tmp.join("foo-0.1.0-py3-none-any.whl");
        let dst_whl = tmp.join("foo-0.1.0-py3-none-any.injected.whl");
        fs::write(&src_whl, &buf)?;
        inject_source_extras(&src_whl, &dst_whl, &tmp)?;

        // Read back, assert the missing files were injected and the
        // denied ones (tests/, README.md) were not.
        let bytes = fs::read(&dst_whl)?;
        let mut zip = ZipArchive::new(Cursor::new(&bytes))?;
        let mut names: Vec<String> = (0..zip.len())
            .map(|i| zip.by_index(i).unwrap().name().to_string())
            .collect();
        names.sort();
        assert!(
            names.iter().any(|n| n == "config/extension.toml"),
            "config/extension.toml must be injected; got {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "foo/sub/mod.py"),
            "foo/sub/mod.py must be injected; got {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.starts_with("tests/")),
            "tests/ must be denied; got {names:?}"
        );
        assert!(
            !names.iter().any(|n| n == "README.md"),
            "README.md at root must be denied; got {names:?}"
        );

        // Existing wheel entries pass through untouched.
        let mut existing_init = String::new();
        zip.by_name("foo/__init__.py")?
            .read_to_string(&mut existing_init)?;
        assert_eq!(existing_init, "# top-level only\n");

        // RECORD picked up one line per extra and kept its self-line
        // (empty hash/size) at the end.
        let mut record = String::new();
        zip.by_name("foo-0.1.0.dist-info/RECORD")?
            .read_to_string(&mut record)?;
        assert!(
            record.contains("config/extension.toml,sha256="),
            "record:\n{record}"
        );
        assert!(
            record.contains("foo/sub/mod.py,sha256="),
            "record:\n{record}"
        );
        assert!(
            record
                .lines()
                .last()
                .unwrap()
                .starts_with("foo-0.1.0.dist-info/RECORD,,"),
            "RECORD self-line must be last; got:\n{record}",
        );

        // Each new RECORD line is well-formed: path,sha256=<b64>,<size>
        for line in record.lines() {
            if line.starts_with("config/") || line.starts_with("foo/sub/") {
                let parts: Vec<&str> = line.splitn(3, ',').collect();
                assert_eq!(parts.len(), 3, "bad record line: {line}");
                assert!(parts[1].starts_with("sha256="), "bad hash field: {line}");
                let size: usize = parts[2].parse().expect("size must be integer");
                assert!(size > 0);
            }
        }

        let _ = fs::remove_dir_all(&tmp);
        Ok(())
    }

    #[test]
    fn empty_walk_falls_through_as_copy() -> Result<()> {
        // If source_root has nothing the deny-list permits, the wheel
        // is copied verbatim. In production the wheel lives under
        // download_dir, NOT inside the source tree, so keep that split
        // here -- otherwise the wheel file itself would be walked and
        // injected as an extra.
        let base =
            std::env::temp_dir().join(format!("retread-inject-empty-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let src_root = base.join("src");
        let wheels = base.join("wheels");
        fs::create_dir_all(&src_root)?;
        fs::create_dir_all(&wheels)?;
        // Source root contains only denied things.
        fs::create_dir_all(src_root.join("tests"))?;
        fs::write(src_root.join("README.md"), b"x")?;

        // Minimal valid wheel, placed OUTSIDE the source root.
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut w = ZipWriter::new(Cursor::new(&mut buf));
            let opts = SimpleFileOptions::default();
            w.start_file("p/__init__.py", opts)?;
            w.write_all(b"x")?;
            w.start_file("p-0.1.0.dist-info/RECORD", opts)?;
            w.write_all(b"p/__init__.py,sha256=dummy,1\np-0.1.0.dist-info/RECORD,,\n")?;
            w.finish()?;
        }
        let src = wheels.join("p-0.1.0.whl");
        let dst = wheels.join("p-0.1.0.out.whl");
        fs::write(&src, &buf)?;
        inject_source_extras(&src, &dst, &src_root)?;
        assert_eq!(fs::read(&src)?, fs::read(&dst)?);
        let _ = fs::remove_dir_all(&base);
        Ok(())
    }
}
