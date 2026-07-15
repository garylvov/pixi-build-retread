//! Wheel METADATA surgery.
//!
//! Given a downloaded `.whl` and a [`RelaxPolicy`], produce a new `.whl`
//! whose `*.dist-info/METADATA` file has every `Requires-Dist:` line's
//! exact `==X.Y.Z` pins widened to a range per the policy. The RECORD
//! file is updated in lock-step so pip's hash check still passes at
//! install time.
//!
//! Rationale: `retread-relax` historically only affected the conda
//! `run_dependencies` we emit. uv on the consumer side ignores that --
//! it reads the wheel's own METADATA from `$PREFIX/.../dist-info/`.
//! Rewriting the wheel itself loosens both sides simultaneously.

use std::fmt::Write as _;
use std::io::{Cursor, Read, Write};
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, anyhow};
use sha2::{Digest, Sha256};
use uv_pep508::Requirement;
use uv_pep508::uv_pep440::{Operator, Version};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::config::RelaxPolicy;

/// 3-way outcome for the per-`Requires-Dist` mapper passed to
/// [`rewrite_wheel_with`] / [`rewrite_metadata_text_with`].
///
/// - `Keep` — emit the original line unchanged (no rewrite cost).
/// - `Replace(s)` — substitute `s` as the new requirement value; `s` MUST
///   differ from the original line (never `Replace(identical-bytes)`; that
///   would flip the `did_change` signal and drift courier's `ShadowSrc`
///   decision for every wheel).
/// - `Drop` — omit the line entirely from the emitted METADATA. Used by
///   Phase 2.8 to strip orphan direct-URL `Requires-Dist` lines whose
///   target is absent from the resolved bundle closure.
#[derive(Debug, PartialEq, Eq)]
pub enum LineAction {
    Keep,
    Replace(String),
    Drop,
}

/// Read `src` (a `.whl`), apply `relax` to every `Requires-Dist:` line in
/// the root-level `*.dist-info/METADATA` entry, and write the result to
/// `dst`. The destination wheel's RECORD entry for METADATA is updated
/// with the new sha256 + size so `pip install` still verifies cleanly.
///
/// Returns the new sha256 of the rewritten wheel (useful for recipe
/// generation).
pub fn rewrite_wheel(src: &Path, dst: &Path, relax: RelaxPolicy) -> Result<String> {
    rewrite_wheel_with(src, dst, &|line| match relax_pep508(line, relax).ok() {
        None => LineAction::Keep,
        Some(s) if s == line => LineAction::Keep,
        Some(s) => LineAction::Replace(s),
    })
    .map(|(sha, _)| sha)
}

/// Generic core of [`rewrite_wheel`]: apply `map` to every
/// `Requires-Dist:` value. [`LineAction::Keep`] leaves a line unchanged,
/// [`LineAction::Replace`] substitutes, [`LineAction::Drop`] omits the line
/// entirely. When no line changes or is dropped, the output is a hard link
/// to the input where possible (same filesystem) so isaac-scale wheels cost
/// nothing to "ship"; falls back to a copy. v1.6.1: emit-pypi uses this to
/// bake its override semantics directly into the shipped wheels' METADATA.
pub(crate) fn rewrite_wheel_with(
    src: &Path,
    dst: &Path,
    map: &dyn Fn(&str) -> LineAction,
) -> Result<(String, bool)> {
    let bytes = std::fs::read(src).with_context(|| format!("reading {}", src.display()))?;
    let mut archive = ZipArchive::new(Cursor::new(&bytes))
        .with_context(|| format!("opening zip {}", src.display()))?;

    // Find the root-level dist-info directory (one with exactly one `/`).
    let mut metadata_name = None;
    let mut record_name = None;
    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        let name = entry.name();
        if name.matches('/').count() != 1 {
            continue;
        }
        if name.ends_with(".dist-info/METADATA") {
            metadata_name = Some(name.to_string());
        } else if name.ends_with(".dist-info/RECORD") {
            record_name = Some(name.to_string());
        }
    }
    let metadata_name = metadata_name
        .ok_or_else(|| anyhow!("no root-level .dist-info/METADATA in {}", src.display()))?;
    let record_name = record_name
        .ok_or_else(|| anyhow!("no root-level .dist-info/RECORD in {}", src.display()))?;

    // Read original METADATA + RECORD.
    let mut metadata_str = String::new();
    archive
        .by_name(&metadata_name)?
        .read_to_string(&mut metadata_str)?;
    let mut record_str = String::new();
    archive
        .by_name(&record_name)?
        .read_to_string(&mut record_str)?;

    // Rewrite METADATA via the mapper.
    let new_metadata = rewrite_metadata_text_with(&metadata_str, map)?;
    if new_metadata == metadata_str {
        // Nothing changed; hard-link when possible (free for
        // multi-GB wheels on the same filesystem), else copy. Atomic:
        // land the hard-link/copy at a same-directory temp path first,
        // then rename over `dst` -- never remove `dst` before its
        // replacement is fully in place, so a process/node death
        // mid-operation leaves either the old `dst` or the new one,
        // never neither/a truncated stub.
        let tmp = crate::wheel::atomic_tmp_path(dst);
        if std::fs::hard_link(src, &tmp).is_err() {
            let _ = std::fs::remove_file(&tmp);
            std::fs::copy(src, &tmp)?;
        }
        crate::wheel::commit_atomic_write(&tmp, dst)?;
        let h = sha256_hex(&bytes);
        return Ok((h, false));
    }
    let new_metadata_bytes = new_metadata.as_bytes();
    // RECORD hash lines use PEP 376's urlsafe-base64-nopad form (what
    // bdist_wheel writes). The hex form previously written here was
    // tolerated by pip on the conda path but is wrong per spec, and
    // blueprint-mode wheels are consumed by uv directly.
    let new_metadata_sha = crate::wheel_inject::sha256_base64_urlsafe_nopad(new_metadata_bytes);
    let new_record = update_record_line(
        &record_str,
        &metadata_name,
        &new_metadata_sha,
        new_metadata_bytes.len(),
    )?;

    // Write new wheel zip. Iterate every entry; substitute METADATA/RECORD,
    // pass everything else through. Preserve compression and timestamps so
    // tools that inspect the wheel see a minimally-different file.
    //
    // Atomic write: build in a same-directory temp file, then rename
    // over `dst` only once every byte is flushed -- a process/node
    // death mid-write can never leave a truncated wheel at `dst` for a
    // later run's mtime-only `is_fresh()` check to mistake for a valid
    // cache hit (the exact failure mode proven in run 9: a corrupted
    // `*.autodata.whl` fed straight into this function as `src`).
    let (tmp, dst_file) = crate::wheel::create_atomic_tmp(dst)?;
    let mut writer = ZipWriter::new(dst_file);

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        let compression = match entry.compression() {
            CompressionMethod::Stored => CompressionMethod::Stored,
            _ => CompressionMethod::Deflated,
        };
        // Pin the entry timestamp explicitly (the fixed 1980 DOS epoch) so the
        // rewrite is byte-deterministic regardless of whether the `zip` crate's
        // `time` feature ever gets unified on by a future dependency -- without
        // this, `default()` would fall back to wall-clock time and silently
        // break the shadow-rewrite cache's byte-identity guarantee.
        let options = SimpleFileOptions::default()
            .compression_method(compression)
            .last_modified_time(zip::DateTime::default());

        writer.start_file(&name, options)?;
        if name == metadata_name {
            writer.write_all(new_metadata_bytes)?;
        } else if name == record_name {
            writer.write_all(new_record.as_bytes())?;
        } else {
            std::io::copy(&mut entry, &mut writer)?;
        }
    }
    let mut finished = writer.finish()?;
    finished.flush()?;
    drop(finished);
    crate::wheel::commit_atomic_write(&tmp, dst)?;

    // sha256 of the rewritten wheel file (for recipe.yaml's source: sha256).
    let dst_bytes = std::fs::read(dst)?;
    Ok((sha256_hex(&dst_bytes), true))
}

/// Generic form: apply `map` to every `Requires-Dist:` value.
/// [`LineAction::Keep`] (including for unparseable lines) keeps the
/// original line, so a confusing line can never corrupt a wheel.
/// [`LineAction::Drop`] omits the line entirely (used for orphan URL deps).
fn rewrite_metadata_text_with(content: &str, map: &dyn Fn(&str) -> LineAction) -> Result<String> {
    let mut out = String::with_capacity(content.len());
    let mut in_headers = true;
    for line in content.split_inclusive('\n') {
        if in_headers && (line == "\n" || line == "\r\n") {
            // Blank line ends the RFC 822 headers; body follows untouched.
            in_headers = false;
        }
        if in_headers && let Some(value) = line.strip_prefix("Requires-Dist: ") {
            let trimmed = value.trim_end_matches(['\r', '\n']);
            match map(trimmed) {
                LineAction::Keep => {}
                LineAction::Replace(rewritten) => {
                    out.push_str("Requires-Dist: ");
                    out.push_str(&rewritten);
                    // Preserve the original line ending.
                    if line.ends_with("\r\n") {
                        out.push_str("\r\n");
                    } else {
                        out.push('\n');
                    }
                    continue;
                }
                LineAction::Drop => {
                    // Omit the line entirely (orphan URL dep strip).
                    continue;
                }
            }
        }
        out.push_str(line);
    }
    Ok(out)
}

/// Parse one PEP 508 requirement string. If it carries a single exact
/// `==X.Y.Z` specifier, widen it per `policy` and return the rebuilt
/// requirement (still PEP 508 syntax). All other shapes pass through.
pub(crate) fn relax_pep508(raw: &str, policy: RelaxPolicy) -> Result<String> {
    let req: Requirement =
        Requirement::from_str(raw).map_err(|e| anyhow!("parsing `{raw}`: {e}"))?;
    // `python` is off-limits to every relax policy; see the matching
    // guard in src/relax.rs::translate. METADATA rewrites must stay in
    // lock-step or pip on the consumer side would still see the widened
    // python pin and re-resolve against it.
    if req.name.as_ref().eq_ignore_ascii_case("python") {
        return Ok(raw.to_string());
    }
    let Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) = req.version_or_url.as_ref() else {
        return Ok(raw.to_string());
    };
    let specs_vec: Vec<_> = specs.iter().collect();

    // Single exact `==X` pin -> widen per policy.
    if specs_vec.len() == 1 && *specs_vec[0].operator() == Operator::Equal {
        if let Some(new_spec) = widen_exact_to_pep508(specs_vec[0].version(), policy) {
            return Ok(rebuild_requirement(&req, &new_spec));
        }
        return Ok(raw.to_string());
    }

    // Range specs: only StrongMajor / CondaAware mutate them (strip
    // upper-bound clauses). Major / Minor / Patch / None pass through.
    // TODO(conda-aware): CondaAware currently strips uppers
    // unconditionally, identical to StrongMajor. The intended design
    // probes the workspace's conda channels per-spec and only strips
    // when zero candidates satisfy the bound -- see
    // RelaxPolicy::CondaAware doc in src/config.rs.
    if matches!(policy, RelaxPolicy::StrongMajor | RelaxPolicy::CondaAware) {
        let stripped = strip_upper_bounds_pep508(&specs_vec);
        if stripped.is_empty() {
            // All bounds were uppers -> emit the requirement with no
            // version constraint at all (effectively `pkg *`).
            return Ok(rebuild_requirement(&req, ""));
        }
        let joined = stripped.join(",");
        // Detect no-op (kept set identical to original): cheap and
        // avoids spuriously-modified METADATA hashes when nothing changed.
        let original = specs_vec
            .iter()
            .map(|s| format!("{}{}", op_str(*s.operator()), s.version()))
            .collect::<Vec<_>>()
            .join(",");
        if joined == original {
            return Ok(raw.to_string());
        }
        return Ok(rebuild_requirement(&req, &joined));
    }

    Ok(raw.to_string())
}

fn op_str(op: Operator) -> &'static str {
    match op {
        Operator::Equal => "==",
        Operator::NotEqual => "!=",
        Operator::LessThan => "<",
        Operator::LessThanEqual => "<=",
        Operator::GreaterThan => ">",
        Operator::GreaterThanEqual => ">=",
        Operator::TildeEqual => "~=",
        Operator::ExactEqual => "===",
        Operator::EqualStar => "==",
        Operator::NotEqualStar => "!=",
    }
}

/// Same shape as relax.rs's strip_upper_bounds but emits PEP 508
/// spec strings rather than VersionSpecifier objects. Dropped: `<X`,
/// `<=X`. `~=X.Y[.Z...]` keeps its full lower bound. Everything else
/// passes through.
fn strip_upper_bounds_pep508(specs: &[&uv_pep508::uv_pep440::VersionSpecifier]) -> Vec<String> {
    let mut kept = Vec::with_capacity(specs.len());
    for spec in specs {
        match spec.operator() {
            Operator::LessThan | Operator::LessThanEqual => {}
            Operator::TildeEqual => {
                let r = spec.version().release();
                if r.is_empty() {
                    continue;
                }
                kept.push(format!(">={}", spec.version()));
            }
            other => kept.push(format!("{}{}", op_str(*other), spec.version())),
        }
    }
    kept
}

fn widen_exact_to_pep508(v: &Version, policy: RelaxPolicy) -> Option<String> {
    let r = v.release();
    if r.is_empty() {
        return None;
    }
    let major = r[0];
    let minor = r.get(1).copied().unwrap_or(0);
    let patch = r.get(2).copied().unwrap_or(0);
    // Elide trailing-zero segments per PEP 440 normalization, so
    // `Pillow==12.0.0` minor-relaxes to `Pillow>=12,<13` (not `>=12.0,<13`).
    let lower_minor = if minor > 0 {
        format!("{major}.{minor}")
    } else {
        format!("{major}")
    };
    let lower_patch = if patch > 0 {
        format!("{major}.{minor}.{patch}")
    } else if minor > 0 {
        format!("{major}.{minor}")
    } else {
        format!("{major}")
    };
    // The `*WithLastResort` variants behave IDENTICALLY to their base
    // at translate/widen-exact time; the cascade is a separate
    // post-translate probe pass in handler.rs. Grouped here so the
    // pattern stays exhaustive and so adding more bases doesn't drift
    // the two sides apart.
    match policy {
        RelaxPolicy::None => None,
        // Tiered cascade emits at patch widening on the wheel side
        // too; conda-side escalation is independent (no wheel rewrite).
        RelaxPolicy::Patch
        | RelaxPolicy::PatchWithLastResort
        | RelaxPolicy::PatchThenMinorThenMajorThenLastResort => {
            Some(format!(">={lower_patch},<{major}.{}", minor + 1))
        }
        RelaxPolicy::Minor | RelaxPolicy::MinorWithLastResort => {
            Some(format!(">={lower_minor},<{}", major + 1))
        }
        // TODO(conda-aware): grouped with Major/StrongMajor as a
        // stopgap; the real probe-and-decide logic is not wired up.
        RelaxPolicy::Major
        | RelaxPolicy::MajorWithLastResort
        | RelaxPolicy::StrongMajor
        | RelaxPolicy::CondaAware => Some(format!(">={major}")),
    }
}

pub(crate) fn rebuild_requirement(req: &Requirement, new_spec: &str) -> String {
    let extras = if req.extras.is_empty() {
        String::new()
    } else {
        let names: Vec<String> = req.extras.iter().map(|e| e.to_string()).collect();
        format!("[{}]", names.join(","))
    };
    let marker_str = req.marker.try_to_string().unwrap_or_default();
    let marker_part = if marker_str.is_empty() {
        String::new()
    } else {
        format!(" ; {marker_str}")
    };
    format!("{}{}{}{}", req.name, extras, new_spec, marker_part)
}

/// Replace the RECORD line for `entry_name` with a new hash and size.
/// RECORD format: `<path>,sha256=<urlsafe-b64-nopad>,<size>` per PEP
/// 376. The line for RECORD itself has empty hash/size by convention;
/// leave those alone.
fn update_record_line(
    record: &str,
    entry_name: &str,
    new_sha: &str,
    new_size: usize,
) -> Result<String> {
    let mut out = String::with_capacity(record.len());
    let mut found = false;
    for line in record.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            out.push_str(line);
            continue;
        }
        let mut fields = trimmed.splitn(3, ',');
        let path = fields.next().unwrap_or("");
        if path == entry_name {
            out.push_str(&format!("{path},sha256={new_sha},{new_size}"));
            if line.ends_with("\r\n") {
                out.push_str("\r\n");
            } else {
                out.push('\n');
            }
            found = true;
        } else {
            out.push_str(line);
        }
    }
    if !found {
        return Err(anyhow!("RECORD has no entry for {entry_name}"));
    }
    Ok(out)
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let mut out = String::with_capacity(64);
    for b in hasher.finalize() {
        write!(&mut out, "{b:02x}").expect("write to String");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_exact_pin_to_minor_range() {
        let raw = "Pillow==12.0.0";
        let out = relax_pep508(raw, RelaxPolicy::Minor).unwrap();
        // uv_pep508 normalizes names to lowercase per PEP 503; pip
        // accepts either case so this is fine on the install side.
        assert_eq!(out, "pillow>=12,<13");
    }

    #[test]
    fn rewrites_exact_pin_to_patch_range() {
        let raw = "starlette==0.45.3";
        let out = relax_pep508(raw, RelaxPolicy::Patch).unwrap();
        assert_eq!(out, "starlette>=0.45.3,<0.46");
    }

    #[test]
    fn rewrites_with_major() {
        let raw = "numpy==1.26.0";
        let out = relax_pep508(raw, RelaxPolicy::Major).unwrap();
        assert_eq!(out, "numpy>=1");
    }

    #[test]
    fn tiered_cascade_rewrites_exact_pin_to_patch_range() {
        // The tiered cascade mirrors Patch at wheel rewrite time; the
        // conda-side escalation is independent.
        let raw = "numpy==1.26.4";
        let out = relax_pep508(raw, RelaxPolicy::PatchThenMinorThenMajorThenLastResort).unwrap();
        assert_eq!(out, "numpy>=1.26.4,<1.27");
    }

    #[test]
    fn passes_through_ranges_unchanged() {
        let raw = "requests>=2.0,<3";
        let out = relax_pep508(raw, RelaxPolicy::Minor).unwrap();
        assert_eq!(out, "requests>=2.0,<3");
    }

    #[test]
    fn preserves_marker() {
        // pywin32==306 has a single-segment release; minor relax should
        // emit `>=306,<307`. The marker `sys_platform == "win32"` must be
        // preserved.
        let raw = r#"pywin32==306 ; sys_platform == "win32""#;
        let out = relax_pep508(raw, RelaxPolicy::Minor).unwrap();
        assert!(out.starts_with("pywin32>=306,<307"), "got: {out}");
        assert!(out.contains("sys_platform"), "got: {out}");
    }

    #[test]
    fn preserves_extras() {
        let raw = "requests[socks]==2.32.5";
        let out = relax_pep508(raw, RelaxPolicy::Minor).unwrap();
        assert_eq!(out, "requests[socks]>=2.32,<3");
    }

    #[test]
    fn rewrite_metadata_text_skips_non_requires_lines() {
        let m = "Metadata-Version: 2.1\n\
                 Name: foo\n\
                 Version: 1.0.0\n\
                 Requires-Dist: numpy==1.26.0\n\
                 Requires-Dist: scipy==1.15.3\n\
                 \n\
                 Body line that mentions Requires-Dist: should not change\n";
        let out = rewrite_metadata_text_with(
            m,
            &|line| match relax_pep508(line, RelaxPolicy::Minor).ok() {
                None => LineAction::Keep,
                Some(s) if s == line => LineAction::Keep,
                Some(s) => LineAction::Replace(s),
            },
        )
        .unwrap();
        assert!(out.contains("numpy>=1.26,<2"));
        assert!(out.contains("scipy>=1.15,<2"));
        assert!(out.contains("Body line that mentions Requires-Dist: should not change"));
        assert!(out.contains("Metadata-Version: 2.1"));
        assert!(out.contains("Name: foo"));
    }

    #[test]
    fn strong_major_strips_upper_bounds_in_wheel_metadata() {
        // pyglet<2 in a wheel's METADATA gets its upper bound
        // dropped under strong-major, so the post-D wheel doesn't
        // re-introduce the cap to uv when installed.
        let out = relax_pep508("pyglet<2", RelaxPolicy::StrongMajor).unwrap();
        assert_eq!(out, "pyglet", "got: {out}");

        let out = relax_pep508("numpy>=1.26,<2", RelaxPolicy::StrongMajor).unwrap();
        assert_eq!(out, "numpy>=1.26", "got: {out}");

        let out = relax_pep508("requests~=2.0", RelaxPolicy::StrongMajor).unwrap();
        assert_eq!(out, "requests>=2.0", "got: {out}");

        let out = relax_pep508("requests~=2.0.4", RelaxPolicy::StrongMajor).unwrap();
        assert_eq!(out, "requests>=2.0.4", "got: {out}");

        // Exact pins still widen.
        let out = relax_pep508("numpy==1.26.4", RelaxPolicy::StrongMajor).unwrap();
        assert_eq!(out, "numpy>=1");

        // Major still leaves ranges alone.
        let out = relax_pep508("pyglet<2", RelaxPolicy::Major).unwrap();
        assert_eq!(out, "pyglet<2");
    }

    #[test]
    fn strong_major_preserves_extras_and_markers_when_stripping() {
        // The rebuilt requirement must preserve `pkg[extra1,extra2]`
        // and `; marker` segments. Without this, requirements like
        // `torch[extras]>=2.0,<3` would lose their extras when D
        // strips the upper bound.
        let out = relax_pep508("torch[cuda,extras]>=2.0,<3", RelaxPolicy::StrongMajor).unwrap();
        assert!(out.contains("torch[cuda,extras]"), "got: {out}");
        assert!(out.contains(">=2.0"));
        assert!(!out.contains("<3"));

        let out = relax_pep508(
            r#"pyglet<2 ; python_version >= "3.10""#,
            RelaxPolicy::StrongMajor,
        )
        .unwrap();
        // uv_pep508 normalizes `python_version` to `python_full_version`
        // -- either form proves the marker survived the strip.
        assert!(
            out.contains("python_version") || out.contains("python_full_version"),
            "marker dropped; got: {out}",
        );
        // `pyglet` with no version, marker preserved.
        assert!(out.starts_with("pyglet"));
        assert!(!out.contains("<2"));
    }

    /// Build a minimal in-memory wheel zip for test use.
    fn make_test_wheel_bytes(dist: &str, version: &str, requires: &[&str]) -> Vec<u8> {
        use std::io::Write as _;
        let normalized = dist.replace('-', "_");
        let di = format!("{normalized}-{version}.dist-info");
        let mut metadata = format!("Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n");
        for req in requires {
            metadata.push_str(&format!("Requires-Dist: {req}\n"));
        }
        let metadata_bytes = metadata.into_bytes();
        let wheel_file = b"Wheel-Version: 1.0\nTag: py3-none-any\n".to_vec();
        let record = format!("{di}/METADATA,,\n{di}/WHEEL,,\n{di}/RECORD,,\n").into_bytes();
        let mut buf = Vec::new();
        let mut zip = ZipWriter::new(std::io::Cursor::new(&mut buf));
        let opts = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Stored)
            .last_modified_time(zip::DateTime::default());
        for (name, body) in [
            (format!("{di}/METADATA"), metadata_bytes.as_slice()),
            (format!("{di}/WHEEL"), wheel_file.as_slice()),
            (format!("{di}/RECORD"), record.as_slice()),
        ] {
            zip.start_file(&name, opts).unwrap();
            zip.write_all(body).unwrap();
        }
        zip.finish().unwrap();
        buf
    }

    /// Test 6 (Phase 2.8): `LineAction::Drop` omits exactly the targeted line.
    ///
    /// Three `Requires-Dist` lines; the mapper drops the middle one.
    /// The output METADATA must contain the first and third lines byte-identically
    /// and must omit the second entirely (not replace it with empty, not leave a
    /// blank line).
    #[test]
    fn line_action_drop_omits_line() {
        let metadata = "Metadata-Version: 2.1\n\
                        Name: foo\n\
                        Version: 1.0.0\n\
                        Requires-Dist: aaa==1.0\n\
                        Requires-Dist: robomimic @ git+https://github.com/example/robomimic.git@v0.4.0\n\
                        Requires-Dist: zzz>=2\n\
                        \n\
                        Body text.\n";

        // Drop the URL line (middle), keep the others.
        let out = rewrite_metadata_text_with(metadata, &|line| {
            if line.starts_with("robomimic @") {
                LineAction::Drop
            } else {
                LineAction::Keep
            }
        })
        .unwrap();

        // The dropped line must not appear at all.
        assert!(
            !out.contains("robomimic"),
            "dropped line still present: {out:?}"
        );
        // The kept lines must appear byte-identically.
        assert!(
            out.contains("Requires-Dist: aaa==1.0\n"),
            "first line missing: {out:?}"
        );
        assert!(
            out.contains("Requires-Dist: zzz>=2\n"),
            "third line missing: {out:?}"
        );
        // Headers and body must survive.
        assert!(out.contains("Metadata-Version: 2.1"));
        assert!(out.contains("Body text."));
        // Exactly two Requires-Dist lines remain.
        assert_eq!(
            out.matches("Requires-Dist: ").count(),
            2,
            "unexpected Requires-Dist count: {out:?}"
        );
    }

    /// Test 7 (Amendment 3 / Phase 2.8): `LineAction` refactor is byte- AND
    /// signal-identical to the old `Option<String>`-based path.
    ///
    /// Drive an UNCHANGED wheel (no override, no drop) through the refactored
    /// `rewrite_wheel_with` and assert:
    ///   - `did_change` is `false` (the ShadowSrc signal is unaffected).
    ///   - The output sha256 equals the input sha256 (byte-identical).
    ///
    /// Then drive a CHANGED wheel (one line gets `Replace`) and confirm
    /// `did_change` flips to `true`.
    #[test]
    fn line_action_refactor_unchanged_wheel_parity() {
        let pid = std::process::id();
        let tmp = std::env::temp_dir().join(format!("retread-wheel-rewrite-parity-{pid}"));
        std::fs::create_dir_all(&tmp).unwrap();
        let src = tmp.join("src.whl");
        let dst_unchanged = tmp.join("dst_unchanged.whl");
        let dst_changed = tmp.join("dst_changed.whl");

        let wheel_bytes = make_test_wheel_bytes("mylib", "1.0.0", &["requests>=2,<3"]);
        std::fs::write(&src, &wheel_bytes).unwrap();
        let src_sha = sha256_hex(&wheel_bytes);

        // Mapper returns Keep for every line → no change expected.
        let (sha_same, did_change_same) =
            rewrite_wheel_with(&src, &dst_unchanged, &|_| LineAction::Keep).unwrap();
        assert!(
            !did_change_same,
            "did_change must be false when mapper returns only Keep"
        );
        assert_eq!(
            sha_same, src_sha,
            "sha256 must be identical for an all-Keep mapper"
        );

        // Mapper returns Replace for the one Requires-Dist line → change expected.
        let (_, did_change_rep) = rewrite_wheel_with(&src, &dst_changed, &|line| {
            if line == "requests>=2,<3" {
                LineAction::Replace("requests>=2".to_string())
            } else {
                LineAction::Keep
            }
        })
        .unwrap();
        assert!(
            did_change_rep,
            "did_change must be true when a line is replaced"
        );

        // Cleanup (best-effort, non-fatal if it fails).
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn update_record_line_updates_target() {
        let record = "foo/__init__.py,sha256=AAAA,42\n\
                      foo-1.0.0.dist-info/METADATA,sha256=OLD,100\n\
                      foo-1.0.0.dist-info/RECORD,,\n";
        let out =
            update_record_line(record, "foo-1.0.0.dist-info/METADATA", "NEWHASH", 200).unwrap();
        assert!(out.contains("foo-1.0.0.dist-info/METADATA,sha256=NEWHASH,200"));
        assert!(out.contains("foo/__init__.py,sha256=AAAA,42"));
        assert!(out.contains("foo-1.0.0.dist-info/RECORD,,"));
    }
}
