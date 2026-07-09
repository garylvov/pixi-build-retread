//! Fix #20: auto-repaired T1 (conda-as-truth) overrides whose conflict
//! originated in a **backend uv-closure** (a pack's
//! `[package.build.config.retread-wheels]` resolution, computed deep inside
//! the JSON-RPC build backend) are written into that failing PACK's own
//! `[package.build.config.retread-overrides]` table -- the exact table the
//! manual torch trio already lives in, and the ONLY override table the
//! pack's uv closure actually consumes.
//!
//! Why not the workspace manifest: `retread solve`/`retread lock`'s
//! original T1 path wrote `pypi-options.dependency-overrides` in the
//! WORKSPACE pixi.toml, which is provably inert for this class -- the
//! pack's closure never reads it (no `dependency-overrides` handling
//! anywhere in `uv_closure.rs`/`handler`/`config.rs`), so the override was
//! applied and silently ignored (override20 brief ACCEPTANCE RUN #6b:
//! iter-2 conflict byte-identical to iter-1). The pack table arrives in
//! `handler`'s `effective.overrides` with no special casing and is folded
//! into the uv `override-dependencies`, so it actually takes effect; it
//! also participates in `courier::config_fingerprint`, so writing one
//! busts cold-solve replay automatically.
//!
//! Run-scoped rollback: a repair run that later EXHAUSTS must not leave a
//! half-written pack override behind (defect #3 parity for the workspace
//! manifest). Each touched pack pixi.toml is snapshotted once per run under
//! the workspace's `.retread/pack-overrides-bak/`, restored on
//! exhaustion/interrupt/crash, and dropped on convergence.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use toml_edit::{DocumentMut, Item, Table, Value};

/// Sentinel appended to every auto-written pack override so it is
/// distinguishable from a user's manual entry (and could later be swept).
pub const OVERRIDE_SENTINEL: &str = "# retread:override";

/// Write (or replace) `package = "spec"  # retread:override <date>` into
/// `pack_pixi`'s `[package.build.config.retread-overrides]` table,
/// creating the nested tables if absent. `spec` is the full conda-style
/// spec (e.g. `"==2.10.0"`); the value is written verbatim so it matches
/// the manual-trio format the pack already uses.
pub fn write_override(pack_pixi: &Path, package: &str, spec: &str) -> Result<()> {
    let text = std::fs::read_to_string(pack_pixi)
        .with_context(|| format!("failed to read pack manifest {}", pack_pixi.display()))?;
    let mut doc = text
        .parse::<DocumentMut>()
        .with_context(|| format!("failed to parse pack manifest {}", pack_pixi.display()))?;

    let package_tbl = ensure_table(doc.as_table_mut(), "package");
    let build_tbl = ensure_table(package_tbl, "build");
    let config_tbl = ensure_table(build_tbl, "config");
    let overrides_tbl = ensure_table(config_tbl, "retread-overrides");

    let mut value = Value::from(spec.to_string());
    value
        .decor_mut()
        .set_suffix(format!("  {OVERRIDE_SENTINEL} {}", local_date()));
    overrides_tbl[package] = Item::Value(value);

    write_atomic(pack_pixi, doc.to_string().as_bytes())
}

/// Navigate to (creating if needed) a child table, marking any created
/// table implicit so a fresh `[package.build.config.retread-overrides]`
/// serializes as a single dotted header rather than a chain of empty
/// `[package]` / `[package.build]` stanzas.
fn ensure_table<'a>(parent: &'a mut Table, key: &str) -> &'a mut Table {
    if !parent.contains_key(key) {
        let mut t = Table::new();
        t.set_implicit(true);
        parent.insert(key, Item::Table(t));
    }
    parent
        .get_mut(key)
        .and_then(Item::as_table_mut)
        .expect("ensured table exists")
}

fn snapshot_dir(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(".retread").join("pack-overrides-bak")
}

/// Stable per-path basename for a pack pixi.toml's snapshot.
fn snapshot_stem(pack_pixi: &Path) -> String {
    format!("{:x}", Sha256::digest(pack_pixi.to_string_lossy().as_bytes()))
}

/// Snapshot `pack_pixi` under the workspace's `.retread/` before the run's
/// first write to it. No-op if a snapshot already exists (first snapshot
/// of the run wins). Records the origin path alongside so rollback needs
/// no external index.
pub fn ensure_snapshot(workspace_dir: &Path, pack_pixi: &Path) -> Result<()> {
    let dir = snapshot_dir(workspace_dir);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create {}", dir.display()))?;
    let stem = snapshot_stem(pack_pixi);
    let bak = dir.join(format!("{stem}.toml"));
    if bak.exists() {
        return Ok(());
    }
    std::fs::copy(pack_pixi, &bak)
        .with_context(|| format!("failed to snapshot {}", pack_pixi.display()))?;
    std::fs::write(
        dir.join(format!("{stem}.path")),
        pack_pixi.to_string_lossy().as_bytes(),
    )
    .with_context(|| "failed to write pack snapshot path record".to_string())?;
    Ok(())
}

/// Restore every pack pixi.toml snapshotted this run, then drop the whole
/// snapshot dir. Used on any non-converged exit (exhaustion / interrupt /
/// crash / per-env `--keep-going` restore).
pub fn rollback_all(workspace_dir: &Path) -> Result<()> {
    let dir = snapshot_dir(workspace_dir);
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("path") {
            continue;
        }
        let origin = std::fs::read_to_string(&path)?;
        let origin = Path::new(origin.trim());
        let bak = path.with_extension("toml");
        if bak.exists() {
            std::fs::copy(&bak, origin)
                .with_context(|| format!("failed to restore {}", origin.display()))?;
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

/// Drop the run's pack snapshots without restoring -- called once the run
/// converges (the overrides are now part of the green lock and stay).
pub fn cleanup_all(workspace_dir: &Path) -> Result<()> {
    let dir = snapshot_dir(workspace_dir);
    if dir.is_dir() {
        std::fs::remove_dir_all(&dir)
            .with_context(|| format!("failed to remove {}", dir.display()))?;
    }
    Ok(())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("toml.retread-tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("failed to write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("failed to rename into {}", path.display()))?;
    Ok(())
}

fn local_date() -> String {
    std::process::Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "1970-01-01".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "retread-pack-overrides-test-{tag}-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn writes_override_into_pack_table_with_sentinel() {
        let tmp = TempDir::new("write");
        let pack = tmp.path().join("pixi.toml");
        std::fs::write(
            &pack,
            "[package]\nname = \"isaac-pack-latest\"\nversion = \"6.0.0\"\n\n\
             [package.build.config.retread-overrides]\ntinyobjloader = \"==2.0.0rc13\"\n",
        )
        .unwrap();
        write_override(&pack, "torch", "==2.10.0").unwrap();
        let text = std::fs::read_to_string(&pack).unwrap();
        assert!(text.contains("torch = \"==2.10.0\"  # retread:override"));
        // Existing manual entry preserved.
        assert!(text.contains("tinyobjloader = \"==2.0.0rc13\""));
        // Reparses as valid TOML with the override in the right table.
        let doc = text.parse::<DocumentMut>().unwrap();
        let spec = doc["package"]["build"]["config"]["retread-overrides"]["torch"]
            .as_str()
            .unwrap();
        assert_eq!(spec, "==2.10.0");
    }

    #[test]
    fn creates_overrides_table_when_absent() {
        let tmp = TempDir::new("create");
        let pack = tmp.path().join("pixi.toml");
        std::fs::write(&pack, "[package]\nname = \"p\"\nversion = \"1.0\"\n").unwrap();
        write_override(&pack, "torch", "==2.10.0").unwrap();
        let text = std::fs::read_to_string(&pack).unwrap();
        let doc = text.parse::<DocumentMut>().unwrap();
        assert_eq!(
            doc["package"]["build"]["config"]["retread-overrides"]["torch"]
                .as_str()
                .unwrap(),
            "==2.10.0"
        );
    }

    #[test]
    fn snapshot_rollback_restores_pack_manifest() {
        let tmp = TempDir::new("rollback");
        let ws = tmp.path();
        let pack_dir = ws.join("pypi-packs/p");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack = pack_dir.join("pixi.toml");
        let original = "[package]\nname = \"p\"\nversion = \"1.0\"\n";
        std::fs::write(&pack, original).unwrap();

        ensure_snapshot(ws, &pack).unwrap();
        write_override(&pack, "torch", "==2.10.0").unwrap();
        assert!(std::fs::read_to_string(&pack).unwrap().contains("torch"));

        rollback_all(ws).unwrap();
        assert_eq!(std::fs::read_to_string(&pack).unwrap(), original);
        assert!(!snapshot_dir(ws).exists());
    }

    #[test]
    fn cleanup_keeps_the_written_override() {
        let tmp = TempDir::new("cleanup");
        let ws = tmp.path();
        let pack = ws.join("pixi.toml");
        std::fs::write(&pack, "[package]\nname = \"p\"\nversion = \"1.0\"\n").unwrap();
        ensure_snapshot(ws, &pack).unwrap();
        write_override(&pack, "torch", "==2.10.0").unwrap();
        cleanup_all(ws).unwrap();
        assert!(!snapshot_dir(ws).exists());
        assert!(std::fs::read_to_string(&pack).unwrap().contains("torch"));
    }
}
