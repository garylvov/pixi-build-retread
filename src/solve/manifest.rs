use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use toml_edit::{DocumentMut, Item, Table, Value};

use super::repair::{Reason, Strategy};

pub const PIN_SENTINEL: &str = "# retread:pin";
pub const WIDEN_SENTINEL: &str = "# retread:widen";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EntrySnapshot {
    pub value: Option<String>,
    pub suffix: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedEdit {
    pub package: String,
    pub strategy: Strategy,
    pub table: TableKind,
    pub before: EntrySnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TableKind {
    Conda,
    Pypi,
    Override,
}

pub struct ManifestEditor {
    path: PathBuf,
    doc: DocumentMut,
}

impl ManifestEditor {
    pub fn open(path: PathBuf) -> Result<Self> {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let doc = text
            .parse::<DocumentMut>()
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(Self { path, doc })
    }

    pub fn project_dir(&self) -> &Path {
        self.path.parent().unwrap_or_else(|| Path::new("."))
    }

    pub fn reload(&mut self) -> Result<()> {
        let text = std::fs::read_to_string(&self.path)?;
        self.doc = text.parse::<DocumentMut>()?;
        Ok(())
    }

    pub fn write_atomic(&self) -> Result<()> {
        write_atomic(&self.path, self.doc.to_string().as_bytes())
    }

    pub fn smoke_modules(&self) -> Vec<String> {
        let Some(table) = self
            .doc
            .as_table()
            .get("tool")
            .and_then(|i| i.get("retread"))
            .and_then(|i| i.get("smoke-test"))
            .and_then(Item::as_table)
        else {
            return Vec::new();
        };
        table
            .get("modules")
            .and_then(Item::as_value)
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn entry_snapshot(&self, feature: &str, kind: TableKind, package: &str) -> EntrySnapshot {
        let Some(item) = self.get_table(feature, kind).and_then(|t| t.get(package)) else {
            return EntrySnapshot {
                value: None,
                suffix: None,
            };
        };
        EntrySnapshot {
            value: item
                .as_value()
                .and_then(Value::as_str)
                .map(ToString::to_string),
            suffix: item_suffix(item),
        }
    }

    pub fn restore_entry(
        &mut self,
        feature: &str,
        kind: TableKind,
        package: &str,
        snapshot: &EntrySnapshot,
    ) {
        match &snapshot.value {
            Some(value) => {
                let suffix = snapshot.suffix.as_deref().unwrap_or("");
                self.set_string(feature, kind, package, value, suffix);
            }
            None => {
                if let Some(table) = self.get_table_mut_existing(feature, kind) {
                    table.remove(package);
                }
                self.drop_empty_path(feature, kind);
            }
        }
    }

    pub fn set_conda_pin(
        &mut self,
        feature: &str,
        package: &str,
        version: &str,
        reason: Reason,
    ) -> AppliedEdit {
        let before = self.entry_snapshot(feature, TableKind::Conda, package);
        self.set_string(
            feature,
            TableKind::Conda,
            package,
            &format!("=={version}"),
            &pin_suffix(reason),
        );
        AppliedEdit {
            package: package.to_string(),
            strategy: Strategy::Conda,
            table: TableKind::Conda,
            before,
        }
    }

    pub fn set_pypi_pin(
        &mut self,
        feature: &str,
        package: &str,
        version: &str,
        reason: Reason,
    ) -> AppliedEdit {
        let before = self.entry_snapshot(feature, TableKind::Pypi, package);
        self.set_string(
            feature,
            TableKind::Pypi,
            package,
            &format!("=={version}"),
            &pin_suffix(reason),
        );
        AppliedEdit {
            package: package.to_string(),
            strategy: Strategy::PypiDep,
            table: TableKind::Pypi,
            before,
        }
    }

    pub fn set_pypi_override(
        &mut self,
        feature: &str,
        package: &str,
        version: &str,
        reason: Reason,
    ) -> AppliedEdit {
        let before = self.entry_snapshot(feature, TableKind::Override, package);
        self.set_string(
            feature,
            TableKind::Override,
            package,
            &format!("=={version}"),
            &pin_suffix(reason),
        );
        AppliedEdit {
            package: package.to_string(),
            strategy: Strategy::PypiOverride,
            table: TableKind::Override,
            before,
        }
    }

    pub fn set_conda_widen(&mut self, feature: &str, package: &str, spec: &str) -> AppliedEdit {
        let before = self.entry_snapshot(feature, TableKind::Conda, package);
        self.set_string(feature, TableKind::Conda, package, spec, &widen_suffix());
        AppliedEdit {
            package: package.to_string(),
            strategy: Strategy::WidenConda,
            table: TableKind::Conda,
            before,
        }
    }

    pub fn remove_retread_entry(
        &mut self,
        feature: &str,
        kind: TableKind,
        package: &str,
    ) -> Option<AppliedEdit> {
        let snapshot = self.entry_snapshot(feature, kind, package);
        if snapshot.value.is_none() || !snapshot_has_retread_sentinel(&snapshot) {
            return None;
        }
        let table = self.get_table_mut_existing(feature, kind)?;
        table.remove(package);
        self.drop_empty_path(feature, kind);
        Some(AppliedEdit {
            package: package.to_string(),
            strategy: match kind {
                TableKind::Conda => Strategy::Conda,
                TableKind::Pypi => Strategy::PypiDep,
                TableKind::Override => Strategy::PypiOverride,
            },
            table: kind,
            before: snapshot,
        })
    }

    pub fn has_user_entry(&self, feature: &str, kind: TableKind, package: &str) -> bool {
        let snap = self.entry_snapshot(feature, kind, package);
        snap.value.is_some() && !snapshot_has_retread_sentinel(&snap)
    }

    pub fn sentinel_strategies(&self) -> Vec<(String, Strategy)> {
        let mut out = Vec::new();
        for (feature, kind, package, suffix) in self.sentinel_entries() {
            let strategy = if suffix.contains(WIDEN_SENTINEL) {
                Strategy::WidenConda
            } else {
                match kind {
                    TableKind::Conda => Strategy::Conda,
                    TableKind::Pypi => Strategy::PypiDep,
                    TableKind::Override => Strategy::PypiOverride,
                }
            };
            let _ = feature;
            out.push((package, strategy));
        }
        out
    }

    pub fn clean_pins(&mut self) -> usize {
        let entries: Vec<(String, TableKind, String, String)> = self.sentinel_entries();
        let mut removed = 0;
        for (feature, kind, package, suffix) in entries {
            if suffix.contains(PIN_SENTINEL) || suffix.contains(WIDEN_SENTINEL) {
                if let Some(table) = self.get_table_mut_existing(&feature, kind)
                    && table.remove(&package).is_some()
                {
                    removed += 1;
                }
                self.drop_empty_path(&feature, kind);
            }
        }
        removed
    }

    fn sentinel_entries(&self) -> Vec<(String, TableKind, String, String)> {
        let mut out = Vec::new();
        for kind in [TableKind::Conda, TableKind::Pypi, TableKind::Override] {
            if let Some(table) = self.get_table("default", kind) {
                collect_sentinel_entries(&mut out, "default", kind, table);
            }
        }
        if let Some(features) = self.doc.as_table().get("feature").and_then(Item::as_table) {
            for (feature_name, feature_item) in features.iter() {
                for kind in [TableKind::Conda, TableKind::Pypi, TableKind::Override] {
                    let table = match kind {
                        TableKind::Conda => feature_item.get("dependencies"),
                        TableKind::Pypi => feature_item.get("pypi-dependencies"),
                        TableKind::Override => feature_item
                            .get("pypi-options")
                            .and_then(|i| i.get("dependency-overrides")),
                    }
                    .and_then(Item::as_table);
                    if let Some(table) = table {
                        collect_sentinel_entries(&mut out, feature_name, kind, table);
                    }
                }
            }
        }
        out
    }

    fn get_table(&self, feature: &str, kind: TableKind) -> Option<&Table> {
        if feature == "default" {
            return match kind {
                TableKind::Conda => self.doc.as_table().get("dependencies"),
                TableKind::Pypi => self.doc.as_table().get("pypi-dependencies"),
                TableKind::Override => self
                    .doc
                    .as_table()
                    .get("pypi-options")
                    .and_then(|i| i.get("dependency-overrides")),
            }
            .and_then(Item::as_table);
        }
        let feature_item = self.doc.as_table().get("feature")?.get(feature)?;
        match kind {
            TableKind::Conda => feature_item.get("dependencies"),
            TableKind::Pypi => feature_item.get("pypi-dependencies"),
            TableKind::Override => feature_item
                .get("pypi-options")
                .and_then(|i| i.get("dependency-overrides")),
        }
        .and_then(Item::as_table)
    }

    fn get_table_mut_existing(&mut self, feature: &str, kind: TableKind) -> Option<&mut Table> {
        if feature == "default" {
            return match kind {
                TableKind::Conda => self.doc.as_table_mut().get_mut("dependencies"),
                TableKind::Pypi => self.doc.as_table_mut().get_mut("pypi-dependencies"),
                TableKind::Override => self
                    .doc
                    .as_table_mut()
                    .get_mut("pypi-options")
                    .and_then(|i| i.get_mut("dependency-overrides")),
            }
            .and_then(Item::as_table_mut);
        }
        let feature_item = self
            .doc
            .as_table_mut()
            .get_mut("feature")?
            .get_mut(feature)?;
        match kind {
            TableKind::Conda => feature_item.get_mut("dependencies"),
            TableKind::Pypi => feature_item.get_mut("pypi-dependencies"),
            TableKind::Override => feature_item
                .get_mut("pypi-options")
                .and_then(|i| i.get_mut("dependency-overrides")),
        }
        .and_then(Item::as_table_mut)
    }

    fn get_table_mut_create(&mut self, feature: &str, kind: TableKind) -> &mut Table {
        if feature == "default" {
            return match kind {
                TableKind::Conda => ensure_table(self.doc.as_table_mut(), "dependencies"),
                TableKind::Pypi => ensure_table(self.doc.as_table_mut(), "pypi-dependencies"),
                TableKind::Override => {
                    let pypi_options = ensure_table(self.doc.as_table_mut(), "pypi-options");
                    ensure_table(pypi_options, "dependency-overrides")
                }
            };
        }
        let feature_table = ensure_table(self.doc.as_table_mut(), "feature");
        let named_feature = ensure_table(feature_table, feature);
        match kind {
            TableKind::Conda => ensure_table(named_feature, "dependencies"),
            TableKind::Pypi => ensure_table(named_feature, "pypi-dependencies"),
            TableKind::Override => {
                let pypi_options = ensure_table(named_feature, "pypi-options");
                ensure_table(pypi_options, "dependency-overrides")
            }
        }
    }

    fn set_string(
        &mut self,
        feature: &str,
        kind: TableKind,
        package: &str,
        value: &str,
        suffix: &str,
    ) {
        let table = self.get_table_mut_create(feature, kind);
        let mut value = Value::from(value.to_string());
        value.decor_mut().set_suffix(suffix);
        table[package] = Item::Value(value);
    }

    fn drop_empty_path(&mut self, feature: &str, kind: TableKind) {
        let empty = self
            .get_table(feature, kind)
            .map(Table::is_empty)
            .unwrap_or(false);
        if !empty {
            return;
        }
        if feature == "default" {
            match kind {
                TableKind::Conda => {
                    self.doc.as_table_mut().remove("dependencies");
                }
                TableKind::Pypi => {
                    self.doc.as_table_mut().remove("pypi-dependencies");
                }
                TableKind::Override => {
                    if let Some(options) = self
                        .doc
                        .as_table_mut()
                        .get_mut("pypi-options")
                        .and_then(Item::as_table_mut)
                    {
                        options.remove("dependency-overrides");
                    }
                }
            }
            return;
        }
        if let Some(feature_item) = self
            .doc
            .as_table_mut()
            .get_mut("feature")
            .and_then(|i| i.get_mut(feature))
        {
            match kind {
                TableKind::Conda => {
                    if let Some(table) = feature_item.as_table_mut() {
                        table.remove("dependencies");
                    }
                }
                TableKind::Pypi => {
                    if let Some(table) = feature_item.as_table_mut() {
                        table.remove("pypi-dependencies");
                    }
                }
                TableKind::Override => {
                    if let Some(options) = feature_item
                        .get_mut("pypi-options")
                        .and_then(Item::as_table_mut)
                    {
                        options.remove("dependency-overrides");
                    }
                }
            }
        }
    }
}

pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(|s| s.to_str())
            .map(|s| format!("{s}."))
            .unwrap_or_default()
    ));
    std::fs::write(&tmp, bytes).with_context(|| format!("failed to write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("failed to rename {} over {}", tmp.display(), path.display()))?;
    Ok(())
}

pub fn copy_atomic(src: &Path, dst: &Path) -> Result<()> {
    let bytes = std::fs::read(src).with_context(|| format!("failed to read {}", src.display()))?;
    write_atomic(dst, &bytes)
}

pub fn restore_bytes_atomic(dst: &Path, bytes: &[u8]) -> Result<()> {
    write_atomic(dst, bytes)
}

fn ensure_table<'a>(table: &'a mut Table, key: &str) -> &'a mut Table {
    table
        .entry(key)
        .or_insert(Item::Table(Table::new()))
        .as_table_mut()
        .expect("inserted table")
}

fn item_suffix(item: &Item) -> Option<String> {
    item.as_value()
        .and_then(|v| v.decor().suffix())
        .and_then(|raw| raw.as_str())
        .map(ToString::to_string)
}

fn snapshot_has_retread_sentinel(snapshot: &EntrySnapshot) -> bool {
    snapshot
        .suffix
        .as_deref()
        .map(|s| s.contains(PIN_SENTINEL) || s.contains(WIDEN_SENTINEL))
        .unwrap_or(false)
}

fn collect_sentinel_entries(
    out: &mut Vec<(String, TableKind, String, String)>,
    feature: &str,
    kind: TableKind,
    table: &Table,
) {
    for (package, item) in table.iter() {
        if let Some(suffix) = item_suffix(item)
            && (suffix.contains(PIN_SENTINEL) || suffix.contains(WIDEN_SENTINEL))
        {
            out.push((feature.to_string(), kind, package.to_string(), suffix));
        }
    }
}

fn pin_suffix(reason: Reason) -> String {
    format!("  # retread:pin {} {}", local_date(), reason.slug())
}

fn widen_suffix() -> String {
    format!("  # retread:widen {}", local_date())
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
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temp_manifest(text: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "retread-solve-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pixi.toml");
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn adds_feature_scoped_sentinel_pins_and_migrates_owned_entries_only() {
        let path = temp_manifest(
            r#"[workspace]
channels = ["conda-forge"]

[feature.gpu.dependencies]
numpy = "==1.26.4"
"#,
        );
        let mut editor = ManifestEditor::open(path.clone()).unwrap();
        assert!(editor.has_user_entry("gpu", TableKind::Conda, "numpy"));
        editor.set_conda_pin("gpu", "torch", "2.7.1", Reason::CondaBoundary);
        editor.remove_retread_entry("gpu", TableKind::Conda, "numpy");
        editor.set_pypi_pin("gpu", "torch", "2.7.1", Reason::NoCandidatesEscalation);
        editor.write_atomic().unwrap();
        let out = std::fs::read_to_string(path).unwrap();
        assert!(out.contains("numpy = \"==1.26.4\""));
        assert!(out.contains("torch = \"==2.7.1\"  # retread:pin"));
        assert!(out.contains("[feature.gpu.pypi-dependencies]"));
    }

    #[test]
    fn clean_pins_removes_only_retread_sentinels() {
        let path = temp_manifest(
            r#"[dependencies]
numpy = "==1.26.4"  # user-owned
mujoco = ">=3.10.3,<4"  # retread:widen 2026-07-07

[pypi-dependencies]
torch = "==2.7.1"  # retread:pin 2026-07-07 conda-boundary
"#,
        );
        let mut editor = ManifestEditor::open(path.clone()).unwrap();
        assert_eq!(editor.clean_pins(), 2);
        editor.write_atomic().unwrap();
        let out = std::fs::read_to_string(path).unwrap();
        assert!(out.contains("numpy = \"==1.26.4\""));
        assert!(!out.contains("mujoco"));
        assert!(!out.contains("torch"));
    }

    #[test]
    fn restore_entry_reverts_single_widen() {
        let path = temp_manifest(
            r#"[dependencies]
mujoco = "==3.5.0"
"#,
        );
        let mut editor = ManifestEditor::open(path.clone()).unwrap();
        let edit = editor.set_conda_widen("default", "mujoco", ">=3.10.3,<4");
        editor.restore_entry("default", edit.table, "mujoco", &edit.before);
        editor.write_atomic().unwrap();
        let out = std::fs::read_to_string(path).unwrap();
        assert!(out.contains("mujoco = \"==3.5.0\""));
        assert!(!out.contains("retread:widen"));
    }
}
