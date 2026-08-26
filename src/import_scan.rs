//! Native Python import scanning.
//!
//! This module deliberately reports raw module names. In particular, it does
//! not decide whether a module is from the standard library or which
//! distribution provides it; those are policy questions for later phases.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use ignore::WalkBuilder;
use ruff_python_ast::{
    visitor::{self, Visitor},
    Expr, Stmt,
};
use ruff_python_parser::parse_module;

/// A Python top-level module imported by one or more source files.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportHit {
    pub module: String,
    pub conditional: bool,
    pub files: Vec<PathBuf>,
}

#[derive(Default)]
struct AccumulatedHit {
    conditional: bool,
    files: BTreeSet<PathBuf>,
}

/// Scan Python source below `root` and return the imported top-level modules.
///
/// Invalid Python files and unreadable paths are skipped. Relative imports and
/// absolute imports whose top-level name is in `own_top_level` are excluded.
pub fn scan_imports(root: &Path, own_top_level: &BTreeSet<String>) -> Vec<ImportHit> {
    let mut hits = BTreeMap::new();

    for entry in WalkBuilder::new(root)
        .standard_filters(true)
        .build()
        .flatten()
    {
        let path = entry.path();
        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
            || path.extension().is_none_or(|extension| extension != "py")
        {
            continue;
        }

        let Ok(source) = std::fs::read_to_string(path) else {
            continue;
        };
        let file = path.strip_prefix(root).unwrap_or(path).to_path_buf();
        scan_source(&source, file, own_top_level, &mut hits);
    }

    hits.into_iter()
        .map(|(module, hit)| ImportHit {
            module,
            conditional: hit.conditional,
            files: hit.files.into_iter().collect(),
        })
        .collect()
}

fn scan_source(
    source: &str,
    file: PathBuf,
    own_top_level: &BTreeSet<String>,
    hits: &mut BTreeMap<String, AccumulatedHit>,
) {
    let Ok(module) = parse_module(source) else {
        return;
    };

    let mut visitor = ImportVisitor {
        own_top_level,
        hits,
        file,
        conditional_depth: 0,
    };
    visitor.visit_body(&module.syntax().body);
}

struct ImportVisitor<'a> {
    own_top_level: &'a BTreeSet<String>,
    hits: &'a mut BTreeMap<String, AccumulatedHit>,
    file: PathBuf,
    conditional_depth: usize,
}

impl ImportVisitor<'_> {
    fn record(&mut self, module: &str) {
        let top_level = module.split('.').next().unwrap_or_default();
        if top_level.is_empty() || self.own_top_level.contains(top_level) {
            return;
        }

        let hit = self.hits.entry(top_level.to_owned()).or_default();
        hit.conditional |= self.conditional_depth > 0;
        hit.files.insert(self.file.clone());
    }
}

impl<'a> Visitor<'a> for ImportVisitor<'_> {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::Import(import) => {
                for alias in &import.names {
                    self.record(alias.name.as_str());
                }
            }
            // A non-zero level is a relative import (`from .x import y`) and
            // therefore necessarily belongs to the scanned source tree.
            Stmt::ImportFrom(import) if import.level == 0 => {
                if let Some(module) = &import.module {
                    self.record(module.as_str());
                }
            }
            Stmt::Try(_) => {
                self.conditional_depth += 1;
                visitor::walk_stmt(self, stmt);
                self.conditional_depth -= 1;
                return;
            }
            Stmt::If(if_stmt) if contains_type_checking(&if_stmt.test) => {
                self.conditional_depth += 1;
                visitor::walk_stmt(self, stmt);
                self.conditional_depth -= 1;
                return;
            }
            _ => {}
        }

        visitor::walk_stmt(self, stmt);
    }
}

fn contains_type_checking(test: &Expr) -> bool {
    struct Finder(bool);

    impl<'a> Visitor<'a> for Finder {
        fn visit_expr(&mut self, expr: &'a Expr) {
            match expr {
                Expr::Name(name) => self.0 |= name.id.as_str() == "TYPE_CHECKING",
                Expr::Attribute(attribute) => self.0 |= attribute.attr.as_str() == "TYPE_CHECKING",
                _ => visitor::walk_expr(self, expr),
            }
        }
    }

    let mut finder = Finder(false);
    finder.visit_expr(test);
    finder.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan_inline(source: &str, own_top_level: &[&str]) -> Vec<ImportHit> {
        let own_top_level = own_top_level
            .iter()
            .map(|name| (*name).to_owned())
            .collect();
        let mut hits = BTreeMap::new();
        scan_source(
            source,
            PathBuf::from("inline.py"),
            &own_top_level,
            &mut hits,
        );
        hits.into_iter()
            .map(|(module, hit)| ImportHit {
                module,
                conditional: hit.conditional,
                files: hit.files.into_iter().collect(),
            })
            .collect()
    }

    #[test]
    fn drops_relative_imports() {
        assert!(scan_inline("from . import child\nfrom ..parent import child\n", &[]).is_empty());
    }

    #[test]
    fn drops_first_party_absolute_imports() {
        assert!(scan_inline("import mypkg.internal\n", &["mypkg"]).is_empty());
    }

    #[test]
    fn flags_try_imports_as_conditional() {
        let hits = scan_inline(
            "try:\n    import optional_dep\nexcept ImportError:\n    pass\n",
            &[],
        );
        assert_eq!(hits[0].module, "optional_dep");
        assert!(hits[0].conditional);
    }

    #[test]
    fn reports_only_top_level_module_name() {
        let hits = scan_inline("import a.b.c\n", &[]);
        assert_eq!(hits[0].module, "a");
    }

    #[test]
    fn scans_the_deception_fixture() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tasks/retread-4-11/p4n-import-detect/artifacts/deception_fixture");
        let hits = scan_imports(&root, &BTreeSet::from(["mypkg".to_owned()]));

        // Standard-library filtering is deliberately outside phase 1. This
        // assertion checks the fixture's third-party subset while also making
        // explicit that the scanner returns raw module names.
        let third_party: BTreeSet<_> = hits
            .iter()
            .filter(|hit| !matches!(hit.module.as_str(), "os" | "sys" | "json" | "typing"))
            .map(|hit| hit.module.as_str())
            .collect();
        assert_eq!(
            third_party,
            BTreeSet::from(["cv2", "numpy", "pandas", "scipy", "ujson"])
        );

        let conditional: BTreeSet<_> = hits
            .iter()
            .filter(|hit| hit.conditional)
            .map(|hit| hit.module.as_str())
            .collect();
        assert_eq!(conditional, BTreeSet::from(["pandas", "ujson"]));
    }
}
