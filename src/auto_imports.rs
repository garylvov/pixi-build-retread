//! P4: join what the code imports to what the closure provides.
//!
//! `import_scan` answers *what does this source tree import*.
//! `wheel_modules` answers *which distribution provides that module*.
//! This joins them into REQUIREMENTS and records where each came from.
//!
//! It deliberately does NOT decide whether a requirement is satisfiable.
//! Retread already has exactly one resolution path -- closure -> auto-route
//! loop -> `prefer_conda_match` (parselmouth + FALLBACK + retread-name-map)
//! -> `validated_conda_route` probe -> conda run-dep or PyPI. An
//! auto-detected dependency goes through THAT, exactly like a declared one,
//! so conda routing, probing and relaxation all apply unchanged.
//!
//! An earlier version of this module built a private wheel-only index and
//! adjudicated satisfaction itself. It reported `numpy` -- 186 importing
//! files -- as MISSING, because conda-provided packages were invisible to it
//! by construction. That was a second, weaker resolver alongside the real
//! one. Emitting requirements instead deletes the whole failure mode.
//!
//! Provenance is the point. A dep that appears with no explanation is exactly
//! the hand-transcribed dep wall this work exists to replace.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::import_scan::{scan_imports_non_stdlib, ImportHit};
use crate::wheel_modules::{wheel_modules, WheelModules};

/// Why a module is believed to be provided by a given distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceSource {
    /// The providing wheel named it in `*.dist-info/top_level.txt`.
    TopLevelTxt,
    /// Derived from the providing wheel's RECORD paths.
    Record,
}

/// One import turned into a requirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedImport {
    pub module: String,
    /// Distribution NAME to require, e.g. `opencv-python` for `cv2`. This is
    /// a requirement to hand to the solver, NOT an assertion that it is
    /// satisfiable -- the solver decides that.
    pub provider: Option<String>,
    pub source: Option<ProvenanceSource>,
    /// True when EVERY import site sat inside a try/except.
    pub conditional: bool,
    /// Files that import it. Retained so a refusal can name them.
    pub files: Vec<PathBuf>,
}

/// Extras inferable from submodule imports, e.g. `etils.epath` -> `epath`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtraHint {
    pub distribution: String,
    pub module: String,
    pub submodule: String,
    /// The extra this WOULD name. A suggestion, never a decision: many
    /// distributions do not name extras after submodules (`requests[security]`
    /// has no `requests.security`), so this must be intersected with the
    /// dist's real Provides-Extra list before it is acted on.
    pub candidate_extra: String,
}

/// The full join.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AutoImportReport {
    /// Imports that map to a distribution name. These become REQUIREMENTS,
    /// injected alongside the entry's declared deps and resolved by the same
    /// path. Presence here says nothing about satisfiability.
    pub requirements: Vec<ResolvedImport>,
    /// Imports that map to NO distribution name at all. This is a NAMING
    /// gap, which is what `retread-name-map` exists to fix -- it is NOT a
    /// claim that the package is absent. Conda-provided modules do not
    /// appear here: they map to a name and are routed to conda downstream.
    pub unmapped: Vec<ResolvedImport>,
    /// Submodule-derived extras candidates.
    pub extra_hints: Vec<ExtraHint>,
    /// Wheels that failed to parse, by path and reason. NEVER silently
    /// dropped: a wheel we could not read is not a wheel that provides
    /// nothing, and conflating the two would produce false MISSING rows.
    pub unreadable: Vec<(PathBuf, String)>,
}


/// Does this entry have a local source tree to scan, and where is it?
///
/// Auto-import scanning is a PER-ENTRY property. The predicate is retread's
/// own source-built test, `wheel_entry_metadata_provenance`
/// (`handler/mod.rs:3772`): `path || git || from`. Reusing it rather than
/// inventing a second notion of "has source" keeps the two from drifting.
///
/// * `path=` -- the tree is the path, absolute or joined to `source_dir`,
///   matching `handler/mod.rs:13613-13617`.
/// * `git=` / `from=` -- the tree is a checkout the caller has already
///   materialised (`source_build::git_checkout_root`), so the caller passes
///   it; this function will not guess a checkout location.
/// * `version=` / `url=` -- NO tree. Returns None. That is not an error: an
///   index wheel has no source for us to read, and its dependencies are
///   already declared in its own metadata.
///
/// BUNDLES NEED NO SPECIAL RULE. Within a group, entries resolve
/// individually -- `handler/mod.rs:5239` drives per-entry futures through
/// `buffered(N)` and merges `sub_bundles` afterwards -- so each entry
/// contributes requirements to its own resolve. A `bundle` grouping entries
/// of mixed kinds needs no arbitration: the ones with trees are scanned, the
/// ones without contribute nothing.
pub fn entry_source_tree(
    has_path: Option<&str>,
    is_git_backed: bool,
    source_dir: &Path,
    materialised_checkout: Option<&Path>,
) -> Option<PathBuf> {
    if let Some(path) = has_path {
        let p = Path::new(path);
        return Some(if p.is_absolute() { p.to_path_buf() } else { source_dir.join(p) });
    }
    if is_git_backed {
        // Only what the caller actually materialised. Never guessed.
        return materialised_checkout.map(Path::to_path_buf);
    }
    None
}

/// PEP 503 normalisation: the requirement form of a name.
fn normalize_dist_name(module: &str) -> String {
    module.replace('_', "-").to_lowercase()
}

/// Index of module name -> providing wheel, built from the closure.
struct ClosureIndex {
    by_module: BTreeMap<String, (String, ProvenanceSource)>,
    submodules: BTreeMap<String, BTreeSet<String>>,
    /// Declared extras per top-level module, from the providing wheel.
    extras: BTreeMap<String, BTreeSet<String>>,
    unreadable: Vec<(PathBuf, String)>,
}

fn build_index(wheels: &[PathBuf]) -> ClosureIndex {
    let mut by_module = BTreeMap::new();
    let mut submodules: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut extras: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut unreadable = Vec::new();
    for w in wheels {
        let WheelModules { modules, submodules: subs, from_top_level_txt, provides_extra } =
            match wheel_modules(w) {
                Ok(m) => m,
                Err(e) => {
                    unreadable.push((w.clone(), format!("{e:#}")));
                    continue;
                }
            };
        // The DISTRIBUTION name, not the wheel filename: a requirement is
        // `opencv-python`, not `opencv_python-4.10.0.84-py3-none-any.whl`.
        // Wheel filenames are `{name}-{version}-...`, and PEP 503 normalises
        // the underscore form back to hyphens.
        let file = w.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let name = file
            .split('-')
            .next()
            .filter(|n| !n.is_empty())
            .map(|n| n.replace('_', "-"))
            .unwrap_or_else(|| w.display().to_string());
        let source = if from_top_level_txt {
            ProvenanceSource::TopLevelTxt
        } else {
            ProvenanceSource::Record
        };
        for m in &modules {
            if !provides_extra.is_empty() {
                extras.entry(m.clone()).or_default().extend(provides_extra.iter().cloned());
            }
        }
        for m in modules {
            // First writer wins; a later duplicate does not silently displace
            // the earlier provider.
            by_module.entry(m).or_insert_with(|| (name.clone(), source));
        }
        for s in subs {
            if let Some((top, sub)) = s.split_once('.') {
                submodules
                    .entry(top.to_string())
                    .or_default()
                    .insert(sub.to_string());
            }
        }
    }
    ClosureIndex { by_module, submodules, extras, unreadable }
}

/// Scan `root`, resolve every non-stdlib import against `wheels`.
///
/// `own_top_level` are the modules this package itself provides; they are
/// excluded by the scanner, not reported as missing.
pub fn resolve_imports(
    root: &Path,
    own_top_level: &BTreeSet<String>,
    python_version: &str,
    wheels: &[PathBuf],
) -> AutoImportReport {
    let index = build_index(wheels);
    let mut report = AutoImportReport { unreadable: index.unreadable, ..Default::default() };

    for ImportHit { module, conditional, files } in
        scan_imports_non_stdlib(root, own_top_level, python_version)
    {
        match index.by_module.get(&module) {
            Some((provider, source)) => {
                let declared = index.extras.get(&module);
                if let Some(subs) = index.submodules.get(&module) {
                    for sub in subs {
                        // A submodule is only an extras candidate if the dist
                        // DECLARES that extra. Without this every submodule
                        // qualifies: 917 candidates on a real closure.
                        if !declared.is_some_and(|d| d.contains(sub)) {
                            continue;
                        }
                        report.extra_hints.push(ExtraHint {
                            distribution: provider.clone(),
                            module: module.clone(),
                            submodule: format!("{module}.{sub}"),
                            candidate_extra: sub.clone(),
                        });
                    }
                }
                report.requirements.push(ResolvedImport {
                    module,
                    provider: Some(provider.clone()),
                    source: Some(*source),
                    conditional,
                    files,
                });
            }
            // No closure wheel ships this module. That is NOT a verdict of
            // absence: conda-provided packages (numpy, imageio, jax) are not
            // in the wheel closure at all, and the module name is itself a
            // perfectly good distribution name for them. Emit it as a
            // requirement and let the auto-route loop map it to conda and the
            // solver decide. Only a module we cannot even NAME stays unmapped,
            // and after this fallback that set is empty by construction --
            // which is correct: naming a requirement is not the same as
            // claiming it resolves.
            None => report.requirements.push(ResolvedImport {
                module: module.clone(),
                provider: Some(normalize_dist_name(&module)),
                source: None,
                conditional,
                files,
            }),
        }
    }
    report
}

impl AutoImportReport {
    /// Human-readable text for imports that map to no distribution NAME.
    /// This is a naming gap, not an absent package -- the wording must not
    /// claim the dependency is missing from the environment.
    pub fn explain_unmapped(&self) -> String {
        let mut out = String::new();
        for m in &self.unmapped {
            out.push_str(&format!(
                "`{}` could not be mapped to a distribution name{}\n",
                m.module,
                if m.conditional { " (every import site is inside a try/except)" } else { "" }
            ));
            for f in m.files.iter().take(5) {
                out.push_str(&format!("    imported by {}\n", f.display()));
            }
            if m.files.len() > 5 {
                out.push_str(&format!("    ... and {} more\n", m.files.len() - 5));
            }
            out.push_str(
                "    it may be first-party, or need a retread-name-map entry\n",
            );
        }
        if !self.unreadable.is_empty() {
            out.push_str(&format!(
                "\n{} wheel(s) could not be read; a name above may come from one:\n",
                self.unreadable.len()
            ));
            for (p, why) in self.unreadable.iter().take(5) {
                out.push_str(&format!("    {}: {why}\n", p.display()));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(label: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "retread-autoimp-{label}-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn wheel(dir: &Path, name: &str, entries: &[(&str, &str)]) -> PathBuf {
        let path = dir.join(name);
        let f = std::fs::File::create(&path).unwrap();
        let mut z = zip::ZipWriter::new(f);
        let o: zip::write::FileOptions<()> =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for (n, b) in entries {
            z.start_file(*n, o).unwrap();
            z.write_all(b.as_bytes()).unwrap();
        }
        z.finish().unwrap();
        path
    }

    fn py(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    /// The headline join: an `import cv2` resolves to the opencv-python wheel
    /// because that wheel says it ships cv2. No alias table involved.
    #[test]
    fn cv2_resolves_to_the_opencv_wheel_with_provenance() {
        let root = tmp("cv2");
        let wd = tmp("cv2-wheels");
        py(&root, "app.py", "import cv2\nimport os\n");
        let w = wheel(
            &wd,
            "opencv_python-4.10.0.84-py3-none-any.whl",
            &[
                ("opencv_python-4.10.0.84.dist-info/top_level.txt", "cv2\n"),
                ("opencv_python-4.10.0.84.dist-info/RECORD", "cv2/__init__.py,,\n"),
            ],
        );
        let r = resolve_imports(&root, &BTreeSet::new(), "3.11", &[w]);
        assert!(r.unmapped.is_empty(), "missing: {:?}", r.unmapped);
        assert_eq!(r.requirements.len(), 1, "os is stdlib and must not appear");
        let hit = &r.requirements[0];
        assert_eq!(hit.module, "cv2");
        assert_eq!(
            hit.provider.as_deref(),
            Some("opencv-python"),
            "a requirement is a DISTRIBUTION NAME, not a wheel filename"
        );
        assert_eq!(hit.source, Some(ProvenanceSource::TopLevelTxt));
        assert!(!hit.files.is_empty(), "the importing file must be recorded");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(wd);
    }

    /// An import no closure wheel ships must still become a REQUIREMENT,
    /// named after the module. conda-provided packages (numpy, imageio, jax)
    /// are not in the wheel closure at all; the earlier design reported them
    /// MISSING, which meant numpy -- 186 importing files -- read as an absent
    /// dependency. Naming a requirement is not claiming it resolves; the
    /// auto-route loop and the solver decide that.
    #[test]
    fn import_with_no_closure_wheel_still_becomes_a_requirement() {
        let root = tmp("unknown");
        let wd = tmp("unknown-wheels");
        py(&root, "pkg/thing.py", "import numpy\nimport some_thing\n");
        let w = wheel(
            &wd,
            "other-1.0-py3-none-any.whl",
            &[("other-1.0.dist-info/top_level.txt", "other\n")],
        );
        let r = resolve_imports(&root, &BTreeSet::new(), "3.11", &[w]);
        assert!(
            r.unmapped.is_empty(),
            "every import is nameable; nothing should be unmapped: {:?}",
            r.unmapped
        );
        let names: Vec<&str> =
            r.requirements.iter().filter_map(|x| x.provider.as_deref()).collect();
        assert!(names.contains(&"numpy"), "numpy must be a requirement: {names:?}");
        assert!(
            names.contains(&"some-thing"),
            "underscores normalise to hyphens per PEP 503: {names:?}"
        );
        // No wheel supplied the answer, so there is no wheel-derived provenance.
        let np = r.requirements.iter().find(|x| x.module == "numpy").unwrap();
        assert_eq!(np.source, None, "provenance is None when no wheel named it");
        assert!(!np.files.is_empty(), "the importing file is still recorded");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(wd);
    }

    /// An unreadable wheel must NOT silently become a MISSING row -- that
    /// would blame the source for a defect in the closure.
    #[test]
    fn unreadable_wheel_is_reported_not_silently_treated_as_empty() {
        let root = tmp("badwheel");
        let wd = tmp("badwheel-wheels");
        py(&root, "a.py", "import something\n");
        let bad = wd.join("broken-1.0-py3-none-any.whl");
        std::fs::write(&bad, b"this is not a zip").unwrap();
        let r = resolve_imports(&root, &BTreeSet::new(), "3.11", &[bad]);
        assert_eq!(r.unreadable.len(), 1, "the bad wheel must be reported");
        let msg = r.explain_unmapped();
        assert!(
            msg.contains("could not be read"),
            "the refusal must disclose the unreadable wheel: {msg}"
        );
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(wd);
    }

    /// Submodule imports surface an extras CANDIDATE, and the type name says
    /// candidate because many dists do not name extras after submodules.
    #[test]
    fn submodule_yields_an_extra_candidate() {
        let root = tmp("extras");
        let wd = tmp("extras-wheels");
        py(&root, "m.py", "from etils import epath\n");
        let w = wheel(
            &wd,
            "etils-1.13.0-py3-none-any.whl",
            &[
                (
                    "etils-1.13.0.dist-info/RECORD",
                    "etils/__init__.py,,\netils/epath/__init__.py,,\netils/internal/x.py,,\n",
                ),
                (
                    "etils-1.13.0.dist-info/METADATA",
                    "Name: etils\nProvides-Extra: epath\nProvides-Extra: enp\n\nbody\n",
                ),
            ],
        );
        let r = resolve_imports(&root, &BTreeSet::new(), "3.11", &[w]);
        assert_eq!(r.requirements.len(), 1);
        assert_eq!(r.requirements[0].source, Some(ProvenanceSource::Record), "etils ships no top_level.txt");
        assert!(
            r.extra_hints.iter().any(|h| h.candidate_extra == "epath"),
            "got {:?}",
            r.extra_hints
        );
        // `internal` is a real submodule but NOT a declared extra. Emitting it
        // would be the PIL[BmpImagePlugin] failure: on a real closure the
        // unfiltered heuristic produced 917 candidates, 899 of them noise.
        assert!(
            !r.extra_hints.iter().any(|h| h.candidate_extra == "internal"),
            "undeclared submodules must not become extras candidates: {:?}",
            r.extra_hints
        );
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(wd);
    }

    /// Own modules are the package's own code, not dependencies.
    #[test]
    fn own_top_level_is_never_reported_missing() {
        let root = tmp("own");
        py(&root, "mypkg/a.py", "import mypkg\nimport mypkg.helper\n");
        let own: BTreeSet<String> = ["mypkg".to_string()].into_iter().collect();
        let r = resolve_imports(&root, &own, "3.11", &[]);
        assert!(r.unmapped.is_empty(), "own code must not read as a dep: {:?}", r.unmapped);
        let _ = std::fs::remove_dir_all(root);
    }
    /// Diagnostic against a REAL source tree and a REAL closure. `#[ignore]`
    /// because it depends on the local checkout and wheel store; run with
    /// `cargo test --lib real_tree -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn real_tree_against_real_closure() {
        let root = std::path::Path::new("/oscar/data/stellex/glvov/imprint-data/src/imprint");
        let index = std::path::Path::new("/users/glvov/.claude/jobs/78d7a173/tmp/wheel_index.txt");
        let (Ok(list), true) = (std::fs::read_to_string(index), root.is_dir()) else {
            eprintln!("real tree or wheel index unavailable; skipping");
            return;
        };
        let wheels: Vec<PathBuf> = list.lines().map(PathBuf::from).collect();
        let own: BTreeSet<String> = ["imprint".to_string()].into_iter().collect();
        let t0 = std::time::Instant::now();
        let r = resolve_imports(root, &own, "3.11", &wheels);
        eprintln!(
            "\nwheels={}  elapsed={:?}\nrequirements={}  unmapped={}  extra_hints={}  unreadable={}",
            wheels.len(),
            t0.elapsed(),
            r.requirements.len(),
            r.unmapped.len(),
            r.extra_hints.len(),
            r.unreadable.len()
        );
        let via_record = r
            .requirements
            .iter()
            .filter(|x| x.source == Some(ProvenanceSource::Record))
            .count();
        eprintln!("resolved via RECORD fallback: {via_record}");
        eprintln!("\n--- first 25 UNMAPPED (naming gaps, NOT absent packages) ---");
        for m in r.unmapped.iter().take(25) {
            eprintln!(
                "  {:<28} conditional={:<5} files={}",
                m.module,
                m.conditional,
                m.files.len()
            );
        }
        eprintln!("\n--- sample extras candidates ---");
        for h in r.extra_hints.iter().take(8) {
            eprintln!("  {} -> {}[{}]", h.submodule, h.module, h.candidate_extra);
        }
    }

    /// The entry-kind gate. An index wheel (`version=`/`url=`) has no source
    /// tree; returning None is correct, not an error -- its dependencies are
    /// declared in its own metadata and it is not ours to scan.
    #[test]
    fn only_source_bearing_entries_have_a_tree() {
        let src = Path::new("/ws/pack");
        // path=, relative -> joined to source_dir (handler/mod.rs:13617)
        assert_eq!(
            entry_source_tree(Some("./sub/pkg"), false, src, None),
            Some(PathBuf::from("/ws/pack/./sub/pkg"))
        );
        // path=, absolute -> used as-is
        assert_eq!(
            entry_source_tree(Some("/abs/pkg"), false, src, None),
            Some(PathBuf::from("/abs/pkg"))
        );
        // git= with a materialised checkout -> that checkout
        assert_eq!(
            entry_source_tree(None, true, src, Some(Path::new("/cache/co"))),
            Some(PathBuf::from("/cache/co"))
        );
        // git= WITHOUT one -> None. Never guess a checkout location.
        assert_eq!(entry_source_tree(None, true, src, None), None);
        // version= / url= -> no tree at all
        assert_eq!(entry_source_tree(None, false, src, None), None);
    }

}
