//! Tree subpackage enumeration (N27-RETREAD-113).
//!
//! # What this replaces
//!
//! Nine of the nineteen production packs each hand-typed the subpackage list of
//! ONE git tree -- IsaacLab -- as `[retread-wheels]` entries that differ only in
//! their `subdirectory`:
//!
//! ```toml
//! isaaclab        = { from = "isaaclab", subdirectory = "source/isaaclab" }
//! isaaclab-assets = { from = "isaaclab", subdirectory = "source/isaaclab_assets" }
//! isaaclab-tasks  = { from = "isaaclab", subdirectory = "source/isaaclab_tasks" }
//! # ... 3 to 12 more, per pack, re-typed
//! ```
//!
//! That is 56 lines across those nine packs, all of it an enumeration of one
//! directory level of one tree at a rev the pack ALREADY pins in
//! `[retread-git-sources]`. The pack manifest is supposed to be a pointer;
//! retread derives the rest. This module is the deriver.
//!
//! # The rule
//!
//! [`crate::config::SubpackageRule`] declares the tree once (`from`) plus an
//! enumeration rule (`glob`, e.g. `source/*`). [`enumerate`] walks that one
//! directory level of the checkout and keeps the children that are directories
//! carrying a Python build file ([`SUBPACKAGE_BUILD_FILES`]); a directory
//! without one is not a distribution and is skipped, not built. The result is
//! sorted by name, so the expansion is deterministic for a given (url, rev) --
//! the enumeration adds NO new input to resolution identity beyond the rev the
//! pack already pinned.
//!
//! # Why it refuses
//!
//! Deriving an enumeration means the set can change without anyone editing the
//! pack. That is exactly the hazard: an upstream addition (or an operator
//! bumping `rev`) would silently start building a new wheel into a production
//! pack. So the rule carries `expect`, the arity of the included set, and
//! [`apply`] refuses when the tree disagrees:
//!
//! * a name in `exclude` the tree does not contain -> refuse (stale exclusion:
//!   the subpackage was removed or renamed upstream);
//! * an included arity other than `expect` -> refuse, printing the full
//!   included set so the operator can see which name is new or gone.
//!
//! Inclusion stays a decision written into the pack. The derivation removes the
//! typing, not the judgement.
//!
//! # Where the expansion lands
//!
//! [`expand`] rewrites the rules into ordinary [`WheelEntry`] values in
//! `RetreadConfig::retread_wheels` before `resolve_all` groups entries into
//! bundles, so every downstream consumer -- the BFS, `pip wheel` in
//! `source_build`, the auto-data walk, the courier spec -- sees precisely what a
//! hand-typed pack produced and knows nothing about this module.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Result, anyhow};

use crate::config::{RetreadConfig, SubpackageRule, WheelEntry};

/// Files whose presence in a directory makes it a buildable Python project.
///
/// `pip wheel` accepts either: PEP 517/518 via `pyproject.toml`, or the legacy
/// setuptools path via `setup.py`. A directory under the glob carrying neither
/// is tree furniture (IsaacLab's `source/` has none today, but a `docs/` or a
/// `.vscode/` under a glob'd level is exactly the case this guards) and must
/// not become a wheel entry.
pub const SUBPACKAGE_BUILD_FILES: &[&str] = &["pyproject.toml", "setup.py"];

/// What one directory level of a tree turned out to hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Enumeration {
    /// Directory names carrying a build file, sorted.
    pub found: Vec<String>,
    /// Directory names the glob selected that carry NO build file, sorted.
    /// Reported, never built.
    pub skipped: Vec<String>,
}

/// The `[retread-wheels]` key a tree directory becomes.
///
/// PyPI/conda distribution names use `-`; Python package directories use `_`.
/// Every one of the 56 hand-typed entries this replaces followed exactly this
/// mapping (`source/isaaclab_assets` -> `isaaclab-assets`), which is why the
/// expansion can reproduce them byte-for-byte.
pub fn entry_key_for(dir_name: &str) -> String {
    dir_name.replace('_', "-")
}

/// Enumerate one directory level of `tree_root` under `prefix`
/// (`None` = the root itself), splitting build-able projects from the rest.
///
/// Ordering is by name so two runs over the same rev produce the same entries
/// in the same order; readdir order is not.
pub fn enumerate(tree_root: &Path, prefix: Option<&str>) -> Result<Enumeration> {
    let level = match prefix {
        Some(prefix) => tree_root.join(prefix),
        None => tree_root.to_path_buf(),
    };
    let entries = std::fs::read_dir(&level).map_err(|e| {
        anyhow!(
            "enumerating subpackages: cannot read `{}` in the checkout: {e}",
            level.display()
        )
    })?;
    let mut found = BTreeSet::new();
    let mut skipped = BTreeSet::new();
    for entry in entries {
        let entry = entry.map_err(|e| {
            anyhow!(
                "enumerating subpackages under `{}`: {e}",
                level.display()
            )
        })?;
        if !entry.file_type().is_ok_and(|t| t.is_dir()) {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            return Err(anyhow!(
                "enumerating subpackages under `{}`: directory name is not UTF-8",
                level.display()
            ));
        };
        if name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        if SUBPACKAGE_BUILD_FILES
            .iter()
            .any(|file| path.join(file).is_file())
        {
            found.insert(name);
        } else {
            skipped.insert(name);
        }
    }
    Ok(Enumeration {
        found: found.into_iter().collect(),
        skipped: skipped.into_iter().collect(),
    })
}

/// Apply a rule's `exclude` + `expect` to an [`Enumeration`], returning the
/// included directory names, or refusing.
///
/// `rev` appears only in the refusal text: the operator's next move is either
/// to edit the pack or to notice the rev moved under them, and both need it.
pub fn apply(
    rule_name: &str,
    rule: &SubpackageRule,
    rev: &str,
    enumeration: &Enumeration,
) -> Result<Vec<String>> {
    let found: BTreeSet<&str> = enumeration.found.iter().map(String::as_str).collect();
    for name in &rule.exclude {
        if !found.contains(name.as_str()) {
            return Err(anyhow!(
                "retread-subpackages `{rule_name}`: `exclude` names `{name}`, which the \
                 tree at rev {rev} does not contain (found: {}). An exclusion that \
                 matches nothing is stale -- the subpackage was removed or renamed \
                 upstream, and the pack must say what it wants now.",
                render(&enumeration.found),
            ));
        }
    }
    let excluded: BTreeSet<&str> = rule.exclude.iter().map(String::as_str).collect();
    let included: Vec<String> = enumeration
        .found
        .iter()
        .filter(|name| !excluded.contains(name.as_str()))
        .cloned()
        .collect();
    if included.len() != rule.expect {
        return Err(anyhow!(
            "retread-subpackages `{rule_name}`: `expect = {}` but the tree at rev {rev} \
             yields {} included subpackage(s): {}. The enumeration is derived, but which \
             subpackages a pack ships is a decision -- add the new name to `exclude` (to \
             keep it out) or set `expect = {}` (to build it). Excluded: {}. Skipped (no \
             {}): {}.",
            rule.expect,
            included.len(),
            render(&included),
            included.len(),
            render(&rule.exclude),
            SUBPACKAGE_BUILD_FILES.join(" or "),
            render(&enumeration.skipped),
        ));
    }
    for key in rule.extras.keys() {
        if !included.iter().any(|name| name == key) {
            return Err(anyhow!(
                "retread-subpackages `{rule_name}`: `extras` names `{key}`, which is not \
                 an included subpackage of the tree at rev {rev} (included: {})",
                render(&included),
            ));
        }
    }
    Ok(included)
}

fn render(names: &[String]) -> String {
    if names.is_empty() {
        "<none>".to_string()
    } else {
        names.join(",")
    }
}

/// The evidence row a rule prints, so a build log shows what the tree actually
/// held rather than only what got built.
pub fn subpackage_row(
    rule: &SubpackageRule,
    rev: &str,
    enumeration: &Enumeration,
    included: &[String],
) -> String {
    format!(
        "### PACK SUBPACKAGES from={} rev={} rule={} found={} names={} included={} excluded={} skipped={}",
        rule.from,
        rev,
        rule.glob,
        enumeration.found.len(),
        render(&enumeration.found),
        included.len(),
        render(&rule.exclude),
        render(&enumeration.skipped),
    )
}

/// Expand every rule in `config.subpackages` into `config.retread_wheels`.
///
/// `checkouts` maps rule name -> (rev, materialized checkout root). The caller
/// owns fetching, because this crate's checkout path is async and this function
/// is the pure, testable half: hand it a directory and it produces entries or a
/// refusal.
///
/// Rules are consumed (`config.subpackages` is left empty) so a second call is
/// a no-op and no downstream reader can see a rule it does not understand.
pub fn expand(
    config: &mut RetreadConfig,
    checkouts: &BTreeMap<String, (String, std::path::PathBuf)>,
) -> Result<Vec<String>> {
    let rules = std::mem::take(&mut config.subpackages);
    let mut rows = Vec::new();
    for (rule_name, rule) in &rules {
        rule.validate(rule_name)?;
        if !config.git_sources.contains_key(&rule.from) {
            return Err(anyhow!(
                "retread-subpackages `{rule_name}`: `from = \"{}\"` names no \
                 [retread-git-sources] entry",
                rule.from
            ));
        }
        let (rev, root) = checkouts.get(rule_name).ok_or_else(|| {
            anyhow!("retread-subpackages `{rule_name}`: no checkout was materialized")
        })?;
        let prefix = rule.glob_prefix(rule_name)?;
        let enumeration = enumerate(root, prefix)?;
        let included = apply(rule_name, rule, rev, &enumeration)?;
        rows.push(subpackage_row(rule, rev, &enumeration, &included));
        for dir_name in &included {
            let key = entry_key_for(dir_name);
            if config.retread_wheels.contains_key(&key) {
                return Err(anyhow!(
                    "retread-subpackages `{rule_name}`: the enumeration produces \
                     `{key}` (from `{dir_name}`) but [retread-wheels] already declares \
                     it. One declaration per wheel: drop the typed entry, or add \
                     `{dir_name}` to this rule's `exclude`."
                ));
            }
            let subdirectory = match prefix {
                Some(prefix) => format!("{prefix}/{dir_name}"),
                None => dir_name.clone(),
            };
            let entry = WheelEntry {
                from: Some(rule.from.clone()),
                subdirectory: Some(subdirectory),
                extras: rule.extras.get(dir_name).cloned().unwrap_or_default(),
                ..WheelEntry::default()
            };
            entry.validate(&key)?;
            config.retread_wheels.insert(key, entry);
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique temp dir per call, std-only: this crate does not declare
    /// `tempfile` as a dependency, and `handler::replay_tests` / `deps_from`
    /// already use exactly this shape. The atomic counter (not just pid+time)
    /// is what makes two calls inside one nanosecond distinct.
    fn unique_tmp_dir() -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "retread-subpackages-test-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&dir).expect("fixture tmp dir");
        dir
    }

    fn tree(root: &Path, dirs: &[(&str, Option<&str>)]) {
        for (name, build_file) in dirs {
            let dir = root.join("source").join(name);
            std::fs::create_dir_all(&dir).expect("fixture dir");
            if let Some(file) = build_file {
                std::fs::write(dir.join(file), b"# fixture\n").expect("fixture build file");
            }
        }
    }

    fn rule(expect: usize, exclude: &[&str]) -> SubpackageRule {
        SubpackageRule {
            from: "isaaclab".into(),
            glob: "source/*".into(),
            expect,
            exclude: exclude.iter().map(|s| (*s).to_string()).collect(),
            extras: BTreeMap::new(),
        }
    }

    /// N subpackage dirs plus one dir with no build file: found = N, sorted,
    /// and the build-file-less dir is reported as skipped rather than built.
    #[test]
    fn enumeration_keeps_build_file_dirs_sorted_and_skips_the_rest() {
        let tmp = unique_tmp_dir();
        tree(
            tmp.as_path(),
            &[
                ("isaaclab_tasks", Some("pyproject.toml")),
                ("isaaclab", Some("pyproject.toml")),
                ("isaaclab_assets", Some("setup.py")),
                ("docs", None),
            ],
        );
        let found = enumerate(tmp.as_path(), Some("source")).expect("enumerate");
        assert_eq!(
            found.found,
            vec!["isaaclab", "isaaclab_assets", "isaaclab_tasks"],
            "found must be exactly the build-able dirs, sorted"
        );
        assert_eq!(found.skipped, vec!["docs"]);
    }

    /// The mutation guard for the build-file requirement: if `enumerate` kept
    /// every directory, `docs` would land in `found` and this assertion is what
    /// goes red.
    #[test]
    fn a_directory_without_a_build_file_is_never_a_subpackage() {
        let tmp = unique_tmp_dir();
        tree(
            tmp.as_path(),
            &[("isaaclab", Some("pyproject.toml")), ("docs", None)],
        );
        let found = enumerate(tmp.as_path(), Some("source")).expect("enumerate");
        assert!(
            !found.found.iter().any(|n| n == "docs"),
            "a dir with neither {} is not a distribution",
            SUBPACKAGE_BUILD_FILES.join(" nor ")
        );
        assert_eq!(found.found.len(), 1);
    }

    #[test]
    fn exclude_removes_k_from_the_included_set() {
        let tmp = unique_tmp_dir();
        tree(
            tmp.as_path(),
            &[
                ("isaaclab", Some("pyproject.toml")),
                ("isaaclab_assets", Some("pyproject.toml")),
                ("isaaclab_contrib", Some("pyproject.toml")),
                ("isaaclab_tasks", Some("pyproject.toml")),
            ],
        );
        let enumeration = enumerate(tmp.as_path(), Some("source")).expect("enumerate");
        assert_eq!(enumeration.found.len(), 4);
        let included = apply("isaaclab", &rule(3, &["isaaclab_contrib"]), "cafe", &enumeration)
            .expect("exclude must apply");
        assert_eq!(
            included,
            vec!["isaaclab", "isaaclab_assets", "isaaclab_tasks"]
        );
    }

    /// THE safety property: a subpackage the tree grows that the pack neither
    /// includes (via `expect`) nor excludes refuses, and the refusal names it.
    #[test]
    fn a_new_subpackage_refuses_and_is_named() {
        let tmp = unique_tmp_dir();
        tree(
            tmp.as_path(),
            &[
                ("isaaclab", Some("pyproject.toml")),
                ("isaaclab_assets", Some("pyproject.toml")),
                ("isaaclab_ppisp", Some("pyproject.toml")),
            ],
        );
        let enumeration = enumerate(tmp.as_path(), Some("source")).expect("enumerate");
        let err = apply("isaaclab", &rule(2, &[]), "deadbeef", &enumeration)
            .expect_err("a third subpackage under expect = 2 must refuse");
        let text = format!("{err:#}");
        assert!(
            text.contains("isaaclab_ppisp"),
            "the refusal must name the subpackage that appeared: {text}"
        );
        assert!(text.contains("expect = 2"), "{text}");
        assert!(text.contains("deadbeef"), "the rev must be in the refusal: {text}");
    }

    /// The other direction: an exclusion that matches nothing is a removal or a
    /// rename upstream, and is equally loud.
    #[test]
    fn a_stale_exclusion_refuses() {
        let tmp = unique_tmp_dir();
        tree(tmp.as_path(), &[("isaaclab", Some("pyproject.toml"))]);
        let enumeration = enumerate(tmp.as_path(), Some("source")).expect("enumerate");
        let err = apply("isaaclab", &rule(1, &["isaaclab_gone"]), "beef", &enumeration)
            .expect_err("stale exclusion must refuse");
        assert!(format!("{err:#}").contains("isaaclab_gone"), "{err:#}");
    }

    #[test]
    fn glob_shape_is_checked_before_any_tree_is_touched() {
        let mut bad = rule(1, &[]);
        bad.glob = "source/*/*".into();
        assert!(bad.validate("r").is_err(), "two stars is not a level");
        bad.glob = "source".into();
        assert!(bad.validate("r").is_err(), "no star is not an enumeration");
        bad.glob = "../*".into();
        assert!(bad.validate("r").is_err(), "escaping the tree must refuse");
        bad.glob = "*".into();
        bad.validate("r").expect("bare `*` enumerates the root");
        assert_eq!(bad.glob_prefix("r").expect("prefix"), None);
    }

    /// `isaaclab-2.3x-pack` at the rev it pins (IsaacLab v2.3.2,
    /// 37ddf626871758333d6ed89cf64ad702aef127d0) has `source/` = {isaaclab,
    /// isaaclab_assets, isaaclab_contrib, isaaclab_mimic, isaaclab_rl,
    /// isaaclab_tasks} and types five entries, omitting `isaaclab_contrib`.
    /// The expansion must reproduce those five entry keys and subdirectories
    /// EXACTLY -- that equality is what makes the staged manifest a rename of
    /// the same build, not a new one.
    #[test]
    fn expansion_reproduces_the_staged_isaaclab_pack_entries_exactly() {
        let tmp = unique_tmp_dir();
        tree(
            tmp.as_path(),
            &[
                ("isaaclab", Some("pyproject.toml")),
                ("isaaclab_assets", Some("pyproject.toml")),
                ("isaaclab_contrib", Some("pyproject.toml")),
                ("isaaclab_mimic", Some("pyproject.toml")),
                ("isaaclab_rl", Some("pyproject.toml")),
                ("isaaclab_tasks", Some("pyproject.toml")),
            ],
        );
        let rev = "37ddf626871758333d6ed89cf64ad702aef127d0";
        let mut config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": {},
            "retread-git-sources": {
                "isaaclab": {
                    "url": "https://github.com/isaac-sim/IsaacLab.git",
                    "rev": rev,
                }
            },
            "retread-subpackages": {
                "isaaclab": {
                    "from": "isaaclab",
                    "glob": "source/*",
                    "expect": 5,
                    "exclude": ["isaaclab_contrib"],
                    "extras": { "isaaclab_rl": ["all"] },
                }
            },
        }))
        .expect("config parses");
        let mut checkouts = BTreeMap::new();
        checkouts.insert(
            "isaaclab".to_string(),
            (rev.to_string(), tmp.as_path().to_path_buf()),
        );
        let rows = expand(&mut config, &checkouts).expect("expansion");

        // Exactly the five `from = "isaaclab"` entries the pack types today.
        let derived: Vec<(String, String, Vec<String>)> = config
            .retread_wheels
            .iter()
            .map(|(key, entry)| {
                (
                    key.clone(),
                    entry.subdirectory.clone().unwrap_or_default(),
                    entry.extras.clone(),
                )
            })
            .collect();
        let typed: Vec<(String, String, Vec<String>)> = vec![
            ("isaaclab", "source/isaaclab", vec![]),
            ("isaaclab-assets", "source/isaaclab_assets", vec![]),
            ("isaaclab-mimic", "source/isaaclab_mimic", vec![]),
            ("isaaclab-rl", "source/isaaclab_rl", vec!["all"]),
            ("isaaclab-tasks", "source/isaaclab_tasks", vec![]),
        ]
        .into_iter()
        .map(|(k, s, e)| {
            (
                k.to_string(),
                s.to_string(),
                e.into_iter().map(str::to_string).collect(),
            )
        })
        .collect();
        assert_eq!(
            derived, typed,
            "the enumerated set must equal the pack's current typed set exactly"
        );
        for entry in config.retread_wheels.values() {
            assert_eq!(entry.from.as_deref(), Some("isaaclab"));
        }
        assert!(config.subpackages.is_empty(), "rules are consumed");
        assert_eq!(rows.len(), 1);
        assert!(
            rows[0].contains("found=6") && rows[0].contains("included=5"),
            "{}",
            rows[0]
        );
        assert!(rows[0].starts_with("### PACK SUBPACKAGES from=isaaclab"), "{}", rows[0]);
    }

    /// Reader/writer: an enumerated key colliding with a typed entry is a
    /// double declaration of one wheel, not a silent precedence rule.
    #[test]
    fn a_typed_entry_colliding_with_the_enumeration_refuses() {
        let tmp = unique_tmp_dir();
        tree(tmp.as_path(), &[("isaaclab_rl", Some("pyproject.toml"))]);
        let mut config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": {
                "isaaclab-rl": { "from": "isaaclab", "subdirectory": "source/isaaclab_rl" }
            },
            "retread-git-sources": {
                "isaaclab": { "url": "https://example.com/x.git", "rev": "abc" }
            },
            "retread-subpackages": {
                "isaaclab": { "from": "isaaclab", "glob": "source/*", "expect": 1 }
            },
        }))
        .expect("config parses");
        let mut checkouts = BTreeMap::new();
        checkouts.insert(
            "isaaclab".to_string(),
            ("abc".to_string(), tmp.as_path().to_path_buf()),
        );
        let err = expand(&mut config, &checkouts).expect_err("collision must refuse");
        assert!(format!("{err:#}").contains("isaaclab-rl"), "{err:#}");
    }

    #[test]
    fn a_rule_naming_an_unknown_git_source_refuses() {
        let tmp = unique_tmp_dir();
        tree(tmp.as_path(), &[("a", Some("setup.py"))]);
        let mut config: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": {},
            "retread-subpackages": {
                "r": { "from": "nope", "glob": "source/*", "expect": 1 }
            },
        }))
        .expect("config parses");
        let mut checkouts = BTreeMap::new();
        checkouts.insert("r".to_string(), ("abc".to_string(), tmp.as_path().to_path_buf()));
        let err = expand(&mut config, &checkouts).expect_err("unknown source must refuse");
        assert!(format!("{err:#}").contains("retread-git-sources"), "{err:#}");
    }

    #[test]
    fn extras_naming_an_excluded_subpackage_refuses() {
        let tmp = unique_tmp_dir();
        tree(
            tmp.as_path(),
            &[
                ("isaaclab", Some("pyproject.toml")),
                ("isaaclab_contrib", Some("pyproject.toml")),
            ],
        );
        let enumeration = enumerate(tmp.as_path(), Some("source")).expect("enumerate");
        let mut r = rule(1, &["isaaclab_contrib"]);
        r.extras
            .insert("isaaclab_contrib".into(), vec!["all".into()]);
        let err = apply("isaaclab", &r, "abc", &enumeration).expect_err("must refuse");
        assert!(format!("{err:#}").contains("isaaclab_contrib"), "{err:#}");
    }

    #[test]
    fn entry_keys_use_dashes() {
        assert_eq!(entry_key_for("isaaclab_tasks_experimental"), "isaaclab-tasks-experimental");
        assert_eq!(entry_key_for("isaaclab"), "isaaclab");
    }
}
