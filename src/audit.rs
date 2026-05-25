//! Audit record of what retread saw and what it emitted, per bundle.
//!
//! Serialized as `retread-audit.json` next to the generated
//! `recipe.yaml`. Two purposes:
//!
//! 1. Debug: when the conda solve fails on a transitive pin you didn't
//!    write, grep the audit for the offending package name and you'll
//!    see exactly which bundled wheel pulled it in, what the original
//!    `Requires-Dist:` line said, and what retread emitted to conda.
//!
//! 2. Manual mirroring: the `pixi_toml_blocks` field renders ready-to-
//!    paste TOML for the workspace's `[dependencies]` and
//!    `[feature.X.pypi-options.dependency-overrides]` tables. Useful
//!    when you want to force the workspace's PyPI side (uv) to match
//!    retread's bundled versions exactly -- the same pattern your
//!    `[feature.isaaclab.pypi-options.dependency-overrides]` block
//!    uses today, but auto-derived from the bundle instead of
//!    hand-maintained.
//!
//! The audit is purely informational; nothing else in retread reads
//! it back. Removing the file or skipping its write has no effect on
//! the build.

use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// v0.12.0+: per-wheel record of the auto-data-files inject phase
/// (phase 1.6). Surfaces which wheel carried the upstream checkout-
/// root tree as `.data/data/lib/<rel>` entries (lands at
/// `$CONDA_PREFIX/lib/<rel>` after install), and which sibling wheels
/// were dedup'd out because they share the same checkout. Purely
/// informational -- grep this to find the .kit files / repo-root data
/// without unpacking the .conda.
#[derive(Debug, Clone, Serialize)]
pub struct AutoDataReport {
    /// Repo on disk that was walked (parent of the entry's subdirectory
    /// for git/named-git entries; absent for path/url/spec entries).
    pub checkout_root: PathBuf,
    /// Number of files emitted as `<dist>-<ver>.data/data/lib/<rel>`.
    pub file_count: usize,
    /// Subdirectories of the checkout that were skipped because
    /// sibling entries in this bundle already shipped them as wheels.
    pub skip_subdirs: Vec<PathBuf>,
}

/// One wheel's pre-D view: the literal `Requires-Dist:` lines as they
/// appear in upstream's wheel METADATA (before any D rewriting or
/// translate).
#[derive(Debug, Serialize)]
pub struct WheelAudit {
    pub name: String,
    pub version: String,
    pub requires_dist: Vec<String>,
    /// v0.12.0+: extras the user requested on this entry (any form --
    /// PyPI, git, path, named-git). Empty when none requested.
    /// Surfaces what got fed to the extras BFS so debugging "did
    /// extras=[all] actually expand to the right sub-wheels?" is one
    /// grep away. `#[serde(default)]` keeps old audit JSON parseable.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extras: Vec<String>,
    /// v0.12.0+: present when this wheel carried the auto-data-files
    /// inject for its checkout root. `None` for: PyPI/URL/path entries
    /// (no checkout root), and for git/named-git entries 2..N within a
    /// bundle that share a checkout with an earlier entry (dedup'd).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_data: Option<AutoDataReport>,
    /// v0.12.0+: present when this wheel was the 2..N entry from a
    /// checkout root already served by an earlier wheel. Records the
    /// shared checkout so debugging "why didn't this wheel carry the
    /// apps/?" is obvious.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_data_dedup_skipped_root: Option<PathBuf>,
}

/// One emitted conda run-dep: the spec retread sends to pixi, plus the
/// PyPI name it came from (recovered by reversing conda_name_from) so a
/// reader can correlate emit ↔ upstream pin.
#[derive(Debug, Serialize)]
pub struct EmittedDep {
    pub name: String,
    pub spec: String,
}

/// TOML fragments the user can copy-paste into their workspace to
/// mirror the bundle exactly. Each is a string ready to drop under
/// the corresponding header.
#[derive(Debug, Serialize)]
pub struct PixiTomlBlocks {
    /// Body of a `[dependencies]` or `[feature.X.dependencies]` table.
    /// One line per emitted conda run-dep:
    ///   `numpy = ">=1"`
    ///   `pyglet = "*"`
    /// Drop into the workspace to constrain its conda solve to the
    /// same versions retread chose.
    pub dependencies: String,

    /// Body of a `[pypi-options.dependency-overrides]` or
    /// `[feature.X.pypi-options.dependency-overrides]` table. One
    /// line per wheel in the bundle, pinned to that wheel's exact
    /// version:
    ///   `isaacsim = "==5.1.0.0"`
    ///   `pytorch3d = "==0.7.8+5043d15pt2.7.0cu128"`
    /// Drop into the workspace to force uv (PyPI side) to pick the
    /// same versions retread bundled. This is the block whose
    /// strict-pinning fights with conda picks -- mirroring it
    /// exactly is the surest fix.
    pub pypi_options_dependency_overrides: String,
}

/// All info about one conda output retread produced.
#[derive(Debug, Serialize)]
pub struct BundleAudit {
    pub conda_name: String,
    pub version: String,
    pub wheels: Vec<WheelAudit>,
    pub emitted_run_deps: Vec<EmittedDep>,
    pub pixi_toml_blocks: PixiTomlBlocks,
}

/// Top-level audit: keyed by the (conda_name, python_version) pair
/// since the same source package may emit different outputs per
/// python variant.
#[derive(Debug, Serialize, Default)]
pub struct Audit {
    pub bundles: BTreeMap<String, BundleAudit>,
}

/// Build the `[pypi-options.dependency-overrides]` TOML body. The
/// PyPI override format is `pkg = "constraint"`; PyPI names keep
/// their original casing/separators per PEP 503, but pixi normalizes
/// to lowercase-dash on read so emitting lowercase-dash is safe and
/// matches what the user's hand-maintained overrides look like.
fn render_pypi_overrides(wheels: &[WheelAudit]) -> String {
    let mut lines = Vec::with_capacity(wheels.len());
    for w in wheels {
        let pypi_name = w.name.to_ascii_lowercase().replace('_', "-");
        // Quote-escape the version (PEP 440 local-version `+` is legal
        // in TOML bare strings but be safe).
        let escaped = w.version.replace('"', "\\\"");
        lines.push(format!("{pypi_name} = \"=={escaped}\""));
    }
    lines.join("\n")
}

/// Build the `[dependencies]` TOML body from the conda run-deps
/// retread emitted. Each line is `<conda_name> = "<spec>"` where
/// `<spec>` is the conda match-spec emitted (e.g. ">=1", "<2",
/// "==5.1.0.0", "*"). The leading `<name> ` prefix from the emitted
/// spec is stripped if present so the right-hand side is JUST the
/// version constraint.
fn render_conda_deps(emitted: &[EmittedDep]) -> String {
    let mut lines = Vec::with_capacity(emitted.len());
    for d in emitted {
        // Spec may be "<name> <version-constraint>" or just "<name>".
        // Strip the leading name to get the constraint part.
        let constraint = d
            .spec
            .strip_prefix(&d.name)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .unwrap_or("*");
        let escaped = constraint.replace('"', "\\\"");
        lines.push(format!("{} = \"{}\"", d.name, escaped));
    }
    lines.join("\n")
}

impl BundleAudit {
    /// Construct a BundleAudit including the rendered TOML blocks.
    pub fn new(
        conda_name: String,
        version: String,
        wheels: Vec<WheelAudit>,
        emitted_run_deps: Vec<EmittedDep>,
    ) -> Self {
        let pypi_options_dependency_overrides = render_pypi_overrides(&wheels);
        let dependencies = render_conda_deps(&emitted_run_deps);
        Self {
            conda_name,
            version,
            wheels,
            emitted_run_deps,
            pixi_toml_blocks: PixiTomlBlocks {
                dependencies,
                pypi_options_dependency_overrides,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wheel(name: &str, version: &str, requires: &[&str]) -> WheelAudit {
        WheelAudit {
            name: name.into(),
            version: version.into(),
            requires_dist: requires.iter().map(|s| (*s).to_string()).collect(),
            extras: vec![],
            auto_data: None,
            auto_data_dedup_skipped_root: None,
        }
    }

    #[test]
    fn pypi_overrides_block_pins_every_wheel_to_exact_version() {
        // The pypi-options block is the one whose strictness matters
        // most -- it's what the user pastes into the workspace to
        // force uv (PyPI side) to match retread's bundled versions
        // exactly. Each line must be valid pixi.toml syntax:
        //     `pkg = "==X.Y.Z"`
        let wheels = vec![
            wheel("isaacsim", "5.1.0.0", &[]),
            wheel("isaaclab", "0.51.1", &[]),
            wheel("pytorch3d", "0.7.8+5043d15pt2.7.0cu128", &[]),
            // Underscore in PyPI name -> dash in pixi.toml override
            // (PEP 503 normalization, pixi reads either).
            wheel("typing_extensions", "4.12.2", &[]),
        ];
        let block = render_pypi_overrides(&wheels);

        // Each wheel produces one line, ==<exact-version>.
        assert!(block.contains("isaacsim = \"==5.1.0.0\""), "block:\n{block}");
        assert!(block.contains("isaaclab = \"==0.51.1\""));
        // PEP 440 local-version identifier must survive verbatim --
        // this is the case (pytorch3d) that fails on the PyPI side
        // when the version doesn't exactly match.
        assert!(
            block.contains("pytorch3d = \"==0.7.8+5043d15pt2.7.0cu128\""),
            "+local identifier dropped; block:\n{block}",
        );
        // Underscore normalized to dash.
        assert!(block.contains("typing-extensions = \"==4.12.2\""));

        // Each line must match the TOML key-equals-quoted-value shape
        // (no special chars in the constraint side that would need
        // escaping for our supported version syntax).
        for line in block.lines() {
            assert!(
                line.contains(" = \"") && line.ends_with('"'),
                "line must be a valid `key = \"value\"` pair; got: {line}",
            );
        }
    }

    #[test]
    fn conda_deps_block_renders_spec_constraints() {
        let emitted = vec![
            EmittedDep { name: "numpy".into(), spec: "numpy >=1".into() },
            EmittedDep { name: "pyglet".into(), spec: "pyglet".into() }, // no constraint
            EmittedDep { name: "pytorch".into(), spec: "pytorch >=2.7,<3".into() },
        ];
        let block = render_conda_deps(&emitted);
        assert!(block.contains("numpy = \">=1\""), "block:\n{block}");
        // `pyglet` (no constraint) renders as `pyglet = "*"`.
        assert!(block.contains("pyglet = \"*\""));
        assert!(block.contains("pytorch = \">=2.7,<3\""));

        for line in block.lines() {
            assert!(
                line.contains(" = \"") && line.ends_with('"'),
                "line must be a valid `key = \"value\"` pair; got: {line}",
            );
        }
    }

    #[test]
    fn bundle_audit_serializes_as_well_formed_json() {
        let audit = BundleAudit::new(
            "isaac-pack".into(),
            "5.1.0.0".into(),
            vec![wheel("isaacsim", "5.1.0.0", &["numpy==1.26.0", "pyglet<2"])],
            vec![
                EmittedDep { name: "numpy".into(), spec: "numpy >=1".into() },
                EmittedDep { name: "pyglet".into(), spec: "pyglet".into() },
            ],
        );
        let json = serde_json::to_string_pretty(&audit).unwrap();
        // Spot-check shape rather than exact bytes (field order is
        // stable but values may evolve).
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["conda_name"], "isaac-pack");
        assert_eq!(parsed["wheels"][0]["requires_dist"][1], "pyglet<2");
        assert!(
            parsed["pixi_toml_blocks"]["pypi_options_dependency_overrides"]
                .as_str()
                .unwrap()
                .contains("isaacsim = \"==5.1.0.0\""),
            "json:\n{json}",
        );
    }
}
