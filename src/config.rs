//! User-facing configuration for the retread backend.
//!
//! Lives under `[build.config]` in the consumer's `pixi.toml`.

use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct RetreadConfig {
    /// Wheels to repack, keyed by package name. The map key serves two
    /// purposes:
    /// 1. For [`WheelEntry::is_spec`] form, it is the PyPI distribution
    ///    name to resolve on the simple index.
    /// 2. For [`WheelEntry::is_url`] form, it is the name of the resulting
    ///    conda package.
    ///
    /// Example matching pixi's `[pypi-dependencies]` syntax:
    /// ```toml
    /// [build.config.retread-wheels]
    /// isaacsim = { version = "==5.1.0", index = "https://pypi.nvidia.com", extras = ["all", "extscache"] }
    /// mujoco   = { version = "==3.5.0", index = "https://py.mujoco.org" }
    /// foo      = { url = "https://example.com/foo-1.whl", sha256 = "..." }   # direct URL fallback
    /// ```
    #[serde(rename = "retread-wheels", alias = "wheels")]
    pub retread_wheels: BTreeMap<String, WheelEntry>,

    /// How aggressively to widen dependency pins from the wheel's METADATA.
    ///
    /// - `none`: keep pins as-is (== stays ==)
    /// - `patch`: ==X.Y.Z -> >=X.Y.Z,<X.Y+1
    /// - `minor` (default): ==X.Y.Z -> >=X.Y,<X+1
    /// - `major`: ==X.Y.Z -> >=X (drop upper bound)
    #[serde(default, rename = "retread-relax", alias = "relax")]
    pub relax: RelaxPolicy,

    /// Per-dependency overrides, applied after the relax policy. Map of
    /// PyPI name -> conda match-spec (e.g. `"*"`, `">=2.7"`).
    #[serde(default, rename = "retread-overrides", alias = "overrides")]
    pub overrides: BTreeMap<String, String>,

    /// PyPI -> conda name mapping overrides on top of the built-in identity
    /// mapping. Use for the common drift cases (`opencv-python-headless` ->
    /// `py-opencv`, etc.).
    #[serde(default, rename = "retread-name-map", alias = "name-map", alias = "name_map")]
    pub name_map: BTreeMap<String, String>,

    /// PyPI names to drop from the conda run-deps entirely. Use for
    /// upstream-pinned deps that don't exist on the target conda channel
    /// (Windows-only shims like `idna-ssl`, `pywin32`) or otherwise can't
    /// be satisfied. The wheel still gets installed; conda just won't
    /// require these at solve time.
    #[serde(default, rename = "retread-drop-deps", alias = "drop-deps", alias = "drop_deps")]
    pub drop_deps: Vec<String>,

    /// When true (default), retread tries to resolve every exact-pinned
    /// transitive `Requires-Dist` line on the entry's PyPI index (with
    /// public PyPI as fallback). If a wheel exists, it's pip-installed
    /// into the conda package and dropped from the conda run-deps.
    /// Eliminates the need to manually `retread-overrides` /
    /// `retread-drop-deps` every PyPI package that isn't on conda-forge
    /// (`aiodns`, `qdldl`, ...).
    ///
    /// Deps in `retread-conda-deps` are never auto-bundled -- they
    /// always emit as conda run-deps. Set this to `false` to disable
    /// auto-bundling entirely.
    #[serde(default = "default_true", rename = "retread-auto-bundle", alias = "auto-bundle")]
    pub auto_bundle: bool,

    /// PyPI names that must stay as conda run-deps even when
    /// `retread-auto-bundle` is on. Use for ABI-sensitive packages where
    /// installing both a bundled wheel and the conda-channel version
    /// causes a collision -- typically the scientific stack (`numpy`,
    /// `scipy`, `pytorch`, `pandas`, ...). Empty by default; retread
    /// intentionally does NOT hard-code a list.
    #[serde(default, rename = "retread-conda-deps", alias = "conda-deps")]
    pub conda_deps: Vec<String>,

    /// Conda build number for the produced packages. Bump to force
    /// re-resolution downstream after a policy change.
    #[serde(default, rename = "retread-build-number", alias = "build-number")]
    pub build_number: u64,

    /// Python version(s) to build for, as a fallback when the workspace
    /// does not declare `[workspace.build-variants] python = [...]`.
    ///
    /// Accepts either a single string ("3.11") or a list of strings
    /// (`["3.11", "3.12"]`). When the workspace's variant configuration
    /// provides `python`, that wins; this field is purely a convenience for
    /// single-Python workspaces. The default is `3.11`.
    #[serde(default)]
    pub python: Option<PythonSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum PythonSpec {
    One(String),
    Many(Vec<String>),
}

impl PythonSpec {
    pub fn as_versions(&self) -> Vec<String> {
        match self {
            Self::One(v) => vec![v.clone()],
            Self::Many(v) => v.clone(),
        }
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RelaxPolicy {
    None,
    Patch,
    #[default]
    Minor,
    Major,
}

/// Either a direct URL ({url, sha256?}) or a PyPI-style spec
/// ({version, index?, extras?}). Validated by [`WheelEntry::validate`] at
/// initialize time.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct WheelEntry {
    // ---- URL form ----
    /// Direct URL to a `.whl`. When set, all other fields are ignored except
    /// `sha256` (which becomes a verification check).
    #[serde(default)]
    pub url: Option<url::Url>,

    /// SHA-256 of the wheel. Used for verification if `url` is set, or as a
    /// pinning anchor for [`WheelEntry::is_spec`] form (rare).
    #[serde(default)]
    pub sha256: Option<String>,

    // ---- Spec form ----
    /// PEP 440 version. Accepts both `5.1.0` and `==5.1.0` (the leading
    /// `==` is stripped). Only exact pins are supported by the resolver
    /// today; range syntax (`>=5.1,<6`) is planned.
    #[serde(default)]
    pub version: Option<String>,

    /// PEP 503 simple index URL. Defaults to PyPI public.
    #[serde(default)]
    pub index: Option<String>,

    /// Extras to follow when expanding this wheel into its transitive
    /// dependency set. Each named extra adds every `Requires-Dist: name ;
    /// extra == "X"` line from the wheel's METADATA to the wheel set,
    /// resolved against the same index.
    #[serde(default)]
    pub extras: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_retread_wheels_key() {
        // Mirrors the inline-table syntax from the README and examples.
        let json = serde_json::json!({
            "retread-wheels": {
                "isaacsim": {
                    "version": "==5.1.0",
                    "index": "https://pypi.nvidia.com",
                    "extras": ["all", "extscache"],
                },
                "foo": { "url": "https://example.com/foo-1.whl", "sha256": "abc" }
            },
            "retread-relax": "minor",
            "retread-build-number": 0,
            "overrides": { "numpy": ">=1.26,<2" },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.retread_wheels.len(), 2);
        assert_eq!(cfg.relax, RelaxPolicy::Minor);
        assert!(cfg.retread_wheels.contains_key("isaacsim"));
        let isaac = &cfg.retread_wheels["isaacsim"];
        assert_eq!(isaac.normalized_version().unwrap(), "5.1.0");
        assert!(isaac.is_spec());
        assert_eq!(isaac.extras, vec!["all", "extscache"]);
    }

    #[test]
    fn legacy_unprefixed_keys_still_parse() {
        // One-release migration cushion: pre-0.4 `wheels`, `relax`, and
        // `build-number` keys without the `retread-` prefix should still
        // deserialize so users have time to update their manifests.
        let json = serde_json::json!({
            "wheels": { "foo": { "version": "1.2.3" } },
            "relax": "patch",
            "build-number": 7,
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.retread_wheels.len(), 1);
        assert_eq!(cfg.relax, RelaxPolicy::Patch);
        assert_eq!(cfg.build_number, 7);
    }

    #[test]
    fn rejects_entry_with_both_url_and_version() {
        let entry = WheelEntry {
            url: Some("https://example.com/x.whl".parse().unwrap()),
            version: Some("1.0".into()),
            ..Default::default()
        };
        assert!(entry.validate("x").is_err());
    }

    #[test]
    fn rejects_entry_with_neither_url_nor_version() {
        let entry = WheelEntry::default();
        assert!(entry.validate("x").is_err());
    }

    #[test]
    fn rejects_extras_on_url_form() {
        let entry = WheelEntry {
            url: Some("https://example.com/x.whl".parse().unwrap()),
            extras: vec!["all".into()],
            ..Default::default()
        };
        assert!(entry.validate("x").is_err());
    }

    #[test]
    fn python_spec_accepts_string_or_list() {
        let one: PythonSpec = serde_json::from_value(serde_json::json!("3.11")).unwrap();
        assert_eq!(one.as_versions(), vec!["3.11"]);
        let many: PythonSpec =
            serde_json::from_value(serde_json::json!(["3.11", "3.12"])).unwrap();
        assert_eq!(many.as_versions(), vec!["3.11", "3.12"]);
    }
}

impl WheelEntry {
    pub fn is_url(&self) -> bool {
        self.url.is_some()
    }
    pub fn is_spec(&self) -> bool {
        !self.is_url() && self.version.is_some()
    }

    /// Validate that the entry has exactly one form. Returns the normalized
    /// version (with leading `==` stripped) when in spec form.
    pub fn validate(&self, name: &str) -> Result<()> {
        if self.is_url() && self.version.is_some() {
            return Err(anyhow!(
                "wheel `{name}`: set either `url` or `version`, not both"
            ));
        }
        if !self.is_url() && self.version.is_none() {
            return Err(anyhow!(
                "wheel `{name}`: requires either `url = ...` or `version = ...`"
            ));
        }
        if self.is_url() && !self.extras.is_empty() {
            return Err(anyhow!(
                "wheel `{name}`: `extras` is only meaningful for the PyPI \
                 spec form (with `version`), not for direct URLs"
            ));
        }
        Ok(())
    }

    /// Normalized version string (leading `==` stripped). Only meaningful
    /// for spec-form entries.
    pub fn normalized_version(&self) -> Option<String> {
        self.version
            .as_ref()
            .map(|v| v.trim().trim_start_matches("==").trim().to_string())
    }

    /// Default index when in spec form.
    pub fn index_url(&self) -> String {
        self.index
            .clone()
            .unwrap_or_else(|| "https://pypi.org/simple/".to_string())
    }
}
