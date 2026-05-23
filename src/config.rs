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
    #[serde(default)]
    pub relax: RelaxPolicy,

    /// Per-dependency overrides, applied after the relax policy. Map of
    /// PyPI name -> conda match-spec (e.g. `"*"`, `">=2.7"`).
    #[serde(default)]
    pub overrides: BTreeMap<String, String>,

    /// PyPI -> conda name mapping overrides on top of the built-in identity
    /// mapping. Use for the common drift cases (`opencv-python-headless` ->
    /// `py-opencv`, etc.).
    #[serde(default)]
    pub name_map: BTreeMap<String, String>,

    /// Conda build number. Bump to force re-resolution downstream after a
    /// policy change.
    #[serde(default)]
    pub build_number: u64,
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
