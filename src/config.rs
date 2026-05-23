//! User-facing configuration for the retread backend.
//!
//! Lives under `[build.config]` in the consumer's `pixi.toml`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct RetreadConfig {
    /// One or more wheels to repack. Each entry is a direct URL plus an
    /// optional SHA-256 for verification. Explicit URLs match the
    /// recipe.yaml pattern from the upstream Isaac Sim repackaging work and
    /// avoid the wheel-selection ambiguity of PyPI Simple resolution.
    pub wheels: Vec<WheelEntry>,

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

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct WheelEntry {
    /// Direct URL to the `.whl` file.
    pub url: url::Url,

    /// Optional SHA-256 of the wheel. If provided, the download is verified
    /// against this hash; otherwise the hash computed at fetch time is used
    /// only for caching / lock-file invalidation.
    #[serde(default)]
    pub sha256: Option<String>,
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

