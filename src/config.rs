//! User-facing configuration for the retread backend.
//!
//! Lives under `[build.config]` in the consumer's `pixi.toml`.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};
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
    /// - `minor`: ==X.Y.Z -> >=X.Y,<X+1
    /// - `major`: ==X.Y.Z -> >=X (drop upper bound)
    /// - `patch-then-minor-then-major-then-last-resort` (default,
    ///   v0.30.0+): emit at patch initially; the cascade widens
    ///   progressively on solve failure. See "Relax policies" in
    ///   the README.
    #[serde(default, rename = "retread-relax", alias = "relax")]
    pub relax: RelaxPolicy,

    /// Per-dependency overrides, applied after the relax policy. Map of
    /// PyPI name -> conda match-spec (e.g. `"*"`, `">=2.7"`).
    #[serde(default, rename = "retread-overrides", alias = "overrides")]
    pub overrides: BTreeMap<String, String>,

    /// PyPI -> conda name mapping overrides on top of the built-in identity
    /// mapping. Use for the common drift cases (`opencv-python-headless` ->
    /// `py-opencv`, etc.).
    #[serde(
        default,
        rename = "retread-name-map",
        alias = "name-map",
        alias = "name_map"
    )]
    pub name_map: BTreeMap<String, String>,

    /// PyPI names to drop from the conda run-deps entirely. Use for
    /// upstream-pinned deps that don't exist on the target conda channel
    /// (Windows-only shims like `idna-ssl`, `pywin32`) or otherwise can't
    /// be satisfied. The wheel still gets installed; conda just won't
    /// require these at solve time.
    #[serde(
        default,
        rename = "retread-drop-deps",
        alias = "drop-deps",
        alias = "drop_deps"
    )]
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
    #[serde(
        default = "default_true",
        rename = "retread-auto-bundle",
        alias = "auto-bundle"
    )]
    pub auto_bundle: bool,

    /// PyPI names that must stay as conda run-deps even when
    /// `retread-auto-bundle` is on. Use for ABI-sensitive packages where
    /// installing both a bundled wheel and the conda-channel version
    /// causes a collision -- typically the scientific stack (`numpy`,
    /// `scipy`, `pytorch`, `pandas`, ...). Empty by default; retread
    /// intentionally does NOT hard-code a list.
    #[serde(default, rename = "retread-conda-deps", alias = "conda-deps")]
    pub conda_deps: Vec<String>,

    /// v1.4.0: default `bundle` group for every `[retread-wheels]`
    /// entry that doesn't set its own `bundle` field. Saves repeating
    /// `bundle = "<name>"` on every entry of a single-output pack:
    ///
    /// ```toml
    /// [package.build.config]
    /// retread-bundle = "isaac-pack"
    /// ```
    ///
    /// A per-entry `bundle` still wins, so mixed grouping (some
    /// entries in the default group, some elsewhere or standalone)
    /// stays expressible. Unset preserves the historical behavior:
    /// entries without `bundle` each produce their own conda output.
    #[serde(default, rename = "retread-bundle", alias = "default-bundle")]
    pub default_bundle: Option<String>,

    /// Named git sources, referenced from `[retread-wheels]` entries
    /// via `from = "<name>"`. Avoids repeating `git = "..."` + `rev =
    /// "..."` across many sub-package entries from the same repo.
    /// Example:
    ///
    /// ```toml
    /// [package.build.config.retread-git-sources]
    /// isaaclab = { url = "https://github.com/isaac-sim/IsaacLab.git", rev = "deadbeef" }
    ///
    /// [package.build.config.retread-wheels]
    /// isaaclab        = { from = "isaaclab", subdirectory = "source/isaaclab" }
    /// isaaclab-assets = { from = "isaaclab", subdirectory = "source/isaaclab_assets" }
    /// ```
    #[serde(default, rename = "retread-git-sources", alias = "git-sources")]
    pub git_sources: BTreeMap<String, NamedGitSource>,

    /// Conda build number for the produced packages. Bump to force
    /// re-resolution downstream after a policy change.
    #[serde(default, rename = "retread-build-number", alias = "build-number")]
    pub build_number: u64,

    /// v1.5.8: zstd compression level for the produced .conda
    /// artifact, forwarded to rattler-build as
    /// `--package-format conda:<level>`. Unset uses rattler-build's
    /// default (high). For locally-consumed packs full of
    /// shared-library / asset payloads that barely compress (Isaac
    /// Sim, extscache), a LOW level (1-3) cuts the packaging stage of
    /// pixi's "preparing packages" phase dramatically for a few
    /// percent more disk:
    ///
    /// ```toml
    /// [package.build.config]
    /// retread-compression-level = 1
    /// ```
    #[serde(
        default,
        rename = "retread-compression-level",
        alias = "compression-level"
    )]
    pub compression_level: Option<u32>,

    /// DEPRECATED (v2.0.0), parsed and ignored. Use `retread-courier`
    /// instead. This field is retained so `deny_unknown_fields` does not
    /// reject old manifests that still carry it -- it parses but has no
    /// effect. Remove it from your `[package.build.config]`.
    #[serde(default, rename = "retread-emit-pypi", alias = "emit-pypi")]
    pub emit_pypi: bool,

    /// v2.1.0: courier mode -- the DEFAULT. Emit a metadata-only conda
    /// package that ships the bundle's built/relax-changed wheels + a
    /// committed `retread-<bundle>.lock.json` as data, declares the solved
    /// conda run-deps plus `uv`, and installs the PyPI wheels at env link
    /// time via a post-link `retread install` (uv hardlink; index wheels
    /// fetched on demand). The consumer keeps ONE clean conda line and zero
    /// machine-written manifest bytes; nothing is committed to git.
    ///
    /// Defaults to `true`. Set `retread-courier = false` to opt OUT to the
    /// legacy conda-artifact path (a full conda package carrying the wheels
    /// as a payload). Courier REQUIRES the consumer workspace to enable
    /// post-link scripts (`<workspace>/.pixi/config.toml` with
    /// `run-post-link-scripts = "insecure"`); without it the wheels are
    /// never installed -- the shipped conda `activate.d` guard prints a loud
    /// banner on every activation so this failure is never silent.
    ///
    /// ```toml
    /// [package.build.config]
    /// retread-courier = false  # opt out (legacy artifact path)
    /// ```
    #[serde(
        default = "default_true",
        rename = "retread-courier",
        alias = "courier"
    )]
    pub courier: bool,

    /// DEPRECATED (v2.0.0), parsed and ignored. Use `retread-courier`
    /// instead. This field is retained so `deny_unknown_fields` does not
    /// reject old manifests that still carry it -- it parses but has no
    /// effect. Remove it from your `[package.build.config]`.
    #[serde(default, rename = "retread-blueprint", alias = "blueprint")]
    pub blueprint: BlueprintMode,

    /// DEPRECATED (v2.0.0), parsed and ignored. Was `retread-blueprint-sync`
    /// (`"script"` | `"fence"`). Both the blueprint fence path and the
    /// install-script path have been superseded by `retread-courier`. The
    /// field is retained so `deny_unknown_fields` does not reject old
    /// manifests -- it parses but has no effect. Remove it from your
    /// `[package.build.config]`.
    #[serde(default, rename = "retread-blueprint-sync", alias = "blueprint-sync")]
    pub blueprint_sync: Option<String>,

    /// Python version(s) to build for, as a fallback when the workspace
    /// does not declare `[workspace.build-variants] python = [...]`.
    ///
    /// Accepts either a single string ("3.11") or a list of strings
    /// (`["3.11", "3.12"]`). When the workspace's variant configuration
    /// provides `python`, that wins; this field is purely a convenience for
    /// single-Python workspaces. The default is `3.11`.
    ///
    /// Also used as the fallback in conda/build_v1 when pixi forwards a
    /// bare-major variant value (`"3"`) — see handler.rs::conda_build_v1.
    /// The bare name `python` is also accepted as a legacy alias.
    #[serde(default, rename = "retread-python", alias = "python")]
    pub python: Option<PythonSpec>,
}

/// `retread-blueprint` accepts `false` (off), `true` (blueprint +
/// full conda artifact), or `"only"` (blueprint + stub conda
/// artifact).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(from = "BlueprintModeRepr")]
pub enum BlueprintMode {
    #[default]
    Off,
    Full,
    Only,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum BlueprintModeRepr {
    Flag(bool),
    Mode(String),
}

impl From<BlueprintModeRepr> for BlueprintMode {
    fn from(r: BlueprintModeRepr) -> Self {
        match r {
            BlueprintModeRepr::Flag(false) => Self::Off,
            BlueprintModeRepr::Flag(true) => Self::Full,
            BlueprintModeRepr::Mode(s) if s.eq_ignore_ascii_case("only") => Self::Only,
            BlueprintModeRepr::Mode(s) => {
                // deny_unknown_fields can't see inside the value; be loud
                // rather than silently off.
                tracing::warn!(value = %s, "unknown retread-blueprint value; treating as true");
                Self::Full
            }
        }
    }
}

impl BlueprintMode {
    pub fn is_on(self) -> bool {
        !matches!(self, Self::Off)
    }
    pub fn is_only(self) -> bool {
        matches!(self, Self::Only)
    }
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
    Minor,
    Major,
    /// Major widening AND aggressive range-spec relaxing: drops every
    /// upper bound on every requirement (`<X`, `<=X`, the `<Y` half of
    /// `>=X,<Y`, the implicit upper of `~=X.Y`). Lower bounds (`>=`,
    /// `>`) stay so conda doesn't pick something pre-historic. Use
    /// when upstream caps are blocking the conda solve and you trust
    /// the conda side to pick a working version.
    #[serde(rename = "strong-major")]
    StrongMajor,
    /// v0.19.0+ "with-last-resort" family. Each behaves IDENTICALLY
    /// to its base (patch / minor / major) at translate time, plus an
    /// automated cascade for deps whose post-translate conda spec
    /// turns out unsatisfiable on the workspace's channels. Cascade
    /// per dep, in order:
    ///   (1) try conda with the base-relaxed spec (probe)
    ///   (2) try PyPI wheel (BFS)            -- wired since v0.13.x
    ///   (3) try PyPI sdist (BFS fallback)   -- wired since v0.18.0
    ///   (4) try conda with `*` (any version) -- last-resort widening
    ///   (5) try PyPI bundle with empty spec  -- not yet implemented
    /// Widening only triggers for deps that FAIL step 1; zero cost for
    /// the common case where parselmouth-routed deps satisfy their
    /// strict spec. `minor-with-last-resort` is the recommended
    /// default; `patch-with-last-resort` for super-strict envs (still
    /// auto-widens when forced); `major-with-last-resort` mostly
    /// equivalent to plain `major` since major already widens broadly,
    /// but exists for symmetry. Surgical alternative to `strong-major`
    /// which strips upper bounds bundle-wide regardless of need.
    #[serde(rename = "patch-with-last-resort")]
    PatchWithLastResort,
    #[serde(rename = "minor-with-last-resort")]
    MinorWithLastResort,
    #[serde(rename = "major-with-last-resort")]
    MajorWithLastResort,
    /// v0.30.0+ tiered cascade. At translate time emits at the narrowest
    /// (patch) widening, then per-dep escalates only when probes prove
    /// the current widening level is unsatisfiable. Per dep, in order:
    ///   (1) probe conda at the patch-widened spec
    ///   (2) probe PyPI for a wheel matching the patch range -> bundle + drop conda emit
    ///   (3) probe conda at the minor-widened spec
    ///   (4) probe PyPI at the minor range -> bundle + drop conda emit
    ///   (5) probe conda at the major-widened spec
    ///   (6) probe PyPI at the major range -> bundle + drop conda emit
    ///   (7) widen the emitted conda spec to `*` (any version)
    /// At each conda step the workspace's conda solver picks the highest
    /// in-range candidate; at each PyPI step the PyPI resolver picks the
    /// highest in-range wheel. Decisions per step land in the audit
    /// under stage `tiered-cascade-stepN-{conda,pypi}`. Use this when
    /// you want strict-by-default behavior with automatic recovery
    /// across multiple widening levels before reaching for `*`.
    #[default]
    #[serde(rename = "patch-then-minor-then-major-then-last-resort")]
    PatchThenMinorThenMajorThenLastResort,
    /// TODO(conda-aware): NOT YET IMPLEMENTED. The variant deserializes
    /// and accepts the value `"conda-aware"` from user config, but the
    /// probe layer described below does not exist -- at translate time
    /// this currently behaves IDENTICALLY to `StrongMajor` (strips every
    /// upper bound unconditionally). Do not document this option in the
    /// README until the probe is wired up. For "strict by default, widen
    /// only when needed" semantics today, use `minor` + per-package
    /// `retread-overrides` entries, OR use `minor-with-last-resort`
    /// which automates the widening for unsatisfiable deps.
    ///
    /// Intended design when implemented: per-dep adaptive widening.
    /// Starts at major (exact pins widen, ranges pass through). Then
    /// for each emitted spec containing an upper bound (`<`, `<=`,
    /// `~=`), retread probes the workspace's conda channels: if zero
    /// candidates satisfy the spec under the workspace's python,
    /// retread strips the upper bound and re-emits for that one dep.
    /// Specs without upper bounds skip the probe. All decisions land
    /// in `retread-audit.json` under `probe_results[]`.
    #[serde(rename = "conda-aware")]
    CondaAware,
}

impl RelaxPolicy {
    /// True for any `*-with-last-resort` variant. Used by the pre/post
    /// widen passes' simpler "widen unsat -> `*`" mutation path.
    pub fn has_last_resort(self) -> bool {
        matches!(
            self,
            RelaxPolicy::PatchWithLastResort
                | RelaxPolicy::MinorWithLastResort
                | RelaxPolicy::MajorWithLastResort,
        )
    }

    /// True for the v0.30.0+ tiered cascade variant. Triggers the
    /// patch -> minor -> major -> last-resort escalation in both
    /// widening passes (pre-emit + post-emit).
    pub fn has_tiered_cascade(self) -> bool {
        matches!(self, RelaxPolicy::PatchThenMinorThenMajorThenLastResort)
    }

    /// True if mutation (override injection / spec rewriting) is
    /// allowed for unsat probes. Both passes always *probe* and
    /// *record* regardless of policy; they only *mutate* when this
    /// returns true.
    pub fn allows_widening_mutation(self) -> bool {
        self.has_last_resort() || self.has_tiered_cascade()
    }
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

    // ---- Local source form ----
    /// Local path to a Python project. retread runs `pip wheel --no-deps`
    /// against this path to produce a wheel, then runs it through the
    /// usual METADATA-rewrite + bundle pipeline. Relative paths resolve
    /// against the source package's pixi.toml directory.
    #[serde(default)]
    pub path: Option<String>,

    // ---- Git source form ----
    /// HTTPS git URL of a Python project. retread clones at `rev` and
    /// runs `pip wheel --no-deps` on `subdirectory` (defaulting to ".").
    #[serde(default)]
    pub git: Option<String>,

    /// Git revision (commit SHA, tag, or branch) for `git` source.
    /// Required when `git` is set.
    #[serde(default)]
    pub rev: Option<String>,

    /// Subdirectory within the git clone containing the Python project
    /// to build. Defaults to "." (the repo root).
    #[serde(default)]
    pub subdirectory: Option<String>,

    /// If true, after the wheel is built and the conda env is
    /// materialized, the user is expected to run
    /// `pip install -e <path> --no-deps --force-reinstall` to overlay
    /// editable on top of the bundled installation. retread doesn't
    /// run this for you -- see the example pixi.toml's
    /// `overlay-editable` task. The wheel is still built normally so
    /// retread can rewrite METADATA + emit a coherent dep set.
    #[serde(default)]
    pub editable: bool,

    // ---- Named git source reference ----
    /// Reference to a `[retread-git-sources]` entry. The named entry
    /// provides `url` + `rev`; this wheel entry only contributes
    /// `subdirectory` (defaulting to "."). Lets many sub-packages from
    /// the same monorepo share a single rev declaration.
    #[serde(default)]
    pub from: Option<String>,

    /// Group entries that share this string into a single conda output.
    /// The output's name is the bundle string. All wheels from grouped
    /// entries become the bundle's wheels (primary = first entry's
    /// primary, others as extras). Without this field each entry
    /// produces its own conda output named after the entry key.
    ///
    /// Use this when you want a single workspace declaration (e.g.
    /// `isaac-pack = { path = "./isaac-pack" }`) to install every
    /// wheel in the pack -- pixi-build only builds outputs the
    /// workspace declared, so grouping into one output sidesteps the
    /// N-line declaration problem at the cost of a bigger artifact.
    #[serde(default)]
    pub bundle: Option<String>,
}

/// A named (url, rev) pair used to share a git source across many
/// `[retread-wheels]` entries.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct NamedGitSource {
    pub url: String,
    pub rev: String,
}

impl WheelEntry {
    pub fn is_url(&self) -> bool {
        self.url.is_some()
    }
    pub fn is_spec(&self) -> bool {
        !self.is_url() && !self.is_path() && !self.is_git() && self.version.is_some()
    }
    pub fn is_path(&self) -> bool {
        self.path.is_some()
    }
    pub fn is_git(&self) -> bool {
        self.git.is_some()
    }
    pub fn is_named_git(&self) -> bool {
        self.from.is_some()
    }

    /// Validate that the entry has exactly one form.
    pub fn validate(&self, name: &str) -> Result<()> {
        let form_count = [
            self.url.is_some(),
            self.version.is_some(),
            self.path.is_some(),
            self.git.is_some(),
            self.from.is_some(),
        ]
        .into_iter()
        .filter(|b| *b)
        .count();
        if form_count == 0 {
            return Err(anyhow!(
                "wheel `{name}`: requires one of `url`, `version`, `path`, `git`, or `from`"
            ));
        }
        if form_count > 1 {
            return Err(anyhow!(
                "wheel `{name}`: set exactly ONE of `url`, `version`, `path`, `git`, `from`"
            ));
        }
        if self.is_git() && self.rev.is_none() {
            return Err(anyhow!(
                "wheel `{name}`: `git` requires `rev` (commit, tag, or branch)"
            ));
        }
        if self.is_url() && !self.extras.is_empty() {
            return Err(anyhow!(
                "wheel `{name}`: `extras` is not meaningful on the bare-URL \
                 form (the upstream filename doesn't carry a project name to \
                 attach extras to). Use the `version`, `path`, `git`, or \
                 `from` form instead."
            ));
        }
        // v0.12.0+: extras now valid on path/git/named-git. The wheel
        // built from those sources carries `Requires-Dist: ...; extra
        // == "<name>"` lines in METADATA, and the handler runs the
        // standard extras BFS on them (just like PyPI form), pulling
        // the extras' deps via PyPI Simple.
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
    fn parses_retread_compression_level_key() {
        // v1.5.8. Invariant #6: every new config field needs a serde
        // test or stale binaries reject user pixi.toml with "unknown
        // field" during upgrades.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-compression-level": 3,
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.compression_level, Some(3));

        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.compression_level, None);
    }

    #[test]
    fn parses_retread_emit_pypi_key() {
        // DEPRECATED (v2.0.0): the field is parsed-and-ignored so old
        // manifests survive an upgrade without a "unknown field" rejection.
        // Invariant #6: the field must still parse correctly.
        // The value is now ignored; use `retread-courier` instead.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-emit-pypi": true,
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.emit_pypi);

        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(!cfg.emit_pypi, "unset defaults to off");
    }

    #[test]
    fn parses_retread_courier_key() {
        // v2.1.0: courier is the DEFAULT. Invariant #6: deny_unknown_fields
        // rejects unknown keys, so the flag needs a serde test.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-courier": true,
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.courier);

        // unset -> defaults to ON (courier is the default mode, v2.1.0+).
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.courier, "unset defaults to courier");

        // explicit opt-out to the legacy artifact path.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-courier": false,
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(!cfg.courier, "explicit false opts out to legacy");
    }

    #[test]
    fn deprecated_blueprint_sync_key_still_parses() {
        // DEPRECATED (v2.0.0): the field is parsed-and-ignored; the
        // deprecation contract (Invariant #6) requires that old manifests
        // carrying this key still parse rather than failing with "unknown
        // field". The value is now ignored; use `retread-courier` instead.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-blueprint-sync": "script",
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.blueprint_sync.as_deref(), Some("script"));

        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.blueprint_sync, None, "absent by default");
    }

    #[test]
    fn parses_retread_blueprint_key() {
        // DEPRECATED (v2.0.0): the field is parsed-and-ignored so old
        // manifests survive an upgrade without a "unknown field" rejection
        // (Invariant #6). The value is now ignored; use `retread-courier`
        // instead. is_on() / is_only() are called here to keep the methods
        // reachable (they are referenced from handler/mod.rs deprecation warn).
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-blueprint": true,
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.blueprint.is_on());
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-blueprint": "only",
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.blueprint.is_only());

        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(!cfg.blueprint.is_on(), "unset defaults to off");
    }

    #[test]
    fn parses_retread_bundle_key() {
        // v1.4.0: pack-wide default bundle group. Invariant #6: every
        // new config field needs a serde test or stale binaries reject
        // user pixi.toml with "unknown field" during upgrades.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-bundle": "isaac-pack",
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.default_bundle.as_deref(), Some("isaac-pack"));

        // Unset stays None (historical per-entry-output behavior).
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.default_bundle, None);
    }

    #[test]
    fn legacy_unprefixed_keys_still_parse() {
        // Kept for back-compat; removing would hard-error old manifests
        // via deny_unknown_fields. The serde aliases (`alias = "wheels"`,
        // `alias = "relax"`, etc.) accept the pre-0.4 unprefixed keys
        // indefinitely so existing pixi.toml files keep loading.
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
    fn parses_bundle_field_on_entry() {
        // Regression: when the `bundle` field was added to WheelEntry,
        // a stale retread without the field would reject the user's
        // pixi.toml with `unknown field bundle, expected one of ...`
        // (because of `deny_unknown_fields`). This test pins the
        // parse-side contract: any new optional field added to
        // WheelEntry must be tested through serde deserialization
        // BEFORE we ship it, or stale binaries break every user's
        // pixi.toml during the upgrade window.
        let json = serde_json::json!({
            "retread-wheels": {
                "isaacsim": {
                    "version": "==5.1.0",
                    "index": "https://pypi.nvidia.com",
                    "extras": ["all"],
                    "bundle": "isaac-pack",
                },
                "isaaclab": {
                    "from": "isaaclab",
                    "subdirectory": "source/isaaclab",
                    "bundle": "isaac-pack",
                },
                "loner": { "version": "==1.0" }
            }
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(
            cfg.retread_wheels["isaacsim"].bundle.as_deref(),
            Some("isaac-pack")
        );
        assert_eq!(
            cfg.retread_wheels["isaaclab"].bundle.as_deref(),
            Some("isaac-pack")
        );
        // Default is None when omitted -- one entry == one output.
        assert_eq!(cfg.retread_wheels["loner"].bundle, None);
    }

    #[test]
    fn parses_tiered_cascade_policy() {
        // v0.30.0+ tiered cascade variant. New string value; pin the
        // serde rename so stale binaries don't reject pixi.toml with
        // "unknown variant" during the upgrade window.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "1.2.3" } },
            "retread-relax": "patch-then-minor-then-major-then-last-resort",
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(
            cfg.relax,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort
        );
        assert!(cfg.relax.has_tiered_cascade());
        assert!(cfg.relax.allows_widening_mutation());
        assert!(!cfg.relax.has_last_resort());
    }

    #[test]
    fn relax_policy_helper_semantics() {
        assert!(RelaxPolicy::MinorWithLastResort.has_last_resort());
        assert!(RelaxPolicy::MinorWithLastResort.allows_widening_mutation());
        assert!(!RelaxPolicy::MinorWithLastResort.has_tiered_cascade());

        assert!(!RelaxPolicy::Minor.has_last_resort());
        assert!(!RelaxPolicy::Minor.has_tiered_cascade());
        assert!(!RelaxPolicy::Minor.allows_widening_mutation());
    }

    #[test]
    fn rejects_unknown_field_on_entry() {
        // Mirror image of the above: deny_unknown_fields must fire on
        // anything we haven't added to WheelEntry. If this test ever
        // starts failing, it means we shipped a permissive parser and
        // the next time we add a field, stale binaries won't break --
        // but neither will typos in user pixi.toml. Trade-off
        // documented; current default is strict.
        let json = serde_json::json!({
            "retread-wheels": {
                "foo": { "version": "==1.0", "totally-bogus-key": "x" }
            }
        });
        let err = serde_json::from_value::<RetreadConfig>(json).unwrap_err();
        assert!(
            err.to_string().contains("totally-bogus-key"),
            "unknown-field error should name the offender; got: {err}",
        );
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

    /// v0.12.0+: extras are valid on git/path/named-git entries. The
    /// extras' deps come through METADATA's `Requires-Dist: ...; extra
    /// == "<name>"` lines on the built wheel and get BFS-resolved via
    /// PyPI Simple by the handler, same as for the spec form.
    #[test]
    fn accepts_extras_on_path_and_git_and_named_git() {
        let path_entry = WheelEntry {
            path: Some("./local-pkg".into()),
            extras: vec!["rl_games".into()],
            ..Default::default()
        };
        path_entry.validate("a").expect("extras valid on path form");

        let git_entry = WheelEntry {
            git: Some("https://example.com/repo.git".into()),
            rev: Some("HEAD".into()),
            extras: vec!["rl_games".into()],
            ..Default::default()
        };
        git_entry.validate("b").expect("extras valid on git form");

        let named_git_entry = WheelEntry {
            from: Some("isaaclab".into()),
            extras: vec!["rl_games".into()],
            ..Default::default()
        };
        named_git_entry
            .validate("c")
            .expect("extras valid on named-git form");
    }

    #[test]
    fn python_spec_accepts_string_or_list() {
        let one: PythonSpec = serde_json::from_value(serde_json::json!("3.11")).unwrap();
        assert_eq!(one.as_versions(), vec!["3.11"]);
        let many: PythonSpec = serde_json::from_value(serde_json::json!(["3.11", "3.12"])).unwrap();
        assert_eq!(many.as_versions(), vec!["3.11", "3.12"]);
    }
}
