//! v0.32.0+: workspace pixi.toml parser. Reads the consumer workspace's
//! `[dependencies]`, `[pypi-dependencies]`, `[environments]`, and
//! `[feature.X.*]` tables so retread's cascade can:
//!
//! 1. Mirror the workspace's pinned versions at the last-resort step
//!    (replaces v0.31.0's `read_workspace_pins`).
//! 2. Compute per-env effective channels + deps when
//!    `retread-per-env` is set, so each emitted output reflects
//!    that env's actual constraints (not a union across envs).
//! 3. Walk per-env conda deps' `depends` arrays to extract transitive
//!    constraints (e.g. ros-humble-joint-state-publisher pins
//!    `numpy <2`, so retread should emit numpy compatibly for any
//!    env that includes it).
//!
//! Best-effort: every parse function returns `Option<...>` so missing
//! files or malformed sections don't kill the build. Callers treat
//! `None` as "no info; behave like the workspace doesn't exist."

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

use crate::relax::{CondaMatchSpec, CondaName, PypiKey};

pub(crate) const DEFAULT_PYPI_INDEX: &str = "https://pypi.org/simple/";
pub(crate) const WORKSPACE_TARGET_ENV: &str = "RETREAD_WORKSPACE_TARGET_JSON";

/// One named Pixi workspace platform declaration.
///
/// Pixi 0.73 permits several named profiles to share one conda subdir. Keeping
/// the name and the complete declared virtual-package map is load-bearing:
/// reducing this to `subdir -> glibc` makes a later rich linux-64 profile
/// silently overwrite an earlier one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceProfile {
    pub name: String,
    pub subdir: String,
    pub declared_virtual_packages: BTreeMap<String, String>,
}

impl WorkspaceProfile {
    pub fn semantic_contract(&self) -> WorkspaceTargetContract {
        WorkspaceTargetContract {
            subdir: self.subdir.clone(),
            declared_virtual_packages: self.declared_virtual_packages.clone(),
            // Pixi owns the default/detected virtual-package table. Retread
            // deliberately does not duplicate it: an exact orchestrated call
            // supplies this map from `pixi workspace platform list --json`.
            detected_virtual_packages: BTreeMap::new(),
        }
    }
}

/// Name-independent target contract used by resolution, sidecar, and cache
/// identities. Two profile aliases may share work only when these canonical
/// maps are byte-for-byte equal.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WorkspaceTargetContract {
    pub subdir: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub declared_virtual_packages: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub detected_virtual_packages: BTreeMap<String, String>,
}

/// Whether Pixi 0.73 maps a declared virtual package into its detection
/// override model. Raw forward-compatible packages, `__unix`, and
/// `__cuda_arch` remain part of declared platform identity but are not
/// preserved in the detected list.
pub(crate) fn pixi_detects_declared_virtual_package(name: &str) -> bool {
    matches!(
        name,
        "cuda" | "archspec" | "glibc" | "musl" | "eglibc" | "linux" | "macos" | "windows"
    )
}

impl WorkspaceTargetContract {
    pub fn declared_glibc(&self) -> Option<(u32, u32)> {
        self.declared_virtual_packages
            .get("glibc")
            .and_then(|value| crate::glibc::parse_glibc_version(value))
    }

    /// Effective glibc floor. An orchestrator's Pixi-detected contract is
    /// authoritative; direct source-consumer inference can use only values
    /// explicitly declared on the selected rich profile.
    pub fn effective_glibc(&self) -> Option<(u32, u32)> {
        self.detected_virtual_packages
            .get("glibc")
            .or_else(|| self.declared_virtual_packages.get("glibc"))
            .and_then(|value| crate::glibc::parse_glibc_version(value))
    }

    /// Virtual-package overrides in the schema consumed by Retread's conda
    /// solver. Pixi-detected values replace declaration-only values when an
    /// exact envelope is present, except that Pixi 0.73 omits declared CUDA
    /// compute capability from its detected list.
    pub fn system_requirements(&self) -> BTreeMap<String, String> {
        let packages = if self.detected_virtual_packages.is_empty() {
            &self.declared_virtual_packages
        } else {
            &self.detected_virtual_packages
        };
        let mut requirements: BTreeMap<String, String> = packages
            .iter()
            .filter_map(|(name, value)| match name.as_str() {
                "glibc" => Some(("libc".to_string(), value.clone())),
                "cuda" | "cuda_arch" | "linux" | "osx" | "windows" | "win" => {
                    Some((name.clone(), value.clone()))
                }
                "macos" => Some(("osx".to_string(), value.clone())),
                "archspec" => Some((
                    name.clone(),
                    value
                        .rsplit_once('=')
                        .map_or(value.as_str(), |(_, build)| build)
                        .to_string(),
                )),
                _ => None,
            })
            .collect();
        if !self.detected_virtual_packages.is_empty()
            && let Some(cuda_arch) = self.declared_virtual_packages.get("cuda_arch")
        {
            requirements
                .entry("cuda_arch".to_string())
                .or_insert_with(|| cuda_arch.clone());
        }
        requirements
    }
}

/// Provenance for a uniquely inferred semantic target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedWorkspaceTarget {
    pub contract: WorkspaceTargetContract,
    pub profiles: Vec<String>,
    pub environments: Vec<String>,
}

/// Exact target selected by a workspace-level orchestrator. The nested
/// profile is shaped like one entry from `pixi workspace platform list
/// --json`; the backend validates it against the manifest before trusting it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceTargetEnvelope {
    pub schema: u32,
    pub environment: String,
    pub profile: WorkspaceTargetEnvelopeProfile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceTargetEnvelopeProfile {
    pub name: String,
    pub subdir: String,
    #[serde(default)]
    pub virtual_packages: Vec<String>,
    #[serde(default)]
    pub detected_virtual_packages: Vec<String>,
}

impl WorkspaceTargetEnvelope {
    /// Read an optional exact target selection supplied by a workspace
    /// orchestrator. Parsing happens once during backend initialization; the
    /// manifest/source validation remains request-specific.
    pub fn from_process_env() -> Result<Option<Self>> {
        let Some(raw) = std::env::var_os(WORKSPACE_TARGET_ENV) else {
            return Ok(None);
        };
        let raw = raw
            .into_string()
            .map_err(|_| anyhow!("{WORKSPACE_TARGET_ENV} must contain UTF-8 JSON"))?;
        serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse {WORKSPACE_TARGET_ENV}"))
            .map(Some)
    }
}

/// Parsed workspace pixi.toml. Only the fields retread cares about.
#[derive(Debug, Default, Clone)]
pub struct WorkspaceManifest {
    /// Top-level `[workspace].channels`.
    pub channels: Vec<String>,
    /// Top-level `[workspace].channel-priority` setting. `None` when
    /// unspecified. Used by the solve check to match what pixi's
    /// own solver will do -- without this, retread defaults to
    /// Strict and reports false unsats when the workspace actually
    /// runs Disabled (pytorch + conda-forge mixed-channel envs).
    pub channel_priority: Option<String>,
    /// Declaration-ordered workspace platform profiles. Names remain distinct
    /// even when several profiles share one conda subdir.
    pub target_profiles: Vec<WorkspaceProfile>,
    /// Top-level `[dependencies]` — conda deps with a version spec.
    /// Used for transitive constraint extraction (each maps to a real
    /// channel package whose `depends` array we walk).
    pub dependencies: BTreeMap<String, String>,
    /// Top-level `[dependencies]` path-form entries
    /// (`<name> = { path = "...", ... }`). Used by output
    /// autodiscovery to find which workspace declarations reference
    /// THIS source package.
    pub path_dependencies: BTreeMap<String, String>,
    /// Top-level conda source entries (`git = ...` / `url = ...`). These are
    /// not local Retread producers, but retaining them is load-bearing for
    /// target overlays and active-feature source conflict detection.
    pub source_dependencies: BTreeMap<String, String>,
    /// Top-level `[target.<platform>.dependencies]`, keyed by Pixi platform.
    /// Target-specific entries overlay the ordinary top-level dependencies.
    pub target_dependencies: Vec<(String, TargetDependencyDef)>,
    /// Top-level `[pypi-dependencies]`, keyed by canonical PEP 503
    /// package name. Version requirements are preserved verbatim;
    /// direct URL/path/git declarations are represented by `"*"`
    /// because the workspace owns their source without declaring a
    /// registry version constraint.
    pub pypi_dependencies: BTreeMap<String, String>,
    /// Per-environment definitions from `[environments]`.
    pub environments: BTreeMap<String, EnvironmentDef>,
    /// Per-feature blocks from `[feature.X.*]`.
    pub features: BTreeMap<String, FeatureDef>,
    /// v0.37.0+ (D1): top-level `[system-requirements]`. Keys are
    /// pixi-schema names (`cuda`, `libc`, `macos`, `archspec`, ...);
    /// values are the declared version. Mapped to rattler virtual
    /// packages (`cuda`->`__cuda`, `libc`->`__glibc`, ...) at the
    /// solve_check boundary so retread's solve matches pixi's instead
    /// of defaulting to the build host's detected virtual packages.
    pub system_requirements: BTreeMap<String, String>,
    /// pixi 0.71+ rich platform declarations from `[workspace].platforms`,
    /// keyed by pixi platform string (`linux-64`, `linux-aarch64`, ...).
    pub platform_glibc: BTreeMap<String, String>,
    /// pixi 0.71+ rich platform `cuda = "..."` declarations, keyed the same
    /// way as [`Self::platform_glibc`]. Folded into
    /// `effective_system_requirements` (rich wins over the deprecated
    /// `[system-requirements]` table, mirroring `declared_glibc`).
    pub platform_cuda: BTreeMap<String, String>,
    /// Top-level `[pypi-options]`. Keeping the primary index separate
    /// from extras preserves whether pixi's implicit public-PyPI default
    /// was explicitly replaced by `index-url`.
    pub pypi_options: PypiOptions,
}

#[derive(Debug, Default, Clone)]
pub struct EnvironmentDef {
    /// Features this env activates, in declaration order.
    pub features: Vec<String>,
    /// If true, the implicit "default" feature (top-level
    /// `[dependencies]` etc.) is NOT inherited.
    pub no_default_feature: bool,
}

/// The index-bearing subset of a pixi `[pypi-options]` table.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PypiOptions {
    /// Explicit replacement for pixi's default PyPI index.
    pub index_url: Option<String>,
    /// Additional indexes, in declaration order.
    pub extra_index_urls: Vec<String>,
}

impl PypiOptions {
    /// Raw declared index candidates, retaining the historical
    /// `index-url`-then-extras order used by fingerprints and effective
    /// environment views.
    fn declared_index_urls(&self) -> impl Iterator<Item = &String> {
        self.index_url.iter().chain(self.extra_index_urls.iter())
    }
}

#[derive(Debug, Default, Clone)]
pub struct FeatureDef {
    /// Optional `[feature.X].platforms` restriction. `None` is unrestricted;
    /// `Some(empty)` supports no platform and is therefore not equivalent.
    pub platforms: Option<Vec<String>>,
    /// `[feature.X.channels]`.
    pub channels: Vec<String>,
    /// `[feature.X.dependencies]` — version-pinned conda deps.
    pub dependencies: BTreeMap<String, String>,
    /// `[feature.X.dependencies]` path-form entries. Output
    /// autodiscovery walks these to find features that reference the
    /// source package retread is building for.
    pub path_dependencies: BTreeMap<String, String>,
    /// `[feature.X.dependencies]` conda git/URL sources. A source declaration
    /// may coexist with a registry constraint, but conflicts with an active
    /// path declaration for the same package.
    pub source_dependencies: BTreeMap<String, String>,
    /// `[feature.X.target.<platform>.dependencies]`, keyed by Pixi platform.
    pub target_dependencies: Vec<(String, TargetDependencyDef)>,
    /// `[feature.X.pypi-dependencies]`, with the same canonical-name
    /// and direct-source semantics as
    /// [`WorkspaceManifest::pypi_dependencies`].
    pub pypi_dependencies: BTreeMap<String, String>,
    /// v0.37.0+ (D1): `[feature.X.system-requirements]`. Same shape as
    /// the top-level table; unioned per active env with feature-wins
    /// precedence by `effective_system_requirements`.
    pub system_requirements: BTreeMap<String, String>,
    /// `[feature.X.pypi-options]`, with primary and extra indexes kept
    /// distinct for the same reason as the top-level options.
    pub pypi_options: PypiOptions,
}

/// Version and path dependencies declared under one target selector.
#[derive(Debug, Default, Clone)]
pub struct TargetDependencyDef {
    pub dependencies: BTreeMap<String, String>,
    pub path_dependencies: BTreeMap<String, String>,
    pub source_dependencies: BTreeMap<String, String>,
    pub pypi_dependencies: BTreeMap<String, String>,
}

impl WorkspaceManifest {
    /// Open `workspace_dir/pixi.toml` and parse. Returns `None` if the
    /// file is missing or fundamentally malformed; individual
    /// malformed entries are skipped rather than aborting.
    ///
    /// v1.4.0: memoized per (path, mtime). `conda_outputs` and its
    /// callees load the manifest 5+ times per request; the file never
    /// changes mid-request, so re-reading + re-parsing the same TOML
    /// was pure waste. The mtime key keeps the memo correct if the
    /// file IS edited while the backend process is alive.
    pub fn load(workspace_dir: &Path) -> Option<Self> {
        use std::collections::HashMap;
        use std::path::PathBuf;
        use std::sync::{Mutex, OnceLock};
        use std::time::SystemTime;

        static CACHE: OnceLock<Mutex<HashMap<PathBuf, (SystemTime, WorkspaceManifest)>>> =
            OnceLock::new();

        let pixi_toml = workspace_dir.join("pixi.toml");
        let mtime = std::fs::metadata(&pixi_toml).ok()?.modified().ok()?;
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some((cached_mtime, manifest)) = cache.lock().unwrap().get(&pixi_toml)
            && *cached_mtime == mtime
        {
            return Some(manifest.clone());
        }
        let bytes = std::fs::read_to_string(&pixi_toml).ok()?;
        let manifest = Self::from_toml_source(&bytes)?;
        cache
            .lock()
            .unwrap()
            .insert(pixi_toml, (mtime, manifest.clone()));
        Some(manifest)
    }

    /// Build from an already-parsed TOML value. Useful for testing
    /// without touching the filesystem.
    pub fn from_toml(parsed: &toml::Value) -> Self {
        let mut out = Self::default();

        if let Some(workspace) = parsed.get("workspace").and_then(|v| v.as_table()) {
            if let Some(chans) = workspace.get("channels").and_then(|v| v.as_array()) {
                out.channels = chans
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
            }
            out.channel_priority = workspace
                .get("channel-priority")
                .or_else(|| workspace.get("channel_priority"))
                .and_then(|v| v.as_str())
                .map(String::from);
            if let Some(platforms) = workspace.get("platforms").and_then(|v| v.as_array()) {
                for platform in platforms {
                    let Some(profile) = parse_workspace_profile(platform) else {
                        continue;
                    };
                    if let Some(existing) = out
                        .target_profiles
                        .iter()
                        .position(|candidate| candidate.name == profile.name)
                    {
                        // Pixi rejects duplicate rich platform names. Keep
                        // parsing best-effort here, with the final declaration
                        // replacing an identical subdir-shaped entry.
                        out.target_profiles[existing] = profile;
                    } else {
                        out.target_profiles.push(profile);
                    }
                }
                derive_unambiguous_platform_values(
                    &out.target_profiles,
                    "glibc",
                    &mut out.platform_glibc,
                );
                derive_unambiguous_platform_values(
                    &out.target_profiles,
                    "cuda",
                    &mut out.platform_cuda,
                );
            }
        }

        if let Some(deps) = parsed.get("dependencies").and_then(|v| v.as_table()) {
            for (name, value) in deps {
                match classify_dep_value(value) {
                    DepKind::Version(spec) => {
                        out.dependencies.insert(name.clone(), spec);
                    }
                    DepKind::Path(path) => {
                        out.path_dependencies.insert(name.clone(), path);
                    }
                    DepKind::Source(source) => {
                        out.source_dependencies.insert(name.clone(), source);
                    }
                    DepKind::Other => {}
                }
            }
        }
        out.target_dependencies = parse_target_dependencies(parsed);

        out.pypi_dependencies = parse_pypi_dependencies(parsed);

        // v0.37.0+ (D1): top-level [system-requirements]. Scalar values
        // (`cuda = "12"`) stored verbatim; table form
        // (`libc = { family = "glibc", version = "2.35" }`) takes the
        // version field. Unrecognized shapes are skipped.
        if let Some(sysreqs) = parsed
            .get("system-requirements")
            .or_else(|| parsed.get("system_requirements"))
            .and_then(|v| v.as_table())
        {
            for (k, v) in sysreqs {
                if let Some(s) = parse_system_requirement_value(k, v) {
                    out.system_requirements.insert(k.clone(), s);
                }
            }
        }

        out.pypi_options = parse_pypi_options(parsed);

        if let Some(envs) = parsed.get("environments").and_then(|v| v.as_table()) {
            for (name, value) in envs {
                if let Some(def) = parse_env_def(value) {
                    out.environments.insert(name.clone(), def);
                }
            }
        }
        // Pixi always materializes the implicit `default` environment, even
        // when the manifest declares only named environments (or no
        // `[environments]` table at all). It activates the top-level default
        // feature and has no additional named features. Preserve an explicit
        // user declaration when one exists.
        out.environments.entry("default".to_string()).or_default();

        if let Some(features) = parsed.get("feature").and_then(|v| v.as_table()) {
            for (name, fvalue) in features {
                let mut def = FeatureDef::default();
                if let Some(fmap) = fvalue.as_table() {
                    if let Some(platforms) = fmap.get("platforms").and_then(|v| v.as_array()) {
                        def.platforms = Some(
                            platforms
                                .iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect(),
                        );
                    }
                    if let Some(chans) = fmap.get("channels").and_then(|v| v.as_array()) {
                        def.channels = chans
                            .iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect();
                    }
                    if let Some(deps) = fmap.get("dependencies").and_then(|v| v.as_table()) {
                        for (dep_name, dep_value) in deps {
                            match classify_dep_value(dep_value) {
                                DepKind::Version(spec) => {
                                    def.dependencies.insert(dep_name.clone(), spec);
                                }
                                DepKind::Path(path) => {
                                    def.path_dependencies.insert(dep_name.clone(), path);
                                }
                                DepKind::Source(source) => {
                                    def.source_dependencies.insert(dep_name.clone(), source);
                                }
                                DepKind::Other => {}
                            }
                        }
                    }
                    def.target_dependencies = parse_target_dependencies(fvalue);
                    def.pypi_dependencies = parse_pypi_dependencies(fvalue);
                    // v0.37.0+ (D1): per-feature system-requirements.
                    if let Some(sysreqs) = fmap
                        .get("system-requirements")
                        .or_else(|| fmap.get("system_requirements"))
                        .and_then(|v| v.as_table())
                    {
                        for (k, v) in sysreqs {
                            if let Some(s) = parse_system_requirement_value(k, v) {
                                def.system_requirements.insert(k.clone(), s);
                            }
                        }
                    }
                    def.pypi_options = parse_pypi_options(fvalue);
                }
                out.features.insert(name.clone(), def);
            }
        }

        let feature_names = out.features.keys().cloned().collect::<Vec<_>>();
        extend_profiles_with_feature_subdirs(
            &mut out.target_profiles,
            feature_names
                .iter()
                .filter_map(|name| out.features.get(name)),
        );

        out
    }

    /// Parse a manifest while retaining the declaration order of target
    /// selectors. `toml::Value` uses a sorted map in this crate's current
    /// feature set, while Pixi deliberately gives the last matching
    /// `[target.<selector>]` table overwrite precedence. Production loads and
    /// this module's fixtures therefore recover the order from `toml_edit`.
    fn from_toml_source(source: &str) -> Option<Self> {
        let parsed: toml::Value = toml::from_str(source).ok()?;
        let mut manifest = Self::from_toml(&parsed);
        let document = source.parse::<toml_edit::DocumentMut>().ok()?;

        // `from_toml` has only lexical map order. Remove its feature-derived
        // bare profiles and append them again below in manifest declaration
        // order, matching Pixi's platform-list order.
        let declared_profile_names: BTreeSet<String> = parsed
            .get("workspace")
            .and_then(|workspace| workspace.get("platforms"))
            .and_then(toml::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(parse_workspace_profile)
            .map(|profile| profile.name)
            .collect();
        manifest
            .target_profiles
            .retain(|profile| declared_profile_names.contains(&profile.name));

        if let Some(targets) = document
            .get("target")
            .and_then(toml_edit::Item::as_table_like)
        {
            reorder_target_dependencies(
                &mut manifest.target_dependencies,
                targets.iter().map(|(selector, _)| selector),
            );
        }
        let mut declared_feature_order = Vec::new();
        if let Some(features) = document
            .get("feature")
            .and_then(toml_edit::Item::as_table_like)
        {
            for (feature_name, feature_item) in features.iter() {
                declared_feature_order.push(feature_name.to_string());
                let Some(feature) = manifest.features.get_mut(feature_name) else {
                    continue;
                };
                let Some(targets) = feature_item
                    .get("target")
                    .and_then(toml_edit::Item::as_table_like)
                else {
                    continue;
                };
                reorder_target_dependencies(
                    &mut feature.target_dependencies,
                    targets.iter().map(|(selector, _)| selector),
                );
            }
        }
        extend_profiles_with_feature_subdirs(
            &mut manifest.target_profiles,
            declared_feature_order
                .iter()
                .filter_map(|name| manifest.features.get(name)),
        );
        Some(manifest)
    }

    /// Pixi 0.73's exact named-profile selection for one environment.
    ///
    /// A feature restriction may name either a profile or a bare subdir. If
    /// an exact named profile is referenced, other profiles for that same
    /// subdir are suppressed. All active feature restrictions must accept a
    /// profile; an absent restriction is unrestricted, while `platforms = []`
    /// accepts none.
    pub fn environment_profiles(&self, env_name: &str) -> Result<Vec<&WorkspaceProfile>> {
        let env = self
            .environments
            .get(env_name)
            .ok_or_else(|| anyhow!("workspace environment `{env_name}` does not exist"))?;
        let active_features: Vec<&FeatureDef> = env
            .features
            .iter()
            .filter_map(|name| self.features.get(name))
            .collect();
        let exact_names: BTreeSet<&str> = active_features
            .iter()
            .filter_map(|feature| feature.platforms.as_ref())
            .flatten()
            .map(String::as_str)
            .collect();
        let exact_subdirs: BTreeSet<&str> = self
            .target_profiles
            .iter()
            .filter(|profile| exact_names.contains(profile.name.as_str()))
            .map(|profile| profile.subdir.as_str())
            .collect();

        Ok(self
            .target_profiles
            .iter()
            .filter(|profile| {
                active_features.iter().all(|feature| {
                    feature.platforms.as_ref().is_none_or(|restrictions| {
                        restrictions.iter().any(|restriction| {
                            restriction == &profile.name || restriction == &profile.subdir
                        })
                    })
                })
            })
            .filter(|profile| {
                exact_names.contains(profile.name.as_str())
                    || !exact_subdirs.contains(profile.subdir.as_str())
            })
            .collect())
    }

    /// Pixi's legacy `[system-requirements]` migration is active only for a
    /// subdir-only workspace. In that mode each environment resolves a
    /// synthesized rich profile made from its active features' requirements.
    fn uses_legacy_platform_composition(&self) -> bool {
        let subdir_only = self.target_profiles.iter().all(|profile| {
            profile.name == profile.subdir && profile.declared_virtual_packages.is_empty()
        });
        let has_legacy_requirements = !self.system_requirements.is_empty()
            || self
                .features
                .values()
                .any(|feature| !feature.system_requirements.is_empty());
        subdir_only && has_legacy_requirements
    }

    /// Return the concrete profile Pixi exposes for one base manifest profile
    /// in one environment. The base name remains the provenance key used for
    /// source discovery; this effective name is what target selectors and the
    /// exact CLI envelope see.
    fn effective_profile_for_env(
        &self,
        env_name: &str,
        base_profile: &WorkspaceProfile,
    ) -> WorkspaceProfile {
        let declared_virtual_packages = self
            .contract_for_env_profile(env_name, base_profile)
            .declared_virtual_packages;
        let name = if self.uses_legacy_platform_composition() {
            synthesize_workspace_profile_name(&base_profile.subdir, &declared_virtual_packages)
        } else {
            base_profile.name.clone()
        };
        WorkspaceProfile {
            name,
            subdir: base_profile.subdir.clone(),
            declared_virtual_packages,
        }
    }

    /// Infer the unique complete target contract used by concrete workspace
    /// consumers of `source_dir`. The build RPC exposes only a conda subdir,
    /// so divergent same-subdir consumers are rejected instead of choosing a
    /// declaration by iteration order.
    pub fn resolve_target_for_source(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        requested_subdir: &str,
        envelope: Option<&WorkspaceTargetEnvelope>,
    ) -> Result<Option<ResolvedWorkspaceTarget>> {
        if let Some(envelope) = envelope {
            return self
                .validate_target_envelope(workspace_dir, source_dir, requested_subdir, envelope)
                .map(Some);
        }

        // A declared, active source consumer whose feature intersection has
        // no reachable workspace profile is not an inactive source. Treating
        // it as `None` would silently fall back to host/subdir identity.
        let declared_envs: BTreeSet<String> = self
            .try_discover_outputs_for_source_inner(
                workspace_dir,
                source_dir,
                requested_subdir,
                None,
                false,
            )?
            .into_iter()
            .flat_map(|output| output.envs)
            .collect();
        let unreachable: Vec<String> = declared_envs
            .iter()
            .filter(|env| {
                self.environment_profiles(env)
                    .is_ok_and(|profiles| profiles.is_empty())
            })
            .cloned()
            .collect();
        if !unreachable.is_empty() {
            bail!(
                "retread source `{}` is consumed by workspace environments with no reachable target profile: {}",
                source_dir.display(),
                unreachable.join(",")
            );
        }

        let mut contracts: BTreeMap<WorkspaceTargetContract, (BTreeSet<String>, BTreeSet<String>)> =
            BTreeMap::new();
        for profile in self
            .target_profiles
            .iter()
            .filter(|profile| profile.subdir == requested_subdir)
        {
            let envs = self.try_consuming_envs_for_profile(workspace_dir, source_dir, profile)?;
            for env in envs {
                let entry = contracts
                    .entry(self.contract_for_env_profile(&env, profile))
                    .or_default();
                entry.0.insert(profile.name.clone());
                entry.1.insert(env);
            }
        }

        match contracts.len() {
            0 => Ok(None),
            1 => {
                let (contract, (profiles, environments)) = contracts.into_iter().next().unwrap();
                let target = ResolvedWorkspaceTarget {
                    contract,
                    profiles: profiles.into_iter().collect(),
                    environments: environments.into_iter().collect(),
                };
                self.validate_resolved_profile_views(&target)?;
                Ok(Some(target))
            }
            _ => {
                let detail = contracts
                    .into_iter()
                    .map(|(contract, (profiles, environments))| {
                        format!(
                            "profiles=[{}] environments=[{}] virtual-packages={}",
                            profiles.into_iter().collect::<Vec<_>>().join(","),
                            environments.into_iter().collect::<Vec<_>>().join(","),
                            render_virtual_packages(&contract.declared_virtual_packages),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                bail!(
                    "retread source `{}` has ambiguous workspace target contracts for `{requested_subdir}`: {detail}; run through an exact target envelope",
                    source_dir.display()
                )
            }
        }
    }

    /// Rehydrate profile/environment provenance from an already validated
    /// semantic contract carried by [`crate::pypi::ResolutionTarget`]. This
    /// lets downstream solve/fingerprint helpers stay profile-correct without
    /// trusting process-global envelope state a second time.
    pub fn resolve_source_for_contract(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        contract: &WorkspaceTargetContract,
    ) -> Option<ResolvedWorkspaceTarget> {
        let mut profile_envs: Vec<(&WorkspaceProfile, String)> = Vec::new();
        for profile in self
            .target_profiles
            .iter()
            .filter(|profile| profile.subdir == contract.subdir)
        {
            for env in self
                .try_consuming_envs_for_profile(workspace_dir, source_dir, profile)
                .ok()?
            {
                if self
                    .contract_for_env_profile(&env, profile)
                    .declared_virtual_packages
                    == contract.declared_virtual_packages
                {
                    profile_envs.push((profile, env));
                }
            }
        }
        if profile_envs.is_empty() {
            return None;
        }
        let profiles = profile_envs
            .iter()
            .map(|(profile, _)| profile.name.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let environments = profile_envs
            .into_iter()
            .map(|(_, env)| env)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let target = ResolvedWorkspaceTarget {
            contract: contract.clone(),
            profiles,
            environments,
        };
        self.validate_resolved_profile_views(&target).ok()?;
        Some(target)
    }

    fn validate_target_envelope(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        requested_subdir: &str,
        envelope: &WorkspaceTargetEnvelope,
    ) -> Result<ResolvedWorkspaceTarget> {
        if envelope.schema != 1 {
            bail!(
                "unsupported {WORKSPACE_TARGET_ENV} schema {}; expected 1",
                envelope.schema
            );
        }
        if envelope.profile.subdir != requested_subdir {
            bail!(
                "{WORKSPACE_TARGET_ENV} selected `{}` but the build RPC requested `{requested_subdir}`",
                envelope.profile.subdir
            );
        }
        let selected = self.environment_profiles(&envelope.environment)?;
        let matching_profiles: Vec<(&WorkspaceProfile, WorkspaceProfile)> = selected
            .into_iter()
            .map(|base| {
                let effective = self.effective_profile_for_env(&envelope.environment, base);
                (base, effective)
            })
            .filter(|(_, effective)| {
                effective.name == envelope.profile.name
                    && effective.subdir == envelope.profile.subdir
            })
            .collect();
        let (profile, effective_profile) = match matching_profiles.as_slice() {
            [(profile, effective)] => (*profile, effective),
            [] => {
                bail!(
                    "{WORKSPACE_TARGET_ENV} profile `{}` is not an effective target of workspace environment `{}`",
                    envelope.profile.name,
                    envelope.environment
                )
            }
            _ => {
                bail!(
                    "{WORKSPACE_TARGET_ENV} profile `{}` ambiguously maps to multiple manifest profiles for environment `{}`",
                    envelope.profile.name,
                    envelope.environment
                )
            }
        };

        let envs = self.try_consuming_envs_for_profile(workspace_dir, source_dir, profile)?;
        if !envs.contains(&envelope.environment) {
            bail!(
                "workspace environment `{}` does not consume retread source `{}` on profile `{}`",
                envelope.environment,
                source_dir.display(),
                profile.name
            );
        }

        let profile_declared = parse_virtual_package_lines(&envelope.profile.virtual_packages)
            .context("invalid target envelope virtual_packages")?;
        for (name, value) in &profile_declared {
            if effective_profile.declared_virtual_packages.get(name) != Some(value) {
                bail!(
                    "{WORKSPACE_TARGET_ENV} profile `{}` advertises `{name}={value}` not declared by its effective workspace target ({})",
                    envelope.profile.name,
                    render_virtual_packages(&effective_profile.declared_virtual_packages)
                );
            }
        }
        let detected = parse_virtual_package_lines(&envelope.profile.detected_virtual_packages)
            .context("invalid target envelope detected_virtual_packages")?;
        for (name, value) in &effective_profile.declared_virtual_packages {
            if !pixi_detects_declared_virtual_package(name) {
                continue;
            }
            let preserved = detected.get(name).is_some_and(|detected| {
                detected == value
                    || (name == "archspec" && detected.ends_with(&format!("={value}")))
            });
            if !preserved {
                bail!(
                    "{WORKSPACE_TARGET_ENV} detected contract does not preserve declared `{name}={value}`"
                );
            }
        }
        let declared = effective_profile.declared_virtual_packages.clone();
        validate_detected_target_shape(
            requested_subdir,
            &effective_profile.declared_virtual_packages,
            &detected,
            &envelope.profile.name,
        )?;

        let target = ResolvedWorkspaceTarget {
            contract: WorkspaceTargetContract {
                subdir: profile.subdir.clone(),
                declared_virtual_packages: declared,
                detected_virtual_packages: detected,
            },
            // The complete semantic contract is name-independent and may
            // share artifact identity with aliases. Consumer provenance is
            // not: an exact envelope authorizes only the selected pair, so
            // dependency/emission discovery must not union unrelated aliases.
            profiles: vec![profile.name.clone()],
            environments: vec![envelope.environment.clone()],
        };
        self.validate_resolved_profile_views(&target)?;
        Ok(target)
    }

    fn contract_for_env_profile(
        &self,
        env_name: &str,
        profile: &WorkspaceProfile,
    ) -> WorkspaceTargetContract {
        let mut declared_virtual_packages = profile.declared_virtual_packages.clone();
        for (name, value) in
            self.effective_legacy_system_requirements_for_target(env_name, &profile.subdir)
        {
            let name = match name.as_str() {
                "libc" => "glibc".to_string(),
                "macos" => "macos".to_string(),
                other => normalize_virtual_package_name(other),
            };
            // Rich profile declarations replace deprecated feature-level
            // system requirements, matching Pixi's precedence.
            declared_virtual_packages
                .entry(name)
                .or_insert_with(|| canonical_virtual_package_value(&value));
        }
        WorkspaceTargetContract {
            subdir: profile.subdir.clone(),
            declared_virtual_packages,
            detected_virtual_packages: BTreeMap::new(),
        }
    }

    fn effective_profile_names_for_resolved_env(
        &self,
        env_name: &str,
        target: &ResolvedWorkspaceTarget,
    ) -> Result<Vec<String>> {
        let authorized: BTreeSet<&str> = target.profiles.iter().map(String::as_str).collect();
        let names: Vec<String> = self
            .environment_profiles(env_name)?
            .into_iter()
            .filter(|profile| authorized.contains(profile.name.as_str()))
            .filter(|profile| {
                self.contract_for_env_profile(env_name, profile)
                    .declared_virtual_packages
                    == target.contract.declared_virtual_packages
            })
            .map(|profile| self.effective_profile_for_env(env_name, profile).name)
            .collect();
        if names.is_empty() {
            bail!(
                "workspace environment `{env_name}` has no authorized profile for target contract {}",
                render_virtual_packages(&target.contract.declared_virtual_packages)
            );
        }
        Ok(names)
    }

    fn effective_profile_name_for_resolved_env(
        &self,
        env_name: &str,
        target: &ResolvedWorkspaceTarget,
    ) -> Result<String> {
        let names = self.effective_profile_names_for_resolved_env(env_name, target)?;
        let first = names[0].clone();
        let first_dependencies =
            self.effective_dependencies_for_profile(env_name, &first, &target.contract.subdir);
        let first_paths = self.effective_path_dependencies_with_provenance_for_profile(
            env_name,
            &first,
            &target.contract.subdir,
        )?;
        let first_pypi =
            self.effective_pypi_dependencies_for_profile(env_name, &first, &target.contract.subdir);
        for name in names.iter().skip(1) {
            let dependencies =
                self.effective_dependencies_for_profile(env_name, name, &target.contract.subdir);
            let paths = self.effective_path_dependencies_with_provenance_for_profile(
                env_name,
                name,
                &target.contract.subdir,
            )?;
            let pypi = self.effective_pypi_dependencies_for_profile(
                env_name,
                name,
                &target.contract.subdir,
            );
            if dependencies != first_dependencies || paths != first_paths || pypi != first_pypi {
                bail!(
                    "workspace environment `{env_name}` has same-contract profile aliases with divergent target selector views: `{first}` and `{name}`; use an exact target envelope"
                );
            }
        }
        Ok(first)
    }

    fn validate_resolved_profile_views(&self, target: &ResolvedWorkspaceTarget) -> Result<()> {
        for environment in &target.environments {
            self.effective_profile_name_for_resolved_env(environment, target)?;
        }
        let mut signature: Option<(
            String,
            Vec<(
                String,
                BTreeMap<String, String>,
                BTreeMap<String, String>,
                BTreeMap<String, String>,
                BTreeMap<String, String>,
            )>,
        )> = None;
        for profile_name in &target.profiles {
            let Some(profile) = self
                .target_profiles
                .iter()
                .find(|profile| &profile.name == profile_name)
            else {
                bail!("resolved target names unknown manifest profile `{profile_name}`");
            };
            let effective_name = target
                .environments
                .iter()
                .find(|environment| {
                    self.environment_profiles(environment)
                        .is_ok_and(|profiles| {
                            profiles
                                .iter()
                                .any(|selected| selected.name == profile.name)
                        })
                })
                .map_or_else(
                    || profile.name.clone(),
                    |environment| self.effective_profile_for_env(environment, profile).name,
                );
            let candidate = self.target_selector_signature(&effective_name, &profile.subdir);
            if let Some((first_name, first)) = &signature {
                if first != &candidate {
                    bail!(
                        "same-contract profile aliases have divergent target selector views: `{first_name}` and `{effective_name}`; use an exact target envelope"
                    );
                }
            } else {
                signature = Some((effective_name, candidate));
            }
        }
        Ok(())
    }

    fn target_selector_signature(
        &self,
        target_name: &str,
        target_subdir: &str,
    ) -> Vec<(
        String,
        BTreeMap<String, String>,
        BTreeMap<String, String>,
        BTreeMap<String, String>,
        BTreeMap<String, String>,
    )> {
        let mut signature = vec![(
            DEFAULT_FEATURE.to_string(),
            dependency_overlay_for_target(
                &self.dependencies,
                &self.target_dependencies,
                target_name,
                target_subdir,
            ),
            path_dependency_overlay_for_target(
                &self.path_dependencies,
                &self.target_dependencies,
                target_name,
                target_subdir,
            ),
            source_dependency_overlay_for_target(
                &self.source_dependencies,
                &self.target_dependencies,
                target_name,
                target_subdir,
            ),
            pypi_dependency_overlay_for_target(
                &self.pypi_dependencies,
                &self.target_dependencies,
                target_name,
                target_subdir,
            ),
        )];
        signature.extend(self.features.iter().map(|(feature_name, feature)| {
            (
                feature_name.clone(),
                dependency_overlay_for_target(
                    &feature.dependencies,
                    &feature.target_dependencies,
                    target_name,
                    target_subdir,
                ),
                path_dependency_overlay_for_target(
                    &feature.path_dependencies,
                    &feature.target_dependencies,
                    target_name,
                    target_subdir,
                ),
                source_dependency_overlay_for_target(
                    &feature.source_dependencies,
                    &feature.target_dependencies,
                    target_name,
                    target_subdir,
                ),
                pypi_dependency_overlay_for_target(
                    &feature.pypi_dependencies,
                    &feature.target_dependencies,
                    target_name,
                    target_subdir,
                ),
            )
        }));
        signature
    }

    pub fn effective_dependencies_for_resolved_env(
        &self,
        env_name: &str,
        target: &ResolvedWorkspaceTarget,
    ) -> Result<BTreeMap<String, String>> {
        let profile_name = self.effective_profile_name_for_resolved_env(env_name, target)?;
        Ok(self.effective_dependencies_for_profile(
            env_name,
            &profile_name,
            &target.contract.subdir,
        ))
    }

    pub fn effective_path_dependencies_for_resolved_env(
        &self,
        env_name: &str,
        target: &ResolvedWorkspaceTarget,
    ) -> Result<BTreeMap<String, String>> {
        let profile_name = self.effective_profile_name_for_resolved_env(env_name, target)?;
        let paths = self.effective_path_dependencies_with_provenance_for_profile(
            env_name,
            &profile_name,
            &target.contract.subdir,
        )?;
        let mut out = BTreeMap::new();
        for (name, declarations) in paths {
            out.insert(name, declarations[0].0.clone());
        }
        Ok(out)
    }

    /// Compute the effective conda deps an environment will solve
    /// against. Order: top-level [dependencies] (unless
    /// no-default-feature), then each active feature in declaration
    /// order. Later entries OVERRIDE earlier ones with the same name
    /// (matches pixi's precedence: feature deps override workspace
    /// defaults).
    pub fn effective_dependencies(&self, env_name: &str) -> BTreeMap<String, String> {
        self.effective_dependencies_for_target(env_name, crate::glibc::current_pixi_platform())
    }

    /// Exact-target form of [`Self::effective_dependencies`]. Target-specific
    /// dependency tables overlay their ordinary default/feature table at each
    /// precedence layer.
    pub fn effective_dependencies_for_target(
        &self,
        env_name: &str,
        target_subdir: &str,
    ) -> BTreeMap<String, String> {
        self.effective_dependencies_for_profile(env_name, target_subdir, target_subdir)
    }

    fn effective_dependencies_for_profile(
        &self,
        env_name: &str,
        target_name: &str,
        target_subdir: &str,
    ) -> BTreeMap<String, String> {
        let Some(env) = self.environments.get(env_name) else {
            return BTreeMap::new();
        };
        let mut out = BTreeMap::new();
        if !env.no_default_feature {
            overlay_effective_version_dependency_layer(
                &mut out,
                &self.dependencies,
                &self.path_dependencies,
                &self.target_dependencies,
                target_name,
                target_subdir,
            );
        }
        for feat_name in &env.features {
            let Some(feat) = self.features.get(feat_name) else {
                continue;
            };
            overlay_effective_version_dependency_layer(
                &mut out,
                &feat.dependencies,
                &feat.path_dependencies,
                &feat.target_dependencies,
                target_name,
                target_subdir,
            );
        }
        out
    }

    /// Compute the path packages activated by one concrete environment.
    ///
    /// This is the path-source counterpart of
    /// [`Self::effective_dependencies_for_target`]. Keeping the same overlay
    /// order is load-bearing for sibling-pack composition: a target-local
    /// registry dependency replaces an inherited path source, while a
    /// target-local path source replaces an inherited registry dependency.
    pub fn effective_path_dependencies_for_target(
        &self,
        env_name: &str,
        target_subdir: &str,
    ) -> BTreeMap<String, String> {
        self.effective_path_dependencies_with_provenance_for_profile(
            env_name,
            target_subdir,
            target_subdir,
        )
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(name, paths)| {
            let distinct: BTreeSet<&str> = paths.iter().map(|(path, _)| path.as_str()).collect();
            (distinct.len() == 1).then(|| (name, paths[0].0.clone()))
        })
        .collect()
    }

    /// Exact final path-source view plus the feature whose dependency layer
    /// supplied it. Cross-kind replacement is handled between feature layers:
    /// a later registry requirement removes an earlier path source, while a
    /// later path source replaces either representation.
    fn effective_path_dependencies_with_provenance_for_profile(
        &self,
        env_name: &str,
        target_name: &str,
        target_subdir: &str,
    ) -> Result<BTreeMap<String, Vec<(String, String)>>> {
        let Some(env) = self.environments.get(env_name) else {
            return Ok(BTreeMap::new());
        };
        let mut out = BTreeMap::new();
        let mut active_sources = BTreeMap::new();
        if !env.no_default_feature {
            overlay_effective_path_dependency_layer(
                &mut out,
                &mut active_sources,
                &self.dependencies,
                &self.path_dependencies,
                &self.source_dependencies,
                &self.target_dependencies,
                target_name,
                target_subdir,
                DEFAULT_FEATURE,
            )?;
        }
        for feat_name in &env.features {
            let Some(feat) = self.features.get(feat_name) else {
                continue;
            };
            overlay_effective_path_dependency_layer(
                &mut out,
                &mut active_sources,
                &feat.dependencies,
                &feat.path_dependencies,
                &feat.source_dependencies,
                &feat.target_dependencies,
                target_name,
                target_subdir,
                feat_name,
            )?;
        }
        Ok(out)
    }

    /// Compute the effective PyPI declarations for an environment. Pixi
    /// merges dependency maps across active features, so every distinct
    /// requirement survives. Target tables overwrite the ordinary map only
    /// within their own feature before the active feature maps are merged.
    pub fn effective_pypi_dependencies(&self, env_name: &str) -> BTreeMap<String, Vec<String>> {
        self.effective_pypi_dependencies_for_target(env_name, crate::glibc::current_pixi_platform())
    }

    pub fn effective_pypi_dependencies_for_target(
        &self,
        env_name: &str,
        target_subdir: &str,
    ) -> BTreeMap<String, Vec<String>> {
        self.effective_pypi_dependencies_for_profile(env_name, target_subdir, target_subdir)
    }

    pub fn effective_pypi_dependencies_for_resolved_env(
        &self,
        env_name: &str,
        target: &ResolvedWorkspaceTarget,
    ) -> Result<BTreeMap<String, Vec<String>>> {
        let profile_name = self.effective_profile_name_for_resolved_env(env_name, target)?;
        Ok(self.effective_pypi_dependencies_for_profile(
            env_name,
            &profile_name,
            &target.contract.subdir,
        ))
    }

    fn effective_pypi_dependencies_for_profile(
        &self,
        env_name: &str,
        target_name: &str,
        target_subdir: &str,
    ) -> BTreeMap<String, Vec<String>> {
        let Some(env) = self.environments.get(env_name) else {
            return BTreeMap::new();
        };
        let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut merge_layer = |layer: BTreeMap<String, String>| {
            for (name, spec) in layer {
                let requirements = out.entry(name).or_default();
                if !requirements.contains(&spec) {
                    requirements.push(spec);
                }
            }
        };
        if !env.no_default_feature {
            merge_layer(pypi_dependency_overlay_for_target(
                &self.pypi_dependencies,
                &self.target_dependencies,
                target_name,
                target_subdir,
            ));
        }
        for feat_name in &env.features {
            let Some(feat) = self.features.get(feat_name) else {
                continue;
            };
            merge_layer(pypi_dependency_overlay_for_target(
                &feat.pypi_dependencies,
                &feat.target_dependencies,
                target_name,
                target_subdir,
            ));
        }
        out
    }

    /// Every PyPI index the workspace declares anywhere: top-level
    /// `[pypi-options]` first, then each feature's in name order,
    /// deduped preserving first occurrence. The cascade's PyPI
    /// fallback is a harmless try-in-order chain, so a flat union is
    /// the right shape -- consulting an env-inactive feature's index
    /// can only find a wheel, never mis-route a dep.
    pub fn all_pypi_index_urls(&self) -> Vec<String> {
        let mut out = Vec::new();
        let all = self.pypi_options.declared_index_urls().chain(
            self.features
                .values()
                .flat_map(|f| f.pypi_options.declared_index_urls()),
        );
        for url in all {
            push_unique_index_url(&mut out, url.clone());
        }
        out
    }

    /// Complete workspace-wide PyPI resolution chain.
    ///
    /// Top-level extras retain declaration priority. They are followed by
    /// the explicitly configured primary index, or public PyPI when pixi's
    /// default has not been suppressed. Feature candidates follow in stable
    /// feature-name order. Equivalent trailing-slash spellings are deduped.
    pub fn resolution_pypi_index_urls(&self) -> Vec<String> {
        let mut out = Vec::new();
        for url in &self.pypi_options.extra_index_urls {
            push_unique_index_url(&mut out, url.clone());
        }
        push_unique_index_url(
            &mut out,
            self.pypi_options
                .index_url
                .clone()
                .unwrap_or_else(|| DEFAULT_PYPI_INDEX.to_string()),
        );
        for feature in self.features.values() {
            for url in feature.pypi_options.declared_index_urls() {
                push_unique_index_url(&mut out, url.clone());
            }
        }
        out
    }

    /// Pack-scoped canonical fingerprint of every resolution-affecting workspace
    /// solve input for the given source package, for the courier lock's
    /// `inputs_hash` (grizzly H1).
    ///
    /// Scoped to the envs that reference `source_dir` via
    /// `discover_outputs_for_source`: only channels, conda/PyPI deps,
    /// system-requirements, and pypi-index-urls from those envs (computed via
    /// the per-env `effective_*` getters) are folded. This eliminates over-coupling to
    /// unrelated envs in the same workspace -- pixi solves each env
    /// independently, so non-referencing envs cannot affect THIS pack's
    /// resolution and should not invalidate its cached lock.
    ///
    /// Returns empty String when no env references the pack (symmetric with
    /// `courier_channel_set`'s empty-Vec return; the caller guards
    /// `if !workspace_fp.is_empty()`). Both producer (`build_one`) and
    /// replayer (`conda_outputs`) call this with the same
    /// `(workspace_dir, source_dir)` pair, so the fingerprints always agree.
    pub fn solve_fingerprint(&self, workspace_dir: &Path, source_dir: &Path) -> String {
        self.solve_fingerprint_for_target(
            workspace_dir,
            source_dir,
            crate::glibc::current_pixi_platform(),
        )
    }

    /// Exact-target form of [`Self::solve_fingerprint`]. This keeps rich
    /// platform system requirements from the build host out of a foreign
    /// target's resolution inputs.
    pub fn solve_fingerprint_for_target(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        target_subdir: &str,
    ) -> String {
        let outputs =
            self.discover_outputs_for_source_for_target(workspace_dir, source_dir, target_subdir);
        if outputs.is_empty() {
            return String::new();
        }
        // Union all env names across every discovered output (stable sort via
        // BTreeSet so parts order is deterministic).
        let mut all_envs: BTreeSet<String> = BTreeSet::new();
        for output in &outputs {
            for env in &output.envs {
                all_envs.insert(env.clone());
            }
        }
        let env_vec: Vec<String> = all_envs.into_iter().collect();

        self.solve_fingerprint_for_envs(&env_vec, target_subdir, None)
    }

    pub fn solve_fingerprint_for_resolved_target(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        target: &ResolvedWorkspaceTarget,
    ) -> String {
        let envs = self
            .precise_consuming_envs_for_resolved_target(workspace_dir, source_dir, target)
            .unwrap_or_default();
        if envs.is_empty() {
            return String::new();
        }
        self.solve_fingerprint_for_envs(&envs, &target.contract.subdir, Some(target))
    }

    fn solve_fingerprint_for_envs(
        &self,
        envs: &[String],
        target_subdir: &str,
        target: Option<&ResolvedWorkspaceTarget>,
    ) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(p) = &self.channel_priority {
            parts.push(format!("ws-channel-priority:{p}"));
        }
        for env in envs {
            for c in self.effective_channels(env) {
                parts.push(format!("scoped-env:{env}:channel:{c}"));
            }
            let dependencies = target.map_or_else(
                || self.effective_dependencies_for_target(env, target_subdir),
                |target| {
                    self.effective_dependencies_for_resolved_env(env, target)
                        .unwrap_or_default()
                },
            );
            for (k, v) in dependencies {
                parts.push(format!("scoped-env:{env}:dep:{k}={v}"));
            }
            let pypi_dependencies = target.map_or_else(
                || self.effective_pypi_dependencies_for_target(env, target_subdir),
                |target| {
                    self.effective_pypi_dependencies_for_resolved_env(env, target)
                        .unwrap_or_default()
                },
            );
            for (k, requirements) in pypi_dependencies {
                for requirement in requirements {
                    parts.push(format!("scoped-env:{env}:pypi-dep:{k}={requirement}"));
                }
            }
            let system_requirements = target.map_or_else(
                || self.effective_system_requirements_for_target(env, target_subdir),
                |target| self.effective_system_requirements_for_contract(env, &target.contract),
            );
            for (k, v) in system_requirements {
                parts.push(format!("scoped-env:{env}:sysreq:{k}={v}"));
            }
            for u in self.effective_pypi_index_urls(env) {
                parts.push(format!("scoped-env:{env}:pypi-index:{u}"));
            }
        }
        parts.join("\n")
    }

    pub fn effective_system_requirements(&self, env_name: &str) -> BTreeMap<String, String> {
        self.effective_system_requirements_for_target(
            env_name,
            crate::glibc::current_pixi_platform(),
        )
    }

    /// Return an environment's system requirements for the requested Pixi
    /// platform. Legacy feature-scoped requirements apply to every target;
    /// rich `[workspace].platforms` values are overlaid only from the exact
    /// target entry.
    pub fn effective_system_requirements_for_target(
        &self,
        env_name: &str,
        target_subdir: &str,
    ) -> BTreeMap<String, String> {
        if !self.environments.contains_key(env_name) {
            return BTreeMap::new();
        }
        let mut out = self.effective_legacy_system_requirements_for_target(env_name, target_subdir);
        // pixi 0.71+ rich `[workspace].platforms` declarations replace the
        // deprecated `[system-requirements]` table. They are workspace-wide
        // (per platform, not per env/feature) and, matching
        // `declared_glibc`'s precedence, win over any legacy declaration.
        if let Some(glibc) = self.platform_glibc.get(target_subdir) {
            out.insert("libc".to_string(), glibc.clone());
        }
        if let Some(cuda) = self.platform_cuda.get(target_subdir) {
            out.insert("cuda".to_string(), cuda.clone());
        }
        out
    }

    /// Exact rich-profile form of [`Self::effective_system_requirements`].
    /// The complete target contract, including Pixi-detected values from an
    /// envelope when present, wins over deprecated feature declarations.
    pub fn effective_system_requirements_for_contract(
        &self,
        env_name: &str,
        contract: &WorkspaceTargetContract,
    ) -> BTreeMap<String, String> {
        if !self.environments.contains_key(env_name) {
            return BTreeMap::new();
        }
        let mut out =
            self.effective_legacy_system_requirements_for_target(env_name, &contract.subdir);
        out.extend(contract.system_requirements());
        out
    }

    fn effective_legacy_system_requirements_for_target(
        &self,
        env_name: &str,
        target_subdir: &str,
    ) -> BTreeMap<String, String> {
        let mut requirements = self.effective_legacy_system_requirements(env_name);
        requirements
            .retain(|name, _| legacy_system_requirement_applies_to_subdir(name, target_subdir));
        requirements
    }

    fn effective_legacy_system_requirements(&self, env_name: &str) -> BTreeMap<String, String> {
        let Some(env) = self.environments.get(env_name) else {
            return BTreeMap::new();
        };
        let mut out = BTreeMap::new();
        if !env.no_default_feature {
            for (k, v) in &self.system_requirements {
                merge_legacy_system_requirement(&mut out, k, v);
            }
        }
        for feat_name in &env.features {
            let Some(feat) = self.features.get(feat_name) else {
                continue;
            };
            for (k, v) in &feat.system_requirements {
                merge_legacy_system_requirement(&mut out, k, v);
            }
        }
        out
    }

    /// Declared glibc floor for installer manylinux relaxation. pixi 0.71+
    /// rich `[workspace].platforms` entries win over legacy
    /// `[system-requirements]`; when `env_name` is unknown, union the legacy
    /// top-level and feature declarations and take the max.
    pub fn declared_glibc(&self, env_name: Option<&str>) -> Option<(u32, u32)> {
        self.declared_glibc_for_target(crate::glibc::current_pixi_platform(), env_name)
    }

    /// Exact-target form of [`Self::declared_glibc`].
    pub fn declared_glibc_for_target(
        &self,
        target_subdir: &str,
        env_name: Option<&str>,
    ) -> Option<(u32, u32)> {
        if let Some(v) = self
            .platform_glibc
            .get(target_subdir)
            .and_then(|s| crate::glibc::parse_glibc_version(s))
        {
            return Some(v);
        }

        if let Some(env_name) = env_name {
            return self
                .effective_legacy_system_requirements_for_target(env_name, target_subdir)
                .get("libc")
                .and_then(|s| crate::glibc::parse_glibc_version(s));
        }

        if !target_subdir.starts_with("linux-") {
            return None;
        }

        let mut versions = Vec::new();
        if let Some(v) = self
            .system_requirements
            .get("libc")
            .and_then(|s| crate::glibc::parse_glibc_version(s))
        {
            versions.push(v);
        }
        for feature in self.features.values() {
            if let Some(v) = feature
                .system_requirements
                .get("libc")
                .and_then(|s| crate::glibc::parse_glibc_version(s))
            {
                versions.push(v);
            }
        }
        versions.into_iter().max()
    }

    /// Effective PyPI index URLs for one env: top-level (unless
    /// no-default-feature) then each active feature's, deduped first-seen.
    /// Mirrors `effective_channels`'s default-inheritance + no-default-feature
    /// semantics exactly.
    pub fn effective_pypi_index_urls(&self, env_name: &str) -> Vec<String> {
        let Some(env) = self.environments.get(env_name) else {
            return Vec::new();
        };
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        if !env.no_default_feature {
            for u in self.pypi_options.declared_index_urls() {
                let normalized = u.trim_end_matches('/').to_string();
                if seen.insert(normalized) {
                    out.push(u.clone());
                }
            }
        }
        for feat_name in &env.features {
            let Some(feat) = self.features.get(feat_name) else {
                continue;
            };
            for u in feat.pypi_options.declared_index_urls() {
                let normalized = u.trim_end_matches('/').to_string();
                if seen.insert(normalized) {
                    out.push(u.clone());
                }
            }
        }
        out
    }

    /// Compute the effective channels an environment will solve
    /// against. Workspace top-level channels first (unless
    /// no-default-feature), then each active feature's channels
    /// appended in declaration order. Duplicates dropped while
    /// preserving first-seen order.
    pub fn effective_channels(&self, env_name: &str) -> Vec<String> {
        let Some(env) = self.environments.get(env_name) else {
            return Vec::new();
        };
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        if !env.no_default_feature {
            for c in &self.channels {
                if seen.insert(c.clone()) {
                    out.push(c.clone());
                }
            }
        }
        for feat_name in &env.features {
            let Some(feat) = self.features.get(feat_name) else {
                continue;
            };
            for c in &feat.channels {
                if seen.insert(c.clone()) {
                    out.push(c.clone());
                }
            }
        }
        out
    }
}

/// What kind of pixi.toml dep entry this is. Git/URL entries are retained as
/// non-path sources: Retread does not build them, but they replace inherited
/// paths within one DependencyMap and conflict with path sources contributed
/// by another active feature.
enum DepKind {
    /// A version-pinned conda dep — used for transitive extraction.
    Version(String),
    /// A path-form dep pointing at a local source package — used by
    /// output autodiscovery to match against the source dir retread
    /// is building.
    Path(String),
    /// A non-path conda source (`git = ...` / `url = ...`). The rendered
    /// declaration is retained so diagnostics can identify both sources.
    Source(String),
    Other,
}

fn classify_dep_value(value: &toml::Value) -> DepKind {
    match value {
        toml::Value::String(s) => DepKind::Version(s.clone()),
        toml::Value::Table(t) => {
            if let Some(path) = t.get("path").and_then(|v| v.as_str()) {
                return DepKind::Path(path.to_string());
            }
            if t.contains_key("git") || t.contains_key("url") {
                return DepKind::Source(value.to_string());
            }
            match t.get("version").and_then(|v| v.as_str()) {
                Some(v) => DepKind::Version(v.to_string()),
                None => DepKind::Other,
            }
        }
        _ => DepKind::Other,
    }
}

fn parse_target_dependencies(root: &toml::Value) -> Vec<(String, TargetDependencyDef)> {
    let mut out = Vec::new();
    let Some(targets) = root.get("target").and_then(toml::Value::as_table) else {
        return out;
    };
    for (target_subdir, target_value) in targets {
        let mut target = TargetDependencyDef::default();
        if let Some(dependencies) = target_value
            .get("dependencies")
            .and_then(toml::Value::as_table)
        {
            for (name, value) in dependencies {
                match classify_dep_value(value) {
                    DepKind::Version(spec) => {
                        target.dependencies.insert(name.clone(), spec);
                    }
                    DepKind::Path(path) => {
                        target.path_dependencies.insert(name.clone(), path);
                    }
                    DepKind::Source(source) => {
                        target.source_dependencies.insert(name.clone(), source);
                    }
                    DepKind::Other => {}
                }
            }
        }
        target.pypi_dependencies = parse_pypi_dependencies(target_value);
        if !target.dependencies.is_empty()
            || !target.path_dependencies.is_empty()
            || !target.source_dependencies.is_empty()
            || !target.pypi_dependencies.is_empty()
        {
            out.push((target_subdir.clone(), target));
        }
    }
    out
}

fn reorder_target_dependencies<'a>(
    targets: &mut Vec<(String, TargetDependencyDef)>,
    declaration_order: impl Iterator<Item = &'a str>,
) {
    let mut by_selector: BTreeMap<String, TargetDependencyDef> = targets.drain(..).collect();
    for selector in declaration_order {
        if let Some(target) = by_selector.remove(selector) {
            targets.push((selector.to_string(), target));
        }
    }
    // Malformed or unsupported entries are parsed best-effort. Keep any
    // survivors deterministic after the declaration-ordered entries.
    targets.extend(by_selector);
}

fn dependency_overlay_for_target(
    base: &BTreeMap<String, String>,
    targets: &[(String, TargetDependencyDef)],
    target_name: &str,
    target_subdir: &str,
) -> BTreeMap<String, String> {
    let mut out = base.clone();
    for target in matching_target_dependencies(targets, target_name, target_subdir) {
        // A target table overwrites the ordinary dependency regardless of
        // representation. A target-local path source therefore removes an
        // inherited version matchspec before exact-target versions are
        // overlaid below.
        for name in target
            .path_dependencies
            .keys()
            .chain(target.source_dependencies.keys())
        {
            out.remove(name);
        }
        out.extend(target.dependencies.clone());
    }
    out
}

fn path_dependency_overlay_for_target(
    base: &BTreeMap<String, String>,
    targets: &[(String, TargetDependencyDef)],
    target_name: &str,
    target_subdir: &str,
) -> BTreeMap<String, String> {
    let mut out = base.clone();
    for target in matching_target_dependencies(targets, target_name, target_subdir) {
        // The inverse cross-kind overwrite: a target-local registry
        // matchspec replaces an inherited path source with the same name.
        for name in target
            .dependencies
            .keys()
            .chain(target.source_dependencies.keys())
        {
            out.remove(name);
        }
        out.extend(target.path_dependencies.clone());
    }
    out
}

fn source_dependency_overlay_for_target(
    base: &BTreeMap<String, String>,
    targets: &[(String, TargetDependencyDef)],
    target_name: &str,
    target_subdir: &str,
) -> BTreeMap<String, String> {
    let mut out = base.clone();
    for target in matching_target_dependencies(targets, target_name, target_subdir) {
        // A target-local registry or path declaration replaces an inherited
        // git/URL source within this DependencyMap. Conversely, a target-local
        // git/URL source becomes the effective non-path tombstone.
        for name in target
            .dependencies
            .keys()
            .chain(target.path_dependencies.keys())
        {
            out.remove(name);
        }
        out.extend(target.source_dependencies.clone());
    }
    out
}

fn pypi_dependency_overlay_for_target(
    base: &BTreeMap<String, String>,
    targets: &[(String, TargetDependencyDef)],
    target_name: &str,
    target_subdir: &str,
) -> BTreeMap<String, String> {
    let mut out = base.clone();
    for target in matching_target_dependencies(targets, target_name, target_subdir) {
        out.extend(target.pypi_dependencies.clone());
    }
    out
}

fn overlay_effective_path_dependency_layer(
    out: &mut BTreeMap<String, Vec<(String, String)>>,
    active_sources: &mut BTreeMap<String, Vec<(String, String)>>,
    _versions: &BTreeMap<String, String>,
    paths: &BTreeMap<String, String>,
    sources: &BTreeMap<String, String>,
    targets: &[(String, TargetDependencyDef)],
    target_name: &str,
    target_subdir: &str,
    provenance: &str,
) -> Result<()> {
    // Target tables overwrite only within this feature. Across active
    // features Pixi merges DependencyMaps, so a registry constraint in one
    // feature must not erase a path source supplied by another.
    let effective_paths =
        path_dependency_overlay_for_target(paths, targets, target_name, target_subdir);
    let effective_sources =
        source_dependency_overlay_for_target(sources, targets, target_name, target_subdir);
    for name in effective_paths.keys() {
        if let Some(conflicting) = active_sources.get(name) {
            let sources = conflicting
                .iter()
                .map(|(source, feature)| format!("{feature}:{source}"))
                .collect::<Vec<_>>()
                .join(",");
            bail!(
                "active dependency features select conflicting path and git/URL sources for `{name}` on profile `{target_name}`: {provenance}:path conflicts with {sources}"
            );
        }
    }
    for (name, source) in &effective_sources {
        if let Some(conflicting) = out.get(name) {
            let paths = conflicting
                .iter()
                .map(|(path, feature)| format!("{feature}:{path}"))
                .collect::<Vec<_>>()
                .join(",");
            bail!(
                "active dependency features select conflicting path and git/URL sources for `{name}` on profile `{target_name}`: {provenance}:{source} conflicts with {paths}"
            );
        }
    }
    for (name, path) in effective_paths {
        out.entry(name)
            .or_default()
            .push((path, provenance.to_string()));
    }
    for (name, source) in effective_sources {
        active_sources
            .entry(name)
            .or_default()
            .push((source, provenance.to_string()));
    }
    Ok(())
}

fn overlay_effective_version_dependency_layer(
    out: &mut BTreeMap<String, String>,
    versions: &BTreeMap<String, String>,
    _paths: &BTreeMap<String, String>,
    targets: &[(String, TargetDependencyDef)],
    target_name: &str,
    target_subdir: &str,
) {
    let effective_versions =
        dependency_overlay_for_target(versions, targets, target_name, target_subdir);
    for (name, spec) in effective_versions {
        match out.entry(name) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(spec);
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                merge_conda_version_spec(entry.get_mut(), &spec);
            }
        }
    }
}

fn merge_conda_version_spec(existing: &mut String, incoming: &str) {
    if existing == incoming || incoming.trim().is_empty() || incoming.trim() == "*" {
        return;
    }
    if existing.trim().is_empty() || existing.trim() == "*" {
        *existing = incoming.to_string();
        return;
    }
    *existing = format!("{existing},{incoming}");
}

/// Pixi target selectors are cumulative and declaration ordered. Exact named
/// rich profiles and `*` globs match the workspace profile name; conda
/// subdirs and OS-family selectors match the underlying subdir. Later
/// matching tables overwrite earlier ones within one feature.
fn matching_target_dependencies<'a>(
    targets: &'a [(String, TargetDependencyDef)],
    target_name: &str,
    target_subdir: &str,
) -> Vec<&'a TargetDependencyDef> {
    targets
        .iter()
        .filter(|(selector, _)| target_selector_matches(selector, target_name, target_subdir))
        .map(|(_, target)| target)
        .collect()
}

fn target_selector_matches(selector: &str, target_name: &str, target_subdir: &str) -> bool {
    match selector {
        // Pixi delegates this selector to `Platform::is_unix()`. Besides
        // Linux and macOS, Pixi 0.73 classifies every FreeBSD subdir and the
        // Emscripten target as Unix; WASI and z/OS are deliberately excluded.
        "unix" => {
            target_subdir.starts_with("linux-")
                || target_subdir.starts_with("osx-")
                || target_subdir.starts_with("freebsd-")
                || target_subdir == "emscripten-wasm32"
        }
        "linux" => target_subdir.starts_with("linux-"),
        "osx" | "macos" => target_subdir.starts_with("osx-"),
        "win" => target_subdir.starts_with("win-"),
        _ if selector == target_subdir || selector == target_name => true,
        _ if selector.contains('*') => {
            star_glob_matches(selector.as_bytes(), target_name.as_bytes())
        }
        _ => false,
    }
}

/// Pixi platform globs admit `*` as their only wildcard and match the full,
/// case-sensitive profile name.
fn star_glob_matches(pattern: &[u8], value: &[u8]) -> bool {
    let (mut pattern_index, mut value_index) = (0, 0);
    let (mut star_index, mut star_value_index) = (None, 0);
    while value_index < value.len() {
        if pattern_index < pattern.len() && pattern[pattern_index] == value[value_index] {
            pattern_index += 1;
            value_index += 1;
        } else if pattern_index < pattern.len() && pattern[pattern_index] == b'*' {
            star_index = Some(pattern_index);
            pattern_index += 1;
            star_value_index = value_index;
        } else if let Some(star) = star_index {
            pattern_index = star + 1;
            star_value_index += 1;
            value_index = star_value_index;
        } else {
            return false;
        }
    }
    pattern[pattern_index..].iter().all(|byte| *byte == b'*')
}

/// One output retread should emit, discovered from the workspace's
/// path-deps referencing this source package.
#[derive(Debug, Clone)]
pub struct DiscoveredOutput {
    /// The dep name as declared in the workspace pixi.toml. This is
    /// what pixi expects retread to emit (e.g. "isaac-pack-physx").
    pub name: String,
    /// Features that declare this path-dep. Usually exactly one.
    pub declaring_features: Vec<String>,
    /// Envs that include any of the declaring features, in
    /// stable-sorted order. retread unions transitive constraints
    /// across these envs when building this output's emission.
    pub envs: Vec<String>,
}

impl WorkspaceManifest {
    /// Walk the manifest looking for path-deps whose `path` resolves
    /// (joined to `workspace_dir`, normalized) to `source_dir`. Returns
    /// one `DiscoveredOutput` per unique dep NAME found. The
    /// `default` feature is the top-level `[dependencies]` block,
    /// inherited by every env that doesn't set `no-default-feature`.
    ///
    /// If nothing references the source package, returns an empty
    /// Vec; the caller falls back to single-default-output behavior
    /// (named after `[package].name`).
    pub fn discover_outputs_for_source(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
    ) -> Vec<DiscoveredOutput> {
        self.discover_outputs_for_source_for_target(
            workspace_dir,
            source_dir,
            crate::glibc::current_pixi_platform(),
        )
    }

    /// Exact-target form of [`Self::discover_outputs_for_source`]. Path
    /// dependencies from the matching target table participate alongside the
    /// ordinary dependency tables; declarations for other targets do not.
    pub fn discover_outputs_for_source_for_target(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        target_subdir: &str,
    ) -> Vec<DiscoveredOutput> {
        self.try_discover_outputs_for_source_inner(
            workspace_dir,
            source_dir,
            target_subdir,
            None,
            true,
        )
        .unwrap_or_default()
    }

    /// Profile-aware discovery used by exact target inference. Consumers are
    /// retained only when Pixi would solve that environment for this named
    /// profile.
    fn discover_outputs_for_source_for_profile(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        profile: &WorkspaceProfile,
    ) -> Vec<DiscoveredOutput> {
        self.try_discover_outputs_for_source_inner(
            workspace_dir,
            source_dir,
            &profile.subdir,
            Some(profile),
            true,
        )
        .unwrap_or_default()
    }

    fn try_consuming_envs_for_profile(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        profile: &WorkspaceProfile,
    ) -> Result<BTreeSet<String>> {
        Ok(self
            .try_discover_outputs_for_source_inner(
                workspace_dir,
                source_dir,
                &profile.subdir,
                Some(profile),
                true,
            )?
            .into_iter()
            .flat_map(|output| output.envs)
            .collect())
    }

    /// Discover outputs for all profile aliases represented by one resolved
    /// semantic target. This is the exact counterpart callers should use
    /// after [`Self::resolve_target_for_source`].
    pub fn discover_outputs_for_source_for_resolved_target(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        target: &ResolvedWorkspaceTarget,
    ) -> Vec<DiscoveredOutput> {
        let mut merged: BTreeMap<String, (BTreeSet<String>, BTreeSet<String>)> = BTreeMap::new();
        let allowed_environments: BTreeSet<&str> =
            target.environments.iter().map(String::as_str).collect();
        for profile_name in &target.profiles {
            let Some(profile) = self
                .target_profiles
                .iter()
                .find(|profile| &profile.name == profile_name)
            else {
                continue;
            };
            for mut output in
                self.discover_outputs_for_source_for_profile(workspace_dir, source_dir, profile)
            {
                output
                    .envs
                    .retain(|env| allowed_environments.contains(env.as_str()));
                if output.envs.is_empty() {
                    continue;
                }
                let entry = merged.entry(output.name).or_default();
                entry.0.extend(output.declaring_features);
                entry.1.extend(output.envs);
            }
        }
        merged
            .into_iter()
            .map(|(name, (declaring_features, envs))| DiscoveredOutput {
                name,
                declaring_features: declaring_features.into_iter().collect(),
                envs: envs.into_iter().collect(),
            })
            .collect()
    }

    fn try_discover_outputs_for_source_inner(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        target_subdir: &str,
        base_profile: Option<&WorkspaceProfile>,
        filter_profiles: bool,
    ) -> Result<Vec<DiscoveredOutput>> {
        // Resolve source_dir for comparison. Failing canonicalize is
        // tolerable -- fall back to the raw path.
        let source_canon = canonical_or_self(source_dir);
        let mut matches: BTreeMap<String, (BTreeSet<String>, BTreeSet<String>)> = BTreeMap::new();

        // Evaluate every concrete environment after all of its dependency
        // layers have been overlaid. Looking only at raw declaring features
        // creates false consumers when a later feature or target selector
        // replaces the source path with a registry requirement or another
        // path under the same package name.
        for env_name in self.environments.keys() {
            let selected_profiles = self.environment_profiles(env_name)?;
            let candidate_profiles: Vec<&WorkspaceProfile> = match base_profile {
                Some(profile)
                    if !filter_profiles
                        || selected_profiles
                            .iter()
                            .any(|selected| selected.name == profile.name) =>
                {
                    vec![profile]
                }
                Some(_) => Vec::new(),
                None if self.target_profiles.is_empty() => Vec::new(),
                None if filter_profiles => selected_profiles
                    .into_iter()
                    .filter(|profile| profile.subdir == target_subdir)
                    .collect(),
                None => self
                    .target_profiles
                    .iter()
                    .filter(|profile| profile.subdir == target_subdir)
                    .collect(),
            };
            if !self.target_profiles.is_empty() && candidate_profiles.is_empty() {
                continue;
            }

            let target_names: Vec<String> = if candidate_profiles.is_empty() {
                vec![target_subdir.to_string()]
            } else {
                candidate_profiles
                    .into_iter()
                    .map(|profile| self.effective_profile_for_env(env_name, profile).name)
                    .collect()
            };
            for target_name in target_names {
                for (name, paths) in self.effective_path_dependencies_with_provenance_for_profile(
                    env_name,
                    &target_name,
                    target_subdir,
                )? {
                    let mut locations: BTreeMap<std::path::PathBuf, Vec<(String, String)>> =
                        BTreeMap::new();
                    for (raw_path, declaring_feature) in paths {
                        let candidate = std::path::PathBuf::from(&raw_path);
                        let joined = if candidate.is_absolute() {
                            candidate
                        } else {
                            workspace_dir.join(candidate)
                        };
                        locations
                            .entry(canonical_or_self(&joined))
                            .or_default()
                            .push((raw_path, declaring_feature));
                    }
                    if locations.len() > 1 {
                        let detail = locations
                            .values()
                            .flatten()
                            .map(|(path, feature)| format!("{feature}:{path}"))
                            .collect::<Vec<_>>()
                            .join(",");
                        bail!(
                            "workspace environment `{env_name}` selects conflicting sources for `{name}` on profile `{target_name}`: {detail}"
                        );
                    }
                    let Some((location, declarations)) = locations.into_iter().next() else {
                        continue;
                    };
                    if location != source_canon {
                        continue;
                    }
                    let entry = matches.entry(name).or_default();
                    entry
                        .0
                        .extend(declarations.into_iter().map(|(_, feature)| feature));
                    entry.1.insert(env_name.clone());
                }
            }
        }

        Ok(matches
            .into_iter()
            .map(|(name, (declaring_features, envs))| DiscoveredOutput {
                name,
                declaring_features: declaring_features.into_iter().collect(),
                envs: envs.into_iter().collect(),
            })
            .collect())
    }

    /// Return the concrete workspace environments that consume the pack at
    /// `source_dir`, but only when path-dependency discovery maps it to at
    /// least one active `[environments]` entry. `None` deliberately abstains
    /// for the feature-only and all-features fallback tiers: those supersets
    /// are useful for validation, but are not precise enough to authorize
    /// ownership-driven routing or dependency removal.
    pub fn precise_consuming_envs(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
    ) -> Option<Vec<String>> {
        self.precise_consuming_envs_for_target(
            workspace_dir,
            source_dir,
            crate::glibc::current_pixi_platform(),
        )
    }

    pub fn precise_consuming_envs_for_target(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        target_subdir: &str,
    ) -> Option<Vec<String>> {
        let envs: BTreeSet<String> = self
            .discover_outputs_for_source_for_target(workspace_dir, source_dir, target_subdir)
            .into_iter()
            .flat_map(|output| output.envs)
            .collect();
        (!envs.is_empty()).then(|| envs.into_iter().collect())
    }

    pub fn precise_consuming_envs_for_resolved_target(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        target: &ResolvedWorkspaceTarget,
    ) -> Option<Vec<String>> {
        let envs: BTreeSet<String> = self
            .discover_outputs_for_source_for_resolved_target(workspace_dir, source_dir, target)
            .into_iter()
            .flat_map(|output| output.envs)
            .collect();
        (!envs.is_empty()).then(|| envs.into_iter().collect())
    }

    /// Compute the effective conda deps an environment would solve
    /// against, restricted to the UNION across multiple envs. Used by
    /// output autodiscovery: when a single discovered output is
    /// referenced by features active in multiple envs, retread
    /// produces ONE bundle whose constraints satisfy ALL of them.
    /// Conflicting constraints become the cascade's problem
    /// (intersection becomes empty -> falls through to PyPI bundle).
    pub fn union_effective_dependencies(
        &self,
        env_names: &[String],
    ) -> BTreeMap<String, Vec<String>> {
        self.union_effective_dependencies_for_target(
            env_names,
            crate::glibc::current_pixi_platform(),
        )
    }

    pub fn union_effective_dependencies_for_target(
        &self,
        env_names: &[String],
        target_subdir: &str,
    ) -> BTreeMap<String, Vec<String>> {
        let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for env in env_names {
            for (k, v) in self.effective_dependencies_for_target(env, target_subdir) {
                let entry = out.entry(k).or_default();
                if !entry.contains(&v) {
                    entry.push(v);
                }
            }
        }
        out
    }

    pub fn union_effective_dependencies_for_resolved_target(
        &self,
        env_names: &[String],
        target: &ResolvedWorkspaceTarget,
    ) -> BTreeMap<String, Vec<String>> {
        let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for env in env_names {
            let dependencies = self
                .effective_dependencies_for_resolved_env(env, target)
                .unwrap_or_default();
            for (name, spec) in dependencies {
                let entry = out.entry(name).or_default();
                if !entry.contains(&spec) {
                    entry.push(spec);
                }
            }
        }
        out
    }

    /// P1: check whether the manifest declares an environment with the
    /// given name. Used to filter `env_names` in `conda_outputs` before
    /// leveling + solving so typos and removed envs never reach the solve
    /// tasks (a missing env's effective_dependencies returns empty, which
    /// produces a trivially-sat result and hides the misconfiguration).
    pub fn has_environment(&self, name: &str) -> bool {
        self.environments.contains_key(name)
    }

    /// v0.35.0+: find which feature's `[feature.X.dependencies]`
    /// table (or the implicit top-level `default` feature) declares
    /// a conda dep with the given name, scoped to the active features
    /// of `env_name`. Used by the conflict classifier to attach
    /// workspace-edit suggestions to the right TOML block.
    ///
    /// Returns:
    /// - `Some("default")` -- the dep is in top-level [dependencies]
    /// - `Some("<feature>")` -- declared in `[feature.<feature>.dependencies]`
    /// - `None` -- env doesn't exist, or no active feature declares it
    pub fn find_declaring_feature(&self, env_name: &str, dep_name: &str) -> Option<String> {
        self.find_declaring_feature_for_target(
            env_name,
            dep_name,
            crate::glibc::current_pixi_platform(),
        )
    }

    pub fn find_declaring_feature_for_target(
        &self,
        env_name: &str,
        dep_name: &str,
        target_subdir: &str,
    ) -> Option<String> {
        let env = self.environments.get(env_name)?;
        // Default first (top-level [dependencies]). If env opts out
        // via no-default-feature, skip.
        if !env.no_default_feature
            && dependency_overlay_for_target(
                &self.dependencies,
                &self.target_dependencies,
                target_subdir,
                target_subdir,
            )
            .contains_key(dep_name)
        {
            return Some("default".to_string());
        }
        for feat_name in &env.features {
            let Some(feat) = self.features.get(feat_name) else {
                continue;
            };
            if dependency_overlay_for_target(
                &feat.dependencies,
                &feat.target_dependencies,
                target_subdir,
                target_subdir,
            )
            .contains_key(dep_name)
            {
                return Some(feat_name.clone());
            }
        }
        None
    }

    /// Compute effective channels across multiple envs, in
    /// declaration order, deduped.
    pub fn union_effective_channels(&self, env_names: &[String]) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for env in env_names {
            for c in self.effective_channels(env) {
                if seen.insert(c.clone()) {
                    out.push(c);
                }
            }
        }
        out
    }

    /// The canonical conda-channel set folded into the courier inputs_hash.
    /// Derived purely from the manifest + paths (NOT from any RPC's
    /// params.channels, which pixi forwards inconsistently across conda/outputs
    /// vs conda/build_v1 for multi-env workspaces -- the cause of replay never
    /// firing). The union of effective channels across EVERY env that references
    /// `source_dir`, in declaration order, deduped. Empty when no env references
    /// it (both sites then agree on empty; solve_fingerprint still folds the
    /// declared channels, so signal is not lost).
    pub fn courier_channel_set(&self, workspace_dir: &Path, source_dir: &Path) -> Vec<String> {
        self.courier_channel_set_for_target(
            workspace_dir,
            source_dir,
            crate::glibc::current_pixi_platform(),
        )
    }

    pub fn courier_channel_set_for_target(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        target_subdir: &str,
    ) -> Vec<String> {
        let outputs =
            self.discover_outputs_for_source_for_target(workspace_dir, source_dir, target_subdir);
        if outputs.is_empty() {
            return Vec::new();
        }
        // Union all env names across every discovered output that references
        // this source_dir. Use a BTreeSet for stable dedup before passing to
        // union_effective_channels (which then deduplicates channels).
        let mut all_envs: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for output in &outputs {
            for env in &output.envs {
                all_envs.insert(env.clone());
            }
        }
        let env_vec: Vec<String> = all_envs.into_iter().collect();
        self.union_effective_channels(&env_vec)
    }

    pub fn courier_channel_set_for_resolved_target(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        target: &ResolvedWorkspaceTarget,
    ) -> Vec<String> {
        let envs = self
            .precise_consuming_envs_for_resolved_target(workspace_dir, source_dir, target)
            .unwrap_or_default();
        self.union_effective_channels(&envs)
    }

    /// Env-aware conda dependency specs for the pack built from
    /// `source_dir`, keyed by package name to every DISTINCT spec
    /// declared for it across the envs that actually consume this
    /// pack. Fix for the "env-scoped oracle blind spot": auto-route's
    /// co-installability check used to validate exact conda pins only
    /// against the `default` env's deps, so a pin that satisfied
    /// `default` but violated another consuming env's range (e.g.
    /// `pillow >=11,<12`) sailed through the check and only blew up
    /// the workspace lock later. Feeding every distinct spec for a
    /// name into the same solve (multiple matchspecs on one package
    /// name) makes the solver require a single version satisfying
    /// ALL of them -- exactly the co-install constraint pixi itself
    /// will enforce.
    ///
    /// Mapping precedence (mirrors [`Self::courier_channel_set`]'s
    /// pack -> envs discovery, with fallbacks for when it can't
    /// resolve to a concrete env):
    /// 1. `discover_outputs_for_source` finds the path-deps whose
    ///    `path` resolves to `source_dir`; union the `envs` across
    ///    every discovered output name (there's usually exactly one).
    /// 2. If that yields at least one env, union `effective_dependencies`
    ///    over those envs (via [`Self::union_effective_dependencies`]) --
    ///    the precise, unambiguous case.
    /// 3. If outputs were found but none map to an active env (a
    ///    declaring feature exists but no `[environments]` entry
    ///    activates it), fall back to the union of the declaring
    ///    features' own `dependencies` tables -- still scoped to the
    ///    pack, just not env-precise.
    /// 4. If nothing at all references `source_dir` (workspace doesn't
    ///    declare it, or path resolution failed), conservative
    ///    superset: union across every feature's dependencies plus the
    ///    top-level default -- "else all features" per the fix spec.
    pub fn consuming_env_dependencies(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
    ) -> BTreeMap<String, Vec<String>> {
        self.consuming_env_dependencies_for_target(
            workspace_dir,
            source_dir,
            crate::glibc::current_pixi_platform(),
        )
    }

    pub fn consuming_env_dependencies_for_target(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        target_subdir: &str,
    ) -> BTreeMap<String, Vec<String>> {
        let outputs =
            self.discover_outputs_for_source_for_target(workspace_dir, source_dir, target_subdir);
        if !outputs.is_empty() {
            let mut envs: BTreeSet<String> = BTreeSet::new();
            let mut features: BTreeSet<String> = BTreeSet::new();
            for output in &outputs {
                envs.extend(output.envs.iter().cloned());
                features.extend(output.declaring_features.iter().cloned());
            }
            if !envs.is_empty() {
                let env_vec: Vec<String> = envs.into_iter().collect();
                return self.union_effective_dependencies_for_target(&env_vec, target_subdir);
            }
            // Tier 3: declaring features exist but no active env
            // reaches them -- union their raw dependency tables.
            let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for feat_name in &features {
                let deps = if feat_name == DEFAULT_FEATURE {
                    dependency_overlay_for_target(
                        &self.dependencies,
                        &self.target_dependencies,
                        target_subdir,
                        target_subdir,
                    )
                } else if let Some(f) = self.features.get(feat_name) {
                    dependency_overlay_for_target(
                        &f.dependencies,
                        &f.target_dependencies,
                        target_subdir,
                        target_subdir,
                    )
                } else {
                    continue;
                };
                for (k, v) in deps {
                    let entry = out.entry(k).or_default();
                    if !entry.contains(&v) {
                        entry.push(v);
                    }
                }
            }
            if !out.is_empty() {
                return out;
            }
        }
        // Tier 4: nothing maps to this pack at all -- conservative
        // superset over every feature (plus the top-level default).
        let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (k, v) in dependency_overlay_for_target(
            &self.dependencies,
            &self.target_dependencies,
            target_subdir,
            target_subdir,
        ) {
            out.entry(k).or_default().push(v);
        }
        for feat in self.features.values() {
            for (k, v) in dependency_overlay_for_target(
                &feat.dependencies,
                &feat.target_dependencies,
                target_subdir,
                target_subdir,
            ) {
                let entry = out.entry(k).or_default();
                if !entry.contains(&v) {
                    entry.push(v);
                }
            }
        }
        out
    }

    pub fn consuming_env_dependencies_for_resolved_target(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        target: &ResolvedWorkspaceTarget,
    ) -> BTreeMap<String, Vec<String>> {
        let envs = self
            .precise_consuming_envs_for_resolved_target(workspace_dir, source_dir, target)
            .unwrap_or_default();
        if envs.is_empty() {
            return BTreeMap::new();
        }
        self.union_effective_dependencies_for_resolved_target(&envs, target)
    }

    /// System requirements (cuda, libc, ...) unioned over the envs that
    /// actually CONSUME the pack at `source_dir` -- the system-requirement
    /// counterpart of [`Self::consuming_env_dependencies`], and the run-34
    /// fix for the co-install oracle's `__cuda` blind spot: the check fed
    /// `effective_system_requirements("default")`, but a
    /// `no-default-feature` env like pm-isaaclab declares `cuda = "12"`
    /// under ITS OWN feature -- with no `__cuda` virtual package, the
    /// trial set containing the env's `pytorch-gpu ==2.7.0` (cuda-only
    /// builds) was unsat REGARDLESS of the routed pins, the greedy
    /// retry-solve could never heal it, and every route (`pytorch
    /// ==2.10.0` included) sailed through "unchecked, applying the round
    /// unchanged".
    ///
    /// Same env-discovery tiers as `consuming_env_dependencies`; the
    /// fallback (nothing maps) unions the default env with every
    /// feature's declaration (conservative superset). Workspace-wide
    /// `[workspace].platforms` glibc/cuda declarations win last, exactly
    /// like `effective_system_requirements`.
    pub fn consuming_env_system_requirements(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
    ) -> BTreeMap<String, String> {
        self.consuming_env_system_requirements_for_target(
            workspace_dir,
            source_dir,
            crate::glibc::current_pixi_platform(),
        )
    }

    /// Exact-target form of [`Self::consuming_env_system_requirements`].
    pub fn consuming_env_system_requirements_for_target(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        target_subdir: &str,
    ) -> BTreeMap<String, String> {
        let mut out: BTreeMap<String, String> = BTreeMap::new();
        let outputs =
            self.discover_outputs_for_source_for_target(workspace_dir, source_dir, target_subdir);
        let mut envs: BTreeSet<String> = BTreeSet::new();
        let mut features: BTreeSet<String> = BTreeSet::new();
        for output in &outputs {
            envs.extend(output.envs.iter().cloned());
            features.extend(output.declaring_features.iter().cloned());
        }
        if !envs.is_empty() {
            for env in &envs {
                for (k, v) in self.effective_system_requirements_for_target(env, target_subdir) {
                    out.insert(k, v);
                }
            }
            return out;
        }
        if !features.is_empty() {
            for feat_name in &features {
                if let Some(f) = self.features.get(feat_name) {
                    for (k, v) in &f.system_requirements {
                        out.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        if out.is_empty() {
            // Conservative superset: default env + every feature.
            out = self.effective_system_requirements_for_target("default", target_subdir);
            for feat in self.features.values() {
                for (k, v) in &feat.system_requirements {
                    out.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
        }
        out.retain(|name, _| legacy_system_requirement_applies_to_subdir(name, target_subdir));
        // Workspace-wide platform declarations win (parity with
        // effective_system_requirements).
        if let Some(glibc) = self.platform_glibc.get(target_subdir) {
            out.insert("libc".to_string(), glibc.clone());
        }
        if let Some(cuda) = self.platform_cuda.get(target_subdir) {
            out.insert("cuda".to_string(), cuda.clone());
        }
        out
    }

    pub fn consuming_env_system_requirements_for_resolved_target(
        &self,
        workspace_dir: &Path,
        source_dir: &Path,
        target: &ResolvedWorkspaceTarget,
    ) -> BTreeMap<String, String> {
        let envs = self
            .precise_consuming_envs_for_resolved_target(workspace_dir, source_dir, target)
            .unwrap_or_default();
        let mut out = BTreeMap::new();
        for env in envs {
            out.extend(self.effective_system_requirements_for_contract(&env, &target.contract));
        }
        out
    }
}

/// Sentinel name used to represent the implicit "default" feature
/// (top-level [dependencies]). Pixi reserves this internally so users
/// can't define a feature literally called "default".
const DEFAULT_FEATURE: &str = "default";

fn canonical_or_self(p: &Path) -> std::path::PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

/// v0.32.0+: solve the env's direct conda deps coherently against the
/// workspace's channels, then accumulate the `depends` constraints of
/// the ACTUAL selected builds. The result is a map
/// `dep_name -> list of constraint strings` telling retread what each
/// of THIS env's already-pinned conda packages requires from OTHER
/// packages.
///
/// Example: gsi-ros2 declares `ros-humble-joint-state-publisher = "*"`.
/// Robostack's latest py3.11 build of that package depends on
/// `numpy >=1.26,<2`. After this pass, the returned map contains
/// `{"numpy": [">=1.26,<2"]}`. retread's cascade then knows to emit
/// numpy compatibly with that range.
///
/// Self-references (deps the bundle is itself producing) are filtered
/// by the caller via `bundle_names`. Path/git deps are already
/// filtered at parse time (no version pin to query).
fn transitive_solve_specs(
    deps: &BTreeMap<String, String>,
    bundle_names: &HashSet<PypiKey>,
) -> Vec<CondaMatchSpec> {
    deps.iter()
        .filter_map(|(dep_name, dep_spec)| {
            let conda_name = CondaName::new(dep_name.as_str());
            if bundle_names.contains(&conda_name.key()) {
                return None;
            }
            Some(conda_name.match_spec(dep_spec.as_str()))
        })
        .collect()
}

fn fold_transitive_constraints(
    solved_records: &[rattler_conda_types::RepoDataRecord],
    bundle_names: &HashSet<PypiKey>,
) -> BTreeMap<PypiKey, Vec<String>> {
    let mut out: BTreeMap<PypiKey, Vec<String>> = BTreeMap::new();
    for record in solved_records {
        let selected_name = CondaName::new(record.package_record.name.as_normalized());
        if bundle_names.contains(&selected_name.key()) {
            continue;
        }
        // P3 (grizzly #6): walk `depends` AND `constrains`. A
        // `run_constrained` entry imposes a real constraint in pixi's
        // actual solve whenever the constrained package is present --
        // skipping them was an input-parity hole vs the v0.37.0
        // "retread's verdict predicts pixi's" contract. The skip set
        // stays python/python_abi-ONLY deliberately: recording a
        // workspace-imposed anchor constraint (cuda-version,
        // libstdcxx-ng) is INPUT-side parity; the rule never to widen
        // beyond the ABI compatibility band is EMISSION-side and
        // enforced at its own three layers (classifier, refinement
        // re-check, output invariant).
        for (trans_name, trans_spec) in constraint_lines(
            &record.package_record.depends,
            &record.package_record.constrains,
        ) {
            let transitive_key = CondaName::new(trans_name.as_str()).key();
            out.entry(transitive_key).or_default().push(trans_spec);
        }
    }
    out
}

pub async fn extract_transitive_constraints(
    manifest: &WorkspaceManifest,
    env_name: &str,
    target_python: &str,
    conda_channels: &[rattler_conda_types::ChannelUrl],
    bundle_names: &HashSet<PypiKey>,
) -> BTreeMap<PypiKey, Vec<String>> {
    extract_transitive_constraints_for_target(
        manifest,
        env_name,
        target_python,
        crate::glibc::current_pixi_platform(),
        conda_channels,
        bundle_names,
    )
    .await
}

/// Target-aware workspace transitive solve. The target subdir controls both
/// the conda repodata selected by the solver and the rich platform system
/// requirements injected as virtual packages.
pub async fn extract_transitive_constraints_for_target(
    manifest: &WorkspaceManifest,
    env_name: &str,
    target_python: &str,
    target_subdir: &str,
    conda_channels: &[rattler_conda_types::ChannelUrl],
    bundle_names: &HashSet<PypiKey>,
) -> BTreeMap<PypiKey, Vec<String>> {
    let deps = manifest.effective_dependencies_for_target(env_name, target_subdir);
    let system_requirements =
        manifest.effective_system_requirements_for_target(env_name, target_subdir);
    extract_transitive_constraints_with_inputs(
        manifest,
        env_name,
        target_python,
        target_subdir,
        conda_channels,
        bundle_names,
        deps,
        system_requirements,
        None,
    )
    .await
}

pub async fn extract_transitive_constraints_for_contract(
    manifest: &WorkspaceManifest,
    env_name: &str,
    target_python: &str,
    contract: &WorkspaceTargetContract,
    conda_channels: &[rattler_conda_types::ChannelUrl],
    bundle_names: &HashSet<PypiKey>,
) -> BTreeMap<PypiKey, Vec<String>> {
    let deps = manifest.effective_dependencies_for_target(env_name, &contract.subdir);
    let system_requirements =
        manifest.effective_system_requirements_for_contract(env_name, contract);
    extract_transitive_constraints_with_inputs(
        manifest,
        env_name,
        target_python,
        &contract.subdir,
        conda_channels,
        bundle_names,
        deps,
        system_requirements,
        Some(&contract.detected_virtual_packages),
    )
    .await
}

pub async fn extract_transitive_constraints_for_resolved_target(
    manifest: &WorkspaceManifest,
    env_name: &str,
    target_python: &str,
    target: &ResolvedWorkspaceTarget,
    conda_channels: &[rattler_conda_types::ChannelUrl],
    bundle_names: &HashSet<PypiKey>,
) -> BTreeMap<PypiKey, Vec<String>> {
    let deps = manifest
        .effective_dependencies_for_resolved_env(env_name, target)
        .unwrap_or_default();
    let system_requirements =
        manifest.effective_system_requirements_for_contract(env_name, &target.contract);
    extract_transitive_constraints_with_inputs(
        manifest,
        env_name,
        target_python,
        &target.contract.subdir,
        conda_channels,
        bundle_names,
        deps,
        system_requirements,
        Some(&target.contract.detected_virtual_packages),
    )
    .await
}

async fn extract_transitive_constraints_with_inputs(
    manifest: &WorkspaceManifest,
    env_name: &str,
    target_python: &str,
    target_subdir: &str,
    conda_channels: &[rattler_conda_types::ChannelUrl],
    bundle_names: &HashSet<PypiKey>,
    deps: BTreeMap<String, String>,
    system_requirements: BTreeMap<String, String>,
    detected_virtual_packages: Option<&BTreeMap<String, String>>,
) -> BTreeMap<PypiKey, Vec<String>> {
    let channel_priority = match manifest.channel_priority.as_deref() {
        Some("disabled") => rattler_solve::ChannelPriority::Disabled,
        _ => rattler_solve::ChannelPriority::Strict,
    };
    let solve_specs = transitive_solve_specs(&deps, bundle_names);

    let solved_records = match crate::conda_solve::solve_selected_records_for_target(
        conda_channels,
        &solve_specs,
        target_python,
        target_subdir,
        channel_priority,
        &system_requirements,
        detected_virtual_packages,
        rattler_solve::SolveStrategy::LowestVersionDirect,
    )
    .await
    {
        Ok(records) => records,
        Err(reasons) => {
            tracing::warn!(
                env = %env_name,
                reasons = ?reasons,
                "workspace: coherent solve for transitive extraction failed; skipping transitive constraints"
            );
            return BTreeMap::new();
        }
    };

    fold_transitive_constraints(&solved_records, bundle_names)
}

/// P3 (grizzly #6): the depends + constrains line walk for one solved
/// record. `constrains` (run_constrained) entries impose real
/// constraints in pixi's solve whenever the constrained package is
/// present; they share the dep-line shape so the same parsing applies.
/// Skips stay python/python_abi-ONLY: recording a workspace-imposed
/// anchor constraint (cuda-version, libstdcxx-ng) is INPUT-side
/// parity; the rule never to widen beyond the ABI compatibility band
/// is EMISSION-side, enforced at its own three layers. Empty/`*` specs impose nothing and would
/// corrupt the comma-AND join, so they're dropped.
fn constraint_lines(depends: &[String], constrains: &[String]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for raw in depends.iter().chain(constrains.iter()) {
        let Some((trans_name, trans_spec)) = split_conda_dep_line(raw) else {
            continue;
        };
        if trans_name == "python" || trans_name == "python_abi" {
            continue;
        }
        if trans_spec.is_empty() || trans_spec == "*" {
            continue;
        }
        out.push((trans_name, trans_spec));
    }
    out
}

/// Parse one top-level or feature-scoped `[pypi-dependencies]` table.
/// Names use the same PEP 503 canonical form as routing. Registry version
/// strings stay intact; direct sources carry ownership but no comparable
/// registry constraint, so they become `"*"`.
fn parse_pypi_dependencies(container: &toml::Value) -> BTreeMap<String, String> {
    let Some(deps) = container
        .get("pypi-dependencies")
        .or_else(|| container.get("pypi_dependencies"))
        .and_then(|v| v.as_table())
    else {
        return BTreeMap::new();
    };

    deps.iter()
        .filter_map(|(raw_name, value)| {
            let name = crate::relax::canonical_conda_name(raw_name);
            if name.is_empty() {
                return None;
            }
            let spec = match value {
                toml::Value::String(spec) => Some(spec.clone()),
                toml::Value::Table(detail) => {
                    if detail.contains_key("url")
                        || detail.contains_key("path")
                        || detail.contains_key("git")
                    {
                        Some("*".to_string())
                    } else {
                        detail
                            .get("version")
                            .and_then(|v| v.as_str())
                            .map(String::from)
                    }
                }
                _ => None,
            }?;
            Some((name, spec))
        })
        .collect()
}

/// Parse the index-bearing fields from a `[pypi-options]` table nested
/// under `container` (the manifest root or a `[feature.X]` value).
fn parse_pypi_options(container: &toml::Value) -> PypiOptions {
    let Some(opts) = container
        .get("pypi-options")
        .or_else(|| container.get("pypi_options"))
        .and_then(|v| v.as_table())
    else {
        return PypiOptions::default();
    };
    let index_url = opts
        .get("index-url")
        .or_else(|| opts.get("index_url"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let extra_index_urls = opts
        .get("extra-index-urls")
        .or_else(|| opts.get("extra_index_urls"))
        .and_then(|v| v.as_array())
        .map(|extra| {
            extra
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    PypiOptions {
        index_url,
        extra_index_urls,
    }
}

fn push_unique_index_url(indexes: &mut Vec<String>, index: String) {
    if !indexes
        .iter()
        .any(|existing| existing.trim_end_matches('/') == index.trim_end_matches('/'))
    {
        indexes.push(index);
    }
}

/// v0.37.0+ (D1): parse one `[system-requirements]` value. pixi allows
/// either a bare scalar (`cuda = "12"`, sometimes a number) or a table
/// (`libc = { family = "glibc", version = "2.35" }`). Scalars are kept
/// verbatim; tables contribute their `version` field. Anything else
/// returns `None` so the caller skips it.
fn parse_system_requirement_value(key: &str, v: &toml::Value) -> Option<String> {
    if let Some(s) = v.as_str() {
        return Some(s.to_string());
    }
    if let Some(i) = v.as_integer() {
        return Some(i.to_string());
    }
    if let Some(f) = v.as_float() {
        return Some(f.to_string());
    }
    if let Some(t) = v.as_table() {
        if key == "libc"
            && let Some(family) = t.get("family").and_then(|x| x.as_str())
            && family != "glibc"
        {
            return None;
        }
        if let Some(ver) = t.get("version").and_then(|x| x.as_str()) {
            return Some(ver.to_string());
        }
    }
    None
}

/// Pixi 0.73 migrates the deprecated system-requirements table into each
/// concrete platform profile only when that requirement applies there.
/// Legacy `archspec` is intentionally not migrated; Pixi supplies the
/// profile's baseline architecture through detection instead.
fn legacy_system_requirement_applies_to_subdir(name: &str, subdir: &str) -> bool {
    match name.trim().trim_start_matches("__") {
        "libc" | "glibc" | "linux" => subdir.starts_with("linux-"),
        "macos" | "osx" => subdir.starts_with("osx-"),
        "windows" | "win" => subdir.starts_with("win-"),
        // Pixi 0.73 migrates legacy CUDA to every supported target except
        // macOS, including FreeBSD, Emscripten, WASI, and z/OS.
        "cuda" => !subdir.starts_with("osx-"),
        "archspec" => false,
        _ => false,
    }
}

/// Pixi composes repeated version-like legacy requirements as minimum floors,
/// so the strongest (highest) active declaration wins irrespective of feature
/// order. Non-version keys retain ordinary later-layer precedence.
fn merge_legacy_system_requirement(out: &mut BTreeMap<String, String>, key: &str, value: &str) {
    let version_floor = matches!(
        key,
        "cuda" | "libc" | "glibc" | "linux" | "macos" | "osx" | "windows" | "win"
    );
    if version_floor
        && let Some(existing) = out.get(key)
        && compare_numeric_versions(existing, value).is_some_and(|ordering| ordering.is_gt())
    {
        return;
    }
    out.insert(key.to_string(), value.to_string());
}

fn compare_numeric_versions(left: &str, right: &str) -> Option<std::cmp::Ordering> {
    fn components(value: &str) -> Option<Vec<u64>> {
        value
            .split('.')
            .map(str::parse::<u64>)
            .collect::<Result<Vec<_>, _>>()
            .ok()
    }
    let mut left = components(left)?;
    let mut right = components(right)?;
    let width = left.len().max(right.len());
    left.resize(width, 0);
    right.resize(width, 0);
    Some(left.cmp(&right))
}

fn extend_profiles_with_feature_subdirs<'a>(
    profiles: &mut Vec<WorkspaceProfile>,
    features: impl Iterator<Item = &'a FeatureDef>,
) {
    // Pixi extends only a legacy subdir-only platform list. Once a rich
    // profile exists, feature restrictions must reference a declared profile
    // or subdir and Pixi reports unknown references instead of inventing one.
    if !profiles.iter().all(|profile| {
        profile.name == profile.subdir && profile.declared_virtual_packages.is_empty()
    }) {
        return;
    }
    for feature in features {
        let Some(restrictions) = &feature.platforms else {
            continue;
        };
        for restriction in restrictions {
            if !is_known_conda_subdir(restriction)
                || profiles.iter().any(|profile| profile.name == *restriction)
            {
                continue;
            }
            profiles.push(WorkspaceProfile {
                name: restriction.clone(),
                subdir: restriction.clone(),
                declared_virtual_packages: BTreeMap::new(),
            });
        }
    }
}

const RAW_FRIENDLY_SYNTHESIS_PREFIX: &str = "@raw-friendly:";

fn profile_virtual_package_synthesis_key(raw_key: &str, normalized_key: &str) -> String {
    if raw_key.starts_with("__")
        && matches!(
            normalized_key,
            "cuda" | "glibc" | "linux" | "macos" | "windows"
        )
    {
        format!("{RAW_FRIENDLY_SYNTHESIS_PREFIX}{normalized_key}")
    } else if raw_key.starts_with("__") {
        raw_key.to_string()
    } else {
        normalized_key.to_string()
    }
}

fn normalize_raw_profile_virtual_package_value(
    raw_key: &str,
    normalized_key: &str,
    value: &str,
    subdir: &str,
) -> String {
    if !raw_key.starts_with("__") {
        return value.to_string();
    }
    match normalized_key {
        "archspec"
            if pixi_073_default_archspec(subdir)
                .is_some_and(|expected| value == format!("0={expected}")) =>
        {
            pixi_073_default_archspec(subdir).unwrap().to_string()
        }
        // Pixi renders both the default raw Unix package and a raw `0`
        // override as `__unix` in platform-list JSON. Raw provenance remains
        // in the synthesis map, where those spellings still get distinct
        // auto-names.
        "unix" => String::new(),
        "cuda" | "glibc" | "linux" | "macos" | "windows" => {
            value.strip_suffix("=0").unwrap_or(value).to_string()
        }
        _ => value.to_string(),
    }
}

/// Parse one Pixi workspace platform declaration while retaining its profile
/// name and every declared virtual package. A bare subdir is itself a named
/// profile; Pixi synthesizes a stable name for an unnamed rich declaration.
fn parse_workspace_profile(v: &toml::Value) -> Option<WorkspaceProfile> {
    if let Some(subdir) = v.as_str() {
        return Some(WorkspaceProfile {
            name: subdir.to_string(),
            subdir: subdir.to_string(),
            declared_virtual_packages: BTreeMap::new(),
        });
    }

    let table = v.as_table()?;
    let explicit_name = table.get("name").and_then(toml::Value::as_str);
    let subdir = match table.get("platform").and_then(toml::Value::as_str) {
        Some(platform) => platform.to_string(),
        None => explicit_name
            .filter(|name| is_known_conda_subdir(name))?
            .to_string(),
    };
    let mut declared_virtual_packages = BTreeMap::new();
    let mut synthesis_virtual_packages = BTreeMap::new();
    for (raw_key, raw_value) in table {
        if raw_key == "name" || raw_key == "platform" {
            continue;
        }
        let key = normalize_virtual_package_name(raw_key);
        if key == "cuda"
            && let Some(cuda) = raw_value.as_table()
        {
            let driver = cuda.get("driver").and_then(parse_scalarish_value);
            let arch = cuda.get("arch").and_then(parse_scalarish_value);
            if let Some(driver) = driver {
                synthesis_virtual_packages.insert("cuda".to_string(), driver.clone());
                declared_virtual_packages
                    .insert("cuda".to_string(), canonical_virtual_package_value(&driver));
                if let Some(arch) = arch {
                    synthesis_virtual_packages.insert("cuda_arch".to_string(), arch.clone());
                    declared_virtual_packages.insert(
                        "cuda_arch".to_string(),
                        canonical_virtual_package_value(&arch),
                    );
                }
            }
            continue;
        }
        if let Some(value) = parse_profile_virtual_package_value(&key, raw_value) {
            let semantic_value =
                normalize_raw_profile_virtual_package_value(raw_key, &key, &value, &subdir);
            let synthesis_key = profile_virtual_package_synthesis_key(raw_key, &key);
            let synthesis_value = if synthesis_key.starts_with(RAW_FRIENDLY_SYNTHESIS_PREFIX) {
                semantic_value.clone()
            } else {
                parse_profile_virtual_package_name_value(&key, raw_value)
                    .unwrap_or_else(|| semantic_value.clone())
            };
            synthesis_virtual_packages.insert(synthesis_key, synthesis_value);
            declared_virtual_packages.insert(key, semantic_value);
        }
    }
    let name = table
        .get("name")
        .and_then(toml::Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| synthesize_workspace_profile_name(&subdir, &synthesis_virtual_packages));
    Some(WorkspaceProfile {
        name,
        subdir,
        declared_virtual_packages,
    })
}

fn is_known_conda_subdir(value: &str) -> bool {
    matches!(
        value,
        "linux-32"
            | "linux-64"
            | "linux-aarch64"
            | "linux-armv6l"
            | "linux-armv7l"
            | "linux-loongarch64"
            | "linux-ppc64le"
            | "linux-ppc64"
            | "linux-ppc"
            | "linux-s390x"
            | "linux-riscv32"
            | "linux-riscv64"
            | "freebsd-32"
            | "freebsd-64"
            | "freebsd-arm64"
            | "osx-64"
            | "osx-arm64"
            | "win-32"
            | "win-64"
            | "win-arm64"
            | "emscripten-wasm32"
            | "wasi-wasm32"
            | "zos-z"
    )
}

/// Pixi 0.73-compatible unnamed rich-platform name synthesis. This baseline
/// is used only to interpret manifest profile names; it is deliberately not
/// injected into Retread's resolution contract. The latter hashes only
/// manifest-verified declarations (or Pixi's exact detected envelope), so an
/// evolving Pixi default table cannot silently change solve compatibility.
fn synthesize_workspace_profile_name(subdir: &str, declared: &BTreeMap<String, String>) -> String {
    const FRIENDLY_ORDER: &[&str] = &["cuda", "archspec", "glibc", "linux", "macos", "windows"];

    let mut parts = vec![subdir.to_string()];
    let mut consumed = BTreeSet::new();
    for friendly in FRIENDLY_ORDER {
        let aliases: &[&str] = if *friendly == "macos" {
            &["macos", "osx"]
        } else {
            std::slice::from_ref(friendly)
        };
        let Some((key, value, raw_friendly)) = aliases.iter().find_map(|key| {
            declared
                .get(*key)
                .map(|value| ((*key).to_string(), value, false))
                .or_else(|| {
                    let raw_key = format!("{RAW_FRIENDLY_SYNTHESIS_PREFIX}{key}");
                    declared.get(&raw_key).map(|value| (raw_key, value, true))
                })
        }) else {
            continue;
        };
        consumed.insert(key.clone());
        if !raw_friendly && is_pixi_073_profile_name_default(subdir, &key, value) {
            continue;
        }
        parts.push(format!(
            "{friendly}-{}",
            sanitize_workspace_profile_name_segment(value)
        ));
    }
    for (key, value) in declared {
        if consumed.contains(key) || is_pixi_073_profile_name_default(subdir, key, value) {
            continue;
        }
        let display_key = key.trim_start_matches('_');
        parts.push(format!(
            "{}-{}",
            sanitize_workspace_profile_name_segment(display_key),
            sanitize_workspace_profile_name_segment(value)
        ));
    }
    parts.join("-")
}

fn is_pixi_073_profile_name_default(subdir: &str, key: &str, value: &str) -> bool {
    let value = canonical_virtual_package_value(value);
    match key {
        "__unix" => value == "0=0",
        "linux" => subdir.starts_with("linux-") && value == "4.18",
        "glibc" => subdir.starts_with("linux-") && value == "2.28",
        "macos" => subdir.starts_with("osx-") && value == "13",
        "windows" => subdir.starts_with("win-") && value == "10",
        "archspec" => pixi_073_default_archspec(subdir) == Some(value.as_str()),
        "__archspec" => pixi_073_default_archspec(subdir)
            .is_some_and(|expected| value == format!("0={expected}")),
        _ => false,
    }
}

fn sanitize_workspace_profile_name_segment(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut last_was_dash = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            out.push(character);
            last_was_dash = false;
        } else if !last_was_dash {
            out.push('-');
            last_was_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

fn derive_unambiguous_platform_values(
    profiles: &[WorkspaceProfile],
    virtual_package: &str,
    out: &mut BTreeMap<String, String>,
) {
    let subdirs: BTreeSet<&str> = profiles
        .iter()
        .map(|profile| profile.subdir.as_str())
        .collect();
    for subdir in subdirs {
        let values: BTreeSet<Option<&str>> = profiles
            .iter()
            .filter(|profile| profile.subdir == subdir)
            .map(|profile| {
                profile
                    .declared_virtual_packages
                    .get(virtual_package)
                    .map(String::as_str)
            })
            .collect();
        if values.len() == 1
            && let Some(Some(value)) = values.into_iter().next()
        {
            out.insert(subdir.to_string(), value.to_string());
        }
    }
}

fn normalize_virtual_package_name(name: &str) -> String {
    match name.trim().trim_start_matches("__") {
        "osx" => "macos".to_string(),
        "win" => "windows".to_string(),
        name => name.to_string(),
    }
}

fn canonical_virtual_package_value(value: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        return String::new();
    }
    let mut components = value.splitn(2, '=');
    let head = components.next().unwrap_or_default();
    let tail = components.next();
    let canonical_head = if head
        .split('.')
        .all(|component| !component.is_empty() && component.chars().all(|c| c.is_ascii_digit()))
    {
        let mut pieces: Vec<&str> = head.split('.').collect();
        while pieces.len() > 1 && pieces.last() == Some(&"0") {
            pieces.pop();
        }
        pieces.join(".")
    } else {
        head.to_string()
    };
    match tail {
        Some(tail) => format!("{canonical_head}={tail}"),
        None => canonical_head,
    }
}

fn validate_detected_target_shape(
    subdir: &str,
    declared: &BTreeMap<String, String>,
    detected: &BTreeMap<String, String>,
    profile_name: &str,
) -> Result<()> {
    let (required, forbidden): (&[&str], &[&str]) = if subdir.starts_with("linux-") {
        (
            &["unix", "linux", "glibc", "archspec"],
            &["macos", "windows"],
        )
    } else if subdir.starts_with("osx-") {
        (
            &["unix", "macos", "archspec"],
            &["linux", "glibc", "windows"],
        )
    } else if subdir.starts_with("win-") {
        (
            &["windows", "archspec"],
            &["unix", "linux", "glibc", "macos"],
        )
    } else if subdir.starts_with("freebsd-") || subdir == "emscripten-wasm32" {
        (&["unix"], &["linux", "glibc", "macos", "windows"])
    } else if subdir == "wasi-wasm32" || subdir == "zos-z" {
        (
            &[],
            &["unix", "linux", "glibc", "macos", "windows", "archspec"],
        )
    } else {
        (&[], &[])
    };
    for required in required {
        if !detected.contains_key(*required) {
            bail!(
                "{WORKSPACE_TARGET_ENV} detected contract for `{profile_name}` is missing `{required}`"
            );
        }
    }
    for forbidden in forbidden {
        if detected.contains_key(*forbidden) {
            bail!(
                "{WORKSPACE_TARGET_ENV} detected contract for `{profile_name}` contains incompatible `{forbidden}` for `{subdir}`"
            );
        }
    }

    // An explicit archspec was already checked against the detected value by
    // declaration preservation above. Otherwise Pixi emits a deterministic
    // subdir baseline; validate it so a forged envelope cannot cross arches.
    if !declared.contains_key("archspec")
        && let Some(expected_arch) = pixi_073_default_archspec(subdir)
    {
        let actual = detected.get("archspec").map(|value| {
            value
                .rsplit_once('=')
                .map_or(value.as_str(), |(_, build)| build)
        });
        if actual != Some(expected_arch) {
            bail!(
                "{WORKSPACE_TARGET_ENV} detected archspec for `{profile_name}` is `{}`, expected `{expected_arch}` for `{subdir}`",
                actual.unwrap_or("<missing>")
            );
        }
    }
    Ok(())
}

fn pixi_073_default_archspec(subdir: &str) -> Option<&'static str> {
    match subdir {
        "linux-32" | "win-32" => Some("x86"),
        "linux-64" | "osx-64" | "win-64" => Some("x86_64"),
        "linux-aarch64" | "win-arm64" => Some("aarch64"),
        "osx-arm64" => Some("m1"),
        "linux-ppc64le" => Some("ppc64le"),
        "linux-ppc64" => Some("ppc64"),
        "linux-ppc" => Some("ppc"),
        "linux-s390x" => Some("s390x"),
        "linux-riscv32" => Some("riscv32"),
        "linux-riscv64" => Some("riscv64"),
        "linux-loongarch64" => Some("loongarch64"),
        // Pixi 0.73 currently detects the aarch64 baseline for both legacy
        // 32-bit ARM conda subdirs.
        "linux-armv6l" | "linux-armv7l" => Some("aarch64"),
        _ => None,
    }
}

fn parse_virtual_package_lines(lines: &[String]) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for raw in lines {
        let raw = raw.trim();
        if raw.is_empty() {
            bail!("virtual-package entries must not be empty");
        }
        // `pixi workspace platform list --json` normally emits compact
        // `name=value` entries, but coupled CUDA declarations retain their
        // TOML display (`cuda = { driver = "12", arch = "8.6" }`). Parse any
        // valid one-entry TOML rendering before falling back to conda's raw
        // virtual-package spelling.
        if let Ok(rendered) = toml::from_str::<toml::Value>(raw)
            && let Some(table) = rendered.as_table()
            && table.len() == 1
        {
            let (raw_name, rendered_value) = table.iter().next().unwrap();
            let name = normalize_virtual_package_name(raw_name);
            if name == "cuda"
                && let Some(cuda) = rendered_value.as_table()
            {
                let driver = cuda.get("driver").and_then(parse_scalarish_value);
                let arch = cuda.get("arch").and_then(parse_scalarish_value);
                let Some(driver) = driver else {
                    bail!("rendered CUDA virtual-package entry `{raw}` has no driver");
                };
                insert_virtual_package_value(
                    &mut out,
                    "cuda".to_string(),
                    canonical_virtual_package_value(&driver),
                )?;
                if let Some(arch) = arch {
                    insert_virtual_package_value(
                        &mut out,
                        "cuda_arch".to_string(),
                        canonical_virtual_package_value(&arch),
                    )?;
                }
                continue;
            }
            if let Some(value) = parse_profile_virtual_package_value(&name, rendered_value) {
                insert_virtual_package_value(&mut out, name, value)?;
                continue;
            }
        }
        let (name, value) = raw
            .split_once('=')
            .map_or((raw, ""), |(name, value)| (name, value));
        let name = normalize_virtual_package_name(name);
        if name.is_empty() {
            bail!("virtual-package entry `{raw}` has no package name");
        }
        let value = canonical_virtual_package_value(value);
        insert_virtual_package_value(&mut out, name, value)?;
    }
    Ok(out)
}

fn insert_virtual_package_value(
    out: &mut BTreeMap<String, String>,
    name: String,
    value: String,
) -> Result<()> {
    let solver_name = format!("__{name}");
    if solver_name
        .parse::<rattler_conda_types::PackageName>()
        .is_err()
    {
        bail!("virtual-package name `{name}` is not a valid conda package name");
    }
    if let Some(previous) = out.insert(name.clone(), value.clone())
        && previous != value
    {
        bail!(
            "virtual-package `{name}` appears with conflicting values `{previous}` and `{value}`"
        );
    }
    Ok(())
}

fn render_virtual_packages(packages: &BTreeMap<String, String>) -> String {
    packages
        .iter()
        .map(|(name, value)| {
            if value.is_empty() {
                name.clone()
            } else {
                format!("{name}={value}")
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_profile_virtual_package_value(key: &str, value: &toml::Value) -> Option<String> {
    if key == "glibc" {
        return parse_glibc_value(value).map(|value| canonical_virtual_package_value(&value));
    }
    if let Some(value) = parse_scalarish_value(value) {
        return Some(canonical_virtual_package_value(&value));
    }
    let table = value.as_table()?;
    let version = table.get("version").and_then(parse_scalarish_value);
    let build = table.get("build").and_then(parse_scalarish_value);
    match (version, build) {
        (Some(version), Some(build)) => Some(format!(
            "{}={build}",
            canonical_virtual_package_value(&version)
        )),
        (Some(version), None) => Some(canonical_virtual_package_value(&version)),
        (None, Some(build)) => Some(format!("={build}")),
        (None, None) => None,
    }
}

/// Value rendering used only by Pixi's synthesized profile name. Unlike the
/// semantic contract canonicalizer, this preserves a declared spelling such
/// as `12.0`, because Pixi names that profile `...-cuda-12-0` even though
/// Retread safely normalizes the compatibility value to `12` for hashing.
fn parse_profile_virtual_package_name_value(key: &str, value: &toml::Value) -> Option<String> {
    if key == "glibc" {
        return parse_glibc_value(value);
    }
    if let Some(value) = parse_scalarish_value(value) {
        return Some(value);
    }
    let table = value.as_table()?;
    let version = table.get("version").and_then(parse_scalarish_value);
    let build = table.get("build").and_then(parse_scalarish_value);
    match (version, build) {
        (Some(version), Some(build)) => Some(format!("{version}={build}")),
        (Some(version), None) => Some(version),
        (None, Some(build)) => Some(format!("={build}")),
        (None, None) => None,
    }
}

/// Render a scalar-ish TOML value (string, float, integer) as the string
/// form pixi's own schema accepts for rich-platform virtual-package
/// versions (`cuda = "12"` / `cuda = 12`).
fn parse_scalarish_value(v: &toml::Value) -> Option<String> {
    if let Some(s) = v.as_str() {
        return Some(s.to_string());
    }
    if let Some(f) = v.as_float() {
        return Some(f.to_string());
    }
    if let Some(i) = v.as_integer() {
        return Some(i.to_string());
    }
    None
}

fn parse_glibc_value(v: &toml::Value) -> Option<String> {
    if let Some(s) = v.as_str() {
        return Some(s.to_string());
    }
    if let Some(f) = v.as_float() {
        return Some(f.to_string());
    }
    if let Some(i) = v.as_integer() {
        return Some(i.to_string());
    }
    if let Some(t) = v.as_table() {
        let major = t
            .get("major")
            .or_else(|| t.get("maj"))
            .and_then(|x| x.as_integer())?;
        let minor = t
            .get("minor")
            .or_else(|| t.get("min"))
            .and_then(|x| x.as_integer())?;
        return Some(format!("{major}.{minor}"));
    }
    None
}

/// Split a conda match-spec line into (name, spec). The line is what
/// repodata `depends` entries look like: name + optional whitespace +
/// optional spec + optional build-string. We only care about (name,
/// spec); build strings are dropped because retread re-derives them.
///
/// Examples:
///   "numpy >=1.26,<2"          -> Some(("numpy", ">=1.26,<2"))
///   "python_abi 3.11.* *_cp311" -> Some(("python_abi", "3.11.*"))
///   "libstdcxx >=12"           -> Some(("libstdcxx", ">=12"))
///   "libgcc-ng"                -> Some(("libgcc-ng", ""))
///   ""                         -> None
fn split_conda_dep_line(raw: &str) -> Option<(String, String)> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut parts = trimmed.splitn(3, char::is_whitespace);
    let name = parts.next()?.to_string();
    let spec = parts.next().unwrap_or("").trim().to_string();
    Some((name, spec))
}

fn parse_env_def(value: &toml::Value) -> Option<EnvironmentDef> {
    // Bare array form: `env = ["feat1", "feat2"]` (pixi shorthand for
    // `{ features = [...] }`). `default = []` is the common spelling for
    // "just the default feature".
    if let Some(arr) = value.as_array() {
        return Some(EnvironmentDef {
            features: arr
                .iter()
                .filter_map(|f| f.as_str().map(String::from))
                .collect(),
            no_default_feature: false,
        });
    }
    let table = value.as_table()?;
    let mut features = Vec::new();
    if let Some(arr) = table.get("features").and_then(|v| v.as_array()) {
        for f in arr {
            if let Some(name) = f.as_str() {
                features.push(name.to_string());
            }
        }
    }
    let no_default_feature = table
        .get("no-default-feature")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Some(EnvironmentDef {
        features,
        no_default_feature,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rattler_conda_types::{PackageRecord, RepoDataRecord, VersionWithSource};
    use std::str::FromStr;
    use url::Url;

    fn ws_toml(text: &str) -> WorkspaceManifest {
        WorkspaceManifest::from_toml_source(text).unwrap()
    }

    fn temp_workspace(prefix: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "retread-{prefix}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn repo_record(name: &str, version: &str, depends: &[&str]) -> RepoDataRecord {
        let mut package_record = PackageRecord::new(
            name.parse().unwrap(),
            VersionWithSource::from_str(version).unwrap(),
            "h123456_0".to_string(),
        );
        package_record.subdir = "linux-64".to_string();
        package_record.depends = depends.iter().map(|dep| (*dep).to_string()).collect();
        RepoDataRecord {
            package_record,
            file_name: format!("{name}-{version}-h123456_0.conda"),
            url: Url::parse(&format!(
                "https://example.invalid/linux-64/{name}-{version}-h123456_0.conda"
            ))
            .unwrap(),
            channel: Some("https://example.invalid".into()),
        }
    }

    #[test]
    fn extract_transitive_constraints_underscored_env() {
        let ws = ws_toml(
            r#"
[environments]
cuda-env = { features = ["cuda"], no-default-feature = true }

[feature.cuda.dependencies]
cuda-nvcc_linux-64 = "12.9.*"
"#,
        );
        let deps = ws.effective_dependencies("cuda-env");
        let bundle_names = HashSet::new();
        let solve_specs = transitive_solve_specs(&deps, &bundle_names);

        assert_eq!(solve_specs.len(), 1);
        assert_eq!(
            solve_specs[0].as_str(),
            "cuda-nvcc_linux-64 12.9.*",
            "conda solver input must preserve the declared underscore"
        );

        let records = vec![
            repo_record("cuda-nvcc_linux-64", "12.9.1", &["cuda-version >=12.9,<13"]),
            repo_record("cuda-version", "12.9", &[]),
        ];
        // The test-only solver helper predates the production type wall and
        // still accepts rendered strings. Keep this conversion adjacent to
        // that cfg(test)-only boundary; production receives CondaMatchSpec.
        let rendered_specs = solve_specs
            .iter()
            .map(|spec| spec.as_str().to_owned())
            .collect::<Vec<_>>();
        let solved = crate::conda_solve::solve_records_for_test(&records, &rendered_specs, "3.11")
            .expect("raw underscored conda dependency should solve");
        assert!(
            solved.iter().any(|record| {
                record.package_record.name.as_normalized() == "cuda-nvcc_linux-64"
            })
        );

        let constraints = fold_transitive_constraints(&solved, &bundle_names);
        assert_eq!(
            constraints.get(&PypiKey::from_pypi("cuda-version")),
            Some(&vec![">=12.9,<13".to_string()]),
            "the selected underscored package's transitive constraint must survive"
        );
    }

    /// pixi 0.71 introduced a STRUCTURED `[workspace.conda-pypi-map]`
    /// form: per-channel OBJECTS carrying a mapping location plus a
    /// `mapping-mode`, replacing the old flat `channel = "url-or-path"`
    /// strings. retread does not consume the map, but its manifest walk
    /// must TOLERATE the new shape: parsing must succeed and every field
    /// retread does read must be unaffected. (Previously zero coverage.)
    #[test]
    fn tolerates_pixi071_structured_conda_pypi_map() {
        let ws = ws_toml(
            r#"
[workspace]
channels = ["conda-forge", "https://prefix.dev/garylvov"]
platforms = ["linux-64"]

[workspace.conda-pypi-map]
conda-forge = { location = "https://example.com/conda-forge-map.json", mapping-mode = "canonical" }
"https://prefix.dev/garylvov" = { location = "./local-map.json", mapping-mode = "manual" }

[dependencies]
python = "==3.12"
numpy = ">=1.26,<3"
"#,
        );
        // The structured map neither aborts the parse nor bleeds into
        // the fields retread reads.
        assert_eq!(
            ws.channels,
            vec![
                "conda-forge".to_string(),
                "https://prefix.dev/garylvov".to_string()
            ]
        );
        assert_eq!(
            ws.dependencies.get("numpy").map(String::as_str),
            Some(">=1.26,<3")
        );
        assert_eq!(
            ws.dependencies.get("python").map(String::as_str),
            Some("==3.12")
        );

        // Legacy flat-string form (pixi <=0.70) must keep parsing too.
        let legacy = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]
conda-pypi-map = { conda-forge = "https://example.com/map.json" }

[dependencies]
python = "==3.12"
"#,
        );
        assert_eq!(legacy.channels, vec!["conda-forge".to_string()]);
        assert_eq!(
            legacy.dependencies.get("python").map(String::as_str),
            Some("==3.12")
        );
    }

    #[test]
    fn parses_top_level_dependencies_and_channels() {
        let ws = ws_toml(
            r#"
[workspace]
channels = ["https://prefix.dev/conda-forge"]

[dependencies]
numpy = "==1.26.4"
torch = { version = ">=2.7", channel = "pytorch" }
isaac-pack = { path = "./isaac-pack" }
"#,
        );
        assert_eq!(ws.channels, vec!["https://prefix.dev/conda-forge"]);
        assert_eq!(
            ws.dependencies.get("numpy").map(String::as_str),
            Some("==1.26.4")
        );
        assert_eq!(
            ws.dependencies.get("torch").map(String::as_str),
            Some(">=2.7")
        );
        // path-form entries are not version pins; skipped.
        assert!(!ws.dependencies.contains_key("isaac-pack"));
    }

    #[test]
    fn parses_pypi_dependencies_with_canonical_owned_specs() {
        let ws = ws_toml(
            r#"
[pypi-dependencies]
NumPy = ">=1.26,<3"
"direct.git" = { git = "https://example.com/direct.git", rev = "abc123" }
direct_url = { url = "https://example.com/archive.whl" }
local-path = { path = "../local-project", editable = true }
with-extras = { version = "~=4.0", extras = ["speedups"] }

[feature.gpu.pypi-dependencies]
Torch_Vision = "==0.24"
"feature.direct" = { git = "https://example.com/feature.git" }
"#,
        );

        assert_eq!(
            ws.pypi_dependencies.get("numpy").map(String::as_str),
            Some(">=1.26,<3")
        );
        assert_eq!(
            ws.pypi_dependencies.get("with-extras").map(String::as_str),
            Some("~=4.0")
        );
        for direct in ["direct-git", "direct-url", "local-path"] {
            assert_eq!(
                ws.pypi_dependencies.get(direct).map(String::as_str),
                Some("*"),
                "direct source {direct} must remain workspace-owned"
            );
        }
        let gpu = &ws.features["gpu"].pypi_dependencies;
        assert_eq!(gpu.get("torch-vision").map(String::as_str), Some("==0.24"));
        assert_eq!(gpu.get("feature-direct").map(String::as_str), Some("*"));
    }

    #[test]
    fn effective_pypi_dependencies_merge_all_feature_requirements_and_respect_no_default() {
        let ws = ws_toml(
            r#"
[pypi-dependencies]
Shared_Dep = ">=1"
top-only = "==1"

[environments]
layered = { features = ["gpu"] }
isolated = { features = ["gpu"], no-default-feature = true }

[feature.gpu.pypi-dependencies]
shared-dep = "==2"
feature_only = { path = "../feature-only" }
"#,
        );

        let layered = ws.effective_pypi_dependencies("layered");
        assert_eq!(
            layered.get("shared-dep"),
            Some(&vec![">=1".to_string(), "==2".to_string()])
        );
        assert_eq!(layered.get("top-only"), Some(&vec!["==1".to_string()]));
        assert_eq!(layered.get("feature-only"), Some(&vec!["*".to_string()]));

        let isolated = ws.effective_pypi_dependencies("isolated");
        assert_eq!(isolated.get("shared-dep"), Some(&vec!["==2".to_string()]));
        assert_eq!(isolated.get("feature-only"), Some(&vec!["*".to_string()]));
        assert!(!isolated.contains_key("top-only"));
        assert!(ws.effective_pypi_dependencies("missing").is_empty());
    }

    #[test]
    fn parses_environments_and_features() {
        let ws = ws_toml(
            r#"
[environments]
gsi = { features = ["isaaclab", "gpu"] }
gsf = { features = ["flash_sac"], no-default-feature = true }

[feature.isaaclab]

[feature.isaaclab.dependencies]
isaac-pack = { path = "./isaac-pack" }
cmake = "==4.1.0"

[feature.gpu]
channels = ["https://prefix.dev/pytorch"]

[feature.gpu.dependencies]
pytorch-gpu = ">=2.7"

[feature.flash_sac]
"#,
        );
        let gsi = ws.environments.get("gsi").unwrap();
        assert_eq!(gsi.features, vec!["isaaclab", "gpu"]);
        assert!(!gsi.no_default_feature);

        let gsf = ws.environments.get("gsf").unwrap();
        assert!(gsf.no_default_feature);

        let isaaclab = ws.features.get("isaaclab").unwrap();
        assert_eq!(
            isaaclab.dependencies.get("cmake").map(String::as_str),
            Some("==4.1.0"),
        );
        // path-form skipped.
        assert!(!isaaclab.dependencies.contains_key("isaac-pack"));

        let gpu = ws.features.get("gpu").unwrap();
        assert_eq!(gpu.channels, vec!["https://prefix.dev/pytorch"]);
    }

    #[test]
    fn effective_dependencies_unions_features_with_top_level() {
        let ws = ws_toml(
            r#"
[dependencies]
python = "==3.11"
numpy = ">=1.26"

[environments]
gsi = { features = ["physx"] }

[feature.physx]

[feature.physx.dependencies]
# Active feature DependencyMaps are conjunctive rather than last-wins.
numpy = "<2"
pinocchio = ">=3.6"
"#,
        );
        let eff = ws.effective_dependencies("gsi");
        assert_eq!(eff.get("python").map(String::as_str), Some("==3.11"));
        assert_eq!(eff.get("numpy").map(String::as_str), Some(">=1.26,<2"));
        assert_eq!(eff.get("pinocchio").map(String::as_str), Some(">=3.6"));
    }

    #[test]
    fn effective_dependencies_no_default_feature_skips_top_level() {
        let ws = ws_toml(
            r#"
[dependencies]
python = "==3.11"
torch = "==2.7"

[environments]
standalone = { features = ["flash"], no-default-feature = true }

[feature.flash]

[feature.flash.dependencies]
python = "==3.12"
"#,
        );
        let eff = ws.effective_dependencies("standalone");
        // python from the feature, NOT the top-level.
        assert_eq!(eff.get("python").map(String::as_str), Some("==3.12"));
        // torch absent — top-level skipped.
        assert!(!eff.contains_key("torch"));
    }

    #[test]
    fn target_dependencies_overlay_top_level_and_feature_tables() {
        let ws = ws_toml(
            r#"
[dependencies]
shared = "1"
arch-pkg = "0"

[target.linux-64.dependencies]
arch-pkg = "64"
x86-only = "*"

[target.linux-aarch64.dependencies]
arch-pkg = "arm"
arm-only = "*"

[environments]
sim = { features = ["sim"] }

[feature.sim.dependencies]
feature-shared = "1"

[feature.sim.target.linux-64.dependencies]
feature-arch = "64"

[feature.sim.target.linux-aarch64.dependencies]
feature-arch = "arm"
arm-feature-only = "*"
"#,
        );

        let x86 = ws.effective_dependencies_for_target("sim", "linux-64");
        assert_eq!(x86.get("shared").map(String::as_str), Some("1"));
        assert_eq!(x86.get("arch-pkg").map(String::as_str), Some("64"));
        assert_eq!(x86.get("feature-arch").map(String::as_str), Some("64"));
        assert!(x86.contains_key("x86-only"));
        assert!(!x86.contains_key("arm-only"));
        assert!(!x86.contains_key("arm-feature-only"));

        let arm = ws.effective_dependencies_for_target("sim", "linux-aarch64");
        assert_eq!(arm.get("shared").map(String::as_str), Some("1"));
        assert_eq!(arm.get("arch-pkg").map(String::as_str), Some("arm"));
        assert_eq!(arm.get("feature-arch").map(String::as_str), Some("arm"));
        assert!(arm.contains_key("arm-only"));
        assert!(arm.contains_key("arm-feature-only"));
        assert!(!arm.contains_key("x86-only"));
        assert_eq!(
            ws.find_declaring_feature_for_target("sim", "arm-only", "linux-aarch64")
                .as_deref(),
            Some("default")
        );
        assert_eq!(
            ws.find_declaring_feature_for_target("sim", "arm-feature-only", "linux-aarch64")
                .as_deref(),
            Some("sim")
        );
        assert_eq!(
            ws.find_declaring_feature_for_target("sim", "arm-only", "linux-64"),
            None
        );
    }

    #[test]
    fn target_dependency_selectors_apply_generic_before_exact() {
        let ws = ws_toml(
            r#"
[dependencies]
base = "1"
arch-pkg = "base"

[target.unix.dependencies]
unix-only = "*"
arch-pkg = "unix"

[target.linux.dependencies]
linux-only = "*"
arch-pkg = "linux"

[target.linux-aarch64.dependencies]
arm-only = "*"
arch-pkg = "arm"

[environments]
sim = { features = ["sim"] }

[feature.sim.target.unix.dependencies]
feature-unix = "*"
feature-arch = "unix"

[feature.sim.target.linux.dependencies]
feature-linux = "*"
feature-arch = "linux"

[feature.sim.target.linux-aarch64.dependencies]
feature-arm = "*"
feature-arch = "arm"
"#,
        );

        let x86 = ws.effective_dependencies_for_target("sim", "linux-64");
        assert_eq!(x86.get("arch-pkg").map(String::as_str), Some("linux"));
        assert_eq!(x86.get("feature-arch").map(String::as_str), Some("linux"));
        assert!(x86.contains_key("unix-only"));
        assert!(x86.contains_key("linux-only"));
        assert!(x86.contains_key("feature-unix"));
        assert!(x86.contains_key("feature-linux"));
        assert!(!x86.contains_key("arm-only"));
        assert!(!x86.contains_key("feature-arm"));

        let arm = ws.effective_dependencies_for_target("sim", "linux-aarch64");
        assert_eq!(arm.get("arch-pkg").map(String::as_str), Some("arm"));
        assert_eq!(arm.get("feature-arch").map(String::as_str), Some("arm"));
        for expected in [
            "unix-only",
            "linux-only",
            "arm-only",
            "feature-unix",
            "feature-linux",
            "feature-arm",
        ] {
            assert!(arm.contains_key(expected), "missing {expected}");
        }

        let mac = ws.effective_dependencies_for_target("sim", "osx-arm64");
        assert_eq!(mac.get("arch-pkg").map(String::as_str), Some("unix"));
        assert_eq!(mac.get("feature-arch").map(String::as_str), Some("unix"));
        assert!(mac.contains_key("unix-only"));
        assert!(!mac.contains_key("linux-only"));
    }

    #[test]
    fn unix_target_selector_includes_freebsd_and_emscripten_but_not_wasi_or_zos() {
        let ws = ws_toml(
            r#"
[target.unix.dependencies]
top-unix = "1"
top-unix-path = { path = "./top-unix" }
[target.unix.pypi-dependencies]
top-unix-pypi = "==1"

[feature.sim.target.unix.dependencies]
feature-unix = "1"
feature-unix-path = { path = "./feature-unix" }
[feature.sim.target.unix.pypi-dependencies]
feature-unix-pypi = "==2"

[environments]
sim = { features = ["sim"] }
"#,
        );

        for subdir in ["freebsd-64", "freebsd-arm64", "emscripten-wasm32"] {
            let dependencies = ws.effective_dependencies_for_target("sim", subdir);
            assert_eq!(
                dependencies.get("top-unix").map(String::as_str),
                Some("1"),
                "top-level Unix dependency missing for {subdir}"
            );
            assert_eq!(
                dependencies.get("feature-unix").map(String::as_str),
                Some("1"),
                "feature Unix dependency missing for {subdir}"
            );

            let paths = ws.effective_path_dependencies_for_target("sim", subdir);
            assert_eq!(
                paths.get("top-unix-path").map(String::as_str),
                Some("./top-unix")
            );
            assert_eq!(
                paths.get("feature-unix-path").map(String::as_str),
                Some("./feature-unix")
            );

            let pypi = ws.effective_pypi_dependencies_for_target("sim", subdir);
            assert_eq!(pypi.get("top-unix-pypi"), Some(&vec!["==1".to_string()]));
            assert_eq!(
                pypi.get("feature-unix-pypi"),
                Some(&vec!["==2".to_string()])
            );
        }

        for subdir in ["wasi-wasm32", "zos-z"] {
            assert!(
                ws.effective_dependencies_for_target("sim", subdir)
                    .is_empty(),
                "Unix selector unexpectedly matched {subdir}"
            );
            assert!(
                ws.effective_path_dependencies_for_target("sim", subdir)
                    .is_empty(),
                "Unix path selector unexpectedly matched {subdir}"
            );
            assert!(
                ws.effective_pypi_dependencies_for_target("sim", subdir)
                    .is_empty(),
                "Unix PyPI selector unexpectedly matched {subdir}"
            );
        }
    }

    #[test]
    fn named_and_glob_target_selectors_follow_declaration_order() {
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "p1", platform = "linux-64", glibc = "2.28" },
  { name = "p3", platform = "linux-64", glibc = "2.35" },
]
[feature.sim.dependencies]
foo = "base"
bar = "base"
[feature.sim.target.'p*'.dependencies]
foo = "glob-first"
[feature.sim.target.p3.dependencies]
foo = "exact-last"
bar = "exact-first"
[feature.sim.target.'*3'.dependencies]
bar = "glob-last"
[environments]
sim = { features = ["sim"], no-default-feature = true }
"#,
        );
        let p1 = ws.effective_dependencies_for_profile("sim", "p1", "linux-64");
        assert_eq!(p1.get("foo").map(String::as_str), Some("glob-first"));
        assert_eq!(p1.get("bar").map(String::as_str), Some("base"));

        let p3 = ws.effective_dependencies_for_profile("sim", "p3", "linux-64");
        assert_eq!(p3.get("foo").map(String::as_str), Some("exact-last"));
        assert_eq!(p3.get("bar").map(String::as_str), Some("glob-last"));
    }

    #[test]
    fn inline_target_maps_follow_declaration_order() {
        let ws = ws_toml(
            r#"
target = { p1 = { dependencies = { root-order = "exact-first" } }, "p*" = { dependencies = { root-order = "glob-last" } } }

[workspace]
platforms = [
  { name = "p1", platform = "linux-64", glibc = "2.28" },
]
[feature.sim]
target = { p1 = { dependencies = { feature-order = "exact-first" } }, "p*" = { dependencies = { feature-order = "glob-last" } } }
[environments]
sim = { features = ["sim"] }
"#,
        );

        let p1 = ws.effective_dependencies_for_profile("sim", "p1", "linux-64");
        assert_eq!(p1.get("root-order").map(String::as_str), Some("glob-last"));
        assert_eq!(
            p1.get("feature-order").map(String::as_str),
            Some("glob-last")
        );
    }

    #[test]
    fn toml_1_1_multiline_inline_tables_preserve_target_order() {
        let ws = ws_toml(
            r#"
target = {
  p1 = {
    dependencies = {
      root-order = "exact-first",
    },
  },
  "p*" = {
    dependencies = {
      root-order = "glob-last",
    },
  },
}

[workspace]
platforms = [
  {
    name = "p1",
    platform = "linux-64",
    glibc = "2.28",
  },
]

[environments]
sim = {
  features = [],
  no-default-feature = false,
}
"#,
        );

        assert_eq!(ws.target_profiles[0].name, "p1");
        let p1 = ws.effective_dependencies_for_profile("sim", "p1", "linux-64");
        assert_eq!(p1.get("root-order").map(String::as_str), Some("glob-last"));
    }

    #[test]
    fn named_target_registry_override_is_local_to_that_profile() {
        let tmp = temp_workspace("named-target-source");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "p1", platform = "linux-64", glibc = "2.28" },
  { name = "p3", platform = "linux-64", glibc = "2.35" },
]
[feature.source.dependencies]
demo-pack = { path = "./demo-pack" }
[feature.source.target.p3.dependencies]
demo-pack = ">=2"
[environments]
sim = { features = ["source"], no-default-feature = true }
"#,
        );
        let p1 = ws
            .target_profiles
            .iter()
            .find(|profile| profile.name == "p1")
            .unwrap();
        let p3 = ws
            .target_profiles
            .iter()
            .find(|profile| profile.name == "p3")
            .unwrap();
        assert_eq!(
            ws.try_consuming_envs_for_profile(&tmp, &source, p1)
                .unwrap(),
            BTreeSet::from(["sim".to_string()])
        );
        assert!(
            ws.try_consuming_envs_for_profile(&tmp, &source, p3)
                .unwrap()
                .is_empty()
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn same_contract_aliases_with_divergent_target_views_fail_closed() {
        let tmp = temp_workspace("target-alias-divergence");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "p1", platform = "linux-64", glibc = "2.28" },
  { name = "p1-alias", platform = "linux-64", glibc = "2.28" },
]
[feature.source.dependencies]
demo-pack = { path = "./demo-pack" }
[feature.source.target.p1.dependencies]
profile-only = "1"
[feature.source.target.p1-alias.dependencies]
profile-only = "2"
[feature.old]
platforms = ["p1"]
[feature.alias]
platforms = ["p1-alias"]
[environments]
old = { features = ["source", "old"], no-default-feature = true }
alias = { features = ["source", "alias"], no-default-feature = true }
"#,
        );
        let error = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("divergent target selector views"), "{error}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn target_scoped_pypi_aliases_fail_closed_and_exact_scope_keeps_its_requirements() {
        let tmp = temp_workspace("target-pypi-alias-divergence");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "p1", platform = "linux-64", glibc = "2.28" },
  { name = "p1-alias", platform = "linux-64", glibc = "2.28" },
]
[feature.source.dependencies]
demo-pack = { path = "./demo-pack" }
[feature.source.pypi-dependencies]
profile-wheel = ">=0"
[feature.source.target.p1.pypi-dependencies]
profile-wheel = "==1"
[feature.source.target.p1-alias.pypi-dependencies]
profile-wheel = "==2"
[feature.old]
platforms = ["p1"]
[feature.alias]
platforms = ["p1-alias"]
[environments]
old = { features = ["source", "old"], no-default-feature = true }
alias = { features = ["source", "alias"], no-default-feature = true }
"#,
        );

        let error = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("divergent target selector views"), "{error}");

        let envelope = WorkspaceTargetEnvelope {
            schema: 1,
            environment: "old".to_string(),
            profile: WorkspaceTargetEnvelopeProfile {
                name: "p1".to_string(),
                subdir: "linux-64".to_string(),
                virtual_packages: Vec::new(),
                detected_virtual_packages: vec![
                    "glibc=2.28".to_string(),
                    "linux=4.18".to_string(),
                    "__archspec=1=x86_64".to_string(),
                    "__unix".to_string(),
                ],
            },
        };
        let target = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", Some(&envelope))
            .unwrap()
            .unwrap();
        assert_eq!(
            ws.effective_pypi_dependencies_for_resolved_env("old", &target)
                .unwrap()
                .get("profile-wheel"),
            Some(&vec!["==1".to_string()]),
        );
        let fingerprint = ws.solve_fingerprint_for_resolved_target(&tmp, &source, &target);
        assert!(fingerprint.contains("pypi-dep:profile-wheel===1"));
        assert!(!fingerprint.contains("pypi-dep:profile-wheel===2"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn source_consumer_with_no_reachable_profile_fails_closed() {
        let tmp = temp_workspace("target-unreachable-consumer");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = ["linux-64"]
[feature.broken]
platforms = []
[feature.broken.dependencies]
demo-pack = { path = "./demo-pack" }
[environments]
broken = { features = ["broken"], no-default-feature = true }
"#,
        );
        let error = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", None)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("no reachable target profile: broken"),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn target_dependencies_overwrite_across_version_and_path_kinds() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-ws-target-kind-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let base_source = tmp.join("base-source");
        let arm_source = tmp.join("arm-source");
        std::fs::create_dir_all(&base_source).unwrap();
        std::fs::create_dir_all(&arm_source).unwrap();
        let ws = ws_toml(
            r#"
[dependencies]
swap-to-path = "1"
swap-to-version = { path = "./base-source" }

[target.linux-aarch64.dependencies]
swap-to-path = { path = "./arm-source" }
swap-to-version = "2"

[environments]
default = []
"#,
        );

        let x86 = ws.effective_dependencies_for_target("default", "linux-64");
        assert_eq!(x86.get("swap-to-path").map(String::as_str), Some("1"));
        assert!(!x86.contains_key("swap-to-version"));
        let arm = ws.effective_dependencies_for_target("default", "linux-aarch64");
        assert!(!arm.contains_key("swap-to-path"));
        assert_eq!(arm.get("swap-to-version").map(String::as_str), Some("2"));

        assert!(
            ws.discover_outputs_for_source_for_target(&tmp, &arm_source, "linux-64")
                .is_empty()
        );
        let arm_outputs =
            ws.discover_outputs_for_source_for_target(&tmp, &arm_source, "linux-aarch64");
        assert_eq!(arm_outputs.len(), 1);
        assert_eq!(arm_outputs[0].name, "swap-to-path");

        let x86_outputs = ws.discover_outputs_for_source_for_target(&tmp, &base_source, "linux-64");
        assert_eq!(x86_outputs.len(), 1);
        assert_eq!(x86_outputs[0].name, "swap-to-version");
        assert!(
            ws.discover_outputs_for_source_for_target(&tmp, &base_source, "linux-aarch64")
                .is_empty()
        );

        assert_eq!(
            ws.find_declaring_feature_for_target("default", "swap-to-path", "linux-aarch64"),
            None
        );
        assert_eq!(
            ws.find_declaring_feature_for_target("default", "swap-to-version", "linux-aarch64")
                .as_deref(),
            Some("default")
        );

        std::fs::remove_dir_all(tmp).ok();
    }

    #[test]
    fn target_path_dependencies_are_discovered_only_for_requested_target() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-ws-target-path-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let source = tmp.join("jetson-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[environments]
jetson = { features = ["jetson"] }

[feature.jetson.target.linux-aarch64.dependencies]
jetson-pack = { path = "./jetson-pack" }
"#,
        );

        assert!(
            ws.discover_outputs_for_source_for_target(&tmp, &source, "linux-64")
                .is_empty()
        );
        let arm = ws.discover_outputs_for_source_for_target(&tmp, &source, "linux-aarch64");
        assert_eq!(arm.len(), 1);
        assert_eq!(arm[0].name, "jetson-pack");
        assert_eq!(arm[0].declaring_features, vec!["jetson"]);
        assert_eq!(arm[0].envs, vec!["jetson"]);

        std::fs::remove_dir_all(tmp).ok();
    }

    // v0.37.0 D1: system-requirements parsing + effective rollup.
    // (Restored after an accidental revert; mirrors the gigastrap layout
    // where [feature.gpu] declares cuda=12 and [feature.isaaclab] libc.)
    #[test]
    fn effective_system_requirements_unions_features_over_top_level() {
        let ws = ws_toml(
            r#"
[system-requirements]
cuda = "11"

[environments]
gpu = { features = ["g", "lab"] }

[feature.g]
[feature.g.system-requirements]
cuda = "12"

[feature.lab]
[feature.lab.system-requirements]
libc = { family = "glibc", version = "2.35" }
"#,
        );
        let sr = ws.effective_system_requirements("gpu");
        // feature cuda (12) overrides top-level (11); table form -> version.
        assert_eq!(sr.get("cuda").map(String::as_str), Some("12"));
        assert_eq!(sr.get("libc").map(String::as_str), Some("2.35"));
    }

    #[test]
    fn effective_legacy_system_requirements_keep_highest_active_floor() {
        let ws = ws_toml(
            r#"
[workspace]
platforms = ["linux-64"]

[feature.high.system-requirements]
cuda = "12.6"
libc = "2.35"
[feature.low.system-requirements]
cuda = "12.0"
libc = "2.28"

[environments]
gpu = { features = ["high", "low"], no-default-feature = true }
"#,
        );
        let requirements = ws.effective_system_requirements_for_target("gpu", "linux-64");
        assert_eq!(requirements.get("cuda").map(String::as_str), Some("12.6"));
        assert_eq!(requirements.get("libc").map(String::as_str), Some("2.35"));
    }

    #[test]
    fn effective_system_requirements_no_default_feature_skips_top_level() {
        let ws = ws_toml(
            r#"
[system-requirements]
cuda = "12"

[environments]
standalone = { features = ["f"], no-default-feature = true }

[feature.f]
[feature.f.system-requirements]
libc = "2.39"
"#,
        );
        let sr = ws.effective_system_requirements("standalone");
        assert_eq!(sr.get("libc").map(String::as_str), Some("2.39"));
        assert!(!sr.contains_key("cuda")); // top-level skipped
    }

    #[test]
    fn effective_system_requirements_rich_platforms_win_over_legacy() {
        let ws = ws_toml(&format!(
            r#"
[workspace]
platforms = [{{ platform = "{plat}", glibc = "2.35", cuda = "12.0" }}]

[system-requirements]
libc = "2.50"
cuda = "11"

[environments]
default = []
"#,
            plat = crate::glibc::current_pixi_platform()
        ));
        let sr = ws.effective_system_requirements("default");
        // Rich platform declarations replace the deprecated table.
        assert_eq!(sr.get("libc").map(String::as_str), Some("2.35"));
        assert_eq!(sr.get("cuda").map(String::as_str), Some("12"));
    }

    #[test]
    fn effective_system_requirements_rich_platform_other_platform_ignored() {
        let other = if crate::glibc::current_pixi_platform() == "linux-64" {
            "linux-aarch64"
        } else {
            "linux-64"
        };
        let ws = ws_toml(&format!(
            r#"
[workspace]
platforms = [{{ platform = "{other}", glibc = "2.39", cuda = "13" }}]

[system-requirements]
libc = "2.34"

[environments]
default = []
"#,
        ));
        let sr = ws.effective_system_requirements("default");
        // Legacy declaration stands; the rich entry targets another platform.
        assert_eq!(sr.get("libc").map(String::as_str), Some("2.34"));
        assert!(!sr.contains_key("cuda"));
    }

    #[test]
    fn target_system_requirements_select_exact_rich_platform() {
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { platform = "linux-64", glibc = "2.35", cuda = "12" },
  { platform = "linux-aarch64", glibc = "2.39", cuda = "13" },
]

[system-requirements]
libc = "2.17"
cuda = "11"

[environments]
default = []
"#,
        );
        let x86 = ws.effective_system_requirements_for_target("default", "linux-64");
        let arm = ws.effective_system_requirements_for_target("default", "linux-aarch64");
        assert_eq!(x86.get("libc").map(String::as_str), Some("2.35"));
        assert_eq!(x86.get("cuda").map(String::as_str), Some("12"));
        assert_eq!(arm.get("libc").map(String::as_str), Some("2.39"));
        assert_eq!(arm.get("cuda").map(String::as_str), Some("13"));
        assert!(
            ws.effective_system_requirements_for_target("missing", "linux-aarch64")
                .is_empty()
        );
        assert_eq!(
            ws.declared_glibc_for_target("linux-64", None),
            Some((2, 35))
        );
        assert_eq!(
            ws.declared_glibc_for_target("linux-aarch64", None),
            Some((2, 39))
        );
    }

    #[test]
    fn rich_platform_cuda_only_entry_is_parsed() {
        let ws = ws_toml(
            r#"
[workspace]
platforms = ["linux-aarch64", { platform = "linux-64", cuda = "12" }]
"#,
        );
        assert_eq!(
            ws.platform_cuda.get("linux-64").map(String::as_str),
            Some("12")
        );
        assert!(ws.platform_glibc.is_empty());
    }

    #[test]
    fn rich_platform_cuda_table_and_subdir_name_shorthand_are_parsed() {
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "gpu", platform = "linux-64", cuda = { driver = "12.0", arch = "8.6" } },
  { name = "osx-arm64" },
]
"#,
        );
        assert_eq!(ws.target_profiles[0].name, "gpu");
        assert_eq!(
            ws.target_profiles[0]
                .declared_virtual_packages
                .get("cuda")
                .map(String::as_str),
            Some("12")
        );
        assert_eq!(
            ws.target_profiles[0]
                .declared_virtual_packages
                .get("cuda_arch")
                .map(String::as_str),
            Some("8.6")
        );
        assert_eq!(ws.target_profiles[1].name, "osx-arm64");
        assert_eq!(ws.target_profiles[1].subdir, "osx-arm64");
    }

    #[test]
    fn unnamed_cuda_table_profile_name_includes_cuda_arch_in_pixi_order() {
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { platform = "linux-64", cuda = { driver = "12.0", arch = "8.6" }, glibc = "2.35", linux = "4.18" },
]
"#,
        );
        assert_eq!(
            ws.target_profiles[0].name,
            "linux-64-cuda-12-0-glibc-2-35-cuda-arch-8-6"
        );
    }

    #[test]
    fn unnamed_profile_names_elide_pixi073_subdir_defaults() {
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { platform = "linux-ppc64le", cuda = "12.0", glibc = "2.28", linux = "4.18", archspec = "ppc64le" },
  { platform = "linux-loongarch64", glibc = "2.28", linux = "4.18", archspec = "loongarch64" },
  { platform = "osx-arm64", macos = "13.0", archspec = "m1" },
  { platform = "win-64", windows = "10.0", archspec = "x86_64" },
]
"#,
        );
        assert_eq!(
            ws.target_profiles
                .iter()
                .map(|profile| profile.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "linux-ppc64le-cuda-12-0",
                "linux-loongarch64",
                "osx-arm64",
                "win-64",
            ],
        );
        assert_eq!(
            pixi_073_default_archspec("linux-loongarch64"),
            Some("loongarch64")
        );
    }

    #[test]
    fn unnamed_profile_names_distinguish_raw_and_friendly_pixi073_shapes() {
        let profile_name = |declaration: &str| {
            ws_toml(&format!(
                "[workspace]\nplatforms = [{{ platform = \"linux-64\", {declaration} }}]\n"
            ))
            .target_profiles[0]
                .name
                .clone()
        };

        assert_eq!(profile_name("archspec = \"x86_64\""), "linux-64");
        assert_eq!(
            profile_name("archspec = \"0=x86_64\""),
            "linux-64-archspec-0-x86-64"
        );
        assert_eq!(profile_name("__archspec = \"0=x86_64\""), "linux-64");
        assert_eq!(
            profile_name("__archspec = \"1=x86_64\""),
            "linux-64-archspec-1-x86-64"
        );
        assert_eq!(profile_name("__unix = \"0=0\""), "linux-64");
        assert_eq!(profile_name("__unix = \"0\""), "linux-64-unix-0");

        let raw = ws_toml(
            r#"
[workspace]
platforms = [
  { platform = "linux-64", __unix = "0=0", __archspec = "0=x86_64", __glibc = "2.28=0", __linux = "4.18=0" },
  { platform = "linux-64", __unix = "0", __archspec = "1=x86_64", __glibc = "2.35=0", __linux = "5.15=0" },
]
"#,
        );
        assert_eq!(
            raw.target_profiles
                .iter()
                .map(|profile| profile.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "linux-64-glibc-2-28-linux-4-18",
                "linux-64-glibc-2-35-linux-5-15-archspec-1-x86-64-unix-0",
            ]
        );
        assert_eq!(
            raw.target_profiles[0].declared_virtual_packages,
            BTreeMap::from([
                ("archspec".to_string(), "x86_64".to_string()),
                ("glibc".to_string(), "2.28".to_string()),
                ("linux".to_string(), "4.18".to_string()),
                ("unix".to_string(), String::new()),
            ])
        );

        let remaining = ws_toml(
            r#"
[workspace]
platforms = [
  { platform = "freebsd-64", __unix = "0=0" },
  { platform = "emscripten-wasm32", __unix = "0=0" },
  { platform = "wasi-wasm32" },
  { platform = "zos-z" },
]
"#,
        );
        assert_eq!(
            remaining
                .target_profiles
                .iter()
                .map(|profile| profile.name.as_str())
                .collect::<Vec<_>>(),
            vec!["freebsd-64", "emscripten-wasm32", "wasi-wasm32", "zos-z"]
        );
    }

    #[test]
    fn rich_platform_osx_and_raw_win_aliases_use_pixi_json_names() {
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "mac-test", platform = "osx-64", osx = "14.2" },
  { name = "win-test", platform = "win-64", __win = "7.7" },
]
"#,
        );
        assert_eq!(
            ws.target_profiles[0].declared_virtual_packages,
            BTreeMap::from([("macos".to_string(), "14.2".to_string())])
        );
        assert_eq!(
            ws.target_profiles[1].declared_virtual_packages,
            BTreeMap::from([("windows".to_string(), "7.7".to_string())])
        );
    }

    #[test]
    fn exact_virtual_package_parser_rejects_names_the_solver_cannot_represent() {
        let error = parse_virtual_package_lines(&["bad/name=1".to_string()]).unwrap_err();
        assert!(
            error.to_string().contains("valid conda package name"),
            "{error:#}"
        );
    }

    #[test]
    fn named_rich_profiles_and_feature_assignments_are_retained_without_last_write() {
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "p1", platform = "linux-64", cuda = 12, glibc = "2.28", linux = "4.18" },
  { name = "p3", platform = "linux-64", cuda = "12", glibc = "2.35", linux = "4.18" },
  "linux-aarch64",
]

[feature.old]
platforms = ["p1"]

[feature.new]
platforms = ["p3"]

[feature.nowhere]
platforms = []
"#,
        );

        assert_eq!(
            ws.target_profiles
                .iter()
                .map(|profile| (profile.name.as_str(), profile.subdir.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("p1", "linux-64"),
                ("p3", "linux-64"),
                ("linux-aarch64", "linux-aarch64"),
            ]
        );
        assert_eq!(
            ws.target_profiles[0]
                .declared_virtual_packages
                .get("glibc")
                .map(String::as_str),
            Some("2.28")
        );
        assert_eq!(
            ws.features["old"].platforms.as_deref(),
            Some(["p1".to_string()].as_slice())
        );
        assert_eq!(ws.features["nowhere"].platforms, Some(Vec::new()));
        assert!(
            !ws.platform_glibc.contains_key("linux-64"),
            "same-subdir rich profiles must not collapse to the last declaration"
        );
    }

    #[test]
    fn unnamed_rich_profiles_use_pixi_synthesized_names_without_overwriting() {
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  "linux-64",
  { platform = "linux-64", cuda = "12.0", glibc = "2.28", linux = "4.18" },
  { platform = "linux-64", cuda = "12", glibc = "2.35", linux = "4.18" },
]

[feature.old]
platforms = ["linux-64-cuda-12-0"]
[feature.new]
platforms = ["linux-64-cuda-12-glibc-2-35"]
[environments]
old = { features = ["old"], no-default-feature = true }
new = { features = ["new"], no-default-feature = true }
"#,
        );
        assert_eq!(
            ws.target_profiles
                .iter()
                .map(|profile| profile.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "linux-64",
                "linux-64-cuda-12-0",
                "linux-64-cuda-12-glibc-2-35",
            ]
        );
        assert_eq!(
            ws.environment_profiles("old").unwrap()[0].name,
            "linux-64-cuda-12-0"
        );
        assert_eq!(
            ws.environment_profiles("new").unwrap()[0].name,
            "linux-64-cuda-12-glibc-2-35"
        );
    }

    #[test]
    fn environment_profile_selection_matches_pixi_named_profile_intersection() {
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "p1", platform = "linux-64", glibc = "2.28" },
  { name = "p3", platform = "linux-64", glibc = "2.35" },
]

[feature.p1]
platforms = ["p1"]
[feature.p3]
platforms = ["p3"]
[feature.generic]
platforms = ["linux-64"]
[feature.empty]
platforms = []
[feature.free]

[environments]
old = { features = ["p1"] }
new = { features = ["p3"] }
generic = { features = ["generic"] }
free = { features = ["free"] }
conflict = { features = ["p1", "p3"] }
empty = { features = ["empty"] }
"#,
        );
        let names = |env: &str| {
            ws.environment_profiles(env)
                .unwrap()
                .into_iter()
                .map(|profile| profile.name.as_str())
                .collect::<Vec<_>>()
        };
        assert_eq!(names("old"), vec!["p1"]);
        assert_eq!(names("new"), vec!["p3"]);
        assert_eq!(names("generic"), vec!["p1", "p3"]);
        assert_eq!(names("free"), vec!["p1", "p3"]);
        assert!(names("conflict").is_empty());
        assert!(names("empty").is_empty());
    }

    #[test]
    fn feature_referenced_bare_subdirs_extend_legacy_workspace_in_declaration_order() {
        let tmp = temp_workspace("feature-referenced-subdirs");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = ["linux-64"]
[feature.z]
platforms = ["win-64", "osx-arm64"]
[feature.z.dependencies]
demo-pack = { path = "./demo-pack" }
[feature.a]
platforms = ["linux-aarch64", "win-64"]
[environments]
mac = { features = ["z"], no-default-feature = true }
"#,
        );
        assert_eq!(
            ws.target_profiles
                .iter()
                .map(|profile| profile.name.as_str())
                .collect::<Vec<_>>(),
            vec!["linux-64", "win-64", "osx-arm64", "linux-aarch64"],
        );

        let direct = ws
            .resolve_target_for_source(&tmp, &source, "osx-arm64", None)
            .unwrap()
            .unwrap();
        assert_eq!(direct.profiles, vec!["osx-arm64"]);
        assert_eq!(direct.environments, vec!["mac"]);
        assert!(direct.contract.declared_virtual_packages.is_empty());

        let envelope = WorkspaceTargetEnvelope {
            schema: 1,
            environment: "mac".to_string(),
            profile: WorkspaceTargetEnvelopeProfile {
                name: "osx-arm64".to_string(),
                subdir: "osx-arm64".to_string(),
                virtual_packages: Vec::new(),
                detected_virtual_packages: vec![
                    "cuda=12.9".to_string(),
                    "macos=13".to_string(),
                    "__archspec=1=m1".to_string(),
                    "__unix".to_string(),
                ],
            },
        };
        let exact = ws
            .resolve_target_for_source(&tmp, &source, "osx-arm64", Some(&envelope))
            .unwrap()
            .unwrap();
        assert_eq!(exact.profiles, vec!["osx-arm64"]);
        assert_eq!(exact.environments, vec!["mac"]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn bare_subdir_feature_composes_with_exact_rich_profile() {
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  "linux-64",
  { name = "p1", platform = "linux-64", glibc = "2.28" },
]
[feature.linux]
platforms = ["linux-64"]
[feature.p1]
platforms = ["p1"]
[environments]
plain = { features = ["linux"] }
rich = { features = ["linux", "p1"] }
"#,
        );
        let names = |env: &str| {
            ws.environment_profiles(env)
                .unwrap()
                .into_iter()
                .map(|profile| profile.name.as_str())
                .collect::<Vec<_>>()
        };
        assert_eq!(names("plain"), vec!["linux-64"]);
        assert_eq!(names("rich"), vec!["p1"]);
    }

    #[test]
    fn source_target_inference_distinguishes_p1_p3_and_fails_closed_on_ambiguity() {
        let tmp = temp_workspace("target-inference");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "p1", platform = "linux-64", cuda = "12", glibc = "2.28", linux = "4.18" },
  { name = "p3", platform = "linux-64", cuda = "12", glibc = "2.35", linux = "4.18" },
]

[feature.old]
platforms = ["p1"]
[feature.old.dependencies]
demo-pack = { path = "./demo-pack" }

[feature.new]
platforms = ["p3"]
[feature.new.dependencies]
demo-pack = { path = "./demo-pack" }

[environments]
old = { features = ["old"], no-default-feature = true }
new = { features = ["new"], no-default-feature = true }
"#,
        );
        let error = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ambiguous workspace target contracts"));
        assert!(error.contains("profiles=[p1] environments=[old]"));
        assert!(error.contains("profiles=[p3] environments=[new]"));
        assert!(error.contains("glibc=2.28"));
        assert!(error.contains("glibc=2.35"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn source_target_inference_accepts_one_unique_profile_contract() {
        let tmp = temp_workspace("target-unique");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "p1", platform = "linux-64", cuda = "12", glibc = "2.28", linux = "4.18" },
  { name = "p3", platform = "linux-64", cuda = "12", glibc = "2.35", linux = "4.18" },
]

[feature.old]
platforms = ["p1"]
[feature.old.dependencies]
demo-pack = { path = "./demo-pack" }

[feature.new]
platforms = ["p3"]
[feature.new.dependencies]
unrelated = "*"

[environments]
old = { features = ["old"], no-default-feature = true }
new = { features = ["new"], no-default-feature = true }
"#,
        );
        let target = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", None)
            .unwrap()
            .unwrap();
        assert_eq!(target.profiles, vec!["p1"]);
        assert_eq!(target.environments, vec!["old"]);
        assert_eq!(target.contract.effective_glibc(), Some((2, 28)));
        assert!(target.contract.detected_virtual_packages.is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn source_target_inference_distinguishes_environment_legacy_requirements() {
        let tmp = temp_workspace("target-legacy-ambiguity");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = ["linux-64"]

[feature.old]
platforms = ["linux-64"]
[feature.old.dependencies]
demo-pack = { path = "./demo-pack" }
[feature.old.system-requirements]
libc = "2.28"

[feature.new]
platforms = ["linux-64"]
[feature.new.dependencies]
demo-pack = { path = "./demo-pack" }
[feature.new.system-requirements]
libc = "2.35"

[environments]
old = { features = ["old"], no-default-feature = true }
new = { features = ["new"], no-default-feature = true }
"#,
        );
        let error = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ambiguous workspace target contracts"));
        assert!(error.contains("environments=[old]"));
        assert!(error.contains("environments=[new]"));
        assert!(error.contains("glibc=2.28"));
        assert!(error.contains("glibc=2.35"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn legacy_system_requirements_follow_pixi_platform_applicability() {
        let tmp = temp_workspace("target-legacy-platform-applicability");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  "linux-64",
  "osx-64",
  "win-64",
  "freebsd-64",
  "emscripten-wasm32",
  "wasi-wasm32",
  "zos-z",
]
[dependencies]
demo-pack = { path = "./demo-pack" }
[system-requirements]
cuda = "12.6"
libc = { family = "glibc", version = "2.35" }
linux = "5.15"
macos = "14.2"
archspec = "x86_64_v3"
"#,
        );

        let linux = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", None)
            .unwrap()
            .unwrap();
        assert_eq!(
            linux.contract.declared_virtual_packages,
            BTreeMap::from([
                ("cuda".to_string(), "12.6".to_string()),
                ("glibc".to_string(), "2.35".to_string()),
                ("linux".to_string(), "5.15".to_string()),
            ])
        );

        let mac = ws
            .resolve_target_for_source(&tmp, &source, "osx-64", None)
            .unwrap()
            .unwrap();
        assert_eq!(
            mac.contract.declared_virtual_packages,
            BTreeMap::from([("macos".to_string(), "14.2".to_string())])
        );

        let windows = ws
            .resolve_target_for_source(&tmp, &source, "win-64", None)
            .unwrap()
            .unwrap();
        assert_eq!(
            windows.contract.declared_virtual_packages,
            BTreeMap::from([("cuda".to_string(), "12.6".to_string())])
        );

        for subdir in ["freebsd-64", "emscripten-wasm32", "wasi-wasm32", "zos-z"] {
            let target = ws
                .resolve_target_for_source(&tmp, &source, subdir, None)
                .unwrap()
                .unwrap();
            assert_eq!(
                target.contract.declared_virtual_packages,
                BTreeMap::from([("cuda".to_string(), "12.6".to_string())]),
                "Pixi migrates legacy CUDA to {subdir}"
            );
        }

        let envelope = WorkspaceTargetEnvelope {
            schema: 1,
            environment: "default".to_string(),
            profile: WorkspaceTargetEnvelopeProfile {
                name: "osx-64-macos-14-2".to_string(),
                subdir: "osx-64".to_string(),
                virtual_packages: vec!["macos=14.2".to_string()],
                detected_virtual_packages: vec![
                    "cuda=12.9".to_string(),
                    "macos=14.2".to_string(),
                    "__archspec=1=x86_64".to_string(),
                    "__unix".to_string(),
                ],
            },
        };
        let exact = ws
            .resolve_target_for_source(&tmp, &source, "osx-64", Some(&envelope))
            .unwrap()
            .unwrap();
        assert_eq!(
            exact.contract.declared_virtual_packages,
            mac.contract.declared_virtual_packages
        );
        assert!(
            !exact
                .contract
                .declared_virtual_packages
                .contains_key("archspec")
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn rich_profile_requirement_wins_over_environment_legacy_requirement() {
        let tmp = temp_workspace("target-rich-wins-legacy");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "p3", platform = "linux-64", glibc = "2.35", linux = "4.18" },
]

[feature.old]
platforms = ["p3"]
[feature.old.dependencies]
demo-pack = { path = "./demo-pack" }
[feature.old.system-requirements]
libc = "2.28"

[feature.new]
platforms = ["p3"]
[feature.new.dependencies]
demo-pack = { path = "./demo-pack" }
[feature.new.system-requirements]
libc = "2.40"

[environments]
old = { features = ["old"], no-default-feature = true }
new = { features = ["new"], no-default-feature = true }
"#,
        );
        let target = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", None)
            .unwrap()
            .unwrap();
        assert_eq!(target.contract.effective_glibc(), Some((2, 35)));
        assert_eq!(target.profiles, vec!["p3"]);
        assert_eq!(target.environments, vec!["new", "old"]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn exact_target_envelope_is_validated_and_carries_detected_contract() {
        let tmp = temp_workspace("target-envelope");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "p1", platform = "linux-64", cuda = "12", glibc = "2.28", linux = "4.18" },
  { name = "p1-alias", platform = "linux-64", cuda = "12", glibc = "2.28", linux = "4.18" },
]

[feature.old]
platforms = ["p1"]
[feature.old.dependencies]
demo-pack = { path = "./demo-pack" }
[feature.alias]
platforms = ["p1-alias"]
[feature.alias.dependencies]
demo-pack = { path = "./demo-pack" }

[environments]
old = { features = ["old"], no-default-feature = true }
alias = { features = ["alias"], no-default-feature = true }
"#,
        );
        let envelope = WorkspaceTargetEnvelope {
            schema: 1,
            environment: "old".to_string(),
            profile: WorkspaceTargetEnvelopeProfile {
                name: "p1".to_string(),
                subdir: "linux-64".to_string(),
                // Pixi's JSON display elides subdir-default glibc/Linux from
                // the declared list even when the manifest spells them out.
                virtual_packages: vec!["cuda=12".to_string()],
                detected_virtual_packages: vec![
                    "cuda=12".to_string(),
                    "glibc=2.28".to_string(),
                    "linux=4.18".to_string(),
                    "__archspec=1=x86_64".to_string(),
                    "__unix".to_string(),
                ],
            },
        };
        let target = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", Some(&envelope))
            .unwrap()
            .unwrap();
        assert_eq!(target.contract.effective_glibc(), Some((2, 28)));
        assert_eq!(
            target.contract.detected_virtual_packages.get("archspec"),
            Some(&"1=x86_64".to_string())
        );
        assert_eq!(
            target.contract.detected_virtual_packages.get("unix"),
            Some(&String::new())
        );
        assert_eq!(target.profiles, vec!["p1"]);
        assert_eq!(target.environments, vec!["old"]);
        let scoped_outputs =
            ws.discover_outputs_for_source_for_resolved_target(&tmp, &source, &target);
        assert_eq!(scoped_outputs.len(), 1);
        assert_eq!(scoped_outputs[0].envs, vec!["old"]);

        let mut invalid = envelope;
        invalid.profile.detected_virtual_packages = vec!["glibc=2.28".to_string()];
        assert!(
            ws.resolve_target_for_source(&tmp, &source, "linux-64", Some(&invalid))
                .unwrap_err()
                .to_string()
                .contains("does not preserve declared")
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn resolved_target_inputs_keep_named_profile_selectors() {
        let tmp = temp_workspace("resolved-target-selector-inputs");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "p1", platform = "linux-64", glibc = "2.28" },
  { name = "p3", platform = "linux-64", glibc = "2.35" },
]
[dependencies]
demo-pack = { path = "./demo-pack" }
[target.p1.dependencies]
profile-input = "1"
[target.p3.dependencies]
profile-input = "3"
[feature.old]
platforms = ["p1"]
[feature.new]
platforms = ["p3"]
[environments]
old = { features = ["old"] }
new = { features = ["new"] }
"#,
        );
        let envelope = |environment: &str, name: &str, glibc: &str| WorkspaceTargetEnvelope {
            schema: 1,
            environment: environment.to_string(),
            profile: WorkspaceTargetEnvelopeProfile {
                name: name.to_string(),
                subdir: "linux-64".to_string(),
                virtual_packages: (glibc != "2.28")
                    .then(|| format!("glibc={glibc}"))
                    .into_iter()
                    .collect(),
                detected_virtual_packages: vec![
                    format!("glibc={glibc}"),
                    "linux=4.18".to_string(),
                    "__archspec=1=x86_64".to_string(),
                    "__unix".to_string(),
                ],
            },
        };
        let old = ws
            .resolve_target_for_source(
                &tmp,
                &source,
                "linux-64",
                Some(&envelope("old", "p1", "2.28")),
            )
            .unwrap()
            .unwrap();
        let new = ws
            .resolve_target_for_source(
                &tmp,
                &source,
                "linux-64",
                Some(&envelope("new", "p3", "2.35")),
            )
            .unwrap()
            .unwrap();

        let old_fingerprint = ws.solve_fingerprint_for_resolved_target(&tmp, &source, &old);
        assert!(old_fingerprint.contains("dep:profile-input=1"));
        assert!(!old_fingerprint.contains("dep:profile-input=3"));
        let new_fingerprint = ws.solve_fingerprint_for_resolved_target(&tmp, &source, &new);
        assert!(new_fingerprint.contains("dep:profile-input=3"));
        assert!(!new_fingerprint.contains("dep:profile-input=1"));
        assert_eq!(
            ws.union_effective_dependencies_for_resolved_target(&["new".to_string()], &new,)["profile-input"],
            vec!["3"]
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn exact_target_envelope_overlays_verified_environment_legacy_requirements() {
        let tmp = temp_workspace("target-envelope-legacy");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = ["linux-64"]

[feature.old]
platforms = ["linux-64"]
[feature.old.dependencies]
demo-pack = { path = "./demo-pack" }
[feature.old.system-requirements]
libc = "2.35"

[environments]
old = { features = ["old"], no-default-feature = true }
"#,
        );
        let envelope = WorkspaceTargetEnvelope {
            schema: 1,
            environment: "old".to_string(),
            profile: WorkspaceTargetEnvelopeProfile {
                name: "linux-64-glibc-2-35".to_string(),
                subdir: "linux-64".to_string(),
                virtual_packages: vec!["glibc=2.35".to_string()],
                detected_virtual_packages: vec![
                    "glibc=2.35".to_string(),
                    "linux=4.18".to_string(),
                    "__archspec=1=x86_64".to_string(),
                    "__unix".to_string(),
                ],
            },
        };
        let target = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", Some(&envelope))
            .unwrap()
            .unwrap();
        assert_eq!(target.contract.declared_glibc(), Some((2, 35)));
        assert_eq!(target.contract.effective_glibc(), Some((2, 35)));
        assert_eq!(
            target.contract.detected_virtual_packages.get("glibc"),
            Some(&"2.35".to_string())
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn exact_target_envelope_accepts_live_pixi073_jetson_composed_profile() {
        let tmp = temp_workspace("target-envelope-jetson-live");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = ["linux-aarch64"]

[feature.jetson]
platforms = ["linux-aarch64"]
[feature.jetson.dependencies]
demo-pack = { path = "./demo-pack" }
[feature.jetson.system-requirements]
cuda = "12.6"
libc = { family = "glibc", version = "2.35" }
linux = "5.15"

[environments]
jetson = { features = ["jetson"], no-default-feature = true }
"#,
        );
        let envelope = WorkspaceTargetEnvelope {
            schema: 1,
            environment: "jetson".to_string(),
            profile: WorkspaceTargetEnvelopeProfile {
                name: "linux-aarch64-cuda-12-6-glibc-2-35-linux-5-15".to_string(),
                subdir: "linux-aarch64".to_string(),
                virtual_packages: vec![
                    "cuda=12.6".to_string(),
                    "glibc=2.35".to_string(),
                    "linux=5.15".to_string(),
                ],
                detected_virtual_packages: vec![
                    "cuda=12.6".to_string(),
                    "glibc=2.35".to_string(),
                    "linux=5.15".to_string(),
                    "__archspec=1=aarch64".to_string(),
                    "__unix".to_string(),
                ],
            },
        };
        let target = ws
            .resolve_target_for_source(&tmp, &source, "linux-aarch64", Some(&envelope))
            .unwrap()
            .unwrap();
        assert_eq!(target.profiles, vec!["linux-aarch64"]);
        assert_eq!(target.environments, vec!["jetson"]);
        assert_eq!(
            target.contract.declared_virtual_packages,
            BTreeMap::from([
                ("cuda".to_string(), "12.6".to_string()),
                ("glibc".to_string(), "2.35".to_string()),
                ("linux".to_string(), "5.15".to_string()),
            ])
        );

        let mut wrong_arch = envelope;
        wrong_arch.profile.detected_virtual_packages = vec![
            "cuda=12.6".to_string(),
            "glibc=2.35".to_string(),
            "linux=5.15".to_string(),
            "__archspec=1=x86_64".to_string(),
            "__unix".to_string(),
        ];
        let error = ws
            .resolve_target_for_source(&tmp, &source, "linux-aarch64", Some(&wrong_arch))
            .unwrap_err()
            .to_string();
        assert!(error.contains("expected `aarch64`"), "{error}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn exact_target_envelope_keeps_declared_cuda_arch_when_pixi_omits_detection() {
        let tmp = temp_workspace("target-envelope-cuda-table");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "gpu", platform = "linux-64", cuda = { driver = "12.0", arch = "8.6" }, archspec = "x86_64_v3", glibc = "2.35", linux = "4.18" },
]
[feature.gpu]
platforms = ["gpu"]
[feature.gpu.dependencies]
demo-pack = { path = "./demo-pack" }
[environments]
gpu = { features = ["gpu"], no-default-feature = true }
"#,
        );
        let envelope = WorkspaceTargetEnvelope {
            schema: 1,
            environment: "gpu".to_string(),
            profile: WorkspaceTargetEnvelopeProfile {
                name: "gpu".to_string(),
                subdir: "linux-64".to_string(),
                virtual_packages: vec![
                    "cuda = { driver = \"12.0\", arch = \"8.6\" }".to_string(),
                    "archspec=x86_64_v3".to_string(),
                    "glibc=2.35".to_string(),
                    "linux=4.18".to_string(),
                ],
                detected_virtual_packages: vec![
                    "cuda=12".to_string(),
                    "glibc=2.35".to_string(),
                    "linux=4.18".to_string(),
                    "__archspec=1=x86_64_v3".to_string(),
                    "__unix".to_string(),
                ],
            },
        };
        let target = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", Some(&envelope))
            .unwrap()
            .unwrap();
        assert_eq!(
            target.contract.declared_virtual_packages.get("cuda"),
            Some(&"12".to_string())
        );
        assert_eq!(
            target.contract.declared_virtual_packages.get("cuda_arch"),
            Some(&"8.6".to_string())
        );
        let system_requirements = target.contract.system_requirements();
        assert_eq!(
            system_requirements.get("cuda_arch"),
            Some(&"8.6".to_string())
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn exact_target_envelope_retains_unknown_declared_vp_omitted_by_pixi_detection() {
        let tmp = temp_workspace("target-envelope-future-vp");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "future", platform = "linux-64", __future_pkg = "7.2" },
]
[feature.future]
platforms = ["future"]
[feature.future.dependencies]
demo-pack = { path = "./demo-pack" }
[environments]
future = { features = ["future"], no-default-feature = true }
"#,
        );
        let envelope = WorkspaceTargetEnvelope {
            schema: 1,
            environment: "future".to_string(),
            profile: WorkspaceTargetEnvelopeProfile {
                name: "future".to_string(),
                subdir: "linux-64".to_string(),
                virtual_packages: vec!["__future_pkg=7.2".to_string()],
                detected_virtual_packages: vec![
                    "glibc=2.28".to_string(),
                    "linux=4.18".to_string(),
                    "__archspec=1=x86_64".to_string(),
                    "__unix".to_string(),
                ],
            },
        };
        let resolved = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", Some(&envelope))
            .unwrap()
            .unwrap();
        assert_eq!(
            resolved
                .contract
                .declared_virtual_packages
                .get("future_pkg"),
            Some(&"7.2".to_string())
        );
        assert!(
            !resolved
                .contract
                .system_requirements()
                .contains_key("future_pkg")
        );

        let target = crate::pypi::ResolutionTarget::try_for_contract_on_subdir(
            "3.11",
            "linux-64",
            resolved.contract.clone(),
        )
        .unwrap();
        assert_eq!(target.target_contract(), Some(&resolved.contract));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn exact_target_envelope_accepts_pixi073_macos_and_windows_alias_rendering() {
        let tmp = temp_workspace("target-envelope-platform-aliases");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = [
  { name = "mac-test", platform = "osx-64", osx = "14.2" },
  { name = "win-test", platform = "win-64", __win = "7.7" },
]
[feature.mac]
platforms = ["mac-test"]
[feature.mac.dependencies]
demo-pack = { path = "./demo-pack" }
[feature.windows]
platforms = ["win-test"]
[feature.windows.dependencies]
demo-pack = { path = "./demo-pack" }
[environments]
mac = { features = ["mac"], no-default-feature = true }
windows = { features = ["windows"], no-default-feature = true }
"#,
        );
        let mac_envelope = WorkspaceTargetEnvelope {
            schema: 1,
            environment: "mac".to_string(),
            profile: WorkspaceTargetEnvelopeProfile {
                name: "mac-test".to_string(),
                subdir: "osx-64".to_string(),
                virtual_packages: vec!["macos=14.2".to_string()],
                detected_virtual_packages: vec![
                    "cuda=12.9".to_string(),
                    "macos=14.2".to_string(),
                    "__archspec=1=x86_64".to_string(),
                    "__unix".to_string(),
                ],
            },
        };
        let mac = ws
            .resolve_target_for_source(&tmp, &source, "osx-64", Some(&mac_envelope))
            .unwrap()
            .unwrap();
        assert_eq!(
            mac.contract.declared_virtual_packages.get("macos"),
            Some(&"14.2".to_string())
        );
        crate::pypi::ResolutionTarget::try_for_contract_on_subdir("3.11", "osx-64", mac.contract)
            .unwrap();
        let mut wrong_mac_arch = mac_envelope.clone();
        wrong_mac_arch.profile.detected_virtual_packages = vec![
            "cuda=12.9".to_string(),
            "macos=14.2".to_string(),
            "__archspec=1=aarch64".to_string(),
            "__unix".to_string(),
        ];
        let error = ws
            .resolve_target_for_source(&tmp, &source, "osx-64", Some(&wrong_mac_arch))
            .unwrap_err()
            .to_string();
        assert!(error.contains("expected `x86_64`"), "{error}");

        let windows_envelope = WorkspaceTargetEnvelope {
            schema: 1,
            environment: "windows".to_string(),
            profile: WorkspaceTargetEnvelopeProfile {
                name: "win-test".to_string(),
                subdir: "win-64".to_string(),
                virtual_packages: vec!["windows=7.7".to_string()],
                detected_virtual_packages: vec![
                    "cuda=12.9".to_string(),
                    "windows=7.7".to_string(),
                    "__archspec=1=x86_64".to_string(),
                ],
            },
        };
        let windows = ws
            .resolve_target_for_source(&tmp, &source, "win-64", Some(&windows_envelope))
            .unwrap()
            .unwrap();
        assert_eq!(
            windows.contract.declared_virtual_packages.get("windows"),
            Some(&"7.7".to_string())
        );
        crate::pypi::ResolutionTarget::try_for_contract_on_subdir(
            "3.11",
            "win-64",
            windows.contract,
        )
        .unwrap();
        let mut wrong_windows_arch = windows_envelope;
        wrong_windows_arch.profile.detected_virtual_packages = vec![
            "cuda=12.9".to_string(),
            "windows=7.7".to_string(),
            "__archspec=1=aarch64".to_string(),
        ];
        let error = ws
            .resolve_target_for_source(&tmp, &source, "win-64", Some(&wrong_windows_arch))
            .unwrap_err()
            .to_string();
        assert!(error.contains("expected `x86_64`"), "{error}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn exact_target_envelope_accepts_pixi073_empty_and_unix_only_platforms() {
        let tmp = temp_workspace("target-envelope-empty-and-unix-only");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = ["freebsd-64", "emscripten-wasm32", "wasi-wasm32", "zos-z"]
[dependencies]
demo-pack = { path = "./demo-pack" }
[environments]
default = []
"#,
        );

        let envelope =
            |subdir: &str, detected_virtual_packages: Vec<String>| WorkspaceTargetEnvelope {
                schema: 1,
                environment: "default".to_string(),
                profile: WorkspaceTargetEnvelopeProfile {
                    name: subdir.to_string(),
                    subdir: subdir.to_string(),
                    virtual_packages: Vec::new(),
                    detected_virtual_packages,
                },
            };

        for subdir in ["freebsd-64", "emscripten-wasm32"] {
            let target = ws
                .resolve_target_for_source(
                    &tmp,
                    &source,
                    subdir,
                    Some(&envelope(subdir, vec!["__unix".to_string()])),
                )
                .unwrap()
                .unwrap();
            assert_eq!(
                target.contract.detected_virtual_packages,
                BTreeMap::from([("unix".to_string(), String::new())])
            );

            let error = ws
                .resolve_target_for_source(
                    &tmp,
                    &source,
                    subdir,
                    Some(&envelope(subdir, Vec::new())),
                )
                .unwrap_err()
                .to_string();
            assert!(error.contains("is missing `unix`"), "{error}");
        }

        for subdir in ["wasi-wasm32", "zos-z"] {
            let target = ws
                .resolve_target_for_source(
                    &tmp,
                    &source,
                    subdir,
                    Some(&envelope(subdir, Vec::new())),
                )
                .unwrap()
                .unwrap();
            assert!(target.contract.detected_virtual_packages.is_empty());
            assert!(target.contract.system_requirements().is_empty());
            assert!(
                ws.effective_system_requirements_for_contract("default", &target.contract)
                    .is_empty()
            );
        }

        let error = ws
            .resolve_target_for_source(
                &tmp,
                &source,
                "wasi-wasm32",
                Some(&envelope("wasi-wasm32", vec!["__unix".to_string()])),
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("contains incompatible `unix`"), "{error}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn declared_glibc_parses_legacy_scalar_and_table_forms() {
        let ws = ws_toml(
            r#"
[system-requirements]
libc = "2.34"

[feature.newer.system-requirements]
libc = { family = "glibc", version = "2.35" }
"#,
        );
        assert_eq!(ws.declared_glibc(None), Some((2, 35)));
    }

    #[test]
    fn declared_glibc_ignores_legacy_musl_family() {
        let ws = ws_toml(
            r#"
[system-requirements]
libc = { family = "musl", version = "1.2" }
"#,
        );
        assert_eq!(ws.system_requirements.get("libc"), None);
        assert_eq!(ws.declared_glibc(None), None);
    }

    #[test]
    fn declared_glibc_parses_rich_platforms_and_ignores_wrong_platform() {
        let other = if crate::glibc::current_pixi_platform() == "linux-64" {
            "linux-aarch64"
        } else {
            "linux-64"
        };
        let ws = ws_toml(&format!(
            r#"
[workspace]
platforms = [
  "{other}",
  {{ platform = "{other}", glibc = "2.39" }},
  {{ platform = "{}", glibc = "2.35", cuda = "12" }},
]
"#,
            crate::glibc::current_pixi_platform()
        ));
        assert_eq!(ws.declared_glibc(None), Some((2, 35)));
    }

    #[test]
    fn declared_glibc_rich_platform_wins_over_legacy() {
        let ws = ws_toml(&format!(
            r#"
[workspace]
platforms = [{{ platform = "{}", glibc = "2.35" }}]

[system-requirements]
libc = "2.50"
"#,
            crate::glibc::current_pixi_platform()
        ));
        assert_eq!(ws.declared_glibc(None), Some((2, 35)));
    }

    #[test]
    fn declared_glibc_env_scope_uses_effective_legacy_requirements() {
        let ws = ws_toml(
            r#"
[system-requirements]
libc = "2.34"

[environments]
gpu = { features = ["sim"] }

[feature.sim.system-requirements]
libc = "2.36"
"#,
        );
        assert_eq!(ws.declared_glibc(Some("gpu")), Some((2, 36)));
    }

    #[test]
    fn effective_channels_appends_features_after_workspace() {
        let ws = ws_toml(
            r#"
[workspace]
channels = ["https://prefix.dev/conda-forge"]

[environments]
ros = { features = ["ros2"] }

[feature.ros2]
channels = ["https://prefix.dev/robostack-humble", "https://prefix.dev/conda-forge"]
"#,
        );
        let chans = ws.effective_channels("ros");
        // Top-level conda-forge first, then robostack-humble appended.
        // Duplicate conda-forge in feature is dropped.
        assert_eq!(
            chans,
            vec![
                "https://prefix.dev/conda-forge",
                "https://prefix.dev/robostack-humble",
            ],
        );
    }

    #[test]
    fn split_conda_dep_line_extracts_name_and_spec() {
        assert_eq!(
            split_conda_dep_line("numpy >=1.26,<2"),
            Some(("numpy".into(), ">=1.26,<2".into())),
        );
        assert_eq!(
            split_conda_dep_line("python_abi 3.11.* *_cp311"),
            Some(("python_abi".into(), "3.11.*".into())),
        );
        assert_eq!(
            split_conda_dep_line("libgcc-ng"),
            Some(("libgcc-ng".into(), "".into())),
        );
        assert_eq!(split_conda_dep_line(""), None);
        assert_eq!(split_conda_dep_line("   "), None);
    }

    #[test]
    fn constraint_lines_include_constrains_and_keep_anchors() {
        // P3 (grizzly #6): run_constrained entries are recorded
        // alongside depends. Anchors like cuda-version are KEPT --
        // recording a workspace-imposed constraint is input-side
        // parity; never widening it beyond the ABI compatibility band
        // is emission-side and enforced elsewhere. python/python_abi
        // stay skipped (relax never
        // touches them; they'd clutter the comma-join).
        let depends = vec![
            "numpy >=1.26,<2".to_string(),
            "python_abi 3.11.* *_cp311".to_string(),
        ];
        let constrains = vec![
            "cuda-version ==12.8".to_string(),
            "tensorboard >=2.0".to_string(),
            "python >=3.9".to_string(),
            "anything *".to_string(),
        ];
        let lines = constraint_lines(&depends, &constrains);
        let names: Vec<&str> = lines.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"numpy"));
        assert!(
            names.contains(&"cuda-version"),
            "anchor CONSTRAINT must be recorded (input parity): {names:?}"
        );
        assert!(names.contains(&"tensorboard"));
        assert!(!names.contains(&"python"));
        assert!(!names.contains(&"python_abi"));
        assert!(!names.contains(&"anything"), "`*` imposes nothing");
        let cuda = lines.iter().find(|(n, _)| n == "cuda-version").unwrap();
        assert_eq!(cuda.1, "==12.8");
    }

    #[test]
    fn parses_pypi_options_index_urls() {
        let ws = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]

[pypi-options]
index-url = "https://pypi.nvidia.com"
extra-index-urls = ["https://download.pytorch.org/whl/cu128"]

[feature.sim.pypi-options]
extra-index-urls = ["https://py.mujoco.org"]

[feature.gpu.pypi-options]
index-url = "https://pypi.nvidia.com"
"#,
        );
        assert_eq!(
            ws.pypi_options,
            PypiOptions {
                index_url: Some("https://pypi.nvidia.com".to_string()),
                extra_index_urls: vec!["https://download.pytorch.org/whl/cu128".to_string()],
            },
        );
        assert_eq!(
            ws.features["sim"].pypi_options,
            PypiOptions {
                index_url: None,
                extra_index_urls: vec!["https://py.mujoco.org".to_string()],
            },
        );
        // Raw rollup retains historical index-url-then-extras order;
        // feature names are stable because the manifest uses a BTreeMap.
        assert_eq!(
            ws.all_pypi_index_urls(),
            vec![
                "https://pypi.nvidia.com".to_string(),
                "https://download.pytorch.org/whl/cu128".to_string(),
                "https://py.mujoco.org".to_string(),
            ],
        );
    }

    #[test]
    fn resolution_indexes_preserve_explicit_extra_priority() {
        let ws = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]

[pypi-options]
extra-index-urls = ["https://pypi.org/simple", "https://pypi.nvidia.com", "https://py.mujoco.org"]
"#,
        );
        assert_eq!(
            ws.resolution_pypi_index_urls(),
            vec![
                "https://pypi.org/simple".to_string(),
                "https://pypi.nvidia.com".to_string(),
                "https://py.mujoco.org".to_string(),
            ],
        );
    }

    #[test]
    fn resolution_indexes_append_implicit_default_after_extras() {
        let ws = ws_toml(
            r#"
[pypi-options]
extra-index-urls = ["https://pypi.nvidia.com", "https://py.mujoco.org"]
"#,
        );
        assert_eq!(
            ws.resolution_pypi_index_urls(),
            vec![
                "https://pypi.nvidia.com".to_string(),
                "https://py.mujoco.org".to_string(),
                DEFAULT_PYPI_INDEX.to_string(),
            ],
        );
    }

    #[test]
    fn resolution_indexes_honor_explicit_default_override() {
        let ws = ws_toml(
            r#"
[pypi-options]
index-url = "https://packages.example/simple"
extra-index-urls = ["https://extra.example/simple"]

[feature.zeta.pypi-options]
extra-index-urls = ["https://zeta.example/simple"]

[feature.alpha.pypi-options]
index-url = "https://alpha.example/simple"
"#,
        );
        let indexes = ws.resolution_pypi_index_urls();
        assert_eq!(
            indexes,
            vec![
                "https://extra.example/simple".to_string(),
                "https://packages.example/simple".to_string(),
                "https://alpha.example/simple".to_string(),
                "https://zeta.example/simple".to_string(),
            ],
        );
        assert!(
            !indexes.iter().any(|url| {
                url.trim_end_matches('/') == DEFAULT_PYPI_INDEX.trim_end_matches('/')
            })
        );
    }

    #[test]
    fn pypi_options_absent_yields_empty() {
        let ws = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]
"#,
        );
        assert_eq!(ws.pypi_options, PypiOptions::default());
        assert!(ws.all_pypi_index_urls().is_empty());
        assert_eq!(
            ws.resolution_pypi_index_urls(),
            vec![DEFAULT_PYPI_INDEX.to_string()]
        );
    }

    #[test]
    fn parses_channel_priority_setting() {
        // Explicit "disabled"
        let ws = ws_toml(
            r#"
[workspace]
channels = ["a"]
channel-priority = "disabled"
"#,
        );
        assert_eq!(ws.channel_priority.as_deref(), Some("disabled"));

        // Explicit "strict"
        let ws = ws_toml(
            r#"
[workspace]
channels = ["a"]
channel-priority = "strict"
"#,
        );
        assert_eq!(ws.channel_priority.as_deref(), Some("strict"));

        // Underscore alias parsed too
        let ws = ws_toml(
            r#"
[workspace]
channels = ["a"]
channel_priority = "disabled"
"#,
        );
        assert_eq!(ws.channel_priority.as_deref(), Some("disabled"));

        // Unspecified -> None (caller picks default)
        let ws = ws_toml(
            r#"
[workspace]
channels = ["a"]
"#,
        );
        assert_eq!(ws.channel_priority, None);
    }

    #[test]
    fn find_declaring_feature_locates_pin() {
        let ws = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]

[dependencies]
numpy = "==1.26.4"

[environments]
gsi = { features = ["gpu"] }
standalone = { features = ["x"], no-default-feature = true }

[feature.gpu]

[feature.gpu.dependencies]
pytorch-gpu = ">=2.7"

[feature.x]

[feature.x.dependencies]
something = "*"
"#,
        );
        // Top-level [dependencies] -> "default"
        assert_eq!(
            ws.find_declaring_feature("gsi", "numpy").as_deref(),
            Some("default"),
        );
        // Feature-scoped
        assert_eq!(
            ws.find_declaring_feature("gsi", "pytorch-gpu").as_deref(),
            Some("gpu"),
        );
        // no-default-feature env -> skip top-level
        assert_eq!(ws.find_declaring_feature("standalone", "numpy"), None);
        // Unknown dep
        assert_eq!(ws.find_declaring_feature("gsi", "asdf"), None);
        // Unknown env
        assert_eq!(ws.find_declaring_feature("nope", "numpy"), None);
    }

    #[test]
    fn discover_outputs_finds_features_referencing_source_dir() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-ws-discover-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(tmp.join("isaac-pack")).unwrap();
        let ws = ws_toml(
            r#"
[environments]
gsi      = { features = ["isaaclab_physx"] }
gsi-ros2 = { features = ["isaaclab_physx", "ros2"] }
gsn      = { features = ["isaaclab_newton"] }
unrelated = { features = ["misc"] }

[feature.isaaclab_physx.dependencies]
isaac-pack-physx = { path = "./isaac-pack" }

[feature.isaaclab_newton.dependencies]
isaac-pack-newton = { path = "./isaac-pack" }

[feature.ros2]

[feature.misc.dependencies]
some-other-pkg = "*"
"#,
        );
        let outputs = ws.discover_outputs_for_source(&tmp, &tmp.join("isaac-pack"));
        // Sort for stable assertion.
        let mut by_name: BTreeMap<String, &DiscoveredOutput> =
            outputs.iter().map(|o| (o.name.clone(), o)).collect();
        let physx = by_name.remove("isaac-pack-physx").expect("physx output");
        // gsi AND gsi-ros2 use feature.isaaclab_physx -> both envs.
        assert_eq!(physx.envs, vec!["gsi".to_string(), "gsi-ros2".to_string()]);
        let newton = by_name.remove("isaac-pack-newton").expect("newton output");
        assert_eq!(newton.envs, vec!["gsn".to_string()]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn discover_outputs_merge_active_feature_dependency_maps() {
        let tmp = temp_workspace("discover-final-overlay");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = ["linux-64"]

[feature.source.dependencies]
demo-pack = { path = "./demo-pack" }
[feature.registry.dependencies]
demo-pack = ">=2"
[feature.same-source.dependencies]
demo-pack = { path = "./demo-pack" }

[environments]
keeps-source = { features = ["source"], no-default-feature = true }
uses-registry = { features = ["source", "registry"], no-default-feature = true }
uses-same-source = { features = ["source", "same-source"], no-default-feature = true }
"#,
        );
        let outputs = ws.discover_outputs_for_source_for_target(&tmp, &source, "linux-64");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name, "demo-pack");
        assert_eq!(outputs[0].declaring_features, vec!["same-source", "source"]);
        assert_eq!(
            outputs[0].envs,
            vec!["keeps-source", "uses-registry", "uses-same-source"]
        );
        assert_eq!(
            ws.effective_dependencies_for_target("uses-registry", "linux-64")
                .get("demo-pack")
                .map(String::as_str),
            Some(">=2")
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn target_local_git_or_url_source_tombstones_inherited_path_consumption() {
        for (case, source_spec) in [
            (
                "git",
                r#"{ git = "https://example.invalid/demo.git", rev = "0123456789abcdef" }"#,
            ),
            (
                "url",
                r#"{ url = "https://example.invalid/demo-1.0-0.conda" }"#,
            ),
        ] {
            let tmp = temp_workspace(&format!("target-{case}-tombstone"));
            let source = tmp.join("demo-pack");
            std::fs::create_dir_all(&source).unwrap();
            let ws = ws_toml(&format!(
                r#"
[workspace]
platforms = [
  {{ name = "p1", platform = "linux-64", glibc = "2.28" }},
  {{ name = "p3", platform = "linux-64", glibc = "2.35" }},
]
[feature.source.dependencies]
demo-pack = {{ path = "./demo-pack" }}
[feature.source.target.p3.dependencies]
demo-pack = {source_spec}
[environments]
sim = {{ features = ["source"], no-default-feature = true }}
"#,
            ));
            let p1 = ws
                .target_profiles
                .iter()
                .find(|profile| profile.name == "p1")
                .unwrap();
            let p3 = ws
                .target_profiles
                .iter()
                .find(|profile| profile.name == "p3")
                .unwrap();
            assert_eq!(
                ws.try_consuming_envs_for_profile(&tmp, &source, p1)
                    .unwrap(),
                BTreeSet::from(["sim".to_string()]),
                "{case} control profile must retain the inherited path",
            );
            assert!(
                ws.try_consuming_envs_for_profile(&tmp, &source, p3)
                    .unwrap()
                    .is_empty(),
                "target-local {case} source must replace the inherited path",
            );
            let target = ws
                .resolve_target_for_source(&tmp, &source, "linux-64", None)
                .unwrap()
                .unwrap();
            assert_eq!(target.profiles, vec!["p1"]);
            std::fs::remove_dir_all(tmp).ok();
        }
    }

    #[test]
    fn active_feature_path_and_git_or_url_sources_fail_closed() {
        for (case, source_spec) in [
            (
                "git",
                r#"{ git = "https://example.invalid/demo.git", rev = "0123456789abcdef" }"#,
            ),
            (
                "url",
                r#"{ url = "https://example.invalid/demo-1.0-0.conda" }"#,
            ),
        ] {
            let tmp = temp_workspace(&format!("feature-{case}-source-conflict"));
            let source = tmp.join("demo-pack");
            std::fs::create_dir_all(&source).unwrap();
            let ws = ws_toml(&format!(
                r#"
[workspace]
platforms = ["linux-64"]
[feature.local.dependencies]
demo-pack = {{ path = "./demo-pack" }}
[feature.remote.dependencies]
demo-pack = {source_spec}
[environments]
broken = {{ features = ["local", "remote"], no-default-feature = true }}
"#,
            ));
            let error = ws
                .resolve_target_for_source(&tmp, &source, "linux-64", None)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("conflicting path and git/URL sources for `demo-pack`"),
                "{case}: {error}",
            );
            std::fs::remove_dir_all(tmp).ok();
        }
    }

    #[test]
    fn conflicting_feature_sources_fail_closed_before_target_inference() {
        let tmp = temp_workspace("discover-conflicting-sources");
        let source = tmp.join("demo-pack");
        let replacement = tmp.join("replacement-pack");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::create_dir_all(&replacement).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = ["linux-64"]

[feature.source.dependencies]
demo-pack = { path = "./demo-pack" }
[feature.replacement.dependencies]
demo-pack = { path = "./replacement-pack" }
[environments]
broken = { features = ["source", "replacement"], no-default-feature = true }
"#,
        );
        let error = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", None)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("conflicting sources for `demo-pack`"),
            "{error}"
        );

        let envelope = WorkspaceTargetEnvelope {
            schema: 1,
            environment: "broken".to_string(),
            profile: WorkspaceTargetEnvelopeProfile {
                name: "linux-64".to_string(),
                subdir: "linux-64".to_string(),
                virtual_packages: Vec::new(),
                detected_virtual_packages: vec![
                    "glibc=2.28".to_string(),
                    "linux=4.18".to_string(),
                    "__archspec=1=x86_64".to_string(),
                    "__unix".to_string(),
                ],
            },
        };
        let error = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", Some(&envelope))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("conflicting sources for `demo-pack`"),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn implicit_default_environment_consumes_top_level_path_source() {
        let tmp = temp_workspace("discover-implicit-default");
        let source = tmp.join("demo-pack");
        std::fs::create_dir_all(&source).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
platforms = ["linux-64"]

[dependencies]
demo-pack = { path = "./demo-pack" }
"#,
        );
        assert!(ws.has_environment("default"));
        let outputs = ws.discover_outputs_for_source_for_target(&tmp, &source, "linux-64");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].declaring_features, vec!["default"]);
        assert_eq!(outputs[0].envs, vec!["default"]);

        let target = ws
            .resolve_target_for_source(&tmp, &source, "linux-64", None)
            .unwrap()
            .unwrap();
        assert_eq!(target.profiles, vec!["linux-64"]);
        assert_eq!(target.environments, vec!["default"]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn discover_outputs_empty_when_no_features_reference_source() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-ws-discover-empty-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(tmp.join("isaac-pack")).unwrap();
        let ws = ws_toml(
            r#"
[environments]
gsi = { features = ["x"] }

[feature.x.dependencies]
some-pkg = "==1.0"
"#,
        );
        let outputs = ws.discover_outputs_for_source(&tmp, &tmp.join("isaac-pack"));
        assert!(outputs.is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn precise_consuming_envs_returns_sorted_active_envs() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-ws-precise-envs-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(tmp.join("demo-pack")).unwrap();
        let ws = ws_toml(
            r#"
[environments]
zeta = { features = ["pack"] }
alpha = { features = ["pack"] }
unrelated = { features = ["other"] }

[feature.pack.dependencies]
demo-pack = { path = "./demo-pack" }

[feature.other.dependencies]
unrelated = "*"
"#,
        );

        assert_eq!(
            ws.precise_consuming_envs(&tmp, &tmp.join("demo-pack")),
            Some(vec!["alpha".to_string(), "zeta".to_string()])
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn precise_consuming_envs_abstains_for_ambiguous_fallbacks() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-ws-ambiguous-envs-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(tmp.join("demo-pack")).unwrap();
        std::fs::create_dir_all(tmp.join("unmapped-pack")).unwrap();
        let ws = ws_toml(
            r#"
[environments]
active = { features = ["other"] }

[feature.pack.dependencies]
demo-pack = { path = "./demo-pack" }

[feature.other.dependencies]
unrelated = "*"
"#,
        );

        // Tier 3: the source is declared, but no concrete env activates it.
        assert_eq!(
            ws.precise_consuming_envs(&tmp, &tmp.join("demo-pack")),
            None
        );
        // Tier 4: no declaration maps to this source at all.
        assert_eq!(
            ws.precise_consuming_envs(&tmp, &tmp.join("unmapped-pack")),
            None
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn consuming_env_dependencies_scopes_to_the_pack_env_range() {
        // envoracle fix fixture: the workspace declares `pillow >=11,<12`
        // for the env that consumes `hover-pack` (not the top-level
        // `default`, which auto-route's old hardcoded lookup used).
        // `consuming_env_dependencies` must surface THAT range so the
        // co-installability check can catch an exact `pillow ==12.3.0`
        // auto-route pin instead of missing it.
        let tmp = std::env::temp_dir().join(format!(
            "retread-ws-envdeps-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(tmp.join("hover-pack")).unwrap();
        let ws = ws_toml(
            r#"
[dependencies]
numpy = "*"

[environments]
hover = { features = ["hover"] }

[feature.hover.dependencies]
hover-pack = { path = "./hover-pack" }
pillow = ">=11,<12"
"#,
        );
        let deps = ws.consuming_env_dependencies(&tmp, &tmp.join("hover-pack"));
        assert_eq!(deps.get("pillow"), Some(&vec![">=11,<12".to_string()]));
        // The default env's `numpy` isn't pulled in unless `hover`
        // actually inherits the default feature (it does here), but the
        // key assertion is that the SCOPED env's own range is present
        // and precise -- not silently dropped in favor of `default`.
        assert!(deps.contains_key("numpy"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn consuming_env_system_requirements_surfaces_feature_scoped_cuda() {
        // Run-34 fixture: a `no-default-feature` env (pm-isaaclab shape)
        // declares `cuda = "12"` under ITS OWN feature. The co-install
        // oracle fed `effective_system_requirements("default")` -- no
        // `__cuda` virtual -- so the trial set containing the env's
        // cuda-only pins was unsat regardless of the routed candidates
        // and every route sailed through unchecked. The consuming-env
        // lookup must surface the feature-scoped cuda.
        let tmp = std::env::temp_dir().join(format!(
            "retread-ws-envsysreq-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(tmp.join("pm-pack")).unwrap();
        let ws = ws_toml(
            r#"
[dependencies]
numpy = "*"

[environments]
pm = { features = ["pm"], no-default-feature = true }

[feature.pm.dependencies]
pm-pack = { path = "./pm-pack" }
pytorch-gpu = "==2.7.0"

[feature.pm.system-requirements]
cuda = "12"
"#,
        );
        let sysreqs = ws.consuming_env_system_requirements(&tmp, &tmp.join("pm-pack"));
        assert_eq!(
            sysreqs.get("cuda").map(String::as_str),
            Some("12"),
            "feature-scoped cuda must reach the co-install oracle: {sysreqs:?}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn consuming_env_dependencies_unions_multiple_consuming_envs() {
        // Two envs both consume the pack with DIFFERENT pillow ranges;
        // both distinct specs must appear so the solve is required to
        // satisfy both simultaneously (a real conflict here is a
        // pre-existing workspace misconfiguration, not something
        // auto-route should paper over).
        let tmp = std::env::temp_dir().join(format!(
            "retread-ws-envdeps-union-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(tmp.join("hover-pack")).unwrap();
        let ws = ws_toml(
            r#"
[environments]
hover-a = { features = ["a"] }
hover-b = { features = ["b"] }

[feature.a.dependencies]
hover-pack = { path = "./hover-pack" }
pillow = ">=11,<12"

[feature.b.dependencies]
hover-pack = { path = "./hover-pack" }
pillow = ">=10,<11"
"#,
        );
        let deps = ws.consuming_env_dependencies(&tmp, &tmp.join("hover-pack"));
        let mut specs = deps.get("pillow").cloned().unwrap_or_default();
        specs.sort();
        assert_eq!(specs, vec![">=10,<11".to_string(), ">=11,<12".to_string()]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn consuming_env_dependencies_falls_back_to_all_features_when_pack_unreferenced() {
        // Nothing in the workspace references this source_dir at all
        // (e.g. discovery running before the workspace declares the
        // path-dep, or a standalone build). Conservative superset:
        // union across every feature rather than silently returning
        // nothing (which would make the co-install check blind again).
        let ws = ws_toml(
            r#"
[dependencies]
numpy = "*"

[feature.a.dependencies]
pillow = ">=11,<12"

[feature.b.dependencies]
scipy = "*"
"#,
        );
        let tmp = std::env::temp_dir().join("retread-ws-envdeps-unreferenced-nonexistent");
        let deps = ws.consuming_env_dependencies(&tmp, &tmp.join("no-such-pack"));
        assert_eq!(deps.get("pillow"), Some(&vec![">=11,<12".to_string()]));
        assert_eq!(deps.get("scipy"), Some(&vec!["*".to_string()]));
        assert_eq!(deps.get("numpy"), Some(&vec!["*".to_string()]));
    }

    #[test]
    fn missing_env_returns_empty() {
        let ws = WorkspaceManifest::default();
        assert!(ws.effective_dependencies("nonexistent").is_empty());
        assert!(ws.effective_channels("nonexistent").is_empty());
    }

    #[test]
    fn effective_pypi_index_urls_unions_features_with_top_level() {
        let ws = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]

[pypi-options]
index-url = "https://pypi.nvidia.com"
extra-index-urls = ["https://download.pytorch.org/whl/cu128"]

[environments]
sim = { features = ["mujoco"] }

[feature.mujoco.pypi-options]
extra-index-urls = ["https://py.mujoco.org"]
"#,
        );
        let urls = ws.effective_pypi_index_urls("sim");
        assert_eq!(
            urls,
            vec![
                "https://pypi.nvidia.com".to_string(),
                "https://download.pytorch.org/whl/cu128".to_string(),
                "https://py.mujoco.org".to_string(),
            ],
        );
    }

    #[test]
    fn effective_pypi_index_urls_no_default_feature_skips_top_level() {
        let ws = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]

[pypi-options]
index-url = "https://pypi.nvidia.com"

[environments]
standalone = { features = ["f"], no-default-feature = true }

[feature.f.pypi-options]
extra-index-urls = ["https://py.mujoco.org"]
"#,
        );
        let urls = ws.effective_pypi_index_urls("standalone");
        // Top-level nvidia index skipped because no-default-feature.
        assert!(!urls.contains(&"https://pypi.nvidia.com".to_string()));
        assert_eq!(urls, vec!["https://py.mujoco.org".to_string()]);
    }

    #[test]
    fn effective_pypi_index_urls_deduplicates() {
        let ws = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]

[pypi-options]
index-url = "https://pypi.nvidia.com"

[environments]
gpu = { features = ["g"] }

[feature.g.pypi-options]
index-url = "https://pypi.nvidia.com"
"#,
        );
        let urls = ws.effective_pypi_index_urls("gpu");
        // nvidia appears in both top-level and feature -- only first occurrence kept.
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0], "https://pypi.nvidia.com");
    }

    #[test]
    fn solve_fingerprint_uses_requested_platform_requirements() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-ws-target-fp-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let src = tmp.join("my-pack");
        std::fs::create_dir_all(&src).unwrap();
        let ws = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]
platforms = [
  { platform = "linux-64", glibc = "2.35", cuda = "12" },
  { platform = "linux-aarch64", glibc = "2.39", cuda = "13" },
]

[environments]
sim = { features = ["sim"] }

[feature.sim.dependencies]
my-pack = { path = "./my-pack" }
"#,
        );

        let x86 = ws.solve_fingerprint_for_target(&tmp, &src, "linux-64");
        let arm = ws.solve_fingerprint_for_target(&tmp, &src, "linux-aarch64");
        assert!(x86.contains("sysreq:libc=2.35"));
        assert!(x86.contains("sysreq:cuda=12"));
        assert!(arm.contains("sysreq:libc=2.39"));
        assert!(arm.contains("sysreq:cuda=13"));
        assert_ne!(x86, arm);

        std::fs::remove_dir_all(tmp).ok();
    }

    #[test]
    fn solve_fingerprint_is_pack_scoped_to_referencing_envs() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-ws-fp-scope-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(tmp.join("my-pack")).unwrap();
        let src = tmp.join("my-pack");

        // Manifest A: one env referencing the pack via feature.base.
        let ws_a = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]

[environments]
env-a = { features = ["base"] }

[feature.base.dependencies]
my-pack = { path = "./my-pack" }
numpy = "==1.26.4"

[feature.base]
channels = ["conda-forge"]
"#,
        );
        let fp_a = ws_a.solve_fingerprint(&tmp, &src);
        assert!(
            !fp_a.is_empty(),
            "must be non-empty when an env references pack"
        );

        // Manifest B = A + an UNRELATED env (new feature, NO path-dep to src/).
        // Adding an env that does NOT reference the pack must NOT change the fingerprint.
        let ws_b = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]

[environments]
env-a   = { features = ["base"] }
env-unrelated = { features = ["other"] }

[feature.base.dependencies]
my-pack = { path = "./my-pack" }
numpy = "==1.26.4"

[feature.base]
channels = ["conda-forge"]

[feature.other.dependencies]
torch = ">=2.7"

[feature.other]
channels = ["conda-forge", "pytorch"]
"#,
        );
        let fp_b = ws_b.solve_fingerprint(&tmp, &src);
        assert_eq!(
            fp_a, fp_b,
            "adding an unrelated env must not change the pack-scoped fingerprint"
        );

        // Manifest C = A but MUTATE the referencing env (add a dep to feature.base).
        // This must change the fingerprint.
        let ws_c = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]

[environments]
env-a = { features = ["base"] }

[feature.base.dependencies]
my-pack = { path = "./my-pack" }
numpy = "==1.26.4"
pinocchio = ">=3.6"

[feature.base]
channels = ["conda-forge"]
"#,
        );
        let fp_c = ws_c.solve_fingerprint(&tmp, &src);
        assert_ne!(
            fp_a, fp_c,
            "mutating the referencing env must change the fingerprint"
        );

        // Determinism: two calls on same manifest are byte-identical.
        let fp_a2 = ws_a.solve_fingerprint(&tmp, &src);
        assert_eq!(fp_a, fp_a2, "solve_fingerprint must be deterministic");

        // Empty: manifest where no env references src/ -> returns "".
        let ws_empty = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]

[environments]
env-a = { features = ["other"] }

[feature.other.dependencies]
torch = ">=2.7"
"#,
        );
        let fp_empty = ws_empty.solve_fingerprint(&tmp, &src);
        assert_eq!(
            fp_empty, "",
            "no env references the pack -> fingerprint must be empty"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn solve_fingerprint_tracks_scoped_pypi_dependencies() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-ws-fp-pypi-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(tmp.join("my-pack")).unwrap();
        let src = tmp.join("my-pack");

        let ws_v1 = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]

[environments]
consumer = { features = ["pack"] }

[feature.pack.dependencies]
my-pack = { path = "./my-pack" }

[feature.pack.pypi-dependencies]
FSSpec = "==1.0"
"#,
        );
        let ws_v2 = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]

[environments]
consumer = { features = ["pack"] }

[feature.pack.dependencies]
my-pack = { path = "./my-pack" }

[feature.pack.pypi-dependencies]
fsspec = "==2.0"
"#,
        );

        let fp_v1 = ws_v1.solve_fingerprint(&tmp, &src);
        let fp_v2 = ws_v2.solve_fingerprint(&tmp, &src);
        assert!(fp_v1.contains("scoped-env:consumer:pypi-dep:fsspec===1.0"));
        assert_ne!(
            fp_v1, fp_v2,
            "changing a consuming env's PyPI declaration must invalidate its solve fingerprint"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Regression test for the courier cold-solve-replay channel hash bug.
    ///
    /// The bug: pixi forwards different `params.channels` to conda/outputs
    /// (per-env subset) vs conda/build_v1 (union/other set) for multi-env
    /// workspaces, so `courier_inputs_hash` at the replayer and `config_fp`
    /// at the producer would never match.
    ///
    /// The fix: both sites call `courier_channel_set(workspace_dir, source_dir)`
    /// which derives the channel set from the manifest identically.
    ///
    /// This test verifies:
    /// (a) `courier_channel_set` is byte-identical across two calls (the
    ///     producer==replayer guarantee).
    /// (b) It equals the UNION of channels across ALL envs referencing the
    ///     source dir (not just one env's subset).
    #[test]
    fn courier_channel_set_is_manifest_derived_union_across_all_envs() {
        let tmp = std::env::temp_dir().join(format!(
            "retread-ws-courier-channels-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(tmp.join("my-pack")).unwrap();

        // Two envs reference the same source pack, each with a different
        // channel set. `genesis` has only `conda-forge`; `genesis-gpu` adds
        // `robostack-humble`. The bug would have caused the two RPCs to
        // disagree on which channel list to hash. The fix unions them.
        let ws = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]

[environments]
genesis     = { features = ["base"] }
genesis-gpu = { features = ["base", "gpu"] }

[feature.base.dependencies]
my-pack = { path = "./my-pack" }

[feature.base]
channels = ["conda-forge"]

[feature.gpu]
channels = ["conda-forge", "robostack-humble"]
"#,
        );

        let src = tmp.join("my-pack");
        // (a) byte-identical across two calls -- simulates producer + replayer
        let first = ws.courier_channel_set(&tmp, &src);
        let second = ws.courier_channel_set(&tmp, &src);
        assert_eq!(
            first, second,
            "producer and replayer must agree on channels"
        );

        // (b) must be the union across ALL envs (conda-forge + robostack-humble),
        // NOT just one env's subset. If only `genesis`'s channels were used we'd
        // get ["conda-forge"]; if only `genesis-gpu` we'd also get both but via
        // a different path. The assertion catches any single-env shortcut.
        assert!(
            first.contains(&"conda-forge".to_string()),
            "union must contain conda-forge: {:?}",
            first
        );
        assert!(
            first.contains(&"robostack-humble".to_string()),
            "union must contain robostack-humble (from genesis-gpu): {:?}",
            first
        );

        // Stable length: dedup means no duplicates
        let unique_count = first.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!(
            first.len(),
            unique_count,
            "no duplicate channels in union: {:?}",
            first
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
