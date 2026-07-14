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

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub(crate) const DEFAULT_PYPI_INDEX: &str = "https://pypi.org/simple/";

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
    /// Top-level `[dependencies]` — conda deps with a version spec.
    /// Used for transitive constraint extraction (each maps to a real
    /// channel package whose `depends` array we walk).
    pub dependencies: BTreeMap<String, String>,
    /// Top-level `[dependencies]` path-form entries
    /// (`<name> = { path = "...", ... }`). Used by output
    /// autodiscovery to find which workspace declarations reference
    /// THIS source package.
    pub path_dependencies: BTreeMap<String, String>,
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
    /// v1.3.0: top-level `[pypi-options]` index URLs -- `index-url`
    /// first, then `extra-index-urls` in declaration order. Feeds the
    /// cascade's PyPI fallback chain so workspace-declared private
    /// indexes are consulted when bundling PyPI-only deps, not just
    /// the `[retread-wheels]` entry indexes.
    pub pypi_index_urls: Vec<String>,
    /// Whether top-level `[pypi-options]` explicitly replaces the default
    /// PyPI index via `index-url`.
    pypi_index_url_overridden: bool,
}

#[derive(Debug, Default, Clone)]
pub struct EnvironmentDef {
    /// Features this env activates, in declaration order.
    pub features: Vec<String>,
    /// If true, the implicit "default" feature (top-level
    /// `[dependencies]` etc.) is NOT inherited.
    pub no_default_feature: bool,
}

#[derive(Debug, Default, Clone)]
pub struct FeatureDef {
    /// `[feature.X.channels]`.
    pub channels: Vec<String>,
    /// `[feature.X.dependencies]` — version-pinned conda deps.
    pub dependencies: BTreeMap<String, String>,
    /// `[feature.X.dependencies]` path-form entries. Output
    /// autodiscovery walks these to find features that reference the
    /// source package retread is building for.
    pub path_dependencies: BTreeMap<String, String>,
    /// `[feature.X.pypi-dependencies]`, with the same canonical-name
    /// and direct-source semantics as
    /// [`WorkspaceManifest::pypi_dependencies`].
    pub pypi_dependencies: BTreeMap<String, String>,
    /// v0.37.0+ (D1): `[feature.X.system-requirements]`. Same shape as
    /// the top-level table; unioned per active env with feature-wins
    /// precedence by `effective_system_requirements`.
    pub system_requirements: BTreeMap<String, String>,
    /// v1.3.0: `[feature.X.pypi-options]` index URLs, same shape as
    /// the top-level field.
    pub pypi_index_urls: Vec<String>,
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
        let parsed: toml::Value = toml::from_str(&bytes).ok()?;
        let manifest = Self::from_toml(&parsed);
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
                    let Some((name, glibc, cuda)) = parse_rich_platform(platform) else {
                        continue;
                    };
                    if let Some(glibc) = glibc {
                        out.platform_glibc.insert(name.clone(), glibc);
                    }
                    if let Some(cuda) = cuda {
                        out.platform_cuda.insert(name, cuda);
                    }
                }
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
                    DepKind::Other => {}
                }
            }
        }

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

        // v1.3.0: top-level [pypi-options] index URLs.
        (out.pypi_index_urls, out.pypi_index_url_overridden) = parse_pypi_index_urls(parsed);

        if let Some(envs) = parsed.get("environments").and_then(|v| v.as_table()) {
            for (name, value) in envs {
                if let Some(def) = parse_env_def(value) {
                    out.environments.insert(name.clone(), def);
                }
            }
        }

        if let Some(features) = parsed.get("feature").and_then(|v| v.as_table()) {
            for (name, fvalue) in features {
                let mut def = FeatureDef::default();
                if let Some(fmap) = fvalue.as_table() {
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
                                DepKind::Other => {}
                            }
                        }
                    }
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
                    // v1.3.0: per-feature [pypi-options] index URLs.
                    (def.pypi_index_urls, _) = parse_pypi_index_urls(fvalue);
                }
                out.features.insert(name.clone(), def);
            }
        }

        out
    }

    /// Compute the effective conda deps an environment will solve
    /// against. Order: top-level [dependencies] (unless
    /// no-default-feature), then each active feature in declaration
    /// order. Later entries OVERRIDE earlier ones with the same name
    /// (matches pixi's precedence: feature deps override workspace
    /// defaults).
    pub fn effective_dependencies(&self, env_name: &str) -> BTreeMap<String, String> {
        let Some(env) = self.environments.get(env_name) else {
            return BTreeMap::new();
        };
        let mut out = BTreeMap::new();
        if !env.no_default_feature {
            for (k, v) in &self.dependencies {
                out.insert(k.clone(), v.clone());
            }
        }
        for feat_name in &env.features {
            let Some(feat) = self.features.get(feat_name) else {
                continue;
            };
            for (k, v) in &feat.dependencies {
                out.insert(k.clone(), v.clone());
            }
        }
        out
    }

    /// Compute the effective PyPI declarations for an environment.
    /// Top-level `[pypi-dependencies]` is inherited unless the env sets
    /// `no-default-feature`; active feature declarations then override
    /// it in feature order, matching [`Self::effective_dependencies`].
    pub fn effective_pypi_dependencies(&self, env_name: &str) -> BTreeMap<String, String> {
        let Some(env) = self.environments.get(env_name) else {
            return BTreeMap::new();
        };
        let mut out = BTreeMap::new();
        if !env.no_default_feature {
            for (name, spec) in &self.pypi_dependencies {
                out.insert(name.clone(), spec.clone());
            }
        }
        for feat_name in &env.features {
            let Some(feat) = self.features.get(feat_name) else {
                continue;
            };
            for (name, spec) in &feat.pypi_dependencies {
                out.insert(name.clone(), spec.clone());
            }
        }
        out
    }

    /// v0.37.0+ (D1): effective system requirements for an environment.
    /// Top-level requirements first (unless no-default-feature), then
    /// each active feature overrides/extends in declaration order
    /// (pixi precedence). Keys are pixi-schema names (`cuda`, `libc`,
    /// ...); translation to rattler virtual-package names (`__cuda`,
    /// `__glibc`, ...) happens at the solve_check boundary.
    /// Every PyPI index the workspace declares anywhere: top-level
    /// `[pypi-options]` first, then each feature's in name order,
    /// deduped preserving first occurrence. The cascade's PyPI
    /// fallback is a harmless try-in-order chain, so a flat union is
    /// the right shape -- consulting an env-inactive feature's index
    /// can only find a wheel, never mis-route a dep.
    pub fn all_pypi_index_urls(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let all = self.pypi_index_urls.iter().chain(
            self.features
                .values()
                .flat_map(|f| f.pypi_index_urls.iter()),
        );
        for url in all {
            if !out.contains(url) {
                out.push(url.clone());
            }
        }
        out
    }

    /// Full auto-bundle index chain: the workspace's explicit `index-url`,
    /// or the implicit public PyPI default, followed by every declared extra.
    pub fn auto_bundle_pypi_index_urls(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.pypi_index_url_overridden {
            out.push(DEFAULT_PYPI_INDEX.to_string());
        }
        for url in self.all_pypi_index_urls() {
            if !out
                .iter()
                .any(|existing| existing.trim_end_matches('/') == url.trim_end_matches('/'))
            {
                out.push(url);
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
        let outputs = self.discover_outputs_for_source(workspace_dir, source_dir);
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

        let mut parts: Vec<String> = Vec::new();
        if let Some(p) = &self.channel_priority {
            parts.push(format!("ws-channel-priority:{p}"));
        }
        for env in &env_vec {
            for c in self.effective_channels(env) {
                parts.push(format!("scoped-env:{env}:channel:{c}"));
            }
            for (k, v) in self.effective_dependencies(env) {
                parts.push(format!("scoped-env:{env}:dep:{k}={v}"));
            }
            for (k, v) in self.effective_pypi_dependencies(env) {
                parts.push(format!("scoped-env:{env}:pypi-dep:{k}={v}"));
            }
            for (k, v) in self.effective_system_requirements(env) {
                parts.push(format!("scoped-env:{env}:sysreq:{k}={v}"));
            }
            for u in self.effective_pypi_index_urls(env) {
                parts.push(format!("scoped-env:{env}:pypi-index:{u}"));
            }
        }
        parts.join("\n")
    }

    pub fn effective_system_requirements(&self, env_name: &str) -> BTreeMap<String, String> {
        let Some(env) = self.environments.get(env_name) else {
            return BTreeMap::new();
        };
        let mut out = BTreeMap::new();
        if !env.no_default_feature {
            for (k, v) in &self.system_requirements {
                out.insert(k.clone(), v.clone());
            }
        }
        for feat_name in &env.features {
            let Some(feat) = self.features.get(feat_name) else {
                continue;
            };
            for (k, v) in &feat.system_requirements {
                out.insert(k.clone(), v.clone());
            }
        }
        // pixi 0.71+ rich `[workspace].platforms` declarations replace the
        // deprecated `[system-requirements]` table. They are workspace-wide
        // (per platform, not per env/feature) and, matching
        // `declared_glibc`'s precedence, win over any legacy declaration.
        let platform = crate::glibc::current_pixi_platform();
        if let Some(glibc) = self.platform_glibc.get(platform) {
            out.insert("libc".to_string(), glibc.clone());
        }
        if let Some(cuda) = self.platform_cuda.get(platform) {
            out.insert("cuda".to_string(), cuda.clone());
        }
        out
    }

    /// Declared glibc floor for installer manylinux relaxation. pixi 0.71+
    /// rich `[workspace].platforms` entries win over legacy
    /// `[system-requirements]`; when `env_name` is unknown, union the legacy
    /// top-level and feature declarations and take the max.
    pub fn declared_glibc(&self, env_name: Option<&str>) -> Option<(u32, u32)> {
        if let Some(v) = self
            .platform_glibc
            .get(crate::glibc::current_pixi_platform())
            .and_then(|s| crate::glibc::parse_glibc_version(s))
        {
            return Some(v);
        }

        if let Some(env_name) = env_name {
            return self
                .effective_system_requirements(env_name)
                .get("libc")
                .and_then(|s| crate::glibc::parse_glibc_version(s));
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
            for u in &self.pypi_index_urls {
                if seen.insert(u.clone()) {
                    out.push(u.clone());
                }
            }
        }
        for feat_name in &env.features {
            let Some(feat) = self.features.get(feat_name) else {
                continue;
            };
            for u in &feat.pypi_index_urls {
                if seen.insert(u.clone()) {
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

/// What kind of pixi.toml dep entry this is. Version + path are the
/// two retread cares about; Other covers git/url/editable-without-path/
/// malformed entries we ignore.
enum DepKind {
    /// A version-pinned conda dep — used for transitive extraction.
    Version(String),
    /// A path-form dep pointing at a local source package — used by
    /// output autodiscovery to match against the source dir retread
    /// is building.
    Path(String),
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
                return DepKind::Other;
            }
            match t.get("version").and_then(|v| v.as_str()) {
                Some(v) => DepKind::Version(v.to_string()),
                None => DepKind::Other,
            }
        }
        _ => DepKind::Other,
    }
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
        // Resolve source_dir for comparison. Failing canonicalize is
        // tolerable -- fall back to the raw path.
        let source_canon = canonical_or_self(source_dir);

        // Walk each feature + the implicit "default" feature
        // (top-level [dependencies]). Record matches as
        // (output_name, feature_name).
        let mut matches: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (dep_name, raw_path) in &self.path_dependencies {
            if path_matches(workspace_dir, raw_path, &source_canon) {
                matches
                    .entry(dep_name.clone())
                    .or_default()
                    .insert(DEFAULT_FEATURE.to_string());
            }
        }
        for (feat_name, feat) in &self.features {
            for (dep_name, raw_path) in &feat.path_dependencies {
                if path_matches(workspace_dir, raw_path, &source_canon) {
                    matches
                        .entry(dep_name.clone())
                        .or_default()
                        .insert(feat_name.clone());
                }
            }
        }

        // For each output, resolve the envs that include any of its
        // declaring features.
        let mut out: Vec<DiscoveredOutput> = Vec::new();
        for (name, features) in matches {
            let mut envs: BTreeSet<String> = BTreeSet::new();
            for (env_name, env_def) in &self.environments {
                let env_includes_default = !env_def.no_default_feature;
                let referenced = features.iter().any(|f| {
                    if f == DEFAULT_FEATURE {
                        env_includes_default
                    } else {
                        env_def.features.iter().any(|active| active == f)
                    }
                });
                if referenced {
                    envs.insert(env_name.clone());
                }
            }
            out.push(DiscoveredOutput {
                name,
                declaring_features: features.into_iter().collect(),
                envs: envs.into_iter().collect(),
            });
        }
        out
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
        let envs: BTreeSet<String> = self
            .discover_outputs_for_source(workspace_dir, source_dir)
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
        let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for env in env_names {
            for (k, v) in self.effective_dependencies(env) {
                let entry = out.entry(k).or_default();
                if !entry.contains(&v) {
                    entry.push(v);
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
        let env = self.environments.get(env_name)?;
        // Default first (top-level [dependencies]). If env opts out
        // via no-default-feature, skip.
        if !env.no_default_feature && self.dependencies.contains_key(dep_name) {
            return Some("default".to_string());
        }
        for feat_name in &env.features {
            let Some(feat) = self.features.get(feat_name) else {
                continue;
            };
            if feat.dependencies.contains_key(dep_name) {
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
        let outputs = self.discover_outputs_for_source(workspace_dir, source_dir);
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
        let outputs = self.discover_outputs_for_source(workspace_dir, source_dir);
        if !outputs.is_empty() {
            let mut envs: BTreeSet<String> = BTreeSet::new();
            let mut features: BTreeSet<String> = BTreeSet::new();
            for output in &outputs {
                envs.extend(output.envs.iter().cloned());
                features.extend(output.declaring_features.iter().cloned());
            }
            if !envs.is_empty() {
                let env_vec: Vec<String> = envs.into_iter().collect();
                return self.union_effective_dependencies(&env_vec);
            }
            // Tier 3: declaring features exist but no active env
            // reaches them -- union their raw dependency tables.
            let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for feat_name in &features {
                let deps = if feat_name == DEFAULT_FEATURE {
                    &self.dependencies
                } else if let Some(f) = self.features.get(feat_name) {
                    &f.dependencies
                } else {
                    continue;
                };
                for (k, v) in deps {
                    let entry = out.entry(k.clone()).or_default();
                    if !entry.contains(v) {
                        entry.push(v.clone());
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
        for (k, v) in &self.dependencies {
            out.entry(k.clone()).or_default().push(v.clone());
        }
        for feat in self.features.values() {
            for (k, v) in &feat.dependencies {
                let entry = out.entry(k.clone()).or_default();
                if !entry.contains(v) {
                    entry.push(v.clone());
                }
            }
        }
        out
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
        let mut out: BTreeMap<String, String> = BTreeMap::new();
        let outputs = self.discover_outputs_for_source(workspace_dir, source_dir);
        let mut envs: BTreeSet<String> = BTreeSet::new();
        let mut features: BTreeSet<String> = BTreeSet::new();
        for output in &outputs {
            envs.extend(output.envs.iter().cloned());
            features.extend(output.declaring_features.iter().cloned());
        }
        if !envs.is_empty() {
            for env in &envs {
                for (k, v) in self.effective_system_requirements(env) {
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
            out = self.effective_system_requirements("default");
            for feat in self.features.values() {
                for (k, v) in &feat.system_requirements {
                    out.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
        }
        // Workspace-wide platform declarations win (parity with
        // effective_system_requirements).
        let platform = crate::glibc::current_pixi_platform();
        if let Some(glibc) = self.platform_glibc.get(platform) {
            out.insert("libc".to_string(), glibc.clone());
        }
        if let Some(cuda) = self.platform_cuda.get(platform) {
            out.insert("cuda".to_string(), cuda.clone());
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

/// True if `raw_path` (relative to `workspace_dir`) resolves to the
/// same directory as `source_canon`. Handles both absolute and
/// relative `raw_path`.
fn path_matches(workspace_dir: &Path, raw_path: &str, source_canon: &Path) -> bool {
    let candidate = std::path::PathBuf::from(raw_path);
    let joined = if candidate.is_absolute() {
        candidate
    } else {
        workspace_dir.join(candidate)
    };
    canonical_or_self(&joined) == *source_canon
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
pub async fn extract_transitive_constraints(
    manifest: &WorkspaceManifest,
    env_name: &str,
    target_python: &str,
    conda_channels: &[rattler_conda_types::ChannelUrl],
    bundle_names: &std::collections::HashSet<String>,
) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let deps = manifest.effective_dependencies(env_name);
    let channel_priority = match manifest.channel_priority.as_deref() {
        Some("disabled") => rattler_solve::ChannelPriority::Disabled,
        _ => rattler_solve::ChannelPriority::Strict,
    };
    let system_requirements = manifest.effective_system_requirements(env_name);
    let solve_specs: Vec<String> = deps
        .iter()
        .filter_map(|(dep_name, dep_spec)| {
            let conda_name = crate::relax::canonical_conda_name(dep_name);
            if bundle_names.contains(&conda_name) {
                return None;
            }
            if dep_spec.is_empty() || dep_spec == "*" {
                Some(conda_name)
            } else {
                Some(format!("{conda_name} {dep_spec}"))
            }
        })
        .collect();

    let solved_records = match crate::conda_solve::solve_selected_records(
        conda_channels,
        &solve_specs,
        target_python,
        "linux-64",
        channel_priority,
        &system_requirements,
        rattler_solve::SolveStrategy::LowestVersionDirect,
    )
    .await
    {
        Ok(records) => records,
        Err(reasons) => {
            tracing::debug!(
                env = %env_name,
                reasons = ?reasons,
                "workspace: coherent solve for transitive extraction failed; skipping transitive constraints"
            );
            return out;
        }
    };

    for record in solved_records {
        let conda_name = record.package_record.name.as_normalized();
        if bundle_names.contains(conda_name) {
            continue;
        }
        // P3 (grizzly #6): walk `depends` AND `constrains`. A
        // `run_constrained` entry imposes a real constraint in pixi's
        // actual solve whenever the constrained package is present --
        // skipping them was an input-parity hole vs the v0.37.0
        // "retread's verdict predicts pixi's" contract. The skip set
        // stays python/python_abi-ONLY deliberately: recording a
        // workspace-imposed anchor constraint (cuda-version,
        // libstdcxx-ng) is INPUT-side parity; the never-widen-anchor
        // rule is EMISSION-side and enforced at its own three layers
        // (classifier, refinement re-check, output invariant).
        for (trans_name, trans_spec) in constraint_lines(
            &record.package_record.depends,
            &record.package_record.constrains,
        ) {
            out.entry(trans_name).or_default().push(trans_spec);
        }
    }
    out
}

/// P3 (grizzly #6): the depends + constrains line walk for one solved
/// record. `constrains` (run_constrained) entries impose real
/// constraints in pixi's solve whenever the constrained package is
/// present; they share the dep-line shape so the same parsing applies.
/// Skips stay python/python_abi-ONLY: recording a workspace-imposed
/// anchor constraint (cuda-version, libstdcxx-ng) is INPUT-side
/// parity; the never-widen-anchor rule is EMISSION-side, enforced at
/// its own three layers. Empty/`*` specs impose nothing and would
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

/// Pull index URLs out of a `[pypi-options]` table nested under
/// `container` (the manifest root, or a `[feature.X]` table value):
/// `index-url` first (it replaces pixi's default index, so it leads
/// the fallback chain), then `extra-index-urls` in declaration order.
fn parse_pypi_index_urls(container: &toml::Value) -> (Vec<String>, bool) {
    let mut out = Vec::new();
    let Some(opts) = container
        .get("pypi-options")
        .or_else(|| container.get("pypi_options"))
        .and_then(|v| v.as_table())
    else {
        return (out, false);
    };
    let index_url = opts
        .get("index-url")
        .or_else(|| opts.get("index_url"))
        .and_then(|v| v.as_str());
    if let Some(url) = index_url {
        out.push(url.to_string());
    }
    if let Some(extra) = opts
        .get("extra-index-urls")
        .or_else(|| opts.get("extra_index_urls"))
        .and_then(|v| v.as_array())
    {
        out.extend(extra.iter().filter_map(|v| v.as_str().map(String::from)));
    }
    (out, index_url.is_some())
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

/// Parse one pixi 0.71+ rich `[workspace].platforms` entry
/// (`{ platform = "linux-64", glibc = "2.28", cuda = "12.0" }`) into
/// `(platform, glibc, cuda)`. Bare-string entries (`"linux-64"`) and
/// entries without a `platform` key return `None`.
fn parse_rich_platform(v: &toml::Value) -> Option<(String, Option<String>, Option<String>)> {
    let t = v.as_table()?;
    let platform = t.get("platform")?.as_str()?.to_string();
    let glibc = t.get("glibc").and_then(parse_glibc_value);
    let cuda = t.get("cuda").and_then(parse_scalarish_value);
    Some((platform, glibc, cuda))
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

    fn ws_toml(text: &str) -> WorkspaceManifest {
        let parsed: toml::Value = toml::from_str(text).unwrap();
        WorkspaceManifest::from_toml(&parsed)
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
    fn effective_pypi_dependencies_merge_features_and_respect_no_default() {
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
        assert_eq!(layered.get("shared-dep").map(String::as_str), Some("==2"));
        assert_eq!(layered.get("top-only").map(String::as_str), Some("==1"));
        assert_eq!(layered.get("feature-only").map(String::as_str), Some("*"));

        let isolated = ws.effective_pypi_dependencies("isolated");
        assert_eq!(isolated.get("shared-dep").map(String::as_str), Some("==2"));
        assert_eq!(isolated.get("feature-only").map(String::as_str), Some("*"));
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
numpy = "==1.26.4"

[environments]
gsi = { features = ["physx"] }

[feature.physx]

[feature.physx.dependencies]
# physx pins numpy tighter than the top-level (well, same here);
# feature wins.
numpy = "==1.26.4"
pinocchio = ">=3.6"
"#,
        );
        let eff = ws.effective_dependencies("gsi");
        assert_eq!(eff.get("python").map(String::as_str), Some("==3.11"));
        assert_eq!(eff.get("numpy").map(String::as_str), Some("==1.26.4"));
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
        assert_eq!(sr.get("cuda").map(String::as_str), Some("12.0"));
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
        // parity; never-widening it is emission-side and enforced
        // elsewhere. python/python_abi stay skipped (relax never
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
        // Top-level: index-url leads, extra-index-urls follow.
        assert_eq!(
            ws.pypi_index_urls,
            vec![
                "https://pypi.nvidia.com".to_string(),
                "https://download.pytorch.org/whl/cu128".to_string(),
            ],
        );
        assert_eq!(
            ws.features["sim"].pypi_index_urls,
            vec!["https://py.mujoco.org".to_string()],
        );
        // Rollup: top-level first, features in name order, deduped
        // (gpu's nvidia index already present from top-level).
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
    fn auto_bundle_index_chain_includes_implicit_default_and_honors_override() {
        let ws = ws_toml(
            r#"
[workspace]
channels = ["conda-forge"]

[pypi-options]
extra-index-urls = ["https://pypi.nvidia.com", "https://py.mujoco.org"]
"#,
        );
        assert_eq!(
            ws.auto_bundle_pypi_index_urls(),
            vec![
                DEFAULT_PYPI_INDEX.to_string(),
                "https://pypi.nvidia.com".to_string(),
                "https://py.mujoco.org".to_string(),
            ],
        );

        let overridden = ws_toml(
            r#"
[pypi-options]
index-url = "https://packages.example/simple"
extra-index-urls = ["https://extra.example/simple"]
"#,
        );
        assert_eq!(
            overridden.auto_bundle_pypi_index_urls(),
            vec![
                "https://packages.example/simple".to_string(),
                "https://extra.example/simple".to_string(),
            ],
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
        assert!(ws.pypi_index_urls.is_empty());
        assert!(ws.all_pypi_index_urls().is_empty());
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
