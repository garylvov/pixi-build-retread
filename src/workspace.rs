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
    /// v1.3.0: top-level `[pypi-options]` index URLs -- `index-url`
    /// first, then `extra-index-urls` in declaration order. Feeds the
    /// cascade's PyPI fallback chain so workspace-declared private
    /// indexes are consulted when bundling PyPI-only deps, not just
    /// the `[retread-wheels]` entry indexes.
    pub pypi_index_urls: Vec<String>,
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
                if let Some(s) = parse_system_requirement_value(v) {
                    out.system_requirements.insert(k.clone(), s);
                }
            }
        }

        // v1.3.0: top-level [pypi-options] index URLs.
        out.pypi_index_urls = parse_pypi_index_urls(parsed);

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
                    // v0.37.0+ (D1): per-feature system-requirements.
                    if let Some(sysreqs) = fmap
                        .get("system-requirements")
                        .or_else(|| fmap.get("system_requirements"))
                        .and_then(|v| v.as_table())
                    {
                        for (k, v) in sysreqs {
                            if let Some(s) = parse_system_requirement_value(v) {
                                def.system_requirements.insert(k.clone(), s);
                            }
                        }
                    }
                    // v1.3.0: per-feature [pypi-options] index URLs.
                    def.pypi_index_urls = parse_pypi_index_urls(fvalue);
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

    let solved_records = match crate::solve_check::solve_selected_records(
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

/// v0.37.0+ (D1): parse one `[system-requirements]` value. pixi allows
/// either a bare scalar (`cuda = "12"`, sometimes a number) or a table
/// (`libc = { family = "glibc", version = "2.35" }`). Scalars are kept
/// verbatim; tables contribute their `version` field. Anything else
/// returns `None` so the caller skips it.
/// Pull index URLs out of a `[pypi-options]` table nested under
/// `container` (the manifest root, or a `[feature.X]` table value):
/// `index-url` first (it replaces pixi's default index, so it leads
/// the fallback chain), then `extra-index-urls` in declaration order.
fn parse_pypi_index_urls(container: &toml::Value) -> Vec<String> {
    let mut out = Vec::new();
    let Some(opts) = container
        .get("pypi-options")
        .or_else(|| container.get("pypi_options"))
        .and_then(|v| v.as_table())
    else {
        return out;
    };
    if let Some(url) = opts
        .get("index-url")
        .or_else(|| opts.get("index_url"))
        .and_then(|v| v.as_str())
    {
        out.push(url.to_string());
    }
    if let Some(extra) = opts
        .get("extra-index-urls")
        .or_else(|| opts.get("extra_index_urls"))
        .and_then(|v| v.as_array())
    {
        out.extend(extra.iter().filter_map(|v| v.as_str().map(String::from)));
    }
    out
}

fn parse_system_requirement_value(v: &toml::Value) -> Option<String> {
    if let Some(s) = v.as_str() {
        return Some(s.to_string());
    }
    if let Some(i) = v.as_integer() {
        return Some(i.to_string());
    }
    if let Some(f) = v.as_float() {
        return Some(f.to_string());
    }
    if let Some(t) = v.as_table()
        && let Some(ver) = t.get("version").and_then(|x| x.as_str())
    {
        return Some(ver.to_string());
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
    fn missing_env_returns_empty() {
        let ws = WorkspaceManifest::default();
        assert!(ws.effective_dependencies("nonexistent").is_empty());
        assert!(ws.effective_channels("nonexistent").is_empty());
    }
}
