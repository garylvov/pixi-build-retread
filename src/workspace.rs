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
}

impl WorkspaceManifest {
    /// Open `workspace_dir/pixi.toml` and parse. Returns `None` if the
    /// file is missing or fundamentally malformed; individual
    /// malformed entries are skipped rather than aborting.
    pub fn load(workspace_dir: &Path) -> Option<Self> {
        let pixi_toml = workspace_dir.join("pixi.toml");
        let bytes = std::fs::read_to_string(&pixi_toml).ok()?;
        let parsed: toml::Value = toml::from_str(&bytes).ok()?;
        Some(Self::from_toml(&parsed))
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
fn path_matches(
    workspace_dir: &Path,
    raw_path: &str,
    source_canon: &Path,
) -> bool {
    let candidate = std::path::PathBuf::from(raw_path);
    let joined = if candidate.is_absolute() {
        candidate
    } else {
        workspace_dir.join(candidate)
    };
    canonical_or_self(&joined) == *source_canon
}

/// v0.32.0+: walk each conda dep an env declares, fetch its latest
/// target-python-compatible build from the workspace's channels, and
/// accumulate the `depends` constraints those builds carry. The result
/// is a map `dep_name -> list of constraint strings` telling retread
/// what each of THIS env's already-pinned conda packages requires
/// from OTHER packages.
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

    for (dep_name, dep_spec) in deps {
        let conda_name = conda_normalize(&dep_name);
        if bundle_names.contains(&conda_name) {
            // The bundle itself provides this; its constraints are
            // irrelevant for retread's own emission.
            continue;
        }
        let depends = crate::probe::fetch_latest_build_depends(
            conda_channels,
            &conda_name,
            &dep_spec,
            Some(target_python),
        )
        .await;
        for raw in depends {
            // Each entry looks like `"numpy >=1.26,<2"` or `"libstdcxx >=12"`
            // or `"python_abi 3.11.* *_cp311"` (build-string-bearing).
            // Skip python / python_abi — relax policy never widens
            // those and they'd just clutter the constraint map.
            let Some((trans_name, trans_spec)) = split_conda_dep_line(&raw) else {
                continue;
            };
            if trans_name == "python" || trans_name == "python_abi" {
                continue;
            }
            // Skip no-op specs. Empty + `*` both mean "any version"
            // and impose zero constraint. Including either in the
            // comma-AND join produces invalid match-specs that the
            // conda parser rejects (e.g. `pytorch >=1.4,==2.10.0,*`
            // -- the `*` collides syntactically with the version
            // operators). Filtering them here is safe because they
            // don't drop any actual constraint information.
            if trans_spec.is_empty() || trans_spec == "*" {
                continue;
            }
            out.entry(trans_name).or_default().push(trans_spec);
        }
    }
    out
}

/// Conda-normalize a package name: lowercase + replace `_` with `-`.
/// Mirrors `handler::conda_name_from` but kept local so workspace.rs
/// doesn't have to depend on handler.rs.
fn conda_normalize(s: &str) -> String {
    s.to_ascii_lowercase().replace('_', "-")
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
        assert_eq!(ws.dependencies.get("numpy").map(String::as_str), Some("==1.26.4"));
        assert_eq!(ws.dependencies.get("torch").map(String::as_str), Some(">=2.7"));
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
