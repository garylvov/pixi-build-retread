use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::manifest::{AppliedEdit, EntrySnapshot, ManifestEditor, TableKind, write_atomic};
use super::parse::Conflict;
use crate::handler::PypiToCondaMap;
use crate::relax::canonical_conda_name;

// Copied from src/conflict_classifier.rs:52 so solve has no dependency on cascade-era modules.
pub(crate) const ABI_ANCHOR_NAMES: &[&str] = &[
    // Python ABI: widening `python` strips the cp-tag constraint; every
    // wheel-emitted dep then collapses to "any python" and pixi picks
    // 3.14 against a 3.11-only workspace.
    "python",
    "python_abi",
    "pypy",
    // glibc / libc: the C runtime ABI. Conda-forge encodes the floor
    // via `__glibc` virtual + `libc` direct; widening these would
    // claim retread's output runs on any libc, which is almost never
    // true (we ship native wheels with manylinux_2_xx tags).
    "libc",
    "glibc",
    "__glibc",
    // libstdcxx / libcxx: the C++ runtime ABI. PyTorch and friends
    // pin libstdcxx-ng tightly; the workspace has matching pins;
    // widening retread's emission lets the solver pick a libstdcxx
    // older than what the wheels need.
    "libstdcxx-ng",
    "libstdcxx",
    "libcxx",
    "libcxx-devel",
    // CUDA runtime: workspace pins `cuda-version ==12.8` for a reason
    // (driver match, sm arch). Widening lets the solver pick cuda 13
    // and break every cuda-bindings/cuda-toolkit/torch interaction.
    "cuda-version",
    "__cuda",
    // Other rattler virtual packages (`__linux`, `__osx`, `__win`,
    // `__unix`, `__archspec`) are caught by the `__` prefix check
    // below. Arch-tagged compilers + binutils are caught by the
    // prefix list. `*_compiler` suffix is caught by the predicate.
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    #[serde(rename = "widen-conda")]
    WidenConda,
    Conda,
    PypiDep,
    PypiOverride,
}

impl Strategy {
    pub fn as_str(self) -> &'static str {
        match self {
            Strategy::WidenConda => "widen-conda",
            Strategy::Conda => "conda",
            Strategy::PypiDep => "pypi_dep",
            Strategy::PypiOverride => "pypi_override",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    CondaBoundary,
    PypiInternal,
    NoCandidatesEscalation,
}

impl Reason {
    pub fn slug(self) -> &'static str {
        match self {
            Reason::CondaBoundary => "conda-boundary",
            Reason::PypiInternal => "pypi-internal",
            Reason::NoCandidatesEscalation => "no-candidates-escalation",
        }
    }
}

/// `[tool.retread] relax-preference` -- which side of a `CondaWidenNeeded`
/// conflict retread relaxes first. Conda manifest = source of truth by
/// default: a pypi requirement conflicting with a conda pin gets a pypi
/// dependency-override (T1) rather than widening the conda pin (T2).
/// `--prefer-pypi` / `relax-preference = "pypi"` restores the historical
/// widen-first order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelaxPreference {
    Conda,
    Pypi,
}

impl RelaxPreference {
    pub fn from_config_str(s: &str) -> Self {
        if s.eq_ignore_ascii_case("pypi") {
            RelaxPreference::Pypi
        } else {
            RelaxPreference::Conda
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WidenCeilingPolicy {
    NextMajor,
    None,
    SameMinor,
}

impl WidenCeilingPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            WidenCeilingPolicy::NextMajor => "next-major",
            WidenCeilingPolicy::None => "none",
            WidenCeilingPolicy::SameMinor => "same-minor",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveLedger {
    pub version: u32,
    pub manifest: String,
    pub runs: Vec<LedgerRun>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerRun {
    pub started: String,
    pub env: String,
    pub manifest_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pixi_version: Option<String>,
    pub attempts: Vec<LedgerAttempt>,
    pub outcome: String,
    pub smoke_failed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kept_pins: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reverted: Vec<RevertedEdit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerAttempt {
    pub iter: u32,
    pub package: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    pub tier: String,
    pub strategy: String,
    pub conflict: String,
    pub source: String,
    pub ts: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_spec: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_spec: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ceiling_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<EntrySnapshot>,
    #[serde(default)]
    pub failed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevertedEdit {
    pub package: String,
    pub strategy: String,
}

impl SolveLedger {
    pub fn load(path: &Path, manifest_display: String) -> Result<Self> {
        if !path.exists() {
            return Ok(Self {
                version: 1,
                manifest: manifest_display,
                runs: Vec::new(),
            });
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let mut ledger: Self = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        if ledger.version != 1 {
            ledger.version = 1;
        }
        ledger.manifest = manifest_display;
        Ok(ledger)
    }

    pub fn write_atomic(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = serde_json::to_vec_pretty(self)?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn start_run(
        &mut self,
        env: String,
        manifest_sha256: String,
        pixi_version: Option<String>,
    ) -> usize {
        self.runs.push(LedgerRun {
            started: timestamp(),
            env,
            manifest_sha256,
            pixi_version,
            attempts: Vec::new(),
            outcome: "running".to_string(),
            smoke_failed: false,
            kept_pins: None,
            reverted: Vec::new(),
        });
        self.runs.len() - 1
    }

    pub fn finish_run(&mut self, run_idx: usize, outcome: &str) {
        if let Some(run) = self.runs.get_mut(run_idx) {
            run.outcome = outcome.to_string();
        }
    }

    pub fn seed_tried_state(
        &self,
        manifest_path: &Path,
        manifest_sha256: &str,
        editor: &ManifestEditor,
    ) -> TriedState {
        let manifest_key = manifest_path.to_string_lossy();
        let same_hash = self
            .runs
            .last()
            .map(|r| r.manifest_sha256 == manifest_sha256)
            .unwrap_or(false);
        let mut state = TriedState::default();
        if same_hash {
            for run in &self.runs {
                for a in &run.attempts {
                    state.mark(&a.package, parse_strategy(&a.strategy), a.failed);
                }
            }
        } else {
            for (package, strategy) in editor.sentinel_strategies() {
                state.mark(&package, strategy, false);
            }
        }
        let _ = manifest_key;
        state
    }
}

#[derive(Debug, Default, Clone)]
pub struct TriedState {
    tried: BTreeMap<String, BTreeSet<Strategy>>,
    failed: BTreeMap<String, BTreeSet<Strategy>>,
}

impl TriedState {
    pub fn has(&self, package: &str, strategy: Strategy) -> bool {
        self.tried
            .get(package)
            .map(|s| s.contains(&strategy))
            .unwrap_or(false)
    }

    pub fn mark(&mut self, package: &str, strategy: Strategy, failed: bool) {
        self.tried
            .entry(package.to_string())
            .or_default()
            .insert(strategy);
        if failed {
            self.failed
                .entry(package.to_string())
                .or_default()
                .insert(strategy);
        }
    }
}

pub struct RepairPlanner {
    pub feature: String,
    pub ceiling_policy: WidenCeilingPolicy,
    pub relax_preference: RelaxPreference,
    current_run_attempts: HashSet<(String, String, Strategy)>,
    /// PyPI name -> conda-forge candidate names, from the same
    /// parselmouth-backed dataset (`crate::handler::load_pypi_to_conda_map`,
    /// FALLBACK_PYPI_TO_CONDA merged in) the courier/auto-route path uses to
    /// build its `name_map`. Lets `CondaWidenNeeded` conflicts (which name
    /// the *pypi* package, e.g. `torch`) find a conda pin declared under a
    /// differently-named variant package (e.g. `pytorch-gpu`). Empty by
    /// default -- callers that don't wire it in (tests, `--offline`-style
    /// use) fall back to exact-name matching only.
    conda_name_map: PypiToCondaMap,
}

/// One conflict target flowing through the repair tiers: which package and
/// version the fix applies to, on which solve iteration, for which parsed
/// conflict. Bundled so every tier helper shares one signature.
struct PinTarget<'a> {
    package: &'a str,
    version: &'a str,
    iter: u32,
    conflict: &'a Conflict,
}

/// Optional ledger metadata for one attempt (only widens populate most of it).
#[derive(Default)]
struct AttemptDetails {
    old_spec: Option<String>,
    new_spec: Option<String>,
    ceiling_policy: Option<String>,
    before: Option<EntrySnapshot>,
    failed: bool,
}

#[derive(Debug)]
pub struct RepairOutcome {
    pub attempt: LedgerAttempt,
    pub extra_attempts: Vec<LedgerAttempt>,
    pub applied: Vec<AppliedEdit>,
    pub summary_line: String,
    /// Fix #20: set instead of (never alongside) `applied` when this
    /// repair's target came from a backend uv-closure conflict
    /// (`Conflict::CondaWidenNeeded::pack_name` is `Some`). The WORKSPACE
    /// manifest is left untouched; the caller (`retread solve`'s
    /// `driver.rs` / `retread lock`'s `lock.rs`) is responsible for
    /// actually writing this into the PACK manifest's
    /// `[package.build.config.retread-overrides]` table (via
    /// `crate::pack_overrides::write_override`) -- kept out of `repair()`
    /// itself so a `--dry-run` repair (which only ever mutates a throwaway
    /// `ManifestEditor`) never side-effects a real pack file.
    pub pack_override: Option<PackOverrideWrite>,
}

/// One pending pack-manifest `retread-overrides` write a repair wants
/// persisted. See [`RepairOutcome::pack_override`].
#[derive(Debug, Clone)]
pub struct PackOverrideWrite {
    /// The bundle/pack name (as the uv-closure conflict reported it).
    pub bundle: String,
    /// Absolute path to the pack's `pixi.toml` the override is written to.
    pub pack_pixi: PathBuf,
    /// PyPI package name the override applies to (e.g. `"torch"`).
    pub package: String,
    /// Conda-style spec written verbatim (e.g. `"==2.10.0"`).
    pub spec: String,
}

impl RepairPlanner {
    pub fn new(feature: String) -> Self {
        Self {
            feature,
            ceiling_policy: WidenCeilingPolicy::NextMajor,
            relax_preference: RelaxPreference::Conda,
            current_run_attempts: HashSet::new(),
            conda_name_map: PypiToCondaMap::new(),
        }
    }

    /// Wires in the parselmouth-backed pypi<->conda name family (see
    /// `conda_name_map` doc comment). `driver::run` loads this once per
    /// invocation via `crate::handler::load_pypi_to_conda_map` and passes it
    /// down so conflict matching agrees with the courier/auto-route path.
    pub fn with_conda_name_map(mut self, map: PypiToCondaMap) -> Self {
        self.conda_name_map = map;
        self
    }

    /// Single-sourced construction for both `retread solve` (driver.rs) and
    /// `retread lock` (lock.rs): wires the conda-pypi name map (aaf58c6) and
    /// relax-preference (59b5b40) identically so the two drivers can't drift
    /// out of sync again (the `retread lock` path previously constructed a
    /// bare planner and silently lost both fixes).
    pub fn configured(
        feature: String,
        conda_name_map: PypiToCondaMap,
        relax_preference: RelaxPreference,
    ) -> Self {
        Self::new(feature)
            .with_relax_preference(relax_preference)
            .with_conda_name_map(conda_name_map)
    }

    /// Resolves the conda-table name under which a `CondaWidenNeeded`
    /// conflict's pypi `package` is actually pinned by the user in the
    /// given `feature`, checking the parselmouth-backed name family before
    /// falling back to an exact-name match. Returns `None` if no candidate
    /// (including the pypi name itself) is a user-owned conda entry in
    /// that feature.
    fn resolve_conda_pin_name_in(
        &self,
        editor: &ManifestEditor,
        feature: &str,
        package: &str,
    ) -> Option<String> {
        let canon = canonical_conda_name(package);
        if let Some(candidates) = self.conda_name_map.get(&canon) {
            for candidate in candidates {
                if let Some(found) = self.resolve_conda_pin_name_or_variant(editor, feature, candidate) {
                    return Some(found);
                }
            }
        }
        // Exact-name fallback -- covers the identical-name case (conda
        // package name == pypi package name) and callers that didn't wire
        // in a name map at all.
        self.resolve_conda_pin_name_or_variant(editor, feature, package)
    }

    /// Checks `name` itself, then its `-gpu`/`-cpu` meta-package variants,
    /// for a user-owned conda entry. Covers the class of split accelerator
    /// meta-packages (`pytorch` / `pytorch-gpu` / `pytorch-cpu`,
    /// `tensorflow` / `tensorflow-gpu` / `tensorflow-cpu`, etc.) whose
    /// variant names aren't all individually listed as name-map candidates
    /// -- so a pin on ANY variant of a mapped conda name family is
    /// discoverable from the pypi name, not just the exact candidates the
    /// name map happens to enumerate.
    fn resolve_conda_pin_name_or_variant(
        &self,
        editor: &ManifestEditor,
        feature: &str,
        name: &str,
    ) -> Option<String> {
        if editor.has_user_entry(feature, TableKind::Conda, name) {
            return Some(name.to_string());
        }
        for suffix in ["-gpu", "-cpu"] {
            // Don't double-suffix a candidate that's already variant-named
            // (e.g. the name map may already list "pytorch-gpu" directly).
            if name.ends_with(suffix) {
                continue;
            }
            let variant = format!("{name}{suffix}");
            if editor.has_user_entry(feature, TableKind::Conda, &variant) {
                return Some(variant);
            }
        }
        None
    }

    /// Cross-feature version of [`Self::resolve_conda_pin_name`]: scans
    /// every feature table (own feature checked first, so existing
    /// single-feature callers see no behavior change) for a user-owned
    /// conda entry matching `package` (directly or via the name map),
    /// returning `(owning_feature, conda_table_name)`. Backs both the
    /// attribution-chain walk (bug: conflict misattributed to a leaf
    /// package with no user pin) and feature-scoped repair placement
    /// (bug: edits always landing in `default` instead of the consuming
    /// feature/pack that actually declares the pin). Used only on the
    /// WORKSPACE (non-pack) path; backend-closure conflicts resolve their
    /// override version via [`Self::resolve_pack_override`] instead.
    fn resolve_conda_pin_owner(
        &self,
        editor: &ManifestEditor,
        package: &str,
    ) -> Option<(String, String)> {
        if let Some(name) = self.resolve_conda_pin_name_in(editor, &self.feature, package) {
            return Some((self.feature.clone(), name));
        }
        for feature in editor.feature_names() {
            if feature == self.feature {
                continue;
            }
            if let Some(name) = self.resolve_conda_pin_name_in(editor, &feature, package) {
                return Some((feature, name));
            }
        }
        None
    }

    /// Fix #20: resolve the exact override version for a BACKEND-closure
    /// conflict from the CONSUMING env's conda deps, scoped to the failing
    /// `pack_name`. Walks candidate packages (the conflict's own package
    /// first, then its `requiring_chain` outward -- e.g. cuda-bindings
    /// footer -> torch -> ...), mapping each pypi name to its conda name
    /// family (name map + `-gpu`/`-cpu` variants) and looking it up in
    /// `WorkspaceManifest::consuming_env_dependencies(pack)` -- the exact
    /// env-scoped, sentinel-AGNOSTIC dep map fix 6f9524a already exposes.
    ///
    /// Two things this gets right that the old workspace `resolve_conda_
    /// pin_owner` scan got wrong (override20 brief ACCEPTANCE RUN #6b):
    /// (defect 2) it is scoped to the consuming env, so an unrelated
    /// feature's pin (e.g. `feature.holosoma`, not in `isaaclab-gpu-latest`)
    /// is never matched; and it reads the RAW dep spec, so the consuming
    /// env's own `# retread:pin`-sentineled torch-family pins (in
    /// `feature.gpu`) ARE the source of truth -- exactly the conda pins the
    /// owner policy says to relax the pypi side onto. Returns
    /// `(pypi_package, exact_version)` for the first untried candidate with
    /// an exact `==X` consuming-env pin, or `None`.
    fn resolve_pack_override(
        &self,
        editor: &ManifestEditor,
        tried: &TriedState,
        conflict_package: &str,
        requiring_chain: &[String],
        pack_name: &str,
    ) -> Option<(String, String)> {
        let ws = crate::workspace::WorkspaceManifest::load(editor.project_dir())?;
        let pack_dir = resolve_pack_dir(&ws, editor.project_dir(), pack_name)?;
        let deps = ws.consuming_env_dependencies(editor.project_dir(), &pack_dir);

        let mut candidates: Vec<&str> = vec![conflict_package];
        candidates.extend(requiring_chain.iter().map(String::as_str));
        for pkg in candidates {
            if tried.has(pkg, Strategy::PypiOverride) || is_abi_anchor(pkg) {
                continue;
            }
            for conda_name in conda_name_family(pkg, &self.conda_name_map) {
                if let Some(specs) = deps.get(&conda_name)
                    && let Some(version) = exact_pin_version(specs)
                {
                    return Some((pkg.to_string(), version));
                }
            }
        }
        None
    }

    /// Runs [`Self::pypi_override_from_conda`] with `self.feature`
    /// temporarily switched to `feature` (the feature that owns the conda
    /// pin the override is derived from), so the emitted
    /// `pypi-options.dependency-overrides` entry lands in the same
    /// feature/pack table as the pin, not wherever the planner happened to
    /// be constructed for (`default` for `retread lock`). Restores
    /// `self.feature` before returning either way. `None` means the tier
    /// was unavailable (ABI anchor / already tried / oscillation guard) --
    /// caller falls through to the next tier.
    fn try_pypi_override_in_feature(
        &mut self,
        editor: &mut ManifestEditor,
        tried: &mut TriedState,
        target: &PinTarget<'_>,
        conda_version: &str,
        feature: &str,
    ) -> Option<RepairOutcome> {
        let saved = std::mem::replace(&mut self.feature, feature.to_string());
        let result = self.pypi_override_from_conda(editor, tried, target, conda_version);
        self.feature = saved;
        result.ok()
    }

    pub fn with_relax_preference(mut self, pref: RelaxPreference) -> Self {
        self.relax_preference = pref;
        self
    }

    pub fn repair(
        &mut self,
        editor: &mut ManifestEditor,
        tried: &mut TriedState,
        conflict: &Conflict,
        iter: u32,
    ) -> std::result::Result<RepairOutcome, String> {
        match conflict {
            Conflict::NoCandidates { package, version } => {
                let target = PinTarget {
                    package,
                    version,
                    iter,
                    conflict,
                };
                self.no_candidates(editor, tried, &target)
            }
            Conflict::CondaBoundary { package, version } => {
                let target = PinTarget {
                    package,
                    version,
                    iter,
                    conflict,
                };
                self.boundary(editor, tried, &target, Reason::CondaBoundary)
            }
            Conflict::PypiInternal { package, version } => {
                let target = PinTarget {
                    package,
                    version,
                    iter,
                    conflict,
                };
                self.override_pin(editor, tried, &target, Reason::PypiInternal)
            }
            Conflict::CondaWidenNeeded {
                package,
                op,
                floor,
                conda_version,
                requiring_chain,
                ..
            } => {
                let target = PinTarget {
                    package,
                    version: floor,
                    iter,
                    conflict,
                };
                self.conda_widen_needed(editor, tried, &target, op, conda_version, requiring_chain)
            }
        }
    }

    /// `CondaWidenNeeded` dispatch: conda-as-truth (T1, default) tries a
    /// pypi dependency-override derived from the conda pin first, and only
    /// falls back to widening the conda pin itself (T2) for tool-owned
    /// (sentineled) pins, when `relax-preference = "pypi"` / `--prefer-pypi`
    /// is set, or when the override tier is unavailable (ABI anchor,
    /// already tried, oscillation guard).
    fn conda_widen_needed(
        &mut self,
        editor: &mut ManifestEditor,
        tried: &mut TriedState,
        target: &PinTarget<'_>,
        op: &str,
        conda_version: &str,
        requiring_chain: &[String],
    ) -> std::result::Result<RepairOutcome, String> {
        let package = target.package;
        let pack_name = match target.conflict {
            Conflict::CondaWidenNeeded { pack_name, .. } => pack_name.clone(),
            _ => None,
        };

        // Fix #20: a BACKEND uv-closure conflict (`pack_name: Some`) is
        // resolved on its own path -- the override version is derived from
        // the CONSUMING env's conda deps (env-scoped + sentinel-agnostic;
        // defect 2) and WRITTEN to the pack's own retread-overrides table
        // (the only table the pack's closure reads; defect 1). The
        // workspace `resolve_conda_pin_owner` scan below is NOT used for
        // this class -- it matched an unrelated feature's pin and wrote a
        // table the closure never reads (ACCEPTANCE RUN #6b).
        if self.relax_preference == RelaxPreference::Conda
            && let Some(bundle) = &pack_name
            && let Some((pkg, version)) =
                self.resolve_pack_override(editor, tried, package, requiring_chain, bundle)
            && let Some(out) = self.try_pack_override(tried, target, &pkg, &version, bundle, editor)
        {
            return Ok(out);
        }

        // Resolve which conda-table name (and feature) actually carries the
        // user's pin -- may differ from the pypi `package` name (e.g. pypi
        // `torch` pinned via conda `pytorch-gpu`) and from the feature the
        // planner was constructed for (e.g. `retread lock`'s `default`);
        // see `resolve_conda_pin_owner`.
        let owner = self.resolve_conda_pin_owner(editor, package);
        if self.relax_preference == RelaxPreference::Conda && pack_name.is_none() {
            if let Some((feature, _name)) = owner.clone() {
                if !tried.has(package, Strategy::PypiOverride)
                    && let Some(out) = self.try_pypi_override_in_feature(
                        editor,
                        tried,
                        target,
                        conda_version,
                        &feature,
                    )
                {
                    return Ok(out);
                }
            } else {
                // ATTRIBUTION FIX: the footer package (the transitive
                // SYMPTOM the conda-provenance regex names, e.g.
                // `cuda-bindings`) has no user conda pin of its own, so T1
                // as written can never fire and every tier below would
                // chase a spec that can never satisfy the real (anchor-
                // derived) conda constraint. Walk the bounded "Because X
                // depends on <footer>" requiring-chain outward (nearest
                // requirer first, e.g. `torch`) until we find an ancestor
                // that DOES carry a user conda pin (directly or via the
                // name map, e.g. `torch` -> `pytorch-gpu`), and re-attribute
                // the T1 override to THAT package/version instead.
                for requirer in requiring_chain {
                    if tried.has(requirer.as_str(), Strategy::PypiOverride) {
                        continue;
                    }
                    let Some((requirer_feature, conda_name)) =
                        self.resolve_conda_pin_owner(editor, requirer)
                    else {
                        continue;
                    };
                    let pin_value = editor
                        .entry_snapshot(&requirer_feature, TableKind::Conda, &conda_name)
                        .value;
                    let Some(pin_value) = pin_value else {
                        continue;
                    };
                    let reattributed = PinTarget {
                        package: requirer.as_str(),
                        version: target.version,
                        iter: target.iter,
                        conflict: target.conflict,
                    };
                    if let Some(out) = self.try_pypi_override_in_feature(
                        editor,
                        tried,
                        &reattributed,
                        &pin_value,
                        &requirer_feature,
                    ) {
                        return Ok(out);
                    }
                }
            }
        }
        let (widen_feature, widen_target_name) =
            owner.unwrap_or_else(|| (self.feature.clone(), package.to_string()));
        let saved = std::mem::replace(&mut self.feature, widen_feature);
        let result = if !tried.has(package, Strategy::WidenConda) {
            self.widen(editor, tried, target, op, &widen_target_name)
        } else {
            self.boundary(editor, tried, target, Reason::CondaBoundary)
        };
        self.feature = saved;
        result
    }

    /// Fix #20 (defect 1): emit a [`PackOverrideWrite`] for a
    /// backend-closure conflict -- the override is written to the failing
    /// PACK's own `[package.build.config.retread-overrides]` table (the
    /// only override table the pack's uv closure consumes; a workspace
    /// `pypi-options.dependency-overrides` write is inert here, see
    /// override20 brief ACCEPTANCE RUN #6b). No manifest is touched by
    /// `repair()` itself; the caller performs the pack write, so
    /// `--dry-run` stays side-effect-free. `pkg`/`version` come from
    /// [`Self::resolve_pack_override`] (consuming-env, sentinel-agnostic).
    fn try_pack_override(
        &mut self,
        tried: &mut TriedState,
        target: &PinTarget<'_>,
        pkg: &str,
        version: &str,
        bundle: &str,
        editor: &ManifestEditor,
    ) -> Option<RepairOutcome> {
        if is_abi_anchor(pkg) {
            return None;
        }
        // Oscillation guard shares the planner-run set with the other tiers.
        if self
            .guard_oscillation(pkg, version, Strategy::PypiOverride)
            .is_err()
        {
            return None;
        }
        let pack_pixi = crate::workspace::WorkspaceManifest::load(editor.project_dir())
            .and_then(|ws| resolve_pack_dir(&ws, editor.project_dir(), bundle))
            .map(|dir| dir.join("pixi.toml"))?;
        tried.mark(pkg, Strategy::PypiOverride, false);
        let new_spec = format!("=={version}");
        // The ledger attempt is attributed to the re-attributed package
        // (e.g. `torch`), not the footer symptom (`cuda-bindings`).
        let attempt_target = PinTarget {
            package: pkg,
            version,
            iter: target.iter,
            conflict: target.conflict,
        };
        Some(RepairOutcome {
            attempt: self.ledger_attempt(
                &attempt_target,
                Strategy::PypiOverride,
                "retread",
                AttemptDetails {
                    old_spec: None,
                    new_spec: Some(new_spec.clone()),
                    ..AttemptDetails::default()
                },
            ),
            extra_attempts: Vec::new(),
            summary_line: format!(
                "would add [{bundle} :: retread-overrides] {pkg} = \"{new_spec}\"  (tier: pypi_override; conda-as-truth; pack manifest)",
            ),
            applied: Vec::new(),
            pack_override: Some(PackOverrideWrite {
                bundle: bundle.to_string(),
                pack_pixi,
                package: pkg.to_string(),
                spec: new_spec,
            }),
        })
    }

    /// T1: relax the pypi requirement to accept the conda-provided version
    /// instead of touching the (user-owned) conda pin. The conda pin is
    /// left byte-for-byte untouched -- conda manifest is source of truth.
    /// This is the WORKSPACE path (`pack_name: None`); backend-closure
    /// conflicts use [`Self::try_pack_override`] instead.
    fn pypi_override_from_conda(
        &mut self,
        editor: &mut ManifestEditor,
        tried: &mut TriedState,
        target: &PinTarget<'_>,
        conda_version: &str,
    ) -> std::result::Result<RepairOutcome, String> {
        let package = target.package;
        self.guard_anchor(package)?;
        let version = strip_version_op(conda_version);
        self.guard_oscillation(package, version, Strategy::PypiOverride)?;

        let edit = editor.set_pypi_override_from_conda(&self.feature, package, version);
        let old_spec = edit.before.value.clone();
        tried.mark(package, Strategy::PypiOverride, false);
        Ok(RepairOutcome {
            attempt: self.ledger_attempt(
                target,
                Strategy::PypiOverride,
                "retread",
                AttemptDetails {
                    old_spec,
                    new_spec: Some(format!("=={version}")),
                    ..AttemptDetails::default()
                },
            ),
            extra_attempts: Vec::new(),
            summary_line: format!(
                "would add [{}] {} = \"=={}\"  (tier: pypi_override; conda-as-truth)",
                table_label(&self.feature, TableKind::Override),
                package,
                version
            ),
            applied: vec![edit],
            pack_override: None,
        })
    }

    fn no_candidates(
        &mut self,
        editor: &mut ManifestEditor,
        tried: &mut TriedState,
        target: &PinTarget<'_>,
    ) -> std::result::Result<RepairOutcome, String> {
        let package = target.package;
        if tried.has(package, Strategy::Conda) && !tried.has(package, Strategy::PypiDep) {
            self.pypi_pin(editor, tried, target, Reason::NoCandidatesEscalation)
        } else if tried.has(package, Strategy::PypiDep)
            && !tried.has(package, Strategy::PypiOverride)
        {
            self.override_pin(editor, tried, target, Reason::NoCandidatesEscalation)
        } else {
            Err(package.to_string())
        }
    }

    fn boundary(
        &mut self,
        editor: &mut ManifestEditor,
        tried: &mut TriedState,
        target: &PinTarget<'_>,
        reason: Reason,
    ) -> std::result::Result<RepairOutcome, String> {
        let package = target.package;
        if !tried.has(package, Strategy::Conda) {
            if editor.has_user_entry(&self.feature, TableKind::Conda, package) {
                tried.mark(package, Strategy::Conda, false);
                let attempt =
                    self.ledger_attempt(target, Strategy::Conda, "user", AttemptDetails::default());
                return self.pypi_pin(editor, tried, target, reason).map(|mut out| {
                    out.summary_line = format!(
                        "{}; skipped user-owned conda pin and {}",
                        describe_attempt(&attempt),
                        out.summary_line
                    );
                    out.extra_attempts.push(attempt);
                    out
                });
            }
            return self.conda_pin(editor, tried, target, reason);
        }
        if !tried.has(package, Strategy::PypiDep) {
            return self.pypi_pin(editor, tried, target, reason);
        }
        if !tried.has(package, Strategy::PypiOverride) {
            return self.override_pin(editor, tried, target, reason);
        }
        Err(package.to_string())
    }

    fn conda_pin(
        &mut self,
        editor: &mut ManifestEditor,
        tried: &mut TriedState,
        target: &PinTarget<'_>,
        reason: Reason,
    ) -> std::result::Result<RepairOutcome, String> {
        let (package, version) = (target.package, target.version);
        self.guard_anchor(package)?;
        self.guard_oscillation(package, version, Strategy::Conda)?;
        let edit = editor.set_conda_pin(&self.feature, package, version, reason);
        tried.mark(package, Strategy::Conda, false);
        Ok(self.outcome(
            target,
            Strategy::Conda,
            vec![edit],
            AttemptDetails::default(),
        ))
    }

    fn pypi_pin(
        &mut self,
        editor: &mut ManifestEditor,
        tried: &mut TriedState,
        target: &PinTarget<'_>,
        reason: Reason,
    ) -> std::result::Result<RepairOutcome, String> {
        let (package, version) = (target.package, target.version);
        self.guard_anchor(package)?;
        self.guard_oscillation(package, version, Strategy::PypiDep)?;
        let mut edits = Vec::new();
        if let Some(edit) = editor.remove_retread_entry(&self.feature, TableKind::Conda, package) {
            edits.push(edit);
        }
        edits.push(editor.set_pypi_pin(&self.feature, package, version, reason));
        tried.mark(package, Strategy::PypiDep, false);
        Ok(self.outcome(target, Strategy::PypiDep, edits, AttemptDetails::default()))
    }

    fn override_pin(
        &mut self,
        editor: &mut ManifestEditor,
        tried: &mut TriedState,
        target: &PinTarget<'_>,
        reason: Reason,
    ) -> std::result::Result<RepairOutcome, String> {
        let (package, version) = (target.package, target.version);
        self.guard_anchor(package)?;
        if tried.has(package, Strategy::PypiOverride) {
            return Err(package.to_string());
        }
        self.guard_oscillation(package, version, Strategy::PypiOverride)?;
        let mut edits = Vec::new();
        if let Some(edit) = editor.remove_retread_entry(&self.feature, TableKind::Pypi, package) {
            edits.push(edit);
        }
        edits.push(editor.set_pypi_override(&self.feature, package, version, reason));
        tried.mark(package, Strategy::PypiOverride, false);
        Ok(self.outcome(
            target,
            Strategy::PypiOverride,
            edits,
            AttemptDetails::default(),
        ))
    }

    fn widen(
        &mut self,
        editor: &mut ManifestEditor,
        tried: &mut TriedState,
        target: &PinTarget<'_>,
        op: &str,
        conda_name: &str,
    ) -> std::result::Result<RepairOutcome, String> {
        let (package, floor) = (target.package, target.version);
        // Guard on the actual conda-table name being widened (may differ
        // from the pypi `package` name via the name-family match), so ABI
        // anchors are protected regardless of which side named the conflict.
        self.guard_anchor(conda_name)?;
        let spec = widen_spec(op, floor, self.ceiling_policy);
        self.guard_oscillation(package, &spec, Strategy::WidenConda)?;
        let edit = editor.set_conda_widen(&self.feature, conda_name, &spec);
        let old_spec = edit.before.value.clone();
        tried.mark(package, Strategy::WidenConda, false);
        Ok(self.outcome(
            target,
            Strategy::WidenConda,
            vec![edit.clone()],
            AttemptDetails {
                old_spec,
                new_spec: Some(spec),
                ceiling_policy: Some(self.ceiling_policy.as_str().to_string()),
                before: Some(edit.before),
                failed: false,
            },
        ))
    }

    fn outcome(
        &self,
        target: &PinTarget<'_>,
        strategy: Strategy,
        applied: Vec<AppliedEdit>,
        details: AttemptDetails,
    ) -> RepairOutcome {
        let (package, version) = (target.package, target.version);
        let summary_line = match strategy {
            Strategy::WidenConda => format!(
                "would add [{}] {} = \"{}\"  (tier: widen-conda)",
                table_label(&self.feature, TableKind::Conda),
                package,
                details.new_spec.as_deref().unwrap_or("")
            ),
            Strategy::Conda => format!(
                "would add [{}] {} = \"=={}\"  (tier: conda)",
                table_label(&self.feature, TableKind::Conda),
                package,
                version
            ),
            Strategy::PypiDep => format!(
                "would add [{}] {} = \"=={}\"  (tier: pypi_dep)",
                table_label(&self.feature, TableKind::Pypi),
                package,
                version
            ),
            Strategy::PypiOverride => format!(
                "would add [{}] {} = \"=={}\"  (tier: pypi_override)",
                table_label(&self.feature, TableKind::Override),
                package,
                version
            ),
        };
        RepairOutcome {
            attempt: self.ledger_attempt(target, strategy, "retread", details),
            extra_attempts: Vec::new(),
            summary_line,
            applied,
            pack_override: None,
        }
    }

    fn ledger_attempt(
        &self,
        target: &PinTarget<'_>,
        strategy: Strategy,
        source: &str,
        details: AttemptDetails,
    ) -> LedgerAttempt {
        LedgerAttempt {
            iter: target.iter,
            package: target.package.to_string(),
            version: Some(target.version.to_string()),
            tier: strategy.as_str().to_string(),
            strategy: strategy.as_str().to_string(),
            conflict: target.conflict.kind().to_string(),
            source: source.to_string(),
            ts: timestamp(),
            old_spec: details.old_spec,
            new_spec: details.new_spec,
            ceiling_policy: details.ceiling_policy,
            before: details.before,
            failed: details.failed,
        }
    }

    fn guard_anchor(&self, package: &str) -> std::result::Result<(), String> {
        if is_abi_anchor(package) {
            eprintln!(
                "retread solve: refusing to auto-pin ABI anchor {package}; edit the manifest manually"
            );
            return Err(package.to_string());
        }
        Ok(())
    }

    fn guard_oscillation(
        &mut self,
        package: &str,
        version_or_spec: &str,
        strategy: Strategy,
    ) -> std::result::Result<(), String> {
        let key = (package.to_string(), version_or_spec.to_string(), strategy);
        if !self.current_run_attempts.insert(key) {
            return Err(package.to_string());
        }
        Ok(())
    }
}

pub fn append_attempt(
    ledger: &mut SolveLedger,
    ledger_path: &Path,
    run_idx: usize,
    attempt: LedgerAttempt,
) -> Result<()> {
    if let Some(run) = ledger.runs.get_mut(run_idx) {
        run.attempts.push(attempt);
    }
    ledger.write_atomic(ledger_path)
}

pub fn mark_last_widen_failed(
    ledger: &mut SolveLedger,
    ledger_path: &Path,
    run_idx: usize,
    package: &str,
) -> Result<()> {
    if let Some(run) = ledger.runs.get_mut(run_idx)
        && let Some(attempt) = run
            .attempts
            .iter_mut()
            .rev()
            .find(|a| a.package == package && a.strategy == Strategy::WidenConda.as_str())
    {
        attempt.failed = true;
    }
    ledger.write_atomic(ledger_path)
}

pub fn manifest_sha256(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub fn ledger_path(project_dir: &Path) -> PathBuf {
    project_dir.join(".retread").join("solve-ledger.json")
}

pub fn retread_dir(project_dir: &Path) -> PathBuf {
    project_dir.join(".retread")
}

pub fn snapshot_path(project_dir: &Path) -> PathBuf {
    retread_dir(project_dir).join("pixi.toml.bak")
}

/// Directory each iteration's raw (pre-parse, ANSI-stripped) solver error
/// text is persisted under -- so a triage session after an exhausted run
/// isn't limited to the terse repair-table summary (which survived the
/// 2026-07-08 acceptance-#5 run but wasn't enough to see what the *next*
/// iteration's conflict text actually named).
pub fn conflict_trace_dir(project_dir: &Path) -> PathBuf {
    retread_dir(project_dir).join("solve-conflicts")
}

/// Path for one iteration's captured conflict text: `<run>-<iter>.txt`
/// under [`conflict_trace_dir`]. `run` is the ledger run name (e.g. "lock"
/// or an environment name); `iter` is the 1-based repair iteration.
pub fn conflict_trace_path(project_dir: &Path, run: &str, iter: u32) -> PathBuf {
    let safe_run: String = run
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    conflict_trace_dir(project_dir).join(format!("{safe_run}-{iter}.txt"))
}

/// Writes the flattened solver stderr for one iteration to
/// [`conflict_trace_path`]; best-effort (a write failure here must never
/// abort a solve run, so errors are swallowed after an eprintln).
pub fn persist_conflict_trace(project_dir: &Path, run: &str, iter: u32, text: &str) {
    let path = conflict_trace_path(project_dir, run, iter);
    if let Some(parent) = path.parent()
        && let Err(err) = std::fs::create_dir_all(parent)
    {
        eprintln!("retread: could not create {}: {err}", parent.display());
        return;
    }
    if let Err(err) = std::fs::write(&path, text) {
        eprintln!("retread: could not write conflict trace {}: {err}", path.display());
    }
}

pub fn truncate_ledger_runs(ledger_path: &Path, manifest_display: String) -> Result<()> {
    let ledger = SolveLedger {
        version: 1,
        manifest: manifest_display,
        runs: Vec::new(),
    };
    if let Some(parent) = ledger_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_atomic(ledger_path, &serde_json::to_vec_pretty(&ledger)?)?;
    Ok(())
}

pub fn timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

/// Resolves `pack_name` (a bundle name, e.g. `"isaac-pack-latest"`, as
/// named by a uv-closure conflict's `computing uv closure for bundle`
/// line -- which always matches the workspace's own path-dep name for
/// that pack) to its source directory, by scanning the workspace
/// manifest's path-dependency declarations (default feature first, then
/// every named feature). `None` if nothing declares it.
pub fn resolve_pack_dir(
    ws: &crate::workspace::WorkspaceManifest,
    workspace_dir: &Path,
    pack_name: &str,
) -> Option<PathBuf> {
    let raw = ws
        .path_dependencies
        .get(pack_name)
        .or_else(|| ws.features.values().find_map(|f| f.path_dependencies.get(pack_name)))?;
    let candidate = PathBuf::from(raw);
    Some(if candidate.is_absolute() {
        candidate
    } else {
        workspace_dir.join(candidate)
    })
}

/// Conda-name candidates for a pypi `pkg`, in match order: the name-map
/// family (parselmouth-backed, e.g. `torch` -> `pytorch`/`pytorch-cpu`)
/// plus each candidate's `-gpu`/`-cpu` meta-package variants, then the
/// canonical name itself with the same variants. Mirrors
/// `RepairPlanner::resolve_conda_pin_name_in`'s lookup order but yields
/// plain names for a MAP lookup rather than checking a manifest table.
fn conda_name_family(pkg: &str, name_map: &PypiToCondaMap) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push_with_variants = |name: &str| {
        for candidate in variants_of(name) {
            if !out.contains(&candidate) {
                out.push(candidate);
            }
        }
    };
    let canon = canonical_conda_name(pkg);
    if let Some(candidates) = name_map.get(&canon) {
        for c in candidates {
            push_with_variants(c);
        }
    }
    push_with_variants(pkg);
    out
}

/// A conda name plus its `-gpu`/`-cpu` accelerator meta-package variants
/// (the name itself first). Skips double-suffixing an already-variant name.
fn variants_of(name: &str) -> Vec<String> {
    let mut out = vec![name.to_string()];
    for suffix in ["-gpu", "-cpu"] {
        if !name.ends_with(suffix) {
            out.push(format!("{name}{suffix}"));
        }
    }
    out
}

/// The exact `X` from a single `==X` spec in `specs`, or `None` if there
/// isn't exactly one exact-equality pin (a range like `>=1,<2`, or two
/// conflicting specs, can't be relaxed onto a single override version).
fn exact_pin_version(specs: &[String]) -> Option<String> {
    let exacts: Vec<&str> = specs
        .iter()
        .filter_map(|s| s.trim().strip_prefix("=="))
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    match exacts.as_slice() {
        [only] => Some((*only).to_string()),
        _ => None,
    }
}

pub fn is_abi_anchor(name: &str) -> bool {
    ABI_ANCHOR_NAMES.contains(&name)
        || name.starts_with("__")
        || name.ends_with("_compiler")
        || [
            "gcc_",
            "gxx_",
            "g++_",
            "gfortran_",
            "clang_",
            "clangxx_",
            "binutils_",
            "ld_",
            "sysroot_",
        ]
        .iter()
        .any(|p| name.starts_with(p))
}

fn widen_spec(op: &str, floor: &str, policy: WidenCeilingPolicy) -> String {
    let lower = format!("{op}{floor}");
    match policy {
        WidenCeilingPolicy::None => lower,
        WidenCeilingPolicy::NextMajor => match next_major(floor) {
            Some(ceil) => format!("{lower},<{ceil}"),
            None => lower,
        },
        WidenCeilingPolicy::SameMinor => match next_minor(floor) {
            Some(ceil) => format!("{lower},<{ceil}"),
            None => lower,
        },
    }
}

/// Strips a leading comparison operator (`==`, `>=`, `>`, ...) off a
/// `Conflict::CondaWidenNeeded::conda_version` string (e.g. `"==2.10.0"` ->
/// `"2.10.0"`), so it can be re-emitted verbatim as a pypi override pin.
fn strip_version_op(spec: &str) -> &str {
    spec.trim_start_matches(|c: char| !c.is_ascii_digit())
}

fn next_major(version: &str) -> Option<String> {
    let major = version.split('.').next()?.parse::<u64>().ok()?;
    Some((major + 1).to_string())
}

fn next_minor(version: &str) -> Option<String> {
    let mut parts = version.split('.');
    let major = parts.next()?.parse::<u64>().ok()?;
    let minor = parts.next().unwrap_or("0").parse::<u64>().ok()?;
    Some(format!("{major}.{}", minor + 1))
}

fn parse_strategy(raw: &str) -> Strategy {
    match raw {
        "widen-conda" => Strategy::WidenConda,
        "pypi_dep" => Strategy::PypiDep,
        "pypi_override" => Strategy::PypiOverride,
        _ => Strategy::Conda,
    }
}

fn describe_attempt(attempt: &LedgerAttempt) -> String {
    format!(
        "recorded {} {} from {}",
        attempt.package, attempt.strategy, attempt.source
    )
}

fn table_label(feature: &str, kind: TableKind) -> String {
    if feature == "default" {
        match kind {
            TableKind::Conda => "dependencies".to_string(),
            TableKind::Pypi => "pypi-dependencies".to_string(),
            TableKind::Override => "pypi-options.dependency-overrides".to_string(),
        }
    } else {
        match kind {
            TableKind::Conda => format!("feature.{feature}.dependencies"),
            TableKind::Pypi => format!("feature.{feature}.pypi-dependencies"),
            TableKind::Override => format!("feature.{feature}.pypi-options.dependency-overrides"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use super::super::parse::{ConflictParser, RegexConflictParser};

    fn temp_manifest(text: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "retread-solve-repair-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pixi.toml");
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn ladder_widens_then_pins_then_migrates_then_overrides() {
        let path = temp_manifest("[dependencies]\n");
        let mut editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let widen = Conflict::CondaWidenNeeded {
            package: "mujoco".into(),
            op: ">=".into(),
            floor: "3.10.3".into(),
            conda_version: "3.5.0".into(),
            requiring_chain: Vec::new(),
            pack_name: None,
        };
        let out = planner.repair(&mut editor, &mut tried, &widen, 1).unwrap();
        assert_eq!(out.attempt.strategy, "widen-conda");
        assert_eq!(out.attempt.new_spec.as_deref(), Some(">=3.10.3,<4"));

        let boundary = Conflict::CondaBoundary {
            package: "numpy".into(),
            version: "2.3.1".into(),
        };
        assert_eq!(
            planner
                .repair(&mut editor, &mut tried, &boundary, 2)
                .unwrap()
                .attempt
                .strategy,
            "conda"
        );
        let no_candidates = Conflict::NoCandidates {
            package: "numpy".into(),
            version: "2.3.1".into(),
        };
        assert_eq!(
            planner
                .repair(&mut editor, &mut tried, &no_candidates, 3)
                .unwrap()
                .attempt
                .strategy,
            "pypi_dep"
        );
        assert_eq!(
            planner
                .repair(&mut editor, &mut tried, &no_candidates, 4)
                .unwrap()
                .attempt
                .strategy,
            "pypi_override"
        );
    }

    #[test]
    fn user_owned_conda_pin_skips_tier_one() {
        let path = temp_manifest("[dependencies]\nnumpy = \"==1.26.4\"\n");
        let mut editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());
        let conflict = Conflict::CondaBoundary {
            package: "numpy".into(),
            version: "2.3.1".into(),
        };
        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .unwrap();
        assert_eq!(out.attempt.strategy, "pypi_dep");
        assert!(tried.has("numpy", Strategy::Conda));
    }

    #[test]
    fn ledger_round_trips_and_seeds_from_hash_or_sentinels() {
        let path = temp_manifest(
            "[dependencies]\nnumpy = \"==2.3.1\"  # retread:pin 2026-07-07 conda-boundary\n",
        );
        let editor = ManifestEditor::open(path.clone()).unwrap();
        let manifest_hash = manifest_sha256(&path).unwrap();
        let ledger_path = path.parent().unwrap().join(".retread/solve-ledger.json");
        let mut ledger = SolveLedger::load(&ledger_path, "pixi.toml".into()).unwrap();
        let run = ledger.start_run(
            "default".into(),
            manifest_hash.clone(),
            Some("pixi 0.70.0".into()),
        );
        ledger.runs[run].attempts.push(LedgerAttempt {
            iter: 1,
            package: "torch".into(),
            version: Some("2.7.1".into()),
            tier: "conda".into(),
            strategy: "conda".into(),
            conflict: "CondaBoundary".into(),
            source: "retread".into(),
            ts: timestamp(),
            old_spec: None,
            new_spec: None,
            ceiling_policy: None,
            before: None,
            failed: false,
        });
        ledger.write_atomic(&ledger_path).unwrap();
        let loaded = SolveLedger::load(&ledger_path, "pixi.toml".into()).unwrap();
        let tried = loaded.seed_tried_state(&path, &manifest_hash, &editor);
        assert!(tried.has("torch", Strategy::Conda));

        let tried_from_sentinels = loaded.seed_tried_state(&path, "different", &editor);
        assert!(tried_from_sentinels.has("numpy", Strategy::Conda));
        assert!(!tried_from_sentinels.has("torch", Strategy::Conda));
    }

    #[test]
    fn conda_widen_needed_user_pin_emits_override_and_leaves_conda_pin_untouched() {
        // The live torch fixture: conda pin ==2.10.0 (user-owned, no
        // sentinel), pypi side wants >=2.11.0. Conda-as-truth (T1, default
        // relax-preference) must emit a pypi dependency-override pinned to
        // the conda-provided version, and must NOT touch the conda pin.
        let path = temp_manifest("[dependencies]\ntorch = \"==2.10.0\"\n");
        let mut editor = ManifestEditor::open(path.clone()).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());
        assert_eq!(planner.relax_preference, RelaxPreference::Conda);

        let conflict = Conflict::CondaWidenNeeded {
            package: "torch".into(),
            op: ">=".into(),
            floor: "2.11.0".into(),
            conda_version: "==2.10.0".into(),
            requiring_chain: Vec::new(),
            pack_name: None,
        };
        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .unwrap();
        assert_eq!(out.attempt.strategy, "pypi_override");
        assert_eq!(out.attempt.new_spec.as_deref(), Some("==2.10.0"));
        assert!(!tried.has("torch", Strategy::Conda));
        assert!(!tried.has("torch", Strategy::WidenConda));
        assert!(tried.has("torch", Strategy::PypiOverride));

        editor.write_atomic().unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("torch = \"==2.10.0\"\n")); // conda pin untouched, no sentinel
        assert!(text.contains("[pypi-options.dependency-overrides]"));
        assert!(text.contains("torch = \"==2.10.0\"  # retread:override"));
    }

    #[tokio::test]
    async fn conda_widen_needed_matches_conda_pin_via_name_map_pytorch_gpu() {
        // Live gap this closes: pypi requirement `torch==2.11.0` conflicts
        // with a conda pin declared as `pytorch-gpu ==2.10.0` -- a
        // DIFFERENT name than the pypi package. Without the parselmouth-
        // backed name family, `has_user_entry(.., "torch")` never finds the
        // `pytorch-gpu` pin, so T1 is skipped entirely and the ladder falls
        // through to (wrongly) widening/creating a bogus `torch` conda
        // entry. With the name map wired in, T1 must fire: emit a pypi
        // override for `torch`, and leave `pytorch-gpu` byte-for-byte
        // untouched.
        //
        // Uses the PRODUCTION map-loading path (`load_pypi_to_conda_map`),
        // not a hand-inserted map -- a prior version of this test built its
        // own `torch -> [pytorch, pytorch-cpu, pytorch-gpu]` map by hand,
        // which masked the real data gap: FALLBACK_PYPI_TO_CONDA didn't
        // actually carry the `torch -> pytorch-gpu` entry (parselmouth's
        // `pytorch-gpu` is a meta-package with no pypi names of its own),
        // so production code returned None here even though this test
        // passed (see lock-succ-brief.md ACCEPTANCE RUN #4). The parselmouth
        // network fetch inside `load_pypi_to_conda_map` may fail in this
        // sandbox (no network) -- that's fine, the FALLBACK_PYPI_TO_CONDA
        // merge runs unconditionally regardless of fetch outcome.
        let path = temp_manifest("[dependencies]\npytorch-gpu = \"==2.10.0\"\n");
        let mut editor = ManifestEditor::open(path.clone()).unwrap();
        let mut tried = TriedState::default();
        let name_map = crate::handler::load_pypi_to_conda_map().await;
        assert!(
            name_map
                .get("torch")
                .is_some_and(|c| c.iter().any(|n| n == "pytorch-gpu")),
            "production pypi->conda map must carry torch -> pytorch-gpu \
             (FALLBACK_PYPI_TO_CONDA entry) or this test is vacuous"
        );
        let mut planner = RepairPlanner::new("default".into()).with_conda_name_map(name_map);
        assert_eq!(planner.relax_preference, RelaxPreference::Conda);

        let conflict = Conflict::CondaWidenNeeded {
            package: "torch".into(),
            op: ">=".into(),
            floor: "2.11.0".into(),
            conda_version: "==2.10.0".into(),
            requiring_chain: Vec::new(),
            pack_name: None,
        };
        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .unwrap();
        assert_eq!(out.attempt.strategy, "pypi_override");
        assert_eq!(out.attempt.new_spec.as_deref(), Some("==2.10.0"));
        assert!(!tried.has("torch", Strategy::Conda));
        assert!(!tried.has("torch", Strategy::WidenConda));
        assert!(tried.has("torch", Strategy::PypiOverride));

        editor.write_atomic().unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        // conda pin untouched, no sentinel:
        assert!(text.contains("pytorch-gpu = \"==2.10.0\"\n"));
        assert!(text.contains("[pypi-options.dependency-overrides]"));
        assert!(text.contains("torch = \"==2.10.0\"  # retread:override"));
        // No bogus `torch` conda entry was created alongside `pytorch-gpu`:
        // the only `torch = ` line in the whole manifest is the override.
        assert_eq!(text.matches("torch = ").count(), 1);
    }

    #[test]
    fn conda_widen_needed_sentineled_pin_takes_widen_tier() {
        // A tool-written (sentineled) conda pin is eligible for T2 widen
        // directly -- it isn't a user-owned pin conda-as-truth should
        // protect.
        let path = temp_manifest(
            "[dependencies]\nmujoco = \"==3.5.0\"  # retread:pin 2026-07-07 conda-boundary\n",
        );
        let mut editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let conflict = Conflict::CondaWidenNeeded {
            package: "mujoco".into(),
            op: ">=".into(),
            floor: "3.10.3".into(),
            conda_version: "==3.5.0".into(),
            requiring_chain: Vec::new(),
            pack_name: None,
        };
        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .unwrap();
        assert_eq!(out.attempt.strategy, "widen-conda");
    }

    #[test]
    fn backend_closure_conflict_writes_pack_table_not_workspace() {
        // Fix #20 (defect 1, override20 brief ACCEPTANCE RUN #6b): a
        // `CondaWidenNeeded` conflict carrying `pack_name` came from the
        // PACK's own uv closure inside the build backend -- a workspace
        // `pypi-options.dependency-overrides` write is provably inert for
        // that class (the closure never reads it). The T1 override must
        // target the PACK's `[package.build.config.retread-overrides]`
        // table, and the workspace manifest must stay untouched.
        let manifest_text = "[dependencies]\ntorch = \"==2.10.0\"\n\n\
             [feature.gpu.dependencies]\n\
             isaac-pack-latest = { path = \"./pypi-packs/isaac-pack-latest\" }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaac-pack-latest");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_pixi = pack_dir.join("pixi.toml");
        std::fs::write(&pack_pixi, "[package]\nname = \"isaac-pack-latest\"\nversion = \"6.0.0\"\n")
            .unwrap();

        let mut editor = ManifestEditor::open(path.clone()).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let conflict = Conflict::CondaWidenNeeded {
            package: "torch".into(),
            op: ">=".into(),
            floor: "2.11.0".into(),
            conda_version: "==2.10.0".into(),
            requiring_chain: Vec::new(),
            pack_name: Some("isaac-pack-latest".into()),
        };
        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .unwrap();
        assert_eq!(out.attempt.strategy, "pypi_override");
        assert!(
            out.applied.is_empty(),
            "a backend-closure override must not touch any workspace table"
        );
        let po = out.pack_override.expect("expected a pack-override write");
        assert_eq!(po.bundle, "isaac-pack-latest");
        assert_eq!(po.package, "torch");
        assert_eq!(po.spec, "==2.10.0");
        assert_eq!(po.pack_pixi, pack_pixi);

        // Workspace manifest untouched.
        editor.write_atomic().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), manifest_text);

        // Caller (lock.rs / driver.rs) performs the write -- simulate it
        // and confirm the override lands in the PACK's retread-overrides
        // table, the exact table the build backend consumes.
        crate::pack_overrides::write_override(&po.pack_pixi, &po.package, &po.spec).unwrap();
        let pack_text = std::fs::read_to_string(&pack_pixi).unwrap();
        assert!(pack_text.contains("torch = \"==2.10.0\"  # retread:override"));
        let doc = pack_text.parse::<toml_edit::DocumentMut>().unwrap();
        assert_eq!(
            doc["package"]["build"]["config"]["retread-overrides"]["torch"]
                .as_str()
                .unwrap(),
            "==2.10.0"
        );
    }

    #[tokio::test]
    async fn pack_override_scoped_to_consuming_env_uses_gpu_pin_not_holosoma() {
        // Fix #20 (defect 2, override20 brief ACCEPTANCE RUN #6b): the old
        // workspace pin-owner scan matched `feature.holosoma`'s unsentineled
        // `pytorch-gpu ==2.10.0` -- even though `holosoma` is NOT a feature
        // of the failing env (`isaaclab-gpu-latest`). The pack-override path
        // resolves the version from the CONSUMING env's conda deps
        // (`consuming_env_dependencies`), which (a) excludes holosoma
        // entirely and (b) reads the RAW spec, so `feature.gpu`'s
        // `# retread:pin`-sentineled `pytorch-gpu ==2.10.0` IS the source of
        // truth (conda = truth per owner policy).
        //
        // To make the point unmistakable, holosoma pins a DIFFERENT version
        // (==9.9.9): if scope leaked, the override would be 9.9.9; scoped
        // correctly it must be gpu's 2.10.0.
        let manifest_text = r#"[dependencies]

[feature.gpu.dependencies]
pytorch-gpu = "==2.10.0"  # retread:pin 2026-07-08 conda-boundary

[feature.holosoma.dependencies]
pytorch-gpu = "==9.9.9"

[feature.isaaclab-latest.dependencies]
isaac-pack-latest = { path = "./pypi-packs/isaac-pack-latest" }

[environments]
isaaclab-gpu-latest = { features = ["gpu", "isaaclab-latest"] }
holosoma-gpu = { features = ["holosoma"] }
"#;
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaac-pack-latest");
        std::fs::create_dir_all(&pack_dir).unwrap();
        std::fs::write(
            pack_dir.join("pixi.toml"),
            "[package]\nname = \"isaac-pack-latest\"\nversion = \"6.0.0\"\n",
        )
        .unwrap();
        let mut editor = ManifestEditor::open(path.clone()).unwrap();
        let mut tried = TriedState::default();
        // Production name map so torch -> pytorch-gpu resolves (the
        // FALLBACK_PYPI_TO_CONDA merge runs even without network).
        let name_map = crate::handler::load_pypi_to_conda_map().await;
        let mut planner = RepairPlanner::new("default".into()).with_conda_name_map(name_map);

        // The conflict names the cuda-bindings footer; the requiring chain
        // reaches torch (which maps to the consuming env's pytorch-gpu pin).
        let conflict = Conflict::CondaWidenNeeded {
            package: "cuda-bindings".into(),
            op: ">=".into(),
            floor: "13.0.3".into(),
            conda_version: ">=12".into(),
            requiring_chain: vec!["torch".into(), "isaacsim-core".into()],
            pack_name: Some("isaac-pack-latest".into()),
        };
        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .unwrap();
        assert_eq!(out.attempt.strategy, "pypi_override");
        let po = out.pack_override.expect("expected a pack-override write");
        assert_eq!(po.package, "torch");
        assert_eq!(
            po.spec, "==2.10.0",
            "must use feature.gpu's consuming-env pin, NOT holosoma's ==9.9.9"
        );
        // Workspace manifest untouched (no edit in any feature).
        assert!(out.applied.is_empty());
        editor.write_atomic().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), manifest_text);
    }

    #[test]
    fn relax_preference_pypi_restores_widen_first_even_for_user_pin() {
        let path = temp_manifest("[dependencies]\ntorch = \"==2.10.0\"\n");
        let mut editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner =
            RepairPlanner::new("default".into()).with_relax_preference(RelaxPreference::Pypi);

        let conflict = Conflict::CondaWidenNeeded {
            package: "torch".into(),
            op: ">=".into(),
            floor: "2.11.0".into(),
            conda_version: "==2.10.0".into(),
            requiring_chain: Vec::new(),
            pack_name: None,
        };
        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .unwrap();
        assert_eq!(out.attempt.strategy, "widen-conda");
    }

    #[test]
    fn abi_guard_covers_exact_list_and_patterns() {
        assert!(is_abi_anchor("python"));
        assert!(is_abi_anchor("__linux"));
        assert!(is_abi_anchor("cxx_compiler"));
        assert!(is_abi_anchor("sysroot_linux-64"));
        assert!(!is_abi_anchor("numpy"));
    }

    // ---- P0 acceptance fixture (lock-succ-brief.md acceptance run #1/#2):
    // real uv-closure text, cuda-bindings footer (no user conda pin), torch
    // named in the "Because torch==2.11.0 depends on cuda-bindings..."
    // requiring clause, pytorch-gpu carrying the actual user conda pin in a
    // NON-default feature (mirrors the root workspace's `feature.gpu`).

    const UV_CLOSURE_CUDA_BINDINGS: &str =
        include_str!("../../tests/fixtures/solve_errors/uv_closure_cuda_bindings_widen.txt");

    #[tokio::test]
    async fn conda_widen_needed_reattributes_footer_to_requiring_package_with_conda_pin() {
        // cuda-bindings (the footer/symptom package) has no conda pin
        // anywhere; pytorch-gpu (torch's conda-side name) is pinned under
        // `feature.gpu`, not `default`. Before the attribution fix, T1
        // never fired (resolve_conda_pin_name("cuda-bindings") == None)
        // and the ladder chased an unwidenable cuda-bindings spec against
        // the ABI-anchored cuda-version pin. After the fix: repair walks
        // the parsed requiring_chain (torch -> isaacsim-core ->
        // isaacsim[all]), finds torch's pytorch-gpu pin under
        // `feature.gpu`, and emits the pypi override THERE, for `torch`,
        // pinned to the conda-provided version -- leaving pytorch-gpu and
        // cuda-bindings untouched.
        //
        // Uses the PRODUCTION map-loading path (see sibling test above for
        // why a hand-inserted map masks real data gaps -- this is the exact
        // scenario ACCEPTANCE RUN #4 hit against the live backend).
        let manifest_text = concat!(
            "[dependencies]\n\n",
            "[feature.gpu.dependencies]\n",
            "pytorch-gpu = \"==2.10.0\"  # retread:pin\n\n",
            "[feature.isaaclab-latest.dependencies]\n",
            "isaac-pack-latest = { path = \"./pypi-packs/isaac-pack-latest\" }\n\n",
            "[environments]\n",
            "isaaclab-gpu-latest = { features = [\"gpu\", \"isaaclab-latest\"] }\n",
        );
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaac-pack-latest");
        std::fs::create_dir_all(&pack_dir).unwrap();
        std::fs::write(pack_dir.join("pixi.toml"), "[package]\nname = \"p\"\nversion = \"6.0\"\n")
            .unwrap();
        let mut editor = ManifestEditor::open(path.clone()).unwrap();
        let mut tried = TriedState::default();
        let name_map = crate::handler::load_pypi_to_conda_map().await;
        assert!(
            name_map
                .get("torch")
                .is_some_and(|c| c.iter().any(|n| n == "pytorch-gpu")),
            "production pypi->conda map must carry torch -> pytorch-gpu \
             (FALLBACK_PYPI_TO_CONDA entry) or this test is vacuous"
        );
        // Planner constructed for the `default` feature, exactly like
        // `retread lock`'s LOCK_FEATURE.
        let mut planner = RepairPlanner::new("default".into()).with_conda_name_map(name_map);

        let parser = RegexConflictParser::new();
        let conflict = parser
            .parse(UV_CLOSURE_CUDA_BINDINGS)
            .expect("real acceptance fixture must parse");
        assert_eq!(conflict.kind(), "CondaWidenNeeded");
        if let Conflict::CondaWidenNeeded { package, .. } = &conflict {
            assert_eq!(package, "cuda-bindings");
        }

        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .unwrap();
        assert_eq!(out.attempt.strategy, "pypi_override");
        assert_eq!(out.attempt.package, "torch");
        assert_eq!(out.attempt.new_spec.as_deref(), Some("==2.10.0"));

        // Fix #20: this fixture is a BACKEND uv-closure conflict
        // (`pack_name: Some("isaac-pack-latest")`), so the T1 override
        // targets the PACK's retread-overrides table (defect 1), and the
        // version is resolved from the CONSUMING env's `feature.gpu` pin --
        // sentinel-agnostic, env-scoped (defect 2). Attribution (fix
        // #16/#18: footer cuda-bindings -> requiring torch -> pytorch-gpu)
        // is unchanged; only the write TARGET/scoping moved.
        let po = out.pack_override.expect("expected a pack-override write");
        assert_eq!(po.bundle, "isaac-pack-latest");
        assert_eq!(po.package, "torch");
        assert_eq!(po.spec, "==2.10.0");
        assert!(out.applied.is_empty());

        editor.write_atomic().unwrap();
        // The workspace manifest is byte-for-byte untouched.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), manifest_text);
    }

    // ---- Fix #19 acceptance fixture (lock-succ-brief.md ACCEPTANCE RUN
    // #5): iter 2 of the same run. Iter 1 already emitted a `torch`
    // pypi_override (tried-state carries it); the iter-2 uv-closure report
    // re-conflicts on cuda-bindings, and the requiring chain now names
    // BOTH torch (already tried -- must be skipped, not fallen back from)
    // and torchvision (untried, with its own exact-name conda pin). Before
    // the chain-walk fix, the single-path walk only ever surfaced torch,
    // so with torch tried the walk found nothing and fell through to the
    // unwidenable cuda-bindings ladder (EXIT=3, exhausted). After the fix,
    // the walk continues past torch to torchvision and fires T1 there.
    const UV_CLOSURE_CUDA_BINDINGS_ITER2_TORCHVISION: &str = include_str!(
        "../../tests/fixtures/solve_errors/uv_closure_cuda_bindings_widen_iter2_torchvision.txt"
    );

    #[tokio::test]
    async fn conda_widen_needed_continues_chain_walk_past_tried_package_to_sibling_pin_owner() {
        let manifest_text = concat!(
            "[dependencies]\n\n",
            "[feature.gpu.dependencies]\n",
            "pytorch-gpu = \"==2.10.0\"  # retread:pin\n",
            "torchvision = \"==0.25.0\"  # retread:pin\n\n",
            "[feature.isaaclab-latest.dependencies]\n",
            "isaac-pack-latest = { path = \"./pypi-packs/isaac-pack-latest\" }\n\n",
            "[environments]\n",
            "isaaclab-gpu-latest = { features = [\"gpu\", \"isaaclab-latest\"] }\n",
        );
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaac-pack-latest");
        std::fs::create_dir_all(&pack_dir).unwrap();
        std::fs::write(pack_dir.join("pixi.toml"), "[package]\nname = \"p\"\nversion = \"6.0\"\n")
            .unwrap();
        let mut editor = ManifestEditor::open(path.clone()).unwrap();
        let mut tried = TriedState::default();
        // Iter 1 already tried (and applied) a `torch` pypi_override --
        // seed the exact tried-state the second repair() call would see.
        tried.mark("torch", Strategy::PypiOverride, false);

        let name_map = crate::handler::load_pypi_to_conda_map().await;
        let mut planner = RepairPlanner::new("default".into()).with_conda_name_map(name_map);

        let parser = RegexConflictParser::new();
        let conflict = parser
            .parse(UV_CLOSURE_CUDA_BINDINGS_ITER2_TORCHVISION)
            .expect("iter-2 acceptance fixture must parse");
        if let Conflict::CondaWidenNeeded {
            requiring_chain, ..
        } = &conflict
        {
            assert_eq!(
                requiring_chain,
                &vec![
                    "torch".to_string(),
                    "torchvision".to_string(),
                    "isaacsim-core".to_string(),
                    "isaacsim[all]".to_string(),
                ]
            );
        } else {
            panic!("expected CondaWidenNeeded, got {conflict:?}");
        }

        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 2)
            .unwrap();
        assert_eq!(out.attempt.strategy, "pypi_override");
        assert_eq!(out.attempt.package, "torchvision");
        assert_eq!(out.attempt.new_spec.as_deref(), Some("==0.25.0"));

        // Fix #20: backend-closure conflict -> the chain-walked torchvision
        // override targets the PACK's retread-overrides table (defect 1),
        // versioned from the consuming env's `feature.gpu` torchvision pin
        // (defect 2). The fix #19 chain walk (skip already-tried torch,
        // continue to the sibling torchvision) is what's under test and is
        // unchanged: resolve_pack_override's candidate loop skips torch
        // (tried) and stops at torchvision.
        let po = out.pack_override.expect("expected a pack-override write");
        assert_eq!(po.bundle, "isaac-pack-latest");
        assert_eq!(po.package, "torchvision");
        assert_eq!(po.spec, "==0.25.0");
        assert!(out.applied.is_empty());

        editor.write_atomic().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), manifest_text);
        // torch's tried-state stayed marked -- no second torch attempt.
        assert!(tried.has("torch", Strategy::PypiOverride));
    }
}
