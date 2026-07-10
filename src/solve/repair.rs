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
            // Only seed strategies that actually FAILED. A ledger entry with
            // `failed: false` means the repair worked in-memory when it was
            // applied; if the run later died on an unrelated conflict, that
            // fix was never persisted to `.retread/auto-overrides.json`, so
            // there is nothing to oscillate against -- the correct behavior
            // on the next invocation is to retry it, not to treat it as
            // already-exhausted. Seeding successes here (regardless of
            // whether they ended up persisted) previously caused
            // `deps_from_pin_conflict` et al. to refuse a retry with
            // "exhausted repair strategies" even though the repair had
            // never actually failed. `TriedState::has` doesn't distinguish
            // tried-and-failed from tried-and-succeeded, so the only way to
            // preserve oscillation protection for genuine failures while
            // unblocking good-but-unpersisted repairs is to never mark the
            // latter into `tried` at all.
            for run in &self.runs {
                for a in &run.attempts {
                    if a.failed {
                        state.mark(&a.package, parse_strategy(&a.strategy), true);
                    }
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
    /// Fix #20 (relocated to the ledger by fix #22): set instead of (never
    /// alongside) `applied` when this repair's target came from a backend
    /// uv-closure conflict (`Conflict::CondaWidenNeeded::pack_name` is
    /// `Some`). The WORKSPACE manifest is left untouched; the caller
    /// (`retread solve`'s `driver.rs` / `retread lock`'s `lock.rs`) is
    /// responsible for actually recording this in the workspace's
    /// `.retread/auto-overrides.json` ledger (via
    /// `crate::pack_overrides::write_override`) -- kept out of `repair()`
    /// itself so a `--dry-run` repair (which only ever mutates a throwaway
    /// `ManifestEditor`) never side-effects the ledger.
    pub pack_override: Option<PackOverrideWrite>,
}

/// Which ledger sink a [`PackOverrideWrite`] targets. Generic fallback
/// engine addition: doctrine-(v) UN-ROUTE writes to
/// `AutoOverrideLedger::unrouted` (`pack_overrides::write_unroute`)
/// instead of the ordinary spec-override sink (`pack_overrides::
/// write_override`) every other tier uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackOverrideKind {
    /// Ordinary `retread-overrides`-shaped spec write (existing behavior,
    /// every tier before the generic fallback engine).
    Override,
    /// Force this package to stay a PyPI wheel (`retread-keep-pypi`),
    /// abandoning its conda auto-route -- `spec` is unused for this kind.
    Unroute,
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
    /// Conda-style spec written verbatim (e.g. `"==2.10.0"`). Unused when
    /// `kind` is [`PackOverrideKind::Unroute`].
    pub spec: String,
    /// Which ledger sink this write targets. Defaults to `Override` at
    /// every pre-existing call site (added by the generic fallback
    /// engine; see [`PackOverrideKind`]).
    pub kind: PackOverrideKind,
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
                if let Some(found) =
                    self.resolve_conda_pin_name_or_variant(editor, feature, candidate)
                {
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
            Conflict::DepsFromPin {
                package,
                pinned_version,
                op,
                floor,
                pack_name,
                ..
            } => {
                let target = PinTarget {
                    package,
                    version: floor,
                    iter,
                    conflict,
                };
                self.deps_from_pin_conflict(
                    editor,
                    tried,
                    &target,
                    pinned_version,
                    op,
                    pack_name.as_deref(),
                )
            }
            Conflict::NoWheelTransitive {
                requirer,
                requirer_pin,
                pack_name,
                ..
            } => {
                let target = PinTarget {
                    package: requirer,
                    version: requirer_pin,
                    iter,
                    conflict,
                };
                self.no_wheel_transitive_conflict(editor, tried, &target, pack_name.as_deref())
            }
            Conflict::CondaRangeVsPackPin {
                package,
                conda_range,
                pack_demand,
                pack_name,
            } => {
                let target = PinTarget {
                    package,
                    version: pack_demand,
                    iter,
                    conflict,
                };
                self.conda_range_vs_pack_pin(editor, tried, &target, conda_range, pack_name)
            }
            Conflict::NestedCondaCap {
                package,
                pack_name,
                pack_demand,
                cap_op,
                cap_version,
                ..
            } => {
                let target = PinTarget {
                    package,
                    version: pack_demand,
                    iter,
                    conflict,
                };
                self.nested_conda_cap(editor, tried, &target, pack_name, cap_op, cap_version)
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

    /// Fix #20 (defect 1; sink relocated to the ledger by fix #22): emit a
    /// [`PackOverrideWrite`] for a backend-closure conflict -- the override
    /// is recorded in the workspace's `.retread/auto-overrides.json`
    /// ledger and merged in-memory into the failing PACK's effective
    /// `retread-overrides` at `Handler::initialize` time (the only
    /// override table the pack's uv closure consumes; a workspace
    /// `pypi-options.dependency-overrides` write is inert here, see
    /// override20 brief ACCEPTANCE RUN #6b). No manifest, and no pack
    /// pixi.toml, is touched by `repair()` itself; the caller performs the
    /// ledger write, so `--dry-run` stays side-effect-free. `pkg`/`version`
    /// come from [`Self::resolve_pack_override`] (consuming-env,
    /// sentinel-agnostic).
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
                "would add [{bundle} :: retread-overrides] {pkg} = \"{new_spec}\"  (tier: pypi_override; conda-as-truth; .retread/auto-overrides.json ledger)",
            ),
            applied: Vec::new(),
            pack_override: Some(PackOverrideWrite {
                bundle: bundle.to_string(),
                pack_pixi,
                package: pkg.to_string(),
                spec: new_spec,
                kind: PackOverrideKind::Override,
            }),
        })
    }

    /// `DepsFromPin` dispatch: relax `package`'s exact pin to a
    /// `{op}{floor}` floor, but ONLY when `deps_from_owns_exact_pin`
    /// confirms the pin actually originates from `bundle`'s own
    /// `retread-deps-from` root(s) -- doctrine limits auto-relaxing exact
    /// pins with no conda involvement at all to that one provenance;
    /// anything else (a hand-authored `[retread-wheels]` exact pin, for
    /// instance) is left alone and this tier reports exhausted instead of
    /// guessing. Mirrors `try_pack_override`'s ledger sink
    /// (`.retread/auto-overrides.json`, never the pack's own `pixi.toml`)
    /// but writes a `>=`/`>` floor spec instead of a conda-derived `==`.
    fn deps_from_pin_conflict(
        &mut self,
        editor: &mut ManifestEditor,
        tried: &mut TriedState,
        target: &PinTarget<'_>,
        pinned_version: &str,
        op: &str,
        pack_name: Option<&str>,
    ) -> std::result::Result<RepairOutcome, String> {
        let package = target.package;
        let floor = target.version;
        self.guard_anchor(package)?;
        let Some(bundle) = pack_name else {
            return Err(package.to_string());
        };
        let new_spec = format!("{op}{floor}");
        self.guard_oscillation(package, &new_spec, Strategy::PypiOverride)?;
        if tried.has(package, Strategy::PypiOverride) {
            return Err(package.to_string());
        }
        if !deps_from_owns_exact_pin(editor.project_dir(), bundle, package) {
            eprintln!(
                "retread: {package}=={pinned_version} conflict is not attributable to a \
                 retread-deps-from root of bundle `{bundle}`; refusing to auto-relax"
            );
            return Err(package.to_string());
        }
        let pack_pixi = crate::workspace::WorkspaceManifest::load(editor.project_dir())
            .and_then(|ws| resolve_pack_dir(&ws, editor.project_dir(), bundle))
            .map(|dir| dir.join("pixi.toml"))
            .ok_or_else(|| package.to_string())?;
        tried.mark(package, Strategy::PypiOverride, false);
        eprintln!(
            "retread: deps-from pin relaxed {package} =={pinned_version} -> {new_spec} (upstream-advisory)"
        );
        Ok(RepairOutcome {
            attempt: self.ledger_attempt(
                target,
                Strategy::PypiOverride,
                "retread",
                AttemptDetails {
                    old_spec: Some(format!("=={pinned_version}")),
                    new_spec: Some(new_spec.clone()),
                    ..AttemptDetails::default()
                },
            ),
            extra_attempts: Vec::new(),
            summary_line: format!(
                "would add [{bundle} :: retread-overrides] {package} = \"{new_spec}\"  (tier: pypi_override; deps-from-advisory; .retread/auto-overrides.json ledger)",
            ),
            applied: Vec::new(),
            pack_override: Some(PackOverrideWrite {
                bundle: bundle.to_string(),
                pack_pixi,
                package: package.to_string(),
                spec: new_spec,
                kind: PackOverrideKind::Override,
            }),
        })
    }

    /// `NoWheelTransitive` dispatch: `package` (e.g.
    /// `antlr4-python3-runtime`) is wheel-less across the whole range
    /// `requirer`'s own exact pin wildcards it into, and the backend's
    /// sdist-only self-heal ladder (conda-route, then sdist auto-build --
    /// see `uv_closure.rs`) already exhausted both rungs before this
    /// conflict ever reached the parser. The only route left is relaxing
    /// `requirer`'s (e.g. `hydra-core`) own exact pin to a `>=` floor, so
    /// uv is free to pick a newer `requirer` release whose metadata may no
    /// longer wildcard-pin the wheel-less package -- same
    /// `deps_from_owns_exact_pin` ownership doctrine as `DepsFromPin`
    /// (never guess at a non-deps-from-owned pin), just checked against
    /// `requirer` (`target.package`, already renamed by the `repair()`
    /// dispatch above) instead of the wheel-less package itself.
    fn no_wheel_transitive_conflict(
        &mut self,
        editor: &mut ManifestEditor,
        tried: &mut TriedState,
        target: &PinTarget<'_>,
        pack_name: Option<&str>,
    ) -> std::result::Result<RepairOutcome, String> {
        let requirer = target.package;
        let requirer_pin = target.version;
        self.guard_anchor(requirer)?;
        let Some(bundle) = pack_name else {
            return Err(requirer.to_string());
        };
        let new_spec = format!(">={requirer_pin}");
        self.guard_oscillation(requirer, &new_spec, Strategy::PypiOverride)?;
        if tried.has(requirer, Strategy::PypiOverride) {
            return Err(requirer.to_string());
        }
        if !deps_from_owns_exact_pin(editor.project_dir(), bundle, requirer) {
            eprintln!(
                "retread: {requirer}=={requirer_pin} conflict is not attributable to a \
                 retread-deps-from root of bundle `{bundle}`; refusing to auto-relax"
            );
            return Err(requirer.to_string());
        }
        let pack_pixi = crate::workspace::WorkspaceManifest::load(editor.project_dir())
            .and_then(|ws| resolve_pack_dir(&ws, editor.project_dir(), bundle))
            .map(|dir| dir.join("pixi.toml"))
            .ok_or_else(|| requirer.to_string())?;
        tried.mark(requirer, Strategy::PypiOverride, false);
        eprintln!(
            "retread: deps-from pin relaxed {requirer} =={requirer_pin} -> {new_spec} \
             (no-wheel-transitive; upstream-advisory)"
        );
        Ok(RepairOutcome {
            attempt: self.ledger_attempt(
                target,
                Strategy::PypiOverride,
                "retread",
                AttemptDetails {
                    old_spec: Some(format!("=={requirer_pin}")),
                    new_spec: Some(new_spec.clone()),
                    ..AttemptDetails::default()
                },
            ),
            extra_attempts: Vec::new(),
            summary_line: format!(
                "would add [{bundle} :: retread-overrides] {requirer} = \"{new_spec}\"  (tier: pypi_override; deps-from-advisory; no-wheel-transitive; .retread/auto-overrides.json ledger)",
            ),
            applied: Vec::new(),
            pack_override: Some(PackOverrideWrite {
                bundle: bundle.to_string(),
                pack_pixi,
                package: requirer.to_string(),
                spec: new_spec,
                kind: PackOverrideKind::Override,
            }),
        })
    }

    /// `CondaRangeVsPackPin` dispatch (tenth fix, deps-from hardening
    /// series): the workspace's own two-sided conda RANGE pin for
    /// `package` is real owner intent (conda-as-truth) and must never be
    /// widened; `pack_name`'s own exact companion demand is AUTO-ROUTED
    /// (derived from whatever uv's own closure happened to lock inside
    /// the pack -- `handler/mod.rs`'s `bundle.auto_routed` emission is the
    /// only mechanism that produces an exact companion conda run-dep for a
    /// pack this backend composed, so it is never a hand-authored
    /// constraint) and is not a real constraint at all. Repair injects the
    /// workspace's own range, translated to PEP440, as a
    /// `retread-overrides` entry in the PACK's own ledger
    /// (`.retread/auto-overrides.json`, merged into its uv closure at
    /// `Handler::initialize` time -- same sink as `try_pack_override` /
    /// `deps_from_pin_conflict`) so uv re-locks `package` inside the
    /// workspace's range; the pack's auto-routed companion pin then
    /// follows to a version inside that range on its next render.
    ///
    /// Guardrail: only fires when `pack_name` resolves to one of THIS
    /// workspace's own composed packs (`resolve_pack_dir` succeeds) --
    /// the only packs whose exact companion conda pins are provably
    /// auto-routed by this backend's own render pipeline; an unresolvable
    /// `pack_name` (a foreign/third-party conda package) is refused rather
    /// than guessed at. ABI anchors are exempt (never touched, matching
    /// every other override tier). This tier never widens any conda pin,
    /// so it runs unconditionally ahead of `conda_widen_needed`'s
    /// widen-conda rung for the (disjoint) conflict shape it handles.
    fn conda_range_vs_pack_pin(
        &mut self,
        editor: &ManifestEditor,
        tried: &mut TriedState,
        target: &PinTarget<'_>,
        conda_range: &str,
        pack_name: &str,
    ) -> std::result::Result<RepairOutcome, String> {
        let package = target.package;
        self.guard_anchor(package)?;
        let pack_pixi = crate::workspace::WorkspaceManifest::load(editor.project_dir())
            .and_then(|ws| resolve_pack_dir(&ws, editor.project_dir(), pack_name))
            .map(|dir| dir.join("pixi.toml"))
            .ok_or_else(|| package.to_string())?;
        let new_spec = translate_conda_range_to_pep440(conda_range);
        // Tried/oscillation state is keyed by (package, PACK) -- run 11
        // showed the same package (`setuptools`) auto-routed at an exact
        // pin by SEVERAL sibling packs (isaaclab-2.3x-pack,
        // unitree-rl-gym/robogen owners); a package-only key would let the
        // first pack's repair permanently block the identical (and equally
        // legitimate) repair for the next pack the follow-up conflict
        // names. One shot per distinct (package, pack) pair; a REPEAT of
        // an already-repaired pair still exhausts (the repair provably
        // didn't stick).
        let tried_key = format!("{package}@{pack_name}");
        self.guard_oscillation(&tried_key, &new_spec, Strategy::PypiOverride)?;
        if tried.has(&tried_key, Strategy::PypiOverride) {
            return Err(package.to_string());
        }
        tried.mark(&tried_key, Strategy::PypiOverride, false);
        eprintln!(
            "retread: workspace conda range {conda_range} for {package} injected into pack \
             `{pack_name}` as pypi override {new_spec} (conda-as-truth; auto-routed pack pin not \
             widened)"
        );
        Ok(RepairOutcome {
            attempt: self.ledger_attempt(
                target,
                Strategy::PypiOverride,
                "retread",
                AttemptDetails {
                    old_spec: Some(format!("=={}", target.version)),
                    new_spec: Some(new_spec.clone()),
                    ..AttemptDetails::default()
                },
            ),
            extra_attempts: Vec::new(),
            summary_line: format!(
                "would add [{pack_name} :: retread-overrides] {package} = \"{new_spec}\"  (tier: pypi_override; workspace-range-vs-auto-routed-pin; .retread/auto-overrides.json ledger)",
            ),
            applied: Vec::new(),
            pack_override: Some(PackOverrideWrite {
                bundle: pack_name.to_string(),
                pack_pixi,
                package: package.to_string(),
                spec: new_spec,
                kind: PackOverrideKind::Override,
            }),
        })
    }

    /// `NestedCondaCap` dispatch (eleventh fix, deps-from hardening
    /// series, run 13): a package a prior `CondaRangeVsPackPin` repair
    /// already narrowed for this pack fails again because a DEEPER conda
    /// run-dependency of the same pack (reached transitively, e.g.
    /// `dex-retargeting -> pytorch`) imposes an additional cap this
    /// conflict names. Reads the ledgered override
    /// (`.retread/auto-overrides.json`, via `pack_overrides::
    /// overrides_for_pack`) that the earlier `CondaRangeVsPackPin` repair
    /// wrote for `(package, pack_name)`, intersects it with the newly
    /// discovered cap (`cap_op`/`cap_version`), and re-writes the
    /// narrowed range into the SAME ledger sink -- the pack's own render
    /// then follows the narrower range on its next re-lock, exactly like
    /// the original override did.
    ///
    /// Termination: each successful narrowing strictly shrinks the range
    /// (a repeated intersection can only tighten or leave it unchanged),
    /// so `guard_oscillation` refusing an EXACT repeat of a
    /// previously-attempted (key, spec) pair is sufficient to bound the
    /// loop -- a narrowing that stops making progress surfaces as a
    /// normal repair exhaustion instead of looping forever. An
    /// intersection with no viable version at all (the cap's ceiling at
    /// or below the range's own floor) is refused up front with a clear
    /// message rather than emitted as a nonsensical override.
    ///
    /// Guardrail: refuses (rather than guesses a range from scratch) when
    /// no `CondaRangeVsPackPin` override already exists for this exact
    /// `(package, pack_name)` pair -- this tier only ever narrows an
    /// existing injected range, it never originates one.
    fn nested_conda_cap(
        &mut self,
        editor: &ManifestEditor,
        tried: &mut TriedState,
        target: &PinTarget<'_>,
        pack_name: &str,
        cap_op: &str,
        cap_version: &str,
    ) -> std::result::Result<RepairOutcome, String> {
        let package = target.package;
        self.guard_anchor(package)?;
        let workspace_dir = editor.project_dir();
        let pack_pixi = crate::workspace::WorkspaceManifest::load(workspace_dir)
            .and_then(|ws| resolve_pack_dir(&ws, workspace_dir, pack_name))
            .map(|dir| dir.join("pixi.toml"))
            .ok_or_else(|| package.to_string())?;
        let existing = crate::pack_overrides::overrides_for_pack(workspace_dir, &pack_pixi)
            .get(package)
            .cloned()
            .ok_or_else(|| {
                eprintln!(
                    "retread: NestedCondaCap fired for {package} in pack `{pack_name}` but no \
                     prior CondaRangeVsPackPin override exists to narrow; refusing to originate \
                     a range from scratch"
                );
                package.to_string()
            })?;
        let narrowed = intersect_range_with_cap(&existing, cap_op, cap_version).map_err(|reason| {
            eprintln!(
                "retread: exhausted -- narrowing {package} in pack `{pack_name}` from {existing} \
                 with nested cap {cap_op}{cap_version} produced an empty range ({reason}); no \
                 version can satisfy both the workspace range and the nested conda cap"
            );
            package.to_string()
        })?;
        // Keyed distinctly from `conda_range_vs_pack_pin`'s own
        // `{package}@{pack_name}` tried-key (that key is marked
        // exhausted after ONE shot, see the comment there) -- each
        // narrowing here produces a strictly different spec, so
        // `guard_oscillation` (keyed on the resulting spec, not just the
        // package/pack pair) is what actually bounds this loop.
        let tried_key = format!("{package}@{pack_name}#narrow");
        self.guard_oscillation(&tried_key, &narrowed, Strategy::PypiOverride)?;
        tried.mark(&tried_key, Strategy::PypiOverride, false);
        eprintln!(
            "retread: narrowed {package} override in pack `{pack_name}`: {existing} -> {narrowed} \
             (nested conda cap {cap_op}{cap_version} via a transitive conda run-dep intersected)"
        );
        Ok(RepairOutcome {
            attempt: self.ledger_attempt(
                target,
                Strategy::PypiOverride,
                "retread",
                AttemptDetails {
                    old_spec: Some(existing),
                    new_spec: Some(narrowed.clone()),
                    ..AttemptDetails::default()
                },
            ),
            extra_attempts: Vec::new(),
            summary_line: format!(
                "would add [{pack_name} :: retread-overrides] {package} = \"{narrowed}\"  (tier: pypi_override; nested-cap-intersection; .retread/auto-overrides.json ledger)",
            ),
            applied: Vec::new(),
            pack_override: Some(PackOverrideWrite {
                bundle: pack_name.to_string(),
                pack_pixi,
                package: package.to_string(),
                spec: narrowed,
                kind: PackOverrideKind::Override,
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
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
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
        eprintln!(
            "retread: could not write conflict trace {}: {err}",
            path.display()
        );
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
    let raw = ws.path_dependencies.get(pack_name).or_else(|| {
        ws.features
            .values()
            .find_map(|f| f.path_dependencies.get(pack_name))
    })?;
    let candidate = PathBuf::from(raw);
    Some(if candidate.is_absolute() {
        candidate
    } else {
        workspace_dir.join(candidate)
    })
}

/// Re-verifies that `package`'s exact pin actually originates from
/// `bundle`'s own `retread-deps-from` root(s) -- the ONLY provenance
/// doctrine allows auto-relaxing an exact pypi pin with no conda
/// involvement at all. The CLI ladder runs outside the JSON-RPC build
/// backend process (where `deps_from_exact_pinned_names` normally runs,
/// see `handler/mod.rs`), so this reads the bundle's `pixi.toml` straight
/// off disk (same file the backend would parse into `RetreadConfig`) and
/// re-fetches its deps-from source(s) via the same
/// `deps_from::resolve_deps_from_roots` fetcher the backend closure uses
/// -- cached under `.retread/deps-from-verify-cache` so a second repair
/// attempt for the same bundle doesn't re-clone/re-fetch. `repair()` is
/// synchronous (many call sites, including plain `#[test]`s, call it
/// directly), so the async fetch is bridged in with
/// `block_in_place`+`block_on` rather than threading `async` through the
/// whole repair ladder -- safe here because `main.rs` runs
/// `#[tokio::main]`'s default multi-thread scheduler.
fn deps_from_owns_exact_pin(project_dir: &Path, bundle: &str, package: &str) -> bool {
    let Some(ws) = crate::workspace::WorkspaceManifest::load(project_dir) else {
        return false;
    };
    let Some(pack_dir) = resolve_pack_dir(&ws, project_dir, bundle) else {
        return false;
    };
    let Ok(text) = std::fs::read_to_string(pack_dir.join("pixi.toml")) else {
        return false;
    };
    let Ok(value) = text.parse::<toml::Value>() else {
        return false;
    };
    let Some(config_value) = value
        .get("package")
        .and_then(|p| p.get("build"))
        .and_then(|b| b.get("config"))
        .cloned()
    else {
        return false;
    };
    // Only `deps_from` is needed here, so deserialize a minimal partial
    // view rather than the full `RetreadConfig` (which requires fields
    // like `retread-wheels` that a deps-from-only re-parse has no reason
    // to demand -- this check must not spuriously fail just because some
    // OTHER config field's shape changed or is absent).
    #[derive(serde::Deserialize)]
    struct DepsFromOnlyConfig {
        #[serde(default, rename = "retread-deps-from", alias = "deps-from")]
        deps_from: crate::config::DepsFromSpec,
    }
    let Ok(cfg) = config_value.try_into::<DepsFromOnlyConfig>() else {
        return false;
    };
    if cfg.deps_from.is_empty() {
        return false;
    }
    let cache_dir = retread_dir(project_dir).join("deps-from-verify-cache");
    let sources = cfg.deps_from.as_slice().to_vec();
    let roots = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(crate::deps_from::resolve_deps_from_roots(
            &sources, &pack_dir, &cache_dir,
        ))
    });
    let Ok(roots) = roots else {
        return false;
    };
    crate::handler::deps_from_exact_pinned_names(&roots).contains(&canonical_conda_name(package))
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

/// Translates a conda-solver two-sided range string (e.g. `">=68,<81"`, as
/// captured from `<pkg> >=X,<Y cannot be installed...` prose) into a
/// PEP440-valid `retread-overrides`/`pypi-options.dependency-overrides`
/// spec. Conda and PEP440 share the same relational operators
/// (`>=`/`>`/`<=`/`<`/`==`), so this is close to a pass-through -- the two
/// normalizations actually needed are conda's bare `=` (never valid in
/// PEP440, which requires `==`) and preserving a trailing conda glob
/// build (e.g. `83.0.0.*`) verbatim, since PEP440 already accepts that
/// unchanged under `==` (arbitrary equality, PEP 440 §Version matching).
/// A clause with no recognized leading operator is passed through
/// unchanged rather than dropped, so a translation gap surfaces as an
/// obviously-wrong override version instead of silently vanishing.
pub(crate) fn translate_conda_range_to_pep440(range: &str) -> String {
    range
        .split(',')
        .map(|clause| translate_conda_clause_to_pep440(clause.trim()))
        .collect::<Vec<_>>()
        .join(",")
}

fn translate_conda_clause_to_pep440(clause: &str) -> String {
    // Longest-prefix-first so `==`/`<=`/`>=` are matched before the bare
    // `=`/`<`/`>` fallbacks (an `=` check first would also match `==`'s
    // leading char and truncate the operator).
    const OPS: &[&str] = &[">=", "<=", "==", ">", "<", "="];
    for op in OPS {
        if let Some(rest) = clause.strip_prefix(op) {
            let normalized_op = if *op == "=" { "==" } else { *op };
            return format!("{normalized_op}{}", rest.trim());
        }
    }
    clause.to_string()
}

/// Eleventh fix: intersects an existing PEP440 range override (e.g.
/// `">=68,<81"`, as written by a prior `CondaRangeVsPackPin` repair) with
/// a newly discovered upper-bound cap (`cap_op`/`cap_version`, e.g.
/// `"<"`/`"76"`, parsed from a `NestedCondaCap` conflict). Keeps the
/// existing floor (if any) untouched -- only the ceiling half narrows --
/// and keeps whichever ceiling (existing vs. cap) is tighter, so a
/// SECOND `NestedCondaCap` naming a looser cap than one already applied
/// is a no-op rather than an accidental widen. Returns `Err` describing
/// the empty-intersection case when the cap's ceiling is at or below the
/// range's own floor: no version can satisfy both.
fn intersect_range_with_cap(
    existing: &str,
    cap_op: &str,
    cap_version: &str,
) -> std::result::Result<String, String> {
    use rattler_conda_types::Version;
    use std::str::FromStr;

    let cap_v = Version::from_str(cap_version)
        .map_err(|e| format!("unparseable cap version {cap_version}: {e}"))?;

    let mut floor: Option<(String, String)> = None;
    let mut ceil: Option<(String, String)> = None;
    for clause in existing.split(',') {
        let clause = clause.trim();
        for op in ["==", ">=", "<=", ">", "<"] {
            if let Some(rest) = clause.strip_prefix(op) {
                let rest = rest.trim().to_string();
                match op {
                    ">=" | ">" => floor = Some((op.to_string(), rest)),
                    "<=" | "<" => ceil = Some((op.to_string(), rest)),
                    _ => {}
                }
                break;
            }
        }
    }

    let narrower_ceil = match &ceil {
        Some((_, existing_ceil_str)) => match Version::from_str(existing_ceil_str) {
            Ok(existing_ceil_v) if existing_ceil_v <= cap_v => ceil.clone().unwrap(),
            _ => (cap_op.to_string(), cap_version.to_string()),
        },
        None => (cap_op.to_string(), cap_version.to_string()),
    };

    if let Some((floor_op, floor_str)) = &floor {
        let floor_v = Version::from_str(floor_str)
            .map_err(|e| format!("unparseable floor version {floor_str}: {e}"))?;
        let ceil_v = Version::from_str(&narrower_ceil.1)
            .map_err(|e| format!("unparseable ceiling version {}: {e}", narrower_ceil.1))?;
        let empty = match narrower_ceil.0.as_str() {
            "<" => floor_v >= ceil_v,
            "<=" => floor_v > ceil_v,
            _ => false,
        };
        if empty {
            return Err(format!(
                "floor {floor_op}{floor_str} vs narrowed ceiling {}{}",
                narrower_ceil.0, narrower_ceil.1
            ));
        }
    }
    let mut out = String::new();
    if let Some((op, v)) = &floor {
        out.push_str(op);
        out.push_str(v);
    }
    if !out.is_empty() {
        out.push(',');
    }
    out.push_str(&narrower_ceil.0);
    out.push_str(&narrower_ceil.1);
    Ok(out)
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

// ---------------------------------------------------------------------------
// Generic ownership-driven fallback repair engine
// ---------------------------------------------------------------------------
//
// Twelve rungs, each a hand-written parser + repair tier for ONE prose
// shape a solver failure can take, is a treadmill: every new shape (run
// 15's PyPI-vs-conda-forge metadata skew was the 13th) demanded another
// bespoke `Conflict` variant, another regex, another tier. This engine
// ends the treadmill: it fires ONLY when none of the specific parsers
// above matched (`RegexConflictParser::parse` returned `None`), scans the
// ENTIRE error text for every `<package> <spec>` mention regardless of
// surrounding prose (`extract_generic_mentions`), classifies each
// mentioned package by WHO owns the knob that could fix it (this
// workspace's own hand-written pin? a `retread-deps-from` root? an
// already-ledgered override? an auto-routed pack pin? an ABI anchor? or
// someone else's untouchable transitive?), and generates a repair
// candidate for each OWNED mention, tried in a fixed doctrine order, one
// per ladder iteration -- reusing every existing tier's mechanics
// (`intersect_range_with_cap`, `deps_from_owns_exact_pin`,
// `translate_conda_range_to_pep440`, the ledger, tried-state,
// oscillation guard) rather than re-implementing them.

/// Who -- if anyone -- owns the knob that could relax a mentioned
/// package's conflicting spec. Doctrine priority mirrors the specific
/// tiers this engine generalizes: a hand-written workspace pin is real
/// owner intent (never touched by relaxing IT; the auto-routed side
/// yields instead); a `deps-from` exact pin is an upstream advisory
/// (safe to relax); an already-ledgered override is a PRIOR repair this
/// engine can narrow further; an auto-routed pack pin is provably not
/// hand-authored (see `CondaRangeVsPackPin`'s doc comment) so it's safe
/// to override OR un-route; an ABI anchor is immutable; anything else is
/// someone else's transitive and untouchable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ownership {
    /// A `retread-deps-from` root's own exact pin (see
    /// `deps_from_owns_exact_pin`) -- upstream advisory, safe to relax.
    DepsFromExactPin,
    /// A pack's own auto-routed exact companion pin (a NAMED pack's
    /// `<pack> <ver> would require <package> ==<v>` clause) -- provably
    /// not hand-authored (`handler/mod.rs`'s `bundle.auto_routed`
    /// emission is the only mechanism that produces one).
    AutoRoutedPackPin,
    /// A PRIOR repair (this engine's or a specific tier's) already wrote
    /// a `.retread/auto-overrides.json` entry for this (package, pack).
    LedgeredOverride,
    /// A hand-written entry in the WORKSPACE's own `pixi.toml` (any
    /// feature, conda or pypi table) -- real owner intent, never
    /// widened; the auto-routed side yields toward it instead (doctrine
    /// (i), mirroring `CondaRangeVsPackPin`).
    WorkspacePin,
    /// One of `ABI_ANCHOR_NAMES` / the `is_abi_anchor` predicate --
    /// immutable guardrail, never touched.
    AbiAnchor,
    /// None of the above: someone else's transitive dependency this
    /// workspace has no ownership stake in. Untouchable.
    Unowned,
}

impl Ownership {
    pub fn as_str(self) -> &'static str {
        match self {
            Ownership::DepsFromExactPin => "deps-from-exact-pin",
            Ownership::AutoRoutedPackPin => "auto-routed-pack-pin",
            Ownership::LedgeredOverride => "ledgered-override",
            Ownership::WorkspacePin => "workspace-pin",
            Ownership::AbiAnchor => "abi-anchor",
            Ownership::Unowned => "unowned",
        }
    }
}

/// One candidate repair the fallback engine could apply, in doctrine
/// priority order (`tier`, ascending): (1) an owned auto-routed pin has a
/// visible workspace range for the SAME package -- override the pack's
/// pin toward that range; (2) a package already has a ledgered override
/// for this pack, and the mentions include a tighter cap -- intersect;
/// (3) a `deps-from`-owned exact pin -- relax to a `>=` floor; (5) two of
/// a pack's OWN emitted pins contradict each other (no workspace pin, no
/// external constrainer -- conda-forge repackage metadata skew) --
/// UN-ROUTE the requirer so it ships as a PyPI wheel instead; (6) an
/// auto-routed pin whose routed conda build requires an ABI anchor
/// version the env can't provide (a `triton ==3.6.0` needing
/// `cuda-version 12.9`+ against a cuda-12.8 env) and has no workspace
/// range to override toward -- UN-ROUTE the pin so its self-contained
/// PyPI wheel (no conda anchor run-dep) ships at the exact resolved
/// version instead. (Tier 4,
/// "override the mentioned package toward whatever the conda closure
/// provides", is the specific tiers' T1 pypi-override and is already
/// covered by tiers 1/3 for every mention this generic scan can attribute
/// an owner to; kept out of the numbering gap intentionally rather than
/// force a redundant candidate.)
#[derive(Debug, Clone)]
struct FallbackCandidate {
    tier: u8,
    package: String,
    pack_name: String,
    kind: PackOverrideKind,
    old_spec: Option<String>,
    new_spec: String,
    ownership: Ownership,
    tried_key: String,
    reason: String,
}

impl RepairPlanner {
    /// Classifies a single mentioned package's ownership within a
    /// `pack_name`-scoped conflict (or the workspace, when `pack_name` is
    /// `None`). See [`Ownership`]'s doc comment for the priority
    /// reasoning; this only ever returns ONE ownership per package (the
    /// highest-priority provenance that actually applies), since a
    /// package can be relaxed via only one real knob at a time.
    fn classify_mention_ownership(
        &self,
        editor: &ManifestEditor,
        pack_name: Option<&str>,
        package: &str,
    ) -> Ownership {
        if is_abi_anchor(package) {
            return Ownership::AbiAnchor;
        }
        let Some(bundle) = pack_name else {
            // No pack context at all: the workspace-manifest-widen path
            // (a hand-written pin colliding with an unscoped pypi
            // requirement) is already covered by the specific
            // `CondaWidenNeeded`/`CondaBoundary` tiers, so the only
            // ownership this generic engine recognizes here is that same
            // hand-written pin (informational -- callers skip it);
            // anything else is someone else's untouchable transitive.
            if self.has_workspace_pin(editor, package) {
                return Ownership::WorkspacePin;
            }
            return Ownership::Unowned;
        };
        // NOTE: deliberately does NOT check for a hand-written workspace
        // pin on `package` here, even though one may ALSO exist -- inside
        // a NAMED pack's conflict, the pack's own auto-routed pin and the
        // workspace's hand-written pin for the SAME name are two
        // different knobs (doctrine (i), `CondaRangeVsPackPin`'s own
        // precedent: the workspace range is real intent and is never
        // touched, but the PACK's companion pin still gets overridden
        // toward it). `generate_fallback_candidates`' tier-1 branch looks
        // up the workspace pin itself via `workspace_conda_pin` when it
        // needs the value; `has_workspace_pin` gates the UN-ROUTE tier
        // separately, for the same reason.
        let Some(pack_pixi) = crate::workspace::WorkspaceManifest::load(editor.project_dir())
            .and_then(|ws| resolve_pack_dir(&ws, editor.project_dir(), bundle))
            .map(|d| d.join("pixi.toml"))
        else {
            return Ownership::Unowned;
        };
        // (c) a prior repair already ledgered an override for this
        // (package, pack) -- checked before deps-from/auto-routed so a
        // SECOND fallback pass narrows the existing entry instead of
        // reclassifying it from scratch.
        if crate::pack_overrides::overrides_for_pack(editor.project_dir(), &pack_pixi)
            .contains_key(package)
        {
            return Ownership::LedgeredOverride;
        }
        // (a) a `retread-deps-from` root's own exact pin.
        if deps_from_owns_exact_pin(editor.project_dir(), bundle, package) {
            return Ownership::DepsFromExactPin;
        }
        // (b) everything else mentioned inside a NAMED pack's own
        // uv-closure conflict is, by the same doctrine `CondaRangeVsPackPin`
        // / `NestedCondaCap` already rely on, part of that pack's own
        // auto-routed/rendered closure (no OTHER mechanism produces a
        // pack-scoped exact companion conda run-dep) -- treat it as an
        // owned, relaxable/un-routable knob rather than an untouchable
        // transitive.
        Ownership::AutoRoutedPackPin
    }

    /// The workspace's own conda-range pin for `package`, if any feature
    /// declares one directly -- used by doctrine tier (i) to translate
    /// toward the SAME spec `CondaRangeVsPackPin` would inject.
    fn workspace_conda_pin(&self, editor: &ManifestEditor, package: &str) -> Option<String> {
        for feature in editor.feature_names() {
            if editor.has_user_entry(&feature, TableKind::Conda, package) {
                let snap = editor.entry_snapshot(&feature, TableKind::Conda, package);
                if let Some(value) = snap.value {
                    return Some(value);
                }
            }
        }
        None
    }

    /// Whether ANY feature declares a hand-written conda or pypi entry
    /// for `package` -- real owner intent per doctrine (d), regardless of
    /// whether it's also mentioned inside a pack-scoped conflict. Used to
    /// gate the UN-ROUTE tier (5): a metadata-skew un-route only makes
    /// sense between two of a PACK's own emitted pins with no external
    /// constrainer, so a package the workspace itself pins is excluded.
    fn has_workspace_pin(&self, editor: &ManifestEditor, package: &str) -> bool {
        editor.feature_names().iter().any(|feature| {
            editor.has_user_entry(feature, TableKind::Conda, package)
                || editor.has_user_entry(feature, TableKind::Pypi, package)
        })
    }

    /// Doctrine-ordered candidate generation over every mention the
    /// generic extractor found. Only mentions inside a NAMED pack's
    /// conflict (`pack_name: Some`) can be attributed a resolvable owner
    /// today -- the workspace-manifest-widen path (`pack_name: None`) is
    /// already covered by the specific `CondaWidenNeeded`/`CondaBoundary`
    /// tiers, so the fallback engine's UNIQUE value-add is entirely on
    /// the pack-scoped side.
    fn generate_fallback_candidates(
        &self,
        editor: &ManifestEditor,
        pack_name: Option<&str>,
        mentions: &[super::parse::Mention],
    ) -> Vec<FallbackCandidate> {
        let mut out = Vec::new();
        let Some(bundle) = pack_name else {
            return out;
        };
        let project_dir = editor.project_dir();
        let Some(pack_pixi) = crate::workspace::WorkspaceManifest::load(project_dir)
            .and_then(|ws| resolve_pack_dir(&ws, project_dir, bundle))
            .map(|d| d.join("pixi.toml"))
        else {
            return out;
        };
        let existing_overrides = crate::pack_overrides::overrides_for_pack(project_dir, &pack_pixi);

        for m in mentions {
            let ownership = self.classify_mention_ownership(editor, pack_name, &m.package);
            match ownership {
                Ownership::AbiAnchor | Ownership::Unowned | Ownership::WorkspacePin => continue,
                Ownership::AutoRoutedPackPin => {
                    // Tier 1: a workspace range for the SAME package is
                    // visible -- override the pack's auto-routed pin
                    // toward it (same mechanics as `conda_range_vs_pack_pin`).
                    if let Some(range) = self.workspace_conda_pin(editor, &m.package) {
                        let new_spec = translate_conda_range_to_pep440(&range);
                        out.push(FallbackCandidate {
                            tier: 1,
                            package: m.package.clone(),
                            pack_name: bundle.to_string(),
                            kind: PackOverrideKind::Override,
                            old_spec: Some(m.spec.clone()),
                            new_spec,
                            ownership,
                            tried_key: format!("{}@{bundle}#fallback-range", m.package),
                            reason: format!(
                                "generic fallback: workspace range {range} for {} overridden into pack `{bundle}`",
                                m.package
                            ),
                        });
                    }
                }
                Ownership::LedgeredOverride => {
                    // Tier 2: a prior repair already narrowed/overrode
                    // this package for this pack; if this mention carries
                    // a tighter upper-bound cap, intersect (same
                    // mechanics as `nested_conda_cap`).
                    if let Some(existing) = existing_overrides.get(&m.package)
                        && let Some((cap_op, cap_version)) = extract_cap_clause(&m.spec)
                        && let Ok(narrowed) =
                            intersect_range_with_cap(existing, &cap_op, &cap_version)
                        && &narrowed != existing
                    {
                        out.push(FallbackCandidate {
                            tier: 2,
                            package: m.package.clone(),
                            pack_name: bundle.to_string(),
                            kind: PackOverrideKind::Override,
                            old_spec: Some(existing.clone()),
                            new_spec: narrowed,
                            ownership,
                            tried_key: format!("{}@{bundle}#fallback-narrow", m.package),
                            reason: format!(
                                "generic fallback: nested cap {cap_op}{cap_version} narrows existing pack override for {}",
                                m.package
                            ),
                        });
                    }
                }
                Ownership::DepsFromExactPin => {
                    // Tier 3: relax the deps-from-owned exact pin to a
                    // `>=` floor (same doctrine as `deps_from_pin_conflict`
                    // / `no_wheel_transitive_conflict`).
                    if let Some((op, version)) = leading_clause(&m.spec) {
                        let new_spec = format!(">={version}");
                        let _ = op;
                        out.push(FallbackCandidate {
                            tier: 3,
                            package: m.package.clone(),
                            pack_name: bundle.to_string(),
                            kind: PackOverrideKind::Override,
                            old_spec: Some(m.spec.clone()),
                            new_spec,
                            ownership,
                            tried_key: format!("{}@{bundle}#fallback-relax", m.package),
                            reason: format!(
                                "generic fallback: deps-from exact pin {} relaxed",
                                m.package
                            ),
                        });
                    }
                }
            }
        }

        // Tier 5: metadata-skew UN-ROUTE. A mention `m` (e.g. `pillow
        // <11.0,>=9.2.0`) names a `requirer` (e.g. `moviepy`); if that
        // requirer is ITSELF mentioned with an exact pin owned by this
        // same pack (an `AutoRoutedPackPin`, i.e. the pack's own
        // top-level emission -- not a workspace pin, not someone else's
        // transitive), the conflict is between two of the pack's OWN
        // emitted pins: no relax candidate makes sense (neither side is
        // externally constrained), so un-route the requirer instead of
        // guessing a version.
        for m in mentions {
            let Some(requirer) = &m.requirer else {
                continue;
            };
            if is_abi_anchor(&m.package) || self.has_workspace_pin(editor, &m.package) {
                continue;
            }
            let Some(req_mention) = mentions
                .iter()
                .find(|mm| mm.package.eq_ignore_ascii_case(requirer) && mm.spec.starts_with("=="))
            else {
                continue;
            };
            if self.has_workspace_pin(editor, &req_mention.package) {
                continue;
            }
            let req_ownership =
                self.classify_mention_ownership(editor, pack_name, &req_mention.package);
            if req_ownership != Ownership::AutoRoutedPackPin {
                continue;
            }
            out.push(FallbackCandidate {
                tier: 5,
                package: req_mention.package.clone(),
                pack_name: bundle.to_string(),
                kind: PackOverrideKind::Unroute,
                old_spec: Some(req_mention.spec.clone()),
                new_spec: String::new(),
                ownership: req_ownership,
                tried_key: format!("{}@{bundle}#fallback-unroute", req_mention.package),
                reason: format!(
                    "generic fallback: metadata skew ({} {} contradicts {}'s own emitted pin) -- un-routing {}",
                    m.package, m.spec, req_mention.package, req_mention.package
                ),
            });
        }

        // Tier 6: ABI-anchor un-route. An auto-routed pack pin (e.g.
        // `triton ==3.6.0`) whose OWN routed conda build requires an ABI
        // anchor version the workspace/env can't provide (`cuda-version
        // >=12.9,<13` against a cuda-12.8 env -> "for which no candidates
        // were found") has NO relaxable knob: the anchor itself is
        // immutable (tier-loop `continue`s on `AbiAnchor`) and the pin has
        // no visible workspace range for tier 1 to override toward. The
        // cure mirrors tier 5 -- un-route the pin so it ships as its
        // self-contained PyPI wheel (whose manylinux build carries no
        // conda `cuda-version` run-dep) instead of forcing conda to find a
        // build of the EXACT pinned version against an incompatible anchor.
        // The exact version is preserved (the wheel closure already
        // resolved it), so a consumer like torch that pins `triton==X`
        // stays satisfied -- which is why un-route, not relax-to-range, is
        // correct here: relaxing the conda pin would let conda install a
        // triton whose version contradicts torch's own PyPI metadata.
        for m in mentions {
            if !is_abi_anchor(&m.package) {
                continue;
            }
            let Some(requirer) = &m.requirer else {
                continue;
            };
            // Respect real owner intent: a hand-written workspace pin on
            // the requirer is never un-routed out from under the user.
            if self.has_workspace_pin(editor, requirer) {
                continue;
            }
            let Some(req_mention) = mentions
                .iter()
                .find(|mm| mm.package.eq_ignore_ascii_case(requirer) && mm.spec.starts_with("=="))
            else {
                continue;
            };
            if self.classify_mention_ownership(editor, pack_name, &req_mention.package)
                != Ownership::AutoRoutedPackPin
            {
                continue;
            }
            out.push(FallbackCandidate {
                tier: 6,
                package: req_mention.package.clone(),
                pack_name: bundle.to_string(),
                kind: PackOverrideKind::Unroute,
                old_spec: Some(req_mention.spec.clone()),
                new_spec: String::new(),
                ownership: Ownership::AutoRoutedPackPin,
                tried_key: format!("{}@{bundle}#fallback-abi-unroute", req_mention.package),
                reason: format!(
                    "generic fallback: auto-routed pin {} {} requires ABI anchor {} {} the env can't satisfy -- un-routing {} to PyPI",
                    req_mention.package, req_mention.spec, m.package, m.spec, req_mention.package
                ),
            });
        }

        out.sort_by_key(|c| c.tier);
        out.dedup_by(|a, b| {
            a.tier == b.tier && a.package == b.package && a.pack_name == b.pack_name
        });
        out
    }

    fn apply_fallback_candidate(
        &mut self,
        tried: &mut TriedState,
        cand: &FallbackCandidate,
        iter: u32,
    ) -> Option<RepairOutcome> {
        if tried.has(&cand.tried_key, Strategy::PypiOverride) {
            return None;
        }
        let strategy_spec = match cand.kind {
            PackOverrideKind::Override => cand.new_spec.clone(),
            PackOverrideKind::Unroute => format!("unroute:{}", cand.package),
        };
        if self
            .guard_oscillation(&cand.tried_key, &strategy_spec, Strategy::PypiOverride)
            .is_err()
        {
            return None;
        }
        tried.mark(&cand.tried_key, Strategy::PypiOverride, false);
        eprintln!(
            "retread: generic repair [{}] {} {} -> {} ({}, fallback engine)",
            cand.tier,
            cand.package,
            cand.old_spec.as_deref().unwrap_or("(none)"),
            match cand.kind {
                PackOverrideKind::Override => cand.new_spec.as_str(),
                PackOverrideKind::Unroute => "pypi (un-routed)",
            },
            cand.ownership.as_str(),
        );
        let summary_line = match cand.kind {
            PackOverrideKind::Override => format!(
                "would add [{} :: retread-overrides] {} = \"{}\"  (tier: pypi_override; {}; fallback engine)",
                cand.pack_name, cand.package, cand.new_spec, cand.reason
            ),
            PackOverrideKind::Unroute => format!(
                "would add [{} :: retread-keep-pypi] {}  (tier: pypi_override; un-route; {}; fallback engine)",
                cand.pack_name, cand.package, cand.reason
            ),
        };
        let new_spec_ledgered = match cand.kind {
            PackOverrideKind::Override => cand.new_spec.clone(),
            PackOverrideKind::Unroute => "keep-pypi".to_string(),
        };
        Some(RepairOutcome {
            attempt: LedgerAttempt {
                iter,
                package: cand.package.clone(),
                version: cand.old_spec.clone(),
                tier: Strategy::PypiOverride.as_str().to_string(),
                strategy: Strategy::PypiOverride.as_str().to_string(),
                // Distinct from every specific `Conflict::kind()` tag
                // (`NoCandidates`, `CondaWidenNeeded`, ...) -- honestly
                // marks this attempt as the generic engine's own, not a
                // specific rung's, in the ledger/audit trail.
                conflict: "GenericFallback".to_string(),
                source: "retread-fallback".to_string(),
                ts: timestamp(),
                old_spec: cand.old_spec.clone(),
                new_spec: Some(new_spec_ledgered),
                ceiling_policy: None,
                before: None,
                failed: false,
            },
            extra_attempts: Vec::new(),
            summary_line,
            applied: Vec::new(),
            pack_override: None, // caller resolves pack_pixi + writes; see generic_fallback_repair
        })
    }

    /// Generic ownership-driven fallback repair engine entry point.
    /// Called by `retread lock` / `retread solve` ONLY when
    /// `RegexConflictParser::parse` returned `None` for this iteration's
    /// solver error (every specific rung already declined it). Returns
    /// `None` when there is truly nothing actionable (no mentions found,
    /// or no mention could be attributed an owner) -- the caller then
    /// falls through to its existing "unparseable" dead end unchanged.
    /// Returns `Some(Ok(outcome))` when a candidate was applied (caller
    /// must still perform the pack-ledger write itself, same as every
    /// other pack-scoped tier -- see `RepairOutcome::pack_override`'s doc
    /// comment); `Some(Err(package))` when candidates exist but every one
    /// is already tried/oscillation-guarded (exhausted, not
    /// unparseable).
    pub fn generic_fallback_repair(
        &mut self,
        editor: &ManifestEditor,
        tried: &mut TriedState,
        stderr: &str,
        iter: u32,
    ) -> Option<(
        std::result::Result<RepairOutcome, String>,
        Option<PackOverrideWrite>,
    )> {
        let parser = super::parse::RegexConflictParser::new();
        let mentions = parser.extract_generic_mentions(stderr);
        if mentions.is_empty() {
            return None;
        }
        let pack_name = parser.extract_bundle_name(stderr);
        let candidates = self.generate_fallback_candidates(editor, pack_name.as_deref(), &mentions);
        if candidates.is_empty() {
            return None;
        }
        let project_dir = editor.project_dir();
        for cand in &candidates {
            let Some(outcome) = self.apply_fallback_candidate(tried, cand, iter) else {
                continue;
            };
            let Some(pack_pixi) = crate::workspace::WorkspaceManifest::load(project_dir)
                .and_then(|ws| resolve_pack_dir(&ws, project_dir, &cand.pack_name))
                .map(|d| d.join("pixi.toml"))
            else {
                continue;
            };
            let po = PackOverrideWrite {
                bundle: cand.pack_name.clone(),
                pack_pixi,
                package: cand.package.clone(),
                spec: cand.new_spec.clone(),
                kind: cand.kind,
            };
            return Some((Ok(outcome), Some(po)));
        }
        Some((Err(candidates[0].package.clone()), None))
    }
}

/// The first clause of a possibly-two-sided spec string (e.g. `"==2.2.1"`
/// -> `("==", "2.2.1")`, `">=68,<81"` -> `(">=", "68")`).
fn leading_clause(spec: &str) -> Option<(String, String)> {
    let clause = spec.split(',').next()?.trim();
    const OPS: &[&str] = &[">=", "<=", "==", ">", "<"];
    for op in OPS {
        if let Some(rest) = clause.strip_prefix(op) {
            return Some((op.to_string(), rest.trim().to_string()));
        }
    }
    None
}

/// The first upper-bound (`<`/`<=`) clause in a possibly-multi-clause
/// spec string (e.g. `"<11.0,>=9.2.0"` -> `("<", "11.0")`), for tier 2's
/// intersect candidate.
fn extract_cap_clause(spec: &str) -> Option<(String, String)> {
    for clause in spec.split(',') {
        let clause = clause.trim();
        for op in ["<=", "<"] {
            if let Some(rest) = clause.strip_prefix(op) {
                return Some((op.to_string(), rest.trim().to_string()));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::super::parse::{ConflictParser, RegexConflictParser};
    use super::*;

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
        // Genuinely FAILED attempt -- must still be seeded (oscillation
        // protection across runs must survive the fix).
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
            failed: true,
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
    fn seed_tried_state_only_seeds_failed_ledger_entries() {
        // Regression test for the sentry-sdk desync (proof run 4 of
        // pm-isaaclab): a repair that RESOLVED its conflict in-memory
        // (`failed: false`) but was never persisted to
        // `.retread/auto-overrides.json` because the overall lock later
        // died on an unrelated conflict must NOT block a retry on the next
        // invocation. A genuinely failed attempt must still block retry
        // (oscillation protection across runs).
        let path = temp_manifest("[dependencies]\n");
        let editor = ManifestEditor::open(path.clone()).unwrap();
        let manifest_hash = manifest_sha256(&path).unwrap();
        let ledger_path = path.parent().unwrap().join(".retread/solve-ledger.json");
        let mut ledger = SolveLedger::load(&ledger_path, "pixi.toml".into()).unwrap();
        let run = ledger.start_run(
            "default".into(),
            manifest_hash.clone(),
            Some("pixi 0.70.0".into()),
        );
        fn attempt(package: &str, failed: bool) -> LedgerAttempt {
            LedgerAttempt {
                iter: 1,
                package: package.into(),
                version: Some("1.0.0".into()),
                tier: "pypi".into(),
                strategy: "pypi_override".into(),
                conflict: "DepsFromPin".into(),
                source: "retread".into(),
                ts: timestamp(),
                old_spec: None,
                new_spec: None,
                ceiling_policy: None,
                before: None,
                failed,
            }
        }
        // succeeded in-memory but never persisted (the sentry-sdk case).
        ledger.runs[run].attempts.push(attempt("sentry-sdk", false));
        // genuinely failed and must remain blocked.
        ledger.runs[run].attempts.push(attempt("antlr4", true));
        ledger.write_atomic(&ledger_path).unwrap();

        let loaded = SolveLedger::load(&ledger_path, "pixi.toml".into()).unwrap();
        let tried = loaded.seed_tried_state(&path, &manifest_hash, &editor);
        assert!(
            !tried.has("sentry-sdk", Strategy::PypiOverride),
            "successful-but-unpersisted repair must be retried, not treated as exhausted"
        );
        assert!(
            tried.has("antlr4", Strategy::PypiOverride),
            "genuinely failed repair must still block retry (oscillation protection)"
        );
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
        // that class (the closure never reads it). Fix #22: the T1
        // override is recorded in the workspace's `.retread/
        // auto-overrides.json` ledger (not the pack's pixi.toml, and
        // still not the workspace manifest).
        let manifest_text = "[dependencies]\ntorch = \"==2.10.0\"\n\n\
             [feature.gpu.dependencies]\n\
             isaac-pack-latest = { path = \"./pypi-packs/isaac-pack-latest\" }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaac-pack-latest");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_pixi = pack_dir.join("pixi.toml");
        std::fs::write(
            &pack_pixi,
            "[package]\nname = \"isaac-pack-latest\"\nversion = \"6.0.0\"\n",
        )
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
        // and confirm the override lands in the workspace ledger, and the
        // pack manifest stays byte-identical.
        crate::pack_overrides::write_override(
            &project_dir,
            &po.pack_pixi,
            &po.bundle,
            &po.package,
            &po.spec,
            &out.attempt.conflict,
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&pack_pixi).unwrap(),
            "[package]\nname = \"isaac-pack-latest\"\nversion = \"6.0.0\"\n",
            "the pack manifest must stay byte-identical after an auto repair"
        );
        let overrides = crate::pack_overrides::overrides_for_pack(&project_dir, &pack_pixi);
        assert_eq!(overrides.get("torch").map(String::as_str), Some("==2.10.0"));
    }

    #[test]
    fn translate_conda_range_to_pep440_basic_and_glob() {
        assert_eq!(translate_conda_range_to_pep440(">=68,<81"), ">=68,<81");
        // conda's bare `=` normalizes to PEP440's `==`.
        assert_eq!(translate_conda_range_to_pep440("=68"), "==68");
        // conda glob build passes through verbatim under `==` (already
        // PEP440-valid arbitrary equality).
        assert_eq!(translate_conda_range_to_pep440("==83.0.0.*"), "==83.0.0.*");
        assert_eq!(translate_conda_range_to_pep440(">=1,<=2"), ">=1,<=2");
        // Malformed clause (no recognized operator) passes through
        // unchanged rather than being silently dropped.
        assert_eq!(translate_conda_range_to_pep440("garbage"), "garbage");
    }

    #[test]
    fn conda_range_vs_pack_pin_injects_workspace_range_into_pack_not_workspace() {
        // Tenth fix (deps-from hardening): a `CondaRangeVsPackPin`
        // conflict must inject the WORKSPACE'S OWN range into the named
        // pack's own ledger/closure as a pypi override -- never widen the
        // workspace's conda pin, and never touch the workspace manifest.
        let manifest_text = "[dependencies]\nsetuptools = \">=68,<81\"\n\n\
             [feature.gpu.dependencies]\n\
             \"isaaclab-2.3x-pack\" = { path = \"./pypi-packs/isaaclab-2.3x-pack\" }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaaclab-2.3x-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_pixi = pack_dir.join("pixi.toml");
        std::fs::write(
            &pack_pixi,
            "[package]\nname = \"isaaclab-2.3x-pack\"\nversion = \"0.54.2\"\n",
        )
        .unwrap();

        let mut editor = ManifestEditor::open(path.clone()).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let conflict = Conflict::CondaRangeVsPackPin {
            package: "setuptools".into(),
            conda_range: ">=68,<81".into(),
            pack_demand: "83.0.0".into(),
            pack_name: "isaaclab-2.3x-pack".into(),
        };
        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .unwrap();
        assert_eq!(out.attempt.strategy, "pypi_override");
        assert_eq!(out.attempt.old_spec.as_deref(), Some("==83.0.0"));
        assert_eq!(out.attempt.new_spec.as_deref(), Some(">=68,<81"));
        assert!(
            out.applied.is_empty(),
            "must not touch the workspace manifest"
        );
        let po = out.pack_override.expect("expected a pack-override write");
        assert_eq!(po.bundle, "isaaclab-2.3x-pack");
        assert_eq!(po.package, "setuptools");
        assert_eq!(po.spec, ">=68,<81");
        assert_eq!(po.pack_pixi, pack_pixi);

        // Workspace manifest's own conda range is byte-identical -- never
        // widened.
        editor.write_atomic().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), manifest_text);
    }

    #[test]
    fn conda_range_vs_pack_pin_fires_once_per_distinct_pack_owner() {
        // Run 11 named several sibling packs auto-routing the SAME package
        // (setuptools) at exact pins. The tried-state is keyed by
        // (package, pack): each distinct pack owner gets one repair shot;
        // a REPEAT of an already-repaired pair exhausts.
        let manifest_text = "[dependencies]\nsetuptools = \">=68,<81\"\n\n\
             [feature.gpu.dependencies]\n\
             \"isaaclab-2.3x-pack\" = { path = \"./pypi-packs/isaaclab-2.3x-pack\" }\n\
             \"unitree-rl-gym\" = { path = \"./pypi-packs/unitree-rl-gym\" }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        for pack in ["isaaclab-2.3x-pack", "unitree-rl-gym"] {
            let pack_dir = project_dir.join("pypi-packs").join(pack);
            std::fs::create_dir_all(&pack_dir).unwrap();
            std::fs::write(
                pack_dir.join("pixi.toml"),
                format!("[package]\nname = \"{pack}\"\nversion = \"1.0.0\"\n"),
            )
            .unwrap();
        }

        let mut editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let conflict_for = |pack: &str| Conflict::CondaRangeVsPackPin {
            package: "setuptools".into(),
            conda_range: ">=68,<81".into(),
            pack_demand: "83.0.0".into(),
            pack_name: pack.into(),
        };

        let out1 = planner
            .repair(
                &mut editor,
                &mut tried,
                &conflict_for("isaaclab-2.3x-pack"),
                1,
            )
            .expect("first pack owner must repair");
        assert_eq!(
            out1.pack_override.as_ref().unwrap().bundle,
            "isaaclab-2.3x-pack"
        );
        let out2 = planner
            .repair(&mut editor, &mut tried, &conflict_for("unitree-rl-gym"), 2)
            .expect("a DIFFERENT pack owner of the same package must also repair");
        assert_eq!(
            out2.pack_override.as_ref().unwrap().bundle,
            "unitree-rl-gym"
        );
        // Repeat of an already-repaired (package, pack) pair exhausts.
        assert!(
            planner
                .repair(
                    &mut editor,
                    &mut tried,
                    &conflict_for("isaaclab-2.3x-pack"),
                    3
                )
                .is_err(),
            "a repeat of an already-repaired pair must exhaust (repair didn't stick)"
        );
    }

    #[test]
    fn conda_range_vs_pack_pin_refuses_when_pack_is_not_ours() {
        // Guardrail: a `pack_name` that doesn't resolve to one of this
        // workspace's own composed packs (no `path = ...` dependency
        // entry found) must be refused rather than guessed at -- there is
        // no way to confirm its conda companion pin is auto-routed.
        let path = temp_manifest("[dependencies]\nsetuptools = \">=68,<81\"\n");
        let mut editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let conflict = Conflict::CondaRangeVsPackPin {
            package: "setuptools".into(),
            conda_range: ">=68,<81".into(),
            pack_demand: "83.0.0".into(),
            pack_name: "not-a-real-pack".into(),
        };
        let result = planner.repair(&mut editor, &mut tried, &conflict, 1);
        assert!(result.is_err());
    }

    #[test]
    fn conda_range_vs_pack_pin_exempts_abi_anchors() {
        let path = temp_manifest("[dependencies]\npython = \">=3.10,<3.12\"\n");
        let mut editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let conflict = Conflict::CondaRangeVsPackPin {
            package: "python".into(),
            conda_range: ">=3.10,<3.12".into(),
            pack_demand: "3.13.0".into(),
            pack_name: "isaaclab-2.3x-pack".into(),
        };
        let result = planner.repair(&mut editor, &mut tried, &conflict, 1);
        assert!(result.is_err(), "ABI anchors must never be auto-repaired");
    }

    #[test]
    fn end_to_end_setuptools_range_fixture_repairs_via_pack_not_widen() {
        // Regression test for run 11's exhaustion (depsfrom-proof-brief.md):
        // the real captured `.retread/solve-conflicts` trace, parsed and
        // repaired end-to-end, must land a pack override -- not a
        // widen-conda repair on the workspace's own `setuptools` pin.
        const CONDA_INCOMPATIBLE_SETUPTOOLS_RANGE: &str = include_str!(
            "../../tests/fixtures/solve_errors/conda_incompatible_setuptools_range.txt"
        );
        let manifest_text = "[dependencies]\nsetuptools = \">=68,<81\"\n\n\
             [feature.gpu.dependencies]\n\
             \"isaaclab-2.3x-pack\" = { path = \"./pypi-packs/isaaclab-2.3x-pack\" }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaaclab-2.3x-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        std::fs::write(
            pack_dir.join("pixi.toml"),
            "[package]\nname = \"isaaclab-2.3x-pack\"\nversion = \"0.54.2\"\n",
        )
        .unwrap();

        let parser = RegexConflictParser::new();
        let conflict = parser
            .parse(CONDA_INCOMPATIBLE_SETUPTOOLS_RANGE)
            .expect("fixture must parse");

        let mut editor = ManifestEditor::open(path.clone()).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());
        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .expect("repair must succeed on the first rung, no ladder exhaustion");
        assert_eq!(out.attempt.strategy, "pypi_override");
        let po = out
            .pack_override
            .expect("must be a pack override, not a workspace widen");
        assert_eq!(po.bundle, "isaaclab-2.3x-pack");
        assert_eq!(po.package, "setuptools");
        assert_eq!(po.spec, ">=68,<81");

        editor.write_atomic().unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            manifest_text,
            "the workspace's own setuptools range must never be widened"
        );
    }

    #[test]
    fn end_to_end_run13_nested_cap_fixture_narrows_pack_override() {
        // Regression test for run 13's EXIT=2 (depsfrom-proof-brief.md): the
        // real captured `.retread/solve-conflicts/lock-3.txt` trace, parsed
        // and repaired end-to-end against a ledger already carrying the
        // prior `CondaRangeVsPackPin` repair, must narrow that override --
        // not fail to parse, and not touch the workspace manifest.
        const NESTED_CONDA_CAP_PYTORCH_SETUPTOOLS: &str = include_str!(
            "../../tests/fixtures/solve_errors/nested_conda_cap_pytorch_setuptools.txt"
        );
        let manifest_text = "[dependencies]\nsetuptools = \">=68,<81\"\n\n\
             [feature.gpu.dependencies]\n\
             \"isaaclab-2.3x-pack\" = { path = \"./pypi-packs/isaaclab-2.3x-pack\" }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaaclab-2.3x-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_pixi = pack_dir.join("pixi.toml");
        std::fs::write(
            &pack_pixi,
            "[package]\nname = \"isaaclab-2.3x-pack\"\nversion = \"0.54.2\"\n",
        )
        .unwrap();
        // Iteration 2's `CondaRangeVsPackPin` repair (already proven in run
        // 13) already landed this override before iteration 3's nested-cap
        // conflict fired.
        crate::pack_overrides::write_override(
            &project_dir,
            &pack_pixi,
            "isaaclab-2.3x-pack",
            "setuptools",
            ">=68,<81",
            "run-13 iter-2 CondaRangeVsPackPin repair",
        )
        .unwrap();

        let parser = RegexConflictParser::new();
        let conflict = parser
            .parse(NESTED_CONDA_CAP_PYTORCH_SETUPTOOLS)
            .expect("run-13 fixture must parse (was EXIT=2/unparseable before this fix)");

        let mut editor = ManifestEditor::open(path.clone()).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());
        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 3)
            .expect("repair must narrow the existing override, not exhaust");
        assert_eq!(out.attempt.strategy, "pypi_override");
        assert_eq!(out.attempt.old_spec.as_deref(), Some(">=68,<81"));
        assert_eq!(out.attempt.new_spec.as_deref(), Some(">=68,<76"));
        let po = out
            .pack_override
            .expect("must be a pack override, not a workspace widen");
        assert_eq!(po.bundle, "isaaclab-2.3x-pack");
        assert_eq!(po.package, "setuptools");
        assert_eq!(po.spec, ">=68,<76");

        editor.write_atomic().unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            manifest_text,
            "the workspace's own setuptools range must never be widened"
        );
    }

    #[test]
    fn intersect_range_with_cap_narrows_ceiling_only() {
        assert_eq!(
            intersect_range_with_cap(">=68,<81", "<", "76").unwrap(),
            ">=68,<76"
        );
    }

    #[test]
    fn intersect_range_with_cap_is_noop_when_cap_looser_than_existing_ceiling() {
        // A second, LOOSER cap than one already applied must not widen the
        // range back out.
        assert_eq!(
            intersect_range_with_cap(">=68,<76", "<", "81").unwrap(),
            ">=68,<76"
        );
    }

    #[test]
    fn intersect_range_with_cap_handles_bare_floor_with_no_existing_ceiling() {
        assert_eq!(
            intersect_range_with_cap(">=68", "<", "76").unwrap(),
            ">=68,<76"
        );
    }

    #[test]
    fn intersect_range_with_cap_reports_empty_intersection() {
        // Floor at or above the cap's ceiling: no version satisfies both.
        let err = intersect_range_with_cap(">=76,<81", "<", "76").unwrap_err();
        assert!(
            err.contains("floor"),
            "error should explain the clash: {err}"
        );
    }

    #[test]
    fn nested_conda_cap_narrows_an_existing_pack_override() {
        // Run 13 (eleventh fix): a package a prior `CondaRangeVsPackPin`
        // repair already injected into a pack's ledger fails again because
        // a deeper conda run-dep of the SAME pack imposes an additional
        // cap. `NestedCondaCap` must read the ledgered override back and
        // intersect it, not originate a fresh range.
        let manifest_text = "[dependencies]\nsetuptools = \">=68,<81\"\n\n\
             [feature.gpu.dependencies]\n\
             \"isaaclab-2.3x-pack\" = { path = \"./pypi-packs/isaaclab-2.3x-pack\" }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaaclab-2.3x-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_pixi = pack_dir.join("pixi.toml");
        std::fs::write(
            &pack_pixi,
            "[package]\nname = \"isaaclab-2.3x-pack\"\nversion = \"0.54.2\"\n",
        )
        .unwrap();
        // Simulate the PRIOR `CondaRangeVsPackPin` repair having already
        // been persisted to the ledger (this is a caller-side write in
        // production, not something `repair()` itself does).
        crate::pack_overrides::write_override(
            &project_dir,
            &pack_pixi,
            "isaaclab-2.3x-pack",
            "setuptools",
            ">=68,<81",
            "prior CondaRangeVsPackPin repair",
        )
        .unwrap();

        let mut editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let conflict = Conflict::NestedCondaCap {
            package: "setuptools".into(),
            pack_name: "isaaclab-2.3x-pack".into(),
            pack_demand: "80.10.2".into(),
            cap_op: "<".into(),
            cap_version: "76".into(),
            via: "pytorch".into(),
        };
        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .expect("must narrow the existing override, not exhaust");
        assert_eq!(out.attempt.strategy, "pypi_override");
        assert_eq!(out.attempt.old_spec.as_deref(), Some(">=68,<81"));
        assert_eq!(out.attempt.new_spec.as_deref(), Some(">=68,<76"));
        let po = out.pack_override.expect("expected a pack-override write");
        assert_eq!(po.bundle, "isaaclab-2.3x-pack");
        assert_eq!(po.package, "setuptools");
        assert_eq!(po.spec, ">=68,<76");
    }

    #[test]
    fn nested_conda_cap_refuses_to_originate_a_range_from_scratch() {
        // Guardrail: this tier only ever NARROWS an existing
        // `CondaRangeVsPackPin` override -- if none was ever recorded for
        // this (package, pack) pair, refuse rather than guess a range.
        let manifest_text = "[dependencies]\nsetuptools = \">=68,<81\"\n\n\
             [feature.gpu.dependencies]\n\
             \"isaaclab-2.3x-pack\" = { path = \"./pypi-packs/isaaclab-2.3x-pack\" }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaaclab-2.3x-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        std::fs::write(
            pack_dir.join("pixi.toml"),
            "[package]\nname = \"isaaclab-2.3x-pack\"\nversion = \"0.54.2\"\n",
        )
        .unwrap();

        let mut editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());
        let conflict = Conflict::NestedCondaCap {
            package: "setuptools".into(),
            pack_name: "isaaclab-2.3x-pack".into(),
            pack_demand: "80.10.2".into(),
            cap_op: "<".into(),
            cap_version: "76".into(),
            via: "pytorch".into(),
        };
        let result = planner.repair(&mut editor, &mut tried, &conflict, 1);
        assert!(result.is_err());
    }

    #[test]
    fn nested_conda_cap_convergence_loop_is_bounded_by_oscillation_guard() {
        // Each narrowing must strictly shrink (or the repair is refused as
        // empty), so a REPEAT of the identical resulting spec -- the same
        // conflict reported again with no progress made -- must be refused
        // by the existing oscillation guard rather than looping forever.
        let manifest_text = "[dependencies]\nsetuptools = \">=68,<81\"\n\n\
             [feature.gpu.dependencies]\n\
             \"isaaclab-2.3x-pack\" = { path = \"./pypi-packs/isaaclab-2.3x-pack\" }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaaclab-2.3x-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_pixi = pack_dir.join("pixi.toml");
        std::fs::write(
            &pack_pixi,
            "[package]\nname = \"isaaclab-2.3x-pack\"\nversion = \"0.54.2\"\n",
        )
        .unwrap();
        crate::pack_overrides::write_override(
            &project_dir,
            &pack_pixi,
            "isaaclab-2.3x-pack",
            "setuptools",
            ">=68,<76",
            "prior narrowing",
        )
        .unwrap();

        let mut editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());
        // Cap identical to the ceiling the ledger already carries -- the
        // intersection is a no-op, producing the SAME spec as last time.
        let conflict = Conflict::NestedCondaCap {
            package: "setuptools".into(),
            pack_name: "isaaclab-2.3x-pack".into(),
            pack_demand: "75.9.0".into(),
            cap_op: "<".into(),
            cap_version: "76".into(),
            via: "pytorch".into(),
        };
        // First attempt succeeds (marks the resulting spec as tried this run).
        planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .expect("first narrowing attempt succeeds even though it's a no-op change");
        // A second, identical attempt (e.g. the same conflict resurfacing
        // unchanged) must be refused -- no progress, bounded loop.
        assert!(
            planner
                .repair(&mut editor, &mut tried, &conflict, 2)
                .is_err(),
            "repeating the identical narrowed spec must be refused, not loop forever"
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
        std::fs::write(
            pack_dir.join("pixi.toml"),
            "[package]\nname = \"p\"\nversion = \"6.0\"\n",
        )
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
        std::fs::write(
            pack_dir.join("pixi.toml"),
            "[package]\nname = \"p\"\nversion = \"6.0\"\n",
        )
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

    // ---- Fix #21 (exactpin-fix-brief.md): the full "companion chaining"
    // scenario acceptance-final.md observed live but couldn't complete --
    // a fix-#20 torch pypi_override reaches the pack's uv closure, which
    // re-emits its torchvision companion as an exact `==` conda run-dep,
    // clashing with the workspace's own exact torchvision pin; repairing
    // THAT surfaces the same shape one layer deeper for torchaudio. Each
    // hop is the direct (non-JSON-RPC) rattler conda-solver prose fix #21
    // teaches `parse_conda_incompatible` to read (`conda_incompatible_
    // exact` + `would_require_exact`), carrying `pack_name` so every hop
    // resolves via `resolve_pack_override`/`try_pack_override` -- never
    // the workspace conda-pin-owner path. Three sequential `repair()`
    // calls (mocking the driver's iter loop) must converge to exactly the
    // manual trio the acceptance brief pinned by hand: torch==2.10.0,
    // torchvision==0.25.0, torchaudio==2.10.0.
    const CONDA_INCOMPATIBLE_TORCHVISION_EXACT: &str =
        include_str!("../../tests/fixtures/solve_errors/conda_incompatible_torchvision_exact.txt");
    const CONDA_INCOMPATIBLE_TORCHAUDIO_EXACT: &str = concat!(
        "Cannot solve the request because of: torchaudio ==2.10.0 cannot be\n",
        "installed because there are no viable options:\n",
        "  torchaudio 2.10.0 would require\n",
        "     python_abi 3.13.*, for which no candidates were found.\n",
        "The following packages are incompatible\n",
        "isaac-pack-latest * can be installed with any of the following options:\n",
        "   isaac-pack-latest 6.1.11 would require\n",
        "      torchaudio ==2.11.0, which can be installed with any of the following options:\n",
        "         torchaudio 2.11.0\n",
    );

    #[tokio::test]
    async fn pack_override_chain_converges_torch_torchvision_torchaudio_exact_pin() {
        let manifest_text = concat!(
            "[dependencies]\n\n",
            "[feature.gpu.dependencies]\n",
            "torch = \"==2.10.0\"  # retread:pin\n",
            "torchvision = \"==0.25.0\"  # retread:pin\n",
            "torchaudio = \"==2.10.0\"  # retread:pin\n\n",
            "[feature.isaaclab-latest.dependencies]\n",
            "isaac-pack-latest = { path = \"./pypi-packs/isaac-pack-latest\" }\n\n",
            "[environments]\n",
            "isaaclab-gpu-latest = { features = [\"gpu\", \"isaaclab-latest\"] }\n",
        );
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaac-pack-latest");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_pixi = pack_dir.join("pixi.toml");
        std::fs::write(
            &pack_pixi,
            "[package]\nname = \"isaac-pack-latest\"\nversion = \"6.1.11\"\n",
        )
        .unwrap();

        let mut editor = ManifestEditor::open(path.clone()).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());
        let parser = RegexConflictParser::new();

        // iter 1: torch, backend-closure-style conflict (as fix #20 shipped
        // it), pack_name already Some -- resolves straight to
        // resolve_pack_override without needing the parser fixture.
        let torch_conflict = Conflict::CondaWidenNeeded {
            package: "torch".into(),
            op: ">=".into(),
            floor: "2.11.0".into(),
            conda_version: "==2.10.0".into(),
            requiring_chain: Vec::new(),
            pack_name: Some("isaac-pack-latest".into()),
        };
        let out1 = planner
            .repair(&mut editor, &mut tried, &torch_conflict, 1)
            .unwrap();
        let po1 = out1.pack_override.expect("iter 1 expected a pack override");
        assert_eq!(po1.package, "torch");
        assert_eq!(po1.spec, "==2.10.0");
        assert!(out1.applied.is_empty());
        crate::pack_overrides::write_override(
            &project_dir,
            &po1.pack_pixi,
            &po1.bundle,
            &po1.package,
            &po1.spec,
            &out1.attempt.conflict,
        )
        .unwrap();

        // iter 2: torchvision, the fix #21 exact-pin companion shape
        // (verbatim acceptance-final.md fixture).
        let torchvision_conflict = parser
            .parse(CONDA_INCOMPATIBLE_TORCHVISION_EXACT)
            .expect("torchvision exact-pin fixture must parse (fix #21)");
        let out2 = planner
            .repair(&mut editor, &mut tried, &torchvision_conflict, 2)
            .unwrap();
        let po2 = out2.pack_override.expect("iter 2 expected a pack override");
        assert_eq!(po2.package, "torchvision");
        assert_eq!(po2.spec, "==0.25.0");
        assert!(out2.applied.is_empty());
        crate::pack_overrides::write_override(
            &project_dir,
            &po2.pack_pixi,
            &po2.bundle,
            &po2.package,
            &po2.spec,
            &out2.attempt.conflict,
        )
        .unwrap();

        // iter 3: torchaudio, same shape one layer deeper.
        let torchaudio_conflict = parser
            .parse(CONDA_INCOMPATIBLE_TORCHAUDIO_EXACT)
            .expect("torchaudio exact-pin fixture must parse (fix #21)");
        let out3 = planner
            .repair(&mut editor, &mut tried, &torchaudio_conflict, 3)
            .unwrap();
        let po3 = out3.pack_override.expect("iter 3 expected a pack override");
        assert_eq!(po3.package, "torchaudio");
        assert_eq!(po3.spec, "==2.10.0");
        assert!(out3.applied.is_empty());
        crate::pack_overrides::write_override(
            &project_dir,
            &po3.pack_pixi,
            &po3.bundle,
            &po3.package,
            &po3.spec,
            &out3.attempt.conflict,
        )
        .unwrap();

        // Converged to exactly the manual trio; workspace manifest AND the
        // pack manifest were never touched by any of the three hops -- all
        // three overrides landed in the ledger.
        editor.write_atomic().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), manifest_text);
        assert_eq!(
            std::fs::read_to_string(&pack_pixi).unwrap(),
            "[package]\nname = \"isaac-pack-latest\"\nversion = \"6.1.11\"\n"
        );
        let overrides = crate::pack_overrides::overrides_for_pack(&project_dir, &pack_pixi);
        assert_eq!(overrides.get("torch").map(String::as_str), Some("==2.10.0"));
        assert_eq!(
            overrides.get("torchvision").map(String::as_str),
            Some("==0.25.0")
        );
        assert_eq!(
            overrides.get("torchaudio").map(String::as_str),
            Some("==2.10.0")
        );
    }

    // ---- deps-from intrinsic-pypi-conflict repair (wandb/sentry-sdk
    // proof, step4-lock-run2.log): a `retread-deps-from` root pins two
    // packages whose PyPI metadata mutually conflicts, with NO conda pin
    // involved at all. `deps_from_pin_conflict` must confirm ownership
    // (re-fetching the bundle's deps-from source(s) off disk) before
    // relaxing the exact pin to a floor -- these tests use `#[tokio::test
    // (flavor = "multi_thread")]` because `deps_from_owns_exact_pin`
    // bridges its async fetch via `block_in_place`, which panics on the
    // (default) current-thread test runtime.

    fn deps_from_pack(project_dir: &std::path::Path, requirements_txt: &str) -> PathBuf {
        let pack_dir = project_dir.join("pypi-packs/deps-from-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        std::fs::write(
            pack_dir.join("pixi.toml"),
            concat!(
                "[package]\nname = \"deps-from-pack\"\nversion = \"0.1.0\"\n\n",
                "[package.build.config]\n",
                "retread-deps-from = \"requirements.txt\"\n",
            ),
        )
        .unwrap();
        std::fs::write(pack_dir.join("requirements.txt"), requirements_txt).unwrap();
        pack_dir
    }

    fn deps_from_manifest(pack_rel: &str) -> String {
        format!(
            concat!(
                "[dependencies]\n\n",
                "[feature.pm.dependencies]\n",
                "deps-from-pack = {{ path = \"{}\" }}\n\n",
                "[environments]\n",
                "pm = {{ features = [\"pm\"] }}\n",
            ),
            pack_rel
        )
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn deps_from_pin_conflict_relaxes_owned_exact_pin_to_floor() {
        let manifest_text = deps_from_manifest("./pypi-packs/deps-from-pack");
        let path = temp_manifest(&manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        deps_from_pack(&project_dir, "sentry-sdk==1.38.0\nwandb==0.23.0\n");

        let mut editor = ManifestEditor::open(path.clone()).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let conflict = Conflict::DepsFromPin {
            package: "sentry-sdk".into(),
            pinned_version: "1.38.0".into(),
            op: ">=".into(),
            floor: "2.0.0".into(),
            requirer: "wandb".into(),
            pack_name: Some("deps-from-pack".into()),
        };
        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .expect("deps-from-owned exact pin must relax, not exhaust");
        assert_eq!(out.attempt.strategy, "pypi_override");
        assert_eq!(out.attempt.package, "sentry-sdk");
        assert_eq!(out.attempt.new_spec.as_deref(), Some(">=2.0.0"));
        assert!(
            out.applied.is_empty(),
            "workspace manifest must stay untouched"
        );

        let po = out.pack_override.expect("expected a pack-override write");
        assert_eq!(po.bundle, "deps-from-pack");
        assert_eq!(po.package, "sentry-sdk");
        assert_eq!(po.spec, ">=2.0.0");

        editor.write_atomic().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), manifest_text);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn deps_from_pin_conflict_refuses_non_deps_from_participant() {
        // The bundle's deps-from root doesn't mention `sentry-sdk` at all
        // (e.g. the requirements file changed, or the conflict was
        // misattributed) -- ownership can't be confirmed, so repair must
        // report exhausted rather than guess at relaxing an arbitrary
        // pypi pin with no provenance.
        let manifest_text = deps_from_manifest("./pypi-packs/deps-from-pack");
        let path = temp_manifest(&manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        deps_from_pack(&project_dir, "numpy==1.26.0\n");

        let mut editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let conflict = Conflict::DepsFromPin {
            package: "sentry-sdk".into(),
            pinned_version: "1.38.0".into(),
            op: ">=".into(),
            floor: "2.0.0".into(),
            requirer: "wandb".into(),
            pack_name: Some("deps-from-pack".into()),
        };
        let err = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .expect_err("non-deps-from-owned pin must not be auto-relaxed");
        assert_eq!(err, "sentry-sdk");
    }

    #[test]
    fn deps_from_pin_conflict_exempts_abi_anchor_before_any_disk_io() {
        // `python`/`python_abi`/etc. are never relaxable regardless of
        // deps-from provenance (same guardrail every other repair tier
        // reuses) -- and the anchor guard must fire BEFORE any pack
        // lookup, so this needs no pack on disk at all (pack_name points
        // at a bundle that doesn't exist in the manifest).
        let path = temp_manifest("[dependencies]\n");
        let mut editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let conflict = Conflict::DepsFromPin {
            package: "python".into(),
            pinned_version: "3.11.15".into(),
            op: ">=".into(),
            floor: "3.12".into(),
            requirer: "some-pkg".into(),
            pack_name: Some("no-such-pack".into()),
        };
        let err = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .expect_err("ABI anchor must never be relaxed");
        assert_eq!(err, "python");
    }

    // ---- no-wheel-transitive repair (hydra-core/antlr4-python3-runtime
    // proof, step4-lock-run3.log): a `retread-deps-from` root's own exact
    // pin wildcard-pins a transitive dependency into a version range with
    // NO usable wheel under `--no-build` at all -- reached only after the
    // backend's sdist-only self-heal ladder (conda-route, sdist
    // auto-build) already exhausted both rungs for the wheel-less
    // package. Repair must relax the REQUIRER's (`hydra-core`) own exact
    // pin, not the wheel-less package (there is nothing to relax on a
    // package retread doesn't pin at all).

    #[tokio::test(flavor = "multi_thread")]
    async fn no_wheel_transitive_relaxes_owned_requirer_pin_to_floor() {
        let manifest_text = deps_from_manifest("./pypi-packs/deps-from-pack");
        let path = temp_manifest(&manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        deps_from_pack(&project_dir, "hydra-core==1.3.2\n");

        let mut editor = ManifestEditor::open(path.clone()).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let conflict = Conflict::NoWheelTransitive {
            package: "antlr4-python3-runtime".into(),
            range: ">=4.9,<=4.9.3".into(),
            requirer: "hydra-core".into(),
            requirer_pin: "1.3.2".into(),
            pack_name: Some("deps-from-pack".into()),
        };
        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .expect("deps-from-owned requirer pin must relax, not exhaust");
        assert_eq!(out.attempt.strategy, "pypi_override");
        assert_eq!(out.attempt.package, "hydra-core");
        assert_eq!(out.attempt.new_spec.as_deref(), Some(">=1.3.2"));
        assert!(
            out.applied.is_empty(),
            "workspace manifest must stay untouched"
        );

        let po = out.pack_override.expect("expected a pack-override write");
        assert_eq!(po.bundle, "deps-from-pack");
        assert_eq!(po.package, "hydra-core");
        assert_eq!(po.spec, ">=1.3.2");

        editor.write_atomic().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), manifest_text);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn no_wheel_transitive_refuses_non_deps_from_requirer() {
        // The bundle's deps-from root doesn't mention `hydra-core` at
        // all -- ownership can't be confirmed, so repair must report
        // exhausted rather than guess at relaxing an arbitrary pin.
        let manifest_text = deps_from_manifest("./pypi-packs/deps-from-pack");
        let path = temp_manifest(&manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        deps_from_pack(&project_dir, "numpy==1.26.0\n");

        let mut editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let conflict = Conflict::NoWheelTransitive {
            package: "antlr4-python3-runtime".into(),
            range: ">=4.9,<=4.9.3".into(),
            requirer: "hydra-core".into(),
            requirer_pin: "1.3.2".into(),
            pack_name: Some("deps-from-pack".into()),
        };
        let err = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .expect_err("non-deps-from-owned requirer pin must not be auto-relaxed");
        assert_eq!(err, "hydra-core");
    }

    #[test]
    fn no_wheel_transitive_exempts_abi_anchor_before_any_disk_io() {
        let path = temp_manifest("[dependencies]\n");
        let mut editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let conflict = Conflict::NoWheelTransitive {
            package: "antlr4-python3-runtime".into(),
            range: ">=4.9,<=4.9.3".into(),
            requirer: "python".into(),
            requirer_pin: "3.11.15".into(),
            pack_name: Some("no-such-pack".into()),
        };
        let err = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .expect_err("ABI anchor must never be relaxed");
        assert_eq!(err, "python");
    }

    // ---- generic ownership-driven fallback repair engine ------------------

    const NESTED_CONDA_CAP_FALLBACK_FIXTURE: &str =
        include_str!("../../tests/fixtures/solve_errors/nested_conda_cap_pytorch_setuptools.txt");
    const PILLOW_MOVIEPY_FALLBACK_FIXTURE: &str = include_str!(
        "../../tests/fixtures/solve_errors/pypi_conda_metadata_skew_pillow_moviepy.txt"
    );
    const TRITON_CUDA_VERSION_FALLBACK_FIXTURE: &str =
        include_str!("../../tests/fixtures/solve_errors/uv_closure_triton_cuda_version_abi.txt");

    #[test]
    fn classify_mention_ownership_abi_anchor_wins_regardless_of_pack() {
        let path = temp_manifest("[dependencies]\n");
        let editor = ManifestEditor::open(path).unwrap();
        let planner = RepairPlanner::new("default".into());
        assert_eq!(
            planner.classify_mention_ownership(&editor, Some("some-pack"), "python"),
            Ownership::AbiAnchor
        );
        assert_eq!(
            planner.classify_mention_ownership(&editor, None, "python"),
            Ownership::AbiAnchor
        );
    }

    #[test]
    fn classify_mention_ownership_workspace_pin_when_no_pack_context() {
        let path = temp_manifest("[dependencies]\nnumpy = \">=1.26,<2\"\n");
        let editor = ManifestEditor::open(path).unwrap();
        let planner = RepairPlanner::new("default".into());
        assert_eq!(
            planner.classify_mention_ownership(&editor, None, "numpy"),
            Ownership::WorkspacePin
        );
        assert_eq!(
            planner.classify_mention_ownership(&editor, None, "some-other-package"),
            Ownership::Unowned
        );
    }

    #[test]
    fn classify_mention_ownership_unowned_when_pack_unresolvable() {
        let path = temp_manifest("[dependencies]\n");
        let editor = ManifestEditor::open(path).unwrap();
        let planner = RepairPlanner::new("default".into());
        assert_eq!(
            planner.classify_mention_ownership(&editor, Some("no-such-pack"), "setuptools"),
            Ownership::Unowned
        );
    }

    #[test]
    fn classify_mention_ownership_auto_routed_pack_pin_when_named_pack_resolves() {
        let manifest_text = "[dependencies]\n\n\
             [feature.gpu.dependencies]\n\
             \"isaaclab-2.3x-pack\" = { path = \"./pypi-packs/isaaclab-2.3x-pack\" }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaaclab-2.3x-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        std::fs::write(
            pack_dir.join("pixi.toml"),
            "[package]\nname = \"isaaclab-2.3x-pack\"\nversion = \"0.54.2\"\n",
        )
        .unwrap();
        let editor = ManifestEditor::open(path).unwrap();
        let planner = RepairPlanner::new("default".into());
        assert_eq!(
            planner.classify_mention_ownership(&editor, Some("isaaclab-2.3x-pack"), "setuptools"),
            Ownership::AutoRoutedPackPin
        );
    }

    #[test]
    fn classify_mention_ownership_ledgered_override_when_a_prior_repair_already_wrote_one() {
        let manifest_text = "[dependencies]\n\n\
             [feature.gpu.dependencies]\n\
             \"isaaclab-2.3x-pack\" = { path = \"./pypi-packs/isaaclab-2.3x-pack\" }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaaclab-2.3x-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_pixi = pack_dir.join("pixi.toml");
        std::fs::write(
            &pack_pixi,
            "[package]\nname = \"isaaclab-2.3x-pack\"\nversion = \"0.54.2\"\n",
        )
        .unwrap();
        crate::pack_overrides::write_override(
            &project_dir,
            &pack_pixi,
            "isaaclab-2.3x-pack",
            "setuptools",
            ">=68,<81",
            "prior repair",
        )
        .unwrap();
        let editor = ManifestEditor::open(path).unwrap();
        let planner = RepairPlanner::new("default".into());
        assert_eq!(
            planner.classify_mention_ownership(&editor, Some("isaaclab-2.3x-pack"), "setuptools"),
            Ownership::LedgeredOverride
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn classify_mention_ownership_deps_from_exact_pin_when_bundle_owns_it() {
        let manifest_text = deps_from_manifest("./pypi-packs/deps-from-pack");
        let path = temp_manifest(&manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        deps_from_pack(&project_dir, "sentry-sdk==1.38.0\n");
        let editor = ManifestEditor::open(path).unwrap();
        let planner = RepairPlanner::new("default".into());
        assert_eq!(
            planner.classify_mention_ownership(&editor, Some("deps-from-pack"), "sentry-sdk"),
            Ownership::DepsFromExactPin
        );
    }

    #[test]
    fn generate_fallback_candidates_orders_by_doctrine_tier() {
        // Synthetic mentions spanning tiers 1 (workspace-range-vs-owned-pin),
        // 2 (ledgered-override intersect), and 5 (metadata-skew un-route),
        // deliberately listed out of order -- the generator must sort by
        // doctrine tier regardless of extraction order.
        let manifest_text = "[dependencies]\nsetuptools = \">=68,<81\"\n\n\
             [feature.gpu.dependencies]\n\
             \"isaaclab-2.3x-pack\" = { path = \"./pypi-packs/isaaclab-2.3x-pack\" }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaaclab-2.3x-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_pixi = pack_dir.join("pixi.toml");
        std::fs::write(
            &pack_pixi,
            "[package]\nname = \"isaaclab-2.3x-pack\"\nversion = \"0.54.2\"\n",
        )
        .unwrap();
        crate::pack_overrides::write_override(
            &project_dir,
            &pack_pixi,
            "isaaclab-2.3x-pack",
            "pillow",
            ">=9.2.0,<11.3.0",
            "prior repair",
        )
        .unwrap();
        let editor = ManifestEditor::open(path).unwrap();
        let planner = RepairPlanner::new("default".into());
        let mentions = vec![
            // Tier 5 candidate material (listed first, out of order).
            super::super::parse::Mention {
                package: "pillow".into(),
                spec: "<11.0,>=9.2.0".into(),
                requirer: Some("moviepy".into()),
            },
            super::super::parse::Mention {
                package: "moviepy".into(),
                spec: "==2.2.1".into(),
                requirer: None,
            },
            // Tier 2 candidate material: a tighter cap than the ledgered
            // pillow override above.
            super::super::parse::Mention {
                package: "pillow".into(),
                spec: "<10.0".into(),
                requirer: Some("some-other-dep".into()),
            },
            // Tier 1 candidate material: setuptools has a visible
            // workspace range AND an auto-routed exact companion demand.
            super::super::parse::Mention {
                package: "setuptools".into(),
                spec: "==83.0.0".into(),
                requirer: Some("isaaclab-2.3x-pack".into()),
            },
        ];
        let candidates =
            planner.generate_fallback_candidates(&editor, Some("isaaclab-2.3x-pack"), &mentions);
        let tiers: Vec<u8> = candidates.iter().map(|c| c.tier).collect();
        let mut sorted = tiers.clone();
        sorted.sort();
        assert_eq!(tiers, sorted, "candidates must already be in tier order");
        assert!(
            tiers.contains(&1),
            "expected a tier-1 (workspace-range) candidate: {candidates:?}"
        );
        assert!(
            tiers.contains(&2),
            "expected a tier-2 (ledgered-override intersect) candidate: {candidates:?}"
        );
        assert!(
            tiers.contains(&5),
            "expected a tier-5 (un-route) candidate: {candidates:?}"
        );
    }

    #[test]
    fn end_to_end_run13_nested_cap_via_generic_fallback_without_specific_rung() {
        // Run 13 shape, fed through the GENERIC fallback engine directly
        // (never `parser.parse()`, never the specific `NestedCondaCap`
        // rung) -- proves the fallback engine alone can narrow an existing
        // pack override via the generic extractor + ownership
        // classification + tier-2 intersect candidate.
        let manifest_text = "[dependencies]\nsetuptools = \">=68,<81\"\n\n\
             [feature.gpu.dependencies]\n\
             \"isaaclab-2.3x-pack\" = { path = \"./pypi-packs/isaaclab-2.3x-pack\" }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaaclab-2.3x-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_pixi = pack_dir.join("pixi.toml");
        std::fs::write(
            &pack_pixi,
            "[package]\nname = \"isaaclab-2.3x-pack\"\nversion = \"0.54.2\"\n",
        )
        .unwrap();
        // Simulate the PRIOR `CondaRangeVsPackPin` repair having already
        // been persisted to the ledger (same setup as
        // `nested_conda_cap_narrows_an_existing_pack_override`, but here
        // the generic engine -- not the specific `NestedCondaCap` tier --
        // must find and narrow it).
        crate::pack_overrides::write_override(
            &project_dir,
            &pack_pixi,
            "isaaclab-2.3x-pack",
            "setuptools",
            ">=68,<81",
            "prior CondaRangeVsPackPin repair",
        )
        .unwrap();

        let editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let (result, po) = planner
            .generic_fallback_repair(&editor, &mut tried, NESTED_CONDA_CAP_FALLBACK_FIXTURE, 1)
            .expect("generic fallback engine must find an actionable mention");
        let outcome = result.expect("must narrow the existing override, not exhaust");
        assert_eq!(outcome.attempt.strategy, "pypi_override");
        assert_eq!(outcome.attempt.old_spec.as_deref(), Some(">=68,<81"));
        assert_eq!(outcome.attempt.new_spec.as_deref(), Some(">=68,<76"));
        let po = po.expect("expected a pack-override write");
        assert_eq!(po.bundle, "isaaclab-2.3x-pack");
        assert_eq!(po.package, "setuptools");
        assert_eq!(po.spec, ">=68,<76");
        assert_eq!(po.kind, PackOverrideKind::Override);
    }

    #[test]
    fn end_to_end_run15_pypi_conda_metadata_skew_unroutes_moviepy() {
        // Run 15 shape: NO specific rung recognizes this at all (asserted
        // separately in `parse.rs`'s
        // `pypi_conda_metadata_skew_stays_unparseable_to_every_specific_rung`)
        // -- the generic fallback engine must classify `moviepy` as an
        // auto-routed pack pin implicated in a metadata-skew conflict with
        // `pillow` (also the pack's own emission, no workspace pin
        // touching either) and generate an UN-ROUTE candidate for
        // `moviepy`, writing to the ledger's `unrouted` sink rather than
        // an override spec.
        let manifest_text = "[dependencies]\n\n\
             [feature.pm.dependencies]\n\
             \"protomotions-deps-pack\" = { path = \"./pypi-packs/protomotions-deps-pack\" }\n\n\
             [environments]\npm = { features = [\"pm\"] }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/protomotions-deps-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        std::fs::write(
            pack_dir.join("pixi.toml"),
            "[package]\nname = \"protomotions-deps-pack\"\nversion = \"3.1\"\n",
        )
        .unwrap();

        let editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let (result, po) = planner
            .generic_fallback_repair(&editor, &mut tried, PILLOW_MOVIEPY_FALLBACK_FIXTURE, 1)
            .expect("generic fallback engine must find an actionable mention");
        let outcome = result.expect("must un-route moviepy, not exhaust");
        assert_eq!(outcome.attempt.strategy, "pypi_override");
        let po = po.expect("expected a pack-override write");
        assert_eq!(po.bundle, "protomotions-deps-pack");
        assert_eq!(po.package, "moviepy");
        assert_eq!(po.kind, PackOverrideKind::Unroute);

        // And the ledger sink is actually the un-route table, not the
        // ordinary overrides table.
        crate::pack_overrides::write_unroute(
            &project_dir,
            &po.pack_pixi,
            &po.bundle,
            &po.package,
            &outcome.attempt.conflict,
        )
        .unwrap();
        let unrouted = crate::pack_overrides::unrouted_for_pack(&project_dir, &po.pack_pixi);
        assert!(unrouted.contains("moviepy"));

        // And `merge_ledger_overrides` actually feeds it into
        // `RetreadConfig.keep_pypi`, the knob the real auto-route sweep
        // reads (`plan_auto_route_round` / `AutoRouteOptions.keep_pypi`).
        let mut config: crate::config::RetreadConfig =
            serde_json::from_value(serde_json::json!({"retread-wheels": {}})).unwrap();
        crate::pack_overrides::merge_ledger_overrides(&mut config, &project_dir, &po.pack_pixi);
        assert!(config.keep_pypi.contains(&"moviepy".to_string()));
    }

    #[test]
    fn end_to_end_run16c_triton_cuda_version_abi_unroutes_triton() {
        // Run 16c shape (lock-7.txt): an auto-routed pack pin
        // `triton ==3.6.0` whose routed conda build requires the ABI
        // anchor `cuda-version >=12.9,<13`, against a workspace pinned to
        // cuda 12.8 -> "for which no candidates were found". NO specific
        // rung recognizes it (the previous run died with "could not parse
        // solver error"), and tier 1 declines (no workspace conda range on
        // triton). The generic engine's tier-6 ABI-anchor un-route must
        // fire: un-route triton so its self-contained PyPI wheel ships at
        // the exact resolved version instead of forcing conda to find a
        // 3.6.0 build against an incompatible cuda anchor.
        let manifest_text = "[dependencies]\n\n\
             [feature.pm.dependencies]\n\
             \"protomotions-deps-pack\" = { path = \"./pypi-packs/protomotions-deps-pack\" }\n\n\
             [environments]\npm = { features = [\"pm\"] }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/protomotions-deps-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        std::fs::write(
            pack_dir.join("pixi.toml"),
            "[package]\nname = \"protomotions-deps-pack\"\nversion = \"3.1\"\n",
        )
        .unwrap();

        let editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());

        let (result, po) = planner
            .generic_fallback_repair(&editor, &mut tried, TRITON_CUDA_VERSION_FALLBACK_FIXTURE, 1)
            .expect("generic fallback engine must find an actionable mention");
        let outcome = result.expect("must un-route triton, not exhaust");
        assert_eq!(outcome.attempt.strategy, "pypi_override");
        let po = po.expect("expected a pack-override write");
        assert_eq!(po.bundle, "protomotions-deps-pack");
        assert_eq!(po.package, "triton");
        assert_eq!(po.kind, PackOverrideKind::Unroute);

        // The ledger sink is the un-route table, and it feeds
        // `RetreadConfig.keep_pypi` -- the knob the auto-route sweep reads.
        crate::pack_overrides::write_unroute(
            &project_dir,
            &po.pack_pixi,
            &po.bundle,
            &po.package,
            &outcome.attempt.conflict,
        )
        .unwrap();
        let unrouted = crate::pack_overrides::unrouted_for_pack(&project_dir, &po.pack_pixi);
        assert!(unrouted.contains("triton"));
    }

    #[test]
    fn generic_fallback_repair_returns_none_when_no_mentions_found() {
        let path = temp_manifest("[dependencies]\n");
        let editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());
        assert!(
            planner
                .generic_fallback_repair(&editor, &mut tried, "totally unrelated text", 1)
                .is_none()
        );
    }

    #[test]
    fn generic_fallback_repair_exhausts_on_repeat_rather_than_looping() {
        let manifest_text = "[dependencies]\nsetuptools = \">=68,<81\"\n\n\
             [feature.gpu.dependencies]\n\
             \"isaaclab-2.3x-pack\" = { path = \"./pypi-packs/isaaclab-2.3x-pack\" }\n";
        let path = temp_manifest(manifest_text);
        let project_dir = path.parent().unwrap().to_path_buf();
        let pack_dir = project_dir.join("pypi-packs/isaaclab-2.3x-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_pixi = pack_dir.join("pixi.toml");
        std::fs::write(
            &pack_pixi,
            "[package]\nname = \"isaaclab-2.3x-pack\"\nversion = \"0.54.2\"\n",
        )
        .unwrap();
        crate::pack_overrides::write_override(
            &project_dir,
            &pack_pixi,
            "isaaclab-2.3x-pack",
            "setuptools",
            ">=68,<76",
            "already narrowed to the fixture's own cap",
        )
        .unwrap();

        let editor = ManifestEditor::open(path).unwrap();
        let mut tried = TriedState::default();
        let mut planner = RepairPlanner::new("default".into());
        // The existing override already matches what the nested cap would
        // narrow it to -- intersecting is a no-op, so tier 2 offers no
        // candidate at all for this mention (and none of the other
        // mentions in this fixture attribute an owner either: `setuptools`
        // itself is workspace-pinned, which gates tier 5's un-route). The
        // engine must recognize there is nothing actionable and return
        // `None` (falls through to the caller's normal "unparseable" path)
        // rather than manufacture a no-op repair or loop.
        assert!(
            planner
                .generic_fallback_repair(&editor, &mut tried, NESTED_CONDA_CAP_FALLBACK_FIXTURE, 1)
                .is_none(),
            "a no-op intersection must not be offered as a candidate"
        );
    }
}
