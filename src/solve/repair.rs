use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::manifest::{AppliedEdit, EntrySnapshot, ManifestEditor, TableKind, write_atomic};
use super::parse::Conflict;

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
}

impl RepairPlanner {
    pub fn new(feature: String) -> Self {
        Self {
            feature,
            ceiling_policy: WidenCeilingPolicy::NextMajor,
            relax_preference: RelaxPreference::Conda,
            current_run_attempts: HashSet::new(),
        }
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
            } => {
                let target = PinTarget {
                    package,
                    version: floor,
                    iter,
                    conflict,
                };
                self.conda_widen_needed(editor, tried, &target, op, conda_version)
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
    ) -> std::result::Result<RepairOutcome, String> {
        let package = target.package;
        if self.relax_preference == RelaxPreference::Conda {
            let user_owned_conda_pin =
                editor.has_user_entry(&self.feature, TableKind::Conda, package);
            if user_owned_conda_pin && !tried.has(package, Strategy::PypiOverride) {
                match self.pypi_override_from_conda(editor, tried, target, conda_version) {
                    Ok(out) => return Ok(out),
                    // Override tier unavailable (ABI anchor / oscillation
                    // guard) -- fall through to widen/boundary below.
                    Err(_) => {}
                }
            }
        }
        if !tried.has(package, Strategy::WidenConda) {
            self.widen(editor, tried, target, op)
        } else {
            self.boundary(editor, tried, target, Reason::CondaBoundary)
        }
    }

    /// T1: relax the pypi requirement to accept the conda-provided version
    /// instead of touching the (user-owned) conda pin. The conda pin is
    /// left byte-for-byte untouched -- conda manifest is source of truth.
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
    ) -> std::result::Result<RepairOutcome, String> {
        let (package, floor) = (target.package, target.version);
        self.guard_anchor(package)?;
        let spec = widen_spec(op, floor, self.ceiling_policy);
        self.guard_oscillation(package, &spec, Strategy::WidenConda)?;
        let edit = editor.set_conda_widen(&self.feature, package, &spec);
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
        };
        let out = planner
            .repair(&mut editor, &mut tried, &conflict, 1)
            .unwrap();
        assert_eq!(out.attempt.strategy, "widen-conda");
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
}
