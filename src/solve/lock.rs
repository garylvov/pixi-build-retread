//! `retread lock`: `pixi lock` subprocess with bounded, ledgered auto-repair.
//!
//! Thin sibling of `retread solve` that reuses the same repair ladder
//! (`RepairPlanner`), conflict parser (`RegexConflictParser`), manifest
//! editor (`ManifestEditor`), and ledger (`SolveLedger`) but drives `pixi
//! lock` instead of `pixi install` and bounds itself by a repair *count*
//! (`--max-repairs`, default 10) rather than a raw iteration count.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::Result;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use super::error::{EXIT_EXHAUSTED, EXIT_INTERRUPTED, EXIT_MAX_ITERS, EXIT_OK, EXIT_UNPARSEABLE};
use super::manifest::{AppliedEdit, ManifestEditor, copy_atomic};
use super::parse::{ConflictParser, RegexConflictParser, tail};
use super::repair::{
    PackOverrideKind, PackOverrideWrite, RelaxPreference, RepairPlanner, SolveLedger, Strategy,
    append_attempt, is_abi_anchor, ledger_path, manifest_sha256, mark_last_widen_failed,
    persist_conflict_trace, retread_dir, snapshot_path,
};

// Run-27 (imprint deps-from proof): a two-pack workspace with dozens of
// auto-routed exact pins legitimately needs a repair tail far deeper than
// 10 -- run 27 burned its whole budget on 10 correct repairs and exited
// EXIT_MAX_ITERS one repair short of the next layer. Repairs are cheap,
// ledgered, conservative relaxations; the budget exists to stop runaway
// loops (which the oscillation guard + TriedState already prevent
// per-package), not to be the binding constraint on a healthy ladder.
const DEFAULT_MAX_REPAIRS: u32 = 50;
const LOCK_FEATURE: &str = "default";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockArgs {
    pub manifest: PathBuf,
    pub max_repairs: u32,
    pub no_repair: bool,
}

impl Default for LockArgs {
    fn default() -> Self {
        Self {
            manifest: PathBuf::from("pixi.toml"),
            max_repairs: DEFAULT_MAX_REPAIRS,
            no_repair: false,
        }
    }
}

pub mod args {
    use super::*;

    pub fn parse(argv: &[String]) -> anyhow::Result<LockArgs> {
        let mut args = LockArgs::default();
        let mut it = argv.iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--manifest" => {
                    let p = it.next().ok_or_else(|| {
                        anyhow::anyhow!("retread lock: --manifest <path> required")
                    })?;
                    args.manifest = PathBuf::from(p);
                }
                "--max-repairs" => {
                    let raw = it.next().ok_or_else(|| {
                        anyhow::anyhow!("retread lock: --max-repairs <N> required")
                    })?;
                    args.max_repairs = raw.parse().map_err(|_| {
                        anyhow::anyhow!("retread lock: bad --max-repairs value {raw}")
                    })?;
                }
                "--no-repair" => args.no_repair = true,
                other => anyhow::bail!("retread lock: unknown arg {other}"),
            }
        }
        Ok(args)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn argv(parts: &[&str]) -> Vec<String> {
            parts.iter().map(|s| s.to_string()).collect()
        }

        #[test]
        fn defaults_are_manifest_and_fifty_repairs() {
            let args = parse(&argv(&[])).unwrap();
            assert_eq!(args.manifest, PathBuf::from("pixi.toml"));
            assert_eq!(args.max_repairs, DEFAULT_MAX_REPAIRS);
            assert_eq!(args.max_repairs, 50);
            assert!(!args.no_repair);
        }

        #[test]
        fn parses_all_flags() {
            let args = parse(&argv(&["--manifest", "x.toml", "--max-repairs", "3"])).unwrap();
            assert_eq!(args.manifest, PathBuf::from("x.toml"));
            assert_eq!(args.max_repairs, 3);

            let args = parse(&argv(&["--no-repair"])).unwrap();
            assert!(args.no_repair);
        }

        #[test]
        fn rejects_unknown_flag() {
            assert!(parse(&argv(&["--bogus"])).is_err());
        }
    }
}

pub async fn run(args: LockArgs) -> Result<i32> {
    run_with_pixi_bin(args, "pixi").await
}

/// One applied repair, kept around only to print the closing summary table.
struct RepairRecord {
    package: String,
    strategy: Strategy,
    old_spec: Option<String>,
    new_spec: String,
    feature: String,
}

struct PendingEdit {
    package: String,
    strategy: Strategy,
    edits: Vec<AppliedEdit>,
}

impl PendingEdit {
    fn from_edits(edits: Vec<AppliedEdit>) -> Option<Self> {
        let last = edits.last()?;
        Some(Self {
            package: last.package.clone(),
            strategy: last.strategy,
            edits,
        })
    }

    fn restore(&self, editor: &mut ManifestEditor) {
        for edit in self.edits.iter().rev() {
            editor.restore_entry(&edit.feature, edit.table, &edit.package, &edit.before);
        }
    }
}

async fn run_with_pixi_bin(args: LockArgs, pixi_bin: &str) -> Result<i32> {
    let manifest_path = normalize_manifest(&args.manifest);
    if !manifest_path.exists() {
        eprintln!(
            "retread lock: manifest {} not found",
            manifest_path.display()
        );
        return Ok(super::error::EXIT_USAGE);
    }

    if args.no_repair {
        return run_passthrough(pixi_bin, &manifest_path).await;
    }

    let mut editor = ManifestEditor::open(manifest_path.clone())?;
    let project_dir = editor.project_dir().to_path_buf();
    let ledger_path = ledger_path(&project_dir);
    let manifest_display = manifest_path
        .strip_prefix(&project_dir)
        .unwrap_or(&manifest_path)
        .to_string_lossy()
        .to_string();

    let parser = RegexConflictParser::new();
    // Same parselmouth-backed pypi<->conda name family + relax-preference
    // config that `retread solve` (driver.rs) wires into its planner --
    // shared via `RepairPlanner::configured` so the two drivers can't drift
    // (see aaf58c6 conda-pypi name map, 59b5b40 conda-as-truth preference).
    let conda_name_map = crate::handler::load_pypi_to_conda_map().await;
    let relax_preference = RelaxPreference::from_config_str(&editor.relax_preference());
    let mut ledger = SolveLedger::load(&ledger_path, manifest_display)?;
    let manifest_hash = manifest_sha256(&manifest_path)?;
    let mut tried = ledger.seed_tried_state(&manifest_path, &manifest_hash, &editor);
    let mut planner =
        RepairPlanner::configured(LOCK_FEATURE.to_string(), conda_name_map, relax_preference);

    let run_idx = ledger.start_run(
        "lock".to_string(),
        manifest_sha256(&manifest_path)?,
        pixi_version(&project_dir, pixi_bin).await,
    );
    ledger.write_atomic(&ledger_path)?;

    // Run-26 stale-snapshot bug: a run that dies without reaching any
    // finish path (signal kill, run 25) or exits keeping its progress
    // (max_repairs, run 27) leaves its pre-run `.bak` snapshots behind.
    // `ensure_snapshot`'s first-write-wins no-op then makes the NEXT
    // run's rollback restore to the PREVIOUS run's start state, wiping
    // repairs that run legitimately persisted (run 26 rolled the ledger
    // back to run-25-START, erasing run 25's tier-7 overrides). The
    // current on-disk state at run start IS this run's baseline -- drop
    // any stale snapshot so this run's own `ensure_snapshot` captures it.
    cleanup_snapshot(&project_dir)?;
    crate::pack_overrides::cleanup_all(&project_dir)?;

    let mut pending_edit: Option<PendingEdit> = None;
    let mut records: Vec<RepairRecord> = Vec::new();
    let mut repairs_applied: u32 = 0;
    // Defensive hard cap on raw loop iterations (widen-revert retries don't
    // consume a repair slot); bounded by the ladder depth per package so
    // this should never bite in practice.
    let hard_cap = args.max_repairs.saturating_mul(2) + 8;

    for _ in 1..=hard_cap {
        let lock = match run_pixi_lock(pixi_bin, &project_dir).await {
            Ok(lock) => lock,
            Err(err) => {
                // Unexpected crash (spawn/IO failure), not a normal solver
                // failure -- roll back any repairs applied so far.
                rollback_snapshot(&project_dir, &manifest_path)?;
                crate::pack_overrides::rollback_all(&project_dir)?;
                ledger.finish_run(run_idx, "crashed");
                ledger.write_atomic(&ledger_path)?;
                return Err(err);
            }
        };
        if lock.interrupted {
            rollback_snapshot(&project_dir, &manifest_path)?;
            crate::pack_overrides::rollback_all(&project_dir)?;
            ledger.finish_run(run_idx, "interrupted");
            ledger.write_atomic(&ledger_path)?;
            return Ok(EXIT_INTERRUPTED);
        }
        if lock.success {
            ledger.finish_run(run_idx, "converged");
            ledger.write_atomic(&ledger_path)?;
            cleanup_snapshot(&project_dir)?;
            crate::pack_overrides::cleanup_all(&project_dir)?;
            print_summary(&records);
            return Ok(EXIT_OK);
        }

        let stripped = parser.strip_ansi(&lock.stderr);
        let parsed = parser.parse(&stripped);
        persist_conflict_trace(&project_dir, "lock", repairs_applied + 1, &stripped);

        if let Some(conflict) = parsed {
            if repairs_applied >= args.max_repairs {
                ledger.finish_run(run_idx, "max_repairs");
                ledger.write_atomic(&ledger_path)?;
                eprintln!(
                    "retread lock: max repairs ({}) reached; last conflict unresolved for manifest {}",
                    args.max_repairs,
                    manifest_path.display()
                );
                eprintln!("{}", tail(&lock.raw_stderr, 4000));
                print_summary(&records);
                return Ok(EXIT_MAX_ITERS);
            }
            ensure_snapshot(&project_dir, &manifest_path)?;
            let iter = repairs_applied + 1;
            let repair = planner.repair(&mut editor, &mut tried, &conflict, iter);
            match repair {
                Ok(out) => {
                    editor.write_atomic()?;
                    if let Some(po) = &out.pack_override {
                        // Fix #22: snapshot the ledger before this run's
                        // first write to it so an exhausted run rolls it
                        // back, then append the override into `.retread/
                        // auto-overrides.json` -- NOT the pack's pixi.toml.
                        crate::pack_overrides::ensure_snapshot(&project_dir)?;
                        crate::pack_overrides::write_override(
                            &project_dir,
                            &po.pack_pixi,
                            &po.bundle,
                            &po.package,
                            &po.spec,
                            &out.attempt.conflict,
                        )?;
                    }
                    for attempt in out.extra_attempts {
                        append_attempt(&mut ledger, &ledger_path, run_idx, attempt)?;
                    }
                    append_attempt(&mut ledger, &ledger_path, run_idx, out.attempt.clone())?;
                    record_repair(
                        &mut records,
                        &out.attempt,
                        &out.applied,
                        out.pack_override.as_ref(),
                    );
                    pending_edit = PendingEdit::from_edits(out.applied);
                    repairs_applied += 1;
                    continue;
                }
                Err(package) => {
                    // Exhaustion never converges -- unlike the
                    // `max_repairs` bound (which intentionally keeps
                    // whatever progress was made so a human can pick up
                    // where the ladder left off), a dead-end tier means
                    // every repair applied *this run* is provably useless.
                    // Roll the manifest back to the pre-run snapshot so no
                    // orphaned `# retread:override`/`:pin`/`:widen` entry
                    // survives a failed run (and any `.retread/
                    // overrides.json` entry recorded this run too).
                    rollback_snapshot(&project_dir, &manifest_path)?;
                    crate::pack_overrides::rollback_all(&project_dir)?;
                    ledger.finish_run(run_idx, "exhausted");
                    ledger.write_atomic(&ledger_path)?;
                    if is_abi_anchor(&package) {
                        eprintln!(
                            "retread lock: ABI-anchor package {package} cannot be auto-repaired; edit the manifest manually"
                        );
                    } else {
                        eprintln!("retread lock: exhausted repair strategies for {package}");
                    }
                    print_summary(&records);
                    return Ok(EXIT_EXHAUSTED);
                }
            }
        }

        // Fallback: no specific parser matched this iteration's error --
        // try the generic ownership-driven engine (ends the
        // rung-per-error-shape treadmill; see `RepairPlanner::
        // generic_fallback_repair`) before declaring the conflict
        // unparseable. Bounded by the same `--max-repairs` budget as the
        // specific-conflict path -- but, unlike the old `repairs_applied <
        // args.max_repairs` short-circuit on the CALL itself, the budget
        // check happens AFTER confirming the engine actually recognizes
        // this shape as actionable (run-21 bug: a real, repairable
        // tier-7 two-pack clash landed as the 11th needed repair against
        // a 10-repair budget; short-circuiting the call entirely skipped
        // straight to "unparseable" below, which rolls EVERY prior
        // in-run repair back -- silently discarding 10 good repairs and
        // misreporting a budget problem as a parse failure). Report this
        // honestly as `max_repairs` instead, exactly like the
        // specific-conflict path above, which keeps whatever progress
        // this run made rather than wiping it.
        if let Some(result) = planner.generic_fallback_repair_batch(
            &editor,
            &mut tried,
            &stripped,
            repairs_applied + 1,
        ) {
            if repairs_applied >= args.max_repairs {
                ledger.finish_run(run_idx, "max_repairs");
                ledger.write_atomic(&ledger_path)?;
                eprintln!(
                    "retread lock: max repairs ({}) reached; last conflict unresolved (generic fallback) for manifest {}",
                    args.max_repairs,
                    manifest_path.display()
                );
                eprintln!("{}", tail(&lock.raw_stderr, 4000));
                print_summary(&records);
                return Ok(EXIT_MAX_ITERS);
            }
            match result {
                Ok(applied) => {
                    // Batched: every entry is an independent, distinct-
                    // (package, pack) ledger override derived from this
                    // SAME error -- land them all before the next (multi-
                    // minute) full re-render instead of one per iteration.
                    for (out, po) in &applied {
                        if let Some(po) = po {
                            crate::pack_overrides::ensure_snapshot(&project_dir)?;
                            match po.kind {
                                PackOverrideKind::Override => {
                                    crate::pack_overrides::write_override(
                                        &project_dir,
                                        &po.pack_pixi,
                                        &po.bundle,
                                        &po.package,
                                        &po.spec,
                                        &out.attempt.conflict,
                                    )?;
                                }
                                PackOverrideKind::Unroute => {
                                    crate::pack_overrides::write_unroute(
                                        &project_dir,
                                        &po.pack_pixi,
                                        &po.bundle,
                                        &po.package,
                                        &out.attempt.conflict,
                                    )?;
                                }
                            }
                        }
                        append_attempt(&mut ledger, &ledger_path, run_idx, out.attempt.clone())?;
                        record_repair(&mut records, &out.attempt, &out.applied, po.as_ref());
                    }
                    // The fallback engine never edits the workspace
                    // manifest directly (every candidate is a pack-ledger
                    // write) -- nothing for `pending_edit`'s post-widen
                    // revert-on-unsat recovery to track.
                    pending_edit = None;
                    repairs_applied += applied.len() as u32;
                    continue;
                }
                Err(package) => {
                    rollback_snapshot(&project_dir, &manifest_path)?;
                    crate::pack_overrides::rollback_all(&project_dir)?;
                    ledger.finish_run(run_idx, "exhausted");
                    ledger.write_atomic(&ledger_path)?;
                    eprintln!(
                        "retread lock: exhausted generic fallback repair strategies for {package}"
                    );
                    print_summary(&records);
                    return Ok(EXIT_EXHAUSTED);
                }
            }
        }

        // Unparseable this round. If the last applied edit was a conda
        // widen and this failure looks like the generic post-widen unsat
        // dump, revert just that widen and retry (same recovery `retread
        // solve` performs) rather than giving up outright.
        if let Some(pending) = pending_edit.take() {
            if pending.strategy == Strategy::WidenConda
                && RegexConflictParser::is_post_widen_conda_unsat(&stripped)
            {
                eprintln!(
                    "retread lock: post-widen conda unsat for {}; reverting only that widen",
                    pending.package
                );
                pending.restore(&mut editor);
                editor.write_atomic()?;
                tried.mark(&pending.package, Strategy::WidenConda, true);
                mark_last_widen_failed(&mut ledger, &ledger_path, run_idx, &pending.package)?;
                records.retain(|r| {
                    !(r.package == pending.package && r.strategy == Strategy::WidenConda)
                });
                continue;
            }
            pending.restore(&mut editor);
            editor.write_atomic()?;
        }

        // An unparseable conflict never converges either -- like
        // exhaustion, every repair applied *this run* (workspace edits AND
        // any pack-table `retread-overrides` writes) is provably useless,
        // since there's no way to plan a next repair off an error we
        // can't read. Roll everything back to the pre-run snapshot rather
        // than leaving an orphaned pack override behind (see fix #21:
        // iter-1's `torch = "==2.10.0"  # retread:override` survived a
        // later unparseable iter-2 until this was added).
        rollback_snapshot(&project_dir, &manifest_path)?;
        crate::pack_overrides::rollback_all(&project_dir)?;
        ledger.finish_run(run_idx, "unparseable");
        ledger.write_atomic(&ledger_path)?;
        eprintln!("retread lock: could not parse solver error; manifest kept as-is");
        // Run-38 shape recognition: a workspace exact pin reduced to a
        // wrong-python build tail carries NO repairable chain in the tree
        // itself (resolvo shows only the leftover cpXX builds' python_abi
        // dead end), but the shape is diagnosable -- name the pin and
        // point at the real cause class so the failure isn't a bare
        // "could not parse".
        if let Some(diag) = super::parse::diagnose_abi_build_tail(&stripped) {
            eprintln!("{diag}");
        }
        eprintln!("{}", tail(&lock.raw_stderr, 4000));
        print_summary(&records);
        return Ok(EXIT_UNPARSEABLE);
    }

    ledger.finish_run(run_idx, "max_iters");
    ledger.write_atomic(&ledger_path)?;
    eprintln!("retread lock: internal loop bound reached without converging");
    print_summary(&records);
    Ok(EXIT_MAX_ITERS)
}

fn record_repair(
    records: &mut Vec<RepairRecord>,
    attempt: &super::repair::LedgerAttempt,
    applied: &[AppliedEdit],
    pack_override: Option<&PackOverrideWrite>,
) {
    let old_spec = applied
        .last()
        .and_then(|e| e.before.value.clone())
        .or_else(|| attempt.old_spec.clone());
    let new_spec = attempt
        .new_spec
        .clone()
        .or_else(|| attempt.version.as_ref().map(|v| format!("=={v}")))
        .unwrap_or_default();
    // Label with the feature the edit actually landed in (may differ from
    // the planner's ambient `LOCK_FEATURE` when the repair re-attributed
    // to a pin owner in another feature/pack table -- see fix #16/#19),
    // not a hardcoded constant (fix #19: the repair table previously
    // always printed `default` here even when the edit landed elsewhere).
    // Fix #20: a `pack_override` repair never touches the workspace
    // manifest (`applied` is empty), so label it with the bundle's
    // pack-manifest scope instead of falling back to `LOCK_FEATURE`.
    let feature = applied
        .last()
        .map(|e| e.feature.clone())
        .or_else(|| pack_override.map(|po| format!("{} (pack)", po.bundle)))
        .unwrap_or_else(|| LOCK_FEATURE.to_string());
    records.push(RepairRecord {
        package: attempt.package.clone(),
        strategy: parse_strategy(&attempt.strategy),
        old_spec,
        new_spec,
        feature,
    });
}

fn parse_strategy(raw: &str) -> Strategy {
    match raw {
        "widen-conda" => Strategy::WidenConda,
        "pypi_dep" => Strategy::PypiDep,
        "pypi_override" => Strategy::PypiOverride,
        _ => Strategy::Conda,
    }
}

async fn run_passthrough(pixi_bin: &str, manifest_path: &Path) -> Result<i32> {
    let project_dir = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let mut cmd = Command::new(pixi_bin);
    cmd.arg("lock").current_dir(&project_dir);
    passthrough_env(&mut cmd);
    let status = cmd
        .status()
        .await
        .map_err(|e| anyhow::anyhow!("retread lock: failed to spawn {pixi_bin} lock: {e}"))?;
    Ok(status.code().unwrap_or(1))
}

struct LockResult {
    success: bool,
    interrupted: bool,
    stderr: String,
    raw_stderr: String,
}

fn passthrough_env(cmd: &mut Command) {
    // `PIXI_BUILD_BACKEND_OVERRIDE` is already inherited by child processes
    // (we never call env_clear); set it explicitly so the passthrough is
    // not silently dependent on that default.
    if let Ok(val) = std::env::var("PIXI_BUILD_BACKEND_OVERRIDE") {
        cmd.env("PIXI_BUILD_BACKEND_OVERRIDE", val);
    }
}

async fn run_pixi_lock(pixi_bin: &str, project_dir: &Path) -> Result<LockResult> {
    let mut cmd = Command::new(pixi_bin);
    cmd.arg("lock")
        .arg("--color=never")
        .current_dir(project_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("PIXI_COLOR", "never");
    passthrough_env(&mut cmd);

    let mut child = cmd
        .spawn()
        .map_err(|e| anyhow::anyhow!("retread lock: failed to spawn {pixi_bin} lock: {e}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        if let Some(mut pipe) = stdout {
            let _ = pipe.read_to_end(&mut buf).await;
        }
        buf
    });
    let stderr_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        if let Some(mut pipe) = stderr {
            let _ = pipe.read_to_end(&mut buf).await;
        }
        buf
    });

    tokio::select! {
        status = child.wait() => {
            let status = status.map_err(|e| anyhow::anyhow!("retread lock: pixi lock failed: {e}"))?;
            let _stdout = stdout_task.await.unwrap_or_default();
            let stderr = stderr_task.await.unwrap_or_default();
            let raw_stderr = String::from_utf8_lossy(&stderr).to_string();
            Ok(LockResult {
                success: status.success(),
                interrupted: false,
                stderr: raw_stderr.clone(),
                raw_stderr,
            })
        }
        _ = tokio::signal::ctrl_c() => {
            let _ = child.kill().await;
            let _ = stdout_task.await;
            let _ = stderr_task.await;
            Ok(LockResult {
                success: false,
                interrupted: true,
                stderr: String::new(),
                raw_stderr: String::new(),
            })
        }
    }
}

async fn pixi_version(project_dir: &Path, pixi_bin: &str) -> Option<String> {
    let out = Command::new(pixi_bin)
        .arg("--version")
        .current_dir(project_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await
        .ok()?;
    if out.status.success() {
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        None
    }
}

fn normalize_manifest(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn ensure_snapshot(project_dir: &Path, manifest_path: &Path) -> Result<()> {
    let retread = retread_dir(project_dir);
    std::fs::create_dir_all(&retread)?;
    let bak = snapshot_path(project_dir);
    if !bak.exists() {
        copy_atomic(manifest_path, &bak)?;
    }
    Ok(())
}

fn rollback_snapshot(project_dir: &Path, manifest_path: &Path) -> Result<()> {
    let bak = snapshot_path(project_dir);
    if bak.exists() {
        copy_atomic(&bak, manifest_path)?;
        let _ = std::fs::remove_file(&bak);
    }
    Ok(())
}

fn cleanup_snapshot(project_dir: &Path) -> Result<()> {
    let bak = snapshot_path(project_dir);
    if bak.exists() {
        std::fs::remove_file(bak)?;
    }
    Ok(())
}

fn print_summary(records: &[RepairRecord]) {
    if records.is_empty() {
        println!("retread lock: no manifest repairs applied");
        return;
    }
    println!("retread lock: repairs applied");
    let header = format!(
        "{:<24} {:<20} {:<24} {:<10} {}",
        "package", "old_spec", "new_spec", "tier", "feature"
    );
    println!("{header}");
    for r in records {
        println!(
            "{:<24} {:<20} {:<24} {:<10} {}",
            r.package,
            r.old_spec.as_deref().unwrap_or("(none)"),
            r.new_spec,
            r.strategy.as_str(),
            r.feature
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "retread-lock-test-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Writes a fake `pixi` executable that fails (dumping `stderr_fixture`)
    /// for the first `fail_count` invocations of `lock`, then succeeds.
    /// `pixi --version` and any other subcommand always succeeds trivially.
    /// State is a plain counter file scoped to this test's own temp dir, so
    /// parallel tests never share process-wide env/PATH state.
    fn write_fake_pixi(dir: &Path, fail_count: u32, stderr_fixture: &str) -> PathBuf {
        let counter = dir.join("counter");
        std::fs::write(&counter, b"0").unwrap();
        let stderr_path = dir.join("stderr.txt");
        std::fs::write(&stderr_path, stderr_fixture).unwrap();
        let script_path = dir.join("pixi");
        let script = format!(
            r#"#!/bin/bash
if [ "$1" != "lock" ]; then
  exit 0
fi
n=$(cat "{counter}")
n=$((n + 1))
echo "$n" > "{counter}"
if [ "$n" -le {fail_count} ]; then
  cat "{stderr_path}" >&2
  exit 1
fi
exit 0
"#,
            counter = counter.display(),
            fail_count = fail_count,
            stderr_path = stderr_path.display(),
        );
        let mut f = std::fs::File::create(&script_path).unwrap();
        f.write_all(script.as_bytes()).unwrap();
        let mut perm = f.metadata().unwrap().permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(&script_path, perm).unwrap();
        script_path
    }

    /// Like `write_fake_pixi`, but plays a DIFFERENT stderr fixture on each
    /// successive `lock` invocation (clamped to the last entry once the
    /// list is exhausted) and always fails. Lets a test drive a specific
    /// multi-iteration sequence -- e.g. iter-1 a parseable pack-override
    /// conflict, iter-2 unparseable garbage -- deterministically.
    fn write_fake_pixi_sequence(dir: &Path, fixtures: &[&str]) -> PathBuf {
        let counter = dir.join("counter");
        std::fs::write(&counter, b"0").unwrap();
        let fixtures_dir = dir.join("seq-fixtures");
        std::fs::create_dir_all(&fixtures_dir).unwrap();
        for (i, fx) in fixtures.iter().enumerate() {
            std::fs::write(fixtures_dir.join(format!("{i}.txt")), fx).unwrap();
        }
        let last = fixtures.len().saturating_sub(1);
        let script_path = dir.join("pixi");
        let script = format!(
            r#"#!/bin/bash
if [ "$1" != "lock" ]; then
  exit 0
fi
n=$(cat "{counter}")
n=$((n + 1))
echo "$n" > "{counter}"
idx=$((n - 1))
if [ "$idx" -gt {last} ]; then
  idx={last}
fi
cat "{fixtures_dir}/$idx.txt" >&2
exit 1
"#,
            counter = counter.display(),
            last = last,
            fixtures_dir = fixtures_dir.display(),
        );
        let mut f = std::fs::File::create(&script_path).unwrap();
        f.write_all(script.as_bytes()).unwrap();
        let mut perm = f.metadata().unwrap().permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(&script_path, perm).unwrap();
        script_path
    }

    fn write_manifest(dir: &Path, text: &str) -> PathBuf {
        let path = dir.join("pixi.toml");
        std::fs::write(&path, text).unwrap();
        path
    }

    /// The lock-path regression this module previously shipped: `retread
    /// lock` constructed `RepairPlanner::new(LOCK_FEATURE)` bare, so the
    /// parselmouth name map (aaf58c6) and relax-preference (59b5b40) were
    /// silently inert in the lock path even though `retread solve` had
    /// them. This exercises the exact planner construction
    /// `run_with_pixi_bin` now performs (via `RepairPlanner::configured`
    /// with `LOCK_FEATURE`) against the acceptance-conflict fixture: pypi
    /// `torch>=2.11.0` vs a conda pin declared under the DIFFERENT name
    /// `pytorch-gpu ==2.10.0`. With the name map wired, conda-as-truth T1
    /// must fire -- pypi override for `torch`, `pytorch-gpu` untouched --
    /// instead of falling through to widening a bogus `torch` conda entry.
    #[tokio::test]
    async fn lock_planner_resolves_torch_to_pytorch_gpu_via_name_map() {
        use super::super::parse::Conflict;
        use super::super::repair::{RelaxPreference, TriedState};

        // Uses the PRODUCTION map-loading path (`crate::handler::
        // load_pypi_to_conda_map`, the exact call `run_with_pixi_bin` makes
        // above), not a hand-inserted map -- a prior version of this test
        // built its own `torch -> [pytorch, pytorch-cpu, pytorch-gpu]` map,
        // which masked a real data gap: FALLBACK_PYPI_TO_CONDA didn't carry
        // `torch -> pytorch-gpu` (parselmouth's `pytorch-gpu` entry is a
        // meta-package with no pypi names), so this test passed while the
        // live tool failed identically 4 times (lock-succ-brief.md
        // ACCEPTANCE RUNs #1-#4). The parselmouth network fetch may fail in
        // this sandbox -- fine, the FALLBACK merge runs unconditionally.
        let dir = temp_dir("namemap");
        let path = write_manifest(&dir, "[dependencies]\npytorch-gpu = \"==2.10.0\"\n");
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
        // Same construction as run_with_pixi_bin.
        let relax_preference = RelaxPreference::from_config_str(&editor.relax_preference());
        let mut planner =
            RepairPlanner::configured(LOCK_FEATURE.to_string(), name_map, relax_preference);

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

        editor.write_atomic().unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("pytorch-gpu = \"==2.10.0\"\n")); // conda pin untouched
        assert!(text.contains("[pypi-options.dependency-overrides]"));
        assert!(text.contains("torch = \"==2.10.0\"  # retread:override"));
        assert_eq!(text.matches("torch = ").count(), 1);
    }

    const CONDA_BOUNDARY_SINGLE_LINE: &str =
        include_str!("../../tests/fixtures/solve_errors/conda_boundary_single_line.txt");
    const UNPARSEABLE: &str =
        include_str!("../../tests/fixtures/solve_errors/unparseable_network_error.txt");

    #[tokio::test]
    async fn converges_after_one_repair() {
        let dir = temp_dir("converge");
        let manifest = write_manifest(&dir, "[dependencies]\n");
        let pixi = write_fake_pixi(&dir, 1, CONDA_BOUNDARY_SINGLE_LINE);

        let args = LockArgs {
            manifest,
            max_repairs: 10,
            no_repair: false,
        };
        let code = run_with_pixi_bin(args, pixi.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(code, EXIT_OK);

        let manifest_text = std::fs::read_to_string(dir.join("pixi.toml")).unwrap();
        assert!(manifest_text.contains("numpy"));
        // Snapshot must be cleaned up on convergence.
        assert!(!snapshot_path(&dir).exists());
    }

    #[tokio::test]
    async fn max_repairs_bound_stops_with_manifest_kept() {
        let dir = temp_dir("maxrepairs");
        let manifest = write_manifest(&dir, "[dependencies]\n");
        // numpy's ladder is conda -> pypi_dep -> pypi_override (3 repairs)
        // before exhausting; cap at 1 repair so we hit the bound first.
        let pixi = write_fake_pixi(&dir, 50, CONDA_BOUNDARY_SINGLE_LINE);

        let args = LockArgs {
            manifest,
            max_repairs: 1,
            no_repair: false,
        };
        let code = run_with_pixi_bin(args, pixi.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(code, EXIT_MAX_ITERS);

        // One repair was kept (bounded exhaustion keeps applied repairs).
        let manifest_text = std::fs::read_to_string(dir.join("pixi.toml")).unwrap();
        assert!(manifest_text.contains("numpy"));
    }

    #[tokio::test]
    async fn exhaustion_rolls_back_to_pre_run_manifest_snapshot() {
        // P0 regression (lock-succ-brief.md acceptance runs #1/#2): both
        // real runs left a dead `# retread:override` entry in pixi.toml
        // after the ladder exhausted -- repairs that never converged must
        // not survive the run. numpy's ladder is conda -> pypi_dep ->
        // pypi_override (3 successful repairs, each converging to the SAME
        // unresolved conflict since the fake pixi always fails) then
        // exhausts on the 4th attempt (override tier already tried).
        let dir = temp_dir("exhaustrollback");
        let original = "[dependencies]\n";
        let manifest = write_manifest(&dir, original);
        let pixi = write_fake_pixi(&dir, 50, CONDA_BOUNDARY_SINGLE_LINE);

        let args = LockArgs {
            manifest,
            max_repairs: 10,
            no_repair: false,
        };
        let code = run_with_pixi_bin(args, pixi.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(code, EXIT_EXHAUSTED);

        // Manifest byte-identical to the pre-run snapshot -- every
        // intermediate repair (conda pin, then pypi dep, then pypi
        // override) was rolled back, not just the last one.
        let manifest_text = std::fs::read_to_string(dir.join("pixi.toml")).unwrap();
        assert_eq!(manifest_text, original);
        assert!(!manifest_text.contains("numpy"));
        assert!(!snapshot_path(&dir).exists());
    }

    #[tokio::test]
    async fn abi_anchor_conflict_stops_without_repair() {
        let dir = temp_dir("abianchor");
        let manifest = write_manifest(&dir, "[dependencies]\n");
        let fixture = CONDA_BOUNDARY_SINGLE_LINE.replace("numpy", "python");
        let pixi = write_fake_pixi(&dir, 50, &fixture);

        let args = LockArgs {
            manifest,
            max_repairs: 10,
            no_repair: false,
        };
        let code = run_with_pixi_bin(args, pixi.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(code, EXIT_EXHAUSTED);

        let manifest_text = std::fs::read_to_string(dir.join("pixi.toml")).unwrap();
        assert!(!manifest_text.contains("python"));
    }

    #[tokio::test]
    async fn unparseable_conflict_stops_and_keeps_manifest_untouched() {
        let dir = temp_dir("unparseable");
        let manifest = write_manifest(&dir, "[dependencies]\n");
        let pixi = write_fake_pixi(&dir, 50, UNPARSEABLE);

        let args = LockArgs {
            manifest,
            max_repairs: 10,
            no_repair: false,
        };
        let code = run_with_pixi_bin(args, pixi.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(code, EXIT_UNPARSEABLE);

        let manifest_text = std::fs::read_to_string(dir.join("pixi.toml")).unwrap();
        assert_eq!(manifest_text, "[dependencies]\n");
    }

    const SENTRY_SDK_TWO_PACK: &str =
        include_str!("../../tests/fixtures/solve_errors/conda_two_pack_sentry_sdk.txt");

    /// Run-21 regression: a conflict shape ONLY the generic fallback
    /// engine recognizes (no specific `Conflict` parser matches tier-7's
    /// two-pack same-package clash) that surfaces exactly when the
    /// `--max-repairs` budget is already spent must be reported as
    /// `max_repairs` (budget exhausted, progress kept) -- not silently
    /// short-circuited into the "unparseable" catch-all, which rolls
    /// EVERY repair this run made back to the pre-run snapshot and prints
    /// a misleading "could not parse solver error" even though the
    /// engine, given one more repair slot, would have floored the
    /// dead-end pin and converged (verified directly against this same
    /// fixture in `solve::repair::tests::
    /// end_to_end_run18_two_pack_sentry_sdk_floors_the_dead_end_exact_pin`).
    #[tokio::test]
    async fn generic_only_conflict_at_budget_reports_max_repairs_not_unparseable() {
        let dir = temp_dir("genericbudget");
        let manifest_text = "[dependencies]\n\n\
             [feature.pm-isaaclab.dependencies]\n\
             \"protomotions-deps-pack\" = { path = \"./pypi-packs/protomotions-deps-pack\" }\n\
             \"isaaclab-2.3x-pack\" = { path = \"./pypi-packs/isaaclab-2.3x-pack\" }\n\n\
             [environments]\npm-isaaclab = { features = [\"pm-isaaclab\"] }\n";
        let manifest = write_manifest(&dir, manifest_text);
        for (name, version) in [
            ("protomotions-deps-pack", "3.1"),
            ("isaaclab-2.3x-pack", "0.54.2"),
        ] {
            let pack_dir = dir.join(format!("pypi-packs/{name}"));
            std::fs::create_dir_all(&pack_dir).unwrap();
            std::fs::write(
                pack_dir.join("pixi.toml"),
                format!("[package]\nname = \"{name}\"\nversion = \"{version}\"\n"),
            )
            .unwrap();
        }
        let pixi = write_fake_pixi(&dir, 50, SENTRY_SDK_TWO_PACK);

        // Budget already exhausted (0) before this conflict is even seen
        // -- mirrors run 21, where 10 unrelated repairs had already
        // consumed the whole budget before the sentry-sdk clash was the
        // sole remaining error.
        let args = LockArgs {
            manifest,
            max_repairs: 0,
            no_repair: false,
        };
        let code = run_with_pixi_bin(args, pixi.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(
            code, EXIT_MAX_ITERS,
            "an actionable-but-budget-starved generic-fallback conflict must report \
             max_repairs, not unparseable"
        );

        // Unlike "unparseable" (blanket rollback), "max_repairs" keeps
        // whatever the run accomplished -- here, nothing was applied (the
        // budget was 0 from the start), so the manifest is simply
        // untouched, but critically via the KEEP path, not the ROLLBACK
        // path (snapshot must be gone either way once the run finishes,
        // but the outcome/exit code is the load-bearing assertion above).
        let manifest_after = std::fs::read_to_string(dir.join("pixi.toml")).unwrap();
        assert_eq!(manifest_after, manifest_text);
    }

    /// Same shape, but ONE specific-conflict repair already consumed the
    /// only budget slot before the generic-only sentry-sdk conflict shows
    /// up on the next iteration -- the prior repair must survive (kept,
    /// not rolled back) exactly like the existing specific-conflict
    /// `max_repairs_bound_stops_with_manifest_kept` case.
    #[tokio::test]
    async fn generic_only_conflict_after_budget_spent_keeps_prior_repair() {
        let dir = temp_dir("genericbudgetprior");
        let manifest_text = "[dependencies]\n\n\
             [feature.pm-isaaclab.dependencies]\n\
             \"protomotions-deps-pack\" = { path = \"./pypi-packs/protomotions-deps-pack\" }\n\
             \"isaaclab-2.3x-pack\" = { path = \"./pypi-packs/isaaclab-2.3x-pack\" }\n\n\
             [environments]\npm-isaaclab = { features = [\"pm-isaaclab\"] }\n";
        let manifest = write_manifest(&dir, manifest_text);
        for (name, version) in [
            ("protomotions-deps-pack", "3.1"),
            ("isaaclab-2.3x-pack", "0.54.2"),
        ] {
            let pack_dir = dir.join(format!("pypi-packs/{name}"));
            std::fs::create_dir_all(&pack_dir).unwrap();
            std::fs::write(
                pack_dir.join("pixi.toml"),
                format!("[package]\nname = \"{name}\"\nversion = \"{version}\"\n"),
            )
            .unwrap();
        }
        // iter 1: a plain specific-conflict repair (numpy conda boundary)
        // that consumes the single repair slot; iter 2+: the generic-only
        // sentry-sdk two-pack clash, clamped for every call after.
        let pixi =
            write_fake_pixi_sequence(&dir, &[CONDA_BOUNDARY_SINGLE_LINE, SENTRY_SDK_TWO_PACK]);

        let args = LockArgs {
            manifest,
            max_repairs: 1,
            no_repair: false,
        };
        let code = run_with_pixi_bin(args, pixi.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(code, EXIT_MAX_ITERS);

        // iter 1's numpy repair must have survived -- the generic-only
        // conflict on iter 2 hitting the exhausted budget must NOT roll
        // it back (the run-21 bug: it used to fall through to the
        // "unparseable" path, which rolls back every repair the run
        // made).
        let manifest_after = std::fs::read_to_string(dir.join("pixi.toml")).unwrap();
        assert!(
            manifest_after.contains("numpy"),
            "iter 1's repair must be kept, not rolled back: {manifest_after}"
        );
    }

    /// Run-26 stale-snapshot regression: a `.bak` snapshot left behind by
    /// a PREVIOUS run (signal kill / max_repairs exit never cleaned it)
    /// must not become a later run's rollback baseline -- pre-fix,
    /// `rollback_snapshot` on an unparseable failure would restore the
    /// STALE bak over the manifest, silently reverting state a previous
    /// run legitimately persisted (run 26 wiped run 25's tier-7 ledger
    /// overrides this way). Run-start hygiene must drop stale snapshots
    /// so the current on-disk state is this run's baseline.
    #[tokio::test]
    async fn stale_snapshot_from_a_previous_run_is_not_this_runs_rollback_baseline() {
        let dir = temp_dir("stalesnap");
        let current = "[dependencies]\n# current, post-previous-run state\n";
        let manifest = write_manifest(&dir, current);
        // Simulate the leftover: a stale pre-run snapshot from an older run
        // whose content differs from the current manifest.
        std::fs::create_dir_all(dir.join(".retread")).unwrap();
        std::fs::write(
            snapshot_path(&dir),
            "[dependencies]\n# STALE state from a killed previous run\n",
        )
        .unwrap();
        let pixi = write_fake_pixi(&dir, 50, UNPARSEABLE);

        let args = LockArgs {
            manifest,
            max_repairs: 10,
            no_repair: false,
        };
        let code = run_with_pixi_bin(args, pixi.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(code, EXIT_UNPARSEABLE);

        let manifest_text = std::fs::read_to_string(dir.join("pixi.toml")).unwrap();
        assert_eq!(
            manifest_text, current,
            "the stale snapshot must not clobber the current manifest on rollback"
        );
    }

    #[tokio::test]
    async fn no_repair_is_plain_passthrough() {
        let dir = temp_dir("norepair");
        let manifest = write_manifest(&dir, "[dependencies]\n");
        // Fails once; --no-repair must not touch the manifest or retry.
        let pixi = write_fake_pixi(&dir, 1, CONDA_BOUNDARY_SINGLE_LINE);

        let args = LockArgs {
            manifest,
            max_repairs: 10,
            no_repair: true,
        };
        let code = run_with_pixi_bin(args, pixi.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(code, 1);
        let manifest_text = std::fs::read_to_string(dir.join("pixi.toml")).unwrap();
        assert_eq!(manifest_text, "[dependencies]\n");

        // Second run (fresh counter isn't reset, but the fake script's
        // fail_count was already exhausted) succeeds -> passthrough exit 0.
        let args2 = LockArgs {
            manifest: dir.join("pixi.toml"),
            max_repairs: 10,
            no_repair: true,
        };
        let code2 = run_with_pixi_bin(args2, pixi.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(code2, 0);
    }

    // ---- Fix #19 (lock-succ-brief.md ACCEPTANCE RUN #5, feature-column
    // finding): `record_repair` previously hardcoded every repair-table row
    // to `LOCK_FEATURE` ("default"), even when the edit landed in a named
    // feature (e.g. `feature.gpu`) via the conda-pin-owner re-attribution
    // in `conda_widen_needed`. Placement was already correct (fix #16);
    // only the printed LABEL lied. This exercises `record_repair` directly
    // against an `AppliedEdit` carrying a non-default feature and checks
    // the resulting `RepairRecord.feature` reflects it, not `LOCK_FEATURE`.
    #[test]
    fn record_repair_labels_feature_the_edit_actually_landed_in() {
        use super::super::manifest::{EntrySnapshot, TableKind};
        use super::super::repair::{LedgerAttempt, Strategy};

        let applied = vec![AppliedEdit {
            package: "torchvision".to_string(),
            strategy: Strategy::PypiOverride,
            table: TableKind::Override,
            before: EntrySnapshot {
                value: None,
                suffix: None,
            },
            feature: "feature.gpu".to_string(),
        }];
        let attempt = LedgerAttempt {
            iter: 2,
            package: "torchvision".to_string(),
            version: Some("0.25.0".to_string()),
            tier: "pypi_override".to_string(),
            strategy: "pypi_override".to_string(),
            conflict: "CondaWidenNeeded".to_string(),
            source: "retread".to_string(),
            ts: "2026-07-08T00:00:00Z".to_string(),
            old_spec: None,
            new_spec: Some("==0.25.0".to_string()),
            ceiling_policy: None,
            before: None,
            failed: false,
        };
        let mut records = Vec::new();
        record_repair(&mut records, &attempt, &applied, None);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].feature, "feature.gpu");
        assert_ne!(records[0].feature, LOCK_FEATURE);
    }

    // ---- Fix #19 (lock-succ-brief.md item 2, conflict-text capture):
    // acceptance run #5 left triage blind after exhaustion -- only the
    // terse repair table survived, no raw per-iteration solver text. Each
    // iteration's flattened conflict text must now be persisted under
    // `.retread/solve-conflicts/<run>-<iter>.txt` so a future session can
    // see exactly what the NEXT iteration's uv report named.
    #[tokio::test]
    async fn persists_raw_conflict_text_per_iteration() {
        let dir = temp_dir("conflicttrace");
        let manifest = write_manifest(&dir, "[dependencies]\n");
        let pixi = write_fake_pixi(&dir, 50, CONDA_BOUNDARY_SINGLE_LINE);

        let args = LockArgs {
            manifest,
            max_repairs: 1,
            no_repair: false,
        };
        let code = run_with_pixi_bin(args, pixi.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(code, EXIT_MAX_ITERS);

        let trace = dir
            .join(".retread")
            .join("solve-conflicts")
            .join("lock-1.txt");
        assert!(
            trace.exists(),
            "expected conflict trace at {}",
            trace.display()
        );
        let text = std::fs::read_to_string(&trace).unwrap();
        assert!(text.contains("numpy"));
    }

    // ---- Fix #21 (exactpin-fix-brief.md / acceptance-final.md secondary
    // defect): iter-1 wrote a PACK override (`torch = "==2.10.0"
    // # retread:override`) into the pack's own manifest; iter-2 then hits a
    // conflict the parser can't read (unparseable). Before this fix,
    // `run_with_pixi_bin`'s `unparseable` exit only restored the (empty, in
    // this case) `pending_edit` -- it never called `pack_overrides::
    // rollback_all`, so the pack's manifest kept the iter-1 override even
    // though the run never converged. Live capture: acceptance-final.md's
    // "Secondary defect (minor)" note.
    const PACK_EXACT_PIN_TORCH_CONFLICT: &str = concat!(
        "Error: failed to solve requirements of environment 'isaaclab-gpu-latest'\n",
        "Cannot solve the request because of: torch ==2.10.0 cannot be\n",
        "installed because there are no viable options:\n",
        "  torch 2.10.0 would require\n",
        "     python_abi 3.13.*, for which no candidates were found.\n",
        "The following packages are incompatible\n",
        "isaac-pack-latest * can be installed with any of the following options:\n",
        "   isaac-pack-latest 6.1.11 would require\n",
        "      torch ==2.11.0, which can be installed with any of the following options:\n",
        "         torch 2.11.0\n",
    );
    const GARBAGE_UNPARSEABLE: &str = "some totally unrecognized solver failure text\n";

    #[tokio::test]
    async fn unparseable_after_pack_override_rolls_back_pack_manifest_too() {
        let dir = temp_dir("unparseable-pack-rollback");
        let manifest_text = concat!(
            "[dependencies]\n\n",
            "[feature.gpu.dependencies]\n",
            "torch = \"==2.10.0\"  # retread:pin\n\n",
            "[feature.isaaclab-latest.dependencies]\n",
            "isaac-pack-latest = { path = \"./pypi-packs/isaac-pack-latest\" }\n\n",
            "[environments]\n",
            "isaaclab-gpu-latest = { features = [\"gpu\", \"isaaclab-latest\"] }\n",
        );
        let manifest = write_manifest(&dir, manifest_text);
        let pack_dir = dir.join("pypi-packs/isaac-pack-latest");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_pixi = pack_dir.join("pixi.toml");
        let pack_text = "[package]\nname = \"isaac-pack-latest\"\nversion = \"6.1.11\"\n";
        std::fs::write(&pack_pixi, pack_text).unwrap();

        let pixi =
            write_fake_pixi_sequence(&dir, &[PACK_EXACT_PIN_TORCH_CONFLICT, GARBAGE_UNPARSEABLE]);

        let args = LockArgs {
            manifest,
            max_repairs: 10,
            no_repair: false,
        };
        let code = run_with_pixi_bin(args, pixi.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(code, EXIT_UNPARSEABLE);

        // Both the workspace manifest AND the pack's own manifest (where
        // iter-1's pack-override write landed) must be byte-identical to
        // their pre-run state -- no orphaned override survives a run that
        // never converged.
        assert_eq!(
            std::fs::read_to_string(dir.join("pixi.toml")).unwrap(),
            manifest_text
        );
        assert_eq!(std::fs::read_to_string(&pack_pixi).unwrap(), pack_text);
        assert!(!snapshot_path(&dir).exists());
        // Fix #22: iter-1's pack override landed in the ledger, not the
        // pack manifest -- confirm the ledger itself was also rolled back
        // (it didn't exist before this run, so it must not exist after).
        assert!(
            !crate::pack_overrides::ledger_path(&dir).exists(),
            "the auto-overrides ledger must be rolled back on an unparseable run too"
        );
    }
}
