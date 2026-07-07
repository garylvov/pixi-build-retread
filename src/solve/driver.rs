use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::Result;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use crate::status;
use crate::workspace::WorkspaceManifest;

use super::args::SolveArgs;
use super::error::{EXIT_OK, EXIT_SMOKE_FAILED, SolveError};
use super::manifest::{AppliedEdit, ManifestEditor, copy_atomic, restore_bytes_atomic};
use super::parse::{Conflict, ConflictParser, RegexConflictParser, tail};
use super::repair::{
    LedgerAttempt, RepairPlanner, SolveLedger, Strategy, TriedState, append_attempt, ledger_path,
    manifest_sha256, mark_last_widen_failed, retread_dir, snapshot_path, truncate_ledger_runs,
};

pub async fn run(args: SolveArgs) -> Result<i32> {
    let manifest_path = normalize_manifest(&args.manifest);
    if !manifest_path.exists() {
        let err = SolveError::Usage(format!(
            "retread solve: manifest {} not found",
            manifest_path.display()
        ));
        eprintln!("{err}");
        return Ok(err.exit_code());
    }

    let mut editor = ManifestEditor::open(manifest_path.clone())?;
    let project_dir = editor.project_dir().to_path_buf();
    let ledger_path = ledger_path(&project_dir);
    let manifest_display = manifest_path
        .strip_prefix(&project_dir)
        .unwrap_or(&manifest_path)
        .to_string_lossy()
        .to_string();

    if args.clean_pins {
        ensure_snapshot(&project_dir, &manifest_path)?;
        let removed = editor.clean_pins();
        editor.write_atomic()?;
        truncate_ledger_runs(&ledger_path, manifest_display)?;
        cleanup_snapshot(&project_dir)?;
        status::tty(&format!("retread solve: removed {removed} retread pins"));
        println!("retread solve: removed {removed} retread pins");
        return Ok(EXIT_OK);
    }

    let pixi_version = pixi_version(&project_dir).await;
    warn_if_uncalibrated(pixi_version.as_deref());
    let parser = RegexConflictParser::new();
    let mut ledger = SolveLedger::load(&ledger_path, manifest_display)?;
    let manifest_hash = manifest_sha256(&manifest_path)?;
    let mut tried = ledger.seed_tried_state(&manifest_path, &manifest_hash, &editor);
    let envs = if args.environments.is_empty() {
        vec![None]
    } else {
        args.environments.iter().map(|e| Some(e.clone())).collect()
    };

    let mut any_failed_code = EXIT_OK;
    for env in envs {
        let feature = resolve_feature(&manifest_path, env.as_deref(), args.feature.as_deref());
        let env_checkpoint = if args.keep_going {
            Some(std::fs::read(&manifest_path)?)
        } else {
            None
        };
        let run_idx = if args.dry_run {
            None
        } else {
            let idx = ledger.start_run(
                env.clone().unwrap_or_else(|| "default".into()),
                manifest_sha256(&manifest_path)?,
                pixi_version.clone(),
            );
            ledger.write_atomic(&ledger_path)?;
            Some(idx)
        };
        let result = run_env(
            &args,
            &project_dir,
            &manifest_path,
            &mut editor,
            &parser,
            &mut ledger,
            &ledger_path,
            run_idx,
            &mut tried,
            feature,
            env.as_deref(),
        )
        .await;
        match result {
            Ok(code) if code == EXIT_OK => {
                if let Some(run_idx) = run_idx {
                    ledger.finish_run(run_idx, "converged");
                    ledger.write_atomic(&ledger_path)?;
                }
            }
            Ok(code) if code == EXIT_SMOKE_FAILED => {
                if let Some(run_idx) = run_idx {
                    if let Some(run) = ledger.runs.get_mut(run_idx) {
                        run.outcome = "converged".into();
                        run.smoke_failed = true;
                    }
                    ledger.write_atomic(&ledger_path)?;
                }
                cleanup_snapshot(&project_dir)?;
                return Ok(code);
            }
            Ok(code) => {
                any_failed_code = code;
            }
            Err(err) => {
                let code = err.exit_code();
                if args.keep_going && code != 130 {
                    if let Some(bytes) = env_checkpoint {
                        restore_bytes_atomic(&manifest_path, &bytes)?;
                        editor.reload()?;
                    }
                    if let Some(run_idx) = run_idx {
                        ledger.finish_run(run_idx, outcome_for_code(code));
                        ledger.write_atomic(&ledger_path)?;
                    }
                    any_failed_code = code;
                    continue;
                }
                match code {
                    2 => {
                        if let Some(run_idx) = run_idx {
                            ledger.finish_run(run_idx, "unparseable");
                            ledger.write_atomic(&ledger_path)?;
                        }
                    }
                    3 => {
                        rollback_snapshot(&project_dir, &manifest_path)?;
                        if let Some(run_idx) = run_idx {
                            ledger.finish_run(run_idx, "exhausted");
                            ledger.write_atomic(&ledger_path)?;
                        }
                    }
                    4 => {
                        rollback_snapshot(&project_dir, &manifest_path)?;
                        if let Some(run_idx) = run_idx {
                            ledger.finish_run(run_idx, "max_iters");
                            ledger.write_atomic(&ledger_path)?;
                        }
                    }
                    130 => {
                        rollback_snapshot(&project_dir, &manifest_path)?;
                        if let Some(run_idx) = run_idx {
                            ledger.finish_run(run_idx, "interrupted");
                            ledger.write_atomic(&ledger_path)?;
                        }
                    }
                    _ => {
                        rollback_snapshot(&project_dir, &manifest_path)?;
                    }
                }
                print_error(&err);
                return Ok(code);
            }
        }
    }
    cleanup_snapshot(&project_dir)?;
    Ok(any_failed_code)
}

async fn run_env(
    args: &SolveArgs,
    project_dir: &Path,
    manifest_path: &Path,
    editor: &mut ManifestEditor,
    parser: &RegexConflictParser,
    ledger: &mut SolveLedger,
    ledger_path: &Path,
    run_idx: Option<usize>,
    tried: &mut TriedState,
    feature: String,
    env: Option<&str>,
) -> std::result::Result<i32, SolveError> {
    let mut planner = RepairPlanner::new(feature.clone());
    let mut pending_edit: Option<PendingEdit> = None;
    let smoke_modules = if args.smoke_modules.is_empty() {
        editor.smoke_modules()
    } else {
        args.smoke_modules.clone()
    };
    for iter in 1..=args.max_iters {
        status::tty(&format!(
            "retread solve: pixi install iteration {iter}/{}{}",
            args.max_iters,
            env.map(|e| format!(" ({e})")).unwrap_or_default()
        ));
        let install = run_pixi_install(project_dir, env).await?;
        if install.interrupted {
            return Err(SolveError::Interrupted);
        }
        if install.success {
            if args.dry_run {
                println!("retread solve: pixi install already succeeds");
                return Ok(EXIT_OK);
            }
            if !args.no_smoke_test && !smoke_modules.is_empty() {
                if let Err(err) = super::smoke::run_smoke(project_dir, env, &smoke_modules).await {
                    if let Some(solve_err) = err.downcast_ref::<SolveError>() {
                        match solve_err {
                            SolveError::SmokeFailed { stderr_tail, .. } => {
                                eprintln!("{stderr_tail}");
                                return Ok(EXIT_SMOKE_FAILED);
                            }
                            SolveError::Usage(msg) => return Err(SolveError::Usage(msg.clone())),
                            _ => return Err(SolveError::Usage(solve_err.to_string())),
                        }
                    }
                    return Err(SolveError::Usage(err.to_string()));
                }
            }
            print_success_summary(
                run_idx
                    .and_then(|idx| ledger.runs.get(idx))
                    .map(|r| &r.attempts[..])
                    .unwrap_or(&[]),
            );
            return Ok(EXIT_OK);
        }

        let stripped = parser.strip_ansi(&install.stderr);
        let parsed = parser.parse(&stripped);
        if let Some(conflict) = parsed {
            if let Some(pending) = &pending_edit
                && pending.conflict_package != conflict.package()
            {
                pending_edit = None;
            }
            if args.dry_run {
                let mut dry_editor = ManifestEditor::open(manifest_path.to_path_buf())
                    .map_err(|e| SolveError::Usage(e.to_string()))?;
                let out = planner
                    .repair(&mut dry_editor, tried, &conflict, iter)
                    .map_err(|package| SolveError::Exhausted { package })?;
                println!("{}", out.summary_line);
                return Ok(EXIT_OK);
            }
            ensure_snapshot(project_dir, manifest_path).map_err(|e| SolveError::Usage(e.to_string()))?;
            let out = planner
                .repair(editor, tried, &conflict, iter)
                .map_err(|package| SolveError::Exhausted { package })?;
            editor
                .write_atomic()
                .map_err(|e| SolveError::Usage(e.to_string()))?;
            if let Some(run_idx) = run_idx {
                for attempt in out.extra_attempts {
                    append_attempt(ledger, ledger_path, run_idx, attempt)
                        .map_err(|e| SolveError::Usage(e.to_string()))?;
                }
                append_attempt(ledger, ledger_path, run_idx, out.attempt.clone())
                    .map_err(|e| SolveError::Usage(e.to_string()))?;
            }
            pending_edit = PendingEdit::from_edits(out.applied, conflict.package().to_string());
            continue;
        }

        if let Some(pending) = pending_edit.take() {
            if pending.strategy == Strategy::WidenConda {
                if RegexConflictParser::is_post_widen_conda_unsat(&stripped) {
                    eprintln!(
                        "retread solve: post-widen conda unsat for {}; reverting only that widen",
                        pending.package
                    );
                }
                pending.restore(editor, &feature);
                editor
                    .write_atomic()
                    .map_err(|e| SolveError::Usage(e.to_string()))?;
                tried.mark(&pending.package, Strategy::WidenConda, true);
                if let Some(run_idx) = run_idx {
                    mark_last_widen_failed(ledger, ledger_path, run_idx, &pending.package)
                        .map_err(|e| SolveError::Usage(e.to_string()))?;
                }
                continue;
            }
            pending.restore(editor, &feature);
            editor
                .write_atomic()
                .map_err(|e| SolveError::Usage(e.to_string()))?;
            if let Some(run_idx) = run_idx {
                if let Some(run) = ledger.runs.get_mut(run_idx) {
                    run.kept_pins = Some(run.attempts.len().saturating_sub(1));
                    run.reverted.push(super::repair::RevertedEdit {
                        package: pending.package,
                        strategy: pending.strategy.as_str().into(),
                    });
                }
                ledger
                    .write_atomic(ledger_path)
                    .map_err(|e| SolveError::Usage(e.to_string()))?;
            }
        }
        eprintln!("retread solve: could not parse solver error; manifest restored");
        eprintln!("{}", tail(&install.raw_stderr, 4000));
        return Err(SolveError::Unparseable {
            stderr_tail: tail(&install.raw_stderr, 4000),
        });
    }
    Err(SolveError::MaxIters(args.max_iters))
}

#[derive(Debug)]
struct PendingEdit {
    package: String,
    strategy: Strategy,
    edits: Vec<AppliedEdit>,
    conflict_package: String,
}

impl PendingEdit {
    fn from_edits(edits: Vec<AppliedEdit>, conflict_package: String) -> Option<Self> {
        let last = edits.last()?;
        Some(Self {
            package: last.package.clone(),
            strategy: last.strategy,
            edits,
            conflict_package,
        })
    }

    fn restore(&self, editor: &mut ManifestEditor, feature: &str) {
        for edit in self.edits.iter().rev() {
            editor.restore_entry(feature, edit.table, &edit.package, &edit.before);
        }
    }
}

struct InstallResult {
    success: bool,
    interrupted: bool,
    stderr: String,
    raw_stderr: String,
}

async fn run_pixi_install(project_dir: &Path, env: Option<&str>) -> std::result::Result<InstallResult, SolveError> {
    let mut cmd = Command::new("pixi");
    cmd.arg("install");
    if let Some(env) = env {
        cmd.arg("-e").arg(env);
    }
    cmd.arg("--color=never")
        .cwd(project_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("PIXI_COLOR", "never");

    let mut child = cmd
        .spawn()
        .map_err(|e| SolveError::Usage(format!("retread solve: failed to spawn pixi: {e}")))?;
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();
    let stdout_task = tokio::spawn(async move {
        let mut stdout = stdout;
        let mut buf = Vec::new();
        if let Some(mut pipe) = stdout.take() {
            pipe.read_to_end(&mut buf).await?;
        }
        Ok::<Vec<u8>, std::io::Error>(buf)
    });
    let stderr_task = tokio::spawn(async move {
        let mut stderr = stderr;
        let mut buf = Vec::new();
        if let Some(mut pipe) = stderr.take() {
            pipe.read_to_end(&mut buf).await?;
        }
        Ok::<Vec<u8>, std::io::Error>(buf)
    });
    tokio::select! {
        status = child.wait() => {
            let status = status.map_err(|e| SolveError::Usage(format!("retread solve: pixi failed: {e}")))?;
            let _stdout = stdout_task.await
                .map_err(|e| SolveError::Usage(format!("retread solve: failed joining pixi stdout reader: {e}")))?
                .map_err(|e| SolveError::Usage(format!("retread solve: failed reading pixi stdout: {e}")))?;
            let stderr = stderr_task.await
                .map_err(|e| SolveError::Usage(format!("retread solve: failed joining pixi stderr reader: {e}")))?
                .map_err(|e| SolveError::Usage(format!("retread solve: failed reading pixi stderr: {e}")))?;
            let raw_stderr = String::from_utf8_lossy(&stderr).to_string();
            Ok(InstallResult {
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
            Ok(InstallResult {
                success: false,
                interrupted: true,
                stderr: String::new(),
                raw_stderr: String::new(),
            })
        }
    }
}

async fn pixi_version(project_dir: &Path) -> Option<String> {
    let out = Command::new("pixi")
        .arg("--version")
        .cwd(project_dir)
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

fn warn_if_uncalibrated(pixi_version: Option<&str>) {
    let Some(version) = pixi_version else {
        return;
    };
    let calibrated = RegexConflictParser::CALIBRATED
        .iter()
        .any(|prefix| version.contains(prefix));
    if !calibrated {
        eprintln!(
            "retread solve: parser calibrated for pixi 0.70.x; running against {version} -- conflict parsing may fail (exit 2). Capture fixtures and extend CALIBRATED."
        );
    }
}

fn resolve_feature(manifest_path: &Path, env: Option<&str>, explicit: Option<&str>) -> String {
    if let Some(feature) = explicit {
        return feature.to_string();
    }
    if let Some(env) = env
        && let Some(ws) = manifest_path
            .parent()
            .and_then(WorkspaceManifest::load)
        && let Some(def) = ws.environments.get(env)
        && def.no_default_feature
        && def.features.len() == 1
    {
        let feature = def.features[0].clone();
        eprintln!("retread solve: targeting feature {feature} for env {env}");
        return feature;
    }
    if env.is_some() {
        eprintln!(
            "retread solve: default-feature pins re-solve all environments; use --feature for scoped repair when possible"
        );
    }
    "default".to_string()
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

fn outcome_for_code(code: i32) -> &'static str {
    match code {
        2 => "unparseable",
        3 => "exhausted",
        4 => "max_iters",
        130 => "interrupted",
        _ => "failed",
    }
}

fn print_error(err: &SolveError) {
    eprintln!("{err}");
    match err {
        SolveError::Unparseable { stderr_tail }
        | SolveError::SmokeFailed { stderr_tail, .. } => {
            eprintln!("{stderr_tail}");
        }
        _ => {}
    }
}

fn print_success_summary(attempts: &[LedgerAttempt]) {
    if attempts.is_empty() {
        println!("retread solve: pixi install converged without manifest repairs");
        return;
    }
    println!("retread solve: converged repairs");
    for tier in ["widen-conda", "conda", "pypi_dep", "pypi_override"] {
        let group: Vec<_> = attempts.iter().filter(|a| a.strategy == tier).collect();
        if group.is_empty() {
            continue;
        }
        println!("{tier}:");
        for a in group {
            if let Some(spec) = &a.new_spec {
                println!("  {} = \"{}\"", a.package, spec);
            } else if let Some(version) = &a.version {
                println!("  {} = \"=={}\"", a.package, version);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_resolution_uses_single_no_default_feature() {
        let dir = std::env::temp_dir().join(format!(
            "retread-solve-driver-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = dir.join("pixi.toml");
        std::fs::write(
            &manifest,
            r#"[workspace]
channels = ["conda-forge"]

[environments]
gpu = { features = ["cuda"], no-default-feature = true }

[feature.cuda.dependencies]
python = "3.11.*"
"#,
        )
        .unwrap();
        assert_eq!(resolve_feature(&manifest, Some("gpu"), None), "cuda");
        assert_eq!(resolve_feature(&manifest, Some("gpu"), Some("manual")), "manual");
    }

    #[test]
    fn snapshot_rollback_restores_manifest_bytes() {
        let dir = std::env::temp_dir().join(format!(
            "retread-solve-rollback-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = dir.join("pixi.toml");
        std::fs::write(&manifest, b"[dependencies]\nnumpy = \"==1.26.4\"\n").unwrap();
        ensure_snapshot(&dir, &manifest).unwrap();
        std::fs::write(&manifest, b"[dependencies]\nnumpy = \"==2.3.1\"\n").unwrap();
        rollback_snapshot(&dir, &manifest).unwrap();
        assert_eq!(
            std::fs::read(&manifest).unwrap(),
            b"[dependencies]\nnumpy = \"==1.26.4\"\n"
        );
        assert!(!snapshot_path(&dir).exists());
    }
}
