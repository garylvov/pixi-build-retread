use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use regex::Regex;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use crate::status;
use crate::workspace::WorkspaceManifest;

use super::args::SolveArgs;
use super::error::{
    EXIT_EXHAUSTED, EXIT_INTERRUPTED, EXIT_OK, EXIT_SMOKE_FAILED, EXIT_UNPARSEABLE, SolveError,
};
use super::manifest::{AppliedEdit, ManifestEditor, copy_atomic, restore_bytes_atomic};
use super::parse::{ConflictParser, RegexConflictParser, tail};
use super::repair::{
    LedgerAttempt, RelaxPreference, RepairPlanner, SolveLedger, Strategy, TriedState,
    append_attempt, ledger_path, manifest_sha256, mark_last_widen_failed, persist_conflict_trace,
    retread_dir, snapshot_path, truncate_ledger_runs,
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

    if args.audit {
        let report = run_workspace_audit(&manifest_path, "pixi").await?;
        eprint!("{}", report.render());
        return Ok(report.exit_code());
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
    // Same parselmouth-backed pypi<->conda name family the courier/auto-
    // route path builds its `name_map` from (FALLBACK_PYPI_TO_CONDA merged
    // over the fetched parselmouth data) -- loaded once per `retread solve`
    // invocation and threaded into every env's RepairPlanner so
    // CondaWidenNeeded matching agrees with emission (e.g. pypi `torch`
    // resolves to a user's `pytorch-gpu` conda pin). Best-effort: falls back
    // to an empty map (exact-name matching only) if the fetch fails.
    let conda_name_map = crate::handler::load_pypi_to_conda_map().await;
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
            EnvRunCtx {
                args: &args,
                project_dir: &project_dir,
                manifest_path: &manifest_path,
                parser: &parser,
                ledger_path: &ledger_path,
                run_idx,
                conda_name_map: &conda_name_map,
            },
            &mut editor,
            &mut ledger,
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
                crate::pack_overrides::cleanup_all(&project_dir)?;
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
                    // Fix #20: pack-manifest overrides written for THIS env
                    // are workspace-scoped side effects -- roll them back so
                    // a failed env's override doesn't survive into the next
                    // env's attempt.
                    crate::pack_overrides::rollback_all(&project_dir)?;
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
                        crate::pack_overrides::rollback_all(&project_dir)?;
                        if let Some(run_idx) = run_idx {
                            ledger.finish_run(run_idx, "exhausted");
                            ledger.write_atomic(&ledger_path)?;
                        }
                    }
                    4 => {
                        rollback_snapshot(&project_dir, &manifest_path)?;
                        crate::pack_overrides::rollback_all(&project_dir)?;
                        if let Some(run_idx) = run_idx {
                            ledger.finish_run(run_idx, "max_iters");
                            ledger.write_atomic(&ledger_path)?;
                        }
                    }
                    130 => {
                        rollback_snapshot(&project_dir, &manifest_path)?;
                        crate::pack_overrides::rollback_all(&project_dir)?;
                        if let Some(run_idx) = run_idx {
                            ledger.finish_run(run_idx, "interrupted");
                            ledger.write_atomic(&ledger_path)?;
                        }
                    }
                    _ => {
                        rollback_snapshot(&project_dir, &manifest_path)?;
                        crate::pack_overrides::rollback_all(&project_dir)?;
                    }
                }
                print_error(&err);
                return Ok(code);
            }
        }
    }
    cleanup_snapshot(&project_dir)?;
    crate::pack_overrides::cleanup_all(&project_dir)?;
    Ok(any_failed_code)
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct AuditConflict {
    package: String,
    requirements: Vec<String>,
    remediation: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct AuditEnvironmentFailure {
    conflicts: Vec<AuditConflict>,
    fallback: Option<String>,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct AuditReport {
    checked: usize,
    environments: BTreeMap<String, AuditEnvironmentFailure>,
    interrupted: bool,
}

impl AuditReport {
    fn conflict_count(&self) -> usize {
        self.environments
            .values()
            .map(|failure| failure.conflicts.len())
            .sum()
    }

    fn conflicting_environment_count(&self) -> usize {
        self.environments
            .values()
            .filter(|failure| !failure.conflicts.is_empty())
            .count()
    }

    fn exit_code(&self) -> i32 {
        if self.interrupted {
            EXIT_INTERRUPTED
        } else if self
            .environments
            .values()
            .any(|failure| failure.fallback.is_some())
        {
            EXIT_UNPARSEABLE
        } else if !self.environments.is_empty() {
            EXIT_EXHAUSTED
        } else {
            EXIT_OK
        }
    }

    fn render(&self) -> String {
        if self.environments.is_empty() && !self.interrupted {
            return format!(
                "retread solve audit: all {} environments passed\n",
                self.checked
            );
        }

        let mut out = String::new();
        let _ = writeln!(
            out,
            "retread solve audit: {} environments checked; {} dependency conflicts across {} environments",
            self.checked,
            self.conflict_count(),
            self.conflicting_environment_count()
        );
        for (environment, failure) in &self.environments {
            let _ = writeln!(out, "\nenvironment `{environment}`:");
            for conflict in &failure.conflicts {
                let _ = writeln!(out, "  package `{}`:", conflict.package);
                let _ = writeln!(out, "    conflicting requirements:");
                for requirement in &conflict.requirements {
                    let _ = writeln!(out, "      - {requirement}");
                }
                let _ = writeln!(out, "    remediation: {}", conflict.remediation);
            }
            if let Some(fallback) = &failure.fallback {
                let _ = writeln!(
                    out,
                    "  solve failed without a scoped dependency conflict:\n    {fallback}"
                );
            }
        }
        if self.interrupted {
            let _ = writeln!(out, "\naudit interrupted");
        }
        out
    }
}

/// A disposable sibling workspace containing one selected environment (and
/// any peers that deliberately share its Pixi solve-group).
///
/// Pixi 0.73's environment selector controls installation, but lock refresh
/// still visits every outdated environment. Filtering the manifest is
/// therefore load-bearing: otherwise every iteration can abort on the same
/// unrelated first environment.
struct AuditShadow {
    root: PathBuf,
    manifest: PathBuf,
}

impl AuditShadow {
    fn create(
        project_dir: &Path,
        manifest_path: &Path,
        manifest_source: &str,
        manifest: &toml::Value,
        environment: &str,
    ) -> Result<Self> {
        let root = create_audit_shadow_dir(project_dir)?;
        let shadow = Self {
            manifest: root.join("pixi.toml"),
            root,
        };
        populate_audit_shadow(&shadow, project_dir, manifest_path)?;
        let filtered = filtered_audit_manifest(manifest_source, manifest, environment)?;
        std::fs::write(&shadow.manifest, filtered).with_context(|| {
            format!(
                "retread solve audit: failed writing {}",
                shadow.manifest.display()
            )
        })?;
        Ok(shadow)
    }
}

impl Drop for AuditShadow {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn create_audit_shadow_dir(project_dir: &Path) -> Result<PathBuf> {
    static NEXT_SHADOW: AtomicU64 = AtomicU64::new(0);

    let parent = project_dir.parent().ok_or_else(|| {
        anyhow::anyhow!(
            "retread solve audit: workspace {} has no parent directory",
            project_dir.display()
        )
    })?;
    let workspace_name = project_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("workspace");
    for _ in 0..100 {
        let serial = NEXT_SHADOW.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{workspace_name}-retread-audit-{}-{serial}",
            std::process::id()
        ));
        match std::fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "retread solve audit: failed creating sibling shadow workspace {}",
                        candidate.display()
                    )
                });
            }
        }
    }
    anyhow::bail!("retread solve audit: could not allocate a unique shadow workspace")
}

fn populate_audit_shadow(
    shadow: &AuditShadow,
    project_dir: &Path,
    manifest_path: &Path,
) -> Result<()> {
    let manifest_name = manifest_path.file_name();
    for entry in std::fs::read_dir(project_dir).with_context(|| {
        format!(
            "retread solve audit: failed reading workspace {}",
            project_dir.display()
        )
    })? {
        let entry = entry?;
        let name = entry.file_name();
        if Some(name.as_os_str()) == manifest_name
            || name == "pixi.toml"
            || name == "pixi.lock"
            || name == ".pixi"
            || name == ".retread"
        {
            continue;
        }
        symlink_audit_entry(
            &entry.path(),
            &shadow.root.join(&name),
            entry.file_type()?.is_dir(),
        )?;
    }

    let local_config = project_dir.join(".pixi").join("config.toml");
    if local_config.is_file() {
        let shadow_pixi = shadow.root.join(".pixi");
        std::fs::create_dir(&shadow_pixi)?;
        std::fs::copy(&local_config, shadow_pixi.join("config.toml")).with_context(|| {
            format!(
                "retread solve audit: failed copying local Pixi config {}",
                local_config.display()
            )
        })?;
    }
    Ok(())
}

#[cfg(unix)]
fn symlink_audit_entry(source: &Path, destination: &Path, _is_dir: bool) -> Result<()> {
    std::os::unix::fs::symlink(source, destination).with_context(|| {
        format!(
            "retread solve audit: failed linking {} into shadow workspace",
            source.display()
        )
    })
}

#[cfg(windows)]
fn symlink_audit_entry(source: &Path, destination: &Path, is_dir: bool) -> Result<()> {
    let result = if is_dir {
        std::os::windows::fs::symlink_dir(source, destination)
    } else {
        std::os::windows::fs::symlink_file(source, destination)
    };
    result.with_context(|| {
        format!(
            "retread solve audit: failed linking {} into shadow workspace",
            source.display()
        )
    })
}

fn filtered_audit_manifest(
    manifest_source: &str,
    manifest: &toml::Value,
    environment: &str,
) -> Result<String> {
    let source_root = manifest.as_table().ok_or_else(|| {
        anyhow::anyhow!("retread solve audit: manifest root must be a TOML table")
    })?;
    let source_environments = match source_root.get("environments") {
        Some(value) => Some(value.as_table().ok_or_else(|| {
            anyhow::anyhow!("retread solve audit: `environments` must be a TOML table")
        })?),
        None => None,
    };
    let selected = source_environments.and_then(|envs| envs.get(environment));
    if environment != "default" && selected.is_none() {
        anyhow::bail!(
            "retread solve audit: environment `{environment}` disappeared while filtering manifest"
        );
    }

    let selected_group = selected.and_then(environment_solve_group);
    let mut kept_environments = BTreeSet::new();
    if let Some(group) = selected_group {
        if let Some(source_environments) = source_environments {
            for (name, definition) in source_environments {
                if environment_solve_group(definition) == Some(group) {
                    kept_environments.insert(name.clone());
                }
            }
        }
    } else if selected.is_some() {
        kept_environments.insert(environment.to_string());
    }

    let preserve_declared_default =
        environment == "default" || kept_environments.contains("default");
    if environment != "default" {
        kept_environments.insert("default".to_string());
    }

    // Preserve declaration order, comments, and formatting outside the
    // environment table. In particular, target-selector order can affect
    // Pixi's last-match-wins semantics and must not be reordered by a
    // `toml::Value` round trip.
    let mut document = manifest_source
        .parse::<toml_edit::DocumentMut>()
        .context("retread solve audit: failed parsing editable manifest")?;
    if kept_environments.is_empty() {
        document.as_table_mut().remove("environments");
    } else {
        let environments = document
            .get_mut("environments")
            .and_then(toml_edit::Item::as_table_like_mut)
            .ok_or_else(|| {
                anyhow::anyhow!("retread solve audit: `environments` must be a TOML table")
            })?;
        let names = environments
            .iter()
            .map(|(name, _)| name.to_string())
            .collect::<Vec<_>>();
        for name in names {
            if !kept_environments.contains(&name) {
                environments.remove(&name);
            }
        }
        if !preserve_declared_default {
            environments.remove("default");
            let mut inert_default = toml_edit::InlineTable::new();
            inert_default.insert("features", toml_edit::Value::Array(toml_edit::Array::new()));
            inert_default.insert("no-default-feature", toml_edit::Value::from(true));
            environments.insert(
                "default",
                toml_edit::Item::Value(toml_edit::Value::InlineTable(inert_default)),
            );
        }
    }
    Ok(document.to_string())
}

fn environment_solve_group(definition: &toml::Value) -> Option<&str> {
    definition
        .as_table()?
        .get("solve-group")
        .or_else(|| definition.as_table()?.get("solve_group"))
        .and_then(toml::Value::as_str)
}

async fn run_workspace_audit(manifest_path: &Path, pixi_bin: &str) -> Result<AuditReport> {
    let manifest_source = std::fs::read_to_string(manifest_path).with_context(|| {
        format!(
            "retread solve audit: failed reading manifest {}",
            manifest_path.display()
        )
    })?;
    let manifest: toml::Value = toml::from_str(&manifest_source).with_context(|| {
        format!(
            "retread solve audit: failed parsing manifest {}",
            manifest_path.display()
        )
    })?;
    let environments: Vec<String> = WorkspaceManifest::from_toml(&manifest)
        .environments
        .into_keys()
        .collect();
    let project_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let parser = RegexConflictParser::new();
    let mut report = AuditReport::default();

    for environment in environments {
        status::tty(&format!(
            "retread solve audit: validating environment {environment}"
        ));
        report.checked += 1;
        let shadow = match AuditShadow::create(
            project_dir,
            manifest_path,
            &manifest_source,
            &manifest,
            &environment,
        ) {
            Ok(shadow) => shadow,
            Err(error) => {
                report.environments.insert(
                    environment,
                    AuditEnvironmentFailure {
                        fallback: Some(error.to_string()),
                        ..AuditEnvironmentFailure::default()
                    },
                );
                continue;
            }
        };
        let result =
            match run_pixi_audit(&shadow.root, &shadow.manifest, &environment, pixi_bin).await {
                Ok(result) => result,
                Err(error) => {
                    report.environments.insert(
                        environment,
                        AuditEnvironmentFailure {
                            fallback: Some(error.to_string()),
                            ..AuditEnvironmentFailure::default()
                        },
                    );
                    continue;
                }
            };
        if result.interrupted {
            report.interrupted = true;
            report.environments.insert(
                environment,
                AuditEnvironmentFailure {
                    fallback: Some("pixi audit solve was interrupted".into()),
                    ..AuditEnvironmentFailure::default()
                },
            );
            break;
        }

        let mut conflicts = extract_audit_conflicts(&parser, &result.raw_stderr);
        conflicts.sort();
        conflicts.dedup();
        if result.success && conflicts.is_empty() {
            continue;
        }
        let fallback = conflicts.is_empty().then(|| {
            let flattened = collapse_diagnostic_whitespace(&parser.strip_ansi(&result.raw_stderr));
            if flattened.is_empty() {
                "pixi update --dry-run failed without stderr".to_string()
            } else {
                tail(&flattened, 2000)
            }
        });
        report.environments.insert(
            environment,
            AuditEnvironmentFailure {
                conflicts,
                fallback,
            },
        );
    }
    Ok(report)
}

fn extract_audit_conflicts(parser: &RegexConflictParser, stderr: &str) -> Vec<AuditConflict> {
    static CONFLICT: OnceLock<Regex> = OnceLock::new();

    let flattened = collapse_diagnostic_whitespace(&parser.strip_ansi(stderr));
    let conflict = CONFLICT.get_or_init(|| {
        Regex::new(concat!(
            r"dependency conflict in environments?\b.*?: `([^`]+)` ",
            r"requirements are mutually unsatisfiable: (.*?)\. ",
            r"(Resolve by pinning one side, or use `retread-relax`, ",
            r"`retread-overrides`, or `retread-drop-deps` in the pack manifest ",
            r"\(see README\)\.)",
        ))
        .expect("valid audit conflict regex")
    });
    let mut seen = BTreeSet::new();
    conflict
        .captures_iter(&flattened)
        .filter_map(|captures| {
            let package = captures.get(1)?.as_str().to_string();
            let requirements = captures
                .get(2)?
                .as_str()
                .split(';')
                .map(str::trim)
                .filter(|clause| !clause.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>();
            let remediation = captures.get(3)?.as_str().to_string();
            let conflict = AuditConflict {
                package,
                requirements,
                remediation,
            };
            seen.insert(conflict.clone()).then_some(conflict)
        })
        .collect()
}

fn collapse_diagnostic_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Shared, read-only solve-session state threaded into each per-environment
/// run; bundled into one parameter object to keep `run_env`'s arity sane.
#[derive(Clone, Copy)]
struct EnvRunCtx<'a> {
    args: &'a SolveArgs,
    project_dir: &'a Path,
    manifest_path: &'a Path,
    parser: &'a RegexConflictParser,
    ledger_path: &'a Path,
    run_idx: Option<usize>,
    conda_name_map: &'a Arc<crate::handler::PypiToCondaMap>,
}

async fn run_env(
    ctx: EnvRunCtx<'_>,
    editor: &mut ManifestEditor,
    ledger: &mut SolveLedger,
    tried: &mut TriedState,
    feature: String,
    env: Option<&str>,
) -> std::result::Result<i32, SolveError> {
    let EnvRunCtx {
        args,
        project_dir,
        manifest_path,
        parser,
        ledger_path,
        run_idx,
        conda_name_map,
    } = ctx;
    let relax_preference = if args.prefer_pypi {
        RelaxPreference::Pypi
    } else {
        RelaxPreference::from_config_str(&editor.relax_preference())
    };
    let mut planner =
        RepairPlanner::configured(feature.clone(), conda_name_map.clone(), relax_preference);
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
            if !args.no_smoke_test
                && !smoke_modules.is_empty()
                && let Err(err) = super::smoke::run_smoke(project_dir, env, &smoke_modules).await
            {
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
        let trace_run = env.unwrap_or("default");
        persist_conflict_trace(project_dir, trace_run, iter, &stripped);
        if let Some(conflict) = parsed {
            if args.dry_run {
                let mut dry_editor = ManifestEditor::open(manifest_path.to_path_buf())
                    .map_err(|e| SolveError::Usage(e.to_string()))?;
                let out = planner
                    .repair(&mut dry_editor, tried, &conflict, iter)
                    .map_err(|package| SolveError::Exhausted { package })?;
                println!("{}", out.summary_line);
                return Ok(EXIT_OK);
            }
            ensure_snapshot(project_dir, manifest_path)
                .map_err(|e| SolveError::Usage(e.to_string()))?;
            let out = planner
                .repair(editor, tried, &conflict, iter)
                .map_err(|package| SolveError::Exhausted { package })?;
            editor
                .write_atomic()
                .map_err(|e| SolveError::Usage(e.to_string()))?;
            if let Some(po) = &out.pack_override {
                // Fix #22: snapshot the ledger before this run's first
                // write to it so an exhausted/interrupted run rolls it
                // back, then append the override into `.retread/
                // auto-overrides.json` -- NOT the pack's pixi.toml.
                crate::pack_overrides::ensure_snapshot(project_dir)
                    .map_err(|e| SolveError::Usage(e.to_string()))?;
                crate::pack_overrides::write_override(
                    project_dir,
                    &po.pack_pixi,
                    &po.bundle,
                    &po.package,
                    &po.spec,
                    &out.attempt.conflict,
                )
                .map_err(|e| SolveError::Usage(e.to_string()))?;
            }
            if let Some(run_idx) = run_idx {
                for attempt in out.extra_attempts {
                    append_attempt(ledger, ledger_path, run_idx, attempt)
                        .map_err(|e| SolveError::Usage(e.to_string()))?;
                }
                append_attempt(ledger, ledger_path, run_idx, out.attempt.clone())
                    .map_err(|e| SolveError::Usage(e.to_string()))?;
            }
            pending_edit = PendingEdit::from_edits(out.applied);
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
                pending.restore(editor);
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
            pending.restore(editor);
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

struct InstallResult {
    success: bool,
    interrupted: bool,
    stderr: String,
    raw_stderr: String,
}

async fn run_pixi_audit(
    project_dir: &Path,
    manifest_path: &Path,
    environment: &str,
    pixi_bin: &str,
) -> std::result::Result<InstallResult, SolveError> {
    let mut cmd = Command::new(pixi_bin);
    cmd.arg("update")
        .arg("--dry-run")
        .arg("--manifest-path")
        .arg(manifest_path)
        .arg("-e")
        .arg(environment)
        .arg("--color=never")
        .current_dir(project_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("PIXI_COLOR", "never");
    capture_pixi_output(cmd).await
}

async fn run_pixi_install(
    project_dir: &Path,
    env: Option<&str>,
) -> std::result::Result<InstallResult, SolveError> {
    let mut cmd = Command::new("pixi");
    cmd.arg("install");
    if let Some(env) = env {
        cmd.arg("-e").arg(env);
    }
    cmd.arg("--color=never")
        .current_dir(project_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("PIXI_COLOR", "never");
    capture_pixi_output(cmd).await
}

async fn capture_pixi_output(mut cmd: Command) -> std::result::Result<InstallResult, SolveError> {
    let mut child = cmd
        .spawn()
        .map_err(|e| SolveError::Usage(format!("retread solve: failed to spawn pixi: {e}")))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
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
        && let Some(ws) = manifest_path.parent().and_then(WorkspaceManifest::load)
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
        SolveError::Unparseable { stderr_tail } | SolveError::SmokeFailed { stderr_tail, .. } => {
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

    #[cfg(unix)]
    fn audit_test_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "retread-solve-audit-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[cfg(unix)]
    fn write_audit_pixi(dir: &Path) -> (PathBuf, PathBuf) {
        use std::os::unix::fs::PermissionsExt;

        let log = dir.join("pixi-invocations.log");
        let script = dir.join("pixi");
        let body = format!(
            r#"#!/bin/bash
printf '%s\n' "$*" >> "{log}"
if [ "$1" != "update" ]; then
  exit 97
fi
environment=""
manifest=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-e" ]; then
    shift
    environment="$1"
  elif [ "$1" = "--manifest-path" ]; then
    shift
    manifest="$1"
  fi
  shift
done
cp "$manifest" "{dir}/seen-$environment.toml"
case "$environment" in
  alpha)
    cat >&2 <<'EOF'
dependency conflict in environment 'alpha' for bundle 'alpha-pack' (platform linux-64, python 3.11): `numpy` requirements are mutually unsatisfiable: `<2` required by wheel `alpha-root==1.0`; `>=2` required by wheel `alpha-peer==1.0`. Resolve by pinning one side, or use `retread-relax`, `retread-overrides`, or `retread-drop-deps` in the pack manifest (see README).
EOF
    exit 1
    ;;
  beta)
    cat >&2 <<'EOF'
dependency conflict in environment 'beta' for bundle 'beta-pack' (platform linux-64, python 3.11): `psutil` requirements are mutually unsatisfiable: `==5.9.8` required by wheel `beta-root==1.0`; `>=7,<8` required by workspace conda fact. Resolve by pinning one side, or use `retread-relax`, `retread-overrides`, or `retread-drop-deps` in the pack manifest (see README).
EOF
    exit 1
    ;;
esac
exit 0
"#,
            dir = dir.display(),
            log = log.display(),
        );
        std::fs::write(&script, body).unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&script, permissions).unwrap();
        (script, log)
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn audit_mode_checks_every_environment_and_consolidates_conflicts() {
        let dir = audit_test_dir();
        let manifest = dir.join("pixi.toml");
        let manifest_source = r#"[workspace]
channels = ["conda-forge"]
platforms = ["linux-64"]

[environments]
beta = []
alpha = []
default = []

# Keep this deliberately non-lexical: audit filtering must preserve target
# selector declaration order because Pixi applies last-match-wins semantics.
[target.z.dependencies]
python = ">=3.11"

[target.a.dependencies]
python = "<3.13"
"#;
        std::fs::write(&manifest, manifest_source).unwrap();
        let (pixi, log) = write_audit_pixi(&dir);

        let report = run_workspace_audit(&manifest, pixi.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(report.checked, 3);
        assert_eq!(report.conflict_count(), 2);
        assert_eq!(report.exit_code(), EXIT_EXHAUSTED);
        let rendered = report.render();
        assert_eq!(rendered.matches("retread solve audit:").count(), 1);
        assert!(rendered.contains("environment `alpha`:"));
        assert!(rendered.contains("package `numpy`"));
        assert!(rendered.contains("- `<2` required by wheel `alpha-root==1.0`"));
        assert!(rendered.contains("- `>=2` required by wheel `alpha-peer==1.0`"));
        assert!(rendered.contains("environment `beta`:"));
        assert!(rendered.contains("package `psutil`"));
        assert!(rendered.contains("- `>=7,<8` required by workspace conda fact"));
        assert_eq!(rendered.matches("remediation:").count(), 2);
        assert!(
            rendered.find("environment `alpha`").unwrap()
                < rendered.find("environment `beta`").unwrap()
        );

        let invocations = std::fs::read_to_string(&log).unwrap();
        let invocations = invocations.lines().collect::<Vec<_>>();
        assert_eq!(invocations.len(), 3);
        for (line, environment) in invocations.iter().zip(["alpha", "beta", "default"]) {
            assert!(
                line.starts_with("update --dry-run --manifest-path "),
                "{line}"
            );
            assert!(
                line.ends_with(&format!("-e {environment} --color=never")),
                "{line}"
            );
            let words = line.split_whitespace().collect::<Vec<_>>();
            let manifest_arg = PathBuf::from(
                words[words
                    .iter()
                    .position(|word| *word == "--manifest-path")
                    .unwrap()
                    + 1],
            );
            assert!(
                !manifest_arg.exists(),
                "shadow should be removed after {environment}"
            );
        }

        for environment in ["alpha", "beta"] {
            let seen_source =
                std::fs::read_to_string(dir.join(format!("seen-{environment}.toml"))).unwrap();
            assert!(
                seen_source.find("[target.z.dependencies]").unwrap()
                    < seen_source.find("[target.a.dependencies]").unwrap(),
                "audit filtering reordered target selectors for {environment}"
            );
            assert!(
                seen_source.contains(
                    "# Keep this deliberately non-lexical: audit filtering must preserve target"
                ),
                "audit filtering dropped manifest comments for {environment}"
            );
            let seen: toml::Value = toml::from_str(&seen_source).unwrap();
            let names = seen["environments"]
                .as_table()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>();
            assert_eq!(names, vec![environment, "default"]);
            assert_eq!(
                seen["environments"]["default"]["no-default-feature"].as_bool(),
                Some(true)
            );
        }
        assert_eq!(std::fs::read_to_string(&manifest).unwrap(), manifest_source);
        assert!(!dir.join(".retread").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

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
        assert_eq!(
            resolve_feature(&manifest, Some("gpu"), Some("manual")),
            "manual"
        );
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
