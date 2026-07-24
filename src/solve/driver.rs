use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use crate::status;
use crate::workspace::WorkspaceManifest;

use super::args::SolveArgs;
use super::error::{
    EXIT_EXHAUSTED, EXIT_INTERRUPTED, EXIT_OK, EXIT_SMOKE_FAILED, EXIT_UNPARSEABLE, SolveError,
};
use super::manifest::{AppliedEdit, ManifestEditor, copy_atomic, restore_bytes_atomic};
use super::parse::{
    Conflict, ConflictParser, RegexConflictParser, RetreadConflictSuggestion,
    RetreadMutuallyUnsatisfiable, tail,
};
use super::repair::{
    LedgerAttempt, RelaxPreference, RepairPlanner, SolveLedger, Strategy, TriedState,
    append_attempt, ledger_path, manifest_sha256, mark_last_widen_failed, persist_conflict_trace,
    retread_dir, snapshot_path, truncate_ledger_runs,
};

pub async fn run(args: SolveArgs) -> Result<i32> {
    run_with_pixi_bin(args, "pixi").await
}

async fn run_with_pixi_bin(args: SolveArgs, pixi_bin: &str) -> Result<i32> {
    let manifest_path = normalize_manifest(&args.manifest);
    if !manifest_path.exists() {
        let err = SolveError::Usage(format!(
            "retread solve: manifest {} not found",
            manifest_path.display()
        ));
        eprintln!("{err}");
        return Ok(err.exit_code());
    }

    if !args.edit_manifest {
        let report = run_workspace_audit(&manifest_path, pixi_bin).await?;
        if args.apply_ledger
            && let Some(reason) = report.apply_blocker()
        {
            eprintln!("retread solve: --apply-ledger refused: {reason}");
            eprint!("{}", report.render(AuditApplication::Refused));
            return Ok(report.exit_code());
        }
        let application = if args.apply_ledger {
            apply_audit_ledger(&manifest_path, &report)?;
            AuditApplication::Applied
        } else {
            AuditApplication::NotRequested
        };
        eprint!("{}", report.render(application));
        if args.apply_ledger {
            return Ok(EXIT_OK);
        }
        return Ok(report.exit_code());
    }

    eprintln!(
        "WARNING: `retread solve --edit-manifest` enables the discouraged legacy repair loop; \
         it may edit {}. Each exact proposed edit will be printed before it is written.",
        manifest_path.display()
    );
    run_legacy_manifest_edit(args).await
}

async fn run_legacy_manifest_edit(args: SolveArgs) -> Result<i32> {
    let manifest_path = normalize_manifest(&args.manifest);
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
        eprintln!(
            "WARNING: editing {} now: remove {removed} retread-managed manifest pins",
            manifest_path.display()
        );
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct AuditScopeFailure {
    conflicts: Vec<RetreadMutuallyUnsatisfiable>,
    fallback: Option<String>,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct AuditReport {
    checked: usize,
    scopes: BTreeMap<String, AuditScopeFailure>,
    interrupted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AuditApplication {
    NotRequested,
    Applied,
    Refused,
}

impl AuditReport {
    fn record_conflicts(
        &mut self,
        conflicts: impl IntoIterator<Item = RetreadMutuallyUnsatisfiable>,
    ) {
        for conflict in conflicts {
            let failure = self.scopes.entry(conflict.scope.clone()).or_default();
            if !failure.conflicts.contains(&conflict) {
                failure.conflicts.push(conflict);
                failure.conflicts.sort();
            }
        }
    }

    fn conflict_count(&self) -> usize {
        self.scopes
            .values()
            .map(|failure| failure.conflicts.len())
            .sum()
    }

    fn conflicting_scope_count(&self) -> usize {
        self.scopes
            .values()
            .filter(|failure| !failure.conflicts.is_empty())
            .count()
    }

    fn apply_blocker(&self) -> Option<String> {
        if self.interrupted {
            return Some("the audit was interrupted; no ledger entries were written".into());
        }
        let fallback_scopes = self
            .scopes
            .iter()
            .filter_map(|(scope, failure)| failure.fallback.as_ref().map(|_| format!("`{scope}`")))
            .collect::<Vec<_>>();
        if fallback_scopes.is_empty() {
            None
        } else {
            Some(format!(
                "the audit did not produce structured conflicts for {}; no ledger entries were \
                 written",
                fallback_scopes.join(", ")
            ))
        }
    }

    fn exit_code(&self) -> i32 {
        if self.interrupted {
            EXIT_INTERRUPTED
        } else if self
            .scopes
            .values()
            .any(|failure| failure.fallback.is_some())
        {
            EXIT_UNPARSEABLE
        } else if !self.scopes.is_empty() {
            EXIT_EXHAUSTED
        } else {
            EXIT_OK
        }
    }

    fn render(&self, application: AuditApplication) -> String {
        if self.scopes.is_empty() && !self.interrupted {
            return format!(
                "retread solve: read-only audit passed for all {} environments\n",
                self.checked
            );
        }

        let mut out = String::new();
        let _ = writeln!(
            out,
            "retread solve: read-only audit checked {} environments; {} dependency conflicts across {} scopes",
            self.checked,
            self.conflict_count(),
            self.conflicting_scope_count()
        );
        for (scope, failure) in &self.scopes {
            let _ = writeln!(out, "\nscope {scope}:");
            for conflict in &failure.conflicts {
                let _ = writeln!(
                    out,
                    "  pack `{}` (bundle `{}`, platform {}, python {}):",
                    conflict.suggestion.pack_manifest(),
                    conflict.bundle,
                    conflict.platform,
                    conflict.python
                );
                let _ = writeln!(out, "    package `{}`", conflict.package);
                let _ = writeln!(out, "    conflicting wheels/requirements:");
                for requirement in &conflict.requirements {
                    let _ = writeln!(
                        out,
                        "      - `{}` required by {}",
                        requirement.spec, requirement.source
                    );
                }
                let label = if application == AuditApplication::Applied {
                    "applied to pack-overrides ledger"
                } else {
                    "proposed relaxation (not applied)"
                };
                let _ = writeln!(out, "    {label}:");
                for line in conflict
                    .suggestion
                    .render_toml(&conflict.package)
                    .trim_end()
                    .lines()
                {
                    let _ = writeln!(out, "      {line}");
                }
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
        } else if application == AuditApplication::Applied {
            let _ = writeln!(
                out,
                "\nretread solve: ledger proposals applied; pixi.toml was not modified"
            );
        } else if application == AuditApplication::Refused {
            let _ = writeln!(
                out,
                "\nretread solve: no changes applied because the audit was incomplete"
            );
        } else {
            let _ = writeln!(
                out,
                "\nretread solve: no changes applied; rerun with --apply-ledger to persist these proposals"
            );
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
        let workspace = WorkspaceManifest::from_toml(manifest);
        populate_audit_shadow(&shadow, project_dir, manifest_path, &workspace)?;
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
    workspace: &WorkspaceManifest,
) -> Result<()> {
    let manifest_name = manifest_path.file_name();
    let copied_entries = audit_mutable_path_entries(project_dir, workspace)?;
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
        let source = entry.path();
        let destination = shadow.root.join(&name);
        if copied_entries.contains(&source) {
            copy_audit_tree(&source, &destination)?;
        } else {
            symlink_audit_entry(&source, &destination, entry.file_type()?.is_dir())?;
        }
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

    // The audit must observe the same effective pack configuration as the
    // real workspace. Copy only the durable pack-overrides ledger; solve
    // traces/snapshots remain deliberately absent from the disposable
    // workspace. A symlink is rejected by `copy_audit_tree`, so a relative
    // backend write can never traverse back through this path.
    let ledger = crate::pack_overrides::ledger_path(project_dir);
    if ledger.exists() {
        let shadow_ledger = crate::pack_overrides::ledger_path(&shadow.root);
        std::fs::create_dir_all(
            shadow_ledger
                .parent()
                .expect("pack-overrides ledger always has a parent"),
        )?;
        copy_audit_tree(&ledger, &shadow_ledger)?;
    }
    Ok(())
}

/// Top-level workspace entries containing a manifest path dependency.
///
/// Build backends are allowed to write probe logs and caches beside their
/// source manifest even during `pixi update --dry-run`. Those directories
/// must therefore be copied into the disposable shadow, never symlinked back
/// to the real workspace. External (`..`/absolute) path dependencies cannot
/// be isolated without rewriting the manifest, so fail closed before pixi is
/// launched.
fn audit_mutable_path_entries(
    project_dir: &Path,
    workspace: &WorkspaceManifest,
) -> Result<BTreeSet<PathBuf>> {
    let paths = workspace
        .path_dependencies
        .values()
        .chain(
            workspace
                .target_dependencies
                .iter()
                .flat_map(|(_, target)| target.path_dependencies.values()),
        )
        .chain(
            workspace
                .features
                .values()
                .flat_map(|feature| feature.path_dependencies.values()),
        )
        .chain(workspace.features.values().flat_map(|feature| {
            feature
                .target_dependencies
                .iter()
                .flat_map(|(_, target)| target.path_dependencies.values())
        }));

    let mut entries = BTreeSet::new();
    for raw in paths {
        let path = Path::new(raw);
        if path.is_absolute() {
            anyhow::bail!(
                "retread solve audit: external path dependency `{raw}` cannot be isolated \
                 read-only"
            );
        }
        let mut top = None;
        for component in path.components() {
            match component {
                std::path::Component::CurDir => {}
                std::path::Component::Normal(name) => {
                    top.get_or_insert_with(|| PathBuf::from(name));
                }
                std::path::Component::ParentDir
                | std::path::Component::RootDir
                | std::path::Component::Prefix(_) => {
                    anyhow::bail!(
                        "retread solve audit: external path dependency `{raw}` cannot be \
                         isolated read-only"
                    );
                }
            }
        }
        let Some(top) = top else {
            anyhow::bail!(
                "retread solve audit: workspace-root path dependency `{raw}` cannot be \
                 isolated read-only"
            );
        };
        entries.insert(project_dir.join(top));
    }
    Ok(entries)
}

fn copy_audit_tree(source: &Path, destination: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(source).with_context(|| {
        format!(
            "retread solve audit: failed inspecting mutable path source {}",
            source.display()
        )
    })?;
    if metadata.file_type().is_symlink() {
        anyhow::bail!(
            "retread solve audit: symlink {} inside a mutable path source cannot be isolated \
             read-only",
            source.display()
        );
    }
    if metadata.is_dir() {
        std::fs::create_dir(destination).with_context(|| {
            format!(
                "retread solve audit: failed creating copied path source {}",
                destination.display()
            )
        })?;
        for entry in std::fs::read_dir(source)? {
            let entry = entry?;
            copy_audit_tree(&entry.path(), &destination.join(entry.file_name()))?;
        }
    } else if metadata.is_file() {
        std::fs::copy(source, destination).with_context(|| {
            format!(
                "retread solve audit: failed copying mutable path source {}",
                source.display()
            )
        })?;
    } else {
        anyhow::bail!(
            "retread solve audit: unsupported file type in mutable path source {}",
            source.display()
        );
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
                report.scopes.insert(
                    format!("in environment '{environment}'"),
                    AuditScopeFailure {
                        fallback: Some(error.to_string()),
                        ..AuditScopeFailure::default()
                    },
                );
                continue;
            }
        };
        let result =
            match run_pixi_audit(&shadow.root, &shadow.manifest, &environment, pixi_bin).await {
                Ok(result) => result,
                Err(error) => {
                    report.scopes.insert(
                        format!("in environment '{environment}'"),
                        AuditScopeFailure {
                            fallback: Some(error.to_string()),
                            ..AuditScopeFailure::default()
                        },
                    );
                    continue;
                }
            };
        if result.interrupted {
            report.interrupted = true;
            report.scopes.insert(
                format!("in environment '{environment}'"),
                AuditScopeFailure {
                    fallback: Some("pixi audit solve was interrupted".into()),
                    ..AuditScopeFailure::default()
                },
            );
            break;
        }

        let mut conflicts = parser.parse_retread_conflicts(&result.raw_stderr);
        conflicts.sort();
        conflicts.dedup();
        if result.success && conflicts.is_empty() {
            continue;
        }
        if conflicts.is_empty() {
            let flattened = collapse_diagnostic_whitespace(&parser.strip_ansi(&result.raw_stderr));
            let fallback = if flattened.is_empty() {
                "pixi update --dry-run failed without stderr".to_string()
            } else {
                tail(&flattened, 2000)
            };
            report.scopes.insert(
                format!("in environment '{environment}'"),
                AuditScopeFailure {
                    fallback: Some(fallback),
                    ..AuditScopeFailure::default()
                },
            );
            continue;
        }
        report.record_conflicts(conflicts);
    }
    Ok(report)
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum AuditLedgerChoice {
    DropDependency,
    Override { spec: String },
    RootPin { spec: String, bundle_group: String },
}

#[derive(Clone, Debug)]
struct AuditLedgerProposal {
    representative: RetreadMutuallyUnsatisfiable,
    pack_pixi: PathBuf,
    package: String,
    choice: AuditLedgerChoice,
    bundles: BTreeSet<String>,
    provenance: BTreeSet<String>,
}

fn apply_audit_ledger(manifest_path: &Path, report: &AuditReport) -> Result<usize> {
    if let Some(reason) = report.apply_blocker() {
        anyhow::bail!("retread solve: --apply-ledger refused: {reason}");
    }

    let mut proposals: BTreeMap<(PathBuf, String), AuditLedgerProposal> = BTreeMap::new();
    for failure in report.scopes.values() {
        for retread in &failure.conflicts {
            if !retread.suggestion_from_backend {
                anyhow::bail!(
                    "retread solve: `{}` has no embedded Track-2 suggestion; refusing to infer \
                     an --apply-ledger action (the read-only fallback remains available)",
                    retread.package
                );
            }
            let pack_pixi = resolve_audit_pack_manifest(manifest_path, retread)?;
            let (package, choice) = match &retread.suggestion {
                RetreadConflictSuggestion::DropDependency { .. } => {
                    (retread.package.clone(), AuditLedgerChoice::DropDependency)
                }
                RetreadConflictSuggestion::Override { package, spec, .. } => (
                    package.clone(),
                    AuditLedgerChoice::Override { spec: spec.clone() },
                ),
                RetreadConflictSuggestion::RootPin {
                    package,
                    spec,
                    bundle_group,
                    ..
                } => (
                    package.clone(),
                    AuditLedgerChoice::RootPin {
                        spec: spec.clone(),
                        bundle_group: bundle_group.clone(),
                    },
                ),
            };
            let pack_identity = pack_pixi
                .parent()
                .unwrap_or(&pack_pixi)
                .canonicalize()
                .unwrap_or_else(|_| pack_pixi.parent().unwrap_or(&pack_pixi).to_path_buf());
            let package_identity = crate::relax::PypiKey::from_pypi(&package).into_string();
            let key = (pack_identity, package_identity);
            let provenance = retread.provenance();
            match proposals.get_mut(&key) {
                Some(existing) if existing.choice == choice => {
                    existing.bundles.insert(retread.bundle.clone());
                    existing.provenance.insert(provenance);
                }
                Some(existing) => {
                    anyhow::bail!(
                        "retread solve: divergent Track-2 suggestions target `{}` in {} \
                         ({:?} versus {:?}); no ledger entries were written",
                        package,
                        pack_pixi.display(),
                        existing.choice,
                        choice
                    );
                }
                None => {
                    proposals.insert(
                        key,
                        AuditLedgerProposal {
                            representative: retread.clone(),
                            pack_pixi,
                            package,
                            choice,
                            bundles: BTreeSet::from([retread.bundle.clone()]),
                            provenance: BTreeSet::from([provenance]),
                        },
                    );
                }
            }
        }
    }

    // Exercise the same planner dispatch as the live parser before opening
    // the ledger or performing any write. Its contract for this structured
    // type is planning-only: no ManifestEditor edit and no implicit ledger
    // mutation.
    let mut editor = ManifestEditor::open(manifest_path.to_path_buf())?;
    let project_dir = editor.project_dir().to_path_buf();
    let mut planner = RepairPlanner::new("default".to_string());
    let mut tried = TriedState::default();
    for proposal in proposals.values() {
        let parsed = Conflict::RetreadMutuallyUnsatisfiable(proposal.representative.clone());
        let plan = planner
            .repair(&mut editor, &mut tried, &parsed, 1)
            .map_err(|package| {
                anyhow::anyhow!("retread solve: no ledger proposal available for `{package}`")
            })?;
        if !plan.applied.is_empty() || plan.pack_override.is_some() {
            anyhow::bail!(
                "retread solve: internal error: retread-owned conflict planned a legacy edit"
            );
        }
    }

    let applied = proposals.len();
    let mut updates = Vec::with_capacity(applied);
    let mut also_invalidate = BTreeSet::new();
    for proposal in proposals.into_values() {
        let bundle = proposal
            .bundles
            .first()
            .expect("every ledger proposal has a triggering bundle")
            .clone();
        let provenance = proposal
            .provenance
            .into_iter()
            .collect::<Vec<_>>()
            .join("; ");
        let action = match proposal.choice {
            AuditLedgerChoice::DropDependency => {
                crate::pack_overrides::SolveLedgerAction::DropDependency
            }
            AuditLedgerChoice::Override { spec } => {
                crate::pack_overrides::SolveLedgerAction::Override { spec }
            }
            AuditLedgerChoice::RootPin { spec, bundle_group } => {
                crate::pack_overrides::SolveLedgerAction::RootPin { spec, bundle_group }
            }
        };
        for also_bundle in &proposal.bundles {
            if also_bundle != &bundle {
                also_invalidate.insert(also_bundle.clone());
            }
        }
        updates.push(crate::pack_overrides::SolveLedgerUpdate {
            pack_pixi: proposal.pack_pixi,
            bundle,
            package: proposal.package,
            provenance,
            action,
        });
    }
    crate::pack_overrides::write_solve_updates(&project_dir, &updates)?;
    for bundle in also_invalidate {
        crate::pack_overrides::invalidate_pixi_source_metadata(&project_dir, &bundle);
    }
    Ok(applied)
}

fn resolve_audit_pack_manifest(
    manifest_path: &Path,
    conflict: &RetreadMutuallyUnsatisfiable,
) -> Result<PathBuf> {
    let project_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let source = std::fs::read_to_string(manifest_path).with_context(|| {
        format!(
            "retread solve: failed reading manifest {} while verifying bundle ownership",
            manifest_path.display()
        )
    })?;
    let parsed: toml::Value = toml::from_str(&source).with_context(|| {
        format!(
            "retread solve: failed parsing manifest {} while verifying bundle ownership",
            manifest_path.display()
        )
    })?;
    let workspace = WorkspaceManifest::from_toml(&parsed);
    let bundle_identity = crate::relax::PypiKey::from_pypi(&conflict.bundle);
    let mut raw_paths = BTreeSet::new();
    let mut collect_matching = |dependencies: &BTreeMap<String, String>| {
        raw_paths.extend(
            dependencies
                .iter()
                .filter(|(name, _)| crate::relax::PypiKey::from_pypi(name) == bundle_identity)
                .map(|(_, path)| path.clone()),
        );
    };
    collect_matching(&workspace.path_dependencies);
    for (_, target) in &workspace.target_dependencies {
        collect_matching(&target.path_dependencies);
    }
    for feature in workspace.features.values() {
        collect_matching(&feature.path_dependencies);
        for (_, target) in &feature.target_dependencies {
            collect_matching(&target.path_dependencies);
        }
    }
    let mut owned_manifests = BTreeSet::new();
    for raw in raw_paths {
        let pack_dir = PathBuf::from(&raw);
        let pack_dir = if pack_dir.is_absolute() {
            pack_dir
        } else {
            project_dir.join(pack_dir)
        };
        let pack_manifest = pack_dir.join("pixi.toml");
        if pack_manifest.is_file() {
            owned_manifests.insert(
                pack_manifest
                    .canonicalize()
                    .unwrap_or_else(|_| pack_manifest.clone()),
            );
        }
    }
    let authoritative = match owned_manifests.len() {
        1 => owned_manifests
            .into_iter()
            .next()
            .expect("one owned pack manifest"),
        0 => anyhow::bail!(
            "retread solve: bundle `{}` is not an existing path dependency in {}",
            conflict.bundle,
            manifest_path.display()
        ),
        count => anyhow::bail!(
            "retread solve: bundle `{}` resolves to {count} different pack manifests in {}; \
             refusing an ambiguous ledger write",
            conflict.bundle,
            manifest_path.display()
        ),
    };

    let suggested = PathBuf::from(conflict.suggestion.pack_manifest());
    let suggested = if suggested.is_absolute() {
        suggested
    } else {
        project_dir.join(suggested)
    };
    let suggested = suggested.canonicalize().with_context(|| {
        format!(
            "retread solve: proposed pack manifest {} for bundle `{}` does not exist",
            suggested.display(),
            conflict.bundle
        )
    })?;
    if suggested != authoritative {
        anyhow::bail!(
            "retread solve: proposed pack manifest {} does not own bundle `{}`; {} does",
            suggested.display(),
            conflict.bundle,
            authoritative.display()
        );
    }
    Ok(authoritative)
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
            if let Conflict::RetreadMutuallyUnsatisfiable(retread) = &conflict {
                eprintln!(
                    "retread solve: the backend supplied a ledger relaxation for `{}`; \
                     --edit-manifest will not translate it into a pixi.toml edit",
                    retread.package
                );
                eprintln!(
                    "proposed relaxation for {} (not applied):",
                    retread.suggestion.pack_manifest()
                );
                for line in retread
                    .suggestion
                    .render_toml(&retread.package)
                    .trim_end()
                    .lines()
                {
                    eprintln!("  {line}");
                }
                eprintln!(
                    "rerun without --edit-manifest for a read-only audit, or use \
                     --apply-ledger to persist the proposal"
                );
                return Err(SolveError::Exhausted {
                    package: retread.package.clone(),
                });
            }
            ensure_snapshot(project_dir, manifest_path)
                .map_err(|e| SolveError::Usage(e.to_string()))?;
            let out = planner
                .repair(editor, tried, &conflict, iter)
                .map_err(|package| SolveError::Exhausted { package })?;
            eprintln!(
                "WARNING: editing {} now: {}",
                manifest_path.display(),
                out.summary_line
            );
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
    fn write_track4_workspace(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
        use std::os::unix::fs::PermissionsExt;

        let manifest = dir.join("pixi.toml");
        std::fs::write(
            &manifest,
            r#"[workspace]
channels = ["conda-forge"]
platforms = ["linux-64"]

[dependencies]
robotics_output = { path = "pypi-packs/robotics-pack" }

[environments]
default = []
"#,
        )
        .unwrap();
        let pack_dir = dir.join("pypi-packs").join("robotics-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_manifest = pack_dir.join("pixi.toml");
        std::fs::write(
            &pack_manifest,
            r#"[package]
name = "robotics-output"
version = "1.0.0"

[package.build.config]
retread-auto-bundle = true
"#,
        )
        .unwrap();

        let pixi = dir.join("pixi-track4");
        let body = r#"#!/bin/bash
touch pypi-packs/robotics-pack/audit-write-sentinel
cat >&2 <<'EOF'
dependency conflict in environment 'default' for bundle 'robotics-output' (target profile 'linux-64', platform linux-64, python 3.11): `numpy` requirements are mutually unsatisfiable: `==1.26.4` required by wheel `old-extension==1.0.0` Requires-Dist `numpy==1.26.4`; `>=2,<3` required by wheel `new-extension==2.0.0` Requires-Dist `numpy>=2,<3`. Resolve by pinning one side, or use `retread-relax`, `retread-overrides`, or `retread-drop-deps` in the pack manifest (see README).

Suggested fix in pypi-packs/robotics-pack/pixi.toml:
# Edit the existing [package.build.config] table.
retread-drop-deps = ["numpy"]
# Alternative 1:
# numpy = "==1.26.4"
# Alternative 2:
# numpy = ">=2,<3"
EOF
exit 1
"#;
        std::fs::write(&pixi, body).unwrap();
        let mut permissions = std::fs::metadata(&pixi).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&pixi, permissions).unwrap();
        (manifest, pack_manifest, pixi)
    }

    fn track4_conflict(suggestion: RetreadConflictSuggestion) -> RetreadMutuallyUnsatisfiable {
        RetreadMutuallyUnsatisfiable {
            scope: "in environment 'default'".into(),
            bundle: "robotics-output".into(),
            platform: "linux-64".into(),
            python: "3.11".into(),
            package: "numpy".into(),
            requirements: vec![
                super::super::parse::RetreadConflictRequirement {
                    spec: "==1.26.4".into(),
                    source: "wheel `old-extension==1.0.0`".into(),
                },
                super::super::parse::RetreadConflictRequirement {
                    spec: ">=2,<3".into(),
                    source: "wheel `new-extension==2.0.0`".into(),
                },
            ],
            suggestion,
            suggestion_from_backend: true,
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn default_is_byte_identical_read_only_and_apply_writes_only_pack_ledger() {
        let dir = audit_test_dir();
        let (manifest, pack_manifest, pixi) = write_track4_workspace(&dir);
        let manifest_before = std::fs::read(&manifest).unwrap();
        let pack_before = std::fs::read(&pack_manifest).unwrap();
        let manifest_hash_before = manifest_sha256(&manifest).unwrap();
        let pack_hash_before = manifest_sha256(&pack_manifest).unwrap();

        let read_only_code = run_with_pixi_bin(
            SolveArgs {
                manifest: manifest.clone(),
                ..SolveArgs::default()
            },
            pixi.to_str().unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(read_only_code, EXIT_EXHAUSTED);
        assert_eq!(std::fs::read(&manifest).unwrap(), manifest_before);
        assert_eq!(std::fs::read(&pack_manifest).unwrap(), pack_before);
        assert_eq!(manifest_sha256(&manifest).unwrap(), manifest_hash_before);
        assert_eq!(manifest_sha256(&pack_manifest).unwrap(), pack_hash_before);
        assert!(
            !pack_manifest
                .parent()
                .unwrap()
                .join("audit-write-sentinel")
                .exists(),
            "audit backend write escaped the disposable pack copy"
        );
        assert!(
            !crate::pack_overrides::ledger_path(&dir).exists(),
            "default solve wrote the pack-overrides ledger"
        );
        assert!(!dir.join(".retread").exists());

        let apply_code = run_with_pixi_bin(
            SolveArgs {
                manifest: manifest.clone(),
                apply_ledger: true,
                ..SolveArgs::default()
            },
            pixi.to_str().unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(apply_code, EXIT_OK);
        assert_eq!(std::fs::read(&manifest).unwrap(), manifest_before);
        assert_eq!(std::fs::read(&pack_manifest).unwrap(), pack_before);
        assert_eq!(manifest_sha256(&manifest).unwrap(), manifest_hash_before);
        assert_eq!(manifest_sha256(&pack_manifest).unwrap(), pack_hash_before);
        assert!(
            !pack_manifest
                .parent()
                .unwrap()
                .join("audit-write-sentinel")
                .exists(),
            "apply audit backend write escaped the disposable pack copy"
        );

        let ledger = crate::pack_overrides::AutoOverrideLedger::load(&dir).unwrap();
        let dropped = &ledger.dropped_dependencies["pypi-packs/robotics-pack"]["numpy"];
        assert_eq!(dropped.bundle, "robotics-output");
        assert!(dropped.provenance.contains("old-extension==1.0.0"));
        assert!(dropped.provenance.contains("new-extension==2.0.0"));
        assert!(!dir.join(".retread/solve-ledger.json").exists());
        assert!(!dir.join(".retread/solve-conflicts").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn audit_shadow_uses_supplied_manifest_and_effective_ledger_without_write_through() {
        let dir = audit_test_dir();
        let default_manifest = dir.join("pixi.toml");
        std::fs::write(
            &default_manifest,
            "[dependencies]\nwrong = { path = \"../must-not-be-loaded\" }\n",
        )
        .unwrap();
        let custom_manifest = dir.join("custom-workspace.toml");
        let custom_source = r#"[workspace]
channels = ["conda-forge"]
platforms = ["linux-64"]

[dependencies]
custom-output = { path = "custom-packs/source" }

[environments]
default = []
"#;
        std::fs::write(&custom_manifest, custom_source).unwrap();
        let pack_dir = dir.join("custom-packs/source");
        std::fs::create_dir_all(&pack_dir).unwrap();
        let pack_manifest = pack_dir.join("pixi.toml");
        std::fs::write(
            &pack_manifest,
            "[package]\nname = \"custom-output\"\nversion = \"1.0.0\"\n",
        )
        .unwrap();
        crate::pack_overrides::write_override(
            &dir,
            &pack_manifest,
            "custom-output",
            "numpy",
            ">=2",
            "existing effective decision",
        )
        .unwrap();
        let ledger_before = std::fs::read(crate::pack_overrides::ledger_path(&dir)).unwrap();

        let parsed: toml::Value = toml::from_str(custom_source).unwrap();
        {
            let shadow =
                AuditShadow::create(&dir, &custom_manifest, custom_source, &parsed, "default")
                    .unwrap();
            let shadow_pack = shadow.root.join("custom-packs/source/pixi.toml");
            assert!(shadow_pack.is_file());
            assert!(
                !std::fs::symlink_metadata(shadow_pack.parent().unwrap())
                    .unwrap()
                    .file_type()
                    .is_symlink()
            );
            assert_eq!(
                crate::pack_overrides::overrides_for_pack(&shadow.root, &shadow_pack)
                    .get("numpy")
                    .map(String::as_str),
                Some(">=2")
            );
            assert_eq!(
                std::fs::read(crate::pack_overrides::ledger_path(&shadow.root)).unwrap(),
                ledger_before
            );
            std::fs::write(
                shadow_pack.parent().unwrap().join("audit-write-sentinel"),
                b"shadow only",
            )
            .unwrap();
        }

        assert!(!pack_dir.join("audit-write-sentinel").exists());
        assert_eq!(
            std::fs::read(crate::pack_overrides::ledger_path(&dir)).unwrap(),
            ledger_before
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn apply_refuses_incomplete_divergent_or_unowned_proposals_before_writing() {
        let dir = audit_test_dir();
        let (manifest, pack_manifest, _) = write_track4_workspace(&dir);
        let manifest_before = std::fs::read(&manifest).unwrap();
        let pack_before = std::fs::read(&pack_manifest).unwrap();
        let drop = track4_conflict(RetreadConflictSuggestion::DropDependency {
            pack_manifest: "pypi-packs/robotics-pack/pixi.toml".into(),
            alternatives: Vec::new(),
        });

        let incomplete = AuditReport {
            checked: 2,
            scopes: BTreeMap::from([
                (
                    drop.scope.clone(),
                    AuditScopeFailure {
                        conflicts: vec![drop.clone()],
                        fallback: None,
                    },
                ),
                (
                    "in environment 'other'".into(),
                    AuditScopeFailure {
                        fallback: Some("unstructured solver failure".into()),
                        ..AuditScopeFailure::default()
                    },
                ),
            ]),
            interrupted: false,
        };
        assert!(apply_audit_ledger(&manifest, &incomplete).is_err());
        assert!(!crate::pack_overrides::ledger_path(&dir).exists());

        let mut override_conflict = drop.clone();
        override_conflict.scope = "in environment 'peer'".into();
        override_conflict.suggestion = RetreadConflictSuggestion::Override {
            pack_manifest: "pypi-packs/robotics-pack/pixi.toml".into(),
            package: "numpy".into(),
            spec: ">=2,<3".into(),
        };
        let divergent = AuditReport {
            checked: 2,
            scopes: BTreeMap::from([
                (
                    drop.scope.clone(),
                    AuditScopeFailure {
                        conflicts: vec![drop.clone()],
                        fallback: None,
                    },
                ),
                (
                    override_conflict.scope.clone(),
                    AuditScopeFailure {
                        conflicts: vec![override_conflict],
                        fallback: None,
                    },
                ),
            ]),
            interrupted: false,
        };
        assert!(apply_audit_ledger(&manifest, &divergent).is_err());
        assert!(!crate::pack_overrides::ledger_path(&dir).exists());

        let evil_dir = dir.join("pypi-packs/not-the-owner");
        std::fs::create_dir_all(&evil_dir).unwrap();
        std::fs::write(
            evil_dir.join("pixi.toml"),
            "[package]\nname = \"not-the-owner\"\nversion = \"1\"\n",
        )
        .unwrap();
        let mut unowned = drop;
        unowned.suggestion = RetreadConflictSuggestion::DropDependency {
            pack_manifest: "pypi-packs/not-the-owner/pixi.toml".into(),
            alternatives: Vec::new(),
        };
        let unowned_report = AuditReport {
            checked: 1,
            scopes: BTreeMap::from([(
                unowned.scope.clone(),
                AuditScopeFailure {
                    conflicts: vec![unowned],
                    fallback: None,
                },
            )]),
            interrupted: false,
        };
        assert!(apply_audit_ledger(&manifest, &unowned_report).is_err());
        assert!(!crate::pack_overrides::ledger_path(&dir).exists());
        assert_eq!(std::fs::read(&manifest).unwrap(), manifest_before);
        assert_eq!(std::fs::read(&pack_manifest).unwrap(), pack_before);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn structured_scope_deduplicates_shared_solve_group_conflicts() {
        let mut conflict = track4_conflict(RetreadConflictSuggestion::DropDependency {
            pack_manifest: "pypi-packs/robotics-pack/pixi.toml".into(),
            alternatives: Vec::new(),
        });
        conflict.scope = "in environments 'cpu', 'gpu'".into();
        let mut report = AuditReport {
            checked: 2,
            ..AuditReport::default()
        };
        report.record_conflicts([conflict.clone()]);
        report.record_conflicts([conflict]);

        assert_eq!(report.conflict_count(), 1);
        assert_eq!(report.conflicting_scope_count(), 1);
        let rendered = report.render(AuditApplication::NotRequested);
        assert_eq!(
            rendered
                .matches("scope in environments 'cpu', 'gpu':")
                .count(),
            1
        );
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
        let rendered = report.render(AuditApplication::NotRequested);
        assert_eq!(
            rendered.matches("retread solve: read-only audit").count(),
            1
        );
        assert!(rendered.contains("scope in environment 'alpha':"));
        assert!(rendered.contains(
            "pack `pypi-packs/alpha-pack/pixi.toml` (bundle `alpha-pack`, platform linux-64, python 3.11)"
        ));
        assert!(rendered.contains("package `numpy`"));
        assert!(rendered.contains("- `<2` required by wheel `alpha-root==1.0`"));
        assert!(rendered.contains("- `>=2` required by wheel `alpha-peer==1.0`"));
        assert!(rendered.contains("scope in environment 'beta':"));
        assert!(rendered.contains("package `psutil`"));
        assert!(rendered.contains("- `>=7,<8` required by workspace conda fact"));
        assert_eq!(
            rendered
                .matches("proposed relaxation (not applied):")
                .count(),
            2
        );
        assert!(rendered.contains("retread-drop-deps = [\"numpy\"]"));
        assert!(
            rendered.find("scope in environment 'alpha'").unwrap()
                < rendered.find("scope in environment 'beta'").unwrap()
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
