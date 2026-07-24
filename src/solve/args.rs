use std::path::PathBuf;

use super::error::SolveError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveArgs {
    pub manifest: PathBuf,
    pub environments: Vec<String>,
    pub feature: Option<String>,
    pub max_iters: u32,
    pub no_smoke_test: bool,
    pub keep_going: bool,
    pub smoke_modules: Vec<String>,
    pub dry_run: bool,
    pub clean_pins: bool,
    /// Audit every manifest environment without applying repair edits.
    /// This is retained as an explicit alias; audit is now the default.
    pub audit: bool,
    /// Persist Track-2 proposals to `.retread/auto-overrides.json`.
    /// This never authorizes a pixi.toml edit.
    pub apply_ledger: bool,
    /// Discouraged compatibility gate for the pre-Track-4 manifest-editing
    /// repair loop. No solve path may write pixi.toml without this flag.
    pub edit_manifest: bool,
    /// Overrides `[tool.retread] relax-preference` to `"pypi"` for this
    /// run: widen the conda pin before trying a pypi dependency-override
    /// (the historical order, predating conda-as-truth).
    pub prefer_pypi: bool,
}

impl Default for SolveArgs {
    fn default() -> Self {
        Self {
            manifest: PathBuf::from("pixi.toml"),
            environments: Vec::new(),
            feature: None,
            max_iters: 50,
            no_smoke_test: false,
            keep_going: false,
            smoke_modules: Vec::new(),
            dry_run: false,
            clean_pins: false,
            audit: false,
            apply_ledger: false,
            edit_manifest: false,
            prefer_pypi: false,
        }
    }
}

pub fn parse(argv: &[String]) -> anyhow::Result<SolveArgs> {
    let mut args = SolveArgs::default();
    let mut non_clean_flag_seen = false;
    let mut audit_incompatible_flag: Option<String> = None;
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--manifest" => {
                let p = it
                    .next()
                    .ok_or_else(|| SolveError::Usage("--manifest <path> required".into()))?;
                args.manifest = PathBuf::from(p);
            }
            "-e" | "--environment" => {
                non_clean_flag_seen = true;
                audit_incompatible_flag.get_or_insert_with(|| a.clone());
                let env = it
                    .next()
                    .ok_or_else(|| SolveError::Usage(format!("{a} <env> required")))?;
                args.environments.push(env.clone());
            }
            "--feature" => {
                non_clean_flag_seen = true;
                audit_incompatible_flag.get_or_insert_with(|| a.clone());
                let feature = it
                    .next()
                    .ok_or_else(|| SolveError::Usage("--feature <name> required".into()))?;
                args.feature = Some(feature.clone());
            }
            "--max-iters" => {
                non_clean_flag_seen = true;
                audit_incompatible_flag.get_or_insert_with(|| a.clone());
                let raw = it
                    .next()
                    .ok_or_else(|| SolveError::Usage("--max-iters <N> required".into()))?;
                args.max_iters = raw.parse().map_err(|_| {
                    SolveError::Usage(format!("retread solve: bad --max-iters value {raw}"))
                })?;
            }
            "--no-smoke-test" => {
                non_clean_flag_seen = true;
                audit_incompatible_flag.get_or_insert_with(|| a.clone());
                args.no_smoke_test = true;
            }
            "--keep-going" => {
                non_clean_flag_seen = true;
                audit_incompatible_flag.get_or_insert_with(|| a.clone());
                args.keep_going = true;
            }
            "--smoke" => {
                non_clean_flag_seen = true;
                audit_incompatible_flag.get_or_insert_with(|| a.clone());
                let raw = it.next().ok_or_else(|| {
                    SolveError::Usage("--smoke <module>[,<module>...] required".into())
                })?;
                for module in raw.split(',').map(str::trim).filter(|m| !m.is_empty()) {
                    if !args.smoke_modules.iter().any(|m| m == module) {
                        args.smoke_modules.push(module.to_string());
                    }
                }
            }
            "--dry-run" => {
                non_clean_flag_seen = true;
                audit_incompatible_flag.get_or_insert_with(|| a.clone());
                args.dry_run = true;
            }
            "--clean-pins" => {
                audit_incompatible_flag.get_or_insert_with(|| a.clone());
                args.clean_pins = true;
            }
            "--audit" | "--all-environments" => args.audit = true,
            "--apply-ledger" => args.apply_ledger = true,
            "--edit-manifest" => args.edit_manifest = true,
            "--prefer-pypi" => {
                non_clean_flag_seen = true;
                audit_incompatible_flag.get_or_insert_with(|| a.clone());
                args.prefer_pypi = true;
            }
            other => anyhow::bail!("retread solve: unknown arg {other}"),
        }
    }

    if args.clean_pins && non_clean_flag_seen {
        return Err(SolveError::Usage(
            "retread solve: --clean-pins is mutually exclusive with every flag except \
             --manifest and its required --edit-manifest opt-in"
                .into(),
        )
        .into());
    }
    if args.audit
        && let Some(flag) = audit_incompatible_flag
    {
        return Err(SolveError::Usage(format!(
            "retread solve: --audit is mutually exclusive with {flag}"
        ))
        .into());
    }
    if args.edit_manifest && args.audit {
        return Err(SolveError::Usage(
            "retread solve: --edit-manifest is mutually exclusive with --audit".into(),
        )
        .into());
    }
    if args.edit_manifest && args.apply_ledger {
        return Err(SolveError::Usage(
            "retread solve: --apply-ledger never authorizes --edit-manifest".into(),
        )
        .into());
    }
    if args.audit && args.apply_ledger {
        return Err(SolveError::Usage(
            "retread solve: --audit is explicitly read-only and cannot be combined with \
             --apply-ledger; use --apply-ledger by itself"
                .into(),
        )
        .into());
    }
    if args.apply_ledger
        && let Some(flag) = audit_incompatible_flag
    {
        return Err(SolveError::Usage(format!(
            "retread solve: --apply-ledger is mutually exclusive with legacy repair flag {flag}"
        ))
        .into());
    }
    if args.clean_pins && !args.edit_manifest {
        return Err(SolveError::Usage(
            "retread solve: --clean-pins requires the separate discouraged --edit-manifest opt-in"
                .into(),
        )
        .into());
    }
    if !args.edit_manifest
        && !args.audit
        && !args.apply_ledger
        && let Some(flag) = audit_incompatible_flag
    {
        return Err(SolveError::Usage(format!(
            "retread solve: legacy repair flag {flag} requires the separate discouraged \
             --edit-manifest opt-in; omit legacy flags for the default read-only audit"
        ))
        .into());
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
    fn parses_repeated_envs_and_smoke_modules() {
        let args = parse(&argv(&[
            "--edit-manifest",
            "--manifest",
            "x.toml",
            "-e",
            "gpu",
            "--environment",
            "cpu",
            "--feature",
            "isaac",
            "--smoke",
            "torch,isaaclab",
            "--smoke",
            "torch",
        ]))
        .unwrap();
        assert_eq!(args.manifest, PathBuf::from("x.toml"));
        assert_eq!(args.environments, vec!["gpu", "cpu"]);
        assert_eq!(args.feature.as_deref(), Some("isaac"));
        assert_eq!(args.smoke_modules, vec!["torch", "isaaclab"]);
    }

    #[test]
    fn parses_prefer_pypi_flag() {
        let args = parse(&argv(&["--edit-manifest", "--prefer-pypi"])).unwrap();
        assert!(args.prefer_pypi);
        let default_args = parse(&argv(&[])).unwrap();
        assert!(!default_args.prefer_pypi);
    }

    #[test]
    fn parses_audit_and_all_environments_alias() {
        assert!(parse(&argv(&["--audit"])).unwrap().audit);
        assert!(parse(&argv(&["--all-environments"])).unwrap().audit);
    }

    #[test]
    fn default_is_read_only_and_apply_ledger_is_explicit() {
        let default_args = parse(&argv(&[])).unwrap();
        assert!(!default_args.apply_ledger);
        assert!(!default_args.edit_manifest);

        let apply = parse(&argv(&["--apply-ledger"])).unwrap();
        assert!(apply.apply_ledger);
        assert!(!apply.edit_manifest);
    }

    #[test]
    fn legacy_manifest_repair_requires_separate_discouraged_opt_in() {
        let err = parse(&argv(&["-e", "gpu"])).unwrap_err();
        assert!(err.to_string().contains("requires"));
        assert!(err.to_string().contains("--edit-manifest"));

        let legacy = parse(&argv(&["--edit-manifest", "-e", "gpu"])).unwrap();
        assert!(legacy.edit_manifest);
        assert_eq!(legacy.environments, ["gpu"]);

        let err = parse(&argv(&["--apply-ledger", "--edit-manifest"])).unwrap_err();
        assert!(err.to_string().contains("never authorizes"));
        let err = parse(&argv(&["--audit", "--apply-ledger"])).unwrap_err();
        assert!(err.to_string().contains("explicitly read-only"));
    }

    #[test]
    fn clean_pins_cannot_edit_without_manifest_opt_in() {
        let err = parse(&argv(&["--clean-pins"])).unwrap_err();
        assert!(err.to_string().contains("--edit-manifest"));
        assert!(
            parse(&argv(&["--clean-pins", "--edit-manifest"]))
                .unwrap()
                .clean_pins
        );
    }

    #[test]
    fn audit_rejects_repair_flags() {
        let err = parse(&argv(&["--audit", "-e", "gpu"])).unwrap_err();
        assert!(err.to_string().contains("mutually exclusive with -e"));
        let err = parse(&argv(&["--audit", "--clean-pins"])).unwrap_err();
        assert!(
            err.to_string()
                .contains("--audit is mutually exclusive with --clean-pins")
        );
    }

    #[test]
    fn clean_pins_rejects_other_flags() {
        let err = parse(&argv(&["--clean-pins", "--dry-run"])).unwrap_err();
        assert!(
            err.to_string()
                .contains("--clean-pins is mutually exclusive")
        );
    }
}
