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
            prefer_pypi: false,
        }
    }
}

pub fn parse(argv: &[String]) -> anyhow::Result<SolveArgs> {
    let mut args = SolveArgs::default();
    let mut non_clean_flag_seen = false;
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
                let env = it
                    .next()
                    .ok_or_else(|| SolveError::Usage(format!("{a} <env> required")))?;
                args.environments.push(env.clone());
            }
            "--feature" => {
                non_clean_flag_seen = true;
                let feature = it
                    .next()
                    .ok_or_else(|| SolveError::Usage("--feature <name> required".into()))?;
                args.feature = Some(feature.clone());
            }
            "--max-iters" => {
                non_clean_flag_seen = true;
                let raw = it
                    .next()
                    .ok_or_else(|| SolveError::Usage("--max-iters <N> required".into()))?;
                args.max_iters = raw.parse().map_err(|_| {
                    SolveError::Usage(format!("retread solve: bad --max-iters value {raw}"))
                })?;
            }
            "--no-smoke-test" => {
                non_clean_flag_seen = true;
                args.no_smoke_test = true;
            }
            "--keep-going" => {
                non_clean_flag_seen = true;
                args.keep_going = true;
            }
            "--smoke" => {
                non_clean_flag_seen = true;
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
                args.dry_run = true;
            }
            "--clean-pins" => {
                args.clean_pins = true;
            }
            "--prefer-pypi" => {
                non_clean_flag_seen = true;
                args.prefer_pypi = true;
            }
            other => anyhow::bail!("retread solve: unknown arg {other}"),
        }
    }

    if args.clean_pins && non_clean_flag_seen {
        return Err(SolveError::Usage(
            "retread solve: --clean-pins is mutually exclusive with every flag except --manifest"
                .into(),
        )
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
        let args = parse(&argv(&["--prefer-pypi"])).unwrap();
        assert!(args.prefer_pypi);
        let default_args = parse(&argv(&[])).unwrap();
        assert!(!default_args.prefer_pypi);
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
