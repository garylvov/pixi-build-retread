use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use anyhow::Result;
use regex::Regex;
use tokio::process::Command;

use super::error::SolveError;
use super::parse::tail;

pub async fn run_smoke(project_dir: &Path, env: Option<&str>, modules: &[String]) -> Result<()> {
    if modules.is_empty() {
        return Ok(());
    }
    let valid = Regex::new(r"^[A-Za-z_][A-Za-z0-9_.]*$").expect("valid smoke module regex");
    for module in modules {
        if !valid.is_match(module) {
            return Err(SolveError::Usage(format!(
                "retread solve: invalid smoke module {module}"
            ))
            .into());
        }
    }
    let code = modules
        .iter()
        .map(|m| format!("import {m}"))
        .collect::<Vec<_>>()
        .join("; ");
    let mut cmd = Command::new("pixi");
    cmd.arg("run");
    if let Some(env) = env {
        cmd.arg("-e").arg(env);
    }
    cmd.arg("--color=never")
        .arg("python")
        .arg("-c")
        .arg(code)
        .cwd(project_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("PIXI_COLOR", "never");
    let output = tokio::time::timeout(Duration::from_secs(600), cmd.output())
        .await
        .map_err(|_| SolveError::SmokeFailed {
            module: modules.join(","),
            stderr_tail: "smoke test timed out after 600s".into(),
        })??;
    if output.status.success() {
        return Ok(());
    }
    Err(SolveError::SmokeFailed {
        module: modules.join(","),
        stderr_tail: tail(&String::from_utf8_lossy(&output.stderr), 4000),
    }
    .into())
}
