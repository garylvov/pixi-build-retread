//! pixi-build-retread: a pixi build backend that repacks PyPI wheels as
//! conda packages with relaxed dependency pins.
//!
//! Speaks line-delimited JSON-RPC 2.0 over stdin/stdout, per the pixi build
//! protocol (`crates/pixi_build_types`, API version 4).

use std::path::PathBuf;

use pixi_build_retread::{fasttmp, handler, installer, rpc, solve};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // v2.0.0 courier: `retread install --lock <path> [--prefix <p>]` is
    // invoked from the courier conda package's post-link script to install
    // the bundle's PyPI wheels into the active env. It is NOT the JSON-RPC
    // build-backend path -- handle it before the transport starts. Prefix
    // defaults to $PREFIX (set by conda post-link) then $CONDA_PREFIX.
    // `retread verify` is the cheap activate.d guard: marker + installed
    // distribution metadata only, no network or mutation.
    let argv: Vec<String> = std::env::args().collect();
    if matches!(argv.get(1).map(String::as_str), Some("fast")) {
        run_fast(&argv[2..])?;
        return Ok(());
    }

    if matches!(
        argv.get(1).map(String::as_str),
        Some("install" | "verify" | "solve")
    ) {
        let cmd = argv[1].as_str();
        if cmd == "solve" {
            let args = solve::args::parse(&argv[2..])?;
            let code = solve::run(args).await?;
            std::process::exit(code);
        }
        let mut lock: Option<String> = None;
        let mut prefix: Option<String> = None;
        let mut full = false;
        let mut it = argv[2..].iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--lock" => lock = it.next().cloned(),
                "--prefix" => prefix = it.next().cloned(),
                "--full" if cmd == "verify" => full = true,
                other => anyhow::bail!("retread {cmd}: unknown arg {other}"),
            }
        }
        let lock = lock.ok_or_else(|| anyhow::anyhow!("retread {cmd}: --lock <path> required"))?;
        let prefix = prefix
            .or_else(|| std::env::var("PREFIX").ok())
            .or_else(|| std::env::var("CONDA_PREFIX").ok())
            .ok_or_else(|| {
                anyhow::anyhow!("retread {cmd}: --prefix <p> or $PREFIX/$CONDA_PREFIX required")
            })?;
        let lock = std::path::Path::new(&lock);
        let prefix = std::path::Path::new(&prefix);
        return match cmd {
            "install" => installer::run(lock, prefix),
            "verify" => installer::verify(lock, prefix, full),
            _ => unreachable!("matched above"),
        };
    }

    // Log to stderr only — stdout is reserved for the JSON-RPC transport.
    // Per-bundle probe + routing decisions ALSO land on disk as part of
    // the audit JSON (retread-audit-<bundle>.json next to the pack's
    // pixi.toml). That audit is what to read when pixi swallows stderr
    // and you can't see this stream.
    // Filter: PIXI_BUILD_RETREAD_LOG (NOT RUST_LOG -- common gotcha).
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_env("PIXI_BUILD_RETREAD_LOG")
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        "pixi-build-retread starting"
    );

    let handler = handler::Handler::new();
    rpc::serve(move |method, params| {
        let handler = handler.clone();
        async move { handler.dispatch(method, params).await }
    })
    .await
}

struct FastCli {
    workspace: Option<PathBuf>,
    print_env: bool,
    preflight_locks: Option<PathBuf>,
    preflight_lock_helper: Option<PathBuf>,
    cmd: Vec<String>,
}

fn run_fast(args: &[String]) -> anyhow::Result<()> {
    let parsed = parse_fast_args(args)?;
    if let Some(path) = parsed.preflight_lock_helper {
        return fasttmp::preflight_lock_helper(&path);
    }
    if let Some(path) = parsed.preflight_locks {
        return fasttmp::preflight_locks(&path);
    }

    let workspace = match parsed.workspace {
        Some(dir) => fasttmp::find_workspace_root(&dir)?,
        None => fasttmp::find_workspace_root(&std::env::current_dir()?)?,
    };
    let cfg = fasttmp::FastTmpConfig::load(&workspace);
    let engaged = fasttmp::engage(&workspace, &cfg)?;

    if parsed.print_env {
        if let Some(engaged) = engaged.as_ref() {
            print!("{}", fasttmp::shell_exports(engaged));
        } else {
            eprintln!(
                "retread fast-tmp: disengaged for {} (mode off or filesystem not slow)",
                workspace.display()
            );
        }
        return Ok(());
    }
    if parsed.cmd.is_empty() {
        anyhow::bail!(
            "retread fast: expected command after `--` (or use --print-env / --preflight-locks)"
        );
    }

    match engaged.as_ref() {
        Some(engaged) => {
            fasttmp::check_env_eviction(&workspace, &engaged.ns);
            fasttmp::print_mapping(engaged);
            fasttmp::run_frozen_install_if_slurm(&workspace, &engaged.env)?;
            exec_fast_command(&parsed.cmd, Some(&engaged.env), false)
        }
        None => {
            eprintln!(
                "retread fast-tmp: disengaged for {} (mode off or filesystem not slow)",
                workspace.display()
            );
            exec_fast_command(&parsed.cmd, None, fasttmp::inherited_fasttmp_stale())
        }
    }
}

fn parse_fast_args(args: &[String]) -> anyhow::Result<FastCli> {
    let mut out = FastCli {
        workspace: None,
        print_env: false,
        preflight_locks: None,
        preflight_lock_helper: None,
        cmd: Vec::new(),
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--" => {
                out.cmd = args[i + 1..].to_vec();
                break;
            }
            "--workspace" => {
                i += 1;
                let Some(value) = args.get(i) else {
                    anyhow::bail!("retread fast: --workspace requires a directory");
                };
                out.workspace = Some(PathBuf::from(value.as_str()));
            }
            "--print-env" => out.print_env = true,
            "--preflight-locks" => {
                i += 1;
                let Some(value) = args.get(i) else {
                    anyhow::bail!("retread fast: --preflight-locks requires a shared path");
                };
                out.preflight_locks = Some(PathBuf::from(value.as_str()));
            }
            "--preflight-lock-helper" => {
                i += 1;
                let Some(value) = args.get(i) else {
                    anyhow::bail!("retread fast: --preflight-lock-helper requires a lock path");
                };
                out.preflight_lock_helper = Some(PathBuf::from(value.as_str()));
            }
            other if other.starts_with("--") => {
                anyhow::bail!("retread fast: unknown arg {other}");
            }
            _ => {
                out.cmd = args[i..].to_vec();
                break;
            }
        }
        i += 1;
    }
    Ok(out)
}

#[cfg(unix)]
fn exec_fast_command(
    cmd: &[String],
    env: Option<&[(String, String)]>,
    remove_stale_fast_env: bool,
) -> anyhow::Result<()> {
    use std::os::unix::process::CommandExt;

    let mut command = std::process::Command::new(&cmd[0]);
    command.args(&cmd[1..]);
    if remove_stale_fast_env {
        fasttmp::remove_stale_fast_env_from_command(&mut command);
    }
    if let Some(env) = env {
        command.envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    }
    let err = command.exec();
    Err(anyhow::anyhow!("retread fast: exec {} failed: {err}", cmd[0]))
}

#[cfg(not(unix))]
fn exec_fast_command(
    cmd: &[String],
    env: Option<&[(String, String)]>,
    remove_stale_fast_env: bool,
) -> anyhow::Result<()> {
    let mut command = std::process::Command::new(&cmd[0]);
    command.args(&cmd[1..]);
    if remove_stale_fast_env {
        fasttmp::remove_stale_fast_env_from_command(&mut command);
    }
    if let Some(env) = env {
        command.envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    }
    let status = command.status()?;
    std::process::exit(status.code().unwrap_or(1));
}
