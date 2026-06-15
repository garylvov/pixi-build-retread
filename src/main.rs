//! pixi-build-retread: a pixi build backend that repacks PyPI wheels as
//! conda packages with relaxed dependency pins.
//!
//! Speaks line-delimited JSON-RPC 2.0 over stdin/stdout, per the pixi build
//! protocol (`crates/pixi_build_types`, API version 4).

use pixi_build_retread::{handler, installer, rpc};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // v2.0.0 courier: `retread install --lock <path> [--prefix <p>]` is
    // invoked from the courier conda package's post-link script to install
    // the bundle's PyPI wheels into the active env. It is NOT the JSON-RPC
    // build-backend path -- handle it before the transport starts. Prefix
    // defaults to $PREFIX (set by conda post-link) then $CONDA_PREFIX.
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).map(String::as_str) == Some("install") {
        let mut lock: Option<String> = None;
        let mut prefix: Option<String> = None;
        let mut it = argv[2..].iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--lock" => lock = it.next().cloned(),
                "--prefix" => prefix = it.next().cloned(),
                other => anyhow::bail!("retread install: unknown arg {other}"),
            }
        }
        let lock =
            lock.ok_or_else(|| anyhow::anyhow!("retread install: --lock <path> required"))?;
        let prefix = prefix
            .or_else(|| std::env::var("PREFIX").ok())
            .or_else(|| std::env::var("CONDA_PREFIX").ok())
            .ok_or_else(|| {
                anyhow::anyhow!("retread install: --prefix <p> or $PREFIX/$CONDA_PREFIX required")
            })?;
        return installer::run(std::path::Path::new(&lock), std::path::Path::new(&prefix));
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
