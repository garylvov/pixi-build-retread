//! pixi-build-retread: a pixi build backend that repacks PyPI wheels as
//! conda packages with relaxed dependency pins.
//!
//! Speaks line-delimited JSON-RPC 2.0 over stdin/stdout, per the pixi build
//! protocol (`crates/pixi_build_types`, API version 4).

use pixi_build_retread::{handler, rpc};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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
