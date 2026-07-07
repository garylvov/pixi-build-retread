//! Line-delimited JSON-RPC 2.0 over stdin/stdout.
//!
//! Each message is a complete JSON-RPC 2.0 frame on a single line, terminated
//! by `\n`. This matches the transport in
//! `pixi/crates/pixi_build_frontend/src/jsonrpc/stdio.rs`.

use std::future::Future;

use futures::StreamExt;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tokio::io::{AsyncWriteExt, BufWriter, Stdout};
use tokio::sync::Mutex;
use tokio_util::codec::{FramedRead, LinesCodec};

/// Standard JSON-RPC 2.0 error codes.
pub const PARSE_ERROR: i32 = -32700;
pub const INVALID_REQUEST: i32 = -32600;
pub const METHOD_NOT_FOUND: i32 = -32601;
pub const INVALID_PARAMS: i32 = -32602;
pub const INTERNAL_ERROR: i32 = -32603;

#[derive(Debug)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    pub data: Option<Value>,
}

impl RpcError {
    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self {
            code: INVALID_PARAMS,
            message: msg.into(),
            data: None,
        }
    }
    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            code: INTERNAL_ERROR,
            message: msg.into(),
            data: None,
        }
    }
}

impl<E: std::fmt::Display> From<E> for RpcError {
    fn from(e: E) -> Self {
        Self::internal(e.to_string())
    }
}

/// Reads requests from stdin, dispatches them to `handler`, writes responses
/// to stdout. Returns when stdin closes (EOF).
pub async fn serve<H, F>(handler: H) -> anyhow::Result<()>
where
    H: Fn(String, Value) -> F + Send + Sync + 'static,
    F: Future<Output = Result<Value, RpcError>> + Send,
{
    let stdin = tokio::io::stdin();
    let stdout = Mutex::new(BufWriter::new(tokio::io::stdout()));
    let mut frames = FramedRead::new(stdin, LinesCodec::new());

    while let Some(line) = frames.next().await {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(error = %e, "stdin read error");
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }

        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                write_error(&stdout, Value::Null, PARSE_ERROR, e.to_string()).await?;
                continue;
            }
        };

        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = match req.get("method").and_then(|m| m.as_str()) {
            Some(m) => m.to_string(),
            None => {
                write_error(&stdout, id, INVALID_REQUEST, "missing method".to_string()).await?;
                continue;
            }
        };
        let params = req.get("params").cloned().unwrap_or(Value::Null);

        // Notifications (no id) get no response.
        let is_notification = req.get("id").is_none();

        tracing::debug!(method = %method, "rpc request");
        match handler(method.clone(), params).await {
            Ok(result) => {
                if !is_notification {
                    write_response(&stdout, id, result).await?;
                }
            }
            Err(err) => {
                if !is_notification {
                    write_error(&stdout, id, err.code, err.message).await?;
                }
            }
        }
    }

    Ok(())
}

async fn write_response(
    out: &Mutex<BufWriter<Stdout>>,
    id: Value,
    result: Value,
) -> anyhow::Result<()> {
    let msg = json!({ "jsonrpc": "2.0", "id": id, "result": result });
    write_line(out, &msg).await
}

async fn write_error(
    out: &Mutex<BufWriter<Stdout>>,
    id: Value,
    code: i32,
    message: String,
) -> anyhow::Result<()> {
    let msg = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message }
    });
    write_line(out, &msg).await
}

async fn write_line(out: &Mutex<BufWriter<Stdout>>, msg: &Value) -> anyhow::Result<()> {
    let mut s = serde_json::to_string(msg)?;
    // Pixi's transport strips embedded newlines and terminates with \n.
    s = s.replace('\n', "");
    s.push('\n');
    let mut guard = out.lock().await;
    guard.write_all(s.as_bytes()).await?;
    guard.flush().await?;
    Ok(())
}

/// Helper: deserialize JSON-RPC params into a typed struct.
pub fn parse_params<T: DeserializeOwned>(params: Value) -> Result<T, RpcError> {
    serde_json::from_value(params)
        .map_err(|e| RpcError::invalid_params(format!("invalid params: {e}")))
}

/// Helper: serialize a typed result into JSON-RPC `result`.
pub fn ok<T: Serialize>(value: T) -> Result<Value, RpcError> {
    serde_json::to_value(value).map_err(RpcError::from)
}
