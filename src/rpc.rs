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
    /// Formats with `{:#}`, NOT `{}`. For `anyhow::Error` the alternate flag
    /// renders the whole cause chain (`outer: middle: root`); plain `{}`
    /// prints only the outermost context and silently discards every
    /// `.context()` and the underlying I/O/parse error. Pixi's frontend
    /// surfaces this string verbatim (and `build_dispatch.rs` `.expect()`s
    /// it), so anything dropped here is unrecoverable for the operator.
    /// Types whose `Display` ignores `#` are unaffected.
    fn from(e: E) -> Self {
        Self::internal(format!("{e:#}"))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Guard for fix/obs-rpc-errors: `?` on an `anyhow::Error` must carry the
    /// whole cause chain onto the wire. With the old `e.to_string()` this
    /// asserted string is just "outer", and every `.context()` plus the root
    /// cause is destroyed before pixi ever sees it.
    #[test]
    fn anyhow_errors_keep_their_full_cause_chain_on_the_wire() {
        let error: anyhow::Error = anyhow::anyhow!("root cause: no such file")
            .context("while reading the pack manifest")
            .context("conda/build_v1 failed");
        let rpc = RpcError::from(error);
        assert_eq!(rpc.code, INTERNAL_ERROR);
        assert!(
            rpc.message.contains("conda/build_v1 failed"),
            "{}",
            rpc.message
        );
        assert!(
            rpc.message.contains("while reading the pack manifest"),
            "the intermediate context must survive; got: {}",
            rpc.message
        );
        assert!(
            rpc.message.contains("root cause: no such file"),
            "the ROOT cause must survive; got: {}",
            rpc.message
        );
    }

    #[test]
    fn plain_display_errors_are_unchanged() {
        let rpc = RpcError::from(std::io::Error::other("boom"));
        assert_eq!(rpc.message, "boom");
    }

    /// N27-RETREAD-180, the STATIC half. `serve` above takes
    /// `tokio::io::stdout()` and never gives it back: for the whole life of the
    /// process, stdout is the JSON-RPC frame channel and nothing else may write
    /// a byte to it. A single `println!` from a handler put a `### PACK
    /// SUBPACKAGES` row in front of the first response frame and the frontend
    /// died `Unparseable message: expected value at line 1 column 1` (SUBCERT-1,
    /// relock 6162669). That defect was already forbidden IN PROSE, twice, in
    /// `handler/mod.rs` -- and prose is not a reader, so it was added anyway.
    /// This is the reader.
    ///
    /// TWO TIERS, because "reachable from the RPC loop" is not a property a
    /// grep can compute:
    ///
    /// * `FORBIDDEN_ROOTS` -- `src/handler/` (every handler method, including
    ///   `initialize`, plus `auto_bundle.rs` and everything else in that tree)
    ///   and `src/rpc.rs` itself. These files ONLY ever execute with the
    ///   transport live. Allowance here is zero and there is no way to add one
    ///   short of editing this test, which is the point.
    /// * everything else under `src/` -- allowed only by NAME, with a reason,
    ///   in `ALLOWED`. A new module that grows a stdout write has to be
    ///   classified by a human as CLI-only or test-only before it can land,
    ///   which is the classification nobody performed for `initialize`.
    ///
    /// Falsifiability: put `println!` back at the `### PACK SUBPACKAGES` loop in
    /// `handler/mod.rs` and this fails, naming the file and the count. That is
    /// the mutation the landing gate ran.
    #[test]
    fn no_println_reaches_the_json_rpc_channel() {
        // Assembled, not written literally, for two reasons: this file is
        // itself inside `FORBIDDEN_ROOTS`, so a literal needle would make the
        // test match its own source and invent an offender (the same shape as
        // a process scan matching its own argv); and `eprintln!` CONTAINS
        // `println!` as a substring, so the match has to reject a preceding
        // identifier character or every correct stderr row would count.
        const NEEDLE: &str = concat!("print", "ln", "!");

        /// Files under `src/` permitted to write to stdout, each with the
        /// reason it cannot be in the RPC loop. Verified by reading the ONLY
        /// call sites: `src/main.rs` dispatches every verb from `argv[1]` and
        /// each arm `return`s, so `rpc::serve` is reached only by the
        /// argv-less transport invocation -- a verb and the transport can
        /// never both run in one process.
        const ALLOWED: &[(&str, &str)] = &[
            (
                "main.rs",
                "the CLI's own stdout. Every verb arm returns before the \
                 `rpc::serve` call at the bottom of `main`.",
            ),
            (
                "store_reap.rs",
                "`retread store-reap`: `run`/`reap_one` are called from \
                 main.rs's argv dispatch and from this module's own tests. No \
                 handler references `store_reap::`.",
            ),
            (
                "sdist_metadata.rs",
                "`retread sdist-meta` / `sdist-meta-key` / `sdist-meta-tags`: \
                 `run`, `key_main` and `run_tags` are argv verbs whose printed \
                 rows are the verb's output, consumed by the shell halves.",
            ),
            (
                "solve/driver.rs",
                "`retread solve`: the convergence report is the verb's output. \
                 The only symbol a handler takes from `crate::solve` is the \
                 pure `is_abi_anchor`.",
            ),
            (
                "solve/lock.rs",
                "`retread lock`: the repair report is the verb's output, read \
                 by the relock templates.",
            ),
            (
                "thread_budget.rs",
                "inside `#[cfg(all(test, target_os = \"linux\"))] mod tests` -- \
                 `unseeded_multiprocess_acquire_helper` is a re-executed test \
                 child and its stdout IS the channel its parent test reads. \
                 Not compiled into the shipped binary.",
            ),
            (
                "wheel.rs",
                "inside `#[cfg(test)]`: diagnostic output of the single-pass \
                 strict-metadata guard. Not compiled into the shipped binary.",
            ),
            (
                "uv_closure.rs",
                "inside `#[cfg(test)]`: diagnostic output of the built-source \
                 validation memo guard. Not compiled into the shipped binary.",
            ),
        ];

        /// Trees where stdout is forbidden outright: they run only while the
        /// transport owns the channel.
        const FORBIDDEN_ROOTS: &[&str] = &["handler", "rpc.rs"];

        /// Count INVOCATIONS of the needle macro, line by line. Three
        /// exclusions, each of them a false positive this guard measured on the
        /// tree it guards:
        ///
        /// * anything after a `//` on the line. `handler/mod.rs` discusses this
        ///   exact macro in three separate comment blocks -- including the two
        ///   that forbid it -- and a guard that counted prose would have been
        ///   red on the very commit that fixed the defect.
        /// * a hit that is the tail of a longer identifier. `eprintln!`
        ///   CONTAINS the needle, so without this every correct stderr row in
        ///   the crate would be an offender.
        /// * a hit not followed by a macro delimiter, which is how the needle
        ///   appears inside backticks in a doc comment.
        fn hits(text: &str, needle: &str) -> usize {
            let mut n = 0;
            for line in text.lines() {
                // Only the code half of the line. A `//` inside a string
                // literal truncates early, which can only DROP later hits on
                // that line, never invent one.
                let code = line.split("//").next().unwrap_or("");
                let bytes = code.as_bytes();
                let mut from = 0;
                while let Some(rel) = code[from..].find(needle) {
                    let at = from + rel;
                    let prev_ok = at == 0 || {
                        let p = bytes[at - 1];
                        !(p.is_ascii_alphanumeric() || p == b'_')
                    };
                    let after = at + needle.len();
                    let next_ok = matches!(bytes.get(after), Some(b'(' | b'[' | b'{'));
                    if prev_ok && next_ok {
                        n += 1;
                    }
                    from = after;
                }
            }
            n
        }

        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        let mut stack = vec![src.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("read_dir under src/") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                    files.push(path);
                }
            }
        }
        assert!(
            files.len() > 20,
            "the scan found only {} .rs files under {} -- it is not reading the \
             tree it thinks it is, and a guard that inspects nothing cannot fail",
            files.len(),
            src.display(),
        );

        let mut forbidden = Vec::new();
        let mut unclassified = Vec::new();
        for path in &files {
            let rel = path
                .strip_prefix(&src)
                .expect("under src/")
                .to_string_lossy()
                .replace('\\', "/");
            let text = std::fs::read_to_string(path).expect("read a source file");
            let n = hits(&text, NEEDLE);
            if n == 0 {
                continue;
            }
            let in_forbidden = FORBIDDEN_ROOTS.iter().any(|root| {
                rel == *root || rel.starts_with(&format!("{root}/"))
            });
            if in_forbidden {
                forbidden.push(format!("{rel} ({n})"));
            } else if !ALLOWED.iter().any(|(name, _)| *name == rel) {
                unclassified.push(format!("{rel} ({n})"));
            }
        }

        assert!(
            forbidden.is_empty(),
            "a stdout write appeared in a module that only runs while \
             `rpc::serve` owns stdout -- it will interleave a line into the \
             JSON-RPC frame stream and the frontend will die `Unparseable \
             message: expected value at line 1 column 1` (N27-RETREAD-180). \
             Use `eprintln!`; the harness tees backend stderr into \
             `<arm>.backend.log`. Offenders: {forbidden:?}",
        );
        assert!(
            unclassified.is_empty(),
            "a stdout write appeared in a module under src/ that is neither \
             forbidden nor on this test's reasoned allow-list. Establish from \
             its CALL SITES whether it can execute while the transport is up: \
             if it can, use `eprintln!`; if it cannot (an argv verb, or \
             `#[cfg(test)]` code), add it to ALLOWED with that reason. \
             Unclassified: {unclassified:?}",
        );

        // The allow-list is a reader too: an entry for a file that no longer
        // writes to stdout is a stale exemption that would silently cover a
        // future one.
        let stale: Vec<&str> = ALLOWED
            .iter()
            .map(|(name, _)| *name)
            .filter(|name| {
                let path = src.join(name);
                match std::fs::read_to_string(&path) {
                    Ok(text) => hits(&text, NEEDLE) == 0,
                    Err(_) => true,
                }
            })
            .collect();
        assert!(
            stale.is_empty(),
            "ALLOWED names {stale:?}, which no longer write to stdout (or no \
             longer exist). Drop the entry -- a stale exemption is a hole.",
        );
    }
}
