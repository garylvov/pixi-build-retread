//! Test the JSON-RPC protocol contract pixi actually uses.
//!
//! Spawns the release binary as a subprocess, writes line-delimited
//! JSON-RPC 2.0 requests to its stdin, reads responses from stdout, and
//! ASSERTS THAT EVERY LINE OF STDOUT PARSES AS VALID JSON-RPC. This is
//! what caught us before: pip/git/rattler-build writing progress to
//! stdout corrupts the protocol, which cargo's normal test harness
//! cannot detect because stdout isn't a channel there.
//!
//! Run with:
//!
//! ```bash
//! cargo build --release && \
//!   cargo test --test jsonrpc_protocol -- --include-ignored
//! ```

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::{json, Value};

fn backend_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/release/pixi-build-retread")
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn assert_release_built() {
    let bin = backend_binary();
    assert!(
        bin.exists(),
        "release binary not found at {} -- run `cargo build --release` first",
        bin.display()
    );
}

/// Send a sequence of JSON-RPC requests to the backend and collect
/// responses. Critically, every stdout line MUST be valid JSON --
/// anything else means a subprocess (pip/git/rattler-build) corrupted
/// the protocol channel.
fn drive_backend(requests: &[Value]) -> (Vec<Value>, String) {
    assert_release_built();
    let mut child = Command::new(backend_binary())
        .env("PIXI_BUILD_RETREAD_LOG", "info")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn backend");

    {
        let stdin = child.stdin.as_mut().expect("stdin");
        for req in requests {
            let line = serde_json::to_string(req).unwrap();
            writeln!(stdin, "{line}").expect("write request");
        }
        // Close stdin so the backend exits cleanly after responding.
    }
    // Closing stdin requires dropping the handle; this happens at end
    // of the block above.
    drop(child.stdin.take());

    let output = child.wait_with_output().expect("wait_with_output");
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf-8");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    let mut responses = Vec::new();
    for (i, line) in stdout.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parsed: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => panic!(
                "stdout line {i} is NOT valid JSON-RPC -- some subprocess \
                 corrupted the protocol channel. \
                 error: {e}\nline: {trimmed}\n--- full stderr ---\n{stderr}"
            ),
        };
        assert_eq!(
            parsed.get("jsonrpc").and_then(Value::as_str),
            Some("2.0"),
            "response missing jsonrpc:'2.0' field: {trimmed}"
        );
        responses.push(parsed);
    }

    if !output.status.success() {
        panic!(
            "backend exited with {:?}\nstdout:\n{stdout}\n--- stderr ---\n{stderr}",
            output.status.code()
        );
    }
    (responses, stderr)
}

#[test]
#[ignore = "requires release build + network for parselmouth fetch"]
fn negotiate_initialize_outputs_round_trip() {
    // Smallest possible request set: negotiate -> initialize ->
    // conda/outputs with one URL-form wheel that resolves without
    // building anything heavy. This catches the protocol-corruption
    // class even on a fast wheel.
    let tmp = std::env::temp_dir().join(format!(
        "retread-rpc-test-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp).unwrap();

    let requests = vec![
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "negotiateCapabilities",
            "params": { "capabilities": {} }
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "initialize",
            "params": {
                "manifestPath": tmp.join("pixi.toml"),
                "sourceDirectory": &tmp,
                "configuration": {
                    "retread-wheels": {
                        "tomli": { "version": "==2.0.1" }
                    }
                }
            }
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "conda/outputs",
            "params": {
                "hostPlatform": "linux-64",
                "buildPlatform": "linux-64",
                "channels": [],
                "workDirectory": &tmp,
            }
        }),
    ];

    let (responses, _stderr) = drive_backend(&requests);
    assert_eq!(responses.len(), 3, "got: {responses:#?}");

    // Verify the conda/outputs response has at least one output named tomli.
    let outputs = responses[2]
        .get("result")
        .and_then(|r| r.get("outputs"))
        .and_then(Value::as_array)
        .expect("conda/outputs result.outputs missing or wrong shape");
    assert!(
        outputs
            .iter()
            .any(|o| o.get("metadata").and_then(|m| m.get("name")) == Some(&json!("tomli"))),
        "expected an output named 'tomli', got: {outputs:#?}"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
#[ignore = "requires release build + pip; exercises pip wheel which would corrupt stdout if unguarded"]
fn path_source_does_not_corrupt_stdout() {
    // The pip-wheel path used to dump 'Collecting setuptools...' etc.
    // to OUR stdout, corrupting the JSON-RPC channel. This test wires
    // up a path-source entry against a fixture project that requires
    // pip to actually run a build (with [build-system].requires forcing
    // pip's isolated build env, which prints lots of progress). If any
    // of that leaks to stdout, drive_backend()'s "every line must be
    // valid JSON" assertion fails.
    let fixture = fixtures_dir().join("sample_with_buildtime_dep");
    assert!(
        fixture.exists(),
        "fixture missing at {} -- regenerate via tests/fixtures/",
        fixture.display()
    );

    let tmp = std::env::temp_dir().join(format!(
        "retread-rpc-path-test-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp).unwrap();

    let requests = vec![
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "negotiateCapabilities",
            "params": { "capabilities": {} }
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "initialize",
            "params": {
                "manifestPath": tmp.join("pixi.toml"),
                "sourceDirectory": &fixture,
                "configuration": {
                    "retread-wheels": {
                        "retread-sample": { "path": "." }
                    }
                }
            }
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "conda/outputs",
            "params": {
                "hostPlatform": "linux-64",
                "buildPlatform": "linux-64",
                "channels": [],
                "workDirectory": &tmp,
            }
        }),
    ];

    let (responses, _stderr) = drive_backend(&requests);
    assert_eq!(responses.len(), 3, "got: {responses:#?}");
    // Validate the outputs response succeeded (no error field).
    assert!(
        responses[2].get("error").is_none(),
        "conda/outputs returned error: {:#?}",
        responses[2]
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
#[ignore = "requires release build + network for PyPI simple-index lookup"]
fn broken_entry_surfaces_with_entry_name() {
    // Regression: conda_outputs used to swallow resolve_all errors per
    // python variant and log a tracing::warn. On a single-variant build
    // (the common case -- no [workspace.build-variants] python list),
    // that meant a broken entry produced empty `outputs`, which pixi
    // reports as the bare "the package 'X' is not provided by the project
    // located at './Y'" -- with no mention of WHICH entry failed or WHY.
    // This test pins the fail-fast contract: a deterministically-broken
    // entry must produce an error response whose message names the entry.
    let tmp = std::env::temp_dir().join(format!(
        "retread-rpc-broken-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp).unwrap();

    let requests = vec![
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "negotiateCapabilities",
            "params": { "capabilities": {} }
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "initialize",
            "params": {
                "manifestPath": tmp.join("pixi.toml"),
                "sourceDirectory": &tmp,
                "configuration": {
                    "retread-wheels": {
                        // tomli is a real PyPI package; 999.999.999 will
                        // never exist, so pypi::resolve bails with
                        // "no wheels match tomli == 999.999.999".
                        "tomli": { "version": "==999.999.999" }
                    }
                }
            }
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "conda/outputs",
            "params": {
                "hostPlatform": "linux-64",
                "buildPlatform": "linux-64",
                "channels": [],
                "workDirectory": &tmp,
            }
        }),
    ];

    let (responses, _stderr) = drive_backend(&requests);
    assert_eq!(responses.len(), 3, "got: {responses:#?}");

    let err = responses[2]
        .get("error")
        .unwrap_or_else(|| panic!("conda/outputs should have errored: {:#?}", responses[2]));
    let msg = err
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("error has no message: {err:#?}"));
    assert!(
        msg.contains("tomli"),
        "error message must name the offending entry `tomli`, got: {msg}"
    );

    std::fs::remove_dir_all(&tmp).ok();
}
