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

use serde_json::{Value, json};

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
    let tmp = std::env::temp_dir().join(format!("retread-rpc-test-{}", std::process::id()));
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

    let tmp = std::env::temp_dir().join(format!("retread-rpc-path-test-{}", std::process::id()));
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
    let tmp = std::env::temp_dir().join(format!("retread-rpc-broken-{}", std::process::id()));
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

/// The enumerated git tree both subpackage transport arms drive.
///
/// `alpha` and `beta` carry the two build files `SUBPACKAGE_BUILD_FILES`
/// accepts; `gamma` carries one and is EXCLUDED by the rule; `docs` carries
/// none and is SKIPPED. That makes the row's `found=`, `included=`,
/// `excluded=` and `skipped=` fields all non-trivial, and makes a derived
/// count of 2 a number the tree really produces -- so a row that reached
/// stderr truncated or empty would not satisfy either arm's assertions.
///
/// Returns `(tmp, tree, cache, rev)`. Everything lives under one `temp_dir()`
/// directory named with pid + nanos, so two arms running concurrently cannot
/// collide and nothing is written near the live worktree (N27-RETREAD-181).
fn subpackage_fixture_tree(tag: &str) -> (PathBuf, PathBuf, PathBuf, String) {
    let tmp = std::env::temp_dir().join(format!(
        "retread-rpc-subpkg-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    let tree = tmp.join("tree");
    let cache = tmp.join("cache");
    std::fs::create_dir_all(&cache).unwrap();

    for (dir, build_file) in [
        ("alpha", Some("pyproject.toml")),
        ("beta", Some("setup.py")),
        ("gamma", Some("pyproject.toml")),
        ("docs", None),
    ] {
        let d = tree.join("source").join(dir);
        std::fs::create_dir_all(&d).unwrap();
        match build_file {
            Some(name) => std::fs::write(d.join(name), b"# fixture\n").unwrap(),
            None => std::fs::write(d.join("README.md"), b"not a distribution\n").unwrap(),
        }
    }

    // A local repo, committed with an inline identity: the gate redirects HOME
    // at a private tree, so there is no global git user to inherit and a
    // `git commit` without `-c` would refuse.
    let git = |args: &[&str]| {
        let out = Command::new("git")
            .args(args)
            .current_dir(&tree)
            .output()
            .expect("run git");
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        String::from_utf8(out.stdout).expect("git stdout is utf-8")
    };
    std::fs::create_dir_all(&tree).unwrap();
    let init = Command::new("git")
        .args(["init", "-q", "--initial-branch=main"])
        .current_dir(&tree)
        .output()
        .expect("git init");
    assert!(
        init.status.success(),
        "git init failed: {}",
        String::from_utf8_lossy(&init.stderr),
    );
    git(&["add", "-A"]);
    git(&[
        "-c",
        "user.email=guard@retread.invalid",
        "-c",
        "user.name=retread guard",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        "subpackage fixture tree",
    ]);
    let rev = git(&["rev-parse", "HEAD"]).trim().to_string();
    assert_eq!(rev.len(), 40, "expected a full sha, got {rev:?}");
    (tmp, tree, cache, rev)
}

/// N27-RETREAD-180, the REAL-TRANSPORT half of the guard.
///
/// THE DEFECT THIS EXISTS FOR. `RetreadHandler::initialize` expands
/// `[retread-subpackages]` rules and reports each one as a
/// `### PACK SUBPACKAGES ...` row. It reported it with `println!`, and
/// `src/rpc.rs`'s `serve` owns `tokio::io::stdout()` as the JSON-RPC frame
/// channel for the life of the process -- so the row went out AHEAD of the
/// initialize response and pixi's frontend died with `could not initialize the
/// build-backend ... Unparseable message: expected value at line 1 column 1`
/// on the first pack that declared a rule (SUBCERT-1, relock 6162669,
/// environment `hover-gpu`, 18 s of lock wall, no cert).
///
/// WHY NO EXISTING TEST COULD SEE IT, which is the whole reason this one is
/// shaped the way it is. Every test of the capability calls
/// `crate::subpackages::expand` in process and asserts on the returned
/// `Vec<String>`. B43's gate ran 1976 library tests and 13 integration targets
/// and the row never crossed a socket in any of them: in a cargo test, stdout
/// is not a channel, so a `println!` there is invisible by construction. The
/// only instrument that sees this class is a REAL PIPE to a REAL PROCESS, which
/// is what `drive_backend` is -- its per-line `serde_json::from_str` on stdout
/// is the assertion that fails.
///
/// WHY IT IS NOT `#[ignore]`, unlike its three neighbours. Those three need the
/// network (PyPI simple-index lookups) or pip. This one needs neither: the
/// enumerated tree is a git repo this test creates in its own temp dir and
/// `ensure_git_checkout` clones over the local filesystem, and the request
/// sequence stops at `initialize`, before anything resolves. So it runs in the
/// gate's integration stage on every landing, offline, in about a second.
///
/// FALSIFIABILITY, and the landing gate ran exactly this: change the
/// `eprintln!("{row}")` in `initialize` back to `println!` and this test fails
/// inside `drive_backend` with `stdout line 0 is NOT valid JSON-RPC`.
#[test]
fn subpackage_expansion_row_does_not_corrupt_stdout() {
    let (tmp, tree, cache, rev) = subpackage_fixture_tree("typed-plus-rule");

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
                "cacheDirectory": &cache,
                "configuration": {
                    // ONE typed entry beside the rule: the half-converted shape.
                    // `retread-wheels` carries no `#[serde(default)]`, so
                    // omitting the table entirely is still `[build.config]:
                    // missing field `retread-wheels`` (measured: gate 6166699);
                    // an EMPTY table plus a rule used to be
                    // `[build.config].wheels must list at least one wheel`
                    // (measured: gate 6169659) because the emptiness refusal ran
                    // above the expansion -- that was N27-RETREAD-190, it is
                    // fixed, and the arm that proves it is
                    // `a_derived_only_pack_initializes_over_the_real_transport`
                    // below. `tomli` resolves nothing at initialize (the
                    // sequence stops there) and its key cannot collide with
                    // `alpha`/`beta`, so the request stays offline and the
                    // derived entries land beside a typed one.
                    "retread-wheels": {
                        "tomli": { "version": "==2.0.1" }
                    },
                    "retread-git-sources": {
                        "fixture": { "url": tree.to_string_lossy(), "rev": &rev }
                    },
                    "retread-subpackages": {
                        "fixture": {
                            "from": "fixture",
                            "glob": "source/*",
                            "expect": 2,
                            "exclude": ["gamma"]
                        }
                    }
                }
            }
        }),
    ];

    // THE ASSERTION THAT CATCHES THE DEFECT is inside drive_backend: every
    // stdout line must parse as a JSON-RPC 2.0 frame.
    let (responses, stderr) = drive_backend(&requests);
    assert_eq!(
        responses.len(),
        2,
        "expected one frame per request and NOTHING else on stdout; got: {responses:#?}\
         \n--- stderr ---\n{stderr}"
    );
    assert!(
        responses[1].get("error").is_none(),
        "initialize must succeed with a rule declared -- if this errors the row \
         below never printed and the stdout assertion proved nothing: {:#?}",
        responses[1],
    );

    // The row still has to be EMITTED, on stderr. Without this half the defect
    // could be "fixed" by deleting the print, which would trade a corrupt
    // channel for a silent expansion -- and the expansion is derived, so the
    // row is the only evidence of which subpackages a pack actually built.
    let row = stderr
        .lines()
        .find(|l| l.contains("### PACK SUBPACKAGES"))
        .unwrap_or_else(|| {
            panic!("no `### PACK SUBPACKAGES` row on stderr\n--- stderr ---\n{stderr}")
        });
    let expected: Vec<String> = vec![
        "from=fixture".to_string(),
        format!("rev={rev}"),
        "rule=source/*".to_string(),
        "found=3".to_string(),
        "names=alpha,beta,gamma".to_string(),
        "included=2".to_string(),
        "excluded=gamma".to_string(),
        "skipped=docs".to_string(),
    ];
    for want in &expected {
        assert!(
            row.contains(want.as_str()),
            "the stderr row is missing `{want}`: {row}"
        );
    }

    // N27-RETREAD-190: the wheel-set census for this same pack, on the same
    // channel. One typed entry plus a two-subpackage rule is 1 + 2 = 3.
    let census = stderr
        .lines()
        .find(|l| l.contains("### PACK WHEEL SET"))
        .unwrap_or_else(|| {
            panic!("no `### PACK WHEEL SET` row on stderr\n--- stderr ---\n{stderr}")
        });
    for want in ["declared=1", "derived=2", "total=3"] {
        assert!(
            census.contains(want),
            "the wheel-set row is missing `{want}`: {census}"
        );
    }

    std::fs::remove_dir_all(&tmp).ok();
}

/// N27-RETREAD-190, the REAL-TRANSPORT half of the guard.
///
/// THE DEFECT THIS EXISTS FOR. `Handler::initialize`'s
/// `config.retread_wheels.is_empty()` refusal stood ~215 lines ABOVE the
/// `[retread-subpackages]` expansion, so a pack whose wheel set is ENTIRELY
/// derived -- an empty `[retread-wheels]` plus one rule, which is the end state
/// `crate::subpackages`'s module doc says the capability exists to reach and the
/// shape the operator's "the pack manifest points at its requirements file and
/// retread derives the rest" directive asks for -- was refused
/// `[build.config].wheels must list at least one wheel` before a single rule was
/// read. It was found by the arm above (gate 6169659), not by reading.
///
/// WHY IT IS A SEPARATE ARM AND NOT AN EXTRA ASSERTION ON ITS NEIGHBOUR. The
/// neighbour's pack declares a typed entry, so its wheel table is never empty
/// and it cannot witness this ordering at all -- that is exactly why the defect
/// survived HOTFIX-180's landing gate. Only a pack with NO typed entry reaches
/// the moved check.
///
/// WHY IT IS NOT `#[ignore]`. Same reason as its neighbour: the tree is a local
/// git repo the fixture creates, `ensure_git_checkout` clones over the local
/// filesystem, and the sequence stops at `initialize`. Offline, about a second.
///
/// FALSIFIABILITY: move the emptiness check back above the expansion and this
/// arm fails at `initialize must succeed`, with the refusal on the wire.
#[test]
fn a_derived_only_pack_initializes_over_the_real_transport() {
    let (tmp, tree, cache, rev) = subpackage_fixture_tree("derived-only");

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
                "cacheDirectory": &cache,
                "configuration": {
                    // NOT scaffolding: an empty table is the whole point. The
                    // key must still be present -- `retread-wheels` carries no
                    // `#[serde(default)]`, so omitting it is a parse refusal,
                    // a different failure that would mask this one.
                    "retread-wheels": {},
                    "retread-git-sources": {
                        "fixture": { "url": tree.to_string_lossy(), "rev": &rev }
                    },
                    "retread-subpackages": {
                        "fixture": {
                            "from": "fixture",
                            "glob": "source/*",
                            "expect": 2,
                            "exclude": ["gamma"]
                        }
                    }
                }
            }
        }),
    ];

    let (responses, stderr) = drive_backend(&requests);
    assert_eq!(
        responses.len(),
        2,
        "expected one frame per request and NOTHING else on stdout; got: {responses:#?}\
         \n--- stderr ---\n{stderr}"
    );
    assert!(
        responses[1].get("error").is_none(),
        "initialize must succeed for a pack whose wheel set is entirely derived \
         (N27-RETREAD-190): {:#?}\n--- stderr ---\n{stderr}",
        responses[1],
    );

    // Both rows, on stderr, with the arities the tree really produces: nothing
    // was declared, two subpackages were derived, and the set is those two.
    let census = stderr
        .lines()
        .find(|l| l.contains("### PACK WHEEL SET"))
        .unwrap_or_else(|| {
            panic!("no `### PACK WHEEL SET` row on stderr\n--- stderr ---\n{stderr}")
        });
    for want in ["declared=0", "derived=2", "total=2"] {
        assert!(
            census.contains(want),
            "the wheel-set row is missing `{want}`: {census}"
        );
    }
    assert!(
        stderr.lines().any(|l| l.contains("### PACK SUBPACKAGES")
            && l.contains("included=2")
            && l.contains("excluded=gamma")),
        "the expansion evidence row must still be emitted for a derived-only pack\
         \n--- stderr ---\n{stderr}"
    );

    std::fs::remove_dir_all(&tmp).ok();
}
