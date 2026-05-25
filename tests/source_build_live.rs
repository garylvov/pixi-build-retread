//! Live integration tests for `pip wheel` source-building, including
//! the build-isolation gotcha.
//!
//! The `sample_with_buildtime_dep` fixture declares `toml` as a
//! `[build-system].requires` entry and imports it from `setup.py`. If
//! retread's `build_wheel_from_path` ever flips back to passing
//! `--no-build-isolation`, this test fails with
//! `ModuleNotFoundError: No module named 'toml'` — exactly the failure
//! mode we hit against IsaacLab on first attempt.

use std::fs;
use std::path::PathBuf;

use pixi_build_retread::source_build::build_wheel_from_path;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn tempdir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "retread-source-build-test-{label}-{}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[tokio::test]
#[ignore = "requires pip on PATH; downloads build deps from PyPI"]
async fn builds_wheel_with_isolated_build_deps() {
    // The fixture's setup.py imports `toml`. If pip's build isolation
    // installs build-system requires (the default behavior), this
    // succeeds. If retread regresses to --no-build-isolation, this
    // fails with ModuleNotFoundError.
    let src = fixture("sample_with_buildtime_dep");
    let out = tempdir("isolated");
    let wheel = build_wheel_from_path(&src, &out, "3.11")
        .await
        .expect("build_wheel_from_path failed");

    assert!(wheel.exists(), "wheel should be produced");
    let fname = wheel.file_name().unwrap().to_str().unwrap();
    assert!(
        fname.starts_with("retread_sample-") && fname.ends_with(".whl"),
        "unexpected wheel filename: {fname}"
    );

    fs::remove_dir_all(&out).ok();
}
