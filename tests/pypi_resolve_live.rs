//! Live tests for the PEP 503 simple-index resolver.
//!
//! Two halves:
//! * fast, runs by default — small public PyPI package, exercises the
//!   resolve -> fetch -> parse path through `pixi_build_retread::pypi`
//!   and confirms the resolver works against any standard simple index.
//! * `#[ignore]`'d — verifies the index-agnostic story by hitting
//!   `https://pypi.nvidia.com` (provides `#sha256=` fragments) AND
//!   `https://py.mujoco.org` (does NOT). Both must round-trip cleanly.

use pixi_build_retread::pypi::{WheelTarget, resolve};
use std::str::FromStr;
use uv_pep508::uv_pep440::VersionSpecifiers;

fn linux64(py: &str) -> WheelTarget {
    WheelTarget {
        python_version: py.to_string(),
        conda_subdir: "linux-64".to_string(),
    }
}

/// `resolve` takes parsed PEP 440 specifiers; the tests express the pin as a
/// bare version, so wrap it as an exact `==` spec.
fn eq(version: &str) -> VersionSpecifiers {
    VersionSpecifiers::from_str(&format!("=={version}")).unwrap()
}

#[tokio::test]
async fn resolves_small_public_pypi_package() {
    // `tomli` is a stable, tiny, pure-python wheel. The resolve targets
    // public PyPI and confirms the simple-index round-trip works against
    // the canonical index (not just NVIDIA's).
    let r = resolve(
        "https://pypi.org/simple/",
        "tomli",
        &eq("2.0.1"),
        &linux64("3.11"),
    )
    .await
    .expect("resolve tomli 2.0.1");
    assert_eq!(r.filename, "tomli-2.0.1-py3-none-any.whl");
    assert!(r.sha256.is_some(), "public PyPI advertises sha256");
    // The URL itself may carry a `#sha256=...` fragment, so check the path.
    assert!(r.url.path().ends_with(".whl"));
}

#[tokio::test]
#[ignore = "hits pypi.nvidia.com; bandwidth-light but external dep"]
async fn resolves_isaacsim_with_pep440_normalized_version() {
    // The user writes `version = "==5.1.0"` (the natural form). The index
    // serves wheels at `isaacsim-5.1.0.0-cp311-...` (four-component, with
    // trailing zero). PEP 440 normalizes trailing zeros, so these must be
    // treated as equivalent.
    let r = resolve(
        "https://pypi.nvidia.com",
        "isaacsim",
        &eq("5.1.0"),
        &linux64("3.11"),
    )
    .await
    .expect("`5.1.0` should resolve to the `5.1.0.0` wheel");
    assert!(r.filename.starts_with("isaacsim-5.1.0.0"));
}

#[tokio::test]
#[ignore = "hits pypi.nvidia.com; bandwidth-light but external dep"]
async fn resolves_isaacsim_kernel_via_nvidia_index() {
    let r = resolve(
        "https://pypi.nvidia.com",
        "isaacsim-kernel",
        &eq("5.1.0.0"),
        &linux64("3.11"),
    )
    .await
    .expect("resolve isaacsim-kernel via nvidia index");
    assert_eq!(
        r.filename,
        "isaacsim_kernel-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl"
    );
    assert_eq!(
        r.sha256.as_deref(),
        Some("f981290a70ad89f169fb9c47bd55833665677879ebc46ed3992756c617ac02fc"),
        "NVIDIA index advertises sha256 in URL fragment"
    );
}

#[tokio::test]
#[ignore = "hits py.mujoco.org; bandwidth-light but external dep"]
async fn resolves_mujoco_via_index_without_sha256() {
    // py.mujoco.org follows PEP 503 but does not include `#sha256=` in URL
    // fragments. Our resolver must still pick a wheel; sha256 falls back to
    // being computed by `fetch_wheel` at download time.
    //
    // mujoco wheels also use PEP 425 compressed tag sets
    // (`manylinux_2_27_x86_64.manylinux_2_28_x86_64`) -- another path the
    // resolver must handle.
    //
    // py.mujoco.org ships only the latest dev wheels; we resolve against
    // that single available version + a Python version the wheels are built
    // for. If this test starts failing because the index churns, refresh
    // the version/python combo below.
    let r = resolve(
        "https://py.mujoco.org",
        "mujoco",
        &eq("3.9.0.dev920201340"),
        &linux64("3.13"),
    )
    .await
    .expect("resolve mujoco via py.mujoco.org");
    assert!(r.filename.contains("cp313"), "got: {}", r.filename);
    assert!(r.filename.contains("manylinux"), "got: {}", r.filename);
    assert!(r.filename.contains("x86_64"), "got: {}", r.filename);
    assert!(
        r.sha256.is_none(),
        "py.mujoco.org does not advertise sha256 fragments"
    );
}
