//! Live integration tests that hit the network. Run with:
//!
//! ```bash
//! cargo test --test wheel_fetch_live -- --include-ignored
//! ```
//!
//! Two flavors:
//!
//! * `fetch_small_pure_python_wheel` — pulls a tiny pure-python wheel from
//!   the canonical PyPI mirror and exercises the full
//!   `fetch_wheel` -> zip-extract -> `parse_metadata` path the backend runs
//!   in `conda/outputs` and `conda/build_v1`. Cheap enough to run by default.
//!
//! * `fetch_isaacsim_kernel_end_to_end` — does the same but against the real
//!   `pypi.nvidia.com` isaacsim-kernel wheel (~70 MB), then drives our
//!   relax pipeline and recipe generator on the live `Requires-Dist`. This
//!   is the closest thing to "what the backend actually does at request
//!   time." Gated behind `#[ignore]` because it costs bandwidth; the
//!   committed `tests/fixtures/*.METADATA.txt` snapshots cover the same
//!   ground without network for normal test runs.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use pixi_build_retread::config::{RelaxPolicy, RetreadConfig};
use pixi_build_retread::recipe::{BundleSource, build_bundle_recipe, to_yaml};
use pixi_build_retread::wheel::{fetch_wheel, read_metadata};

fn tempdir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("retread-test-{label}-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[tokio::test]
async fn fetch_small_pure_python_wheel() {
    // `tomli` is small (~12 KB), pure-python, and stable. We use it purely
    // to exercise the fetch + zip + METADATA parse code path the backend
    // runs at request time, not the relax math (which is covered by the
    // snapshot tests).
    let url: url::Url = "https://files.pythonhosted.org/packages/97/75/10a9ebee3fd790d20926a90a2547f0bf78f371b2f13aa822c759680ca7b9/tomli-2.0.1-py3-none-any.whl"
        .parse()
        .unwrap();
    let dir = tempdir("tomli");
    let path = fetch_wheel(&url, None, &dir).await.expect("fetch tomli");
    assert!(path.exists(), "downloaded wheel should exist on disk");

    let meta = read_metadata(&path).expect("parse tomli METADATA");
    assert_eq!(meta.name.to_ascii_lowercase(), "tomli");
    assert!(meta.is_pure_python, "tomli is py3-none-any");
    assert!(
        !meta.sha256.is_empty(),
        "fetch_wheel + read_metadata should populate sha256"
    );

    fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
#[ignore = "downloads ~70MB from pypi.nvidia.com; run with --include-ignored"]
async fn fetch_isaacsim_kernel_end_to_end() {
    // The full pipeline the backend runs for one wheel: download via
    // `fetch_wheel`, parse via `read_metadata`, then generate the recipe.
    // Asserts the same compatibility properties as the snapshot tests, but
    // against a wheel fetched live -- guards against regressions in the
    // fetch/extract path that snapshot tests don't reach.
    let url: url::Url = "https://pypi.nvidia.com/isaacsim-kernel/isaacsim_kernel-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl"
        .parse()
        .unwrap();

    let dir = tempdir("isaacsim_kernel");
    let path = fetch_wheel(&url, None, &dir)
        .await
        .expect("fetching isaacsim-kernel wheel");
    let meta = read_metadata(&path).expect("parsing live isaacsim-kernel METADATA");

    assert_eq!(
        meta.name.to_ascii_lowercase().replace('_', "-"),
        "isaacsim-kernel"
    );
    assert!(!meta.is_pure_python, "isaacsim-kernel is platform-specific");

    let config = RetreadConfig {
        retread_wheels: BTreeMap::new(),
        relax: RelaxPolicy::Minor,
        overrides: BTreeMap::new(),
        name_map: BTreeMap::new(),
        build_number: 0,
        drop_deps: Vec::new(),
        auto_bundle: false,
        conda_deps: Vec::new(),
        git_sources: std::collections::BTreeMap::new(),
        python: None,
    };
    let recipe = build_bundle_recipe(
        "isaacsim-kernel",
        &[BundleSource {
            pypi_name: &meta.name,
            url: &url,
            metadata: &meta,
        }],
        &config,
        "3.11",
        None,
    )
    .unwrap();
    let yaml = to_yaml(&recipe).unwrap();

    // Same compatibility assertion as the snapshot test, but on live data.
    assert!(
        yaml.lines().any(|l| {
            let t = l.trim_start().trim_start_matches("- ").trim();
            t.starts_with("numpy >=1.26,<")
        }),
        "live wheel should yield widened numpy range:\n{yaml}"
    );

    // No exact-version pins should remain anywhere under minor relax.
    let exact_pin_lines: Vec<&str> = yaml
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            // Skip the package version line and python pin.
            if t.starts_with("version:") || t.contains("python ") {
                return false;
            }
            t.starts_with("- ") && t.contains(" ==")
        })
        .collect();
    assert!(
        exact_pin_lines.is_empty(),
        "minor relax should produce no `name ==X` lines, found: {exact_pin_lines:#?}\nrecipe:\n{yaml}"
    );

    fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn cached_wheel_not_redownloaded() {
    // A second fetch_wheel call with the same URL should reuse the on-disk
    // file rather than re-hitting the network. Verifies the cache check that
    // matters for repeated `conda/outputs` calls on the same workspace.
    let url: url::Url = "https://files.pythonhosted.org/packages/97/75/10a9ebee3fd790d20926a90a2547f0bf78f371b2f13aa822c759680ca7b9/tomli-2.0.1-py3-none-any.whl"
        .parse()
        .unwrap();
    let dir = tempdir("tomli-cache");
    let first = fetch_wheel(&url, None, &dir).await.unwrap();
    let first_mtime = fs::metadata(&first).unwrap().modified().unwrap();

    let second = fetch_wheel(&url, None, &dir).await.unwrap();
    let second_mtime = fs::metadata(&second).unwrap().modified().unwrap();

    assert_eq!(first, second, "same path returned");
    assert_eq!(
        first_mtime, second_mtime,
        "cached wheel should not have been overwritten"
    );

    fs::remove_dir_all(&dir).ok();
}
