//! v2.0.0 courier: producer-side staging.
//!
//! Stages the bundle's built + relax-changed ("shadow") wheels + the
//! generated meta-wheel + the committed lock into a staging dir, WITHOUT
//! touching the consumer pixi.toml. The courier conda package then ships
//! these as data and installs them at link time via `retread install`.
//!
//! FROZEN CONTRACT (phase 0): [`CourierStaged`] is the output consumed by
//! `build_one` (WS-C) and the [`stage`] signature is what WS-C compiles
//! against. WS-A implements the body. Do not change these shapes without a
//! stop-the-world re-freeze.

use std::collections::HashSet;
use std::path::Path;

use crate::config::RetreadConfig;
use crate::emit_pypi::EmitWheel;
use crate::lock::RetreadLock;

/// Everything `build_one` needs to build the courier recipe + write the lock.
pub struct CourierStaged {
    /// `file://` URLs of the staged artifacts (shipped wheels + the lock
    /// json) for the courier recipe's `source:` list.
    pub source_urls: Vec<String>,
    /// The assembled, ready-to-write install lock.
    pub lock: RetreadLock,
    /// The solved conda run-deps for the courier recipe's `requirements.run`
    /// (uv + pixi-build-retread are appended by `build_courier_recipe`).
    pub run_deps: Vec<String>,
}

/// Stage the courier artifacts. Built wheels (`must_ship()`) AND index
/// wheels whose metadata relax CHANGED are written to `staging_dir` (they
/// ship in the conda package as `Origin::Built`); unchanged index wheels are
/// recorded `Origin::Index` with their upstream url + sha256. Builds the
/// `<bundle>-pypi` meta-wheel (the lock's single `root_requirement`),
/// collects prerelease pins, computes `inputs_hash`, and writes the lock
/// json into `staging_dir`. NEVER writes the consumer manifest.
#[allow(unused_variables, clippy::too_many_arguments)]
pub async fn stage(
    config: &RetreadConfig,
    bundle_name: &str,
    version: &str,
    python: &str,
    emit_wheels: &[EmitWheel],
    conda_capable: &HashSet<String>,
    run_deps: &[String],
    index_urls: &[String],
    source_dir: &Path,
    staging_dir: &Path,
) -> anyhow::Result<CourierStaged> {
    todo!("WS-A: stage built/shadow wheels + meta-wheel + lock into staging_dir")
}
