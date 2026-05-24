//! End-to-end integration test: run a real `pixi install --dry-run`
//! against a workspace that combines `ros-humble-*` (conda) with
//! `isaacsim[all,extscache]` (PyPI, repacked by retread). This is the
//! literal scenario from prefix-dev/pixi#5230.
//!
//! Heavy. Requires `pixi` and `rattler-build` on PATH plus several
//! gigabytes of channel downloads. `#[ignore]`'d by default; run with:
//!
//! ```bash
//! cargo test --test e2e_ros_isaacsim -- --include-ignored --nocapture
//! ```
//!
//! What this proves vs. the snapshot tests:
//! - The backend speaks pixi-build correctly when pixi (not our own
//!   harness) is the JSON-RPC client.
//! - The PyPI resolver + extras expansion holds together for the full
//!   22-wheel isaacsim metapackage.
//! - The relax policy actually lets the conda solve converge with ros2
//!   sitting alongside.
//!
//! Failures here are the ones that matter. The override block in
//! `[package.build.config.retread-overrides]` documents which upstream conflicts retread
//! could NOT resolve on its own (e.g. pillow disagrees with itself across
//! isaacsim sub-wheels); update those as we learn what breaks.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn backend_binary() -> PathBuf {
    repo_root().join("target/release/pixi-build-retread")
}

fn tempdir() -> PathBuf {
    let dir = env::temp_dir().join(format!(
        "retread-e2e-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

const ROOT_MANIFEST: &str = r#"
[workspace]
name = "retread-e2e"
channels = [
    "https://prefix.dev/conda-forge",
    "https://prefix.dev/robostack-humble",
]
platforms = ["linux-64"]
preview = ["pixi-build"]

# The realistic conflict surface: a chunk of ros-humble that touches the
# same transitive deps isaacsim pins hard (opencv, numpy, pillow, scipy,
# packaging, ...). Without retread, the conda solve finishes but the PyPI
# solve fails because pixi forwards conda's versions as constraints into
# uv. Repacking isaacsim via retread eliminates that round-trip.
[dependencies]
python = "==3.11"
isaacsim = { path = "./isaacsim-repack" }
ros-humble-ros-core = "*"
ros-humble-rviz2 = "*"
ros-humble-rosbag2 = "*"
ros-humble-xacro = "*"
ros-humble-joint-state-publisher = "*"
ros-humble-tf2-ros = "*"
colcon-common-extensions = "*"
colcon-ros = "*"
"#;

const REPACK_MANIFEST: &str = r#"
[package]
name = "isaacsim"
version = "5.1.0"

[package.build]
backend = { name = "pixi-build-retread", version = "*", channels = ["https://prefix.dev/garylvov", "https://prefix.dev/conda-forge"] }

[package.build.config]
retread-relax        = "minor"
retread-build-number = 0
# Keep these on the conda side instead of auto-bundling. ABI-sensitive
# C extensions where the workspace's conda version must be the one
# Python imports.
retread-conda-deps = ["numpy", "scipy", "torch", "pytorch", "pandas"]
# retread-drop-deps would go here for deps that have no conda
# counterpart. Windows-only shims (idna-ssl, pywin32, ...) are
# auto-dropped on non-Windows targets, so nothing to add for those.

[package.build.config.retread-wheels]
isaacsim = { version = "==5.1.0", index = "https://pypi.nvidia.com", extras = ["all", "extscache"] }

# retread-overrides intentionally absent. With retread-auto-bundle
# on (default), `aiodns`, `qdldl`, etc. are auto-fetched from PyPI
# and pip-installed into the bundle -- no manual overrides needed.
# numpy stays on the conda side via retread-conda-deps above.

[package.build.config.retread-name-map]
opencv-python-headless = "py-opencv"
# PyPI "torch" -> conda-forge "pytorch". Required because the
# retread-conda-deps list above keeps torch on the conda side.
torch = "pytorch"
"#;

fn ensure_release_build() {
    if backend_binary().exists() {
        return;
    }
    eprintln!("building pixi-build-retread (release)...");
    let status = Command::new("cargo")
        .args(["build", "--release", "--bin", "pixi-build-retread"])
        .current_dir(repo_root())
        .status()
        .expect("cargo build failed to spawn");
    assert!(status.success(), "cargo build --release failed");
}

fn require_on_path(tool: &str) {
    let found = Command::new("which")
        .arg(tool)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !found {
        panic!(
            "the e2e test requires `{tool}` on PATH (try `pixi run -- {tool} --version` or \
             activate the pixi env at the project root)"
        );
    }
}

#[test]
#[ignore = "heavy: needs pixi + rattler-build on PATH, downloads several GB"]
fn ros2_humble_plus_isaacsim_solves() {
    require_on_path("pixi");
    require_on_path("rattler-build");
    ensure_release_build();

    let work = tempdir();
    let repack = work.join("isaacsim-repack");
    fs::create_dir_all(&repack).unwrap();
    fs::write(work.join("pixi.toml"), ROOT_MANIFEST).unwrap();
    fs::write(repack.join("pixi.toml"), REPACK_MANIFEST).unwrap();

    eprintln!("workspace: {}", work.display());
    eprintln!("backend  : {}", backend_binary().display());

    // We want the backend to find rattler-build at runtime; piggyback on the
    // caller's PATH (the test runner is expected to be in an env that has it).
    let override_spec = format!(
        "pixi-build-retread={}",
        backend_binary().to_string_lossy()
    );

    let output = Command::new("pixi")
        .args(["lock", "--no-progress"])
        .env("PIXI_BUILD_BACKEND_OVERRIDE", &override_spec)
        // Stream backend stderr so failures are diagnosable.
        .env("PIXI_BUILD_RETREAD_LOG", "debug")
        .current_dir(&work)
        .output()
        .expect("`pixi lock` failed to spawn");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        panic!(
            "`pixi lock` failed (exit {:?})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
            output.status.code(),
        );
    }
    assert!(
        work.join("pixi.lock").exists(),
        "pixi.lock should be written on success\n--- stderr ---\n{stderr}"
    );

    // Leave the tempdir around on success so it can be inspected manually.
    eprintln!(
        "ok. Inspect resolved lock at {}/pixi.lock",
        work.display()
    );
}
