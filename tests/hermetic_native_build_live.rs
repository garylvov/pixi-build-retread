//! Live end-to-end coverage for hermetic native sdist builds.
//!
//! This test is ignored by default because it needs network access to PyPI
//! and conda-forge plus `uv` and `rattler-build >=0.70`. The host must be
//! Linux x86_64 with glibc newer than 2.17 so the declared 2.17 target floor
//! forces Retread's hermetic retry. A host compiler is deliberately not a
//! prerequisite: the provisioned conda compiler must perform the native build.
//!
//! Run it explicitly with:
//!
//! ```bash
//! env -u SLURM_JOB_ID rustup run 1.97.0 cargo test \
//!   --test hermetic_native_build_live closure_blocking_evdev_entry_builds_hermetically \
//!   -- --include-ignored --test-threads=4 --nocapture
//! ```

use std::ffi::OsString;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use pixi_build_retread::handler::Handler;
use pixi_build_retread::wheel::read_metadata;
use serde_json::json;
use sha2::{Digest, Sha256};

const TARGET_GLIBC: &str = "2.17";
const TARGET_PLATFORM_TAG: &str = "manylinux_2_17_x86_64";

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "retread-hermetic-evdev-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create hermetic live-test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!("preserving failed live-test directory {}", self.0.display());
            return;
        }
        fs::remove_dir_all(&self.0).ok();
    }
}

struct ScopedEnvironment(Vec<(&'static str, Option<OsString>)>);

impl ScopedEnvironment {
    fn set(values: Vec<(&'static str, OsString)>) -> Self {
        let mut previous = Vec::with_capacity(values.len());
        for (key, value) in values {
            previous.push((key, std::env::var_os(key)));
            // SAFETY: this ignored integration-test binary contains one test.
            // It mutates its environment before spawning build work and does
            // not run concurrent code that reads these variables.
            unsafe { std::env::set_var(key, value) };
        }
        Self(previous)
    }
}

impl Drop for ScopedEnvironment {
    fn drop(&mut self) {
        for (key, value) in self.0.drain(..).rev() {
            // SAFETY: all build work has completed before this guard drops,
            // under the same single-test condition described above.
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

fn require_on_path(tool: &str) {
    let available = Command::new(tool)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    assert!(
        available,
        "live hermetic build test requires `{tool}` on PATH"
    );
}

fn collect_evdev_wheels(path: &Path, wheels: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries {
        let entry = entry.expect("read live-test directory entry");
        let file_type = entry.file_type().expect("read live-test file type");
        let path = entry.path();
        if file_type.is_dir() {
            collect_evdev_wheels(&path, wheels);
        } else if file_type.is_file()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.to_ascii_lowercase().starts_with("evdev-") && name.ends_with(".whl")
                })
        {
            wheels.push(path);
        }
    }
}

fn assert_native_archive_tag(
    wheel: &Path,
    patchelf: &Path,
    inspection_dir: &Path,
    test_root: &Path,
) {
    let metadata = read_metadata(wheel).expect("read built evdev wheel metadata");
    assert_eq!(metadata.name.to_ascii_lowercase(), "evdev");
    assert_eq!(metadata.version, "1.7.1");
    assert!(!metadata.is_pure_python, "evdev wheel must be native");

    let file = fs::File::open(wheel).expect("open built wheel archive");
    let mut archive = zip::ZipArchive::new(file).expect("parse built wheel archive");
    let mut has_native_payload = false;
    let mut wheel_metadata = None;
    fs::create_dir_all(inspection_dir).expect("create native-member inspection directory");
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).expect("read wheel archive entry");
        if entry.name().ends_with(".so") {
            has_native_payload = true;
            let member = entry.name().to_string();
            let mut payload = Vec::new();
            entry
                .read_to_end(&mut payload)
                .expect("read native wheel member");
            let test_root = test_root.to_string_lossy();
            assert!(
                !payload
                    .windows(test_root.len())
                    .any(|window| window == test_root.as_bytes()),
                "native wheel member `{member}` embeds the live-test cache root `{test_root}`"
            );
            assert!(
                !payload
                    .windows(b"hermetic-build-envs".len())
                    .any(|window| window == b"hermetic-build-envs"),
                "native wheel member `{member}` embeds the hermetic environment cache path"
            );
            let extracted = inspection_dir.join(
                Path::new(&member)
                    .file_name()
                    .expect("native member has a filename"),
            );
            fs::write(&extracted, payload).expect("write native member for RPATH inspection");
            let output = Command::new(patchelf)
                .arg("--print-rpath")
                .arg(&extracted)
                .output()
                .expect("run provisioned patchelf");
            assert!(
                output.status.success(),
                "patchelf failed to inspect native member `{member}`: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let rpath = String::from_utf8(output.stdout).expect("patchelf RPATH is UTF-8");
            let absolute: Vec<&str> = rpath
                .trim()
                .split(':')
                .filter(|entry| !entry.is_empty() && Path::new(entry).is_absolute())
                .collect();
            assert!(
                absolute.is_empty(),
                "native wheel member `{member}` retains absolute RPATH entries {absolute:?}"
            );
        }
        if entry.name().ends_with(".dist-info/WHEEL") {
            let mut contents = String::new();
            entry
                .read_to_string(&mut contents)
                .expect("read WHEEL metadata");
            wheel_metadata = Some(contents);
        }
    }
    assert!(
        has_native_payload,
        "evdev wheel should contain shared objects"
    );

    let wheel_metadata = wheel_metadata.expect("wheel archive contains .dist-info/WHEEL");
    let tags: Vec<&str> = wheel_metadata
        .lines()
        .filter_map(|line| line.strip_prefix("Tag: "))
        .collect();
    assert!(!tags.is_empty(), "WHEEL metadata should advertise a tag");
    assert!(
        tags.iter()
            .all(|tag| tag.rsplit('-').next() == Some(TARGET_PLATFORM_TAG)),
        "all WHEEL tags must use `{TARGET_PLATFORM_TAG}`, got {tags:?}"
    );
}

#[tokio::test]
#[ignore = "live: needs PyPI/conda-forge, uv, and rattler-build >=0.70"]
async fn closure_blocking_evdev_entry_builds_hermetically() {
    assert_eq!(std::env::consts::OS, "linux", "test requires Linux");
    assert_eq!(std::env::consts::ARCH, "x86_64", "test requires x86_64");
    require_on_path("uv");
    require_on_path("rattler-build");

    let test_dir = TestDirectory::new();
    let _environment = ScopedEnvironment::set(vec![
        ("RETREAD_HERMETIC_BUILDS", OsString::from("1")),
        ("RETREAD_NO_REPLAY", OsString::from("1")),
        ("RETREAD_FAST_TMP", OsString::from("off")),
        (
            "RETREAD_CACHE_DIR",
            test_dir.path().join("retread-cache").into_os_string(),
        ),
        (
            "RETREAD_WHEEL_STORE",
            test_dir.path().join("wheel-store").into_os_string(),
        ),
        (
            "UV_CACHE_DIR",
            test_dir.path().join("uv-cache").into_os_string(),
        ),
    ]);

    let workspace = test_dir.path().join("workspace");
    let pack = workspace.join("pack");
    fs::create_dir_all(&pack).expect("create live-test pack");
    fs::write(
        workspace.join("pixi.toml"),
        r#"[workspace]
channels = ["conda-forge"]
platforms = [{ platform = "linux-64", glibc = "2.17" }]

[dependencies]
hermetic-live-pack = { path = "./pack" }
"#,
    )
    .expect("write live-test workspace manifest");
    fs::write(
        pack.join("pixi.toml"),
        r#"[package]
name = "hermetic-live-pack"
version = "1.0.0"
"#,
    )
    .expect("write live-test pack manifest");
    let handler = Handler::new();
    handler
        .dispatch(
            "initialize".to_string(),
            json!({
                "manifestPath": pack.join("pixi.toml"),
                "sourceDirectory": &pack,
                "workspaceDirectory": &workspace,
                "cacheDirectory": test_dir.path().join("handler-cache"),
                "configuration": {
                    "retread-wheels": { "evdev": { "version": "==1.7.1" } },
                    "retread-python": "3.11",
                    "retread-auto-bundle": true,
                    "retread-auto-route": false,
                    "retread-keep-pypi": ["evdev"],
                    "retread-name-map": { "evdev": "" },
                    "retread-relax": "none",
                    "retread-hermetic": true
                }
            }),
        )
        .await
        .expect("initialize live-test handler");
    let outputs = handler
        .dispatch(
            "conda/outputs".to_string(),
            json!({
                "hostPlatform": "linux-64",
                "buildPlatform": "linux-64",
                "channels": ["https://prefix.dev/conda-forge"],
                "workDirectory": test_dir.path().join("work"),
                "variantConfiguration": { "python": ["3.11"] }
            }),
        )
        .await
        .expect("resolve and materialize the evdev sdist fallback");
    assert!(
        outputs
            .get("outputs")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|outputs| !outputs.is_empty()),
        "conda/outputs should return the materialized bundle: {outputs:#}"
    );

    let mut evdev_wheels = Vec::new();
    collect_evdev_wheels(test_dir.path(), &mut evdev_wheels);
    assert!(
        !evdev_wheels.is_empty(),
        "bundled-sdist fallback should materialize an evdev wheel under {}",
        test_dir.path().display()
    );
    let exact_tagged = evdev_wheels.iter().find(|wheel| {
        wheel
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.contains(TARGET_PLATFORM_TAG))
    });
    assert!(
        exact_tagged.is_some(),
        "expected a sysroot-derived `{TARGET_PLATFORM_TAG}` evdev wheel, found {evdev_wheels:?}; \
         the test host must have glibc newer than {TARGET_GLIBC} to force the hermetic path"
    );
    let marker_path = fs::read_dir(test_dir.path().join("retread-cache/hermetic-build-envs/v6"))
        .expect("read hermetic environment cache")
        .map(|entry| entry.expect("read hermetic environment entry").path())
        .find(|path| path.join("complete.json").is_file())
        .expect("find solve-digest-keyed hermetic environment")
        .join("complete.json");
    let marker: serde_json::Value =
        serde_json::from_slice(&fs::read(&marker_path).unwrap_or_else(|error| {
            panic!(
                "read hermetic environment marker {}: {error}",
                marker_path.display()
            )
        }))
        .expect("parse hermetic environment marker");
    assert_eq!(marker["selected_sysroot"], json!([2, 17]));
    assert_eq!(marker["platform_tag"], TARGET_PLATFORM_TAG);
    let sysroot = Path::new(
        marker["sysroot_path"]
            .as_str()
            .expect("hermetic marker records its sysroot path"),
    );
    let build_prefix = sysroot
        .parent()
        .and_then(Path::parent)
        .expect("sysroot is nested under the hermetic build prefix");
    let patchelf = build_prefix.join("bin/patchelf");
    assert!(
        patchelf.is_file(),
        "hermetic environment should provision {}",
        patchelf.display()
    );
    assert_native_archive_tag(
        exact_tagged.unwrap(),
        &patchelf,
        &test_dir.path().join("native-member-inspection"),
        test_dir.path(),
    );
}

async fn run_deterministic_native_build(root: &Path, work_name: &str) -> PathBuf {
    let workspace = root.join("determinism-workspace");
    let pack = workspace.join("pack");
    let native = pack.join("native");
    fs::create_dir_all(&native).expect("create deterministic native source");
    fs::write(
        workspace.join("pixi.toml"),
        r#"[workspace]
channels = ["conda-forge"]
platforms = [{ platform = "linux-64", glibc = "2.17" }]

[dependencies]
determinism-pack = { path = "./pack" }
"#,
    )
    .expect("write determinism workspace");
    fs::write(
        pack.join("pixi.toml"),
        r#"[package]
name = "determinism-pack"
version = "1.0.0"
"#,
    )
    .expect("write determinism pack");
    fs::write(
        native.join("pyproject.toml"),
        r#"[build-system]
requires = ["setuptools==80.9.0", "wheel==0.45.1"]
build-backend = "setuptools.build_meta"
"#,
    )
    .expect("write deterministic pyproject");
    fs::write(
        native.join("setup.py"),
        r#"from setuptools import Extension, setup
setup(name="deterministic-native", version="1.0.0",
      ext_modules=[Extension("deterministic_native", ["module.cpp"], language="c++")])
"#,
    )
    .expect("write deterministic setup.py");
    fs::write(
        native.join("module.cpp"),
        r#"#include <Python.h>
#include <string>
static PyObject* ping(PyObject*, PyObject*) {
    std::string value("deterministic-native");
    return PyUnicode_FromStringAndSize(value.data(), value.size());
}
static PyMethodDef methods[] = {{"ping", ping, METH_NOARGS, nullptr}, {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "deterministic_native", nullptr, -1, methods};
PyMODINIT_FUNC PyInit_deterministic_native() { return PyModule_Create(&module); }
"#,
    )
    .expect("write deterministic C++ extension");

    let handler = Handler::new();
    handler
        .dispatch(
            "initialize".to_string(),
            json!({
                "manifestPath": pack.join("pixi.toml"),
                "sourceDirectory": &pack,
                "workspaceDirectory": &workspace,
                "cacheDirectory": root.join(format!("handler-cache-{work_name}")),
                "configuration": {
                    "retread-wheels": {
                        "deterministic-native": { "path": "./native" }
                    },
                    "retread-python": "3.11",
                    "retread-auto-bundle": true,
                    "retread-auto-route": false,
                    "retread-keep-pypi": ["deterministic-native"],
                    "retread-name-map": { "deterministic-native": "" },
                    "retread-relax": "none",
                    "retread-hermetic": true
                }
            }),
        )
        .await
        .expect("initialize determinism handler");
    handler
        .dispatch(
            "conda/outputs".to_string(),
            json!({
                "hostPlatform": "linux-64",
                "buildPlatform": "linux-64",
                "channels": ["https://prefix.dev/conda-forge"],
                "workDirectory": root.join(work_name),
                "variantConfiguration": { "python": ["3.11"] }
            }),
        )
        .await
        .expect("materialize deterministic native wheel");
    let cache_root = PathBuf::from(
        std::env::var_os("RETREAD_CACHE_DIR").expect("determinism test sets RETREAD_CACHE_DIR"),
    );
    let mut wheels = Vec::new();
    collect_named_wheels(
        &cache_root.join("built-wheels"),
        "deterministic_native-",
        &mut wheels,
    );
    wheels
        .into_iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.contains(TARGET_PLATFORM_TAG))
        })
        .max_by_key(|path| fs::metadata(path).and_then(|meta| meta.modified()).ok())
        .expect("find deterministic hermetic wheel")
}

fn collect_named_wheels(path: &Path, prefix: &str, wheels: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries {
        let entry = entry.expect("read deterministic wheel tree");
        let path = entry.path();
        if entry
            .file_type()
            .expect("read wheel tree file type")
            .is_dir()
        {
            collect_named_wheels(&path, prefix, wheels);
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(prefix) && name.ends_with(".whl"))
        {
            wheels.push(path);
        }
    }
}

#[tokio::test]
#[ignore = "live: two cold wheel-cache native rebuilds need PyPI/conda-forge"]
async fn hermetic_native_cold_rebuilds_are_byte_identical() {
    assert_eq!(std::env::consts::OS, "linux");
    assert_eq!(std::env::consts::ARCH, "x86_64");
    require_on_path("uv");
    require_on_path("rattler-build");
    let test_dir = TestDirectory::new();
    let retread_cache = std::env::var_os("RETREAD_TEST_HERMETIC_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| test_dir.path().join("retread-cache"));
    let _environment = ScopedEnvironment::set(vec![
        ("RETREAD_HERMETIC_BUILDS", OsString::from("1")),
        ("RETREAD_NO_REPLAY", OsString::from("1")),
        ("RETREAD_FAST_TMP", OsString::from("off")),
        ("RETREAD_CACHE_DIR", retread_cache.clone().into_os_string()),
        (
            "RETREAD_WHEEL_STORE",
            test_dir.path().join("wheel-store").into_os_string(),
        ),
        (
            "UV_CACHE_DIR",
            test_dir.path().join("uv-cache").into_os_string(),
        ),
    ]);
    let first = run_deterministic_native_build(test_dir.path(), "work-one").await;
    let first_bytes = fs::read(&first).expect("read first cold wheel");
    let built_wheels = retread_cache.join("built-wheels");
    if built_wheels.exists() {
        fs::remove_dir_all(&built_wheels)
            .expect("clear built-wheel cache between deterministic builds");
    }
    for lock in [
        test_dir
            .path()
            .join("determinism-workspace/pack/retread-determinism-pack.lock.json"),
        test_dir
            .path()
            .join("determinism-workspace/pack/retread.lock.json"),
    ] {
        if lock.exists() {
            fs::remove_file(lock).expect("remove replay lock");
        }
    }
    let second = run_deterministic_native_build(test_dir.path(), "work-two").await;
    let second_bytes = fs::read(&second).expect("read second cold wheel");
    let first_hash = format!("{:x}", Sha256::digest(first_bytes));
    let second_hash = format!("{:x}", Sha256::digest(second_bytes));
    eprintln!("determinism SHA-256 #1: {first_hash}");
    eprintln!("determinism SHA-256 #2: {second_hash}");
    assert_eq!(first_hash, second_hash);
}
