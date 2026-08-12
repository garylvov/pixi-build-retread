use super::audit_report::format_packagespec;
use super::auto_bundle::{
    ExtraDepSource, conda_probe_spec, extra_dep_source_from_url, pep508_extra_dep,
    prefer_conda_match, validated_conda_route,
};
use super::*;

#[test]
fn direct_sdist_only_entry_reuses_closure_built_wheel_store_source() {
    let store = std::env::temp_dir().join(format!(
        "retread-closure-built-entry-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    let sha256 = "a".repeat(64);
    let filename = "evdev-1.7.1-cp311-cp311-manylinux_2_34_x86_64.whl";
    std::fs::create_dir_all(store.join(&sha256)).unwrap();
    std::fs::write(store.join(&sha256).join(filename), b"wheel fixture").unwrap();
    let entry = WheelEntry {
        version: Some("==1.7.1".to_string()),
        ..Default::default()
    };
    let source = crate::lock::SdistWheelSource {
        index: "https://pypi.org/simple/".to_string(),
        name: "evdev".to_string(),
        version: "1.7.1".to_string(),
        sdist_url: "https://files.pythonhosted.org/evdev-1.7.1.tar.gz#sha256=fixture".to_string(),
    };
    let wheel = crate::lock::LockWheel {
        name: "evdev".to_string(),
        version: "1.7.1".to_string(),
        origin: crate::lock::Origin::Built,
        filename: filename.to_string(),
        url: None,
        sha256: Some(sha256.clone()),
        requires_dist: Vec::new(),
        must_ship: true,
        upstream_url: None,
        git_source: None,
        sdist_source: Some(source.clone()),
    };

    let (overridden, captured) =
        closure_built_entry_override("evdev", &entry, &[wheel], &store).unwrap();
    assert_eq!(overridden.sha256.as_deref(), Some(sha256.as_str()));
    assert_eq!(
        overridden.url.as_ref().unwrap().to_file_path().unwrap(),
        store.join(&sha256).join(filename),
    );
    assert_eq!(captured.unwrap().sdist_url, source.sdist_url);
    let _ = std::fs::remove_dir_all(store);
}
use crate::config::RelaxPolicy;
use crate::constraint::Provenance;
use crate::index_chain::{IndexPurpose, PUBLIC_PYPI, index_chain};
use crate::relax::{CondaName, CondaTarget, NameMap, PypiKey};
use std::collections::BTreeMap;

#[derive(Clone)]
struct SharedLogWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for SharedLogWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn deprecated_system_requirements_rpc_shape_omits_glibc_contract() {
    use pixi_build_types::procedures::conda_outputs::CondaOutputsParams;

    // Pixi 0.73 sends only workspace build variants here. Deprecated
    // system-requirements affect its selected rich platform envelope, not the
    // typed conda/outputs parameters.
    let params: CondaOutputsParams = serde_json::from_value(serde_json::json!({
        "channels": [],
        "hostPlatform": "linux-64",
        "buildPlatform": "linux-64",
        "variantConfiguration": {},
        "variantFiles": [],
        "workDirectory": "/tmp/retread-rpc-glibc"
    }))
    .unwrap();
    assert_eq!(params.host_platform.as_str(), "linux-64");
    assert!(
        params
            .variant_configuration
            .as_ref()
            .is_some_and(BTreeMap::is_empty)
    );
}

#[cfg(unix)]
fn unique_test_dir(label: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("retread-{label}-{}-{nanos}", std::process::id()))
}

#[cfg(unix)]
#[tokio::test]
async fn conda_outputs_disk_fill_lock_coalesces_same_key_only() {
    let root = unique_test_dir("outputs-fill-lock");
    let first_path = root.join("same.json");
    let other_path = root.join("other.json");
    let first = acquire_conda_outputs_disk_cache_fill_lock(&first_path)
        .await
        .unwrap();

    // A different memo key must remain independently lockable.
    let other = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        acquire_conda_outputs_disk_cache_fill_lock(&other_path),
    )
    .await
    .expect("different output-memo keys must not serialize")
    .unwrap();
    drop(other);

    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let mut waiter = tokio::spawn(async move {
        started_tx.send(()).unwrap();
        acquire_conda_outputs_disk_cache_fill_lock(&first_path)
            .await
            .unwrap()
    });
    started_rx.await.unwrap();
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(50), &mut waiter)
            .await
            .is_err(),
        "a same-key contender must wait for the active first compute",
    );

    drop(first);
    let second = tokio::time::timeout(std::time::Duration::from_secs(2), waiter)
        .await
        .expect("same-key contender did not acquire after publication")
        .unwrap();
    drop(second);
    let _ = std::fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn initialize_preflight_repairs_dangling_pixi_bld_symlink_target() {
    let root = unique_test_dir("pixi-bld-symlink");
    let workspace = root.join("workspace");
    let pixi = workspace.join(".pixi");
    let target = root.join("tmp-build").join("pixi-bld");
    std::fs::create_dir_all(&pixi).unwrap();
    std::fs::write(workspace.join("pixi.toml"), "[workspace]\nchannels = []\n").unwrap();
    std::os::unix::fs::symlink(&target, pixi.join("bld")).unwrap();

    assert!(
        !target.exists(),
        "test setup must start with a dangling symlink"
    );
    // Pass in_slurm_job=false explicitly: this test is about the repair
    // logic, and reading the ambient SLURM_JOB_ID made it fail for everyone
    // running inside an allocation.
    ensure_pixi_bld_symlink_target_inner(Some(&workspace), false).unwrap();

    assert!(
        target.is_dir(),
        "preflight should create only the symlink target"
    );
    assert!(
        std::fs::symlink_metadata(pixi.join("bld"))
            .unwrap()
            .file_type()
            .is_symlink(),
        "preflight must not replace the .pixi/bld symlink"
    );
    assert_eq!(std::fs::read_link(pixi.join("bld")).unwrap(), target);

    let _ = std::fs::remove_dir_all(root);
}

// -----------------------------------------------------------------
// v0.46.0: BFS prefer-conda picker. Regression coverage for the bug
// where `torch` (whose parselmouth inverted map is ambiguous) fell
// through to PyPI and got bundled at latest (2.12.0), clobbering
// conda's pinned pytorch at install. The merged name_map (which
// carries the FALLBACK torch->pytorch) must win over the ambiguous
// parselmouth candidates so the BFS routes it to conda.
// -----------------------------------------------------------------

fn pypi_map(pairs: &[(&str, &[&str])]) -> PypiToCondaMap {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.iter().map(|s| s.to_string()).collect()))
        .collect()
}

fn name_map(pairs: &[(&str, &str)]) -> NameMap {
    pairs
        .iter()
        .map(|(pypi, conda)| {
            (
                PypiKey::from_pypi(pypi),
                CondaTarget::Mapped(CondaName::new(*conda)),
            )
        })
        .collect()
}

fn bundle_auto_route(name: &str, version: &str, provenance: Provenance) -> BundleAutoRoute {
    BundleAutoRoute {
        route: crate::uv_closure::AutoRoutedPackage {
            pypi_name: name.to_string(),
            conda_name: name.to_string(),
            pypi_version: version.to_string(),
            conda_version: version.to_string(),
            channel: "https://conda.example.invalid/linux-64".to_string(),
            input_requirements: Vec::new(),
        },
        provenance,
        workspace_provider: None,
    }
}

#[test]
fn spec_from_str_handles_build_string() {
    // Cold-solve replay round-trips emitted conda run-deps (incl. a
    // build-tagged python_abi like `3.12.* *_cp312`) through spec_from_str.
    // It must split the build string from the version, not feed the whole
    // tail to the version parser (which rejects it) -- the bug that made
    // replay silently fall through to the cascade.
    let ns = spec_from_str("python_abi 3.12.* *_cp312").expect("build-string spec must parse");
    assert_eq!(ns.name, "python_abi");
    match ns.spec {
        PackageSpec::Binary(b) => {
            assert!(b.version.is_some(), "version parsed");
            assert!(b.build.is_some(), "build string parsed");
        }
        _ => panic!("expected binary spec"),
    }
    // plain "name version" and bare "name" still work.
    assert!(spec_from_str("numpy >=1.26").is_ok());
    assert!(spec_from_str("pip").is_ok());
}

#[test]
fn pick_conda_target_name_map_wins_over_ambiguous_parselmouth() {
    // THE torch regression: parselmouth lists several conda
    // candidates for `torch` with NO identity match (torch != any).
    // Parselmouth alone -> None -> bundle. With the FALLBACK
    // torch->pytorch in the merged name_map, the picker returns
    // pytorch so the BFS routes it to conda instead of bundling.
    let parselmouth = pypi_map(&[("torch", &["pytorch", "pytorch-cpu", "pytorch-gpu"])]);
    let name_map = name_map(&[("torch", "pytorch")]);
    assert_eq!(
        pick_conda_target(&PypiKey::from_pypi("torch"), &name_map, &parselmouth)
            .as_ref()
            .map(CondaName::as_spec),
        Some("pytorch"),
    );
}

#[test]
fn conda_route_rejects_transitive_workspace_pypi_provider() {
    let protected = BTreeMap::from([(CondaName::new("pytorch"), PypiKey::from_pypi("torch"))]);

    assert_eq!(
        selected_workspace_pypi_provider_conflicts(["python", "tensordict", "pytorch"], &protected,),
        vec![(CondaName::new("pytorch"), PypiKey::from_pypi("torch"),)],
        "a routed package may not pull a conda provider owned by the workspace's PyPI solve",
    );
    assert!(
        selected_workspace_pypi_provider_conflicts(["python", "tensordict"], &protected,)
            .is_empty(),
        "the route remains valid when its solved closure does not shadow the PyPI provider",
    );
}

#[test]
fn provider_conflict_names_the_mutable_route_root() {
    use rattler_conda_types::{PackageRecord, RepoDataRecord, VersionWithSource};
    use std::str::FromStr;
    use url::Url;

    let record = |name: &str, depends: &[&str]| {
        let mut package_record = PackageRecord::new(
            name.parse().unwrap(),
            VersionWithSource::from_str("1.0.0").unwrap(),
            "h123_0".to_string(),
        );
        package_record.subdir = "linux-64".to_string();
        package_record.depends = depends.iter().map(|dep| (*dep).to_string()).collect();
        RepoDataRecord {
            package_record,
            file_name: format!("{name}-1.0.0-h123_0.conda"),
            url: Url::parse(&format!(
                "https://example.invalid/linux-64/{name}-1.0.0-h123_0.conda"
            ))
            .unwrap(),
            channel: Some("https://example.invalid".into()),
        }
    };
    let records = vec![
        record("tensordict", &["torchopt >=0.7"]),
        record("torchopt", &["pytorch >=2.0"]),
        record("pytorch", &["python >=3.11"]),
        record("unrelated", &["python >=3.11"]),
    ];
    let route = |name: &str| crate::uv_closure::CondaRouteSpec {
        pypi_name: PypiKey::from_pypi(name),
        conda_name: CondaName::new(name),
        spec: String::new(),
    };

    assert_eq!(
        routed_roots_reaching_provider(
            &records,
            &[route("tensordict"), route("unrelated")],
            &CondaName::new("pytorch"),
        ),
        vec![CondaName::new("tensordict")],
        "the reason hint must identify the route to reject, not only its transitive provider",
    );
}

#[test]
fn pick_conda_target_ambiguous_parselmouth_without_name_map_is_none() {
    // Documents the pre-fix behavior: ambiguous parselmouth + no
    // curated answer -> None (caller leaves it on the PyPI/bundle
    // path). This is exactly why the FALLBACK entry is load-bearing.
    let parselmouth = pypi_map(&[("torch", &["pytorch", "pytorch-cpu", "pytorch-gpu"])]);
    let name_map = NameMap::new();
    assert_eq!(
        pick_conda_target(&PypiKey::from_pypi("torch"), &name_map, &parselmouth),
        None
    );
}

#[test]
fn pick_conda_target_parselmouth_identity_match_wins() {
    // numpy -> numpy is an identity match among the candidates; no
    // name_map entry needed.
    let parselmouth = pypi_map(&[("numpy", &["numpy", "manifpy"])]);
    let name_map = NameMap::new();
    assert_eq!(
        pick_conda_target(&PypiKey::from_pypi("numpy"), &name_map, &parselmouth)
            .as_ref()
            .map(CondaName::as_spec),
        Some("numpy"),
    );
}

#[tokio::test]
async fn pick_conda_target_single_parselmouth_candidate() {
    let parselmouth = pypi_map(&[("some-pypi-only", &["the-conda-name"])]);
    let name_map = NameMap::new();
    assert_eq!(
        pick_conda_target(
            &PypiKey::from_pypi("some-pypi-only"),
            &name_map,
            &parselmouth,
        )
        .as_ref()
        .map(CondaName::as_spec),
        Some("the-conda-name"),
    );

    // The process snapshot canonicalizes candidate order and shares one Arc
    // across concurrent consumers instead of cloning the full mapping.
    let canonical = finalize_pypi_to_conda_map(pypi_map(&[(
        "unordered",
        &["z-provider", "a-provider", "z-provider"],
    )]));
    assert_eq!(canonical["unordered"], ["a-provider", "z-provider"]);

    let cell = Arc::new(tokio::sync::OnceCell::new());
    let fetches = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let spawn_load = |cell: Arc<tokio::sync::OnceCell<Arc<PypiToCondaMap>>>,
                      fetches: Arc<std::sync::atomic::AtomicUsize>| {
        tokio::spawn(async move {
            load_pypi_to_conda_map_with(&cell, || async move {
                fetches.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                tokio::task::yield_now().await;
                Ok(pypi_map(&[("shared", &["provider"])]))
            })
            .await
        })
    };
    let first = spawn_load(cell.clone(), fetches.clone());
    let second = spawn_load(cell, fetches.clone());
    let first = first.await.unwrap();
    let second = second.await.unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(fetches.load(std::sync::atomic::Ordering::SeqCst), 1);

    // A failed first fetch publishes one stable fallback snapshot rather than
    // allowing route semantics to change later in the same backend process.
    let fallback_cell = tokio::sync::OnceCell::new();
    let fallback = load_pypi_to_conda_map_with(&fallback_cell, || async {
        Err(anyhow!("injected fetch failure"))
    })
    .await;
    let cached = load_pypi_to_conda_map_with(&fallback_cell, || async {
        panic!("cached fallback must suppress a later fetch");
        #[allow(unreachable_code)]
        Ok(PypiToCondaMap::new())
    })
    .await;
    assert!(Arc::ptr_eq(&fallback, &cached));
    assert!(
        cached["torch"]
            .iter()
            .any(|candidate| candidate == "pytorch-gpu"),
        "the process-stable fallback must retain curated mappings"
    );
    assert_eq!(
        cached["ray"],
        ["ray-core"],
        "Ray routing must survive a total Parselmouth fetch failure"
    );
    let effective = effective_name_map(&NameMap::new(), &cached);
    assert_eq!(
        effective
            .get(&PypiKey::from_pypi("ray"))
            .and_then(CondaTarget::mapped_name)
            .map(CondaName::as_spec),
        Some("ray-core"),
    );
}

#[test]
fn parselmouth_retry_statuses_are_transient_only() {
    assert!(retryable_parselmouth_status(
        reqwest::StatusCode::REQUEST_TIMEOUT
    ));
    assert!(retryable_parselmouth_status(
        reqwest::StatusCode::TOO_MANY_REQUESTS
    ));
    assert!(retryable_parselmouth_status(
        reqwest::StatusCode::SERVICE_UNAVAILABLE
    ));
    assert!(!retryable_parselmouth_status(
        reqwest::StatusCode::UNAUTHORIZED
    ));
    assert!(!retryable_parselmouth_status(
        reqwest::StatusCode::NOT_FOUND
    ));
}

#[test]
fn pick_conda_target_unknown_dep_is_none() {
    // Not in name_map and not in parselmouth -> stays on PyPI.
    let parselmouth = pypi_map(&[("torch", &["pytorch"])]);
    let name_map = NameMap::new();
    assert_eq!(
        pick_conda_target(
            &PypiKey::from_pypi("totally-unknown-pkg"),
            &name_map,
            &parselmouth,
        ),
        None,
    );
}

#[test]
fn pick_conda_target_user_name_map_overrides_parselmouth_identity() {
    // A user retread-name-map entry beats even a parselmouth identity
    // match -- the curated answer is authoritative.
    let parselmouth = pypi_map(&[("opencv-python-headless", &["opencv-python-headless"])]);
    let name_map = name_map(&[("opencv-python-headless", "py-opencv")]);
    assert_eq!(
        pick_conda_target(
            &PypiKey::from_pypi("opencv-python-headless"),
            &name_map,
            &parselmouth,
        )
        .as_ref()
        .map(CondaName::as_spec),
        Some("py-opencv"),
    );
}

// -------------------------------------------------------------
// v0.37.0 D4: join_transitive_to_overrides clause-level dedup.
// -------------------------------------------------------------

#[test]
fn join_transitive_dedups_clauses_across_input_specs() {
    // The exact junk shape from shipped meta-v0:
    //   setuptools >=41.0.0,>=59.6.0,<80,>=59.6.0,<=79.0.1
    // Two `>=59.6.0` clauses survived because the input had
    // them embedded in two different parent spec strings.
    // Clause-level dedup must collapse to ONE `>=59.6.0`.
    let mut input: BTreeMap<String, Vec<String>> = BTreeMap::new();
    input.insert(
        "setuptools".to_string(),
        vec![
            ">=41.0.0".to_string(),
            ">=59.6.0,<80".to_string(),
            ">=59.6.0,<=79.0.1".to_string(),
        ],
    );
    let out = join_transitive_to_overrides(input);
    let joined = out.get("setuptools").expect("setuptools should be present");
    // Count occurrences of `>=59.6.0` — must be exactly one.
    let occurrences = joined.matches(">=59.6.0").count();
    assert_eq!(
        occurrences, 1,
        "expected one `>=59.6.0` clause after dedup, got `{joined}`",
    );
    // All other clauses preserved.
    for needed in [">=41.0.0", "<80", "<=79.0.1"] {
        assert!(
            joined.contains(needed),
            "missing clause `{needed}` in `{joined}`",
        );
    }
}

#[test]
fn join_transitive_filters_star_and_empty_clauses() {
    // Mixed input where some entries are bare `*` or empty —
    // these impose no constraint and including them produces
    // invalid match-specs like `pytorch >=1,*,==2`. Drop them
    // at the clause boundary.
    let mut input: BTreeMap<String, Vec<String>> = BTreeMap::new();
    input.insert(
        "pytorch".to_string(),
        vec![
            ">=1.4".to_string(),
            "*".to_string(),
            "".to_string(),
            "==2.10.0,*".to_string(),
        ],
    );
    let out = join_transitive_to_overrides(input);
    let joined = out.get("pytorch").expect("pytorch should be present");
    assert!(
        !joined.contains('*'),
        "`*` clauses must be filtered out, got `{joined}`",
    );
    assert!(joined.contains(">=1.4"));
    assert!(joined.contains("==2.10.0"));
}

#[test]
fn join_transitive_falls_back_when_build_strings_leak_through() {
    // v0.37.0 regression: if `extract_transitive_constraints`
    // ever fails to strip a build-string before pushing to the
    // override map, `join_transitive_to_overrides` must still
    // produce a parseable result. Tests the VersionSpec
    // validation fallback path: the cleaned join `>=1.4,2.10.0
    // cuda*_mkl*303,...` is not a valid VersionSpec, so we
    // expect the function to either skip the dep or fall back to
    // SOMETHING that at minimum doesn't crash. The current
    // implementation uses plain-concat as fallback; assert that
    // at least the dep doesn't disappear into a panic.
    let mut input: BTreeMap<String, Vec<String>> = BTreeMap::new();
    input.insert(
        "pytorch".to_string(),
        vec![
            ">=1.4".to_string(),
            "2.10.0 cuda*_mkl*303".to_string(),
            ">=2.10.0,<2.11.0a0".to_string(),
        ],
    );
    // No panic, no crash. Either skipped or fallback string.
    let out = join_transitive_to_overrides(input);
    if let Some(s) = out.get("pytorch") {
        // If kept, the entry must not be empty.
        assert!(!s.is_empty());
    }
}

#[test]
fn join_transitive_result_parses_as_version_spec() {
    // Contract: every emitted joined spec must be a parseable
    // VersionSpec. If it isn't, the cascade chokes downstream.
    // This test asserts the fallback path doesn't get
    // exercised under normal inputs.
    use rattler_conda_types::{ParseStrictness, VersionSpec};
    let mut input: BTreeMap<String, Vec<String>> = BTreeMap::new();
    input.insert(
        "numpy".to_string(),
        vec![
            ">=1.26".to_string(),
            ">=1.26,<2".to_string(),
            "<3".to_string(),
        ],
    );
    let out = join_transitive_to_overrides(input);
    let joined = out.get("numpy").expect("numpy should be present");
    assert!(
        VersionSpec::from_str(joined, ParseStrictness::Lenient).is_ok(),
        "joined spec must be parseable, got `{joined}`",
    );
}

#[test]
fn emission_workspace_snapshot_cannot_override_wheel_metadata() {
    // Regression: discovery independently solved the consuming workspace and
    // happened to select aiohttp 3.14, whose transitive constraint is
    // aiohappyeyeballs>=2.5. Applying that provisional selection as a
    // `retread-overrides` entry replaced BOTH authoritative requirements from
    // the routed Isaac Sim closure (==2.4.4 and >=2.3.0), after route planning
    // had already selected the compatible conda package 2.4.4.
    let mut bundle = solo_bundle("isaacsim-kernel-pack", vec!["aiohappyeyeballs==2.4.4"]);
    bundle.extras.push(rw(
        "aiohttp",
        meta("aiohttp", "3.11.11", vec!["aiohappyeyeballs>=2.3.0"], true),
    ));
    bundle.auto_routed.push(bundle_auto_route(
        "aiohappyeyeballs",
        "2.4.4",
        Provenance::PriorSelection,
    ));

    let mut base_config = cfg();
    base_config
        .overrides
        .insert("manual-intent".to_string(), "==1.2.3".to_string());
    let emission = DiscoveredEmission {
        output_name: "isaaclab-2-3x-pack".to_string(),
        channels: Vec::new(),
        transitive_overrides: BTreeMap::from([
            ("aiohappyeyeballs".to_string(), ">=2.5.0".to_string()),
            ("unrelated-derived".to_string(), ">=9".to_string()),
        ]),
        envs: vec!["uwlab-gpu".to_string()],
    };

    let (bundle, effective) = apply_emission(&bundle, &base_config, &emission);
    assert_eq!(bundle.conda_name, "isaaclab-2-3x-pack");
    assert_eq!(
        effective.overrides, base_config.overrides,
        "explicit overrides must retain their authority, and no provisional \
         workspace selection may be promoted"
    );

    let output = produce_output(
        &bundle,
        &effective,
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .expect("the real Isaac Sim requirements intersect at aiohappyeyeballs 2.4.4");
    let deps: Vec<(String, String)> = output
        .run_dependencies
        .depends
        .iter()
        .map(|dep| (dep.name.clone(), format_packagespec(&dep.spec)))
        .collect();
    let aiohappy = deps
        .iter()
        .find(|(name, _)| name == "aiohappyeyeballs")
        .map(|(_, spec)| {
            VersionSpec::from_str(spec, rattler_conda_types::ParseStrictness::Lenient).unwrap()
        })
        .expect("aiohappyeyeballs run dependency");
    assert!(
        aiohappy.matches(&rattler_conda_types::Version::from_str("2.4.4").unwrap()),
        "{deps:?}"
    );
    assert!(
        !aiohappy.matches(&rattler_conda_types::Version::from_str("2.4.5").unwrap()),
        "the strict 2.4.4 wheel pin must survive the compatible route envelope: {deps:?}"
    );
}

// -------------------------------------------------------------
// v0.37.0: pythons_for bare-major rejection (D2).
// -------------------------------------------------------------

#[test]
fn pythons_for_rejects_bare_major_variant() {
    // Pixi sometimes forwards `["3"]` to pixi-build backends when
    // the workspace's `build-variants` declaration doesn't reach
    // cleanly. retread must reject this and fall back to the
    // configured python or DEFAULT_PYTHON. Without the rejection,
    // every downstream concept (`__cpython` virtual package,
    // `python 3.*` emitted spec, ABI invariant checks) breaks
    // silently. This test pins the rejection.
    let mut variants: BTreeMap<String, Vec<VariantValue>> = BTreeMap::new();
    variants.insert(
        "python".to_string(),
        vec![VariantValue::String("3".to_string())],
    );
    let cfg = RetreadConfig {
        resolver: Default::default(),
        auto_route: true,
        // v4.6: legacy sweep semantics for the pre-v4.6 test matrix.
        route_policy: crate::config::RoutePolicy::Aggressive,
        route_include: vec![],
        keep_pypi: vec![],
        force_conda: vec![],
        sdist_build: Default::default(),
        hermetic: true,
        retread_wheels: BTreeMap::new(),
        relax: RelaxPolicy::Minor,
        overrides: BTreeMap::new(),
        name_map: BTreeMap::new(),
        shadow_libs: BTreeMap::new(),
        build_number: 0,
        drop_deps: Vec::new(),
        auto_bundle: false,
        conda_deps: Vec::new(),
        default_bundle: None,
        compression_level: None,
        compression_threads: None,
        emit_pypi: false,
        bundle_mode: crate::config::BundleMode::Fat,
        courier: false,
        courier_mode: Default::default(),
        blueprint: Default::default(),
        blueprint_sync: Default::default(),
        git_sources: std::collections::BTreeMap::new(),
        python: None,
        pin_version: false,
        deps_from: Default::default(),
        ledger_overrides: Default::default(),
        pack_manifest_path: None,
    };
    let result = pythons_for(&cfg, Some(&variants));
    assert_eq!(
        result,
        vec![DEFAULT_PYTHON.to_string()],
        "bare-major variant must fall through to DEFAULT_PYTHON, got {result:?}",
    );
}

#[test]
fn pythons_for_accepts_dotted_variant() {
    // Sanity check the happy path: a properly forwarded `"3.11"`
    // variant survives validation and gets used directly.
    let mut variants: BTreeMap<String, Vec<VariantValue>> = BTreeMap::new();
    variants.insert(
        "python".to_string(),
        vec![VariantValue::String("3.11".to_string())],
    );
    let cfg = RetreadConfig {
        resolver: Default::default(),
        auto_route: true,
        // v4.6: legacy sweep semantics for the pre-v4.6 test matrix.
        route_policy: crate::config::RoutePolicy::Aggressive,
        route_include: vec![],
        keep_pypi: vec![],
        force_conda: vec![],
        sdist_build: Default::default(),
        hermetic: true,
        retread_wheels: BTreeMap::new(),
        relax: RelaxPolicy::Minor,
        overrides: BTreeMap::new(),
        name_map: BTreeMap::new(),
        shadow_libs: BTreeMap::new(),
        build_number: 0,
        drop_deps: Vec::new(),
        auto_bundle: false,
        conda_deps: Vec::new(),
        default_bundle: None,
        compression_level: None,
        compression_threads: None,
        emit_pypi: false,
        bundle_mode: crate::config::BundleMode::Fat,
        courier: false,
        courier_mode: Default::default(),
        blueprint: Default::default(),
        blueprint_sync: Default::default(),
        git_sources: std::collections::BTreeMap::new(),
        python: None,
        pin_version: false,
        deps_from: Default::default(),
        ledger_overrides: Default::default(),
        pack_manifest_path: None,
    };
    let result = pythons_for(&cfg, Some(&variants));
    assert_eq!(result, vec!["3.11".to_string()]);
}

#[test]
fn pythons_for_filters_bare_major_keeps_dotted() {
    // Mixed list: `["3", "3.11"]` — drop the bare, keep the dotted.
    // Realistic scenario if pixi ever sent a hybrid set.
    let mut variants: BTreeMap<String, Vec<VariantValue>> = BTreeMap::new();
    variants.insert(
        "python".to_string(),
        vec![
            VariantValue::String("3".to_string()),
            VariantValue::String("3.11".to_string()),
            VariantValue::String("3.12".to_string()),
        ],
    );
    let cfg = RetreadConfig {
        resolver: Default::default(),
        auto_route: true,
        // v4.6: legacy sweep semantics for the pre-v4.6 test matrix.
        route_policy: crate::config::RoutePolicy::Aggressive,
        route_include: vec![],
        keep_pypi: vec![],
        force_conda: vec![],
        sdist_build: Default::default(),
        hermetic: true,
        retread_wheels: BTreeMap::new(),
        relax: RelaxPolicy::Minor,
        overrides: BTreeMap::new(),
        name_map: BTreeMap::new(),
        shadow_libs: BTreeMap::new(),
        build_number: 0,
        drop_deps: Vec::new(),
        auto_bundle: false,
        conda_deps: Vec::new(),
        default_bundle: None,
        compression_level: None,
        compression_threads: None,
        emit_pypi: false,
        bundle_mode: crate::config::BundleMode::Fat,
        courier: false,
        courier_mode: Default::default(),
        blueprint: Default::default(),
        blueprint_sync: Default::default(),
        git_sources: std::collections::BTreeMap::new(),
        python: None,
        pin_version: false,
        deps_from: Default::default(),
        ledger_overrides: Default::default(),
        pack_manifest_path: None,
    };
    let result = pythons_for(&cfg, Some(&variants));
    assert_eq!(result, vec!["3.11".to_string(), "3.12".to_string()]);
}

#[test]
fn courier_pure_python_bundle_is_platform_specific_not_noarch() {
    // Regression (newton-pack-latest): a courier pack with ALL pure-python
    // wheels must NOT be advertised noarch. It ships the native retread
    // installer + a python-specific lock and the courier recipe is
    // `noarch: None`; advertising noarch made pixi request a noarch build
    // that rattler-build rejects ("target-platform cannot be noarch").
    // Pure-python bundle: meta(.., platform_specific=false) -> is_pure_python.
    let bundle = Bundle {
        conda_name: "pure-py-pack".into(),
        primary: rw(
            "pure-py-pack",
            meta("pure-py-pack", "1.0.0", vec!["numpy>=1.21"], false),
        ),
        extras: vec![],
        probe_decisions: vec![],
        solve_diagnostics: BTreeMap::new(),
        auto_routed: vec![],
        auto_dropped: Default::default(),
        uv_closure_names: Default::default(),
        uv_dependency_graph: Default::default(),
        workspace_conda_versions: Default::default(),
        workspace_conda_provider_facts: Default::default(),
    };

    let courier_cfg = RetreadConfig {
        resolver: Default::default(),
        auto_route: true,
        // v4.6: legacy sweep semantics for the pre-v4.6 test matrix.
        route_policy: crate::config::RoutePolicy::Aggressive,
        route_include: vec![],
        keep_pypi: vec![],
        force_conda: vec![],
        sdist_build: Default::default(),
        hermetic: true,
        courier: true,
        ..cfg()
    };
    let out = produce_output(
        &bundle,
        &courier_cfg,
        Platform::Linux64,
        "3.11",
        &[],
        Some("deadbeef"),
        None,
    )
    .unwrap();
    assert_eq!(
        out.metadata.subdir,
        Platform::Linux64,
        "courier output must be platform-specific, not noarch"
    );
    assert!(
        out.metadata.noarch.is_none(),
        "courier output must not be noarch"
    );
    let advertised_host = out
        .host_dependencies
        .as_ref()
        .expect("courier metadata must preserve host run-export derivation");
    let advertised_host_names: Vec<&str> = advertised_host
        .depends
        .iter()
        .map(|dep| dep.name.as_str())
        .collect();
    assert_eq!(advertised_host_names, ["python", "pip"]);

    // Control: the legacy (non-courier) path still emits noarch for a
    // pure-python bundle.
    let legacy =
        produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    assert_eq!(legacy.metadata.subdir, Platform::NoArch);
}

#[test]
fn produce_output_emits_auto_routed_conda_run_deps() {
    // M2 (v4.3.0): packages the uv auto-route loop moved to conda must
    // land as conda run-deps of the stub output — including deep
    // transitives no shipped wheel's Requires-Dist names (scipy below).
    // For a name a wheel DOES declare (numpy), the auto-route pin wins
    // over the wheel's looser spec (first-insert dedup). Since the
    // bounded-range fix, non-deps-from, non-anchor, non-overridden pins
    // are emitted as `>=locked,<next-major`. NumPy is an ABI anchor, so its
    // route selection is confined to the selected within-minor ABI band.
    let bundle = Bundle {
        conda_name: "auto-pack".into(),
        primary: rw(
            "auto-pack",
            meta("auto-pack", "1.0.0", vec!["numpy>=1.21"], false),
        ),
        extras: vec![],
        probe_decisions: vec![],
        solve_diagnostics: BTreeMap::new(),
        auto_routed: vec![
            bundle_auto_route("numpy", "2.1.0", Provenance::PriorSelection),
            bundle_auto_route("scipy", "1.14.1", Provenance::PriorSelection),
        ],
        auto_dropped: Default::default(),
        uv_closure_names: Default::default(),
        uv_dependency_graph: Default::default(),
        workspace_conda_versions: Default::default(),
        workspace_conda_provider_facts: Default::default(),
    };
    let out = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let deps: Vec<(String, String)> = out
        .run_dependencies
        .depends
        .iter()
        .map(|d| (d.name.clone(), format_packagespec(&d.spec)))
        .collect();
    assert!(
        deps.contains(&("numpy".to_string(), ">=2.1,<2.2".to_string())),
        "auto-routed numpy must emit the selected within-minor ABI band: {deps:?}"
    );
    assert!(
        deps.contains(&("scipy".to_string(), ">=1.14.1,<2".to_string())),
        "auto-routed transitive scipy must be a run-dep even though no \
         shipped wheel names it: {deps:?}"
    );
    // No duplicate numpy from the wheel's Requires-Dist.
    assert_eq!(
        deps.iter().filter(|(n, _)| n == "numpy").count(),
        1,
        "{deps:?}"
    );

    // Control: without auto_routed the wheel's own (relaxed) spec is
    // emitted and scipy is absent.
    let plain = Bundle {
        auto_routed: vec![],
        uv_closure_names: Default::default(),
        uv_dependency_graph: Default::default(),
        ..bundle
    };
    let out = produce_output(&plain, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let deps: Vec<(String, String)> = out
        .run_dependencies
        .depends
        .iter()
        .map(|d| (d.name.clone(), format_packagespec(&d.spec)))
        .collect();
    assert!(
        deps.iter()
            .any(|(n, s)| n == "numpy" && s != ">=2.1.0,<3" && !s.trim().is_empty()),
        "{deps:?}"
    );
    assert!(!deps.iter().any(|(n, _)| n == "scipy"), "{deps:?}");
}

#[test]
fn auto_route_envelope_does_not_override_index_metadata_cap() {
    let mut bundle = solo_bundle("pillow-cap-pack", vec!["pillow<11.1"]);
    bundle.auto_routed.push(bundle_auto_route(
        "pillow",
        "12.3.0",
        Provenance::PriorSelection,
    ));

    let error = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None)
        .expect_err("the generated compatibility envelope must not replace wheel metadata");
    let message = format!("{error:#}");
    assert!(message.contains("mutually unsatisfiable"), "{message}");
    assert!(message.contains("pillow<11.1"), "{message}");
    assert!(message.contains(">=12.3.0"), "{message}");
    assert!(message.contains("<13"), "{message}");
}

#[test]
fn final_emission_opts_into_minimal_stale_cap_relaxation() {
    let mut bundle = solo_bundle("relax-pack", vec!["demo>=1,<2", "demo<99"]);
    bundle.extras.push(rw(
        "current-wheel",
        meta("current-wheel", "2.0.0", vec!["demo>=2.1,<3"], true),
    ));
    let mut config = cfg();
    config.relax = RelaxPolicy::PatchThenMinorThenMajorThenLastResort;

    let output =
        produce_output(&bundle, &config, Platform::Linux64, "3.11", &[], None, None).unwrap();
    let spec = output
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "demo")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("demo run dependency");
    let parsed =
        VersionSpec::from_str(&spec, rattler_conda_types::ParseStrictness::Lenient).unwrap();
    assert!(parsed.matches(&rattler_conda_types::Version::from_str("2.1").unwrap()));
    assert!(!parsed.matches(&rattler_conda_types::Version::from_str("3").unwrap()));
}

#[test]
fn within_major_relaxation_becomes_the_final_structured_bundle_record() {
    let mut bundle = solo_bundle("relax-record-pack", vec!["demo==1.24.0"]);
    bundle.extras.push(rw(
        "newer-consumer",
        meta("newer-consumer", "2.0.0", vec!["demo>=1.26"], true),
    ));
    let mut config = cfg();
    config.relax = RelaxPolicy::PatchThenMinorThenMajorThenLastResort;

    let (output, relaxations) = produce_output_pending_relaxations(
        &bundle,
        &config,
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    let emitted = output
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "demo")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("demo run dependency");
    let emitted =
        VersionSpec::from_str(&emitted, rattler_conda_types::ParseStrictness::Lenient).unwrap();
    assert!(emitted.matches(&rattler_conda_types::Version::from_str("1.26").unwrap()));
    assert!(!emitted.matches(&rattler_conda_types::Version::from_str("2").unwrap()));

    let target = ResolutionTarget::for_subdir("3.11", "linux-64");
    let manifest = bundled_relaxations_for_output(
        "relax-record-pack",
        "relax-record-pack",
        &target,
        &[],
        &relaxations,
    )
    .expect("safe within-major relaxation must produce a bundled record");
    assert_eq!(manifest.relaxations.len(), 1);
    let record = &manifest.relaxations[0];
    assert_eq!(record.package, "demo");
    assert_eq!(record.original_spec, "==1.24.0");
    assert_eq!(record.resulting_spec, ">=1.24,<2");
    assert_eq!(record.tier, RelaxPolicy::Minor);
    assert_eq!(
        record.kind,
        crate::relaxation_record::RelaxationRecordKind::ExactPinWidened
    );
    assert_eq!(record.scope.platform, "linux-64");
    assert_eq!(record.scope.python, "3.11");
    assert_eq!(record.scope.environments, Vec::<String>::new());
    assert_eq!(record.scope.targets, Vec::<String>::new());
    assert!(
        record
            .involved_wheels
            .iter()
            .any(|wheel| wheel.contains("newer-consumer==2.0.0"))
    );
    let reparsed: RelaxationManifest =
        serde_json::from_str(&manifest.to_pretty_json().unwrap()).unwrap();
    assert_eq!(reparsed, manifest);
    assert!(
        manifest
            .activate_script()
            .contains("[retread] auto-relaxed: demo ==1.24.0 -> >=1.24,<2 (exact pin widened)")
    );
}

#[test]
fn satisfiable_final_emission_has_no_bundle_record() {
    let mut bundle = solo_bundle("strict-record-pack", vec!["demo>=1"]);
    bundle.extras.push(rw(
        "compatible-consumer",
        meta("compatible-consumer", "2.0.0", vec!["demo<2"], true),
    ));
    let config = cfg();
    let (_, relaxations) = produce_output_pending_relaxations(
        &bundle,
        &config,
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    assert!(relaxations.is_empty());
    assert!(
        bundled_relaxations_for_output(
            "strict-record-pack",
            "strict-record-pack",
            &ResolutionTarget::for_subdir("3.11", "linux-64"),
            &[],
            &relaxations,
        )
        .is_none(),
        "strict packages must inject neither warning file"
    );
}

#[test]
fn noarch_relaxation_record_uses_the_concrete_resolution_platform() {
    let mut bundle = solo_bundle(
        "win-relax-record-pack",
        vec!["demo==1.24.0 ; sys_platform == 'win32'"],
    );
    bundle.extras.push(rw(
        "newer-win-consumer",
        meta(
            "newer-win-consumer",
            "2.0.0",
            vec!["demo>=1.26 ; sys_platform == 'win32'"],
            true,
        ),
    ));
    let mut config = cfg();
    config.relax = RelaxPolicy::PatchThenMinorThenMajorThenLastResort;

    let (_, win_relaxations) = produce_output_pending_relaxations(
        &bundle,
        &config,
        Platform::Win64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        win_relaxations.len(),
        1,
        "the concrete Win target must retain and relax both marked constraints"
    );
    let (_, noarch_relaxations) = produce_output_pending_relaxations(
        &bundle,
        &config,
        Platform::NoArch,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    assert!(
        noarch_relaxations.is_empty(),
        "evaluating a noarch artifact as a target would silently lose the Win relaxation record"
    );
}

#[test]
fn relaxation_parity_allows_run_exports_but_rejects_replaced_ranges() {
    let mut bundle = solo_bundle("relax-parity-pack", vec!["demo==1.24.0"]);
    bundle.extras.push(rw(
        "newer-consumer",
        meta("newer-consumer", "2.0.0", vec!["demo>=1.26"], true),
    ));
    let mut config = cfg();
    config.relax = RelaxPolicy::PatchThenMinorThenMajorThenLastResort;
    let (output, _) = produce_output_pending_relaxations(
        &bundle,
        &config,
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    let mut build_dependencies = output
        .run_dependencies
        .depends
        .iter()
        .map(|dependency| {
            let spec = format_packagespec(&dependency.spec);
            if spec.is_empty() {
                dependency.name.clone()
            } else {
                format!("{} {spec}", dependency.name)
            }
        })
        .collect::<Vec<_>>();
    build_dependencies.push("python_abi 3.11.* *_cp311".to_string());
    assert!(
        run_dependencies_match(&output.run_dependencies.depends, Some(&build_dependencies))
            .unwrap(),
        "Pixi-added run exports are a valid superset of advertised deps"
    );
    let mut intersected_dependencies = build_dependencies.clone();
    intersected_dependencies.push("demo >=1.27,<2".to_string());
    assert!(
        !run_dependencies_match(
            &output.run_dependencies.depends,
            Some(&intersected_dependencies)
        )
        .unwrap(),
        "a same-name extra must not silently intersect the recorded result"
    );
    let demo = build_dependencies
        .iter_mut()
        .find(|dependency| dependency.starts_with("demo "))
        .unwrap();
    *demo = "demo >=1.27,<2".to_string();
    assert!(
        !run_dependencies_match(&output.run_dependencies.depends, Some(&build_dependencies))
            .unwrap(),
        "replacing the recorded result with a tighter build spec must fail closed"
    );
}

#[test]
fn final_emission_bands_numpy_exact_pin_but_preserves_cuda_cap() {
    let bundle = solo_bundle("anchor-pack", vec!["numpy==1.26.4", "cuda>=12.8,<13"]);
    let mut config = cfg();
    config.relax = RelaxPolicy::StrongMajor;

    let output =
        produce_output(&bundle, &config, Platform::Linux64, "3.11", &[], None, None).unwrap();
    let emitted = output
        .run_dependencies
        .depends
        .iter()
        .map(|dependency| {
            (
                dependency.name.as_str(),
                VersionSpec::from_str(
                    &format_packagespec(&dependency.spec),
                    rattler_conda_types::ParseStrictness::Lenient,
                )
                .unwrap(),
            )
        })
        .collect::<BTreeMap<_, _>>();

    let numpy = &emitted["numpy"];
    assert!(numpy.matches(&rattler_conda_types::Version::from_str("1.26.4").unwrap()));
    assert!(
        numpy.matches(&rattler_conda_types::Version::from_str("1.26.5").unwrap()),
        "the NumPy exact pin must admit a higher within-minor patch"
    );
    assert!(
        !numpy.matches(&rattler_conda_types::Version::from_str("1.27").unwrap()),
        "the NumPy exact pin must not widen beyond its minor ABI band"
    );
    let cuda = &emitted["cuda"];
    assert!(cuda.matches(&rattler_conda_types::Version::from_str("12.8").unwrap()));
    assert!(
        !cuda.matches(&rattler_conda_types::Version::from_str("13").unwrap()),
        "the CUDA cap must not be stripped"
    );
}

#[test]
fn authoritative_numpy_pin_uses_conflicting_workspace_patch_fact_only_for_validation() {
    let mut bundle = solo_bundle("anchor-pack", vec!["numpy==1.26.0"]);
    bundle
        .workspace_conda_versions
        .insert("numpy".to_string(), "1.26.4".to_string());
    let mut config = cfg();
    config.relax = RelaxPolicy::PatchThenMinorThenMajorThenLastResort;

    let (output, relaxations) = produce_output_pending_relaxations(
        &bundle,
        &config,
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    let emitted = output
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "numpy")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("NumPy run dependency");
    let emitted =
        VersionSpec::from_str(&emitted, rattler_conda_types::ParseStrictness::Lenient).unwrap();
    assert!(emitted.matches(&rattler_conda_types::Version::from_str("1.26.1").unwrap()));
    assert!(emitted.matches(&rattler_conda_types::Version::from_str("1.26.4").unwrap()));
    assert!(!emitted.matches(&rattler_conda_types::Version::from_str("1.27").unwrap()));
    assert_eq!(relaxations.len(), 1);
    assert!(
        relaxations[0].to_string().contains(">=1.26.0,<1.27"),
        "{relaxations:?}"
    );

    let mut compatible = solo_bundle("compatible-anchor-pack", vec!["numpy>=1.26,<1.27"]);
    compatible
        .workspace_conda_versions
        .insert("numpy".to_string(), "1.26.4".to_string());
    let (output, relaxations) = produce_output_pending_relaxations(
        &compatible,
        &config,
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    let emitted = output
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "numpy")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("compatible NumPy run dependency");
    let emitted =
        VersionSpec::from_str(&emitted, rattler_conda_types::ParseStrictness::Lenient).unwrap();
    assert!(
        emitted.matches(&rattler_conda_types::Version::from_str("1.26.5").unwrap()),
        "a compatible precise fact must not be attached to or narrow the emitted range"
    );
    assert!(relaxations.is_empty());

    let mut advisory = solo_bundle("advisory-pack", vec!["starlette>=0.49,<0.50"]);
    advisory.primary.metadata_provenance = Provenance::SourceBuiltRelaxed;
    advisory
        .workspace_conda_versions
        .insert("starlette".to_string(), "0.45.3".to_string());
    let output = produce_output(
        &advisory,
        &config,
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    let emitted = output
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "starlette")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("advisory Starlette run dependency");
    let emitted =
        VersionSpec::from_str(&emitted, rattler_conda_types::ParseStrictness::Lenient).unwrap();
    assert!(emitted.matches(&rattler_conda_types::Version::from_str("0.45.3").unwrap()));
    assert!(
        !emitted.matches(&rattler_conda_types::Version::from_str("0.45.4").unwrap()),
        "the pre-existing advisory-only workspace fact path must retain its exact intersection"
    );
}

#[test]
fn emission_merges_alias_constraints_by_conda_target() {
    let bundle = solo_bundle("alias-pack", vec!["alpha-provider>=1", "beta-provider<2"]);
    let mut config = cfg();
    config.name_map = name_map(&[
        ("alpha-provider", "shared-provider"),
        ("beta-provider", "shared-provider"),
    ]);
    config.relax = RelaxPolicy::None;

    let output =
        produce_output(&bundle, &config, Platform::Linux64, "3.11", &[], None, None).unwrap();
    let shared = output
        .run_dependencies
        .depends
        .iter()
        .filter(|dependency| dependency.name == "shared-provider")
        .collect::<Vec<_>>();
    assert_eq!(shared.len(), 1, "{shared:?}");
    let spec = VersionSpec::from_str(
        &format_packagespec(&shared[0].spec),
        rattler_conda_types::ParseStrictness::Lenient,
    )
    .unwrap();
    assert!(spec.matches(&rattler_conda_types::Version::from_str("1.5").unwrap()));
    assert!(!spec.matches(&rattler_conda_types::Version::from_str("0.9").unwrap()));
    assert!(!spec.matches(&rattler_conda_types::Version::from_str("2").unwrap()));
}

#[test]
fn conflicting_alias_constraints_to_one_conda_target_fail_closed() {
    let emit = |reverse: bool| {
        let mut requirements = vec!["alpha-provider<2", "beta-provider>=2"];
        if reverse {
            requirements.reverse();
        }
        let bundle = solo_bundle("alias-conflict-pack", requirements);
        let mut config = cfg();
        config.name_map = name_map(&[
            ("alpha-provider", "shared-provider"),
            ("beta-provider", "shared-provider"),
        ]);
        config.relax = RelaxPolicy::None;
        let error = produce_output(&bundle, &config, Platform::Linux64, "3.11", &[], None, None)
            .expect_err("both aliases must participate in one strict intersection");
        format!("{error:#}")
    };

    let forward = emit(false);
    let backward = emit(true);
    assert_eq!(forward, backward);
    assert!(forward.contains("alpha-provider<2"), "{forward}");
    assert!(forward.contains("beta-provider>=2"), "{forward}");
    assert!(forward.contains("mutually unsatisfiable"), "{forward}");
}

#[test]
fn mapped_anchor_alias_native_star_override_fails_closed() {
    let bundle = solo_bundle("mapped-anchor-pack", vec!["numpy==1.26.4"]);
    let mut config = cfg();
    config.name_map = name_map(&[("numpy", "array-runtime")]);
    config
        .overrides
        .insert("array-runtime".to_string(), "*".to_string());

    let error = produce_output(&bundle, &config, Platform::Linux64, "3.11", &[], None, None)
        .expect_err("a mapped NumPy target must not bypass the ABI star-spec check");
    let message = format!("{error:#}");
    assert!(message.contains("ABI invariant"), "{message}");
    assert!(message.contains("array-runtime"), "{message}");
    assert!(message.contains("empty/*"), "{message}");
}

#[test]
fn hidden_many_to_one_anchor_alias_vetoes_relaxation() {
    let mut bundle = solo_bundle("hidden-anchor-pack", vec!["array-provider==1.26.4"]);
    bundle.extras.push(rw(
        "newer-consumer",
        meta(
            "newer-consumer",
            "1.0.0",
            vec!["array-provider>=2,<3"],
            true,
        ),
    ));
    let mut config = cfg();
    config.name_map = name_map(&[
        ("array-provider", "shared-array-runtime"),
        ("numpy", "shared-array-runtime"),
    ]);
    config.relax = RelaxPolicy::PatchThenMinorThenMajorThenLastResort;

    let error = produce_output(&bundle, &config, Platform::Linux64, "3.11", &[], None, None)
        .expect_err("a semantic NumPy alias must veto exact-pin widening");
    let message = format!("{error:#}");
    assert!(message.contains("mutually unsatisfiable"), "{message}");
    assert!(message.contains("array-provider==1.26.4"), "{message}");
    assert!(message.contains("array-provider>=2,<3"), "{message}");
}

#[test]
fn abi_postcheck_rejects_divergent_workspace_anchor_versions() {
    let mut bundle = solo_bundle("divergent-anchor-pack", vec!["numpy>=2.1,<3"]);
    bundle.workspace_conda_provider_facts.insert(
        "numpy".to_string(),
        WorkspaceCondaProviderFact {
            selected_versions: BTreeSet::from(["1.26.4".to_string(), "2.1.0".to_string()]),
            declared_specs: BTreeSet::new(),
            present_in_all_consumers: true,
        },
    );
    assert!(bundle.workspace_conda_versions.is_empty());

    let error = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None)
        .expect_err("the emitted NumPy range must cover every precise consumer selection");
    let message = format!("{error:#}");
    assert!(
        message.contains("does not cover workspace pin"),
        "{message}"
    );
    assert!(message.contains("numpy==1.26.4"), "{message}");
}

#[test]
fn final_emission_is_deterministic_under_wheel_and_requirement_permutations() {
    let emit = |reverse: bool| {
        let mut bundle = solo_bundle("permutation-pack", vec!["helper>=1", "demo==1.2.3"]);
        bundle.extras = vec![
            rw(
                "newer-wheel",
                meta(
                    "newer-wheel",
                    "2.0.0",
                    vec!["demo==1.2.4", "helper<3"],
                    true,
                ),
            ),
            rw(
                "noise-wheel",
                meta("noise-wheel", "1.0.0", vec!["helper>=1", "noise<2"], true),
            ),
        ];
        if reverse {
            bundle.primary.metadata.requires_dist.reverse();
            for wheel in &mut bundle.extras {
                wheel.metadata.requires_dist.reverse();
            }
            bundle.extras.reverse();
        }
        let mut config = cfg();
        config.relax = RelaxPolicy::PatchThenMinorThenMajorThenLastResort;
        produce_output(&bundle, &config, Platform::Linux64, "3.11", &[], None, None)
            .unwrap()
            .run_dependencies
            .depends
            .into_iter()
            .map(|dependency| {
                (
                    dependency.name.as_str().to_string(),
                    format_packagespec(&dependency.spec),
                )
            })
            .collect::<Vec<_>>()
    };

    let forward = emit(false);
    let backward = emit(true);
    assert_eq!(forward, backward);
    let demo = forward
        .iter()
        .find(|(name, _)| name == "demo")
        .map(|(_, spec)| {
            VersionSpec::from_str(spec, rattler_conda_types::ParseStrictness::Lenient).unwrap()
        })
        .expect("demo run dependency");
    assert!(demo.matches(&rattler_conda_types::Version::from_str("1.2.4").unwrap()));
    assert!(!demo.matches(&rattler_conda_types::Version::from_str("1.2.5").unwrap()));
}

fn insert_exact_workspace_provider(bundle: &mut Bundle, name: &str, version: &str) {
    bundle
        .workspace_conda_versions
        .insert(name.to_string(), version.to_string());
    bundle.workspace_conda_provider_facts.insert(
        name.to_string(),
        WorkspaceCondaProviderFact {
            selected_versions: BTreeSet::from([version.to_string()]),
            declared_specs: BTreeSet::new(),
            present_in_all_consumers: true,
        },
    );
}

#[test]
fn workspace_fact_ownership_respects_pypi_intent_and_mapping_direction() {
    let mut bundle = solo_bundle("source-pack", vec!["starlette>=0.40,<0.46"]);
    bundle.primary.metadata_provenance = Provenance::SourceBuiltRelaxed;
    insert_exact_workspace_provider(&mut bundle, "starlette", "0.45.3");
    let mut config = cfg();
    config.keep_pypi.push("starlette".to_string());
    bundle.apply_workspace_conda_fact_ownership(
        &config,
        &config.name_map,
        &BTreeSet::new(),
        &BTreeSet::new(),
    );
    assert!(bundle.auto_dropped.is_empty());

    let output =
        produce_output(&bundle, &config, Platform::Linux64, "3.11", &[], None, None).unwrap();
    let spec = output
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name.as_str() == "starlette")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("starlette must remain emitted");
    assert!(spec.contains(">=0.40"), "{spec}");
    assert!(spec.contains("<0.46"), "{spec}");
    assert!(!spec.contains("==0.45.3"), "{spec}");

    // Exclusions apply to the whole declared alias group: a PyPI-side keep
    // for torch also protects its mapped pytorch conda fact and route.
    let mut bundle = solo_bundle("source-pack", vec!["torch>=2"]);
    insert_exact_workspace_provider(&mut bundle, "pytorch", "2.7.0");
    bundle.auto_routed.push(BundleAutoRoute {
        route: crate::uv_closure::AutoRoutedPackage {
            pypi_name: "torch".to_string(),
            conda_name: "pytorch".to_string(),
            pypi_version: "2.7.0".to_string(),
            conda_version: "2.7.0".to_string(),
            channel: "https://conda.example.invalid/linux-64".to_string(),
            input_requirements: Vec::new(),
        },
        provenance: Provenance::PriorSelection,
        workspace_provider: None,
    });
    let mapped_bundle = bundle.clone();
    let mut config = cfg();
    config.name_map = name_map(&[("torch", "pytorch")]);
    config.keep_pypi.push("torch".to_string());

    bundle.apply_workspace_conda_fact_ownership(
        &config,
        &config.name_map,
        &BTreeSet::new(),
        &BTreeSet::new(),
    );

    assert!(bundle.auto_dropped.is_empty());
    assert_eq!(bundle.auto_routed.len(), 1);

    // Positive direction: a pytorch conda fact provides both the explicitly
    // mapped PyPI torch identity and the otherwise-unmapped PyPI pytorch
    // identity. The stale torch -> pytorch route is pruned by its PyPI key.
    let mut owned_bundle = mapped_bundle.clone();
    config.keep_pypi.clear();
    owned_bundle.apply_workspace_conda_fact_ownership(
        &config,
        &config.name_map,
        &BTreeSet::new(),
        &BTreeSet::new(),
    );
    assert_eq!(
        owned_bundle.auto_dropped,
        HashSet::from(["pytorch".to_string(), "torch".to_string()])
    );
    assert!(owned_bundle.auto_routed.is_empty());

    // Reverse direction is invalid: `torch -> pytorch` does not let a torch
    // conda fact provide PyPI torch when the configured provider is absent.
    let mut reverse_bundle = mapped_bundle;
    reverse_bundle.workspace_conda_versions.clear();
    reverse_bundle.workspace_conda_provider_facts.clear();
    insert_exact_workspace_provider(&mut reverse_bundle, "torch", "2.7.0");
    reverse_bundle.apply_workspace_conda_fact_ownership(
        &config,
        &config.name_map,
        &BTreeSet::new(),
        &BTreeSet::new(),
    );
    assert!(reverse_bundle.auto_dropped.is_empty());
    assert_eq!(reverse_bundle.auto_routed.len(), 1);

    // Mapping chains are not equivalence classes. Protecting PyPI foo, whose
    // provider is bar, must not exclude the independent baz fact that owns
    // PyPI bar through the separate `bar -> baz` edge.
    let mut chained_bundle = solo_bundle("source-pack", vec!["bar>=1"]);
    insert_exact_workspace_provider(&mut chained_bundle, "baz", "1.0.0");
    let mut chained_config = cfg();
    chained_config.name_map = name_map(&[("foo", "bar"), ("bar", "baz")]);
    chained_bundle.apply_workspace_conda_fact_ownership(
        &chained_config,
        &chained_config.name_map,
        &BTreeSet::new(),
        &BTreeSet::from(["foo".to_string()]),
    );
    assert_eq!(
        chained_bundle.auto_dropped,
        HashSet::from(["bar".to_string(), "baz".to_string()])
    );

    // Overrides may be keyed by the translated conda provider. With
    // `foo -> bar`, an override named bar must veto ownership of PyPI foo even
    // when bar is also a PyPI key mapped onward to baz.
    let mut provider_override_bundle = solo_bundle("source-pack", vec!["foo>=1"]);
    insert_exact_workspace_provider(&mut provider_override_bundle, "bar", "1.0.0");
    let mut provider_override_config = cfg();
    provider_override_config.name_map = name_map(&[("foo", "bar"), ("bar", "baz")]);
    provider_override_config
        .overrides
        .insert("bar".to_string(), "*".to_string());
    provider_override_bundle.apply_workspace_conda_fact_ownership(
        &provider_override_config,
        &provider_override_config.name_map,
        &BTreeSet::new(),
        &BTreeSet::new(),
    );
    assert!(provider_override_bundle.auto_dropped.is_empty());

    // Inferred mappings cannot create ownership, but they still participate
    // in exclusions because emission honors conda-provider-keyed overrides
    // under the effective map. Here the explicit fact name map is intentionally
    // empty: foo owns only by same-name identity, then the effective
    // `foo -> bar` emission edge lets the manual bar override veto that drop.
    let mut inferred_override_bundle = solo_bundle("source-pack", vec!["foo>=1"]);
    insert_exact_workspace_provider(&mut inferred_override_bundle, "foo", "1.0.0");
    let mut inferred_override_config = cfg();
    inferred_override_config.name_map = name_map(&[("foo", "bar")]);
    inferred_override_config
        .overrides
        .insert("bar".to_string(), "*".to_string());
    inferred_override_bundle.apply_workspace_conda_fact_ownership(
        &inferred_override_config,
        &NameMap::new(),
        &BTreeSet::new(),
        &BTreeSet::new(),
    );
    assert!(inferred_override_bundle.auto_dropped.is_empty());

    // An explicit disabled mapping is wheel-side intent and vetoes same-name
    // fact ownership.
    let mut disabled_bundle = solo_bundle("source-pack", vec!["numpy>=2"]);
    insert_exact_workspace_provider(&mut disabled_bundle, "numpy", "2.1.0");
    let mut disabled_config = cfg();
    disabled_config
        .name_map
        .insert(PypiKey::from_pypi("numpy"), CondaTarget::Disabled);
    disabled_bundle.apply_workspace_conda_fact_ownership(
        &disabled_config,
        &disabled_config.name_map,
        &BTreeSet::new(),
        &BTreeSet::new(),
    );
    assert!(disabled_bundle.auto_dropped.is_empty());

    // A disabled PyPI key is not a provider-wide veto. Another explicit edge
    // may still use the same-named conda fact.
    let mut disabled_alias_bundle = solo_bundle("source-pack", vec!["bar>=1"]);
    insert_exact_workspace_provider(&mut disabled_alias_bundle, "foo", "1.0.0");
    let mut disabled_alias_config = cfg();
    disabled_alias_config
        .name_map
        .insert(PypiKey::from_pypi("foo"), CondaTarget::Disabled);
    disabled_alias_config.name_map.insert(
        PypiKey::from_pypi("bar"),
        CondaTarget::Mapped(CondaName::new("foo")),
    );
    disabled_alias_bundle.apply_workspace_conda_fact_ownership(
        &disabled_alias_config,
        &disabled_alias_config.name_map,
        &BTreeSet::new(),
        &BTreeSet::new(),
    );
    assert_eq!(
        disabled_alias_bundle.auto_dropped,
        HashSet::from(["bar".to_string()])
    );

    // Auto-dropped identities remain PyPI-typed at emission. Owning PyPI bar
    // must not suppress a different raw PyPI foo requirement merely because
    // an effective routing map translates foo to conda bar.
    let mut typed_bundle = solo_bundle("source-pack", vec!["foo>=1"]);
    typed_bundle.auto_dropped.insert("bar".to_string());
    let mut typed_config = cfg();
    typed_config.name_map = name_map(&[("foo", "bar")]);
    let typed_output = produce_output(
        &typed_bundle,
        &typed_config,
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    assert!(
        typed_output
            .run_dependencies
            .depends
            .iter()
            .any(|dependency| dependency.name.as_str() == "bar"),
        "PyPI bar ownership must not drop a distinct PyPI foo requirement"
    );
}

fn partial_psutil_provider_bundle() -> Bundle {
    let mut bundle = solo_bundle("source-pack", vec!["psutil==5.9.8"]);
    bundle.workspace_conda_provider_facts.insert(
        "psutil".to_string(),
        WorkspaceCondaProviderFact {
            selected_versions: BTreeSet::from(["7.2.2".to_string()]),
            declared_specs: BTreeSet::from([">=7,<8".to_string()]),
            present_in_all_consumers: false,
        },
    );
    bundle.auto_routed.push(bundle_auto_route(
        "psutil",
        "5.9.8",
        Provenance::PriorSelection,
    ));
    bundle
}

fn assert_partial_provider_veto_preserves_route(
    label: &str,
    mut bundle: Bundle,
    config: RetreadConfig,
    dynamic_keep_pypi: BTreeSet<PypiKey>,
    protected_roots: BTreeSet<String>,
) {
    assert!(
        bundle.workspace_conda_versions.is_empty(),
        "{label}: this guard must exercise provider-only ownership"
    );
    assert_eq!(bundle.auto_routed.len(), 1, "{label}: stale route fixture");
    assert!(
        bundle.auto_routed[0].workspace_provider.is_none(),
        "{label}: fixture route must start unannotated"
    );
    let original_route = bundle.auto_routed[0].route.clone();
    let original_provenance = bundle.auto_routed[0].provenance.clone();

    bundle.apply_workspace_conda_fact_ownership(
        &config,
        &config.name_map,
        &dynamic_keep_pypi,
        &protected_roots,
    );

    assert!(
        bundle.auto_dropped.is_empty(),
        "{label}: explicit PyPI intent must veto the typed drop"
    );
    assert_eq!(
        bundle.auto_routed.len(),
        1,
        "{label}: veto must preserve the original route"
    );
    assert_eq!(
        bundle.auto_routed[0].route, original_route,
        "{label}: veto must not replace the route"
    );
    assert_eq!(
        bundle.auto_routed[0].provenance, original_provenance,
        "{label}: veto must preserve route provenance"
    );
    assert!(
        bundle.auto_routed[0].workspace_provider.is_none(),
        "{label}: veto must not annotate the route as workspace-owned"
    );
}

#[test]
fn partial_workspace_provider_ownership_respects_every_pypi_veto() {
    let mut override_config = cfg();
    override_config
        .overrides
        .insert("psutil".to_string(), "*".to_string());
    assert!(override_config.ledger_overrides.is_empty());
    assert_partial_provider_veto_preserves_route(
        "manual non-ledger override",
        partial_psutil_provider_bundle(),
        override_config,
        BTreeSet::new(),
        BTreeSet::new(),
    );

    let mut keep_config = cfg();
    keep_config.keep_pypi.push("psutil".to_string());
    assert_partial_provider_veto_preserves_route(
        "configured keep-pypi",
        partial_psutil_provider_bundle(),
        keep_config,
        BTreeSet::new(),
        BTreeSet::new(),
    );

    assert_partial_provider_veto_preserves_route(
        "dynamic keep-pypi",
        partial_psutil_provider_bundle(),
        cfg(),
        BTreeSet::from([PypiKey::from_pypi("psutil")]),
        BTreeSet::new(),
    );

    assert_partial_provider_veto_preserves_route(
        "protected root",
        partial_psutil_provider_bundle(),
        cfg(),
        BTreeSet::new(),
        BTreeSet::from(["psutil".to_string()]),
    );

    let mut materialized_bundle = partial_psutil_provider_bundle();
    materialized_bundle
        .extras
        .push(rw("psutil", meta("psutil", "7.2.2", Vec::new(), false)));
    assert_partial_provider_veto_preserves_route(
        "materialized wheel",
        materialized_bundle,
        cfg(),
        BTreeSet::new(),
        BTreeSet::new(),
    );

    let mut disabled_config = cfg();
    disabled_config
        .name_map
        .insert(PypiKey::from_pypi("psutil"), CondaTarget::Disabled);
    assert_partial_provider_veto_preserves_route(
        "disabled name mapping",
        partial_psutil_provider_bundle(),
        disabled_config,
        BTreeSet::new(),
        BTreeSet::new(),
    );
}

#[test]
fn pypi_wildcard_exclusion_round_trips_through_conda_emission() {
    let bundle = solo_bundle("jupyter-pack", vec!["jupyter-core!=5.0.*,>=4.12"]);

    let output =
        produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let dependency = output
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "jupyter-core")
        .expect("the routed PyPI dependency must be emitted to conda");
    let rendered = format_packagespec(&dependency.spec);
    let conda_spec =
        VersionSpec::from_str(&rendered, rattler_conda_types::ParseStrictness::Lenient)
            .unwrap_or_else(|error| panic!("emitted constraint `{rendered}` is invalid: {error}"));
    for accepted in ["4.12", "4.99", "5.1", "6.0"] {
        let version = rattler_conda_types::Version::from_str(accepted).unwrap();
        assert!(
            conda_spec.matches(&version),
            "{rendered} rejected {accepted}"
        );
    }
    for rejected in ["4.11.9", "5.0", "5.0.9"] {
        let version = rattler_conda_types::Version::from_str(rejected).unwrap();
        assert!(
            !conda_spec.matches(&version),
            "{rendered} accepted excluded {rejected}"
        );
    }

    let target = WheelTarget {
        python_version: "3.11".to_string(),
        conda_subdir: "linux-64".to_string(),
        max_glibc: None,
    };
    let routes = emitted_bundle_route_specs(&bundle, &cfg(), &target).unwrap();
    assert!(
        routes
            .iter()
            .any(|route| route.pypi_name == PypiKey::from_pypi("jupyter-core")),
        "joint route assembly must accept the preserved PyPI constraint: {routes:?}"
    );
}

#[test]
fn pypi_conda_only_constraint_still_fails_closed() {
    let dep = CondaDep {
        pypi_name: PypiKey::from_pypi("synthetic"),
        name: "synthetic".to_string(),
        spec: "1.2|1.3".to_string(),
        constraint_origin: CondaConstraintOrigin::Pypi {
            original_specifiers: "1.2|1.3".to_string(),
            effective_specifiers: "1.2|1.3".to_string(),
        },
    };
    let error = translated_emission_constraint(
        "synthetic (1.2|1.3)",
        &dep,
        &Provenance::IndexWheelMetadata,
    )
    .expect_err("non-PEP PyPI-origin syntax must not bypass finalization");
    let message = format!("{error:#}");
    assert!(message.contains("conda-only spec `1.2|1.3`"), "{message}");
    assert!(
        message.contains("PyPI-origin constraints may not bypass shared finalization"),
        "{message}"
    );

    let exact_bundle = solo_bundle("exact-pack", vec!["synthetic===1.2"]);
    let exact_error = produce_output(
        &exact_bundle,
        &cfg(),
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .expect_err("conda-inexpressible valid PEP 440 must also fail closed");
    let exact_message = format!("{exact_error:#}");
    assert!(
        exact_message.contains("preserved PEP 440 constraint `===1.2`"),
        "{exact_message}"
    );
    assert!(
        exact_message.contains("has no conda representation"),
        "{exact_message}"
    );
    assert!(
        exact_message.contains("PyPI-origin constraints may not bypass shared finalization"),
        "{exact_message}"
    );
}

#[test]
fn explicit_native_conda_alternation_remains_allowed() {
    let dep = CondaDep {
        pypi_name: PypiKey::from_pypi("synthetic"),
        name: "synthetic".to_string(),
        spec: "1.2|1.3".to_string(),
        constraint_origin: CondaConstraintOrigin::ExplicitOverride,
    };
    let translated =
        translated_emission_constraint("synthetic>=1", &dep, &Provenance::IndexWheelMetadata)
            .expect("explicit override is the documented native-conda boundary");
    assert!(translated.specifiers.is_empty());
    assert_eq!(translated.native_conda_override.as_deref(), Some("1.2|1.3"));
    assert_eq!(translated.provenance, Provenance::UvOverride);
}

#[test]
fn produce_output_omits_workspace_owned_auto_drops() {
    let mut bundle = solo_bundle(
        "owned-pack",
        vec!["numpy>=2.1,<3", "gym==0.23.1", "requests>=2.31"],
    );
    bundle.auto_dropped = ["numpy", "gym"]
        .into_iter()
        .map(crate::relax::canonical_conda_name)
        .collect();

    let out = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let names: Vec<&str> = out
        .run_dependencies
        .depends
        .iter()
        .map(|dep| dep.name.as_str())
        .collect();
    assert!(!names.contains(&"numpy"), "{names:?}");
    assert!(!names.contains(&"gym"), "{names:?}");
    assert!(names.contains(&"requests"), "{names:?}");
}

#[test]
fn speculative_emissions_are_debug_only_and_final_output_logs_once() {
    let mut bundle = solo_bundle(
        "owned-pack",
        vec!["numpy>=2.1,<3", "gym==0.23.1", "requests>=2.31"],
    );
    bundle.auto_dropped = ["numpy", "gym"]
        .into_iter()
        .map(crate::relax::canonical_conda_name)
        .collect();

    let logs = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .with_writer({
            let logs = std::sync::Arc::clone(&logs);
            move || SharedLogWriter(std::sync::Arc::clone(&logs))
        })
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        for _ in 0..3 {
            produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
        }
        let output =
            produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
        log_final_bundle_outputs(&CondaOutputsResult {
            outputs: vec![output],
            input_globs: Default::default(),
        });
    });

    let logs = String::from_utf8(logs.lock().unwrap().clone()).unwrap();
    assert_eq!(logs.matches("bundle run-deps emitted").count(), 1, "{logs}");
    assert!(
        !logs.contains("dropping wheel dependency owned by a workspace conda provider"),
        "{logs}"
    );
}

#[test]
fn bundle_probe_metrics_aggregate_retries_and_zero_probe_bundles() {
    let logs = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .with_writer({
            let logs = std::sync::Arc::clone(&logs);
            move || SharedLogWriter(std::sync::Arc::clone(&logs))
        })
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let metrics = std::sync::Arc::new(BundleProbeMetrics::new("retried-pack"));
        let retry_owner = std::sync::Arc::clone(&metrics);

        let first = metrics.enter();
        let overlapping = metrics.enter();
        drop(overlapping);
        drop(first);
        let first_finished = metrics
            .timing
            .lock()
            .unwrap()
            .finished
            .expect("the first probe wave records its completion");
        std::thread::sleep(std::time::Duration::from_millis(2));
        assert_eq!(
            metrics.timing.lock().unwrap().finished,
            Some(first_finished),
            "post-probe work must not extend the recorded finish time",
        );

        let second = metrics.enter();
        drop(second);
        let timing = metrics.timing.lock().unwrap();
        assert_eq!(metrics.probes.load(std::sync::atomic::Ordering::Relaxed), 3);
        assert_eq!(timing.rounds, 2);
        assert_eq!(timing.active, 0);
        assert!(
            timing.finished.unwrap() > first_finished,
            "wall time must span from the first probe start through the final round",
        );
        drop(timing);

        drop(metrics);
        assert!(
            logs.lock()
                .unwrap()
                .windows(b"bench: bundle route probes finished".len())
                .all(|window| window != b"bench: bundle route probes finished"),
            "a retry owner must keep the one bundle summary open",
        );
        drop(retry_owner);

        drop(BundleProbeMetrics::new("zero-probe-pack"));
    });

    let logs = String::from_utf8(logs.lock().unwrap().clone()).unwrap();
    assert_eq!(
        logs.matches("bench: bundle route probes finished").count(),
        2,
        "{logs}",
    );
    assert_eq!(logs.matches("bundle=retried-pack").count(), 1, "{logs}");
    assert_eq!(logs.matches("bundle=zero-probe-pack").count(), 1, "{logs}");
    assert!(logs.contains("probes=3"), "{logs}");
    assert!(logs.contains("rounds=2"), "{logs}");
    assert!(logs.contains("probes=0"), "{logs}");
    assert!(logs.contains("rounds=0"), "{logs}");
}

#[test]
fn produce_output_softens_deps_from_floor_pin_to_floor_spec() {
    // conda-as-truth fix: an auto-routed package whose root ORIGINATED
    // from a `retread-deps-from` exact pin (`deps_from_floor`)
    // must be emitted as a `>=` floor, not the usual exact `==` pin --
    // otherwise a sibling pack's own exact conda pin for the same name
    // (e.g. `setuptools ==83.0.0`) hard-conflicts with this pack's
    // `setuptools ==69.5.1` at workspace conda-solve time.
    let mut deps_from_route =
        bundle_auto_route("setuptools", "69.5.1", Provenance::DepsFromRelaxed);
    deps_from_route
        .route
        .input_requirements
        .push(crate::uv_closure::AutoRouteInputRequirement {
            specifiers: ">=69.5.1".to_string(),
            source: "retread-deps-from root `setuptools==69.5.1`".to_string(),
            provenance: Provenance::DepsFromRelaxed,
            role: crate::uv_closure::AutoRouteInputRole::Requirement,
        });
    let bundle = Bundle {
        conda_name: "protomotions-pack".into(),
        primary: rw(
            "protomotions-pack",
            meta("protomotions-pack", "1.0.0", vec![], false),
        ),
        extras: vec![],
        probe_decisions: vec![],
        solve_diagnostics: BTreeMap::new(),
        auto_routed: vec![
            deps_from_route,
            bundle_auto_route("numpy", "2.1.0", Provenance::PriorSelection),
        ],
        auto_dropped: Default::default(),
        uv_closure_names: Default::default(),
        uv_dependency_graph: Default::default(),
        workspace_conda_versions: Default::default(),
        workspace_conda_provider_facts: Default::default(),
    };
    let out = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let deps: Vec<(String, String)> = out
        .run_dependencies
        .depends
        .iter()
        .map(|d| (d.name.clone(), format_packagespec(&d.spec)))
        .collect();
    assert!(
        deps.contains(&("setuptools".to_string(), ">=69.5.1".to_string())),
        "deps-from-originated exact pin must be softened to an unbounded \
         floor (not a bounded range -- the deps-from softening path takes \
         priority over the bounded-range path): {deps:?}"
    );
    assert!(
        deps.contains(&("numpy".to_string(), ">=2.1,<2.2".to_string())),
        "ABI-anchor auto-routes emit a within-minor band: {deps:?}"
    );
}

#[test]
fn produce_output_preserves_deps_from_bare_and_range_specs() {
    let mut pandas = bundle_auto_route("pandas", "3.0.3", Provenance::DepsFromRelaxed);
    pandas
        .route
        .input_requirements
        .push(crate::uv_closure::AutoRouteInputRequirement {
            specifiers: String::new(),
            source: "retread-deps-from root `pandas`".to_string(),
            provenance: Provenance::DepsFromRelaxed,
            role: crate::uv_closure::AutoRouteInputRole::Requirement,
        });
    let mut scipy = bundle_auto_route("scipy", "1.17.0", Provenance::DepsFromRelaxed);
    scipy
        .route
        .input_requirements
        .push(crate::uv_closure::AutoRouteInputRequirement {
            specifiers: ">=1.15,<1.18".to_string(),
            source: "retread-deps-from root `scipy>=1.15,<1.18`".to_string(),
            provenance: Provenance::DepsFromRelaxed,
            role: crate::uv_closure::AutoRouteInputRole::Requirement,
        });
    let bundle = Bundle {
        conda_name: "upstream-requirements-pack".into(),
        primary: rw(
            "upstream-requirements-pack",
            meta("upstream-requirements-pack", "1.0.0", vec![], false),
        ),
        extras: vec![],
        probe_decisions: vec![],
        solve_diagnostics: BTreeMap::new(),
        auto_routed: vec![pandas, scipy],
        auto_dropped: Default::default(),
        uv_closure_names: Default::default(),
        uv_dependency_graph: Default::default(),
        workspace_conda_versions: Default::default(),
        workspace_conda_provider_facts: Default::default(),
    };

    let out = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let deps: BTreeMap<String, String> = out
        .run_dependencies
        .depends
        .iter()
        .map(|dep| (dep.name.clone(), format_packagespec(&dep.spec)))
        .collect();
    assert_eq!(deps.get("pandas").map(String::as_str), Some(""));
    assert_eq!(deps.get("scipy").map(String::as_str), Some(">=1.15,<1.18"));
}

// --- bounded_range_ceiling / auto-routed pin emission ----------------------

#[test]
fn bounded_range_ceiling_caps_at_next_major() {
    assert_eq!(bounded_range_ceiling("1.26.4").as_deref(), Some("2"));
    assert_eq!(bounded_range_ceiling("12.8").as_deref(), Some("13"));
    assert_eq!(bounded_range_ceiling("2.0.0").as_deref(), Some("3"));
}

#[test]
fn bounded_range_ceiling_zero_x_caps_at_next_minor() {
    // semver: pre-1.0 releases treat the minor component as the
    // breaking axis, so 0.x auto-routed pins cap at the next MINOR
    // instead of jumping straight to `<1`.
    assert_eq!(bounded_range_ceiling("0.20.1").as_deref(), Some("0.21"));
    assert_eq!(bounded_range_ceiling("0.9.0").as_deref(), Some("0.10"));
    assert_eq!(bounded_range_ceiling("0").as_deref(), Some("0.1"));
}

#[test]
fn bounded_range_ceiling_unparseable_returns_none() {
    assert!(bounded_range_ceiling("not-a-version").is_none());
    assert!(bounded_range_ceiling("").is_none());
}

#[test]
fn bounded_range_ceiling_overflow_returns_none() {
    assert!(bounded_range_ceiling("18446744073709551615.1").is_none());
    assert!(bounded_range_ceiling("0.18446744073709551615").is_none());
}

#[test]
fn produce_output_auto_routed_pin_widens_to_bounded_range() {
    // The core fix under test: a plain (non-floor) auto-routed pin
    // widens from the exact `==X.Y.Z` uv resolved to `>=X.Y.Z,<next-major`
    // -- floored at the locked version so the uv-solved closure it was
    // resolved against still installs, capped so the conda solver can't
    // pick something wildly newer than what the pack was ever tested with.
    let bundle = Bundle {
        auto_routed: vec![bundle_auto_route(
            "pandas",
            "2.2.3",
            Provenance::PriorSelection,
        )],
        ..solo_bundle("range-pack", vec![])
    };
    let out = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let deps: Vec<(String, String)> = out
        .run_dependencies
        .depends
        .iter()
        .map(|d| (d.name.clone(), format_packagespec(&d.spec)))
        .collect();
    assert!(
        deps.contains(&("pandas".to_string(), ">=2.2.3,<3".to_string())),
        "{deps:?}"
    );
}

#[test]
fn produce_output_auto_routed_pin_zero_x_widens_to_next_minor() {
    let bundle = Bundle {
        auto_routed: vec![bundle_auto_route(
            "etgen",
            "0.20.1",
            Provenance::PriorSelection,
        )],
        ..solo_bundle("range-pack-zero-x", vec![])
    };
    let out = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let deps: Vec<(String, String)> = out
        .run_dependencies
        .depends
        .iter()
        .map(|d| (d.name.clone(), format_packagespec(&d.spec)))
        .collect();
    assert!(
        deps.contains(&("etgen".to_string(), ">=0.20.1,<0.21".to_string())),
        "0.x auto-routed pin must cap at the next MINOR: {deps:?}"
    );
}

#[test]
fn produce_output_auto_routed_abi_anchor_stays_within_minor() {
    // ABI anchors (python/python_abi/libc/cuda family) may move to a newer
    // patch in the same minor ABI band, but never across that boundary.
    let bundle = Bundle {
        auto_routed: vec![bundle_auto_route(
            "cuda-version",
            "12.8",
            Provenance::PriorSelection,
        )],
        ..solo_bundle("anchor-pack", vec![])
    };
    let out = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let deps: Vec<(String, String)> = out
        .run_dependencies
        .depends
        .iter()
        .map(|d| (d.name.clone(), format_packagespec(&d.spec)))
        .collect();
    assert!(
        deps.contains(&("cuda-version".to_string(), ">=12.8,<12.9".to_string())),
        "ABI anchor auto-routed pin must stay within-minor: {deps:?}"
    );
}

#[test]
fn produce_output_mapped_pypi_abi_anchor_stays_within_minor() {
    let mut route = bundle_auto_route("numpy", "2.1.0", Provenance::PriorSelection);
    route.route.conda_name = "array-runtime".to_string();
    let bundle = Bundle {
        auto_routed: vec![route],
        ..solo_bundle("mapped-anchor-pack", vec![])
    };
    let out = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let mapped = out
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "array-runtime")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("mapped NumPy route");
    assert_eq!(mapped, ">=2.1,<2.2");
}

#[test]
fn produce_output_auto_routed_ledger_override_still_widens() {
    // Run-31 regression: an override merged from the
    // .retread/auto-overrides.json ledger (repair-engine-derived pypi
    // steering knob) lands in config.overrides exactly like a
    // hand-written [retread-overrides] entry -- but it must NOT trigger
    // the manual-override exemption, or every package ANY repair touched
    // re-emits an exact ==locked conda pin and the exact-pin conflict
    // class the bounded ranges eliminate comes straight back
    // (isaaclab-2.3x-pack re-emitted `setuptools ==80.10.2` right after
    // tier-1 ledgered `setuptools >=68,<81`).
    let mut config = cfg();
    config
        .overrides
        .insert("setuptools".to_string(), ">=68,<81".to_string());
    config.ledger_overrides.insert("setuptools".to_string());
    let bundle = Bundle {
        auto_routed: vec![bundle_auto_route(
            "setuptools",
            "80.10.2",
            Provenance::PriorSelection,
        )],
        ..solo_bundle("ledger-pack", vec![])
    };
    let out = produce_output(&bundle, &config, Platform::Linux64, "3.11", &[], None, None).unwrap();
    let deps: Vec<(String, String)> = out
        .run_dependencies
        .depends
        .iter()
        .map(|d| (d.name.clone(), format_packagespec(&d.spec)))
        .collect();
    assert!(
        deps.contains(&("setuptools".to_string(), ">=80.10.2,<81".to_string())),
        "ledger-derived override must NOT freeze the conda pin to exact; \
         bounded range expected: {deps:?}"
    );
}

#[test]
fn produce_output_auto_routed_manual_override_stays_exact() {
    // Hand-written intent (a `retread-overrides` entry for this exact
    // name) wins over the auto-derived bounded range.
    let mut config = cfg();
    config
        .overrides
        .insert("sentry-sdk".to_string(), "==1.2.3".to_string());
    let bundle = Bundle {
        auto_routed: vec![bundle_auto_route(
            "sentry-sdk",
            "1.2.3",
            Provenance::PriorSelection,
        )],
        ..solo_bundle("override-pack", vec![])
    };
    let out = produce_output(&bundle, &config, Platform::Linux64, "3.11", &[], None, None).unwrap();
    let deps: Vec<(String, String)> = out
        .run_dependencies
        .depends
        .iter()
        .map(|d| (d.name.clone(), format_packagespec(&d.spec)))
        .collect();
    assert!(
        deps.contains(&("sentry-sdk".to_string(), "==1.2.3".to_string())),
        "manually-overridden name must stay exact, not widen: {deps:?}"
    );

    let mut anchor_config = cfg();
    anchor_config
        .overrides
        .insert("numpy".to_string(), "==1.26.0".to_string());
    let anchor = Bundle {
        auto_routed: vec![bundle_auto_route(
            "numpy",
            "1.26.0",
            Provenance::PriorSelection,
        )],
        ..solo_bundle("anchor-override-pack", vec!["numpy>=1.26"])
    };
    let (output, warnings) = produce_output_pending_relaxations(
        &anchor,
        &anchor_config,
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    let numpy = output
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "numpy")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("manually-overridden NumPy route");
    assert_eq!(numpy, "==1.26.0");
    assert!(warnings.is_empty(), "{warnings:?}");

    anchor_config.ledger_overrides.insert("numpy".to_string());
    let (output, warnings) = produce_output_pending_relaxations(
        &anchor,
        &anchor_config,
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    let numpy = output
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "numpy")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("ledger-overridden NumPy route");
    assert_eq!(numpy, ">=1.26,<1.27");
    assert_eq!(warnings.len(), 1, "{warnings:?}");
}

#[test]
fn produce_output_never_emits_uv_closure_members_as_conda_run_deps() {
    // Regression (isaaclab-viral-pack / isaac-pack-latest, 2026-07-08):
    // a pypi-only external with NO conda candidate (`isaacsim-kernel`) and
    // an external whose spec no channel satisfies (`aiodns >=3.1.1` vs
    // conda-forge's 3.0.0 cap) are members of the exported uv closure —
    // the uv install set provides them at install time. The run-dep
    // translation loop leaked them as CONDA run-deps, making every isaac
    // env's conda solve unsatisfiable. Only conda-routed packages may be
    // emitted.
    let mut bundle = solo_bundle(
        "viral-pack",
        vec![
            "isaacsim-kernel==5.1.0.0", // pypi-only: no conda candidate anywhere
            "aiodns==3.1.1",            // conda caps at 3.0.0: spec unsatisfiable
            "requests>=2.31",           // control: NOT in the closure -> emitted
        ],
    );
    bundle.uv_closure_names = ["isaacsim-kernel", "aiodns"]
        .iter()
        .map(|n| crate::relax::canonical_conda_name(n))
        .collect();

    let out = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let deps: Vec<(String, String)> = out
        .run_dependencies
        .depends
        .iter()
        .map(|d| (d.name.clone(), format_packagespec(&d.spec)))
        .collect();
    assert!(
        !deps.iter().any(|(n, _)| n == "isaacsim-kernel"),
        "pypi-only closure member must never become a conda run-dep: {deps:?}"
    );
    assert!(
        !deps.iter().any(|(n, _)| n == "aiodns"),
        "closure member with conda-unsatisfiable spec must never become a \
         conda run-dep: {deps:?}"
    );
    assert!(
        deps.iter().any(|(n, _)| n == "requests"),
        "non-closure dep must still be emitted (conda-routed path): {deps:?}"
    );
}

#[test]
fn produce_output_closure_gate_keeps_auto_routed_pins_and_base_deps_undoubled() {
    // Companion to the closure gate: an auto-routed package (repodata
    // hit) DOES appear, exact-pinned, even when a shipped wheel names it
    // with a looser spec; and a bundle base-dep (`<entry>-*` family
    // member living in the uv closure) is not duplicated as a conda dep.
    let mut bundle = solo_bundle(
        "isaacsim",
        vec![
            "numpy>=1.21",            // auto-routed below: pin must win
            "isaacsim-kernel==6.0.0", // base-dep in the closure: no conda dep
        ],
    );
    bundle.auto_routed = vec![bundle_auto_route(
        "numpy",
        "2.1.0",
        Provenance::PriorSelection,
    )];
    bundle.uv_closure_names = ["isaacsim-kernel", "numpy"]
        .iter()
        .map(|n| crate::relax::canonical_conda_name(n))
        .collect();

    let out = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let deps: Vec<(String, String)> = out
        .run_dependencies
        .depends
        .iter()
        .map(|d| (d.name.clone(), format_packagespec(&d.spec)))
        .collect();
    assert!(
        deps.contains(&("numpy".to_string(), ">=2.1,<2.2".to_string())),
        "auto-routed ABI anchor must appear within-minor banded even though \
         it is also named by a wheel and present in uv pins pre-route: \
         {deps:?}"
    );
    assert_eq!(
        deps.iter().filter(|(n, _)| n == "numpy").count(),
        1,
        "auto-routed pin must not be duplicated by the wheel's spec: {deps:?}"
    );
    assert!(
        !deps.iter().any(|(n, _)| n == "isaacsim-kernel"),
        "bundle base-dep provided by the uv closure must not be duplicated \
         as a conda run-dep: {deps:?}"
    );
}

#[test]
fn produce_output_reflects_overrides_for_refinement_widening() {
    // The regression: simulate what conda_outputs does post-
    // v0.36.4 — apply the union'd overrides into effective and
    // re-render via produce_output. Without v0.36.4 this widen
    // path was inert: refinement widened a local copy of
    // effective.overrides, but the output pushed to pixi was
    // never rebuilt. Pixi then saw the original pinned spec.
    //
    // Names below are synthetic placeholders — the code path
    // under test is package-agnostic (no hardcoded list of
    // "widenable" deps anywhere). `dep-widened` will receive an
    // override, `dep-untouched` is the control whose spec must
    // survive the rebuild verbatim.
    let widened_name = "dep-widened";
    let control_name = "dep-untouched";
    let widened_pin = "==9.9.9";
    let control_pin = "==1.0.0";
    let bundle = solo_bundle(
        "synthetic-bundle",
        vec![
            &format!("{widened_name}{widened_pin}"),
            &format!("{control_name}{control_pin}"),
        ],
    );

    // Baseline rendering: no widening yet. The widened-name
    // dep lands at a non-wildcard spec (exact shape depends on
    // the configured relax policy; we only assert it's not `*`).
    let narrow =
        produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let narrow_widened_spec = narrow
        .run_dependencies
        .depends
        .iter()
        .find(|d| d.name == widened_name)
        .map(|d| format_packagespec(&d.spec))
        .expect("widened-name dep should appear in baseline run-deps");
    assert_ne!(
        narrow_widened_spec.trim(),
        "*",
        "baseline must be tighter than `*`, got `{narrow_widened_spec}`",
    );
    let narrow_control_spec = narrow
        .run_dependencies
        .depends
        .iter()
        .find(|d| d.name == control_name)
        .map(|d| format_packagespec(&d.spec))
        .expect("control dep should appear in baseline run-deps");

    // Simulate v0.36.4's post-refinement rebuild: the union of
    // per-env widenings has `dep-widened -> *`. Apply it via
    // effective.overrides and re-render.
    let mut rebuild_effective = cfg();
    rebuild_effective
        .overrides
        .insert(widened_name.to_string(), "*".to_string());
    let widened = produce_output(
        &bundle,
        &rebuild_effective,
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    let widened_spec = widened
        .run_dependencies
        .depends
        .iter()
        .find(|d| d.name == widened_name)
        .map(|d| format_packagespec(&d.spec))
        .expect("widened-name dep should still appear after the rebuild");
    // Before v0.36.4 this assertion would fail: the rebuilt
    // output emitted the same narrow spec because the
    // produce_output call never used the widened override.
    assert!(
        widened_spec.trim() == "*" || widened_spec.trim().is_empty(),
        "rebuild should ship the widened spec (`*`), got `{widened_spec}`",
    );

    // Control dep had no override — its emitted spec must match
    // the baseline rendering byte-for-byte so we know the
    // rebuild only mutates what the cascade widened.
    let control_spec_after = widened
        .run_dependencies
        .depends
        .iter()
        .find(|d| d.name == control_name)
        .map(|d| format_packagespec(&d.spec))
        .expect("control dep should remain in emitted run-deps");
    assert_eq!(
        control_spec_after, narrow_control_spec,
        "non-widened deps must render identically before and after rebuild",
    );
}

// v0.32.0+: workspace pin/transitive-constraint parsing moved to
// src/workspace.rs (see WorkspaceManifest::extract_transitive_constraints
// + discover_outputs_for_source). Tests for that live there.

fn cfg() -> RetreadConfig {
    RetreadConfig {
        resolver: Default::default(),
        auto_route: true,
        // v4.6: legacy sweep semantics for the pre-v4.6 test matrix.
        route_policy: crate::config::RoutePolicy::Aggressive,
        route_include: vec![],
        keep_pypi: vec![],
        force_conda: vec![],
        sdist_build: Default::default(),
        hermetic: true,
        retread_wheels: BTreeMap::new(),
        relax: RelaxPolicy::Minor,
        overrides: BTreeMap::new(),
        name_map: BTreeMap::new(),
        shadow_libs: BTreeMap::new(),
        build_number: 0,
        drop_deps: Vec::new(),
        auto_bundle: false,
        conda_deps: Vec::new(),
        default_bundle: None,
        compression_level: None,
        compression_threads: None,
        emit_pypi: false,
        bundle_mode: crate::config::BundleMode::Fat,
        courier: false,
        courier_mode: Default::default(),
        blueprint: Default::default(),
        blueprint_sync: Default::default(),
        git_sources: std::collections::BTreeMap::new(),
        python: None,
        pin_version: false,
        deps_from: Default::default(),
        ledger_overrides: Default::default(),
        pack_manifest_path: None,
    }
}

fn meta(name: &str, version: &str, requires: Vec<&str>, platform_specific: bool) -> WheelMetadata {
    WheelMetadata {
        name: name.into(),
        version: version.into(),
        requires_dist: requires.into_iter().map(String::from).collect(),
        retread_conda_run_dependencies: vec![],
        is_pure_python: !platform_specific,
        sha256: format!("sha-{name}"),
        filename: if platform_specific {
            format!(
                "{}-{version}-cp311-none-manylinux_2_35_x86_64.whl",
                name.replace('-', "_")
            )
        } else {
            format!("{}-{version}-py3-none-any.whl", name.replace('-', "_"))
        },
    }
}

fn rw(pypi: &str, m: WheelMetadata) -> ResolvedWheel {
    let url: url::Url = format!("https://example.com/{pypi}.whl").parse().unwrap();
    ResolvedWheel {
        pypi_name: pypi.to_string(),
        upstream_url: Some(url.clone()),
        git_source: None,
        sdist_source: None,
        metadata_provenance: Provenance::IndexWheelMetadata,
        url,
        metadata: m,
        extras_requested: vec![],
        auto_data: None,
        auto_data_dedup_skipped_root: None,
    }
}

fn solo_bundle(name: &str, requires: Vec<&str>) -> Bundle {
    Bundle {
        conda_name: name.into(),
        primary: rw(name, meta(name, "1.0.0", requires, true)),
        extras: vec![],
        probe_decisions: vec![],
        solve_diagnostics: BTreeMap::new(),
        auto_routed: vec![],
        auto_dropped: Default::default(),
        uv_closure_names: Default::default(),
        uv_dependency_graph: Default::default(),
        workspace_conda_versions: Default::default(),
        workspace_conda_provider_facts: Default::default(),
    }
}

#[test]
fn auto_routed_underscored_conda_name_emits_raw() {
    let mut bundle = solo_bundle("underscore-pack", vec!["cuda-nvcc-linux-64==12.9.1"]);
    bundle.auto_routed.push(BundleAutoRoute {
        route: crate::uv_closure::AutoRoutedPackage {
            pypi_name: "cuda-nvcc-linux-64".to_string(),
            conda_name: "cuda-nvcc_linux-64".to_string(),
            pypi_version: "12.9.1".to_string(),
            conda_version: "12.9.1".to_string(),
            channel: "https://conda.example.invalid/linux-64".to_string(),
            input_requirements: Vec::new(),
        },
        provenance: Provenance::PriorSelection,
        workspace_provider: None,
    });

    let output =
        produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let routed = output
        .run_dependencies
        .depends
        .iter()
        .filter(|dependency| dependency.name.as_str() != "python")
        .collect::<Vec<_>>();
    assert_eq!(routed.len(), 1, "raw/folded aliases must dedupe by key");
    assert_eq!(routed[0].name.as_str(), "cuda-nvcc_linux-64");

    let target = WheelTarget {
        python_version: "3.11".to_string(),
        conda_subdir: "linux-64".to_string(),
        max_glibc: None,
    };
    let route_specs = emitted_bundle_route_specs(&bundle, &cfg(), &target).unwrap();
    let route = route_specs
        .iter()
        .find(|route| route.pypi_name == PypiKey::from_pypi("cuda-nvcc-linux-64"))
        .expect("the raw auto-route must reach the co-solve boundary");
    assert_eq!(route.conda_name.as_spec(), "cuda-nvcc_linux-64");
    assert_eq!(
        route.match_spec().as_str(),
        "cuda-nvcc_linux-64 >=12.9,<12.10"
    );
}

// -----------------------------------------------------------------
// v1.3.0: cascade PyPI index fallback chain + step-8 auto-bundle
// gating. The chain must mirror auto_bundle_transitives' (entry
// indexes first, then public PyPI) so NVIDIA-only deps resolvable
// only on pypi.nvidia.com can be bundled by the cascade instead of
// falling through to a doomed conda emission.
// -----------------------------------------------------------------

fn spec_entry(version: &str, index: Option<&str>) -> WheelEntry {
    let mut v = serde_json::json!({ "version": version });
    if let Some(idx) = index {
        v["index"] = serde_json::Value::String(idx.to_string());
    }
    serde_json::from_value(v).unwrap()
}

#[test]
fn localize_wheel_source_prefers_cached_copy() {
    let root = std::env::temp_dir().join(format!("retread-localize-{}", std::process::id()));
    let entry_dir = root.join("someentry");
    std::fs::create_dir_all(&entry_dir).unwrap();
    std::fs::write(root.join("cached_top-1.0-py3-none-any.whl"), b"x").unwrap();
    std::fs::write(entry_dir.join("cached_sub-1.0-py3-none-any.whl"), b"x").unwrap();

    // Cached at the top level -> file://.
    let up: url::Url = "https://pypi.org/p/cached_top-1.0-py3-none-any.whl"
        .parse()
        .unwrap();
    let localized = localize_wheel_source(&up, &root);
    assert_eq!(localized.scheme(), "file");
    assert!(
        localized
            .path()
            .ends_with("cached_top-1.0-py3-none-any.whl")
    );

    // Cached in a per-entry subdir -> file://.
    let up: url::Url = "https://pypi.org/p/cached_sub-1.0-py3-none-any.whl"
        .parse()
        .unwrap();
    assert_eq!(localize_wheel_source(&up, &root).scheme(), "file");

    // Not cached -> upstream URL untouched.
    let up: url::Url = "https://pypi.org/p/absent-1.0-py3-none-any.whl"
        .parse()
        .unwrap();
    assert_eq!(localize_wheel_source(&up, &root), up);

    // file:// (the D-rewritten primary) passes through untouched.
    let f: url::Url =
        url::Url::from_file_path(root.join("cached_top-1.0-py3-none-any.whl")).unwrap();
    assert_eq!(localize_wheel_source(&f, &root), f);

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn relaxed_retry_specs_table() {
    // v1.5.9 patch-drift fallback math: only single exact pins under a
    // relaxing policy produce a retry range; everything else resolves
    // exact-first with no fallback.
    use std::str::FromStr as _;
    let specs = |s: &str| VersionSpecifiers::from_str(s).unwrap();

    // Exact 4-component NVIDIA-style pin widens under the tiered
    // cascade policy (mirrors Patch at rewrite time) -- this is the
    // range that ADMITS the drifted 6.0.0.1, which is why it is only
    // a fallback.
    let relaxed = relaxed_retry_specs(
        "isaacsim-kernel",
        &specs("==6.0.0.0"),
        RelaxPolicy::PatchThenMinorThenMajorThenLastResort,
        &AbiAliasGraph::new(),
    )
    .expect("exact pin must produce a retry range");
    // PEP 440 normalization may render >=6.0.0.0 as >=6; assert
    // semantically: the range admits both the exact version and the
    // drifted next patch (which is WHY it is only a fallback).
    let v = |s: &str| uv_pep508::uv_pep440::Version::from_str(s).unwrap();
    assert!(relaxed.contains(&v("6.0.0.0")), "got {relaxed}");
    assert!(relaxed.contains(&v("6.0.0.1")), "got {relaxed}");
    assert!(!relaxed.contains(&v("6.1.0")), "got {relaxed}");
    assert_ne!(relaxed, specs("==6.0.0.0"));

    // Range specs pass through relax untouched -> no fallback.
    assert!(
        relaxed_retry_specs(
            "etgen",
            &specs(">=0.8.2,<0.9"),
            RelaxPolicy::Minor,
            &AbiAliasGraph::new(),
        )
        .is_none()
    );
    // Policy None never falls back.
    assert!(
        relaxed_retry_specs(
            "foo",
            &specs("==1.0.0"),
            RelaxPolicy::None,
            &AbiAliasGraph::new(),
        )
        .is_none()
    );
}

#[test]
fn relaxed_retry_specs_vetoes_hidden_cuda_alias() {
    use std::str::FromStr as _;

    let package = "nvidia-cuda-runtime-cu12";
    let specs = VersionSpecifiers::from_str("==12.8.90").unwrap();
    let mut aliases = AbiAliasGraph::new();
    add_abi_alias_edge(&mut aliases, package, "cuda");

    assert!(!crate::solve::is_abi_anchor(package));
    assert!(is_semantic_abi_anchor(package, &aliases));
    for policy in [RelaxPolicy::Major, RelaxPolicy::StrongMajor] {
        assert!(
            relaxed_retry_specs(package, &specs, policy, &aliases).is_none(),
            "{policy:?} retry must not widen a hidden ABI-anchor exact pin"
        );
    }
}

#[test]
fn vendored_filter_matches_underscore_pypi_name() {
    // P2 (grizzly #2) through the real consumer: a bundle wheel whose
    // recorded pypi_name uses underscores ("opencv_python") must still
    // suppress the conda run-dep emitted for "opencv-python". Pre-P2
    // the vendored set was seeded RAW while the query side was
    // canonical, so the filter missed and the dep shipped.
    let mut bundle = solo_bundle("isaac-pack", vec!["opencv-python==4.9.0", "numpy==1.26.0"]);
    bundle.extras.push(rw(
        "opencv_python",
        meta("opencv_python", "4.9.0", vec![], true),
    ));
    let output =
        produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let names: Vec<String> = output
        .run_dependencies
        .depends
        .iter()
        .map(|d| d.name.as_str().to_string())
        .collect();
    assert!(
        !names.iter().any(|n| n.contains("opencv")),
        "underscore-named vendored wheel must suppress the emission; got: {names:?}"
    );
    assert!(names.iter().any(|n| n == "numpy"), "control dep survives");
}

#[test]
fn already_covered_dotted_name() {
    // P2 dotted-name class: a skip set seeded from a "ruamel.yaml"
    // wheel covers the canonical "ruamel-yaml" query and vice versa.
    let mut set = std::collections::HashSet::new();
    set.insert(crate::relax::canonical_conda_name("ruamel.yaml"));
    assert!(crate::relax::already_covered(&set, "ruamel-yaml", None));
    assert!(crate::relax::already_covered(
        &set,
        "x",
        Some("ruamel.yaml")
    ));
    assert!(!crate::relax::already_covered(&set, "ruamel", None));
}

#[test]
fn name_mapped_dep_dropped_by_pypi_name() {
    // v1.4.0 regression (found via examples/isaac6): the cascade
    // bundles a wheel and records the drop under the PYPI name
    // ("tinyobjloader"), but name_map translates the emission to a
    // different conda name ("tinyobjloader-python"). The drop and
    // vendored filters must match on EITHER name -- before this,
    // the doomed conda run-dep shipped alongside the bundled wheel
    // and the solve died with "No candidates were found for
    // tinyobjloader-python".
    let mut config = cfg();
    config.name_map.insert(
        PypiKey::from_pypi("tinyobjloader"),
        CondaTarget::Mapped(CondaName::new("tinyobjloader-python")),
    );

    // Case 1: drop recorded under the PyPI name (what the cascade
    // pushes).
    let mut dropped_cfg = config.clone();
    dropped_cfg.drop_deps.push("tinyobjloader".into());
    let bundle = solo_bundle(
        "isaac-pack-6",
        vec!["tinyobjloader==2.0.0rc13", "numpy==1.26.0"],
    );
    let output = produce_output(
        &bundle,
        &dropped_cfg,
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    let names: Vec<String> = output
        .run_dependencies
        .depends
        .iter()
        .map(|d| d.name.as_str().to_string())
        .collect();
    assert!(
        !names.iter().any(|n| n.contains("tinyobjloader")),
        "pypi-name drop must also drop the name-mapped conda emission; got: {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "numpy"),
        "control dep must survive"
    );

    // Case 2: the wheel is vendored in the bundle under the PyPI
    // name (what try_pypi_bundle pushes to extras).
    let mut vendored_bundle = solo_bundle("isaac-pack-6", vec!["tinyobjloader==2.0.0rc13"]);
    vendored_bundle.extras.push(rw(
        "tinyobjloader",
        meta("tinyobjloader", "2.0.0rc13", vec![], true),
    ));
    let output = produce_output(
        &vendored_bundle,
        &config,
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    let names: Vec<String> = output
        .run_dependencies
        .depends
        .iter()
        .map(|d| d.name.as_str().to_string())
        .collect();
    assert!(
        !names.iter().any(|n| n.contains("tinyobjloader")),
        "vendored wheel must drop the name-mapped conda emission; got: {names:?}"
    );
}

#[test]
fn bundle_group_default_and_precedence() {
    // v1.4.0 retread-bundle: per-entry bundle > pack default >
    // standalone (entry's own name).
    let plain = spec_entry("==1.0", None);
    let mut grouped = spec_entry("==1.0", None);
    grouped.bundle = Some("other-pack".into());

    // No default: entry without bundle is standalone.
    assert_eq!(bundle_group_for("foo", &plain, None), "foo");
    // Default fills in for entries without a bundle.
    assert_eq!(
        bundle_group_for("foo", &plain, Some("isaac-pack")),
        "isaac-pack"
    );
    // Per-entry bundle wins over the default (mixed mode).
    assert_eq!(
        bundle_group_for("foo", &grouped, Some("isaac-pack")),
        "other-pack"
    );
}

// -----------------------------------------------------------------
// index_chain: order, dedup, and terminal fallback semantics
// -----------------------------------------------------------------

#[test]
fn index_chain_entry_before_workspace() {
    // The canonical use case: entry index + workspace indexes.
    // Public PyPI should land last, entry index first.
    let result = index_chain(
        ["https://pypi.nvidia.com".to_string()],
        &[
            "https://download.pytorch.org/whl/cu128".to_string(),
            PUBLIC_PYPI.to_string(),
        ],
        IndexPurpose::RootResolve,
    );
    assert_eq!(
        result,
        vec![
            "https://pypi.nvidia.com".to_string(),
            "https://download.pytorch.org/whl/cu128".to_string(),
            PUBLIC_PYPI.to_string(),
        ],
    );
}

#[test]
fn index_chain_dedups_trailing_slash() {
    // public PyPI appears in extra without trailing slash -- must not
    // be added twice even though the stored constant has a trailing slash.
    let result = index_chain(
        ["https://pypi.nvidia.com".to_string()],
        &["https://pypi.org/simple".to_string()],
        IndexPurpose::RootResolve,
    );
    assert_eq!(
        result,
        vec![
            "https://pypi.nvidia.com".to_string(),
            "https://pypi.org/simple".to_string(),
        ],
        "public PyPI without trailing slash should suppress the appended PUBLIC_PYPI",
    );
}

#[test]
fn index_chain_empty_inputs_append_public() {
    let result = index_chain(std::iter::empty::<String>(), &[], IndexPurpose::RootResolve);
    assert_eq!(result, vec![PUBLIC_PYPI.to_string()]);
}

#[test]
fn index_chain_deduplicates_repeated_entry_indexes() {
    // Two primary items that are the same URL (one with, one without slash)
    // should only appear once.
    let result = index_chain(
        [
            "https://pypi.nvidia.com".to_string(),
            "https://pypi.nvidia.com/".to_string(),
        ],
        &[],
        IndexPurpose::RootResolve,
    );
    assert_eq!(
        result,
        vec![
            "https://pypi.nvidia.com".to_string(),
            PUBLIC_PYPI.to_string(),
        ],
    );
}

#[test]
fn index_chain_explicit_override_replaces_public() {
    let workspace = vec![
        "https://packages.example/simple".to_string(),
        "https://extra.example/simple".to_string(),
    ];
    assert_eq!(
        index_chain(
            std::iter::empty::<String>(),
            &workspace,
            IndexPurpose::RootResolve,
        ),
        vec![
            "https://packages.example/simple".to_string(),
            "https://extra.example/simple".to_string(),
        ],
        "an explicit workspace index-url replaces public PyPI; index_chain must not re-append it",
    );
}

#[tokio::test]
async fn resolve_bundle_bfs_falls_back_to_default() {
    // The workspace arg is the COMPLETE chain (resolution_pypi_index_urls
    // already appends pixi's implicit public default), so public PyPI is the
    // terminal fallback the mock serves.
    let workspace = vec![
        "https://workspace.example/simple".to_string(),
        PUBLIC_PYPI.to_string(),
    ];
    let indexes = index_chain(
        ["https://entry.example/simple".to_string()],
        &workspace,
        IndexPurpose::RootResolve,
    );
    let requires_dist = vec!["root-child>=1".to_string()];
    let mut work = std::collections::VecDeque::new();
    seed_worklist(
        &requires_dist,
        &[],
        &indexes,
        "root-",
        &std::collections::HashSet::new(),
        &mut work,
        None,
        &std::collections::HashSet::new(),
    )
    .unwrap();
    let pending = work.pop_front().expect("the BFS must seed root-child");
    assert_eq!(pending.pypi_name, "root-child");
    let PendingSource::Pypi {
        specifiers,
        indexes: pending_indexes,
    } = &pending.source
    else {
        panic!("an ordinary Requires-Dist must seed a PyPI pending source");
    };
    assert_eq!(specifiers.to_string(), ">=1");
    assert_eq!(pending_indexes, &indexes);

    let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let fetch = {
        let calls = std::sync::Arc::clone(&calls);
        move |name: String, specs: VersionSpecifiers, index: String| {
            let calls = std::sync::Arc::clone(&calls);
            async move {
                assert_eq!(name, "root-child");
                assert_eq!(specs.to_string(), ">=1");
                calls.lock().unwrap().push(index.clone());
                if index.trim_end_matches('/') == PUBLIC_PYPI.trim_end_matches('/') {
                    Ok(index)
                } else {
                    Err(crate::pypi::pypi_index_miss(format!(
                        "root-child is absent from {index}"
                    )))
                }
            }
        }
    };
    let selected = fetch_from_pypi_index_chain(
        pending_indexes,
        |index| fetch(pending.pypi_name.clone(), specifiers.clone(), index),
        "root-child fixture exhausted its index chain".to_string(),
    )
    .await
    .unwrap();
    assert_eq!(selected, PUBLIC_PYPI);
    assert_eq!(*calls.lock().unwrap(), indexes);
}

#[test]
fn resolve_bundle_bfs_descendants_inherit_full_chain() {
    let indexes = index_chain(
        ["https://entry.example/simple".to_string()],
        &["https://workspace.example/simple".to_string()],
        IndexPurpose::RootResolve,
    );
    let parent = PendingSource::Pypi {
        specifiers: VersionSpecifiers::empty(),
        indexes: indexes.clone(),
    };
    let inherited = bfs_descendant_indexes(&parent, &[]);
    assert_eq!(inherited, indexes);
    let git_parent = PendingSource::Git {
        url: "https://example.com/project.git".to_string(),
        rev: Some("abc123".to_string()),
        subdirectory: None,
    };
    assert_eq!(bfs_descendant_indexes(&git_parent, &indexes), indexes);
    let url_parent = PendingSource::Url {
        wheel_url: url::Url::parse("https://example.com/project.whl").unwrap(),
    };
    assert_eq!(bfs_descendant_indexes(&url_parent, &indexes), indexes);

    let mut grandchildren = std::collections::VecDeque::new();
    seed_worklist(
        &["root-grandchild>=1".to_string()],
        &[],
        &inherited,
        "root-",
        &std::collections::HashSet::new(),
        &mut grandchildren,
        None,
        &std::collections::HashSet::new(),
    )
    .unwrap();
    let grandchild = grandchildren
        .pop_front()
        .expect("the descendant BFS must seed root-grandchild");
    let PendingSource::Pypi {
        indexes: grandchild_indexes,
        ..
    } = grandchild.source
    else {
        panic!("the grandchild must remain a PyPI pending source");
    };
    assert_eq!(grandchild_indexes, indexes);
}

#[test]
fn tiered_cascade_step8_respects_retread_conda_deps() {
    // Step 8 must never reroute a dep the user explicitly forced to
    // the conda side. With empty channels the probe is indecisive,
    // which already gates step 8 -- so to pin the conda_deps guard
    // specifically, assert the predicate the gate uses.
    let mut effective = cfg();
    effective.conda_deps.push("isaacsim_kernel".into());
    let forced = effective
        .conda_deps
        .iter()
        .any(|n| canonical_conda_name(n) == "isaacsim-kernel");
    assert!(
        forced,
        "conda_deps guard must normalize names the way emission does"
    );
}

#[test]
fn built_in_win_only_dropped_on_linux() {
    // idna-ssl is in BUILT_IN_WIN_ONLY. Targeting linux-64, it must
    // not appear in run-deps even though it has no explicit
    // `sys_platform == "win32"` marker.
    let bundle = solo_bundle("isaacsim", vec!["idna-ssl==1.1.0", "numpy==1.26.0"]);
    let output =
        produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let names: Vec<String> = output
        .run_dependencies
        .depends
        .iter()
        .map(|d| d.name.clone())
        .collect();
    assert!(
        !names.iter().any(|n| n == "idna-ssl"),
        "idna-ssl auto-drop on linux failed; got: {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "numpy"),
        "numpy must still be emitted; got: {names:?}"
    );
}

#[test]
fn built_in_win_only_kept_on_windows() {
    // Same input, win-64 target. The auto-drop is non-Windows-only,
    // so idna-ssl is expected to remain.
    let bundle = solo_bundle("isaacsim", vec!["idna-ssl==1.1.0"]);
    let output = produce_output(&bundle, &cfg(), Platform::Win64, "3.11", &[], None, None).unwrap();
    let names: Vec<String> = output
        .run_dependencies
        .depends
        .iter()
        .map(|d| d.name.clone())
        .collect();
    assert!(
        names.iter().any(|n| n == "idna-ssl"),
        "idna-ssl should NOT be auto-dropped on win-64; got: {names:?}"
    );
}

#[test]
fn explicit_override_beats_built_in_win_only() {
    // If a user actually needs idna-ssl on linux, retread-overrides
    // is the documented escape hatch. Setting it to any spec must
    // cancel the built-in auto-drop.
    let mut config = cfg();
    config
        .overrides
        .insert("idna-ssl".to_string(), "*".to_string());
    let bundle = solo_bundle("isaacsim", vec!["idna-ssl==1.1.0"]);
    let output =
        produce_output(&bundle, &config, Platform::Linux64, "3.11", &[], None, None).unwrap();
    let names: Vec<String> = output
        .run_dependencies
        .depends
        .iter()
        .map(|d| d.name.clone())
        .collect();
    assert!(
        names.iter().any(|n| n == "idna-ssl"),
        "retread-overrides should cancel the auto-drop; got: {names:?}"
    );
}

#[test]
fn user_drop_deps_silently_drops() {
    // User-specified drop happens at debug level (no warn), unlike
    // the built-in auto-drop which warns. Behavior parity: dep is
    // not emitted.
    let mut config = cfg();
    config.drop_deps.push("requests".to_string());
    let bundle = solo_bundle("foo", vec!["requests==2.32.0", "numpy==1.26.0"]);
    let output =
        produce_output(&bundle, &config, Platform::Linux64, "3.11", &[], None, None).unwrap();
    let names: Vec<String> = output
        .run_dependencies
        .depends
        .iter()
        .map(|d| d.name.clone())
        .collect();
    assert!(
        !names.iter().any(|n| n == "requests"),
        "requests should be dropped per retread-drop-deps; got: {names:?}"
    );
    assert!(names.iter().any(|n| n == "numpy"));
}

#[test]
fn vendored_sub_packages_dropped_from_run_deps() {
    // Mirror the isaacsim bundle: primary depends on sub-packages,
    // sub-packages depend on each other, all are vendored together.
    let bundle = Bundle {
        conda_name: "isaacsim".into(),
        primary: rw(
            "isaacsim",
            meta(
                "isaacsim",
                "5.1.0.0",
                vec!["isaacsim-kernel==5.1.0.0 ; extra == \"all\""],
                true,
            ),
        ),
        extras: vec![
            rw(
                "isaacsim-kernel",
                meta(
                    "isaacsim-kernel",
                    "5.1.0.0",
                    vec!["numpy==1.26.0", "Pillow==11.3.0"],
                    true,
                ),
            ),
            rw(
                "isaacsim-core",
                meta(
                    "isaacsim-core",
                    "5.1.0.0",
                    vec!["isaacsim-kernel==5.1.0.0", "numpy==1.26.0", "scipy==1.15.3"],
                    true,
                ),
            ),
        ],
        probe_decisions: vec![],
        solve_diagnostics: BTreeMap::new(),
        auto_routed: vec![],
        auto_dropped: Default::default(),
        uv_closure_names: Default::default(),
        uv_dependency_graph: Default::default(),
        workspace_conda_versions: Default::default(),
        workspace_conda_provider_facts: Default::default(),
    };

    let output =
        produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let dep_names: Vec<String> = output
        .run_dependencies
        .depends
        .iter()
        .map(|d| d.name.clone())
        .collect();
    assert!(
        !dep_names.iter().any(|n| n == "isaacsim-kernel"),
        "isaacsim-kernel is vendored and must NOT appear in run-deps; got: {dep_names:?}"
    );
    assert!(
        !dep_names.iter().any(|n| n == "isaacsim-core"),
        "isaacsim-core is vendored and must NOT appear in run-deps; got: {dep_names:?}"
    );
    assert!(
        dep_names.iter().any(|n| n == "numpy"),
        "numpy must appear (deduped from multiple wheels); got: {dep_names:?}"
    );
    assert!(
        dep_names.iter().any(|n| n == "pillow"),
        "pillow must appear; got: {dep_names:?}"
    );
    assert!(
        dep_names.iter().any(|n| n == "scipy"),
        "scipy must appear; got: {dep_names:?}"
    );
}

#[tokio::test]
#[ignore = "spawns pip wheel; needs PATH with pip + setuptools (same prerequisites as path_source_does_not_corrupt_stdout)"]
async fn d_rewrites_metadata_on_the_wheel_the_recipe_will_source() {
    // Regression for the silent-D bug: ResolvedWheel.url used to
    // carry the pre-D URL (upstream for PyPI form, file:// of the
    // unrewritten built wheel for path / git). That meant the conda
    // emission saw the relaxed pins but the wheel rattler-build
    // actually copied into the conda package still had the strict
    // pins -- pixi forwarded conda's choice to uv as a hard pin, uv
    // then read site-packages METADATA and found a conflicting
    // strict pin, and the solve failed with the classic "PyPI
    // packages have been pinned by the conda solve" error.
    // Contract: after a non-None relax policy, ResolvedWheel.url
    // points at a file:// path whose METADATA matches what we
    // emitted into conda run-deps.
    let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sample_with_buildtime_dep");
    let tmp = std::env::temp_dir().join(format!("retread-d-on-disk-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let entry = crate::config::WheelEntry {
        path: Some(fixture.to_string_lossy().to_string()),
        ..Default::default()
    };
    let target = crate::pypi::ResolutionTarget::for_subdir("3.11", "linux-64");

    let (resolved, original_rd) = materialize_and_rewrite(
        &entry,
        "retread-sample",
        None,
        &target,
        &tmp,
        &fixture,
        &tmp,
        RelaxPolicy::Minor,
        &std::collections::BTreeMap::new(),
        None,
        EntryAuditInfo::default(),
    )
    .await
    .expect("materialize_and_rewrite");

    assert_eq!(
        resolved.url.scheme(),
        "file",
        "recipe source URL must be file:// of the rewritten wheel; got {}",
        resolved.url,
    );
    let on_disk = resolved.url.to_file_path().expect("file path from URL");
    let on_disk_meta =
        crate::wheel::read_metadata(&on_disk).expect("read METADATA from wheel-on-disk");

    // v1.5.9: the ORIGINAL (pre-D) lines keep the exact upstream pin
    // -- the BFS resolves sub-wheels from these so pinned families
    // stay patch-consistent -- while the on-disk wheel carries the
    // relaxed form for the uv/emission side.
    assert!(
        original_rd.iter().any(|l| l.contains("==0.49.1")),
        "original requires_dist must keep the exact pin; got {original_rd:?}",
    );

    let starlette_lines: Vec<&String> = on_disk_meta
        .requires_dist
        .iter()
        .filter(|l| l.to_lowercase().contains("starlette"))
        .collect();
    assert!(
        !starlette_lines.is_empty(),
        "fixture must declare starlette; got Requires-Dist: {:?}",
        on_disk_meta.requires_dist,
    );
    assert!(
        starlette_lines.iter().all(|l| !l.contains("==0.49.1")),
        "wheel on disk must have starlette pin relaxed (minor relax => `>=0.49,<1`), \
             but at least one Requires-Dist still reads `==0.49.1`: {:?}",
        on_disk_meta.requires_dist,
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn bundle_field_groups_entries_into_one_output() {
    // Contract: two [retread-wheels] entries with the same `bundle`
    // field collapse to ONE conda output containing both wheels.
    // Verified at the produce_output level by constructing a Bundle
    // whose conda_name is the bundle group's name and whose
    // primary+extras list covers wheels from both source entries.
    //
    // resolve_all's grouping is integration-shaped (it shells out
    // to PyPI/pip); the contract this test pins is the downstream
    // behavior produce_output exposes: the bundle's conda_name
    // drives the output name, and all wheels in the bundle
    // contribute their Requires-Dist to the merged run-deps.
    let bundle = Bundle {
        conda_name: "isaac-pack".into(),
        primary: rw(
            "isaacsim",
            meta("isaacsim", "5.1.0.0", vec!["numpy==1.26.0"], true),
        ),
        extras: vec![
            rw(
                "isaaclab",
                meta("isaaclab", "0.51.1", vec!["scipy==1.15.0"], true),
            ),
            rw(
                "pytorch3d",
                meta(
                    "pytorch3d",
                    "0.7.8+5043d15pt2.7.0cu128",
                    vec!["pillow==11.0.0"],
                    true,
                ),
            ),
        ],
        probe_decisions: vec![],
        solve_diagnostics: BTreeMap::new(),
        auto_routed: vec![],
        auto_dropped: Default::default(),
        uv_closure_names: Default::default(),
        uv_dependency_graph: Default::default(),
        workspace_conda_versions: Default::default(),
        workspace_conda_provider_facts: Default::default(),
    };

    let output =
        produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();

    // Output name is the bundle's conda_name, not any one entry name.
    assert_eq!(
        output.metadata.name.as_normalized(),
        "isaac-pack",
        "merged bundle's conda output should be named after the bundle group",
    );
    // All three wheels' Requires-Dist flow into the merged run-deps.
    let dep_names: Vec<String> = output
        .run_dependencies
        .depends
        .iter()
        .map(|d| d.name.clone())
        .collect();
    for sib_dep in ["numpy", "scipy", "pillow"] {
        assert!(
            dep_names.iter().any(|n| n == sib_dep),
            "merged bundle should emit {sib_dep} from one of its wheels; got: {dep_names:?}",
        );
    }
}

// Regression: handoff-2026-05-24. The merged-bundle primary (alphabetically
// first entry in [retread-wheels]) was `isaaclab` -- a `py3-none-any`
// wheel. After D rewrite the file on disk is named
// `isaaclab-0.51.1-py3-none-any.relaxed.whl`. `WheelMetadata.is_pure_python`
// was computed from `filename.contains("-none-any.whl")`, which is false
// on the relaxed filename. `produce_output` then took the wheel-tag
// fallback branch and `python_version_from_wheel_tag` returned bare
// `"3"` (from the `py3` tag), so the emitted run-dep was `python 3.*`.
// The conda solver, given `python 3.*`, picked python 3.14 and implied
// `python_abi 3.14.* *_cp314`, which collided with the workspace's
// `python==3.11` pin -- producing the user-facing error:
//   isaac-pack 0.51.1 would require python_abi 3.14.* *_cp314
// Fix: `is_pure_python` is now derived from the canonical PEP 425 tag
// via `wheel::is_pure_python_wheel_filename`, which strips the
// cosmetic `.relaxed.whl` suffix before checking the platform tag.
// This test pins the *end-to-end* contract: a relaxed pure-Python
// primary must yield `python {workspace_python_version}.*`, NOT
// `python 3.*`. Previously only `wheel.rs` had unit coverage of the
// helper; nothing asserted the downstream produce_output behavior, so
// the bug shipped silently.
#[test]
fn relaxed_pure_python_primary_pins_python_to_workspace_variant() {
    use crate::wheel::is_pure_python_wheel_filename;

    // Construct a bundle whose primary mirrors the post-D state of
    // the `isaaclab` wheel that fooled the previous filename check.
    let filename = "isaaclab-0.51.1-py3-none-any.relaxed.whl".to_string();
    // Helper-driven is_pure_python -- this is exactly what wheel.rs's
    // read_metadata path produces for the on-disk relaxed wheel.
    let is_pure = is_pure_python_wheel_filename(&filename);
    assert!(
        is_pure,
        "regression guard: helper must report relaxed py3-none-any wheel as pure"
    );
    let primary = WheelMetadata {
        name: "isaaclab".into(),
        version: "0.51.1".into(),
        requires_dist: vec![],
        retread_conda_run_dependencies: vec![],
        is_pure_python: is_pure,
        sha256: "sha".into(),
        filename,
    };
    let bundle = Bundle {
        conda_name: "isaac-pack".into(),
        primary: ResolvedWheel {
            pypi_name: "isaaclab".into(),
            url: "https://example.com/isaaclab-0.51.1-py3-none-any.relaxed.whl"
                .parse()
                .unwrap(),
            upstream_url: Some(
                "https://example.com/isaaclab-0.51.1-py3-none-any.relaxed.whl"
                    .parse()
                    .unwrap(),
            ),
            git_source: None,
            sdist_source: None,
            metadata_provenance: Provenance::SourceBuiltRelaxed,
            metadata: primary,
            extras_requested: vec![],
            auto_data: None,
            auto_data_dedup_skipped_root: None,
        },
        extras: vec![],
        probe_decisions: vec![],
        solve_diagnostics: BTreeMap::new(),
        auto_routed: vec![],
        auto_dropped: Default::default(),
        uv_closure_names: Default::default(),
        uv_dependency_graph: Default::default(),
        workspace_conda_versions: Default::default(),
        workspace_conda_provider_facts: Default::default(),
    };

    let output =
        produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();

    // The conda output's variant must be the workspace's 3.11, not the
    // bare-major "3" parsed from the py3 tag.
    let variant_python = output
        .metadata
        .variant
        .get("python")
        .map(|v| v.to_string())
        .expect("variant.python must be set");
    assert_eq!(
        variant_python, "3.11",
        "variant.python must inherit the workspace's variant, not the wheel tag's bare major"
    );

    // Run-dep must be `python 3.11.*`, NOT `python 3.*` (which lets the
    // solver pick 3.14 and triggers the python_abi 3.14 collision).
    // rattler's NamelessMatchSpec Debug format is structural:
    //   StrictRange(StartsWith, StrictVersion(Version { version: [[0], [3], [11]], local: [] }))
    // so we search for the segment list `[0], [3], [11]` -- bare-major
    // would render as `[0], [3]` with only TWO components. Anchor with
    // both `[3]` and `[11]` to catch the minor; absence of `[11]` is the
    // smoking gun for the regression we just fixed.
    let python_spec = output
        .run_dependencies
        .depends
        .iter()
        .find(|d| d.name == "python")
        .map(|d| format!("{:?}", d.spec))
        .expect("python in run_deps");
    assert!(
        python_spec.contains("[3]") && python_spec.contains("[11]"),
        "run-dep python must pin to 3.11 (look for [3] and [11] segments); got: {python_spec}"
    );
    assert!(
        python_spec.contains("StartsWith"),
        "run-dep python must use StartsWith range (== `3.11.*`); got: {python_spec}"
    );

    // build_v1 reads the python version from the output's variant when
    // constructing the recipe; so as long as variant.python is right,
    // the recipe is too. The variant assertion above already covers
    // this -- pinning the contract here in case the read path changes.
}

#[test]
fn bare_major_workspace_python_input_still_emits_a_minor_anchor() {
    let bundle = solo_bundle("foo", vec![]);
    let output = produce_output(&bundle, &cfg(), Platform::Linux64, "3", &[], None, None).unwrap();
    let emitted = output
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "python")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("python run dependency");
    assert!(
        !is_bare_major_spec(&emitted),
        "post-emission ABI check requires a minor anchor, got {emitted}"
    );
}

#[test]
fn bare_major_abi_anchor_emission_auto_completes_within_major_cap_and_warning() {
    for original in [">=2.0", ">=2"] {
        // The deps-from route suppresses the incidental selected-version pin;
        // the constraint itself retains its real index-wheel provenance.
        let mut numpy = bundle_auto_route("numpy", "2.2.6", Provenance::DepsFromRelaxed);
        numpy
            .route
            .input_requirements
            .push(crate::uv_closure::AutoRouteInputRequirement {
                specifiers: original.to_string(),
                source: format!("wheel `jax==0.7.2` Requires-Dist `numpy{original}`"),
                provenance: Provenance::IndexWheelMetadata,
                role: crate::uv_closure::AutoRouteInputRole::Requirement,
            });
        let mut bundle = solo_bundle("flashsac-pack", vec![]);
        bundle.auto_routed.push(numpy);

        let (output, warnings) = produce_output_pending_relaxations(
            &bundle,
            &cfg(),
            Platform::Linux64,
            "3.11",
            &[],
            None,
            None,
        )
        .unwrap();
        let emitted = output
            .run_dependencies
            .depends
            .iter()
            .find(|dependency| dependency.name == "numpy")
            .map(|dependency| format_packagespec(&dependency.spec))
            .expect("NumPy run dependency");
        assert_eq!(emitted, ">=2.0,<3", "{original}");
        assert_eq!(warnings.len(), 1, "{original}: {warnings:?}");

        let warning = warnings[0].to_string();
        assert!(warning.contains("RETREAD AUTO-COMPLETED ABI anchor"));
        assert!(warning.contains("bundle `flashsac-pack`"));
        assert!(warning.contains("package `numpy`"));
        assert!(warning.contains(&format!("`{original}` -> `>=2.0,<3`")));
        assert!(warning.contains("wheel `jax==0.7.2`"));

        let manifest = bundled_relaxations_for_output(
            "flashsac-pack",
            "flashsac-pack",
            &ResolutionTarget::for_subdir("3.11", "linux-64"),
            &[],
            &warnings,
        )
        .expect("ABI-anchor completion must reach the courier warning payload");
        assert_eq!(manifest.records().len(), 1);
        assert_eq!(
            manifest.records()[0].kind,
            crate::relaxation_record::RelaxationRecordKind::AbiAnchorCapCompleted
        );
        let hook = manifest.activate_script();
        assert!(hook.contains("bundle `flashsac-pack`"));
        assert!(hook.contains(&format!("numpy {original} -> >=2.0,<3")));
    }
}

#[test]
fn abi_anchor_exact_pin_normalizes_to_within_minor_band() {
    for (original, normalized) in [
        ("==1.26.0", ">=1.26,<1.27"),
        (">=1.23.5,==1.26.0,~=1.26.0,<2.5", ">=1.26.0,<1.27"),
        ("==2.0.0", ">=2.0,<2.1"),
    ] {
        assert_eq!(
            widen_exact_abi_anchor_spec_to_minor_band(original).as_deref(),
            Some(normalized),
            "{original}"
        );
    }

    let normalized = widen_exact_abi_anchor_spec_to_minor_band(">=1.23.5,==1.26.0,~=1.26.0,<2.5")
        .expect("joined exact anchor must normalize");
    let normalized = VersionSpecifiers::from_str(&normalized).unwrap();
    assert!(normalized.contains(&Version::from_str("1.26.0").unwrap()));
    assert!(
        normalized.contains(&Version::from_str("1.26.4").unwrap()),
        "higher patches in the same NumPy ABI band must remain selectable"
    );
    assert!(!normalized.contains(&Version::from_str("1.27").unwrap()));

    for not_an_exact_pin in [">=1.26,<1.27", "==1.26.*", ">=1.26.0,<=1.26.0"] {
        assert!(
            widen_exact_abi_anchor_spec_to_minor_band(not_an_exact_pin).is_none(),
            "{not_an_exact_pin}"
        );
    }

    let excluded = widen_exact_abi_anchor_spec_to_minor_band("==1.26.0,!=1.26.2")
        .expect("an exclusion must survive exact-pin banding");
    let excluded = VersionSpecifiers::from_str(&excluded).unwrap();
    assert!(!excluded.contains(&Version::from_str("1.26.2").unwrap()));
    assert!(excluded.contains(&Version::from_str("1.26.4").unwrap()));
}

#[test]
fn abi_anchor_exact_pin_band_preserves_joined_source_upper_bound() {
    let mut bundle = solo_bundle("anchor-cobound-pack", vec!["numpy<=1.26.0"]);
    bundle.auto_routed.push(bundle_auto_route(
        "numpy",
        "1.26.0",
        Provenance::PriorSelection,
    ));

    let (output, warnings) = produce_output_pending_relaxations(
        &bundle,
        &cfg(),
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    let emitted = output
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "numpy")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("NumPy run dependency");

    assert_eq!(emitted, ">=1.26,<=1.26.0");
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    let emitted =
        VersionSpec::from_str(&emitted, rattler_conda_types::ParseStrictness::Lenient).unwrap();
    assert!(
        !emitted.matches(&rattler_conda_types::Version::from_str("1.26.4").unwrap()),
        "the independent <=1.26.0 source bound must remain load-bearing"
    );
}

#[test]
fn abi_anchor_exact_pin_band_resolves_cross_pack_patch_shape() {
    let mut isaaclab = solo_bundle(
        "isaaclab-2.3x-pack",
        vec!["numpy>=1.23.5", "numpy~=1.26.0", "numpy<2.5"],
    );
    isaaclab.auto_routed.push(bundle_auto_route(
        "numpy",
        "1.26.0",
        Provenance::PriorSelection,
    ));
    let (output, warnings) = produce_output_pending_relaxations(
        &isaaclab,
        &cfg(),
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    let emitted = output
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "numpy")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("isaaclab NumPy run dependency");
    assert_eq!(emitted, ">=1.26.0,<1.27");
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0]
            .to_string()
            .contains("RETREAD AUTO-WIDENED ABI anchor exact pin")
    );

    let candidate = rattler_conda_types::Version::from_str("1.26.4").unwrap();
    let isaaclab =
        VersionSpec::from_str(&emitted, rattler_conda_types::ParseStrictness::Lenient).unwrap();
    let protomotions = VersionSpec::from_str(
        ">=1.26.4,<2.7",
        rattler_conda_types::ParseStrictness::Lenient,
    )
    .unwrap();
    assert!(
        isaaclab.matches(&candidate) && protomotions.matches(&candidate),
        "NumPy 1.26.4 must satisfy both independently-emitted pack constraints"
    );
}

#[test]
fn abi_anchor_cap_completion_intersects_compatible_clauses_and_stays_source_scoped() {
    for (original, normalized) in [
        (">=2.0,!=2.1", ">=2.0,!=2.1,<3"),
        (">=2.0,<4", ">=2.0,<4,<3"),
        (">=2,<3", ">=2.0,<3"),
    ] {
        assert_eq!(
            auto_complete_bare_major_abi_anchor_spec(original).as_deref(),
            Some(normalized),
            "{original}"
        );
        assert!(
            is_auto_completed_abi_anchor_spec(normalized),
            "{normalized}"
        );
    }
    for already_safe in [">=2.0,<3", ">=2.0,<3,!=2.1"] {
        assert!(
            auto_complete_bare_major_abi_anchor_spec(already_safe).is_none(),
            "{already_safe}"
        );
        assert!(
            is_auto_completed_abi_anchor_spec(already_safe),
            "{already_safe}"
        );
    }
    for irreconcilable in ["!", "<3", ">=2.0,<2", ">=18446744073709551615"] {
        assert!(
            auto_complete_bare_major_abi_anchor_spec(irreconcilable).is_none(),
            "{irreconcilable}"
        );
    }

    assert!(
        check_output_abi_invariants(
            &[("numpy".to_string(), ">=2.0,<3".to_string())],
            &[],
            &BTreeMap::new(),
            &BTreeMap::new(),
            &AbiAliasGraph::new(),
        )
        .is_empty()
    );
    // Epoch 46: emit-pypi's anchor floor completion writes the canonical
    // within-major band into shipped wheel METADATA, so the invariant accepts
    // the canonical form from embedded origins too — no longer source-scoped
    // to retread-emitted conda contracts.
    let embedded = check_output_abi_invariants(
        &[],
        &[("consumer-wheel".to_string(), "numpy>=2.0,<3".to_string())],
        &BTreeMap::new(),
        &BTreeMap::new(),
        &AbiAliasGraph::new(),
    );
    assert!(embedded.is_empty(), "{embedded:?}");
    // A genuinely open bare-major embedded anchor still violates.
    let open_bare = check_output_abi_invariants(
        &[],
        &[("consumer-wheel".to_string(), "numpy>=2".to_string())],
        &BTreeMap::new(),
        &BTreeMap::new(),
        &AbiAliasGraph::new(),
    );
    assert_eq!(open_bare.len(), 1, "{open_bare:?}");
    assert!(open_bare[0].contains("wheel `consumer-wheel` embeds"));
}

#[test]
fn abi_anchor_cap_completion_normalizes_qualified_version_floors() {
    for (original, normalized) in [
        (">=2!2.0", ">=2!2.0,<2!3"),
        (">=2!2.0rc1", ">=2!2.0rc1,<2!3"),
        (">=2.0rc1", ">=2.0rc1,<3"),
        (">=2.0a1", ">=2.0a1,<3"),
        (">=2.0b2", ">=2.0b2,<3"),
        (">=2.0.post1", ">=2.0.post1,<3"),
        (">=2.0.dev0", ">=2.0.dev0,<3"),
    ] {
        assert_eq!(
            auto_complete_bare_major_abi_anchor_spec(original).as_deref(),
            Some(normalized),
            "{original}: bare-major={}",
            is_bare_major_spec(original),
        );
        assert!(
            is_auto_completed_abi_anchor_spec(normalized),
            "{normalized}"
        );
    }
    for already_capped in [
        ">=2!2.0,<2!3",
        ">=2.0rc1,<3",
        ">=2.0.post1,<2.3",
        ">=2.0.dev0,~=2.1",
    ] {
        assert!(
            auto_complete_bare_major_abi_anchor_spec(already_capped).is_none(),
            "{already_capped}",
        );
    }
    assert!(auto_complete_bare_major_abi_anchor_spec("==2.0rc1").is_none());
}

#[cfg(unix)]
#[test]
fn abi_anchor_cap_completion_sha_bound_lock_replay_round_trips() {
    const BUNDLE: &str = "abi-cap-replay-pack";
    const VERSION: &str = "1.0.0";
    const INPUTS_HASH: &str = "abi-cap-record-hash";
    const ORIGINAL: &str = ">=2.0";
    const NORMALIZED: &str = ">=2.0,<3";

    let _env_guard = TEST_ENV_MUTEX.lock().unwrap();
    let target = ResolutionTarget::from_wheel_target(
        crate::pypi::WheelTarget {
            python_version: "3.11".to_string(),
            conda_subdir: "linux-64".to_string(),
            max_glibc: None,
        },
        None,
    );
    let completion = auto_bundle::abi_anchor_cap_completion(
        BUNDLE,
        &PypiKey::from_pypi("numpy"),
        ORIGINAL,
        NORMALIZED,
        vec!["wheel `consumer==1` Requires-Dist `numpy>=2.0`".to_string()],
    );
    let manifest = bundled_relaxations_for_output(BUNDLE, BUNDLE, &target, &[], &[completion])
        .expect("ABI-anchor completion must produce a durable manifest");
    assert_eq!(
        manifest.schema_version,
        crate::relaxation_record::RELAXATION_MANIFEST_SCHEMA
    );
    assert_eq!(crate::relaxation_record::RELAXATION_MANIFEST_SCHEMA, 2);

    let sha256 = "11".repeat(32);
    let filename = format!("{}-{VERSION}-py3-none-any.whl", BUNDLE.replace('-', "_"));
    let lock = crate::lock::RetreadLock {
        schema: crate::lock::SCHEMA,
        retread_version: env!("CARGO_PKG_VERSION").to_string(),
        bundle: BUNDLE.to_string(),
        version: VERSION.to_string(),
        python: "3.11".to_string(),
        target_subdir: "linux-64".to_string(),
        target_contract: None,
        target_identity: None,
        target_scope: None,
        exact_workspace_envelope: false,
        inputs_hash: INPUTS_HASH.to_string(),
        root_requirements: vec![format!("{BUNDLE}=={VERSION}")],
        wheels: vec![crate::lock::LockWheel {
            name: BUNDLE.to_string(),
            version: VERSION.to_string(),
            origin: crate::lock::Origin::Index,
            filename: filename.clone(),
            url: Some(format!("https://example.com/{filename}")),
            sha256: Some(sha256.clone()),
            requires_dist: vec![],
            must_ship: false,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }],
        abi_context: Some(crate::lock::LockAbiContext {
            wheels: vec![crate::lock::LockWheelAbiMetadata {
                name: BUNDLE.to_string(),
                sha256,
                requires_dist: vec![],
            }],
        }),
        relaxations: manifest.records().to_vec(),
        conda_run_deps: vec![crate::lock::CondaDep {
            name: "numpy".to_string(),
            spec: NORMALIZED.to_string(),
        }],
        index_urls: vec!["https://pypi.org/simple/".to_string()],
        prerelease: BTreeMap::new(),
        shadow_libs: BTreeMap::new(),
        declared_glibc: None,
        resolution_glibc: None,
        conda_capable: vec!["numpy".to_string()],
        entry_specs: vec![format!("{BUNDLE}=={VERSION}")],
        wheel_store: None,
    };

    let root = unique_test_dir("abi-cap-record-replay");
    std::fs::create_dir_all(&root).unwrap();
    let lock_path = root.join(crate::lock::RetreadLock::file_name_for_target(
        BUNDLE, &target,
    ));
    std::fs::write(&lock_path, lock.to_pretty_json().unwrap()).unwrap();

    let reloaded = crate::lock::RetreadLock::load(&lock_path).unwrap();
    assert_eq!(reloaded.schema, crate::lock::SCHEMA);
    assert_eq!(crate::lock::SCHEMA, 18);
    assert_eq!(reloaded.relaxations, manifest.records());
    assert_eq!(
        reloaded.relaxations[0].kind,
        crate::relaxation_record::RelaxationRecordKind::AbiAnchorCapCompleted
    );
    let replay_manifest = RelaxationManifest::new(BUNDLE, reloaded.relaxations.clone()).unwrap();
    replay_manifest.validate_for(BUNDLE, &target).unwrap();

    let replay = replay_from_lock_for_target(
        &lock_path,
        INPUTS_HASH,
        true,
        &target,
        BUNDLE,
        Platform::Linux64,
        0,
        false,
        &[],
        &BTreeMap::new(),
        &BTreeMap::new(),
        &AbiAliasGraph::new(),
    )
    .expect("schema-current SHA-bound lock must validate without stale relaxation errors")
    .expect("matching SHA-bound lock must replay");
    let numpy = replay
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "numpy")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("replayed NumPy run dependency");
    assert_eq!(numpy, NORMALIZED);

    std::fs::remove_dir_all(root).ok();
}

#[cfg(unix)]
#[test]
fn abi_anchor_exact_pin_widening_sha_bound_lock_replay_round_trips() {
    const BUNDLE: &str = "abi-exact-replay-pack";
    const VERSION: &str = "1.0.0";
    const INPUTS_HASH: &str = "abi-exact-record-hash";
    const ORIGINAL: &str = "==1.26.0";
    const NORMALIZED: &str = ">=1.26,<1.27";

    let _env_guard = TEST_ENV_MUTEX.lock().unwrap();
    let target = ResolutionTarget::from_wheel_target(
        crate::pypi::WheelTarget {
            python_version: "3.11".to_string(),
            conda_subdir: "linux-64".to_string(),
            max_glibc: None,
        },
        None,
    );
    let mut bundle = solo_bundle(BUNDLE, vec![]);
    bundle.auto_routed.push(bundle_auto_route(
        "numpy",
        "1.26.0",
        Provenance::PriorSelection,
    ));
    let (output, warnings) = produce_output_pending_relaxations(
        &bundle,
        &cfg(),
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    let emitted = output
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "numpy")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("emitted NumPy run dependency");
    assert_eq!(emitted, NORMALIZED);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    let warning = warnings[0].to_string();
    assert!(
        warning.contains("RETREAD AUTO-WIDENED ABI anchor exact pin"),
        "{warning}"
    );
    assert!(
        warning.contains(&format!("`{ORIGINAL}` -> `{NORMALIZED}`")),
        "{warning}"
    );

    let manifest = bundled_relaxations_for_output(BUNDLE, BUNDLE, &target, &[], &warnings)
        .expect("ABI-anchor exact-pin widening must produce a durable manifest");
    assert_eq!(
        manifest.schema_version,
        crate::relaxation_record::RELAXATION_MANIFEST_SCHEMA
    );
    assert_eq!(crate::relaxation_record::RELAXATION_MANIFEST_SCHEMA, 2);
    assert_eq!(manifest.records().len(), 1);
    assert_eq!(manifest.records()[0].original_spec, ORIGINAL);
    assert_eq!(manifest.records()[0].resulting_spec, NORMALIZED);
    assert_eq!(
        manifest.records()[0].kind,
        crate::relaxation_record::RelaxationRecordKind::ExactPinWidened
    );

    let sha256 = "22".repeat(32);
    let filename = format!("{}-{VERSION}-py3-none-any.whl", BUNDLE.replace('-', "_"));
    let lock = crate::lock::RetreadLock {
        schema: crate::lock::SCHEMA,
        retread_version: env!("CARGO_PKG_VERSION").to_string(),
        bundle: BUNDLE.to_string(),
        version: VERSION.to_string(),
        python: "3.11".to_string(),
        target_subdir: "linux-64".to_string(),
        target_contract: None,
        target_identity: None,
        target_scope: None,
        exact_workspace_envelope: false,
        inputs_hash: INPUTS_HASH.to_string(),
        root_requirements: vec![format!("{BUNDLE}=={VERSION}")],
        wheels: vec![crate::lock::LockWheel {
            name: BUNDLE.to_string(),
            version: VERSION.to_string(),
            origin: crate::lock::Origin::Index,
            filename: filename.clone(),
            url: Some(format!("https://example.com/{filename}")),
            sha256: Some(sha256.clone()),
            requires_dist: vec![],
            must_ship: false,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }],
        abi_context: Some(crate::lock::LockAbiContext {
            wheels: vec![crate::lock::LockWheelAbiMetadata {
                name: BUNDLE.to_string(),
                sha256,
                requires_dist: vec![],
            }],
        }),
        relaxations: manifest.records().to_vec(),
        conda_run_deps: vec![crate::lock::CondaDep {
            name: "numpy".to_string(),
            spec: NORMALIZED.to_string(),
        }],
        index_urls: vec!["https://pypi.org/simple/".to_string()],
        prerelease: BTreeMap::new(),
        shadow_libs: BTreeMap::new(),
        declared_glibc: None,
        resolution_glibc: None,
        conda_capable: vec!["numpy".to_string()],
        entry_specs: vec![format!("{BUNDLE}=={VERSION}")],
        wheel_store: None,
    };

    let root = unique_test_dir("abi-exact-record-replay");
    std::fs::create_dir_all(&root).unwrap();
    let lock_path = root.join(crate::lock::RetreadLock::file_name_for_target(
        BUNDLE, &target,
    ));
    std::fs::write(&lock_path, lock.to_pretty_json().unwrap()).unwrap();

    let reloaded = crate::lock::RetreadLock::load(&lock_path).unwrap();
    assert_eq!(reloaded.schema, crate::lock::SCHEMA);
    assert_eq!(crate::lock::SCHEMA, 18);
    assert_eq!(reloaded.relaxations, manifest.records());
    assert_eq!(
        reloaded.relaxations[0].kind,
        crate::relaxation_record::RelaxationRecordKind::ExactPinWidened
    );
    let replay_manifest = RelaxationManifest::new(BUNDLE, reloaded.relaxations.clone()).unwrap();
    replay_manifest.validate_for(BUNDLE, &target).unwrap();

    let replay = replay_from_lock_for_target(
        &lock_path,
        INPUTS_HASH,
        true,
        &target,
        BUNDLE,
        Platform::Linux64,
        0,
        false,
        &[],
        &BTreeMap::new(),
        &BTreeMap::new(),
        &AbiAliasGraph::new(),
    )
    .expect("schema-current SHA-bound lock must validate the exact-pin relaxation record")
    .expect("matching SHA-bound exact-pin lock must replay");
    let numpy = replay
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "numpy")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("replayed NumPy run dependency");
    assert_eq!(numpy, NORMALIZED);

    std::fs::remove_dir_all(root).ok();
}

#[test]
fn abi_anchor_emission_preserves_banded_anchor_and_exact_non_anchor_specs() {
    let mut config = cfg();
    config.relax = RelaxPolicy::None;
    let mut numpy = bundle_auto_route("numpy", "2.0.1", Provenance::DepsFromRelaxed);
    numpy
        .route
        .input_requirements
        .push(crate::uv_closure::AutoRouteInputRequirement {
            specifiers: ">=2.0,<2.3".to_string(),
            source: "typed NumPy input `>=2.0,<2.3`".to_string(),
            provenance: Provenance::DepsFromRelaxed,
            role: crate::uv_closure::AutoRouteInputRole::Requirement,
        });
    let mut bundle = solo_bundle("anchor-pack", vec![]);
    bundle.auto_routed.push(numpy);
    let (output, warnings) = produce_output_pending_relaxations(
        &bundle,
        &config,
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    let emitted = output
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "numpy")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("NumPy run dependency");
    assert_eq!(emitted, ">=2.0,<2.3");
    assert!(warnings.is_empty(), "{warnings:?}");

    let bundle = solo_bundle("ordinary-pack", vec!["packaging==2.0.1"]);
    let (output, warnings) = produce_output_pending_relaxations(
        &bundle,
        &config,
        Platform::Linux64,
        "3.11",
        &[],
        None,
        None,
    )
    .unwrap();
    let emitted = output
        .run_dependencies
        .depends
        .iter()
        .find(|dependency| dependency.name == "packaging")
        .map(|dependency| format_packagespec(&dependency.spec))
        .expect("packaging run dependency");
    assert_eq!(emitted, "==2.0.1");
    assert!(warnings.is_empty());
}

#[test]
fn abi_anchor_cap_completion_leaves_unparseable_specs_fail_closed() {
    assert!(auto_complete_bare_major_abi_anchor_spec("!").is_none());
    for spec in [
        "!",
        "===2.0.0",
        "==1!2.0.0",
        "==2.0rc1",
        "==2.0.post1",
        "==2.0.dev0",
        "==2.0.0+local",
        "==2.18446744073709551615",
    ] {
        assert!(
            widen_exact_abi_anchor_spec_to_minor_band(spec).is_none(),
            "{spec}"
        );
    }
    let violations = check_output_abi_invariants(
        &[("numpy".to_string(), "!".to_string())],
        &[],
        &BTreeMap::new(),
        &BTreeMap::new(),
        &AbiAliasGraph::new(),
    );
    assert_eq!(violations.len(), 1, "{violations:?}");
    assert!(violations[0].contains("ABI invariant"));
    assert!(violations[0].contains("bare-major"));
}

#[test]
fn output_abi_invariant_rejects_star_bare_major_and_uncovered_workspace_pin() {
    let overrides = BTreeMap::new();
    let python_workspace =
        BTreeMap::from([("python".to_string(), BTreeSet::from(["3.11".to_string()]))]);
    for spec in [
        "",
        "*",
        "3.*",
        ">=3",
        "3.*|4.*",
        ">=3,<4,!=3.5",
        ">=3,<4,>0.1",
        ">0.1,>=3,<4",
        ">=3,<4,<99.1",
    ] {
        let violations = check_output_abi_invariants(
            &[("python".to_string(), spec.to_string())],
            &[],
            &python_workspace,
            &overrides,
            &BTreeMap::new(),
        );
        assert_eq!(violations.len(), 1, "{spec}: {violations:?}");
    }

    let uncovered = check_output_abi_invariants(
        &[("python".to_string(), ">=3.12,<4".to_string())],
        &[],
        &python_workspace,
        &overrides,
        &BTreeMap::new(),
    );
    assert_eq!(uncovered.len(), 1, "{uncovered:?}");
    assert!(uncovered[0].contains("does not cover workspace pin"));
}

#[test]
fn bare_major_detection_uses_effective_parsed_bounds() {
    for spec in [
        ">=3,<4,>0.1",
        ">0.1,>=3,<4",
        ">=3,<4,<99.1",
        ">=3.0,<4",
        ">=3,<4.0",
        "<4.0",
    ] {
        assert!(
            is_bare_major_spec(spec),
            "redundant textual minor bound must not hide bare-major range `{spec}`"
        );
    }
    for spec in [
        ">=3,<4,>3.1",
        ">=3,<3.12",
        ">=3.11,<4",
        ">=3.0.1,<4",
        "==12.0",
    ] {
        assert!(
            !is_bare_major_spec(spec),
            "effective minor bound must protect narrowed range `{spec}`"
        );
    }
    for spec in ["", "!", ">=4,<3", ">=3,<"] {
        assert!(
            is_bare_major_spec(spec),
            "empty, unsatisfiable, or unparseable ABI constraint must fail closed: `{spec}`"
        );
    }
}

#[test]
fn bare_major_detection_accepts_single_segment_exact_pin() {
    assert!(
        !is_bare_major_spec("==3"),
        "an exact equality is concrete regardless of release segment count"
    );
}

#[test]
fn bare_major_detection_fails_closed_for_exclusion_emptied_ranges() {
    for spec in [
        "==3.1,!=3.1",
        ">=3.1,<3.2,!=3.1.*",
        "==3.1.0dev1,!=3.1.*",
        "!=3.1",
    ] {
        assert!(
            is_bare_major_spec(spec),
            "empty or exclusion-only ABI constraint must fail closed: `{spec}`"
        );
    }
}

#[test]
fn bare_major_detection_fails_closed_when_prefix_ceiling_overflows() {
    let spec = "!=3.18446744073709551615.*";
    assert!(
        is_bare_major_spec(spec),
        "an unrepresentable prefix ceiling must fail closed: `{spec}`"
    );

    let requirement = format!("numpy{spec}");
    let violations = check_output_abi_invariants(
        &[],
        &[("consumer-wheel".to_string(), requirement.clone())],
        &BTreeMap::new(),
        &BTreeMap::new(),
        &AbiAliasGraph::new(),
    );
    assert_eq!(violations.len(), 1, "{requirement}: {violations:?}");
    assert!(
        violations[0].contains("unsatisfiable under PEP 440"),
        "{requirement}: {violations:?}"
    );
}

#[test]
fn bare_major_effective_ranges_model_negative_compatible_exclusion() {
    use rattler_conda_types::version_spec::{LogicalOperator, StrictRangeOperator};
    use rattler_conda_types::{StrictVersion, Version};
    use std::str::FromStr as _;

    let version = Version::from_str("3.1").unwrap();
    let contradictory = VersionSpec::Group(
        LogicalOperator::And,
        vec![
            VersionSpec::StrictRange(
                StrictRangeOperator::Compatible,
                StrictVersion(version.clone()),
            ),
            VersionSpec::StrictRange(StrictRangeOperator::NotCompatible, StrictVersion(version)),
        ],
    );
    assert!(
        effective_version_ranges(&contradictory).is_empty(),
        "compatible and negative-compatible clauses must form an empty branch"
    );
}

#[test]
fn bare_major_detection_checks_prefix_and_compatible_predicates() {
    for spec in [
        "3.1.*,==3.2a1",
        "3.1.*,==3.1post1",
        "~=3.1,==4a1",
        "3.1.*,>=3.2a1,<3.2",
        "~=3.1,>=4a1,<4",
    ] {
        assert!(
            is_bare_major_spec(spec),
            "an empty or unproved predicate range must fail closed: `{spec}`"
        );
    }
    for spec in ["3.1.*", "~=3.1", "3.1.*,>=3.1.5,<3.2"] {
        assert!(
            !is_bare_major_spec(spec),
            "a witnessed minor predicate range should remain accepted: `{spec}`"
        );
    }
}

#[test]
fn bare_major_detection_merges_disjunctive_range_envelopes() {
    for spec in [
        "<=3.1|>3.1",
        "<3.1|>=3.1",
        "<3.1|>3.1",
        ">=3,<3.1|>=3.1,<4",
        ">=3,<3.1|>3.2,<4",
        "==3|>=4,<5",
    ] {
        assert!(
            is_bare_major_spec(spec),
            "minor-looking internal OR boundaries must not hide an unconstrained union: `{spec}`"
        );
    }
    assert!(
        !is_bare_major_spec("==3.1|==3.2"),
        "disjoint concrete alternatives retain their load-bearing boundaries"
    );
}

#[test]
fn output_abi_invariant_accepts_minor_pin_and_rejects_anchor_override() {
    let workspace = BTreeMap::from([
        ("python".to_string(), BTreeSet::from(["3.11".to_string()])),
        (
            "cuda-version".to_string(),
            BTreeSet::from(["12.8".to_string()]),
        ),
    ]);
    let emitted = vec![
        ("python".to_string(), "3.11.*".to_string()),
        ("cuda-version".to_string(), "==12.8".to_string()),
        ("packaging".to_string(), "*".to_string()),
    ];
    assert!(
        check_output_abi_invariants(
            &emitted,
            &[],
            &workspace,
            &BTreeMap::new(),
            &BTreeMap::new(),
        )
        .is_empty()
    );

    for spec in ["*", ">=1,<2,>0.1", ">=1.0,<2"] {
        let overrides = BTreeMap::from([("numpy".to_string(), spec.to_string())]);
        let violations =
            check_output_abi_invariants(&[], &[], &workspace, &overrides, &BTreeMap::new());
        assert_eq!(violations.len(), 1, "{spec}: {violations:?}");
        assert!(violations[0].contains("retread-overrides[numpy]"));
    }
}

#[test]
fn output_abi_invariant_checks_embedded_wheel_metadata_and_aliases() {
    let workspace = BTreeMap::from([("numpy".to_string(), BTreeSet::from(["1.26.4".to_string()]))]);
    let embedded = vec![("consumer-wheel".to_string(), "numpy>=1".to_string())];
    let violations = check_output_abi_invariants(
        &[],
        &embedded,
        &workspace,
        &BTreeMap::new(),
        &AbiAliasGraph::new(),
    );
    assert_eq!(violations.len(), 1, "{violations:?}");
    assert!(violations[0].contains("wheel `consumer-wheel` embeds"));
    assert!(violations[0].contains("bare-major"));

    let safe = vec![("consumer-wheel".to_string(), "numpy>=1.26,<2".to_string())];
    assert!(
        check_output_abi_invariants(
            &[],
            &safe,
            &workspace,
            &BTreeMap::new(),
            &AbiAliasGraph::new(),
        )
        .is_empty()
    );

    let mut aliases = AbiAliasGraph::new();
    add_abi_alias_edge(&mut aliases, "array-provider", "shared-runtime");
    add_abi_alias_edge(&mut aliases, "numpy", "shared-runtime");
    let hidden_alias = vec![(
        "consumer-wheel".to_string(),
        "array-provider>=1".to_string(),
    )];
    let violations =
        check_output_abi_invariants(&[], &hidden_alias, &workspace, &BTreeMap::new(), &aliases);
    assert_eq!(violations.len(), 1, "{violations:?}");
    assert!(violations[0].contains("array-provider >=1"));
}

#[test]
fn emitted_concrete_anchor_covers_bare_source_wheel_requirement() {
    let violations = check_output_abi_invariants(
        &[("numpy".to_string(), ">=2.3,<2.4".to_string())],
        &[("openmesh".to_string(), "numpy".to_string())],
        &BTreeMap::new(),
        &BTreeMap::new(),
        &AbiAliasGraph::new(),
    );
    assert!(violations.is_empty(), "{violations:?}");
}

#[test]
fn output_abi_invariant_rejects_pep440_local_exclusion_contradictions() {
    for requirement in ["numpy==3.1+cuda,!=3.1", "numpy==1!3.1+cuda,!=1!3.1"] {
        let violations = check_output_abi_invariants(
            &[],
            &[("consumer-wheel".to_string(), requirement.to_string())],
            &BTreeMap::new(),
            &BTreeMap::new(),
            &AbiAliasGraph::new(),
        );
        assert_eq!(violations.len(), 1, "{requirement}: {violations:?}");
        assert!(
            violations[0].contains("unsatisfiable under PEP 440"),
            "{requirement}: {violations:?}"
        );
    }
}

#[test]
fn cross_output_siblings_appear_as_run_deps() {
    // Contract: when a pack emits multiple outputs (isaacsim,
    // isaaclab, isaaclab-arena, pytorch3d, ...), each output's
    // run_dependencies must include every sibling at exact version.
    // Then the workspace pixi.toml can declare just one name and
    // conda transitively pulls the rest; without this, every
    // [retread-wheels] entry needs its own line in the consumer's
    // pixi.toml or its conda package sits unused. Also pins that
    // PEP 440 local version identifiers (the +5043d15... in
    // pytorch3d's version) survive verbatim in the run-dep spec.
    let bundle = solo_bundle("isaacsim", vec![]);
    let siblings = vec![
        ("isaacsim".to_string(), "1.0.0".to_string()), // self -- must be skipped
        ("isaaclab".to_string(), "0.51.1".to_string()),
        ("isaaclab-arena".to_string(), "0.4.2".to_string()),
        (
            "pytorch3d".to_string(),
            "0.7.8+5043d15pt2.7.0cu128".to_string(),
        ),
    ];
    let output = produce_output(
        &bundle,
        &cfg(),
        Platform::Linux64,
        "3.11",
        &siblings,
        None,
        None,
    )
    .unwrap();

    let dep_names: Vec<String> = output
        .run_dependencies
        .depends
        .iter()
        .map(|d| d.name.clone())
        .collect();

    // Self must not appear in own run-deps.
    assert_eq!(
        dep_names.iter().filter(|n| *n == "isaacsim").count(),
        0,
        "self should not be a sibling run-dep; got {dep_names:?}",
    );
    for sib in ["isaaclab", "isaaclab-arena", "pytorch3d"] {
        assert!(
            dep_names.iter().any(|n| n == sib),
            "missing sibling run-dep {sib}; got: {dep_names:?}",
        );
    }

    // Spec for the pytorch3d sibling must preserve the +local
    // version identifier. conda's match-spec parser splits the
    // local segment into tokens (e.g. `[5043, 'd', 15, 'pt', 2]`),
    // so the original string isn't contiguous in Debug -- check
    // for the distinctive token components and the `local:` marker
    // proving the +local segment was stored at all.
    let pytorch3d_spec = output
        .run_dependencies
        .depends
        .iter()
        .find(|d| d.name == "pytorch3d")
        .map(|d| format!("{:?}", d.spec))
        .expect("pytorch3d in deps");
    assert!(
        pytorch3d_spec.contains("local:"),
        "pytorch3d sibling pin lost the +local segment entirely; got: {pytorch3d_spec}",
    );
    for token in ["5043", "pt", "cu", "128"] {
        assert!(
            pytorch3d_spec.contains(token),
            "pytorch3d sibling pin missing token `{token}` from +local segment; got: {pytorch3d_spec}",
        );
    }
}

#[test]
fn prefer_conda_match_identifies_mapped_targets() {
    // Contract: entries in the effective name map are eligible for a
    // requirement-specific conda probe. The probe result, not this name-only
    // lookup, makes the final routing decision.
    let name_map = name_map(&[
        ("torch", "pytorch"),
        ("numpy", "numpy"),
        ("opencv-python-headless", "py-opencv"),
    ]);

    // Parselmouth-known => probe the mapped conda target.
    assert!(prefer_conda_match(&PypiKey::from_pypi("torch"), &name_map));
    assert!(prefer_conda_match(&PypiKey::from_pypi("numpy"), &name_map));
    // FALLBACK_PYPI_TO_CONDA entry survives the same way.
    assert!(prefer_conda_match(
        &PypiKey::from_pypi("opencv-python-headless"),
        &name_map
    ));

    // Unknown to parselmouth => no mapped conda target from this lookup.
    assert!(!prefer_conda_match(&PypiKey::from_pypi("qdldl"), &name_map));
    assert!(!prefer_conda_match(
        &PypiKey::from_pypi("asteval"),
        &name_map
    ));
    assert!(!prefer_conda_match(
        &PypiKey::from_pypi("aiodns"),
        &name_map
    ));
}

#[test]
fn source_wheel_auto_bundle_empty_conda_candidates_stays_on_pypi() {
    let bare = uv_pep508::uv_pep440::VersionSpecifiers::empty();
    assert_eq!(conda_probe_spec(&bare), "*");

    let empty = crate::probe::ProbeResult {
        package: "zmq".into(),
        spec: "*".into(),
        channels_consulted: vec!["https://conda.example/noarch".into()],
        satisfiable: Some(false),
        matching_candidates: 0,
    };
    assert!(
        !validated_conda_route(&empty),
        "an empty requirement-specific candidate set must take the auto-bundle PyPI path"
    );

    let indecisive = crate::probe::ProbeResult {
        satisfiable: None,
        channels_consulted: Vec::new(),
        ..empty.clone()
    };
    assert!(
        !validated_conda_route(&indecisive),
        "routing without a validated candidate must fail closed to PyPI"
    );

    let inconsistent = crate::probe::ProbeResult {
        satisfiable: Some(true),
        ..empty.clone()
    };
    assert!(
        !validated_conda_route(&inconsistent),
        "a true verdict without a candidate must not authorize conda routing"
    );

    let matched = crate::probe::ProbeResult {
        satisfiable: Some(true),
        matching_candidates: 1,
        ..empty
    };
    assert!(validated_conda_route(&matched));
}

#[test]
fn auto_bundle_sdist_fetch_preserves_replay_provenance() {
    let built_url = url::Url::from_file_path("/cache/zmq-0.0.0-py3-none-any.whl").unwrap();
    let sdist_url =
        url::Url::parse("https://files.pythonhosted.org/packages/zmq-0.0.0.tar.gz#sha256=abc123")
            .unwrap();
    let (upstream, source, metadata_provenance) = bfs_fetch_provenance(
        &built_url,
        Some(SdistProv {
            index: "https://pypi.org/simple/".into(),
            name: "zmq".into(),
            version: "0.0.0".into(),
            sdist_url: sdist_url.clone(),
        }),
    );

    assert!(
        upstream.is_none(),
        "machine-local built wheel URL must not become replay provenance"
    );
    let source = source.expect("sdist replay source");
    assert_eq!(metadata_provenance, Provenance::SourceBuiltRelaxed);
    assert_eq!(source.name, "zmq");
    assert_eq!(source.version, "0.0.0");
    assert_eq!(source.sdist_url, sdist_url.to_string());

    let wheel_url = url::Url::parse("https://files.pythonhosted.org/zmq-1.0.whl").unwrap();
    let (upstream, source, metadata_provenance) = bfs_fetch_provenance(&wheel_url, None);
    assert_eq!(upstream.as_ref(), Some(&wheel_url));
    assert!(source.is_none());
    assert_eq!(metadata_provenance, Provenance::IndexWheelMetadata);
}

#[test]
fn wheel_entry_origin_marks_every_source_build_advisory() {
    let path = WheelEntry {
        path: Some("../isaaclab".to_string()),
        ..WheelEntry::default()
    };
    let git = WheelEntry {
        git: Some("https://example.invalid/project.git".to_string()),
        rev: Some("deadbeef".to_string()),
        ..WheelEntry::default()
    };
    let named_git = WheelEntry {
        from: Some("upstream".to_string()),
        ..WheelEntry::default()
    };
    for entry in [&path, &git, &named_git] {
        let provenance = wheel_entry_metadata_provenance(entry);
        assert_eq!(provenance, Provenance::SourceBuiltRelaxed);
        assert_eq!(
            crate::constraint::authority(&provenance),
            crate::constraint::Authority::Advisory,
        );
    }

    let direct_url = WheelEntry {
        url: Some("https://example.invalid/project.whl".parse().unwrap()),
        ..WheelEntry::default()
    };
    let index = WheelEntry {
        version: Some("1.2.3".to_string()),
        ..WheelEntry::default()
    };
    for entry in [&direct_url, &index] {
        let provenance = wheel_entry_metadata_provenance(entry);
        assert_eq!(provenance, Provenance::IndexWheelMetadata);
        assert_eq!(
            crate::constraint::authority(&provenance),
            crate::constraint::Authority::Authoritative,
        );
    }
}

#[test]
fn path_built_wheel_floor_is_advisory_during_finalization() {
    let path = WheelEntry {
        path: Some("../isaaclab".to_string()),
        ..WheelEntry::default()
    };
    let path_provenance = wheel_entry_metadata_provenance(&path);
    let constraints = vec![
        crate::constraint::Constraint {
            specifiers: ">=0.49.1,<0.50".parse().unwrap(),
            provenance: path_provenance,
            source: "path-built wheel `isaaclab` Requires-Dist".to_string(),
            origin_id: crate::constraint::ConstraintOriginId::from_parts(
                "handler-test-wheel-requirement",
                ["isaaclab", "starlette", ">=0.49.1,<0.50"],
            ),
        },
        crate::constraint::Constraint {
            specifiers: ">=0.40,<0.46".parse().unwrap(),
            provenance: Provenance::IndexWheelMetadata,
            source: "index wheel `fastapi` Requires-Dist".to_string(),
            origin_id: crate::constraint::ConstraintOriginId::from_parts(
                "handler-test-wheel-requirement",
                ["fastapi", "starlette", ">=0.40,<0.46"],
            ),
        },
    ];
    let finalized = crate::constraint::finalize(&PypiKey::from_pypi("starlette"), &constraints)
        .expect("a path-built wheel floor must yield to an authoritative cap");
    assert!(finalized.contains(&"0.45.3".parse().unwrap()));
    assert!(!finalized.contains(&"0.49.1".parse().unwrap()));
}

#[test]
fn pep508_extra_dep_accepts_range_specifier() {
    // Regression: isaacsim's METADATA has
    //   Requires-Dist: isaacsim-extscache-kit>=5 ; extra == "extscache"
    // We used to require an exact `==X.Y.Z` pin on extras-gated
    // requirements and bail otherwise. NVIDIA publishes plenty of
    // metapackages that use `>=` to gate to a major series, so range
    // resolution is a hard requirement -- we now resolve the highest
    // matching version off the index instead of refusing.
    let dep = pep508_extra_dep(
        "isaacsim-extscache-kit>=5 ; extra == \"extscache\"",
        "extscache",
    )
    .expect("pep508 parse")
    .expect("extras-gated dep, got None");
    assert_eq!(dep.name, "isaacsim-extscache-kit");
    let specs = match &dep.source {
        ExtraDepSource::Pypi(s) => s,
        other => panic!("expected PyPI source, got {other:?}"),
    };
    assert!(
        specs.to_string().contains(">=5"),
        "specifiers must preserve `>=5`, got: {specs}",
    );
}

/// v0.12.0+: URL Requires-Dist (PEP 508 `pkg @ <url>`) is parsed
/// into `ExtraDepSource::Git` / `Url` instead of bailing. Mirrors
/// IsaacLab's `rl_games` extra:
///   Requires-Dist: rl-games @ git+https://.../rl_games.git@python3.11 ; extra == "rl_games"
#[test]
fn pep508_extra_dep_handles_git_url() {
    let dep = pep508_extra_dep(
            "rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11 ; extra == \"rl_games\"",
            "rl_games",
        )
        .expect("pep508 parse")
        .expect("extras-gated dep, got None");
    assert_eq!(dep.name, "rl-games");
    match dep.source {
        ExtraDepSource::Git {
            url,
            rev,
            subdirectory,
        } => {
            assert_eq!(url, "https://github.com/isaac-sim/rl_games.git");
            assert_eq!(rev.as_deref(), Some("python3.11"));
            assert_eq!(subdirectory, None, "no #subdirectory= in this URL");
        }
        other => panic!("expected Git source, got {other:?}"),
    }
}

#[test]
fn pep508_extra_dep_handles_direct_url() {
    let dep = pep508_extra_dep(
        "foo @ https://example.com/foo-1.2.3-py3-none-any.whl ; extra == \"foo\"",
        "foo",
    )
    .expect("pep508 parse")
    .expect("extras-gated dep, got None");
    assert_eq!(dep.name, "foo");
    match dep.source {
        ExtraDepSource::Url(u) => {
            assert_eq!(u.as_str(), "https://example.com/foo-1.2.3-py3-none-any.whl");
        }
        other => panic!("expected Url source, got {other:?}"),
    }
}

/// Regression: extras-gated bare names (no version, no URL) are
/// legal PEP 508 -- e.g. IsaacLab's
///   Requires-Dist: tqdm; extra == "sb3"
///   Requires-Dist: gym; extra == "rl_games"
///   Requires-Dist: rich; extra == "sb3"
/// Previously made retread bail with "no version or URL". Now
/// resolves as PyPI with empty specifiers -> latest matching the
/// target python.
#[test]
fn pep508_extra_dep_handles_bare_name() {
    let dep = pep508_extra_dep("tqdm; extra == \"sb3\"", "sb3")
        .expect("pep508 parse")
        .expect("extras-gated bare-name dep, got None");
    assert_eq!(dep.name, "tqdm");
    match dep.source {
        ExtraDepSource::Pypi(specs) => {
            assert_eq!(
                specs.to_string(),
                "",
                "bare name -> empty specifiers, got: {specs}"
            );
        }
        other => panic!("expected PyPI source, got {other:?}"),
    }
}

/// git URL without `@<rev>` -> Some(url), None for rev. Pipeline
/// will default to HEAD when synthesizing the WheelEntry.
#[test]
fn extra_dep_source_from_url_git_without_rev() {
    let url: url::Url = "git+https://github.com/foo/bar.git".parse().unwrap();
    match extra_dep_source_from_url(&url).expect("parse") {
        ExtraDepSource::Git {
            url,
            rev,
            subdirectory,
        } => {
            assert_eq!(url, "https://github.com/foo/bar.git");
            assert_eq!(rev, None);
            assert_eq!(subdirectory, None, "no #subdirectory= in this URL");
        }
        other => panic!("expected Git, got {other:?}"),
    }
}

/// A-0 regression: `#subdirectory=<path>` fragment must be stripped from
/// the rev string and surfaced as a separate `subdirectory` field.
///
/// Without this fix, `rfind('@')` returns the `@` before `<rev>` but
/// includes `#subdirectory=src/foo` as part of `rev`, corrupting it to
/// `"ce11136#subdirectory=src/foo"`. The checkout cache key (sha256 of
/// url+"\0"+rev) then differs between the producer and a replay that
/// correctly splits the fragment, and the git checkout itself fails
/// because git doesn't recognize the junk rev.
///
/// This test is RED on the pre-fix tree (rev contains the fragment) and
/// GREEN after the fix (rev and subdirectory are split correctly).
#[test]
fn extra_dep_source_from_url_git_subdirectory_fragment_is_stripped() {
    // Typical format: git+https://host/repo@<rev>#subdirectory=<subdir>
    let url: url::Url =
        "git+https://github.com/newton-sim/newton.git@ce11136#subdirectory=src/newton"
            .parse()
            .unwrap();
    match extra_dep_source_from_url(&url).expect("parse") {
        ExtraDepSource::Git {
            url,
            rev,
            subdirectory,
        } => {
            assert_eq!(url, "https://github.com/newton-sim/newton.git");
            assert_eq!(
                rev.as_deref(),
                Some("ce11136"),
                "rev must NOT contain #subdirectory= fragment"
            );
            assert_eq!(
                subdirectory.as_deref(),
                Some("src/newton"),
                "subdirectory must be parsed from the fragment"
            );
        }
        other => panic!("expected Git, got {other:?}"),
    }
}

/// SHA-only rev with no #subdirectory= remains unaffected by the fix.
#[test]
fn extra_dep_source_from_url_git_sha_rev_no_fragment() {
    let url: url::Url =
        "git+https://github.com/newton-sim/newton.git@8de7e456deadbeef1234567890abcdef12345678"
            .parse()
            .unwrap();
    match extra_dep_source_from_url(&url).expect("parse") {
        ExtraDepSource::Git {
            url,
            rev,
            subdirectory,
        } => {
            assert_eq!(url, "https://github.com/newton-sim/newton.git");
            assert_eq!(
                rev.as_deref(),
                Some("8de7e456deadbeef1234567890abcdef12345678")
            );
            assert_eq!(subdirectory, None);
        }
        other => panic!("expected Git, got {other:?}"),
    }
}

// -----------------------------------------------------------------
// P1: has_environment + absent-env filtering.
// Manifest with envs {a}; filtering [a, b] keeps only [a].
// -----------------------------------------------------------------

#[test]
fn has_environment_returns_true_for_declared_env() {
    use crate::workspace::WorkspaceManifest;
    let manifest: toml::Value = toml::from_str(
        r#"
[environments]
a = { features = ["feat-a"] }

[feature.feat-a.dependencies]
numpy = ">=1"
"#,
    )
    .unwrap();
    let ws = WorkspaceManifest::from_toml(&manifest);
    assert!(ws.has_environment("a"), "env 'a' is declared");
    assert!(!ws.has_environment("b"), "env 'b' is not declared");
    assert!(!ws.has_environment("feat-a"), "features are not envs");
}

#[test]
fn absent_env_filter_keeps_declared_drops_absent() {
    use crate::workspace::WorkspaceManifest;
    let manifest: toml::Value = toml::from_str(
        r#"
[environments]
a = { features = [] }
"#,
    )
    .unwrap();
    let ws = WorkspaceManifest::from_toml(&manifest);

    // Simulate the retain logic from conda_outputs.
    let mut env_names = vec!["a".to_string(), "b".to_string(), "__default__".to_string()];
    env_names.retain(|n| {
        if n == "__default__" {
            return true;
        }
        ws.has_environment(n)
    });

    // "a" kept, "b" dropped, "__default__" always kept.
    assert_eq!(env_names, vec!["a", "__default__"]);
}

// -----------------------------------------------------------------
// P1: cache key changes when manifest mtime changes.
// We exercise `conda_outputs_cache_key` directly with two different
// synthetic `Option<SystemTime>` values and assert the keys differ.
// -----------------------------------------------------------------

#[test]
fn cache_key_changes_when_manifest_mtime_changes() {
    use pixi_build_types::procedures::conda_outputs::CondaOutputsParams;
    use rattler_conda_types::Platform;

    // Build minimal CondaOutputsParams with identical platform/channels/variant.
    let make_params = || CondaOutputsParams {
        host_platform: Platform::Linux64,
        build_platform: Platform::Linux64,
        channels: vec![],
        variant_configuration: None,
        variant_files: None,
        work_directory: std::path::PathBuf::new(),
    };

    // Two different synthetic mtimes.
    let t0 = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000);
    let t1 = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000_001);

    let key0 = conda_outputs_cache_key(&make_params(), Some(t0), "none");
    let key1 = conda_outputs_cache_key(&make_params(), Some(t1), "none");
    let key_none = conda_outputs_cache_key(&make_params(), None, "none");

    // Different mtimes -> different keys.
    assert_ne!(key0, key1, "distinct mtimes must produce distinct keys");
    // None sentinel differs from a real mtime.
    assert_ne!(
        key0, key_none,
        "None mtime must not collide with a real mtime"
    );
    // Identical params + identical mtime -> same key (deterministic).
    let key0_dup = conda_outputs_cache_key(&make_params(), Some(t0), "none");
    assert_eq!(
        key0, key0_dup,
        "same params+mtime must produce identical key"
    );

    // Run-12 regression: an auto-overrides ledger change (fix #22 pack
    // repairs write ONLY the ledger, never a manifest, so the mtime
    // component never moves) must bust the key too -- otherwise a repair
    // iteration's fresh backend cache-hits the STALE pack render.
    let key_ledger = conda_outputs_cache_key(&make_params(), Some(t0), "abcdef0123456789");
    assert_ne!(
        key0, key_ledger,
        "distinct auto-overrides ledger fingerprints must produce distinct keys"
    );

    // Run-31 regression: the backend build identity (crate version + git
    // hash from build.rs) must be IN the key, so a binary built from a
    // different commit can never reuse this build's cached pack renders
    // (run 30 served the bounded-range binary pre-fix exact-pin renders).
    let ident = backend_build_identity();
    assert!(
        !ident.is_empty() && ident.contains('+'),
        "build identity must be `version+githash`, got `{ident}`"
    );
    assert!(
        ident.starts_with(env!("CARGO_PKG_VERSION")),
        "build identity must start with the crate version, got `{ident}`"
    );
    assert!(
        key0.ends_with(ident),
        "cache key must fold in the backend build identity: key=`{key0}` ident=`{ident}`"
    );
}

#[test]
fn cache_key_partitions_same_subdir_rich_target_contracts() {
    use pixi_build_types::procedures::conda_outputs::CondaOutputsParams;
    use rattler_conda_types::Platform;

    let params = CondaOutputsParams {
        host_platform: Platform::Linux64,
        build_platform: Platform::Linux64,
        channels: vec![],
        variant_configuration: None,
        variant_files: None,
        work_directory: std::path::PathBuf::new(),
    };
    let contract = |glibc: &str| WorkspaceTargetContract {
        subdir: "linux-64".to_string(),
        declared_virtual_packages: BTreeMap::from([
            ("cuda".to_string(), "12".to_string()),
            ("glibc".to_string(), glibc.to_string()),
            ("linux".to_string(), "4.18".to_string()),
        ]),
        detected_virtual_packages: BTreeMap::new(),
    };
    let p1 = ResolutionTarget::try_for_contract("0.0", contract("2.28")).unwrap();
    let p3 = ResolutionTarget::try_for_contract("0.0", contract("2.35")).unwrap();

    let p1_key = conda_outputs_cache_key_for_target(&params, None, "none", &p1, None, "");
    let p3_key = conda_outputs_cache_key_for_target(&params, None, "none", &p3, None, "");
    assert_ne!(
        p1_key, p3_key,
        "p1 and p3 share linux-64 but must never share conda/outputs metadata"
    );

    let output_for = |target: &ResolutionTarget| {
        assemble_conda_output(
            "demo-pack",
            "1.0.0",
            "3.11",
            false,
            true,
            Vec::new(),
            std::collections::HashSet::new(),
            Platform::Linux64,
            0,
            Some(&target.resolution_identity()),
            false,
            &[],
        )
        .unwrap()
    };
    assert_ne!(
        output_for(&p1).metadata.build,
        output_for(&p3).metadata.build,
        "non-courier rich targets also need distinct conda build identities"
    );
}

#[test]
fn cache_key_partitions_exact_workspace_consumer_scope() {
    use pixi_build_types::procedures::conda_outputs::CondaOutputsParams;
    use rattler_conda_types::Platform;

    let params = CondaOutputsParams {
        host_platform: Platform::Linux64,
        build_platform: Platform::Linux64,
        channels: vec![],
        variant_configuration: None,
        variant_files: None,
        work_directory: std::path::PathBuf::new(),
    };
    let contract = WorkspaceTargetContract {
        subdir: "linux-64".to_string(),
        declared_virtual_packages: BTreeMap::from([("glibc".to_string(), "2.28".to_string())]),
        detected_virtual_packages: BTreeMap::new(),
    };
    let target = ResolutionTarget::try_for_contract("0.0", contract.clone()).unwrap();
    let old = ResolvedWorkspaceTarget {
        contract: contract.clone(),
        profiles: vec!["p1".to_string()],
        environments: vec!["old".to_string()],
    };
    let alias = ResolvedWorkspaceTarget {
        contract: contract.clone(),
        profiles: vec!["p1-alias".to_string()],
        environments: vec!["alias".to_string()],
    };
    let aggregate = ResolvedWorkspaceTarget {
        contract: contract.clone(),
        profiles: vec!["p1-alias".to_string(), "p1".to_string()],
        environments: vec!["alias".to_string(), "old".to_string()],
    };
    let permuted_with_duplicates = ResolvedWorkspaceTarget {
        contract,
        profiles: vec!["p1".to_string(), "p1-alias".to_string(), "p1".to_string()],
        environments: vec!["old".to_string(), "alias".to_string(), "old".to_string()],
    };
    let key = |scope| conda_outputs_cache_key_for_target(&params, None, "none", &target, scope, "");

    assert_ne!(key(Some(&old)), key(Some(&alias)));
    assert_ne!(key(Some(&old)), key(Some(&aggregate)));
    assert_eq!(
        key(Some(&aggregate)),
        key(Some(&permuted_with_duplicates)),
        "logical consumer scope identity must ignore ordering and duplicates"
    );
    assert_eq!(key(None), key(None));

    let old_target = target.clone().with_workspace_scope(old).unwrap();
    let alias_target = target.clone().with_workspace_scope(alias).unwrap();
    assert_ne!(
        old_target.resolution_identity(),
        alias_target.resolution_identity(),
        "exact consumer scope must partition resolution, lock, and build identity"
    );
    let output_for = |target: &ResolutionTarget| {
        assemble_conda_output(
            "demo-pack",
            "1.0.0",
            "3.11",
            false,
            true,
            Vec::new(),
            std::collections::HashSet::new(),
            Platform::Linux64,
            0,
            Some(&target.resolution_identity()),
            false,
            &[],
        )
        .unwrap()
    };
    assert_ne!(
        output_for(&old_target).metadata.build,
        output_for(&alias_target).metadata.build,
        "same-contract environments may emit different dependency metadata"
    );
}

#[test]
fn dynamic_courier_cold_and_replay_use_staged_content_build_identity() {
    let target = ResolutionTarget::for_subdir("3.11", "linux-64");
    let staged_hash = "0123456789abcdef";
    let expected = courier_build_string_for_target(&target, staged_hash, 4, false);

    let cold = resolved_courier_build(None, &target, staged_hash, 4, false);
    let replay = resolved_courier_build(None, &target, staged_hash, 4, false);
    assert_eq!(cold, expected);
    assert_eq!(replay, expected);
    assert_ne!(
        cold, "py311_4",
        "dynamic builds must not use the legacy namespace"
    );
    assert_eq!(
        resolved_courier_build(Some("advertised"), &target, staged_hash, 4, false),
        "advertised",
        "an explicit Pixi build identity remains authoritative"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn artifact_lookup_requires_exact_build_identity() {
    let dir = unique_test_dir("exact-artifact-build");
    std::fs::create_dir_all(&dir).unwrap();
    let p1 = dir.join("demo-1.0.0-py311_h1111111111_0.conda");
    let p3 = dir.join("demo-1.0.0-py311_h3333333333_0.conda");
    std::fs::write(&p1, b"p1").unwrap();
    std::fs::write(&p3, b"p3").unwrap();

    let found = find_conda_artifact(&dir, "demo", "1.0.0", "py311_h3333333333_0")
        .await
        .unwrap();
    assert_eq!(found, p3, "must not return the first same-version sibling");
    let error = find_conda_artifact(&dir, "demo", "1.0.0", "py311_h2222222222_0")
        .await
        .unwrap_err();
    assert!(error.to_string().contains("py311_h2222222222_0"));

    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn noarch_build_uses_host_prefix_as_resolution_platform() {
    assert_eq!(
        resolution_subdir_for_build(
            Platform::NoArch,
            Some(Platform::Linux64),
            None,
            Some("py311_h0123456789_0"),
        )
        .unwrap(),
        Platform::Linux64,
    );
    assert!(
        resolution_subdir_for_build(Platform::NoArch, None, None, Some("py311_h0123456789_0"),)
            .is_err(),
        "a rich noarch output must not fall back to a noarch resolution target"
    );
    assert_eq!(
        resolution_subdir_for_build(Platform::NoArch, None, None, Some("py311_0")).unwrap(),
        Platform::NoArch,
        "legacy noarch identity retains the old fallback"
    );
    assert_eq!(
        resolution_subdir_for_build(Platform::LinuxAarch64, Some(Platform::Linux64), None, None,)
            .unwrap(),
        Platform::LinuxAarch64,
        "platform-specific artifact identity remains authoritative"
    );

    let linux = ResolutionTarget::for_subdir("3.11", "linux-64");
    assert!(validate_resolution_artifact_subdir(&linux, Platform::NoArch).is_ok());
    assert!(validate_resolution_artifact_subdir(&linux, Platform::LinuxAarch64).is_err());
}

#[cfg(unix)]
#[tokio::test]
async fn pure_noarch_prepared_handoff_retains_rich_target_contract() {
    let contract = |glibc: &str| WorkspaceTargetContract {
        subdir: "linux-64".to_string(),
        declared_virtual_packages: BTreeMap::from([("glibc".to_string(), glibc.to_string())]),
        detected_virtual_packages: BTreeMap::new(),
    };
    let p1 = ResolutionTarget::try_for_contract("3.11", contract("2.28")).unwrap();
    let p3 = ResolutionTarget::try_for_contract("3.11", contract("2.35")).unwrap();
    let mut base_bundle = solo_bundle("pure-target-pack", vec![]);
    base_bundle.primary.metadata.is_pure_python = true;
    base_bundle.primary.metadata.filename = "pure_target_pack-1.0.0-py3-none-any.whl".to_string();
    let config = cfg();
    let emission = DiscoveredEmission {
        output_name: "pure-target-pack".to_string(),
        channels: Vec::new(),
        transitive_overrides: BTreeMap::new(),
        envs: vec!["old".to_string()],
    };
    let advertised = produce_output(
        &base_bundle,
        &config,
        Platform::Linux64,
        "3.11",
        &[],
        Some(&p1.resolution_identity()),
        None,
    )
    .unwrap();
    assert_eq!(advertised.metadata.subdir, Platform::NoArch);
    let work_dir = unique_test_dir("rich-noarch-prepared");
    let plan = Arc::new(ResolvedTargetPlan {
        local_wheel_stamps: capture_local_wheel_stamps(std::slice::from_ref(&base_bundle)),
        materialized: vec![base_bundle],
        base_config: config.clone(),
        restore_relaxations: vec![],
        declared_config: config,
        target: p1.clone(),
        work_directory: work_dir.clone(),
        workspace_manifest_mtime: None,
        auto_overrides_fingerprint: "none".to_string(),
    });
    let prepared = PreparedBuild {
        locator_id: 0,
        plan,
        bundle_index: 0,
        emission,
        advertised: PreparedOutputIdentity::from_metadata(&advertised.metadata),
        advertised_run_dependencies: advertised.run_dependencies.clone(),
        relaxations: None,
        incremental_version_override: None,
    };
    let request = pixi_build_types::procedures::conda_build_v1::CondaBuildV1Output {
        name: advertised.metadata.name.clone(),
        version: Some(advertised.metadata.version.clone()),
        build: Some(advertised.metadata.build.clone()),
        subdir: advertised.metadata.subdir,
        variant: advertised.metadata.variant.clone(),
    };
    let handler = Handler::default();
    {
        let mut state = handler.state.write().await;
        state.generation = 19;
    }
    let transaction = handler.begin_prepared_transaction(19).await.unwrap();
    assert!(
        handler
            .publish_prepared_builds(19, transaction, "rich-noarch".to_string(), vec![prepared])
            .await
    );
    assert!(
        handler
            .lookup_prepared_build_for_target(19, &work_dir, None, Some("3.11"), &p3, &request,)
            .await
            .is_none(),
        "same-subdir p3 contract must not consume p1's noarch plan"
    );
    assert!(
        handler
            .lookup_prepared_build_for_target(19, &work_dir, None, Some("3.11"), &p1, &request,)
            .await
            .is_some(),
        "noarch artifact identity must retain the p1 resolution contract"
    );
}

#[cfg(unix)]
#[test]
fn handler_source_resolution_does_not_inherit_sibling_profile_glibc() {
    use rattler_conda_types::Platform;

    let workspace = unique_test_dir("handler-rich-target");
    let p1_source = workspace.join("p1-pack");
    let p3_source = workspace.join("p3-pack");
    std::fs::create_dir_all(&p1_source).unwrap();
    std::fs::create_dir_all(&p3_source).unwrap();
    std::fs::write(
        workspace.join("pixi.toml"),
        r#"
[workspace]
platforms = [
  { name = "p1", platform = "linux-64", cuda = "12", glibc = "2.28", linux = "4.18" },
  { name = "p3", platform = "linux-64", cuda = "12", glibc = "2.35", linux = "4.18" },
]
[feature.p1]
platforms = ["p1"]
[feature.p1.dependencies]
p1-pack = { path = "./p1-pack" }
[feature.p3]
platforms = ["p3"]
[feature.p3.dependencies]
p3-pack = { path = "./p3-pack" }
[environments]
old = { features = ["p1"], no-default-feature = true }
new = { features = ["p3"], no-default-feature = true }
"#,
    )
    .unwrap();

    let p1 = resolve_workspace_target_for_source(Some(&workspace), &p1_source, "linux-64", None)
        .unwrap()
        .unwrap();
    let p3 = resolve_workspace_target_for_source(Some(&workspace), &p3_source, "linux-64", None)
        .unwrap()
        .unwrap();
    assert_eq!(p1.contract.effective_glibc(), Some((2, 28)));
    assert_eq!(p3.contract.effective_glibc(), Some((2, 35)));

    let p1_target =
        wheel_target_for_contract(Platform::Linux64, "3.11", Some(&p1.contract)).unwrap();
    let p3_target =
        wheel_target_for_contract(Platform::Linux64, "3.11", Some(&p3.contract)).unwrap();
    assert_eq!(p1_target.declared_glibc(), Some((2, 28)));
    assert_eq!(p3_target.declared_glibc(), Some((2, 35)));
    assert_ne!(
        p1_target.resolution_identity(),
        p3_target.resolution_identity()
    );

    std::fs::remove_dir_all(workspace).ok();
}

#[test]
fn auto_overrides_fingerprint_tracks_ledger_content_not_mtime() {
    let dir = std::env::temp_dir().join(format!(
        "retread-auto-fp-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(dir.join(".retread")).unwrap();

    // No workspace dir / no ledger -> "none" sentinel.
    assert_eq!(auto_overrides_fingerprint(None), "none");
    assert_eq!(auto_overrides_fingerprint(Some(&dir)), "none");

    let ledger = dir.join(".retread").join("auto-overrides.json");
    std::fs::write(&ledger, br#"{"packs":{}}"#).unwrap();
    let fp_empty = auto_overrides_fingerprint(Some(&dir));
    assert_ne!(fp_empty, "none");

    std::fs::write(
        &ledger,
        br#"{"packs":{"pypi-packs/isaaclab-2.3x-pack":{"setuptools":{"spec":">=68,<81"}}}}"#,
    )
    .unwrap();
    let fp_with_override = auto_overrides_fingerprint(Some(&dir));
    assert_ne!(
        fp_empty, fp_with_override,
        "a ledger write must change the fingerprint"
    );

    // Rollback (restoring the previous BYTES) must restore the previous
    // fingerprint -- this is why the fingerprint hashes content, not mtime.
    std::fs::write(&ledger, br#"{"packs":{}}"#).unwrap();
    assert_eq!(auto_overrides_fingerprint(Some(&dir)), fp_empty);

    std::fs::remove_dir_all(&dir).ok();
}

// -----------------------------------------------------------------
// v2.11.0: cross-process conda/outputs disk cache. Two DIFFERENT
// "processes" (simulated here by two independent calls, since the
// on-disk cache is exactly what lets a fresh process avoid a cold
// recompute) round-trip a CondaOutputsResult through
// write_conda_outputs_disk_cache / read_conda_outputs_disk_cache.
// -----------------------------------------------------------------

#[tokio::test]
async fn conda_outputs_cache_and_prepared_handoff_round_trip() {
    use pixi_build_types::procedures::conda_outputs::CondaOutputsResult;

    let cache_dir = std::env::temp_dir().join(format!(
        "retread-conda-outputs-disk-cache-test-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&cache_dir).unwrap();
    let cache_key = "linux-64|linux-64|https://conda.anaconda.org/conda-forge|None|123456";
    let source_dir = std::path::Path::new("/workspace/pypi-packs/some-pack");
    let path = conda_outputs_disk_cache_path(&cache_dir, cache_key, source_dir);

    // Nothing written yet -- a fresh process must fall back to cold compute.
    assert!(read_conda_outputs_disk_cache(&path).await.is_none());

    let result = CondaOutputsResult {
        outputs: Default::default(),
        input_globs: Default::default(),
    };
    write_conda_outputs_disk_cache(&path, &result).await;

    let loaded = read_conda_outputs_disk_cache(&path)
        .await
        .expect("a second process must be able to load what the first one wrote");
    assert_eq!(loaded.outputs.len(), result.outputs.len());

    // No leftover .tmp file from the atomic write-then-rename.
    let leftovers: Vec<_> = std::fs::read_dir(&cache_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
        .collect();
    assert!(
        leftovers.is_empty(),
        "temp file must be renamed away, not left behind"
    );

    // A real outputs computation may hand its exact typed resolution to the
    // matching build request. Optional version/build fields narrow only when
    // present; duplicates are deliberately ambiguous and fall back.
    let work_dir = cache_dir.join("prepared-work");
    let mut declared_config = cfg();
    declared_config.courier = true;
    let base_bundle = solo_bundle("prepared-pack", vec![]);
    let emission = DiscoveredEmission {
        output_name: "prepared-pack-env".to_string(),
        channels: Vec::new(),
        transitive_overrides: BTreeMap::new(),
        envs: Vec::new(),
    };
    let (advertised_bundle, advertised_config) =
        apply_emission(&base_bundle, &declared_config, &emission);
    let advertised_output = produce_output(
        &advertised_bundle,
        &advertised_config,
        Platform::Linux64,
        "3.11",
        &[],
        Some("prepared-hash"),
        None,
    )
    .unwrap();
    let plan = Arc::new(ResolvedTargetPlan {
        local_wheel_stamps: capture_local_wheel_stamps(std::slice::from_ref(&base_bundle)),
        materialized: vec![base_bundle],
        base_config: declared_config.clone(),
        restore_relaxations: vec![],
        declared_config: declared_config.clone(),
        target: crate::pypi::ResolutionTarget::for_subdir("3.11", "linux-64"),
        work_directory: work_dir.clone(),
        workspace_manifest_mtime: None,
        auto_overrides_fingerprint: "none".to_string(),
    });
    let prepared = PreparedBuild {
        locator_id: 0,
        plan,
        bundle_index: 0,
        emission,
        advertised: PreparedOutputIdentity::from_metadata(&advertised_output.metadata),
        advertised_run_dependencies: advertised_output.run_dependencies.clone(),
        relaxations: None,
        incremental_version_override: None,
    };
    let request = pixi_build_types::procedures::conda_build_v1::CondaBuildV1Output {
        name: advertised_output.metadata.name.clone(),
        version: None,
        build: None,
        subdir: advertised_output.metadata.subdir,
        variant: advertised_output.metadata.variant.clone(),
    };
    let handler = Handler::default();
    {
        let mut state = handler.state.write().await;
        state.generation = 7;
    }
    let transaction = handler.begin_prepared_transaction(7).await.unwrap();
    assert!(
        handler
            .publish_prepared_builds(
                7,
                transaction,
                "prepared-key".to_string(),
                vec![prepared.clone()],
            )
            .await
    );
    let hit = handler
        .lookup_prepared_build(7, &work_dir, None, None, &request)
        .await
        .expect("one partial-identity candidate must reuse the advertised plan");
    assert_eq!(hit.bundle.conda_name, "prepared-pack-env");
    assert!(matches!(
        hit.bundle.primary.metadata_provenance,
        Provenance::IndexWheelMetadata
    ));
    assert_eq!(
        hit.prepared.advertised.build, advertised_output.metadata.build,
        "a dynamic build request must retain the advertised content-addressed build"
    );
    handler
        .retain_prepared_for_memory_cache_hit("prepared-key", &work_dir)
        .await;
    assert!(
        handler
            .lookup_prepared_build(7, &work_dir, None, Some("3.11"), &request)
            .await
            .is_some(),
        "the same handler/cache key may retain its typed plan"
    );
    assert!(
        handler
            .lookup_prepared_build(7, Path::new("/other-work"), None, Some("3.11"), &request,)
            .await
            .is_none(),
        "work directories must not share prepared plans"
    );
    handler
        .consume_prepared_build(7, hit.transaction, hit.prepared.locator_id)
        .await;
    assert!(
        handler
            .lookup_prepared_build(7, &work_dir, None, Some("3.11"), &request)
            .await
            .is_none(),
        "a successful build must release its locator and final plan Arc"
    );

    let transaction = handler.begin_prepared_transaction(7).await.unwrap();
    let py312_bundle = solo_bundle("prepared-pack", vec![]);
    let py312_plan = Arc::new(ResolvedTargetPlan {
        local_wheel_stamps: capture_local_wheel_stamps(std::slice::from_ref(&py312_bundle)),
        materialized: vec![py312_bundle],
        base_config: declared_config.clone(),
        restore_relaxations: vec![],
        declared_config: declared_config.clone(),
        target: crate::pypi::ResolutionTarget::for_subdir("3.12", "linux-64"),
        work_directory: work_dir.clone(),
        workspace_manifest_mtime: None,
        auto_overrides_fingerprint: "none".to_string(),
    });
    let py312_prepared = PreparedBuild {
        locator_id: 1,
        plan: py312_plan,
        bundle_index: 0,
        emission: prepared.emission.clone(),
        advertised: prepared.advertised.clone(),
        advertised_run_dependencies: prepared.advertised_run_dependencies.clone(),
        relaxations: None,
        incremental_version_override: None,
    };
    handler
        .publish_prepared_builds(
            7,
            transaction,
            "ambiguous-key".to_string(),
            vec![prepared.clone(), py312_prepared],
        )
        .await;
    assert!(
        handler
            .lookup_prepared_build(7, &work_dir, None, None, &request)
            .await
            .is_none(),
        "a missing Python minor must not guess among advertised Python plans"
    );
    assert!(
        handler
            .lookup_prepared_build(7, &work_dir, None, Some("3.11"), &request)
            .await
            .is_some(),
        "an explicit Python minor may uniquely select its advertised plan"
    );

    // A newer initialize/output transaction must reject an older in-flight
    // publication, and an unrelated memory-cache hit must clear prior plans.
    let stale_transaction = handler.begin_prepared_transaction(7).await.unwrap();
    {
        let mut state = handler.state.write().await;
        state.generation = 8;
        state.prepared_transaction += 1;
        state.prepared_builds.clear();
    }
    assert!(
        !handler
            .publish_prepared_builds(
                7,
                stale_transaction,
                "stale-key".to_string(),
                vec![prepared.clone()],
            )
            .await
    );
    let transaction = handler.begin_prepared_transaction(8).await.unwrap();
    handler
        .publish_prepared_builds(
            8,
            transaction,
            "prepared-key".to_string(),
            vec![prepared.clone()],
        )
        .await;
    handler
        .retain_prepared_for_memory_cache_hit("other-key", &work_dir)
        .await;
    assert!(
        handler
            .lookup_prepared_build(8, &work_dir, None, Some("3.11"), &request)
            .await
            .is_none()
    );

    // Local wheel reuse additionally requires the exact file stamp and a
    // readable ZIP central directory. Corruption must return to resolve_all.
    let local_wheel = cache_dir.join("prepared-pack-1.0.0-py3-none-any.whl");
    {
        use std::io::Write;
        let file = std::fs::File::create(&local_wheel).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        zip.start_file(
            "prepared_pack-1.0.0.dist-info/METADATA",
            zip::write::FileOptions::<'_, ()>::default(),
        )
        .unwrap();
        zip.write_all(b"Metadata-Version: 2.1\nName: prepared-pack\nVersion: 1.0.0\n")
            .unwrap();
        zip.finish().unwrap();
    }
    let mut local_bundle = solo_bundle("prepared-pack", vec![]);
    local_bundle.primary.url = url::Url::from_file_path(&local_wheel).unwrap();
    let local_stamps = capture_local_wheel_stamps(std::slice::from_ref(&local_bundle));
    let local_plan = Arc::new(ResolvedTargetPlan {
        materialized: vec![local_bundle],
        base_config: declared_config.clone(),
        restore_relaxations: vec![],
        declared_config,
        target: crate::pypi::ResolutionTarget::for_subdir("3.11", "linux-64"),
        work_directory: work_dir.clone(),
        workspace_manifest_mtime: None,
        auto_overrides_fingerprint: "none".to_string(),
        local_wheel_stamps: local_stamps,
    });
    let local_prepared = PreparedBuild {
        locator_id: 0,
        plan: local_plan,
        bundle_index: 0,
        emission: hit.prepared.emission.clone(),
        advertised: hit.prepared.advertised.clone(),
        advertised_run_dependencies: hit.prepared.advertised_run_dependencies.clone(),
        relaxations: None,
        incremental_version_override: None,
    };
    let transaction = handler.begin_prepared_transaction(8).await.unwrap();
    handler
        .publish_prepared_builds(
            8,
            transaction,
            "local-key".to_string(),
            vec![local_prepared],
        )
        .await;
    assert!(
        handler
            .lookup_prepared_build(8, &work_dir, None, Some("3.11"), &request)
            .await
            .is_some()
    );
    std::fs::write(&local_wheel, b"not a wheel").unwrap();
    assert!(
        handler
            .lookup_prepared_build(8, &work_dir, None, Some("3.11"), &request)
            .await
            .is_none(),
        "a replaced/corrupt localized wheel must force normal resolution"
    );

    let _ = std::fs::remove_dir_all(&cache_dir);
}

/// An incremental metadata response may intentionally retain the committed
/// pack version even when the newly added source sorts first and has another
/// wheel version. That identity is safe only while the exact cold plan from
/// conda/outputs and the independently rediscovered build-time lock agree.
#[tokio::test]
async fn incremental_cold_fallback_requires_exact_prepared_plan() {
    let work_dir = std::env::temp_dir().join(format!(
        "retread-incremental-prepared-plan-{}",
        std::process::id()
    ));
    let mut config = cfg();
    config.courier = true;
    let mut base_bundle = solo_bundle("incremental-pack", vec![]);
    base_bundle.primary.metadata.version = "2.0.0".to_string();
    let emission = DiscoveredEmission {
        output_name: "incremental-pack".to_string(),
        channels: Vec::new(),
        transitive_overrides: BTreeMap::new(),
        envs: Vec::new(),
    };
    let (advertised_bundle, effective) = apply_emission(&base_bundle, &config, &emission);
    let output = produce_output(
        &advertised_bundle,
        &effective,
        Platform::Linux64,
        "3.11",
        &[],
        Some("incremental-prepared-hash"),
        Some("1.0.0"),
    )
    .unwrap();
    assert_eq!(output.metadata.version.to_string(), "1.0.0");
    assert_eq!(advertised_bundle.primary.metadata.version, "2.0.0");

    let plan = Arc::new(ResolvedTargetPlan {
        local_wheel_stamps: capture_local_wheel_stamps(std::slice::from_ref(&base_bundle)),
        materialized: vec![base_bundle],
        base_config: config.clone(),
        restore_relaxations: vec![],
        declared_config: config,
        target: ResolutionTarget::for_subdir("3.11", "linux-64"),
        work_directory: work_dir.clone(),
        workspace_manifest_mtime: None,
        auto_overrides_fingerprint: "none".to_string(),
    });
    let prepared = PreparedBuild {
        locator_id: 0,
        plan,
        bundle_index: 0,
        emission,
        advertised: PreparedOutputIdentity::from_metadata(&output.metadata),
        advertised_run_dependencies: output.run_dependencies.clone(),
        relaxations: None,
        incremental_version_override: Some("1.0.0".to_string()),
    };
    let mut sibling = prepared.clone();
    sibling.locator_id = 1;
    sibling.emission.output_name = "incremental-pack-sibling".to_string();
    sibling.advertised.name = "incremental-pack-sibling".to_string();
    let request = pixi_build_types::procedures::conda_build_v1::CondaBuildV1Output {
        name: output.metadata.name.clone(),
        version: Some(output.metadata.version.clone()),
        build: Some(output.metadata.build.clone()),
        subdir: output.metadata.subdir,
        variant: output.metadata.variant.clone(),
    };
    let handler = Handler::default();
    {
        let mut state = handler.state.write().await;
        state.generation = 11;
    }
    let transaction = handler.begin_prepared_transaction(11).await.unwrap();
    let cache_key = format!("incremental-multi-key-{}", std::process::id());
    assert!(
        handler
            .publish_prepared_builds(11, transaction, cache_key.clone(), vec![prepared, sibling],)
            .await
    );
    CONDA_OUTPUTS_CACHE
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
        .lock()
        .unwrap()
        .insert(
            cache_key.clone(),
            CondaOutputsMemo {
                result: pixi_build_types::procedures::conda_outputs::CondaOutputsResult {
                    outputs: vec![output.clone()],
                    input_globs: Default::default(),
                },
                requires_prepared_plan: true,
            },
        );

    let hit = handler
        .lookup_prepared_build(11, &work_dir, None, Some("3.11"), &request)
        .await
        .expect("incremental metadata must retain its exact cold plan");
    assert_eq!(hit.bundle.primary.metadata.version, "2.0.0");
    assert_eq!(
        hit.prepared.incremental_version_override.as_deref(),
        Some("1.0.0")
    );
    assert!(
        validate_prepared_incremental_version_handoff(
            hit.prepared.incremental_version_override.as_deref(),
            Some("1.0.0"),
            &hit.bundle.conda_name,
        )
        .is_ok()
    );
    assert!(
        validate_advertised_courier_version(
            &hit.bundle,
            Some("1.0.0"),
            hit.prepared.incremental_version_override.as_deref(),
        )
        .is_ok()
    );
    assert!(
        handler
            .retain_prepared_for_memory_cache_hit(&cache_key, &work_dir)
            .await,
        "an in-memory metadata memo may be reused only with its typed plan"
    );

    handler
        .consume_prepared_build(11, transaction, hit.prepared.locator_id)
        .await;
    assert!(
        handler
            .lookup_prepared_build(11, &work_dir, None, Some("3.11"), &request)
            .await
            .is_none()
    );
    assert!(
        !CONDA_OUTPUTS_CACHE
            .get()
            .unwrap()
            .lock()
            .unwrap()
            .contains_key(&cache_key),
        "consuming one output must evict the incremental memo even while sibling plans remain"
    );
    assert!(
        handler
            .retain_prepared_for_memory_cache_hit(&cache_key, &work_dir)
            .await,
        "the sibling plan remains, proving memo eviction does not rely on an empty plan set"
    );
    assert!(
        reject_unprepared_incremental_fallback(Some("1.0.0"), "incremental-pack").is_err(),
        "build_v1 must stop before fresh resolve_all when the typed plan is unavailable"
    );
    assert!(
        validate_advertised_courier_version(&hit.bundle, Some("1.0.0"), None).is_err(),
        "freshly resolved v2 bytes must never be relabeled as the stale v1 output"
    );
}

/// Metadata-to-build regression for a late incremental escalation. Metadata
/// has already advertised the committed lock version; if build-time channels
/// or newer repodata make the new source entry's BFS choose conda, localized
/// merge must escalate and the cold courier package must keep that same
/// advertised version even when its new primary wheel sorts first.
#[test]
fn routed_incremental_cold_fallback_keeps_advertised_lock_version() {
    let mut config = cfg();
    config.courier = true;
    config.name_map.insert(
        PypiKey::from_pypi("ray"),
        CondaTarget::Mapped(CondaName::new("ray-core")),
    );
    config
        .overrides
        .insert("ray".to_string(), "==2.49.1".to_string());

    let mut cold_bundle = solo_bundle("a-new-source-root", vec!["ray>=2.40,<3"]);
    cold_bundle.primary.metadata.version = "2.0.0".to_string();
    cold_bundle
        .probe_decisions
        .push(crate::audit::ProbeDecision {
            stage: "bfs".to_string(),
            pypi_name: "ray".to_string(),
            conda_name: "ray-core".to_string(),
            spec: "==2.49.1".to_string(),
            target_python: "3.11".to_string(),
            channels_consulted: Vec::new(),
            satisfiable: None,
            matching_candidates: 0,
            routing_decision: "short-circuit-explicit-override".to_string(),
        });
    assert!(
        incremental_bundle_requires_cold_resolve(&cold_bundle),
        "the new BFS route must exercise the late cold-escalation branch"
    );

    let metadata = produce_output(
        &cold_bundle,
        &config,
        Platform::Linux64,
        "3.11",
        &[],
        Some("routed-cold-hash"),
        Some("1.0.0"),
    )
    .unwrap();
    let advertised_version = metadata.metadata.version.to_string();
    assert_eq!(advertised_version, "1.0.0");
    assert_eq!(cold_bundle.primary.metadata.version, "2.0.0");

    let fallback_version = match incremental_version_plan(Some(&advertised_version), "1.0.0") {
        IncrementalVersionPlan::Attempt { fallback_version } => fallback_version,
        other => panic!("matching metadata/lock versions must attempt localized build: {other:?}"),
    };
    assert_eq!(
        courier_pack_version(&cold_bundle, Some(&fallback_version)),
        advertised_version,
        "late cold fallback must build the package identity metadata advertised"
    );
    assert!(
        validate_advertised_courier_version(
            &cold_bundle,
            Some(&advertised_version),
            Some(&fallback_version),
        )
        .is_ok(),
        "a build-time-positive incremental fallback may retain the committed lock version"
    );
}

/// A route retained from an unchanged grouped entry does not prove that the
/// newly-added entry will route. Metadata must keep the ordinary incremental
/// lock version so a successful localized build stays on the fast path. If a
/// stale/cold metadata response instead advertised another version, build_v1
/// must skip localized output rather than return the lock's version.
#[test]
fn unrelated_existing_route_keeps_incremental_fast_path_version_safe() {
    let mut grouped_cold_bundle = solo_bundle("a-new", vec![]);
    grouped_cold_bundle.primary.metadata.version = "2.0.0".to_string();
    grouped_cold_bundle
        .probe_decisions
        .push(crate::audit::ProbeDecision {
            stage: "bfs".to_string(),
            pypi_name: "old-native-dep".to_string(),
            conda_name: "old-native-dep".to_string(),
            spec: ">=1".to_string(),
            target_python: "3.11".to_string(),
            channels_consulted: vec!["conda-forge".to_string()],
            satisfiable: Some(true),
            matching_candidates: 1,
            routing_decision: "short-circuit".to_string(),
        });
    assert!(incremental_bundle_requires_cold_resolve(
        &grouped_cold_bundle
    ));

    assert_eq!(
        incremental_version_plan(Some("1.0.0"), "1.0.0"),
        IncrementalVersionPlan::Attempt {
            fallback_version: "1.0.0".to_string(),
        },
        "an unrelated retained route must not disable a valid localized add"
    );
    assert_eq!(
        incremental_version_plan(Some("2.0.0"), "1.0.0"),
        IncrementalVersionPlan::Cold,
        "a build request for cold metadata must never return lock.version"
    );
    assert_eq!(courier_pack_version(&grouped_cold_bundle, None), "2.0.0");
    assert!(
        validate_advertised_courier_version(&grouped_cold_bundle, Some("2.0.0"), None).is_ok(),
        "cold metadata may build only when the fresh primary version still matches"
    );
}

/// conda/outputs may detect an incremental add and advertise `lock.version`,
/// while build_v1 later fails that detection gate because the lock, config,
/// environment, or cache changed between RPCs. Without a persisted marker,
/// the cold build cannot prove that metadata's version came from that lock.
/// It must reject a newly resolved primary version instead of relabeling its
/// bytes under the stale advertised identity.
#[test]
fn disappeared_build_incremental_detection_rejects_primary_version_drift() {
    let mut config = cfg();
    config.courier = true;
    config.retread_wheels.insert(
        "a-new-primary".to_string(),
        WheelEntry {
            version: Some(">=1,<3".to_string()),
            ..WheelEntry::default()
        },
    );
    let mut cold_bundle = solo_bundle("a-new-primary", vec![]);
    cold_bundle.primary.metadata.version = "2.0.0".to_string();

    let metadata = produce_output(
        &cold_bundle,
        &config,
        Platform::Linux64,
        "3.11",
        &[],
        Some("detection-drift-hash"),
        Some("1.0.0"),
    )
    .unwrap();
    let advertised = metadata.metadata.version.to_string();
    let target = ResolutionTarget::for_subdir("3.11", "linux-64");
    let unchanged_manifest_build = current_courier_build_for_input_bundle(
        &config,
        "a-new-primary",
        &target,
        None,
        None,
        Path::new("/source"),
    );
    assert!(
        validate_advertised_courier_build(
            &config,
            "a-new-primary",
            &target,
            None,
            None,
            Path::new("/source"),
            Some(&unchanged_manifest_build),
        )
        .is_ok(),
        "a ranged input can keep the same manifest hash while resolving a newer primary"
    );
    assert_eq!(
        courier_pack_version(&cold_bundle, None),
        "2.0.0",
        "an ordinary cold build derives its identity from the bytes it just resolved"
    );
    assert!(
        validate_advertised_courier_version(&cold_bundle, Some(&advertised), None).is_err(),
        "a disappeared incremental detector must fail closed on primary version drift"
    );
    assert!(
        !advertised_version_matches(Some("2.0.0"), "1.0.0"),
        "replay must not return a lock version different from the requested output"
    );
    assert!(advertised_version_matches(None, "1.0.0"));
}

#[test]
fn conda_outputs_disk_cache_path_differs_by_key() {
    let cache_dir = std::env::temp_dir().join(format!(
        "retread-conda-outputs-disk-cache-path-test-{}",
        std::process::id()
    ));
    let source_dir = std::path::Path::new("/workspace/pypi-packs/some-pack");
    let p1 = conda_outputs_disk_cache_path(&cache_dir, "key-a", source_dir);
    let p2 = conda_outputs_disk_cache_path(&cache_dir, "key-b", source_dir);
    assert_ne!(p1, p2, "distinct cache keys must not collide on disk");
}

// v3.0.1 regression (#8): two DIFFERENT source packages (e.g.
// isaaclab-viral-pack and isaaclab-unitree-pack) built in the same
// workspace can share an IDENTICAL conda_outputs_cache_key (same
// platform/channels/variant/workspace-mtime), because CondaOutputsParams
// carries no package identity at all. Without source_dir folded into the
// disk-cache path, package B's process would load package A's cached
// CondaOutputsResult -- surfacing as pixi's "the package '<B>' is not
// provided by the project located at '<B's path>' (did you mean '<A>'?)".
#[test]
fn conda_outputs_disk_cache_path_differs_by_source_dir_even_with_identical_cache_key() {
    let cache_dir = std::env::temp_dir().join(format!(
        "retread-conda-outputs-disk-cache-collision-test-{}",
        std::process::id()
    ));
    let same_cache_key = "linux-64|linux-64|https://prefix.dev/conda-forge|None|123456";
    let viral = conda_outputs_disk_cache_path(
        &cache_dir,
        same_cache_key,
        std::path::Path::new("/workspace/pypi-packs/isaaclab-viral-pack"),
    );
    let unitree = conda_outputs_disk_cache_path(
        &cache_dir,
        same_cache_key,
        std::path::Path::new("/workspace/pypi-packs/isaaclab-unitree-pack"),
    );
    assert_ne!(
        viral, unitree,
        "sibling packages with identical params must not collide on the disk cache"
    );
}

// ── Pack-version mismatch fix: metadata-phase version_override ─────────────
//
// When an incremental-add is in flight, bundle.primary is the BTreeMap-first
// entry (e.g. iniconfig==2.0.0 sorts before isaacsim==6.0.0.1).  Without
// the override the metadata would promise "2.0.0" while the build would
// produce "6.0.0.1" → pixi rejects the package.  With the override
// (lock.version) both phases agree on "6.0.0.1".

/// produce_output with version_override=None returns bundle.primary version.
#[test]
fn produce_output_without_override_uses_primary_version() {
    // Bundle whose primary is "iniconfig" (sorts first alphabetically).
    let bundle = solo_bundle("iniconfig", vec![]);
    // primary.metadata.version == "1.0.0" (solo_bundle default)
    let out = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    assert_eq!(
        out.metadata.version.to_string(),
        "1.0.0",
        "no override: version must equal bundle.primary.metadata.version"
    );
}

/// produce_output with version_override=Some("6.0.0.1") returns the override.
#[test]
fn produce_output_with_override_uses_lock_version() {
    // Bundle whose primary is "iniconfig" (as if BTreeMap-first after adding it
    // alongside isaacsim==6.0.0.1).  primary.metadata.version == "1.0.0".
    let bundle = solo_bundle("iniconfig", vec![]);
    let out = produce_output(
        &bundle,
        &cfg(),
        Platform::Linux64,
        "3.11",
        &[],
        None,
        Some("6.0.0.1"),
    )
    .unwrap();
    assert_eq!(
        out.metadata.version.to_string(),
        "6.0.0.1",
        "version_override must win over bundle.primary.metadata.version"
    );
}

/// version_override propagates into the siblings list: the sibling entry
/// for a bundle whose version was overridden must use the lock version.
#[test]
fn siblings_with_override_use_lock_version() {
    // Two bundles: isaacsim (primary, version "6.0.0.1") and iniconfig
    // (BTreeMap-first, primary version "2.0.0").
    let main_bundle = solo_bundle("isaacsim", vec![]);
    // siblings list: override iniconfig to "6.0.0.1" (the lock version) so
    // the cross-link run-dep tracks the correct version.
    let siblings: Vec<(String, String)> = vec![
        ("isaacsim".to_string(), "6.0.0.1".to_string()),
        ("iniconfig".to_string(), "6.0.0.1".to_string()),
    ];
    // main bundle's produce_output should embed the sibling version correctly.
    let out = produce_output(
        &main_bundle,
        &cfg(),
        Platform::Linux64,
        "3.11",
        &siblings,
        None,
        Some("6.0.0.1"),
    )
    .unwrap();
    assert_eq!(
        out.metadata.version.to_string(),
        "6.0.0.1",
        "version_override must be reflected in the emitted metadata version"
    );
}

// --- retread-deps-from root assembly --------------------------------------

#[test]
fn root_req_name_normalizes_pep508_names() {
    assert_eq!(
        root_req_name("Foo_Bar[extra]==1.0"),
        Some("foo-bar".to_string())
    );
    assert_eq!(
        root_req_name("tensordict==0.9.0"),
        Some("tensordict".to_string())
    );
    // Unparsable garbage doesn't panic; caller falls back to the raw string.
    assert_eq!(root_req_name("!!! not a requirement !!!"), None);
}

#[test]
fn dedupe_roots_last_wins_keeps_last_occurrence_by_name() {
    let roots = vec![
        "isaacsim==5.1.0".to_string(),
        "tensordict==0.9.0".to_string(),
        // deps-from re-states isaacsim with a different pin: this must win.
        "isaacsim==5.2.0".to_string(),
    ];
    let deduped = dedupe_roots_last_wins(roots);
    assert_eq!(
        deduped,
        vec![
            "isaacsim==5.2.0".to_string(),
            "tensordict==0.9.0".to_string(),
        ]
    );
}

#[test]
fn dedupe_roots_last_wins_preserves_order_when_no_collisions() {
    let roots = vec!["a==1".to_string(), "b==2".to_string(), "c==3".to_string()];
    assert_eq!(dedupe_roots_last_wins(roots.clone()), roots);
}

#[test]
fn deps_from_roots_are_not_first_party_for_workspace_ownership() {
    let dependencies = vec![
        "gr00t".to_string(),
        "torch".to_string(),
        "torchvision>=0.22,<0.23".to_string(),
        "!!! preserve unparseable input !!!".to_string(),
    ];
    let deps_from_root_names =
        std::collections::BTreeSet::from(["torch".to_string(), "torchvision".to_string()]);

    assert_eq!(
        workspace_ownership_planning_dependencies(&dependencies, &deps_from_root_names),
        vec![
            "gr00t".to_string(),
            "!!! preserve unparseable input !!!".to_string(),
        ],
        "configured artifacts remain protected, while imported upstream roots may be supplied by every precise consumer's explicit PyPI dependency",
    );
}

/// End-to-end root-assembly test: a `retread-deps-from` local source's
/// PEP 508 lines make it into the root set `uv_group_closure` extends,
/// combined + deduped against a `[retread-wheels]` root — without driving
/// an actual uv solve (which `uv_group_closure` would need network /
/// the `uv` binary for). Exercises the exact two calls `uv_group_closure`
/// makes: `deps_from::resolve_deps_from` then
/// `dedupe_roots_last_wins`.
#[tokio::test]
async fn deps_from_roots_reach_closure_input_root_set() {
    let workspace = unique_test_dir("deps-from-roots");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::write(
        workspace.join("requirements_isaaclab.txt"),
        "tensordict==0.9.0\nlightning\nrtree==1.2.0\n",
    )
    .unwrap();

    let deps_from = vec![crate::deps_from::DepSource::Local(
        std::path::PathBuf::from("requirements_isaaclab.txt"),
    )];
    let cache_dir = workspace.join("cache");

    // Mirrors uv_group_closure: `[retread-wheels]` roots built first...
    let mut roots: Vec<String> = vec!["isaacsim==5.1.0".to_string()];
    // ...then retread-deps-from roots fetched + parsed + appended...
    let deps_from_parsed = crate::deps_from::resolve_deps_from(&deps_from, &workspace, &cache_dir)
        .await
        .expect("resolve_deps_from should succeed");
    roots.extend(deps_from_parsed.pypi_roots);
    // ...then deduped by name (last wins).
    let roots = dedupe_roots_last_wins(roots);

    assert_eq!(
        roots,
        vec![
            "isaacsim==5.1.0".to_string(),
            "tensordict==0.9.0".to_string(),
            "lightning".to_string(),
            "rtree==1.2.0".to_string(),
        ],
        "deps-from's parsed requirements must reach the root set fed to the uv closure"
    );

    std::fs::remove_dir_all(&workspace).ok();
}

#[test]
fn deps_from_conda_floors_apply_only_to_explicit_active_bare_roots() {
    let mut constraints = crate::uv_closure::ConstraintSet::default();
    let floors = vec![
        crate::deps_from::AdvisoryCondaFloor {
            conda_name: "pytorch".to_string(),
            floor_spec: ">=2.1.0".to_string(),
            source: "environment.yaml".to_string(),
        },
        crate::deps_from::AdvisoryCondaFloor {
            conda_name: "python".to_string(),
            floor_spec: ">=3.9.16".to_string(),
            source: "environment.yaml".to_string(),
        },
    ];
    let roots = vec![
        "torch ; sys_platform == 'linux'".to_string(),
        "python-package ; sys_platform == 'win32'".to_string(),
    ];
    let name_map = name_map(&[("torch", "pytorch"), ("python-package", "python")]);

    apply_deps_from_conda_floors(
        &mut constraints,
        &floors,
        &roots,
        &name_map,
        &BTreeSet::new(),
        &BTreeMap::new(),
        &[],
        "linux-64",
        "3.11",
    )
    .unwrap();

    assert_eq!(constraints.constraints, vec!["torch>=2.1.0"]);
    let provenance = &constraints.provenance["torch"];
    assert_eq!(provenance.provenance, Provenance::DepsFromRelaxed);
    assert_eq!(provenance.source, "deps-from-conda-advisory");
    assert_eq!(provenance.env, "environment.yaml");
    assert!(
        crate::uv_closure::attribute_conflict(
            "torch>=3 is incompatible with torch>=2.1.0",
            &constraints.provenance,
        )
        .is_empty(),
        "advisory floors must not authorize repair attribution"
    );
}

#[test]
fn deps_from_conda_floors_fail_closed_on_ambiguous_name_map() {
    let mut constraints = crate::uv_closure::ConstraintSet::default();
    let floors = vec![crate::deps_from::AdvisoryCondaFloor {
        conda_name: "pytorch".to_string(),
        floor_spec: ">=2.1.0".to_string(),
        source: "environment.yaml".to_string(),
    }];
    let name_map = name_map(&[("torch", "pytorch"), ("torch-alt", "pytorch")]);

    apply_deps_from_conda_floors(
        &mut constraints,
        &floors,
        &["torch".to_string()],
        &name_map,
        &BTreeSet::new(),
        &BTreeMap::new(),
        &[],
        "linux-64",
        "3.11",
    )
    .unwrap();

    assert!(constraints.constraints.is_empty());
    assert!(constraints.provenance.is_empty());
}

#[test]
fn deps_from_conda_floors_preserve_authoritative_inputs() {
    let mut constraints = crate::uv_closure::ConstraintSet::default();
    constraints.constraints.push("numpy>=2".to_string());
    constraints.provenance.insert(
        "numpy".to_string(),
        crate::uv_closure::ConstraintProvenance {
            constraint: "numpy>=2".to_string(),
            conda_name: "numpy".to_string(),
            conda_version: ">=2".to_string(),
            source: "workspace-solved".to_string(),
            env: "default".to_string(),
            provenance: Provenance::WorkspaceCondaFact("default".to_string()),
        },
    );
    let original_constraints = constraints.constraints.clone();
    let original_provenance = constraints.provenance.clone();
    let floors = ["numpy", "pandas", "scipy", "requests"]
        .into_iter()
        .map(|name| crate::deps_from::AdvisoryCondaFloor {
            conda_name: name.to_string(),
            floor_spec: ">=1".to_string(),
            source: "environment.yaml".to_string(),
        })
        .collect::<Vec<_>>();
    let name_map = ["numpy", "pandas", "scipy", "requests"]
        .into_iter()
        .map(|name| {
            (
                PypiKey::from_pypi(name),
                CondaTarget::Mapped(CondaName::new(name)),
            )
        })
        .collect::<NameMap>();
    let roots = vec![
        "numpy".to_string(),
        "pandas".to_string(),
        "scipy".to_string(),
        "requests==2.32.0".to_string(),
    ];

    apply_deps_from_conda_floors(
        &mut constraints,
        &floors,
        &roots,
        &name_map,
        &BTreeSet::new(),
        &BTreeMap::from([("pandas".to_string(), "==2.2.0".to_string())]),
        &["scipy".to_string()],
        "linux-64",
        "3.11",
    )
    .unwrap();

    assert_eq!(constraints.constraints, original_constraints);
    assert_eq!(constraints.provenance, original_provenance);
}

// --- deps_from_exact_pinned_names (conda-as-truth pin-softening) ----------

#[test]
fn deps_from_exact_pinned_names_flags_exact_and_triple_equal() {
    let roots = vec![
        "setuptools==69.5.1".to_string(),
        "tensordict===0.9.0".to_string(),
    ];
    let names = deps_from_exact_pinned_names(&roots);
    assert!(names.contains("setuptools"));
    assert!(names.contains("tensordict"));
    assert_eq!(names.len(), 2);
}

#[test]
fn deps_from_exact_pinned_names_ignores_non_exact_specs() {
    let roots = vec![
        "typer>=0.6.1".to_string(),
        "lightning".to_string(),
        "pkg[cli]>=1.9.4,<2".to_string(),
        "rtree!=1.2.0".to_string(),
    ];
    let names = deps_from_exact_pinned_names(&roots);
    assert!(
        names.is_empty(),
        "non-exact specs must not be flagged: {names:?}"
    );
}

#[test]
fn deps_from_exact_pinned_names_ignores_unparseable_lines() {
    let roots = vec!["!!! not a requirement !!!".to_string()];
    assert!(deps_from_exact_pinned_names(&roots).is_empty());
}

#[test]
fn deps_from_exact_pinned_names_canonicalizes_conda_name() {
    // Underscore/dot/dash normalization must match `canonical_conda_name`
    // so lookups against `AutoRoutedPackage::pypi_name` (PEP 503-canonical)
    // succeed regardless of the deps-from file's original spelling.
    let roots = vec!["My_Package.Name==1.0.0".to_string()];
    let names = deps_from_exact_pinned_names(&roots);
    assert!(names.contains("my-package-name"));
}

// ── Run 9: is_fresh() self-heals a corrupted cached wheel ───────────────────

#[cfg(unix)]
fn write_zip(path: &std::path::Path) {
    let file = std::fs::File::create(path).unwrap();
    let mut writer = zip::ZipWriter::new(file);
    writer
        .start_file("a.txt", zip::write::SimpleFileOptions::default())
        .unwrap();
    use std::io::Write as _;
    writer.write_all(b"hello").unwrap();
    writer.finish().unwrap();
}

/// A cached `.whl` newer than its input, but not a real zip (a node died
/// mid-write before the atomic rename landed -- or, pre-fix, before an
/// in-place write finished): `is_fresh` must reject it as NOT fresh AND
/// delete it, instead of trusting the mtime and handing a corrupt file to
/// the next pipeline phase (the run-9 "Could not find EOCD" failure).
#[cfg(unix)]
#[test]
fn is_fresh_self_heals_corrupted_cached_wheel() {
    let dir = unique_test_dir("is-fresh-selfheal");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("raw-1.0-py3-none-any.whl");
    let output = dir.join("raw-1.0-py3-none-any.injected.whl");

    std::fs::write(&input, b"raw wheel bytes").unwrap();
    // Truncated/garbage output, deliberately made newer than the input so
    // the plain mtime check alone would call it "fresh".
    std::fs::write(&output, b"not a valid zip, truncated mid-write").unwrap();
    let future = std::time::SystemTime::now() + std::time::Duration::from_secs(60);
    let out_file = std::fs::File::options().write(true).open(&output).unwrap();
    out_file.set_modified(future).unwrap();

    assert!(
        !is_fresh(&output, &input).unwrap(),
        "a corrupted cached wheel must never be reported fresh, even if newer than its input"
    );
    assert!(
        !output.exists(),
        "is_fresh must remove the corrupted cache file so the caller rebuilds it"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A genuinely valid, up-to-date cached wheel is still reported fresh and
/// left untouched -- the self-heal check must not false-positive on good
/// cache hits.
#[cfg(unix)]
#[test]
fn is_fresh_accepts_valid_cached_wheel() {
    let dir = unique_test_dir("is-fresh-valid");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("raw-1.0-py3-none-any.whl");
    let output = dir.join("raw-1.0-py3-none-any.injected.whl");

    std::fs::write(&input, b"raw wheel bytes").unwrap();
    write_zip(&output);
    let future = std::time::SystemTime::now() + std::time::Duration::from_secs(60);
    let out_file = std::fs::File::options().write(true).open(&output).unwrap();
    out_file.set_modified(future).unwrap();

    assert!(
        is_fresh(&output, &input).unwrap(),
        "a valid, newer-than-input cached wheel must be reported fresh"
    );
    assert!(output.exists(), "a valid cache hit must not be deleted");

    let _ = std::fs::remove_dir_all(&dir);
}

// ── Option D: cold-path lock-parity recovery ─────────────────────────────────
//
// docs/RETREAD_DETERMINISM_FIX_DESIGN.md. `conda/outputs` advertises a
// run-dependency set computed at lock time; `conda_build_v1`'s cold path
// re-derives it from live repodata plus a read-modify-write heal-facts ledger
// and can get a DIFFERENT set, which the gate then refuses. The drift is
// nondeterministic, so these tests CONSTRUCT the failing state (a bundle with
// one injected extra auto-route) instead of racing for it.

/// Render an emitted output's run deps back into the `"<name> <spec>"` strings
/// pixi records in pixi.lock and forwards as `params.run_dependencies`.
fn advertised_run_dependency_strings(output: &CondaOutput) -> Vec<String> {
    output
        .run_dependencies
        .depends
        .iter()
        .map(|dependency| {
            let spec = format_packagespec(&dependency.spec);
            if spec.is_empty() {
                dependency.name.clone()
            } else {
                format!("{} {}", dependency.name, spec)
            }
        })
        .collect()
}

fn recovery_target() -> crate::pypi::ResolutionTarget {
    crate::pypi::ResolutionTarget::from_wheel_target(
        crate::pypi::WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "linux-64".into(),
            max_glibc: None,
        },
        None,
    )
}

/// A committed pack lock whose `conda_run_deps` are exactly `run_deps`, split
/// the way `courier::parse_conda_deps` splits forwarded run-dep strings.
fn recovery_lock(
    bundle: &str,
    version: &str,
    inputs_hash: &str,
    run_deps: &[String],
) -> crate::lock::RetreadLock {
    let filename = format!(
        "{}-{version}-cp311-cp311-manylinux_2_17_x86_64.whl",
        bundle.replace('-', "_"),
    );
    let sha256 = "11".repeat(32);
    crate::lock::RetreadLock {
        schema: crate::lock::SCHEMA,
        retread_version: "0.0.1".into(),
        bundle: bundle.into(),
        version: version.into(),
        python: "3.11".into(),
        target_subdir: "linux-64".into(),
        target_contract: None,
        target_identity: None,
        target_scope: None,
        exact_workspace_envelope: false,
        resolution_glibc: None,
        inputs_hash: inputs_hash.into(),
        root_requirements: Vec::new(),
        wheels: vec![crate::lock::LockWheel {
            name: bundle.into(),
            version: version.into(),
            origin: crate::lock::Origin::Index,
            filename: filename.clone(),
            url: Some(format!("https://example.com/{filename}")),
            sha256: Some(sha256.clone()),
            requires_dist: vec![],
            must_ship: false,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }],
        conda_run_deps: run_deps
            .iter()
            .map(|raw| {
                let mut parts = raw.splitn(2, ' ');
                crate::lock::CondaDep {
                    name: parts.next().unwrap_or_default().to_string(),
                    spec: parts.next().unwrap_or_default().to_string(),
                }
            })
            .collect(),
        index_urls: vec!["https://pypi.org/simple/".into()],
        prerelease: BTreeMap::new(),
        shadow_libs: BTreeMap::new(),
        declared_glibc: None,
        conda_capable: vec![],
        entry_specs: vec![],
        wheel_store: None,
        abi_context: Some(crate::lock::LockAbiContext {
            wheels: vec![crate::lock::LockWheelAbiMetadata {
                name: bundle.into(),
                sha256,
                requires_dist: vec![],
            }],
        }),
        relaxations: vec![],
    }
}

/// The advertised (lock-time) bundle and the drifted (cold re-derivation)
/// bundle: identical except for ONE extra auto-routed conda dep, which is the
/// exact shape the observed failures take ("present in the rebuilt output but
/// not advertised: virtualenv, zipp"). Deterministic — no timing, no network,
/// no cache state.
fn advertised_and_drifted_bundles() -> (Bundle, Bundle) {
    let advertised = solo_bundle("robogen-pack", vec!["packaging>=23", "certifi"]);
    let mut drifted = advertised.clone();
    drifted.auto_routed.push(BundleAutoRoute {
        route: crate::uv_closure::AutoRoutedPackage {
            pypi_name: "zipp".to_string(),
            conda_name: "zipp".to_string(),
            pypi_version: "3.19.2".to_string(),
            conda_version: "3.19.2".to_string(),
            channel: "https://conda.example.invalid/noarch".to_string(),
            input_requirements: Vec::new(),
        },
        provenance: Provenance::PriorSelection,
        workspace_provider: None,
    });
    (advertised, drifted)
}

fn emit_for_recovery(bundle: &Bundle) -> CondaOutput {
    produce_output(bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None)
        .expect("emission must succeed")
}

fn courier_cfg() -> RetreadConfig {
    RetreadConfig {
        courier: true,
        ..cfg()
    }
}

/// The cold-path candidate the failure arm retains: identity matched, run
/// dependencies drifted.
fn recovery_candidate(bundle: &Bundle, inputs_hash: Option<&str>) -> ColdMismatchedCandidate {
    ColdMismatchedCandidate {
        bundle: bundle.clone(),
        effective: courier_cfg(),
        courier_hash: inputs_hash.map(ToString::to_string),
    }
}

/// Write a committed pack lock where `plan_cold_mismatch_recovery` looks for
/// it, and return its directory.
fn commit_recovery_lock(
    label: &str,
    version: &str,
    inputs_hash: &str,
    run_deps: &[String],
) -> std::path::PathBuf {
    let dir = unique_test_dir(label);
    std::fs::create_dir_all(&dir).unwrap();
    let lock = recovery_lock("robogen-pack", version, inputs_hash, run_deps);
    let path = dir.join(crate::lock::RetreadLock::file_name_for_target(
        "robogen-pack",
        &recovery_target(),
    ));
    std::fs::write(&path, lock.to_pretty_json().unwrap()).unwrap();
    dir
}

/// T1 (fails before / passes after): the constructed drift state is exactly
/// the one the gate refuses today, and the committed lock — which still
/// reproduces the advertisement — recovers it.
#[test]
fn cold_mismatch_recovery_replays_when_lock_matches_advertised() {
    let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
    let (advertised_bundle, drifted_bundle) = advertised_and_drifted_bundles();
    let advertised = advertised_run_dependency_strings(&emit_for_recovery(&advertised_bundle));

    // Sanity: the advertised strings faithfully round-trip, so a failure below
    // is real drift and not a rendering artefact of the test.
    assert!(
        run_dependencies_match(
            &emit_for_recovery(&advertised_bundle)
                .run_dependencies
                .depends,
            Some(&advertised),
        )
        .unwrap(),
        "the advertised rendering must match the emission it came from",
    );

    // The failing state the gate produces today.
    let candidate = emit_for_recovery(&drifted_bundle);
    assert!(
        !output_run_dependencies_match(&candidate, Some(&advertised)).unwrap(),
        "the injected auto-route must make the gate reject this candidate",
    );
    let delta = run_dependency_delta(&candidate, Some(&advertised));
    assert!(
        delta.contains("present in the rebuilt output but not advertised: zipp"),
        "gate delta must name the drifted dep, got: {delta}",
    );

    // The committed lock still reproduces the advertisement. Drive the whole
    // failure-arm decision (candidate retention → ABI context → lock location →
    // lock load → parity), not just its innermost comparison; everything after
    // this point is the shared `materialize_from_lock_for_target` replay the
    // top replay gate already uses.
    let dir = commit_recovery_lock("cold-recovery-match", "1.0.0", "inputs-hash-a", &advertised);
    let target = recovery_target();
    match plan_cold_mismatch_recovery(
        Some(recovery_candidate(&drifted_bundle, Some("inputs-hash-a"))),
        false,
        &courier_cfg(),
        &target,
        &dir,
        Some(&advertised),
        Some("1.0.0"),
    ) {
        ColdRecoveryPlan::Replay { bundle_name, lock } => {
            assert_eq!(bundle_name, "robogen-pack");
            let replayed: Vec<String> = lock
                .conda_run_deps
                .iter()
                .map(lock_run_dep_string)
                .collect();
            assert_eq!(
                replayed, advertised,
                "the replayed build must carry the advertised run deps exactly",
            );
        }
        ColdRecoveryPlan::Refuse { reason } => panic!(
            "a committed lock that reproduces the advertisement must recover the build, \
             got refusal: {reason}",
        ),
        ColdRecoveryPlan::NotAttempted => {
            panic!("recovery must be attempted for a courier pack with a drifted candidate")
        }
    }

    // RETREAD_NO_REPLAY=1 must disable this recovery too, not only the other
    // replay paths.
    // SAFETY: serialised by TEST_ENV_MUTEX; no concurrent env access.
    unsafe {
        std::env::set_var("RETREAD_NO_REPLAY", "1");
    }
    let outcome = plan_cold_mismatch_recovery(
        Some(recovery_candidate(&drifted_bundle, Some("inputs-hash-a"))),
        false,
        &courier_cfg(),
        &target,
        &dir,
        Some(&advertised),
        Some("1.0.0"),
    );
    // SAFETY: serialised by TEST_ENV_MUTEX; no concurrent env access.
    unsafe {
        std::env::remove_var("RETREAD_NO_REPLAY");
    }
    match outcome {
        ColdRecoveryPlan::Refuse { reason } => assert!(
            reason.contains("RETREAD_NO_REPLAY"),
            "the escape hatch must be named in the refusal, got: {reason}",
        ),
        other => panic!(
            "RETREAD_NO_REPLAY=1 must disable lock-parity recovery, got {}",
            recovery_plan_label(&other),
        ),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

fn recovery_plan_label(plan: &ColdRecoveryPlan) -> &'static str {
    match plan {
        ColdRecoveryPlan::NotAttempted => "NotAttempted",
        ColdRecoveryPlan::Replay { .. } => "Replay",
        ColdRecoveryPlan::Refuse { .. } => "Refuse",
    }
}

/// T2: a lock that disagrees with the advertisement must never build. This is
/// the hand-re-locked-pack state; the refusal has to name which record
/// disagrees, because "0 exact matches" alone is undiagnosable.
#[test]
fn cold_mismatch_recovery_refuses_when_lock_also_differs() {
    let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
    let (advertised_bundle, drifted_bundle) = advertised_and_drifted_bundles();
    let advertised = advertised_run_dependency_strings(&emit_for_recovery(&advertised_bundle));
    // The lock records its own third set: it dropped `certifi` and gained
    // `virtualenv`.
    let mut lock_deps: Vec<String> = advertised
        .iter()
        .filter(|dep| !dep.starts_with("certifi"))
        .cloned()
        .collect();
    lock_deps.push("virtualenv >=20".to_string());

    let dir = commit_recovery_lock(
        "cold-recovery-lock-differs",
        "1.0.0",
        "inputs-hash-a",
        &lock_deps,
    );
    match plan_cold_mismatch_recovery(
        Some(recovery_candidate(&drifted_bundle, Some("inputs-hash-a"))),
        false,
        &courier_cfg(),
        &recovery_target(),
        &dir,
        Some(&advertised),
        Some("1.0.0"),
    ) {
        ColdRecoveryPlan::Refuse { reason } => {
            assert!(
                reason.contains("committed lock also differs"),
                "refusal must say the LOCK is the disagreeing record, got: {reason}",
            );
            assert!(
                reason.contains("present in the committed lock but not advertised: virtualenv"),
                "refusal must name what the lock has and the advertisement does not, \
                 got: {reason}",
            );
            assert!(
                reason.contains("advertised but absent from the committed lock: certifi"),
                "refusal must name what the advertisement has and the lock does not, \
                 got: {reason}",
            );
        }
        other => panic!(
            "a lock that does not vouch for the advertisement must never be replayed, got {}",
            recovery_plan_label(&other),
        ),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// T3: no lock, an unusable lock, and a version-drifted lock all fail closed,
/// with distinguishable reasons.
#[test]
fn cold_mismatch_recovery_refuses_without_lock_or_on_hash_mismatch() {
    let advertised = vec!["python 3.11.*".to_string(), "certifi".to_string()];

    match recover_cold_mismatch_from_lock(Ok(None), Some(&advertised), Some("1.0.0")) {
        ColdMismatchRecovery::Refuse { reason } => assert!(
            reason.contains("no committed lock vouches"),
            "missing/hash-mismatched lock must say so, got: {reason}",
        ),
        ColdMismatchRecovery::ReplayFromLock(_) => panic!("no lock must never recover"),
    }

    let broken = recover_cold_mismatch_from_lock(
        Err(anyhow!("courier replay rejected by ABI invariant: numpy")),
        Some(&advertised),
        Some("1.0.0"),
    );
    match broken {
        ColdMismatchRecovery::Refuse { reason } => {
            assert!(
                reason.contains("committed lock unusable"),
                "an unusable lock must be reported as such, got: {reason}",
            );
            assert!(
                reason.contains("ABI invariant"),
                "the loader's own diagnosis must survive into the refusal, got: {reason}",
            );
        }
        ColdMismatchRecovery::ReplayFromLock(_) => panic!("an unusable lock must never recover"),
    }

    // Identity matched on version, so a lock recording a different version is
    // a different package: refuse rather than build it.
    let lock = recovery_lock("robogen-pack", "2.0.0", "inputs-hash-a", &advertised);
    match recover_cold_mismatch_from_lock(Ok(Some(lock)), Some(&advertised), Some("1.0.0")) {
        ColdMismatchRecovery::Refuse { reason } => assert!(
            reason.contains("records version `2.0.0`")
                && reason.contains("advertised version `1.0.0`"),
            "a version-drifted lock must name both versions, got: {reason}",
        ),
        ColdMismatchRecovery::ReplayFromLock(_) => {
            panic!("a lock for another version must never recover")
        }
    }

    // A missing lock file on disk reaches the same refusal through the wiring.
    let dir = unique_test_dir("cold-recovery-no-lock");
    std::fs::create_dir_all(&dir).unwrap();
    let (_, drifted_bundle) = advertised_and_drifted_bundles();
    match plan_cold_mismatch_recovery(
        Some(recovery_candidate(&drifted_bundle, Some("inputs-hash-a"))),
        false,
        &courier_cfg(),
        &recovery_target(),
        &dir,
        Some(&advertised),
        Some("1.0.0"),
    ) {
        ColdRecoveryPlan::Refuse { reason } => assert!(
            reason.contains("no committed lock vouches"),
            "an absent lock file must refuse, got: {reason}",
        ),
        other => panic!(
            "an absent lock file must refuse, got {}",
            recovery_plan_label(&other),
        ),
    }

    // Non-courier packs have no lock at all: the existing error must stand
    // completely unchanged (no appended reason).
    assert!(
        matches!(
            plan_cold_mismatch_recovery(
                Some(recovery_candidate(&drifted_bundle, None)),
                false,
                &cfg(),
                &recovery_target(),
                &dir,
                Some(&advertised),
                Some("1.0.0"),
            ),
            ColdRecoveryPlan::NotAttempted,
        ),
        "non-courier packs must not grow a lock-parity note on their error",
    );

    // Identity never matched (e.g. a rebuilt version): nothing to recover.
    assert!(
        matches!(
            plan_cold_mismatch_recovery(
                None,
                false,
                &courier_cfg(),
                &recovery_target(),
                &dir,
                Some(&advertised),
                Some("1.0.0"),
            ),
            ColdRecoveryPlan::NotAttempted,
        ),
        "no identity-matched candidate means no recovery attempt",
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// T4: the parity check is the SAME matcher the gate uses, so pixi's
/// post-`conda/outputs` `python_abi` run-export is tolerated identically on
/// both comparisons. A second, subtly different comparison would fail here.
#[test]
fn cold_mismatch_recovery_applies_python_abi_allowance_symmetrically() {
    let lock_deps = vec![
        "python 3.11.*".to_string(),
        "certifi".to_string(),
        "packaging >=23".to_string(),
    ];
    let mut advertised = lock_deps.clone();
    advertised.push("python_abi 3.11.* *_cp311".to_string());

    let lock = recovery_lock("robogen-pack", "1.0.0", "inputs-hash-a", &lock_deps);
    assert!(
        matches!(
            recover_cold_mismatch_from_lock(Ok(Some(lock)), Some(&advertised), Some("1.0.0")),
            ColdMismatchRecovery::ReplayFromLock(_),
        ),
        "pixi's injected python_abi must not defeat lock parity",
    );

    // The allowance is one-directional by design: a lock carrying a dep the
    // advertisement never mentioned is still a real disagreement.
    let mut extra = lock_deps.clone();
    extra.push("zipp ==3.19.2".to_string());
    let lock = recovery_lock("robogen-pack", "1.0.0", "inputs-hash-a", &extra);
    assert!(
        matches!(
            recover_cold_mismatch_from_lock(Ok(Some(lock)), Some(&lock_deps), Some("1.0.0")),
            ColdMismatchRecovery::Refuse { .. },
        ),
        "a lock with an unadvertised dep must not pass parity",
    );
}

/// T5 (the "don't break the 13 working environments" guard): when the gate
/// passes, none of the recovery code may run — no lock load, no extra I/O.
///
/// Two halves. The behavioural half proves an undrifted candidate passes the
/// gate, so the failure arm (the only place recovery lives) is never entered.
/// The structural half pins the recovery INSIDE that arm: it fails if anyone
/// hoists the lock load above the gate, or drops the wiring altogether.
#[test]
fn cold_mismatch_recovery_never_consulted_when_candidate_matches() {
    let (advertised_bundle, _) = advertised_and_drifted_bundles();
    let advertised = advertised_run_dependency_strings(&emit_for_recovery(&advertised_bundle));
    assert!(
        output_run_dependencies_match(&emit_for_recovery(&advertised_bundle), Some(&advertised))
            .unwrap(),
        "an undrifted candidate must pass the gate, which is what keeps the \
         recovery branch unreachable for the environments that build today",
    );

    let src = include_str!("mod.rs");
    let loop_start = src
        .find("let mut matching_bundles = Vec::new();")
        .expect("cold-path candidate loop must be findable");
    let gate = src
        .find("] = matching_bundles.as_slice()")
        .expect("cold-path gate must be findable");
    assert!(loop_start < gate, "the loop precedes the gate");
    let before_gate = &src[loop_start..gate];
    for forbidden in [
        "load_replayable_lock",
        "materialize_from_lock_for_target",
        "plan_cold_mismatch_recovery",
        "recover_cold_mismatch_from_lock",
        "replay_abi_context_for_bundle",
    ] {
        assert!(
            !before_gate.contains(forbidden),
            "`{forbidden}` must not run before the cold-path gate decides: every \
             environment that builds today passes the gate and must pay no extra I/O",
        );
    }
    let failure_arm = &src[gate..];
    assert!(
        failure_arm.contains("plan_cold_mismatch_recovery("),
        "lock-parity recovery must be wired into the cold-path failure arm",
    );

    // `plan_cold_mismatch_recovery` performs the only cold-path lock read, so
    // one call site is the whole story: the gate-passing path cannot reach it.
    assert_eq!(
        src.matches("plan_cold_mismatch_recovery(").count(),
        2,
        "exactly one definition and one call site; a second caller would need \
         its own proof that the gate-passing path cannot reach it",
    );
}

/// T6: two identity-matching mismatched candidates cannot happen today
/// (identity includes the package name), but if they ever do, no lock may be
/// attributed to either — the existing ambiguity error must stand.
#[test]
fn cold_mismatch_recovery_disabled_for_ambiguous_identity() {
    let _env_guard = super::TEST_ENV_MUTEX.lock().unwrap();
    let (advertised_bundle, drifted_bundle) = advertised_and_drifted_bundles();
    let advertised = advertised_run_dependency_strings(&emit_for_recovery(&advertised_bundle));
    // A lock that WOULD have vouched for the advertisement, so the refusal can
    // only come from the ambiguity guard.
    let dir = commit_recovery_lock(
        "cold-recovery-ambiguous",
        "1.0.0",
        "inputs-hash-a",
        &advertised,
    );

    match plan_cold_mismatch_recovery(
        Some(recovery_candidate(&drifted_bundle, Some("inputs-hash-a"))),
        true,
        &courier_cfg(),
        &recovery_target(),
        &dir,
        Some(&advertised),
        Some("1.0.0"),
    ) {
        ColdRecoveryPlan::Refuse { reason } => assert!(
            reason.contains("refusing lock-parity recovery") && reason.contains("unambiguously"),
            "the ambiguity refusal must say why, got: {reason}",
        ),
        other => panic!(
            "an ambiguous identity match must never pick a lock, got {}",
            recovery_plan_label(&other),
        ),
    }

    let _ = std::fs::remove_dir_all(&dir);
}
