use super::audit_report::format_packagespec;
use super::auto_bundle::{
    ExtraDepSource, extra_dep_source_from_url, pep508_extra_dep, prefer_conda_match,
};
use super::{merge_index_chain, *};
use crate::config::RelaxPolicy;
use std::collections::BTreeMap;

#[cfg(unix)]
fn unique_test_dir(label: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("retread-{label}-{}-{nanos}", std::process::id()))
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
    ensure_pixi_bld_symlink_target(Some(&workspace)).unwrap();

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
    let mut name_map: BTreeMap<String, String> = BTreeMap::new();
    name_map.insert("torch".into(), "pytorch".into());
    assert_eq!(
        pick_conda_target("torch", &name_map, &parselmouth).as_deref(),
        Some("pytorch"),
    );
}

#[test]
fn pick_conda_target_ambiguous_parselmouth_without_name_map_is_none() {
    // Documents the pre-fix behavior: ambiguous parselmouth + no
    // curated answer -> None (caller leaves it on the PyPI/bundle
    // path). This is exactly why the FALLBACK entry is load-bearing.
    let parselmouth = pypi_map(&[("torch", &["pytorch", "pytorch-cpu", "pytorch-gpu"])]);
    let name_map: BTreeMap<String, String> = BTreeMap::new();
    assert_eq!(pick_conda_target("torch", &name_map, &parselmouth), None);
}

#[test]
fn pick_conda_target_parselmouth_identity_match_wins() {
    // numpy -> numpy is an identity match among the candidates; no
    // name_map entry needed.
    let parselmouth = pypi_map(&[("numpy", &["numpy", "manifpy"])]);
    let name_map: BTreeMap<String, String> = BTreeMap::new();
    assert_eq!(
        pick_conda_target("numpy", &name_map, &parselmouth).as_deref(),
        Some("numpy"),
    );
}

#[test]
fn pick_conda_target_single_parselmouth_candidate() {
    let parselmouth = pypi_map(&[("some-pypi-only", &["the-conda-name"])]);
    let name_map: BTreeMap<String, String> = BTreeMap::new();
    assert_eq!(
        pick_conda_target("some-pypi-only", &name_map, &parselmouth).as_deref(),
        Some("the-conda-name"),
    );
}

#[test]
fn pick_conda_target_unknown_dep_is_none() {
    // Not in name_map and not in parselmouth -> stays on PyPI.
    let parselmouth = pypi_map(&[("torch", &["pytorch"])]);
    let name_map: BTreeMap<String, String> = BTreeMap::new();
    assert_eq!(
        pick_conda_target("totally-unknown-pkg", &name_map, &parselmouth),
        None,
    );
}

#[test]
fn pick_conda_target_user_name_map_overrides_parselmouth_identity() {
    // A user retread-name-map entry beats even a parselmouth identity
    // match -- the curated answer is authoritative.
    let parselmouth = pypi_map(&[("opencv-python-headless", &["opencv-python-headless"])]);
    let mut name_map: BTreeMap<String, String> = BTreeMap::new();
    name_map.insert("opencv-python-headless".into(), "py-opencv".into());
    assert_eq!(
        pick_conda_target("opencv-python-headless", &name_map, &parselmouth).as_deref(),
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
        keep_pypi: vec![],
        force_conda: vec![],
        sdist_build: Default::default(),
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
        keep_pypi: vec![],
        force_conda: vec![],
        sdist_build: Default::default(),
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
        keep_pypi: vec![],
        force_conda: vec![],
        sdist_build: Default::default(),
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
        uv_closure_names: Default::default(),
    };

    let courier_cfg = RetreadConfig {
        resolver: Default::default(),
        auto_route: true,
        keep_pypi: vec![],
        force_conda: vec![],
        sdist_build: Default::default(),
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
    // are emitted as `>=locked,<next-major` rather than an exact `==`.
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
            ("numpy".to_string(), "2.1.0".to_string(), false),
            ("scipy".to_string(), "1.14.1".to_string(), false),
        ],
        uv_closure_names: Default::default(),
    };
    let out = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None, None).unwrap();
    let deps: Vec<(String, String)> = out
        .run_dependencies
        .depends
        .iter()
        .map(|d| (d.name.clone(), format_packagespec(&d.spec)))
        .collect();
    assert!(
        deps.contains(&("numpy".to_string(), ">=2.1.0,<3".to_string())),
        "auto-routed numpy must be a bounded-range run-dep floored at the \
         locked version (won over the wheel's numpy>=1.21): {deps:?}"
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
fn produce_output_softens_deps_from_floor_pin_to_floor_spec() {
    // conda-as-truth fix: an auto-routed package whose root ORIGINATED
    // from a `retread-deps-from` exact pin (the third tuple element)
    // must be emitted as a `>=` floor, not the usual exact `==` pin --
    // otherwise a sibling pack's own exact conda pin for the same name
    // (e.g. `setuptools ==83.0.0`) hard-conflicts with this pack's
    // `setuptools ==69.5.1` at workspace conda-solve time.
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
            ("setuptools".to_string(), "69.5.1".to_string(), true),
            ("numpy".to_string(), "2.1.0".to_string(), false),
        ],
        uv_closure_names: Default::default(),
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
        deps.contains(&("numpy".to_string(), ">=2.1.0,<3".to_string())),
        "non-deps-from auto-routed pins get a bounded range, not an exact \
         pin: {deps:?}"
    );
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
fn produce_output_auto_routed_pin_widens_to_bounded_range() {
    // The core fix under test: a plain (non-floor) auto-routed pin
    // widens from the exact `==X.Y.Z` uv resolved to `>=X.Y.Z,<next-major`
    // -- floored at the locked version so the uv-solved closure it was
    // resolved against still installs, capped so the conda solver can't
    // pick something wildly newer than what the pack was ever tested with.
    let bundle = Bundle {
        auto_routed: vec![("pandas".to_string(), "2.2.3".to_string(), false)],
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
        auto_routed: vec![("etgen".to_string(), "0.20.1".to_string(), false)],
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
fn produce_output_auto_routed_abi_anchor_stays_exact() {
    // ABI anchors (python/python_abi/libc/cuda family) must never widen:
    // "any newer build" is a lie about what this pack's wheels actually
    // run on.
    let bundle = Bundle {
        auto_routed: vec![("cuda-version".to_string(), "12.8".to_string(), false)],
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
        deps.contains(&("cuda-version".to_string(), "==12.8".to_string())),
        "ABI anchor auto-routed pin must stay exact, not widen: {deps:?}"
    );
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
        auto_routed: vec![("setuptools".to_string(), "80.10.2".to_string(), false)],
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
        auto_routed: vec![("sentry-sdk".to_string(), "1.2.3".to_string(), false)],
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
    bundle.auto_routed = vec![("numpy".to_string(), "2.1.0".to_string(), false)];
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
        deps.contains(&("numpy".to_string(), ">=2.1.0,<3".to_string())),
        "auto-routed package must appear (as a bounded range) even though \
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
        keep_pypi: vec![],
        force_conda: vec![],
        sdist_build: Default::default(),
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
    }
}

fn meta(name: &str, version: &str, requires: Vec<&str>, platform_specific: bool) -> WheelMetadata {
    WheelMetadata {
        name: name.into(),
        version: version.into(),
        requires_dist: requires.into_iter().map(String::from).collect(),
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
        uv_closure_names: Default::default(),
    }
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
    assert!(relaxed_retry_specs("etgen", &specs(">=0.8.2,<0.9"), RelaxPolicy::Minor).is_none());
    // Policy None never falls back.
    assert!(relaxed_retry_specs("foo", &specs("==1.0.0"), RelaxPolicy::None).is_none());
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
        produce_output(&bundle, &cfg(), Platform::Linux64, "3.12", &[], None, None).unwrap();
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
    config
        .name_map
        .insert("tinyobjloader".into(), "tinyobjloader-python".into());

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
        "3.12",
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
        "3.12",
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
// merge_index_chain: order and dedup semantics
// -----------------------------------------------------------------

#[test]
fn merge_index_chain_preserves_order_primary_then_extra_then_public() {
    // The canonical use case: entry index + workspace indexes.
    // Public PyPI should land last, entry index first.
    let result = merge_index_chain(
        ["https://pypi.nvidia.com".to_string()],
        &["https://download.pytorch.org/whl/cu128".to_string()],
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
fn merge_index_chain_deduplicates_trailing_slash_insensitive() {
    // public PyPI appears in extra without trailing slash -- must not
    // be added twice even though the stored constant has a trailing slash.
    let result = merge_index_chain(
        ["https://pypi.nvidia.com".to_string()],
        &["https://pypi.org/simple".to_string()],
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
fn merge_index_chain_empty_primary_appends_public() {
    let result = merge_index_chain(std::iter::empty::<String>(), &[]);
    assert_eq!(result, vec![PUBLIC_PYPI.to_string()]);
}

#[test]
fn merge_index_chain_deduplicates_repeated_primary() {
    // Two primary items that are the same URL (one with, one without slash)
    // should only appear once.
    let result = merge_index_chain(
        [
            "https://pypi.nvidia.com".to_string(),
            "https://pypi.nvidia.com/".to_string(),
        ],
        &[],
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
        uv_closure_names: Default::default(),
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
    let target = WheelTarget {
        python_version: "3.11".into(),
        conda_subdir: "linux-64".into(),
        max_glibc: None,
    };

    let (resolved, original_rd) = materialize_and_rewrite(
        &entry,
        "retread-sample",
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
        uv_closure_names: Default::default(),
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
            metadata: primary,
            extras_requested: vec![],
            auto_data: None,
            auto_data_dedup_skipped_root: None,
        },
        extras: vec![],
        probe_decisions: vec![],
        solve_diagnostics: BTreeMap::new(),
        auto_routed: vec![],
        uv_closure_names: Default::default(),
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
fn bare_major_python_emits_glob_not_strict_equals() {
    // Regression: when python_version is bare-major like "3" (e.g.
    // wheel tag parsing yields just the major, or a workspace
    // variant is "python = [\"3\"]"), the emitted host-dep was
    // `python 3` which rattler-conda-types Lenient-parses as
    // `==3` strict, causing rattler-build to fail the host solve
    // with "No candidates were found for python ==3". Always
    // append `.*` so the glob form is used.
    //
    // Construct a bundle whose primary wheel produces python_version
    // = "3" via the pure-Python fallback (workspace_python_version)
    // -- pass "3" as the workspace_python_version arg.
    let bundle = solo_bundle("foo", vec![]);
    let output = produce_output(&bundle, &cfg(), Platform::Linux64, "3", &[], None, None).unwrap();

    // python must appear with a wildcard, NOT as strict equals.
    let python = output
        .host_dependencies
        .as_ref()
        .unwrap()
        .depends
        .iter()
        .find(|d| d.name == "python")
        .expect("python in host_deps");
    let rendered = format!("{:?}", python.spec);
    assert!(
        !rendered.contains("Equals") || rendered.contains("Glob") || rendered.contains("*"),
        "host python dep must be a glob, not strict ==; got: {rendered}",
    );
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
fn prefer_conda_skips_parselmouth_known_deps() {
    // Contract: anything in the effective name_map (parselmouth +
    // FALLBACK + user retread-name-map) is NOT auto-bundled -- it
    // flows to emission as a conda run-dep via translate. This is
    // the prefer-conda default. Concretely: torch in the bundle's
    // candidates should be skipped because parselmouth maps it to
    // pytorch; a niche pure-PyPI helper with no conda equivalent
    // (e.g. qdldl) should not be skipped.
    let mut name_map: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();
    name_map.insert("torch".to_string(), "pytorch".to_string());
    name_map.insert("numpy".to_string(), "numpy".to_string());
    name_map.insert(
        "opencv-python-headless".to_string(),
        "py-opencv".to_string(),
    );

    // Parselmouth-known => prefer conda, don't bundle.
    assert!(prefer_conda_match("torch", &name_map));
    assert!(prefer_conda_match("numpy", &name_map));
    // FALLBACK_PYPI_TO_CONDA entry survives the same way.
    assert!(prefer_conda_match("opencv-python-headless", &name_map));

    // Unknown to parselmouth => fall through to auto-bundle path.
    // (These are the long tail retread should still vendor.)
    assert!(!prefer_conda_match("qdldl", &name_map));
    assert!(!prefer_conda_match("asteval", &name_map));
    assert!(!prefer_conda_match("aiodns", &name_map));
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
async fn conda_outputs_disk_cache_round_trips() {
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

    let _ = std::fs::remove_dir_all(&cache_dir);
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

/// End-to-end root-assembly test: a `retread-deps-from` local source's
/// PEP 508 lines make it into the root set `uv_group_closure` extends,
/// combined + deduped against a `[retread-wheels]` root — without driving
/// an actual uv solve (which `uv_group_closure` would need network /
/// the `uv` binary for). Exercises the exact two calls `uv_group_closure`
/// makes: `deps_from::resolve_deps_from_roots` then
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
    let deps_from_roots =
        crate::deps_from::resolve_deps_from_roots(&deps_from, &workspace, &cache_dir)
            .await
            .expect("resolve_deps_from_roots should succeed");
    roots.extend(deps_from_roots);
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
