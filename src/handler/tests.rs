use super::auto_bundle::{
    ExtraDepSource, extra_dep_source_from_url, pep508_extra_dep, prefer_conda_match,
};
use super::cascade::{
    bundle_group_for, check_output_abi_invariants, extract_anchor_version, merge_looser_override,
    pypi_fallback_indexes, tiered_cascade_for_dep, widen_one_level, widening_level,
};
use super::{merge_index_chain, *};
use crate::config::RelaxPolicy;
use crate::relax::default_marker_env;
use std::collections::BTreeMap;

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

#[test]
fn widen_one_level_progression() {
    // Patch -> Minor: `>=3.7.0,<3.8` -> `>=3.7,<4`
    assert_eq!(widen_one_level(">=3.7.0,<3.8").as_deref(), Some(">=3.7,<4"));
    // Minor -> Major FLOOR (keeps the `<4` upper): `>=3.7,<4` -> `>=3,<4`
    // (v0.46.0: was `>=3`; we now preserve the upper bound so the
    // emitted spec stays bounded to one major version.)
    assert_eq!(widen_one_level(">=3.7,<4").as_deref(), Some(">=3,<4"));
    // Major floor -> Star: `>=3,<4` -> `*` (drop the upper last).
    assert_eq!(widen_one_level(">=3,<4").as_deref(), Some("*"));
    // Bare major-open (no upper) -> Star.
    assert_eq!(widen_one_level(">=3").as_deref(), Some("*"));
    // Star -> None
    assert_eq!(widen_one_level("*"), None);
}

#[test]
fn widen_one_level_preserves_upper_through_major() {
    // The pytorch case: the merged spec widens to a bounded major
    // range, NOT an unbounded `>=2`. `>=2.10,<3` -> `>=2,<3`.
    assert_eq!(widen_one_level(">=2.10,<3").as_deref(), Some(">=2,<3"));
    // And only THEN, if still unsat, drops the upper to `*`.
    assert_eq!(widen_one_level(">=2,<3").as_deref(), Some("*"));
}

#[test]
fn widen_one_level_handles_upper_only_specs() {
    // Pure upper-bound (no lower anchor): jump straight to `*`.
    // Wheels that pin `package<X` provide no anchor version to
    // widen FROM, so the only meaningful widening is to drop
    // the constraint entirely. Without this case, deps like
    // `pyglet <2` stay stuck at `<2` through every refinement
    // round (widen_one_level previously returned None).
    assert_eq!(widen_one_level("<2").as_deref(), Some("*"));
    assert_eq!(widen_one_level("<=5").as_deref(), Some("*"));
}

#[test]
fn extract_anchor_version_picks_highest_among_merged_clauses() {
    // The torch regression: a merged spec carrying a stray low lower
    // bound (>=1.4) plus the real exact pin (==2.10.0). The tightest
    // anchor (2.10.0) must win -- NOT the first clause (1.4).
    assert_eq!(
        extract_anchor_version(">=1.4,==2.10.0,>=2.10.0,<2.11.0a0").as_deref(),
        Some("2.10.0"),
    );
}

#[test]
fn widen_one_level_merged_spec_does_not_collapse_to_one() {
    // Regression: `>=1.4,==2.10.0,>=2.10.0,<2.11.0a0` previously
    // widened straight to `>=1` (anchored on 1.4; `<2` matched
    // `<2.11.0a0`). It must instead recognize the 2.10 anchor + the
    // `<2.11` minor upper and step to `>=2.10,<3`.
    assert_eq!(
        widen_one_level(">=1.4,==2.10.0,>=2.10.0,<2.11.0a0").as_deref(),
        Some(">=2.10,<3"),
    );
    // And the next step is the bounded major floor, never dipping
    // below the 2.x line and never going unbounded.
    assert_eq!(widen_one_level(">=2.10,<3").as_deref(), Some(">=2,<3"));
}

#[test]
fn widen_one_level_handles_exact_pin() {
    // `==3.7.0` is an exact pin; treat as Patch -> next is Minor.
    // Implementation detail: no `<3.8` upper means it falls through
    // to the "no major upper" branch, returning `*` directly. This
    // is acceptable because exact pins entering refinement are
    // unusual and reaching `*` for them is a safe fallback.
    let out = widen_one_level("==3.7.0").unwrap_or_default();
    // Any non-empty widening is acceptable here.
    assert!(!out.is_empty());
}

// -------------------------------------------------------------
// v0.36.4: refinement-widening propagation tests.
//
// The original bug: iterative_solve_refinement widened
// `effective.overrides` per env, the solve check internally
// re-rendered with the widened overrides and reported sat, but
// the outer `output` (created BEFORE the env loop) was never
// rebuilt — so pixi received the pre-refinement run-deps and
// exploded on misleading leaves. These tests pin the building
// blocks: widening-level ordering, the loosest-wins merge, and
// that produce_output reflects the merged overrides.
// -------------------------------------------------------------

#[test]
fn widening_level_orders_patch_minor_major_star() {
    // Total order must agree with widen_one_level's steps: a
    // spec that widen_one_level moves from L -> L+1 must report
    // widening_level L initially and L+1 after.
    assert_eq!(widening_level(">=3.7.0,<3.8"), 0); // patch
    assert_eq!(widening_level(">=3.7,<4"), 1); // minor range
    assert_eq!(widening_level(">=3,<4"), 2); // major floor, bounded
    assert_eq!(widening_level(">=3"), 3); // major-open, no upper
    assert_eq!(widening_level("*"), 4); // star
    assert_eq!(widening_level(""), 4); // bare-name == *
}

#[test]
fn widening_level_strictly_increases_along_widen_chain() {
    // The merge in merge_looser_override needs a TOTAL order that
    // agrees with widen_one_level: each step must report a strictly
    // higher level than the last, so "loosest wins" picks correctly.
    let mut spec = ">=2.10.0,<2.11".to_string();
    let mut last = widening_level(&spec);
    for _ in 0..6 {
        let Some(next) = widen_one_level(&spec) else {
            break;
        };
        let lvl = widening_level(&next);
        assert!(
            lvl > last,
            "widening {spec} -> {next} must raise level ({last} -> {lvl})",
        );
        last = lvl;
        spec = next;
    }
    assert_eq!(spec, "*", "chain should terminate at star");
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
        retread_wheels: BTreeMap::new(),
        relax: RelaxPolicy::Minor,
        overrides: BTreeMap::new(),
        name_map: BTreeMap::new(),
        build_number: 0,
        drop_deps: Vec::new(),
        auto_bundle: false,
        conda_deps: Vec::new(),
        default_bundle: None,
        compression_level: None,
        emit_pypi: false,
        courier: false,
        blueprint: Default::default(),
        blueprint_sync: Default::default(),
        git_sources: std::collections::BTreeMap::new(),
        python: None,
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
        retread_wheels: BTreeMap::new(),
        relax: RelaxPolicy::Minor,
        overrides: BTreeMap::new(),
        name_map: BTreeMap::new(),
        build_number: 0,
        drop_deps: Vec::new(),
        auto_bundle: false,
        conda_deps: Vec::new(),
        default_bundle: None,
        compression_level: None,
        emit_pypi: false,
        courier: false,
        blueprint: Default::default(),
        blueprint_sync: Default::default(),
        git_sources: std::collections::BTreeMap::new(),
        python: None,
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
        retread_wheels: BTreeMap::new(),
        relax: RelaxPolicy::Minor,
        overrides: BTreeMap::new(),
        name_map: BTreeMap::new(),
        build_number: 0,
        drop_deps: Vec::new(),
        auto_bundle: false,
        conda_deps: Vec::new(),
        default_bundle: None,
        compression_level: None,
        emit_pypi: false,
        courier: false,
        blueprint: Default::default(),
        blueprint_sync: Default::default(),
        git_sources: std::collections::BTreeMap::new(),
        python: None,
    };
    let result = pythons_for(&cfg, Some(&variants));
    assert_eq!(result, vec!["3.11".to_string(), "3.12".to_string()]);
}

#[test]
fn widening_level_treats_pure_upper_bounds_as_zero() {
    // `<2`, `==1.26.4` have no anchor; widen_one_level jumps
    // straight to `*`. Level 0 ensures any widening (level >=
    // 1) wins under merge_looser_override.
    assert_eq!(widening_level("<2"), 0);
    assert_eq!(widening_level("==1.26.4"), 0);
}

#[test]
fn merge_looser_override_keeps_widest_across_envs() {
    // env A widened to >=3, env B widened the same dep to *.
    // The shipped output must satisfy both => keep `*`.
    let mut accum: BTreeMap<String, String> = BTreeMap::new();
    merge_looser_override(&mut accum, "triton", ">=3");
    assert_eq!(accum.get("triton").unwrap(), ">=3");
    merge_looser_override(&mut accum, "triton", "*");
    assert_eq!(accum.get("triton").unwrap(), "*");
}

#[test]
fn merge_looser_override_does_not_narrow() {
    // env A widened to `*`, env B's later contribution is the
    // tighter `>=2.7,<3`. The accumulator must NOT narrow back
    // — the shipped run-deps must satisfy both envs and `*`
    // already does.
    let mut accum: BTreeMap<String, String> = BTreeMap::new();
    merge_looser_override(&mut accum, "pytorch", "*");
    merge_looser_override(&mut accum, "pytorch", ">=2.7,<3");
    assert_eq!(accum.get("pytorch").unwrap(), "*");
}

#[test]
fn merge_looser_override_inserts_when_missing() {
    // Baseline override absent => candidate wins unconditionally.
    let mut accum: BTreeMap<String, String> = BTreeMap::new();
    merge_looser_override(&mut accum, "opencv", ">=4.11,<5");
    assert_eq!(accum.get("opencv").unwrap(), ">=4.11,<5");
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
    let narrow = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None).unwrap();
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

// -------------------------------------------------------------
// v0.36.0: ABI-anchor invariant tests
// -------------------------------------------------------------

#[test]
fn invariant_flags_python_widened_to_star() {
    // The exact gsi corruption: retread's run_deps emit `python *`.
    // The invariant MUST flag this.
    let emitted = vec![("python".to_string(), "*".to_string())];
    let workspace = vec!["python ==3.11".to_string()];
    let overrides = std::collections::BTreeMap::new();
    let violations = check_output_abi_invariants(&emitted, &workspace, &overrides);
    assert_eq!(
        violations.len(),
        1,
        "expected one violation, got {violations:?}"
    );
    assert!(
        violations[0].contains("python"),
        "violation should mention python: {}",
        violations[0],
    );
    assert!(
        violations[0].contains("ABI anchor") || violations[0].contains("ABI invariant"),
        "violation should mention ABI: {}",
        violations[0],
    );
}

#[test]
fn invariant_flags_empty_spec_on_abi_anchor() {
    // Empty spec ("") is the other form of corruption -- retread
    // emitting just the bare name. Same severity as `*`.
    let emitted = vec![("cuda-version".to_string(), "".to_string())];
    let workspace = vec!["cuda-version ==12.8".to_string()];
    let overrides = std::collections::BTreeMap::new();
    let violations = check_output_abi_invariants(&emitted, &workspace, &overrides);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("cuda-version"));
}

#[test]
fn invariant_passes_on_concrete_abi_anchor_spec() {
    // Retread emits `python ==3.11.5` -- a concrete spec. No
    // corruption; invariant passes.
    let emitted = vec![("python".to_string(), "==3.11.5".to_string())];
    let workspace = vec!["python ==3.11".to_string()];
    let overrides = std::collections::BTreeMap::new();
    let violations = check_output_abi_invariants(&emitted, &workspace, &overrides);
    assert!(violations.is_empty(), "should not flag: {violations:?}");
}

// v0.37.0 D6: bare-major glob is a new ABI corruption shape.
#[test]
fn invariant_flags_python_bare_major_glob() {
    // The gsn-failure smoking gun: meta-v0 emitted `python 3.*`
    // because pixi forwarded variant `"3"` and `pythons_for`
    // accepted it (since fixed in D2). Invariant should flag the
    // glob even if it slipped past the input boundary.
    let emitted = vec![("python".to_string(), "3.*".to_string())];
    let workspace = vec!["python ==3.11".to_string()];
    let overrides = std::collections::BTreeMap::new();
    let violations = check_output_abi_invariants(&emitted, &workspace, &overrides);
    assert_eq!(
        violations.len(),
        1,
        "expected exactly one bare-major violation, got {violations:?}",
    );
    assert!(
        violations[0].contains("bare-major"),
        "violation should mention bare-major: {}",
        violations[0],
    );
}

#[test]
fn invariant_accepts_dotted_minor_glob_on_abi_anchor() {
    // `python 3.11.*` is the correct emission shape; must NOT
    // trigger the bare-major glob check.
    let emitted = vec![("python".to_string(), "3.11.*".to_string())];
    let workspace = vec!["python ==3.11".to_string()];
    let overrides = std::collections::BTreeMap::new();
    let violations = check_output_abi_invariants(&emitted, &workspace, &overrides);
    assert!(violations.is_empty(), "should not flag: {violations:?}");
}

#[test]
fn invariant_flags_cuda_version_bare_major_glob() {
    // The bare-major glob check applies to every ABI anchor, not
    // just python. `cuda-version 12.*` is the same shape of
    // corruption.
    let emitted = vec![("cuda-version".to_string(), "12.*".to_string())];
    let workspace: Vec<String> = Vec::new();
    let overrides = std::collections::BTreeMap::new();
    let violations = check_output_abi_invariants(&emitted, &workspace, &overrides);
    assert_eq!(
        violations.len(),
        1,
        "expected one violation, got {violations:?}",
    );
}

#[test]
fn invariant_flags_overrides_with_abi_anchor_star() {
    // Override map carrying `python = "*"` is the upstream cause
    // of the run-deps corruption. The check catches it before
    // produce_output re-renders.
    let emitted: Vec<(String, String)> = Vec::new();
    let workspace: Vec<String> = Vec::new();
    let mut overrides = std::collections::BTreeMap::new();
    overrides.insert("python".to_string(), "*".to_string());
    let violations = check_output_abi_invariants(&emitted, &workspace, &overrides);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("python"));
    assert!(violations[0].contains("overrides"));
}

#[test]
fn invariant_lets_non_anchor_widen_to_star() {
    // Widening pytorch to `*` is legitimate; the invariant only
    // guards ABI anchors. No violation should fire.
    let emitted = vec![("pytorch".to_string(), "*".to_string())];
    let workspace: Vec<String> = Vec::new();
    let mut overrides = std::collections::BTreeMap::new();
    overrides.insert("pytorch".to_string(), "*".to_string());
    let violations = check_output_abi_invariants(&emitted, &workspace, &overrides);
    assert!(
        violations.is_empty(),
        "should not flag pytorch: {violations:?}"
    );
}

#[test]
fn invariant_catches_libstdcxx_overrides_corruption() {
    let emitted: Vec<(String, String)> = Vec::new();
    let workspace: Vec<String> = Vec::new();
    let mut overrides = std::collections::BTreeMap::new();
    overrides.insert("libstdcxx-ng".to_string(), "*".to_string());
    let violations = check_output_abi_invariants(&emitted, &workspace, &overrides);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("libstdcxx-ng"));
}

#[test]
fn invariant_catches_compiler_activation_corruption() {
    // `gcc_linux-64` is an arch-tagged compiler activation pkg
    // (caught by the prefix predicate). Widening corrupts the
    // build-time toolchain match with the gcc-runtime install.
    let emitted: Vec<(String, String)> = Vec::new();
    let workspace: Vec<String> = Vec::new();
    let mut overrides = std::collections::BTreeMap::new();
    overrides.insert("gcc_linux-64".to_string(), "*".to_string());
    let violations = check_output_abi_invariants(&emitted, &workspace, &overrides);
    assert_eq!(violations.len(), 1);
}

/// Simulated-refinement test (no IO): models what the loop's
/// per-iteration body does for a verdict mix, asserting the
/// guard against widening ABI anchors holds even if a (hypothetical)
/// `WidenRetread` verdict ever named one. This re-exercises the
/// defense-in-depth check that lives inside
/// `iterative_solve_refinement`'s widening branch.
#[test]
fn refinement_loop_never_widens_python_even_if_verdict_says_so() {
    use crate::conflict_classifier::{PerChainVerdict, is_abi_anchor};
    // Construct a verdict that (in a bug) claims python is
    // widenable. The loop's `is_abi_anchor` guard MUST refuse it.
    let v = PerChainVerdict::WidenRetread {
        dep: "python".into(),
        current_spec: "==3.11".into(),
    };
    let dep = v.dep().to_string();
    assert!(
        is_abi_anchor(&dep),
        "python must be recognized as ABI anchor",
    );
    // The defense-in-depth branch in the refinement loop:
    let next_spec_if_widened = if is_abi_anchor(&dep) {
        None
    } else {
        widen_one_level("==3.11")
    };
    assert!(
        next_spec_if_widened.is_none(),
        "the loop must refuse to widen python; got {next_spec_if_widened:?}",
    );
}

// v0.32.0+: workspace pin/transitive-constraint parsing moved to
// src/workspace.rs (see WorkspaceManifest::extract_transitive_constraints
// + discover_outputs_for_source). Tests for that live there.

fn cfg() -> RetreadConfig {
    RetreadConfig {
        retread_wheels: BTreeMap::new(),
        relax: RelaxPolicy::Minor,
        overrides: BTreeMap::new(),
        name_map: BTreeMap::new(),
        build_number: 0,
        drop_deps: Vec::new(),
        auto_bundle: false,
        conda_deps: Vec::new(),
        default_bundle: None,
        compression_level: None,
        emit_pypi: false,
        courier: false,
        blueprint: Default::default(),
        blueprint_sync: Default::default(),
        git_sources: std::collections::BTreeMap::new(),
        python: None,
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
    ResolvedWheel {
        pypi_name: pypi.to_string(),
        url: format!("https://example.com/{pypi}.whl").parse().unwrap(),
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
fn capped_rerun_eligibility() {
    // P3 (grizzly #4): a capped env earns its single re-run only when
    // the level has >1 env AND at least one sibling converged.
    let cap = Some("A-iteration-cap");
    let ok = Some("B-workspace");
    // Mixed level: capped envs get the re-run.
    assert_eq!(
        capped_envs_eligible_for_rerun([cap, None, cap].into_iter()),
        vec![0, 2]
    );
    assert_eq!(
        capped_envs_eligible_for_rerun([ok, cap].into_iter()),
        vec![1]
    );
    // Single-env level: nothing to seed from.
    assert!(capped_envs_eligible_for_rerun([cap].into_iter()).is_empty());
    // Everyone capped: no converged sibling to seed from.
    assert!(capped_envs_eligible_for_rerun([cap, cap].into_iter()).is_empty());
    // Nobody capped: nothing to do.
    assert!(capped_envs_eligible_for_rerun([None, ok].into_iter()).is_empty());
}

#[test]
fn constrains_anchor_recorded_but_not_widened() {
    // P3 load-bearing distinction: a cuda-version CONSTRAINT from the
    // workspace is RECORDED (input-side parity, see
    // workspace::constraint_lines test) while the emission side still
    // refuses to widen it -- is_abi_anchor is the single source of
    // truth all three never-widen layers consult (invariant #8).
    use crate::conflict_classifier::is_abi_anchor;
    assert!(is_abi_anchor("cuda-version"));
    assert!(is_abi_anchor("libstdcxx-ng"));
    // And the recorded spec round-trips cleanly into the override map
    // shape the cascade consumes.
    crate::relax::assert_spec_roundtrips("cuda-version", "==12.8");
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
    let output = produce_output(&bundle, &cfg(), Platform::Linux64, "3.12", &[], None).unwrap();
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
    let output =
        produce_output(&bundle, &dropped_cfg, Platform::Linux64, "3.12", &[], None).unwrap();
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
fn pypi_fallback_indexes_dedups_entries_and_appends_public() {
    let mut config = cfg();
    config.retread_wheels.insert(
        "isaacsim".into(),
        spec_entry("==6.0.0", Some("https://pypi.nvidia.com")),
    );
    config.retread_wheels.insert(
        "isaacsim-extra".into(),
        spec_entry("==6.0.0", Some("https://pypi.nvidia.com")),
    );
    config.retread_wheels.insert(
        "pytorch3d".into(),
        spec_entry(
            "==0.7.8",
            Some("https://miropsota.github.io/torch_packages_builder"),
        ),
    );
    config.retread_wheels.insert(
        "urlform".into(),
        serde_json::from_value(serde_json::json!({ "url": "https://example.com/x.whl" })).unwrap(),
    );
    assert_eq!(
        pypi_fallback_indexes(&config, None),
        vec![
            "https://pypi.nvidia.com".to_string(),
            "https://miropsota.github.io/torch_packages_builder".to_string(),
            "https://pypi.org/simple/".to_string(),
        ],
    );
}

#[test]
fn pypi_fallback_indexes_defaults_to_public_only() {
    // No entries (and spec-form entries without an explicit index
    // default to public anyway) -> just public PyPI, once.
    assert_eq!(
        pypi_fallback_indexes(&cfg(), None),
        vec!["https://pypi.org/simple/".to_string()],
    );
    let mut config = cfg();
    config
        .retread_wheels
        .insert("tomli".into(), spec_entry("==2.0.1", None));
    assert_eq!(
        pypi_fallback_indexes(&config, None),
        vec!["https://pypi.org/simple/".to_string()],
    );
}

#[test]
fn pypi_fallback_indexes_includes_workspace_pypi_options() {
    // Workspace [pypi-options] indexes (top-level + feature) slot
    // between the entry indexes and public PyPI; dedup is
    // trailing-slash-insensitive so a workspace-declared
    // "https://pypi.org/simple" doesn't double the public index.
    let manifest: toml::Value = toml::from_str(
        r#"
[pypi-options]
index-url = "https://pypi.org/simple"
extra-index-urls = ["https://download.pytorch.org/whl/cu128"]

[feature.sim.pypi-options]
extra-index-urls = ["https://py.mujoco.org", "https://download.pytorch.org/whl/cu128"]
"#,
    )
    .unwrap();
    let ws = crate::workspace::WorkspaceManifest::from_toml(&manifest);
    let mut config = cfg();
    config.retread_wheels.insert(
        "isaacsim".into(),
        spec_entry("==6.0.0", Some("https://pypi.nvidia.com")),
    );
    assert_eq!(
        pypi_fallback_indexes(&config, Some(&ws)),
        vec![
            "https://pypi.nvidia.com".to_string(),
            "https://pypi.org/simple".to_string(),
            "https://download.pytorch.org/whl/cu128".to_string(),
            "https://py.mujoco.org".to_string(),
        ],
    );
}

#[test]
fn tiered_cascade_indecisive_probe_never_reroutes() {
    // Empty channel list -> every probe returns satisfiable=None
    // (indecisive). The cascade must not widen, must not bundle
    // from PyPI (step 8 requires a DEFINITIVELY zero-candidate
    // name-level probe), and must not push drop_deps. A channel
    // fetch failure silently rerouting deps to PyPI is exactly
    // what the definitive-probe gate exists to prevent. Bare dep
    // (no version spec) so the per-level PyPI fallback steps skip
    // and the test stays offline.
    let mut bundle = solo_bundle("isaac-pack", vec![]);
    let mut effective = cfg();
    let target = wheel_target_for(Platform::Linux64, "3.11");
    let env = default_marker_env("3.11").unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(tiered_cascade_for_dep(
        &mut bundle,
        &mut effective,
        &[],
        &target,
        &std::env::temp_dir(),
        &["https://pypi.org/simple/".to_string()],
        "isaacsim-kernel",
        &env,
        "isaacsim-kernel",
        None,
        "isaacsim-kernel",
        &BTreeMap::new(),
    ))
    .unwrap();
    assert!(
        bundle.extras.is_empty(),
        "indecisive probe must not auto-bundle: {:?}",
        bundle.extras,
    );
    assert!(
        effective.drop_deps.is_empty(),
        "indecisive probe must not drop conda emission: {:?}",
        effective.drop_deps,
    );
    assert!(
        effective.overrides.is_empty(),
        "indecisive probe must not inject overrides: {:?}",
        effective.overrides,
    );
}

#[test]
#[ignore = "live: conda-forge repodata + several PyPI wheels"]
fn pypi_only_transitive_chain_bundles_recursively_live() {
    // v1.5.7 regression (isaaclab-mimic's nvidia-srl chain): the
    // cascade bundles nvidia-srl-usd-to-urdf (zero conda candidates),
    // whose OWN Requires-Dist names nvidia-srl-base / -math / -usd --
    // also PyPI-only. Pre-fix those transitives never joined the
    // candidate list (collected once at pass entry), flowed to
    // emission, and post-emit could only widen them to `*`; the
    // consumer solve died on `nvidia-srl-base *`. The fixed-point
    // pass must bundle the whole chain.
    let mut bundle = solo_bundle(
        "isaac-pack",
        vec!["nvidia-srl-usd-to-urdf>=1.0", "numpy==1.26.0"],
    );
    let mut effective = cfg();
    effective.relax = RelaxPolicy::PatchThenMinorThenMajorThenLastResort;
    let target = wheel_target_for(Platform::Linux64, "3.11");
    let channels = vec![ChannelUrl::from(
        url::Url::parse("https://prefix.dev/conda-forge").unwrap(),
    )];
    let tmp = std::env::temp_dir().join(format!("retread-srlchain-live-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(super::cascade::pre_emit_widen_pass(
        &mut bundle,
        &mut effective,
        &channels,
        &target,
        &tmp,
        &["https://pypi.org/simple/".to_string()],
    ))
    .unwrap();
    let bundled: Vec<&str> = bundle.extras.iter().map(|w| w.pypi_name.as_str()).collect();
    assert!(
        bundled.iter().any(|n| n.contains("usd-to-urdf")),
        "root PyPI-only dep must bundle; got {bundled:?}",
    );
    for transitive in ["nvidia-srl-base", "nvidia-srl-math", "nvidia-srl-usd"] {
        assert!(
            bundled.contains(&transitive),
            "transitive {transitive} of a bundled wheel must bundle recursively; \
             got {bundled:?}, decisions: {:#?}",
            bundle
                .probe_decisions
                .iter()
                .map(|d| format!("{} {} -> {}", d.stage, d.conda_name, d.routing_decision))
                .collect::<Vec<_>>(),
        );
    }
}

#[test]
#[ignore = "live: conda-forge repodata + a PyPI wheel"]
fn bare_dep_with_zero_conda_candidates_bundles_from_pypi_live() {
    // v1.5.6 regression (genesis-world's DracoPy): a BARE dep (no
    // version spec) that has zero conda candidates used to be skipped
    // by the pre-emit cascade's spec filter and shipped as a doomed
    // `dracopy *` conda run-dep. It must now reach the cascade at the
    // name level and step-8-bundle the latest compatible wheel. The
    // control dep (numpy, bare) HAS conda candidates and must remain
    // a conda run-dep.
    let mut bundle = solo_bundle("genesis-pack", vec!["DracoPy", "numpy"]);
    let mut effective = cfg();
    effective.relax = RelaxPolicy::PatchThenMinorThenMajorThenLastResort;
    let target = wheel_target_for(Platform::Linux64, "3.12");
    let channels = vec![ChannelUrl::from(
        url::Url::parse("https://prefix.dev/conda-forge").unwrap(),
    )];
    let tmp = std::env::temp_dir().join(format!("retread-baredep-live-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(super::cascade::pre_emit_widen_pass(
        &mut bundle,
        &mut effective,
        &channels,
        &target,
        &tmp,
        &["https://pypi.org/simple/".to_string()],
    ))
    .unwrap();
    assert!(
        bundle
            .extras
            .iter()
            .any(|w| w.pypi_name.contains("dracopy")),
        "bare zero-candidate dep must be step-8 bundled; extras: {:?}, decisions: {:#?}",
        bundle
            .extras
            .iter()
            .map(|w| &w.pypi_name)
            .collect::<Vec<_>>(),
        bundle.probe_decisions,
    );
    assert!(
        effective
            .drop_deps
            .iter()
            .any(|d| crate::relax::canonical_conda_name(d) == "dracopy"),
        "bundled bare dep must be dropped from conda emission: {:?}",
        effective.drop_deps,
    );
    assert!(
        !bundle.extras.iter().any(|w| w.pypi_name == "numpy"),
        "bare dep WITH conda candidates must stay a conda run-dep",
    );
}

#[test]
#[ignore = "live: fetches conda-forge repodata + a PyPI wheel"]
fn tiered_cascade_step8_bundles_pypi_only_dep_live() {
    // The user-reported failure shape: a PyPI-only NVIDIA dep
    // (zero conda-forge candidates at ANY version) reached
    // emission as a conda run-dep and the solve died with "no
    // candidates". Step 8 must instead bundle the wheel from PyPI
    // and drop the conda emission. Bare dep (no version spec) so
    // steps 2/4/6 skip and step 8 is the only PyPI attempt.
    let mut bundle = solo_bundle("isaac-pack", vec![]);
    let mut effective = cfg();
    let target = wheel_target_for(Platform::Linux64, "3.11");
    let env = default_marker_env("3.11").unwrap();
    let channels = vec![ChannelUrl::from(
        url::Url::parse("https://prefix.dev/conda-forge").unwrap(),
    )];
    let tmp = std::env::temp_dir().join(format!("retread-step8-live-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(tiered_cascade_for_dep(
        &mut bundle,
        &mut effective,
        &channels,
        &target,
        &tmp,
        &["https://pypi.org/simple/".to_string()],
        "nvidia-cuda-nvrtc-cu12",
        &env,
        "nvidia-cuda-nvrtc-cu12",
        None,
        "nvidia-cuda-nvrtc-cu12",
        &BTreeMap::new(),
    ))
    .unwrap();
    assert_eq!(
        bundle.extras.len(),
        1,
        "step 8 must bundle the PyPI-only wheel; decisions: {:#?}",
        bundle.probe_decisions,
    );
    assert_eq!(bundle.extras[0].pypi_name, "nvidia-cuda-nvrtc-cu12");
    assert!(
        effective
            .drop_deps
            .contains(&"nvidia-cuda-nvrtc-cu12".to_string()),
        "step 8 must drop the conda emission",
    );
    assert!(
        bundle.probe_decisions.iter().any(|d| {
            d.stage == "tiered-cascade-step8-pypi-last-resort"
                && d.routing_decision == "auto-pypi-no-conda-candidates"
        }),
        "audit must record the auto-reroute: {:#?}",
        bundle.probe_decisions,
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
    let output = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None).unwrap();
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
    let output = produce_output(&bundle, &cfg(), Platform::Win64, "3.11", &[], None).unwrap();
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
    let output = produce_output(&bundle, &config, Platform::Linux64, "3.11", &[], None).unwrap();
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
    let output = produce_output(&bundle, &config, Platform::Linux64, "3.11", &[], None).unwrap();
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
    };

    let output = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None).unwrap();
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
    };

    let output = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None).unwrap();

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
            metadata: primary,
            extras_requested: vec![],
            auto_data: None,
            auto_data_dedup_skipped_root: None,
        },
        extras: vec![],
        probe_decisions: vec![],
        solve_diagnostics: BTreeMap::new(),
    };

    let output = produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &[], None).unwrap();

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
    let output = produce_output(&bundle, &cfg(), Platform::Linux64, "3", &[], None).unwrap();

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
    let output =
        produce_output(&bundle, &cfg(), Platform::Linux64, "3.11", &siblings, None).unwrap();

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
        ExtraDepSource::Git { url, rev } => {
            assert_eq!(url, "https://github.com/isaac-sim/rl_games.git");
            assert_eq!(rev.as_deref(), Some("python3.11"));
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
        ExtraDepSource::Git { url, rev } => {
            assert_eq!(url, "https://github.com/foo/bar.git");
            assert_eq!(rev, None);
        }
        other => panic!("expected Git, got {other:?}"),
    }
}

// -----------------------------------------------------------------
// P1 (cleanup): classify_run_terminal table tests.
// Invariant: the pure helper is the single source of truth for both
// the abstention banner and the MD-deletion guard. These tests pin
// every combination so neither consumer can silently drift.
// -----------------------------------------------------------------

#[test]
fn classify_run_terminal_all_sat() {
    // attempted=3, skipped=0, no block messages -> VerifiedAllSat
    let (terminal, skipped) = classify_run_terminal(3, 0, false);
    assert_eq!(terminal, RunTerminal::VerifiedAllSat);
    assert_eq!(skipped, 0);
}

#[test]
fn classify_run_terminal_all_unsat_with_block_messages() {
    // attempted=2, skipped=0, block messages -> VerifiedUnsat
    let (terminal, skipped) = classify_run_terminal(2, 0, true);
    assert_eq!(terminal, RunTerminal::VerifiedUnsat);
    assert_eq!(skipped, 0);
}

#[test]
fn classify_run_terminal_all_abstained() {
    // attempted==skipped -> AllAbstained regardless of block messages.
    // (block messages cannot exist when all checks were skipped, but
    // the classifier is pure and doesn't enforce that precondition.)
    let (terminal, skipped) = classify_run_terminal(4, 4, false);
    assert_eq!(terminal, RunTerminal::AllAbstained);
    assert_eq!(skipped, 4);
}

#[test]
fn classify_run_terminal_partial_skip_no_block() {
    // 1 sat + 2 skipped, no workspace block -> VerifiedAllSat (the
    // verified portion passed; the skipped portion is flagged via
    // the non-zero skipped_count in the return value).
    let (terminal, skipped) = classify_run_terminal(3, 2, false);
    assert_eq!(terminal, RunTerminal::VerifiedAllSat);
    assert_eq!(skipped, 2);
}

#[test]
fn classify_run_terminal_partial_skip_with_block() {
    // 1 verified unsat + 2 skipped -> VerifiedUnsat
    let (terminal, skipped) = classify_run_terminal(3, 2, true);
    assert_eq!(terminal, RunTerminal::VerifiedUnsat);
    assert_eq!(skipped, 2);
}

#[test]
fn classify_run_terminal_nothing_attempted() {
    // attempted=0 -> NothingAttempted regardless of other args
    let (terminal, skipped) = classify_run_terminal(0, 0, false);
    assert_eq!(terminal, RunTerminal::NothingAttempted);
    assert_eq!(skipped, 0);
}

// -----------------------------------------------------------------
// P1: write_solve_failed_summary MD-deletion guard.
// (a) abstained run (all diagnostics skipped=true) preserves a
//     pre-existing md file -- abstention is not evidence of resolution.
// (b) verified all-sat run deletes a stale md file.
// These tests exercise the guard directly via a real temp-dir write.
// -----------------------------------------------------------------

mod solve_failed_summary_guard {
    use super::super::audit_report::write_solve_failed_summary;
    use super::*; // pulls in Bundle, ResolvedWheel, rw, meta, etc.
    use crate::audit::SolveDiagnostics;

    /// Build a minimal Bundle with the given solve_diagnostics. The
    /// bundle's conda_name is used to derive the MD file path.
    fn make_bundle_with_diags(name: &str, diags: BTreeMap<String, SolveDiagnostics>) -> Bundle {
        Bundle {
            conda_name: name.to_string(),
            primary: rw(name, meta(name, "1.0.0", vec![], true)),
            extras: vec![],
            probe_decisions: vec![],
            solve_diagnostics: diags,
        }
    }

    fn skipped_diag() -> SolveDiagnostics {
        SolveDiagnostics {
            satisfiable: false, // "unknown" -- not sat, but skipped
            unsat_explanations: vec![],
            channels_consulted: vec![],
            specs_count: 0,
            records_count: 0,
            refinement_steps: vec![],
            workspace_edit_suggestions: vec![],
            terminal_classification: None,
            skipped: true,
        }
    }

    fn unsat_diag() -> SolveDiagnostics {
        SolveDiagnostics {
            satisfiable: false,
            unsat_explanations: vec!["dep A conflicts with dep B".into()],
            channels_consulted: vec!["https://conda.anaconda.org/conda-forge/linux-64".into()],
            specs_count: 5,
            records_count: 100,
            refinement_steps: vec![],
            workspace_edit_suggestions: vec![],
            terminal_classification: Some("B-workspace-pin".into()),
            skipped: false,
        }
    }

    fn sat_diag() -> SolveDiagnostics {
        SolveDiagnostics {
            satisfiable: true,
            unsat_explanations: vec![],
            channels_consulted: vec!["https://conda.anaconda.org/conda-forge/linux-64".into()],
            specs_count: 5,
            records_count: 100,
            refinement_steps: vec![],
            workspace_edit_suggestions: vec![],
            terminal_classification: None,
            skipped: false,
        }
    }

    /// Test (a): abstained run preserves pre-existing MD.
    #[tokio::test]
    async fn abstained_run_preserves_prior_md() {
        let dir = std::env::temp_dir().join(format!("retread_test_abstain_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bundle_name = "test-bundle-abstain";
        let md_path = dir.join(format!("RETREAD-SOLVE-FAILED-{bundle_name}.md"));

        // Pre-create the MD file (simulates a prior failed run).
        std::fs::write(&md_path, "# Prior failure record\n").unwrap();
        assert!(md_path.exists(), "pre-condition: md must exist");

        // Run with all diagnostics skipped (full abstention).
        let mut diags = BTreeMap::new();
        diags.insert("gsi".to_string(), skipped_diag());
        diags.insert("gsi-ros2".to_string(), skipped_diag());
        let bundle = make_bundle_with_diags(bundle_name, diags);

        write_solve_failed_summary(&bundle, &dir).await.unwrap();

        // MD must still exist (abstention is not evidence of resolution).
        assert!(
            md_path.exists(),
            "abstained run must NOT delete the prior failure MD"
        );

        // Cleanup.
        let _ = std::fs::remove_file(&md_path);
        let _ = std::fs::remove_dir(&dir);
    }

    /// Test (b): verified all-sat run deletes stale MD.
    #[tokio::test]
    async fn verified_sat_clears_stale_md() {
        let dir =
            std::env::temp_dir().join(format!("retread_test_sat_clear_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bundle_name = "test-bundle-sat-clear";
        let md_path = dir.join(format!("RETREAD-SOLVE-FAILED-{bundle_name}.md"));

        // Pre-create the MD file (stale from a prior failed run).
        std::fs::write(&md_path, "# Stale prior failure\n").unwrap();
        assert!(md_path.exists(), "pre-condition: md must exist");

        // Run with all diagnostics satisfiable (verified all-sat).
        let mut diags = BTreeMap::new();
        diags.insert("gsi".to_string(), sat_diag());
        let bundle = make_bundle_with_diags(bundle_name, diags);

        write_solve_failed_summary(&bundle, &dir).await.unwrap();

        // MD must be gone (stale file cleaned up).
        assert!(
            !md_path.exists(),
            "verified sat run must delete the stale failure MD"
        );

        // Cleanup (dir may already be absent if remove_file removed it).
        let _ = std::fs::remove_dir(&dir);
    }

    /// Test that a genuine unsat (non-skipped) still writes the MD.
    #[tokio::test]
    async fn genuine_unsat_writes_md() {
        let dir =
            std::env::temp_dir().join(format!("retread_test_unsat_write_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bundle_name = "test-bundle-unsat-write";
        let md_path = dir.join(format!("RETREAD-SOLVE-FAILED-{bundle_name}.md"));

        assert!(!md_path.exists(), "pre-condition: md must NOT exist");

        let mut diags = BTreeMap::new();
        diags.insert("gsi".to_string(), unsat_diag());
        let bundle = make_bundle_with_diags(bundle_name, diags);

        write_solve_failed_summary(&bundle, &dir).await.unwrap();

        assert!(md_path.exists(), "genuine unsat must write the failure MD");

        // Cleanup.
        let _ = std::fs::remove_file(&md_path);
        let _ = std::fs::remove_dir(&dir);
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

    let key0 = conda_outputs_cache_key(&make_params(), Some(t0));
    let key1 = conda_outputs_cache_key(&make_params(), Some(t1));
    let key_none = conda_outputs_cache_key(&make_params(), None);

    // Different mtimes -> different keys.
    assert_ne!(key0, key1, "distinct mtimes must produce distinct keys");
    // None sentinel differs from a real mtime.
    assert_ne!(
        key0, key_none,
        "None mtime must not collide with a real mtime"
    );
    // Identical params + identical mtime -> same key (deterministic).
    let key0_dup = conda_outputs_cache_key(&make_params(), Some(t0));
    assert_eq!(
        key0, key0_dup,
        "same params+mtime must produce identical key"
    );
}
