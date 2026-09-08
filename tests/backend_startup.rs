#[test]
fn backend_startup_reports_package_version() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_pixi-build-retread"))
        .env("PIXI_BUILD_RETREAD_LOG", "info")
        .output()
        .expect("the backend binary must launch");

    assert!(
        output.status.success(),
        "backend exited with {}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("pixi-build-retread starting"), "{stderr}");
    assert!(stderr.contains(env!("CARGO_PKG_VERSION")), "{stderr}");
}

// ---------------------------------------------------------------------------
// SHIM-AUTO-2 / N27-RETREAD-65. The guards for the C34 job 6066294 arm-3 shape.
//
// WHAT HAPPENED. The arm ran, against `binsnaps/integration-5ebfbb6`,
//
//   pixi-build-retread path-source-manifest --workspace <ws> --out <ws>/pixi.toml
//                      --pack <ws>/pypi-packs/isaaclab-2.3x-pack
//                      --pack <ws>/pypi-packs/protomotions-deps-pack
//
// on a binary built BEFORE `path-source-manifest` existed. `argv[1]` matched no
// arm of the dispatch chain, fell through to the JSON-RPC transport, read a
// closed stdin, and returned 0. The driver's `rc != 0` check passed. Its `--out`
// file still held the CANONICAL manifest (md5 9711eb99…, 45298 B) where the
// EFFECTIVE one (md5 4ad488b9…, 45393 B) was required, and only that second
// assertion caught it.
//
// The two tests below are the two halves that were missing, and each fails on
// the binary that produced the incident:
//   * `unknown_verb_refuses_…` fails because that binary exits 0;
//   * `path_source_manifest_writes_…` fails because that binary leaves `--out`
//     byte-identical to the canonical input.
//
// THE BYTE-EXACT 4ad488b9… IS DELIBERATELY NOT ASSERTED HERE. It is a property
// of imprint-data's 45298-byte production manifest, which does not live in this
// repository; vendoring a copy would create a second authority for a file this
// crate does not own, and it would go stale the first time the manifest moved.
// What that md5 ENCODES is asserted instead — `--out` is written, it differs
// from the canonical input, every declared path source is repointed at its
// shim, the shim exists and carries its generated marker, and `--check` accepts
// exactly that file and refuses a different one. The byte-exact assertion stays
// with the caller that pins the production manifest: the C34 driver's
// `EXPECT_EFFECTIVE_MD5`, which was already correct and already reading `--out`.

fn scratch(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "retread-shimauto2-{label}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const PACE_REL: &str = "third_party/pace-sim2real/source/pace_sim2real";
const PACK_REL: &str = "pypi-packs/isaaclab-2.3x-pack";
const SHIM_REL: &str = "pypi-packs/isaaclab-2.3x-pack/sources/pace-sim2real";

/// A workspace in the production shape: a CANONICAL manifest whose
/// `[pypi-dependencies]` entry points at the REAL tree (not at the shim), a
/// pack directory, and the tree's own `*.egg-info/PKG-INFO`.
///
/// `records = true` also writes the hand-editable record file. It is a
/// PARAMETER and not a fixture constant because the two states are two
/// different production claims: `imprint-data` has NO record files anywhere
/// (`find imprint-data -maxdepth 4 -path '*path-sources*'` returns nothing),
/// so `false` is the shape the daily relock actually meets, and `true` is the
/// shape where a record exists and may only CONFIRM what the tree already says.
///
/// THE MANIFEST DECLARES THE PACK, and that line is load-bearing rather than
/// scenery: the derivation asks which pack a path source sits BESIDE, so a
/// workspace that names a pack only on the command line has not said what the
/// pack is for. The live manifest declares it the same way — `[feature.pace]`
/// carries `"isaaclab-2.3x-pack" = { path = "./pypi-packs/isaaclab-2.3x-pack" }`
/// next to its `pace_sim2real` entry.
fn canonical_workspace_opt(label: &str, records: bool) -> std::path::PathBuf {
    let root = scratch(label);
    std::fs::write(
        root.join("pixi.toml"),
        format!(
            "[workspace]\nname = \"ws\"\n\n# a prose mention of {PACE_REL} must NOT move\n\
             [dependencies]\n\
             \"isaaclab-2.3x-pack\" = {{ path = \"./{PACK_REL}\" }}\n\
             [pypi-dependencies]\n\
             pace_sim2real = {{ path = \"{PACE_REL}\", editable = true }}\n"
        ),
    )
    .unwrap();
    std::fs::create_dir_all(root.join(PACE_REL).join("pace_sim2real.egg-info")).unwrap();
    std::fs::write(
        root.join(PACE_REL)
            .join("pace_sim2real.egg-info")
            .join("PKG-INFO"),
        "Metadata-Version: 2.1\nName: pace_sim2real\nVersion: 0.1.2\n\
         Requires-Python: >=3.10\nRequires-Dist: psutil\nRequires-Dist: cmaes\n\nbody\n",
    )
    .unwrap();
    std::fs::create_dir_all(root.join(PACK_REL)).unwrap();
    std::fs::write(root.join(PACK_REL).join("pixi.toml"), "[package]\n").unwrap();
    if records {
        std::fs::create_dir_all(root.join(PACK_REL).join("path-sources")).unwrap();
        std::fs::write(
            root.join(PACK_REL).join("path-sources").join("pace-sim2real.toml"),
            format!(
                "# Path-source metadata record for `pace-sim2real`.\n\
                 path = \"{PACE_REL}\"\nversion = \"0.1.2\"\n\
                 requires-python = \">=3.10\"\ndependencies = [\"psutil\", \"cmaes\"]\n"
            ),
        )
        .unwrap();
    }
    root
}

fn canonical_workspace(label: &str) -> std::path::PathBuf {
    canonical_workspace_opt(label, true)
}

fn retread(args: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_pixi-build-retread"))
        .args(args)
        .output()
        .expect("the backend binary must launch")
}

#[test]
fn unknown_verb_refuses_instead_of_starting_the_transport() {
    // The C34 argv, verbatim in shape, with a verb this binary does not know.
    // A binary that answers 0 here is one that will report a capability it does
    // not have as a pass, which is what job 6066294 arm 3 measured.
    let root = canonical_workspace("unknown-verb");
    let out = root.join("pixi.toml");
    let before = std::fs::read(&out).unwrap();

    let output = retread(&[
        "path-source-manifest-that-does-not-exist",
        "--workspace",
        root.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--pack",
        root.join(PACK_REL).to_str().unwrap(),
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !output.status.success(),
        "an unhandled verb exited 0 -- this is the C34 6066294 arm-3 defect \
         (N27-RETREAD-65): the caller cannot tell a missing capability from a \
         successful one.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("unknown verb"),
        "the refusal must name itself so a caller can act on it: {stderr}"
    );
    assert!(
        !stderr.contains("pixi-build-retread starting"),
        "an unhandled verb must not reach the JSON-RPC transport at all: {stderr}"
    );
    assert_eq!(
        std::fs::read(&out).unwrap(),
        before,
        "a refused invocation must leave --out untouched"
    );
}

#[test]
fn no_argument_invocation_is_still_the_transport() {
    // The falsifier for the guard above: pixi launches this binary with NO
    // arguments, and that path must stay exactly as it was. A refusal broad
    // enough to catch the transport would be a worse defect than the one being
    // fixed. (`backend_startup_reports_package_version` asserts the same shape;
    // this one states WHY it may not be tightened.)
    let output = retread(&[]);
    assert!(
        output.status.success(),
        "the zero-argument invocation is the build-backend transport and must \
         keep working: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn path_source_manifest_writes_the_effective_manifest_to_out() {
    // The reader for the verb's writer, driven through the CLI the C34 driver
    // calls -- not through the library function, because the incident was that
    // the CLI arm did not exist while the library function did.
    let root = canonical_workspace("effective-out");
    let out = root.join("pixi.toml"); // the production shape: --out IS the canonical file
    let canonical = std::fs::read_to_string(&out).unwrap();

    let output = retread(&[
        "path-source-manifest",
        "--workspace",
        root.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--pack",
        root.join(PACK_REL).to_str().unwrap(),
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "path-source-manifest refused.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    let effective = std::fs::read_to_string(&out).unwrap();
    assert_ne!(
        effective, canonical,
        "--out still holds the CANONICAL manifest. This is the exact row C34 \
         6066294 arm 3 printed (md5 9711eb99... where 4ad488b9... was wanted): \
         the verb reported success and wrote nothing.\nstdout:\n{stdout}"
    );
    assert!(
        effective.contains(&format!("path = \"{SHIM_REL}\"")),
        "the declared path source was not repointed at its shim:\n{effective}"
    );
    assert!(
        !effective.contains(&format!("path = \"{PACE_REL}\"")),
        "a `path =` entry still names the real tree, so the PEP 517 metadata \
         build this transform exists to remove is still in the lock:\n{effective}"
    );
    assert!(
        effective.contains(&format!("# a prose mention of {PACE_REL} must NOT move")),
        "the rewrite is line-wise and key-scoped; it must not touch prose:\n{effective}"
    );

    let shim = root.join(SHIM_REL).join("pyproject.toml");
    let shim_text = std::fs::read_to_string(&shim)
        .unwrap_or_else(|e| panic!("no shim at {}: {e}", shim.display()));
    assert!(
        shim_text.contains("pixi-build-retread (retread-path-source-metadata)"),
        "the shim carries no generated marker, so nothing downstream can tell \
         it from a hand edit:\n{shim_text}"
    );

    // AND THE ASSERTION READS `--out`. `--check` is the same transform compared
    // against the file at `--out`, so these two calls prove the criterion the
    // C34 driver applies is aimed at the artefact the verb writes -- the
    // hypothesis (H3) that the driver was checking the wrong file, closed by a
    // test rather than by inspection.
    let recheck = retread(&[
        "path-source-manifest",
        "--workspace",
        root.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--pack",
        root.join(PACK_REL).to_str().unwrap(),
        "--check",
    ]);
    assert!(
        recheck.status.success(),
        "--check refused the file the verb had just written to --out: {}",
        String::from_utf8_lossy(&recheck.stderr)
    );

    let elsewhere = root.join("pixi.toml.canonical-copy");
    std::fs::write(&elsewhere, &canonical).unwrap();
    let mismatch = retread(&[
        "path-source-manifest",
        "--workspace",
        root.to_str().unwrap(),
        "--out",
        elsewhere.to_str().unwrap(),
        "--pack",
        root.join(PACK_REL).to_str().unwrap(),
        "--check",
    ]);
    assert_eq!(
        mismatch.status.code(),
        Some(4),
        "--check must exit 4 when --out is the canonical manifest rather than \
         the effective one -- that is the C34 condition, and a --check that \
         passed on it would be a guard that cannot fail: {}",
        String::from_utf8_lossy(&mismatch.stderr)
    );
}

#[test]
fn path_source_manifest_derives_its_records_with_no_record_file_on_disk() {
    // THE PRODUCTION SHAPE, END TO END, THROUGH THE REAL BINARY. `imprint-data`
    // has no `path-sources/` directory anywhere and never has, so the daily
    // relock's default `PACK_SHIM_MODE=on` refused at 2m26s (job 6073129) and
    // C35 arm 3's 138 s only ever passed because `c34_threearm.sh`'s
    // `pack_stage_content` MANUFACTURED the record into its own workspace from
    // a shell literal. This drives the same verb the daily template calls, over
    // a workspace with NOTHING under `path-sources/`, and requires it to
    // produce the same effective manifest.
    let root = canonical_workspace_opt("derive-no-record", false);
    assert!(
        !root.join(PACK_REL).join("path-sources").exists(),
        "this test is only meaningful with no record on disk"
    );
    let out = root.join("pixi.toml");
    let canonical = std::fs::read_to_string(&out).unwrap();

    let output = retread(&[
        "path-source-manifest",
        "--workspace",
        root.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--pack",
        root.join(PACK_REL).to_str().unwrap(),
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "the verb still requires a record file.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    // The derivation says so in its own row, and says it derived rather than
    // confirmed -- there is nothing on disk to confirm.
    assert!(
        stdout.contains("### PATH SOURCE RECORD project=pace-sim2real")
            && stdout.contains(&format!("source={PACE_REL}"))
            && stdout.contains("derived=yes"),
        "no derivation row, so nothing can read where the record came from:\n{stdout}"
    );

    let effective = std::fs::read_to_string(&out).unwrap();
    assert_ne!(effective, canonical, "the verb wrote nothing:\n{stdout}");
    assert!(effective.contains(&format!("path = \"{SHIM_REL}\"")), "{effective}");
    assert!(!effective.contains(&format!("path = \"{PACE_REL}\"")), "{effective}");

    // The shim is materialized from the DERIVED record and carries every field
    // uv needs statically -- the whole point of the transform.
    let shim_text =
        std::fs::read_to_string(root.join(SHIM_REL).join("pyproject.toml")).unwrap();
    for field in ["name", "version", "requires-python", "dependencies"] {
        assert!(shim_text.contains(field), "{field} missing:\n{shim_text}");
    }
    assert!(shim_text.contains("0.1.2"), "the tree's version:\n{shim_text}");
    assert!(shim_text.contains("psutil"), "the tree's deps:\n{shim_text}");

    // AND IT IS THE SAME DOCUMENT THE RECORD FILE PRODUCES, so the two
    // producers cannot drift: a record may confirm, never override.
    let with_record = canonical_workspace_opt("derive-with-record", true);
    let out2 = with_record.join("pixi.toml");
    let confirmed = retread(&[
        "path-source-manifest",
        "--workspace",
        with_record.to_str().unwrap(),
        "--out",
        out2.to_str().unwrap(),
        "--pack",
        with_record.join(PACK_REL).to_str().unwrap(),
    ]);
    assert!(confirmed.status.success(), "{}", String::from_utf8_lossy(&confirmed.stderr));
    assert!(
        String::from_utf8_lossy(&confirmed.stdout).contains("derived=confirmed"),
        "a record that agrees must report itself as a CONFIRMATION"
    );
    assert_eq!(
        std::fs::read_to_string(&out2).unwrap(),
        effective,
        "the derived and the confirmed effective manifests differ"
    );

    // A record that DISAGREES is a refusal naming both sides, never a silent
    // preference. Without this the confirmation above could be a no-op.
    std::fs::write(
        with_record.join(PACK_REL).join("path-sources").join("pace-sim2real.toml"),
        format!(
            "path = \"{PACE_REL}\"\nversion = \"9.9\"\n\
             requires-python = \">=3.10\"\ndependencies = [\"psutil\", \"cmaes\"]\n"
        ),
    )
    .unwrap();
    let lying = retread(&[
        "path-source-manifest",
        "--workspace",
        with_record.to_str().unwrap(),
        "--out",
        out2.to_str().unwrap(),
        "--pack",
        with_record.join(PACK_REL).to_str().unwrap(),
    ]);
    assert!(!lying.status.success(), "a lying record was accepted");
    let err = String::from_utf8_lossy(&lying.stderr);
    assert!(err.contains("9.9") && err.contains("0.1.2"), "both sides must be named: {err}");
}

// WHERE THESE RUN. The certifying gate (`tools/gate_build.sh`) runs
// `cargo test --lib` and only BUILDS `--all-targets`, so the three tests above
// are compiled but not executed by it. That is why the decision itself lives in
// `pixi_build_retread::cli_verbs`, whose guards ARE lib tests and ARE counted in
// the gate's split. These are the end-to-end half -- they drive the real binary
// over a real workspace -- and the lane that lands this runs them explicitly
// with `cargo test --test backend_startup` in the same job as the gate. A guard
// nobody executes is not a guard (law 3), and saying which job executes which
// half is the whole point of writing it down here.
