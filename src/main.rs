//! pixi-build-retread: a pixi build backend that repacks PyPI wheels as
//! conda packages with relaxed dependency pins.
//!
//! Speaks line-delimited JSON-RPC 2.0 over stdin/stdout, per the pixi build
//! protocol (`crates/pixi_build_types`, API version 4).

use std::ffi::OsStr;
use std::io::Write as _;
use std::path::PathBuf;

use pixi_build_retread::{fasttmp, handler, installer, rpc, solve};
use tracing_subscriber::EnvFilter;

fn main() -> anyhow::Result<()> {
    // `retread env-seed` -- handled FIRST, and the position is the whole
    // design. The production wrapper's job is to export the reproducible
    // interpreter hash seed BEFORE it launches pixi, and it asks this binary
    // what to export so the value has exactly one authority. If the verb sat
    // behind the RPC preflight it would refuse for want of the very variable
    // it exists to supply, and the wrapper could never bootstrap. It therefore
    // runs before panic hooks, Tokio, tracing and preflight, and touches
    // nothing: it prints a constant and exits.
    if let Some(code) = env_seed_command_exit_code() {
        std::process::exit(code);
    }
    pixi_build_retread::panic_hook::install_global_panic_hook();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| {
            let mut stderr = std::io::stderr().lock();
            let _ = writeln!(
                stderr,
                "retread: fatal: failed to build Tokio runtime: {error}"
            );
            let _ = stderr.flush();
            anyhow::anyhow!("failed to build Tokio runtime: {error}")
        })?;
    runtime.block_on(async_main())
}

/// `retread env-seed` prints the reproducible interpreter hash seed and exits.
///
/// Returns `None` when this is not an `env-seed` invocation, so `main` falls
/// through to its ordinary path. A malformed invocation (`env-seed` with any
/// further argument) exits 2 and prints nothing on stdout -- the wrapper reads
/// stdout with `$(...)`, so a diagnostic there would be exported AS the seed.
fn env_seed_command_exit_code() -> Option<i32> {
    let mut arguments = std::env::args_os();
    let _program = arguments.next();
    if arguments.next().as_deref()
        != Some(OsStr::new(pixi_build_retread::uv_closure::ENV_SEED_VERB))
    {
        return None;
    }
    if arguments.next().is_some() {
        eprintln!(
            "retread {}: takes no arguments",
            pixi_build_retread::uv_closure::ENV_SEED_VERB
        );
        return Some(2);
    }
    let mut stdout = std::io::stdout().lock();
    match stdout
        .write_all(pixi_build_retread::env_seed_verb_output().as_bytes())
        .and_then(|()| stdout.flush())
    {
        Ok(()) => Some(0),
        Err(error) => {
            eprintln!("retread env-seed: writing stdout: {error}");
            Some(1)
        }
    }
}

async fn async_main() -> anyhow::Result<()> {
    // v2.0.0 courier: `retread install --lock <path> [--prefix <p>]` is
    // invoked from the courier conda package's post-link script to install
    // the bundle's PyPI wheels into the active env. It is NOT the JSON-RPC
    // build-backend path -- handle it before the transport starts. Prefix
    // defaults to $PREFIX (set by conda post-link) then $CONDA_PREFIX.
    // `retread verify` is the cheap activate.d guard: marker + installed
    // distribution metadata only, no network or mutation.
    let argv: Vec<String> = std::env::args().collect();
    if matches!(argv.get(1).map(String::as_str), Some("fast")) {
        run_fast(&argv[2..])?;
        return Ok(());
    }

    // `retread preflight` — explicit environment check for callers (Slurm
    // scripts, CI) that want to fail at second zero instead of twenty minutes
    // into a staged run. Diagnostics go to stderr; stdout stays clean.
    if matches!(argv.get(1).map(String::as_str), Some("preflight")) {
        return match pixi_build_retread::uv_closure::preflight().await {
            Ok((bin, version)) => {
                eprintln!(
                    "preflight OK: uv {version} at {} (retread {})",
                    bin.display(),
                    env!("CARGO_PKG_VERSION")
                );
                Ok(())
            }
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(error.exit_code());
            }
        };
    }

    // `retread repodata-universe [--cache-root <dir>]` -- print the conda
    // candidate universe a lock would resolve against, WITHOUT solving,
    // fetching or refreshing anything.
    //
    // It exists so the harness has ONE implementation of the fingerprint. A
    // shell that folded `sha256sum` output its own way would be a second
    // implementation of a comparison rule, and the first time the two disagreed
    // the disagreement would read as a moved universe. This verb and the
    // backend's own rows call `repodata::universe_digest_of`, so they cannot.
    if matches!(argv.get(1).map(String::as_str), Some("repodata-universe")) {
        return run_repodata_universe(&argv[2..]);
    }

    // `retread sharded-universe [--repodata-dir <dir>]` -- fingerprint the
    // SHARDED repodata indexes the pixi SOLVER read (N27-RETREAD-81).
    //
    // WHY THIS IS A VERB AND NOT A BACKEND ROW. The backend never sees these
    // documents: pixi's frontend fetches the sharded index through
    // `rattler_repodata_gateway`, solves against it, and only then spawns us.
    // There is no moment inside this process at which the sharded index has
    // been read, so a `tracing` row in `repodata.rs` would be a row about a
    // file this process never opened. The honest producer runs AFTER the lock,
    // from the harness, over the directory the solve actually used -- exactly
    // the shape `repodata-universe` already has.
    //
    // AND IT IS A SECOND UNIVERSE, NOT A WIDER ONE. `repodata_universe=` keeps
    // covering precisely the classic documents it always covered, because the
    // built-output store's adoption rule is keyed on that number and widening
    // it would make every published record unadoptable in one commit.
    if matches!(argv.get(1).map(String::as_str), Some("sharded-universe")) {
        return run_sharded_universe(&argv[2..]);
    }

    if matches!(argv.get(1).map(String::as_str), Some("migrate-overrides")) {
        return run_migrate_overrides(&argv[2..]);
    }

    // `retread store-reap
    //      [--store built-wheels|git-snapshots|shadow|build-requirements|hermetic-envs|sdist-metadata|built-outputs|path-source-metadata|route-probe-verdicts|all]
    //      [--root <dir>]... [--dry-run|--apply]
    //      [--max-age-days <n>] [--route-probe-max-age-days <n>] [--bytes]`
    //
    // STORE-REAP-2, and it exists because of law 2. The persistent-store
    // reapers had no production caller that could ever reach a persistent
    // store: the only call site is the backend's `initialize`, and every
    // relock this harness runs job-scopes `XDG_CACHE_HOME`, so the root that
    // resolves is empty by construction and deleted by the job's cleanup.
    // This is the reap-only invocation `p6v_lock_reap.sh` had the shape of.
    //
    // IT IS A DRY RUN UNLESS `--apply` IS TYPED. A dry run walks and decides
    // with the SAME code -- `courier::ReapMode` is a parameter of the reapers,
    // not a second walk -- and then creates nothing and renames nothing.
    if matches!(argv.get(1).map(String::as_str), Some("store-reap")) {
        let args = pixi_build_retread::store_reap::parse_args(&argv[2..])?;
        let code = pixi_build_retread::store_reap::run(&args)?;
        std::process::exit(code);
    }

    // `retread sdist-meta-key (--revision-http <file> | --probe)
    //      --uv-version <s> --python-tag <s> --backend <s> --pythonhashseed <s>
    //      [--store-root <dir>]`
    //
    // SDIST-META-3 REPLACED `--sdist-sha256` WITH `--revision-http`, and the
    // reason is a measurement rather than a preference: the sdist sha256 the
    // old flag carried IS NOT IN `revision.http` any more. `od -c` on the same
    // shard in the same tree shows `91 92 a6 "Sha256" d9 40 <64 hex>` on a file
    // written 2026-09-04 and the EMPTY array `90` on one written 2026-09-07;
    // job 6043382 arm 1 refused all six sdists with
    // `no-sha256-in-revision.http` and admitted nothing. The URL and the ETag
    // are byte-identical across both shapes, so the key is built on those and
    // the verb reads the file itself. `--probe` is the seeder's one call to
    // learn where the store is before it holds an entry.
    //
    // SDIST-META-2, and it exists for the same reason `repodata-universe`
    // does: the prepared-sdist-metadata store has TWO halves in the harness --
    // the post-lock harvester that writes an entry and the scoper's seeder that
    // reads one -- and if each folded its own `sha256sum` the two would
    // eventually disagree, at which point the store would read as permanently
    // COLD rather than as broken. This verb is the one derivation both call,
    // and it prints the entry path as well as the key so neither half joins the
    // store's path segments itself either.
    //
    // Every input is an ARGUMENT. The seeder derives the key for the arm it is
    // about to scope, standing outside that build, so reading the current
    // process's environment would be the wrong answer as well as the wrong
    // shape.
    //
    // SDM-PYTAG-1 (N27-RETREAD-24) MADE `python_tag` MEASURABLE. It was the one
    // key field with no producer: declared on argv and then carried, which
    // MERGE-B32 ran over the canonical 27-environment workspace as a single
    // `cp310`. That lock resolves FOUR interpreters (cp312 x16, cp311 x7,
    // cp310 x5, cp38 x1), so the carried value was wrong for 22 of 27 and would
    // have admitted cp312-built metadata under a cp310 key. `--python-tag` is
    // still accepted for a single-environment caller; `--python-tag-from-lock
    // <pixi.lock> --env <name>` derives it, and passing BOTH is refused as a
    // ROW plus `CARRIED_TAG_EXIT` so a wrapper can tell a stale call site from
    // a typo.
    if matches!(argv.get(1).map(String::as_str), Some("sdist-meta-key")) {
        let code = pixi_build_retread::sdist_metadata::key_main(&argv[2..])?;
        std::process::exit(code);
    }

    // `retread sdist-meta-python-tags (--lock <pixi.lock> [--subdir <s>]
    //      [--env <name>] | --prefix <dir> --env <name>)`
    //
    // THE PRODUCER of the fifth key field, printing one
    // `### SDIST_META_KEY env=<name> python_tag=<cpXY> source=<lock|interpreter>`
    // row per environment plus a `TOTAL … distinct_tags=<n>` summary. The
    // harvester and the seeder both read those rows instead of carrying a
    // literal; `distinct_tags` is what tells a single-overlay seeder that one
    // key cannot serve every environment (N27-RETREAD-25's wiring).
    if matches!(
        argv.get(1).map(String::as_str),
        Some("sdist-meta-python-tags")
    ) {
        let args = pixi_build_retread::sdist_metadata::parse_tag_args(&argv[2..])?;
        let code = pixi_build_retread::sdist_metadata::run_tags(&args)?;
        std::process::exit(code);
    }

    // `retread path-source-refresh` -- the operator-facing half of
    // `retread-path-source-metadata`. The pack's `path-sources/<project>.toml`
    // record is the SOURCE OF TRUTH, and every backend initialize refuses when
    // it disagrees with the tree. This verb is how the operator adopts the
    // tree's current facts: it prints the per-field diff, and `--write`
    // rewrites the record so the change lands as one reviewable diff. It never
    // touches the tree and never touches the generated shim.
    if matches!(argv.get(1).map(String::as_str), Some("path-source-refresh")) {
        return run_path_source_refresh(&argv[2..]);
    }

    // `retread path-source-manifest` -- the pre-lock half of
    // `retread-path-source-metadata`, and the reason the canonical manifest
    // never has to carry the repoint. It reads the canonical `pixi.toml`,
    // materializes each pack's shims, and writes an EFFECTIVE manifest whose
    // path-source entries point at those shims. The lock driver stages that
    // file into the disposable workspace it already builds. Nothing is written
    // to the canonical tree, and `--no-pack-shims` reproduces its bytes.
    if matches!(
        argv.get(1).map(String::as_str),
        Some("path-source-manifest")
    ) {
        return run_path_source_manifest(&argv[2..]);
    }

    if matches!(
        argv.get(1).map(String::as_str),
        Some("install" | "verify" | "solve" | "lock")
    ) {
        let cmd = argv[1].as_str();
        if cmd == "solve" {
            let args = solve::args::parse(&argv[2..])?;
            let code = solve::run(args).await?;
            std::process::exit(code);
        }
        if cmd == "lock" {
            let args = solve::lock::args::parse(&argv[2..])?;
            let code = solve::lock::run(args).await?;
            std::process::exit(code);
        }
        let mut lock: Option<String> = None;
        let mut prefix: Option<String> = None;
        let mut full = false;
        let mut it = argv[2..].iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--lock" => lock = it.next().cloned(),
                "--prefix" => prefix = it.next().cloned(),
                "--full" if cmd == "verify" => full = true,
                other => anyhow::bail!("retread {cmd}: unknown arg {other}"),
            }
        }
        let lock = lock.ok_or_else(|| anyhow::anyhow!("retread {cmd}: --lock <path> required"))?;
        let prefix = prefix
            .or_else(|| std::env::var("PREFIX").ok())
            .or_else(|| std::env::var("CONDA_PREFIX").ok())
            .ok_or_else(|| {
                anyhow::anyhow!("retread {cmd}: --prefix <p> or $PREFIX/$CONDA_PREFIX required")
            })?;
        let lock = std::path::Path::new(&lock);
        let prefix = std::path::Path::new(&prefix);
        return match cmd {
            "install" => installer::run(lock, prefix).await,
            "verify" => installer::verify(lock, prefix, full),
            _ => unreachable!("matched above"),
        };
    }

    // AN UNHANDLED VERB IS A REFUSAL, NOT A SILENT BACKEND START
    // (N27-RETREAD-65, SHIM-AUTO-2). Every arm above `return`s or
    // `std::process::exit`s, so an `argv[1]` that reaches this line was
    // dispatched by NOBODY, and until now it fell through to the transport
    // below and exited 0 on a closed stdin -- which made C34 job 6066294 arm 3
    // read a verb its binsnap did not carry as a pass. The condition is
    // STRUCTURAL (fall-through), not a second copy of the dispatch table; the
    // decision and its message are `cli_verbs`, which is a library module so
    // that `cargo test --lib` -- the certifying gate -- actually runs the
    // guards for it.
    if let Some(verb) = pixi_build_retread::cli_verbs::verb_candidate(&argv) {
        anyhow::bail!(pixi_build_retread::cli_verbs::unknown_verb_message(
            verb,
            env!("CARGO_PKG_VERSION")
        ));
    }

    // AUTOMATIC PREFLIGHT. Runs on every RPC invocation before the transport
    // starts, so a misconfigured uv fails here in milliseconds instead of
    // surfacing twenty minutes into a staged build. Diagnostics go to stderr;
    // stdout is the JSON-RPC channel and MUST stay clean.
    if let Err(error) = pixi_build_retread::uv_closure::preflight().await {
        eprintln!("{error}");
        std::process::exit(error.exit_code());
    }

    // Log to stderr only — stdout is reserved for the JSON-RPC transport.
    // Per-bundle probe + routing decisions ALSO land on disk as part of
    // the audit JSON (retread-audit-<bundle>.json next to the pack's
    // pixi.toml). That audit is what to read when pixi swallows stderr
    // and you can't see this stream.
    // Filter: PIXI_BUILD_RETREAD_LOG (NOT RUST_LOG -- common gotcha).
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_env("PIXI_BUILD_RETREAD_LOG")
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        "pixi-build-retread starting"
    );

    let handler = handler::Handler::new();
    let served = rpc::serve(move |method, params| {
        let handler = handler.clone();
        async move { handler.dispatch(method, params).await }
    })
    .await;
    // N27-RETREAD-160. The run-total `### ROUTE PROBE STORE` row. `serve`
    // returns when the frontend closes stdin, which is the last moment this
    // process is alive and the first at which every per-bundle handle has
    // dropped its counters into the totals. A no-op unless the shared store
    // was configured, so the default regime prints nothing new.
    pixi_build_retread::route_probe_cache::emit_run_total();
    served
}

/// `retread migrate-overrides --workspace <dir> --pack <pack pixi.toml>`:
/// fix #22 one-shot. Moves fix #20-era auto-written `# retread:override`
/// entries out of a pack's pixi.toml and into the workspace's
/// `.retread/auto-overrides.json` ledger, leaving any genuinely manual
/// (un-sentineled) `retread-overrides` entries untouched. Safe to re-run
/// (no-op once migrated).
/// `retread path-source-refresh --pack <dir> --workspace <dir> [--project <p>] [--write]`
///
/// Reads each `<pack>/path-sources/<project>.toml`, reads what the real tree
/// says about itself, and reports every field that moved. Exit 0 = the records
/// agree with the trees. Exit 3 = drift, and (without `--write`) nothing was
/// changed. `--write` rewrites the drifted records from the tree.
fn run_path_source_refresh(args: &[String]) -> anyhow::Result<()> {
    use pixi_build_retread::path_source_metadata as psm;

    let mut pack: Option<PathBuf> = None;
    let mut workspace: Option<PathBuf> = None;
    let mut only: Option<String> = None;
    let mut records_dir = psm::RECORDS_DIR_DEFAULT.to_string();
    let mut write = false;
    let mut shims = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--pack" => {
                pack = Some(PathBuf::from(it.next().ok_or_else(|| {
                    anyhow::anyhow!("path-source-refresh: --pack <dir> requires a value")
                })?));
            }
            "--workspace" => {
                workspace = Some(PathBuf::from(it.next().ok_or_else(|| {
                    anyhow::anyhow!("path-source-refresh: --workspace <dir> requires a value")
                })?));
            }
            "--project" => {
                only = Some(
                    it.next()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "path-source-refresh: --project <name> requires a value"
                            )
                        })?
                        .clone(),
                );
            }
            "--records-dir" => {
                records_dir = it
                    .next()
                    .ok_or_else(|| {
                        anyhow::anyhow!("path-source-refresh: --records-dir <rel> requires a value")
                    })?
                    .clone();
            }
            "--write" => write = true,
            "--shims" => shims = true,
            other => anyhow::bail!("path-source-refresh: unknown arg {other}"),
        }
    }
    let pack = pack
        .ok_or_else(|| anyhow::anyhow!("path-source-refresh: --pack <pack directory> required"))?;
    let workspace = workspace
        .ok_or_else(|| anyhow::anyhow!("path-source-refresh: --workspace <dir> required"))?;

    let records = psm::load_records(&pack, &records_dir)?;
    if records.is_empty() {
        anyhow::bail!(
            "path-source-refresh: {}/{} holds no <project>.toml record",
            pack.display(),
            records_dir
        );
    }
    let mut drifted = 0usize;
    let mut checked = 0usize;
    for record in &records {
        if let Some(only) = only.as_deref()
            && only != record.project
        {
            continue;
        }
        checked += 1;
        let real = workspace.join(&record.entry.path);
        if !real.is_dir() {
            anyhow::bail!(
                "path-source-refresh: {} names path = \"{}\", which is not a \
                 directory under {}",
                record.file.display(),
                record.entry.path,
                workspace.display()
            );
        }
        let facts = psm::tree_facts(&real)?;
        match psm::check_drift(record, &facts) {
            Ok(()) => println!(
                "path-source-refresh: {} agrees with {} (version {})",
                record.file.display(),
                real.display(),
                record.entry.version
            ),
            Err(error) => {
                drifted += 1;
                println!("path-source-refresh: DRIFT {}\n  {error:#}", record.project);
                let refreshed = psm::record_from_tree(&record.project, &record.entry, &facts)?;
                let before = psm::render_record(&record.project, &record.entry);
                let after = psm::render_record(&record.project, &refreshed);
                for (b, a) in before.lines().zip(after.lines()) {
                    if b != a {
                        println!("  - {b}");
                        println!("  + {a}");
                    }
                }
                if write {
                    std::fs::write(&record.file, &after)?;
                    println!("  WROTE {}", record.file.display());
                }
            }
        }
    }
    if checked == 0 {
        anyhow::bail!(
            "path-source-refresh: no record matched --project {}",
            only.unwrap_or_default()
        );
    }
    // `--shims` is how a developer materializes the pack content they COMMIT.
    // It runs the SAME writer `Handler::initialize` runs, so a committed shim
    // and a regenerated one are the same bytes and initialize is a no-op.
    if shims {
        if drifted > 0 && !write {
            anyhow::bail!(
                "path-source-refresh: refusing to generate shims from {drifted} \
                 drifted record(s). Fix the records first (--write adopts the \
                 tree's facts)."
            );
        }
        for outcome in psm::generate_shims(&pack, &workspace, &records_dir, only.as_deref())? {
            println!(
                "path-source-refresh: shim {} {}",
                outcome.verb(),
                outcome.shim().display()
            );
        }
    }
    if drifted > 0 && !write {
        eprintln!(
            "path-source-refresh: {drifted} record(s) disagree with their trees and \
             nothing was written. Re-run with --write to adopt the tree's facts, or \
             edit the record(s) by hand."
        );
        std::process::exit(3);
    }
    Ok(())
}

/// `retread path-source-manifest --workspace <dir> --pack <dir> [--pack <dir>…]
///  --out <file> [--records-dir <rel>] [--no-pack-shims] [--check]`
///
/// Emits the EFFECTIVE manifest: the canonical `<workspace>/pixi.toml` with
/// every declared path source repointed at the pack shim that carries its
/// static PEP 621 metadata. The canonical file is read and never written.
///
/// Exit 0 = `--out` holds the effective manifest (and, without
/// `--no-pack-shims`, every shim is materialized). Exit 4 (`--check` only) =
/// the file at `--out` is not what the transform produces.
fn run_path_source_manifest(args: &[String]) -> anyhow::Result<()> {
    use pixi_build_retread::path_source_metadata as psm;

    let mut packs: Vec<PathBuf> = Vec::new();
    let mut workspace: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut records_dir = psm::RECORDS_DIR_DEFAULT.to_string();
    let mut shims = true;
    let mut derive_editable = true;
    let mut generated_records = true;
    let mut check = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--pack" => packs.push(PathBuf::from(it.next().ok_or_else(|| {
                anyhow::anyhow!("path-source-manifest: --pack <dir> requires a value")
            })?)),
            "--workspace" => {
                workspace = Some(PathBuf::from(it.next().ok_or_else(|| {
                    anyhow::anyhow!("path-source-manifest: --workspace <dir> requires a value")
                })?));
            }
            "--out" => {
                out = Some(PathBuf::from(it.next().ok_or_else(|| {
                    anyhow::anyhow!("path-source-manifest: --out <file> requires a value")
                })?));
            }
            "--records-dir" => {
                records_dir = it
                    .next()
                    .ok_or_else(|| {
                        anyhow::anyhow!("path-source-manifest: --records-dir <rel> requires a value")
                    })?
                    .clone();
            }
            // The opt-out. It is a real control, not a comment: it reproduces
            // the canonical bytes, so a lock driver can bisect the transform
            // itself without editing anything.
            "--no-pack-shims" => shims = false,
            // METAGEN-1's opt-out, the CLI half of
            // `retread-derive-editable-metadata = false`. With it, a source
            // that states its dependencies NOWHERE derives none instead of
            // building them -- which is what the pre-METAGEN-1 verb did, and
            // what the byte-identity guard drives.
            "--no-derive-editable-metadata" => derive_editable = false,
            // METAGEN-1. The record half: with this, the verb derives the
            // metadata but writes NO record into the pack. It exists so the
            // staged-first workflow has a real control -- a run that shows
            // what WOULD be written without writing it.
            "--no-generated-records" => generated_records = false,
            "--check" => check = true,
            other => anyhow::bail!("path-source-manifest: unknown arg {other}"),
        }
    }
    let workspace = workspace
        .ok_or_else(|| anyhow::anyhow!("path-source-manifest: --workspace <dir> required"))?;
    let out = out.ok_or_else(|| anyhow::anyhow!("path-source-manifest: --out <file> required"))?;
    let canonical = workspace.join("pixi.toml");
    let manifest_text = std::fs::read_to_string(&canonical)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", canonical.display()))?;

    if shims && packs.is_empty() {
        anyhow::bail!(
            "path-source-manifest: at least one --pack <dir> is required (or \
             pass --no-pack-shims to copy the canonical manifest through)"
        );
    }
    // METAGEN-1. The verb builds the same last-resort reader the backend does,
    // from the same constructor, so the two producers of one capability cannot
    // drift -- the defect SHIM-AUTO-3's comment in `path_source_metadata`
    // records having already happened once here.
    let deriver = pixi_build_retread::derived_editable_metadata::ConfiguredDeriver::from_config(
        Some(derive_editable),
        None,
        &|key| std::env::var(key).ok(),
        std::env::temp_dir().join(format!("retread-metagen-{}", std::process::id())),
    );
    let dynamic = deriver
        .as_ref()
        .map(|d| d as &dyn psm::DynamicDependencySource);
    let (text, rewrites, outcomes, derived, metadata_rows) = psm::effective_manifest_text(
        &packs,
        &workspace,
        &records_dir,
        &manifest_text,
        shims,
        dynamic,
        generated_records,
    )?;
    // THE DERIVED-METADATA ROWS, before the records that carry them. One per
    // source whose dependencies had to be BUILT because the tree states them
    // nowhere; a run that prints none is a run in which every source stated its
    // own facts, which is the fast and common case.
    for row in &metadata_rows {
        println!("{row}");
    }
    // WHERE EACH RECORD CAME FROM, before anything that uses it. Nothing under
    // `<pack>/path-sources/` has to exist: the record is derived from the
    // manifest entry and the tree's own metadata, and a file that IS there can
    // only confirm it (`derived=confirmed`). A run whose rows all say
    // `derived=yes` wrote nothing into the workspace to get there.
    for record in &derived {
        println!("{}", record.row());
    }
    if !shims {
        println!(
            "### PACK SHIM disabled (--no-pack-shims): {} copied unchanged, the \
             frontend will run a PEP 517 metadata build for every path source",
            canonical.display()
        );
    }
    for rewrite in &rewrites {
        println!("{}", rewrite.row());
    }
    for outcome in &outcomes {
        println!(
            "### PACK SHIM FILE {} {}",
            outcome.verb(),
            outcome.shim().display()
        );
    }

    if check {
        // The derived-vs-canonical diff, line for line. It is printed BEFORE
        // the verdict because it is the thing an auditor reads: a transform
        // that moved a line nobody expected is a defect even when `--check`
        // is otherwise satisfied, and a check that only says match/no-match
        // cannot show that.
        let canon: Vec<&str> = manifest_text.lines().collect();
        let eff: Vec<&str> = text.lines().collect();
        if canon.len() == eff.len() {
            for (n, (a, b)) in canon.iter().zip(eff.iter()).enumerate() {
                if a != b {
                    println!("### PACK SHIM DIFF line={} -{}", n + 1, a);
                    println!("### PACK SHIM DIFF line={} +{}", n + 1, b);
                }
            }
        } else {
            println!(
                "### PACK SHIM DIFF line-count moved {} -> {} (the transform is \
                 line-preserving, so this is a defect)",
                canon.len(),
                eff.len()
            );
        }
        let found = std::fs::read_to_string(&out).unwrap_or_default();
        if found == text {
            println!("### PACK SHIM CHECK {} matches", out.display());
            return Ok(());
        }
        eprintln!(
            "path-source-manifest: {} is not the manifest this transform produces \
             ({} bytes on disk vs {} generated). Re-run without --check.",
            out.display(),
            found.len(),
            text.len()
        );
        std::process::exit(4);
    }

    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("creating {}: {e}", parent.display()))?;
    }
    // Sibling temp + rename: the effective manifest is routinely written into a
    // workspace that was `cp -al`'d out of a shared mirror, where a write in
    // place would reach every other link.
    let tmp = out.with_extension("retread-tmp");
    std::fs::write(&tmp, text.as_bytes())
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &out)
        .map_err(|e| anyhow::anyhow!("renaming {} -> {}: {e}", tmp.display(), out.display()))?;
    println!(
        "### PACK SHIM MANIFEST {} ({} bytes) from {}",
        out.display(),
        std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0),
        canonical.display()
    );
    Ok(())
}

fn run_migrate_overrides(args: &[String]) -> anyhow::Result<()> {
    let mut workspace: Option<PathBuf> = None;
    let mut pack: Option<PathBuf> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--workspace" => {
                workspace = Some(PathBuf::from(it.next().ok_or_else(|| {
                    anyhow::anyhow!("retread migrate-overrides: --workspace <dir> requires a value")
                })?));
            }
            "--pack" => {
                pack = Some(PathBuf::from(it.next().ok_or_else(|| {
                    anyhow::anyhow!(
                        "retread migrate-overrides: --pack <pixi.toml> requires a value"
                    )
                })?));
            }
            other => anyhow::bail!("retread migrate-overrides: unknown arg {other}"),
        }
    }
    let workspace = workspace
        .ok_or_else(|| anyhow::anyhow!("retread migrate-overrides: --workspace <dir> required"))?;
    let pack = pack.ok_or_else(|| {
        anyhow::anyhow!("retread migrate-overrides: --pack <pack pixi.toml> required")
    })?;
    if !pack.is_file() {
        anyhow::bail!(
            "retread migrate-overrides: pack manifest {} not found",
            pack.display()
        );
    }
    let migrated =
        pixi_build_retread::pack_overrides::migrate_pack_toml_entries(&workspace, &pack)?;
    if migrated.is_empty() {
        println!(
            "retread migrate-overrides: no sentineled auto overrides found in {} (already migrated, or none ever written)",
            pack.display()
        );
    } else {
        println!(
            "retread migrate-overrides: moved {} override(s) from {} into {}: {}",
            migrated.len(),
            pack.display(),
            pixi_build_retread::pack_overrides::ledger_path(&workspace).display(),
            migrated.join(", "),
        );
    }
    Ok(())
}

struct FastCli {
    workspace: Option<PathBuf>,
    print_env: bool,
    preflight_locks: Option<PathBuf>,
    preflight_lock_helper: Option<PathBuf>,
    persist: Option<String>,
    cmd: Vec<String>,
}

fn run_fast(args: &[String]) -> anyhow::Result<()> {
    let parsed = parse_fast_args(args)?;
    if let Some(path) = parsed.preflight_lock_helper {
        return fasttmp::preflight_lock_helper(&path);
    }
    if let Some(path) = parsed.preflight_locks {
        return fasttmp::preflight_locks(&path);
    }
    if parsed.persist.is_some() {
        anyhow::bail!(
            "retread fast --persist is disabled: Pixi environments embed their job-local detached prefix and cannot be safely restored into another job root; use the shared package cache plus `pixi install --frozen`"
        );
    }

    let workspace = match parsed.workspace {
        Some(dir) => fasttmp::find_workspace_root(&dir)?,
        None => fasttmp::find_workspace_root(&std::env::current_dir()?)?,
    };
    let cfg = fasttmp::FastTmpConfig::load(&workspace);
    let engaged = fasttmp::engage(&workspace, &cfg)?;

    if parsed.print_env {
        if let Some(engaged) = engaged.as_ref() {
            print!("{}", fasttmp::shell_exports(engaged));
        } else {
            // `--print-env` is commonly sourced. If it inherited a retread
            // overlay from an older SLURM job, emit cleanup commands so Pixi
            // is not left pointing at a dead job-local config/cache path.
            print!("{}", fasttmp::shell_stale_cleanup());
            eprintln!(
                "retread fast-tmp: disengaged for {} (mode off or filesystem not slow)",
                workspace.display()
            );
        }
        return Ok(());
    }
    if parsed.cmd.is_empty() {
        anyhow::bail!(
            "retread fast: expected command after `--` (or use --print-env / --preflight-locks)"
        );
    }

    match engaged.as_ref() {
        Some(engaged) => {
            fasttmp::check_env_eviction(&workspace, &engaged.ns);
            fasttmp::print_mapping(engaged);
            exec_fast_command(&parsed.cmd, Some(&engaged.env), false)
        }
        None => {
            eprintln!(
                "retread fast-tmp: disengaged for {} (mode off or filesystem not slow)",
                workspace.display()
            );
            exec_fast_command(
                &parsed.cmd,
                None,
                fasttmp::inherited_fasttmp_cleanup_needed(),
            )
        }
    }
}

fn parse_fast_args(args: &[String]) -> anyhow::Result<FastCli> {
    let mut out = FastCli {
        workspace: None,
        print_env: false,
        preflight_locks: None,
        preflight_lock_helper: None,
        persist: None,
        cmd: Vec::new(),
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--" => {
                out.cmd = args[i + 1..].to_vec();
                break;
            }
            "--workspace" => {
                i += 1;
                let Some(value) = args.get(i) else {
                    anyhow::bail!("retread fast: --workspace requires a directory");
                };
                out.workspace = Some(PathBuf::from(value.as_str()));
            }
            "--print-env" => out.print_env = true,
            "--persist" => {
                // Retain parsing solely to produce a targeted compatibility
                // error; snapshots are unsafe until Pixi envs are relocatable.
                match args.get(i + 1) {
                    Some(value) if !value.starts_with('-') => {
                        i += 1;
                        out.persist = Some(value.clone());
                    }
                    _ => out.persist = Some("all".to_string()),
                }
            }
            "--preflight-locks" => {
                i += 1;
                let Some(value) = args.get(i) else {
                    anyhow::bail!("retread fast: --preflight-locks requires a shared path");
                };
                out.preflight_locks = Some(PathBuf::from(value.as_str()));
            }
            "--preflight-lock-helper" => {
                i += 1;
                let Some(value) = args.get(i) else {
                    anyhow::bail!("retread fast: --preflight-lock-helper requires a lock path");
                };
                out.preflight_lock_helper = Some(PathBuf::from(value.as_str()));
            }
            other if other.starts_with("--") => {
                anyhow::bail!("retread fast: unknown arg {other}");
            }
            _ => {
                out.cmd = args[i..].to_vec();
                break;
            }
        }
        i += 1;
    }
    Ok(out)
}

#[cfg(unix)]
fn exec_fast_command(
    cmd: &[String],
    env: Option<&[(String, String)]>,
    remove_stale_fast_env: bool,
) -> anyhow::Result<()> {
    use std::os::unix::process::CommandExt;

    let mut command = std::process::Command::new(&cmd[0]);
    command.args(&cmd[1..]);
    if remove_stale_fast_env {
        fasttmp::remove_stale_fast_env_from_command(&mut command);
    }
    if let Some(env) = env {
        command.envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    }
    let err = command.exec();
    Err(anyhow::anyhow!(
        "retread fast: exec {} failed: {err}",
        cmd[0]
    ))
}

#[cfg(not(unix))]
fn exec_fast_command(
    cmd: &[String],
    env: Option<&[(String, String)]>,
    remove_stale_fast_env: bool,
) -> anyhow::Result<()> {
    let mut command = std::process::Command::new(&cmd[0]);
    command.args(&cmd[1..]);
    if remove_stale_fast_env {
        fasttmp::remove_stale_fast_env_from_command(&mut command);
    }
    if let Some(env) = env {
        command.envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    }
    let status = command.status()?;
    std::process::exit(status.code().unwrap_or(1));
}

/// `retread repodata-universe [--cache-root <dir>]`.
///
/// `--cache-root` defaults to `$RATTLER_CACHE_DIR`, i.e. exactly what
/// `repodata::cache_root_from` resolves for the backend, so running the verb
/// with the harness's own environment reads the harness's own snapshot.
/// Diagnostics to stderr; the one summary line to stdout so a job header can
/// capture it.
fn run_repodata_universe(args: &[String]) -> anyhow::Result<()> {
    let mut cache_root: Option<PathBuf> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--cache-root" => {
                cache_root = Some(PathBuf::from(it.next().ok_or_else(|| {
                    anyhow::anyhow!("retread repodata-universe: --cache-root needs a path")
                })?));
            }
            other => anyhow::bail!("retread repodata-universe: unknown arg {other}"),
        }
    }
    let cache_root = match cache_root {
        Some(root) => root,
        None => std::env::var_os("RATTLER_CACHE_DIR")
            .map(PathBuf::from)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "retread repodata-universe: pass --cache-root <dir> or set RATTLER_CACHE_DIR"
                )
            })?,
    };
    let documents = pixi_build_retread::repodata::universe_from_cache_root(&cache_root)?;
    if documents.is_empty() {
        // A snapshot with no documents is not a universe, and a header that
        // printed a digest for it would be stating a fact nobody has.
        anyhow::bail!(
            "retread repodata-universe: no repodata documents under {}/retread-repodata",
            cache_root.display()
        );
    }
    for document in &documents {
        eprintln!(
            "repodata {}  sha256={} bytes={}",
            document.label(),
            document.sha256,
            document.bytes
        );
    }
    println!(
        "{}",
        pixi_build_retread::repodata::universe_summary_line(&documents)
    );
    Ok(())
}

/// `retread sharded-universe [--repodata-dir <dir>]`.
///
/// N27-RETREAD-81. Fingerprints the SHARDED repodata indexes under a pixi
/// `repodata/` directory -- the documents the SOLVER read, and the only place
/// `run_exports` ever comes from.
///
/// `--repodata-dir` defaults to `$PIXI_CACHE_DIR/repodata`, which is what
/// `fasttmp` points the job's pixi at, so running the verb with the relock's
/// own environment reads the relock's own indexes.
///
/// THE PER-INDEX ROWS GO TO STDOUT, NOT STDERR, and that is a deliberate
/// difference from `repodata-universe`. That verb's per-document lines are
/// diagnostics beside one summary line; these rows ARE the evidence a
/// comparator reads back, and evidence a harness has to merge two streams to
/// find is evidence that goes missing.
///
/// AN EMPTY DIRECTORY IS NOT AN ERROR HERE, which is the other difference.
/// `repodata-universe` bails on an empty snapshot because a classic universe
/// with no documents means the backend consulted nothing and any digest printed
/// for it would be a fact nobody has. The sharded case is not symmetric: a lock
/// solved entirely from pixi's own `.solv`/`pkgs` caches legitimately touches
/// no shard index, and refusing there would turn a real state into a failed
/// job. It prints `indexes=0` instead, which a comparator reads as a universe
/// and an operator reads as loud.
fn run_sharded_universe(args: &[String]) -> anyhow::Result<()> {
    let mut repodata_dir: Option<PathBuf> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--repodata-dir" => {
                repodata_dir = Some(PathBuf::from(it.next().ok_or_else(|| {
                    anyhow::anyhow!("retread sharded-universe: --repodata-dir needs a path")
                })?));
            }
            other => anyhow::bail!("retread sharded-universe: unknown arg {other}"),
        }
    }
    let repodata_dir = match repodata_dir {
        Some(dir) => dir,
        None => std::env::var_os("PIXI_CACHE_DIR")
            .map(|root| PathBuf::from(root).join("repodata"))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "retread sharded-universe: pass --repodata-dir <dir> or set PIXI_CACHE_DIR"
                )
            })?,
    };
    let indexes = pixi_build_retread::sharded_repodata::indexes_from_dir(&repodata_dir);
    for index in &indexes {
        println!(
            "{}",
            pixi_build_retread::sharded_repodata::sharded_index_row(index)
        );
    }
    println!(
        "{}",
        pixi_build_retread::sharded_repodata::sharded_universe_line(&indexes)
    );
    Ok(())
}
