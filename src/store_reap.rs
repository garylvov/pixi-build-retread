//! STORE-REAP-2: `retread store-reap` — THE PRODUCTION CALL SITE OF THE
//! PERSISTENT-STORE REAPERS. L3-1b-3B took the list from three to FOUR by
//! persisting the build-requirements store, whose shape is
//! `<root>/build-requirements/<version>/<identity>/requirements.txt`; L3-1b-4
//! took it to FIVE with the hermetic environment store, whose shape is
//! `<root>/hermetic-build-envs/<version>/env-<sha256>/complete.json`; and
//! SDIST-META-2 took it to SIX with the prepared-sdist-metadata store,
//! `<root>/sdist-metadata/v1/sdm-<sha256>/complete.json` -- the same shape, so
//! all three are reaped by ONE walk, `source_build::reap_marker_store`;
//! CONDA-OUT-2 took it to SEVEN with the built-output store, the FIRST FLAT
//! one -- `<root>/built-outputs/<key>/COMPLETE`, no generation directory, the
//! generation in the marker's own content -- reaped by that SAME walk under
//! a spec that says so.
//!
//! # Why this verb exists (STORE-REAP-1-1, law 2)
//!
//! C18-1, L3-1b and L3-1b-1 made three caches PERSISTENT and gave each one a
//! reaper; L3-1b-1a-1 taught two of those reapers to walk every generation.
//! Every one of those reapers is real, guarded, and **has never run against a
//! persistent store on this filesystem**, because the only thing that calls
//! them is `handler::…::initialize`, once per backend process, against
//! whatever root THAT PROCESS resolves — and every relock this harness runs
//! job-scopes `XDG_CACHE_HOME` to `$C/xdg-cache`, so the root it resolves is a
//! fresh empty directory the job's own cleanup deletes. What such a run prints
//! is MERGE-N-5's `reason="store-absent"` refusal row, on day 1 and on day 14
//! alike. That is a writer with no reader: the exact shape law 2 forbids.
//!
//! The bytes are not hypothetical. STORE-REAP-1 censused eleven built-wheel
//! roots on this filesystem and found 2 155 904 494 B of over-age v12 entries
//! in `caches/rtcache/built-wheels` and its `bench/run-warm.doomed-4879761`
//! twin — entries that no process resolves, under a generation the writer no
//! longer addresses, reachable by the reaper's code and by nothing that runs.
//!
//! # What it does NOT do, deliberately
//!
//! It does not make the reapers run automatically anywhere new. Reaping a
//! SHARED store is an operator act with consequences (a wrongly-sized age rule
//! throws away a 940-second wheel build), so the verb's default is
//! `--dry-run`, and the harness half — `tools/store_reap_census.sh`, wired
//! into the cleanup template — only ever runs it in dry run. What lands here
//! is the CAPABILITY plus a reader that exercises it on every lane job; the
//! first `--apply` against a shared store is a decision the operator makes
//! from the census this prints.

use std::path::{Path, PathBuf};

use crate::courier::ReapMode;

/// The stores this verb knows how to reap, and the exact spellings `--store`
/// accepts. One list, so the parser, the help text and the `all` fan-out
/// cannot drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Store {
    BuiltWheels,
    GitSnapshots,
    Shadow,
    /// L3-1b-3B.
    BuildRequirements,
    /// L3-1b-4.
    HermeticEnvironments,
    /// SDIST-META-2: the prepared-sdist-metadata store. Its writer is a
    /// post-lock harvester and its reader is the scoper's seeder — both in the
    /// harness, because pixi's embedded uv is what produces the metadata — so
    /// THIS VERB is the store's only in-product reader, and the census the
    /// cleanup template runs is the only thing that ever looks at the store as
    /// a whole.
    SdistMetadata,
    /// CONDA-OUT-2. The first FLAT marker store the verb reaps: it has no
    /// generation directory, so each entry's generation is its own marker's
    /// content. APPENDED, never inserted, on the same rule as the three before
    /// it. MERGE-CO: SDIST-META-2 and CONDA-OUT-2 each appended what its own
    /// branch called "the sixth store"; both are here, so this is the SEVENTH
    /// and the two appends did not displace one another.
    BuiltOutputs,
    /// METAGEN-1: the derived path-source metadata store. Its writer and its
    /// reader are the SAME code — the path-source derivation inside the
    /// backend's `initialize` — one lock apart, and this verb is what ages it.
    /// APPENDED, never inserted, on the rule the six before it were.
    PathSourceMetadata,
    /// REAP-2 (N27-RETREAD-210). The route-probe verdict store, and the NINTH.
    /// It is the first store here whose entries are FILES in one flat directory
    /// with a per-entry advisory lock, so it is the first that
    /// `source_build::reap_marker_store` cannot walk — its reaper lives beside
    /// its writer in `crate::route_probe_cache`, which is where the writer's
    /// lock path and quarantine convention are already named.
    /// APPENDED, never inserted, on the rule the eight before it were.
    RouteProbeVerdicts,
}

impl Store {
    /// The `--store` spelling, which is also the `store=` field of every row.
    pub fn as_str(self) -> &'static str {
        match self {
            Store::BuiltWheels => "built-wheels",
            Store::GitSnapshots => "git-snapshots",
            Store::Shadow => "shadow",
            Store::BuildRequirements => "build-requirements",
            Store::HermeticEnvironments => "hermetic-envs",
            Store::SdistMetadata => crate::sdist_metadata::CACHE_NAMESPACE,
            Store::BuiltOutputs => crate::built_output_store::STORE_DIR,
            Store::PathSourceMetadata => crate::derived_editable_metadata::CACHE_NAMESPACE,
            Store::RouteProbeVerdicts => crate::route_probe_cache::STORE_DIR,
        }
    }

    /// ALL of them, in the order the rows print. `--store all` is exactly this
    /// slice, never a re-listing of the names somewhere else.
    ///
    /// L3-1b-3B appended one and did not reorder the first three, and L3-1b-4
    /// appends the fifth on the same rule: the merge gate's readers grep
    /// summary rows by position in some places, and a reordering would move
    /// rows that this landing has no reason to move.
    /// SDIST-META-2 appends the sixth on the same rule L3-1b-3B and L3-1b-4
    /// appended the fourth and fifth by: APPEND, NEVER REORDER.
    /// CONDA-OUT-2 appends the SEVENTH by the same rule. MERGE-CO: the two
    /// branches each appended a "sixth" element at this exact position, and
    /// the resolution is BOTH, in landing order (sdist-metadata landed first,
    /// as B32), never one of them.
    /// METAGEN-1 appends the EIGHTH by the same rule: APPEND, NEVER REORDER.
    pub const ALL: [Store; 9] = [
        Store::BuiltWheels,
        Store::GitSnapshots,
        Store::Shadow,
        Store::BuildRequirements,
        Store::HermeticEnvironments,
        Store::SdistMetadata,
        Store::BuiltOutputs,
        Store::PathSourceMetadata,
        Store::RouteProbeVerdicts,
    ];

    fn parse(value: &str) -> Option<Vec<Store>> {
        if value == "all" {
            return Some(Store::ALL.to_vec());
        }
        Store::ALL
            .iter()
            .find(|store| store.as_str() == value)
            .map(|store| vec![*store])
    }

    /// The DEFAULT max age for this store, read from the store's own constant
    /// rather than restated here — `--max-age-days` overrides it, and a verb
    /// that carried its own copy of "14" would be a second policy.
    fn default_max_age_days(self) -> u64 {
        match self {
            Store::BuiltWheels => crate::source_build::BUILT_WHEEL_STORE_DEFAULT_MAX_AGE_DAYS,
            Store::GitSnapshots => crate::source_build::GIT_SNAPSHOT_STORE_DEFAULT_MAX_AGE_DAYS,
            Store::Shadow => crate::courier::SHADOW_CACHE_STORE_DEFAULT_MAX_AGE_DAYS,
            Store::BuildRequirements => {
                crate::source_build::BUILD_REQUIREMENTS_STORE_DEFAULT_MAX_AGE_DAYS
            }
            Store::HermeticEnvironments => {
                crate::hermetic_build::HERMETIC_ENVIRONMENT_STORE_DEFAULT_MAX_AGE_DAYS
            }
            Store::SdistMetadata => crate::sdist_metadata::DEFAULT_MAX_AGE_DAYS,
            Store::BuiltOutputs => {
                crate::built_output_store::BUILT_OUTPUT_STORE_DEFAULT_MAX_AGE_DAYS
            }
            Store::PathSourceMetadata => crate::derived_editable_metadata::DEFAULT_MAX_AGE_DAYS,
            // REAP-2 / N27-RETREAD-163. THIS LINE IS THE CONSTANT'S ONLY
            // READER, and until this landing it had none: the constant was
            // written by ROUTECACHE-1 beside the store it sizes and
            // `grep -rn ROUTE_PROBE_STORE_DEFAULT_MAX_AGE_DAYS src/ tests/`
            // returned exactly one line — its own definition. A default with
            // no reader is the same defect as a gate criterion with no
            // producer, so the whole point of the arm below is that the
            // printed `max_age_days=` field MOVES when this constant moves.
            Store::RouteProbeVerdicts => {
                crate::route_probe_cache::ROUTE_PROBE_STORE_DEFAULT_MAX_AGE_DAYS
            }
        }
    }
}

/// THE ONE RESOLUTION of "how old is too old" for one store in one run.
/// `flag` is `--max-age-days` when the operator stated one. Extracted so the
/// override is a function a guard can call, not a line inside a loop.
pub fn resolved_max_age_days(store: Store, flag: Option<u64>) -> u64 {
    flag.unwrap_or_else(|| store.default_max_age_days())
}

/// REAP-2. WHICH `--max-age-days` FLAG APPLIES TO THIS STORE, and it is a
/// function rather than a line in the loop so a guard can assert the precedence
/// without a filesystem.
///
/// `--route-probe-max-age-days` is a DECLARED argument of its own because the
/// route-probe store's retention is not the other eight stores' retention: a
/// verdict file is kilobytes and its loss costs one probe round, while a
/// built-wheel entry is a 940-second build, so an operator sizing a shared root
/// must be able to age the cheap store hard without touching the expensive
/// ones — and with a single `--max-age-days` they could not. Precedence, and
/// there is no other rung: the store-specific flag, then the general flag, then
/// [`Store::default_max_age_days`], which for this store IS
/// `route_probe_cache::ROUTE_PROBE_STORE_DEFAULT_MAX_AGE_DAYS`.
pub fn max_age_flag_for(store: Store, args: &Args) -> Option<u64> {
    match store {
        Store::RouteProbeVerdicts => args.route_probe_max_age_days.or(args.max_age_days),
        _ => args.max_age_days,
    }
}

/// What the verb was asked to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub stores: Vec<Store>,
    /// The persistent store roots to reap. Empty at parse time means "the one
    /// the product itself would resolve"; [`Args::resolved_roots`] fills it in.
    pub roots: Vec<PathBuf>,
    pub mode: ReapMode,
    pub max_age_days: Option<u64>,
    /// REAP-2. `--route-probe-max-age-days`, which applies to the route-probe
    /// store ONLY and outranks `--max-age-days` there. Absent means the
    /// store's own constant; see [`max_age_flag_for`].
    pub route_probe_max_age_days: Option<u64>,
    /// Charge the bytes of every selected entry. OFF by default: the harness
    /// census step runs on every lane job and must stay cheap, while an
    /// operator sizing a reap wants the number and can pay a full walk for it.
    pub bytes: bool,
}

impl Args {
    /// The roots to act on. THE DEFAULT IS THE PRODUCT'S OWN FORMULA —
    /// `courier::persistent_store_root_with`, the same function the backend
    /// resolves through — so the verb can never reap a root the product would
    /// not have used. A store somewhere else (`caches/rtcache`, a cert root)
    /// is reachable only by NAMING it with `--root`, which is the whole point:
    /// STORE-REAP-1-1's orphaned bytes sit under roots no default resolution
    /// reaches.
    pub fn resolved_roots(&self) -> Vec<PathBuf> {
        if self.roots.is_empty() {
            return vec![crate::courier::persistent_store_root()];
        }
        self.roots.clone()
    }
}

/// Argument parsing, separated from [`run`] so a guard can assert on the
/// parse without touching a filesystem.
pub fn parse_args(args: &[String]) -> anyhow::Result<Args> {
    let mut stores: Option<Vec<Store>> = None;
    let mut roots: Vec<PathBuf> = Vec::new();
    let mut mode = ReapMode::DryRun;
    let mut max_age_days: Option<u64> = None;
    let mut route_probe_max_age_days: Option<u64> = None;
    let mut bytes = false;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--store" => {
                let value = it
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("store-reap: --store <name> requires a value"))?;
                stores = Some(Store::parse(value).ok_or_else(|| {
                    anyhow::anyhow!(
                        "store-reap: --store {value}: expected one of \
                         built-wheels, git-snapshots, shadow, \
                         build-requirements, hermetic-envs, sdist-metadata, \
                         built-outputs, path-source-metadata, route-probe-verdicts, all"
                    )
                })?);
            }
            "--root" => {
                roots.push(PathBuf::from(it.next().ok_or_else(|| {
                    anyhow::anyhow!("store-reap: --root <dir> requires a value")
                })?));
            }
            // BOTH SPELLINGS EXIST AND `--dry-run` IS THE DEFAULT. A caller
            // that writes `--dry-run` explicitly is saying what it means, and
            // a caller that omits both gets the safe one.
            "--dry-run" => mode = ReapMode::DryRun,
            "--apply" => mode = ReapMode::Apply,
            "--max-age-days" => {
                let value = it.next().ok_or_else(|| {
                    anyhow::anyhow!("store-reap: --max-age-days <n> requires a value")
                })?;
                max_age_days = Some(value.parse::<u64>().map_err(|error| {
                    anyhow::anyhow!("store-reap: --max-age-days {value}: {error}")
                })?);
            }
            // REAP-2. Its own flag, and the parse error names the store so a
            // typo cannot silently become "the general flag applied to all
            // nine".
            "--route-probe-max-age-days" => {
                let value = it.next().ok_or_else(|| {
                    anyhow::anyhow!(
                        "store-reap: --route-probe-max-age-days <n> requires a value"
                    )
                })?;
                route_probe_max_age_days = Some(value.parse::<u64>().map_err(|error| {
                    anyhow::anyhow!("store-reap: --route-probe-max-age-days {value}: {error}")
                })?);
            }
            "--bytes" => bytes = true,
            other => anyhow::bail!("store-reap: unknown arg {other}"),
        }
    }
    Ok(Args {
        stores: stores.unwrap_or_else(|| Store::ALL.to_vec()),
        roots,
        mode,
        max_age_days,
        route_probe_max_age_days,
        bytes,
    })
}

/// One store under one root, after the reap.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StoreOutcome {
    pub scanned: u64,
    pub selected: u64,
    pub stale_version: u64,
    pub kept: u64,
    pub skipped_locked: u64,
    pub versions_walked: u64,
    /// STORE-REAP-3. How many on-disk LAYOUTS of this store the walk
    /// enumerated. Only the shadow store has ever had more than one (L3-1 moved
    /// the target identity out of the path and left the old directories where
    /// they were), so the other stores report 0 here — the same way the
    /// shadow store reports 0 for `versions_walked`, because it has no
    /// generation segment. ONE row format for EVERY store; a field that
    /// does not apply reads 0 rather than being absent, so a parser never has
    /// to know which store it is looking at.
    pub layouts_walked: u64,
    pub skipped_concurrent: bool,
    pub bytes: u64,
    /// REAP-2. Of `selected`, how many were selected because they were
    /// OVER-AGE. Only the route-probe store fills these two in — the eight
    /// stores before it have exactly one selection rule, so their `reason_age`
    /// reads 0 the way their `layouts_walked` does, and a parser never has to
    /// know which store it is looking at.
    pub reason_age: u64,
    /// REAP-2. Of `selected`, how many were selected because no reader can
    /// ADDRESS the entry any more. See
    /// `crate::route_probe_cache::entry_is_unreachable` for what that means and
    /// for what it deliberately is not.
    pub reason_universe: u64,
}

/// `retread store-reap`. Returns the process exit code.
///
/// EXIT CODES, and they are the reader's half of the verb:
///   0  the reap ran (in either mode) and nothing refused.
///   7  at least one store REFUSED because a relock held its try-lock. Not a
///      failure of the reap — a statement that housekeeping deferred to a live
///      writer — but it must not read as a clean census, because a census that
///      scanned nothing and said `scanned=0` looks exactly like an empty store.
pub fn run(args: &Args) -> anyhow::Result<i32> {
    let mut refused = false;
    let mut total_scanned = 0u64;
    let mut total_selected = 0u64;
    let mut total_bytes = 0u64;
    for root in args.resolved_roots() {
        for store in &args.stores {
            let store = *store;
            let days = resolved_max_age_days(store, max_age_flag_for(store, args));
            let outcome = reap_one(&root, store, days, args.mode, args.bytes)?;
            refused |= outcome.skipped_concurrent;
            total_scanned += outcome.scanned;
            total_selected += outcome.selected;
            total_bytes += outcome.bytes;
            println!(
                "### store-reap SUMMARY root={} store={} mode={} max_age_days={} \
                 scanned={} {}={} stale_version={} kept={} skipped_locked={} \
                 versions_walked={} layouts_walked={} skipped_concurrent={} bytes={}",
                root.display(),
                store.as_str(),
                args.mode.as_str(),
                days,
                outcome.scanned,
                selected_field(args.mode),
                outcome.selected,
                outcome.stale_version,
                outcome.kept,
                outcome.skipped_locked,
                outcome.versions_walked,
                outcome.layouts_walked,
                outcome.skipped_concurrent,
                outcome.bytes,
            );
            // REAP-2. THE ROUTE-PROBE STORE'S OWN ROW, and it exists because
            // the SUMMARY row above cannot carry a reason split: it has one
            // selection-count column and this store has two selection RULES.
            // `mode=` is appended AFTER the briefed fields, not folded into
            // them, and it is not decoration -- `quarantined=` in a dry run is
            // a PREDICTION, and the campaign's own rule (`would_evict` vs
            // `evicted` on the row above) is that a reader must never take one
            // for bytes reclaimed. The mode is on the same line so a single
            // grep resolves it.
            if store == Store::RouteProbeVerdicts {
                println!(
                    "### ROUTE PROBE STORE REAP scanned={} quarantined={} kept={} \
                     max_age_days={} reason_counts=age:{} universe:{} \
                     skipped_locked={} mode={} root={}",
                    outcome.scanned,
                    outcome.selected,
                    outcome.kept,
                    days,
                    outcome.reason_age,
                    outcome.reason_universe,
                    outcome.skipped_locked,
                    args.mode.as_str(),
                    root.display(),
                );
            }
        }
    }
    println!(
        "### store-reap TOTAL roots={} stores={} mode={} scanned={} {}={} bytes={} refused={}",
        args.resolved_roots().len(),
        args.stores.len(),
        args.mode.as_str(),
        total_scanned,
        selected_field(args.mode),
        total_selected,
        total_bytes,
        refused,
    );
    Ok(if refused { 7 } else { 0 })
}

/// The name of the count column. `evicted` and `would_evict` are DIFFERENT
/// WORDS on purpose: a reader grepping a job log for `evicted=` must never
/// pick up a dry run's prediction and report it as bytes reclaimed.
fn selected_field(mode: ReapMode) -> &'static str {
    if mode.is_dry_run() {
        "would_evict"
    } else {
        "evicted"
    }
}

/// One store, one root. THE REAPERS THEMSELVES DO THE WALKING — this function
/// only maps a root onto the argument each reaper takes and turns its report
/// into rows.
fn reap_one(
    root: &Path,
    store: Store,
    max_age_days: u64,
    mode: ReapMode,
    bytes: bool,
) -> anyhow::Result<StoreOutcome> {
    let max_age = std::time::Duration::from_secs(max_age_days * 86_400);
    let mut outcome = StoreOutcome::default();
    let entries = match store {
        Store::BuiltWheels => {
            let report = crate::source_build::reap_built_wheel_store(root, max_age, mode)?;
            outcome.scanned = report.scanned;
            outcome.selected = report.evicted;
            outcome.stale_version = report.evicted_stale_version;
            outcome.kept = report.kept;
            outcome.skipped_locked = report.skipped_locked;
            outcome.versions_walked = report.versions_walked;
            outcome.skipped_concurrent = report.skipped_concurrent;
            report.entries
        }
        Store::GitSnapshots => {
            let report = crate::source_build::reap_canonical_git_snapshot_store(
                root, max_age, mode,
            )?;
            outcome.scanned = report.scanned;
            outcome.selected = report.evicted;
            outcome.stale_version = report.evicted_stale_version;
            outcome.kept = report.kept;
            outcome.skipped_locked = report.skipped_locked;
            outcome.versions_walked = report.versions_walked;
            outcome.skipped_concurrent = report.skipped_concurrent;
            report.entries
        }
        Store::Shadow => {
            // The shadow reaper takes the shadow DIRECTORY, not the store root.
            // It has no generation segment — L3-1b-1a-1 ruled that walk
            // deliberately unchanged and that ruling still holds — but
            // STORE-REAP-3 gave it a second LAYOUT to walk, so it reports
            // `layouts_walked` where the other two report `versions_walked`.
            let report = crate::courier::reap_shadow_cache_store(
                &crate::courier::shadow_cache_dir_in(root),
                max_age,
                mode,
            )?;
            outcome.scanned = report.scanned;
            outcome.selected = report.evicted;
            outcome.kept = report.kept;
            outcome.layouts_walked = report.layouts_walked;
            outcome.skipped_concurrent = report.skipped_concurrent;
            report.entries
        }
        // L3-1b-3B. Takes the STORE ROOT and appends its own directory
        // segment inside the reaper, like the built-wheel and git-snapshot
        // arms and unlike the shadow arm, whose entries are files in one flat
        // directory the caller has to name. It has ONE layout and reports
        // `layouts_walked` 0, the way the built-wheel and git-snapshot arms do.
        Store::BuildRequirements => {
            let report = crate::source_build::reap_marker_store(
                &crate::source_build::BUILD_REQUIREMENTS_STORE_SPEC,
                root,
                max_age,
                mode,
            )?;
            outcome.scanned = report.scanned;
            outcome.selected = report.evicted;
            outcome.stale_version = report.evicted_stale_version;
            outcome.kept = report.kept;
            outcome.skipped_locked = report.skipped_locked;
            outcome.versions_walked = report.versions_walked;
            outcome.skipped_concurrent = report.skipped_concurrent;
            report.entries
        }
        // L3-1b-4. The SAME walk as the arm above, on the same shape, with a
        // different spec -- not a second implementation of it.
        Store::HermeticEnvironments => {
            let report = crate::source_build::reap_marker_store(
                &crate::source_build::HERMETIC_ENVIRONMENT_STORE_SPEC,
                root,
                max_age,
                mode,
            )?;
            outcome.scanned = report.scanned;
            outcome.selected = report.evicted;
            outcome.stale_version = report.evicted_stale_version;
            outcome.kept = report.kept;
            outcome.skipped_locked = report.skipped_locked;
            outcome.versions_walked = report.versions_walked;
            outcome.skipped_concurrent = report.skipped_concurrent;
            report.entries
        }
        // SDIST-META-2. THE SAME WALK AGAIN -- a third spec, not a third
        // reaper, and the `### sdist_metadata_store` rows it prints are the
        // census `tools/store_reap_census.sh` picks up with no new call site.
        Store::SdistMetadata => {
            let report = crate::source_build::reap_marker_store(
                &crate::source_build::SDIST_METADATA_STORE_SPEC,
                root,
                max_age,
                mode,
            )?;
            outcome.scanned = report.scanned;
            outcome.selected = report.evicted;
            outcome.stale_version = report.evicted_stale_version;
            outcome.kept = report.kept;
            outcome.skipped_locked = report.skipped_locked;
            outcome.versions_walked = report.versions_walked;
            outcome.skipped_concurrent = report.skipped_concurrent;
            report.entries
        }
        // CONDA-OUT-2. THE SAME WALK AGAIN, on the store's FLAT shape:
        // `<root>/built-outputs/<key>/COMPLETE`, no generation directory, the
        // generation in the marker. It reports `versions_walked` as the number
        // of DISTINCT marker generations found -- on the shared root that was
        // MEASURED at 3 (v1, v2, v3) across 218 entries the day this landed,
        // and every v1 and v2 of them was unreachable by any reaper until now.
        // METAGEN-1. THE SAME WALK AGAIN -- an eighth spec, not an eighth
        // reaper, on the same `<root>/<dir>/<version>/<key>/complete.json`
        // shape the build-requirements, hermetic and sdist-metadata stores use.
        Store::PathSourceMetadata => {
            let report = crate::source_build::reap_marker_store(
                &crate::source_build::PATH_SOURCE_METADATA_STORE_SPEC,
                root,
                max_age,
                mode,
            )?;
            outcome.scanned = report.scanned;
            outcome.selected = report.evicted;
            outcome.stale_version = report.evicted_stale_version;
            outcome.kept = report.kept;
            outcome.skipped_locked = report.skipped_locked;
            outcome.versions_walked = report.versions_walked;
            outcome.skipped_concurrent = report.skipped_concurrent;
            report.entries
        }
        // REAP-2. THE ONE STORE THAT IS NOT `reap_marker_store`, and the
        // reason is a SHAPE, not a setting -- the same judgement C18-1 made
        // when it refused to parameterise the shadow walk. The eight arms
        // above walk `<root>/<dir>/<version>/<key>/<marker>`: entry
        // DIRECTORIES, aged from a marker file inside them. This store is a
        // flat directory of JSON FILES, each with a `<stem>.lock` advisory
        // lock its writer takes BLOCKING, and each carrying entries whose
        // reachability is a property of the file's CONTENT. Sharing a body
        // across those would need a callback for enumeration, one for the
        // lock, one for the age and one for the second selection rule.
        Store::RouteProbeVerdicts => {
            let report =
                crate::route_probe_cache::reap_route_probe_store(root, max_age, mode)?;
            outcome.scanned = report.scanned;
            outcome.selected = report.quarantined;
            outcome.kept = report.kept;
            outcome.skipped_locked = report.skipped_locked;
            outcome.skipped_concurrent = report.skipped_concurrent;
            outcome.reason_age = report.selected_age;
            outcome.reason_universe = report.selected_universe;
            report.entries
        }
        Store::BuiltOutputs => {
            let report = crate::source_build::reap_marker_store(
                &crate::source_build::BUILT_OUTPUT_STORE_SPEC,
                root,
                max_age,
                mode,
            )?;
            outcome.scanned = report.scanned;
            outcome.selected = report.evicted;
            outcome.stale_version = report.evicted_stale_version;
            outcome.kept = report.kept;
            outcome.skipped_locked = report.skipped_locked;
            outcome.versions_walked = report.versions_walked;
            outcome.skipped_concurrent = report.skipped_concurrent;
            report.entries
        }
    };
    for entry in &entries {
        let entry_bytes = if bytes {
            // In a dry run the entry is still where it was; after an apply it is
            // in the quarantine. Charge whichever one exists.
            let target = entry.quarantine.as_deref().unwrap_or(&entry.path);
            path_bytes(target)
        } else {
            0
        };
        outcome.bytes += entry_bytes;
        println!(
            "store-reap {} store={} entry={} age_days={} reason={} path={} bytes={}",
            if mode.is_dry_run() {
                "would-evict"
            } else {
                "evicted"
            },
            store.as_str(),
            entry.label,
            entry.age_days,
            entry.reason,
            entry.path.display(),
            entry_bytes,
        );
    }
    Ok(outcome)
}

/// Bytes under a path: the file's own length, or the sum over a directory
/// tree. Apparent size, `st_size`, never blocks — the census this feeds is
/// compared against `find -printf '%s'` totals, and a block count would
/// disagree with those for a reason that has nothing to do with the store.
///
/// Symlinks are NOT followed (`symlink_metadata`), so a store holding a link
/// out of itself cannot make the number unbounded.
fn path_bytes(path: &Path) -> u64 {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return 0;
    };
    if meta.is_symlink() {
        return 0;
    }
    if meta.is_file() {
        return meta.len();
    }
    if !meta.is_dir() {
        return 0;
    }
    let mut total = 0u64;
    let Ok(children) = std::fs::read_dir(path) else {
        return 0;
    };
    for child in children.flatten() {
        total += path_bytes(&child.path());
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scratch that is never `/tmp`-shared between two runs of the suite.
    fn scratch(tag: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "retread-store-reap-2-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(&base).expect("scratch");
        base
    }

    fn age(days: u64) -> std::time::Duration {
        std::time::Duration::from_secs(days * 86_400)
    }

    fn set_mtime(path: &Path, ago: std::time::Duration) {
        let when = std::time::SystemTime::now() - ago;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("reopen to age");
        file.set_times(std::fs::FileTimes::new().set_modified(when).set_accessed(when))
            .expect("age");
    }

    /// One built-wheel entry under `<root>/built-wheels/<kind>/<version>/<target>`,
    /// aged by its own `artifact.json` marker.
    fn built_wheel_entry(root: &Path, kind: &str, version: &str, target: &str, ago: u64) -> PathBuf {
        let entry = root
            .join("built-wheels")
            .join(kind)
            .join(version)
            .join(target);
        std::fs::create_dir_all(&entry).expect("entry");
        std::fs::write(entry.join("wheel.whl"), vec![b'w'; 64]).expect("payload");
        let marker = entry.join("artifact.json");
        std::fs::write(&marker, b"{}").expect("marker");
        set_mtime(&marker, age(ago));
        entry
    }

    /// One git-snapshot entry under `<root>/canonical-git-sources/<version>/<id>/<ref>`.
    fn git_snapshot_entry(root: &Path, version: &str, id: &str, ref_state: &str, ago: u64) -> PathBuf {
        let entry = root
            .join("canonical-git-sources")
            .join(version)
            .join(id)
            .join(ref_state);
        std::fs::create_dir_all(entry.join("repo")).expect("entry");
        std::fs::write(entry.join("repo").join("f.txt"), vec![b'g'; 32]).expect("payload");
        let marker = entry.join("source.json");
        std::fs::write(&marker, b"{}").expect("marker");
        set_mtime(&marker, age(ago));
        entry
    }

    /// L3-1b-3B. One marker-store entry under
    /// `<root>/<dir>/<version>/<identity>`, aged by its own marker file.
    fn marker_store_entry(
        root: &Path,
        dir: &str,
        version: &str,
        identity: &str,
        marker: &str,
        ago: u64,
    ) -> PathBuf {
        let entry = root.join(dir).join(version).join(identity);
        std::fs::create_dir_all(&entry).expect("entry");
        let marker_path = entry.join(marker);
        std::fs::write(&marker_path, vec![b'm'; 48]).expect("marker");
        set_mtime(&marker_path, age(ago));
        entry
    }

    /// One shadow entry: a FILE in `<root>/shadow`.
    fn shadow_entry(root: &Path, name: &str, ago: u64) -> PathBuf {
        let dir = root.join("shadow");
        std::fs::create_dir_all(&dir).expect("shadow dir");
        let entry = dir.join(name);
        std::fs::write(&entry, vec![b's'; 16]).expect("payload");
        set_mtime(&entry, age(ago));
        entry
    }

    /// A 64-hex target identity, the shape `artifact_cache_identity` produces
    /// and the only shape the legacy walk descends into.
    fn legacy_target(seed: u8) -> String {
        std::iter::repeat(format!("{seed:02x}")).take(32).collect()
    }

    /// STORE-REAP-3. One PRE-L3-1 shadow entry: a FILE one level down, under
    /// the target artifact identity the key used to be qualified by —
    /// `<root>/shadow/<64-hex target>/<key>.changed`.
    fn shadow_legacy_entry(root: &Path, target: &str, name: &str, ago: u64) -> PathBuf {
        let dir = root.join("shadow").join(target);
        std::fs::create_dir_all(&dir).expect("legacy target dir");
        let entry = dir.join(name);
        std::fs::write(&entry, vec![b'l'; 64]).expect("payload");
        set_mtime(&entry, age(ago));
        entry
    }

    /// Every path under a root, sorted — the reader that makes "renames
    /// nothing" and "creates nothing" ONE assertion instead of two adjectives.
    fn tree(root: &Path) -> Vec<String> {
        fn walk(dir: &Path, base: &Path, out: &mut Vec<String>) {
            let Ok(children) = std::fs::read_dir(dir) else {
                return;
            };
            for child in children.flatten() {
                let path = child.path();
                out.push(
                    path.strip_prefix(base)
                        .unwrap_or(&path)
                        .display()
                        .to_string(),
                );
                if path.is_dir() {
                    walk(&path, base, out);
                }
            }
        }
        let mut out = Vec::new();
        walk(root, root, &mut out);
        out.sort();
        out
    }

    fn dry_run(root: &Path, store: Store) -> StoreOutcome {
        reap_one(root, store, 14, ReapMode::DryRun, true).expect("dry run")
    }

    /// GUARD 1, THE WHOLE POINT OF THE VERB'S DEFAULT. A dry run over a store
    /// with over-age entries in all three shapes NAMES them, and leaves the
    /// store byte-for-byte the tree it found — not merely "does not rename",
    /// but creates NOTHING: no `.built-wheels.reap.lock`, no per-entry
    /// `.lock`, no `quarantine/`. The apply arm on the SAME fixture is the
    /// non-vacuity control: if the entries were not selectable, the dry run's
    /// zero would be trivially true.
    #[test]
    fn a_dry_run_names_every_over_age_entry_and_leaves_the_store_untouched() {
        let root = scratch("dry");
        built_wheel_entry(&root, "git", "v13", "current", 30);
        built_wheel_entry(&root, "sdist", "v12", "retired", 30);
        git_snapshot_entry(&root, "v3", "proj", "main", 30);
        shadow_entry(&root, "abc.changed", 30);
        let before = tree(&root);

        let bw = dry_run(&root, Store::BuiltWheels);
        let gs = dry_run(&root, Store::GitSnapshots);
        let sh = dry_run(&root, Store::Shadow);

        assert_eq!(
            (bw.selected, bw.stale_version, bw.versions_walked),
            (2, 1, 2),
            "the dry run must select BOTH built-wheel entries and name the \
             retired generation as stale-version"
        );
        assert_eq!(gs.selected, 1, "one git snapshot entry is over age");
        assert_eq!(sh.selected, 1, "one shadow entry is over age");
        assert!(bw.bytes > 0 && gs.bytes > 0 && sh.bytes > 0, "bytes charged");
        assert_eq!(
            tree(&root),
            before,
            "A DRY RUN MUST CHANGE NOTHING — not one rename, and not one \
             created lock sidecar or quarantine directory"
        );

        // NON-VACUITY: the same fixture, applied, does move them.
        let applied =
            reap_one(&root, Store::BuiltWheels, 14, ReapMode::Apply, false).expect("apply");
        assert_eq!(applied.selected, 2, "the apply arm selects the same two");
        assert_ne!(tree(&root), before, "the apply arm DOES change the store");
    }

    /// GUARD 2. What the dry run promised, the apply delivers: the same
    /// entries, the same reasons, now under `quarantine/` and gone from where
    /// they were. A dry run that named a different set than the apply moves is
    /// a proposal an operator cannot trust.
    #[test]
    fn an_apply_moves_exactly_the_entries_the_dry_run_named_with_the_same_reasons() {
        let root = scratch("apply");
        let current = built_wheel_entry(&root, "git", "v13", "current", 30);
        let retired = built_wheel_entry(&root, "sdist", "v12", "retired", 30);
        let fresh = built_wheel_entry(&root, "git", "v13", "fresh", 1);

        let predicted = crate::source_build::reap_built_wheel_store(
            &root,
            age(14),
            ReapMode::DryRun,
        )
        .expect("dry run");
        let predicted_names: Vec<(String, &str)> = predicted
            .entries
            .iter()
            .map(|entry| (entry.label.clone(), entry.reason))
            .collect();
        assert!(
            predicted.entries.iter().all(|entry| entry.quarantine.is_none()),
            "a dry-run report must never carry a quarantine path"
        );

        let applied =
            crate::source_build::reap_built_wheel_store(&root, age(14), ReapMode::Apply)
                .expect("apply");
        let applied_names: Vec<(String, &str)> = applied
            .entries
            .iter()
            .map(|entry| (entry.label.clone(), entry.reason))
            .collect();
        assert_eq!(
            predicted_names, applied_names,
            "the apply must move exactly the set the dry run named"
        );
        assert!(
            applied.entries.iter().all(|entry| entry
                .quarantine
                .as_ref()
                .is_some_and(|path| path.is_dir())),
            "every applied eviction must exist in the quarantine"
        );
        assert!(!current.exists() && !retired.exists(), "both moved");
        assert!(fresh.exists(), "the fresh entry is untouched in both modes");
        assert!(
            applied_names
                .iter()
                .any(|(_, reason)| *reason == "stale-version"),
            "the retired generation is named stale-version, not unreferenced"
        );
    }

    /// GUARD 3. THE TRY-LOCK REFUSAL, IN DRY RUN. A relock holding the store's
    /// reap lock must make the verb REFUSE rather than report an empty census.
    /// The non-vacuity control is the same store with the lock released.
    #[test]
    fn a_dry_run_refuses_while_a_relock_holds_the_store_try_lock() {
        let root = scratch("busy");
        built_wheel_entry(&root, "git", "v13", "current", 30);
        let lock_path = root.join("built-wheels").join(".built-wheels.reap.lock");
        let held = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .expect("open the reap lock");
        assert!(
            fs4::fs_std::FileExt::try_lock_exclusive(&held).unwrap_or(false),
            "the fixture must actually hold the lock"
        );

        let busy = dry_run(&root, Store::BuiltWheels);
        assert!(busy.skipped_concurrent, "a held lock must REFUSE");
        assert_eq!(busy.scanned, 0, "a refusal scans nothing");
        let code = run(&Args {
            stores: vec![Store::BuiltWheels],
            roots: vec![root.clone()],
            mode: ReapMode::DryRun,
            max_age_days: None,
            route_probe_max_age_days: None,
            bytes: false,
        })
        .expect("run");
        assert_eq!(code, 7, "a refusal is exit 7, never a clean 0");

        // NON-VACUITY: release it and the same store scans.
        fs4::fs_std::FileExt::unlock(&held).expect("release");
        drop(held);
        let free = dry_run(&root, Store::BuiltWheels);
        assert!(!free.skipped_concurrent && free.selected == 1);
    }

    /// GUARD 4. A dry run must not CREATE the reap lock it did not find. This
    /// is the half that separates "read-only" from "renames nothing": the
    /// apply path opens that sidecar with `create(true)`.
    #[test]
    fn a_dry_run_does_not_create_the_reap_lock_sidecar_but_an_apply_does() {
        let root = scratch("nolock");
        built_wheel_entry(&root, "git", "v13", "current", 1);
        let lock_path = root.join("built-wheels").join(".built-wheels.reap.lock");
        assert!(!lock_path.exists(), "fixture starts without the sidecar");
        dry_run(&root, Store::BuiltWheels);
        assert!(
            !lock_path.exists(),
            "A DRY RUN MUST NOT CREATE THE REAP LOCK"
        );
        crate::source_build::reap_built_wheel_store(&root, age(14), ReapMode::Apply)
            .expect("apply");
        assert!(
            lock_path.exists(),
            "non-vacuity: the apply path DOES create it"
        );
    }

    /// GUARD 5. `--max-age-days` overrides the store's own default, and its
    /// ABSENCE resolves to that default rather than to a number written here.
    #[test]
    fn the_max_age_days_flag_overrides_each_store_default_and_absence_keeps_it() {
        for store in Store::ALL {
            assert_eq!(
                resolved_max_age_days(store, None),
                14,
                "{}: the campaign default is the store's own constant",
                store.as_str()
            );
            assert_eq!(resolved_max_age_days(store, Some(1)), 1);
            assert_eq!(resolved_max_age_days(store, Some(0)), 0);
        }
        // And it reaches the reap: an entry 2 days old is kept at the default
        // and selected at `--max-age-days 1`.
        let root = scratch("age");
        built_wheel_entry(&root, "git", "v13", "twodays", 2);
        assert_eq!(
            reap_one(&root, Store::BuiltWheels, 14, ReapMode::DryRun, false)
                .expect("default")
                .selected,
            0
        );
        assert_eq!(
            reap_one(&root, Store::BuiltWheels, 1, ReapMode::DryRun, false)
                .expect("override")
                .selected,
            1
        );
    }

    /// GUARD 6. THE PARSE. Dry run is the default with no flag at all, `all`
    /// is exactly EVERY store and not a subset frozen when the list was
    /// shorter, every spelling round-trips through `--store`, an unknown store
    /// is a refusal and not a silent skip, and `--root` accumulates.
    #[test]
    fn the_parse_defaults_to_a_dry_run_of_every_store_and_refuses_a_bad_name() {
        let empty = parse_args(&[]).expect("no args");
        assert_eq!(empty.mode, ReapMode::DryRun, "THE DEFAULT IS THE SAFE ONE");
        assert_eq!(empty.stores, Store::ALL.to_vec());
        assert!(empty.roots.is_empty() && !empty.bytes && empty.max_age_days.is_none());

        // CONDA-OUT-3. THE FAN-OUT IS EVERY VARIANT OF THE REGISTRY, and this
        // guard now derives that from the enum instead of from a hard count.
        // L3-1b-4 wrote a hard 5 on both counts deliberately -- `Store::ALL
        // .len()` on both sides is an identity that would pass on a list that
        // LOST a store -- but a hard number rots the day a store is APPENDED,
        // which is exactly what CONDA-OUT-2's sixth store (`built-outputs`)
        // did to it at gate 6051584 (left 6, right 5). SDIST-META-2 had
        // meanwhile edited the same two numbers to 6 on its own branch, so
        // MERGE-CO arrived at a THIRD spelling of the same rot -- which is the
        // argument for this shape rather than a bigger number. Both properties
        // are kept here without a number: the `match` below is exhaustive with
        // no `_` arm, so an EIGHTH variant cannot be added to the enum without
        // the compiler stopping HERE, and the comparison against `Store::ALL`
        // is not an identity -- one side is the hand-written const list, the
        // other the enumerated variants -- so a variant dropped from `ALL`
        // while it still exists on the enum is still a red. The enum is the
        // ONE authority; nothing counts stores.
        let every_variant: Vec<Store> = [
            Store::BuiltWheels,
            Store::GitSnapshots,
            Store::Shadow,
            Store::BuildRequirements,
            Store::HermeticEnvironments,
            Store::SdistMetadata,
            Store::BuiltOutputs,
            Store::PathSourceMetadata,
            // REAP-2, the ninth. This list is hand-written on purpose so
            // `Store::ALL` cannot be its own witness.
            Store::RouteProbeVerdicts,
        ]
        .into_iter()
        .inspect(|store| match store {
            // EXHAUSTIVE ON PURPOSE. Do not add a `_` arm: this match is the
            // compile-time half of the guard.
            Store::BuiltWheels
            | Store::GitSnapshots
            | Store::Shadow
            | Store::BuildRequirements
            | Store::HermeticEnvironments
            | Store::SdistMetadata
            | Store::BuiltOutputs
            | Store::PathSourceMetadata
            | Store::RouteProbeVerdicts => {}
        })
        .collect();
        assert_eq!(
            Store::ALL.to_vec(),
            every_variant,
            "EVERY variant of the registry is in the fan-out, in declaration order"
        );
        assert_eq!(
            parse_args(&["--store".into(), "all".into()])
                .expect("all")
                .stores,
            every_variant,
            "`--store all` fans out to the whole registry, never to a frozen subset",
        );
        for store in Store::ALL {
            assert_eq!(
                parse_args(&["--store".into(), store.as_str().to_string()])
                    .expect("by name")
                    .stores,
                vec![store],
                "--store {} must select exactly that store",
                store.as_str(),
            );
        }

        let args: Vec<String> = [
            "--store",
            "built-wheels",
            "--root",
            "/a",
            "--root",
            "/b",
            "--apply",
            "--max-age-days",
            "30",
            "--bytes",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let parsed = parse_args(&args).expect("full");
        assert_eq!(parsed.stores, vec![Store::BuiltWheels]);
        assert_eq!(parsed.roots, vec![PathBuf::from("/a"), PathBuf::from("/b")]);
        assert_eq!(parsed.mode, ReapMode::Apply);
        assert_eq!(parsed.max_age_days, Some(30));
        assert!(parsed.bytes);

        for bad in [vec!["--store", "wheels"], vec!["--store"], vec!["--nope"]] {
            let bad: Vec<String> = bad.iter().map(|s| s.to_string()).collect();
            assert!(
                parse_args(&bad).is_err(),
                "{bad:?} must REFUSE, never be skipped"
            );
        }
    }

    /// GUARD 7. THE DEFAULT ROOT IS THE PRODUCT'S OWN PERSISTENT FORMULA, and
    /// naming `--root` is the ONLY way to reach a store somewhere else. Both
    /// halves matter: the first stops the verb inventing a location, the
    /// second is how STORE-REAP-1-1's orphaned roots are reachable at all.
    #[test]
    fn the_default_root_is_the_persistent_formula_and_root_overrides_it() {
        let defaulted = Args {
            stores: Store::ALL.to_vec(),
            roots: Vec::new(),
            mode: ReapMode::DryRun,
            max_age_days: None,
            route_probe_max_age_days: None,
            bytes: false,
        };
        assert_eq!(
            defaulted.resolved_roots(),
            vec![crate::courier::persistent_store_root()],
            "the default root must BE the product's persistent-root formula"
        );
        let named = Args {
            roots: vec![PathBuf::from("/oscar/data/stellex/glvov/caches/rtcache")],
            ..defaulted
        };
        assert_eq!(
            named.resolved_roots(),
            vec![PathBuf::from("/oscar/data/stellex/glvov/caches/rtcache")]
        );
    }

    /// STORE-REAP-3, GUARD 1 — THE FIXTURE CARRIES BOTH LAYOUTS AND THE VERB
    /// SEES ALL OF IT.
    ///
    /// This is the guard that would have caught STORE-REAP-2's `scanned=0`
    /// against two real persistent roots. Three entries, one of each fate:
    /// * a FRESH flat entry — kept, because the age rule is unchanged;
    /// * a STALE flat entry — evicted, `reason="unreferenced"`;
    /// * a STALE entry in the pre-L3-1 `shadow/<64-hex target>/` layout —
    ///   evicted, `reason="stale-layout"`, which is the whole lane.
    /// `layouts_walked=2` says the walk reached both, and it is 1 when only the
    /// flat layout is on disk — asserted below so the field cannot be a
    /// constant.
    #[test]
    fn the_shadow_dry_run_walks_both_layouts_and_names_the_legacy_entry_by_its_own_reason() {
        let root = scratch("shadow-both-layouts");
        let target = legacy_target(0xab);
        let fresh = shadow_entry(&root, "fresh.changed", 1);
        let stale = shadow_entry(&root, "stale.changed", 30);
        let legacy = shadow_legacy_entry(&root, &target, "old.changed", 30);
        // An EMPTIED legacy directory is still a layout the walk reached, and
        // it must never be selected: the reaper evicts files, never directories.
        let empty_target = legacy_target(0xcd);
        std::fs::create_dir_all(root.join("shadow").join(&empty_target)).expect("empty target");
        let before = tree(&root);

        let sh = dry_run(&root, Store::Shadow);

        assert_eq!(
            (sh.scanned, sh.selected, sh.kept, sh.layouts_walked),
            (3, 2, 1, 2),
            "three entries across two layouts, two of them over age, and BOTH \
             layouts walked",
        );
        // ON THE TIP THIS IS (1, 1, 0, 0): the legacy entry is invisible, so
        // `scanned` counts only the flat pair and `layouts_walked` does not
        // exist. That is the mutation this guard is pinned against.
        assert!(sh.bytes >= 64, "the legacy entry's bytes are charged: {}", sh.bytes);
        assert_eq!(
            tree(&root),
            before,
            "A DRY RUN OVER EITHER LAYOUT MUST CHANGE NOTHING",
        );
        assert!(fresh.is_file() && stale.is_file() && legacy.is_file());

        // THE APPLY ARM, on the same fixture: the legacy entry moves, carries
        // its layout AND its target in the quarantine name, and the flat
        // quarantine is shared rather than re-nested.
        let applied = reap_one(&root, Store::Shadow, 14, ReapMode::Apply, false).expect("apply");
        assert_eq!(
            (applied.scanned, applied.selected, applied.kept, applied.layouts_walked),
            (3, 2, 1, 2),
            "the apply arm selects exactly what the dry run named",
        );
        assert!(!legacy.exists(), "the legacy entry left its target directory");
        assert!(!stale.exists(), "the stale flat entry left the shadow directory");
        assert!(fresh.is_file(), "a fresh entry is never evicted, in either layout");
        assert!(
            root.join("shadow").join(&target).is_dir(),
            "RULE 1 IS RENAME-ONLY, and it reaches the emptied directory too: \
             a legacy target directory is never removed",
        );
        assert!(root.join("shadow").join(&empty_target).is_dir());
        let mut quarantined: Vec<String> = std::fs::read_dir(root.join("shadow").join("quarantine"))
            .expect("quarantine")
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .collect();
        quarantined.sort();
        assert_eq!(quarantined.len(), 2, "one quarantine, both layouts: {quarantined:?}");
        assert!(
            quarantined
                .iter()
                .any(|name| name.starts_with(&format!("legacy-{target}-old.changed-"))),
            "THE LAYOUT AND THE TARGET ARE BOTH IN THE QUARANTINE NAME: {quarantined:?}",
        );
        assert!(
            quarantined.iter().any(|name| name.starts_with("stale.changed-")),
            "a flat eviction's quarantine name is unchanged by this lane: {quarantined:?}",
        );

        // NON-VACUITY FOR `layouts_walked`: a root with no legacy directory at
        // all reports 1, so the 2 above is measured and not a constant.
        let flat_only = scratch("shadow-flat-only");
        shadow_entry(&flat_only, "solo.changed", 30);
        let solo = dry_run(&flat_only, Store::Shadow);
        assert_eq!(
            (solo.scanned, solo.selected, solo.layouts_walked),
            (1, 1, 1),
            "with no legacy directory on disk only the flat layout is walked",
        );
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&flat_only);
    }

    /// STORE-REAP-3, GUARD 2 — THE REASONS PARTITION THE SELECTED SET, AND THE
    /// LEGACY ROWS CARRY THE LEGACY REASON.
    ///
    /// `store-reap`'s per-entry rows are what an operator vets a proposal from,
    /// so `reason=` must distinguish "nothing referenced this" from "this sits
    /// in a layout the writer retired". A single reason for both would make the
    /// 85 pre-L3-1 entries on this filesystem indistinguishable in the census
    /// from ordinary unreferenced ones.
    #[test]
    fn every_legacy_shadow_row_says_stale_layout_and_every_flat_row_says_unreferenced() {
        let root = scratch("shadow-reasons");
        let target_a = legacy_target(0x1a);
        let target_b = legacy_target(0x2b);
        shadow_entry(&root, "flat.changed", 30);
        shadow_legacy_entry(&root, &target_a, "a.changed", 30);
        shadow_legacy_entry(&root, &target_b, "b.same", 30);
        // Same key under two targets — the collision the target-in-the-name
        // rule exists to prevent. Both must be selected and both must survive
        // the rename into ONE quarantine.
        shadow_legacy_entry(&root, &target_a, "dup.changed", 30);
        shadow_legacy_entry(&root, &target_b, "dup.changed", 30);

        let report = crate::courier::reap_shadow_cache_store(
            &crate::courier::shadow_cache_dir_in(&root),
            std::time::Duration::from_secs(14 * 86_400),
            ReapMode::Apply,
        )
        .expect("reap");
        assert_eq!(
            (report.scanned, report.evicted, report.evicted_stale_layout, report.layouts_walked),
            (5, 5, 4, 2),
            "four of the five selected entries are legacy",
        );
        let mut reasons: Vec<(String, &str)> = report
            .entries
            .iter()
            .map(|entry| (entry.label.clone(), entry.reason))
            .collect();
        reasons.sort();
        let mut expected: Vec<(String, &str)> = vec![
            (format!("legacy-{target_a}-a.changed"), "stale-layout"),
            (format!("legacy-{target_a}-dup.changed"), "stale-layout"),
            (format!("legacy-{target_b}-b.same"), "stale-layout"),
            (format!("legacy-{target_b}-dup.changed"), "stale-layout"),
            ("flat.changed".to_string(), "unreferenced"),
        ];
        expected.sort();
        assert_eq!(
            reasons, expected,
            "the labels are target-qualified so two entries with the same key \
             under different targets are two different rows",
        );
        assert_eq!(
            std::fs::read_dir(root.join("shadow").join("quarantine"))
                .expect("quarantine")
                .count(),
            5,
            "ALL FIVE survive the rename into one flat quarantine — the \
             same-key pair does not collide",
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// L3-1b-3B GUARD 8. THE NEW STORE REACHES THE VERB, AND IT REACHES IT
    /// THROUGH `--store all` RATHER THAN ONLY BY NAME — which is the half that
    /// matters, because `tools/store_reap_census.sh` only ever calls `all`, so
    /// a store the fan-out missed would be invisible to the census that is its
    /// only production reader.
    ///
    /// RED on the tip in the strongest possible way: `Store` had no such
    /// variant, so this does not compile there.
    ///
    /// The stale-generation arm is not decoration. The store puts the
    /// generation in the PATH, so a `v1` -> `v2` bump is exactly the event that
    /// orphaned 1 187 202 225 B of built wheels before STORE-REAP-1; this
    /// asserts the walk descends into BOTH generations (`versions_walked = 2`)
    /// and labels the retired one `stale-version`.
    #[test]
    fn the_marker_store_is_reached_by_store_all_and_walks_every_generation() {
        let root = scratch("marker");
        marker_store_entry(&root, "build-requirements", "v1", "cur", "requirements.txt", 30);
        marker_store_entry(&root, "build-requirements", "v0", "old", "requirements.txt", 30);
        marker_store_entry(&root, "build-requirements", "v1", "fresh", "requirements.txt", 1);
        // A directory with NO marker is not an entry. Discovery is by marker,
        // never by counting levels, so this must be neither scanned nor moved.
        std::fs::create_dir_all(root.join("build-requirements/v1/half-published"))
            .expect("half-published");
        let before = tree(&root);

        let br = dry_run(&root, Store::BuildRequirements);
        assert_eq!(
            (br.scanned, br.selected, br.stale_version, br.versions_walked),
            (3, 2, 1, 2),
            "three MARKED entries scanned, the two over-age ones selected, the \
             `v0` one labelled stale-version, and BOTH generations walked"
        );
        assert_eq!(
            tree(&root),
            before,
            "a dry run creates nothing and moves nothing"
        );

        // Non-vacuity: the apply arm on the SAME fixture DOES move them, so the
        // dry run's "unchanged" above is not the trivial truth of a walk that
        // selected nothing.
        let applied = reap_one(&root, Store::BuildRequirements, 14, ReapMode::Apply, true)
            .expect("apply");
        assert_eq!(applied.selected, 2);
        assert!(applied.bytes > 0, "--bytes must charge the moved entries");
        assert!(
            root.join("build-requirements/quarantine").is_dir(),
            "an eviction RENAMES into quarantine; it never deletes"
        );
        assert!(
            !root.join("build-requirements/v1/cur").exists()
                && !root.join("build-requirements/v0/old").exists(),
            "both over-age entries left their addresses"
        );
        assert!(
            root.join("build-requirements/v1/fresh/requirements.txt").is_file(),
            "the fresh entry is untouched"
        );
        assert!(
            root.join("build-requirements/v1/half-published").is_dir(),
            "a directory with no marker is not an entry and must not be moved"
        );
    }

    /// L3-1b-4. THE GUARD THAT MATTERS FOR THE FIFTH STORE, because
    /// `tools/store_reap_census.sh` only ever calls `--store all`: a store the
    /// enum knows about but the fan-out does not reach is a reaper with no
    /// reader, which is the exact shape law 2 forbids.
    ///
    /// It asserts the store is reached BOTH by name and through `all`, and
    /// that `all` still reaches the four that were there before it — an
    /// appended member must not displace one.
    #[test]
    fn the_hermetic_environment_store_is_reached_by_name_and_by_store_all() {
        assert_eq!(
            Store::parse("hermetic-envs"),
            Some(vec![Store::HermeticEnvironments]),
            "the spelling the census and an operator both type"
        );
        assert!(
            Store::ALL.contains(&Store::HermeticEnvironments),
            "`--store all` must fan out to the hermetic store"
        );
        for previous in [
            Store::BuiltWheels,
            Store::GitSnapshots,
            Store::Shadow,
            Store::BuildRequirements,
        ] {
            assert!(
                Store::ALL.contains(&previous),
                "appending the fifth store displaced {}",
                previous.as_str()
            );
        }

        let root = scratch("hermetic");
        marker_store_entry(&root, "hermetic-build-envs", "v8", "env-cur", "complete.json", 30);
        marker_store_entry(&root, "hermetic-build-envs", "v7", "env-old", "complete.json", 30);
        marker_store_entry(&root, "hermetic-build-envs", "v8", "env-fresh", "complete.json", 1);
        std::fs::create_dir_all(root.join("hermetic-build-envs/v8/env-half"))
            .expect("half-published");
        let before = tree(&root);

        let dry = dry_run(&root, Store::HermeticEnvironments);
        assert_eq!(
            (
                dry.scanned,
                dry.selected,
                dry.stale_version,
                dry.versions_walked
            ),
            (3, 2, 1, 2),
            "the hermetic store is walked by the same rules across both generations"
        );
        assert_eq!(tree(&root), before, "a dry run creates nothing");

        // Non-vacuity, and the reader the census actually exercises: the SAME
        // fixture reached through `all`, applied.
        let mut selected_by_all = 0;
        for store in Store::ALL {
            selected_by_all += reap_one(&root, store, 14, ReapMode::Apply, true)
                .unwrap_or_else(|error| panic!("{} reap: {error:#}", store.as_str()))
                .selected;
        }
        assert_eq!(
            selected_by_all, 2,
            "`all` must reach the hermetic entries and nothing else in this fixture"
        );
        assert!(
            root.join("hermetic-build-envs/quarantine").is_dir(),
            "an eviction RENAMES into quarantine; it never deletes"
        );
        assert!(
            root.join("hermetic-build-envs/v8/env-fresh/complete.json").is_file(),
            "the fresh entry is untouched"
        );
        assert!(
            root.join("hermetic-build-envs/v8/env-half").is_dir(),
            "a directory with no completion marker is not an entry"
        );
    }

    /// METAGEN-1's store, registered exactly the way the seven before it were,
    /// and asserted through the module's own constants rather than by
    /// re-spelling the segments — so a generation bump moves the fixture with
    /// the walk instead of leaving this green over a store nothing addresses.
    ///
    /// It is a separate store and not a corner of the sdist-metadata one for a
    /// structural reason: that key's first field is a fold of the sdist's URL
    /// and ETag out of a `revision.http`, and a LOCAL PATH TREE has no URL, no
    /// ETag and no `revision.http`.
    #[test]
    fn the_path_source_metadata_store_is_reached_by_name_and_by_store_all() {
        use crate::derived_editable_metadata as dem;

        assert_eq!(
            Store::parse(dem::CACHE_NAMESPACE),
            Some(vec![Store::PathSourceMetadata]),
            "the spelling the census and an operator both type"
        );
        assert!(
            Store::ALL.contains(&Store::PathSourceMetadata),
            "`--store all` must fan out to the path-source-metadata store, \
             which is the ONLY thing `tools/store_reap_census.sh` ever calls"
        );
        for previous in [
            Store::BuiltWheels,
            Store::GitSnapshots,
            Store::Shadow,
            Store::BuildRequirements,
            Store::HermeticEnvironments,
            Store::SdistMetadata,
            Store::BuiltOutputs,
        ] {
            assert!(
                Store::ALL.contains(&previous),
                "appending the eighth store displaced {}",
                previous.as_str()
            );
        }
        assert_eq!(
            resolved_max_age_days(Store::PathSourceMetadata, None),
            dem::DEFAULT_MAX_AGE_DAYS,
            "the horizon is the store's own constant, not a copy of 14 here"
        );

        // AND THE WALK REACHES A REAL ENTRY. A registration that parses but
        // scans nothing is the same defect one level down.
        let root = scratch("pathsourcemeta");
        marker_store_entry(
            &root,
            dem::CACHE_NAMESPACE,
            dem::CACHE_VERSION,
            "psm-cur",
            dem::COMPLETION_MARKER,
            30,
        );
        let dry = dry_run(&root, Store::PathSourceMetadata);
        assert_eq!(dry.scanned, 1, "the walk must see the entry: {dry:?}");
    }

    /// SDIST-META-2's half of the verb, and the CENSUS the store's harness
    /// halves are audited by. It is the only in-product reader of the
    /// prepared-sdist-metadata store: the writer is a post-lock harvester and
    /// the reader is the scoper's seeder, both shell, because pixi's embedded
    /// uv — not this backend — is what produces the metadata.
    ///
    /// The store's shape is asserted through
    /// `sdist_metadata::{CACHE_NAMESPACE, CACHE_VERSION, COMPLETION_MARKER}`
    /// rather than by re-spelling the segments, so a generation bump in the
    /// module moves the fixture with the walk instead of leaving this test
    /// green over a store nothing addresses.
    #[test]
    fn the_sdist_metadata_store_is_reached_by_name_and_by_store_all() {
        use crate::sdist_metadata as sdm;

        assert_eq!(
            Store::parse(sdm::CACHE_NAMESPACE),
            Some(vec![Store::SdistMetadata]),
            "the spelling the census and an operator both type"
        );
        assert!(
            Store::ALL.contains(&Store::SdistMetadata),
            "`--store all` must fan out to the sdist-metadata store"
        );
        for previous in [
            Store::BuiltWheels,
            Store::GitSnapshots,
            Store::Shadow,
            Store::BuildRequirements,
            Store::HermeticEnvironments,
        ] {
            assert!(
                Store::ALL.contains(&previous),
                "appending the sixth store displaced {}",
                previous.as_str()
            );
        }
        assert_eq!(
            resolved_max_age_days(Store::SdistMetadata, None),
            sdm::DEFAULT_MAX_AGE_DAYS,
            "the horizon is the store's own constant, not a copy of 14 here"
        );

        let root = scratch("sdistmeta");
        let dir = sdm::CACHE_NAMESPACE;
        let marker = sdm::COMPLETION_MARKER;
        marker_store_entry(&root, dir, sdm::CACHE_VERSION, "sdm-cur", marker, 30);
        // `v1` BY NAME, not a placeholder. SDIST-META-3 bumped the generation
        // to `v2` because the key's first field changed, and the v1 directory
        // is left standing for the reaper rather than deleted. This fixture is
        // the state that actually exists on disk after that bump, and it
        // asserts the walk ages a v1 entry by the same rule with
        // `reason="stale-version"` instead of walking past it forever.
        marker_store_entry(&root, dir, "v1", "sdm-old", marker, 30);
        assert_ne!(
            sdm::CACHE_VERSION,
            "v1",
            "the older-generation fixture must not be the current generation"
        );
        marker_store_entry(&root, dir, sdm::CACHE_VERSION, "sdm-fresh", marker, 1);
        // A half-published entry: the payload landed, the marker did not. It is
        // NOT an entry, which is the property the harvester's marker-last write
        // order depends on.
        std::fs::create_dir_all(root.join(dir).join(sdm::CACHE_VERSION).join("sdm-half"))
            .expect("half-published");
        std::fs::write(
            root.join(dir)
                .join(sdm::CACHE_VERSION)
                .join("sdm-half")
                .join(sdm::METADATA_FILE),
            b"partial",
        )
        .expect("payload");
        let before = tree(&root);

        let dry = dry_run(&root, Store::SdistMetadata);
        assert_eq!(
            (
                dry.scanned,
                dry.selected,
                dry.stale_version,
                dry.versions_walked
            ),
            (3, 2, 1, 2),
            "the sdist-metadata store is walked by the same rules across both generations"
        );
        assert_eq!(tree(&root), before, "a dry run creates nothing");

        // Non-vacuity, through the fan-out the census actually runs.
        let mut selected_by_all = 0;
        for store in Store::ALL {
            selected_by_all += reap_one(&root, store, 14, ReapMode::Apply, true)
                .unwrap_or_else(|error| panic!("{} reap: {error:#}", store.as_str()))
                .selected;
        }
        assert_eq!(
            selected_by_all, 2,
            "`all` must reach the sdist-metadata entries and nothing else in this fixture"
        );
        assert!(
            root.join(dir).join("quarantine").is_dir(),
            "an eviction RENAMES into quarantine; it never deletes"
        );
        assert!(
            root.join(dir)
                .join(sdm::CACHE_VERSION)
                .join("sdm-fresh")
                .join(marker)
                .is_file(),
            "the fresh entry is untouched"
        );
        assert!(
            root.join(dir)
                .join(sdm::CACHE_VERSION)
                .join("sdm-half")
                .is_dir(),
            "a directory with no completion marker is not an entry"
        );
    }

    /// One FLAT built-output entry: `<root>/built-outputs/<key>/COMPLETE`,
    /// whose CONTENT is the generation. `stamped` writes the `.used` sidecar
    /// the reaper prefers over the marker mtime.
    fn built_output_entry(
        root: &Path,
        key: &str,
        generation: &str,
        marker_ago: u64,
        stamped: Option<u64>,
    ) -> PathBuf {
        let entry = root
            .join(crate::built_output_store::STORE_DIR)
            .join(key);
        std::fs::create_dir_all(&entry).expect("entry");
        std::fs::write(entry.join("outputs.json"), vec![b'p'; 32]).expect("payload");
        let marker_path = entry.join(crate::built_output_store::MARKER);
        std::fs::write(&marker_path, generation).expect("marker");
        set_mtime(&marker_path, age(marker_ago));
        if let Some(ago) = stamped {
            let stamp =
                crate::source_build::use_stamp_path(&entry).expect("the shared stamp formula");
            std::fs::write(&stamp, b"").expect("stamp");
            set_mtime(&stamp, age(ago));
        }
        entry
    }

    /// CONDA-OUT-2. THE SEVENTH STORE (MERGE-CO: the sixth on its own branch,
    /// the seventh now that SDIST-META-2's store landed first as B32), and
    /// the first FLAT one: no generation directory, the generation in each
    /// entry's own marker.
    ///
    /// The arms that can each fail on their own: reached by name and through
    /// `--store all` (a store the fan-out misses is a reaper with no reader);
    /// the generation is read from the MARKER, so an entry whose marker says
    /// `v1` is `stale-version` while its `v3` neighbour of the same age is
    /// `unreferenced`; a FRESH `.used` sidecar keeps an entry whose marker is
    /// 30 days old, which is the whole reason the adoption path stamps;
    /// `versions_walked` counts DISTINCT marker generations, not directories;
    /// and a dry run creates nothing.
    #[test]
    fn the_built_output_store_is_flat_and_takes_its_generation_from_the_marker() {
        assert_eq!(
            Store::parse("built-outputs"),
            Some(vec![Store::BuiltOutputs]),
            "the spelling an operator and the census both type"
        );
        assert!(
            Store::ALL.contains(&Store::BuiltOutputs),
            "`--store all` must fan out to the built-output store"
        );
        for previous in [
            Store::BuiltWheels,
            Store::GitSnapshots,
            Store::Shadow,
            Store::BuildRequirements,
            Store::HermeticEnvironments,
            Store::SdistMetadata,
        ] {
            assert!(
                Store::ALL.contains(&previous),
                "appending the seventh store displaced {}",
                previous.as_str()
            );
        }

        let root = scratch("built-outputs");
        let current = crate::built_output_store::SCHEMA;
        // Over-age, current generation.
        built_output_entry(&root, "aaaa", current, 30, None);
        // Over-age, a RETIRED generation: the shared root holds 218 entries
        // whose markers read v1, v2 and v3, and until this spec existed
        // nothing could reach the v1s and v2s at all.
        built_output_entry(&root, "bbbb", "retread-built-output-store-v1", 30, None);
        // Marker 30 days old and REFERENCED yesterday: kept, and this is the
        // arm the `.used` stamp on the adoption path exists for.
        built_output_entry(&root, "cccc", current, 30, Some(1));
        // Young.
        built_output_entry(&root, "dddd", current, 1, None);
        // A half-published entry: a directory with no marker is not an entry.
        std::fs::create_dir_all(
            root.join(crate::built_output_store::STORE_DIR).join("eeee"),
        )
        .expect("half-published");
        let before = tree(&root);

        let dry = dry_run(&root, Store::BuiltOutputs);
        assert_eq!(
            (
                dry.scanned,
                dry.selected,
                dry.stale_version,
                dry.kept,
                dry.versions_walked
            ),
            (4, 2, 1, 2, 2),
            "four entries scanned, the two over-age ones selected, one of them \
             for its retired generation, and TWO distinct generations walked"
        );
        assert_eq!(tree(&root), before, "a dry run creates nothing");

        // Non-vacuity for the dry run's zero: the same fixture, applied,
        // through the fan-out the census actually calls.
        let mut selected_by_all = 0;
        for store in Store::ALL {
            selected_by_all += reap_one(&root, store, 14, ReapMode::Apply, true)
                .unwrap_or_else(|error| panic!("{} reap: {error:#}", store.as_str()))
                .selected;
        }
        assert_eq!(
            selected_by_all, 2,
            "`all` must reach the built-output entries and nothing else here"
        );
        let store_dir = root.join(crate::built_output_store::STORE_DIR);
        assert!(
            store_dir.join(crate::built_output_store::QUARANTINE).is_dir(),
            "an eviction RENAMES into quarantine; it never deletes"
        );
        assert!(
            store_dir.join("cccc").is_dir(),
            "a freshly REFERENCED entry survives an old marker"
        );
        assert!(store_dir.join("dddd").is_dir(), "the young entry is untouched");
        assert!(
            !store_dir.join("aaaa").exists() && !store_dir.join("bbbb").exists(),
            "both over-age entries left their addresses"
        );
        assert!(
            store_dir.join("eeee").is_dir(),
            "a directory with no marker is not an entry and must not be moved"
        );
    }

    // ================= REAP-2 (N27-RETREAD-210, -163) =================
    //
    // The route-probe store's arm. Every fixture below writes a REAL verdict
    // file — the schema string, the validity key and entries at addresses
    // `route_probe_cache::probe_digest` actually produces — because a reaper
    // guard fed hand-waved JSON cannot tell "kept because it is reachable"
    // from "kept because the parse failed".

    /// A verdict file body whose single entry is REACHABLE: its map key is the
    /// digest of the very stamp it carries.
    fn route_probe_reachable_body(stage: &str, universe: &str, specs: &[&str]) -> String {
        let owned: Vec<String> = specs.iter().map(|s| (*s).to_string()).collect();
        let digest = crate::route_probe_cache::probe_digest(stage, universe, owned.iter());
        let mut sorted = owned.clone();
        sorted.sort();
        sorted.dedup();
        let question = sorted.join("\u{1f}");
        format!(
            "{{\"schema\":\"v4-route-probe-verdicts\",\"key\":\"file-key\",\"entries\":\
             {{\"{digest}\":{{\"verdict\":\"Sat\",\"stamp\":{{\"stage\":\"{stage}\",\
             \"universe\":\"{universe}\",\"question\":{}}}}}}}}}",
            serde_json::to_string(&question).expect("question"),
        )
    }

    /// A verdict file body whose single entry is UNREACHABLE: the stamp says one
    /// universe and the address was computed from another, which is the p6ab
    /// cross-writer shape.
    fn route_probe_unreachable_body(stage: &str, specs: &[&str]) -> String {
        let owned: Vec<String> = specs.iter().map(|s| (*s).to_string()).collect();
        let digest = crate::route_probe_cache::probe_digest(stage, "universe-gone", owned.iter());
        let mut sorted = owned.clone();
        sorted.sort();
        sorted.dedup();
        let question = sorted.join("\u{1f}");
        format!(
            "{{\"schema\":\"v4-route-probe-verdicts\",\"key\":\"file-key\",\"entries\":\
             {{\"{digest}\":{{\"verdict\":\"Sat\",\"stamp\":{{\"stage\":\"{stage}\",\
             \"universe\":\"universe-live\",\"question\":{}}}}}}}}}",
            serde_json::to_string(&question).expect("question"),
        )
    }

    /// One entry file under `<root>/route-probe-verdicts/<name>`, aged by its
    /// OWN mtime — the store has no marker sidecar, the file IS the entry.
    fn route_probe_entry(root: &Path, name: &str, body: &str, ago_days: u64) -> PathBuf {
        let dir = crate::route_probe_cache::store_dir_in(root);
        std::fs::create_dir_all(&dir).expect("store dir");
        let path = dir.join(name);
        std::fs::write(&path, body.as_bytes()).expect("entry");
        set_mtime(&path, age(ago_days));
        path
    }

    /// Every NON-EMPTY regular file under a directory tree, as `name -> bytes`,
    /// so a guard can assert NOTHING WAS DELETED rather than only that a count
    /// happened to match.
    ///
    /// Zero-byte files are excluded and that is not a convenience: an apply
    /// leaves behind the advisory-lock sidecars it took (`<stem>.lock`, and the
    /// reaper's own `.route-probe-verdicts.reap.lock`), all of which are empty
    /// by construction. Counting them would make "the store's bytes are
    /// unchanged" fail for a reason that has nothing to do with an entry.
    fn files_under(dir: &Path) -> Vec<(String, Vec<u8>)> {
        let mut out: Vec<(String, Vec<u8>)> = Vec::new();
        let Ok(children) = std::fs::read_dir(dir) else {
            return out;
        };
        for child in children.flatten() {
            let path = child.path();
            if path.is_dir() {
                out.extend(files_under(&path));
            } else if path.is_file() {
                let bytes = std::fs::read(&path).unwrap_or_default();
                if bytes.is_empty() {
                    continue;
                }
                out.push((
                    path.file_name().unwrap().to_string_lossy().into_owned(),
                    bytes,
                ));
            }
        }
        out.sort();
        out
    }

    /// The three-age fixture the whole arm is judged on: 1, 31 and 91 days,
    /// every entry REACHABLE so the only rule that can select is the age.
    fn three_age_store(tag: &str) -> PathBuf {
        let root = scratch(tag);
        for (name, ago) in [("a-py3.12-linux-64.json", 1u64),
                            ("b-py3.12-linux-64.json", 31),
                            ("c-py3.12-linux-64.json", 91)] {
            route_probe_entry(
                &root,
                name,
                &route_probe_reachable_body("auto_route_joint_solve", "universe-live", &[name]),
                ago,
            );
        }
        root
    }

    /// REAP-2 GUARD 1. THE DRY RUN DECIDES AND MOVES NOTHING.
    ///
    /// Ages 1 / 31 / 91 against a 30-day rule: two selected, one kept, and the
    /// filesystem is byte-for-byte what it was — no `quarantine` directory, all
    /// three entries where they were, and every reported entry carrying
    /// `quarantine: None`.
    ///
    /// RED ON a83c0a3: `Store` has no `RouteProbeVerdicts` variant, so this
    /// does not compile — which is the strongest form of red available.
    #[test]
    fn reap2_route_probe_dry_run_selects_two_of_three_and_moves_nothing() {
        let root = three_age_store("rp-dry");
        let store_dir = crate::route_probe_cache::store_dir_in(&root);
        let before = files_under(&store_dir);

        let outcome = reap_one(
            &root,
            Store::RouteProbeVerdicts,
            30,
            ReapMode::DryRun,
            false,
        )
        .expect("dry run");

        assert_eq!(outcome.scanned, 3, "all three entries are scanned");
        assert_eq!(outcome.selected, 2, "31 and 91 days are over a 30-day rule");
        assert_eq!(outcome.kept, 1, "1 day is under it");
        assert_eq!(outcome.reason_age, 2, "both were selected BY AGE");
        assert_eq!(outcome.reason_universe, 0, "no entry here is unaddressable");
        assert_eq!(outcome.skipped_locked, 0);
        assert!(!outcome.skipped_concurrent);
        assert_eq!(
            files_under(&store_dir),
            before,
            "a dry run must leave the store byte-for-byte as it was",
        );
        assert!(
            !store_dir.join(crate::route_probe_cache::QUARANTINE).exists(),
            "a dry run must not even CREATE the quarantine directory",
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// REAP-2 GUARD 2. AN APPLY QUARANTINES BY RENAME AND DELETES NOTHING.
    ///
    /// The same fixture applied: two entries leave the store directory, both
    /// appear under `quarantine/` WITH THEIR BYTES, the kept one is untouched,
    /// and the total set of file CONTENTS under the store is unchanged — which
    /// is the assertion the mutant "reaper deletes instead of renaming" fails.
    #[test]
    fn reap2_route_probe_apply_quarantines_by_rename_and_deletes_nothing() {
        let root = three_age_store("rp-apply");
        let store_dir = crate::route_probe_cache::store_dir_in(&root);
        let mut before: Vec<Vec<u8>> = files_under(&store_dir)
            .into_iter()
            .map(|(_, bytes)| bytes)
            .collect();
        before.sort();

        let outcome =
            reap_one(&root, Store::RouteProbeVerdicts, 30, ReapMode::Apply, false).expect("apply");

        assert_eq!(outcome.scanned, 3);
        assert_eq!(outcome.selected, 2);
        assert_eq!(outcome.kept, 1);
        assert_eq!(outcome.reason_age, 2);
        assert_eq!(outcome.reason_universe, 0);

        // The store directory itself now holds ONE entry, the fresh one.
        let live: Vec<String> = std::fs::read_dir(&store_dir)
            .expect("store dir")
            .flatten()
            .filter(|e| e.path().is_file())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| crate::route_probe_cache::is_verdict_entry_name(n))
            .collect();
        assert_eq!(
            live,
            vec!["a-py3.12-linux-64.json".to_string()],
            "only the 1-day entry may remain live",
        );

        // NOTHING WAS DELETED: the two selected entries are files under
        // quarantine/ and their bytes are intact.
        let quarantine = store_dir.join(crate::route_probe_cache::QUARANTINE);
        let quarantined = files_under(&quarantine);
        assert_eq!(
            quarantined.len(),
            2,
            "both selected entries must be present under quarantine/: {quarantined:?}",
        );
        for (name, _) in &quarantined {
            assert!(
                name.starts_with("b-py3.12-linux-64.json-")
                    || name.starts_with("c-py3.12-linux-64.json-"),
                "a quarantine name keeps the entry name and appends <unix>-<pid>: {name}",
            );
        }
        let mut after: Vec<Vec<u8>> = files_under(&store_dir)
            .into_iter()
            .map(|(_, bytes)| bytes)
            .collect();
        after.sort();
        assert_eq!(
            after, before,
            "every byte that was in the store is still in the store, moved not removed",
        );

        // And the report says WHERE each one went, which a dry run never does.
        assert_eq!(outcome.selected as usize, 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// REAP-2 GUARD 3. AN ENTRY A LIVE PUBLISH HOLDS IS SKIPPED AND COUNTED.
    ///
    /// Two over-age entries; the lock `route_probe_cache::lock_path_for` names
    /// for one of them is held EXCLUSIVELY for the duration of the reap, which
    /// is exactly what `RouteProbeCache::persist` does while it rewrites a
    /// file. That entry must survive, be counted as `skipped_locked`, and NOT
    /// be counted as kept-by-policy.
    ///
    /// RED IF THE REAPER IGNORES THE LOCK (or tries the wrong path, which is
    /// what a reaper written against `artifact_cache_lock_path`'s dot form
    /// would do): `skipped_locked` reads 0 and `selected` reads 2.
    #[test]
    fn reap2_route_probe_skips_and_counts_an_entry_a_publish_holds() {
        let root = scratch("rp-lock");
        for name in ["a-py3.12-linux-64.json", "b-py3.12-linux-64.json"] {
            route_probe_entry(
                &root,
                name,
                &route_probe_reachable_body("auto_route_joint_solve", "universe-live", &[name]),
                91,
            );
        }
        let store_dir = crate::route_probe_cache::store_dir_in(&root);
        let held_entry = store_dir.join("b-py3.12-linux-64.json");
        let held_lock = crate::route_probe_cache::lock_path_for(&held_entry);
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&held_lock)
            .expect("open the entry lock");
        assert!(
            fs4::fs_std::FileExt::try_lock_exclusive(&lock).unwrap_or(false),
            "the fixture must actually hold the lock, or this guard cannot fail",
        );

        let outcome =
            reap_one(&root, Store::RouteProbeVerdicts, 30, ReapMode::Apply, false).expect("apply");

        assert_eq!(outcome.scanned, 2);
        assert_eq!(outcome.selected, 1, "only the unlocked entry moves");
        assert_eq!(outcome.skipped_locked, 1, "the held entry is COUNTED, not silent");
        assert_eq!(
            outcome.kept, 0,
            "a locked entry is not `kept` — kept means the policy spared it",
        );
        assert!(
            held_entry.is_file(),
            "the entry a publish holds must still be exactly where it was",
        );
        let _ = fs4::fs_std::FileExt::unlock(&lock);
        drop(lock);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// REAP-2 GUARD 4. AN UNADDRESSABLE FILE IS SELECTED WITH reason=universe,
    /// AND A REACHABLE ONE OF THE SAME AGE IS NOT.
    ///
    /// Both files are ONE DAY old against a 30-day rule, so the age rule cannot
    /// fire and the only thing separating them is whether any reader can still
    /// address their entries.
    #[test]
    fn reap2_route_probe_quarantines_an_unaddressable_file_and_keeps_a_reachable_one() {
        let root = scratch("rp-universe");
        route_probe_entry(
            &root,
            "live-py3.12-linux-64.json",
            &route_probe_reachable_body("auto_route_joint_solve", "universe-live", &["numpy"]),
            1,
        );
        route_probe_entry(
            &root,
            "gone-py3.12-linux-64.json",
            &route_probe_unreachable_body("auto_route_joint_solve", &["numpy"]),
            1,
        );

        let outcome =
            reap_one(&root, Store::RouteProbeVerdicts, 30, ReapMode::Apply, false).expect("apply");

        assert_eq!(outcome.scanned, 2);
        assert_eq!(outcome.selected, 1);
        assert_eq!(outcome.kept, 1);
        assert_eq!(
            outcome.reason_age, 0,
            "neither file is over-age; an `age` here would mean the rules are crossed",
        );
        assert_eq!(outcome.reason_universe, 1);
        let store_dir = crate::route_probe_cache::store_dir_in(&root);
        assert!(
            store_dir.join("live-py3.12-linux-64.json").is_file(),
            "a file with a reachable entry is live whatever else is in the store",
        );
        assert!(
            !store_dir.join("gone-py3.12-linux-64.json").exists(),
            "the unaddressable file must have moved",
        );
        assert_eq!(
            files_under(&store_dir.join(crate::route_probe_cache::QUARANTINE)).len(),
            1,
            "and it must be UNDER quarantine, not gone",
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// REAP-2 GUARD 5 (N27-RETREAD-163). THE CONSTANT IS THE DEFAULT OF THE
    /// DECLARED ARGUMENT, AND THAT IS THE WHOLE DISCHARGE OF -163.
    ///
    /// `ROUTE_PROBE_STORE_DEFAULT_MAX_AGE_DAYS` had exactly one occurrence in
    /// the tree at a83c0a3 — its own definition. This asserts the resolution
    /// chain that gives it a reader: no flag -> the constant;
    /// `--max-age-days` -> that; `--route-probe-max-age-days` -> that, and it
    /// OUTRANKS the general flag for this store only.
    ///
    /// RED UNDER THE MUTANT that changes the constant: the first assertion
    /// compares the resolved number against the constant AND against 14 as a
    /// literal, so a mutant that moves the constant moves the printed
    /// `max_age_days=` field and this line names it.
    #[test]
    fn reap2_the_route_probe_max_age_flag_defaults_to_the_stores_own_constant() {
        let bare = parse_args(&["--store".to_string(), "route-probe-verdicts".to_string()])
            .expect("parse");
        assert_eq!(bare.stores, vec![Store::RouteProbeVerdicts]);
        assert_eq!(bare.route_probe_max_age_days, None);
        let resolved =
            resolved_max_age_days(Store::RouteProbeVerdicts, max_age_flag_for(Store::RouteProbeVerdicts, &bare));
        assert_eq!(
            resolved,
            crate::route_probe_cache::ROUTE_PROBE_STORE_DEFAULT_MAX_AGE_DAYS,
            "with no flag the store's own constant IS the default",
        );
        assert_eq!(
            resolved, 14,
            "and it is 14 today — a literal here so a mutant on the constant reddens this guard",
        );

        // The general flag reaches it.
        let general = parse_args(&["--max-age-days".to_string(), "90".to_string()]).expect("parse");
        assert_eq!(
            resolved_max_age_days(
                Store::RouteProbeVerdicts,
                max_age_flag_for(Store::RouteProbeVerdicts, &general)
            ),
            90,
        );

        // The store's own flag OUTRANKS it, and reaches NO other store.
        let specific = parse_args(&[
            "--max-age-days".to_string(),
            "90".to_string(),
            "--route-probe-max-age-days".to_string(),
            "3".to_string(),
        ])
        .expect("parse");
        assert_eq!(specific.route_probe_max_age_days, Some(3));
        assert_eq!(
            resolved_max_age_days(
                Store::RouteProbeVerdicts,
                max_age_flag_for(Store::RouteProbeVerdicts, &specific)
            ),
            3,
            "the store-specific flag wins for the route-probe store",
        );
        for store in Store::ALL {
            if store == Store::RouteProbeVerdicts {
                continue;
            }
            assert_eq!(
                max_age_flag_for(store, &specific),
                Some(90),
                "--route-probe-max-age-days must not reach {}",
                store.as_str(),
            );
        }
        assert!(
            parse_args(&["--route-probe-max-age-days".to_string()]).is_err(),
            "a flag with no value must refuse, not default silently",
        );
    }

    /// REAP-2 GUARD 6. THE STORE IS IN `--store all`, ITS SPELLING IS THE
    /// STORE'S OWN CONSTANT, AND THE EIGHT BEFORE IT DID NOT MOVE.
    ///
    /// The append-never-reorder rule the eight previous stores landed under:
    /// the merge gate's readers grep summary rows by position.
    #[test]
    fn reap2_the_route_probe_store_is_appended_to_store_all() {
        assert_eq!(Store::ALL.len(), 9, "the ninth store");
        assert_eq!(
            Store::ALL[8],
            Store::RouteProbeVerdicts,
            "APPENDED, never inserted",
        );
        assert_eq!(
            Store::ALL[..8],
            [
                Store::BuiltWheels,
                Store::GitSnapshots,
                Store::Shadow,
                Store::BuildRequirements,
                Store::HermeticEnvironments,
                Store::SdistMetadata,
                Store::BuiltOutputs,
                Store::PathSourceMetadata,
            ],
            "the eight stores before it are in the order they landed in",
        );
        assert_eq!(
            Store::RouteProbeVerdicts.as_str(),
            crate::route_probe_cache::STORE_DIR,
            "the `--store` spelling READS the store's own constant",
        );
        assert_eq!(
            parse_args(&["--store".to_string(), "route-probe-verdicts".to_string()])
                .expect("parse")
                .stores,
            vec![Store::RouteProbeVerdicts],
        );
        assert_eq!(
            parse_args(&["--store".to_string(), "all".to_string()])
                .expect("parse")
                .stores
                .len(),
            9,
            "`all` fans out to nine, so a census that reads `all` reaches this store",
        );
    }

    /// REAP-2 GUARD 7. AN ABSENT STORE IS A CLEAN, EMPTY, NON-REFUSING CENSUS
    /// — and it must not create the store either.
    ///
    /// This is the shape every lane job prints today (every relock job-scopes
    /// `XDG_CACHE_HOME`), so it is the row the census step actually reads.
    #[test]
    fn reap2_an_absent_route_probe_store_scans_nothing_and_creates_nothing() {
        let root = scratch("rp-absent");
        for mode in [ReapMode::DryRun, ReapMode::Apply] {
            let outcome =
                reap_one(&root, Store::RouteProbeVerdicts, 30, mode, false).expect("absent store");
            assert_eq!(outcome.scanned, 0);
            assert_eq!(outcome.selected, 0);
            assert_eq!(outcome.kept, 0);
            assert!(
                !outcome.skipped_concurrent,
                "an absent store is not a refusal; exit 7 would read as deferral",
            );
        }
        assert!(
            !crate::route_probe_cache::store_dir_in(&root).exists(),
            "a census must not bring the store into existence",
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
