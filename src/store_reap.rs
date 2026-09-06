//! STORE-REAP-2: `retread store-reap` — THE PRODUCTION CALL SITE OF THE THREE
//! PERSISTENT-STORE REAPERS.
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
}

impl Store {
    /// The `--store` spelling, which is also the `store=` field of every row.
    pub fn as_str(self) -> &'static str {
        match self {
            Store::BuiltWheels => "built-wheels",
            Store::GitSnapshots => "git-snapshots",
            Store::Shadow => "shadow",
        }
    }

    /// ALL of them, in the order the rows print. `--store all` is exactly this
    /// slice, never a re-listing of the names somewhere else.
    pub const ALL: [Store; 3] = [Store::BuiltWheels, Store::GitSnapshots, Store::Shadow];

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
        }
    }
}

/// THE ONE RESOLUTION of "how old is too old" for one store in one run.
/// `flag` is `--max-age-days` when the operator stated one. Extracted so the
/// override is a function a guard can call, not a line inside a loop.
pub fn resolved_max_age_days(store: Store, flag: Option<u64>) -> u64 {
    flag.unwrap_or_else(|| store.default_max_age_days())
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
                         built-wheels, git-snapshots, shadow, all"
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
            "--bytes" => bytes = true,
            other => anyhow::bail!("store-reap: unknown arg {other}"),
        }
    }
    Ok(Args {
        stores: stores.unwrap_or_else(|| Store::ALL.to_vec()),
        roots,
        mode,
        max_age_days,
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
    pub skipped_concurrent: bool,
    pub bytes: u64,
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
            let days = resolved_max_age_days(store, args.max_age_days);
            let outcome = reap_one(&root, store, days, args.mode, args.bytes)?;
            refused |= outcome.skipped_concurrent;
            total_scanned += outcome.scanned;
            total_selected += outcome.selected;
            total_bytes += outcome.bytes;
            println!(
                "### store-reap SUMMARY root={} store={} mode={} max_age_days={} \
                 scanned={} {}={} stale_version={} kept={} skipped_locked={} \
                 versions_walked={} skipped_concurrent={} bytes={}",
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
                outcome.skipped_concurrent,
                outcome.bytes,
            );
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
            // The shadow reaper takes the shadow DIRECTORY, not the store root:
            // its entries are files in one flat directory with no generation
            // segment (L3-1b-1a-1 ruled that walk deliberately unchanged).
            let report = crate::courier::reap_shadow_cache_store(
                &crate::courier::shadow_cache_dir_in(root),
                max_age,
                mode,
            )?;
            outcome.scanned = report.scanned;
            outcome.selected = report.evicted;
            outcome.kept = report.kept;
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

    /// One shadow entry: a FILE in `<root>/shadow`.
    fn shadow_entry(root: &Path, name: &str, ago: u64) -> PathBuf {
        let dir = root.join("shadow");
        std::fs::create_dir_all(&dir).expect("shadow dir");
        let entry = dir.join(name);
        std::fs::write(&entry, vec![b's'; 16]).expect("payload");
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
    /// is exactly the three stores, an unknown store is a refusal and not a
    /// silent skip, and `--root` accumulates.
    #[test]
    fn the_parse_defaults_to_a_dry_run_of_all_three_stores_and_refuses_a_bad_name() {
        let empty = parse_args(&[]).expect("no args");
        assert_eq!(empty.mode, ReapMode::DryRun, "THE DEFAULT IS THE SAFE ONE");
        assert_eq!(empty.stores, Store::ALL.to_vec());
        assert!(empty.roots.is_empty() && !empty.bytes && empty.max_age_days.is_none());

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
}
