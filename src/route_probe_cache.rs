//! On-disk memo for conda route-probe verdicts (fix f17).
//!
//! Motivation. Every rejected route probe in
//! [`crate::handler::auto_bundle`] costs one full conda co-solve plus a
//! serial PyPI restore fetch, and the whole batch runs inside pixi's
//! single conda-solve permit with no log output. `isaac-pack-latest`
//! (python 3.12) executed 100 probes -- all
//! `joint-co-solve-rejected-to-pypi`, all `matching_candidates: 0` --
//! on EVERY lock, cold and warm, because nothing on disk remembered the
//! previous run's verdicts. The heal-facts memo
//! (`crate::uv_closure::HealFacts`) is a different cache: it replays the
//! HEALED closure, and it only ever existed for the packs that had one
//! (`~/.cache/rattler/retread-heal-facts/` holds
//! `isaaclab-2-3x-pack-py3.11-linux-64.json` and three others -- no
//! `isaac-pack-latest` file at all), so that pack fell through to a cold
//! probe storm every time.
//!
//! This module memoizes the co-solve VERDICT itself, keyed by everything
//! that can change the answer:
//!   * the normalized probe spec set (the question),
//!   * the channel set + repodata identity (the candidate universe),
//!   * the target python / platform subdir,
//!   * a policy fingerprint (channel priority, system requirements,
//!     detected virtual packages, workspace deps, workspace PyPI
//!     providers, and the RESOLUTION POLICY -- the auto-imports
//!     injection gate, which decides the spec set the question is asked
//!     about; see `crate::handler::resolution_policy_fingerprint`).
//! A hit skips the probe entirely. Any change to the key invalidates the
//! whole file (it is rewritten from empty), so a stale verdict can never
//! be replayed against a universe it was not learned in.
//!
//! `Skipped` verdicts are NEVER cached: they mean the check could not
//! run (no repodata on disk / offline), which is a property of the
//! machine's cache state, not of the question.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Bumped whenever the on-disk shape or the key inputs change, so an old
/// file is discarded rather than misread.
///
/// v3 (fix p6c): the file-level validity key gained a resolution-policy field
/// carrying the auto-imports injection gate, so every v2 file -- any of which
/// may hold verdicts learned with injection ON, written at an OFF run's
/// address -- is invalidated once.
///
/// v4 (fix p6ac): every entry now carries the CLOSURE it was computed from --
/// the stage tag, the candidate-universe fingerprint and the normalized spec
/// set -- inside the record, not only hashed into the map key. A v3 entry has
/// no such stamp, so it cannot be checked and is discarded once.
const SCHEMA: &str = "v4-route-probe-verdicts";

/// Directory under the retread cache root holding one file per
/// (bundle, python minor, subdir).
const DIR: &str = "retread-route-probe-verdicts";

/// A decisive co-solve verdict, safe to replay under an unchanged key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum CachedVerdict {
    Sat,
    Unsat(Vec<String>),
    ExactUnsat(Vec<String>),
}

impl CachedVerdict {
    /// `None` for verdicts that must not be memoized.
    pub(crate) fn from_verdict(verdict: &crate::uv_closure::CoInstallVerdict) -> Option<Self> {
        match verdict {
            crate::uv_closure::CoInstallVerdict::Sat => Some(Self::Sat),
            crate::uv_closure::CoInstallVerdict::Unsat(reasons) => {
                Some(Self::Unsat(reasons.clone()))
            }
            crate::uv_closure::CoInstallVerdict::ExactUnsat(reasons) => {
                Some(Self::ExactUnsat(reasons.clone()))
            }
            // Indecisive: a property of this machine's repodata cache
            // state, not of the question. Never memoized.
            crate::uv_closure::CoInstallVerdict::Skipped(_) => None,
        }
    }
}

impl From<CachedVerdict> for crate::uv_closure::CoInstallVerdict {
    fn from(cached: CachedVerdict) -> Self {
        match cached {
            CachedVerdict::Sat => Self::Sat,
            CachedVerdict::Unsat(reasons) => Self::Unsat(reasons),
            CachedVerdict::ExactUnsat(reasons) => Self::ExactUnsat(reasons),
        }
    }
}

/// The CLOSURE a verdict was computed from, carried INSIDE the record.
///
/// The map key is `sha256(stage, universe, specs)`. Hashing an input into an
/// address makes a mismatched entry *unreachable* only while every writer
/// derives the address the same way; it does not make the entry
/// *unmisreadable*. This campaign runs several binaries concurrently against
/// one shared pixi cache root, so "every writer derives it the same way" is an
/// assumption, not a fact -- and a verdict adopted from the wrong closure is
/// exactly how a bundle's vendored set moves without its closure moving
/// (p6ab-5: `isaaclab-hover-pack` read `n_wheels` 68 / 76 / 93 on one binary
/// and one manifest, and the 40 route-probe questions behind that number were
/// answered 10-hits-0-solves in one run and 37-live-solves in the next).
///
/// So the record states its own provenance and [`RouteProbeCache::lookup`]
/// refuses anything whose stamp disagrees with what the reader computed. Same
/// shape as `crate::built_output_store::Record` (C11): stamped identity is
/// checked on read, and a refusal never yields the payload.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct EntryStamp {
    /// Which probe stage asked (`auto_route_joint_solve`, ...).
    #[serde(default)]
    pub(crate) stage: String,
    /// `crate::conda_solve::reachable_universe_digest_shared` for this
    /// question: the candidate universe the verdict is only valid within.
    #[serde(default)]
    pub(crate) universe: String,
    /// The normalized, sorted, deduplicated spec set -- the closure the
    /// co-solve actually ran over -- as a single canonical string.
    #[serde(default)]
    pub(crate) question: String,
}

impl EntryStamp {
    /// Build the stamp from the same inputs [`probe_digest`] hashes, so the
    /// address and the record can never describe different questions.
    pub(crate) fn new<S: AsRef<str>>(
        stage: &str,
        universe: &str,
        specs: impl Iterator<Item = S>,
    ) -> Self {
        Self {
            stage: stage.to_string(),
            universe: universe.to_string(),
            question: normalized_question(specs).join(&QUESTION_SEPARATOR.to_string()),
        }
    }
}

/// A stored verdict plus the closure it was computed from.
///
/// `stamp` is `serde(default)` so a truncated or hand-edited entry decodes to
/// an EMPTY stamp rather than failing the whole file -- and an empty stamp can
/// never equal a reader's stamp, so such an entry is refused, which is the
/// behaviour we want.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct StoredEntry {
    verdict: CachedVerdict,
    #[serde(default)]
    stamp: EntryStamp,
}

/// The normalized spec list shared by [`probe_digest`] and [`EntryStamp`].
fn normalized_question<S: AsRef<str>>(specs: impl Iterator<Item = S>) -> Vec<String> {
    let mut normalized: Vec<String> = specs.map(|s| s.as_ref().trim().to_string()).collect();
    normalized.sort();
    normalized.dedup();
    normalized
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct VerdictFile {
    #[serde(default)]
    schema: String,
    /// Hex digest of every non-question input (channels + repodata
    /// identity + target + policy). A mismatch discards the file.
    #[serde(default)]
    key: String,
    #[serde(default)]
    entries: BTreeMap<String, StoredEntry>,
}

/// Hex sha256 of one probe ENTRY key: the stage tag, the normalized,
/// sorted, deduplicated spec strings (the QUESTION), and the fingerprint of
/// the candidate universe that question can actually reach (the UNIVERSE,
/// from `crate::conda_solve::reachable_universe_digest_shared`).
///
/// The universe moved from the file-level validity key to the entry key in
/// v2. The old file-level input was `crate::repodata::repodata_identity`,
/// the on-disk repodata cache file's LENGTH and MTIME -- so an identical
/// document re-fetched after its 30-minute TTL invalidated every verdict in
/// the file. Measured: jobs 5598763 arm A vs arm B (one node, one job, one
/// manifest, differing only in which directory held the repodata cache)
/// produced a DIFFERENT validity key for all 14 bundles, and job 5611846
/// (fresh workspace, warm shared caches) discarded 13 of 14 verdict files
/// and re-executed all 315 probes against 116 cache hits.
pub(crate) fn probe_digest<S: AsRef<str>>(
    stage: &str,
    universe: &str,
    specs: impl Iterator<Item = S>,
) -> String {
    let normalized = normalized_question(specs);
    let mut h = Sha256::new();
    h.update(b"retread-probe-question\0");
    h.update(stage.as_bytes());
    h.update([0xffu8]);
    h.update(b"universe\0");
    h.update(universe.as_bytes());
    h.update([0xffu8]);
    for spec in &normalized {
        h.update(spec.as_bytes());
        h.update([0u8]);
    }
    format!("{:x}", h.finalize())
}

/// Hex sha256 of the cache-VALIDITY key. `policy_fields` is an ordered
/// list of `(tag, values)` describing everything except the question and the
/// candidate universe.
///
/// The universe is NOT in here any more (v2). It is per-ENTRY, keyed on the
/// reachable-record content that entry's own solve consulted, so an upstream
/// upload of an unrelated package no longer discards the whole file. Whole-
/// document keying would not have worked either: the conda-forge linux-64
/// repodata measurably changes inside an hour (637,578,869 bytes at 06:54
/// EDT vs 637,595,538 at 08:02 EDT on 2026-09-02, different sha256).
pub(crate) fn validity_key(
    channels: &[String],
    python: &str,
    subdir: &str,
    policy_fields: &[(&str, Vec<String>)],
) -> String {
    let mut h = Sha256::new();
    let mut field = |tag: &str, values: &[String]| {
        h.update(tag.as_bytes());
        h.update([0xffu8]);
        for value in values {
            h.update(value.as_bytes());
            h.update([0u8]);
        }
    };
    field("schema", &[SCHEMA.to_string()]);
    field("channels", channels);
    field("python", &[python.to_string()]);
    field("subdir", &[subdir.to_string()]);
    for (tag, values) in policy_fields {
        field(tag, values);
    }
    format!("{:x}", h.finalize())
}

/// Path of the verdict file for one (bundle, python minor, subdir).
pub(crate) fn cache_path(cache_dir: &Path, bundle: &str, python: &str, subdir: &str) -> PathBuf {
    cache_dir
        .join(DIR)
        .join(format!(
            "{}-py{}-{subdir}.json",
            bundle.replace(['/', '\\', ' '], "-"),
            python_minor(python),
        ))
}

/// The store's directory name under a persistent root, and the `--store`
/// spelling `retread store-reap` accepts. Named here, beside the layout it
/// describes, exactly as `crate::built_output_store::STORE_DIR` is, so the
/// reaper's spec READS it instead of carrying a second copy.
pub const STORE_DIR: &str = "route-probe-verdicts";

/// Refused entries are RENAMED in here, never deleted. Same name and same
/// rule as `crate::built_output_store::QUARANTINE`.
pub const QUARANTINE: &str = "quarantine";

/// How long an entry file may go unreferenced before a reaper may select it.
/// The same 14 days every other persistent store in this backend uses.
pub const ROUTE_PROBE_STORE_DEFAULT_MAX_AGE_DAYS: u64 = 14;

/// REAP-2. Name of the reaper's own store-wide try-lock, a dot-sidecar beside
/// the entry files it scans. NEVER a blocking lock: a process that cannot take
/// it does not reap, so a live relock is never made to wait on housekeeping.
/// Same shape and same rule as `source_build`'s `.built-wheels.reap.lock`.
const ROUTE_PROBE_REAP_LOCK_NAME: &str = ".route-probe-verdicts.reap.lock";

/// The separator [`EntryStamp`] joins the normalized spec set with, named once
/// so the writer that joins and the reaper that splits cannot drift apart.
const QUESTION_SEPARATOR: char = '\u{1f}';

/// REAP-2. `<root>/route-probe-verdicts`, the ONE formula for where the shared
/// store's entry files live. [`shared_cache_path`] and the reaper both go
/// through it, so a reaper can never walk a directory the writer does not use.
pub fn store_dir_in(root: &Path) -> PathBuf {
    root.join(STORE_DIR)
}

/// REAP-2, AND THIS IS A CORRECTION, NOT AN ADDITION.
///
/// THE ONE definition of the advisory lock that guards one verdict file's
/// read-modify-write. [`RouteProbeCache::persist`] took this path inline
/// (`self.path.with_extension("lock")`), so a reaper written against this
/// campaign's OTHER convention — `source_build::artifact_cache_lock_path`'s
/// dot form `.{file}.lock` — would have try-locked a path no writer ever
/// touches and reported `skipped_locked=0` over a live publish: a criterion
/// with no producer, which is the law-2 defect this lane exists to close.
/// So the writer and the reaper now READ THE SAME FUNCTION.
///
/// The shape is `<stem>.lock` beside the entry (`k-py3.12-linux-64.json` ->
/// `k-py3.12-linux-64.lock`) because that is what the writer has always used
/// and moving it would orphan every lock a running relock holds. It is not a
/// dotfile, which is why the reaper's enumeration filters by `.json` rather
/// than by a leading dot.
pub fn lock_path_for(entry: &Path) -> PathBuf {
    entry.with_extension("lock")
}

/// REAP-2. Is this directory entry's name one of the store's ENTRY files?
///
/// The store directory holds four other things and every one of them would be
/// a bug to reap: the `<stem>.lock` sidecars, the `<stem>.tmp<pid>` files a
/// concurrent `persist` renames from, the `quarantine` directory, and the
/// reaper's own dot-lock. Only a published `*.json` is an entry.
pub fn is_verdict_entry_name(name: &str) -> bool {
    !name.starts_with('.') && name.ends_with(".json")
}

/// REAP-2. Can any reader ever ADDRESS this entry again?
///
/// The entry's map key is `probe_digest(stage, universe, specs)` and v4 records
/// the same three inputs in the entry's own [`EntryStamp`]. So recomputing the
/// digest from the stamp and comparing it to the key is a complete, offline
/// answer to "is this entry reachable": a record whose stamp does not reproduce
/// its own address can never be returned to anybody, because
/// [`RouteProbeCache::lookup`] looks up BY the digest and then refuses anything
/// whose stamp disagrees. `serde(default)` makes a v3 remnant or a truncated
/// entry decode to an EMPTY stamp, which cannot reproduce any address, so those
/// are unreachable too.
///
/// **WHAT THIS IS NOT, STATED HERE BECAUSE THE DIFFERENCE DECIDES A ROW.**
/// It is not "the universe this entry was learned in no longer exists". That
/// question is NOT decidable from the store: `conda_solve::
/// reachable_universe_digest` hashes the reachable CANDIDATE SET for one
/// question, and nothing on disk enumerates the universes that are currently
/// live — `repodata::snapshot_documents_at` yields document identities, not
/// universe digests — so answering it would need a re-solve per entry, which a
/// reaper must never do. What IS decidable, and what the `universe` reason
/// counts, is an address the universe component can no longer reach.
pub(crate) fn entry_is_unreachable(key: &str, stamp: &EntryStamp) -> bool {
    if stamp.stage.is_empty() && stamp.universe.is_empty() && stamp.question.is_empty() {
        return true;
    }
    let specs: Vec<&str> = if stamp.question.is_empty() {
        Vec::new()
    } else {
        stamp.question.split(QUESTION_SEPARATOR).collect()
    };
    probe_digest(&stamp.stage, &stamp.universe, specs.iter()) != key
}

/// REAP-2. Why one entry file was selected. `&'static str` so it is the same
/// value in the per-entry row, the reason tally and the guard's assertion.
const REASON_AGE: &str = "age";
const REASON_UNIVERSE: &str = "universe";

/// REAP-2. What the route-probe reap did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RouteProbeReapReport {
    pub(crate) scanned: u64,
    /// Entries RENAMED into `quarantine/` (apply) or that would be (dry run).
    pub(crate) quarantined: u64,
    pub(crate) kept: u64,
    /// Of `quarantined`, how many were selected because the file was older
    /// than `max_age`.
    pub(crate) selected_age: u64,
    /// Of `quarantined`, how many were selected because every entry in the
    /// file is unaddressable — see [`entry_is_unreachable`].
    pub(crate) selected_universe: u64,
    /// A live `persist` held the entry's own advisory lock, so the entry was
    /// left exactly where it was and COUNTED. The reader half of
    /// [`lock_path_for`].
    pub(crate) skipped_locked: u64,
    /// Another process held the store-wide try-lock and this one backed off
    /// without scanning anything.
    pub(crate) skipped_concurrent: bool,
    pub(crate) entries: Vec<crate::courier::ReapedEntry>,
}

/// REAP-2, THE REAPER. Quarantine every verdict file that is over-age or that
/// no reader can address any more.
///
/// The three rules C18-1 wrote and every reaper in this backend keeps:
/// 1. **It never deletes.** A selected entry is RENAMED to
///    `<store>/quarantine/<name>-<unix>-<pid>`. Reclaiming a quarantine is a
///    separate, operator-visible act.
/// 2. **It never blocks anyone.** Both the store-wide lock and the per-entry
///    [`lock_path_for`] lock are TRY-locks; an entry a live `persist` holds is
///    skipped and counted, never waited on. Note that `persist` itself takes
///    that lock BLOCKING, which is correct for a writer and would be a wedge
///    for housekeeping.
/// 3. **It re-reads the age with the lock held.** A publish can land between
///    the scan and the rename, so the mtime is re-stated after the entry lock
///    is taken and an entry that became fresh in that window is kept.
///
/// `max_age` zero DISABLES the age rule (the shape every other reaper in this
/// backend uses for "off"); the unaddressable rule still runs, because an entry
/// nobody can reach is not a retention decision.
pub(crate) fn reap_route_probe_store(
    root: &Path,
    max_age: std::time::Duration,
    mode: crate::courier::ReapMode,
) -> anyhow::Result<RouteProbeReapReport> {
    use crate::courier::{ReapLockOutcome, ReapedEntry, take_entry_lock, take_reap_lock};
    use anyhow::Context as _;

    let mut report = RouteProbeReapReport::default();
    let store_dir = store_dir_in(root);
    let max_age_days = max_age.as_secs() / 86_400;
    if !store_dir.is_dir() {
        tracing::info!(
            store = %store_dir.display(),
            scanned = 0,
            quarantined = 0,
            kept = 0,
            max_age_days,
            mode = mode.as_str(),
            reason = "store-absent",
            "route_probe_store reap",
        );
        return Ok(report);
    }
    let reap_lock_path = store_dir.join(ROUTE_PROBE_REAP_LOCK_NAME);
    let _reap_lock = match take_reap_lock(&reap_lock_path, mode)? {
        ReapLockOutcome::Held(lock) => Some(lock),
        ReapLockOutcome::Absent => None,
        ReapLockOutcome::Busy | ReapLockOutcome::Unopenable => {
            report.skipped_concurrent = true;
            tracing::info!(
                store = %store_dir.display(),
                mode = mode.as_str(),
                "route_probe_store reap skipped=concurrent",
            );
            return Ok(report);
        }
    };
    let quarantine_root = store_dir.join(QUARANTINE);
    let now = std::time::SystemTime::now();
    let mut names: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&store_dir)
        .with_context(|| format!("reading the route-probe store {}", store_dir.display()))?
    {
        let entry = entry
            .with_context(|| format!("reading an entry of {}", store_dir.display()))?;
        let Some(name) = entry.file_name().to_str().map(|s| s.to_string()) else {
            continue;
        };
        if !is_verdict_entry_name(&name) {
            continue;
        }
        names.push(name);
    }
    names.sort();
    for name in names {
        let entry_path = store_dir.join(&name);
        if !entry_path.is_file() {
            continue;
        }
        report.scanned += 1;
        let age = entry_age(&entry_path, now);
        let over_age = !max_age.is_zero() && age.is_some_and(|age| age > max_age);
        let unreachable = file_is_unaddressable(&entry_path);
        if !over_age && !unreachable {
            report.kept += 1;
            continue;
        }
        // Rule 2: the entry's OWN lock, try-only. A held lock is a live
        // `persist`, which is a use, so the entry stays and is counted.
        let entry_lock_path = lock_path_for(&entry_path);
        let _entry_lock = match take_entry_lock(&entry_lock_path, mode) {
            ReapLockOutcome::Held(lock) => Some(lock),
            ReapLockOutcome::Absent => None,
            ReapLockOutcome::Busy => {
                report.skipped_locked += 1;
                tracing::info!(
                    entry = %name,
                    mode = mode.as_str(),
                    "route_probe_store skipped=locked",
                );
                continue;
            }
            // "I cannot look" is not "a writer holds it": the entry is kept and
            // NOT counted as locked, the distinction `ReapLockOutcome` exists
            // for.
            ReapLockOutcome::Unopenable => {
                report.kept += 1;
                continue;
            }
        };
        // Rule 3: re-state the decision with the lock held. Only the AGE can
        // change under us (a publish rewrites the file, which both refreshes
        // the mtime and replaces its entries), so re-reading the age is
        // sufficient and re-parsing would be a second full read for nothing.
        let fresh_age = entry_age(&entry_path, std::time::SystemTime::now());
        let over_age = !max_age.is_zero() && fresh_age.is_some_and(|age| age > max_age);
        let unreachable = if over_age {
            unreachable
        } else {
            file_is_unaddressable(&entry_path)
        };
        if !over_age && !unreachable {
            report.kept += 1;
            continue;
        }
        // AGE FIRST: an over-age file is a retention decision whatever its
        // entries say, and a reader counting reclaimed bytes against a
        // retention policy must not have them attributed to a defect.
        let reason = if over_age { REASON_AGE } else { REASON_UNIVERSE };
        let age_days = fresh_age.map_or(0, |age| age.as_secs() / 86_400);
        let stamp_unix = now
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let quarantine =
            quarantine_root.join(format!("{name}-{stamp_unix}-{}", std::process::id()));
        // THE DRY RUN STOPS HERE, one statement before the first thing that
        // writes: scanned, aged, re-aged and selected by exactly the rules the
        // apply path uses, and from here on nothing is created and nothing is
        // renamed.
        if mode.is_dry_run() {
            report.quarantined += 1;
            if reason == REASON_AGE {
                report.selected_age += 1;
            } else {
                report.selected_universe += 1;
            }
            report.entries.push(ReapedEntry {
                label: name.clone(),
                path: entry_path.clone(),
                quarantine: None,
                age_days,
                reason,
            });
            tracing::info!(
                entry = %name, age_days, max_age_days, reason,
                mode = mode.as_str(),
                "route_probe_store would-evict",
            );
            continue;
        }
        if let Err(error) = std::fs::create_dir_all(&quarantine_root) {
            tracing::warn!(
                store = %store_dir.display(), error = %error,
                "could not create the route-probe quarantine; nothing evicted",
            );
            report.kept += 1;
            continue;
        }
        // Rule 1: RENAME. Never `remove_file`. A reader already holding the
        // file keeps reading valid bytes; the next one misses and re-probes.
        if let Err(error) = std::fs::rename(&entry_path, &quarantine) {
            tracing::warn!(
                entry = %name, error = %error,
                "route_probe_store eviction could not rename; entry kept",
            );
            report.kept += 1;
            continue;
        }
        report.quarantined += 1;
        if reason == REASON_AGE {
            report.selected_age += 1;
        } else {
            report.selected_universe += 1;
        }
        report.entries.push(ReapedEntry {
            label: name.clone(),
            path: entry_path.clone(),
            quarantine: Some(quarantine.clone()),
            age_days,
            reason,
        });
        tracing::info!(
            entry = %name, age_days, max_age_days, reason,
            quarantine = %quarantine.display(),
            mode = mode.as_str(),
            "route_probe_store evicted",
        );
    }
    tracing::info!(
        store = %store_dir.display(),
        scanned = report.scanned,
        quarantined = report.quarantined,
        selected_age = report.selected_age,
        selected_universe = report.selected_universe,
        kept = report.kept,
        skipped_locked = report.skipped_locked,
        max_age_days,
        mode = mode.as_str(),
        "route_probe_store reap",
    );
    Ok(report)
}

/// How long ago this entry file was last written. `None` when the mtime cannot
/// be read or lies in the future (clock skew across nodes), which reads as
/// "not over-age" — a reaper must never evict on a clock it cannot trust.
fn entry_age(path: &Path, now: std::time::SystemTime) -> Option<std::time::Duration> {
    let modified = std::fs::metadata(path).and_then(|meta| meta.modified()).ok()?;
    now.duration_since(modified).ok()
}

/// Is EVERY entry in this file unaddressable? A file with at least one
/// reachable entry is live, and an EMPTY or unparseable file is NOT selected
/// on this rule: "I could not read it" is not "nobody can reach it", and an
/// unreadable file is left for the age rule to age out.
fn file_is_unaddressable(path: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(file) = serde_json::from_str::<VerdictFile>(&text) else {
        return false;
    };
    if file.entries.is_empty() {
        return false;
    }
    file.entries
        .iter()
        .all(|(key, entry)| entry_is_unreachable(key, &entry.stamp))
}

/// Where a `RouteProbeCache`'s file lives, for the printed row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StoreMode {
    /// `retread-route-probe-store` absent: the pre-existing, per-bundle,
    /// job-scoped file under the handler's `cache_dir`. Byte-for-byte the
    /// behaviour that shipped before this key existed.
    PerJob,
    /// `retread-route-probe-store = "<root>"`: one shared, durable file per
    /// (validity key, python minor, subdir) under `<root>/route-probe-verdicts/`.
    Shared,
}

impl StoreMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::PerJob => "off",
            Self::Shared => "shared",
        }
    }
}

/// The harness's reach at this key. THIS IS NOT A LEGACY SHIM and it is not
/// a preference for environment over arguments.
///
/// `crate::config::RetreadConfig::route_probe_store` is the DECLARED control
/// and it wins whenever it is set: a misspelling there is a load error rather
/// than a silent no-op, which an env var can never be. But the packs whose
/// `[build.config]` would carry it live in `imprint-data`, a shared tree no
/// lane may edit, so a manifest key alone would be a control with no reachable
/// caller in this campaign — the reader/writer defect, not a safeguard against
/// it. `crate::built_output_store::BuiltOutputStore::from_config` resolves the
/// same two-rung ladder for the same reason, and the harness turns THAT store
/// on through exactly this rung.
pub const STORE_ENV: &str = "RETREAD_ROUTE_PROBE_STORE";

/// Resolve the shared store root: the config key first, then [`STORE_ENV`].
/// `None` is the default and is byte-for-byte the behaviour that shipped
/// before this key existed.
pub fn resolve_store_root(configured: Option<&Path>) -> Option<PathBuf> {
    match configured {
        Some(path) => Some(path.to_path_buf()),
        None => match std::env::var_os(STORE_ENV) {
            Some(value) if !value.is_empty() => Some(PathBuf::from(value)),
            _ => None,
        },
    }
}

fn python_minor(python: &str) -> String {
    python
        .split('.')
        .take(2)
        .collect::<Vec<_>>()
        .join(".")
        .replace(['/', '\\'], "-")
}

/// Path of the SHARED verdict file, whose address contains no bundle and no
/// job.
///
/// THE FILENAME IS THE VALIDITY KEY, AND THAT IS THE WHOLE POINT. Dropping
/// `safe_bundle` from [`cache_path`] and keeping one file per (python, subdir)
/// would be wrong: the file carries a [`validity_key`] and
/// [`RouteProbeCache::open`] DISCARDS the entire file when that key differs,
/// so two bundles whose policy fields differ — a different `workspace-deps`
/// set is enough — would alternately wipe each other's verdicts on every
/// open, turning a shared store into a slower cold path. Putting the key IN
/// the address makes a key mismatch on open structurally impossible: bundles
/// that legitimately share a universe share a file, and bundles that do not
/// simply address different files.
///
/// What IS shared, and what the 29.5 % cross-pack overlap PACKPROF-1 measured
/// rides on, is the ENTRY address: [`probe_digest`] hashes only the stage, the
/// universe and the normalized spec set — the bundle is not in it.
pub(crate) fn shared_cache_path(
    store_root: &Path,
    validity_key: &str,
    python: &str,
    subdir: &str,
) -> PathBuf {
    store_root.join(STORE_DIR).join(format!(
        "{validity_key}-py{}-{subdir}.json",
        python_minor(python),
    ))
}

/// Run-wide totals across every [`RouteProbeCache`] this process opened,
/// folded in on each handle's drop so the run row is a SUM of the per-bundle
/// rows and not a second, independently derived number.
#[derive(Debug, Default)]
pub(crate) struct StoreTotals {
    pub(crate) bundles: usize,
    pub(crate) consulted: usize,
    pub(crate) hits: usize,
    pub(crate) misses: usize,
    pub(crate) published: usize,
    pub(crate) refused: usize,
    pub(crate) refused_stage: usize,
    pub(crate) refused_universe: usize,
    pub(crate) refused_question: usize,
    pub(crate) mode: Option<StoreMode>,
    pub(crate) root: Option<PathBuf>,
}

static RUN_TOTALS: Mutex<Option<StoreTotals>> = Mutex::new(None);

/// Print `### ROUTE PROBE STORE ... bundle=<run total>` once, at the end of
/// the process, from the run totals every handle folded in on drop.
///
/// STDERR, NEVER STDOUT, for the reason `handler::conda_outputs`'s
/// `### built-outputs REFUSED` row states: `rpc.rs` owns `tokio::io::stdout()`
/// as the JSON-RPC channel, and the harness tees backend stderr into
/// `<arm>.backend.log`.
pub fn emit_run_total() {
    let totals = RUN_TOTALS.lock().expect("route probe store totals").take();
    let Some(totals) = totals else {
        return;
    };
    eprintln!(
        "### ROUTE PROBE STORE mode={} root={} bundle=<run total> bundles={} \
         consulted={} hit={} miss={} published={} refused={} reason=stage:{},universe:{},question:{}",
        totals.mode.map_or("off", StoreMode::as_str),
        totals
            .root
            .as_ref()
            .map_or_else(|| "-".to_string(), |root| root.display().to_string()),
        totals.bundles,
        totals.consulted,
        totals.hits,
        totals.misses,
        totals.published,
        totals.refused,
        totals.refused_stage,
        totals.refused_universe,
        totals.refused_question,
    );
}

/// Why a present entry was refused. Every arm is a MISS at the call site; the
/// variant exists so the row names which one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RefusalCounts {
    stage: usize,
    universe: usize,
    question: usize,
}

impl RefusalCounts {
    fn total(self) -> usize {
        self.stage + self.universe + self.question
    }
}

/// Process-lifetime handle over one verdict file.
#[derive(Debug)]
pub(crate) struct RouteProbeCache {
    path: PathBuf,
    key: String,
    /// The bundle that opened this handle. LABEL ONLY — it is deliberately
    /// absent from the shared address, and a guard holds that separation.
    bundle: String,
    mode: StoreMode,
    entries: Mutex<BTreeMap<String, StoredEntry>>,
    hits: AtomicUsize,
    misses: AtomicUsize,
    /// Verdicts written back to disk by this handle.
    published: AtomicUsize,
    /// Entries that were PRESENT at the reader's address but whose stamped
    /// closure disagreed with the reader's. Counted separately from a miss so
    /// "this never happens" stays a measurable claim rather than a belief.
    refusals: AtomicUsize,
    refusal_reasons: Mutex<RefusalCounts>,
}

impl RouteProbeCache {
    /// Load (or start empty on a key/schema mismatch -- that IS the
    /// invalidation). A read error is non-fatal: worst case is a cold
    /// probe round, never a wrong verdict.
    pub(crate) fn open(path: PathBuf, key: String) -> Self {
        Self::open_labelled(path, key, String::new(), StoreMode::PerJob)
    }

    /// [`Self::open`] plus the two fields the printed row needs. The label and
    /// the mode change no address and no verdict.
    pub(crate) fn open_labelled(
        path: PathBuf,
        key: String,
        bundle: String,
        mode: StoreMode,
    ) -> Self {
        let entries = match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<VerdictFile>(&text) {
                Ok(file) if file.schema == SCHEMA && file.key == key => file.entries,
                Ok(_) => {
                    tracing::debug!(
                        path = %path.display(),
                        "route probe cache: key or schema changed; discarding verdicts",
                    );
                    BTreeMap::new()
                }
                Err(error) => {
                    tracing::debug!(
                        path = %path.display(), %error,
                        "route probe cache: unreadable verdict file; starting empty",
                    );
                    BTreeMap::new()
                }
            },
            Err(_) => BTreeMap::new(),
        };
        Self {
            path,
            key,
            bundle,
            mode,
            entries: Mutex::new(entries),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            published: AtomicUsize::new(0),
            refusals: AtomicUsize::new(0),
            refusal_reasons: Mutex::new(RefusalCounts::default()),
        }
    }

    /// Replay a verdict, but only one whose stamped closure IS the reader's.
    ///
    /// An entry at the right address with the wrong stamp is REFUSED: it
    /// yields `None` (so the caller probes, exactly as on a miss) and is
    /// counted and logged, never silently adopted.
    pub(crate) fn lookup(&self, digest: &str, stamp: &EntryStamp) -> Option<CachedVerdict> {
        let entry = self
            .entries
            .lock()
            .expect("route probe cache mutex")
            .get(digest)
            .cloned();
        match entry {
            Some(entry) if entry.stamp == *stamp => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(entry.verdict)
            }
            Some(entry) => {
                self.refusals.fetch_add(1, Ordering::Relaxed);
                self.misses.fetch_add(1, Ordering::Relaxed);
                {
                    let mut reasons =
                        self.refusal_reasons.lock().expect("route probe refusal counts");
                    if entry.stamp.stage != stamp.stage {
                        reasons.stage += 1;
                    } else if entry.stamp.universe != stamp.universe {
                        reasons.universe += 1;
                    } else {
                        reasons.question += 1;
                    }
                }
                tracing::warn!(
                    path = %self.path.display(),
                    digest = %digest,
                    stored_stage = %entry.stamp.stage,
                    stored_universe = %entry.stamp.universe,
                    reader_stage = %stamp.stage,
                    reader_universe = %stamp.universe,
                    "route probe cache: refused an entry whose stamped closure is not \
                     the reader's; probing instead of adopting",
                );
                None
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Copy the entry a REPUBLISH is about to displace into
    /// `<dir>/quarantine/`, before it is displaced.
    ///
    /// **THIS IS THE WRITE PATH, DELIBERATELY, AND IT IS NOT THE READ PATH.**
    /// An earlier draft quarantined at `lookup` time, on refusal. That is
    /// wrong twice over: it destroys, on a mere READ, an entry that is
    /// perfectly valid for the writer whose stamp it carries — p6ac's own
    /// control (`refusal must be about the stamp, not about refusing
    /// everything`) asserts that entry is still served to that reader, and it
    /// went red — and a read path that mutates the store turns every
    /// concurrent reader into a writer. Displacement is the only moment an
    /// entry actually stops existing, so displacement is where the copy
    /// belongs.
    ///
    /// It is copied, never deleted — the same rule as
    /// `crate::built_output_store::BuiltOutputStore::quarantine_refused`, for
    /// the same reason: a displaced entry is the only artifact of a
    /// disagreement between two writers, and deleting it destroys the evidence
    /// that the disagreement happened. A store this backend writes never
    /// `rm`s; eviction is a reaper's job and a reaper logs.
    ///
    /// Best-effort: a quarantine that cannot be written does not block the
    /// republish. Worst case is the pre-existing behaviour — a silent
    /// overwrite — never a wrong verdict.
    fn quarantine_displaced(&self, digest: &str, entry: &StoredEntry) {
        let Some(dir) = self.path.parent() else {
            return;
        };
        let stem = self
            .path
            .file_stem()
            .map_or_else(|| "verdicts".to_string(), |s| s.to_string_lossy().into_owned());
        let quarantine = dir.join(QUARANTINE);
        if std::fs::create_dir_all(&quarantine).is_err() {
            return;
        }
        let target = quarantine.join(format!("{stem}-{digest}-{}.json", std::process::id()));
        if let Ok(text) = serde_json::to_string(entry) {
            let tmp = quarantine.join(format!(".{stem}-{digest}-{}.tmp", std::process::id()));
            if std::fs::write(&tmp, text.as_bytes()).is_ok()
                && std::fs::rename(&tmp, &target).is_ok()
            {
                tracing::warn!(
                    quarantined = %target.display(), digest = %digest,
                    stored_stage = %entry.stamp.stage,
                    stored_universe = %entry.stamp.universe,
                    "route probe cache: quarantined the entry a republish displaced \
                     (copied aside, never deleted)",
                );
                return;
            }
            let _ = std::fs::remove_file(&tmp);
        }
    }

    pub(crate) fn record(&self, digest: &str, stamp: &EntryStamp, verdict: CachedVerdict) {
        self.published.fetch_add(1, Ordering::Relaxed);
        let (snapshot, displaced) = {
            let mut entries = self.entries.lock().expect("route probe cache mutex");
            let displaced = entries.insert(
                digest.to_string(),
                StoredEntry {
                    verdict,
                    stamp: stamp.clone(),
                },
            );
            (entries.clone(), displaced)
        };
        // Only a genuine DISAGREEMENT is evidence. Re-recording the identical
        // (verdict, stamp) — the ordinary shape of two workers answering the
        // same question in one run — displaces nothing worth keeping, and
        // quarantining it would bury the real disagreements in noise.
        if let Some(old) = displaced.filter(|old| old.stamp != *stamp) {
            self.quarantine_displaced(digest, &old);
        }
        self.persist(snapshot);
    }

    /// Entries refused for a stamp mismatch since this handle was opened.
    pub(crate) fn refusals(&self) -> usize {
        self.refusals.load(Ordering::Relaxed)
    }

    /// `(hits, misses)` since this handle was opened.
    pub(crate) fn stats(&self) -> (usize, usize) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.lock().expect("route probe cache mutex").len()
    }

    /// Verdicts this handle wrote back.
    pub(crate) fn published(&self) -> usize {
        self.published.load(Ordering::Relaxed)
    }

    /// The one `### ROUTE PROBE STORE` row for this handle. Built here, in the
    /// only place that owns the counters, so the per-bundle row and the run
    /// total can never be two independently derived numbers.
    pub(crate) fn store_row(&self) -> String {
        let (hits, misses) = self.stats();
        let reasons = *self.refusal_reasons.lock().expect("route probe refusal counts");
        let root = self
            .path
            .parent()
            .map_or_else(|| self.path.display().to_string(), |p| p.display().to_string());
        format!(
            "### ROUTE PROBE STORE mode={} root={} bundle={} consulted={} hit={} miss={} \
             published={} refused={} reason=stage:{},universe:{},question:{}",
            self.mode.as_str(),
            root,
            if self.bundle.is_empty() {
                "-"
            } else {
                self.bundle.as_str()
            },
            hits + misses,
            hits,
            misses,
            self.published(),
            reasons.total(),
            reasons.stage,
            reasons.universe,
            reasons.question,
        )
    }

    /// Write `entries` back to disk WITHOUT dropping entries another writer
    /// learned in the meantime.
    ///
    /// The previous version rewrote the whole file from this handle's
    /// in-memory snapshot. Two processes probing the same (bundle, python,
    /// subdir) -- which is the normal shape of a relock, one bundle per
    /// worker over a shared cache dir -- each rendered the file from the
    /// entries THEY had, and whoever renamed last silently erased the other's
    /// verdicts. The loss is invisible: the next run just re-probes.
    ///
    /// So the write is: take an exclusive advisory lock on a sidecar `.lock`
    /// file, re-read what is on disk under it, UNION the two entry sets
    /// (ours wins on a shared digest -- same question, same universe, so the
    /// verdicts agree), write a temp file and rename it into place. A file
    /// whose schema or validity key does not match ours is not merged: it
    /// belongs to a different question universe and is replaced, which is the
    /// invalidation [`RouteProbeCache::open`] already performs.
    ///
    /// The lock is advisory and best-effort: if it cannot be taken the write
    /// still happens. Worst case is the old behaviour (a lost entry, a re-probe
    /// next run), never a wrong verdict.
    fn persist(&self, entries: BTreeMap<String, StoredEntry>) {
        if let Some(parent) = self.path.parent() {
            if std::fs::create_dir_all(parent).is_err() {
                return;
            }
        }

        // Held for the read-modify-write below; dropped (unlocking) on return.
        let lock_path = lock_path_for(&self.path);
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .ok();
        if let Some(lock) = lock.as_ref() {
            let _ = fs4::fs_std::FileExt::lock_exclusive(lock);
        }

        let mut merged = match std::fs::read_to_string(&self.path) {
            Ok(text) => match serde_json::from_str::<VerdictFile>(&text) {
                Ok(file) if file.schema == SCHEMA && file.key == self.key => file.entries,
                _ => BTreeMap::new(),
            },
            Err(_) => BTreeMap::new(),
        };
        let recovered = merged.len();
        merged.extend(entries);

        let file = VerdictFile {
            schema: SCHEMA.to_string(),
            key: self.key.clone(),
            entries: merged,
        };
        if let Ok(text) = serde_json::to_string(&file) {
            let tmp = self
                .path
                .with_extension(format!("tmp{}", std::process::id()));
            if std::fs::write(&tmp, text.as_bytes()).is_ok()
                && std::fs::rename(&tmp, &self.path).is_ok()
            {
                tracing::trace!(
                    path = %self.path.display(), recovered,
                    "route probe cache: persisted (merged with on-disk entries)",
                );
            }
            let _ = std::fs::remove_file(&tmp);
        }

        if let Some(lock) = lock.as_ref() {
            let _ = fs4::fs_std::FileExt::unlock(lock);
        }
    }
}

/// The per-bundle `### ROUTE PROBE STORE` row, and the fold into the run
/// totals, happen HERE — on the last `Arc`'s drop — because that is the one
/// moment the counters are final and the handler is not holding a lock.
///
/// **ONLY [`StoreMode::Shared`] EMITS, AND THAT IS A LANDING-SAFETY DECISION,
/// NOT AN OVERSIGHT.** With the `retread-route-probe-store` key absent the
/// backend must be byte-for-byte what 0be408a was, stderr included — a lane
/// guard `cmp`s exactly that — so a new row in the default regime would itself
/// be the regression. `mode=` stays in the row because the row must state its
/// own regime rather than leave a reader to infer it from the root path; today
/// it therefore only ever reads `shared`.
impl Drop for RouteProbeCache {
    fn drop(&mut self) {
        if self.mode != StoreMode::Shared {
            return;
        }
        eprintln!("{}", self.store_row());

        let (hits, misses) = self.stats();
        let reasons = *self.refusal_reasons.lock().expect("route probe refusal counts");
        let mut totals = RUN_TOTALS.lock().expect("route probe store totals");
        let totals = totals.get_or_insert_with(StoreTotals::default);
        totals.bundles += 1;
        totals.consulted += hits + misses;
        totals.hits += hits;
        totals.misses += misses;
        totals.published += self.published();
        totals.refused += reasons.total();
        totals.refused_stage += reasons.stage;
        totals.refused_universe += reasons.universe;
        totals.refused_question += reasons.question;
        totals.mode = Some(self.mode);
        if totals.root.is_none() {
            totals.root = self.path.parent().map(Path::to_path_buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "retread-route-probe-cache-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn key_for(policy: &str) -> String {
        validity_key(
            &["https://prefix.dev/conda-forge/".to_string()],
            "3.12",
            "linux-64",
            &[("policy", vec![policy.to_string()])],
        )
    }

    /// Guard (a): a second run over the same probe set executes ZERO
    /// probes and returns identical verdicts.
    #[test]
    fn second_run_over_the_same_probe_set_executes_no_probes() {
        let dir = tmp_dir("same-key");
        let path = cache_path(&dir, "isaac-pack-latest", "3.12", "linux-64");
        let key = key_for("strict");

        let questions: Vec<(String, EntryStamp)> = (0..100)
            .map(|i| {
                let specs = [format!("absl-py=={i}.0")];
                (
                    probe_digest("auto_route_joint_solve", "universe-rev-1", specs.iter()),
                    EntryStamp::new("auto_route_joint_solve", "universe-rev-1", specs.iter()),
                )
            })
            .collect();

        let mut cold_executions = 0usize;
        {
            let cache = RouteProbeCache::open(path.clone(), key.clone());
            for (digest, stamp) in &questions {
                if cache.lookup(digest, stamp).is_none() {
                    cold_executions += 1;
                    cache.record(
                        digest,
                        stamp,
                        CachedVerdict::ExactUnsat(vec!["no candidates".into()]),
                    );
                }
            }
            assert_eq!(cache.stats(), (0, 100), "cold run is all misses");
        }
        assert_eq!(cold_executions, 100);

        let mut warm_executions = 0usize;
        let cache = RouteProbeCache::open(path, key);
        for (digest, stamp) in &questions {
            match cache.lookup(digest, stamp) {
                Some(verdict) => assert_eq!(
                    verdict,
                    CachedVerdict::ExactUnsat(vec!["no candidates".into()]),
                    "replayed verdict must be identical",
                ),
                None => {
                    warm_executions += 1;
                    cache.record(digest, stamp, CachedVerdict::Sat);
                }
            }
        }
        assert_eq!(warm_executions, 0, "warm run must execute zero probes");
        assert_eq!(cache.stats(), (100, 0), "warm run is all hits");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Guard (b): a changed key re-executes every probe.
    #[test]
    fn changed_key_reexecutes_every_probe() {
        let dir = tmp_dir("changed-key");
        let path = cache_path(&dir, "isaac-pack-latest", "3.12", "linux-64");
        let questions: Vec<(String, EntryStamp)> = (0..8)
            .map(|i| {
                let specs = [format!("absl-py=={i}.0")];
                (
                    probe_digest("auto_route_joint_solve", "universe-rev-1", specs.iter()),
                    EntryStamp::new("auto_route_joint_solve", "universe-rev-1", specs.iter()),
                )
            })
            .collect();
        {
            let cache = RouteProbeCache::open(path.clone(), key_for("strict"));
            for (digest, stamp) in &questions {
                cache.record(digest, stamp, CachedVerdict::Sat);
            }
            assert_eq!(cache.len(), 8);
        }
        // Same file, different POLICY -> whole file discarded.
        let cache = RouteProbeCache::open(path.clone(), key_for("disabled"));
        assert_eq!(cache.len(), 0, "a changed key must discard every verdict");
        let mut executions = 0usize;
        for (digest, stamp) in &questions {
            if cache.lookup(digest, stamp).is_none() {
                executions += 1;
            }
        }
        assert_eq!(executions, 8, "every probe must be re-executed");
        assert_eq!(cache.stats(), (0, 8));

        // And the original key still resolves to the original file name.
        assert_eq!(
            path,
            cache_path(&dir, "isaac-pack-latest", "3.12", "linux-64"),
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// GUARD (a) -- p6ac. The store's CONTENTS may not decide an answer.
    ///
    /// Same question, same universe; one reader over an EMPTY store and one
    /// over a store already holding 500 unrelated verdicts must replay the
    /// same verdict, and the extra entries must not turn into hits.
    ///
    /// This is the property p6ab-5 needed and could not state: `n_wheels` for
    /// `isaaclab-hover-pack` read 68 in four canonical relocks and 93 in the
    /// next two on the same binary and manifest, and the 40 route-probe
    /// questions behind it were answered 10-hits/0-solves in one run and
    /// 3-hits/37-live-solves in the next. Whether warmth alone can move an
    /// answer had to become a test rather than an argument.
    #[test]
    fn p6ac_store_contents_beyond_the_question_cannot_change_the_answer() {
        let dir = tmp_dir("p6ac-contents");
        let key = key_for("strict");
        let specs = ["torch==2.5.1".to_string(), "sympy==1.13.1".to_string()];
        let digest = probe_digest("auto_route_joint_solve", "universe-rev-7", specs.iter());
        let stamp = EntryStamp::new("auto_route_joint_solve", "universe-rev-7", specs.iter());
        let answer = CachedVerdict::Unsat(vec!["pytorch 2.7.1 shadows torch".to_string()]);

        // Reader 1: cold store. Misses, solves, records.
        let cold_path = cache_path(&dir, "cold-pack", "3.11", "linux-64");
        let cold = RouteProbeCache::open(cold_path.clone(), key.clone());
        assert_eq!(cold.lookup(&digest, &stamp), None, "a cold store must miss");
        cold.record(&digest, &stamp, answer.clone());
        let cold_answer = RouteProbeCache::open(cold_path, key.clone())
            .lookup(&digest, &stamp)
            .expect("the recorded verdict replays");

        // Reader 2: the SAME question against a store stuffed with 500
        // verdicts for other questions, recorded first.
        let warm_path = cache_path(&dir, "warm-pack", "3.11", "linux-64");
        {
            let warm = RouteProbeCache::open(warm_path.clone(), key.clone());
            for i in 0..500usize {
                let other = [format!("filler-{i}==1.0")];
                warm.record(
                    &probe_digest("auto_route_joint_solve", "universe-rev-7", other.iter()),
                    &EntryStamp::new("auto_route_joint_solve", "universe-rev-7", other.iter()),
                    CachedVerdict::Sat,
                );
            }
            warm.record(&digest, &stamp, answer.clone());
        }
        let warm = RouteProbeCache::open(warm_path, key);
        let warm_answer = warm.lookup(&digest, &stamp).expect("warm store replays");

        assert_eq!(
            cold_answer, warm_answer,
            "a store holding 500 extra verdicts must answer this question \
             exactly as an empty one did",
        );
        assert_eq!(warm_answer, answer);
        assert_eq!(
            warm.stats(),
            (1, 0),
            "the 500 unrelated entries must not be consulted for this question",
        );
        assert_eq!(warm.refusals(), 0, "nothing here is stamped wrongly");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// GUARD (b) -- p6ac. An entry at the reader's address whose stamped
    /// closure is NOT the reader's is refused, never adopted.
    ///
    /// The address is `sha256(stage, universe, specs)`, so on the tip an entry
    /// can only be reached by a writer that derived the address the same way.
    /// That is an assumption about every binary sharing the pixi cache root,
    /// not a fact -- several binaries run concurrently against one root in
    /// this campaign. C11 made the built-output store *unmisreadable* rather
    /// than merely unreachable; this does the same here.
    #[test]
    fn p6ac_an_entry_stamped_with_another_closure_is_refused_not_adopted() {
        let dir = tmp_dir("p6ac-stamp");
        let path = cache_path(&dir, "isaaclab-hover-pack", "3.11", "linux-64");
        let _ = std::fs::remove_file(&path);
        let key = key_for("strict");

        let specs = ["torch==2.5.1".to_string()];
        let digest = probe_digest("auto_route_joint_solve", "universe-rev-7", specs.iter());
        let mine = EntryStamp::new("auto_route_joint_solve", "universe-rev-7", specs.iter());
        // Same address, a DIFFERENT closure behind it: another universe, and
        // a different spec set. Only a writer with a different address rule
        // can produce this -- which is exactly the case the stamp is for.
        let theirs = EntryStamp::new(
            "auto_route_joint_solve",
            "universe-rev-2",
            ["torch==2.7.0".to_string()].iter(),
        );
        assert_ne!(mine, theirs);

        {
            let writer = RouteProbeCache::open(path.clone(), key.clone());
            writer.record(&digest, &theirs, CachedVerdict::Sat);
        }

        let reader = RouteProbeCache::open(path.clone(), key.clone());
        assert_eq!(reader.len(), 1, "the entry IS on disk at our address");
        assert_eq!(
            reader.lookup(&digest, &mine),
            None,
            "a verdict computed from another closure must never be adopted",
        );
        assert_eq!(reader.refusals(), 1, "and the refusal must be counted");
        assert_eq!(reader.stats(), (0, 1), "a refusal is a miss, not a hit");

        // Control: the SAME store answers the stamp it actually holds.
        assert_eq!(
            reader.lookup(&digest, &theirs),
            Some(CachedVerdict::Sat),
            "refusal must be about the stamp, not about refusing everything",
        );
        assert_eq!(reader.refusals(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A v3 file has no stamps at all. Its entries decode with an EMPTY stamp,
    /// which can never be a reader's, so nothing from it is adopted -- and the
    /// SCHEMA bump discards the file outright before that even matters. Both
    /// belts are asserted here because only one of them survives a future
    /// schema bump that forgets the other.
    #[test]
    fn p6ac_an_unstamped_legacy_entry_is_never_adopted() {
        let dir = tmp_dir("p6ac-legacy");
        let path = cache_path(&dir, "legacy-pack", "3.11", "linux-64");
        let key = key_for("strict");
        let specs = ["torch==2.5.1".to_string()];
        let digest = probe_digest("auto_route_joint_solve", "universe-rev-7", specs.iter());
        let mine = EntryStamp::new("auto_route_joint_solve", "universe-rev-7", specs.iter());

        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // Hand-written v3 shape: the value is the bare verdict, no stamp.
        std::fs::write(
            &path,
            format!(
                r#"{{"schema":"{SCHEMA}","key":"{key}","entries":{{"{digest}":{{"verdict":"Sat"}}}}}}"#
            ),
        )
        .unwrap();

        let reader = RouteProbeCache::open(path.clone(), key);
        assert_eq!(reader.len(), 1, "the unstamped entry decoded");
        assert_eq!(
            reader.lookup(&digest, &mine),
            None,
            "an entry that cannot say what it was computed from is not evidence",
        );
        assert_eq!(reader.refusals(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `Skipped` is indecisive and must never be memoized.
    #[test]
    fn skipped_verdicts_are_not_cached() {
        assert!(
            CachedVerdict::from_verdict(&crate::uv_closure::CoInstallVerdict::Skipped(
                "no repodata".into()
            ))
            .is_none()
        );
        assert_eq!(
            CachedVerdict::from_verdict(&crate::uv_closure::CoInstallVerdict::Sat),
            Some(CachedVerdict::Sat),
        );
    }

    /// The question digest is order-insensitive but content-sensitive.
    #[test]
    fn probe_digest_is_order_insensitive_and_content_sensitive() {
        let a = probe_digest("solve", "u1", ["b==1", "a==2"].iter());
        let b = probe_digest("solve", "u1", ["a==2", "b==1"].iter());
        let c = probe_digest("solve", "u1", ["a==2", "b==2"].iter());
        let d = probe_digest("standalone", "u1", ["a==2", "b==1"].iter());
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d, "the stage tag is part of the question");
    }

    /// The candidate universe is per-ENTRY now (v2): a changed reachable
    /// universe must change the entry key, and a changed WORKSPACE PATH must
    /// not. This is the guard for the measured defect -- job 5611846 (fresh
    /// workspace, warm shared caches) re-executed all 315 probes because the
    /// universe used to live in the file-level validity key and was derived
    /// from the repodata cache file's mtime.
    #[test]
    fn universe_is_part_of_the_entry_key_and_the_workspace_path_is_not() {
        let specs = ["numpy==2.1", "python==3.12"];
        let same = probe_digest("solve", "universe-abc", specs.iter());
        assert_eq!(
            same,
            probe_digest("solve", "universe-abc", specs.iter()),
            "an unchanged universe must reproduce the entry key",
        );
        assert_ne!(
            same,
            probe_digest("solve", "universe-xyz", specs.iter()),
            "a changed candidate universe must change the entry key",
        );

        // Two workspaces at different paths, same everything else: the
        // validity key must be identical. Before v2 it could not be --
        // `repodata_identity` fed it the cache file's length and mtime.
        let workspace_a = std::path::Path::new("/oscar/ws.RUN-A-1/pixi.toml");
        let workspace_b = std::path::Path::new("/oscar/ws.RUN-B-2/deeper/pixi.toml");
        let key_a = validity_key(
            &["https://prefix.dev/conda-forge/".to_string()],
            "3.12",
            "linux-64",
            &[
                ("policy", vec!["strict".to_string()]),
                (
                    "workspace-deps",
                    vec![format!("root={}", workspace_a.parent().is_some())],
                ),
            ],
        );
        let key_b = validity_key(
            &["https://prefix.dev/conda-forge/".to_string()],
            "3.12",
            "linux-64",
            &[
                ("policy", vec!["strict".to_string()]),
                (
                    "workspace-deps",
                    vec![format!("root={}", workspace_b.parent().is_some())],
                ),
            ],
        );
        assert_eq!(
            key_a, key_b,
            "the workspace path must not reach the validity key",
        );
    }
    /// Guard (iii): two writers of the SAME verdict file, each holding
    /// entries the other never saw, must both survive.
    ///
    /// This is the shape a relock actually takes -- one worker per bundle
    /// over a shared cache dir, several of them probing the same
    /// (bundle, python, subdir) file. `persist` used to render the whole
    /// file from the writing handle's own in-memory snapshot, so whoever
    /// renamed last erased the other's verdicts, invisibly: the only
    /// symptom is a re-probe next run.
    ///
    /// Part 1 is the deterministic interleave (B opened before A wrote, so
    /// B's snapshot cannot contain A's entry). Part 2 is a real thread
    /// storm, which additionally exercises the advisory lock: with the lock
    /// removed the read-modify-write races and entries go missing.
    #[test]
    fn concurrent_persists_of_disjoint_entries_all_survive() {
        let dir = tmp_dir("concurrent-persist");
        let path = cache_path(&dir, "isaac-pack-latest", "3.12", "linux-64");
        let _ = std::fs::remove_file(&path);
        let key = key_for("strict");
        let question = |n: usize| {
            probe_digest(
                "auto_route_joint_solve",
                "universe-rev-1",
                [format!("absl-py=={n}.0")].iter(),
            )
        };
        let stamp = |n: usize| {
            EntryStamp::new(
                "auto_route_joint_solve",
                "universe-rev-1",
                [format!("absl-py=={n}.0")].iter(),
            )
        };

        // --- Part 1: deterministic interleave. ---
        let writer_a = RouteProbeCache::open(path.clone(), key.clone());
        let writer_b = RouteProbeCache::open(path.clone(), key.clone());
        writer_a.record(&question(1), &stamp(1), CachedVerdict::Sat);
        // B's snapshot still holds nothing of A's; the old persist rendered
        // the file from exactly that snapshot.
        writer_b.record(
            &question(2),
            &stamp(2),
            CachedVerdict::Unsat(vec!["no".to_string()]),
        );

        let reader = RouteProbeCache::open(path.clone(), key.clone());
        assert_eq!(
            reader.lookup(&question(1), &stamp(1)),
            Some(CachedVerdict::Sat),
            "the first writer's verdict was erased by the second writer's persist",
        );
        assert_eq!(
            reader.lookup(&question(2), &stamp(2)),
            Some(CachedVerdict::Unsat(vec!["no".to_string()])),
            "the second writer's own verdict must be on disk",
        );

        // --- Part 2: eight concurrent writers, five entries each. ---
        let _ = std::fs::remove_file(&path);
        std::thread::scope(|scope| {
            for writer in 0..8usize {
                let path = path.clone();
                let key = key.clone();
                scope.spawn(move || {
                    let cache = RouteProbeCache::open(path, key);
                    for entry in 0..5usize {
                        let n = 100 + writer * 5 + entry;
                        cache.record(&question(n), &stamp(n), CachedVerdict::Sat);
                    }
                });
            }
        });

        let reader = RouteProbeCache::open(path.clone(), key.clone());
        let missing: Vec<usize> = (0..40)
            .map(|i| 100 + i)
            .filter(|n| reader.lookup(&question(*n), &stamp(*n)).is_none())
            .collect();
        assert!(
            missing.is_empty(),
            "{} of 40 concurrently written verdicts were lost: {missing:?}",
            missing.len(),
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ===================================================================
    // N27-RETREAD-160 (ROUTECACHE-1). The shared, durable route-probe store.
    //
    // Every guard below is falsified by one of the two mutants the lane
    // brief names, and each test says which:
    //   MUTANT A -- `shared_cache_path` puts the bundle back in the address.
    //   MUTANT B -- `EntryStamp::new` drops the universe from the stamp.
    // ===================================================================

    fn shared_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "retread-route-probe-store-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("shared store root");
        dir
    }

    /// GUARD (a). Two BUNDLES asking one overlapping question consult ONE
    /// file and the second one HITS -- exactly one probe is executed for the
    /// shared question.
    ///
    /// This is the 29.5 % PACKPROF-1 measured (49 of 166 (bundle, digest)
    /// pairs asked by a bundle that was not the first to ask), and it is the
    /// whole reason the store is shared.
    ///
    /// RED UNDER MUTANT A: with the bundle back in the address the two
    /// bundles open different files, the second executes its own probe, and
    /// `executed` is 2 rather than 1.
    #[test]
    fn n27_160_two_bundles_with_an_overlapping_question_execute_one_probe() {
        let root = shared_root("overlap");
        let key = key_for("strict");

        // The address the two bundles resolve to. The bundle is NOT an
        // argument -- that is the property under test.
        let path_a = shared_cache_path(&root, &key, "3.11", "linux-64");
        let path_b = shared_cache_path(&root, &key, "3.11", "linux-64");
        assert_eq!(
            path_a, path_b,
            "two bundles under one validity key must address ONE shared file",
        );
        assert!(
            !path_a.to_string_lossy().contains("pack"),
            "the shared address must not name a bundle: {}",
            path_a.display(),
        );

        let shared_specs = ["numpy==1.26.4".to_string(), "torch==2.7.0".to_string()];
        let shared_digest =
            probe_digest("auto_route_joint_solve", "universe-rev-1", shared_specs.iter());
        let shared_stamp =
            EntryStamp::new("auto_route_joint_solve", "universe-rev-1", shared_specs.iter());
        // Something only the FIRST bundle asks, so the test cannot pass by
        // the two bundles happening to have identical question sets.
        let only_a = ["skrl==1.4.3".to_string()];
        let only_a_digest = probe_digest("auto_route_joint_solve", "universe-rev-1", only_a.iter());
        let only_a_stamp =
            EntryStamp::new("auto_route_joint_solve", "universe-rev-1", only_a.iter());

        let mut executed = 0usize;
        {
            let cache = RouteProbeCache::open_labelled(
                shared_cache_path(&root, &key, "3.11", "linux-64"),
                key.clone(),
                "isaaclab-2-3x-pack".to_string(),
                StoreMode::Shared,
            );
            for (digest, stamp) in [
                (&shared_digest, &shared_stamp),
                (&only_a_digest, &only_a_stamp),
            ] {
                if cache.lookup(digest, stamp).is_none() {
                    executed += 1;
                    cache.record(digest, stamp, CachedVerdict::Sat);
                }
            }
            assert_eq!(cache.published(), 2, "bundle A publishes both verdicts");
        }
        assert_eq!(executed, 2, "bundle A is cold and pays for both questions");

        let cache_b = RouteProbeCache::open_labelled(
            shared_cache_path(&root, &key, "3.11", "linux-64"),
            key.clone(),
            "isaac-pack-latest".to_string(),
            StoreMode::Shared,
        );
        let mut executed_b = 0usize;
        if cache_b.lookup(&shared_digest, &shared_stamp).is_none() {
            executed_b += 1;
        }
        assert_eq!(
            executed_b, 0,
            "the second bundle must HIT the question the first already answered",
        );
        assert_eq!(
            cache_b.stats(),
            (1, 0),
            "the hit is counted as a hit, not as a miss",
        );
        assert_eq!(cache_b.published(), 0, "a hit publishes nothing");
        assert!(
            cache_b.store_row().contains("consulted=1 hit=1 miss=0 published=0"),
            "the printed row must carry the hit: {}",
            cache_b.store_row(),
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// GUARD (b). A rolled universe is a MISS and a REPUBLISH, and the entry
    /// the roll invalidated is QUARANTINED, never deleted.
    ///
    /// Two halves, because the universe reaches the store two ways:
    ///   1. it is in the ADDRESS, so a roll simply asks a new question --
    ///      the old entry is still in the file afterwards; and
    ///   2. it is in the STAMP, so an entry that somehow sits AT the reader's
    ///      address with another universe is refused, copied into
    ///      `quarantine/`, dropped from the live file, and republished.
    ///
    /// RED UNDER MUTANT B: with the universe out of the stamp, half 2's
    /// planted entry compares EQUAL to the reader's stamp and is adopted --
    /// no refusal, no quarantine file, and the assert on `refusals()` fails.
    #[test]
    fn n27_160_a_rolled_universe_misses_republishes_and_quarantines_the_old_entry() {
        let root = shared_root("roll");
        let key = key_for("strict");
        let path = shared_cache_path(&root, &key, "3.11", "linux-64");
        let specs = ["numpy==1.26.4".to_string()];

        let d_old = probe_digest("auto_route_joint_solve", "universe-rev-1", specs.iter());
        let s_old = EntryStamp::new("auto_route_joint_solve", "universe-rev-1", specs.iter());
        let d_new = probe_digest("auto_route_joint_solve", "universe-rev-2", specs.iter());
        let s_new = EntryStamp::new("auto_route_joint_solve", "universe-rev-2", specs.iter());
        assert_ne!(d_old, d_new, "the universe is part of the entry address");

        {
            let cache = RouteProbeCache::open_labelled(
                path.clone(),
                key.clone(),
                "isaaclab-2-3x-pack".to_string(),
                StoreMode::Shared,
            );
            cache.record(&d_old, &s_old, CachedVerdict::Sat);
        }

        // Half 1: the roll is a miss at a NEW address, and the old entry
        // survives untouched.
        {
            let cache = RouteProbeCache::open_labelled(
                path.clone(),
                key.clone(),
                "isaaclab-2-3x-pack".to_string(),
                StoreMode::Shared,
            );
            assert!(
                cache.lookup(&d_new, &s_new).is_none(),
                "a rolled universe must MISS",
            );
            cache.record(&d_new, &s_new, CachedVerdict::Sat);
            assert_eq!(cache.published(), 1, "the roll republishes exactly once");
            assert!(
                cache.lookup(&d_old, &s_old).is_some(),
                "the pre-roll entry is not deleted by the roll",
            );
        }

        // Half 2: an entry AT the reader's address whose stamp names another
        // universe -- refused, quarantined, dropped, republished.
        let planted = shared_root("roll-planted");
        let planted_path = shared_cache_path(&planted, &key, "3.11", "linux-64");
        {
            let cache = RouteProbeCache::open_labelled(
                planted_path.clone(),
                key.clone(),
                "isaaclab-2-3x-pack".to_string(),
                StoreMode::Shared,
            );
            // Deliberately stored at the NEW address under the OLD stamp:
            // the shape a second writer deriving the address by another rule
            // would leave behind.
            cache.record(&d_new, &s_old, CachedVerdict::Sat);
        }
        let reader = RouteProbeCache::open_labelled(
            planted_path.clone(),
            key.clone(),
            "isaac-pack-latest".to_string(),
            StoreMode::Shared,
        );
        assert!(
            reader.lookup(&d_new, &s_new).is_none(),
            "an entry stamped with another universe must be refused, not adopted",
        );
        assert_eq!(reader.refusals(), 1, "the refusal is counted");
        assert!(
            reader.store_row().contains("refused=1 reason=stage:0,universe:1,question:0"),
            "the row must name the refusal reason: {}",
            reader.store_row(),
        );

        // NOTHING is quarantined by the READ. A refused entry is still the
        // valid answer for the writer whose stamp it carries -- p6ac's own
        // control asserts that -- so a read never destroys it.
        let quarantine = planted_path.parent().unwrap().join(QUARANTINE);
        assert!(
            !quarantine.exists(),
            "a refused READ must not mutate the store",
        );

        // The REPUBLISH displaces it, and the displacement is what is copied
        // aside.
        reader.record(&d_new, &s_new, CachedVerdict::Sat);
        assert_eq!(reader.published(), 1, "the refusal is followed by a republish");
        assert_eq!(
            reader.lookup(&d_new, &s_new),
            Some(CachedVerdict::Sat),
            "the republished verdict is the reader's own",
        );

        let quarantined: Vec<PathBuf> = std::fs::read_dir(&quarantine)
            .expect("quarantine directory")
            .map(|e| e.expect("quarantine entry").path())
            .collect();
        assert_eq!(
            quarantined.len(),
            1,
            "the displaced entry is copied aside, never deleted: {quarantined:?}",
        );
        let kept = std::fs::read_to_string(&quarantined[0]).expect("quarantined entry");
        assert!(
            kept.contains("universe-rev-1"),
            "the quarantined copy is the displaced entry itself: {kept}",
        );

        // Re-recording the IDENTICAL entry displaces nothing worth keeping, so
        // the quarantine does not grow -- a store whose evidence directory
        // fills with non-events is a store nobody reads.
        reader.record(&d_new, &s_new, CachedVerdict::Sat);
        assert_eq!(
            std::fs::read_dir(&quarantine).unwrap().count(),
            1,
            "an idempotent re-record must not quarantine anything",
        );

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&planted);
    }

    /// GUARD (c). With the key ABSENT the store does not exist: the address
    /// is the pre-existing per-bundle, job-scoped one, the on-disk bytes are
    /// the pre-existing shape, and NOTHING is printed.
    ///
    /// The exact JSON text is pinned rather than described. A new field on
    /// `VerdictFile` or `StoredEntry` -- the ordinary way a store's wire
    /// format drifts -- turns this red even though every behavioural
    /// assertion around it would still pass.
    #[test]
    fn n27_160_with_the_key_absent_the_behaviour_is_byte_identical() {
        let dir = tmp_dir("off-identical");
        let key = key_for("strict");

        let path = cache_path(&dir, "isaaclab-2.3x-pack", "3.11.9", "linux-64");
        assert!(
            path.ends_with("retread-route-probe-verdicts/isaaclab-2.3x-pack-py3.11-linux-64.json"),
            "the per-job address is unchanged: {}",
            path.display(),
        );

        let specs = ["numpy==1.26.4".to_string()];
        let digest = probe_digest("s", "u", specs.iter());
        let stamp = EntryStamp::new("s", "u", specs.iter());
        let cache = RouteProbeCache::open(path.clone(), key.clone());
        assert_eq!(
            cache.mode,
            StoreMode::PerJob,
            "the default handle is the per-job one",
        );
        cache.record(&digest, &stamp, CachedVerdict::Sat);
        drop(cache);

        let written = std::fs::read_to_string(&path).expect("verdict file");
        let expected = format!(
            "{{\"schema\":\"{SCHEMA}\",\"key\":\"{key}\",\"entries\":{{\"{digest}\":\
             {{\"verdict\":\"Sat\",\"stamp\":{{\"stage\":\"s\",\"universe\":\"u\",\
             \"question\":\"numpy==1.26.4\"}}}}}}}}"
        );
        assert_eq!(
            written, expected,
            "the per-job on-disk bytes must be exactly what 0be408a wrote",
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// GUARD (d). TWO PROCESSES, not two threads: the file is a shared
    /// resource across `pixi lock`'s concurrent backends, and threads share
    /// the in-process `Mutex` that makes the interesting failure impossible.
    ///
    /// Each child publishes a disjoint block of verdicts into ONE shared
    /// file. Afterwards the file parses (no torn write), holds every verdict
    /// both children published (no lost update), and a third reader HITS all
    /// of them.
    ///
    /// The child role is this same test binary re-executed with
    /// `N27_160_CHILD` set, so the code under test in the child is the
    /// production `persist` path and not a re-implementation of it.
    #[test]
    fn n27_160_two_processes_publish_into_one_file_without_tearing_it() {
        let key = key_for("strict");
        let question = |n: usize| {
            probe_digest(
                "auto_route_joint_solve",
                "universe-rev-1",
                [format!("pkg-{n}==1.0")].iter(),
            )
        };
        let stamp = |n: usize| {
            EntryStamp::new(
                "auto_route_joint_solve",
                "universe-rev-1",
                [format!("pkg-{n}==1.0")].iter(),
            )
        };

        if let Ok(spec) = std::env::var("N27_160_CHILD") {
            let (path, base) = spec.split_once('|').expect("child spec is <path>|<base>");
            let base: usize = base.parse().expect("child base");
            let cache = RouteProbeCache::open_labelled(
                PathBuf::from(path),
                key,
                format!("child-{base}"),
                StoreMode::Shared,
            );
            for i in 0..40usize {
                let n = base + i;
                cache.record(&question(n), &stamp(n), CachedVerdict::Sat);
            }
            return;
        }

        let root = shared_root("two-process");
        let path = shared_cache_path(&root, &key, "3.11", "linux-64");
        let exe = std::env::current_exe().expect("test binary");
        let children: Vec<std::process::Child> = [1000usize, 2000usize]
            .into_iter()
            .map(|base| {
                std::process::Command::new(&exe)
                    .args([
                        "--exact",
                        "route_probe_cache::tests::\
                         n27_160_two_processes_publish_into_one_file_without_tearing_it",
                        "--nocapture",
                    ])
                    .env("N27_160_CHILD", format!("{}|{base}", path.display()))
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .expect("spawn child writer")
            })
            .collect();
        for mut child in children {
            let status = child.wait().expect("child writer");
            assert!(status.success(), "child writer failed: {status}");
        }

        // No torn file: it parses, at the right schema and key.
        let text = std::fs::read_to_string(&path).expect("shared verdict file");
        let file: VerdictFile = serde_json::from_str(&text).expect("the shared file is not torn");
        assert_eq!(file.schema, SCHEMA);
        assert_eq!(file.key, key);
        assert_eq!(
            file.entries.len(),
            80,
            "both processes' publications survive the union write",
        );

        // No leftover staging files beside it.
        let strays: Vec<String> = std::fs::read_dir(path.parent().unwrap())
            .expect("store dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp"))
            .collect();
        assert!(strays.is_empty(), "publisher temp files left behind: {strays:?}");

        // A third reader hits every one of them.
        let reader = RouteProbeCache::open_labelled(
            path.clone(),
            key.clone(),
            "reader".to_string(),
            StoreMode::Shared,
        );
        let missing: Vec<usize> = [1000usize, 2000usize]
            .into_iter()
            .flat_map(|base| (0..40).map(move |i| base + i))
            .filter(|n| reader.lookup(&question(*n), &stamp(*n)).is_none())
            .collect();
        assert!(
            missing.is_empty(),
            "{} of 80 cross-process verdicts were lost: {missing:?}",
            missing.len(),
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// The lane's standing separation, held as its own guard so it cannot be
    /// undone silently: the SHARED address names no bundle, while the
    /// per-job address still does.
    ///
    /// RED UNDER MUTANT A, directly.
    #[test]
    fn n27_160_the_shared_address_is_bundle_free_and_the_per_job_one_is_not() {
        let root = shared_root("address");
        let key = key_for("strict");
        let a = shared_cache_path(&root, &key, "3.11", "linux-64");
        let b = shared_cache_path(&root, &key, "3.11", "linux-64");
        assert_eq!(a, b);
        assert_eq!(
            a.parent().unwrap().file_name().unwrap(),
            std::ffi::OsStr::new(STORE_DIR),
            "the shared store lives under its own named directory",
        );
        assert_eq!(
            a.file_name().unwrap().to_string_lossy(),
            format!("{key}-py3.11-linux-64.json"),
            "the validity key IS the shared filename, so an open can never \
             discard another bundle's verdicts",
        );

        // The declared control WINS over the harness rung, and the harness
        // rung is named here so the guard fails if it is ever renamed out
        // from under the launcher that exports it.
        assert_eq!(
            resolve_store_root(Some(&root)),
            Some(root.clone()),
            "the config key is the control and it wins",
        );
        assert_eq!(STORE_ENV, "RETREAD_ROUTE_PROBE_STORE");

        let per_job_x = cache_path(&root, "pack-x", "3.11", "linux-64");
        let per_job_y = cache_path(&root, "pack-y", "3.11", "linux-64");
        assert_ne!(
            per_job_x, per_job_y,
            "the per-job address still separates bundles -- that is what \
             `off` means",
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// REAP-2 GUARD. THE WRITER'S LOCK AND THE REAPER'S LOCK ARE ONE PATH.
    ///
    /// This is the guard for the correction, and it is written so it FAILS if
    /// anybody ever re-inlines the writer's lock path or points the reaper at
    /// this campaign's OTHER convention. Half 1 states the shape as a literal
    /// (`<stem>.lock`, not `.{file}.lock`) so a silent convention swap cannot
    /// stay green by both sides moving together; half 2 runs a real publish and
    /// asserts the file `persist` created is the file `lock_path_for` names.
    ///
    /// RED WITHOUT THE FIX: with `persist` still computing its own path, half 2
    /// passes only by coincidence -- and it is exactly that coincidence the
    /// reaper cannot rely on, which is why half 1 pins the shape too.
    #[test]
    fn reap2_the_writer_and_the_reaper_share_one_lock_path() {
        let root = shared_root("lockpath");
        let key = key_for("strict");
        let path = shared_cache_path(&root, &key, "3.12", "linux-64");

        // Half 1: the SHAPE, as a literal. `<stem>.lock` beside the entry.
        let named = lock_path_for(&path);
        assert_eq!(
            named.file_name().and_then(|n| n.to_str()),
            Some(format!("{}-py3.12-linux-64.lock", key).as_str()),
            "the store's lock is <stem>.lock beside the entry, not a dotfile: {}",
            named.display(),
        );
        assert_eq!(
            named.parent(),
            path.parent(),
            "the lock must be a sibling of the entry it guards",
        );
        assert!(
            !named
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with('.')),
            "a dot-prefixed lock would be invisible to the reaper's `.json` filter",
        );

        // Half 2: a real publish, and the file it leaves behind.
        assert!(!named.exists(), "nothing has published yet");
        let specs = ["numpy==1.26.4".to_string()];
        let digest = probe_digest("auto_route_joint_solve", "u1", specs.iter());
        let stamp = EntryStamp::new("auto_route_joint_solve", "u1", specs.iter());
        {
            let cache = RouteProbeCache::open_labelled(
                path.clone(),
                key.clone(),
                "pack-a".to_string(),
                StoreMode::Shared,
            );
            cache.record(&digest, &stamp, CachedVerdict::Sat);
        }
        assert!(path.is_file(), "the entry must have been published");
        assert!(
            named.is_file(),
            "`persist` must have created the lock the reaper tries: {}",
            named.display(),
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// REAP-2 GUARD. `entry_is_unreachable` answers "can any reader address
    /// this entry again" from the record alone, and the three cases are the
    /// three the store actually produces.
    ///
    /// RED IF THE PREDICATE IS INVERTED OR STUBBED: a stub returning `false`
    /// fails cases 2 and 3, one returning `true` fails case 1.
    #[test]
    fn reap2_an_entry_is_unreachable_exactly_when_its_stamp_misses_its_address() {
        let specs = ["numpy==1.26.4".to_string(), "torch==2.7.0".to_string()];
        let digest = probe_digest("auto_route_joint_solve", "universe-rev-1", specs.iter());
        let stamp = EntryStamp::new("auto_route_joint_solve", "universe-rev-1", specs.iter());

        // 1. The normal entry: the stamp reproduces its own address.
        assert!(
            !entry_is_unreachable(&digest, &stamp),
            "a self-consistent entry is reachable and must never be reaped by reason=universe",
        );
        // 2. A v3 remnant or a truncated record: `serde(default)` gives it an
        //    EMPTY stamp, which can reproduce no address at all.
        assert!(
            entry_is_unreachable(&digest, &EntryStamp::default()),
            "an unstamped entry can never be adopted, so nothing can reach it",
        );
        // 3. The p6ab shape: an entry sitting at an address its own universe
        //    does not hash to.
        let wrong = EntryStamp::new("auto_route_joint_solve", "universe-rev-2", specs.iter());
        assert!(
            entry_is_unreachable(&digest, &wrong),
            "an entry whose universe does not hash to its key is unreachable",
        );
        // ... and it IS reachable at its own address, so the predicate is
        // about the PAIR and not about the stamp in isolation.
        let wrong_digest = probe_digest("auto_route_joint_solve", "universe-rev-2", specs.iter());
        assert!(
            !entry_is_unreachable(&wrong_digest, &wrong),
            "the predicate must compare the stamp against THIS key, not judge the stamp alone",
        );
    }

    /// REAP-2 GUARD. The store's own filter: only a published `*.json` is an
    /// entry, and every other thing the store directory legitimately holds is
    /// NOT one. Without this the reaper would quarantine live locks, a
    /// concurrent publish's temp file, or the quarantine directory itself.
    #[test]
    fn reap2_only_a_published_json_is_an_entry() {
        for name in [
            "abc-py3.12-linux-64.json",
            "abc-py3.11-osx-arm64.json",
        ] {
            assert!(is_verdict_entry_name(name), "{name} is an entry");
        }
        for name in [
            // the writer's advisory lock
            "abc-py3.12-linux-64.lock",
            // a concurrent `persist`'s temp file (`with_extension("tmp<pid>")`)
            "abc-py3.12-linux-64.tmp12345",
            // the reaper's own store-wide try-lock
            ROUTE_PROBE_REAP_LOCK_NAME,
            // the quarantine directory
            QUARANTINE,
            // a dot-prefixed json is a sidecar by convention, never an entry
            ".abc-py3.12-linux-64.json",
        ] {
            assert!(!is_verdict_entry_name(name), "{name} is NOT an entry");
        }
    }
}
