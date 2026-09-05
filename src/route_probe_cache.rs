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
            question: normalized_question(specs).join("\u{1f}"),
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
    let minor = python
        .split('.')
        .take(2)
        .collect::<Vec<_>>()
        .join(".")
        .replace(['/', '\\'], "-");
    let safe_bundle = bundle.replace(['/', '\\', ' '], "-");
    cache_dir
        .join(DIR)
        .join(format!("{safe_bundle}-py{minor}-{subdir}.json"))
}

/// Process-lifetime handle over one bundle's verdict file.
#[derive(Debug)]
pub(crate) struct RouteProbeCache {
    path: PathBuf,
    key: String,
    entries: Mutex<BTreeMap<String, StoredEntry>>,
    hits: AtomicUsize,
    misses: AtomicUsize,
    /// Entries that were PRESENT at the reader's address but whose stamped
    /// closure disagreed with the reader's. Counted separately from a miss so
    /// "this never happens" stays a measurable claim rather than a belief.
    refusals: AtomicUsize,
}

impl RouteProbeCache {
    /// Load (or start empty on a key/schema mismatch -- that IS the
    /// invalidation). A read error is non-fatal: worst case is a cold
    /// probe round, never a wrong verdict.
    pub(crate) fn open(path: PathBuf, key: String) -> Self {
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
            entries: Mutex::new(entries),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            refusals: AtomicUsize::new(0),
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

    pub(crate) fn record(&self, digest: &str, stamp: &EntryStamp, verdict: CachedVerdict) {
        let snapshot = {
            let mut entries = self.entries.lock().expect("route probe cache mutex");
            entries.insert(
                digest.to_string(),
                StoredEntry {
                    verdict,
                    stamp: stamp.clone(),
                },
            );
            entries.clone()
        };
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
        let lock_path = self.path.with_extension("lock");
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
}
