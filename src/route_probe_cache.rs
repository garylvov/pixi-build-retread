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
//!     providers).
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
const SCHEMA: &str = "v1-route-probe-verdicts";

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

#[derive(Debug, Default, Serialize, Deserialize)]
struct VerdictFile {
    #[serde(default)]
    schema: String,
    /// Hex digest of every non-question input (channels + repodata
    /// identity + target + policy). A mismatch discards the file.
    #[serde(default)]
    key: String,
    #[serde(default)]
    entries: BTreeMap<String, CachedVerdict>,
}

/// Hex sha256 of one probe QUESTION: the stage tag plus the normalized,
/// sorted, deduplicated spec strings.
pub(crate) fn probe_digest<S: AsRef<str>>(stage: &str, specs: impl Iterator<Item = S>) -> String {
    let mut normalized: Vec<String> = specs.map(|s| s.as_ref().trim().to_string()).collect();
    normalized.sort();
    normalized.dedup();
    let mut h = Sha256::new();
    h.update(b"retread-probe-question\0");
    h.update(stage.as_bytes());
    h.update([0xffu8]);
    for spec in &normalized {
        h.update(spec.as_bytes());
        h.update([0u8]);
    }
    format!("{:x}", h.finalize())
}

/// Hex sha256 of the cache-VALIDITY key. `policy_fields` is an ordered
/// list of `(tag, values)` describing everything except the question.
pub(crate) fn validity_key(
    channels: &[String],
    repodata_identity: &str,
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
    field("repodata", &[repodata_identity.to_string()]);
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
    entries: Mutex<BTreeMap<String, CachedVerdict>>,
    hits: AtomicUsize,
    misses: AtomicUsize,
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
        }
    }

    pub(crate) fn lookup(&self, digest: &str) -> Option<CachedVerdict> {
        let hit = self
            .entries
            .lock()
            .expect("route probe cache mutex")
            .get(digest)
            .cloned();
        if hit.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    pub(crate) fn record(&self, digest: &str, verdict: CachedVerdict) {
        let snapshot = {
            let mut entries = self.entries.lock().expect("route probe cache mutex");
            entries.insert(digest.to_string(), verdict);
            entries.clone()
        };
        self.persist(snapshot);
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

    fn persist(&self, entries: BTreeMap<String, CachedVerdict>) {
        let file = VerdictFile {
            schema: SCHEMA.to_string(),
            key: self.key.clone(),
            entries,
        };
        let Ok(text) = serde_json::to_string(&file) else {
            return;
        };
        if let Some(parent) = self.path.parent() {
            if std::fs::create_dir_all(parent).is_err() {
                return;
            }
        }
        let tmp = self.path.with_extension(format!("tmp{}", std::process::id()));
        if std::fs::write(&tmp, text.as_bytes()).is_ok() {
            let _ = std::fs::rename(&tmp, &self.path);
        }
        let _ = std::fs::remove_file(&tmp);
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

    fn key_for(repodata: &str) -> String {
        validity_key(
            &["https://prefix.dev/conda-forge/".to_string()],
            repodata,
            "3.12",
            "linux-64",
            &[("policy", vec!["strict".to_string()])],
        )
    }

    /// Guard (a): a second run over the same probe set executes ZERO
    /// probes and returns identical verdicts.
    #[test]
    fn second_run_over_the_same_probe_set_executes_no_probes() {
        let dir = tmp_dir("same-key");
        let path = cache_path(&dir, "isaac-pack-latest", "3.12", "linux-64");
        let key = key_for("repodata-rev-1");

        let questions: Vec<String> = (0..100)
            .map(|i| probe_digest("auto_route_joint_solve", [format!("absl-py=={i}.0")].iter()))
            .collect();

        let mut cold_executions = 0usize;
        {
            let cache = RouteProbeCache::open(path.clone(), key.clone());
            for digest in &questions {
                if cache.lookup(digest).is_none() {
                    cold_executions += 1;
                    cache.record(digest, CachedVerdict::ExactUnsat(vec!["no candidates".into()]));
                }
            }
            assert_eq!(cache.stats(), (0, 100), "cold run is all misses");
        }
        assert_eq!(cold_executions, 100);

        let mut warm_executions = 0usize;
        let cache = RouteProbeCache::open(path, key);
        for digest in &questions {
            match cache.lookup(digest) {
                Some(verdict) => assert_eq!(
                    verdict,
                    CachedVerdict::ExactUnsat(vec!["no candidates".into()]),
                    "replayed verdict must be identical",
                ),
                None => {
                    warm_executions += 1;
                    cache.record(digest, CachedVerdict::Sat);
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
        let questions: Vec<String> = (0..8)
            .map(|i| probe_digest("auto_route_joint_solve", [format!("absl-py=={i}.0")].iter()))
            .collect();
        {
            let cache = RouteProbeCache::open(path.clone(), key_for("repodata-rev-1"));
            for digest in &questions {
                cache.record(digest, CachedVerdict::Sat);
            }
            assert_eq!(cache.len(), 8);
        }
        // Same file, different repodata identity -> whole file discarded.
        let cache = RouteProbeCache::open(path.clone(), key_for("repodata-rev-2"));
        assert_eq!(cache.len(), 0, "a changed key must discard every verdict");
        let mut executions = 0usize;
        for digest in &questions {
            if cache.lookup(digest).is_none() {
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
        let a = probe_digest("solve", ["b==1", "a==2"].iter());
        let b = probe_digest("solve", ["a==2", "b==1"].iter());
        let c = probe_digest("solve", ["a==2", "b==2"].iter());
        let d = probe_digest("standalone", ["a==2", "b==1"].iter());
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d, "the stage tag is part of the question");
    }
}
