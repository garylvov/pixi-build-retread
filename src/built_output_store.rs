//! Shared, content-addressed store of already-computed BUILD OUTPUTS.
//!
//! # Why this exists
//!
//! `conda/outputs` is the expensive half of a `pixi lock` against a workspace
//! full of local path-source packs: it materializes wheels, runs the
//! auto-bundle cascade, and route-probes every candidate. Measured on the
//! canonical 27-environment manifest, that RPC is >90% of the backend window
//! and 14 of them run per lock.
//!
//! `handler::conda_outputs` already memoizes that result twice — once in
//! process (`CONDA_OUTPUTS_CACHE`) and once on disk for a sibling backend
//! process. Neither survives a fresh workspace:
//!
//! * the disk memo is written under the handler's `cache_dir`, which
//!   `fasttmp` redirects into a **job-scoped** namespace whenever the pixi
//!   cache lives on a slow filesystem, so nothing outlives the job; and
//! * its key folds the workspace `pixi.toml`'s **mtime** and the pack's
//!   **absolute `source_dir`**, both of which move when a workspace is staged
//!   at a new path — so even in a shared directory it could never hit.
//!
//! This module adds a third tier that is deliberately neither: a store root
//! the operator names (`retread-built-output-store`), keyed on a digest that
//! consults no path, no mtime and no job id — the same discipline
//! `courier_inputs_hash` already applies to the pack build string.
//!
//! # Layout
//!
//! ```text
//! <store>/<key>/outputs.json     the payload
//! <store>/<key>/COMPLETE         the marker; an entry without it is a MISS
//! <store>/tmp-<key>-<pid>/       a publisher's private staging dir
//! ```
//!
//! Publication is: write payload into `tmp-<key>-<pid>`, fsync it, rename the
//! directory to `<key>` (the first rename wins — a loser removes its own tmp
//! and treats the winner's entry as authoritative), then write the marker.
//! The marker is written LAST and inside the already-renamed directory, so a
//! reader can never observe a complete-looking entry with a partial payload:
//! a crash between rename and marker leaves an entry that every reader treats
//! as a miss and that the next publisher replaces.
//!
//! # What it is NOT
//!
//! It is not a correctness mechanism and never a source of truth. Every read
//! and write failure falls back to the ordinary cold compute. It is also not
//! consulted for an output that carries a job-local prepared plan — the same
//! `requires_prepared_plan` guard that already keeps those out of the
//! cross-process disk memo keeps them out of here.

use std::path::{Path, PathBuf};

/// Bumped whenever the payload's meaning or serialization changes. Folded
/// into every key, so an old entry is invisible rather than misread.
pub const SCHEMA: &str = "retread-built-output-store-v1";

/// The payload filename inside an entry.
const PAYLOAD: &str = "outputs.json";

/// The completeness marker. Written last; its absence means "miss".
const MARKER: &str = "COMPLETE";

/// A store root the operator opted into. Absent = today's behaviour exactly.
#[derive(Debug, Clone)]
pub struct BuiltOutputStore {
    root: PathBuf,
}

/// What a lookup did, so the caller can log one loud line either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lookup {
    Hit,
    /// No entry directory at all.
    Miss,
    /// The entry exists but has no marker (a crashed or in-flight publish) or
    /// its payload does not deserialize. Treated exactly like a miss, and the
    /// next successful publish replaces it.
    Incomplete,
}

impl BuiltOutputStore {
    /// Resolve the store from the pack's config, falling back to the legacy
    /// environment override. `None` means the feature is off, which is the
    /// default and is byte-for-byte the behaviour that shipped before it.
    ///
    /// The config key is the supported control (an argument, not an ambient
    /// environment variable, so a misspelling is a load error rather than a
    /// silent no-op); the env var stays only so an existing harness that sets
    /// it keeps working.
    pub fn from_config(configured: Option<&Path>) -> Option<Self> {
        let root = match configured {
            Some(path) => path.to_path_buf(),
            None => match std::env::var_os("RETREAD_BUILT_OUTPUT_STORE") {
                Some(value) if !value.is_empty() => PathBuf::from(value),
                _ => return None,
            },
        };
        Some(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn entry(&self, key: &str) -> PathBuf {
        self.root.join(key)
    }

    /// Read a stored payload. `Ok(None)` covers every "not usable" case; the
    /// [`Lookup`] tells the caller which one so the log line is honest.
    pub fn get(&self, key: &str) -> (Lookup, Option<Vec<u8>>) {
        let entry = self.entry(key);
        if !entry.is_dir() {
            return (Lookup::Miss, None);
        }
        if !entry.join(MARKER).is_file() {
            return (Lookup::Incomplete, None);
        }
        match std::fs::read(entry.join(PAYLOAD)) {
            Ok(bytes) => (Lookup::Hit, Some(bytes)),
            Err(_) => (Lookup::Incomplete, None),
        }
    }

    /// How long a publisher waits for a rival that already renamed its entry
    /// into place to write the marker, before concluding the entry is the
    /// residue of a crashed publish and replacing it. A publish writes the
    /// marker microseconds after the rename, so this only ever elapses for a
    /// genuinely abandoned entry.
    const MARKER_GRACE: std::time::Duration = std::time::Duration::from_millis(750);

    /// Publish `payload` under `key`.
    ///
    /// Returns `Ok(true)` when this call's own directory became the entry and
    /// `Ok(false)` when a concurrent publisher won the rename first — in
    /// which case this call removed its staging directory and left the
    /// winner's entry untouched. Both are success: the store holds exactly
    /// one entry for the key either way, and by construction both publishers
    /// computed the same bytes, because the key is a digest of the inputs.
    ///
    /// An INCOMPLETE entry (marker missing) is replaced, but only after
    /// [`Self::MARKER_GRACE`] has passed without a marker appearing — so a
    /// rival that is mid-publish is never destroyed, while the residue of a
    /// crashed publish never becomes a permanent hole in the store.
    pub fn publish(&self, key: &str, payload: &[u8]) -> std::io::Result<bool> {
        self.publish_inner(key, payload, true)
    }

    fn publish_inner(
        &self,
        key: &str,
        payload: &[u8],
        may_reclaim: bool,
    ) -> std::io::Result<bool> {
        let entry = self.entry(key);
        if entry.join(MARKER).is_file() {
            return Ok(false);
        }
        std::fs::create_dir_all(&self.root)?;
        // The staging name must be unique per CALL, not per process. Two
        // threads of one process publishing the same key share a pid, and a
        // pid-only name made them race on one staging directory -- the full
        // test suite caught exactly that under parallel load while the
        // filtered run passed. The counter plus a clock reading keeps threads
        // apart; the pid keeps processes apart.
        static NEXT_STAGING: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = format!(
            "{}-{}-{}",
            std::process::id(),
            NEXT_STAGING.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        );
        let tmp = self.root.join(format!("tmp-{key}-{unique}"));
        std::fs::create_dir_all(&tmp)?;
        let staged = tmp.join(PAYLOAD);
        std::fs::write(&staged, payload)?;
        // fsync the payload before it can be reached under the entry name, so
        // a reader that sees the marker cannot read a short file.
        if let Ok(file) = std::fs::File::open(&staged) {
            let _ = file.sync_all();
        }
        match std::fs::rename(&tmp, &entry) {
            Ok(()) => {
                // Marker LAST, and inside the directory that is already in
                // place: until it lands, every reader sees a miss.
                std::fs::write(entry.join(MARKER), SCHEMA.as_bytes())?;
                Ok(true)
            }
            Err(error) => {
                // Renaming onto a non-empty directory is how a loser finds
                // out it lost. Drop our staging dir either way.
                let _ = std::fs::remove_dir_all(&tmp);
                if !entry.exists() {
                    // Nothing is there, so the rename failed for a real
                    // filesystem reason. Report it; the caller falls back to
                    // an ordinary cold compute.
                    return Err(error);
                }
                let deadline = std::time::Instant::now() + Self::MARKER_GRACE;
                while std::time::Instant::now() < deadline {
                    if entry.join(MARKER).is_file() {
                        return Ok(false);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                if entry.join(MARKER).is_file() {
                    return Ok(false);
                }
                if !may_reclaim {
                    return Err(error);
                }
                // Abandoned: no marker after the grace window. Reclaim it.
                let _ = std::fs::remove_dir_all(&entry);
                self.publish_inner(key, payload, false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The repo idiom for test scratch space (see `import_scan`,
    /// `hermetic_build`): a uniquely named directory under the process temp
    /// dir, removed by the guard on drop so a failing assert cannot leak.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "retread-bos-{tag}-{}-{n}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("scratch dir");
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn store(dir: &Path) -> BuiltOutputStore {
        BuiltOutputStore::from_config(Some(dir)).expect("configured root yields a store")
    }

    #[test]
    fn unset_config_and_unset_env_means_no_store() {
        // The default must be "the feature does not exist", so a workspace
        // that never opts in behaves exactly as it did before.
        let previous = std::env::var_os("RETREAD_BUILT_OUTPUT_STORE");
        unsafe { std::env::remove_var("RETREAD_BUILT_OUTPUT_STORE") };
        let resolved = BuiltOutputStore::from_config(None);
        if let Some(previous) = previous {
            unsafe { std::env::set_var("RETREAD_BUILT_OUTPUT_STORE", previous) };
        }
        assert!(
            resolved.is_none(),
            "no config and no env must yield no store"
        );
    }

    #[test]
    fn a_published_entry_is_read_back_and_a_missing_one_is_a_miss() {
        let dir = Scratch::new("roundtrip");
        let store = store(dir.path());
        assert_eq!(store.get("k1").0, Lookup::Miss);
        assert!(store.publish("k1", b"payload-bytes").unwrap());
        let (lookup, bytes) = store.get("k1");
        assert_eq!(lookup, Lookup::Hit);
        assert_eq!(bytes.as_deref(), Some(&b"payload-bytes"[..]));
        assert_eq!(store.get("k2").0, Lookup::Miss);
    }

    #[test]
    fn an_entry_without_its_marker_is_a_miss_and_is_replaced() {
        // The crash window: the directory is renamed into place but the
        // marker was never written. A reader must not serve that payload,
        // and the next publisher must be able to fix it.
        let dir = Scratch::new("incomplete");
        let store = store(dir.path());
        let entry = dir.path().join("k1");
        std::fs::create_dir_all(&entry).unwrap();
        std::fs::write(entry.join(PAYLOAD), b"torn").unwrap();
        assert_eq!(store.get("k1").0, Lookup::Incomplete);
        assert!(
            store.get("k1").1.is_none(),
            "an unmarked entry serves nothing"
        );

        assert!(store.publish("k1", b"repaired").unwrap());
        let (lookup, bytes) = store.get("k1");
        assert_eq!(lookup, Lookup::Hit);
        assert_eq!(bytes.as_deref(), Some(&b"repaired"[..]));
    }

    #[test]
    fn a_marked_entry_with_no_payload_is_incomplete_not_a_hit() {
        let dir = Scratch::new("nopayload");
        let store = store(dir.path());
        let entry = dir.path().join("k1");
        std::fs::create_dir_all(&entry).unwrap();
        std::fs::write(entry.join(MARKER), SCHEMA).unwrap();
        assert_eq!(store.get("k1").0, Lookup::Incomplete);
    }

    #[test]
    fn concurrent_publishers_of_one_key_leave_exactly_one_entry() {
        // Eight threads publish the same key at once. Exactly one entry may
        // exist afterwards, it must be complete, and no `tmp-` staging
        // directory may be left behind.
        let dir = Scratch::new("concurrent");
        let root = dir.path().to_path_buf();
        let winners: Vec<bool> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let root = root.clone();
                    scope.spawn(move || {
                        BuiltOutputStore::from_config(Some(&root))
                            .unwrap()
                            .publish("shared-key", b"same-bytes")
                            .expect("publish must not error under contention")
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(
            winners.iter().any(|won| *won),
            "at least one publisher must have created the entry"
        );
        let entries: Vec<String> = std::fs::read_dir(&root)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            entries,
            vec!["shared-key".to_string()],
            "exactly one entry and no leftover staging dirs: {entries:?}"
        );
        let store = BuiltOutputStore::from_config(Some(&root)).unwrap();
        assert_eq!(store.get("shared-key").0, Lookup::Hit);
    }

    #[test]
    fn publishing_over_a_complete_entry_is_a_no_op() {
        let dir = Scratch::new("nooverwrite");
        let store = store(dir.path());
        assert!(store.publish("k1", b"first").unwrap());
        assert!(
            !store.publish("k1", b"second").unwrap(),
            "a complete entry is never overwritten"
        );
        assert_eq!(store.get("k1").1.as_deref(), Some(&b"first"[..]));
    }
}
