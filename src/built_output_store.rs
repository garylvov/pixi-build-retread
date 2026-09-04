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

/// Bumped whenever the stored RECORD's wire format changes. Folded into
/// every key AND written into every record, so an old entry is invisible
/// rather than misread — and, if it is read anyway, refused rather than
/// decoded.
///
/// v2 (fix p6c): the key gained the resolution-policy fingerprint (the
/// auto-imports injection gate). Every v1 entry may hold a POST-INJECTION
/// payload stored at an injection-OFF address, so all of them are invalidated
/// once.
///
/// v3 (fix C11): the payload stopped being a bare `CondaOutputsResult` and
/// became [`Record`] — an envelope carrying this schema, the emission-schema
/// constant and the FULL input digest the reader independently recomputes.
/// The key stopped folding the backend's GIT HASH (see
/// `handler::backend_behaviour_identity`), so entries now survive a backend
/// rebuild; the envelope is what keeps a surviving entry from being MISREAD
/// after that survival became possible.
pub const SCHEMA: &str = "retread-built-output-store-v3";

/// The EMISSION-SEMANTICS version of the payload: what the backend decided,
/// as opposed to how it is spelled on disk ([`SCHEMA`]).
///
/// This is the constant that replaces the git hash. Until C11 the key folded
/// `CARGO_PKG_VERSION + "+" + RETREAD_GIT_HASH`, so every rebuild made all
/// existing entries unreachable — measured as `miss=14 hit=0` on two
/// consecutive canonical relocks that differed only by a binary. Dropping the
/// hash is what makes the store useful across binaries; this constant is what
/// keeps that safe.
///
/// **Any commit that changes what `conda/outputs` EMITS for unchanged inputs
/// must bump this.** That includes: the auto-bundle cascade's selection or
/// ordering, route-probe acceptance, pin rendering or relaxation, injected
/// dependencies, and the `input_globs` set. It does NOT include changes that
/// cannot move the emitted bytes for fixed inputs — logging, timing,
/// diagnostics, error text, or a pure refactor.
///
/// Why a hand-bumped constant and not a hash of the emitting code: the code
/// that decides these outputs is spread across `handler/mod.rs`,
/// `uv_closure.rs`, `pypi.rs`, `pack_overrides.rs` and `source_build.rs`, so a
/// hash of ONE module is false comfort and a hash of ALL of `src/` is
/// precisely the git hash this change exists to remove. A curated file list is
/// the same human discipline as this constant with none of its visibility: a
/// file omitted from the list fails silently, while a missing bump here is a
/// named, greppable line in review. The residual risk is bounded on three
/// sides — `CARGO_PKG_VERSION` is also in the identity, so every release bump
/// invalidates the store regardless; the record carries the producing git hash
/// for post-hoc audit; and a stale hit is a stale RESOLUTION, never a wrong
/// artifact, because everything downstream re-validates bytes.
pub const BUILT_OUTPUT_SCHEMA: &str = "retread-built-output-emission-1";

/// Why a stored record was not usable. Every arm is a MISS at the call site;
/// the variant exists so the log line names which one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The bytes are not a [`Record`] at all — including every pre-v3 entry,
    /// which is a bare payload object with none of the envelope's fields.
    Undecodable,
    /// A record written by a different wire schema.
    Schema { found: String },
    /// A record emitted by different emission semantics.
    Emission { found: String },
    /// A record whose own input digest disagrees with the one the reader
    /// computed for the address it looked up. Cannot happen through an honest
    /// publish; it catches a truncated-key collision, a hand-moved entry, and
    /// a publisher that addressed and stamped a record from different inputs.
    Inputs { found: String },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Undecodable => write!(f, "not a built-output record (pre-v3 or corrupt)"),
            Refusal::Schema { found } => {
                write!(f, "record schema `{found}` != `{SCHEMA}`")
            }
            Refusal::Emission { found } => {
                write!(
                    f,
                    "record emission schema `{found}` != `{BUILT_OUTPUT_SCHEMA}`"
                )
            }
            Refusal::Inputs { found } => {
                write!(f, "record input digest `{found}` != the digest of the inputs this lookup was built from")
            }
        }
    }
}

/// The stored envelope.
///
/// The reader accepts a record only when all three stamped identities match
/// what it computed itself. That is the difference between an entry being
/// *unreachable* (the pre-C11 property, bought with the git hash) and an entry
/// being *unmisreadable* (the C11 property, bought with these fields) — and
/// only the second one survives the git hash being dropped from the key.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Record {
    pub schema: String,
    pub emission_schema: String,
    /// The FULL sha256 over the key material. The store address is only the
    /// first 16 bytes of it, so this is an independent check and not a
    /// restatement of the directory name.
    pub inputs_digest: String,
    /// AUDIT ONLY. Never consulted by [`decode`]; it exists so an operator can
    /// read which binary produced an adopted entry — the fact the git hash
    /// used to buy by making the entry unreachable.
    pub produced_by: String,
    pub payload: serde_json::Value,
    /// The `advertised_identity` records the COLD compute wrote as a side
    /// effect of producing `payload`.
    ///
    /// A store hit returns before that loop runs, so without these an adopted
    /// output leaves no record and `conda/build_v1` re-derives the build string
    /// from the ADOPTING workspace's live inputs — job 5723770 (`p19-depadd`,
    /// `hit=14 miss=0`) died exactly there, on `courier inputs changed between
    /// conda/outputs and conda/build_v1`. Carrying them in the record is what
    /// makes an adoption leave the same on-disk state a cold compute leaves.
    /// `serde(default)` so the field is additive within this schema.
    #[serde(default)]
    pub advertised: serde_json::Value,
}

/// A record this reader accepted: the payload plus the cold pass's side
/// effects, which the caller must restore before it can behave as if it had
/// computed the payload itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accepted {
    pub payload: serde_json::Value,
    pub advertised: serde_json::Value,
}

/// Wrap a payload for publication.
pub fn encode<T: serde::Serialize, A: serde::Serialize>(
    inputs_digest: &str,
    produced_by: &str,
    payload: &T,
    advertised: &A,
) -> Result<Vec<u8>, serde_json::Error> {
    let record = Record {
        schema: SCHEMA.to_string(),
        emission_schema: BUILT_OUTPUT_SCHEMA.to_string(),
        inputs_digest: inputs_digest.to_string(),
        produced_by: produced_by.to_string(),
        payload: serde_json::to_value(payload)?,
        advertised: serde_json::to_value(advertised)?,
    };
    serde_json::to_vec(&record)
}

/// Unwrap a stored record, refusing anything whose stamped identity does not
/// match this reader. A refusal never yields the payload.
pub fn decode(bytes: &[u8], expected_inputs_digest: &str) -> Result<Accepted, Refusal> {
    let record: Record = serde_json::from_slice(bytes).map_err(|_| Refusal::Undecodable)?;
    if record.schema != SCHEMA {
        return Err(Refusal::Schema {
            found: record.schema,
        });
    }
    if record.emission_schema != BUILT_OUTPUT_SCHEMA {
        return Err(Refusal::Emission {
            found: record.emission_schema,
        });
    }
    if record.inputs_digest != expected_inputs_digest {
        return Err(Refusal::Inputs {
            found: record.inputs_digest,
        });
    }
    Ok(Accepted {
        payload: record.payload,
        advertised: record.advertised,
    })
}

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

    // ---------------------------------------------------------------
    // C11 guard (d): a record written by an older schema is REFUSED, not
    // misread. Every arm here is red on the pre-C11 reader, which was a bare
    // `serde_json::from_slice::<CondaOutputsResult>` with no envelope at all.
    // ---------------------------------------------------------------

    /// The shape of the pre-v3 payload: exactly what `serde_json::to_vec`
    /// wrote before C11, i.e. the bare result object.
    fn legacy_payload() -> Vec<u8> {
        use pixi_build_types::procedures::conda_outputs::CondaOutputsResult;
        serde_json::to_vec(&CondaOutputsResult {
            outputs: Default::default(),
            input_globs: Default::default(),
        })
        .unwrap()
    }

    #[test]
    fn a_pre_v3_bare_payload_is_refused_not_decoded() {
        use pixi_build_types::procedures::conda_outputs::CondaOutputsResult;
        let bytes = legacy_payload();

        // NON-VACUITY: these bytes really are a payload the OLD reader would
        // have adopted -- they deserialize through the exact call the pre-C11
        // hit path made. If this stops holding, the guard below is testing
        // that garbage is rejected, which proves nothing.
        let _as_old_reader: CondaOutputsResult = serde_json::from_slice(&bytes)
            .expect("the legacy fixture must deserialize the way the pre-C11 reader did");

        assert_eq!(
            decode(&bytes, "any-digest"),
            Err(Refusal::Undecodable),
            "a pre-v3 entry must be refused"
        );
    }

    #[test]
    fn a_record_from_another_schema_or_emission_or_input_set_is_refused() {
        let good = encode(
            "digest-a",
            "1.2.3+deadbeef",
            &serde_json::json!({"outputs": []}),
            &serde_json::json!([{"name": "pack", "build": "py311_hdeadbeef_loose_5"}]),
        )
        .unwrap();

        // Positive control first: the honest round trip must work, or every
        // refusal below is trivially satisfiable.
        let accepted = decode(&good, "digest-a").unwrap();
        assert_eq!(
            accepted.payload,
            serde_json::json!({"outputs": []}),
            "a record this reader wrote must decode to exactly its payload"
        );
        // The cold pass's side effects travel with the payload, or an adoption
        // is not equivalent to the compute it stands in for (job 5723770).
        assert_eq!(
            accepted.advertised,
            serde_json::json!([{"name": "pack", "build": "py311_hdeadbeef_loose_5"}]),
            "the advertised-identity records must survive the round trip"
        );

        let tamper = |field: &str, value: &str| {
            let mut record: serde_json::Value = serde_json::from_slice(&good).unwrap();
            record[field] = serde_json::Value::String(value.to_string());
            serde_json::to_vec(&record).unwrap()
        };

        // (d) an older WIRE schema.
        assert_eq!(
            decode(&tamper("schema", "retread-built-output-store-v2"), "digest-a"),
            Err(Refusal::Schema {
                found: "retread-built-output-store-v2".to_string()
            }),
        );
        // (c) an older EMISSION schema -- the hand-bumped constant. A binary
        // that bumped it must not adopt what the previous one emitted.
        assert_eq!(
            decode(
                &tamper("emission_schema", "retread-built-output-emission-0"),
                "digest-a"
            ),
            Err(Refusal::Emission {
                found: "retread-built-output-emission-0".to_string()
            }),
        );
        // (b) a record stamped with a different input digest, i.e. an entry
        // that landed at this address without standing for these inputs --
        // the truncated-key collision the git hash never covered.
        assert_eq!(
            decode(&good, "digest-b"),
            Err(Refusal::Inputs {
                found: "digest-a".to_string()
            }),
        );

        // And no refusal ever yields the payload.
        for bytes in [
            tamper("schema", "retread-built-output-store-v2"),
            tamper("emission_schema", "retread-built-output-emission-0"),
        ] {
            assert!(decode(&bytes, "digest-a").is_err());
        }
    }

    #[test]
    fn the_producing_binary_is_recorded_but_never_gates_acceptance() {
        // The git hash left the KEY; it must still be readable off an entry,
        // and it must not be able to refuse one -- that was the whole trade.
        let bytes = encode(
            "digest-a",
            "9.9.9+cafebabe",
            &serde_json::json!({"outputs": []}),
            &serde_json::json!([]),
        )
        .unwrap();
        let record: Record = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(record.produced_by, "9.9.9+cafebabe");
        assert!(
            decode(&bytes, "digest-a").is_ok(),
            "a record from another binary must still be adoptable"
        );
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
