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

/// The store's directory name under a persistent root, and the `--store`
/// spelling `retread store-reap` accepts. Named here, beside the layout it
/// describes, so the reaper's spec READS it instead of carrying a second copy
/// (the two-copies-of-a-generation-string defect STORE-REAP-3 measured).
pub const STORE_DIR: &str = "built-outputs";

/// CONDA-OUT-2. How long an entry may go unreferenced before
/// `retread store-reap --store built-outputs` selects it. The same 14 days
/// every other persistent store in this backend uses; a store with no horizon
/// is a leak, and a horizon spelled differently here would be a second policy.
pub const BUILT_OUTPUT_STORE_DEFAULT_MAX_AGE_DAYS: u64 = 14;

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
    /// CONDA-OUT-2. The record states which repodata documents the resolution
    /// it holds actually consulted, and at least one of them is no longer
    /// present, byte-identical, under this reader's cache root -- so the world
    /// that answer was true in is not this reader's world. `recorded` is how
    /// many documents the record named; `missing` is the first one that is not
    /// here, spelled `<channel>/<subdir>@<sha256>`.
    ///
    /// An EMPTY recorded set is this refusal too, and deliberately: a record
    /// written before this field existed states NOTHING about the world it was
    /// resolved in, and nothing can never be shown to still hold. Same shape,
    /// same `serde(default)` and the same reasoning as
    /// `handler::advertised_identity::AdvertisedIdentityRecord::repodata_universe`.
    Universe { recorded: usize, missing: String },
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
            Refusal::Universe { recorded, missing } => {
                write!(
                    f,
                    "repodata_universe mismatch: of the {recorded} document(s) the stored resolution consulted, `{missing}` is not present under this reader's cache root"
                )
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
    /// CONDA-OUT-2, AUDIT ONLY. `repodata::universe_digest_of` over exactly
    /// the set in [`consulted_repodata`](Record::consulted_repodata), so an
    /// operator can compare an entry against a `repodata-universe` row with
    /// one grep instead of a set diff. It is NOT the acceptance test, and a
    /// reader never compares it: the set is, for the reasons below.
    #[serde(default)]
    pub repodata_universe: String,
    /// CONDA-OUT-2. **The world this answer was true in**: every repodata
    /// document the producing resolution actually consulted, as
    /// `repodata::universe_documents()` recorded them.
    ///
    /// THE DEFECT THIS CLOSES (law 2). Until this field the store recorded a
    /// universe nowhere and read one nowhere: the only `repodata_universe` in a
    /// stored record sat NESTED inside `advertised`, where nothing at this
    /// layer looked at it. `restore_advertised_identities` then wrote those
    /// records into the adopting job's cache dir, and
    /// `advertised_identity::load_record` refused them for the universe
    /// mismatch several RPCs later -- so a stale adoption surfaced as "there
    /// was no record" inside `conda/build_v1`, which is job 5723770's shape.
    /// The refusal existed; it just arrived after the adoption had already been
    /// taken and in a place that misnames it.
    ///
    /// WHY THIS AND NOT THE DIGEST, MEASURED ON THIS BOX. Every one of the 15
    /// tip-produced entries in the shared store was written inside one
    /// 44-minute window carrying `f78473b8878daa23`, and the SAME shared root
    /// folded to `654cab14d9e1d23b` five hours later. `universe_digest`'s own
    /// doc boards the reason as p6ad-6-1: the whole-root digest moves when an
    /// unrelated lane drops an unrelated document in. Folding that digest into
    /// the key, or demanding it be equal on read, re-addresses or refuses the
    /// whole store several times a day for changes that cannot touch this
    /// pack's resolution -- which is C11's "unreachable rather than
    /// unmisreadable" defect reintroduced with a faster clock.
    ///
    /// WHY CONTAINMENT AND NOT EQUALITY. The universe that can move the
    /// payload is the set the resolution CONSULTED, and that set is known only
    /// AFTER the resolution -- the reader has not resolved anything when it
    /// looks up, so it cannot compute an equal digest to compare against. What
    /// it can decide, before resolving, is whether the world the stored answer
    /// was true in is still intact inside its own: every recorded
    /// `(channel, subdir, sha256, bytes)` still present exactly. That refuses
    /// every genuine channel move and is immune to the unrelated-document
    /// false positive.
    ///
    /// `serde(default)` so the field is additive within this schema, exactly as
    /// `advertised` was. A record without it decodes to an EMPTY set, and an
    /// empty set is a refusal: a record that states nothing about its world
    /// cannot be shown to still hold in this one.
    #[serde(default)]
    pub consulted_repodata: Vec<crate::repodata::RepodataDocument>,
}

/// A record this reader accepted: the payload plus the cold pass's side
/// effects, which the caller must restore before it can behave as if it had
/// computed the payload itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accepted {
    pub payload: serde_json::Value,
    pub advertised: serde_json::Value,
}

/// How one consulted document is spelled in a refusal row: enough to identify
/// it and to say WHICH of its facts moved, and nothing that is a path.
///
/// N27-RETREAD-62 added the URL-derived channel key, because it is now half of
/// the comparison and a record written before the fix carries none -- a
/// refusal that printed only the URL could not be told apart from a refusal
/// for moved content.
fn document_label(document: &crate::repodata::RepodataDocument) -> String {
    let key = if document.channel_key.is_empty() {
        "<no channel key: a record older than N27-RETREAD-62>"
    } else {
        document.channel_key.as_str()
    };
    format!(
        "{}/{}#{}@{}",
        document.channel, document.subdir, key, document.sha256
    )
}

/// Wrap a payload for publication.
///
/// `consulted` is [`Record::consulted_repodata`]: the documents this
/// resolution read. It is the writer's half of the adoption rule and there is
/// no publish path that omits it -- a record with an empty set is refused by
/// every reader, so a caller that could not name its documents publishes an
/// entry nobody will ever adopt rather than one anybody might adopt blind.
pub fn encode<T: serde::Serialize, A: serde::Serialize>(
    inputs_digest: &str,
    produced_by: &str,
    repodata_universe: &str,
    consulted: &[crate::repodata::RepodataDocument],
    payload: &T,
    advertised: &A,
) -> Result<Vec<u8>, serde_json::Error> {
    let mut consulted_repodata = consulted.to_vec();
    // SORTED AND DEDUPED at the writer, the same normalisation
    // `repodata::universe_digest_of` applies, so two publishers of one key
    // cannot write two orderings of one world.
    consulted_repodata.sort();
    consulted_repodata.dedup();
    let record = Record {
        schema: SCHEMA.to_string(),
        emission_schema: BUILT_OUTPUT_SCHEMA.to_string(),
        inputs_digest: inputs_digest.to_string(),
        produced_by: produced_by.to_string(),
        payload: serde_json::to_value(payload)?,
        advertised: serde_json::to_value(advertised)?,
        repodata_universe: repodata_universe.to_string(),
        consulted_repodata,
    };
    serde_json::to_vec(&record)
}

/// Unwrap a stored record, refusing anything whose stamped identity does not
/// match this reader. A refusal never yields the payload.
/// CONDA-OUT-2: `reader_documents` is the reader's own on-disk snapshot
/// (`repodata::snapshot_documents_at`). An adoption requires that every
/// document the stored resolution consulted is still in it, byte-identical.
///
/// N27-RETREAD-62: "still in it" is decided on
/// [`RepodataDocument::adoption_identity`](crate::repodata::RepodataDocument::adoption_identity)
/// -- the URL-derived channel key, the subdir and the content hash -- and
/// NEVER on the whole document, because `channel` is a human label whose two
/// producers spell it differently by construction. Comparing it made every
/// stored record unadoptable by every reader, including the one that wrote it.
pub fn decode(
    bytes: &[u8],
    expected_inputs_digest: &str,
    reader_documents: &[crate::repodata::RepodataDocument],
) -> Result<Accepted, Refusal> {
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
    // The universe check is LAST of the four on purpose: it is the only one
    // that costs a set membership over the reader's snapshot, and the three
    // cheap identity checks have already thrown out every record that is not
    // even about these inputs.
    if record.consulted_repodata.is_empty() {
        return Err(Refusal::Universe {
            recorded: 0,
            missing: "<the record names no consulted document>".to_string(),
        });
    }
    // N27-RETREAD-62. THE COMPARISON IS `adoption_identity`, NOT THE WHOLE
    // DOCUMENT. Comparing whole documents compared `channel` too, and
    // `channel` has two producers that spell it differently by construction:
    // the writer holds the URL `sparse()` was called with, the reader holds
    // only a filename and rendered `conda_forge#<hex>` from it. No record
    // could ever be adopted -- relock 6063566 read `hit=0 miss=14
    // published=14` with every refusal naming a document the SAME process's
    // `conda_universe` row listed as consulted eleven seconds later. The rule
    // CONDA-OUT-2 wrote is unchanged: containment of the consulted set in the
    // reader's world, decided on CONTENT and on a URL-derived key, never on a
    // path.
    if let Some(missing) = record.consulted_repodata.iter().find(|document| {
        document.channel_key.is_empty()
            || !reader_documents
                .iter()
                .any(|present| present.adoption_identity() == document.adoption_identity())
    }) {
        return Err(Refusal::Universe {
            recorded: record.consulted_repodata.len(),
            missing: document_label(missing),
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
///
/// Its CONTENT is [`SCHEMA`], and CONDA-OUT-2 made that load-bearing: this
/// store has no generation directory, so the marker's own bytes are where an
/// entry's generation is written and are what
/// `retread store-reap --store built-outputs` reads to decide `stale-version`.
/// Public for the reaper's spec, which must never carry a second copy of it.
pub const MARKER: &str = "COMPLETE";

/// Where the reaper moves a selected entry. Named here, beside the layout,
/// because [`BuiltOutputStore::get`] must never walk into it and the reaper
/// must never walk it as an entry.
pub const QUARANTINE: &str = "quarantine";

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

    /// CONDA-OUT-2. Record that a lock REFERENCED this entry, by touching the
    /// `.used` sidecar the reaper ages from.
    ///
    /// Without this the reaper ages every entry from its PUBLISH time, so the
    /// entry a relock adopts every night is evicted on its fourteenth day for
    /// being unreferenced -- which is exactly false. It is the same sidecar,
    /// written by the same formula (`source_build::use_stamp_path`), that the
    /// canonical-git-snapshot and hermetic-environment stores stamp on their
    /// own hit arms; a second formula here is how two stores' sidecars
    /// silently diverge.
    ///
    /// Best effort and silent on failure: a read-only or full store must cost
    /// a stamp, never a lock.
    pub fn stamp_used(&self, key: &str) {
        if let Some(stamp) = crate::source_build::use_stamp_path(&self.entry(key)) {
            let _ = std::fs::write(stamp, b"");
        }
    }

    /// CONDA-OUT-2. Move a REFUSED entry aside so the cold compute that
    /// replaces it can publish at the same address.
    ///
    /// THE HOLE THIS CLOSES. [`Self::publish`] returns `Ok(false)` over any
    /// entry that carries the marker, so before this an entry this binary
    /// refuses -- a stale universe, a stale emission schema -- occupied its
    /// address FOREVER: every later run refused it and no later run could
    /// replace it. A refusal without a repair is not a safety property, it is
    /// a permanently cold address, and on a shared store it is permanent for
    /// everyone.
    ///
    /// The entry is RENAMED into [`QUARANTINE`], never deleted: another
    /// process, on another binary or another snapshot, may still be able to
    /// adopt what this one refused, and a rename leaves that recoverable while
    /// a delete does not. `retread store-reap` is what empties the quarantine.
    ///
    /// Returns whether an entry was moved. Best effort: a failed rename simply
    /// leaves the refusal standing, which is today's behaviour.
    pub fn quarantine_refused(&self, key: &str) -> bool {
        let entry = self.entry(key);
        if !entry.is_dir() {
            return false;
        }
        let quarantine_root = self.root.join(QUARANTINE);
        if std::fs::create_dir_all(&quarantine_root).is_err() {
            return false;
        }
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default();
        let target = quarantine_root.join(format!("{key}-{stamp}-{}", std::process::id()));
        if std::fs::rename(&entry, &target).is_err() {
            return false;
        }
        if let Some(sidecar) = crate::source_build::use_stamp_path(&entry) {
            let _ = std::fs::remove_file(sidecar);
        }
        true
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

    /// CONDA-OUT-2 fixtures. One repodata document, named the way the registry
    /// names them.
    fn document(channel: &str, sha256: &str) -> crate::repodata::RepodataDocument {
        crate::repodata::RepodataDocument {
            channel: channel.to_string(),
            subdir: "linux-64".to_string(),
            sha256: sha256.to_string(),
            bytes: 4096,
            channel_key: crate::repodata::channel_subdir_key(channel, "linux-64"),
        }
    }

    /// The world a fixture resolution consulted.
    fn consulted_world() -> Vec<crate::repodata::RepodataDocument> {
        vec![
            document("https://conda.anaconda.org/conda-forge", "aa11"),
            document("https://conda.anaconda.org/nvidia", "bb22"),
        ]
    }

    fn encoded(
        inputs_digest: &str,
        consulted: &[crate::repodata::RepodataDocument],
    ) -> Vec<u8> {
        encode(
            inputs_digest,
            "1.2.3+deadbeef",
            &crate::repodata::universe_digest_of(consulted),
            consulted,
            &serde_json::json!({"outputs": []}),
            &serde_json::json!([{"name": "pack", "build": "py311_hdeadbeef_loose_5"}]),
        )
        .unwrap()
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
            decode(&bytes, "any-digest", &consulted_world()),
            Err(Refusal::Undecodable),
            "a pre-v3 entry must be refused"
        );
    }

    #[test]
    fn a_record_from_another_schema_or_emission_or_input_set_is_refused() {
        let world = consulted_world();
        let good = encoded("digest-a", &world);

        // Positive control first: the honest round trip must work, or every
        // refusal below is trivially satisfiable.
        let accepted = decode(&good, "digest-a", &world).unwrap();
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
            decode(&tamper("schema", "retread-built-output-store-v2"), "digest-a", &world),
            Err(Refusal::Schema {
                found: "retread-built-output-store-v2".to_string()
            }),
        );
        // (c) an older EMISSION schema -- the hand-bumped constant. A binary
        // that bumped it must not adopt what the previous one emitted.
        assert_eq!(
            decode(
                &tamper("emission_schema", "retread-built-output-emission-0"),
                "digest-a",
                &world
            ),
            Err(Refusal::Emission {
                found: "retread-built-output-emission-0".to_string()
            }),
        );
        // (b) a record stamped with a different input digest, i.e. an entry
        // that landed at this address without standing for these inputs --
        // the truncated-key collision the git hash never covered.
        assert_eq!(
            decode(&good, "digest-b", &world),
            Err(Refusal::Inputs {
                found: "digest-a".to_string()
            }),
        );

        // And no refusal ever yields the payload.
        for bytes in [
            tamper("schema", "retread-built-output-store-v2"),
            tamper("emission_schema", "retread-built-output-emission-0"),
        ] {
            assert!(decode(&bytes, "digest-a", &world).is_err());
        }
    }

    #[test]
    fn the_producing_binary_is_recorded_but_never_gates_acceptance() {
        // The git hash left the KEY; it must still be readable off an entry,
        // and it must not be able to refuse one -- that was the whole trade.
        let world = consulted_world();
        let bytes = encode(
            "digest-a",
            "9.9.9+cafebabe",
            &crate::repodata::universe_digest_of(&world),
            &world,
            &serde_json::json!({"outputs": []}),
            &serde_json::json!([]),
        )
        .unwrap();
        let record: Record = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(record.produced_by, "9.9.9+cafebabe");
        assert!(
            decode(&bytes, "digest-a", &world).is_ok(),
            "a record from another binary must still be adoptable"
        );
    }

    // ---------------------------------------------------------------
    // CONDA-OUT-2. The repodata-universe rule, and the repair that makes a
    // refusal survivable. Every arm below is RED on c0ccc0d, whose `decode`
    // took no world at all and whose `publish` could never replace a marked
    // entry.
    // ---------------------------------------------------------------

    #[test]
    fn a_record_whose_consulted_documents_are_all_still_here_is_adopted() {
        // The POSITIVE CONTROL for the whole rule, and the one that keeps the
        // three refusals below from being satisfiable by refusing everything:
        // the reader's world may be STRICTLY LARGER than the record's -- an
        // unrelated lane dropping an unrelated document in must NOT refuse
        // anything, which is the false positive p6ad-6-1 boards against the
        // whole-root digest and the entire reason this is containment.
        let world = consulted_world();
        let mut reader = world.clone();
        reader.push(document("https://conda.anaconda.org/some-other-lane", "cc33"));
        let bytes = encoded("digest-a", &world);
        let accepted = decode(&bytes, "digest-a", &reader)
            .expect("a superset world must still adopt");
        assert_eq!(accepted.payload, serde_json::json!({"outputs": []}));
    }

    #[test]
    fn a_record_is_refused_when_a_document_it_consulted_moved_or_left() {
        let world = consulted_world();
        let bytes = encoded("digest-a", &world);

        // (a) the document is GONE from the reader's root.
        let mut short = world.clone();
        short.pop();
        match decode(&bytes, "digest-a", &short) {
            Err(Refusal::Universe { recorded, missing }) => {
                assert_eq!(recorded, 2);
                assert_eq!(
                    missing,
                    format!(
                        "https://conda.anaconda.org/nvidia/linux-64#{}@bb22",
                        crate::repodata::channel_subdir_key(
                            "https://conda.anaconda.org/nvidia",
                            "linux-64"
                        )
                    )
                );
            }
            other => panic!("an absent consulted document must refuse: {other:?}"),
        }

        // (b) the document is STILL THERE under the same name and its BYTES
        // MOVED. This is the real hazard -- a channel refresh in place -- and
        // it is invisible to any check that compares names.
        let moved: Vec<_> = world
            .iter()
            .map(|document| {
                let mut document = document.clone();
                if document.channel.ends_with("conda-forge") {
                    document.sha256 = "ffff".to_string();
                }
                document
            })
            .collect();
        assert!(
            matches!(
                decode(&bytes, "digest-a", &moved),
                Err(Refusal::Universe { .. })
            ),
            "a refreshed document must refuse the answer resolved against the old one"
        );

        // (c) the reader could not list its root at all: EMPTY reader world.
        // The failure direction is a miss, never an adoption on no evidence.
        assert!(matches!(
            decode(&bytes, "digest-a", &[]),
            Err(Refusal::Universe { .. })
        ));

        // And no refusal ever yields the payload.
        assert!(decode(&bytes, "digest-a", &short).is_err());
    }

    #[test]
    fn a_record_that_names_no_consulted_document_is_refused() {
        // Every one of the 15 tip-produced entries in the shared store on
        // 2026-09-07 is this record: valid schema, valid emission, valid input
        // digest, and NOTHING said about the world it was resolved in. A
        // reader that adopts it is adopting on an unstated claim.
        let world = consulted_world();
        let mut record: serde_json::Value =
            serde_json::from_slice(&encoded("digest-a", &world)).unwrap();
        record
            .as_object_mut()
            .unwrap()
            .remove("consulted_repodata");
        let bytes = serde_json::to_vec(&record).unwrap();

        // NON-VACUITY: the bytes really are a record this reader would
        // otherwise take -- only the world claim is gone.
        let decoded: Record = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded.schema, SCHEMA);
        assert_eq!(decoded.inputs_digest, "digest-a");

        assert_eq!(
            decode(&bytes, "digest-a", &world),
            Err(Refusal::Universe {
                recorded: 0,
                missing: "<the record names no consulted document>".to_string(),
            }),
        );
    }

    #[test]
    fn the_consulted_set_is_normalised_by_the_writer() {
        // Two publishers of one key must not write two orderings of one world,
        // or byte-comparing two entries for the same key becomes meaningless.
        let world = consulted_world();
        let mut reversed = world.clone();
        reversed.reverse();
        reversed.push(reversed[0].clone());
        assert_eq!(
            encoded("digest-a", &world),
            encoded("digest-a", &reversed),
            "the writer must sort and dedup the consulted set"
        );
    }

    // ---------------------------------------------------------------
    // N27-RETREAD-62. THE TWO HALVES OF THE ADOPTION RULE MUST NAME ONE
    // DOCUMENT THE SAME WAY.
    //
    // Every CONDA-OUT-2 guard above builds BOTH halves of the comparison with
    // the same fixture helper, so all of them pass while production cannot
    // adopt anything: in production the reader's half comes from
    // `repodata::universe_from_cache_root` (which has only a filename and
    // renders `channel` as `<slug>#<hex>`) and the writer's from
    // `repodata::record_document` (which has the URL `sparse()` was called
    // with). MEASURED on relock 6063566: hit=0 miss=14 published=14.
    //
    // The guards below take the reader's half from the REAL reader producer,
    // so a fixture can never be the only thing that compares.
    // ---------------------------------------------------------------

    /// A document as the shared repodata cache stores it: the filename scheme
    /// `repodata::disk_cache_path` writes, which is the only thing the reader
    /// half ever sees.
    fn plant(root: &Path, channel_url: &str, subdir: &str, body: &[u8]) {
        let dir = root.join("retread-repodata");
        std::fs::create_dir_all(&dir).expect("repodata dir");
        let slug = channel_url
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or("channel")
            .replace(|c: char| !c.is_ascii_alphanumeric(), "_");
        let key = crate::repodata::channel_subdir_key(channel_url, subdir);
        std::fs::write(dir.join(format!("{slug}--{subdir}--{key}.json")), body)
            .expect("planting a repodata document");
    }

    /// The writer's half, built EXACTLY as `repodata::record_document` builds
    /// it: the channel URL it called `sparse()` with, the subdir, the content
    /// identity of the very bytes it parsed, and the URL-derived key. The sha
    /// and byte count are taken from the reader's own fold of the same file,
    /// because that is the same fingerprint function both halves run.
    fn as_the_writer_named_it(
        reader: &[crate::repodata::RepodataDocument],
        channel_url: &str,
        subdir: &str,
    ) -> crate::repodata::RepodataDocument {
        let key = crate::repodata::channel_subdir_key(channel_url, subdir);
        let seen = reader
            .iter()
            .find(|document| document.channel_key == key && document.subdir == subdir)
            .expect("the planted document must be in the reader's fold");
        crate::repodata::RepodataDocument {
            channel: channel_url.to_string(),
            subdir: subdir.to_string(),
            sha256: seen.sha256.clone(),
            bytes: seen.bytes,
            channel_key: key,
        }
    }

    const N27_62_CHANNEL: &str = "https://prefix.dev/conda-forge";
    const N27_62_BODY: &[u8] = br#"{"packages":{"n27-62":{"build":"0"}}}"#;

    /// N27-RETREAD-62 GUARD 1 — TWO JOBS, TWO CACHE ROOTS, ONE CONTENT: THE
    /// SECOND ADOPTS.
    ///
    /// RED ON 5ebfbb6: `decode` compared whole `RepodataDocument`s, and the
    /// reader's `channel` (`conda_forge#<hex>`, rendered from the filename)
    /// can never equal the writer's (`https://prefix.dev/conda-forge`), so
    /// this decode returned `Refusal::Universe` naming a document that is
    /// demonstrably right there -- which is the shape of every one of the 14
    /// refusals in relock 6063566's backend log.
    ///
    /// MUTATION ARM: restore path identity to the comparison -- swap
    /// `document.adoption_identity()` back to whole-document containment in
    /// `decode`, or make `universe_from_cache_root_inner` stop recovering
    /// `channel_key` from the filename's third field -- and this goes red.
    #[test]
    fn n27_62_a_record_published_under_one_cache_root_is_adopted_under_another() {
        let publisher = Scratch::new("n27-62-publisher");
        let requester = Scratch::new("n27-62-requester");
        plant(publisher.path(), N27_62_CHANNEL, "linux-64", N27_62_BODY);
        plant(requester.path(), N27_62_CHANNEL, "linux-64", N27_62_BODY);

        let published_world =
            crate::repodata::universe_from_cache_root(publisher.path()).unwrap();
        let consulted =
            vec![as_the_writer_named_it(&published_world, N27_62_CHANNEL, "linux-64")];
        let bytes = encoded("digest-n27-62", &consulted);

        // The requester's world comes from the REAL reader producer over a
        // DIFFERENT directory. Identical content, different path.
        let reader = crate::repodata::universe_from_cache_root(requester.path()).unwrap();
        assert_eq!(reader.len(), 1, "one planted document");

        // NON-VACUITY, and the whole defect in one line: the two halves are
        // NOT equal as whole documents. If they were, this guard would pass on
        // the broken code too.
        assert_ne!(
            reader[0], consulted[0],
            "the reader's label and the writer's URL must still differ -- \
             otherwise this guard is not about the thing that was wrong"
        );
        assert_eq!(
            reader[0].adoption_identity(),
            consulted[0].adoption_identity(),
            "and their ADOPTION identity must be one thing"
        );

        let accepted = decode(&bytes, "digest-n27-62", &reader)
            .expect("identical content under another cache root must adopt");
        assert_eq!(accepted.payload, serde_json::json!({"outputs": []}));
    }

    /// N27-RETREAD-62 GUARD 2 — DIFFERENT CONTENT STILL REFUSES.
    ///
    /// The rule CONDA-OUT-2 wrote is not weakened by naming the channel
    /// differently: a channel that refreshed in place under the SAME name and
    /// the SAME URL-derived key must still refuse, or guard 1 would have been
    /// bought by comparing nothing.
    #[test]
    fn n27_62_a_requester_whose_document_content_moved_still_refuses() {
        let publisher = Scratch::new("n27-62-moved-pub");
        let requester = Scratch::new("n27-62-moved-req");
        plant(publisher.path(), N27_62_CHANNEL, "linux-64", N27_62_BODY);
        plant(
            requester.path(),
            N27_62_CHANNEL,
            "linux-64",
            br#"{"packages":{"n27-62":{"build":"1"}}}"#,
        );

        let published_world =
            crate::repodata::universe_from_cache_root(publisher.path()).unwrap();
        let consulted =
            vec![as_the_writer_named_it(&published_world, N27_62_CHANNEL, "linux-64")];
        let bytes = encoded("digest-n27-62", &consulted);
        let reader = crate::repodata::universe_from_cache_root(requester.path()).unwrap();

        // Same channel, same subdir, same key -- ONLY the bytes moved.
        assert_eq!(reader[0].channel_key, consulted[0].channel_key);
        assert_ne!(reader[0].sha256, consulted[0].sha256);
        match decode(&bytes, "digest-n27-62", &reader) {
            Err(Refusal::Universe { recorded, missing }) => {
                assert_eq!(recorded, 1);
                assert!(
                    missing.contains(&consulted[0].sha256),
                    "the refusal must name the content that moved: {missing}"
                );
            }
            other => panic!("a refreshed document must refuse: {other:?}"),
        }
    }

    /// N27-RETREAD-62 GUARD 3 — A RECORD WRITTEN BEFORE THIS FIX IS REFUSED,
    /// NOT ADOPTED BY ACCIDENT.
    ///
    /// `channel_key` is `serde(default)`, so the 14 records the CO and B35
    /// rounds published decode with an EMPTY key. An empty key must never
    /// match a reader document (whose key is never empty), and the refusal
    /// must SAY so -- otherwise the first operator to see it reads it as a
    /// channel that moved and goes looking for a refresh that never happened.
    #[test]
    fn n27_62_a_record_without_a_channel_key_is_refused_and_says_why() {
        let requester = Scratch::new("n27-62-legacy");
        plant(requester.path(), N27_62_CHANNEL, "linux-64", N27_62_BODY);
        let reader = crate::repodata::universe_from_cache_root(requester.path()).unwrap();
        let mut legacy = as_the_writer_named_it(&reader, N27_62_CHANNEL, "linux-64");
        legacy.channel_key = String::new();
        let bytes = encoded("digest-n27-62", &[legacy]);

        match decode(&bytes, "digest-n27-62", &reader) {
            Err(Refusal::Universe { recorded, missing }) => {
                assert_eq!(recorded, 1);
                assert!(
                    missing.contains("older than N27-RETREAD-62"),
                    "the refusal must name the reason, not a phantom move: {missing}"
                );
            }
            other => panic!("a keyless record must refuse: {other:?}"),
        }
    }

    #[test]
    fn a_refused_entry_can_be_replaced_and_only_through_the_quarantine() {
        // THE HOLE. `publish` refuses to overwrite a marked entry, so without
        // `quarantine_refused` the FIRST record this binary refuses owns its
        // address for ever and that key is permanently cold for every job
        // sharing the store. Mutation arm for this guard: delete the
        // `quarantine_refused` call in `handler::conda_outputs` and the second
        // publish below goes back to `false` with the stale bytes still served.
        let dir = Scratch::new("repair");
        let store = store(dir.path());
        assert!(store.publish("k1", b"stale").unwrap());
        assert!(
            !store.publish("k1", b"fresh").unwrap(),
            "the pre-existing no-overwrite rule must still hold"
        );

        assert!(store.quarantine_refused("k1"), "the entry must move aside");
        assert_eq!(
            store.get("k1").0,
            Lookup::Miss,
            "a quarantined entry is a miss, not a hit and not incomplete"
        );
        assert!(store.publish("k1", b"fresh").unwrap());
        assert_eq!(store.get("k1").1.as_deref(), Some(&b"fresh"[..]));

        // RECOVERABLE, never deleted: another binary may still be able to
        // adopt what this one refused.
        let quarantined: Vec<String> = std::fs::read_dir(dir.path().join(QUARANTINE))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(quarantined.len(), 1, "{quarantined:?}");
        assert!(quarantined[0].starts_with("k1-"), "{quarantined:?}");
        assert_eq!(
            std::fs::read(dir.path().join(QUARANTINE).join(&quarantined[0]).join(PAYLOAD))
                .unwrap(),
            b"stale".to_vec(),
            "the refused bytes must survive the move"
        );

        // Nothing to move is not an error.
        assert!(!store.quarantine_refused("never-published"));
    }

    #[test]
    fn an_adoption_stamps_the_sidecar_the_reaper_ages_from() {
        // Without this the reaper ages an entry from its PUBLISH, so the entry
        // a nightly relock adopts every night is evicted on its fourteenth day
        // for being "unreferenced" -- which is exactly false. Same sidecar and
        // same formula as the git-snapshot and hermetic-environment stores.
        let dir = Scratch::new("stamp");
        let store = store(dir.path());
        assert!(store.publish("k1", b"payload").unwrap());
        let stamp = crate::source_build::use_stamp_path(&dir.path().join("k1"))
            .expect("the shared formula must name a stamp for this entry");
        assert!(!stamp.exists(), "nothing stamps before an adoption");
        assert_eq!(store.get("k1").0, Lookup::Hit);
        store.stamp_used("k1");
        assert!(
            stamp.is_file(),
            "an adoption must leave the `.used` sidecar the reaper reads"
        );
        // A DOTFILE, so the reaper's `read_dir_names` can never walk it as an
        // entry of the flat store.
        assert!(
            stamp.file_name().unwrap().to_string_lossy().starts_with('.'),
            "the stamp must be a dotfile beside the entry"
        );
    }

    #[test]
    fn the_marker_carries_the_generation_the_reaper_reads() {
        // The flat store has no generation directory, so `store-reap
        // --store built-outputs` reads each entry's generation out of its
        // marker. That is only true while the marker's CONTENT is `SCHEMA`.
        let dir = Scratch::new("generation");
        let store = store(dir.path());
        assert!(store.publish("k1", b"payload").unwrap());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("k1").join(MARKER)).unwrap(),
            SCHEMA,
            "the reaper's `stale-version` rule reads this byte-for-byte"
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
