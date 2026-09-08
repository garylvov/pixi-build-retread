//! SDIST-META-2: THE PERSISTENT PREPARED-SDIST-METADATA STORE, and the ONE
//! authority for its entry key.
//!
//! SDIST-META-3 REPLACED THE KEY'S FIRST FIELD AND BUMPED THE GENERATION. See
//! "The source identity" below; the short version is that the field the key was
//! built on IS NOT IN THE FILE any more, and the two shapes of the file were
//! measured side by side rather than argued about.
//!
//! # What the store holds, and why it is not the shared `sdists-v9`
//!
//! SDIST-META-1 measured what uv writes for a prepared sdist and found the
//! metadata output and the build tree living in the same shard:
//! `sdists-v9/pypi/<name>/<version>/<rev>/metadata.msgpack` beside
//! `<rev>/src/`, where C31-4's poison lives — a `CMakeCache.txt` recording the
//! ABSOLUTE compiler paths of whichever workspace built the sdist first. The
//! shard as a whole is therefore unshareable, which is why the shared cache is
//! scoped away per job by `retread_scope_sdist_builds`.
//!
//! The metadata half is not. It is `Metadata23` — name, version,
//! `requires_dist`, `requires_python`, extras — 26 to 1599 bytes, and it
//! records the resolution's OUTPUT with none of the build environment's
//! absolute paths. So this store carries EXACTLY:
//!
//!   * [`METADATA_FILE`] — the verbatim uv bytes,
//!   * [`REVISION_FILE`] — the shard's `revision.http`, so a reader can
//!     reconstruct the shard pointer uv resolves through,
//!   * [`COMPLETION_MARKER`] — the provenance record, written LAST.
//!
//! and nothing else. [`PERMITTED_ENTRY_NAMES`] is that list, and it is enforced
//! on admission by shape rather than by enumerating forbidden names: `src/`,
//! `build-setuptools/`, `*.egg-info/`, a built wheel and every future build
//! artefact are all UNREPRESENTABLE, not merely checked for.
//!
//! # The source identity, and why it is not the sdist's sha256 (SDIST-META-3)
//!
//! SDIST-META-1 specified the key's first field as "the sdist's own sha256,
//! read out of `revision.http`". SDIST-META-2 implemented that, ran it, and
//! job 6043382 arm 1 refused ALL SIX sdists with
//! `refused:no-sha256-in-revision.http` — `admitted=0`, and the hit arm it was
//! meant to feed measured nothing.
//!
//! THE MEASUREMENT, `od -c` on the same two shards in the same cache tree, is
//! the whole argument. `openmesh/1.2.1/revision.http` as written on 2026-09-04
//! (589 bytes) begins:
//!
//! ```text
//!   92 b0 "lv_60yeIGVHYQ5Fk"  91 92 a6 "Sha256" d9 40 "6fd3fa41…af3c"
//!   "https://files.pythonhosted.org/packages/66/34/fd86…/openmesh-1.2.1.tar.gz"
//!   "\"0a2fe2633fcae73bd4fd76aeab0be1c7\""  <rkyv trailer>
//! ```
//!
//! and the SAME dist's file as written on 2026-09-07 (515 bytes) begins:
//!
//! ```text
//!   92 b0 "wsfpCFMrKawpvqCO"  90
//!   "https://files.pythonhosted.org/packages/66/34/fd86…/openmesh-1.2.1.tar.gz"
//!   "\"0a2fe2633fcae73bd4fd76aeab0be1c7\""  <rkyv trailer>
//! ```
//!
//! `91 92 a6 …` is a one-element array holding `["Sha256", <64 hex>]`; `90` is
//! the EMPTY array. The hashes list is gone. `strings | grep -c '[0-9a-f]{64}'`
//! is 1 on all the old shards and 0 on all six fresh ones. A digest that is
//! absent cannot be a key field, and a key rule that reads it when it happens
//! to be there and refuses when it is not is TWO rules wearing one name.
//!
//! What survived the change, byte for byte, is the URL and the ETag: identical
//! across both shapes for both dists, which is unsurprising once said out loud
//! — they are what uv REVALIDATES the artefact with, so they are the artefact's
//! identity as far as the cache is concerned. So [`SourceIdentity`] is the URL
//! and the ETag, [`source_digest`] folds them into one hex field, and that is
//! the key's first field under BOTH file shapes. The digest, present or absent,
//! is not read at all — which is what makes this ONE rule.
//!
//! # Why the key is ours and not uv's
//!
//! Read out of uv's `url_metadata`/`url_revision`: uv's own lookup validates
//! only `metadata.matches(name, version)`, and its revision is a function of
//! the sdist URL + ETag. It contains NOTHING about the build inputs — not the
//! interpreter, not the PEP 517 backend, not `PYTHONHASHSEED`. That is adequate
//! inside one job and is exactly the hole a cross-job store must close, so
//! [`entry_key`] adds what uv omits rather than copying uv's key.
//!
//! The fifth field is there by MEASUREMENT, not caution. SDIST-META-1 found
//! gym 0.26.2's two revisions in the live shared cache at the same length
//! (1599 B) with different md5s, the requirements inside `extra == 'testing'`
//! permuted — a set-iteration order, i.e. `PYTHONHASHSEED`. Five of the six
//! sdists in the arm are byte-identical across their revisions; exactly one is
//! seed-sensitive, and with the seed pinned it is stable. An entry built under
//! a different seed is a DIFFERENT ENTRY.
//!
//! # Why the FILE is read here and not in the shell
//!
//! The store's writer is a post-lock harvester and its reader is the scoper's
//! seeder — two shell halves that must derive the SAME key or the store
//! silently never hits. `retread sdist-meta-key` is the one derivation both
//! call, and since SDIST-META-3 it is also the one READER OF THE FILE's layout:
//! it prints `rev=`, `url=`, `etag=` and `source_digest=` so neither half runs
//! `strings` over `revision.http` itself. The previous split — shell finding
//! the digest, product hashing it — is precisely how a layout change turned
//! into six silent refusals instead of one loud one.
//!
//! # What is deliberately NOT here (law 2)
//!
//! There is no `retread-sdist-metadata-store` config key and no backend call
//! site, because the backend is NOT the producer of this metadata: pixi's
//! embedded uv writes it, under `pixi-overlay/uv-cache`, and pixi 0.73.0 does
//! not read `UV_CACHE_DIR` at all. A config key wired into `handler::initialize`
//! would be a stamped directive with no consumer — the same defect law 2 names.
//! The store's production readers are `retread store-reap --store
//! sdist-metadata` (via `tools/store_reap_census.sh`, on every lane job) and
//! this verb (via the harvester and the seeder). Both are live.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The store directory under the persistent root. Read by
/// [`crate::source_build::SDIST_METADATA_STORE_SPEC`] rather than re-spelled
/// there, so the reaper cannot drift from the writer.
pub const CACHE_NAMESPACE: &str = "sdist-metadata";

/// The generation segment. `reap_marker_store` walks EVERY generation it finds
/// and ages an entry under an older one by the same rule with
/// `reason="stale-version"`, so a bump here orphans nothing — the v1 entries
/// SDIST-META-2 could never write stay visible to the census and to the reaper
/// and are simply never served to a v2 seeder.
///
/// BUMPED TO `v2` BY SDIST-META-3, because the key's first field changed from
/// the sdist sha256 to [`source_digest`]. Two entries under one generation
/// keyed by two different rules is the one thing a content-addressed store may
/// not have.
pub const CACHE_VERSION: &str = "v2";

/// The file whose presence MAKES a directory an entry — for the reaper's
/// discovery and for the seeder's completeness test alike. Written LAST, after
/// both payload files are in place, so a half-written entry is invisible to
/// both.
pub const COMPLETION_MARKER: &str = "complete.json";

/// The `<row> reap` / `<row> evicted` stem, so an operator greps one word.
pub const STORE_ROW: &str = "sdist_metadata_store";

/// The store-wide reap try-lock: a DOTFILE beside the generations, so the
/// generation walk can never mistake it for one.
pub const STORE_REAP_LOCK: &str = ".sdist-metadata.reap.lock";

/// uv's own name for the prepared-metadata output
/// (`uv-distribution/src/source/mod.rs`, `const METADATA`). Ours is the same
/// name because the seeder writes the file uv will read.
pub const METADATA_FILE: &str = "metadata.msgpack";

/// The shard-level revision pointer uv resolves BEFORE it reads the metadata.
pub const REVISION_FILE: &str = "revision.http";

/// EXACTLY what an entry directory may contain. Admission refuses anything
/// else — see the module docs for why this is a shape rule and not a list of
/// forbidden names.
pub const PERMITTED_ENTRY_NAMES: [&str; 3] = [METADATA_FILE, REVISION_FILE, COMPLETION_MARKER];

/// The domain-separation tag of [`entry_key`]. Bumping it invalidates every
/// entry, which is the point: it is how a change in WHAT the key covers is
/// announced. `v2` since SDIST-META-3 changed the first field.
pub const KEY_SCHEMA: &str = "retread-sdist-metadata-cache-v2";

/// The domain-separation tag of [`source_digest`], separate from
/// [`KEY_SCHEMA`] so the two hashes can never be confused for one another in a
/// log or in a marker.
pub const SOURCE_SCHEMA: &str = "retread-sdist-source-identity-v1";

/// The rendered key's prefix, so a directory name is self-identifying in a
/// census row.
pub const KEY_PREFIX: &str = "sdm-";

/// A ceiling on the `revision.http` we will read. The live files measure 507 to
/// 589 bytes; a megabyte is four orders of magnitude of headroom and still
/// refuses to load something that is not this file at all.
pub const REVISION_HTTP_MAX_BYTES: u64 = 1 << 20;

/// The scheme the sdist URL starts with, and the anchor the URL scan finds. uv
/// writes the PyPI file URL verbatim.
const URL_MARKER: &[u8] = b"https://";

/// How long an unreferenced entry may sit before the reaper QUARANTINES it, in
/// days. 14, the same horizon as the other five stores, on the standing
/// argument that housekeeping horizons which differ for no stated reason are
/// separate things to get wrong. `0` disables the reaper.
pub const DEFAULT_MAX_AGE_DAYS: u64 = 14;

// ── revision.http, the ONE reader of its layout ─────────────────────────────

/// What [`parse_revision_http`] recovers: the shard pointer uv resolves through
/// and the two fields that identify the artefact under BOTH measured shapes of
/// the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevisionPointer {
    /// The revision directory name — the msgpack string the file OPENS with,
    /// and the directory `metadata.msgpack` sits in. Both shell halves used to
    /// take this with `strings -a | head -1`; it is printed by the verb now so
    /// the file has one reader.
    pub revision_id: String,
    /// The artefact's identity: URL and ETag.
    pub source: SourceIdentity,
}

/// The sdist URL and the ETag uv revalidates it with. See the module docs: this
/// pair is what SURVIVED the uv/pixi change that deleted the sha256, and it is
/// therefore what the key is built on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceIdentity {
    pub url: String,
    pub etag: String,
}

/// Fold [`SourceIdentity`] into the key's first field: sha256 over
/// [`SOURCE_SCHEMA`] and the two LENGTH-PREFIXED halves, hex.
///
/// Length-prefixed for [`entry_key`]'s reason: without it a URL that ends where
/// an ETag begins is indistinguishable from the reverse split.
pub fn source_digest(source: &SourceIdentity) -> String {
    let mut h = Sha256::new();
    h.update(SOURCE_SCHEMA.as_bytes());
    h.update([0u8]);
    for part in [source.url.as_str(), source.etag.as_str()] {
        h.update((part.len() as u64).to_be_bytes());
        h.update(part.as_bytes());
    }
    format!("{:x}", h.finalize())
}

/// THE ONE PARSER of `revision.http`, and it is deliberately a GUARDED FIXED
/// LAYOUT rather than a general msgpack reader.
///
/// The file is msgpack for its first few bytes and then uv's rkyv-archived
/// cache policy, which is why a general msgpack decoder would not read the
/// whole thing anyway. What is parsed, and nothing else:
///
///  1. `0x92` — the two-element array the file opens with. Anything else is
///     refused by shape rather than skipped past.
///  2. A msgpack string header (`fixstr` `0xa0..=0xbf`, or `str8` `0xd9`) and
///     its bytes — the revision id.
///  3. The FIRST `https://` anywhere after it, run to the first `"` — the URL.
///  4. Between that `"` and the next one — the ETag. A weak-validator `W/`
///     immediately before the quote is folded onto the front of the ETag rather
///     than swallowed into the URL, so `W/"abc"` and `"abc"` are DIFFERENT
///     identities, which is what RFC 9110 says they are.
///
/// Step 2 is the whole of the msgpack. The hashes list that sits between it and
/// the URL — `91 92 a6 "Sha256" d9 40 <64 hex>` on an old shard, `90` on a
/// fresh one — is STEPPED OVER, not decoded. That is the ONE-RULE property:
/// the same code path, and the same key, whether the digest is there or not.
pub fn parse_revision_http(bytes: &[u8]) -> anyhow::Result<RevisionPointer> {
    let head = *bytes
        .first()
        .ok_or_else(|| anyhow::anyhow!("sdist-meta-key: revision.http is empty"))?;
    if head != 0x92 {
        anyhow::bail!(
            "sdist-meta-key: revision.http does not open with the 2-element msgpack array \
             (first byte 0x{head:02x}, expected 0x92); this is not a uv revision pointer"
        );
    }
    let (id_start, id_len) = match bytes.get(1).copied() {
        Some(b @ 0xa0..=0xbf) => (2usize, (b & 0x1f) as usize),
        Some(0xd9) => (
            3usize,
            *bytes.get(2).ok_or_else(|| {
                anyhow::anyhow!("sdist-meta-key: revision.http truncated in its str8 length")
            })? as usize,
        ),
        other => anyhow::bail!(
            "sdist-meta-key: revision.http byte 1 is {other:?}, not a msgpack string header; \
             the revision id cannot be read"
        ),
    };
    let id_end = id_start + id_len;
    let raw_id = bytes.get(id_start..id_end).ok_or_else(|| {
        anyhow::anyhow!("sdist-meta-key: revision.http truncated inside its revision id")
    })?;
    let revision_id = std::str::from_utf8(raw_id)
        .map_err(|e| anyhow::anyhow!("sdist-meta-key: revision id is not utf-8: {e}"))?
        .to_string();
    if revision_id.is_empty() || revision_id.contains('/') {
        anyhow::bail!(
            "sdist-meta-key: revision.http names revision id {revision_id:?}, which cannot be a \
             directory name"
        );
    }

    // THE URL. Searched for from the end of the revision id, so the hashes
    // list -- whether it is `91 92 a6 "Sha256" d9 40 <64 hex>` or the empty
    // `90` -- is stepped over without being read.
    let tail = &bytes[id_end..];
    let url_at = tail
        .windows(URL_MARKER.len())
        .position(|w| w == URL_MARKER)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "sdist-meta-key: revision.http (rev {revision_id}) carries no `https://` source \
                 URL; the artefact has no identity to key on"
            )
        })?;
    let rest = &tail[url_at..];
    let url_len = rest
        .iter()
        .position(|&b| b == b'"' || b < 0x21 || b > 0x7e)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "sdist-meta-key: revision.http (rev {revision_id}) has a source URL that never \
                 ends; refusing rather than keying on a run-on read"
            )
        })?;
    let mut url = std::str::from_utf8(&rest[..url_len])
        .map_err(|e| anyhow::anyhow!("sdist-meta-key: source URL is not utf-8: {e}"))?
        .to_string();
    if rest.get(url_len) != Some(&b'"') {
        anyhow::bail!(
            "sdist-meta-key: revision.http (rev {revision_id}) has no ETag after its source URL \
             {url}; an entry keyed without the ETag would be served to a different artefact at \
             the same URL"
        );
    }
    // The weak-validator prefix belongs to the ETAG, not to the URL.
    let mut weak = "";
    if let Some(stripped) = url.strip_suffix("W/") {
        weak = "W/";
        url = stripped.to_string();
    }
    let after_open = &rest[url_len + 1..];
    let close = after_open
        .iter()
        .position(|&b| b == b'"')
        .ok_or_else(|| {
            anyhow::anyhow!(
                "sdist-meta-key: revision.http (rev {revision_id}) opens an ETag and never closes \
                 it"
            )
        })?;
    let etag_body = std::str::from_utf8(&after_open[..close])
        .map_err(|e| anyhow::anyhow!("sdist-meta-key: ETag is not utf-8: {e}"))?;
    if etag_body.is_empty() {
        anyhow::bail!(
            "sdist-meta-key: revision.http (rev {revision_id}) carries an EMPTY ETag for {url}; \
             an empty identity field collapses distinct artefacts onto one entry"
        );
    }
    if url.is_empty() {
        anyhow::bail!("sdist-meta-key: revision.http (rev {revision_id}) carries an empty URL");
    }
    Ok(RevisionPointer {
        revision_id,
        source: SourceIdentity {
            url,
            etag: format!("{weak}{etag_body}"),
        },
    })
}

/// Read and parse a `revision.http` off disk, refusing a file that is too large
/// to be one.
pub fn read_revision_http(path: &Path) -> anyhow::Result<RevisionPointer> {
    let meta = std::fs::metadata(path)
        .map_err(|e| anyhow::anyhow!("sdist-meta-key: cannot stat {}: {e}", path.display()))?;
    if meta.len() > REVISION_HTTP_MAX_BYTES {
        anyhow::bail!(
            "sdist-meta-key: {} is {} bytes, over the {REVISION_HTTP_MAX_BYTES}-byte ceiling for \
             a revision pointer",
            path.display(),
            meta.len()
        );
    }
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("sdist-meta-key: cannot read {}: {e}", path.display()))?;
    parse_revision_http(&bytes)
        .map_err(|e| anyhow::anyhow!("{e} (file {})", path.display()))
}

/// The five facts an entry is identified by. Named fields rather than a tuple
/// because the ORDER is load-bearing — two fields swapped is a key that still
/// looks well-formed and matches nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyInputs {
    /// [`source_digest`] of the artefact's URL and ETag. Name+version are NOT
    /// sufficient: that is precisely what uv validates and precisely the hole a
    /// persistent store must close. This replaced the sdist sha256 in
    /// SDIST-META-3 — see the module docs.
    pub source_digest: String,
    /// The producing uv's version string. `metadata.msgpack` is `rmp_serde` of
    /// uv's `Metadata23`; a struct change across uv versions is a SILENT
    /// misread, not a parse error, so the producer's version is a key field
    /// rather than a footnote.
    pub uv_version: String,
    /// `cp3XX` of the interpreter the build used. Absent from uv's key and
    /// required in ours: a `setup.py` may branch on `sys.version_info`.
    pub python_tag: String,
    /// The PEP 517 backend AND its version, e.g. `setuptools==84.0.0`.
    pub backend: String,
    /// The `PYTHONHASHSEED` the build ran under. See the module docs: this one
    /// is here by measurement.
    pub pythonhashseed: String,
}

impl KeyInputs {
    /// The five fields IN KEY ORDER. The hash and every diagnostic read this
    /// one function, so a field added to the struct without being added to the
    /// key cannot pass unnoticed.
    pub fn fields(&self) -> [(&'static str, &str); 5] {
        [
            ("source_digest", self.source_digest.as_str()),
            ("uv_version", self.uv_version.as_str()),
            ("python_tag", self.python_tag.as_str()),
            ("backend", self.backend.as_str()),
            ("pythonhashseed", self.pythonhashseed.as_str()),
        ]
    }

    /// Refuse a field that is empty or blank. An empty field is not a
    /// conservative default: it COLLAPSES two builds that differ in that field
    /// onto one key, which is a wrong hit rather than a miss.
    pub fn validate(&self) -> anyhow::Result<()> {
        for (name, value) in self.fields() {
            if value.trim().is_empty() {
                anyhow::bail!(
                    "sdist-meta-key: {name} is empty; an empty key field collapses \
                     distinct builds onto one entry"
                );
            }
        }
        Ok(())
    }
}

/// THE ENTRY KEY. sha256 over [`KEY_SCHEMA`] and the five LENGTH-PREFIXED
/// fields of [`KeyInputs`], rendered `sdm-<hex>`.
///
/// Length-prefixed for the reason the hermetic store's key is: without it,
/// `("ab", "c")` and `("a", "bc")` hash identically, and two builds that differ
/// only in where a boundary falls would share an entry.
pub fn entry_key(inputs: &KeyInputs) -> String {
    let mut identity = Sha256::new();
    identity.update(KEY_SCHEMA.as_bytes());
    identity.update([0u8]);
    for (_, value) in inputs.fields() {
        identity.update((value.len() as u64).to_be_bytes());
        identity.update(value.as_bytes());
    }
    format!("{KEY_PREFIX}{:x}", identity.finalize())
}

/// The store root: whatever was named, else
/// [`crate::courier::persistent_store_root_with`] and nothing else. There is
/// deliberately no `RETREAD_CACHE_DIR` branch — that absence is the property
/// that makes the store outlive the job.
pub fn store_root_with(configured: Option<&Path>, env: &dyn Fn(&str) -> Option<String>) -> PathBuf {
    match configured {
        Some(root) => root.to_path_buf(),
        None => crate::courier::persistent_store_root_with(env),
    }
}

/// `<root>/sdist-metadata/v2` — the generation directory the entries sit in,
/// and the directory the harvester and the seeder both address.
pub fn generation_dir(store_root: &Path) -> PathBuf {
    store_root.join(CACHE_NAMESPACE).join(CACHE_VERSION)
}

/// `<root>/sdist-metadata/v2/<key>`.
pub fn entry_dir(store_root: &Path, key: &str) -> PathBuf {
    generation_dir(store_root).join(key)
}

// ── `python_tag`, THE ONE KEY FIELD THAT HAD NO PRODUCER (SDM-PYTAG-1) ──────
//
// N27-RETREAD-24. Four of [`KeyInputs`]'s five fields are MEASURED at the point
// of use: `source_digest` out of the entry's own `revision.http`, `uv_version`
// as the pixi binary's sha256, `backend` read back out of the arm's lock log,
// `pythonhashseed` from the harness pin. The fifth, `python_tag`, was DECLARED
// on argv and then CARRIED: MERGE-B32's proof ran the store against the
// canonical 27-environment workspace with one `cp310` on the command line,
// inherited verbatim from SDIST-META-3 arm 1's row.
//
// THE MEASUREMENT THAT SAYS WHY THAT IS NOT SURVIVABLE, taken from the retained
// canonical lock `mergeB32/artifacts/pixi.lock.MDA-6054364.cert`: the 27
// environments resolve FOUR distinct interpreters —
//
//   cp312  16 envs (default, cpu, gpu, test, test-gpu, tensorboard-tools,
//                   newton-gpu, isaaclab-gpu-latest, ros2-{humble,jazzy}-{cpu,gpu},
//                   pm-newton-gpu 3.12.14, …)
//   cp311   7 envs (groot-sonic-gpu, holosoma, pace, pm-isaaclab,
//                   unitree-rl-lab-gpu, uwlab-gpu, viral-gpu; flashsac-gpu 3.11.16)
//   cp310   5 envs (hover-gpu, jetson 3.10.20, pm-mujoco, robogen, sage)
//   cp38    1 env  (unitree-rl-gym, python 3.8.0)
//
// — so a carried `cp310` is WRONG FOR 22 OF 27. On the read side that is a miss
// (the recomputed key does not equal the entry's directory name, and the seeder
// skips with a row); on the WRITE side it is worse, because an entry produced
// by a cp312 environment is admitted under a cp310 key and a later cp310 arm
// gets served metadata built by a different interpreter, which is precisely the
// cross-job collision the fifth field exists to prevent.
//
// So the tag is DERIVED HERE, per environment, from the environment's own
// resolved interpreter, and printed as a row a shell half can read:
//
//     ### SDIST_META_KEY env=<name> python_tag=<cpXY> source=<lock|interpreter>
//
// Two sources, both measurements and neither an execution: the lock's own
// `python-<version>-<build>` conda artefact (`source=lock`, the shape the
// relock templates have in hand before an environment exists), and an installed
// environment's `lib/python<major>.<minor>` directory (`source=interpreter`,
// for a caller holding a prefix rather than a lock). Running the interpreter
// was rejected: the seeder derives keys for an environment it is standing
// OUTSIDE of, which is the same argument [`ArmFacts`] makes.

/// The row stem every consumer greps for. One word, so `grep SDIST_META_KEY`
/// over an arm's wall finds every environment's tag and nothing else.
pub const TAG_ROW_STEM: &str = "### SDIST_META_KEY";

/// The subdir the tags are read on when a caller names none. The store's
/// readers are the merge relock templates and every one of them locks
/// `linux-64`; a caller on another platform passes `--subdir`.
pub const DEFAULT_SUBDIR: &str = "linux-64";

/// The exit code for "this caller is still on the carried-value path". DISTINCT
/// from a usage error and from [`REVISION_REFUSED_EXIT`] so a wrapper can tell
/// a stale call site from a bad file.
pub const CARRIED_TAG_EXIT: i32 = 4;

/// Where a tag was measured. Printed, because "cp312" alone does not say
/// whether anything was actually read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagSource {
    /// The `python` conda artefact locked for this environment.
    Lock,
    /// An installed prefix's `lib/python<major>.<minor>` directory.
    Interpreter,
}

impl TagSource {
    pub fn as_str(self) -> &'static str {
        match self {
            TagSource::Lock => "lock",
            TagSource::Interpreter => "interpreter",
        }
    }
}

/// One environment's measured tag, and the row it prints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvPythonTag {
    pub env: String,
    /// The version verbatim as measured — `3.12.14`, `3.8.0`. Kept beside the
    /// tag because the tag is lossy and the row is the audit record.
    pub python_version: String,
    pub python_tag: String,
    pub source: TagSource,
}

impl EnvPythonTag {
    /// THE ROW. `python_version` rides on the end rather than in the middle so
    /// that a consumer's `sed -n 's/.*python_tag=\([^ ]*\).*/\1/p` is stable if
    /// more fields are ever appended.
    pub fn row(&self) -> String {
        format!(
            "{TAG_ROW_STEM} env={} python_tag={} source={} python_version={}",
            self.env,
            self.python_tag,
            self.source.as_str(),
            self.python_version,
        )
    }
}

/// `3.12.14` -> `cp312`, `3.8.0` -> `cp38`. THE ONE PLACE the major/minor
/// collapse happens, so the harvester and the seeder cannot spell it two ways.
///
/// Refuses anything it cannot read rather than returning a plausible default:
/// an empty or guessed tag COLLAPSES two interpreters onto one key, which
/// [`KeyInputs::validate`] refuses for exactly the same reason.
pub fn python_tag_from_version(version: &str) -> anyhow::Result<String> {
    let mut parts = version.trim().split('.');
    let major = parts.next().unwrap_or("");
    let minor = parts.next().unwrap_or("");
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(major) || !digits(minor) {
        anyhow::bail!(
            "sdist-meta python tag: `{version}` is not a <major>.<minor>[.<patch>] CPython \
             version; a guessed tag collapses two interpreters onto one entry key"
        );
    }
    Ok(format!("cp{major}{minor}"))
}

/// Every environment `lock_text` locks ON `subdir`, each with the tag of the
/// interpreter IT resolved — never one value carried across all of them, and
/// never an environment that is not on this platform at all.
///
/// AN ENVIRONMENT PRESENT HERE WITH NO `python` IS AN `Err` NAMING IT: a key
/// field that cannot be measured is the state SDM-PYTAG-1 exists to stop being
/// papered over. AN ENVIRONMENT ABSENT FROM THIS SUBDIR IS NOT THAT, and job
/// 6058035 is why the two are separated — `jetson` locks only `linux-aarch64`
/// in the canonical workspace, and treating that as an unmeasurable field made
/// the producer refuse a lock it had read correctly. Absent environments come
/// back in [`TagSet::absent`] and are counted in the row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagSet {
    pub tags: Vec<EnvPythonTag>,
    /// Environments the lock has that lock nothing on this subdir.
    pub absent: Vec<String>,
}

pub fn env_python_tags_from_lock(lock_text: &str, subdir: &str) -> anyhow::Result<TagSet> {
    let locked = crate::workspace::locked_python_versions_by_env(lock_text, subdir)?;
    if locked.present.is_empty() {
        anyhow::bail!(
            "sdist-meta python tag: the lock locks no environment on {subdir} (it has {}); \
             there is nothing to derive a per-environment tag from",
            if locked.absent.is_empty() {
                "no environments at all".to_string()
            } else {
                locked.absent.join(", ")
            }
        );
    }
    let mut untagged: Vec<&str> = Vec::new();
    let mut out = Vec::with_capacity(locked.present.len());
    for (env, version) in &locked.present {
        let Some(version) = version else {
            untagged.push(env.as_str());
            continue;
        };
        out.push(EnvPythonTag {
            env: env.clone(),
            python_tag: python_tag_from_version(version)?,
            python_version: version.clone(),
            source: TagSource::Lock,
        });
    }
    if !untagged.is_empty() {
        anyhow::bail!(
            "sdist-meta python tag: {} environment(s) lock packages on {subdir} but no \
             `python`, so their key field cannot be measured: {}",
            untagged.len(),
            untagged.join(", ")
        );
    }
    Ok(TagSet {
        tags: out,
        absent: locked.absent,
    })
}

/// The tag of an INSTALLED environment, read from `<prefix>/lib/python<M>.<m>`.
///
/// The directory is read, not the interpreter run: the caller derives keys for
/// an environment it is standing outside of, and an execution here would also
/// make the verb refuse on a prefix that exists but cannot be entered.
pub fn env_python_tag_from_prefix(env: &str, prefix: &Path) -> anyhow::Result<EnvPythonTag> {
    let lib = prefix.join("lib");
    let entries = std::fs::read_dir(&lib)
        .map_err(|e| anyhow::anyhow!("sdist-meta python tag: cannot read {}: {e}", lib.display()))?;
    let mut found: Option<String> = None;
    for entry in entries {
        let entry = entry.map_err(|e| {
            anyhow::anyhow!("sdist-meta python tag: cannot read {}: {e}", lib.display())
        })?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(version) = name.strip_prefix("python") else {
            continue;
        };
        if !version.starts_with(|c: char| c.is_ascii_digit()) {
            continue;
        }
        // `python3.12` and `python3.12t` (free-threaded) are two interpreters,
        // and the trailing marker is part of neither this rule nor the tag —
        // refuse rather than silently fold them together.
        if python_tag_from_version(version).is_err() {
            continue;
        }
        match &found {
            Some(first) if first.as_str() != version => anyhow::bail!(
                "sdist-meta python tag: {} holds two interpreters ({first} and {version}); the \
                 interpreter this environment builds under cannot be read",
                lib.display()
            ),
            Some(_) => {}
            None => found = Some(version.to_owned()),
        }
    }
    let Some(version) = found else {
        anyhow::bail!(
            "sdist-meta python tag: no `lib/python<major>.<minor>` under {}; this prefix's key \
             field cannot be measured",
            prefix.display()
        )
    };
    Ok(EnvPythonTag {
        env: env.to_owned(),
        python_tag: python_tag_from_version(&version)?,
        python_version: version,
        source: TagSource::Interpreter,
    })
}

/// The distinct tags in a set of rows, in first-seen order.
pub fn distinct_tags(tags: &[EnvPythonTag]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for tag in tags {
        if !out.iter().any(|seen| seen == &tag.python_tag) {
            out.push(tag.python_tag.clone());
        }
    }
    out
}

// ── `retread sdist-meta-python-tags` ────────────────────────────────────────

/// What the tag verb was asked. Exactly one source; there is no precedence rule
/// between a lock and a prefix because a precedence rule is how the wrong one
/// wins silently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TagArgs {
    Lock {
        lock: PathBuf,
        subdir: String,
        /// Restrict to one environment. `None` prints every one, which is what
        /// a relock template wants: it locks all of them in one pixi call.
        env: Option<String>,
    },
    Prefix {
        env: String,
        prefix: PathBuf,
    },
}

pub fn parse_tag_args(args: &[String]) -> anyhow::Result<TagArgs> {
    let mut lock: Option<PathBuf> = None;
    let mut prefix: Option<PathBuf> = None;
    let mut env: Option<String> = None;
    let mut subdir: Option<String> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || -> anyhow::Result<String> {
            it.next()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("sdist-meta-python-tags: {arg} requires a value"))
        };
        match arg.as_str() {
            "--lock" => lock = Some(PathBuf::from(value()?)),
            "--prefix" => prefix = Some(PathBuf::from(value()?)),
            "--env" => env = Some(value()?),
            "--subdir" => subdir = Some(value()?),
            // THE CARRIED-VALUE PATH, REFUSED BY NAME. A caller that reaches
            // the producer while still holding a literal tag has not been
            // converted, it has been double-wired, and accepting the flag here
            // would let the carried value win over the measured one.
            "--python-tag" => anyhow::bail!(
                "sdist-meta-python-tags: --python-tag is the carried value this verb exists to \
                 replace; it derives the tag per environment and takes no literal"
            ),
            other => anyhow::bail!(
                "sdist-meta-python-tags: unknown arg {other}; expected (--lock <pixi.lock> \
                 [--subdir <s>] [--env <name>] | --prefix <dir> --env <name>)"
            ),
        }
    }
    match (lock, prefix) {
        (Some(_), Some(_)) => anyhow::bail!(
            "sdist-meta-python-tags: --lock and --prefix are exclusive; they are two \
             measurements of the same field and a precedence rule between them is how the \
             wrong one wins silently"
        ),
        (None, None) => anyhow::bail!(
            "sdist-meta-python-tags: one of --lock <pixi.lock> or --prefix <dir> is required"
        ),
        (Some(lock), None) => Ok(TagArgs::Lock {
            lock,
            subdir: subdir.unwrap_or_else(|| DEFAULT_SUBDIR.to_string()),
            env,
        }),
        (None, Some(prefix)) => {
            if subdir.is_some() {
                anyhow::bail!(
                    "sdist-meta-python-tags: --subdir belongs to --lock; an installed prefix \
                     is already one platform"
                );
            }
            let env = env.ok_or_else(|| {
                anyhow::anyhow!("sdist-meta-python-tags: --prefix requires --env <name>")
            })?;
            Ok(TagArgs::Prefix { env, prefix })
        }
    }
}

/// Resolve the rows a [`TagArgs`] asks for.
pub fn tag_rows(args: &TagArgs) -> anyhow::Result<TagSet> {
    match args {
        TagArgs::Lock { lock, subdir, env } => {
            let text = std::fs::read_to_string(lock).map_err(|e| {
                anyhow::anyhow!("sdist-meta-python-tags: cannot read {}: {e}", lock.display())
            })?;
            let all = env_python_tags_from_lock(&text, subdir)?;
            let Some(env) = env else { return Ok(all) };
            let picked: Vec<EnvPythonTag> =
                all.tags.iter().filter(|t| &t.env == env).cloned().collect();
            if picked.is_empty() {
                // NAME WHICH KIND OF ABSENCE IT IS. "no such environment" and
                // "that environment is not on this platform" send a caller to
                // two different fixes, and job 6058035 is the run that proved
                // the second one is real (`jetson`, linux-aarch64 only).
                let where_ = if all.absent.iter().any(|a| a == env) {
                    format!("environment `{env}` locks nothing on {subdir}")
                } else {
                    format!("no environment `{env}` at all")
                };
                anyhow::bail!(
                    "sdist-meta-python-tags: {} has {where_}; on {subdir} it has: {}",
                    lock.display(),
                    all.tags
                        .iter()
                        .map(|t| t.env.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            Ok(TagSet {
                tags: picked,
                absent: all.absent,
            })
        }
        TagArgs::Prefix { env, prefix } => Ok(TagSet {
            tags: vec![env_python_tag_from_prefix(env, prefix)?],
            absent: Vec::new(),
        }),
    }
}


pub fn run_tags(args: &TagArgs) -> anyhow::Result<i32> {
    let set = tag_rows(args)?;
    let rows = &set.tags;
    for row in rows {
        println!("{}", row.row());
    }
    let distinct = distinct_tags(rows);
    let source = rows.first().map(|r| r.source.as_str()).unwrap_or("none");
    // THE SUMMARY IS THE ROW THAT MATTERS TO THE SEEDER. `distinct_tags=1` is
    // the only shape a single overlay can be seeded for under one key; anything
    // higher and the consumer must key per environment or refuse. It is printed
    // rather than enforced here because this verb is the PRODUCER — the policy
    // belongs to the call site (N27-RETREAD-25).
    println!(
        "{TAG_ROW_STEM} TOTAL envs={} distinct_tags={} tags={} absent_on_subdir={} source={source}",
        rows.len(),
        distinct.len(),
        distinct.join(","),
        if set.absent.is_empty() {
            "0".to_string()
        } else {
            format!("{}:{}", set.absent.len(), set.absent.join(","))
        },
    );
    Ok(0)
}

// ── `retread sdist-meta-key` ────────────────────────────────────────────────

/// The refusal exit code for a `revision.http` this verb cannot read. It is
/// DISTINCT from a usage error so the harvester can turn it into a per-shard
/// row and keep going, while a usage error still stops the job.
pub const REVISION_REFUSED_EXIT: i32 = 3;

/// The four facts the CALLER supplies. Every one is an ARGUMENT: the seeder
/// derives the key for the arm it is about to scope while standing outside that
/// build, so an environment lookup here would be the wrong answer as well as
/// the wrong shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmFacts {
    pub uv_version: String,
    pub python_tag: String,
    pub backend: String,
    pub pythonhashseed: String,
}

/// A caller that passed BOTH a literal `--python-tag` and the per-environment
/// derivation. Its own type so [`key_main`] can turn it into a ROW and
/// [`CARRIED_TAG_EXIT`], rather than one more anonymous usage error: this is
/// the defect SDM-PYTAG-1 is about, and a wrapper needs to be able to tell it
/// apart from a typo.
#[derive(Debug)]
pub struct CarriedTagRefusal(pub String);

impl std::fmt::Display for CarriedTagRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CarriedTagRefusal {}

/// Where `sdist-meta-key` gets its `python_tag`. Not an `Option<String>` plus a
/// second `Option<PathBuf>`: two optionals admit the both-set state as a value,
/// and the whole point here is that the both-set state has no meaning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PythonTagChoice {
    /// The literal a caller declared. Still accepted — a single-environment
    /// caller measures its own tag and passes it — but no longer the only way.
    Literal(String),
    /// Derived from the named environment's own locked interpreter.
    FromLock {
        lock: PathBuf,
        subdir: String,
        env: String,
    },
}

/// What the verb was asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub uv_version: String,
    /// SDM-PYTAG-1: a CHOICE, not a string. See [`PythonTagChoice`].
    pub python_tag: PythonTagChoice,
    pub backend: String,
    pub pythonhashseed: String,
    /// The `revision.http` to key on. `None` is `--probe`: resolve and print
    /// the store paths WITHOUT a key. The seeder needs exactly that — one call
    /// to learn where the store is before it has any entry in hand — and
    /// before SDIST-META-3 it faked it by passing `--sdist-sha256 0`, a
    /// placeholder key input that only worked because the field was a free
    /// string.
    pub revision_http: Option<PathBuf>,
    /// `None` means the persistent root the product itself would resolve.
    pub store_root: Option<PathBuf>,
}

impl Args {
    /// The four facts, with `python_tag` MEASURED if it was not declared.
    ///
    /// Returns the [`EnvPythonTag`] alongside so the caller can print the same
    /// row the producer verb prints — one row shape, one reader.
    pub fn resolve_facts(&self) -> anyhow::Result<(ArmFacts, Option<EnvPythonTag>)> {
        let (python_tag, measured) = match &self.python_tag {
            PythonTagChoice::Literal(tag) => (tag.clone(), None),
            PythonTagChoice::FromLock { lock, subdir, env } => {
                let args = TagArgs::Lock {
                    lock: lock.clone(),
                    subdir: subdir.clone(),
                    env: Some(env.clone()),
                };
                let row = tag_rows(&args)?.tags.into_iter().next().ok_or_else(|| {
                    anyhow::anyhow!("sdist-meta-key: no tag row for environment `{env}`")
                })?;
                (row.python_tag.clone(), Some(row))
            }
        };
        Ok((
            ArmFacts {
                uv_version: self.uv_version.clone(),
                python_tag,
                backend: self.backend.clone(),
                pythonhashseed: self.pythonhashseed.clone(),
            },
            measured,
        ))
    }
}

pub fn parse_args(args: &[String]) -> anyhow::Result<Args> {
    let mut uv_version: Option<String> = None;
    let mut python_tag: Option<String> = None;
    let mut backend: Option<String> = None;
    let mut pythonhashseed: Option<String> = None;
    let mut store_root: Option<PathBuf> = None;
    let mut revision_http: Option<PathBuf> = None;
    let mut probe = false;
    let mut tag_lock: Option<PathBuf> = None;
    let mut tag_env: Option<String> = None;
    let mut tag_subdir: Option<String> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || -> anyhow::Result<String> {
            it.next()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("sdist-meta-key: {arg} requires a value"))
        };
        match arg.as_str() {
            "--revision-http" => revision_http = Some(PathBuf::from(value()?)),
            "--probe" => probe = true,
            "--uv-version" => uv_version = Some(value()?),
            "--python-tag" => python_tag = Some(value()?),
            "--python-tag-from-lock" => tag_lock = Some(PathBuf::from(value()?)),
            "--env" => tag_env = Some(value()?),
            "--subdir" => tag_subdir = Some(value()?),
            "--backend" => backend = Some(value()?),
            "--pythonhashseed" => pythonhashseed = Some(value()?),
            "--store-root" => store_root = Some(PathBuf::from(value()?)),
            other => anyhow::bail!(
                "sdist-meta-key: unknown arg {other}; expected (--revision-http <file> | --probe) \
                 --uv-version (--python-tag <cpXY> | --python-tag-from-lock <pixi.lock> --env \
                 <name> [--subdir <s>]) --backend --pythonhashseed [--store-root]"
            ),
        }
    }
    // `--sdist-sha256` is GONE rather than accepted-and-ignored. A caller still
    // passing it is a caller that has not been taught the new rule, and a silent
    // acceptance would key it on something it did not intend.
    let missing = |name: &str| anyhow::anyhow!("sdist-meta-key: --{name} is required");
    // SDM-PYTAG-1. THE BOTH-SET STATE IS THE DEFECT, SO IT IS THE REFUSAL.
    //
    // MERGE-B32 ran the store over the canonical 27-environment workspace with
    // ONE `cp310` on argv, carried from a single-environment arm's row; four
    // distinct interpreters are locked there, so 22 environments would have
    // been keyed under an interpreter they do not use. A caller holding both a
    // literal and a lock is that caller mid-conversion, and letting either win
    // is how the carried value survives the fix.
    let tag_choice = match (python_tag, tag_lock) {
        (Some(_), Some(lock)) => {
            return Err(CarriedTagRefusal(format!(
                "sdist-meta-key: --python-tag (a carried literal) and --python-tag-from-lock {} \
                 (the measured value) were both given; they are two answers for one key field \
                 and this verb will not pick between them",
                lock.display()
            ))
            .into());
        }
        (Some(tag), None) => {
            if tag_env.is_some() || tag_subdir.is_some() {
                anyhow::bail!(
                    "sdist-meta-key: --env/--subdir belong to --python-tag-from-lock; with a \
                     literal --python-tag they name an environment nothing is read from"
                );
            }
            PythonTagChoice::Literal(tag)
        }
        (None, Some(lock)) => PythonTagChoice::FromLock {
            lock,
            subdir: tag_subdir.unwrap_or_else(|| DEFAULT_SUBDIR.to_string()),
            env: tag_env.ok_or_else(|| {
                anyhow::anyhow!("sdist-meta-key: --python-tag-from-lock requires --env <name>")
            })?,
        },
        (None, None) => {
            return Err(missing("python-tag (or --python-tag-from-lock <pixi.lock> --env <name>)"));
        }
    };
    let uv_version = uv_version.ok_or_else(|| missing("uv-version"))?;
    let backend = backend.ok_or_else(|| missing("backend"))?;
    let pythonhashseed = pythonhashseed.ok_or_else(|| missing("pythonhashseed"))?;
    match (probe, revision_http.is_some()) {
        (true, true) => anyhow::bail!(
            "sdist-meta-key: --probe and --revision-http are exclusive; --probe resolves the \
             store WITHOUT deriving a key"
        ),
        (false, false) => anyhow::bail!(
            "sdist-meta-key: one of --revision-http <file> or --probe is required"
        ),
        _ => {}
    }
    Ok(Args {
        uv_version,
        python_tag: tag_choice,
        backend,
        pythonhashseed,
        revision_http,
        store_root,
    })
}

/// Parse and run `sdist-meta-key`, TURNING THE CARRIED-VALUE REFUSAL INTO A ROW
/// AND [`CARRIED_TAG_EXIT`] rather than an anonymous usage error.
///
/// The seeder reads rows; a refusal that only reaches stderr with the same exit
/// code as a typo is a refusal it cannot act on differently (law 9).
pub fn key_main(argv: &[String]) -> anyhow::Result<i32> {
    match parse_args(argv) {
        Ok(args) => run(&args),
        Err(err) => {
            if let Some(refusal) = err.downcast_ref::<CarriedTagRefusal>() {
                println!(
                    "### SDIST-META KEY refused=carried-python-tag version={CACHE_VERSION}"
                );
                eprintln!("{refusal}");
                return Ok(CARRIED_TAG_EXIT);
            }
            Err(err)
        }
    }
}

/// Print the key and the paths derived from it, plus every field READ OUT OF
/// the file. The paths are printed because the shell halves would otherwise
/// each join the store's segments themselves; `rev=`, `url=` and `etag=` are
/// printed because they would otherwise each run `strings` over
/// `revision.http` themselves — which is the same second-implementation defect,
/// and the one that actually bit (SDIST-META-2, `admitted=0`).
pub fn run(args: &Args) -> anyhow::Result<i32> {
    let root = store_root_with(args.store_root.as_deref(), &|k| std::env::var(k).ok());
    // MEASURE FIRST, INCLUDING ON `--probe`. A caller whose lock cannot yield
    // the environment's tag must find that out on the one probe call it makes
    // before it has any entry in hand, not on entry number one.
    let (facts, measured) = args.resolve_facts()?;
    if let Some(row) = &measured {
        println!("{}", row.row());
    }
    let Some(path) = args.revision_http.as_deref() else {
        println!(
            "### SDIST-META KEY probe=1 version={CACHE_VERSION} store={} entry=<probe>",
            generation_dir(&root).display(),
        );
        return Ok(0);
    };
    let pointer = match read_revision_http(path) {
        Ok(p) => p,
        Err(e) => {
            // A REFUSAL, NOT A PANIC AND NOT A DEFAULT. It goes to stdout as a
            // row so the harvester can attribute it to the shard, and to stderr
            // so a human running the verb by hand sees it.
            println!("### SDIST-META KEY refused=revision-http version={CACHE_VERSION}");
            eprintln!("{e}");
            return Ok(REVISION_REFUSED_EXIT);
        }
    };
    let digest = source_digest(&pointer.source);
    let inputs = KeyInputs {
        source_digest: digest.clone(),
        uv_version: facts.uv_version.clone(),
        python_tag: facts.python_tag.clone(),
        backend: facts.backend.clone(),
        pythonhashseed: facts.pythonhashseed.clone(),
    };
    inputs.validate()?;
    let key = entry_key(&inputs);
    println!(
        "### SDIST-META KEY key={key} version={CACHE_VERSION} rev={} source_digest={digest} \
         url={} etag={} store={} entry={}",
        pointer.revision_id,
        pointer.source.url,
        pointer.source.etag,
        generation_dir(&root).display(),
        entry_dir(&root, &key).display(),
    );
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE FRESH SHAPE, byte for byte as measured on 2026-09-07 in
    /// `…-sdm2-6043382-a1/pixi/uv-cache/sdists-v9/pypi/openmesh/1.2.1`: the
    /// two-element array, the 16-byte revision id, the EMPTY hashes array
    /// (`0x90`), the URL, the quoted ETag.
    fn fresh_bytes() -> Vec<u8> {
        let mut v = vec![0x92u8, 0xb0];
        v.extend_from_slice(b"wsfpCFMrKawpvqCO");
        v.push(0x90);
        v.extend_from_slice(
            b"https://files.pythonhosted.org/packages/66/34/fd86/openmesh-1.2.1.tar.gz\
              \"0a2fe2633fcae73bd4fd76aeab0be1c7\"",
        );
        v.extend_from_slice(&[0u8; 16]);
        v
    }

    /// THE STALE SHAPE, as measured on the 2026-09-04 file for the SAME dist:
    /// identical but for the revision id and the one-element hashes array
    /// holding `["Sha256", <64 hex>]`. The URL and the ETag are the same bytes
    /// — that is the measurement the whole key rule rests on.
    fn stale_bytes() -> Vec<u8> {
        let mut v = vec![0x92u8, 0xb0];
        v.extend_from_slice(b"lv_60yeIGVHYQ5Fk");
        v.extend_from_slice(&[0x91, 0x92, 0xa6]);
        v.extend_from_slice(b"Sha256");
        v.extend_from_slice(&[0xd9, 0x40]);
        v.extend_from_slice(
            b"6fd3fa41a68148e4a7523f562426aa9758bf65ccc6642abcf79c37bae9c6af3c",
        );
        v.extend_from_slice(
            b"https://files.pythonhosted.org/packages/66/34/fd86/openmesh-1.2.1.tar.gz\
              \"0a2fe2633fcae73bd4fd76aeab0be1c7\"",
        );
        v.extend_from_slice(&[0u8; 16]);
        v
    }

    fn facts() -> ArmFacts {
        ArmFacts {
            uv_version: "0.9.5".to_string(),
            python_tag: "cp310".to_string(),
            backend: "setuptools==84.0.0".to_string(),
            pythonhashseed: "0".to_string(),
        }
    }

    fn inputs() -> KeyInputs {
        let p = parse_revision_http(&fresh_bytes()).expect("the fresh shape parses");
        KeyInputs {
            source_digest: source_digest(&p.source),
            uv_version: facts().uv_version,
            python_tag: facts().python_tag,
            backend: facts().backend,
            pythonhashseed: facts().pythonhashseed,
        }
    }

    /// SDIST-META-3's REASON FOR EXISTING. The fresh shard — the one the whole
    /// arm now writes — carries NO sha256 anywhere, and it must still key.
    /// SDIST-META-2's rule refused exactly this file, six times, `admitted=0`.
    #[test]
    fn the_fresh_shape_with_no_digest_still_derives_a_key() {
        let bytes = fresh_bytes();
        assert_eq!(
            bytes
                .windows(64)
                .filter(|w| w.iter().all(|b| b.is_ascii_hexdigit()))
                .count(),
            0,
            "the fixture must have NO 64-hex run, or it is not the fresh shape"
        );
        let p = parse_revision_http(&bytes).expect("the fresh shape must parse");
        assert_eq!(p.revision_id, "wsfpCFMrKawpvqCO");
        assert_eq!(p.source.etag, "0a2fe2633fcae73bd4fd76aeab0be1c7");
        assert!(p.source.url.ends_with("openmesh-1.2.1.tar.gz"));
        assert!(entry_key(&inputs()).starts_with(KEY_PREFIX));
    }

    /// ONE RULE, NOT TWO. The stale shape carries a digest; the fresh one does
    /// not; the URL and the ETag are identical. The key rule must therefore
    /// produce the SAME source digest for both — which is only true if the
    /// sha256 is not read at all. If someone reinstates "use the digest when it
    /// is there", this goes red.
    #[test]
    fn the_digest_is_ignored_so_both_file_shapes_give_one_key() {
        let fresh = parse_revision_http(&fresh_bytes()).expect("fresh parses");
        let stale = parse_revision_http(&stale_bytes()).expect("stale parses");
        assert_ne!(
            fresh.revision_id, stale.revision_id,
            "the two shapes differ in their revision id; the fixtures are real"
        );
        assert_eq!(
            stale_bytes()
                .windows(64)
                .filter(|w| w.iter().all(|b| b.is_ascii_hexdigit()))
                .count(),
            1,
            "the stale fixture must carry exactly the one 64-hex digest"
        );
        assert_eq!(fresh.source, stale.source, "the identity must be the same");
        assert_eq!(
            source_digest(&fresh.source),
            source_digest(&stale.source),
            "the digest field moved the key; the rule read the sha256 after all"
        );
    }

    /// NO ETAG IS A REFUSAL THAT NAMES THE ETAG, not a key over the URL alone.
    /// Two artefacts can share a URL across a re-upload; the ETag is what tells
    /// them apart, and a store that keyed without it would serve the first to
    /// the second.
    #[test]
    fn a_revision_without_an_etag_is_refused_by_name() {
        let mut v = vec![0x92u8, 0xb0];
        v.extend_from_slice(b"wsfpCFMrKawpvqCO");
        v.push(0x90);
        v.extend_from_slice(b"https://files.pythonhosted.org/x/openmesh-1.2.1.tar.gz");
        v.extend_from_slice(&[0u8; 8]);
        let error = parse_revision_http(&v).expect_err("a missing ETag must refuse");
        assert!(
            error.to_string().contains("ETag"),
            "the refusal must NAME the ETag: {error}"
        );
    }

    /// A file that is not a revision pointer at all is refused by SHAPE, on its
    /// first byte, rather than scanned hopefully for something URL-like.
    #[test]
    fn a_file_that_is_not_a_revision_pointer_is_refused_on_its_first_byte() {
        let error = parse_revision_http(b"https://example.invalid/x.tar.gz\"e\"")
            .expect_err("a bare URL is not a revision pointer");
        assert!(error.to_string().contains("0x92"), "{error}");
        assert!(
            parse_revision_http(&[]).is_err(),
            "an empty file must refuse"
        );
    }

    /// A WEAK VALIDATOR IS A DIFFERENT ETAG. RFC 9110 says `W/"x"` and `"x"`
    /// are not the same validator, so they must not be the same key — and the
    /// `W/` must not be swallowed into the URL either.
    #[test]
    fn a_weak_validator_belongs_to_the_etag_and_moves_the_key() {
        let mut v = vec![0x92u8, 0xb0];
        v.extend_from_slice(b"wsfpCFMrKawpvqCO");
        v.push(0x90);
        v.extend_from_slice(b"https://files.pythonhosted.org/x/o.tar.gz W/\"abc\"");
        let v: Vec<u8> = v.into_iter().filter(|&b| b != b' ').collect();
        let weak = parse_revision_http(&v).expect("weak validator parses");
        assert_eq!(weak.source.etag, "W/abc");
        assert!(
            weak.source.url.ends_with("o.tar.gz"),
            "the W/ leaked into the URL: {}",
            weak.source.url
        );
        let mut strong_bytes = vec![0x92u8, 0xb0];
        strong_bytes.extend_from_slice(b"wsfpCFMrKawpvqCO");
        strong_bytes.push(0x90);
        strong_bytes.extend_from_slice(b"https://files.pythonhosted.org/x/o.tar.gz\"abc\"");
        let strong = parse_revision_http(&strong_bytes).expect("strong parses");
        assert_ne!(
            source_digest(&weak.source),
            source_digest(&strong.source),
            "W/\"abc\" and \"abc\" must not share an entry"
        );
    }

    /// The key is a FUNCTION of the five fields: same inputs, same key, across
    /// calls. Without this the two shell halves could differ and the store
    /// would read as permanently cold.
    #[test]
    fn the_key_is_deterministic_for_the_same_inputs() {
        assert_eq!(entry_key(&inputs()), entry_key(&inputs()));
        assert!(entry_key(&inputs()).starts_with(KEY_PREFIX));
        assert_eq!(entry_key(&inputs()).len(), KEY_PREFIX.len() + 64);
    }

    /// EVERY field moves the key. This is the guard that fails if a field is
    /// dropped from `KeyInputs::fields` — the exact defect that would make the
    /// store serve a `PYTHONHASHSEED`-permuted entry to an arm that pinned a
    /// different seed.
    #[test]
    fn every_key_field_changes_the_key() {
        let base = entry_key(&inputs());
        let mutations: [(&str, fn(&mut KeyInputs)); 5] = [
            ("source_digest", |i| i.source_digest.push('x')),
            ("uv_version", |i| i.uv_version = "0.12.5".to_string()),
            ("python_tag", |i| i.python_tag = "cp312".to_string()),
            ("backend", |i| i.backend = "setuptools==85.0.0".to_string()),
            ("pythonhashseed", |i| i.pythonhashseed = "1".to_string()),
        ];
        for (name, mutate) in mutations {
            let mut moved = inputs();
            mutate(&mut moved);
            assert_ne!(
                entry_key(&moved),
                base,
                "{name} did not move the key; it is not covered by the hash"
            );
        }
    }

    /// Both halves of the SOURCE identity move the source digest, and it is
    /// length-prefixed so a boundary shift between them is visible.
    #[test]
    fn both_halves_of_the_source_identity_move_its_digest() {
        let base = SourceIdentity {
            url: "https://h/ab".to_string(),
            etag: "c".to_string(),
        };
        let other_url = SourceIdentity {
            url: "https://h/ac".to_string(),
            ..base.clone()
        };
        let other_etag = SourceIdentity {
            etag: "d".to_string(),
            ..base.clone()
        };
        let shifted = SourceIdentity {
            url: "https://h/a".to_string(),
            etag: "bc".to_string(),
        };
        assert_ne!(source_digest(&base), source_digest(&other_url));
        assert_ne!(source_digest(&base), source_digest(&other_etag));
        assert_ne!(
            source_digest(&base),
            source_digest(&shifted),
            "the halves are not length-prefixed; a boundary shift is invisible"
        );
    }

    /// The length prefixes are load-bearing: without them a boundary shift
    /// between two adjacent fields is invisible to the hash.
    #[test]
    fn a_boundary_shift_between_fields_moves_the_key() {
        let mut a = inputs();
        a.python_tag = "cp31".to_string();
        a.backend = "0setuptools==84.0.0".to_string();
        let mut b = inputs();
        b.python_tag = "cp310".to_string();
        b.backend = "setuptools==84.0.0".to_string();
        assert_ne!(entry_key(&a), entry_key(&b));
    }

    /// An empty field is REFUSED, not hashed. Hashing it would collapse every
    /// build that lacks the fact onto one entry.
    #[test]
    fn an_empty_key_field_is_refused() {
        let mut empty = inputs();
        empty.pythonhashseed = "   ".to_string();
        let error = empty.validate().expect_err("blank field must refuse");
        assert!(
            error.to_string().contains("pythonhashseed"),
            "the refusal must NAME the field: {error}"
        );
    }

    /// The store shape, asserted against the constants the spec and both shell
    /// halves read — so a rename here shows up as a failure rather than as a
    /// store nobody writes to. `v2` since SDIST-META-3: an entry written under
    /// the old key rule must not be addressable by the new one.
    #[test]
    fn the_store_paths_are_root_namespace_version_key() {
        let root = Path::new("/tmp/root");
        let key = entry_key(&inputs());
        assert_eq!(CACHE_VERSION, "v2");
        assert!(
            KEY_SCHEMA.ends_with("-v2"),
            "the generation and the key schema must be bumped together: {KEY_SCHEMA}"
        );
        assert_eq!(
            entry_dir(root, &key),
            root.join("sdist-metadata").join("v2").join(&key)
        );
        assert_eq!(generation_dir(root), root.join("sdist-metadata").join("v2"));
        assert_ne!(
            entry_dir(root, &key),
            root.join("sdist-metadata").join("v1").join(&key),
            "a v1 entry must not be addressable as a v2 one"
        );
    }

    /// The permitted set IS the safe subset SDIST-META-1 measured: the two uv
    /// files and the marker, and no `src/`.
    #[test]
    fn the_permitted_entry_names_are_exactly_the_safe_subset() {
        assert_eq!(
            PERMITTED_ENTRY_NAMES,
            ["metadata.msgpack", "revision.http", "complete.json"]
        );
    }

    /// The default root is the PERSISTENT one and has no `RETREAD_CACHE_DIR`
    /// branch. Reinstating one turns this red.
    #[test]
    fn the_default_store_root_is_persistent_not_the_job_local_redirect() {
        let env = |key: &str| match key {
            "XDG_CACHE_HOME" => Some("/xdg".to_string()),
            "RETREAD_CACHE_DIR" => Some("/job-local/caches/retread".to_string()),
            _ => None,
        };
        assert_eq!(store_root_with(None, &env), PathBuf::from("/xdg/retread"));
        assert_eq!(
            store_root_with(Some(Path::new("/named")), &env),
            PathBuf::from("/named")
        );
    }

    #[test]
    fn parse_requires_the_four_facts_and_exactly_one_source() {
        let base = [
            "--uv-version",
            "0.9.5",
            "--python-tag",
            "cp310",
            "--backend",
            "setuptools==84.0.0",
            "--pythonhashseed",
            "0",
        ];
        let full: Vec<String> = base
            .iter()
            .chain(["--revision-http", "/r.http", "--store-root", "/named"].iter())
            .map(|s| s.to_string())
            .collect();
        let args = parse_args(&full).expect("full argv parses");
        assert_eq!(args.store_root, Some(PathBuf::from("/named")));
        assert_eq!(args.revision_http, Some(PathBuf::from("/r.http")));
        assert_eq!(
            args.python_tag,
            PythonTagChoice::Literal("cp310".to_string())
        );
        assert_eq!(
            args.resolve_facts().expect("a literal needs no lock").0.python_tag,
            "cp310"
        );

        let probe: Vec<String> = base
            .iter()
            .chain(["--probe"].iter())
            .map(|s| s.to_string())
            .collect();
        assert!(
            parse_args(&probe).expect("probe parses").revision_http.is_none(),
            "--probe must resolve the store without a key"
        );

        let neither: Vec<String> = base.iter().map(|s| s.to_string()).collect();
        assert!(
            parse_args(&neither).is_err(),
            "neither --revision-http nor --probe must refuse"
        );

        let both: Vec<String> = base
            .iter()
            .chain(["--probe", "--revision-http", "/r.http"].iter())
            .map(|s| s.to_string())
            .collect();
        assert!(parse_args(&both).is_err(), "both sources must refuse");

        let short: Vec<String> = base[..6]
            .iter()
            .chain(["--probe"].iter())
            .map(|s| s.to_string())
            .collect();
        let error = parse_args(&short).expect_err("a missing fact must refuse");
        assert!(error.to_string().contains("pythonhashseed"), "{error}");

        // The RETIRED flag is refused, not ignored. A caller still passing it
        // has not been taught the new rule.
        let retired: Vec<String> = base
            .iter()
            .chain(["--probe", "--sdist-sha256", "abc"].iter())
            .map(|s| s.to_string())
            .collect();
        assert!(
            parse_args(&retired).is_err(),
            "--sdist-sha256 must be REFUSED, not silently ignored"
        );
    }

    // ── SDM-PYTAG-1: the fifth key field, measured per environment ──────────

    /// A scratch directory, the same shape `store_reap`'s tests use. No
    /// `tempfile` dependency is added for four tests.
    fn scratch(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "retread-sdm-pytag-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    /// TWO ENVIRONMENTS ON TWO INTERPRETERS, which is the canonical
    /// workspace's shape in miniature: `mergeB32/artifacts/pixi.lock.MDA-6054364.cert`
    /// locks cp312, cp311, cp310 and cp38 across its 27 environments.
    ///
    /// `legacy` also carries `python-dotenv`, and `gpu` carries `python_abi`,
    /// because both are packages a substring search for `python-3.` or
    /// `python` finds first in a real lock.
    fn two_env_lock() -> &'static str {
        "version: 6\n\
         environments:\n\
        \x20 gpu:\n\
        \x20   channels:\n\
        \x20   - url: https://conda.anaconda.org/conda-forge/\n\
        \x20   packages:\n\
        \x20     linux-64:\n\
        \x20     - conda: https://prefix.dev/conda-forge/linux-64/python_abi-3.12-8_cp312.conda\n\
        \x20     - conda: https://prefix.dev/conda-forge/linux-64/python-3.12.0-hab00c5b_0_cpython.conda\n\
        \x20     - pypi: https://files.pythonhosted.org/packages/aa/torch-2.7.0-cp312-cp312-linux_x86_64.whl\n\
        \x20 legacy:\n\
        \x20   packages:\n\
        \x20     linux-64:\n\
        \x20     - conda: https://prefix.dev/conda-forge/linux-64/python-dotenv-1.0.1-pyhd8ed1ab_0.conda\n\
        \x20     - conda: https://prefix.dev/conda-forge/linux-64/python-3.8.0-h357f687_5.tar.bz2\n\
         packages: []\n"
    }

    #[test]
    fn the_tag_is_the_major_minor_and_a_version_it_cannot_read_is_refused() {
        assert_eq!(python_tag_from_version("3.12.14").unwrap(), "cp312");
        assert_eq!(python_tag_from_version("3.8.0").unwrap(), "cp38");
        assert_eq!(python_tag_from_version("3.10").unwrap(), "cp310");
        for bad in ["", "3", "3.", "3.x", "cp312", "3.12t"] {
            assert!(
                python_tag_from_version(bad).is_err(),
                "`{bad}` must refuse rather than guess a tag"
            );
        }
    }

    #[test]
    fn two_environments_on_two_pythons_derive_two_different_tags() {
        let set = env_python_tags_from_lock(two_env_lock(), "linux-64")
            .expect("the fixture lock parses");
        let rows = set.tags;
        assert!(set.absent.is_empty(), "both environments are on linux-64");
        assert_eq!(rows.len(), 2, "every environment gets a row: {rows:?}");
        let gpu = rows.iter().find(|r| r.env == "gpu").expect("gpu row");
        let legacy = rows.iter().find(|r| r.env == "legacy").expect("legacy row");
        assert_eq!(gpu.python_tag, "cp312");
        assert_eq!(legacy.python_tag, "cp38");
        assert_ne!(
            gpu.python_tag, legacy.python_tag,
            "two interpreters must not collapse onto one tag"
        );
        assert_eq!(distinct_tags(&rows).len(), 2);
        assert_eq!(
            gpu.row(),
            "### SDIST_META_KEY env=gpu python_tag=cp312 source=lock python_version=3.12.0"
        );
        assert_eq!(gpu.source, TagSource::Lock);
    }

    /// THE POINT OF THE WHOLE LANE. Same artefact, same uv, same backend, same
    /// seed; only the environment differs — and the key must differ, because
    /// serving one environment's metadata to the other is exactly the
    /// cross-interpreter collision the fifth field exists to prevent.
    #[test]
    fn the_per_environment_tags_move_the_entry_key() {
        let rows = env_python_tags_from_lock(two_env_lock(), "linux-64").unwrap().tags;
        let key_for = |tag: &str| {
            entry_key(&KeyInputs {
                source_digest: "d".repeat(64),
                uv_version: "0.9.5".to_string(),
                python_tag: tag.to_string(),
                backend: "setuptools==84.0.0".to_string(),
                pythonhashseed: "0".to_string(),
            })
        };
        let keys: Vec<String> = rows.iter().map(|r| key_for(&r.python_tag)).collect();
        assert_ne!(
            keys[0], keys[1],
            "two environments on two interpreters must not share one entry key"
        );
        // And the CARRIED value is neither of them for at least one of the two,
        // which is the state MERGE-B32 ran: one `cp310` for 27 environments.
        let carried = key_for("cp310");
        assert!(
            keys.iter().all(|k| k != &carried),
            "the carried cp310 key must match no environment in this lock"
        );
    }

    #[test]
    fn an_environment_that_locks_no_python_is_refused_by_name() {
        let lock = "environments:\n\
                   \x20 headless:\n\
                   \x20   packages:\n\
                   \x20     linux-64:\n\
                   \x20     - conda: https://prefix.dev/conda-forge/linux-64/python_abi-3.12-8_cp312.conda\n\
                    packages: []\n";
        let error = env_python_tags_from_lock(lock, "linux-64")
            .expect_err("an unmeasurable key field must refuse");
        assert!(error.to_string().contains("headless"), "{error}");
    }

    /// AN ENVIRONMENT ON ANOTHER PLATFORM IS NOT AN UNMEASURABLE ONE, and it
    /// took a real run to separate them: job 6058035 drove the producer at the
    /// canonical `pixi.lock.MDA-6054364.cert` and it REFUSED with
    /// `1 environment(s) lock no `python` on linux-64 … : jetson` — because
    /// `jetson`'s packages sit under the lock-v7 platform key whose `subdir` is
    /// `linux-aarch64`. Refusing a whole 27-environment lock over an
    /// environment that is not on this platform is the producer being wrong,
    /// not the lock. This fixture is that shape in miniature.
    #[test]
    fn an_environment_absent_on_this_subdir_is_not_a_missing_interpreter() {
        let lock = "platforms:\n\
                   - name: p1\n\
                   \x20 subdir: linux-64\n\
                   - name: p5\n\
                   \x20 subdir: linux-aarch64\n\
                    environments:\n\
                   \x20 gpu:\n\
                   \x20   packages:\n\
                   \x20     p1:\n\
                   \x20     - conda: https://prefix.dev/conda-forge/linux-64/python-3.12.0-hab00c5b_0_cpython.conda\n\
                   \x20 jetson:\n\
                   \x20   packages:\n\
                   \x20     p5:\n\
                   \x20     - conda: https://prefix.dev/conda-forge/linux-aarch64/python-3.10.20-h4f76b5d_1_cpython.conda\n\
                    packages: []\n";
        let set = env_python_tags_from_lock(lock, "linux-64")
            .expect("an off-platform environment must not refuse the lock");
        assert_eq!(set.tags.len(), 1);
        assert_eq!(set.tags[0].env, "gpu");
        assert_eq!(set.absent, vec!["jetson".to_string()]);

        // …and asking for it BY NAME still refuses, naming which absence it is.
        let dir = scratch("aarch64");
        let path = dir.join("pixi.lock");
        std::fs::write(&path, lock).unwrap();
        let error = tag_rows(&TagArgs::Lock {
            lock: path,
            subdir: "linux-64".to_string(),
            env: Some("jetson".to_string()),
        })
        .expect_err("an off-platform environment has no tag on this subdir");
        assert!(
            error.to_string().contains("locks nothing on linux-64"),
            "the refusal must say WHICH absence it is: {error}"
        );

        // The same lock read on the OTHER subdir flips which is which.
        let other = env_python_tags_from_lock(lock, "linux-aarch64").unwrap();
        assert_eq!(other.tags.len(), 1);
        assert_eq!(other.tags[0].python_tag, "cp310");
        assert_eq!(other.absent, vec!["gpu".to_string()]);
    }

    #[test]
    fn an_environment_the_lock_does_not_have_is_refused_with_the_list() {
        let dir = scratch("lock");
        let lock = dir.join("pixi.lock");
        std::fs::write(&lock, two_env_lock()).unwrap();
        let error = tag_rows(&TagArgs::Lock {
            lock,
            subdir: "linux-64".to_string(),
            env: Some("nope".to_string()),
        })
        .expect_err("an unknown environment must refuse");
        let text = error.to_string();
        assert!(text.contains("nope"), "{text}");
        assert!(text.contains("gpu"), "the refusal must name what IS there: {text}");
    }

    /// THE CARRIED-VALUE PATH, REFUSED WITH ITS OWN EXIT CODE AND ITS OWN ROW.
    #[test]
    fn a_literal_tag_beside_the_lock_is_refused_as_the_carried_value() {
        let argv: Vec<String> = [
            "--probe",
            "--uv-version",
            "0.9.5",
            "--python-tag",
            "cp310",
            "--python-tag-from-lock",
            "/nowhere/pixi.lock",
            "--env",
            "gpu",
            "--backend",
            "setuptools==84.0.0",
            "--pythonhashseed",
            "0",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let error = parse_args(&argv).expect_err("both tag sources must refuse");
        assert!(
            error.downcast_ref::<CarriedTagRefusal>().is_some(),
            "the carried-value refusal must be its own type, not an anonymous \
             usage error: {error}"
        );
        // …and it reaches the caller as a code a wrapper can branch on, never
        // as the same 1 a typo produces.
        assert_eq!(key_main(&argv).expect("the refusal is a code, not an Err"), CARRIED_TAG_EXIT);
        assert_ne!(CARRIED_TAG_EXIT, REVISION_REFUSED_EXIT);
    }

    #[test]
    fn the_producer_verb_takes_no_literal_tag_at_all() {
        let argv: Vec<String> = ["--lock", "/p.lock", "--python-tag", "cp310"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let error = parse_tag_args(&argv).expect_err("the producer must take no literal");
        assert!(error.to_string().contains("carried value"), "{error}");

        assert!(
            parse_tag_args(&["--lock".to_string(), "/p.lock".to_string(), "--prefix".to_string(), "/p".to_string()])
                .is_err(),
            "a lock and a prefix are two measurements and must not both be given"
        );
        assert!(
            parse_tag_args(&["--prefix".to_string(), "/p".to_string()]).is_err(),
            "--prefix without --env names nothing"
        );
    }

    #[test]
    fn a_lock_derived_tag_reaches_the_key_and_prints_its_row() {
        let dir = scratch("lock");
        let lock = dir.join("pixi.lock");
        std::fs::write(&lock, two_env_lock()).unwrap();
        let argv: Vec<String> = [
            "--probe",
            "--uv-version",
            "0.9.5",
            "--python-tag-from-lock",
            lock.to_str().unwrap(),
            "--env",
            "legacy",
            "--backend",
            "setuptools==84.0.0",
            "--pythonhashseed",
            "0",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let args = parse_args(&argv).expect("the derived form parses");
        let (facts, measured) = args.resolve_facts().expect("the lock yields the tag");
        assert_eq!(facts.python_tag, "cp38");
        let measured = measured.expect("a derived tag carries its row");
        assert_eq!(measured.env, "legacy");
        assert_eq!(measured.source, TagSource::Lock);
    }

    #[test]
    fn an_installed_prefix_is_read_from_its_lib_directory_and_never_run() {
        let dir = scratch("prefix");
        std::fs::create_dir_all(dir.join("lib/python3.11/site-packages")).unwrap();
        std::fs::create_dir_all(dir.join("lib/pythonqt")).unwrap();
        let row = env_python_tag_from_prefix("gpu", &dir).expect("one interpreter");
        assert_eq!(row.python_tag, "cp311");
        assert_eq!(row.source, TagSource::Interpreter);
        assert_eq!(
            row.row(),
            "### SDIST_META_KEY env=gpu python_tag=cp311 source=interpreter python_version=3.11"
        );

        let empty = scratch("empty");
        std::fs::create_dir_all(empty.join("lib")).unwrap();
        assert!(
            env_python_tag_from_prefix("gpu", &empty).is_err(),
            "a prefix whose key field cannot be measured must refuse"
        );
    }
}
