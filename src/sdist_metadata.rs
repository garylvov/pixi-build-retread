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

/// What the verb was asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub facts: ArmFacts,
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

pub fn parse_args(args: &[String]) -> anyhow::Result<Args> {
    let mut uv_version: Option<String> = None;
    let mut python_tag: Option<String> = None;
    let mut backend: Option<String> = None;
    let mut pythonhashseed: Option<String> = None;
    let mut store_root: Option<PathBuf> = None;
    let mut revision_http: Option<PathBuf> = None;
    let mut probe = false;
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
            "--backend" => backend = Some(value()?),
            "--pythonhashseed" => pythonhashseed = Some(value()?),
            "--store-root" => store_root = Some(PathBuf::from(value()?)),
            other => anyhow::bail!(
                "sdist-meta-key: unknown arg {other}; expected (--revision-http <file> | --probe) \
                 --uv-version --python-tag --backend --pythonhashseed [--store-root]"
            ),
        }
    }
    // `--sdist-sha256` is GONE rather than accepted-and-ignored. A caller still
    // passing it is a caller that has not been taught the new rule, and a silent
    // acceptance would key it on something it did not intend.
    let missing = |name: &str| anyhow::anyhow!("sdist-meta-key: --{name} is required");
    let facts = ArmFacts {
        uv_version: uv_version.ok_or_else(|| missing("uv-version"))?,
        python_tag: python_tag.ok_or_else(|| missing("python-tag"))?,
        backend: backend.ok_or_else(|| missing("backend"))?,
        pythonhashseed: pythonhashseed.ok_or_else(|| missing("pythonhashseed"))?,
    };
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
        facts,
        revision_http,
        store_root,
    })
}

/// Print the key and the paths derived from it, plus every field READ OUT OF
/// the file. The paths are printed because the shell halves would otherwise
/// each join the store's segments themselves; `rev=`, `url=` and `etag=` are
/// printed because they would otherwise each run `strings` over
/// `revision.http` themselves — which is the same second-implementation defect,
/// and the one that actually bit (SDIST-META-2, `admitted=0`).
pub fn run(args: &Args) -> anyhow::Result<i32> {
    let root = store_root_with(args.store_root.as_deref(), &|k| std::env::var(k).ok());
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
        uv_version: args.facts.uv_version.clone(),
        python_tag: args.facts.python_tag.clone(),
        backend: args.facts.backend.clone(),
        pythonhashseed: args.facts.pythonhashseed.clone(),
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
        assert_eq!(args.facts.python_tag, "cp310");

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
}
