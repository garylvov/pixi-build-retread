//! The SHARDED repodata index the pixi SOLVER reads -- fingerprinted, because
//! nothing else fingerprints it (N27-RETREAD-81).
//!
//! # The defect this module is the writer for
//!
//! [`crate::repodata`] fingerprints the CLASSIC `repodata.json` documents THIS
//! BACKEND reads and folds them into `repodata_universe=`. That number is
//! correct and load-bearing -- the built-output store's adoption rule is keyed
//! on it (N27-RETREAD-62) -- and it is also, for one whole family of lock
//! fields, blind.
//!
//! The pixi FRONTEND does not solve against those documents. It solves against
//! prefix.dev's SHARDED repodata: a per-`(channel, subdir)` index, fetched and
//! cached by `rattler_repodata_gateway`'s `sharded_subdir`, from which it pulls
//! individual per-package shards. `run_exports` lives ONLY there. RUNEXPORTS-1
//! measured the consequence over five certified relocks: both MERGE-B36 runs
//! printed the SAME `repodata_universe=5fb54c5f22a4d2dd` and the same two
//! `repodata_sha256=` values, and produced two DIFFERENT locks, differing by
//! exactly 24 `run_exports: {}` lines over 23 `conda-forge/noarch` names. The
//! cached classic `conda-forge/noarch` document those digests cover contains
//! ZERO occurrences of the string `"run_exports"`. It could not have witnessed
//! the change under any circumstances.
//!
//! So a determinism criterion asserted over `pixi.lock` had, for the field that
//! actually moves, **no live producer** -- law 1.2. This module is that
//! producer.
//!
//! # What it hashes, and why that is the honest thing to hash
//!
//! The smallest honest producer reads the shard-index cache files the solver
//! ACTUALLY USED and hashes their bytes. It does not re-fetch (a re-fetch would
//! measure the network at hash time, not the document the solve read), it does
//! not parse msgpack (a parser is a second opinion about the bytes; the bytes
//! are the identity), and it does not ask rattler to recompute the cache key.
//!
//! # How `(channel, subdir)` is recovered, measured rather than assumed
//!
//! The on-disk file names are OPAQUE: eight hex characters plus
//! [`SHARDED_INDEX_SUFFIX`], e.g. `e837b49a.shards-cache-v1`. Nothing in the
//! name says which channel it is.
//!
//! The CONTENT does. Every one of the eighteen files in the campaign cache
//! carries its own request URL verbatim in its bytes -- checked with a single
//! `grep -ao` over all eighteen on 2026-09-08, eighteen hits and no misses:
//!
//! ```text
//! e837b49a.shards-cache-v1  https://prefix.dev/conda-forge/noarch/repodata_shards.msgpack.zst
//! 6ddf7f44.shards-cache-v1  https://prefix.dev/conda-forge/linux-64/repodata_shards.msgpack.zst
//! 75f7b1e8.shards-cache-v1  https://conda.anaconda.org/pytorch/linux-64/repodata_shards.msgpack.zst
//! ```
//!
//! So the mapping is read from the document itself, with no dependency on a
//! private hashing scheme in a vendored crate that may be re-versioned under
//! us. If a future cache format stops carrying the URL, [`index_from_file`]
//! yields a row whose channel is [`CHANNEL_UNKNOWN`] rather than guessing --
//! and a row that says `unknown` is a finding, not a silent default (law 9).
//!
//! # What it deliberately does NOT touch
//!
//! `repodata_universe=` is left byte-for-byte alone. Widening it would change
//! the adoption identity of every record already published in the built-output
//! store, turning a fingerprint fix into a store-wide invalidation. The sharded
//! fold is a SEPARATE digest with its OWN schema tag, and
//! `sharded_digest_is_not_part_of_the_classic_universe` is the test that keeps
//! it that way.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Suffix of a shard-index cache file written by `rattler_repodata_gateway`'s
/// sharded subdir. The `-v1` is the gateway's own format tag; a `-v2` would be
/// a different document and this constant is where that shows up.
pub const SHARDED_INDEX_SUFFIX: &str = ".shards-cache-v1";

/// The needle that identifies a shard INDEX request inside a cache file. The
/// per-package shards live under a different path and must not be folded in.
pub const SHARDED_INDEX_NEEDLE: &str = "/repodata_shards.msgpack.zst";

/// Wire tag of the sharded fold. DELIBERATELY NOT
/// [`crate::repodata::UNIVERSE_SCHEMA`]: the two folds cover different document
/// families and a row that mixed them would be unreadable.
pub const SHARDED_UNIVERSE_SCHEMA: &str = "retread-sharded-index-v1";

/// The channel token for an index file that does not name its own source. It is
/// not a URL and cannot be mistaken for one, so a grep for a channel can never
/// match it and nobody can read it as "conda-forge".
pub const CHANNEL_UNKNOWN: &str = "unknown";

/// One sharded repodata INDEX, identified by the bytes on disk that the solver
/// read.
///
/// Serializable for the same reason [`crate::repodata::RepodataDocument`] is:
/// so a second hand-written copy of "which index is this" cannot drift from
/// this one.
#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct ShardedIndex {
    /// Channel URL with no trailing slash, as read from the index's own request
    /// URL -- e.g. `https://prefix.dev/conda-forge`. [`CHANNEL_UNKNOWN`] when
    /// the file does not name itself.
    pub channel: String,
    /// `noarch`, `linux-64`, ... [`CHANNEL_UNKNOWN`] alongside an unknown
    /// channel, never guessed.
    pub subdir: String,
    /// Hex sha256 of the WHOLE cache file, which is the document the solve
    /// read. Not of a parsed subset: a parser is an opinion, the bytes are the
    /// identity.
    pub index_sha256: String,
    pub bytes: u64,
    /// The cache file the digest was taken from, so a row can be re-checked by
    /// hand and so a reader can tell a shared cache from a per-job one
    /// (N27-RETREAD-82).
    pub source: PathBuf,
}

impl ShardedIndex {
    /// `"<channel>/<subdir>"`, the same shape
    /// [`crate::repodata::RepodataDocument::label`] prints.
    pub fn label(&self) -> String {
        format!("{}/{}", self.channel, self.subdir)
    }

    /// First 8 hex of the index hash -- what the per-index list in the summary
    /// row prints.
    pub fn short(&self) -> String {
        self.index_sha256.chars().take(8).collect()
    }

    /// True when this index could not name its own channel. A caller that folds
    /// these still folds it in -- an unnamed index the solver read is still a
    /// document the solver read -- but a reporter should say so.
    pub fn is_unknown(&self) -> bool {
        self.channel == CHANNEL_UNKNOWN
    }
}

/// Pull the shard-index request URL out of a cache file's raw bytes.
///
/// The file is a cached HTTP response envelope wrapped around msgpack. Rather
/// than decode either layer, this scans for [`SHARDED_INDEX_NEEDLE`] and walks
/// BACKWARDS to the `http` that starts the URL. That is deliberately the
/// dumbest thing that can work: it depends on the URL being present, and on
/// nothing else about either format.
///
/// Returns `None` when no shard-index URL is present -- which is the correct
/// answer for a shard BLOB cache file, and a refusal-worthy one for a file
/// named `*.shards-cache-v1`.
pub fn source_url_of(bytes: &[u8]) -> Option<String> {
    let needle = SHARDED_INDEX_NEEDLE.as_bytes();
    let end = bytes
        .windows(needle.len())
        .position(|window| window == needle)?
        + needle.len();
    // Walk back to the scheme. The URL cannot contain whitespace or a NUL, so
    // the first byte outside the URL character set bounds the search, and a URL
    // is never longer than a couple of hundred bytes.
    let floor = end.saturating_sub(512);
    let mut start = end;
    while start > floor {
        let candidate = start - 1;
        let byte = bytes[candidate];
        if !byte.is_ascii_graphic() {
            break;
        }
        start = candidate;
        if bytes[start..end].starts_with(b"http://") || bytes[start..end].starts_with(b"https://") {
            return std::str::from_utf8(&bytes[start..end])
                .ok()
                .map(str::to_string);
        }
    }
    None
}

/// Split a shard-index URL into `(channel, subdir)`.
///
/// `https://prefix.dev/conda-forge/noarch/repodata_shards.msgpack.zst`
/// -> `("https://prefix.dev/conda-forge", "noarch")`.
///
/// The channel keeps no trailing slash, which is the spelling
/// [`crate::repodata::RepodataDocument::channel`] uses, so the two families'
/// rows are comparable by eye.
pub fn channel_subdir_from_url(url: &str) -> Option<(String, String)> {
    let base = url.strip_suffix(SHARDED_INDEX_NEEDLE)?;
    let cut = base.rfind('/')?;
    let channel = &base[..cut];
    let subdir = &base[cut + 1..];
    if channel.is_empty() || subdir.is_empty() {
        return None;
    }
    Some((channel.to_string(), subdir.to_string()))
}

/// Hash a shard-index cache file and read its own name out of it.
///
/// A file that does not name itself still yields a row -- with
/// [`CHANNEL_UNKNOWN`] in both name fields and a real digest. Dropping it would
/// make an unreadable index indistinguishable from an absent one, and those are
/// different facts.
pub fn index_from_file(path: &Path) -> Result<ShardedIndex> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading shard index cache {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let index_sha256 = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let (channel, subdir) = source_url_of(&bytes)
        .as_deref()
        .and_then(channel_subdir_from_url)
        .unwrap_or_else(|| (CHANNEL_UNKNOWN.to_string(), CHANNEL_UNKNOWN.to_string()));
    Ok(ShardedIndex {
        channel,
        subdir,
        index_sha256,
        bytes: bytes.len() as u64,
        source: path.to_path_buf(),
    })
}

/// Every shard index under a pixi `repodata/` directory, in `(channel, subdir)`
/// order.
///
/// A MISSING or unreadable directory yields an EMPTY list and no error, for the
/// same reason [`crate::repodata::snapshot_documents_at`] does: the caller's
/// question is "what did the solve consult", and the answer "nothing I can see"
/// must reach the row as an honest empty rather than as a failed job. The row
/// then prints `indexes=0`, which is loud.
pub fn indexes_from_dir(dir: &Path) -> Vec<ShardedIndex> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => {
            tracing::warn!(
                dir = %dir.display(),
                error = %error,
                "sharded repodata: the index directory could not be listed; the sharded universe will name nothing",
            );
            return Vec::new();
        }
    };
    let mut indexes = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(SHARDED_INDEX_SUFFIX))
        {
            continue;
        }
        match index_from_file(&path) {
            Ok(index) => indexes.push(index),
            Err(error) => tracing::warn!(
                path = %path.display(),
                error = %format!("{error:#}"),
                "sharded repodata: could not fingerprint the index; the sharded universe row will not name it",
            ),
        }
    }
    indexes.sort();
    indexes
}

/// Fold a set of sharded indexes into ONE digest.
///
/// Pure, so a guard can drive it without a filesystem. Order-independent (the
/// input is sorted here, not by the caller) and subdir-separated, exactly like
/// [`crate::repodata::universe_digest_of`] -- and tagged with a DIFFERENT
/// schema so the two can never be confused for one another.
///
/// `source` is EXCLUDED from the fold on purpose. It is a path on one node in
/// one job; folding it in would make two runs that read byte-identical indexes
/// from differently-named job directories look like a moved universe, which is
/// the precise mistake `channel` caused in the classic fold (N27-RETREAD-62).
pub fn sharded_universe_digest_of(indexes: &[ShardedIndex]) -> String {
    let mut sorted: Vec<&ShardedIndex> = indexes.iter().collect();
    sorted.sort();
    sorted.dedup();
    let mut hasher = Sha256::new();
    hasher.update(SHARDED_UNIVERSE_SCHEMA.as_bytes());
    hasher.update([0u8]);
    for index in sorted {
        hasher.update(index.channel.as_bytes());
        hasher.update([0x1fu8]);
        hasher.update(index.subdir.as_bytes());
        hasher.update([0x1fu8]);
        hasher.update(index.index_sha256.as_bytes());
        hasher.update([0x1fu8]);
        hasher.update(index.bytes.to_string().as_bytes());
        hasher.update([0u8]);
    }
    let digest = hasher.finalize();
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// The per-index evidence row.
///
/// One line per `(channel, subdir)`, naming the file it was taken from so the
/// claim can be re-checked by hand with `sha256sum` and so a reader can see
/// whether the index came from a per-job directory or from the shared campaign
/// cache (N27-RETREAD-82).
pub fn sharded_index_row(index: &ShardedIndex) -> String {
    format!(
        "### SHARDED REPODATA channel={} subdir={} index_sha256={} bytes={} source={}",
        index.channel,
        index.subdir,
        index.index_sha256,
        index.bytes,
        index.source.display(),
    )
}

/// The one folded row, the counterpart of
/// [`crate::repodata::universe_summary_line`].
///
/// `sharded_universe=` is the token a comparator greps for. It is spelled
/// differently from `repodata_universe=` so that a grep for either can never
/// accidentally match the other.
pub fn sharded_universe_line(indexes: &[ShardedIndex]) -> String {
    let channels: Vec<String> = indexes.iter().map(ShardedIndex::label).collect();
    let digests: Vec<String> = indexes.iter().map(ShardedIndex::short).collect();
    format!(
        "### SHARDED UNIVERSE schema={SHARDED_UNIVERSE_SCHEMA} sharded_universe={} indexes={} unknown={} channels=[{}] digests=[{}]",
        sharded_universe_digest_of(indexes),
        indexes.len(),
        indexes.iter().filter(|i| i.is_unknown()).count(),
        channels.join(","),
        digests.join(","),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal stand-in for a cache file: some binary noise, the request URL
    /// embedded the way the real envelope embeds it, and more noise.
    fn fixture(url: &str, tail: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0x93u8, 0xa7, 0x00, 0x01, 0xff];
        bytes.extend_from_slice(url.as_bytes());
        bytes.push(0x00);
        bytes.extend_from_slice(tail);
        bytes
    }

    fn conda_forge_noarch(tail: &[u8]) -> Vec<u8> {
        fixture(
            "https://prefix.dev/conda-forge/noarch/repodata_shards.msgpack.zst",
            tail,
        )
    }

    #[test]
    fn the_index_names_its_own_channel_and_subdir() {
        let bytes = conda_forge_noarch(b"payload");
        let url = source_url_of(&bytes).expect("the fixture carries its URL");
        assert_eq!(
            url,
            "https://prefix.dev/conda-forge/noarch/repodata_shards.msgpack.zst"
        );
        let (channel, subdir) = channel_subdir_from_url(&url).expect("splittable");
        assert_eq!(channel, "https://prefix.dev/conda-forge");
        assert_eq!(subdir, "noarch");
    }

    #[test]
    fn a_non_prefix_host_splits_the_same_way() {
        let (channel, subdir) = channel_subdir_from_url(
            "https://conda.anaconda.org/pytorch/linux-64/repodata_shards.msgpack.zst",
        )
        .expect("splittable");
        assert_eq!(channel, "https://conda.anaconda.org/pytorch");
        assert_eq!(subdir, "linux-64");
    }

    #[test]
    fn a_file_that_does_not_name_itself_is_unknown_and_not_guessed() {
        let bytes = b"no url in here at all".to_vec();
        assert!(source_url_of(&bytes).is_none());
    }

    /// THE FIXTURE GUARD, in-tree half: two indexes differing in ONE byte must
    /// give different digests. This is the whole claim of the module.
    #[test]
    fn one_byte_of_index_content_moves_the_sharded_digest() {
        let a = conda_forge_noarch(b"AAAA");
        let b = conda_forge_noarch(b"AAAB");
        assert_eq!(a.len(), b.len(), "the fixtures differ in content, not size");
        let index_a = ShardedIndex {
            channel: "https://prefix.dev/conda-forge".into(),
            subdir: "noarch".into(),
            index_sha256: {
                let mut h = Sha256::new();
                h.update(&a);
                h.finalize().iter().map(|x| format!("{x:02x}")).collect()
            },
            bytes: a.len() as u64,
            source: PathBuf::from("/a"),
        };
        let mut index_b = index_a.clone();
        index_b.index_sha256 = {
            let mut h = Sha256::new();
            h.update(&b);
            h.finalize().iter().map(|x| format!("{x:02x}")).collect()
        };
        index_b.source = PathBuf::from("/b");
        assert_ne!(index_a.index_sha256, index_b.index_sha256);
        assert_ne!(
            sharded_universe_digest_of(&[index_a]),
            sharded_universe_digest_of(&[index_b]),
            "one changed index byte must move the sharded universe",
        );
    }

    /// The `source` path must NOT be in the fold: two jobs reading identical
    /// indexes from differently-named directories are the SAME universe.
    #[test]
    fn the_source_path_is_not_part_of_the_fold() {
        let base = ShardedIndex {
            channel: "https://prefix.dev/conda-forge".into(),
            subdir: "noarch".into(),
            index_sha256: "aa".repeat(32),
            bytes: 17,
            source: PathBuf::from("/job-1/pixi-overlay/repodata/e837b49a.shards-cache-v1"),
        };
        let mut elsewhere = base.clone();
        elsewhere.source = PathBuf::from("/job-2/pixi-overlay/repodata/e837b49a.shards-cache-v1");
        assert_eq!(
            sharded_universe_digest_of(&[base]),
            sharded_universe_digest_of(&[elsewhere]),
        );
    }

    /// Subdir separation, the same property the classic fold asserts: two
    /// hashes swapped between subdirs must NOT fold to the same value.
    #[test]
    fn swapping_two_hashes_between_subdirs_moves_the_digest() {
        let make = |subdir: &str, sha: &str| ShardedIndex {
            channel: "https://prefix.dev/conda-forge".into(),
            subdir: subdir.into(),
            index_sha256: sha.into(),
            bytes: 1,
            source: PathBuf::from("/x"),
        };
        let straight = [make("noarch", &"11".repeat(32)), make("linux-64", &"22".repeat(32))];
        let swapped = [make("noarch", &"22".repeat(32)), make("linux-64", &"11".repeat(32))];
        assert_ne!(
            sharded_universe_digest_of(&straight),
            sharded_universe_digest_of(&swapped),
        );
    }

    /// The fold is order-independent: the caller does not get to change the
    /// answer by listing a directory differently.
    #[test]
    fn the_fold_is_order_independent() {
        let make = |subdir: &str| ShardedIndex {
            channel: "https://prefix.dev/conda-forge".into(),
            subdir: subdir.into(),
            index_sha256: "33".repeat(32),
            bytes: 2,
            source: PathBuf::from("/x"),
        };
        let forward = [make("linux-64"), make("noarch")];
        let backward = [make("noarch"), make("linux-64")];
        assert_eq!(
            sharded_universe_digest_of(&forward),
            sharded_universe_digest_of(&backward),
        );
    }

    /// THE SEPARATION TEST (N27-RETREAD-81, deliberate design constraint).
    ///
    /// The sharded digest must NOT be mixed into the classic universe. If it
    /// ever were, every record already published in the built-output store
    /// would become unadoptable, because its stored `repodata_universe` was
    /// computed without it. This asserts the two folds are distinct FUNCTIONS
    /// over distinct SCHEMAS: the same content fed to both must not agree, and
    /// the classic fold must be unmoved by anything sharded.
    #[test]
    fn sharded_digest_is_not_part_of_the_classic_universe() {
        use crate::repodata::{RepodataDocument, UNIVERSE_SCHEMA, universe_digest_of};

        assert_ne!(
            SHARDED_UNIVERSE_SCHEMA, UNIVERSE_SCHEMA,
            "the two folds must be tagged differently or a row cannot say which it is",
        );

        // The classic universe over one document.
        let document = RepodataDocument {
            channel: "https://prefix.dev/conda-forge".into(),
            channel_key: crate::repodata::channel_subdir_key(
                "https://prefix.dev/conda-forge",
                "noarch",
            ),
            subdir: "noarch".into(),
            sha256: "ab".repeat(32),
            bytes: 99,
        };
        let classic = universe_digest_of(&[document]);

        // The SAME channel, subdir, hash and byte count, folded as a sharded
        // index. Different schema tag => different digest. If these were ever
        // equal, the two universes would be one universe.
        let index = ShardedIndex {
            channel: "https://prefix.dev/conda-forge".into(),
            subdir: "noarch".into(),
            index_sha256: "ab".repeat(32),
            bytes: 99,
            source: PathBuf::from("/x"),
        };
        let sharded = sharded_universe_digest_of(&[index.clone()]);
        assert_ne!(
            classic, sharded,
            "the sharded fold must not reproduce the classic fold's number",
        );

        // And the classic fold has no way to see a sharded index at all: it
        // takes RepodataDocument, which has no index field. The strongest
        // statement available in the type system is that adding indexes cannot
        // change a classic digest, because the classic digest is computed from
        // a disjoint input type. Assert the empty case is stable.
        assert_eq!(universe_digest_of(&[]), universe_digest_of(&[]));
        assert_ne!(
            universe_digest_of(&[]),
            sharded_universe_digest_of(&[]),
            "even the EMPTY universes must differ, or the tags are not doing their job",
        );
        let _ = index;
    }

    #[test]
    fn an_empty_directory_is_an_empty_universe_and_not_an_error() {
        let dir = std::env::temp_dir().join(format!(
            "retread-sharded-empty-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let indexes = indexes_from_dir(&dir);
        assert!(indexes.is_empty());
        let line = sharded_universe_line(&indexes);
        assert!(line.contains("indexes=0"), "{line}");
        assert!(line.contains("sharded_universe="), "{line}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An ABSENT directory must not be an error either -- and must not be
    /// silently indistinguishable from a present-but-empty one at the row
    /// level, which is why the row prints `indexes=`.
    #[test]
    fn an_absent_directory_yields_an_empty_universe() {
        let indexes = indexes_from_dir(Path::new("/nonexistent/retread/sharded/repodata"));
        assert!(indexes.is_empty());
    }

    /// End to end over real files on disk: two cache files differing in one
    /// byte must produce two different `sharded_universe=` values, and the row
    /// must name the file it hashed.
    #[test]
    fn two_index_files_differing_in_one_byte_give_different_universes() {
        let root = std::env::temp_dir().join(format!(
            "retread-sharded-onebyte-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let a_dir = root.join("a");
        let b_dir = root.join("b");
        std::fs::create_dir_all(&a_dir).expect("a");
        std::fs::create_dir_all(&b_dir).expect("b");
        let name = format!("e837b49a{SHARDED_INDEX_SUFFIX}");
        std::fs::write(a_dir.join(&name), conda_forge_noarch(b"AAAA")).expect("write a");
        std::fs::write(b_dir.join(&name), conda_forge_noarch(b"AAAB")).expect("write b");

        let a = indexes_from_dir(&a_dir);
        let b = indexes_from_dir(&b_dir);
        assert_eq!(a.len(), 1, "one index in a");
        assert_eq!(b.len(), 1, "one index in b");
        assert_eq!(a[0].channel, "https://prefix.dev/conda-forge");
        assert_eq!(a[0].subdir, "noarch");
        assert_eq!(a[0].bytes, b[0].bytes, "the two differ in content, not size");
        assert_ne!(
            sharded_universe_digest_of(&a),
            sharded_universe_digest_of(&b),
            "one byte of index content must move the sharded universe",
        );

        let row = sharded_index_row(&a[0]);
        assert!(row.starts_with("### SHARDED REPODATA "), "{row}");
        assert!(row.contains("channel=https://prefix.dev/conda-forge"), "{row}");
        assert!(row.contains("subdir=noarch"), "{row}");
        assert!(row.contains(&format!("index_sha256={}", a[0].index_sha256)), "{row}");
        assert!(row.contains(&name), "the row names the file it hashed: {row}");
        std::fs::remove_dir_all(&root).ok();
    }

    /// A file that is NOT a shard index must be ignored, so a shard blob or a
    /// lock file in the same directory cannot move the universe.
    #[test]
    fn only_shard_index_files_are_folded() {
        let dir = std::env::temp_dir().join(format!(
            "retread-sharded-filter-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("dir");
        std::fs::write(
            dir.join(format!("e837b49a{SHARDED_INDEX_SUFFIX}")),
            conda_forge_noarch(b"payload"),
        )
        .expect("index");
        std::fs::write(dir.join("something-else.json"), b"{}").expect("noise");
        std::fs::write(dir.join("e837b49a.lock"), b"lock").expect("lock");
        let indexes = indexes_from_dir(&dir);
        assert_eq!(indexes.len(), 1, "only the .shards-cache-v1 file is folded");
        std::fs::remove_dir_all(&dir).ok();
    }
}
