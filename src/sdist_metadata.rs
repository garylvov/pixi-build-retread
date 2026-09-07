//! SDIST-META-2: THE PERSISTENT PREPARED-SDIST-METADATA STORE, and the ONE
//! authority for its entry key.
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
//! # Why the key is ours and not uv's
//!
//! Read out of uv's `url_metadata`/`url_revision`: uv's own lookup validates
//! only `metadata.matches(name, version)`, and its revision is a function of
//! the sdist URL + sha256 + ETag. It contains NOTHING about the build inputs —
//! not the interpreter, not the PEP 517 backend, not `PYTHONHASHSEED`. That is
//! adequate inside one job and is exactly the hole a cross-job store must
//! close, so [`entry_key`] adds what uv omits rather than copying uv's key.
//!
//! The fifth field is there by MEASUREMENT, not caution. SDIST-META-1 found
//! gym 0.26.2's two revisions in the live shared cache at the same length
//! (1599 B) with different md5s, the requirements inside `extra == 'testing'`
//! permuted — a set-iteration order, i.e. `PYTHONHASHSEED`. Five of the six
//! sdists in the arm are byte-identical across their revisions; exactly one is
//! seed-sensitive, and with the seed pinned it is stable. An entry built under
//! a different seed is a DIFFERENT ENTRY.
//!
//! # Why the key lives in the product and not in the shell
//!
//! The store's writer is a post-lock harvester and its reader is the scoper's
//! seeder — two shell halves that must derive the SAME key or the store
//! silently never hits. `retread sdist-meta-key` is the one derivation both
//! call. A shell that folded `sha256sum` its own way would be a second
//! implementation of the key, and the first time the two disagreed the
//! disagreement would read as a cold cache rather than as a defect.
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

/// The generation segment. `reap_marker_store` walks EVERY generation it finds,
/// so a bump here orphans nothing.
pub const CACHE_VERSION: &str = "v1";

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
/// announced.
pub const KEY_SCHEMA: &str = "retread-sdist-metadata-cache-v1";

/// The rendered key's prefix, so a directory name is self-identifying in a
/// census row.
pub const KEY_PREFIX: &str = "sdm-";

/// How long an unreferenced entry may sit before the reaper QUARANTINES it, in
/// days. 14, the same horizon as the other five stores, on the standing
/// argument that housekeeping horizons which differ for no stated reason are
/// separate things to get wrong. `0` disables the reaper.
pub const DEFAULT_MAX_AGE_DAYS: u64 = 14;

/// The five facts an entry is identified by. Named fields rather than a tuple
/// because the ORDER is load-bearing — two fields swapped is a key that still
/// looks well-formed and matches nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyInputs {
    /// The sdist artefact's own sha256, read from `revision.http`'s digest.
    /// Name+version are NOT sufficient: that is precisely what uv validates
    /// and precisely the hole a persistent store must close.
    pub sdist_sha256: String,
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
            ("sdist_sha256", self.sdist_sha256.as_str()),
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

/// `<root>/sdist-metadata/v1` — the generation directory the entries sit in,
/// and the directory the harvester and the seeder both address.
pub fn generation_dir(store_root: &Path) -> PathBuf {
    store_root.join(CACHE_NAMESPACE).join(CACHE_VERSION)
}

/// `<root>/sdist-metadata/v1/<key>`.
pub fn entry_dir(store_root: &Path, key: &str) -> PathBuf {
    generation_dir(store_root).join(key)
}

// ── `retread sdist-meta-key` ────────────────────────────────────────────────

/// What the verb was asked. Every field is an ARGUMENT: the harvester and the
/// seeder must be able to derive a key for inputs that are not their own
/// process's environment (the seeder asks "what key would THIS arm want?"
/// while standing outside the build), so an environment lookup here would be
/// the wrong shape as well as against the standing rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub inputs: KeyInputs,
    /// `None` means the persistent root the product itself would resolve.
    pub store_root: Option<PathBuf>,
}

pub fn parse_args(args: &[String]) -> anyhow::Result<Args> {
    let mut sdist_sha256: Option<String> = None;
    let mut uv_version: Option<String> = None;
    let mut python_tag: Option<String> = None;
    let mut backend: Option<String> = None;
    let mut pythonhashseed: Option<String> = None;
    let mut store_root: Option<PathBuf> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || -> anyhow::Result<String> {
            it.next()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("sdist-meta-key: {arg} requires a value"))
        };
        match arg.as_str() {
            "--sdist-sha256" => sdist_sha256 = Some(value()?),
            "--uv-version" => uv_version = Some(value()?),
            "--python-tag" => python_tag = Some(value()?),
            "--backend" => backend = Some(value()?),
            "--pythonhashseed" => pythonhashseed = Some(value()?),
            "--store-root" => store_root = Some(PathBuf::from(value()?)),
            other => anyhow::bail!(
                "sdist-meta-key: unknown arg {other}; expected --sdist-sha256 --uv-version \
                 --python-tag --backend --pythonhashseed [--store-root]"
            ),
        }
    }
    let missing = |name: &str| anyhow::anyhow!("sdist-meta-key: --{name} is required");
    let inputs = KeyInputs {
        sdist_sha256: sdist_sha256.ok_or_else(|| missing("sdist-sha256"))?,
        uv_version: uv_version.ok_or_else(|| missing("uv-version"))?,
        python_tag: python_tag.ok_or_else(|| missing("python-tag"))?,
        backend: backend.ok_or_else(|| missing("backend"))?,
        pythonhashseed: pythonhashseed.ok_or_else(|| missing("pythonhashseed"))?,
    };
    inputs.validate()?;
    Ok(Args { inputs, store_root })
}

/// Print the key and the two paths derived from it. The PATHS are printed
/// because the shell halves would otherwise each join the segments themselves,
/// which is the same second-implementation defect the key itself avoids.
pub fn run(args: &Args) -> anyhow::Result<i32> {
    let key = entry_key(&args.inputs);
    let root = store_root_with(args.store_root.as_deref(), &|k| std::env::var(k).ok());
    println!(
        "### SDIST-META KEY key={key} version={CACHE_VERSION} store={} entry={}",
        generation_dir(&root).display(),
        entry_dir(&root, &key).display(),
    );
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> KeyInputs {
        KeyInputs {
            sdist_sha256: "6fd3fa41a68148e4a752".to_string(),
            uv_version: "0.9.5".to_string(),
            python_tag: "cp310".to_string(),
            backend: "setuptools==84.0.0".to_string(),
            pythonhashseed: "0".to_string(),
        }
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
            ("sdist_sha256", |i| i.sdist_sha256.push('x')),
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
    /// store nobody writes to.
    #[test]
    fn the_store_paths_are_root_namespace_version_key() {
        let root = Path::new("/tmp/root");
        let key = entry_key(&inputs());
        assert_eq!(
            entry_dir(root, &key),
            root.join("sdist-metadata").join("v1").join(&key)
        );
        assert_eq!(generation_dir(root), root.join("sdist-metadata").join("v1"));
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
    fn parse_requires_all_five_fields_and_accepts_a_named_root() {
        let argv: Vec<String> = [
            "--sdist-sha256",
            "abc",
            "--uv-version",
            "0.9.5",
            "--python-tag",
            "cp310",
            "--backend",
            "setuptools==84.0.0",
            "--pythonhashseed",
            "0",
            "--store-root",
            "/named",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let args = parse_args(&argv).expect("full argv parses");
        assert_eq!(args.store_root, Some(PathBuf::from("/named")));
        assert_eq!(args.inputs.python_tag, "cp310");

        let short: Vec<String> = argv[..8].to_vec();
        let error = parse_args(&short).expect_err("a missing field must refuse");
        assert!(error.to_string().contains("pythonhashseed"), "{error}");

        let unknown = vec!["--seed".to_string(), "0".to_string()];
        assert!(parse_args(&unknown).is_err(), "unknown args must refuse");
    }
}
