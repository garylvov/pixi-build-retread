//! Cross-pass record of the resolution inputs behind one advertised output
//! identity.
//!
//! `conda/outputs` advertises a content-addressed build string computed from
//! `courier_inputs_hash`, whose `config_fingerprint` folds
//! `workspace_solve_fingerprint` -- and that fingerprint reads the *on-disk*
//! lock files of co-activated sibling packs. In a cold multi-pack relock a
//! sibling writes its lock BETWEEN this pack's metadata pass and its build
//! pass, so `conda/build_v1` recomputes a different hash than the one it was
//! asked to build and the identity gate refuses ("0 exact matches for
//! advertised output ... identity differs").
//!
//! The in-memory `prepared_builds` handoff cannot cover this: it is keyed by
//! the metadata pass's `work_directory`, which pixi changes for the build
//! phase, and it dies with the process. This store is its durable companion:
//! the metadata pass persists the exact fingerprint it resolved under, keyed
//! by the identity it advertised, and the build pass resolves under that same
//! fingerprint instead of re-reading whatever siblings happen to exist now.
//!
//! Records are content-addressed by the identity they describe, so two
//! concurrent writers write byte-identical bytes; publication is still
//! temp-file + rename so no reader ever sees a partial file.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Bumped whenever the recorded field set changes meaning. A record with a
/// different schema is ignored, which degrades to today's recompute.
pub(crate) const SCHEMA: u32 = 3;

/// Short digest of the RELAX RULE in force, folded into every record address.
///
/// `config.relax` is what decides which metadata relaxations the advertising
/// pass applied, and it is already part of the build string's own inputs hash
/// (`RetreadLock::compute_inputs_hash_for_target` takes `format!("{:?}",
/// config.relax)`). Folding the same spelling here keeps the record's address
/// a function of the same rule, so a record written under one relax policy can
/// never answer a request resolved under another.
pub(crate) fn relax_digest(config: &crate::config::RetreadConfig) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(format!("{:?}", config.relax).as_bytes());
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// The resolution inputs one advertised output identity was computed from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct AdvertisedIdentityRecord {
    pub schema: u32,
    pub name: String,
    pub version: String,
    pub build: String,
    pub subdir: String,
    /// `ResolutionTarget::resolution_identity()` of the advertising pass.
    pub target_identity: String,
    pub python_version: String,
    /// The exact `workspace_solve_fingerprint` string folded into the
    /// `config_fingerprint` that produced `build`.
    pub workspace_fp: String,
    /// The exact `run_dependencies.depends` this pass sent to pixi in
    /// `conda/outputs`, rendered as `name spec` lines.
    ///
    /// Pixi solved the consuming environment against THIS list; the build pass
    /// re-deriving a different one (a missing `python_abi`, an extra
    /// auto-routed name) is drift, not a correction, so the record -- not the
    /// re-derivation -- decides what the package emits.
    #[serde(default)]
    pub run_depends: Vec<String>,
    /// The exact `run_dependencies.constraints` (conda `constrains`) advertised
    /// alongside `run_depends`. Re-derived on the build pass today
    /// (`bundle_emitted_constrains`), so it drifts for the same reasons.
    #[serde(default)]
    pub run_constrains: Vec<String>,
}

impl AdvertisedIdentityRecord {
    /// A record is usable only when it describes the very output the build
    /// request names, resolved for the very same target. Anything else is a
    /// stale or foreign record and must be ignored rather than trusted.
    pub(crate) fn describes(
        &self,
        name: &str,
        version: Option<&str>,
        subdir: &str,
        target_identity: &str,
        python_version: &str,
    ) -> bool {
        self.schema == SCHEMA
            && self.name == name
            && self.subdir == subdir
            && self.target_identity == target_identity
            && self.python_version == python_version
            && version.is_none_or(|version| self.version == version)
    }
}

/// Path of the record for one advertised identity.
///
/// The build string is already content-addressed over the resolution inputs;
/// the digest additionally separates packs (source dir), outputs (name) and
/// subdirs so two packs can never collide on one file.
///
/// p6i: the digest also folds [`SCHEMA`] and the RELAX RULE
/// ([`relax_digest`]). Without the schema, a record written by an older field
/// set occupies the address a newer one computes and is only rejected after
/// being read (`describes`), which costs a read and leaves the old file to be
/// re-read forever. Without the relax rule, changing `retread-relax` -- which
/// changes what the metadata pass ADVERTISED -- leaves the old record at the
/// same address, and `describes` does not compare it, so a stale relaxation
/// record is adopted as authoritative for what the package emits.
pub(crate) fn record_path(
    cache_dir: &Path,
    source_dir: &Path,
    name: &str,
    subdir: &str,
    build: &str,
    relax_digest: &str,
) -> PathBuf {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let schema = SCHEMA.to_string();
    for part in [
        schema.as_str(),
        source_dir.to_string_lossy().as_ref(),
        name,
        subdir,
        build,
        relax_digest,
    ] {
        hasher.update(part.as_bytes());
        hasher.update(b"\0");
    }
    let digest = hasher.finalize();
    let hex: String = digest.iter().take(16).map(|b| format!("{b:02x}")).collect();
    cache_dir
        .join("retread-advertised-identity")
        .join(format!("{hex}.json"))
}

/// Publish the record for one advertised identity. Best effort: a store that
/// cannot be written must never fail the RPC, it only costs the build pass its
/// reuse and returns it to today's recompute-and-refuse behaviour.
pub(crate) async fn write_record(
    cache_dir: &Path,
    source_dir: &Path,
    record: &AdvertisedIdentityRecord,
    relax_digest: &str,
) {
    let path = record_path(
        cache_dir,
        source_dir,
        &record.name,
        &record.subdir,
        &record.build,
        relax_digest,
    );
    let Some(parent) = path.parent() else {
        return;
    };
    if let Err(error) = tokio::fs::create_dir_all(parent).await {
        tracing::debug!(error = %error, path = %parent.display(), "advertised identity: could not create store dir");
        return;
    }
    let bytes = match serde_json::to_vec(record) {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::debug!(error = %error, "advertised identity: serialize failed");
            return;
        }
    };
    // Per-process temp then rename: several backend children advertise
    // concurrently, and a reader must never observe a partial file.
    let tmp_path = path.with_extension(format!("json.tmp.{}", std::process::id()));
    if let Err(error) = tokio::fs::write(&tmp_path, &bytes).await {
        tracing::debug!(error = %error, path = %tmp_path.display(), "advertised identity: write failed");
        return;
    }
    if let Err(error) = tokio::fs::rename(&tmp_path, &path).await {
        tracing::debug!(error = %error, path = %path.display(), "advertised identity: rename failed");
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return;
    }
    tracing::debug!(
        path = %path.display(),
        output = %record.name,
        build = %record.build,
        "advertised identity: recorded the resolution inputs behind this identity",
    );
}

/// Load the record for an advertised identity, or `None` when there is none
/// (missing, unreadable, stale schema, or describing something else).
pub(crate) async fn load_record(
    cache_dir: &Path,
    source_dir: &Path,
    name: &str,
    version: Option<&str>,
    subdir: &str,
    build: &str,
    target_identity: &str,
    python_version: &str,
    relax_digest: &str,
) -> Option<AdvertisedIdentityRecord> {
    let path = record_path(cache_dir, source_dir, name, subdir, build, relax_digest);
    let bytes = tokio::fs::read(&path).await.ok()?;
    let record: AdvertisedIdentityRecord = serde_json::from_slice(&bytes).ok()?;
    let record = record
        .describes(name, version, subdir, target_identity, python_version)
        .then_some(record)?;
    // INFO, not DEBUG: whether a build request reproduced its advertised
    // identity from the record or fell back to recompute-and-refuse is the one
    // fact a post-hoc audit of a cold relock needs, and a DEBUG-only line is
    // absent from every default-level backend log.
    tracing::info!(
        path = %path.display(),
        output = %record.name,
        build = %record.build,
        "advertised identity: loaded the recorded resolution inputs for this build request",
    );
    Some(record)
}

/// The live workspace solve fingerprint, read from whatever sibling locks are
/// on disk RIGHT NOW.
///
/// PRIVATE ON PURPOSE. This is the function whose result drifts between the
/// metadata pass and the build pass, and every identity refusal this module
/// exists to fix was a gate calling it directly. It is unreachable from
/// `handler::mod` (and from every other module), so the only fingerprint a
/// gate can obtain is [`EffectiveWorkspaceFp`], which has folded the record in.
fn live_workspace_fp(
    workspace_manifest: Option<&crate::workspace::WorkspaceManifest>,
    workspace_dir: &Path,
    source_dir: &Path,
    target: &crate::pypi::ResolutionTarget,
) -> String {
    workspace_manifest
        .map(|m| super::workspace_solve_fingerprint(m, workspace_dir, source_dir, target))
        .unwrap_or_default()
}

/// The ONE workspace solve fingerprint a build request derives identity from.
///
/// Structural enforcement of "identity is a function of the inputs the
/// ADVERTISING pass saw, never of which pass is running": [`live_workspace_fp`]
/// is private to this module, so [`resolve`](Self::resolve) is the only
/// constructor in the crate. Every identity gate in `conda/build_v1` -- the
/// WS-B replay hash, the cold candidate hash, the lock-parity recovery
/// fingerprint, `validate_advertised_courier_build`, and `build_one`'s packing
/// fingerprint -- takes this type by value or reference, so a gate added later
/// physically cannot bypass the record: there is no other way to make one, and
/// the compiler, not a code-review convention, is what keeps stragglers out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EffectiveWorkspaceFp(String);

impl EffectiveWorkspaceFp {
    /// Resolve the fingerprint for this build request: the recorded one when
    /// the advertising pass left a record for the requested identity, the live
    /// one (and today's drift gate) when it did not.
    pub(crate) fn resolve(
        record: Option<&AdvertisedIdentityRecord>,
        workspace_manifest: Option<&crate::workspace::WorkspaceManifest>,
        workspace_dir: &Path,
        source_dir: &Path,
        target: &crate::pypi::ResolutionTarget,
    ) -> Self {
        Self(workspace_fp_for_build(
            record,
            live_workspace_fp(workspace_manifest, workspace_dir, source_dir, target),
        ))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

/// The workspace solve fingerprint the build pass must resolve under.
///
/// With a record: the fingerprint the advertised identity was computed from,
/// so the identity reproduces no matter which sibling packs have written their
/// locks since. Without one: today's freshly computed fingerprint, and today's
/// drift gate behaviour.
pub(crate) fn workspace_fp_for_build(
    record: Option<&AdvertisedIdentityRecord>,
    computed: String,
) -> String {
    match record {
        Some(record) if record.workspace_fp != computed => {
            tracing::info!(
                output = %record.name,
                build = %record.build,
                "advertised identity: resolving under the recorded metadata-pass workspace \
                 fingerprint; sibling packs wrote locks between the two passes",
            );
            record.workspace_fp.clone()
        }
        Some(record) => record.workspace_fp.clone(),
        None => computed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Any stable relax digest: these tests are about the record, not the rule.
    const TEST_RELAX: &str = "relax-test-digest";

    fn tempdir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "retread-advertised-identity-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn record() -> AdvertisedIdentityRecord {
        AdvertisedIdentityRecord {
            schema: SCHEMA,
            name: "protomotions-deps-pack".to_string(),
            version: "3.1".to_string(),
            build: "py311_h3c24f86882_loose_0".to_string(),
            subdir: "linux-64".to_string(),
            target_identity: "linux-64-cuda-12-glibc-2-35".to_string(),
            python_version: "3.11".to_string(),
            workspace_fp: "metadata-pass-fp".to_string(),
            run_depends: vec![
                "python 3.11.*".to_string(),
                "python_abi 3.11.* *_cp311".to_string(),
            ],
            run_constrains: vec!["numpy >=1.26".to_string()],
        }
    }

    /// Guard (turn 14): the advertised OUTPUT round-trips through the store --
    /// both halves of it. `conda/build_v1` emits the recorded `depends` AND the
    /// recorded `constrains`; dropping either from the record silently returns
    /// that half to the build pass's re-derivation, which is the drift this
    /// record exists to eliminate.
    #[tokio::test]
    async fn a_record_round_trips_the_advertised_depends_and_constrains() {
        let dir = tempdir("advertised-output").canonicalize().unwrap();
        let cache = dir.join("cache");
        let source = dir.join("pack");
        let record = record();
        write_record(&cache, &source, &record, TEST_RELAX).await;
        let loaded = load_record(
            &cache,
            &source,
            &record.name,
            Some(&record.version),
            &record.subdir,
            &record.build,
            &record.target_identity,
            &record.python_version,
            TEST_RELAX,
        )
        .await
        .expect("record must load");
        assert_eq!(loaded.run_depends, record.run_depends);
        assert_eq!(loaded.run_constrains, record.run_constrains);
        assert_eq!(loaded, record);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Guard: the record survives pixi moving `work_directory` between the
    /// metadata pass and the build pass, and survives the two passes running
    /// from different current directories. `record_path` is keyed by
    /// (cache_dir, source_dir, name, subdir, build) ONLY -- if a work dir, a
    /// CWD, or any per-phase path is ever folded in, the build pass looks in a
    /// place the metadata pass never wrote and this test fails.
    #[tokio::test]
    async fn a_record_written_from_one_work_dir_view_is_found_from_another() {
        let dir = tempdir("work-dir-view").canonicalize().unwrap();
        let cache = dir.join("cache");
        let source = dir.join("pack");
        let metadata_work_dir = dir.join(".pixi/bld/pack/metadata-phase");
        let build_work_dir = dir.join(".pixi/bld/pack/AbC123-build-phase");
        std::fs::create_dir_all(&metadata_work_dir).unwrap();
        std::fs::create_dir_all(&build_work_dir).unwrap();
        let record = record();

        // The path must live under the cache root and carry no trace of
        // either phase's work directory...
        let path = record_path(
            &cache,
            &source,
            &record.name,
            &record.subdir,
            &record.build,
            TEST_RELAX,
        );
        assert!(
            path.starts_with(cache.join("retread-advertised-identity")),
            "{path:?}"
        );
        let rendered = path.to_string_lossy().to_string();
        assert!(!rendered.contains("metadata-phase"), "{rendered}");
        assert!(!rendered.contains("AbC123-build-phase"), "{rendered}");
        assert!(metadata_work_dir.exists() && build_work_dir.exists());

        // ...and a record the metadata phase wrote must load for the build
        // phase, which passes a different `work_directory` for the same
        // (source_dir, name, subdir, build).
        write_record(&cache, &source, &record, TEST_RELAX).await;
        let loaded = load_record(
            &cache,
            &source,
            &record.name,
            Some(&record.version),
            &record.subdir,
            &record.build,
            &record.target_identity,
            &record.python_version,
            TEST_RELAX,
        )
        .await;
        assert_eq!(loaded.as_ref(), Some(&record));

        // Non-vacuous: a DIFFERENT source dir (another pack) must miss, so the
        // assertion above is not merely "any record loads".
        let other_source = dir.join("other-pack");
        assert!(
            load_record(
                &cache,
                &other_source,
                &record.name,
                Some(&record.version),
                &record.subdir,
                &record.build,
                &record.target_identity,
                &record.python_version,
                TEST_RELAX,
            )
            .await
            .is_none()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Guard (a): the metadata pass's record is retrievable by the exact
    /// identity it advertised.
    #[tokio::test]
    async fn record_round_trips_keyed_by_the_advertised_build_string() {
        let dir = tempdir("round-trip");
        let cache = dir.join("cache");
        let source = dir.join("pack");
        let record = record();
        write_record(&cache, &source, &record, TEST_RELAX).await;
        let loaded = load_record(
            &cache,
            &source,
            &record.name,
            Some(&record.version),
            &record.subdir,
            &record.build,
            &record.target_identity,
            &record.python_version,
            TEST_RELAX,
        )
        .await;
        assert_eq!(loaded.as_ref(), Some(&record));
    }

    /// Guard (b): a build pass whose own sibling fingerprint has since moved
    /// still resolves under the fingerprint the identity was advertised from.
    #[test]
    fn a_moved_sibling_fingerprint_does_not_override_the_record() {
        let record = record();
        assert_eq!(
            workspace_fp_for_build(Some(&record), "build-pass-fp-with-new-siblings".to_string()),
            "metadata-pass-fp",
        );
    }

    /// Guard (c): no record -> today's freshly computed fingerprint, and
    /// today's drift-gate behaviour.
    #[test]
    fn without_a_record_the_freshly_computed_fingerprint_stands() {
        assert_eq!(
            workspace_fp_for_build(None, "build-pass-fp".to_string()),
            "build-pass-fp",
        );
    }

    /// A record for another output/target is never trusted for this one.
    #[tokio::test]
    async fn a_record_for_another_target_is_ignored() {
        let dir = tempdir("foreign-record");
        let cache = dir.join("cache");
        let source = dir.join("pack");
        let record = record();
        write_record(&cache, &source, &record, TEST_RELAX).await;
        assert!(
            load_record(
                &cache,
                &source,
                &record.name,
                Some(&record.version),
                &record.subdir,
                &record.build,
                "linux-64-cuda-13-glibc-2-35",
                &record.python_version,
                TEST_RELAX,
            )
            .await
            .is_none()
        );
        assert!(
            load_record(
                &cache,
                &source,
                &record.name,
                Some("9.9"),
                &record.subdir,
                &record.build,
                &record.target_identity,
                &record.python_version,
                TEST_RELAX,
            )
            .await
            .is_none()
        );
        assert!(
            load_record(
                &cache,
                &source,
                &record.name,
                None,
                &record.subdir,
                "py311_h0a4ba3c452_loose_0",
                &record.target_identity,
                &record.python_version,
                TEST_RELAX,
            )
            .await
            .is_none()
        );
    }
}
