//! v2.0.0 courier: the committed install lock.
//!
//! Records every resolution decision the backend made so a consumer can
//! install the bundle's PyPI wheels at link time (via `retread install`,
//! invoked from the conda package's post-link script) WITHOUT re-running
//! the cascade and WITHOUT committing any wheels to git:
//!
//! - `built` wheels (git/path sources, on no index) ship inside the conda
//!   package under `share/retread/wheels/<filename>` and install from
//!   there.
//! - `index` wheels (isaacsim, nvidia, ... -- the multi-GB bulk) are NOT
//!   shipped; they install by fetching their recorded `url` at link time,
//!   on uv's fast hardlink path.
//!
//! The lock is small (KB), human-diffable, and committed next to the pack
//! manifest as `retread-<bundle>.lock.json`. It is the single source of
//! truth the installer reads.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// How a wheel reaches the consumer at install time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Origin {
    /// Shipped inside the conda package (built from a git/path source, on
    /// no index). Installed from `share/retread/wheels/<filename>`.
    Built,
    /// Fetched from `url` at link time (lives on a public/extra index).
    Index,
}

/// One wheel in the bundle's install set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockWheel {
    pub name: String,
    pub version: String,
    pub origin: Origin,
    /// Standardized wheel filename (the basename installed/shipped).
    pub filename: String,
    /// Upstream URL for `Origin::Index` wheels; `None` for `Built`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// sha256 of the wheel file (index-wheel verification; reproducibility).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

/// A conda run-dep retread routed to the conda side (a shared transitive
/// the consumer's conda solve provides). Carried so the courier conda
/// package can declare them as its own run-deps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CondaDep {
    pub name: String,
    pub spec: String,
}

/// The committed install lock for one bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetreadLock {
    pub schema: u32,
    pub retread_version: String,
    pub bundle: String,
    pub version: String,
    /// Target python (e.g. "3.11"); the wheel set is python-specific.
    pub python: String,
    /// sha256 over the canonicalized resolution inputs (entry specs +
    /// ordered index chain + relax policy + python + retread version).
    /// Reproducibility gate (req #5) AND the cold-solve replay key (req #4):
    /// `conda/outputs` replays this lock's emitted outputs (skipping the
    /// probe cascade) iff a freshly computed inputs hash matches this.
    #[serde(default)]
    pub inputs_hash: String,
    /// The PEP 508 requirements `retread install` hands to uv (the meta
    /// requirement that drives the closure -- typically the bundle entries
    /// pinned to their resolved versions). uv resolves these against the
    /// shipped find-links wheels + `index_urls`, into the active conda env
    /// (preferring conda-installed dists, so shared transitives stay conda).
    #[serde(default)]
    pub root_requirements: Vec<String>,
    /// Wheels to install at link time (built ship in pkg + index fetched).
    pub wheels: Vec<LockWheel>,
    /// Shared transitives routed to conda (the courier package's run-deps).
    pub conda_run_deps: Vec<CondaDep>,
    /// Index chain for fetching `Origin::Index` wheels + any prerelease
    /// deps, in priority order (entry indexes, then public PyPI).
    pub index_urls: Vec<String>,
    /// Prerelease pins (`name` -> `==X`) uv needs to opt those deps into
    /// prerelease resolution (it only honors them from direct reqs +
    /// overrides). Empty when the bundle pins no prereleases.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub prerelease: BTreeMap<String, String>,
}

pub const SCHEMA: u32 = 4;

/// Bump EMIT_EPOCH in the SAME commit as ANY change that can alter the bytes
/// retread emits for identical manifest inputs (relax/version-selection
/// algorithm; wheel-rewrite logic incl. RECORD/metadata/dist-info/layout;
/// auto-bundle selection or fetching; meta-wheel / courier package format;
/// conda recipe metadata; index-chain merge order; the compute_inputs_hash
/// domain prefix or folded fields). Do NOT bump for docs/tests/logging/
/// refactors with identical output, dep bumps that don't change emitted bytes,
/// or SCHEMA-only lock-format changes. When in doubt, bump it: a needless bump
/// costs one cold solve per pack (cheap, self-healing); a missed bump causes
/// silent stale-cache reuse for consumers (expensive, invisible).
/// SCHEMA = on-disk lock FORMAT; EMIT_EPOCH = emitted-output SEMANTICS for
/// identical inputs -- bump independently.
pub const EMIT_EPOCH: u32 = 1;

impl RetreadLock {
    /// File name for a bundle's lock next to the pack manifest.
    pub fn file_name(bundle: &str) -> String {
        format!("retread-{bundle}.lock.json")
    }

    /// Canonical inputs hash. EVERY producer (the courier staging that
    /// writes the lock) and replayer (`conda/outputs` deciding whether to
    /// skip the cascade) MUST call this -- never hand-roll the digest, or
    /// the two sides disagree and replay silently never fires (or fires
    /// stale). Entry specs are sorted so ordering is not significant; the
    /// index chain order IS significant (it is resolution priority).
    ///
    /// `config_fingerprint` folds in EVERY remaining resolution-affecting
    /// manifest input that is not already covered above -- overrides,
    /// name-map, drop-deps, conda-deps, auto-bundle, build-number, and the
    /// conda channel list. Without it, editing any of those (e.g. genesis's
    /// `retread-name-map`) would leave the hash unchanged and a stale,
    /// POISONED lock would replay. It is produced by the shared
    /// `courier::config_fingerprint` so producer and replayer always agree.
    ///
    /// The hash folds the emit epoch (+ the exact retread version when
    /// `retread-pin-version` is set). Routine retread upgrades that do NOT
    /// change emitted bytes for identical inputs only bump `EMIT_EPOCH` when
    /// the output semantics change, so existing committed locks replay without
    /// a cold re-solve on every upgrade.
    pub fn compute_inputs_hash(
        entry_specs: &[String],
        index_urls: &[String],
        relax: &str,
        python: &str,
        emit_epoch: u32,
        pin_version: Option<&str>,
        config_fingerprint: &str,
    ) -> String {
        use sha2::{Digest, Sha256};
        let mut sorted = entry_specs.to_vec();
        sorted.sort();
        let mut h = Sha256::new();
        h.update(b"retread-inputs-v5\n");
        for s in &sorted {
            h.update(s.as_bytes());
            h.update(b"\n");
        }
        h.update(b"--indexes--\n");
        for u in index_urls {
            h.update(u.as_bytes());
            h.update(b"\n");
        }
        h.update(b"--meta--\n");
        h.update(relax.as_bytes());
        h.update(b"\n");
        h.update(python.as_bytes());
        h.update(b"\n");
        h.update(b"--epoch--\n");
        h.update(emit_epoch.to_le_bytes());
        if let Some(v) = pin_version {
            h.update(b"--pinver--\n");
            h.update(v.as_bytes());
        }
        h.update(b"\n--config--\n");
        h.update(config_fingerprint.as_bytes());
        format!("{:x}", h.finalize())
    }

    /// Read a lock back (consumer side / cold-start replay).
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("reading lock {}", path.display()))?;
        serde_json::from_slice(&bytes).with_context(|| format!("parsing lock {}", path.display()))
    }

    /// Serialize to pretty JSON for committing.
    pub fn to_pretty_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("serializing retread lock")
    }

    /// Idempotency marker file (under `<prefix>/share/retread/`). The
    /// installer writes the lock's content hash here on success and
    /// no-ops when it already matches -- safe to re-run on every relink.
    pub fn marker_name(&self) -> String {
        format!("{}.installed", self.bundle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_roundtrips() {
        let lock = RetreadLock {
            schema: SCHEMA,
            retread_version: "2.0.0".into(),
            bundle: "isaac-pack".into(),
            version: "5.1.0".into(),
            python: "3.11".into(),
            inputs_hash: "deadbeef".into(),
            root_requirements: vec!["isaac-pack-pypi==5.1.0".into()],
            wheels: vec![
                LockWheel {
                    name: "isaaclab".into(),
                    version: "0.51.1".into(),
                    origin: Origin::Built,
                    filename: "isaaclab-0.51.1-py3-none-any.whl".into(),
                    url: None,
                    sha256: None,
                },
                LockWheel {
                    name: "isaacsim-core".into(),
                    version: "5.1.0.0".into(),
                    origin: Origin::Index,
                    filename: "isaacsim_core-5.1.0.0-cp311-...whl".into(),
                    url: Some("https://pypi.nvidia.com/isaacsim-core/...whl".into()),
                    sha256: Some("abc123".into()),
                },
            ],
            conda_run_deps: vec![CondaDep {
                name: "torchaudio".into(),
                spec: ">=2.7,<3".into(),
            }],
            index_urls: vec![
                "https://pypi.nvidia.com".into(),
                "https://pypi.org/simple/".into(),
            ],
            prerelease: BTreeMap::from([("gmpy2".into(), "==2.1.0a4".into())]),
        };
        let json = lock.to_pretty_json().unwrap();
        let back: RetreadLock = serde_json::from_str(&json).unwrap();
        assert_eq!(back.bundle, "isaac-pack");
        assert_eq!(back.wheels.len(), 2);
        assert_eq!(back.wheels[0].origin, Origin::Built);
        assert_eq!(back.wheels[1].origin, Origin::Index);
        assert_eq!(
            back.prerelease.get("gmpy2").map(String::as_str),
            Some("==2.1.0a4")
        );
        assert_eq!(
            RetreadLock::file_name(&back.bundle),
            "retread-isaac-pack.lock.json"
        );
        assert_eq!(back.inputs_hash, "deadbeef");
        assert_eq!(back.wheels[1].sha256.as_deref(), Some("abc123"));
    }

    #[test]
    fn inputs_hash_stable_and_order_sensitive() {
        let h1 = RetreadLock::compute_inputs_hash(
            &["b==1".into(), "a==2".into()],
            &[
                "https://pypi.nvidia.com".into(),
                "https://pypi.org/simple/".into(),
            ],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "cfg",
        );
        // entry order does NOT matter (sorted)
        let h2 = RetreadLock::compute_inputs_hash(
            &["a==2".into(), "b==1".into()],
            &[
                "https://pypi.nvidia.com".into(),
                "https://pypi.org/simple/".into(),
            ],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "cfg",
        );
        assert_eq!(h1, h2, "entry-spec order must not change the hash");
        // index order DOES matter (resolution priority)
        let h3 = RetreadLock::compute_inputs_hash(
            &["a==2".into(), "b==1".into()],
            &[
                "https://pypi.org/simple/".into(),
                "https://pypi.nvidia.com".into(),
            ],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "cfg",
        );
        assert_ne!(h1, h3, "index chain order must change the hash");
    }

    #[test]
    fn inputs_hash_covers_config_fingerprint() {
        // The config fingerprint folds in name-map / overrides / drop-deps /
        // conda-deps / build-number / auto-bundle / channels. Changing it
        // MUST change the hash, or a manifest edit replays a poisoned lock.
        let base = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "name-map=opencv-python:py-opencv",
        );
        let changed = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "name-map=opencv-python:opencv",
        );
        assert_ne!(
            base, changed,
            "config fingerprint change must change the inputs hash"
        );
    }

    #[test]
    fn inputs_hash_epoch_changes_hash() {
        // Bumping EMIT_EPOCH must invalidate any committed lock so a cold
        // re-solve is forced on output-semantics changes.
        let h1 = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "cfg",
        );
        let h2 = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            2,
            None,
            "cfg",
        );
        assert_ne!(h1, h2, "changing emit_epoch must change the hash");
    }

    #[test]
    fn inputs_hash_pin_version_differs_from_none() {
        // With retread-pin-version=true, Some("1.0.0") must differ from None.
        let without_pin = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "cfg",
        );
        let with_pin = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            Some("1.0.0"),
            "cfg",
        );
        assert_ne!(
            without_pin, with_pin,
            "pin_version=Some must differ from None"
        );
    }

    #[test]
    fn inputs_hash_pin_version_two_versions_differ_only_when_pinned() {
        // The whole point: two different retread versions produce the SAME hash
        // under pin_version=None (epoch-gated replay) but DIFFERENT hashes
        // under pin_version=Some (strict reproducibility).
        let none_v1 = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "cfg",
        );
        let none_v2 = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            None,
            "cfg",
        );
        assert_eq!(
            none_v1, none_v2,
            "same epoch, no pin => identical hash regardless of retread version"
        );

        let pinned_v1 = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            Some("1.7.2"),
            "cfg",
        );
        let pinned_v2 = RetreadLock::compute_inputs_hash(
            &["a==2".into()],
            &["https://pypi.org/simple/".into()],
            "patch-then-minor",
            "3.11",
            1,
            Some("1.7.3"),
            "cfg",
        );
        assert_ne!(
            pinned_v1, pinned_v2,
            "different versions under pin_version=Some must differ"
        );
    }
}
