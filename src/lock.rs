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

pub const SCHEMA: u32 = 2;

impl RetreadLock {
    /// File name for a bundle's lock next to the pack manifest.
    pub fn file_name(bundle: &str) -> String {
        format!("retread-{bundle}.lock.json")
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
            root_requirements: vec!["isaac-pack-pypi==5.1.0".into()],
            wheels: vec![
                LockWheel {
                    name: "isaaclab".into(),
                    version: "0.51.1".into(),
                    origin: Origin::Built,
                    filename: "isaaclab-0.51.1-py3-none-any.whl".into(),
                    url: None,
                },
                LockWheel {
                    name: "isaacsim-core".into(),
                    version: "5.1.0.0".into(),
                    origin: Origin::Index,
                    filename: "isaacsim_core-5.1.0.0-cp311-...whl".into(),
                    url: Some("https://pypi.nvidia.com/isaacsim-core/...whl".into()),
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
    }
}
