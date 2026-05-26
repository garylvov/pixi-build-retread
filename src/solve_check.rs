//! v0.33.0+: pre-emission solve check.
//!
//! After retread builds a per-env CondaOutput, run a real conda solve
//! over (workspace effective deps + retread's emitted run-deps) against
//! the workspace's channels. If unsat, capture the explanation strings
//! and persist them to the audit. This catches cross-package conflicts
//! that the per-dep probe layer can't see:
//!
//!   - retread emits `cuda-bindings >=13.0.3,<14` (from a wheel's
//!     Requires-Dist)
//!   - workspace pins `cuda-toolkit 12.8.*`
//!   - cuda-bindings 13.x's `depends` says it needs `cuda 13.x`
//!   - cuda-toolkit 12.8.*'s `depends` says cuda 12.8.*
//!   - same dep (`cuda`) with incompatible ranges across two
//!     workspace-/retread-pinned packages -> solver explanation
//!
//! The per-dep probe never catches this because it only asks "does the
//! spec I'd emit have ANY candidate on the channel" -- not "does that
//! candidate compose with the rest of the workspace."
//!
//! Cost: one full conda solve per (python_variant, discovered_output).
//! For a typical isaac-pack with 1 output this adds ~1-3 seconds to
//! conda/outputs. Repodata is already cached in memory by probe.rs so
//! we don't pay the fetch cost again.

use std::collections::HashSet;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{anyhow, Context, Result};
use rattler_conda_types::{
    Channel, ChannelConfig, ChannelUrl, GenericVirtualPackage, MatchSpec, PackageName,
    ParseStrictness, RepoData, RepoDataRecord, Version,
};
use rattler_solve::{resolvo, ChannelPriority, SolveStrategy, SolverImpl, SolverTask};
use serde::Serialize;
use sha2::{Digest, Sha256};

/// Result of running the solve check.
#[derive(Debug, Clone, Serialize)]
pub struct SolveOutcome {
    /// True if the solver found a valid solution.
    pub satisfiable: bool,
    /// Per-leaf explanation strings from the rattler solver when
    /// `satisfiable == false`. Each string names a constraint chain
    /// the solver couldn't reconcile.
    pub unsat_explanations: Vec<String>,
    /// Channels actually consulted (after disk-cache reads).
    pub channels_consulted: Vec<String>,
    /// Total specs the solver was asked to satisfy.
    pub specs_count: usize,
    /// Total available records across all channels.
    pub records_count: usize,
}

impl SolveOutcome {
    pub fn unreachable() -> Self {
        Self {
            satisfiable: false,
            unsat_explanations: vec![
                "solve-check skipped: no repodata could be loaded".into(),
            ],
            channels_consulted: Vec::new(),
            specs_count: 0,
            records_count: 0,
        }
    }
}

/// Run a conda solve over the combined spec set. `specs` must include
/// both retread's emitted run-deps for this output AND the workspace's
/// effective conda deps for the env this output is built for. Target
/// python is encoded as a virtual package (`__cuda` not included; if
/// you need to model GPU/CUDA-only constraints, add them via the
/// caller).
///
/// `target_subdir` is the linux-64/osx-64/etc. selector. retread today
/// targets linux-64 only.
pub async fn run_solve_check(
    channels: &[ChannelUrl],
    specs: &[String],
    target_python: &str,
    target_subdir: &str,
) -> SolveOutcome {
    // Parse the specs first; bad input shouldn't be hidden behind
    // network IO. Skip specs that don't parse (rare; logged at debug).
    let mut parsed_specs: Vec<MatchSpec> = Vec::with_capacity(specs.len());
    for raw in specs {
        match MatchSpec::from_str(raw, ParseStrictness::Lenient) {
            Ok(s) => parsed_specs.push(s),
            Err(e) => {
                tracing::debug!(spec = %raw, error = %e, "solve-check: skipping unparseable spec");
            }
        }
    }

    let mut all_records: Vec<RepoDataRecord> = Vec::new();
    let mut consulted: Vec<String> = Vec::new();
    for channel_url in channels {
        let channel_str = channel_url.url().as_str().trim_end_matches('/').to_string();
        for subdir in [target_subdir, "noarch"] {
            let Ok(Some(bytes)) = read_disk_cache(&channel_str, subdir).await else {
                continue;
            };
            let repo_data: RepoData = match serde_json::from_slice(&bytes) {
                Ok(r) => r,
                Err(e) => {
                    tracing::debug!(
                        channel = %channel_str,
                        subdir = %subdir,
                        error = %e,
                        "solve-check: failed to parse cached repodata",
                    );
                    continue;
                }
            };
            // Build a Channel for record URL synthesis. ChannelConfig
            // is required by the constructor; default config is fine
            // because we'll never resolve relative aliases here.
            let cfg = ChannelConfig::default_with_root_dir(std::env::temp_dir());
            let channel = match Channel::from_str(&channel_str, &cfg) {
                Ok(c) => c,
                Err(e) => {
                    tracing::debug!(
                        channel = %channel_str,
                        error = %e,
                        "solve-check: failed to build Channel for record URL synthesis",
                    );
                    continue;
                }
            };
            let records = repo_data.into_repo_data_records(&channel);
            consulted.push(format!("{channel_str}/{subdir}"));
            all_records.extend(records);
        }
    }

    if all_records.is_empty() {
        return SolveOutcome {
            satisfiable: false,
            unsat_explanations: vec![
                "solve-check skipped: no repodata available from disk cache".into(),
            ],
            channels_consulted: consulted,
            specs_count: parsed_specs.len(),
            records_count: 0,
        };
    }

    // Construct virtual packages for the target python. We pin
    // __cpython here so the solver respects python_abi constraints.
    // Other virtual packages (linux, glibc, cuda, archspec) are
    // intentionally NOT injected -- adding them risks the solver
    // accepting solutions that depend on host info we don't actually
    // know about at retread time. Per-dep python_abi filtering in the
    // probe layer already eliminates cross-python false-sats.
    let mut virtual_packages: Vec<GenericVirtualPackage> = Vec::new();
    if let Ok(v) = Version::from_str(target_python) {
        if let Ok(name) = PackageName::from_str("__cpython") {
            virtual_packages.push(GenericVirtualPackage {
                name,
                version: v,
                build_string: String::new(),
            });
        }
    }

    let records_count = all_records.len();
    let specs_count = parsed_specs.len();

    let task = SolverTask {
        available_packages: vec![&all_records],
        locked_packages: Vec::new(),
        pinned_packages: Vec::new(),
        virtual_packages,
        specs: parsed_specs,
        constraints: Vec::new(),
        timeout: Some(std::time::Duration::from_secs(60)),
        channel_priority: ChannelPriority::Strict,
        exclude_newer: None,
        min_age: None,
        strategy: SolveStrategy::default(),
    };
    let mut solver = resolvo::Solver;
    match solver.solve(task) {
        Ok(_solution) => SolveOutcome {
            satisfiable: true,
            unsat_explanations: Vec::new(),
            channels_consulted: consulted,
            specs_count,
            records_count,
        },
        Err(rattler_solve::SolveError::Unsolvable(reasons)) => SolveOutcome {
            satisfiable: false,
            unsat_explanations: reasons,
            channels_consulted: consulted,
            specs_count,
            records_count,
        },
        Err(other) => SolveOutcome {
            satisfiable: false,
            unsat_explanations: vec![format!("solver error: {other}")],
            channels_consulted: consulted,
            specs_count,
            records_count,
        },
    }
}

/// Compute the disk-cache path for repodata. Must match probe.rs's
/// layout exactly -- we read what that module wrote.
fn disk_cache_path(channel_url: &str, subdir: &str) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(channel_url.as_bytes());
    hasher.update(b"|");
    hasher.update(subdir.as_bytes());
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(16);
    for b in &digest[..8] {
        hex.push_str(&format!("{b:02x}"));
    }
    let dir = dirs_cache_root().join("retread-repodata");
    let slug = channel_url
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or("channel")
        .replace(|c: char| !c.is_ascii_alphanumeric(), "_");
    dir.join(format!("{slug}--{subdir}--{hex}.json"))
}

fn dirs_cache_root() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache").join("rattler").join("cache")
    } else {
        std::env::temp_dir().join("retread-cache")
    }
}

const REPODATA_TTL: std::time::Duration = std::time::Duration::from_secs(30 * 60);

async fn read_disk_cache(channel_url: &str, subdir: &str) -> Result<Option<Vec<u8>>> {
    let path = disk_cache_path(channel_url, subdir);
    let meta = match tokio::fs::metadata(&path).await {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("stat {}", path.display())),
    };
    let mtime = meta
        .modified()
        .with_context(|| format!("mtime {}", path.display()))?;
    let age = std::time::SystemTime::now()
        .duration_since(mtime)
        .unwrap_or(std::time::Duration::ZERO);
    if age > REPODATA_TTL {
        // Stale; probe.rs will refresh on its next call but we don't
        // refetch here -- the cache is best-effort for solve check.
        return Ok(None);
    }
    let bytes = tokio::fs::read(&path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;
    Ok(Some(bytes))
}

#[allow(dead_code)]
fn _quiet_anyhow_warning() -> Result<()> {
    Err(anyhow!("unused"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_outcome_unreachable_default() {
        let o = SolveOutcome::unreachable();
        assert!(!o.satisfiable);
        assert_eq!(o.unsat_explanations.len(), 1);
    }
}
