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
    /// v0.34.0+: refinement passes the cascade made before reaching
    /// this outcome (each pass widened one or more retread-emitted
    /// deps to `*`). Empty when the first solve succeeded.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refinement_steps: Vec<crate::audit::RefinementStep>,
    /// v0.35.0+: workspace-edit suggestions from the classifier.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub workspace_edit_suggestions: Vec<crate::conflict_classifier::WorkspaceEditSuggestion>,
    /// v0.35.0+: terminal classification name (A/AExhausted/B/C/None).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_classification: Option<String>,
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
            refinement_steps: Vec::new(),
            workspace_edit_suggestions: Vec::new(),
            terminal_classification: None,
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
/// v0.37.0+: build the rattler virtual-package set the solver
/// should see. Order of operations:
///   1. Host detection (`__archspec`, `__linux`, `__glibc`, `__cuda`,
///      `__osx`, ...) — provides defaults for keys the workspace
///      doesn't constrain explicitly.
///   2. `__cpython` override from the variant-derived `target_python`
///      so transitive `python_abi` constraints resolve consistently
///      with what pixi will install.
///   3. Workspace `[feature.X.system-requirements]` overrides — pixi
///      treats these as authoritative; retread must too. Keys map:
///        cuda     -> __cuda
///        libc     -> __glibc   (linux; value is glibc version)
///        macos    -> __osx
///        archspec -> __archspec (build-string encoded)
///        linux    -> __linux
///      Unrecognized keys are trace-logged + skipped (forward-compat).
///
/// Without (3), retread's solve_check sees the BUILD HOST's virtual
/// packages while pixi's actual solve sees the WORKSPACE-declared
/// ones — the asymmetry produced "retread sat, pixi unsat" for the
/// gsn gymnasium failure that motivated v0.37.0. Extracted into a
/// pure function so the mapping logic is unit-testable without
/// running a full solve.
pub fn build_virtual_packages(
    target_python: &str,
    system_requirements: &std::collections::BTreeMap<String, String>,
) -> Vec<GenericVirtualPackage> {
    let mut virtual_packages: Vec<GenericVirtualPackage> =
        match rattler_virtual_packages::VirtualPackage::detect(
            &rattler_virtual_packages::VirtualPackageOverrides::default(),
        ) {
            Ok(vps) => vps.into_iter().map(GenericVirtualPackage::from).collect(),
            Err(e) => {
                tracing::debug!(
                    error = %e,
                    "solve-check: host virtual package detection failed; falling back to minimal set",
                );
                Vec::new()
            }
        };
    if let Ok(v) = Version::from_str(target_python) {
        if let Ok(name) = PackageName::from_str("__cpython") {
            virtual_packages.retain(|vp| vp.name.as_normalized() != "__cpython");
            virtual_packages.push(GenericVirtualPackage {
                name,
                version: v,
                build_string: String::new(),
            });
        }
    }
    for (req_key, req_value) in system_requirements {
        let (vp_name, is_build_string) = match req_key.as_str() {
            "cuda" => ("__cuda", false),
            "libc" | "glibc" => ("__glibc", false),
            "macos" | "osx" => ("__osx", false),
            // Archspec is unconditionally a build-string virtual
            // package -- even values like "1" or "10" that LOOK
            // like versions are arch identifiers; treat as such.
            "archspec" => ("__archspec", true),
            "linux" => ("__linux", false),
            other => {
                tracing::trace!(
                    key = %other,
                    "solve-check: ignoring unrecognized system-requirement key (not in pixi schema)",
                );
                continue;
            }
        };
        let Ok(name) = PackageName::from_str(vp_name) else {
            continue;
        };
        let (version, build_string) = if is_build_string {
            let Ok(v1) = Version::from_str("1.0") else {
                continue;
            };
            (v1, req_value.clone())
        } else {
            match Version::from_str(req_value) {
                Ok(v) => (v, String::new()),
                Err(_) => {
                    let Ok(v1) = Version::from_str("1.0") else {
                        continue;
                    };
                    (v1, req_value.clone())
                }
            }
        };
        virtual_packages.retain(|vp| vp.name.as_normalized() != vp_name);
        virtual_packages.push(GenericVirtualPackage {
            name,
            version,
            build_string,
        });
    }
    virtual_packages
}

pub async fn run_solve_check(
    channels: &[ChannelUrl],
    specs: &[String],
    target_python: &str,
    target_subdir: &str,
    channel_priority: ChannelPriority,
    system_requirements: &std::collections::BTreeMap<String, String>,
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
    // Dedup filenames as we collect. Across channel/subdir
    // boundaries (e.g. the same package appearing in both linux-64
    // and noarch repodata, or duplicated across channel mirrors)
    // rattler_solve aborts with `DuplicateRecords` before ever
    // attempting a solve. Keep first-seen filename and skip the
    // rest -- they would have been treated as the same package by
    // the solver anyway. Channel priority is still honored because
    // we iterate channels in `channels` order.
    let mut seen_filenames: HashSet<String> = HashSet::new();
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
            for rec in records {
                if seen_filenames.insert(rec.file_name.clone()) {
                    all_records.push(rec);
                }
            }
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
            refinement_steps: Vec::new(),
            workspace_edit_suggestions: Vec::new(),
            terminal_classification: None,
        };
    }

    let virtual_packages = build_virtual_packages(target_python, system_requirements);

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
        channel_priority,
        exclude_newer: None,
        min_age: None,
        // v0.37.0+: pinned explicitly to match pixi's own default.
        // rattler_solve 4.1.x's `SolveStrategy::default()` is
        // `Highest`; pixi also defaults to Highest. If pixi ever
        // changes its default (or starts honoring a workspace-level
        // override like `[workspace].solve-strategy`), retread
        // should mirror — divergence here means retread's solver
        // explores branches in a different order than pixi's,
        // producing different (numpy, cuda, python_abi) tuples for
        // the same spec set and silently disagreeing on
        // satisfiability. Pin explicitly so the contract is
        // visible in code review.
        strategy: SolveStrategy::Highest,
    };
    let mut solver = resolvo::Solver;
    match solver.solve(task) {
        Ok(_solution) => SolveOutcome {
            satisfiable: true,
            unsat_explanations: Vec::new(),
            channels_consulted: consulted,
            specs_count,
            records_count,
            refinement_steps: Vec::new(),
            workspace_edit_suggestions: Vec::new(),
            terminal_classification: None,
        },
        Err(rattler_solve::SolveError::Unsolvable(reasons)) => SolveOutcome {
            satisfiable: false,
            unsat_explanations: reasons,
            channels_consulted: consulted,
            specs_count,
            records_count,
            refinement_steps: Vec::new(),
            workspace_edit_suggestions: Vec::new(),
            terminal_classification: None,
        },
        Err(other) => SolveOutcome {
            satisfiable: false,
            unsat_explanations: vec![format!("solver error: {other}")],
            channels_consulted: consulted,
            specs_count,
            records_count,
            refinement_steps: Vec::new(),
            workspace_edit_suggestions: Vec::new(),
            terminal_classification: None,
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

/// v0.34.0+: parse rattler_solve's tree-formatted unsat explanation
/// strings to find the package names that are the ENTRY POINTS of the
/// conflict graph. These are the deps the solver couldn't satisfy at
/// the top level -- widening any of them in retread's emission gives
/// the solver more freedom to backtrack.
///
/// Returns `["triton"]` for the typical triton-cuda chain. Multiple
/// sibling top-level conflicts return multiple names.
pub fn extract_blocking_dep_names(unsat_strs: &[String]) -> Vec<String> {
    extract_blocking_chains(unsat_strs)
        .into_iter()
        .map(|c| c.name)
        .collect()
}

/// v0.35.0+: richer version of `extract_blocking_dep_names`. Preserves
/// the version data we used to throw away after extracting the name:
/// the spec the blocked package was looking for, the versions the
/// solver tried + rejected, and the transitive requirement that caused
/// each rejection.
///
/// The classifier (`conflict_classifier::classify_unsat`) consumes
/// these chains to produce a structured `ConflictClassification` --
/// distinguishing "retread-side widenable" from "workspace-pin-
/// dominated" failures.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct BlockingChain {
    /// Top-level package name (e.g. `"torchvision"`).
    pub name: String,
    /// Spec that was looked up at the top level (e.g. `">=0.22.0"`).
    /// Empty when the parser couldn't extract one.
    pub current_spec: String,
    /// Concrete versions the solver tried and rejected
    /// (e.g. `["0.25.0", "0.26.0"]`). Deduped, sorted in encountered
    /// order. May be empty when the rattler explanation didn't list
    /// individual versions.
    pub rejected_versions: Vec<String>,
    /// Transitive requirement that caused the rejection
    /// (e.g. `"pytorch >=2.10.0,<2.11.0a0"`). Empty when the chain
    /// doesn't dive deeper or the parse missed it.
    pub transitive_requirement: String,
    /// v0.36.1+: true when rattler said "can be installed with any
    /// of the following options" (the dep is satisfiable in
    /// isolation but listed as part of a multi-dep incompatibility
    /// group). false when "cannot be installed" (genuinely
    /// blocking). The classifier uses this to skip suggestion
    /// derivation for installable chains -- they're context, not
    /// the root cause.
    #[serde(default)]
    pub installable: bool,
}

pub fn extract_blocking_chains(unsat_strs: &[String]) -> Vec<BlockingChain> {
    let mut out: Vec<BlockingChain> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for raw in unsat_strs {
        // Walk lines; whenever a top-level marker is found, parse the
        // chain rooted at that line + its descendants (denoted by
        // deeper indentation).
        let lines: Vec<&str> = raw.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            let trimmed_start = line
                .strip_prefix("├─ ")
                .or_else(|| line.strip_prefix("└─ "));
            let Some(rest) = trimmed_start else { continue };
            let token = rest.split_whitespace().next().unwrap_or("");
            if token.is_empty() {
                continue;
            }
            let name = token.split('[').next().unwrap_or(token);
            if name.chars().any(|c| matches!(c, '<' | '>' | '=' | ',' | '!' | '~' | '*')) {
                continue;
            }
            if !seen.insert(name.to_string()) {
                continue;
            }
            // Extract the spec: everything after the name up to
            // " cannot be installed" / " can be installed" / EOL.
            // Also track whether this chain is genuinely blocking
            // ("cannot be installed") or just listed as context
            // ("can be installed with any of the following options").
            let after_name = rest[token.len()..].trim_start();
            let (current_spec, installable) =
                if let Some(end) = after_name.find(" cannot be installed") {
                    (after_name[..end].trim().to_string(), false)
                } else if let Some(end) = after_name.find(" can be installed") {
                    (after_name[..end].trim().to_string(), true)
                } else {
                    (String::new(), false)
                };

            // Walk descendant lines (deeper indentation than the
            // current line) to find a `would require` block and the
            // versions it enumerates.
            let mut rejected_versions: Vec<String> = Vec::new();
            let mut rejected_seen: HashSet<String> = HashSet::new();
            let mut transitive_requirement: String = String::new();
            for tail in lines.iter().skip(idx + 1) {
                if !tail.starts_with('│') && !tail.starts_with(' ') && !tail.starts_with('├')
                    && !tail.starts_with('└') && !tail.starts_with('\t')
                {
                    // Back to top-level, this chain is over.
                    if !tail.is_empty() {
                        break;
                    }
                }
                // The "X 1.0 | 1.0 | 1.0 ... would require" line lists
                // candidate versions. Extract the part after the name
                // and before "would require".
                if let Some(would_idx) = tail.find(" would require") {
                    let before = &tail[..would_idx];
                    // Strip leading tree decoration: any chars in
                    // `│├└─ \t`.
                    let payload = before
                        .trim_start_matches(|c: char| {
                            matches!(c, '│' | '├' | '└' | '─' | ' ' | '\t')
                        });
                    // payload looks like `<name> 0.25.0 | 0.25.0 | ...`
                    // Drop the name (first whitespace-separated token).
                    let mut it = payload.splitn(2, char::is_whitespace);
                    let _ = it.next();
                    if let Some(versions_part) = it.next() {
                        for v in versions_part.split('|') {
                            let v = v.trim();
                            if v.is_empty() {
                                continue;
                            }
                            if rejected_seen.insert(v.to_string()) {
                                rejected_versions.push(v.to_string());
                            }
                        }
                    }
                    continue;
                }
                // Lines like `└─ pytorch >=2.10.0,<2.11.0a0, which
                // cannot be installed` are the transitive requirement
                // chain. Capture the FIRST such line per top-level
                // chain (the closest cause).
                if transitive_requirement.is_empty() {
                    let strip = tail.trim_start_matches(|c: char| {
                        matches!(c, '│' | '├' | '└' | '─' | ' ' | '\t')
                    });
                    // Strip trailing ", which cannot be installed..." /
                    // ", for which no candidates...".
                    let end = strip
                        .find(", which")
                        .or_else(|| strip.find(", for which"))
                        .unwrap_or(strip.len());
                    let candidate = strip[..end].trim().to_string();
                    // Only keep if it looks like `<name> <spec>` not
                    // a versions-list line. Version-enumeration
                    // lines like `pytorch-gpu 2.7.1 | 2.7.1` contain
                    // `|` (the rattler separator); those are
                    // candidates, not transitive requirements.
                    let looks_like_version_list = candidate.contains(" | ");
                    if !candidate.contains(" would require")
                        && !looks_like_version_list
                        && !candidate.is_empty()
                        && candidate
                            .chars()
                            .next()
                            .map_or(false, |c| c.is_ascii_alphabetic() || c == '_')
                    {
                        transitive_requirement = candidate;
                    }
                }
            }

            out.push(BlockingChain {
                name: name.to_string(),
                current_spec,
                rejected_versions,
                transitive_requirement,
                installable,
            });
        }
    }
    out
}

#[cfg(test)]
mod parse_tests {
    use super::*;

    #[test]
    fn extract_blocking_dep_names_pulls_top_level_only() {
        let unsat = vec![
            "The following packages are incompatible\n\
             ├─ triton >=3.7.0,<3.8 cannot be installed because there are no viable options:\n\
             │  ├─ triton 3.7.0 | 3.7.0 would require\n\
             │  │  └─ cuda-version >=13.0,<14, which cannot be installed because there are no viable options:\n\
             │  │     ├─ cuda-version 13.2, which conflicts with the versions reported above.\n\
             └─ another-pkg >=1.0 cannot be installed because there are no viable options:\n\
             │  └─ stuff"
                .to_string(),
        ];
        let blocking = extract_blocking_dep_names(&unsat);
        // Both `triton` and `another-pkg` are top-level conflict
        // entry points. Nested deps (cuda-version) are NOT.
        assert!(blocking.contains(&"triton".to_string()), "expected triton in {blocking:?}");
        assert!(blocking.contains(&"another-pkg".to_string()), "expected another-pkg in {blocking:?}");
        assert!(!blocking.contains(&"cuda-version".to_string()), "cuda-version is nested, not top-level: {blocking:?}");
    }

    #[test]
    fn extract_blocking_dep_names_dedups() {
        let unsat = vec![
            "├─ pkg cannot be installed\n├─ pkg also cannot be installed".to_string(),
        ];
        let blocking = extract_blocking_dep_names(&unsat);
        assert_eq!(blocking, vec!["pkg".to_string()]);
    }

    #[test]
    fn extract_blocking_chains_captures_versions_and_transitive() {
        let unsat = vec![
            "The following packages are incompatible\n\
             └─ torchvision >=0.22.0 cannot be installed because there are no viable options:\n   \
                ├─ torchvision 0.25.0 | 0.25.0 | 0.26.0 would require\n   \
                │  └─ pytorch >=2.10.0,<2.11.0a0, which cannot be installed because there are no viable options:\n   \
                │     └─ pytorch 2.10.0 would require\n   \
                │        └─ cuda-version >=12.9,<13, for which no candidates were found."
                .to_string(),
        ];
        let chains = extract_blocking_chains(&unsat);
        assert_eq!(chains.len(), 1);
        let c = &chains[0];
        assert_eq!(c.name, "torchvision");
        assert_eq!(c.current_spec, ">=0.22.0");
        // Versions deduped, in encountered order.
        assert!(
            c.rejected_versions.contains(&"0.25.0".to_string())
                && c.rejected_versions.contains(&"0.26.0".to_string()),
            "rejected_versions = {:?}",
            c.rejected_versions,
        );
        // Transitive picks the closest cause (pytorch), not the
        // deepest (cuda-version).
        assert!(
            c.transitive_requirement.starts_with("pytorch "),
            "transitive_requirement = {:?}",
            c.transitive_requirement,
        );
    }

    #[test]
    fn extract_blocking_chains_handles_multiple_top_levels() {
        let unsat = vec![
            "The following packages are incompatible\n\
             ├─ pkga >=1 cannot be installed because there are no viable options:\n   \
                └─ stuff\n\
             └─ pkgb cannot be installed because there are no viable options:\n   \
                └─ more stuff"
                .to_string(),
        ];
        let chains = extract_blocking_chains(&unsat);
        let names: Vec<&str> = chains.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"pkga"), "names = {names:?}");
        assert!(names.contains(&"pkgb"), "names = {names:?}");
    }

    #[test]
    fn extract_blocking_chains_empty_fields_when_unparseable() {
        // No "would require" or transitive; chain still emits with
        // empty rejected_versions / transitive_requirement.
        let unsat = vec!["├─ foo cannot be installed because reasons".to_string()];
        let chains = extract_blocking_chains(&unsat);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].name, "foo");
        assert!(chains[0].rejected_versions.is_empty());
        assert!(chains[0].transitive_requirement.is_empty());
    }

    #[test]
    fn extract_blocking_dep_names_skips_specifier_tokens() {
        // Lines whose first token after the marker is a version
        // operator (rare; defensive) shouldn't accidentally count.
        let unsat = vec!["├─ >=1.0 noisy".to_string()];
        assert!(extract_blocking_dep_names(&unsat).is_empty());
    }
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

    // -------------------------------------------------------------
    // v0.37.0 T1: build_virtual_packages exposes the workspace -> rattler
    // virtual-package mapping for unit testing. The full end-to-end
    // run_solve_check parity test against synthetic RepoData is left
    // as future work (it would require committing repodata fixtures
    // under tests/fixtures/); these unit tests pin the contract that
    // the mapping is correct, which is the load-bearing piece. If
    // these pass and the cascade is wired right, retread's solver
    // sees the same virtual packages pixi's will.
    // -------------------------------------------------------------

    fn vp_lookup<'a>(vps: &'a [GenericVirtualPackage], name: &str) -> Option<&'a GenericVirtualPackage> {
        vps.iter().find(|vp| vp.name.as_normalized() == name)
    }

    #[test]
    fn build_virtual_packages_injects_cpython_from_target_python() {
        let sysreqs = std::collections::BTreeMap::new();
        let vps = build_virtual_packages("3.11", &sysreqs);
        let cp = vp_lookup(&vps, "__cpython").expect("__cpython must be present");
        // Version parses as `3.11.0` (zero-extended).
        assert!(cp.version.to_string().starts_with("3.11"));
    }

    #[test]
    fn build_virtual_packages_maps_workspace_cuda_to_double_underscore() {
        // The load-bearing case for v0.37.0 D1: workspace declares
        // `cuda = "12"` in `[feature.gpu.system-requirements]`. Without
        // this mapping, retread sees the build host's `__cuda` (which
        // may be absent or a different version), pixi sees __cuda 12,
        // they disagree on whether a `__cuda >=12` dep is installable.
        let mut sysreqs = std::collections::BTreeMap::new();
        sysreqs.insert("cuda".into(), "12".into());
        let vps = build_virtual_packages("3.11", &sysreqs);
        let cuda = vp_lookup(&vps, "__cuda").expect("__cuda must be injected");
        assert!(
            cuda.version.to_string().starts_with("12"),
            "expected __cuda 12.x, got {}",
            cuda.version,
        );
    }

    #[test]
    fn build_virtual_packages_maps_libc_to_glibc() {
        // pixi's schema names the glibc requirement `libc` (or `glibc`).
        // The matching virtual package is `__glibc` (linux-specific
        // family). Without this mapping, packages that declare
        // `__glibc >=2.35` would see the host's __glibc instead of
        // the workspace's declaration.
        let mut sysreqs = std::collections::BTreeMap::new();
        sysreqs.insert("libc".into(), "2.35".into());
        let vps = build_virtual_packages("3.11", &sysreqs);
        let glibc = vp_lookup(&vps, "__glibc").expect("__glibc must be injected");
        assert_eq!(glibc.version.to_string(), "2.35");
    }

    #[test]
    fn build_virtual_packages_overrides_host_detection() {
        // Workspace value MUST win over host detection. The build host
        // probably has a __glibc version different from the workspace's
        // declared one; the workspace's value is authoritative because
        // that's what pixi uses.
        let mut sysreqs = std::collections::BTreeMap::new();
        sysreqs.insert("libc".into(), "9.99".into());
        let vps = build_virtual_packages("3.11", &sysreqs);
        // Exactly ONE __glibc entry (overriding any host-detected one).
        let count = vps
            .iter()
            .filter(|vp| vp.name.as_normalized() == "__glibc")
            .count();
        assert_eq!(count, 1, "must replace, not append, the host's __glibc");
        let glibc = vp_lookup(&vps, "__glibc").unwrap();
        assert_eq!(glibc.version.to_string(), "9.99");
    }

    #[test]
    fn build_virtual_packages_archspec_encodes_as_build_string() {
        // Archspec values are NOT versions (`x86_64`, `aarch64`); they
        // belong in the build-string slot. Pin that encoding.
        let mut sysreqs = std::collections::BTreeMap::new();
        sysreqs.insert("archspec".into(), "x86_64".into());
        let vps = build_virtual_packages("3.11", &sysreqs);
        let arch = vp_lookup(&vps, "__archspec").expect("__archspec must be injected");
        assert_eq!(arch.build_string, "x86_64");
    }

    #[test]
    fn build_virtual_packages_skips_unknown_keys() {
        // Forward-compat: future pixi schema additions shouldn't kill
        // the solve check. Unknown keys log + skip.
        let mut sysreqs = std::collections::BTreeMap::new();
        sysreqs.insert("something-new-in-pixi".into(), "12".into());
        let vps_baseline = build_virtual_packages("3.11", &std::collections::BTreeMap::new());
        let vps = build_virtual_packages("3.11", &sysreqs);
        // No __something-new-in-pixi (or any other) was added.
        assert_eq!(vps.len(), vps_baseline.len(),
            "unknown system-requirement keys must not add virtual packages");
    }
}
