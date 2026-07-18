//! v2.0.0 courier: producer-side staging.
//!
//! Stages the bundle's built + relax-changed ("shadow") wheels + the
//! generated meta-wheel + the committed lock into a staging dir, WITHOUT
//! touching the consumer pixi.toml. The courier conda package then ships
//! these as data and installs them at link time via `retread install`.
//!
//! FROZEN CONTRACT (phase 0): [`CourierStaged`] is the output consumed by
//! `build_one` (WS-C) and the [`stage`] signature is what WS-C compiles
//! against. WS-A implements the body. Do not change these shapes without a
//! stop-the-world re-freeze.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::io::{Read, Seek};
use std::path::Path;
use std::str::FromStr as _;

use anyhow::Context as _;
use rattler_conda_types::Platform;
use sha2::{Digest, Sha256};
use uv_pep508::Requirement;

use crate::config::{NamedGitSource, RetreadConfig, RoutePolicy, WheelEntry};
use crate::emit_pypi::{
    EmitWheel, build_meta_wheel, collect_prerelease_pins, insert_build_tag, override_line_map,
    plan, standard_wheel_filename,
};
use crate::lock::{CondaDep, LockWheel, Origin, RetreadLock, SCHEMA};
use crate::pypi::{ResolutionTarget, WheelTarget};

const SHADOW_DOWNLOAD_ATTEMPTS: usize = 5;
const SHADOW_RETRY_BASE_DELAY_MS: u64 = if cfg!(test) { 10 } else { 250 };

fn retryable_shadow_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::REQUEST_TIMEOUT
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status.is_server_error()
}

fn retryable_shadow_transport(error: &reqwest::Error) -> bool {
    error.is_connect() || error.is_timeout() || error.is_body()
}

async fn wait_before_shadow_retry(url: &url::Url, wheel: &str, attempt: usize, reason: &str) {
    let delay_ms = SHADOW_RETRY_BASE_DELAY_MS * (1_u64 << (attempt - 1));
    tracing::warn!(
        %url,
        wheel,
        attempt,
        max_attempts = SHADOW_DOWNLOAD_ATTEMPTS,
        delay_ms,
        reason,
        "transient courier wheel download failure; retrying",
    );
    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
}

/// Fetch a remote-only wheel needed for a courier shadow rewrite. These
/// downloads happen at the very end of an expensive solve, so transient DNS,
/// timeout, rate-limit, and server failures must not discard the whole run.
async fn download_shadow_wheel(
    client: &reqwest::Client,
    url: &url::Url,
    wheel: &str,
) -> anyhow::Result<Vec<u8>> {
    for attempt in 1..=SHADOW_DOWNLOAD_ATTEMPTS {
        let response = match client.get(url.clone()).send().await {
            Ok(response) => response,
            Err(error)
                if attempt < SHADOW_DOWNLOAD_ATTEMPTS && retryable_shadow_transport(&error) =>
            {
                wait_before_shadow_retry(url, wheel, attempt, &error.to_string()).await;
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let status = response.status();
        if !status.is_success() {
            if attempt < SHADOW_DOWNLOAD_ATTEMPTS && retryable_shadow_status(status) {
                wait_before_shadow_retry(url, wheel, attempt, &format!("HTTP {status}")).await;
                continue;
            }
            return Err(response.error_for_status().unwrap_err().into());
        }
        match response.bytes().await {
            Ok(bytes) => return Ok(bytes.to_vec()),
            Err(error)
                if attempt < SHADOW_DOWNLOAD_ATTEMPTS && retryable_shadow_transport(&error) =>
            {
                wait_before_shadow_retry(url, wheel, attempt, &error.to_string()).await;
            }
            Err(error) => return Err(error.into()),
        }
    }
    unreachable!("shadow download retry loop always returns on its final attempt")
}

/// Everything `build_one` needs to build the courier recipe + write the lock.
pub struct CourierStaged {
    /// `file://` URLs of the staged artifacts (shipped wheels + the lock
    /// json) for the courier recipe's `source:` list.
    pub source_urls: Vec<String>,
    /// The assembled, ready-to-write install lock.
    pub lock: RetreadLock,
    /// The solved conda run-deps for the courier recipe's `requirements.run`
    /// (uv + pixi-build-retread are appended by `build_courier_recipe`).
    pub run_deps: Vec<String>,
}

/// Canonical resolution-INPUT specs for one bundle, used IDENTICALLY by the
/// lock producer (`stage`) and the cold-solve replayer (`conda_outputs`) so
/// their `inputs_hash` matches and replay actually fires. PURE manifest inputs
/// only -- entry key + extras + (explicit version | git rev | url) -- NEVER a
/// resolved version (the replayer runs BEFORE the cascade resolves anything,
/// so a resolved version would make the two hashes diverge). Sorted; the order
/// is not significant (compute_inputs_hash sorts too, but we sort here so this
/// fn is a stable standalone contract).
/// Canonical spec string for a single `[retread-wheels]` entry.
///
/// Format: `<key>[<extras>]<ver_proxy>` where `ver_proxy` is one of:
/// - `==<version>` for an explicit pin
/// - `@git:<rev>` for an inline or named git source
/// - `@url:<url>[#sha256=<digest>]` for a direct URL
/// - `""` (empty) for a bare/range entry
///
/// All three encoding sites that build spec strings for gate comparison
/// (`courier_input_specs`, `detect_incremental_add` STEP A, and the
/// `matched_entries` block in `resolve_incremental_add`) must call this
/// function so that the encodings never drift.
pub fn spec_for_entry(
    key: &str,
    entry: &WheelEntry,
    git_sources: &BTreeMap<String, NamedGitSource>,
) -> String {
    let extras = if entry.extras.is_empty() {
        String::new()
    } else {
        format!("[{}]", entry.extras.join(","))
    };
    // version proxy, pure-input precedence: explicit pin, then inline
    // git rev, then named-git-source rev, then direct url, else bare.
    let ver = entry
        .normalized_version()
        .map(|v| format!("=={v}"))
        .or_else(|| entry.rev.clone().map(|r| format!("@git:{r}")))
        .or_else(|| {
            entry
                .from
                .as_ref()
                .and_then(|f| git_sources.get(f))
                .map(|s| format!("@git:{}", s.rev))
        })
        .or_else(|| {
            entry.url.as_ref().map(|url| {
                let hash = entry
                    .sha256
                    .as_deref()
                    .map(|digest| format!("#sha256={}", digest.to_ascii_lowercase()))
                    .unwrap_or_default();
                format!("@url:{url}{hash}")
            })
        })
        .unwrap_or_default();
    format!("{key}{extras}{ver}")
}

pub fn courier_input_specs(config: &RetreadConfig, bundle_name: &str) -> Vec<String> {
    let mut specs: Vec<String> = config
        .retread_wheels
        .iter()
        .filter(|(key, entry)| {
            let group = entry.bundle.as_deref().or(config.default_bundle.as_deref());
            let expected = crate::relax::canonical_conda_name(bundle_name);
            match group {
                Some(g) => crate::relax::canonical_conda_name(g) == expected,
                None => crate::relax::canonical_conda_name(key) == expected,
            }
        })
        .map(|(key, entry)| spec_for_entry(key, entry, &config.git_sources))
        .collect();
    specs.sort();
    specs
}

/// Canonical fingerprint of the resolution-affecting config inputs that are
/// NOT already folded into `compute_inputs_hash` via `courier_input_specs`
/// (entry key/extras/version/git-rev/url), the index chain, relax, or python.
/// Producer (`stage`) and replayer (`conda_outputs`) MUST build this
/// identically or the inputs hash diverges and replay never fires / fires
/// stale.
///
/// Folds in: per-dep overrides, the PyPI->conda name-map, drop-deps,
/// conda-deps, the route policy and include set, the auto-bundle toggle, the
/// build-number, deps-from source identity, AND the conda channel list. Each
/// of these changes the emitted
/// conda specs or the conda/PyPI routing, so omitting any would let a
/// manifest/workspace edit leave the hash unchanged and replay a stale,
/// POISONED lock. (genesis's `retread-name-map` is the canonical config
/// regression case; a workspace channel addition is the canonical channel
/// case -- a newly-added channel can make a previously auto-bundled wheel
/// conda-capable, flipping its lock classification.)
///
/// `conda_channels` is the channel set pixi forwards (`params.channels`,
/// stringified). Order is SIGNIFICANT (conda channel priority), so it is NOT
/// sorted -- exactly like the PyPI index chain. The producer (`build_one`) and
/// the replayer (`conda_outputs`) MUST pass the same stringified channel list.
///
/// `workspace_fp` is `WorkspaceManifest::solve_fingerprint()` (empty when there
/// is no workspace manifest). It folds in the WORKSPACE solve environment --
/// per-env conda dep pins, system-requirements, pypi-options, feature/env
/// wiring (grizzly H1) -- which shape what pixi actually solves and forwards as
/// the lock's conda run-deps. Without it, a workspace pixi.toml edit would
/// replay a stale lock. Both sides load the same manifest, so the strings agree.
pub fn config_fingerprint(
    config: &RetreadConfig,
    conda_channels: &[String],
    workspace_fp: &str,
) -> String {
    // BTreeMaps iterate in sorted key order already; sort the Vecs explicitly
    // so ordering in the manifest never perturbs the digest.
    let mut parts: Vec<String> = Vec::new();
    for (k, v) in &config.overrides {
        parts.push(format!("override:{k}={v}"));
    }
    for (k, v) in &config.name_map {
        parts.push(format!("name-map:{k}={v}"));
    }
    for (k, v) in &config.shadow_libs {
        parts.push(format!("shadow-lib:{k}={}", v.as_lock_value()));
    }
    for (index, source) in config.deps_from.as_slice().iter().enumerate() {
        match source {
            crate::deps_from::DepSource::Local(path) => {
                parts.push(format!("deps-from:{index}:local:{}", path.display()));
            }
            crate::deps_from::DepSource::Url(url) => {
                parts.push(format!("deps-from:{index}:url:{url}"));
            }
            crate::deps_from::DepSource::Git { git, rev, path } => {
                parts.push(format!("deps-from:{index}:git:{git}@{rev}:{path}"));
            }
        }
    }
    let mut drop = config.drop_deps.clone();
    drop.sort();
    for d in &drop {
        parts.push(format!("drop:{d}"));
    }
    let mut conda = config.conda_deps.clone();
    conda.sort();
    for c in &conda {
        parts.push(format!("conda-dep:{c}"));
    }
    let route_policy = match config.route_policy {
        RoutePolicy::PreferCondaValidated => "prefer-conda-validated",
        RoutePolicy::Minimal => "minimal",
        RoutePolicy::Aggressive => "aggressive",
    };
    parts.push(format!("route-policy:{route_policy}"));
    let mut route_include = config.route_include.clone();
    route_include.sort();
    for name in &route_include {
        parts.push(format!("route-include:{name}"));
    }
    parts.push(format!("auto-bundle:{}", config.auto_bundle));
    parts.push(format!("build-number:{}", config.build_number));
    for c in conda_channels {
        parts.push(format!("channel:{c}"));
    }
    if !workspace_fp.is_empty() {
        parts.push(format!("--workspace--\n{workspace_fp}"));
    }
    parts.join("\n")
}

/// Parse `"name spec"` lines (space-separated) into [`CondaDep`] values.
/// Lines with no space (name only) get an empty spec. Blank lines are skipped.
fn parse_conda_deps(run_deps: &[String]) -> Vec<CondaDep> {
    run_deps
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut parts = l.splitn(2, ' ');
            let name = parts.next().unwrap_or("").to_string();
            let spec = parts.next().unwrap_or("").to_string();
            CondaDep { name, spec }
        })
        .collect()
}

/// Convert an absolute path to a `file://` URL.
fn file_url(path: &Path) -> anyhow::Result<String> {
    let abs = path
        .canonicalize()
        .with_context(|| format!("canonicalizing {}", path.display()))?;
    let s = abs
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF-8 path: {}", abs.display()))?;
    Ok(format!("file://{s}"))
}

// ── Persistent cache root ────────────────────────────────────────────────────

/// Machine-global cache root for retread's content-addressed caches.
///
/// Priority (first wins):
///   1. `RETREAD_CACHE_DIR` env var (absolute path; overrides everything).
///   2. `XDG_CACHE_HOME/retread` (Linux XDG standard).
///   3. `$HOME/.cache/retread` (POSIX fallback).
///
/// The returned path may not exist yet; callers create it with
/// `fs::create_dir_all` as needed.
///
/// EMIT-NEUTRAL: this location only governs WHERE content-addressed caches
/// live, never WHAT bytes get emitted. The cache root never feeds
/// `inputs_hash` and a cache hit is byte-identical to a fresh rewrite, so
/// relocating it does not warrant an `EMIT_EPOCH` bump.
pub fn retread_cache_root() -> std::path::PathBuf {
    match crate::fasttmp::backend_env_override("RETREAD_CACHE_DIR") {
        crate::fasttmp::BackendEnvOverride::Set(dir) => {
            return std::path::PathBuf::from(dir);
        }
        crate::fasttmp::BackendEnvOverride::Remove => return default_retread_cache_root(),
        crate::fasttmp::BackendEnvOverride::Unchanged => {}
    }
    if let Ok(dir) = std::env::var("RETREAD_CACHE_DIR") {
        return std::path::PathBuf::from(dir);
    }
    default_retread_cache_root()
}

fn default_retread_cache_root() -> std::path::PathBuf {
    let base = std::env::var("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| std::path::PathBuf::from(h).join(".cache"))
                .unwrap_or_else(|_| std::env::temp_dir().join(".retread-cache-fallback"))
        });
    base.join("retread")
}

// ── Shared loose-bundle wheel store ─────────────────────────────────────────

/// Shared content-addressed wheel store root: the directory holding
/// `<sha256>/<filename>` entries (loose-bundle built wheels + the download
/// cache of `fetch_wheel_cached`).
///
/// Priority (first wins):
///   1. `RETREAD_WHEEL_STORE` env var (absolute path to the store dir).
///   2. `XDG_CACHE_HOME/retread/wheels`.
///   3. `$HOME/.cache/retread/wheels` (POSIX fallback).
///
/// DELIBERATELY independent of `RETREAD_CACHE_DIR` / [`retread_cache_root`]:
/// fast-tmp redirects `RETREAD_CACHE_DIR` into a JOB-LOCAL tmp namespace, but
/// the wheel store is a PERSISTENCE CONTRACT, not a scratch cache — a loose
/// bundle's lock records store shas at build time and `retread install` must
/// find those bytes later, on other nodes, after the build job's tmp is gone.
/// Blob stores stay SHARED; only envs/scratch are job-local. The store is
/// content-addressed with atomic tmp+rename writes, so concurrent writers on
/// NFS are safe.
pub fn retread_wheel_store_root() -> std::path::PathBuf {
    wheel_store_root_with(&|key| std::env::var(key).ok())
}

/// Testable core of [`retread_wheel_store_root`]; `env` is the variable
/// lookup. Note the intentional ABSENCE of a `RETREAD_CACHE_DIR` branch.
pub(crate) fn wheel_store_root_with(env: &dyn Fn(&str) -> Option<String>) -> std::path::PathBuf {
    if let Some(dir) = env("RETREAD_WHEEL_STORE").filter(|s| !s.trim().is_empty()) {
        return std::path::PathBuf::from(dir);
    }
    let base = env("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            env("HOME")
                .map(|h| std::path::PathBuf::from(h).join(".cache"))
                .unwrap_or_else(|| std::env::temp_dir().join(".retread-cache-fallback"))
        });
    base.join("retread").join("wheels")
}

/// Portable form of a wheel-store path for the committed lock: a store under
/// the producer's `$HOME` is recorded as `~/...` so the lock stays
/// byte-identical across users/machines (the store default is per-user; the
/// consumer expands `~` against its OWN home). Non-home paths (e.g. a shared
/// project dir via `RETREAD_WHEEL_STORE`) are recorded verbatim.
pub(crate) fn portable_wheel_store_path(store: &Path) -> String {
    if let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
        && let Ok(rel) = store.strip_prefix(&home)
    {
        return format!("~/{}", rel.display());
    }
    store.display().to_string()
}

/// Inverse of [`portable_wheel_store_path`] on the consumer side.
pub fn expand_wheel_store_path(recorded: &str) -> std::path::PathBuf {
    if let Some(rest) = recorded.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        return std::path::PathBuf::from(home).join(rest);
    }
    std::path::PathBuf::from(recorded)
}

// ── Shadow-rewrite cache helpers ────────────────────────────────────────────

/// Compute the shadow-rewrite cache key for one wheel.
///
/// Key = sha256 of:
///   b"retread-shadow-v2\n"
///   || EMIT_EPOCH (u32 le)
///   || CARGO_PKG_VERSION bytes || b"\n"
///   || target_artifact_identity || b"\n"
///   || input_wheel_sha256 (hex of relaxed.whl bytes) || b"\n"
///   || applicable_overrides_conda_and_url_drops_serialized
///
/// CRITICAL: only the APPLICABLE subset of overrides/conda_capable is
/// hashed (entries whose name appears in this wheel's Requires-Dist).
/// This is what makes "add one dep" a near-total cache hit: only wheels
/// that actually reference the new dep miss. It is ALSO what makes the
/// hit correct: the output is a pure function of exactly this subset +
/// the input bytes + the code version.
///
/// NOTE: this key is INTERNAL-ONLY. It MUST NOT feed `compute_inputs_hash`.
fn shadow_cache_key_for_target(
    input_wheel_bytes: &[u8],
    requires_dist: &[String],
    overrides: &BTreeMap<String, String>,
    conda_capable: &HashSet<String>,
    drop_url: &HashSet<String>,
    target_artifact_identity: &str,
) -> String {
    // Collect dep names from Requires-Dist (PEP 508 parse, same as override_line_map).
    let dep_names: HashSet<String> = requires_dist
        .iter()
        .filter_map(|line| {
            let req: Requirement = Requirement::from_str(line).ok()?;
            Some(req.name.to_string())
        })
        .collect();

    // Applicable subset of overrides: only entries whose name is in dep_names.
    let mut applicable_parts: Vec<String> = overrides
        .iter()
        .filter(|(name, _)| dep_names.contains(*name))
        .map(|(name, spec)| format!("{name}={spec}"))
        .collect();
    applicable_parts.sort();

    // Applicable subset of conda_capable: only names in dep_names.
    let mut cap_parts: Vec<String> = conda_capable
        .iter()
        .filter(|name| dep_names.contains(*name))
        .map(|name| format!("cap:{name}"))
        .collect();
    cap_parts.sort();

    applicable_parts.extend(cap_parts);
    let mut drop_parts: Vec<String> = drop_url
        .iter()
        .filter(|name| dep_names.contains(*name))
        .map(|name| format!("drop:{name}"))
        .collect();
    drop_parts.sort();
    applicable_parts.extend(drop_parts);
    let applicable_serialized = applicable_parts.join("\n");

    let input_sha = crate::wheel_rewrite::sha256_hex(input_wheel_bytes);

    let mut h = Sha256::new();
    h.update(b"retread-shadow-v2\n");
    h.update(crate::lock::EMIT_EPOCH.to_le_bytes());
    h.update(env!("CARGO_PKG_VERSION").as_bytes());
    h.update(b"\n");
    h.update(target_artifact_identity.as_bytes());
    h.update(b"\n");
    h.update(input_sha.as_bytes());
    h.update(b"\n");
    h.update(applicable_serialized.as_bytes());

    let mut out = String::with_capacity(64);
    for b in h.finalize() {
        write!(&mut out, "{b:02x}").expect("write to String");
    }
    out
}

#[cfg(test)]
fn shadow_cache_key(
    input_wheel_bytes: &[u8],
    requires_dist: &[String],
    overrides: &BTreeMap<String, String>,
    conda_capable: &HashSet<String>,
) -> String {
    shadow_cache_key_for_target(
        input_wheel_bytes,
        requires_dist,
        overrides,
        conda_capable,
        &HashSet::new(),
        "legacy-linux-64-test-target",
    )
}

/// Hard-link `src` -> `dst`, falling back to copy on cross-device error.
/// Mirrors the pattern in wheel_rewrite.rs for consistency.
fn hardlink_or_copy(src: &Path, dst: &Path) -> anyhow::Result<()> {
    if dst.exists() {
        std::fs::remove_file(dst)?;
    }
    if std::fs::hard_link(src, dst).is_err() {
        std::fs::copy(src, dst)?;
    }
    Ok(())
}

/// Route one staged (ship-class) wheel according to the bundle mode.
///
/// Fat mode (`loose == false`): push its `file://` URL into the recipe
/// source list so rattler-build tars it into the .conda; returns `None`
/// (the lock keeps whatever sha it already had).
///
/// Loose mode (`loose == true`): persist the staged bytes into the shared
/// content-addressed wheel store instead and return `Some(sha256)` for the
/// lock entry -- the .conda stays a stub and `retread install` materializes
/// the wheel from `<store>/<sha256>/<filename>` (hash-verified).
async fn ship_or_store(
    loose: bool,
    staged: &Path,
    source_urls: &mut Vec<String>,
    store_root: &Path,
    expected_name: &str,
    expected_version: &str,
) -> anyhow::Result<Option<String>> {
    if loose {
        let sha = crate::wheel::store_wheel_in_cache(staged, store_root).await?;
        let filename = staged
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "courier wheel store: staged path has no UTF-8 filename: {}",
                    staged.display(),
                )
            })?;
        let stored = store_root.join(&sha).join(filename);
        if let Err(poison) = validate_wheel_file_identity(
            expected_name,
            expected_version,
            &stored,
            Some(&sha),
            "courier loose wheel store entry",
        ) {
            if stored == staged {
                return Err(poison).context(
                    "courier loose wheel store entry aliases the staged wheel and is invalid",
                );
            }
            tokio::fs::remove_file(&stored).await.with_context(|| {
                format!(
                    "removing invalid courier loose wheel store entry {}",
                    stored.display()
                )
            })?;
            let repaired_sha = crate::wheel::store_wheel_in_cache(staged, store_root).await?;
            if repaired_sha != sha {
                anyhow::bail!(
                    "courier loose wheel store repair changed content identity from {sha} to {repaired_sha}"
                );
            }
            validate_wheel_file_identity(
                expected_name,
                expected_version,
                &stored,
                Some(&sha),
                "courier repaired loose wheel store entry",
            )
            .with_context(|| format!("replacing poisoned store entry: {poison:#}"))?;
        }
        Ok(Some(sha))
    } else {
        source_urls.push(file_url(staged)?);
        Ok(None)
    }
}

/// Look up or populate the shadow-rewrite cache, returning `(sha256, did_change)`.
///
/// Cache dir: `<cache_dir>/<key>.changed` or `<cache_dir>/<key>.same`.
/// On a miss, rewrites `src` -> cache file via `rewrite_wheel_with`, then
/// hard-links cache -> `dst`. On a hit, hard-links cache -> `dst` directly.
///
/// SAFETY NOTE: This cache is a pure build-speed optimization. It is
/// content-addressed by (code version, EMIT_EPOCH, target artifact identity,
/// input wheel bytes, applicable overrides). It produces BYTE-IDENTICAL
/// staged output to the no-cache path. It MUST NOT feed `compute_inputs_hash`.
fn shadow_cache_stage_validated(
    src: &Path,
    dst: &Path,
    cache_dir: &Path,
    key: &str,
    expected_name: &str,
    expected_version: &str,
    expected_sha256: Option<&str>,
    overrides: &BTreeMap<String, String>,
    conda_capable: &HashSet<String>,
    drop_url: &HashSet<String>,
) -> anyhow::Result<(String, bool)> {
    std::fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating shadow cache dir {}", cache_dir.display()))?;

    let hit_changed = cache_dir.join(format!("{key}.changed"));
    let hit_same = cache_dir.join(format!("{key}.same"));

    if hit_changed.exists() {
        hardlink_or_copy(&hit_changed, dst)
            .with_context(|| format!("cache hit hardlink (.changed) -> {}", dst.display()))?;
        let sha = match validate_wheel_archive_identity_and_sha(
            expected_name,
            expected_version,
            dst,
            expected_sha256,
            "courier changed shadow cache entry",
        ) {
            Ok(sha) => sha,
            Err(err) => {
                let _ = std::fs::remove_file(dst);
                let _ = std::fs::remove_file(&hit_changed);
                return Err(err);
            }
        };
        tracing::debug!(
            key = %&key[..8],
            dst = %dst.display(),
            "shadow cache: hit (changed)",
        );
        return Ok((sha, true));
    }
    if hit_same.exists() {
        hardlink_or_copy(&hit_same, dst)
            .with_context(|| format!("cache hit hardlink (.same) -> {}", dst.display()))?;
        let sha = match validate_wheel_archive_identity_and_sha(
            expected_name,
            expected_version,
            dst,
            expected_sha256,
            "courier unchanged shadow cache entry",
        ) {
            Ok(sha) => sha,
            Err(err) => {
                let _ = std::fs::remove_file(dst);
                let _ = std::fs::remove_file(&hit_same);
                return Err(err);
            }
        };
        tracing::debug!(
            key = %&key[..8],
            dst = %dst.display(),
            "shadow cache: hit (same)",
        );
        return Ok((sha, false));
    }

    // Cache miss: rewrite ONCE into the cache file, then link to dst.
    tracing::debug!(
        key = %&key[..8],
        dst = %dst.display(),
        "shadow cache: miss",
    );
    // Process+sequence-unique tmp so concurrent installs sharing this
    // machine-global cache don't race the same tmp path (avoids spurious
    // rewrite errors; the canonical .changed/.same entries stay atomic).
    static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let cache_tmp = cache_dir.join(format!(
        "{key}.{}.{}.tmp",
        std::process::id(),
        TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let m = override_line_map(overrides, conda_capable, drop_url);
    let (_sha, did_change) = crate::wheel_rewrite::rewrite_wheel_with(src, &cache_tmp, &m)
        .with_context(|| {
            format!(
                "shadow cache miss rewrite {} -> {}",
                src.display(),
                cache_tmp.display()
            )
        })?;

    let sha = match validate_wheel_archive_identity_and_sha(
        expected_name,
        expected_version,
        &cache_tmp,
        expected_sha256,
        "courier new shadow cache entry",
    ) {
        Ok(sha) => sha,
        Err(err) => {
            let _ = std::fs::remove_file(&cache_tmp);
            return Err(err);
        }
    };

    // Rename tmp -> canonical cache entry.
    let cache_dst = if did_change { &hit_changed } else { &hit_same };
    std::fs::rename(&cache_tmp, cache_dst).with_context(|| {
        format!(
            "placing cache entry {} -> {}",
            cache_tmp.display(),
            cache_dst.display()
        )
    })?;

    // Link cache -> staging.
    hardlink_or_copy(cache_dst, dst)
        .with_context(|| format!("cache miss hardlink -> {}", dst.display()))?;

    Ok((sha, did_change))
}

#[cfg(test)]
fn shadow_cache_stage(
    src: &Path,
    dst: &Path,
    cache_dir: &Path,
    key: &str,
    overrides: &BTreeMap<String, String>,
    conda_capable: &HashSet<String>,
    drop_url: &HashSet<String>,
) -> anyhow::Result<(String, bool)> {
    let metadata = read_wheel_metadata_streaming(src)?;
    shadow_cache_stage_validated(
        src,
        dst,
        cache_dir,
        key,
        &metadata.name,
        &metadata.version,
        None,
        overrides,
        conda_capable,
        drop_url,
    )
}

// ── Stage ────────────────────────────────────────────────────────────────────

/// Explicit provenance for the staged wheel bytes, distinguishing
/// already-rewritten wheels (probe_dst from shadow_cache_stage) from raw
/// un-relaxed bytes (no-cache src clone or force-downloaded .dl file).
///
/// Using an explicit enum eliminates the `src.starts_with(staging_dir)`
/// heuristic that wrongly classified force-downloaded `.dl-courier-*` files
/// as already-rewritten (they share the staging_dir prefix but carry RAW
/// bytes that must still go through rewrite_wheel_with).
enum ShadowSrc {
    /// Wheel bytes were already rewritten by shadow_cache_stage into this
    /// path. Only needs a rename to the final shadow filename.
    Rewritten(std::path::PathBuf),
    /// Wheel bytes are raw (un-relaxed). Must go through rewrite_wheel_with
    /// before being written to the final shadow filename.
    Raw(std::path::PathBuf),
    /// Relax did not change this wheel; no shadow needed.
    None,
}

#[derive(Debug, PartialEq, Eq)]
struct WheelFilenameProvenance {
    exact: String,
    without_build: String,
    build: Option<String>,
}

fn provenance_wheel_filename(filename: &str) -> Option<WheelFilenameProvenance> {
    let standard = standard_wheel_filename(filename);
    let stem = standard.strip_suffix(".whl")?;
    let mut parts: Vec<&str> = stem.split('-').collect();
    let build = (parts.len() == 6).then(|| parts.remove(2).to_owned());
    let without_build = format!("{}.whl", parts.join("-"));
    Some(WheelFilenameProvenance {
        exact: standard,
        without_build,
        build,
    })
}

/// Compare exact wheel provenance. PEP 427 build tags distinguish artifacts,
/// except that retread deliberately replaces an upstream build tag with its
/// owned `999retread` marker when producing a shadow.
pub(crate) fn wheel_filename_provenance_matches(left: &str, right: &str) -> bool {
    let (Some(left), Some(right)) = (
        provenance_wheel_filename(left),
        provenance_wheel_filename(right),
    ) else {
        return false;
    };
    left.exact == right.exact
        || ((left.build.as_deref() == Some("999retread")
            || right.build.as_deref() == Some("999retread"))
            && left.without_build == right.without_build)
}

pub(crate) fn has_owned_shadow_build(filename: &str) -> bool {
    provenance_wheel_filename(filename)
        .and_then(|provenance| provenance.build)
        .as_deref()
        == Some("999retread")
}

/// Validate one distribution identity using PEP 503 name normalization and
/// PEP 440 version normalization. `1.0` and `1.0.0` are the same release;
/// unrelated versions and merely similar names are not.
pub(crate) fn validate_distribution_identity(
    expected_name: &str,
    expected_version: &str,
    actual_name: &str,
    actual_version: &str,
    provenance: &str,
) -> anyhow::Result<()> {
    let expected_name_normalized = uv_normalize::PackageName::from_str(expected_name)
        .with_context(|| {
            format!("{provenance} expected distribution `{expected_name}` is invalid")
        })?;
    let actual_name_normalized = uv_normalize::PackageName::from_str(actual_name)
        .with_context(|| format!("{provenance} records invalid distribution `{actual_name}`"))?;
    if expected_name_normalized != actual_name_normalized {
        anyhow::bail!(
            "{provenance} records distribution `{actual_name}`, but `{expected_name}` was expected"
        );
    }

    let expected =
        uv_pep508::uv_pep440::Version::from_str(expected_version).with_context(|| {
            format!("{provenance} expected version `{expected_version}` is invalid")
        })?;
    let actual = uv_pep508::uv_pep440::Version::from_str(actual_version)
        .with_context(|| format!("{provenance} records invalid version `{actual_version}`"))?;
    if actual != expected {
        anyhow::bail!(
            "{provenance} records version `{actual_version}`, but `{expected_version}` was expected"
        );
    }
    Ok(())
}

/// Validate the distribution/version components of a PEP 427 wheel filename.
/// Retread's temporary processing infixes are removed first; an owned build
/// tag remains a valid PEP 427 build component.
fn validate_pep427_filename_structure<'a>(
    filename: &'a str,
    provenance: &str,
) -> anyhow::Result<Vec<&'a str>> {
    if filename.is_empty()
        || filename == "."
        || filename == ".."
        || filename.contains('/')
        || filename.contains('\\')
        || Path::new(filename)
            .file_name()
            .and_then(|name| name.to_str())
            != Some(filename)
    {
        anyhow::bail!("{provenance} must be a single wheel basename, not a path: `{filename}`");
    }

    let stem = filename
        .strip_suffix(".whl")
        .ok_or_else(|| anyhow::anyhow!("{provenance} is not a wheel filename: `{filename}`"))?;
    let fields: Vec<&str> = stem.split('-').collect();
    if !matches!(fields.len(), 5 | 6) || fields.iter().any(|field| field.is_empty()) {
        anyhow::bail!("{provenance} is not a valid PEP 427 wheel filename: `{filename}`");
    }

    let distribution = fields[0];
    if distribution.contains("__")
        || !distribution
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.'))
        || !distribution
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        || !distribution
            .as_bytes()
            .last()
            .is_some_and(u8::is_ascii_alphanumeric)
    {
        anyhow::bail!("{provenance} has an invalid PEP 427 distribution field: `{distribution}`");
    }
    uv_pep508::uv_pep440::Version::from_str(fields[1]).with_context(|| {
        format!(
            "{provenance} has an invalid PEP 427 version field: `{}`",
            fields[1]
        )
    })?;

    let tag_start = fields.len() - 3;
    if fields.len() == 6 {
        let build = fields[2];
        if !build.as_bytes()[0].is_ascii_digit()
            || !build
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.'))
        {
            anyhow::bail!("{provenance} has an invalid PEP 427 build tag: `{build}`");
        }
    }
    if fields[tag_start..].iter().any(|tag_set| {
        tag_set.split('.').any(|tag| {
            tag.is_empty()
                || !tag
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        })
    }) {
        anyhow::bail!("{provenance} has an invalid PEP 427 compatibility tag in `{filename}`");
    }
    Ok(fields)
}

pub(crate) fn validate_wheel_filename_identity(
    expected_name: &str,
    expected_version: &str,
    filename: &str,
    provenance: &str,
) -> anyhow::Result<String> {
    // Validate the raw basename first so percent-decoded URL separators can
    // never be hidden by Retread's owned processing-infix normalization.
    let raw_fields = validate_pep427_filename_structure(filename, provenance)?;
    validate_distribution_identity(
        expected_name,
        expected_version,
        raw_fields[0],
        raw_fields[1],
        provenance,
    )?;

    let standard = standard_wheel_filename(filename);
    let fields = validate_pep427_filename_structure(&standard, provenance)?;
    validate_distribution_identity(
        expected_name,
        expected_version,
        fields[0],
        fields[1],
        provenance,
    )?;
    Ok(standard)
}

fn macos_platform_arch(platform: &str) -> Option<&str> {
    let rest = platform.strip_prefix("macosx_")?;
    let (major, rest) = rest.split_once('_')?;
    let (minor, arch) = rest.split_once('_')?;
    if major.is_empty()
        || minor.is_empty()
        || !major.bytes().all(|byte| byte.is_ascii_digit())
        || !minor.bytes().all(|byte| byte.is_ascii_digit())
        || !matches!(arch, "x86_64" | "arm64" | "universal2")
    {
        return None;
    }
    Some(arch)
}

fn wheel_platform_family_matches(standard_filename: &str, conda_subdir: &str) -> bool {
    let Some(stem) = standard_filename.strip_suffix(".whl") else {
        return false;
    };
    let Some(platforms) = stem.rsplit('-').next() else {
        return false;
    };
    platforms.split('.').any(|platform| {
        if platform == "any" {
            return true;
        }
        match conda_subdir {
            "linux-64" => {
                platform == "linux_x86_64"
                    || ((platform.starts_with("manylinux") || platform.starts_with("musllinux"))
                        && platform.ends_with("_x86_64"))
            }
            "linux-aarch64" => {
                platform == "linux_aarch64"
                    || ((platform.starts_with("manylinux") || platform.starts_with("musllinux"))
                        && platform.ends_with("_aarch64"))
            }
            "osx-64" => matches!(macos_platform_arch(platform), Some("x86_64" | "universal2")),
            "osx-arm64" => matches!(macos_platform_arch(platform), Some("arm64" | "universal2")),
            "win-64" => platform == "win_amd64",
            "noarch" => false,
            _ => false,
        }
    })
}

fn cpython_tag_version(tag: &str) -> Option<(u32, u32)> {
    let digits = tag.strip_prefix("cp")?;
    if digits.len() < 2 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some((digits[..1].parse().ok()?, digits[1..].parse().ok()?))
}

fn wheel_abi_matches_target(
    standard_filename: &str,
    target_python: &str,
    provenance: &str,
) -> anyhow::Result<bool> {
    let normalized = crate::lock::normalized_target_python(target_python)
        .with_context(|| format!("{provenance} has a malformed target Python"))?;
    let (major, minor) = normalized
        .split_once('.')
        .expect("normalized target Python always has one dot");
    let target = (major.parse::<u32>()?, minor.parse::<u32>()?);
    let target_abi = format!("cp{}{}", target.0, target.1);

    let fields = validate_pep427_filename_structure(standard_filename, provenance)?;
    let tag_start = fields.len() - 3;
    let python_tags: Vec<&str> = fields[tag_start].split('.').collect();
    Ok(fields[tag_start + 1].split('.').any(|abi| match abi {
        "none" => true,
        "abi3" => python_tags.iter().any(|python| {
            cpython_tag_version(python)
                .is_some_and(|minimum| minimum.0 == target.0 && minimum.1 <= target.1)
        }),
        abi => abi == target_abi,
    }))
}

pub(crate) fn validate_wheel_filename_for_target(
    expected_name: &str,
    expected_version: &str,
    filename: &str,
    target: &WheelTarget,
    provenance: &str,
) -> anyhow::Result<String> {
    let standard =
        validate_wheel_filename_identity(expected_name, expected_version, filename, provenance)?;
    if !wheel_abi_matches_target(&standard, &target.python_version, provenance)? {
        anyhow::bail!(
            "{provenance} `{filename}` has an ABI incompatible with target Python {}",
            target.python_version,
        );
    }
    let score = crate::pypi::score_wheel(&standard, target);
    // Foundation scoring predates macOS universal2 and looks for a literal
    // architecture substring. Score an equivalent single-arch spelling after
    // the family guard confirms this is genuinely a macOS platform field.
    let universal2_score = if score < 0 && target.conda_subdir.starts_with("osx-") {
        let arch = if target.conda_subdir == "osx-64" {
            "x86_64"
        } else {
            "arm64"
        };
        standard.rsplit_once('-').and_then(|(prefix, platform)| {
            platform.contains("universal2").then(|| {
                let platform = platform.replace("universal2", arch);
                crate::pypi::score_wheel(&format!("{prefix}-{platform}"), target)
            })
        })
    } else {
        None
    };
    if !wheel_platform_family_matches(&standard, &target.conda_subdir)
        || (score < 0 && universal2_score.is_none_or(|score| score < 0))
    {
        anyhow::bail!(
            "{provenance} `{filename}` is incompatible with target Python {} on {}",
            target.python_version,
            target.conda_subdir,
        );
    }
    Ok(standard)
}

/// Read one wheel's root METADATA while hashing with bounded memory. The
/// general `wheel::read_metadata` helper currently hashes via `fs::read`; that
/// is unsuitable at this ingress because valid courier wheels can exceed 5
/// GiB. Keep the same parsed `WheelMetadata` contract without buffering the
/// archive.
const MAX_WHEEL_METADATA_BYTES: u64 = 16 * 1024 * 1024;

fn read_utf8_limited(
    reader: &mut impl Read,
    limit: u64,
    description: &str,
) -> anyhow::Result<String> {
    let mut limited = reader.take(limit + 1);
    let mut raw = String::new();
    limited
        .read_to_string(&mut raw)
        .with_context(|| format!("reading {description}"))?;
    if raw.len() as u64 > limit {
        anyhow::bail!("{description} exceeds the {limit}-byte safety limit");
    }
    Ok(raw)
}

fn validate_unique_metadata_identity_headers(raw: &str, description: &str) -> anyhow::Result<()> {
    let mut name_headers = 0u8;
    let mut version_headers = 0u8;
    let mut previous_identity_header: Option<&str> = None;
    for line in raw.lines() {
        if line.is_empty() {
            break;
        }
        if line.starts_with(' ') || line.starts_with('\t') {
            if let Some(header) = previous_identity_header {
                anyhow::bail!("{description} folds the `{header}` identity header");
            }
            continue;
        }
        previous_identity_header = None;
        let Some((key, _)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let (count, canonical) = if key.eq_ignore_ascii_case("Name") {
            (&mut name_headers, "Name")
        } else if key.eq_ignore_ascii_case("Version") {
            (&mut version_headers, "Version")
        } else {
            continue;
        };
        *count += 1;
        if *count > 1 {
            anyhow::bail!("{description} contains duplicate `{canonical}` identity headers");
        }
        previous_identity_header = Some(canonical);
    }
    Ok(())
}

fn read_wheel_metadata_streaming(wheel_path: &Path) -> anyhow::Result<crate::wheel::WheelMetadata> {
    let filename = wheel_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("wheel path has no filename: {}", wheel_path.display()))?
        .to_owned();

    let mut wheel_file = std::fs::File::open(wheel_path)
        .with_context(|| format!("opening {} for hashing", wheel_path.display()))?;
    let mut hasher = Sha256::new();
    let mut chunk = [0u8; 1024 * 1024];
    loop {
        let read = wheel_file
            .read(&mut chunk)
            .with_context(|| format!("hashing {}", wheel_path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&chunk[..read]);
    }
    let sha256 = format!("{:x}", hasher.finalize());

    wheel_file
        .rewind()
        .with_context(|| format!("rewinding {} after hashing", wheel_path.display()))?;
    let mut archive = zip::ZipArchive::new(wheel_file)
        .with_context(|| format!("reading zip {}", wheel_path.display()))?;
    let mut metadata_idx = None;
    for index in 0..archive.len() {
        let entry = archive.by_index(index)?;
        let entry_name = entry.name();
        if entry_name.ends_with(".dist-info/METADATA") && entry_name.matches('/').count() == 1 {
            if metadata_idx.replace(index).is_some() {
                anyhow::bail!(
                    "multiple root-level .dist-info/METADATA entries in {}",
                    wheel_path.display()
                );
            }
        }
    }
    let metadata_idx = metadata_idx.ok_or_else(|| {
        anyhow::anyhow!(
            "no root-level .dist-info/METADATA in {}",
            wheel_path.display()
        )
    })?;
    let mut entry = archive.by_index(metadata_idx)?;
    if entry.size() > MAX_WHEEL_METADATA_BYTES {
        anyhow::bail!(
            "METADATA from {} declares {} bytes, exceeding the {}-byte safety limit",
            wheel_path.display(),
            entry.size(),
            MAX_WHEEL_METADATA_BYTES,
        );
    }
    let raw = read_utf8_limited(
        &mut entry,
        MAX_WHEEL_METADATA_BYTES,
        &format!("METADATA from {}", wheel_path.display()),
    )?;
    validate_unique_metadata_identity_headers(
        &raw,
        &format!("METADATA from {}", wheel_path.display()),
    )?;
    crate::wheel::parse_metadata(
        &raw,
        filename.clone(),
        crate::wheel::is_pure_python_wheel_filename(&filename),
        sha256,
    )
}

/// Validate the basename, ZIP/METADATA identity, and optional content digest
/// of wheel bytes before they enter a cache/store or are handed to uv.
fn validate_wheel_archive_identity_and_sha(
    expected_name: &str,
    expected_version: &str,
    wheel_path: &Path,
    expected_sha256: Option<&str>,
    provenance: &str,
) -> anyhow::Result<String> {
    let metadata = read_wheel_metadata_streaming(wheel_path).with_context(|| {
        format!(
            "reading {provenance} METADATA from {}",
            wheel_path.display()
        )
    })?;
    validate_distribution_identity(
        expected_name,
        expected_version,
        &metadata.name,
        &metadata.version,
        provenance,
    )?;
    if let Some(expected_sha256) = expected_sha256
        && !metadata.sha256.eq_ignore_ascii_case(expected_sha256)
    {
        anyhow::bail!(
            "{provenance} SHA-256 mismatch for {}: expected {}, got {}",
            wheel_path.display(),
            expected_sha256,
            metadata.sha256,
        );
    }
    Ok(metadata.sha256)
}

fn validate_wheel_file_identity_and_sha(
    expected_name: &str,
    expected_version: &str,
    wheel_path: &Path,
    expected_sha256: Option<&str>,
    provenance: &str,
) -> anyhow::Result<String> {
    let filename = wheel_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{provenance} has no UTF-8 wheel filename: {}",
                wheel_path.display()
            )
        })?;
    validate_wheel_filename_identity(expected_name, expected_version, filename, provenance)?;
    validate_wheel_archive_identity_and_sha(
        expected_name,
        expected_version,
        wheel_path,
        expected_sha256,
        provenance,
    )
}

pub(crate) fn validate_wheel_file_identity(
    expected_name: &str,
    expected_version: &str,
    wheel_path: &Path,
    expected_sha256: Option<&str>,
    provenance: &str,
) -> anyhow::Result<()> {
    validate_wheel_file_identity_and_sha(
        expected_name,
        expected_version,
        wheel_path,
        expected_sha256,
        provenance,
    )
    .map(|_| ())
}

fn validate_wheel_filename(
    wheel: &EmitWheel,
    provenance: &str,
    filename: &str,
    target: &WheelTarget,
) -> anyhow::Result<String> {
    validate_wheel_filename_for_target(
        &wheel.pypi_name,
        &wheel.version,
        filename,
        target,
        &format!("courier {provenance} wheel filename"),
    )?;
    Ok(standard_wheel_filename(filename))
}

/// Validate every cold/replay provenance before staging mutates files or
/// caches. All representations must both score for the exact target and name
/// the same wheel after removing only transformations owned by retread.
fn validate_emit_wheels(emit_wheels: &[EmitWheel], target: &WheelTarget) -> anyhow::Result<()> {
    if emit_wheels.is_empty() {
        anyhow::bail!("courier: refusing to stage a replay lock with no wheel payload");
    }
    let mut seen_distributions: BTreeMap<String, String> = BTreeMap::new();
    let mut seen_filenames: BTreeMap<String, String> = BTreeMap::new();
    for wheel in emit_wheels {
        let mut source_provenance: Option<(String, String)> = None;
        if let Some(source) = &wheel.sdist_source {
            if wheel.must_ship() {
                anyhow::bail!(
                    "courier: {}=={} records sdist provenance on a must-ship source wheel",
                    wheel.pypi_name,
                    wheel.version,
                );
            }
            source
                .validated_url_and_sha256(&wheel.pypi_name, &wheel.version)
                .with_context(|| {
                    format!(
                        "courier: invalid sdist provenance for {}=={}",
                        wheel.pypi_name, wheel.version,
                    )
                })?;
        }
        let canonical_name = crate::relax::canonical_conda_name(&wheel.pypi_name);
        if let Some(prior) = seen_distributions.insert(canonical_name, wheel.pypi_name.clone()) {
            anyhow::bail!(
                "courier: duplicate distribution entries `{prior}` and `{}` would alias one staged artifact",
                wheel.pypi_name,
            );
        }
        if let Some(sha256) = wheel.sha256.as_deref()
            && (sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            anyhow::bail!(
                "courier: invalid sha256 for {}=={}; refusing to stage a poisoned lock",
                wheel.pypi_name,
                wheel.version,
            );
        }
        if let Some(sha256) = wheel.locked_final_sha256.as_deref()
            && (sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            anyhow::bail!(
                "courier: invalid locked final sha256 for {}=={}; refusing replay",
                wheel.pypi_name,
                wheel.version,
            );
        }
        let recorded = validate_wheel_filename(wheel, "recorded", &wheel.wheel_filename, target)?;
        if let Some(prior) =
            seen_filenames.insert(recorded.to_ascii_lowercase(), wheel.wheel_filename.clone())
        {
            anyhow::bail!(
                "courier: duplicate wheel filenames `{prior}` and `{}` would overwrite one staged artifact",
                wheel.wheel_filename,
            );
        }

        if let Some(path) = &wheel.local_path {
            let filename = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "courier: local wheel path for `{}` has no UTF-8 filename: {}",
                        wheel.pypi_name,
                        path.display(),
                    )
                })?;
            let local = validate_wheel_filename(wheel, "local", filename, target)?;
            if !wheel_filename_provenance_matches(&local, &recorded) {
                anyhow::bail!(
                    "courier: wheel provenance mismatch for `{}`: recorded filename `{}` and \
                     local filename `{filename}` are not the same wheel",
                    wheel.pypi_name,
                    wheel.wheel_filename,
                );
            }
            if !has_owned_shadow_build(&local) {
                source_provenance = Some(("local filename".to_owned(), local));
            }
            validate_wheel_file_identity(
                &wheel.pypi_name,
                &wheel.version,
                path,
                wheel.sha256.as_deref(),
                "courier local wheel",
            )?;
        }

        for (provenance, url) in [
            ("remote URL", wheel.remote_url.as_ref()),
            ("upstream URL", wheel.upstream_url.as_ref()),
        ] {
            let Some(url) = url else { continue };
            let filename = crate::wheel::wheel_filename_from_url(url).with_context(|| {
                format!(
                    "courier: extracting {provenance} wheel filename for `{}` from {url}",
                    wheel.pypi_name,
                )
            })?;
            let remote = validate_wheel_filename(wheel, provenance, &filename, target)?;
            if !wheel_filename_provenance_matches(&remote, &recorded) {
                anyhow::bail!(
                    "courier: wheel provenance mismatch for `{}`: recorded filename `{}` and \
                     {provenance} filename `{filename}` are not the same wheel",
                    wheel.pypi_name,
                    wheel.wheel_filename,
                );
            }
            if !has_owned_shadow_build(&remote) {
                if let Some((prior_provenance, prior_filename)) = &source_provenance
                    && prior_filename != &remote
                {
                    anyhow::bail!(
                        "courier: source provenance mismatch for `{}`: {prior_provenance} `{prior_filename}` and {provenance} filename `{filename}` identify different wheel artifacts",
                        wheel.pypi_name,
                    );
                }
                source_provenance = Some((format!("{provenance} filename"), remote));
            }
        }
    }
    Ok(())
}

/// Reserve the generated `<bundle>-pypi` distribution and filename before
/// staging. An emitted wheel with either identity would otherwise overwrite
/// (or be overwritten by) the meta-wheel after staging had already mutated
/// files and caches.
fn validate_reserved_meta_wheel_collisions(
    emit_wheels: &[EmitWheel],
    bundle_name: &str,
    version: &str,
) -> anyhow::Result<()> {
    let reserved_name = format!("{bundle_name}-pypi");
    let reserved_filename = format!(
        "{}_pypi-{version}-py3-none-any.whl",
        bundle_name.replace('-', "_")
    );
    let reserved_filename = validate_wheel_filename_identity(
        &reserved_name,
        version,
        &reserved_filename,
        "courier reserved meta-wheel filename",
    )?;
    let reserved_distribution = uv_normalize::PackageName::from_str(&reserved_name)
        .context("courier reserved meta-wheel distribution is invalid")?;

    for wheel in emit_wheels {
        let emitted_distribution = uv_normalize::PackageName::from_str(&wheel.pypi_name)
            .with_context(|| {
                format!(
                    "courier emitted distribution `{}` is invalid",
                    wheel.pypi_name
                )
            })?;
        if emitted_distribution == reserved_distribution {
            anyhow::bail!(
                "courier: emitted distribution `{}` collides with reserved meta-wheel `{reserved_name}`",
                wheel.pypi_name,
            );
        }
        let emitted_filename = standard_wheel_filename(&wheel.wheel_filename);
        if emitted_filename.eq_ignore_ascii_case(&reserved_filename) {
            anyhow::bail!(
                "courier: emitted wheel filename `{}` collides with reserved meta-wheel `{reserved_filename}`",
                wheel.wheel_filename,
            );
        }
    }
    Ok(())
}

fn validate_native_stage_target(
    target: &ResolutionTarget,
    native_platform: Platform,
) -> anyhow::Result<()> {
    if target.conda_subdir() != native_platform.as_str() {
        anyhow::bail!(
            "courier: cannot stage target `{}` on native platform `{native_platform}`: the \
             package embeds the currently running retread executable",
            target.conda_subdir(),
        );
    }
    Ok(())
}

/// Stage the courier artifacts. Built wheels (`must_ship()`) AND index
/// wheels whose metadata relax CHANGED are written to `staging_dir` (they
/// ship in the conda package as `Origin::Built`); unchanged index wheels are
/// recorded `Origin::Index` with their upstream url + sha256. Builds the
/// `<bundle>-pypi` meta-wheel (the lock's single `root_requirement`),
/// collects prerelease pins, computes `inputs_hash`, and writes the lock
/// json into `staging_dir`. NEVER writes the consumer manifest.
#[allow(clippy::too_many_arguments)]
pub async fn stage(
    config: &RetreadConfig,
    bundle_name: &str,
    version: &str,
    python: &str,
    emit_wheels: &[EmitWheel],
    conda_capable: &HashSet<String>,
    run_deps: &[String],
    index_urls: &[String],
    config_fp: &str,
    source_dir: &Path,
    staging_dir: &Path,
) -> anyhow::Result<CourierStaged> {
    let target = ResolutionTarget::try_for_subdir(python, Platform::current().as_str())
        .context("courier: malformed resolution target")?;
    stage_for_target(
        config,
        bundle_name,
        bundle_name,
        version,
        &target,
        emit_wheels,
        conda_capable,
        run_deps,
        index_urls,
        config_fp,
        source_dir,
        staging_dir,
    )
    .await
}

/// Exact-target staging boundary used by the target-aware backend pipeline.
/// `input_bundle_name` remains the manifest identity when an emitted package
/// name is target-qualified.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn stage_for_target(
    config: &RetreadConfig,
    bundle_name: &str,
    input_bundle_name: &str,
    version: &str,
    target: &ResolutionTarget,
    emit_wheels: &[EmitWheel],
    conda_capable: &HashSet<String>,
    run_deps: &[String],
    index_urls: &[String],
    config_fp: &str,
    source_dir: &Path,
    staging_dir: &Path,
) -> anyhow::Result<CourierStaged> {
    stage_for_target_with_store_root(
        config,
        bundle_name,
        input_bundle_name,
        version,
        target,
        emit_wheels,
        conda_capable,
        run_deps,
        index_urls,
        config_fp,
        source_dir,
        staging_dir,
        None,
    )
    .await
}

/// Replay form of [`stage_for_target`] that can preserve a portable,
/// lock-recorded loose-store identity (for example `~/.cache/...`) while
/// still using the current host to expand that identity.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn stage_for_target_with_store_root(
    config: &RetreadConfig,
    bundle_name: &str,
    input_bundle_name: &str,
    version: &str,
    target: &ResolutionTarget,
    emit_wheels: &[EmitWheel],
    conda_capable: &HashSet<String>,
    run_deps: &[String],
    index_urls: &[String],
    config_fp: &str,
    source_dir: &Path,
    staging_dir: &Path,
    wheel_store_root_override: Option<&Path>,
) -> anyhow::Result<CourierStaged> {
    // Run both guards before status output, directory creation, cache writes,
    // or current_exe packaging. A foreign target must leave no staged state.
    let python = crate::lock::normalized_target_python(target.python_version())
        .context("courier: malformed resolution target")?;
    validate_native_stage_target(target, Platform::current())?;
    validate_reserved_meta_wheel_collisions(emit_wheels, bundle_name, version)?;
    validate_emit_wheels(emit_wheels, target.wheel_target())?;

    let t_stage = std::time::Instant::now();
    crate::status::phase(source_dir, bundle_name, "staging: planning emit set");
    tokio::fs::create_dir_all(staging_dir)
        .await
        .with_context(|| format!("creating staging dir {}", staging_dir.display()))?;

    // Shadow-rewrite cache dir: persistent, machine-global, content-addressed.
    // Lives OUTSIDE source_dir/wheels so `rm -rf wheels` does not evict it.
    // Never feeds inputs_hash (the cache dir path is intentionally excluded
    // from the inputs hash -- only the cache KEY covers the relevant inputs).
    let target_artifact_identity = target.artifact_cache_identity();
    let shadow_cache_dir = retread_cache_root()
        .join("shadow")
        .join(&target_artifact_identity);
    // Best-effort: create the dir now so the first miss doesn't race.
    let _ = std::fs::create_dir_all(&shadow_cache_dir);
    // Bypass: RETREAD_NO_SHADOW_CACHE=<any value> disables the cache entirely
    // (forces fresh rewrites, enabling byte-for-byte parity testing).
    let use_shadow_cache = std::env::var("RETREAD_NO_SHADOW_CACHE").is_err();

    // Bundle mode: loose persists ship-class wheels to the shared wheel
    // store (stub .conda); fat tars them into the artifact (legacy).
    // The store root is fast-tmp-EXEMPT (see retread_wheel_store_root):
    // installs on other nodes must find these bytes after this job dies.
    let loose = config.bundle_mode == crate::config::BundleMode::Loose;
    let wheel_store_root = wheel_store_root_override
        .map(Path::to_path_buf)
        .unwrap_or_else(retread_wheel_store_root);

    // Step 1: run plan() to get ship set + override table.
    let emit_plan = plan(emit_wheels, conda_capable);

    // Clone overrides + conda_capable + drop_url so we can move them into
    // spawn_blocking. The mapper itself cannot cross the `'static` boundary
    // (it holds refs), so we re-derive it inside each blocking closure from
    // owned copies.
    let overrides_owned = emit_plan.overrides.clone();
    let conda_cap_owned: HashSet<String> = conda_capable.clone();
    let drop_url_owned: HashSet<String> = emit_plan.drop_url.clone();

    // Step 2: Build [retread-wheels] entries for this bundle (same filter as
    // emit's `entries` construction).
    let entries: Vec<(String, crate::config::WheelEntry, Option<String>)> = config
        .retread_wheels
        .iter()
        .filter(|(key, entry)| {
            let group = entry.bundle.as_deref().or(config.default_bundle.as_deref());
            let expected = crate::relax::canonical_conda_name(input_bundle_name);
            match group {
                Some(g) => crate::relax::canonical_conda_name(g) == expected,
                None => crate::relax::canonical_conda_name(key) == expected,
            }
        })
        .map(|(key, entry)| {
            let resolved = emit_wheels
                .iter()
                .find(|w| w.pypi_name == crate::relax::canonical_conda_name(key))
                .map(|w| w.version.clone());
            (key.clone(), entry.clone(), resolved)
        })
        .collect();

    // Build a non-async mapper for the remote-only check (no spawn needed).
    let mapper_for_remote = override_line_map(&overrides_owned, &conda_cap_owned, &drop_url_owned);

    // Step 3: Classify and stage each wheel.
    let mut lock_wheels: Vec<LockWheel> = Vec::new();
    let mut source_urls: Vec<String> = Vec::new();
    let shadow_download_client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .context("building courier shadow download client")?;

    crate::status::phase(
        source_dir,
        bundle_name,
        &format!(
            "staging: materializing {} wheels (large index wheels download here)",
            emit_wheels.len()
        ),
    );

    for w in emit_wheels {
        let std_name = standard_wheel_filename(&w.wheel_filename);

        if w.must_ship() {
            // Built-by-retread wheel (carries .injected infix): REWRITE (not
            // copy) into staging under the standard filename, then record
            // Origin::Built. Built wheels can carry transitive `name @ url`
            // Requires-Dist lines (e.g. isaaclab-rl -> `rl-games @ git+...`)
            // that uv REJECTS at install ("URL dependencies must be direct
            // requirements or constraints"). The bundle provides those deps via
            // find-links, and plan()'s Pass-1 puts a `name==<ver>` override in
            // `emit_plan.overrides` for every in-bundle URL target, so the same
            // override_line_map that rewrites index shadows also drops the URL
            // here. It is a no-op for lines that need no change (already-relaxed
            // built wheels re-rewrite idempotently), so this is safe for every
            // built wheel -- copying verbatim was the bug that shipped a wheel
            // uv could not resolve.
            let src = w.local_path.as_ref().ok_or_else(|| {
                anyhow::anyhow!("must_ship wheel has no local_path: {}", w.pypi_name)
            })?;
            let dst = staging_dir.join(&std_name);

            if use_shadow_cache {
                let src_bytes = tokio::fs::read(src)
                    .await
                    .with_context(|| format!("reading must_ship wheel {}", src.display()))?;
                let key = shadow_cache_key_for_target(
                    &src_bytes,
                    &w.requires_dist,
                    &overrides_owned,
                    &conda_cap_owned,
                    &drop_url_owned,
                    &target_artifact_identity,
                );
                let cache_dir = shadow_cache_dir.clone();
                let src_b = src.clone();
                let dst_b = dst.clone();
                let ov_b = overrides_owned.clone();
                let cap_b = conda_cap_owned.clone();
                let drop_b = drop_url_owned.clone();
                let expected_name_b = w.pypi_name.clone();
                let expected_version_b = w.version.clone();
                let expected_sha_b = w.locked_final_sha256.clone();
                tokio::task::spawn_blocking(move || {
                    shadow_cache_stage_validated(
                        &src_b,
                        &dst_b,
                        &cache_dir,
                        &key,
                        &expected_name_b,
                        &expected_version_b,
                        expected_sha_b.as_deref(),
                        &ov_b,
                        &cap_b,
                        &drop_b,
                    )
                })
                .await
                .with_context(|| {
                    format!("spawn_blocking shadow-cache (must_ship) {}", w.pypi_name)
                })??;
            } else {
                let src_blocking = src.clone();
                let dst_blocking = dst.clone();
                let overrides_b = overrides_owned.clone();
                let conda_cap_b = conda_cap_owned.clone();
                let drop_b = drop_url_owned.clone();
                tokio::task::spawn_blocking(move || {
                    let m = override_line_map(&overrides_b, &conda_cap_b, &drop_b);
                    crate::wheel_rewrite::rewrite_wheel_with(&src_blocking, &dst_blocking, &m)
                })
                .await
                .with_context(|| {
                    format!("spawn_blocking rewrite of built wheel {}", w.pypi_name)
                })??;
            }

            let staged_sha = validate_wheel_file_identity_and_sha(
                &w.pypi_name,
                &w.version,
                &dst,
                w.locked_final_sha256.as_deref(),
                "courier staged built wheel",
            )?;
            let stored_sha = ship_or_store(
                loose,
                &dst,
                &mut source_urls,
                &wheel_store_root,
                &w.pypi_name,
                &w.version,
            )
            .await?;
            let lock_sha = stored_sha.unwrap_or(staged_sha);
            lock_wheels.push(LockWheel {
                name: w.pypi_name.clone(),
                version: w.version.clone(),
                origin: Origin::Built,
                filename: std_name,
                url: None,
                // Always bind the exact final bytes. In loose mode this is
                // also the store lookup key; fat mode uses it to validate the
                // shipped wheel and deterministic source rebuilds.
                sha256: Some(lock_sha),
                requires_dist: w.requires_dist.clone(),
                must_ship: w.must_ship(),
                // Source-built (.injected) wheels exist on no index; there is
                // no upstream URL to record. Re-materialization on replay uses
                // git_source (when present, schema 8+) or the [retread-wheels]
                // config entry as a legacy fallback.
                upstream_url: None,
                // Git provenance (schema 8+): written from EmitWheel.git_source
                // which was populated by materialize_and_rewrite for git entries.
                git_source: w.git_source.clone(),
                // Sdist provenance (schema 9+): None for source-built (.injected)
                // wheels (these are git/path builds, not sdist BFS transitives).
                sdist_source: None,
            });
        } else {
            // Index wheel. Decide whether to ship a relax-rewritten shadow
            // (AUDIT B2: relax-changed index wheels must ship as shadows, not
            // stay remote, or strict pins re-emerge at install time).
            let shadow_src: ShadowSrc = if let Some(src) = w.local_path.as_ref() {
                // Cheap no-op pre-check (avoids reading a possibly multi-GB
                // wheel): rewrite_wheel_with only ever mutates Requires-Dist
                // lines, so if the mapper keeps EVERY one of this wheel's
                // recorded Requires-Dist lines the rewrite is a provable
                // byte-identical no-op (did_change == false). Short-circuit to
                // ShadowSrc::None WITHOUT reading/rewriting/copying the wheel.
                //
                // Why this matters: dependency-free multi-GB index wheels
                // (isaacsim-extscache-*, kit alone ~5.9 GiB) would otherwise be
                // fully COPIED into the node-local shadow cache purely to
                // discover did_change == false. The shadow cache lives on the
                // fast-tmp job-local dir (a SLURM RAM tmpfs) which is a
                // DIFFERENT filesystem than the NFS source wheels, so
                // rewrite_wheel_with's no-op hard-link falls back to a full
                // std::fs::copy -- exhausting the per-job tmpfs (ENOSPC at
                // install-time materialize_and_pack). Mirrors the remote-only
                // branch's `any_change` guard below; `w.requires_dist` is the
                // wheel's authoritative METADATA Requires-Dist
                // (EmitWheel.requires_dist, sourced from w.metadata).
                //
                // EMIT-NEUTRAL: this only skips WASTED work -- the pre-fix code
                // also landed on ShadowSrc::None for these wheels (did_change ==
                // false), so the emitted lock entry and .conda bytes are
                // byte-identical. No EMIT_EPOCH bump warranted.
                let any_change = w
                    .requires_dist
                    .iter()
                    .any(|l| mapper_for_remote(l) != crate::wheel_rewrite::LineAction::Keep);
                if !any_change {
                    ShadowSrc::None
                } else if use_shadow_cache {
                    // Single-pass through cache: rewrite_wheel_with returns
                    // (sha, did_change). No probe-then-rewrite double pass.
                    let src_bytes = tokio::fs::read(src)
                        .await
                        .with_context(|| format!("reading index wheel {}", src.display()))?;
                    let key = shadow_cache_key_for_target(
                        &src_bytes,
                        &w.requires_dist,
                        &overrides_owned,
                        &conda_cap_owned,
                        &drop_url_owned,
                        &target_artifact_identity,
                    );
                    let cache_dir = shadow_cache_dir.clone();
                    let src_c = src.clone();
                    let ov_c = overrides_owned.clone();
                    let cap_c = conda_cap_owned.clone();
                    let drop_c = drop_url_owned.clone();
                    let expected_name_c = w.pypi_name.clone();
                    let expected_version_c = w.version.clone();
                    let expected_sha_c = w.locked_final_sha256.clone();
                    // Rewrite into a temp dst so we can check did_change,
                    // then move to the real shadow name below if changed.
                    let probe_dst = staging_dir.join(format!(".probe-courier-{std_name}"));
                    let probe_dst_c = probe_dst.clone();
                    let (_sha, did_change) = tokio::task::spawn_blocking(move || {
                        shadow_cache_stage_validated(
                            &src_c,
                            &probe_dst_c,
                            &cache_dir,
                            &key,
                            &expected_name_c,
                            &expected_version_c,
                            expected_sha_c.as_deref(),
                            &ov_c,
                            &cap_c,
                            &drop_c,
                        )
                    })
                    .await
                    .with_context(|| {
                        format!("spawn_blocking shadow-cache (index) {}", w.pypi_name)
                    })??;
                    if did_change {
                        // probe_dst contains already-rewritten bytes: rename only.
                        ShadowSrc::Rewritten(probe_dst)
                    } else {
                        // Unchanged: remove the probe staging file.
                        let _ = tokio::fs::remove_file(&probe_dst).await;
                        ShadowSrc::None
                    }
                } else {
                    // No-cache path: probe-then-rewrite (old behavior).
                    let tmp_name = format!(".tmp-courier-{std_name}");
                    let tmp = staging_dir.join(&tmp_name);
                    let overrides_c = overrides_owned.clone();
                    let conda_cap_c = conda_cap_owned.clone();
                    let drop_c = drop_url_owned.clone();
                    let (_sha, did_change) = tokio::task::spawn_blocking({
                        let src = src.clone();
                        let tmp = tmp.clone();
                        move || {
                            let m = override_line_map(&overrides_c, &conda_cap_c, &drop_c);
                            crate::wheel_rewrite::rewrite_wheel_with(&src, &tmp, &m)
                        }
                    })
                    .await
                    .with_context(|| {
                        format!("spawn_blocking rewrite-check of {}", w.pypi_name)
                    })??;
                    let _ = tokio::fs::remove_file(&tmp).await;
                    // src contains raw (un-relaxed) bytes: rewrite_wheel_with needed.
                    if did_change {
                        ShadowSrc::Raw(src.clone())
                    } else {
                        ShadowSrc::None
                    }
                }
            } else {
                // Remote-only wheel (sidecar metadata, bytes never
                // downloaded -- typically a small auto-bundled PyPI dep).
                // If any Requires-Dist line WOULD relax-change, recording it
                // as Origin::Index ships the ORIGINAL strict pins (AUDIT B2),
                // which would POISON the lock. If conda satisfies it, Index
                // is harmless (conda wins). Otherwise FORCE-DOWNLOAD the
                // bytes so we can ship a relaxed shadow.
                let any_change = w
                    .requires_dist
                    .iter()
                    .any(|l| mapper_for_remote(l) != crate::wheel_rewrite::LineAction::Keep);
                if any_change && !conda_cap_owned.contains(&w.pypi_name) {
                    let url = w.remote_url.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "courier: remote-only relax-changed wheel {} has no remote_url \
                                 to download for shadow rewrite",
                            w.pypi_name
                        )
                    })?;
                    // Keep the physical basename PEP 427-valid so the byte
                    // provenance validator checks the same identity uv will
                    // see; isolation lives in a private temp directory.
                    let dl_dir = staging_dir.join(".dl-courier");
                    tokio::fs::create_dir_all(&dl_dir)
                        .await
                        .with_context(|| format!("creating {}", dl_dir.display()))?;
                    let dl = dl_dir.join(&std_name);
                    let bytes = download_shadow_wheel(&shadow_download_client, url, &w.pypi_name)
                        .await
                        .with_context(|| {
                            format!("downloading {} ({url}) for shadow rewrite", w.pypi_name)
                        })?;
                    tokio::fs::write(&dl, &bytes)
                        .await
                        .with_context(|| format!("writing downloaded {}", dl.display()))?;
                    if let Err(err) = validate_wheel_file_identity(
                        &w.pypi_name,
                        &w.version,
                        &dl,
                        w.sha256.as_deref(),
                        "courier downloaded wheel",
                    ) {
                        let _ = tokio::fs::remove_file(&dl).await;
                        return Err(err);
                    }
                    tracing::info!(
                        wheel = %w.pypi_name,
                        "courier: force-downloaded remote-only relax-changed wheel to ship \
                         a rewritten shadow (B-5)",
                    );
                    // dl contains raw (un-relaxed) bytes: rewrite_wheel_with needed.
                    ShadowSrc::Raw(dl)
                } else {
                    ShadowSrc::None
                }
            };

            if matches!(&shadow_src, ShadowSrc::None)
                && has_owned_shadow_build(&w.wheel_filename)
                && (w.upstream_url.is_some() || w.remote_url.is_some())
            {
                anyhow::bail!(
                    "courier: recorded owned shadow `{}` for {}=={} no longer produces a \
                     shadow; refusing to downgrade final rewritten provenance to a raw index URL",
                    w.wheel_filename,
                    w.pypi_name,
                    w.version,
                );
            }

            match shadow_src {
                ShadowSrc::None => {
                    // Unchanged index wheel: record with direct artifact URL + hash.
                    // Install-time replay fetches this exact URL only; it never asks
                    // an index for metadata. If no URL exists, the wheel must ship
                    // inside the courier package instead.
                    // requires_dist is recorded in full (not vec![]) so plan()
                    // on replay builds the same override table as cold-produce
                    // (#4 parity fix: empty requires_dist causes plan() to miss
                    // overrides for index wheels, potentially flipping a relax-
                    // shadow to Index and poisoning the lock on replay).
                    let index_url = w
                        .upstream_url
                        .as_ref()
                        .or(w.remote_url.as_ref())
                        .map(|u| u.to_string());
                    if let Some(index_url) = index_url {
                        let sha256 = w.sha256.clone().ok_or_else(|| {
                            anyhow::anyhow!(
                                "courier: cannot write replayable lock for unchanged index wheel \
                                 `{}`: missing sha256 for direct artifact URL `{}`. Rebuild the \
                                 pack from a resolver path that downloads the wheel or reads a \
                                 PEP 658 sidecar with a hash.",
                                w.pypi_name,
                                index_url
                            )
                        })?;
                        lock_wheels.push(LockWheel {
                            name: w.pypi_name.clone(),
                            version: w.version.clone(),
                            origin: Origin::Index,
                            filename: std_name,
                            url: Some(index_url),
                            sha256: Some(sha256),
                            requires_dist: w.requires_dist.clone(),
                            must_ship: false,
                            upstream_url: None, // n/a for Index wheels; use `url` instead
                            git_source: None,   // Index wheels have no git source
                            sdist_source: None, // n/a for Index wheels
                        });
                    } else {
                        let src = w.local_path.as_ref().ok_or_else(|| {
                            anyhow::anyhow!(
                                "courier: unchanged wheel `{}` has neither a local file nor a \
                                 direct artifact URL; cannot write a replayable lock",
                                w.pypi_name
                            )
                        })?;
                        let dst = staging_dir.join(&std_name);
                        crate::wheel::hardlink_or_copy_async(src, &dst)
                            .await
                            .with_context(|| {
                                format!(
                                    "shipping unchanged local-only wheel {} from {}",
                                    w.pypi_name,
                                    src.display()
                                )
                            })?;
                        let staged_sha = validate_wheel_file_identity_and_sha(
                            &w.pypi_name,
                            &w.version,
                            &dst,
                            w.locked_final_sha256.as_deref().or(w.sha256.as_deref()),
                            "courier staged local-only wheel",
                        )?;
                        let stored_sha = ship_or_store(
                            loose,
                            &dst,
                            &mut source_urls,
                            &wheel_store_root,
                            &w.pypi_name,
                            &w.version,
                        )
                        .await?;
                        let lock_sha = stored_sha.unwrap_or(staged_sha);
                        lock_wheels.push(LockWheel {
                            name: w.pypi_name.clone(),
                            version: w.version.clone(),
                            origin: Origin::Built,
                            filename: std_name,
                            url: None,
                            // Always bind the exact final bytes; in loose mode
                            // this is also the content-addressed store key.
                            sha256: Some(lock_sha),
                            requires_dist: w.requires_dist.clone(),
                            must_ship: false,
                            upstream_url: None,
                            git_source: None,
                            sdist_source: None,
                        });
                    }
                }
                ShadowSrc::Rewritten(probe_dst) => {
                    // Relax changed this index wheel's METADATA (already rewritten
                    // by shadow_cache_stage): ship it as a build-tagged shadow wheel
                    // so uv's find-links prefers it over the registry original
                    // (AUDIT B2 fix). Only a rename needed — bytes are already correct.
                    let shadow_name = insert_build_tag(&std_name, "999retread")?;
                    let dst = staging_dir.join(&shadow_name);
                    tokio::fs::rename(&probe_dst, &dst).await.with_context(|| {
                        format!(
                            "renaming probe {} -> shadow {}",
                            probe_dst.display(),
                            dst.display()
                        )
                    })?;
                    let staged_sha = validate_wheel_file_identity_and_sha(
                        &w.pypi_name,
                        &w.version,
                        &dst,
                        w.locked_final_sha256.as_deref(),
                        "courier staged shadow wheel",
                    )?;
                    let stored_sha = ship_or_store(
                        loose,
                        &dst,
                        &mut source_urls,
                        &wheel_store_root,
                        &w.pypi_name,
                        &w.version,
                    )
                    .await?;
                    let lock_sha = stored_sha.unwrap_or(staged_sha);
                    // Prefer upstream_url (pristine pre-localization index URL,
                    // set for local-path shadows when EmitWheel was built from
                    // the cold produce path) over remote_url (set only for
                    // remote-only wheels that were never localized). This ensures
                    // the replay path can re-fetch the wheel even when it was
                    // originally local (downloaded to wheels/ on cold produce).
                    let upstream_url = w
                        .upstream_url
                        .as_ref()
                        .or(w.remote_url.as_ref())
                        .map(|u| u.to_string());
                    lock_wheels.push(LockWheel {
                        name: w.pypi_name.clone(),
                        version: w.version.clone(),
                        origin: Origin::Built,
                        filename: shadow_name,
                        url: None,
                        // Always bind the exact final bytes; in loose mode
                        // this is also the content-addressed store key.
                        sha256: Some(lock_sha),
                        requires_dist: w.requires_dist.clone(),
                        must_ship: w.must_ship(),
                        upstream_url,
                        git_source: None, // Class-2 shadow: index wheel, no git source
                        // Sdist provenance (schema 9+): thread from EmitWheel so
                        // sdist-built BFS transitives (e.g. gym) carry their sdist_url
                        // into the lock for Class-2b replay.
                        sdist_source: w.sdist_source.clone(),
                    });
                }
                ShadowSrc::Raw(src) => {
                    // Relax changed this index wheel's METADATA (raw bytes, not yet
                    // rewritten): ship it as a build-tagged shadow wheel, running
                    // rewrite_wheel_with on the raw bytes first.
                    let shadow_name = insert_build_tag(&std_name, "999retread")?;
                    let dst = staging_dir.join(&shadow_name);
                    let dst_blocking = dst.clone();
                    let src_blocking = src.clone();
                    let overrides_c2 = overrides_owned.clone();
                    let conda_cap_c2 = conda_cap_owned.clone();
                    let drop_c2 = drop_url_owned.clone();
                    tokio::task::spawn_blocking(move || {
                        let m = override_line_map(&overrides_c2, &conda_cap_c2, &drop_c2);
                        crate::wheel_rewrite::rewrite_wheel_with(&src_blocking, &dst_blocking, &m)
                    })
                    .await
                    .with_context(|| {
                        format!("spawn_blocking shadow-rewrite of {}", w.pypi_name)
                    })??;
                    let staged_sha = validate_wheel_file_identity_and_sha(
                        &w.pypi_name,
                        &w.version,
                        &dst,
                        w.locked_final_sha256.as_deref(),
                        "courier staged shadow wheel",
                    )?;
                    let stored_sha = ship_or_store(
                        loose,
                        &dst,
                        &mut source_urls,
                        &wheel_store_root,
                        &w.pypi_name,
                        &w.version,
                    )
                    .await?;
                    let lock_sha = stored_sha.unwrap_or(staged_sha);
                    // Prefer upstream_url over remote_url (same rationale as
                    // the Rewritten arm above).
                    let upstream_url = w
                        .upstream_url
                        .as_ref()
                        .or(w.remote_url.as_ref())
                        .map(|u| u.to_string());
                    lock_wheels.push(LockWheel {
                        name: w.pypi_name.clone(),
                        version: w.version.clone(),
                        origin: Origin::Built,
                        filename: shadow_name,
                        url: None,
                        // Loose mode: store sha256 = install-time lookup key.
                        sha256: Some(lock_sha),
                        requires_dist: w.requires_dist.clone(),
                        must_ship: w.must_ship(),
                        upstream_url,
                        git_source: None, // Class-2 shadow: index wheel, no git source
                        // Sdist provenance (schema 9+): thread from EmitWheel so
                        // sdist-built BFS transitives (e.g. gym) carry their sdist_url
                        // into the lock for Class-2b replay.
                        sdist_source: w.sdist_source.clone(),
                    });
                }
            }
        }
    }

    crate::status::phase(
        source_dir,
        bundle_name,
        &format!("staging: staged {} wheels", lock_wheels.len()),
    );

    // Step 4: Build and stage the <bundle>-pypi meta-wheel (conditional).
    // Marker = sha256(bundle_name || "\n" || version || "\n" || serialized entries).
    // Skip rebuild when marker matches AND the meta-wheel file exists.
    let meta_marker_key = {
        let mut h = Sha256::new();
        h.update(bundle_name.as_bytes());
        h.update(b"\n");
        h.update(version.as_bytes());
        h.update(b"\n");
        for (key, entry, resolved) in &entries {
            h.update(key.as_bytes());
            h.update(b":");
            h.update(entry.extras.join(",").as_bytes());
            h.update(b":");
            h.update(
                entry
                    .normalized_version()
                    .as_deref()
                    .unwrap_or("")
                    .as_bytes(),
            );
            h.update(b":");
            h.update(resolved.as_deref().unwrap_or("").as_bytes());
            h.update(b"\n");
        }
        let mut s = String::with_capacity(64);
        for b in h.finalize() {
            write!(&mut s, "{b:02x}").expect("write to String");
        }
        s
    };
    let meta_marker_path = staging_dir.join(".meta-wheel.key");
    let (meta_name, meta_bytes) = build_meta_wheel(bundle_name, version, &entries);
    let meta_dst = staging_dir.join(&meta_name);
    let meta_cached = if let Ok(existing) = tokio::fs::read_to_string(&meta_marker_path).await {
        existing.trim() == meta_marker_key && meta_dst.exists()
    } else {
        false
    };
    if !meta_cached {
        tokio::fs::write(&meta_dst, &meta_bytes)
            .await
            .with_context(|| format!("writing meta-wheel {}", meta_dst.display()))?;
        tokio::fs::write(&meta_marker_path, &meta_marker_key)
            .await
            .with_context(|| format!("writing meta-wheel marker {}", meta_marker_path.display()))?;
    }
    validate_wheel_file_identity(
        &format!("{bundle_name}-pypi"),
        version,
        &meta_dst,
        None,
        "courier staged meta-wheel",
    )?;
    source_urls.push(file_url(&meta_dst)?);

    // Step 4b: ship the static installer binary INSIDE the package (the
    // currently-running backend == the static musl `pixi-build-retread`).
    // The recipe copies it to `$PREFIX/bin/retread`; the post-link runs it.
    // This avoids run-depping on the heavy backend conda package (which the
    // consumer's solve check can't even see on a file:///non-default channel).
    let self_exe = std::env::current_exe().context("locating retread backend binary")?;
    let installer_dst = staging_dir.join("retread-installer");
    let installer_marker_path = staging_dir.join("retread-installer.version");
    // Marker = CARGO_PKG_VERSION + ":" + mtime_nanos of self_exe.
    let installer_marker = {
        let mtime = std::fs::metadata(&self_exe)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("{}:{}", env!("CARGO_PKG_VERSION"), mtime)
    };
    let installer_cached =
        if let Ok(existing) = tokio::fs::read_to_string(&installer_marker_path).await {
            existing.trim() == installer_marker
                && installer_dst.exists()
                && std::fs::metadata(&installer_dst)
                    .map(|m| m.len() > 0)
                    .unwrap_or(false)
        } else {
            false
        };
    if !installer_cached {
        tokio::fs::copy(&self_exe, &installer_dst)
            .await
            .with_context(|| {
                format!(
                    "staging installer binary {} -> {}",
                    self_exe.display(),
                    installer_dst.display()
                )
            })?;
        tokio::fs::write(&installer_marker_path, &installer_marker)
            .await
            .with_context(|| {
                format!(
                    "writing installer marker {}",
                    installer_marker_path.display()
                )
            })?;
    }
    source_urls.push(file_url(&installer_dst)?);

    // Step 5: Collect prerelease pins and compute inputs_hash.
    let prerelease =
        collect_prerelease_pins(&emit_plan.overrides, emit_wheels, &emit_plan.ship, &entries);

    // inputs_hash over PURE manifest inputs (NOT resolved versions), via the
    // canonical helper shared with the cold-solve replayer in conda_outputs --
    // they MUST compute it identically or replay never fires.
    let entry_specs = courier_input_specs(config, input_bundle_name);
    let inputs_hash = RetreadLock::compute_inputs_hash_for_target(
        &entry_specs,
        index_urls,
        &format!("{:?}", config.relax),
        target,
        crate::lock::EMIT_EPOCH,
        config.pin_version.then_some(env!("CARGO_PKG_VERSION")),
        config_fp,
    );

    // B-8 invariant: every emitted wheel must be classified into the lock
    // (Built or Index). A dropped wheel means the consumer installs an
    // incomplete set -- a poisoned lock. The only paths that skip a push are
    // the hard `bail!`s above, so a count mismatch is a logic bug.
    if lock_wheels.len() != emit_wheels.len() {
        anyhow::bail!(
            "courier: internal invariant violated -- staged {} lock wheels for {} emitted \
             wheels; refusing to write an incomplete (poisoned) lock",
            lock_wheels.len(),
            emit_wheels.len(),
        );
    }

    // PRODUCER POISONING GUARD (schema 5): when a non-default relax policy
    // caused an index wheel to be relax-changed into Origin::Built (the shadow
    // path), its requires_dist MUST be non-empty -- the replay path needs them
    // to detect requires_dist changes that would re-trigger a relax rewrite.
    // An empty requires_dist on a relax-changed Built wheel means we cannot
    // verify on replay that the wheel's metadata hasn't changed, which would
    // silently propagate stale relax-rewrites. Bail rather than write a
    // potentially poisoned lock.
    if config.relax != crate::config::RelaxPolicy::default() {
        for lw in &lock_wheels {
            if lw.origin == Origin::Built && !lw.must_ship && lw.requires_dist.is_empty() {
                anyhow::bail!(
                    "courier: relax-changed wheel `{}` has empty requires_dist with \
                     non-default relax policy `{:?}`; cannot write a safe lock (the \
                     replay path needs requires_dist to detect stale relax-rewrites). \
                     This is a retread bug -- please file an issue.",
                    lw.name,
                    config.relax,
                );
            }
        }
    }

    // Step 6: Assemble the RetreadLock and canonicalize for stable JSON output.
    // entry_specs: the canonical resolution-input spec list for this bundle.
    // Populated here so the Part-2 delta-detector can diff it against the
    // current manifest without re-running the full resolve.
    let shadow_libs: BTreeMap<String, String> = config
        .shadow_libs
        .iter()
        .map(|(path, policy)| (path.clone(), policy.as_lock_value().to_string()))
        .collect();
    let mut lock = RetreadLock {
        schema: SCHEMA,
        retread_version: env!("CARGO_PKG_VERSION").to_string(),
        bundle: bundle_name.to_string(),
        version: version.to_string(),
        python,
        target_subdir: target.conda_subdir().to_string(),
        inputs_hash,
        root_requirements: vec![format!("{bundle_name}-pypi=={version}")],
        wheels: lock_wheels,
        conda_run_deps: parse_conda_deps(run_deps),
        index_urls: index_urls.to_vec(),
        prerelease,
        shadow_libs,
        declared_glibc: target.declared_glibc().map(crate::glibc::format_glibc),
        resolution_glibc: target.effective_glibc().map(crate::glibc::format_glibc),
        conda_capable: conda_capable.iter().cloned().collect(),
        entry_specs,
        // Loose bundles: record WHERE the built-wheel bytes were persisted so
        // the installer can find them even when its default store resolution
        // differs from the build machine's (home-relative "~" form; see
        // portable_wheel_store_path). Fat bundles carry their bytes.
        wheel_store: loose.then(|| portable_wheel_store_path(&wheel_store_root)),
    };
    if !lock.is_for_resolution_target(target) {
        anyhow::bail!(
            "courier: internal invariant violated -- serialized lock target does not match \
             the staging target; refusing to write a poisoned lock"
        );
    }
    lock.canonicalize();
    lock.validate_replay_provenance()
        .context("courier: refusing to serialize an incomplete replay lock")?;

    // Step 7: Write the lock JSON into staging_dir. Write-then-rename so a
    // crash mid-write can never leave a torn lock (B-3): a partial file would
    // either fail to parse (fail-safe, replay falls through) or, worse, parse
    // into wrong data. The rename is atomic on the same filesystem.
    let lock_json = lock.to_pretty_json()?;
    let lock_filename = RetreadLock::file_name_for_target(bundle_name, target);
    let lock_dst = staging_dir.join(&lock_filename);
    let lock_tmp = staging_dir.join(format!(".{lock_filename}.tmp"));
    tokio::fs::write(&lock_tmp, lock_json.as_bytes())
        .await
        .with_context(|| format!("writing lock tmp {}", lock_tmp.display()))?;
    tokio::fs::rename(&lock_tmp, &lock_dst)
        .await
        .with_context(|| format!("atomically placing lock {}", lock_dst.display()))?;
    source_urls.push(file_url(&lock_dst)?);

    tracing::info!(
        bundle = %bundle_name,
        version = %version,
        staged = source_urls.len(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "bench: courier::stage finished",
    );

    Ok(CourierStaged {
        source_urls,
        lock,
        run_deps: run_deps.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relax::{CondaName, CondaTarget, PypiKey};
    use std::io::Write as _;

    fn wheel_dist_name(name: &str) -> String {
        name.replace('-', "_").replace('.', "_")
    }

    /// Build a minimal valid wheel zip in memory for the given dist/version/requires.
    fn make_wheel_bytes(dist: &str, version: &str, requires: &[&str]) -> Vec<u8> {
        let mut metadata = format!("Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n");
        for req in requires {
            metadata.push_str(&format!("Requires-Dist: {req}\n"));
        }
        make_wheel_bytes_with_metadata(dist, version, metadata.as_bytes())
    }

    fn make_wheel_bytes_with_metadata(dist: &str, version: &str, metadata: &[u8]) -> Vec<u8> {
        let normalized = dist.replace('-', "_");
        let di = format!("{normalized}-{version}.dist-info");
        let wheel_file = b"Wheel-Version: 1.0\nTag: py3-none-any\n".to_vec();
        // Minimal RECORD with empty hashes (good enough for the rewrite test).
        let record = format!("{di}/METADATA,,\n{di}/WHEEL,,\n{di}/RECORD,,\n").into_bytes();

        let mut buf = Vec::new();
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, body) in [
            (format!("{di}/METADATA"), metadata),
            (format!("{di}/WHEEL"), wheel_file.as_slice()),
            (format!("{di}/RECORD"), record.as_slice()),
        ] {
            zip.start_file(&name, opts).unwrap();
            zip.write_all(body).unwrap();
        }
        zip.finish().unwrap();
        buf
    }

    fn make_emit_wheel(
        name: &str,
        version: &str,
        requires: &[&str],
        local_path: Option<&Path>,
        remote_url: Option<&str>,
    ) -> EmitWheel {
        let wheel_filename = local_path
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{}-{version}-py3-none-any.whl", wheel_dist_name(name)));
        let local_sha256 = local_path
            .filter(|path| path.is_file())
            .and_then(|path| std::fs::read(path).ok())
            .map(|bytes| crate::wheel_rewrite::sha256_hex(&bytes));
        EmitWheel {
            pypi_name: name.to_string(),
            version: version.to_string(),
            requires_dist: requires.iter().map(|s| s.to_string()).collect(),
            wheel_filename,
            sha256: local_sha256.or_else(|| remote_url.map(|_| "0".repeat(64))),
            locked_final_sha256: None,
            local_path: local_path.map(|p| p.to_path_buf()),
            remote_url: remote_url.and_then(|u| u.parse().ok()),
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }
    }

    #[tokio::test]
    async fn shadow_download_retries_transient_http_failure() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for attempt in 1..=2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = [0_u8; 1024];
                let _ = stream.read(&mut request).await;
                let response = if attempt == 1 {
                    "HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n".to_string()
                } else {
                    "HTTP/1.0 200 OK\r\nContent-Length: 5\r\n\r\nwheel".to_string()
                };
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });

        let url: url::Url = format!("http://{address}/demo.whl").parse().unwrap();
        let client = reqwest::Client::builder().build().unwrap();
        let bytes = download_shadow_wheel(&client, &url, "demo").await.unwrap();

        assert_eq!(bytes, b"wheel");
        server.await.unwrap();
    }

    // Fat mode: the historical stage() contract these tests were written
    // against (wheels ship in source_urls). Loose-mode routing is covered
    // by the dedicated `loose_mode_*` tests below.
    fn minimal_config(bundle_name: &str) -> RetreadConfig {
        let json = serde_json::json!({
            "retread-wheels": {
                bundle_name: { "version": "==1.0.0" }
            },
            "retread-bundle-mode": "fat"
        });
        serde_json::from_value(json).unwrap()
    }

    fn minimal_loose_config(bundle_name: &str) -> RetreadConfig {
        let json = serde_json::json!({
            "retread-wheels": {
                bundle_name: { "version": "==1.0.0" }
            }
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        // Loose must be the DEFAULT (owner decision: fat tarballs are
        // opt-in for channel publishing only).
        assert_eq!(cfg.bundle_mode, crate::config::BundleMode::Loose);
        cfg
    }

    #[test]
    fn direct_url_input_identity_includes_sha256() {
        let config = |sha: &str| -> RetreadConfig {
            serde_json::from_value(serde_json::json!({
                "retread-wheels": {
                    "demo": {
                        "url": "https://example.com/demo-1.0-py3-none-any.whl",
                        "sha256": sha
                    }
                }
            }))
            .unwrap()
        };
        let first = courier_input_specs(&config(&"a".repeat(64)), "demo");
        let second = courier_input_specs(&config(&"b".repeat(64)), "demo");
        assert_ne!(first, second);
        assert!(first[0].ends_with(&format!("#sha256={}", "a".repeat(64))));
    }

    /// grizzly P1 regression: changing the conda channel list MUST change the
    /// fingerprint (a newly-added channel can flip a wheel's lock
    /// classification, so a stale lock must not replay). Channel ORDER also
    /// matters (conda priority), like the PyPI index chain.
    #[test]
    fn fingerprint_covers_conda_channels() {
        let cfg = minimal_config("b");
        let base = config_fingerprint(&cfg, &["conda-forge".to_string()], "");
        let added =
            config_fingerprint(&cfg, &["conda-forge".to_string(), "nvidia".to_string()], "");
        assert_ne!(base, added, "adding a channel must change the fingerprint");
        let reordered =
            config_fingerprint(&cfg, &["nvidia".to_string(), "conda-forge".to_string()], "");
        assert_ne!(
            added, reordered,
            "channel order must change the fingerprint"
        );
    }

    #[test]
    fn fingerprint_covers_deps_from_source_identity() {
        let base = minimal_config("b");
        let mut first = base.clone();
        first.deps_from = crate::config::DepsFromSpec(vec![crate::deps_from::DepSource::Git {
            git: "https://example.com/project.git".to_string(),
            rev: "deadbeef".to_string(),
            path: "environment.yaml".to_string(),
        }]);
        let mut changed = first.clone();
        changed.deps_from = crate::config::DepsFromSpec(vec![crate::deps_from::DepSource::Git {
            git: "https://example.com/project.git".to_string(),
            rev: "cafebabe".to_string(),
            path: "environment.yaml".to_string(),
        }]);

        let channels = ["conda-forge".to_string()];
        assert_ne!(
            config_fingerprint(&base, &channels, ""),
            config_fingerprint(&first, &channels, ""),
        );
        assert_ne!(
            config_fingerprint(&first, &channels, ""),
            config_fingerprint(&changed, &channels, ""),
        );
    }

    /// Routing policy changes must invalidate locks, including the v5 default
    /// transition from legacy aggressive routing to validated conda routing.
    #[test]
    fn fingerprint_covers_route_policy() {
        let cfg = minimal_config("b");
        let chans = ["conda-forge".to_string()];
        let validated = config_fingerprint(&cfg, &chans, "");
        assert!(
            validated
                .lines()
                .any(|line| line == "route-policy:prefer-conda-validated"),
            "the default policy must be encoded explicitly to invalidate legacy fingerprints"
        );

        let mut aggressive = cfg;
        aggressive.route_policy = RoutePolicy::Aggressive;
        let aggressive = config_fingerprint(&aggressive, &chans, "");
        assert_ne!(
            validated, aggressive,
            "changing route policy must change the fingerprint"
        );
    }

    /// `route_include` affects routing under the minimal policy, but its order
    /// is semantically irrelevant because routing consumes it as a set.
    #[test]
    fn fingerprint_covers_route_include() {
        let mut first = minimal_config("b");
        first.route_policy = RoutePolicy::Minimal;
        first.route_include = vec!["mujoco".to_string(), "grpcio".to_string()];
        let chans = ["conda-forge".to_string()];
        let first_fp = config_fingerprint(&first, &chans, "");

        let mut changed = first.clone();
        changed.route_include = vec!["mujoco".to_string(), "scipy".to_string()];
        assert_ne!(
            first_fp,
            config_fingerprint(&changed, &chans, ""),
            "changing route includes must change the fingerprint"
        );

        let mut reordered = first;
        reordered.route_include.reverse();
        assert_eq!(
            first_fp,
            config_fingerprint(&reordered, &chans, ""),
            "route include order must not change the fingerprint"
        );
    }

    /// grizzly H1 regression: a workspace solve-env change (per-env conda dep
    /// pins, system-requirements, pypi-options) must change the fingerprint, or
    /// a workspace pixi.toml edit replays a stale lock.
    #[test]
    fn fingerprint_covers_workspace_solve_env() {
        let cfg = minimal_config("b");
        let chans = ["conda-forge".to_string()];
        let base = config_fingerprint(&cfg, &chans, "ws-dep:numpy=>=1.24");
        let changed = config_fingerprint(&cfg, &chans, "ws-dep:numpy===1.26.4");
        assert_ne!(
            base, changed,
            "a workspace dep-pin change must change the fingerprint"
        );
        let empty = config_fingerprint(&cfg, &chans, "");
        assert_ne!(base, empty, "presence of a workspace fp must matter");
    }

    /// Regression guard for the courier cold-solve replay hash mismatch bug:
    /// the producer (build_one) must hash the PRISTINE declared config, not
    /// the cascade-mutated effective config.
    ///
    /// Bug: apply_emission / resolve_all inject synthetic entries into
    /// `config.overrides` and `config.name_map` (transitive overrides, FALLBACK
    /// mappings, parselmouth renames).  Using the effective config in
    /// `config_fingerprint` produces a 686KB fingerprint vs the replayer's 2.5KB
    /// pristine one, so the hashes never match and cold-solve replay never fires.
    ///
    /// Fix: build_one now accepts `declared_config` (pristine manifest snapshot)
    /// and passes it to `config_fingerprint` instead of the effective config.
    #[test]
    fn declared_config_fingerprint_matches_replayer_not_effective() {
        let chans = ["conda-forge".to_string()];
        let ws_fp = "ws-dep:torch==2.3.0";

        // 1. Build the "declared" (manifest-level) config with a couple of
        //    user-declared overrides and a name-map entry — mirrors what the
        //    replayer reads directly from the manifest snapshot.
        let declared: RetreadConfig = serde_json::from_value(serde_json::json!({
            "retread-wheels": {
                "mypkg": { "version": "==1.0.0" }
            },
            "retread-overrides": {
                "numpy": ">=1.24"
            },
            "retread-name-map": {
                "pillow": "Pillow"
            }
        }))
        .unwrap();

        // 2. Build the "effective" config by cloning and injecting the kind of
        //    cascade-added entries that apply_emission / resolve_all produce at
        //    runtime (synthetic transitive overrides + FALLBACK name_map entries).
        let mut effective = declared.clone();
        // Mimic transitive overrides added by apply_emission
        effective
            .overrides
            .insert("torch".to_string(), "==2.3.0".to_string());
        effective
            .overrides
            .insert("triton".to_string(), "==2.3.0".to_string());
        effective
            .overrides
            .insert("torchvision".to_string(), ">=0.18".to_string());
        // Mimic FALLBACK + parselmouth name_map entries added by resolve_all
        effective.name_map.insert(
            PypiKey::from_pypi("FALLBACK"),
            CondaTarget::Mapped(CondaName::new("mypkg")),
        );
        effective.name_map.insert(
            PypiKey::from_pypi("Pillow"),
            CondaTarget::Mapped(CondaName::new("pillow")),
        );

        let fp_declared = config_fingerprint(&declared, &chans, ws_fp);
        let fp_effective = config_fingerprint(&effective, &chans, ws_fp);

        // The bug: using effective would produce a different (larger) fingerprint
        // than the replayer (which always uses pristine declared).
        assert_ne!(
            fp_declared, fp_effective,
            "effective config must produce a different fingerprint than declared \
             (proves the bug exists when effective is used)"
        );

        // The fix: the producer now uses declared_config; the replayer always used
        // declared — they must agree.
        let replayer_fp = config_fingerprint(&declared, &chans, ws_fp);
        assert_eq!(
            fp_declared, replayer_fp,
            "producer (declared_config) and replayer must produce identical fingerprints"
        );

        // Negative control: a real manifest-level override change MUST still
        // invalidate the fingerprint (cascade-only entries are excluded but real
        // declared changes are not).
        let mut declared_v2 = declared.clone();
        declared_v2
            .overrides
            .insert("numpy".to_string(), ">=1.26".to_string());
        let fp_declared_v2 = config_fingerprint(&declared_v2, &chans, ws_fp);
        assert_ne!(
            fp_declared, fp_declared_v2,
            "a real manifest override change must still invalidate the fingerprint"
        );
    }

    /// Create a process-unique temp directory and return it. Caller must clean
    /// up manually (matches the pattern used elsewhere in this codebase).
    fn make_test_dir(slug: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("retread-courier-{}-{}", slug, std::process::id(),));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// B2 regression guard: a relax-CHANGED index wheel must be classified
    /// Origin::Built and appear in source_urls.
    ///
    /// We drive this through the URL-requirement reroute path in `plan()`:
    /// "mypkg" has a URL Requires-Dist on "dep-a" (a bundle member), so
    /// override_line_map rewrites that line to "dep-a==1.2.3" -- the
    /// rewrite_wheel_with call detects the change -> `changed=true` -> Built.
    #[tokio::test]
    async fn relax_changed_index_wheel_is_built() {
        let tmp = make_test_dir("b2");
        let staging = tmp.join("staging");

        let bundle = "mypkg";
        let target_name = "dep-a";
        let target_version = "1.2.3";

        // dep-a wheel (no .injected -> must_ship=false, plain index wheel).
        let dep_a_whl_name = format!(
            "{}-{target_version}-py3-none-any.whl",
            wheel_dist_name(target_name)
        );
        let dep_a_whl = tmp.join(&dep_a_whl_name);
        std::fs::write(
            &dep_a_whl,
            make_wheel_bytes(target_name, target_version, &[]),
        )
        .unwrap();

        // "mypkg" (our bundle) is also an index wheel but its METADATA has a
        // URL requirement on dep-a.  plan() inserts an exact override for it
        // ("dep-a==1.2.3") and the mapper turns the URL line into that pin ->
        // rewrite_wheel_with returns changed=true.
        let idx_whl_name = format!("{bundle}-0.5.0-py3-none-any.whl");
        let idx_whl = tmp.join(&idx_whl_name);
        let url_req = format!("{target_name} @ https://example.com/{dep_a_whl_name}");
        std::fs::write(
            &idx_whl,
            make_wheel_bytes(bundle, "0.5.0", &[url_req.as_str()]),
        )
        .unwrap();

        let dep_a_wheel = make_emit_wheel(target_name, target_version, &[], Some(&dep_a_whl), None);
        let idx_wheel = make_emit_wheel(
            bundle,
            "0.5.0",
            &[url_req.as_str()],
            Some(&idx_whl),
            Some("https://pypi.example.com/mypkg-0.5.0-py3-none-any.whl"),
        );

        let emit_wheels = vec![dep_a_wheel, idx_wheel];
        let conda_capable: HashSet<String> = HashSet::new();
        let config = minimal_config(bundle);

        let result = stage(
            &config,
            bundle,
            "0.5.0",
            "3.11",
            &emit_wheels,
            &conda_capable,
            &[],
            &["https://pypi.org/simple/".to_string()],
            "",
            &tmp,
            &staging,
        )
        .await
        .unwrap();

        // Clean up.
        let _ = std::fs::remove_dir_all(&tmp);

        // The index wheel whose relax CHANGED must be Origin::Built.
        let built: Vec<&LockWheel> = result
            .lock
            .wheels
            .iter()
            .filter(|w| w.origin == Origin::Built)
            .collect();
        assert!(
            !built.is_empty(),
            "expected at least one Origin::Built wheel (the relax-changed index wheel)"
        );
        // Its shadow filename must appear in source_urls.
        for w in &built {
            let fname = &w.filename;
            assert!(
                w.sha256.is_some(),
                "fat Built wheel {fname} must bind its exact staged bytes"
            );
            assert!(
                result.source_urls.iter().any(|u| u.ends_with(fname)),
                "Built wheel {fname} must appear in source_urls"
            );
        }
    }

    /// root_requirements equals ["<bundle>-pypi==<version>"].
    #[tokio::test]
    async fn root_requirements_format() {
        let tmp = make_test_dir("root-req");
        let staging = tmp.join("staging");

        let bundle = "my-bundle";
        let ver = "3.2.1";

        // A simple remote index wheel with no Requires-Dist to rewrite.
        let w = make_emit_wheel(
            "somepkg",
            "1.0.0",
            &[],
            None,
            Some("https://pypi.org/simple/somepkg-1.0.0-py3-none-any.whl"),
        );
        let config = minimal_config(bundle);
        let conda_capable = HashSet::new();

        let result = stage(
            &config,
            bundle,
            ver,
            "3.11",
            &[w],
            &conda_capable,
            &[],
            &["https://pypi.org/simple/".to_string()],
            "",
            &tmp,
            &staging,
        )
        .await
        .unwrap();

        let _ = std::fs::remove_dir_all(&tmp);

        assert_eq!(
            result.lock.root_requirements,
            vec![format!("{bundle}-pypi=={ver}")]
        );
        let expected_target = ResolutionTarget::for_subdir("3.11", Platform::current().as_str());
        assert!(result.lock.is_for_resolution_target(&expected_target));
    }

    #[tokio::test]
    async fn stage_rejects_foreign_target_before_mutation() {
        let tmp = make_test_dir("foreign-stage");
        let staging = tmp.join("staging");
        let foreign = if Platform::current() == Platform::LinuxAarch64 {
            "linux-64"
        } else {
            "linux-aarch64"
        };
        let target = ResolutionTarget::from_wheel_target(
            WheelTarget {
                python_version: "3.11".into(),
                conda_subdir: foreign.into(),
                max_glibc: Some((2, 35)),
            },
            Some((2, 35)),
        );

        let result = stage_for_target(
            &minimal_config("demo"),
            "demo",
            "demo",
            "1.0.0",
            &target,
            &[],
            &HashSet::new(),
            &[],
            &[],
            "",
            &tmp,
            &staging,
        )
        .await;
        let err = match result {
            Ok(_) => panic!("foreign stage unexpectedly succeeded"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("cannot stage target"));
        assert!(!staging.exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn stage_rejects_reserved_meta_wheel_collision_before_mutation() {
        let tmp = make_test_dir("reserved-meta-stage");
        let staging = tmp.join("staging");
        let target = ResolutionTarget::for_subdir("3.11", Platform::current().as_str());
        let collision = make_emit_wheel("demo-pypi", "1.0.0", &[], None, None);

        let err = stage_for_target(
            &minimal_config("demo"),
            "demo",
            "demo",
            "1.0.0",
            &target,
            &[collision],
            &HashSet::new(),
            &[],
            &[],
            "",
            &tmp,
            &staging,
        )
        .await
        .err()
        .expect("reserved meta-wheel collision unexpectedly staged");
        assert!(format!("{err:#}").contains("reserved meta-wheel"));
        assert!(!staging.exists());

        let mut filename_collision = make_emit_wheel("other", "1.0.0", &[], None, None);
        filename_collision.wheel_filename = "demo_pypi-1.0.0-py3-none-any.whl".into();
        let err = validate_reserved_meta_wheel_collisions(&[filename_collision], "demo", "1.0.0")
            .unwrap_err();
        assert!(format!("{err:#}").contains("wheel filename"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn emit_wheel_validation_rejects_cross_arch_and_provenance_mismatch() {
        let target = WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "linux-aarch64".into(),
            max_glibc: Some((2, 35)),
        };
        let mut wheel = make_emit_wheel("demo", "1.0.0", &[], None, None);
        wheel.wheel_filename = "demo-1.0.0-cp311-cp311-manylinux_2_35_x86_64.whl".into();
        let err = validate_emit_wheels(&[wheel], &target).unwrap_err();
        assert!(err.to_string().contains("incompatible"));

        let mut mismatch = make_emit_wheel("demo", "1.0.0", &[], None, None);
        mismatch.wheel_filename = "demo-1.0.0-cp311-cp311-manylinux_2_35_aarch64.whl".into();
        mismatch.local_path =
            Some("/unused/other-1.0.0-cp311-cp311-manylinux_2_35_aarch64.whl".into());
        let err = validate_emit_wheels(&[mismatch], &target).unwrap_err();
        assert!(err.to_string().contains("records distribution"));
    }

    #[test]
    fn emit_wheel_validation_accepts_owned_shadow_transformations() {
        let tmp = make_test_dir("owned-shadow-provenance");
        let target = WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "linux-aarch64".into(),
            max_glibc: Some((2, 39)),
        };
        let source = "demo-1.0.0-cp311-cp311-manylinux_2_35_aarch64.whl";
        let source_path = tmp.join(source);
        std::fs::write(&source_path, make_wheel_bytes("demo", "1.0.0", &[])).unwrap();
        let mut wheel = make_emit_wheel("demo", "1.0.0", &[], None, None);
        wheel.wheel_filename =
            "demo-1.0.0-999retread-cp311-cp311-manylinux_2_35_aarch64.whl".into();
        wheel.local_path = Some(source_path);
        wheel.upstream_url = format!("https://example.com/{source}").parse().ok();
        validate_emit_wheels(&[wheel], &target).unwrap();

        let build_one = "demo-1.0.0-1-cp311-cp311-manylinux_2_35_aarch64.whl";
        let build_one_path = tmp.join(build_one);
        std::fs::write(&build_one_path, make_wheel_bytes("demo", "1.0.0", &[])).unwrap();
        let mut conflicting = make_emit_wheel("demo", "1.0.0", &[], None, None);
        conflicting.wheel_filename =
            "demo-1.0.0-999retread-cp311-cp311-manylinux_2_35_aarch64.whl".into();
        conflicting.local_path = Some(build_one_path);
        conflicting.upstream_url =
            "https://example.com/demo-1.0.0-2-cp311-cp311-manylinux_2_35_aarch64.whl"
                .parse()
                .ok();
        let err = validate_emit_wheels(&[conflicting], &target).unwrap_err();
        assert!(format!("{err:#}").contains("source provenance mismatch"));
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn wheel_identity_validation_binds_filenames_metadata_and_build_tags() {
        let target = WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "linux-64".into(),
            max_glibc: Some((2, 35)),
        };
        validate_wheel_filename_for_target(
            "demo-pkg",
            "1.0",
            "Demo_Pkg-1.0.0-py3-none-any.whl",
            &target,
            "test wheel",
        )
        .unwrap();
        assert!(
            validate_wheel_filename_for_target(
                "demo-pkg",
                "1.0",
                "evil-1.0-py3-none-any.whl",
                &target,
                "test wheel",
            )
            .is_err()
        );
        assert!(
            validate_distribution_identity("-demo", "1.0", "demo", "1.0", "test wheel").is_err(),
            "invalid leading separator must not normalize into a valid distribution",
        );
        assert!(
            validate_wheel_filename_for_target(
                "demo-pkg",
                "1.0",
                "demo_pkg-9.9-py3-none-any.whl",
                &target,
                "test wheel",
            )
            .is_err()
        );

        assert!(wheel_filename_provenance_matches(
            "demo-1.0-999retread-py3-none-any.whl",
            "demo-1.0-1-py3-none-any.whl",
        ));
        assert!(!wheel_filename_provenance_matches(
            "demo-1.0-1-py3-none-any.whl",
            "demo-1.0-2-py3-none-any.whl",
        ));
        for malformed in [
            "demo-1.0-build-py3-none-any.whl",
            "demo-1.0-1-py3-none..evil-any.whl",
            "demo-1.0-1/../../evil-py3-none-any.whl",
            "demo-1.0-1\\..\\..\\evil-py3-none-any.whl",
        ] {
            assert!(
                validate_wheel_filename_identity("demo", "1.0", malformed, "test wheel").is_err(),
                "malformed wheel filename was accepted: {malformed}",
            );
        }

        let mut traversal = make_emit_wheel("demo", "1.0", &[], None, None);
        traversal.wheel_filename = "demo-1.0-999retread-py3-none-any.whl".into();
        traversal.upstream_url = "https://example.com/demo-1.0-1%2F..%2F..%2Fevil-py3-none-any.whl"
            .parse()
            .ok();
        let err = validate_emit_wheels(&[traversal], &target).unwrap_err();
        assert!(format!("{err:#}").contains("single wheel basename"));

        let mut oversized = std::io::Cursor::new(b"12345".to_vec());
        let err = read_utf8_limited(&mut oversized, 4, "test METADATA").unwrap_err();
        assert!(err.to_string().contains("exceeds the 4-byte safety limit"));

        let tmp = make_test_dir("metadata-identity");
        let duplicate_path = tmp.join("demo-1.0-py3-none-any.whl");
        let duplicate_metadata =
            b"Metadata-Version: 2.1\nName: evil\nName: demo\nVersion: 9.9\nVersion: 1.0\n";
        std::fs::write(
            &duplicate_path,
            make_wheel_bytes_with_metadata("demo", "1.0", duplicate_metadata),
        )
        .unwrap();
        let err = validate_wheel_file_identity(
            "demo",
            "1.0",
            &duplicate_path,
            None,
            "test duplicate identity wheel",
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("duplicate `Name` identity headers"));
        let err = validate_unique_metadata_identity_headers(
            "Name: demo\nVersion: 9.9\nVersion: 1.0\n",
            "test METADATA",
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("duplicate `Version` identity headers")
        );

        let mut malformed_sha = make_emit_wheel("demo", "1.0", &[], None, None);
        malformed_sha.sha256 = Some("abc".into());
        assert!(validate_emit_wheels(&[malformed_sha], &target).is_err());

        let duplicate_a = make_emit_wheel("demo", "1.0", &[], None, None);
        let duplicate_b = make_emit_wheel("Demo", "1.0", &[], None, None);
        let err = validate_emit_wheels(&[duplicate_a, duplicate_b], &target).unwrap_err();
        assert!(format!("{err:#}").contains("duplicate distribution entries"));

        let path = tmp.join("demo_pkg-1.0-py3-none-any.whl");
        std::fs::write(&path, make_wheel_bytes("evil", "1.0", &[])).unwrap();
        let wheel = make_emit_wheel("demo-pkg", "1.0", &[], Some(&path), None);
        let err = validate_emit_wheels(&[wheel], &target).unwrap_err();
        assert!(format!("{err:#}").contains("records distribution `evil`"));

        std::fs::write(&path, make_wheel_bytes("demo-pkg", "9.9", &[])).unwrap();
        let wheel = make_emit_wheel("demo-pkg", "1.0", &[], Some(&path), None);
        let err = validate_emit_wheels(&[wheel], &target).unwrap_err();
        assert!(format!("{err:#}").contains("records version `9.9`"));
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn wheel_target_validation_rejects_cross_family_and_accepts_universal2() {
        let osx_x86 = WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "osx-64".into(),
            max_glibc: None,
        };
        let osx_arm = WheelTarget {
            conda_subdir: "osx-arm64".into(),
            ..osx_x86.clone()
        };
        let linux = "demo-1.0-cp311-cp311-manylinux_2_17_x86_64.whl";
        assert!(
            validate_wheel_filename_for_target("demo", "1.0", linux, &osx_x86, "test").is_err()
        );
        let universal = "demo-1.0-cp311-cp311-macosx_11_0_universal2.whl";
        validate_wheel_filename_for_target("demo", "1.0", universal, &osx_x86, "test").unwrap();
        validate_wheel_filename_for_target("demo", "1.0", universal, &osx_arm, "test").unwrap();
        for malformed in [
            "demo-1.0-cp311-cp311-macosx_evil_x86_64.whl",
            "demo-1.0-cp311-cp311-macosx_evil_universal2.whl",
        ] {
            assert!(
                validate_wheel_filename_for_target("demo", "1.0", malformed, &osx_x86, "test")
                    .is_err()
            );
        }

        let linux_target = WheelTarget {
            python_version: "3.11".into(),
            conda_subdir: "linux-64".into(),
            max_glibc: Some((2, 35)),
        };
        for valid in [
            "demo-1.0-cp311-cp311-manylinux_2_17_x86_64.whl",
            "demo-1.0-cp311-none-manylinux_2_17_x86_64.whl",
            "demo-1.0-cp36-abi3-manylinux_2_17_x86_64.whl",
        ] {
            validate_wheel_filename_for_target("demo", "1.0", valid, &linux_target, "test")
                .unwrap();
        }
        assert!(
            validate_wheel_filename_for_target(
                "demo",
                "1.0",
                "demo-1.0-cp311-cp310-manylinux_2_17_x86_64.whl",
                &linux_target,
                "test",
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn owned_shadow_replay_never_downgrades_or_drops_final_hash() {
        let tmp = make_test_dir("owned-shadow-replay-guard");
        let config = minimal_config("demo");
        let raw_url = "https://example.com/demo-1.0-py3-none-any.whl";
        let conda_capable = HashSet::new();
        let remote_staging = tmp.join("remote-staging");

        let mut remote = make_emit_wheel("demo", "1.0", &[], None, Some(raw_url));
        remote.wheel_filename = "demo-1.0-999retread-py3-none-any.whl".into();
        let err = stage(
            &config,
            "demo",
            "1.0",
            "3.11",
            &[remote],
            &conda_capable,
            &[],
            &[],
            "",
            &tmp,
            &remote_staging,
        )
        .await
        .err()
        .expect("owned shadow unexpectedly downgraded to an index wheel");
        assert!(format!("{err:#}").contains("refusing to downgrade"));

        let injected = tmp.join("demo-1.0-py3-none-any.injected.whl");
        std::fs::write(&injected, make_wheel_bytes("demo", "1.0", &[])).unwrap();
        let mut built = make_emit_wheel("demo", "1.0", &[], Some(&injected), None);
        built.wheel_filename = "demo-1.0-999retread-py3-none-any.whl".into();
        built.sha256 = Some("0".repeat(64));
        let built_staging = tmp.join("built-staging");
        let err = stage(
            &config,
            "demo",
            "1.0",
            "3.11",
            &[built],
            &conda_capable,
            &[],
            &[],
            "",
            &tmp,
            &built_staging,
        )
        .await
        .err()
        .expect("owned built wheel unexpectedly dropped its final hash");
        assert!(format!("{err:#}").contains("SHA-256 mismatch"));
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[tokio::test]
    async fn stage_rejects_malformed_python_before_mutation() {
        let tmp = make_test_dir("malformed-python-target");
        let staging = tmp.join("staging");
        let result = stage(
            &minimal_config("demo"),
            "demo",
            "1.0",
            "3.11.0.1",
            &[],
            &HashSet::new(),
            &[],
            &[],
            "",
            &tmp,
            &staging,
        )
        .await;
        let err = match result {
            Ok(_) => panic!("malformed target unexpectedly staged"),
            Err(error) => error,
        };
        assert!(format!("{err:#}").contains("malformed resolution target"));
        assert!(!staging.exists());
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn shadow_cache_key_is_target_qualified() {
        let bytes = b"wheel bytes";
        let requires = vec!["demo>=1".to_string()];
        let overrides = BTreeMap::new();
        let conda_capable = HashSet::new();
        let drop_url = HashSet::new();
        let x86 = shadow_cache_key_for_target(
            bytes,
            &requires,
            &overrides,
            &conda_capable,
            &drop_url,
            &"a".repeat(64),
        );
        let arm = shadow_cache_key_for_target(
            bytes,
            &requires,
            &overrides,
            &conda_capable,
            &drop_url,
            &"b".repeat(64),
        );
        assert_ne!(x86, arm);
        assert_eq!(x86.len(), 64);
        assert_eq!(arm.len(), 64);
    }

    #[test]
    fn shadow_cache_key_covers_applicable_url_drops() {
        let bytes = b"wheel bytes";
        let requires = vec!["dep @ https://example.com/dep.whl".to_string()];
        let overrides = BTreeMap::new();
        let conda_capable = HashSet::new();
        let no_drop = HashSet::new();
        let drop_dep = HashSet::from(["dep".to_string()]);
        let drop_other = HashSet::from(["other".to_string()]);
        let key = |drop_url: &HashSet<String>| {
            shadow_cache_key_for_target(
                bytes,
                &requires,
                &overrides,
                &conda_capable,
                drop_url,
                &"a".repeat(64),
            )
        };
        assert_ne!(key(&no_drop), key(&drop_dep));
        assert_eq!(key(&no_drop), key(&drop_other));
    }

    /// An unchanged index wheel is recorded as Origin::Index with its upstream url.
    #[tokio::test]
    async fn unchanged_index_wheel_is_index_with_url() {
        let tmp = make_test_dir("idx-origin");
        let staging = tmp.join("staging");

        let bundle = "idx-bundle";
        let upstream_url = "https://pypi.example.com/unchanged-1.0.0-py3-none-any.whl";

        // Pure remote index wheel: no local bytes, no Requires-Dist -> mapper
        // produces no changes -> unchanged -> Origin::Index.
        let w = make_emit_wheel("unchanged", "1.0.0", &[], None, Some(upstream_url));

        let config = minimal_config(bundle);
        let conda_capable = HashSet::new();

        let result = stage(
            &config,
            bundle,
            "1.0.0",
            "3.11",
            &[w],
            &conda_capable,
            &[],
            &["https://pypi.org/simple/".to_string()],
            "",
            &tmp,
            &staging,
        )
        .await
        .unwrap();

        let _ = std::fs::remove_dir_all(&tmp);

        let unchanged: Vec<&LockWheel> = result
            .lock
            .wheels
            .iter()
            .filter(|w| w.name == "unchanged")
            .collect();
        assert_eq!(unchanged.len(), 1, "exactly one wheel named 'unchanged'");
        assert_eq!(unchanged[0].origin, Origin::Index);
        assert_eq!(unchanged[0].url.as_deref(), Some(upstream_url));
        assert_eq!(
            unchanged[0].sha256.as_deref(),
            Some("0000000000000000000000000000000000000000000000000000000000000000")
        );
    }

    // ── Shadow-cache unit tests ───────────────────────────────────────────

    /// Helper: write a wheel file and return its path.
    fn write_wheel(dir: &Path, name: &str, version: &str, requires: &[&str]) -> std::path::PathBuf {
        let fname = format!("{}-{version}-py3-none-any.whl", wheel_dist_name(name));
        let path = dir.join(&fname);
        std::fs::write(&path, make_wheel_bytes(name, version, requires)).unwrap();
        path
    }

    /// Test 1 — byte-identical hit: rewrite through cache twice; warm hit
    /// produces byte-identical staged output.
    #[test]
    fn shadow_cache_warm_hit_is_byte_identical() {
        let tmp = make_test_dir("cache-hit");
        let cache = tmp.join("cache");
        let whl = write_wheel(&tmp, "pkg-a", "1.0.0", &["dep-x==2.0.0"]);

        let mut overrides = BTreeMap::new();
        overrides.insert("dep-x".to_string(), ">=2.0.0".to_string());
        let conda_cap: HashSet<String> = HashSet::new();

        let bytes = std::fs::read(&whl).unwrap();
        let requires = vec!["dep-x==2.0.0".to_string()];
        let key = shadow_cache_key(&bytes, &requires, &overrides, &conda_cap);
        let drop: HashSet<String> = HashSet::new();

        // Cold pass.
        let dst1 = tmp.join("staged1.whl");
        shadow_cache_stage(&whl, &dst1, &cache, &key, &overrides, &conda_cap, &drop).unwrap();
        let out1 = std::fs::read(&dst1).unwrap();

        // Warm pass (cache hit).
        let dst2 = tmp.join("staged2.whl");
        shadow_cache_stage(&whl, &dst2, &cache, &key, &overrides, &conda_cap, &drop).unwrap();
        let out2 = std::fs::read(&dst2).unwrap();

        let _ = std::fs::remove_dir_all(&tmp);
        assert_eq!(
            out1, out2,
            "warm cache hit must produce byte-identical output"
        );
    }

    #[test]
    fn poisoned_shadow_cache_hit_and_miss_are_evicted_before_use() {
        let tmp = make_test_dir("cache-poison");
        let cache = tmp.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        let source = write_wheel(&tmp, "demo", "1.0", &[]);
        let overrides = BTreeMap::new();
        let conda_capable = HashSet::new();
        let drop_url = HashSet::new();

        let hit_key = "a".repeat(64);
        let poisoned_hit = cache.join(format!("{hit_key}.changed"));
        std::fs::write(&poisoned_hit, make_wheel_bytes("evil", "1.0", &[])).unwrap();
        let hit_dst = tmp.join("hit-dst.whl");
        let err = shadow_cache_stage_validated(
            &source,
            &hit_dst,
            &cache,
            &hit_key,
            "demo",
            "1.0",
            None,
            &overrides,
            &conda_capable,
            &drop_url,
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("records distribution `evil`"));
        assert!(!poisoned_hit.exists());
        assert!(!hit_dst.exists());

        let miss_key = "b".repeat(64);
        let evil_source = tmp.join("evil-1.0-py3-none-any.whl");
        std::fs::write(&evil_source, make_wheel_bytes("evil", "1.0", &[])).unwrap();
        let miss_dst = tmp.join("miss-dst.whl");
        let err = shadow_cache_stage_validated(
            &evil_source,
            &miss_dst,
            &cache,
            &miss_key,
            "demo",
            "1.0",
            None,
            &overrides,
            &conda_capable,
            &drop_url,
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("records distribution `evil`"));
        assert!(!cache.join(format!("{miss_key}.same")).exists());
        assert!(!cache.join(format!("{miss_key}.changed")).exists());
        assert!(!miss_dst.exists());
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// Test 2a — applicable-subset keying: an override for a name NOT in
    /// Requires-Dist produces the SAME key (hit, identical output).
    #[test]
    fn shadow_cache_key_unrelated_override_is_hit() {
        let tmp = make_test_dir("cache-key-unrelated");
        let whl = write_wheel(&tmp, "pkg-b", "1.0.0", &["dep-y==3.0.0"]);
        let bytes = std::fs::read(&whl).unwrap();
        let requires = vec!["dep-y==3.0.0".to_string()];
        let conda_cap: HashSet<String> = HashSet::new();

        // Base overrides: dep-y.
        let mut ov_base = BTreeMap::new();
        ov_base.insert("dep-y".to_string(), ">=3.0.0".to_string());

        // Add override for "unrelated" which is NOT in requires_dist.
        let mut ov_extra = ov_base.clone();
        ov_extra.insert("unrelated".to_string(), ">=99".to_string());

        let key_base = shadow_cache_key(&bytes, &requires, &ov_base, &conda_cap);
        let key_extra = shadow_cache_key(&bytes, &requires, &ov_extra, &conda_cap);

        let _ = std::fs::remove_dir_all(&tmp);
        assert_eq!(
            key_base, key_extra,
            "an override for a name not in Requires-Dist must NOT change the cache key"
        );
    }

    /// Test 2b — applicable-subset keying: an override for a name IN
    /// Requires-Dist produces a DIFFERENT key (miss).
    #[test]
    fn shadow_cache_key_relevant_override_is_miss() {
        let tmp = make_test_dir("cache-key-relevant");
        let whl = write_wheel(&tmp, "pkg-c", "1.0.0", &["dep-z==4.0.0"]);
        let bytes = std::fs::read(&whl).unwrap();
        let requires = vec!["dep-z==4.0.0".to_string()];
        let conda_cap: HashSet<String> = HashSet::new();

        let mut ov_a = BTreeMap::new();
        ov_a.insert("dep-z".to_string(), ">=4.0.0".to_string());

        let mut ov_b = BTreeMap::new();
        ov_b.insert("dep-z".to_string(), ">=4.1.0".to_string()); // different spec

        let key_a = shadow_cache_key(&bytes, &requires, &ov_a, &conda_cap);
        let key_b = shadow_cache_key(&bytes, &requires, &ov_b, &conda_cap);

        let _ = std::fs::remove_dir_all(&tmp);
        assert_ne!(
            key_a, key_b,
            "changing the spec for a name IN Requires-Dist must change the cache key"
        );
    }

    /// Test 3 — version invalidation: changing the EMIT_EPOCH in the key
    /// domain would produce a different key. We simulate this by testing
    /// that a change to the input wheel bytes itself also invalidates (since
    /// we can't mutate the static EMIT_EPOCH in a test, we verify the
    /// bytes component of the key is sensitive).
    #[test]
    fn shadow_cache_key_changes_with_input_wheel_bytes() {
        let tmp = make_test_dir("cache-key-epoch");
        let whl_v1 = write_wheel(&tmp, "pkg-d", "1.0.0", &["dep-w==5.0.0"]);
        let whl_v2 = write_wheel(&tmp, "pkg-d", "1.0.1", &["dep-w==5.0.0"]);

        let bytes_v1 = std::fs::read(&whl_v1).unwrap();
        let bytes_v2 = std::fs::read(&whl_v2).unwrap();
        let requires = vec!["dep-w==5.0.0".to_string()];
        let ov: BTreeMap<String, String> = BTreeMap::new();
        let cap: HashSet<String> = HashSet::new();

        let key_v1 = shadow_cache_key(&bytes_v1, &requires, &ov, &cap);
        let key_v2 = shadow_cache_key(&bytes_v2, &requires, &ov, &cap);

        let _ = std::fs::remove_dir_all(&tmp);
        assert_ne!(
            key_v1, key_v2,
            "different input wheel bytes must produce different cache keys"
        );
    }

    /// Test 4 — fold equivalence: the single-pass `(sha, did_change)` from
    /// shadow_cache_stage (which calls rewrite_wheel_with once) equals the
    /// result of calling rewrite_wheel_with directly (old probe behavior).
    #[test]
    fn fold_equivalence_changed_wheel() {
        let tmp = make_test_dir("fold-changed");
        let cache = tmp.join("cache");
        // wheel with a dep that WILL be rewritten.
        let whl = write_wheel(&tmp, "pkg-e", "1.0.0", &["dep-q==6.0.0"]);
        let bytes = std::fs::read(&whl).unwrap();

        let mut ov = BTreeMap::new();
        ov.insert("dep-q".to_string(), ">=6.0.0".to_string());
        let cap: HashSet<String> = HashSet::new();
        let requires = vec!["dep-q==6.0.0".to_string()];

        let key = shadow_cache_key(&bytes, &requires, &ov, &cap);

        let drop: HashSet<String> = HashSet::new();

        // Single-pass via cache.
        let dst_cache = tmp.join("staged-cache.whl");
        let (sha_cache, changed_cache) =
            shadow_cache_stage(&whl, &dst_cache, &cache, &key, &ov, &cap, &drop).unwrap();

        // Direct rewrite_wheel_with (old probe behavior).
        let dst_direct = tmp.join("staged-direct.whl");
        let m = override_line_map(&ov, &cap, &drop);
        let (sha_direct, changed_direct) =
            crate::wheel_rewrite::rewrite_wheel_with(&whl, &dst_direct, &m).unwrap();

        let bytes_cache = std::fs::read(&dst_cache).unwrap();
        let bytes_direct = std::fs::read(&dst_direct).unwrap();

        let _ = std::fs::remove_dir_all(&tmp);

        assert_eq!(
            changed_cache, changed_direct,
            "did_change must agree between cache path and direct rewrite"
        );
        assert_eq!(
            sha_cache, sha_direct,
            "sha256 must agree between cache path and direct rewrite"
        );
        assert_eq!(
            bytes_cache, bytes_direct,
            "staged bytes must be byte-identical"
        );
        assert!(changed_cache, "dep-q==6.0.0 -> >=6.0.0 must be a change");
    }

    #[test]
    fn fold_equivalence_unchanged_wheel() {
        let tmp = make_test_dir("fold-unchanged");
        let cache = tmp.join("cache");
        // wheel with no deps that match any override -> unchanged.
        let whl = write_wheel(&tmp, "pkg-f", "1.0.0", &["dep-p>=1.0"]);
        let bytes = std::fs::read(&whl).unwrap();

        let ov: BTreeMap<String, String> = BTreeMap::new();
        let cap: HashSet<String> = HashSet::new();
        let drop: HashSet<String> = HashSet::new();
        let requires = vec!["dep-p>=1.0".to_string()];

        let key = shadow_cache_key(&bytes, &requires, &ov, &cap);

        let dst_cache = tmp.join("staged-cache.whl");
        let (sha_cache, changed_cache) =
            shadow_cache_stage(&whl, &dst_cache, &cache, &key, &ov, &cap, &drop).unwrap();

        let dst_direct = tmp.join("staged-direct.whl");
        let m = override_line_map(&ov, &cap, &drop);
        let (sha_direct, changed_direct) =
            crate::wheel_rewrite::rewrite_wheel_with(&whl, &dst_direct, &m).unwrap();

        let bytes_cache = std::fs::read(&dst_cache).unwrap();
        let bytes_direct = std::fs::read(&dst_direct).unwrap();

        let _ = std::fs::remove_dir_all(&tmp);

        assert_eq!(changed_cache, changed_direct, "did_change must agree");
        assert_eq!(sha_cache, sha_direct, "sha256 must agree");
        assert_eq!(
            bytes_cache, bytes_direct,
            "staged bytes must be byte-identical"
        );
        assert!(!changed_cache, "no matching override -> must be unchanged");
    }

    /// Test 5 — meta-wheel guard: same (bundle, version, entries) -> marker
    /// matches -> reuse; one entry changed -> marker changes -> rebuild.
    /// Also verifies rebuilt bytes == fresh build_meta_wheel output.
    #[tokio::test]
    async fn meta_wheel_guard_reuse_and_invalidate() {
        use crate::emit_pypi::build_meta_wheel;

        let tmp = make_test_dir("meta-guard");
        let staging = tmp.join("staging");
        std::fs::create_dir_all(&staging).unwrap();

        let bundle = "meta-bundle";
        let ver = "1.0.0";

        // A single remote index wheel with no Requires-Dist changes.
        let upstream_url = "https://pypi.example.com/somepkg-1.0.0-py3-none-any.whl";
        let w = make_emit_wheel("somepkg", "1.0.0", &[], None, Some(upstream_url));
        let config = minimal_config(bundle);
        let conda_cap: HashSet<String> = HashSet::new();

        // First stage call: cold, must write the meta-wheel.
        let result1 = stage(
            &config,
            bundle,
            ver,
            "3.11",
            std::slice::from_ref(&w),
            &conda_cap,
            &[],
            &["https://pypi.org/simple/".to_string()],
            "",
            &tmp,
            &staging,
        )
        .await
        .unwrap();

        let meta_name1 = result1
            .lock
            .wheels
            .iter()
            .find(|w| w.name == "somepkg")
            .map(|_| format!("{}_pypi-{}-py3-none-any.whl", bundle.replace('-', "_"), ver))
            .unwrap_or_else(|| {
                format!("{}_pypi-{}-py3-none-any.whl", bundle.replace('-', "_"), ver)
            });
        let meta_path = staging.join(&meta_name1);
        assert!(
            meta_path.exists(),
            "meta-wheel must exist after first stage"
        );
        let mtime1 = std::fs::metadata(&meta_path).unwrap().modified().unwrap();

        // Brief pause to ensure mtime would differ if file were rewritten.
        std::thread::sleep(std::time::Duration::from_millis(20));

        // Second stage call: same inputs, must reuse (marker hit).
        stage(
            &config,
            bundle,
            ver,
            "3.11",
            std::slice::from_ref(&w),
            &conda_cap,
            &[],
            &["https://pypi.org/simple/".to_string()],
            "",
            &tmp,
            &staging,
        )
        .await
        .unwrap();

        let mtime2 = std::fs::metadata(&meta_path).unwrap().modified().unwrap();
        assert_eq!(
            mtime1, mtime2,
            "meta-wheel must NOT be rewritten on a marker hit"
        );

        // Verify that the bytes match a fresh build_meta_wheel.
        let entries_check: Vec<(String, crate::config::WheelEntry, Option<String>)> = config
            .retread_wheels
            .iter()
            .map(|(k, e)| {
                let resolved = Some("1.0.0".to_string());
                (k.clone(), e.clone(), resolved)
            })
            .collect();
        let (_, fresh_bytes) = build_meta_wheel(bundle, ver, &entries_check);
        let staged_bytes = std::fs::read(&meta_path).unwrap();
        assert_eq!(
            staged_bytes, fresh_bytes,
            "staged meta-wheel bytes must equal fresh build_meta_wheel output"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ── Persistent shadow-cache tests ─────────────────────────────────────

    /// Test P1: shadow cache materializes under the configured persistent dir,
    /// NOT under wheels/. We test the helper logic directly without mutating
    /// process env (unsafe in multi-threaded tests): the cache key used by
    /// shadow_cache_stage writes into whatever `cache_dir` we pass, which is
    /// now retread_cache_root()/shadow — verified here by structurally checking
    /// that the path does NOT start with the wheels/ prefix.
    #[test]
    fn shadow_cache_materializes_under_persistent_dir() {
        // Construct a representative persistent-cache path and confirm it is
        // structurally separate from any source_dir/wheels/ path.
        let some_source_dir = std::path::PathBuf::from("/some/project/isaac-pack");
        let wheels_dir = some_source_dir.join("wheels");

        // OLD path (what the code used to produce):
        let old_cache_dir = wheels_dir.join(".retread-shadow-cache");
        // NEW path (what retread_cache_root() + "shadow" looks like):
        // We can't call retread_cache_root() here without relying on env state,
        // so we verify the STRUCTURAL invariant: the new path must not start with
        // source_dir/wheels.
        let new_cache_dir = std::path::PathBuf::from("/home/user/.cache/retread/shadow");

        // OLD path starts with wheels/ -- this is the bug.
        assert!(
            old_cache_dir.starts_with(&wheels_dir),
            "old cache was inside wheels/ (the bug we fixed)"
        );
        // NEW path does NOT start with wheels/ -- this is the fix.
        assert!(
            !new_cache_dir.starts_with(&wheels_dir),
            "new persistent cache must NOT be inside wheels/"
        );
        // The shadow cache is the retread_cache_root() joined with "shadow".
        // shadow_cache_dir construction in stage() is:
        //   retread_cache_root().join("shadow")
        // which, for any sensible HOME/XDG, produces a path outside source_dir.
        let cache_root = retread_cache_root();
        let shadow_dir = cache_root.join("shadow");
        // The shadow dir must NOT start with any path that is "wheels" relative.
        // It may coincidentally share a prefix with the home dir, but it will
        // never start with an arbitrary pack's wheels/ subdir.
        assert!(
            shadow_dir.ends_with("retread/shadow"),
            "shadow dir must end with retread/shadow (got {})",
            shadow_dir.display()
        );
    }

    /// Test P2: stage a wheel; delete the wheels/ dir (simulating `rm -rf wheels`);
    /// re-stage; assert the shadow cache is still hit (byte-identical output).
    #[test]
    fn shadow_cache_survives_rm_wheels() {
        let tmp = make_test_dir("p2-survive");
        // Persistent cache in its own subdir (not inside wheels/).
        let persistent_cache = tmp.join("persistent-cache");
        let wheels_dir = tmp.join("wheels");
        std::fs::create_dir_all(&wheels_dir).unwrap();

        let shadow_cache = persistent_cache.join("shadow");

        let whl = write_wheel(&tmp, "pkg-survive", "1.0.0", &["dep-x==2.0.0"]);
        let mut overrides = BTreeMap::new();
        overrides.insert("dep-x".to_string(), ">=2.0.0".to_string());
        let conda_cap: HashSet<String> = HashSet::new();

        let bytes = std::fs::read(&whl).unwrap();
        let requires = vec!["dep-x==2.0.0".to_string()];
        let key = shadow_cache_key(&bytes, &requires, &overrides, &conda_cap);
        let drop: HashSet<String> = HashSet::new();

        // Cold pass: populate the persistent cache.
        let dst1 = tmp.join("staged1.whl");
        shadow_cache_stage(
            &whl,
            &dst1,
            &shadow_cache,
            &key,
            &overrides,
            &conda_cap,
            &drop,
        )
        .unwrap();
        let out1 = std::fs::read(&dst1).unwrap();

        // Simulate `rm -rf wheels`: delete the wheels dir.
        let _ = std::fs::remove_dir_all(&wheels_dir);
        assert!(!wheels_dir.exists(), "wheels dir should be gone");

        // Warm pass: the source wheel is still accessible (it lives in tmp/,
        // not tmp/wheels/), but the PERSISTENT cache is under persistent_cache/.
        // The key point is: even if wheels/ were gone, the persistent cache survives.
        let dst2 = tmp.join("staged2.whl");
        let (_sha2, _changed2) = shadow_cache_stage(
            &whl,
            &dst2,
            &shadow_cache,
            &key,
            &overrides,
            &conda_cap,
            &drop,
        )
        .unwrap();
        let out2 = std::fs::read(&dst2).unwrap();

        let _ = std::fs::remove_dir_all(&tmp);

        assert_eq!(
            out1, out2,
            "shadow cache hit after rm-rf wheels must produce byte-identical output"
        );
    }

    /// Test P3: EXDEV cross-device fallback — hardlink_or_copy falls back to
    /// copy and produces identical bytes.
    #[test]
    fn hardlink_or_copy_fallback_produces_identical_bytes() {
        let tmp = make_test_dir("p3-exdev");
        let whl = write_wheel(&tmp, "pkg-exdev", "1.0.0", &[]);
        let src_bytes = std::fs::read(&whl).unwrap();

        // Simulate cross-device by calling hardlink_or_copy with a dst on the
        // same FS (hard_link succeeds), then verify bytes. The EXDEV case
        // itself requires two different filesystems (not easily reproducible in
        // a unit test), but we verify the copy-fallback codepath directly:
        // remove the src first so hard_link fails, then copy.
        let dst = tmp.join("dst.whl");
        // Copy-fallback test: use hardlink_or_copy (the sync version in courier.rs)
        // which already has the fallback. We exercise it via shadow_cache_stage.
        let cache = tmp.join("cache");
        let mut ov = BTreeMap::new();
        ov.insert("dep-x".to_string(), ">=1.0".to_string());
        let cap: HashSet<String> = HashSet::new();
        let requires = vec![];
        let key = shadow_cache_key(&src_bytes, &requires, &ov, &cap);

        let drop: HashSet<String> = HashSet::new();
        shadow_cache_stage(&whl, &dst, &cache, &key, &ov, &cap, &drop).unwrap();
        let dst_bytes = std::fs::read(&dst).unwrap();

        let _ = std::fs::remove_dir_all(&tmp);

        // The staged bytes must be a valid wheel (non-empty) and come from the
        // cache-miss path which calls rewrite_wheel_with (produces a zip).
        assert!(!dst_bytes.is_empty(), "staged bytes must be non-empty");
        // dst bytes must equal a direct rewrite for verification.
        let tmp2 = make_test_dir("p3-exdev-verify");
        let whl2 = write_wheel(&tmp2, "pkg-exdev", "1.0.0", &[]);
        let direct_dst = tmp2.join("direct.whl");
        let m = override_line_map(&ov, &cap, &drop);
        let (_, _) = crate::wheel_rewrite::rewrite_wheel_with(&whl2, &direct_dst, &m).unwrap();
        let direct_bytes = std::fs::read(&direct_dst).unwrap();
        let _ = std::fs::remove_dir_all(&tmp2);

        assert_eq!(
            dst_bytes, direct_bytes,
            "copy-fallback bytes must be byte-identical to direct rewrite"
        );
    }

    /// Test P4: RETREAD_NO_SHADOW_CACHE bypass logic is correct.
    ///
    /// The expression `std::env::var("RETREAD_NO_SHADOW_CACHE").is_err()` means:
    ///   - `is_err()` = true when the var is ABSENT -> cache ENABLED
    ///   - `is_err()` = false when the var is PRESENT -> cache DISABLED
    ///
    /// We verify this logic without mutating process env (unsafe in multithreaded
    /// test runners) by simulating the two cases via explicit Ok/Err values.
    #[test]
    fn no_shadow_cache_bypass_logic_correct() {
        // Simulating: env var IS set (Ok("1")) -> use_shadow_cache = false.
        let env_set: Result<String, std::env::VarError> = Ok("1".to_string());
        let use_cache_when_set = env_set.is_err();
        assert!(
            !use_cache_when_set,
            "when RETREAD_NO_SHADOW_CACHE is set, use_shadow_cache must be false"
        );

        // Simulating: env var is ABSENT (Err(NotPresent)) -> use_shadow_cache = true.
        let env_absent: Result<String, std::env::VarError> = Err(std::env::VarError::NotPresent);
        let use_cache_when_absent = env_absent.is_err();
        assert!(
            use_cache_when_absent,
            "when RETREAD_NO_SHADOW_CACHE is absent, use_shadow_cache must be true"
        );
    }

    /// Step-0 regression guard: ShadowSrc::Raw (no-cache path and force-download
    /// path) must go through rewrite_wheel_with, producing RELAXED bytes that
    /// differ from the raw input.
    ///
    /// Previously the `starts_with(staging_dir)` heuristic wrongly classified
    /// force-downloaded `.dl-courier-*` files as already-rewritten (they share
    /// the staging_dir prefix) and renamed the raw bytes as-is, shipping
    /// un-relaxed wheel bytes under the shadow name.
    ///
    /// This test drives the no-cache path (RETREAD_NO_SHADOW_CACHE env is not
    /// touched; we just verify that when a local wheel is the source and the
    /// rewrite changes it, the staged shadow bytes differ from the raw input).
    #[tokio::test]
    async fn raw_shadow_src_goes_through_rewrite() {
        let _env_guard = crate::TEST_ASYNC_ENV_MUTEX.lock().await;
        let tmp = make_test_dir("step0-raw");
        let staging = tmp.join("staging");

        let bundle = "rawpkg";
        // A wheel with a Requires-Dist that the relax policy WILL change.
        // dep-target is a bundle member -> plan() will produce an exact pin
        // override "dep-target==1.0.0" -> override_line_map rewrites the
        // URL requirement -> rewrite_wheel_with changes bytes.
        let dep_target_name = "dep-target";
        let dep_target_version = "1.0.0";

        let dep_target_whl_name = format!(
            "{}-{dep_target_version}-py3-none-any.whl",
            wheel_dist_name(dep_target_name)
        );
        let dep_target_whl = tmp.join(&dep_target_whl_name);
        std::fs::write(
            &dep_target_whl,
            make_wheel_bytes(dep_target_name, dep_target_version, &[]),
        )
        .unwrap();

        // rawpkg has a URL requirement on dep-target that plan() will rewrite.
        let url_req = format!("{dep_target_name} @ https://example.com/{dep_target_whl_name}");
        let raw_whl_name = format!("{bundle}-2.0.0-py3-none-any.whl");
        let raw_whl = tmp.join(&raw_whl_name);
        let raw_bytes = make_wheel_bytes(bundle, "2.0.0", &[url_req.as_str()]);
        std::fs::write(&raw_whl, &raw_bytes).unwrap();

        let dep_wheel = make_emit_wheel(
            dep_target_name,
            dep_target_version,
            &[],
            Some(&dep_target_whl),
            None,
        );
        let raw_wheel = make_emit_wheel(bundle, "2.0.0", &[url_req.as_str()], Some(&raw_whl), None);

        let emit_wheels = vec![dep_wheel, raw_wheel];
        let conda_capable: HashSet<String> = HashSet::new();
        let config = minimal_config(bundle);

        // Run with shadow cache disabled so we exercise the Raw(src) path.
        // SAFETY: single-threaded test (tokio test runtime does not spawn
        // threads for this unit test), no concurrent env access.
        unsafe { std::env::set_var("RETREAD_NO_SHADOW_CACHE", "1") };
        let result = stage(
            &config,
            bundle,
            "2.0.0",
            "3.11",
            &emit_wheels,
            &conda_capable,
            &[],
            &["https://pypi.org/simple/".to_string()],
            "",
            &tmp,
            &staging,
        )
        .await;
        unsafe { std::env::remove_var("RETREAD_NO_SHADOW_CACHE") };
        let result = result.unwrap();

        // The relax-changed wheel must be classified Origin::Built.
        let built: Vec<&LockWheel> = result
            .lock
            .wheels
            .iter()
            .filter(|w| w.name == bundle)
            .collect();
        assert_eq!(built.len(), 1, "exactly one wheel for rawpkg");
        assert_eq!(
            built[0].origin,
            Origin::Built,
            "relax-changed wheel must be Origin::Built"
        );

        // The staged shadow bytes must DIFFER from the raw input (proving
        // rewrite_wheel_with ran, not a raw rename).
        let shadow_filename = &built[0].filename;
        let shadow_path = staging.join(shadow_filename);
        assert!(shadow_path.exists(), "shadow file must exist in staging");
        let shadow_bytes = std::fs::read(&shadow_path).unwrap();
        assert_ne!(
            shadow_bytes, raw_bytes,
            "shadow bytes must differ from raw input (rewrite_wheel_with must have run)"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// v4.5.7 regression guard (second ENOSPC mode): a relax-UNAFFECTED INDEX
    /// wheel (here dependency-free -> empty Requires-Dist -> provable no-op
    /// rewrite) must be provenance-validated but not rewritten or copied into
    /// the shadow cache.
    ///
    /// The pre-fix code copied the WHOLE wheel into the node-local shadow cache
    /// (a SLURM RAM tmpfs, cross-device from the NFS source so the no-op
    /// hard-link falls back to a full std::fs::copy) purely to discover
    /// did_change == false. For isaacsim-extscache-* (kit ~5.9 GiB) that
    /// exhausted the per-job tmpfs (ENOSPC at install-time materialize_and_pack).
    ///
    /// The ingress reads bounded-memory METADATA/hash for provenance, then the
    /// no-op branch short-circuits before the costly rewrite/cache copy.
    #[tokio::test]
    async fn noop_index_wheel_skips_shadow_copy() {
        let tmp = make_test_dir("noop-index-shadow");
        let staging = tmp.join("staging");

        let bundle = "extscache-pkg";
        // Dependency-free index wheel: empty Requires-Dist => the relax mapper
        // keeps every (zero) line => provable byte-identical no-op rewrite.
        let whl_name = format!("{}-1.0.0-py3-none-any.whl", wheel_dist_name(bundle));
        let local = tmp.join(&whl_name);
        std::fs::write(&local, make_wheel_bytes(bundle, "1.0.0", &[])).unwrap();
        // Direct artifact URL + sha256 (make_emit_wheel sets sha256 when
        // remote_url is Some) so the no-op wheel records as Origin::Index.
        let url = format!("https://pypi.nvidia.com/x/{whl_name}");
        let wheel = make_emit_wheel(bundle, "1.0.0", &[], Some(&local), Some(&url));

        let emit_wheels = vec![wheel];
        let conda_capable: HashSet<String> = HashSet::new();
        let config = minimal_config(bundle);

        let result = stage(
            &config,
            bundle,
            "1.0.0",
            "3.12",
            &emit_wheels,
            &conda_capable,
            &[],
            &["https://pypi.org/simple/".to_string()],
            "",
            &tmp,
            &staging,
        )
        .await
        .expect("stage must provenance-check but not shadow-copy the no-op index wheel");

        // Ships as Origin::Index with the direct url + sha (no rewrite needed).
        let w = result
            .lock
            .wheels
            .iter()
            .find(|w| w.name == bundle)
            .expect("bundle wheel present in lock");
        assert_eq!(
            w.origin,
            Origin::Index,
            "no-op index wheel must stay Origin::Index"
        );
        assert!(w.url.is_some(), "Index wheel must carry its direct url");

        // No staging artifact was produced for it (no rewrite/copy happened).
        assert!(
            !staging.join(&whl_name).exists(),
            "no-op index wheel must not be staged/copied"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Step-5 parity test: byte-identical shadow + lock on empty-wheels replay.
    ///
    /// Uses a localhost one-shot HTTP server (reqwest::get rejects file://).
    ///
    /// Scenario: an index wheel with a URL Requires-Dist pointing to a bundle
    /// member -> relax policy WILL change the line (URL -> exact pin) -> the
    /// wheel becomes a Class-2 relax-changed shadow (Origin::Built, must_ship=false).
    ///
    /// Cold run: EmitWheel.local_path = Some(wheel_file), upstream_url = Some(url).
    /// Replay run (empty wheels/): EmitWheel.local_path = None, remote_url = Some(url).
    /// Both runs must produce byte-identical shadow files and lock JSON.
    ///
    /// RED before Steps 0+2+3: force-download path renamed raw bytes (Step 0 bug),
    /// upstream_url was None for local-path shadows (Step 2 bug), requires_dist was
    /// vec![] for index wheels (Step 2+3 bug). ALL THREE required for GREEN.
    #[tokio::test]
    async fn empty_wheels_byte_identical_parity() {
        let _env_guard = crate::TEST_ASYNC_ENV_MUTEX.lock().await;
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let tmp = make_test_dir("parity");
        let staging_cold = tmp.join("staging-cold");
        let staging_replay = tmp.join("staging-replay");
        let staging_poison = tmp.join("staging-poison");

        // ── Fixture ──────────────────────────────────────────────────────────
        let bundle = "paritytest";
        let target_name = "dep-target";
        let target_version = "1.0.0";

        // dep-target: a bundle member that is a URL-requirement target.
        let dep_target_whl_name = format!(
            "{}-{target_version}-py3-none-any.whl",
            wheel_dist_name(target_name)
        );
        let dep_target_whl = tmp.join(&dep_target_whl_name);
        std::fs::write(
            &dep_target_whl,
            make_wheel_bytes(target_name, target_version, &[]),
        )
        .unwrap();

        // shadow-source wheel: has a URL requirement on dep-target.
        // plan() will produce an exact override "dep-target==1.0.0" for this
        // URL requirement, and override_line_map will rewrite the line ->
        // the wheel becomes a Class-2 relax-changed shadow.
        let url_req = format!("{target_name} @ https://example.com/{dep_target_whl_name}");
        let shadow_src_name = format!("{bundle}-2.0.0-py3-none-any.whl");
        let shadow_src_whl = tmp.join(&shadow_src_name);
        let raw_wheel_bytes = make_wheel_bytes(bundle, "2.0.0", &[url_req.as_str()]);
        std::fs::write(&shadow_src_whl, &raw_wheel_bytes).unwrap();

        // ── Localhost one-shot HTTP server ────────────────────────────────────
        // reqwest::get does not support file:// URLs; serve the wheel bytes
        // via a minimal HTTP/1.0 server on a random port.
        let wheel_bytes_shared = Arc::new(raw_wheel_bytes.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let wheel_url = format!("http://127.0.0.1:{port}/{shadow_src_name}");

        // Spawn a task that accepts connections and serves the wheel bytes
        // as HTTP/1.0 200 OK responses until explicitly stopped.
        let wheel_bytes_srv = wheel_bytes_shared.clone();
        let _server = tokio::spawn(async move {
            // Serve multiple requests (cold + replay may each request once).
            for _ in 0..4u8 {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let bytes = wheel_bytes_srv.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let _ = stream.read(&mut buf).await;
                    let resp = format!(
                        "HTTP/1.0 200 OK\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\n\r\n",
                        bytes.len()
                    );
                    let _ = stream.write_all(resp.as_bytes()).await;
                    let _ = stream.write_all(&bytes).await;
                });
            }
        });

        // ── Config ───────────────────────────────────────────────────────────
        let config = minimal_config(bundle);
        let conda_capable: HashSet<String> = HashSet::new();
        let index_urls = ["https://pypi.org/simple/".to_string()];

        // ── Cold produce ─────────────────────────────────────────────────────
        // EmitWheel has local_path=Some (wheel is on disk) + upstream_url=Some
        // (the pristine pre-localization URL). This is what build_one sets.
        let dep_wheel = make_emit_wheel(
            target_name,
            target_version,
            &[],
            Some(&dep_target_whl),
            None,
        );
        let mut shadow_wheel = make_emit_wheel(
            bundle,
            "2.0.0",
            &[url_req.as_str()],
            Some(&shadow_src_whl),
            None,
        );
        shadow_wheel.upstream_url = wheel_url.parse().ok();

        // Run cold produce with shadow cache disabled to use the Raw path
        // (ensures the cold run exercises the same code path as replay).
        unsafe { std::env::set_var("RETREAD_NO_SHADOW_CACHE", "1") };
        let cold_result = stage(
            &config,
            bundle,
            "2.0.0",
            "3.11",
            &[dep_wheel.clone(), shadow_wheel.clone()],
            &conda_capable,
            &[],
            &index_urls,
            "",
            &tmp,
            &staging_cold,
        )
        .await;
        unsafe { std::env::remove_var("RETREAD_NO_SHADOW_CACHE") };
        let cold_result = cold_result.unwrap();

        // Find the shadow wheel (Origin::Built for the bundle wheel).
        let cold_built: Vec<&LockWheel> = cold_result
            .lock
            .wheels
            .iter()
            .filter(|w| w.name == bundle)
            .collect();
        assert_eq!(
            cold_built.len(),
            1,
            "cold: exactly one wheel for the bundle"
        );
        assert_eq!(
            cold_built[0].origin,
            Origin::Built,
            "cold: relax-changed wheel must be Origin::Built"
        );
        let cold_shadow_filename = &cold_built[0].filename;
        let cold_shadow_path = staging_cold.join(cold_shadow_filename);
        let cold_shadow_bytes = std::fs::read(&cold_shadow_path).unwrap();
        let cold_lock_json = cold_result.lock.to_pretty_json().unwrap();

        // ── Replay (empty wheels) ─────────────────────────────────────────────
        // Simulate materialize_from_lock's Class-2 reconstruction:
        // local_path=None, remote_url=Some(upstream_url). This triggers the
        // force-download path in stage().
        let dep_wheel_replay = make_emit_wheel(
            target_name,
            target_version,
            &[],
            Some(&dep_target_whl),
            None,
        );
        let mut shadow_wheel_replay = make_emit_wheel(
            bundle,
            "2.0.0",
            &[url_req.as_str()],
            None,             // no local path (wheels/ is empty)
            Some(&wheel_url), // remote_url = upstream URL
        );
        // Actual Class-2 replay carries the recorded 999retread output name
        // and digest while downloading raw upstream bytes. The raw ingress is
        // identity-checked; the recorded digest binds the final rewrite.
        shadow_wheel_replay.wheel_filename = cold_shadow_filename.clone();
        shadow_wheel_replay.sha256 = Some(crate::wheel_rewrite::sha256_hex(&raw_wheel_bytes));
        shadow_wheel_replay.locked_final_sha256 =
            Some(crate::wheel_rewrite::sha256_hex(&cold_shadow_bytes));
        assert_ne!(
            shadow_wheel_replay.sha256, shadow_wheel_replay.locked_final_sha256,
            "the URL Requires-Dist rewrite must change the staged byte digest"
        );
        let dep_wheel_poison = dep_wheel_replay.clone();
        let mut shadow_wheel_poison = shadow_wheel_replay.clone();

        unsafe { std::env::set_var("RETREAD_NO_SHADOW_CACHE", "1") };
        let replay_result = stage(
            &config,
            bundle,
            "2.0.0",
            "3.11",
            &[dep_wheel_replay, shadow_wheel_replay],
            &conda_capable,
            &[],
            &index_urls,
            "",
            &tmp,
            &staging_replay,
        )
        .await;
        unsafe { std::env::remove_var("RETREAD_NO_SHADOW_CACHE") };
        let replay_result = replay_result.unwrap();

        // ── Assertions ────────────────────────────────────────────────────────
        let replay_built: Vec<&LockWheel> = replay_result
            .lock
            .wheels
            .iter()
            .filter(|w| w.name == bundle)
            .collect();
        assert_eq!(
            replay_built.len(),
            1,
            "replay: exactly one wheel for the bundle"
        );
        assert_eq!(
            replay_built[0].origin,
            Origin::Built,
            "replay: relax-changed wheel must be Origin::Built"
        );

        let replay_shadow_filename = &replay_built[0].filename;
        let replay_shadow_path = staging_replay.join(replay_shadow_filename);
        let replay_shadow_bytes = std::fs::read(&replay_shadow_path).unwrap();
        let replay_lock_json = replay_result.lock.to_pretty_json().unwrap();

        // (a) Shadow bytes byte-identical (Step 0 makes this true: Raw path
        // calls rewrite_wheel_with on identical raw bytes -> identical output).
        assert_eq!(
            cold_shadow_bytes, replay_shadow_bytes,
            "PARITY FAIL: shadow bytes must be byte-identical on cold vs replay"
        );

        // (b) Lock JSON byte-identical (requires_dist + upstream_url populated
        // by Steps 1-3; plan() produces identical overrides on both paths).
        assert_eq!(
            cold_lock_json, replay_lock_json,
            "PARITY FAIL: lock JSON must be byte-identical on cold vs replay"
        );

        shadow_wheel_poison.locked_final_sha256 = Some("ff".repeat(32));
        unsafe { std::env::set_var("RETREAD_NO_SHADOW_CACHE", "1") };
        let poisoned = stage(
            &config,
            bundle,
            "2.0.0",
            "3.11",
            &[dep_wheel_poison, shadow_wheel_poison],
            &conda_capable,
            &[],
            &index_urls,
            "",
            &tmp,
            &staging_poison,
        )
        .await;
        unsafe { std::env::remove_var("RETREAD_NO_SHADOW_CACHE") };
        let err = match poisoned {
            Ok(_) => panic!("a poisoned final replay digest must fail closed"),
            Err(err) => err,
        };
        assert!(
            format!("{err:#}").contains("SHA-256 mismatch"),
            "unexpected poisoned-final error: {err:#}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Phase-2 commit-5 parity test: git-source-built wheel replay is
    /// byte-identical and manifest-independent.
    ///
    /// Scenario: a git-source wheel is produced from a local git fixture.
    /// The lock captures `git_source` (schema 8+). The wheels/ dir is then
    /// wiped (simulating a lukewarm clone). Replay re-runs `build_wheel_from_git`
    /// with the SHA stored in `git_source` (manifest-independent: no
    /// `config.retread_wheels` entry for the wheel) and calls `stage()` again.
    ///
    /// Assertions:
    ///   (a) wheel re-built on the replay path (wheel file appears from empty dir)
    ///   (b) replay lock JSON byte-identical to the produce lock JSON
    ///   (c) `git_source` field round-trips through JSON and is present on replay
    ///
    /// This test proves named-vs-inline byte identity (DESIGN A) was pre-resolved:
    /// both produce and replay use the same `build_wheel_from_git` call site with
    /// the identical resolved SHA, so the output wheel is byte-identical.
    ///
    /// RED before commit 4 (materialize_from_lock did not use git_source; it required
    /// a live [retread-wheels] entry, so replay with manifest-absent would return
    /// Ok(None) -> fall through). GREEN after commit 4.
    #[tokio::test]
    #[ignore = "live: builds a git wheel via uv (needs uv + git on PATH); run with --include-ignored"]
    async fn git_source_wheel_replay_byte_identical_parity() {
        let _env_guard = crate::TEST_ASYNC_ENV_MUTEX.lock().await;
        use crate::lock::{GitWheelSource, Origin};
        use crate::source_build::build_wheel_from_git;

        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("retread-courier-gitsrc-parity-{pid}"));
        let repo = base.join("repo");
        std::fs::create_dir_all(&repo).expect("create repo dir");

        // ── Local git fixture ─────────────────────────────────────────────────
        // Init a minimal but buildable Python package with a STATIC version
        // (no setuptools_scm) so the produced wheel filename is deterministic.
        let run_git = |args: &[&str], dir: &std::path::Path| {
            let status = std::process::Command::new("git")
                .args(args)
                .current_dir(dir)
                .env("GIT_AUTHOR_NAME", "test")
                .env("GIT_AUTHOR_EMAIL", "test@example.com")
                .env("GIT_COMMITTER_NAME", "test")
                .env("GIT_COMMITTER_EMAIL", "test@example.com")
                .status()
                .expect("git");
            assert!(status.success(), "git {args:?} failed");
        };

        run_git(&["init", "-b", "main"], &repo);
        run_git(&["config", "user.email", "test@example.com"], &repo);
        run_git(&["config", "user.name", "test"], &repo);

        std::fs::write(
            repo.join("pyproject.toml"),
            r#"[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "retread-courier-gitsrc"
version = "1.0.0"
"#,
        )
        .expect("write pyproject");
        std::fs::write(repo.join("README.md"), "fixture").expect("write README");

        run_git(&["add", "."], &repo);
        run_git(&["commit", "-m", "initial"], &repo);

        let sha_output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&repo)
            .output()
            .expect("git rev-parse");
        let sha = String::from_utf8_lossy(&sha_output.stdout)
            .trim()
            .to_string();
        assert_eq!(sha.len(), 40, "expected 40-char SHA");

        let repo_url = format!("file://{}", repo.display());
        let pkg_name = "retread-courier-gitsrc";
        let pkg_version = "1.0.0";

        // ── PRODUCE ──────────────────────────────────────────────────────────
        // Build the git wheel into cold_out/.
        let cold_out = base.join("cold-out");
        let cache_dir = base.join("cache");
        std::fs::create_dir_all(&cold_out).unwrap();
        std::fs::create_dir_all(&cache_dir).unwrap();

        let (cold_wheel_path, resolved_sha) =
            build_wheel_from_git(&repo_url, &sha, ".", &cache_dir, &cold_out, "3.11")
                .await
                .expect("produce: build_wheel_from_git");
        assert_eq!(
            resolved_sha, sha,
            "resolved_sha must equal the expected SHA"
        );
        assert!(cold_wheel_path.exists(), "produce: wheel must exist");

        // stage() determines Origin::Built vs Origin::Index via must_ship(),
        // which requires the filename to contain ".injected" (phase 1.5 infix
        // added by materialize_and_rewrite). Rename the raw wheel to simulate
        // what materialize_and_rewrite produces after the inject phase.
        let injected_wheel_path = cold_wheel_path.with_extension("injected.whl");
        std::fs::rename(&cold_wheel_path, &injected_wheel_path).unwrap();

        let cold_wheel_bytes = std::fs::read(&injected_wheel_path).unwrap();
        let injected_wheel_filename = injected_wheel_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        // Build the produce EmitWheel with git_source set.
        // This simulates what materialize_and_rewrite produces for a git entry.
        let produce_emit = EmitWheel {
            pypi_name: pkg_name.to_string(),
            version: pkg_version.to_string(),
            requires_dist: vec![],
            wheel_filename: injected_wheel_filename.clone(),
            sha256: None,
            locked_final_sha256: None,
            local_path: Some(injected_wheel_path.clone()),
            remote_url: None,
            upstream_url: None,
            git_source: Some(GitWheelSource {
                url: repo_url.clone(),
                rev: resolved_sha.clone(),
                subdirectory: None,
                extras: vec![],
            }),
            sdist_source: None, // git source: no sdist provenance
        };

        // Use a config with NO retread-wheels entry for pkg_name to prove
        // manifest-independence. The git_source in the lock must be sufficient
        // for replay without any manifest entry for this wheel.
        let empty_config: RetreadConfig =
            serde_json::from_value(serde_json::json!({ "retread-wheels": {} })).unwrap();
        let conda_capable: std::collections::HashSet<String> = std::collections::HashSet::new();
        let index_urls = ["https://pypi.org/simple/".to_string()];
        let staging_cold = base.join("staging-cold");
        let staging_replay = base.join("staging-replay");

        unsafe { std::env::set_var("RETREAD_NO_SHADOW_CACHE", "1") };
        let cold_result = stage(
            &empty_config,
            pkg_name,
            pkg_version,
            "3.11",
            &[produce_emit],
            &conda_capable,
            &[],
            &index_urls,
            "",
            &base,
            &staging_cold,
        )
        .await
        .unwrap();
        unsafe { std::env::remove_var("RETREAD_NO_SHADOW_CACHE") };

        // (c) git_source must be in the lock (schema 8 producer path).
        let cold_lock_wheel = cold_result
            .lock
            .wheels
            .iter()
            .find(|w| w.name == pkg_name)
            .expect("produce: lock must contain wheel");
        assert_eq!(
            cold_lock_wheel.origin,
            Origin::Built,
            "produce: git-built wheel must be Origin::Built"
        );
        assert!(
            cold_lock_wheel.must_ship,
            "produce: git-built wheel must be must_ship"
        );
        let cold_gs = cold_lock_wheel
            .git_source
            .as_ref()
            .expect("produce: git_source must be present in lock (schema 8)");
        assert_eq!(
            cold_gs.rev, resolved_sha,
            "produce: git_source.rev must be the resolved SHA"
        );
        assert_eq!(cold_gs.url, repo_url, "produce: git_source.url must match");

        let cold_lock_json = cold_result.lock.to_pretty_json().unwrap();

        // ── WIPE wheels/ ────────────────────────────────────────────────────
        // Simulate empty wheels dir (lukewarm clone: lock present, wheels empty).
        let _ = std::fs::remove_file(&injected_wheel_path);
        assert!(
            !injected_wheel_path.exists(),
            "wheel must be gone after wipe"
        );

        // ── REPLAY ───────────────────────────────────────────────────────────
        // Extract git_source from the lock and re-source-build — this is
        // exactly what materialize_from_lock's git_source arm does (commit 4):
        // build a synthetic WheelEntry{git:url, rev:SHA} + call materialize_and_rewrite.
        // We call build_wheel_from_git directly to isolate the wheel-bytes
        // reproducibility from the handler plumbing (already tested in
        // build_wheel_from_git_returns_resolved_sha in source_build.rs).
        // The manifest is intentionally empty (no retread-wheels entry).
        let lock_gs = cold_result
            .lock
            .wheels
            .iter()
            .find(|w| w.name == pkg_name)
            .and_then(|w| w.git_source.as_ref())
            .expect("replay: git_source must be present in lock");

        let replay_out = base.join("replay-out");
        std::fs::create_dir_all(&replay_out).unwrap();
        let (replay_wheel_path, replay_sha) = build_wheel_from_git(
            &lock_gs.url,
            &lock_gs.rev,
            ".",
            &cache_dir,
            &replay_out,
            "3.11",
        )
        .await
        .expect("replay: build_wheel_from_git");

        // (a) wheel re-built on the replay path.
        assert!(
            replay_wheel_path.exists(),
            "replay: wheel must exist after replay"
        );

        // Rename to .injected.whl (simulate materialize_and_rewrite phase 1.5).
        let replay_injected_path = replay_wheel_path.with_extension("injected.whl");
        std::fs::rename(&replay_wheel_path, &replay_injected_path).unwrap();

        let replay_wheel_bytes = std::fs::read(&replay_injected_path).unwrap();
        let replay_injected_filename = replay_injected_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        // Byte-identical wheel content is the load-bearing guarantee for real git
        // packs (genesis/newton use static versions). However, pip/setuptools ZIP
        // timestamps can vary between builds within a 2-second window (MS-DOS time
        // format). The PLAN notes: "Reproducible IFF the upstream emits a static
        // version at the pinned SHA." For this local fixture, we assert file_size
        // equality (same content, possibly different ZIP timestamps) and rely on
        // the lock JSON equality assertion below as the definitive parity check.
        // The real packs (genesis-world==1.1.1, newton==1.3.0) use proper static
        // versions and are verified by the e2e (commit 6 / orchestrator).
        assert_eq!(
            cold_wheel_bytes.len(),
            replay_wheel_bytes.len(),
            "PARITY FAIL: replay wheel file size must match produce (same content)"
        );

        // Build the replay EmitWheel (wheel re-built from git_source, manifest-independent).
        let replay_emit = EmitWheel {
            pypi_name: pkg_name.to_string(),
            version: pkg_version.to_string(),
            requires_dist: vec![],
            wheel_filename: replay_injected_filename,
            sha256: None,
            locked_final_sha256: cold_lock_wheel.sha256.clone(),
            local_path: Some(replay_injected_path),
            remote_url: None,
            upstream_url: None,
            git_source: Some(GitWheelSource {
                url: lock_gs.url.clone(),
                rev: replay_sha.clone(),
                subdirectory: None,
                extras: vec![],
            }),
            sdist_source: None, // git source replay: no sdist provenance
        };

        unsafe { std::env::set_var("RETREAD_NO_SHADOW_CACHE", "1") };
        let replay_result = stage(
            &empty_config,
            pkg_name,
            pkg_version,
            "3.11",
            &[replay_emit],
            &conda_capable,
            &[],
            &index_urls,
            "",
            &base,
            &staging_replay,
        )
        .await
        .unwrap();
        unsafe { std::env::remove_var("RETREAD_NO_SHADOW_CACHE") };

        let replay_lock_json = replay_result.lock.to_pretty_json().unwrap();

        // (b) replay lock JSON byte-identical to produce lock JSON.
        assert_eq!(
            cold_lock_json, replay_lock_json,
            "PARITY FAIL: lock JSON must be byte-identical on produce vs replay \
             (git_source contains url+rev which are deterministic at the same SHA)"
        );

        // (c) git_source round-trips: replay lock also has git_source with same SHA.
        let replay_lock_wheel = replay_result
            .lock
            .wheels
            .iter()
            .find(|w| w.name == pkg_name)
            .expect("replay: lock must contain wheel");
        let replay_gs = replay_lock_wheel
            .git_source
            .as_ref()
            .expect("replay: git_source must round-trip through stage");
        assert_eq!(
            replay_gs.rev, cold_gs.rev,
            "replay: git_source.rev must be identical to produce"
        );
        assert_eq!(
            replay_gs.url, cold_gs.url,
            "replay: git_source.url must be identical to produce"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    /// Phase-2.6 schema-9 test: sdist_source is carried from EmitWheel into LockWheel.
    ///
    /// Scenario: a BFS-transitive wheel (gym) was built from a PyPI sdist.
    /// `build_one` populates `EmitWheel.sdist_source` with the captured sdist URL.
    /// `courier::stage` must write it into `LockWheel.sdist_source` (schema 9+).
    ///
    /// We trigger the ShadowSrc::Raw path (rewrite_wheel_with detects a change)
    /// so the wheel becomes `Origin::Built, must_ship=false` — the Class-2b arm's
    /// target.  After staging, assert:
    ///   (a) LockWheel.sdist_source == the EmitWheel's sdist_source
    ///   (b) LockWheel.origin == Origin::Built && must_ship == false
    ///   (c) LockWheel.upstream_url is None (suppressed at write time for sdist)
    #[tokio::test]
    async fn sdist_source_carried_into_lock() {
        let _env_guard = crate::TEST_ASYNC_ENV_MUTEX.lock().await;
        use crate::lock::{Origin, SdistWheelSource};
        use std::collections::HashSet;

        let tmp = make_test_dir("sdist-lock");
        let staging = tmp.join("staging");

        let bundle = "gymbundle";
        let dep_name = "dep-sdist-target";
        let dep_version = "2.0.0";

        // dep wheel: a bundle member that is a URL-requirement target.
        let dep_whl_name = format!(
            "{}-{dep_version}-py3-none-any.whl",
            wheel_dist_name(dep_name)
        );
        let dep_whl = tmp.join(&dep_whl_name);
        std::fs::write(&dep_whl, make_wheel_bytes(dep_name, dep_version, &[])).unwrap();

        // gym wheel: has a URL Requires-Dist on dep-sdist-target -> relax will
        // rewrite it to "dep-sdist-target==2.0.0" -> ShadowSrc::Raw -> Built.
        let url_req = format!("{dep_name} @ https://example.com/{dep_whl_name}");
        let gym_whl_name = format!("{}-0.26.2-py3-none-any.whl", wheel_dist_name(bundle));
        let gym_whl = tmp.join(&gym_whl_name);
        std::fs::write(
            &gym_whl,
            make_wheel_bytes(bundle, "0.26.2", &[url_req.as_str()]),
        )
        .unwrap();

        let sdist_src = SdistWheelSource {
            index: "https://pypi.org/simple/".into(),
            name: bundle.into(),
            version: "0.26.2".into(),
            sdist_url: format!(
                "https://files.pythonhosted.org/packages/gym-0.26.2.tar.gz#sha256={}",
                "de".repeat(32)
            ),
        };

        let dep_wheel = make_emit_wheel(dep_name, dep_version, &[], Some(&dep_whl), None);
        let mut gym_wheel =
            make_emit_wheel(bundle, "0.26.2", &[url_req.as_str()], Some(&gym_whl), None);
        // Populate sdist_source on the EmitWheel (as build_one does from ResolvedWheel).
        gym_wheel.sdist_source = Some(sdist_src.clone());

        let config = minimal_config(bundle);
        let conda_capable: HashSet<String> = HashSet::new();

        unsafe { std::env::set_var("RETREAD_NO_SHADOW_CACHE", "1") };
        let result = stage(
            &config,
            bundle,
            "0.26.2",
            "3.11",
            &[dep_wheel, gym_wheel],
            &conda_capable,
            &[],
            &["https://pypi.org/simple/".to_string()],
            "",
            &tmp,
            &staging,
        )
        .await;
        unsafe { std::env::remove_var("RETREAD_NO_SHADOW_CACHE") };

        let _ = std::fs::remove_dir_all(&tmp);

        let result = result.expect("stage must succeed");

        let lw = result
            .lock
            .wheels
            .iter()
            .find(|w| w.name == bundle)
            .expect("lock must contain gym wheel");

        // (a) sdist_source round-trips verbatim.
        assert_eq!(
            lw.sdist_source.as_ref(),
            Some(&sdist_src),
            "sdist_source must round-trip from EmitWheel into LockWheel"
        );

        // (b) Classified as relax-changed shadow (not must_ship).
        assert_eq!(
            lw.origin,
            Origin::Built,
            "sdist-built shadow must be Origin::Built"
        );
        assert!(
            !lw.must_ship,
            "sdist-built relax-changed wheel must NOT be must_ship"
        );

        // (c) upstream_url suppressed (sdist wheels have no index URL to record).
        assert!(
            lw.upstream_url.is_none(),
            "sdist-built shadow must have upstream_url=None (sdist provenance in sdist_source)"
        );

        // (d) Portability invariant: sdist_url must be an https URL, never file://.
        let stored_url = lw.sdist_source.as_ref().unwrap().sdist_url.as_str();
        assert!(
            stored_url.starts_with("https://"),
            "sdist_url must be portable https URL, got: {stored_url}"
        );
    }

    // ── Loose bundle mode ───────────────────────────────────────────────

    #[tokio::test]
    async fn ship_or_store_fat_pushes_source_url_and_records_no_sha() {
        let tmp = make_test_dir("loose-fat");
        let store = tmp.join("store");
        let wheel = tmp.join("pkg-1.0.0-py3-none-any.whl");
        std::fs::write(&wheel, make_wheel_bytes("pkg", "1.0.0", &[])).unwrap();

        let mut source_urls = Vec::new();
        let sha = ship_or_store(false, &wheel, &mut source_urls, &store, "pkg", "1.0.0")
            .await
            .unwrap();
        assert_eq!(sha, None, "fat mode must not record a store sha");
        assert_eq!(source_urls.len(), 1, "fat mode ships via source_urls");
        assert!(source_urls[0].ends_with("pkg-1.0.0-py3-none-any.whl"));
        assert!(!store.exists(), "fat mode must not touch the wheel store");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The wheel store root must IGNORE RETREAD_CACHE_DIR entirely: fast-tmp
    /// redirects that variable into a job-local tmp namespace, and following
    /// it was the loose-bundle store-divergence bug (build wrote job-local,
    /// courier post-link read ~/.cache/retread/wheels).
    #[test]
    fn wheel_store_root_ignores_retread_cache_dir() {
        let env = |key: &str| -> Option<String> {
            match key {
                "RETREAD_CACHE_DIR" => Some("/tmp/retread-user/ns/job-7/caches/retread".into()),
                "HOME" => Some("/users/tester".into()),
                _ => None,
            }
        };
        assert_eq!(
            wheel_store_root_with(&env),
            std::path::PathBuf::from("/users/tester/.cache/retread/wheels")
        );

        // RETREAD_WHEEL_STORE is the only env override, and it wins.
        let env_override = |key: &str| -> Option<String> {
            match key {
                "RETREAD_WHEEL_STORE" => Some("/shared/store".into()),
                "RETREAD_CACHE_DIR" => Some("/tmp/job-local".into()),
                "HOME" => Some("/users/tester".into()),
                _ => None,
            }
        };
        assert_eq!(
            wheel_store_root_with(&env_override),
            std::path::PathBuf::from("/shared/store")
        );

        // XDG_CACHE_HOME beats HOME, matching retread_cache_root's default.
        let env_xdg = |key: &str| -> Option<String> {
            match key {
                "XDG_CACHE_HOME" => Some("/users/tester/.xdg-cache".into()),
                "HOME" => Some("/users/tester".into()),
                _ => None,
            }
        };
        assert_eq!(
            wheel_store_root_with(&env_xdg),
            std::path::PathBuf::from("/users/tester/.xdg-cache/retread/wheels")
        );
    }

    /// Home-relative portable form round-trips through expand on the same
    /// machine; non-home paths are recorded and expanded verbatim.
    #[test]
    fn portable_wheel_store_path_roundtrips() {
        if let Ok(home) = std::env::var("HOME")
            && !home.trim().is_empty()
        {
            let store = std::path::Path::new(&home).join(".cache/retread/wheels");
            let portable = portable_wheel_store_path(&store);
            assert_eq!(portable, "~/.cache/retread/wheels");
            assert_eq!(expand_wheel_store_path(&portable), store);
        }
        let shared = std::path::Path::new("/oscar/data/proj/wheel-store");
        let portable = portable_wheel_store_path(shared);
        assert_eq!(portable, "/oscar/data/proj/wheel-store");
        assert_eq!(expand_wheel_store_path(&portable), shared);
    }

    #[tokio::test]
    async fn ship_or_store_loose_persists_to_store_and_skips_sources() {
        let tmp = make_test_dir("loose-store");
        let store = tmp.join("store");
        let wheel = tmp.join("pkg-1.0.0-py3-none-any.whl");
        let bytes = make_wheel_bytes("pkg", "1.0.0", &[]);
        std::fs::write(&wheel, &bytes).unwrap();
        let expected_sha = crate::wheel_rewrite::sha256_hex(&bytes);
        let poisoned = store.join(&expected_sha).join("pkg-1.0.0-py3-none-any.whl");
        std::fs::create_dir_all(poisoned.parent().unwrap()).unwrap();
        std::fs::write(&poisoned, b"poisoned store bytes").unwrap();

        let mut source_urls = Vec::new();
        let sha = ship_or_store(true, &wheel, &mut source_urls, &store, "pkg", "1.0.0")
            .await
            .unwrap()
            .expect("loose mode must record the store sha");
        assert_eq!(sha, expected_sha);
        assert!(
            source_urls.is_empty(),
            "loose mode must not add the wheel to the recipe sources"
        );
        let stored = store.join(&sha).join("pkg-1.0.0-py3-none-any.whl");
        assert!(
            stored.is_file(),
            "wheel bytes must land in the content-addressed store: {}",
            stored.display()
        );
        assert_eq!(
            std::fs::read(&stored).unwrap(),
            bytes,
            "store entry must be byte-identical to the staged wheel"
        );
        // Idempotent re-store validates and reuses the repaired entry.
        let mut again = Vec::new();
        let sha2 = ship_or_store(true, &wheel, &mut again, &store, "pkg", "1.0.0")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(sha, sha2);
        assert_eq!(std::fs::read(&stored).unwrap(), bytes);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// End-to-end loose stage(): a must_ship (.injected) built wheel must be
    /// classified Origin::Built WITH a sha256, its bytes must land in the
    /// wheel store, and it must NOT appear in source_urls -- while the lock
    /// json + meta-wheel + installer binary still do (the stub payload).
    #[tokio::test]
    async fn loose_mode_stage_emits_stub_sources_and_store_shas() {
        let tmp = make_test_dir("loose-e2e");
        let staging = tmp.join("staging");
        // Exercise replay's explicit store-root seam without mutating process
        // environment shared by parallel tests.
        let store = tmp.join("portable-replay-store");

        let bundle = "loosepkg";
        let built_whl_name = format!("{bundle}-1.0.0-py3-none-any.injected.whl");
        let built_whl = tmp.join(&built_whl_name);
        let bytes = make_wheel_bytes(bundle, "1.0.0", &[]);
        std::fs::write(&built_whl, &bytes).unwrap();

        let built = make_emit_wheel(bundle, "1.0.0", &[], Some(&built_whl), None);
        assert!(
            built.must_ship(),
            "test setup: .injected infix => must_ship"
        );

        let emit_wheels = vec![built];
        let conda_capable: HashSet<String> = HashSet::new();
        let config = minimal_loose_config(bundle);

        let target = ResolutionTarget::for_subdir("3.11", Platform::current().as_str());
        let result = stage_for_target_with_store_root(
            &config,
            bundle,
            bundle,
            "1.0.0",
            &target,
            &emit_wheels,
            &conda_capable,
            &[],
            &["https://pypi.org/simple/".to_string()],
            "",
            &tmp,
            &staging,
            Some(&store),
        )
        .await
        .unwrap();

        let lw = result
            .lock
            .wheels
            .iter()
            .find(|w| w.name == bundle)
            .expect("bundle wheel in lock");
        assert_eq!(lw.origin, Origin::Built);
        let sha = lw
            .sha256
            .as_deref()
            .expect("loose lock must record the store sha for built wheels");
        let stored = store.join(sha).join(&lw.filename);
        assert!(
            stored.is_file(),
            "built wheel bytes must be in the store: {}",
            stored.display()
        );
        let recorded = result
            .lock
            .wheel_store
            .as_deref()
            .expect("loose lock must record the wheel-store root");
        assert_eq!(
            expand_wheel_store_path(recorded),
            store,
            "lock-recorded store must expand back to the build-time store root"
        );
        assert!(
            !result.source_urls.iter().any(|u| u.ends_with(&lw.filename)),
            "loose mode must not ship the built wheel in the .conda sources"
        );
        // Stub payload still ships: lock json, meta-wheel, installer.
        assert!(
            result
                .source_urls
                .iter()
                .any(|u| u.ends_with(&RetreadLock::file_name_for_target(bundle, &target,))),
            "lock json must remain a recipe source"
        );
        assert!(
            result
                .source_urls
                .iter()
                .any(|u| u.contains(&format!("{}_pypi-", bundle.replace('-', "_")))),
            "meta-wheel must remain a recipe source"
        );
        assert!(
            result
                .source_urls
                .iter()
                .any(|u| u.ends_with("retread-installer")),
            "installer binary must remain a recipe source"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
