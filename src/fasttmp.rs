//! First-class fast-tmp support for slow workspace filesystems.
//!
//! The wrapper-facing entry point is [`engage`]. The backend-facing entry
//! point is [`engage_backend`], which prepares the same job-scoped namespace
//! without mutating workspace symlinks.

use std::collections::{HashMap, HashSet};
use std::ffi::{CString, OsStr, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};
use xxhash_rust::xxh3::xxh3_64;

const DEFAULT_TMP_ROOT: &str = "/tmp";
const DEFAULT_ESTIMATE_BYTES: u64 = 80 * 1024 * 1024 * 1024;
const PROBE_THRESHOLD_MS: f64 = 5.0;
/// Job-local Pixi config overlay. Pixi 0.70 no longer reads the legacy
/// `PIXI_DETACHED_ENVIRONMENTS` variable; detached environments must be
/// enabled in config, while the cache directory has a supported env override.
const PIXI_FASTTMP_CONFIG: &str = "pixi-fasttmp-config.toml";
/// Tracks the overlay installed by retread so a later job can distinguish it
/// from a user-supplied `PIXI_CONFIG_FILE` and restore the latter.
const RETREAD_PIXI_CONFIG_MARKER: &str = "RETREAD_FAST_TMP_PIXI_CONFIG_FILE";
const RETREAD_BASE_PIXI_CONFIG: &str = "RETREAD_FAST_TMP_BASE_PIXI_CONFIG_FILE";
const RETREAD_MANAGED_KEYS: &str = "RETREAD_FAST_TMP_MANAGED_KEYS";
const RETREAD_BASE_ENV_JSON: &str = "RETREAD_FAST_TMP_BASE_ENV_JSON";
const RETREAD_EXPECTED_ENV_JSON: &str = "RETREAD_FAST_TMP_EXPECTED_ENV_JSON";
const FAST_WORKSPACE_LINK_LOCK: &str = ".retread-fast-envs-link.lock";
const RETREAD_BLD_OWNER_MARKER: &str = ".retread-bld-owner";
const RETREAD_BLD_OWNER_FORMAT: &str = "pixi-build-retread fast-tmp bld v1";
const BLD_IDLE_THRESHOLD: Duration = Duration::from_secs(90);

/// User-facing variables Retread may own. Inherited ownership metadata is
/// untrusted because `--print-env` is evaluated by a shell, so cleanup must
/// never emit an inherited key outside this fixed allowlist.
const MANAGEABLE_ENV_KEYS: &[&str] = &[
    "PIXI_CACHE_DIR",
    "RATTLER_CACHE_DIR",
    "UV_CACHE_DIR",
    "RETREAD_CACHE_DIR",
    "UV_LOCK_TIMEOUT",
    "PIXI_CONFIG_FILE",
    "PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR",
    "PIXI_DETACHED_ENVIRONMENTS",
];

#[cfg(test)]
const FAST_ENV_KEYS: &[&str] = &[
    "PIXI_CACHE_DIR",
    "RATTLER_CACHE_DIR",
    "UV_CACHE_DIR",
    "RETREAD_CACHE_DIR",
    "UV_LOCK_TIMEOUT",
    "PIXI_CONFIG_FILE",
    "PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR",
    "PIXI_DETACHED_ENVIRONMENTS",
    RETREAD_PIXI_CONFIG_MARKER,
    RETREAD_BASE_PIXI_CONFIG,
    RETREAD_MANAGED_KEYS,
    RETREAD_BASE_ENV_JSON,
    RETREAD_EXPECTED_ENV_JSON,
    "RETREAD_FAST_TMP_NS_JOB",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsClass {
    LocalFast,
    Network(&'static str),
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastTmpMode {
    Auto,
    On,
    Off,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobCacheMode {
    Shared,
    Tmp,
}

#[derive(Debug, Clone)]
pub struct FastTmpConfig {
    pub mode: FastTmpMode,
    pub tmp_root: PathBuf,
    pub budget_bytes: Option<u64>,
    pub blob_caches: BlobCacheMode,
    pub shared_cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Namespace {
    pub root: PathBuf,
}

impl Namespace {
    pub fn rattler_cache_dir(&self) -> PathBuf {
        self.root.join("caches").join("rattler")
    }

    pub fn uv_cache_dir(&self) -> PathBuf {
        self.root.join("caches").join("uv")
    }

    pub fn retread_cache_dir(&self) -> PathBuf {
        self.root.join("caches").join("retread")
    }

    pub fn bld_dir(&self) -> PathBuf {
        self.root.join("bld")
    }

    pub fn envs_dir(&self) -> PathBuf {
        self.root.join("envs")
    }

    pub fn pixi_config_file(&self) -> PathBuf {
        self.root.join(PIXI_FASTTMP_CONFIG)
    }
}

#[derive(Debug, Clone)]
pub struct EngagedFastTmp {
    pub env: Vec<(String, String)>,
    pub ns: Namespace,
}

#[derive(Debug, Clone, Default)]
struct BackendEnv {
    pairs: Vec<(String, String)>,
    remove_fast_vars: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendEnvOverride {
    Unchanged,
    Set(String),
    Remove,
}

static BACKEND_ENV: OnceLock<Mutex<BackendEnv>> = OnceLock::new();
static PROBE_CACHE: OnceLock<Mutex<HashMap<PathBuf, bool>>> = OnceLock::new();
static CORRUPT_COPYBACK_ONCE_USED: AtomicBool = AtomicBool::new(false);

impl Default for FastTmpConfig {
    fn default() -> Self {
        Self {
            mode: FastTmpMode::Auto,
            tmp_root: PathBuf::from(DEFAULT_TMP_ROOT),
            budget_bytes: None,
            blob_caches: BlobCacheMode::Shared,
            shared_cache_dir: None,
        }
    }
}

impl FastTmpConfig {
    pub fn load(workspace_dir: &Path) -> Self {
        let mut cfg = Self::default();
        let pixi_toml = workspace_dir.join("pixi.toml");
        if let Ok(text) = fs::read_to_string(&pixi_toml) {
            match toml::from_str::<toml::Value>(&text) {
                Ok(parsed) => cfg.apply_toml(&parsed),
                Err(e) => warn_msg(&format!(
                    "retread fast-tmp: could not parse {} for fast-tmp config: {e}",
                    pixi_toml.display()
                )),
            }
        }

        if let Ok(mode) = std::env::var("RETREAD_FAST_TMP") {
            match parse_mode(&mode) {
                Some(mode) => cfg.mode = mode,
                None => warn_msg(&format!(
                    "retread fast-tmp: ignoring invalid RETREAD_FAST_TMP={mode:?}; expected on|off|auto"
                )),
            }
        }
        if let Ok(root) = std::env::var("RETREAD_FAST_TMP_ROOT") {
            cfg.tmp_root = PathBuf::from(root);
        }
        if let Ok(budget) = std::env::var("RETREAD_FAST_TMP_BUDGET_BYTES") {
            match parse_bytes_value(&budget) {
                Some(bytes) => cfg.budget_bytes = Some(bytes),
                None => warn_msg(&format!(
                    "retread fast-tmp: ignoring invalid RETREAD_FAST_TMP_BUDGET_BYTES={budget:?}"
                )),
            }
        }
        if let Ok(shared) = std::env::var("RETREAD_SHARED_CACHE_DIR") {
            cfg.shared_cache_dir = Some(PathBuf::from(shared));
        }
        if cfg.tmp_root.is_relative() {
            cfg.tmp_root = std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(&cfg.tmp_root);
        }
        cfg
    }

    fn apply_toml(&mut self, parsed: &toml::Value) {
        let Some(table) = parsed
            .get("tool")
            .and_then(toml::Value::as_table)
            .and_then(|tool| tool.get("retread"))
            .and_then(toml::Value::as_table)
            .and_then(|retread| retread.get("fast-tmp").or_else(|| retread.get("fast_tmp")))
            .and_then(toml::Value::as_table)
        else {
            return;
        };

        if let Some(v) = table.get("mode") {
            match v.as_str().and_then(parse_mode) {
                Some(mode) => self.mode = mode,
                None => warn_msg("retread fast-tmp: ignoring invalid tool.retread.fast-tmp.mode"),
            }
        }
        if let Some(v) = table.get("tmp-root").or_else(|| table.get("tmp_root")) {
            if let Some(s) = v.as_str() {
                self.tmp_root = PathBuf::from(s);
            } else {
                warn_msg("retread fast-tmp: ignoring invalid tool.retread.fast-tmp.tmp-root");
            }
        }
        if let Some(v) = table
            .get("budget-bytes")
            .or_else(|| table.get("budget_bytes"))
        {
            if let Some(bytes) = toml_value_as_u64(v) {
                self.budget_bytes = Some(bytes);
            } else {
                warn_msg("retread fast-tmp: ignoring invalid tool.retread.fast-tmp.budget-bytes");
            }
        }
        if let Some(v) = table
            .get("blob-caches")
            .or_else(|| table.get("blob_caches"))
        {
            match v.as_str().and_then(parse_blob_mode) {
                Some(mode) => self.blob_caches = mode,
                None => {
                    warn_msg(
                        "retread fast-tmp: ignoring invalid tool.retread.fast-tmp.blob-caches",
                    );
                }
            }
        }
        if let Some(v) = table
            .get("shared-cache-dir")
            .or_else(|| table.get("shared_cache_dir"))
        {
            if let Some(s) = v.as_str() {
                self.shared_cache_dir = Some(PathBuf::from(s));
            } else {
                warn_msg(
                    "retread fast-tmp: ignoring invalid tool.retread.fast-tmp.shared-cache-dir",
                );
            }
        }
    }
}

fn parse_mode(s: &str) -> Option<FastTmpMode> {
    match s.to_ascii_lowercase().as_str() {
        "auto" => Some(FastTmpMode::Auto),
        "on" | "1" | "true" => Some(FastTmpMode::On),
        "off" | "0" | "false" => Some(FastTmpMode::Off),
        _ => None,
    }
}

fn env_flag_truthy(key: &str) -> bool {
    std::env::var(key).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on" | "y" | "t"
        )
    })
}

fn parse_blob_mode(s: &str) -> Option<BlobCacheMode> {
    match s.to_ascii_lowercase().as_str() {
        "shared" => Some(BlobCacheMode::Shared),
        "tmp" => Some(BlobCacheMode::Tmp),
        _ => None,
    }
}

fn toml_value_as_u64(v: &toml::Value) -> Option<u64> {
    match v {
        toml::Value::Integer(i) => (*i >= 0).then_some(*i as u64),
        toml::Value::String(s) => parse_bytes_value(s),
        _ => None,
    }
}

fn parse_bytes_value(s: &str) -> Option<u64> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return None;
    }
    let (digits, suffix): (String, String) = trimmed
        .chars()
        .partition(|c| c.is_ascii_digit() || *c == '.');
    let value: f64 = digits.parse().ok()?;
    let multiplier = match suffix.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "k" | "kb" | "kib" => 1024.0,
        "m" | "mb" | "mib" => 1024.0 * 1024.0,
        "g" | "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
        "t" | "tb" | "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    Some((value * multiplier) as u64)
}

pub fn classify_fs(path: &Path) -> FsClass {
    if let Ok(force) = std::env::var("RETREAD_FAST_TMP_FORCE_FS") {
        return classify_forced_fs(&force);
    }
    let Some(magic) = statfs_magic(path) else {
        return FsClass::Unknown;
    };
    classify_magic(magic)
}

fn classify_forced_fs(force: &str) -> FsClass {
    match force.to_ascii_lowercase().as_str() {
        "nfs" | "nfs4" => FsClass::Network("nfs"),
        "smb" => FsClass::Network("smb"),
        "smb2" => FsClass::Network("smb2"),
        "cifs" => FsClass::Network("cifs"),
        "ceph" => FsClass::Network("ceph"),
        "lustre" => FsClass::Network("lustre"),
        "gfs2" => FsClass::Network("gfs2"),
        "afs" | "openafs" => FsClass::Network("afs"),
        "9p" | "v9fs" => FsClass::Network("9p"),
        "ocfs2" => FsClass::Network("ocfs2"),
        "ext4" | "ext3" | "ext2" | "xfs" | "btrfs" | "tmpfs" | "overlay" => FsClass::LocalFast,
        "fuse" | "unknown" => FsClass::Unknown,
        _ => FsClass::Unknown,
    }
}

fn classify_magic(magic: u64) -> FsClass {
    match magic {
        0x6969 => FsClass::Network("nfs"),
        0x517B => FsClass::Network("smb"),
        0xFE534D42 => FsClass::Network("smb2"),
        0xFF534D42 => FsClass::Network("cifs"),
        0x73757245 => FsClass::Network("coda"),
        0x00C36400 => FsClass::Network("ceph"),
        0x0BD00BD0 => FsClass::Network("lustre"),
        0x01161970 => FsClass::Network("gfs2"),
        0x6B414653 => FsClass::Network("afs"),
        0x5346414F => FsClass::Network("openafs"),
        0x01021997 => FsClass::Network("9p"),
        0x7461636F => FsClass::Network("ocfs2"),
        0xEF53 | 0x58465342 | 0x9123683E | 0x01021994 | 0x794C7630 => FsClass::LocalFast,
        0x65735546 => FsClass::Unknown,
        _ => FsClass::Unknown,
    }
}

pub fn probe_write_latency(path: &Path) -> Option<Duration> {
    let pixi = path.join(".pixi");
    if let Err(e) = fs::create_dir_all(&pixi) {
        warn_msg(&format!(
            "retread fast-tmp: write-latency probe could not create {}: {e}",
            pixi.display()
        ));
        return None;
    }
    let probe = pixi.join(format!(".retread-fs-probe.{}", std::process::id()));
    let mut file = match OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&probe)
    {
        Ok(file) => file,
        Err(e) => {
            warn_msg(&format!(
                "retread fast-tmp: write-latency probe could not open {}: {e}",
                probe.display()
            ));
            return None;
        }
    };
    let buf = [0_u8; 4096];
    let mut samples = Vec::with_capacity(8);
    for _ in 0..8 {
        let started = Instant::now();
        if let Err(e) = file.write_all(&buf).and_then(|_| file.sync_all()) {
            warn_msg(&format!(
                "retread fast-tmp: write-latency probe failed at {}: {e}",
                probe.display()
            ));
            let _ = fs::remove_file(&probe);
            return None;
        }
        samples.push(started.elapsed());
    }
    let _ = fs::remove_file(&probe);
    samples.sort_unstable();
    samples.get(samples.len() / 2).copied()
}

pub fn is_slow(path: &Path, cfg: &FastTmpConfig) -> bool {
    match cfg.mode {
        FastTmpMode::Off => return false,
        FastTmpMode::On => return true,
        FastTmpMode::Auto => {}
    }
    match classify_fs(path) {
        FsClass::LocalFast => false,
        FsClass::Network(_) => true,
        FsClass::Unknown => {
            let key = path.to_path_buf();
            if let Some(verdict) = PROBE_CACHE
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .unwrap()
                .get(&key)
                .copied()
            {
                return verdict;
            }
            let threshold_ms = std::env::var("RETREAD_FAST_TMP_PROBE_THRESHOLD_MS")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(PROBE_THRESHOLD_MS);
            let verdict = probe_write_latency(path)
                .map(|d| d.as_secs_f64() * 1000.0 > threshold_ms)
                .unwrap_or(false);
            PROBE_CACHE
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .unwrap()
                .insert(key, verdict);
            verdict
        }
    }
}

pub fn fs_check_path(path: &Path) -> PathBuf {
    let mut current = path.to_path_buf();
    loop {
        if current.exists() {
            return current;
        }
        if !current.pop() {
            return path.to_path_buf();
        }
    }
}

pub fn namespace(cfg: &FastTmpConfig, workspace_root: &Path) -> Namespace {
    let canonical =
        fs::canonicalize(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf());
    let hash = workspace_hash(&canonical);
    let job = current_job_component();
    Namespace {
        root: cfg
            .tmp_root
            .join(user_namespace_component())
            .join(hash)
            .join(job),
    }
}

pub fn shared_blob_cache_dir(cfg: &FastTmpConfig, workspace_root: &Path) -> PathBuf {
    cfg.shared_cache_dir
        .clone()
        .unwrap_or_else(|| workspace_root.join(".retread-cache"))
}

pub fn engage(workspace_root: &Path, cfg: &FastTmpConfig) -> Result<Option<EngagedFastTmp>> {
    engage_inner(workspace_root, cfg, true)
}

pub fn engage_backend(
    workspace_root: &Path,
    cfg: &FastTmpConfig,
) -> Result<Option<EngagedFastTmp>> {
    let engaged = engage_inner(workspace_root, cfg, false)?;
    let backend = BackendEnv {
        pairs: engaged.as_ref().map(|e| e.env.clone()).unwrap_or_default(),
        remove_fast_vars: engaged.is_none() && inherited_fasttmp_cleanup_needed(),
    };
    *BACKEND_ENV
        .get_or_init(|| Mutex::new(BackendEnv::default()))
        .lock()
        .unwrap() = backend;
    Ok(engaged)
}

fn engage_inner(
    workspace_root: &Path,
    cfg: &FastTmpConfig,
    wrapper_side_effects: bool,
) -> Result<Option<EngagedFastTmp>> {
    if cfg.mode == FastTmpMode::Off {
        if wrapper_side_effects {
            cleanup_owned_workspace_links(workspace_root)?;
        }
        return Ok(None);
    }
    let stale = inherited_fasttmp_stale();
    if stale {
        warn_msg(&format!(
            "retread fast-tmp: inherited namespace for job {:?} is stale under current job {}; refreshing fast-tmp state",
            std::env::var("RETREAD_FAST_TMP_NS_JOB").ok(),
            current_job_marker()
        ));
    }
    if cfg.mode == FastTmpMode::Auto && !is_slow(workspace_root, cfg) {
        if wrapper_side_effects {
            cleanup_owned_workspace_links(workspace_root)?;
        }
        return Ok(None);
    }

    ensure_valid_managed_metadata()?;
    validate_pixi_version()?;

    let canonical = fs::canonicalize(workspace_root)
        .with_context(|| format!("canonicalizing workspace {}", workspace_root.display()))?;
    fs::create_dir_all(&cfg.tmp_root)
        .with_context(|| format!("creating tmp root {}", cfg.tmp_root.display()))?;
    enforce_tmp_user_dir(&cfg.tmp_root)?;
    enforce_budget(&cfg.tmp_root, workspace_root, cfg)?;

    let ns = namespace(cfg, &canonical);
    prepare_namespace_dirs(&ns, &canonical, cfg, workspace_root)?;
    if wrapper_side_effects {
        validate_workspace_local_detached_config(workspace_root, &ns)?;
    }
    let engaged = with_file_lock(&ns.root.join(".engage.lock"), || {
        let env = desired_env_pairs(cfg, workspace_root, &ns)?;
        if wrapper_side_effects && !in_slurm_job() {
            let pixi = workspace_root.join(".pixi");
            fs::create_dir_all(&pixi).with_context(|| format!("creating {}", pixi.display()))?;
            setup_bld_symlink(workspace_root, &ns)?;
        } else if wrapper_side_effects {
            tracing::info!(
                workspace = %workspace_root.display(),
                namespace = %ns.root.display(),
                "retread fast-tmp: SLURM job context; leaving workspace .pixi/envs and .pixi/bld untouched"
            );
        }
        if wrapper_side_effects && !in_slurm_job() {
            warn_if_real_envs_dir(workspace_root);
        }
        Ok(EngagedFastTmp {
            env,
            ns: ns.clone(),
        })
    })?;
    Ok(Some(engaged))
}

fn validate_workspace_local_detached_config(workspace_root: &Path, ns: &Namespace) -> Result<()> {
    let path = workspace_root.join(".pixi").join("config.toml");
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| format!("reading {}", path.display()));
        }
    };
    let config: toml::Table =
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let Some(detached) = config.get("detached-environments") else {
        return Ok(());
    };
    match detached {
        toml::Value::Boolean(true) => Ok(()),
        toml::Value::String(raw) => {
            let configured = if raw == "~" || raw.starts_with("~/") {
                let home = std::env::var_os("HOME").ok_or_else(|| {
                    anyhow!(
                        "{} uses ~ for detached-environments but HOME is unset",
                        path.display()
                    )
                })?;
                PathBuf::from(home).join(raw.trim_start_matches('~').trim_start_matches('/'))
            } else {
                PathBuf::from(raw)
            };
            let configured = if configured.is_absolute() {
                configured
            } else {
                // This is workspace-local `.pixi/config.toml`; Pixi resolves
                // its relative path values from that config's directory, not
                // from whichever cwd happened to invoke `--workspace`.
                path.parent().unwrap_or(workspace_root).join(configured)
            };
            let expected = fs::canonicalize(ns.envs_dir()).with_context(|| {
                format!("canonicalizing detached root {}", ns.envs_dir().display())
            })?;
            let compatible = fs::canonicalize(&configured)
                .ok()
                .is_some_and(|resolved| resolved == expected);
            if compatible {
                Ok(())
            } else {
                bail!(
                    "{} sets detached-environments to {}, which overrides retread fast-tmp's job-local root {}. Remove the local setting or set it to true.",
                    path.display(),
                    configured.display(),
                    expected.display()
                )
            }
        }
        _ => bail!(
            "{} has incompatible detached-environments={detached}; remove the local setting or set it to true so retread fast-tmp can select the job-local root",
            path.display()
        ),
    }
}

fn prepare_namespace_dirs(
    ns: &Namespace,
    canonical_workspace: &Path,
    cfg: &FastTmpConfig,
    workspace_root: &Path,
) -> Result<()> {
    for dir in [
        ns.rattler_cache_dir(),
        ns.uv_cache_dir(),
        ns.retread_cache_dir(),
        ns.bld_dir(),
        ns.envs_dir(),
    ] {
        fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    set_dir_mode_0700(&ns.root);
    write_namespace_ownership_proof(ns, canonical_workspace)?;
    if cfg.blob_caches == BlobCacheMode::Shared {
        let shared = shared_blob_cache_dir(cfg, workspace_root);
        fs::create_dir_all(&shared)
            .with_context(|| format!("creating shared retread blob cache {}", shared.display()))?;
    }
    prepare_pixi_config_overlay(ns)?;
    Ok(())
}

fn write_namespace_ownership_proof(ns: &Namespace, canonical_workspace: &Path) -> Result<()> {
    fs::create_dir_all(ns.bld_dir())
        .with_context(|| format!("creating {}", ns.bld_dir().display()))?;
    write_bld_owner_marker(ns, canonical_workspace)?;
    fs::write(
        ns.root.join("workspace-path"),
        canonical_workspace.to_string_lossy().as_bytes(),
    )
    .with_context(|| {
        format!(
            "writing fast-tmp workspace breadcrumb {}",
            ns.root.join("workspace-path").display()
        )
    })?;
    Ok(())
}

fn bld_owner_marker_contents(canonical_workspace: &Path) -> Vec<u8> {
    format!(
        "{RETREAD_BLD_OWNER_FORMAT}\nworkspace={}\n",
        canonical_workspace.display()
    )
    .into_bytes()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OwnershipMarkerStatus {
    Absent,
    Valid,
    Invalid,
}

fn ownership_marker_status(
    marker: &Path,
    expected: &[u8],
    description: &str,
) -> Result<OwnershipMarkerStatus> {
    match fs::symlink_metadata(marker) {
        Ok(metadata) if metadata.file_type().is_file() && metadata.len() <= 16 * 1024 => {}
        Ok(_) => return Ok(OwnershipMarkerStatus::Invalid),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(OwnershipMarkerStatus::Absent);
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("checking {description} {}", marker.display()));
        }
    }
    let contents =
        fs::read(marker).with_context(|| format!("reading {description} {}", marker.display()))?;
    Ok(if contents == expected {
        OwnershipMarkerStatus::Valid
    } else {
        OwnershipMarkerStatus::Invalid
    })
}

fn write_bld_owner_marker(ns: &Namespace, canonical_workspace: &Path) -> Result<()> {
    let marker = ns.bld_dir().join(RETREAD_BLD_OWNER_MARKER);
    let expected = bld_owner_marker_contents(canonical_workspace);
    match fs::symlink_metadata(&marker) {
        Ok(metadata) if !metadata.file_type().is_file() => {
            bail!(
                "retread fast-tmp ownership marker {} is not a regular file",
                marker.display()
            );
        }
        Ok(_) => {
            if fs::read(&marker)
                .with_context(|| format!("reading bld ownership marker {}", marker.display()))?
                == expected
            {
                return Ok(());
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("checking bld ownership marker {}", marker.display()));
        }
    }
    fs::write(&marker, expected)
        .with_context(|| format!("writing bld ownership marker {}", marker.display()))
}

/// Return the Pixi config file that was active before retread installed its
/// overlay. A sourced `fast --print-env` can outlive a SLURM job; in that case
/// `PIXI_CONFIG_FILE` still names our old overlay, while the base marker keeps
/// the user's original override (if any).
fn base_pixi_config_file() -> Option<OsString> {
    let current = std::env::var_os("PIXI_CONFIG_FILE");
    let marker = std::env::var_os(RETREAD_PIXI_CONFIG_MARKER);
    let base = if current.is_some() && current == marker && current_managed_env_state().is_some() {
        std::env::var_os(RETREAD_BASE_PIXI_CONFIG).filter(|value| !value.is_empty())
    } else {
        current
    }?;
    let path = PathBuf::from(base);
    if path.is_absolute() {
        Some(path.into_os_string())
    } else {
        // The overlay query runs from the job namespace, not the caller's
        // cwd. Resolve first so a relative user PIXI_CONFIG_FILE keeps the
        // exact meaning it had when `fast` was invoked.
        Some(
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(path)
                .into_os_string(),
        )
    }
}

/// Build a private, job-local Pixi config overlay from Pixi's *effective*
/// configuration. `PIXI_CONFIG_FILE` replaces Pixi's normal system/user
/// config search, so a minimal file would silently discard authentication,
/// mirrors, TLS policy, proxy settings, and `run-post-link-scripts`. Asking
/// Pixi to render the base (system + user/override) config first preserves
/// those settings. The query deliberately runs outside the workspace: Pixi
/// will merge `.pixi/config.toml` on top of this overlay later, exactly once.
fn prepare_pixi_config_overlay(ns: &Namespace) -> Result<()> {
    if env_flag_truthy("PIXI_NO_CONFIG") {
        bail!(
            "retread fast-tmp cannot enable supported Pixi detached environments while PIXI_NO_CONFIG is set; unset PIXI_NO_CONFIG or turn fast-tmp off"
        );
    }

    let rendered = if let Some(base_path) = base_pixi_config_file() {
        // Pixi 0.70.2's install/info commands honor PIXI_CONFIG_FILE, but its
        // `config list` subcommand does not. Because an explicit override
        // replaces the system/user search layer, parsing that one file is the
        // exact base layer we need to preserve.
        let base_path = PathBuf::from(base_path);
        let base = fs::read_to_string(&base_path)
            .with_context(|| format!("reading base Pixi config {}", base_path.display()))?;
        render_pixi_config_overlay_from_toml(&base)
            .with_context(|| format!("parsing base Pixi config {}", base_path.display()))?
    } else {
        let mut command = Command::new("pixi");
        command
            .args(["config", "list", "--json"])
            .current_dir(&ns.root)
            .stdin(Stdio::null())
            .env_remove("PIXI_CONFIG_FILE")
            .env_remove("PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR")
            .env_remove("PIXI_DETACHED_ENVIRONMENTS")
            .env_remove(RETREAD_PIXI_CONFIG_MARKER)
            .env_remove(RETREAD_BASE_PIXI_CONFIG);
        let effective_json = match command.output() {
            Ok(output) if output.status.success() => output.stdout,
            Ok(output) => {
                bail!(
                    "`pixi config list --json` failed while preparing fast-tmp config ({}): {}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr).trim()
                );
            }
            #[cfg(test)]
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => b"{}".to_vec(),
            Err(error) => {
                return Err(error).context("reading effective Pixi config for fast-tmp overlay");
            }
        };
        render_pixi_config_overlay(&effective_json)?
    };
    atomic_write_private(&ns.pixi_config_file(), rendered.as_bytes())
        .context("writing job-local Pixi fast-tmp config overlay")
}

fn render_pixi_config_overlay(effective_json: &[u8]) -> Result<String> {
    let config: toml::Table = serde_json::from_slice(effective_json)
        .context("parsing effective Pixi config JSON for fast-tmp overlay")?;
    render_pixi_config_table(config)
}

fn render_pixi_config_overlay_from_toml(effective_toml: &str) -> Result<String> {
    let config: toml::Table =
        toml::from_str(effective_toml).context("parsing effective Pixi config TOML")?;
    render_pixi_config_table(config)
}

fn render_pixi_config_table(mut config: toml::Table) -> Result<String> {
    // Pixi 0.70 requires this config switch. The supported cache env below
    // supplies the job-local path; retaining a boolean here also avoids
    // baking a redundant path into the effective user configuration.
    config.insert(
        "detached-environments".to_string(),
        toml::Value::Boolean(true),
    );
    toml::to_string_pretty(&config)
        .context("serializing effective Pixi config for fast-tmp overlay")
}

fn atomic_write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("config overlay path {} has no parent", path.display()))?;
    let tmp = parent.join(format!(
        ".{PIXI_FASTTMP_CONFIG}.{}.{}",
        std::process::id(),
        unique_nonce()
    ));
    let result = (|| -> Result<()> {
        fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
        set_file_mode_0600(&tmp);
        fs::rename(&tmp, path).with_context(|| {
            format!(
                "atomically replacing {} with {}",
                path.display(),
                tmp.display()
            )
        })?;
        let _ = fsync_dir(parent);
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

#[cfg(unix)]
fn set_file_mode_0600(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn set_file_mode_0600(_path: &Path) {}

#[derive(Debug, Clone)]
struct ManagedEnvState {
    managed: HashSet<String>,
    base: HashMap<String, Option<String>>,
    expected: HashMap<String, String>,
}

fn current_managed_env_state() -> Option<ManagedEnvState> {
    let managed_raw = std::env::var(RETREAD_MANAGED_KEYS).ok()?;
    let base_raw = std::env::var(RETREAD_BASE_ENV_JSON).ok()?;
    let expected_raw = std::env::var(RETREAD_EXPECTED_ENV_JSON).ok()?;
    let base: HashMap<String, Option<String>> = serde_json::from_str(&base_raw).ok()?;
    let expected: HashMap<String, String> = serde_json::from_str(&expected_raw).ok()?;
    let managed = managed_raw
        .split(',')
        .filter(|key| MANAGEABLE_ENV_KEYS.contains(key))
        .map(str::to_owned)
        .collect::<HashSet<_>>();
    if managed
        .iter()
        .any(|key| !base.contains_key(key) || !expected.contains_key(key))
    {
        return None;
    }
    for (key, current) in [
        (RETREAD_MANAGED_KEYS, managed_raw.as_str()),
        (RETREAD_BASE_ENV_JSON, base_raw.as_str()),
    ] {
        if expected.get(key).map(String::as_str) != Some(current) {
            return None;
        }
    }
    for key in [
        RETREAD_PIXI_CONFIG_MARKER,
        RETREAD_BASE_PIXI_CONFIG,
        "RETREAD_FAST_TMP_NS_JOB",
    ] {
        let current = std::env::var(key).ok()?;
        if expected.get(key) != Some(&current) {
            return None;
        }
    }
    Some(ManagedEnvState {
        managed,
        base,
        expected,
    })
}

fn has_new_managed_metadata() -> bool {
    [
        RETREAD_MANAGED_KEYS,
        RETREAD_BASE_ENV_JSON,
        RETREAD_EXPECTED_ENV_JSON,
        RETREAD_PIXI_CONFIG_MARKER,
        RETREAD_BASE_PIXI_CONFIG,
    ]
    .iter()
    .any(|key| std::env::var_os(key).is_some())
}

fn ensure_valid_managed_metadata() -> Result<()> {
    if has_new_managed_metadata() && current_managed_env_state().is_none() {
        bail!(
            "retread fast-tmp: inherited environment ownership metadata is missing, malformed, or changed; refusing to re-engage and overwrite inherited Pixi values"
        );
    }
    Ok(())
}

fn desired_env_pairs(
    cfg: &FastTmpConfig,
    workspace_root: &Path,
    ns: &Namespace,
) -> Result<Vec<(String, String)>> {
    ensure_valid_managed_metadata()?;
    let blob_cache = match cfg.blob_caches {
        BlobCacheMode::Shared => shared_blob_cache_dir(cfg, workspace_root),
        BlobCacheMode::Tmp => ns.rattler_cache_dir(),
    };
    let previous = current_managed_env_state();
    let mut out = Vec::new();
    let mut managed = Vec::new();
    let mut base = HashMap::new();
    let mut expected = HashMap::new();
    push_managed_env(
        &mut out,
        &mut managed,
        &mut base,
        &mut expected,
        previous.as_ref(),
        "PIXI_CACHE_DIR",
        blob_cache.to_string_lossy().into_owned(),
        false,
    );
    push_managed_env(
        &mut out,
        &mut managed,
        &mut base,
        &mut expected,
        previous.as_ref(),
        "RATTLER_CACHE_DIR",
        blob_cache.to_string_lossy().into_owned(),
        false,
    );
    push_managed_env(
        &mut out,
        &mut managed,
        &mut base,
        &mut expected,
        previous.as_ref(),
        "UV_CACHE_DIR",
        ns.uv_cache_dir().to_string_lossy().into_owned(),
        false,
    );
    // NOTE: this redirect deliberately does NOT move the loose-bundle wheel
    // store. The store is resolved by courier::retread_wheel_store_root(),
    // which ignores RETREAD_CACHE_DIR precisely so blob stores stay SHARED
    // while scratch caches go job-local (a job-local store dies with the
    // job, breaking `retread install` on other nodes / later jobs).
    push_managed_env(
        &mut out,
        &mut managed,
        &mut base,
        &mut expected,
        previous.as_ref(),
        "RETREAD_CACHE_DIR",
        ns.retread_cache_dir().to_string_lossy().into_owned(),
        false,
    );
    push_managed_env(
        &mut out,
        &mut managed,
        &mut base,
        &mut expected,
        previous.as_ref(),
        "UV_LOCK_TIMEOUT",
        "1800".to_string(),
        false,
    );
    // Pixi 0.70+ requires detached-environments in config and exposes the
    // destination through PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR. Retain the
    // old variable only for shell-level backward compatibility; correctness
    // is version-gated on the supported 0.70+ interface below.
    push_managed_env(
        &mut out,
        &mut managed,
        &mut base,
        &mut expected,
        previous.as_ref(),
        "PIXI_CONFIG_FILE",
        ns.pixi_config_file().to_string_lossy().into_owned(),
        true,
    );
    push_managed_env(
        &mut out,
        &mut managed,
        &mut base,
        &mut expected,
        previous.as_ref(),
        "PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR",
        ns.envs_dir().to_string_lossy().into_owned(),
        true,
    );
    push_managed_env(
        &mut out,
        &mut managed,
        &mut base,
        &mut expected,
        previous.as_ref(),
        "PIXI_DETACHED_ENVIRONMENTS",
        ns.envs_dir().to_string_lossy().into_owned(),
        true,
    );
    managed.sort();
    managed.dedup();
    let config_marker = ns.pixi_config_file().to_string_lossy().into_owned();
    let base_config_marker = base
        .get("PIXI_CONFIG_FILE")
        .cloned()
        .flatten()
        .unwrap_or_default();
    let managed_marker = managed.join(",");
    let base_json = serde_json::to_string(&base).expect("serializing fast-tmp base environment");
    let job_marker = current_job_marker();
    for (key, value) in [
        (RETREAD_PIXI_CONFIG_MARKER, config_marker.as_str()),
        (RETREAD_BASE_PIXI_CONFIG, base_config_marker.as_str()),
        (RETREAD_MANAGED_KEYS, managed_marker.as_str()),
        (RETREAD_BASE_ENV_JSON, base_json.as_str()),
        ("RETREAD_FAST_TMP_NS_JOB", job_marker.as_str()),
    ] {
        expected.insert(key.to_string(), value.to_string());
        out.push((key.to_string(), value.to_string()));
    }
    out.push((
        RETREAD_EXPECTED_ENV_JSON.to_string(),
        serde_json::to_string(&expected).expect("serializing fast-tmp expected environment"),
    ));
    Ok(out)
}

fn push_managed_env(
    out: &mut Vec<(String, String)>,
    managed: &mut Vec<String>,
    base: &mut HashMap<String, Option<String>>,
    expected: &mut HashMap<String, String>,
    previous: Option<&ManagedEnvState>,
    key: &str,
    value: String,
    force: bool,
) {
    let current = std::env::var_os(key).map(|value| value.to_string_lossy().into_owned());
    let previous_claimed = previous.is_some_and(|state| state.managed.contains(key));
    let previous_owned = previous_claimed
        && previous
            .and_then(|state| state.expected.get(key))
            .is_some_and(|expected| current.as_deref() == Some(expected.as_str()));
    if !force && previous_claimed && !previous_owned {
        return;
    }
    let legacy_owned = previous.is_none()
        && std::env::var_os("RETREAD_FAST_TMP_NS_JOB").is_some()
        && current.as_deref().is_some_and(|current| {
            looks_like_retread_namespace_value(current)
                || (key == "UV_LOCK_TIMEOUT" && current == "1800")
        });
    if force || previous_owned || legacy_owned || current.is_none() {
        let original = if previous_owned {
            previous
                .and_then(|state| state.base.get(key))
                .cloned()
                .flatten()
        } else if legacy_owned {
            None
        } else if key == "PIXI_CONFIG_FILE" {
            current.map(|current| {
                let path = PathBuf::from(&current);
                if path.is_absolute() {
                    current
                } else {
                    std::env::current_dir()
                        .unwrap_or_else(|_| PathBuf::from("."))
                        .join(path)
                        .to_string_lossy()
                        .into_owned()
                }
            })
        } else {
            current
        };
        base.insert(key.to_string(), original);
        expected.insert(key.to_string(), value.clone());
        managed.push(key.to_string());
        out.push((key.to_string(), value));
    }
}

fn validate_pixi_version() -> Result<()> {
    let output = Command::new("pixi")
        .arg("--version")
        .stdin(Stdio::null())
        .output();
    let stdout = match output {
        Ok(output) if output.status.success() => output.stdout,
        Ok(output) => {
            bail!(
                "`pixi --version` failed while validating fast-tmp compatibility ({}): {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )
        }
        #[cfg(test)]
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => b"pixi 0.70.2\n".to_vec(),
        Err(error) => return Err(error).context("running `pixi --version` for fast-tmp"),
    };
    validate_pixi_version_text(&String::from_utf8_lossy(&stdout))
}

fn validate_pixi_version_text(output: &str) -> Result<()> {
    let version = output
        .split_whitespace()
        .find(|part| part.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
        .ok_or_else(|| anyhow!("could not parse Pixi version from {output:?}"))?;
    let mut parts = version.split('.');
    let major = parts.next().and_then(|part| part.parse::<u64>().ok());
    let minor = parts.next().and_then(|part| part.parse::<u64>().ok());
    if !matches!((major, minor), (Some(major), Some(minor)) if major > 0 || minor >= 70) {
        bail!(
            "retread fast-tmp requires Pixi >=0.70 for supported detached-environments config; found {version}"
        );
    }
    Ok(())
}

/// Return true only for an absolute path whose lexical representation has no
/// `.` or `..` components. This deliberately does not require the path to
/// exist, so dangling legacy Retread links can still be identified safely.
fn is_normalized_absolute_path(path: &Path) -> bool {
    if !path.is_absolute() {
        return false;
    }
    let mut rebuilt = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => rebuilt.push(prefix.as_os_str()),
            Component::RootDir => rebuilt.push(component.as_os_str()),
            Component::Normal(value) => rebuilt.push(value),
            Component::CurDir | Component::ParentDir => return false,
        }
    }
    rebuilt.as_os_str() == path.as_os_str()
}

pub fn backend_env_override(key: &str) -> BackendEnvOverride {
    let Some(env) = BACKEND_ENV.get() else {
        return BackendEnvOverride::Unchanged;
    };
    let env = env.lock().unwrap().clone();
    if let Some((_, value)) = env.pairs.iter().find(|(candidate, _)| candidate == key) {
        return BackendEnvOverride::Set(value.clone());
    }
    if env.remove_fast_vars
        && let Some((_, value)) = cleanup_env_actions()
            .into_iter()
            .find(|(candidate, _)| candidate == key)
    {
        return match value {
            Some(value) => BackendEnvOverride::Set(value),
            None => BackendEnvOverride::Remove,
        };
    }
    BackendEnvOverride::Unchanged
}

pub fn apply_backend_env(cmd: &mut tokio::process::Command) {
    let env = BACKEND_ENV
        .get_or_init(|| Mutex::new(BackendEnv::default()))
        .lock()
        .unwrap()
        .clone();
    if env.remove_fast_vars {
        for (key, value) in cleanup_env_actions() {
            match value {
                Some(value) => {
                    cmd.env(key, value);
                }
                None => {
                    cmd.env_remove(key);
                }
            }
        }
    }
    for (key, value) in env.pairs {
        cmd.env(key, value);
    }
}

pub fn inherited_fasttmp_stale() -> bool {
    std::env::var("RETREAD_FAST_TMP_NS_JOB")
        .ok()
        .is_some_and(|seen| seen != current_job_marker())
}

pub fn inherited_fasttmp_cleanup_needed() -> bool {
    has_new_managed_metadata() || std::env::var_os("RETREAD_FAST_TMP_NS_JOB").is_some()
}

fn cleanup_env_actions() -> Vec<(String, Option<String>)> {
    let mut actions: HashMap<String, Option<String>> = HashMap::new();
    if has_new_managed_metadata() {
        let Some(state) = current_managed_env_state() else {
            warn_msg(
                "retread fast-tmp: inherited environment ownership metadata is missing, malformed, or changed; preserving all environment values",
            );
            return Vec::new();
        };
        for key in &state.managed {
            let Some(expected) = state.expected.get(key) else {
                continue;
            };
            if std::env::var(key).ok().as_deref() == Some(expected.as_str()) {
                actions.insert(key.clone(), state.base.get(key).cloned().unwrap_or(None));
            }
        }
        for key in [
            RETREAD_PIXI_CONFIG_MARKER,
            RETREAD_BASE_PIXI_CONFIG,
            RETREAD_MANAGED_KEYS,
            RETREAD_BASE_ENV_JSON,
            RETREAD_EXPECTED_ENV_JSON,
            "RETREAD_FAST_TMP_NS_JOB",
        ] {
            actions.insert(key.to_string(), None);
        }
    } else if std::env::var_os("RETREAD_FAST_TMP_NS_JOB").is_some() {
        // Committed pre-marker releases only set variables that were absent.
        // Remove values whose current shape still proves they are Retread's;
        // shared/user cache paths and any values changed later are preserved.
        for key in [
            "PIXI_CACHE_DIR",
            "RATTLER_CACHE_DIR",
            "PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR",
            "PIXI_DETACHED_ENVIRONMENTS",
            "UV_CACHE_DIR",
            "RETREAD_CACHE_DIR",
        ] {
            if std::env::var(key)
                .ok()
                .as_deref()
                .is_some_and(looks_like_retread_namespace_value)
            {
                actions.insert(key.to_string(), None);
            }
        }
        if std::env::var("UV_LOCK_TIMEOUT").ok().as_deref() == Some("1800") {
            actions.insert("UV_LOCK_TIMEOUT".to_string(), None);
        }
        actions.insert("RETREAD_FAST_TMP_NS_JOB".to_string(), None);
    }
    let mut actions = actions.into_iter().collect::<Vec<_>>();
    actions.sort_by(|a, b| a.0.cmp(&b.0));
    actions
}

fn looks_like_retread_namespace_value(value: &str) -> bool {
    let path = Path::new(value);
    if !is_normalized_absolute_path(path) {
        return false;
    }
    let parts = path
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str(),
            _ => None,
        })
        .collect::<Vec<_>>();
    parts.windows(4).any(|parts| {
        parts[0].starts_with("retread-")
            && parts[1].len() == 12
            && parts[1].bytes().all(|byte| byte.is_ascii_hexdigit())
            && (parts[2] == "nojob"
                || parts[2]
                    .strip_prefix("job-")
                    .is_some_and(|job| !job.is_empty()))
    })
}

pub fn current_job_marker() -> String {
    std::env::var("SLURM_JOB_ID")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "nojob".to_string())
}

pub fn in_slurm_job() -> bool {
    std::env::var_os("SLURM_JOB_ID").is_some()
}

fn current_job_component() -> String {
    match std::env::var("SLURM_JOB_ID") {
        Ok(id) if !id.is_empty() => format!("job-{id}"),
        _ => "nojob".to_string(),
    }
}

fn workspace_hash(path: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    let digest = hasher.finalize();
    digest.iter().take(6).map(|b| format!("{b:02x}")).collect()
}

fn user_namespace_component() -> String {
    if let Some(user) = std::env::var_os("USER").and_then(|u| {
        let s = u.to_string_lossy().trim().to_string();
        (!s.is_empty()).then_some(s)
    }) {
        let sanitized = user
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>();
        format!("retread-{sanitized}")
    } else {
        format!("retread-uid{}", current_uid())
    }
}

fn enforce_tmp_user_dir(tmp_root: &Path) -> Result<()> {
    let user_dir = tmp_root.join(user_namespace_component());
    match fs::symlink_metadata(&user_dir) {
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                bail!(
                    "retread fast-tmp refuses to use symlinked namespace dir {}",
                    user_dir.display()
                );
            }
            if !meta.is_dir() {
                bail!(
                    "retread fast-tmp namespace path exists but is not a directory: {}",
                    user_dir.display()
                );
            }
            if metadata_uid(&meta) != current_uid() {
                bail!(
                    "retread fast-tmp namespace dir {} is owned by uid {}, not current uid {}",
                    user_dir.display(),
                    metadata_uid(&meta),
                    current_uid()
                );
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(&user_dir)
                .with_context(|| format!("creating fast-tmp user dir {}", user_dir.display()))?;
            set_dir_mode_0700(&user_dir);
        }
        Err(e) => {
            return Err(e)
                .with_context(|| format!("checking fast-tmp user dir {}", user_dir.display()));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OwnedWorkspaceNamespaceTarget {
    namespace_root: PathBuf,
    job_component: String,
    relative: PathBuf,
}

fn owned_workspace_namespace_target(
    workspace_root: &Path,
    raw_target: &Path,
    matches_relative: impl Fn(&Path) -> bool,
) -> Result<Option<OwnedWorkspaceNamespaceTarget>> {
    if !is_normalized_absolute_path(raw_target) {
        return Ok(None);
    }
    let canonical_workspace = fs::canonicalize(workspace_root)
        .with_context(|| format!("canonicalizing workspace {}", workspace_root.display()))?;
    for candidate_root in raw_target.ancestors().skip(1) {
        let Ok(relative) = raw_target.strip_prefix(candidate_root) else {
            continue;
        };
        if !matches_relative(relative) {
            continue;
        }
        let Some(job) = candidate_root.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if job != "nojob" && !job.strip_prefix("job-").is_some_and(|id| !id.is_empty()) {
            continue;
        }

        // Strongest proof: a live namespace breadcrumb identifies the exact
        // canonical workspace even across tmp-root/user naming changes.
        if fs::read_to_string(candidate_root.join("workspace-path"))
            .ok()
            .is_some_and(|recorded| Path::new(&recorded) == canonical_workspace)
        {
            return Ok(Some(OwnedWorkspaceNamespaceTarget {
                namespace_root: candidate_root.to_path_buf(),
                job_component: job.to_string(),
                relative: relative.to_path_buf(),
            }));
        }

        // Strict fallback for an evicted/dangling legacy namespace: reproduce
        // the exact user + canonical-workspace hash portion of Retread's path.
        let hash_matches = candidate_root
            .parent()
            .and_then(Path::file_name)
            .and_then(|value| value.to_str())
            .is_some_and(|hash| hash == workspace_hash(&canonical_workspace));
        let user_matches = candidate_root
            .parent()
            .and_then(Path::parent)
            .and_then(Path::file_name)
            .and_then(|value| value.to_str())
            .is_some_and(|user| user == user_namespace_component());
        if hash_matches && user_matches {
            return Ok(Some(OwnedWorkspaceNamespaceTarget {
                namespace_root: candidate_root.to_path_buf(),
                job_component: job.to_string(),
                relative: relative.to_path_buf(),
            }));
        }
    }
    Ok(None)
}

fn workspace_link_target_is_owned(workspace_root: &Path, raw_target: &Path) -> Result<bool> {
    Ok(
        owned_workspace_namespace_target(workspace_root, raw_target, |relative| {
            relative == Path::new("bld")
        })?
        .is_some(),
    )
}

fn pixi_envs_workspace_component(
    workspace_root: &Path,
    canonical_workspace: &Path,
) -> Result<String> {
    let manifest_path = workspace_root.join("pixi.toml");
    let manifest = fs::read_to_string(&manifest_path).with_context(|| {
        format!(
            "reading Pixi workspace manifest {}",
            manifest_path.display()
        )
    })?;
    let parsed: toml::Value = toml::from_str(&manifest).with_context(|| {
        format!(
            "parsing Pixi workspace manifest {}",
            manifest_path.display()
        )
    })?;
    let display_name = parsed
        .get("workspace")
        .and_then(|workspace| workspace.get("name"))
        .and_then(toml::Value::as_str)
        .or_else(|| {
            parsed
                .get("project")
                .and_then(|project| project.get("name"))
                .and_then(toml::Value::as_str)
        })
        .or_else(|| {
            parsed
                .get("package")
                .and_then(|package| package.get("name"))
                .and_then(toml::Value::as_str)
        })
        .map(str::to_string)
        .or_else(|| {
            canonical_workspace
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "workspace".to_string());
    Ok(format!(
        "{display_name}-{}",
        xxh3_64(canonical_workspace.to_string_lossy().as_bytes())
    ))
}

fn is_pixi_envs_namespace_relative(relative: &Path, workspace_component: &str) -> bool {
    let mut components = relative.iter();
    components.next().is_some_and(|value| value == "envs")
        && components
            .next()
            .is_some_and(|value| value == workspace_component)
        && components.next().is_some_and(|value| value == "envs")
        && components.next().is_none()
}

fn envs_target_components_are_real_directories(
    configured_tmp_root: &Path,
    raw_target: &Path,
) -> Result<bool> {
    let Ok(relative) = raw_target.strip_prefix(configured_tmp_root) else {
        return Ok(false);
    };
    let mut current = configured_tmp_root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(value) = component else {
            return Ok(false);
        };
        current.push(value);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return Ok(false);
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(true),
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "checking fast-tmp namespace component {}",
                        current.display()
                    )
                });
            }
        }
    }
    Ok(true)
}

fn envs_namespace_marker_status(
    workspace_root: &Path,
    namespace_root: &Path,
) -> Result<OwnershipMarkerStatus> {
    let canonical_workspace = fs::canonicalize(workspace_root)
        .with_context(|| format!("canonicalizing workspace {}", workspace_root.display()))?;
    let workspace_path = canonical_workspace.to_string_lossy();
    let breadcrumb = ownership_marker_status(
        &namespace_root.join("workspace-path"),
        workspace_path.as_bytes(),
        "fast-tmp workspace breadcrumb",
    )?;
    let bld_marker = ownership_marker_status(
        &namespace_root.join("bld").join(RETREAD_BLD_OWNER_MARKER),
        &bld_owner_marker_contents(&canonical_workspace),
        "bld ownership marker",
    )?;
    Ok(
        if matches!(breadcrumb, OwnershipMarkerStatus::Invalid)
            || matches!(bld_marker, OwnershipMarkerStatus::Invalid)
        {
            OwnershipMarkerStatus::Invalid
        } else if matches!(breadcrumb, OwnershipMarkerStatus::Valid)
            || matches!(bld_marker, OwnershipMarkerStatus::Valid)
        {
            OwnershipMarkerStatus::Valid
        } else {
            OwnershipMarkerStatus::Absent
        },
    )
}

fn workspace_envs_link_target(
    workspace_root: &Path,
    configured_tmp_root: &Path,
    raw_target: &Path,
    allow_dangling_shape_proof: bool,
) -> Result<Option<OwnedWorkspaceNamespaceTarget>> {
    let canonical_workspace = fs::canonicalize(workspace_root)
        .with_context(|| format!("canonicalizing workspace {}", workspace_root.display()))?;
    let pixi_workspace_component =
        pixi_envs_workspace_component(workspace_root, &canonical_workspace)?;
    let Some(owned) = owned_workspace_namespace_target(workspace_root, raw_target, |relative| {
        is_pixi_envs_namespace_relative(relative, &pixi_workspace_component)
    })?
    else {
        return Ok(None);
    };
    let workspace_component_matches = owned
        .namespace_root
        .parent()
        .and_then(Path::file_name)
        .and_then(|value| value.to_str())
        .is_some_and(|value| value == workspace_hash(&canonical_workspace));
    let user_component_matches = owned
        .namespace_root
        .parent()
        .and_then(Path::parent)
        .and_then(Path::file_name)
        .and_then(|value| value.to_str())
        .is_some_and(|value| value == user_namespace_component());
    if !workspace_component_matches || !user_component_matches {
        return Ok(None);
    }

    let expected_namespace_root = configured_tmp_root
        .join(user_namespace_component())
        .join(workspace_hash(&canonical_workspace))
        .join(&owned.job_component);
    if owned.namespace_root != expected_namespace_root
        || !envs_target_components_are_real_directories(configured_tmp_root, raw_target)?
    {
        return Ok(None);
    }
    match envs_namespace_marker_status(workspace_root, &owned.namespace_root)? {
        OwnershipMarkerStatus::Valid => Ok(Some(owned)),
        OwnershipMarkerStatus::Absent if allow_dangling_shape_proof => Ok(Some(owned)),
        OwnershipMarkerStatus::Absent | OwnershipMarkerStatus::Invalid => Ok(None),
    }
}

/// Atomically move `source` to an absent `destination`. Linux's
/// `RENAME_NOREPLACE` is the only primitive used for workspace bld mutation:
/// it works for symlinks, files, and directories without ever replacing a
/// concurrently installed user path.
#[cfg(target_os = "linux")]
fn rename_with_flags(
    source: &Path,
    destination: &Path,
    flags: rustix::fs::RenameFlags,
) -> std::io::Result<()> {
    use rustix::fs::renameat_with;

    let source_parent = source.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} has no parent", source.display()),
        )
    })?;
    let destination_parent = destination.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} has no parent", destination.display()),
        )
    })?;
    if source_parent != destination_parent {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "safe rename requires one lexical parent: {} != {}",
                source_parent.display(),
                destination_parent.display()
            ),
        ));
    }
    let source_name = source.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} has no file name", source.display()),
        )
    })?;
    let destination_name = destination.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} has no file name", destination.display()),
        )
    })?;
    // Open once: even if `.pixi` itself is concurrently renamed/replaced, both
    // operands remain anchored to the exact same directory inode.
    let directory = File::open(source_parent)?;
    renameat_with(&directory, source_name, &directory, destination_name, flags)
        .map_err(std::io::Error::from)
}

#[cfg(target_os = "linux")]
fn rename_noreplace(source: &Path, destination: &Path) -> std::io::Result<()> {
    rename_with_flags(source, destination, rustix::fs::RenameFlags::NOREPLACE)
}

#[cfg(not(target_os = "linux"))]
fn rename_noreplace(_source: &Path, _destination: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "safe workspace bld mutation requires Linux renameat2(RENAME_NOREPLACE)",
    ))
}

fn restore_quarantined_path(quarantine: &Path, link: &Path) -> Result<()> {
    match rename_noreplace(quarantine, link) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => bail!(
            "retread fast-tmp: {} changed concurrently; preserved the displaced path at {} without replacing the newer workspace path",
            link.display(),
            quarantine.display()
        ),
        Err(error) => Err(error).with_context(|| {
            format!(
                "restoring concurrently displaced path {} from {}; the path remains quarantined",
                link.display(),
                quarantine.display()
            )
        }),
    }
}

/// On non-SLURM wrapper disengage, remove only a `.pixi/bld` symlink proven to
/// be ours. SLURM never mutates either workspace link.
fn cleanup_owned_workspace_links(workspace_root: &Path) -> Result<()> {
    if in_slurm_job() {
        return Ok(());
    }
    let pixi = workspace_root.join(".pixi");
    match fs::metadata(&pixi) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => return Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error).with_context(|| format!("checking {}", pixi.display())),
    }
    with_file_lock(&pixi.join(FAST_WORKSPACE_LINK_LOCK), || {
        remove_owned_workspace_symlink(workspace_root, &pixi.join("bld"))
    })
}

fn remove_owned_workspace_symlink(workspace_root: &Path, link: &Path) -> Result<()> {
    remove_owned_workspace_symlink_with_hook(workspace_root, link, || {})
}

fn remove_owned_workspace_symlink_with_hook(
    workspace_root: &Path,
    link: &Path,
    before_quarantine: impl FnOnce(),
) -> Result<()> {
    let Ok(metadata) = fs::symlink_metadata(link) else {
        return Ok(());
    };
    if !metadata.file_type().is_symlink() {
        return Ok(());
    }
    let initial = fs::read_link(link)
        .with_context(|| format!("reading existing symlink {}", link.display()))?;
    if !workspace_link_target_is_owned(workspace_root, &initial)? {
        return Ok(());
    }

    // Ownership must still hold at the actual mutation point. Move the path
    // aside atomically, inspect exactly what moved, and delete only the same
    // Retread-owned symlink we observed. If a user raced us, restore their
    // inode only when the original name is still free; never replace a newer
    // path they installed there.
    before_quarantine();
    let parent = link
        .parent()
        .ok_or_else(|| anyhow!("symlink path {} has no parent", link.display()))?;
    let quarantine = parent.join(format!(
        ".{}.retread-quarantine.{}.{}",
        link.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("link"),
        std::process::id(),
        unique_nonce()
    ));
    match rename_noreplace(link, &quarantine) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "quarantining candidate Retread symlink {} as {}",
                    link.display(),
                    quarantine.display()
                )
            });
        }
    }

    let moved_target = fs::symlink_metadata(&quarantine)
        .ok()
        .filter(|metadata| metadata.file_type().is_symlink())
        .and_then(|_| fs::read_link(&quarantine).ok());
    let unchanged_and_owned = if moved_target.as_deref() == Some(initial.as_path()) {
        match workspace_link_target_is_owned(workspace_root, &initial) {
            Ok(owned) => owned,
            Err(error) => {
                restore_quarantined_path(&quarantine, link)?;
                return Err(error).context(
                    "revalidating Retread bld ownership after quarantine; restored the workspace path",
                );
            }
        }
    } else {
        false
    };
    if unchanged_and_owned {
        return fs::remove_file(&quarantine).with_context(|| {
            format!(
                "removing quarantined Retread-owned symlink {}",
                quarantine.display()
            )
        });
    }

    restore_quarantined_path(&quarantine, link)?;
    bail!(
        "retread fast-tmp: {} changed concurrently; restored it and refused cleanup",
        link.display()
    )
}

fn setup_bld_symlink(workspace_root: &Path, ns: &Namespace) -> Result<()> {
    setup_bld_symlink_with_hook(workspace_root, ns, || {})
}

#[derive(Debug)]
enum ExpectedBldPath {
    Missing,
    StaleSymlink {
        target: PathBuf,
        reason: StaleSymlinkReason,
    },
    Directory {
        device: u64,
        inode: u64,
        proof: BldDirectoryProof,
    },
}

#[derive(Debug, Clone, Copy)]
enum BldDirectoryProof {
    Empty,
    RetreadMarked,
    Idle,
}

#[derive(Debug)]
enum StaleSymlinkReason {
    Dangling,
    OwnedNamespace,
}

impl ExpectedBldPath {
    fn stale_kind(&self) -> Option<&'static str> {
        match self {
            Self::Missing => None,
            Self::StaleSymlink {
                reason: StaleSymlinkReason::Dangling,
                ..
            } => Some("dangling symlink"),
            Self::StaleSymlink {
                reason: StaleSymlinkReason::OwnedNamespace,
                ..
            } => Some("previous Retread namespace symlink"),
            Self::Directory { .. } => Some("real build directory"),
        }
    }
}

fn bld_has_valid_owner_marker(workspace_root: &Path, directory: &Path) -> Result<bool> {
    let marker = directory.join(RETREAD_BLD_OWNER_MARKER);
    let canonical_workspace = fs::canonicalize(workspace_root)
        .with_context(|| format!("canonicalizing workspace {}", workspace_root.display()))?;
    Ok(ownership_marker_status(
        &marker,
        &bld_owner_marker_contents(&canonical_workspace),
        "bld ownership marker",
    )? == OwnershipMarkerStatus::Valid)
}

fn newest_bld_entry_mtime(directory: &Path) -> Result<Option<SystemTime>> {
    let mut pending = vec![directory.to_path_buf()];
    let mut newest: Option<SystemTime> = None;
    while let Some(current) = pending.pop() {
        for entry in fs::read_dir(&current)
            .with_context(|| format!("reading real build directory {}", current.display()))?
        {
            let entry = entry.with_context(|| format!("reading entry in {}", current.display()))?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .with_context(|| format!("checking build-directory entry {}", path.display()))?;
            let modified = metadata
                .modified()
                .with_context(|| format!("reading modification time for {}", path.display()))?;
            if newest.is_none_or(|current| modified > current) {
                newest = Some(modified);
            }
            if metadata.file_type().is_dir() {
                pending.push(path);
            }
        }
    }
    Ok(newest)
}

fn bld_directory_proof(
    workspace_root: &Path,
    directory: &Path,
    observed_at: SystemTime,
) -> Result<Option<BldDirectoryProof>> {
    if bld_has_valid_owner_marker(workspace_root, directory)? {
        return Ok(Some(BldDirectoryProof::RetreadMarked));
    }
    let Some(newest) = newest_bld_entry_mtime(directory)? else {
        return Ok(Some(BldDirectoryProof::Empty));
    };
    if observed_at
        .duration_since(newest)
        .is_ok_and(|age| age > BLD_IDLE_THRESHOLD)
    {
        Ok(Some(BldDirectoryProof::Idle))
    } else {
        Ok(None)
    }
}

fn ensure_bld_target_is_disjoint(link: &Path, target: &Path) -> Result<()> {
    let canonical_link =
        fs::canonicalize(link).with_context(|| format!("canonicalizing {}", link.display()))?;
    let canonical_target =
        fs::canonicalize(target).with_context(|| format!("canonicalizing {}", target.display()))?;
    if canonical_link.starts_with(&canonical_target)
        || canonical_target.starts_with(&canonical_link)
    {
        bail!(
            "retread fast-tmp refuses to auto-heal {} because build target {} overlaps it",
            link.display(),
            target.display()
        );
    }
    Ok(())
}

fn setup_bld_symlink_with_hook(
    workspace_root: &Path,
    ns: &Namespace,
    before_mutation: impl FnOnce(),
) -> Result<()> {
    let pixi = workspace_root.join(".pixi");
    fs::create_dir_all(&pixi).with_context(|| format!("creating {}", pixi.display()))?;
    let link = pixi.join("bld");
    let target = ns.bld_dir();
    let lock_path = pixi.join(FAST_WORKSPACE_LINK_LOCK);
    let Some(_lock) = try_open_and_lock(&lock_path)? else {
        bail!(
            "retread fast-tmp: {} may be changing concurrently; another fast-tmp build holds {}, so refusing to replace it",
            link.display(),
            lock_path.display()
        );
    };
    let expected = match fs::symlink_metadata(&link) {
        Ok(meta) if meta.file_type().is_symlink() => {
            let current = fs::read_link(&link)
                .with_context(|| format!("reading build symlink {}", link.display()))?;
            if current == target {
                return Ok(());
            }
            let reason = match fs::metadata(&link) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    StaleSymlinkReason::Dangling
                }
                Ok(_) if workspace_link_target_is_owned(workspace_root, &current)? => {
                    StaleSymlinkReason::OwnedNamespace
                }
                Ok(_) => {
                    bail!(
                        "retread fast-tmp refuses to replace live unowned symlink {} -> {}. Move it aside or remove it before enabling fast-tmp.",
                        link.display(),
                        current.display()
                    );
                }
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("checking target of {}", link.display()));
                }
            };
            ExpectedBldPath::StaleSymlink {
                target: current,
                reason,
            }
        }
        Ok(meta) if meta.is_dir() => {
            ensure_bld_target_is_disjoint(&link, &target)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt as _;

                let Some(proof) = bld_directory_proof(workspace_root, &link, SystemTime::now())?
                else {
                    bail!(
                        "retread fast-tmp refuses to auto-heal real build directory {} because it shows activity within the last {} seconds and may belong to a live Pixi/non-Retread build. Remove {} manually only if you know it is stale, then retry.",
                        link.display(),
                        BLD_IDLE_THRESHOLD.as_secs(),
                        link.display()
                    );
                };
                ExpectedBldPath::Directory {
                    device: meta.dev(),
                    inode: meta.ino(),
                    proof,
                }
            }
            #[cfg(not(unix))]
            {
                bail!(
                    "retread fast-tmp cannot safely auto-heal real build directory {} on this platform",
                    link.display()
                );
            }
        }
        Ok(_) => {
            bail!(
                "retread fast-tmp refuses to replace non-directory real path {}. Move it aside before enabling fast-tmp.",
                link.display()
            );
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => ExpectedBldPath::Missing,
        Err(e) => return Err(e).with_context(|| format!("checking {}", link.display())),
    };
    before_mutation();
    atomic_symlink_replace(workspace_root, &target, &link, &expected)?;
    let now = fs::read_link(&link).with_context(|| format!("reading {}", link.display()))?;
    if now != target {
        bail!(
            "retread fast-tmp failed to point {} at {}; it points at {}",
            link.display(),
            target.display(),
            now.display()
        );
    }
    if let Some(kind) = expected.stale_kind() {
        warn_msg(&format!(
            "retread fast-tmp: auto-healed stale {kind} {} to point at {}",
            link.display(),
            target.display()
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn quarantined_bld_matches(
    workspace_root: &Path,
    quarantine: &Path,
    expected: &ExpectedBldPath,
) -> Result<bool> {
    use std::os::unix::fs::MetadataExt as _;

    let Ok(metadata) = fs::symlink_metadata(quarantine) else {
        return Ok(false);
    };
    match expected {
        ExpectedBldPath::Missing => Ok(false),
        ExpectedBldPath::StaleSymlink { target, reason } => {
            if !metadata.file_type().is_symlink()
                || fs::read_link(quarantine).ok().as_deref() != Some(target.as_path())
            {
                return Ok(false);
            }
            match reason {
                StaleSymlinkReason::Dangling => match fs::metadata(quarantine) {
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(true),
                    Ok(_) => Ok(false),
                    Err(error) => Err(error)
                        .with_context(|| format!("rechecking target of {}", quarantine.display())),
                },
                StaleSymlinkReason::OwnedNamespace => {
                    workspace_link_target_is_owned(workspace_root, target)
                }
            }
        }
        ExpectedBldPath::Directory { device, inode, .. } => {
            if !metadata.is_dir() || metadata.dev() != *device || metadata.ino() != *inode {
                return Ok(false);
            }
            Ok(bld_directory_proof(workspace_root, quarantine, SystemTime::now())?.is_some())
        }
    }
}

fn remove_quarantined_bld_path(
    quarantine: &Path,
    expected: &ExpectedBldPath,
) -> std::io::Result<()> {
    match expected {
        ExpectedBldPath::Missing => Ok(()),
        ExpectedBldPath::StaleSymlink { .. } => fs::remove_file(quarantine),
        ExpectedBldPath::Directory {
            proof: BldDirectoryProof::Empty,
            ..
        } => fs::remove_dir(quarantine),
        ExpectedBldPath::Directory { .. } => fs::remove_dir_all(quarantine),
    }
}

#[cfg(unix)]
fn atomic_symlink_replace(
    workspace_root: &Path,
    target: &Path,
    link: &Path,
    expected: &ExpectedBldPath,
) -> Result<()> {
    let parent = link
        .parent()
        .ok_or_else(|| anyhow!("symlink path {} has no parent", link.display()))?;
    let tmp = parent.join(format!(
        ".{}.retread-tmp.{}.{}",
        link.file_name().and_then(|s| s.to_str()).unwrap_or("link"),
        std::process::id(),
        unique_nonce()
    ));
    std::os::unix::fs::symlink(target, &tmp).with_context(|| {
        format!(
            "creating temporary symlink {} -> {}",
            tmp.display(),
            target.display()
        )
    })?;
    let result = (|| -> Result<()> {
        if matches!(expected, ExpectedBldPath::Missing) {
            return match rename_noreplace(&tmp, link) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => bail!(
                    "retread fast-tmp: {} changed concurrently; refusing to replace the newer workspace path",
                    link.display()
                ),
                Err(error) => Err(error).with_context(|| {
                    format!(
                        "installing Retread bld symlink {} without replacing a concurrently created path",
                        link.display()
                    )
                }),
            };
        }

        let quarantine = parent.join(format!(
            ".{}.retread-quarantine.{}.{}",
            link.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("link"),
            std::process::id(),
            unique_nonce()
        ));
        rename_noreplace(link, &quarantine).with_context(|| {
            format!(
                "quarantining {} for ownership revalidation before replacement",
                link.display()
            )
        })?;

        let unchanged_and_stale =
            match quarantined_bld_matches(workspace_root, &quarantine, expected) {
                Ok(matches) => matches,
                Err(error) => {
                    restore_quarantined_path(&quarantine, link)?;
                    return Err(error).context(
                    "revalidating stale bld path before replacement; restored the workspace path",
                );
                }
            };
        if !unchanged_and_stale {
            restore_quarantined_path(&quarantine, link)?;
            bail!(
                "retread fast-tmp: {} changed concurrently; restored it and refused replacement",
                link.display()
            );
        }

        match rename_noreplace(&tmp, link) {
            Ok(()) => remove_quarantined_bld_path(&quarantine, expected).with_context(|| {
                format!("removing replaced stale bld path {}", quarantine.display())
            }),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                remove_quarantined_bld_path(&quarantine, expected).with_context(|| {
                    format!("removing displaced stale bld path {}", quarantine.display())
                })?;
                bail!(
                    "retread fast-tmp: {} changed concurrently; refusing to replace the newer workspace path",
                    link.display()
                )
            }
            Err(error) => {
                restore_quarantined_path(&quarantine, link)?;
                Err(error).with_context(|| {
                    format!(
                        "installing Retread bld symlink {} after ownership revalidation",
                        link.display()
                    )
                })
            }
        }
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

#[cfg(not(unix))]
fn atomic_symlink_replace(
    _workspace_root: &Path,
    _target: &Path,
    _link: &Path,
    _expected: &ExpectedBldPath,
) -> Result<()> {
    bail!("retread fast-tmp symlink setup is only supported on Unix")
}

#[derive(Debug, Clone, Copy)]
enum StaleEnvsSymlinkReason {
    Dangling,
    DanglingTargetPrepared,
    ForeignJob,
}

#[derive(Debug, Clone)]
struct ExpectedEnvsSymlink {
    raw_target: PathBuf,
    owned: OwnedWorkspaceNamespaceTarget,
    current_job_component: String,
    reason: StaleEnvsSymlinkReason,
}

/// Backend-startup repair for Pixi's detached `.pixi/envs` link. This runs
/// beside the older bld-target preflight, before Pixi tries to create an
/// environment through a stale job-local symlink.
pub fn heal_stale_envs_symlink_at_backend_startup(workspace_root: &Path) -> Result<()> {
    let cfg = FastTmpConfig::load(workspace_root);
    let current_ns = match cfg.mode {
        FastTmpMode::Off => None,
        FastTmpMode::Auto if !is_slow(workspace_root, &cfg) => None,
        FastTmpMode::Auto | FastTmpMode::On => Some(namespace(&cfg, workspace_root)),
    };
    heal_stale_envs_symlink(workspace_root, &cfg.tmp_root, current_ns.as_ref())
}

fn heal_stale_envs_symlink(
    workspace_root: &Path,
    configured_tmp_root: &Path,
    current_ns: Option<&Namespace>,
) -> Result<()> {
    heal_stale_envs_symlink_with_warning(workspace_root, configured_tmp_root, current_ns, warn_msg)
}

#[cfg(target_os = "linux")]
fn heal_stale_envs_symlink_with_warning(
    workspace_root: &Path,
    configured_tmp_root: &Path,
    current_ns: Option<&Namespace>,
    emit_warning: impl Fn(&str),
) -> Result<()> {
    let pixi = workspace_root.join(".pixi");
    match fs::metadata(&pixi) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => return Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error).with_context(|| format!("checking {}", pixi.display())),
    }

    // Lock-free fast path: decide whether a repair could even be needed
    // BEFORE creating/locking the link lock file. Opening that lock is a
    // WRITE to the workspace; on a quota-blocked (or otherwise read-only)
    // filesystem it fails with EDQUOT even when the envs symlink is
    // perfectly healthy, and that spurious failure took down backend
    // startup for every environment (observed as `pixi install` runs that
    // produced empty envs). Reads suffice to prove "nothing to do"; the
    // repair path below re-validates everything under the lock, so this
    // pre-pass introduces no race.
    let link = pixi.join("envs");
    match fs::symlink_metadata(&link) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            if let Ok(raw_target) = fs::read_link(&link)
                && fs::metadata(&raw_target).is_ok()
            {
                // Live target: no repair. Preserve the live-foreign-job
                // warning the locked path would have emitted.
                if let Some(owned) = workspace_envs_link_target(
                    workspace_root,
                    configured_tmp_root,
                    &raw_target,
                    false,
                )? && current_ns.is_some()
                    && owned.job_component != current_job_component()
                {
                    emit_warning(&live_foreign_envs_warning(
                        workspace_root,
                        &link,
                        &owned.job_component,
                        &current_job_component(),
                    ));
                }
                return Ok(());
            }
        }
        Ok(_) => return Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| format!("checking {}", link.display()));
        }
    }

    let directory = AnchoredEnvsDirectory::open(&pixi)
        .with_context(|| format!("opening envs-link directory {}", pixi.display()))?;
    let lock_path = pixi.join(FAST_WORKSPACE_LINK_LOCK);
    let _lock = directory.lock(&lock_path)?;
    let link_name = OsStr::new("envs");
    let stat = match directory.stat(link_name) {
        Ok(stat) if rustix::fs::FileType::from_raw_mode(stat.st_mode).is_symlink() => stat,
        Ok(_) => return Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| format!("checking {}", link.display()));
        }
    };
    debug_assert!(rustix::fs::FileType::from_raw_mode(stat.st_mode).is_symlink());

    let raw_target = directory
        .read_link(link_name)
        .with_context(|| format!("reading {}", link.display()))?;
    let dangling = match fs::metadata(&raw_target) {
        Ok(_) => false,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
        Err(error) => {
            return Err(error).with_context(|| format!("checking target {}", raw_target.display()));
        }
    };
    let Some(owned) =
        workspace_envs_link_target(workspace_root, configured_tmp_root, &raw_target, dangling)?
    else {
        return Ok(());
    };
    let current_job = current_job_component();

    let Some(current_ns) = current_ns else {
        if !dangling {
            return Ok(());
        }
        let expected = ExpectedEnvsSymlink {
            raw_target,
            owned,
            current_job_component: current_job,
            reason: StaleEnvsSymlinkReason::Dangling,
        };
        atomic_remove_stale_envs_symlink(
            &directory,
            link_name,
            workspace_root,
            configured_tmp_root,
            &link,
            &expected,
        )?;
        emit_warning(&format!(
            "retread fast-tmp: removed dangling envs symlink while fast-tmp is disabled (old job {}) at {}",
            envs_job_label(&expected.owned.job_component),
            link.display()
        ));
        return Ok(());
    };

    let foreign_job = owned.job_component != current_job;
    if !dangling {
        if foreign_job {
            emit_warning(&live_foreign_envs_warning(
                workspace_root,
                &link,
                &owned.job_component,
                &current_job,
            ));
        }
        return Ok(());
    }
    let reason = if foreign_job {
        StaleEnvsSymlinkReason::ForeignJob
    } else {
        StaleEnvsSymlinkReason::DanglingTargetPrepared
    };
    let tmp_root = current_ns.root.ancestors().nth(3).ok_or_else(|| {
        anyhow!(
            "Retread envs namespace {} has no tmp root",
            current_ns.root.display()
        )
    })?;
    fs::create_dir_all(tmp_root)
        .with_context(|| format!("creating tmp root {}", tmp_root.display()))?;
    enforce_tmp_user_dir(tmp_root)?;
    let new_target = current_ns.root.join(&owned.relative);
    fs::create_dir_all(&new_target)
        .with_context(|| format!("creating current envs target {}", new_target.display()))?;
    set_dir_mode_0700(&current_ns.root);
    let canonical_workspace = fs::canonicalize(workspace_root)
        .with_context(|| format!("canonicalizing workspace {}", workspace_root.display()))?;
    write_namespace_ownership_proof(current_ns, &canonical_workspace)?;
    let expected = ExpectedEnvsSymlink {
        raw_target,
        owned,
        current_job_component: current_job,
        reason,
    };
    if let Err(error) = atomic_replace_stale_envs_symlink(
        &directory,
        link_name,
        workspace_root,
        configured_tmp_root,
        &link,
        &new_target,
        &expected,
    ) {
        let became_live = matches!(expected.reason, StaleEnvsSymlinkReason::ForeignJob)
            && directory.read_link(link_name).ok().as_deref()
                == Some(expected.raw_target.as_path())
            && fs::metadata(&expected.raw_target).is_ok();
        if became_live {
            emit_warning(&live_foreign_envs_warning(
                workspace_root,
                &link,
                &expected.owned.job_component,
                &expected.current_job_component,
            ));
            return Ok(());
        }
        return Err(error);
    }
    let installed = directory
        .read_link(link_name)
        .with_context(|| format!("reading {}", link.display()))?;
    if installed != new_target {
        bail!(
            "retread fast-tmp failed to point {} at {}; it points at {}",
            link.display(),
            new_target.display(),
            installed.display()
        );
    }
    emit_warning(&format!(
        "retread fast-tmp: healed stale envs symlink (old job {} -> job {}) {} -> {}",
        envs_job_label(&expected.owned.job_component),
        envs_job_label(&expected.current_job_component),
        link.display(),
        new_target.display()
    ));
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn heal_stale_envs_symlink_with_warning(
    _workspace_root: &Path,
    _configured_tmp_root: &Path,
    _current_ns: Option<&Namespace>,
    _emit_warning: impl Fn(&str),
) -> Result<()> {
    Ok(())
}

fn envs_job_label(job_component: &str) -> &str {
    job_component.strip_prefix("job-").unwrap_or(job_component)
}

fn live_foreign_envs_warning(
    workspace_root: &Path,
    link: &Path,
    foreign_job_component: &str,
    current_job_component: &str,
) -> String {
    format!(
        "retread fast-tmp: LIVE JOB COLLISION: refusing to repoint {} from live job {} to current job {} because two live jobs share workspace {}; use a per-job workspace or disable fast-tmp",
        link.display(),
        envs_job_label(foreign_job_component),
        envs_job_label(current_job_component),
        workspace_root.display()
    )
}

fn expected_envs_raw_target_matches(
    workspace_root: &Path,
    configured_tmp_root: &Path,
    raw_target: &Path,
    expected: &ExpectedEnvsSymlink,
) -> Result<bool> {
    if raw_target != expected.raw_target {
        return Ok(false);
    }
    let Some(owned) =
        workspace_envs_link_target(workspace_root, configured_tmp_root, raw_target, true)?
    else {
        return Ok(false);
    };
    if owned != expected.owned {
        return Ok(false);
    }
    match expected.reason {
        StaleEnvsSymlinkReason::Dangling => match fs::metadata(raw_target) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(true),
            Ok(_) => Ok(false),
            Err(error) => {
                Err(error).with_context(|| format!("rechecking target {}", raw_target.display()))
            }
        },
        StaleEnvsSymlinkReason::DanglingTargetPrepared => Ok(true),
        StaleEnvsSymlinkReason::ForeignJob
            if owned.job_component != expected.current_job_component =>
        {
            match fs::metadata(raw_target) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(true),
                Ok(_) => Ok(false),
                Err(error) => Err(error)
                    .with_context(|| format!("rechecking target {}", raw_target.display())),
            }
        }
        StaleEnvsSymlinkReason::ForeignJob => Ok(false),
    }
}

#[cfg(target_os = "linux")]
fn restore_anchored_envs_quarantine(
    directory: &AnchoredEnvsDirectory,
    quarantine_name: &OsStr,
    quarantine_display: &Path,
    quarantine_identity: AnchoredPathIdentity,
    link_name: &OsStr,
    link_display: &Path,
) -> Result<()> {
    if directory.identity(quarantine_name).with_context(|| {
        format!(
            "checking quarantined envs symlink {} before restoration",
            quarantine_display.display()
        )
    })? != quarantine_identity
    {
        bail!(
            "retread fast-tmp: quarantine {} changed concurrently; preserved it and refused to overwrite {}",
            quarantine_display.display(),
            link_display.display()
        );
    }
    match directory.stat(link_name) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Ok(_) => bail!(
            "retread fast-tmp: {} changed concurrently; preserved the displaced path at {} without replacing the newer workspace path",
            link_display.display(),
            quarantine_display.display()
        ),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "checking {} before restoring {}",
                    link_display.display(),
                    quarantine_display.display()
                )
            });
        }
    }
    directory
        .rename_over(quarantine_name, link_name)
        .with_context(|| {
            format!(
                "atomically restoring {} from {}",
                link_display.display(),
                quarantine_display.display()
            )
        })
}

#[cfg(target_os = "linux")]
fn atomic_remove_stale_envs_symlink(
    directory: &AnchoredEnvsDirectory,
    link_name: &OsStr,
    workspace_root: &Path,
    configured_tmp_root: &Path,
    link: &Path,
    expected: &ExpectedEnvsSymlink,
) -> Result<()> {
    let parent = link
        .parent()
        .ok_or_else(|| anyhow!("symlink path {} has no parent", link.display()))?;
    let quarantine_name = OsString::from(format!(
        ".envs.retread-quarantine.{}.{}",
        std::process::id(),
        unique_nonce()
    ));
    let quarantine = parent.join(&quarantine_name);
    if !expected_envs_symlink_matches_at(
        directory,
        link_name,
        workspace_root,
        configured_tmp_root,
        expected,
    )? {
        bail!(
            "retread fast-tmp: {} changed or became live concurrently; refused envs-link removal",
            link.display()
        );
    }
    match directory.stat(&quarantine_name) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Ok(_) => bail!(
            "retread fast-tmp: refusing to overwrite unexpected quarantine path {}",
            quarantine.display()
        ),
        Err(error) => {
            return Err(error).with_context(|| format!("checking {}", quarantine.display()));
        }
    }
    directory
        .rename_over(link_name, &quarantine_name)
        .with_context(|| {
            format!(
                "atomically quarantining stale envs symlink {} as {}",
                link.display(),
                quarantine.display()
            )
        })?;
    let quarantine_identity = directory.identity(&quarantine_name).with_context(|| {
        format!(
            "identifying quarantined envs symlink {}",
            quarantine.display()
        )
    })?;
    let matches = match expected_envs_symlink_matches_at(
        directory,
        &quarantine_name,
        workspace_root,
        configured_tmp_root,
        expected,
    ) {
        Ok(matches) => matches,
        Err(error) => {
            restore_anchored_envs_quarantine(
                directory,
                &quarantine_name,
                &quarantine,
                quarantine_identity,
                link_name,
                link,
            )?;
            return Err(error).context(
                "revalidating stale envs symlink before mutation; restored the workspace path",
            );
        }
    };
    if !matches {
        restore_anchored_envs_quarantine(
            directory,
            &quarantine_name,
            &quarantine,
            quarantine_identity,
            link_name,
            link,
        )?;
        bail!(
            "retread fast-tmp: {} changed concurrently; restored it and refused envs-link mutation",
            link.display()
        );
    }
    if directory.identity(&quarantine_name).with_context(|| {
        format!(
            "rechecking quarantined envs symlink {} before unlink",
            quarantine.display()
        )
    })? != quarantine_identity
    {
        bail!(
            "retread fast-tmp: quarantine {} changed concurrently; preserved the newer path",
            quarantine.display()
        );
    }
    let still_dangling = match expected_envs_symlink_matches_at(
        directory,
        &quarantine_name,
        workspace_root,
        configured_tmp_root,
        expected,
    ) {
        Ok(matches) => matches,
        Err(error) => {
            restore_anchored_envs_quarantine(
                directory,
                &quarantine_name,
                &quarantine,
                quarantine_identity,
                link_name,
                link,
            )?;
            return Err(error).context(
                "final dangling-target check failed; restored the workspace envs symlink",
            );
        }
    };
    if !still_dangling {
        restore_anchored_envs_quarantine(
            directory,
            &quarantine_name,
            &quarantine,
            quarantine_identity,
            link_name,
            link,
        )?;
        bail!(
            "retread fast-tmp: target of {} became live concurrently; restored it and refused envs-link removal",
            link.display()
        );
    }
    match directory.unlink(&quarantine_name) {
        Ok(()) => Ok(()),
        Err(error) => {
            restore_anchored_envs_quarantine(
                directory,
                &quarantine_name,
                &quarantine,
                quarantine_identity,
                link_name,
                link,
            )?;
            Err(error).with_context(|| {
                format!(
                    "removing quarantined stale envs symlink {}",
                    quarantine.display()
                )
            })
        }
    }
}

#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AnchoredPathIdentity {
    device: u64,
    inode: u64,
}

/// All names involved in the envs-link swap stay relative to one open
/// directory. A concurrent rename/replacement of `.pixi` therefore cannot
/// redirect validation, publication, or cleanup into a different directory.
#[cfg(target_os = "linux")]
struct AnchoredEnvsDirectory {
    directory: File,
}

#[cfg(target_os = "linux")]
impl AnchoredEnvsDirectory {
    fn open(parent: &Path) -> std::io::Result<Self> {
        Ok(Self {
            directory: File::open(parent)?,
        })
    }

    fn stat(&self, name: &OsStr) -> std::io::Result<rustix::fs::Stat> {
        rustix::fs::statat(&self.directory, name, rustix::fs::AtFlags::SYMLINK_NOFOLLOW)
            .map_err(std::io::Error::from)
    }

    fn identity(&self, name: &OsStr) -> std::io::Result<AnchoredPathIdentity> {
        let stat = self.stat(name)?;
        Ok(AnchoredPathIdentity {
            device: stat.st_dev,
            inode: stat.st_ino,
        })
    }

    fn read_link(&self, name: &OsStr) -> std::io::Result<PathBuf> {
        use std::os::unix::ffi::OsStringExt as _;

        let target = rustix::fs::readlinkat(&self.directory, name, Vec::new())
            .map_err(std::io::Error::from)?;
        Ok(PathBuf::from(OsString::from_vec(target.into_bytes())))
    }

    fn symlink(&self, target: &Path, name: &OsStr) -> std::io::Result<()> {
        rustix::fs::symlinkat(target, &self.directory, name).map_err(std::io::Error::from)
    }

    fn rename_over(&self, source: &OsStr, destination: &OsStr) -> std::io::Result<()> {
        rustix::fs::renameat(&self.directory, source, &self.directory, destination)
            .map_err(std::io::Error::from)
    }

    fn unlink(&self, name: &OsStr) -> std::io::Result<()> {
        rustix::fs::unlinkat(&self.directory, name, rustix::fs::AtFlags::empty())
            .map_err(std::io::Error::from)
    }

    fn lock(&self, lock_display: &Path) -> Result<FileLock> {
        let file = rustix::fs::openat(
            &self.directory,
            FAST_WORKSPACE_LINK_LOCK,
            rustix::fs::OFlags::CREATE
                | rustix::fs::OFlags::WRONLY
                | rustix::fs::OFlags::CLOEXEC
                | rustix::fs::OFlags::NOFOLLOW,
            rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
        )
        .map(File::from)
        .map_err(std::io::Error::from)
        .with_context(|| format!("opening anchored lock file {}", lock_display.display()))?;
        fs4::fs_std::FileExt::lock_exclusive(&file)
            .with_context(|| format!("locking {}", lock_display.display()))?;
        Ok(FileLock { file })
    }
}

#[cfg(target_os = "linux")]
fn expected_envs_symlink_matches_at(
    directory: &AnchoredEnvsDirectory,
    name: &OsStr,
    workspace_root: &Path,
    configured_tmp_root: &Path,
    expected: &ExpectedEnvsSymlink,
) -> Result<bool> {
    let stat = match directory.stat(name) {
        Ok(stat) => stat,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error).context("checking anchored envs-link entry"),
    };
    if !rustix::fs::FileType::from_raw_mode(stat.st_mode).is_symlink() {
        return Ok(false);
    }
    let raw_target = directory
        .read_link(name)
        .context("reading anchored envs-link entry")?;
    expected_envs_raw_target_matches(workspace_root, configured_tmp_root, &raw_target, expected)
}

#[cfg(target_os = "linux")]
fn cleanup_prepared_envs_symlink(
    directory: &AnchoredEnvsDirectory,
    tmp_name: &OsStr,
    tmp_display: &Path,
    prepared_identity: AnchoredPathIdentity,
) -> Result<()> {
    match directory.identity(tmp_name) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("checking temporary envs symlink {}", tmp_display.display())),
        Ok(identity) if identity != prepared_identity => bail!(
            "retread fast-tmp: temporary path {} changed concurrently; preserved the newer path",
            tmp_display.display()
        ),
        Ok(_) => directory.unlink(tmp_name).with_context(|| {
            format!(
                "removing uninstalled temporary envs symlink {}",
                tmp_display.display()
            )
        }),
    }
}

#[cfg(target_os = "linux")]
fn atomic_replace_stale_envs_symlink(
    directory: &AnchoredEnvsDirectory,
    link_name: &OsStr,
    workspace_root: &Path,
    configured_tmp_root: &Path,
    link: &Path,
    new_target: &Path,
    expected: &ExpectedEnvsSymlink,
) -> Result<()> {
    atomic_replace_stale_envs_symlink_with_hook(
        directory,
        link_name,
        workspace_root,
        configured_tmp_root,
        link,
        new_target,
        expected,
        |_| {},
    )
}

#[cfg(target_os = "linux")]
fn atomic_replace_stale_envs_symlink_with_hook(
    directory: &AnchoredEnvsDirectory,
    link_name: &OsStr,
    workspace_root: &Path,
    configured_tmp_root: &Path,
    link: &Path,
    new_target: &Path,
    expected: &ExpectedEnvsSymlink,
    before_rename: impl FnOnce(&Path),
) -> Result<()> {
    let parent = link
        .parent()
        .ok_or_else(|| anyhow!("symlink path {} has no parent", link.display()))?;
    let tmp_name = OsString::from(format!(
        ".envs.retread-tmp.{}.{}",
        std::process::id(),
        unique_nonce()
    ));
    let tmp = parent.join(&tmp_name);
    directory.symlink(new_target, &tmp_name).with_context(|| {
        format!(
            "creating temporary envs symlink {} -> {}",
            tmp.display(),
            new_target.display()
        )
    })?;
    let prepared_identity = match directory.identity(&tmp_name) {
        Ok(identity) => identity,
        Err(error) => {
            let cleanup = directory.unlink(&tmp_name);
            if let Err(cleanup_error) = cleanup {
                return Err(cleanup_error).with_context(|| {
                    format!(
                        "cleaning temporary envs symlink {} after identity check failed: {error}",
                        tmp.display()
                    )
                });
            }
            return Err(error)
                .with_context(|| format!("identifying temporary envs symlink {}", tmp.display()));
        }
    };

    let result = (|| -> Result<()> {
        if !expected_envs_symlink_matches_at(
            directory,
            link_name,
            workspace_root,
            configured_tmp_root,
            expected,
        )? {
            bail!(
                "retread fast-tmp: {} changed concurrently; refused envs-link replacement",
                link.display()
            );
        }
        before_rename(&tmp);
        if directory.identity(&tmp_name).with_context(|| {
            format!(
                "rechecking prepared envs symlink {} before publication",
                tmp.display()
            )
        })? != prepared_identity
        {
            bail!(
                "retread fast-tmp: prepared envs symlink {} changed concurrently; refused publication",
                tmp.display()
            );
        }
        // This is the final check immediately adjacent to publication. In
        // particular, a foreign target that became live while we prepared the
        // replacement is never repointed, even transiently.
        if !expected_envs_symlink_matches_at(
            directory,
            link_name,
            workspace_root,
            configured_tmp_root,
            expected,
        )? {
            bail!(
                "retread fast-tmp: {} changed or became live concurrently; refused envs-link replacement",
                link.display()
            );
        }
        // POSIX rename-over is one atomic namespace operation and is supported
        // by the NFS/Lustre workspace filesystems fast-tmp is designed for.
        // The caller holds the flock opened through this same directory FD,
        // which serializes this final proof/publication pair with writers.
        directory
            .rename_over(&tmp_name, link_name)
            .with_context(|| {
                format!(
                    "atomically renaming prepared Retread envs symlink {} over {}",
                    tmp.display(),
                    link.display()
                )
            })
    })();

    if let Err(operation_error) = result {
        if let Err(cleanup_error) =
            cleanup_prepared_envs_symlink(directory, &tmp_name, &tmp, prepared_identity)
        {
            return Err(cleanup_error).context(format!(
                "envs-link replacement also failed: {operation_error:#}"
            ));
        }
        return Err(operation_error);
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn atomic_replace_stale_envs_symlink(
    _workspace_root: &Path,
    _configured_tmp_root: &Path,
    _link: &Path,
    _new_target: &Path,
    _expected: &ExpectedEnvsSymlink,
) -> Result<()> {
    bail!("retread fast-tmp envs symlink setup is only supported on Unix")
}

fn warn_if_real_envs_dir(workspace_root: &Path) {
    let envs = workspace_root.join(".pixi").join("envs");
    let Ok(meta) = fs::symlink_metadata(&envs) else {
        return;
    };
    if meta.file_type().is_symlink() || !meta.is_dir() {
        return;
    }
    let non_empty = fs::read_dir(&envs)
        .ok()
        .and_then(|mut it| it.next())
        .is_some();
    if non_empty {
        warn_msg(&format!(
            "retread fast-tmp: {} is a real non-empty directory; Pixi detached-environments may not take effect until it is moved or removed",
            envs.display()
        ));
    }
}

pub fn check_env_eviction(workspace_root: &Path, ns: &Namespace) {
    if in_slurm_job() {
        return;
    }
    let envs = workspace_root.join(".pixi").join("envs");
    let Ok(meta) = fs::symlink_metadata(&envs) else {
        return;
    };
    if !meta.file_type().is_symlink() {
        return;
    }
    let Ok(raw_target) = fs::read_link(&envs) else {
        return;
    };
    let target = if raw_target.is_absolute() {
        raw_target
    } else {
        envs.parent().unwrap_or(workspace_root).join(raw_target)
    };
    let cold = !target.exists()
        || (target.starts_with(ns.envs_dir())
            && fs::read_dir(&target)
                .ok()
                .map(|mut it| it.next().is_none())
                .unwrap_or(false));
    if cold {
        warn_msg(&format!(
            "retread fast-tmp: envs evicted or cold at {}; pixi will reinstall from pixi.lock",
            target.display()
        ));
    }
}

pub fn print_mapping(engaged: &EngagedFastTmp) {
    eprintln!("retread fast-tmp namespace: {}", engaged.ns.root.display());
    for (key, value) in &engaged.env {
        eprintln!("retread fast-tmp env: {key}={value}");
    }
}

pub fn shell_exports(engaged: &EngagedFastTmp) -> String {
    let mut out = String::new();
    for (key, value) in &engaged.env {
        out.push_str("export ");
        out.push_str(key);
        out.push('=');
        out.push_str(&shell_quote(value));
        out.push('\n');
    }
    out
}

/// Shell commands for disengaging a previously sourced fast-tmp environment,
/// including same-job `RETREAD_FAST_TMP=off`. Only keys recorded as managed
/// are changed, and every overwritten user value is restored.
pub fn shell_stale_cleanup() -> String {
    if !inherited_fasttmp_cleanup_needed() {
        return String::new();
    }
    let mut out = String::new();
    for (key, value) in cleanup_env_actions() {
        match value {
            Some(value) => {
                out.push_str("export ");
                out.push_str(&key);
                out.push('=');
                out.push_str(&shell_quote(&value));
                out.push('\n');
            }
            None => {
                out.push_str("unset ");
                out.push_str(&key);
                out.push('\n');
            }
        }
    }
    out
}

fn shell_quote(s: &str) -> String {
    let mut out = String::from("'");
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

pub fn remove_stale_fast_env_from_command(cmd: &mut Command) {
    if inherited_fasttmp_cleanup_needed() {
        for (key, value) in cleanup_env_actions() {
            match value {
                Some(value) => {
                    cmd.env(key, value);
                }
                None => {
                    cmd.env_remove(key);
                }
            }
        }
    }
}

pub fn find_workspace_root(start: &Path) -> Result<PathBuf> {
    let mut dir = if start.is_file() {
        start
            .parent()
            .ok_or_else(|| anyhow!("{} has no parent", start.display()))?
            .to_path_buf()
    } else {
        start.to_path_buf()
    };
    loop {
        if dir.join("pixi.toml").is_file() {
            return Ok(dir);
        }
        if !dir.pop() {
            break;
        }
    }
    bail!(
        "retread fast: could not find pixi.toml walking up from {}",
        start.display()
    )
}

pub fn preflight_locks(shared_path: &Path) -> Result<()> {
    fs::create_dir_all(shared_path)
        .with_context(|| format!("creating shared preflight path {}", shared_path.display()))?;
    let lock_path = shared_path.join(".retread-fast-preflight.lock");
    let lock = open_and_lock(&lock_path)?;
    let exe = std::env::current_exe().context("locating current executable for lock helper")?;
    let lock_arg = lock_path.to_string_lossy().to_string();
    let mut child = Command::new(exe)
        .args(["fast", "--preflight-lock-helper", &lock_arg])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawning retread fast lock preflight helper")?;
    std::thread::sleep(Duration::from_millis(250));
    if let Some(status) = child
        .try_wait()
        .context("checking lock preflight helper status")?
    {
        drop(lock);
        bail!(
            "retread fast lock preflight failed: helper exited before parent released the lock ({status}); shared-side flock did not block"
        );
    }
    drop(lock);
    let output = child
        .wait_with_output()
        .context("waiting for lock preflight helper")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "retread fast lock preflight helper failed with status {}: {}",
            output.status,
            stderr.trim()
        );
    }
    eprintln!(
        "retread fast lock preflight passed for {} (local helper); run the helper under srun on new clusters when cross-node NFS locking has not been validated",
        shared_path.display()
    );
    Ok(())
}

pub fn preflight_lock_helper(lock_path: &Path) -> Result<()> {
    let _lock = open_and_lock(lock_path)?;
    Ok(())
}

pub fn stage_dir(ns: &Namespace, token: &str) -> PathBuf {
    ns.bld_dir().join(format!(
        "out-{token}-{}-{}",
        std::process::id(),
        unique_nonce()
    ))
}

pub fn copy_back_artifacts(
    stage_dir: &Path,
    output_dir: &Path,
    returned_output_file: &Path,
) -> Result<PathBuf> {
    fs::create_dir_all(output_dir)
        .with_context(|| format!("creating output dir {}", output_dir.display()))?;
    let artifacts = collect_artifacts(stage_dir)?;
    if artifacts.is_empty() {
        bail!(
            "retread fast-tmp staged build produced no .conda/.tar.bz2 artifacts under {}",
            stage_dir.display()
        );
    }
    let _lock = open_and_lock(&output_dir.join(".retread-copyback.lock"))?;
    let mut returned_final = None;
    for src in artifacts {
        let rel = src.strip_prefix(stage_dir).with_context(|| {
            format!(
                "staged artifact {} is not under stage dir {}",
                src.display(),
                stage_dir.display()
            )
        })?;
        let dst = output_dir.join(rel);
        copy_one_verified(&src, &dst)?;
        if src == returned_output_file {
            returned_final = Some(dst);
        }
    }
    returned_final.ok_or_else(|| {
        anyhow!(
            "retread fast-tmp did not copy returned artifact {} from stage {}",
            returned_output_file.display(),
            stage_dir.display()
        )
    })
}

fn collect_artifacts(root: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    collect_artifacts_inner(root, &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_artifacts_inner(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            collect_artifacts_inner(&path, out)?;
        } else if meta.is_file()
            && path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(is_conda_artifact_name)
        {
            out.push(path);
        }
    }
    Ok(())
}

fn is_conda_artifact_name(name: &str) -> bool {
    name.ends_with(".conda") || name.ends_with(".tar.bz2")
}

fn copy_one_verified(src: &Path, dst: &Path) -> Result<()> {
    let parent = dst
        .parent()
        .ok_or_else(|| anyhow!("destination artifact {} has no parent", dst.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("creating artifact output dir {}", parent.display()))?;
    let part = parent.join(format!(
        ".{}.part",
        dst.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("artifact")
    ));
    let mut last_mismatch = None;
    for attempt in 0..2 {
        fs::copy(src, &part).with_context(|| {
            format!(
                "copying staged artifact {} to part file {}",
                src.display(),
                part.display()
            )
        })?;
        maybe_corrupt_copyback_part(&part)?;
        File::open(&part)
            .and_then(|f| f.sync_all())
            .with_context(|| format!("fsyncing copy-back part file {}", part.display()))?;
        fs::rename(&part, dst).with_context(|| {
            format!(
                "atomically placing copy-backed artifact {} -> {}",
                part.display(),
                dst.display()
            )
        })?;
        let _ = fsync_dir(parent);
        let src_size = fs::metadata(src)?.len();
        let dst_size = fs::metadata(dst)?.len();
        let src_hash = sha256_file(src)?;
        let dst_hash = sha256_file(dst)?;
        if src_size == dst_size && src_hash == dst_hash {
            return Ok(());
        }
        last_mismatch = Some((src_size, dst_size, src_hash, dst_hash));
        if attempt == 0 {
            warn_msg(&format!(
                "retread fast-tmp: copy-back verification mismatch for {}; retrying once",
                dst.display()
            ));
        }
    }
    let (src_size, dst_size, src_hash, dst_hash) = last_mismatch.unwrap();
    bail!(
        "retread fast-tmp copy-back verification failed for {} -> {}: source size/hash {}/{} destination size/hash {}/{}",
        src.display(),
        dst.display(),
        src_size,
        src_hash,
        dst_size,
        dst_hash
    )
}

fn maybe_corrupt_copyback_part(part: &Path) -> Result<()> {
    let Ok(mode) = std::env::var("RETREAD_FAST_TMP_CORRUPT_COPYBACK") else {
        return Ok(());
    };
    let corrupt = match mode.as_str() {
        "once" => !CORRUPT_COPYBACK_ONCE_USED.swap(true, Ordering::SeqCst),
        "always" => true,
        _ => false,
    };
    if corrupt {
        OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(part)
            .with_context(|| {
                format!("opening copy-back corruption test hook {}", part.display())
            })?;
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("reading {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn fsync_dir(path: &Path) -> std::io::Result<()> {
    File::open(path).and_then(|f| f.sync_all())
}

fn enforce_budget(tmp_root: &Path, workspace_root: &Path, cfg: &FastTmpConfig) -> Result<()> {
    let estimate = estimate_need_bytes(workspace_root);
    let (budget, source) = effective_budget(tmp_root, cfg)?;
    if let Some(budget) = budget {
        if estimate > budget {
            bail!(
                "retread fast-tmp budget too small: estimated need {} bytes > budget {} bytes from {}. Increase --mem, set RETREAD_FAST_TMP_BUDGET_BYTES/[tool.retread.fast-tmp] budget-bytes, or turn fast-tmp off for this job.",
                estimate,
                budget,
                source
            );
        }
    } else if is_tmpfs(tmp_root) {
        warn_msg(&format!(
            "retread fast-tmp: tmp-root {} is tmpfs/RAM but no cgroup or SLURM memory budget was readable; expected RAM cost is about {} bytes",
            tmp_root.display(),
            estimate
        ));
    }
    Ok(())
}

fn estimate_need_bytes(workspace_root: &Path) -> u64 {
    let lock = workspace_root.join("pixi.lock");
    let Ok(text) = fs::read_to_string(lock) else {
        return DEFAULT_ESTIMATE_BYTES;
    };
    let mut total = 0_u64;
    for line in text.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("size:") else {
            continue;
        };
        if let Some(bytes) = parse_bytes_value(rest.trim()) {
            total = total.saturating_add(bytes);
        }
    }
    if total == 0 {
        DEFAULT_ESTIMATE_BYTES
    } else {
        total
    }
}

fn effective_budget(tmp_root: &Path, cfg: &FastTmpConfig) -> Result<(Option<u64>, &'static str)> {
    if let Some(bytes) = cfg.budget_bytes {
        return Ok((Some(bytes), "configured budget-bytes"));
    }
    if is_tmpfs(tmp_root) {
        if let Some(bytes) = cgroup_v2_available_memory()? {
            return Ok((
                Some(bytes.saturating_mul(75) / 100),
                "cgroup v2 memory.max-current",
            ));
        }
        if let Some(bytes) = slurm_memory_budget_bytes() {
            return Ok((
                Some(bytes.saturating_mul(75) / 100),
                "SLURM memory environment",
            ));
        }
        return Ok((None, "unavailable tmpfs memory budget"));
    }
    Ok((statvfs_available_bytes(tmp_root), "statvfs on non-tmpfs"))
}

fn slurm_memory_budget_bytes() -> Option<u64> {
    if let Ok(mem) = std::env::var("SLURM_MEM_PER_NODE")
        && let Some(mb) = parse_slurm_mb(&mem)
    {
        return Some(mb.saturating_mul(1024 * 1024));
    }
    let per_cpu = std::env::var("SLURM_MEM_PER_CPU")
        .ok()
        .and_then(|s| parse_slurm_mb(&s))?;
    let cpus = std::env::var("SLURM_CPUS_ON_NODE")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())?;
    Some(per_cpu.saturating_mul(cpus).saturating_mul(1024 * 1024))
}

fn parse_slurm_mb(s: &str) -> Option<u64> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.chars().all(|c| c.is_ascii_digit()) {
        return trimmed.parse().ok();
    }
    parse_bytes_value(trimmed).map(|bytes| bytes / (1024 * 1024))
}

fn cgroup_v2_available_memory() -> Result<Option<u64>> {
    let Ok(cgroup) = fs::read_to_string("/proc/self/cgroup") else {
        return Ok(None);
    };
    let Some(rel) = cgroup.lines().find_map(|line| line.strip_prefix("0::")) else {
        return Ok(None);
    };
    let dir = Path::new("/sys/fs/cgroup").join(rel.trim_start_matches('/'));
    let max = fs::read_to_string(dir.join("memory.max")).ok();
    let current = fs::read_to_string(dir.join("memory.current")).ok();
    let (Some(max), Some(current)) = (max, current) else {
        return Ok(None);
    };
    let max = max.trim();
    if max == "max" {
        return Ok(None);
    }
    let max = max.parse::<u64>().ok();
    let current = current.trim().parse::<u64>().ok();
    Ok(match (max, current) {
        (Some(max), Some(current)) if max > current => Some(max - current),
        _ => None,
    })
}

fn with_file_lock<T>(lock_path: &Path, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let _lock = open_and_lock(lock_path)?;
    f()
}

struct FileLock {
    file: File,
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs4::fs_std::FileExt::unlock(&self.file);
    }
}

fn open_and_lock(lock_path: &Path) -> Result<FileLock> {
    let file = open_lock_file(lock_path)?;
    fs4::fs_std::FileExt::lock_exclusive(&file)
        .with_context(|| format!("locking {}", lock_path.display()))?;
    Ok(FileLock { file })
}

fn try_open_and_lock(lock_path: &Path) -> Result<Option<FileLock>> {
    let file = open_lock_file(lock_path)?;
    if fs4::fs_std::FileExt::try_lock_exclusive(&file)
        .with_context(|| format!("try-locking {}", lock_path.display()))?
    {
        Ok(Some(FileLock { file }))
    } else {
        Ok(None)
    }
}

fn open_lock_file(lock_path: &Path) -> Result<File> {
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock_path)
        .with_context(|| format!("opening lock file {}", lock_path.display()))
}

fn unique_nonce() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn warn_msg(msg: &str) {
    eprintln!("retread warning: {msg}");
    tracing::warn!("{msg}");
}

#[cfg(unix)]
fn set_dir_mode_0700(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(meta) = fs::metadata(path) {
        let mut perms = meta.permissions();
        perms.set_mode(0o700);
        let _ = fs::set_permissions(path, perms);
    }
}

#[cfg(not(unix))]
fn set_dir_mode_0700(_path: &Path) {}

#[cfg(unix)]
fn metadata_uid(meta: &fs::Metadata) -> u32 {
    use std::os::unix::fs::MetadataExt;
    meta.uid()
}

#[cfg(not(unix))]
fn metadata_uid(_meta: &fs::Metadata) -> u32 {
    0
}

#[cfg(target_os = "linux")]
fn current_uid() -> u32 {
    unsafe { sys::getuid() }
}

#[cfg(not(target_os = "linux"))]
fn current_uid() -> u32 {
    0
}

#[cfg(target_os = "linux")]
fn statfs_magic(path: &Path) -> Option<u64> {
    let c_path = c_path(path).ok()?;
    let mut stat: sys::StatFs = unsafe { std::mem::zeroed() };
    let rc = unsafe { sys::statfs(c_path.as_ptr(), &mut stat) };
    (rc == 0).then_some(stat.f_type as u64)
}

#[cfg(not(target_os = "linux"))]
fn statfs_magic(_path: &Path) -> Option<u64> {
    None
}

fn is_tmpfs(path: &Path) -> bool {
    if let Ok(force) = std::env::var("RETREAD_FAST_TMP_FORCE_FS")
        && force.eq_ignore_ascii_case("tmpfs")
    {
        return true;
    }
    statfs_magic(path) == Some(0x01021994)
}

#[cfg(target_os = "linux")]
fn statvfs_available_bytes(path: &Path) -> Option<u64> {
    let c_path = c_path(path).ok()?;
    let mut stat: sys::StatVfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { sys::statvfs(c_path.as_ptr(), &mut stat) };
    if rc != 0 {
        return None;
    }
    Some((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
}

#[cfg(not(target_os = "linux"))]
fn statvfs_available_bytes(_path: &Path) -> Option<u64> {
    None
}

#[cfg(unix)]
fn c_path(path: &Path) -> Result<CString> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(path.as_os_str().as_bytes()).with_context(|| {
        format!(
            "path contains interior NUL and cannot be passed to statfs/statvfs: {}",
            path.display()
        )
    })
}

#[cfg(target_os = "linux")]
mod sys {
    use std::os::raw::{c_char, c_int, c_long, c_ulong};

    #[repr(C)]
    pub struct FsId {
        pub val: [c_int; 2],
    }

    #[repr(C)]
    pub struct StatFs {
        pub f_type: c_long,
        pub f_bsize: c_long,
        pub f_blocks: c_ulong,
        pub f_bfree: c_ulong,
        pub f_bavail: c_ulong,
        pub f_files: c_ulong,
        pub f_ffree: c_ulong,
        pub f_fsid: FsId,
        pub f_namelen: c_long,
        pub f_frsize: c_long,
        pub f_flags: c_long,
        pub f_spare: [c_long; 4],
    }

    #[repr(C)]
    pub struct StatVfs {
        pub f_bsize: c_ulong,
        pub f_frsize: c_ulong,
        pub f_blocks: c_ulong,
        pub f_bfree: c_ulong,
        pub f_bavail: c_ulong,
        pub f_files: c_ulong,
        pub f_ffree: c_ulong,
        pub f_favail: c_ulong,
        pub f_fsid: c_ulong,
        pub f_flag: c_ulong,
        pub f_namemax: c_ulong,
        pub f_spare: [c_int; 6],
    }

    unsafe extern "C" {
        pub fn statfs(path: *const c_char, buf: *mut StatFs) -> c_int;
        pub fn statvfs(path: *const c_char, buf: *mut StatVfs) -> c_int;
        pub fn getuid() -> u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        saved: Vec<(&'static str, Option<OsString>)>,
    }

    impl EnvGuard {
        fn new(keys: &[&'static str]) -> Self {
            let saved = keys
                .iter()
                .map(|k| (*k, std::env::var_os(k)))
                .collect::<Vec<_>>();
            for key in keys {
                unsafe { std::env::remove_var(key) };
            }
            Self { saved }
        }

        fn set(&self, key: &str, value: &str) {
            unsafe { std::env::set_var(key, value) };
        }

        fn remove(&self, key: &str) {
            unsafe { std::env::remove_var(key) };
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in self.saved.drain(..) {
                match value {
                    Some(value) => unsafe { std::env::set_var(key, value) },
                    None => unsafe { std::env::remove_var(key) },
                }
            }
        }
    }

    fn fasttmp_env_keys() -> Vec<&'static str> {
        let mut keys = FAST_ENV_KEYS.to_vec();
        keys.extend([
            "RETREAD_FAST_TMP",
            "RETREAD_FAST_TMP_ROOT",
            "RETREAD_FAST_TMP_BUDGET_BYTES",
            "RETREAD_SHARED_CACHE_DIR",
            "RETREAD_FAST_TMP_FORCE_FS",
            "RETREAD_FAST_TMP_PROBE_THRESHOLD_MS",
            "RETREAD_FAST_TMP_CORRUPT_COPYBACK",
            "PIXI_NO_CONFIG",
            "SLURM_JOB_ID",
            "SLURM_MEM_PER_NODE",
            "SLURM_MEM_PER_CPU",
            "SLURM_CPUS_ON_NODE",
        ]);
        keys
    }

    fn tmp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "retread-fasttmp-{label}-{}-{}",
            std::process::id(),
            unique_nonce()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_workspace(dir: &Path) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("pixi.toml"), "[workspace]\nchannels = []\n").unwrap();
    }

    fn retread_envs_target(tmp_root: &Path, workspace: &Path, job: &str) -> PathBuf {
        let canonical = fs::canonicalize(workspace).unwrap();
        let pixi_workspace_component =
            pixi_envs_workspace_component(workspace, &canonical).unwrap();
        tmp_root
            .join(user_namespace_component())
            .join(workspace_hash(&canonical))
            .join(job)
            .join("envs")
            .join(pixi_workspace_component)
            .join("envs")
    }

    fn write_envs_namespace_ownership_proof(target: &Path, workspace: &Path) {
        let namespace_root = target.ancestors().nth(3).unwrap();
        let canonical_workspace = fs::canonicalize(workspace).unwrap();
        fs::create_dir_all(namespace_root.join("bld")).unwrap();
        fs::write(
            namespace_root.join("workspace-path"),
            canonical_workspace.to_string_lossy().as_bytes(),
        )
        .unwrap();
        fs::write(
            namespace_root.join("bld").join(RETREAD_BLD_OWNER_MARKER),
            bld_owner_marker_contents(&canonical_workspace),
        )
        .unwrap();
    }

    fn pair_map(pairs: &[(String, String)]) -> HashMap<String, String> {
        pairs.iter().cloned().collect()
    }

    #[cfg(unix)]
    fn install_fake_pixi(root: &Path, version_command: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let bin = root.join("fake-bin");
        fs::create_dir_all(&bin).unwrap();
        let pixi = bin.join("pixi");
        fs::write(
            &pixi,
            format!(
                "#!/bin/sh\n\
                 if test \"$1\" = --version; then\n\
                   {version_command}\n\
                 elif test \"$1\" = config; then\n\
                   printf '%s\\n' '{{}}'\n\
                 else\n\
                   exit 90\n\
                 fi\n"
            ),
        )
        .unwrap();
        fs::set_permissions(&pixi, fs::Permissions::from_mode(0o755)).unwrap();
        bin
    }

    #[test]
    fn forced_fs_classification() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        guard.set("RETREAD_FAST_TMP_FORCE_FS", "nfs");
        assert_eq!(classify_fs(Path::new("/")), FsClass::Network("nfs"));
        guard.set("RETREAD_FAST_TMP_FORCE_FS", "ext4");
        assert_eq!(classify_fs(Path::new("/")), FsClass::LocalFast);
        guard.set("RETREAD_FAST_TMP_FORCE_FS", "garbage");
        assert_eq!(classify_fs(Path::new("/")), FsClass::Unknown);
    }

    #[test]
    fn config_cascade_env_beats_toml_beats_defaults() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let ws = tmp_dir("config");
        fs::write(
            ws.join("pixi.toml"),
            r#"
[workspace]
channels = []

[tool.retread.fast-tmp]
mode = "off"
tmp-root = "/toml/tmp"
budget-bytes = "123M"
blob-caches = "tmp"
"#,
        )
        .unwrap();
        let cfg = FastTmpConfig::load(&ws);
        assert_eq!(cfg.mode, FastTmpMode::Off);
        assert_eq!(cfg.tmp_root, PathBuf::from("/toml/tmp"));
        assert_eq!(cfg.budget_bytes, Some(123 * 1024 * 1024));
        assert_eq!(cfg.blob_caches, BlobCacheMode::Tmp);

        guard.set("RETREAD_FAST_TMP", "on");
        guard.set("RETREAD_FAST_TMP_ROOT", "/env/tmp");
        guard.set("RETREAD_FAST_TMP_BUDGET_BYTES", "456M");
        let cfg = FastTmpConfig::load(&ws);
        assert_eq!(cfg.mode, FastTmpMode::On);
        assert_eq!(cfg.tmp_root, PathBuf::from("/env/tmp"));
        assert_eq!(cfg.budget_bytes, Some(456 * 1024 * 1024));

        fs::remove_dir_all(ws).ok();
    }

    #[test]
    fn relative_tmp_root_is_resolved_from_invocation_cwd() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let ws = tmp_dir("relative-root");
        write_workspace(&ws);
        guard.set("RETREAD_FAST_TMP_ROOT", "relative-fast-root");
        let cfg = FastTmpConfig::load(&ws);
        assert_eq!(
            cfg.tmp_root,
            std::env::current_dir().unwrap().join("relative-fast-root")
        );
        assert!(cfg.tmp_root.is_absolute());
        fs::remove_dir_all(ws).ok();
    }

    #[test]
    fn pixi_no_config_uses_boolean_semantics() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("no-config-bool");
        let base = root.join("base-config.toml");
        fs::write(&base, "tls-no-verify = false\n").unwrap();
        guard.set("PIXI_CONFIG_FILE", base.to_str().unwrap());
        let ns = Namespace {
            root: root.join("namespace"),
        };
        fs::create_dir_all(&ns.root).unwrap();

        for value in ["0", "false", "no", "off", ""] {
            guard.set("PIXI_NO_CONFIG", value);
            prepare_pixi_config_overlay(&ns).unwrap();
        }
        for value in ["1", "true", "yes", "on", "Y", "t"] {
            guard.set("PIXI_NO_CONFIG", value);
            assert!(
                prepare_pixi_config_overlay(&ns)
                    .unwrap_err()
                    .to_string()
                    .contains("PIXI_NO_CONFIG")
            );
        }
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn pixi_overlay_preserves_effective_security_and_network_config() {
        let effective = br#"{
            "authentication-override-file": "/secure/pixi-auth.json",
            "tls-no-verify": false,
            "tls-root-certs": "system",
            "run-post-link-scripts": "insecure",
            "mirrors": {
                "https://conda.example.invalid": ["https://mirror.example.invalid"]
            },
            "pypi-config": {
                "index-url": "https://pypi.example.invalid/simple"
            },
            "proxy-config": {
                "https": "http://proxy.example.invalid:8080"
            }
        }"#;
        let rendered = render_pixi_config_overlay(effective).unwrap();
        let config: toml::Table = toml::from_str(&rendered).unwrap();

        assert_eq!(
            config
                .get("detached-environments")
                .and_then(toml::Value::as_bool),
            Some(true)
        );
        assert_eq!(
            config
                .get("authentication-override-file")
                .and_then(toml::Value::as_str),
            Some("/secure/pixi-auth.json")
        );
        assert_eq!(
            config
                .get("run-post-link-scripts")
                .and_then(toml::Value::as_str),
            Some("insecure")
        );
        assert_eq!(
            config.get("tls-root-certs").and_then(toml::Value::as_str),
            Some("system")
        );
        assert_eq!(
            config
                .get("pypi-config")
                .and_then(toml::Value::as_table)
                .and_then(|table| table.get("index-url"))
                .and_then(toml::Value::as_str),
            Some("https://pypi.example.invalid/simple")
        );
        assert!(config.contains_key("mirrors"));
        assert!(config.contains_key("proxy-config"));

        let from_override = render_pixi_config_overlay_from_toml(
            "authentication-override-file = \"/private/auth.json\"\n\
             run-post-link-scripts = \"insecure\"\n",
        )
        .unwrap();
        let override_config: toml::Table = toml::from_str(&from_override).unwrap();
        assert_eq!(
            override_config
                .get("authentication-override-file")
                .and_then(toml::Value::as_str),
            Some("/private/auth.json")
        );
        assert_eq!(
            override_config
                .get("detached-environments")
                .and_then(toml::Value::as_bool),
            Some(true)
        );
    }

    #[test]
    fn pixi_version_validation_requires_0_70_or_newer() {
        for supported in ["pixi 0.70.0", "pixi 0.70.2\n", "pixi 0.99.1", "pixi 1.0.0"] {
            validate_pixi_version_text(supported).unwrap();
        }
        for unsupported in ["pixi 0.69.9", "pixi 0.1.0", "pixi version unknown", ""] {
            assert!(
                validate_pixi_version_text(unsupported).is_err(),
                "{unsupported:?}"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn engage_emits_supported_and_legacy_pixi_detached_interfaces() {
        use std::os::unix::fs::PermissionsExt;

        let _lock = ENV_MUTEX.lock().unwrap();
        let mut env_keys = fasttmp_env_keys();
        env_keys.extend(["PATH", "RETREAD_TEST_EXPECTED_CONFIG_CWD"]);
        let guard = EnvGuard::new(&env_keys);
        let root = tmp_dir("pixi-detached");
        let ws = root.join("workspace");
        write_workspace(&ws);
        guard.set("RETREAD_FAST_TMP_FORCE_FS", "nfs");
        guard.set("RETREAD_FAST_TMP_ROOT", root.join("tmp").to_str().unwrap());
        guard.set("RETREAD_FAST_TMP_BUDGET_BYTES", "200G");

        let cfg = FastTmpConfig::load(&ws);
        let expected_ns = namespace(&cfg, &ws);
        let fake_bin = root.join("fake-bin");
        fs::create_dir_all(&fake_bin).unwrap();
        let fake_pixi = fake_bin.join("pixi");
        fs::write(
            &fake_pixi,
            r#"#!/bin/sh
if test "$1" = --version; then
  printf '%s\n' 'pixi 0.70.2'
elif test "$1" = config; then
  test "$#" = 3 || exit 41
  test "$2" = list || exit 42
  test "$3" = --json || exit 43
  test "$PWD" = "$RETREAD_TEST_EXPECTED_CONFIG_CWD" || exit 44
  test -z "$PIXI_CONFIG_FILE" || exit 45
  printf '%s\n' '{"tls-no-verify":false,"run-post-link-scripts":"insecure"}'
else
  exit 60
fi
"#,
        )
        .unwrap();
        fs::set_permissions(&fake_pixi, fs::Permissions::from_mode(0o755)).unwrap();
        guard.set("PATH", fake_bin.to_str().unwrap());
        guard.set(
            "RETREAD_TEST_EXPECTED_CONFIG_CWD",
            expected_ns.root.to_str().unwrap(),
        );

        let engaged = engage(&ws, &cfg).unwrap().unwrap();
        let env = engaged.env.iter().cloned().collect::<HashMap<_, _>>();
        assert_eq!(
            env.get("PIXI_CONFIG_FILE").map(String::as_str),
            engaged.ns.pixi_config_file().to_str()
        );
        assert_eq!(
            env.get("PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR")
                .map(String::as_str),
            engaged.ns.envs_dir().to_str()
        );
        assert_eq!(
            env.get("PIXI_DETACHED_ENVIRONMENTS").map(String::as_str),
            engaged.ns.envs_dir().to_str()
        );
        assert_eq!(
            env.get(RETREAD_BASE_PIXI_CONFIG).map(String::as_str),
            Some("")
        );
        let overlay: toml::Table =
            toml::from_str(&fs::read_to_string(engaged.ns.pixi_config_file()).unwrap()).unwrap();
        assert_eq!(
            overlay
                .get("detached-environments")
                .and_then(toml::Value::as_bool),
            Some(true)
        );
        assert_eq!(
            overlay
                .get("run-post-link-scripts")
                .and_then(toml::Value::as_str),
            Some("insecure")
        );
        assert_eq!(
            fs::metadata(engaged.ns.pixi_config_file())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn stale_shell_cleanup_restores_user_pixi_config() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("stale-shell-cleanup");
        let ws = root.join("workspace");
        write_workspace(&ws);
        guard.set("SLURM_JOB_ID", "old-job");
        guard.set("PIXI_CONFIG_FILE", "/secure/user-pixi-config.toml");
        let ns = Namespace {
            root: root.join("namespace").join("job-old-job"),
        };
        for (key, value) in desired_env_pairs(&FastTmpConfig::default(), &ws, &ns).unwrap() {
            guard.set(&key, &value);
        }

        // The sourced ownership metadata still describes the old job exactly;
        // only the scheduler's current job identity changed.
        guard.set("SLURM_JOB_ID", "new-job");

        let cleanup = shell_stale_cleanup();
        assert!(cleanup.contains("unset PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR\n"));
        assert!(cleanup.contains("unset RETREAD_FAST_TMP_PIXI_CONFIG_FILE\n"));
        assert!(cleanup.contains("unset RETREAD_FAST_TMP_NS_JOB\n"));
        assert!(cleanup.contains("export PIXI_CONFIG_FILE='/secure/user-pixi-config.toml'\n"));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn managed_env_root_transition_replaces_owned_values_and_restores_base() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("managed-root-transition");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let cfg = FastTmpConfig {
            blob_caches: BlobCacheMode::Tmp,
            ..FastTmpConfig::default()
        };
        let ns_a = Namespace {
            root: root.join("root-a").join("job-42"),
        };
        let ns_b = Namespace {
            root: root.join("root-b").join("job-42"),
        };

        let first = desired_env_pairs(&cfg, &ws, &ns_a).unwrap();
        for (key, value) in &first {
            guard.set(key, value);
        }
        let second = desired_env_pairs(&cfg, &ws, &ns_b).unwrap();
        let second_map = pair_map(&second);
        for (key, expected) in [
            ("PIXI_CACHE_DIR", ns_b.rattler_cache_dir()),
            ("RATTLER_CACHE_DIR", ns_b.rattler_cache_dir()),
            ("UV_CACHE_DIR", ns_b.uv_cache_dir()),
            ("RETREAD_CACHE_DIR", ns_b.retread_cache_dir()),
            ("PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR", ns_b.envs_dir()),
            ("PIXI_DETACHED_ENVIRONMENTS", ns_b.envs_dir()),
        ] {
            assert_eq!(
                second_map.get(key).map(String::as_str),
                expected.to_str(),
                "{key} must transition to root B"
            );
        }

        for (key, value) in &second {
            guard.set(key, value);
        }
        let actions = cleanup_env_actions().into_iter().collect::<HashMap<_, _>>();
        for key in [
            "PIXI_CACHE_DIR",
            "RATTLER_CACHE_DIR",
            "UV_CACHE_DIR",
            "RETREAD_CACHE_DIR",
            "PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR",
            "PIXI_DETACHED_ENVIRONMENTS",
        ] {
            assert_eq!(actions.get(key), Some(&None), "{key} base must stay unset");
        }
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn forced_pixi_values_restore_originals_and_unowned_cache_is_preserved() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("managed-base-restore");
        let ws = root.join("workspace");
        write_workspace(&ws);
        guard.set("PIXI_CONFIG_FILE", "/secure/user-pixi.toml");
        guard.set(
            "PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR",
            "/shared/user-detached",
        );
        guard.set("PIXI_DETACHED_ENVIRONMENTS", "/shared/legacy-detached");
        guard.set("UV_CACHE_DIR", "/shared/user-uv");
        let ns = Namespace {
            root: root.join("namespace").join("job-1"),
        };

        let pairs = desired_env_pairs(&FastTmpConfig::default(), &ws, &ns).unwrap();
        assert!(
            !pairs.iter().any(|(key, _)| key == "UV_CACHE_DIR"),
            "an unowned user cache must not be overridden"
        );
        for (key, value) in &pairs {
            guard.set(key, value);
        }
        let actions = cleanup_env_actions().into_iter().collect::<HashMap<_, _>>();
        assert_eq!(
            actions.get("PIXI_CONFIG_FILE"),
            Some(&Some("/secure/user-pixi.toml".to_string()))
        );
        assert_eq!(
            actions.get("PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR"),
            Some(&Some("/shared/user-detached".to_string()))
        );
        assert_eq!(
            actions.get("PIXI_DETACHED_ENVIRONMENTS"),
            Some(&Some("/shared/legacy-detached".to_string()))
        );
        assert!(!actions.contains_key("UV_CACHE_DIR"));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn legacy_same_job_values_are_cleaned_and_not_saved_as_base() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("legacy-cleanup");
        let legacy_ns = root
            .join(user_namespace_component())
            .join("012345abcdef")
            .join("nojob");
        guard.set("RETREAD_FAST_TMP_NS_JOB", "nojob");
        guard.set(
            "UV_CACHE_DIR",
            legacy_ns.join("caches/uv").to_str().unwrap(),
        );
        guard.set(
            "RETREAD_CACHE_DIR",
            legacy_ns.join("caches/retread").to_str().unwrap(),
        );
        guard.set(
            "PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR",
            legacy_ns.join("envs").to_str().unwrap(),
        );
        guard.set("PIXI_CACHE_DIR", "/shared/user-pixi-cache");
        guard.set("RATTLER_CACHE_DIR", "/shared/user-rattler-cache");
        guard.set("UV_LOCK_TIMEOUT", "1800");

        assert!(inherited_fasttmp_cleanup_needed());
        let actions = cleanup_env_actions().into_iter().collect::<HashMap<_, _>>();
        assert_eq!(actions.get("UV_CACHE_DIR"), Some(&None));
        assert_eq!(actions.get("RETREAD_CACHE_DIR"), Some(&None));
        assert_eq!(
            actions.get("PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR"),
            Some(&None)
        );
        assert!(!actions.contains_key("PIXI_CACHE_DIR"));
        assert!(!actions.contains_key("RATTLER_CACHE_DIR"));
        assert_eq!(actions.get("UV_LOCK_TIMEOUT"), Some(&None));

        let ws = root.join("workspace");
        write_workspace(&ws);
        let ns_b = Namespace {
            root: root.join("new-root").join("nojob"),
        };
        let pairs = desired_env_pairs(&FastTmpConfig::default(), &ws, &ns_b).unwrap();
        assert_eq!(
            pair_map(&pairs).get("UV_CACHE_DIR").map(String::as_str),
            ns_b.uv_cache_dir().to_str()
        );
        let base_json = pair_map(&pairs).remove(RETREAD_BASE_ENV_JSON).unwrap();
        let base: HashMap<String, Option<String>> = serde_json::from_str(&base_json).unwrap();
        assert_eq!(base.get("UV_CACHE_DIR"), Some(&None));
        assert_eq!(base.get("RETREAD_CACHE_DIR"), Some(&None));
        assert_eq!(base.get("UV_LOCK_TIMEOUT"), Some(&None));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn reengagement_preserves_changed_user_values_and_reowns_forced_pixi_values() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("managed-user-change");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let ns_a = Namespace {
            root: root.join("root-a").join("nojob"),
        };
        let ns_b = Namespace {
            root: root.join("root-b").join("nojob"),
        };
        for (key, value) in desired_env_pairs(&FastTmpConfig::default(), &ws, &ns_a).unwrap() {
            guard.set(&key, &value);
        }
        guard.set("UV_CACHE_DIR", "/user/changed-after-source");
        guard.set(
            "PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR",
            "/user/changed-detached-after-source",
        );

        let second = desired_env_pairs(&FastTmpConfig::default(), &ws, &ns_b).unwrap();
        let second_map = pair_map(&second);
        assert!(!second_map.contains_key("UV_CACHE_DIR"));
        assert_eq!(
            second_map
                .get("PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR")
                .map(String::as_str),
            ns_b.envs_dir().to_str()
        );
        let base: HashMap<String, Option<String>> =
            serde_json::from_str(second_map.get(RETREAD_BASE_ENV_JSON).unwrap()).unwrap();
        assert!(!base.contains_key("UV_CACHE_DIR"));
        assert_eq!(
            base.get("PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR"),
            Some(&Some("/user/changed-detached-after-source".to_string()))
        );

        for (key, value) in second {
            guard.set(&key, &value);
        }
        let actions = cleanup_env_actions().into_iter().collect::<HashMap<_, _>>();
        assert!(!actions.contains_key("UV_CACHE_DIR"));
        assert_eq!(
            actions.get("PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR"),
            Some(&Some("/user/changed-detached-after-source".to_string()))
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn cleanup_fails_closed_for_missing_or_malformed_ownership_metadata() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("managed-metadata-fail-closed");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let ns = Namespace {
            root: root.join("namespace").join("nojob"),
        };
        let pairs = desired_env_pairs(&FastTmpConfig::default(), &ws, &ns).unwrap();
        for (key, value) in &pairs {
            guard.set(key, value);
        }

        guard.remove(RETREAD_EXPECTED_ENV_JSON);
        assert!(cleanup_env_actions().is_empty());
        for (key, value) in &pairs {
            guard.set(key, value);
        }
        guard.set(RETREAD_BASE_ENV_JSON, "not-json");
        assert!(cleanup_env_actions().is_empty());
        for (key, value) in &pairs {
            guard.set(key, value);
        }
        guard.set(RETREAD_EXPECTED_ENV_JSON, "not-json");
        assert!(cleanup_env_actions().is_empty());
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn cleanup_quotes_base_values_and_rejects_injected_managed_names() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("managed-shell-quoting");
        let ws = root.join("workspace");
        write_workspace(&ws);
        guard.set("PIXI_CONFIG_FILE", "/safe/user's config.toml");
        let ns = Namespace {
            root: root.join("namespace").join("nojob"),
        };
        let pairs = desired_env_pairs(&FastTmpConfig::default(), &ws, &ns).unwrap();
        for (key, value) in &pairs {
            guard.set(key, value);
        }
        let cleanup = shell_stale_cleanup();
        assert!(cleanup.contains("export PIXI_CONFIG_FILE='/safe/user'\\''s config.toml'\n"));

        guard.set(
            RETREAD_MANAGED_KEYS,
            "UV_CACHE_DIR,EVIL_KEY\nexport RETREAD_PWNED=1",
        );
        let cleanup = shell_stale_cleanup();
        assert!(cleanup.is_empty());
        assert!(!cleanup.contains("EVIL_KEY"));
        assert!(!cleanup.contains("RETREAD_PWNED"));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn backend_cleanup_override_suppresses_inherited_legacy_cache() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let legacy = std::env::temp_dir()
            .join(user_namespace_component())
            .join("012345abcdef")
            .join("job-old")
            .join("caches/retread");
        guard.set("RETREAD_FAST_TMP_NS_JOB", "old");
        guard.set("RETREAD_CACHE_DIR", legacy.to_str().unwrap());

        let backend = BACKEND_ENV.get_or_init(|| Mutex::new(BackendEnv::default()));
        let previous = backend.lock().unwrap().clone();
        *backend.lock().unwrap() = BackendEnv {
            pairs: Vec::new(),
            remove_fast_vars: true,
        };
        assert_eq!(
            backend_env_override("RETREAD_CACHE_DIR"),
            BackendEnvOverride::Remove
        );
        *backend.lock().unwrap() = previous;
    }

    #[test]
    fn relative_base_pixi_config_is_made_absolute_before_overlay_cwd_changes() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        guard.set("PIXI_CONFIG_FILE", "relative/config.toml");
        assert_eq!(
            PathBuf::from(base_pixi_config_file().unwrap()),
            std::env::current_dir()
                .unwrap()
                .join("relative/config.toml")
        );
    }

    #[test]
    fn mode_off_short_circuits_slow_detection() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        guard.set("RETREAD_FAST_TMP_FORCE_FS", "nfs");
        let cfg = FastTmpConfig {
            mode: FastTmpMode::Off,
            ..FastTmpConfig::default()
        };
        assert!(!is_slow(Path::new("/definitely/not/needed"), &cfg));
    }

    #[cfg(unix)]
    #[test]
    fn namespace_is_canonical_and_job_scoped() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        guard.set("SLURM_JOB_ID", "1234");
        let root = tmp_dir("namespace");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let alias = root.join("alias");
        std::os::unix::fs::symlink(&ws, &alias).unwrap();
        let cfg = FastTmpConfig {
            tmp_root: root.join("tmp"),
            ..FastTmpConfig::default()
        };
        let a = namespace(&cfg, &ws);
        let b = namespace(&cfg, &alias);
        assert_eq!(a.root, b.root);
        assert!(a.root.ends_with("job-1234"));

        guard.remove("SLURM_JOB_ID");
        let c = namespace(&cfg, &ws);
        assert_ne!(a.root, c.root);
        assert!(c.root.ends_with("nojob"));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn pixi_envs_component_honors_legacy_project_and_package_names() {
        let root = tmp_dir("pixi-envs-name-fallbacks");
        for (directory, manifest, expected_name) in [
            (
                "project",
                "[project]\nname = \"legacy-project\"\nchannels = []\n",
                "legacy-project",
            ),
            (
                "package",
                "[workspace]\nchannels = []\n\n[package]\nname = \"package-name\"\nversion = \"1.0.0\"\n",
                "package-name",
            ),
        ] {
            let workspace = root.join(directory);
            fs::create_dir_all(&workspace).unwrap();
            fs::write(workspace.join("pixi.toml"), manifest).unwrap();
            let canonical = fs::canonicalize(&workspace).unwrap();

            assert_eq!(
                pixi_envs_workspace_component(&workspace, &canonical).unwrap(),
                format!(
                    "{expected_name}-{}",
                    xxh3_64(canonical.to_string_lossy().as_bytes())
                )
            );
        }
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn probe_failure_is_not_slow() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        guard.set("RETREAD_FAST_TMP_FORCE_FS", "garbage");
        let root = tmp_dir("probe-failure");
        let file = root.join("not-a-dir");
        fs::write(&file, "x").unwrap();
        let cfg = FastTmpConfig::default();
        assert!(!is_slow(&file, &cfg));
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn engage_is_idempotent_and_respects_existing_env() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("engage");
        let ws = root.join("workspace");
        write_workspace(&ws);
        guard.set("RETREAD_FAST_TMP_FORCE_FS", "nfs");
        guard.set("RETREAD_FAST_TMP_ROOT", root.join("tmp").to_str().unwrap());
        guard.set("RETREAD_FAST_TMP_BUDGET_BYTES", "200G");
        guard.set("UV_CACHE_DIR", "/preexisting/uv");
        let cfg = FastTmpConfig::load(&ws);
        let first = engage(&ws, &cfg).unwrap().unwrap();
        let second = engage(&ws, &cfg).unwrap().unwrap();
        assert_eq!(first.ns.root, second.ns.root);
        assert_eq!(
            fs::read_link(ws.join(".pixi").join("bld")).unwrap(),
            first.ns.bld_dir()
        );
        assert_eq!(
            fs::read(first.ns.bld_dir().join(RETREAD_BLD_OWNER_MARKER)).unwrap(),
            bld_owner_marker_contents(&fs::canonicalize(&ws).unwrap())
        );
        assert!(
            !first.env.iter().any(|(k, _)| k == "UV_CACHE_DIR"),
            "pre-set UV_CACHE_DIR must not be overridden"
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn setup_auto_heals_stale_real_bld_directory() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let root = tmp_dir("real-bld");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let link = ws.join(".pixi/bld");
        fs::create_dir_all(&link).unwrap();
        let stale_file = link.join("stale-scratch");
        fs::write(&stale_file, b"regenerable").unwrap();
        let stale_time = SystemTime::now()
            .checked_sub(BLD_IDLE_THRESHOLD + Duration::from_secs(5))
            .unwrap();
        File::options()
            .write(true)
            .open(&stale_file)
            .unwrap()
            .set_times(fs::FileTimes::new().set_modified(stale_time))
            .unwrap();
        let ns = Namespace {
            root: root.join("new-namespace").join("nojob"),
        };
        fs::create_dir_all(ns.bld_dir()).unwrap();

        setup_bld_symlink(&ws, &ns).unwrap();

        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read_link(&link).unwrap(), ns.bld_dir());
        assert!(!link.join("stale-scratch").exists());
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn setup_refuses_recent_unmarked_real_bld_directory_without_deleting_it() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let root = tmp_dir("recent-foreign-bld");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let link = ws.join(".pixi/bld");
        fs::create_dir_all(&link).unwrap();
        fs::write(link.join("keep"), b"possibly live foreign build").unwrap();
        let ns = Namespace {
            root: root.join("new-namespace").join("nojob"),
        };
        fs::create_dir_all(ns.bld_dir()).unwrap();

        let error = setup_bld_symlink(&ws, &ns).unwrap_err().to_string();

        assert!(error.contains("may belong to a live Pixi/non-Retread build"));
        assert!(fs::symlink_metadata(&link).unwrap().is_dir());
        assert_eq!(
            fs::read(link.join("keep")).unwrap(),
            b"possibly live foreign build"
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn setup_auto_heals_empty_real_bld_directory() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let root = tmp_dir("empty-real-bld");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let link = ws.join(".pixi/bld");
        fs::create_dir_all(&link).unwrap();
        let ns = Namespace {
            root: root.join("new-namespace").join("nojob"),
        };
        fs::create_dir_all(ns.bld_dir()).unwrap();

        setup_bld_symlink(&ws, &ns).unwrap();

        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read_link(&link).unwrap(), ns.bld_dir());
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn setup_auto_heals_recent_retread_marked_real_bld_directory() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let root = tmp_dir("marked-real-bld");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let link = ws.join(".pixi/bld");
        fs::create_dir_all(&link).unwrap();
        fs::write(link.join("scratch"), b"recent but Retread-owned").unwrap();
        fs::write(
            link.join(RETREAD_BLD_OWNER_MARKER),
            bld_owner_marker_contents(&fs::canonicalize(&ws).unwrap()),
        )
        .unwrap();
        let ns = Namespace {
            root: root.join("new-namespace").join("nojob"),
        };
        fs::create_dir_all(ns.bld_dir()).unwrap();

        setup_bld_symlink(&ws, &ns).unwrap();

        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read_link(&link).unwrap(), ns.bld_dir());
        assert!(!link.join("scratch").exists());
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn setup_rejects_stale_bld_directory_that_contains_target() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let root = tmp_dir("overlapping-bld");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let link = ws.join(".pixi/bld");
        fs::create_dir_all(&link).unwrap();
        let ns = Namespace {
            root: link.join("namespace"),
        };
        fs::create_dir_all(ns.bld_dir()).unwrap();
        fs::write(ns.bld_dir().join("keep"), b"live target").unwrap();

        let error = setup_bld_symlink(&ws, &ns).unwrap_err().to_string();

        assert!(error.contains("overlaps"));
        assert!(fs::symlink_metadata(&link).unwrap().is_dir());
        assert_eq!(fs::read(ns.bld_dir().join("keep")).unwrap(), b"live target");
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn setup_keeps_correct_bld_symlink_without_mutation() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let root = tmp_dir("correct-bld");
        let ws = root.join("workspace");
        write_workspace(&ws);
        fs::create_dir_all(ws.join(".pixi")).unwrap();
        let ns = Namespace {
            root: root.join("namespace").join("nojob"),
        };
        fs::create_dir_all(ns.bld_dir()).unwrap();
        let link = ws.join(".pixi/bld");
        std::os::unix::fs::symlink(ns.bld_dir(), &link).unwrap();

        setup_bld_symlink_with_hook(&ws, &ns, || {
            panic!("correct symlink must be a no-op");
        })
        .unwrap();

        assert_eq!(fs::read_link(&link).unwrap(), ns.bld_dir());
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn setup_auto_heals_dangling_bld_symlink() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let root = tmp_dir("dangling-bld");
        let ws = root.join("workspace");
        write_workspace(&ws);
        fs::create_dir_all(ws.join(".pixi")).unwrap();
        let dead_target = root.join("missing-old-job/bld");
        let link = ws.join(".pixi/bld");
        std::os::unix::fs::symlink(&dead_target, &link).unwrap();
        let ns = Namespace {
            root: root.join("new-namespace").join("nojob"),
        };
        fs::create_dir_all(ns.bld_dir()).unwrap();

        setup_bld_symlink(&ws, &ns).unwrap();

        assert_eq!(fs::read_link(&link).unwrap(), ns.bld_dir());
        assert!(!dead_target.exists());
        fs::remove_dir_all(root).ok();
    }

    /// Guard for the 4.10.82 fix: a LIVE envs symlink must be verifiable
    /// with reads alone. Opening the link lock is a workspace WRITE, and on
    /// a quota-blocked filesystem it failed with EDQUOT even when there was
    /// nothing to heal -- taking down backend startup for every environment
    /// while `pixi install` still exited 0 with an empty env.
    #[cfg(target_os = "linux")]
    #[test]
    fn backend_heal_needs_no_workspace_write_when_envs_symlink_is_live() {
        use std::os::unix::fs::PermissionsExt as _;

        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("readonly-live-envs");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let tmp_root = root.join("tmp");
        let target = retread_envs_target(&tmp_root, &ws, "job-4305035");
        fs::create_dir_all(&target).unwrap();
        std::os::unix::fs::symlink(&target, pixi.join("envs")).unwrap();
        guard.set("SLURM_JOB_ID", "4305035");
        guard.set("RETREAD_FAST_TMP", "on");
        guard.set("RETREAD_FAST_TMP_ROOT", tmp_root.to_str().unwrap());

        // Read-only .pixi: any attempt to create the lock file would EACCES.
        fs::set_permissions(&pixi, fs::Permissions::from_mode(0o555)).unwrap();
        let result = heal_stale_envs_symlink_at_backend_startup(&ws);
        fs::set_permissions(&pixi, fs::Permissions::from_mode(0o755)).unwrap();
        result.expect("live envs symlink must not require workspace writes");
        assert!(
            !pixi.join(FAST_WORKSPACE_LINK_LOCK).exists(),
            "no lock file may be created on the no-repair path",
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn backend_heals_dangling_retread_envs_symlink_to_current_job() {
        use std::os::unix::fs::MetadataExt as _;

        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("dangling-envs");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let tmp_root = root.join("tmp");
        let target = retread_envs_target(&tmp_root, &ws, "job-4305035");
        let link = pixi.join("envs");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let inode_before = fs::symlink_metadata(&link).unwrap().ino();
        guard.set("SLURM_JOB_ID", "4305035");
        guard.set("RETREAD_FAST_TMP", "on");
        guard.set("RETREAD_FAST_TMP_ROOT", tmp_root.to_str().unwrap());

        heal_stale_envs_symlink_at_backend_startup(&ws).unwrap();

        assert_eq!(fs::read_link(&link).unwrap(), target);
        assert_ne!(fs::symlink_metadata(&link).unwrap().ino(), inode_before);
        assert!(target.is_dir());
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn backend_preserves_live_foreign_job_retread_envs_symlink_and_warns() {
        use std::cell::RefCell;
        use std::os::unix::fs::MetadataExt as _;

        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("foreign-job-envs");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let tmp_root = root.join("tmp");
        let old_target = retread_envs_target(&tmp_root, &ws, "job-4305034");
        let new_target = retread_envs_target(&tmp_root, &ws, "job-4305035");
        fs::create_dir_all(&old_target).unwrap();
        fs::write(old_target.join("keep"), b"old job environment").unwrap();
        write_envs_namespace_ownership_proof(&old_target, &ws);
        let link = pixi.join("envs");
        std::os::unix::fs::symlink(&old_target, &link).unwrap();
        let inode_before = fs::symlink_metadata(&link).unwrap().ino();
        guard.set("SLURM_JOB_ID", "4305035");
        guard.set("RETREAD_FAST_TMP", "on");
        guard.set("RETREAD_FAST_TMP_ROOT", tmp_root.to_str().unwrap());
        let cfg = FastTmpConfig::load(&ws);
        let current_ns = namespace(&cfg, &ws);
        let warnings = RefCell::new(Vec::new());

        heal_stale_envs_symlink_with_warning(&ws, &cfg.tmp_root, Some(&current_ns), |message| {
            warnings.borrow_mut().push(message.to_string())
        })
        .unwrap();

        assert_eq!(fs::read_link(&link).unwrap(), old_target);
        assert_eq!(fs::symlink_metadata(&link).unwrap().ino(), inode_before);
        assert!(!new_target.exists());
        assert_eq!(
            fs::read(old_target.join("keep")).unwrap(),
            b"old job environment"
        );
        let warnings = warnings.into_inner();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        let warning = &warnings[0];
        assert!(warning.contains("LIVE JOB COLLISION"), "{warning}");
        assert!(
            warning.contains("two live jobs share workspace"),
            "{warning}"
        );
        assert!(warning.contains("4305034"), "{warning}");
        assert!(warning.contains("4305035"), "{warning}");
        assert!(warning.contains("per-job workspace"), "{warning}");
        assert!(warning.contains("disable fast-tmp"), "{warning}");
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn backend_heals_dangling_foreign_job_retread_envs_symlink() {
        use std::os::unix::fs::MetadataExt as _;

        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("dangling-foreign-job-envs");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let tmp_root = root.join("tmp");
        let old_target = retread_envs_target(&tmp_root, &ws, "job-4305034");
        let new_target = retread_envs_target(&tmp_root, &ws, "job-4305035");
        let link = pixi.join("envs");
        std::os::unix::fs::symlink(&old_target, &link).unwrap();
        let inode_before = fs::symlink_metadata(&link).unwrap().ino();
        guard.set("SLURM_JOB_ID", "4305035");
        guard.set("RETREAD_FAST_TMP", "on");
        guard.set("RETREAD_FAST_TMP_ROOT", tmp_root.to_str().unwrap());

        heal_stale_envs_symlink_at_backend_startup(&ws).unwrap();

        assert_eq!(fs::read_link(&link).unwrap(), new_target);
        assert_ne!(fs::symlink_metadata(&link).unwrap().ino(), inode_before);
        assert!(new_target.is_dir());
        assert!(!old_target.exists());
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn healed_namespace_proves_later_live_foreign_job_collision() {
        use std::cell::RefCell;

        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("healed-live-foreign-proof");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let tmp_root = root.join("tmp");
        let old_target = retread_envs_target(&tmp_root, &ws, "job-4305034");
        let healed_target = retread_envs_target(&tmp_root, &ws, "job-4305035");
        let link = pixi.join("envs");
        std::os::unix::fs::symlink(&old_target, &link).unwrap();
        guard.set("SLURM_JOB_ID", "4305035");
        guard.set("RETREAD_FAST_TMP", "on");
        guard.set("RETREAD_FAST_TMP_ROOT", tmp_root.to_str().unwrap());

        heal_stale_envs_symlink_at_backend_startup(&ws).unwrap();
        assert_eq!(fs::read_link(&link).unwrap(), healed_target);

        guard.set("SLURM_JOB_ID", "4305036");
        let cfg = FastTmpConfig::load(&ws);
        let next_ns = namespace(&cfg, &ws);
        let warnings = RefCell::new(Vec::new());
        heal_stale_envs_symlink_with_warning(&ws, &cfg.tmp_root, Some(&next_ns), |message| {
            warnings.borrow_mut().push(message.to_string())
        })
        .unwrap();

        assert_eq!(fs::read_link(&link).unwrap(), healed_target);
        let warnings = warnings.into_inner();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains("LIVE JOB COLLISION"),
            "{}",
            warnings[0]
        );
        assert!(warnings[0].contains("4305035"), "{}", warnings[0]);
        assert!(warnings[0].contains("4305036"), "{}", warnings[0]);
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn backend_keeps_correct_retread_envs_symlink_untouched() {
        use std::os::unix::fs::MetadataExt as _;

        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("correct-envs");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let tmp_root = root.join("tmp");
        let target = retread_envs_target(&tmp_root, &ws, "job-4305035");
        fs::create_dir_all(&target).unwrap();
        write_envs_namespace_ownership_proof(&target, &ws);
        let link = pixi.join("envs");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let inode_before = fs::symlink_metadata(&link).unwrap().ino();
        guard.set("SLURM_JOB_ID", "4305035");
        guard.set("RETREAD_FAST_TMP", "on");
        guard.set("RETREAD_FAST_TMP_ROOT", tmp_root.to_str().unwrap());

        heal_stale_envs_symlink_at_backend_startup(&ws).unwrap();

        assert_eq!(fs::read_link(&link).unwrap(), target);
        assert_eq!(fs::symlink_metadata(&link).unwrap().ino(), inode_before);
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn backend_preserves_non_retread_user_envs_symlink() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("user-envs-link");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let user_target = root.join("missing-user-envs");
        let link = pixi.join("envs");
        std::os::unix::fs::symlink(&user_target, &link).unwrap();
        guard.set("SLURM_JOB_ID", "4305035");
        guard.set("RETREAD_FAST_TMP", "on");
        guard.set("RETREAD_FAST_TMP_ROOT", root.join("tmp").to_str().unwrap());

        heal_stale_envs_symlink_at_backend_startup(&ws).unwrap();

        assert_eq!(fs::read_link(&link).unwrap(), user_target);
        assert!(!user_target.exists());
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn backend_preserves_prefix_shaped_unowned_envs_symlink() {
        use std::os::unix::fs::MetadataExt as _;

        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("prefix-shaped-user-envs");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let rogue_tmp_root = root.join("rogue-tmp");
        let configured_tmp_root = root.join("managed-tmp");
        let user_target = retread_envs_target(&rogue_tmp_root, &ws, "job-4305034");
        let managed_target = retread_envs_target(&configured_tmp_root, &ws, "job-4305035");
        write_envs_namespace_ownership_proof(&user_target, &ws);
        let link = pixi.join("envs");
        std::os::unix::fs::symlink(&user_target, &link).unwrap();
        let inode_before = fs::symlink_metadata(&link).unwrap().ino();
        guard.set("SLURM_JOB_ID", "4305035");
        guard.set("RETREAD_FAST_TMP", "on");
        guard.set(
            "RETREAD_FAST_TMP_ROOT",
            configured_tmp_root.to_str().unwrap(),
        );

        heal_stale_envs_symlink_at_backend_startup(&ws).unwrap();

        assert_eq!(fs::read_link(&link).unwrap(), user_target);
        assert_eq!(fs::symlink_metadata(&link).unwrap().ino(), inode_before);
        assert!(!user_target.exists());
        assert!(!managed_target.exists());
        assert!(!configured_tmp_root.exists());
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn backend_preserves_forged_pixi_workspace_hash_envs_symlink() {
        use std::os::unix::fs::MetadataExt as _;

        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("forged-pixi-workspace-hash");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let configured_tmp_root = root.join("managed-tmp");
        let canonical_workspace = fs::canonicalize(&ws).unwrap();
        let user_target = configured_tmp_root
            .join(user_namespace_component())
            .join(workspace_hash(&canonical_workspace))
            .join("job-4305034")
            .join("envs")
            .join("workspace-1")
            .join("envs");
        let link = pixi.join("envs");
        std::os::unix::fs::symlink(&user_target, &link).unwrap();
        let inode_before = fs::symlink_metadata(&link).unwrap().ino();
        guard.set("SLURM_JOB_ID", "4305035");
        guard.set("RETREAD_FAST_TMP", "on");
        guard.set(
            "RETREAD_FAST_TMP_ROOT",
            configured_tmp_root.to_str().unwrap(),
        );

        heal_stale_envs_symlink_at_backend_startup(&ws).unwrap();

        assert_eq!(fs::read_link(&link).unwrap(), user_target);
        assert_eq!(fs::symlink_metadata(&link).unwrap().ino(), inode_before);
        assert!(!configured_tmp_root.exists());
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn backend_preserves_real_envs_directory() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("real-envs-backend");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let envs = ws.join(".pixi/envs");
        fs::create_dir_all(&envs).unwrap();
        fs::write(envs.join("keep"), b"user environment").unwrap();
        guard.set("SLURM_JOB_ID", "4305035");
        guard.set("RETREAD_FAST_TMP", "on");
        guard.set("RETREAD_FAST_TMP_ROOT", root.join("tmp").to_str().unwrap());

        heal_stale_envs_symlink_at_backend_startup(&ws).unwrap();

        assert!(fs::symlink_metadata(&envs).unwrap().is_dir());
        assert_eq!(fs::read(envs.join("keep")).unwrap(), b"user environment");
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn backend_fast_tmp_off_removes_dangling_retread_envs_symlink() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("off-dangling-envs");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let target = retread_envs_target(&root.join("tmp"), &ws, "job-4305034");
        let link = pixi.join("envs");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        guard.set("SLURM_JOB_ID", "4305035");
        guard.set("RETREAD_FAST_TMP", "off");
        guard.set("RETREAD_FAST_TMP_ROOT", root.join("tmp").to_str().unwrap());

        heal_stale_envs_symlink_at_backend_startup(&ws).unwrap();

        assert!(fs::symlink_metadata(&link).is_err());
        assert!(!target.exists());
        assert!(
            fs::read_dir(&pixi)
                .unwrap()
                .filter_map(Result::ok)
                .all(|entry| !entry
                    .file_name()
                    .to_string_lossy()
                    .contains("retread-quarantine"))
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn envs_replacement_renames_prepared_symlink_over_existing_link() {
        use std::cell::Cell;
        use std::os::unix::fs::MetadataExt as _;

        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("atomic-envs-replace");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let tmp_root = root.join("tmp");
        let old_target = retread_envs_target(&tmp_root, &ws, "job-4305034");
        let new_target = retread_envs_target(&tmp_root, &ws, "job-4305035");
        fs::create_dir_all(&new_target).unwrap();
        let link = pixi.join("envs");
        std::os::unix::fs::symlink(&old_target, &link).unwrap();
        let inode_before = fs::symlink_metadata(&link).unwrap().ino();
        guard.set("SLURM_JOB_ID", "4305035");
        let owned = workspace_envs_link_target(&ws, &tmp_root, &old_target, true)
            .unwrap()
            .unwrap();
        let expected = ExpectedEnvsSymlink {
            raw_target: old_target.clone(),
            owned,
            current_job_component: current_job_component(),
            reason: StaleEnvsSymlinkReason::ForeignJob,
        };
        let rename_observed = Cell::new(false);
        let _link_lock = open_and_lock(&pixi.join(FAST_WORKSPACE_LINK_LOCK)).unwrap();
        let directory = AnchoredEnvsDirectory::open(&pixi).unwrap();

        atomic_replace_stale_envs_symlink_with_hook(
            &directory,
            OsStr::new("envs"),
            &ws,
            &tmp_root,
            &link,
            &new_target,
            &expected,
            |prepared| {
                assert_eq!(fs::read_link(&link).unwrap(), old_target);
                assert_eq!(fs::read_link(prepared).unwrap(), new_target);
                assert!(
                    prepared
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .starts_with(".envs.retread-tmp.")
                );
                rename_observed.set(true);
            },
        )
        .unwrap();

        assert!(rename_observed.get());
        assert_eq!(fs::read_link(&link).unwrap(), new_target);
        assert_ne!(fs::symlink_metadata(&link).unwrap().ino(), inode_before);
        assert!(
            fs::read_dir(&pixi)
                .unwrap()
                .filter_map(Result::ok)
                .all(|entry| !entry.file_name().to_string_lossy().contains("retread-tmp"))
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn envs_rename_stays_anchored_when_pixi_directory_moves() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("envs-pixi-directory-race");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let displaced_pixi = ws.join(".pixi-displaced");
        let tmp_root = root.join("tmp");
        let old_target = retread_envs_target(&tmp_root, &ws, "job-4305034");
        let new_target = retread_envs_target(&tmp_root, &ws, "job-4305035");
        fs::create_dir_all(&new_target).unwrap();
        let user_target = root.join("user-envs");
        let link = pixi.join("envs");
        std::os::unix::fs::symlink(&old_target, &link).unwrap();
        guard.set("SLURM_JOB_ID", "4305035");
        let owned = workspace_envs_link_target(&ws, &tmp_root, &old_target, true)
            .unwrap()
            .unwrap();
        let expected = ExpectedEnvsSymlink {
            raw_target: old_target,
            owned,
            current_job_component: current_job_component(),
            reason: StaleEnvsSymlinkReason::ForeignJob,
        };
        let _link_lock = open_and_lock(&pixi.join(FAST_WORKSPACE_LINK_LOCK)).unwrap();
        let directory = AnchoredEnvsDirectory::open(&pixi).unwrap();

        atomic_replace_stale_envs_symlink_with_hook(
            &directory,
            OsStr::new("envs"),
            &ws,
            &tmp_root,
            &link,
            &new_target,
            &expected,
            |_| {
                fs::rename(&pixi, &displaced_pixi).unwrap();
                fs::create_dir_all(&pixi).unwrap();
                std::os::unix::fs::symlink(&user_target, &link).unwrap();
            },
        )
        .unwrap();

        assert_eq!(fs::read_link(&link).unwrap(), user_target);
        assert_eq!(
            fs::read_link(displaced_pixi.join("envs")).unwrap(),
            new_target
        );
        assert!(
            fs::read_dir(&displaced_pixi)
                .unwrap()
                .filter_map(Result::ok)
                .all(|entry| !entry.file_name().to_string_lossy().contains("retread-tmp"))
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn envs_rename_refuses_concurrently_installed_user_symlink() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("envs-user-link-race");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let tmp_root = root.join("tmp");
        let old_target = retread_envs_target(&tmp_root, &ws, "job-4305034");
        let new_target = retread_envs_target(&tmp_root, &ws, "job-4305035");
        fs::create_dir_all(&new_target).unwrap();
        let user_target = root.join("user-envs");
        let link = pixi.join("envs");
        std::os::unix::fs::symlink(&old_target, &link).unwrap();
        guard.set("SLURM_JOB_ID", "4305035");
        let owned = workspace_envs_link_target(&ws, &tmp_root, &old_target, true)
            .unwrap()
            .unwrap();
        let expected = ExpectedEnvsSymlink {
            raw_target: old_target,
            owned,
            current_job_component: current_job_component(),
            reason: StaleEnvsSymlinkReason::ForeignJob,
        };
        let _link_lock = open_and_lock(&pixi.join(FAST_WORKSPACE_LINK_LOCK)).unwrap();
        let directory = AnchoredEnvsDirectory::open(&pixi).unwrap();

        let error = atomic_replace_stale_envs_symlink_with_hook(
            &directory,
            OsStr::new("envs"),
            &ws,
            &tmp_root,
            &link,
            &new_target,
            &expected,
            |_| {
                fs::remove_file(&link).unwrap();
                std::os::unix::fs::symlink(&user_target, &link).unwrap();
            },
        )
        .unwrap_err();
        let error = format!("{error:#}");

        assert!(
            error.contains("changed or became live concurrently"),
            "{error}"
        );
        assert_eq!(fs::read_link(&link).unwrap(), user_target);
        assert!(
            fs::read_dir(&pixi)
                .unwrap()
                .filter_map(Result::ok)
                .all(|entry| !entry.file_name().to_string_lossy().contains("retread-tmp"))
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn envs_rename_refuses_replaced_prepared_symlink() {
        use std::cell::RefCell;

        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("envs-prepared-link-race");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let tmp_root = root.join("tmp");
        let old_target = retread_envs_target(&tmp_root, &ws, "job-4305034");
        let new_target = retread_envs_target(&tmp_root, &ws, "job-4305035");
        fs::create_dir_all(&new_target).unwrap();
        let user_target = root.join("user-prepared-envs");
        let displaced_prepared = pixi.join(".envs.displaced-prepared");
        let link = pixi.join("envs");
        std::os::unix::fs::symlink(&old_target, &link).unwrap();
        guard.set("SLURM_JOB_ID", "4305035");
        let owned = workspace_envs_link_target(&ws, &tmp_root, &old_target, true)
            .unwrap()
            .unwrap();
        let expected = ExpectedEnvsSymlink {
            raw_target: old_target.clone(),
            owned,
            current_job_component: current_job_component(),
            reason: StaleEnvsSymlinkReason::ForeignJob,
        };
        let _link_lock = open_and_lock(&pixi.join(FAST_WORKSPACE_LINK_LOCK)).unwrap();
        let directory = AnchoredEnvsDirectory::open(&pixi).unwrap();
        let prepared_path = RefCell::new(None);

        let error = atomic_replace_stale_envs_symlink_with_hook(
            &directory,
            OsStr::new("envs"),
            &ws,
            &tmp_root,
            &link,
            &new_target,
            &expected,
            |prepared| {
                *prepared_path.borrow_mut() = Some(prepared.to_path_buf());
                fs::rename(prepared, &displaced_prepared).unwrap();
                std::os::unix::fs::symlink(&user_target, prepared).unwrap();
            },
        )
        .unwrap_err();
        let error = format!("{error:#}");

        assert!(error.contains("preserved the newer path"), "{error}");
        assert_eq!(fs::read_link(&link).unwrap(), old_target);
        let prepared_path = prepared_path.into_inner().unwrap();
        assert_eq!(fs::read_link(&prepared_path).unwrap(), user_target);
        assert_eq!(fs::read_link(displaced_prepared).unwrap(), new_target);
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn envs_rename_never_repoints_foreign_link_if_target_becomes_live() {
        use std::os::unix::fs::MetadataExt as _;

        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("envs-foreign-target-race");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let tmp_root = root.join("tmp");
        let old_target = retread_envs_target(&tmp_root, &ws, "job-4305034");
        let new_target = retread_envs_target(&tmp_root, &ws, "job-4305035");
        fs::create_dir_all(&new_target).unwrap();
        let link = pixi.join("envs");
        std::os::unix::fs::symlink(&old_target, &link).unwrap();
        let inode_before = fs::symlink_metadata(&link).unwrap().ino();
        guard.set("SLURM_JOB_ID", "4305035");
        let owned = workspace_envs_link_target(&ws, &tmp_root, &old_target, true)
            .unwrap()
            .unwrap();
        let expected = ExpectedEnvsSymlink {
            raw_target: old_target.clone(),
            owned,
            current_job_component: current_job_component(),
            reason: StaleEnvsSymlinkReason::ForeignJob,
        };
        let _link_lock = open_and_lock(&pixi.join(FAST_WORKSPACE_LINK_LOCK)).unwrap();
        let directory = AnchoredEnvsDirectory::open(&pixi).unwrap();

        let error = atomic_replace_stale_envs_symlink_with_hook(
            &directory,
            OsStr::new("envs"),
            &ws,
            &tmp_root,
            &link,
            &new_target,
            &expected,
            |_| fs::create_dir_all(&old_target).unwrap(),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("became live concurrently"), "{error}");
        assert_eq!(fs::read_link(&link).unwrap(), old_target);
        assert_eq!(fs::symlink_metadata(&link).unwrap().ino(), inode_before);
        assert!(old_target.is_dir());
        assert!(
            fs::read_dir(&pixi)
                .unwrap()
                .filter_map(Result::ok)
                .all(|entry| !entry.file_name().to_string_lossy().contains("retread-tmp"))
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn backend_envs_heal_serializes_two_ranks() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("concurrent-envs");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let tmp_root = root.join("tmp");
        let old_target = retread_envs_target(&tmp_root, &ws, "job-4305034");
        let new_target = retread_envs_target(&tmp_root, &ws, "job-4305035");
        let link = pixi.join("envs");
        std::os::unix::fs::symlink(&old_target, &link).unwrap();
        guard.set("SLURM_JOB_ID", "4305035");
        guard.set("RETREAD_FAST_TMP", "on");
        guard.set("RETREAD_FAST_TMP_ROOT", tmp_root.to_str().unwrap());
        let start = std::sync::Arc::new(std::sync::Barrier::new(3));

        std::thread::scope(|scope| {
            let first_start = start.clone();
            let first_ws = ws.clone();
            let first = scope.spawn(move || {
                first_start.wait();
                heal_stale_envs_symlink_at_backend_startup(&first_ws)
            });
            let second_start = start.clone();
            let second_ws = ws.clone();
            let second = scope.spawn(move || {
                second_start.wait();
                heal_stale_envs_symlink_at_backend_startup(&second_ws)
            });
            start.wait();
            first.join().unwrap().unwrap();
            second.join().unwrap().unwrap();
        });

        assert_eq!(fs::read_link(&link).unwrap(), new_target);
        assert!(new_target.is_dir());
        assert!(
            fs::read_dir(&pixi)
                .unwrap()
                .filter_map(Result::ok)
                .all(|entry| {
                    let name = entry.file_name();
                    let name = name.to_string_lossy();
                    !name.contains("retread-quarantine") && !name.contains("retread-tmp")
                })
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn setup_refuses_to_heal_while_workspace_lock_is_held() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let root = tmp_dir("locked-bld");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let link = ws.join(".pixi/bld");
        fs::create_dir_all(&link).unwrap();
        fs::write(link.join("keep"), b"live scratch").unwrap();
        let ns = Namespace {
            root: root.join("new-namespace").join("nojob"),
        };
        fs::create_dir_all(ns.bld_dir()).unwrap();
        let concurrent_lock =
            open_and_lock(&ws.join(".pixi").join(FAST_WORKSPACE_LINK_LOCK)).unwrap();

        let error = setup_bld_symlink(&ws, &ns).unwrap_err().to_string();

        assert!(error.contains("another fast-tmp build holds"));
        assert!(fs::symlink_metadata(&link).unwrap().is_dir());
        assert_eq!(fs::read(link.join("keep")).unwrap(), b"live scratch");
        drop(concurrent_lock);
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn root_transition_replaces_only_proven_retread_links() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("bld-root-transition");
        let ws = root.join("workspace");
        write_workspace(&ws);
        fs::create_dir_all(ws.join(".pixi")).unwrap();

        let old_ns = Namespace {
            root: root.join("old-root").join("job-1234"),
        };
        let new_ns = Namespace {
            root: root.join("new-root").join("job-1234"),
        };
        fs::create_dir_all(&old_ns.root).unwrap();
        fs::write(
            old_ns.root.join("workspace-path"),
            fs::canonicalize(&ws).unwrap().to_string_lossy().as_bytes(),
        )
        .unwrap();
        let old_target = old_ns.bld_dir();
        let new_target = new_ns.bld_dir();
        fs::create_dir_all(&old_target).unwrap();
        fs::create_dir_all(&new_target).unwrap();
        let link = ws.join(".pixi").join("bld");
        std::os::unix::fs::symlink(&old_target, &link).unwrap();

        setup_bld_symlink(&ws, &new_ns).unwrap();
        assert_eq!(fs::read_link(&link).unwrap(), new_target);
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn disengage_removes_owned_links_but_preserves_unowned_links() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("link-cleanup");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let owned_ns = root.join("owned").join("job-old");
        fs::create_dir_all(owned_ns.join("envs")).unwrap();
        fs::create_dir_all(owned_ns.join("bld")).unwrap();
        fs::write(
            owned_ns.join("workspace-path"),
            fs::canonicalize(&ws).unwrap().to_string_lossy().as_bytes(),
        )
        .unwrap();
        let envs_target = owned_ns.join("envs");
        std::os::unix::fs::symlink(&envs_target, pixi.join("envs")).unwrap();
        std::os::unix::fs::symlink(owned_ns.join("bld"), pixi.join("bld")).unwrap();

        let cfg = FastTmpConfig {
            mode: FastTmpMode::Off,
            ..FastTmpConfig::default()
        };
        assert!(engage(&ws, &cfg).unwrap().is_none());
        assert_eq!(fs::read_link(pixi.join("envs")).unwrap(), envs_target);
        assert!(fs::symlink_metadata(pixi.join("bld")).is_err());

        let user_bld = root.join("user-bld");
        std::os::unix::fs::symlink(&user_bld, pixi.join("bld")).unwrap();
        assert!(engage(&ws, &cfg).unwrap().is_none());
        assert_eq!(fs::read_link(pixi.join("envs")).unwrap(), envs_target);
        assert_eq!(fs::read_link(pixi.join("bld")).unwrap(), user_bld);
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn unowned_symlinks_are_rejected_and_dangling_legacy_links_are_owned() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("link-ownership");
        let ws = root.join("workspace");
        write_workspace(&ws);
        fs::create_dir_all(ws.join(".pixi")).unwrap();
        let link = ws.join(".pixi/bld");
        let user_target = root.join("unrelated-user-bld");
        fs::create_dir_all(&user_target).unwrap();
        std::os::unix::fs::symlink(&user_target, &link).unwrap();
        let ns = Namespace {
            root: root.join("new-namespace").join("nojob"),
        };
        fs::create_dir_all(ns.bld_dir()).unwrap();
        let error = setup_bld_symlink(&ws, &ns).unwrap_err().to_string();
        assert!(error.contains("unowned symlink"));
        assert_eq!(fs::read_link(&link).unwrap(), user_target);

        fs::remove_file(&link).unwrap();
        let canonical = fs::canonicalize(&ws).unwrap();
        let dangling = root
            .join(user_namespace_component())
            .join(workspace_hash(&canonical))
            .join("job-evicted")
            .join("bld");
        std::os::unix::fs::symlink(&dangling, &link).unwrap();
        assert!(workspace_link_target_is_owned(&ws, &dangling).unwrap());
        cleanup_owned_workspace_links(&ws).unwrap();
        assert!(fs::symlink_metadata(&link).is_err());
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cleanup_restores_a_concurrently_swapped_unowned_bld_link() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("bld-cleanup-race");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let owned_ns = root.join("owned").join("job-old");
        fs::create_dir_all(owned_ns.join("bld")).unwrap();
        fs::write(
            owned_ns.join("workspace-path"),
            fs::canonicalize(&ws).unwrap().to_string_lossy().as_bytes(),
        )
        .unwrap();
        let link = pixi.join("bld");
        std::os::unix::fs::symlink(owned_ns.join("bld"), &link).unwrap();
        let user_target = root.join("user-bld");

        let error = remove_owned_workspace_symlink_with_hook(&ws, &link, || {
            fs::remove_file(&link).unwrap();
            std::os::unix::fs::symlink(&user_target, &link).unwrap();
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("changed concurrently"));
        assert_eq!(fs::read_link(&link).unwrap(), user_target);
        assert!(
            fs::read_dir(&pixi)
                .unwrap()
                .filter_map(Result::ok)
                .all(|entry| !entry.file_name().to_string_lossy().contains("quarantine"))
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cleanup_restores_a_concurrently_swapped_real_bld_directory() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("bld-cleanup-directory-race");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let owned_ns = root.join("owned").join("job-old");
        fs::create_dir_all(owned_ns.join("bld")).unwrap();
        fs::write(
            owned_ns.join("workspace-path"),
            fs::canonicalize(&ws).unwrap().to_string_lossy().as_bytes(),
        )
        .unwrap();
        let link = pixi.join("bld");
        std::os::unix::fs::symlink(owned_ns.join("bld"), &link).unwrap();

        let error = remove_owned_workspace_symlink_with_hook(&ws, &link, || {
            fs::remove_file(&link).unwrap();
            fs::create_dir(&link).unwrap();
            fs::write(link.join("keep"), b"user directory").unwrap();
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("changed concurrently"));
        assert_eq!(fs::read(link.join("keep")).unwrap(), b"user directory");
        assert!(
            fs::read_dir(&pixi)
                .unwrap()
                .filter_map(Result::ok)
                .all(|entry| !entry.file_name().to_string_lossy().contains("quarantine"))
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn setup_never_overwrites_concurrently_created_symlink_or_directory() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("bld-setup-races");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let link = pixi.join("bld");
        let new_ns = Namespace {
            root: root.join("new").join("job-new"),
        };
        fs::create_dir_all(new_ns.bld_dir()).unwrap();

        let user_target = root.join("user-bld");
        let error = setup_bld_symlink_with_hook(&ws, &new_ns, || {
            std::os::unix::fs::symlink(&user_target, &link).unwrap();
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("changed concurrently"));
        assert_eq!(fs::read_link(&link).unwrap(), user_target);

        fs::remove_file(&link).unwrap();
        let old_ns = root.join("old").join("job-old");
        fs::create_dir_all(old_ns.join("bld")).unwrap();
        fs::write(
            old_ns.join("workspace-path"),
            fs::canonicalize(&ws).unwrap().to_string_lossy().as_bytes(),
        )
        .unwrap();
        std::os::unix::fs::symlink(old_ns.join("bld"), &link).unwrap();
        let error = setup_bld_symlink_with_hook(&ws, &new_ns, || {
            fs::remove_file(&link).unwrap();
            fs::create_dir(&link).unwrap();
            fs::write(link.join("keep"), b"user directory").unwrap();
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("changed concurrently"));
        assert_eq!(fs::read(link.join("keep")).unwrap(), b"user directory");

        fs::write(
            link.join(RETREAD_BLD_OWNER_MARKER),
            bld_owner_marker_contents(&fs::canonicalize(&ws).unwrap()),
        )
        .unwrap();
        let displaced = pixi.join("bld-before-directory-race");
        let error = setup_bld_symlink_with_hook(&ws, &new_ns, || {
            fs::rename(&link, &displaced).unwrap();
            fs::create_dir(&link).unwrap();
            fs::write(link.join("keep"), b"newer directory").unwrap();
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("changed concurrently"));
        assert_eq!(fs::read(link.join("keep")).unwrap(), b"newer directory");
        assert_eq!(fs::read(displaced.join("keep")).unwrap(), b"user directory");
        assert!(
            fs::read_dir(&pixi)
                .unwrap()
                .filter_map(Result::ok)
                .all(|entry| {
                    let name = entry.file_name();
                    let name = name.to_string_lossy();
                    !name.contains("retread-quarantine") && !name.contains("retread-tmp")
                })
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn quarantine_restore_never_overwrites_a_newer_workspace_path() {
        let root = tmp_dir("bld-restore-no-replace");
        let pixi = root.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let quarantine = pixi.join(".bld.retread-quarantine-test");
        fs::create_dir(&quarantine).unwrap();
        fs::write(quarantine.join("keep"), b"displaced directory").unwrap();
        let link = pixi.join("bld");
        let newer_target = root.join("newer-user-bld");
        std::os::unix::fs::symlink(&newer_target, &link).unwrap();

        let error = restore_quarantined_path(&quarantine, &link)
            .unwrap_err()
            .to_string();
        assert!(error.contains("without replacing"));
        assert_eq!(fs::read_link(&link).unwrap(), newer_target);
        assert_eq!(
            fs::read(quarantine.join("keep")).unwrap(),
            b"displaced directory"
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn no_replace_rename_rejects_distinct_parent_directories() {
        let root = tmp_dir("rename-parent-identity");
        let first = root.join("first");
        let second = root.join("second");
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&second).unwrap();
        let source = first.join("source");
        let destination = second.join("destination");
        fs::write(&source, b"keep").unwrap();

        let error = rename_noreplace(&source, &destination).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(fs::read(&source).unwrap(), b"keep");
        assert!(!destination.exists());
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn invalid_managed_metadata_aborts_before_namespace_or_link_mutation() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("invalid-metadata-engage");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let old_ns = root.join("old").join("job-old");
        fs::create_dir_all(old_ns.join("bld")).unwrap();
        fs::write(
            old_ns.join("workspace-path"),
            fs::canonicalize(&ws).unwrap().to_string_lossy().as_bytes(),
        )
        .unwrap();
        let old_bld = old_ns.join("bld");
        let envs_raw = PathBuf::from("../user-envs-relative");
        std::os::unix::fs::symlink(&old_bld, pixi.join("bld")).unwrap();
        std::os::unix::fs::symlink(&envs_raw, pixi.join("envs")).unwrap();

        let tmp_root = root.join("new-tmp");
        guard.set("RETREAD_FAST_TMP_FORCE_FS", "nfs");
        guard.set("RETREAD_FAST_TMP_ROOT", tmp_root.to_str().unwrap());
        guard.set("RETREAD_FAST_TMP_BUDGET_BYTES", "200G");
        guard.set(RETREAD_BASE_ENV_JSON, "not-json");
        let error = engage(&ws, &FastTmpConfig::load(&ws))
            .unwrap_err()
            .to_string();
        assert!(error.contains("ownership metadata"));
        assert!(!tmp_root.exists());
        assert_eq!(fs::read_link(pixi.join("bld")).unwrap(), old_bld);
        assert_eq!(fs::read_link(pixi.join("envs")).unwrap(), envs_raw);
        fs::remove_dir_all(root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn failed_pixi_version_validation_preserves_existing_links_byte_for_byte() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let mut keys = fasttmp_env_keys();
        keys.push("PATH");
        let guard = EnvGuard::new(&keys);
        let root = tmp_dir("version-failure-atomicity");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let old_ns = root.join("old").join("job-old");
        fs::create_dir_all(old_ns.join("envs")).unwrap();
        fs::create_dir_all(old_ns.join("bld")).unwrap();
        fs::write(
            old_ns.join("workspace-path"),
            fs::canonicalize(&ws).unwrap().to_string_lossy().as_bytes(),
        )
        .unwrap();
        let old_envs = old_ns.join("envs");
        let old_bld = old_ns.join("bld");
        std::os::unix::fs::symlink(&old_envs, pixi.join("envs")).unwrap();
        std::os::unix::fs::symlink(&old_bld, pixi.join("bld")).unwrap();

        let fake_bin = install_fake_pixi(&root, "printf '%s\\n' 'pixi 0.69.9'");
        guard.set("PATH", fake_bin.to_str().unwrap());
        guard.set("RETREAD_FAST_TMP_FORCE_FS", "nfs");
        guard.set(
            "RETREAD_FAST_TMP_ROOT",
            root.join("new-tmp").to_str().unwrap(),
        );
        guard.set("RETREAD_FAST_TMP_BUDGET_BYTES", "200G");
        let cfg = FastTmpConfig::load(&ws);
        let error = engage(&ws, &cfg).unwrap_err().to_string();
        assert!(error.contains("requires Pixi >=0.70"));
        assert_eq!(fs::read_link(pixi.join("envs")).unwrap(), old_envs);
        assert_eq!(fs::read_link(pixi.join("bld")).unwrap(), old_bld);
        fs::remove_dir_all(root).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn real_envs_path_is_preserved_during_engagement() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let mut keys = fasttmp_env_keys();
        keys.push("PATH");
        let guard = EnvGuard::new(&keys);
        let root = tmp_dir("real-envs");
        let ws = root.join("workspace");
        write_workspace(&ws);
        fs::create_dir_all(ws.join(".pixi/envs")).unwrap();
        fs::write(ws.join(".pixi/envs/keep"), b"user environment").unwrap();
        guard.set("RETREAD_FAST_TMP_FORCE_FS", "nfs");
        guard.set("RETREAD_FAST_TMP_ROOT", root.join("tmp").to_str().unwrap());
        guard.set("RETREAD_FAST_TMP_BUDGET_BYTES", "200G");
        let fake_bin = install_fake_pixi(&root, "printf '%s\\n' 'pixi 0.70.2'");
        guard.set("PATH", fake_bin.to_str().unwrap());
        assert!(engage(&ws, &FastTmpConfig::load(&ws)).unwrap().is_some());
        assert_eq!(
            fs::read(ws.join(".pixi/envs/keep")).unwrap(),
            b"user environment"
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn slurm_engagement_and_disengagement_preserve_workspace_links_byte_for_byte() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let mut keys = fasttmp_env_keys();
        keys.push("PATH");
        let guard = EnvGuard::new(&keys);
        let root = tmp_dir("slurm-workspace-links");
        let ws = root.join("workspace");
        write_workspace(&ws);
        let pixi = ws.join(".pixi");
        fs::create_dir_all(&pixi).unwrap();
        let envs_raw = PathBuf::from("../user-envs-relative");
        let bld_raw = PathBuf::from("../user-bld-relative");
        std::os::unix::fs::symlink(&envs_raw, pixi.join("envs")).unwrap();
        std::os::unix::fs::symlink(&bld_raw, pixi.join("bld")).unwrap();

        let fake_bin = install_fake_pixi(&root, "printf '%s\\n' 'pixi 0.70.2'");
        guard.set("PATH", fake_bin.to_str().unwrap());
        guard.set("SLURM_JOB_ID", "481516");
        guard.set("RETREAD_FAST_TMP_FORCE_FS", "nfs");
        guard.set("RETREAD_FAST_TMP_ROOT", root.join("tmp").to_str().unwrap());
        guard.set("RETREAD_FAST_TMP_BUDGET_BYTES", "200G");
        let cfg = FastTmpConfig::load(&ws);
        assert!(engage(&ws, &cfg).unwrap().is_some());
        assert_eq!(fs::read_link(pixi.join("envs")).unwrap(), envs_raw);
        assert_eq!(fs::read_link(pixi.join("bld")).unwrap(), bld_raw);

        let off = FastTmpConfig {
            mode: FastTmpMode::Off,
            ..cfg
        };
        assert!(engage(&ws, &off).unwrap().is_none());
        assert_eq!(fs::read_link(pixi.join("envs")).unwrap(), envs_raw);
        assert_eq!(fs::read_link(pixi.join("bld")).unwrap(), bld_raw);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn workspace_local_detached_config_must_not_override_fasttmp_root() {
        let root = tmp_dir("local-detached-config");
        let ws = root.join("workspace");
        write_workspace(&ws);
        fs::create_dir_all(ws.join(".pixi")).unwrap();
        let ns = Namespace {
            root: root.join("job-root"),
        };
        fs::create_dir_all(ns.envs_dir()).unwrap();
        let local = ws.join(".pixi").join("config.toml");

        fs::write(&local, "detached-environments = false\n").unwrap();
        let error = validate_workspace_local_detached_config(&ws, &ns)
            .unwrap_err()
            .to_string();
        assert!(error.contains("incompatible detached-environments"));

        fs::write(&local, "detached-environments = true\n").unwrap();
        validate_workspace_local_detached_config(&ws, &ns).unwrap();

        let mut compatible = toml::Table::new();
        compatible.insert(
            "detached-environments".to_string(),
            toml::Value::String(ns.envs_dir().to_string_lossy().into_owned()),
        );
        fs::write(&local, toml::to_string(&compatible).unwrap()).unwrap();
        validate_workspace_local_detached_config(&ws, &ns).unwrap();

        fs::write(&local, "detached-environments = \"../../job-root/envs\"\n").unwrap();
        validate_workspace_local_detached_config(&ws, &ns).unwrap();

        compatible.insert(
            "detached-environments".to_string(),
            toml::Value::String(root.join("other").to_string_lossy().into_owned()),
        );
        fs::write(&local, toml::to_string(&compatible).unwrap()).unwrap();
        let error = validate_workspace_local_detached_config(&ws, &ns)
            .unwrap_err()
            .to_string();
        assert!(error.contains("overrides retread fast-tmp"));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn budget_override_rejects_default_estimate() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("budget");
        let ws = root.join("workspace");
        write_workspace(&ws);
        guard.set("RETREAD_FAST_TMP_FORCE_FS", "nfs");
        guard.set("RETREAD_FAST_TMP_ROOT", root.join("tmp").to_str().unwrap());
        guard.set("RETREAD_FAST_TMP_BUDGET_BYTES", "1G");
        let cfg = FastTmpConfig::load(&ws);
        let err = engage_backend(&ws, &cfg).unwrap_err().to_string();
        assert!(err.contains("budget too small"));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn copy_back_retries_after_injected_corruption() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        guard.set("RETREAD_FAST_TMP_CORRUPT_COPYBACK", "once");
        CORRUPT_COPYBACK_ONCE_USED.store(false, std::sync::atomic::Ordering::SeqCst);
        let root = tmp_dir("copyback");
        let stage = root.join("stage").join("linux-64");
        let out = root.join("out");
        fs::create_dir_all(&stage).unwrap();
        let artifact = stage.join("pkg-1.0-0.conda");
        fs::write(&artifact, b"conda bytes").unwrap();
        let final_path =
            copy_back_artifacts(root.join("stage").as_path(), &out, &artifact).unwrap();
        assert_eq!(fs::read(&final_path).unwrap(), b"conda bytes");
        assert_eq!(final_path, out.join("linux-64").join("pkg-1.0-0.conda"));
        fs::remove_dir_all(root).ok();
    }
}
