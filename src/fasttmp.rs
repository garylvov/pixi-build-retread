//! First-class fast-tmp support for slow workspace filesystems.
//!
//! The wrapper-facing entry point is [`engage`]. The backend-facing entry
//! point is [`engage_backend`], which prepares the same job-scoped namespace
//! without mutating workspace symlinks.

use std::collections::HashMap;
use std::ffi::CString;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};

const DEFAULT_TMP_ROOT: &str = "/tmp";
const DEFAULT_ESTIMATE_BYTES: u64 = 80 * 1024 * 1024 * 1024;
const PROBE_THRESHOLD_MS: f64 = 5.0;
/// Relative (to workspace root) default location of NFS-persisted env snapshots.
const DEFAULT_PERSIST_DIR: &str = ".pixi/envs-persist";
/// Hash file written alongside each snapshot; contains the sha256 hex of the
/// workspace lock(s) the snapshot was built from.
const ENV_HASH_FILE: &str = ".retread-env-hash";
/// Cap on parallel copy workers (benchmarked sweet spot on NFS: ~60).
const MAX_COPY_WORKERS: usize = 60;

const FAST_ENV_KEYS: &[&str] = &[
    "PIXI_CACHE_DIR",
    "RATTLER_CACHE_DIR",
    "UV_CACHE_DIR",
    "RETREAD_CACHE_DIR",
    "UV_LOCK_TIMEOUT",
    "PIXI_DETACHED_ENVIRONMENTS",
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
    /// Where env snapshots persist on the shared/NFS side. Relative paths are
    /// resolved against the workspace root. Default: `.pixi/envs-persist`.
    pub persist_dir: Option<PathBuf>,
    /// Worker count for parallel env copies. Default: min(60, 4*allocated
    /// cpus) — SLURM/cgroup-aware, ncpu fallback; see effective_copy_workers.
    pub copy_workers: Option<usize>,
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
            persist_dir: None,
            copy_workers: None,
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
        if let Ok(dir) = std::env::var("RETREAD_FAST_TMP_PERSIST_DIR") {
            cfg.persist_dir = Some(PathBuf::from(dir));
        }
        if let Ok(workers) = std::env::var("RETREAD_FAST_TMP_COPY_WORKERS") {
            match workers.trim().parse::<usize>() {
                Ok(n) if n > 0 => cfg.copy_workers = Some(n),
                _ => warn_msg(&format!(
                    "retread fast-tmp: ignoring invalid RETREAD_FAST_TMP_COPY_WORKERS={workers:?}"
                )),
            }
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
        if let Some(v) = table
            .get("persist-dir")
            .or_else(|| table.get("persist_dir"))
        {
            if let Some(s) = v.as_str() {
                self.persist_dir = Some(PathBuf::from(s));
            } else {
                warn_msg("retread fast-tmp: ignoring invalid tool.retread.fast-tmp.persist-dir");
            }
        }
        if let Some(v) = table
            .get("copy-workers")
            .or_else(|| table.get("copy_workers"))
        {
            match v.as_integer() {
                Some(n) if n > 0 => self.copy_workers = Some(n as usize),
                _ => {
                    warn_msg(
                        "retread fast-tmp: ignoring invalid tool.retread.fast-tmp.copy-workers",
                    );
                }
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
        remove_fast_vars: engaged.is_none() && inherited_fasttmp_stale(),
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
        return Ok(None);
    }
    let stale = inherited_fasttmp_stale();
    if stale {
        warn_msg(&format!(
            "retread fast-tmp: inherited namespace for job {:?} is stale under current job {}; re-engaging",
            std::env::var("RETREAD_FAST_TMP_NS_JOB").ok(),
            current_job_marker()
        ));
    }
    if cfg.mode == FastTmpMode::Auto && !stale && !is_slow(workspace_root, cfg) {
        return Ok(None);
    }

    let canonical = fs::canonicalize(workspace_root)
        .with_context(|| format!("canonicalizing workspace {}", workspace_root.display()))?;
    fs::create_dir_all(&cfg.tmp_root)
        .with_context(|| format!("creating tmp root {}", cfg.tmp_root.display()))?;
    enforce_tmp_user_dir(&cfg.tmp_root)?;
    enforce_budget(&cfg.tmp_root, workspace_root, cfg)?;

    let ns = namespace(cfg, &canonical);
    prepare_namespace_dirs(&ns, &canonical, cfg, workspace_root)?;
    let engaged = with_file_lock(&ns.root.join(".engage.lock"), || {
        if wrapper_side_effects && !in_slurm_job() {
            setup_bld_symlink(workspace_root, &ns)?;
        } else if wrapper_side_effects && in_slurm_job() {
            tracing::info!(
                workspace = %workspace_root.display(),
                namespace = %ns.root.display(),
                "retread fast-tmp: SLURM job context; not touching workspace .pixi/bld or .pixi/envs symlinks"
            );
        }
        warn_if_real_envs_dir(workspace_root);
        Ok(EngagedFastTmp {
            env: desired_env_pairs(cfg, workspace_root, &ns, stale),
            ns,
        })
    })?;
    Ok(Some(engaged))
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
    if cfg.blob_caches == BlobCacheMode::Shared {
        let shared = shared_blob_cache_dir(cfg, workspace_root);
        fs::create_dir_all(&shared)
            .with_context(|| format!("creating shared retread blob cache {}", shared.display()))?;
    }
    Ok(())
}

fn desired_env_pairs(
    cfg: &FastTmpConfig,
    workspace_root: &Path,
    ns: &Namespace,
    stale: bool,
) -> Vec<(String, String)> {
    let blob_cache = match cfg.blob_caches {
        BlobCacheMode::Shared => shared_blob_cache_dir(cfg, workspace_root),
        BlobCacheMode::Tmp => ns.rattler_cache_dir(),
    };
    let mut out = Vec::new();
    push_env_if_needed(&mut out, "PIXI_CACHE_DIR", &blob_cache, stale);
    push_env_if_needed(&mut out, "RATTLER_CACHE_DIR", &blob_cache, stale);
    push_env_if_needed(&mut out, "UV_CACHE_DIR", &ns.uv_cache_dir(), stale);
    // NOTE: this redirect deliberately does NOT move the loose-bundle wheel
    // store. The store is resolved by courier::retread_wheel_store_root(),
    // which ignores RETREAD_CACHE_DIR precisely so blob stores stay SHARED
    // while scratch caches go job-local (a job-local store dies with the
    // job, breaking `retread install` on other nodes / later jobs).
    push_env_if_needed(
        &mut out,
        "RETREAD_CACHE_DIR",
        &ns.retread_cache_dir(),
        stale,
    );
    push_env_str_if_needed(&mut out, "UV_LOCK_TIMEOUT", "1800", stale);
    push_env_if_needed(
        &mut out,
        "PIXI_DETACHED_ENVIRONMENTS",
        &ns.envs_dir(),
        stale,
    );
    push_env_str_if_needed(
        &mut out,
        "RETREAD_FAST_TMP_NS_JOB",
        &current_job_marker(),
        true,
    );
    out
}

fn push_env_if_needed(out: &mut Vec<(String, String)>, key: &str, value: &Path, stale: bool) {
    if stale || std::env::var_os(key).is_none() {
        out.push((key.to_string(), value.to_string_lossy().to_string()));
    }
}

fn push_env_str_if_needed(out: &mut Vec<(String, String)>, key: &str, value: &str, stale: bool) {
    if stale || std::env::var_os(key).is_none() {
        out.push((key.to_string(), value.to_string()));
    }
}

pub fn backend_env_value(key: &str) -> Option<String> {
    BACKEND_ENV.get().and_then(|env| {
        env.lock()
            .unwrap()
            .pairs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    })
}

pub fn apply_backend_env(cmd: &mut tokio::process::Command) {
    let env = BACKEND_ENV
        .get_or_init(|| Mutex::new(BackendEnv::default()))
        .lock()
        .unwrap()
        .clone();
    if env.remove_fast_vars {
        for key in FAST_ENV_KEYS {
            cmd.env_remove(key);
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

fn setup_bld_symlink(workspace_root: &Path, ns: &Namespace) -> Result<()> {
    let pixi = workspace_root.join(".pixi");
    fs::create_dir_all(&pixi).with_context(|| format!("creating {}", pixi.display()))?;
    let link = pixi.join("bld");
    let target = ns.bld_dir();
    match fs::symlink_metadata(&link) {
        Ok(meta) if meta.file_type().is_symlink() => {
            let current = fs::read_link(&link)
                .with_context(|| format!("reading build symlink {}", link.display()))?;
            if current == target {
                return Ok(());
            }
            atomic_symlink_replace(&target, &link)?;
        }
        Ok(_) => {
            bail!(
                "retread fast-tmp refuses to replace real path {}. Move it aside before enabling fast-tmp.",
                link.display()
            );
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            atomic_symlink_replace(&target, &link)?;
        }
        Err(e) => return Err(e).with_context(|| format!("checking {}", link.display())),
    }
    let now = fs::read_link(&link).with_context(|| format!("reading {}", link.display()))?;
    if now != target {
        bail!(
            "retread fast-tmp failed to point {} at {}; it points at {}",
            link.display(),
            target.display(),
            now.display()
        );
    }
    Ok(())
}

#[cfg(unix)]
fn atomic_symlink_replace(target: &Path, link: &Path) -> Result<()> {
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
    match fs::rename(&tmp, link) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e).with_context(|| {
                format!(
                    "atomically replacing symlink {} with target {}",
                    link.display(),
                    target.display()
                )
            })
        }
    }
}

#[cfg(not(unix))]
fn atomic_symlink_replace(_target: &Path, _link: &Path) -> Result<()> {
    bail!("retread fast-tmp symlink setup is only supported on Unix")
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

// ---------------------------------------------------------------------------
// Env materialization fast path (parallel NFS copy) + persist-back.
//
// PREFIX IDENTITY INVARIANT: a snapshot under `envs-persist` was produced from
// an env whose baked-in prefix (shebangs, activation scripts, conda-meta,
// text-relocated paths) is the workspace's `<workspace>/.pixi/envs/<env>` NFS
// path identity. Copying those bytes into the job-local namespace
// (`ns.envs_dir()/<env>`) is only correct because the engage path points
// `<workspace>/.pixi/envs` (a symlink) at `ns.envs_dir()` — i.e. the env is
// always *reached through* the identical `<workspace>/.pixi/envs/<env>` path
// on every node. Never hand out the raw job-tmp path as the env prefix, and
// never materialize a snapshot somewhere that is not behind that symlink.
// ---------------------------------------------------------------------------

/// Stats from one parallel tree copy.
#[derive(Debug, Clone, Copy, Default)]
pub struct CopyStats {
    pub files: u64,
    pub bytes: u64,
    pub dirs: u64,
    pub symlinks: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryKind {
    File,
    Symlink,
}

/// Resolved persist dir for this workspace (config or `.pixi/envs-persist`).
pub fn persist_dir(cfg: &FastTmpConfig, workspace_root: &Path) -> PathBuf {
    match &cfg.persist_dir {
        Some(dir) if dir.is_absolute() => dir.clone(),
        Some(dir) => workspace_root.join(dir),
        None => workspace_root.join(DEFAULT_PERSIST_DIR),
    }
}

/// Worker count: config wins, else min(60, 4*allocated_cpus).
///
/// "Allocated" is allocation-aware: on shared nodes the process is often
/// confined to a slice of the machine, and sizing off the full core count
/// oversubscribes the copy pool. Precedence:
/// 1. `$SLURM_CPUS_ON_NODE` (cpus SLURM granted on this node)
/// 2. cgroup v2 cpu quota (`/sys/fs/cgroup/cpu.max`, quota/period rounded up)
/// 3. `available_parallelism()` (the historical formula)
pub fn effective_copy_workers(cfg: &FastTmpConfig) -> usize {
    if let Some(n) = cfg.copy_workers {
        return n.max(1);
    }
    let ncpu = allocated_cpus().unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });
    MAX_COPY_WORKERS.min(4 * ncpu).max(1)
}

/// CPUs actually allocated to this process, when a scheduler or cgroup
/// says so. `None` = no allocation signal; caller falls back to ncpu.
fn allocated_cpus() -> Option<usize> {
    if let Ok(v) = std::env::var("SLURM_CPUS_ON_NODE")
        && let Ok(n) = v.trim().parse::<usize>()
        && n > 0
    {
        return Some(n);
    }
    cgroup_v2_cpu_quota(Path::new("/sys/fs/cgroup/cpu.max"))
}

/// Parse a cgroup v2 `cpu.max` file ("<quota> <period>" or "max <period>")
/// into a whole-cpu count (quota/period, rounded up). "max" = unconstrained.
fn cgroup_v2_cpu_quota(path: &Path) -> Option<usize> {
    let raw = fs::read_to_string(path).ok()?;
    let mut it = raw.split_whitespace();
    let quota = it.next()?;
    if quota == "max" {
        return None;
    }
    let quota: u64 = quota.parse().ok()?;
    let period: u64 = it.next()?.parse().ok()?;
    if quota == 0 || period == 0 {
        return None;
    }
    Some(quota.div_ceil(period).max(1) as usize)
}

/// sha256 hex over the workspace lock(s) a snapshot is keyed by. Currently
/// this is `pixi.lock`; if the retread lock format grows sibling lock files
/// they must be hashed here too (sorted, concatenated) so snapshots invalidate.
pub fn current_lock_hash(workspace_root: &Path) -> Option<String> {
    let lock = workspace_root.join("pixi.lock");
    let bytes = fs::read(&lock).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Some(
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    )
}

fn snapshot_hash(snapshot: &Path) -> Option<String> {
    fs::read_to_string(snapshot.join(ENV_HASH_FILE))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Test hook: fail the copy of any entry whose relative path contains the
/// value of RETREAD_FAST_TMP_COPY_FAIL_SUBSTR. Used to exercise the
/// partial-failure fallback without real I/O errors.
fn maybe_inject_copy_failure(rel: &Path) -> Result<()> {
    if let Ok(needle) = std::env::var("RETREAD_FAST_TMP_COPY_FAIL_SUBSTR")
        && !needle.is_empty()
        && rel.to_string_lossy().contains(&needle)
    {
        bail!(
            "retread fast-tmp: injected copy failure for {} (test hook)",
            rel.display()
        );
    }
    Ok(())
}

/// Copy `src` tree to `dst` with `workers` threads. Pre-creates the directory
/// tree first (so workers never race on mkdir), then fans file/symlink copies
/// out over a shared work queue. Permissions are preserved: `fs::copy` carries
/// file modes, symlinks are recreated as symlinks (targets untouched), and
/// directory modes are applied deepest-first after the file pass so read-only
/// dirs cannot block their own population. On any worker error the copy stops
/// and the error is returned; the caller owns cleanup of the partial `dst`.
pub fn parallel_copy_tree(src: &Path, dst: &Path, workers: usize) -> Result<CopyStats> {
    let mut dirs: Vec<(PathBuf, u32)> = Vec::new();
    let mut entries: Vec<(PathBuf, EntryKind)> = Vec::new();
    walk_tree(src, Path::new(""), &mut dirs, &mut entries)?;

    fs::create_dir_all(dst).with_context(|| format!("creating {}", dst.display()))?;
    for (rel, _mode) in &dirs {
        let dir = dst.join(rel);
        fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    }

    let next = AtomicUsize::new(0);
    let bytes = AtomicU64::new(0);
    let failed: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let workers = workers.max(1).min(entries.len().max(1));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= entries.len() || failed.lock().unwrap().is_some() {
                        break;
                    }
                    let (rel, kind) = &entries[i];
                    match copy_entry(src, dst, rel, *kind) {
                        Ok(copied) => {
                            bytes.fetch_add(copied, Ordering::Relaxed);
                        }
                        Err(e) => {
                            let mut slot = failed.lock().unwrap();
                            if slot.is_none() {
                                *slot = Some(e);
                            }
                            break;
                        }
                    }
                }
            });
        }
    });
    if let Some(e) = failed.into_inner().unwrap() {
        return Err(e);
    }

    // Directory modes last, deepest-first (walk order is parent-before-child).
    for (rel, mode) in dirs.iter().rev() {
        set_mode(&dst.join(rel), *mode);
    }

    let (files, symlinks) = entries
        .iter()
        .fold((0_u64, 0_u64), |(f, s), (_, k)| match k {
            EntryKind::File => (f + 1, s),
            EntryKind::Symlink => (f, s + 1),
        });
    Ok(CopyStats {
        files,
        bytes: bytes.into_inner(),
        dirs: dirs.len() as u64,
        symlinks,
    })
}

fn walk_tree(
    root: &Path,
    rel: &Path,
    dirs: &mut Vec<(PathBuf, u32)>,
    entries: &mut Vec<(PathBuf, EntryKind)>,
) -> Result<()> {
    let abs = root.join(rel);
    for entry in fs::read_dir(&abs).with_context(|| format!("reading {}", abs.display()))? {
        let entry = entry.with_context(|| format!("reading entry in {}", abs.display()))?;
        let child_rel = rel.join(entry.file_name());
        let ft = entry
            .file_type()
            .with_context(|| format!("file type of {}", entry.path().display()))?;
        if ft.is_symlink() {
            entries.push((child_rel, EntryKind::Symlink));
        } else if ft.is_dir() {
            let mode = entry.metadata().map(unix_mode).unwrap_or(0o755);
            dirs.push((child_rel.clone(), mode));
            walk_tree(root, &child_rel, dirs, entries)?;
        } else {
            entries.push((child_rel, EntryKind::File));
        }
    }
    Ok(())
}

fn copy_entry(src: &Path, dst: &Path, rel: &Path, kind: EntryKind) -> Result<u64> {
    maybe_inject_copy_failure(rel)?;
    let from = src.join(rel);
    let to = dst.join(rel);
    match kind {
        EntryKind::Symlink => {
            let target = fs::read_link(&from)
                .with_context(|| format!("reading symlink {}", from.display()))?;
            if fs::symlink_metadata(&to).is_ok() {
                fs::remove_file(&to).with_context(|| format!("removing stale {}", to.display()))?;
            }
            make_symlink(&target, &to)?;
            Ok(0)
        }
        EntryKind::File => fs::copy(&from, &to)
            .with_context(|| format!("copying {} -> {}", from.display(), to.display())),
    }
}

#[cfg(unix)]
fn make_symlink(target: &Path, link: &Path) -> Result<()> {
    std::os::unix::fs::symlink(target, link).with_context(|| {
        format!(
            "creating symlink {} -> {}",
            link.display(),
            target.display()
        )
    })
}

#[cfg(not(unix))]
fn make_symlink(_target: &Path, link: &Path) -> Result<()> {
    bail!(
        "retread fast-tmp: symlink materialization is only supported on Unix ({})",
        link.display()
    )
}

#[cfg(unix)]
fn unix_mode(meta: fs::Metadata) -> u32 {
    use std::os::unix::fs::MetadataExt;
    meta.mode() & 0o7777
}

#[cfg(not(unix))]
fn unix_mode(_meta: fs::Metadata) -> u32 {
    0o755
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(mode));
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: u32) {}

/// Try to materialize all persisted env snapshots into the job-local
/// namespace. Returns `true` only when every snapshot present in the persist
/// dir carried a hash matching the current lock and was copied (or already
/// present) — in that case the frozen `pixi install` can be skipped entirely.
/// Any hash miss/absence, or any copy failure (partial dest is deleted),
/// returns `false` so the caller falls back to the existing install path.
pub fn materialize_persisted_envs(
    workspace_root: &Path,
    cfg: &FastTmpConfig,
    ns: &Namespace,
) -> bool {
    let Some(lock_hash) = current_lock_hash(workspace_root) else {
        return false;
    };
    let pdir = persist_dir(cfg, workspace_root);
    let Ok(read) = fs::read_dir(&pdir) else {
        return false;
    };
    let workers = effective_copy_workers(cfg);
    let mut materialized_any = false;
    for entry in read.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Skip in-flight tmp/old swap dirs and stray dotfiles.
        if name_str.starts_with('.') {
            continue;
        }
        let snapshot = entry.path();
        if !snapshot.is_dir() {
            continue;
        }
        match snapshot_hash(&snapshot) {
            Some(hash) if hash == lock_hash => {}
            other => {
                warn_msg(&format!(
                    "retread fast-tmp: snapshot {} {}; falling back to frozen install",
                    snapshot.display(),
                    if other.is_none() {
                        "has no .retread-env-hash"
                    } else {
                        "lock hash does not match current lock"
                    },
                ));
                return false;
            }
        }
        let dest = ns.envs_dir().join(&name);
        if fs::read_dir(&dest)
            .ok()
            .and_then(|mut it| it.next())
            .is_some()
        {
            // Already materialized in this job namespace.
            materialized_any = true;
            continue;
        }
        let started = Instant::now();
        match parallel_copy_tree(&snapshot, &dest, workers) {
            Ok(stats) => {
                // The hash file is snapshot metadata, not env content.
                let _ = fs::remove_file(dest.join(ENV_HASH_FILE));
                log_copy_timing("materialized", &name_str, &stats, started.elapsed());
                materialized_any = true;
            }
            Err(e) => {
                warn_msg(&format!(
                    "retread fast-tmp: parallel materialization of {} failed ({e:#}); removing partial dest and falling back to frozen install",
                    snapshot.display()
                ));
                let _ = fs::remove_dir_all(&dest);
                return false;
            }
        }
    }
    materialized_any
}

/// `retread fast --persist <env>`: parallel-copy the job-local env back to the
/// NFS persist dir and stamp it with the current lock hash. Atomic: copies to
/// `persist-dir/.tmp-<pid>-<nonce>` then rename-swaps into place; a previous
/// snapshot is renamed aside first and removed after the swap.
pub fn persist_env(
    workspace_root: &Path,
    cfg: &FastTmpConfig,
    ns: &Namespace,
    env_name: &str,
) -> Result<()> {
    if env_name.is_empty()
        || env_name.starts_with('.')
        || env_name.contains('/')
        || env_name.contains("..")
    {
        bail!("retread fast --persist: invalid env name {env_name:?}");
    }
    let src = ns.envs_dir().join(env_name);
    if !src.is_dir() {
        bail!(
            "retread fast --persist: job-local env not found at {}",
            src.display()
        );
    }
    let lock_hash = current_lock_hash(workspace_root).ok_or_else(|| {
        anyhow!(
            "retread fast --persist: no readable pixi.lock in {}; refusing to persist an unkeyed snapshot",
            workspace_root.display()
        )
    })?;
    let pdir = persist_dir(cfg, workspace_root);
    fs::create_dir_all(&pdir)
        .with_context(|| format!("creating persist dir {}", pdir.display()))?;
    let tmp = pdir.join(format!(".tmp-{}-{}", std::process::id(), unique_nonce()));
    let workers = effective_copy_workers(cfg);
    let started = Instant::now();
    let stats = match parallel_copy_tree(&src, &tmp, workers) {
        Ok(stats) => stats,
        Err(e) => {
            let _ = fs::remove_dir_all(&tmp);
            return Err(e).with_context(|| {
                format!(
                    "retread fast --persist: parallel copy {} -> {} failed; partial tmp removed",
                    src.display(),
                    tmp.display()
                )
            });
        }
    };
    fs::write(tmp.join(ENV_HASH_FILE), format!("{lock_hash}\n"))
        .with_context(|| format!("writing {}", tmp.join(ENV_HASH_FILE).display()))?;

    let dest = pdir.join(env_name);
    let old = pdir.join(format!(".old-{}-{}", std::process::id(), unique_nonce()));
    let had_old = match fs::rename(&dest, &old) {
        Ok(()) => true,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(e) => {
            let _ = fs::remove_dir_all(&tmp);
            return Err(e)
                .with_context(|| format!("moving previous snapshot {} aside", dest.display()));
        }
    };
    if let Err(e) = fs::rename(&tmp, &dest) {
        // Best effort: restore the old snapshot, drop the tmp copy.
        if had_old {
            let _ = fs::rename(&old, &dest);
        }
        let _ = fs::remove_dir_all(&tmp);
        return Err(e)
            .with_context(|| format!("swapping snapshot {} -> {}", tmp.display(), dest.display()));
    }
    if had_old {
        let _ = fs::remove_dir_all(&old);
    }
    log_copy_timing("persisted", env_name, &stats, started.elapsed());
    Ok(())
}

/// `retread fast --persist` with the env omitted (or the literal `all`):
/// persist every env directory present under the job-local envs root.
/// Non-directories and dotted entries are skipped; each env goes through
/// `persist_env`, so every snapshot keeps its own per-env hash stamp.
pub fn persist_all_envs(workspace_root: &Path, cfg: &FastTmpConfig, ns: &Namespace) -> Result<()> {
    let envs_dir = ns.envs_dir();
    let entries = fs::read_dir(&envs_dir)
        .with_context(|| format!("reading job-local envs dir {}", envs_dir.display()))?;
    let mut names: Vec<String> = Vec::new();
    for entry in entries {
        let entry = entry.with_context(|| format!("reading entry in {}", envs_dir.display()))?;
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        names.push(name);
    }
    if names.is_empty() {
        bail!(
            "retread fast --persist: no job-local envs found under {}",
            envs_dir.display()
        );
    }
    names.sort();
    for name in &names {
        persist_env(workspace_root, cfg, ns, name)?;
    }
    Ok(())
}

fn log_copy_timing(verb: &str, env_name: &str, stats: &CopyStats, elapsed: Duration) {
    let secs = elapsed.as_secs_f64();
    let total = stats.files + stats.symlinks;
    let rate = if secs > 0.0 { total as f64 / secs } else { 0.0 };
    let msg = format!(
        "retread fast-tmp: {verb} env {env_name}: {total} files ({} symlinks, {} dirs), {} bytes, {secs:.2} s, {rate:.0} files/s",
        stats.symlinks, stats.dirs, stats.bytes
    );
    eprintln!("{msg}");
    tracing::info!("{msg}");
}

pub fn run_frozen_install_if_slurm(
    workspace_root: &Path,
    cfg: &FastTmpConfig,
    ns: &Namespace,
    env: &[(String, String)],
) -> Result<()> {
    if !in_slurm_job() {
        return Ok(());
    }
    // Fast path: byte-for-byte parallel copy of persisted snapshots keyed by
    // the current lock hash. When it fully succeeds the frozen install (and
    // its lock check) is redundant — the snapshot was built from this lock.
    if materialize_persisted_envs(workspace_root, cfg, ns) {
        return Ok(());
    }
    let lock_status = Command::new("pixi")
        .args(["lock", "--check"])
        .current_dir(workspace_root)
        .envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .stdin(Stdio::null())
        .status()
        .context("running `pixi lock --check` before SLURM frozen install")?;
    if !lock_status.success() {
        bail!(
            "pixi.lock is not up to date; run `retread solve` / `pixi install` once on a login node (or one designated job) before fanning out."
        );
    }
    let install_status = Command::new("pixi")
        .args(["install", "--frozen"])
        .current_dir(workspace_root)
        .envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .stdin(Stdio::null())
        .status()
        .context("running `pixi install --frozen` inside SLURM job")?;
    if !install_status.success() {
        bail!("`pixi install --frozen` failed with status {install_status}");
    }
    Ok(())
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
    if inherited_fasttmp_stale() {
        for key in FAST_ENV_KEYS {
            cmd.env_remove(key);
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
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock_path)
        .with_context(|| format!("opening lock file {}", lock_path.display()))?;
    fs4::fs_std::FileExt::lock_exclusive(&file)
        .with_context(|| format!("locking {}", lock_path.display()))?;
    Ok(FileLock { file })
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
            "RETREAD_FAST_TMP_PERSIST_DIR",
            "RETREAD_FAST_TMP_COPY_WORKERS",
            "RETREAD_FAST_TMP_COPY_FAIL_SUBSTR",
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

    #[cfg(unix)]
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
        assert!(
            !first.env.iter().any(|(k, _)| k == "UV_CACHE_DIR"),
            "pre-set UV_CACHE_DIR must not be overridden"
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn engage_rejects_real_bld_path() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("real-bld");
        let ws = root.join("workspace");
        write_workspace(&ws);
        fs::create_dir_all(ws.join(".pixi").join("bld")).unwrap();
        guard.set("RETREAD_FAST_TMP_FORCE_FS", "nfs");
        guard.set("RETREAD_FAST_TMP_ROOT", root.join("tmp").to_str().unwrap());
        guard.set("RETREAD_FAST_TMP_BUDGET_BYTES", "200G");
        let cfg = FastTmpConfig::load(&ws);
        let err = engage(&ws, &cfg).unwrap_err().to_string();
        assert!(err.contains("refuses to replace real path"));
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

    #[cfg(unix)]
    fn write_fixture_env(env_dir: &Path) {
        use std::os::unix::fs::PermissionsExt;
        fs::create_dir_all(env_dir.join("bin")).unwrap();
        fs::create_dir_all(env_dir.join("lib").join("python3.11").join("pkg")).unwrap();
        fs::create_dir_all(env_dir.join("empty")).unwrap();
        fs::write(env_dir.join("bin").join("tool"), b"#!/bin/sh\necho hi\n").unwrap();
        fs::set_permissions(
            env_dir.join("bin").join("tool"),
            fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        fs::write(
            env_dir
                .join("lib")
                .join("python3.11")
                .join("pkg")
                .join("mod.py"),
            b"x = 1\n",
        )
        .unwrap();
        fs::set_permissions(
            env_dir
                .join("lib")
                .join("python3.11")
                .join("pkg")
                .join("mod.py"),
            fs::Permissions::from_mode(0o644),
        )
        .unwrap();
        // Relative symlink (like lib/libfoo.so -> libfoo.so.1) and a dangling
        // absolute one (conda envs contain both; both must copy as symlinks).
        std::os::unix::fs::symlink("tool", env_dir.join("bin").join("tool-alias")).unwrap();
        std::os::unix::fs::symlink(
            "/definitely/not/a/real/target",
            env_dir.join("lib").join("dangling"),
        )
        .unwrap();
        fs::set_permissions(env_dir.join("empty"), fs::Permissions::from_mode(0o700)).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn parallel_copy_preserves_tree_perms_and_symlinks() {
        use std::os::unix::fs::PermissionsExt;
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::new(&fasttmp_env_keys());
        let root = tmp_dir("pcopy");
        let src = root.join("src");
        write_fixture_env(&src);
        let dst = root.join("dst");

        let stats = parallel_copy_tree(&src, &dst, 8).unwrap();
        assert_eq!(stats.files, 2);
        assert_eq!(stats.symlinks, 2);
        assert_eq!(stats.dirs, 5);
        assert_eq!(
            stats.bytes,
            fs::metadata(src.join("bin").join("tool")).unwrap().len()
                + fs::metadata(
                    src.join("lib")
                        .join("python3.11")
                        .join("pkg")
                        .join("mod.py")
                )
                .unwrap()
                .len()
        );

        assert_eq!(
            fs::read(dst.join("bin").join("tool")).unwrap(),
            b"#!/bin/sh\necho hi\n"
        );
        assert_eq!(
            fs::metadata(dst.join("bin").join("tool"))
                .unwrap()
                .permissions()
                .mode()
                & 0o7777,
            0o755
        );
        assert_eq!(
            fs::metadata(dst.join("empty"))
                .unwrap()
                .permissions()
                .mode()
                & 0o7777,
            0o700
        );
        let alias = dst.join("bin").join("tool-alias");
        assert!(
            fs::symlink_metadata(&alias)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read_link(&alias).unwrap(), PathBuf::from("tool"));
        let dangling = dst.join("lib").join("dangling");
        assert!(
            fs::symlink_metadata(&dangling)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            fs::read_link(&dangling).unwrap(),
            PathBuf::from("/definitely/not/a/real/target")
        );
        fs::remove_dir_all(root).ok();
    }

    #[cfg(unix)]
    struct PersistFixture {
        root: PathBuf,
        ws: PathBuf,
        ns: Namespace,
        cfg: FastTmpConfig,
    }

    #[cfg(unix)]
    fn persist_fixture(label: &str) -> PersistFixture {
        let root = tmp_dir(label);
        let ws = root.join("workspace");
        write_workspace(&ws);
        fs::write(ws.join("pixi.lock"), b"version: 6\npackages: []\n").unwrap();
        let ns = Namespace {
            root: root.join("nsroot"),
        };
        fs::create_dir_all(ns.envs_dir()).unwrap();
        let cfg = FastTmpConfig {
            persist_dir: Some(root.join("persist")),
            copy_workers: Some(4),
            ..FastTmpConfig::default()
        };
        PersistFixture { root, ws, ns, cfg }
    }

    #[cfg(unix)]
    #[test]
    fn materialize_gates_on_lock_hash() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::new(&fasttmp_env_keys());
        let fx = persist_fixture("hashgate");
        let snap = persist_dir(&fx.cfg, &fx.ws).join("default");
        write_fixture_env(&snap);
        let hash = current_lock_hash(&fx.ws).unwrap();

        // No hash file -> miss -> no materialization.
        assert!(!materialize_persisted_envs(&fx.ws, &fx.cfg, &fx.ns));
        assert!(!fx.ns.envs_dir().join("default").exists());

        // Wrong hash -> miss.
        fs::write(snap.join(ENV_HASH_FILE), "deadbeef\n").unwrap();
        assert!(!materialize_persisted_envs(&fx.ws, &fx.cfg, &fx.ns));
        assert!(!fx.ns.envs_dir().join("default").exists());

        // Matching hash -> materialized, hash metadata file not copied along.
        fs::write(snap.join(ENV_HASH_FILE), format!("{hash}\n")).unwrap();
        assert!(materialize_persisted_envs(&fx.ws, &fx.cfg, &fx.ns));
        let dest = fx.ns.envs_dir().join("default");
        assert_eq!(
            fs::read(dest.join("bin").join("tool")).unwrap(),
            b"#!/bin/sh\necho hi\n"
        );
        assert!(!dest.join(ENV_HASH_FILE).exists());
        assert!(dest.join("bin").join("tool-alias").is_symlink());

        // Idempotent: second call sees a warm dest and still reports success.
        assert!(materialize_persisted_envs(&fx.ws, &fx.cfg, &fx.ns));
        fs::remove_dir_all(fx.root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn materialize_partial_failure_cleans_dest_and_falls_back() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let fx = persist_fixture("partial");
        let snap = persist_dir(&fx.cfg, &fx.ws).join("default");
        write_fixture_env(&snap);
        let hash = current_lock_hash(&fx.ws).unwrap();
        fs::write(snap.join(ENV_HASH_FILE), format!("{hash}\n")).unwrap();

        guard.set("RETREAD_FAST_TMP_COPY_FAIL_SUBSTR", "mod.py");
        assert!(!materialize_persisted_envs(&fx.ws, &fx.cfg, &fx.ns));
        assert!(
            !fx.ns.envs_dir().join("default").exists(),
            "partial dest must be deleted so the frozen install starts clean"
        );

        guard.remove("RETREAD_FAST_TMP_COPY_FAIL_SUBSTR");
        assert!(materialize_persisted_envs(&fx.ws, &fx.cfg, &fx.ns));
        fs::remove_dir_all(fx.root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn persist_env_swaps_atomically_and_stamps_hash() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let fx = persist_fixture("persist");
        let local = fx.ns.envs_dir().join("default");
        write_fixture_env(&local);

        persist_env(&fx.ws, &fx.cfg, &fx.ns, "default").unwrap();
        let pdir = persist_dir(&fx.cfg, &fx.ws);
        let snap = pdir.join("default");
        let hash = current_lock_hash(&fx.ws).unwrap();
        assert_eq!(
            fs::read_to_string(snap.join(ENV_HASH_FILE)).unwrap().trim(),
            hash
        );
        assert_eq!(
            fs::read(snap.join("bin").join("tool")).unwrap(),
            b"#!/bin/sh\necho hi\n"
        );
        assert!(snap.join("bin").join("tool-alias").is_symlink());

        // Re-persist replaces the old snapshot and leaves no tmp/old debris.
        fs::write(local.join("bin").join("extra"), b"new").unwrap();
        persist_env(&fx.ws, &fx.cfg, &fx.ns, "default").unwrap();
        assert_eq!(fs::read(snap.join("bin").join("extra")).unwrap(), b"new");
        let leftovers: Vec<String> = fs::read_dir(&pdir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with(".tmp-") || n.starts_with(".old-"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "swap debris left behind: {leftovers:?}"
        );

        // Failed persist removes the tmp dir and keeps the old snapshot.
        guard.set("RETREAD_FAST_TMP_COPY_FAIL_SUBSTR", "mod.py");
        assert!(persist_env(&fx.ws, &fx.cfg, &fx.ns, "default").is_err());
        assert!(snap.join("bin").join("extra").exists());
        let tmp_left = fs::read_dir(&pdir)
            .unwrap()
            .flatten()
            .any(|e| e.file_name().to_string_lossy().starts_with(".tmp-"));
        assert!(!tmp_left, "failed persist must remove its tmp dir");

        assert!(persist_env(&fx.ws, &fx.cfg, &fx.ns, "../evil").is_err());
        fs::remove_dir_all(fx.root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn persist_all_envs_persists_every_env_dir() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let _guard = EnvGuard::new(&fasttmp_env_keys());
        let fx = persist_fixture("persistall");

        // Nothing job-local yet -> error, not a silent no-op.
        assert!(persist_all_envs(&fx.ws, &fx.cfg, &fx.ns).is_err());

        write_fixture_env(&fx.ns.envs_dir().join("default"));
        write_fixture_env(&fx.ns.envs_dir().join("gpu"));
        // Skipped: non-directory and dotted entries in the envs root.
        fs::write(fx.ns.envs_dir().join("stray-file"), b"not an env").unwrap();
        fs::create_dir_all(fx.ns.envs_dir().join(".hidden")).unwrap();

        persist_all_envs(&fx.ws, &fx.cfg, &fx.ns).unwrap();

        let pdir = persist_dir(&fx.cfg, &fx.ws);
        let hash = current_lock_hash(&fx.ws).unwrap();
        for env in ["default", "gpu"] {
            let snap = pdir.join(env);
            assert_eq!(
                fs::read_to_string(snap.join(ENV_HASH_FILE)).unwrap().trim(),
                hash,
                "per-env hash stamp missing for {env}"
            );
            assert_eq!(
                fs::read(snap.join("bin").join("tool")).unwrap(),
                b"#!/bin/sh\necho hi\n"
            );
        }
        assert!(!pdir.join("stray-file").exists());
        assert!(!pdir.join(".hidden").exists());
        fs::remove_dir_all(fx.root).ok();
    }

    #[test]
    fn copy_workers_config_and_default() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::new(&fasttmp_env_keys());
        let ws = tmp_dir("workers");
        fs::write(
            ws.join("pixi.toml"),
            r#"
[workspace]
channels = []

[tool.retread.fast-tmp]
persist-dir = "/nfs/persist"
copy-workers = 12
"#,
        )
        .unwrap();
        let cfg = FastTmpConfig::load(&ws);
        assert_eq!(cfg.persist_dir, Some(PathBuf::from("/nfs/persist")));
        assert_eq!(cfg.copy_workers, Some(12));
        assert_eq!(effective_copy_workers(&cfg), 12);
        assert_eq!(persist_dir(&cfg, &ws), PathBuf::from("/nfs/persist"));

        guard.set("RETREAD_FAST_TMP_COPY_WORKERS", "3");
        guard.set("RETREAD_FAST_TMP_PERSIST_DIR", "rel/persist");
        let cfg = FastTmpConfig::load(&ws);
        assert_eq!(cfg.copy_workers, Some(3));
        assert_eq!(persist_dir(&cfg, &ws), ws.join("rel/persist"));

        let default_cfg = FastTmpConfig::default();
        let n = effective_copy_workers(&default_cfg);
        assert!((1..=MAX_COPY_WORKERS).contains(&n));
        assert_eq!(
            persist_dir(&default_cfg, &ws),
            ws.join(".pixi/envs-persist")
        );

        // Allocation-aware default: SLURM_CPUS_ON_NODE beats ncpu...
        guard.remove("RETREAD_FAST_TMP_COPY_WORKERS");
        guard.set("SLURM_CPUS_ON_NODE", "3");
        assert_eq!(effective_copy_workers(&default_cfg), 12);
        // ...but never beats an explicit copy-workers override, and junk
        // values fall back to the ncpu formula.
        let pinned = FastTmpConfig {
            copy_workers: Some(2),
            ..FastTmpConfig::default()
        };
        assert_eq!(effective_copy_workers(&pinned), 2);
        guard.set("SLURM_CPUS_ON_NODE", "banana");
        let n = effective_copy_workers(&default_cfg);
        assert!((1..=MAX_COPY_WORKERS).contains(&n));
        guard.remove("SLURM_CPUS_ON_NODE");
        fs::remove_dir_all(ws).ok();
    }

    #[test]
    fn cgroup_v2_cpu_quota_parses_quota_and_max() {
        let dir = tmp_dir("cgroup");
        let f = dir.join("cpu.max");
        // 2.5 cpus rounds up to 3.
        fs::write(&f, "250000 100000\n").unwrap();
        assert_eq!(cgroup_v2_cpu_quota(&f), Some(3));
        fs::write(&f, "100000 100000\n").unwrap();
        assert_eq!(cgroup_v2_cpu_quota(&f), Some(1));
        // Unconstrained and malformed both mean "no signal".
        fs::write(&f, "max 100000\n").unwrap();
        assert_eq!(cgroup_v2_cpu_quota(&f), None);
        fs::write(&f, "garbage\n").unwrap();
        assert_eq!(cgroup_v2_cpu_quota(&f), None);
        assert_eq!(cgroup_v2_cpu_quota(&dir.join("absent")), None);
        fs::remove_dir_all(dir).ok();
    }
}
