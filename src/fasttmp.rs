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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};

const DEFAULT_TMP_ROOT: &str = "/tmp";
const DEFAULT_ESTIMATE_BYTES: u64 = 80 * 1024 * 1024 * 1024;
const PROBE_THRESHOLD_MS: f64 = 5.0;

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
        cfg
    }

    fn apply_toml(&mut self, parsed: &toml::Value) {
        let Some(table) = parsed
            .get("tool")
            .and_then(toml::Value::as_table)
            .and_then(|tool| tool.get("retread"))
            .and_then(toml::Value::as_table)
            .and_then(|retread| {
                retread
                    .get("fast-tmp")
                    .or_else(|| retread.get("fast_tmp"))
            })
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
    let canonical = fs::canonicalize(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf());
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

pub fn engage_backend(workspace_root: &Path, cfg: &FastTmpConfig) -> Result<Option<EngagedFastTmp>> {
    let engaged = engage_inner(workspace_root, cfg, false)?;
    let backend = BackendEnv {
        pairs: engaged
            .as_ref()
            .map(|e| e.env.clone())
            .unwrap_or_default(),
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
    push_env_if_needed(&mut out, "RETREAD_CACHE_DIR", &ns.retread_cache_dir(), stale);
    push_env_str_if_needed(&mut out, "UV_LOCK_TIMEOUT", "1800", stale);
    push_env_if_needed(
        &mut out,
        "PIXI_DETACHED_ENVIRONMENTS",
        &ns.envs_dir(),
        stale,
    );
    push_env_str_if_needed(&mut out, "RETREAD_FAST_TMP_NS_JOB", &current_job_marker(), true);
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
    BACKEND_ENV
        .get()
        .and_then(|env| {
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
        link.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("link"),
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

pub fn run_frozen_install_if_slurm(workspace_root: &Path, env: &[(String, String)]) -> Result<()> {
    if !in_slurm_job() {
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
    ns.bld_dir()
        .join(format!("out-{token}-{}-{}", std::process::id(), unique_nonce()))
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
        dst.file_name().and_then(|s| s.to_str()).unwrap_or("artifact")
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
            .with_context(|| format!("opening copy-back corruption test hook {}", part.display()))?;
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
            return Ok((Some(bytes.saturating_mul(75) / 100), "cgroup v2 memory.max-current"));
        }
        if let Some(bytes) = slurm_memory_budget_bytes() {
            return Ok((Some(bytes.saturating_mul(75) / 100), "SLURM memory environment"));
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
        assert_eq!(fs::read_link(ws.join(".pixi").join("bld")).unwrap(), first.ns.bld_dir());
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
        let final_path = copy_back_artifacts(root.join("stage").as_path(), &out, &artifact).unwrap();
        assert_eq!(fs::read(&final_path).unwrap(), b"conda bytes");
        assert_eq!(final_path, out.join("linux-64").join("pkg-1.0-0.conda"));
        fs::remove_dir_all(root).ok();
    }
}
