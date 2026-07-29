//! Per-user, node-local compression-thread sharing for rattler-build children.
//!
//! Every coordinated build publishes a nonce-scoped lease in a local registry.
//! Registry mutations are serialized with `flock`, and every lease records its
//! immutable grant. New grants consume only unallocated tokens, so the sum of
//! coordinated live grants never exceeds the effective budget. A solo build
//! can still receive the full budget; later contenders wait for tokens rather
//! than changing the thread count of an already-running child.
//!
//! Coordination is intentionally per user, not cross-UID. The default `/tmp`
//! registry is UID-namespaced, owned by that UID, and mode 0700. Hosts running
//! builds as several users still need a higher-level node quota.

use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use std::sync::atomic::AtomicBool;

const COMPRESSION_THREADS_ENV: &str = "RETREAD_COMPRESSION_THREADS";
const COMPRESSION_BUDGET_ENV: &str = "RETREAD_COMPRESSION_BUDGET";
const THREAD_LEASE_DIR_ENV: &str = "RETREAD_THREAD_LEASE_DIR";
const FALLBACK_AVAILABLE_PARALLELISM: usize = 4;
const REGISTRY_LOCK_FILE: &str = "registry.lock";
const LEASE_FILE_PREFIX: &str = "lease-";
const LEASE_FILE_SUFFIX: &str = ".json";
const NONCE_BYTES: usize = 16;
const NONCE_HEX_LEN: usize = NONCE_BYTES * 2;
const MAX_NONCE_ATTEMPTS: usize = 16;
const MAX_LEASE_BYTES: u64 = 16 * 1024;
const LEASE_STALE_AFTER: Duration = Duration::from_secs(6 * 60 * 60);
const CAPACITY_RETRY_DELAY: Duration = Duration::from_millis(25);

static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
static FORCE_REGISTRY_LOCK_FAILURE: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompressionThreadSource {
    Override,
    BudgetShare,
    Default,
    RegistryFallback,
}

impl CompressionThreadSource {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Override => "override",
            Self::BudgetShare => "budget-share",
            Self::Default => "default",
            Self::RegistryFallback => "registry-fallback",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompressionThreadDecision {
    pub(crate) threads: NonZeroUsize,
    pub(crate) active_leases: usize,
    pub(crate) budget: NonZeroUsize,
    pub(crate) source: CompressionThreadSource,
}

#[derive(Clone)]
struct RegistryDir {
    path: PathBuf,
    dir: Arc<File>,
}

#[derive(Clone)]
struct LeaseRegistration {
    registry: RegistryDir,
    file_name: String,
    pid: u32,
    nonce: String,
}

/// A live compression-budget lease.
///
/// Call [`Self::release`] immediately after the rattler-build child exits.
/// `Drop` retries the same best-effort cleanup on spawn/wait error paths.
pub(crate) struct CompressionThreadLease {
    decision: CompressionThreadDecision,
    registration: Option<LeaseRegistration>,
    released: bool,
}

impl CompressionThreadLease {
    pub(crate) fn decision(&self) -> CompressionThreadDecision {
        self.decision
    }

    /// Attach the actual compression consumer to this lease.
    ///
    /// This is deliberately non-fatal: losing registry accounting must never
    /// turn a successfully spawned packaging child into a failed build.
    pub(crate) async fn record_child(&self, child_pid: u32) {
        let Some(registration) = self.registration.clone() else {
            return;
        };
        let registry_path = registration.registry.path.clone();
        match tokio::task::spawn_blocking(move || update_child_identity(&registration, child_pid))
            .await
        {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!(
                    error = %error,
                    child_pid,
                    lease_dir = %registry_path.display(),
                    "failed to attach rattler-build child to compression lease; continuing",
                );
            }
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    child_pid,
                    lease_dir = %registry_path.display(),
                    "compression child-tracking task failed; continuing",
                );
            }
        }
    }

    pub(crate) fn release(&mut self) {
        if self.released {
            return;
        }
        let Some(registration) = &self.registration else {
            self.released = true;
            return;
        };
        match remove_registered_lease(registration) {
            Ok(()) => self.released = true,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    pid = registration.pid,
                    nonce = %registration.nonce,
                    lease_dir = %registration.registry.path.display(),
                    "failed to remove compression thread lease",
                );
            }
        }
    }
}

impl Drop for CompressionThreadLease {
    fn drop(&mut self) {
        self.release();
    }
}

/// Acquire this build's compression budget lease.
///
/// Registry setup, locking, parsing, pruning, and persistence failures all
/// degrade to a conservative nonzero grant. They never fail packaging.
pub(crate) async fn acquire(config_threads: Option<NonZeroUsize>) -> CompressionThreadLease {
    let available_parallelism = available_parallelism();
    let settings = resolve_settings(config_threads, available_parallelism);
    if let Some(threads) = settings.thread_env_override {
        // Highest-precedence valid overrides are an explicit opt-out from
        // coordination. In particular, do not resolve or touch a registry.
        return detached_lease(threads, settings.budget, CompressionThreadSource::Override);
    }

    let fallback_budget = settings.budget;
    let fallback_config_threads = settings.config_threads;
    match tokio::task::spawn_blocking(move || acquire_resolved(settings, std::process::id())).await
    {
        Ok(lease) => lease,
        Err(error) => {
            let threads = conservative_fallback(fallback_budget, fallback_config_threads);
            tracing::warn!(
                error = %error,
                fallback_threads = threads.get(),
                budget = fallback_budget.get(),
                "compression lease task failed; continuing with conservative fallback",
            );
            detached_lease(
                threads,
                fallback_budget,
                CompressionThreadSource::RegistryFallback,
            )
        }
    }
}

fn available_parallelism() -> NonZeroUsize {
    std::thread::available_parallelism().unwrap_or_else(|error| {
        tracing::warn!(
            error = %error,
            fallback = FALLBACK_AVAILABLE_PARALLELISM,
            "could not determine available parallelism for rattler-build compression",
        );
        NonZeroUsize::new(FALLBACK_AVAILABLE_PARALLELISM)
            .expect("fallback available parallelism is nonzero")
    })
}

#[derive(Debug)]
enum NumericEnv {
    Missing,
    Valid(NonZeroUsize),
    Invalid(OsString),
}

fn read_numeric_env(name: &str) -> NumericEnv {
    let Some(raw) = std::env::var_os(name) else {
        return NumericEnv::Missing;
    };
    match raw.to_str().and_then(|value| value.parse().ok()) {
        Some(value) => NumericEnv::Valid(value),
        None => NumericEnv::Invalid(raw),
    }
}

fn warn_invalid_numeric_env(name: &str, value: &OsString) {
    tracing::warn!(
        variable = name,
        value = ?value,
        "invalid numeric environment override; expected an integer >= 1; ignoring value",
    );
}

#[derive(Debug, Clone, Copy)]
struct AcquireSettings {
    budget: NonZeroUsize,
    budget_is_overridden: bool,
    thread_env_override: Option<NonZeroUsize>,
    config_threads: Option<NonZeroUsize>,
}

fn resolve_settings(
    config_threads: Option<NonZeroUsize>,
    available_parallelism: NonZeroUsize,
) -> AcquireSettings {
    let compression_threads = read_numeric_env(COMPRESSION_THREADS_ENV);
    let compression_budget = read_numeric_env(COMPRESSION_BUDGET_ENV);

    let budget = match &compression_budget {
        NumericEnv::Valid(value) => *value,
        NumericEnv::Invalid(value) => {
            warn_invalid_numeric_env(COMPRESSION_BUDGET_ENV, value);
            available_parallelism
        }
        NumericEnv::Missing => available_parallelism,
    };
    let budget_is_overridden = matches!(compression_budget, NumericEnv::Valid(_));

    let thread_env_override = match &compression_threads {
        NumericEnv::Valid(value) => Some(*value),
        NumericEnv::Invalid(value) => {
            warn_invalid_numeric_env(COMPRESSION_THREADS_ENV, value);
            // Invalid process-wide input is absent for precedence purposes;
            // retain the lower-precedence manifest configuration.
            None
        }
        NumericEnv::Missing => None,
    };

    AcquireSettings {
        budget,
        budget_is_overridden,
        thread_env_override,
        config_threads,
    }
}

#[cfg(test)]
fn acquire_for_pid(
    config_threads: Option<NonZeroUsize>,
    available_parallelism: NonZeroUsize,
    pid: u32,
) -> CompressionThreadLease {
    let settings = resolve_settings(config_threads, available_parallelism);
    if let Some(threads) = settings.thread_env_override {
        return detached_lease(threads, settings.budget, CompressionThreadSource::Override);
    }
    acquire_resolved(settings, pid)
}

fn acquire_resolved(settings: AcquireSettings, pid: u32) -> CompressionThreadLease {
    match acquire_coordinated(settings, pid) {
        Ok(lease) => lease,
        Err(error) => {
            let threads = conservative_fallback(settings.budget, settings.config_threads);
            tracing::warn!(
                error = %error,
                pid,
                fallback_threads = threads.get(),
                budget = settings.budget.get(),
                "compression thread registry unavailable; continuing with conservative fallback",
            );
            detached_lease(
                threads,
                settings.budget,
                CompressionThreadSource::RegistryFallback,
            )
        }
    }
}

fn conservative_fallback(
    budget: NonZeroUsize,
    config_threads: Option<NonZeroUsize>,
) -> NonZeroUsize {
    let threads = (budget.get() / 8).max(1);
    let threads = config_threads.map_or(threads, |configured| threads.min(configured.get()));
    NonZeroUsize::new(threads).expect("the conservative compression fallback is nonzero")
}

fn detached_lease(
    threads: NonZeroUsize,
    budget: NonZeroUsize,
    source: CompressionThreadSource,
) -> CompressionThreadLease {
    CompressionThreadLease {
        decision: CompressionThreadDecision {
            threads,
            active_leases: 0,
            budget,
            source,
        },
        registration: None,
        released: false,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LeaseRecord {
    pid: u32,
    starttime: u64,
    granted_threads: usize,
    nonce: String,
    refreshed_at_unix_secs: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    child_pid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    child_starttime: Option<u64>,
}

struct LoadedLease {
    file_name: String,
    record: LeaseRecord,
}

enum RegistryAttempt {
    Collision,
    Waiting,
    Granted {
        threads: NonZeroUsize,
        active_leases: usize,
    },
}

#[cfg(target_os = "linux")]
fn acquire_coordinated(settings: AcquireSettings, pid: u32) -> Result<CompressionThreadLease> {
    acquire_coordinated_with_nonce_generator(settings, pid, random_nonce)
}

#[cfg(target_os = "linux")]
fn acquire_coordinated_with_nonce_generator(
    settings: AcquireSettings,
    pid: u32,
    mut nonce_generator: impl FnMut() -> Result<String>,
) -> Result<CompressionThreadLease> {
    let registry = prepare_registry_dir(&lease_registry_dir())?;
    let starttime = process_starttime(pid)?
        .ok_or_else(|| anyhow!("compression lease owner PID {pid} is not alive"))?;

    for _ in 0..MAX_NONCE_ATTEMPTS {
        let nonce = nonce_generator()?;
        let file_name = lease_file_name(pid, &nonce);
        let registration = LeaseRegistration {
            registry: registry.clone(),
            file_name,
            pid,
            nonce,
        };
        let mut registered = false;

        loop {
            match registry_admission_pass(&registration, starttime, settings, registered) {
                Ok(RegistryAttempt::Collision) => break,
                Ok(RegistryAttempt::Waiting) => {
                    registered = true;
                    std::thread::sleep(CAPACITY_RETRY_DELAY);
                }
                Ok(RegistryAttempt::Granted {
                    threads,
                    active_leases,
                }) => {
                    let source = if settings.config_threads.is_some() {
                        CompressionThreadSource::Override
                    } else if active_leases > 1 || settings.budget_is_overridden {
                        CompressionThreadSource::BudgetShare
                    } else {
                        CompressionThreadSource::Default
                    };
                    return Ok(CompressionThreadLease {
                        decision: CompressionThreadDecision {
                            threads,
                            active_leases,
                            budget: settings.budget,
                            source,
                        },
                        registration: Some(registration),
                        released: false,
                    });
                }
                Err(error) => {
                    if registered && let Err(cleanup_error) = remove_registered_lease(&registration)
                    {
                        tracing::warn!(
                            error = %cleanup_error,
                            pid,
                            nonce = %registration.nonce,
                            lease_dir = %registration.registry.path.display(),
                            "failed to roll back pending compression lease",
                        );
                    }
                    return Err(error);
                }
            }
        }
    }

    bail!("failed to allocate a unique compression lease nonce after {MAX_NONCE_ATTEMPTS} attempts")
}

#[cfg(not(target_os = "linux"))]
fn acquire_coordinated(_settings: AcquireSettings, _pid: u32) -> Result<CompressionThreadLease> {
    bail!("secure local compression lease registries are unsupported on this platform")
}

#[cfg(target_os = "linux")]
fn registry_admission_pass(
    registration: &LeaseRegistration,
    starttime: u64,
    settings: AcquireSettings,
    registered: bool,
) -> Result<RegistryAttempt> {
    with_registry_lock(&registration.registry, || {
        if !registered {
            match rustix::fs::statat(
                &*registration.registry.dir,
                &registration.file_name,
                rustix::fs::AtFlags::SYMLINK_NOFOLLOW,
            ) {
                Ok(_) => return Ok(RegistryAttempt::Collision),
                Err(rustix::io::Errno::NOENT) => {}
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "checking compression lease path {}",
                            registration
                                .registry
                                .path
                                .join(&registration.file_name)
                                .display()
                        )
                    });
                }
            }
        }
        let now = unix_now_secs()?;
        let leases = load_and_prune_leases(&registration.registry, now)?;
        let own = leases
            .iter()
            .find(|lease| lease.file_name == registration.file_name);
        if let Some(own) = own
            && own.record.granted_threads > 0
        {
            let threads = NonZeroUsize::new(own.record.granted_threads)
                .expect("a positive stored grant is nonzero");
            return Ok(RegistryAttempt::Granted {
                threads,
                active_leases: leases.len(),
            });
        }

        let active_leases = leases.len() + usize::from(own.is_none());
        let used_threads = leases.iter().try_fold(0usize, |total, lease| {
            total
                .checked_add(lease.record.granted_threads)
                .ok_or_else(|| anyhow!("compression lease grant sum overflowed"))
        })?;
        let remaining = settings.budget.get().saturating_sub(used_threads);
        let fair_share = settings.budget.get() / active_leases;
        let mut grant = remaining.min(fair_share);
        if let Some(config_threads) = settings.config_threads {
            // The manifest value remains the requested per-build ceiling, but
            // cannot bypass the aggregate node budget during contention.
            grant = grant.min(config_threads.get());
        }
        if remaining > 0 {
            grant = grant.max(1).min(remaining);
        }

        let mut record = own.map_or_else(
            || LeaseRecord {
                pid: registration.pid,
                starttime,
                granted_threads: 0,
                nonce: registration.nonce.clone(),
                refreshed_at_unix_secs: now,
                child_pid: None,
                child_starttime: None,
            },
            |lease| lease.record.clone(),
        );

        if grant == 0 {
            if own.is_none() {
                write_record_atomic(
                    &registration.registry,
                    &registration.file_name,
                    &record,
                    false,
                )?;
            }
            return Ok(RegistryAttempt::Waiting);
        }

        record.granted_threads = grant;
        record.refreshed_at_unix_secs = now;
        write_record_atomic(
            &registration.registry,
            &registration.file_name,
            &record,
            own.is_some(),
        )?;
        Ok(RegistryAttempt::Granted {
            threads: NonZeroUsize::new(grant).expect("registry grants are positive"),
            active_leases,
        })
    })
}

fn lease_registry_dir() -> PathBuf {
    if let Some(path) = std::env::var_os(THREAD_LEASE_DIR_ENV)
        && !path.is_empty()
    {
        return PathBuf::from(path);
    }
    if let Some(runtime_dir) = std::env::var_os("XDG_RUNTIME_DIR")
        && !runtime_dir.is_empty()
    {
        return PathBuf::from(runtime_dir).join("retread-thread-leases");
    }
    fallback_lease_registry_dir()
}

#[cfg(unix)]
fn fallback_lease_registry_dir() -> PathBuf {
    PathBuf::from(format!(
        "/tmp/retread-{}-thread-leases",
        nix::unistd::Uid::effective().as_raw()
    ))
}

#[cfg(not(unix))]
fn fallback_lease_registry_dir() -> PathBuf {
    std::env::temp_dir().join(format!("retread-{}-thread-leases", std::process::id()))
}

#[cfg(target_os = "linux")]
fn prepare_registry_dir(path: &Path) -> Result<RegistryDir> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};

    let mut builder = fs::DirBuilder::new();
    builder.mode(0o700);
    match builder.create(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("creating lease registry {}", path.display()));
        }
    }

    let fd = rustix::fs::open(
        path,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .with_context(|| {
        format!(
            "opening lease registry directory without following symlinks {}",
            path.display()
        )
    })?;
    let dir = File::from(fd);
    let metadata = dir
        .metadata()
        .with_context(|| format!("reading lease registry metadata {}", path.display()))?;
    let expected_uid = nix::unistd::Uid::effective().as_raw();
    let actual_mode = metadata.mode() & 0o7777;
    if metadata.uid() != expected_uid || actual_mode != 0o700 {
        bail!(
            "lease registry {} must be owned by uid {} with mode 0700; found uid {} mode {:04o}",
            path.display(),
            expected_uid,
            metadata.uid(),
            actual_mode,
        );
    }

    let filesystem = rustix::fs::fstatfs(&dir)
        .with_context(|| format!("checking lease registry filesystem {}", path.display()))?;
    let filesystem_type = filesystem.f_type as u32;
    if !filesystem_is_local(filesystem_type) {
        bail!(
            "lease registry {} is on unsupported remote or unknown filesystem type 0x{:08x}",
            path.display(),
            filesystem_type,
        );
    }

    Ok(RegistryDir {
        path: path.to_owned(),
        dir: Arc::new(dir),
    })
}

#[cfg(target_os = "linux")]
fn filesystem_is_local(filesystem_type: u32) -> bool {
    matches!(
        filesystem_type,
        0x0000_ef53 // ext2/3/4
            | 0x0102_1994 // tmpfs
            | 0x2fc1_2fc1 // zfs
            | 0x3153_464a // jfs
            | 0x5265_4973 // reiserfs
            | 0x5846_5342 // xfs
            | 0x794c_7630 // overlayfs
            | 0x8584_58f6 // ramfs
            | 0x9123_683e // btrfs
            | 0xf2f5_2010 // f2fs
    )
}

#[cfg(target_os = "linux")]
struct RegistryLock(File);

#[cfg(target_os = "linux")]
impl RegistryLock {
    fn acquire(registry: &RegistryDir) -> Result<Self> {
        let file = open_file_at(
            registry,
            REGISTRY_LOCK_FILE,
            rustix::fs::OFlags::RDWR | rustix::fs::OFlags::CREATE,
        )
        .with_context(|| {
            format!(
                "opening compression registry lock without following symlinks {}",
                registry.path.join(REGISTRY_LOCK_FILE).display()
            )
        })?;
        let metadata = file.metadata().with_context(|| {
            format!(
                "reading compression registry lock metadata {}",
                registry.path.join(REGISTRY_LOCK_FILE).display()
            )
        })?;
        if !metadata.is_file() {
            bail!(
                "compression registry lock is not a regular file: {}",
                registry.path.join(REGISTRY_LOCK_FILE).display()
            );
        }
        #[cfg(test)]
        if FORCE_REGISTRY_LOCK_FAILURE.load(Ordering::SeqCst) {
            bail!("injected compression registry lock-exclusive failure");
        }
        fs4::fs_std::FileExt::lock_exclusive(&file).with_context(|| {
            format!(
                "locking compression thread lease registry {}",
                registry.path.display()
            )
        })?;
        Ok(Self(file))
    }
}

#[cfg(target_os = "linux")]
impl Drop for RegistryLock {
    fn drop(&mut self) {
        if let Err(error) = fs4::fs_std::FileExt::unlock(&self.0) {
            tracing::warn!(
                error = %error,
                "failed to unlock compression thread lease registry",
            );
        }
    }
}

#[cfg(target_os = "linux")]
fn with_registry_lock<T>(
    registry: &RegistryDir,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let _lock = RegistryLock::acquire(registry)?;
    operation()
}

#[cfg(target_os = "linux")]
fn open_file_at(
    registry: &RegistryDir,
    file_name: &str,
    flags: rustix::fs::OFlags,
) -> Result<File> {
    let fd = rustix::fs::openat(
        &*registry.dir,
        file_name,
        flags
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::NONBLOCK
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
    )?;
    Ok(File::from(fd))
}

#[cfg(target_os = "linux")]
fn load_and_prune_leases(registry: &RegistryDir, now: u64) -> Result<Vec<LoadedLease>> {
    let mut live = Vec::new();
    let mut entries = rustix::fs::Dir::read_from(&*registry.dir)
        .with_context(|| format!("reading lease registry {}", registry.path.display()))?;
    for entry in &mut entries {
        let entry =
            entry.with_context(|| format!("reading lease entry {}", registry.path.display()))?;
        let file_name = entry
            .file_name()
            .to_str()
            .with_context(|| {
                format!(
                    "lease registry contains a non-UTF-8 entry in {}",
                    registry.path.display()
                )
            })?
            .to_owned();
        if let Some(pid) = legacy_lease_pid(&file_name) {
            if process_starttime(pid)?.is_some() {
                bail!(
                    "live legacy compression lease for PID {pid} has no recorded grant; \
                     cannot safely account aggregate tokens"
                );
            }
            unlink_file_at(registry, &file_name).with_context(|| {
                format!(
                    "removing dead legacy compression lease {}",
                    registry.path.join(&file_name).display()
                )
            })?;
            continue;
        }
        let Some(identity) = parse_lease_file_name(&file_name) else {
            continue;
        };
        let Some(record) = read_lease_record(registry, &file_name, identity)? else {
            continue;
        };
        if lease_is_live(&record, now)? {
            live.push(LoadedLease { file_name, record });
        } else {
            unlink_file_at(registry, &file_name).with_context(|| {
                format!(
                    "removing stale compression lease {}",
                    registry.path.join(&file_name).display()
                )
            })?;
        }
    }
    Ok(live)
}

fn legacy_lease_pid(file_name: &str) -> Option<u32> {
    file_name.parse().ok().filter(|pid| *pid != 0)
}

#[derive(Clone, Copy)]
struct LeaseFileIdentity<'a> {
    pid: u32,
    nonce: &'a str,
}

fn parse_lease_file_name(file_name: &str) -> Option<LeaseFileIdentity<'_>> {
    let body = file_name
        .strip_prefix(LEASE_FILE_PREFIX)?
        .strip_suffix(LEASE_FILE_SUFFIX)?;
    let (pid, nonce) = body.split_once('-')?;
    let pid = pid.parse().ok().filter(|pid| *pid != 0)?;
    if nonce.len() != NONCE_HEX_LEN || !nonce.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    Some(LeaseFileIdentity { pid, nonce })
}

#[cfg(target_os = "linux")]
fn read_lease_record(
    registry: &RegistryDir,
    file_name: &str,
    identity: LeaseFileIdentity<'_>,
) -> Result<Option<LeaseRecord>> {
    let mut file = match open_file_at(registry, file_name, rustix::fs::OFlags::RDONLY) {
        Ok(file) => file,
        Err(error) if root_io_error_kind(&error) == Some(std::io::ErrorKind::NotFound) => {
            return Ok(None);
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "opening compression lease without following symlinks {}",
                    registry.path.join(file_name).display()
                )
            });
        }
    };
    let metadata = file.metadata().with_context(|| {
        format!(
            "reading compression lease metadata {}",
            registry.path.join(file_name).display()
        )
    })?;
    if !metadata.is_file() {
        bail!(
            "compression lease is not a regular file: {}",
            registry.path.join(file_name).display()
        );
    }
    if metadata.len() > MAX_LEASE_BYTES {
        bail!(
            "compression lease exceeds {} bytes: {}",
            MAX_LEASE_BYTES,
            registry.path.join(file_name).display()
        );
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut bytes).with_context(|| {
        format!(
            "reading compression lease {}",
            registry.path.join(file_name).display()
        )
    })?;
    let record: LeaseRecord = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "parsing compression lease {}",
            registry.path.join(file_name).display()
        )
    })?;
    validate_lease_record(&record, identity).with_context(|| {
        format!(
            "validating compression lease {}",
            registry.path.join(file_name).display()
        )
    })?;
    Ok(Some(record))
}

#[cfg(target_os = "linux")]
fn root_io_error_kind(error: &anyhow::Error) -> Option<std::io::ErrorKind> {
    if error
        .chain()
        .any(|cause| cause.downcast_ref::<rustix::io::Errno>() == Some(&rustix::io::Errno::NOENT))
    {
        return Some(std::io::ErrorKind::NotFound);
    }
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<std::io::Error>())
        .map(std::io::Error::kind)
}

fn validate_lease_record(record: &LeaseRecord, identity: LeaseFileIdentity<'_>) -> Result<()> {
    if record.pid != identity.pid {
        bail!(
            "lease PID {} does not match filename PID {}",
            record.pid,
            identity.pid
        );
    }
    if record.nonce != identity.nonce {
        bail!("lease nonce does not match filename nonce");
    }
    if record.starttime == 0 {
        bail!("lease process starttime must be nonzero");
    }
    if record.refreshed_at_unix_secs == 0 {
        bail!("lease refresh timestamp must be nonzero");
    }
    match (record.child_pid, record.child_starttime) {
        (Some(pid), Some(starttime)) if pid != 0 && starttime != 0 => {}
        (None, None) => {}
        _ => bail!("lease child PID and starttime must be present together and nonzero"),
    }
    Ok(())
}

fn lease_is_live(record: &LeaseRecord, now: u64) -> Result<bool> {
    if matches!(
        process_identity_matches(record.pid, record.starttime),
        Ok(true)
    ) {
        return Ok(true);
    }
    if match (record.child_pid, record.child_starttime) {
        (Some(pid), Some(starttime)) => {
            matches!(process_identity_matches(pid, starttime), Ok(true))
        }
        (None, None) => false,
        _ => unreachable!("validated leases have paired child identity fields"),
    } {
        return Ok(true);
    }
    Ok(now.saturating_sub(record.refreshed_at_unix_secs) <= LEASE_STALE_AFTER.as_secs())
}

fn process_identity_matches(pid: u32, expected_starttime: u64) -> Result<bool> {
    Ok(process_starttime(pid)? == Some(expected_starttime))
}

#[cfg(target_os = "linux")]
fn process_starttime(pid: u32) -> Result<Option<u64>> {
    let path = PathBuf::from(format!("/proc/{pid}/stat"));
    let stat = match fs::read_to_string(&path) {
        Ok(stat) => stat,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("reading process identity {}", path.display()));
        }
    };
    // `comm` is parenthesized and may itself contain spaces or `)`. Field 3
    // begins after the final `)`, and starttime is field 22 (suffix index 19).
    let comm_end = stat.rfind(')').ok_or_else(|| {
        anyhow!("malformed process stat for PID {pid}: missing command terminator")
    })?;
    let starttime = stat[comm_end + 1..]
        .split_whitespace()
        .nth(19)
        .ok_or_else(|| anyhow!("malformed process stat for PID {pid}: missing starttime"))?
        .parse()
        .with_context(|| format!("parsing process starttime for PID {pid}"))?;
    Ok(Some(starttime))
}

#[cfg(not(target_os = "linux"))]
fn process_starttime(_pid: u32) -> Result<Option<u64>> {
    Ok(None)
}

#[cfg(target_os = "linux")]
fn write_record_atomic(
    registry: &RegistryDir,
    file_name: &str,
    record: &LeaseRecord,
    replace: bool,
) -> Result<()> {
    let temp_id = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
    let temp_name = format!(
        ".lease-tmp-{}-{}-{temp_id}",
        std::process::id(),
        record.nonce
    );
    let mut temp = open_file_at(
        registry,
        &temp_name,
        rustix::fs::OFlags::WRONLY | rustix::fs::OFlags::CREATE | rustix::fs::OFlags::EXCL,
    )
    .with_context(|| {
        format!(
            "creating temporary compression lease {}",
            registry.path.join(&temp_name).display()
        )
    })?;
    let write_result = (|| -> Result<()> {
        serde_json::to_writer(&mut temp, record).with_context(|| {
            format!(
                "serializing compression lease {}",
                registry.path.join(file_name).display()
            )
        })?;
        temp.write_all(b"\n").with_context(|| {
            format!(
                "finishing compression lease {}",
                registry.path.join(file_name).display()
            )
        })?;
        temp.sync_all().with_context(|| {
            format!(
                "syncing compression lease {}",
                registry.path.join(file_name).display()
            )
        })?;
        Ok(())
    })();
    drop(temp);
    if let Err(error) = write_result {
        let _ = unlink_file_at(registry, &temp_name);
        return Err(error);
    }

    let rename_result = if replace {
        rustix::fs::renameat(&*registry.dir, &temp_name, &*registry.dir, file_name)
    } else {
        rustix::fs::renameat_with(
            &*registry.dir,
            &temp_name,
            &*registry.dir,
            file_name,
            rustix::fs::RenameFlags::NOREPLACE,
        )
    };
    if let Err(error) = rename_result {
        let _ = unlink_file_at(registry, &temp_name);
        return Err(error).with_context(|| {
            format!(
                "atomically publishing compression lease {}",
                registry.path.join(file_name).display()
            )
        });
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn unlink_file_at(registry: &RegistryDir, file_name: &str) -> Result<()> {
    match rustix::fs::unlinkat(&*registry.dir, file_name, rustix::fs::AtFlags::empty()) {
        Ok(()) => Ok(()),
        Err(rustix::io::Errno::NOENT) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

#[cfg(target_os = "linux")]
fn update_child_identity(registration: &LeaseRegistration, child_pid: u32) -> Result<()> {
    with_registry_lock(&registration.registry, || {
        let child_starttime = process_starttime(child_pid)?
            .ok_or_else(|| anyhow!("rattler-build child PID {child_pid} already exited"))?;
        let identity = parse_lease_file_name(&registration.file_name)
            .ok_or_else(|| anyhow!("invalid owned compression lease filename"))?;
        let Some(mut record) =
            read_lease_record(&registration.registry, &registration.file_name, identity)?
        else {
            // Cancellation can release the lease while an asynchronous child
            // update is queued. Never recreate a lease after that release.
            return Ok(());
        };
        if record.pid != registration.pid || record.nonce != registration.nonce {
            bail!("owned compression lease identity changed before child update");
        }
        record.child_pid = Some(child_pid);
        record.child_starttime = Some(child_starttime);
        record.refreshed_at_unix_secs = unix_now_secs()?;
        write_record_atomic(
            &registration.registry,
            &registration.file_name,
            &record,
            true,
        )
    })
}

#[cfg(not(target_os = "linux"))]
fn update_child_identity(_registration: &LeaseRegistration, _child_pid: u32) -> Result<()> {
    bail!("compression child lease tracking is unsupported on this platform")
}

#[cfg(target_os = "linux")]
fn remove_registered_lease(registration: &LeaseRegistration) -> Result<()> {
    with_registry_lock(&registration.registry, || {
        let identity = parse_lease_file_name(&registration.file_name)
            .ok_or_else(|| anyhow!("invalid owned compression lease filename"))?;
        let Some(record) =
            read_lease_record(&registration.registry, &registration.file_name, identity)?
        else {
            return Ok(());
        };
        if record.pid != registration.pid || record.nonce != registration.nonce {
            bail!(
                "refusing to remove compression lease whose PID or nonce no longer matches owner"
            );
        }
        unlink_file_at(&registration.registry, &registration.file_name).with_context(|| {
            format!(
                "removing compression lease {}",
                registration
                    .registry
                    .path
                    .join(&registration.file_name)
                    .display()
            )
        })
    })
}

#[cfg(not(target_os = "linux"))]
fn remove_registered_lease(_registration: &LeaseRegistration) -> Result<()> {
    bail!("compression lease removal is unsupported on this platform")
}

fn lease_file_name(pid: u32, nonce: &str) -> String {
    format!("{LEASE_FILE_PREFIX}{pid}-{nonce}{LEASE_FILE_SUFFIX}")
}

fn random_nonce() -> Result<String> {
    let mut bytes = [0u8; NONCE_BYTES];
    File::open("/dev/urandom")
        .context("opening /dev/urandom for compression lease nonce")?
        .read_exact(&mut bytes)
        .context("reading compression lease nonce")?;
    let mut nonce = String::with_capacity(NONCE_HEX_LEN);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut nonce, "{byte:02x}").expect("writing to a String cannot fail");
    }
    Ok(nonce)
}

fn unix_now_secs() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_secs())
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use std::collections::HashSet;
    use std::io::{BufRead, BufReader};
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt, symlink};
    use std::process::{Child, ChildStdin, Command, Stdio};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex, MutexGuard};
    use std::thread;
    use std::time::Instant;

    use super::*;

    const MULTIPROCESS_HELPER_ENV: &str = "RETREAD_THREAD_BUDGET_RACE_HELPER";
    const RACE_EVENT_PREFIX: &str = "RETREAD_THREAD_BUDGET_ACQUIRED ";

    static ENV_MUTEX: Mutex<()> = Mutex::new(());
    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);
    static NEXT_TEST_NONCE: AtomicU64 = AtomicU64::new(1);

    struct TestEnvironment {
        root: PathBuf,
        dir: PathBuf,
        saved: Vec<(&'static str, Option<OsString>)>,
    }

    impl TestEnvironment {
        fn new(label: &str) -> (MutexGuard<'static, ()>, Self) {
            let guard = ENV_MUTEX.lock().unwrap();
            let unique = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "retread-thread-budget-{label}-{}-{nanos}-{unique}",
                std::process::id()
            ));
            create_private_dir(&root);
            let dir = root.join("registry");
            create_private_dir(&dir);
            let keys = [
                COMPRESSION_THREADS_ENV,
                COMPRESSION_BUDGET_ENV,
                THREAD_LEASE_DIR_ENV,
                MULTIPROCESS_HELPER_ENV,
            ];
            let saved = keys
                .into_iter()
                .map(|key| (key, std::env::var_os(key)))
                .collect();
            // SAFETY: every test in this module serializes mutations of these
            // feature-specific environment variables with ENV_MUTEX.
            unsafe {
                std::env::remove_var(COMPRESSION_THREADS_ENV);
                std::env::remove_var(COMPRESSION_BUDGET_ENV);
                std::env::remove_var(MULTIPROCESS_HELPER_ENV);
                std::env::set_var(THREAD_LEASE_DIR_ENV, &dir);
            }
            (guard, Self { root, dir, saved })
        }

        fn set(&self, key: &'static str, value: impl AsRef<std::ffi::OsStr>) {
            // SAFETY: the TestEnvironment's mutex guard remains live.
            unsafe { std::env::set_var(key, value) };
        }

        fn set_registry_path(&mut self, path: PathBuf) {
            self.dir = path;
            // SAFETY: the TestEnvironment's mutex guard remains live.
            unsafe { std::env::set_var(THREAD_LEASE_DIR_ENV, &self.dir) };
        }
    }

    impl Drop for TestEnvironment {
        fn drop(&mut self) {
            for (key, value) in &self.saved {
                // SAFETY: the TestEnvironment's mutex guard remains live
                // until after this value is dropped.
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(key, value),
                        None => std::env::remove_var(key),
                    }
                }
            }
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn create_private_dir(path: &Path) {
        let mut builder = fs::DirBuilder::new();
        builder.mode(0o700);
        builder.create(path).unwrap();
    }

    struct ChildGuard(Child);

    impl Drop for ChildGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    struct ForcedRegistryLockFailure;

    impl ForcedRegistryLockFailure {
        fn new() -> Self {
            FORCE_REGISTRY_LOCK_FAILURE.store(true, Ordering::SeqCst);
            Self
        }
    }

    impl Drop for ForcedRegistryLockFailure {
        fn drop(&mut self) {
            FORCE_REGISTRY_LOCK_FAILURE.store(false, Ordering::SeqCst);
        }
    }

    struct RaceChild {
        child: Child,
        input: ChildStdin,
    }

    impl Drop for RaceChild {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    #[derive(Clone)]
    struct SharedWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().write(bytes)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.0.lock().unwrap().flush()
        }
    }

    fn nonzero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    fn test_nonce() -> String {
        format!(
            "{:0width$x}",
            NEXT_TEST_NONCE.fetch_add(1, Ordering::Relaxed),
            width = NONCE_HEX_LEN
        )
    }

    fn make_record(pid: u32, starttime: u64, granted_threads: usize) -> LeaseRecord {
        LeaseRecord {
            pid,
            starttime,
            granted_threads,
            nonce: test_nonce(),
            refreshed_at_unix_secs: unix_now_secs().unwrap(),
            child_pid: None,
            child_starttime: None,
        }
    }

    fn expire_record(record: &mut LeaseRecord) {
        record.refreshed_at_unix_secs = unix_now_secs()
            .unwrap()
            .saturating_sub(LEASE_STALE_AFTER.as_secs() + 1);
    }

    fn seed_record(registry_path: &Path, record: &LeaseRecord) -> String {
        let registry = prepare_registry_dir(registry_path).unwrap();
        let file_name = lease_file_name(record.pid, &record.nonce);
        with_registry_lock(&registry, || {
            write_record_atomic(&registry, &file_name, record, false)
        })
        .unwrap();
        file_name
    }

    fn snapshot_records(registry_path: &Path) -> Vec<LeaseRecord> {
        let registry = prepare_registry_dir(registry_path).unwrap();
        with_registry_lock(&registry, || {
            Ok(load_and_prune_leases(&registry, unix_now_secs()?)?
                .into_iter()
                .map(|lease| lease.record)
                .collect())
        })
        .unwrap()
    }

    fn assert_fallback(lease: &CompressionThreadLease, budget: usize) {
        assert_eq!(
            lease.decision().threads,
            conservative_fallback(nonzero(budget), None)
        );
        assert_eq!(
            lease.decision().source,
            CompressionThreadSource::RegistryFallback
        );
        assert_eq!(lease.decision().active_leases, 0);
    }

    #[test]
    fn single_build_gets_full_budget() {
        let (_guard, _env) = TestEnvironment::new("single");
        let mut lease = acquire_for_pid(None, nonzero(60), std::process::id());
        assert_eq!(lease.decision().threads.get(), 60);
        assert_eq!(lease.decision().active_leases, 1);
        assert_eq!(lease.decision().source, CompressionThreadSource::Default);
        lease.release();
    }

    #[test]
    fn compression_budget_override_replaces_parallelism() {
        let (_guard, env) = TestEnvironment::new("budget-override");
        env.set(COMPRESSION_BUDGET_ENV, "24");

        let mut lease = acquire_for_pid(None, nonzero(60), std::process::id());
        assert_eq!(lease.decision().threads.get(), 24);
        assert_eq!(lease.decision().budget.get(), 24);
        assert_eq!(
            lease.decision().source,
            CompressionThreadSource::BudgetShare
        );
        lease.release();
    }

    #[test]
    fn invalid_thread_env_falls_through_to_manifest_config() {
        let (_guard, env) = TestEnvironment::new("invalid-precedence");
        env.set(COMPRESSION_THREADS_ENV, "not-a-number");
        env.set(COMPRESSION_BUDGET_ENV, "13");

        let warnings = Arc::new(Mutex::new(Vec::new()));
        let warning_writer = Arc::clone(&warnings);
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_writer(move || SharedWriter(Arc::clone(&warning_writer)))
            .finish();
        let mut lease = tracing::subscriber::with_default(subscriber, || {
            acquire_for_pid(Some(nonzero(9)), nonzero(60), std::process::id())
        });
        let warnings = String::from_utf8(warnings.lock().unwrap().clone()).unwrap();

        assert_eq!(lease.decision().threads.get(), 9);
        assert_eq!(lease.decision().source, CompressionThreadSource::Override);
        assert!(warnings.contains(COMPRESSION_THREADS_ENV), "{warnings}");
        lease.release();
    }

    #[test]
    fn invalid_thread_env_manifest_caps_registry_fallback() {
        let (_guard, env) = TestEnvironment::new("invalid-precedence-fallback");
        env.set(COMPRESSION_THREADS_ENV, "-1");
        fs::create_dir(env.dir.join(REGISTRY_LOCK_FILE)).unwrap();

        let mut lease = acquire_for_pid(Some(nonzero(1)), nonzero(80), std::process::id());

        assert_eq!(lease.decision().threads.get(), 1);
        assert_eq!(
            lease.decision().source,
            CompressionThreadSource::RegistryFallback
        );
        lease.release();
    }

    #[test]
    fn manifest_config_is_preserved_as_coordinated_ceiling() {
        let (_guard, _env) = TestEnvironment::new("config");
        let mut lease = acquire_for_pid(Some(nonzero(5)), nonzero(60), std::process::id());
        assert_eq!(lease.decision().threads.get(), 5);
        assert_eq!(lease.decision().source, CompressionThreadSource::Override);
        assert!(lease.registration.is_some());
        lease.release();
    }

    #[test]
    fn valid_thread_override_never_touches_registry() {
        let (_guard, mut env) = TestEnvironment::new("override-short-circuit");
        let unusable = env.root.join("not-a-directory");
        fs::write(&unusable, b"sentinel").unwrap();
        env.set_registry_path(unusable.clone());
        env.set(COMPRESSION_THREADS_ENV, "7");
        env.set(COMPRESSION_BUDGET_ENV, "2");

        let mut lease = acquire_for_pid(Some(nonzero(3)), nonzero(60), std::process::id());

        assert_eq!(lease.decision().threads.get(), 7);
        assert_eq!(lease.decision().budget.get(), 2);
        assert_eq!(lease.decision().source, CompressionThreadSource::Override);
        assert!(lease.registration.is_none());
        assert_eq!(fs::read(&unusable).unwrap(), b"sentinel");
        lease.release();
    }

    #[test]
    fn unwritable_registry_uses_conservative_fallback() {
        let (_guard, env) = TestEnvironment::new("unwritable");
        fs::set_permissions(&env.dir, fs::Permissions::from_mode(0o500)).unwrap();

        let mut lease = acquire_for_pid(None, nonzero(40), std::process::id());

        assert_fallback(&lease, 40);
        fs::set_permissions(&env.dir, fs::Permissions::from_mode(0o700)).unwrap();
        lease.release();
    }

    #[test]
    fn malformed_lease_file_uses_conservative_fallback() {
        let (_guard, env) = TestEnvironment::new("malformed");
        let file_name = lease_file_name(std::process::id(), &test_nonce());
        fs::write(env.dir.join(file_name), b"{ definitely not valid json").unwrap();

        let mut lease = acquire_for_pid(None, nonzero(40), std::process::id());

        assert_fallback(&lease, 40);
        lease.release();
    }

    #[test]
    fn lock_acquisition_failure_uses_conservative_fallback() {
        let (_guard, env) = TestEnvironment::new("lock-failure");
        fs::create_dir(env.dir.join(REGISTRY_LOCK_FILE)).unwrap();

        let mut lease = acquire_for_pid(None, nonzero(40), std::process::id());

        assert_fallback(&lease, 40);
        lease.release();
    }

    #[test]
    fn exclusive_lock_failure_uses_conservative_fallback() {
        let (_guard, _env) = TestEnvironment::new("exclusive-lock-failure");
        let _forced_failure = ForcedRegistryLockFailure::new();

        let mut lease = acquire_for_pid(None, nonzero(40), std::process::id());

        assert_fallback(&lease, 40);
        lease.release();
    }

    #[test]
    fn symlinked_registry_directory_is_rejected_without_following() {
        let (_guard, mut env) = TestEnvironment::new("registry-symlink");
        fs::remove_dir(&env.dir).unwrap();
        let attacker_dir = env.root.join("attacker");
        create_private_dir(&attacker_dir);
        let registry_link = env.root.join("registry-link");
        symlink(&attacker_dir, &registry_link).unwrap();
        env.set_registry_path(registry_link);

        let mut lease = acquire_for_pid(None, nonzero(40), std::process::id());

        assert_fallback(&lease, 40);
        assert_eq!(fs::read_dir(&attacker_dir).unwrap().count(), 0);
        lease.release();
    }

    #[test]
    fn symlinked_lock_file_is_rejected_without_truncating_target() {
        let (_guard, env) = TestEnvironment::new("lock-symlink");
        let sentinel = env.root.join("lock-target");
        fs::write(&sentinel, b"do not truncate").unwrap();
        symlink(&sentinel, env.dir.join(REGISTRY_LOCK_FILE)).unwrap();

        let mut lease = acquire_for_pid(None, nonzero(40), std::process::id());

        assert_fallback(&lease, 40);
        assert_eq!(fs::read(&sentinel).unwrap(), b"do not truncate");
        lease.release();
    }

    #[test]
    fn symlinked_lease_file_is_rejected_without_truncating_target() {
        let (_guard, env) = TestEnvironment::new("lease-symlink");
        let sentinel = env.root.join("lease-target");
        fs::write(&sentinel, b"do not truncate").unwrap();
        let file_name = lease_file_name(std::process::id(), &test_nonce());
        symlink(&sentinel, env.dir.join(file_name)).unwrap();

        let mut lease = acquire_for_pid(None, nonzero(40), std::process::id());

        assert_fallback(&lease, 40);
        assert_eq!(fs::read(&sentinel).unwrap(), b"do not truncate");
        lease.release();
    }

    #[test]
    fn lease_shaped_fifo_uses_fallback_without_blocking() {
        let (_guard, env) = TestEnvironment::new("lease-fifo");
        let registry = prepare_registry_dir(&env.dir).unwrap();
        let file_name = lease_file_name(std::process::id(), &test_nonce());
        rustix::fs::mkfifoat(
            &*registry.dir,
            &file_name,
            rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
        )
        .unwrap();
        let started = Instant::now();

        let mut lease = acquire_for_pid(None, nonzero(40), std::process::id());

        assert_fallback(&lease, 40);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "opening a lease-shaped FIFO blocked"
        );
        lease.release();
    }

    #[test]
    fn non_private_registry_mode_is_rejected() {
        let (_guard, env) = TestEnvironment::new("registry-mode");
        fs::set_permissions(&env.dir, fs::Permissions::from_mode(0o755)).unwrap();

        let mut lease = acquire_for_pid(None, nonzero(40), std::process::id());

        assert_fallback(&lease, 40);
        fs::set_permissions(&env.dir, fs::Permissions::from_mode(0o700)).unwrap();
        lease.release();
    }

    #[test]
    fn remote_and_unknown_filesystem_types_are_rejected() {
        assert!(!filesystem_is_local(0x0000_6969), "NFS");
        assert!(!filesystem_is_local(0x0bd0_0bd0), "Lustre");
        assert!(!filesystem_is_local(0xff53_4d42), "CIFS");
        assert!(!filesystem_is_local(0x00c3_6400), "Ceph");
        assert!(!filesystem_is_local(0x0102_1997), "9p");
        assert!(!filesystem_is_local(0x6573_5546), "FUSE");
        assert!(!filesystem_is_local(0xdead_beef), "unknown");
        assert!(filesystem_is_local(0x0102_1994), "tmpfs");
        assert!(filesystem_is_local(0x0000_ef53), "ext");
        assert!(filesystem_is_local(0x794c_7630), "overlayfs");
    }

    #[test]
    fn remote_override_registry_uses_fallback_when_test_fs_is_remote() {
        let (_guard, mut env) = TestEnvironment::new("remote-override");
        let current_dir = std::env::current_dir().unwrap();
        let current_fd = rustix::fs::open(
            &current_dir,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )
        .unwrap();
        let filesystem_type = rustix::fs::fstatfs(&current_fd).unwrap().f_type as u32;
        if filesystem_is_local(filesystem_type) {
            return;
        }
        let remote_dir = current_dir.join(format!(
            ".retread-thread-budget-remote-test-{}-{}",
            std::process::id(),
            NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        create_private_dir(&remote_dir);
        env.set_registry_path(remote_dir.clone());

        let mut lease = acquire_for_pid(None, nonzero(40), std::process::id());

        assert_fallback(&lease, 40);
        lease.release();
        fs::remove_dir(remote_dir).unwrap();
    }

    #[test]
    fn dead_pid_lease_is_retained_until_ttl() {
        let (_guard, env) = TestEnvironment::new("dead-pid");
        let mut child = Command::new("sleep").arg("30").spawn().unwrap();
        let pid = child.id();
        let starttime = process_starttime(pid).unwrap().unwrap();
        child.kill().unwrap();
        child.wait().unwrap();
        assert_eq!(process_starttime(pid).unwrap(), None);
        let stale_name = seed_record(&env.dir, &make_record(pid, starttime, 2));

        let mut lease = acquire_for_pid(Some(nonzero(1)), nonzero(8), std::process::id());

        assert!(env.dir.join(stale_name).is_file());
        assert_eq!(lease.decision().active_leases, 2);
        lease.release();
    }

    #[test]
    fn lease_older_than_ttl_with_dead_identity_is_pruned() {
        let (_guard, env) = TestEnvironment::new("expired-dead-pid");
        let mut child = Command::new("sleep").arg("30").spawn().unwrap();
        let pid = child.id();
        let starttime = process_starttime(pid).unwrap().unwrap();
        child.kill().unwrap();
        child.wait().unwrap();
        assert_eq!(process_starttime(pid).unwrap(), None);
        let mut record = make_record(pid, starttime, 2);
        expire_record(&mut record);
        let stale_name = seed_record(&env.dir, &record);

        let mut lease = acquire_for_pid(Some(nonzero(1)), nonzero(8), std::process::id());

        assert!(!env.dir.join(stale_name).exists());
        assert_eq!(lease.decision().active_leases, 1);
        lease.release();
    }

    #[test]
    fn dead_legacy_pid_lease_is_pruned() {
        let (_guard, env) = TestEnvironment::new("dead-legacy-pid");
        let mut child = Command::new("sleep").arg("30").spawn().unwrap();
        let pid = child.id();
        child.kill().unwrap();
        child.wait().unwrap();
        assert_eq!(process_starttime(pid).unwrap(), None);
        let legacy_path = env.dir.join(pid.to_string());
        fs::write(&legacy_path, b"").unwrap();

        let mut lease = acquire_for_pid(Some(nonzero(1)), nonzero(8), std::process::id());

        assert!(!legacy_path.exists());
        assert_eq!(lease.decision().active_leases, 1);
        lease.release();
    }

    #[test]
    fn live_legacy_pid_lease_uses_conservative_fallback() {
        let (_guard, env) = TestEnvironment::new("live-legacy-pid");
        let legacy_path = env.dir.join(std::process::id().to_string());
        fs::write(&legacy_path, b"").unwrap();

        let mut lease = acquire_for_pid(None, nonzero(40), std::process::id());

        assert_fallback(&lease, 40);
        assert!(legacy_path.is_file());
        lease.release();
    }

    #[test]
    fn lease_older_than_ttl_with_wrong_starttime_is_pruned() {
        let (_guard, env) = TestEnvironment::new("wrong-starttime");
        let pid = std::process::id();
        let starttime = process_starttime(pid).unwrap().unwrap();
        let mut record = make_record(pid, starttime.saturating_add(1), 2);
        expire_record(&mut record);
        let stale_name = seed_record(&env.dir, &record);

        let mut lease = acquire_for_pid(Some(nonzero(1)), nonzero(8), std::process::id());

        assert!(!env.dir.join(stale_name).exists());
        assert_eq!(lease.decision().active_leases, 1);
        lease.release();
    }

    #[test]
    fn lease_older_than_ttl_with_live_validated_identity_survives_pruning() {
        let (_guard, env) = TestEnvironment::new("expired");
        let pid = std::process::id();
        let starttime = process_starttime(pid).unwrap().unwrap();
        let mut record = make_record(pid, starttime, 2);
        expire_record(&mut record);
        let live_name = seed_record(&env.dir, &record);

        let mut lease = acquire_for_pid(Some(nonzero(1)), nonzero(8), std::process::id());

        assert!(env.dir.join(live_name).is_file());
        assert_eq!(lease.decision().active_leases, 2);
        lease.release();
    }

    #[test]
    fn lease_records_pid_starttime_grant_and_nonce() {
        let (_guard, _env) = TestEnvironment::new("contents");
        let mut lease = acquire_for_pid(Some(nonzero(3)), nonzero(16), std::process::id());
        let registration = lease.registration.as_ref().unwrap();
        let identity = parse_lease_file_name(&registration.file_name).unwrap();
        let record = read_lease_record(&registration.registry, &registration.file_name, identity)
            .unwrap()
            .unwrap();

        assert_eq!(record.pid, std::process::id());
        assert_eq!(
            record.starttime,
            process_starttime(std::process::id()).unwrap().unwrap()
        );
        assert_eq!(record.granted_threads, 3);
        assert_eq!(record.nonce, registration.nonce);
        assert_eq!(record.nonce.len(), NONCE_HEX_LEN);
        lease.release();
    }

    #[test]
    fn same_pid_leases_have_distinct_nonces_and_release_independently() {
        let (_guard, env) = TestEnvironment::new("same-pid");
        let mut first = acquire_for_pid(Some(nonzero(5)), nonzero(60), std::process::id());
        let mut second = acquire_for_pid(Some(nonzero(5)), nonzero(60), std::process::id());
        let first_name = first.registration.as_ref().unwrap().file_name.clone();
        let second_name = second.registration.as_ref().unwrap().file_name.clone();

        assert_ne!(first_name, second_name);
        assert!(env.dir.join(&first_name).is_file());
        assert!(env.dir.join(&second_name).is_file());
        first.release();
        assert!(!env.dir.join(first_name).exists());
        assert!(env.dir.join(&second_name).is_file());
        second.release();
        assert!(!env.dir.join(second_name).exists());
    }

    #[test]
    fn nonce_collision_retries_with_new_independent_lease() {
        let (_guard, env) = TestEnvironment::new("nonce-collision");
        let pid = std::process::id();
        let starttime = process_starttime(pid).unwrap().unwrap();
        let existing_record = make_record(pid, starttime, 2);
        let collision_nonce = existing_record.nonce.clone();
        let existing_name = seed_record(&env.dir, &existing_record);
        let new_nonce = test_nonce();
        let mut nonces = [collision_nonce, new_nonce.clone()].into_iter();
        let settings = resolve_settings(Some(nonzero(3)), nonzero(8));

        let mut lease = acquire_coordinated_with_nonce_generator(settings, pid, || {
            Ok(nonces.next().expect("nonce retry should be bounded"))
        })
        .unwrap();
        let new_name = lease.registration.as_ref().unwrap().file_name.clone();
        let records = snapshot_records(&env.dir);

        assert_eq!(lease.registration.as_ref().unwrap().nonce, new_nonce);
        assert_ne!(new_name, existing_name);
        assert!(env.dir.join(&existing_name).is_file());
        assert!(env.dir.join(&new_name).is_file());
        assert_eq!(records.len(), 2);
        assert_eq!(
            records
                .iter()
                .find(|record| record.nonce == existing_record.nonce)
                .unwrap()
                .granted_threads,
            2
        );
        assert_eq!(
            records
                .iter()
                .find(|record| record.nonce == new_nonce)
                .unwrap()
                .granted_threads,
            3
        );

        lease.release();

        assert!(env.dir.join(&existing_name).is_file());
        assert!(!env.dir.join(new_name).exists());
        let records = snapshot_records(&env.dir);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].nonce, existing_record.nonce);
        assert_eq!(records[0].granted_threads, 2);
    }

    #[test]
    fn record_child_persists_pid_and_starttime() {
        let (_guard, _env) = TestEnvironment::new("record-child");
        let mut lease = acquire_for_pid(Some(nonzero(2)), nonzero(16), std::process::id());
        let child = ChildGuard(Command::new("sleep").arg("30").spawn().unwrap());
        let child_pid = child.0.id();
        update_child_identity(lease.registration.as_ref().unwrap(), child_pid).unwrap();
        let registration = lease.registration.as_ref().unwrap();
        let identity = parse_lease_file_name(&registration.file_name).unwrap();
        let record = read_lease_record(&registration.registry, &registration.file_name, identity)
            .unwrap()
            .unwrap();

        assert_eq!(record.child_pid, Some(child_pid));
        assert_eq!(
            record.child_starttime,
            process_starttime(child_pid).unwrap()
        );
        lease.release();
    }

    #[test]
    fn lease_older_than_ttl_with_live_child_survives_when_parent_identity_is_stale() {
        let (_guard, env) = TestEnvironment::new("orphan-child");
        let child = ChildGuard(Command::new("sleep").arg("30").spawn().unwrap());
        let child_pid = child.0.id();
        let mut record = make_record(
            std::process::id(),
            process_starttime(std::process::id())
                .unwrap()
                .unwrap()
                .saturating_add(1),
            2,
        );
        record.child_pid = Some(child_pid);
        record.child_starttime = process_starttime(child_pid).unwrap();
        expire_record(&mut record);
        let live_name = seed_record(&env.dir, &record);

        let mut lease = acquire_for_pid(Some(nonzero(1)), nonzero(8), std::process::id());

        assert!(env.dir.join(live_name).is_file());
        assert_eq!(lease.decision().active_leases, 2);
        lease.release();
    }

    #[test]
    fn expired_wrong_child_starttime_is_pruned_when_parent_identity_is_stale() {
        let (_guard, env) = TestEnvironment::new("wrong-child-starttime");
        let child = ChildGuard(Command::new("sleep").arg("30").spawn().unwrap());
        let child_pid = child.0.id();
        let mut record = make_record(
            std::process::id(),
            process_starttime(std::process::id())
                .unwrap()
                .unwrap()
                .saturating_add(1),
            2,
        );
        record.child_pid = Some(child_pid);
        record.child_starttime = process_starttime(child_pid)
            .unwrap()
            .map(|starttime| starttime.saturating_add(1));
        expire_record(&mut record);
        let stale_name = seed_record(&env.dir, &record);

        let mut lease = acquire_for_pid(Some(nonzero(1)), nonzero(8), std::process::id());

        assert!(!env.dir.join(stale_name).exists());
        assert_eq!(lease.decision().active_leases, 1);
        lease.release();
    }

    #[test]
    fn child_update_does_not_resurrect_released_lease() {
        let (_guard, env) = TestEnvironment::new("no-resurrection");
        let mut lease = acquire_for_pid(Some(nonzero(2)), nonzero(16), std::process::id());
        let registration = lease.registration.as_ref().unwrap().clone();
        let file_name = registration.file_name.clone();
        lease.release();
        let child = ChildGuard(Command::new("sleep").arg("30").spawn().unwrap());

        update_child_identity(&registration, child.0.id()).unwrap();

        assert!(!env.dir.join(file_name).exists());
    }

    #[test]
    fn unseeded_multiprocess_acquisitions_never_exceed_budget() {
        const PROCESS_COUNT: usize = 6;
        const BUDGET: usize = 12;

        let (_guard, env) = TestEnvironment::new("multiprocess-race");
        let executable = std::env::current_exe().unwrap();
        let (event_tx, event_rx) = mpsc::channel();
        let mut readers = Vec::new();
        let mut children = Vec::new();

        for index in 0..PROCESS_COUNT {
            let mut child = Command::new(&executable)
                .args([
                    "--exact",
                    "thread_budget::tests::unseeded_multiprocess_acquire_helper",
                    "--nocapture",
                ])
                .env(MULTIPROCESS_HELPER_ENV, "1")
                .env(THREAD_LEASE_DIR_ENV, &env.dir)
                .env(COMPRESSION_BUDGET_ENV, BUDGET.to_string())
                .env_remove(COMPRESSION_THREADS_ENV)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap();
            let input = child.stdin.take().unwrap();
            let output = child.stdout.take().unwrap();
            let event_tx = event_tx.clone();
            readers.push(thread::spawn(move || {
                let mut sent = false;
                for line in BufReader::new(output).lines() {
                    let line = line.unwrap();
                    if !sent && let Some(grant) = line.strip_prefix(RACE_EVENT_PREFIX) {
                        event_tx
                            .send((index, grant.parse::<usize>().unwrap()))
                            .unwrap();
                        sent = true;
                    }
                }
            }));
            children.push(RaceChild { child, input });
        }
        drop(event_tx);

        for child in &mut children {
            child.input.write_all(b"start\n").unwrap();
            child.input.flush().unwrap();
        }

        let (first_index, first_grant) = event_rx.recv_timeout(Duration::from_secs(30)).unwrap();
        assert_eq!(first_grant, BUDGET);

        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let records = snapshot_records(&env.dir);
            let total = records
                .iter()
                .map(|record| record.granted_threads)
                .sum::<usize>();
            assert!(total <= BUDGET, "{total} > {BUDGET}");
            if records.len() == PROCESS_COUNT {
                break;
            }
            assert!(Instant::now() < deadline, "contenders did not all register");
            thread::sleep(Duration::from_millis(10));
        }
        children[first_index].input.write_all(b"release\n").unwrap();
        children[first_index].input.flush().unwrap();

        let mut acquired = HashSet::from([first_index]);
        while acquired.len() < PROCESS_COUNT {
            let (index, grant) = event_rx.recv_timeout(Duration::from_secs(30)).unwrap();
            assert!(
                acquired.insert(index),
                "duplicate acquisition from child {index}"
            );
            assert!(grant > 0);
            let records = snapshot_records(&env.dir);
            let total = records
                .iter()
                .map(|record| record.granted_threads)
                .sum::<usize>();
            assert!(
                total <= BUDGET,
                "aggregate grant {total} exceeded budget {BUDGET} after child {index} acquired"
            );
        }

        let final_records = snapshot_records(&env.dir);
        assert!(
            final_records
                .iter()
                .filter(|record| record.granted_threads > 0)
                .count()
                >= 2,
            "the race never exercised concurrent positive grants"
        );
        assert!(
            final_records
                .iter()
                .map(|record| record.granted_threads)
                .sum::<usize>()
                <= BUDGET
        );

        for (index, child) in children.iter_mut().enumerate() {
            if index != first_index {
                child.input.write_all(b"release\n").unwrap();
                child.input.flush().unwrap();
            }
        }
        for child in &mut children {
            assert!(child.child.wait().unwrap().success());
        }
        for reader in readers {
            reader.join().unwrap();
        }
    }

    #[test]
    fn unseeded_multiprocess_acquire_helper() {
        if std::env::var_os(MULTIPROCESS_HELPER_ENV).is_none() {
            return;
        }
        let stdin = std::io::stdin();
        let mut commands = stdin.lock().lines();
        assert_eq!(commands.next().unwrap().unwrap(), "start");
        let budget = std::env::var(COMPRESSION_BUDGET_ENV)
            .unwrap()
            .parse()
            .unwrap();
        let mut lease = acquire_for_pid(None, nonzero(budget), std::process::id());
        println!("{RACE_EVENT_PREFIX}{}", lease.decision().threads.get());
        std::io::stdout().flush().unwrap();
        assert_eq!(commands.next().unwrap().unwrap(), "release");
        lease.release();
    }
}
