//! Node-wide compression-thread sharing for rattler-build subprocesses.
//!
//! Every active build publishes its backend's PID-named lease in a node-local
//! registry. Registry mutations are serialized with `flock`, so each build
//! can prune dead owners and take an equal share of the effective node budget
//! immediately before it spawns rattler-build.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const COMPRESSION_THREADS_ENV: &str = "RETREAD_COMPRESSION_THREADS";
const COMPRESSION_BUDGET_ENV: &str = "RETREAD_COMPRESSION_BUDGET";
const THREAD_LEASE_DIR_ENV: &str = "RETREAD_THREAD_LEASE_DIR";
const FALLBACK_AVAILABLE_PARALLELISM: usize = 4;
const REGISTRY_LOCK_FILE: &str = "registry.lock";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompressionThreadSource {
    Override,
    BudgetShare,
    Default,
}

impl CompressionThreadSource {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Override => "override",
            Self::BudgetShare => "budget-share",
            Self::Default => "default",
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

/// A live compression-budget lease.
///
/// Call [`Self::release`] immediately after the rattler-build child exits.
/// `Drop` retries the same best-effort cleanup on spawn/wait error paths.
pub(crate) struct CompressionThreadLease {
    decision: CompressionThreadDecision,
    registry_dir: PathBuf,
    pid: u32,
    released: bool,
}

impl CompressionThreadLease {
    pub(crate) fn decision(&self) -> CompressionThreadDecision {
        self.decision
    }

    pub(crate) fn release(&mut self) {
        if self.released {
            return;
        }
        match remove_lease(&self.registry_dir, self.pid) {
            Ok(()) => self.released = true,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    pid = self.pid,
                    lease_dir = %self.registry_dir.display(),
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

/// Acquire this build's node-wide compression budget lease.
pub(crate) async fn acquire(
    config_threads: Option<NonZeroUsize>,
) -> Result<CompressionThreadLease> {
    tokio::task::spawn_blocking(move || {
        let available_parallelism = std::thread::available_parallelism().unwrap_or_else(|error| {
            tracing::warn!(
                error = %error,
                fallback = FALLBACK_AVAILABLE_PARALLELISM,
                "could not determine available parallelism for rattler-build compression",
            );
            NonZeroUsize::new(FALLBACK_AVAILABLE_PARALLELISM)
                .expect("fallback available parallelism is nonzero")
        });
        acquire_for_pid(config_threads, available_parallelism, std::process::id())
    })
    .await
    .context("compression thread lease task panicked")?
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
        "invalid numeric environment override; expected an integer >= 1; using default",
    );
}

fn acquire_for_pid(
    config_threads: Option<NonZeroUsize>,
    available_parallelism: NonZeroUsize,
    pid: u32,
) -> Result<CompressionThreadLease> {
    acquire_for_pid_with_hook(config_threads, available_parallelism, pid, || {})
}

fn acquire_for_pid_with_hook(
    config_threads: Option<NonZeroUsize>,
    available_parallelism: NonZeroUsize,
    pid: u32,
    after_registry_lock: impl FnOnce(),
) -> Result<CompressionThreadLease> {
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

    // An invalid highest-precedence thread override deliberately falls back
    // to automatic budget sharing rather than activating a lower-precedence
    // manifest override.
    let fixed_threads = match &compression_threads {
        NumericEnv::Valid(value) => Some(*value),
        NumericEnv::Invalid(value) => {
            warn_invalid_numeric_env(COMPRESSION_THREADS_ENV, value);
            None
        }
        NumericEnv::Missing => config_threads,
    };

    let registry_dir = lease_registry_dir();
    let active_leases = registry_pass_with_hook(&registry_dir, pid, after_registry_lock)?;
    let (threads, source) = fixed_threads.map_or_else(
        || {
            (
                shared_threads(budget, active_leases),
                if active_leases > 1 || budget_is_overridden {
                    CompressionThreadSource::BudgetShare
                } else {
                    CompressionThreadSource::Default
                },
            )
        },
        |threads| (threads, CompressionThreadSource::Override),
    );

    Ok(CompressionThreadLease {
        decision: CompressionThreadDecision {
            threads,
            active_leases,
            budget,
            source,
        },
        registry_dir,
        pid,
        released: false,
    })
}

fn shared_threads(budget: NonZeroUsize, active_leases: usize) -> NonZeroUsize {
    let active_leases = active_leases.max(1);
    NonZeroUsize::new((budget.get() / active_leases).max(1))
        .expect("the compression thread share is always at least one")
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
        nix::unistd::Uid::current().as_raw()
    ))
}

#[cfg(not(unix))]
fn fallback_lease_registry_dir() -> PathBuf {
    std::env::temp_dir().join(format!("retread-{}-thread-leases", std::process::id()))
}

struct RegistryLock(File);

impl RegistryLock {
    fn acquire(registry_dir: &Path) -> Result<Self> {
        fs::create_dir_all(registry_dir).with_context(|| {
            format!(
                "creating compression thread lease registry {}",
                registry_dir.display()
            )
        })?;
        let lock_path = registry_dir.join(REGISTRY_LOCK_FILE);
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .with_context(|| {
                format!(
                    "opening compression thread lease registry lock {}",
                    lock_path.display()
                )
            })?;
        fs4::fs_std::FileExt::lock_exclusive(&file).with_context(|| {
            format!(
                "locking compression thread lease registry {}",
                lock_path.display()
            )
        })?;
        Ok(Self(file))
    }
}

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

fn with_registry_lock<T>(registry_dir: &Path, operation: impl FnOnce() -> Result<T>) -> Result<T> {
    let _lock = RegistryLock::acquire(registry_dir)?;
    operation()
}

fn registry_pass_with_hook(
    registry_dir: &Path,
    pid: u32,
    after_lock: impl FnOnce(),
) -> Result<usize> {
    with_registry_lock(registry_dir, || {
        after_lock();
        prune_stale_leases(registry_dir)?;
        refresh_lease(registry_dir, pid)?;
        match count_active_leases(registry_dir) {
            Ok(active_leases) => Ok(active_leases),
            Err(error) => {
                let lease_path = registry_dir.join(pid.to_string());
                if let Err(cleanup_error) = fs::remove_file(&lease_path)
                    && cleanup_error.kind() != std::io::ErrorKind::NotFound
                {
                    tracing::warn!(
                        error = %cleanup_error,
                        path = %lease_path.display(),
                        "failed to roll back compression lease after registry error",
                    );
                }
                Err(error)
            }
        }
    })
}

fn prune_stale_leases(registry_dir: &Path) -> Result<()> {
    for entry in fs::read_dir(registry_dir).with_context(|| {
        format!(
            "reading compression thread lease registry {}",
            registry_dir.display()
        )
    })? {
        let entry = entry.with_context(|| {
            format!(
                "reading entry in compression thread lease registry {}",
                registry_dir.display()
            )
        })?;
        if !entry
            .file_type()
            .with_context(|| format!("reading lease type {}", entry.path().display()))?
            .is_file()
        {
            continue;
        }
        let Some(pid) = lease_pid(&entry.file_name()) else {
            continue;
        };
        if pid_is_alive(pid) {
            continue;
        }
        match fs::remove_file(entry.path()) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("removing stale compression lease for PID {pid}"));
            }
        }
    }
    Ok(())
}

fn refresh_lease(registry_dir: &Path, pid: u32) -> Result<()> {
    let lease_path = registry_dir.join(pid.to_string());
    OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&lease_path)
        .with_context(|| format!("creating compression thread lease {}", lease_path.display()))?;
    Ok(())
}

fn count_active_leases(registry_dir: &Path) -> Result<usize> {
    let mut active = 0;
    for entry in fs::read_dir(registry_dir).with_context(|| {
        format!(
            "counting compression thread leases in {}",
            registry_dir.display()
        )
    })? {
        let entry = entry.with_context(|| {
            format!(
                "reading entry in compression thread lease registry {}",
                registry_dir.display()
            )
        })?;
        if entry
            .file_type()
            .with_context(|| format!("reading lease type {}", entry.path().display()))?
            .is_file()
            && lease_pid(&entry.file_name()).is_some()
        {
            active += 1;
        }
    }
    Ok(active)
}

fn lease_pid(file_name: &std::ffi::OsStr) -> Option<u32> {
    file_name.to_str()?.parse().ok().filter(|pid| *pid != 0)
}

#[cfg(unix)]
fn pid_is_alive(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    match nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None) {
        Ok(()) | Err(nix::errno::Errno::EPERM) => true,
        Err(nix::errno::Errno::ESRCH) => false,
        Err(error) => {
            tracing::warn!(
                error = %error,
                pid,
                "could not check compression lease PID; retaining lease",
            );
            true
        }
    }
}

#[cfg(not(unix))]
fn pid_is_alive(pid: u32) -> bool {
    pid == std::process::id()
}

fn remove_lease(registry_dir: &Path, pid: u32) -> Result<()> {
    with_registry_lock(registry_dir, || {
        let lease_path = registry_dir.join(pid.to_string());
        match fs::remove_file(&lease_path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error)
                .with_context(|| format!("removing compression lease {}", lease_path.display())),
        }
    })
}

#[cfg(all(test, unix))]
mod tests {
    use std::io::Write;
    use std::process::{Child, Command};
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier, Mutex, MutexGuard};
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use super::*;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());
    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    struct TestEnvironment {
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
            let dir = std::env::temp_dir().join(format!(
                "retread-thread-budget-{label}-{}-{nanos}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&dir).unwrap();
            let keys = [
                COMPRESSION_THREADS_ENV,
                COMPRESSION_BUDGET_ENV,
                THREAD_LEASE_DIR_ENV,
            ];
            let saved = keys
                .into_iter()
                .map(|key| (key, std::env::var_os(key)))
                .collect();
            // SAFETY: all tests in this module serialize mutations of these
            // feature-specific environment variables with ENV_MUTEX.
            unsafe {
                std::env::remove_var(COMPRESSION_THREADS_ENV);
                std::env::remove_var(COMPRESSION_BUDGET_ENV);
                std::env::set_var(THREAD_LEASE_DIR_ENV, &dir);
            }
            (guard, Self { dir, saved })
        }

        fn set(&self, key: &'static str, value: &str) {
            // SAFETY: the TestEnvironment's mutex guard remains live.
            unsafe { std::env::set_var(key, value) };
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
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    struct ChildGuard(Child);

    impl Drop for ChildGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
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

    fn seed_live_lease(registry_dir: &Path, pid: u32) {
        with_registry_lock(registry_dir, || refresh_lease(registry_dir, pid)).unwrap();
    }

    #[test]
    fn share_math_single_build_gets_full_budget() {
        let (_guard, _env) = TestEnvironment::new("share-one");
        assert_eq!(shared_threads(nonzero(60), 1).get(), 60);
    }

    #[test]
    fn share_math_six_builds_divides_budget() {
        let (_guard, _env) = TestEnvironment::new("share-six");
        assert_eq!(shared_threads(nonzero(60), 6).get(), 10);
    }

    #[test]
    fn share_math_never_returns_zero() {
        let (_guard, _env) = TestEnvironment::new("share-floor");
        assert_eq!(shared_threads(nonzero(2), 6).get(), 1);
    }

    #[test]
    fn stale_dead_pid_lease_is_pruned() {
        let (_guard, env) = TestEnvironment::new("stale");
        let dead_pid = (0..16)
            .find_map(|_| {
                let mut child = Command::new("sh").args(["-c", "exit 0"]).spawn().unwrap();
                let pid = child.id();
                child.wait().unwrap();
                (!pid_is_alive(pid)).then_some(pid)
            })
            .expect("a reaped child PID should be dead");
        seed_live_lease(&env.dir, dead_pid);
        let stale_path = env.dir.join(dead_pid.to_string());

        let mut lease =
            acquire_for_pid(None, nonzero(8), std::process::id()).expect("acquire lease");

        assert!(!stale_path.exists());
        assert_eq!(lease.decision().active_leases, 1);
        lease.release();
    }

    #[test]
    fn compression_threads_override_bypasses_budget_math() {
        let (_guard, env) = TestEnvironment::new("threads-override");
        env.set(COMPRESSION_THREADS_ENV, "7");
        env.set(COMPRESSION_BUDGET_ENV, "2");

        let mut lease = acquire_for_pid(Some(nonzero(3)), nonzero(60), std::process::id())
            .expect("acquire lease");
        let decision = lease.decision();

        assert_eq!(decision.threads.get(), 7);
        assert_eq!(decision.budget.get(), 2);
        assert_eq!(decision.source, CompressionThreadSource::Override);
        lease.release();
    }

    #[test]
    fn compression_budget_override_replaces_parallelism() {
        let (_guard, env) = TestEnvironment::new("budget-override");
        env.set(COMPRESSION_BUDGET_ENV, "24");

        let mut lease =
            acquire_for_pid(None, nonzero(60), std::process::id()).expect("acquire lease");
        let decision = lease.decision();

        assert_eq!(decision.threads.get(), 24);
        assert_eq!(decision.budget.get(), 24);
        assert_eq!(decision.source, CompressionThreadSource::BudgetShare);
        lease.release();
    }

    #[test]
    fn invalid_numeric_overrides_warn_and_use_defaults() {
        let (_guard, env) = TestEnvironment::new("invalid");
        env.set(COMPRESSION_THREADS_ENV, "not-a-number");
        env.set(COMPRESSION_BUDGET_ENV, "0");

        let warnings = Arc::new(Mutex::new(Vec::new()));
        let warning_writer = Arc::clone(&warnings);
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_writer(move || SharedWriter(Arc::clone(&warning_writer)))
            .finish();
        let mut lease = tracing::subscriber::with_default(subscriber, || {
            acquire_for_pid(Some(nonzero(9)), nonzero(13), std::process::id())
        })
        .expect("invalid overrides must not panic");
        let decision = lease.decision();
        let warnings = String::from_utf8(warnings.lock().unwrap().clone()).unwrap();

        assert_eq!(decision.threads.get(), 13);
        assert_eq!(decision.budget.get(), 13);
        assert_eq!(decision.source, CompressionThreadSource::Default);
        assert!(warnings.contains(COMPRESSION_THREADS_ENV), "{warnings}");
        assert!(warnings.contains(COMPRESSION_BUDGET_ENV), "{warnings}");
        lease.release();
    }

    #[test]
    fn config_threads_override_is_preserved() {
        let (_guard, _env) = TestEnvironment::new("config-override");
        let mut lease = acquire_for_pid(Some(nonzero(5)), nonzero(60), std::process::id())
            .expect("acquire lease");
        assert_eq!(lease.decision().threads.get(), 5);
        assert_eq!(lease.decision().source, CompressionThreadSource::Override);
        lease.release();
    }

    #[test]
    fn lease_release_removes_own_pid_file() {
        let (_guard, env) = TestEnvironment::new("release");
        let lease_path = env.dir.join(std::process::id().to_string());
        let mut lease =
            acquire_for_pid(None, nonzero(8), std::process::id()).expect("acquire lease");

        assert!(lease_path.is_file());
        lease.release();
        assert!(!lease_path.exists());
    }

    #[test]
    fn flock_serializes_concurrent_registry_passes() {
        let (_guard, env) = TestEnvironment::new("flock-race");
        env.set(COMPRESSION_BUDGET_ENV, "16");
        let children = (0..4)
            .map(|_| ChildGuard(Command::new("sleep").arg("30").spawn().unwrap()))
            .collect::<Vec<_>>();
        let pids = children
            .iter()
            .map(|child| child.0.id())
            .collect::<Vec<_>>();
        // Model the required steady-state point after every contender has a
        // live PID lease, then race their complete refresh/count decisions.
        for &pid in &pids {
            seed_live_lease(&env.dir, pid);
        }

        let barrier = Arc::new(Barrier::new(pids.len()));
        let in_critical_section = Arc::new(AtomicUsize::new(0));
        let max_in_critical_section = Arc::new(AtomicUsize::new(0));
        let handles = pids
            .iter()
            .copied()
            .map(|pid| {
                let barrier = Arc::clone(&barrier);
                let in_critical_section = Arc::clone(&in_critical_section);
                let max_in_critical_section = Arc::clone(&max_in_critical_section);
                thread::spawn(move || {
                    barrier.wait();
                    acquire_for_pid_with_hook(None, nonzero(60), pid, || {
                        let concurrent = in_critical_section.fetch_add(1, Ordering::SeqCst) + 1;
                        max_in_critical_section.fetch_max(concurrent, Ordering::SeqCst);
                        thread::sleep(Duration::from_millis(20));
                        in_critical_section.fetch_sub(1, Ordering::SeqCst);
                    })
                    .expect("concurrent registry pass")
                })
            })
            .collect::<Vec<_>>();
        let mut leases = handles
            .into_iter()
            .map(|handle| handle.join().expect("registry thread panicked"))
            .collect::<Vec<_>>();

        assert_eq!(max_in_critical_section.load(Ordering::SeqCst), 1);
        assert!(
            leases
                .iter()
                .all(|lease| lease.decision().active_leases == 4)
        );
        let total_threads = leases
            .iter()
            .map(|lease| lease.decision().threads.get())
            .sum::<usize>();
        assert!(total_threads <= 16);
        for lease in &mut leases {
            lease.release();
        }
    }
}
