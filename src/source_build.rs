//! Build a `.whl` from a local path or git checkout via `pip wheel`.
//!
//! Used by `[retread-wheels]` entries that take `path = "..."` or
//! `git = "..."` instead of the PyPI `version + index` form. The
//! produced wheel goes through the same auto-bundle + METADATA-rewrite
//! pipeline as any PyPI-resolved wheel.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use tokio::process::Command;

/// Build a wheel from a local source tree using `uv pip wheel --no-deps`.
///
/// `--no-deps` is the critical flag: it stops the build from fetching the
/// project's transitive runtime dependencies (which for things like
/// isaaclab is GBs of torch + CUDA wheels). retread already handles the
/// dependency story via the bundle + conda emission; we only need the
/// raw wheel for METADATA inspection and packaging.
///
/// `python_version` (e.g. "3.11", "3.13") tells uv which interpreter to
/// use for the build. uv downloads python-build-standalone on demand
/// (cached under `~/.cache/uv/python/`) so retread itself doesn't need
/// to ship any python -- ANY python version the user asks for works
/// without rebuilding retread.
///
/// Returns the path to the produced `.whl` (inside `out_dir`).
pub async fn build_wheel_from_path(
    source: &Path,
    out_dir: &Path,
    python_version: &str,
) -> Result<PathBuf> {
    tokio::fs::create_dir_all(out_dir)
        .await
        .with_context(|| format!("creating wheel output dir {}", out_dir.display()))?;

    // Cache reuse: if out_dir already holds a built wheel, return it
    // instead of re-running uv (build + isolated env setup takes 30-60s
    // per package for IsaacLab-sized sources). To force a rebuild after
    // editing the source, delete the per-entry folder under
    // `<pack>/wheels/<entry_name>/`.
    if let Some(cached) = newest_wheel_in(out_dir).await? {
        tracing::info!(
            source = %source.display(),
            wheel = %cached.display(),
            "reusing cached wheel (delete the folder to force rebuild)",
        );
        return Ok(cached);
    }

    tracing::info!(
        source = %source.display(),
        python = %python_version,
        "building wheel via uv build --wheel (this can take a minute; uv downloads python if missing)",
    );
    // `uv build --wheel`: build the project at `source` into a wheel.
    // (uv doesn't expose `uv pip wheel` -- the PEP 517 build pipeline
    // lives under the top-level `uv build` command.) `--python <ver>`
    // tells uv which interpreter to use; `UV_PYTHON_DOWNLOADS=automatic`
    // enables auto-fetching of python-build-standalone binaries when
    // the requested version isn't installed locally. uv build only
    // builds the project's own wheel -- it doesn't fetch runtime deps
    // -- so no equivalent of pip's `--no-deps` is needed.
    let py_arg = format!("--python={python_version}");
    let out_arg = format!("--out-dir={}", out_dir.display());
    run_capturing_uv(&[
        "build",
        "--wheel",
        &py_arg,
        &out_arg,
        &source.display().to_string(),
    ])
    .await?;
    find_built_wheel(out_dir).await
}

/// Return the wheel in `dir` with the latest mtime, or `None` if `dir`
/// is missing or contains no .whl. Used by the cache-reuse path so
/// repeated solves don't re-run `pip wheel`.
async fn newest_wheel_in(dir: &Path) -> Result<Option<PathBuf>> {
    if !dir.exists() {
        return Ok(None);
    }
    let mut read = tokio::fs::read_dir(dir)
        .await
        .with_context(|| format!("opening wheel-cache dir {}", dir.display()))?;
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    while let Some(entry) = read
        .next_entry()
        .await
        .with_context(|| format!("reading wheel-cache dir {}", dir.display()))?
    {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !name.ends_with(".whl") {
            continue;
        }
        // Skip our own post-processed wheels so we always reuse the
        // raw pip-wheel output and re-run inject+D on it. Match on
        // SUBSTRING (not just .ends_with) so multi-suffix names like
        // `foo.injected.autodata.whl` are filtered too -- otherwise
        // the cache lookup picks the post-processed wheel as the new
        // "raw" input, the next pipeline run suffixes it AGAIN, and
        // the filename grows by ~18 chars per solve until pip wheel /
        // git clone hits ENAMETOOLONG. Burned a multi-version-bump
        // debug session on exactly this. Add every new suffix here
        // when introducing a new pipeline phase.
        const RETREAD_SUFFIXES: &[&str] = &[".injected.", ".autodata.", ".relaxed."];
        if RETREAD_SUFFIXES.iter().any(|s| name.contains(s)) {
            continue;
        }
        let mtime = entry
            .metadata()
            .await
            .with_context(|| format!("stat'ing wheel {}", path.display()))?
            .modified()
            .with_context(|| format!("reading mtime of {}", path.display()))?;
        if best.as_ref().is_none_or(|(t, _)| mtime > *t) {
            best = Some((mtime, path));
        }
    }
    Ok(best.map(|(_, p)| p))
}

/// v0.18.0+: download a PyPI sdist (`.tar.gz` / `.zip`) and run
/// `uv build --wheel` on it. Used as the BFS fallback when a dep is
/// sdist-only on PyPI (gym, classic-control, ...). Output is a normal
/// wheel that re-enters the bundle pipeline.
///
/// `out_dir` should be per-entry (e.g. `<pack>/wheels/<entry>/`) so
/// cache reuse and cleanup match what other materialize paths do.
/// Returns the path to the built wheel.
pub async fn build_wheel_from_sdist_url(
    sdist_url: &url::Url,
    out_dir: &Path,
    python_version: &str,
    expected_sha256: Option<&str>,
) -> Result<PathBuf> {
    let expected_sha256 = expected_sha256
        .map(normalize_sha256)
        .transpose()
        .context("validating expected sdist sha256")?;

    // Cache identity is supplied by the caller and must include the source
    // digest and immutable target. Validate the digest syntax before looking
    // at that cache so malformed replay provenance never reaches filesystem
    // or network work.
    if let Some(cached) = newest_wheel_in(out_dir).await? {
        tracing::info!(
            sdist = %sdist_url,
            wheel = %cached.display(),
            "reusing cached wheel from previous sdist build",
        );
        return Ok(cached);
    }

    // Pull the sdist filename out of the URL.
    let filename = sdist_url
        .path_segments()
        .and_then(|mut s| s.next_back())
        .filter(|f| !f.is_empty())
        .ok_or_else(|| anyhow!("sdist URL {sdist_url} has no filename component"))?
        .to_string();
    let sdist_path = out_dir.join(&filename);

    tracing::info!(
        url = %sdist_url,
        dst = %sdist_path.display(),
        "downloading sdist for last-resort wheel build",
    );
    let bytes = reqwest::get(sdist_url.clone())
        .await
        .with_context(|| format!("downloading sdist {sdist_url}"))?
        .error_for_status()
        .with_context(|| format!("sdist HTTP error for {sdist_url}"))?
        .bytes()
        .await
        .with_context(|| format!("reading sdist body from {sdist_url}"))?;

    // An sdist executes arbitrary build-backend code. Verify the downloaded
    // bytes before writing them into the build cache and, critically, before
    // invoking `uv build`.
    if let Some(expected) = expected_sha256.as_deref() {
        verify_sha256(&bytes, expected)
            .with_context(|| format!("sdist content verification failed for {sdist_url}"))?;
    }

    tokio::fs::create_dir_all(out_dir)
        .await
        .with_context(|| format!("creating sdist-build out dir {}", out_dir.display()))?;
    tokio::fs::write(&sdist_path, &bytes)
        .await
        .with_context(|| format!("writing sdist to {}", sdist_path.display()))?;

    tracing::info!(
        sdist = %sdist_path.display(),
        python = %python_version,
        "uv build --wheel on sdist (downloads python if needed)",
    );
    let py_arg = format!("--python={python_version}");
    let out_arg = format!("--out-dir={}", out_dir.display());
    run_capturing_uv(&[
        "build",
        "--wheel",
        &py_arg,
        &out_arg,
        &sdist_path.display().to_string(),
    ])
    .await?;
    let wheel_path = find_built_wheel(out_dir).await?;

    // DETERMINISM GUARD (Amendment 3): detect non-reproducible setuptools_scm
    // versions, mirroring the identical guard in build_wheel_from_git.
    // A wheel whose filename contains .devN, .dYYYYMMDD, or +g<sha> was built
    // without a release tag — its version/filename will DRIFT across calendar
    // days even when the sdist URL is pinned, causing lock drift on replay.
    // For static released versions (e.g. gym 0.26.2) this is a silent no-op.
    if wheel_path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(is_nondeterministic_version)
    {
        tracing::warn!(
            sdist_url = %sdist_url,
            filename = %wheel_path.display(),
            "sdist-built wheel has a non-reproducible setuptools_scm version \
             (contains .devN, .dYYYYMMDD, or +g<sha>). The wheel filename \
             will DRIFT across calendar days even when the sdist URL is \
             pinned, causing lock drift on replay. Fix: ensure the sdist's \
             build backend emits a static release version, or set \
             SETUPTOOLS_SCM_PRETEND_VERSION=<version> in the build env.",
        );
    }

    Ok(wheel_path)
}

fn normalize_sha256(value: &str) -> Result<String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("expected sdist sha256 must be exactly 64 hexadecimal characters");
    }
    Ok(value.to_ascii_lowercase())
}

fn verify_sha256(bytes: &[u8], expected: &str) -> Result<()> {
    use sha2::{Digest, Sha256};

    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != expected {
        bail!("sdist sha256 mismatch: expected {expected}, got {actual}");
    }
    Ok(())
}

/// Shared cross-pack cache directory for a built git wheel, keyed by
/// (repo url, resolved commit sha, subdirectory, python version).
///
/// Layout: `<retread cache root>/built-wheels/git/<slug>/<key12>/<raw>.whl`
/// (same slug/short-hash hierarchy rationale as [`git_checkout_root`]).
/// Every pack that pins the same (url, rev, subdir) reuses ONE build --
/// previously each pack rebuilt identical wheels into its own
/// `pypi-packs/<pack>/wheels/<entry>/` dir (isaac-pack and
/// isaac-pack-latest both building IsaacLab's 15-member group at the same
/// rev was the motivating case; ~30-60 s of `uv build` per member).
pub fn git_wheel_cache_dir(
    url: &str,
    sha: &str,
    subdirectory: &str,
    python_version: &str,
) -> PathBuf {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"retread-git-wheel-v1\n");
    hasher.update(url.as_bytes());
    hasher.update(b"\0");
    hasher.update(sha.as_bytes());
    hasher.update(b"\0");
    hasher.update(subdirectory.as_bytes());
    hasher.update(b"\0");
    hasher.update(python_version.as_bytes());
    let digest = hasher.finalize();
    let key12: String = digest.iter().take(6).map(|b| format!("{b:02x}")).collect();
    let mut slug = git_slug(url);
    slug.truncate(24);
    crate::courier::retread_cache_root()
        .join("built-wheels")
        .join("git")
        .join(slug)
        .join(key12)
}

/// Look up the shared git-wheel cache; on a hit, materialize the raw wheel
/// into `out_dir` (hardlink, copy on EXDEV) so the downstream
/// inject/autodata/relax pipeline finds it exactly where a fresh build
/// would have put it. Returns the out_dir wheel path, or `None` on miss.
async fn git_wheel_cache_lookup(cache_wheel_dir: &Path, out_dir: &Path) -> Result<Option<PathBuf>> {
    let Some(cached) = newest_wheel_in(cache_wheel_dir).await? else {
        return Ok(None);
    };
    tokio::fs::create_dir_all(out_dir)
        .await
        .with_context(|| format!("creating wheel output dir {}", out_dir.display()))?;
    let dst = out_dir.join(cached.file_name().expect("wheel path has a filename"));
    if !dst.exists() {
        crate::wheel::hardlink_or_copy_async(&cached, &dst)
            .await
            .with_context(|| {
                format!(
                    "materializing shared-cache git wheel {} -> {}",
                    cached.display(),
                    dst.display()
                )
            })?;
    }
    tracing::info!(
        cache = %cached.display(),
        wheel = %dst.display(),
        "reusing shared-cache git wheel (cross-pack; delete the cache dir to force rebuild)",
    );
    Ok(Some(dst))
}

/// Best-effort population of the shared git-wheel cache after a successful
/// build. Failure only warns: the pack-local out_dir copy is authoritative,
/// the shared cache is purely a cross-pack build-speed optimization.
async fn git_wheel_cache_store(wheel: &Path, cache_wheel_dir: &Path) {
    let store = async {
        tokio::fs::create_dir_all(cache_wheel_dir).await?;
        let filename = wheel
            .file_name()
            .ok_or_else(|| anyhow!("wheel path has no filename: {}", wheel.display()))?;
        let dst = cache_wheel_dir.join(filename);
        if dst.exists() {
            return Ok::<_, anyhow::Error>(());
        }
        let tmp = cache_wheel_dir.join(format!(
            "{}.{}.tmp",
            filename.to_string_lossy(),
            std::process::id()
        ));
        crate::wheel::hardlink_or_copy_async(wheel, &tmp).await?;
        if let Err(e) = tokio::fs::rename(&tmp, &dst).await {
            let _ = tokio::fs::remove_file(&tmp).await;
            if !dst.exists() {
                return Err(e.into());
            }
        }
        Ok(())
    };
    if let Err(e) = store.await {
        tracing::warn!(
            wheel = %wheel.display(),
            cache = %cache_wheel_dir.display(),
            error = %format!("{e:#}"),
            "could not populate shared git-wheel cache (non-fatal)",
        );
    }
}

/// Compute a git wheel entry's source directory without accessing it.
///
/// This is path derivation only; callers that read the returned tree must hold
/// the checkout lease obtained by this module's build/checkout boundary.
pub fn git_source_root(url: &str, rev: &str, subdirectory: &str, cache_dir: &Path) -> PathBuf {
    git_checkout_root(url, rev, cache_dir).join(subdirectory)
}

/// Compute the on-disk *checkout* directory for a (url, rev) pair --
/// the parent of each wheel entry's source subdirectory. Used as a pure
/// identity/grouping key by auto-data planning so the WHOLE upstream repo (minus
/// `.gitignore`'d paths and minus subdirectories already shipped as
/// wheels by sibling entries in the same bundle) can ride along into
/// the conda env at `$PREFIX/lib/<rel>`.
///
/// This function performs no synchronization and confers no permission to read
/// the returned path. Checkout consumers must retain the [`GitCheckout`] lease
/// returned by the internal checkout/build boundary.
///
/// Layout (v0.13.3+): cache_dir / retread-git-clones / <slug> /
/// <sha12> / ... -- a HIERARCHY rather than a single flat dirname.
/// This is what pip/uv do (the wheel itself stays a normal PEP 427
/// filename; disambiguation rides in parent directories). Each path
/// component is independently bounded:
///   - `<slug>`: repo-name slug, truncated to 24 chars
///   - `<sha12>`: 12 hex chars of sha256(url + "\0" + rev)
///
/// Previously the (slug + raw 40-char git SHA) was flattened into one
/// 60+ char component; combined with the rattler cache prefix and
/// deep upstream repo internals (IsaacLab's nested test/snapshot
/// trees), full pathnames pushed against PATH_MAX and triggered
/// ENAMETOOLONG on git checkout. Splitting into a hierarchy also lets
/// multiple revs of the same repo share a parent dir, which is nicer
/// for inspection. Hashing the rev also kills any chance of `/`,
/// `@`, or `:` from a branch-name rev leaking into the on-disk path.
pub fn git_checkout_root(url: &str, rev: &str, cache_dir: &Path) -> PathBuf {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    hasher.update(b"\0");
    hasher.update(rev.as_bytes());
    let digest = hasher.finalize();
    let sha12: String = digest.iter().take(6).map(|b| format!("{b:02x}")).collect();
    let mut slug = git_slug(url);
    // The slug strips `https___github.com_`; cap whatever's left so
    // big-org/long-name repos don't blow the slug component.
    slug.truncate(24);
    cache_dir.join("retread-git-clones").join(slug).join(sha12)
}

const CHECKOUT_READY_MARKER: &str = ".retread-checkout-ready-v1";
const CHECKOUT_LOCK_POLL: Duration = Duration::from_millis(10);

#[cfg(test)]
static CHECKOUT_REPAIRS: OnceLock<Mutex<HashMap<PathBuf, usize>>> = OnceLock::new();

#[cfg(test)]
fn record_checkout_repair(clone_dir: &Path) {
    let mut repairs = CHECKOUT_REPAIRS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("checkout-repair test counter poisoned");
    *repairs.entry(clone_dir.to_path_buf()).or_default() += 1;
}

#[cfg(test)]
fn checkout_repair_count(clone_dir: &Path) -> usize {
    CHECKOUT_REPAIRS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("checkout-repair test counter poisoned")
        .get(clone_dir)
        .copied()
        .unwrap_or_default()
}

#[cfg(test)]
fn signal_checkout_test_path(variable: &str) -> Result<()> {
    let Some(path) = std::env::var_os(variable) else {
        return Ok(());
    };
    std::fs::write(PathBuf::from(&path), b"ready\n").with_context(|| {
        format!(
            "writing checkout test signal {} from {variable}",
            PathBuf::from(path).display()
        )
    })
}

#[cfg(test)]
async fn pause_initializer_after_exclusive_for_test() -> Result<()> {
    let Some(release) = std::env::var_os("RETREAD_TEST_EXCLUSIVE_RELEASE") else {
        return Ok(());
    };
    signal_checkout_test_path("RETREAD_TEST_EXCLUSIVE_HELD")?;
    let release = PathBuf::from(release);
    loop {
        if release.try_exists().with_context(|| {
            format!(
                "checking checkout test release signal {}",
                release.display()
            )
        })? {
            return Ok(());
        }
        tokio::time::sleep(CHECKOUT_LOCK_POLL).await;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CloneIdentity {
    digest: [u8; 32],
    marker_contents: Vec<u8>,
}

impl CloneIdentity {
    fn new(url: &str, rev: &str) -> Self {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(url.as_bytes());
        hasher.update(b"\0");
        hasher.update(rev.as_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        let digest_hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        let marker_contents =
            format!("pixi-build-retread-checkout-v1\n{digest_hex}\n").into_bytes();
        Self {
            digest,
            marker_contents,
        }
    }
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct ProcessLockKey {
    device: u64,
    inode: u64,
}

#[cfg(not(unix))]
type ProcessLockKey = PathBuf;

struct ProcessCloneLock {
    file: File,
    lock_path: PathBuf,
    identity: CloneIdentity,
    local: Arc<tokio::sync::RwLock<()>>,
    shared: AtomicBool,
    poisoned: AtomicBool,
    #[cfg(test)]
    os_acquisitions: std::sync::atomic::AtomicUsize,
}

fn process_clone_locks() -> &'static Mutex<HashMap<ProcessLockKey, Arc<ProcessCloneLock>>> {
    // Strong entries intentionally retain one fd per checkout touched by this
    // backend process. That bounded process-lifetime cost is what guarantees a
    // later worker can never reopen and flock the same inode a second time.
    static LOCKS: OnceLock<Mutex<HashMap<ProcessLockKey, Arc<ProcessCloneLock>>>> = OnceLock::new();
    LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(unix)]
fn process_lock_key(file: &File, _lock_path: &Path) -> std::io::Result<ProcessLockKey> {
    use std::os::unix::fs::MetadataExt;

    let metadata = file.metadata()?;
    Ok(ProcessLockKey {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

#[cfg(not(unix))]
fn process_lock_key(_file: &File, lock_path: &Path) -> std::io::Result<ProcessLockKey> {
    std::fs::canonicalize(lock_path)
}

/// Open the clone lock and merge it into the process registry before any
/// `flock` call. On Unix the key is the opened file's `(st_dev, st_ino)`, so
/// symlink/path aliases cannot make this process lock the same inode twice.
fn registered_clone_lock(
    lock_path: &Path,
    identity: &CloneIdentity,
) -> Result<Arc<ProcessCloneLock>> {
    let candidate = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .with_context(|| format!("opening git-clone lock file {}", lock_path.display()))?;
    let key = process_lock_key(&candidate, lock_path)
        .with_context(|| format!("identifying git-clone lock file {}", lock_path.display()))?;
    let mut locks = process_clone_locks()
        .lock()
        .map_err(|_| anyhow!("git-clone process lock registry is poisoned"))?;
    if let Some(existing) = locks.get(&key) {
        if existing.identity.digest != identity.digest {
            bail!(
                "git-clone cache-key collision: lock inode {} is already bound to a different full (url, rev) digest",
                existing.lock_path.display(),
            );
        }
        #[cfg(test)]
        signal_checkout_test_path("RETREAD_TEST_SECOND_REGISTRATION")?;
        // `candidate` is dropped here without ever being flocked. Every worker
        // in this process therefore flocks only the registry's one open
        // description, even though identifying an alias requires opening it.
        return Ok(Arc::clone(existing));
    }

    let lock = Arc::new(ProcessCloneLock {
        file: candidate,
        lock_path: lock_path.to_path_buf(),
        identity: identity.clone(),
        local: Arc::new(tokio::sync::RwLock::new(())),
        shared: AtomicBool::new(false),
        poisoned: AtomicBool::new(false),
        #[cfg(test)]
        os_acquisitions: std::sync::atomic::AtomicUsize::new(0),
    });
    locks.insert(key, Arc::clone(&lock));
    Ok(lock)
}

struct CloneReadLease {
    _local: tokio::sync::OwnedRwLockReadGuard<()>,
    lock: Arc<ProcessCloneLock>,
}

/// A checked-out source tree plus its process-local reader lease. The process
/// registry retains one cross-process SH flock for the life of the process;
/// this lease multiplexes that one OS lock across worker tasks without any
/// worker reopening or re-flocking the lock inode.
#[derive(Clone)]
pub(crate) struct GitCheckout {
    root: PathBuf,
    lease: Arc<CloneReadLease>,
}

impl GitCheckout {
    pub(crate) fn root(&self) -> &Path {
        &self.root
    }
}

impl std::fmt::Debug for GitCheckout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GitCheckout")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

impl PartialEq for GitCheckout {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root && self.lease.lock.identity == other.lease.lock.identity
    }
}

impl Eq for GitCheckout {}

#[derive(Debug)]
enum CheckoutMarkerState {
    Missing,
    Matching,
    Invalid,
}

fn checkout_ready_marker(clone_dir: &Path) -> PathBuf {
    clone_dir.join(".git").join(CHECKOUT_READY_MARKER)
}

fn checkout_marker_state(
    clone_dir: &Path,
    identity: &CloneIdentity,
) -> Result<CheckoutMarkerState> {
    let marker = checkout_ready_marker(clone_dir);
    match std::fs::read(&marker) {
        Ok(contents) if contents == identity.marker_contents => Ok(CheckoutMarkerState::Matching),
        Ok(_) => Ok(CheckoutMarkerState::Invalid),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(CheckoutMarkerState::Missing)
        }
        Err(error) => {
            Err(error).with_context(|| format!("reading git-checkout marker {}", marker.display()))
        }
    }
}

fn invalid_marker_error(clone_dir: &Path) -> anyhow::Error {
    anyhow!(
        "git-checkout marker {} is malformed or belongs to a different full (url, rev) identity; refusing to mutate a published checkout",
        checkout_ready_marker(clone_dir).display(),
    )
}

struct PendingOsLock {
    lock: Arc<ProcessCloneLock>,
    armed: bool,
}

impl PendingOsLock {
    fn acquired(lock: Arc<ProcessCloneLock>) -> Self {
        #[cfg(test)]
        lock.os_acquisitions.fetch_add(1, Ordering::Relaxed);
        Self { lock, armed: true }
    }

    fn commit_shared(mut self) {
        self.lock.shared.store(true, Ordering::Release);
        self.armed = false;
    }

    fn downgrade_and_commit(mut self) -> Result<()> {
        #[cfg(unix)]
        fs4::fs_std::FileExt::lock_shared(&self.lock.file).with_context(|| {
            format!(
                "downgrading git-clone lock to shared {}",
                self.lock.lock_path.display()
            )
        })?;

        // Windows LockFileEx layers a SH lock over an EX lock rather than
        // replacing it. Keep the same File/open description, but transition
        // through an unlock after the marker is published. Any interposing EX
        // owner rechecks that marker and therefore performs no mutation.
        #[cfg(not(unix))]
        {
            fs4::fs_std::FileExt::unlock(&self.lock.file).with_context(|| {
                format!(
                    "unlocking exclusive git-clone lock before shared transition {}",
                    self.lock.lock_path.display()
                )
            })?;
            self.armed = false;
            fs4::fs_std::FileExt::lock_shared(&self.lock.file).with_context(|| {
                format!(
                    "shared-locking git-clone lock after transition {}",
                    self.lock.lock_path.display()
                )
            })?;
            self.armed = true;
        }

        self.lock.shared.store(true, Ordering::Release);
        self.armed = false;
        Ok(())
    }
}

impl Drop for PendingOsLock {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let Err(error) = fs4::fs_std::FileExt::unlock(&self.lock.file) {
            self.lock.poisoned.store(true, Ordering::Release);
            tracing::error!(
                lock = %self.lock.lock_path.display(),
                error = %error,
                "failed to release uncommitted git-clone lock; process entry poisoned",
            );
        }
    }
}

enum InitialOsAccess {
    Ready,
    Initialize(PendingOsLock),
}

/// Establish this process's one OS lock for a clone. All probes are
/// nonblocking: cold losers never queue an EX request that the winner's
/// process-lifetime SH lock could strand forever.
fn acquire_initial_os_access(
    lock: Arc<ProcessCloneLock>,
    clone_dir: &Path,
) -> Result<InitialOsAccess> {
    loop {
        if lock.poisoned.load(Ordering::Acquire) {
            bail!(
                "git-clone lock {} is poisoned after an unlock failure",
                lock.lock_path.display()
            );
        }

        match checkout_marker_state(clone_dir, &lock.identity)? {
            CheckoutMarkerState::Invalid => return Err(invalid_marker_error(clone_dir)),
            CheckoutMarkerState::Matching => {
                let acquired =
                    fs4::fs_std::FileExt::try_lock_shared(&lock.file).with_context(|| {
                        format!(
                            "try-shared-locking git-clone lock file {}",
                            lock.lock_path.display()
                        )
                    })?;
                if acquired {
                    let pending = PendingOsLock::acquired(Arc::clone(&lock));
                    match checkout_marker_state(clone_dir, &lock.identity)? {
                        CheckoutMarkerState::Matching => {
                            pending.commit_shared();
                            return Ok(InitialOsAccess::Ready);
                        }
                        CheckoutMarkerState::Missing => {
                            drop(pending);
                        }
                        CheckoutMarkerState::Invalid => {
                            drop(pending);
                            return Err(invalid_marker_error(clone_dir));
                        }
                    }
                } else {
                    #[cfg(test)]
                    signal_checkout_test_path("RETREAD_TEST_SHARED_BLOCKED")?;
                }
            }
            CheckoutMarkerState::Missing => {
                if fs4::fs_std::FileExt::try_lock_exclusive(&lock.file).with_context(|| {
                    format!(
                        "try-exclusive-locking git-clone lock file {}",
                        lock.lock_path.display()
                    )
                })? {
                    let pending = PendingOsLock::acquired(Arc::clone(&lock));
                    // EX->SH replacement is not atomic on every flock
                    // implementation. An interposing process may have published
                    // readiness since our unlocked observation, so recheck under EX.
                    match checkout_marker_state(clone_dir, &lock.identity)? {
                        CheckoutMarkerState::Missing => {
                            return Ok(InitialOsAccess::Initialize(pending));
                        }
                        CheckoutMarkerState::Matching => {
                            pending.downgrade_and_commit()?;
                            return Ok(InitialOsAccess::Ready);
                        }
                        CheckoutMarkerState::Invalid => {
                            drop(pending);
                            return Err(invalid_marker_error(clone_dir));
                        }
                    }
                }
            }
        }
        std::thread::sleep(CHECKOUT_LOCK_POLL);
    }
}

fn publish_checkout_ready(clone_dir: &Path, identity: &CloneIdentity) -> Result<()> {
    let marker = checkout_ready_marker(clone_dir);
    let marker_parent = marker
        .parent()
        .ok_or_else(|| anyhow!("git-checkout marker has no parent: {}", marker.display()))?;
    let temporary = marker_parent.join(format!("{CHECKOUT_READY_MARKER}.tmp"));
    let publish = || -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&temporary)
            .with_context(|| format!("creating git-checkout marker {}", temporary.display()))?;
        file.write_all(&identity.marker_contents)
            .with_context(|| format!("writing git-checkout marker {}", temporary.display()))?;
        file.sync_all()
            .with_context(|| format!("syncing git-checkout marker {}", temporary.display()))?;
        std::fs::rename(&temporary, &marker).with_context(|| {
            format!(
                "publishing git-checkout marker {} -> {}",
                temporary.display(),
                marker.display()
            )
        })?;
        #[cfg(unix)]
        File::open(marker_parent)
            .and_then(|directory| directory.sync_all())
            .with_context(|| {
                format!(
                    "syncing git-checkout marker dir {}",
                    marker_parent.display()
                )
            })?;
        Ok(())
    };
    if let Err(error) = publish() {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(())
}

/// Populate or repair an UNPUBLISHED checkout while its one-time initializer
/// holds the clone's EX lock. This is never called after the ready marker has
/// been published; published warm checkouts are read-only cache entries.
async fn clone_and_checkout(clone_dir: &Path, url: &str, rev: &str) -> Result<()> {
    if !clone_dir.exists() {
        tracing::info!(url = %url, rev = %rev, "cloning git source");
        // Clone shallow without checkout. Use a two-step fetch so we can
        // target arbitrary commits (not just branch/tag tips).
        run_silent(
            Command::new("git")
                .arg("clone")
                .arg("--filter=blob:none")
                .arg("--no-checkout")
                .arg(url)
                .arg(clone_dir),
            "git clone",
        )
        .await?;
    } else {
        tracing::debug!(path = %clone_dir.display(), "git source already cached");
    }

    if checkout_rev_robust(clone_dir, rev).await? {
        return Ok(());
    }
    // Fetch the specific rev AND its reachable tags so that
    // setuptools_scm can find a release tag and emit a static version
    // (e.g. "1.1.1") rather than a drifting dev/date suffix (e.g.
    // "1.1.1.dev4+g1234567.d20250101").
    run_silent(
        Command::new("git")
            .args(["fetch", "--tags", "origin", rev])
            .current_dir(clone_dir),
        "git fetch --tags",
    )
    .await?;
    if !checkout_rev_robust(clone_dir, "FETCH_HEAD").await? {
        bail!(
            "git checkout FETCH_HEAD failed even after cleaning the working \
             tree, in clone at {}",
            clone_dir.display()
        );
    }
    Ok(())
}

/// Perform the legacy working-tree repair and full-reclone fallback exactly
/// once, before publishing this `(url, rev)` checkout. Existing v4.8.1 cache
/// directories have no marker, so their first access under the new protocol
/// receives this one migration repair. `pending` is threaded through every
/// unabortable filesystem job so none can continue after EX is released.
async fn initialize_checkout_once(
    clone_dir: &Path,
    url: &str,
    rev: &str,
    pending: PendingOsLock,
) -> Result<PendingOsLock> {
    if let Err(error) = clone_and_checkout(clone_dir, url, rev).await {
        tracing::warn!(
            url = %url, rev = %rev, error = %format!("{error:#}"),
            path = %clone_dir.display(),
            "unpublished git clone/checkout failed after working-tree repair; \
             wiping the clone dir and re-cloning once before publication",
        );
        let remove_dir = clone_dir.to_path_buf();
        let pending = tokio::task::spawn_blocking(move || -> Result<PendingOsLock> {
            std::fs::remove_dir_all(&remove_dir)
                .with_context(|| format!("wiping corrupted clone dir {}", remove_dir.display()))?;
            Ok(pending)
        })
        .await
        .context("git-checkout wipe task panicked")??;
        clone_and_checkout(clone_dir, url, rev)
            .await
            .with_context(|| {
                format!("re-clone after wiping corrupted dir still failed for {url}@{rev}")
            })?;
        return Ok(pending);
    }
    Ok(pending)
}

async fn initialize_process_clone_lock(
    lock: Arc<ProcessCloneLock>,
    clone_dir: PathBuf,
    url: String,
    rev: String,
    _writer: tokio::sync::OwnedRwLockWriteGuard<()>,
) -> Result<()> {
    if lock.shared.load(Ordering::Acquire) {
        return Ok(());
    }
    if lock.poisoned.load(Ordering::Acquire) {
        bail!(
            "git-clone lock {} is poisoned after an unlock failure",
            lock.lock_path.display()
        );
    }

    let access = {
        let lock = Arc::clone(&lock);
        let clone_dir = clone_dir.clone();
        tokio::task::spawn_blocking(move || acquire_initial_os_access(lock, &clone_dir))
            .await
            .context("git-clone OS-lock task panicked")??
    };
    let InitialOsAccess::Initialize(pending) = access else {
        return Ok(());
    };

    #[cfg(test)]
    pause_initializer_after_exclusive_for_test().await?;

    // `pending` is cancellation-safe and the outer caller awaits this work via
    // a detached Tokio task. The EX lock therefore spans clone/repair, marker
    // publication, and the same-fd EX->SH transition as one transaction.
    let pending = initialize_checkout_once(&clone_dir, &url, &rev, pending).await?;

    let identity = lock.identity.clone();
    let marker_clone_dir = clone_dir.clone();
    let pending = tokio::task::spawn_blocking(move || -> Result<PendingOsLock> {
        publish_checkout_ready(&marker_clone_dir, &identity)?;
        Ok(pending)
    })
    .await
    .context("git-checkout marker task panicked")??;

    // Downgrade and publish the in-process Shared state inside one blocking
    // closure: no async cancellation point may split those two state changes.
    tokio::task::spawn_blocking(move || pending.downgrade_and_commit())
        .await
        .context("git-clone downgrade task panicked")??;
    Ok(())
}

/// Return a reader lease for an immutable, published `(url, rev)` checkout.
///
/// The first process to observe a missing ready marker takes EX, performs the
/// one-time clone/repair, publishes the marker, and downgrades its ONE registry
/// fd to SH. Other processes poll nonblocking until they can take SH. Within a
/// process, all tasks share that one fd through an async RwLock; no task ever
/// flocks the inode a second time.
pub(crate) async fn ensure_git_checkout(
    url: &str,
    rev: &str,
    cache_dir: &Path,
) -> Result<GitCheckout> {
    // Delegate to git_checkout_root so the layout stays in sync. (Was
    // duplicated here before v0.13.3 -- update both or the resolver
    // half stops finding the cached clone the cloner half just made.)
    let clone_dir = git_checkout_root(url, rev, cache_dir);
    let parent = clone_dir.parent().unwrap();
    tokio::fs::create_dir_all(parent).await.with_context(|| {
        format!(
            "creating git-clone parent dir {} (for url={url}, rev={rev}, target={})",
            parent.display(),
            clone_dir.display(),
        )
    })?;

    let lock_path = clone_dir.with_extension("lock");
    let identity = CloneIdentity::new(url, rev);
    let lock = {
        let lock_path = lock_path.clone();
        let identity = identity.clone();
        tokio::task::spawn_blocking(move || registered_clone_lock(&lock_path, &identity))
            .await
            .context("git-clone registry task panicked")??
    };

    if !lock.shared.load(Ordering::Acquire) {
        let writer = Arc::clone(&lock.local).write_owned().await;
        if !lock.shared.load(Ordering::Acquire) {
            // The spawned task owns `writer`. Dropping this caller's JoinHandle
            // detaches rather than cancels an in-flight mutating transaction.
            let initializer = tokio::spawn(initialize_process_clone_lock(
                Arc::clone(&lock),
                clone_dir.clone(),
                url.to_string(),
                rev.to_string(),
                writer,
            ));
            initializer
                .await
                .context("git-clone initializer task panicked")??;
        }
    }

    if lock.poisoned.load(Ordering::Acquire) {
        bail!(
            "git-clone lock {} is poisoned after an unlock failure",
            lock.lock_path.display()
        );
    }
    let local = Arc::clone(&lock.local).read_owned().await;
    if !lock.shared.load(Ordering::Acquire) {
        bail!(
            "git-clone lock {} reached reader path before SH initialization",
            lock.lock_path.display()
        );
    }

    // Validate the full identity even on the process-local Shared fast path.
    // This catches a truncated checkout-path hash collision or marker
    // deletion/corruption without ever re-enabling a destructive writer.
    let marker = checkout_ready_marker(&clone_dir);
    let marker_contents = tokio::fs::read(&marker)
        .await
        .with_context(|| format!("reading published git-checkout marker {}", marker.display()))?;
    if marker_contents != identity.marker_contents {
        return Err(invalid_marker_error(&clone_dir));
    }

    Ok(GitCheckout {
        root: clone_dir,
        lease: Arc::new(CloneReadLease {
            _local: local,
            lock,
        }),
    })
}

/// Result of a git wheel build. Keeping this value alive keeps the checkout's
/// process-local reader lease alive; internal callers pass that same lease
/// through all subsequent source-tree injection reads.
#[derive(Debug)]
pub(crate) struct GitWheelBuild {
    wheel_path: PathBuf,
    resolved_sha: String,
    checkout: GitCheckout,
}

impl GitWheelBuild {
    pub(crate) fn into_parts(self) -> (PathBuf, String, GitCheckout) {
        (self.wheel_path, self.resolved_sha, self.checkout)
    }
}

/// Internal leased build boundary. The handler retains this result across its
/// subsequent source-tree injection phases.
pub(crate) async fn build_wheel_from_git_leased(
    url: &str,
    rev: &str,
    subdirectory: &str,
    cache_dir: &Path,
    out_dir: &Path,
    python_version: &str,
) -> Result<GitWheelBuild> {
    build_wheel_from_git_inner(url, rev, subdirectory, cache_dir, out_dir, python_version).await
}

/// Build a wheel from a clone-once git checkout and return its path plus the
/// resolved commit SHA. The checkout stays under the process's retained SH
/// lock for this complete operation. `rev` may be a commit, tag, or branch.
///
/// The emitted wheel filename is also checked for non-reproducible
/// `setuptools_scm` markers (`.devN`, `.dYYYYMMDD`, and `+g<sha>`).
pub async fn build_wheel_from_git(
    url: &str,
    rev: &str,
    subdirectory: &str,
    cache_dir: &Path,
    out_dir: &Path,
    python_version: &str,
) -> Result<(PathBuf, String)> {
    let build =
        build_wheel_from_git_inner(url, rev, subdirectory, cache_dir, out_dir, python_version)
            .await?;
    let (wheel_path, resolved_sha, _checkout) = build.into_parts();
    Ok((wheel_path, resolved_sha))
}

async fn build_wheel_from_git_inner(
    url: &str,
    rev: &str,
    subdirectory: &str,
    cache_dir: &Path,
    out_dir: &Path,
    python_version: &str,
) -> Result<GitWheelBuild> {
    // NOTE on the shared cross-pack wheel cache: the lookup deliberately
    // happens AFTER clone+checkout (below), not here. Callers derive
    // `source_root` from the checkout for the auto-data inject phase, so the
    // clone must exist even on a cache hit. The clone is machine-shared per
    // (url, rev) and a no-op when warm; the cache only needs to skip the
    // expensive per-pack `uv build`.
    let checkout = ensure_git_checkout(url, rev, cache_dir).await?;
    let clone_dir = checkout.root();

    let source_dir = clone_dir.join(subdirectory);
    if !source_dir.exists() {
        bail!(
            "subdirectory `{subdirectory}` not found in clone at {}",
            clone_dir.display()
        );
    }

    // Resolve the ACTUAL commit SHA after checkout. This converts branch
    // names, tags, and "HEAD" to a stable 40-char SHA that the lock can
    // store. Keying on the resolved SHA (rather than the original `rev`
    // string) ensures a lukewarm replay clones the exact same commit even
    // when the original rev was a moving ref like a branch name.
    let resolved_sha = run_output(
        Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&clone_dir),
        "git rev-parse HEAD",
    )
    .await?;
    let resolved_sha = resolved_sha.trim().to_string();

    // Cross-pack shared-cache lookup, now that a moving rev (branch/tag)
    // has been resolved to an exact sha. A hit skips the `uv build` (the
    // expensive part; the clone above was needed anyway to resolve the sha).
    let shared = git_wheel_cache_dir(url, &resolved_sha, subdirectory, python_version);
    if let Some(wheel) = git_wheel_cache_lookup(&shared, out_dir).await? {
        return Ok(GitWheelBuild {
            wheel_path: wheel,
            resolved_sha,
            checkout,
        });
    }

    // DETERMINISM GUARD: detect non-reproducible setuptools_scm versions.
    // A wheel whose version contains .devN, .dYYYYMMDD, or +g<sha> segments
    // was built without a reachable tag at the pinned SHA. Its filename (and
    // therefore the lock entry's `version` + `filename` fields) will DRIFT
    // across calendar days even when the commit SHA is unchanged, producing
    // a lock that is not byte-identical on replay. The `git fetch --tags`
    // above is cheap insurance; this warn fires when it was not enough.
    let wheel_path = build_wheel_from_path(&source_dir, out_dir, python_version).await?;
    if wheel_path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(is_nondeterministic_version)
    {
        tracing::warn!(
            url = %url,
            rev = %rev,
            resolved_sha = %resolved_sha,
            filename = %wheel_path.display(),
            "git-source wheel has a non-reproducible setuptools_scm version \
             (contains .devN, .dYYYYMMDD, or +g<sha>). The wheel filename \
             will DRIFT across calendar days even when the commit SHA is \
             pinned, causing lock drift on replay. Fix: ensure the upstream \
             repo has a reachable tag at the pinned commit, or set \
             SETUPTOOLS_SCM_PRETEND_VERSION=<version> in the build env.",
        );
    }

    // Populate the shared cross-pack cache (best-effort).
    git_wheel_cache_store(&wheel_path, &shared).await;

    Ok(GitWheelBuild {
        wheel_path,
        resolved_sha,
        checkout,
    })
}

/// Returns `true` when a wheel filename contains markers of a
/// non-reproducible `setuptools_scm`-style version:
/// - `.devN` — development distance (e.g. `1.1.1.dev4`)
/// - `.dYYYYMMDD` — local date segment (e.g. `+g1234.d20250101`)
/// - `+g<hex>` — local git-hash segment
///
/// These cause the version/filename to change daily even for a pinned
/// commit SHA, breaking byte-identical lock replay.
pub fn is_nondeterministic_version(filename: &str) -> bool {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        // Matches any of:
        //   .devN    (development distance)
        //   .dYYYYMMDD  (local date in setuptools_scm local segment)
        //   +g<hexchars>  (local git-hash segment)
        regex::Regex::new(r"(?:\.dev\d+|\.d\d{8}|\+g[0-9a-f]+)").unwrap()
    });
    re.is_match(filename)
}

/// Run a command silently and return its trimmed stdout as a `String`.
/// Fails if the command exits non-zero.
async fn run_output(cmd: &mut Command, label: &str) -> Result<String> {
    let output = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .with_context(|| format!("spawning {label}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "{label} failed (status {}): {}",
            output.status,
            stderr.trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(unix)]
struct UvProcessGroupGuard {
    pgid: nix::unistd::Pid,
    armed: bool,
}

#[cfg(unix)]
impl UvProcessGroupGuard {
    fn new(pgid: u32) -> Result<Self> {
        let pgid = i32::try_from(pgid).context("uv process id exceeds Unix pid_t range")?;
        Ok(Self {
            pgid: nix::unistd::Pid::from_raw(pgid),
            armed: true,
        })
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

#[cfg(unix)]
impl Drop for UvProcessGroupGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        match nix::sys::signal::killpg(self.pgid, nix::sys::signal::Signal::SIGKILL) {
            Ok(()) | Err(nix::errno::Errno::ESRCH) => {}
            Err(error) => tracing::warn!(
                pgid = self.pgid.as_raw(),
                error = %error,
                "failed to kill cancelled uv build process group",
            ),
        }
    }
}

/// Invoke `uv` with the given args, capturing stdout + stderr so neither
/// leaks to retread's stdout (which is the JSON-RPC channel to pixi).
/// Sets `UV_PYTHON_DOWNLOADS=automatic` so missing pythons are fetched
/// on demand without user intervention.
async fn run_capturing_uv(args: &[&str]) -> Result<()> {
    // The callers above have already exhausted their wheel-cache paths. Hold
    // the process-wide permit only for the real build subprocess so nested
    // handler concurrency cannot multiply expensive `uv build` work.
    let _build_permit = crate::concurrency::acquire_build_permit().await;
    let mut cmd = Command::new("uv");
    for arg in args {
        cmd.arg(arg);
    }
    crate::fasttmp::apply_backend_env(&mut cmd);
    #[cfg(unix)]
    cmd.process_group(0);
    let child = cmd
        .env("UV_PYTHON_DOWNLOADS", "automatic")
        // If pixi cancels a backend request, dropping this future must also
        // terminate the direct uv child. The Unix process-group guard below
        // additionally terminates Python/compiler descendants.
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawning uv (is it on PATH? expected via retread's runtime dep)")?;
    // Keep this guard declared after `_build_permit`: Rust drops locals in
    // reverse declaration order, so cancellation kills the complete process
    // group before releasing capacity to a replacement build.
    #[cfg(unix)]
    let mut process_group = UvProcessGroupGuard::new(
        child
            .id()
            .context("spawned uv process has no operating-system pid")?,
    )?;
    let output = child
        .wait_with_output()
        .await
        .context("waiting for uv build subprocess")?;
    #[cfg(unix)]
    process_group.disarm();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        tracing::error!(stdout = %stdout, stderr = %stderr, args = ?args, "uv failed");
        // Include stderr in the bail message so pixi surfaces the
        // actual uv error (usage / network / build failure) instead
        // of a bare "status 2".
        let snippet = stderr.trim();
        let snippet = if snippet.len() > 2000 {
            format!("{}...(truncated)", &snippet[..2000])
        } else {
            snippet.to_string()
        };
        bail!("uv {:?} failed (status {}): {snippet}", args, output.status,);
    }
    if !stdout.trim().is_empty() {
        tracing::debug!(stdout = %stdout, "uv output");
    }
    Ok(())
}

/// Run a child process, capturing stdout + stderr so neither leaks to
/// retread's stdout (which is the JSON-RPC channel). Fail with the
/// captured output attached if the child exits non-zero. v0.13.4+:
/// stderr is included in the bail message so the underlying tool's
/// real error (e.g. git's "Cannot create file '<some-200-char-name>':
/// File name too long") surfaces in the pixi JSON-RPC error instead
/// of getting buried in trace logs that nobody reads. The single
/// "status N" we used to emit was useless for diagnosing upstream
/// issues like ENAMETOOLONG on git checkout.
async fn run_silent(cmd: &mut Command, label: &str) -> Result<()> {
    let output = cmd
        // Clone/checkout/clean/fetch run while the one-time EX transaction is
        // armed. If that task is aborted during runtime shutdown, do not let a
        // detached git child continue mutating after the RAII guard unlocks.
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .with_context(|| format!("spawning {label} (is the tool on PATH?)"))?;
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::error!(label = %label, stdout = %stdout, stderr = %stderr, "{label} failed");
        // Inline the stderr snippet (and stdout if non-empty -- some
        // tools shovel errors to stdout). Cap at 4KB so a runaway
        // child doesn't drown the JSON-RPC error.
        let snippet_for = |s: &str| -> String {
            let s = s.trim();
            if s.len() > 4096 {
                format!("{}...(truncated)", &s[..4096])
            } else {
                s.to_string()
            }
        };
        let stderr_snip = snippet_for(&stderr);
        let stdout_snip = snippet_for(&stdout);
        let detail = match (stderr_snip.is_empty(), stdout_snip.is_empty()) {
            (true, true) => String::new(),
            (false, true) => format!(": {stderr_snip}"),
            (true, false) => format!(": (stdout) {stdout_snip}"),
            (false, false) => format!(": {stderr_snip} | (stdout) {stdout_snip}"),
        };
        bail!("{label} failed (status {}){detail}", output.status);
    }
    Ok(())
}

/// Like [`run_silent`] but returns `Ok(false)` instead of failing when
/// the child exits non-zero. Used by paths that have a fallback (e.g.,
/// `git checkout` -> `git fetch` -> `git checkout`).
async fn try_run_silent(cmd: &mut Command) -> Result<bool> {
    let output = cmd
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .context("spawning subprocess")?;
    Ok(output.status.success())
}

/// Check out `target` (a rev, tag, branch, or `FETCH_HEAD`) in `clone_dir`,
/// self-healing the two corruption modes a pre-v3.0.0 concurrent-resolve
/// race could leave behind (#8): stray untracked files blocking the
/// checkout ("untracked working tree files would be overwritten"), or a
/// previous checkout simply parked on the wrong commit. This helper is only
/// used by the one-time EX initializer before it publishes readiness; it must
/// never run against a warm checkout that readers can access.
///
/// Returns `Ok(false)` (not an error) when `target` isn't resolvable at
/// all in this clone_dir -- the caller's fallback is to `git fetch` first.
async fn checkout_rev_robust(clone_dir: &Path, target: &str) -> Result<bool> {
    if try_run_silent(
        Command::new("git")
            .args(["checkout", target])
            .current_dir(clone_dir),
    )
    .await?
    {
        return Ok(true);
    }
    // First attempt failed -- most likely stray untracked files left by a
    // prior corrupted run. `git clean -fdx` clears untracked AND
    // gitignored files (safe here: clone_dir only ever holds the
    // checkout itself, wheel output lands in a separate out_dir), then
    // retry once. If `target` still isn't resolvable locally, this
    // second attempt fails the same way a doomed checkout always would.
    #[cfg(test)]
    record_checkout_repair(clone_dir);
    run_silent(
        Command::new("git")
            .args(["clean", "-fdx"])
            .current_dir(clone_dir),
        "git clean -fdx (repairing a corrupted checkout)",
    )
    .await?;
    try_run_silent(
        Command::new("git")
            .args(["checkout", target])
            .current_dir(clone_dir),
    )
    .await
}

async fn find_built_wheel(dir: &Path) -> Result<PathBuf> {
    let mut read = tokio::fs::read_dir(dir)
        .await
        .with_context(|| format!("opening wheel-build dir {}", dir.display()))?;
    let mut latest: Option<PathBuf> = None;
    while let Some(entry) = read
        .next_entry()
        .await
        .with_context(|| format!("reading wheel-build dir {}", dir.display()))?
    {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.ends_with(".whl") {
            latest = Some(path);
        }
    }
    latest.ok_or_else(|| anyhow!("no .whl produced in {}", dir.display()))
}

/// Sanitize a git URL into a filesystem-safe slug for cache key.
fn git_slug(url: &str) -> String {
    url.replace(['/', ':', '@'], "_")
        .replace("https___github.com_", "")
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHECKOUT_SUBPROCESS_HELPER: &str = "source_build::tests::git_checkout_subprocess_helper";
    const UV_BUILD_LIMIT_SUBPROCESS_HELPER: &str =
        "source_build::tests::uv_build_limit_subprocess_helper";

    struct CheckoutTestChild {
        child: Option<std::process::Child>,
        label: String,
    }

    impl CheckoutTestChild {
        fn spawn(label: &str, mode: &str, environment: &[(&str, String)]) -> Self {
            Self::spawn_exact(
                label,
                CHECKOUT_SUBPROCESS_HELPER,
                mode,
                environment,
            )
        }

        fn spawn_exact(
            label: &str,
            test_name: &str,
            mode: &str,
            environment: &[(&str, String)],
        ) -> Self {
            let mut command = std::process::Command::new(
                std::env::current_exe().expect("locate current Rust test executable"),
            );
            command
                .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
                .env("RETREAD_TEST_CHILD_MODE", mode)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            for (name, value) in environment {
                command.env(name, value);
            }
            let child = command.spawn().expect("spawn checkout test subprocess");
            Self {
                child: Some(child),
                label: label.to_string(),
            }
        }

        fn wait_for_signal(&mut self, path: &Path, timeout: Duration) {
            let deadline = std::time::Instant::now() + timeout;
            loop {
                if path.exists() {
                    return;
                }
                if let Some(status) = self
                    .child
                    .as_mut()
                    .expect("checkout test child already consumed")
                    .try_wait()
                    .expect("poll checkout test subprocess")
                {
                    let output = self
                        .child
                        .take()
                        .expect("checkout test child already consumed")
                        .wait_with_output()
                        .expect("collect checkout test subprocess output");
                    panic!(
                        "{} exited before signal {} ({status}):\nstdout:\n{}\nstderr:\n{}",
                        self.label,
                        path.display(),
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr),
                    );
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "{} timed out waiting for signal {}",
                    self.label,
                    path.display(),
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        }

        fn finish(mut self, timeout: Duration) {
            let deadline = std::time::Instant::now() + timeout;
            loop {
                let status = self
                    .child
                    .as_mut()
                    .expect("checkout test child already consumed")
                    .try_wait()
                    .expect("poll checkout test subprocess");
                if let Some(status) = status {
                    let output = self
                        .child
                        .take()
                        .expect("checkout test child already consumed")
                        .wait_with_output()
                        .expect("collect checkout test subprocess output");
                    assert!(
                        status.success(),
                        "{} failed ({status}):\nstdout:\n{}\nstderr:\n{}",
                        self.label,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr),
                    );
                    return;
                }
                if std::time::Instant::now() >= deadline {
                    let mut child = self
                        .child
                        .take()
                        .expect("checkout test child already consumed");
                    let _ = child.kill();
                    let output = child
                        .wait_with_output()
                        .expect("collect timed-out checkout test subprocess output");
                    panic!(
                        "{} timed out and was killed:\nstdout:\n{}\nstderr:\n{}",
                        self.label,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr),
                    );
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn recorded_uv_processes(
        state_dir: &Path,
        prefix: &str,
    ) -> std::collections::BTreeMap<String, u32> {
        let mut recorded = std::collections::BTreeMap::new();
        for entry in std::fs::read_dir(state_dir).expect("read fake-uv state directory") {
            let entry = entry.expect("read fake-uv state entry");
            let name = entry.file_name();
            let Some(id) = name.to_str().and_then(|name| name.strip_prefix(prefix)) else {
                continue;
            };
            let Ok(pid) = std::fs::read_to_string(entry.path()) else {
                continue;
            };
            let Ok(pid) = pid.trim().parse() else {
                continue;
            };
            recorded.insert(id.to_string(), pid);
        }
        recorded
    }

    #[cfg(target_os = "linux")]
    fn started_uv_processes(state_dir: &Path) -> std::collections::BTreeMap<String, u32> {
        recorded_uv_processes(state_dir, "started-")
    }

    #[cfg(target_os = "linux")]
    fn uv_grandchildren(state_dir: &Path) -> std::collections::BTreeMap<String, u32> {
        recorded_uv_processes(state_dir, "grandchild-")
    }

    #[cfg(target_os = "linux")]
    fn process_is_running(pid: u32) -> bool {
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
            return false;
        };
        // A killed child may briefly remain as a zombie until Tokio reaps it;
        // that is terminated for this regression's purposes.
        stat.rsplit_once(") ")
            .and_then(|(_, rest)| rest.chars().next())
            .is_some_and(|state| state != 'Z')
    }

    #[cfg(target_os = "linux")]
    async fn wait_for_test_condition(label: &str, mut condition: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !condition() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {label}"));
    }

    /// Subprocess-isolated because the production semaphore and parsed
    /// environment setting are intentionally initialized once per process.
    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn uv_build_limit_subprocess_helper() {
        if std::env::var("RETREAD_TEST_CHILD_MODE").as_deref() != Ok("uv-build-limit") {
            return;
        }
        let state_dir = PathBuf::from(
            std::env::var_os("RETREAD_TEST_UV_STATE").expect("missing fake-uv state directory"),
        );
        let ids = ["one", "two", "three"];
        let mut tasks = Vec::new();
        for id in ids {
            tasks.push(Some(tokio::spawn(async move {
                run_capturing_uv(&["build", id]).await
            })));
        }

        wait_for_test_condition("two capped uv children", || {
            started_uv_processes(&state_dir).len() >= 2
        })
        .await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        let first_wave = started_uv_processes(&state_dir);
        assert_eq!(
            first_wave.len(),
            2,
            "cap=2 allowed a third uv build to start: {first_wave:?}",
        );

        let cancelled_id = first_wave.keys().next().expect("one started uv id").clone();
        let cancelled_pid = first_wave[&cancelled_id];
        let cancelled_grandchild_pid = uv_grandchildren(&state_dir)
            .get(&cancelled_id)
            .copied()
            .expect("started fake uv recorded its grandchild pid");
        let cancelled_index = ids
            .iter()
            .position(|id| *id == cancelled_id)
            .expect("started uv id belongs to a task");
        let cancelled = tasks[cancelled_index]
            .take()
            .expect("cancelled task still present");
        cancelled.abort();
        assert!(
            cancelled.await.expect_err("aborted build task completed").is_cancelled(),
            "build task did not report cancellation",
        );
        wait_for_test_condition("cancelled uv child termination", || {
            !process_is_running(cancelled_pid)
        })
        .await;
        wait_for_test_condition("cancelled uv grandchild termination", || {
            !process_is_running(cancelled_grandchild_pid)
        })
        .await;
        wait_for_test_condition("third uv child after permit release", || {
            started_uv_processes(&state_dir).len() == 3
        })
        .await;

        for task in tasks.into_iter().flatten() {
            task.abort();
            assert!(task.await.expect_err("aborted build task completed").is_cancelled());
        }
        for pid in started_uv_processes(&state_dir).into_values() {
            wait_for_test_condition("fake uv cleanup", || !process_is_running(pid)).await;
        }
        for pid in uv_grandchildren(&state_dir).into_values() {
            wait_for_test_condition("fake uv grandchild cleanup", || !process_is_running(pid))
                .await;
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn uv_builds_are_capped_and_killed_on_cancellation() {
        use std::os::unix::fs::PermissionsExt;

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "retread-uv-build-limit-{}-{unique}",
            std::process::id()
        ));
        let bin_dir = base.join("bin");
        let state_dir = base.join("state");
        std::fs::create_dir_all(&bin_dir).unwrap();
        std::fs::create_dir_all(&state_dir).unwrap();
        let fake_uv = bin_dir.join("uv");
        std::fs::write(
            &fake_uv,
            b"#!/bin/sh\nsleep 30 &\ngrandchild=$!\ngrandchild_tmp=\"${RETREAD_TEST_UV_STATE}/.grandchild-$2-$$\"\nprintf '%s\\n' \"$grandchild\" > \"$grandchild_tmp\"\nmv \"$grandchild_tmp\" \"${RETREAD_TEST_UV_STATE}/grandchild-$2\"\nstarted_tmp=\"${RETREAD_TEST_UV_STATE}/.started-$2-$$\"\nprintf '%s\\n' \"$$\" > \"$started_tmp\"\nmv \"$started_tmp\" \"${RETREAD_TEST_UV_STATE}/started-$2\"\nwait \"$grandchild\"\n",
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&fake_uv).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&fake_uv, permissions).unwrap();
        let path = format!(
            "{}:{}",
            bin_dir.display(),
            std::env::var_os("PATH")
                .unwrap_or_default()
                .to_string_lossy()
        );

        CheckoutTestChild::spawn_exact(
            "uv build concurrency/cancellation probe",
            UV_BUILD_LIMIT_SUBPROCESS_HELPER,
            "uv-build-limit",
            &[
                ("RETREAD_MAX_CONCURRENT_BUILDS", "2".to_string()),
                (
                    "RETREAD_TEST_UV_STATE",
                    state_dir.display().to_string(),
                ),
                ("PATH", path),
            ],
        )
        .finish(Duration::from_secs(15));
        let _ = std::fs::remove_dir_all(&base);
    }

    impl Drop for CheckoutTestChild {
        fn drop(&mut self) {
            let Some(child) = self.child.as_mut() else {
                return;
            };
            if child.try_wait().ok().flatten().is_none() {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
    }

    async fn wait_for_checkout_signal(variable: &str) {
        let path = PathBuf::from(
            std::env::var_os(variable)
                .unwrap_or_else(|| panic!("checkout test subprocess missing {variable}")),
        );
        tokio::time::timeout(Duration::from_secs(5), async {
            while !path.exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {variable} at {}", path.display()));
    }

    #[test]
    fn git_slug_strips_github_prefix() {
        assert_eq!(
            git_slug("https://github.com/isaac-sim/IsaacLab.git"),
            "isaac-sim_IsaacLab.git"
        );
    }

    /// v0.13.3+ regression: every on-disk path component in the
    /// checkout-root path is independently bounded. Layout is
    /// cache/retread-git-clones/<slug<=24>/<sha12>, so the longest
    /// component should be the 24-char slug cap. Previously the
    /// (slug + 40-char raw SHA) flattened into one 60+ char
    /// component; combined with the rattler cache prefix and deep
    /// IsaacLab internals, pathnames tripped ENAMETOOLONG on git
    /// checkout.
    #[test]
    fn checkout_root_components_are_short() {
        let cache = std::path::Path::new("/tmp/cache");
        let p = git_checkout_root(
            "https://github.com/isaac-sim/IsaacLab.git",
            "867cbf9b7b4edbb03f32e1209c585a38cb3d8edf",
            cache,
        );
        let comps: Vec<String> = p
            .components()
            .filter_map(|c| c.as_os_str().to_str().map(String::from))
            .collect();
        // Last component is the 12-hex sha; second-to-last is the
        // slug (<=24 chars). Neither anywhere near NAME_MAX / 255.
        let last = comps.last().expect("at least one component");
        let parent = &comps[comps.len() - 2];
        assert_eq!(
            last.len(),
            12,
            "sha12 must be exactly 12 chars; got: {last}"
        );
        assert!(parent.len() <= 24, "slug must be <=24 chars; got {parent}");
    }

    /// The process registry must unify repeated opens before flocking, and its
    /// actual production RwLock must exclude the one-time writer from readers
    /// in both directions.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn process_clone_reader_and_writer_guards_never_overlap() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "retread-clone-rwlock-{}-{unique}",
            std::process::id()
        ));
        let clone_dir = base
            .join("retread-git-clones")
            .join("slug")
            .join("abcdef012345");
        std::fs::create_dir_all(clone_dir.parent().unwrap()).unwrap();
        let lock_path = clone_dir.with_extension("lock");
        let identity = CloneIdentity::new("https://example.com/repo.git", "rev");
        let first = registered_clone_lock(&lock_path, &identity).unwrap();
        let second = registered_clone_lock(&lock_path, &identity).unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "same inode must map to one process lock entry"
        );

        let writer = Arc::clone(&first.local).write_owned().await;
        let (reader_entered_tx, mut reader_entered_rx) = tokio::sync::oneshot::channel();
        let (reader_release_tx, reader_release_rx) = tokio::sync::oneshot::channel();
        let reader_lock = Arc::clone(&second.local);
        let reader_task = tokio::spawn(async move {
            let _reader = reader_lock.read_owned().await;
            let _ = reader_entered_tx.send(());
            let _ = reader_release_rx.await;
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut reader_entered_rx)
                .await
                .is_err(),
            "reader entered while the one-time writer was held"
        );
        drop(writer);
        tokio::time::timeout(Duration::from_secs(2), &mut reader_entered_rx)
            .await
            .expect("reader did not enter after writer release")
            .expect("reader entry sender dropped");

        let (writer_entered_tx, mut writer_entered_rx) = tokio::sync::oneshot::channel();
        let writer_lock = Arc::clone(&first.local);
        let writer_task = tokio::spawn(async move {
            let _writer = writer_lock.write_owned().await;
            let _ = writer_entered_tx.send(());
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut writer_entered_rx)
                .await
                .is_err(),
            "writer entered while a reader was held"
        );
        let _ = reader_release_tx.send(());
        reader_task.await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), &mut writer_entered_rx)
            .await
            .expect("writer did not enter after reader release")
            .expect("writer entry sender dropped");
        writer_task.await.unwrap();

        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn process_clone_registry_unifies_path_aliases_by_inode() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "retread-clone-alias-{}-{unique}",
            std::process::id()
        ));
        let real_cache = base.join("real-cache");
        let alias_cache = base.join("alias-cache");
        let real_parent = real_cache.join("retread-git-clones").join("slug");
        std::fs::create_dir_all(&real_parent).unwrap();
        std::os::unix::fs::symlink(&real_cache, &alias_cache).unwrap();
        let real_lock = real_parent.join("abcdef012345.lock");
        let alias_lock = alias_cache
            .join("retread-git-clones")
            .join("slug")
            .join("abcdef012345.lock");
        let identity = CloneIdentity::new("https://example.com/alias.git", "rev");
        let real = registered_clone_lock(&real_lock, &identity).unwrap();
        let alias = registered_clone_lock(&alias_lock, &identity).unwrap();
        assert!(
            Arc::ptr_eq(&real, &alias),
            "two paths to one lock inode created separate process entries"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Different (url, rev) pairs must NOT collide on disk -- the
    /// rev is the only thing distinguishing two checkouts of the same
    /// repo at different revisions.
    #[test]
    fn checkout_root_distinct_revs_do_not_collide() {
        let cache = std::path::Path::new("/tmp/cache");
        let a = git_checkout_root("https://example.com/r.git", "rev-a", cache);
        let b = git_checkout_root("https://example.com/r.git", "rev-b", cache);
        assert_ne!(a, b);
    }

    /// And two DIFFERENT repos at the same revision name must also
    /// differ (the url is hashed into the key alongside the rev).
    #[test]
    fn checkout_root_distinct_urls_do_not_collide() {
        let cache = std::path::Path::new("/tmp/cache");
        let a = git_checkout_root("https://example.com/r1.git", "main", cache);
        let b = git_checkout_root("https://example.com/r2.git", "main", cache);
        assert_ne!(a, b);
    }

    // ---------------------------------------------------------------------------
    // Determinism guard: is_nondeterministic_version
    // ---------------------------------------------------------------------------

    #[test]
    fn deterministic_version_not_flagged() {
        // Static release versions must NOT trigger the guard.
        assert!(!is_nondeterministic_version("mylib-1.1.1-py3-none-any.whl"));
        assert!(!is_nondeterministic_version(
            "newton-1.3.0-py3-none-any.whl"
        ));
        assert!(!is_nondeterministic_version(
            "genesis_world-1.1.1-py3-none-any.whl"
        ));
        assert!(!is_nondeterministic_version(
            "foo-2.0.0rc1-py3-none-any.whl"
        ));
        assert!(!is_nondeterministic_version(
            "bar-0.1.0.post1-py3-none-any.whl"
        ));
    }

    #[test]
    fn dev_version_is_flagged() {
        // .devN suffix (development distance without local segment).
        assert!(is_nondeterministic_version(
            "mylib-1.1.1.dev4-py3-none-any.whl"
        ));
        assert!(is_nondeterministic_version(
            "mylib-0.1.dev123-py3-none-any.whl"
        ));
    }

    #[test]
    fn date_segment_is_flagged() {
        // .dYYYYMMDD local date segment produced by setuptools_scm.
        assert!(is_nondeterministic_version(
            "mylib-1.0.dev4+g1234567.d20250101-py3-none-any.whl"
        ));
        assert!(is_nondeterministic_version(
            "mylib-1.0.dev0+g0000000.d20991231-py3-none-any.whl"
        ));
    }

    #[test]
    fn local_git_sha_segment_is_flagged() {
        // +g<hexchars> local git-hash segment.
        assert!(is_nondeterministic_version(
            "mylib-1.0+gabcdef0-py3-none-any.whl"
        ));
        assert!(is_nondeterministic_version(
            "mylib-2.0.post0+g1234abc-py3-none-any.whl"
        ));
    }

    /// Determinism guard (Amendment 3): build_wheel_from_sdist_url must warn
    /// on a non-reproducible version and be silent on a static one.
    /// Tests the guard logic that was added to mirror build_wheel_from_git.
    #[test]
    fn sdist_determinism_guard_matches_git_guard() {
        // Static released version (e.g. gym 0.26.2) — NO warn.
        assert!(
            !is_nondeterministic_version("gym-0.26.2-py3-none-any.whl"),
            "gym 0.26.2 is a static version; determinism guard must NOT fire"
        );
        // .dYYYYMMDD date segment — MUST warn.
        assert!(
            is_nondeterministic_version("mypkg-1.0.dev4+g1234567.d20250101-py3-none-any.whl"),
            "setuptools_scm date suffix must trigger determinism guard"
        );
        // .devN without date — MUST warn.
        assert!(
            is_nondeterministic_version("mypkg-0.1.dev5-py3-none-any.whl"),
            ".devN suffix must trigger determinism guard"
        );
        // +g<sha> local version — MUST warn.
        assert!(
            is_nondeterministic_version("mypkg-1.0+gabcdef0-py3-none-any.whl"),
            "+g<sha> local version must trigger determinism guard"
        );
    }

    #[tokio::test]
    async fn sdist_hash_mismatch_fails_before_cache_write_or_build() {
        use std::io::{Read as _, Write as _};

        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 2048];
            let _ = stream.read(&mut request).unwrap();
            let body = b"untrusted-sdist-bytes";
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len(),
            )
            .unwrap();
            stream.write_all(body).unwrap();
        });
        let url = url::Url::parse(&format!("http://{address}/demo-1.0.tar.gz")).unwrap();
        let out_dir = std::env::temp_dir().join(format!(
            "retread-sdist-hash-mismatch-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));

        let err = build_wheel_from_sdist_url(&url, &out_dir, "3.11", Some(&"00".repeat(32)))
            .await
            .unwrap_err();
        server.join().unwrap();

        assert!(format!("{err:#}").contains("sdist sha256 mismatch"));
        assert!(
            !out_dir.exists(),
            "unverified sdist bytes must not publish a cache directory"
        );
    }

    // ---------------------------------------------------------------------------
    // Local git fixture: build_wheel_from_git resolves and retains checkout
    // ---------------------------------------------------------------------------

    /// Verifies that `build_wheel_from_git` returns a 40-character resolved
    /// SHA and that the SHA is stable (calling again with the same rev returns
    /// the same SHA). Uses a minimal local git repo so no network access is
    /// required and CI stays fast.
    #[tokio::test]
    #[ignore = "live: builds a git wheel via uv (needs uv + git on PATH); run with --include-ignored"]
    async fn build_wheel_from_git_returns_resolved_sha() {
        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("retread-gitfixture-{pid}"));
        let repo = base.join("repo");
        std::fs::create_dir_all(&repo).expect("create repo dir");

        // Init git repo.
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

        // Write a minimal but buildable Python package.
        std::fs::write(
            repo.join("pyproject.toml"),
            r#"[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "retread-test-fixture"
version = "0.1.0"
"#,
        )
        .expect("write pyproject");
        std::fs::write(repo.join("README.md"), "test fixture").expect("write README");

        run_git(&["add", "."], &repo);
        run_git(&["commit", "-m", "initial"], &repo);

        // Get the commit SHA directly.
        let sha_output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&repo)
            .output()
            .expect("git rev-parse");
        let expected_sha = String::from_utf8_lossy(&sha_output.stdout)
            .trim()
            .to_string();
        assert_eq!(
            expected_sha.len(),
            40,
            "git rev-parse HEAD must be 40 chars"
        );

        let cache_dir = base.join("cache");
        let out_dir = base.join("out");
        std::fs::create_dir_all(&cache_dir).expect("cache dir");
        std::fs::create_dir_all(&out_dir).expect("out dir");
        let repo_url = format!("file://{}", repo.display());

        let (wheel_path, resolved_sha) =
            build_wheel_from_git(&repo_url, &expected_sha, ".", &cache_dir, &out_dir, "3.11")
                .await
                .expect("build_wheel_from_git");

        // The returned SHA must match what git reports.
        assert_eq!(
            resolved_sha, expected_sha,
            "resolved_sha must equal the commit SHA"
        );
        assert_eq!(resolved_sha.len(), 40, "resolved_sha must be 40 hex chars");
        // A wheel must have been produced.
        assert!(
            wheel_path.extension().is_some_and(|e| e == "whl"),
            "built file must be a .whl"
        );
        // The static version "0.1.0" must NOT be flagged as non-deterministic.
        let filename = wheel_path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("filename");
        assert!(
            !is_nondeterministic_version(filename),
            "a static version should not be flagged: {filename}"
        );

        // Cleanup.
        let _ = std::fs::remove_dir_all(&base);
    }

    struct GitCheckoutFixture {
        base: PathBuf,
        cache: PathBuf,
        url: String,
        rev1: String,
        rev2: String,
    }

    impl Drop for GitCheckoutFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.base);
        }
    }

    fn run_fixture_git(args: &[&str], directory: &Path) -> String {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(directory)
            .env("GIT_AUTHOR_NAME", "retread-test")
            .env("GIT_AUTHOR_EMAIL", "retread-test@example.com")
            .env("GIT_COMMITTER_NAME", "retread-test")
            .env("GIT_COMMITTER_EMAIL", "retread-test@example.com")
            .output()
            .expect("spawn fixture git");
        assert!(
            output.status.success(),
            "git {args:?} failed in {}: {}",
            directory.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn git_checkout_fixture(label: &str) -> GitCheckoutFixture {
        static NEXT_FIXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let base =
            std::env::temp_dir().join(format!("retread-{label}-{}-{sequence}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let repo = base.join("repo");
        let cache = base.join("cache");
        std::fs::create_dir_all(&repo).expect("create fixture repo");
        std::fs::create_dir_all(&cache).expect("create fixture cache");
        run_fixture_git(&["init", "-b", "main"], &repo);
        std::fs::write(repo.join("base.txt"), "base\n").expect("write base file");
        run_fixture_git(&["add", "."], &repo);
        run_fixture_git(&["commit", "-m", "base"], &repo);
        let rev1 = run_fixture_git(&["rev-parse", "HEAD"], &repo);
        std::fs::write(repo.join("extra.txt"), "tracked-content\n")
            .expect("write second-revision file");
        run_fixture_git(&["add", "."], &repo);
        run_fixture_git(&["commit", "-m", "add extra"], &repo);
        let rev2 = run_fixture_git(&["rev-parse", "HEAD"], &repo);
        let url = format!("file://{}", repo.display());
        GitCheckoutFixture {
            base,
            cache,
            url,
            rev1,
            rev2,
        }
    }

    fn seed_unmarked_checkout(fixture: &GitCheckoutFixture, checkout_rev: &str) -> PathBuf {
        let clone_dir = git_checkout_root(&fixture.url, &fixture.rev2, &fixture.cache);
        std::fs::create_dir_all(clone_dir.parent().unwrap()).expect("create clone parent");
        let clone_arg = clone_dir.to_str().expect("utf8 clone path");
        run_fixture_git(
            &["clone", "--no-checkout", &fixture.url, clone_arg],
            &fixture.base,
        );
        run_fixture_git(&["checkout", "--force", checkout_rev], &clone_dir);
        assert!(
            !checkout_ready_marker(&clone_dir).exists(),
            "legacy fixture must begin unpublished"
        );
        clone_dir
    }

    /// Entry point used only by parent tests that need a killable process
    /// boundary. Running it without a mode is an intentional no-op so the
    /// normal libtest pass can include this helper safely.
    #[test]
    fn git_checkout_subprocess_helper() {
        let Ok(mode) = std::env::var("RETREAD_TEST_CHILD_MODE") else {
            return;
        };
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("build checkout test runtime");

        match mode.as_str() {
            "same-process-readers" => runtime.block_on(async {
                let fixture = git_checkout_fixture("child-two-readers");
                let barrier = Arc::new(tokio::sync::Barrier::new(3));

                let first_barrier = Arc::clone(&barrier);
                let first_url = fixture.url.clone();
                let first_rev = fixture.rev2.clone();
                let first_cache = fixture.cache.clone();
                let first = tokio::spawn(async move {
                    first_barrier.wait().await;
                    ensure_git_checkout(&first_url, &first_rev, &first_cache).await
                });

                let second_barrier = Arc::clone(&barrier);
                let second_url = fixture.url.clone();
                let second_rev = fixture.rev2.clone();
                let second_cache = fixture.cache.clone();
                let second = tokio::spawn(async move {
                    second_barrier.wait().await;
                    ensure_git_checkout(&second_url, &second_rev, &second_cache).await
                });

                barrier.wait().await;
                wait_for_checkout_signal("RETREAD_TEST_EXCLUSIVE_HELD").await;
                wait_for_checkout_signal("RETREAD_TEST_SECOND_REGISTRATION").await;
                assert!(
                    !first.is_finished() && !second.is_finished(),
                    "a checkout reader completed while initialization was paused under EX"
                );
                let release = PathBuf::from(
                    std::env::var_os("RETREAD_TEST_EXCLUSIVE_RELEASE")
                        .expect("missing initializer release path"),
                );
                std::fs::write(&release, b"release\n").expect("release checkout initializer");

                let (first, second) = tokio::time::timeout(Duration::from_secs(10), async {
                    (first.await, second.await)
                })
                .await
                .expect("same-process checkout readers deadlocked");
                let first = first
                    .expect("first reader task panicked")
                    .expect("first reader failed");
                let second = second
                    .expect("second reader task panicked")
                    .expect("second reader failed");
                assert!(Arc::ptr_eq(&first.lease.lock, &second.lease.lock));
                assert_eq!(
                    first.lease.lock.os_acquisitions.load(Ordering::Relaxed),
                    1,
                    "same-process callers took more than one OS flock"
                );
            }),
            "warm-reader" => runtime.block_on(async {
                let url = std::env::var("RETREAD_TEST_GIT_URL").expect("missing git URL");
                let rev = std::env::var("RETREAD_TEST_GIT_REV").expect("missing git rev");
                let cache = PathBuf::from(
                    std::env::var_os("RETREAD_TEST_GIT_CACHE").expect("missing git cache"),
                );
                let _checkout = tokio::time::timeout(
                    Duration::from_secs(10),
                    ensure_git_checkout(&url, &rev, &cache),
                )
                .await
                .expect("warm reader timed out")
                .expect("warm reader failed");
                signal_checkout_test_path("RETREAD_TEST_READER_ENTERED")
                    .expect("signal warm reader entry");
            }),
            other => panic!("unknown checkout test child mode `{other}`"),
        }
    }

    /// The first access repairs an unmarked legacy tree and publishes it. Any
    /// later warm access is read-only: a sentinel that `git clean -fdx` would
    /// remove and an index lock that would force wipe/reclone both survive.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn git_checkout_repairs_once_then_warm_tree_is_immutable() {
        let fixture = git_checkout_fixture("clone-once");
        let clone_dir = seed_unmarked_checkout(&fixture, &fixture.rev1);
        std::fs::write(clone_dir.join("extra.txt"), "stray-untracked\n")
            .expect("write checkout conflict");

        let first = tokio::time::timeout(
            Duration::from_secs(10),
            ensure_git_checkout(&fixture.url, &fixture.rev2, &fixture.cache),
        )
        .await
        .expect("initial checkout repair timed out")
        .expect("initial checkout repair failed");
        assert_eq!(
            std::fs::read_to_string(clone_dir.join("extra.txt")).unwrap(),
            "tracked-content\n"
        );
        assert!(matches!(
            checkout_marker_state(&clone_dir, &CloneIdentity::new(&fixture.url, &fixture.rev2))
                .unwrap(),
            CheckoutMarkerState::Matching
        ));
        assert_eq!(
            first.lease.lock.os_acquisitions.load(Ordering::Relaxed),
            1,
            "initialization must take the OS flock exactly once"
        );
        assert_eq!(
            checkout_repair_count(&clone_dir),
            1,
            "the unpublished checkout must run destructive repair exactly once"
        );

        let sentinel = clone_dir.join("warm-sentinel.txt");
        let index_lock = clone_dir.join(".git").join("index.lock");
        std::fs::write(&sentinel, "preserve me\n").unwrap();
        std::fs::write(&index_lock, "force any checkout to fail\n").unwrap();
        let marker_before = std::fs::read(checkout_ready_marker(&clone_dir)).unwrap();
        let head_log = clone_dir.join(".git").join("logs").join("HEAD");
        let head_log_before = std::fs::read(&head_log).unwrap();

        let url = fixture.url.clone();
        let rev = fixture.rev2.clone();
        let cache = fixture.cache.clone();
        let second_task =
            tokio::spawn(async move { ensure_git_checkout(&url, &rev, &cache).await });
        let second = tokio::time::timeout(Duration::from_secs(2), second_task)
            .await
            .expect("same-process warm reader deadlocked")
            .expect("warm reader task panicked")
            .expect("warm reader failed");
        assert!(Arc::ptr_eq(&first.lease.lock, &second.lease.lock));
        assert_eq!(
            second.lease.lock.os_acquisitions.load(Ordering::Relaxed),
            1,
            "warm reader must reuse the process's one OS flock"
        );
        let fresh_reader_entered = fixture.base.join("fresh-reader-entered");
        CheckoutTestChild::spawn(
            "fresh-process warm checkout reader",
            "warm-reader",
            &[
                ("RETREAD_TEST_GIT_URL", fixture.url.clone()),
                ("RETREAD_TEST_GIT_REV", fixture.rev2.clone()),
                (
                    "RETREAD_TEST_GIT_CACHE",
                    fixture.cache.display().to_string(),
                ),
                (
                    "RETREAD_TEST_READER_ENTERED",
                    fresh_reader_entered.display().to_string(),
                ),
            ],
        )
        .finish(Duration::from_secs(15));
        assert!(
            fresh_reader_entered.exists(),
            "fresh process never acquired the warm shared-reader path"
        );
        assert_eq!(std::fs::read_to_string(&sentinel).unwrap(), "preserve me\n");
        assert!(index_lock.exists(), "warm access ran checkout or recloned");
        assert_eq!(
            std::fs::read(checkout_ready_marker(&clone_dir)).unwrap(),
            marker_before
        );
        assert_eq!(std::fs::read(&head_log).unwrap(), head_log_before);
        assert_eq!(
            checkout_repair_count(&clone_dir),
            1,
            "warm access must not run destructive repair again"
        );
    }

    /// The subprocess boundary makes this regression killable even if a future
    /// implementation blocks a runtime thread in `flock`. Test hooks pause the
    /// winner after EX and prove the second worker registered before release.
    #[test]
    fn same_process_two_thread_checkout_reads_do_not_deadlock() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let signals = std::env::temp_dir().join(format!(
            "retread-two-reader-signals-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&signals).unwrap();
        let exclusive_held = signals.join("exclusive-held");
        let exclusive_release = signals.join("exclusive-release");
        let second_registration = signals.join("second-registration");
        CheckoutTestChild::spawn(
            "same-process checkout reader probe",
            "same-process-readers",
            &[
                (
                    "RETREAD_TEST_EXCLUSIVE_HELD",
                    exclusive_held.display().to_string(),
                ),
                (
                    "RETREAD_TEST_EXCLUSIVE_RELEASE",
                    exclusive_release.display().to_string(),
                ),
                (
                    "RETREAD_TEST_SECOND_REGISTRATION",
                    second_registration.display().to_string(),
                ),
            ],
        )
        .finish(Duration::from_secs(15));
        assert!(exclusive_held.exists());
        assert!(second_registration.exists());
        assert!(exclusive_release.exists());
        let _ = std::fs::remove_dir_all(&signals);
    }

    /// A production reader must observe the same cross-process writer lock as
    /// the one-time initializer. The child signals only after an SH attempt is
    /// denied, so the exclusion assertion does not depend on scheduler timing.
    #[cfg(unix)]
    #[test]
    fn published_checkout_reader_waits_for_cross_process_writer() {
        let fixture = git_checkout_fixture("cross-process-rw");
        let clone_dir = seed_unmarked_checkout(&fixture, &fixture.rev2);
        publish_checkout_ready(&clone_dir, &CloneIdentity::new(&fixture.url, &fixture.rev2))
            .expect("publish cross-process reader fixture");

        let lock_path = clone_dir.with_extension("lock");
        let writer = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .expect("open cross-process writer lock");
        fs4::fs_std::FileExt::lock_exclusive(&writer).expect("acquire cross-process writer lock");

        let shared_blocked = fixture.base.join("shared-blocked");
        let reader_entered = fixture.base.join("reader-entered");
        let mut child = CheckoutTestChild::spawn(
            "cross-process shared-reader probe",
            "warm-reader",
            &[
                ("RETREAD_TEST_GIT_URL", fixture.url.clone()),
                ("RETREAD_TEST_GIT_REV", fixture.rev2.clone()),
                (
                    "RETREAD_TEST_GIT_CACHE",
                    fixture.cache.display().to_string(),
                ),
                (
                    "RETREAD_TEST_SHARED_BLOCKED",
                    shared_blocked.display().to_string(),
                ),
                (
                    "RETREAD_TEST_READER_ENTERED",
                    reader_entered.display().to_string(),
                ),
            ],
        );
        child.wait_for_signal(&shared_blocked, Duration::from_secs(5));
        assert!(
            !reader_entered.exists(),
            "shared reader entered while the cross-process writer held EX"
        );

        fs4::fs_std::FileExt::unlock(&writer).expect("release cross-process writer lock");
        child.finish(Duration::from_secs(10));
        assert!(
            reader_entered.exists(),
            "shared reader did not enter after the writer released EX"
        );
    }

    /// Preserve v3.0.3's deep-corruption recovery only at the migration
    /// boundary: an unmarked legacy clone may be wiped once, before readers.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unmarked_git_dir_corruption_is_recloned_before_publication() {
        let fixture = git_checkout_fixture("legacy-reclone");
        let clone_dir = seed_unmarked_checkout(&fixture, &fixture.rev1);
        std::fs::write(clone_dir.join(".git").join("index.lock"), "stale\n").unwrap();

        let checkout = tokio::time::timeout(
            Duration::from_secs(10),
            ensure_git_checkout(&fixture.url, &fixture.rev2, &fixture.cache),
        )
        .await
        .expect("legacy re-clone timed out")
        .expect("legacy re-clone failed");
        assert_eq!(checkout.root(), clone_dir);
        assert!(!clone_dir.join(".git").join("index.lock").exists());
        assert!(matches!(
            checkout_marker_state(&clone_dir, &CloneIdentity::new(&fixture.url, &fixture.rev2))
                .unwrap(),
            CheckoutMarkerState::Matching
        ));
    }

    /// A failed initializer must release EX and leave the process registry
    /// retryable on the same fd. Otherwise one bad clone attempt permanently
    /// wedges every later worker in the same backend process.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_checkout_initialization_releases_lock_for_retry() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "retread-retry-init-{}-{unique}",
            std::process::id()
        ));
        let repo = base.join("repo-created-after-failure");
        let cache = base.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        let url = format!("file://{}", repo.display());

        let first = tokio::time::timeout(
            Duration::from_secs(10),
            ensure_git_checkout(&url, "main", &cache),
        )
        .await
        .expect("failing checkout attempt deadlocked");
        assert!(first.is_err(), "missing source unexpectedly initialized");
        let clone_dir = git_checkout_root(&url, "main", &cache);
        assert!(
            !checkout_ready_marker(&clone_dir).exists(),
            "failed initialization published readiness"
        );

        std::fs::create_dir_all(&repo).unwrap();
        run_fixture_git(&["init", "-b", "main"], &repo);
        std::fs::write(repo.join("after-retry.txt"), "ready\n").unwrap();
        run_fixture_git(&["add", "."], &repo);
        run_fixture_git(&["commit", "-m", "available"], &repo);

        let checkout = tokio::time::timeout(
            Duration::from_secs(10),
            ensure_git_checkout(&url, "main", &cache),
        )
        .await
        .expect("retry after failed initialization deadlocked")
        .expect("retry after failed initialization failed");
        assert_eq!(checkout.root(), clone_dir);
        assert_eq!(
            std::fs::read_to_string(clone_dir.join("after-retry.txt")).unwrap(),
            "ready\n"
        );
        assert_eq!(
            checkout.lease.lock.os_acquisitions.load(Ordering::Relaxed),
            2,
            "retry must reuse the registry fd after releasing the failed EX"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn git_wheel_cache_dir_is_keyed_by_all_inputs() {
        let a = git_wheel_cache_dir(
            "https://github.com/x/y.git",
            "a".repeat(40).as_str(),
            "sub",
            "3.11",
        );
        // Same inputs => same dir (deterministic).
        let a2 = git_wheel_cache_dir(
            "https://github.com/x/y.git",
            "a".repeat(40).as_str(),
            "sub",
            "3.11",
        );
        assert_eq!(a, a2);
        // Any input change => different dir.
        let b = git_wheel_cache_dir(
            "https://github.com/x/y.git",
            "b".repeat(40).as_str(),
            "sub",
            "3.11",
        );
        let c = git_wheel_cache_dir(
            "https://github.com/x/y.git",
            "a".repeat(40).as_str(),
            "other",
            "3.11",
        );
        let d = git_wheel_cache_dir(
            "https://github.com/x/y.git",
            "a".repeat(40).as_str(),
            "sub",
            "3.12",
        );
        let e = git_wheel_cache_dir(
            "https://github.com/x/z.git",
            "a".repeat(40).as_str(),
            "sub",
            "3.11",
        );
        for other in [&b, &c, &d, &e] {
            assert_ne!(&a, other);
        }
    }

    #[tokio::test]
    async fn git_wheel_cache_store_then_lookup_round_trips() {
        let base =
            std::env::temp_dir().join(format!("retread-gitwheel-cache-{}", std::process::id()));
        let cache_dir = base.join("shared");
        let out_dir = base.join("out");
        let built_dir = base.join("built");
        std::fs::create_dir_all(&built_dir).unwrap();

        // Empty cache: miss.
        let miss = git_wheel_cache_lookup(&cache_dir, &out_dir).await.unwrap();
        assert!(miss.is_none(), "empty cache must miss");

        // Store a freshly-"built" wheel, then look it up into out_dir.
        let wheel = built_dir.join("pkg-1.0.0-py3-none-any.whl");
        std::fs::write(&wheel, b"wheel bytes").unwrap();
        git_wheel_cache_store(&wheel, &cache_dir).await;
        assert!(
            cache_dir.join("pkg-1.0.0-py3-none-any.whl").is_file(),
            "store must persist the raw wheel into the shared cache dir"
        );

        let hit = git_wheel_cache_lookup(&cache_dir, &out_dir)
            .await
            .unwrap()
            .expect("populated cache must hit");
        assert_eq!(hit, out_dir.join("pkg-1.0.0-py3-none-any.whl"));
        assert_eq!(std::fs::read(&hit).unwrap(), b"wheel bytes");

        // Post-processed variants must never be served from the cache
        // (they are pack/config-specific and regenerate downstream).
        std::fs::write(
            cache_dir.join("pkg-1.0.0-py3-none-any.injected.whl"),
            b"processed",
        )
        .unwrap();
        let hit2 = git_wheel_cache_lookup(&cache_dir, &out_dir)
            .await
            .unwrap()
            .expect("raw wheel still present");
        assert!(
            !hit2.to_string_lossy().contains(".injected."),
            "cache lookup must skip retread-processed variants: {}",
            hit2.display()
        );

        let _ = std::fs::remove_dir_all(&base);
    }
}
