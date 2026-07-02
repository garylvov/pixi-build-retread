//! Build a `.whl` from a local path or git checkout via `pip wheel`.
//!
//! Used by `[retread-wheels]` entries that take `path = "..."` or
//! `git = "..."` instead of the PyPI `version + index` form. The
//! produced wheel goes through the same auto-bundle + METADATA-rewrite
//! pipeline as any PyPI-resolved wheel.

use std::path::{Path, PathBuf};
use std::process::Stdio;

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
) -> Result<PathBuf> {
    tokio::fs::create_dir_all(out_dir)
        .await
        .with_context(|| format!("creating sdist-build out dir {}", out_dir.display()))?;

    // Cache: if a wheel was already built here, reuse it.
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
        .and_then(|f| if f.is_empty() { None } else { Some(f) })
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

/// Compute the on-disk source-tree directory that
/// [`build_wheel_from_git`] will (or did) build from, without doing
/// any clone work. Lets callers feed the same directory into
/// `wheel_inject::inject` after the wheel is built.
pub fn git_source_root(url: &str, rev: &str, subdirectory: &str, cache_dir: &Path) -> PathBuf {
    git_checkout_root(url, rev, cache_dir).join(subdirectory)
}

/// Compute the on-disk *checkout* directory for a (url, rev) pair --
/// the parent of [`git_source_root`]'s subdirectory join. Used by the
/// v0.12.0 auto-data-files inject so the WHOLE upstream repo (minus
/// `.gitignore`'d paths and minus subdirectories already shipped as
/// wheels by sibling entries in the same bundle) can ride along into
/// the conda env at `$PREFIX/lib/<rel>`.
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

/// Ensure `clone_dir` is a git clone of `url` checked out at `rev`.
/// Clones (with `--no-checkout`) only if `clone_dir` doesn't exist yet;
/// otherwise reuses it. Always runs the checkout dance regardless --
/// see the v3.0.2 comment at the call site for why an existing
/// clone_dir can't be trusted blindly. Callers hold the per-clone_dir
/// lock for the duration; this function does not lock.
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

/// Clone a git URL at a specific revision into `cache_dir`, then build
/// the wheel for `subdirectory` (relative to the repo root, defaulting
/// to ".").
///
/// Clones are cached by (url, rev) so repeated `conda/outputs` calls
/// for the same workspace don't re-clone. `rev` can be a commit SHA,
/// tag, or branch name.
///
/// Returns `(wheel_path, resolved_sha)` where `resolved_sha` is the
/// 40-character git SHA obtained from `git rev-parse HEAD` after
/// checkout. This is the **canonical** commit identity that should be
/// stored in `GitWheelSource.rev` so that a branch/tag/HEAD ref at
/// produce time is pinned to a specific commit in the lock.
///
/// # Determinism guard
///
/// After the build, the emitted wheel filename is checked for markers
/// that indicate a non-reproducible `setuptools_scm` version:
/// - `.devN` segments (e.g. `1.0.dev4`)
/// - `.dYYYYMMDD` date segments (e.g. `1.0.dev4+g1234567.d20250101`)
/// - local `+g<sha>` segments
///
/// When detected, `tracing::warn!` is emitted. Such versions drift
/// across calendar days even when the commit SHA is pinned, causing
/// `lock drift` (the filename/version in the lock changes every day
/// even though the inputs have not changed). To fix this the upstream
/// project must tag a release or set `SETUPTOOLS_SCM_PRETEND_VERSION`.
pub async fn build_wheel_from_git(
    url: &str,
    rev: &str,
    subdirectory: &str,
    cache_dir: &Path,
    out_dir: &Path,
    python_version: &str,
) -> Result<(PathBuf, String)> {
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

    // Multiple [retread-wheels] entries commonly share one (url, rev) --
    // e.g. IsaacLab's 14+ `from = "isaaclab"` entries that differ only by
    // `subdirectory` -- and clone into the SAME clone_dir. Without a lock,
    // concurrent resolves (either multiple retread backend processes
    // solving different environments in parallel, since one retread
    // process only serializes RPCs within itself, or overlapping
    // sibling-entry resolves) race on that one shared working tree:
    // one's `git checkout` can land mid-way through another's `git
    // fetch`, leaving HEAD parked on the wrong commit or aborting with
    // "untracked working tree files would be overwritten". A per-(url,
    // rev) exclusive file lock (same mechanism rattler_cache uses to
    // guard its package cache dir) serializes clone/fetch/checkout so
    // only one resolver ever mutates a given clone_dir at a time; the
    // rest block on the lock, then see the completed checkout below.
    let lock_path = clone_dir.with_extension("lock");
    let lock_file = {
        let lock_path = lock_path.clone();
        tokio::task::spawn_blocking(move || -> Result<std::fs::File> {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .open(&lock_path)
                .with_context(|| format!("opening git-clone lock file {}", lock_path.display()))?;
            // fs4's lock_exclusive is a blocking syscall (flock/LockFileEx)
            // regardless of file type, so it must run on a blocking thread
            // rather than the async executor -- same pattern rattler_cache
            // uses for its own package-cache global lock.
            fs4::fs_std::FileExt::lock_exclusive(&file)
                .with_context(|| format!("locking git-clone lock file {}", lock_path.display()))?;
            Ok(file)
        })
        .await
        .context("git-clone lock task panicked")??
    };

    // v3.0.2 (#8 follow-up): ALWAYS run the full clone-if-missing +
    // checkout dance, even when clone_dir already existed. A directory
    // that survived a pre-v3.0.0 race (fetched `rev` but never actually
    // checked it out, checked out an unrelated commit, or left untracked
    // files mid-checkout that block any future `git checkout`)
    // previously slipped past unrepaired -- every subsequent run reused
    // that broken checkout forever, since `clone_dir.exists()` alone
    // can't tell a healthy checkout from a corrupted one.
    // `clone_and_checkout` is a cheap, safe no-op when the tree is
    // already correct (common case), and self-heals a corrupted
    // WORKING TREE via `checkout_rev_robust`'s `git clean -fdx` retry.
    //
    // v3.0.3: `git clean -fdx` only repairs the working tree -- it can't
    // fix a corrupted `.git` itself (bad refs, missing objects, a stale
    // `index.lock`), which is what #8 hit next: "git checkout FETCH_HEAD
    // failed even after cleaning the working tree." A clone_dir that
    // took concurrent hits from multiple pre-lock resolvers over its
    // lifetime can end up broken at that deeper level. When the
    // working-tree-level repair still isn't enough, wipe clone_dir
    // entirely and re-clone from scratch once -- the only fix that's
    // correct regardless of what kind of corruption is actually there.
    if let Err(e) = clone_and_checkout(&clone_dir, url, rev).await {
        tracing::warn!(
            url = %url, rev = %rev, error = %format!("{e:#}"),
            path = %clone_dir.display(),
            "git clone/checkout failed even after working-tree repair; \
             wiping the clone dir and re-cloning from scratch",
        );
        tokio::fs::remove_dir_all(&clone_dir)
            .await
            .with_context(|| format!("wiping corrupted clone dir {}", clone_dir.display()))?;
        clone_and_checkout(&clone_dir, url, rev)
            .await
            .with_context(|| {
                format!(
                    "re-clone after wiping corrupted dir still failed for {url}@{rev}"
                )
            })?;
    }

    // Release the lock now that the clone_dir holds a complete checkout;
    // the remaining work (reading subdirectory, `git rev-parse`, building
    // the wheel) only reads the tree and is safe to run concurrently with
    // other entries once the checkout itself is settled.
    tokio::task::spawn_blocking(move || fs4::fs_std::FileExt::unlock(&lock_file))
        .await
        .context("git-clone unlock task panicked")?
        .with_context(|| format!("unlocking git-clone lock file {}", lock_path.display()))?;

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

    Ok((wheel_path, resolved_sha))
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

/// Invoke `uv` with the given args, capturing stdout + stderr so neither
/// leaks to retread's stdout (which is the JSON-RPC channel to pixi).
/// Sets `UV_PYTHON_DOWNLOADS=automatic` so missing pythons are fetched
/// on demand without user intervention.
async fn run_capturing_uv(args: &[&str]) -> Result<()> {
    let mut cmd = Command::new("uv");
    for arg in args {
        cmd.arg(arg);
    }
    let output = cmd
        .env("UV_PYTHON_DOWNLOADS", "automatic")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .context("spawning uv (is it on PATH? expected via retread's runtime dep)")?;
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
/// previous checkout simply parked on the wrong commit. A plain `git
/// checkout` on an already-correct tree is a fast no-op, so this is safe
/// to call unconditionally rather than only on a fresh clone.
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

    // ---------------------------------------------------------------------------
    // Local git fixture: build_wheel_from_git returns (path, resolved_sha)
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

    /// v3.0.2 regression (#8): a clone_dir that survived a prior
    /// corrupted run -- e.g. one left an UNTRACKED file at a path the
    /// target commit also tracks -- must self-heal on the next
    /// `build_wheel_from_git` call instead of failing forever with
    /// "untracked working tree files would be overwritten by checkout."
    #[tokio::test]
    #[ignore = "live: builds a git wheel via uv (needs uv + git on PATH); run with --include-ignored"]
    async fn build_wheel_from_git_self_heals_untracked_file_conflict() {
        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("retread-githeal-{pid}"));
        let repo = base.join("repo");
        std::fs::create_dir_all(&repo).expect("create repo dir");

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
        let rev_parse_head = |dir: &std::path::Path| -> String {
            let out = std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(dir)
                .output()
                .expect("git rev-parse");
            String::from_utf8_lossy(&out.stdout).trim().to_string()
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
name = "retread-test-fixture"
version = "0.1.0"
"#,
        )
        .expect("write pyproject");
        run_git(&["add", "."], &repo);
        run_git(&["commit", "-m", "initial"], &repo);
        let rev1 = rev_parse_head(&repo);

        // Second commit adds a TRACKED "extra.txt". This is the rev every
        // resolve below asks for -- matching the real bug, where every
        // racing [retread-wheels] entry names the exact SAME `rev`, so
        // they all hash to the exact same clone_dir (git_checkout_root
        // keys on url+rev; different revs never collide, which is correct
        // and is NOT what's under test here).
        std::fs::write(repo.join("extra.txt"), "tracked-content").expect("write extra.txt");
        run_git(&["add", "."], &repo);
        run_git(&["commit", "-m", "add extra.txt"], &repo);
        let rev2 = rev_parse_head(&repo);

        let cache_dir = base.join("cache");
        let out_dir = base.join("out");
        std::fs::create_dir_all(&cache_dir).expect("cache dir");
        std::fs::create_dir_all(&out_dir).expect("out dir");
        let repo_url = format!("file://{}", repo.display());

        // First resolve: populates clone_dir at rev2 (the rev every call
        // below will keep asking for).
        build_wheel_from_git(&repo_url, &rev2, ".", &cache_dir, &out_dir, "3.11")
            .await
            .expect("initial build_wheel_from_git");
        let clone_dir = git_checkout_root(&repo_url, &rev2, &cache_dir);

        // Simulate the corruption a pre-v3.0.2 race could leave behind:
        // HEAD parked on an EARLIER commit (rev1, which lacks extra.txt)
        // plus a stray UNTRACKED extra.txt with different content sitting
        // in the working tree. `git checkout rev2` from this state must
        // create extra.txt (rev1 -> rev2 changes it), but an untracked
        // file already occupies that path -- exactly reproducing "The
        // following untracked working tree files would be overwritten by
        // checkout: extra.txt" from issue #8, without needing real
        // concurrency to trigger it.
        run_git(&["checkout", "--force", &rev1], &clone_dir);
        assert_eq!(
            rev_parse_head(&clone_dir),
            rev1,
            "setup: clone_dir must be parked on rev1"
        );
        std::fs::write(clone_dir.join("extra.txt"), "stray-untracked-content")
            .expect("write stray untracked file");

        // Resolving rev2 again now must self-heal (clean + checkout)
        // rather than failing with "untracked working tree files would be
        // overwritten".
        let (_, resolved_sha) =
            build_wheel_from_git(&repo_url, &rev2, ".", &cache_dir, &out_dir, "3.11")
                .await
                .expect("self-healing build_wheel_from_git must succeed");
        assert_eq!(resolved_sha, rev2);
        assert_eq!(
            std::fs::read_to_string(clone_dir.join("extra.txt")).expect("read extra.txt"),
            "tracked-content",
            "the stray untracked file must be replaced by the tracked one, not left in place"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    /// v3.0.3 regression (#8): when a clone_dir is corrupted at the
    /// `.git` level (not just the working tree), `checkout_rev_robust`'s
    /// working-tree-only repair (`git clean -fdx`) can't fix it and
    /// every checkout attempt keeps failing -- exactly what #8 hit next:
    /// "git checkout FETCH_HEAD failed even after cleaning the working
    /// tree." `build_wheel_from_git` must fall back to wiping clone_dir
    /// and re-cloning from scratch rather than erroring out forever.
    /// Simulated here with a stale `.git/index.lock` (a realistic
    /// leftover from a process killed mid-checkout before the flock fix
    /// existed): git refuses EVERY checkout while it's present, and
    /// `git clean` doesn't touch `.git/` at all, so only a full wipe
    /// recovers.
    #[tokio::test]
    #[ignore = "live: builds a git wheel via uv (needs uv + git on PATH); run with --include-ignored"]
    async fn build_wheel_from_git_recovers_from_corrupted_git_dir_by_recloning() {
        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("retread-gitrecover-{pid}"));
        let repo = base.join("repo");
        std::fs::create_dir_all(&repo).expect("create repo dir");

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
name = "retread-test-fixture"
version = "0.1.0"
"#,
        )
        .expect("write pyproject");
        run_git(&["add", "."], &repo);
        run_git(&["commit", "-m", "initial"], &repo);
        let sha_output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&repo)
            .output()
            .expect("git rev-parse");
        let rev = String::from_utf8_lossy(&sha_output.stdout)
            .trim()
            .to_string();

        let cache_dir = base.join("cache");
        let out_dir = base.join("out");
        std::fs::create_dir_all(&cache_dir).expect("cache dir");
        std::fs::create_dir_all(&out_dir).expect("out dir");
        let repo_url = format!("file://{}", repo.display());

        // First resolve: populates clone_dir correctly.
        build_wheel_from_git(&repo_url, &rev, ".", &cache_dir, &out_dir, "3.11")
            .await
            .expect("initial build_wheel_from_git");
        let clone_dir = git_checkout_root(&repo_url, &rev, &cache_dir);

        // Corrupt at the .git level: a stale index.lock blocks EVERY
        // checkout attempt, and `git clean` never touches `.git/`, so
        // checkout_rev_robust's working-tree repair cannot fix this --
        // only wiping clone_dir and recloning can.
        std::fs::write(clone_dir.join(".git").join("index.lock"), "")
            .expect("write stale index.lock");

        let (_, resolved_sha) = build_wheel_from_git(&repo_url, &rev, ".", &cache_dir, &out_dir, "3.11")
            .await
            .expect("must recover by wiping and re-cloning, not error out");
        assert_eq!(resolved_sha, rev);
        assert!(
            !clone_dir.join(".git").join("index.lock").exists(),
            "the fresh clone must not carry over the stale lock file"
        );

        let _ = std::fs::remove_dir_all(&base);
    }
}
