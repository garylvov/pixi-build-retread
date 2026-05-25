//! Build a `.whl` from a local path or git checkout via `pip wheel`.
//!
//! Used by `[retread-wheels]` entries that take `path = "..."` or
//! `git = "..."` instead of the PyPI `version + index` form. The
//! produced wheel goes through the same auto-bundle + METADATA-rewrite
//! pipeline as any PyPI-resolved wheel.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{anyhow, bail, Context, Result};
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
    run_capturing_uv(
        &[
            "build",
            "--wheel",
            &py_arg,
            &out_arg,
            &source.display().to_string(),
        ],
    )
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
        const RETREAD_SUFFIXES: &[&str] = &[
            ".injected.",
            ".autodata.",
            ".relaxed.",
        ];
        if RETREAD_SUFFIXES.iter().any(|s| name.contains(s)) {
            continue;
        }
        let mtime = entry
            .metadata()
            .await
            .with_context(|| format!("stat'ing wheel {}", path.display()))?
            .modified()
            .with_context(|| format!("reading mtime of {}", path.display()))?;
        if best.as_ref().map_or(true, |(t, _)| mtime > *t) {
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
        .and_then(|s| s.last())
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
    run_capturing_uv(
        &[
            "build",
            "--wheel",
            &py_arg,
            &out_arg,
            &sdist_path.display().to_string(),
        ],
    )
    .await?;
    find_built_wheel(out_dir).await
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
///   - <slug>: repo-name slug, truncated to 24 chars
///   - <sha12>: 12 hex chars of sha256(url + "\0" + rev)
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
    let sha12: String = digest
        .iter()
        .take(6)
        .map(|b| format!("{b:02x}"))
        .collect();
    let mut slug = git_slug(url);
    // The slug strips `https___github.com_`; cap whatever's left so
    // big-org/long-name repos don't blow the slug component.
    slug.truncate(24);
    cache_dir
        .join("retread-git-clones")
        .join(slug)
        .join(sha12)
}

/// Clone a git URL at a specific revision into `cache_dir`, then build
/// the wheel for `subdirectory` (relative to the repo root, defaulting
/// to ".").
///
/// Clones are cached by (url, rev) so repeated `conda/outputs` calls
/// for the same workspace don't re-clone. `rev` can be a commit SHA,
/// tag, or branch name.
pub async fn build_wheel_from_git(
    url: &str,
    rev: &str,
    subdirectory: &str,
    cache_dir: &Path,
    out_dir: &Path,
    python_version: &str,
) -> Result<PathBuf> {
    // Delegate to git_checkout_root so the layout stays in sync. (Was
    // duplicated here before v0.13.3 -- update both or the resolver
    // half stops finding the cached clone the cloner half just made.)
    let clone_dir = git_checkout_root(url, rev, cache_dir);

    if !clone_dir.exists() {
        let parent = clone_dir.parent().unwrap();
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| {
                format!(
                    "creating git-clone parent dir {} (for url={url}, rev={rev}, target={})",
                    parent.display(),
                    clone_dir.display(),
                )
            })?;
        tracing::info!(url = %url, rev = %rev, "cloning git source");
        // Clone shallow without checkout. Use a two-step fetch so we can
        // target arbitrary commits (not just branch/tag tips).
        run_silent(
            Command::new("git")
                .arg("clone")
                .arg("--filter=blob:none")
                .arg("--no-checkout")
                .arg(url)
                .arg(&clone_dir),
            "git clone",
        )
        .await?;
        // First-try checkout; if it fails (server doesn't expose arbitrary
        // commits) fetch the rev explicitly and try again.
        let checkout_ok = try_run_silent(
            Command::new("git")
                .args(["checkout", rev])
                .current_dir(&clone_dir),
        )
        .await?;
        if !checkout_ok {
            run_silent(
                Command::new("git")
                    .args(["fetch", "origin", rev])
                    .current_dir(&clone_dir),
                "git fetch",
            )
            .await?;
            run_silent(
                Command::new("git")
                    .args(["checkout", "FETCH_HEAD"])
                    .current_dir(&clone_dir),
                "git checkout FETCH_HEAD",
            )
            .await?;
        }
    } else {
        tracing::debug!(path = %clone_dir.display(), "git source already cached");
    }

    let source_dir = clone_dir.join(subdirectory);
    if !source_dir.exists() {
        bail!(
            "subdirectory `{subdirectory}` not found in clone at {}",
            clone_dir.display()
        );
    }
    build_wheel_from_path(&source_dir, out_dir, python_version).await
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
        bail!(
            "uv {:?} failed (status {}): {snippet}",
            args, output.status,
        );
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
    /// component should be the 24-char slug cap. Previously the (slug
    /// + 40-char raw SHA) flattened into one 60+ char component;
    /// combined with the rattler cache prefix and deep IsaacLab
    /// internals, pathnames tripped ENAMETOOLONG on git checkout.
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
        assert_eq!(last.len(), 12, "sha12 must be exactly 12 chars; got: {last}");
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
}
