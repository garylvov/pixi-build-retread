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

use std::collections::HashSet;
use std::path::Path;

use anyhow::Context as _;

use crate::config::RetreadConfig;
use crate::emit_pypi::{
    EmitWheel, build_meta_wheel, collect_prerelease_pins, insert_build_tag, override_line_map,
    plan, standard_wheel_filename,
};
use crate::lock::{CondaDep, LockWheel, Origin, RetreadLock, SCHEMA};

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
pub fn courier_input_specs(config: &RetreadConfig, bundle_name: &str) -> Vec<String> {
    let mut specs: Vec<String> = config
        .retread_wheels
        .iter()
        .filter(|(key, entry)| {
            let group = entry.bundle.as_deref().or(config.default_bundle.as_deref());
            match group {
                Some(g) => g == bundle_name,
                None => key.as_str() == bundle_name,
            }
        })
        .map(|(key, entry)| {
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
                        .and_then(|f| config.git_sources.get(f))
                        .map(|s| format!("@git:{}", s.rev))
                })
                .or_else(|| entry.url.as_ref().map(|u| format!("@url:{u}")))
                .unwrap_or_default();
            format!("{key}{extras}{ver}")
        })
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
/// conda-deps, the auto-bundle toggle, the build-number, AND the conda channel
/// list. Each of these changes the emitted conda specs or the conda/PyPI
/// routing, so omitting any would let a manifest/workspace edit leave the hash
/// unchanged and replay a stale, POISONED lock. (genesis's `retread-name-map`
/// is the canonical config regression case; a workspace channel addition is
/// the canonical channel case -- a newly-added channel can make a previously
/// auto-bundled wheel conda-capable, flipping its lock classification.)
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
    // Suppress unused_variables warning on source_dir (used by callers for
    // context but not needed internally; staging_dir is the write target).
    let _ = source_dir;

    tokio::fs::create_dir_all(staging_dir)
        .await
        .with_context(|| format!("creating staging dir {}", staging_dir.display()))?;

    // Step 1: run plan() to get ship set + override table.
    let emit_plan = plan(emit_wheels, conda_capable);

    // Clone overrides + conda_capable so we can move them into spawn_blocking.
    // The mapper itself cannot cross the `'static` boundary (it holds refs),
    // so we re-derive it inside each blocking closure from owned copies.
    let overrides_owned = emit_plan.overrides.clone();
    let conda_cap_owned: HashSet<String> = conda_capable.clone();

    // Step 2: Build [retread-wheels] entries for this bundle (same filter as
    // emit's `entries` construction).
    let entries: Vec<(String, crate::config::WheelEntry, Option<String>)> = config
        .retread_wheels
        .iter()
        .filter(|(key, entry)| {
            let group = entry.bundle.as_deref().or(config.default_bundle.as_deref());
            match group {
                Some(g) => g == bundle_name,
                None => key.as_str() == bundle_name,
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
    let mapper_for_remote = override_line_map(&overrides_owned, &conda_cap_owned);

    // Step 3: Classify and stage each wheel.
    let mut lock_wheels: Vec<LockWheel> = Vec::new();
    let mut source_urls: Vec<String> = Vec::new();

    for w in emit_wheels {
        let std_name = standard_wheel_filename(&w.wheel_filename);

        if w.must_ship() {
            // Built-by-retread wheel (carries .injected infix): copy to
            // staging_dir under the standard filename and record Origin::Built.
            let src = w.local_path.as_ref().ok_or_else(|| {
                anyhow::anyhow!("must_ship wheel has no local_path: {}", w.pypi_name)
            })?;
            let dst = staging_dir.join(&std_name);
            tokio::fs::copy(src, &dst).await.with_context(|| {
                format!("staging built wheel {} -> {}", src.display(), dst.display())
            })?;
            source_urls.push(file_url(&dst)?);
            lock_wheels.push(LockWheel {
                name: w.pypi_name.clone(),
                version: w.version.clone(),
                origin: Origin::Built,
                filename: std_name,
                url: None,
                sha256: None,
            });
        } else {
            // Index wheel. Decide whether to ship a relax-rewritten shadow
            // (AUDIT B2: relax-changed index wheels must ship as shadows, not
            // stay remote, or strict pins re-emerge at install time). Produce
            // `(changed, shadow_src)`: when `changed`, `shadow_src` holds the
            // local bytes to rewrite.
            let (changed, shadow_src): (bool, Option<std::path::PathBuf>) = if let Some(src) =
                w.local_path.as_ref()
            {
                // We have the bytes: run the rewrite into a temp file to
                // determine whether the mapper changes any line.
                let tmp_name = format!(".tmp-courier-{std_name}");
                let tmp = staging_dir.join(&tmp_name);
                let overrides_c = overrides_owned.clone();
                let conda_cap_c = conda_cap_owned.clone();
                let (_sha, did_change) = tokio::task::spawn_blocking({
                    let src = src.clone();
                    let tmp = tmp.clone();
                    move || {
                        let m = override_line_map(&overrides_c, &conda_cap_c);
                        crate::wheel_rewrite::rewrite_wheel_with(&src, &tmp, &m)
                    }
                })
                .await
                .with_context(|| format!("spawn_blocking rewrite-check of {}", w.pypi_name))??;
                // Remove the probe tmp -- we re-stage into the real dst below.
                let _ = tokio::fs::remove_file(&tmp).await;
                (did_change, did_change.then(|| src.clone()))
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
                    .any(|l| mapper_for_remote(l).is_some());
                if any_change && !conda_cap_owned.contains(&w.pypi_name) {
                    let url = w.remote_url.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "courier: remote-only relax-changed wheel {} has no remote_url \
                                 to download for shadow rewrite",
                            w.pypi_name
                        )
                    })?;
                    let dl = staging_dir.join(format!(".dl-courier-{std_name}"));
                    let bytes = reqwest::get(url.clone())
                        .await
                        .and_then(|r| r.error_for_status())
                        .with_context(|| {
                            format!("downloading {} ({url}) for shadow rewrite", w.pypi_name)
                        })?
                        .bytes()
                        .await
                        .with_context(|| format!("reading bytes of {}", w.pypi_name))?;
                    tokio::fs::write(&dl, &bytes)
                        .await
                        .with_context(|| format!("writing downloaded {}", dl.display()))?;
                    tracing::info!(
                        wheel = %w.pypi_name,
                        "courier: force-downloaded remote-only relax-changed wheel to ship \
                         a rewritten shadow (B-5)",
                    );
                    (true, Some(dl))
                } else {
                    (false, None)
                }
            };

            if changed {
                // Relax changed this index wheel's METADATA: ship it as a
                // build-tagged shadow wheel so uv's find-links prefers it over
                // the registry original (AUDIT B2 fix).
                let src = shadow_src.expect("changed=true implies shadow_src is Some");
                let shadow_name = insert_build_tag(&std_name, "999retread")?;
                let dst = staging_dir.join(&shadow_name);
                let dst_blocking = dst.clone();
                let src_blocking = src.clone();
                let overrides_c2 = overrides_owned.clone();
                let conda_cap_c2 = conda_cap_owned.clone();
                tokio::task::spawn_blocking(move || {
                    let m = override_line_map(&overrides_c2, &conda_cap_c2);
                    crate::wheel_rewrite::rewrite_wheel_with(&src_blocking, &dst_blocking, &m)
                })
                .await
                .with_context(|| format!("spawn_blocking shadow-rewrite of {}", w.pypi_name))??;
                source_urls.push(file_url(&dst)?);
                lock_wheels.push(LockWheel {
                    name: w.pypi_name.clone(),
                    version: w.version.clone(),
                    origin: Origin::Built,
                    filename: shadow_name,
                    url: None,
                    sha256: None,
                });
            } else {
                // Unchanged index wheel: record with upstream url.
                // sha256 is not carried by EmitWheel; the installer verifies
                // at fetch time from the index's sidecar hash.
                let upstream_url = w.remote_url.as_ref().map(|u| u.to_string());
                lock_wheels.push(LockWheel {
                    name: w.pypi_name.clone(),
                    version: w.version.clone(),
                    origin: Origin::Index,
                    filename: std_name,
                    url: upstream_url,
                    sha256: None,
                });
            }
        }
    }

    // Step 4: Build and stage the <bundle>-pypi meta-wheel.
    let (meta_name, meta_bytes) = build_meta_wheel(bundle_name, version, &entries);
    let meta_dst = staging_dir.join(&meta_name);
    tokio::fs::write(&meta_dst, &meta_bytes)
        .await
        .with_context(|| format!("writing meta-wheel {}", meta_dst.display()))?;
    source_urls.push(file_url(&meta_dst)?);

    // Step 4b: ship the static installer binary INSIDE the package (the
    // currently-running backend == the static musl `pixi-build-retread`).
    // The recipe copies it to `$PREFIX/bin/retread`; the post-link runs it.
    // This avoids run-depping on the heavy backend conda package (which the
    // consumer's solve check can't even see on a file:///non-default channel).
    let self_exe = std::env::current_exe().context("locating retread backend binary")?;
    let installer_dst = staging_dir.join("retread-installer");
    tokio::fs::copy(&self_exe, &installer_dst)
        .await
        .with_context(|| {
            format!(
                "staging installer binary {} -> {}",
                self_exe.display(),
                installer_dst.display()
            )
        })?;
    source_urls.push(file_url(&installer_dst)?);

    // Step 5: Collect prerelease pins and compute inputs_hash.
    let prerelease =
        collect_prerelease_pins(&emit_plan.overrides, emit_wheels, &emit_plan.ship, &entries);

    // inputs_hash over PURE manifest inputs (NOT resolved versions), via the
    // canonical helper shared with the cold-solve replayer in conda_outputs --
    // they MUST compute it identically or replay never fires.
    let inputs_hash = RetreadLock::compute_inputs_hash(
        &courier_input_specs(config, bundle_name),
        index_urls,
        &format!("{:?}", config.relax),
        python,
        env!("CARGO_PKG_VERSION"),
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

    // Step 6: Assemble the RetreadLock.
    let lock = RetreadLock {
        schema: SCHEMA,
        retread_version: env!("CARGO_PKG_VERSION").to_string(),
        bundle: bundle_name.to_string(),
        version: version.to_string(),
        python: python.to_string(),
        inputs_hash,
        root_requirements: vec![format!("{bundle_name}-pypi=={version}")],
        wheels: lock_wheels,
        conda_run_deps: parse_conda_deps(run_deps),
        index_urls: index_urls.to_vec(),
        prerelease,
    };

    // Step 7: Write the lock JSON into staging_dir. Write-then-rename so a
    // crash mid-write can never leave a torn lock (B-3): a partial file would
    // either fail to parse (fail-safe, replay falls through) or, worse, parse
    // into wrong data. The rename is atomic on the same filesystem.
    let lock_json = lock.to_pretty_json()?;
    let lock_filename = RetreadLock::file_name(bundle_name);
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
        "courier: staged artifacts",
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
    use std::io::Write as _;

    /// Build a minimal valid wheel zip in memory for the given dist/version/requires.
    fn make_wheel_bytes(dist: &str, version: &str, requires: &[&str]) -> Vec<u8> {
        let normalized = dist.replace('-', "_");
        let di = format!("{normalized}-{version}.dist-info");
        let mut metadata = format!("Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n");
        for req in requires {
            metadata.push_str(&format!("Requires-Dist: {req}\n"));
        }
        let metadata_bytes = metadata.into_bytes();
        let wheel_file = b"Wheel-Version: 1.0\nTag: py3-none-any\n".to_vec();
        // Minimal RECORD with empty hashes (good enough for the rewrite test).
        let record = format!("{di}/METADATA,,\n{di}/WHEEL,,\n{di}/RECORD,,\n").into_bytes();

        let mut buf = Vec::new();
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, body) in [
            (format!("{di}/METADATA"), metadata_bytes.as_slice()),
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
            .unwrap_or_else(|| format!("{name}-{version}-py3-none-any.whl"));
        EmitWheel {
            pypi_name: name.to_string(),
            version: version.to_string(),
            requires_dist: requires.iter().map(|s| s.to_string()).collect(),
            wheel_filename,
            local_path: local_path.map(|p| p.to_path_buf()),
            remote_url: remote_url.and_then(|u| u.parse().ok()),
        }
    }

    fn minimal_config(bundle_name: &str) -> RetreadConfig {
        let json = serde_json::json!({
            "retread-wheels": {
                bundle_name: { "version": "==1.0.0" }
            }
        });
        serde_json::from_value(json).unwrap()
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
        let dep_a_whl_name = format!("{target_name}-{target_version}-py3-none-any.whl");
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
    }
}
