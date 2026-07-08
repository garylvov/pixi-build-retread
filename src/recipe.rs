//! Generate a rattler-build `recipe.yaml` for a bundle of repacked wheels.
//!
//! The bundle pattern: one conda package whose `source:` list contains every
//! wheel in the bundle (the user's named entry plus extras-derived
//! sub-wheels). All wheels are pip-installed into the same prefix at build
//! time. Mirrors comment 24 of prefix-dev/pixi#5230.

use std::collections::HashSet;

use serde::Serialize;

use crate::config::RetreadConfig;
use crate::relax::{default_marker_env, emit_python_version, translate};
use crate::wheel::WheelMetadata;

#[derive(Debug, Serialize)]
pub struct Recipe {
    pub schema_version: u32,
    pub package: Package,
    pub source: Vec<Source>,
    pub build: Build,
    pub requirements: Requirements,
    pub about: About,
}

#[derive(Debug, Serialize)]
pub struct Package {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Serialize)]
pub struct Source {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Build {
    pub number: u64,
    /// Explicit build string. When set, rattler-build uses this verbatim
    /// instead of synthesizing one from the variant + build number. Used by
    /// the courier path to embed the content-addressed `inputs_hash` prefix
    /// so pixi cache-hits are invalidated whenever content changes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub string: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noarch: Option<String>,
    pub script: String,
    /// Per rattler-build's recipe schema, `binary_relocation` lives under
    /// `build.dynamic_linking`, NOT at the top level of `build`. See the
    /// dynamic_linking section in rattler-build docs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_linking: Option<DynamicLinking>,
}

/// rattler-build's `build.dynamic_linking` group. Only emit fields we
/// actually set, so we don't accidentally override rattler-build's
/// defaults for anything else.
#[derive(Debug, Serialize)]
pub struct DynamicLinking {
    /// Skip rattler-build's patchelf/relink pass on bundled `.so` files.
    /// Vendor wheels (NVIDIA Omniverse, manylinux) ship with pre-baked
    /// rpaths that point into their own extscache trees. rattler-build's
    /// default behavior rewrites those to be prefix-relative, which
    /// (a) overflows the original DT_RPATH slot for many of NVIDIA's libs
    /// (`× error new value is longer than old value`) and (b) trips a
    /// goblin ELF parser panic on libs whose string tables contain
    /// non-UTF8 bytes (Failed to parse the ELF file: invalid utf8). Both
    /// fire during the "Packaging new files" phase. Disabling the pass
    /// keeps the wheels' original rpaths -- which is what they were built
    /// to use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_relocation: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct Requirements {
    pub host: Vec<String>,
    pub run: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct About {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

fn shell_ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// Input to [`build_bundle_recipe`]: one wheel that should appear in the
/// recipe's `source:` list. The metadata feeds run-deps and platform
/// detection.
pub struct BundleSource<'a> {
    /// PEP 503 normalized name (e.g. "isaacsim-kernel"). Used to filter
    /// out vendored deps from the run list.
    pub pypi_name: &'a str,
    pub url: &'a url::Url,
    pub metadata: &'a WheelMetadata,
}

/// Build a recipe for a multi-wheel bundle. The conda package name comes
/// from the bundle (not any single wheel's METADATA); the version comes from
/// the primary wheel (the first source).
///
/// All wheels in `sources` are pip-installed into the same prefix at build
/// time with `--no-deps`. Deps that name any of the bundled wheels are
/// dropped from the run-deps because they're vendored.
pub fn build_bundle_recipe(
    conda_name: &str,
    sources: &[BundleSource<'_>],
    config: &RetreadConfig,
    workspace_python_version: &str,
    run_override: Option<&[String]>,
    payload: bool,
) -> anyhow::Result<Recipe> {
    let primary = sources
        .first()
        .ok_or_else(|| anyhow::anyhow!("bundle must have at least one source"))?;
    // Prefer the primary wheel's tag (it pins the cpXY ABI), but fall back
    // to the workspace python whenever the wheel only carries a bare-major
    // tag (`py3-none-any`). Shared with `handler::produce_output` via
    // `emit_python_version` so the recipe and the conda/outputs metadata
    // always agree on the same dotted X.Y.
    let python_version = emit_python_version(&primary.metadata.filename, workspace_python_version);
    let python_pin = format!("python {python_version}.*");

    // Run-deps: PREFER the exact specs pixi solved/locked with, forwarded by
    // pixi in `CondaBuildV1Params.run_dependencies` (-> `run_override`). This
    // guarantees the BUILT package's run-deps MATCH what the solve produced --
    // including cascade widenings the metadata applied (e.g. `pytorch >=1`).
    // Re-deriving from each wheel's requires_dist here (the fallback below)
    // diverges from the solve and can comma-join the raw, un-widened
    // transitive override into a malformed spec like
    // `pytorch >=1.4,2.10.0,>=2.10.0,<2.11.0a0`, which rattler-build rejects
    // ("missing range specifier for '2.10.0'"). pixi's specs are already
    // parsed MatchSpecs, so they round-trip cleanly.
    let run: Vec<String> = if let Some(over) = run_override {
        let mut r: Vec<String> = over.to_vec();
        // The solved run-deps normally include `python`; if a host (older
        // pixi) ever omits it, keep the package importable.
        if !r.iter().any(|s| s == "python" || s.starts_with("python ")) {
            r.insert(0, python_pin.clone());
        }
        r
    } else {
        // Fallback for older pixi that doesn't forward run_dependencies in
        // the build params: derive from each wheel's requires_dist.
        let env = default_marker_env(&python_version)?;
        let vendored: HashSet<String> = sources.iter().map(|s| s.pypi_name.to_string()).collect();
        let mut run = vec![python_pin.clone()];
        let mut seen: HashSet<String> = HashSet::from(["python".to_string()]);
        for source in sources {
            for raw in &source.metadata.requires_dist {
                match translate(raw, &env, &config.name_map, &config.overrides, config.relax) {
                    Ok(Some(dep)) => {
                        let dep_name = dep.name.clone();
                        if vendored.contains(&dep_name) {
                            continue;
                        }
                        if seen.insert(dep_name) {
                            run.push(dep.to_string());
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        tracing::warn!(req = %raw, error = %e, "could not translate requirement; dropping");
                    }
                }
            }
        }
        run
    };

    let host = vec![python_pin, "pip".to_string()];

    let any_platform_specific = sources.iter().any(|s| !s.metadata.is_pure_python);
    let noarch = if any_platform_specific {
        None
    } else {
        Some("python".to_string())
    };

    // v1.7.0 `retread-blueprint = "only"`: payload-skip mode. The
    // recipe keeps its REAL shape -- version from the primary wheel,
    // noarch/subdir derived from the actual wheel set, the solved
    // run-deps -- so pixi's lock and identity checks (name/version/
    // build/subdir) hold and consuming envs keep their conda
    // transitives. Only the wheel payload is omitted: no sources, a
    // no-op script, and therefore seconds of packaging instead of
    // minutes of zstd. (An earlier empty-stub shape flipped noarch and
    // zeroed run-deps; pixi rejected the artifact and the lock lied.)
    let recipe_sources = if payload {
        sources
            .iter()
            .map(|s| Source {
                url: s.url.to_string(),
                sha256: Some(s.metadata.sha256.clone()),
            })
            .collect()
    } else {
        Vec::new()
    };

    Ok(Recipe {
        schema_version: 1,
        package: Package {
            name: conda_name.to_string(),
            version: primary.metadata.version.clone(),
        },
        source: recipe_sources,
        build: Build {
            number: config.build_number,
            // Non-courier path: no content-addressed string; rattler-build
            // synthesizes the standard `py{XY}_{build_number}` string.
            string: None,
            noarch,
            // Vendor wheels (Omniverse, manylinux) ship pre-baked rpaths;
            // rattler-build's default relocation pass either overflows the
            // original DT_RPATH slot or chokes on non-UTF8 in some .so
            // string tables. Skip the patchelf step. Only meaningful for
            // platform-specific bundles -- noarch has no native libs.
            dynamic_linking: if any_platform_specific {
                Some(DynamicLinking {
                    binary_relocation: Some(false),
                })
            } else {
                None
            },
            // `--no-deps` is essential: conda solves deps from the run: list,
            // not from pip re-resolving Requires-Dist at install time.
            script: if payload {
                "${{ PYTHON }} -m pip install *.whl -vv --no-deps --no-build-isolation".to_string()
            } else {
                // Payload-skip: the package carries metadata only; the
                // content is consumed through the blueprint find-links.
                "echo retread-blueprint=only: payload skipped".to_string()
            },
        },
        requirements: Requirements { host, run },
        about: About {
            license: None,
            summary: None,
        },
    })
}

/// v2.0.0 courier: build the recipe for a metadata-only "courier" conda
/// package. Unlike [`build_bundle_recipe`], the package carries NO installed
/// wheel payload. Instead it ships the bundle's built/shadow wheels + the
/// committed lock as data under `$PREFIX/share/retread/`, declares the
/// solved conda run-deps (so shared transitives stay conda) plus `uv` and
/// `pixi-build-retread` (the installer binary), and writes a conda
/// **post-link** script that runs `retread install` at env link time to
/// install exact wheel files with `uv --no-deps --offline`. Missing
/// unchanged index wheels are direct-fetched from the lock's URL+hash; the
/// huge index wheels never enter the conda package, so packaging is seconds
/// and nothing is committed to git.
///
/// `source_urls` are file:// URLs to the staged wheels + the lock json
/// (rattler-build copies them into `$SRC_DIR`); the build script stages
/// them into the install prefix. `run_deps` is the solved conda run-dep
/// list (uv + pixi-build-retread are appended here).
pub fn build_courier_recipe(
    conda_name: &str,
    version: &str,
    python_version: &str,
    run_deps: &[String],
    source_urls: &[String],
    // Content-addressed build string to embed in the recipe. When `Some`,
    // rattler-build records it verbatim so the on-disk artifact name matches
    // what `conda/outputs` advertised to pixi. When `None` (direct tests
    // without a pixi context), rattler-build synthesizes the string itself.
    expected_build: Option<&str>,
) -> Recipe {
    let python_pin = format!("python {python_version}.*");
    let lock_filename = crate::lock::RetreadLock::file_name(conda_name);

    let mut run: Vec<String> = run_deps.to_vec();
    if !run
        .iter()
        .any(|s| s == "python" || s.starts_with("python "))
    {
        run.insert(0, python_pin.clone());
    }
    // Only `uv` is a conda run-dep (always on conda-forge -> the consumer's
    // pre-emission solve check + lock find it). We do NOT run-dep on
    // `pixi-build-retread`: that would drag the heavy backend (rattler-build,
    // ...) into the consumer env AND the solve check can't see it on a
    // file:// / non-default channel. Instead the static installer binary
    // SHIPS inside this package (staged as `retread-installer`, copied to
    // `$PREFIX/bin/retread` by the build script below).
    if !run.iter().any(|s| s == "uv" || s.starts_with("uv ")) {
        run.push("uv".to_string());
    }

    // Build script: stage wheels + lock under $PREFIX/share/retread, install
    // the shipped static `retread` binary into $PREFIX/bin, then emit the
    // conda post-link script (literal $PREFIX -- expanded at LINK time, not
    // build time -- via the quoted heredoc). The post-link runs the installer
    // and must fail the conda link if the PyPI payload cannot be installed;
    // otherwise pixi can report success for an incomplete environment.
    let post_link = format!("$PREFIX/bin/.{conda_name}-post-link.sh");
    // Loud-failure guard: an activate.d script runs on every activation
    // (regardless of the post-link toggle) and warns when the marker is absent
    // or no longer matches the actual installed wheel state.
    let activate_guard = format!("$PREFIX/etc/conda/activate.d/zzz-retread-{conda_name}.sh");
    let deactivate_guard = format!("$PREFIX/etc/conda/deactivate.d/zzz-retread-{conda_name}.sh");
    let var_pack = shell_ident(conda_name);
    let script = format!(
        "set -euo pipefail\n\
         SHARE=\"$PREFIX/share/retread\"\n\
         WHEELS=\"$SHARE/{conda_name}/wheels\"\n\
         mkdir -p \"$WHEELS\" \"$PREFIX/bin\" \"$PREFIX/etc/conda/activate.d\" \"$PREFIX/etc/conda/deactivate.d\"\n\
         cp \"$SRC_DIR\"/*.whl \"$WHEELS\"/ 2>/dev/null || true\n\
         cp \"$SRC_DIR\"/{lock_filename} \"$SHARE\"/\n\
         cp \"$SRC_DIR\"/retread-installer \"$PREFIX/bin/retread\"\n\
         chmod +x \"$PREFIX/bin/retread\"\n\
         cat > \"{post_link}\" <<'POSTLINK'\n\
         #!/bin/bash\n\
         set -euo pipefail\n\
         \"$PREFIX/bin/retread\" install --lock \"$PREFIX/share/retread/{lock_filename}\" --prefix \"$PREFIX\"\n\
         POSTLINK\n\
         chmod +x \"{post_link}\"\n\
         cat > \"{activate_guard}\" <<'ACTIVATE'\n\
         #!/bin/bash\n\
         # SELF-HEAL guard, sourced on every activation. The bundle's PyPI wheels\n\
         # are installed by the post-link as a side effect that conda/pixi does\n\
         # NOT track, so any loss of that payload -- env moved, node-local /tmp\n\
         # env wiped while conda-meta survived, or pixi treating the package as\n\
         # already-satisfied and skipping the post-link on relink -- is invisible\n\
         # to pixi and never repaired. On activation, if the payload no longer\n\
         # verifies, re-run the installer to restore it into the CURRENT prefix\n\
         # from the conda-tracked lock + shipped wheels + shipped retread binary\n\
         # (all of which survive the payload loss). Cheap no-op when healthy\n\
         # (verify is marker + stat checks). This file is SOURCED into the\n\
         # user's interactive shell, so by default it must never enable errexit,\n\
         # exit, or return nonzero. RETREAD_GUARD_STRICT=1 opts into return 1.\n\
         case \":${{LD_LIBRARY_PATH:-}}:\" in\n\
         *\":$CONDA_PREFIX/lib:\"*) ;;\n\
         *) export _RETREAD_SAVED_LDLP_{var_pack}=\"${{LD_LIBRARY_PATH-__unset__}}\"\n\
            export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\" ;;\n\
         esac\n\
         RETREAD_LOCK=\"$CONDA_PREFIX/share/retread/{lock_filename}\"\n\
         RETREAD_BROKEN_FILE=\"$CONDA_PREFIX/share/retread/{conda_name}.broken\"\n\
         RETREAD_REPAIR_LOG=\"$CONDA_PREFIX/share/retread/{conda_name}.repair.log\"\n\
         RETREAD_SKIP_HEAL=0\n\
         if [ -f \"$RETREAD_BROKEN_FILE\" ]; then\n\
         now=$(date +%s 2>/dev/null || echo 0)\n\
         mtime=$(date -r \"$RETREAD_BROKEN_FILE\" +%s 2>/dev/null || echo 0)\n\
         age=$((now - mtime))\n\
         if [ \"$age\" -lt 300 ]; then RETREAD_SKIP_HEAL=1; fi\n\
         fi\n\
         if [ \"$RETREAD_SKIP_HEAL\" = \"1\" ]; then\n\
         echo \"retread: '{conda_name}' is marked broken; skipping auto-repair during 300s backoff. Retry manually:\" >&2\n\
         echo \"  \\\"$CONDA_PREFIX/bin/retread\\\" install --lock \\\"$RETREAD_LOCK\\\" --prefix \\\"$CONDA_PREFIX\\\"\" >&2\n\
         export RETREAD_BROKEN_{var_pack}=1\n\
         elif ! \"$CONDA_PREFIX/bin/retread\" verify --lock \"$RETREAD_LOCK\" --prefix \"$CONDA_PREFIX\" >/dev/null 2>&1; then\n\
         echo \"retread: '{conda_name}' PyPI wheels missing from this env; repairing...\" >&2\n\
         if \"$CONDA_PREFIX/bin/retread\" install --lock \"$RETREAD_LOCK\" --prefix \"$CONDA_PREFIX\" >\"$RETREAD_REPAIR_LOG\" 2>&1; then\n\
         rm -f \"$RETREAD_BROKEN_FILE\" \"$RETREAD_REPAIR_LOG\"\n\
         unset RETREAD_BROKEN_{var_pack}\n\
         else\n\
         tail -n 80 \"$RETREAD_REPAIR_LOG\" >&2 2>/dev/null || true\n\
         mkdir -p \"$(dirname \"$RETREAD_BROKEN_FILE\")\"\n\
         {{ date -u 2>/dev/null || date; tail -n 80 \"$RETREAD_REPAIR_LOG\" 2>/dev/null || true; }} > \"$RETREAD_BROKEN_FILE\"\n\
         export RETREAD_BROKEN_{var_pack}=1\n\
         echo \"retread: '{conda_name}' auto-repair FAILED; env is incomplete. Retry manually:\" >&2\n\
         echo \"  \\\"$CONDA_PREFIX/bin/retread\\\" install --lock \\\"$RETREAD_LOCK\\\" --prefix \\\"$CONDA_PREFIX\\\"\" >&2\n\
         if [ \"${{RETREAD_GUARD_STRICT:-0}}\" = \"1\" ]; then return 1 2>/dev/null || true; fi\n\
         fi\n\
         fi\n\
         ACTIVATE\n\
         chmod +x \"{activate_guard}\"\n\
         cat > \"{deactivate_guard}\" <<'DEACTIVATE'\n\
         #!/bin/bash\n\
         if [ \"${{_RETREAD_SAVED_LDLP_{var_pack}-}}\" = \"__unset__\" ]; then unset LD_LIBRARY_PATH\n\
         elif [ -n \"${{_RETREAD_SAVED_LDLP_{var_pack}+x}}\" ]; then\n\
         export LD_LIBRARY_PATH=\"$_RETREAD_SAVED_LDLP_{var_pack}\"\n\
         fi\n\
         unset _RETREAD_SAVED_LDLP_{var_pack}\n\
         DEACTIVATE\n\
         chmod +x \"{deactivate_guard}\"\n"
    );

    let source = source_urls
        .iter()
        .map(|u| Source {
            url: u.clone(),
            sha256: None,
        })
        .collect();

    Recipe {
        schema_version: 1,
        package: Package {
            name: conda_name.to_string(),
            version: version.to_string(),
        },
        source,
        build: Build {
            number: 0,
            // Content-addressed string: embed so the on-disk artifact name
            // matches what conda/outputs advertised to pixi (prevents stale
            // cache hits when content changes but metadata stays the same).
            string: expected_build.map(str::to_string),
            // Platform + python specific: the staged wheels are cpXY/manylinux
            // and the lock is python-specific, so the package must not be
            // noarch (build string carries the python variant via the run pin).
            noarch: None,
            // Wheels ship as .whl zips (no extracted .so), so rattler-build's
            // relocation pass has nothing to rewrite -- leave defaults.
            dynamic_linking: None,
            script,
        },
        requirements: Requirements {
            host: vec![python_pin],
            run,
        },
        about: About {
            license: None,
            summary: None,
        },
    }
}

pub fn to_yaml(recipe: &Recipe) -> anyhow::Result<String> {
    Ok(serde_yaml::to_string(recipe)?)
}

#[cfg(test)]
mod courier_tests {
    use super::*;

    #[test]
    fn courier_recipe_shape() {
        let r = build_courier_recipe(
            "isaac-pack",
            "5.1.0",
            "3.11",
            &["torchaudio >=2.7,<3".to_string(), "numpy <2".to_string()],
            &[
                "file:///x/isaaclab-0.51.1-py3-none-any.whl".to_string(),
                "file:///x/retread-isaac-pack.lock.json".to_string(),
            ],
            None,
        );
        // run-deps: solved deps + the installer essentials, no dup python.
        assert!(
            r.requirements
                .run
                .iter()
                .any(|s| s == "torchaudio >=2.7,<3")
        );
        assert!(r.requirements.run.iter().any(|s| s == "uv"));
        // The installer binary SHIPS in the package (not a run-dep), so the
        // heavy backend never pollutes the consumer env.
        assert!(
            !r.requirements.run.iter().any(|s| s == "pixi-build-retread"),
            "courier must NOT run-dep on the backend"
        );
        assert!(r.requirements.run.iter().any(|s| s.starts_with("python ")));
        // no payload pip-install; ships wheels + lock + the static binary as
        // data + a post-link that runs the shipped `retread` installer.
        assert!(r.build.script.contains("share/retread"));
        assert!(r.build.script.contains(".isaac-pack-post-link.sh"));
        assert!(
            r.build.script.contains("cp \"$SRC_DIR\"/retread-installer"),
            "must stage the shipped installer binary"
        );
        assert!(
            r.build
                .script
                .contains("\"$PREFIX/bin/retread\" install --lock"),
            "post-link must run the shipped installer"
        );
        assert!(
            r.build.script.contains("set -euo pipefail"),
            "post-link must fail closed when the installer fails"
        );
        assert!(
            // `|| echo "` (with a message) is the failure-downgrade pattern;
            // the guard's `date ... || echo 0` fallbacks are benign.
            !r.build.script.contains("post-link install failed")
                && !r.build.script.contains("|| echo \""),
            "post-link must not downgrade installer failure to success"
        );
        assert!(
            r.build.script.contains("retread-isaac-pack.lock.json"),
            "post-link must reference the bundle lock"
        );
        // sources are the staged wheels + lock (no sha pinning needed).
        assert_eq!(r.source.len(), 2);
        assert!(r.build.noarch.is_none());
        // it must NOT pip-install a payload like the conda-artifact recipe.
        assert!(!r.build.script.contains("pip install *.whl"));
        // The quoted heredoc must be valid: body + terminator at column 0
        // (Rust's `\` string-continuation eats the source indentation). A
        // leading-space terminator would never close the heredoc and would
        // swallow the rest of the build script.
        assert!(
            r.build.script.contains("\n#!/bin/bash\n"),
            "post-link heredoc body must start at column 0"
        );
        assert!(
            r.build.script.contains("\nPOSTLINK\n"),
            "heredoc terminator must be at column 0 to close the heredoc"
        );
        // Self-heal guard: a conda activate.d script that verifies the payload
        // on every activation and repairs it when missing.
        assert!(
            r.build
                .script
                .contains("etc/conda/activate.d/zzz-retread-isaac-pack.sh"),
            "must ship an activate.d guard"
        );
        assert!(
            r.build
                .script
                .contains("etc/conda/deactivate.d/zzz-retread-isaac-pack.sh"),
            "must ship a matching deactivate.d hook"
        );
        assert!(
            r.build
                .script
                .contains("LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib"),
            "activate.d must prepend $CONDA_PREFIX/lib to LD_LIBRARY_PATH"
        );
        assert!(
            r.build.script.contains("_RETREAD_SAVED_LDLP_isaac_pack"),
            "activate/deactivate hooks must use a sanitized pack-specific saved variable"
        );
        assert!(
            r.build
                .script
                .contains("\"$CONDA_PREFIX/bin/retread\" verify --lock"),
            "guard must verify marker plus installed wheel metadata"
        );
        assert!(
            r.build.script.contains("retread-isaac-pack.lock.json"),
            "guard must verify against the bundle lock"
        );
        // Self-heal: on a failed verify the guard must RUN the installer to
        // repair the payload (not merely warn), so an env whose non-conda-
        // tracked wheels were lost (moved / node-local /tmp wiped / post-link
        // skipped on relink) is restored on next activation.
        assert!(
            r.build
                .script
                .contains("\"$CONDA_PREFIX/bin/retread\" install --lock"),
            "guard must self-heal by running the installer when verify fails"
        );
        assert!(
            r.build.script.contains("repairing..."),
            "guard should announce the repair"
        );
        // Sourced into the user's shell -> must not carry set -e / exit that
        // would abort activation on a repair failure.
        let activate_body = r
            .build
            .script
            .split("<<'ACTIVATE'\n")
            .nth(1)
            .and_then(|s| s.split("\nACTIVATE\n").next())
            .expect("activate.d heredoc body");
        assert!(
            !activate_body.contains("set -e") && !activate_body.contains("\nexit "),
            "activate.d guard is sourced; it must not set -e or exit"
        );
        assert!(
            activate_body.contains("RETREAD_GUARD_STRICT"),
            "return 1 must be strict-mode-only"
        );
        assert!(
            r.build.script.contains("\nACTIVATE\n"),
            "activate.d heredoc terminator must be at column 0"
        );
        assert!(
            r.build.script.contains("\nDEACTIVATE\n"),
            "deactivate.d heredoc terminator must be at column 0"
        );
    }

    // End-to-end: render the REAL courier build script, run it with bash to
    // materialize a prefix, then drive the generated activate.d guard against a
    // STUB `retread` binary to prove the self-heal control flow: when the
    // payload is missing (verify fails) the guard actually RUNS the installer
    // and restores it; when the payload is present (verify passes) the guard is
    // a no-op and never invokes install. This reproduces "marker absent, lock
    // present, prefix has no payload" (issue #9 / detached-/tmp Scenario B).
    #[test]
    fn courier_activate_guard_self_heals_missing_payload_and_noops_when_present() {
        use std::process::Command;

        let bash = match which("bash") {
            Some(b) => b,
            None => {
                eprintln!("skipping: bash not on PATH");
                return;
            }
        };

        let root = std::env::temp_dir().join(format!(
            "retread-selfheal-e2e-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let src = root.join("src");
        let prefix = root.join("prefix");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&prefix).unwrap();
        // Build script cp's the lock + installer under set -euo pipefail.
        std::fs::write(src.join("retread-isaac-pack.lock.json"), "{}").unwrap();
        std::fs::write(src.join("retread-installer"), "#!/bin/sh\n").unwrap();

        let recipe = build_courier_recipe("isaac-pack", "5.1.0", "3.11", &[], &[], None);
        let build = Command::new(&bash)
            .arg("-c")
            .arg(&recipe.build.script)
            .env("SRC_DIR", &src)
            .env("PREFIX", &prefix)
            .output()
            .expect("run build script");
        assert!(
            build.status.success(),
            "build script failed: {}",
            String::from_utf8_lossy(&build.stderr)
        );

        // Overwrite the shipped binary with a stub: `verify` succeeds iff the
        // payload sentinel exists; `install` creates the sentinel and logs.
        let stub = "#!/bin/bash\n\
             log=\"$CONDA_PREFIX/retread-stub.log\"\n\
             payload=\"$CONDA_PREFIX/lib/python3.11/site-packages/isaaclab/__init__.py\"\n\
             case \"$1\" in\n\
             verify) [ -f \"$payload\" ] ;;\n\
             install) echo \"install $(date +%s.%N)\" >> \"$log\"; mkdir -p \"$(dirname \"$payload\")\"; echo x > \"$payload\" ;;\n\
             *) exit 0 ;;\n\
             esac\n";
        std::fs::write(prefix.join("bin/retread"), stub).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(
                prefix.join("bin/retread"),
                std::fs::Permissions::from_mode(0o755),
            )
            .unwrap();
        }

        let guard = prefix.join("etc/conda/activate.d/zzz-retread-isaac-pack.sh");
        let deactivate = prefix.join("etc/conda/deactivate.d/zzz-retread-isaac-pack.sh");
        assert!(guard.is_file(), "activate.d guard not shipped");
        assert!(deactivate.is_file(), "deactivate.d hook not shipped");
        let source_guard = || {
            Command::new(&bash)
                .arg("-c")
                .arg(format!(". \"{}\"", guard.display()))
                .env("CONDA_PREFIX", &prefix)
                .output()
                .expect("source guard")
        };
        let log = prefix.join("retread-stub.log");
        let payload = prefix.join("lib/python3.11/site-packages/isaaclab/__init__.py");

        // Payload absent -> guard must self-heal: invoke install, restore it.
        assert!(!payload.exists(), "payload should start absent");
        let out = source_guard();
        assert!(payload.exists(), "self-heal must restore the payload");
        assert_eq!(
            std::fs::read_to_string(&log)
                .unwrap_or_default()
                .lines()
                .count(),
            1,
            "self-heal must invoke install exactly once"
        );
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("repairing"),
            "self-heal should announce the repair"
        );

        // Payload present -> guard is a no-op: verify passes, install NOT called.
        let out2 = source_guard();
        assert_eq!(
            std::fs::read_to_string(&log)
                .unwrap_or_default()
                .lines()
                .count(),
            1,
            "healthy env must not trigger another install (no wasted uv run)"
        );
        assert!(
            !String::from_utf8_lossy(&out2.stderr).contains("repairing"),
            "healthy env must be silent"
        );

        let ldlp = Command::new(&bash)
            .arg("-c")
            .arg(format!(
                ". \"{}\" >/dev/null 2>&1; . \"{}\" >/dev/null 2>&1; printf '%s' \"$LD_LIBRARY_PATH\"",
                guard.display(),
                guard.display()
            ))
            .env("CONDA_PREFIX", &prefix)
            .env("LD_LIBRARY_PATH", "/already")
            .output()
            .expect("double-source guard");
        assert!(ldlp.status.success());
        assert_eq!(
            String::from_utf8_lossy(&ldlp.stdout),
            format!("{}:/already", prefix.join("lib").display()),
            "activate.d must prepend prefix/lib exactly once"
        );

        let restored = Command::new(&bash)
            .arg("-c")
            .arg(format!(
                ". \"{}\" >/dev/null 2>&1; . \"{}\"; printf '%s' \"${{LD_LIBRARY_PATH-__unset__}}\"",
                guard.display(),
                deactivate.display()
            ))
            .env("CONDA_PREFIX", &prefix)
            .env("LD_LIBRARY_PATH", "/before")
            .output()
            .expect("source deactivate");
        assert!(restored.status.success());
        assert_eq!(String::from_utf8_lossy(&restored.stdout), "/before");

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn courier_activate_guard_broken_sentinel_backoff_and_strict_status() {
        use std::process::Command;

        let bash = match which("bash") {
            Some(b) => b,
            None => {
                eprintln!("skipping: bash not on PATH");
                return;
            }
        };

        let root = std::env::temp_dir().join(format!(
            "retread-broken-e2e-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let src = root.join("src");
        let prefix = root.join("prefix");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&prefix).unwrap();
        std::fs::write(src.join("retread-isaac-pack.lock.json"), "{}").unwrap();
        std::fs::write(src.join("retread-installer"), "#!/bin/sh\n").unwrap();

        let recipe = build_courier_recipe("isaac-pack", "5.1.0", "3.11", &[], &[], None);
        let build = Command::new(&bash)
            .arg("-c")
            .arg(&recipe.build.script)
            .env("SRC_DIR", &src)
            .env("PREFIX", &prefix)
            .output()
            .expect("run build script");
        assert!(
            build.status.success(),
            "build script failed: {}",
            String::from_utf8_lossy(&build.stderr)
        );

        let stub = "#!/bin/bash\n\
             log=\"$CONDA_PREFIX/retread-stub.log\"\n\
             case \"$1\" in\n\
             verify) exit 1 ;;\n\
             install) echo install >> \"$log\"; echo failed repair >&2; exit 42 ;;\n\
             *) exit 0 ;;\n\
             esac\n";
        std::fs::write(prefix.join("bin/retread"), stub).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(
                prefix.join("bin/retread"),
                std::fs::Permissions::from_mode(0o755),
            )
            .unwrap();
        }

        let guard = prefix.join("etc/conda/activate.d/zzz-retread-isaac-pack.sh");
        let source = |strict: bool| {
            let mut cmd = Command::new(&bash);
            cmd.arg("-c")
                .arg(format!(". \"{}\"; printf 'status:%s broken:%s' \"$?\" \"${{RETREAD_BROKEN_isaac_pack-0}}\"", guard.display()))
                .env("CONDA_PREFIX", &prefix);
            if strict {
                cmd.env("RETREAD_GUARD_STRICT", "1");
            }
            cmd.output().expect("source guard")
        };

        let first = source(false);
        assert!(
            first.status.success(),
            "default guard must not fail activation"
        );
        assert!(
            String::from_utf8_lossy(&first.stdout).contains("status:0 broken:1"),
            "default failure should set broken env and return zero: {}",
            String::from_utf8_lossy(&first.stdout)
        );
        let broken = prefix.join("share/retread/isaac-pack.broken");
        assert!(broken.is_file(), "failed heal must write .broken sentinel");
        assert_eq!(
            std::fs::read_to_string(prefix.join("retread-stub.log"))
                .unwrap_or_default()
                .lines()
                .count(),
            1,
            "first failure invokes installer once"
        );

        let second = source(false);
        assert!(second.status.success());
        assert_eq!(
            std::fs::read_to_string(prefix.join("retread-stub.log"))
                .unwrap_or_default()
                .lines()
                .count(),
            1,
            "second activation within backoff must not invoke installer again"
        );
        assert!(
            String::from_utf8_lossy(&second.stderr).contains("backoff"),
            "backoff should be announced"
        );

        std::fs::remove_file(&broken).unwrap();
        let strict = source(true);
        assert!(
            String::from_utf8_lossy(&strict.stdout).contains("status:1 broken:1"),
            "strict mode should make the sourced guard return 1: {}",
            String::from_utf8_lossy(&strict.stdout)
        );

        std::fs::remove_dir_all(&root).ok();
    }

    fn which(bin: &str) -> Option<std::path::PathBuf> {
        std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|d| d.join(bin))
                .find(|p| p.is_file())
        })
    }

    #[test]
    fn courier_recipe_with_expected_build_sets_string_field() {
        let r = build_courier_recipe(
            "mypack",
            "1.0.0",
            "3.11",
            &[],
            &[],
            Some("py311_habcdef0123_0"),
        );
        assert_eq!(
            r.build.string.as_deref(),
            Some("py311_habcdef0123_0"),
            "expected_build must be forwarded to build.string"
        );
    }

    #[test]
    fn courier_recipe_without_expected_build_has_no_string_field() {
        let r = build_courier_recipe("mypack", "1.0.0", "3.11", &[], &[], None);
        assert!(
            r.build.string.is_none(),
            "None expected_build must leave build.string absent"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RelaxPolicy;
    use std::collections::BTreeMap;

    fn cfg() -> RetreadConfig {
        RetreadConfig {
            resolver: Default::default(),
            auto_route: true,
            keep_pypi: vec![],
            force_conda: vec![],
            retread_wheels: BTreeMap::new(),
            relax: RelaxPolicy::Minor,
            overrides: BTreeMap::new(),
            name_map: BTreeMap::new(),
            shadow_libs: BTreeMap::new(),
            build_number: 0,
            drop_deps: Vec::new(),
            auto_bundle: false,
            conda_deps: Vec::new(),
            default_bundle: None,
            compression_level: None,
            emit_pypi: false,
            bundle_mode: crate::config::BundleMode::Fat,
            courier: false,
            blueprint: Default::default(),
            blueprint_sync: Default::default(),
            git_sources: std::collections::BTreeMap::new(),
            python: None,
            pin_version: false,
        }
    }

    fn one_source<'a>(meta: &'a WheelMetadata, url: &'a url::Url) -> Vec<BundleSource<'a>> {
        vec![BundleSource {
            pypi_name: &meta.name,
            url,
            metadata: meta,
        }]
    }

    #[test]
    fn renders_recipe_with_widened_pins() {
        let meta = WheelMetadata {
            name: "example-pkg".into(),
            version: "1.2.3".into(),
            requires_dist: vec![
                "numpy==1.26.4".into(),
                "torch==2.7.1".into(),
                "requests>=2.0".into(),
            ],
            is_pure_python: false,
            sha256: "deadbeef".into(),
            filename: "example_pkg-1.2.3-cp311-none-manylinux_2_35_x86_64.whl".into(),
        };
        let url: url::Url =
            "https://example.com/example_pkg-1.2.3-cp311-none-manylinux_2_35_x86_64.whl"
                .parse()
                .unwrap();
        let r = build_bundle_recipe(
            "example-pkg",
            &one_source(&meta, &url),
            &cfg(),
            "3.11",
            None,
            true,
        )
        .unwrap();
        let yaml = to_yaml(&r).unwrap();
        assert!(yaml.contains("python 3.11.*"), "yaml:\n{yaml}");
        assert!(yaml.contains("numpy >=1.26,<2"), "yaml:\n{yaml}");
        assert!(yaml.contains("torch >=2.7,<3"), "yaml:\n{yaml}");
        assert!(yaml.contains("requests >=2.0"), "yaml:\n{yaml}");
        assert!(!yaml.contains("noarch"), "should be platform-specific");
        // Platform-specific bundles must disable rattler-build's patchelf
        // pass -- NVIDIA's libs have rpath slots too short to rewrite and
        // some have non-UTF8 in their string tables that crashes goblin.
        // rattler-build's schema places `binary_relocation` under the
        // `dynamic_linking` group; emitting it at the top level of `build`
        // produces "unknown field 'binary_relocation'" at solve time.
        let dl = r
            .build
            .dynamic_linking
            .as_ref()
            .expect("platform-specific bundle must populate build.dynamic_linking");
        assert_eq!(dl.binary_relocation, Some(false));
        // YAML check pins down the exact nesting rattler-build expects.
        assert!(
            yaml.contains("dynamic_linking:") && yaml.contains("binary_relocation: false"),
            "expected `dynamic_linking:` with nested `binary_relocation: false`; yaml:\n{yaml}",
        );
    }

    #[test]
    fn pure_python_gets_noarch() {
        let meta = WheelMetadata {
            name: "pure".into(),
            version: "0.1.0".into(),
            requires_dist: vec![],
            is_pure_python: true,
            sha256: "abc".into(),
            filename: "pure-0.1.0-py3-none-any.whl".into(),
        };
        let url = "https://example.com/pure-0.1.0-py3-none-any.whl"
            .parse()
            .unwrap();
        let r = build_bundle_recipe("pure", &one_source(&meta, &url), &cfg(), "3.11", None, true)
            .unwrap();
        assert_eq!(r.build.noarch.as_deref(), Some("python"));
        // noarch bundles have nothing to relocate -- don't emit the field
        // (and don't risk poisoning future rattler-build default changes).
        assert!(r.build.dynamic_linking.is_none());
    }

    #[test]
    fn bundle_emits_multiple_sources_and_drops_vendored() {
        // Two wheels in a bundle: a metapackage that depends on its sibling.
        // The sibling's pypi_name matches the metapackage's `Requires-Dist`,
        // so it must be dropped from the conda run-deps (vendored).
        let primary = WheelMetadata {
            name: "isaacsim".into(),
            version: "5.1.0.0".into(),
            requires_dist: vec!["isaacsim-kernel==5.1.0.0".into(), "numpy==1.26.4".into()],
            is_pure_python: false,
            sha256: "primary_sha".into(),
            filename: "isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl".into(),
        };
        let primary_url: url::Url = "https://pypi.nvidia.com/isaacsim/isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl".parse().unwrap();
        let kernel = WheelMetadata {
            name: "isaacsim-kernel".into(),
            version: "5.1.0.0".into(),
            requires_dist: vec!["pillow==12.0.0".into()],
            is_pure_python: false,
            sha256: "kernel_sha".into(),
            filename: "isaacsim_kernel-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl".into(),
        };
        let kernel_url: url::Url = "https://pypi.nvidia.com/isaacsim-kernel/isaacsim_kernel-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl".parse().unwrap();

        let sources = vec![
            BundleSource {
                pypi_name: "isaacsim",
                url: &primary_url,
                metadata: &primary,
            },
            BundleSource {
                pypi_name: "isaacsim-kernel",
                url: &kernel_url,
                metadata: &kernel,
            },
        ];
        let r = build_bundle_recipe("isaacsim", &sources, &cfg(), "3.11", None, true).unwrap();
        let yaml = to_yaml(&r).unwrap();

        assert_eq!(r.source.len(), 2, "two sources in the recipe");
        assert!(
            yaml.contains("numpy >=1.26,<2"),
            "primary dep stays: {yaml}"
        );
        assert!(
            yaml.contains("pillow >=12.0,<13"),
            "extras dep stays: {yaml}"
        );
        assert!(
            !yaml.contains("isaacsim-kernel >="),
            "vendored sibling must NOT appear in run-deps: {yaml}"
        );
    }

    #[test]
    fn run_override_is_used_verbatim_not_rederived() {
        // When pixi forwards the solved run-deps (CondaBuildV1Params.
        // run_dependencies -> run_override), the recipe must use them as-is,
        // NOT re-derive from requires_dist. This keeps the built package's
        // deps identical to what the solve locked (cascade-widened) and avoids
        // re-emitting the raw transitive override that rattler-build rejects
        // as a malformed spec ("missing range specifier for '2.10.0'").
        let meta = WheelMetadata {
            name: "isaacsim".into(),
            version: "5.1.0.0".into(),
            // requires_dist that, if re-derived, would emit a tight torch pin.
            requires_dist: vec!["torch==2.10.0".into(), "numpy==1.26.4".into()],
            is_pure_python: false,
            sha256: "s".into(),
            filename: "isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl".into(),
        };
        let url: url::Url =
            "https://example.com/isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl"
                .parse()
                .unwrap();
        let over = vec![
            "python 3.11.*".to_string(),
            "pytorch >=1".to_string(), // the cascade-widened spec pixi solved with
            "numpy >=1.26,<2".to_string(),
        ];
        let r = build_bundle_recipe(
            "isaacsim",
            &one_source(&meta, &url),
            &cfg(),
            "3.11",
            Some(&over),
            true,
        )
        .unwrap();
        assert!(
            r.requirements.run.iter().any(|s| s == "pytorch >=1"),
            "must use the widened override verbatim: {:?}",
            r.requirements.run
        );
        assert!(
            !r.requirements.run.iter().any(|s| s.contains("2.10.0")),
            "must NOT re-derive the tight torch pin from requires_dist: {:?}",
            r.requirements.run
        );
        assert!(
            r.requirements.run.iter().any(|s| s.starts_with("python ")),
            "python must remain in run-deps: {:?}",
            r.requirements.run
        );
    }
}
