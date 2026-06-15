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
/// uv-hardlink the wheels in (fetching index wheels on demand). The huge
/// index wheels never enter the conda package, so packaging is seconds and
/// nothing is committed to git.
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
    // build time -- via the quoted heredoc). The post-link runs the installer;
    // a failure is logged loudly but does not abort linking (the conda
    // metadata is still valid; the user can re-run `retread install`).
    let post_link = format!("$PREFIX/bin/.{conda_name}-post-link.sh");
    // A4 loud-failure guard: courier is the default mode, so a consumer who
    // forgets `run-post-link-scripts = "insecure"` would otherwise get a
    // SILENTLY broken env (post-link never runs, wheels never installed). We
    // also ship a conda activate.d script -- which runs on EVERY activation
    // regardless of the post-link toggle -- that checks the installer's success
    // marker and, if absent, prints a loud actionable banner. Turns an
    // invisible failure into one the user sees on the next `pixi run`/`shell`.
    let activate_guard = format!("$PREFIX/etc/conda/activate.d/zzz-retread-{conda_name}.sh");
    let script = format!(
        "set -euo pipefail\n\
         SHARE=\"$PREFIX/share/retread\"\n\
         WHEELS=\"$SHARE/{conda_name}/wheels\"\n\
         mkdir -p \"$WHEELS\" \"$PREFIX/bin\" \"$PREFIX/etc/conda/activate.d\"\n\
         cp \"$SRC_DIR\"/*.whl \"$WHEELS\"/ 2>/dev/null || true\n\
         cp \"$SRC_DIR\"/{lock_filename} \"$SHARE\"/\n\
         cp \"$SRC_DIR\"/retread-installer \"$PREFIX/bin/retread\"\n\
         chmod +x \"$PREFIX/bin/retread\"\n\
         cat > \"{post_link}\" <<'POSTLINK'\n\
         #!/bin/bash\n\
         \"$PREFIX/bin/retread\" install --lock \"$PREFIX/share/retread/{lock_filename}\" --prefix \"$PREFIX\" || echo 'retread: post-link install failed; run `retread install` manually' >&2\n\
         POSTLINK\n\
         chmod +x \"{post_link}\"\n\
         cat > \"{activate_guard}\" <<'ACTIVATE'\n\
         #!/bin/bash\n\
         # retread courier guard ({conda_name}): warn loudly if the bundle's\n\
         # PyPI wheels were never installed (post-link did not run -- almost\n\
         # always run-post-link-scripts is not enabled in .pixi/config.toml).\n\
         if [ ! -f \"$CONDA_PREFIX/share/retread/{conda_name}.installed\" ]; then\n\
         echo \"######################################################################\" >&2\n\
         echo \"# retread: bundle '{conda_name}' PyPI wheels are NOT installed.\" >&2\n\
         echo \"# The post-link installer did not run. Almost always this means\" >&2\n\
         echo \"# post-link scripts are disabled. Add to <workspace>/.pixi/config.toml:\" >&2\n\
         echo '#     run-post-link-scripts = \"insecure\"' >&2\n\
         echo \"# then re-run: pixi install   (see the pack README security note).\" >&2\n\
         echo \"# Or install now:\" >&2\n\
         echo \"#   retread install --lock \\\"$CONDA_PREFIX/share/retread/{lock_filename}\\\" --prefix \\\"$CONDA_PREFIX\\\"\" >&2\n\
         echo \"######################################################################\" >&2\n\
         fi\n\
         ACTIVATE\n\
         chmod +x \"{activate_guard}\"\n"
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
        // A4 loud-failure guard: a conda activate.d script that warns on every
        // activation when the wheels are not installed (missing toggle).
        assert!(
            r.build
                .script
                .contains("etc/conda/activate.d/zzz-retread-isaac-pack.sh"),
            "must ship an activate.d guard"
        );
        assert!(
            r.build
                .script
                .contains("$CONDA_PREFIX/share/retread/isaac-pack.installed"),
            "guard must check the installer success marker"
        );
        assert!(
            r.build
                .script
                .contains("run-post-link-scripts = \"insecure\""),
            "guard banner must name the toggle the user must enable"
        );
        assert!(
            r.build.script.contains("\nACTIVATE\n"),
            "activate.d heredoc terminator must be at column 0"
        );
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
            retread_wheels: BTreeMap::new(),
            relax: RelaxPolicy::Minor,
            overrides: BTreeMap::new(),
            name_map: BTreeMap::new(),
            build_number: 0,
            drop_deps: Vec::new(),
            auto_bundle: false,
            conda_deps: Vec::new(),
            default_bundle: None,
            compression_level: None,
            emit_pypi: false,
            courier: false,
            blueprint: Default::default(),
            blueprint_sync: Default::default(),
            git_sources: std::collections::BTreeMap::new(),
            python: None,
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
