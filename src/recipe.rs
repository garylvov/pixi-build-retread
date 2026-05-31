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
                        let dep_name = dep.0.split_whitespace().next().unwrap_or("").to_string();
                        if vendored.contains(&dep_name) {
                            continue;
                        }
                        if seen.insert(dep_name) {
                            run.push(dep.0);
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

    let recipe_sources = sources
        .iter()
        .map(|s| Source {
            url: s.url.to_string(),
            sha256: Some(s.metadata.sha256.clone()),
        })
        .collect();

    Ok(Recipe {
        schema_version: 1,
        package: Package {
            name: conda_name.to_string(),
            version: primary.metadata.version.clone(),
        },
        source: recipe_sources,
        build: Build {
            number: config.build_number,
            noarch,
            // Vendor wheels (Omniverse, manylinux) ship pre-baked rpaths;
            // rattler-build's default relocation pass either overflows the
            // original DT_RPATH slot or chokes on non-UTF8 in some .so
            // string tables. Skip the patchelf step. Only meaningful for
            // platform-specific bundles -- noarch has no native libs.
            dynamic_linking: if any_platform_specific {
                Some(DynamicLinking { binary_relocation: Some(false) })
            } else {
                None
            },
            // `--no-deps` is essential: conda solves deps from the run: list,
            // not from pip re-resolving Requires-Dist at install time.
            script: "${{ PYTHON }} -m pip install *.whl -vv --no-deps --no-build-isolation"
                .to_string(),
        },
        requirements: Requirements { host, run },
        about: About {
            license: None,
            summary: None,
        },
    })
}

pub fn to_yaml(recipe: &Recipe) -> anyhow::Result<String> {
    Ok(serde_yaml::to_string(recipe)?)
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
            git_sources: std::collections::BTreeMap::new(),
            python: None,
        }
    }

    fn one_source<'a>(
        meta: &'a WheelMetadata,
        url: &'a url::Url,
    ) -> Vec<BundleSource<'a>> {
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
        let url: url::Url = "https://example.com/example_pkg-1.2.3-cp311-none-manylinux_2_35_x86_64.whl".parse().unwrap();
        let r = build_bundle_recipe("example-pkg", &one_source(&meta, &url), &cfg(), "3.11", None).unwrap();
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
        let url = "https://example.com/pure-0.1.0-py3-none-any.whl".parse().unwrap();
        let r = build_bundle_recipe("pure", &one_source(&meta, &url), &cfg(), "3.11", None).unwrap();
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
            requires_dist: vec![
                "isaacsim-kernel==5.1.0.0".into(),
                "numpy==1.26.4".into(),
            ],
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
        let r = build_bundle_recipe("isaacsim", &sources, &cfg(), "3.11", None).unwrap();
        let yaml = to_yaml(&r).unwrap();

        assert_eq!(r.source.len(), 2, "two sources in the recipe");
        assert!(yaml.contains("numpy >=1.26,<2"), "primary dep stays: {yaml}");
        assert!(yaml.contains("pillow >=12.0,<13"), "extras dep stays: {yaml}");
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
        let r = build_bundle_recipe("isaacsim", &one_source(&meta, &url), &cfg(), "3.11", Some(&over))
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
