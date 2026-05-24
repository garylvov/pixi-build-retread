//! Generate a rattler-build `recipe.yaml` for a bundle of repacked wheels.
//!
//! The bundle pattern: one conda package whose `source:` list contains every
//! wheel in the bundle (the user's named entry plus extras-derived
//! sub-wheels). All wheels are pip-installed into the same prefix at build
//! time. Mirrors comment 24 of prefix-dev/pixi#5230.

use std::collections::HashSet;

use serde::Serialize;

use crate::config::RetreadConfig;
use crate::relax::{default_marker_env, python_version_from_wheel_tag, translate};
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
) -> anyhow::Result<Recipe> {
    let primary = sources
        .first()
        .ok_or_else(|| anyhow::anyhow!("bundle must have at least one source"))?;
    let python_version = python_version_from_wheel_tag(&primary.metadata.filename)
        .unwrap_or_else(|| "3".to_string());
    let env = default_marker_env(&python_version)?;

    let python_pin = if python_version.contains('.') {
        format!("python {python_version}.*")
    } else {
        format!("python {python_version}")
    };

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
        let r = build_bundle_recipe("example-pkg", &one_source(&meta, &url), &cfg()).unwrap();
        let yaml = to_yaml(&r).unwrap();
        assert!(yaml.contains("python 3.11.*"), "yaml:\n{yaml}");
        assert!(yaml.contains("numpy >=1.26,<2"), "yaml:\n{yaml}");
        assert!(yaml.contains("torch >=2.7,<3"), "yaml:\n{yaml}");
        assert!(yaml.contains("requests >=2.0"), "yaml:\n{yaml}");
        assert!(!yaml.contains("noarch"), "should be platform-specific");
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
        let r = build_bundle_recipe("pure", &one_source(&meta, &url), &cfg()).unwrap();
        assert_eq!(r.build.noarch.as_deref(), Some("python"));
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
        let r = build_bundle_recipe("isaacsim", &sources, &cfg()).unwrap();
        let yaml = to_yaml(&r).unwrap();

        assert_eq!(r.source.len(), 2, "two sources in the recipe");
        assert!(yaml.contains("numpy >=1.26,<2"), "primary dep stays: {yaml}");
        assert!(yaml.contains("pillow >=12.0,<13"), "extras dep stays: {yaml}");
        assert!(
            !yaml.contains("isaacsim-kernel >="),
            "vendored sibling must NOT appear in run-deps: {yaml}"
        );
    }
}
