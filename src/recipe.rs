//! Generate a rattler-build `recipe.yaml` for one repacked wheel.

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

/// Build a recipe from a wheel's parsed metadata, applying the relax policy
/// and name mapping from the retread config.
pub fn build_recipe(
    metadata: &WheelMetadata,
    source_url: &url::Url,
    config: &RetreadConfig,
) -> anyhow::Result<Recipe> {
    let python_version = python_version_from_wheel_tag(&metadata.filename)
        .unwrap_or_else(|| "3".to_string());
    let env = default_marker_env(&python_version)?;

    let python_pin = if python_version.contains('.') {
        format!("python {python_version}.*")
    } else {
        format!("python {python_version}")
    };

    let mut run = vec![python_pin.clone()];

    for raw in &metadata.requires_dist {
        match translate(raw, &env, &config.name_map, &config.overrides, config.relax) {
            Ok(Some(dep)) => run.push(dep.0),
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(req = %raw, error = %e, "could not translate requirement; dropping");
            }
        }
    }

    let host = vec![python_pin, "pip".to_string()];

    let noarch = if metadata.is_pure_python {
        Some("python".to_string())
    } else {
        None
    };

    Ok(Recipe {
        schema_version: 1,
        package: Package {
            name: metadata.name.to_ascii_lowercase().replace('_', "-"),
            version: metadata.version.clone(),
        },
        source: vec![Source {
            url: source_url.to_string(),
            sha256: Some(metadata.sha256.clone()),
        }],
        build: Build {
            number: config.build_number,
            noarch,
            // `--no-deps` is essential: conda solves deps from the run: list,
            // not from pip resolving Requires-Dist again at install time.
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
            wheels: BTreeMap::new(),
            relax: RelaxPolicy::Minor,
            overrides: BTreeMap::new(),
            name_map: BTreeMap::new(),
            build_number: 0,
        }
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
        let r = build_recipe(&meta, &url, &cfg()).unwrap();
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
        let r = build_recipe(&meta, &url, &cfg()).unwrap();
        assert_eq!(r.build.noarch.as_deref(), Some("python"));
    }
}
