//! pixi >0.70 wire-protocol compatibility shims.
//!
//! Pixi Build API version 5 (pixi 0.72+, `prefix-dev/pixi#6328`) serializes
//! the `spec` field of every `CondaBuildV1Dependency` in `conda/build_v1`
//! params as a STRUCTURED MatchSpec object (`{"name": "python", "version":
//! "3.12.*"}`) instead of the API-4 display string (`"python 3.12.*"`).
//! Retread pins `pixi_build_types` at an API-4 rev whose
//! `CondaBuildV1Dependency::spec` is `DisplayFromStr`, so v5 params fail
//! deserialization with `invalid type: map, expected a string`.
//!
//! Rather than bumping the whole `pixi_build_types` + rattler pin set (which
//! would break the wire types against pixi 0.70.x in the other direction),
//! this module normalizes incoming `conda/build_v1` params BEFORE typed
//! deserialization: any map-form `spec` is deserialized through our pinned
//! `rattler_conda_types::MatchSpec` (whose structured serde field names match
//! the v5 wire format, including `condition`) and re-rendered as its display
//! string. String-form specs (API <= 4, pixi <= 0.70.x) pass through
//! untouched, so one binary speaks both protocol revisions.
//!
//! Known lossiness: a v5 `condition` (`if(...)` conditional dependency)
//! cannot be represented in the API-4 string form; it is dropped with a
//! warning. Retread never receives conditional deps from its own emitted
//! packages today, so this only fires for exotic workspace-authored source
//! packages.

use serde_json::Value;

/// Normalize v5-structured `conda/build_v1` params into the API-4 shape the
/// pinned `pixi_build_types::CondaBuildV1Params` deserializes. Idempotent;
/// API-4 params pass through byte-identical.
pub fn normalize_conda_build_v1_params(mut params: Value) -> Value {
    // Every location where `Vec<CondaBuildV1Dependency>` appears in the
    // params schema (camelCase wire form).
    for prefix_key in ["buildPrefix", "hostPrefix"] {
        if let Some(prefix) = params.get_mut(prefix_key) {
            normalize_dependency_array(prefix.get_mut("dependencies"));
            normalize_dependency_array(prefix.get_mut("constraints"));
        }
    }
    normalize_dependency_array(params.get_mut("runDependencies"));
    normalize_dependency_array(params.get_mut("runConstraints"));
    if let Some(run_exports) = params.get_mut("runExports") {
        for key in [
            "weak",
            "strong",
            "noarch",
            "weakConstrains",
            "strongConstrains",
        ] {
            normalize_dependency_array(run_exports.get_mut(key));
        }
    }
    // API 5 addition: extra dependency groups (map group -> Vec<dep>). The
    // pinned type has no such field (serde ignores it), but normalize anyway
    // so a future field addition deserializes cleanly.
    if let Some(Value::Object(groups)) = params.get_mut("extraDependencies") {
        for (_, deps) in groups.iter_mut() {
            normalize_dependency_array(Some(deps));
        }
    }
    params
}

/// Rewrite each `{"spec": {...}, ...}` element's map-form spec into the
/// display-string form. Non-array values and string-form specs are left
/// untouched.
fn normalize_dependency_array(deps: Option<&mut Value>) {
    let Some(Value::Array(deps)) = deps else {
        return;
    };
    for dep in deps.iter_mut() {
        let Some(spec) = dep.get_mut("spec") else {
            continue;
        };
        if !spec.is_object() {
            continue;
        }
        match structured_spec_to_string(spec) {
            Some(rendered) => *spec = Value::String(rendered),
            None => {
                tracing::warn!(
                    spec = %spec,
                    "conda/build_v1: could not normalize structured match spec; \
                     leaving as-is (typed deserialization will fail)"
                );
            }
        }
    }
}

/// Deserialize a structured (API 5) match spec through the pinned rattler
/// `MatchSpec` and render it as the canonical display string. Returns `None`
/// when the object doesn't fit the pinned MatchSpec schema.
fn structured_spec_to_string(spec: &Value) -> Option<String> {
    let mut parsed: rattler_conda_types::MatchSpec = serde_json::from_value(spec.clone()).ok()?;
    if parsed.condition.is_some() {
        tracing::warn!(
            spec = %spec,
            "conda/build_v1: dropping `condition` from structured match spec; \
             conditional (`if(...)`) dependencies are not representable in the \
             API-4 string form retread consumes"
        );
    }
    // rattler's Display only renders `subdir` as part of a `channel/subdir`
    // prefix; a subdir WITHOUT a channel would come out as a bare `::` that
    // does not re-parse. Drop it (the host/build prefix is already solved by
    // pixi; retread never routes on prefix-dep subdirs).
    if parsed.channel.is_none() && parsed.subdir.take().is_some() {
        tracing::debug!(
            spec = %spec,
            "conda/build_v1: dropping channel-less `subdir` from structured match spec"
        );
    }
    Some(parsed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn structured_specs_are_rendered_to_strings() {
        // Shape captured from a real pixi 0.72.1 conda/build_v1 request.
        let params = json!({
            "channels": ["https://prefix.dev/conda-forge"],
            "hostPrefix": {
                "prefix": "/tmp/host",
                "platform": "linux-64",
                "dependencies": [
                    {"spec": {"name": "python", "version": "3.12.*"}, "source": null},
                    {"spec": {"name": "pip"}, "source": null},
                ],
                "constraints": [],
                "packages": [],
            },
            "output": {
                "name": "tiny-pack",
                "version": "0.9.0",
                "build": null,
                "subdir": "linux-64",
                "variant": {},
            },
            "workDirectory": "/tmp/work",
        });
        let out = normalize_conda_build_v1_params(params);
        let deps = &out["hostPrefix"]["dependencies"];
        assert_eq!(deps[0]["spec"], json!("python 3.12.*"));
        assert_eq!(deps[1]["spec"], json!("pip"));
        // And the pinned typed params must now deserialize.
        let typed: Result<pixi_build_types::procedures::conda_build_v1::CondaBuildV1Params, _> =
            serde_json::from_value(out);
        let typed = typed.expect("normalized params must deserialize into pinned type");
        let host = typed.host_prefix.expect("host prefix");
        assert_eq!(host.dependencies.len(), 2);
        assert_eq!(host.dependencies[0].spec.to_string(), "python 3.12.*");
    }

    #[test]
    fn string_specs_pass_through_unchanged() {
        let params = json!({
            "runDependencies": [
                {"spec": "numpy >=1.26,<2", "source": null},
            ],
        });
        let out = normalize_conda_build_v1_params(params.clone());
        assert_eq!(out, params);
    }

    #[test]
    fn run_exports_and_extra_groups_are_normalized() {
        let params = json!({
            "runExports": {
                "weak": [{"spec": {"name": "libfoo", "version": ">=1.0,<2"}}],
                "strong": [],
                "noarch": [],
                "weakConstrains": [],
                "strongConstrains": [],
            },
            "runConstraints": [{"spec": {"name": "openmp", "version": "<0.0a0"}}],
            "extraDependencies": {
                "cuda": [{"spec": {"name": "cuda-version", "version": "==12.8"}}],
            },
        });
        let out = normalize_conda_build_v1_params(params);
        assert_eq!(
            out["runExports"]["weak"][0]["spec"],
            json!("libfoo >=1.0,<2")
        );
        assert_eq!(out["runConstraints"][0]["spec"], json!("openmp <0.0a0"));
        assert_eq!(
            out["extraDependencies"]["cuda"][0]["spec"],
            json!("cuda-version ==12.8")
        );
    }

    #[test]
    fn richer_structured_fields_survive() {
        let params = json!({
            "runDependencies": [
                {"spec": {"name": "foo", "version": ">=1.2", "build": "py312*", "subdir": "linux-64"}},
            ],
        });
        let out = normalize_conda_build_v1_params(params);
        let s = out["runDependencies"][0]["spec"].as_str().unwrap();
        // Exact display format is rattler's business; assert the components.
        assert!(s.contains("foo"), "{s}");
        assert!(s.contains(">=1.2"), "{s}");
        assert!(s.contains("py312"), "{s}");
        // A channel-less `subdir` is deliberately dropped: rattler's Display
        // can only render it as part of a `channel/subdir::` prefix, and a
        // bare `::` would not re-parse.
        assert!(!s.contains("linux-64"), "{s}");
        // Must round-trip through the pinned string parser.
        let reparsed: Result<rattler_conda_types::MatchSpec, _> = std::str::FromStr::from_str(s);
        reparsed.expect("normalized spec string must re-parse");
    }
}
