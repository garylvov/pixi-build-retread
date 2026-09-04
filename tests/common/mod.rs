//! Shared fixtures for the integration tests in `tests/`.
//!
//! WHY THIS EXISTS. `tests/isaacsim_relax.rs` and `tests/wheel_fetch_live.rs`
//! each carried a hand-written `RetreadConfig { .. }` struct literal naming
//! every field. `RetreadConfig` gains fields regularly, and a struct literal
//! that names them all stops compiling the moment one is added — which is
//! exactly what happened: both targets had been dead for some time, and the
//! campaign gate (`cargo test --lib`) never builds an integration test, so
//! nothing said so.
//!
//! Building the config through serde instead of through a literal is what
//! makes that class of breakage impossible: every optional key on
//! `RetreadConfig` carries `#[serde(default)]`, so a NEW field simply takes
//! its default here and this helper keeps compiling. It is also the pattern
//! the library's own unit tests already use (`handler::mod::empty_config`,
//! `handler::auto_bundle::test_config`).
//!
//! Keys are spelled exactly as `RetreadConfig`'s `rename` attributes require —
//! the struct is `#[serde(deny_unknown_fields)]`, so a typo is a loud panic
//! here, never a silently ignored setting.

use pixi_build_retread::config::RetreadConfig;

/// The config both integration tests were built around.
///
/// Every key below is one the old struct literals set to something OTHER than
/// the field's own serde default, OR one whose value the tests actively depend
/// on and which therefore must not drift with a default change:
///
/// * `retread-wheels` — required (no `#[serde(default)]`); the tests feed
///   metadata in directly rather than through a wheel entry.
/// * `retread-relax = "minor"` — the whole point of `isaacsim_relax.rs`.
/// * `retread-route-policy = "aggressive"` — the pre-v4.6 legacy sweep
///   semantics the test matrix was written against (default is
///   `prefer-conda-validated`).
/// * `retread-bundle-mode = "fat"` — the tests render a full recipe, not a
///   loose stub (default is `loose`).
/// * `retread-auto-bundle = false` and `retread-courier = false` — both
///   DEFAULT TO TRUE; the literals turned them off and the tests' expectations
///   assume that.
/// * `retread-auto-route = true` and `retread-hermetic = true` — these happen
///   to match today's defaults, but the literals stated them, so they are
///   stated here too rather than inherited.
///
/// Intentionally stripped, as the original comment said: no overrides, no
/// name-map. The point is proving `relax = "minor"` alone is enough to let
/// ros2 + isaacsim coexist. Compare gigastrap's
/// `[feature.isaaclab.pypi-options.dependency-overrides]` block — that is what
/// should NOT be needed once retread is in the loop.
pub fn baseline_config() -> RetreadConfig {
    serde_json::from_value(serde_json::json!({
        "retread-wheels": {},
        "retread-relax": "minor",
        "retread-route-policy": "aggressive",
        "retread-bundle-mode": "fat",
        "retread-auto-bundle": false,
        "retread-courier": false,
        "retread-auto-route": true,
        "retread-hermetic": true,
    }))
    .expect("baseline_config keys must match RetreadConfig's serde renames")
}
