//! The verb surface of the `retread` CLI, and the refusal for an argument that
//! is not one (N27-RETREAD-65, SHIM-AUTO-2).
//!
//! WHY THIS IS A LIBRARY MODULE AND NOT TEN LINES IN `main.rs`. `main.rs` is a
//! binary target: `cargo test --lib` -- which is what the certifying gate runs
//! -- cannot reach a `#[cfg(test)]` module inside it, so a guard written there
//! would be a guard no gate ever executes, which law 3 counts as a defect in
//! its own right. The decision is therefore a pure function of `argv` that
//! lives here with its tests, and `main.rs` holds only the call site.
//!
//! THE INCIDENT THIS EXISTS FOR. `async_main`'s dispatch chain is a run of
//! `if matches!(argv.get(1) …)` arms, every one of which `return`s or
//! `std::process::exit`s. Anything else fell through to the JSON-RPC
//! build-backend transport -- which pixi launches with NO arguments at all --
//! where an already-closed stdin reads as EOF and the process returns `Ok(())`.
//! A verb this binary did not have was therefore a SUCCESSFUL NO-OP.
//!
//! C34 job 6066294 arm 3 ran
//!
//! ```text
//! pixi-build-retread path-source-manifest --workspace <ws> --out <ws>/pixi.toml \
//!                    --pack <ws>/pypi-packs/isaaclab-2.3x-pack \
//!                    --pack <ws>/pypi-packs/protomotions-deps-pack
//! ```
//!
//! against a binary cut before `path-source-manifest` existed. It printed the
//! startup banner, wrote nothing, and exited 0. The caller's `rc != 0` check
//! passed; only its second assertion -- the md5 of the `--out` file, still the
//! canonical `9711eb99…`/45298 B where the effective `4ad488b9…`/45393 B was
//! required -- caught it. A caller without that second assertion would have
//! locked the canonical manifest and published the wall as a measurement of the
//! shim shape.

/// The verbs `async_main` dispatches, for the refusal's DIAGNOSTIC only.
///
/// This list is deliberately not the condition. [`verb_candidate`] fires on
/// fall-through -- the call site sits after every dispatch arm -- so a name
/// missing from here degrades an error message and can never make the binary
/// refuse a verb that works. A second copy of the dispatch table that DECIDED
/// behaviour would go stale exactly once and then break production.
pub const KNOWN_VERBS: &[&str] = &[
    crate::uv_closure::ENV_SEED_VERB,
    "fast",
    "preflight",
    "repodata-universe",
    "sharded-universe",
    "migrate-overrides",
    "store-reap",
    "sdist-meta-key",
    "sdist-meta-python-tags",
    "path-source-refresh",
    "path-source-manifest",
    "install",
    "verify",
    "solve",
    "lock",
];

/// The first argument, if it is shaped like a verb rather than a flag.
///
/// `None` for the zero-argument invocation -- which is how pixi launches this
/// binary and is the one path that must reach the transport -- and `None` for
/// anything beginning with `-`, so a future transport flag is not read as a
/// mistyped verb. Called AFTER every dispatch arm, so a `Some` here means
/// nobody handled it.
pub fn verb_candidate(argv: &[String]) -> Option<&str> {
    argv.get(1)
        .map(String::as_str)
        .filter(|a| !a.starts_with('-'))
}

/// What the binary says while refusing an argument it does not handle.
///
/// It names the version, because the ordinary cause is a caller pinned to a
/// binsnap older than the branch that added the verb, and it names the
/// no-argument transport, because the failure mode being closed is precisely
/// that the transport had been acting as a silent fallback.
pub fn unknown_verb_message(verb: &str, version: &str) -> String {
    format!(
        "retread: unknown verb `{verb}`. This binary is version {version} and handles: {}. \
         The JSON-RPC build-backend transport is the NO-ARGUMENT invocation and is never a \
         fallback for an argument this binary does not know -- a capability that is absent \
         has to say so, or a caller measures the wrong thing and calls it a pass.",
        KNOWN_VERBS.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    /// THE C34 SHAPE. The exact argv job 6066294 arm 3 passed, against a binary
    /// that does not carry the verb, must be recognised as unhandled. A
    /// `None` here is the incident: the caller would go on to the transport and
    /// exit 0.
    #[test]
    fn n27_65_the_c34_argv_is_recognised_as_an_unhandled_verb() {
        let a = argv(&[
            "pixi-build-retread",
            "path-source-manifest",
            "--workspace",
            "/ws",
            "--out",
            "/ws/pixi.toml",
            "--pack",
            "/ws/pypi-packs/isaaclab-2.3x-pack",
            "--pack",
            "/ws/pypi-packs/protomotions-deps-pack",
        ]);
        assert_eq!(verb_candidate(&a), Some("path-source-manifest"));
    }

    /// THE FALSIFIER, and it matters more than the guard above. pixi launches
    /// this binary with no arguments; if the refusal reached that path the fix
    /// would be worse than the defect, because every build in production would
    /// refuse. Stated as a test so it cannot be tightened away by accident.
    #[test]
    fn n27_65_the_no_argument_transport_invocation_is_never_a_verb() {
        assert_eq!(verb_candidate(&argv(&["pixi-build-retread"])), None);
        assert_eq!(verb_candidate(&[]), None);
    }

    /// A leading `-` is a flag, not a mistyped verb.
    #[test]
    fn n27_65_flags_are_not_verbs() {
        assert_eq!(
            verb_candidate(&argv(&["pixi-build-retread", "--version"])),
            None
        );
        assert_eq!(verb_candidate(&argv(&["pixi-build-retread", "-v"])), None);
    }

    /// The message has to carry the two facts a caller needs to act: which
    /// argument was refused, and which binary refused it. C34's operator read
    /// `rc=0` and a stale binsnap and could tell neither.
    #[test]
    fn n27_65_the_refusal_names_the_verb_and_the_version() {
        let message = unknown_verb_message("path-source-manifest", "4.10.90");
        assert!(message.contains("unknown verb"), "{message}");
        assert!(message.contains("path-source-manifest"), "{message}");
        assert!(message.contains("4.10.90"), "{message}");
        assert!(message.contains("NO-ARGUMENT"), "{message}");
    }

    /// The diagnostic list must actually list the verbs, including the one
    /// whose absence caused the incident and the one it is confused with.
    #[test]
    fn n27_65_known_verbs_carries_the_dispatched_names() {
        for verb in [
            "sharded-universe",
            "path-source-manifest",
            "path-source-refresh",
            "sdist-meta-key",
            "store-reap",
            "lock",
            crate::uv_closure::ENV_SEED_VERB,
        ] {
            assert!(KNOWN_VERBS.contains(&verb), "KNOWN_VERBS is missing {verb}");
        }
    }
}
