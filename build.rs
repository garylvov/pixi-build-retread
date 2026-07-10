//! Embeds the backend's build identity (git commit hash) so
//! `conda_outputs_cache_key` can fold it in: a binary built from a
//! different commit must never reuse another build's cached pack renders
//! (run-30 of the retread-deps-from proof hit exactly that -- the
//! bounded-range emission binary was served pre-fix exact-pin renders
//! because the key carried only manifest mtime + overrides-ledger hash).

use std::process::Command;

fn main() {
    // Re-run when HEAD moves (commit/checkout). Best-effort: if .git is
    // missing (crates.io tarball build) the rerun directives are inert.
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs");

    let hash = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    // Mark dirty builds distinctly: two binaries at the same HEAD but
    // with different uncommitted changes must not share a cache.
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    let ident = if dirty { format!("{hash}-dirty") } else { hash };
    println!("cargo:rustc-env=RETREAD_GIT_HASH={ident}");
}
