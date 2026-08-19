//! Library crate exposing retread's internals so they can be exercised by
//! integration tests under `tests/` and reused if we ever embed the backend
//! in another tool. The binary at `src/main.rs` consumes these modules
//! through the crate.

pub mod audit;
pub mod compat;
pub mod concurrency;
pub mod conda_solve;
pub mod config;
pub mod constraint;
pub mod courier;
pub mod deps_from;
pub mod emit_pypi;
pub mod fasttmp;
pub mod glibc;
pub mod handler;
pub mod hermetic_build;
mod index_chain;
pub mod installer;
pub mod lock;
pub mod pack_overrides;
pub mod panic_hook;
pub mod probe;
pub mod pypi;
pub mod recipe;
pub mod relax;
pub(crate) mod relax_decision;
pub mod relaxation_record;
pub mod repair;
pub mod repodata;
pub mod rpc;
pub mod solve;
pub mod source_build;
pub mod status;
pub(crate) mod thread_budget;
pub mod uv_closure;
pub mod wheel;
pub mod wheel_inject;
pub mod wheel_inject_data;
pub mod wheel_rewrite;
pub mod workspace;

// Process environment is global even when Rust's test harness runs modules in
// parallel. Env-sensitive tests share this lock so one module cannot
// transiently switch another module's production path.
#[cfg(test)]
pub(crate) static TEST_ASYNC_ENV_MUTEX: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
