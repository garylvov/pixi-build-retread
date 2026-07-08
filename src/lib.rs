//! Library crate exposing retread's internals so they can be exercised by
//! integration tests under `tests/` and reused if we ever embed the backend
//! in another tool. The binary at `src/main.rs` consumes these modules
//! through the crate.

pub mod audit;
pub mod compat;
pub mod conda_solve;
pub mod config;
pub mod courier;
pub mod emit_pypi;
pub mod fasttmp;
pub mod glibc;
pub mod handler;
pub mod installer;
pub mod lock;
pub mod probe;
pub mod pypi;
pub mod recipe;
pub mod relax;
pub mod repodata;
pub mod rpc;
pub mod solve;
pub mod source_build;
pub mod status;
pub mod uv_closure;
pub mod wheel;
pub mod wheel_inject;
pub mod wheel_inject_data;
pub mod wheel_rewrite;
pub mod workspace;
