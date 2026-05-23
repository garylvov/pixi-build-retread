//! Library crate exposing retread's internals so they can be exercised by
//! integration tests under `tests/` and reused if we ever embed the backend
//! in another tool. The binary at `src/main.rs` consumes these modules
//! through the crate.

pub mod config;
pub mod handler;
pub mod recipe;
pub mod relax;
pub mod rpc;
pub mod wheel;
