//! `retread solve`: pixi subprocess repair loop for manifest-level solver conflicts.

pub mod args;
mod driver;
mod error;
pub mod lock;
mod manifest;
mod parse;
mod repair;
mod smoke;

pub use driver::run;
// v4.2.0: the deleted in-backend conflict classifier's anchor check
// lives on here; re-exported for the anchor-surface pin test in relax.rs.
pub use repair::is_abi_anchor;
