//! Best-effort status lines to the controlling terminal.
//!
//! pixi captures the backend's stdout (the JSON-RPC channel) and BUFFERS its
//! stderr behind its own progress spinner -- so retread's `tracing` logs are
//! invisible during `pixi shell` / `pixi install` unless the user passes `-v`.
//! That makes the slow phases (multi-GB wheel materialization, the conda
//! solve-checks, rattler-build) look like a silent freeze.
//!
//! To keep the user informed regardless of pixi's verbosity, retread writes
//! short status lines straight to `/dev/tty`, which bypasses pixi's stderr
//! capture and reaches the user's terminal directly. No-op when there is no
//! controlling terminal (CI, redirected output), so it never errors and never
//! pollutes the JSON-RPC stdout channel.

use std::io::Write;

/// Write one status line to the controlling terminal, best-effort.
///
/// Cheap and infrequent: called at phase boundaries and inside the slow
/// solve/refine loops, NOT per repodata record. Opening `/dev/tty` per call
/// keeps it stateless and thread-safe.
pub fn tty(msg: &str) {
    if let Ok(mut f) = std::fs::OpenOptions::new().write(true).open("/dev/tty") {
        // Leading CR so we start at column 0 even if pixi's in-place progress
        // bar left the cursor mid-line; trailing newline flushes a full line
        // (pixi redraws its bar underneath).
        let _ = write!(f, "\r[retread] {msg}\n");
        let _ = f.flush();
    }
}
