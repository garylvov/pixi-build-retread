//! Process-wide concurrency controls for expensive retread build work.

use std::sync::OnceLock;

use tokio::sync::{Semaphore, SemaphorePermit};

/// Preserve the historical entry-build window when no valid override is set.
pub const DEFAULT_MAX_CONCURRENT_BUILDS: usize = 6;

/// Parse `RETREAD_MAX_CONCURRENT_BUILDS` without reading process state.
///
/// Invalid values retain the historical default. Zero is treated as a request
/// for serial execution, and values larger than Tokio can represent are
/// clamped before constructing a semaphore.
pub fn parse_max_concurrent_builds(value: Option<&str>) -> usize {
    let parsed = value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_CONCURRENT_BUILDS);
    parsed.clamp(1, Semaphore::MAX_PERMITS)
}

/// Return this backend process's configured build concurrency.
pub fn max_concurrent_builds() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        parse_max_concurrent_builds(
            std::env::var("RETREAD_MAX_CONCURRENT_BUILDS")
                .ok()
                .as_deref(),
        )
    })
}

/// Acquire one process-wide permit for a real source-wheel build subprocess.
///
/// Callers deliberately acquire at the leaf, after all wheel-cache lookups,
/// so cache hits never consume scarce build capacity.
pub(crate) async fn acquire_build_permit() -> SemaphorePermit<'static> {
    static BUILDS: OnceLock<Semaphore> = OnceLock::new();
    BUILDS
        .get_or_init(|| Semaphore::new(max_concurrent_builds()))
        .acquire()
        .await
        .expect("retread's process-wide build semaphore is never closed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_concurrent_builds_parser_handles_defaults_and_bounds() {
        for value in [None, Some(""), Some("   "), Some("nope"), Some("-1")] {
            assert_eq!(
                parse_max_concurrent_builds(value),
                DEFAULT_MAX_CONCURRENT_BUILDS,
                "unexpected result for {value:?}",
            );
        }

        assert_eq!(parse_max_concurrent_builds(Some(" 3 \n")), 3);
        assert_eq!(parse_max_concurrent_builds(Some("0")), 1);
        assert_eq!(
            parse_max_concurrent_builds(Some(&usize::MAX.to_string())),
            Semaphore::MAX_PERMITS,
        );
        assert_eq!(
            parse_max_concurrent_builds(Some("999999999999999999999999999999999999999")),
            DEFAULT_MAX_CONCURRENT_BUILDS,
        );
    }
}
