use thiserror::Error;

pub const EXIT_OK: i32 = 0;
pub const EXIT_USAGE: i32 = 1;
pub const EXIT_UNPARSEABLE: i32 = 2;
pub const EXIT_EXHAUSTED: i32 = 3;
pub const EXIT_MAX_ITERS: i32 = 4;
pub const EXIT_SMOKE_FAILED: i32 = 5;
pub const EXIT_INTERRUPTED: i32 = 130;

#[derive(Debug, Error)]
pub enum SolveError {
    #[error("{0}")]
    Usage(String),
    #[error("retread solve: could not parse conflict")]
    Unparseable { stderr_tail: String },
    #[error("retread solve: exhausted repair strategies for {package}")]
    Exhausted { package: String },
    #[error("retread solve: max iterations reached ({0})")]
    MaxIters(u32),
    #[error("retread solve: smoke test failed importing {module}")]
    SmokeFailed { module: String, stderr_tail: String },
    #[error("retread solve: interrupted")]
    Interrupted,
}

impl SolveError {
    pub fn exit_code(&self) -> i32 {
        match self {
            SolveError::Usage(_) => EXIT_USAGE,
            SolveError::Unparseable { .. } => EXIT_UNPARSEABLE,
            SolveError::Exhausted { .. } => EXIT_EXHAUSTED,
            SolveError::MaxIters(_) => EXIT_MAX_ITERS,
            SolveError::SmokeFailed { .. } => EXIT_SMOKE_FAILED,
            SolveError::Interrupted => EXIT_INTERRUPTED,
        }
    }
}
