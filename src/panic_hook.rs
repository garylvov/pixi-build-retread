//! Process-wide panic diagnostics for the backend binary.

use std::backtrace::Backtrace;
use std::io::Write as _;

/// Install a panic hook that always writes the panic and a forced backtrace.
///
/// The backend is commonly launched behind Pixi's JSON-RPC transport, where a
/// worker panic can otherwise look like an unexplained EOF. Write directly to
/// stderr and ignore I/O errors so diagnostics never cause a second panic.
pub fn install_global_panic_hook() {
    std::panic::set_hook(Box::new(|panic| {
        let thread = std::thread::current();
        let thread_name = thread.name().unwrap_or("<unnamed>");
        let backtrace = Backtrace::force_capture();
        let mut stderr = std::io::stderr().lock();
        let _ = writeln!(
            stderr,
            "retread: fatal panic on thread {thread_name}: {panic}"
        );
        let _ = writeln!(stderr, "retread: forced backtrace:\n{backtrace}");
        let _ = stderr.flush();
    }));
}

#[cfg(test)]
mod tests {
    const PANIC_HOOK_CHILD: &str = "RETREAD_TEST_PANIC_HOOK_CHILD";

    #[test]
    fn panic_hook_child() {
        if std::env::var_os(PANIC_HOOK_CHILD).is_none() {
            return;
        }
        super::install_global_panic_hook();
        panic!("panic-hook regression sentinel");
    }

    #[test]
    fn global_panic_hook_writes_panic_and_backtrace() {
        let output = std::process::Command::new(
            std::env::current_exe().expect("the test executable must have a path"),
        )
        .args([
            "panic_hook::tests::panic_hook_child",
            "--exact",
            "--nocapture",
        ])
        .env(PANIC_HOOK_CHILD, "1")
        .output()
        .expect("the panic-hook child must launch");

        assert!(!output.status.success(), "the child panic must fail");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("retread: fatal panic"), "{stderr}");
        assert!(
            stderr.contains("panic-hook regression sentinel"),
            "{stderr}"
        );
        assert!(stderr.contains("retread: forced backtrace:"), "{stderr}");
        assert!(stderr.contains("panic_hook_child"), "{stderr}");
    }
}
