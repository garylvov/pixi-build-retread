#[test]
fn backend_startup_reports_package_version() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_pixi-build-retread"))
        .env("PIXI_BUILD_RETREAD_LOG", "info")
        .output()
        .expect("the backend binary must launch");

    assert!(
        output.status.success(),
        "backend exited with {}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("pixi-build-retread starting"), "{stderr}");
    assert!(stderr.contains(env!("CARGO_PKG_VERSION")), "{stderr}");
}
