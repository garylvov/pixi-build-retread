//! glibc / manylinux runtime contract support for the courier installer.
//!
//! This module deliberately shells out for both host glibc detection and ELF
//! inspection. The retread binary is static musl, and NVIDIA's vendored ELFs
//! have already exposed parser edge cases in Rust ELF crates; GNU/LLVM readelf
//! is the durable boundary here.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use rattler_conda_types::Platform;
use serde::{Deserialize, Serialize};

use crate::lock::RetreadLock;

static HOST_GLIBC: OnceLock<Option<(u32, u32)>> = OnceLock::new();
static READELF: OnceLock<Option<PathBuf>> = OnceLock::new();

pub(crate) const DEFAULT_SHADOW_LIBS: &[(&str, &str)] = &[
    (
        "isaacsim/kit/kernel/plugins/libpython3.12.so.1.0",
        "conda-lib",
    ),
    ("isaacsim/kit/kernel/plugins/libpython3.12.so", "conda-lib"),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ElfLibInfo {
    pub path: PathBuf,
    pub soname: Option<String>,
    pub max_glibc_need: Option<(u32, u32)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PayloadLib {
    pub rel_path: String,
    pub abs_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RelaxDecision {
    Relax { target: (u32, u32) },
    NotNeeded,
    Undeclared,
    HostUnknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeclaredGlibc {
    pub version: (u32, u32),
    pub source: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct InstalledMarkerAudit {
    pub schema: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_glibc: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relaxed_platform: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declaration_source: Option<String>,
    pub audit: AuditStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fixups: Vec<FixupRecord>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub offenders: Vec<OffenderRecord>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub file_cache: Vec<FileCacheRecord>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum AuditStatus {
    Passed,
    PassedWithWarnings,
    SkippedNoReadelf,
    SkippedByEnv,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct FixupRecord {
    pub path: String,
    pub soname: String,
    pub provider: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct OffenderRecord {
    pub path: String,
    pub needs: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soname: Option<String>,
    pub state: OffenderState,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum OffenderState {
    Shadowed,
    Unshadowed,
    Unreadable,
}

/// Serialized as the compact JSON array requested by the marker spec:
/// `[relative/path.so, size, mtime_ns, "2.34"|null]`.
pub(crate) type FileCacheRecord = (String, u64, u128, Option<String>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RelaxOutcome {
    pub platform: String,
    pub declared: (u32, u32),
    pub declaration_source: &'static str,
}

#[derive(Debug, Clone)]
struct CandidateLib {
    rel_path: String,
    scan_path: PathBuf,
    cache_key: Option<(u64, u128)>,
}

#[derive(Debug, Default)]
struct ReadelfScan {
    infos: BTreeMap<PathBuf, ElfLibInfo>,
    unreadable: BTreeSet<PathBuf>,
}

pub(crate) fn parse_glibc_version(s: &str) -> Option<(u32, u32)> {
    let line = s.lines().next().unwrap_or("");
    for tok in line.split(|c: char| !(c.is_ascii_digit() || c == '.')) {
        if let Some((maj, rest)) = tok.split_once('.') {
            let min: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if min.is_empty() {
                continue;
            }
            if let (Ok(maj), Ok(min)) = (maj.parse::<u32>(), min.parse::<u32>()) {
                return Some((maj, min));
            }
        }
    }
    None
}

pub(crate) fn format_glibc(v: (u32, u32)) -> String {
    format!("{}.{}", v.0, v.1)
}

pub(crate) fn host_glibc() -> Option<(u32, u32)> {
    *HOST_GLIBC.get_or_init(detect_host_glibc)
}

/// Declared glibc floor without a lock in hand: `$RETREAD_DECLARED_GLIBC`
/// first, then the workspace manifest walk (both the pre-0.71
/// `[system-requirements] libc = "X.Y"` form and the 0.71+ rich
/// `platforms = [{ platform = ..., glibc = "X.Y" }]` form, via
/// `WorkspaceManifest::declared_glibc`).
pub(crate) fn declared_glibc_no_lock() -> Option<(u32, u32)> {
    declared_glibc_no_lock_for_target(current_pixi_platform())
}

/// Exact-target form of [`declared_glibc_no_lock`]. Resolution paths use this
/// when selecting artifacts for a platform other than the running host.
pub(crate) fn declared_glibc_no_lock_for_target(target_subdir: &str) -> Option<(u32, u32)> {
    if let Ok(value) = std::env::var("RETREAD_DECLARED_GLIBC")
        && let Some(version) = parse_glibc_version(&value)
    {
        return Some(version);
    }
    resolve_workspace_declared_glibc_for_target(target_subdir)
}

/// The manylinux ceiling for wheel-tag selection on a linux target:
/// `manylinux_X_Y` wheels are acceptable when `(X, Y)` is at or below
/// this. A declaration for the requested target is authoritative on aarch64
/// and on foreign targets. Native linux-64 deliberately retains retread's
/// established `max(declared, host)` behavior. Host detection otherwise only
/// falls back for an undeclared native target and must never leak into a
/// foreign target. `None` means no target-specific ceiling is known.
pub(crate) fn manylinux_ceiling(conda_subdir: &str) -> Option<(u32, u32)> {
    target_glibc_ceiling(
        conda_subdir,
        current_pixi_platform(),
        declared_glibc_no_lock_for_target(conda_subdir),
        host_glibc(),
    )
}

/// Backward-compatible native linux-64 combiner.
pub(crate) fn combine_glibc_ceiling(
    declared: Option<(u32, u32)>,
    host: Option<(u32, u32)>,
) -> Option<(u32, u32)> {
    match (declared, host) {
        (Some(d), Some(h)) => Some(d.max(h)),
        (Some(d), None) => Some(d),
        (None, Some(h)) => Some(h),
        (None, None) => None,
    }
}

/// Pure target resolver for [`manylinux_ceiling`].
///
/// A rich-platform declaration describes the deployment target, not the
/// machine running retread. Native linux-64 retains the established maximum
/// of declaration and host for compatibility. Other targets use the exact
/// declaration, falling back to host glibc only when the target is native.
pub(crate) fn target_glibc_ceiling(
    target_subdir: &str,
    native_subdir: &str,
    declared: Option<(u32, u32)>,
    host: Option<(u32, u32)>,
) -> Option<(u32, u32)> {
    if !target_subdir.starts_with("linux-") {
        return None;
    }
    if target_subdir == "linux-64" && target_subdir == native_subdir {
        return combine_glibc_ceiling(declared, host);
    }
    declared.or_else(|| (target_subdir == native_subdir).then_some(host).flatten())
}

fn detect_host_glibc() -> Option<(u32, u32)> {
    if let Ok(override_value) = std::env::var("RETREAD_HOST_GLIBC") {
        return parse_glibc_version(&override_value);
    }
    for (prog, arg) in [
        ("/usr/bin/getconf", "GNU_LIBC_VERSION"),
        ("getconf", "GNU_LIBC_VERSION"),
        ("/usr/bin/ldd", "--version"),
        ("ldd", "--version"),
    ] {
        if let Ok(out) = Command::new(prog).arg(arg).output() {
            let text = if out.stdout.is_empty() {
                String::from_utf8_lossy(&out.stderr).into_owned()
            } else {
                String::from_utf8_lossy(&out.stdout).into_owned()
            };
            if let Some(v) = parse_glibc_version(&text) {
                return Some(v);
            }
        }
    }
    None
}

pub(crate) fn relax_decision(
    declared: Option<(u32, u32)>,
    host: Option<(u32, u32)>,
) -> RelaxDecision {
    let Some(host) = host else {
        return RelaxDecision::HostUnknown;
    };
    let Some(declared) = declared else {
        return RelaxDecision::Undeclared;
    };
    if declared <= host {
        RelaxDecision::NotNeeded
    } else {
        RelaxDecision::Relax { target: declared }
    }
}

pub(crate) fn parse_readelf_dynamic(text: &str) -> BTreeMap<PathBuf, Option<String>> {
    let mut out: BTreeMap<PathBuf, Option<String>> = BTreeMap::new();
    let mut current: Option<PathBuf> = None;
    for line in text.lines() {
        if let Some(path) = line.trim().strip_prefix("File:") {
            let path = PathBuf::from(path.trim());
            out.entry(path.clone()).or_insert(None);
            current = Some(path);
            continue;
        }
        if line.contains("(SONAME)")
            && let Some(start) = line.find('[')
            && let Some(end) = line[start + 1..].find(']')
        {
            let soname = line[start + 1..start + 1 + end].to_string();
            if let Some(path) = current.as_ref() {
                out.insert(path.clone(), Some(soname));
            }
        }
    }
    out
}

pub(crate) fn parse_readelf_version_needs(text: &str) -> BTreeMap<PathBuf, Option<(u32, u32)>> {
    let mut out: BTreeMap<PathBuf, Option<(u32, u32)>> = BTreeMap::new();
    let mut current: Option<PathBuf> = None;
    for line in text.lines() {
        if let Some(path) = line.trim().strip_prefix("File:") {
            let path = PathBuf::from(path.trim());
            out.entry(path.clone()).or_insert(None);
            current = Some(path);
            continue;
        }
        let Some(path) = current.as_ref() else {
            continue;
        };
        let mut rest = line;
        while let Some(idx) = rest.find("Name: GLIBC_") {
            rest = &rest[idx + "Name: GLIBC_".len()..];
            let token: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.')
                .collect();
            if let Some(v) = parse_glibc_suffix(&token) {
                let entry = out.entry(path.clone()).or_insert(None);
                if entry.is_none_or(|old| v > old) {
                    *entry = Some(v);
                }
            }
        }
    }
    out
}

fn parse_glibc_suffix(s: &str) -> Option<(u32, u32)> {
    let (maj, min) = s.split_once('.')?;
    let maj = maj.parse::<u32>().ok()?;
    let min: String = min.chars().take_while(|c| c.is_ascii_digit()).collect();
    if min.is_empty() {
        return None;
    }
    Some((maj, min.parse::<u32>().ok()?))
}

fn parse_single_dynamic(text: &str) -> Option<Option<String>> {
    let saw_dynamic = text.contains("Dynamic section");
    for line in text.lines() {
        if line.contains("(SONAME)")
            && let Some(start) = line.find('[')
            && let Some(end) = line[start + 1..].find(']')
        {
            return Some(Some(line[start + 1..start + 1 + end].to_string()));
        }
    }
    saw_dynamic.then_some(None)
}

fn parse_single_version_need(text: &str) -> Option<Option<(u32, u32)>> {
    let saw_version = text.contains("Version needs section") || text.contains("Name:");
    let mut max_need: Option<(u32, u32)> = None;
    for line in text.lines() {
        let mut rest = line;
        while let Some(idx) = rest.find("Name: GLIBC_") {
            rest = &rest[idx + "Name: GLIBC_".len()..];
            let token: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.')
                .collect();
            if let Some(v) = parse_glibc_suffix(&token)
                && max_need.is_none_or(|old| v > old)
            {
                max_need = Some(v);
            }
        }
    }
    saw_version.then_some(max_need)
}

pub(crate) fn extract_manylinux_floor(text: &str) -> Option<(u32, u32)> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"manylinux_(\d+)_(\d+)").unwrap());
    re.captures_iter(text)
        .filter_map(|cap| {
            let maj = cap.get(1)?.as_str().parse::<u32>().ok()?;
            let min = cap.get(2)?.as_str().parse::<u32>().ok()?;
            Some((maj, min))
        })
        .max()
}

pub(crate) fn undeclared_glibc_error(
    host: Option<(u32, u32)>,
    floor: Option<(u32, u32)>,
) -> String {
    undeclared_glibc_error_for_target(host, floor, current_pixi_platform())
}

pub(crate) fn undeclared_glibc_error_for_target(
    host: Option<(u32, u32)>,
    floor: Option<(u32, u32)>,
    target_subdir: &str,
) -> String {
    let host = host
        .map(format_glibc)
        .unwrap_or_else(|| "unknown".to_string());
    let floor = floor.map(format_glibc).unwrap_or_else(|| "x.y".to_string());
    format!(
        "retread: uv rejected the only available wheels for their manylinux platform tag\n\
         (wheel floor above this host's glibc {host}). retread will only relax uv's host\n\
         gate up to a glibc version YOUR WORKSPACE DECLARES it can handle. Declare it:\n\n\
           [system-requirements]\n\
           libc = \"{floor}\"        # pixi < 0.71\n\n\
           # or (pixi >= 0.71):\n\
           [workspace]\n\
           platforms = [{{ platform = \"{}\", glibc = \"{floor}\" }}]\n\n\
         then re-run. This declaration is load-bearing: retread verifies at install time\n\
         (GLIBC symbol audit) that every wheel library needing more than glibc {host} is\n\
         shadowed by a conda-provided library, and fails the install if not.",
        target_subdir,
    )
}

pub(crate) fn emit_glibc_relax_warning(
    host: (u32, u32),
    declared: (u32, u32),
    source: &str,
    target: &str,
) {
    let bar = "!".repeat(78);
    let host_s = format_glibc(host);
    let declared_s = format_glibc(declared);
    eprintln!("\n{bar}");
    eprintln!("!!  retread WARNING: RELAXING glibc / manylinux PLATFORM TAG");
    eprintln!("!!  uv rejected a wheel for its manylinux tag; retrying relaxed.");
    eprintln!("!!");
    eprintln!("!!  Host glibc detected: {host_s}");
    eprintln!("!!  Declared glibc floor ({source}): {declared_s}");
    eprintln!("!!  Retrying with uv --python-platform {target}");
    eprintln!("!!");
    eprintln!(
        "!!  Safety is enforced by retread's post-install GLIBC symbol audit; \
         the install will fail in enforce mode if any wheel library needs more \
         than host glibc {host_s} without a clean $PREFIX/lib provider."
    );
    eprintln!("{bar}\n");
    tracing::warn!(
        host_glibc = %host_s,
        declared_glibc = %declared_s,
        declaration_source = %source,
        target_platform = %target,
        "retread: relaxing uv's manylinux host gate to the declared glibc floor",
    );
}

pub(crate) fn current_pixi_platform() -> &'static str {
    Platform::current().as_str()
}

pub(crate) fn resolve_declared_glibc(lock: &RetreadLock) -> Option<DeclaredGlibc> {
    resolve_declared_glibc_for_target(lock, current_pixi_platform())
}

pub(crate) fn resolve_declared_glibc_for_target(
    lock: &RetreadLock,
    target_subdir: &str,
) -> Option<DeclaredGlibc> {
    if let Ok(value) = std::env::var("RETREAD_DECLARED_GLIBC")
        && let Some(version) = parse_glibc_version(&value)
    {
        return Some(DeclaredGlibc {
            version,
            source: "env",
        });
    }
    if let Some(version) = resolve_workspace_declared_glibc_for_target(target_subdir) {
        return Some(DeclaredGlibc {
            version,
            source: "workspace",
        });
    }
    lock_declared_glibc_for_target(lock, target_subdir)
}

fn lock_declared_glibc_for_target(
    lock: &RetreadLock,
    target_subdir: &str,
) -> Option<DeclaredGlibc> {
    // Existing locks do not identify the platform whose declaration they
    // recorded. Preserve the native fallback, but never let an unqualified
    // lock raise a foreign target's compatibility ceiling. Target-qualified
    // lock schemas can relax this guard once the recorded subdir is verified.
    if target_subdir != current_pixi_platform() {
        return None;
    }
    lock.declared_glibc
        .as_deref()
        .and_then(parse_glibc_version)
        .map(|version| DeclaredGlibc {
            version,
            source: "lock",
        })
}

pub(crate) fn resolve_workspace_declared_glibc() -> Option<(u32, u32)> {
    resolve_workspace_declared_glibc_for_target(current_pixi_platform())
}

pub(crate) fn resolve_workspace_declared_glibc_for_target(
    target_subdir: &str,
) -> Option<(u32, u32)> {
    let mut candidates = Vec::new();
    if let Ok(path) = std::env::var("PIXI_PROJECT_MANIFEST") {
        candidates.push(PathBuf::from(path));
    }
    if let Ok(root) = std::env::var("PIXI_PROJECT_ROOT") {
        append_manifest_walk(&mut candidates, Path::new(&root));
    }
    if let Ok(cwd) = std::env::current_dir() {
        append_manifest_walk(&mut candidates, &cwd);
    }

    let mut seen = BTreeSet::new();
    for path in candidates {
        if !seen.insert(path.clone()) {
            continue;
        }
        if let Some(v) = declared_glibc_from_manifest_path(&path, target_subdir) {
            return Some(v);
        }
    }
    None
}

fn append_manifest_walk(out: &mut Vec<PathBuf>, start: &Path) {
    let mut cur = Some(start);
    while let Some(dir) = cur {
        out.push(dir.join("pixi.toml"));
        out.push(dir.join("pyproject.toml"));
        cur = dir.parent();
    }
}

fn declared_glibc_from_manifest_path(path: &Path, target_subdir: &str) -> Option<(u32, u32)> {
    let text = std::fs::read_to_string(path).ok()?;
    let parsed: toml::Value = toml::from_str(&text).ok()?;
    let root = pixi_manifest_root(&parsed)?;
    let ws = crate::workspace::WorkspaceManifest::from_toml(root);
    ws.declared_glibc_for_target(target_subdir, None)
}

fn pixi_manifest_root(parsed: &toml::Value) -> Option<&toml::Value> {
    if parsed.get("workspace").is_some() || parsed.get("package").is_some() {
        return Some(parsed);
    }
    parsed
        .get("tool")
        .and_then(|v| v.as_table())
        .and_then(|t| t.get("pixi"))
}

pub(crate) fn marker_digest_matches(marker: &Path, want: &str) -> bool {
    std::fs::read_to_string(marker)
        .ok()
        .and_then(|have| have.lines().next().map(str::trim).map(str::to_string))
        .is_some_and(|have| have == want)
}

pub(crate) fn parse_marker_audit(text: &str) -> Option<InstalledMarkerAudit> {
    let json = text.lines().skip(1).find(|line| !line.trim().is_empty())?;
    serde_json::from_str(json).ok()
}

pub(crate) fn marker_audit(marker: &Path) -> Option<InstalledMarkerAudit> {
    let text = std::fs::read_to_string(marker).ok()?;
    parse_marker_audit(&text)
}

pub(crate) fn marker_body(digest: &str, audit: &InstalledMarkerAudit) -> Result<String> {
    Ok(format!("{digest}\n{}\n", serde_json::to_string(audit)?))
}

pub(crate) fn verify_marker_state(
    lock: &RetreadLock,
    prefix: &Path,
    marker_text: &str,
) -> Result<InstalledMarkerAudit> {
    let share = prefix.join("share").join("retread");
    let broken = share.join(format!("{}.broken", lock.bundle));
    if broken.exists() {
        let detail = std::fs::read_to_string(&broken).unwrap_or_default();
        bail!(
            "retread verify: bundle {} is marked broken at {}{}{}",
            lock.bundle,
            broken.display(),
            if detail.trim().is_empty() { "" } else { ": " },
            detail.trim()
        );
    }

    let audit = parse_marker_audit(marker_text)
        .ok_or_else(|| anyhow::anyhow!("retread verify: marker has no glibc audit record"))?;
    verify_audit_record(&audit, prefix)?;
    Ok(audit)
}

pub(crate) fn verify_audit_record(audit: &InstalledMarkerAudit, prefix: &Path) -> Result<()> {
    let prefix_lib = prefix.join("lib");
    for fixup in &audit.fixups {
        let path = site_packages_from_fixup_path(prefix, &fixup.path)?;
        let meta = std::fs::symlink_metadata(&path)
            .with_context(|| format!("retread verify: reading fixup path {}", path.display()))?;
        if !meta.file_type().is_symlink() {
            bail!(
                "retread verify: fixup path {} is no longer a symlink",
                path.display()
            );
        }
        let target = std::fs::read_link(&path)
            .with_context(|| format!("retread verify: reading symlink {}", path.display()))?;
        let resolved = absolutize_link_target(&path, &target);
        if !resolved.exists() {
            bail!(
                "retread verify: fixup path {} points at missing target {}",
                path.display(),
                resolved.display()
            );
        }
    }

    for offender in &audit.offenders {
        if offender.state == OffenderState::Shadowed
            && let Some(soname) = offender.soname.as_deref()
        {
            let provider = prefix_lib.join(soname);
            if !provider.exists() {
                bail!(
                    "retread verify: recorded shadow provider {} for {} is missing",
                    provider.display(),
                    offender.path
                );
            }
        }
    }

    if let Some(recorded) = audit.host_glibc.as_deref().and_then(parse_glibc_version)
        && let Some(current) = host_glibc()
        && current < recorded
    {
        bail!(
            "retread verify: host glibc {} is older than marker audit host {}",
            format_glibc(current),
            format_glibc(recorded),
        );
    }

    if matches!(
        audit.audit,
        AuditStatus::SkippedNoReadelf | AuditStatus::SkippedByEnv
    ) {
        eprintln!(
            "retread verify: GLIBC audit for this bundle was {}; install marker is accepted with warning",
            audit_status_str(audit.audit)
        );
    }

    Ok(())
}

fn site_packages_from_fixup_path(prefix: &Path, rel: &str) -> Result<PathBuf> {
    let lib = prefix.join("lib");
    let entries = std::fs::read_dir(&lib).with_context(|| format!("reading {}", lib.display()))?;
    for py in entries {
        let Ok(entry) = py else {
            continue;
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("python") {
            let candidate = entry.path().join("site-packages").join(rel);
            if candidate.exists() || candidate.symlink_metadata().is_ok() {
                return Ok(candidate);
            }
        }
    }
    bail!("retread verify: cannot locate site-packages entry {rel}")
}

pub(crate) fn install_audit(
    lock: &RetreadLock,
    prefix: &Path,
    site_packages: &Path,
    payload_libs: &[PayloadLib],
    previous: Option<&InstalledMarkerAudit>,
    relaxed_platform: Option<String>,
    declaration_source: Option<String>,
) -> Result<InstalledMarkerAudit> {
    cleanup_orphan_backups(site_packages, payload_libs, false)?;
    let fixup_result = apply_shadow_fixups(lock, prefix, site_packages, payload_libs)?;
    let mut audit = audit_payload(
        lock,
        prefix,
        payload_libs,
        previous,
        relaxed_platform,
        declaration_source,
    )?;
    audit.fixups = fixup_result.records;
    if !fixup_result.failures.is_empty() {
        for failure in &fixup_result.failures {
            eprintln!("retread: shadow-lib fixup warning: {failure}");
        }
        if relaxed_audit_enforced(audit.relaxed_platform.as_deref()) {
            bail!(
                "retread: shadow-lib fixup failed under enforced GLIBC audit:\n{}",
                fixup_result.failures.join("\n")
            );
        }
        if audit.audit == AuditStatus::Passed {
            audit.audit = AuditStatus::PassedWithWarnings;
        }
    }
    Ok(audit)
}

pub(crate) fn full_verify_audit(
    lock: &RetreadLock,
    prefix: &Path,
    site_packages: &Path,
    payload_libs: &[PayloadLib],
    previous: Option<&InstalledMarkerAudit>,
) -> Result<InstalledMarkerAudit> {
    cleanup_orphan_backups(site_packages, payload_libs, true)?;
    audit_payload(
        lock,
        prefix,
        payload_libs,
        previous,
        previous.and_then(|p| p.relaxed_platform.clone()),
        previous.and_then(|p| p.declaration_source.clone()),
    )
}

fn audit_payload(
    _lock: &RetreadLock,
    prefix: &Path,
    payload_libs: &[PayloadLib],
    previous: Option<&InstalledMarkerAudit>,
    relaxed_platform: Option<String>,
    declaration_source: Option<String>,
) -> Result<InstalledMarkerAudit> {
    let host = host_glibc();
    let mut record = InstalledMarkerAudit {
        schema: 1,
        host_glibc: host.map(format_glibc),
        relaxed_platform,
        declaration_source,
        audit: AuditStatus::Passed,
        fixups: previous.map(|p| p.fixups.clone()).unwrap_or_default(),
        offenders: Vec::new(),
        file_cache: Vec::new(),
    };

    if std::env::var("RETREAD_SKIP_GLIBC_AUDIT").is_ok() {
        record.audit = AuditStatus::SkippedByEnv;
        return Ok(record);
    }

    let Some(host) = host else {
        eprintln!("retread: host glibc is undetectable; GLIBC symbol audit skipped with warning");
        record.audit = AuditStatus::PassedWithWarnings;
        return Ok(record);
    };

    let Some(_readelf) = readelf(prefix) else {
        record.audit = AuditStatus::SkippedNoReadelf;
        let message = "retread: installed wheels may be above the host glibc floor but cannot audit symbol needs: no readelf found. Add `binutils` to the pack's conda run-deps, or set RETREAD_SKIP_GLIBC_AUDIT=1 to accept the risk.";
        if relaxed_audit_enforced(record.relaxed_platform.as_deref()) {
            bail!("{message}");
        }
        eprintln!("{message}");
        return Ok(record);
    };

    let previous_cache = previous.map(cache_map).unwrap_or_default();
    let candidates = audit_candidates(prefix, payload_libs);
    let mut cached_needs: BTreeMap<String, Option<(u32, u32)>> = BTreeMap::new();
    let mut changed: Vec<PathBuf> = Vec::new();
    let mut scan_by_path: BTreeMap<PathBuf, String> = BTreeMap::new();

    for candidate in &candidates {
        let Some((size, mtime_ns)) = candidate.cache_key else {
            record.offenders.push(OffenderRecord {
                path: candidate.rel_path.clone(),
                needs: "unreadable".to_string(),
                soname: None,
                state: OffenderState::Unreadable,
            });
            record.audit = AuditStatus::PassedWithWarnings;
            continue;
        };
        if let Some((old_size, old_mtime_ns, old_need)) = previous_cache.get(&candidate.rel_path)
            && *old_size == size
            && *old_mtime_ns == mtime_ns
        {
            cached_needs.insert(candidate.rel_path.clone(), *old_need);
        } else {
            changed.push(candidate.scan_path.clone());
            scan_by_path.insert(candidate.scan_path.clone(), candidate.rel_path.clone());
        }
    }

    let scan = inspect_elf_files(prefix, &changed);
    let mut infos_by_rel: BTreeMap<String, ElfLibInfo> = BTreeMap::new();
    for (path, info) in scan.infos {
        let rel = scan_by_path
            .get(&path)
            .cloned()
            .unwrap_or_else(|| path.to_string_lossy().into_owned());
        infos_by_rel.insert(
            rel,
            ElfLibInfo {
                path: info.path,
                soname: info.soname,
                max_glibc_need: info.max_glibc_need,
            },
        );
    }
    for path in scan.unreadable {
        let rel = scan_by_path
            .get(&path)
            .cloned()
            .unwrap_or_else(|| path.to_string_lossy().into_owned());
        record.offenders.push(OffenderRecord {
            path: rel,
            needs: "unreadable".to_string(),
            soname: None,
            state: OffenderState::Unreadable,
        });
        record.audit = AuditStatus::PassedWithWarnings;
    }

    let prefix_lib = prefix.join("lib");
    for candidate in candidates {
        let Some((size, mtime_ns)) = candidate.cache_key else {
            continue;
        };
        let max_need = if let Some(need) = cached_needs.get(&candidate.rel_path) {
            *need
        } else {
            infos_by_rel
                .get(&candidate.rel_path)
                .and_then(|info| info.max_glibc_need)
        };
        record.file_cache.push((
            candidate.rel_path.clone(),
            size,
            mtime_ns,
            max_need.map(format_glibc),
        ));
        let Some(need) = max_need else {
            continue;
        };
        if need <= host {
            continue;
        }

        let soname = infos_by_rel
            .get(&candidate.rel_path)
            .and_then(|info| info.soname.clone())
            .or_else(|| soname_for_one(prefix, &candidate.scan_path))
            .or_else(|| basename_string(&candidate.scan_path));
        let state = if let Some(soname) = soname.as_deref() {
            let provider = prefix_lib.join(soname);
            if provider.exists()
                && max_glibc_need_for_one(prefix, &provider).is_none_or(|pneed| pneed <= host)
            {
                OffenderState::Shadowed
            } else {
                OffenderState::Unshadowed
            }
        } else {
            OffenderState::Unshadowed
        };
        if state == OffenderState::Unshadowed {
            record.audit = AuditStatus::PassedWithWarnings;
        }
        record.offenders.push(OffenderRecord {
            path: candidate.rel_path,
            needs: format_glibc(need),
            soname,
            state,
        });
    }
    record.file_cache.sort_by(|a, b| a.0.cmp(&b.0));

    let unshadowed: Vec<&OffenderRecord> = record
        .offenders
        .iter()
        .filter(|o| o.state == OffenderState::Unshadowed)
        .collect();
    if !unshadowed.is_empty() && relaxed_audit_enforced(record.relaxed_platform.as_deref()) {
        let mut lines = Vec::new();
        for offender in unshadowed {
            let soname = offender.soname.as_deref().unwrap_or("<none>");
            let provider = if soname == "<none>" {
                "<none>".to_string()
            } else {
                prefix_lib.join(soname).display().to_string()
            };
            lines.push(format!(
                "{} needs GLIBC_{} (host {}), SONAME {}, no clean provider at {}",
                offender.path,
                offender.needs,
                format_glibc(host),
                soname,
                provider,
            ));
        }
        bail!(
            "retread: GLIBC symbol audit failed after manylinux relax:\n{}\n\
             Add a [tool.retread.shadow-libs] entry if a provider exists under another name, \
             or stop declaring libc >= {} for wheels this host cannot honor.",
            lines.join("\n"),
            record
                .relaxed_platform
                .as_deref()
                .unwrap_or("<unknown relaxed platform>")
        );
    }
    if !record.offenders.is_empty() && record.audit == AuditStatus::Passed {
        record.audit = AuditStatus::PassedWithWarnings;
    }
    Ok(record)
}

fn relaxed_audit_enforced(relaxed_platform: Option<&str>) -> bool {
    relaxed_platform.is_some()
        && std::env::var("RETREAD_GLIBC_AUDIT")
            .map(|v| v.eq_ignore_ascii_case("enforce"))
            .unwrap_or(false)
}

fn audit_status_str(status: AuditStatus) -> &'static str {
    match status {
        AuditStatus::Passed => "passed",
        AuditStatus::PassedWithWarnings => "passed-with-warnings",
        AuditStatus::SkippedNoReadelf => "skipped-no-readelf",
        AuditStatus::SkippedByEnv => "skipped-by-env",
    }
}

/// Marker-audit file cache keyed by path: (size, mtime ns, parsed max `GLIBC_x.y` need).
type ParsedFileCache = BTreeMap<String, (u64, u128, Option<(u32, u32)>)>;

fn cache_map(audit: &InstalledMarkerAudit) -> ParsedFileCache {
    audit
        .file_cache
        .iter()
        .map(|(path, size, mtime_ns, need)| {
            (
                path.clone(),
                (
                    *size,
                    *mtime_ns,
                    need.as_deref().and_then(parse_glibc_version),
                ),
            )
        })
        .collect()
}

fn audit_candidates(prefix: &Path, payload_libs: &[PayloadLib]) -> Vec<CandidateLib> {
    let mut out = Vec::new();
    let prefix_lib = prefix.join("lib");
    for lib in payload_libs {
        let scan_path = if let Ok(target) = std::fs::read_link(&lib.abs_path) {
            let resolved = absolutize_link_target(&lib.abs_path, &target);
            if path_resolves_under(&resolved, &prefix_lib) {
                resolved
            } else {
                lib.abs_path.clone()
            }
        } else {
            lib.abs_path.clone()
        };
        let cache_key = file_cache_key(&scan_path);
        out.push(CandidateLib {
            rel_path: lib.rel_path.clone(),
            scan_path,
            cache_key,
        });
    }
    out
}

fn file_cache_key(path: &Path) -> Option<(u64, u128)> {
    let meta = std::fs::metadata(path).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        let secs = meta.mtime() as i128;
        let nanos = meta.mtime_nsec() as i128;
        let mtime_ns = secs.saturating_mul(1_000_000_000).saturating_add(nanos);
        Some((meta.len(), mtime_ns.max(0) as u128))
    }
    #[cfg(not(unix))]
    {
        let mtime_ns = meta
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_nanos();
        Some((meta.len(), mtime_ns))
    }
}

struct FixupResult {
    records: Vec<FixupRecord>,
    failures: Vec<String>,
}

fn apply_shadow_fixups(
    lock: &RetreadLock,
    prefix: &Path,
    site_packages: &Path,
    payload_libs: &[PayloadLib],
) -> Result<FixupResult> {
    let shadow_libs = effective_shadow_libs(lock);
    let mut result = FixupResult {
        records: Vec::new(),
        failures: Vec::new(),
    };
    if shadow_libs.is_empty() {
        return Ok(result);
    }
    let prefix_lib = prefix.join("lib");
    for lib in payload_libs {
        let Some((_pattern, policy)) = shadow_libs
            .iter()
            .find(|(pattern, _)| path_pattern_matches(pattern, &lib.rel_path))
        else {
            continue;
        };
        if policy != "conda-lib" {
            result.failures.push(format!(
                "{} has unsupported shadow-lib policy {}",
                lib.rel_path, policy
            ));
            continue;
        }
        let soname = soname_for_one(prefix, &lib.abs_path)
            .or_else(|| basename_string(&lib.abs_path))
            .unwrap_or_else(|| lib.rel_path.clone());
        let provider = prefix_lib.join(&soname);
        if !provider.exists() {
            result.failures.push(format!(
                "{} declares conda-lib shadow for SONAME {}, but provider {} is missing",
                lib.rel_path,
                soname,
                provider.display()
            ));
            continue;
        }
        if let Some(host) = host_glibc()
            && let Some(need) = max_glibc_need_for_one(prefix, &provider)
            && need > host
        {
            result.failures.push(format!(
                "{} provider {} needs GLIBC_{} above host {}",
                lib.rel_path,
                provider.display(),
                format_glibc(need),
                format_glibc(host)
            ));
            continue;
        }
        match replace_with_provider_symlink(site_packages, &lib.abs_path, &provider) {
            Ok(()) => {
                result.records.push(FixupRecord {
                    path: lib.rel_path.clone(),
                    soname,
                    provider: path_relative_to_prefix(prefix, &provider),
                });
            }
            Err(err) => {
                result
                    .failures
                    .push(format!("{} shadow fixup failed: {err:#}", lib.rel_path));
            }
        }
    }
    Ok(result)
}

pub(crate) fn effective_shadow_libs(lock: &RetreadLock) -> BTreeMap<String, String> {
    if !lock.shadow_libs.is_empty() {
        return lock.shadow_libs.clone();
    }
    if lock.wheels.iter().any(|w| {
        matches!(
            normalize_dist_name(&w.name).as_str(),
            "isaacsim" | "isaacsim-kernel"
        )
    }) {
        DEFAULT_SHADOW_LIBS
            .iter()
            .map(|(path, policy)| ((*path).to_string(), (*policy).to_string()))
            .collect()
    } else {
        BTreeMap::new()
    }
}

fn normalize_dist_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_sep = false;
    for c in name.trim().chars() {
        if c == '-' || c == '_' || c == '.' {
            if !prev_sep && !out.is_empty() {
                out.push('-');
            }
            prev_sep = true;
        } else {
            out.push(c.to_ascii_lowercase());
            prev_sep = false;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

fn replace_with_provider_symlink(
    _site_packages: &Path,
    vendored: &Path,
    provider: &Path,
) -> Result<()> {
    let parent = vendored
        .parent()
        .ok_or_else(|| anyhow::anyhow!("{} has no parent directory", vendored.display()))?;
    std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    if let Ok(target) = std::fs::read_link(vendored) {
        let resolved = absolutize_link_target(vendored, &target);
        if path_resolves_same(&resolved, provider) {
            return Ok(());
        }
        if provider
            .parent()
            .is_some_and(|provider_dir| path_resolves_under(&resolved, provider_dir))
        {
            std::fs::remove_file(vendored)
                .with_context(|| format!("removing stale symlink {}", vendored.display()))?;
        } else {
            bail!(
                "{} is a symlink outside the retread provider boundary ({})",
                vendored.display(),
                resolved.display()
            );
        }
    } else if vendored.exists() {
        let backup = backup_path(vendored);
        if backup.exists() {
            std::fs::remove_file(&backup)
                .with_context(|| format!("removing stale backup {}", backup.display()))?;
        }
        std::fs::rename(vendored, &backup).with_context(|| {
            format!(
                "backing up current vendored library {} -> {}",
                vendored.display(),
                backup.display()
            )
        })?;
    }

    let rel = relative_path(parent, provider);
    #[cfg(unix)]
    std::os::unix::fs::symlink(&rel, vendored).with_context(|| {
        format!(
            "creating relative symlink {} -> {}",
            vendored.display(),
            rel.display()
        )
    })?;
    #[cfg(not(unix))]
    std::fs::hard_link(provider, vendored).with_context(|| {
        format!(
            "linking provider {} -> {}",
            provider.display(),
            vendored.display()
        )
    })?;
    Ok(())
}

fn backup_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".retread-orig");
    path.with_file_name(name)
}

fn cleanup_orphan_backups(
    site_packages: &Path,
    payload_libs: &[PayloadLib],
    report_only: bool,
) -> Result<()> {
    let payload: BTreeSet<PathBuf> = payload_libs.iter().map(|l| l.abs_path.clone()).collect();
    let mut stack = vec![site_packages.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.ends_with(".retread-orig") {
                continue;
            }
            let sibling_name = name.trim_end_matches(".retread-orig");
            let sibling = entry.path().with_file_name(sibling_name);
            if !sibling.exists() || !payload.contains(&sibling) {
                if report_only {
                    eprintln!(
                        "retread verify --full: orphan shadow-lib backup {}",
                        path.display()
                    );
                } else {
                    std::fs::remove_file(&path)
                        .with_context(|| format!("removing orphan backup {}", path.display()))?;
                }
            }
        }
    }
    Ok(())
}

fn soname_for_one(prefix: &Path, path: &Path) -> Option<String> {
    let mut scan = inspect_elf_files(prefix, &[path.to_path_buf()]);
    scan.infos.remove(path).and_then(|info| info.soname)
}

fn max_glibc_need_for_one(prefix: &Path, path: &Path) -> Option<(u32, u32)> {
    let mut scan = inspect_elf_files(prefix, &[path.to_path_buf()]);
    scan.infos.remove(path).and_then(|info| info.max_glibc_need)
}

fn inspect_elf_files(prefix: &Path, files: &[PathBuf]) -> ReadelfScan {
    let mut scan = ReadelfScan::default();
    if files.is_empty() {
        return scan;
    }
    let Some(readelf) = readelf(prefix) else {
        for file in files {
            scan.unreadable.insert(file.clone());
        }
        return scan;
    };
    for chunk in files.chunks(50) {
        let dyn_text = run_readelf(&readelf, "-d", chunk);
        let ver_text = run_readelf(&readelf, "-V", chunk);
        let mut dyn_map = dyn_text
            .as_deref()
            .map(parse_readelf_dynamic)
            .unwrap_or_default();
        let mut ver_map = ver_text
            .as_deref()
            .map(parse_readelf_version_needs)
            .unwrap_or_default();
        if chunk.len() == 1 {
            let only = chunk[0].clone();
            if !dyn_map.contains_key(&only)
                && let Some(parsed) = dyn_text.as_deref().and_then(parse_single_dynamic)
            {
                dyn_map.insert(only.clone(), parsed);
            }
            if !ver_map.contains_key(&only)
                && let Some(parsed) = ver_text.as_deref().and_then(parse_single_version_need)
            {
                ver_map.insert(only, parsed);
            }
        }
        for file in chunk {
            if !dyn_map.contains_key(file) && !ver_map.contains_key(file) {
                scan.unreadable.insert(file.clone());
                continue;
            }
            scan.infos.insert(
                file.clone(),
                ElfLibInfo {
                    path: file.clone(),
                    soname: dyn_map.get(file).cloned().flatten(),
                    max_glibc_need: ver_map.get(file).cloned().flatten(),
                },
            );
        }
    }
    scan
}

fn run_readelf(readelf: &Path, flag: &str, files: &[PathBuf]) -> Option<String> {
    let out = Command::new(readelf).arg(flag).args(files).output().ok()?;
    let mut text = String::new();
    text.push_str(&String::from_utf8_lossy(&out.stdout));
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    Some(text)
}

fn readelf(prefix: &Path) -> Option<PathBuf> {
    READELF.get_or_init(|| resolve_readelf(prefix)).clone()
}

fn resolve_readelf(prefix: &Path) -> Option<PathBuf> {
    let bin = prefix.join("bin");
    let direct = bin.join("readelf");
    if command_answers_version(&direct) {
        return Some(direct);
    }
    if let Ok(entries) = std::fs::read_dir(&bin) {
        let mut prefixed: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with("-readelf"))
            })
            .collect();
        prefixed.sort();
        for path in prefixed {
            if command_answers_version(&path) {
                return Some(path);
            }
        }
    }
    for name in ["readelf", "llvm-readelf"] {
        let path = PathBuf::from(name);
        if command_answers_version(&path) {
            return Some(path);
        }
    }
    None
}

fn command_answers_version(path: &Path) -> bool {
    Command::new(path)
        .arg("--version")
        .output()
        .is_ok_and(|out| out.status.success())
}

fn path_pattern_matches(pattern: &str, path: &str) -> bool {
    let pattern_parts: Vec<&str> = pattern.split('/').collect();
    let path_parts: Vec<&str> = path.split('/').collect();
    match_segments(&pattern_parts, &path_parts)
}

fn match_segments(pattern: &[&str], path: &[&str]) -> bool {
    if pattern.is_empty() {
        return path.is_empty();
    }
    if pattern[0] == "**" {
        return match_segments(&pattern[1..], path)
            || (!path.is_empty() && match_segments(pattern, &path[1..]));
    }
    !path.is_empty()
        && match_component(pattern[0], path[0])
        && match_segments(&pattern[1..], &path[1..])
}

fn match_component(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let mut rest = text;
    if !parts[0].is_empty() {
        let Some(stripped) = rest.strip_prefix(parts[0]) else {
            return false;
        };
        rest = stripped;
    }
    for part in &parts[1..parts.len() - 1] {
        if part.is_empty() {
            continue;
        }
        let Some(idx) = rest.find(part) else {
            return false;
        };
        rest = &rest[idx + part.len()..];
    }
    let last = parts.last().copied().unwrap_or("");
    last.is_empty() || rest.ends_with(last)
}

fn basename_string(path: &Path) -> Option<String> {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(str::to_string)
}

fn path_relative_to_prefix(prefix: &Path, path: &Path) -> String {
    path.strip_prefix(prefix)
        .ok()
        .map(path_to_slash_string)
        .unwrap_or_else(|| path.display().to_string())
}

fn path_to_slash_string(path: &Path) -> String {
    path.components()
        .filter_map(|c| match c {
            Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn absolutize_link_target(link_path: &Path, target: &Path) -> PathBuf {
    if target.is_absolute() {
        target.to_path_buf()
    } else {
        link_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(target)
    }
}

fn path_resolves_same(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

fn path_resolves_under(path: &Path, parent: &Path) -> bool {
    let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let parent = std::fs::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf());
    path.starts_with(parent)
}

fn relative_path(from_dir: &Path, to: &Path) -> PathBuf {
    let from_parts = normal_components(from_dir);
    let to_parts = normal_components(to);
    let common = from_parts
        .iter()
        .zip(to_parts.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let mut out = PathBuf::new();
    for _ in common..from_parts.len() {
        out.push("..");
    }
    for part in &to_parts[common..] {
        out.push(part);
    }
    if out.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        out
    }
}

fn normal_components(path: &Path) -> Vec<OsString> {
    path.components()
        .filter_map(|c| match c {
            Component::Normal(s) => Some(s.to_os_string()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::lock::{LockWheel, Origin};

    #[test]
    fn parses_glibc_version_banners() {
        assert_eq!(parse_glibc_version("glibc 2.34\n"), Some((2, 34)));
        assert_eq!(
            parse_glibc_version("ldd (GNU libc) 2.34\nCopyright ..."),
            Some((2, 34))
        );
        assert_eq!(parse_glibc_version("glibc 2.34.9000"), Some((2, 34)));
        assert_eq!(parse_glibc_version("no version here"), None);
    }

    #[test]
    fn relax_decision_matrix() {
        assert_eq!(
            relax_decision(None, Some((2, 34))),
            RelaxDecision::Undeclared
        );
        assert_eq!(
            relax_decision(Some((2, 35)), Some((2, 34))),
            RelaxDecision::Relax { target: (2, 35) }
        );
        assert_eq!(
            relax_decision(Some((2, 39)), Some((2, 34))),
            RelaxDecision::Relax { target: (2, 39) }
        );
        assert_eq!(
            relax_decision(Some((2, 34)), Some((2, 34))),
            RelaxDecision::NotNeeded
        );
        assert_eq!(
            relax_decision(Some((2, 28)), Some((2, 34))),
            RelaxDecision::NotNeeded
        );
        assert_eq!(
            relax_decision(Some((3, 1)), Some((2, 34))),
            RelaxDecision::Relax { target: (3, 1) }
        );
        assert_eq!(
            relax_decision(Some((2, 35)), None),
            RelaxDecision::HostUnknown
        );
        assert_eq!(relax_decision(None, None), RelaxDecision::HostUnknown);
    }

    #[test]
    fn target_glibc_ceiling_never_leaks_host_into_foreign_resolution() {
        assert_eq!(
            target_glibc_ceiling("linux-aarch64", "linux-64", Some((2, 35)), Some((2, 39))),
            Some((2, 35)),
            "the exact Orin deployment declaration is authoritative"
        );
        assert_eq!(
            target_glibc_ceiling("linux-64", "linux-aarch64", Some((2, 35)), Some((2, 39))),
            Some((2, 35)),
            "the exact x86 deployment declaration is authoritative"
        );
        assert_eq!(
            target_glibc_ceiling("linux-aarch64", "linux-64", None, Some((2, 39))),
            None,
            "an x86 host says nothing about an undeclared aarch64 target"
        );
        assert_eq!(
            target_glibc_ceiling("linux-64", "linux-aarch64", None, Some((2, 39))),
            None,
            "an ARM host says nothing about an undeclared x86 target"
        );
        assert_eq!(
            target_glibc_ceiling("linux-aarch64", "linux-aarch64", None, Some((2, 35))),
            Some((2, 35)),
            "host glibc remains a safe fallback for an undeclared native target"
        );
        assert_eq!(
            target_glibc_ceiling("linux-64", "linux-64", Some((2, 35)), Some((2, 39))),
            Some((2, 39)),
            "native x86 retains max(declared, host) compatibility"
        );
        assert_eq!(
            target_glibc_ceiling("osx-arm64", "osx-arm64", Some((2, 39)), Some((2, 39))),
            None,
            "glibc ceilings never apply to non-Linux targets"
        );
    }

    #[test]
    fn unqualified_lock_glibc_never_applies_to_foreign_target() {
        let mut lock = lock_with_shadow("x.so");
        lock.declared_glibc = Some("2.39".to_string());
        let native = current_pixi_platform();
        let foreign = if native == "linux-aarch64" {
            "linux-64"
        } else {
            "linux-aarch64"
        };

        assert!(lock_declared_glibc_for_target(&lock, foreign).is_none());
        let native_declared = lock_declared_glibc_for_target(&lock, native).unwrap();
        assert_eq!(native_declared.version, (2, 39));
        assert_eq!(native_declared.source, "lock");
    }

    #[test]
    fn readelf_dynamic_fixture_parses_sonames() {
        let text = include_str!("../tests/fixtures/readelf/dynamic.soname.txt");
        let parsed = parse_readelf_dynamic(text);
        assert_eq!(
            parsed
                .get(&PathBuf::from(
                    "/env/site/isaacsim/kit/kernel/plugins/libpython3.12.so.1.0"
                ))
                .cloned()
                .flatten()
                .as_deref(),
            Some("libpython3.12.so.1.0")
        );
        assert_eq!(
            parsed
                .get(&PathBuf::from("/env/site/pkg/no-soname.so"))
                .cloned()
                .flatten(),
            None
        );
    }

    #[test]
    fn readelf_version_fixtures_parse_glibc_max() {
        let vendored = include_str!("../tests/fixtures/readelf/libpython_vendored.verneed.txt");
        let conda = include_str!("../tests/fixtures/readelf/libpython_conda.verneed.txt");
        let adversarial = include_str!("../tests/fixtures/readelf/adversarial.verneed.txt");
        assert_eq!(
            parse_readelf_version_needs(vendored)
                .values()
                .next()
                .copied()
                .flatten(),
            Some((2, 35))
        );
        assert_eq!(
            parse_readelf_version_needs(conda)
                .values()
                .next()
                .copied()
                .flatten(),
            Some((2, 10))
        );
        let parsed = parse_readelf_version_needs(adversarial);
        assert_eq!(
            parsed
                .get(&PathBuf::from("/tmp/private.so"))
                .copied()
                .flatten(),
            Some((2, 34))
        );
        assert_eq!(
            parsed
                .get(&PathBuf::from("/tmp/glibcxx-only.so"))
                .copied()
                .flatten(),
            None
        );
    }

    #[test]
    #[ignore = "live: needs readelf on PATH; run with --include-ignored"]
    fn live_readelf_fixture_reports_fake_high_glibc_need() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/readelf/libretread_fake_glibc_9_99.so");
        let Some(readelf) = resolve_readelf(Path::new(env!("CARGO_MANIFEST_DIR"))) else {
            eprintln!("skipping: readelf unavailable");
            return;
        };
        let text = run_readelf(&readelf, "-V", std::slice::from_ref(&fixture))
            .expect("readelf -V fixture");
        let parsed = parse_single_version_need(&text).expect("parse single readelf output");
        assert_eq!(parsed, Some((9, 99)));
    }

    #[test]
    fn manylinux_floor_extraction_takes_max() {
        let text = "reject manylinux_2_34_x86_64 and manylinux_2_35_x86_64";
        assert_eq!(extract_manylinux_floor(text), Some((2, 35)));
    }

    #[test]
    fn undeclared_error_names_both_toml_forms() {
        let msg = undeclared_glibc_error(Some((2, 34)), Some((2, 35)));
        assert!(msg.contains("[system-requirements]"));
        assert!(msg.contains("libc = \"2.35\""));
        assert!(msg.contains("[workspace]"));
        assert!(msg.contains("platforms = [{ platform = \"linux-"));
    }

    #[test]
    fn path_pattern_matching_supports_literal_star_and_globstar() {
        assert!(path_pattern_matches(
            "isaacsim/kit/kernel/*/libpython3.12.so*",
            "isaacsim/kit/kernel/plugins/libpython3.12.so.1.0"
        ));
        assert!(path_pattern_matches(
            "isaacsim/**/libpython3.12.so.1.0",
            "isaacsim/kit/kernel/plugins/libpython3.12.so.1.0"
        ));
        assert!(!path_pattern_matches(
            "isaacsim/kit/*.so",
            "isaacsim/kit/kernel/plugins/libpython3.12.so.1.0"
        ));
    }

    #[test]
    fn marker_parses_digest_line_compatibly() {
        assert!(parse_marker_audit("abc\n").is_none());
        let audit = InstalledMarkerAudit {
            schema: 1,
            host_glibc: Some("2.34".to_string()),
            relaxed_platform: None,
            declaration_source: None,
            audit: AuditStatus::Passed,
            fixups: Vec::new(),
            offenders: Vec::new(),
            file_cache: Vec::new(),
        };
        let body = marker_body("abc", &audit).unwrap();
        assert_eq!(body.lines().next(), Some("abc"));
        assert_eq!(
            parse_marker_audit(&body).unwrap().host_glibc.as_deref(),
            Some("2.34")
        );
    }

    fn tempdir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "retread-glibc-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    unsafe fn restore_env(key: &str, value: Option<std::ffi::OsString>) {
        if let Some(value) = value {
            unsafe { std::env::set_var(key, value) };
        } else {
            unsafe { std::env::remove_var(key) };
        }
    }

    fn lock_with_shadow(path: &str) -> RetreadLock {
        let mut shadow_libs = BTreeMap::new();
        shadow_libs.insert(path.to_string(), "conda-lib".to_string());
        RetreadLock {
            schema: crate::lock::SCHEMA,
            retread_version: "test".to_string(),
            bundle: "test-bundle".to_string(),
            version: "1.0.0".to_string(),
            python: "3.11".to_string(),
            target_subdir: "linux-64".to_string(),
            resolution_glibc: None,
            inputs_hash: "abc".to_string(),
            root_requirements: vec!["test-bundle-pypi==1.0.0".to_string()],
            wheels: Vec::new(),
            conda_run_deps: Vec::new(),
            index_urls: Vec::new(),
            prerelease: BTreeMap::new(),
            shadow_libs,
            declared_glibc: None,
            conda_capable: Vec::new(),
            entry_specs: Vec::new(),
            wheel_store: None,
        }
    }

    fn wheel(name: &str) -> LockWheel {
        LockWheel {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            origin: Origin::Index,
            filename: format!("{name}-1.0.0-py3-none-any.whl"),
            url: Some(format!("https://example.invalid/{name}.whl")),
            sha256: None,
            requires_dist: Vec::new(),
            must_ship: false,
            upstream_url: None,
            git_source: None,
            sdist_source: None,
        }
    }

    #[test]
    fn effective_shadow_libs_defaults_only_for_isaacsim_without_config() {
        let mut lock = lock_with_shadow("custom.so");
        assert_eq!(effective_shadow_libs(&lock).len(), 1);
        assert!(effective_shadow_libs(&lock).contains_key("custom.so"));

        lock.shadow_libs.clear();
        lock.wheels = vec![wheel("isaacsim-kernel")];
        let defaults = effective_shadow_libs(&lock);
        assert!(
            defaults.contains_key("isaacsim/kit/kernel/plugins/libpython3.12.so.1.0"),
            "compiled-in Isaac Sim libpython default must be active when no config overrides it"
        );

        lock.wheels = vec![wheel("unrelated")];
        assert!(effective_shadow_libs(&lock).is_empty());
    }

    #[test]
    fn declared_glibc_precedence_env_manifest_lock() {
        let root = tempdir("declared-precedence");
        let manifest = root.join("pixi.toml");
        let old_declared = std::env::var_os("RETREAD_DECLARED_GLIBC");
        let old_manifest = std::env::var_os("PIXI_PROJECT_MANIFEST");
        let old_root = std::env::var_os("PIXI_PROJECT_ROOT");
        std::fs::write(
            &manifest,
            r#"
[workspace]
platforms = [{ platform = "linux-64", glibc = "2.35" }, { platform = "linux-aarch64", glibc = "2.35" }]
"#,
        )
        .unwrap();
        let mut lock = lock_with_shadow("x.so");
        lock.declared_glibc = Some("2.39".to_string());

        unsafe {
            std::env::remove_var("RETREAD_DECLARED_GLIBC");
            std::env::set_var("PIXI_PROJECT_MANIFEST", &manifest);
            std::env::remove_var("PIXI_PROJECT_ROOT");
        }
        let from_manifest = resolve_declared_glibc(&lock).unwrap();
        assert_eq!(from_manifest.version, (2, 35));
        assert_eq!(from_manifest.source, "workspace");

        unsafe { std::env::set_var("RETREAD_DECLARED_GLIBC", "2.36") };
        let from_env = resolve_declared_glibc(&lock).unwrap();
        assert_eq!(from_env.version, (2, 36));
        assert_eq!(from_env.source, "env");

        unsafe {
            std::env::remove_var("RETREAD_DECLARED_GLIBC");
            std::env::remove_var("PIXI_PROJECT_MANIFEST");
            std::env::remove_var("PIXI_PROJECT_ROOT");
        }
        let from_lock = resolve_declared_glibc(&lock).unwrap();
        assert_eq!(from_lock.version, (2, 39));
        assert_eq!(from_lock.source, "lock");
        unsafe {
            restore_env("RETREAD_DECLARED_GLIBC", old_declared);
            restore_env("PIXI_PROJECT_MANIFEST", old_manifest);
            restore_env("PIXI_PROJECT_ROOT", old_root);
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn fixup_rebacks_up_current_regular_file_on_upgrade() {
        let root = tempdir("rebackup");
        let prefix = root.join("prefix");
        let site = prefix.join("lib/python3.11/site-packages");
        let vendored = site.join("isaacsim/kit/kernel/plugins/libpython3.12.so.1.0");
        let provider = prefix.join("lib/libpython3.12.so.1.0");
        std::fs::create_dir_all(vendored.parent().unwrap()).unwrap();
        std::fs::create_dir_all(provider.parent().unwrap()).unwrap();
        std::fs::write(&provider, "provider").unwrap();
        std::fs::write(&vendored, "v1").unwrap();

        let rel = "isaacsim/kit/kernel/plugins/libpython3.12.so.1.0";
        let lock = lock_with_shadow(rel);
        let payload = vec![PayloadLib {
            rel_path: rel.to_string(),
            abs_path: vendored.clone(),
        }];
        install_audit(&lock, &prefix, &site, &payload, None, None, None).unwrap();
        assert!(
            std::fs::symlink_metadata(&vendored)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            std::fs::read_to_string(backup_path(&vendored)).unwrap(),
            "v1"
        );

        std::fs::remove_file(&vendored).unwrap();
        std::fs::write(&vendored, "v2").unwrap();
        install_audit(&lock, &prefix, &site, &payload, None, None, None).unwrap();
        assert!(
            std::fs::symlink_metadata(&vendored)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            std::fs::read_to_string(backup_path(&vendored)).unwrap(),
            "v2",
            "upgrade path must back up the fresh regular file, not reuse stale v1"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn verify_audit_record_rejects_clobbered_fixup_symlink() {
        let root = tempdir("verify-clobbered");
        let prefix = root.join("prefix");
        let vendored = prefix
            .join("lib/python3.11/site-packages/isaacsim/kit/kernel/plugins/libpython3.12.so.1.0");
        std::fs::create_dir_all(vendored.parent().unwrap()).unwrap();
        std::fs::write(&vendored, "regular").unwrap();
        let audit = InstalledMarkerAudit {
            schema: 1,
            host_glibc: None,
            relaxed_platform: None,
            declaration_source: None,
            audit: AuditStatus::Passed,
            fixups: vec![FixupRecord {
                path: "isaacsim/kit/kernel/plugins/libpython3.12.so.1.0".to_string(),
                soname: "libpython3.12.so.1.0".to_string(),
                provider: "lib/libpython3.12.so.1.0".to_string(),
            }],
            offenders: Vec::new(),
            file_cache: Vec::new(),
        };
        let err = verify_audit_record(&audit, &prefix).unwrap_err();
        assert!(
            format!("{err:#}").contains("no longer a symlink"),
            "unexpected error: {err:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }
}
