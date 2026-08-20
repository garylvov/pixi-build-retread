//! F14: workspace virtual-package requirements the build host cannot satisfy.
//!
//! A cold solve on a CPU batch node (glibc 2.34, no GPU) against a workspace
//! that declares `glibc = "2.35"` / `cuda = "12"` on its named platform is
//! unsatisfiable for a reason that has nothing to do with the request: the
//! host simply does not carry `__glibc >= 2.35` or `__cuda` at all. Pixi
//! itself panics in build dispatch ("missing virtual packages: __cuda >= 12")
//! unless `CONDA_OVERRIDE_CUDA` / `CONDA_OVERRIDE_GLIBC` are exported, and
//! retread's own resolvo solves inherited the same hole because host
//! detection ran with [`VirtualPackageOverrides::default()`] — which is
//! `None` for every field, i.e. it ignores `CONDA_OVERRIDE_*` entirely.
//!
//! This module is the detector/actuator pair for that gap (§1.2 reader/writer):
//! [`solve_overrides`] reads the workspace's declared requirements plus the
//! true host, injects the declared value into retread's OWN solves, and emits
//! ONE loud WARN naming the exports pixi itself still needs. An explicit
//! `CONDA_OVERRIDE_*` from the operator is authoritative and is never
//! second-guessed or double-applied.

use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::OnceLock;

use rattler_conda_types::Version;
use rattler_virtual_packages::{Override, VirtualPackageOverrides};

/// The two requirements a Pixi named platform can declare that a Linux build
/// host may fail to satisfy. `archspec`/`linux` floors are effectively always
/// met on our nodes and are deliberately out of scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VirtualKind {
    Cuda,
    Glibc,
}

impl VirtualKind {
    /// The `__`-prefixed conda virtual-package name.
    pub(crate) fn package_name(self) -> &'static str {
        match self {
            Self::Cuda => "__cuda",
            Self::Glibc => "__glibc",
        }
    }

    /// The environment variable pixi (and rattler) read as an override.
    pub(crate) fn env_var(self) -> &'static str {
        match self {
            Self::Cuda => "CONDA_OVERRIDE_CUDA",
            Self::Glibc => "CONDA_OVERRIDE_GLIBC",
        }
    }
}

/// One declared requirement the host cannot satisfy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VirtualGap {
    pub(crate) kind: VirtualKind,
    /// The value the workspace declared (e.g. `"2.35"`, `"12"`).
    pub(crate) declared: String,
    /// The host's value, or `None` when the host has no such package at all.
    pub(crate) host: Option<String>,
}

/// The host's true virtual-package situation, plus which overrides the
/// operator already set explicitly. Kept as plain data so the planner is a
/// pure function testable on any host.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct HostVirtualPackages {
    pub(crate) cuda: Option<String>,
    pub(crate) glibc: Option<String>,
    /// `CONDA_OVERRIDE_CUDA` is set to a non-empty value in this process.
    pub(crate) cuda_override_set: bool,
    /// `CONDA_OVERRIDE_GLIBC` is set to a non-empty value in this process.
    pub(crate) glibc_override_set: bool,
}

/// The gaps found, in declaration-name order (`__cuda` then `__glibc`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct VirtualOverridePlan {
    pub(crate) gaps: Vec<VirtualGap>,
}

impl VirtualOverridePlan {
    /// Guard-facing: assert a satisfied host produced no override at all.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.gaps.is_empty()
    }

    fn gap(&self, kind: VirtualKind) -> Option<&VirtualGap> {
        self.gaps.iter().find(|gap| gap.kind == kind)
    }

    /// The rattler override set retread's own host detection should run with.
    ///
    /// Starts from [`VirtualPackageOverrides::from_env`] so an explicit
    /// `CONDA_OVERRIDE_*` is honoured (the pre-F14 `default()` silently
    /// ignored it), then pins the declared value for every gap. A gap is by
    /// construction never a key the operator set, so the two never collide.
    pub(crate) fn overrides(&self) -> VirtualPackageOverrides {
        let mut overrides = VirtualPackageOverrides::from_env();
        if let Some(gap) = self.gap(VirtualKind::Cuda) {
            overrides.cuda = Some(Override::String(gap.declared.clone()));
        }
        if let Some(gap) = self.gap(VirtualKind::Glibc) {
            overrides.libc = Some(Override::String(gap.declared.clone()));
        }
        overrides
    }

    /// The single WARN line. `None` when there is nothing to warn about.
    pub(crate) fn warning(&self) -> Option<String> {
        if self.gaps.is_empty() {
            return None;
        }
        let names = self
            .gaps
            .iter()
            .map(|gap| gap.kind.package_name())
            .collect::<Vec<_>>()
            .join("/");
        // Detail and export order is glibc-then-cuda: the glibc floor is the
        // one an operator can actually reason about, so it leads.
        let ordered: Vec<&VirtualGap> = [VirtualKind::Glibc, VirtualKind::Cuda]
            .into_iter()
            .filter_map(|kind| self.gap(kind))
            .collect();
        let details = ordered
            .iter()
            .map(|gap| match (gap.kind, gap.host.as_deref()) {
                (VirtualKind::Glibc, Some(host)) => {
                    format!("host glibc {host} < declared {}", gap.declared)
                }
                (VirtualKind::Glibc, None) => "no glibc".to_string(),
                (VirtualKind::Cuda, Some(host)) => {
                    format!("host CUDA {host} < declared {}", gap.declared)
                }
                (VirtualKind::Cuda, None) => "no CUDA".to_string(),
            })
            .collect::<Vec<_>>()
            .join("; ");
        let exports = ordered
            .iter()
            .map(|gap| format!("{}={}", gap.kind.env_var(), gap.declared))
            .collect::<Vec<_>>()
            .join(" ");
        Some(format!(
            "host lacks {names} required by the workspace ({details}); \
             retread is overriding its own solves; \
             for pixi itself export {exports}"
        ))
    }
}

/// Pure planner: which declared requirements does this host fail to meet?
///
/// `system_requirements` is the pixi-schema map retread already assembles from
/// the manifest's named platforms and `[system-requirements]`
/// (`WorkspaceManifest::effective_system_requirements_for_target`), so `cuda`
/// and `libc`/`glibc` arrive here already resolved for the target platform.
pub(crate) fn plan_virtual_overrides(
    system_requirements: &BTreeMap<String, String>,
    host: &HostVirtualPackages,
) -> VirtualOverridePlan {
    let mut gaps = Vec::new();
    for kind in [VirtualKind::Cuda, VirtualKind::Glibc] {
        let (declared, host_value, override_set) = match kind {
            VirtualKind::Cuda => (
                system_requirements.get("cuda"),
                host.cuda.as_deref(),
                host.cuda_override_set,
            ),
            VirtualKind::Glibc => (
                system_requirements
                    .get("libc")
                    .or_else(|| system_requirements.get("glibc")),
                host.glibc.as_deref(),
                host.glibc_override_set,
            ),
        };
        let Some(declared) = declared else { continue };
        // An explicit operator override is authoritative: respect it, and do
        // not warn or double-apply.
        if override_set {
            continue;
        }
        let satisfied = match host_value {
            None => false,
            Some(host_value) => {
                match (Version::from_str(host_value), Version::from_str(declared)) {
                    (Ok(host_version), Ok(declared_version)) => host_version >= declared_version,
                    // An unparseable version on either side is not evidence of a
                    // gap; a false WARN is worse than a missing one here.
                    _ => true,
                }
            }
        };
        if !satisfied {
            gaps.push(VirtualGap {
                kind,
                declared: declared.clone(),
                host: host_value.map(str::to_string),
            });
        }
    }
    VirtualOverridePlan { gaps }
}

/// Read the host's true `__cuda`/`__glibc` (deliberately WITHOUT applying
/// `CONDA_OVERRIDE_*`, so a gap is detected against reality) and note which
/// overrides the operator already exported.
fn detect_host_virtual_packages() -> HostVirtualPackages {
    let env_set = |name: &str| {
        std::env::var(name)
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false)
    };
    let glibc = match rattler_virtual_packages::LibC::current() {
        Ok(Some(libc)) => Some(libc.version.to_string()),
        Ok(None) => None,
        Err(error) => {
            tracing::debug!(%error, "solve-check: host libc detection failed");
            None
        }
    };
    HostVirtualPackages {
        cuda: rattler_virtual_packages::Cuda::current().map(|cuda| cuda.version.to_string()),
        glibc,
        cuda_override_set: env_set(VirtualKind::Cuda.env_var()),
        glibc_override_set: env_set(VirtualKind::Glibc.env_var()),
    }
}

/// Emit the plan's WARN at most once per process. A solve check runs many
/// times per build; the operator needs the export line once, loudly.
fn warn_once(plan: &VirtualOverridePlan) {
    static WARNED: OnceLock<()> = OnceLock::new();
    let Some(warning) = plan.warning() else {
        return;
    };
    if WARNED.set(()).is_ok() {
        tracing::warn!("{warning}");
    }
}

/// Solve-startup entry point: plan, warn once, and hand back the rattler
/// override set retread's own host detection must run with.
pub(crate) fn solve_overrides(
    system_requirements: &BTreeMap<String, String>,
) -> VirtualPackageOverrides {
    let plan = plan_virtual_overrides(system_requirements, &detect_host_virtual_packages());
    warn_once(&plan);
    plan.overrides()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn imprint_workspace() -> BTreeMap<String, String> {
        // Mirrors imprint-data/pixi.toml's named linux-64 platform.
        BTreeMap::from([
            ("cuda".to_string(), "12".to_string()),
            ("libc".to_string(), "2.35".to_string()),
        ])
    }

    /// (a) declared > host and no `CONDA_OVERRIDE_*`: both overrides applied
    /// and the WARN carries the exact operator-facing export line.
    #[test]
    fn declared_beyond_host_overrides_and_warns() {
        let host = HostVirtualPackages {
            cuda: None,
            glibc: Some("2.34".to_string()),
            cuda_override_set: false,
            glibc_override_set: false,
        };
        let plan = plan_virtual_overrides(&imprint_workspace(), &host);
        assert_eq!(
            plan.gaps
                .iter()
                .map(|gap| gap.kind.package_name())
                .collect::<Vec<_>>(),
            vec!["__cuda", "__glibc"],
        );
        assert_eq!(
            plan.warning().as_deref(),
            Some(
                "host lacks __cuda/__glibc required by the workspace \
                 (host glibc 2.34 < declared 2.35; no CUDA); \
                 retread is overriding its own solves; \
                 for pixi itself export CONDA_OVERRIDE_GLIBC=2.35 CONDA_OVERRIDE_CUDA=12"
            ),
        );
        let overrides = plan.overrides();
        assert_eq!(overrides.cuda, Some(Override::String("12".to_string())));
        assert_eq!(overrides.libc, Some(Override::String("2.35".to_string())));
    }

    /// (b) the operator already exported the override: respect it, do not
    /// re-apply it, and do not warn about the key they already handled.
    #[test]
    fn explicit_env_override_is_respected_without_double_apply() {
        let host = HostVirtualPackages {
            cuda: None,
            glibc: Some("2.34".to_string()),
            cuda_override_set: true,
            glibc_override_set: true,
        };
        let plan = plan_virtual_overrides(&imprint_workspace(), &host);
        assert!(plan.is_empty(), "explicit overrides leave no gap: {plan:?}");
        assert_eq!(plan.warning(), None);
        let overrides = plan.overrides();
        // from_env() semantics: the env var is read by rattler itself, not
        // pinned to a String we substituted on top of it.
        assert_eq!(overrides.cuda, Some(Override::DefaultEnvVar));
        assert_eq!(overrides.libc, Some(Override::DefaultEnvVar));

        // Half-set is still half-detected: only the unset key gaps.
        let half = HostVirtualPackages {
            glibc_override_set: false,
            ..host
        };
        let plan = plan_virtual_overrides(&imprint_workspace(), &half);
        assert_eq!(
            plan.gaps
                .iter()
                .map(|gap| gap.kind.package_name())
                .collect::<Vec<_>>(),
            vec!["__glibc"],
        );
        assert_eq!(
            plan.warning().as_deref(),
            Some(
                "host lacks __glibc required by the workspace \
                 (host glibc 2.34 < declared 2.35); \
                 retread is overriding its own solves; \
                 for pixi itself export CONDA_OVERRIDE_GLIBC=2.35"
            ),
        );
    }

    /// (c) the host already satisfies every declaration: nothing is
    /// overridden and nothing is warned.
    #[test]
    fn satisfied_host_changes_nothing() {
        let host = HostVirtualPackages {
            cuda: Some("12.4".to_string()),
            glibc: Some("2.39".to_string()),
            cuda_override_set: false,
            glibc_override_set: false,
        };
        let plan = plan_virtual_overrides(&imprint_workspace(), &host);
        assert!(plan.is_empty(), "satisfied host has no gap: {plan:?}");
        assert_eq!(plan.warning(), None);
        let overrides = plan.overrides();
        assert_eq!(overrides.cuda, Some(Override::DefaultEnvVar));
        assert_eq!(overrides.libc, Some(Override::DefaultEnvVar));

        // Exactly equal also satisfies (`>=`, not `>`).
        let exact = HostVirtualPackages {
            cuda: Some("12".to_string()),
            glibc: Some("2.35".to_string()),
            ..host
        };
        assert!(plan_virtual_overrides(&imprint_workspace(), &exact).is_empty());
    }

    /// A workspace that declares nothing cannot gap, whatever the host is.
    #[test]
    fn undeclared_requirements_never_gap() {
        let host = HostVirtualPackages {
            cuda: None,
            glibc: Some("2.17".to_string()),
            cuda_override_set: false,
            glibc_override_set: false,
        };
        assert!(plan_virtual_overrides(&BTreeMap::new(), &host).is_empty());
    }

    /// The deprecated `glibc` spelling is read when `libc` is absent.
    #[test]
    fn legacy_glibc_key_is_read() {
        let requirements = BTreeMap::from([("glibc".to_string(), "2.35".to_string())]);
        let host = HostVirtualPackages {
            cuda: None,
            glibc: Some("2.34".to_string()),
            cuda_override_set: false,
            glibc_override_set: false,
        };
        let plan = plan_virtual_overrides(&requirements, &host);
        assert_eq!(plan.gaps.len(), 1);
        assert_eq!(plan.gaps[0].kind, VirtualKind::Glibc);
    }
}
