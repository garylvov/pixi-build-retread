//! v0.36.0: per-chain verdicts (replaces v0.35.0's aggregate-class model).
//!
//! v0.35.0 returned a single `ConflictClass` per unsat -- "if ANY chain
//! is Class A actionable, the whole unsat is Class A, widen everything
//! in `blocking_deps`". That collapsed per-chain information at the
//! exact moment the refinement loop needed it. Real-world consequence:
//! when one chain was `pytorch` (Class A) and another was `python`
//! (workspace-pin-dominates + ABI anchor), the aggregate picked A and
//! the widening loop saw `python` in `emitted_names` (retread DOES
//! emit `python` from wheel Requires-Dist) and widened it to `*`. The
//! resulting CondaOutput shipped `python *` to pixi, pixi's full solve
//! happily picked python 3.14, every other dep's `python_abi 3.11.*`
//! broke. See HANDOFF v0.36.0 entry for the gory details.
//!
//! The fix: `classify_chains()` returns `Vec<PerChainVerdict>` -- one
//! verdict per blocking chain. The refinement loop iterates verdicts
//! and widens ONLY `WidenRetread` entries. `AbiAnchor` is never widened
//! regardless of who emits it (retread, workspace, transitive); the
//! `is_abi_anchor()` predicate is the single source of truth and is
//! also used at translate time (`relax.rs::translate`) so the same
//! protection applies everywhere.
//!
//! The old `ConflictClass` + `classify_unsat()` are KEPT as a derived-
//! from-verdicts API for the audit pipeline + RPC error tagging. Those
//! consumers want a single roll-up label, and that's still the right
//! abstraction THERE -- just not inside the refinement loop.
//!
//! Pure functions over data from `solve_check::BlockingChain`. No IO.

use std::collections::HashSet;

use serde::Serialize;

use crate::solve_check::BlockingChain;

// --------------------------------------------------------------------
// ABI anchors: deps the cascade must never widen.
// --------------------------------------------------------------------

/// Names that are exact matches for ABI-anchor packages. Widening
/// any of these breaks the conda env's ABI contract (python tag,
/// libc, libstdcxx, CUDA runtime, compiler activation packages).
///
/// Sourced from common conda-forge conventions; expanded as new
/// failure modes surface. Anchor list is intentionally conservative:
/// missing entries here just degrade to the v0.35.x behavior (the
/// classifier might widen them); spurious entries here would cause
/// the cascade to give up too early. Bias toward MORE entries -- the
/// cost of a false "never widen" is a slightly less helpful error
/// message; the cost of a false "ok to widen" is a corrupted output
/// that pixi consumes silently.
const ABI_ANCHOR_NAMES: &[&str] = &[
    // Python ABI: widening `python` strips the cp-tag constraint; every
    // wheel-emitted dep then collapses to "any python" and pixi picks
    // 3.14 against a 3.11-only workspace.
    "python",
    "python_abi",
    "pypy",
    // glibc / libc: the C runtime ABI. Conda-forge encodes the floor
    // via `__glibc` virtual + `libc` direct; widening these would
    // claim retread's output runs on any libc, which is almost never
    // true (we ship native wheels with manylinux_2_xx tags).
    "libc",
    "glibc",
    "__glibc",
    // libstdcxx / libcxx: the C++ runtime ABI. PyTorch and friends
    // pin libstdcxx-ng tightly; the workspace has matching pins;
    // widening retread's emission lets the solver pick a libstdcxx
    // older than what the wheels need.
    "libstdcxx-ng",
    "libstdcxx",
    "libcxx",
    "libcxx-devel",
    // CUDA runtime: workspace pins `cuda-version ==12.8` for a reason
    // (driver match, sm arch). Widening lets the solver pick cuda 13
    // and break every cuda-bindings/cuda-toolkit/torch interaction.
    "cuda-version",
    "__cuda",
    // Other rattler virtual packages (`__linux`, `__osx`, `__win`,
    // `__unix`, `__archspec`) are caught by the `__` prefix check
    // below. Arch-tagged compilers + binutils are caught by the
    // prefix list. `*_compiler` suffix is caught by the predicate.
];

/// Predicate forms used in addition to the exact-name list. Cheap
/// substring/prefix checks for families of names.
fn is_abi_anchor_pattern(name: &str) -> bool {
    // Rattler virtual packages always start with `__`.
    if name.starts_with("__") {
        return true;
    }
    // Compiler activation: `c_compiler`, `cxx_compiler`,
    // `fortran_compiler`, etc.
    if name.ends_with("_compiler") {
        return true;
    }
    // Arch-tagged compilers + binutils: `gcc_linux-64`, `gxx_linux-64`,
    // `binutils_linux-64`, `gfortran_linux-64`, `clang_linux-64`,
    // `clangxx_linux-64`, etc. Match the `_<os>-<arch>` suffix shape.
    let abi_tagged_prefixes = [
        "gcc_",
        "gxx_",
        "g++_",
        "gfortran_",
        "clang_",
        "clangxx_",
        "binutils_",
        "ld_",
        "sysroot_",
    ];
    if abi_tagged_prefixes.iter().any(|p| name.starts_with(p)) {
        return true;
    }
    false
}

/// Is `name` an ABI-anchor dependency the cascade must never widen?
/// This is the SINGLE source of truth -- the refinement loop, the
/// translate-time relax, and the post-condition invariant all call
/// this same function.
pub fn is_abi_anchor(name: &str) -> bool {
    ABI_ANCHOR_NAMES.contains(&name) || is_abi_anchor_pattern(name)
}

// --------------------------------------------------------------------
// Per-chain verdict: the new structured output of classification.
// --------------------------------------------------------------------

/// One verdict per blocking chain from rattler's unsat explanation.
///
/// The refinement loop in `handler::iterative_solve_refinement`
/// iterates verdicts and acts per-verdict:
///   * `WidenRetread`: widen this dep one level, re-emit, re-solve.
///   * `AbiAnchor`: stop the loop and surface diagnostics -- this dep
///     must NEVER be widened, regardless of who emits it.
///   * `WorkspacePinDominates`: stop the loop and surface the
///     suggestion; the workspace pin is the floor.
///   * `TransitiveOnly`: stop the loop; not declared by either side.
///   * `AlreadyExhausted`: cascade already widened this to `*`; the
///     real cause is via the transitive requirement.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum PerChainVerdict {
    /// Retread emits this dep, workspace doesn't pin it, and it isn't
    /// an ABI anchor. Safe to widen one level.
    WidenRetread { dep: String, current_spec: String },
    /// ABI anchor (python, libc, cuda-version, *_compiler, ...).
    /// Never widen, regardless of whether retread emits it.
    AbiAnchor { dep: String, reason: String },
    /// Workspace pins this dep (possibly in addition to retread).
    /// Workspace pin is the floor; widening retread's emission won't
    /// help. A suggestion is attached for the audit.
    WorkspacePinDominates {
        dep: String,
        suggestion: Option<WorkspaceEditSuggestion>,
        also_emitted_by_retread: bool,
    },
    /// Cascade already widened this to `*`; the binding constraint is
    /// via the transitive requirement. Can't widen further.
    AlreadyExhausted {
        dep: String,
        current_spec: String,
        transitive_requirement: String,
    },
    /// Surfaced as top-level but declared by neither side. Almost
    /// certainly a transitive that bubbled up.
    TransitiveOnly { dep: String },
}

impl PerChainVerdict {
    /// The dep name this verdict refers to.
    pub fn dep(&self) -> &str {
        match self {
            Self::WidenRetread { dep, .. } => dep,
            Self::AbiAnchor { dep, .. } => dep,
            Self::WorkspacePinDominates { dep, .. } => dep,
            Self::AlreadyExhausted { dep, .. } => dep,
            Self::TransitiveOnly { dep } => dep,
        }
    }

    /// Short label for audit/log output.
    pub fn label(&self) -> &'static str {
        match self {
            Self::WidenRetread { .. } => "widen-retread",
            Self::AbiAnchor { .. } => "abi-anchor",
            Self::WorkspacePinDominates { .. } => "workspace-pin-dominates",
            Self::AlreadyExhausted { .. } => "already-exhausted",
            Self::TransitiveOnly { .. } => "transitive-only",
        }
    }

    /// True if and only if the refinement loop should widen this dep.
    pub fn is_widenable(&self) -> bool {
        matches!(self, Self::WidenRetread { .. })
    }
}

/// Classify every blocking chain into a per-chain verdict. The
/// refinement loop iterates the returned vec and acts on each verdict
/// independently -- NO collapse to an aggregate class.
pub fn classify_chains(
    chains: &[BlockingChain],
    retread_emitted_names: &HashSet<String>,
    workspace_dep_names: &HashSet<String>,
    already_widened_to_star: &HashSet<String>,
    env_name: &str,
) -> Vec<PerChainVerdict> {
    let mut verdicts: Vec<PerChainVerdict> = Vec::with_capacity(chains.len());
    for chain in chains {
        // ABI anchor takes precedence over EVERY other classification.
        // Even if retread happens to be emitting `python` (from a
        // wheel's `Requires-Dist: python>=3.10`), the cascade must
        // never touch it -- python is the workspace's ABI contract.
        if is_abi_anchor(&chain.name) {
            verdicts.push(PerChainVerdict::AbiAnchor {
                dep: chain.name.clone(),
                reason: format!(
                    "`{}` is an ABI anchor; widening it would corrupt the env's ABI contract \
                     (python tag, libc, libstdcxx, cuda runtime, or compiler activation)",
                    chain.name,
                ),
            });
            continue;
        }

        let by_retread = retread_emitted_names.contains(&chain.name);
        let by_workspace = workspace_dep_names.contains(&chain.name);
        let already_exhausted = already_widened_to_star.contains(&chain.name);

        if by_workspace {
            // v0.36.2+: skip suggestion derivation for "installable"
            // chains -- these are listed as part of a multi-dep
            // incompatibility group but aren't themselves the
            // blocker (rattler says "can be installed with any of
            // the following options" rather than "cannot be
            // installed"). Their transitive_requirement field
            // typically captures the version-enumeration line and
            // produces gibberish suggestions like
            // "remove X OR relax `X 1.0 | 1.0`". Mark them as
            // workspace-dominated for the verdict trace but emit
            // no suggestion -- the actionable chain is one of the
            // genuinely blocking siblings.
            let suggestion = if chain.installable {
                None
            } else {
                derive_suggestion(chain, env_name, by_retread)
            };
            verdicts.push(PerChainVerdict::WorkspacePinDominates {
                dep: chain.name.clone(),
                suggestion,
                also_emitted_by_retread: by_retread,
            });
            continue;
        }

        if by_retread && already_exhausted {
            verdicts.push(PerChainVerdict::AlreadyExhausted {
                dep: chain.name.clone(),
                current_spec: chain.current_spec.clone(),
                transitive_requirement: chain.transitive_requirement.clone(),
            });
            continue;
        }

        if by_retread {
            verdicts.push(PerChainVerdict::WidenRetread {
                dep: chain.name.clone(),
                current_spec: chain.current_spec.clone(),
            });
            continue;
        }

        verdicts.push(PerChainVerdict::TransitiveOnly {
            dep: chain.name.clone(),
        });
    }
    verdicts
}

/// One-line human-readable summary for the per-iteration audit row.
/// Aggregates the verdict labels in a stable form.
pub fn summarize_verdicts(verdicts: &[PerChainVerdict]) -> String {
    let parts: Vec<String> = verdicts
        .iter()
        .map(|v| match v {
            PerChainVerdict::WidenRetread { dep, .. } => format!("{dep} (retread will widen)"),
            PerChainVerdict::AbiAnchor { dep, .. } => {
                format!("{dep} (ABI anchor -- NEVER widened)")
            }
            PerChainVerdict::WorkspacePinDominates {
                dep,
                also_emitted_by_retread,
                ..
            } => {
                if *also_emitted_by_retread {
                    format!("{dep} (workspace pin + retread emission -- workspace dominates)")
                } else {
                    format!("{dep} (workspace-pinned -- retread doesn't emit this)")
                }
            }
            PerChainVerdict::AlreadyExhausted {
                dep,
                transitive_requirement,
                ..
            } => format!(
                "{dep} (retread already widened to `*`; chain blocks via transitive `{}`)",
                transitive_requirement,
            ),
            PerChainVerdict::TransitiveOnly { dep } => format!(
                "{dep} (not declared by retread or workspace -- transitive surfaced as top-level)",
            ),
        })
        .collect();
    parts.join("; ")
}

// --------------------------------------------------------------------
// Back-compat: aggregate `ConflictClass` + `classify_unsat`.
// --------------------------------------------------------------------

/// One of four failure modes the audit pipeline rolls up to. KEPT in
/// v0.36.0 as a derived-from-verdicts label for the audit + RPC error
/// tagging. The REFINEMENT LOOP no longer uses this -- it consumes
/// `Vec<PerChainVerdict>` directly so per-chain decisions aren't lost.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum ConflictClass {
    /// At least one verdict is `WidenRetread` -- cascade can act this
    /// iteration.
    A,
    /// Some verdict is `AlreadyExhausted` and nothing else is actionable.
    AExhausted,
    /// Some verdict is `WorkspacePinDominates` or `AbiAnchor`.
    BWorkspaceDominated,
    /// All verdicts are `TransitiveOnly`.
    CWorkspaceOnly,
}

/// One actionable suggestion attached to the user's terminal output
/// + audit MD file.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct WorkspaceEditSuggestion {
    /// Pixi env this suggestion applies to.
    pub env: String,
    /// Feature block the offending pin lives under, if locatable.
    pub feature: Option<String>,
    /// The pin retread thinks should be relaxed.
    pub current_pin: String,
    /// A spec the solver would accept (heuristic).
    pub suggested_pin: String,
    /// One-line human-readable reason.
    pub reason: String,
}

/// Roll-up classification used by the audit pipeline + RPC error tag.
/// Derived from the per-chain verdicts.
#[derive(Debug, Clone, Serialize)]
pub struct ConflictClassification {
    pub class: ConflictClass,
    pub blocking_summary: String,
    pub suggestions: Vec<WorkspaceEditSuggestion>,
}

pub fn classify_unsat(
    chains: &[BlockingChain],
    retread_emitted_names: &HashSet<String>,
    workspace_dep_names: &HashSet<String>,
    already_widened_to_star: &HashSet<String>,
    env_name: &str,
) -> ConflictClassification {
    let verdicts = classify_chains(
        chains,
        retread_emitted_names,
        workspace_dep_names,
        already_widened_to_star,
        env_name,
    );
    let blocking_summary = summarize_verdicts(&verdicts);
    let suggestions: Vec<WorkspaceEditSuggestion> = verdicts
        .iter()
        .filter_map(|v| match v {
            PerChainVerdict::WorkspacePinDominates {
                suggestion: Some(s),
                ..
            } => Some(s.clone()),
            _ => None,
        })
        .collect();

    let any_widen = verdicts
        .iter()
        .any(|v| matches!(v, PerChainVerdict::WidenRetread { .. }));
    let any_workspace_or_abi = verdicts.iter().any(|v| {
        matches!(
            v,
            PerChainVerdict::WorkspacePinDominates { .. } | PerChainVerdict::AbiAnchor { .. },
        )
    });
    let any_exhausted = verdicts
        .iter()
        .any(|v| matches!(v, PerChainVerdict::AlreadyExhausted { .. }));

    let class = if any_widen {
        ConflictClass::A
    } else if any_workspace_or_abi {
        ConflictClass::BWorkspaceDominated
    } else if any_exhausted {
        ConflictClass::AExhausted
    } else {
        ConflictClass::CWorkspaceOnly
    };

    ConflictClassification {
        class,
        blocking_summary,
        suggestions,
    }
}

// --------------------------------------------------------------------
// Suggestion derivation (unchanged from v0.35.0).
// --------------------------------------------------------------------

/// Derive a workspace-edit suggestion from a blocking chain.
fn derive_suggestion(
    chain: &BlockingChain,
    env_name: &str,
    by_retread: bool,
) -> Option<WorkspaceEditSuggestion> {
    if chain.name.is_empty() {
        return None;
    }
    let lower = lower_bound_of(&chain.current_spec);
    let lower_anchor: Option<Vec<u64>> = lower.as_deref().and_then(extract_lower_version);

    let above_lower: Vec<String> = chain
        .rejected_versions
        .iter()
        .filter(|v| match (&lower_anchor, parse_loose(v)) {
            (Some(anchor), parsed) => parsed.as_slice() >= anchor.as_slice(),
            _ => true,
        })
        .cloned()
        .collect();

    let suggested_pin: String;
    let no_cap_works: bool;
    let cap_candidate: Option<(String, Vec<u64>)> = lowest_version(&above_lower)
        .and_then(|lowest| minor_of(&lowest).map(|m| (m, parse_loose(&lowest))));
    let lo_parsed: Option<Vec<u64>> = lower_anchor.clone();
    let cap_useful: bool = match (&cap_candidate, &lo_parsed) {
        (Some((_, cap_v)), Some(lo_v)) => cap_v.as_slice() > lo_v.as_slice(),
        (Some(_), None) => true,
        _ => false,
    };
    if cap_useful {
        let (cap_minor, _) = cap_candidate.unwrap();
        let lo = lower.unwrap_or_default();
        no_cap_works = false;
        suggested_pin = if lo.is_empty() {
            format!("<{cap_minor}")
        } else {
            format!("{lo},<{cap_minor}")
        };
    } else {
        no_cap_works = true;
        suggested_pin = format!(
            "remove `{}` from workspace OR relax `{}`",
            chain.name,
            chain.transitive_requirement.trim(),
        );
    }

    let reason = if no_cap_works {
        format!(
            "every version your current pin `{}` allows transitively requires `{}`, which conflicts with another workspace pin",
            chain.current_spec,
            if chain.transitive_requirement.is_empty() {
                "(unknown transitive)"
            } else {
                chain.transitive_requirement.as_str()
            },
        )
    } else if !chain.transitive_requirement.is_empty() {
        format!(
            "current pin `{}` picks a version requiring `{}`, which conflicts with another workspace pin",
            chain.current_spec, chain.transitive_requirement,
        )
    } else {
        format!(
            "current pin `{}` conflicts with other workspace constraints",
            chain.current_spec,
        )
    };

    Some(WorkspaceEditSuggestion {
        env: env_name.to_string(),
        feature: None,
        current_pin: format!("{} {}", chain.name, chain.current_spec)
            .trim()
            .to_string(),
        suggested_pin: if no_cap_works {
            suggested_pin
        } else {
            format!("{} {}", chain.name, suggested_pin)
        },
        reason: if by_retread {
            format!(
                "{reason} (retread also emits `{}` but the workspace pin dominates)",
                chain.name,
            )
        } else {
            reason
        },
    })
}

fn extract_lower_version(lower: &str) -> Option<Vec<u64>> {
    for prefix in [">=", "==", ">"] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let parsed = parse_loose(rest.trim());
            if !parsed.is_empty() && parsed[0] != u64::MAX {
                return Some(parsed);
            }
        }
    }
    None
}

fn lower_bound_of(spec: &str) -> Option<String> {
    for clause in spec.split(',') {
        let c = clause.trim();
        for op in [">=", "==", ">"] {
            if let Some(rest) = c.strip_prefix(op) {
                return Some(format!("{op}{}", rest.trim()));
            }
        }
    }
    None
}

fn lowest_version(versions: &[String]) -> Option<String> {
    if versions.is_empty() {
        return None;
    }
    let mut indexed: Vec<(Vec<u64>, &String)> =
        versions.iter().map(|v| (parse_loose(v), v)).collect();
    indexed.sort_by(|a, b| a.0.cmp(&b.0));
    indexed.first().map(|(_, v)| (*v).clone())
}

fn parse_loose(v: &str) -> Vec<u64> {
    v.split('.')
        .map(|p| {
            let head: String = p.chars().take_while(|c| c.is_ascii_digit()).collect();
            head.parse::<u64>().unwrap_or(u64::MAX)
        })
        .collect()
}

fn minor_of(v: &str) -> Option<String> {
    let parts: Vec<&str> = v.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    let major: u64 = parts[0].parse().ok()?;
    let minor: u64 = parts[1]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()?;
    Some(format!("{major}.{minor}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_with(name: &str, spec: &str, rejected: Vec<&str>, transitive: &str) -> BlockingChain {
        BlockingChain {
            name: name.into(),
            current_spec: spec.into(),
            rejected_versions: rejected.into_iter().map(String::from).collect(),
            transitive_requirement: transitive.into(),
            installable: false,
        }
    }

    fn hs(items: &[&str]) -> HashSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    // ---------- is_abi_anchor predicate ----------

    #[test]
    fn abi_anchor_recognizes_python() {
        assert!(is_abi_anchor("python"));
        assert!(is_abi_anchor("python_abi"));
    }

    #[test]
    fn abi_anchor_recognizes_libc_family() {
        assert!(is_abi_anchor("libc"));
        assert!(is_abi_anchor("glibc"));
        assert!(is_abi_anchor("__glibc"));
        assert!(is_abi_anchor("libstdcxx-ng"));
        assert!(is_abi_anchor("libcxx"));
    }

    #[test]
    fn abi_anchor_recognizes_cuda_runtime() {
        assert!(is_abi_anchor("cuda-version"));
        assert!(is_abi_anchor("__cuda"));
    }

    #[test]
    fn abi_anchor_recognizes_virtual_packages_via_prefix() {
        assert!(is_abi_anchor("__linux"));
        assert!(is_abi_anchor("__osx"));
        assert!(is_abi_anchor("__archspec"));
        assert!(is_abi_anchor("__win"));
    }

    #[test]
    fn abi_anchor_recognizes_compilers_via_suffix() {
        assert!(is_abi_anchor("c_compiler"));
        assert!(is_abi_anchor("cxx_compiler"));
        assert!(is_abi_anchor("fortran_compiler"));
    }

    #[test]
    fn abi_anchor_recognizes_arch_tagged_compilers() {
        assert!(is_abi_anchor("gcc_linux-64"));
        assert!(is_abi_anchor("gxx_linux-64"));
        assert!(is_abi_anchor("binutils_linux-64"));
        assert!(is_abi_anchor("gfortran_linux-aarch64"));
        assert!(is_abi_anchor("sysroot_linux-64"));
    }

    #[test]
    fn abi_anchor_lets_regular_deps_through() {
        assert!(!is_abi_anchor("pytorch"));
        assert!(!is_abi_anchor("pytorch-gpu"));
        assert!(!is_abi_anchor("triton"));
        assert!(!is_abi_anchor("numpy"));
        assert!(!is_abi_anchor("torchvision"));
        assert!(!is_abi_anchor("opencv"));
    }

    // ---------- classify_chains: per-chain verdicts ----------

    #[test]
    fn classify_chains_widen_retread() {
        let chains = vec![chain_with(
            "triton",
            ">=3.7.0,<3.8",
            vec!["3.7.0"],
            "cuda-version >=13.0,<14",
        )];
        let v = classify_chains(&chains, &hs(&["triton"]), &hs(&[]), &hs(&[]), "gsi");
        assert_eq!(v.len(), 1);
        match &v[0] {
            PerChainVerdict::WidenRetread { dep, current_spec } => {
                assert_eq!(dep, "triton");
                assert_eq!(current_spec, ">=3.7.0,<3.8");
            }
            other => panic!("expected WidenRetread, got {other:?}"),
        }
    }

    #[test]
    fn classify_chains_abi_anchor_beats_retread_emission() {
        // The exact gsi failure mode: retread emits python (from
        // wheel Requires-Dist) AND workspace pins it. v0.35.x would
        // have classified this as "workspace dominates" but the
        // widening loop still widened it because it was also in the
        // blocking list with Class A actionable on another chain.
        // v0.36.0: ABI anchor wins absolutely; never widen, period.
        let chains = vec![chain_with("python", "==3.11", vec!["3.14.5"], "")];
        let v = classify_chains(&chains, &hs(&["python"]), &hs(&["python"]), &hs(&[]), "gsi");
        assert_eq!(v.len(), 1);
        match &v[0] {
            PerChainVerdict::AbiAnchor { dep, .. } => assert_eq!(dep, "python"),
            other => panic!("expected AbiAnchor, got {other:?}"),
        }
    }

    #[test]
    fn classify_chains_cuda_version_is_abi_anchor() {
        let chains = vec![chain_with("cuda-version", "==12.8", vec!["13.0"], "")];
        let v = classify_chains(&chains, &hs(&[]), &hs(&["cuda-version"]), &hs(&[]), "gsi");
        assert_eq!(v.len(), 1);
        assert!(matches!(v[0], PerChainVerdict::AbiAnchor { .. }));
    }

    #[test]
    fn classify_chains_workspace_pin_dominates() {
        let chains = vec![chain_with(
            "pytorch-gpu",
            ">=2.7.1,<3",
            vec!["3.0"],
            "pytorch >=3",
        )];
        let v = classify_chains(&chains, &hs(&[]), &hs(&["pytorch-gpu"]), &hs(&[]), "gsi");
        assert_eq!(v.len(), 1);
        match &v[0] {
            PerChainVerdict::WorkspacePinDominates {
                dep,
                also_emitted_by_retread,
                ..
            } => {
                assert_eq!(dep, "pytorch-gpu");
                assert!(!also_emitted_by_retread);
            }
            other => panic!("expected WorkspacePinDominates, got {other:?}"),
        }
    }

    #[test]
    fn classify_chains_already_exhausted() {
        let chains = vec![chain_with(
            "torchvision",
            "*",
            vec!["0.25.0"],
            "pytorch >=2.10",
        )];
        let v = classify_chains(
            &chains,
            &hs(&["torchvision"]),
            &hs(&[]),
            &hs(&["torchvision"]),
            "gsi",
        );
        assert_eq!(v.len(), 1);
        assert!(matches!(v[0], PerChainVerdict::AlreadyExhausted { .. }));
    }

    #[test]
    fn classify_chains_transitive_only() {
        let chains = vec![chain_with("mystery", ">=1", vec!["1.0"], "")];
        let v = classify_chains(&chains, &hs(&[]), &hs(&[]), &hs(&[]), "gsi");
        assert_eq!(v.len(), 1);
        assert!(matches!(v[0], PerChainVerdict::TransitiveOnly { .. }));
    }

    /// The specific gsi failure that motivated v0.36.0: mixed chains
    /// with pytorch (WidenRetread) + python (AbiAnchor) + cuda-version
    /// (AbiAnchor) + pytorch-gpu (WorkspacePinDominates). Each must
    /// get the right verdict; the aggregate roll-up still calls it
    /// Class A because pytorch is widenable, but the refinement loop
    /// (downstream) won't blindly widen everything in the blocker
    /// list -- it iterates verdicts and ONLY widens WidenRetread.
    #[test]
    fn classify_chains_mixed_gsi_round4_scenario() {
        let chains = vec![
            chain_with("pytorch-gpu", ">=2.7.1,<3", vec![], ""),
            chain_with("pytorch", ">=1", vec![], ""),
            chain_with("python", "==3.11", vec![], ""),
            chain_with("cuda-version", "==12.8", vec![], ""),
        ];
        let v = classify_chains(
            &chains,
            &hs(&["pytorch", "python"]),
            &hs(&["pytorch-gpu", "python", "cuda-version"]),
            &hs(&[]),
            "gsi",
        );
        assert_eq!(v.len(), 4);

        let pytorch = v.iter().find(|x| x.dep() == "pytorch").unwrap();
        assert!(
            matches!(pytorch, PerChainVerdict::WidenRetread { .. }),
            "pytorch should be widenable: {pytorch:?}",
        );

        let pytorch_gpu = v.iter().find(|x| x.dep() == "pytorch-gpu").unwrap();
        assert!(
            matches!(pytorch_gpu, PerChainVerdict::WorkspacePinDominates { .. }),
            "pytorch-gpu should be workspace-dominated: {pytorch_gpu:?}",
        );

        let python = v.iter().find(|x| x.dep() == "python").unwrap();
        assert!(
            matches!(python, PerChainVerdict::AbiAnchor { .. }),
            "python MUST be ABI anchor (the bug): {python:?}",
        );

        let cuda_version = v.iter().find(|x| x.dep() == "cuda-version").unwrap();
        assert!(
            matches!(cuda_version, PerChainVerdict::AbiAnchor { .. }),
            "cuda-version MUST be ABI anchor: {cuda_version:?}",
        );

        // Only ONE verdict is widenable; the cascade has work but
        // must not touch the other three.
        let widenable_count = v.iter().filter(|x| x.is_widenable()).count();
        assert_eq!(widenable_count, 1, "exactly one widenable verdict");
    }

    // ---------- summarize + aggregate roll-up ----------

    #[test]
    fn summarize_verdicts_renders_mixed_chains() {
        let verdicts = vec![
            PerChainVerdict::WidenRetread {
                dep: "pytorch".into(),
                current_spec: ">=1".into(),
            },
            PerChainVerdict::AbiAnchor {
                dep: "python".into(),
                reason: "abi".into(),
            },
        ];
        let s = summarize_verdicts(&verdicts);
        assert!(s.contains("pytorch (retread will widen)"));
        assert!(s.contains("python (ABI anchor -- NEVER widened)"));
    }

    #[test]
    fn aggregate_class_a_when_any_widenable() {
        let chains = vec![
            chain_with("triton", ">=3.7.0,<3.8", vec![], ""),
            chain_with("python", "==3.11", vec![], ""),
        ];
        let out = classify_unsat(&chains, &hs(&["triton"]), &hs(&["python"]), &hs(&[]), "gsi");
        assert_eq!(out.class, ConflictClass::A);
    }

    #[test]
    fn aggregate_class_b_when_no_widenable_but_abi_anchor() {
        let chains = vec![chain_with("python", "==3.11", vec![], "")];
        let out = classify_unsat(&chains, &hs(&["python"]), &hs(&["python"]), &hs(&[]), "gsi");
        assert_eq!(out.class, ConflictClass::BWorkspaceDominated);
    }

    #[test]
    fn aggregate_class_b_when_workspace_dominates() {
        let chains = vec![chain_with(
            "torchvision",
            ">=0.22.0",
            vec!["0.25.0", "0.26.0"],
            "pytorch >=2.10.0,<2.11.0a0",
        )];
        let out = classify_unsat(
            &chains,
            &hs(&["torchvision"]),
            &hs(&["torchvision", "pytorch-gpu"]),
            &hs(&[]),
            "gsi",
        );
        assert_eq!(out.class, ConflictClass::BWorkspaceDominated);
        assert_eq!(out.suggestions.len(), 1);
        assert!(
            out.suggestions[0].suggested_pin.contains("<0.25"),
            "suggested_pin = {}",
            out.suggestions[0].suggested_pin,
        );
    }

    #[test]
    fn suggestion_falls_back_when_every_version_rejected() {
        let chains = vec![chain_with(
            "torchvision",
            ">=0.22.0",
            vec![
                "0.22.0", "0.22.1", "0.23.0", "0.24.0", "0.24.1", "0.25.0", "0.26.0",
            ],
            "pytorch >=2.10.0,<2.11.0a0",
        )];
        let out = classify_unsat(
            &chains,
            &hs(&["torchvision"]),
            &hs(&["torchvision", "pytorch-gpu"]),
            &hs(&[]),
            "gsi",
        );
        assert_eq!(out.class, ConflictClass::BWorkspaceDominated);
        let sug = &out.suggestions[0];
        assert!(
            !sug.suggested_pin.contains("<0.22"),
            "suggested_pin should NOT be empty-range `<0.22`: {}",
            sug.suggested_pin,
        );
        assert!(sug.suggested_pin.contains("remove") || sug.suggested_pin.contains("relax"),);
        assert!(sug.suggested_pin.contains("pytorch"));
    }

    #[test]
    fn aggregate_class_c_when_only_transitive() {
        let chains = vec![chain_with("mystery", ">=1", vec![], "")];
        let out = classify_unsat(&chains, &hs(&[]), &hs(&[]), &hs(&[]), "gsi");
        assert_eq!(out.class, ConflictClass::CWorkspaceOnly);
    }

    #[test]
    fn lower_bound_of_extracts_ge_clause() {
        assert_eq!(lower_bound_of(">=0.22.0"), Some(">=0.22.0".into()));
        assert_eq!(lower_bound_of(">=0.22,<0.25"), Some(">=0.22".into()));
        assert_eq!(lower_bound_of("*"), None);
    }

    #[test]
    fn lowest_version_orders_dotted_numeric() {
        let v = vec![
            "1.10.0".to_string(),
            "1.2.0".to_string(),
            "1.9.0".to_string(),
        ];
        assert_eq!(lowest_version(&v), Some("1.2.0".to_string()));
    }

    #[test]
    fn minor_of_strips_patch() {
        assert_eq!(minor_of("0.25.0"), Some("0.25".into()));
        assert_eq!(minor_of("2.10.0a1"), Some("2.10".into()));
        assert_eq!(minor_of("3"), None);
    }

    #[test]
    fn gigastrap_torchvision_pytorch_fixture() {
        // The v0.35.0 fixture, still valid -- per-chain verdict
        // approach produces WorkspacePinDominates for torchvision.
        let unsat = vec![
            "The following packages are incompatible\n\
             └─ torchvision >=0.22.0 cannot be installed because there are no viable options:\n   \
                ├─ torchvision 0.25.0 | 0.25.0 | 0.26.0 would require\n   \
                │  └─ pytorch >=2.10.0,<2.11.0a0, which cannot be installed because there are no viable options:\n   \
                │     └─ pytorch 2.10.0 would require\n   \
                │        └─ cuda-version >=12.9,<13, for which no candidates were found."
                .to_string(),
        ];
        let chains = crate::solve_check::extract_blocking_chains(&unsat);
        let verdicts = classify_chains(
            &chains,
            &hs(&["torchvision", "pytorch"]),
            &hs(&["torchvision", "pytorch-gpu"]),
            &hs(&[]),
            "gsi",
        );
        let tv = verdicts.iter().find(|v| v.dep() == "torchvision").unwrap();
        assert!(matches!(tv, PerChainVerdict::WorkspacePinDominates { .. }));
    }
}
