//! User-facing configuration for the retread backend.
//!
//! Lives under `[build.config]` in the consumer's `pixi.toml`.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::relax::{CondaName, CondaTarget, NameMap, PypiKey};

fn deserialize_name_map<'de, D>(deserializer: D) -> std::result::Result<NameMap, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = BTreeMap::<String, String>::deserialize(deserializer)?;
    let mut canonical = NameMap::new();
    let mut source_keys = BTreeMap::<PypiKey, String>::new();

    for (raw_key, raw_target) in raw {
        let key = PypiKey::from_pypi(&raw_key);
        if raw_key != key.as_str() {
            tracing::warn!(
                key = %raw_key,
                canonical = %key,
                "retread-name-map key is non-canonical; using its canonical PyPI key",
            );
        }

        let target = if raw_target.is_empty() {
            CondaTarget::Disabled
        } else {
            CondaTarget::Mapped(CondaName::new(raw_target))
        };

        if let Some(existing) = canonical.get(&key) {
            let existing_key = source_keys
                .get(&key)
                .expect("source key accompanies every canonical name-map entry");
            if existing == &target {
                tracing::warn!(
                    canonical = %key,
                    first_key = %existing_key,
                    duplicate_key = %raw_key,
                    target = %target,
                    "duplicate retread-name-map aliases have the same target; deduplicating",
                );
                continue;
            }

            return Err(serde::de::Error::custom(format!(
                "conflicting retread-name-map entries `{existing_key}` = `{existing}` and \
                 `{raw_key}` = `{target}` canonicalize to the same key `{key}`"
            )));
        }

        source_keys.insert(key.clone(), raw_key);
        canonical.insert(key, target);
    }

    Ok(canonical)
}

/// PyPI packages that are Windows-only and frequently declared as
/// unconditional `Requires-Dist` lines by upstream packagers (notably the
/// Isaac Sim wheels). Auto-dropped from run-deps when the target platform
/// isn't Windows, so users don't have to enumerate them in retread-drop-deps.
///
/// Consumed by BOTH resolution paths so there is a single source of truth:
/// the conda run-dep path (`handler::produce_output`) filters these out of
/// the emitted `depends`, and the v4 uv-closure path
/// (`handler::uv_group_closure`) injects them as unmatchable-marker
/// override-dependencies (the same mechanism as `retread-drop-deps`) so uv
/// never resolves them. NVIDIA's index strips the `sys_platform == "win32"`
/// marker from these lines, so the marker-pruning path can't catch them.
///
/// Inclusion criteria: ships wheels only for `win_*` platforms OR the
/// package is exclusively a shim for a Windows-specific subsystem (Win32
/// API, COM, registry, ANSI terminal compat, etc.) such that running it
/// on non-Windows is meaningless. Cross-platform packages (colorama,
/// chardet) do NOT belong here — they just happen to be most-cited on
/// Windows.
pub const BUILT_IN_WIN_ONLY: &[&str] = &[
    "comtypes",       // COM bindings
    "idna-ssl",       // async SSL shim, last release 2017
    "pyreadline",     // readline replacement (deprecated)
    "pyreadline3",    // readline replacement (current)
    "pywin32",        // Win32 API bindings
    "pywin32-ctypes", // ctypes-only fallback for pywin32
    "pywinpty",       // Windows pseudo-terminal (jupyter, IPython)
    "win32-setctime", // ctime setter for Windows files
    "winregistry",    // registry helper (stdlib winreg on Windows)
    "winrt-runtime",  // Windows Runtime API
    "winshell",       // shell helpers
    "wmi",            // Windows Management Instrumentation
];

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct RetreadConfig {
    /// Wheels to repack, keyed by package name. The map key serves two
    /// purposes:
    /// 1. For [`WheelEntry::is_spec`] form, it is the PyPI distribution
    ///    name to resolve on the simple index.
    /// 2. For [`WheelEntry::is_url`] form, it is the name of the resulting
    ///    conda package.
    ///
    /// Example matching pixi's `[pypi-dependencies]` syntax:
    /// ```toml
    /// [build.config.retread-wheels]
    /// isaacsim = { version = "==5.1.0", index = "https://pypi.nvidia.com", extras = ["all", "extscache"] }
    /// mujoco   = { version = "==3.5.0", index = "https://py.mujoco.org" }
    /// foo      = { url = "https://example.com/foo-1.whl", sha256 = "..." }   # direct URL fallback
    /// ```
    #[serde(rename = "retread-wheels", alias = "wheels")]
    pub retread_wheels: BTreeMap<String, WheelEntry>,

    /// How aggressively to widen dependency pins from the wheel's METADATA.
    ///
    /// - `none`: keep pins as-is (== stays ==)
    /// - `patch`: ==X.Y.Z -> >=X.Y.Z,<X.Y+1
    /// - `minor`: ==X.Y.Z -> >=X.Y,<X+1
    /// - `major`: ==X.Y.Z -> >=X (drop upper bound)
    /// - `patch-then-minor-then-major-then-last-resort` (default,
    ///   v0.30.0+): emit at patch initially; the cascade widens
    ///   progressively on solve failure. See "Relax policies" in
    ///   the README.
    #[serde(default, rename = "retread-relax", alias = "relax")]
    pub relax: RelaxPolicy,

    /// Per-dependency overrides, applied after the relax policy. Map of
    /// PyPI name -> conda match-spec (e.g. `"*"`, `">=2.7"`).
    #[serde(default, rename = "retread-overrides", alias = "overrides")]
    pub overrides: BTreeMap<String, String>,

    /// PyPI -> conda name mapping overrides on top of the built-in identity
    /// mapping. Use for the common drift cases (`opencv-python-headless` ->
    /// `py-opencv`, etc.).
    #[serde(
        default,
        deserialize_with = "deserialize_name_map",
        rename = "retread-name-map",
        alias = "name-map",
        alias = "name_map"
    )]
    pub name_map: NameMap,

    /// Shared-library payload replacements applied by the courier installer
    /// after uv installs the wheel payload. Keys are site-packages-relative
    /// path patterns; v1 supports exactly `conda-lib`, meaning replace the
    /// vendored file with a relative symlink to `$PREFIX/lib/<SONAME>`.
    #[serde(
        default,
        rename = "retread-shadow-libs",
        alias = "shadow-libs",
        alias = "shadow_libs"
    )]
    pub shadow_libs: BTreeMap<String, ShadowPolicy>,

    /// PyPI names to drop from the conda run-deps entirely. Use for
    /// upstream-pinned deps that don't exist on the target conda channel
    /// (Windows-only shims like `idna-ssl`, `pywin32`) or otherwise can't
    /// be satisfied. The wheel still gets installed; conda just won't
    /// require these at solve time.
    #[serde(
        default,
        rename = "retread-drop-deps",
        alias = "drop-deps",
        alias = "drop_deps"
    )]
    pub drop_deps: Vec<String>,

    /// When true (default), retread tries to resolve every exact-pinned
    /// transitive `Requires-Dist` line on the entry's PyPI index (with
    /// public PyPI as fallback). If a wheel exists, it's pip-installed
    /// into the conda package and dropped from the conda run-deps.
    /// Eliminates the need to manually `retread-overrides` /
    /// `retread-drop-deps` every PyPI package that isn't on conda-forge
    /// (`aiodns`, `qdldl`, ...).
    ///
    /// Deps in `retread-conda-deps` are never auto-bundled -- they
    /// always emit as conda run-deps. Set this to `false` to disable
    /// auto-bundling entirely.
    #[serde(
        default = "default_true",
        rename = "retread-auto-bundle",
        alias = "auto-bundle"
    )]
    pub auto_bundle: bool,

    /// PyPI names that must stay as conda run-deps even when
    /// `retread-auto-bundle` is on. Use for ABI-sensitive packages where
    /// installing both a bundled wheel and the conda-channel version
    /// causes a collision -- typically the scientific stack (`numpy`,
    /// `scipy`, `pytorch`, `pandas`, ...). Empty by default; retread
    /// intentionally does NOT hard-code a list.
    #[serde(default, rename = "retread-conda-deps", alias = "conda-deps")]
    pub conda_deps: Vec<String>,

    /// v1.4.0: default `bundle` group for every `[retread-wheels]`
    /// entry that doesn't set its own `bundle` field. Saves repeating
    /// `bundle = "<name>"` on every entry of a single-output pack:
    ///
    /// ```toml
    /// [package.build.config]
    /// retread-bundle = "isaac-pack"
    /// ```
    ///
    /// A per-entry `bundle` still wins, so mixed grouping (some
    /// entries in the default group, some elsewhere or standalone)
    /// stays expressible. Unset preserves the historical behavior:
    /// entries without `bundle` each produce their own conda output.
    #[serde(default, rename = "retread-bundle", alias = "default-bundle")]
    pub default_bundle: Option<String>,

    /// Named git sources, referenced from `[retread-wheels]` entries
    /// via `from = "<name>"`. Avoids repeating `git = "..."` + `rev =
    /// "..."` across many sub-package entries from the same repo.
    /// Example:
    ///
    /// ```toml
    /// [package.build.config.retread-git-sources]
    /// isaaclab = { url = "https://github.com/isaac-sim/IsaacLab.git", rev = "deadbeef" }
    ///
    /// [package.build.config.retread-wheels]
    /// isaaclab        = { from = "isaaclab", subdirectory = "source/isaaclab" }
    /// isaaclab-assets = { from = "isaaclab", subdirectory = "source/isaaclab_assets" }
    /// ```
    #[serde(default, rename = "retread-git-sources", alias = "git-sources")]
    pub git_sources: BTreeMap<String, NamedGitSource>,

    /// Conda build number for the produced packages. Bump to force
    /// re-resolution downstream after a policy change.
    #[serde(default, rename = "retread-build-number", alias = "build-number")]
    pub build_number: u64,

    /// v1.5.8: zstd compression level for the produced .conda
    /// artifact, forwarded to rattler-build as
    /// `--package-format conda:<level>`. Unset uses rattler-build's
    /// default (high). For locally-consumed packs full of
    /// shared-library / asset payloads that barely compress (Isaac
    /// Sim, extscache), a LOW level (1-3) cuts the packaging stage of
    /// pixi's "preparing packages" phase dramatically for a few
    /// percent more disk:
    ///
    /// ```toml
    /// [package.build.config]
    /// retread-compression-level = 1
    /// ```
    #[serde(
        default,
        rename = "retread-compression-level",
        alias = "compression-level"
    )]
    pub compression_level: Option<u32>,

    /// Number of rattler-build compression threads for this pack.
    /// The process-wide `RETREAD_COMPRESSION_THREADS` environment override
    /// takes precedence. Unset shares the node-wide compression budget with
    /// other active retread builds.
    #[serde(
        default,
        rename = "retread-compression-threads",
        alias = "compression-threads"
    )]
    pub compression_threads: Option<NonZeroUsize>,

    /// DEPRECATED (v2.0.0), parsed and ignored. Use `retread-courier`
    /// instead. This field is retained so `deny_unknown_fields` does not
    /// reject old manifests that still carry it -- it parses but has no
    /// effect. Remove it from your `[package.build.config]`.
    #[serde(default, rename = "retread-emit-pypi", alias = "emit-pypi")]
    pub emit_pypi: bool,

    /// v2.1.0: courier mode -- the DEFAULT. Emit a metadata-only conda
    /// package that ships the bundle's built/relax-changed wheels + a
    /// committed `retread-<bundle>.lock.json` as data, declares the solved
    /// conda run-deps plus `uv`, and installs the PyPI wheels at env link
    /// time via a post-link `retread install` (uv hardlink; index wheels
    /// fetched on demand). The consumer keeps ONE clean conda line and zero
    /// machine-written manifest bytes; nothing is committed to git.
    ///
    /// Defaults to `true`. Set `retread-courier = false` to opt OUT to the
    /// legacy conda-artifact path (a full conda package carrying the wheels
    /// as a payload). In the default `courier-mode = "post-link"`, courier
    /// wants the consumer workspace to enable post-link scripts
    /// (`<workspace>/.pixi/config.toml` with
    /// `run-post-link-scripts = "insecure"`) so the wheels are installed
    /// eagerly at `pixi install` time; without it, `pixi install` SKIPS the
    /// post-link (pixi refuses to run it) but the env is not left broken --
    /// the shipped conda `activate.d` guard runs `retread verify`/`install`
    /// on every activation regardless of the post-link toggle, and installs
    /// the wheels lazily on first `pixi run`/`pixi shell` instead. Set
    /// `courier-mode = "activation"` to skip emitting the post-link script
    /// entirely and rely solely on that lazy activate.d install -- this
    /// drops the `run-post-link-scripts = "insecure"` requirement at the
    /// cost of a slower first activation (see [`CourierMode`]).
    ///
    /// ```toml
    /// [package.build.config]
    /// retread-courier = false  # opt out (legacy artifact path)
    /// ```
    #[serde(
        default = "default_true",
        rename = "retread-courier",
        alias = "courier"
    )]
    pub courier: bool,

    /// v4.3.0: how the courier bundle installs its PyPI wheel payload into
    /// the consumer env (`retread-courier-mode`).
    ///
    /// - `"post-link"` (the DEFAULT): install eagerly at env link time via a
    ///   conda post-link script (`retread install` runs during `pixi
    ///   install`). Requires the consumer's `<workspace>/.pixi/config.toml`
    ///   to set `run-post-link-scripts = "insecure"` for the install to run
    ///   at link time (see [`RetreadConfig::courier`] doc); without it, the
    ///   activate.d guard still installs lazily on first activation.
    /// - `"activation"`: never emit a post-link script at all. The wheels
    ///   are installed lazily by the same `retread install` call, but only
    ///   the first time the env is activated (`pixi run` / `pixi shell`).
    ///   No `run-post-link-scripts = "insecure"` config is needed anywhere.
    ///   Trade-off: `pixi install` finishes faster (no wheel install), but
    ///   the first `pixi run`/`pixi shell` after install pays that cost
    ///   instead, blocking on the install synchronously.
    ///
    /// ```toml
    /// [package.build.config]
    /// retread-courier-mode = "activation"  # no insecure config needed
    /// ```
    #[serde(default, rename = "retread-courier-mode", alias = "courier-mode")]
    pub courier_mode: CourierMode,

    /// Which packages the auto-route loop may move from the wheel closure
    /// to conda run-deps, and how conda candidates are validated.
    ///
    /// - `"prefer-conda-validated"` (the DEFAULT): consider every closure
    ///   wheel for conda routing, but route only when an available conda
    ///   version satisfies the wheel requirement and remains compatible
    ///   with the consuming workspace's declared conda facts.
    /// - `"minimal"`: apply the same validation, but cap eligibility at the
    ///   ABI/binary whitelist --
    ///   python/python_abi, the torch family (torch/pytorch/pytorch-gpu/
    ///   pytorch-cpu/torchvision/torchaudio, checked on both the pypi and
    ///   the mapped conda name), the cuda-* family (cuda-version,
    ///   cudatoolkit, cuda-toolkit, ...) -- plus anything the pack lists
    ///   in `retread-route-include` or `force-conda`.
    /// - `"aggressive"`: legacy broad-eligibility behavior -- consider any
    ///   closure wheel whose name exists on the conda channels.
    ///
    /// ```toml
    /// [package.build.config]
    /// retread-route-policy = "aggressive"  # opt back into legacy routing
    /// ```
    #[serde(default, rename = "retread-route-policy", alias = "route-policy")]
    pub route_policy: RoutePolicy,

    /// Extra canonical PyPI names admitted for routing under the
    /// `"minimal"` policy, in addition to the built-in ABI/binary
    /// whitelist. The default `"prefer-conda-validated"` policy already
    /// considers every name, but still validates requirements and workspace
    /// conda facts; this list does not bypass that validation. Ignored under
    /// `"aggressive"`, where every conda-available name is already eligible.
    ///
    /// ```toml
    /// [package.build.config]
    /// retread-route-include = ["mujoco"]
    /// ```
    #[serde(default, rename = "retread-route-include", alias = "route-include")]
    pub route_include: Vec<String>,

    /// v3.2.0: where the courier bundle's shipped wheel BYTES live.
    ///
    /// - `"loose"` (the DEFAULT): the produced .conda is a KB-sized stub
    ///   (lock + meta-wheel + installer binary + post-link script). The
    ///   bundle's built/shadow wheels are persisted at build time to the
    ///   shared content-addressed wheel store
    ///   (`<wheel store root>/<sha256>/<filename>`, see
    ///   [`crate::courier::retread_wheel_store_root`]; fast-tmp-exempt, and
    ///   the lock records the store root used) plus their sha256;
    ///   `retread install` materializes them from the store
    ///   (hash-verified) instead of from the package payload. Wheel bytes
    ///   are never tarred into the conda artifact, dedupe across
    ///   packs/rebuilds, and packaging drops from minutes to seconds.
    ///   OFFLINE STORY: loose + warm store = fully offline install; loose +
    ///   cold store needs network to the locked index URLs, and a
    ///   store-evicted BUILT wheel requires a pack rebuild to re-populate.
    /// - `"fat"`: legacy self-contained artifact -- built/shadow wheels ship
    ///   inside the .conda. Use when publishing to a real channel whose
    ///   consumers do not share a wheel store with the build machine.
    ///
    /// ```toml
    /// [package.build.config]
    /// retread-bundle-mode = "fat"   # opt out of the loose default
    /// ```
    #[serde(default, rename = "retread-bundle-mode", alias = "bundle-mode")]
    pub bundle_mode: BundleMode,

    /// DEPRECATED (v2.0.0), parsed and ignored. Use `retread-courier`
    /// instead. This field is retained so `deny_unknown_fields` does not
    /// reject old manifests that still carry it -- it parses but has no
    /// effect. Remove it from your `[package.build.config]`.
    #[serde(default, rename = "retread-blueprint", alias = "blueprint")]
    pub blueprint: BlueprintMode,

    /// DEPRECATED (v2.0.0), parsed and ignored. Was `retread-blueprint-sync`
    /// (`"script"` | `"fence"`). Both the blueprint fence path and the
    /// install-script path have been superseded by `retread-courier`. The
    /// field is retained so `deny_unknown_fields` does not reject old
    /// manifests -- it parses but has no effect. Remove it from your
    /// `[package.build.config]`.
    #[serde(default, rename = "retread-blueprint-sync", alias = "blueprint-sync")]
    pub blueprint_sync: Option<String>,

    /// When true, fold the exact retread version into the courier inputs_hash
    /// so EVERY retread release forces a cold re-solve (strict reproducibility).
    /// Default false: replay is gated on EMIT_EPOCH, so routine retread upgrades
    /// reuse the committed lock. See lock::EMIT_EPOCH.
    #[serde(default, rename = "retread-pin-version", alias = "pin-version")]
    pub pin_version: bool,

    /// Python version(s) to build for, as a fallback when the workspace
    /// does not declare `[workspace.build-variants] python = [...]`.
    ///
    /// Accepts either a single string ("3.11") or a list of strings
    /// (`["3.11", "3.12"]`). When the workspace's variant configuration
    /// provides `python`, that wins; this field is purely a convenience for
    /// single-Python workspaces. The default is `3.11`.
    ///
    /// Also used as the fallback in conda/build_v1 when pixi forwards a
    /// bare-major variant value (`"3"`) — see handler.rs::conda_build_v1.
    /// The bare name `python` is also accepted as a legacy alias.
    #[serde(default, rename = "retread-python", alias = "python")]
    pub python: Option<PythonSpec>,

    /// DEPRECATED (v4.4.0), parsed and ignored. uv is the ONLY resolver:
    /// the bundle's wheel closure is always computed by a uv subprocess
    /// (ephemeral project + `uv lock` + `uv export --format pylock.toml`).
    /// The old `retread-resolver = "legacy"` cascade/resolvo mirror-solver
    /// was deleted in v4.2.0 and the knob itself removed in v4.4.0. This
    /// field is retained so `deny_unknown_fields` does not reject old
    /// manifests that still carry `retread-resolver`; `initialize` emits a
    /// deprecation warning when it is set. Remove it from your
    /// `[package.build.config]`.
    #[serde(default, rename = "retread-resolver", alias = "resolver")]
    pub resolver: Option<String>,

    /// v4.3.0 (spec-uv-restructure M2): conda-route auto-discovery.
    /// After `uv lock` yields the wheel closure, every closure wheel
    /// whose conda equivalent (name-mapped; identity fallback) exists on
    /// the workspace's channels AT THE RESOLVED VERSION is moved to the
    /// conda side: excluded from the shipped closure, pinned as a uv
    /// constraint, and emitted as a conda run-dep of the stub package.
    /// The lock is then re-run and the sweep repeats to fixpoint (max 5
    /// rounds). Roots and retread-built wheels are never routed.
    /// Default `true`; set `auto-route = false` to ship everything uv
    /// resolves as PyPI wheels (minus force-listed `retread-conda-deps`).
    #[serde(
        default = "default_true",
        rename = "retread-auto-route",
        alias = "auto-route"
    )]
    pub auto_route: bool,

    /// v4.3.0: PyPI names auto-route must NEVER move to conda, even when
    /// a channel has a matching version. Use for packages whose conda
    /// build differs materially from the wheel (patched builds, split
    /// outputs, optional native features). Only affects auto-route;
    /// `retread-conda-deps` (force TO conda) still wins for names listed
    /// there.
    #[serde(default, rename = "retread-keep-pypi", alias = "keep-pypi")]
    pub keep_pypi: Vec<String>,

    /// Inverse of `retread-keep-pypi` for the self-healing un-route
    /// step: PyPI names whose auto-route decision must stick even when
    /// the co-installability solve names them in an unsat core. The
    /// un-route step skips them (with a warning), so a persisting
    /// conflict fails the workspace lock loudly instead of silently
    /// reverting the package to its wheel. Does NOT force a route into
    /// existence — the package still needs a repodata hit (use
    /// `retread-conda-deps` to hard-force a conda dep).
    #[serde(default, rename = "retread-force-conda", alias = "force-conda")]
    pub force_conda: Vec<String>,

    /// v4.4.0: third rung of the sdist-only self-heal ladder (wheel ->
    /// conda-route -> sdist auto-build -> error). When a package has no
    /// wheels at all AND no conda channel carries any version of it,
    /// `"auto"` (default) builds it from its PyPI sdist through the
    /// same machinery git-sourced `[retread-wheels]` entries use, caches
    /// it content-addressed in the shared wheel store, and records it in
    /// the bundle lock with the same provenance a git-built wheel gets.
    /// `"never"` disables the build rung: a conda-route miss surfaces
    /// the original uv error + guidance immediately, exactly as before
    /// this rung existed.
    #[serde(default, rename = "retread-sdist-build", alias = "sdist-build")]
    pub sdist_build: SdistBuildPolicy,

    /// Extra root PyPI dependencies pulled from an externally-maintained
    /// requirements, PEP 621 pyproject, or conda environment file, layered on
    /// top of (or instead of)
    /// `[retread-wheels]` entries. Each source is fetched
    /// (`deps_from::fetch_dep_source`) and parsed
    /// (`deps_from::parse_dep_source`) into typed closure inputs. PEP 508
    /// dependencies become uv roots,
    /// which are added as roots to the bundle's uv closure alongside the
    /// `[retread-wheels]` roots (see `handler::uv_group_closure`).
    ///
    /// For PEP 621 files, `[tool.uv.sources]` is honored as source metadata.
    /// Dependencies mapped to local `path` sources are omitted with a warning
    /// whether the path is missing or exists: retread cannot portably ship an
    /// arbitrary local/editable source tree, and resolving the same bare name
    /// from a registry would select a different project. Relative paths are
    /// resolved against local files and git checkouts; raw URL sources have no
    /// local base and therefore omit relative path sources with a warning.
    ///
    /// `environment.yaml` / `.yml` sources are parsed structurally. Nested
    /// `pip:` entries become PEP 508 roots. Conda export entries discard
    /// channels and build strings and retain only safe `>=` advisory floors.
    /// Those floors never become conda run dependencies: an exported env is a
    /// machine-specific solved/transitive snapshot, not a direct dependency
    /// manifest. A floor may constrain an already-active bare PyPI root only
    /// through a one-to-one edge in this pack's explicit `retread-name-map`;
    /// workspace constraints, configured roots, overrides, and drops win.
    /// Unmapped floors remain inert and visible in debug diagnostics.
    ///
    /// Accepts three TOML shapes (a bare string, a git table, or a list
    /// mixing both -- union of all sources):
    ///
    /// ```toml
    /// [package.build.config]
    /// retread-deps-from = "requirements_isaaclab.txt"   # local path (relative to this pixi.toml)
    /// # or:
    /// retread-deps-from = "https://raw.githubusercontent.com/org/repo/main/requirements.txt"
    /// # or:
    /// retread-deps-from = { git = "https://github.com/org/repo", rev = "deadbeef", path = "requirements.txt" }
    /// # or a list mixing forms:
    /// retread-deps-from = [
    ///     "requirements_base.txt",
    ///     { git = "https://github.com/org/repo", rev = "deadbeef", path = "pyproject.toml" },
    /// ]
    /// ```
    ///
    /// Default: empty (feature off).
    #[serde(default, rename = "retread-deps-from", alias = "deps-from")]
    pub deps_from: DepsFromSpec,

    /// Names of `overrides` entries that were merged from the
    /// `.retread/auto-overrides.json` ledger (repair-engine-derived pypi
    /// overrides, see `pack_overrides::merge_ledger_overrides`) rather
    /// than hand-written in a `[retread-overrides]` table. Never part of
    /// the TOML surface (`serde(skip)`). Run-31 regression this guards:
    /// the conda run-dep emission exempts manually-overridden names from
    /// bounded-range widening ("hand-written intent wins"), but a
    /// ledgered override is a pypi-side steering knob the repair engine
    /// derived -- treating it as manual intent froze the conda pin back
    /// to an exact `==` the moment ANY repair landed for that package,
    /// resurrecting the exact-pin conflict class the ranges exist to
    /// prevent.
    #[serde(skip)]
    pub ledger_overrides: std::collections::BTreeSet<String>,

    /// Exact workspace-relative pack manifest used only for fail-closed
    /// diagnostics. Populated by the build-backend initialize boundary and
    /// excluded from user configuration and emitted artifact fingerprints.
    #[doc(hidden)]
    #[serde(skip)]
    pub pack_manifest_path: Option<String>,
}

/// `retread-deps-from`'s value: a bare source or a list of sources (union).
/// See [`RetreadConfig::deps_from`] for the accepted TOML shapes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DepsFromSpec(pub Vec<crate::deps_from::DepSource>);

impl DepsFromSpec {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_slice(&self) -> &[crate::deps_from::DepSource] {
        &self.0
    }
}

impl<'de> serde::Deserialize<'de> for DepsFromSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            One(crate::deps_from::DepSource),
            Many(Vec<crate::deps_from::DepSource>),
        }
        Ok(match Repr::deserialize(deserializer)? {
            Repr::One(d) => DepsFromSpec(vec![d]),
            Repr::Many(d) => DepsFromSpec(d),
        })
    }
}

impl Serialize for DepsFromSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

/// Per-pack policy for the sdist-only self-heal's build rung
/// (`sdist-build`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SdistBuildPolicy {
    /// Auto-build sdist-only packages with no conda candidate (default).
    #[default]
    Auto,
    /// Never build; a conda-route miss errors immediately (pre-v4.4.0
    /// behavior).
    Never,
}

/// Where the courier bundle's shipped wheel bytes live
/// (`retread-bundle-mode`): the shared content-addressed wheel store
/// (`loose`, default) or inside the .conda artifact (`fat`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BundleMode {
    /// Stub .conda; wheel bytes live in the shared wheel store (default).
    #[default]
    Loose,
    /// Self-contained .conda carrying the wheel payload (legacy; for
    /// channel publishing).
    Fat,
}

/// How the courier bundle installs its wheel payload (`retread-courier-mode`):
/// eagerly via a conda post-link script at env link time (`post-link`,
/// default), or lazily via the activate.d self-heal guard only, on first
/// activation (`activation` -- no post-link, no `insecure` config needed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CourierMode {
    /// Install eagerly at env link time via a conda post-link script.
    /// Wants `run-post-link-scripts = "insecure"` in the consumer's
    /// `.pixi/config.toml` for the install to happen at `pixi install` time.
    #[default]
    PostLink,
    /// Skip the post-link script entirely; the activate.d guard installs
    /// the wheels lazily on first `pixi run` / `pixi shell`. No `insecure`
    /// post-link config required.
    Activation,
}

/// Which packages the auto-route loop may move to conda
/// (`retread-route-policy`): any requirement- and workspace-compatible
/// candidate (`prefer-conda-validated`, default), the ABI/binary whitelist
/// ceiling (`minimal`), or the legacy broad conda-name eligibility
/// (`aggressive`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RoutePolicy {
    /// Prefer conda broadly, but only after the candidate satisfies the wheel
    /// requirement and the consuming workspace's declared conda facts.
    #[default]
    PreferCondaValidated,
    /// Route only the ABI/binary whitelist (python/python_abi, torch
    /// family, cuda-* family) plus `retread-route-include`/`force-conda`
    /// entries, subject to the same validation as the default policy.
    Minimal,
    /// Legacy broad eligibility: consider any closure wheel whose name exists
    /// on conda.
    Aggressive,
}

/// `retread-blueprint` accepts `false` (off), `true` (blueprint +
/// full conda artifact), or `"only"` (blueprint + stub conda
/// artifact).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(from = "BlueprintModeRepr")]
pub enum BlueprintMode {
    #[default]
    Off,
    Full,
    Only,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum BlueprintModeRepr {
    Flag(bool),
    Mode(String),
}

impl From<BlueprintModeRepr> for BlueprintMode {
    fn from(r: BlueprintModeRepr) -> Self {
        match r {
            BlueprintModeRepr::Flag(false) => Self::Off,
            BlueprintModeRepr::Flag(true) => Self::Full,
            BlueprintModeRepr::Mode(s) if s.eq_ignore_ascii_case("only") => Self::Only,
            BlueprintModeRepr::Mode(s) => {
                // deny_unknown_fields can't see inside the value; be loud
                // rather than silently off.
                tracing::warn!(value = %s, "unknown retread-blueprint value; treating as true");
                Self::Full
            }
        }
    }
}

impl BlueprintMode {
    pub fn is_on(self) -> bool {
        !matches!(self, Self::Off)
    }
    pub fn is_only(self) -> bool {
        matches!(self, Self::Only)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum PythonSpec {
    One(String),
    Many(Vec<String>),
}

impl PythonSpec {
    pub fn as_versions(&self) -> Vec<String> {
        match self {
            Self::One(v) => vec![v.clone()],
            Self::Many(v) => v.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ShadowPolicy {
    CondaLib,
}

impl ShadowPolicy {
    pub fn as_lock_value(self) -> &'static str {
        match self {
            ShadowPolicy::CondaLib => "conda-lib",
        }
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RelaxPolicy {
    None,
    Patch,
    Minor,
    Major,
    /// Major widening AND aggressive range-spec relaxing: drops every
    /// upper bound on every requirement (`<X`, `<=X`, the `<Y` half of
    /// `>=X,<Y`, the implicit upper of `~=X.Y`). Lower bounds (`>=`,
    /// `>`) stay so conda doesn't pick something pre-historic. Use
    /// when upstream caps are blocking the conda solve and you trust
    /// the conda side to pick a working version.
    #[serde(rename = "strong-major")]
    StrongMajor,
    /// "With-last-resort" family. Translation still behaves exactly like the
    /// named base policy. In the normal auto-bundle restore path, an otherwise
    /// empty intersection of index-wheel metadata requirements is retried at
    /// the base tier and then at the existing StrongMajor transform: exact
    /// pins become major floors and range upper caps are stripped. The first
    /// satisfiable tier wins and emits a loud, source-rich warning. Already
    /// satisfiable requirements and non-wheel authoritative constraints are
    /// never loosened.
    #[serde(rename = "patch-with-last-resort")]
    PatchWithLastResort,
    #[serde(rename = "minor-with-last-resort")]
    MinorWithLastResort,
    #[serde(rename = "major-with-last-resort")]
    MajorWithLastResort,
    /// Tiered inline conflict recovery. Translation begins at Patch. When the
    /// raw intersection of collected index-wheel metadata is empty in the
    /// auto-bundle restore path, retread retries Patch, Minor, Major, then the
    /// StrongMajor last-resort transform (major floors plus upper-cap
    /// stripping). It stops at the narrowest satisfiable tier and loudly
    /// reports every changed pin/cap with all wheel sources. If no tier can
    /// reconcile the metadata with untouched hard constraints, the original
    /// typed conflict is returned.
    #[default]
    #[serde(rename = "patch-then-minor-then-major-then-last-resort")]
    PatchThenMinorThenMajorThenLastResort,
    /// TODO(conda-aware): NOT YET IMPLEMENTED. The variant deserializes
    /// and accepts the value `"conda-aware"` from user config, but the
    /// probe layer described below does not exist -- at translate time
    /// this currently behaves IDENTICALLY to `StrongMajor` (strips every
    /// upper bound unconditionally). Do not document this option in the
    /// README until the probe is wired up. For "strict by default, widen
    /// only when needed" semantics today, use `minor` + per-package
    /// `retread-overrides` entries, OR use `minor-with-last-resort`
    /// which automates the widening for unsatisfiable deps.
    ///
    /// Intended design when implemented: per-dep adaptive widening.
    /// Starts at major (exact pins widen, ranges pass through). Then
    /// for each emitted spec containing an upper bound (`<`, `<=`,
    /// `~=`), retread probes the workspace's conda channels: if zero
    /// candidates satisfy the spec under the workspace's python,
    /// retread strips the upper bound and re-emits for that one dep.
    /// Specs without upper bounds skip the probe. All decisions land
    /// in `retread-audit.json` under `probe_results[]`.
    #[serde(rename = "conda-aware")]
    CondaAware,
}

impl RelaxPolicy {
    /// True for any `*-with-last-resort` variant.
    pub fn has_last_resort(self) -> bool {
        matches!(
            self,
            RelaxPolicy::PatchWithLastResort
                | RelaxPolicy::MinorWithLastResort
                | RelaxPolicy::MajorWithLastResort,
        )
    }

    /// True for the tiered Patch -> Minor -> Major -> last-resort policy.
    pub fn has_tiered_cascade(self) -> bool {
        matches!(self, RelaxPolicy::PatchThenMinorThenMajorThenLastResort)
    }

    /// True when the policy permits adaptive widening beyond its translation
    /// tier after an unsatisfiable result.
    pub fn allows_widening_mutation(self) -> bool {
        self.has_last_resort() || self.has_tiered_cascade()
    }
}

/// Either a direct URL ({url, sha256?}) or a PyPI-style spec
/// ({version, index?, extras?}). Validated by [`WheelEntry::validate`] at
/// initialize time.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct WheelEntry {
    // ---- URL form ----
    /// Direct URL to a `.whl`. When set, all other fields are ignored except
    /// `sha256` (which becomes a verification check).
    #[serde(default)]
    pub url: Option<url::Url>,

    /// SHA-256 of the wheel. Used for verification if `url` is set, or as a
    /// pinning anchor for [`WheelEntry::is_spec`] form (rare).
    #[serde(default)]
    pub sha256: Option<String>,

    // ---- Spec form ----
    /// PEP 440 version. Accepts both `5.1.0` and `==5.1.0` (the leading
    /// `==` is stripped). Only exact pins are supported by the resolver
    /// today; range syntax (`>=5.1,<6`) is planned.
    #[serde(default)]
    pub version: Option<String>,

    /// PEP 503 simple index URL. Defaults to PyPI public.
    #[serde(default)]
    pub index: Option<String>,

    /// Extras to follow when expanding this wheel into its transitive
    /// dependency set. Each named extra adds every `Requires-Dist: name ;
    /// extra == "X"` line from the wheel's METADATA to the wheel set,
    /// resolved against the same index.
    #[serde(default)]
    pub extras: Vec<String>,

    // ---- Local source form ----
    /// Local path to a Python project. retread runs `pip wheel --no-deps`
    /// against this path to produce a wheel, then runs it through the
    /// usual METADATA-rewrite + bundle pipeline. Relative paths resolve
    /// against the source package's pixi.toml directory.
    #[serde(default)]
    pub path: Option<String>,

    // ---- Git source form ----
    /// HTTPS git URL of a Python project. retread clones at `rev` and
    /// runs `pip wheel --no-deps` on `subdirectory` (defaulting to ".").
    #[serde(default)]
    pub git: Option<String>,

    /// Git revision (commit SHA, tag, or branch) for `git` source.
    /// Required when `git` is set.
    #[serde(default)]
    pub rev: Option<String>,

    /// Subdirectory within the git clone containing the Python project
    /// to build. Defaults to "." (the repo root).
    #[serde(default)]
    pub subdirectory: Option<String>,

    /// If true, after the wheel is built and the conda env is
    /// materialized, the user is expected to run
    /// `pip install -e <path> --no-deps --force-reinstall` to overlay
    /// editable on top of the bundled installation. retread doesn't
    /// run this for you -- see the example pixi.toml's
    /// `overlay-editable` task. The wheel is still built normally so
    /// retread can rewrite METADATA + emit a coherent dep set.
    #[serde(default)]
    pub editable: bool,

    // ---- Named git source reference ----
    /// Reference to a `[retread-git-sources]` entry. The named entry
    /// provides `url` + `rev`; this wheel entry only contributes
    /// `subdirectory` (defaulting to "."). Lets many sub-packages from
    /// the same monorepo share a single rev declaration.
    #[serde(default)]
    pub from: Option<String>,

    /// Group entries that share this string into a single conda output.
    /// The output's name is the bundle string. All wheels from grouped
    /// entries become the bundle's wheels (primary = first entry's
    /// primary, others as extras). Without this field each entry
    /// produces its own conda output named after the entry key.
    ///
    /// Use this when you want a single workspace declaration (e.g.
    /// `isaac-pack = { path = "./isaac-pack" }`) to install every
    /// wheel in the pack -- pixi-build only builds outputs the
    /// workspace declared, so grouping into one output sidesteps the
    /// N-line declaration problem at the cost of a bigger artifact.
    #[serde(default)]
    pub bundle: Option<String>,
}

/// A named (url, rev) pair used to share a git source across many
/// `[retread-wheels]` entries.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct NamedGitSource {
    pub url: String,
    pub rev: String,
}

impl WheelEntry {
    pub fn is_url(&self) -> bool {
        self.url.is_some()
    }
    pub fn is_spec(&self) -> bool {
        !self.is_url() && !self.is_path() && !self.is_git() && self.version.is_some()
    }
    pub fn is_path(&self) -> bool {
        self.path.is_some()
    }
    pub fn is_git(&self) -> bool {
        self.git.is_some()
    }
    pub fn is_named_git(&self) -> bool {
        self.from.is_some()
    }

    /// Normalize the entry in place before validation.
    ///
    /// URL-form entries may carry the wheel hash as a `#sha256=<hex>` URL
    /// fragment (the spelling PEP 503 / pip / uv emit) instead of the
    /// discrete `sha256` field. Such entries deserialize with
    /// `sha256 = None`, so [`fetch_wheel_cached`](crate::wheel::fetch_wheel_cached)
    /// has no content key and silently bypasses the persistent wheel store --
    /// a multi-GiB extscache wheel then redownloads from a flaky CDN on every
    /// cold run. Lift the fragment hash into the discrete field and strip the
    /// fragment from the URL so every later fetch/compare site sees one
    /// canonical key.
    ///
    /// Precedence: an explicit discrete `sha256` wins. If both are present and
    /// disagree, that is a contradiction and we error (a wrong hash would
    /// either poison the cache or wedge verification).
    pub fn normalize(&mut self, name: &str) -> Result<()> {
        let Some(url) = self.url.as_mut() else {
            return Ok(());
        };
        // Pull `sha256=<hex>` out of the URL fragment, if present. The
        // fragment may in principle carry multiple `&`-joined params; only the
        // sha256 one is meaningful here.
        let frag_sha = url.fragment().and_then(|frag| {
            frag.split('&').find_map(|kv| {
                kv.strip_prefix("sha256=")
                    .filter(|h| !h.is_empty())
                    .map(str::to_string)
            })
        });
        if let Some(frag_sha) = frag_sha {
            match &self.sha256 {
                Some(discrete) if !discrete.eq_ignore_ascii_case(&frag_sha) => {
                    return Err(anyhow!(
                        "wheel `{name}`: discrete sha256 `{discrete}` disagrees with the \
                         URL fragment sha256 `{frag_sha}` -- remove one",
                    ));
                }
                // Discrete field present and matching: discrete wins, keep it.
                Some(_) => {}
                // Only the fragment carries the hash: adopt it, lowercased
                // (the store path and every comparison site use lowercase
                // hex; a mixed-case fragment must not fork the cache key).
                None => self.sha256 = Some(frag_sha.to_ascii_lowercase()),
            }
        } else if let Some(frag) = url.fragment()
            && !frag.is_empty()
        {
            tracing::debug!(
                wheel = %name,
                fragment = %frag,
                "wheel url carries a non-sha256 fragment; stripping it",
            );
        }
        // Strip the fragment unconditionally so the fetched/compared URL is
        // canonical (reqwest never transmits fragments anyway, but a lingering
        // `#sha256=` would leak into filenames/logs and diverge from the clean
        // URL recorded elsewhere).
        url.set_fragment(None);
        Ok(())
    }

    /// Validate that the entry has exactly one form.
    pub fn validate(&self, name: &str) -> Result<()> {
        let form_count = [
            self.url.is_some(),
            self.version.is_some(),
            self.path.is_some(),
            self.git.is_some(),
            self.from.is_some(),
        ]
        .into_iter()
        .filter(|b| *b)
        .count();
        if form_count == 0 {
            return Err(anyhow!(
                "wheel `{name}`: requires one of `url`, `version`, `path`, `git`, or `from`"
            ));
        }
        if form_count > 1 {
            return Err(anyhow!(
                "wheel `{name}`: set exactly ONE of `url`, `version`, `path`, `git`, `from`"
            ));
        }
        if self.is_git() && self.rev.is_none() {
            return Err(anyhow!(
                "wheel `{name}`: `git` requires `rev` (commit, tag, or branch)"
            ));
        }
        if self.is_url() && !self.extras.is_empty() {
            return Err(anyhow!(
                "wheel `{name}`: `extras` is not meaningful on the bare-URL \
                 form (the upstream filename doesn't carry a project name to \
                 attach extras to). Use the `version`, `path`, `git`, or \
                 `from` form instead."
            ));
        }
        // v0.12.0+: extras now valid on path/git/named-git. The wheel
        // built from those sources carries `Requires-Dist: ...; extra
        // == "<name>"` lines in METADATA, and the handler runs the
        // standard extras BFS on them (just like PyPI form), pulling
        // the extras' deps via PyPI Simple.
        Ok(())
    }

    /// Normalized version string (leading `==` stripped). Only meaningful
    /// for spec-form entries.
    pub fn normalized_version(&self) -> Option<String> {
        self.version
            .as_ref()
            .map(|v| v.trim().trim_start_matches("==").trim().to_string())
    }

    /// Default index when in spec form.
    pub fn index_url(&self) -> String {
        self.index
            .clone()
            .unwrap_or_else(|| "https://pypi.org/simple/".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_map_noncanonical_key_is_canonicalized() {
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-name-map": {
                "opencv_python_headless": "py-opencv"
            },
        });

        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        let target = cfg
            .name_map
            .get(&PypiKey::from_pypi("opencv-python-headless"))
            .and_then(CondaTarget::mapped_name)
            .expect("non-canonical user key must remain live after config load");
        assert_eq!(target.as_spec(), "py-opencv");
    }

    #[test]
    fn name_map_equal_canonical_aliases_are_deduplicated() {
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-name-map": {
                "opencv-python-headless": "py-opencv",
                "opencv_python_headless": "py-opencv"
            },
        });

        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.name_map.len(), 1);
        assert_eq!(
            cfg.name_map
                .get(&PypiKey::from_pypi("opencv.python.headless"))
                .and_then(CondaTarget::mapped_name)
                .map(CondaName::as_spec),
            Some("py-opencv")
        );
    }

    #[test]
    fn name_map_conflicting_canonical_aliases_are_rejected() {
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-name-map": {
                "opencv-python-headless": "",
                "opencv_python_headless": "py-opencv"
            },
        });

        let error = serde_json::from_value::<RetreadConfig>(json).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("conflicting retread-name-map entries"));
        assert!(message.contains("opencv-python-headless"));
    }

    #[test]
    fn typed_name_map_json_round_trip_preserves_targets() {
        let expected = NameMap::from([
            (
                PypiKey::from_pypi("opencv_python_headless"),
                CondaTarget::Mapped(CondaName::new("py_opencv")),
            ),
            (PypiKey::from_pypi("wheel-only"), CondaTarget::Disabled),
        ]);

        let json = serde_json::to_value(&expected).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "opencv-python-headless": "py_opencv",
                "wheel-only": ""
            })
        );
        let round_trip: NameMap = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip, expected);
    }

    #[test]
    fn parses_retread_wheels_key() {
        // Mirrors the inline-table syntax from the README and examples.
        let json = serde_json::json!({
            "retread-wheels": {
                "isaacsim": {
                    "version": "==5.1.0",
                    "index": "https://pypi.nvidia.com",
                    "extras": ["all", "extscache"],
                },
                "foo": { "url": "https://example.com/foo-1.whl", "sha256": "abc" }
            },
            "retread-relax": "minor",
            "retread-build-number": 0,
            "overrides": { "numpy": ">=1.26,<2" },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.retread_wheels.len(), 2);
        assert_eq!(cfg.relax, RelaxPolicy::Minor);
        assert!(cfg.retread_wheels.contains_key("isaacsim"));
        let isaac = &cfg.retread_wheels["isaacsim"];
        assert_eq!(isaac.normalized_version().unwrap(), "5.1.0");
        assert!(isaac.is_spec());
        assert_eq!(isaac.extras, vec!["all", "extscache"]);
    }

    #[test]
    fn normalize_lifts_url_fragment_sha256() {
        // Fragment-only: the hash rides in `#sha256=` and no discrete field.
        // Mixed-case fragment: adopted LOWERCASED (the store path and every
        // comparison site use lowercase hex).
        let mut entry = WheelEntry {
            url: Some(
                "https://pypi.nvidia.com/foo/foo-1.0-cp312-none-any.whl#sha256=ABC123"
                    .parse()
                    .unwrap(),
            ),
            ..Default::default()
        };
        entry.normalize("foo").unwrap();
        assert_eq!(entry.sha256.as_deref(), Some("abc123"));
        // Fragment stripped from the canonical URL.
        assert_eq!(entry.url.as_ref().unwrap().fragment(), None);
    }

    #[test]
    fn normalize_discrete_sha256_wins_over_fragment() {
        // Both present and equal (ignoring case): discrete field is kept, no error.
        let mut entry = WheelEntry {
            url: Some(
                "https://ex.com/foo-1.0-cp312-none-any.whl#sha256=ABC123"
                    .parse()
                    .unwrap(),
            ),
            sha256: Some("abc123".to_string()),
            ..Default::default()
        };
        entry.normalize("foo").unwrap();
        assert_eq!(entry.sha256.as_deref(), Some("abc123"));
        assert_eq!(entry.url.as_ref().unwrap().fragment(), None);
    }

    #[test]
    fn normalize_mismatched_dual_sha256_errors() {
        // Discrete and fragment disagree: contradiction, must error.
        let mut entry = WheelEntry {
            url: Some(
                "https://ex.com/foo-1.0-cp312-none-any.whl#sha256=deadbeef"
                    .parse()
                    .unwrap(),
            ),
            sha256: Some("abc123".to_string()),
            ..Default::default()
        };
        let err = entry.normalize("foo").unwrap_err().to_string();
        assert!(err.contains("disagrees"), "unexpected error: {err}");
    }

    #[test]
    fn normalize_noop_without_fragment() {
        let mut entry = WheelEntry {
            url: Some("https://ex.com/foo-1.0-cp312-none-any.whl".parse().unwrap()),
            ..Default::default()
        };
        entry.normalize("foo").unwrap();
        assert_eq!(entry.sha256, None);
    }

    #[test]
    fn parses_retread_compression_level_key() {
        // v1.5.8. Invariant #6: every new config field needs a serde
        // test or stale binaries reject user pixi.toml with "unknown
        // field" during upgrades.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-compression-level": 3,
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.compression_level, Some(3));

        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.compression_level, None);
    }

    #[test]
    fn parses_retread_compression_threads_key() {
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-compression-threads": 6,
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.compression_threads.map(NonZeroUsize::get), Some(6));

        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.compression_threads, None);

        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-compression-threads": 0,
        });
        assert!(serde_json::from_value::<RetreadConfig>(json).is_err());
    }

    #[test]
    fn route_policy_defaults_to_prefer_conda_validated() {
        // Invariant #6: every new config field needs a serde test. An absent
        // policy uses conda-facts-first validated routing.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.route_policy, RoutePolicy::PreferCondaValidated);
        assert!(cfg.route_include.is_empty());
    }

    #[test]
    fn parses_all_retread_route_policies_and_include() {
        for (serialized, expected) in [
            ("prefer-conda-validated", RoutePolicy::PreferCondaValidated),
            ("minimal", RoutePolicy::Minimal),
            ("aggressive", RoutePolicy::Aggressive),
        ] {
            let json = serde_json::json!({
                "retread-wheels": { "foo": { "version": "==1.0" } },
                "retread-route-policy": serialized,
                "retread-route-include": ["mujoco", "grpcio"],
            });
            let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
            assert_eq!(cfg.route_policy, expected);
            assert_eq!(cfg.route_include, vec!["mujoco", "grpcio"]);
        }
    }

    #[test]
    fn parses_retread_deps_from_default_empty() {
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.deps_from.is_empty());
    }

    #[test]
    fn parses_retread_deps_from_bare_string_local() {
        // Invariant #6: every new config field needs a serde test.
        let json = serde_json::json!({
            "retread-wheels": {},
            "retread-deps-from": "requirements_isaaclab.txt",
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(
            cfg.deps_from.as_slice(),
            &[crate::deps_from::DepSource::Local(
                std::path::PathBuf::from("requirements_isaaclab.txt")
            )]
        );
    }

    #[test]
    fn parses_retread_deps_from_bare_string_url() {
        let json = serde_json::json!({
            "retread-wheels": {},
            "retread-deps-from": "https://raw.githubusercontent.com/org/repo/main/requirements.txt",
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(
            cfg.deps_from.as_slice(),
            &[crate::deps_from::DepSource::Url(
                "https://raw.githubusercontent.com/org/repo/main/requirements.txt".to_string()
            )]
        );
    }

    #[test]
    fn parses_retread_deps_from_git_table() {
        let json = serde_json::json!({
            "retread-wheels": {},
            "retread-deps-from": {
                "git": "https://github.com/org/repo",
                "rev": "deadbeef",
                "path": "requirements.txt",
            },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(
            cfg.deps_from.as_slice(),
            &[crate::deps_from::DepSource::Git {
                git: "https://github.com/org/repo".to_string(),
                rev: "deadbeef".to_string(),
                path: "requirements.txt".to_string(),
            }]
        );
    }

    #[test]
    fn parses_retread_deps_from_list_mixing_forms() {
        let json = serde_json::json!({
            "retread-wheels": {},
            "retread-deps-from": [
                "requirements_base.txt",
                "https://raw.githubusercontent.com/org/repo/main/requirements.txt",
                {
                    "git": "https://github.com/org/repo",
                    "rev": "deadbeef",
                    "path": "pyproject.toml",
                },
            ],
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(
            cfg.deps_from.as_slice(),
            &[
                crate::deps_from::DepSource::Local(std::path::PathBuf::from(
                    "requirements_base.txt"
                )),
                crate::deps_from::DepSource::Url(
                    "https://raw.githubusercontent.com/org/repo/main/requirements.txt".to_string()
                ),
                crate::deps_from::DepSource::Git {
                    git: "https://github.com/org/repo".to_string(),
                    rev: "deadbeef".to_string(),
                    path: "pyproject.toml".to_string(),
                },
            ]
        );
    }

    #[test]
    fn parses_retread_shadow_libs_key() {
        let json = serde_json::json!({
            "retread-wheels": { "isaacsim": { "version": "==6.0.0.1" } },
            "retread-shadow-libs": {
                "isaacsim/kit/kernel/plugins/libpython3.12.so.1.0": "conda-lib"
            },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(
            cfg.shadow_libs
                .get("isaacsim/kit/kernel/plugins/libpython3.12.so.1.0")
                .copied(),
            Some(ShadowPolicy::CondaLib)
        );

        let json = serde_json::json!({
            "retread-wheels": { "isaacsim": { "version": "==6.0.0.1" } },
            "retread-shadow-libs": {
                "x.so": "copy-from-somewhere"
            },
        });
        let err = serde_json::from_value::<RetreadConfig>(json).unwrap_err();
        assert!(
            err.to_string().contains("copy-from-somewhere"),
            "unknown shadow policy should be a hard config error: {err}"
        );
    }

    #[test]
    fn parses_retread_emit_pypi_key() {
        // DEPRECATED (v2.0.0): the field is parsed-and-ignored so old
        // manifests survive an upgrade without a "unknown field" rejection.
        // Invariant #6: the field must still parse correctly.
        // The value is now ignored; use `retread-courier` instead.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-emit-pypi": true,
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.emit_pypi);

        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(!cfg.emit_pypi, "unset defaults to off");
    }

    #[test]
    fn parses_retread_courier_key() {
        // v2.1.0: courier is the DEFAULT. Invariant #6: deny_unknown_fields
        // rejects unknown keys, so the flag needs a serde test.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-courier": true,
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.courier);

        // unset -> defaults to ON (courier is the default mode, v2.1.0+).
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.courier, "unset defaults to courier");

        // explicit opt-out to the legacy artifact path.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-courier": false,
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(!cfg.courier, "explicit false opts out to legacy");
    }

    #[test]
    fn deprecated_blueprint_sync_key_still_parses() {
        // DEPRECATED (v2.0.0): the field is parsed-and-ignored; the
        // deprecation contract (Invariant #6) requires that old manifests
        // carrying this key still parse rather than failing with "unknown
        // field". The value is now ignored; use `retread-courier` instead.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-blueprint-sync": "script",
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.blueprint_sync.as_deref(), Some("script"));

        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.blueprint_sync, None, "absent by default");
    }

    #[test]
    fn parses_retread_blueprint_key() {
        // DEPRECATED (v2.0.0): the field is parsed-and-ignored so old
        // manifests survive an upgrade without a "unknown field" rejection
        // (Invariant #6). The value is now ignored; use `retread-courier`
        // instead. is_on() / is_only() are called here to keep the methods
        // reachable (they are referenced from handler/mod.rs deprecation warn).
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-blueprint": true,
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.blueprint.is_on());
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-blueprint": "only",
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.blueprint.is_only());

        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(!cfg.blueprint.is_on(), "unset defaults to off");
    }

    #[test]
    fn parses_retread_bundle_key() {
        // v1.4.0: pack-wide default bundle group. Invariant #6: every
        // new config field needs a serde test or stale binaries reject
        // user pixi.toml with "unknown field" during upgrades.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-bundle": "isaac-pack",
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.default_bundle.as_deref(), Some("isaac-pack"));

        // Unset stays None (historical per-entry-output behavior).
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.default_bundle, None);
    }

    #[test]
    fn legacy_unprefixed_keys_still_parse() {
        // Kept for back-compat; removing would hard-error old manifests
        // via deny_unknown_fields. The serde aliases (`alias = "wheels"`,
        // `alias = "relax"`, etc.) accept the pre-0.4 unprefixed keys
        // indefinitely so existing pixi.toml files keep loading.
        let json = serde_json::json!({
            "wheels": { "foo": { "version": "1.2.3" } },
            "relax": "patch",
            "build-number": 7,
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.retread_wheels.len(), 1);
        assert_eq!(cfg.relax, RelaxPolicy::Patch);
        assert_eq!(cfg.build_number, 7);
    }

    #[test]
    fn parses_bundle_field_on_entry() {
        // Regression: when the `bundle` field was added to WheelEntry,
        // a stale retread without the field would reject the user's
        // pixi.toml with `unknown field bundle, expected one of ...`
        // (because of `deny_unknown_fields`). This test pins the
        // parse-side contract: any new optional field added to
        // WheelEntry must be tested through serde deserialization
        // BEFORE we ship it, or stale binaries break every user's
        // pixi.toml during the upgrade window.
        let json = serde_json::json!({
            "retread-wheels": {
                "isaacsim": {
                    "version": "==5.1.0",
                    "index": "https://pypi.nvidia.com",
                    "extras": ["all"],
                    "bundle": "isaac-pack",
                },
                "isaaclab": {
                    "from": "isaaclab",
                    "subdirectory": "source/isaaclab",
                    "bundle": "isaac-pack",
                },
                "loner": { "version": "==1.0" }
            }
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(
            cfg.retread_wheels["isaacsim"].bundle.as_deref(),
            Some("isaac-pack")
        );
        assert_eq!(
            cfg.retread_wheels["isaaclab"].bundle.as_deref(),
            Some("isaac-pack")
        );
        // Default is None when omitted -- one entry == one output.
        assert_eq!(cfg.retread_wheels["loner"].bundle, None);
    }

    #[test]
    fn parses_tiered_cascade_policy() {
        // v0.30.0+ tiered cascade variant. New string value; pin the
        // serde rename so stale binaries don't reject pixi.toml with
        // "unknown variant" during the upgrade window.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "1.2.3" } },
            "retread-relax": "patch-then-minor-then-major-then-last-resort",
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert_eq!(
            cfg.relax,
            RelaxPolicy::PatchThenMinorThenMajorThenLastResort
        );
        assert!(cfg.relax.has_tiered_cascade());
        assert!(cfg.relax.allows_widening_mutation());
        assert!(!cfg.relax.has_last_resort());
    }

    #[test]
    fn relax_policy_helper_semantics() {
        assert!(RelaxPolicy::MinorWithLastResort.has_last_resort());
        assert!(RelaxPolicy::MinorWithLastResort.allows_widening_mutation());
        assert!(!RelaxPolicy::MinorWithLastResort.has_tiered_cascade());

        assert!(!RelaxPolicy::Minor.has_last_resort());
        assert!(!RelaxPolicy::Minor.has_tiered_cascade());
        assert!(!RelaxPolicy::Minor.allows_widening_mutation());
    }

    #[test]
    fn rejects_unknown_field_on_entry() {
        // Mirror image of the above: deny_unknown_fields must fire on
        // anything we haven't added to WheelEntry. If this test ever
        // starts failing, it means we shipped a permissive parser and
        // the next time we add a field, stale binaries won't break --
        // but neither will typos in user pixi.toml. Trade-off
        // documented; current default is strict.
        let json = serde_json::json!({
            "retread-wheels": {
                "foo": { "version": "==1.0", "totally-bogus-key": "x" }
            }
        });
        let err = serde_json::from_value::<RetreadConfig>(json).unwrap_err();
        assert!(
            err.to_string().contains("totally-bogus-key"),
            "unknown-field error should name the offender; got: {err}",
        );
    }

    #[test]
    fn rejects_entry_with_both_url_and_version() {
        let entry = WheelEntry {
            url: Some("https://example.com/x.whl".parse().unwrap()),
            version: Some("1.0".into()),
            ..Default::default()
        };
        assert!(entry.validate("x").is_err());
    }

    #[test]
    fn rejects_entry_with_neither_url_nor_version() {
        let entry = WheelEntry::default();
        assert!(entry.validate("x").is_err());
    }

    #[test]
    fn rejects_extras_on_url_form() {
        let entry = WheelEntry {
            url: Some("https://example.com/x.whl".parse().unwrap()),
            extras: vec!["all".into()],
            ..Default::default()
        };
        assert!(entry.validate("x").is_err());
    }

    /// v0.12.0+: extras are valid on git/path/named-git entries. The
    /// extras' deps come through METADATA's `Requires-Dist: ...; extra
    /// == "<name>"` lines on the built wheel and get BFS-resolved via
    /// PyPI Simple by the handler, same as for the spec form.
    #[test]
    fn accepts_extras_on_path_and_git_and_named_git() {
        let path_entry = WheelEntry {
            path: Some("./local-pkg".into()),
            extras: vec!["rl_games".into()],
            ..Default::default()
        };
        path_entry.validate("a").expect("extras valid on path form");

        let git_entry = WheelEntry {
            git: Some("https://example.com/repo.git".into()),
            rev: Some("HEAD".into()),
            extras: vec!["rl_games".into()],
            ..Default::default()
        };
        git_entry.validate("b").expect("extras valid on git form");

        let named_git_entry = WheelEntry {
            from: Some("isaaclab".into()),
            extras: vec!["rl_games".into()],
            ..Default::default()
        };
        named_git_entry
            .validate("c")
            .expect("extras valid on named-git form");
    }

    #[test]
    fn python_spec_accepts_string_or_list() {
        let one: PythonSpec = serde_json::from_value(serde_json::json!("3.11")).unwrap();
        assert_eq!(one.as_versions(), vec!["3.11"]);
        let many: PythonSpec = serde_json::from_value(serde_json::json!(["3.11", "3.12"])).unwrap();
        assert_eq!(many.as_versions(), vec!["3.11", "3.12"]);
    }

    #[test]
    fn parses_retread_pin_version_key() {
        // Invariant #6: every new config field needs a serde test or stale
        // binaries reject user pixi.toml with "unknown field" during upgrades.

        // unset -> false (default: epoch-gated replay, not strict per-version).
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(!cfg.pin_version, "unset defaults to false");

        // explicit true -> strict per-version reproducibility.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "retread-pin-version": true,
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.pin_version, "explicit true must parse as true");

        // alias "pin-version" also accepted.
        let json = serde_json::json!({
            "retread-wheels": { "foo": { "version": "==1.0" } },
            "pin-version": true,
        });
        let cfg: RetreadConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.pin_version, "alias pin-version must parse as true");
    }
}
