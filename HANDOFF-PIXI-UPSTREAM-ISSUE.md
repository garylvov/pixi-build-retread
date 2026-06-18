# Draft: pixi feature request (NOT FILED — for Gary's review)

Title: Allow `[pypi-options]` (find-links, dependency-overrides, prerelease)
to be sourced from an external file, or support manifest includes

## Problem

Build backends (pixi-build) and code generators have no sanctioned way to
contribute PyPI resolution options to a workspace. Today `find-links`,
`dependency-overrides`, and `prerelease-mode` are workspace-manifest-only:

- No include/extends mechanism in pixi.toml (pixi_manifest has none).
- Global config (`PyPIConfig`) carries only index-url/extra-index-urls/
  keyring/allow-insecure-host.
- `[tool.uv]` in pyproject workspaces is ignored.
- The pixi-build protocol (`CondaOutput`) has no pypi side channel.

Concrete case: pixi-build-retread repacks PyPI stacks (Isaac Sim scale) and
can generate a complete, correct `[feature.X]` block — find-links to fixed
wheels, a dependency-overrides table, prerelease pins. The only delivery
options are (a) machine-editing the user's pixi.toml (a fenced auto-synced
block — users object to ANY machine-written manifest bytes) or (b) bypassing
pixi's solver entirely with a standalone-uv install script (loses single-
lockfile integration; pixi's installer prunes the overlay on pypi re-syncs).

## Proposal (any one of these dissolves the problem)

1. `[pypi-options] options-file = "path/to/generated.toml"` — merge an
   external TOML fragment (find-links/overrides/prerelease) at parse time,
   path relative to the manifest.
2. A general `include = ["path.toml"]` for feature tables.
3. uv-style file refs: `dependency-overrides-file = "overrides.txt"`
   (PEP 508 lines, matching uv's `--overrides`).

Option 1/3 are narrow and keep the manifest the source of truth for WHICH
file is trusted; the file contents can then be machine-owned.

## PRECISE upstream scope (grizzly source audit, pixi rev dc5d0dc) — the real fix

The root cause, from source: uv's resolution inputs are built from ONE object
`IndexLocations` at `crates/pixi_core/src/lock_file/resolve/pypi.rs:440-441`,
sourced ONLY from `environment.pypi_options()`
(`crates/pixi_core/src/lock_file/update.rs:2582` ->
`crates/pixi_manifest/src/features_ext.rs:231-252`, which folds ONLY the
workspace manifest features). The build protocol has NO pypi field in any
backend response (`crates/pixi_build_types/src/procedures/conda_outputs.rs:79`
`CondaOutput` is conda-only; same for conda_build_v1/metadata/initialize), and
a source `[package]` manifest has no pypi fields at all
(`crates/pixi_manifest/src/manifests/target.rs:48-51` `PackageTarget` is
conda-only). Global/.pixi config + UV_* env vars do NOT reach the solve
(`crates/pixi_config/src/lib.rs:343-362` PyPIConfig has only
index-url/extra-index-urls/keyring/insecure-host, and even those are init-only,
`crates/pixi_api/src/workspace/init/mod.rs:126-127`).

### Smallest change that makes a build backend natively delegate to uv (THE ask)
1. Add `pypi_options: Option<PypiOptions>` to `CondaOutput`
   (`crates/pixi_build_types/src/procedures/conda_outputs.rs:79`).
2. Union backend-returned options into the fold at
   `crates/pixi_manifest/src/features_ext.rs:231-252` (consumed at
   `update.rs:2582`, before `pypi.rs:440`).
One protocol field + one merge site. Then a backend (retread) emits find-links
+ meta-wheel + overrides via `CondaOutput`; the consumer keeps ONE clean conda
path-dep line; uv resolves + fast-installs natively; zero hand-written pypi
bytes. This is the exact "backend delegates to uv like the manifest does"
symmetry. Alternative (also small): give `PackageTarget` a `pypi_options` field
+ union source-package pypi-options at the same merge site.

VERDICT on current pixi: NO native path exists. post-link (via
`.pixi/config.toml run-post-link-scripts="insecure"`) is the ONLY mechanism
today for a clean-manifest + uv-fast install; the native conda path forces the
25GB zstd payload. The above PR is what unlocks the truly-native path.

## Evidence of demand

(retread's blueprint mode: build-tagged find-links wheels beating registry
originals, generated meta-wheel, validated on Isaac Sim 6 + IsaacLab — happy
to share the full design notes. The prerelease case is the sharpest: uv only
honors transitive prerelease pins via direct requirements/overrides, so a
generated overrides table is unavoidable, and it has nowhere first-class to
live.)
