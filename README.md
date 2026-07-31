# pixi-build-retread

[![linux-x86_64](https://github.com/garylvov/pixi-build-retread/actions/workflows/ci.yml/badge.svg)](https://github.com/garylvov/pixi-build-retread/actions/workflows/ci.yml)

Bundles PyPI wheel closures into conda packages, crossing the conda↔uv boundary.

## Requirements

- Pixi providing `pixi-build-api-version >=4,<6` (pinned in `recipe/recipe.yaml`).
  Rich platform envelopes (`platforms = [{ platform = "linux-64", glibc = "2.35" }]`)
  need Pixi >= 0.71; recommended: **Pixi 0.73**.
- Commit a workspace `.pixi/config.toml` (a fresh `pixi init` re-includes it):

  ```toml
  run-post-link-scripts = "insecure"   # courier post-link install (retread's own script)

  [concurrency]
  solves = 1
  ```

  One Pixi process per workspace; for parallel installs, one clone per install.
  Alternatively set `retread-courier-mode = "activation"` to skip the post-link
  and install lazily on first activation.

## Setup

```toml
# pixi.toml
[workspace]
preview = ["pixi-build"]
channels = ["https://prefix.dev/conda-forge"]
platforms = [{ platform = "linux-64", glibc = "2.35" }]
[dependencies]
isaac-pack = { path = "./isaac-pack" }

# isaac-pack/pixi.toml
[package]
name = "isaac-pack"
backend = { name = "pixi-build-retread", version = ">=4.4.0", channels = ["https://prefix.dev/garylvov", "https://prefix.dev/pixi-build-backends", "https://prefix.dev/conda-forge"] }
[package.build.config]
retread-bundle = "isaac-pack"
[package.build.config.retread-wheels]
isaacsim = { version = "==5.1.0", index = "https://pypi.nvidia.com" }
```

The backend is published on `prefix.dev/garylvov` (not conda-forge), so the
backend `channels` list must include it.

Run: `pixi install`

## Pack Configuration

A pack's `pixi.toml` (e.g., `examples/isaac6/isaac-pack/`) declares package
metadata and wheels to bundle:

```toml
[package]
name = "isaac-pack-6"
version = "6.0.0.1"
[package.build]
backend = { name = "pixi-build-retread", version = "*", channels = ["https://prefix.dev/garylvov", "https://prefix.dev/pixi-build-backends", "https://prefix.dev/conda-forge"] }
[package.build.config]
retread-python   = "3.12"
retread-bundle   = "isaac-pack-6"
retread-resolver = "uv"        # default
retread-hermetic = true        # default; sysroot-pinned native sdist builds
[package.build.config.retread-wheels]
isaacsim = { version = "==6.0.0.1", index = "https://pypi.nvidia.com" }
```

Dependency-conflict escape hatches, in the same manifest — relax upstream pins
globally, override one dependency, or drop one from the emitted run deps:

```toml
[package.build.config]
retread-relax = "minor"
retread-drop-deps = ["optional-package"]

[package.build.config.retread-overrides]
numpy = ">=1.26,<2"
```

## Commands

| Command | Usage |
|---------|-------|
| install | `retread install --lock <lock.json> --prefix <p>` |
| verify | `retread verify --lock <lock.json> [--full]` |
| solve | repair: `retread solve --manifest pixi.toml -e <env>`; audit all: `retread solve --manifest pixi.toml --audit` |
| fast (env) | `eval "$(pixi-build-retread fast --print-env)"` |

## Fast path

On shared-filesystem machines (SLURM clusters, EC2+EFS),
`eval "$(pixi-build-retread fast --print-env)"` reroutes caches and env storage
to fast machine-local disk; `pixi install` then materializes environments with a
no-resolve frozen rebuild backed by the shared package cache. On fast local
disk, fast-tmp auto-disengages. Defaults work unconfigured; overrides:

```toml
[tool.retread.fast-tmp]
mode = "auto"                  # auto | on | off
tmp-root = "/tmp"
blob-caches = "shared"
```

Env vars `RETREAD_FAST_TMP`, `RETREAD_FAST_TMP_ROOT`,
`RETREAD_FAST_TMP_BUDGET_BYTES`, and `RETREAD_SHARED_CACHE_DIR` override the TOML.

## Threads & caches

Compression threads are budgeted node-wide per user (PID-lease registry): a
solo build gets full parallelism; concurrent builds share the budget. Knobs:
`RETREAD_COMPRESSION_THREADS` (per-build override), `RETREAD_COMPRESSION_BUDGET`
(node budget; default = available parallelism), `RETREAD_THREAD_LEASE_DIR`
(registry location, mainly for tests).

Probe solves are serial by default. `RETREAD_PARALLEL_PROBES=1` opts into the
experimental bounded parallel probe pool; bisection and shared repodata remain
enabled either way.

The shared wheel store (`~/.cache/retread/wheels`, override
`RETREAD_WHEEL_STORE`) is independent of `RETREAD_CACHE_DIR` and fast-tmp;
lock records reference its content SHAs, so do not delete it casually.

Native source builds on `linux-64` use a cached conda compiler environment
pinned to the newest `sysroot_linux-64` no newer than the target glibc floor.
Their wheel tag names that exact sysroot (for example, sysroot 2.28 produces
`manylinux_2_28_x86_64`). Set `retread-hermetic = false` for one pack or
`RETREAD_HERMETIC_BUILDS=0` for the process to retain host-only builds.
Archive policy validation admits Linux x86_64 ET_DYN extensions and
tuple-gated CUDA `.cubin` payloads; standalone ELF executables and static
objects fail closed because auditwheel cannot completely attest them.

## Rust toolchain

Rust `1.97.0` is pinned in `rust-toolchain.toml` (matching CI). Install via
[rustup](https://rustup.rs), which reads the pin automatically.
