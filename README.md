# pixi-build-retread

[![linux-x86_64](https://github.com/garylvov/pixi-build-retread/actions/workflows/ci.yml/badge.svg)](https://github.com/garylvov/pixi-build-retread/actions/workflows/ci.yml)

Bundles PyPI wheel closures into conda packages, crossing the conda↔uv boundary.

## Setup

```toml
# pixi.toml
[workspace]
preview = ["pixi-build"]
channels = ["https://prefix.dev/conda-forge"]
[system-requirements]
libc = "2.35"
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

The backend itself is published on `prefix.dev/garylvov`, not conda-forge, so
`channels` in the backend spec (separate from the workspace `channels` above)
must list it -- omitting it leaves the backend unresolvable.

Run: `pixi install`

Courier installs via a conda post-link script (retread's own, not
third-party), so the workspace `.pixi/config.toml` needs
`run-post-link-scripts = "insecure"`. Set `retread-courier-mode =
"activation"` to skip the post-link and install lazily on first `pixi
run`/`shell` instead -- no insecure config, slower first activation.

## Pack Configuration Example
A pack's `pixi.toml` (e.g., `examples/isaac6/isaac-pack/`) declares package metadata and wheels to bundle:
```toml
[package]
name = "isaac-pack-6"
version = "6.0.0.1"
[package.build]
backend = { name = "pixi-build-retread", version = "*", channels = ["https://prefix.dev/garylvov", "https://prefix.dev/pixi-build-backends", "https://prefix.dev/conda-forge"] }
[package.build.config]
retread-python   = "3.12"
retread-bundle   = "isaac-pack-6"
retread-resolver = "uv"
[package.build.config.retread-wheels]
isaacsim = { version = "==6.0.0.1", index = "https://pypi.nvidia.com" }
# ...
```

Dependency-conflict escape hatches live in that pack manifest. Relax upstream
pins globally, override one dependency explicitly, or omit a dependency from
the generated conda run dependencies:

```toml
[package.build.config]
retread-relax = "minor"
retread-drop-deps = ["optional-package"]

[package.build.config.retread-overrides]
numpy = ">=1.26,<2"
```

## Rust toolchain

Rust `1.97.0` is pinned in `rust-toolchain.toml` (kept in sync with CI's
`dtolnay/rust-toolchain` pin) so `fmt`/`clippy` never drift between local and
CI. Install it with [rustup](https://rustup.rs) (which reads
`rust-toolchain.toml` automatically) -- `pixi exec --spec rust` pulls from
conda-forge, which lags the pinned version and will disagree on `fmt`/
`clippy` output.

## Commands

| Command | Usage |
|---------|-------|
| install | `retread install --lock <lock.json> --prefix <p>` |
| verify | `retread verify --lock <lock.json> [--full]` |
| solve | repair: `retread solve --manifest pixi.toml -e <env>`; audit all: `retread solve --manifest pixi.toml --audit` |
| fast (env) | `eval "$(pixi-build-retread fast --print-env)"` |

## Fast path

For shared-filesystem machines (SLURM clusters, EC2+EFS) where the project and caches
sit on slow shared storage: `eval "$(pixi-build-retread fast --print-env)"` reroutes
caches and env storage to fast machine-local disk (Pixi 0.70 or newer is required).
`pixi install` then materializes the environment with a no-resolve frozen rebuild
backed by the shared package cache. Raw
environment snapshots are intentionally unsupported because Pixi embeds the detached
prefix in scripts and metadata, making snapshots unsafe to move between job roots. On
a local machine with fast disk, fast-tmp auto-disengages.

Configuration is optional: the default `mode = "auto"` engages on slow shared
filesystems and auto-disengages on fast local disk; `RETREAD_FAST_TMP`,
`RETREAD_FAST_TMP_ROOT`, `RETREAD_FAST_TMP_BUDGET_BYTES`, and
`RETREAD_SHARED_CACHE_DIR` override the TOML.

```toml
[tool.retread.fast-tmp]
mode = "auto"                  # auto | on | off
tmp-root = "/tmp"
blob-caches = "shared"
```

## Concurrency & requirements

Retread requires a Pixi that provides `pixi-build-api-version >=4,<6`; that
protocol range is pinned in `recipe/recipe.yaml`. Rich platform envelopes such
as `platforms = [{ platform = "linux-64", glibc = "2.35" }]` require Pixi 0.71
or newer. Pixi 0.73 is the recommended, known-good version. Some sibling
backends, including `pixi-build-rattler-build` 0.4.4.20260707 and newer, require
API v5, so run `pixi self-update` when a workspace mixes backends.

Starting with retread 4.10.44, retread bounds rattler-build compression threads
node-wide per user through a PID-lease registry. A solo build gets full
parallelism; concurrent builds share the budget, and the sum of coordinated
grants never exceeds it. `RETREAD_COMPRESSION_THREADS` is a hard per-build
override, `RETREAD_COMPRESSION_BUDGET` overrides the node budget (which defaults
to available parallelism), and `RETREAD_THREAD_LEASE_DIR` overrides the
registry location, mainly for tests. The registry is node-local; remote
filesystems are rejected and retread uses a conservative fallback.

**Known Pixi issue (not a retread bug):** Pixi's build dispatch is not safe
under concurrent solves involving source packs. Two Pixi processes using one
workspace can panic at
`pixi_core/src/lock_file/resolve/build_dispatch.rs:477` with `could not
initialize build dispatch correctly`. Even one `pixi install --all` can fail
during concurrent environment solves with `build dispatch initialization
failed: failed to build <pack>`, which wraps the underlying error. Until Pixi
fixes this upstream, use `--concurrent-solves 1`, or persist the workspace
setting with `pixi config set --local concurrency.solves 1`. Never run two Pixi
processes against one workspace concurrently; serialize them externally (for
example, with `flock`). For genuinely parallel installs, use one clone per
install.

In Pixi 0.73, the local config command writes `[concurrency]` to
`.pixi/config.toml`; `pixi.toml` does not accept a `concurrency` key. A fresh
`pixi init` ignores `.pixi/*` but explicitly re-includes `.pixi/config.toml`, so
the workspace-local configuration can be committed.

The shared wheel store defaults to `~/.cache/retread/wheels` and can be
overridden with `RETREAD_WHEEL_STORE`. It is deliberately independent of
`RETREAD_CACHE_DIR` and fast-tmp. Do not delete it casually: lock records
reference its content SHAs.

## Config

```toml
[package.build.config]
retread-resolver = "uv"        # default
```
