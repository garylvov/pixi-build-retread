# pixi-build-retread

## this repo is fundamentally flawed and is in the process of being fixed; don't use for now

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

## Rust toolchain

The exact Rust version is pinned in `rust-toolchain.toml` (kept in sync with
CI's `dtolnay/rust-toolchain` pin) so `fmt`/`clippy` never drift between local
and CI. Install it with [rustup](https://rustup.rs) (which reads
`rust-toolchain.toml` automatically) -- `pixi exec --spec rust` pulls from
conda-forge, which lags the pinned version and will disagree on `fmt`/
`clippy` output.

## Commands

| Command | Usage |
|---------|-------|
| install | `retread install --lock <lock.json> --prefix <p>` |
| verify | `retread verify --lock <lock.json> [--full]` |
| solve | `retread solve --manifest pixi.toml -e <env>` |
| fast (env) | `eval "$(pixi-build-retread fast --print-env)"` |
| fast (persist) | `pixi-build-retread fast --persist <env>` |

## Fast path

For shared-filesystem machines (SLURM clusters, EC2+EFS) where the project and caches
sit on slow shared storage: `eval "$(pixi-build-retread fast --print-env)"` reroutes
caches and env storage to fast machine-local disk. `pixi install` then materializes the
env by parallel copy from the `envs-persist` snapshot (~1.5 min for 40 GB, lock hash
match) or a no-resolve frozen rebuild (~3 min); `fast --persist <env>` writes/updates
the snapshot after env changes. On a local machine with fast disk it auto-disengages.

## Config

Default solver is uv; set `retread-resolver = "legacy"` to fall back.

```toml
[package.build.config]
retread-resolver = "uv"        # default; or "legacy" fallback

[tool.retread.fast-tmp]
mode = "auto"                  # auto | on | off
tmp-root = "/tmp"
blob-caches = "shared"
```
