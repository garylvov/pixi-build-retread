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
backend = { name = "pixi-build-retread", version = ">=4.0.0" }
[package.build.config]
retread-bundle = "isaac-pack"
[package.build.config.retread-wheels]
isaacsim = { version = "==5.1.0", index = "https://pypi.nvidia.com" }
```

Run: `pixi install`

## Pack Configuration Example
A pack's `pixi.toml` (e.g., `examples/isaac6/isaac-pack/`) declares package metadata and wheels to bundle:
```toml
[package]
name = "isaac-pack-6"
version = "6.0.0.1"
[package.build]
backend = { name = "pixi-build-retread", version = "*" }
[package.build.config]
retread-python   = "3.12"
retread-bundle   = "isaac-pack-6"
retread-resolver = "uv"
[package.build.config.retread-wheels]
isaacsim = { version = "==6.0.0.1", index = "https://pypi.nvidia.com" }
# ...
```

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
