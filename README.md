# pixi-build-retread

A [pixi](https://pixi.sh) build backend that **repacks PyPI wheels as conda
packages with relaxed dependency pins** — so mixed conda + PyPI workspaces
(notably Isaac Sim + ROS 2) stop fighting upstream's exact-version pins.

Motivated by [prefix-dev/pixi#5230](https://github.com/prefix-dev/pixi/issues/5230).
Automates [@diegoferigo-rai](https://github.com/diegoferigo-rai)'s
hand-written
[Isaac Sim `recipe.yaml`](https://github.com/prefix-dev/pixi/issues/5230#issuecomment-comment-24).

## Use

```toml
[workspace]
preview  = ["pixi-build"]
channels = ["https://prefix.dev/conda-forge"]

[package]
name    = "isaacsim-repack"
version = "5.1.0"

[build]
backend  = { name = "pixi-build-retread", version = "*" }
channels = ["https://prefix.dev/garylvov", "https://prefix.dev/conda-forge"]

[build.config]
relax        = "minor"   # patch | minor | major | none
build-number = 0

# Same syntax as `[pypi-dependencies]`. `version` + optional `index` and
# `extras` resolves on a PEP 503 simple index; `url` + `sha256` is the
# explicit fallback.
[build.config.retread-wheels]
isaacsim = { version = "==5.1.0", index = "https://pypi.nvidia.com", extras = ["all", "extscache"] }

# Escape hatch for upstream conflicts the relax policy can't resolve.
[build.config.overrides]
numpy = ">=1.26,<2"

# PyPI -> conda name remap on top of PEP 503 normalization.
[build.config.name-map]
opencv-python-headless = "py-opencv"
```

Worked example with conda ros2-humble: [`examples/isaacsim/`](examples/isaacsim/).

## Relax policies

| Policy   | `numpy==1.26.4` becomes  |
|----------|--------------------------|
| `none`   | `numpy ==1.26.4`         |
| `patch`  | `numpy >=1.26.4,<1.27`   |
| `minor`  | `numpy >=1.26,<2`        |
| `major`  | `numpy >=1`              |

Non-`==` specifiers (ranges, `~=`, etc.) pass through unchanged.

## Local development

To point pixi at a checkout instead of the published channel, build the
binary and export
[`PIXI_BUILD_BACKEND_OVERRIDE`](https://github.com/prefix-dev/pixi/blob/main/crates/pixi_build_frontend/src/backend_override.rs):

```bash
cargo build --release
export PIXI_BUILD_BACKEND_OVERRIDE=pixi-build-retread=$(pwd)/target/release/pixi-build-retread
# now run pixi anywhere; it'll use this binary instead of the channel copy
```

`rattler-build` must be on `PATH` (retread shells out to it). Add it to
the consumer workspace's `[dependencies]` if it isn't already.

Set up the pre-commit hooks (`cargo fmt`, `clippy`, fast `cargo test`):

```bash
pre-commit install
```

Heavy live tests are gated behind `#[ignore]`:

```bash
cargo test -- --include-ignored
cargo test --test e2e_ros_isaacsim -- --include-ignored --nocapture  # needs pixi + rattler-build
```

## Acknowledgements

The pixi team at [prefix.dev](https://prefix.dev) for pixi itself.
[@ruben-arts](https://github.com/ruben-arts) and
[@tdejager](https://github.com/tdejager) for the
[pixi#5230](https://github.com/prefix-dev/pixi/issues/5230) discussion that
suggested the "extension that finds the correct overrides" shape.
[@diegoferigo-rai](https://github.com/diegoferigo-rai) for the static
`recipe.yaml` this project automates.
