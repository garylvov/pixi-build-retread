# pixi-build-retread

A [pixi](https://pixi.sh) build backend that **repacks PyPI wheels as conda
packages with relaxed dependency pins** — so mixed conda + PyPI workspaces
(notably Isaac Sim + ROS 2) stop fighting upstream's exact-version pins.

Motivated by [prefix-dev/pixi#5230](https://github.com/prefix-dev/pixi/issues/5230).
Automates [@diegoferigo-rai](https://github.com/diegoferigo-rai)'s
hand-written
[Isaac Sim `recipe.yaml`](https://github.com/prefix-dev/pixi/issues/5230#issuecomment-comment-24).

## Requirements

- **pixi >= 0.63.0** — earlier versions only support pixi-build API
  versions 1-3; retread targets API v4 (where the `[package.build]`
  layout, `conda/outputs`, and `conda/build_v1` stabilized). If you
  hit `the constraint pixi-build-api-version >=1,<4 cannot be
  fulfilled`, upgrade pixi (`pixi self-update`) and re-run.
- **rattler-build** on `PATH` whenever pixi invokes retread — the
  conda recipe declares it as a runtime dep, so options that
  install retread via a conda channel handle this automatically.

## Use

retread is a pixi-build backend. pixi-build's model is **workspace
consumes source package** — two `pixi.toml` files. The workspace
depends on the source package via a path, and the source package
declares `[package.build]` pointing at retread.

```
your-project/
├── pixi.toml                # workspace -- your existing manifest
└── isaacsim-repack/
    └── pixi.toml            # source package -- retread config
```

### Workspace `pixi.toml` (your existing one)

Add `preview = ["pixi-build"]` and pull in the source package as a
regular conda dependency:

```toml
[workspace]
preview  = ["pixi-build"]                       # required for source packages
channels = ["https://prefix.dev/conda-forge"]

[dependencies]
isaacsim = { path = "./isaacsim-repack" }       # consumes the source package
# plus whatever else: ros-humble-*, python, pytorch-gpu, ...
```

### Source-package `pixi.toml` (the new file)

This is the entire content of `isaacsim-repack/pixi.toml`:

```toml
[package]
name    = "isaacsim"
version = "5.1.0"

[package.build]
backend = { name = "pixi-build-retread", version = "*", channels = ["https://prefix.dev/garylvov", "https://prefix.dev/conda-forge"] }

[package.build.config]
retread-relax        = "minor"   # patch | minor | major | none
retread-build-number = 0

# Same syntax as pixi's `[pypi-dependencies]`. `version` + optional
# `index` and `extras` resolves on a PEP 503 simple index; `url` +
# `sha256` is the explicit fallback.
[package.build.config.retread-wheels]
isaacsim = { version = "==5.1.0", index = "https://pypi.nvidia.com", extras = ["all", "extscache"] }
```

That's it. No overrides, no name-maps, no drop-deps. The auto-bundle
default handles PyPI-only transitives (aiodns, qdldl, ...) and
parselmouth supplies the standard PyPI<->conda name skews
(torch->pytorch, ...). For edge cases see [Escape hatches](#escape-hatches).

Worked example: [`examples/isaacsim/`](examples/isaacsim/).

## Escape hatches

When the auto-bundle path doesn't produce a working solve, four
opt-in knobs sit in `[package.build.config]`:

```toml
[package.build.config.retread-overrides]
# Replace the spec retread would emit. Use when the relaxed range
# doesn't match what conda channels have (e.g., conda-forge ships
# aiodns 3.0 but isaacsim pins 3.1 -- loosen here).
aiodns = "*"

[package.build.config.retread-name-map]
# Force a PyPI->conda name translation that parselmouth misses.
# (Most known cases are already in retread's FALLBACK list and
# don't need manual entries.)
some-pkg = "different-conda-name"

[package.build.config]
retread-conda-deps = ["pytorch"]    # keep on conda side, don't bundle
retread-drop-deps  = ["weird-shim"] # drop from run-deps entirely
```

## Multi-Python

retread fans `conda/outputs` over every Python version the workspace asks
for, picking the matching wheel from the index per version and skipping
versions whose wheels don't exist on the index (with a tracing warning).

Two ways to declare the target Python(s):

```toml
# In the consumer workspace -- preferred. Pixi forwards this to every
# build backend, so any source package automatically gets the right
# matrix without needing per-backend config.
[workspace.build-variants]
python = ["3.11", "3.12"]
```

```toml
# Or in the source package's [package.build.config] -- a shortcut for users
# who haven't declared build-variants. Accepts a string or a list.
[package.build.config]
python = "3.11"           # single version
# python = ["3.11", "3.12"]   # or a matrix
```

Precedence is `[workspace.build-variants]` > `[package.build.config] python` >
default `3.11`. Each variant point gets its own conda package with a
`build` string of `pyXY_<build_number>`, so pixi can resolve different
workspaces to different builds of the same source package.

## Relax policies

| Policy   | `numpy==1.26.4` becomes  |
|----------|--------------------------|
| `none`   | `numpy ==1.26.4`         |
| `patch`  | `numpy >=1.26.4,<1.27`   |
| `minor`  | `numpy >=1.26,<2`        |
| `major`  | `numpy >=1`              |

Non-`==` specifiers (ranges, `~=`, etc.) pass through unchanged.

## Local development

You can run the backend straight from a local checkout — useful when
trying changes before they hit the channel, or when working from a fork.
Pixi honors
[`PIXI_BUILD_BACKEND_OVERRIDE`](https://github.com/prefix-dev/pixi/blob/main/crates/pixi_build_frontend/src/backend_override.rs)
to substitute the executable while leaving the consumer's `pixi.toml`
unchanged:

```bash
# 1. Clone and build (one time).
git clone https://github.com/garylvov/pixi-build-retread.git
cd pixi-build-retread
cargo build --release

# 2. Export the override pointing at the local binary.
export PIXI_BUILD_BACKEND_OVERRIDE=pixi-build-retread=$(pwd)/target/release/pixi-build-retread

# 3. Work in any consumer workspace -- its [package.build] table still says
#    `backend = { name = "pixi-build-retread", ... }`, but pixi will
#    spawn the local binary instead of the channel copy.
cd ~/your-workspace
pixi install
```

To revert to the channel version: `unset PIXI_BUILD_BACKEND_OVERRIDE`.

`rattler-build` must be on `PATH` whenever retread runs (it shells out
for the actual conda build). Easiest: open a pixi shell from this repo
(`pixi shell` in `pixi-build-retread/`) before exporting the override —
the dev env declares `rattler-build` as a dep.

### Contributing

Pre-commit hooks (`cargo fmt`, `clippy`, fast `cargo test`):

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
