# pixi-build-retread

A [pixi](https://pixi.sh) build backend that **repacks PyPI wheels as conda
packages with relaxed dependency pins**.

Some PyPI packages (Isaac Sim, parts of the ML stack) pin their dependencies
to exact patch versions, which makes a mixed `conda` + `pypi-dependencies`
solve very hard to satisfy alongside other workspace deps. Instead of
patching every conflict by hand with `pypi-options.dependency-overrides`,
retread reads the wheel's `METADATA`, widens the pins per a configurable
policy, and re-emits a real conda package via `rattler-build`.

The pattern is borrowed from
[the manual Isaac Sim recipe.yaml in prefix-dev/pixi#5230](https://github.com/prefix-dev/pixi/issues/5230#issuecomment-4-comment-24)
— automated, declarative, and reproducible.

## How it fits into pixi

retread is a standard pixi build backend (pixi-build API v4). It is
discovered exactly like `pixi-build-python` or `pixi-build-rattler-build`:
add a `[build]` table to your `pixi.toml` and name `pixi-build-retread` as
the backend.

```toml
[workspace]
preview = ["pixi-build"]
channels = ["https://prefix.dev/conda-forge"]

[package]
name = "isaacsim-repack"
version = "5.1.0"

[build]
backend = { name = "pixi-build-retread", version = "*" }
channels = ["https://prefix.dev/conda-forge"]  # plus wherever retread is hosted

[build.config]
relax = "minor"          # patch | minor | major | none
build-number = 0

# Same syntax as pixi's `[pypi-dependencies]`. Use `version` + optional
# `index` and `extras` to resolve from a PEP 503 simple index (PyPI public
# by default), or `url` + `sha256` for an explicit wheel.
[build.config.wheels]
isaacsim = { version = "==5.1.0", index = "https://pypi.nvidia.com", extras = ["all", "extscache"] }
mujoco   = { version = "==3.5.0", index = "https://py.mujoco.org" }
# foo    = { url = "https://example.com/foo-1.whl", sha256 = "..." }   # explicit URL fallback

[build.config.overrides]
# Per-PyPI-name overrides applied after the relax policy. Use to escape-hatch
# specific deps that need exact pinning or full unconstrain.
numpy = "==1.26.4"
torch = ">=2.7"

[build.config.name-map]
# PyPI -> conda name remaps on top of the built-in PEP 503 normalization.
opencv-python-headless = "py-opencv"
```

## Relax policies

| Policy   | `numpy==1.26.4` becomes  |
|----------|--------------------------|
| `none`   | `numpy ==1.26.4`         |
| `patch`  | `numpy >=1.26.4,<1.27`   |
| `minor`  | `numpy >=1.26,<2`        |
| `major`  | `numpy >=1`              |

Specifiers that aren't simple `==` pins (ranges, `~=`, etc.) are passed
through to conda match-spec syntax as-is.

## Local development

Pixi discovers build backends by name from conda channels — there is no
manifest-level `path =` field for backends. Three real options:

### Option A: env-var override (fastest dev loop)

Pixi honors `PIXI_BUILD_BACKEND_OVERRIDE=<name>=<path>` (see
`pixi/crates/pixi_build_frontend/src/backend_override.rs`). Build retread
once, export the env var, leave consumer manifests unchanged:

```bash
cd pixi-build-retread
cargo build --release
export PIXI_BUILD_BACKEND_OVERRIDE=pixi-build-retread=$(pwd)/target/release/pixi-build-retread

# now in any consumer workspace:
pixi install
```

### Option B: local conda channel

Build retread into a directory and add that directory as a channel:

```bash
pixi run -- rattler-build build --recipe recipe/recipe.yaml --output-dir ./local-channel
```

Consumer `pixi.toml`:

```toml
[build]
backend = { name = "pixi-build-retread", version = "*" }
channels = ["file:///abs/path/to/pixi-build-retread/local-channel", "https://prefix.dev/conda-forge"]
```

### Option C: publish to a remote channel

Same recipe, push to prefix.dev. Then consumers just need the channel URL.

`rattler-build` must be on `PATH` whenever retread runs — the binary shells
out to it for the actual conda build. The conda recipe declares
`rattler-build` as a runtime dep, so options B/C handle this automatically;
under option A, add `rattler-build = "*"` to the consumer workspace's
`[dependencies]`.

## Building the conda package

```bash
pixi run -- rattler-build build --recipe recipe/recipe.yaml
```

This produces a `.conda` that you can upload to prefix.dev or any conda
channel of your choosing. Once published, consumers no longer need the
`path =` backend reference — pixi will install the backend from the channel
on demand.

## Limitations / TODO

- Only direct wheel URLs are accepted. PyPI Simple resolution
  (`name==version` -> wheel URL) is not implemented yet.
- Extras are not honored (`isaacsim[all]` repacks `isaacsim`'s base deps
  only). The omnibus recipe in the upstream Isaac Sim work shows what
  multi-source recipes look like; that pattern is planned.
- The marker environment is hardcoded to linux/x86_64/CPython. The
  `host_platform` JSON-RPC parameter is read but not yet fully propagated
  to marker eval. Cross-arch builds will need this.
- `host_dependencies`/`build_dependencies` are coarse — host is just
  `python` + `pip`. We should also include `${{ compiler('c') }}` etc. when
  the wheel is platform-specific and ships shared libs. For now the
  `--no-deps` pip install handles it.

## Development

Hook setup once you've cloned:

```bash
pre-commit install
```

The pre-commit config runs `cargo fmt --check`, `cargo clippy -D warnings`,
and the default `cargo test` suite on every commit. Heavy network tests
(`tests/pypi_resolve_live.rs`'s `#[ignore]`d variants, `tests/e2e_ros_isaacsim.rs`)
are *not* run by the hook — they belong in CI.

To run them locally:

```bash
cargo test -- --include-ignored        # live PyPI + isaacsim wheel fetch
cargo test --test e2e_ros_isaacsim -- --include-ignored --nocapture
```

The e2e test needs `pixi` and `rattler-build` on PATH. Easiest way is
`pixi shell` in the project root, or use the dev env from
`/home/garylvov/projects/pixi`.

## Status

Pre-alpha. Tested only against pixi 0.62.x at the pinned `pixi_build_types`
git revision in `Cargo.toml`. Bump that rev together with the
`pixi-build-api-version` constraint in `recipe/recipe.yaml` when the pixi
protocol moves.

## Acknowledgements

This backend exists because of the design discussion in
[prefix-dev/pixi#5230](https://github.com/prefix-dev/pixi/issues/5230). Thank
you to:

- **The pixi team** at [prefix.dev](https://prefix.dev) for building pixi
  itself, the `pixi-build` protocol, and the broader rattler/conda-forge
  ecosystem this backend plugs into.
- **[@ruben-arts](https://github.com/ruben-arts)** and
  **[@tdejager](https://github.com/tdejager)** for the architectural
  guidance in pixi#5230 — in particular, [@tdejager](https://github.com/tdejager)
  surfaced the idea of "an extension that finds the correct overrides," which
  this backend is one concrete realization of.
- **[@diegoferigo-rai](https://github.com/diegoferigo-rai)** for the
  hand-written Isaac Sim `recipe.yaml` shared in
  [pixi#5230 (comment 24)](https://github.com/prefix-dev/pixi/issues/5230#issuecomment-comment-24),
  which is the static template this project automates.
