# pixi-build-retread

[![linux-x86_64](https://github.com/garylvov/pixi-build-retread/actions/workflows/ci.yml/badge.svg)](https://github.com/garylvov/pixi-build-retread/actions/workflows/ci.yml)

**Retread relaxes strict PyPI dependency pins, prefers the Conda equivalent for any shared transitive, and iteratively reconciles conflicts at the fixed boundary between Pixi's Conda and uv's PyPI solver.**  Ships as a statically linked musl binary: runs on any ``x86_64`` Linux distro, with no glibc version requirement.

[Pixi](https://pixi.sh) resolves Conda packages first, then runs `uv` against PyPI using Conda's selections as hard pins. This handoff frequently fails because upstream wheels often strictly pin their own transitive dependencies. When Conda commits to one version and an upstream wheel demands another, `uv` gets trapped in the middle and the installation fails. Retread solves this by rewriting exact pins into flexible ranges within both the wheel's `METADATA` and the emitted Conda run-dependencies. Using [parselmouth](https://github.com/prefix-dev/parselmouth) and a small fallback table, Retread intercepts shared transitives and routes them to their Conda equivalents *before* `uv` runs. It iteratively re-solves the metadata until the Conda/uv boundary stabilizes, leaving any remaining PyPI-only dependencies untouched in the metadata so `uv` can seamlessly resolve them.

Motivated by [prefix-dev/pixi#5230](https://github.com/prefix-dev/pixi/issues/5230);
automates [@diegoferigo-rai](https://github.com/diegoferigo-rai)'s hand-written
[Isaac Sim `recipe.yaml`](https://github.com/prefix-dev/pixi/issues/5230#issuecomment-comment-24).

The [`examples/gigastrap/`](examples/gigastrap/) workspace (Isaac Sim + IsaacLab + IsaacLab-Arena + pytorch3d, mixed with ROS 2
and a GPU pytorch stack) demonstrates this approach, which can be installed via `pixi install -e gsi`.

## Requirements

- **Pixi >= 0.63.0** — targets pixi-build API v4. Older Pixi → `pixi self-update`.
- **rattler-build** and **uv** on `PATH` when Pixi invokes retread (both are
  declared as runtime deps in the conda recipe, so a channel install handles
  them). uv supplies the python for source builds on demand; retread ships no
  python and pins to none.

## Use

pixi-build's model is **workspace consumes source package** — two separate
`pixi.toml`s. The second one is a *new* manifest you create **inside the
`isaac-pack/` subdirectory**; it is not part of your workspace manifest:

```
your-project/
├── pixi.toml          # 1. workspace -- your existing manifest
└── isaac-pack/        # the source-package directory, can be different name, where the PyPi deps go
    └── pixi.toml      # 2. a SECOND manifest, living inside isaac-pack/ -- retread config
```

### Workspace `pixi.toml` (your existing manifest, at the project root)

Add `preview = ["pixi-build"]` and declare the source package. With
`retread-bundle = "<name>"` set (below), every `[retread-wheels]` entry
collapses into one conda output, so the workspace declares that single name
and gets the whole pack:

```toml
[workspace]
preview  = ["pixi-build"]
channels = ["https://prefix.dev/conda-forge"]

[dependencies]
isaac-pack = { path = "./isaac-pack" }   # one decl pulls the whole pack
# plus your usual deps: python, pytorch-gpu, ros-humble-*, ...
```

### Source-package `pixi.toml` (the second manifest, at `isaac-pack/pixi.toml`)

```toml
[package]
name    = "isaac-pack"
version = "5.1.0"

[package.build]
backend  = { name = "pixi-build-retread", version = ">=1.2.0" }
channels = ["https://prefix.dev/garylvov", "https://prefix.dev/conda-forge"]

[package.build.config]
retread-build-number = 0
# Pack target python. Defaults to 3.11. A list (["3.11","3.12"]) needs every
# entry to ship a wheel for every python -- see Multi-Python.
retread-python       = "3.11"
# Default bundle group (v1.4.0): every entry below lands in this one conda
# output. Omit it and each entry produces its own output; a per-entry
# `bundle = "..."` overrides the default for mixed grouping.
retread-bundle       = "isaac-pack"
# retread-relax is OPTIONAL; default is the solve-driven cascade (see below).

# Named git sources, referenced by `from = "<name>"`. Keeps each rev in one place.
[package.build.config.retread-git-sources.isaaclab]
url = "https://github.com/isaac-sim/IsaacLab.git"
rev = "54cf64beb4eee99bc7b78e0353c8a4a8a13aa2c0"

[package.build.config.retread-git-sources.isaaclab-arena]
url = "https://github.com/isaac-sim/IsaacLab-Arena"
rev = "867cbf9b7b4edbb03f32e1209c585a38cb3d8edf"

# Wheels to repack. Each entry is one of five source forms:
#   version (+ index, extras)                -> PyPI Simple
#   url (+ sha256)                           -> direct download
#   path (+ extras)                          -> uv build local dir
#   git + rev (+ subdirectory, extras)       -> uv build git (inline)
#   from = "<name>" (+ subdirectory, extras) -> uv build git (named source above)
# extras resolve extras-gated Requires-Dist (incl. `pkg @ git+...` / `pkg @
# https://...whl`) as sub-wheels. Mix freely; each gets the same treatment.
[package.build.config.retread-wheels]
isaacsim        = { version = "==5.1.0", index = "https://pypi.nvidia.com", extras = ["all", "extscache"] }
isaaclab        = { from = "isaaclab", subdirectory = "source/isaaclab" }
isaaclab-assets = { from = "isaaclab", subdirectory = "source/isaaclab_assets" }
isaaclab-tasks  = { from = "isaaclab", subdirectory = "source/isaaclab_tasks" }
isaaclab-rl     = { from = "isaaclab", subdirectory = "source/isaaclab_rl", extras = ["all"] }
isaaclab-mimic  = { from = "isaaclab", subdirectory = "source/isaaclab_mimic" }
isaaclab-arena  = { from = "isaaclab-arena" }   # subdirectory defaults to "."
```

Worked example: [`examples/gigastrap/`](examples/gigastrap/).

### Files retread writes into the pack (`<pack>/`)

```
isaac-pack/ # or preferred name
├── pixi.toml
├── retread-audit-<name>.json         # what was relaxed/emitted (after build)
├── retread-probe-trace-<name>.json   # per-env probe + solve diagnostics (during conda/outputs)
├── RETREAD-SOLVE-FAILED-<name>.md    # human summary, only when a solve is UNSAT
└── wheels/<entry>/...                # materialized wheels + post-rewrite *.relaxed.whl
```

- **audit** (post-build): per-wheel pre/post `Requires-Dist`, emitted conda
  run-deps, copy-paste-ready TOML blocks, probe + solve diagnostics.
- **probe-trace** (during `conda/outputs`, always written): per-dep routing
  plus an `{env -> solve_diagnostics}` map with `refinement_steps`. Grep this
  when Pixi prints a misleading leaf error to see what conda's solver really
  tripped on.
- **SOLVE-FAILED.md** (UNSAT only): per-env refinement history, the real unsat
  chain, and an action checklist (which blocker is yours vs retread's). Absent
  means every env solved.

`wheels/` is multi-GB (NVIDIA) — gitignore it. Delete a per-entry folder or
all of `wheels/` to force re-materialization on the next solve.

### Auto-injected checkout-root data

For path/git sources, retread ships the upstream repo root's non-Python assets
(honoring `.gitignore`) as wheel `.data/data/lib/<rel>` entries, which pip
extracts to `$CONDA_PREFIX/lib/<rel>`. This makes `__file__`-relative asset
lookups (the IsaacLab `.kit` pattern) resolve without an editable overlay. One
wheel per bundle carries it (deduped); `wheels[].auto_data` in the audit shows
what shipped.

### Editable overlay (optional)

To *edit* a bundled package live, do NOT also list it as an editable
`pypi-dependency` — uv reads its strict pyproject pins and they collide with
conda's picks. Instead overlay after `pixi install`:

```bash
python3 -m pip install -e ./IsaacLab/source/isaaclab --no-deps --force-reinstall
```

`--no-deps` keeps retread's resolution; this swaps only the importable code.
Wrap it in a Pixi task or activation hook.

## Escape hatches

Most packs need none of these — parselmouth plus a built-in fallback table
(`torch`→`pytorch`, `pywin32`, `opencv-python`→`opencv`, …) handle the common
name skews, and the cascade widens automatically. When the auto path still
doesn't solve, opt-in knobs live in `[package.build.config]`:

```toml
[package.build.config.retread-overrides]
aiodns = "*"                          # replace the spec retread would emit

[package.build.config.retread-name-map]
some-pkg = "different-conda-name"     # force a PyPI->conda name parselmouth misses

[package.build.config]
retread-conda-deps = ["pytorch"]      # keep on conda side, don't bundle
retread-drop-deps  = ["weird-shim"]   # drop from run-deps entirely
```

## Relax policies

| Policy | `numpy==1.26.4` → | `pyglet<2` → | Auto-widen unsat? |
|---|---|---|---|
| `patch-then-minor-then-major-then-last-resort` ★ (Default) | `>=1.26.4,<1.27` | `<2` | yes (progressive) |
| `none` | `==1.26.4` | `<2` | no |
| `patch` | `>=1.26.4,<1.27` | `<2` | no |
| `minor` | `>=1.26,<2` | `<2` | no |
| `major` | `>=1` | `<2` | no |
| `strong-major` | `>=1` | `pyglet` (cap stripped) | no |
| `*-with-last-resort` (patch/minor/major) | as base | `<2` | yes (`*`) |


★ = **default** (omit `retread-relax`). Non-`==` specs pass through unchanged
except under `strong-major`, which strips upper bounds. `python` is exempt
from every policy (widening it loses ABI meaning / breaks rattler-build).

**The default, solve-driven cascade:** start from the narrowest safe rewrite,
run a real `rattler_solve` over (workspace deps + emitted run-deps) on the
workspace's channels, and on unsat iteratively widen only the retread-emitted
blocker that is actually causing the Pixi-to-uv solver handoff to fail. If
the dominant constraint belongs to the *workspace*, retread stops widening,
surfaces that conflict clearly, and emits a workspace-edit suggestion instead
of thrashing. When a dep provably has **zero conda candidates at any version**
(PyPI-only packages — `isaacsim-*`, `nvidia-*-cu1x`, …), the cascade bundles
the wheel from PyPI and drops the conda emission automatically — no
`retread-drop-deps` needed. The index fallback chain is built from the
manifests, nothing vendor-specific: each `[retread-wheels]` entry's `index`,
then any workspace `[pypi-options]` / `[feature.X.pypi-options]` `index-url` +
`extra-index-urls`, then public PyPI. The reroute happens only on a definitive probe
(a channel fetch failure never reroutes) and never for deps the pack pins to
conda via `retread-conda-deps`; the audit records it as
`auto-pypi-no-conda-candidates`. Every step lands in the probe-trace's `refinement_steps`. The
check honors `[workspace].channel-priority` (default `strict`, matching Pixi).
Pixi shows backend errors verbatim, so on a hard conflict you see retread's
diagnostic, not Pixi's misleading leaf.

## Multi-Python

retread is python-agnostic — one artifact per platform. It builds wheels via
`uv pip wheel --python <ver>` (uv fetches python-build-standalone on demand),
fans `conda/outputs` over each requested python, and picks the matching wheel
per version. A multi-python pack therefore works **only if every entry ships a
wheel for every requested python** (otherwise that variant fails fast). Isaac
Sim 5.1.0 is cp311-only on `pypi.nvidia.com`; git/path sources track the
python automatically.

Declare the target python(s), precedence high→low:

```toml
# workspace (preferred -- forwarded to every backend)
[workspace.build-variants]
python = ["3.11", "3.12"]
```
```toml
# or per source package
[package.build.config]
retread-python = "3.11"   # or ["3.11", "3.12"]
```

Each variant gets its own conda build string (`pyXY_<n>`).

### pytorch / CUDA family

When a bundle uses `torch`/`torchaudio`/`torchvision`, the conda solver needs
matching GPU builds. conda-forge ships `pytorch-gpu` plus `torchvision` /
`torchaudio` with `pytorch * cuda*` build tags; pin `cuda-version` to Isaac
Sim's tested CUDA (e.g. `==12.8`). Scope niche channels per-feature
(`[feature.gpu.channels]`) so CPU/ROS envs aren't forced through them.

## Local development

### Option A: `file://` channel (recommended)

```bash
git clone https://github.com/garylvov/pixi-build-retread.git && cd pixi-build-retread
pixi run -- rattler-build build --recipe recipe/recipe.yaml --output-dir ./local-channel
```

Point the source package's backend channels at it:

```toml
[package.build]
backend = { name = "pixi-build-retread", version = "*", channels = [
  "file:///abs/path/to/pixi-build-retread/local-channel",
  "https://prefix.dev/conda-forge",
] }
```

`bash scripts/rebuild-local.sh` does the whole nuke-rebuild-verify dance (bump
`Cargo.toml` + `recipe/recipe.yaml` to the same version first; it aborts on
mismatch). `CONSUMER_PROJECT=/abs/path bash scripts/rebuild-local.sh` also
clears that workspace's Pixi caches. The script exists because three caches
otherwise serve a stale build: the channel's appended `repodata.json`, Pixi's
backend-executable cache (`~/.cache/rattler/cache/backends-v0/`), and retread's
git-clone cache. Never delete `local-channel/noarch/repodata.json` —
rattler-build needs that empty placeholder.

### Option B: `PIXI_BUILD_BACKEND_OVERRIDE` (faster, less deterministic)

```bash
cargo build --release
export PIXI_BUILD_BACKEND_OVERRIDE=pixi-build-retread=$(pwd)/target/release/pixi-build-retread
```

Faster inner loop, but Pixi may reuse cached metadata — use Option A when in
doubt. `rattler-build` must be on `PATH` whenever retread runs.

### Contributing

```bash
pre-commit install                # cargo fmt + clippy + fast cargo test
cargo test -- --include-ignored   # heavy live tests (network, multi-GB)
```

CI (GitHub Actions) runs fmt, clippy, and the fast test suite on every push
and PR.

## Acknowledgements

The [prefix.dev](https://prefix.dev) team for Pixi.
[@ruben-arts](https://github.com/ruben-arts) and
[@tdejager](https://github.com/tdejager) for the
[pixi#5230](https://github.com/prefix-dev/pixi/issues/5230) discussion that
suggested the "extension that finds the correct overrides" shape.
[@diegoferigo-rai](https://github.com/diegoferigo-rai) for the static
`recipe.yaml` this project automates.
