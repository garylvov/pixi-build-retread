# pixi-build-retread - doesn't work yet, fixes inbound

**Retread PyPI wheels as conda packages, with dep versions loosened and
shared deps preferred from conda.**

A [pixi](https://pixi.sh) build backend. Pixi solves conda first, then
runs uv against PyPI with conda's chosen versions forwarded as hard
pins. Upstream PyPI wheels routinely pin their transitives exactly
(`Requires-Dist: numpy==1.26.0`), so those forwarded pins clash with
what the wheel demands and the install fails.

retread fixes both sides: it rewrites each exact pin to a range
(default `>=X.Y,<X+1`) in the wheel's METADATA *and* the emitted
conda run-deps, and reroutes any shared transitive that has a conda
equivalent (via parselmouth + a small fallback table) onto the conda
side *before* uv runs. Deps with no conda equivalent stay bundled in
the wheel and skip uv entirely.

Motivated by [prefix-dev/pixi#5230](https://github.com/prefix-dev/pixi/issues/5230).
Automates [@diegoferigo-rai](https://github.com/diegoferigo-rai)'s
hand-written
[Isaac Sim `recipe.yaml`](https://github.com/prefix-dev/pixi/issues/5230#issuecomment-comment-24).

## Requirements

- **pixi >= 0.63.0**: earlier versions only support pixi-build API
  versions 1-3; retread targets API v4 (where the `[package.build]`
  layout, `conda/outputs`, and `conda/build_v1` stabilized). If you
  hit `the constraint pixi-build-api-version >=1,<4 cannot be
  fulfilled`, upgrade pixi (`pixi self-update`) and re-run.
- **rattler-build** on `PATH` whenever pixi invokes retread. The
  conda recipe declares it as a runtime dep, so installations that
  install retread via a conda channel handle this automatically.
- **uv** on `PATH` whenever retread builds path/git sources. Also
  declared as a runtime dep in the conda recipe. uv provides the
  python interpreter for source builds; retread does NOT ship its
  own python and does NOT pin to one. Any python the workspace
  asks for is downloaded on demand by uv on first use.

## Use

retread is a pixi-build backend. pixi-build's model is **workspace
consumes source package**: two `pixi.toml` files. The workspace
depends on the source package via a path, and the source package
declares `[package.build]` pointing at retread.

```
your-project/
├── pixi.toml                # workspace -- your existing manifest
└── isaacsim-repack/
    └── pixi.toml            # source package -- retread config
```

### Workspace `pixi.toml` (your existing one)

Add `preview = ["pixi-build"]` and declare the source package. Every
`[retread-wheels]` entry sets `bundle = "<name>"` (see the
source-package config below) so they all collapse into one conda
output. The workspace declares that single name and pixi-build
installs every wheel in the pack:

```toml
[workspace]
preview  = ["pixi-build"]
channels = ["https://prefix.dev/conda-forge"]

[dependencies]
isaac-pack = { path = "./isaac-pack" }          # one decl pulls everything
# plus whatever else: ros-humble-*, python, pytorch-gpu, ...
```

### Source-package `pixi.toml` (the new file)

This is the entire content of `isaac-pack/pixi.toml`:

```toml
[package]
name    = "isaacsim"
version = "5.1.0"

[package.build]
# Pin to the latest published retread version (see
# https://prefix.dev/channels/garylvov/packages/pixi-build-retread).
# Bump as new releases ship so pixi reproducibly picks the same
# backend across machines. Use `version = "*"` only if you're OK
# with auto-picking the highest available on every solve.
backend = { name = "pixi-build-retread", version = ">=0.22.0" }
channels = [
  "https://prefix.dev/garylvov",
  "https://prefix.dev/conda-forge",
]

[package.build.config]
# Recommended default. Behaves like `minor` (preserves upstream ranges,
# widens `==X.Y.Z` pins to `>=X.Y,<X+1`) plus an automated last-resort
# cascade for deps whose post-relax conda spec is unsatisfiable on the
# workspace's channels (probes conda, widens to `*` only for those
# specific deps). Eliminates hand-edited `[retread-overrides]` for the
# `pyglet`-class pins. See "Relax policies" below for the full table.
retread-relax        = "minor-with-last-resort"
retread-build-number = 0
# Recommended: pin the python version this pack targets. Without it,
# retread falls back to DEFAULT_PYTHON (3.11) when pixi forwards only a
# bare-major variant value. NVIDIA's isaacsim only ships cp311 on
# `pypi.nvidia.com/isaacsim/`, so the isaac-pack example is locked to
# 3.11. A list (`["3.11", "3.12"]`) works only when every entry has a
# wheel for every requested python -- see the Multi-Python section.
retread-python       = "3.11"

# Named git sources -- referenced by `from = "<name>"` below. Avoids
# repeating the rev across every sub-package entry. Bump the rev here
# when you want to advance the pinned commit.
[package.build.config.retread-git-sources.isaaclab]
url = "https://github.com/isaac-sim/IsaacLab.git"
rev = "54cf64beb4eee99bc7b78e0353c8a4a8a13aa2c0"

[package.build.config.retread-git-sources.isaaclab-arena]
url = "https://github.com/isaac-sim/IsaacLab-Arena"
rev = "867cbf9b7b4edbb03f32e1209c585a38cb3d8edf"

# Wheels to repack. Five source forms accepted:
#  * `version` + optional `index` + optional `extras`  → PyPI Simple
#  * `url` + optional `sha256`                         → direct download
#  * `path = "./relative/dir"` + optional `extras`     → pip wheel local
#  * `git` + `rev` + optional `subdirectory` + optional `extras`
#                                                      → pip wheel git (inline)
#  * `from = "<name>"` + optional `subdirectory` + optional `extras`
#                                                      → pip wheel git (named)
#
# Source-form extras (path/git/from) follow the same BFS the PyPI form
# uses: each extras-gated `Requires-Dist: <pkg>; extra == "<name>"` in
# the built wheel's METADATA gets resolved as a sub-wheel. URL
# Requires-Dist (PEP 508 `pkg @ git+https://...@<rev>` or `pkg @
# https://.../file.whl`) is also handled -- the git URL clones and
# `pip wheel`'s; the direct URL downloads. Unlocks IsaacLab's
# `rl_games` extra (`rl-games @ git+https://github.com/isaac-sim/
# rl_games.git@python3.11`) without hand-maintaining a post-install
# pip install.
#
# Mix freely. Each gets the same METADATA-rewrite + bundle treatment.
[package.build.config.retread-wheels]
isaacsim = { version = "==5.1.0", index = "https://pypi.nvidia.com", 
              extras = ["all", "extscache"] }

# IsaacLab sub-packages -- all six are needed because they reference
# each other (isaaclab pulls isaaclab-assets / isaaclab-tasks;
# isaaclab-rl pulls isaaclab; etc.). All resolve via the named source
# above, so the rev appears in ONE place. `isaaclab-rl` uses
# `extras = ["all"]` to pull in stable-baselines3, skrl, rl-games (via
# its git URL), and rsl-rl -- everything that ships under the upstream
# `all` extra.
isaaclab        = { from = "isaaclab", subdirectory = "source/isaaclab" }
isaaclab-assets = { from = "isaaclab", subdirectory = "source/isaaclab_assets" }
isaaclab-tasks  = { from = "isaaclab", subdirectory = "source/isaaclab_tasks" }
isaaclab-rl     = { from = "isaaclab", subdirectory = "source/isaaclab_rl", extras = ["all"] }
isaaclab-mimic  = { from = "isaaclab", subdirectory = "source/isaaclab_mimic" }
isaaclab-arena  = { from = "isaaclab-arena" }   # subdirectory defaults to "."
```

### Wheel cache + audit (`<pack>/`)

Every wheel retread materializes (direct downloads, PyPI resolves,
`pip wheel` outputs from path/git sources, and the post-D `*.relaxed.whl`
copies) lands inside the source-package directory under
`./wheels/<entry_name>/`. Side-by-side with the pack's `pixi.toml` so
you can inspect or diff the exact bytes rattler-build will pick up.

Each conda output also drops a `retread-audit-<conda_name>.json` next
to `pixi.toml`: per-bundled-wheel pre-D `Requires-Dist`, post-translate
conda run-deps, and copy-paste-ready `[dependencies]` /
`[pypi-options.dependency-overrides]` TOML blocks. Use it to see
exactly what was relaxed, requested, and emitted on the last solve.

```
isaac-pack/
├── pixi.toml
├── retread-audit-isaac-pack.json        # what was relaxed / emitted
└── wheels/
    ├── isaacsim/
    │   ├── isaacsim-5.1.0-cp311-...-x86_64.whl
    │   └── isaacsim-5.1.0-cp311-...-x86_64.relaxed.whl   # post-D
    ├── isaaclab/
    │   └── isaaclab-0.0.1-py3-none-any.relaxed.whl
    └── ...
```

Add `wheels/` (and `retread-audit-*.json` if you don't want it
checked in) to the pack's `.gitignore`; the NVIDIA wheels alone are
multi-GB. To force re-materialization (after bumping a git rev,
changing the relax policy, etc.) delete the per-entry folder or the
whole `wheels/` tree and re-run pixi; retread re-fetches/re-builds on
the next solve:

```bash
rm -rf isaac-pack/wheels                  # nuke everything
rm -rf isaac-pack/wheels/isaaclab         # or just one entry
rm -f  isaac-pack/retread-audit-*.json    # audits are regenerated on every solve
```

You still need to clear pixi's caches alongside it when iterating on
retread itself; see the "Local development" section below for the
full incantation.

### Auto-injected checkout-root data (v0.12.0+)

For path/git/named-git entries, retread walks the **upstream repo
root** (parent of the entry's `subdirectory`) honoring its own
`.gitignore` and ships every non-ignored, non-sibling-subdirectory
file as a wheel `.data/data/lib/<rel>` entry. Pip extracts those to
`$CONDA_PREFIX/lib/<rel>` at install time. Hardcoded floor of names
that are skipped even without `.gitignore` coverage: `__pycache__`,
`.pixi`, `.venv`, `venv`, `node_modules`, `target`.

Why it exists: many packages do `__file__` arithmetic to find non-
Python assets adjacent to the package source (the IsaacLab pattern:
`dirname(__file__) + *[".."] * 4 + "apps"`). When the wheel only
ships `source/<pkg>/`, that arithmetic from
`<env>/lib/python3.X/site-packages/<pkg>/...` lands at
`<env>/lib/<asset>` -- which doesn't exist unless we put it there.
Auto-data inject does. Solves the "everything imports but the .kit
files are missing" class of failure without anyone declaring data
paths manually.

Dedup: when N bundle entries share one checkout (common -- all 6
IsaacLab sub-packages clone the same repo), the auto-data ships on
exactly ONE wheel of the bundle. The walk also skips every
`subdirectory` of every sibling entry sharing the root, so the
Python package source that `pip wheel` already put in site-packages
doesn't get double-shipped into `<env>/lib/source/<pkg>/`.

The `wheels[].auto_data` field in `retread-audit-*.json` surfaces
which wheel carried the inject, how many files it shipped, and
which subdirs were skipped.

### Editable overlay (optional in v0.12.0+; required pre-v0.12.0 for `__file__` asset access)

The auto-data inject above means a bundled IsaacLab is now
self-sufficient at runtime -- `.kit` experience files resolve at
`<env>/lib/apps/` without any editable overlay. The overlay below is
**still useful** if you want to *edit* one of the bundled packages
and have changes picked up live (hot-reload, debugging). Skip this
section entirely if you only need to RUN the bundled code.

If you want to edit one of the bundled packages (typical for IsaacLab
hot-reload), let retread own dep resolution and overlay editable
AFTER `pixi install`. Two rules:

1. **Do NOT also list the package as an editable `pypi-dependency` in
   the workspace `pixi.toml`.** uv reads editable pyproject.toml pins at
   solve time and forwards them to its solver. They collide with
   whatever conda picked for the same transitives -- e.g. IsaacLab
   pinning `pillow==11.2.1` while matplotlib (via conda) brings in
   `pillow==12.0.0`. The failure shows up as:
   > Because isaaclab==X depends on pillow==11.2.1 and pillow==12.0.0,
   > we can conclude that isaaclab==X cannot be used.
   That conflict cannot be fixed from inside retread because uv never
   looks at the wheel METADATA we rewrote -- it reads the editable
   pyproject directly. Solution: take the editable entry OUT of the
   workspace pixi.toml entirely.

2. **Overlay editable in a post-install step**, with `--no-deps`
   (don't re-read the strict pins) and `--force-reinstall` (replace
   the snapshot retread put in site-packages). Invoke via `python3 -m
   pip` so the env's own pip is used regardless of what's on `PATH`:

   ```bash
   python3 -m pip install -e ./IsaacLab/source/isaaclab --no-deps --force-reinstall
   # repeat per sub-package
   ```

   Wrap it in a pixi task or activation hook so it runs after every
   `pixi install`.

The bundle handles dep resolution; the editable overlay swaps only the
importable code in site-packages, without re-introducing the pins.

That's it for the source package. No overrides, no name-maps,
no drop-deps. The auto-bundle default handles PyPI-only transitives
(aiodns, qdldl, ...) and parselmouth supplies the standard
PyPI<->conda name skews (torch->pytorch, ...). For edge cases see
[Escape hatches](#escape-hatches).

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
# Widens a single upstream cap to `*`. Use when an exact conflict in
# your workspace traces back to this pin (check `retread-audit-*.json`
# + the conda solver error). `retread-relax = "strong-major"` strips
# every upper bound at once and avoids per-package overrides.
pyglet = "*"

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

**As of 0.11.0 retread itself is python-agnostic**: one retread
artifact per platform, ever. retread uses `uv pip wheel --python <ver>`
internally, and uv downloads
[python-build-standalone](https://github.com/astral-sh/python-build-standalone)
binaries on demand (cached under `~/.cache/uv/python/`). Any python
the workspace requests just works (no retread rebuild needed for new
python releases).

retread fans `conda/outputs` over every Python version the workspace
asks for, picking the matching wheel from the index per version. If any
bundled entry has no wheel for a requested python, the whole solve for
that python variant fails fast with `no wheel for <name> at <index>
matches target python=<X.Y>`. This is conservative; partial bundles
would silently miss code.

Implication: multi-python packs only work when **every entry has a
wheel for every requested python**. For Isaac Sim 5.1.0 that's py3.11
only (NVIDIA's `pypi.nvidia.com/isaacsim/` ships cp310/cp311, not
cp312). If you need a py3.12 env, build a separate pack against an
upstream that ships cp312 wheels, OR use git/path sources whose wheel
is rebuilt under each python's `uv pip wheel` (then the cp tag tracks
the python automatically).

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
retread-python = "3.11"           # single version
# retread-python = ["3.11", "3.12"]   # or a matrix
```

Precedence is `[workspace.build-variants]` > `[package.build.config] retread-python` >
default `3.11`. Each variant point gets its own conda package with a
`build` string of `pyXY_<build_number>`, so pixi can resolve different
workspaces to different builds of the same source package.

### One pack folder, many python envs (when supported)

A single pack CAN serve multiple python envs in the same workspace --
**only when every entry in the pack ships wheels for every requested
python**. Path/git sources qualify automatically (pip wheel rebuilds
under each python). PyPI sources qualify only if the upstream index
publishes cp tags for every target.

When it does work, one source-package serves N envs:

```toml
# workspace pixi.toml
[workspace]
preview        = ["pixi-build"]
build-variants = { python = ["3.11", "3.12"] }

[feature.py311.dependencies]
python      = "==3.11"
my-pack     = { path = "./my-pack" }

[feature.py312.dependencies]
python      = "==3.12"
my-pack     = { path = "./my-pack" }

[environments]
py311 = ["py311"]
py312 = ["py312"]
```

```toml
# my-pack/pixi.toml
[package.build.config]
retread-python = ["3.11", "3.12"]
```

retread runs `pip wheel` per (entry × python), producing cp311 and
cp312 wheels side-by-side under `my-pack/wheels/<entry>/`. Conda emits
one artifact per python and pixi picks the right one per env.

If any one entry doesn't ship a wheel for one of the requested pythons
(e.g. NVIDIA's `isaacsim` PyPI index only ships cp311), the solve for
that variant bails. In that case use a separate pack per python (or
just one pack with a single `retread-python` value matching the env
that needs it).

## Relax policies

| Policy                       | `numpy==1.26.4` becomes  | `pyglet<2` becomes | Auto-widen unsat? |
|------------------------------|--------------------------|--------------------|--------------------|
| `none`                       | `numpy ==1.26.4`         | `pyglet <2`        | no                 |
| `patch`                      | `numpy >=1.26.4,<1.27`   | `pyglet <2`        | no                 |
| `minor`                      | `numpy >=1.26,<2`        | `pyglet <2`        | no                 |
| `major`                      | `numpy >=1`              | `pyglet <2`        | no                 |
| `strong-major`               | `numpy >=1`              | `pyglet`           | no                 |
| `patch-with-last-resort`     | `numpy >=1.26.4,<1.27`   | `pyglet <2`        | yes (probe + `*` widen) |
| `minor-with-last-resort` ★   | `numpy >=1.26,<2`        | `pyglet <2`        | yes (probe + `*` widen) |
| `major-with-last-resort`     | `numpy >=1`              | `pyglet <2`        | yes (probe + `*` widen) |

Non-`==` specifiers (ranges, `~=`, etc.) pass through unchanged under
every policy except `strong-major`, which additionally strips every
upper-bound clause from range specs (`<X`, `<=X`, the `<Y` half of
`>=A,<B`, the implicit upper of `~=X.Y`) so upstream caps stop
blocking the conda solve. Lower bounds (`>=`, `>`) stay so conda
doesn't pick something pre-historic.

`python` is exempt from every relax policy. Widening would either lose
ABI meaning (e.g. `python >=3`) or trip rattler-build's "missing range
specifier" check (`python 3`), so a `Requires-Dist: python` line is
passed through verbatim no matter what `retread-relax` is set to.

### `*-with-last-resort`: strict-by-default, auto-widens on unsat (v0.19.0+, recommended)

Each `*-with-last-resort` variant behaves IDENTICALLY to its base
(`patch` / `minor` / `major`) at translate time, plus an automated
per-dep cascade that fires ONLY for deps whose emitted conda spec
turns out unsatisfiable on the workspace's channels:

1. Probe conda channels with the base-relaxed spec
2. If satisfiable → emit normally (no behavior change)
3. If unsatisfiable → probe conda with `*` (any version, python-compatible)
4. If satisfied with `*` → inject `pkg = "*"` override → emit widened
5. If still unsatisfiable → log a warning (manual `retread-drop-deps` or
   a separate `retread-wheels` entry is needed; PyPI any-version
   fallback is v0.20 roadmap)

`minor-with-last-resort` is the recommended default — it preserves
the strictness of `minor` for the common case but eliminates the
hand-editing of `retread-overrides` for the pyglet-class pins where
conda-forge's candidates happen to be python-incompatible.

Every widening decision is recorded under `probe_decisions[].stage =
"last-resort-widen"` in `retread-probe-trace-<bundle>.json`, so the
auto-widenings stay auditable. Cost is zero for deps that satisfy
their strict spec.

### Channels: pytorch family

If your workspace has `torch` / `torchaudio` / `torchvision` /
`torchcodec` in any bundle, **add `https://prefix.dev/pytorch` to
your workspace `channels` list** ahead of `conda-forge`:

```toml
[workspace]
channels = ["https://prefix.dev/pytorch", "https://prefix.dev/conda-forge"]
```

conda-forge's torch builds are CPU-only or behind the official
release; the `pytorch` channel ships GPU-enabled cp311 builds.
Without this, retread's probe correctly identifies `torchaudio
>=2.7,<3` as unsatisfiable on conda-forge and falls back to PyPI
(bundling those huge wheels into your conda artifact), which is
slower and disk-heavier than letting the conda solver pull them
from the pytorch channel directly.

### Performance note (v0.19.0)

First solve of a non-trivial bundle (e.g. isaac-pack with isaacsim +
IsaacLab) takes minutes — retread downloads multi-GB wheels, builds
sdists when needed, runs the auto-data inject and probe cascade per
bundle. Subsequent solves hit the wheel cache under `<pack>/wheels/`
and the probe cache under `~/.cache/rattler/cache/retread-probes/`,
so re-solves are usually under a minute. `rattler-build`'s "preparing
packages" step is also slow on first build — it's downloading the
build env (rust toolchain, etc.) into a sandbox. To force a cold
rebuild, use `bash scripts/rebuild-local.sh`; it nukes the right
caches in the right order (see [Iteration: one script does the whole
dance](#iteration-one-script-does-the-whole-dance) below).

### Legacy: "strict by default, widen only when needed" (pre-v0.19.0)

Before `minor-with-last-resort` existed, the manual pattern was
`minor` + per-package `retread-overrides` added one at a time as the
solver complained:

```toml
[package.build.config]
retread-relax = "minor"

[package.build.config.retread-overrides]
# isaaclab pins pyglet<2 but conda-forge's pyglet<2 candidates are all
# python-3.5-only; widen this one entry.
pyglet = "*"
```

This still works and gives explicit control over which widenings
happen. `minor-with-last-resort` automates it for the common case
where the override would just be `pkg = "*"`. Both patterns coexist
fine — user overrides win over auto-widenings.

## Local development

Two ways to run an unreleased build against a real workspace, depending
on how much determinism you want.

### Option A: `file://` channel (true-local, recommended for iteration)

Build retread into a directory, then point the source package's
`[package.build].channels` at that directory. This is the only path that
fully bypasses prefix.dev for the backend itself; pixi installs retread
from the local `.conda` artifact exactly as it would from any channel.

```bash
# 1. Clone and build the .conda artifact.
git clone https://github.com/garylvov/pixi-build-retread.git
cd pixi-build-retread
pixi run -- rattler-build build --recipe recipe/recipe.yaml --output-dir ./local-channel
```

In your source package's `pixi.toml`, point `channels` at the `file://`
URL of `local-channel/`:

```toml
[package.build]
# `version = "*"` is fine here -- the file:// channel only has one
# build of retread (whatever you just produced), so there's nothing
# to disambiguate.
backend = { name = "pixi-build-retread", version = "*", channels = [
  "file:///abs/path/to/pixi-build-retread/local-channel",
  "https://prefix.dev/conda-forge",
] }
```

Repeat the `rattler-build` step whenever you edit retread; pixi picks
up the new `.conda` on the next `pixi install`. To switch to the
published version, swap the file URL for `https://prefix.dev/garylvov`.

#### Iteration: one script does the whole dance

`bash scripts/rebuild-local.sh` from the retread repo root nukes every
cache layer that gets in the way and rebuilds + verifies the new
artifact. Bump `Cargo.toml` AND `recipe/recipe.yaml` to the same
version first; the script aborts if they disagree.

Optional: `CONSUMER_PROJECT=/abs/path/to/workspace bash scripts/rebuild-local.sh`
also nukes that workspace's project-local pixi caches so the next
`pixi install` in the consumer picks up the new retread cleanly.

The rest of this section explains *what* the script touches and *why*
each cache layer matters. Useful when debugging a build that goes
sideways or when iterating without the script:

1. **`rattler-build` APPENDS to `local-channel/linux-64/repodata.json`**
   instead of regenerating it. Delete that one file before each rebuild
   so the channel stops advertising the old version:

   ```bash
   rm -f local-channel/linux-64/pixi-build-retread-*.conda \
         local-channel/linux-64/repodata.json
   # then re-run rattler-build
   ```

   **Do NOT delete `local-channel/noarch/repodata.json`.** retread
   only builds `linux-64`, but rattler-build still scans the noarch
   subdir during build-env resolution and fails with
   `could not find subdir 'noarch'` if the file is missing. The empty
   `{}`-shaped file checked into the repo must stay put. If you do
   nuke it by accident, recreate:

   ```bash
   echo '{"info":{"subdir":"noarch"},"packages":{},"packages.conda":{},"repodata_version":2}' \
     > local-channel/noarch/repodata.json
   ```

2. **pixi caches the retread executable** under
   `~/.cache/rattler/cache/backends-v0/pixi-build-retread-*/`. Even
   after the channel advertises a new version, a build-hash collision
   can make pixi reuse the old binary:

   ```bash
   rm -rf ~/.cache/rattler/cache/backends-v0/pixi-build-retread-*
   ```

3. **retread caches git clones** under
   `~/.cache/rattler/cache/retread-git-clones/`. The cache hits on
   `(url, rev)` so changes to the clone-dir layout (e.g. v0.13.3
   moved from `<slug>-<rev>/` to `<slug>/<sha12>/`) or partial-
   checkout failures leave stale trees that confuse the resolver.
   Nuke unconditionally when iterating on retread:

   ```bash
   rm -rf ~/.cache/rattler/cache/retread-git-clones
   ```

After both nukes + a fresh `rattler-build`, verify the channel actually
sees your new version before solving the consumer workspace:

```bash
grep -o 'pixi-build-retread-[0-9.]*' \
  local-channel/linux-64/repodata.json | sort -u
```

### Option B: `PIXI_BUILD_BACKEND_OVERRIDE` env var (faster loop, less deterministic)

Skip the `rattler-build` step and point pixi at the raw cargo-built
binary. Fast for inner-loop changes but pixi might cache earlier
metadata, leading to confusing "I changed the code but the behavior
didn't change" symptoms. Use Option A when in doubt.

```bash
cd pixi-build-retread
cargo build --release
export PIXI_BUILD_BACKEND_OVERRIDE=pixi-build-retread=$(pwd)/target/release/pixi-build-retread

cd ~/your-workspace
pixi install
```

To revert: `unset PIXI_BUILD_BACKEND_OVERRIDE`.

`rattler-build` must be on `PATH` whenever retread runs (it shells out
for the actual conda build). Easiest: open a pixi shell from this repo
(`pixi shell` in `pixi-build-retread/`) before running pixi in the
consumer workspace. the dev env declares `rattler-build` as a dep.

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
