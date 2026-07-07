# pixi-build-retread

[![linux-x86_64](https://github.com/garylvov/pixi-build-retread/actions/workflows/ci.yml/badge.svg)](https://github.com/garylvov/pixi-build-retread/actions/workflows/ci.yml)

**Retread relaxes strict PyPI dependency pins, prefers the Conda equivalent for any shared transitive, and iteratively reconciles conflicts at the fixed boundary between Pixi's Conda and uv's PyPI solver.** Ships as a statically linked musl binary: runs on any `x86_64` Linux distro, no glibc version requirement.

[Pixi](https://pixi.sh) resolves Conda first, then runs `uv` against PyPI using Conda's picks as hard pins. That handoff fails when an upstream wheel strictly pins a transitive that Conda resolved differently. Retread rewrites those exact pins into ranges (in the wheel `METADATA` and the emitted Conda run-deps), routes shared transitives to their Conda equivalents (via [parselmouth](https://github.com/prefix-dev/parselmouth) + a small fallback table) *before* `uv` runs, and iteratively re-solves until the boundary stabilizes — leaving PyPI-only deps for `uv`.

Motivated by [prefix-dev/pixi#5230](https://github.com/prefix-dev/pixi/issues/5230); automates [@diegoferigo-rai](https://github.com/diegoferigo-rai)'s hand-written Isaac Sim recipe. Worked example: [`examples/gigastrap/`](examples/gigastrap/) (Isaac Sim + IsaacLab + IsaacLab-Arena + pytorch3d + ROS 2 + GPU torch) — `pixi install -e gsi`.

## Requirements

- **Pixi >= 0.63.0** (pixi-build API v4; `pixi self-update` if older).
- **uv** — declared as a conda run-dep in the courier package, so a channel install provides it.
- The **`retread` installer** ships inside the courier package; no separate install, no glibc requirement.

## Quickstart

pixi-build's model is **workspace consumes source package** — two `pixi.toml`s. The second is a *new* manifest inside a subdirectory (`isaac-pack/` here), not part of your workspace:

```
your-project/
├── pixi.toml          # 1. workspace (your existing manifest)
└── isaac-pack/
    └── pixi.toml      # 2. source-package manifest (retread config)
```

**Workspace `pixi.toml`** — add the preview flag and one declaration that pulls the whole pack:

```toml
[workspace]
preview  = ["pixi-build"]
channels = ["https://prefix.dev/conda-forge"]

[dependencies]
isaac-pack = { path = "./isaac-pack" }   # one decl pulls the whole pack
# plus your usual deps: python, pytorch-gpu, ros-humble-*, ...
```

**Source-package `isaac-pack/pixi.toml`** — what to repack:

```toml
[package]
name    = "isaac-pack"
version = "5.1.0"

[package.build]
backend  = { name = "pixi-build-retread", version = ">=2.1.1" }
channels = ["https://prefix.dev/garylvov", "https://prefix.dev/conda-forge"]

[package.build.config]
retread-python = "3.11"          # target python(s); default 3.11
retread-bundle = "isaac-pack"    # collapse every entry below into ONE conda output

# Named git sources, referenced by `from = "<name>"` (keeps each rev in one place).
[package.build.config.retread-git-sources.isaaclab]
url = "https://github.com/isaac-sim/IsaacLab.git"
rev = "54cf64beb4eee99bc7b78e0353c8a4a8a13aa2c0"

# Wheels to repack. Each entry is one of five source forms:
#   version (+ index, extras)            -> PyPI Simple
#   url (+ sha256)                       -> direct download
#   path (+ extras)                      -> uv build local dir
#   git + rev (+ subdirectory, extras)   -> uv build git (inline)
#   from = "<name>" (+ subdirectory)     -> uv build git (named source above)
# `extras` resolve extras-gated Requires-Dist (incl. `pkg @ git+...`) as sub-wheels.
[package.build.config.retread-wheels]
isaacsim        = { version = "==5.1.0", index = "https://pypi.nvidia.com", extras = ["all", "extscache"] }
isaaclab        = { from = "isaaclab", subdirectory = "source/isaaclab" }
isaaclab-rl     = { from = "isaaclab", subdirectory = "source/isaaclab_rl", extras = ["all"] }
```

Then add **`<workspace>/.pixi/config.toml`** for the fast install path:

```toml
run-post-link-scripts = "insecure"
```

`git clone && pixi install` now resolves + links the courier package and its post-link runs `retread install` — installing exact locked wheel files with `uv --no-deps --offline`, hardlinking shipped/cache hits and direct-fetching only missing URL+hash wheels. No backend process, dependency resolution, or index metadata lookup happens on the consumer. (Why the toggle, and the safe alternative: see **Courier** below.)

## Courier — the fast path, and the safe alternative

retread can deliver a pack two ways. **Courier is the default and the fast path; `retread-courier = false` is the safe path.** Pick based on whether you can enable post-link scripts in the consuming workspace.

**Courier (default) — fast.** The pack is *one metadata-light conda package* that bakes in the `retread` installer, the built/shadow wheels, and a committed `retread-<bundle>.lock.json`. pixi links it like any conda package, then a post-link script runs `retread install`, which hands uv the exact wheel file list with `--no-deps --offline` and hardlinks the wheels into the env in **seconds**. Why prefer it:
- the consumer `pixi.toml` keeps **one clean line** (zero machine-written bytes),
- **no wheels in git** — the committed lock is kilobytes and the wheels ride inside the conda package,
- **nothing rebuilds on the consumer** — no backend process, no source build, no solve (a matching lock just replays).

The catch: pixi does not run post-link scripts by default, so the consumer must opt in with `run-post-link-scripts = "insecure"` — a real supply-chain tradeoff (it runs *every* package's post-link, not just retread's; see the toggle below). That's the price of the fast path.

**Safe mode (`retread-courier = false`) — no unsafe toggle.** The legacy conda-artifact path builds an ordinary conda package with the wheels pip-installed *into it at build time*; conda then places them at link time like any package — so **no post-link script and no `insecure` toggle are needed**. The cost is a heavier conda artifact and a slower build, and the wheels live inside that (larger) package rather than being fetched/hardlinked on demand.

Rule of thumb: use **courier** when you control the workspace and want fast, clean installs; use **safe mode** when enabling post-link scripts isn't acceptable in your environment.

<details>
<summary><b>The post-link toggle (fast path vs safe mode)</b></summary>

Courier installs via conda post-link scripts, which pixi does not run by default. Commit `<workspace>/.pixi/config.toml` containing exactly `run-post-link-scripts = "insecure"` (no section header).

> **Security.** This makes `pixi install` auto-run the post-link scripts of **every** conda package in the env, not just retread's — unsandboxed, with your privileges. A supply-chain-shaped risk; enable only for workspaces you trust.

If the toggle is enabled and `retread install` fails, the post-link fails the `pixi install` instead of leaving a half-installed env. If the toggle is missing, pixi skips the post-link and the wheels never install. The package ships an `activate.d` guard that verifies the marker and installed wheel metadata on every `pixi run` / `pixi shell`, so a missing or stale install is not silent:

```
retread: '<pack>' PyPI wheels are NOT installed.
  fast path: set run-post-link-scripts = "insecure" in <workspace>/.pixi/config.toml
  safe mode: set retread-courier = false
```

</details>

<details>
<summary><b>What's committed vs fetched</b></summary>

- **Git** (KB, no wheel bytes): the two `pixi.toml`s, `.pixi/config.toml`, and `retread-<bundle>.lock.json` next to the source pack.
- **Inside the conda package** (built by the backend, never in git): retread-built wheels (git/path sources), local-only wheels, sdist-built wheels, and relax-changed index wheels (shadow copies with rewritten METADATA, so the strict-pinned originals can't sneak back in).
- **Fetched at install** by `retread install`: unchanged index wheels from their recorded direct artifact URLs, verified against the lock's `sha256`, then installed from local files. If the sha-addressed cache already has the file, replay is fully offline.

Gitignore `wheels/` in the pack dir — it's multi-GB for NVIDIA packs and fully reproducible from the lock during pack build. At install time, shipped-only wheel classes cannot be recovered if the courier package payload under `$CONDA_PREFIX/share/retread/<pack>/wheels` is missing; reinstall or rebuild the courier package. Missing unchanged index wheels are recoverable from their locked URL+hash without consulting index metadata.

> The uv-installed PyPI dists live in `$CONDA_PREFIX` but are outside `pixi.lock`, so `pixi list` / `pixi install --frozen` won't show or restore them. After a pack change, `pixi lock` + re-install so pixi re-links the package and re-runs `retread install`.

</details>

<details>
<summary><b>Cold-solve replay & versioning (EMIT_EPOCH)</b></summary>

On `conda/outputs` the backend compares an `inputs_hash` (the `[retread-wheels]` entries, git revs, relax policy, python, workspace channels + per-env deps/system-requirements/pypi-options, and the per-dep config: overrides, name-map, shadow-libs, drop-deps, conda-deps, auto-bundle, build-number) against the committed lock. On a match it replays the lock and skips the cascade entirely (no probe-trace is written).

Replay is **not** keyed on the retread release version — the hash folds an internal `EMIT_EPOCH` that only bumps when a release could change emitted output, so routine retread upgrades reuse the lock instead of cold-solving. For strict reproducibility (re-solve on every retread version), set `retread-pin-version = true`. The courier package's build string is content-addressed on `inputs_hash`, so any content change makes pixi re-extract — no stale cache. (Upgrading across an `EMIT_EPOCH`/scheme change costs one cold solve per pack, then replay resumes.)

</details>

<details>
<summary><b>Editing a bundled package live</b></summary>

Don't list it as an editable `pypi-dependency` (uv reads its strict pins and they collide with conda's picks). Overlay after `pixi install`:

```bash
python3 -m pip install -e ./IsaacLab/source/isaaclab --no-deps --force-reinstall
```

`--no-deps` keeps retread's resolution; this swaps only the importable code.

</details>

## Escape hatches

Most packs need none — parselmouth + a built-in fallback table (`torch`→`pytorch`, `opencv-python`→`opencv`, …) handle common name skews and the cascade widens automatically.

<details>
<summary><b>When the auto path doesn't solve</b></summary>

```toml
[package.build.config.retread-overrides]
aiodns = "*"                          # replace the spec retread would emit

[package.build.config.retread-name-map]
some-pkg = "different-conda-name"     # force a PyPI->conda name parselmouth misses

[package.build.config]
retread-conda-deps = ["pytorch"]      # keep on conda side, don't bundle
retread-drop-deps  = ["weird-shim"]   # drop from run-deps entirely
```

</details>

<details>
<summary><b>Relax policies</b></summary>

| Policy | `numpy==1.26.4` → | `pyglet<2` → | Auto-widen on unsat? |
|---|---|---|---|
| `patch-then-minor-then-major-then-last-resort` ★ default | `>=1.26.4,<1.27` | `<2` | yes (progressive) |
| `none` / `patch` / `minor` / `major` | `==` / `>=1.26.4,<1.27` / `>=1.26,<2` / `>=1` | `<2` | no |
| `strong-major` | `>=1` | `pyglet` (cap stripped) | no |
| `*-with-last-resort` | as base | `<2` | yes (`*`) |

★ default (omit `retread-relax`). `python` is exempt (widening it loses ABI meaning).

The default cascade starts at the narrowest safe rewrite, runs a real `rattler_solve` over (workspace + emitted run-deps) on the workspace's channels, and on unsat widens only the retread-emitted blocker actually causing the handoff to fail. If the dominant constraint belongs to the *workspace*, it stops and surfaces a workspace-edit suggestion instead of thrashing. A dep with **zero conda candidates at any version** (`isaacsim-*`, `nvidia-*-cu1x`, …) is bundled from PyPI and dropped from the conda emission automatically. The index fallback chain is manifest-driven (each entry's `index`, then workspace/feature `[pypi-options]`, then public PyPI); it honors `[workspace].channel-priority` (default `strict`). Every step lands in `retread-probe-trace-<name>.json`; an UNSAT writes `RETREAD-SOLVE-FAILED-<name>.md` with the real conflict chain.

</details>

<details>
<summary><b>Lock stability (favor-lock) &amp; same-repo siblings</b></summary>

**favor-lock** (default-on since 2.10.0): on a re-resolve after a manifest change, retread *prefers each version already in the committed lock* when it still satisfies all constraints, and only deviates when a new dep forces it — minimal-change re-resolves, like pixi's own `favored` solver hint, over a fully-validated graph. Replay (unchanged inputs) and first resolves are unaffected. Disable with `RETREAD_NO_FAVOR_LOCK=1` to always pick highest-compatible.

**Siblings**: when several `retread-wheels` entries are built from one `retread-git-sources` repo in the same bundle, a dep naming a fellow entry (e.g. `isaaclab_visualizers` → `isaaclab`) is satisfied by that sibling wheel — never fetched from PyPI, never emitted as a run-dep. Missing subpackages a broken `setup.py` omits (`packages=[...]` instead of `find_packages()`) are recovered from the source tree automatically.

</details>

<details>
<summary><b>glibc / manylinux auto-relax</b></summary>

A binary index wheel can be published with a manylinux floor newer than the install host's glibc — e.g. Isaac Sim 6 ships only `manylinux_2_35`, but a RHEL 9 host is glibc `2.34`. uv derives manylinux compatibility from the **host** glibc and rejects the only available wheel:

```
× No solution found ... isaacsim[all]==6.0.0.1 has no wheels with a matching
  platform tag (e.g., `manylinux_2_34_x86_64`) ...
```

`retread install` recovers only when the workspace or pack declares the glibc floor it is willing to honor, for example `libc = "2.35"` in `[system-requirements]` or `platforms = [{ platform = "linux-64", glibc = "2.35" }]` under `[workspace]`. The retry target is exactly that declaration, not a host+1 heuristic. The declaration is load-bearing: uv still installs explicit wheel files with `--no-deps --offline`, but the retry adds `--python-platform` so uv's wheel tag check honors the declared floor. After uv installs, retread applies configured `[package.build.config.retread-shadow-libs]` replacements, prepends `$CONDA_PREFIX/lib` from the shipped activate.d hook, and runs a readelf GLIBC symbol audit. The audit records its result in the `.installed` marker; `retread verify --full` reruns it.

The shipped activation guard verifies and self-heals the uv-installed payload on activation. A failed heal writes `$CONDA_PREFIX/share/retread/<pack>.broken` and backs off for 300 seconds; `retread verify` treats that sentinel as failure. Pixi's experimental `use-environment-activation-cache` can cache activate.d output and skip per-activation verification, so workspaces that enable it should run `retread verify --lock ... --prefix ...` in CI or task preflight.

</details>

<details>
<summary><b>Cross-process solve dedup &amp; shared-git-source locking (3.0.0)</b></summary>

pixi solves separate top-level environments that reference the same source package (e.g. `isaaclab-gpu` and `isaaclab-gpu-latest`, or `gsi` and `gsi-ros2`) with **separate** retread backend processes. Two fixes landed for the resulting cross-process contention:

- **Duplicated solve-checks on cold**: each process previously started with an empty in-memory memo and reran the *entire* multi-env solve (every widening attempt, every env) from scratch even when a sibling process had just solved the identical `(params, workspace mtime)` key — burning minutes of repodata-parse + resolvo work per process and leaving other cores idle while it ran serially. `conda/outputs` results are now also memoized to disk under `<cache_dir>/retread-conda-outputs-cache/` (atomic write, best-effort — a read/write failure just falls back to a cold compute), so a process solving a sibling environment reuses the finished result instead of recomputing it.
- **Corrupted git checkouts under concurrent resolves**: `[retread-wheels]` entries that share one `(url, rev)` (e.g. IsaacLab's 14+ `from = "isaaclab"` entries differing only by `subdirectory`) clone into the same on-disk clone dir. Without coordination, concurrent resolves — across wheel entries or across the separate processes above — could race on that one shared working tree, aborting with `git checkout FETCH_HEAD failed ... untracked working tree files would be overwritten` or leaving `HEAD` parked on the wrong commit. Cloning/fetching/checking out now holds an exclusive `flock` on a lock file per clone dir (same mechanism `rattler_cache` uses for its own package cache) so only one resolver mutates a given clone dir at a time.

</details>

<details>
<summary><b>Multi-Python &amp; pytorch/CUDA</b></summary>

One artifact per platform; python-agnostic. retread builds wheels via `uv pip wheel --python <ver>`, fans `conda/outputs` over each requested python, and picks the matching wheel — so a multi-python pack works **only if every entry ships a wheel for every requested python**. Declare it (high→low precedence):

```toml
[workspace.build-variants]        # preferred -- forwarded to every backend
python = ["3.11", "3.12"]
# or per source package: [package.build.config] retread-python = "3.11"
```

**pytorch / CUDA:** for `torch`-family bundles the conda solver needs matching GPU builds — conda-forge ships `pytorch-gpu` + `torchvision`/`torchaudio` with `pytorch * cuda*` tags; pin `cuda-version` (e.g. `==12.8`) and scope niche channels per-feature (`[feature.gpu.channels]`).

</details>

## Local development

<details>
<summary><b>Build locally + point the backend at it</b></summary>

```bash
git clone https://github.com/garylvov/pixi-build-retread.git && cd pixi-build-retread
bash scripts/rebuild-local.sh        # nuke + build + verify into ./local-channel
```

Point the source pack's backend at it:

```toml
backend = { name = "pixi-build-retread", version = "*", channels = [
  "file:///abs/path/to/pixi-build-retread/local-channel",
  "https://prefix.dev/conda-forge",
] }
```

`rebuild-local.sh` requires matching versions (`Cargo.toml` + `recipe/recipe.yaml`; run `cargo check` to refresh `Cargo.lock`). `CONSUMER_PROJECT=/abs/path bash scripts/rebuild-local.sh` also clears that workspace's caches. The script exists because three caches otherwise serve a stale build (channel repodata, pixi's backend-executable cache, retread's git-clone cache); never delete `local-channel/noarch/repodata.json`. Faster, less-deterministic loop: `cargo build --release` + `export PIXI_BUILD_BACKEND_OVERRIDE=pixi-build-retread=$(pwd)/target/release/pixi-build-retread` (`rattler-build` must be on `PATH`).

**Contributing:** `pre-commit install` (fmt + clippy + fast tests); `cargo test -- --include-ignored` for the heavy live tests. CI runs fmt, clippy, the fast suite, the static-musl build, and an `EMIT_EPOCH` guard.

</details>

## Acknowledgements

The [prefix.dev](https://prefix.dev) team for Pixi; [@ruben-arts](https://github.com/ruben-arts) and [@tdejager](https://github.com/tdejager) for the [pixi#5230](https://github.com/prefix-dev/pixi/issues/5230) discussion; [@diegoferigo-rai](https://github.com/diegoferigo-rai) for the static `recipe.yaml` this automates.
