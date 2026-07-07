# pixi-build-retread

[![linux-x86_64](https://github.com/garylvov/pixi-build-retread/actions/workflows/ci.yml/badge.svg)](https://github.com/garylvov/pixi-build-retread/actions/workflows/ci.yml)

A [pixi](https://pixi.sh) build backend that bundles a PyPI wheel closure (Isaac Sim,
IsaacLab, and friends) into a single conda package, crossing the conda&harr;uv boundary in
both directions: strict wheel pins are relaxed into ranges so conda's picks satisfy them,
shared transitives are routed to their conda equivalents, and PyPI-only deps ride inside
the bundle. One `pixi.toml` line on the consumer pulls the whole pack.

Ships as a statically linked musl binary (`x86_64` Linux, no host glibc requirement).
Motivated by [prefix-dev/pixi#5230](https://github.com/prefix-dev/pixi/issues/5230).

## v4 architecture

Four pieces, two of which are allowed to resolve anything:

- **Pack backend** (resolves) — speaks the pixi-build JSON-RPC protocol. Computes the wheel
  closure for each `[retread-wheels]` entry, rewrites strict pins, and emits one
  metadata-light conda package plus a committed `retread-<bundle>.lock.json` recording the
  complete wheel set: name, version, URL, sha256.
- **Courier** (never resolves) — the conda package's post-link script runs
  `retread install`, which **replays the lock**: uv is invoked with `--no-deps` against
  explicit wheel files/URLs, never against index metadata. Shipped and cached wheels are
  hardlinked; missing wheel *bytes* are fetched from their locked URL and hash-verified.
  Reactivation and self-heal never re-resolve — resolution happens only at pack build and
  in `retread solve`.
- **activate.d guard** — every `pixi run` / `pixi shell` runs `retread verify`
  (marker + installed dist metadata, no network). A broken payload self-heals via the same
  no-resolve replay; a failed heal writes a `.broken` sentinel and backs off.
- **glibc handling** — a wheel published with a manylinux floor above the host glibc
  (Isaac Sim: `manylinux_2_35` vs RHEL 9's 2.34) is installed by relaxing uv's platform
  check to the floor **you declared** in `[system-requirements] libc = "..."` — never a
  host+1 guess; an undeclared floor is a hard error with remediation. After install,
  `retread-shadow-libs` replacements are applied (vendored lib &rarr; symlink to
  `$PREFIX/lib/<SONAME>`) and a readelf GLIBC symbol audit verifies every vendored lib
  needing more than host glibc has a shadowing provider.

## Quickstart

Two manifests: your workspace, and a source-package manifest in a subdirectory.

```toml
# pixi.toml (workspace)
[workspace]
preview  = ["pixi-build"]
channels = ["https://prefix.dev/conda-forge"]

[system-requirements]
libc = "2.35"            # REQUIRED for packs with a manylinux floor (Isaac Sim).
                         # Works on glibc-2.34 hosts: this is the floor retread
                         # honors for wheel tags, backed by the shadow-lib audit.

[dependencies]
isaac-pack = { path = "./isaac-pack" }
```

```toml
# isaac-pack/pixi.toml (source package)
[package]
name    = "isaac-pack"
version = "5.1.0"

[package.build]
backend  = { name = "pixi-build-retread", version = ">=4.0.0" }
channels = ["https://prefix.dev/garylvov", "https://prefix.dev/conda-forge"]

[package.build.config]
retread-python = "3.11"
retread-bundle = "isaac-pack"

[package.build.config.retread-wheels]
isaacsim = { version = "==5.1.0", index = "https://pypi.nvidia.com", extras = ["all", "extscache"] }
# also: url+sha256, path, git+rev(+subdirectory), from = "<retread-git-sources name>"

[package.build.config.retread-shadow-libs]
"isaacsim/kit/kernel/plugins/libpython3.12.so.1.0" = "conda-lib"
```

```toml
# .pixi/config.toml (courier post-link opt-in; safe alternative: retread-courier = false)
run-post-link-scripts = "insecure"
```

Then `pixi install`. (The previous README's quickstart omitted the `libc` declaration and
failed on 2.34 hosts — that was [issue #9](https://github.com/garylvov/pixi-build-retread/issues/9).)

## Subcommands

The one binary is both the build backend (no subcommand; JSON-RPC on stdin/stdout) and the
runtime CLI:

```bash
# Install a bundle's wheels into a prefix by replaying the lock (post-link calls this).
retread install --lock share/retread/isaac-pack/retread-isaac-pack.lock.json --prefix "$CONDA_PREFIX"

# Cheap activation guard: marker + dist metadata. --full reruns the GLIBC symbol audit.
retread verify --lock <lock.json> [--prefix <p>] [--full]

# Error-driven repair loop: drives `pixi install`, parses the solver conflict, and
# escalates per package: widen conda pin -> pin conda -> pin/migrate pypi -> pypi override.
# Every injected pin gets a `# retread:pin` sentinel comment; every action is ledgered on
# disk; a post-solve import smoke test gates success. ABI anchors (python/libc/cuda) are
# never auto-widened.
retread solve --manifest pixi.toml -e isaaclab-gpu-latest [--max-iters N] [--dry-run]
retread solve --clean-pins        # strip all sentinel pins, then re-solve from scratch

# Slow-FS (NFS/SLURM) escape hatch: run pixi with env + caches on job-local tmp.
eval "$(pixi-build-retread fast --print-env)"     # export the fast-tmp env into this shell
pixi-build-retread fast -- pixi install           # or wrap a single command
pixi-build-retread fast --persist isaaclab-gpu-latest   # snapshot job-local env back to NFS
```

## Config

```toml
[package.build.config]
retread-resolver = "legacy"   # default: in-backend closure engine
# retread-resolver = "uv"     # experimental: uv-project-based closure computation

[tool.retread.fast-tmp]       # workspace pixi.toml; all optional
mode             = "auto"     # auto (engage when FS probes slow) | on | off
tmp-root         = "/tmp"     # node-local root for envs + caches
budget-bytes     = 50_000_000_000
blob-caches      = "shared"   # keep pixi/uv/rattler blob caches on shared FS
shared-cache-dir = ".pixi-shared-cache"
persist-dir      = ".pixi-envs-persist"   # NFS-side env snapshots
copy-workers     = 16
```

Escape hatches unchanged from 3.x: `retread-overrides`, `retread-name-map`,
`retread-conda-deps`, `retread-drop-deps`, `retread-relax` (policy table), `retread-git-sources`,
`retread-auto-bundle`, `retread-courier = false` (legacy artifact path, no post-link toggle needed).
Backend logs: `PIXI_BUILD_RETREAD_LOG` (not `RUST_LOG`); per-bundle audit JSON lands next to
the pack's `pixi.toml`.

## Consumer workflow on a SLURM/NFS cluster

Installed pixi envs on NFS are slow to link and slow to import from. The `fast` subcommand
treats the env as **disposable and job-local**:

```bash
eval "$(pixi-build-retread fast --print-env)"   # detached envs + caches -> node-local tmp
pixi install                                     # then use pixi normally
pixi run python -c "import isaaclab"
pixi-build-retread fast --persist isaaclab-gpu-latest   # once, from a warm env
```

Materialization on a fresh node is hash-gated: if the persist snapshot's stamped lock hash
matches the workspace lock, the env is parallel-copied from NFS in seconds; otherwise it is
rebuilt with a frozen install (no re-resolve). Blob caches stay on the shared FS so rebuilds
are download-free. Requirements learned the hard way: **`git-lfs` and `rattler-build` must
be on `PATH`** on the build host, or pack builds fail with unhelpful errors mid-closure.

## v4.0.0 breaking changes vs 3.x

- **Lock schema bump** (`SCHEMA = 12`): existing packs cold-solve exactly once to rewrite
  their `retread-<bundle>.lock.json`, then replay resumes.
- **No-resolve replay is mandatory**: courier install, reactivation, and self-heal never
  invoke a resolver. A lock too old to replay is an error, not a silent fallback to
  resolver-backed uv.
- **Undeclared glibc is now an error**: packs whose wheels carry a manylinux floor above
  host glibc previously guessed; now they fail with the exact `[system-requirements]
  libc = "X.Y"` line to add.

## Local development

```bash
git clone https://github.com/garylvov/pixi-build-retread.git && cd pixi-build-retread
bash scripts/rebuild-local.sh   # nuke + build + verify into ./local-channel
```

Point a pack's `backend.channels` at `file:///abs/path/to/local-channel`. Faster loop:
`cargo build --release` + `PIXI_BUILD_BACKEND_OVERRIDE=pixi-build-retread=$(pwd)/target/release/pixi-build-retread`.
`pre-commit install` for fmt + clippy + fast tests; `cargo test -- --include-ignored` for
the heavy live suite.

## Acknowledgements

The [prefix.dev](https://prefix.dev) team for pixi; [@ruben-arts](https://github.com/ruben-arts)
and [@tdejager](https://github.com/tdejager) for the [pixi#5230](https://github.com/prefix-dev/pixi/issues/5230)
discussion; [@diegoferigo-rai](https://github.com/diegoferigo-rai) for the hand-written
Isaac Sim recipe this automates.
