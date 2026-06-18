# Handoff Prompt — pixi-build-retread + gigastrap

You're picking up an ongoing collaboration with Gary Lvov on a Rust
project called **pixi-build-retread**, a pixi-build backend that repacks
PyPI wheels as conda packages with relaxed dependency pins. The
motivation is [prefix-dev/pixi#5230](https://github.com/prefix-dev/pixi/issues/5230)
— mixing complex PyPI packages (notably NVIDIA Isaac Sim) with conda
deps in pixi causes solve failures because pixi's dual-solver
forwards conda's chosen versions to uv as hard pins, which collide
with upstream's strict `Requires-Dist` pins.

## User context

- Gary is technically sophisticated, runs a real robotics monorepo
  called **gigastrap** at `/home/garylvov/projects/gigastrap/`.
- Hates emojis, hates excess prose. Wants concise, direct answers.
- Uses pixi for everything; lives in pixi env at
  `/home/garylvov/projects/pixi/` (the pixi source repo, with a `.pixi/`
  env that has cargo, rust, rattler-build, etc.).
- For ANY `cargo` / `rattler-build` invocation, set
  `PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH"`.
- Iterative loop: he runs `pixi s -e gsn` (or `gsi`) in gigastrap, sees
  what breaks, asks "why didn't tests catch this", we add tests + fix.
- He values architectural pushback. When given "just add an override"
  answers he asks for solution-architect input and a redesign.
- He pushes back on hardcoded lists — prefers parselmouth lookup over
  a baked-in numpy/scipy/pytorch table.
- When stuck, he asks for "more options" or dispatches the-grizzly
  rather than committing to the first suggestion.
- Multiple times he's said "we should be running tests" when a bug
  shipped — he's right; this project has a recurring pattern of tests
  that check retread's emission but never that downstream (pixi /
  rattler-build / conda solver) accepts it. When you add a new field
  or behavior, the test should round-trip through whatever consumes it.

## The project

### Repo + publishing

- Source: `/home/garylvov/projects/pixi-build-retread/`
- GitHub: <https://github.com/garylvov/pixi-build-retread> (push as
  `garylvov` via gh CLI — already authed)
- Published channel: <https://prefix.dev/garylvov> — versions 0.1.0
  through **0.22.0** published. **0.37.1** built locally, awaiting
  prefix.dev upload (see "Standard commands" below for the upload
  one-liner). Local channel at `./local-channel/linux-64/` always
  holds the active iteration version; gigastrap's isaac-pack/pixi.toml
  `[package.build].channels` lists `file://` local-channel BEFORE
  `https://prefix.dev/garylvov` so local-iteration always wins when
  a same-version artifact exists in both. v0.9.9+ artifacts are
  single-arch (linux-64, single hash), not per-python variants —
  retread shells out to `uv build --python <ver>` so the binary is
  python-agnostic.
- License BSD-3-Clause. Author email `gary.lvov@gmail.com`.

### Architecture

retread is a JSON-RPC build backend (line-delimited JSON-RPC 2.0 over
stdio). Pixi spawns it as a subprocess and sends:

1. `negotiateCapabilities` — we say `providesCondaOutputs: true`,
   `providesCondaBuildV1: true`.
2. `initialize` — we read the `[package.build.config]` table as our
   `RetreadConfig`, parse `[retread-wheels]` entries, validate.
3. `conda/outputs` — for each `[retread-wheels]` entry group: resolve
   the primary wheel (URL / PyPI Simple / local path / git / named-git),
   apply D (wheel METADATA surgery per relax policy), parse, optionally
   auto-bundle transitive PyPI-only deps. Return CondaOutput per group.
4. `conda/build_v1` — pick the requested output, generate
   `recipe.yaml` (and `retread-audit.json` alongside it), shell out to
   `rattler-build build`.

### Bundle pattern (THE central concept)

Each `[retread-wheels]` entry produces ONE conda package whose
`source:` list contains N wheels (primary + extras + auto-bundled
transitives). All N wheels are `pip install --no-deps` into the same
conda prefix. This is the pattern from
[pixi#5230 comment 24](https://github.com/prefix-dev/pixi/issues/5230#issuecomment-comment-24).

**Group multiple entries into one output via the `bundle` field**
(added v0.9.9): every entry that sets `bundle = "<name>"` collapses
into one conda output named `<name>`. The workspace then declares a
single `<name> = { path = "./isaac-pack" }` and gets every wheel. Used
by gigastrap's isaac-pack/pixi.toml — all 8 entries set
`bundle = "isaac-pack"`. Without `bundle`, each entry produces its own
conda output and the workspace would need one declaration per entry.

Why bundle: avoids conda's per-package resolution forwarding versions
to uv as hard pins. The bundled wheels are pip-installed into the
conda env directly, side-stepping the conda→uv pin forwarding for
those packages.

### Five source forms for `[retread-wheels]` entries

| Form | Schema | Resolution |
|---|---|---|
| URL | `{ url, sha256? }` | direct download |
| PyPI spec | `{ version, index?, extras? }` | PEP 503 simple-index resolve |
| Local path | `{ path }` | `pip wheel <path> --no-deps` |
| Git inline | `{ git, rev, subdirectory? }` | clone, `pip wheel --no-deps` |
| Named git | `{ from, subdirectory? }` | reference `[retread-git-sources.<name>]` |

PyPI spec accepts ranges (`>=5`, `~=5.1`), not just exact pins — range
resolution picks the highest matching version that has a target-
compatible wheel. Fixed in v0.9.2.

The named-git form lets multiple sub-packages share one `(url, rev)`
declaration. gigastrap uses this for 6 IsaacLab packages from the
same upstream commit.

### D = wheel METADATA surgery

The defining feature of v0.9.x. After downloading or building each
wheel, retread rewrites its `dist-info/METADATA` to apply the relax
policy to `Requires-Dist:` lines. RECORD file is updated in lock-step
so pip's hash check still passes. **The recipe `source:` URL is set
to the post-D wheel** (`*.relaxed.whl`); previously a bug had it
pointing at the pre-D wheel so rattler-build sourced the un-rewritten
file and uv re-saw the strict pins from site-packages. Fixed in v0.9.3.

### Relax policies (RelaxPolicy enum, src/config.rs)

| Policy | `numpy==1.26.4` becomes | `pyglet<2` becomes |
|---|---|---|
| `none` | `numpy ==1.26.4` | `pyglet <2` |
| `patch` | `numpy >=1.26.4,<1.27` | `pyglet <2` (passthrough) |
| `minor` (default) | `numpy >=1.26,<2` | `pyglet <2` (passthrough) |
| `major` | `numpy >=1` | `pyglet <2` (passthrough) |
| `strong-major` (v0.10.0+) | `numpy >=1` | `pyglet` (upper stripped) |
| `conda-aware` (v0.10.0+) | `numpy >=1` | `pyglet` (probe layer is TODO; currently identical to strong-major) |

`strong-major` was added when the conda solver kept rejecting
`pyglet<2` (only conda-forge `pyglet<2` candidates were python-3.5-
only). Drops every upper-bound clause: `<X`, `<=X`, the `<Y` half of
`>=A,<B`, the implicit upper of `~=X.Y`. Lower bounds stay.

`conda-aware` is reserved for "smart-B" — probe the workspace's conda
channels per-emitted-spec, and only strip the upper bound for deps
that actually have zero candidates under the workspace's python. The
intent is to widen only what truly needs widening, leaving the rest
at major's behavior. **Not yet implemented**; currently behaves
identically to strong-major at translate time. To implement: fetch
the channel's repodata.json (zstd-compressed, cache in
`~/.cache/rattler/retread-probes/`), for each emitted spec containing
`<`, `<=`, or `~=`, check whether any candidate satisfies the spec AND
has a python_abi matching the workspace's python; if zero, strip
uppers and emit the relaxed form; if any candidate, leave alone.
Record decisions in the audit's `probe_results[]` field.

### Auto-bundle policy

**Prefer-conda by default (v0.9.6+).** For each PyPI transitive
discovered while expanding a bundled wheel's Requires-Dist, retread
checks the effective `name_map` (parselmouth + FALLBACK_PYPI_TO_CONDA
in `src/handler.rs:60` + user `retread-name-map`). If the dep has an
unambiguous conda equivalent, retread skips bundling — the dep flows
through to emission as a conda run-dep via translate. Only deps with
no conda match get vendored.

Why: bundling everything double-installs ABI-sensitive packages
(numpy, torch, scipy) on top of the conda-installed copy. With
gigastrap's editable PyPI workspace entries commented out (see below),
there's no conflict source on the PyPI side, so prefer-conda is safe
and saves disk + parallelizes downloads.

User opt-outs:
- `retread-conda-deps = [...]` — force this dep to conda even if
  parselmouth missed it
- `retread-drop-deps = [...]` — drop from emission entirely (no
  conda dep, no bundle)
- `retread-overrides = { name = "*" }` — substitute the conda spec
- `retread-name-map = { pypi: conda }` — name skew (most cases
  auto-handled by parselmouth + FALLBACK)

### Cross-output run-deps

When N retread-wheels entries each produce their own conda output
(no `bundle` field), each output's run_dependencies pin every
sibling at exact version. Safety net: if user declares 7 of the 8 in
the workspace pixi.toml, the conda solver fails loudly with
`no candidates were found for <missing-name>` instead of silently
installing partial. **Does NOT pull missing siblings transitively**
because pixi-build only builds outputs the workspace declared —
undeclared outputs are described to pixi but never staged in the
solver's channel.

### Python version: variant build (v0.9.9+)

retread shells out to `pip wheel` for git/path sources. The pip
inherits retread's own env's python; the wheel's cp tag matches that
python. If retread runs under py3.14 but the workspace uses py3.11,
wheels are cp314, conda metadata pins `python_abi 3.14.*`, and the
workspace's solver rejects them.

Fix: retread's `recipe/recipe.yaml` declares
`python ${{ python }}.*` as a variant. `recipe/variants.yaml` lists
the pythons to build for (currently `["3.11", "3.12"]`). Build with
`--variant-config recipe/variants.yaml` and one conda artifact is
produced per python variant.

**Propagation**: gigastrap's workspace pixi.toml has
`[workspace] build-variants = { python = ["3.11"] }` (inline form to
avoid a `[workspace.build-variants]` sub-table header that would
swallow subsequent keys like `conda-pypi-map`). pixi-build forwards
this to source-package backend resolution, so retread's py311 variant
is auto-picked. To migrate to 3.12, edit the one line in gigastrap's
pixi.toml — retread's py312 variant is already built and waiting in
the local-channel.

**To add a new python**: edit `recipe/variants.yaml`, rebuild with
`--variant-config`, delete orphan old variants from the channel.

### Audit artifact (v0.10.0+)

`retread-audit.json` written next to every generated `recipe.yaml`
(at `gigastrap/.pixi/bld/isaac-pack-*/recipe-isaac-pack/`). Fields:

- `wheels[]` — for each bundled wheel: `name`, `version`,
  `requires_dist` (pre-D Requires-Dist lines as they appear in
  upstream METADATA)
- `emitted_run_deps[]` — `name` + `spec` for each conda run-dep
  retread emitted (post-D, post-translate)
- `pixi_toml_blocks.dependencies` — copy-paste-ready
  `[dependencies]` body
- `pixi_toml_blocks.pypi_options_dependency_overrides` — copy-paste-
  ready `[pypi-options.dependency-overrides]` body; one line per
  wheel pinned to `==<exact-version>` for mirroring the bundle onto
  the PyPI side. PEP 440 local-version identifiers (the `+5043d15…`
  in pytorch3d) survive verbatim.

Purely informational; nothing else reads it back. Removing the file
or skipping the write has no effect on the build.

## File map

| File | What lives here |
|---|---|
| `src/audit.rs` | Audit data structures + JSON serialization + TOML block formatters. v0.14.1+ adds `ProbeDecision` (stage / pypi_name / conda_name / spec / target_python / channels_consulted / satisfiable / matching_candidates / routing_decision). Written to `<pack>/retread-audit-<name>.json` AND streamed live to `<pack>/retread-probe-trace-<name>.json` during conda/outputs so the trace survives failed conda solves. |
| `src/config.rs` | TOML schema. v0.19.0+ adds `PatchWithLastResort`/`MinorWithLastResort`/`MajorWithLastResort` policy variants with `has_last_resort()` helper. v0.12.0 lifted `extras` restriction to allow them on path/git/named-git forms (URL still rejected). |
| `src/handler.rs` | Four JSON-RPC methods + the whole materialize pipeline. Phases per primary wheel: (1) fetch/build, (1.5) source-extras inject, (1.6) checkout-root data inject, (2) relax. Bundle merging in `resolve_all` runs `auto_bundle_transitives` then `pre_emit_widen_pass` (v0.30.0+ — always runs; tiered cascade or simple last-resort depending on policy), then `produce_output` (sync), then `post_emit_widen_pass` (v0.23.0+ — also always runs; mutation gated by policy). ~2800 lines. |
| `src/wheel_rewrite.rs` | D: METADATA rewriting + RECORD lock-step update. PEP 508 relax math. `strip_upper_bounds_pep508` for strong-major + conda-aware. The `*WithLastResort` variants delegate to their base policy here (last-resort is in the probe layer, not the rewrite). |
| `src/source_build.rs` | `uv build --wheel` for path + git. v0.13.3 hashed the git-clone cache layout (`<slug>/<sha12>/`) to dodge ENAMETOOLONG. v0.18.0 added `build_wheel_from_sdist_url` for BFS fallback when PyPI Simple has no wheel for the spec (e.g. `gym` which is sdist-only). All subprocess calls use `run_capturing` so stdout (JSON-RPC channel) is never poisoned. v0.13.4 propagates the failing tool's stderr in error messages. |
| `src/pypi.rs` | PEP 503 simple-index resolver. PEP 425 wheel tag matching incl. v0.13.10 abi3 support (cp36-abi3 matches any py >= 3.6). v0.18.0 adds `resolve_sdist` (used by the BFS sdist fallback). |
| `src/recipe.rs` | recipe.yaml generator for rattler-build. `dynamic_linking.binary_relocation: false` for platform-specific bundles. |
| `src/relax.rs` | PEP 508 → conda match-spec translation. `strip_upper_bounds` for strong-major/conda-aware. Marker env builder (`marker_env_for(subdir, py)` + `default_marker_env(py)`). |
| `src/wheel.rs` | Wheel download (reqwest) + METADATA parse. Stable. |
| `src/rpc.rs` | Hand-rolled line-delimited JSON-RPC 2.0 dispatcher. |
| `src/wheel_inject.rs` | v0.9-era source-extras inject (files pip wheel forgot to ship for the entry's own subdirectory). Pre-pip wheel inject phase 1.5. |
| `src/wheel_inject_data.rs` | v0.12.0 auto-data-files inject. Walks the upstream repo's CHECKOUT ROOT (parent of subdir) honoring `.gitignore` + `ALWAYS_SKIP` floor. Emits each file as `<dist>-<ver>.data/data/lib/<rel>` so it lands at `$PREFIX/lib/<rel>` post-install. v0.21.0 refined the `skip_subdirs` filter: under skipped subdirs, only Python source/cache/build-meta is dropped; non-Python files (Kit `extension.toml`, data assets, configs) still ship so Omniverse Kit's `${app}/../source/<ext>` extension scan succeeds. |
| `src/probe.rs` | v0.13.11 initial probe (was prefix.dev `/api/v1/.../variants?limit=500` -- returned 404 silently on every call). v0.22.0 rewrote to use conda `repodata.json[.zst]` directly. Fetches `<channel>/<subdir>/repodata.json.zst`, decompresses with `zstd` crate, parses into `HashMap<name, Vec<VariantInfo>>` indexed by name. Parallelizes per (channel × subdir) via `FuturesUnordered`. In-memory cache (Arc) survives within a process; disk cache (30min TTL) survives across invocations. Filters by spec AND python_constraint (extracted from `depends` array's `python` / `python_abi` entries). v0.32.0 adds `fetch_latest_build_depends` for workspace-transitive constraint extraction. |
| `src/workspace.rs` | v0.32.0+: parses the consumer workspace's pixi.toml's `[workspace]`, `[dependencies]`, `[pypi-dependencies]`, `[environments]`, `[feature.X.*]` tables. Exposes `WorkspaceManifest::discover_outputs_for_source` (autodiscovery of which workspace path-deps point at this source package + which envs activate them) and `extract_transitive_constraints` (reads each env's conda deps' `depends` arrays to find what they require of OTHER deps). Output autodiscovery replaces the briefly-considered `retread-per-env` config: retread emits one output per workspace-declared name. |
| `src/solve_check.rs` | v0.33.0+: pre-emission solve check via `rattler_solve` (resolvo backend). `run_solve_check(channels, specs, target_python, target_subdir)` loads cached repodata, builds `RepoDataRecord`s, runs a full conda solve, returns `SolveOutcome { satisfiable, unsat_explanations, ... }`. Called from conda_outputs after post_emit_widen_pass; outcome persists to `BundleAudit.solve_diagnostics`. Surfaces cross-package conflicts the per-dep probe can't see (cuda-bindings 13 vs cuda-toolkit 12.8). |
| `recipe/recipe.yaml` | Conda recipe for retread itself. Runtime deps: `rattler-build`, `python ${{ python }}.*`, `pip`, `setuptools`, `git`, `pixi-build-api-version >=4,<5`. |
| `recipe/variants.yaml` | Variant list. Currently `python: ["3.11", "3.12"]`. Rebuild produces one artifact per. |
| `scripts/rebuild-local.sh` | v0.20.0+ one-shot nuke-rebuild-verify. Aborts on Cargo.toml/recipe.yaml version mismatch. Nukes 5 cache layers (artifact, channel-repodata, backend exe, retread git-clones, retread probe + repodata). With `CONSUMER_PROJECT=/abs/path` also nukes that workspace's `.pixi/{meta-v0,bld}/isaac*`, `~/.cache/rattler/cache/bld/{metadata,source_metadata}-v0/isaac*`, and the per-pack `retread-{audit,probe-trace}-*.json` + `wheels/` dir. Verifies the new version lands in `repodata.json` before exit. |
| `tests/wheel_fetch_live.rs` | Live tomli + isaacsim wheel fetch. |
| `tests/pypi_resolve_live.rs` | PyPI + NVIDIA + py.mujoco.org Simple resolver. |
| `tests/isaacsim_relax.rs` | Snapshot tests using `tests/fixtures/isaacsim_{kernel,core}.METADATA.txt`. |
| `tests/source_build_live.rs` | path-source `pip wheel` builds. Uses `tests/fixtures/sample_with_buildtime_dep/`. |
| `tests/jsonrpc_protocol.rs` | **The one that catches stdout corruption + per-entry error context.** Spawns the release binary, sends JSON-RPC, asserts every stdout line parses as JSON. `broken_entry_surfaces_with_entry_name` test pins fail-fast on broken entries. |
| `tests/e2e_ros_isaacsim.rs` | Heavy: drives `pixi lock` against a stripped ros2+isaacsim workspace. |
| `tests/fixtures/fetch_metadata.py` | Regenerates METADATA snapshots from real wheels. |

Total: **172 lib tests** as of v0.37.0 (152 on v0.36.4, 146 on
v0.36.0, 125 on v0.35.3; was ~30 in the original handoff). v0.37.0
adds 20 across solve_check (build_virtual_packages mapping x6),
workspace (system-requirements parsing + effective rollup +
build-string preservation x6), and handler (pythons_for bare-major
rejection x3, join_transitive clause-dedup x3, ABI invariant
bare-major coverage x2). v0.36.4 added 6: widening_level ordering +
exact-pin + upper-bound edge cases, merge_looser_override (keeps
widest, never narrows, inserts when missing), and the produce_output
rebuild round-trip. Adds in v0.36.0: 7 `is_abi_anchor` predicate
tests, 6 `classify_chains` per-verdict tests (incl. the
`classify_chains_mixed_gsi_round4_scenario` fixture pinning the gsi
failure that motivated v0.36.0), 7 `check_output_abi_invariants`
tests, and 1 simulated refinement-loop test asserting python is never
widened. Also: bundle field parsing, cross-output deps, prefer-conda
filter, percent-decoding of URLs, strong-major stripping (with extras
+ markers preserved), audit TOML block validity, range version
specifiers, PEP 440 local-version identifiers, bare-major python
emitting glob (not `==3` strict).

## Critical "don't break this" invariants

1. **Stdout discipline**: retread's stdout is the JSON-RPC channel.
   ANY subprocess invocation must capture stdout. The
   `jsonrpc_protocol.rs` test catches regressions.

2. **Per-entry error context**: in `resolve_all`, wrap each
   `resolve_bundle` call with `.with_context(...)` naming the entry
   AND its bundle group. Otherwise pixi just says "the package is
   not provided" with no hint of WHICH entry failed. Also,
   `conda_outputs` MUST propagate errors (it used to swallow them
   per-variant and continue, leaving empty outputs).

3. **RECORD lock-step**: when D rewrites METADATA, RECORD's entry
   for METADATA must be updated with the new sha256 + size.

4. **Recipe source URL = post-D wheel**: never the pre-D upstream
   URL. The `materialize_and_rewrite` function in handler.rs returns
   `file://` of the rewritten wheel. There's a regression test
   (`d_rewrites_metadata_on_the_wheel_the_recipe_will_source`) that
   pins this.

5. **Backend cache invalidation**: pixi caches resolved backend
   binaries at `~/.cache/rattler/cache/backends-v0/pixi-build-retread-*/`,
   AND `rattler-build` APPENDS to `local-channel/linux-64/repodata.json`
   instead of regenerating it. Combined, these two caches mean that
   even after you bump the version, rebuild, and run `pixi clean`,
   pixi can STILL pick up the previous binary because (a) the channel's
   repodata still lists the old version, OR (b) pixi reuses its cached
   executable. Symptom: an error message you literally just deleted
   from `src/handler.rs` re-fires on the next solve. To break out,
   run the full nuke-+-rebuild block from "Standard commands" below
   AS A UNIT -- nuking only the backend cache, or only the artifact,
   or only the project's `.pixi/`, leaves a stale cache somewhere
   upstream. ALSO bump the retread version each rebuild — pixi
   caches by version, so without a bump it may keep the old artifact.

6. **`deny_unknown_fields` strictness on WheelEntry**: adding a new
   optional field requires a serde-deserialization test
   (`parses_<field>_on_entry` in src/config.rs). Without it, stale
   binaries reject user pixi.toml with "unknown field" during the
   upgrade window. The `parses_bundle_field_on_entry` and
   `rejects_unknown_field_on_entry` tests demonstrate the pattern.

7. **TOML scoping in workspace pixi.toml**: when adding
   `build-variants` to `[workspace]`, use the inline form
   (`build-variants = { python = [...] }`). A
   `[workspace.build-variants]` sub-table header would swallow every
   subsequent key-value pair (like `conda-pypi-map`).

8. **ABI anchors must never be widened (v0.36.0+)**: the cascade,
   the translate-time relax, AND the post-condition invariant all
   share one source of truth: `conflict_classifier::is_abi_anchor()`.
   Any new failure mode that requires "don't touch this dep" gets
   added to `ABI_ANCHOR_NAMES` or `is_abi_anchor_pattern`, NOT a
   per-call-site special case. Widening `python`, `python_abi`,
   `cuda-version`, `__cuda`, `__glibc`, `libstdcxx-ng`,
   `*_compiler`, or arch-tagged compilers (`gcc_linux-*` etc.)
   corrupts every downstream env's ABI. The invariant
   `check_output_abi_invariants` catches violations at the
   output level; the classifier filters at verdict-emission time;
   the refinement loop re-checks defense-in-depth. Three layers
   because the previous "one layer" model regressed three times
   between v0.32 and v0.35.

9. **Per-env state isolation in conda_outputs (v0.36.1+)**: the
   env_names inner loop snapshots `(bundle, effective)` BEFORE
   the loop and resets at the start of each iteration. NEVER let
   one env's cascade widenings leak into a sibling env's solve
   check via shared mutable state -- that produces false-sat
   results for envs whose feature set is a strict superset of a
   failing env. solve_diagnostics accumulates outside the loop
   and is transferred back to bundle.solve_diagnostics after.
   See `iterative_solve_refinement` call site in `conda_outputs`.

10. **Channel-priority defaults to Strict (v0.36.3+)**: matches
    pixi's own default. Workspace `[workspace].channel-priority`
    setting wins when present. retread's solve check reads
    `WorkspaceManifest.channel_priority`; falls back to Strict
    when unspecified. Disabled is rarely correct -- it lets
    cross-channel raw-version comparison override channel ORDER,
    which defeats the whole point of listing a channel first.

11. **Per-feature channel scoping is the conventional pattern**:
    workspace top-level `channels` apply to EVERY env. Niche
    channels (pytorch, robostack-humble, prefix.dev/X) belong
    under `[feature.X.channels]` so they only apply to envs that
    activate that feature. Putting a niche channel at workspace
    top-level under strict priority forces every env -- including
    ones that don't want the niche -- to source matching packages
    from there. When pixi's solver returns `... is excluded
    because due to strict channel priority not using this option
    from: '<channel>'`, the answer is **almost always** wrong
    channel scope at the workspace level, NOT a retread bug.

## gigastrap state

- Branch: `memorize_and_ft`. Has uncommitted, unrelated changes.
- Source-package directory: `./isaac-pack/` (renamed from `./isaacsim-repack/`).
- Workspace pixi.toml relevant changes:
  - `preview = ["pixi-build"]` in `[workspace]`
  - `build-variants = { python = ["3.11"] }` in `[workspace]` (inline form)
  - Single conda dep declaration: `isaac-pack = { path = "./isaac-pack" }`
    (under `[feature.isaaclab.dependencies]`)
  - `pytorch3d` commented out of `[feature.gpu.pypi-dependencies]`
    with `# RETREAD:` prefix (moved into isaac-pack)
  - `[feature.gigastrap_sim_physx.pypi-dependencies]` and
    `[feature.gigastrap_sim_newton.pypi-dependencies]` isaaclab editable
    entries commented out with `# RETREAD:` prefix
- Editable overlay handled by `scripts/post-install-gigastrap-sim.bash`
  and `scripts/post-install-gigastrap-sim-newton.bash`: each contains
  an `_overlay_editable` shell function that pip-installs each
  IsaacLab source dir as `--no-deps --force-reinstall` after pixi
  install. Without `--no-deps`, uv would re-read the editable
  pyproject's strict pins (pillow==11.2.1 etc.) and conflict with
  conda's picks. **DISABLED as of v0.21.0** — auto-data inject now
  ships the Kit `extension.toml` files at `$PREFIX/lib/source/<ext>/`
  so Omniverse Kit's extension scanner finds them without the
  editable overlay. The overlay is still useful for HOT-RELOAD editing
  workflows; commented blocks in both scripts so it's a one-line revert.
- isaac-pack/pixi.toml entries (all share `bundle = "isaac-pack"`):
  - `isaacsim` (PyPI extras=[all, extscache])
  - `isaaclab`, `isaaclab-assets`, `isaaclab-tasks`, `isaaclab-mimic` (named-git: isaaclab)
  - `isaaclab-rl` (named-git: isaaclab, **extras=["all"]** — pulls sb3,
    skrl, rl-games via its `git+https://...@python3.11` URL Requires-
    Dist, and rsl-rl)
  - `isaaclab-arena` (named-git: isaaclab-arena)
  - `pytorch3d` (miropsota index, `==0.7.8+5043d15pt2.7.0cu128`)
- isaac-pack's `retread-relax = "patch-then-minor-then-major-then-last-resort"`
  (v0.30.0+ recommended; also the codebase default as of v0.35.3).
  Emits patch widening initially; the cascade widens progressively
  through minor/major/`*` per-dep ONLY when the solve check proves
  the current spec unsat.
- **Workspace channels** (post-v0.36.3 + late-session lessons):
  - `[workspace].channels = ["https://prefix.dev/conda-forge"]`
    (top-level; applies to ALL envs). conda-forge ships GPU
    pytorch via the `pytorch-gpu` package -- the old "conda-forge
    torch is CPU-only" assumption was outdated.
  - `[feature.ros2].channels = ["https://prefix.dev/robostack-humble",
    "https://prefix.dev/conda-forge"]` -- robostack-humble scoped
    to ros2 envs only.
  - `[feature.gpu]` does NOT add any extra channels. conda-forge's
    `pytorch-gpu >=2.7.1,<3` + matching torchvision/torchaudio
    (with `pytorch * cuda*` build tags) work cleanly with workspace
    `cuda-version ==12.8` under strict priority.
  - Earlier setup added `https://prefix.dev/pytorch` at workspace
    top-level (v0.34.5) then at `[feature.gpu]` (v0.36.3). Both
    failed because the pytorch channel doesn't ship a cuda 12.8
    build matrix -- it has cuda 12.6 for older torch or 12.9+ for
    newer. Removed entirely; conda-forge handles both CPU and GPU
    torch coherently now.
- `[workspace].channel-priority = "strict"` (post-v0.36.3). Was
  `disabled` in v0.34.5 as a misread "fix"; strict is the right
  default per pixi conventions.

## Recent fix log (since the original v0.9.0 handoff)

- **v0.36.0** (2026-05-27): PER-CHAIN VERDICTS + ABI-ANCHOR INVARIANT.
  Closes the "each version exposes the next bug" pattern that
  plagued v0.32.0 -> v0.35.3 by adding the structural invariant
  v0.34.0 was missing: **the cascade may never widen an ABI anchor,
  full stop, regardless of who emits it**. v0.35.0's classifier knew
  python was workspace-pinned but the refinement loop's widening
  branch read only the aggregate `ConflictClass` (which picked A
  because pytorch was also a blocker) and walked every name in
  `blocking_deps`, finding python in `emitted_names` (retread emits
  it from wheel Requires-Dist) and widening it to `*`. The corrupted
  output shipped to pixi; pixi's full solve picked python 3.14;
  every transitive `python_abi 3.11.* *_cp311` then unsat-ed against
  a misleading leaf (`gymnasium 1.2.1 ... no candidates`). See the
  `retread-probe-trace-isaac-pack.json` round-4 widening at
  `examples/gigastrap/isaac-pack/` for the smoking gun.
  - **New module shape**: `src/conflict_classifier.rs` exports
    `classify_chains() -> Vec<PerChainVerdict>` and the
    `is_abi_anchor()` predicate. Verdicts:
    `WidenRetread { dep, current_spec }`,
    `AbiAnchor { dep, reason }`,
    `WorkspacePinDominates { dep, suggestion, also_emitted_by_retread }`,
    `AlreadyExhausted { dep, current_spec, transitive_requirement }`,
    `TransitiveOnly { dep }`.
  - **ABI anchor list** (single source of truth in `is_abi_anchor`):
    exact names `python`, `python_abi`, `pypy`, `libc`, `glibc`,
    `__glibc`, `libstdcxx-ng`, `libstdcxx`, `libcxx`, `libcxx-devel`,
    `cuda-version`, `__cuda`; pattern matches `__*` (every rattler
    virtual package), `*_compiler` suffix, and the
    `gcc_/gxx_/g++_/gfortran_/clang_/clangxx_/binutils_/ld_/sysroot_`
    arch-tagged prefixes. Bias: prefer false-positives (more entries)
    over false-negatives -- a spurious "never widen" just degrades
    the error message; a missing entry corrupts the output.
  - **The aggregate `ConflictClass` + `classify_unsat()` are KEPT**
    as a derived-from-verdicts roll-up label for the audit pipeline
    and RPC error tags. The refinement loop no longer uses them.
  - **Refinement-loop rewrite** in
    `handler::iterative_solve_refinement`: iterates verdicts and
    widens ONLY `WidenRetread`. If no verdict is widenable, stops
    the loop and surfaces `derive_class_tag()` (which prefers `B-`
    over `A-exhausted` so the user sees the actionable suggestion
    first). Defense-in-depth: even if `classify_chains` ever leaked
    an ABI anchor into `WidenRetread`, the loop re-checks
    `is_abi_anchor` before mutating `effective.overrides` and skips
    with a `tracing::error!`.
  - **Post-condition invariant** (`check_output_abi_invariants`):
    runs after every `produce_output` call inside the loop. Three
    checks: (1) ABI-anchor names in `run_dependencies.depends` must
    not have empty/`*` specs; (2) workspace-coemitted anchors are
    trace-logged for solver-reconciliation visibility; (3)
    `effective.overrides` must not carry ABI-anchor->`*` entries.
    Violations are logged at `error` level, `debug_assert!`'d so
    test runs fail-fast, and recorded under
    `RefinementStep.invariant_violations`. Does NOT fail the
    cascade -- the invariant is a safety-net, not a precondition.
  - **`RefinementStep` audit struct** gains `verdicts:
    Vec<PerChainVerdict>` (per-iteration verdict transparency) and
    `invariant_violations: Vec<String>`. Existing fields retained
    for back-compat.
  - **`conda_outputs` fail gate tightened**: was
    `all_solve_attempted && !any_solve_passed && !workspace_block_messages.is_empty()`.
    Is now `all_solve_attempted && !workspace_block_messages.is_empty()`.
    The old gate hid silently-corrupted outputs (Bug 1 made 3 of 4
    gigastrap envs falsely "pass" against the pre-emission solve
    check, so retread shipped them and pixi exploded downstream).
    With the ABI-anchor invariant in place, "any actionable
    workspace conflict" is the right gate -- the user gets the
    structured RPC error pointing at `RETREAD-SOLVE-FAILED-*.md`
    even when sibling envs are independently solvable.
  - **Tests**: 146 lib tests pass (was 125 on 0.35.3). New
    coverage: 7 `is_abi_anchor` predicate tests (python, libc,
    cuda runtime, virtual packages, compilers, arch-tagged, regular
    deps); 6 `classify_chains` per-verdict tests including the
    `classify_chains_mixed_gsi_round4_scenario` fixture that pins
    the exact failure mode that motivated v0.36.0; 7
    `check_output_abi_invariants` tests covering python widened to
    `*`, empty spec, concrete spec passes, override map corruption,
    non-anchor pytorch widening (passes), libstdcxx-ng, arch-tagged
    compiler activation; 1 simulated refinement-loop test asserting
    python is never widened even if a buggy verdict says so.
  - **Judgment calls deviating from the brief**:
    1. The brief said "stop the loop and surface suggestions" on
       any non-widenable verdict. I kept the looser policy "widen
       every widenable; stop only when none remain widenable" --
       this lets the cascade widen `pytorch` to resolve a gsi-shape
       conflict on the NEXT round even when `python` (ABI anchor)
       co-appears in this round's blockers. The brief's policy
       would stop after iter 0 in that case, never trying the
       widen-pytorch step, and the user gets a useless workspace
       suggestion. Net effect: same correctness (`python` is
       never widened either way), better cascade progress.
    2. Kept `class_label` as `pub` with `#[allow(dead_code)]`
       instead of deleting it. The audit MD writer at
       `handler.rs:3879` reads `terminal_classification` strings;
       any future code path that produces a `ConflictClass` should
       use the same label set, so the function stays as the public
       canonical mapping.
    3. The invariant's "workspace pin looseness" check (post-cond
       #2 in the brief) is conservative: it only trace-logs rather
       than flagging, because parsing both specs as `VersionSpec`
       and computing intersection is out-of-scope spec-math for
       v0.36.0. The (1) check (empty/`*` on ABI anchor) catches
       the gsi corruption directly; intersection-correctness is a
       future tightening when we hit a failure mode it would catch.
  - **What this version closes**: the "each version exposes the
    next bug" pattern. The structural invariant -- "no ABI anchor
    is ever widened, no matter who emits it" -- is now enforced at
    THREE layers (`relax.rs` translate-time was already exempt for
    `python`; `classify_chains` filters at the classifier level;
    `iterative_solve_refinement`'s widening branch re-checks
    defense-in-depth; `check_output_abi_invariants` post-checks
    the output). Each future "we corrupted the output by widening
    X" report can now be addressed by adding X to
    `ABI_ANCHOR_NAMES` (or adding a new pattern to
    `is_abi_anchor_pattern`) instead of a new policy patch.

- **v0.36.1** (2026-05-27): cross-env state-leak fix + misleading-
  scope wording fix.
  - The env_names loop inside `conda_outputs` was re-using
    `&mut bundle` / `&mut effective` across iterations. So env A's
    cascade widenings (`pytorch -> >=1`, `torchaudio -> >=2.7,<3`)
    leaked into `effective.overrides` for env B's solve, making
    sibling envs falsely `sat=True`. The user caught this asking
    "why did gsi-ros2 sat when gsi unsat? gsi-ros2 ⊇ gsi" -- correct
    intuition; a strict feature superset can't independently satisfy
    if the subset fails.
  - Fix: snapshot `(bundle, effective)` BEFORE the env loop;
    reset both at the start of each iteration. `solve_diagnostics`
    accumulates outside via `accumulated_diagnostics:
    BTreeMap<String, SolveDiagnostics>` and is transferred back to
    `bundle.solve_diagnostics` after the loop ends.
  - Stale "every env failed" wording in the RPC error message was
    triggering even on partial failure. Now tracks
    `envs_failed_with_block: BTreeSet<String>` separately and the
    message says `"1 of 4 envs: [gsi]"` or `"every env (4/4)"`.
  - Parser fix: `extract_blocking_chains` was capturing version-
    enumeration lines (`pytorch-gpu 2.7.1 | 2.7.1`) as
    `transitive_requirement`, producing gibberish suggestions like
    `relax pytorch-gpu 2.7.1 | 2.7.1`. Now skips lines containing
    ` | ` (the rattler version separator). Also tracks
    `installable: bool` per chain (rattler "can be installed"
    vs "cannot be installed") -- installable chains are context,
    not the blocker, so suggestion derivation skips them.

- **v0.36.2** (2026-05-27): iteration cap + suggestion-count
  accuracy.
  - `MAX_REFINEMENT` 5 -> 10. With per-chain verdicts, each round
    may surface a NEW blocker set as earlier widenings unlock
    candidates -- so longer chains need more headroom. 5 was sized
    for the old "widen everything in one round" model.
  - Suggestion count distinguishes real suggestions from
    "see-the-trace" fallback. Message now says either:
    - `"N actionable workspace-edit suggestion(s) at the top of the MD"`
    - `"no auto-suggestion (cascade exhausted; conflict is upstream-
      wheel-vs-workspace-pin and requires manual judgment)"`.
  - MD file gets a "Cascade exhausted -- no auto-suggestion"
    section when no real suggestions exist, with a what-to-look-at
    checklist (per-env classification, refinement steps, verbatim
    unsat chain) + common fixes (bump pytorch-gpu, move to
    pypi-deps, retread-drop-deps).

- **v0.37.1** (2026-05-28): D3a follow-up — strip build strings
  before pushing to the cascade's override map. v0.37.0's
  `split_conda_dep_line` change correctly preserved build strings
  in the spec round-trip (`python_abi 3.11.* *_cp311`,
  `pytorch 2.10.0 cuda*_mkl*303`), but the consumer
  `extract_transitive_constraints` was feeding them verbatim into
  the override map. `join_transitive_to_overrides` then comma-AND'd
  the build-string-bearing specs and produced
  `>=1.4,2.10.0 cuda*_mkl*303,>=2.10.0,<2.11.0a0` — the literal
  space in the middle broke `MatchSpec::from_str` at the
  isaaclab-gpu env. Fix: in `extract_transitive_constraints`, take
  `trans_spec.split_whitespace().next()` before pushing to drop
  the build-string portion. Build strings are still enforced
  transitively by the rattler solver via the package's own
  `depends` array (the override map only carries version
  constraints; that's the contract). Regression test in
  `join_transitive_to_overrides` covers the fallback path if a
  build string ever leaks past this filter.

- **v0.37.0** (2026-05-28): SOLVE-CHECK INPUT PARITY WITH PIXI.
  Architectural correctness pass. After v0.36.4 fixed widening
  propagation, gigastrap's gsi/gsn envs STILL failed in pixi while
  retread's solve_check reported sat=true. Two grizzly investigations
  (initial + independent peer review) identified the root structural
  cause: **retread's solve_check models a different solver input than
  pixi's real solve**, so "sat" in retread does not predict "sat" in
  pixi. Every v0.34→v0.36.4 patch was treating symptoms; the contract
  "retread's verdict predicts pixi's" was never enforced.
  - **D1: system-requirements injection** (the load-bearing fix).
    Workspace `[feature.X.system-requirements]` (e.g.
    `cuda = "12"`, `libc = "2.35"`) tells pixi to inject `__cuda 12`
    / `__glibc 2.35` as virtual packages for the env's solve.
    retread's solve_check previously called
    `rattler_virtual_packages::VirtualPackage::detect()` on the BUILD
    HOST — divergent from pixi's actual solve. Symptom: a workspace
    declaring `cuda = "12"` saw retread succeed on a host with
    `__cuda 13` (host has it), but pixi (using workspace-declared
    `__cuda 12`) rejected `cuda-bindings >=13` deps.
    - `WorkspaceManifest` and `FeatureDef` gain
      `system_requirements: BTreeMap<String, String>`.
    - Parser handles top-level `[system-requirements]` AND
      `[feature.X.system-requirements]`. Scalar values (`cuda = "12"`)
      stored verbatim; table form (`libc = { family = "glibc",
      version = "2.35" }`) takes the `version` field.
    - New `effective_system_requirements(env)` method analogous to
      `effective_dependencies` — top-level + each active feature in
      declaration order with feature-wins precedence.
    - `solve_check::build_virtual_packages(target_python,
      system_requirements)` extracted from `run_solve_check` as a
      pure function. Maps `cuda -> __cuda`, `libc|glibc -> __glibc`,
      `macos|osx -> __osx`, `archspec -> __archspec` (build-string),
      `linux -> __linux`. OVERRIDES host-detected entries — workspace
      values are authoritative. Unrecognized keys trace-log + skip.
    - `run_solve_check` takes a `system_requirements: &BTreeMap<...>`
      parameter. `iterative_solve_refinement` forwards it.
      `conda_outputs` computes per-env via
      `effective_system_requirements`.
  - **D2: reject bare-major python in `pythons_for`**. Pixi
    sometimes forwards `variant_configuration["python"] = ["3"]`
    (just the major) despite the workspace declaring
    `build-variants = { python = ["3.11"] }`. Letting a bare-major
    through poisoned the entire pipeline: solve_check installed
    `__cpython 3.0.0` (not 3.11), `produce_output` emitted `python
    3.*`, ABI checks misread the state. Same validation as
    `conda_build_v1` (handler.rs:944-975): drop bare-major entries
    with a `warn!`, fall back to `config.python` or
    `DEFAULT_PYTHON`. Workaround for an upstream pixi bug; remove
    when pixi stops forwarding bare-major.
  - **D3a: preserve build-string in `split_conda_dep_line`**.
    Previously `splitn(3, ...)` silently discarded the
    `*_cp311`-style build string from transitive constraint lines.
    For most deps the build string is decorative, but for
    `python_abi 3.11.* *_cp311` it distinguishes cpython from pypy.
    Changed to `splitn(2, ...)` so the full spec round-trips through
    `MatchSpec::from_str`. Other deps that benefit: `libstdcxx-ng
    >=12 hb0f4dca_0`, `pytorch 2.7.1 cuda*_mkl*304`.
  - **D3b: documented python_abi filter rationale**. The second
    grizzly's review correctly flagged that REMOVING the filter
    would leak ABI anchors into retread's emission. The original
    grizzly's "two-track" plumbing was over-engineered for the
    actual symptom — the rattler depends graph already enforces
    python_abi transitively via the python concrete package. Filter
    kept; rationale documented at workspace.rs:564-581 so future
    maintainers don't second-guess.
  - **D4: clause-level dedup in `join_transitive_to_overrides`**.
    Pre-0.37 the dedup was at the full-spec-string level. Result
    was junk like
    `setuptools >=41.0.0,>=59.6.0,<80,>=59.6.0,<=79.0.1` in shipped
    meta — two `>=59.6.0` clauses survived because they were
    embedded in different parent specs. Now: split each input by
    `,`, dedup at clause level, rejoin, validate the result parses
    as `VersionSpec` (fallback to plain concat if it doesn't).
  - **D5: pinned `SolveStrategy::Highest` explicitly** in
    solve_check.rs:228. Was `SolveStrategy::default()` (same value,
    but the contract was invisible). Now visible in code review;
    if pixi ever changes its strategy, retread should mirror.
  - **D6: tightened `check_output_abi_invariants`** to flag
    bare-major globs on ABI anchors (`python 3.*`, `cuda-version
    12.*`). Defense-in-depth on top of D2; if a bare-major slips
    past the input boundary in the future, the invariant catches it
    at the output boundary.
  - **T1: `build_virtual_packages` unit tests** pin the workspace-
    requirement → rattler virtual-package mapping (6 tests covering
    cpython, cuda, glibc, archspec build-string encoding, override
    semantics, unknown-key skip). Full end-to-end synthetic-RepoData
    parity test deferred (would require committing repodata
    fixtures); the mapping unit tests cover the load-bearing piece.
  - **T3: `pythons_for` rejection tests** (3 tests: bare-major
    rejected, dotted accepted, mixed list keeps the dotted ones).
  - **T4: build-string preservation tests** in
    `split_conda_dep_line` (regression coverage for the new
    `splitn(2)` shape).
  - **Tests**: 172 lib tests pass (was 152 on v0.36.4). 20 new
    across solve_check, workspace, handler.
  - **Honest framing** (second grizzly's call): the input-divergence
    hypothesis is necessary but not sufficient. Two other structural
    factors contribute to the "each version exposes the next bug"
    pattern and are NOT fixed by v0.37.0:
    1. Cache invalidation races (HANDOFF invariant #5). Even a
       correct fix can ship against a stale backend binary or
       channel repodata.
    2. Multi-layer specification surface drift.
       `extract_transitive_constraints`, `join_transitive_to_overrides`,
       `produce_output`, `post_emit_widen_pass` each transform specs
       without shared invariants. D4 addresses part of (2) but the
       layers' contracts aren't formally enforced.
  - **What this preserves**: v0.36.0 ABI-anchor invariant + classifier;
    v0.36.1 per-env solve-diagnostic isolation; v0.36.4 monotonic
    widening union. All four prior layers stay; v0.37.0 fixes the
    SOLVE-CHECK INPUT that those layers were correctly processing.
  - **What this risks**: D1 makes solve_check stricter. Envs that
    previously called sat may flip to unsat now that workspace-
    declared virtual packages are honored. That's the point — we're
    aligning verdicts — but the cascade may surface workspace
    conflicts v0.36.4 hid.

- **v0.36.4** (2026-05-28): REFINEMENT WIDENING ACTUALLY SHIPS.
  Critical correctness fix. Between v0.34.0 (introduced iterative
  refinement) and v0.36.3, the cascade's widenings were INERT for
  what pixi actually saw. `iterative_solve_refinement` mutated
  `effective.overrides` and re-rendered `produce_output` INTERNALLY
  to re-check via solve_check, but the `output` pushed to pixi was
  the one built BEFORE the env loop. The site that should have
  propagated the widening was a no-op:
  ```rust
  // After refinement we may have widened some overrides; re-derive
  // the run-deps strings from the (potentially updated) output for
  // the next env's loop iteration.
  let _ = &output;
  ```
  — a placeholder comment describing the propagation that never
  actually happened.
  - **Symptom**: retread's solve_check happily reported `sat=true`
    (because it solved against the in-loop widened specs), the
    trace recorded every widening, but pixi received the
    pre-refinement run-deps. With `pytorch ==2.10.0` un-widened,
    pixi's solver transitively required `numpy >=2`, which killed
    the `np126py311` builds of `ros-humble-joint-state-publisher
    2.4.0`. Rattler's unsat tree then fell through to the deepest
    leaf — the 2.3.0 py3.9 builds — and printed the misleading
    `python_abi 3.9.*, for which no candidates were found`. The
    py3.9 mention has nothing to do with the actual cause.
  - **Fix** (handler.rs::conda_outputs):
    1. New helpers `widening_level(spec) -> u8` and
       `merge_looser_override(accum, dep, candidate)`. They
       compute a total-ordered "looseness" (patch=0, minor=1,
       major=2, star=3, exact-pin=0) and union per-env widenings
       monotonically (loosest spec per dep wins).
    2. Before the env loop, snapshot the baseline overrides into
       `accumulated_overrides`. Each iteration runs
       `iterative_solve_refinement` against a fresh snapshot of
       bundle+effective (v0.36.1's isolation preserved for
       diagnostic accuracy), then merges that env's final
       `effective.overrides` into `accumulated_overrides` via
       `merge_looser_override`.
    3. After the env loop, if the accumulated overrides differ
       from baseline, REBUILD the output: restore the base bundle,
       apply `accumulated_overrides` into a fresh effective, call
       `produce_output`, re-run `post_emit_widen_pass`. Replace
       `output` with the rebuilt one. That's what `outputs.push`'s.
  - **Why monotonic union, not snapshot's overrides directly**:
    each env's refinement diverges from the baseline. Env A may
    widen pytorch to `*`; env B may stop at minor; env C may not
    widen pytorch at all. The shipped run-deps must satisfy every
    env, so the emitted spec must be the LOOSEST across envs.
    Taking the max widening level per dep is the natural lattice
    join over `widen_one_level`'s steps.
  - **Why diagnostic isolation stays**: v0.36.1's snapshot/restore
    pattern was the right call for solve-diagnostic correctness.
    The bug it fixed (gsi-ros2 falsely sat because gsi's widening
    leaked in) is real. v0.36.4 doesn't undo that — each env's
    `solve_diagnostics` entry still reflects the env's own
    refinement journey. The new union is a SEPARATE accumulator
    for what to ship; it doesn't feed back into per-env solve
    checks.
  - **Tests added**: 6 new (152 total, was 146). `widening_level`
    coverage for patch/minor/major/star ordering, exact-pin
    handling (`==X.Y.Z` is level 0 not 2), pure-upper-bound
    handling (`<2` is level 0). `merge_looser_override` keeps
    widest across envs, never narrows. New end-to-end test
    `produce_output_reflects_overrides_for_refinement_widening`:
    builds a synthetic two-dep bundle (one to widen, one control),
    asserts that applying the union'd overrides into effective
    and re-rendering ships the widened spec while the control dep
    renders byte-identically to baseline. Before v0.36.4 this
    test would catch the regression because the widened output's
    `dep-widened` would still carry the pre-refinement rendering.
  - **Not touched**: the variant-python="3" mystery in the meta-v0
    cache (workspace says `build-variants = { python = ["3.11"] }`
    but pixi forwarded only `"3"`). Orthogonal: workspace's
    `python ==3.11` in `[dependencies]` dominates regardless. Worth
    investigating in pixi if it recurs.

- **v0.36.3** (2026-05-27): channel-priority default reverted to
  pixi's own default (Strict).
  - v0.34.5 changed retread's default to `Disabled` thinking strict
    was over-rejecting. That was a misread -- strict was doing
    exactly the right thing (each package from the first channel
    that lists it). Disabled let conda-forge's CPU torchaudio
    compete with pytorch channel's GPU torchaudio on raw version
    comparison and pick the wrong one.
  - retread now defaults to `Strict` when the workspace doesn't
    specify `[workspace].channel-priority`. Workspace's explicit
    setting still wins (Disabled, Strict, or unset -> Strict).

- **Late-session channel-priority lessons** (2026-05-27, not a
  retread code change but worth capturing for future debugging):
  - When the unsat chain contains lines like `package X is excluded
    because due to strict channel priority not using this option
    from: 'https://prefix.dev/conda-forge/'`, the conflict is
    almost always **wrong channel scope at the workspace level**,
    not a retread bug. The "channel-priority not using this option"
    string is rattler's most informative diagnostic -- when it
    appears, the answer is always "the channels list is wrong for
    this env."
  - **Per-feature channels** are how pixi handles mixed-toolchain
    envs (CPU/GPU/ROS). Workspace top-level `channels` apply to
    EVERY env. Putting a niche channel (pytorch, robostack-humble)
    in workspace top-level forces every env to source from it under
    strict priority. The clean pattern: declare niche channels under
    `[feature.X.channels]` so they apply only to envs that activate
    that feature.
  - gigastrap's current pattern (post-v0.36.3):
    - `[workspace].channels = ["https://prefix.dev/conda-forge"]`
      (single channel; conda-forge does have GPU pytorch via
      `pytorch-gpu` package -- the old "conda-forge is CPU-only"
      assumption is outdated).
    - `[feature.ros2].channels` adds robostack-humble for ros2 envs.
    - `[feature.gpu]` does NOT add the pytorch channel anymore --
      conda-forge's `pytorch-gpu` + matching torchvision/torchaudio
      (with `pytorch * cuda*` build tags) work cleanly under strict
      priority with cuda-version 12.8.
  - The pytorch channel was tried (v0.34.5->v0.36.3) for GPU torch
    but it only ships cuda 12.6 (older torch) or cuda 12.9+ (newer);
    no cuda 12.8 build matrix. The workspace's `cuda-version ==12.8`
    pin (matching Isaac Sim's tested CUDA) doesn't intersect with
    what pytorch channel ships. conda-forge does have matching
    cuda 12.8 builds.

- v0.9.1: per-entry error context surfaces in pixi (no more silent
  "package not provided")
- v0.9.2: range version specifiers supported in PyPI resolution
  (isaacsim's `isaacsim-extscache-kit>=5 ; extra == "extscache"`)
- v0.9.3: recipe source URL points at the post-D wheel
- v0.9.4-5: `binary_relocation: false`, nested `dynamic_linking`
- v0.9.6: prefer-conda default (parselmouth-mapped deps skip auto-bundle)
- v0.9.7: cross-output sibling deps + percent-decoding of wheel URLs
- v0.9.8: `python ${{ ver }}.*` glob
- v0.9.9: per-python variant build of retread itself; `WheelEntry.bundle`
- v0.10.0: `strong-major` + `conda-aware` policies; `retread-audit.json`
- v0.11.x: relax-policy refinements + range-spec passthrough
- **v0.12.0**: AUTO-DATA-FILES INJECT. `wheel_inject_data.rs` walks the
  upstream repo's checkout root (parent of subdir), respects `.gitignore`,
  ships every non-Python sibling file as `<wheel>.data/data/lib/<rel>`
  so it lands at `$PREFIX/lib/<rel>` after install. Solves the IsaacLab
  `__file__ + 4 up + apps/` pattern where the `.kit` experience files
  live at the repo root, not inside any pip-wheel'd subdir. Per-bundle
  dedup so the same `apps/` ships exactly once across the 6 IsaacLab
  named-git entries. Also lifted `extras = [...]` restriction to allow
  it on path/git/named-git forms (URL still rejected). Also added URL-
  Requires-Dist support in extras BFS (PEP 508 `pkg @ git+...@<rev>`
  and `pkg @ https://.../foo.whl`); unlocks IsaacLab's `rl_games`
  extra without hand-maintaining a post-install pip install.
- **v0.13.x**: ergonomic + bugfix series.
  - **v0.13.7**: wheel-cache skip-list now matches `.contains(".injected.")`
    etc. (was `.ends_with`) so multi-suffix names like
    `*.injected.autodata.whl` don't masquerade as raw wheels and trigger
    suffix-accumulator runaway (was a real production failure: filenames
    grew to 250+ chars per solve until ENAMETOOLONG fired).
  - **v0.13.3**: git-clone cache layout is now `<slug>/<sha12>/` (hierarchy)
    instead of `<slug>-<rev>/` (flat 60+ char name).
  - **v0.13.4**: `run_silent` propagates the failing subprocess's stderr
    in the bail message so git/uv errors actually surface.
  - **v0.13.6**: every phase in `materialize_and_rewrite` and every
    syscall in `source_build.rs` is wrapped with `with_context` so a raw
    `os error 36` no longer surfaces without identifying the offending
    file/operation.
  - **v0.13.7**: extras on git/path entries flow through the extras BFS
    same as PyPI form, with empty-spec ("bare-name") coerced to `*`.
  - **v0.13.8**: prefer-conda BFS short-circuit (skip PyPI resolve when
    parselmouth knows the conda equivalent; URL/git Pending sources
    never short-circuit).
  - **v0.13.9**: BFS recurse on URL/git sub-wheels uses parent entry's
    `index_url()` (was incorrectly passing the name-prefix string as
    an "index URL" -> "relative URL without a base" errors).
  - **v0.13.10**: abi3 wheel-tag matching: `cp36-abi3` now matches any
    target python >= 3.6 (psutil's stable-ABI wheels were being rejected).
  - **v0.13.11**: BFS short-circuit became probe-gated -- if conda
    channels don't have a satisfying version, fall through to PyPI.
- **v0.14.0**: python-aware probe. Each conda variant's `depends`
  string is parsed for `python` / `python_abi` constraints; probe
  rejects variants whose python doesn't match the workspace target.
  Was rejecting gym 0.23.1 (py3.10-only build) for py3.11 target.
- **v0.16.0**: BFS-picker fix. `pypi_to_conda` is the INVERTED parselmouth
  map -- many conda packages list the same PyPI dep, so `.first()`
  silently picked nonsense like `numpy -> manifpy` and `torch -> pytorch-cpu`.
  New picker: identity match wins, else single-candidate, else ambiguous-
  no-identity skips the short-circuit entirely. Plus the audit's
  `ProbeDecision` records every routing choice, AND `<pack>/
  retread-probe-trace-<name>.json` is written during `conda/outputs`
  so the trace survives downstream conda-solve failures.
- **v0.17.0-1**: probe-spec normalization (strip ", " -> ","), bare-name
  spec coerce to `*`.
- **v0.18.0**: BFS sdist fallback. When `pypi::resolve` returns "no
  wheels match", `resolve_sdist` finds the matching tarball and
  `build_wheel_from_sdist_url` runs `uv build --wheel` on it. Unlocks
  gym (and any other sdist-only PyPI package) without hand-maintained
  post-install pip-install.
- **v0.19.0**: `*-with-last-resort` family of relax policies (patch/
  minor/major). PRE-EMIT cascade in `last_resort_widen_pass`: for each
  emitted conda spec that the strict probe says unsat, probe `*` and
  inject `pkg = "*"` into effective overrides. Surgical alternative
  to `strong-major` which strips uppers bundle-wide.
- **v0.21.0**: `skip_subdirs` in auto-data-files inject only drops
  Python source/cache/build-meta (not Kit `extension.toml`, data dirs,
  README, etc.). Solves Omniverse Kit's "untrusted extension" error
  -- Kit's extension scanner finds `${app}/../source/<ext>/config/
  extension.toml` because it's now shipped via the data inject even
  though the Python package source is wheel'd instead.
- **v0.22.0**: probe rewritten to use **conda repodata.json** directly,
  not the prefix.dev `/api/v1/.../variants` endpoint (which was returning
  404 on every call -- silently false-unsat across every probe!).
  Fetches `<channel>/<subdir>/repodata.json.zst` per (channel, subdir),
  parallelizes via `FuturesUnordered`, caches in-memory (Arc) within
  process + on-disk (30min TTL) across invocations.
- **v0.23.0**: POST-EMIT widening. After `produce_output` computes
  `output.run_dependencies.depends`, walk it and probe each spec.
  Definitively-unsat → mutate spec to `*` in place AND record under
  `stage = "post-emit-widen"`. The architecturally-correct probe layer
  -- pre-emit cascade in `resolve_all` can disagree with what
  `produce_output` actually emits (cross-output siblings, marker
  subtleties, dedup ordering), and the post-emit pass probes the
  GROUND TRUTH that conda solver sees.
- **v0.30.0**: TIERED CASCADE policy + always-on probes.
  - New `RelaxPolicy::PatchThenMinorThenMajorThenLastResort`
    (TOML: `patch-then-minor-then-major-then-last-resort`). Translate
    emits at patch widening; the pre-emit pass escalates per dep
    through patch -> PyPI -> minor -> PyPI -> major -> PyPI -> `*`,
    stopping at the first level that satisfies. PyPI hits bundle the
    wheel and push the PyPI name into `effective.drop_deps` so
    translate skips the conda emission. Each step records a
    `ProbeDecision` under stage
    `tiered-cascade-step{1..7}-{conda,pypi,last-resort}`.
  - `last_resort_widen_pass` renamed to `pre_emit_widen_pass`. Both
    widen passes now ALWAYS run (no `has_last_resort()` gate); they
    probe + record regardless of policy, but only mutate when
    `RelaxPolicy::allows_widening_mutation()` is true. Audit captures
    a probe trace even under plain `minor`/`major`/`strong-major` so
    users can see what *would* be widened.
  - New helpers on `RelaxPolicy`: `has_tiered_cascade()`,
    `allows_widening_mutation()`.
- **v0.31.0**: WORKSPACE PIN AT LAST RESORT. The cascade's step 7
  (and the `*-with-last-resort` widening branch) now reads the
  consumer workspace's `pixi.toml` `[dependencies]` table at retread
  time and mirrors any found pin instead of emitting `*`. The
  workspace path comes from `params.workspace_directory` (passed by
  pixi at initialize). Feature-gated deps (`[feature.X.dependencies]`)
  are skipped — retread doesn't know which features are enabled.
  Best-effort lookup; falls through to `*` if the file's missing,
  unparseable, or doesn't pin the dep. Audit decisions record
  `routing_decision = "widened-to-workspace-pin"` vs
  `"widened-to-any-version"` so users can tell where the spec came
  from. `toml = "0.8"` added as a dep.
- **v0.32.0**: AUTODISCOVERED PER-ENV OUTPUTS + TRANSITIVE CONSTRAINTS.
  - New `src/workspace.rs` parses the workspace pixi.toml's
    `[environments]`, `[feature.X.*]`, and path-form deps.
  - `WorkspaceManifest::discover_outputs_for_source(workspace_dir,
    source_dir)` walks every feature's path-deps. Any whose path
    resolves to the source package retread is building becomes a
    discovered output. Output names come straight from the workspace
    dep declarations (e.g. `isaac-pack-physx = { path = "./isaac-pack" }`
    → output named `isaac-pack-physx`).
  - `WorkspaceManifest::extract_transitive_constraints(env, ...)`:
    for each conda dep an env declares, fetches that dep's latest
    target-python-compatible build's `depends` array from repodata
    (new `probe::fetch_latest_build_depends`) and accumulates
    `(transitive_dep_name, spec)` pairs. So if env declares
    `ros-humble-joint-state-publisher = "*"` and its latest py3.11
    build depends on `numpy >=1.26,<2`, retread learns that numpy
    must be `<2` for that env.
  - In `handler::discover_emissions`: per discovered output, union
    transitive constraints across every env that references it (via
    its declaring feature). Comma-AND join the specs into
    `effective.overrides`. The cascade respects overrides as
    authoritative -- workspace transitive wins, no widening past it.
  - **Removed**: `retread-per-env` config field (was briefly added
    pre-release; superseded by autodiscovery before shipping). Also
    removed v0.31.0's `read_workspace_pins` helper -- the
    transitive-constraint path subsumes it.
  - **Bundle content is shared across per-env outputs**: the
    materialize phase (`resolve_all`) runs once per python variant;
    only the cascade + emission re-runs per env. No duplicate wheel
    downloads.
  - **Pre/post-emit widen passes** moved out of `resolve_all` into
    the per-env emission loop so each env runs the cascade with its
    own channels + overrides.
  - **Audit paths** automatically vary per output because
    `retread-{audit,probe-trace}-<bundle.conda_name>.json` and
    `conda_name` now equals the discovered output name.
  - **conda_build_v1** mirrors the autodiscovery: matches requested
    output name against discovered emissions, applies the right one
    to the base bundle, builds.
- **v0.32.1**: `*` filter for transitive constraint joining. Some
  workspace conda deps' `depends` arrays list transitives with bare
  `*` constraints (e.g. transformers/timm declare `pytorch *`). Comma-
  joining `*` with other constraints produced invalid match-specs like
  `pytorch >=1.4,*,==2.10.0` that conda's parser rejected. Fix:
  `workspace::extract_transitive_constraints` and
  `handler::join_transitive_to_overrides` both drop empty + `*` specs
  before joining. They impose zero constraint so dropping is
  information-preserving; including them was syntactically corrupting
  the comma-AND chain.
- **v0.35.0**: CONFLICT CLASSIFIER + WORKSPACE-EDIT SUGGESTIONS.
  Architectural fix from the-grizzly (2026-05-27). Previous versions
  treated every UNSAT as "retread emitted too-narrow" and widened its
  own emission. This whack-a-moled when the binding constraint was
  actually a workspace pin retread can't widen (gigastrap's
  `torchvision >=0.22.0` selecting torchvision 0.25 → pytorch 2.10 vs
  `pytorch-gpu >=2.7.1,<3`). v0.35.0 classifies each UNSAT into:
  - **A-retread-widenable**: cascade widens, continue loop
  - **A-exhausted**: cascade hit `*`, transitive blocks
  - **B-workspace-pin-dominates**: workspace pin is the floor; emit
    a workspace-edit suggestion (with `feature.X.dependencies` block
    located via `WorkspaceManifest::find_declaring_feature`)
  - **C-workspace-only**: dep not declared by either side; surface
    as transitive bubble
  - New module `src/conflict_classifier.rs` (pure functions, 10 unit
    tests including the gigastrap fixture).
  - New `solve_check::BlockingChain` + `extract_blocking_chains()`:
    preserves rejected_versions + transitive_requirement we used to
    discard after extracting the dep name.
  - `SolveOutcome` + `SolveDiagnostics` gain
    `workspace_edit_suggestions` + `terminal_classification`.
  - `RefinementStep` gains per-iteration `classification` +
    `blocking_summary` so the audit shows the verdict on each round.
  - `iterative_solve_refinement` calls the classifier at the top of
    every iteration. Stops loop on Class B/C/AExhausted; only widens
    on Class A.
  - `conda_outputs` fails the RPC with a structured error message
    when EVERY attempted env solve fails AND the classifier produced
    workspace suggestions. Pixi displays backend errors verbatim, so
    the user sees the real fix instead of pixi's misleading leaf.
  - `RETREAD-SOLVE-FAILED-<bundle>.md` puts suggestions FIRST,
    per-env classification SECOND, full unsat chain LAST.
- **v0.34.0**: ITERATIVE SOLVE-DRIVEN REFINEMENT. The cascade's
  per-dep probes can't see cross-package conflicts: they only check
  "is this spec satisfiable in isolation," not "does it co-satisfy
  with workspace pins." So retread emits `triton >=3.7.0,<3.8` (from
  isaacsim's `Requires-Dist: triton==3.7.0`); conda's triton 3.7 needs
  cuda 13; workspace pins cuda 12.8 — the per-dep probe never sees
  the collision. v0.34.0 closes the gap with `iterative_solve_refinement`:
  - After produce_output, run the solve check (per-env).
  - If UNSAT, parse the blocking dep names from rattler's tree-
    formatted Unsolvable explanation
    (`solve_check::extract_blocking_dep_names`).
  - Intersect with retread-emitted dep names. Widen any matched dep
    ONE level (patch -> minor -> major -> `*`) via
    `widen_one_level(current_spec)`. Don't jump to `*` -- the
    tightest spec the solver can backtrack from gives the user the
    closest-to-upstream pin that still solves.
  - Re-run produce_output, re-run solve check. Iterate.
  - Cap at MAX_REFINEMENT = 5 (then surface the residual conflict
    in the audit; usually it's an external workspace-vs-workspace
    conflict retread can't fix).
  - Each iteration logged as `audit::RefinementStep { iteration,
    blocking_deps, widened_deps }`, persisted on
    `SolveDiagnostics.refinement_steps`. Easy to read in the audit:
    "round 0 widened triton, round 1 widened X, eventually sat."
- **v0.33.0**: PRE-EMISSION SOLVE CHECK. New `src/solve_check.rs`
  runs a real conda solve over (workspace effective deps + retread's
  emitted run-deps + workspace transitive constraints) at the end of
  every per-env emission, using `rattler_solve 4.1.0` (resolvo
  backend). Catches cross-package conflicts the per-dep probe layer
  can't see: e.g. retread emits `cuda-bindings >=13.0.3,<14` (from a
  wheel's Requires-Dist), workspace pins `cuda-toolkit 12.8.*`, both
  have valid candidates BUT their `depends` arrays disagree on the
  shared `cuda` runtime version. The solve check surfaces this with
  `SolveError::Unsolvable(Vec<String>)` explanation strings persisted
  to `retread-audit-<name>.json.solve_diagnostics`. Skipped silently
  when no repodata is cached (best-effort). ~1-3s per (python, env)
  emission; repodata cache is reused from probe.rs.

## Outstanding issues / things to investigate

1. **Pre-emit AND post-emit widen, both always-on (v0.30.0+).**
   `pre_emit_widen_pass` (resolve_all) and `post_emit_widen_pass`
   (conda_outputs) both run unconditionally now. They probe + record
   regardless of policy; mutation is gated on
   `RelaxPolicy::allows_widening_mutation()`. They cover NON-OVERLAPPING
   cases: pre-emit re-translates each wheel's requires_dist and can
   bundle PyPI fallbacks (only place to do that since auto-bundle has
   already run); post-emit probes the GROUND TRUTH of what
   produce_output emitted (catches cross-output siblings and dedup
   ordering effects that pre-emit can't predict). Don't try to fold
   them into one site.

2. **`conda-aware` policy's probe layer is not implemented.** The
   enum variant exists but currently behaves identically to
   `strong-major`. Now that v0.22.0 has the real repodata-backed
   probe, the intended design (per-dep adaptive widening: probe the
   spec, only strip uppers if zero candidates satisfy) is reachable.
   The post-emit pass actually IS the conda-aware behavior; the
   `*WithLastResort` family is the user-facing knob. So `conda-aware`
   could be folded into `MinorWithLastResort` as an alias and
   eventually removed. Or kept as the "no relax at all + widen only
   when proven unsat" variant.

2. **Workspace introspection (Option A in old HANDOFF) is no longer
   urgent.** The original concern was conda→uv pin forwarding when
   the workspace had editable PyPI deps for packages retread also
   bundles. The user resolved this by commenting out the editable
   entries in gigastrap's `[feature.X.pypi-dependencies]` and adding
   `_overlay_editable` to the post-install scripts. With editables
   pinned via `--no-deps --force-reinstall`, uv never sees the
   strict pyproject pins. Option A would still be a nice
   architectural cleanup but isn't blocking.

3. **Audit is informational only.** Nothing reads `retread-audit.json`
   back. If we ever want the audit to be authoritative (e.g. a
   `--write-pixi-blocks` mode that updates the user's workspace
   pixi.toml in place), that's new feature work.

4. **No multi-entry test for path/git sources end-to-end via
   jsonrpc_protocol.** Existing live tests cover single entries.

5. **rattler-build's recipe schema may evolve.** v0.9.5 was needed
   when `binary_relocation` moved under `dynamic_linking`. If new
   rattler-build versions reject our generated recipes, look at
   `src/recipe.rs`.

## Standard commands

```bash
cd /home/garylvov/projects/pixi-build-retread

# build everything with pixi's toolchain (glibc, fast dev iteration)
PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH" cargo build --release

# NOTE: the RELEASE artifact (v1.2.0+) is a statically linked musl binary
# so it runs on any x86_64 Linux regardless of glibc version. The recipe
# bootstraps rustup's musl target at build time (conda-forge ships no
# rust-std for *-musl) and disables _FORTIFY_SOURCE for the C objects
# (ring, zstd-sys) because the __memcpy_chk-style fortified symbols are
# glibc-only. See recipe/recipe.yaml's build script -- rebuild-local.sh
# exercises that path; the plain `cargo build` above does NOT.

# FULL local rebuild + cache nuke. Use the script:
#
#   bash scripts/rebuild-local.sh
#   # or, also nuke the consumer workspace's pixi caches:
#   CONSUMER_PROJECT=/home/garylvov/projects/gigastrap bash scripts/rebuild-local.sh
#
# Bump `Cargo.toml` + `recipe/recipe.yaml` to the same version BEFORE
# running -- the script aborts on version mismatch. Inline equivalent
# below (kept here as documentation of what the script touches and
# WHY each cache layer matters). ALWAYS run as a unit if not using
# the script -- doing only part of it is the #1 cause of "I fixed the
# bug, why is it still firing the old error?"
#
# (1) Nuke the previous linux-64 artifact AND the linux-64 channel's
#     repodata.json. rattler-build APPENDS to repodata rather than
#     regenerating, so a stale 0.X.Y entry sticks around even after
#     the .conda file is gone and the channel keeps advertising the
#     old version to pixi. Deleting repodata.json forces regen.
#     IMPORTANT: do NOT delete `local-channel/noarch/repodata.json` --
#     retread only builds linux-64 (it ships a native binary), but
#     rattler-build still scans the channel's noarch subdir for build-
#     env resolution and fails with "could not find subdir 'noarch'"
#     if its repodata.json is missing. The empty `{}`-shaped file
#     that's checked in must stay put.
# (2) Nuke the global pixi backend cache (where pixi caches the retread
#     EXECUTABLE keyed by version). If the cache key collides (same
#     build-hash tail across versions), pixi reuses the OLD binary
#     even after the channel advertises a new one.
# (3) Nuke gigastrap's project-local pixi caches (envs / artifacts /
#     bld / meta). `pixi clean` does this too, but at the cost of
#     re-downloading multi-GB conda packages.
# (4) Nuke retread's git-clone cache. Critical after any change to the
#     git-clone layout (e.g. v0.13.3 moved from a flat
#     `<slug>-<rev>/` dirname to a hierarchical `<slug>/<sha12>/`).
#     Without this, the cloner writes to the NEW layout while the
#     resolver (next time around) hits the cached OLD layout. Also
#     necessary if you suspect a clone hit ENAMETOOLONG mid-checkout
#     and left a half-broken tree behind. Cheap to redo -- shallow
#     clones with `--filter=blob:none`.
# (5) Rebuild the artifact -- regenerates linux-64 repodata.json on
#     the way.
# (6) Sanity-check what the channel now advertises BEFORE the gigastrap
#     solve, so you catch stale-repodata before pixi does.
rm -rf /home/garylvov/projects/pixi-build-retread/local-channel/linux-64/pixi-build-retread-*.conda \
       /home/garylvov/projects/pixi-build-retread/local-channel/linux-64/repodata.json \
       ~/.cache/rattler/cache/backends-v0/pixi-build-retread-* \
       ~/.cache/rattler/cache/retread-git-clones \
       /home/garylvov/projects/gigastrap/.pixi/meta-v0/isaac* \
       /home/garylvov/projects/gigastrap/.pixi/bld/isaac* \
       ~/.cache/rattler/cache/bld/metadata-v0/isaac* \
       ~/.cache/rattler/cache/bld/source_metadata-v0/isaac*

# Recovery: if you DID accidentally delete `local-channel/noarch/repodata.json`
# (rattler-build will fail with "could not find subdir 'noarch'"), recreate
# the empty repodata before the build step:
  mkdir -p /home/garylvov/projects/pixi-build-retread/local-channel/noarch
  echo '{"info":{"subdir":"noarch"},"packages":{},"packages.conda":{},"repodata_version":2}' \
    > /home/garylvov/projects/pixi-build-retread/local-channel/noarch/repodata.json

# rebuild local conda channel (multi-variant)
PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH" \
  rattler-build build --recipe recipe/recipe.yaml \
  --variant-config recipe/variants.yaml \
  --output-dir ./local-channel --target-platform linux-64

# verify the channel actually advertises the version you just built --
# if this prints an older version, repodata.json wasn't regenerated
# (most likely the previous step's `rm` missed it; rerun the nuke).
grep -o 'pixi-build-retread-[0-9.]*' \
  /home/garylvov/projects/pixi-build-retread/local-channel/linux-64/repodata.json | sort -u

# upload to prefix.dev (only after verifying locally works)
PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH" \
  rattler-build upload prefix -vv --channel garylvov \
  local-channel/linux-64/pixi-build-retread-X.Y.Z-py311XXXX_0.conda

# regenerate isaacsim METADATA fixtures
python tests/fixtures/fetch_metadata.py

# all lib tests (62, including new audit + strong-major + bundle field)
PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH" cargo test --lib

# heavy live tests (network, several GB downloads)
PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH" \
  cargo test -- --include-ignored

# the protocol-discipline test (catches stdout corruption)
PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH" \
  cargo test --test jsonrpc_protocol -- --include-ignored

# end-to-end against a stripped gigastrap-like workspace
PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH" \
  cargo test --test e2e_ros_isaacsim -- --include-ignored --nocapture

# inspect the audit after a gigastrap solve
cat /home/garylvov/projects/gigastrap/.pixi/bld/isaac-pack-*/recipe-isaac-pack/retread-audit.json
```

## Working style with this user

- Use Read/Edit for known files; only spawn the henry-hudson explorer
  for genuinely open-ended codebase questions (his global instruction).
- He'll often interrupt mid-task with a new question or pivot. Roll
  with it.
- He'll ask "why didn't tests catch this" when a bug ships. The
  honest answer is usually "the test only checked retread's emission,
  not whether downstream accepted it." Add the missing round-trip
  test.
- He prefers ARCHITECTURAL conversation over piecemeal fixes. When
  the fixes start feeling like whack-a-mole, escalate to
  solution-architect or dispatch the-grizzly for breadth.
- He'll push back on hardcoded lists. Prefer external data sources
  (parselmouth) or user-configurable knobs.
- He pushes back on "you should just rebuild for X" — anything that
  isn't auto-determined is friction. The variant + build-variants
  propagation pattern is the model he likes; copy it for similar
  problems.
- DO NOT commit to gigastrap or push without explicit ask — it's his
  active workspace. retread repo IS yours to push as needed when
  work is committed.
- For commits, always include
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
  trailer.
- ALWAYS bump retread's version in `Cargo.toml` AND
  `recipe/recipe.yaml` before rebuilding. Without a bump, pixi may
  keep using the cached old artifact even after a successful
  rattler-build run.

## What I'd suggest you do first

1. If the user reports a build failure, first ask which env (`gsi`
   vs `gsn`) and what the exact error is. Read it carefully — the
   per-entry error context (v0.9.1) usually names the offending
   `[retread-wheels]` entry.
2. Check `retread-audit.json` for the bundle that failed — the
   `wheels[].requires_dist` and `emitted_run_deps[]` sections tell
   you exactly what retread sent to conda.
3. If a dep's spec has an upper bound that's blocking the solve and
   the bundle uses `retread-relax = "strong-major"`, the bound
   should already be stripped. If not, regenerate the bundle (clear
   caches) and re-check the audit.
4. If conda-aware probing is wanted, implement it (see Outstanding
   issue #1). Otherwise stick with strong-major.
5. Don't forget to clear all the caches listed under Standard
   commands after every rebuild AND bump the version.

Good luck.
