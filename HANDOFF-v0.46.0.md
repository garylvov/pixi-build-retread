# Handoff Prompt — pixi-build-retread @ v0.46.0 (2026-06-05)

You're picking up an ongoing collaboration with **Gary Lvov** on
**pixi-build-retread**, a Rust pixi-build backend that repacks PyPI wheels as
conda packages with relaxed dependency pins. Everything below reflects the
current state. For deep per-version history (v0.9 → v0.37) see the large
`HANDOFF.md` in this directory; for the most recent saga see
`memory/project_python_abi_root_cause.md`.

## Status: shipped, no open tasks

- HEAD = `ebcb241` "v0.46.0: torch routing + over-widening fixes, parent-first
  env solving, conda/outputs memo" — committed and pushed to `origin/main`.
- `pixi-build-retread-0.46.0-hb0f4dca_0.conda` uploaded to **prefix.dev/garylvov**.
- Version `0.46.0` consistent across Cargo.toml, Cargo.lock, recipe/recipe.yaml,
  README.md. **185 lib tests pass.**
- The user's last directive ("preserve `<3` upper, then push") is fully done.
- **Nothing outstanding.** Wait for the user's direction before starting work.

## Who Gary is / how to work with him

- Technically deep; runs a real robotics monorepo **gigastrap** at
  `/home/garylvov/projects/gigastrap/`. The in-repo test workspace is
  `examples/gigastrap/`.
- Hates emojis and excess prose. Concise, direct answers.
- Uses pixi for everything — always use the pixi envs (check the relevant
  pixi.toml). For cargo/rattler-build, the toolchain lives in the pixi env.
- Values architectural pushback over "just add an override." He explicitly and
  repeatedly rejects per-project overrides: **fix things generally.**
- "Why didn't tests catch this?" is a recurring (fair) refrain. When you add a
  behavior, add a test that round-trips through whatever consumes it.
- For codebase exploration use the **henry-hudson-codebase-explorer** agent, not
  manual Grep. Grep only for tiny single-purpose confirmations.

## What the backend does

JSON-RPC 2.0 over stdio. Methods: `negotiateCapabilities`, `initialize`,
`conda/outputs`, `conda/build_v1`. **STDOUT is the JSON-RPC channel — never
print to it.** User-visible status during the metadata phase goes to `/dev/tty`
via `crate::status::tty()` because pixi hides backend stderr during
`conda/outputs` even at `-vv`.

Core problem: pixi solves conda first, then runs uv against PyPI with conda's
chosen versions forwarded as hard pins (pixi#5230). Upstream wheels pin their
transitives exactly, so the forwarded pins clash and install fails. retread
(a) rewrites each exact pin to a range in the wheel METADATA + emitted conda
run-deps, and (b) reroutes any shared transitive with a conda equivalent (via
parselmouth + the FALLBACK_PYPI_TO_CONDA table) onto the conda side so uv never
sees it.

Pipeline phases (all in src/handler.rs, ~5000 lines):
1. **materialize wheels** — BFS `resolve_bundle` + `auto_bundle_transitives`
2. **produce_output** — per-env conda metadata
3. **solve_check cascade** — `iterative_solve_refinement`, progressive widening
4. **recipe + rattler-build** — `conda/build_v1`

## What landed recently (v0.44–0.46)

1. **Parent-first env solving** — `conda_outputs` sorts `env_names` ascending by
   `effective_dependencies` count (base env e.g. `isaaclab-gpu` solves first),
   threads `accumulated_overrides` as `seed_overrides` into
   `iterative_solve_refinement`. Child envs apply seeds strictly looser than
   their own (never ABI anchors) and re-emit once. gsi/gsi-ros2 now solve in 1
   attempt instead of 8. SAFE vs the v0.36.2 false-SAT bug because every
   iteration still runs a real `run_solve_check`.
2. **conda/outputs memo** — `CONDA_OUTPUTS_CACHE`
   (`OnceLock<Mutex<HashMap>>`), key =
   `host|build|sorted_channels|variant_config` (excludes work_directory).
   Collapses pixi's 3× redundant multi-env solves into one.
3. **Generic status text** — de-NVIDIA-ified download/resolve messages. The
   per-attempt tty line shows the dep widened + what changed (`last_action`,
   `summarize_changes`), still one line.
4. **Torch bundling fix (general, no override).** torch was bundled at
   2.12.0+cu130 instead of using conda's pinned pytorch 2.7.1, crashing
   `torch._inductor` ("duplicate template name"). Root cause: the primary BFS
   `resolve_bundle` routed using ONLY parselmouth's ambiguous inverted map and
   ignored `name_map` (where the FALLBACK `torch->pytorch` lived) — so emission
   said `pytorch` while the BFS bundled `torch`, and the bundled wheel clobbered
   conda's pytorch at install. FIX: extracted
   `pick_conda_target(dep, name_map, pypi_to_conda)` — name_map wins over
   ambiguous parselmouth; threaded `&effective.name_map` into `resolve_bundle`.
   Plus a name-level conda fallback in BOTH BFS and `auto_bundle_transitives`:
   if the exact resolved version isn't on conda but the package exists at other
   versions, keep it on conda (ABI-correct). Verified: `torch bundled?: False`,
   nvidia/cuda bundled `[]`, 96→77 wheels.
5. **Over-widening fix.** `pytorch` was collapsing to `>=1`.
   `extract_anchor_version` now picks the HIGHEST anchor among merged clauses
   (was the first — a stray `>=1.4`), and `widen_one_level` PRESERVES the upper
   through the major step: `>=2.10,<3` → `>=2,<3` → `*`. Added a 5th
   `widening_level` (0=patch/minor-upper, 1=minor-range, 2=major-floor,
   3=major-open, 4=star); total order stays monotone (enforced by
   `widening_level_strictly_increases_along_widen_chain`). Verified emitted
   `pytorch >=2,<3`.

## Functions to know (src/handler.rs)

`conda_outputs`, `iterative_solve_refinement`, `widen_one_level`,
`extract_anchor_version`, `widening_level`, `merge_looser_override`,
`pick_conda_target`, `resolve_bundle`, `auto_bundle_transitives`,
`summarize_changes`. FALLBACK_PYPI_TO_CONDA (~src/handler.rs:80) includes
`torch->pytorch`, `pywin32->pywin32`.

Other files: `src/status.rs` (`tty`), `src/wheel.rs` (download msg),
`src/relax.rs` (`emit_python_version` — the v0.40 python_abi fix shared by
produce_output + recipe.rs), `src/solve_check.rs`, `src/probe.rs`,
`src/workspace.rs`.

## Standing invariants — do not break

1. **Stdout discipline.** Every subprocess call captures stdout.
   `tests/jsonrpc_protocol.rs` catches regressions.
2. **ABI anchors are NEVER widened** — `python`, `python_abi`, `cuda-version`,
   `__*` virtual packages, `*_compiler`, arch-tagged compilers. Single source of
   truth in `conflict_classifier::is_abi_anchor`. Three enforcement layers
   (classifier, refinement loop re-check, `check_output_abi_invariants`).
3. **Per-env state isolation** in the `conda_outputs` env loop: snapshot
   `(bundle, effective)` before the loop, reset each iteration. One env's
   widenings must never leak into a sibling's solve (the v0.36.1/.2 false-SAT
   bug). The shipped spec is the LOOSEST across envs via `merge_looser_override`.
4. **Recipe source URL = post-D (rewritten) wheel**, never the upstream URL.
5. **RECORD lock-step** when METADATA is rewritten.
6. **Bump the retread version every rebuild** — pixi caches by version.

## How to verify end-to-end (and the gotchas)

From `examples/gigastrap/`: `pixi s -e gsi`. Then in the env:
`python3 -c "import torch; print(torch.__version__)"` should report conda's
**2.7.1**, NOT 2.12.0+cu130.

- **Caching gotcha (cost ~1hr):** re-verifying a SAME-version rebuild reuses a
  cached built package. The reference that survives nuking `.pixi/{bld,meta-v0}`
  AND `~/.cache/rattler/cache/bld` is
  `.pixi/artifacts-v0/<pack>/<hash>/<pack>-*.conda` (keyed by source hash
  `isaac-pack[a29e085f]` in pixi.lock). To force a true rebuild at an unchanged
  version: `rm -rf .pixi/artifacts-v0`.
- **Hung processes:** stale `pixi s -e gsi` processes (the freeze-at-install
  issue) accumulate over days and collide over the isaac-pack cache. Kill them
  before verifying.
- **Reproduction:** flattening retread's emitted run-deps into a plain pixi
  project and solving does NOT reproduce the source-package metadata bugs
  (always SAT). Reproduce via the real source-package path (`pixi lock` on the
  workspace).

## Conventions

No emojis. Use pixi envs. Commit/push only when asked. `HANDOFF*.md` are
scratch — keep them out of commits. Fix generally, never per-project override.
