# Kill the install script — plan (scratch, never commit)

## STATUS: IMPLEMENTED v1.8.0 (2026-06-12) — grizzly audit PASSED, pending e2e
Grizzly audit verdict: APPROVE-WITH-CHANGES. Both required code fixes applied:
- FIX-1: `collect_prerelease_pins` source 3 now uses the SAME pin precedence
  as `build_meta_wheel` (`entry.normalized_version()` first, then `resolved`),
  so the override row can never disagree with the meta-wheel pin (broke when
  a prerelease-pinned entry's wheel lookup missed -> `resolved` None/stable).
- FIX-2: test extended with the divergence cases (3b skewed resolved, 3c
  resolved None). All green.
Remaining = e2e ONLY (BLOCKER B conda/CUDA sharing; release-build protocol
test --include-ignored; gigastrap fresh `git clone && pixi install`). These
need a real Isaac env (multi-GB/GPU) -- not run here.

## E2E RESULTS (2026-06-12, in progress)
- Gate 2 (channel build): PASS. pixi-build-retread-1.8.0 built + advertised.
- Gate 3 (synthetic pack, BLOCKER A source-3): PASS. A `[retread-wheels]`
  entry pinned to tinyobjloader==2.0.0rc13 produced the
  `[pypi-options.dependency-overrides] "tinyobjloader" = "==2.0.0rc13"` row
  in the auto-synced fence block; `pixi lock` resolved the prerelease from
  PyPI (tinyobjloader is transitive-via-meta-wheel, so ONLY the override row
  could enable its prerelease -> the new code is what made it work). Fence
  mechanics also validated: no install.sh/overrides.txt, blueprint.lock
  written, meta-wheel shipped, loud re-lock warn fired.
- **FINDING-C (NEW, real):** pixi/uv suppresses the default PyPI index when a
  feature declares `find-links` with no `index-url`/`extra-index-urls`. The
  auto-synced fence block sets find-links but declares NO index, so
  index-origin entries (tinyobjloader here; isaacsim from pypi.nvidia.com in
  the real packs) are "not found" -- even a DIRECT dep failed, proving it is
  an INDEX gap, not a prerelease gap. The deleted `uv pip install --find-links`
  kept PyPI; fence does not. Blast radius: workspaces with ZERO pypi index
  config (e.g. isaac6's minimal workspace). gigastrap is SAFE (declares
  extra-index-urls nvidia+mujoco -> default PyPI stays active). Fix candidate:
  render_snippet_blueprint should emit the manifest-driven index chain
  (distinct entry `index` values as extra-index-urls) so the blueprint is
  self-contained. Small, well-scoped (entries already carry `.index`).
- Gate 5 (gigastrap gsi, BLOCKER B conda torch sharing): **PASS (with consumer
  caveat).** Setup: /tmp/gs-e2e copy, blueprint="only", gsi switched to the
  isaac-pack-pypi feature (conda path dep dropped), wheels hardlinked from the
  prior 5.2G materialization, `pixi lock`.
  - CORE torch/CUDA stack stayed CONDA automatically (explicit [feature.gpu]
    deps): pytorch-2.7.1-cuda126, pytorch-gpu, libtorch, torchvision-0.22-cuda,
    cuda-version-12.8. The headline BLOCKER B fear (torch reinstalled from a
    wheel) did NOT happen. Expected-pypi: pytorch3d (999retread blueprint entry),
    torchrl/warp_lang (PyPI-only).
  - SKEW FOUND + MITIGATED: torchaudio (required by the blueprint wheel
    isaacsim_core, NOT an explicit gigastrap conda dep) first resolved from
    pypi @ 2.11.0, skewed vs conda torch 2.7.1 (conda-path had conda
    torchaudio 2.8.0). Root cause: blueprint="only" drops the conda artifact
    whose run-deps used to drag torchaudio in as conda; the pypi meta-wheel
    closure pulls it from pypi instead. NOT script-specific (standalone uv on a
    blueprint-only env behaves the same -- the conda PATH artifact was the
    anchor, not the script). MITIGATION (verified): pin the torch-family
    transitive as an explicit conda dep (`torchaudio = ">=2.7,<3"` in
    [feature.gpu]) -> torchaudio-2.8.0 + torchcodec-0.6.0 both go CONDA.
  - Also re-confirmed on the real pack: FINDING-C fix (extra-index-urls =
    [miropsota, pypi.nvidia.com] + index-url) and BLOCKER A (a real wild
    prerelease, gmpy2==2.1.0a4, pinned in the micro-table).
- Gate 6 (full GPU install + runtime): **PASS.** `pixi install -e gsi` on
  /tmp/gs-e2e succeeded (blueprint via fence). Runtime smoke on the env python:
  torch 2.7.1 / torchvision 0.22.0 / torchaudio 2.8.0 (all conda), CUDA
  available (4 devices), torchaudio C-ext loads, GPU matmul runs -> torch-family
  ABI clean, __cuda detection survived. Blueprint content present: isaac-pack-pypi
  0.51.1 meta-wheel, isaaclab* git-built wheels, isaacsim 5.1.0.0 + all
  isaacsim-* from pypi.nvidia.com (FINDING-C index chain working). Only minor
  pypi-over-conda override: matplotlib-base (not ABI-critical). Kit boot itself
  not run (needs full gigastrap submodules/activation scripts absent from the
  stripped /tmp copy).

## COMMITTED: branch kill-install-script, commit b9d6782. pre-commit all green
(trailing-ws, eof, yaml, toml, merge-conflict, line-ending, cargo fmt, clippy,
cargo test fast). e2e GATE COMPLETE + GREEN through Gate 6.

## CONSUMER CAVEAT (document): when consuming a blueprint in pypi/"only" mode,
pin ABI-sensitive torch-family transitives that the pack's wheels require
(torchaudio, torchcodec, ...) as explicit conda deps in the consuming
workspace. The blueprint's pypi closure will NOT anchor them to your conda
torch otherwise (the dropped conda artifact used to). The core
torch/torchvision/pytorch-gpu/cuda stay conda automatically when declared.

## (history) IMPLEMENTED v1.8.0 — pre-audit snapshot
Done: prerelease-gap fix (`collect_prerelease_pins`, 3 sources) + unit test;
stale-lock guard (`blueprint.lock` marker + loud warn); config
`blueprint_sync` -> deprecated ignored `Option<String>` + warn (BlueprintSync
enum deleted, deny_unknown_fields preserved); emit() Script branch deleted,
fence always-sync, `remove_workspace_block` retained; `render_install_script`
+ `render_overrides_file` deleted; README + migration note; version bump
1.7.2 -> 1.8.0 (Cargo.toml, recipe, Cargo.lock). GREEN: cargo fmt --check,
clippy -D warnings (0), cargo test --lib (251 pass). NOT YET RUN (gates):
release-build protocol test --include-ignored; gigastrap e2e incl. BLOCKER B
(no torch wheel reinstall) + a prerelease entry/shipped case.


Goal: eliminate the generated `install.sh` + `overrides.txt` path. Blueprint
reaches the workspace ONLY through pixi-native fence auto-sync (single
pixi.lock, `pixi list` visibility). Decided with Gary 2026-06-12.

## Why this is now safe (audit findings)

1. **Fence is already fully wired.** `BlueprintSync::Fence` exists; emit()
   already calls `sync_workspace_block` / `remove_workspace_block`; mode
   switches already converge. Killing the script is delete + default-flip +
   validate, NOT new feature work.
2. **The script's headline justification dissolves in fence mode.** The
   activation-hook self-heal existed because the uv overlay was OUTSIDE pixi's
   knowledge, so pixi pruned it on env reconcile. In fence mode the pypi deps
   live in the manifest, pixi LOCKS them, and never prunes them. No self-heal
   needed.
3. **Prerelease is already handled pixi-native.** `render_snippet_blueprint`
   (emit_pypi.rs:604-655) emits a minimal `[pypi-options.dependency-overrides]`
   micro-table with ONE line per prerelease-only floor. Per the in-code proof
   (and commit d8bc4de) this is uv's architectural floor: uv builds its
   explicit-prerelease set only from direct requirements + overrides, so a
   prerelease-only dep NEEDS that override line; build-tags + meta-wheel
   Requires-Dist get filtered before source selection. This fully replaces the
   script's `--prerelease=allow`.

Net: there is no capability the script provides that fence does not, once the
prerelease micro-table is present (it is).

## Deletion surface (precise)

### src/config.rs
- DELETE `BlueprintSync` enum (222-232) and the `blueprint_sync` field
  (204-205) + its doc (196-203). Fence is the only path; no selector remains.
- DELETE test `parses_retread_blueprint_sync_key` (652-669).
- Stale-key handling: confirm config struct does NOT use
  `deny_unknown_fields` (it relies on aliases, so almost certainly not). If so,
  a leftover `retread-blueprint-sync = "script"` is silently ignored and fence
  is used — acceptable. ADD: a one-line `tracing::warn!` at config-load if the
  raw key is present and == "script", telling the user the script path is gone
  and fence is now used. (Needs a captured raw value; if that costs a parse
  hook, downgrade to a README/CHANGELOG migration note only — see Open
  Decision 1.)

### src/emit_pypi.rs
- DELETE `render_install_script` (663-693) + its doc (657-662).
- DELETE `render_overrides_file` (697-708).
- emit() (1065-1093): DELETE the entire Script branch.
- `fence_wanted` (1101-1102) collapses to unconditionally true (blueprint-on OR
  non-blueprint emit-pypi both want fence). Simplify: always
  `sync_workspace_block`; the `else remove_workspace_block` arm becomes
  unreachable from emit(). Decide `remove_workspace_block` fate — Open
  Decision 2.
- DELETE the script-path module-doc lines and update the "Motivation" doc block
  (1-49) to drop the installer framing.
- DELETE tests asserting `render_install_script` / `render_overrides_file` /
  `--prerelease=allow` (~1303-1313).

### README.md
- Rewrite "emit-pypi side-channel" sync-modes section (151-177): remove the
  `"script"` (default) bullet + activation-hook guidance; describe fence as the
  single path.
- Line 174-176: `git clone && pixi install && install.sh` -> `git clone &&
  pixi lock && pixi install` (fence: re-lock then install; no script).
- Keep `retread-blueprint = "only"` description (167-169) unchanged.

### examples/
- grep clean today (no install.sh / blueprint-sync refs). examples/gigastrap
  fence block will be (re)written by the e2e run; verify no stale `[activation]`
  line referencing install.sh remains in any example pixi.toml.

## Validation gates (per HANDOFF-CLEANUP-PLAN convention; end green)
- cargo fmt; cargo clippy -D warnings (0); cargo test --lib.
- release-build protocol test --include-ignored.
- e2e on examples/gigastrap (gsi / blueprint env):
  - nuke the env + pixi caches (scripts/rebuild-local.sh CONSUMER_PROJECT=...).
  - build the pack; confirm NO `install.sh` / `overrides.txt` written under
    retread-pypi/<bundle>/.
  - confirm fence block auto-synced into workspace pixi.toml (static ~7 lines +
    prerelease micro-table if any prerelease-only floors).
  - `pixi lock` then `pixi install -e <env>`: lock references the 999retread
    build-tagged wheels + the meta-wheel; installed METADATA is rewritten;
    prerelease-only deps resolve; import/Kit-boot smoke test passes.
- Version bump: Cargo.toml + recipe/recipe.yaml to the SAME new version (v1.8.0
  — script removal is a behavior change). rebuild-local.sh aborts on mismatch.

## Migration note (CHANGELOG / README)
Existing consumers wired to the old script (a `[tasks]`/`[activation]` line
calling install.sh) must, after upgrading: (a) delete that hand-added line,
(b) let the next build auto-sync the fence block, (c) `pixi lock && pixi
install`. The script + overrides.txt stop being generated.

## REVISION 1 — the-grizzly review (APPROVE-WITH-CHANGES), findings VERIFIED

Grizzly found one real correctness BLOCKER. I verified it at the exact lines:

### BLOCKER A — prerelease gap (CONFIRMED, the headline claim was overstated)
The micro-table (`prerelease_pins`, 1015-1024) is built SOLELY from
`emit_plan.overrides`. Two prerelease classes never reach `overrides`, so the
script's blanket `--prerelease=allow` covered them and fence does NOT:
- **Shipped/injected wheels** — Pass 2 skips `ship.contains(&name)`
  (emit_pypi.rs:265). A git/path-built IsaacLab wheel with a `.dev`/`rc`
  version gets no floor row.
- **`[retread-wheels]` entries** — every entry name is removed from `overrides`
  (emit_pypi.rs:860-864) to protect its own pin. An entry pinned to a
  prerelease gets no row. The meta-wheel's `==X.Yrc` Requires-Dist does NOT
  opt uv into prereleases (uv builds its explicit-prerelease set only from
  direct requirements + overrides; meta-wheel Requires-Dist is filtered before
  source selection — the in-code proof at 624-648).

gigastrap's tinyobjloader is TRANSITIVE, so it survives into `overrides` and
the proof "works" — validating the lucky case, not the general one. A pack with
a prerelease entry or a prerelease-versioned built wheel would fail `pixi lock`
with "pre-releases not enabled."

FIX (small, surgical, in the emit() blueprint block ~1015-1024): build
`prerelease_pins` from THREE sources, unioned:
  (a) existing — `overrides` rows that are `==<prerelease>` (transitive case),
  (b) NEW — every shipped wheel whose own resolved version is a prerelease,
  (c) NEW — every `[retread-wheels]` entry whose pin is a prerelease.
Each emits one `"name" = "==<prerelease>"` row in
`[pypi-options.dependency-overrides]`. KEEP the overrides COMPUTATION
(`render_overrides_file` serializer goes away, the map does not).

### BLOCKER B — conda/CUDA sharing is unproven in fence mode
Standalone uv (the script) "prefers installed dists", so conda torch/CUDA stay
shared (README:161-162). retread builds the blueprint as a pure static function
of the bundle (never reads CONDA_PREFIX) — so nothing guarantees fence-mode
pixi won't try to reinstall torch from a wheel. Cannot be settled statically;
the gigastrap e2e gate MUST assert torch/CUDA are NOT reinstalled from a wheel
after `pixi install`. If they are, that's a real regression the script avoided.

### BLOCKER C — stale-lock guard (now unavoidable, was opt-in)
pixi's satisfiability string-compares the find-links PATH and is blind to
directory contents (BLUEPRINT.md:46-48). After a rebuild that changes wheel
bytes but not versions, a stale pixi.lock silently installs OLD/registry wheels
over the 999retread-tagged ones — reintroducing the exact import breakage the
blueprint exists to prevent. Comment-only is insufficient now that fence is the
ONLY path. Implement the `blueprint.lock` staleness marker (BLUEPRINT.md step
9: config hash + shipped-wheel set) + a loud "re-lock" log on every build.

## RESOLVED decisions (grizzly recs, adopted)
1. **Stale config key:** silent-ignore + one-line `tracing::warn!` when
   `retread-blueprint-sync` is present. NEVER hard-error (bricks existing
   builds). Verify no `deny_unknown_fields` on RetreadConfig first. Keep the
   field as a deprecated, ignored `Option<String>` for one release, then delete.
2. **remove_workspace_block:** RETAIN (tests exercise it; only mechanism that
   strips a stale fence). Don't delete as "dead"; `#[allow(dead_code)]` + TODO
   if truly unreachable, or wire the blueprint-OFF -> strip-fence path.
3. **Replace deleted tests:** keep `parses_*`/script tests deleted, but ADD a
   test asserting a shipped-prerelease AND an entry-prerelease each produce a
   `[pypi-options.dependency-overrides]` row (locks BLOCKER A fix).
4. **Migration:** CHANGELOG + a LOUD first-build log: existing consumers must
   remove the hand-added `install.sh` `[activation]`/`[tasks]` line, else
   `pixi shell`/`pixi run` errors `install.sh: No such file` (env-breaking).

## OPEN decision (needs Gary) — deletion phasing
Grizzly recommends keeping `Script` behind an UNDOCUMENTED escape hatch for one
release (flip default to fence in v1.8.0, delete Script code in v1.9.0) as
insurance until BLOCKER B (conda sharing) is proven green on a real pack.
Gary's stated intent is full removal now. My rec: do the prerelease fix +
stale-lock guard + e2e first; if the e2e proves conda sharing AND the
prerelease fix is green, full-delete in v1.8.0 (we've closed the only gaps).
If e2e shows torch double-install, fall back to phased (default-flip only).
=> Confirm: full-delete-now (gated on green e2e) vs phased (one-release hatch).

## Revised implementation order
0. prerelease-gap fix (BLOCKER A) + unit tests — FIRST, independent of deletion.
1. stale-lock guard (BLOCKER C) — blueprint.lock marker + loud relock log.
2. config: deprecate+ignore blueprint_sync, warn on stale key; delete enum
   later (per phasing decision).
3. emit(): delete Script branch, collapse fence_wanted to always-sync, RETAIN
   remove_workspace_block.
4. delete render_install_script + render_overrides_file (serializer) + script
   tests; keep overrides map computation.
5. README + CHANGELOG rewrite + migration log.
6. version bump (v1.8.0); rebuild-local.sh; gigastrap e2e incl. BLOCKER B
   assertion (no torch wheel reinstall) and a prerelease entry/shipped case.
