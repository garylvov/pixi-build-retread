# Courier architecture (v2.0.0) — design + build plan (scratch, never commit)

## ====================== SHIP-READY (2026-06-15) ======================
COURIER v2.0.0 IS SHIP-READY on branch `courier` (HEAD 378ca87, unpushed).
the-grizzly final verdict: SHIP-WITH-FOLLOWUPS, no blockers. All 5 reqs
validated by e2e (isaac6 full flow + gigastrap lock-level S1). Green at every
phase: cargo fmt, clippy -D warnings, 267 lib tests.

Phases done: 0 (freeze) -> WS-A/B/D (parallel bees) -> WS-C (wire) -> 4
(delete fence, T1) -> 5 (v2.0.0 + README) -> 6 (e2e) -> 7 (grizzly + B-1 fix).
Built with the ralph loop: architect plan -> implement -> grizzly review ->
swe correct, every phase.

The 5 requirements (all PASS, grizzly-verified against source):
 1. zero machine-written manifest bytes (isaac6 consumer pixi.toml byte-identical)
 2. no wheels in git (KB lock committed; shadows in the conda pkg; index fetched)
 3. cached relaxed state (committed retread-<bundle>.lock.json)
 4. fast cold solve (conda/outputs replays the lock, skips the cascade)
 5. reproducible (inputs_hash + idempotency marker)

FOLLOW-UPS (post-2.0.0, tracked; NONE block ship):
 - B-1 DONE (378ca87): replay uv dedup.
 - B-2: remote-only relax-changed wheels (botocore/cryptography/sympy/...) can't
   be shadowed (no local bytes) -> recorded Origin::Index with strict pins;
   warned; mitigated by the S1 --constraints + they're conda-capable. Latent
   risk: force-download to rewrite, or assert conda coverage.
 - B-3: post-link `|| echo` swallows install failure (broken env links quietly).
   Consider a <bundle>.install-failed sentinel a later `pixi run` detects.
 - B-4: 490M isaac6 conda pkg — all isaacsim subpkgs shadowed (relax changed
   intra-bundle pins). Trim intra-bundle shadows (ship original, find-links
   resolves siblings) to shrink.
 - B-5: run_override=None fallback re-derives conda_run_deps from requires_dist,
   which could poison the lock vs what pixi locked. Hard-require forwarded
   run-deps in courier mode.
 - sha256: LockWheel.sha256 unused (index hash rides the url#sha256 fragment);
   populate for explicit verification.
 - DEFERRED decisions for Gary: convert the committed gigastrap example to
   courier? publish 2.0.0 to prefix.dev/garylvov (external consumers need it).
   pixi upstream PR (CondaOutput.pypi_options) would let us drop post-link
   entirely someday (HANDOFF-PIXI-UPSTREAM-ISSUE.md).

Branches: `courier` (this work, unpushed) supersedes `kill-install-script`
(v1.8.0 fence, abandon). To ship: `git push` + PR `courier` -> main.
## =====================================================================


## >>> RALPH LOOP BACKLOG (autonomous, overnight) <<<
Branch `courier`. Per-step protocol for EACH phase below:
1. PLAN with the solution-architect (subagent solution-architect) — concrete
   file-level plan for that phase against current code.
2. IMPLEMENT it (directly or via a swe-worker-bee).
3. REVIEW with the-grizzly (adversarial; find blockers, false-positives ok to
   note). VERIFY grizzly's claims against source before acting (it over-calls).
4. CORRECT the real findings with a swe-worker-bee (or directly).
5. GREEN BAR (mandatory, every phase): `pixi exec --spec rust -- cargo fmt`,
   `... cargo clippy --all-targets -- -D warnings` (zero), `... cargo test --lib`.
   (No host cargo — always `pixi exec --spec rust -- cargo`.)
6. COMMIT on `courier` with a clear message. Update this file's DONE/REMAINING.
NEVER push. NEVER touch committed examples (e2e on /tmp copies only; fence sync
+ post-link write into the workspace). Convert relative dates to absolute.

DONE: Phase 0 (freeze), WS-A/B/D (parallel), WS-C (wire), integration fixes.
Courier proven end-to-end on /tmp/retread-synth (clean manifest, post-link
installs the prerelease wheel, idempotent). Commits up to d9e8e20.
DONE: Phase 4 (delete fence/blueprint injection, T1) — commit 05fbc6c.
  Grizzly APPROVE, no regressions; FINDING-C + prerelease preserved via the
  courier lock+installer. manylinux_floor was DELETED (zero callers; courier
  uses live-interp tags, no system-requirements block). 266 lib tests green.
DONE: Phase 5 (v2.0.0 + README + isaac6 config) — commit ab2ad0b. Version
  1.8.0->2.0.0 (Cargo.toml/recipe/Cargo.lock). README rewritten for courier
  (security note, cold-solve replay, --frozen honesty, migration). .gitignore
  fixed so .pixi/config.toml + retread-*.lock.json are committable. isaac6:
  retread-courier=true + committed .pixi/config.toml. Grizzly APPROVE-W-CHANGES;
  2 README inaccuracies fixed. NOTE: prefix.dev/garylvov 2.0.0 publish still
  pending (external consumers). gigastrap example conversion deferred to P6.

DONE: Phase 6 (e2e) — commits ff728c6 (replay fixes) + c05f5ad (isaac6 lock).
  isaac6 FULL courier e2e green: build, lock (nvidia idx, 67 wheels), ZERO
  manifest injection (consumer byte-identical), post-link installs isaacsim 6
  + imports, idempotent, B2 999retread shadows, cold-solve replay FIRES.
  Two replay bugs found+fixed: (1) inputs_hash divergence -> courier_input_specs
  shared by producer+replayer; (2) spec_from_str couldn't parse build strings
  (python_abi 3.12.* *_cp312) -> replay ERR'd. gigastrap lock-level S1 PASS
  (torch/torchaudio/torchvision conda-routed, none in wheel set) + FINDING-C
  index chain. ALL 5 REQS validated. gigastrap committed-example conversion
  still deferred (decision for Gary; e2e ran on /tmp). prefix.dev 2.0.0 publish
  still pending for external consumers.
DONE: Phase 7 (SHIP) — courier shipped 2026-06-16. Final state:
  - v2.0.0 -> v2.1.3. Replay bug fixed (2.1.2: producer/replayer hashed
    divergent inputs -> threaded pristine declared_config + manifest-derived
    courier_channel_set; empirically verified replay FIRES on a real multi-env
    workspace). v2.1.3: content-addressed shadow-rewrite cache (cheaper staging).
  - Cache VERIFIED on isaac6: warm add-dep staging 63s -> 38s (27 shadow wheels
    cache-hit). POISONING-FREE: cache output byte-identical to fresh rewrite
    (clean parity cache-on == cache-off on identical inputs); the only lock
    variation is pre-existing solver python_abi run-dep non-determinism, which
    reproduces with the cache fully disabled (RETREAD_NO_SHADOW_CACHE=1 twice
    -> differs the same way). Grizzly: "Ship it."
  - prefix.dev/garylvov: 2.1.3 PUBLISHED (supersedes the deferred 2.0.0; external
    consumers now get the latest). main: courier MERGED (72aa4c9) + pushed.
    CI green on main (pre-commit, musl-static, EMIT_EPOCH guard).
  - Lock-poisoning: audited + grizzly-certified NEVER (B-1..B-8, P1, H1).
    --excludes general install fix; URL-dep built-wheel rewrite; content-
    addressed conda build string (fixes pixi cache staleness).
  - pixidock_template (separate repo): migrated to courier API on
    dev/retread-update (5 commits ahead of main); main-merge strategy is
    Gary's call (PR vs fast-merge) -- NOT a courier-repo blocker.
  COURIER_SHIP_READY.

(superseded detail) PHASE 5b — version bump 1.8.0 -> 2.0.0 (Cargo.toml + recipe/recipe.yaml +
  Cargo.lock). README rewrite: clean one-line manifest, committed lock, the
  `.pixi/config.toml run-post-link-scripts="insecure"` toggle WITH a prominent
  security note (T2: clone auto-runs all post-links), the cold-clone flow, and
  an honest note that uv-installed site-packages are outside pixi.lock so
  `pixi install --frozen` won't restore them (T3). Add committed
  examples/isaac6/.pixi/config.toml. Migration note from fence/v1.8.
- PHASE 6 — e2e on isaac6 (cp312, single isaacsim entry) then gigastrap gsi.
  Switch the pack to retread-courier=true; consumer = clean one-line +
  .pixi/config.toml. Build channel (rebuild-local.sh), pixi install, assert:
  no manifest injection, conda pkg small, post-link installs, imports/torch
  stays conda (S1), cold-solve replay fires (probe-trace absent on 2nd build).
  HEAVY (multi-GB/GPU) — /tmp copies only.
- PHASE 7 — final the-grizzly review of the whole courier (all reqs #1-#5),
  fix blockers via swe, green, and write a SHIP-READY summary in this file.

Deferred (note, don't block 2.0.0): content store for built wheels; sha256
verification in installer; pixi upstream PR (HANDOFF-PIXI-UPSTREAM-ISSUE.md).
## <<< END RALPH LOOP BACKLOG >>>


Decided with Gary after rejecting fence (manifest injection) + committing
wheels, and after a pixi source audit proved NO native backend->uv pypi
channel exists (see HANDOFF-PIXI-UPSTREAM-ISSUE.md).

## The idea: conda package = COURIER, not container
The pack stays ONE clean hand-authored conda line in the consumer manifest
(`isaac-pack = { path = "./isaac-pack" }`). retread NEVER writes the consumer
pixi.toml. The conda package carries metadata + a committed lock + a post-link
hook -- NOT the 25GB wheel payload. At env link time the post-link runs
`retread install`, which uv-hardlinks the wheels in (built/shadow wheels ship
in the package; index wheels fetch from their recorded URLs).

Satisfies Gary's 5 hard reqs: (1) zero manifest bytes, (2) no wheels in git
(index wheels fetched; built/shadow ship in the conda pkg, built from source),
(3) committed `retread-<bundle>.lock.json` caches relaxed state (KB, no wheels),
(4) fast install via uv hardlink [cold SOLVE replay = a later phase],
(5) reproducible via the lock.

Enabling cost (NOT manifest): commit `.pixi/config.toml` with
`run-post-link-scripts = "insecure"` (pixi un-ignores config.toml via its
.pixi/.gitignore `*` + `!config.toml`). Empirically proven this session:
post-link runs for path-source packages, can uv-install at link time.

## DONE (branch `courier`)
- Phase 1: src/lock.rs (RetreadLock) + write in build_one. commit 2034073.
- Phase 2: src/installer.rs + `retread install` subcommand. commit 2034073.
- Phase 3a: src/recipe.rs `build_courier_recipe` (additive, tested). THIS commit.

## WEEK AUDIT (grizzly, 2026-06-14) — validated findings folding into 3b
- B1 (BLOCKER): root_requirements always empty -> installer inert. Fix in 3b.
- B2 (BLOCKER, biggest risk): must_ship() recognizes only `.injected`, so a
  relax-CHANGED index wheel (`.relaxed.`) is mis-tagged Index/fetch-upstream ->
  consumer silently re-gets STRICT pins (the exact breakage the project
  prevents). FIX: classify origin by the rewrite `changed` flag (ship changed
  index wheels as shadows), not just `.injected`.
- B3 (BLOCKER): build_courier_recipe is dead code; no config flag; build_one
  still calls build_bundle_recipe + fence. Wire in 3b.
- B4: FALSE POSITIVE (verified: Rust `\` continuation eats the heredoc
  indentation; terminator is at col 0). Locked with an assertion.
- S1 (SHOULD): conda run-deps + uv install = double-install/skew; the
  torchaudio skew (HANDOFF-KILL-SCRIPT) RECURS unless the installer prefers
  conda-installed dists. FIX: installer constrains uv to the prefix's
  installed set (or --no-deps for conda-covered names).
- S3 (SHOULD): record the FULL ordered index chain in the lock + replay
  verbatim; don't hard-code public PyPI as primary (misses private mirrors).
- S4: add installer + e2e tests (only roundtrip + recipe-shape exist).
- T1 (STRATEGIC): courier branch STILL runs fence (sync_workspace_block) ->
  violates the clean-manifest thesis NOW. 3b MUST delete the fence sync.
- T2: post-link "insecure" is supply-chain-shaped (clone auto-runs all
  post-links). Needs a prominent consumer warning (Phase 5).
- T3: cold-SOLVE replay NOT done -> req #4 only partial (fast install, not
  fast solve). Also uv-installed site-packages are outside pixi.lock ->
  `pixi install --frozen` won't restore + reconcile may prune. Be honest in
  docs; verify prune behavior in Phase 6.
- T4: DONE reclaim /tmp/gs-e2e (42GB). Version bump 1.8.0->2.0.0 at Phase 5.

## REMAINING — Phase 3b wiring (the build_one restructure)
The hard integration. In build_one, for the courier path:
1. Run the wheel staging (REUSE emit_pypi::emit's machinery: built wheels +
   shadow wheels for relax-CHANGED index wheels + the `<bundle>-pypi`
   meta-wheel) into a staging dir, but DO NOT write the fence block
   (suppress sync_workspace_block). This also fixes Phase 1's known TODO
   (relax-changed index wheels must ship, not fetch upstream).
2. Lock: set `root_requirements = ["<bundle>-pypi==<version>"]` (the
   meta-wheel drives the closure) and populate `prerelease` via
   collect_prerelease_pins. Mark shipped wheels Origin::Built (in pkg),
   the rest Origin::Index (fetch). Write the lock into the staging dir.
3. Build the courier recipe via build_courier_recipe(conda_name, version,
   python, solved_run_deps, source_urls=[staged wheels.. + lock]). NOTE
   ordering: staging+lock must happen BEFORE the recipe (today emit runs
   AFTER recipe; reorder for courier).
4. rattler-build as today (small pkg now).
Gate the whole thing on a config flag (e.g. retread-courier or reuse
blueprint="only" semantics). Keep the conda-artifact + fence paths working
until Phase 6 is green, THEN Phase 4 deletes fence.

## Phase 4: delete fence/blueprint manifest injection (after courier proven)
emit_pypi.rs: drop sync_workspace_block/remove_workspace_block/
render_snippet_blueprint/fence; KEEP build_meta_wheel + shadow-wheel rewrite +
collect_prerelease_pins (reused by courier staging). config: deprecate
retread-blueprint(-sync)/emit-pypi keys (parse+warn+ignore).

## Phase 5: committable .pixi/config.toml + README
Add committed .pixi/config.toml (run-post-link-scripts="insecure") to the
example(s) + README rewrite (one clean line, committed lock, post-link toggle
with the security note, cold-clone flow). Bump version 1.8.0 -> 2.0.0
(Cargo.toml, recipe, Cargo.lock).

## Phase 6: e2e (synthetic -> isaac6 -> gigastrap gsi)
clean manifest only -> pixi install -> post-link installs -> imports work.
Assert: no manifest injection, conda pkg small, idempotent re-install,
lock committable, torch-family stays conda.

## Deferred follow-ups (not blocking courier ship)
- Cold-SOLVE replay: conda/outputs replays the lock to skip the cascade
  (fast cold solve). Big win for req #4 but independent; layer on later.
- Content store for built wheels (so cold clone doesn't rebuild from git).
- sha256 in lock for index-wheel verification.

## Branch state
- `kill-install-script` (v1.8.0 fence): PARKED, unpushed. Superseded by
  courier. Salvaged: prerelease fix, FINDING-C, stale-lock idea, meta-wheel.
  Decide at Phase 4 whether to abandon or cherry-pick docs.
- `courier`: active.
