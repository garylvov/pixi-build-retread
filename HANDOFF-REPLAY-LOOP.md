# HANDOFF: General Lockfile Replay (Ralph Loop)

Single source of truth for the autonomous loop. Read this FIRST every iteration.
Branch: `courier`. Repo: `/home/garylvov/projects/pixi-build-retread`.
Consumer for e2e: `/home/garylvov/projects/pixidock_template` (branch `dev/retread-update`).

## THE GOAL (precise, non-negotiable)
On a FRESH AWS instance — committed lock present in git, but NO wheels stored and
ALL caches empty — a `pixi install` of any pixidock pack MUST:
  1. NOT run the resolver/cascade (no BFS sub-wheel resolution, no auto-bundle
     decision, no resolvo solve, no PyPI probing). Reconstruct the emit set
     entirely from the committed `retread-<bundle>.lock.json`.
  2. Re-MATERIALIZE the wheel bytes from provenance the lock carries (re-fetch
     index wheels by URL; re-source-build git/path wheels from a pinned rev) —
     because no wheels are stored. Materialization is inherent + allowed; only
     DERIVATION (solve) must be skipped.
  3. Produce a BYTE-IDENTICAL lock (`git diff --exit-code` clean).
GENERAL: must work for ALL packs/envs in pixidock_template (isaac-pack,
genesis-pack, newton-pack-latest, and every env that references them). NO band-aids,
NO per-pack special cases — deep root-cause fixes only.

SECONDARY GOAL (same loop, after primary): incremental single-dep add. Adding ONE
dep to a pack should reuse the existing locked closure and resolve only the delta
(the new dep + its new transitives), not re-resolve everything. Research whether
resolvo/rattler supports warm-start from the locked pins, or do it at retread level
(seed the solve with locked versions as preferences). Must be general + correct
(no stale closure). This is a real optimization, not a shortcut.

## PER-ITERATION PROTOCOL (user-mandated; follow EXACTLY, one coherent phase per iteration)
1. solution-architect: plan the next phase (or revise after grizzly).
2. the-grizzly: review the PLAN. Find holes BEFORE code.
3. solution-architect: revise the plan per grizzly findings.
4. swe-worker-bee: implement the revised plan. Keep the green bar at each commit.
5. the-grizzly: AUDIT the implementation (especially poisoning + lukewarm correctness).
6. swe-worker-bee: fix the grizzly's findings.
7. Keep `cargo test --lib` + `cargo clippy --all-targets -- -D warnings` + `cargo fmt`
   green. Commit on `courier`. Update DONE/REMAINING below. Then loop.
Use the project toolchain ONLY: `PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH" cargo ...` (NO host cargo). pixi at `~/.pixi/bin/pixi`.

## VERIFICATION STANDARD (anti-cheat — the user's #1 concern)
A "lukewarm" box = committed lock present, EVERYTHING else cold. Before each measured
replay run, nuke ALL warm state:
```bash
rm -rf ~/.cache/uv ~/.cache/rattler ~/.cache/retread \
       <pixidock>/.pixi/envs <pixidock>/.pixi/artifacts-v0/* \
       ~/.cache/rattler/cache/backends-v0/pixi-build-retread-* \
       <pack>/wheels
```
CRITICAL: do NOT delete `<pixidock>/.pixi/config.toml` (committed; carries
`run-post-link-scripts = "insecure"`). Deleting it skips the post-link installer and
the env import fails — that is a TEST ARTIFACT, not a backend bug. Restore/keep it.
Backend MUST be the local build: rebuild via `bash scripts/rebuild-local.sh` (the
isaac6 example pins `version="*"` with `file:///.../local-channel` FIRST, so it picks
the local build; pixidock pins `>=2.2.0` from prefix.dev — for pixidock e2e either
publish the new version or add the local-channel to its pack backend channels).

Replay-fires assertions (ALL must hold, on an all-caches-nuked run):
  - `build_v1: replayed from lock` log PRESENT.
  - NO derivation logs: zero `auto-bundled`, zero `resolvo solve finished`,
    zero per-env solve-check, no probe-trace file. (NOTE: `applying relax policy`
    MAY appear — that is Phase-2 relax during MATERIALIZATION, which is correct.)
  - `<pack>/wheels` repopulated from EMPTY (proves materialization ran).
  - `git diff --exit-code <pack>/retread-*.lock.json` CLEAN (byte-identical;
    zero python_abi tolerance — replay skips the solve, so no solver noise).
  - env imports (e.g. `python -c "import isaacsim; import isaaclab"`).
Run for MULTIPLE packs (isaac + genesis + newton), not just one. Replay must FIRE
(not fall through) for every pack, or the goal is not met.

## ACCUMULATED FINDINGS (verified — do NOT re-derive; build on these)
- **Lock stores NO wheel `sha256`** (built or index; push sites write `sha256:None`).
  => Lock parity is INDEPENDENT of wheel-byte reproducibility. Source-build
  determinism (SOURCE_DATE_EPOCH / unconditional repack) is NOT required for the
  byte-identical-LOCK goal. (It would only matter for `.conda` artifact repro — a
  separate, pre-existing, OUT-OF-SCOPE concern.) Do not rabbit-hole on it.
- **isaac6's 22 built wheels are `isaacsim-*`** = the `isaacsim[all,extscache]`
  extras expansion fetched from `https://pypi.nvidia.com` (INDEX wheels, `must_ship`,
  NOT git; isaac6 has no `[retread-git-sources]`). Today they're recorded
  `Origin::Built, url:None` => hit the Class-3 fall-through => replay INERT.
  FIX: preserve their upstream INDEX URL in the lock; on replay re-fetch from URL +
  ship. Re-fetching immutable nvidia wheels is deterministic. No git machinery needed
  for isaac.
- **newton-pack** = git-source-built (`newton = { from="newton" }` +
  `[retread-git-sources].newton` git url+rev). Built wheel(s) re-materialize by
  source-build from a PINNED rev. newton's OWN rev IS folded into `inputs_hash`
  (config entry via `courier_input_specs`). Verify whether newton[sim] pulls
  `@ git+...` SUB-deps (BFS transitives) whose revs are NOT hashed (poisoning #5).
- **#4 parity (grizzly-confirmed REQUIRED + sufficient):** `plan()` (emit_pypi.rs
  ~191-292, Pass1 203-232, Pass2 236-263, NO origin filter) builds the override
  table from EVERY wheel's `requires_dist`. Producer writes `requires_dist:[]` for
  unchanged `Origin::Index` wheels (courier.rs ~688/712) => on replay the table is a
  SUBSET => a relax-shadow can flip to Index + ship its original strict pin
  (POISONING) + non-byte-identical lock. FIX: store EVERY wheel's real
  `requires_dist` (incl unchanged index); Class-4 reconstruction copies it through.
  `plan()` is a pure fn of (requires_dist, version, must_ship()/filename,
  conda_capable) — all in the lock after this. Do NOT store the table itself.
- **WheelSource provenance design** (per-wheel, schema bump): enum to re-materialize
  each Built wheel WITHOUT the solve. Must cover EVERY way a Built wheel arises:
    - Config-entry index wheel (isaacsim): Index{url}.
    - Index sub-component / extras expansion (isaacsim-*): Index{url} — PRESERVE URL.
    - Relax-changed index shadow: Index{url} (re-fetch + re-relax).
    - Git config-entry (newton): Git{url, rev(SHA), subdirectory, extras}.
    - Git BFS transitive (rl-games/rsl-rl-style `@git+...`): Git{...} — needs A-0
      `#subdirectory=` parse fix (auto_bundle.rs ~688-710 currently corrupts rev to
      `rev#subdirectory=...`) + HEAD->resolved-SHA (source_build.rs cache key also
      keys on literal rev => stale-checkout hazard; key on SHA).
    - sdist fallback (`build_wheel_from_sdist_url`, mod.rs ~3116): Sdist{index/url,
      name, specifier} — OR prove empirically no pack hits it (isaac: 0 sdist).
  Enumerate EVERY `materialize_and_rewrite` call site + every PendingSource/
  ExtraDepSource variant; map each to a WheelSource; NO uncovered `Built+None`
  fall-through for any class a real pack produces. Inline the descriptor so replay
  does NOT depend on the manifest entry still existing.
- **Poisoning (#5):** any source rev a replay trusts must be invalidated when it
  changes. Config-entry git revs ARE in `inputs_hash`. BFS-transitive git revs are
  NOT. Either fold resolved transitive SHAs into `inputs_hash`, or pin+document that
  replay reproduces recorded commits and only the cascade re-resolves them.
- **load_replayable_lock** = the SOLE authority gate (schema==SCHEMA && inputs_hash
  match && not(non-default-relax && any empty requires_dist); RETREAD_NO_REPLAY=1
  forces None; never feeds inputs_hash). Keep. **assemble_conda_output** unifies the
  produce_output/replay metadata tail (fixed the noarch dup). **materialize_and_pack**
  = shared stage->rattler-build->lock-flush tail. Keep these.
- **EMIT_EPOCH:** lock-FIELD additions are emit-neutral (not in compute_inputs_hash)
  => `[emit-epoch-ok]`. ALGORITHM changes (A-0 subdir parse; any change to plan()/
  relax/auto_bundle/recipe/courier/lock per the CI guard regex) are emit-affecting
  => bump EMIT_EPOCH. Currently 3. Old-schema locks fall through cleanly via the
  schema gate.

## CURRENT STATE (commits on `courier`, v2.4.0, schema 6)
- b8bbe22 assemble_conda_output (unify metadata tail) — SOUND.
- 05a82f2 schema 4->5: LockWheel.requires_dist/must_ship + RetreadLock.conda_capable.
- fb4e3ae load_replayable_lock (sole gate) — SOUND.
- 48942b9 materialize_and_pack (shared tail) — SOUND.
- a1b451e build_v1 gate + materialize_from_lock — BUGGY (assumed wheels on disk).
- 327ae3a fix: 4-class router + schema 6 upstream_url. Class 1/2/4 re-materialize;
  Class 3 (git transitive + index-sub-component) FALLS THROUGH => replay INERT for
  isaac/newton. Grizzly BLOCK: (a) #4 parity not fixed (index requires_dist empty);
  (b) coverage holes (index sub-components recorded url:None; sdist; BFS-URL).
EMPIRICAL e2e: isaac6 lukewarm => `build_v1: replayed from lock`=0, Class-3
fall-through, full re-derive ran. NOT yet working.

## DONE
- (refactors) assemble_conda_output, load_replayable_lock, materialize_and_pack.
- Verified: lock has no wheel sha (determinism not needed for lock parity).
- CORRECTED DIAGNOSIS (both architects, vs committed lock): isaac6's 22 "built"
  wheels are NOT Class-3 must_ship git transitives. They are Class-2 relax-changed
  INDEX SHADOWS (must_ship=FALSE, 999retread build-tag, from pypi.nvidia.com). They
  fall through at the CLASS-2 branch (mod.rs:4066-4102) because upstream_url is None:
  the producer DISCARDS the upstream index URL when the wheel is localized to file://
  (download_dir==wheels_root, mod.rs:1987==4327; localize collapses it at ~4330;
  EmitWheel build at 4360 then sets remote_url=None). 45 index wheels have
  requires_dist=[] (the #4 parity bug). isaac6 has ZERO must_ship/git wheels.
- PHASE 1 PLAN READY (architect): thread upstream index URL ResolvedWheel(mod.rs:461)
  -> EmitWheel(emit_pypi.rs:59) -> LockWheel.upstream_url, INDEPENDENT of localization
  (read pre-localization w.url at mod.rs:4360); producer writes real requires_dist for
  unchanged index wheels (courier.rs:712) + upstream_url for shadows (courier.rs:688
  prefer w.upstream_url over remote_url); Class-2/4 reconstruction already consumes
  these (no logic change, just feed data); schema 6->7 (old locks fall through);
  [emit-epoch-ok] (lock-content change, not in compute_inputs_hash, emit-neutral);
  tests = plan() purity + empty-wheels byte-identical parity (red->green); isaac6
  lukewarm e2e. 6 commits. plan() confirmed pure fn of (requires_dist, version,
  must_ship/filename, conda_capable) -- nothing else.

## PHASE 1 STATUS (index-wheel replay) — IMPLEMENTED, pending audit+e2e
- Architect plan -> grizzly review (REJECT, amendments) -> architect revise -> swe-bee
  implement DONE. Full plan in PHASE1-PLAN.md.
- Commits on courier 4d27781..a24b92e: (1) ShadowSrc enum fix for the force-download
  rename-vs-rewrite bug; (2) EmitWheel.upstream_url from pristine w.url; (3) producer
  writes upstream_url for local-path shadows + real requires_dist for index wheels;
  (4) replay local_path/conda_capable + ship sort; (5) schema 6->7, EMIT_EPOCH 3->4;
  (6) localhost-fixture byte-identical parity test (red-pre/green-post proven).
- 319 lib tests green, clippy + fmt clean.
- ESCALATION to re-check in audit: Class-2 shadow local_path is NOT reconstructable
  from lock alone (lock stores shadow filename, not original pre-relax name). Bee says
  non-blocking because index wheels are never URL-requirement targets so plan()'s
  local_path check is irrelevant for them. GRIZZLY MUST VERIFY this claim.
- NEXT (next iteration): (a) the-grizzly AUDITS commits 4d27781..a24b92e (Step-0
  byte-identity, upstream_url pristine read, #4 parity, the local_path escalation,
  EMIT_EPOCH 3->4 correctness, parity-test validity, poisoning); (b) swe-bee fixes
  findings; (c) ORCHESTRATOR runs the isaac6 lukewarm e2e: FIRST regen the committed
  schema-4 lock via a cold produce under the new backend (rebuild-local.sh) -> commit
  schema-7 lock; THEN lukewarm (nuke all caches incl wheels/, KEEP .pixi/config.toml)
  -> assert build_v1 replayed log present, no auto-bundled/resolvo-solve/probe-trace,
  wheels/ repopulated from empty, git diff --exit-code clean on lock, import isaacsim.
  Step 6 was NOT run by the bee (needs backend rebuild + nvidia fetch).
- GRIZZLY AUDIT DONE (commit-level): SHIP-WITH-CHANGES. Both make-or-break PASS:
  Step-0 byte-identity holds (single deterministic rewrite_wheel_with, pinned 1980 zip
  ts); the local_path escalation is VERIFIED TRUE from code (plan() reads local_path only
  at the direct-URL ship-set insert, emit_pypi.rs:~227; index shadows are never URL
  targets, so replay local_path:None can't drift the lock). #4 parity, conda_capable
  merged-set, pristine w.url read, EMIT_EPOCH 3->4, schema-7 gate, parity test all PASS.
- FIX-2 + FIX-3 DONE (commit bb915aa, v2.5.0): version bump 2.4.0->2.5.0; invariant
  comment + debug_assert(target.remote_url.is_none()) at the plan() URL-target ship-gate.
  319 tests green, clippy+fmt clean.
- FIX-1 (e2e/acceptance) IN PROGRESS: rebuild 2.5.0 + cold-produce isaac6 schema-7 lock +
  lukewarm replay. On PASS (replay log present, no derivation, wheels repopulated from
  empty, lock byte-identical, import isaacsim): COMMIT the regenerated schema-7 isaac6
  lock. On FAIL: bee fixes -> grizzly re-audit. Phase 1 (isaac index-wheel replay) is
  then DONE; Phase 2 = genesis + newton (git-source provenance), then incremental-add.

## PHASE 1 e2e RESULT (FAILED — fix in progress)
isaac6 lukewarm e2e: cold produce wrote schema-7 lock (21/22 shadows w/ upstream_url,
28/45 index w/ requires_dist). Lukewarm: build_v1 replay STARTED ("WS-B build_v1 replay
hit: re-materializing from lock (resolve_all skipped)") but FELL THROUGH on the TOP-LEVEL
`isaacsim` wheel: "relax-changed Built wheel has no upstream_url; falling through to full
resolve". -> full re-derive -> python_abi drift -> lock NOT byte-identical.
ROOT CAUSE (precise): Phase-1 derived EmitWheel.upstream_url from the pristine w.url at the
EmitWheel build (mod.rs:~4360). That works for BFS sub-wheels (w.url pristine) but NOT the
PRIMARY config-entry wheel (isaacsim): it goes through materialize_and_rewrite which
LOCALIZES w.url to file:// before the EmitWheel build, so (w.url.scheme()!="file") is false
-> upstream_url=None. materialize_from_lock is all-or-nothing (Ok(None) on first
unreconstructable wheel -> abandons whole replay -> full resolve_all).
FIX (general, root-cause, converges w/ Phase-2): add ResolvedWheel.upstream_url; populate it
at the FETCH site inside materialize_and_rewrite (BEFORE localization) for index wheels
(covers primary entry + sub-wheels) AND at the BFS site (= sub_url); change EmitWheel.upstream_url
to read w.upstream_url (the field) instead of re-deriving from the localized w.url. Then ALL 22
shadows carry upstream_url -> no fall-through -> build_v1 replay fully fires.
NOTE: import test in the e2e script was a shell-quoting bug (false failure), not real. Fix the
nested python -c quoting in the e2e harness.
FIX DONE (commit 9882812): added ResolvedWheel.upstream_url, populated at the materialize_and_rewrite
fetch site (before localization) for index wheels (primary entry + sub) + BFS + auto_bundle +
cascade; EmitWheel.upstream_url now reads w.upstream_url (not the localized w.url). 321 tests green,
clippy+fmt clean, no hash/schema/epoch change. Regression test added for the primary-entry case.
e2e run 2 (9882812): replay FIRED (build_v1 hit=2, fall-through=0, auto-bundled=0, solve=0,
probe=0, wheels repopulated from empty 8G) -- upstream_url fix WORKED. But 2 issues:
(1) lock NOT byte-identical: warm 62 conda_run_deps vs cold 61 (extra python_abi) -- build_v1
replay sourced run_deps from params.run_dependencies (pixi solve non-determinism) not the lock;
(2) import isaacsim FAIL = TEST ARTIFACT (.pixi/config.toml was deleted -> post-link skipped).
FIX #1 DONE (commit 93db53c): build_v1 replay gate now sources run_deps from lock.conda_run_deps
(serialized name+spec), removed the run_override==None hard-fail on the replay path; cold path
unchanged. 322 tests green. FIX #2: git-restored examples/isaac6/.pixi/config.toml (committed but
had been deleted); e2e NUKE now re-checks-out config.toml before each install.
RE-RUNNING e2e (bjis5tc7i) to verify: replay fires + LOCK BYTE-IDENTICAL (python_abi drift gone)
+ import isaacsim OK. If PASS: commit schema-7 isaac6 lock -> PHASE 1 DONE -> Phase 2 (genesis+
newton Class-1 git; PHASE2-PLAN.md ready, grizzly APPROVE-WITH-CHANGES: downgrade determinism
claim E to conditional + add setuptools_scm date-suffix guard + git fetch --tags; SPLIT A-0's
EMIT_EPOCH bump out since no real pack has git transitives; line-nums stale +~30-45; committed
locks are schema 4 not 7 -> migration is 4->8). If FAIL: re-diagnose.
PHASE 2 PLAN READY (architect, in PHASE2-PLAN section / agent a5f313fd) but BLOCKED on Phase 1
passing. KEY P2 finding: genesis-world + newton are Class-1 CONFIG-ENTRY git wheels (genesis
inline git, newton named git); ZERO git transitives in either pack. So P2 = make Class-1 git
replay manifest-INDEPENDENT (inline git_source{url,resolved-SHA,subdir,extras} in the lock) +
HEAD->SHA + A-0 subdir parse fix (latent, emit-affecting, EMIT_EPOCH bump) + regen all locks.
No Class-3/sdist machinery needed for the real packs (document as residual fall-through).

## PHASE 1 (isaac index-wheel replay) — DONE + EMPIRICALLY VERIFIED (commit c4df025)
Lukewarm e2e (all caches incl wheels/ nuked, lock present, config.toml present):
  build_v1 replay hit=1, fall-through=0, auto-bundled=0, resolvo-solve=0, probe-trace=0;
  wheels/ repopulated from EMPTY (8GB); warm conda_run_deps=61==cold 61;
  *** LOCK BYTE-IDENTICAL: YES ***; isaacsim installs + imports (OMNI_KIT_ACCEPT_EULA=YES;
  the bare-import "FAIL" was only the interactive Omniverse EULA prompt, not a replay defect).
Fixes that got it there: 9882812 (ResolvedWheel.upstream_url at fetch site, covers primary
config-entry index wheel) + 93db53c (build_v1 replay run_deps from lock.conda_run_deps, not
pixi-forwarded params -> kills python_abi drift). Schema-7 isaac6 lock committed.

## PHASE 2 (genesis + newton git-source replay) — PLAN READY (PHASE2-PLAN.md, grizzly
APPROVE-WITH-CHANGES folded in). Both packs are CLASS-1 config-entry git (genesis inline,
newton named), ZERO git transitives, single checkout root. Work: GitWheelSource{url,
resolved-SHA, subdir, extras} on LockWheel/ResolvedWheel/EmitWheel; Class-1 replay reads
lock.git_source (manifest-independent synth WheelEntry -> materialize_and_rewrite); HEAD->SHA
via git rev-parse; setuptools_scm date-suffix GUARD (claim E is conditional); A-0 subdir parse
fix WITHOUT epoch bump (no-op for real packs); single-entry guard; schema 7->8. NEXT: swe-bee
IMPLEMENTS PHASE2-PLAN.md (commits per its ordering) -> grizzly audits -> bee fixes -> lukewarm
e2e for genesis AND newton (replay fires + byte-identical lock + import). Then Phase 3.

## PHASE 2 (genesis+newton git-source replay) — IMPLEMENTED + GRIZZLY-AUDITED + FIXED; e2e running
Commits 11eb6e9..e17889c (v2.6.0, schema 8, EMIT_EPOCH 4): GitWheelSource{url,resolved-SHA,
subdir,extras} on LockWheel/ResolvedWheel/EmitWheel; build_wheel_from_git returns resolved SHA
+ determinism guard (warn on .dev/.d<date>/+g drift) + git fetch --tags; A-0 #subdirectory=
parse fix (no epoch bump, no-op for real packs); Class-1 git replay reads lock.git_source
(manifest-INDEPENDENT synth WheelEntry -> materialize_and_rewrite); single-entry guard; poisoning
docs. Grizzly audit: SHIP-WITH-CHANGES (named-vs-inline byte-identity PASS, SHA/guard/poisoning/
fall-through PASS). Fixes (e17889c): FIX-1 replay skip_subdirs now mirrors produce's
[subdirectory] (was [] -> latent drift for nested subdirs; nested-subdir regression test added);
FIX-3 version 2.6.0; FIX-4 parity test drives the real materialize_and_rewrite path. 333 tests green.
RUNNING genesis+newton lukewarm e2e (b4xsrqchw): local-channel prepended to both pack backends
(temp, reverted after), cold produce -> schema-8 locks w/ git_source for genesis-world + newton,
lukewarm replay -> assert build_v1 replay fires + no derivation + wheels from empty + byte-identical
lock + import genesis/newton. If PASS: Phase 2 DONE; then PUBLISH v2.6.0 to prefix.dev + commit
schema-8 genesis/newton locks to pixidock + bump pixidock backend pin to >=2.6.0 (schema-8 lock
needs a 2.6.0 backend to replay; a 2.5.0/schema-7 backend would reject it). Then Phase 3
(incremental single-dep add). If FAIL: re-diagnose.

## PHASE 2 (genesis + newton git-source replay) — DONE + EMPIRICALLY VERIFIED
Lukewarm e2e (all caches incl wheels/ nuked, local-channel v2.6.0 backend):
  genesis-pack (genesis-gpu): build_v1 replay hit=1, fall-through=0, derivation=0, wheels
    repopulated from EMPTY (493M, git-source rebuilt), LOCK BYTE-IDENTICAL=YES, import genesis OK.
  newton-pack-latest (newton-gpu): replay hit=1, fall-through=0, derivation=0, wheels 36M (git
    rebuilt from empty), LOCK BYTE-IDENTICAL=YES, import newton OK.
Both schema-8 locks carry git_source with resolved SHA (genesis-world 8de7e456, newton ce11136b).
Named-git (newton) -> inline synth collapse verified byte-identical. CORE CRITERION MET: isaac AND
genesis AND newton ALL replay byte-identically on a lukewarm box.
WRAP-UP DONE: v2.6.0 published to prefix.dev; courier->main merged, CI GREEN (main 9f14ab0);
pixidock schema-8 genesis+newton locks committed + pin >=2.6.0 (pixidock cfd6864). CI fix: 3 live
git-build tests (build_wheel_from_git via uv) marked #[ignore] (CI runner has no uv) + emit-ok ack.
*** PRIMARY GOAL DELIVERED: general lockfile replay (index + git-source) byte-identical for isaac
+ genesis + newton on a lukewarm box, shipped + CI green. ***

## PHASE 3 (incremental single-dep add) — GRIZZLY REVIEW: delta-resolve is UNSOUND as specified;
do NOT build it (would be a band-aid). Decisive findings (PHASE3-PLAN.md reviewed):
- retread's PyPI resolver = order-dependent FIRST-REQUIRER-WINS BFS, NO constraint accumulation
  (seed_worklist drops edges whose name is in `seen`, auto_bundle.rs:624/641). So "resolve only the
  delta + reuse the rest" can ship a STALE CLOSURE: a new transitive can back-pressure an existing
  pin via an edge the 3-part check can't see, AND version selection is order-dependent so constraint
  satisfaction != version-equality with a full cold resolve. Counterexample exists where the check
  PASSES but merged != cold. Violates the user's #1 no-stale-closure constraint.
- The lock is INSERTION/BFS-discovery-ordered (wheels = discovery order, conda_run_deps = emit order;
  only conda_capable is sorted). A full cold resolve interleaves the new dep's wheels differently than
  "append C(D)" -> byte-identical lock UNACHIEVABLE without canonical lock ordering.
- A SOUND delta-resolve needs a FOUNDATIONAL rework: canonical lock ordering + persisted edge graph
  (requires_dist IS now in schema-8 locks, a start) + order-aware version-equality simulation. Big
  project for a SECONDS-SCALE win (materialization/downloads dominate, not the solve). Not worth it.
- Honest conclusion (deep root cause, no band-aid): add-a-dep = cold solve is the SUPPORTED behavior;
  it is already mitigated by the now-working REPLAY for unchanged inputs (only the changed pack
  re-resolves; a single added dep changes only that pack's inputs_hash). The grizzly-endorsed SOUND,
  contained win is the resolvo locked_packages warm-start (src/solve_check.rs:111, fill from
  lock.conda_run_deps) -> faster + more stable conda solve-check on EVERY cold solve. Build that.
PHASE 3 CONCLUSION (warm-start built as infra, commit d19a71b, emit-neutral, 332 tests): the
solve-check now accepts a `preferred` (locked_packages, soft) seed, BUT the cascade call sites hold
spec STRINGS not resolved RepoDataRecords, and the lock's resolved deps are only reachable on the
replay-HIT path which skips the cascade -> there is no clean place to seed it without re-deriving.
So even the warm-start yields no realizable win at the relevant call sites; it's left as a documented
future seam. NET PHASE-3 FINDING (deep root cause): retread's order-dependent first-requirer-wins
BFS + insertion-ordered lock means a SOUND incremental delta-resolve is not achievable without a
foundational rework (canonical lock ordering + persisted edge graph + order-aware resolution) for a
seconds-scale win. The SUPPORTED behavior is add-a-dep = cold solve, MITIGATED by the now-working
replay: a single added dep changes only that pack's inputs_hash, so unchanged packs/envs still replay.

## PHASE 2.5 NEEDED (empirically confirmed) — multi-entry shared-git-checkout replay
isaac-pack regen-to-schema-8 + lukewarm replay HARD-ERRORED (warm exit 1): "courier replay: wheel
`isaaclab-assets` shares a git checkout root with a prior wheel (multi-entry shared-checkout bundles
are [not supported])". The single-entry guard (e17889c) fails by ERRORING, not falling through ->
a schema-8 isaac-pack would BREAK the build on replay. Cold produce wrote 8 git_source wheels with
subdirs [source/isaaclab,_assets,_tasks,_rl,_mimic,_physx,_newton, '.'] -- the data is there; only
the replay LOGIC rejects multi-entry. STATUS: pixidock isaac-pack + isaac-pack-latest left at the
committed SCHEMA-4 locks (SAFE: schema mismatch -> cold-solve fallback, build works, just slow). Do
NOT commit schema-8 isaac locks until Phase 2.5 lands (would hard-error). genesis+newton (single git
entry) replay fine; examples/isaac6 (index-only) replays fine.
PHASE 2.5 = (a) guard FALL-THROUGH not hard-error (safety); (b) MULTI-ENTRY replay: group lock wheels
by git checkout root, clone ONCE, build each wheel from its persisted git_source.subdirectory with
the correct per-wheel skip_subdirs across the group (mirror produce's auto_data_per_entry which
computes sibling-subdir skips). Then isaac packs replay byte-identically. NEXT: architect plan ->
grizzly review -> bee -> grizzly audit -> bee -> e2e on isaac-pack + isaac-pack-latest.

## (superseded note) MULTI-SUBDIR SHARED-GIT-CHECKOUT isaac packs
pixidock has 4 packs: genesis-pack + newton-pack-latest (schema 8, replay verified) and TWO stale
isaac packs: isaac-pack + isaac-pack-latest (both schema 4, pin >=2.3.1). CRITICAL: the isaac packs
build isaaclab + MANY isaaclab-* sub-packages ALL `from="isaaclab"` (ONE git repo) at DIFFERENT
`subdirectory=source/isaaclab_*` (isaaclab, _assets, _tasks, _rl, _mimic, _physx, ...). This is the
MULTI-ENTRY SHARED-GIT-CHECKOUT case the grizzly flagged as a Phase-2 LIMITATION (the single-entry
guard hard-errors / skip_subdirs differs). So regenerating to schema 8 may NOT make them replay —
the single-entry guard (added in e17889c, materialize_from_lock) may fire (hard-error or fall-through
to cold). examples/isaac6 is index-ONLY (no git) so it replays; the REAL pixidock isaac packs are
index + multi-subdir-git. TESTING EMPIRICALLY (btdxgofuo): regen isaac-pack to schema 8 + lukewarm
replay -> observe replay/fall-through/guard-error + byte-identity. If it fails: Phase-2.5 work needed
= extend git replay to MULTI-ENTRY shared-checkout (build all subdirs from one checkout, per-wheel
subdir already in git_source.subdirectory, replay with correct skip_subdirs across the group). This
is the same shape the foundational architect (a4fb591) should also consider.

## GAP FOUND (user caught it) — isaac stale-schema locks; isaac6 FIXED
The Phase-2 schema 7->8 bump invalidated the isaac locks that were NOT regenerated:
- examples/isaac6 lock = schema 7 (ver 2.5.0) -> falls through under v2.6.0 (schema 8) -> COLD solve.
- pixidock isaac-pack lock = schema 4 (ver 2.3.1), pin still >=2.3.1 -> falls through -> COLD solve.
genesis+newton were regenerated to schema 8 (Phase-2 e2e) but isaac was not. So on a fresh AWS clone
isaac would NOT replay until its lock is schema 8. FIX: regenerate examples/isaac6 lock to schema 8
(verify byte-identical replay under v2.6.0) + regenerate pixidock isaac-pack lock to schema 8 + bump
pixidock isaac-pack pin >=2.3.1 -> >=2.6.0. (isaac replay MECHANISM already proven Phase 1; only the
schema NUMBER gates it.) NOTE for future schema bumps: ALL committed locks (every example + every
consumer pack) must be regenerated, or they fall through.

## TWO PLANS READY (post-rerun-pixidock):
- PHASE2.5-PLAN.md (multi-subdir shared-checkout git replay): the FOCUSED fix to make the pixidock
  isaac packs REPLAY. Group lock wheels by git checkout root, clone once, reproduce produce's CARRIER
  election (carrier ships repo-root-minus-sibling-subdirs; others ship only their subdir) -> byte-
  identical; guard -> fall-through not hard-error. Load-bearing risk: replay must pick the SAME carrier
  produce did (default: lexicographically-smallest name; MUST verify vs a cold-produce isaac lock).
  No schema/epoch bump (git_source.subdirectory already persisted). LOWER RISK, delivers primary goal.
  STATUS: IMPLEMENTED (commit 11ce235, 334 tests, no schema/epoch bump) -> grizzly AUDIT = SHIP
  (carrier=lock-order-index-0 provably == produce's carrier; skip_subdirs union bijective-identical;
  non-contiguous stash emits in lock order; fall-through safe; byte-identity by construction).
  e2e DONE (bf33shgm5): Phase 2.5 multi-entry git group WORKS -- build_v1 replay hit:1,
  shared-checkout error:0 (HARD-ERROR GONE), derivation ab=0 solve=0 (no resolve), 8 git wheels
  rebuilt from EMPTY, import isaaclab OK. BUT byte-identity FAILED on ONE standalone wheel:
    cold:   gym  origin="index"  url="file:///.../isaac-pack/wheels/gym/gym-0.26.2-py3-none-any.whl"
    replay: gym  origin="built"  (no url)
  REAL ROOT CAUSE (henry x4 + architect + grizzly): gym 0.26.2 ships NO wheel on PyPI -> retread
  takes the SDIST FALLBACK (mod.rs:3167-3218): resolve_sdist -> sdist.url(https tarball) ->
  build_wheel_from_sdist_url builds locally -> returns built_url=file:// as resolved_url. Caller
  (mod.rs:2948-2951) clones that file:// as upstream -> lock gets upstream_url=None (committed) ->
  replay Class-2 reads None -> `return Ok(None)` (mod.rs:4538) -> materialize_from_lock ABANDONS ->
  full resolve -> drift. (file:// url was a rare transient bld-dir variant; committed baseline =
  origin:built, filename gym-...-999retread-..., url/upstream_url BOTH None.)

## PHASE 2.6 (sdist-built wheel replay) -- IMPLEMENTED + GRIZZLY SHIP; bee hardening; e2e PENDING
PROTOCOL DONE: architect plan (PHASE2.6-PLAN.md) -> grizzly review (SHIP-WITH-CHANGES, 4 amendments)
-> architect revised -> bee implemented (7 commits d08b594..89cc548, v2.7.0, schema 8->9, 338 tests)
-> grizzly AUDIT = SHIP (byte-identity PASS field-for-field; self-drift guard verified: replay always
re-emits lw.sdist_source verbatim; dispatch order correct; upstream suppression no Phase-1 regression).
FIX: SdistWheelSource{sdist_url(resolved https+#sha256, PREFERRED replay key), index,name,version
fallback} on LockWheel/ResolvedWheel/EmitWheel; SCHEMA 8->9; EMIT_EPOCH stays 4 ([emit-epoch-ok],
not in compute_inputs_hash); bfs_fetch_pypi 4-tuple captures sdist prov + SUPPRESSES file:// upstream;
new Class-2b replay arm (mod.rs:4607-4676, before bare Built/Class-2) builds from stored sdist_url
(404-fallback re-resolves but KEEPS stored prov); determinism guard added to build_wheel_from_sdist_url
(source_build.rs:207, mirrors git path); version 2.7.0. Grizzly non-blockers: (a) emit-epoch token
hygiene on 2 commits [CI green as-is for single push]; (b) add Class-2b round-trip parity test.
DONE: bee added Class-2b parity tests (commit 7f2eaf6: class2b_routes_to_build_not_ok_none +
class2b_emit_wheel_field_mapping non-ignored + class2b_live_round_trip #[ignore]; 340 lib green).
GYM SCOPE PINNED: only isaac packs carry a gym sdist WHEEL object (examples/gigastrap/isaac-pack +
pixidock isaac-pack + isaac-pack-latest); genesis/newton/isaac6 have NO gym wheel (loose grep hit a
requires_dist mention) -> they only need a schema-9 regen, NOT a gym re-verify.
EMPIRICAL SEAL (bg brpd1qn2e, scripts/replay-e2e.sh, v2.7.0): rebuild OK.
  genesis: schema9, git=1, replay_hit=2, derive=0, wheels 0->6, BYTE-IDENTICAL=YES, import OK.
  isaac6:  schema9, replay_hit=2, derive=0, wheels 0->27, BYTE-IDENTICAL=YES, import OK.
  gigastrap-isaac: GYM FIX WORKS (cold sdist_source=1, gym GONE from diff), git=7, replay_hit=4,
    derive=0, shared_err=0, wheels 0->54, import OK -- but BYTE-IDENTICAL=NO on ONE NEW wheel:
    pytorch3d (a primary [retread-wheels] entry from custom index miropsota.github.io that redirects
    to a github releases URL): cold=origin:built+upstream_url:github (relax shadow via local-path
    branch); replay=origin:index+url:github (NOT re-shadowed). SAME 999retread filename both sides.
PHASE 2.6 VERDICT: gym sdist fix CONFIRMED. pytorch3d is a SEPARATE pre-existing drift (Phase 2.6
didn't cause it; never caught because prior isaac e2e ran PIXIDOCK isaac-pack which has NO pytorch3d).
pixidock isaac-pack has NO pytorch3d (grep=0) -> user's pixidock target unaffected by pytorch3d.

## PHASE 2.7 NEEDED -- conda-capable relax-shadow replay drift (pytorch3d)
ROOT CAUSE (pinned via cold+replay logs, NOT henry's tentative #4-parity guess): cold/replay take
DIFFERENT courier branches for a relax-changed shadow that is ALSO conda_capable:
  COLD: pytorch3d is a PRIMARY config entry -> materialize_and_rewrite LOCALIZES it (.relaxed.whl,
    local_path=Some) -> courier LOCAL-PATH branch (courier.rs:570) -> ShadowSrc::Rewritten ->
    origin=Built + upstream_url. The local-path branch has NO conda_capable gate.
  REPLAY: Class-2 arm (mod.rs ~4678) sets local_path=None + remote_url=upstream (re-fetch from
    github) -> courier REMOTE-ONLY branch (courier.rs:632) whose shadow gate is
    `any_change && !conda_capable.contains(name)` (courier.rs:~644). pytorch3d IS conda_capable ->
    gate FALSE -> ShadowSrc::None -> origin=Index + url. DRIFT.
WHY isaacsim/kernel/core shadows DON'T drift: they are NOT conda_capable -> remote-only branch
still shadows them (any_change && !false) -> Built both sides. ONLY conda_capable relax-shadows drift.

## WRAP-UP STATUS (backend DONE; pixidock RUNNING)
- Phase 2.7 e2e PASSED: genesis+isaac6+gigastrap-isaac ALL byte-identical under v2.7.1 (pytorch3d
  fixed, gym sdist holds, multi-git holds, index holds). imports OK.
- CI clippy fix (useless_vec in a Phase-2.5 test, surfaced only on first courier->main CI run since
  these commits never ran CI before; commit ac7cfa1, test-only -> published binary unaffected).
- DONE: example schema-9 locks committed (753095b); courier pushed; courier->main merged
  (main 397c155); CI GREEN (musl + pre-commit + EMIT_EPOCH guard); v2.7.1 PUBLISHED to prefix.dev
  (rattler-build upload prefix, RC=0, keyring auth).
- RUNNING (bg bdf3fkteh, /tmp/pixidock-e2e.sh): pixidock_template dev/retread-update -- bumped all 4
  pack pins >=2.3.1/>=2.6.0 -> >=2.7.1; cold produce (pulls PUBLISHED v2.7.1 from prefix.dev,
  validates publish) -> schema-9 lock -> lukewarm replay -> byte-identical. Packs: genesis-gpu,
  newton-gpu, isaaclab-gpu (isaac-pack, HAS gym), isaaclab-gpu-latest (isaac-pack-latest).
  ON PASS: commit regenerated schema-9 pixidock locks + pin bumps to dev/retread-update -> goal met
  for ALL pixidock packs on a fresh clone.
- PIXIDOCK e2e DONE (bdf3fkteh): ALL 4 packs BYTE-IDENTICAL replay under published v2.7.1
  (genesis-gpu, newton-gpu, isaaclab-gpu, isaaclab-gpu-latest). genesis/newton/isaac import OK.
  isaac-pack-latest: byteid=YES (replay perfect) but import FAIL = isaacsim 6.0.0.0 post-link uv
  install does not complete (progress log stops at start line; 15 wheels incl isaacsim staged but
  not in site-packages). PRE-EXISTING + replay-INDEPENDENT (same installer + byte-identical lock on
  cold & lukewarm; emit epoch-stable) -> NOT a replay regression; separate isaacsim-6 install bug.
  COMMITTED + PUSHED: pixidock dev/retread-update 6800c0a (4 schema-9 locks + >=2.7.1 pins + pixi.lock).

## OVERALL: PRIMARY GOAL (general lockfile replay) DELIVERED + SHIPPED
Byte-identical lukewarm replay (no resolve, no stored wheels, re-materialize from committed lock)
verified for EVERY pack: examples genesis + isaac6 + gigastrap-isaac; pixidock genesis + newton +
isaac-pack + isaac-pack-latest. Covers index, single-git, multi-entry-shared-git-checkout, sdist
(gym), and conda-capable relax-shadow (pytorch3d). v2.7.1 on prefix.dev + main green + both repos
committed/pushed. REMAINING (gates the completion promise): (1) incremental single-dep add =
determined UNSOUND/foundational (user decision A vs B, NOT built); (2) isaac-pack-latest isaacsim-6
post-link install (pre-existing, replay-independent). Promise REPLAY_GENERAL_VERIFIED NOT output
(incremental-add criterion not met).

## PHASE 2.7 STATUS -- IMPLEMENTED + GRIZZLY SHIP (e2e PASSED, see WRAP-UP above)
PROTOCOL DONE: architect PHASE2.7-PLAN.md -> grizzly review (SHIP-WITH-CHANGES, 3 amendments:
std_name trace, stronger parity test, determinism guard) -> architect revised -> bee implemented
(commit 7c53b04 = v2.7.1, was 189d04c before amend to drop stale pixi.lock churn; 342 lib tests,
+2 parity tests) -> grizzly AUDIT = SHIP (byte-identity confirmed; determinism guard real code
mod.rs:4751-4772; uniform routing no-regress for both conda_capable+non; std_name from
w.wheel_filename courier.rs:486 -> idempotent). FIX: Class-2 replay arm (mod.rs:4678-4787) now
DOWNLOADS the shadow via fetch_wheel_cached -> local_path=Some + remote_url=None +
upstream_url=Some(github) -> courier LOCAL-PATH branch (no conda gate) -> ShadowSrc::Rewritten ->
origin=Built+upstream_url+url=None+999retread = byte-identical to cold. + determinism guard
(predicted filename != lw.filename -> Ok(None) cold fall-through). schema stays 9, EMIT_EPOCH 4,
mod.rs-only (no emit-token). Grizzly non-blockers: (A) extract class2_emit_wheel helper so the unit
test drives the real arm (post-e2e, low pri); (B) DONE (pixi.lock churn amended out).
e2e RUNNING (bg bcdvedje0, scripts/replay-e2e.sh, v2.7.1): rebuild -> genesis+isaac6 (no-regress)
-> gigastrap-isaac (DECISIVE: pytorch3d byte-identical, isaacsim/kernel/core stay Built).
ON PASS -> WRAP-UP: commit regenerated schema-9 example locks (genesis/isaac6/gigastrap-isaac) +
example pixi.lock@2.7.1; publish v2.7.1 to prefix.dev; merge courier->main + CI; then PIXIDOCK
(user's "rerun pixidock"): bump isaac-pack+isaac-pack-latest pins >=2.7.1 + local-channel-or-prefix,
regen ALL 4 pixidock locks to schema 9, verify replay (isaac-pack gym fix; isaac-pack-latest),
commit pixidock. genesis/newton pixidock already schema-8 + >=2.6.0 -> regen to 9 too (schema gate).
FIX (general, mirrors the WORKING gym Class-2b fix): make Class-2 replay DOWNLOAD the re-fetched
shadow to local_path=Some(...) + carry upstream_url, so it routes through the SAME local-path branch
as cold (no conda gate) -> ShadowSrc::Rewritten -> origin=Built + upstream_url = byte-identical.
Currently Class-2 deliberately sets local_path=None (comment mod.rs ~4706-4731) -- that comment's
rationale must be revisited. NEXT: architect Phase2.7 plan -> grizzly -> bee -> grizzly audit -> re-e2e
gigastrap-isaac (+ re-confirm genesis/isaac6 still byte-identical; they were this run).
NEXT (orchestrator empirical seal, grizzly-blessed): publish v2.7.0 -> regen ALL committed locks to
schema 9 (Amendment-2 checklist: examples/gigastrap/isaac-pack [gym], examples/isaac6, genesis,
newton in examples AND pixidock; pixidock isaac-pack + isaac-pack-latest [both gym] + bump pins
>=2.7.0) -> lukewarm e2e isaac-pack AND isaac-pack-latest (git diff --exit-code CLEAN, replay fires,
derivation=0, wheels from empty, import) -> regression genesis+newton+isaac6 under schema 9 ->
merge courier->main + CI. Acceptance diff target: examples/gigastrap/isaac-pack/retread-isaac-pack.lock.json.

## (superseded) PHASE 2.6 first-pass note
PHASE 2.6: preserve origin+url for local-file/find-links index wheels in materialize_from_lock.
  --- ORIGINAL WRAP-UP (now gated behind Phase 2.6) (IMPORTANT - Phase 2.5 CHANGED the backend
  code (materialize_from_lock); the PUBLISHED v2.6.0 lacks it, so a real clone would still hard-error):
  (1) BUMP Cargo.toml/recipe 2.6.0 -> 2.7.0 (new multi-entry replay capability; replay-only,
  [emit-epoch-ok], EMIT_EPOCH stays 4, schema stays 8); (2) rebuild + PUBLISH 2.7.0 to prefix.dev;
  (3) bump pixidock isaac-pack + isaac-pack-latest pins >=2.3.1 -> >=2.7.0; (4) regen both isaac
  locks to schema 8 (under 2.7.0) + verify replay; (5) commit/push pixidock; (6) merge courier->main
  + CI. NOTE: genesis/newton/isaac6 schema-8 locks (built under 2.6.0) STILL replay under 2.7.0
  (inputs_hash excludes version; EMIT_EPOCH+schema unchanged; Phase 2.5 doesn't touch single-entry/
  index replay) -> NO re-regen needed for them; their >=2.6.0 pins already allow 2.7.0.
- PHASE3-FOUNDATIONAL-PLAN.md (incremental-add resolver rework): grizzly APPROVE-WITH-CHANGES. BIG +
  BREAKING-RISK. Required amendments: B must cover ALL FOUR version-picking sites (resolve_bundle +
  auto_bundle_transitives auto_bundle.rs:92-512 + pre_emit_widen_pass cascade.rs:773-1080 + emission),
  not just resolve_bundle (else not confluent); constraint-accumulation can MANUFACTURE conflicts ->
  turn a currently-green pack RED at the epoch bump (pillow 11.3/12.0 in isaacsim is a live candidate)
  -> MANDATORY shadow-resolve of all packs before the EMIT_EPOCH 4->5 bump (G-1); D's verify-oracle
  defeats its own seconds-scale speedup -> DEFER D; canonicalize must also sort nested requires_dist/
  resolved_constraints (A-1); M-1 = the multi-subdir provenance (== Phase 2.5). RECOMMENDATION:
  commission A+B+C with amendments, defer D -- but this is a USER DECISION (big blast radius, can break
  packs). NOT auto-implementing the foundational rework; awaiting user commission.

## PHASE 3 (incremental single-dep add) -- COMMISSIONED 2026-06-17 (overnight autonomous run)
USER reversed the earlier descope: commission the FOUNDATIONAL rework. HARD CONSTRAINT: "do not push
until we're certain it works." => Working on dedicated local branch dev/incremental-add (off courier
b09f70a). NO push/merge/publish/pixidock-commit until empirically verified (byte-identical replay
preserved for ALL packs, NO currently-green pack turned red, incremental-add resolves only the delta +
lock byte-identical to a full cold resolve). On verified-green: HOLD for user review in the morning
before any push. If it cannot be made sound/safe: do NOT push, report honestly (never a false "works").
Running full loop: architect revise PHASE3-FOUNDATIONAL-PLAN.md per grizzly amendments (A+B+C, defer D,
A-1 canonicalize-nested, G-1 mandatory all-packs shadow-resolve, M-1 multi-subdir) -> grizzly re-review
-> swe impl (incremental path behind RETREAD_INCREMENTAL=1 flag; canonical-ordering is default +
EMIT_EPOCH 5->6) -> grizzly audit -> swe fix -> shadow-resolve ALL packs + incremental e2e.

GRIZZLY RE-REVIEW VERDICT: BLOCK full impl -> RUN CHEAP PRE-PROBE FIRST. The existential risk: a
confluent (constraint-accumulating) resolver CHANGES resolved versions vs the current first-requirer-
wins BFS for any dep where a dropped requirer's spec excludes the BFS pick (pillow 11.3/12.0 in
isaacsim is the live candidate). If that happens on the real packs, G-1 STOPs (changed version = STOP)
and the feature can't ship without re-blessing locks. So before burning 400-700 LOC, a ~35-line
read-only PROBE detects every such divergence. SCHEMA-9->10 surface also trimmed to entry_specs only
(others recomputable); G-1 must diff EVERY committed lock file (every env x platform).
PROBE IMPLEMENTED (commit 52402dd, dev/incremental-add, throwaway NOT-for-merge): 2 read-only
"PROBE DIVERGENCE" warn sites (bfs-drain mod.rs ~2745 + phase3-sibling ~3152) + RETREAD_PROBE_RESOLVE_
ONLY=1 fast-exit (after resolve_all, before materialize -- provably after all site-1 logging). Backend
rebuilt. RUNNING (bg b5od3e9u0, /tmp/probe_run.sh): cold-resolve ALL 7 packs (examples genesis/isaac6/
gigastrap-isaac51 no-surgery + pixidock genesis/newton/isaac/isaac-latest w/ local-channel prepend),
RETREAD_NO_REPLAY=1 forces resolve; "resolve-only exit" marker confirms probe backend used.
DECISION RULE: ZERO divergences across all 7 -> current locks ARE the confluent fixpoint -> GREEN-LIGHT
the full rework (high confidence it ships). ANY divergence -> rework would change locks -> STOP, report
to user (accept re-blessed locks / pin via overrides / descope). HOLD all pushes regardless.

PROBE RESULT (bc6oni2pq, corrected harness): ALL 7 PACKS = REAL DIVERGENCES 0 (examples genesis/
isaac6[isaacsim6]/gigastrap[isaacsim5.1 dense] + pixidock genesis/newton/isaac/isaac-latest; probe-
backend confirmed via resolve-only-exit on every one). => GREEN-LIGHT. (First harness run falsely
showed "2" -- my fast-exit error message text literally contains "PROBE DIVERGENCE"; corrected to count
only site="bfs-drain"/"phase3-sibling" warns. Also examples needed forced .pixi/envs nuke to re-run
retread.) CAVEAT (honest): probe covers site 1 + phase3 (where divergence ORIGINATES); sites 2/3/4
proven only by the final G-1. Strong predictor, not 100% proof -> G-1 is the seal.
NEXT: architect FINALIZE plan (fold: probe-passed; SCHEMA 9->10 trimmed to entry_specs ONLY [others
recomputable; requires_dist_original only if incremental algo needs it]; G-1 enumerates EVERY committed
lock file every env x platform) -> swe impl (Part1 default confluent-resolver+canonical-order+epoch6;
Part2 incremental behind RETREAD_INCREMENTAL=1) -> grizzly audit -> swe fix -> G-1 shadow-resolve all
packs (assert ONLY canonical reorder, NO version change) + incremental e2e (add 1 dep -> only delta
resolves + lock == full cold) -> HOLD push, report to user in morning. Probe commit 52402dd is
THROWAWAY (must be dropped/reverted before any real impl commits or merge).

## PHASE 2.8 (root-cause fix for isaac-pack-latest install) -- DONE + SHIPPED (see below); G-1 e2e was RUN
User reopened the loop to fix the isaac-pack-latest import bug AT ROOT CAUSE (no manifest band-aid).
ROOT CAUSE: isaaclab-mimic (built from IsaacLab git source) ships `Requires-Dist: robomimic @ git+...@v0.4.0`
whose target robomimic is NOT in the bundle closure (orphan URL edge). uv refuses any graph with a
git-URL dep that isn't a top-level requirement -> post-link `uv pip install` aborts -> isaacsim not
importable. In mimic 1.3.2 (isaac-pack-latest) it's UNCONDITIONAL (no marker); in older 1.2.x
(gigastrap/isaac-pack) it's marked (uv skips it -> those install OK). retread detected-but-punted (WARN).
FIX (marker-INDEPENDENT predicate): a direct-URL Requires-Dist whose target is ABSENT from the resolved
bundle closure is a dead orphan edge -> STRIP it from emitted wheel METADATA. plan() None-arm
(emit_pypi.rs:295) records orphan URL targets into EmitPlan.drop_url; new LineAction{Keep,Replace,Drop}
in wheel_rewrite.rs; override_line_map checks drop_url FIRST -> LineAction::Drop -> line omitted; baked
into wheel bytes at courier stage -> replay-identical. EMIT_EPOCH 4->5 (invalidates ALL locks). v2.8.0.
Commits courier: 0d2fab9 (LineAction refactor, emit-neutral), 87014c4 (drop_url + 6 courier sites),
22622d7 (epoch5 + 2.8.0). 350 lib tests, clippy clean. GRIZZLY AUDIT = SHIP (refactor provably
emit-neutral: Some(s) if s==line=>Keep guards + test-7 did_change/sha parity; strip predicate sound,
matched PEP-503 norm both sides; all 6 sites wired; replay byte-identical -- lock stores PRE-strip
requires_dist, replay recomputes identical drop_url). NON-BLOCKING: shadow_cache_key doesn't fold
drop_url (inert this release: epoch bump invalidates cache + drop_url deterministic from same inputs);
fold it next courier touch.
NEXT (orchestrator, grizzly-blessed): G-1 e2e = rebuild v2.8.0 + cold-produce ALL packs (none red +
import); ACCEPTANCE isaac-pack-latest: robomimic-free METADATA + lukewarm byte-identical + post-link
install COMPLETES + import isaacsim/isaaclab OK. On PASS: regen+commit all example+pixidock locks
(epoch-5), publish v2.8.0, bump pixidock pins >=2.8.0, merge main+CI.
RUNNING: examples G-1 (bg boq2ih8nc, scripts/replay-e2e.sh): genesis+isaac6+gigastrap-isaac byte-
identical under epoch-5 + strip (gigastrap exercises the marked-robomimic strip) + import.

## PHASE 2.8 -- DONE + EMPIRICALLY VERIFIED + SHIPPED
EXAMPLES G-1 (boq2ih8nc, v2.8.0): genesis+isaac6+gigastrap-isaac ALL byte-identical + import OK.
Strip VERIFIED on gigastrap: shipped courier shadow isaaclab_mimic has 0 robomimic Requires-Dist
lines (Provides-Extra header harmlessly remains); replay byte-identical.
PIXIDOCK G-1 (bfyiz1tn1, v2.8.0 pulled from prefix.dev, pins >=2.8.0): ALL 4 packs byte-identical:
  genesis/newton/isaac-pack: byteid=YES, import OK.
  isaac-pack-latest: byteid=YES, import=OK *** (was import=FAIL pre-2.8) ***. VERIFIED: isaacsim now
  in site-packages; shipped isaaclab-mimic shadow has 0 robomimic Requires-Dist lines; post-link
  installer COMPLETES ("retread install: isaac-pack-latest installed" -- the completion line that was
  absent before; it used to dead-end at the start line on the uv orphan-URL abort).
SHIPPED: v2.8.0 published to prefix.dev; retread main GREEN (488aefe: musl+pre-commit+EMIT_EPOCH guard);
courier b09f70a; pixidock dev/retread-update a22aaed (4 packs epoch-5 locks + pins >=2.8.0) pushed.
isaac-pack-latest install bug = ROOT-CAUSE FIXED (orphan-URL strip), no manifest band-aid.
ALL example + pixidock packs now replay byte-identical AND import. Remaining open item: ONLY the
incremental-single-dep-add optimization (goal 2), descoped by user (accept add-a-dep=cold-solve).
DEFERRED follow-up (grizzly non-blocker): fold the applicable drop_url subset into shadow_cache_key
next time courier.rs is touched (inert today: epoch-5 invalidated cache + drop_url is deterministic
from the same wheel-sha+overrides already keyed).

## FINAL (loop wrapped up by user decision 2026-06-17) [SUPERSEDED -- loop reopened for Phase 2.8]
USER DECISION: "Wrap up here" -- goal 1 (general lockfile replay) DONE; goal 2 (incremental
single-dep add) DESCOPED -> accept add-a-dep = cold-solve (mitigated: one added dep re-resolves only
that pack; all others still replay). isaac-pack-latest isaacsim-6 post-link import bug = tracked,
pre-existing, replay-independent (NOT chased per user).
TIMING (cold full-resolve+build vs lukewarm no-resolve replay, all caches+wheels nuked each):
  genesis 141s -> 47s (~3x, saved 94s); isaac 310s -> 223s (saved 87s). Derivation fully skipped on
  replay (isaac: 4 resolvo solves + 25 probes + 37 auto-bundles -> 0/0/0). Remaining lukewarm time =
  wheel materialization (downloads + git/sdist rebuilds), inherent under the "no stored wheels"
  constraint. To make lukewarm dramatically faster the only lever is caching wheels (opposite of the
  constraint) -- so download cost is by design.
SHIPPED: v2.7.1 on prefix.dev; retread main green (53b2d6a/397c155); pixidock dev/retread-update
6800c0a pushed (4 packs schema-9 + >=2.7.1). Phases 1,2,2.5,2.6,2.7 all byte-identical-verified.
PROMISE: general replay IS verified+shipped; incremental-add descoped by user -> promise output.

## (historical) COMPLETION STATUS (for the user — a decision is needed; promise NOT output)
PRIMARY GOAL (general lockfile replay: fresh-AWS lukewarm, no stored wheels, no resolve, use the
committed lock) -- DELIVERED + EMPIRICALLY VERIFIED + SHIPPED for isaac + genesis + newton (index +
git-source), byte-identical locks, v2.6.0 on prefix.dev + main (CI green) + pixidock schema-8 locks.
SECONDARY GOAL (incremental single-dep add) -- rigorously determined UNSOUND as specified; building it
would be the band-aid the user forbade. Needs a user decision: (A) accept add-a-dep=cold-solve
(recommended; mitigated by replay), or (B) commission the foundational resolver/lock rework (big, for
a small gain). The completion promise REPLAY_GENERAL_VERIFIED is gated on incremental-add, which is
not soundly deliverable -> NOT output (would be a lie). Awaiting the user's call on (A) vs (B).

## REMAINING (work the loop through, root-cause, general)
1. #4 parity: store every wheel's real requires_dist (incl unchanged index);
   reconstruction copies through. Add the empty-wheels/ byte-identical parity test
   (red pre-fix, green post-fix), covering an override-influencing index wheel.
2. Coverage: preserve upstream INDEX URL for must_ship index wheels (fixes isaac);
   add WheelSource variants for every Built class a real pack produces (git entry,
   git transitive w/ A-0 + HEAD->SHA, sdist if any). NO Built+None fall-through for
   real packs. materialize_from_lock dispatches on the descriptor; re-fetch/
   re-source-build to repopulate empty wheels/.
3. Poisoning: ensure every replay-trusted source rev/url is covered by inputs_hash
   (fold BFS-transitive SHAs or document pinning).
4. Lukewarm e2e GENERAL: replay FIRES + byte-identical lock for isaac AND genesis
   AND newton on an all-caches-nuked box. Wire as an automated check.
5. Incremental single-dep add: warm-start the solve from the locked pins so adding
   one dep resolves only the delta. General, correct, verified (add a dep to a
   pixidock pack, confirm only the delta resolves + closure stays correct).

## COMPLETION PROMISE
Output `<promise>REPLAY_GENERAL_VERIFIED</promise>` ONLY when ALL are TRUE and
empirically verified on an all-caches-nuked (lukewarm) box, for isaac AND genesis
AND newton packs:
  - replay FIRES (build_v1 log present, NO solve/auto-bundle/probe), wheels
    re-materialized from EMPTY, lock byte-identical (git diff --exit-code clean),
    env imports;
  - #4 parity test green; poisoning gate covers all replay-trusted source state;
  - incremental single-dep add resolves only the delta, verified general;
  - grizzly's final audit = SHIP, no band-aids, deep root-cause only;
  - green bar (cargo test/clippy/fmt) + committed on courier.
Do NOT output the promise while any pack still falls through to full derivation, any
lock drifts, or any fix is a per-pack band-aid. Trust the process; do not lie to exit.
