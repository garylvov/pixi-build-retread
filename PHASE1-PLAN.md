# PHASE 1 PLAN (grizzly-reviewed, architect-revised, READY to implement)

Goal: make `conda/build_v1` replay FIRE for the index-wheel (Class-2) case on
examples/isaac6 — general, root-cause, no band-aids. isaac6's 22 "built" wheels
are Class-2 relax-changed INDEX SHADOWS (must_ship=false, 999retread tag, from
pypi.nvidia.com); replay is inert because the lock records upstream_url=None for
them + requires_dist=[] for 45 index wheels.

Toolchain: `PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH" cargo ...` ONLY.
Branch: courier. Keep cargo test --lib + clippy --all-targets -D warnings + fmt green at EVERY commit.
OUT OF SCOPE: git provenance/newton, sdist, A-0 #subdirectory=, HEAD->SHA, incremental-add.

## STEP 0 (FIRST, MANDATORY) — fix courier rename-vs-rewrite bug
Bug: force-download writes `.dl-courier-{std_name}` INSIDE staging_dir (courier.rs:622);
the `changed` block dispatches on `src.starts_with(staging_dir)` (courier.rs:654) -> the
.dl raw file wrongly takes the RENAME branch (657) -> ships RAW un-relaxed bytes under the
999retread shadow name, skipping rewrite_wheel_with. => Class-2 re-fetch on empty wheels/
is NOT byte-identical. The probe_dst path (565) is also in staging_dir but IS already
rewritten (shadow_cache_stage 568) -> rename correct for it. The heuristic conflates them.
FIX (option b, explicit state): replace `(changed, shadow_src): (bool, Option<PathBuf>)`
(courier.rs:544) with a 3-state distinguishing already-rewritten vs raw, e.g.
`enum ShadowSrc { Rewritten(PathBuf), Raw(PathBuf), None }`:
  - cache did_change arm (574-575) -> Rewritten(probe_dst)
  - no-cache arm (600) -> Raw(src)
  - remote-only force-download arm (640) -> Raw(dl)
  - unchanged arms -> None
In the `if changed` block (646-678) dispatch on the state: Rewritten -> RENAME (657-663);
Raw -> rewrite_wheel_with (664-678). DELETE the starts_with(staging_dir) heuristic (654).
Unit test: a Raw force-downloaded .dl goes through rewrite_wheel_with (shadow bytes != raw).
EMIT: courier matches the CI emit regex; if the no-cache/force-download path emits different
bytes for a real pack, BUMP EMIT_EPOCH 3->4 in this commit (lock.rs:148). Else [emit-epoch-ok].
Commit 1: fix(courier): re-rewrite force-downloaded shadow instead of renaming raw bytes.

## STEP 1 — EmitWheel.upstream_url from PRISTINE w.url
Diagnosis: localize_wheel_source (mod.rs:3879-3906) only collapses to file:// when the
file EXISTS; on lukewarm (empty wheels/) it returns the upstream URL unchanged. The URL is
lost at PRODUCE time (warm box) on the local-path branch. build_one zips pristine
bundle.all_wheels() `w` with a SEPARATE localized_urls vec (mod.rs:4327-4374); `w.url` at
4360 IS pristine.
- Add field to EmitWheel (emit_pypi.rs:59-81): `pub upstream_url: Option<url::Url>` (doc:
  pristine pre-localization index URL; None for source-built .injected wheels). Do NOT
  overload remote_url (stays scheme-derived from the localized url at 4372).
- Populate in cold EmitWheel build (mod.rs:4360): `upstream_url: (w.url.scheme()!="file").then(|| w.url.clone())` reading PRISTINE w.url.
- Fix all other EmitWheel literals (compiler-enforced): materialize_from_lock reconstructions
  (mod.rs:3991, 4044, 4094) per Step 3; tests.
Commit 2: feat(emit): EmitWheel.upstream_url carries pristine pre-localization index URL. [emit-epoch-ok]

## STEP 2 — producer writes upstream_url (Class-2) + real requires_dist (Class-4)
courier.rs:
- Shared `changed` block (680-699): change upstream_url (688) to PREFER the new field:
  `let upstream_url = w.upstream_url.as_ref().or(w.remote_url.as_ref()).map(|u| u.to_string());`
  (local-path shadow now gets the pristine nvidia URL; remote-only arm still works.)
- Unchanged-index branch (700-715): change `requires_dist: vec![]` (712) to
  `requires_dist: w.requires_dist.clone()` (the #4 parity fix). Leave upstream_url:None (714);
  Index wheels use `url`. Update the stale comment at 681-687.
Lock fields NOT in compute_inputs_hash (lock.rs:184-212) -> [emit-epoch-ok].
Commit 3: fix(courier): record upstream_url for local-path shadows + real requires_dist for index wheels.

## STEP 3 — replay reconstruction parity contract (local_path + conda_capable + ship sort)
plan() (emit_pypi.rs:191-292) ALSO reads local_path (216 ship-set), must_ship()/filename (199),
version (215/280), conda_capable (278). EmitPlan.ship is a HashSet (180, non-deterministic).
In materialize_from_lock (mod.rs:3975-4103):
- local_path: reconstruct `Some(<source_dir>/wheels/<lw.filename>)` for every locally-
  materialized wheel (Class-1, Class-2, Class-4-when-localized) so plan()'s ship-set matches
  the cold-produce value. swe-bee MUST diff cold-produce EmitWheel.local_path vs replay
  EmitWheel.local_path for isaac6 wheel-for-wheel; if reconstruction is impossible at
  EmitWheel-build time, ESCALATE (plan()'s reliance on live local_path may be the deeper bug;
  do NOT fake it).
- conda_capable: cold path MERGES probe_decisions + config.name_map + load_pypi_to_conda_map
  (mod.rs:4375-4382); replay uses lock.conda_capable (4108-4109). CONFIRM the producer writes
  the MERGED set (sorted) into RetreadLock.conda_capable (lock.rs:120-126); if it writes only
  the probe subset, fix it here. Add a parity assertion.
- ship HashSet: grep `.ship` consumers; SORT anywhere it reaches the committed lock or staged
  package (overrides are already BTreeMap, deterministic).
Commit 4: fix(replay): reconstruct local_path/conda_capable + deterministic ship order for plan() parity. [emit-epoch-ok]

## STEP 4 — schema 6->7 (HYGIENE) + regen isaac6 lock
SCHEMA already 6 (lock.rs:129); committed isaac6 lock is schema-4, ALREADY rejected by the
`!=` gate (mod.rs:4663). Bump 6->7 is HYGIENE (signals new load-bearing semantics), not the
fall-through enabler. requires_dist/upstream_url NOT in compute_inputs_hash -> [emit-epoch-ok];
EMIT_EPOCH stays at Step-0's value. SCHEMA-only change is explicitly NOT an epoch bump.
Regenerate examples/isaac6 committed lock (fresh cold PRODUCE) to schema-7 with new fields.
Commit 5: chore(lock): schema 6->7 + regen isaac6 lock.

## STEP 5 — DETERMINISTIC LOCAL-FIXTURE byte-identical parity test (red->green)
MUST NOT hit pypi.nvidia.com (stage force-downloads via reqwest::get, courier.rs:623;
RETREAD_NO_SHADOW_CACHE only toggles cache not network).
- Check in a tiny raw wheel under tests/fixtures/ with a Requires-Dist line the relax policy
  WILL change (-> becomes a Class-2 shadow). Set upstream_url to a file:// URL (preferred,
  hermetic) OR a localhost one-shot HTTP server if reqwest::get rejects file:// (VERIFY which;
  if file:// needs a code branch, propose it but keep it out of the threading commits).
- Test 1 (unit): plan() purity — same (requires_dist, version, must_ship/filename,
  local_path.is_some(), conda_capable) -> identical EmitPlan (overrides BTreeMap eq + ship as
  SORTED Vec).
- Test 2 (integration): PRODUCE shadow+lock; wipe wheels/; REPLAY via materialize_from_lock ->
  courier::stage with the local fixture; ASSERT (a) re-staged shadow BYTE-IDENTICAL to produce
  (Step 0 makes this true), (b) replay lock byte-identical (to_pretty_json eq), (c) plan()
  identical. RED before Steps 0+2+3, GREEN after — prove red on the pre-fix tree.
Commit 6: test(replay): empty-wheels byte-identical parity with local fixture (red->green).

## STEP 6 — isaac6 lukewarm e2e (verification only, no commit)
`bash scripts/rebuild-local.sh` (isaac6 pins version="*" + local-channel first). Nuke ALL
warm state (handoff lines 43-48) incl the isaac6 pack wheels/, KEEP .pixi/config.toml. pixi
install isaac6. ASSERT: `build_v1: replayed from lock` present; ZERO auto-bundled/resolvo
solve/probe-trace; wheels/ repopulated from EMPTY; `git diff --exit-code` clean on the lock;
import isaacsim + isaaclab. Record results in HANDOFF-REPLAY-LOOP.md DONE/REMAINING.

## ESCALATION FLAGS (don't guess)
- Step 0: kill the starts_with(staging_dir) heuristic; option (b) explicit state preferred.
- Step 3.1: if local_path can't be reconstructed to match cold-produce, ESCALATE (plan()'s
  live-local_path reliance may be the deeper bug). Do NOT fake it.
- Step 5: confirm reqwest::get accepts file://; if not, localhost HTTP one-shot, NEVER real index.
