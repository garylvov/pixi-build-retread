# PHASE 3 PLAN — incremental single-dep add (architect feasibility + design)

VERDICT: FEASIBLE + correctness-able, but the WIN IS MODEST (seconds-scale). It saves DERIVATION
(BFS metadata fetches + auto-bundle probes + solve-check) for unchanged subtrees, NOT
materialization (wheel bytes must still ship/download — dominates wall-clock, same as replay).
resolvo (rattler_solve 4.1.0) DOES expose locked_packages/pinned_packages warm-start, but retread
leaves them empty (src/solve_check.rs:111-123) — and that's only the cheap conda solve-check.
pixi/uv already warm-start their own solves from pixi.lock; retread's cascade is the part that cold-starts.

## COST (where to optimize)
- retread PyPI resolution = its own BFS (bfs_fetch_pypi, src/handler/mod.rs:3120), NOT resolvo.
- resolvo used ONLY for conda solve-check (src/solve_check.rs); already sparse + abstains when no repodata.
- Cold cache: dominated by wheel downloads/repack (materialize_and_rewrite ~3303) — NOT saved by incremental.
- Warm cache: BFS = hashmap lookups; ~32 solves (8 passes x 4 envs); seconds. Incremental saves these.
- Materialization happens even on a hash-match replay -> incremental add still materializes full closure;
  only re-DERIVATION of N-1 unchanged subtrees is skipped.

## DESIGN (if built — fail-closed, no stale closure)
1. PERSIST entry_specs (SCHEMA 8->9, [emit-epoch-ok], NOT in compute_inputs_hash): add
   RetreadLock.entry_specs = courier_input_specs(config,bundle) output (src/courier.rs:49-86), written in stage.
   Today the lock lacks the user's [retread-wheels] specs (root_requirements is just the meta-wheel pin).
2. HOOK at the replay-MISS arm (src/handler/mod.rs:1897, before resolve_all at 1915): new
   try_incremental_resolve(lock, config, ...) -> Option<(Vec<Bundle>, RetreadConfig)>.
3. DELTA DETECT: require everything in inputs_hash EXCEPT entry_specs byte-identical to lock
   (config_fingerprint, index_urls, relax, python, EMIT_EPOCH, pin_version); require entry_specs diff =
   EXACTLY ONE ADDED spec, zero removed/modified. Else -> None (full fallback).
4. RESOLVE only C(D) (new entry's closure) via existing resolve_bundle (mod.rs:2560) + scoped auto_bundle.
   Reuse unchanged subtrees by synthesizing Bundles from lock.wheels (like materialize_from_lock replay).
5. CORRECTNESS CONDITION (the user's hard constraint): re-validate the MERGED closure L u C(D):
   (1) no version conflict on any shared name; (2) every new requirement edge targeting an existing pin
   is satisfied by that pin; (3) every pin satisfies every constraint pointing at it (pure in-memory check
   over requires_dist already in the lock + C(D)'s fetched metadata). If ANY fails -> None -> FULL resolve.
   NO "re-resolve just the touched subtree" (stale-closure-prone). Fail closed.
6. LOCK-CORRECTNESS INVARIANT: incremental-produced lock MUST be BYTE-IDENTICAL to a full cold
   resolve of the same final manifest (provable: unchanged subtrees copied verbatim = cold output for
   unchanged inputs; correctness condition guarantees C(D) doesn't perturb them). Never a superset.
7. (OPTIONAL, separable) resolvo warm-start: fill SolverTask.locked_packages/pinned_packages from
   lock.conda_run_deps (src/solve_check.rs:111) — helps EVERY cold solve, small contained win, no delta machinery.

## COMMITS: (1) schema 8->9 entry_specs [emit-epoch-ok]; (2) delta detector + correctness module (pure,
unit-tested); (3) try_incremental_resolve + hook, behind RETREAD_INCREMENTAL=1 (opt-in until e2e-green);
(4) optional resolvo warm-start; (5) flip default after green e2e. EMIT_EPOCH stays 4.

## TESTS/E2E: unit correctness cases (disjoint=safe; tighten-shared=None/fallback; conflict=None;
same-version-share=safe); byte-identical parity (incremental lock == full cold lock); lukewarm e2e:
add one dep, assert only delta derives (zero re-BFS of unchanged names) AND lock == full cold resolve
(RETREAD_NO_REPLAY=1 + git diff --exit-code); negative e2e: tighten-shared dep -> falls back to full.
Run for genesis + newton + isaac (generality).

## OUT OF SCOPE: multi-dep add, dep removal/modification (-> full resolve); resolvo warm-start as
PRIMARY mechanism (conda-only, low value); subtree-only re-resolve (rejected); materialization savings
(impossible, bytes must ship); conda_outputs path (cold-derives, memoized).

## RECOMMENDATION: build the schema-9 entry_specs + delta detector + correctness module + fail-closed
try_incremental_resolve behind an opt-in flag; treat resolvo warm-start as separate optional win. Honest
cheaper alternative: document add-a-dep = cold solve (status quo) + optionally just the resolvo
locked_packages warm-start. The win is seconds-scale (derivation only); decide if the warm-cache
add-a-dep dev loop justifies the correctness-audit surface.
