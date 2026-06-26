# PHASE3-RESOLVO-PLAN.md

Feasibility-first design for replacing retread's hand-rolled PyPI BFS with a
real retracting solver built on **resolvo**. Written for the-grizzly's review.

Branch: `dev/incremental-add` (HEAD 7b89af7). All citations re-pinned via henry.

---

## 0. TL;DR / BLUF

**Verdict: FEASIBLE, but NOT a few-hundred-LOC drop-in. This is a multi-week
subsystem (~1200-1800 LOC + a discovery/probe pre-pass) with two findings that
force design decisions the user must sign off on BEFORE any code lands.**

The two findings that change the shape of the plan:

1. **resolvo 0.10.3 is pseudo-async** (`NowOrNeverRuntime`, panics on real
   `await`). The `DependencyProvider` methods are `async fn` for API uniformity
   only; the solver polls each future exactly once and panics if it yields.
   `resolvo-0.10.3/src/lib.rs:117-152`. So we CANNOT fetch PyPI metadata or run
   the conda probe lazily inside `get_dependencies`. **Everything must be
   pre-fetched into an in-memory Pool before the synchronous solve.** This is the
   single largest piece of new work and the source of most risk.

2. **The conda-routing probe is version-spec-dependent, not name-only**
   (`mod.rs:2887-2982` probes `probe(name, specifiers.to_string(), python)`,
   where `specifiers` is the live `Requires-Dist` constraint). `pick_conda_target`
   itself is name-only (`auto_bundle.rs:50-66`), but the *decision* to route is
   gated on an async probe at the specific version-spec. Combined with #1, this
   means the conda boundary cannot be answered inside the sync solver and must be
   pre-computed -- which is only possible if we discover all reachable
   `(name, spec)` edges BEFORE solving. That requires a **fixpoint discovery
   pass** (fetch -> find new edges -> fetch again) because resolvo can surface
   edges the pre-pass didn't see.

**The make-or-break soundness question (incremental == cold):** resolvo's
warm-start primitive in this version is `Candidates::favored` (a SOFT hint that
backtracks) wired through `SolverTask.locked_packages`
(`rattler_solve-4.1.0/src/lib.rs:219-282`; resolvo `Candidates` at
`resolvo-0.10.3/src/lib.rs:156-194`). **A soft hint CAN diverge from cold** in
principle. BUT: incremental==cold does NOT actually depend on warm-start
soundness. It depends on resolvo being a **deterministic complete solver** -- if
cold and incremental both ask resolvo for "the highest-preferred consistent
solution over the same constraint set," resolvo (being order-independent and
deterministic) returns the same answer regardless of seeding. The seed is a
search-order optimization, not a result determinant -- PROVIDED we never use the
HARD lock (`pinned_packages` -> `Candidates::locked`) to pin prior versions.
See §D for the full argument and the required verification test. **This is the
one claim that must be empirically verified before PR-4 ships.**

**No lock-format change.** Schema stays 10 (`lock.rs:259`). The resolver is an
internal detail; only the resolved `{name:version}` set is persisted, mapping
cleanly into existing `wheels` + `conda_run_deps`
(`lock.rs:186-239`). We bump `EMIT_EPOCH` 6 -> 7 (`lock.rs:297`) because the
algorithm changes resolved versions; that invalidates all committed locks for a
one-time cold re-solve, which is exactly what the G-1 gate measures.

---

## 1. HONEST SCOPE & RISK ASSESSMENT (required, up front)

### Is this a few-hundred-LOC DependencyProvider?

**No.** The `DependencyProvider` trait impl itself is the small part (~300-400
LOC). The bulk of the work and ALL of the risk is in the **async->sync bridge**:
because resolvo cannot do I/O during the solve, retread must build a complete
in-memory Pool of every reachable candidate + its metadata + its conda-routing
decision, BEFORE solving. That pre-pass (discovery + metadata fetch + batched
probe) is itself a graph traversal of comparable complexity to the current BFS --
we are not deleting the BFS so much as repurposing it into a *discovery* pass and
adding resolvo as the *decision* layer on top.

Realistic LOC: provider ~400, discovery/Pool builder ~500-700, probe pre-compute
+ memo ~200, solution->Bundle mapping ~150, A/B harness ~200, incremental
fast-path ~150. **~1600-1800 LOC net new, multi-week.**

### The 3 hardest / riskiest parts

1. **The discovery fixpoint pre-pass (HIGHEST RISK).** resolvo needs the full
   candidate set up front, but the candidate set is only knowable by resolving.
   We break the cycle with a discovery traversal that over-approximates: fetch
   candidates for every name reachable under ANY version in scope, fetch their
   metadata, evaluate markers, discover new names, repeat to fixpoint. This can
   fetch MORE than the final solution needs (over-fetch) -- acceptable for
   correctness, a perf concern for big packs (isaac6). Risk: the discovery pass
   reintroduces the very ordering/confluence concerns we're trying to escape if
   not done carefully. Mitigation: discovery only needs to be a SUPERSET (it
   doesn't pick versions; resolvo does), so it can be naive/order-independent
   (BTreeMap frontier, fetch-all-compatible-versions). It must NOT make
   route-to-conda decisions that prune the graph -- see #2.

2. **The conda boundary inside a complete solver (HIGH RISK).** Today the BFS
   prunes conda-routed subtrees by `continue` (`mod.rs:2982`) -- it never fetches
   their PyPI transitives. In a resolvo model this becomes: `get_dependencies`
   for a parent OMITS the edges to conda-routed children (they're conda's job),
   and those children are recorded as conda run-deps. But the route decision is
   the async, version-spec-dependent probe (`mod.rs:2887`). So the probe results
   for every `(child_name, child_spec)` edge must be in a pre-computed memo table
   before the solve. The discovery pass collects those `(name, spec)` pairs and
   runs `probe_many` (16-way, `probe.rs:344-360`) to fill the memo. Risk: if
   resolvo explores an edge whose `(name, spec)` wasn't in the memo (because
   discovery's spec set differs from what resolvo actually traverses), the
   provider has no answer. Mitigation: discovery must collect the probe-relevant
   spec for EVERY edge it sees, and the memo must be keyed conservatively (probe
   at the parent edge's exact spec AND at "*" as the BFS already does). If a memo
   miss occurs, fail-closed (panic-free error) and fall back to cold BFS for that
   pack -- never silently guess.

3. **incremental == cold via resolvo (MAKE-OR-BREAK, see §D).** Must prove
   empirically that resolvo cold == resolvo warm-seeded-then-add on the example
   corpus. If resolvo's `favored` hint can change the result, incremental!=cold
   and we are back to square one -- the entire motivating goal fails. The plan
   gates PR-4 on a `RETREAD_VERIFY_INCREMENTAL` oracle test that asserts
   byte-identical locks.

### Is anything INFEASIBLE on resolvo 0.10.3?

**No hard showstopper, but two near-showstoppers handled by design:**

- **Async (handled by pre-fetch).** resolvo's pseudo-async means no lazy I/O.
  Feasible via the discovery pre-pass + Pool. This is the rattler_solve pattern
  exactly: `CondaDependencyProvider` interns ALL records before solving and does
  O(1) hashmap lookups in its callbacks (rattler_solve `resolvo/mod.rs:636-885`,
  confirmed by solve_check.rs's pre-fetch via `load_selected_records_sparse`).
  We mirror it for PyPI.

- **Extras (handled by per-(name,extras) solvables).** The current code uses
  Model B (extras expand to extra dep edges; solvable keyed by name only;
  `Pending.extras` is threaded mutable state -- `auto_bundle.rs:588`,
  `pep508_extra_dep` `auto_bundle.rs:868-903`). resolvo's `get_dependencies`
  is **stateless** (takes only a `SolvableId`), so we CANNOT thread extras as
  mutable state. We must adopt the standard resolvo pattern: a solvable is
  `(name, frozenset(extras), version)`. `requests[cuda]` and `requests` are two
  distinct solvables sharing the same wheel artifact but with different dep edge
  sets, and resolvo must force them to the SAME version via a constraint (the
  "extras imply base at same version" clause). This is well-trodden (uv and
  pixi's pypi resolver do exactly this) but is NOT free -- it's the single
  biggest correctness subtlety in the provider. **STOP-if:** if modeling
  extras-as-solvables proves intractable in 0.10.3's API, fall back to expanding
  extras into the discovery pass only (resolve the union of all requested extras
  per name) -- less precise but matches today's behavior. Recommend trying
  per-(name,extras) first; it's the "right" model.

- **The probe-must-be-pre-computed constraint (handled by discovery+memo).**
  Covered in risk #2.

**Conclusion: feasible. No STOP. But the extras model and the discovery
fixpoint are the two places this could balloon, and incremental==cold must be
proven, not assumed.**

---

## 2. CURRENT STATE (what we're replacing) -- cited

The resolution path that produces a `Bundle` (the clean cut line at
`resolve_state.rs:13-14`, the designated seam):

- **`resolve_bundle` BFS** `mod.rs:2568` -- frontier `BTreeMap<String, Pending>`
  (`mod.rs:2602`, name-sorted = "Pillar 3" confluence), 500-level cap
  (`mod.rs:2750`), constraints in `ResolveState` (`resolve_state.rs:59`) via
  `observe_edge` (`resolve_state.rs:99`) with `NeedsReResolve`->`revoke_chosen`.
- **`bfs_fetch_pypi`** `mod.rs:3289` (async, 8-way buffered) -> `pypi::resolve`
  `pypi.rs:46` (highest compatible wheel, no backtracking, `pypi.rs:99-111`) /
  `resolve_sdist` `pypi.rs:127`.
- **4 version-picking sites:** `pypi::resolve` descending sort (`pypi.rs:99`);
  `auto_bundle_transitives` (`auto_bundle.rs:348`); `pre_emit_widen_pass`
  (`cascade.rs:773`, conda/PyPI 8-step cascade + override injection);
  `produce_output` run-dep union (`mod.rs:4007`, no version picking, just
  translate+dedup first-encountered, sorted `mod.rs:4090`).
- **conda boundary:** `pick_conda_target` (`auto_bundle.rs:50-66`, name-only) +
  two-probe routing (`mod.rs:2887-2982`), `continue` at `mod.rs:2982` prunes the
  conda-routed subtree. `retread-conda-deps` user force-list (`config.rs:96`,
  skip set `auto_bundle.rs:118`, cascade `cascade.rs:1486-1489`).
- **Downstream (UNCHANGED by this work):** `produce_output` `mod.rs:4007` ->
  `Bundle` (`mod.rs:521`: `conda_name, primary, extras, probe_decisions,
  solve_diagnostics`) -> `courier::stage` `courier.rs:410` -> `RetreadLock`
  schema 10 (`lock.rs:186-259`) -> `replay_from_lock` `mod.rs:5530` /
  `materialize_from_lock` `mod.rs:4263`.

---

## A. The PyPI DependencyProvider

### Solvable identity

`Solvable = (NameKey, Version)` where **`NameKey = (canonical_name,
frozenset(extras))`**. This is the per-(name,extras) model (§1 hardest-part #3).
- Base solvable `requests@2.31` and extra solvable `requests[socks]@2.31` are
  distinct `NameId`s sharing a version axis.
- A constraint clause forces: if `requests[socks]@V` is selected, `requests@V`
  must be too (same version). resolvo expresses this as a dependency edge from
  the extras-solvable to a version-pinned requirement on the base name.

### `get_candidates(name)`

From the pre-built Pool only (no I/O):
- versions = index-chain wheels (`merge_index_chain` `mod.rs:227`: entry
  indexes -> workspace `[pypi-options]` -> public PyPI, dedup first-wins)
  **filtered to versions that have a target-compatible wheel** via `pick_best` /
  `score_wheel` logic (`pypi.rs:300-311`; `WheelTarget = {python_version,
  conda_subdir}` `pypi.rs:34-40`).
- Git/Url deps: exactly ONE candidate, a synthetic-version sentinel (the SHA /
  URL), no filtering (`PendingSource::Git/Url` `auto_bundle.rs:596-617`;
  frozen, single-artifact).
- `Candidates.locked`: **NEVER set** (this is the soundness guard -- see §D).
- `Candidates.favored`: set to the prior locked version for incremental warm-start
  (soft hint only).
- `Candidates.excluded`: prerelease versions excluded unless a spec explicitly
  admits them (today's behavior: prereleases pass only if `specifiers.contains(v)`;
  no separate prerelease API in 0.10.3).

### `sort_candidates`

Highest-version-first (matches today's `Highest` strategy and `pypi.rs`
descending sort). Deterministic tiebreak by full version then filename.

### `get_dependencies(solvable)`

Synchronous, from pre-built metadata:
1. Take the wheel's `Requires-Dist`.
2. Evaluate PEP 508 markers against the **fixed** `MarkerEnvironment`
   (`marker_env_for(conda_subdir, python_version)` `relax.rs:411`) WITH the
   solvable's active extras (`marker.evaluate(&env, &active_extras)` --
   `auto_bundle.rs:877-879` pattern). Drop edges whose marker is false.
3. For each surviving edge, consult the **pre-computed conda-route memo**
   keyed `(child_canonical_name, child_spec)`:
   - **routed-to-conda** -> OMIT the edge from resolvo's dependency list; record
     `(child_name, child_spec)` in a side-channel `conda_run_deps` accumulator
     keyed by this solvable. resolvo never resolves it.
   - **stays-on-pypi** -> emit a resolvo `Requirement` over `(child_name+extras,
     spec)`.
4. The extras-solvable additionally emits the "same-version base" requirement.

### Async -> sync integration (CONFIRMED APPROACH)

**Pre-resolve discovery pass, then offline solve.** (Mirrors solve_check.rs's
own pattern of `load_selected_records_sparse` + `spawn_blocking` solve.)

```
DISCOVERY (async, over-approximating, order-independent):
  frontier = BTreeMap of (NameKey -> spec-union) seeded from entry specs
  loop to fixpoint:
    for each NameKey: fetch all index-chain versions w/ compatible wheels
                      (pypi::resolve-style listing), fetch METADATA (PEP 658
                      sidecar preferred), parse Requires-Dist
    evaluate markers (fixed env) -> collect child edges (name, spec, extras)
    record every (child_name, child_spec) into the probe-needs set
    add new NameKeys to frontier
  -> in-memory Pool: all candidates + metadata
PROBE PRE-COMPUTE (async):
  probe_many(probe-needs set, 16-way)  [probe.rs:344-360]
  + the "*" name-level second probe per name (mirrors mod.rs two-probe)
  -> HashMap<(name, spec), Route>  memo
SOLVE (sync, spawn_blocking):
  resolvo Solver over the Pool; provider answers from Pool + memo
  -> Ok(Vec<SolvableId>)  |  Err(Unsolvable(conflict))
```

Memo-miss policy: fail-closed -> error -> (during A/B) fall back to BFS for that
pack; never guess a route.

---

## B. resolvo solution -> Bundle (downstream UNCHANGED)

resolvo returns a consistent `(NameKey -> Version)` set. Map it:
- Each selected PyPI solvable -> a bundled wheel. The primary entry's solvable
  -> `Bundle.primary`; the rest -> `Bundle.extras` (`Bundle` `mod.rs:521`).
  Extras-solvables collapse onto their base wheel artifact (same `(name,version)`
  -> one `ResolvedWheel`, union of requested extras into `extras_requested`).
- The accumulated conda-route side-channel -> conda run-deps. NOTE today
  `conda_run_deps` is derived in `produce_output` (`mod.rs:4007`) by translating
  each bundled wheel's `Requires-Dist` and deduping. **Two options:**
  - **B1 (minimal-change, recommended):** still let `produce_output` derive
    `conda_run_deps` from the bundled wheels exactly as today. The resolvo
    solution only changes WHICH wheels are bundled; `produce_output` is
    unchanged. The conda-route memo is the same logic `produce_output`'s
    `translate()` already applies, so the run-dep set is consistent.
  - **B2:** carry resolvo's side-channel run-deps directly. More precise but
    duplicates `produce_output`. Avoid unless B1 diverges.
- **Everything downstream is byte-for-byte unchanged:** `produce_output` ->
  `courier::stage` (`courier.rs:410`, `canonicalize()` `lock.rs:404`) -> schema
  10 lock -> replay/materialize. **Lock format unchanged. inputs_hash domain
  unchanged** (`compute_inputs_hash` `lock.rs:325-362` encodes inputs +
  EMIT_EPOCH only, not the algorithm).

---

## C. Which of the 4 sites resolvo subsumes

- **resolve_bundle BFS** (site 1) -> subsumed (becomes discovery + solve).
- **auto_bundle_transitives** (site 2, `auto_bundle.rs:348`) -> subsumed; its
  exact-pin transitive bundling is just more dependency edges resolvo handles.
- **pre_emit_widen_pass cascade** (site 3, `cascade.rs:773`) -> **PARTIALLY.**
  The "widen until conda-satisfiable" logic is really the conda-routing decision
  in another guise. The 8-step cascade's PyPI steps fold into the
  DependencyProvider's conda-route memo. **BUT the override-injection / spec-
  widening at emit (rewriting an emitted conda spec `>=1.2,<1.3` -> `*`) is a
  POST-resolution METADATA concern, not a resolution concern** -- like relax, it
  applies to the chosen wheels' emitted specs. **Recommendation: keep the
  widen/override emit step as a post-pass over resolvo's solution; only the
  "should this dep route to conda" part folds into the provider.** Confirm with
  the-grizzly: the cascade currently interleaves routing + emit-widening; the
  split must be clean or the widen post-pass could re-route and desync from
  resolvo's solution.
- **produce_output union** (site 4, `mod.rs:4007`) -> becomes trivial / unchanged
  (resolvo already gave one consistent set; B1 keeps produce_output as-is).

**relax policy:** stays a downstream METADATA rewrite, UNCHANGED. resolvo
resolves on ORIGINAL specs; relax applies to chosen wheels' METADATA at emit
(today: `bfs_fetch_pypi` patch-drift `mod.rs:3253` + `rewrite_wheel`; relax does
NOT gate version selection). No change to relax timing.

---

## D. INCREMENTAL == COLD (the make-or-break)

### What resolvo offers

- `Candidates.favored` (SOFT: solver tries it first, **backtracks if it
  conflicts** -- `resolvo-0.10.3/src/lib.rs:156-194`) via
  `SolverTask.locked_packages` (`rattler_solve-4.1.0/src/lib.rs:219-282`,
  mapped at rattler_solve `resolvo/mod.rs:524-539`).
- `Candidates.locked` (HARD: always selected) via `SolverTask.pinned_packages`.
  retread today always passes `pinned_packages = Vec::new()` (`solve_check.rs`).

### The argument

incremental-add = seed resolvo with the prior locked closure as **favored** (NOT
locked), add the one new `entry_spec`, solve.

**incremental == cold IFF resolvo is deterministic AND the favored seed cannot
change the result vs cold.**

- resolvo is a **complete CDCL/PubGrub-style solver** -> for a fixed constraint
  set and fixed candidate ordering, it has a unique "highest-preferred
  consistent solution." It is order-independent by construction (that's the whole
  point of using it -- §E).
- `favored` is a **search-order hint**: it changes which branch the solver
  explores first, NOT which solution is optimal. A complete solver with a
  deterministic objective (highest-version-preferred + deterministic
  `sort_candidates`) returns the SAME optimum whether or not it was hinted --
  the hint only affects how fast it gets there, and it BACKTRACKS off the hint if
  the hint isn't part of the optimal consistent set.
- Therefore cold (no favored) and incremental (favored = prior closure) over the
  SAME constraint set (roots + new spec) converge to the same solution.

**The one way this breaks:** if we used the HARD lock (`pinned_packages` ->
`Candidates.locked`) to pin prior versions, incremental could "stick" to a stale
version that cold would not pick -> incremental != cold. **GUARD: never set
`Candidates.locked` for incremental. Use `favored` only.** (§A enforces this.)

**The second way it could break:** if resolvo's "highest-preferred" optimum is
not actually unique/deterministic in 0.10.3 (e.g. ties resolved by internal
ordering that depends on insertion order). **This MUST be verified empirically,
not assumed.**

### Required verification (gates PR-4)

`RETREAD_VERIFY_INCREMENTAL` oracle test (mirrors the spec in
PHASE3-FOUNDATIONAL-PLAN.md §4.8): for each example pack, (1) cold-solve full
spec set; (2) cold-solve set-minus-one, then incremental-add the one with favored
seeding; assert the two schema-10 locks are **byte-identical**. Plus the
determinism sub-gate: cold-solve each pack TWICE, assert byte-identical
(PHASE3-FOUNDATIONAL-PLAN.md §7.3 clause 4 -- not yet implemented). If either
fails on any pack, **STOP -- incremental!=cold, report to user.**

---

## E. CONFLUENCE + DETERMINISM

resolvo is a complete solver -> **order-independent by construction**; confluence
is free (it's the reason for the swap; replaces the hand-rolled "Pillar 3"
name-sort confluence argument in `resolve_state.rs`).

Determinism requires: deterministic `sort_candidates` (full version + filename
tiebreak), deterministic Pool insertion (BTreeMap-ordered discovery), and the
existing `canonicalize()` emit (`lock.rs:404`). With `Candidates.locked` never
set, no hidden state.

**Empty-solution / real conflict** (e.g. pillow 11.3 vs 12.0): resolvo returns
`Err(Unsolvable(conflict))` -> `conflict.display_user_friendly` (rattler_solve
`resolvo/mod.rs:972`) -> **fail-closed**, surface the conflict tree to the user
(reuse `extract_blocking_chains` `solve_check.rs:503-678`). Never emit a partial
or guessed solution.

---

## F. EMIT_EPOCH BUMP + G-1 GATE

- **EMIT_EPOCH 6 -> 7** (`lock.rs:297`). resolvo changes resolved versions ->
  emit-affecting -> epoch bump invalidates all committed locks
  (`compute_inputs_hash` folds epoch `lock.rs:354`; replay miss `mod.rs:5488` ->
  cold re-solve). No SCHEMA bump (stays 10).
- **G-1 user-accept gate (manual, mandatory, PHASE3-FOUNDATIONAL-PLAN.md §7.3):**
  for every committed `retread-*.lock.json` (the corpus: `examples/genesis/`,
  `examples/gigastrap/isaac-pack/`, `examples/isaac6/isaac-pack/` -- each has a
  committed lock), dump resolved `{name:version}` under resolvo, diff vs current
  BFS lock, classify each delta: **version-change / RED (newly missing or
  conda<->pypi flip) / clean (canonical reorder only)**. REPORT for user
  sign-off. The ONLY auto-accept is canonical reordering.
- **Determinism run-twice** sub-gate (§D).

---

## G. PHASED PR PLAN

**PR-1 -- provider + Pool + discovery pass behind `RETREAD_RESOLVO=1`
(default OFF, runs ALONGSIDE BFS, does NOT replace it).**
New module `src/handler/resolvo_provider.rs`: solvable model (name,extras,ver),
discovery fixpoint pass, probe pre-compute memo, `DependencyProvider` impl,
solution->Bundle mapping. Gated `std::env::var("RETREAD_RESOLVO")` at the seam
(`mod.rs:1919-1922` / `resolve_state.rs:14`). BFS untouched. New env var follows
the existing `RETREAD_*` pattern (`mod.rs:5474` etc.).

**PR-2 -- A/B harness (the pre-probe).** Resolve every example pack BOTH ways,
diff the resolved `{name:version}` + `conda_run_deps`. This is the
user-certainty measurement: *does resolvo change versions vs BFS, and where?*
Output a per-pack classified diff. **No swap yet.** This is where we learn if the
extras model / discovery / conda-memo actually reproduce BFS results.

**PR-3 -- flip resolvo to default + remove BFS, ONLY after G-1 + user accept.**
Bump EMIT_EPOCH 6->7. Re-bless the example locks (with sign-off). Delete the BFS,
`ResolveState`, `bfs_fetch_pypi`, the 4-site machinery, `pre_emit_widen_pass`
routing half (keep widen-emit post-pass per §C). `produce_output` stays (B1).

**PR-4 -- incremental fast-path on resolvo.** Hook at `mod.rs:1919-1922`
(`try_incremental_resolve`): seed resolvo with prior locked closure as
**favored** (never locked), add new entry_spec, solve. Gate on
`RETREAD_VERIFY_INCREMENTAL` oracle (§D) being GREEN on the full corpus. If the
oracle is red, incremental!=cold -> STOP, do not ship the fast-path (cold path
still works; just no speedup).

**PR-5 -- cleanup.** Remove dead BFS scaffolding, flags, docs. Collapse discovery
over-fetch where safe (perf for isaac6).

**Why this phasing:** PR-1+PR-2 let us MEASURE resolvo-vs-BFS on real packs
BEFORE committing to the swap (the user-certainty gate). If A/B shows resolvo
diverges in ways the user won't bless, we abort cheaply without having deleted
the working BFS.

---

## CONFIRMATIONS (as requested)

- **async-sync integration:** pre-resolve discovery pass fetches all candidates +
  metadata into an in-memory Pool; batched `probe_many` fills a conda-route memo;
  resolvo solves OFFLINE inside `spawn_blocking`. No lazy I/O in callbacks (0.10.3
  pseudo-async panics on await). Mirrors solve_check.rs's existing pre-fetch.
- **conda-boundary:** `pick_conda_target` is name-only but the route gate is the
  version-spec probe; both pre-computed into a `(name,spec)->Route` memo during
  discovery. `get_dependencies` OMITS conda-routed edges (records them as
  run-deps); resolvo never pulls conda subtrees. Memo-miss = fail-closed.
- **incremental==cold:** sound IFF resolvo's optimum is deterministic and we use
  `favored` (soft) NOT `locked` (hard). Argued in §D; **must be empirically
  verified** by `RETREAD_VERIFY_INCREMENTAL` before PR-4. This is the only
  unproven claim.
- **A/B pre-probe phasing:** PR-1 (provider behind flag, alongside BFS) + PR-2
  (A/B diff harness) measure before the swap.
- **scope/risk:** ~1600-1800 LOC, multi-week. Top 3 risks: discovery fixpoint
  pre-pass, conda boundary in a complete solver, incremental==cold proof.
  No hard showstopper on 0.10.3.

**Open sub-decisions for the user / the-grizzly:**
1. Extras model: per-(name,extras) solvables (right, harder) vs extras-union in
   discovery (matches today, looser). Recommend the former, fall back to latter.
2. cascade split (§C): confirm widen-emit can cleanly separate from routing.
3. Accept the one-time cold re-solve + re-bless of all committed locks (EMIT_EPOCH
   bump) -- the user has already accepted re-blessing with sign-off.
