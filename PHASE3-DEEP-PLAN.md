# Phase 3 DEEP Rework — True Cross-Level Confluence + Incremental Single-Dep Add

**Target:** pixi-build-retread, branch `dev/incremental-add`, **HEAD 7b89af7** (Part 1 landed).
**Baseline state:** `SCHEMA = 10`, `EMIT_EPOCH = 6`, `entry_specs` persisted, canonical lock
ordering DONE, constraint-accumulating resolver scaffolding (`ResolveState`, `observe_edge`,
`NeedsReResolve`, `MAX_BFS_ITERATIONS`) **built and live but DORMANT**.
**Author:** The Architect. **Auditor (next):** the-grizzly.
**Status:** DESIGN, implementation-ready. All load-bearing claims re-pinned `file:line` against
HEAD 7b89af7 (re-pinned 2026-06-18; the foundational doc's lines pre-date Part 1 and are stale).

> **Reference, not duplication.** `PHASE3-FOUNDATIONAL-PLAN.md` still holds for: canonical ordering
> design (§2 there — DONE), `entry_specs` persistence (§4.2 there — DONE), and the Part-2 incremental
> *algorithm* shape (§5 there). This document covers ONLY the deeper change those plans deferred:
> making cross-level tightening edges actually fire `NeedsReResolve` so the resolver is genuinely
> confluent, then delivering the incremental fast path on top of it.

---

## 0. The wall, precisely (what the swe + grizzly proved)

The confluent resolver is built. `observe_edge` (`src/handler/resolve_state.rs:99-146`) correctly
intersects constraints and returns one of four outcomes
(`ObserveEdgeResult`, `resolve_state.rs:24-39`):

- `New(Pending)` — name not yet in `state.constraints` → enqueue + fetch.
- `AlreadySatisfied` — seen; intersected constraint still contains the chosen version → no refetch.
- `NeedsReResolve(Pending)` — seen; intersection now **excludes** the chosen version → revoke +
  re-resolve.
- `NonPypiAlreadySeen` — non-PyPI source already seen → treat as satisfied.
- `Err(...)` — provably-empty intersection (two conflicting exact pins) → fail-closed.

The BFS loop **already handles all of these** (`src/handler/mod.rs:2770-2796`): the
`NeedsReResolve` arm calls `state.revoke_chosen(name)`, strips the dep from `extras`, pushes the
tighter `Pending` to `reresolve_queue`, and after the level the queue is re-inserted into `work`
(`mod.rs:2793-2796`) and drained next iteration. `MAX_BFS_ITERATIONS = 500`
(`mod.rs:2750`, checked `:2753-2760`) is the fail-closed backstop.

**The defect — why it never fires.** `seed_worklist` (`src/handler/auto_bundle.rs:621-663`) drops a
dependency edge whose canonical name is already in `seen` at **two guards**:
`auto_bundle.rs:635` and `auto_bundle.rs:652` (`if seen.contains(&dn) { continue; }`). Both call
sites construct `seen` as a **snapshot of `state.constraints.keys()`** taken immediately before the
call (`mod.rs:2688-2696` initial seed; `mod.rs:3170-3180` phase-3 per-sub-wheel seed). So a
tightening edge to an already-committed dep is **filtered out of `work` before it ever reaches
`observe_edge`**. The `observe_edge` loop at `mod.rs:2770` therefore only ever sees names absent
from `constraints` → always hits `New` → `AlreadySatisfied`/`NeedsReResolve` are unreachable for
cross-level edges → the resolver is **first-requirer-wins (order-dependent) for cross-level
constraints** → NOT confluent → incremental == cold not soundly deliverable.

G-1 Phase A was green only because **no current real pack has a cross-level conflict** (the
divergence pre-probe found 0). That is luck, not soundness. The deep rework removes the luck.

---

## 1. THE CORE CHANGE — route EVERY edge through `observe_edge`

### 1.1 Principle

`seed_worklist` must stop being the place that decides "skip already-seen." That decision belongs
to `observe_edge` alone (the single arbiter of the three cases). `seed_worklist` becomes a pure
edge *enumerator*: it emits a `Pending` for **every** dependency edge (gated only by markers/extras,
never by `seen`), and `observe_edge` decides New vs Satisfied vs ReResolve vs Conflict.

This is the minimal, surgical change: the machinery to handle the result already exists and is
correct. We only have to stop dropping the edges before they get there.

### 1.2 The three cases, mapped to existing code

| Case | Input | Desired outcome | Mechanism (already built) |
|------|-------|-----------------|----------------------------|
| (a) **NEW name** | edge to name ∉ `constraints` | fetch + commit | `observe_edge` → `New(Pending)` → `frontier.push` (`mod.rs:2773-2775`) |
| (b) **SEEN, COMPATIBLE** | edge to committed name; intersection still contains chosen version | accumulate constraint, **no refetch** | `observe_edge` → `AlreadySatisfied` (`resolve_state.rs` satisfied arm); loop skips (`mod.rs:2776-2778`). Constraint is updated in-place inside `observe_edge` so the tighter spec is recorded for emission. |
| (c) **SEEN, TIGHTENING** | edge to committed name; intersection **excludes** chosen version | revoke + re-resolve subtree | `observe_edge` → `NeedsReResolve(Pending)` → `revoke_chosen` + strip `extras` + `reresolve_queue.push` (`mod.rs:2779-2788`); re-inserted to `work` (`:2793-2796`) |
| (d) **SEEN, in-flight** | edge to name in `constraints` but not yet `chosen` | accumulate only | `observe_edge` already handles: updates constraint in-place, returns `AlreadySatisfied` (no re-enqueue) — the in-flight fetch will pick up the tightened constraint when it resolves (see §1.5) |
| (e) **EMPTY intersection** | two specs whose AND is unsatisfiable | fail-closed error | `observe_edge` → `Err(...)` (§4) |

### 1.3 The exact `seed_worklist` change

**Remove the two seen-filter guards** at `auto_bundle.rs:635` and `auto_bundle.rs:652`. Keep
everything else (marker evaluation, extras gating, `Pending` construction, index resolution).

**But preserve no-refetch / no-duplicate-work** — this is the subtle part the grizzly will probe.
Today the `seen` filter served two jobs: (1) the cross-level skip (the bug), and (2) avoiding a
flood of duplicate `Pending`s for the same name within one seed. Job (2) is still needed for
efficiency but must NOT drop a *tighter* edge. So:

- `seed_worklist` no longer takes `seen` at all. Drop the parameter (and the two call sites'
  `seen_set` construction at `mod.rs:2688-2696` and `mod.rs:3170-3180`). Signature becomes
  `seed_worklist(requires_dist, extras_requested, index, bundle_prefix, work)`.
- Within a single `seed_worklist` call, **dedup by canonical name with constraint-merge** (not
  drop): if the same name appears twice in one `requires_dist` set, intersect their specifiers into
  one `Pending` (this mirrors the existing in-level merge already done at `mod.rs:2707` /
  `mod.rs:3189` via the `BTreeMap::Entry::Occupied` path — reuse that merge helper, do not
  reinvent). Net: one `Pending` per name per seed, carrying the intersected spec.
- All dedup-vs-tighten-vs-skip decisions across the *whole resolve* now happen exactly once, in
  `observe_edge`. Single arbiter. (This also satisfies E.4 from the foundational plan — "unify the
  drop points.")

### 1.4 The BFS-loop control flow (the now-live re-resolve)

The loop at `mod.rs:2752` (`'levels: loop`) needs only one correctness reinforcement beyond what
exists, because re-resolve now actually runs:

1. **Drain `current_work` → `observe_edge`** (`mod.rs:2770-2790`): unchanged — already routes New →
   frontier, Satisfied → skip, ReResolve → revoke + requeue, Conflict → bail.
2. **Re-insert `reresolve_queue` into `work`** (`mod.rs:2793-2796`): unchanged.
3. **Phase-2 concurrent fetch of `frontier`** (`bfs_fetch_pypi`, `mod.rs:2998`): when a name was
   revoked in step 1, it is re-fetched here against its **tightened** accumulated constraint. ✦ NEW
   REQUIREMENT: the re-fetch must read the spec from `state.constraints[name]` (the accumulated
   intersection), **not** from the original `Pending.specifiers`. Today `bfs_fetch_pypi` is handed
   `pending.specifiers` (`mod.rs:2998`). On a re-resolve, the `Pending` pushed by the
   `NeedsReResolve` arm is the `tighter_pending` that `observe_edge` built from the intersection
   (`resolve_state.rs` NeedsReResolve construction) — confirm it carries the intersected specifiers,
   not the raw triggering edge's. **Grizzly check:** verify `tighter_pending.specifiers ==
   intersect(constraints[name], triggering_edge.specifiers)`. If `observe_edge` currently builds
   `tighter_pending` from only the triggering edge, fix it to carry the full accumulated
   intersection (else the re-resolve under-constrains and the fixpoint is wrong).
4. **Phase-3 per-sub-wheel: commit + recurse-seed** (`commit_chosen` at `mod.rs:3230`,
   `seed_worklist` at `mod.rs:3173`): when a re-resolved wheel commits a **different version**, its
   `Requires-Dist` may differ → its children edges are re-seeded (now WITHOUT the seen-filter) →
   any of those children that tighten an already-committed grandchild fire `NeedsReResolve` in the
   next level. This is the cascade, and it is correct (see §2).

### 1.5 In-flight tightening (case d) — the one genuinely new edge case

When a name is in `constraints` but not yet in `chosen` (it is queued/fetching in the current
level), a tightening edge must update the constraint **before** that name's fetch picks its
version, or the fetch will pick a version the tightening edge excludes. Two safe designs; pick the
simpler:

- **Design A (recommended): observe-all-before-fetch within a level.** The loop already observes
  the entire `current_work` (step 1) *before* the phase-2 fetch (step 3). So all
  same-level tightening edges land in `constraints[name]` before that name is fetched, as long as
  the fetch reads from `constraints[name]` (the §1.4 step-3 requirement) rather than the per-pending
  spec. This makes case (d) collapse into "fetch against the final accumulated constraint" with no
  extra machinery. **This is why §1.4 step 3 reading from `constraints` is load-bearing, not
  cosmetic.**
- Design B (rejected): allow in-flight re-fetch. More complexity, no benefit given A.

**Decision: Design A.** The invariant to enforce + test: *a name is fetched against
`state.constraints[name]` (the fully accumulated intersection at the moment of fetch), never against
a single requirer's raw spec.*

### 1.6 The other three version-pick sites (must also route through `observe_edge`/accumulation)

Confluence requires ALL FOUR pick sites honor accumulation, not just `resolve_bundle`. Re-pinned at
HEAD 7b89af7:

- **Site 1 — `resolve_bundle`/`bfs_fetch_pypi`** (`pypi::resolve` at `mod.rs:3305`, relaxed retry
  `:3309`, fetch dispatch `:2998`): covered by §1.1-1.5.
- **Site 2 — `auto_bundle_transitives`** (`pypi::resolve` at `auto_bundle.rs:445`): runs its own
  fixpoint over already-bundled wheels. It must observe its edges through the SAME `ResolveState` /
  accumulation, not a private first-committed loop. **Design:** thread the resolve's `ResolveState`
  (or a freshly-derived one seeded from the bundle's wheels' `requires_dist`) into
  `auto_bundle_transitives` and route its candidate selection through `observe_edge`. If that is too
  invasive for this PR, the fallback is: have `auto_bundle_transitives` intersect all requirers'
  specs for each transitive name *before* its single `pypi::resolve` call (a local accumulation),
  which yields the same version as the confluent resolver for that name. **Grizzly check:** prove
  site 2's pick == site 1's pick for any name reachable from both.
- **Site 3 — `pre_emit_widen_pass` → `try_pypi_bundle`** (`pypi::resolve` at `cascade.rs:1146`):
  same treatment — its PyPI bundling must select against the accumulated intersection. The conda
  *widening* half is confluent iff its input bundle is (it consumes the bundle, doesn't pick PyPI
  versions independently beyond `try_pypi_bundle`).
- **Site 4 — `produce_output` union** (`mod.rs:4079-4090` sort + first-encountered insert at
  `:4142`): emit the conda spec from `state.constraints[name]` (the intersection) in canonical-name
  order, so emission agrees with resolution. After §1, `chosen`/`constraints` is the single source
  of truth; the first-encountered dedup at `:4142` becomes "emit the accumulated spec for each
  name," deleting the order-dependence rather than sorting around it.

---

## 2. TERMINATION UNDER REAL LOAD (re-resolve now runs)

### 2.1 The monotonicity argument

Define, per canonical name `n`, the accumulated constraint `C(n)` = AND-intersection of every
observed requirer's specifiers. `observe_edge` only ever **intersects** into `C(n)`
(`resolve_state.rs:99-146`); it never widens. So `C(n)` is **monotonically non-increasing** in
feasible-version-set over the resolve's lifetime.

The chosen version `V(n)` is the max version satisfying `C(n)` (via `pypi::resolve` against the
intersected spec). Since the feasible set only shrinks, `V(n)` is **monotonically non-increasing**
over re-resolves of `n`.

### 2.2 Bounding total re-resolves

- The set of reachable names is **finite** (bounded by the transitive closure of the roots over the
  index — a finite DAG of packages).
- Each name `n` has a **finite** set of available versions on the index.
- A re-resolve of `n` fires only on a *strict* tightening that excludes the current `V(n)`, and it
  picks a **strictly lower** `V(n)` (or fails-closed if none feasible). So `n` can be re-resolved
  **at most (number of distinct versions of `n`)** times — each re-resolve strictly descends the
  finite version ladder for `n`.
- Total re-resolves ≤ Σ over names of (versions of that name) — a finite bound.

### 2.3 The cascade (re-resolve of X tightens an already-chosen Y)

Yes, this happens and the plan must own it: re-resolving X to a lower version can change X's
`Requires-Dist`, introducing a tighter edge to an already-chosen Y → Y re-resolves → Y's new
version can tighten Z, etc. **It still converges:**

- Each cascade step is itself a strict tightening of *some* name's `C(·)` (else it wouldn't fire
  `NeedsReResolve`). By §2.1 every name's constraint is monotone-shrinking and every name's chosen
  version is monotone-descending.
- The global potential function Φ = Σ over names of (index-rank of `V(n)`) — i.e. how far down each
  name's version ladder we are — **strictly decreases** on every re-resolve (some name descends; no
  name ever ascends because constraints never widen). Φ is bounded below by 0. A strictly-decreasing
  integer-valued function bounded below **terminates**. ∎
- Edge case: a re-resolve that introduces a *brand-new* name (X's lower version pulls a new dep).
  That adds a name to the finite reachable set but cannot create an infinite chain, because the
  reachable set is still the finite transitive closure over the index, and each new name can only
  descend its own finite ladder. Φ is redefined over the (still finite) reached set; adding a name
  adds a finite, one-time bounded amount to Φ, after which the descent argument applies.

### 2.4 Backstop

`MAX_BFS_ITERATIONS = 500` (`mod.rs:2750`) stays as the fail-closed safety net: if the convergence
argument is ever violated by a bug (e.g. a non-monotone constraint), the resolver **bails with a
clear error** (`mod.rs:2753-2760`) naming a "circular re-resolve in constraint accumulation"
rather than hanging. This is the certainty-bar backstop: a wrong resolver fails loudly, never
silently. Keep the cap; the §2.3 proof says it should never be hit on real packs (the grizzly
should treat any cap-hit in G-1 as a STOP, not a tune-the-cap).

---

## 3. CONFLUENCE PROOF (now genuine, not aspirational)

### 3.1 The claim

With (i) every edge routed through `observe_edge` (§1), (ii) intersect-only accumulation (monotone),
(iii) re-resolve-on-tighten that descends to the unique max-feasible version, (iv) name-keyed work
ordering, and (v) the §2 termination, the resolution `R(roots, index) -> {name: (version, source)}`
is the **unique fixpoint** of the constraint system: the assignment where every name's version is
the max satisfying the AND of all its in-closure requirers' specs. Uniqueness + termination ⇒ `R`
is **order-independent** (confluent): permuting discovery order, level batching, or fetch
concurrency cannot change the result, because all of them converge to the same unique fixpoint.

### 3.2 Why it's now true (vs. Part 1's "ish")

Part 1 was non-confluent *only* because cross-level tightening edges were dropped (§0), so the
fixpoint the resolver reached depended on which requirer arrived first. With those edges no longer
dropped, the resolver reaches the fixpoint that honors *all* constraints regardless of arrival
order. The remaining order-sensitivity (which name is processed first within a level) cannot change
the final assignment because re-resolve repairs any premature pick — a name picked too high by an
early requirer is revoked when the later tighter requirer's edge is observed (§1.2 case c).

### 3.3 How the e2e tests it (the decisive empirical seal)

- **Shuffled-discovery-order determinism harness** (unit): seed the resolver with the same roots but
  permute the order edges are enqueued (shuffle `requires_dist` order, shuffle level batching) N×;
  assert identical `chosen` map AND identical canonical lock bytes every time. **Must include a
  CROSS-LEVEL conflict fixture** (the case Part 1 could not handle): name `C` committed at level 1
  by a loose requirer, then a level-2 sibling requires `C` strictly lower — assert `C` is re-resolved
  to the lower version regardless of permutation, and identical across all permutations.
- **Run the harness across all four pick sites' code paths** (§1.6), not just `resolve_bundle`.
- The G-1 determinism sub-gate (§5, §8) re-confirms confluence on *real* packs by producing each
  lock twice and asserting byte-identity.

---

## 4. EMPTY-INTERSECTION = REAL CONFLICT (the pillow 11.3/12.0 case)

### 4.1 Behavior

When `observe_edge` intersects two specifiers into a provably-empty set (e.g. `==11.3` ∧ `==12.0`,
or `<11.4` ∧ `>=12.0`), it returns `Err(...)` (`resolve_state.rs` empty-intersection arm) and the
resolve **fails-closed**. The resolver MUST NOT silently pick one side (that is exactly the unsound
first-requirer-wins behavior we are removing). The pack genuinely cannot resolve as specified — this
is a **USER decision**, not an auto-fix.

### 4.2 The error must be actionable

The fail-closed error must name:
- the offending canonical package name (`pillow`),
- the two (or more) conflicting requirers (the wheels whose `Requires-Dist` produced the conflicting
  specs) and their specs (`isaacsim-X requires pillow==11.3`, `isaacsim-Y requires pillow==12.0`),
- a suggested remedy: add a `[build.config.overrides]` pin or relax one requirer.

`observe_edge` currently returns `Err` on conflicting *exact* pins. **Extend it** to also detect
empty intersection on range specs (not just exact==exact) and to carry the requirer provenance in
the error (today `observe_edge` may not know *which* wheel supplied the prior spec — thread a
requirer label into the edge so the error can name both sides). **Grizzly check:** confirm the
error message contains both requirers + both specs; a bare "unsatisfiable" is insufficient for a
user decision.

### 4.3 G-1 / e2e surfacing

A pack that goes RED on empty intersection is reported (not silenced) in the G-1 dump (§5): the
output lists it under "RED — manufactured/exposed conflict" with the full requirer detail, for the
user to decide. This is an expected, correct outcome of honoring constraints that were previously
dropped — it means the old lock was unsound for that pack.

---

## 5. THE MANDATORY GATE — fresh G-1 on the DEEPER resolver, version changes REPORTED TO USER

> The user authorized (b) **knowing versions may change.** Because the deeper resolver honors
> constraints currently dropped, it WILL likely change some resolved versions vs the committed
> locks — and the new versions are the **sound** ones (the old locks relied on dropped constraints).
> This gate's job is to **surface every change for explicit user sign-off**, never to auto-bless.

### 5.1 The dump mechanism

Rebuild the throwaway `RETREAD_DUMP_RESOLVED` pattern (a resolve-only-exit that prints
`{canonical_name: version}` for the full resolved closure and exits before materialization). Wire it
at the end of `resolve_bundle`/the cascade, gated on the env var, dumping JSON to stdout or a
sidecar file. (Reuse the prior probe shape; it was reverted as commit `52402dd` — reconstruct, do
not hunt for it.)

### 5.2 The diff procedure (every committed lock, BOTH repos)

Per the foundational G-1 scope (widened): enumerate **every** committed lock, not just 7 names:

```bash
find /home/garylvov/projects/pixi-build-retread -name 'retread-*.lock.json' -not -path '*/.pixi/*'
find <pixidock_template>                         -name 'retread-*.lock.json' -not -path '*/.pixi/*'
```

For each enumerated lock:
1. Extract its current committed `{name: version}` (from `lock.wheels[].{name,version}`).
2. Cold-resolve the same manifest under the DEEPER resolver with `RETREAD_DUMP_RESOLVED=1` →
   new `{name: version}`.
3. Diff. Classify into:
   - **(i) version changes:** names whose resolved version differs (old → new), with the requirer
     that caused the tightening if determinable.
   - **(ii) RED packs:** empty-intersection failures, with the §4.2 requirer detail.
   - **(iii) unchanged** (only canonical reorder): the expected-clean case.

### 5.3 The output + the sign-off rule (the hard gate)

Produce a single report:
```
PACK / ENV / LOCK            STATUS    DETAIL
examples/genesis ...         CLEAN     (reorder only)
pixidock/isaac-pack ...      CHANGED   pillow 11.3 -> 11.3 (unchanged), numpy 1.26.4 -> 1.26.2 (tightened by foo>=..,<1.26.3)
pixidock/isaac-latest ...    RED       pillow: isaacsim-A==11.3 ∧ isaacsim-B==12.0 unsatisfiable -> needs override
...
```

**Rule (carried hard constraint):**
- **Nothing is pushed/merged/published/committed** (neither repo's locks) until this report is
  produced AND **the user explicitly accepts the version changes** in (i) and decides on each RED in
  (ii).
- **Do NOT auto-accept** re-blessed locks. A changed version is a sound correction, but it is the
  user's call to adopt it (it may change runtime behavior of their packs).
- A RED pack is a STOP for that pack until the user adds an override/pin; the resolver does not
  guess.
- Determinism sub-gate (still mandatory): produce each lock **twice**; the two dumps must be
  identical. A difference = residual non-confluence at some site (§1.6) = STOP, do not ship.

This is the difference from foundational-G-1: there, "any changed version = STOP/regression." Here,
under user-authorized (b), "any changed version = REPORT + await explicit user accept." Both forbid
silent shipping; (b) adds a human accept step for the sound changes.

---

## 6. PART 2 — incremental fast-path on the now-confluent resolver

(Algorithm shape per `PHASE3-FOUNDATIONAL-PLAN.md` §5; soundness now actually holds because §3 makes
cold resolve a unique fixpoint.)

### 6.1 Delta detect (narrow, fail-closed)

Behind `RETREAD_INCREMENTAL=1` (default OFF). At the cold-resolve kickoff (the `resolve_all` hook,
foundational §4.3 — re-pin the current `conda_build_v1` decision site and the `resolve_all` call at
implementation time; the function moved with Part 1), attempt incremental **only** when ALL hold:
1. committed lock parses, `lock.schema == SCHEMA` (10);
2. current inputs ≠ `lock.inputs_hash` (else plain replay wins);
3. current `courier_input_specs` vs `lock.entry_specs` (now persisted, §0) differ by **exactly one
   added spec, zero removed/modified**; every other inputs_hash component identical (index_urls,
   relax, python, emit_epoch — now **7**, pin, config_fingerprint) **plus `prerelease`
   byte-identical** (prerelease is not in inputs_hash);
4. the added entry is a plain PyPI-form add (not git/url/path).
Else → full cold resolve.

### 6.2 Resolve only the delta, seeded from the prior closure

Reconstruct the prior `ResolveState` from the committed lock: `chosen` from `lock.wheels[].{name,
version, source}`; `constraints` **re-derived in-memory** by scanning each locked wheel's persisted
`requires_dist` (no new schema field — foundational §4.2 surface trim stands). Seed the confluent
resolver with that state, then observe the new dep's edges. The fixpoint touches only:
- a brand-new name → fetch its subtree;
- an existing name the new edge **tightens past** its chosen version → `NeedsReResolve` → re-resolve
  that subtree (the same machinery as §1, now reused warm);
- an existing name still satisfied → `AlreadySatisfied`, no fetch.

### 6.3 Re-validate the merged closure + byte-identity invariant

After convergence, the merged closure is — **by the §3 unique-fixpoint property** — identical to a
full cold resolve of the N+1 manifest. Serialize through the same `canonicalize` (Part 1, DONE) →
**byte-identical lock.** Fail-closed to full resolve on ANY doubt (empty intersection, git/url
edge, multi-delta, marker mismatch, cap hit, prerelease drift). There is no path that emits an
incremental result not equal to cold.

### 6.4 Oracles

- **Shipped fast path:** `RETREAD_INCREMENTAL=1` only — incremental, no verification overhead.
- **Test-only oracle:** `RETREAD_VERIFY_INCREMENTAL=1` — run incremental, then full cold, assert
  byte-identical; error loudly on mismatch. Off in production (it defeats the speedup). This is the
  e2e parity instrument + permanent CI tripwire.

---

## 7. EMIT_EPOCH / SCHEMA

- **`EMIT_EPOCH 6 → 7`** (`src/lock.rs:267`, currently 6). The deeper resolver **changes emitted
  output** for any pack with a cross-level constraint (it honors edges previously dropped → may pick
  different sound versions → different wheels/specs emitted). This is emit-affecting → epoch bump.
  The bump invalidates all old locks' replay via the epoch in `compute_inputs_hash`
  (`src/lock.rs`), forcing a clean cold re-resolve under the deeper resolver.
- **`SCHEMA` stays 10.** `entry_specs` is already persisted (Part 1); §6.2 re-derives `constraints`
  in-memory; no new persisted field is required. (If §1.4-step-3 / §4.2 implementation uncovers a
  concrete need for persisted pre-relax `requires_dist_original`, add it under a 10→11 bump in the
  SAME PR — but the design does not need it: re-resolve reads live metadata, and relax widens
  versions not names.)
- **All committed locks regenerate** at the re-baseline (§5), gated behind the user-accept of §5.3.

---

## 8. PR SEQUENCE + GREEN BAR + VERIFICATION GATES

Toolchain: `PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH" cargo …` (NO host
cargo). Green bar after **every** commit = all three:
```
cargo test --lib
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```
Commit on `dev/incremental-add` only. **Nothing pushed/merged/published until G-1 (§5) + incremental
e2e (gate D below) pass AND the user signs off on §5.3 version changes.**

| PR | Lands | Default behavior change? | Green bar + gate before next |
|----|-------|--------------------------|-------------------------------|
| **PR-A (core, §1)** | Drop `seed_worklist` seen-filter (`auto_bundle.rs:635/652`) + its param; in-call constraint-merge dedup; fetch reads `state.constraints[name]` (§1.4 step 3); confirm/ fix `tighter_pending` carries full intersection (§1.4 step 3 check). | YES (cross-level edges now honored) | 3 green + §3.3 shuffled-order determinism harness incl. cross-level fixture |
| **PR-B (sites 2-4, §1.6)** | Route `auto_bundle_transitives` (`auto_bundle.rs:445`), `try_pypi_bundle` (`cascade.rs:1146`), `produce_output` union (`mod.rs:4142`) through accumulation. | YES | 3 green + determinism harness across all 4 sites |
| **PR-C (conflict UX, §4)** | Extend `observe_edge` empty-intersection detection to ranges + thread requirer provenance into the actionable error. | YES (clearer errors; same resolve) | 3 green + unit: range-conflict fixture errors with both requirers named |
| **PR-D (epoch, §7)** | `EMIT_EPOCH 6→7`. | YES (invalidates replay) | 3 green |
| **G-1 GATE (§5)** | Rebuild `RETREAD_DUMP_RESOLVED`; dump + diff EVERY lock both repos; determinism run-twice; produce the §5.3 report. | — | **HARD: produce report; user accepts version changes + decides each RED; do NOT proceed otherwise** |
| **PR-E (incremental, §6)** | `try_incremental_resolve` behind `RETREAD_INCREMENTAL=1` (default OFF); re-derive prior state; warm fixpoint; fail-closed; `RETREAD_VERIFY_INCREMENTAL=1` oracle. | **NO (flag off ⇒ inert)** | 3 green + delta-detector unit tests + gate D |
| **PR-F (cleanup)** | E-leftovers (wire/remove solve-check seam, etc.). | NO | 3 green |

### Exact verification gates

- **(A) Confluence unit (§3.3):** shuffled-discovery-order → identical `chosen` + identical canonical
  bytes, N×, **including a cross-level conflict fixture** (the thing Part 1 failed); across all four
  pick sites.
- **(B) Termination unit (§2):** re-resolve-on-tighten fixture (loose-then-tight == tight-from-start);
  cascade fixture (X re-resolve tightens Y → both descend, converges); cap-not-hit on a deep chain;
  empty-intersection → fail-closed error with requirer detail (§4).
- **(C) G-1 fresh dump (§5):** every committed lock, both repos, dumped + diffed under the deeper
  resolver; produced twice (determinism); report of version-changes + REDs; **user sign-off
  required** before any push/commit.
- **(D) Incremental == cold e2e (§6):** on **genesis (light) + ≥1 isaac pack** — cold N, cold N+1,
  incremental N→N+1; assert `lock_inc_N1` byte-identical to `lock_cold_N1`; `RETREAD_VERIFY_INCREMENTAL=1`
  passes; back-pressure fixture (added dep tightens an existing pin) re-resolves and matches cold.
- **(E) Replay-e2e seal:** lukewarm all-caches-nuked replay on genesis + ≥1 isaac under epoch-7:
  `build_v1` replay fires, derivation=0, wheels from empty, `git diff --exit-code` clean, env imports.
  Includes M-1 multi-entry isaac (carrier election survives canonical reorder).
- **(F) Determinism CI gate:** every e2e pack produced twice in one job → byte-identical (standing
  enforcement of §3).

---

## 9. Pre-empting the grizzly (holes I expect to be probed)

1. **"`tighter_pending` under-constrains."** Addressed §1.4 step 3 — must carry the full accumulated
   intersection, not just the triggering edge. Flagged as a mandatory grizzly check + a PR-A test.
2. **"In-flight name fetched before tightening lands."** Addressed §1.5 Design A — observe-all-
   before-fetch within a level + fetch reads `constraints[name]`. Invariant + test specified.
3. **"Re-resolve cascade loops forever."** Addressed §2.3 — strictly-decreasing integer potential Φ
   bounded below; `MAX_BFS_ITERATIONS` backstop; cap-hit = STOP not tune.
4. **"Sites 2/3/4 still order-dependent → not confluent."** Addressed §1.6 + PR-B + the
   all-four-sites determinism harness; the G-1 run-twice on real packs is the empirical seal.
5. **"Removing the seen-filter floods `work` / perf regression."** Addressed §1.3 — in-call
   constraint-merge dedup keeps one `Pending` per name per seed; `observe_edge` does global dedup.
   No flood; the same total edges are observed, just not pre-dropped.
6. **"Empty-intersection error is unactionable."** Addressed §4.2 — must name both requirers + both
   specs + a remedy; PR-C test asserts it.
7. **"Version changes shipped without user knowing."** Addressed §5.3 — hard user-accept gate; no
   auto-bless; both repos held.
8. **"Incremental ≠ cold despite the proof."** Addressed §6.3 + the `RETREAD_VERIFY_INCREMENTAL`
   oracle + gate D byte-identity; fail-closed on any doubt.
9. **"Epoch bump breaks pixidock replay on a fresh clone."** Expected + handled by re-baseline (§5,
   §7) gated behind user accept; the replay-e2e seal (gate E) proves epoch-7 locks replay
   byte-identically before any pixidock commit.

---

## 10. Summary for the auditor

- **`seed_worklist` routing design:** delete the two seen-filter guards (`auto_bundle.rs:635/652`)
  and the `seen` parameter (call sites `mod.rs:2688-2696`, `:3170-3180`); `seed_worklist` becomes a
  pure edge enumerator with in-call constraint-merge dedup; ALL skip/tighten/conflict decisions move
  to the single arbiter `observe_edge` (`resolve_state.rs:99-146`), making `NeedsReResolve`
  (`mod.rs:2779-2788`) reachable. Fetch must read the accumulated `state.constraints[name]`
  (`mod.rs:2998`), and `tighter_pending` must carry the full intersection.
- **Termination:** constraints intersect-only (monotone, `resolve_state.rs`); each re-resolve
  strictly descends a finite version ladder; global potential Φ (Σ version-ranks) strictly decreases,
  bounded below → terminates; cascade converges by the same Φ; `MAX_BFS_ITERATIONS=500`
  (`mod.rs:2750`) fail-closed backstop.
- **EMIT_EPOCH bump:** `6 → 7` (`src/lock.rs:267`) — deeper resolver is emit-affecting; SCHEMA stays
  10 (`entry_specs` already persisted; `constraints` re-derived in-memory).
- **G-1-surfaces-to-user gate:** `RETREAD_DUMP_RESOLVED` dump + diff of EVERY committed lock in BOTH
  repos under the deeper resolver, produced twice (determinism); report version-changes (i) + REDs
  (ii); **user must explicitly accept the sound version changes and decide each RED before ANY
  push/merge/publish/commit.** No auto-bless. RED = empty intersection = real conflict, fail-closed
  with named requirers (§4).
- **Confluence + incremental:** all four pick sites route through accumulation → unique fixpoint →
  order-independent → incremental (warm-seeded from the lock, fail-closed) is provably byte-identical
  to cold; behind `RETREAD_INCREMENTAL=1` (default OFF) with the `RETREAD_VERIFY_INCREMENTAL=1` test
  oracle.

**HARD CONSTRAINT (carried):** nothing pushed/merged/published/committed until G-1 (§5) + incremental
e2e (gate D) pass AND the user has signed off on the version changes G-1 surfaces. Work stays on
`dev/incremental-add`. If any site stays non-confluent, the cap is hit, or the user declines the
changes → STOP, report honestly, do not ship.
