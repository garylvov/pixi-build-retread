# Phase 3 Foundational Rework — Sound, Confluent Resolution + Incremental Single-Dep Add

**Target:** pixi-build-retread, branch `dev/incremental-add` (off `courier` b09f70a).
**Current baseline (re-pinned 2026-06-17):** v2.8.0, `SCHEMA = 9` (`src/lock.rs:235`),
`EMIT_EPOCH = 5` (`src/lock.rs:267`).
**Author:** The Architect. **Auditor:** the-grizzly.
**Status:** IMPLEMENTATION-READY (grizzly re-review amendments folded). Every load-bearing claim is
re-pinned `file:line` against the CURRENT tree (the prior draft cited the v2.6.0 / schema-8 /
epoch-4 tree; all those line numbers were stale and have been corrected here — see §1).

---

## ⊕ PRE-PROBE RESULT — GREEN LIGHT (2026-06-17)

The grizzly's mandated **cheap divergence pre-probe** ran across all 7 packs (examples
`genesis` / `isaac6`[isaacsim 6] / `gigastrap`[isaacsim 5.1 dense]; pixidock `genesis` / `newton` /
`isaac` / `isaac-latest`). **REAL DIVERGENCES = 0 on every pack** (probe backend confirmed via the
resolve-only-exit marker on each). Throwaway probe commit `52402dd` is **reverted**; branch
`dev/incremental-add` is back at `b09f70a`, clean.

**What the probe proves (and what it does NOT):**
- ✅ The current first-requirer-wins BFS never picks a version that a *dropped* requirer's spec would
  exclude. So **the existing committed locks already equal the confluent fixpoint** at the place
  divergence would originate (**site 1**, `resolve_bundle` / `bfs_fetch_pypi`, and the Phase-3 edge
  back-pressure case). ⇒ the rework will NOT change resolved versions ⇒ **G-1's "changed version =
  STOP" is expected to PASS.** This is the green light to implement.
- ⚠️ **Necessary, not sufficient.** The probe instruments the version-selection origin (site 1).
  Confluence at **sites 2/3/4** (`auto_bundle_transitives`, the cascade widen `try_pypi_bundle`,
  and the `produce_output` union) is proven **only by the final G-1 shadow-resolve (§7.3), which
  remains the seal.** The probe lowers the risk; G-1 confirms it. The HARD CONSTRAINT (§7.6) stands:
  nothing ships until G-1 + the incremental e2e prove no version change on any committed lock and
  incremental lock == cold lock byte-identical.

> **The user's bar:** "do not push until we're certain it works." Correctness `>>` speed. A
> wrong-but-fast result is unacceptable. The win is seconds-scale (skips DERIVATION, not wheel
> materialization). Therefore this plan is structured so that **the certainty gates can fail and
> we cleanly DO NOT SHIP** with zero residual risk — the incremental path is opt-in
> (`RETREAD_INCREMENTAL=1`, default OFF), and the only default-on change (Part 1) is reversible up
> to the moment of an epoch bump that is itself gated behind a mandatory all-packs shadow-resolve.

---

## 0. The one-sentence problem and the two-part split

Adding ONE dependency to a pack should reuse the committed locked closure and resolve only the
delta — **and produce a lock BYTE-IDENTICAL to a full cold resolve of the same final manifest.**

This is split into two parts with very different blast radii, and the split is the central safety
mechanism:

- **PART 1 — DEFAULT, emit-affecting (the bridge).** Make a full resolve **confluent**
  (order-independent + constraint-accumulating) at all four version-picking sites, and make the
  lock **canonical** (input-order-independent serialization, including nested structures). Part 1
  changes emitted bytes and therefore bumps `EMIT_EPOCH 5→6` and `SCHEMA 9→10`. It re-baselines
  every committed lock. **This is where all the risk lives**, and it is gated behind the mandatory
  G-1 shadow-resolve (§7.3).
- **PART 2 — OPT-IN, inert by default (the incremental fast path).** Behind
  `RETREAD_INCREMENTAL=1` (default OFF). Detects a single-spec add, seeds the confluent resolver
  from the lock, fetches only the delta's subtree, re-validates the merged closure, and
  fail-closes to a full cold resolve on any doubt. **Inert when the flag is off** — it cannot
  affect any current behavior until explicitly enabled, and even when enabled it can only *decline*
  and defer to cold, never emit a result that differs from cold.

**Why the split is the safety story:** Part 2 is provably equal to cold *only because* Part 1
makes a cold resolve a well-defined, order-independent fixpoint. If Part 1's confluence cannot be
achieved or the G-1 gate turns any pack red, we **ship nothing** — Part 2 never turns on, and we
do not perform Part 1's epoch bump. (See §3.5 — the honest make-or-break.)

---

## 1. Architecture facts this plan rests on (ALL re-pinned `file:line`, current tree)

> Every reference below was re-verified against `dev/incremental-add` on 2026-06-17. Where the
> prior draft was wrong, the **OLD (stale)** value is noted so the grizzly can see the drift.

### 1.1 The PyPI BFS resolver — order-dependent, first-requirer-wins

- `resolve_bundle` — `src/handler/mod.rs:2564` (signature ends `-> Result<Bundle>` at
  `src/handler/mod.rs:2595`). **OLD doc: 2591.** It does **not** call `pypi::resolve` directly; it
  dispatches PyPI-pending deps to `bfs_fetch_pypi` at `src/handler/mod.rs:2915`.
- Worklist state locals:
  - `seen: HashSet<String>` — `src/handler/mod.rs:2597` — holds **canonical conda names**
    (`canonical_conda_name(...)`). **OLD: 2593.**
  - `work: VecDeque<Pending>` — `src/handler/mod.rs:2598`. **OLD: 2594.**
  - `extras` accumulator — `src/handler/mod.rs:2694`. **OLD: 2690.**
  - `probe_decisions` — `src/handler/mod.rs:2602`.
- Primary canonical name seeded into `seen` — `src/handler/mod.rs:2638`
  (`seen.insert(canonical_conda_name(&primary.pypi_name));`). **OLD: 2634.**
- Worklist seeded from the primary wheel's **pre-relax original** `Requires-Dist`:
  `seed_worklist(&primary_original_rd, &entry.extras, &entry.index_url(), &prefix, &seen, &mut work)`
  — `src/handler/mod.rs:2678-2685`. `primary_original_rd` is the 2nd element returned by
  `materialize_and_rewrite` and is bound at `src/handler/mod.rs:2619`. **OLD: 2672-2674.**
- `Pending` struct — `src/handler/auto_bundle.rs:576-582`: `pypi_name: String`,
  `source: PendingSource`, `extras: Vec<String>`. **OLD: 572-578.**
- `PendingSource` enum — `src/handler/auto_bundle.rs:589-610`:
  - `Pypi { specifiers: VersionSpecifiers, index: String }` — `:591-594` (this is where a
    requirement's version constraint lives; **never merged with any other requirer's**).
  - `Git { url, rev: Option<String>, subdirectory: Option<String> }` — `:602-607`.
  - `Url { wheel_url: url::Url }` — `:609`.
  - **OLD doc: 587-590 / 598-605.**
- **DROP POINT #1 (seed_worklist):** `seed_worklist` spans `src/handler/auto_bundle.rs:614-656`.
  Two `if seen.contains(&dn) { continue; }` guards at `:628` (extras-gated branch) and `:645`
  (base-dep branch). A requirement edge whose canonical name is already in `seen` is silently
  discarded; its `specifiers` are never intersected. **OLD: 624 / 641; span 610-652.**
- **DROP POINT #2 (BFS drain):** `src/handler/mod.rs:2711-2716`:
  ```rust
  while let Some(pending) = work.pop_front() {        // 2711
      let dep_conda_name = canonical_conda_name(&pending.pypi_name);
      if !seen.insert(dep_conda_name) {               // 2713 — false => already seen => DROP
          continue;                                   // 2714
      }
      frontier.push(pending);
  }
  ```
  **OLD: 2707-2713.**
- Level loop `'levels: loop {` — `src/handler/mod.rs:2706`. **OLD: 2702.** Prefer-conda name-map +
  channel-probe routing block spans `src/handler/mod.rs:2750-2895` (`pick_conda_target` at `:2758`,
  `crate::probe::probe` at `:2810`). **OLD: "2720+".**
- `bfs_fetch_pypi` — `src/handler/mod.rs:3170-3177`, returns
  `Result<(url::Url, WheelMetadata, String, Option<SdistProv>)>`. Calls
  `pypi::resolve(index, pypi_name, specifiers, target)` at `:3186` (primary) and a relaxed retry at
  `:3190`. **OLD: 3120 / "~3116".**
- **Conclusion (confirms grizzly P1):** the chosen version of every transitive is a pure function
  of *which requirer reached it first in BFS order*. This is the make-or-break defect; §3 repairs
  it.

### 1.2 The FOUR version-picking sites (grizzly amendment **B** — all re-pinned)

> Amendment **B** is folded in: confluence must cover ALL FOUR sites, not just `resolve_bundle`.
> If any one stays order-dependent, the full resolve is not confluent and incremental ≠ cold.
> Henry re-confirmed all four exist and all four are order-dependent today.

| # | Site | `file:line` | Picks | Order-dependent today? |
|---|------|-------------|-------|------------------------|
| **1** | `resolve_bundle` BFS → `bfs_fetch_pypi` | dispatch `mod.rs:2915`, fetch `mod.rs:3170`, `pypi::resolve` `mod.rs:3186` | PyPI version | **Yes** — BFS level/discovery order + two drop points |
| **2** | `auto_bundle_transitives` | `src/handler/auto_bundle.rs:92-100` (sig); `pypi::resolve` at `auto_bundle.rs:438`; own fixpoint loop from `:139` | PyPI version | **Yes** — its own fixpoint iterates unresolved names in a non-canonical order; first-committed wins |
| **3** | `pre_emit_widen_pass` → `try_pypi_bundle` | `src/handler/cascade.rs:773` (sig); helper `try_pypi_bundle` at `cascade.rs:1130`; `pypi::resolve` at `cascade.rs:1143` | PyPI version (step-8 auto-bundle) **and** conda match-spec widening | **Yes** — fixpoint iteration order over unresolved conda deps decides which PyPI wheel is bundled |
| **4** | `produce_output` run-dep union | `src/handler/mod.rs:3888` (fn); union block comment `:3960-3963`; first-encountered-wins `if !seen_dep_names.insert(dep_name.clone()) { continue; }` `:4018-4019` | conda **spec** per dep name | **Yes** — explicitly first-encountered-wins over `bundle.all_wheels()` discovery order |

**OLD doc lines:** site 2 "auto_bundle.rs:92-512" (function is `:92-100` sig; the doc conflated
the whole module region); site 3 "cascade.rs:773-1080" (correct start, but the actual PyPI pick is
in `try_pypi_bundle` at `:1130/:1143`, which the old doc missed); site 4 "mod.rs:3950"
(now `:4018`). **All four must be made confluent (§3.3).**

### 1.3 The cascade

- `iterative_solve_refinement` — `src/handler/cascade.rs:343`, cap `MAX_REFINEMENT = 10`
  (`src/handler/cascade.rs:28`). Widens **conda** match-specs and re-emits via `produce_output`.
- `pre_emit_widen_pass` — `src/handler/cascade.rs:773` — runs the PyPI step-8 auto-bundle
  (`try_pypi_bundle`) *and* conda widening *before* the conda refinement loop (this is site #3).

### 1.4 Conda solve-check + the inert warm-start seam

- `solve_selected_records_from_records` builds the `SolverTask`; `locked_packages: preferred`
  (soft) wired but the public entry forwards `Vec::new()` — the warm-start seam from commit d19a71b
  is structurally present but **always inert**. (`src/solve_check.rs`; exact lines unchanged in
  spirit — the seam is dead-on-arrival until §5.E.1 wires or removes it. Re-pin during impl; not
  load-bearing for confluence.)

### 1.5 Lock structure + ordering (re-pinned; new schema-9 fields flagged)

- `RetreadLock` — `src/lock.rs:184-224`: `schema` (`:185`), `retread_version` (`:186`),
  `bundle` (`:187`), `version` (`:188`), `python` (`:190`), `inputs_hash` (`#[serde(default)]`,
  `:196-197`), `root_requirements: Vec<String>` (`#[serde(default)]`, `:203-204`),
  `wheels: Vec<LockWheel>` (`:206`), `conda_run_deps: Vec<CondaDep>` (`:208`),
  `index_urls: Vec<String>` (`:211`), `prerelease: BTreeMap<String,String>`
  (`skip_if_empty`, `:215-216`), `conda_capable: Vec<String>` (`skip_if_empty`, `:222-223`).
- `LockWheel` — `src/lock.rs:118-171`: `name` (`:120`), `version` (`:121`), `origin` (`:122`),
  `filename` (`:124`), `url?` (`:127`), `sha256?` (`:130`), `requires_dist: Vec<String>`
  (`skip_if_empty`, `:134`), `must_ship: bool` (`:140`), `upstream_url?` (`:153`),
  `git_source: Option<GitWheelSource>` (`:161`), **`sdist_source: Option<SdistWheelSource>`
  (`:169` — NEW since schema 8; this drove SCHEMA 8→9).**
- `CondaDep { name, spec }` — `src/lock.rs:177-180` (no serde attrs).
- `GitWheelSource { url, rev, subdirectory?, extras }` — `src/lock.rs:52-70`.
- `SdistWheelSource { index, name, version, sdist_url }` — `src/lock.rs:91-104` (**NEW**;
  `sdist_url` carries the `#sha256=` fragment).
- **NO `drop_url` field exists on `LockWheel` or `RetreadLock`.** Phase 2.8's orphan-URL strip is
  *recomputed at emit time* from `requires_dist` + bundle membership (`EmitPlan.drop_url`,
  `src/emit_pypi.rs:199-221`), not persisted. **This is load-bearing for byte-identity** (§2.5,
  §4.4).
- **Discovery/emit ordering (confirms grizzly P2):**
  - `wheels[]` built in `courier::stage` loop `for w in emit_wheels { ... }` at
    `src/courier.rs:488` (per-class `lock_wheels.push` at `:550/:710/:750/:794`); `emit_wheels`
    derived from `bundle.all_wheels()` at `src/handler/mod.rs:5042-5075`. **Discovery order.**
  - `conda_run_deps[]` = `parse_conda_deps(run_deps)` at `src/courier.rs:977`; `run_deps` is the
    `produce_output` first-encountered union. **Emit order.**
  - Only `conda_capable` is sorted: `conda_capable_sorted.sort()` at `src/courier.rs:966-967`.
    **OLD: 938-939.**
- **There is no `RetreadLock::canonicalize` method today** (confirmed; `impl RetreadLock`
  `src/lock.rs:269-357` has only `file_name`, `compute_inputs_hash`, `load`, `to_pretty_json`,
  `marker_name`).

### 1.6 inputs_hash, entry_specs, replay (re-pinned)

- `compute_inputs_hash` — `src/lock.rs:295-332`. Signature takes
  `(entry_specs, index_urls, relax, python, emit_epoch, pin_version, config_fingerprint)`.
  Domain `b"retread-inputs-v5\n"` (`:308`). Order: entry_specs **sorted** (local clone `.sort()`,
  `:305-306`), `--indexes--` then index_urls **in order**, `--meta--` relax + python, `--epoch--`
  `emit_epoch.to_le_bytes()`, optional `--pinver--`, `--config--` config_fingerprint.
  **`wheels`/`conda_run_deps`/`prerelease` are NOT read.** (Reordering lock vectors ⇒ hash
  unchanged. `prerelease` is NOT in the hash — a gap the delta-detector must guard, §4.1.)
- `courier_input_specs` — `src/courier.rs:49-86`: one canonical spec per manifest wheel entry
  `"<key>[extras]<ver-proxy>"`, `specs.sort()` at `:84`. Never a resolved version.
- **`root_requirements` is NOT `courier_input_specs`.** Written at `src/courier.rs:975` as
  `vec![format!("{bundle_name}-pypi=={version}")]` — the single meta-wheel pin. **The old doc's §D.1
  assumed the delta-detector could diff against `root_requirements`; it cannot.** This forces a new
  persisted `entry_specs` field (§4.1, §4.2).
- `SCHEMA = 9` (`src/lock.rs:235`); `EMIT_EPOCH = 5` (`src/lock.rs:267`). **OLD: 8 / 4.**
- Replay control flow (re-pinned — the old "1897/1915/4964/4984" lines are all stale):
  - Build entry `conda_build_v1` — `src/handler/mod.rs:1719`. Owns the replay-vs-resolve decision.
  - `load_replayable_lock` — `src/handler/mod.rs:5344`. `RETREAD_NO_REPLAY` gate `:5350`; schema
    exact-equality gate `if lock.schema != crate::lock::SCHEMA { return Ok(None) }` `:5364`;
    inputs_hash match gate `:5368`. **OLD: 4964 / 4984 / 4988.**
  - Replay branch consulted at `conda_build_v1:1836` (`match load_replayable_lock(...)`); replay
    success arm `Ok(Some(lock))` at `:1837` → `materialize_from_lock` at `:1868`.
  - **Cold-resolve kickoff: `resolve_all(...)` at `src/handler/mod.rs:1919`.** This is the hook
    point for Part 2 (§4.3). **OLD: "resolve_all at 1915".**
  - `replay_from_lock` — `:5406`; `materialize_from_lock` — `:4139`; `materialize_and_pack` —
    `:4835`; `assemble_conda_output` — `:3761`. **OLD: 5026 / 4071 / 4460 / 3681.**
- `materialize_from_lock` wheel-class dispatch (`match lw.origin`, from `:4256`):
  - `Origin::Index` `:4257` (Class 4 — unchanged index wheel).
  - `Origin::Built if lw.must_ship` `:4298` (Class 1 / Class 3; git_source sub-dispatch incl.
    **Phase-2.5 multi-entry shared-checkout** group machinery — grizzly amendment **M-1**, §6).
  - `Origin::Built if !lw.must_ship && lw.sdist_source.is_some()` `:4607` (Class 2b — sdist shadow).
  - `Origin::Built` catch-all `:4678` (Class 2 — relax-changed index shadow; conda-capable Class-2
    download path from Phase 2.7).
- **Phase-2.5 multi-entry machinery (M-1):** grouping is inline in `materialize_from_lock`
  (`git_group_members` HashMap `:4188`, group scan `:4193`, carrier election = group index 0
  `:4230`, `auto_data_override` `:4213`, lazy stash `:4340-4420`). Helpers
  `checkout_root_for_entry` (`mod.rs:3313`), `git_checkout_root` (`src/source_build.rs:255`).
- `materialize_and_rewrite` — `src/handler/mod.rs:3367`, returns
  `Result<(ResolvedWheel, Vec<String>)>` where the `Vec<String>` is the **pre-rewrite original
  `Requires-Dist`** (the BFS seed). **OLD: "~3303 / ~3116".**

---

## 2. PART 1 / Section A — Canonical lock ordering (byte-identity precondition)

### A.1 Why ordering is necessary but not sufficient

Byte-identity between a cold resolve and an incremental merge requires the serialized lock to be a
pure function of the *set* of resolved facts, not the *order* they were discovered. Today
`wheels[]` and `conda_run_deps[]` are discovery/emit ordered (§1.5). Even a *correct* merge that
picks identical versions would serialize them in a different sequence. Canonical ordering removes
that degree of freedom. It is **necessary, not sufficient**: if §3 (confluence) is not done, the
two runs pick *different versions* and no ordering saves byte-identity. A and B are a matched pair.

### A.2 Canonical order: pure lexicographic by canonical name, **including nested structures** (amendment A-1)

> **Grizzly amendment A-1 folded in:** canonicalization must ALSO sort the nested structures
> inside each wheel — primarily `LockWheel.requires_dist` — not just the top-level `wheels` vector.
> Otherwise two confluent runs can emit the same *set* of requires_dist lines in different *line
> order* and byte-identity fails. **The §4.2 surface trim REMOVES four would-be nested-sort
> surfaces** (`resolved_constraints`, `chosen_extras`, `requires_dist_original`,
> `marker_env_fingerprint` are all dropped), so A-1 now covers only `requires_dist`, the existing
> `GitWheelSource.extras`, and the new (already-sorted) top-level `entry_specs`.

**Top-level vectors:**
- `wheels[]`: sort by `(canonical_conda_name(name), version, origin-discriminant, filename)`.
  The 4-tuple key is total even if a bundle legitimately ships two wheels with the same canonical
  name (platform fan-out) — no ambiguity for insertion order to leak through.
- `conda_run_deps[]`: sort by `(name, spec)`. Names are unique post-dedup
  (`seen_dep_names`, `mod.rs:4018`); `spec` is a defensive tiebreak.
- `root_requirements[]`: sort lexicographically (today it is a single element; future-proof).
- `entry_specs[]` (new, §4.2): already produced sorted (`courier_input_specs` `:84`); re-sort in
  `canonicalize` defensively so the invariant lives in one place.
- `conda_capable`: already sorted; fold into the canonicalizer (remove the inline
  `src/courier.rs:966-967` sort).
- `index_urls`: **NOT sorted** — chain order is semantically significant and feeds
  `compute_inputs_hash` in order (`src/lock.rs` index block). Leave it.
- `prerelease`: `BTreeMap` (already key-ordered).

**Nested-per-wheel (A-1) — trimmed by the §4.2 surface trim:**
- `LockWheel.requires_dist`: **sort lexicographically.** (Verify this is safe — see A.5 caveat.)
- `GitWheelSource.extras`: sort (already small; make canonical).
- *(No `requires_dist_original` / `resolved_constraints` / `chosen_extras` sort surfaces — those
  fields were dropped, §4.2. If the conditional `requires_dist_original` is later proven necessary,
  add it to this list, sorted A-1, in the same PR.)*

### A.3 Single serialize-path choke point

Add `RetreadLock::canonicalize(&mut self)` in `src/lock.rs` (the `impl` block at `:269-357`). It
sorts every vector listed in A.2, top-level **and** nested, and is **idempotent**. Call it exactly
once, at the end of `courier::stage` immediately before the `RetreadLock { ... }` value at
`src/courier.rs:968-981` is serialized — and also at the end of the Part-2 incremental-merge path
(§4) so both producers share the identical normalizer. **Single source of truth for "canonical."**
Remove the inline `conda_capable_sorted.sort()` (`:966-967`) and route it through `canonicalize`.

Do **not** sort inside `produce_output`, the resolver, or the cascade: those orders are consumed by
the cascade's widen passes and the `ProbeDecision` audit and must stay discovery-ordered for
diagnostic stability. Canonicalization is a *presentation* concern of the persisted artifact.

### A.4 Replay / inputs_hash impact — confirmed safe

- `compute_inputs_hash` (`src/lock.rs:295-332`) reads none of `wheels`/`conda_run_deps`/`prerelease`
  /`requires_dist` (§1.6). **Reordering does not change inputs_hash.** Replay gating
  (`load_replayable_lock` `:5364/:5368`) is unaffected.
- The persisted JSON *bytes* change, so every committed lock is byte-different from the new
  canonical producer. The `SCHEMA 9→10` bump (§7) forces the exact-equality gate (`:5364`) to
  reject every schema-9 lock → full cold resolve → canonical schema-10 lock. After one cold pass
  per pack, all committed locks are canonical and self-consistent.
- `materialize_from_lock` (`:4139`) iterates `lock.wheels` by class, order-independent (each wheel
  materialized by its own provenance). **Re-verify (test) that no code path indexes `wheels[]`
  positionally** — in particular the Phase-2.5 carrier election (`:4230`) picks "group index 0".
  After canonicalization the group's index-0 member is the lexicographically-smallest name; the
  re-baseline regenerates the lock under the same canonicalizer, so produce and replay agree by
  construction. **This is an explicit M-1 check — see §6.**

### A.5 A-1 caveat (must verify, not assume)

Sorting `requires_dist` lines is safe for *metadata equality* but the rewritten wheel **bytes**
bake `requires_dist` in their original file order. The lock's `requires_dist` is only an input to
`plan()` (`src/emit_pypi.rs:230`) / replay reconstruction, which iterate it order-independently
(union/membership). **Verify (henry + a test) that nothing consumes `LockWheel.requires_dist` in a
line-order-sensitive way** before sorting it. If any consumer is order-sensitive, sort a *clone* for
serialization only, or store an explicit canonical copy. Default plan: sort in place after
confirming order-insensitivity; the test in §7.5(a) asserts it.

---

## 3. PART 1 / Section B — Confluent, constraint-accumulating resolution (THE MAKE-OR-BREAK)

This is the linchpin and the centerpiece of the plan. Without it, a cold resolve is not even a
well-defined target, so "incremental == cold" is meaningless.

### 3.1 Definition of confluence (the precondition for everything)

Define resolution as a function `R(roots, index_snapshot) -> {canonical_name -> (version, source)}`
that must satisfy:

1. **Constraint soundness.** For every resolved name `n`, the chosen version satisfies the
   *intersection* of all `Requires-Dist` specifiers from every requirer of `n` in the closure
   (under the active marker env). Today violated at the two drop points (§1.1) and at sites 2/3
   (§1.2).
2. **Order independence (confluence).** `R` is invariant under permutation of discovery order —
   across BFS level batching, the `auto_bundle_transitives` fixpoint, the `pre_emit_widen_pass`
   fixpoint, and `produce_output`'s union. Same roots + same index snapshot ⇒ same result, full
   stop, regardless of concurrency or discovery sequence.
3. **Deterministic tie-break.** Among versions satisfying the intersected constraint, pick the
   existing policy's choice (exact-first then highest-in-range, via `pypi::resolve`) with a fully
   specified source precedence for mixed sources.

**Corollary (the whole point):** if `R` is a deterministic fixpoint, then
`R(roots ∪ {new})` from a warm start equals `R(roots ∪ {new})` cold, because both are the unique
fixpoint of the same `R` on the same inputs (initialization point doesn't change a confluent
monotone fixpoint). **This corollary is the entire soundness basis for Part 2.**

### 3.2 The confluence proof (centerpiece — what makes it testable)

The argument has three pillars; each is independently testable so the grizzly can verify the proof
empirically, not just on paper:

- **Pillar 1 — Constraint accumulation is order-free.** Specifier intersection (PEP 508 AND of
  clauses, which `VersionSpecifiers` already represents as an AND-set) is commutative and
  associative. So the accumulated constraint for a name is independent of the order requirers are
  observed. *Test:* §7.5(a) intersection-permutation unit test.
- **Pillar 2 — Version selection is a deterministic max under the accumulated constraint.**
  `pypi::resolve` against the *final* intersected constraint yields a unique version (exact-first,
  else highest-in-range, against a fixed index snapshot). *Test:* §7.5(a) re-resolve-on-tighten
  test (discover loose, then tighten → same as tight-from-start).
- **Pillar 3 — Iteration converges to a unique fixpoint and is driven in a canonical order.** The
  worklist is keyed by canonical name (a `BTreeMap`/min-heap, not a `VecDeque`), so names are
  (re)processed name-sorted, not discovery-ordered. Tightening is monotone; a name re-resolves at
  most once per distinct version it can take (finite); a hard iteration cap (mirror
  `MAX_REFINEMENT`, `cascade.rs:28`) fail-closes if exceeded. *Test:* §7.5(a) shuffle-N-times
  determinism harness asserts identical `chosen` + identical canonical bytes.

If all three pillars hold across all four sites (§3.3), `R` is confluent and the §3.1 corollary
gives Part 2 its soundness for free.

### 3.3 Confluence must cover ALL FOUR sites (amendment B — explicit)

A single `resolve_closure(roots, index_state) -> ResolvedClosure` abstraction (the `ResolveState`
object, §5.E.3) becomes the **one** place version selection happens. Each of the four sites is
re-pointed at it:

1. **Site 1 — `resolve_bundle` (`mod.rs:2564`).** Replace `seen: HashSet`
   (`:2597`) + the two drop points (`auto_bundle.rs:628/645`, `mod.rs:2713`) with
   `state.observe_edge(name, specifiers, requirer)` that *intersects* instead of *dropping*, and
   re-resolves-on-tighten. Drive the worklist name-sorted (Pillar 3).
2. **Site 2 — `auto_bundle_transitives` (`auto_bundle.rs:92`, pick at `:438`).** Its private
   fixpoint must use the *same* `ResolveState` / accumulation, not its own first-committed-wins
   loop. Either fold it into `resolve_closure` or have it call `state.observe_edge` + the shared
   resolver. **If left order-dependent, the result is non-confluent — B fails.**
3. **Site 3 — `pre_emit_widen_pass` → `try_pypi_bundle` (`cascade.rs:773` / `:1130` / `:1143`).**
   The PyPI bundling half must route version selection through the same accumulation. The conda
   *widening* half (match-spec widening) is downstream of the bundle and is confluent **iff its
   input bundle is confluent** — so once sites 1/2 are confluent and the bundle is canonical, the
   widen pass's PyPI picks are too. Its conda-spec output feeds site 4.
4. **Site 4 — `produce_output` union (`mod.rs:3888`, drop at `:4018`).** Replace
   first-encountered-wins with: iterate `chosen` in **canonical-name order** and emit the conda
   spec derived from the *intersected* constraint for that name (so emission agrees with
   resolution). This deletes the §1.2 second-order dependence rather than sorting around it (A
   sorts the *output*; B makes the *content* order-free).

**Make-or-break note for the grizzly:** if any of sites 2/3/4 cannot be made to use the shared
accumulation (e.g. an architectural coupling that resists it), confluence is not achieved →
**Part 2 is unsound → DO NOT SHIP Part 2, and DO NOT bump the epoch for Part 1** (§3.5).

### 3.4 Implementation option: harden the BFS (B-α), not resolvo (B-β)

- **B-α (recommended):** harden the hand-rolled resolver into a constraint-accumulating fixpoint
  behind `resolve_closure`. Bounded to sites 1-4 + `seed_worklist` + the drop points; does **not**
  touch courier/replay. Est. 400-700 LOC net. No backtracking: an empty intersection (no version
  satisfies all requirers) is a **fail-closed conflict error** (matches today's no-backtrack
  behavior; more honest than silently picking). Mixed pypi+url/git for one name: url/git wins by
  fixed precedence *only if* its version satisfies the accumulated pypi constraint, else conflict.
- **B-β (deferred):** replace the PyPI resolver with a resolvo `DependencyProvider`. True
  backtracking + confluence for free, but a 1000+ LOC subsystem with a long correctness tail and a
  *larger* re-baseline blast radius (it changes *which versions* get picked, not just order).
  Keep `resolve_closure` as the seam so B-β is a future drop-in.

**Recommendation: B-α.** retread's domain (curated co-installable wheel families) rarely needs
backtracking; fail-closed conflict is acceptable and documented, not hidden.

### 3.5 Honest STOP condition (the user's certainty bar)

> **This is the section that lets us cleanly NOT SHIP.** If, during implementation or the G-1 gate
> (§7.3), we find that:
> - confluence cannot be achieved at all four sites (e.g. site 2 or 3 resists shared accumulation), **or**
> - constraint accumulation *manufactures a conflict* that turns a currently-green pack RED (the
>   grizzly named **pillow 11.3 vs 12.0 in isaacsim** as a live candidate: today first-encountered-wins
>   silently picks one; with accumulation the intersection `11.3 ∧ 12.0` is **empty** → conflict), **or**
> - any pack's *resolved versions* (not just order) change at the re-baseline,
>
> then the feature is **not soundly deliverable as-is**. The correct outcome is: **do not bump the
> epoch, do not ship Part 1, leave `RETREAD_INCREMENTAL` off, report honestly.** A manufactured
> conflict is not a bug to paper over — it means the current lock was *unsound* (shipping a closure
> a real resolver rejects), and surfacing it is correct, but it must be surfaced to the USER as a
> decision (add an override), never auto-resolved by silently re-introducing first-wins. The G-1
> gate (§7.3) is precisely the instrument that detects this before any push.

---

## 4. PART 2 / Sections C+D — Persisted graph + incremental fast path (OPT-IN, `RETREAD_INCREMENTAL=1`)

> **Inert by default.** Part 2 is gated on `RETREAD_INCREMENTAL=1`. With the flag off, none of §4
> executes; `conda_build_v1` (`mod.rs:1719`) goes replay-or-cold exactly as today. The flag is read
> once at the start of the incremental hook (§4.3) and at the §4.5 oracle.

### 4.1 The delta-detector needs a persisted entry_specs (root_requirements is wrong)

Per §1.6, `root_requirements` stores only `vec![format!("{bundle_name}-pypi=={version}")]`, **not**
`courier_input_specs`. The delta-detector cannot diff against it. Therefore **persist the entry
specs** so the incremental path can compute the delta in-memory.

Also: `prerelease` is **not** in `inputs_hash` (§1.6). A change to prerelease pins leaves
`inputs_hash` identical but can change the resolved closure. The delta-detector must therefore
**additionally compare `prerelease` byte-for-byte** (current config vs lock) and fall back to full
on any difference. (Fixing prerelease's absence from the hash is out of scope here; the detector
guards it conservatively.)

### 4.2 New persisted field — schema-10 SURFACE TRIM (grizzly amendment: persist ONLY `entry_specs`)

> **Grizzly re-review amendment (folded, decisive):** the prior draft proposed FIVE new fields. Four
> are re-derivable or already-available and were dropped. **The schema-10 bump persists exactly ONE
> new field.** A smaller schema = a smaller re-baseline blast radius + fewer A-1 nested-canonicalization
> surfaces.

**THE FINAL SCHEMA-10 FIELD LIST (decisive):**

Add to `RetreadLock` (`#[serde(default)]`, `skip_serializing_if = "Vec::is_empty"`; old locks parse,
but the exact schema gate `mod.rs:5364` rejects them for replay and forces re-baseline):

- **`entry_specs: Vec<String>`** — exactly `courier_input_specs(config, bundle)` output
  (`src/courier.rs:49-86`), already sorted by `specs.sort()` (`:84`). Written in `courier::stage`
  at the `RetreadLock { ... }` construction (`src/courier.rs:968-981`). **This is the ONE genuinely
  necessary addition:** `root_requirements` (`:975`) is just the meta-wheel pin
  `vec![format!("{bundle_name}-pypi=={version}")]`, so the delta-detector has nothing to diff
  against without a persisted spec list. (Supersedes the old doc's `root_requirements` reuse;
  `root_requirements` stays as-is for back-compat.)

**That is the entire schema-10 surface: one field.** No new `LockWheel` fields. No new structs.

**The four DROPPED fields and the reason each is unnecessary (grizzly rationale):**

1. **`resolved_constraints` — DROPPED (re-derivable).** The constraint graph (`constraints` +
   `requirers`) is re-derived in-memory at warm-start time by scanning the locked wheels' already-
   persisted `requires_dist` (`LockWheel.requires_dist`, `src/lock.rs:134`, present since schema 5).
   For each locked wheel, parse its `requires_dist` edges → reconstruct `requirers[name]` and
   intersect specifiers → `constraints[name]`. This is O(edges) in-memory work with zero fetch, done
   once on the warm-start path; persisting the *result* saves only that cheap recomputation and is
   not worth a wider schema + an extra A-1 nested-sort surface.
2. **`marker_env_fingerprint` — DROPPED (already covered).** `python` is in `inputs_hash`
   (`compute_inputs_hash`, `src/lock.rs:295`); the target platform is structurally fixed per env ×
   subdir (the build target), not a free variable a lock could be reused across; and `marker_env_for`
   (the marker-env builder) is a **pure function** of (platform, python). So the warm start can
   reconstruct the exact marker env from data it already trusts (the inputs_hash match already
   guarantees `python` is identical, and the trigger gate §4.4 re-confirms it). A persisted
   fingerprint adds nothing the gates don't already enforce.
3. **`chosen_extras` — DROPPED (already encoded).** Which extras were active on a config-entry wheel
   is encoded in `entry_specs`' `[extras]` substring (`courier_input_specs` emits `"<key>[extras]…"`,
   `src/courier.rs:49-86`). Transitive extras are recovered when the warm start re-derives edges from
   `requires_dist` (item 1). No separate field needed.
4. **`requires_dist_original` (pre-relax) — DROPPED, with one conditional (read carefully).** relax
   only ever **WIDENS version specifiers; it never changes dependency NAMES.** The warm-start
   correctness check (§4.6) operates on the **set of edge names** and whether the *added* dep
   tightens an existing pin — it seeds from the locked `chosen` versions and does **not** re-derive
   sub-wheel versions from scratch. Edge *names* are identical pre- and post-relax, so the persisted
   post-relax `requires_dist` is sufficient for the name-graph the incremental check needs. **The ONE
   case that would need pre-relax lines** is if a concrete incremental code path provably re-derives
   a sub-wheel's *version* from its parent's original specifiers (rather than reusing the locked
   `chosen` version). The design (§4.5) explicitly does NOT do that — it reuses `chosen` and only
   re-resolves the *affected* subtree against live metadata. **Decision: do NOT persist
   `requires_dist_original`. If, during PR-4 implementation, the swe finds a concrete path that
   provably needs pre-relax lines, add `requires_dist_original: Vec<String>` (serde-default, sorted
   A-1) under the SAME schema-10 bump — do not introduce a second bump.** This is the only
   conditional field, and the trigger is a demonstrated code need, not a precaution.

**Net schema-10:** `RetreadLock.entry_specs: Vec<String>` only (plus the conditional, only-if-proven
`LockWheel.requires_dist_original`). The field does **not** enter `compute_inputs_hash` (it stays a
manifest fingerprint, §1.6). Lock growth: one short string vector — negligible. A-1 nested
canonicalization (§2.2) now applies to `entry_specs` (already sorted) and the existing
`requires_dist` (the §2.2 / §A.5 caveat) only — the four dropped fields remove four would-be
nested-sort surfaces.

### 4.3 The hook (where Part 2 attaches)

In `conda_build_v1` (`mod.rs:1719`), at the cold-resolve kickoff `resolve_all(...)` `:1919`
(reached only after a replay MISS — schema match but inputs_hash differ, or no lock): **if
`RETREAD_INCREMENTAL=1`**, first call `try_incremental_resolve(lock, config, ...) -> Option<...>`.
On `Some`, use its result (then `materialize_and_pack` as usual); on `None`, fall through to the
existing `resolve_all` at `:1919` unchanged. **Flag off ⇒ never even attempt; behavior identical to
today.**

### 4.4 Trigger detection (narrow, fail-closed)

Attempt incremental **only** when ALL hold; else `None` → full cold resolve:

1. `RETREAD_INCREMENTAL=1`.
2. A committed lock parses and `lock.schema == SCHEMA` (10).
3. Current inputs differ from `lock.inputs_hash` (else plain replay already wins, `:5368`).
4. The difference is **exactly one added entry_spec, zero removed/modified.** Compute current
   `courier_input_specs` (`:49`) and diff against `lock.entry_specs` (§4.2). Require
   `added = {one}`, `removed = {}`, and **every other inputs_hash component identical**
   (index_urls, relax, python, emit_epoch, pin, config_fingerprint) — recompute each and compare —
   **plus `prerelease` byte-identical** (§4.1).
5. The new entry is a plain PyPI-form add (not git/url/path — those lack a range to intersect).

Dep removal, modification, multi-add, any other component change, prerelease change, git/url add,
or **any detected conflict → full cold resolve.** Incremental is an optimization that is
provably-equal-to-cold or **not taken**.

### 4.5 What is reused vs re-derived

**Reused (no fetch):** prior `chosen` — every `LockWheel` → (name, version, source). The
constraint graph (`constraints` + `requirers` maps) is **re-derived in-memory** by scanning the
locked wheels' persisted `requires_dist` (`src/lock.rs:134`) — no persisted `resolved_constraints`
field (§4.2 drop #1). Extras come from `entry_specs`' `[extras]` substring + the re-derived edges
(§4.2 drop #3). This in-memory graph rebuild is O(edges), zero fetch.

**Re-derived (fetch only the affected subtree):** materialize the new dep + parse METADATA
(its `Requires-Dist`) → new edges; run §3's `resolve_closure` **seeded** with prior `chosen` +
the re-derived `constraints`. The fixpoint only touches a name when a new/tightened constraint
reaches it:
- new edge to existing name, prior version still satisfies intersection → no fetch, edge recorded;
- new edge that tightens past prior version → re-resolve that name + propagate to children
  (affected subtree only);
- brand-new name → fetch its subtree as cold would.
Re-run prefer-conda routing (`mod.rs:2750-2895`) only for names whose version changed/is new
(probes are cached after first call). Re-run sites 3/4 over the merged bundle (cheap; in-memory).

### 4.6 The correctness guarantee (provable, given §3)

**Claim:** incremental result == full cold resolve, byte-identical after `canonicalize` (§2).
**Proof:** by §3.1 corollary, cold = unique fixpoint of `R` on `roots ∪ {new}`. The warm start
computes the same `R` on the same inputs, merely initialized at the prior fixpoint `R(roots)`;
initialization point does not change a confluent monotone fixpoint. Both serialize through the same
`canonicalize` (§2.3) → byte-identical. The only divergence risk is if the prior `chosen` is not
itself a `R`-fixpoint — which is exactly why **Part 1 must land and ALL locks must be re-baselined
before Part 2 is enabled** (a pre-confluence lock is not a sound seed). Hence the schema/epoch bump
and re-baseline are correctness-load-bearing for Part 2, not cosmetic.

**Fail-closed:** any step that can't be cleanly handled (empty intersection, git/url conflict,
multi-delta, schema mismatch, marker-env reconstruction failure, iteration cap, prerelease drift) →
discard warm start → full cold resolve. (Marker env is reconstructed from `python` [inputs_hash-
guaranteed identical] + the build target platform via the pure `marker_env_for`; no persisted
fingerprint — §4.2 drop #2.) **There is no code path where an incremental result is emitted without
being equal to what cold would emit.** This is the safety property the grizzly verifies.

### 4.7 drop_url byte-identity (a Phase-2.8 hazard the old doc missed)

Phase 2.8's orphan-URL strip is recomputed at emit from bundle membership (`EmitPlan.drop_url`,
`emit_pypi.rs:284-296`), **not persisted** (§1.5). For byte-identity, the incremental merged
bundle's **membership must be identical to cold's** so `plan()` computes the *same* `drop_url`.
Since §3 confluence guarantees the merged `chosen` == cold `chosen` (same set of wheels in the
bundle), `plan()`'s membership-only predicate yields identical `drop_url` on both paths. **Add an
explicit test** (§7.5(b)) that an incremental add whose closure contains an orphan-URL wheel
(gigastrap-style robomimic edge) strips identically. **If membership ever differs, byte-identity
fails — caught by the §4.8 oracle.**

### 4.8 The TEST-ONLY equivalence oracle (amendment D — DEFERRED from shipped path)

> **Grizzly amendment D folded in:** the verify-oracle (run a full cold resolve to verify the
> incremental result) **defeats the seconds-scale speedup**, so it is **DEFERRED from the shipped
> fast path**. It is kept as a **TEST / e2e-only** instrument: `RETREAD_VERIFY_INCREMENTAL=1` runs
> the incremental path, then *also* runs the full cold resolve, and asserts byte-identical locks,
> erroring loudly on mismatch.

- **Shipped fast path** (`RETREAD_INCREMENTAL=1` alone): incremental only, no oracle, fast.
- **Test oracle** (`RETREAD_VERIFY_INCREMENTAL=1`): incremental + cold + byte-assert. Used by §7.5(b)
  parity tests and as a permanent CI regression tripwire. Off in production.

This is the explicit "shipped fast path vs test oracle" distinction the amendment requires.

---

## 5. Section E — Cleanups folded in

1. **Wire-or-remove the inert solve-check seam (d19a71b).** Part 2 carries a prior resolved conda
   set; seed `locked_packages` (soft) on the incremental path's solve-check. If Part 2 is descoped,
   **remove** the dead parameter rather than leave inert infra. (Re-pin solve_check lines at impl;
   not load-bearing for confluence.)
2. **Single canonicalization site** — fold `conda_capable_sorted.sort()` (`courier.rs:966-967`)
   into `RetreadLock::canonicalize` (§2.3).
3. **`ResolveState` object** — replace loose `seen`/`work`/`extras` (`mod.rs:2597/2598/2694`) with
   one `ResolveState { chosen, constraints, requirers, work }`; the seam a future B-β implements.
4. **Unify the drop points** — `auto_bundle.rs:628/645` + `mod.rs:2713` collapse into one
   `state.observe_edge(...)` so there is exactly one skip-vs-intersect-vs-reresolve decision (the
   grizzly's P1 is literally "two drop points, both wrong").
5. **Emission/resolution agreement** — `produce_output` (`mod.rs:4018`) emits from
   `chosen`/`constraints` (site 4, §3.3).
6. **Persist `entry_specs`** (§4.2) — supersedes the old doc's `root_requirements` reuse; leave
   `root_requirements` as-is (meta-wheel pin) for back-compat.
7. **No produce_output/replay dup to remove** — confirmed complementary (Bundle-driven vs
   lock-driven); both funnel through `assemble_conda_output` (`mod.rs:3761`). Listed so the grizzly
   knows it was checked.

---

## 6. Section M-1 — Multi-subdir / multi-entry shared-git-checkout compatibility (amendment M-1)

> **Grizzly amendment M-1 folded in:** the multi-subdir provenance IS the Phase-2.5 multi-entry
> shared-git-checkout machinery (already shipped, §1.6). Ensure canonical ordering (§2) AND the
> incremental path (§4) are compatible with multi-entry git groups.

- **Canonical ordering vs carrier election.** Phase-2.5 elects the carrier as **group index 0**
  (`materialize_from_lock` `:4230`) — the first lock-order member of a git checkout-root group.
  After §2 canonicalization, group index 0 is the **lexicographically-smallest canonical name** in
  the group. The re-baseline regenerates the lock under the same canonicalizer, so produce's
  carrier and replay's carrier are the same wheel **by construction**. **Explicit gate:** a
  cold-produce of an isaac pack (multi-entry: isaaclab + isaaclab_assets/_tasks/_rl/_mimic/_physx/
  _newton, all from one IsaacLab git repo) under schema-10 must replay byte-identically (§7.5(d)),
  proving carrier election survives canonical reordering.
- **Incremental path vs multi-entry.** §4.4 trigger requires the added entry be a plain PyPI add
  (not git/path). A single PyPI add to an isaac pack does **not** touch the git groups (their
  membership/subdirs are unchanged), so the multi-entry stash machinery (`:4340-4420`) replays the
  unchanged git wheels exactly as in a cold resolve. The incremental delta only adds PyPI wheels +
  their PyPI transitives; git-group wheels are copied verbatim from the prior `chosen`. **Test
  (§7.5(d)):** incremental PyPI add to a multi-entry isaac pack == cold, byte-identical, git groups
  untouched.

---

## 7. Section G — Phasing, schema/epoch, and the MANDATORY verification gates

### 7.1 Phasing and PR order (forced by the correctness dependency)

Order is forced: **B (confluence) before A (ordering) before C/D**, because A's re-baseline should
write *sound+canonical* locks in one pass, and D's soundness depends on B.

- **PR-1 (B): confluent resolver.** `ResolveState`, unified `observe_edge` (intersect +
  re-resolve-on-tighten + fail-closed conflict + iteration cap), name-sorted worklist, all four
  sites routed through `resolve_closure`, emit from `chosen` (site 4). Output *content* can change
  for back-pressure cases. **Bump `EMIT_EPOCH 5→6`** (`src/lock.rs:267`) — emission semantics
  changed. Invalidates old locks' replay via the epoch in inputs_hash. Includes the synthetic
  back-pressure test (§7.5(b)). **This PR is the make-or-break; it does not merge until the G-1 gate
  (§7.3) passes.**
- **PR-2 (A): canonical ordering.** `RetreadLock::canonicalize` (top-level + nested A-1), call at
  serialize boundary, fold in conda_capable sort. **Bump `SCHEMA 9→10`** (`src/lock.rs:235`).
- **PR-3 (C): persist `entry_specs`.** The single schema-10 field `RetreadLock.entry_specs`
  (§4.2 surface trim — the other four candidate fields are DROPPED), serde-default, populated in
  `courier::stage` at the `RetreadLock { ... }` construction (`src/courier.rs:968-981`). **Batch
  under the same SCHEMA 9→10 bump as PR-2** to avoid 9→10→11 churn. No behavior change (field
  written, read only by Part 2 which is flag-off by default).
- **PR-4 (D): incremental fast path,** behind `RETREAD_INCREMENTAL=1` (default OFF). Hook at
  `mod.rs:1919` (§4.3), trigger detector (§4.4), seed + fail-closed (§4.5-4.6), test-only oracle
  `RETREAD_VERIFY_INCREMENTAL=1` (§4.8). No schema bump (reads schema-10 fields).
- **PR-5 (E leftovers).**

**Schema/epoch summary:** `EMIT_EPOCH 5→6` (PR-1); `SCHEMA 9→10` (PR-2, carrying PR-3's single
`entry_specs` field). Both rely on existing fail-closed gates: epoch via inputs_hash mismatch
(`:5368`); schema via exact-equality (`:5364`). Old locks rejected, never misread.

### 7.1a SWE implementation sequence — green bar at EVERY step (handoff to swe-worker-bee)

The toolchain is **`PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH" cargo …`**
(NO host cargo). After **each** PR's commit, the green bar is **all three** of:

```
cargo test --lib
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Commit on `dev/incremental-add` only. **No push/merge/publish until G-1 (§7.3) + the incremental
e2e (§7.4) pass.**

| Step | What lands | Default behavior change? | Green bar | Gate before next step |
|------|-----------|--------------------------|-----------|-----------------------|
| **PR-1 (B)** | `ResolveState` + `resolve_closure` seam; unified `observe_edge` at all four sites (`mod.rs:2915/3186`, `auto_bundle.rs:438`, `cascade.rs:1143`, `mod.rs:4018`); name-sorted worklist; fail-closed conflict + iteration cap; emit from `chosen` (site 4). `EMIT_EPOCH 5→6`. | YES (emission semantics; expected NO version change per probe) | 3 commands green + §7.5(a) confluence unit tests + back-pressure fixture §7.5(b) | unit tests green |
| **PR-2 (A)** | `RetreadLock::canonicalize` (top-level + nested A-1: `requires_dist`, `GitWheelSource.extras`); call at serialize boundary `courier.rs:968`; fold in `conda_capable` sort `:966-967`. `SCHEMA 9→10`. | YES (lock byte order) | 3 commands green + §7.5(a) canonicalization unit tests (idempotent, permutation-invariant, inputs_hash-invariant) | unit tests green |
| **PR-3 (C)** | `RetreadLock.entry_specs` (the ONE schema-10 field) populated in `courier::stage`; canonicalize covers it. Batched under PR-2's `SCHEMA 9→10`. | NO (field written; read only by flag-off Part 2) | 3 commands green | — |
| **G-1 GATE** | (§7.3) re-baseline + shadow-resolve **every** committed lock both repos; resolved-set audit; determinism sub-gate. | — | — | **HARD: any version change or non-confluence → STOP, do not bump/ship** |
| **PR-4 (D)** | `try_incremental_resolve` hooked at `resolve_all` `mod.rs:1919` behind `RETREAD_INCREMENTAL=1` (default OFF); trigger detector §4.4; seed + fail-closed §4.5-4.6; in-memory graph rebuild from locked `requires_dist`; TEST-only oracle `RETREAD_VERIFY_INCREMENTAL=1` §4.8. No schema bump. | **NO when flag off (inert)** | 3 commands green + §7.5(a) delta-detector unit tests | §7.4 byte-identity parity + §7.5(b) e2e |
| **PR-5 (E)** | leftovers (wire-or-remove solve-check seam; any cleanup not folded into 1-4). | NO | 3 commands green | — |

**Key safety property for the swe:** PR-4's fast path is **flag-gated and inert when off**, so
**Part 1 (PR-1/2/3) is fully verifiable on its own** — the G-1 gate runs against Part-1-only
backend with `RETREAD_INCREMENTAL` unset. Part 2 is turned on only for the §7.4 parity e2e, never in
the shipped default.

### 7.2 Re-baseline procedure

After PR-1+PR-2+PR-3: for **every committed `retread-*.lock.json` in both repos** (not just the 7
headline packs — see §7.3 for the full enumeration), let the schema gate reject the old lock,
cold-build once → canonical schema-10 sound lock, **audit it (§7.3 G-1)**, then commit. From then on
every producer (cold or incremental) emits byte-identical locks for identical inputs.

### 7.3 GATE G-1 (MANDATORY, the user's certainty gate — amendment G-1, SCOPE WIDENED)

> **Grizzly amendment G-1 folded in and made the centerpiece of acceptance.** Constraint
> accumulation can MANUFACTURE conflicts (pillow 11.3 vs 12.0 in isaacsim) that turn a
> currently-green pack RED. **BEFORE any push/merge/publish and BEFORE the epoch bump is shipped**, a
> shadow cold resolve of **EVERY committed lock across BOTH repos** is MANDATORY.

**SCOPE (grizzly re-review amendment — widened from "7 pack names"):** the gate must enumerate and
diff **every committed `retread-*.lock.json` file** in `pixi-build-retread` (examples) AND
`pixidock_template` — i.e. **every env × platform variant**, not just one lock per headline pack. A
version change could hide in an env or platform variant that the 7 headline names don't surface.

1. **Enumerate every committed lock (decisive — do this first):**
   ```bash
   # both repos; capture the COMPLETE set the gate must cover
   find /home/garylvov/projects/pixi-build-retread -name 'retread-*.lock.json' -not -path '*/.pixi/*'
   find <pixidock_template> -name 'retread-*.lock.json' -not -path '*/.pixi/*'
   ```
   The 7 headline packs are the *minimum*; the gate covers **the full file list output above**
   (each env/platform lock that exists). Record the count; the audit (clause 3) must cover all of
   them.
2. **Cold-produce must succeed (none RED).** For each pack/env, no `resolve_closure` conflict error;
   the build completes and the env imports (same import asserts as the replay e2e harness,
   `scripts/replay-e2e.sh`).
3. **Resolved-set audit (the decisive check), for EVERY enumerated lock.** Diff the new schema-10
   lock's resolved versions against that lock's CURRENT committed resolved versions (name → version
   map, ignoring order and the new `entry_specs` field). **The ONLY allowed change is canonical
   REORDERING.** If ANY lock's *resolved version* for ANY name differs, that is a **manufactured
   conflict / changed resolution** → **STOP, DO NOT SHIP, report to the user** (§3.5). (A changed
   version means accumulation picked differently than first-wins did — that env was relying on the
   unsound behavior; surfacing it is correct but it is a user decision, never an auto-fix.)
4. **Determinism sub-gate (catch residual order leak).** Cold-produce **each** pack/env **twice** in
   the same run and assert the two schema-10 locks are byte-identical to each other. A difference
   here is a residual concurrency/discovery-order leak in site 1/2/3/4 → confluence not achieved →
   **STOP** (§3.5). This is what proves sites 2/3/4 are confluent (the probe only covered site 1).
5. **Conflict surfacing.** If any pack/env REDs on an empty intersection (e.g. pillow), capture the
   conflicting requirers and the offending name; this is the report to the user, not a thing to
   silence.

**G-1 is a hard gate:** PR-1/PR-2/PR-3 do not merge, the epoch is not bumped in any shipped artifact,
and `RETREAD_INCREMENTAL` is not enabled, until **every enumerated lock** passes clauses 2-4.

### 7.4 GATE: byte-identity parity (the headline incremental test)

For **genesis (light)** and **at least one isaac pack** (the M-1 multi-entry case):
1. Cold-resolve manifest with N entries → `lock_cold_N`.
2. Cold-resolve manifest with N+1 entries (one real PyPI dep added) → `lock_cold_N1`.
3. From `lock_cold_N`, add the same dep, run **incremental** (`RETREAD_INCREMENTAL=1`) →
   `lock_inc_N1`.
4. **Assert `lock_inc_N1` byte-identical to `lock_cold_N1`.** (Validates A, B-fixpoint,
   C-reconstruction, D-soundness simultaneously.)
5. Assert `RETREAD_VERIFY_INCREMENTAL=1` passes on step 3 (the test oracle, §4.8).

### 7.5 The full verification gate matrix (the user's "certain it works" bar)

**(a) Unit — confluence + canonicalization + delta-detector:**
- *Confluence:* intersection-permutation (process requirers A-first vs B-first → identical
  `chosen`); re-resolve-on-tighten (loose-then-tight == tight-from-start); shuffle-N-times
  determinism harness (identical `chosen` + identical canonical bytes); conflict fail-closed (empty
  intersection → error, no partial lock). Run the harness **across all four sites'** code paths,
  not just `resolve_bundle`.
- *Canonicalization:* idempotent (`canon(canon(x)) == canon(x)`); permutation-invariant (shuffle
  `wheels`/`conda_run_deps`/nested `requires_dist` → same bytes — the A-1 nested check);
  inputs_hash invariance (reorder vectors → `compute_inputs_hash` unchanged; assert explicitly so a
  future refactor that adds them to the hash trips this).
- *Delta-detector:* disjoint-add → safe (incremental taken); tighten-shared → fallback (full
  taken, asserted via a decline counter/trace); conflict → fallback; same-version-share → safe;
  multi-add / removal / modification / index-change / relax-change / python-change / prerelease-
  change / git-url-add → each declines to full.
- *Correctness module* (the pure merge-validation fn): unit-tested in isolation.

**(b) Byte-identity parity (real added dep):** §7.4 on genesis + one isaac pack, including the
synthetic **back-pressure fixture** (existing closure pins `C==2.9` via a loose requirer; the
added dep `D` requires `C<2.5`; assert incremental re-resolves `C` to highest `<2.5` AND
incremental lock == cold lock byte-for-byte — this is the test that proves the §1.1 defect is fixed
and the warm start is not stale; **run it first**) and the **drop_url** orphan-URL identity check
(§4.7).

**(c) G-1 shadow-resolve — EVERY committed lock in BOTH repos** (§7.3, scope widened): enumerate via
`find … -name 'retread-*.lock.json'` across both repos (every env × platform, not just 7 names);
none RED; each resolved-set audited vs its current lock; the ONLY allowed change is canonical
reordering; a changed version on ANY lock = STOP. Includes the determinism sub-gate (each produced
twice → byte-identical), which is what proves sites 2/3/4 are confluent (the probe covered only
site 1).

**(d) Regression — existing replay still byte-identical lukewarm:** every example + every pixidock
pack still replays byte-identically on an all-caches-nuked box under the schema-10 backend
(`scripts/replay-e2e.sh` standard: `build_v1` replay fires, derivation=0, wheels from empty, `git
diff --exit-code` clean, env imports). **Includes the M-1 multi-entry isaac packs** (carrier
election survives canonicalization, §6).

**(e) Determinism CI gate:** run every e2e pack twice in the same job; assert byte-identical locks
across runs (catches residual concurrency/order leak in A or B). *(This is the same discipline as
G-1 clause 4 §7.3, applied as a standing CI gate so the property is enforced on every future run,
not just at the one-time re-baseline.)*

### 7.6 Rollback / safety

- **Part 2 risk = zero by default:** `RETREAD_INCREMENTAL` off → §4 never executes. Abandoning Part
  2 is flipping/removing a flag.
- **Part 1 risk = the epoch/schema bump only.** If G-1 (§7.3) surfaces a red pack or a changed
  resolved version, **we do not ship the bump** — revert PR-1's `EMIT_EPOCH 5→6` and PR-2's
  `SCHEMA 9→10`, and the tree is back to v2.8.0 behavior. Nothing is pushed until all gates pass; on
  green, **HOLD for user review** before any push/merge/publish/pixidock-commit (per the
  commission's hard constraint).
- **The single decisive STOP rule (§3.5):** if Part 1 reorders a pack's *resolved versions* (not
  just order), or any pack REDs, that is a regression / manufactured conflict → **DO NOT SHIP**,
  report honestly. Never output a false "works."

### 7.7 Honest cost/benefit (restated)

- **A+B (confluence + canonical ordering) are the real value:** they convert "the lock is whatever
  the BFS happened to discover" into "the lock is the unique sound resolution of the inputs." Worth
  doing even if D is never enabled — *but* they carry the epoch-bump blast radius and the G-1 risk.
- **D (incremental add) is seconds-scale** (skips DERIVATION, not materialization — wheel
  downloads/builds dominate wall-clock and run on every path). It is a capability unlock (a *sound*
  incremental add at all), not a big speedup.
- **Correctness-audit surface is large** (four version-picking sites, nested canonicalization, the
  drop_url interaction, multi-entry carrier election). The plan front-loads that surface into the
  G-1 gate and the parity tests precisely so the user's certainty bar is met by *evidence*, not
  assertion. **If at any gate the evidence says it's unsound or turns a pack red, the correct
  outcome is DO NOT SHIP** (§3.5, §7.6).

---

## 8. Summary for the auditor

- **Confluence approach (§3, the centerpiece):** harden the hand-rolled BFS into a
  constraint-accumulating, name-sorted, re-resolve-on-tighten **fixpoint** (B-α) behind a single
  `resolve_closure` / `ResolveState` seam; fail-closed on empty intersection; three testable
  pillars (intersection is order-free; version selection is a deterministic max; iteration
  converges in canonical order). Confluence ⇒ cold resolve is a unique fixpoint ⇒ incremental ==
  cold for free.
- **The four sites (amendment B, all re-pinned):** (1) `resolve_bundle`/`bfs_fetch_pypi`
  `mod.rs:2564/2915/3170`; (2) `auto_bundle_transitives` `auto_bundle.rs:92`, pick `:438`;
  (3) `pre_emit_widen_pass`→`try_pypi_bundle` `cascade.rs:773/1130/1143`; (4) `produce_output`
  union `mod.rs:3888/4018`. All four must route version selection through the shared accumulation,
  else not confluent → STOP.
- **Epoch / schema bumps:** `EMIT_EPOCH 5→6` (PR-1, emission semantics) and `SCHEMA 9→10` (PR-2+PR-3).
  **Schema-10 surface trim (grizzly amendment): exactly ONE new field — `RetreadLock.entry_specs:
  Vec<String>`.** The other four candidates are DROPPED: `resolved_constraints` (re-derived from
  locked `requires_dist`), `marker_env_fingerprint` (`python` in inputs_hash + pure `marker_env_for`),
  `chosen_extras` (encoded in `entry_specs`' `[extras]`), `requires_dist_original` (relax widens
  versions not names; add ONLY if a concrete incremental path provably needs it, same bump). Both
  bumps gated fail-closed (`:5364` schema, `:5368` inputs_hash). Re-baseline every committed lock.
- **Scope split:** Part 1 (A+B+C) is default-on + epoch-bumping; Part 2 (D) is opt-in
  `RETREAD_INCREMENTAL=1` (default OFF, inert when off — Part 1 is verifiable independently). The
  shipped fast path has NO oracle; the TEST-only oracle is `RETREAD_VERIFY_INCREMENTAL=1`
  (amendment D deferred from shipped path).
- **Verification gates (the certainty bar):** (a) unit — confluence/canonicalization/delta-detector;
  (b) byte-identity parity incremental==cold on genesis + one isaac pack, with the back-pressure
  fixture run first + the drop_url identity check; (c) **G-1 MANDATORY shadow-resolve of EVERY
  committed lock in BOTH repos** (every env × platform, enumerated via `find`) — none RED, each
  resolved-set audited, only-reordering allowed, ANY changed version = STOP, each produced twice =
  byte-identical (proves sites 2/3/4 confluent); (d) regression — all existing replay still
  byte-identical lukewarm incl. M-1 multi-entry isaac; (e) determinism CI gate — every pack twice.
- **Amendments folded:** **probe green-light** (top banner; necessary-not-sufficient, site 1 only,
  G-1 is the seal); **schema-10 surface trim** (one field, `entry_specs`); **B** (four sites);
  **A-1** (nested canonicalization, now only `requires_dist` + `GitWheelSource.extras` after the
  trim); **G-1** (mandatory every-lock-both-repos shadow-resolve + determinism sub-gate before
  bump); **D** (verify-oracle deferred to test-only); **M-1** (multi-entry carrier election survives
  canonicalization + incremental leaves git groups untouched).
- **Honest STOP (§3.5, §7.6):** if confluence can't be achieved at all four sites, or accumulation
  manufactures a conflict (pillow 11.3/12.0), or ANY committed lock's resolved versions change → do
  not bump the epoch, do not ship, leave the flag off, report honestly. **Nothing pushed/merged/
  published until G-1 + the incremental e2e pass; work stays on `dev/incremental-add`.**
