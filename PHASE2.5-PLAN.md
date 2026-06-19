# PHASE 2.5 PLAN — Multi-Entry Shared-Git-Checkout Replay

**Branch:** `courier`. **Scope:** make `build_v1` replay work for multi-entry
shared-git-checkout packs (pixidock `isaac-pack` / `isaac-pack-latest`), which
today HARD-ERROR on replay. OUT OF SCOPE: the incremental-add foundational rework
(separate, `PHASE3-FOUNDATIONAL-PLAN.md`).

All file:line citations are against the current tree (`courier`, v2.6.0,
`SCHEMA=8`, `EMIT_EPOCH=4`).

---

## 0. THE PROBLEM, RESTATED FROM CODE

The pixidock isaac packs build ONE git repo (`isaaclab`, declared once in
`[package.build.config.retread-git-sources.isaaclab]`) into 8 wheels, each from a
different `subdirectory=source/isaaclab_{,assets,tasks,rl,mimic,physx,newton}`,
all via `from="isaaclab"`. Cold produce under v2.6.0 already writes a correct
schema-8 lock: each of the 8 `LockWheel.git_source` carries
`{url, rev:resolved-SHA, subdirectory, extras}` (the data IS persisted —
`lock.rs:51-70` `GitWheelSource`).

Replay HARD-ERRORS. The single-entry guard added in e17889c
(`src/handler/mod.rs:4104-4105` declares `seen_git_checkout_roots`; the bail is at
`src/handler/mod.rs:4183-4192`) fires the moment a second wheel resolves to a
checkout root already seen, raising
`"wheel `isaaclab-assets` shares a git checkout root with a prior wheel
(multi-entry shared-checkout bundles are not yet supported...)"`. `materialize_from_lock`
returns `Err`, the warm path exits non-zero, the build FAILS. (Note: this is the
SECOND of two `seen_git_checkout_roots` references — the `HashSet` decl at 4104 and
the `.insert()` check at 4183.)

Single-entry git packs (genesis, newton) and index-only packs (isaac6) replay fine
because their loop never inserts a duplicate root.

---

## 1. PRODUCE PATH FOR MULTI-ENTRY GIT (the behavior replay must reproduce)

`resolve_all` groups `[retread-wheels]` entries by their `bundle` field into one
conda output (`src/handler/mod.rs:2167-2175`). For the isaac pack all 8 isaaclab*
entries land in ONE group. Inside the per-group loop
(`src/handler/mod.rs:2177-2254`) the auto-data dedup is computed BEFORE building any
wheel:

- **`entry_checkouts`** (`mod.rs:2192-2195`): for each entry, the git checkout root
  via `checkout_root_for_entry` (`mod.rs:3249-3268`). For a `from="isaaclab"` entry
  it returns `git_checkout_root(src.url, src.rev, cache_dir)` (`mod.rs:3255-3259`),
  for inline `git=`+`rev=` it returns `git_checkout_root(git_url, rev, cache_dir)`
  (`mod.rs:3260-3264`). All 8 isaaclab entries → the SAME root (same url+rev).

- **`auto_data_per_entry`** (`mod.rs:2197-2221`): the FIRST entry (BTreeMap order)
  that owns a given root carries `Some(AutoDataConfig{checkout_root, skip_subdirs})`;
  every subsequent sibling sharing that root gets `None` (the
  `emitted_auto_data.contains(root)` check at `mod.rs:2201-2203`).
  - **`skip_subdirs` for the carrier** (`mod.rs:2205-2215`) = the subdirectory of
    EVERY entry in the group whose checkout root equals this root (including the
    carrier's own), each defaulting to `"."` when absent. For the isaac pack the
    carrier's `skip_subdirs` = the full set
    `[source/isaaclab, source/isaaclab_assets, ..., source/isaaclab_newton]`
    (8 entries — one is `subdirectory="."` per the empirical note → the literal `.`).

- Each `(entry, auto_data)` pair is built by `resolve_bundle` →
  `materialize_and_rewrite` (`mod.rs:2224-2253`).

**What the carrier's `skip_subdirs` MEANS for the bytes** (`wheel_inject_data.rs:76-78`,
`90-95`, `131-139`): the carrier wheel's phase-1.6 auto-data pass walks the checkout
root honoring `.gitignore` and ships `<repo-root-tree>` into `.data/data/lib/<rel>`,
EXCLUDING every path under any `skip_subdirs` entry. So the carrier ships the repo
root files (apps/, tools/, share/, `.kit` experiences, etc.) but NOT the python
package source of ANY sibling (those are shipped by each sibling's own wheel via
phase-1.5 source inject + pip wheel). Non-carrier siblings get `auto_data=None`
(`mod.rs:3532` `if let Some(cfg) = auto_data.as_ref()`), so phase-1.6 is skipped
entirely for them — their wheel is just `pip wheel <subdir>` + phase-1.5 inject.

**Conclusion — the byte-identity contract replay must reproduce:**
For each git checkout root in a group:
1. exactly ONE wheel is the carrier; it runs phase-1.6 with `skip_subdirs` =
   the set of EVERY group member's `subdirectory` (defaulting to `.`).
2. every other wheel runs with `auto_data=None`.
3. which wheel is the carrier is determined by produce as "first in BTreeMap
   order over the group entries" — see §3 for why replay can pick ANY deterministic
   carrier and still be byte-identical.

---

## 2. REPLAY REDESIGN — group lock wheels by checkout root, reuse the produce build

### 2.1 Current replay shape (what we replace)

`materialize_from_lock` (`src/handler/mod.rs:4071-4448`) loops over `lock.wheels`
ONE AT A TIME (`mod.rs:4107`). For each `Origin::Built && must_ship` wheel with a
`git_source` it (a) inserts the checkout root into `seen_git_checkout_roots` and
bails on a dup (`mod.rs:4181-4192`), (b) synthesizes a single `WheelEntry`
(`mod.rs:4200-4215`), (c) sets `skip_subdirs=[its-own-subdirectory]`
(`mod.rs:4226-4227`) — CORRECT only when one wheel owns the root — and (d) calls
`materialize_and_rewrite` (`mod.rs:4232-4251`).

This is structurally wrong for multi-entry: it would build N carriers (each
shipping the root tree minus only its own subdir) instead of 1 carrier + N−1
non-carriers. That over-ships sibling source under `lib/` in every wheel ≠ produce.

### 2.2 New shape — TWO-PASS over lock.wheels

Replace the per-wheel git arm with a pre-pass that GROUPS git wheels by checkout
root, then dispatches each group through a single shared helper that mirrors
produce's `auto_data_per_entry` derivation.

**Pass A (pre-loop grouping).** Before the `for lw in &lock.wheels` loop
(currently `mod.rs:4107`), build:

```
// key = git_checkout_root(gs.url, gs.rev, cache_dir)
let mut git_groups: BTreeMap<PathBuf, Vec<&LockWheel>> = BTreeMap::new();
for lw in &lock.wheels {
    if lw.origin == Origin::Built && lw.must_ship {
        if let Some(gs) = &lw.git_source {
            let root = git_checkout_root(&gs.url, &gs.rev, cache_dir);
            git_groups.entry(root).or_default().push(lw);
        }
    }
}
```

This is the SAME keying produce uses (`git_checkout_root`, `source_build.rs:230`),
so the partition is identical to produce's `entry_checkouts` partition.

**Pass B (the per-wheel loop).** Keep the existing loop for `Origin::Index`
(Class 4, `mod.rs:4109-4148`) and `Origin::Built && !must_ship` (Class 2,
`mod.rs:4348-4412`) UNCHANGED — they compose fine (see §5). For the
`Origin::Built && must_ship` arm:

- If `lw.git_source.is_some()`: look up its group in `git_groups`. The FIRST time we
  encounter any member of a group, materialize the WHOLE group via the new helper
  `materialize_git_group` (§2.3) and stash the resulting `EmitWheel`s in a
  `BTreeMap<String /*wheel name*/, EmitWheel>` keyed by `lw.name`. Then (and on every
  subsequent member of the group) pull this wheel's `EmitWheel` out of the stash by
  name and push it. A `processed_roots: HashSet<PathBuf>` guards the
  build-the-group-once semantics. (We must keep emitting in `lock.wheels` order so the
  emit_wheels vector preserves lock ordering — the stash decouples "build once" from
  "emit in order".)
- If `lw.git_source.is_none()`: fall through to the existing legacy/manifest arm
  (`mod.rs:4265-4326`) and finally the Class-3 fall-through (`mod.rs:4327-4346`),
  UNCHANGED — but with the FALL-THROUGH-NOT-BAIL fix from §4 layered on the legacy arm.

The `seen_git_checkout_roots` HashSet (`mod.rs:4104-4105`) and the bail
(`mod.rs:4183-4192`) are DELETED. Grouping replaces the guard.

### 2.3 The shared helper — reuse produce's auto_data derivation EXACTLY

The byte-identity guarantee comes from running the SAME derivation produce runs.
Add a free function that takes a group of lock wheels sharing ONE checkout root and
returns one `EmitWheel` per wheel:

```
async fn materialize_git_group(
    root: &Path,                       // the shared git_checkout_root
    group: &[&LockWheel],              // all wheels of this checkout root, lock order
    config: &RetreadConfig,
    target: &WheelTarget,
    download_dir, source_dir, cache_dir: &Path,
) -> Result<Vec<(String /*name*/, crate::emit_pypi::EmitWheel)>>
```

Inside, MIRROR `mod.rs:2196-2221` line-for-line, but driven by `git_source`
instead of `WheelEntry`:

1. Build the per-member subdir list:
   `subdirs[i] = group[i].git_source.subdirectory.as_deref().unwrap_or(".")`.
2. Carrier selection: choose carrier index = the group member that produce would
   have made first. Produce orders by the GROUP's BTreeMap-over-entry-name order
   (`mod.rs:2167-2174` builds the group as a `Vec` in `effective.retread_wheels`
   iteration order, which is the manifest's `BTreeMap<String,WheelEntry>` order;
   then `mod.rs:2197` iterates that Vec). **Byte-identity does NOT depend on WHICH
   member is carrier** (see §3): every non-carrier ships identical bytes
   (`auto_data=None`), and the carrier's bytes depend ONLY on `skip_subdirs`, which is
   the SAME full union regardless of which member is carrier. So we pick a
   DETERMINISTIC carrier = the lock-order-first member of the group (index 0 after
   the BTreeMap grouping in Pass A preserves lock order within the value Vec). This is
   simpler than reconstructing manifest order and is provably byte-equivalent.
3. For each member `i`:
   - `auto_data[i] = if i == carrier { Some(AutoDataConfig{ checkout_root: root.clone(),
     skip_subdirs: ALL subdirs as PathBuf }) } else { None }`. (skip_subdirs = the
     full union — §3.)
   - synth `WheelEntry { git: Some(gs.url), rev: Some(gs.rev), subdirectory:
     gs.subdirectory, extras: gs.extras, ..Default::default() }` — identical to the
     current single-entry synth (`mod.rs:4200-4215`).
   - call `materialize_and_rewrite(&synth, &lw.name, target, download_dir, source_dir,
     cache_dir, config.relax, &config.git_sources, auto_data[i], EntryAuditInfo::default())`
     — the SAME entry point produce uses (`mod.rs:2224`→`resolve_bundle`→
     `materialize_and_rewrite`). The clone happens ONCE per root inside
     `build_wheel_from_git` because `git_checkout_root` is content-addressed by
     (url,rev) (`source_build.rs:230`, `204`); the second+ member reuse the existing
     clone (cache hit). So "clone once per root" is satisfied by the existing
     source-build cache, NOT by special replay code.
   - build the `EmitWheel` exactly as the current git arm does (`mod.rs:4252-4264`):
     `local_path` from `resolved.url` if `file://`, `requires_dist: lw.requires_dist`,
     `git_source: resolved.git_source`, `remote_url/upstream_url: None`.

The single-entry case (genesis/newton) is now just a group of size 1: carrier=only
member, `skip_subdirs=[its subdir]` — IDENTICAL to today's behavior
(`mod.rs:4226-4227` produced `[subdirectory]`; a 1-element union is the same set).
So genesis/newton replay is unchanged by construction. **This collapses the current
single-entry git arm INTO `materialize_git_group`** — delete the bespoke single-entry
synth at `mod.rs:4165-4264` and route ALL git_source wheels through the group helper.

---

## 3. skip_subdirs CORRECTNESS — the byte-identity crux

**Produce computes** (`mod.rs:2205-2215`): the carrier's `skip_subdirs` = `{ e.subdirectory
or "." : e ∈ group, checkout_root(e) == root }`. For the isaac group that's all 8
subdirs.

**Replay reconstructs** the IDENTICAL set from the lock: every member of the
`git_groups[root]` Vec carries its own `git_source.subdirectory`; the union of those
(defaulting to `.`) IS the produce-time set, because the lock persisted EVERY
member's subdir (`lock.rs:59-62`) and the partition-by-root is identical (§2.2 uses
the same `git_checkout_root`). Formally: produce's group-by-checkout and replay's
group-by-checkout are the same equivalence classes over the same wheels, so the
per-class subdir-union is equal.

**Why carrier choice does not affect bytes** (the simplification that makes this
robust):
- Every NON-carrier ships `auto_data=None` → no phase-1.6 → bytes = pip wheel +
  phase-1.5 source inject, which depend only on (url, rev, subdirectory) — all in
  `git_source`, identical regardless of carrier choice.
- The CARRIER ships `auto_data=Some{root, skip_subdirs=FULL UNION}`. The full union
  is carrier-INDEPENDENT (it's all members' subdirs). `inject_checkout_root_data`
  (`wheel_inject_data.rs:90-139`) is a pure function of (src_wheel, checkout_root,
  skip_subdirs). The src_wheel (the carrier's own pip+inject wheel) depends only on
  the carrier's own (url,rev,subdir); checkout_root + skip_subdirs are
  carrier-independent. So the carrier's bytes are fully determined by
  (carrier's own subdir, the full union, the pinned checkout).
- Therefore: pick carrier = group Vec index 0 (lock-order-first). As long as produce
  ALSO produces exactly one carrier with the full-union skip_subdirs (it does,
  `mod.rs:2201-2215`) and all others None (it does), the MULTISET of emitted wheel
  bytes is identical between produce and replay even if produce's carrier was a
  different member. Each wheel is keyed by name in the lock; the per-NAME bytes match
  because for any given name the (subdir, carrier-or-not, skip_subdirs-if-carrier)
  triple is the same on both sides for that name.

  **Subtle check:** could produce's carrier be e.g. `isaaclab` (subdir
  `source/isaaclab`) while replay's lock-order-first carrier is `isaaclab-assets`
  (subdir `source/isaaclab_assets`)? Then `isaaclab`'s wheel would be a non-carrier
  on one side and carrier on the other → DIFFERENT bytes for the `isaaclab` wheel.
  **This breaks per-name byte-identity.** RESOLUTION: replay MUST pick the SAME
  carrier produce picked. Produce's carrier = first group member in
  `effective.retread_wheels` iteration order (a `BTreeMap<String,_>` →
  lexicographic by entry NAME). The lock does NOT store entry names directly, but
  `LockWheel.name` IS the canonical wheel/pypi name derived from the entry name
  (`canonical_conda_name`, see `mod.rs:3316`). **Carrier selection rule (final):**
  pick the group member whose `lw.name` is lexicographically smallest — this matches
  produce's BTreeMap-by-entry-name order WHEN entry name == canonical wheel name
  (true for the isaac pack: entries `isaaclab`, `isaaclab_assets`, ... → wheels
  `isaaclab`, `isaaclab-assets`, ...; `_`→`-` normalization preserves lexicographic
  order across the set). swe-bee MUST verify empirically (e2e §8) that produce's
  carrier (the wheel whose lock entry has the auto-data `.data/data/lib/` payload)
  == the lexicographically-smallest-name member. If normalization ever reorders
  (e.g. a `.`-subdir entry named to sort differently), fall back to: persist the
  carrier flag in the lock (§7 contingency). For the isaac pack the empirical
  subdirs include `'.'` whose entry name is unknown — **swe-bee must confirm the
  carrier identity from a cold-produce lock before relying on name-sort.**

**Determinism of the union ordering:** `skip_subdirs` is a `Vec`, but
`inject_checkout_root_data` normalizes it into a `HashSet` (`wheel_inject_data.rs:133`)
before use, so the Vec ORDER does not affect output bytes. Replay may build the union
in any order. (Produce builds it in group order; replay in lock order — both fine.)

---

## 4. GUARD FALL-THROUGH SAFETY (replace hard-error with cold fall-through)

The bail at `mod.rs:4183-4192` is a latent build-breaker (it `Err`s, killing the
build). After §2 it is DELETED (grouping handles multi-entry). But the new group
helper must ALSO degrade safely: if a group cannot be cleanly replayed, return
`Ok(None)` from `materialize_from_lock` (fall through to full `resolve_all`), NEVER
`Err`. Concretely, `materialize_git_group` returns `Ok(None)`-equivalent (propagate a
sentinel up so `materialize_from_lock` returns `Ok(None)`) when:

- any member of the group has `git_source == None` while another has `Some`
  (mixed provenance within one root — can't reconstruct the full subdir union); OR
- (contingency, if §7 carrier-flag route is taken) the carrier flag is missing/ambiguous.

A genuine BUILD error (clone fails, pip wheel fails, rattler-build fails) stays an
`Err` — that's a real failure, not a "can't replay" — matching the existing contract
(`mod.rs:4054-4056`). The distinction: provenance-insufficiency ⇒ `Ok(None)` (cold
fall-through, build still works, just slow); build-failure ⇒ `Err`.

Net: the build ALWAYS works. Multi-entry packs replay (fast) when provenance is
complete; otherwise cold-resolve (slow) — never hard-error.

---

## 5. INDEX + GIT MIX (composition with Phase-1 index replay)

The isaac bundle has BOTH `isaacsim` index shadows/wheels (Class 2/4) AND the 8-wheel
git group (Class 1). The redesign composes cleanly because:

- Pass A only collects `Origin::Built && must_ship && git_source.is_some()` wheels into
  groups. Index wheels (`Origin::Index`, `mod.rs:4109-4148`) and relax shadows
  (`Origin::Built && !must_ship`, `mod.rs:4348-4412`) are untouched — they still go
  through their existing per-wheel arms in Pass B.
- The emit order is preserved: Pass B walks `lock.wheels` in order; index wheels emit
  inline, git-group members emit from the stash (built on first encounter) in their
  lock position. So `emit_wheels` is in the SAME order produce wrote the lock — a
  prerequisite for byte-identical lock (the lock is insertion-ordered;
  HANDOFF §PHASE-3 confirms wheels are discovery-ordered).
- `materialize_and_pack` (`mod.rs:4429-4446`) consumes the merged `emit_wheels`
  exactly as before. No change to the conda packaging tail.

So Phase-1 (isaac6 index) + Phase-2 (genesis/newton single git) + Phase-2.5
(isaac multi-git) all coexist in `materialize_from_lock`.

---

## 6. DETERMINISM

Each group member is built by `build_wheel_from_git(url, RESOLVED-SHA, subdir, ...)`
(`source_build.rs:272`) from the lock's pinned 40-char SHA (`lock.rs:55-58`). This is
the SAME deterministic build Phase-2 validated for single-entry: the setuptools_scm
date-suffix guard (`source_build.rs:366`, `398`) and `git fetch --tags` apply per
subdir build. The clone is content-addressed (`git_checkout_root`, `source_build.rs:230`)
so all 8 builds share ONE checkout at the pinned SHA. Per-subdir builds are
independent and deterministic given (SHA, subdir). The carrier's phase-1.6 auto-data
walk is a deterministic `.gitignore`-honoring tree walk of the pinned checkout minus
the fixed skip set (`wheel_inject_data.rs:141+`). No new determinism risk beyond
Phase-2's (which is already empirically byte-identical for genesis/newton).

CAVEAT (carry Phase-2's conditional determinism note): if any isaaclab subpackage
uses setuptools_scm and the recorded build was NOT from a clean tagged commit, the
date-suffix guard warns. The lock stores no wheel sha256 (HANDOFF FINDINGS), so
LOCK byte-identity does not depend on wheel-byte reproducibility regardless; the
import-correctness e2e (§8) covers the runtime contract.

---

## 7. SCHEMA / EMIT — no bump

- **No schema bump.** `GitWheelSource.subdirectory` already persists every member's
  subdir (`lock.rs:59-62`); the full union is reconstructable from the existing
  schema-8 fields. Only replay LOGIC changes. `SCHEMA` stays 8 (`lock.rs:190`).
- **No EMIT_EPOCH bump** ⇒ `[emit-epoch-ok]`. EMIT_EPOCH (`lock.rs:215`) bumps only
  when emitted bytes change for identical inputs (`lock.rs:192-202`,
  `compute_inputs_hash` at `courier.rs:263`). Phase-2.5 changes only the REPLAY path
  (`materialize_from_lock`), which is NOT part of `compute_inputs_hash` and does not
  alter cold-produce bytes. Cold produce is byte-for-byte unchanged (we touch no
  produce code, only delete the replay guard + add the replay group helper). Confirm
  the CI emit-guard regex does not flag `materialize_from_lock` edits (it watches
  plan()/relax/auto_bundle/recipe/courier/lock per HANDOFF §EMIT_EPOCH; replay
  grouping in handler/mod.rs is replay-only — bee adds an `[emit-epoch-ok]` ack to
  the commit if the guard trips on mod.rs).
- **Lock unchanged on disk** for isaac packs across Phase-2.5: the schema-8 lock a
  cold v2.6.0 produce already writes is exactly what replay must reproduce. No
  re-emit, no new field. (Contingency carrier-flag route below WOULD be a schema
  field add ⇒ schema bump; avoid it unless §3's name-sort carrier rule fails the
  e2e.)

**Contingency (only if §3 name-sort carrier ≠ produce carrier empirically):** add
`LockWheel.git_carrier: bool` (schema 8→9, lock-field add ⇒ `[emit-epoch-ok]`, old
locks fall through via the schema gate) set by produce when `auto_data.is_some()`,
read by replay to pick the carrier. This is the clean fallback if entry-name vs
canonical-wheel-name ordering ever diverges. Default route is the name-sort rule (no
schema change); the carrier flag is the documented escape hatch.

---

## 8. TEST + e2e

**Unit / lib (red-pre, green-post):**
1. `git_group_skip_subdirs_reconstruction` (pure, no network): construct a `Vec<LockWheel>`
   with 3 wheels sharing one `git_source.url+rev`, subdirs `["pkg_a","pkg_b","."]`, plus
   one wheel from a DIFFERENT rev. Assert the grouping (§2.2 Pass A logic, factored into a
   testable free fn `group_git_wheels(&lock.wheels, cache_dir) -> BTreeMap<PathBuf,Vec<&LockWheel>>`)
   yields 2 groups, and the size-3 group's reconstructed `skip_subdirs` union ==
   `{"pkg_a","pkg_b","."}`. Red pre-fix (fn doesn't exist), green post.
2. `git_group_carrier_is_lex_smallest_name`: assert carrier selection picks the
   lexicographically-smallest `lw.name` member, deterministically.
3. Single-entry regression: a size-1 group reconstructs `skip_subdirs=[its subdir]`
   (proves genesis/newton path is byte-unchanged — guards against §2.3 collapse
   regressing the single-entry case).

**Live multi-entry parity (mark `#[ignore]`, runs uv/pip like the existing
`build_wheel_from_git_returns_resolved_sha` at `source_build.rs:685`):**
4. `multi_entry_shared_checkout_replay_byte_identical`: build a local git fixture =
   ONE repo with 2 subdirs (`pkg_a/`, `pkg_b/`), each a buildable wheel, plus a
   root-level data file (so the carrier's phase-1.6 has something to ship). Run the
   PRODUCE multi-entry path (`resolve_all` group of 2 entries, or directly invoke the
   two `materialize_and_rewrite` calls with the produce-computed auto_data) → capture
   both wheels' bytes + the synthesized git_source values. Then drive the REPLAY group
   helper (§2.3) from a synthesized `Vec<LockWheel>` with those git_sources → assert
   BOTH wheels' bytes are byte-identical to produce, AND the carrier wheel contains the
   root data file under `.data/data/lib/` while the non-carrier does NOT, AND neither
   wheel ships the OTHER's subdir source under `lib/`. Red-pre: with the OLD single-entry
   logic this either bails (guard) or produces 2 carriers (over-ships); green-post.

**e2e (orchestrator, lukewarm box, all caches incl wheels/ nuked, KEEP
`.pixi/config.toml`):**
5. Regenerate `pixidock isaac-pack` + `isaac-pack-latest` schema-8 locks via a cold
   produce under the v2.x backend (rebuild-local.sh; local-channel prepended to the pack
   backend, reverted after — per HANDOFF VERIFICATION STANDARD). COMMIT the schema-8 isaac
   locks ONLY after replay is proven (HANDOFF §PHASE-2.5 currently forbids committing them).
6. Lukewarm replay of isaac-pack: assert `build_v1: replayed from lock` PRESENT;
   group helper log present ("re-source-building git wheel ... class 1" for each of the
   8); ZERO derivation logs (no `auto-bundled`, no `resolvo solve finished`, no probe
   trace); `<pack>/wheels` repopulated from EMPTY; `git diff --exit-code` CLEAN on
   `retread-isaac-pack.lock.json` (byte-identical); `python -c "import isaacsim; import
   isaaclab"` succeeds (OMNI_KIT_ACCEPT_EULA=YES). Repeat for isaac-pack-latest.
7. Re-confirm NO regression: genesis-pack, newton-pack-latest, examples/isaac6 all still
   replay byte-identically (the §2.3 single-entry collapse + Pass-B index/shadow arms
   must be inert for them).

---

## 9. COMMIT ORDERING (each commit keeps `cargo test --lib` + clippy + fmt green)

1. **`refactor: extract group_git_wheels + materialize_git_group (no behavior change yet)`**
   Add the free fn `group_git_wheels` (Pass A) and `materialize_git_group` (§2.3), but do
   NOT wire them in yet. Route the EXISTING single-entry git arm through
   `materialize_git_group` (size-1 groups) to prove the collapse is byte-neutral. Add lib
   tests #1–#3 (red→green). Single-entry guard STILL present (harmless for size-1).
   `[emit-epoch-ok]` (replay-only).

2. **`feat: multi-entry shared-checkout git replay (group by checkout root)`**
   Wire Pass A grouping + Pass B stash into `materialize_from_lock`; DELETE
   `seen_git_checkout_roots` (`mod.rs:4104-4105`) and the bail (`mod.rs:4183-4192`).
   Carrier = lex-smallest name (§3). Add live parity test #4 (`#[ignore]`).
   `[emit-epoch-ok]`.

3. **`fix: replay falls through (Ok(None)) on incomplete git-group provenance, never Err`**
   §4 safety: mixed-provenance group ⇒ `Ok(None)`; build errors stay `Err`. Update the
   `lock.rs:41-50` Phase-2-limitation doc-comment on `GitWheelSource` to describe the
   new multi-entry support + the carrier rule. `[emit-epoch-ok]`.

4. **`docs: version bump + Phase-2.5 notes`** version 2.6.0→2.7.0; HANDOFF update.

5. **(orchestrator, NOT a code commit)** e2e §5–§7. On PASS: separate commit in the
   pixidock repo committing the regenerated schema-8 isaac locks + (if pin gating needs
   it) bump the isaac-pack backend pin to the new published version. On FAIL: bee
   re-diagnoses; if §3 carrier rule diverged → apply the §7 contingency (schema 8→9
   `git_carrier` flag) as commit 2b, re-run.

**Hardest parts (per the brief):** #2 (reuse produce's `materialize_and_rewrite` +
`AutoDataConfig` at replay so byte-identity holds by construction — the helper calls
the IDENTICAL function produce calls) and #3 (skip_subdirs full-union reconstruction +
carrier selection so per-NAME bytes match — the carrier-choice analysis in §3 is the
correctness lynchpin; the lex-smallest-name rule MUST be empirically confirmed against
a cold-produce isaac lock before commit 2 is trusted).
