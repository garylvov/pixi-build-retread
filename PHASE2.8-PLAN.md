# PHASE 2.8 PLAN — Strip dead/orphan direct-URL Requires-Dist from emitted wheel METADATA

Branch: `courier` (HEAD ac7cfa1, v2.7.1, schema 9, EMIT_EPOCH 4).
Author: solution-architect. Audience: the-grizzly (audit before any code).
Toolchain: `PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH" cargo` (NO host cargo).

This is an **additive root-cause fix** the user explicitly requested. General lockfile
replay (Phases 1–2.7) is DONE + shipped. This plan does NOT touch the replay machinery;
it changes the EMITTED wheel METADATA so the bundle is *complete* (uv can resolve it
without a manual workaround). No band-aids, no per-pack special cases.

---

## 0. The bug, root-caused to ground (every claim verified against code via henry)

### 0.1 Symptom

`pixidock isaac-pack-latest` env `isaaclab-gpu-latest` fails `import isaacsim`. The conda
package links fine (post-link ends in `|| echo 'failed'`, so the link "succeeds" even when
the installer aborts). The post-link runs `retread install --lock … --prefix …`, which
builds a `uv pip install` command (src/installer.rs `build_uv_args` lines 79–151) and
spawns it (installer.rs:300). uv aborts the **whole** install with:

```
× Failed to resolve dependencies for `isaaclab-mimic` (v1.3.2)
╰─▶ Package `robomimic` was included as a URL dependency. URL dependencies must be
    expressed as direct requirements or constraints. Consider adding
    `robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0` …
```

The whole bundle install aborts → `isaacsim` never lands in site-packages → import fails.

### 0.2 Why uv aborts

The **shipped** `isaaclab-mimic` (v1.3.2) wheel METADATA carries an **unconditional orphan**
`Requires-Dist: robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0`
— grizzly G-0 unzipped the actual wheel and confirmed it has **NO `; extra == "robomimic"`
marker AND NO `Provides-Extra: robomimic` header**. uv refuses any resolution graph that
contains a git-URL dependency unless it is a *top-level direct requirement* — and
`robomimic` is neither in the bundle nor a top-level root requirement, so uv aborts.

### 0.3 What the wheel actually declares (grizzly G-0, empirically verified)

`isaaclab-mimic` is **built from git source**
(`isaaclab-mimic = { from = "isaaclab", subdirectory = "source/isaaclab_mimic" }`) via
`build_wheel_from_git` → `uv build` (src/handler/mod.rs:3405 named-git / 3478 inline-git,
inside `materialize_and_rewrite` mod.rs:3367–3738).

**Empirical truth (grizzly unzipped the real wheels):**

- **isaaclab_mimic-1.3.2** (THE failing isaac-pack-latest version): ships
  `Requires-Dist: robomimic @ git+…@v0.4.0` with **NO `; extra ==` marker and NO
  `Provides-Extra: robomimic` header**. Upstream made it a genuinely **UNCONDITIONAL
  orphan direct-URL dep** in 1.3.2. It is not extras-gated at all.
- **isaaclab_mimic-1.0.16 / 1.2.3** (older, e.g. the regular `isaac-pack`): ship the
  **marked** form (`; extra == "robomimic"` + `Provides-Extra: robomimic`). This is the
  setuptools `extras_require`-with-`@ git+` story — but it is NOT the 1.3.2 bug.

So the earlier "setuptools dropped the marker" narrative described the 1.2.x wheels, not the
1.3.2 wheel that actually breaks. The 1.3.2 line is unconditional by upstream design.

### 0.4 The fix is marker-INDEPENDENT — the true invariant

The fix does NOT and CANNOT key on the `; extra ==` marker (it is absent in the failing
1.3.2 wheel, and present-but-irrelevant in the 1.2.x wheels). The correct, broader invariant:

> **A direct-URL `Requires-Dist` whose target is ABSENT from the resolved bundle closure is
> an orphan URL edge that uv will reject.** retread's BFS followed every URL edge it was
> supposed to follow (base deps + every *active*-extra URL via `seed_worklist` +
> `pep508_extra_dep`, auto_bundle.rs:614–656 / 861–896). A URL target that is still absent
> from the closure is one retread **deliberately did not bundle** — whether because its
> gating extra was inactive OR because it was an unconditional dep retread chose not to
> follow (the 1.3.2 case). **Either way the package is not in the closure, not wanted, and
> its requirer-line must be stripped so the bundle is resolvable.**

The predicate is therefore purely **bundle-membership-based** and entirely independent of any
marker or `Provides-Extra` header. This is why it handles both the unconditional 1.3.2 orphan
AND the marked 1.2.x form identically: both are direct-URL lines whose target is absent from
the closure.

### 0.5 What retread reliably knows (all already in `plan()`)

For the strip decision, retread knows two things about each `Requires-Dist` line in
`EmitWheel.requires_dist`, both from data already in `plan()` — **no marker read needed**:

1. **It is a direct-URL / git requirement**:
   `matches!(req.version_or_url, Some(VersionOrUrl::Url(_)))` (already tested at
   emit_pypi.rs:231; non-URL lines `continue` BEFORE the bundle lookup, so they are never
   strip candidates).
2. **Its target name is NOT in the bundle closure**:
   `bundle_versions.get(name).is_none()` (the `None` arm at emit_pypi.rs:266 — exactly where
   the WARN fires today).

**Why "URL + absent" is the right predicate (not the marker):** the marker is unavailable in
the real failing wheel (1.3.2 has none). But the bundle-membership map (`bundle_versions`,
emit_pypi.rs:214) is the ground truth of what retread actually resolved and will ship. A URL
target absent from that map is, by construction, an edge retread did not follow — so the
shipped wheel must not advertise it. Bundle-absence is a sound, deterministic, replay-stable
predicate for "dead/orphan URL requirement". The marker is neither available nor needed.

**The historical WARN string** in the handoff log
(`robomimic @ git+…@v0.4.0 ; extra == "robomimic"`) was almost certainly emitted by an OLDER
1.2.x build whose line carried the marker; the predicate is identical for both and the marked
vs unmarked distinction is immaterial to the code.

---

## 1. CLOSURE-MEMBERSHIP ANALYSIS — how retread decides "in the bundle or not"

The strip decision is **bundle-membership-based**, NOT marker- or extras-based. retread does
NOT maintain (and the fix does NOT need) an explicit "set of activated extras". The ground
truth is simply the set of wheels that ended up in the bundle:

- The BFS followed every edge it was supposed to: base deps + every *active*-extra URL via
  `seed_worklist` + the marker evaluator `pep508_extra_dep` (auto_bundle.rs:614–656 /
  861–896, which builds a `MarkerEnvironment` via `default_marker_env(DEFAULT_PYTHON)` and
  evaluates `req.marker.evaluate(env, &[extra])`).
- The resulting closure = the `EmitWheel`s passed to `plan()`. retread does not re-read the
  marker at emit time; the closure already encodes every follow decision.

**Decision:** reuse the existing bundle-membership map (`bundle_versions`, emit_pypi.rs:214).
A direct-URL target is "in the bundle" iff `bundle_versions.get(name).is_some()`, "orphan"
iff `None`. This is precisely the `Some` vs `None` split already present at
emit_pypi.rs:235/266. No new closure tracking, no marker evaluation in `plan()`, and no
order-dependent-BFS hazards (the grizzly's Phase-3 concern) — we only *read* the already-final
closure.

> **A direct-URL `Requires-Dist` is "kept/rerouted" iff its target name is present in the
> bundle (Some-arm); it is "dead/orphan → stripped" iff its target name is absent
> (None-arm).** Marker presence is irrelevant.

### 1.1 SOUNDNESS BOUNDARY — `config.drop_deps` is NOT consulted (Amendment 2)

The strip predicate is membership in `bundle_versions` only. It does **NOT** consult
`config.drop_deps`. This is deliberate and matches pre-existing behavior:

- `config.drop_deps` feeds a *different* mechanism than the extras-BFS: the
  `auto_bundle_transitives` skip set (auto_bundle.rs:114–120) and the emit conda-run-dep
  filter (mod.rs:3930). It does NOT feed `seed_worklist`'s `seen` set
  (mod.rs:2678), which in `resolve_bundle` contains only the primary wheel's own name
  (mod.rs:2597 / 2638).
- Consequence (pre-existing quirk, NOT introduced here): a name in `drop_deps` that is ALSO
  an *active-extra* URL target still gets built + pinned via the `Some`-arm (it is in the
  bundle), NOT stripped. The new strip only fires for URL targets genuinely ABSENT from the
  closure.

**We do NOT wire `drop_deps` into the BFS `seen` set** — that is a separate blast radius and
out of scope for Phase 2.8. The `EmitPlan.drop_url` doc comment (§2.2) MUST state explicitly
that the predicate is bundle-membership-based and does not consult `drop_deps`. G-1 (§3.3)
adds a guard: confirm no committed pack puts a URL-dep name in `drop_deps` in a way that the
new strip would change its handling.

---

## 2. THE FIX — strip dead direct-URL Requires-Dist from the EMITTED wheel METADATA

### 2.1 Recommended approach (decisive; see §5 for the comparison)

**STRIP the dead/orphan direct-URL line from the emitted wheel METADATA.** When a wheel's
`Requires-Dist` line is (a) a direct-URL/git requirement AND (b) its target name is NOT in
the bundle closure, the emitted wheel METADATA must OMIT that line entirely. With the line
gone, uv has no orphan URL dependency and resolves the bundle cleanly. robomimic is genuinely
not in the closure (retread did not bundle it — unconditional-but-not-followed in 1.3.2;
inactive-extra in 1.2.x — same outcome), so removing it is correct, not a workaround. The
decision is marker-INDEPENDENT (§0.4): it fires identically for the unconditional 1.3.2 orphan
and the marked 1.2.x form.

### 2.2 Where the decision is made: `plan()` produces a *drop set*

`plan()` (emit_pypi.rs:213) is the single point that already classifies every
`Requires-Dist` line and already distinguishes the dead case (the `None` arm,
emit_pypi.rs:266). Extend `EmitPlan` with a drop set:

```rust
pub struct EmitPlan {
    pub ship: HashSet<String>,
    pub overrides: BTreeMap<String, String>,
    /// PEP 503 names of DEAD/ORPHAN direct-URL Requires-Dist targets:
    /// direct-URL (git/url) requirements whose target name is ABSENT from
    /// the resolved bundle closure (`bundle_versions`). retread did not
    /// follow them into the bundle -- whether because the gating extra was
    /// inactive (e.g. isaaclab_mimic 1.2.x marked form) OR because they are
    /// unconditional deps retread chose not to bundle (e.g. isaaclab_mimic
    /// 1.3.2, which carries NO `; extra==` marker and NO `Provides-Extra`).
    /// The decision is MARKER-INDEPENDENT: bundle-membership is the sole
    /// predicate. Their Requires-Dist lines are STRIPPED from emitted wheel
    /// METADATA so uv does not see an orphan URL dependency and abort.
    ///
    /// NOTE: this is bundle-MEMBERSHIP-based and does NOT consult
    /// `config.drop_deps` (which feeds the auto_bundle_transitives skip set
    /// + emit conda-run-dep filter, NOT the extras BFS `seen` set). A
    /// drop_deps name that is also an active-extra URL target stays in the
    /// bundle and is pinned via the Some-arm, not stripped (pre-existing
    /// behavior, out of scope). Phase 2.8.
    pub drop_url: HashSet<String>,
}
```

In the `None` arm (emit_pypi.rs:266–275), in addition to the existing WARN (downgrade it to
a one-line INFO that says "stripping dead/orphan direct-URL requirement", since it is no
longer a manual-action warning), insert `drop_url.insert(name)`. Return it on the struct
(emit_pypi.rs:337).

**Predicate (exact, see §4):** a name enters `drop_url` iff, for some wheel, a
`Requires-Dist` line parses to a requirement with `version_or_url == Some(Url(_))` AND
`bundle_versions.get(name).is_none()`. This is byte-for-byte the existing `None`-arm
condition; we are only *recording* it instead of merely warning.

### 2.3 Where the strip happens: teach the METADATA mapper a DROP outcome

**Finding (henry):** the mapper passed to `rewrite_metadata_text_with` (wheel_rewrite.rs:153)
currently returns `Option<String>`: `None` = leave line UNCHANGED, `Some(s)` = REPLACE with
`s`. **There is no DROP path** — the loop always emits either the original or the
replacement line (wheel_rewrite.rs:158–180). So "omit this line" is **not currently
expressible**. We must add a third outcome.

Introduce a 3-way result and thread it through the one generic rewrite primitive:

```rust
// src/wheel_rewrite.rs
pub enum LineAction {
    Keep,            // emit the original line unchanged
    Replace(String), // emit "Requires-Dist: {0}\n"
    Drop,            // omit the line entirely (and its RECORD-irrelevant METADATA presence)
}
```

Change `rewrite_metadata_text_with` (wheel_rewrite.rs:153) and `rewrite_wheel_with`
(wheel_rewrite.rs:45) to take `map: impl Fn(&str) -> LineAction`. In the per-line loop
(wheel_rewrite.rs:158–180):

- `Keep` → `out.push_str(line)` (current `None` behavior).
- `Replace(s)` → current `Some(s)` behavior (re-emit `Requires-Dist: {s}` with the line's
  original terminator).
- `Drop` → emit NOTHING (skip the line; `continue`).

`override_line_map` (emit_pypi.rs:356) becomes a closure returning `LineAction`, capturing
`drop_url` in addition to `overrides`/`conda_capable`:

- If `name` is in `drop_url` → return `LineAction::Drop`. (Check this FIRST, before the
  overrides lookup, so a dropped name never also gets a pin.)
- Else mirror the existing logic EXACTLY: `overrides` hit → `Replace(rebuilt)` ONLY when
  `rebuilt != line`, else `Keep` (preserves the existing byte-identity short-circuit at
  emit_pypi.rs:373 — `(rebuilt != line).then_some(...)`); cap-only conda-capable →
  `Replace(rebuilt)` ONLY when `rebuilt != line`, else `Keep` (emit_pypi.rs:391); otherwise
  `Keep`.

**CRITICAL byte-parity rule (Amendment 3):** the refactor is emit-neutral ONLY if BOTH mappers
preserve the `(rebuilt != line).then_some(...)` short-circuit: return `Keep` when the line is
unchanged, and **NEVER** `Replace(identical-bytes)`. courier keys `ShadowSrc`/`did_change` on
whether the mapper "changed" the line (the old `Some` vs `None` distinction). A
`Replace(same-bytes)` would read as "changed" → flip a wheel from no-rewrite to rewrite →
drift EVERY lock. So `Keep` ≡ the old `None` (no change), `Replace(s)` ≡ the old `Some(s)`
(s strictly differs from the input), `Drop` ≡ the new omit. See §2.4 for the parity test.

**Why the mapper, not the excludes/constraints file (installer.rs:247–276):** the excludes
file is generated *fresh at every `retread install`* from `uv pip list` (it is NOT
committed, NOT in the lock). Putting the fix there would (a) be invisible to the lock /
replay, (b) require robomimic to already be installed to exclude it (it never is), and
(c) not stop uv's *resolution* abort — excludes block uv from *replacing* conda-provided
packages, they do not remove a Requires-Dist edge from the graph. uv aborts at
**resolution**, before install, on the orphan URL edge. The only place that removes the
edge is the wheel METADATA itself. So the strip MUST be in the emitted wheel bytes.

### 2.4 Why this is byte-stable and replay-reproducible

- The strip is applied during `courier::stage` at the SAME rewrite call that already bakes
  the override table into shipped wheel bytes: `override_line_map(&overrides, &conda_cap)`
  → `rewrite_wheel_with(src, dst, &m)` (courier.rs:535–536 no-cache; courier.rs:522–528 +
  `shadow_cache_stage` courier.rs:354 cache path). The drop is part of the SAME
  deterministic single rewrite — no second pass, no new file.
- The stripped METADATA is baked into the shipped wheel. On **replay**, the wheel is
  re-materialized by re-source-building from the pinned git rev and re-running the SAME
  `materialize_and_rewrite` + courier stage with the SAME `plan()` output (the lock carries
  `git_source`, and `plan()` is a pure function of `(requires_dist, version, must_ship,
  conda_capable)` — and now also `bundle_versions`, all reconstructed identically). So the
  same line is stripped identically on replay → byte-identical wheel → byte-identical lock.
- robomimic was never in the bundle, so it is never a `ship`/`override`/lock entry; its
  absence is already part of the deterministic closure. Stripping its requirer-line changes
  the `isaaclab-mimic` wheel bytes consistently on both cold and replay.

### 2.4 REFACTOR BYTE-PARITY GUARD (Amendment 3 — MANDATORY regression test)

The `Option<String>` → `LineAction` refactor touches BOTH mapper bodies and both primitive
signatures. It MUST be provably emit-neutral for the no-change case:

- Mappers changed: the relax lambda (`rewrite_wheel` → `relax_pep508`, wheel_rewrite.rs:35–44
  + the direct test call at wheel_rewrite.rs:461) AND `override_line_map`'s return
  (emit_pypi.rs:356).
- Primitives changed once each: `rewrite_wheel_with` (wheel_rewrite.rs:45) +
  `rewrite_metadata_text_with` (wheel_rewrite.rs:153). courier call sites update for free
  (they pass the closure, not a concrete `Option`/`LineAction`).
- 1:1 mapping in step-1: `None` → `Keep`, `Some(s)` → `Replace(s)`. No `Drop` is introduced
  until step-2. Step-1 alone must produce byte-identical wheels for every existing input.

**Required regression test** (beyond output-byte checks): drive an UNCHANGED wheel
(no override entry, no cap-strip, no drop) through the refactored `rewrite_wheel_with`/
`rewrite_metadata_text_with` and assert **`did_change`/sha PARITY** — i.e. the refactored
primitive reports "no change" (and the courier `ShadowSrc` decision is identical) for a wheel
that the pre-refactor `Option`-based mapper also reported "no change" for. This guards against
a `Replace(identical)` silently flipping the did_change signal and drifting every lock. (Pure
output-byte equality is necessary but NOT sufficient — the did_change SIGNAL is what courier
keys on.)

---

## 3. REPLAY SAFETY + EMIT_EPOCH

### 3.1 This change IS emit-affecting

It changes EMITTED wheel METADATA (removes a `Requires-Dist` line) for any pack whose
bundle contains a dead/orphan direct-URL requirement (isaac-pack-latest via the 1.3.2
unconditional orphan; isaac-pack likely via the marked 1.2.x form). Emitted bytes change ⇒ the
content-addressed build string changes ⇒ a cold-solve replay against an OLD committed lock
would reuse a stale lock. Per the EMIT_EPOCH doc (lock.rs:237–260) and the CI guard
(.github/workflows/ci.yml:108–125, regex matches `emit_pypi.rs`, `wheel_rewrite.rs`,
`courier.rs`), this REQUIRES bumping `EMIT_EPOCH`.

### 3.2 EMIT_EPOCH 4 → 5, and the consequence

- `EMIT_EPOCH` is mixed into `compute_inputs_hash` (lock.rs:316–317, `h.update(b"--epoch--")`
  + `h.update(emit_epoch.to_le_bytes())`; caller courier.rs:909 passes `crate::lock::EMIT_EPOCH`).
- Bumping 4→5 **invalidates `inputs_hash` for ALL committed locks**. Every committed lock
  (every example + every pixidock pack) will fail the `load_replayable_lock` inputs_hash
  gate → cold re-solve until regenerated. This is the same blast radius as a schema bump.
- **Schema stays 9** (no new lock FIELDS; `drop_url` lives only in `EmitPlan`, an in-memory
  produce-time structure, never serialized — the strip is baked into wheel bytes, the lock
  needs no new field). So the gate is the EMIT_EPOCH/inputs_hash one, not schema.

### 3.3 G-1 MANDATORY pre-bump cold-produce-ALL (the grizzly flagged this in the
foundational plan)

An EMIT_EPOCH bump forces a cold re-solve of every pack on the next build, AND this fix
changes emitted bytes for any pack carrying a dead/orphan URL line — which now includes BOTH
isaac packs (isaac-pack-latest via 1.3.2; isaac-pack likely via the marked 1.2.x robomimic
line — the predicate strips both, see §4). The grizzly's standing rule: **before committing
the 4→5 bump, cold-produce ALL packs to ensure none turn red AND all still import.**
Concretely, before the bump is finalized:

1. Build the v2.8.0 backend (rebuild-local.sh).
2. **Cold-produce EVERY committed pack** under the new backend: examples genesis, isaac6,
   gigastrap-isaac; pixidock genesis, newton, isaac-pack, isaac-pack-latest.
3. Assert each cold produce SUCCEEDS (no pack turns red from the re-solve) AND the env
   imports (the strip changes both isaac packs' emitted bytes, so isaac-pack must be confirmed
   green + importable, not just isaac-pack-latest). If any pack goes red or fails to import,
   STOP and re-diagnose before merging — the bump itself must not break a pack.
4. **drop_deps guard (Amendment 2):** confirm no committed pack lists a URL-dep name in
   `config.drop_deps` such that the new strip would change its handling (grep each pack's
   manifest `[retread]` / drop_deps for any name that is also a direct-URL Requires-Dist
   target). Expected: none — but verify, since drop_deps does NOT feed the strip predicate
   (§1.1) and a collision would be a latent surprise.

This is **lower-risk** than the foundational-plan G-1: this fix only REMOVES a dead edge — it
never adds constraints, so it cannot manufacture constraint-accumulation conflicts (the
pillow-style conflict the foundational plan worried about is impossible here). G-1 remains
mandatory only because the EMIT_EPOCH bump forces every pack to cold-re-solve, and any
re-solve carries residual upstream-index drift risk.

---

## 4. NO-REGRESSION — the precise strip predicate

`drop_url` membership (and therefore the `LineAction::Drop` decision) requires ALL of:

1. The `Requires-Dist` line parses (`uv_pep508::Requirement::from_str(line).is_ok()`).
2. `req.version_or_url == Some(VersionOrUrl::Url(_))` — it is a direct-URL/git requirement.
3. `bundle_versions.get(req.name).is_none()` — the target is NOT in the bundle closure.

Confirmed-good (grizzly): the `None`-arm is reachable ONLY for direct-URL deps absent from
the bundle — non-URL lines `continue` at emit_pypi.rs:231 BEFORE the bundle lookup, so they
never reach the drop decision. This MUST NOT strip any of:

- **Normal (non-URL) deps, extras-gated or not** (`foo>=1 ; extra == "x"`, `bar<2`): fail
  (2) — not a URL; `continue` before the lookup (emit_pypi.rs:231). uv accepts these fine;
  left untouched. Marker presence is irrelevant — it is the URL form, not the marker, that
  gates the strip.
- **Active-extra (or otherwise-followed) URL deps** (target IS in the bundle): fail (3) —
  `bundle_versions` hit → goes through the existing `Some` arm; `rebuild_requirement`
  (wheel_rewrite.rs:326, test :577) excises the URL to `name==version` so find-links serves
  it — NOT dropped. (This is the symmetric case; see §6.)
- **Unconditional URL deps that ARE top-level `[retread-wheels]` entries**: such a wheel is
  IN the bundle (it is a config entry), so any Requires-Dist line naming it hits (3)'s
  `Some` arm and is not dropped; and the entry itself is shipped + top-level so uv accepts
  it. (A config-entry URL is exactly the "direct requirement" uv wants.)
- **`python`**: `override_line_map` already early-returns on `name == "python"`
  (emit_pypi.rs:363); keep that guard ahead of the drop check.

**isaac-pack blast radius (grizzly factual correction):** the OLDER isaaclab in `isaac-pack`
ships the MARKED 1.2.x robomimic line (`robomimic @ git+…@v0.4.0 ; extra == "robomimic"` +
`Provides-Extra: robomimic`). The predicate STILL strips it — the marker is irrelevant; it is
a direct-URL line whose target is absent from the closure. So **isaac-pack's emitted bytes
ALSO change** (not just isaac-pack-latest). This is correct (robomimic was never bundled), but
it WIDENS the regen blast radius: G-1 (§3.3) MUST cold-produce isaac-pack and confirm it stays
green AND still imports, and isaac-pack's committed lock must be regenerated at EMIT_EPOCH 5.

### 4.1 Determinism / idempotence

- `bundle_versions` is built from the same `EmitWheel` set on cold and replay → the same
  names hit `None` → the same `drop_url` → the same lines stripped.
- The drop check runs BEFORE the overrides lookup, so a dropped name is never double-handled.
- Re-running the rewrite on an already-stripped wheel is a no-op (the line is already gone;
  `map` is never invoked for a line that does not exist). Idempotent.

---

## 5. ALTERNATIVES CONSIDERED — strip vs promote vs bundle (DECISIVE recommendation)

| Option | What it does | Verdict |
|---|---|---|
| **A. Strip orphan** (RECOMMENDED) | Remove the dead/orphan direct-URL `Requires-Dist` line (target absent from bundle, marker-independent) from emitted wheel METADATA. | **Correct.** robomimic is genuinely not in the closure (retread did not bundle it — unconditional-not-followed in 1.3.2, inactive-extra in 1.2.x). uv sees a complete, resolvable graph. No unwanted package installed. Deterministic, replay-stable, baked into wheel bytes. Minimal: reuses the existing `None`-arm classification + the existing rewrite primitive (one new `LineAction::Drop` outcome). |
| **B. Promote to top-level** | Add `robomimic @ git+…@v0.4.0` to the uv install root requirements / constraints (installer.rs / share files). | **Wrong.** Installs an UNWANTED package (robomimic + its whole git-built closure) into every isaac-pack-latest env, even though the `robomimic` extra was never requested. Bloats the env, can introduce conflicts, and is not in the lock closure (poisons the bundle-completeness invariant). Also the constraints file is install-time/uncommitted → not replay-faithful. |
| **C. Bundle the git dep** | Follow robomimic into the bundle via Phase-2 git machinery (build_wheel_from_git → SHA → git_source lock wheel → top-level). | **Wrong for THIS bug** (same "installs unwanted package" objection as B, plus it would re-resolve robomimic's transitive closure and grow every lock). This is the *symmetric* path and is only correct when the extra IS active — see §6. |

**RECOMMENDATION: Option A (strip orphan).** It is the deepest correct fix: it makes the
emitted bundle internally consistent (every URL edge either resolves within the bundle or
is provably dead and removed), with zero unwanted installs and full replay byte-identity.
B and C both install a package the user never asked for, which is a behavior change, not a
fix.

---

## 6. SYMMETRIC CASE (extra IS activated) — design now, GATE the implementation

When the gating extra IS activated, the BFS already followed the URL requirement and
`build_wheel_from_git`-built it into the bundle (Phase 2 machinery: resolved rev → SHA →
shipped as a `git_source` lock wheel). In `plan()` that target then HITS the `Some` arm
(emit_pypi.rs:236): exact-pin override + force-ship + (today) the consumer must reference it.
The existing code path already handles "URL target IN bundle" — that branch is NOT changed
by this plan.

**The genuinely-missing piece for the active case** is making uv accept the in-bundle git
wheel as a *direct requirement* (uv's rule applies to git URLs even when find-links serves
the wheel). Today emit_pypi rewrites the URL Requires-Dist line to a version pin
(`rebuild_requirement`, override path) so find-links satisfies it by name — which already
sidesteps uv's git-URL rejection for the *active* case. So the active case is ALREADY
handled by the existing override→pin rewrite; no new work is required for the live packs.

**Decision: implement ONLY the strip (orphan) branch now.** The active branch is already
covered by the existing override-to-pin rewrite (verify in §7). If a future pack surfaces
an active-extra git-URL that the override path does NOT cover, extend then — the
`drop_url`/`LineAction` seam does not paint us into a corner (the kept case is the `Some`
arm; the dropped case is the `None` arm; they are disjoint by `bundle_versions` membership).
Document this in the `EmitPlan.drop_url` doc comment so the boundary is explicit.

---

## 7. VERIFICATION + TESTS

### G-0 (grizzly, DONE) — wheel METADATA empirically inspected

Grizzly unzipped the actual wheels: **isaaclab_mimic-1.3.2** ships
`Requires-Dist: robomimic @ git+…@v0.4.0` with **NO `; extra ==` marker and NO
`Provides-Extra: robomimic` header** (genuinely unconditional orphan); **isaaclab_mimic
1.0.16 / 1.2.3** ship the MARKED form. This confirms the fix must be marker-INDEPENDENT and
key on bundle-absence only (§0.3/§0.4). No further G-0 action needed.

### Unit tests (src/emit_pypi.rs + src/wheel_rewrite.rs `#[cfg(test)]`)

1. **`drop_predicate_strips_unconditional_orphan_url`** (Amendment 1 — mirrors the REAL 1.3.2
   bug): `plan()` over a wheel whose `requires_dist` has
   `robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0` with **NO
   `; extra ==` marker** where `robomimic` is NOT among the wheels → `drop_url` contains
   `robomimic`; `overrides`/`ship` do NOT. This is the load-bearing test (the prior plan's
   tests only covered the marked case and would have missed the actual failing wheel).
2. **`drop_predicate_strips_marked_orphan_url`** (the 1.2.x form): same target absent, line is
   `robomimic @ git+…@v0.4.0 ; extra == "robomimic"` → `drop_url` contains `robomimic`. Proves
   marker-independence: both marked and unmarked orphans strip identically.
3. **`drop_predicate_keeps_active_url`**: a URL line whose target `robomimic` IS in the wheel
   set → `drop_url` empty; `overrides["robomimic"]` is an exact pin (existing `Some`-arm
   behavior unchanged).
4. **`drop_predicate_ignores_non_url_extra_dep`** (keep from prior plan): `requires_dist` has
   `foo>=1 ; extra == "x"` (non-URL), `foo` not in bundle → NOT in `drop_url` (`continue`s at
   the URL test, emit_pypi.rs:231, before the bundle lookup). Left for uv (uv accepts non-URL
   lines, marked or not).
5. **`drop_predicate_ignores_url_config_entry`**: a URL-named target that IS a top-level
   bundle entry (in `bundle_versions`) → NOT dropped (`Some` arm).
6. **`line_action_drop_omits_line`** (wheel_rewrite.rs): feed
   `rewrite_metadata_text_with` a METADATA text with three Requires-Dist lines and a map
   that returns `Drop` for the middle one → output OMITS exactly that line, Keeps the other
   two byte-identically, headers/body intact, RECORD updated consistently.
7. **`line_action_refactor_unchanged_wheel_parity`** (Amendment 3 — the byte-parity guard):
   drive an UNCHANGED wheel (no override entry, no cap-strip, no drop) through the refactored
   `rewrite_wheel_with`/`rewrite_metadata_text_with` and assert the **did_change/sha signal**
   reports "no change" — i.e. the result is byte-identical AND the courier `ShadowSrc`
   decision is identical to the pre-refactor `Option`-based path. This is the load-bearing
   guard: a `Replace(identical)` must NOT flip a no-rewrite wheel to a rewrite (which would
   drift EVERY lock). Output-byte equality alone is insufficient — assert the did_change
   SIGNAL specifically. Pair with a `Keep`+`Replace`-mix case to confirm `Replace(s)` still
   changes only when `s != line`.
8. **`override_line_map_drop_precedence`**: a name in BOTH `drop_url` and `overrides`
   resolves to `Drop` (drop wins; no double-handling).

### Live e2e (orchestrator, the decisive seal) — `scripts/replay-e2e.sh` style

Target: pixidock isaac-pack-latest (`isaaclab-gpu-latest`). Sequence:

1. Build v2.8.0 backend (rebuild-local.sh).
2. **G-1 cold-produce ALL packs** (§3.3): cold-produce examples genesis/isaac6/
   gigastrap-isaac + pixidock genesis/newton/isaac-pack/isaac-pack-latest; assert none turn
   red AND all import; run the drop_deps guard (§3.3 step 4).
3. Cold-produce isaac-pack-latest under v2.8.0 → schema-9 lock (EMIT_EPOCH 5).
   Assert the staged `isaaclab-mimic` wheel METADATA has NO `robomimic` line
   (`unzip -p … METADATA | grep -c robomimic` == 0). Do the same assertion for isaac-pack's
   staged isaaclab-mimic wheel (the marked 1.2.x line must ALSO be gone).
4. **Lukewarm replay** (nuke all caches incl wheels/, KEEP `.pixi/config.toml`):
   - `build_v1: replayed from lock` present; derivation = 0 (no auto-bundle / resolvo /
     probe); wheels/ repopulated from EMPTY.
   - `git diff --exit-code` on the lock CLEAN (byte-identical).
   - The replayed `isaaclab-mimic` wheel METADATA is markerless robomimic-FREE too (strip
     re-materialized on replay).
5. **Post-link install succeeds**: `retread install` completes (uv does NOT abort);
   `isaacsim` lands in site-packages.
6. `OMNI_KIT_ACCEPT_EULA=YES python -c "import isaacsim; import isaaclab"` → OK.
7. **Regression**: re-run genesis + newton + isaac6 + gigastrap-isaac + pixidock genesis +
   newton under v2.8.0 → all byte-identical + import OK (G-1 already cold-built them; confirm
   replay too). For **isaac-pack** specifically (emitted bytes CHANGE — marked 1.2.x strip):
   cold→replay→byte-identical + post-link install succeeds + import isaacsim/isaaclab OK (it
   imported fine before; confirm the strip did not regress it).

ON PASS: regenerate + commit ALL example + pixidock locks at EMIT_EPOCH 5 (the bump forces
this); publish v2.8.0 to prefix.dev; bump pixidock pins `>=2.7.1` → `>=2.8.0`; merge
courier→main + CI.

---

## 8. VERSION + CI

- `Cargo.toml` line 3: `version = "2.7.1"` → `"2.8.0"` (emit-affecting feature).
- `Cargo.lock` `pixi-build-retread` (line ~1759): bump to 2.8.0 (`cargo update -p
  pixi-build-retread` or `cargo build` regenerates).
- `recipe/recipe.yaml` line 5: `version: "2.7.1"` → `"2.8.0"`.
- `src/lock.rs:260`: `EMIT_EPOCH: u32 = 4` → `5`; append an epoch-5 note to the doc
  (lock.rs:237–260) describing the dead-URL strip.
- **CI emit-guard** (.github/workflows/ci.yml:89–125): this change touches emit-affecting
  files (`emit_pypi.rs`, `wheel_rewrite.rs`, `courier.rs`). The guard regex
  (ci.yml:110) matches them; the diff to `src/lock.rs` touching `EMIT_EPOCH` (ci.yml:114)
  SATISFIES the guard — **no `[emit-epoch-ok]` token needed** since EMIT_EPOCH is genuinely
  bumped.
- Keep green: `cargo test --lib` + `cargo clippy --all-targets -- -D warnings` + `cargo
  fmt`, all via the pixi toolchain.

---

## 9. IMPLEMENTATION ORDER (for swe-worker-bee; one coherent commit each, green bar每 step)

1. **`LineAction` refactor — PROVABLY emit-neutral** (wheel_rewrite.rs): introduce
   `enum LineAction {Keep,Replace(String),Drop}`; change `rewrite_metadata_text_with`
   (wheel_rewrite.rs:153) + `rewrite_wheel_with` (wheel_rewrite.rs:45) signatures from
   `Fn(&str)->Option<String>` to `Fn(&str)->LineAction`; update the loop
   (Keep→push original / Replace→push `Requires-Dist: {s}` / Drop→`continue` omit). Update
   BOTH mapper bodies 1:1: `None`→`Keep`, `Some(s)`→`Replace(s)` — NEVER `Replace(identical)`
   (Amendment 3). The two mappers are the relax lambda (`relax_pep508`/`rewrite_wheel`,
   wheel_rewrite.rs:35–44 + the direct test call wheel_rewrite.rs:461; Keep/Replace, never
   Drop) and `override_line_map` (emit_pypi.rs:356, updated in step 2). courier call sites
   compile unchanged (they pass the closure). Add unit tests **6 + 7** (the drop-omit test and
   the did_change/sha PARITY test). NO behavior change yet — pure refactor, must be byte- AND
   signal-identical; lands in the same series as the epoch bump.
2. **`drop_url` in `plan()`** (emit_pypi.rs): add `drop_url` to `EmitPlan` (with the
   marker-independent + drop_deps-boundary doc comment from §2.2); populate it in the `None`
   arm (emit_pypi.rs:266); downgrade the WARN to an INFO "stripping dead/orphan direct-URL
   requirement" (no longer a manual-action warning). Make `override_line_map` return
   `LineAction`, checking `drop_url` FIRST (after the `name == "python"` guard at
   emit_pypi.rs:363, before the overrides lookup). Add unit tests **1–5, 8**.
3. **Wire drop set through courier** (courier.rs): pass `emit_plan.drop_url` into
   `override_line_map` at courier.rs:535 (no-cache) AND the cache path courier.rs:353–354 via
   `shadow_cache_stage` — verify BOTH staging branches receive it (a miss drifts cache vs
   no-cache). Confirm the blueprint / meta-wheel path is unaffected (build_meta_wheel
   emit_pypi.rs:426 builds from entry pins, not from `requires_dist`, so it does not need the
   drop set).
4. **EMIT_EPOCH 4→5 + version bumps** (lock.rs:260, Cargo.toml line 3, Cargo.lock,
   recipe/recipe.yaml line 5; append an epoch-5 note to the lock.rs:237–260 doc describing the
   dead/orphan-URL strip). This is the commit the CI guard keys on (no `[emit-epoch-ok]` token
   needed).
5. **G-1 + e2e** (orchestrator): G-0 is DONE (§7). G-1 cold-produce ALL packs (incl
   isaac-pack — emitted bytes change) + drop_deps guard; isaac-pack-latest AND isaac-pack
   cold→replay→install→import; regression suite; then publish + regen ALL locks at EMIT_EPOCH
   5 + pixidock pin bump `>=2.7.1`→`>=2.8.0` + merge.

---

## 10. RESOLVED ITEMS (grizzly audit folded in) + RESIDUAL VERIFY-IN-IMPLEMENTATION

RESOLVED by grizzly (SHIP-WITH-CHANGES):

- **G-0 (DONE)**: grizzly unzipped the real wheels — isaaclab_mimic-1.3.2 is an UNCONDITIONAL
  orphan (no marker, no `Provides-Extra`); 1.0.16/1.2.3 are the marked form. Fix is
  marker-INDEPENDENT (§0.3/§0.4). ✔
- **isaac-pack blast radius (DONE)**: isaac-pack ships the marked 1.2.x robomimic line; the
  predicate strips it too → isaac-pack's emitted bytes ALSO change → its lock regenerates and
  G-1 must confirm it stays green + imports (§4, §3.3, §7 step 7). ✔
- **Predicate soundness (PASS)**: `None`-arm reachable ONLY for URL-deps-absent-from-bundle
  (non-URL `continue`s at emit_pypi.rs:231); active-URL deps hit `Some`-arm →
  `rebuild_requirement` (wheel_rewrite.rs:326, test :577) excises URL to `name==version`;
  RECORD recompute on any METADATA change (wheel_rewrite.rs:98–109). ✔
- **Both courier paths (PASS)**: drop set must flow via `override_line_map` at courier.rs:353–354
  (cache) AND 535–536 (no-cache) — §9 step 3. ✔
- **build_meta_wheel non-hazard (PASS)**: emit_pypi.rs:426 synthesizes from entry pins, not
  `requires_dist` → unaffected. ✔
- **EMIT_EPOCH (PASS)**: in `compute_inputs_hash` (lock.rs:316–317) → 4→5 bump satisfies the
  CI guard (ci.yml:114), no token. ✔
- **strip beats promote/bundle (PASS)**: §5 — A is decisively correct; B/C install an unwanted
  package. ✔
- **drop_deps boundary (DOCUMENTED, Amendment 2)**: strip is bundle-membership-based, does NOT
  consult `config.drop_deps`; G-1 adds the collision guard (§1.1, §3.3 step 4). ✔
- **LineAction byte-parity (Amendment 3)**: 1:1 `None`→`Keep`/`Some(s)`→`Replace(s)`, never
  `Replace(identical)`; did_change/sha parity test #7 (§2.4, §7). ✔

RESIDUAL — verify during implementation (not blocking):

- **Active-extra coverage** (§6): confirm the existing override→pin rewrite already makes
  active-extra git-URL deps acceptable to uv (so we correctly implement ONLY the strip branch
  now). Verified in the §7 e2e (all packs import).
