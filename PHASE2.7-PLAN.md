# PHASE 2.7 PLAN — conda-capable relax-shadow replay drift (pytorch3d)

Branch: `courier`. Base HEAD: `7f2eaf6` (v2.7.0, schema 9, EMIT_EPOCH 4).
Scope: REPLAY-LOGIC-ONLY change in `materialize_from_lock` (`src/handler/mod.rs`).
NO lock-field change, NO schema bump (stays 9), NO EMIT_EPOCH bump (stays 4).
Version bump 2.7.0 → 2.7.1 (replay bugfix).

Every claim below was re-verified by henry-hudson against current code at HEAD
`7f2eaf6`. Line numbers are the CURRENT ones, not the handoff's stale ones.

---

## 0. THE BUG (re-verified, exact)

One wheel drifts between the COLD-produced and REPLAY-produced lock for
`examples/gigastrap/isaac-pack` (the FRESH schema-9 lock produced under v2.7.0 by
`scripts/replay-e2e.sh`, NOT the stale committed schema-4 lock):

- **COLD** `pytorch3d`: `origin="built"`, `must_ship=false`,
  `upstream_url="https://github.com/MiroPsota/torch_packages_builder/releases/download/.../pytorch3d-0.7.8+5043d15pt2.7.0cu128-cp311-cp311-linux_x86_64.whl"`,
  `filename="pytorch3d-0.7.8+5043d15pt2.7.0cu128-999retread-cp311-cp311-linux_x86_64.whl"`,
  NO `url`.
- **REPLAY** `pytorch3d`: `origin="index"`, `url=<the SAME github url>`, NO
  `upstream_url`, SAME `999retread` filename, same `requires_dist`.

`pytorch3d` is the SOLE drifting wheel. `genesis` + `isaac6` packs replayed
byte-identical in the same run and MUST NOT regress.

---

## 1. ROOT CAUSE (every link verified against current code)

### 1.1 `pytorch3d` is a PRIMARY `[retread-wheels]` entry with a redirecting custom index
`examples/gigastrap/isaac-pack/pixi.toml:50`:
```toml
pytorch3d = { version = "==0.7.8+5043d15pt2.7.0cu128", index = "https://miropsota.github.io/torch_packages_builder" }
```
The index host (`miropsota.github.io`) redirects to a `github.com/.../releases/download/...` URL. VERIFIED.

### 1.2 `pytorch3d` IS (runtime-)conda_capable; isaacsim/kernel/core are NOT
`conda_capable` is built in `build_one` (cold path) at `src/handler/mod.rs:5032-5039`:
it merges (a) probe decisions with `matching_candidates > 0`, (b) `config.name_map`
keys, (c) `load_pypi_to_conda_map()` (live parselmouth download from
`PARSELMOUTH_MAPPING_URL`, `mod.rs:338`). Membership is by `canonical_conda_name`
(`src/relax.rs:141-163`, PEP-503 normalization) and a plain `HashSet::contains`.
- `pytorch3d`: NOT in the static `FALLBACK_PYPI_TO_CONDA` (`mod.rs:348-370`); its
  conda_capable status comes from live parselmouth. The empirical COLD-vs-REPLAY
  drift logs PROVE it is conda_capable in practice (only conda_capable relax-shadows
  drift — see 1.6). VERIFIED behaviorally.
- `isaacsim`/`isaacsim-kernel`/`isaacsim-core`: NOT in fallback; closed-source
  NVIDIA PyPI-only → no conda-forge entry → not in parselmouth; conda probe finds 0
  candidates → filtered out at `mod.rs:5035`. NOT conda_capable. VERIFIED.

### 1.3 COLD path → courier LOCAL-PATH branch → `Origin::Built`
A primary config entry is fetched + relaxed by `materialize_and_rewrite`
(`src/handler/mod.rs`):
- the upstream index URL is captured BEFORE localization at `mod.rs:3527-3530`
  (`upstream_url = Some(resolved.url.clone())`), so the lock gets
  `upstream_url=<github URL>`. VERIFIED.
- the wheel is fetched (`fetch_wheel_cached`, `mod.rs:3445`), relaxed to a
  `*.relaxed.whl` (`mod.rs:3647`, rewrite at `mod.rs:3660-3667`), and the returned
  `ResolvedWheel.url` becomes the local `file://` path (`mod.rs:3709-3724`). So
  cold's `EmitWheel` carries `local_path=Some(.relaxed.whl)` +
  `upstream_url=Some(github)`. VERIFIED.
- `courier::stage` takes the LOCAL-PATH sub-branch (`src/courier.rs:570`,
  `if let Some(src) = w.local_path.as_ref()`). The relax rewrite returns
  `did_change=true` → `ShadowSrc::Rewritten` (cache path, `courier.rs:602`) or
  `ShadowSrc::Raw` (no-cache path, `courier.rs:629`).
- **CRITICAL: the local-path branch has NO `conda_capable` gate.** VERIFIED
  (henry read all of `courier.rs:570-631`; the only `conda_capable` check in the
  whole index-wheel arm is at `courier.rs:644`, inside the REMOTE-ONLY sub-branch).
- Both `ShadowSrc::Rewritten` (`courier.rs:702-743`) and `ShadowSrc::Raw`
  (`courier.rs:745-787`) emit: `origin=Origin::Built`, `url=None`,
  `upstream_url = w.upstream_url.or(w.remote_url)` (→ github),
  `filename = insert_build_tag(std_name, "999retread")`. VERIFIED.

### 1.4 REPLAY path → bare `Origin::Built` arm (Class-2) → remote-only branch
On replay, the lock entry (`origin=built`, `must_ship=false`, `sdist_source=None`,
`git_source=None`) routes to the bare `Origin::Built =>` arm = **Class-2**, at
`src/handler/mod.rs:4678-4743` (dispatch order: `Index`@4257, `Built if must_ship`@4298,
`Built if !must_ship && sdist_source.is_some()` = Class-2b @4607, bare `Built` =
Class-2 @4678). VERIFIED.

The log line `mod.rs:4701-4705` confirms:
`"courier replay: re-fetching relax-changed shadow from upstream (class 2)"`.

Class-2 builds an `EmitWheel` with (`mod.rs:4732-4742`):
- `local_path: None` (`mod.rs:4736`) — DELIBERATE (see §2),
- `remote_url: Some(remote_url)` where `remote_url` is parsed from `lw.upstream_url`
  (github) (`mod.rs:4683-4700, 4738`),
- `upstream_url: None` (the EmitWheel field, `mod.rs:4739`),
- `wheel_filename: lw.filename.clone()` (the already-`999retread` name, `mod.rs:4737`).

### 1.5 courier REMOTE-ONLY branch + the conda gate → `Origin::Index` (DRIFT)
Because `local_path=None`, `courier::stage` takes the REMOTE-ONLY sub-branch
(`src/courier.rs:632-674`). Its shadow gate (`courier.rs:644`, VERBATIM):
```rust
if any_change && !conda_cap_owned.contains(&w.pypi_name) {
```
where `any_change` (`courier.rs:640-643`) is true iff any `requires_dist` line would
relax-change. For `pytorch3d`: `any_change=true` BUT `conda_cap_owned.contains("pytorch3d")=true`
→ gate is **FALSE** → `ShadowSrc::None` (`courier.rs:673`) → the `ShadowSrc::None`
arm (`courier.rs:678-701`) emits `origin=Origin::Index`, `url=Some(remote_url)` (github),
`upstream_url=None`. **DRIFT.** The `999retread` filename in the replay lock is
`std_name` echoed (the lock's stored `lw.filename` passed through
`standard_wheel_filename`, no change since it has no retread infix). VERIFIED.

### 1.6 WHY isaacsim/kernel/core (same pack, also Class-2 relax-shadows) DON'T drift
They are NOT conda_capable (§1.2). On the remote-only branch the gate
`any_change && !conda_capable` is TRUE → force-download to `.dl-courier-*`
(`courier.rs:652-664`) → `ShadowSrc::Raw` (`courier.rs:671`) → re-shadowed →
`origin=Built` both sides. **ONLY conda_capable relax-shadows drift.** VERIFIED.

### 1.7 Other conda_capable relax-shadows across packs (coverage)
henry surveyed every `examples/*/pixi.toml`. The drift requires: a PRIMARY
`[retread-wheels]` entry with a custom `index=` (→ gets `local_path` cold) AND
conda_capable membership. Across ALL examples, `pytorch3d` in
`examples/gigastrap/isaac-pack` is the **sole** entry meeting both. `isaacsim`
variants have a custom index but are NOT conda_capable. git-source entries
(`isaaclab*`, `genesis-world`, `newton`) take the `must_ship` branch entirely.
The fix is GENERAL (any conda_capable relax-shadow), so it covers any future such
entry, but `pytorch3d` is the only live one. The user's pixidock isaac-pack has NO
`pytorch3d` (grep=0) → pixidock unaffected. VERIFIED.

---

## 2. WHY Class-2 currently sets `local_path=None` (the guarded concern)

The comment at `src/handler/mod.rs:4706-4731` (VERBATIM-read) states the invariant:

> An `Origin::Index` wheel, or a relax-changed index shadow (`Origin::Built && !must_ship`),
> is NEVER the target of a direct-URL Requires-Dist line. `plan()` in `emit_pypi.rs`
> reads `local_path` ONLY inside the direct-URL ship-set insert (`target.local_path.is_some()`
> branch). Because index shadows are never direct-URL targets, their `local_path` is never
> consulted by `plan()`, so setting it to None is safe... A `debug_assert` in `plan()`
> enforces this: when a wheel enters the ship set via the `local_path` gate, its
> `remote_url` must be None.

**What it protects:** it asserts that giving a Class-2 shadow `local_path=None`
+ `remote_url=Some` is HARMLESS, because `plan()` never reads `local_path` for an
index shadow (index shadows are not direct-URL requirement targets). It does NOT say
`local_path=Some` would be *wrong* — it says `None` is *sufficient*. The
`debug_assert` it references fires only when a wheel enters the ship set via the
`local_path` gate WITH a non-None `remote_url` simultaneously.

**Implication for the fix:** we will set `local_path=Some(downloaded)` AND
`remote_url=None` (mirroring cold's local-path EmitWheel, which has
`local_path=Some` + `remote_url=None`; cold carries upstream in `upstream_url`, not
`remote_url`). This does NOT trip the `debug_assert` (it requires `local_path=Some`
AND `remote_url=Some` together). It does NOT make the shadow a direct-URL
requirement target (plan()'s URL-target set is driven by `requires_dist` content,
not by which EmitWheel fields are set — unchanged). So the invariant is preserved.

**Cross-machine portability:** the downloaded `local_path` is a TRANSIENT build-time
path under `<source_dir>/wheels/` consumed immediately by `courier::stage` within
the SAME process; it is never serialized to the lock (the lock stores `filename`,
`url`, `upstream_url` — never `local_path`). Cold's `local_path` (the `.relaxed.whl`)
is identically transient and identically never serialized. So no `file://`
portability issue is introduced. This is exactly how Class-2b (gym) and Class-1
(git) already operate: both set `local_path=Some(built)` on replay and ship fine.

---

## 3. THE FIX (general, root-cause, byte-identical-by-construction)

Make the Class-2 replay arm reproduce COLD's classification: DOWNLOAD the upstream
shadow to a `local_path=Some(...)` and route it through the courier LOCAL-PATH branch
(no conda gate), exactly as cold does — mirroring the proven Class-2b (gym) and
Class-1 (git) precedents.

### 3.1 Code change — `src/handler/mod.rs`, the Class-2 arm (currently 4678-4743)

Replace the body. Pseudocode (final wording during implementation):

```rust
Origin::Built => {
    // Class 2: relax-changed INDEX shadow (must_ship=false, no sdist/git).
    // The original upstream index URL is in lw.upstream_url (schema 6+).
    let remote_url_opt = lw.upstream_url.as_deref()
        .and_then(|u| url::Url::parse(u).ok());
    let remote_url = match remote_url_opt {
        Some(u) => u,
        None => {
            tracing::warn!(wheel = %lw.name,
                "courier replay: relax-changed Built wheel has no upstream_url \
                 (schema-5 lock); falling through to full resolve");
            return Ok(None);
        }
    };
    tracing::info!(wheel = %lw.name, url = %remote_url,
        "courier replay: re-fetching relax-changed shadow from upstream (class 2)");

    // FIX (Phase 2.7): DOWNLOAD the upstream wheel to a local path and route
    // through courier's LOCAL-PATH branch (ShadowSrc::Rewritten/Raw -> Built),
    // mirroring COLD's materialize_and_rewrite path. The remote-only branch's
    // `!conda_capable` shadow gate would otherwise (mis)classify a conda_capable
    // shadow as Origin::Index on replay, drifting the lock vs cold. Lock stores
    // no wheel sha256, so pass None; fetch_wheel_cached falls back to fetch_wheel
    // (dest_dir.join(wheel_filename_from_url(url))) -> the pristine 5-field
    // upstream filename, identical to what cold fetched pre-relax.
    let fetched = crate::wheel::fetch_wheel_cached(
        &remote_url, None, &download_dir, cache_dir,
    ).await.with_context(|| format!(
        "courier replay Class-2: re-fetching shadow {} from {}", lw.name, remote_url))?;

    // DETERMINISM GUARD (Phase 2.7, mirrors the Phase-2/2.6 source-build guards):
    // assert the re-fetched artifact reproduces the recorded shadow name. The
    // shadow name courier WILL emit is insert_build_tag(standard_wheel_filename(
    // <fetched basename>), "999retread"); it must equal lw.filename. If upstream
    // served a repackaged/differently-named artifact under the same URL, DIVERGE
    // -> fall through to cold re-resolve instead of silently emitting a drifted
    // lock entry. This pins the assumption: replay reproduces the RECORDED
    // artifact name; a repackaged upstream is invalidated only via cold resolve.
    let fetched_base = fetched.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let predicted = crate::emit_pypi::insert_build_tag(
        &crate::emit_pypi::standard_wheel_filename(fetched_base), "999retread")?;
    if predicted != lw.filename {
        tracing::warn!(wheel = %lw.name, predicted = %predicted, recorded = %lw.filename,
            "courier replay Class-2: re-fetched artifact name diverges from recorded \
             shadow filename (upstream repackaged?); falling through to cold resolve");
        return Ok(None);
    }

    crate::emit_pypi::EmitWheel {
        pypi_name: lw.name.clone(),
        version: lw.version.clone(),
        requires_dist: lw.requires_dist.clone(),
        local_path: Some(fetched),
        // lw.filename is the already-999retread shadow name. standard_wheel_filename
        // strips no retread infix from it; insert_build_tag sees 6 fields and
        // REPLACES the 999retread tag with 999retread (idempotent no-op) -> the
        // identical shadow_name as cold. (Equivalently, the on-disk 5-field
        // upstream name would yield the same shadow_name via the 5-field insert.)
        wheel_filename: lw.filename.clone(),
        remote_url: None,        // cold's local-path EmitWheel has remote_url=None
        upstream_url: Some(remote_url.to_string()), // -> courier writes upstream_url=github, url=None
        git_source: None,
        sdist_source: None,
    }
}
```

Notes:
- `download_dir` (= `source_dir.join("wheels")`) and `cache_dir` are already in scope
  in `materialize_from_lock` (`mod.rs:4156`, `mod.rs:4146`). VERIFIED.
- `fetch_wheel_cached(url, None, dest, cache_root)` with `expected_sha256=None` →
  `fetch_wheel` (`wheel.rs:213-214`) → lands at
  `dest_dir.join(wheel_filename_from_url(url))` = the pristine upstream 5-field name.
  This is byte-for-byte the same fetch cold performs at `mod.rs:3445`. VERIFIED.
- `EmitWheel.upstream_url` is `Option<String>` (matches Class-1 usage); courier's
  `Rewritten`/`Raw` arms compute `w.upstream_url.or(w.remote_url)` → github →
  written to `LockWheel.upstream_url`, with `url=None`. So the lock gets
  `upstream_url=github`, `url=None` = cold. VERIFIED at `courier.rs:723-727, 766-770`.
- The old comment block (`mod.rs:4706-4731`) is REPLACED by the new FIX comment;
  its invariant is preserved (see §2) but no longer the rationale for `local_path=None`.

### 3.2 UNIFORM vs gated — make it UNIFORM (apply to ALL Class-2 shadows)

We route EVERY Class-2 shadow (conda_capable AND non-conda_capable) through the
download→local-path branch. Justification (the user's preference for a uniform,
general fix):
- For **conda_capable** shadows (pytorch3d): fixes the drift (Index→Built). REQUIRED.
- For **non-conda_capable** shadows (isaacsim/kernel/core): COLD already produced them
  as `Origin::Built` via the local-path branch (they too are primary-or-BFS index
  wheels localized by `materialize_and_rewrite`). The OLD replay produced them as
  `Origin::Built` via the remote-only force-download → `ShadowSrc::Raw` arm
  (`courier.rs:671 → 745-787`). The NEW replay produces them as `Origin::Built` via
  the local-path → `ShadowSrc::Raw`/`Rewritten` arm. **Both old and new replay emit
  identical fields** (`origin=Built`, `url=None`, `upstream_url=github`,
  `filename=999retread`, full `requires_dist`) because the `Raw` and `Rewritten` arms
  share the SAME LockWheel construction (`courier.rs:728-743` ≡ `760-787`). The ONLY
  behavioral difference is the download MECHANISM: remote-only uses
  `reqwest::get → .dl-courier-*` then `rewrite_wheel_with`; local-path uses
  `fetch_wheel_cached → wheels/<name>` then cache-stage-or-`rewrite_wheel_with`. Both
  end at the same `rewrite_wheel_with` (Step-0 deterministic, pinned-1980-ts repack,
  per accumulated finding). So non-conda_capable shadows ALSO converge to Built and
  do NOT regress. VERIFIED by field-equivalence of the two arms.
- Simpler + general (no per-wheel conda_capable branch in replay; the replay code
  doesn't even need to read `lock.conda_capable` here).

This is preferred over §5's alternative (it is byte-identical by *construction* —
same path cold took — not by *argument about a gate*).

---

## 4. BYTE-IDENTITY CONVERGENCE (field-by-field, COLD vs NEW REPLAY)

Both paths end in courier's LOCAL-PATH branch → `ShadowSrc::Rewritten` (cache on) or
`ShadowSrc::Raw` (cache off). Those two arms (`courier.rs:702-743` / `745-787`) write
IDENTICAL `LockWheel` fields. Field-by-field for `pytorch3d`:

| Field | COLD | NEW REPLAY | Converge? |
|---|---|---|---|
| `origin` | `Built` (Rewritten/Raw arm) | `Built` (same arm) | YES |
| `filename` | `insert_build_tag(std_name,"999retread")` where `std_name`=`standard_wheel_filename("pytorch3d-...-cp311-cp311-linux_x86_64.relaxed.whl")` → 5-field clean → 999retread inserted | `insert_build_tag(std_name,"999retread")` where `std_name`=`standard_wheel_filename(lw.filename)` = the already-999retread name → 6-field → 999retread REPLACED (idempotent) | YES (both = `pytorch3d-0.7.8+5043d15pt2.7.0cu128-999retread-cp311-cp311-linux_x86_64.whl`) |
| `url` | `None` (Rewritten/Raw arm always sets `url:None`) | `None` (same) | YES |
| `upstream_url` | `w.upstream_url.or(w.remote_url)` = github | `w.upstream_url`(=`Some(remote_url)`) = github | YES |
| `requires_dist` | `w.requires_dist.clone()` (real, from fetched wheel metadata) | `lw.requires_dist.clone()` (the stored real `requires_dist`, written full per #4 parity) | YES (stored == cold) |
| `must_ship` | `w.must_ship()` = false (index shadow; basename has no `.injected`) | `w.must_ship()` = false (fetched pristine wheel basename has no `.injected`) | YES* |
| `sha256` | `None` | `None` | YES |
| `git_source` | `None` | `None` | YES |
| `sdist_source` | `w.sdist_source` = `None` | `w.sdist_source` = `None` | YES |

\* `must_ship()`: GRIZZLY-CONFIRMED in code (`emit_pypi.rs:106-116`): it keys ONLY on
whether the wheel basename contains `.injected` — NOT on `local_path`/`git_source`/
`sdist_source` presence. The Class-2 replay EmitWheel's `local_path` is the PRISTINE
fetched wheel (no `.injected` infix), so `must_ship()` returns false, matching cold's
index shadow. (This also explains why Class-1 git is `must_ship=true` — its localized
wheel carries `.injected` — while Class-2b sdist and our Class-2 are `must_ship=false`:
they ship pristine basenames.) VERIFIED, no longer an open item.

**The `filename` line is the load-bearing detail** and it converges because
`insert_build_tag`'s 6-field branch (`emit_pypi.rs:412-415`) REPLACES an existing
build tag (idempotent for `999retread`→`999retread`), while the 5-field branch
(`emit_pypi.rs:408-411`) inserts it. Cold takes the 5-field branch (clean
`.relaxed`-stripped name); replay takes the 6-field branch (already-tagged
`lw.filename`). Both yield the identical string. VERIFIED by henry.

---

## 5. SELF-DRIFT (2nd replay must also be byte-identical)

After the fix, the cold lock and the 1st replay lock are byte-identical. A 2nd
replay reads the SAME lock entry (`origin=built, must_ship=false, sdist_source=None,
git_source=None, upstream_url=github`) → SAME Class-2 arm → re-downloads from the
SAME `upstream_url` → SAME local-path branch → SAME `Rewritten`/`Raw` → SAME
`origin=Built, upstream_url=github, url=None, 999retread filename`. No field is
re-derived differently on the Nth replay (the arm reads only `lw.*`, which are stable
across replays). NO self-drift. The `filename` is doubly stable: even though the
EmitWheel carries the already-999retread `lw.filename`, the 6-field idempotent-replace
keeps it fixed. VERIFIED by construction (cf. Class-2b's identical self-drift
guarantee — the audit note at HANDOFF line ~396).

---

## 6. NO REGRESSION on non-conda_capable Class-2 shadows (filename source TRACED)

This section proves byte-identity for the NON-conda_capable Class-2 shadows
(isaacsim/kernel/core, aiohttp/scipy/matplotlib/...) that currently replay fine via
the remote-only branch, AND that the conda_capable case (pytorch3d) converges too.
The load-bearing question the grizzly flagged: where does `std_name` (the base from
which `insert_build_tag` derives the `999retread` filename) come from inside the
courier LOCAL-PATH branch — the on-disk `local_path` basename, or `w.wheel_filename`?

### 6.1 The `std_name` SOURCE (henry, VERBATIM, courier.rs)
`std_name` is computed ONCE per wheel at the top of the `for w in emit_wheels` loop,
BEFORE the `must_ship()` branch split (`if w.must_ship()` at `courier.rs:488`):
```
   486	        let std_name = standard_wheel_filename(&w.wheel_filename);
```
`standard_wheel_filename` (`emit_pypi.rs:124-129`, VERBATIM):
```rust
pub fn standard_wheel_filename(cached: &str) -> String {
    cached
        .replace(".injected.", ".")
        .replace(".autodata.", ".")
        .replace(".relaxed.", ".")
}
```
**`std_name` derives SOLELY from `w.wheel_filename` (the EmitWheel field) — NEVER from
the on-disk `local_path` basename.** Inside the LOCAL-PATH branch (`courier.rs:570-631`)
the bound `src = w.local_path` is used ONLY as the bytes to read/rewrite; staging
destinations are built from `std_name` (e.g. `courier.rs:589`
`.probe-courier-{std_name}`, `courier.rs:608` `.tmp-courier-{std_name}`). Both shadow
arms call `insert_build_tag(&std_name, "999retread")` against this SAME `std_name`:
`ShadowSrc::Rewritten` at `courier.rs:707`, `ShadowSrc::Raw` at `courier.rs:749`.
VERIFIED by henry.

**Consequence:** the filename is a pure function of `EmitWheel.wheel_filename`. The
downloaded artifact's on-disk name is IRRELEVANT to the emitted shadow filename. Our
Class-2 replay arm sets `wheel_filename = lw.filename.clone()` (the already-`999retread`
recorded name) for BOTH conda_capable and non-conda_capable shadows, so every replay
produces `insert_build_tag(standard_wheel_filename(lw.filename), "999retread")`.

### 6.2 Per-class convergence proof (filename specifically)

**(a) pytorch3d (conda_capable), cold vs NEW replay:**
- COLD: `w.wheel_filename` = the localized `.relaxed.whl` basename
  (`pytorch3d-...-cp311-cp311-linux_x86_64.relaxed.whl`) → `standard_wheel_filename`
  strips `.relaxed.` → 5-field clean name → `insert_build_tag` 5-field branch INSERTS
  `999retread` → `pytorch3d-0.7.8+5043d15pt2.7.0cu128-999retread-cp311-cp311-linux_x86_64.whl`.
- NEW REPLAY: `w.wheel_filename` = `lw.filename` (the already-`999retread` name) →
  `standard_wheel_filename` strips nothing (no retread infix) → 6-field name →
  `insert_build_tag` 6-field branch REPLACES `999retread`→`999retread` (idempotent) →
  the IDENTICAL string. CONVERGES.

**(b) isaacsim/kernel/core (NON-conda_capable), OLD replay vs NEW replay:**
- OLD replay (remote-only branch): `w.wheel_filename` = `lw.filename` (already-999retread)
  → remote-only force-download → `ShadowSrc::Raw` → `insert_build_tag(std_name=…,"999retread")`
  with `std_name = standard_wheel_filename(lw.filename)` (6-field idempotent) → 999retread name.
- NEW replay (local-path branch): `w.wheel_filename` = `lw.filename` (UNCHANGED — we set
  the same field) → `std_name` computed at `courier.rs:486` is the IDENTICAL value
  (still `standard_wheel_filename(lw.filename)`, since `std_name` ignores the downloaded
  basename) → `ShadowSrc::Raw`/`Rewritten` → `insert_build_tag` 6-field idempotent →
  the IDENTICAL 999retread name. CONVERGES.
- The ONLY change for these wheels is the download MECHANISM (`fetch_wheel_cached →
  wheels/<pristine name>` vs `reqwest → .dl-courier-<std_name>`); the bytes fed to
  `rewrite_wheel_with` are the same upstream wheel, the rewrite is deterministic
  (pinned-1980-ts repack), and ALL emitted LockWheel fields (`origin=Built`, `url=None`,
  `upstream_url`, `filename`, `requires_dist`, `must_ship=false`) are produced by the
  SAME `Raw`/`Rewritten` arms. So the lock entry is byte-identical to the old replay
  AND to cold. NO REGRESSION.

Note the determinism guard (§3.1) recomputes `insert_build_tag(standard_wheel_filename(
<fetched basename>), "999retread")` and compares to `lw.filename`. For (a)/(b) the
fetched basename is the pristine 5-field upstream name → 5-field insert → `999retread`
name → equals `lw.filename` → guard passes. The guard's predicted name uses the FETCHED
basename (5-field path), while courier's actual emit uses `lw.filename` (6-field path);
both routes yield the identical `999retread` string (§4 R2), so the guard is a sound
oracle for the emitted name.

### 6.3 Empirical guard
The e2e (§9) asserts `git diff --exit-code` clean on ALL three packs (genesis, isaac6,
gigastrap-isaac), catching any regression. genesis + isaac6 (isaac6 shadows are
non-conda_capable isaacsim-* which already went Built) MUST be re-confirmed
byte-identical (they were this run; the change only alters the Class-2 arm, which
isaac6/genesis exercise via the same `Raw`/`Rewritten` convergence as 6.2(b)).

---

## 7. ALTERNATIVE CONSIDERED: fix the `!conda_capable` gate (REJECTED)

Option B: on the REPLAY path, drop/adjust the `!conda_capable` gate in courier's
remote-only branch (`courier.rs:644`) so a conda_capable shadow is force-downloaded
and re-shadowed (Built) like the non-conda_capable ones.

Rejected, decisively:
- The gate at `courier.rs:644` is on the PRODUCE path too (courier::stage is shared
  cold+replay). The `!conda_capable` skip is a deliberate cold-path optimization
  (AUDIT B2 comment, `courier.rs:635-639`): if conda satisfies a dep, recording it
  `Origin::Index` is harmless (conda wins) and avoids a needless download+shadow.
  Removing/relaxing it would change COLD behavior for remote-only conda_capable
  wheels (small auto-bundled deps) → potential cold lock churn + extra downloads →
  blast radius beyond replay. The handoff mandates replay-logic-ONLY changes.
- Threading a "replay vs produce" flag into `courier::stage` to gate-differently is a
  band-aid (special-casing inside the shared stage), exactly what the loop forbids.
- The download→local-path fix mirrors cold's ACTUAL path → byte-identity by
  construction (same proof shape as the gym Class-2b fix and the git Class-1 fix),
  and is confined to `materialize_from_lock` (replay-only). Strictly cleaner and
  more general. **DECISION: download→local-path (Option A).**

---

## 8. SCHEMA / EMIT_EPOCH / VERSION / CI-GUARD

- **No lock-field change**: we set existing `EmitWheel` fields differently on replay;
  the serialized `LockWheel` shape is unchanged. → **SCHEMA stays 9**
  (`src/lock.rs:235`).
- **inputs_hash unaffected**: `compute_inputs_hash` (`src/lock.rs:288-325`) feeds only
  `entry_specs, index_urls, relax, python, emit_epoch, pin_version, config_fingerprint`
  — NONE of `materialize_from_lock`'s replay routing. VERIFIED. → **EMIT_EPOCH stays 4**
  (`src/lock.rs:260`).
- **CI emit-epoch-guard** (`.github/workflows/ci.yml:103-126`): the watched-file regex
  is `^src/(relax|wheel_rewrite|wheel_inject|wheel_inject_data|emit_pypi|recipe|courier|lock)\.rs$|^src/handler/(auto_bundle|cascade)\.rs$`.
  Our change touches ONLY `src/handler/mod.rs` (the Class-2 arm) — `mod.rs` is NOT in
  the regex → **the guard does NOT fire**. No EMIT_EPOCH bump and no `[emit-epoch-ok]`
  token required. (If implementation ALSO touches `src/courier.rs` — it should NOT;
  the fix is wholly in `mod.rs` — the guard would fire and we'd add `[emit-epoch-ok]`.
  Plan: keep the change in `mod.rs` only.) VERIFIED.
- **Version**: bump `Cargo.toml:3` and `recipe/recipe.yaml:5` from `2.7.0` → `2.7.1`
  (replay bugfix). VERIFIED both at 2.7.0. ALSO update `Cargo.lock` (the
  `pixi-build-retread` package entry) so it stays in sync — run the build once or
  `cargo update -p pixi-build-retread --precise 2.7.1` to regenerate it.

---

## 9. TESTS

Amendment 2 (grizzly): the PURE field-mapping test is the REQUIRED primary. The
unreachable-url `!matches!(Ok(None))` test is DROPPED (false-green: an `Err` also
passes, and it proves nothing about `origin=Built` vs `Index`). Add a localhost-fixture
parity test as the real byte-identity oracle if feasible.

### 9.1 REQUIRED primary — `class2_emit_wheel_field_mapping` (pure sync, no network)
Mirror `class2b_emit_wheel_field_mapping` (`mod.rs:5957-6036`). Construct the EmitWheel
EXACTLY as the new Class-2 arm produces it (with a dummy `local_path` such as a pristine
upstream basename), and assert the full field contract:
- `local_path.is_some()` (routes through courier's local-path branch, no conda gate),
- `remote_url.is_none()` (cold's local-path EmitWheel has `remote_url=None`; also keeps
  the plan() `debug_assert` from firing),
- `upstream_url == Some("https://github.com/.../pytorch3d-...whl".to_string())`,
- `git_source.is_none()`, `sdist_source.is_none()`,
- `wheel_filename == lw.filename`, `version == lw.version`,
- `requires_dist == lw.requires_dist`.
This proves the Built + upstream + no-url + no-double-remote contract deterministically
with no network — the robust style behind Class-2b's contract test.

### 9.2 REQUIRED parity — `class2_replay_cold_byte_identity` (localhost fixture)
Add the byte-identity oracle the shipped Class-2b lacked, mirroring the existing
localhost-fixture parity tests (the Phase-1 localhost parity test + the git/Class-2b
parity tests). Serve a SMALL conda_capable wheel over a localhost HTTP server (reuse the
existing test fixture-server helper used by the Phase-1/git parity tests; henry/the-bee
should locate it — it's the same harness those tests already use). Then:
1. Drive the COLD path: run `courier::stage` on an `EmitWheel` with `local_path=Some(<the
   fetched+relaxed wheel>)`, `upstream_url=Some(localhost url)`, `conda_capable={name}`
   → capture the produced `LockWheel`.
2. Drive the NEW Class-2 REPLAY path: build a `RetreadLock` whose single `LockWheel` is
   that cold entry (`origin=Built`, `must_ship=false`, `upstream_url=Some(localhost)`,
   recorded `999retread` filename, `sdist_source=None`, `git_source=None`), call
   `materialize_from_lock` → it re-fetches from the localhost url, routes local-path,
   and produces a `LockWheel`.
3. ASSERT the two `LockWheel`s are FIELD-FOR-FIELD equal (`origin`, `filename`, `url`,
   `upstream_url`, `requires_dist`, `must_ship`, `sha256`, `git_source`, `sdist_source`).
This is the real byte-identity oracle (proves cold==replay for a conda_capable shadow),
not a routing smoke test. If the localhost-fixture wiring proves heavier than the
iteration budget allows, 9.1 + 9.3 (e2e) still cover the contract + the empirical seal;
prefer to land 9.2 — it is the test that would have CAUGHT this Phase-2.7 drift.

### 9.3 Live round-trip (ignored): `class2_live_round_trip`
`#[tokio::test] #[ignore]` mirroring `class2b_live_round_trip` (`mod.rs:6048-6134`):
real reachable upstream index wheel (a small conda_capable one to keep it cheap, e.g.
a tiny pure-python wheel from PyPI marked conda_capable), assert `Ok(Some)` and that the
produced lock entry is `origin=Built`, `url=None`, `upstream_url=Some`, `filename`
carries `999retread`.

### 9.4 e2e (the empirical seal): `scripts/replay-e2e.sh`
Already exists; runs genesis + isaac6 + gigastrap-isaac, cold-produces schema-9 locks
under the locally-built backend, nukes ALL caches incl `wheels/` (keeps
`.pixi/config.toml`), lukewarm-replays, asserts:
- `build_v1: replayed from lock` present; zero derivation (auto-bundled / resolvo
  solve / probe-trace = 0); `wheels/` repopulated from EMPTY;
- `git diff --exit-code` CLEAN on ALL three lock files — specifically
  `examples/gigastrap/isaac-pack/retread-isaac-pack.lock.json` (the acceptance target,
  pytorch3d no longer drifts) AND `genesis` + `isaac6` (no regression);
- env imports.
Rebuild via `bash scripts/rebuild-local.sh` first (local backend = 2.7.1).

---

## 10. COVERAGE / NON-TOUCH GUARANTEES

- Fix is GENERAL: any conda_capable relax-shadow index wheel now replays as Built
  (the only live one is `pytorch3d`; future ones covered automatically).
- Does NOT touch the working **Class-2b** (gym/sdist) arm (`mod.rs:4607-4676`): that
  arm is matched FIRST by the `sdist_source.is_some()` guard; our change is in the
  bare `Origin::Built` fall-through after it.
- Does NOT touch **Class-1** (git, `mod.rs:4298+`) or **Class-3** (`mod.rs:4570-4588`)
  or **Class-4** (Index, `mod.rs:4257`).
- Does NOT touch `src/courier.rs` (the shared cold+replay stage) — so cold behavior is
  entirely unchanged and the CI emit-guard stays silent.
- Does NOT change any lock FIELD, schema, or inputs_hash → all existing schema-9 locks
  (genesis/newton/isaac6/pixidock) still replay under 2.7.1 unchanged.

---

## 11. IMPLEMENTATION ORDER (for swe-worker-bee)

1. Rewrite the Class-2 arm body (`src/handler/mod.rs:4678-4743`) per §3.1: download via
   `fetch_wheel_cached(&remote_url, None, &download_dir, cache_dir)`; add the §3.1
   determinism guard (compare `insert_build_tag(standard_wheel_filename(<fetched
   basename>), "999retread")` to `lw.filename`, `return Ok(None)` on divergence); set
   `local_path=Some(fetched)`, `remote_url=None`, `upstream_url=Some(remote_url.to_string())`,
   `wheel_filename=lw.filename.clone()`, `sdist_source=None`, `git_source=None`. Replace
   the old `local_path=None` comment block (`mod.rs:4706-4731`) with the new FIX comment
   (preserve the §2 invariant in prose). Keep the schema-5 `upstream_url=None → Ok(None)`
   fall-through. `standard_wheel_filename`/`insert_build_tag` are already `pub` in
   `emit_pypi.rs` (used by the guard); confirm they're importable from `mod.rs`.
   Change is `mod.rs`-ONLY (do NOT touch `src/courier.rs`).
2. Add §9.1 `class2_emit_wheel_field_mapping` (pure, REQUIRED) + §9.2
   `class2_replay_cold_byte_identity` (localhost-fixture parity, REQUIRED — reuse the
   existing fixture-server helper from the Phase-1/git/Class-2b parity tests) + §9.3
   `class2_live_round_trip` (#[ignore]). DO NOT add the dropped unreachable-url
   `!matches!(Ok(None))` test (false-green).
3. Bump `Cargo.toml:3` + `recipe/recipe.yaml:5` 2.7.0 → 2.7.1; sync `Cargo.lock` (§8).
4. Green bar: `PATH=".../pixi/.pixi/envs/default/bin:$PATH" cargo test --lib` +
   `cargo clippy --all-targets -- -D warnings` + `cargo fmt`. Commit on `courier`.
5. Hand to the-grizzly for audit (esp. §4 filename convergence + §6 traced no-regression +
   §5 self-drift + the §9.2 parity oracle), fix findings, then orchestrator runs
   `scripts/replay-e2e.sh`.

`must_ship()` is GRIZZLY-CONFIRMED (keys only on `.injected` basename, `emit_pypi.rs:106-116`)
→ our pristine-basename local_path yields `must_ship=false`. No longer a pre-implementation
blocker (was §11 step 1; removed).

---

## 12. RISKS / OPEN ITEMS (pre-empting the grizzly)

- **R1 (must_ship)** — RESOLVED by the grizzly in code (`emit_pypi.rs:106-116`):
  `must_ship()` keys ONLY on a `.injected` basename. Our replay EmitWheel's `local_path`
  is the pristine fetched wheel (no `.injected`) → `must_ship=false`, matching cold.
  No longer an open item.
- **R2 (filename 5-vs-6-field)** — resolved: idempotent 6-field replace == 5-field
  insert for `999retread` (grizzly confirmed `insert_build_tag` idempotent,
  `emit_pypi.rs:398-420`, test `610-614`). If a future shadow had a DIFFERENT build tag
  upstream, the 6-field branch would REPLACE it with 999retread — exactly what cold does
  too. Aligned. Note: the §3.1 determinism guard also pins this — if the fetched name
  doesn't reduce to `lw.filename`, replay falls through rather than drifting.
- **R3 (sha256=None download + repackaged-upstream)** — `fetch_wheel_cached(None,...)`
  bypasses the persistent cache and calls `fetch_wheel`, landing at
  `dest_dir.join(wheel_filename_from_url)` (the pristine percent-decoded name; grizzly
  confirmed `wheel.rs:212-215/39-52`). Same as cold. The lock stores no sha256
  (accumulated finding). The §3.1 DETERMINISM GUARD upgrades this from "silent trust":
  if upstream served a repackaged/differently-named artifact, the predicted shadow name
  diverges from `lw.filename` → `return Ok(None)` → cold re-resolve. PINNING ASSUMPTION
  (documented): replay reproduces the RECORDED artifact name; a repackaged upstream is
  invalidated only via cold resolve. LOW.
- **R4 (download cost on replay)** — replay already re-materializes wheel bytes by
  design (HANDOFF goal #2); downloading the Class-2 shadow is materialization, not
  derivation. Non-conda_capable Class-2 shadows ALREADY download on replay (remote-only
  force-download); we just switch the primitive. No new derivation. ACCEPTABLE.
- **R5 (genesis/isaac6 regression)** — neither has a conda_capable Class-2 shadow that
  drifts; the e2e re-asserts byte-identity on both. Guarded.
