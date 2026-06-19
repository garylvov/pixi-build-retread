# PHASE 2.6 PLAN — portable replay provenance for sdist-built BFS transitives (gym)

Branch: `courier`. Author: solution-architect. Audience: the-grizzly (audit DONE,
SHIP-WITH-CHANGES) then swe-worker-bee (impl).
All line numbers verified by direct read of the working tree at planning time (commit 11ce235).
Revised once to fold in the-grizzly's 4 blocking + 2 recommended amendments (§11 changelog).

---

## 0. EXECUTIVE SUMMARY (read this first — it overturns the handoff premise)

The PHASE2.6 prompt + HANDOFF-REPLAY-LOOP.md describe the bug as: "cold records gym
`origin=index` + `url=file://.../wheels/gym/...`; replay flips it to `origin=built`,
drops `url`; fix = store gym's portable PyPI **wheel** https URL and re-fetch the wheel."

**That premise is empirically FALSE for the current code.** The grizzly and I read
the COMMITTED locks AND the resolver. The truth:

1. **gym is built from an SDIST, not downloaded as a wheel.** `wheels/gym/` contains
   BOTH `gym-0.26.2.tar.gz` (downloaded sdist) and `gym-0.26.2-py3-none-any.whl`
   whose `dist-info/WHEEL` says `Generator: setuptools (82.0.1)` with a local
   build timestamp and a `licenses/` subdir — i.e. it is a **locally-built** wheel,
   not PyPI's. `src/pypi.rs:124` names gym as the canonical sdist-fallback example
   ("gym, classic-control, packages where the maintainer never uploads wheels").
   So **gym has no portable PyPI wheel URL to store.** The "fix" in the prompt is
   unimplementable as stated.

2. **The COMMITTED cold lock baseline (the acceptance target) records gym as
   `{origin:"built", filename:"gym-0.26.2-999retread-py3-none-any.whl", url:ABSENT,
   upstream_url:ABSENT, must_ship:false}`** — NO url, NO upstream_url, NO
   sdist_source. (Verified by direct read of all three committed isaac locks, §1.1.1.)
   The `999retread` build tag in the filename proves gym **IS relax-rewritten** (via
   `insert_build_tag`, `courier.rs:703`/`741`), reconciling the two henry traces: it
   is a `ShadowSrc::Rewritten`/`Raw` shadow. The `upstream_url=file://` value the
   first plan draft cited was a TRANSIENT `.pixi/bld` artifact (the RARE branch where
   `wheels/gym/` happened to be pre-localized at capture); it is NOT what gets
   committed and is NOT the norm. **The committed baseline is the source of truth.**

3. **`upstream_url` is therefore ABSENT in the committed lock, but the underlying
   capture is non-deterministic** (`mod.rs:2951` reads `resolved_url`, which for the
   sdist path is `built_url`/`file://` at `mod.rs:3198,3215`, varying with `wheels/`
   state — `None` in the common fresh case, a file:// path in the pre-localized
   case). This latent non-determinism is what the sdist descriptor removes for good.

4. **The TRUE current failure on a fresh box is a REPLAY ABANDON, not a file:// fetch.**
   Replay Class-2 (`mod.rs:4518-4540`) reads `lw.upstream_url` = `None` (committed
   baseline) → parses to `None` (`mod.rs:4523-4526`) → **`return Ok(None)`**
   (`mod.rs:4538`) → `materialize_from_lock` ABANDONS the whole replay → falls through
   to **FULL RESOLVE** → python_abi / version drift → lock NOT byte-identical → goal
   violated. (The `reqwest::get(file://)` error path at `courier.rs:650` is only
   reachable in the rare pre-localized-file:// transient; the dominant, committed-lock
   failure is the `Ok(None)` abandon.)

### The correct, general, root-cause fix

gym is one instance of a general class: **a BFS-transitive PyPI dep that PyPI ships
only as an sdist, so retread source-builds it.** The portable provenance for such a
wheel is the **resolved sdist https URL** (`sdist.url` at `mod.rs:3187`, e.g.
`https://files.pythonhosted.org/.../gym-0.26.2.tar.gz#sha256=...`, carrying a PEP-503
sha256 fragment per `pypi.rs:197`; returned straight from the PyPI simple index by
`parse_index_links_any` — verified `pypi.rs:127-177,181-213`), together with the fact
that replay must **re-build from that sdist URL** rather than re-fetch a wheel.

This is exactly the `Sdist{...}` `WheelSource` variant the handoff's own ACCUMULATED
FINDINGS anticipated and conditionally deferred ("sdist fallback ... OR prove
empirically no pack hits it (isaac: 0 sdist)"). **isaac-pack DOES hit it** (gym, via
`isaaclab-rl[all]` → `rl-games` → `gym`). So we implement the variant now, mirroring
the existing `GitWheelSource` design (`lock.rs:52-80`) one-for-one.

**Design: add `SdistWheelSource{index, name, version, sdist_url}` to
LockWheel/ResolvedWheel/EmitWheel (schema 8→9). Cold: when a BFS transitive is
sdist-built, record the sdist provenance (incl. the exact resolved `sdist_url` with
its #sha256). Replay: a new Class-2b arm re-builds directly from the stored
`sdist_url` (falling back to `resolve_sdist(index,name,version)` only if that URL
404s), producing a deterministic, portable, byte-identical lock entry.**

This is *symmetric* with `GitWheelSource` (git source-built wheels: store the
RESOLVED SHA, not HEAD) and `upstream_url` (index-fetched relax shadows): each
Built-wheel materialization mode gets its own inline, manifest-independent provenance
descriptor pinned to a concrete resolved artifact. No gym special-case.

---

## 1. ROOT-CAUSE MECHANISM (verified, file:line)

### 1.1.1 THE COMMITTED BASELINE (the acceptance source of truth)

The grizzly read the COMMITTED locks. The real committed gym entry (schema 4 today,
identical shape across all three isaac locks) is:

```json
{ "name":"gym", "version":"0.26.2", "origin":"built",
  "filename":"gym-0.26.2-999retread-py3-none-any.whl" }
```

i.e. `url`=ABSENT(None), `upstream_url`=ABSENT(None), `must_ship`=false (schema-4 has
no `requires_dist`/`upstream_url`/`git_source` keys; schema-8 regens keep
`url`/`upstream_url` absent too — see the `Fn9ID7Tp3vw` bld dir which has
`upstream_url:None`). The `999retread` build tag proves gym **IS relax-rewritten**
(`insert_build_tag`, `courier.rs:703`/`741`). The `upstream_url=file://` value cited in
the first plan draft was a transient `.pixi/bld` artifact from the RARE pre-localized
branch, NOT the committed norm. **All convergence proofs below target this committed
baseline.** Post-fix, the cold lock entry is IDENTICAL except it ADDS `sdist_source`
(and `requires_dist` from the schema-5+ bump already in place); `url`/`upstream_url`
stay absent.

Cold baseline TODAY (committed):
`{origin:built, filename:gym-0.26.2-999retread-..., url:None, upstream_url:None,
must_ship:false, sdist_source:None}`
Cold POST-FIX:
`{origin:built, filename:gym-0.26.2-999retread-..., url:None, upstream_url:None,
must_ship:false, sdist_source:{index,name,version,sdist_url}, requires_dist:[...]}`

### 1.1 Cold path — how gym becomes a `Built` shadow with NO portable provenance

- **BFS resolves gym.** `bfs_fetch_pypi` (`mod.rs:3120`) calls `pypi::resolve`
  (`mod.rs:3136`). For gym this fails (no compatible wheel on the index), so the
  **sdist fallback** fires (`mod.rs:3167-3216`):
  - `pypi::resolve_sdist(index, name, specifiers)` → `sdist` (`mod.rs:3175`). Its
    `sdist.url` is the **portable PyPI https tarball URL with a #sha256 fragment**
    (`pypi.rs:176` returns the `ResolvedWheel` whose `.url`/`.sha256` came from
    `parse_index_links_any`, `pypi.rs:197-207`).
  - `build_wheel_from_sdist_url(&sdist.url, &sdist_out, py)` builds the wheel
    locally (`mod.rs:3186`, signature `source_build.rs:136`).
  - `built_url = file://<built path>` (`mod.rs:3198`).
  - **`(built_url, metadata)` is returned as `(resolved_url, metadata)`**
    (`mod.rs:3215`), and `bfs_fetch_pypi` returns `(resolved_url, metadata, index)`
    (`mod.rs:3218`). **`sdist.url` is discarded here — THE DISCARD POINT.**
- **BFS phase-3 captures provenance.** For a `PendingSource::Pypi` item
  (`mod.rs:2944-2952`): `upstream = Some(resolved_url.clone())` (`mod.rs:2951`).
  For gym `resolved_url == built_url == file://`, so `upstream = Some(file://)` (a
  machine-local, non-deterministic path — `None` once it is not localized).
  This flows into `ResolvedWheel{ url: sub_url, upstream_url: sub_upstream_url,
  git_source: sub_git_source(=None for Pypi), ... }` (`mod.rs:3062-3070`).
- **`build_one` builds the EmitWheel** (`mod.rs:4837-4865`); `upstream_url =
  w.upstream_url.clone()` (`mod.rs:4860`) — the file://-or-None value above.
- **courier::stage classifies it as a relax shadow.** gym's strict pins
  (`gym_notices>=0.0.4`, etc.) relax-change → `did_change=true`
  (`courier.rs:567,595`) → `ShadowSrc::Rewritten`/`Raw` (`courier.rs:597`/`624`),
  emitting filename `gym-0.26.2-999retread-...` (`insert_build_tag`, `courier.rs:703`).
  The arm writes `origin=Built`, `must_ship=false`, `url=None` (`courier.rs:729`),
  and `upstream_url = w.upstream_url.or(w.remote_url)` (`courier.rs:719-723`) — which
  in the committed norm is **None** (no portable provenance recorded at all).
- **Net committed result:** `{origin:built, must_ship:false, url:None,
  upstream_url:None}`. **Replay has nothing to re-materialize from → it abandons (§1.2).**

### 1.2 Replay path — why it ABANDONS → full resolve → drift

- Schema gate: `materialize_from_lock` dispatches on `lw.origin` (`mod.rs:4188`).
  gym hits `Origin::Built` non-must_ship → **Class-2** (`mod.rs:4518-4540`).
- Class-2 reads `lw.upstream_url` = **None** (committed baseline), parses it
  (`mod.rs:4523-4526`) → `None` → hits the `None` arm (`mod.rs:4527-4540`) and
  **`return Ok(None)`** (`mod.rs:4538`).
- `materialize_from_lock` is all-or-nothing: the first `Ok(None)` ABANDONS the entire
  replay → the caller falls through to **full `resolve_all`** → BFS re-runs, solve
  re-runs, python_abi/version drift → committed lock NOT byte-identical. **This is the
  dominant primary-goal violation** for the committed isaac packs.
- (The `reqwest::get(file://...)` error at `courier.rs:650` is only reachable in the
  RARE transient where a committed lock carried `upstream_url=file://`; no committed
  lock does, so that path is secondary. The fix eliminates both.)

### 1.3 Why this is general, not gym-specific

Any PyPI dep with no index-compatible wheel takes the same sdist fallback
(`mod.rs:3167`). The fix targets the *mechanism* (sdist-built BFS transitive
provenance), so every such dep — present or future, in any pack — is covered.

---

## 2. PRE-EMPTING GRIZZLY HOLES (the prompt's (a)-(d), answered from code)

- **(a) Is `w.upstream_url` Some(https) for gym at the courier write site?**
  **No.** It is `None` in the committed norm (and `Some(file://)` in the rare
  pre-localized transient) — the sdist build path sets `resolved_url=built_url=file://`
  BEFORE upstream capture (`mod.rs:3215` → `mod.rs:2951`), and the committed lock
  records `upstream_url:None`. So the prompt's plan ("prefer w.upstream_url for the
  index url") would store either nothing (→ replay `Ok(None)` abandon, §1.2) or a
  machine-local file:// path; neither fixes portability. **This is why the prompt's
  fix is wrong and we need the sdist descriptor.**
- **(b) Could preferring upstream_url over remote_url break wheels where upstream_url
  is file:// or a non-PyPI index?** It already IS file:// for sdist wheels (the bug).
  For genuine index shadows (isaacsim-* from pypi.nvidia.com) `upstream_url` is
  https (verified: isaac6 lock entries are `https://pypi.nvidia.com/...`,
  `lock.rs`-schema-8). Our fix does NOT touch the https-index-shadow path; it only
  redirects the **sdist-built** subset to the new descriptor (gated on a definitive
  signal, §3.1). pypi.nvidia.com shadows keep `upstream_url=https` and replay via
  Class-2 unchanged → no regression.
- **(c) Does any Origin::Index wheel legitimately need a file:// url (a committed
  find-links source)?** **No** for the real packs: `wheels/` is gitignored
  (verified `git check-ignore .../wheels/gym/...whl` → GITIGNORED; the dir's
  `.gitignore` is `*`). No committed lock has any `"url": "file://"` entry (verified
  by henry across both repos: 0 hits). So no legitimate file:// provenance is being
  clobbered. We still keep the Class-4 file:// filter (`mod.rs:4196`) as a defensive
  no-op for old/odd locks (§3.4).
- **(d) Is gym built from sdist (Sdist source) rather than a downloaded wheel?**
  **YES — confirmed three independent ways** (§0.1): the pypi.rs comment, the
  `.tar.gz` sidecar in `wheels/gym/`, and the built wheel's `WHEEL` metadata
  (`Generator: setuptools (82.0.1)`, local build date, `licenses/` dir). So the
  portable provenance is the **sdist URL + source-build**, not a wheel URL. This is
  the load-bearing correction the prompt asked us to verify.

---

## 3. THE DESIGN (mirrors GitWheelSource exactly)

### 3.1 Detecting "sdist-built" at the BFS site (the only new signal)

`bfs_fetch_pypi` currently returns `(url, metadata, index)` (`mod.rs:3127`,
`mod.rs:3218`) and hides whether the wheel came from a download or an sdist build.
**Extend its return to carry an optional sdist descriptor:**

```rust
// new return type
async fn bfs_fetch_pypi(...) -> Result<(url::Url, WheelMetadata, String, Option<SdistProv>)>
struct SdistProv { index: String, name: String, version: String, sdist_url: url::Url }
```

- Wheel path (`mod.rs:3162-3166`): return `(resolved.url, metadata, index, None)`.
- Sdist path (`mod.rs:3167-3216`): after building, return
  `(built_url, metadata, index, Some(SdistProv{ index: index.into(),
   name: pypi_name.into(), version: metadata.version.clone(),
   sdist_url: sdist.url.clone() }))`. **`sdist.url` is now CAPTURED, not discarded.**
  `version` comes from the built wheel's parsed metadata (`mod.rs:3204-3209`), which
  is the authoritative resolved version (`resolve_sdist` already pinned it).

This is a *definitive* signal (we are literally in the sdist branch), satisfying the
"gated on definitive probe" principle — no heuristic on filenames or URLs.

### 3.2 Schema: add `sdist_source: Option<SdistWheelSource>` (mirror GitWheelSource)

In `src/lock.rs`, add next to `GitWheelSource` (`lock.rs:52-80`):

```rust
/// Provenance for a wheel built locally from a PyPI sdist because the
/// index ships no target-compatible wheel (e.g. gym). On replay,
/// materialize_from_lock re-builds DIRECTLY from the recorded `sdist_url`
/// (the exact resolved https tarball + #sha256), falling back to
/// resolve_sdist(index, name, version) only if that URL 404s ->
/// deterministic, portable, manifest-independent. `None` for index-fetched
/// and git-built wheels.
///
/// POISONING: like git_source.rev, `sdist_source` is NOT folded into
/// inputs_hash (same circularity argument: it is a RESULT of resolution,
/// not an input to it; folding it would require resolving to compute the
/// hash that gates resolution). Replay reproduces the RECORDED sdist
/// artifact verbatim. With the stored https URL + #sha256, the only
/// residual risk is "the artifact was deleted/yanked from PyPI" -- the same
/// documented pinning floor as every other replay-trusted upstream
/// (git commit, index wheel). The #sha256 is a free integrity check on
/// re-fetch and is likewise NOT in inputs_hash.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SdistWheelSource {
    /// PEP 503 simple index base URL the sdist was resolved from.
    /// Human-readable provenance + fallback re-resolution key.
    pub index: String,
    /// PEP 503 normalized project name. Fallback re-resolution key.
    pub name: String,
    /// Resolved version (from the built wheel's METADATA). Fallback key.
    pub version: String,
    /// The EXACT resolved sdist URL (https://files.pythonhosted.org/.../
    /// <name>-<version>.tar.gz#sha256=<hex>). PREFERRED on replay: build
    /// straight from this, skipping a re-resolve. Carries the PEP-503
    /// #sha256 fragment when the index advertised one (pypi.rs:197).
    pub sdist_url: String,
}
```

Add to `LockWheel` (after `git_source`, `lock.rs:128`):
```rust
#[serde(default, skip_serializing_if = "Option::is_none")]
pub sdist_source: Option<SdistWheelSource>,
```

**We store the exact resolved `sdist_url` AND `(index, name, version)`** (grizzly
Amendment 4, OVERRIDING the first draft's index/name/version-only choice). Rationale:
`resolve_sdist` (`pypi.rs:127`) is **not yank-safe** (it never inspects
`data-yanked`) and **not deterministic across index-HTML reordering** (it collects
candidates, stable-sorts by version, and `next()`s — version ties are broken by index
order, `pypi.rs:152-176`). So re-resolving by `(index,name,version)` on a fresh box
can pick a DIFFERENT artifact than cold. Storing the exact `sdist_url` (with #sha256)
makes replay build the IDENTICAL tarball, skips a network resolve, and gives a free
integrity check — mirroring Phase-2's "store the RESOLVED SHA, not HEAD" lesson.
`(index, name, version)` is retained as human-readable provenance and a fallback
re-resolution key if the pinned URL 404s.

Add `sdist_source: Option<SdistWheelSource>` to `ResolvedWheel` (`mod.rs:461-485`,
next to `git_source`) and to `EmitWheel` (`emit_pypi.rs:59-68`, next to `git_source`).

**Schema 8 → 9.** Old schema-8 locks fail the `!=` gate → fall through to full
resolve (safe). Per `lock.rs:184-190`, SCHEMA is on-disk FORMAT, **not** an epoch bump.

### 3.3 Cold producer: thread sdist provenance, suppress file:// upstream_url

- **BFS phase-3** (`mod.rs:2944-2952`, the `PendingSource::Pypi` arm): destructure the
  new 4th element `sdist_prov`. Build `sub_sdist_source = sdist_prov.map(|p|
  SdistWheelSource{ index: p.index, name: p.name, version: p.version,
  sdist_url: p.sdist_url.to_string() })` — **carrying the exact resolved `sdist_url`
  with #sha256** (Amendment 4). Critically, **when `sdist_prov.is_some()`, set
  `upstream = None`** (do NOT store the file:// built_url). When it is a real wheel
  download, `upstream = Some(resolved_url)` as today. Push `sdist_source:
  sub_sdist_source` into the `ResolvedWheel` (`mod.rs:3062-3070`). (Other tuple slots
  unchanged.)
- **`build_one`** (`mod.rs:4837-4865`): add `sdist_source: w.sdist_source.clone()` to
  the EmitWheel literal (next to `git_source`, `mod.rs:4863`). `upstream_url` keeps
  reading `w.upstream_url` (`mod.rs:4860`) — now `None` for sdist wheels.
- **courier::stage** `ShadowSrc::Rewritten`/`Raw` arms (`courier.rs:698-736`,
  `737-775`): add `sdist_source: w.sdist_source.clone()` to both `LockWheel` literals.
  Keep `upstream_url = w.upstream_url.or(w.remote_url)` (`courier.rs:719-723`,
  `758-762`) — for an sdist wheel both are now `None`, so `upstream_url=None` in the
  lock (deterministic). The `ShadowSrc::None` arm (`courier.rs:685-696`) and the git
  Class-1 arm (`courier.rs:~540-562`) also get `sdist_source: None` /
  `w.sdist_source.clone()` to satisfy the struct (None in practice for those).
- **Net cold lock for gym (schema 9):** `origin=built, must_ship=false, url=absent,
  upstream_url=absent, sdist_source={index, name:"gym", version:"0.26.2",
  sdist_url:"https://files.pythonhosted.org/.../gym-0.26.2.tar.gz#sha256=..."},
  requires_dist:[...relaxed...]`. Deterministic (no file:// path), portable (no
  machine-local state). This is byte-identical to §1.1.1's committed baseline PLUS the
  `sdist_source`/`requires_dist` additions.

### 3.4 Replay: new Class-2b arm for sdist-built shadows

In `materialize_from_lock`, the `Origin::Built` non-must_ship arm (`mod.rs:4518`)
currently assumes `upstream_url`. **Branch on `lw.sdist_source` FIRST:**

```rust
Origin::Built if !lw.must_ship && lw.sdist_source.is_some() => {
    // Class 2b: relax shadow built from a PyPI sdist (e.g. gym).
    // PREFER the stored exact sdist_url (with #sha256); fall back to
    // resolve_sdist(index,name,version) only if that URL 404s. Produces
    // the SAME local wheel cold produced -> courier::stage runs the SAME
    // ShadowSrc::Raw/Rewritten rewrite -> byte-identical entry.
    let s = lw.sdist_source.as_ref().unwrap();
    let out = download_dir.join(&s.name);
    let stored_url = url::Url::parse(&s.sdist_url)?;
    let built = match build_wheel_from_sdist_url(&stored_url, &out, &lock.python).await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(name=%s.name, url=%s.sdist_url, err=%format!("{e:#}"),
                "Class-2b: stored sdist_url failed; falling back to resolve_sdist by (index,name,version)");
            let specifiers = VersionSpecifiers::from_str(&format!("=={}", s.version))?;
            let sdist = pypi::resolve_sdist(&s.index, &s.name, &specifiers).await?;
            build_wheel_from_sdist_url(&sdist.url, &out, &lock.python).await?
        }
    };
    EmitWheel {
        pypi_name: lw.name.clone(),
        version: lw.version.clone(),
        requires_dist: lw.requires_dist.clone(),
        local_path: Some(built),          // local-path branch in courier
        wheel_filename: lw.filename.clone(),
        remote_url: None,
        upstream_url: None,
        git_source: None,
        sdist_source: lw.sdist_source.clone(),
    }
}
```

DISPATCH ORDER (grizzly-confirmed): the `match lw.origin` arms must be, in order:
`Origin::Index` (Class-4, `mod.rs:4189`), then `Origin::Built if lw.must_ship`
(Class-1/3 git/path, `mod.rs:4229`), then **this new `Origin::Built if !lw.must_ship
&& lw.sdist_source.is_some()` (Class-2b)**, then the bare `Origin::Built =>` (Class-2
index shadow, currently `mod.rs:4518`). The new arm MUST be inserted **immediately
BEFORE** the bare `Origin::Built =>` at `mod.rs:4518` (Rust match arms are evaluated
top-to-bottom; the bare arm is the catch-all and would otherwise swallow sdist wheels).

**Why this converges byte-identically (§4 proves it):** the Class-2b EmitWheel has
`local_path=Some(built)` (same as cold's `build_one` for a localized sdist wheel),
`remote_url=None`, so courier::stage takes the **local-path branch**
(`courier.rs:567`), computes the SAME `did_change` (same raw sdist-built bytes, same
overrides/conda_capable from the lock), yields the SAME `ShadowSrc::Rewritten`/`Raw`,
and writes the SAME `LockWheel{origin=built, filename=gym-...-999retread-...,
url=None, must_ship=false, upstream_url=None, requires_dist (copied from lock),
sdist_source (copied from lock)}`. **No reqwest of a file:// path, no Ok(None) abandon
→ replay FIRES, fresh-AWS safe.**

### 3.5 Keep the Class-4 file:// filter (`mod.rs:4196`) as a defensive no-op

For https `url` it is already inert (verified: filter only drops file://). With cold
no longer writing file:// anywhere, it never fires for new locks; leave it to guard
malformed/old locks. No change.

---

## 4. BYTE-IDENTITY PROOF (POST-FIX cold vs replay for gym, schema 9)

The acceptance target is the **committed** `examples/gigastrap/isaac-pack/
retread-isaac-pack.lock.json` regenerated under 2.7.0 (§1.1.1 baseline + the
`sdist_source`/`requires_dist` additions). Both cold-regen and replay must emit the
SAME entry.

| Field / step | COLD (regen under 2.7.0) | REPLAY (Class-2b) |
|---|---|---|
| wheel bytes source | `resolve_sdist`+`build_wheel_from_sdist_url(sdist.url)` | `build_wheel_from_sdist_url(stored sdist_url)` — **same exact tarball URL + #sha256** → same uv build |
| EmitWheel.local_path | `Some(built)` | `Some(built)` |
| EmitWheel.remote_url | `None` | `None` |
| EmitWheel.upstream_url | `None` (suppressed §3.3) | `None` |
| EmitWheel.requires_dist | from built metadata | copied from lock (== same metadata) |
| EmitWheel.sdist_source | `Some{index,name,version,sdist_url}` | copied from lock (identical) |
| courier branch | local-path (`courier.rs:567`) | local-path |
| `insert_build_tag` | applied → `999retread` (`courier.rs:703`) | applied → `999retread` |
| did_change | true (single `rewrite_wheel_with`, `courier.rs:614`) | true (SAME single `rewrite_wheel_with`, same overrides/conda_capable from lock) |
| ShadowSrc | `Rewritten` | `Rewritten` |
| LockWheel.origin | built | built |
| LockWheel.filename | `gym-0.26.2-999retread-py3-none-any.whl` | identical |
| LockWheel.url | None (`courier.rs:729`) | None |
| LockWheel.must_ship | false | false |
| LockWheel.upstream_url | None (both srcs None) | None |
| LockWheel.sdist_source | `{index,name,version,sdist_url}` | copied from lock (identical) |

**Shadow-rewrite symmetry (Amendment 1):** gym IS relax-rewritten on BOTH paths
(the `999retread` build tag in the committed filename proves it). Both cold and replay
feed the SAME raw sdist-built bytes through the SAME single `rewrite_wheel_with`
(`courier.rs:614`/`749`) with the SAME override map (from `lock.conda_capable` +
override table). The two henry traces ("unchanged index" vs "rewritten shadow") are
reconciled: gym is the rewritten-shadow case, symmetric across cold/replay.

Both sides emit an identical `LockWheel`. The wheel BYTES need not be identical (lock
stores no wheel sha256 — HANDOFF ACCUMULATED FINDINGS); only the lock ENTRY must match,
and every field above is provably equal.

**Determinism guard (Amendment 3 — now a code change, not prose):**
`build_wheel_from_git` (`source_build.rs:369-386`) already warns via
`is_nondeterministic_version` (`source_build.rs:399`) when a built wheel's filename
carries a `.devN` / `.dYYYYMMDD` / `+g<sha>` setuptools_scm suffix (which would make
the FILENAME — stored in the lock — drift across calendar days). `build_wheel_from_sdist_url`
(`source_build.rs:136`) has **no such guard**. ADD the identical guard to
`build_wheel_from_sdist_url` after the build (warn, do not fail), mirroring the git
path verbatim (same message, swap `rev`→sdist context). This is inert for gym 0.26.2
(static released version) but is REQUIRED for the generality claim: any sdist whose
build backend emits a date/sha-suffixed version would otherwise silently drift the
lock filename on replay. Documented conditional, exactly as Phase-2 claim E did for git.

---

## 5. NO-REGRESSION ANALYSIS

- **genesis / newton / isaac6 committed locks:** henry verified ZERO `"url": "file://"`
  entries in any committed lock in either repo, and isaac6's pypi.nvidia.com shadows
  carry `upstream_url=https://...` (Class-2 index path, untouched by §3.4 since
  `sdist_source` is None for them). genesis/newton are Class-1 git (untouched).
- **Schema 8 → 9** invalidates ALL committed schema-8 locks (genesis, newton, isaac6,
  pixidock genesis/newton). They fall through the `!=` gate → cold solve (safe, slow),
  so **every committed lock must be regenerated under the new backend** (§7 step 8).
  This is unavoidable and identical to prior schema bumps (handoff: "ALL committed
  locks must be regenerated, or they fall through").
- **isaac packs (schema 4 today):** already fall through; regenerated to schema 9.
- **No new fall-through class for real packs:** isaac (gym sdist), genesis/newton
  (git), isaac6 (index) are all covered. Other sdist transitives, if any, hit the
  same Class-2b automatically.
- **Single sdist site (grizzly-confirmed coverage-complete):** only `bfs_fetch_pypi`
  calls `build_wheel_from_sdist_url` (`mod.rs:3186`) + `resolve_sdist` (`mod.rs:3175`).
  The primary-entry, auto_bundle, and cascade paths call `pypi::resolve` only, with NO
  sdist fallback — so there is no other place an sdist-built Built wheel can arise.
  Threading provenance at the one BFS site covers every sdist wheel a real pack
  produces.
- **Determinism guard adds no failure path:** the new `is_nondeterministic_version`
  call in `build_wheel_from_sdist_url` only `tracing::warn!`s; it never errors, so no
  pack regresses from its addition.

## 6. SCHEMA / EPOCH DECISION

- **SCHEMA 8 → 9** (lock FORMAT: new `sdist_source` field + new replay class).
- **EMIT_EPOCH:** stays **4** (grizzly-confirmed). Justification: the change is
  `[emit-epoch-ok]`. The emitted lock CONTENT for gym changes (adds `sdist_source` +
  `requires_dist`), but **none of this is in `compute_inputs_hash`** (verified
  `lock.rs:243-280`: it folds only entry_specs, index_urls, relax, python, epoch,
  pin_version, config_fingerprint — no per-wheel url/origin/upstream/sdist field).
  Per `lock.rs:192-203`, EMIT_EPOCH gates emitted-output SEMANTICS *for identical
  inputs*; a SCHEMA-only format change is explicitly excluded.
  **BUT** the CI emit guard (`.github/workflows/ci.yml:89-126`, regex matches
  `src/(...|courier|lock).rs` and `src/handler/(auto_bundle|cascade).rs`) WILL fire
  because we edit `src/courier.rs` and `src/lock.rs`. Resolution: include
  **`[emit-epoch-ok]`** in the commit messages that touch those files (the guard
  accepts that token as the ack), since inputs_hash is provably unaffected. Editing
  `src/handler/mod.rs` does NOT trip the guard (not in the regex). State this in the
  commit body with the `compute_inputs_hash` citation.

## 7. IMPLEMENTATION ORDER (commits on `courier`; green bar each)

1. `src/lock.rs`: add `SdistWheelSource` struct (incl. `sdist_url`) +
   `LockWheel.sdist_source`; bump `SCHEMA` 8→9; update the schema doc comment + add the
   poisoning doc note (Amendment 5, already in the struct doc §3.2). **`[emit-epoch-ok]`**
   (touches lock.rs; inputs_hash unaffected). Build + `cargo test --lib`.
2. `src/source_build.rs`: add the `is_nondeterministic_version` warn guard to
   `build_wheel_from_sdist_url` (`source_build.rs:136`), mirroring the git path
   (`source_build.rs:369-386`). Amendment 3. (Not in CI guard regex.) Unit-test the
   guard fires on a `.dYYYYMMDD` sdist-built filename and is silent on a static one.
3. `src/handler/mod.rs`: add `ResolvedWheel.sdist_source`; change `bfs_fetch_pypi`
   return to the 4-tuple with `Option<SdistProv{index,name,version,sdist_url}>`;
   capture `sdist.url` (the exact resolved tarball URL with #sha256) in the sdist
   branch; in BFS phase-3 set `upstream=None` when sdist + populate `sub_sdist_source`
   carrying `sdist_url`. (mod.rs not in CI guard regex.)
4. `src/emit_pypi.rs` + `src/handler/mod.rs` build_one: add `EmitWheel.sdist_source`;
   thread `w.sdist_source` in `build_one`. **`[emit-epoch-ok]`** (touches emit_pypi.rs).
5. `src/courier.rs`: add `sdist_source` to all `LockWheel` literals (Rewritten/Raw/None
   index arms + git Class-1 arm). **`[emit-epoch-ok]`**.
6. `src/handler/mod.rs`: add the Class-2b replay arm in `materialize_from_lock`,
   inserted IMMEDIATELY BEFORE the bare `Origin::Built =>` arm at `mod.rs:4518`
   (prefer stored `sdist_url`, fall back to `resolve_sdist`; §3.4).
7. Tests (§8). Bump Cargo.toml/recipe version 2.6.0 → 2.7.0 (new schema + replay
   capability). `cargo test --lib` + `clippy --all-targets -D warnings` + `fmt`.
8. (orchestrator) Publish 2.7.0; bump pixidock backend pins to `>=2.7.0`; then
   **regenerate EVERY committed lock to schema 9 under the 2.7.0 backend** — schema 8→9
   falls ALL of them through to cold solve, so each must be regenerated or it never
   replays. EXPLICIT checklist (Amendment 2):
   - [ ] `/home/garylvov/projects/pixi-build-retread/examples/gigastrap/isaac-pack/retread-isaac-pack.lock.json` (carries gym; **the §4 acceptance diff target**)
   - [ ] `/home/garylvov/projects/pixidock_template/isaac-pack/retread-isaac-pack.lock.json` (carries gym)
   - [ ] `/home/garylvov/projects/pixidock_template/isaac-pack-latest/retread-isaac-pack-latest.lock.json` (carries gym)
   - [ ] `/home/garylvov/projects/pixi-build-retread/examples/genesis/genesis-pack/retread-genesis-pack.lock.json`
   - [ ] `/home/garylvov/projects/pixi-build-retread/examples/isaac6/isaac-pack/retread-isaac-pack-6.lock.json`
   - [ ] `/home/garylvov/projects/pixidock_template/genesis-pack/retread-genesis-pack.lock.json`
   - [ ] `/home/garylvov/projects/pixidock_template/newton-pack-latest/retread-newton-pack-latest.lock.json`
   - [ ] any newton lock under `examples/` if present (grep `retread-newton*.lock.json`)
9. (orchestrator) Lukewarm e2e (§9) on **BOTH isaac-pack AND isaac-pack-latest** (both
   carry gym) PLUS genesis + newton + isaac6 regression. On PASS commit all the
   schema-9 locks.

## 8. TESTS

- **Unit (byte-identity parity for an sdist-built shadow):** a localhost-fixture test
  mirroring the existing index/git parity tests. Serve a tiny sdist (only a `.tar.gz`,
  no `.whl`) over a local HTTP index (reuse the existing test index harness), drive a
  cold produce that takes the sdist fallback, assert the cold `LockWheel` has
  `origin=built, url=None, upstream_url=None,
  sdist_source=Some{index,name,version,sdist_url(=the served https tarball URL)}`; then
  run `materialize_from_lock` on that lock and assert the reconstructed `LockWheel` is
  byte-identical (origin/url/upstream_url/filename/sdist_source/requires_dist all
  equal) AND that replay built from the STORED `sdist_url` (not a re-resolve). Red
  before §3.3-§3.4, green after.
- **Unit (no file:// in lock):** assert neither `LockWheel.upstream_url` nor `.url` nor
  `sdist_source.sdist_url` begins with `file://` for the sdist-shadow fixture (the
  portability invariant — `sdist_url` must be the https tarball, never the local build
  path).
- **Unit (determinism guard, Amendment 3):** assert `build_wheel_from_sdist_url` warns
  via `is_nondeterministic_version` on a `.dYYYYMMDD`-suffixed built filename and is
  silent on a static version (mirror the git guard test at `source_build.rs:687+`).
- **Schema-gate test:** a schema-8 lock is REJECTED by the `!=` gate (falls through to
  cold solve), not mis-replayed under schema 9.
- Keep `cargo test --lib` (currently 334) green; clippy `-D warnings`; fmt.

## 9. LUKEWARM E2E (BOTH isaac packs; existing harness)

Run on **isaac-pack AND isaac-pack-latest** — both carry gym (Amendment 2), so both
exercise Class-2b; a one-pack run would miss a per-pack drift. The acceptance diff
target is the COMMITTED `examples/gigastrap/isaac-pack/retread-isaac-pack.lock.json`
regenerated under 2.7.0 (§4).

For each isaac pack: nuke ALL caches incl `<pack>/wheels` and the uv/rattler/retread
caches, KEEP `.pixi/config.toml`. Use the local 2.7.0 backend. Assert:
- `build_v1: replayed from lock` PRESENT; shared-checkout error = 0 (Phase 2.5 holds);
  derivation = 0 (no `auto-bundled`, no `resolvo solve finished`, no probe-trace).
- `<pack>/wheels` repopulated from EMPTY (gym re-built from the stored sdist_url;
  isaaclab git group rebuilt).
- **`git diff --exit-code` CLEAN on the pack's `retread-*.lock.json`** (gym no longer
  drifts; no file:// path anywhere in the committed lock — incl. `sdist_source.sdist_url`,
  which must be https).
- `python -c "import isaacsim; import isaaclab"` (with `OMNI_KIT_ACCEPT_EULA=YES`;
  watch the known EULA-prompt false-fail).
Run genesis + newton + isaac6 too (regression): all must still replay byte-identically
under schema 9 (they have no sdist transitive, so Class-2b is inert for them).

---

## 10. WHAT THIS PLAN DELIBERATELY DOES NOT DO

- Does NOT store gym's "PyPI wheel https URL" (the prompt's premise) — gym has none;
  it is sdist-built. Storing the file:// built path (prompt's fallback) would be the
  non-portable band-aid the goal forbids.
- Does NOT special-case gym — the `SdistWheelSource` mechanism covers every
  sdist-fallback transitive generally.
- Does NOT change `compute_inputs_hash`, the relax algorithm, version selection, or the
  git/index replay paths — minimal blast radius, symmetric with existing
  `GitWheelSource`/`upstream_url` provenance descriptors.
- Does NOT rely on re-resolution as the primary replay path: per Amendment 4 it FREEZES
  the exact resolved `sdist_url` (with #sha256) and builds from it directly, because
  `resolve_sdist` is neither yank-safe nor reorder-deterministic; re-resolution is the
  fallback only.

---

## 11. REVISION CHANGELOG (grizzly amendments folded in)

- **A1 (blocking) — drift narrative corrected to the COMMITTED baseline:** §0.2-0.4,
  §1.1.1 (new), §1.1, §1.2, §4. Committed gym = `{origin:built,
  filename:gym-...-999retread-..., url:None, upstream_url:None, must_ship:false}`. True
  failure = Class-2 `return Ok(None)` (`mod.rs:4538`) → replay ABANDON → full resolve →
  drift (NOT `reqwest::get(file://)`). gym IS relax-rewritten (`999retread` tag,
  `courier.rs:703`); shadow-rewrite is symmetric across cold/replay (single
  `rewrite_wheel_with`). Post-fix cold = baseline + `sdist_source` + `requires_dist`.
- **A2 (blocking) — explicit regen checklist:** §7 step 8 lists all 7+ committed lock
  paths (the three gym-carrying isaac locks + genesis/newton/isaac6 in both repos).
  §9 e2e covers BOTH isaac-pack AND isaac-pack-latest; acceptance target =
  `examples/gigastrap/isaac-pack/retread-isaac-pack.lock.json`.
- **A3 (blocking) — determinism guard as code:** §4, §7 step 2, §8. Add the
  `is_nondeterministic_version` warn to `build_wheel_from_sdist_url`
  (`source_build.rs:136`), mirroring the git guard (`source_build.rs:369-386,399`).
- **A4 (recommended, adopted) — store the exact `sdist_url`:** §0, §3.2 (struct field),
  §3.3 (capture), §3.4 (prefer stored URL, fallback to `resolve_sdist`), §4, §8, §10.
  `resolve_sdist` is not yank-safe / not reorder-deterministic (`pypi.rs:127,152-176`),
  so the frozen URL + #sha256 is the deterministic, portable replay key.
- **A5 (recommended, adopted) — poisoning doc:** §3.2 struct doc note (sdist_source not
  in inputs_hash; same circularity as git_source.rev; pinning floor = artifact deleted
  from PyPI; #sha256 = free integrity check).
- **Confirmed-good, unchanged:** single sdist site (coverage complete, §5); EMIT_EPOCH
  stays 4 + `[emit-epoch-ok]` (§6); Class-4 file:// filter stays a defensive no-op
  (§3.5); Class-2b dispatched before bare `Origin::Built =>` at `mod.rs:4518` (§3.4);
  version 2.6.0→2.7.0; files-to-touch set (§7).
