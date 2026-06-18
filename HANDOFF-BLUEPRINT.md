# v1.7.0 blueprint mode — validated plan (scratch, never commit)

Goal: ZERO dependency-overrides table in the workspace manifest. Static ~7-line
fenced block (auto-synced by the existing sync_workspace_block); override
semantics live in REWRITTEN WHEELS in find-links; entry pins live in a generated
META-WHEEL. Light blueprint (~160MB measured pre-cap-handling) committable for
cluster `git clone && pixi install` with zero backend execution.

## Proven by kill-experiment (2026-06-12, pixidock_template)
- Build-tagged wheel `isaacsim_core-6.0.0.0-999retread-cp312-...whl` in
  find-links BEATS the registry original at the same version: lock selected it,
  installed METADATA carries the rewritten line. (uv PrioritizedDist merge +
  build-tag tiebreak; pixi hardcodes HashStrategy::None both at solve and
  install, so the hash-first tuple field is always Matched/equal.)
- Meta-wheel isaac_pack_pypi-6.0.0.0-py3-none-any.whl (METADATA 2.1 +
  Requires-Dist) resolves from find-links as a direct pypi-dependency with NO
  index 404 failure; transitive (isaaclab-assets==0.3.0) pulled.
- CONSTRAINT: the meta-wheel pypi-dependencies entry must NOT carry `index=`
  (explicit index bypasses find-links in pixi's resolver provider).

## Implementation steps (architect plan + grizzly mandatory changes)
1. config.rs: `retread-blueprint` bool (sub-mode of retread-emit-pypi), serde test.
2. wheel_rewrite.rs: RECORD line hash -> PEP 376 base64url-nopad in
   update_record_line (hygiene, non-blocking; pixi never validates). Make
   rebuild_requirement pub(crate).
3. emit_pypi.rs `override_line_map(overrides, bundle) -> Fn(&str)->Option<String>`:
   table semantics as a Requires-Dist mapper. MANDATORY: handle CAP-ONLY lines
   (`foo<2`) — the v1.6 table was structurally blind to them (lower_bound
   returns None); relax/strip caps that exclude the bundled/conda-capable
   space. Exact-pin == existing -> None (unchanged; preserves family pins).
4. `insert_build_tag(std_name, "999retread")` — PEP 427 slot after version.
   999 not 1 (robust to upstream build-tagged republishes).
5. Ship policy: changed -> tagged + shipped; unchanged index wheel -> skip;
   unchanged built/.injected or URL-target -> ship untagged (as today).
   Remote (https) wheels: run map over in-memory requires_dist; on hit,
   fetch+rewrite+ship (warn); expect zero on isaac.
6. build_meta_wheel: <bundle>-pypi wheel, METADATA 2.1, Root-Is-Purelib,
   Requires-Dist = [retread-wheels] entry pins w/ extras
   (isaacsim[all,extscache]==6.0.0.0), RECORD b64 hashes.
7. render_snippet_blueprint: find-links + `<bundle>-pypi = "==<ver>"` +
   system-requirements.libc. NO table. prerelease: rewritten lines mention
   prereleases (self-enabling under if-necessary-or-explicit); fallback
   prerelease-mode="allow" if live test fails.
8. emit() branches on config.blueprint; hard switch; fence converges.
9. Staleness marker blueprint.lock (config hash + shipped set). DOC: pixi.lock
   must be (re)locked after blueprint changes — pixi satisfiability is BLIND to
   find-links dir contents (string-compares the path only); a stale lock
   silently installs registry wheels.
10. Live gates: lock references tagged wheels; installed METADATA rewritten;
    zero remote fetches; Kit boot; family ==6.0.0.0 throughout; cluster sim
    (fresh clone, no backend on PATH).

## Current local state (uncommitted, v1.6.1)
- wheel_rewrite: rewrite_wheel_with (generic mapper + hardlink-on-unchanged) DONE.
- emit_pypi: sync_workspace_block + auto-sync wiring (workspace_dir threaded
  through build_one) DONE, 243 tests green.
- Version 1.6.1 in Cargo.toml/recipe; local channel built; NOT shipped.
- pixidock workspace has experiment artifacts in
  isaac-pack/retread-pypi/isaac-pack/wheels/ (999retread + meta wheel) and a
  hand-added `isaac-pack-pypi = "==6.0.0.0"` line in the pasted block.
