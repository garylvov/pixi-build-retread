# Cleanup plan (FINAL, post-adversarial-revision) — 2026-06-11

Pipeline: bloat-killer + grizzly review -> architect plan -> adversarial
re-review (both) -> THIS revision -> final review pass -> implement.
Scratch doc (HANDOFF* convention: never committed).

Every phase ends green: cargo fmt; clippy -D warnings (0); cargo test
--lib; release-build protocol test --include-ignored. Phases 0c/1/2/3/4
also run the examples/isaac6 pixi lock e2e. Version bump per shipped
phase (invariant #5).

## Phase 0a — provably-identical normalizer unify + dead code (behavior-zero)
- `conda_name_simple` in relax.rs = the EXACT shared body of
  handler::conda_name_from (handler.rs:4888) + workspace::conda_normalize
  (workspace.rs:645): lowercase + `_`->`-` only. Swap 29 + 1 call sites.
- Delete dead: check_on_any_channel (:4790), conda_candidates_for (:194),
  class_label (:1969).
- map_name UNTOUCHED (different PEP 503 semantics -> 0c).
- Test: conda_name_simple_matches_legacy_twins property/table test.
- Gate: no e2e (zero behavior change).

## Phase 0b — leaf extraction (behavior-zero, whole-fn moves, NO renames)
- handler.rs tests module (5921-7710) -> handler/tests.rs.
- Leaf submodules: audit_report (~5410-5718), auto_bundle (~4441-5125),
  cascade (~1980-3002 + 3162-3446). pub(crate) visibility only.
- conda_outputs orchestrator + resolve_all STAY in handler.rs
  (invariant #9). Each move its own commit, green between.
- P1-3 author their new tests directly in the new files.

## Phase 0c — PEP 503 adoption (EXPLICIT semantic change)
- canonical_conda_name = map_name's PEP 503 body (relax.rs:87-103,
  collapses `. _ -` runs); map_name does raw-keyed override lookup
  FIRST (preserve :84-86) then canonicalizes. Replace conda_name_simple
  everywhere; delete it.
- assert_spec_roundtrips(name, spec) debug_assert at 3 boundaries:
  join_transitive_to_overrides output, override-map insertion,
  produce_output assembly.
- Test matrix: skip-set class (ruamel.yaml->ruamel-yaml, a__b->a-b,
  trim); name-map class incl. override-keyed-on-RAW guard
  (map_name("ruamel.yaml", {"ruamel.yaml": X}) == X); ABI-compare class
  (is_abi_anchor unchanged for python_abi/cuda-version/libstdcxx-ng —
  proves anchor surface is a no-op); roundtrip trips on embedded-space
  spec.
- Gate: + isaac6 e2e; lock must be byte-identical (no isaac name has
  dots). If the lock changes, STOP and investigate.
- Bookkeeping (bloat-killer final note): deleting conda_name_simple in
  0c also deletes/rewrites the 0a property test against
  canonical_conda_name IN THE SAME COMMIT, else the dead-code gate
  trips.

BOTH REVIEWERS: APPROVE TO IMPLEMENT (2026-06-11).

## Phase 1 — abstention terminal state + cache key
- ONE pure classify_run_terminal(envs_attempted, envs_skipped,
  block_messages) -> RunTerminal used by BOTH banner and MD-guard.
- envs_skipped counter (near :469; else-branch of :919).
- Banner whenever skipped_count > 0: "ABSTAINED for N of M envs ...
  shipped UNVERIFIED" (partial AND full). Audit gains
  skipped_count/all_skipped (serde round-trip test). NOT a hard fail.
- write_solve_failed_summary (:5526): delete prior MD only when run
  verified (>=1 env actually solved) and none unsat; abstained runs
  leave MD intact.
- Typo/absent env: WorkspaceManifest::has_environment; FILTER out of
  env_names before leveling (:655-683) with warn naming the env (never
  reaches trivial-sat / any_solve_passed).
- repodata.rs:38-43 comment corrected: None IS cached for process
  lifetime (short-lived by design); write_atomic rename documented as
  mmap-load-bearing.
- conda_outputs_cache_key (:55): + workspace manifest mtime. Correct
  rationale: manifest loads at :541/617/1525/1747 are mtime-memoized,
  but CONDA_OUTPUTS_CACHE returns memoized results BYPASSING them.
- Tests: classify_run_terminal table (all-sat/all-unsat/all-skipped/
  partial); partial_skip_banners; abstained_run_preserves_prior_md;
  verified_sat_clears_stale_md; absent_env_filtered_before_leveling;
  cache_key_changes_on_manifest_mtime.
- Gate: + isaac6 e2e; nuke repodata cache -> banner fires.

## Phase 2 — skip/seen normalization symmetry (depends 0c)
- Skip set seeded canonical (:4459 -> canonical_conda_name(w.pypi_name));
  BFS seen seeded canonical (:3520); seed_worklist both sides audited.
- already_covered(pypi_name, &skip_set) helper replaces the 3
  dual-namespace sites.
- Tests through real consumers: opencv_python bundled / opencv-python
  transitive not re-bundled; ruamel.yaml BFS dedup; raw-seed asymmetry
  regression.
- Gate: + isaac6 e2e (audit wheel set unchanged).

## Phase 3 — constrains parity + leveling cross-seed (riskiest)
- workspace.rs:610-637: second loop over
  record.package_record.constrains mirroring depends loop EXACTLY:
  split_conda_dep_line + python/python_abi-ONLY skip (NOT full
  is_abi_anchor — recording a workspace-imposed cuda-version constraint
  is INPUT parity; never-widen is EMISSION-side, enforced at its 3
  existing layers) + empty/* skip + assert_spec_roundtrips.
- Capped env in a multi-env level: ONE re-run seeded with completed
  siblings' union, FULL MAX_REFINEMENT budget, bounded to 1; caps again
  -> loud warn + classify capped. Snapshot/restore + union accumulator
  (v0.36.4); levels NOT serialized.
- Tests: constrains included; constrains_anchor_recorded_but_not_widened
  (cuda-version ==12.8 appears in map AND cascade refuses to widen —
  THE load-bearing test); build-string roundtrip;
  capped_env_reruns_once_with_full_budget_then_warns; sibling-widening
  no-leak isolation.
- Gate: + isaac6 e2e; watch for sat->unsat flips.

## Phase 4 — structural consolidation (behavior-zero)
- H1 probe.rs:115-136 -> repodata::sparse_pairs.
- H2 ProbeDecision::from_probe (14 literals).
- H3 index-chain unify + const PUBLIC_PYPI.
- M3 parse_named_spec helper (4 copies).
- M1 CondaDep{name, spec} with Display shim; Display round-trip test
  (empty-spec = bare name, no trailing space) lands BEFORE any .0
  migration; migrate highest-churn sites, rest stays behind shim.
- L2 conda-aware policy: loud warn at policy resolution (variant kept).
- L3 stale-version-comment sweep (keep load-bearing WHY comments).
- L4 verify wheel_rewrite.rs:367 allow(dead_code).
- Gate per sub-step: fmt/clippy/lib; phase end: protocol + isaac6 e2e.

## Phase 5a — config-alias comment fix (standalone trivial)
- Comment-only. Aliases KEPT (invariant #6). No e2e.

## Key distinctions to preserve (for implementer)
- Input-side recording != emission-side widening (P3).
- 0a is mechanical; 0c is semantic — never blend.
- Orchestrator + resolve_all never move (invariant #9).
- Abstention is visible, never fatal (offline best-effort contract).
