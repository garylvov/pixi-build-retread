# Fresh-Eyes Handoff — pixi-build-retread

Read this FIRST. Then read `HANDOFF.md` for the architecture + full
history. Do NOT skim. The previous Claude (me) has been wrong enough
times that you should treat its conclusions as hypotheses, not facts.

## Bottom line

Six retread versions (v0.34 → v0.37.1) over 36 hours. Every version
fixed a real bug. The user's actual command — `pixi s -e gsi` in
`examples/gigastrap/` — STILL fails with the same misleading leaf
the original investigation chased:

```
× failed to solve requirements of environment 'gsi-ros2' for platform 'linux-64'
╰─▶ Cannot solve the request because of: ros-humble-joint-state-publisher * cannot be installed because
    there are no viable options:
    └─ ros-humble-joint-state-publisher 2.3.0 | 2.3.0 | 2.4.0 | 2.4.0 | 2.4.0 | 2.4.0 would require
       └─ python_abi 3.9.*, for which no candidates were found.
```

retread's own solve_check (probe trace at
`examples/gigastrap/isaac-pack/retread-probe-trace-isaac-pack.json`)
reports `gsi-ros2: satisfiable=true` after 7 refinement iterations.
**This disagreement has persisted through every fix.** The pattern
is: fix one technical issue → user reruns → same shape of failure or
a closely-related sibling → fix that → same again.

If you find yourself proposing another patch-on-top, stop and
escalate. The user has explicitly said "I want to nail the root
cause."

## What's been ruled out (verified empirically against current code)

The previous Claude ran two grizzly investigations + one peer review
and shipped v0.37.0 + v0.37.1 with these fixes. **All landed and all
the unit tests pass (173 lib tests).** None resolved the user-visible
failure:

| Fix | Status | What it addressed |
|---|---|---|
| v0.34 iterative refinement | shipped, works | per-env solve-driven widening loop |
| v0.35 conflict classifier | shipped, works | A/B/C verdict categorization |
| v0.36.0 ABI anchor invariant | shipped, works | python/cuda-version never widened |
| v0.36.1 per-env isolation | shipped, works | snapshot/restore around env loop |
| v0.36.2 iteration cap 10 | shipped, works | terminate pathological cascades |
| v0.36.3 channel-priority Strict | shipped, works | match pixi default |
| v0.36.4 widening propagation | shipped, works | the iter widenings actually reach pixi |
| v0.37.0 D1 system-requirements injection | shipped, works | __cuda/__glibc come from workspace, not host |
| v0.37.0 D2 bare-major python reject | shipped, works | filter "3" out of variant_configuration |
| v0.37.0 D3a build-string preserve | shipped, works | split_conda_dep_line returns full spec |
| v0.37.0 D3b python_abi filter kept | shipped, works | filter rationale documented |
| v0.37.0 D4 clause-dedup | shipped, works | killed setuptools comma-junk |
| v0.37.0 D5 SolveStrategy::Highest pinned | shipped, works | explicit instead of default |
| v0.37.0 D6 bare-major glob ABI invariant | shipped, works | `python 3.*` flagged |
| v0.37.1 strip build-string before override map | shipped, works | parse-error from `2.10.0 cuda*_mkl*303` join |

After all this, `pixi s -e gsi` still fails the same way.

## The smoking gun the previous Claude has NOT explained

The shipped meta-v0 at
`examples/gigastrap/.pixi/meta-v0/isaac-pack-Q-8QIiWFpFg/linux_64-CICILP8QpaU.json`:

```json
"build_variants": {"python": ["3.11"]},          // pixi's view of the workspace
"outputs[0].metadata.variant": {"python": "3"},  // what retread emitted
"outputs[0].metadata.build": "py3_0",            // build string derived from variant
"outputs[0].hostDependencies": [
    {"name": "python", "binary": {"version": "3.*"}},
    ...
]
```

**pixi correctly reads `build-variants = { python = ["3.11"] }`** from the
workspace `[workspace]` table (you can see `build_variants: {python:
["3.11"]}` in the meta). **But retread emitted variant `"3"` and host-
dep `python 3.*`.** D2 (bare-major rejection in `handler.rs::pythons_for`)
was supposed to prevent this — it has unit tests pinning the rejection
of `"3"` → fall back to DEFAULT_PYTHON `"3.11"`. The unit tests pass.
The actual emission still ships `"3"`.

Possible explanations the previous Claude HAS NOT confirmed:

1. **pixi reuses a stale cached meta**. The cache is keyed on
   `project_model_hash`, `configuration_hash`, `backend_spec_hash`.
   If those didn't change when retread bumped to 0.37.1, pixi might
   not re-invoke retread at all. The meta's timestamp updated
   (20:46), but that may be a touch, not a regeneration. **TEST**:
   `rm -rf examples/gigastrap/.pixi/meta-v0/isaac-pack-*` then rerun;
   if the meta regenerates with variant "3.11", D2 is firing and the
   old cache was the issue.
2. **pixi forwards a different shape than expected**. Maybe
   `variant_configuration` arrives at retread as something other than
   `Some({"python": ["3"]})` — could be `Some({"python": ["3.11"]})`
   AND `config.python` from `isaac-pack/pixi.toml` overrides to "3".
   Check `examples/gigastrap/isaac-pack/pixi.toml` for `[package.build.config]
   python`. If `retread-python` (or equivalent) is set to "3", that's
   the source — D2 only filters the variant path; `config.python` is
   trusted verbatim.
3. **Spec-decoding subtlety**. `as_versions()` on the
   `config.python` spec may flatten "3.11" to "3" via some PEP 440
   coercion. Audit `src/config.rs::as_versions`.
4. **retread is being called fresh but `python_version` is reset
   between phases**. The outer loop sets `python_version` from
   `pythons_for`; downstream uses `target.python_version`. If a code
   path constructs a `WheelTarget` with a hardcoded major-only string
   somewhere, that path would emit "3" regardless of D2.

The previous Claude added trace logging but never actually ran retread
under `RUST_LOG=info` (or wired into pixi's verbose) to confirm
which of the above is true.

## The deeper open question

Even if you fix the variant=3 issue and retread emits `python 3.11.*`
and the meta variant becomes "3.11", the python_abi 3.9 leaf will
probably STILL fire. Here's why:

The pixi error's `python_abi 3.9.*` is the deepest twig of a rattler
unsat tree, not the cause. Conda-forge's `ros-humble-joint-state-publisher`
has:
- 2 builds at 2.3.0, both `py39` (require `python_abi 3.9.*`)
- 2 builds at 2.4.0, `py311` with `numpy 1.26` (require `python_abi 3.11.* *_cp311`)
- 2 builds at 2.4.0, `py312` with `numpy 2` (require `python_abi 3.12.* *_cp312`)

With workspace `python ==3.11`, the solver SHOULD pick the
`np126py311` 2.4.0 builds. It doesn't. Why?

Hypothesis: something else in the env forces `numpy 2`, which
eliminates the `np126` builds. Then `np2py312` is excluded by
`python ==3.11`. Then 2.3.0 py39 is excluded too. Rattler reports
the LAST attempted candidate's failure leaf — `python_abi 3.9.*` —
not the originating constraint.

What forces numpy 2? Probably retread's emitted `pytorch >=1` →
solver picks pytorch 2.10 → numpy `>=1.23,<3` → solver picks numpy 2.
Then gymnasium 1.2.x's np2 builds satisfy. Then joint-state-publisher
np126 builds DON'T satisfy. Chain fails.

retread's solve_check picks the numpy 2 path internally and says sat.
pixi might (under a subtly different constraint set or strategy) pick
the numpy 1 path and say unsat. Both are valid sat-checks of the same
SAT problem; they happen to find different witnesses.

**The structural disagreement isn't input divergence (v0.37.0 D1 made
the inputs match). It's SOLVER STATE divergence.**

## Concrete next steps a new Claude should take

### Phase 1: prove or disprove the cache hypothesis (5 min)

```bash
cd /home/garylvov/projects/pixi-build-retread/examples/gigastrap
rm -rf .pixi/meta-v0/isaac-pack-*
rm -rf isaac-pack/wheels/
pixi s -e gsi 2>&1 | tee /tmp/pixi-fresh.txt
# Then inspect:
cat .pixi/meta-v0/isaac-pack-*/linux_64-*.json | python3 -c \
  "import sys,json; d=json.load(sys.stdin); o=d['outputs'][0]; \
   print('variant:', o['metadata']['variant']); \
   print('build:', o['metadata']['build']); \
   print('python_dep:', [(i['name'], i.get('binary',{}).get('version')) for i in o['hostDependencies']['depends'] if isinstance(i, dict) and i['name']=='python'])"
```

If `variant: {'python': '3.11'}` now, D2 was always working — the prior
output was a stale cache. If `variant: {'python': '3'}` again, D2 is
genuinely not firing and you need to instrument `pythons_for`.

### Phase 2: instrument pythons_for if Phase 1 shows D2 not firing (10 min)

Edit `src/handler.rs::conda_outputs` line 352-ish:

```rust
let pythons = pythons_for(&config, params.variant_configuration.as_ref());
tracing::error!(
    variant_configuration = ?params.variant_configuration,
    config_python = ?config.python,
    pythons = ?pythons,
    "DEBUG: pythons_for inputs/output",
);
```

`error!` so it survives pixi's log filtering. Rebuild, rerun. Read
stderr in `/tmp/pixi-fresh.txt`. The output tells you exactly what's
arriving and what `pythons_for` is returning.

### Phase 3: confirm or refute the solver-state divergence hypothesis

If retread is now emitting `python 3.11.*` and pixi STILL says
gsi-ros2 unsat with the python_abi 3.9 leaf, the gap is solver-state
not input. Then:

a) Try the trivial workaround the user has refused to accept until
   the root cause is named: add `gymnasium` to `retread-drop-deps` in
   `examples/gigastrap/isaac-pack/pixi.toml`. If THIS makes the
   gsi-ros2 solve succeed, you've proven gymnasium is the bottleneck
   (numpy-2 vs numpy-1.26 conflict). The user still won't accept the
   workaround as the fix, but it confirms the diagnosis.
b) The architectural fix at that point would be in retread's
   solve_check: detect that the chosen pytorch picks numpy in a way
   that conflicts with a workspace dep, surface a Class-B workspace-
   pin suggestion ("either widen pytorch-gpu or drop gymnasium from
   the retread emission"). The conflict classifier
   (`src/conflict_classifier.rs`) has the scaffolding; what's missing
   is the cross-package conflict detector — currently it only walks
   the explicit unsat chains rattler reports.

### Phase 4: if Phase 3 confirms it, the architectural change is big

The right fix may be one of:

i) Have retread emit a LOWER pytorch (e.g. `pytorch ==2.7.*`) when
   the workspace's combined deps include something needing numpy 2.
   Requires retread to model the workspace's COMPLETE numpy graph
   pre-emit.
ii) Have retread DROP gymnasium from its emission when a workspace
    dep also constrains gymnasium. The workspace
    `[pypi-options.dependency-overrides] gymnasium = "==1.3.0"`
    already exists — retread is double-pinning via its bundled wheel's
    Requires-Dist. Auto-detect overlap and skip the conda emission.
iii) Audit every retread-emitted dep against the workspace's
     `[pypi-options.dependency-overrides]` and skip conda emission
     for anything the workspace will install via PyPI anyway.

(iii) is the most architecturally honest. The user's workspace
already declares the PyPI-side resolution; retread is fighting it.

## Things the previous Claude THINKS it knows but you should
verify before acting on

- "v0.37.0 D1 made the inputs match between retread's solve_check and
  pixi's actual solve" — UNVERIFIED. The probe trace shows `__cuda`
  now appears in blocking_deps (which is new), but that doesn't prove
  the rest of the virtual-package set is aligned. Inspect a side-by-
  side of retread's `build_virtual_packages` output vs what
  `rattler_virtual_packages::VirtualPackage::detect()` returns on the
  user's box.
- "The python_abi 3.9 leaf is misleading" — partially true but
  doesn't change the fact that the env is unsat. Don't dismiss it.
  Read the FULL rattler tree from pixi (run with `pixi s -e gsi -vvv`
  or whatever the right flag is) — there should be more context above
  the leaf showing what excluded the py3.11 builds.
- "retread's solve_check returns sat, pixi says unsat" — verified for
  THIS run. Verify it's still true after your changes.

## Don't do these

- Don't add another `retread-drop-deps` entry or `retread-overrides`
  to gigastrap's isaac-pack/pixi.toml and call it fixed. The user has
  refused this multiple times.
- Don't bump retread version + claim it works without re-running
  `pixi s -e gsi` end-to-end. Every prior "fix" passed lib tests but
  failed at the user's actual command.
- Don't write a long design document. Investigate, run things, get
  evidence, then act minimally.
- Don't commit to gigastrap. retread repo IS yours.
- Don't dispatch a third grizzly without a specific failed-prediction
  to feed it. Two grizzlies have run; both produced lucid analyses
  that hadn't yet been actually validated against the user's command.

## Reference

| Path | What |
|---|---|
| `HANDOFF.md` | Full architecture + every fix log v0.9.0 → v0.37.1 |
| `examples/gigastrap/pixi.toml` | The workspace that triggers the bug |
| `examples/gigastrap/isaac-pack/pixi.toml` | The source-package retread builds |
| `examples/gigastrap/isaac-pack/retread-probe-trace-isaac-pack.json` | Last solve_check trace (sat=true on every env) |
| `examples/gigastrap/.pixi/meta-v0/isaac-pack-*/linux_64-*.json` | What pixi sees from retread |
| `src/solve_check.rs` | The solve check + `build_virtual_packages` |
| `src/handler.rs::pythons_for` (~line 1097) | D2 bare-major filter |
| `src/handler.rs::conda_outputs` (~line 341) | The env loop |
| `src/handler.rs::iterative_solve_refinement` (~line 1838) | The widening loop |
| `src/workspace.rs` | Workspace parsing, including `effective_system_requirements` |
| `~/.cache/rattler/cache/retread-repodata/` | Cached repodata (30min TTL) |
| `scripts/rebuild-local.sh` | Nuke-rebuild-verify script |

## Standard commands

`PATH="/home/garylvov/projects/pixi/.pixi/envs/default/bin:$PATH"` is
needed for `cargo` / `rattler-build`. The user runs `pixi s -e gsi`
in `examples/gigastrap/`. Tests: `cargo test --lib` (173 passing as of
v0.37.1). Full rebuild: `bash scripts/rebuild-local.sh`
(or `CONSUMER_PROJECT=/abs/path/to/gigastrap bash scripts/rebuild-local.sh`
to also nuke the consumer's pixi caches — invariant #5 in HANDOFF.md).

Good luck. The user is patient with technical depth but tired of
band-aids. Lead with evidence.
