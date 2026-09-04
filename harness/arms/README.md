# `arms/` — relock harnesses for named campaign arms

`phase_template/` holds the template a new harness is derived FROM. This
directory holds derived arms that are worth keeping verbatim because a later
arm is derived from THEM by SUBSTITUTE-block edits only, so losing one breaks
the derivation chain, not just a run. They are versioned here for the same
reason as `manifests/`: the task tree is not a git repository (CLAUDE.md law 7).
Boarded as C29-5 in `LANE-C-WARM-LOG.md` §29.

| file | md5 | what it is | produced by |
| --- | --- | --- | --- |
| `c29_relock.sh` | `7dfa4383fda0630e604c73b83046a457` | The C29/B5 injection-ON arm: the CANONICAL `imprint-data/pixi.toml` (md5 `9711eb990bfe211d498d1635a60e0d07`) with the a3b2 cession as the ONE pack diff and nothing else. A copy of `c28-phase1/c28_relock.sh` with edits inside `### SUBSTITUTE` only — proved, not asserted. Its `EVIDENCE BEGIN/END` header is the inherited one from `p8_warm_inject.sh`; the arm-specific provenance is the `C29 PROVENANCE (2026-09-04)` block that opens the SUBSTITUTE region. | `LANE-C-WARM-LOG.md` §29 (job 5831726, node2352, `lock rc=1 wall=529s`) |
