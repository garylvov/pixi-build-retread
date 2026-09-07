# `arms/` — relock harnesses for named campaign arms

`phase_template/` holds the template a new harness is derived FROM. This
directory holds derived arms that are worth keeping verbatim because a later
arm is derived from THEM by SUBSTITUTE-block edits only, so losing one breaks
the derivation chain, not just a run. They are versioned here for the same
reason as `manifests/`: the task tree is not a git repository (CLAUDE.md law 7).
Boarded as C29-5 in `LANE-C-WARM-LOG.md` §29.

## The table, and what its columns MEAN

Every derivation in this campaign starts `git cat-file blob <commit>:<path>`
(see `harness/README.md`: the COMMIT, never the checkout, because the checkout
moves under you). So the useful provenance for an arm is not "the md5 of the
file today" — the file is maintained, and back-ports land in it — but **the
commit at which it was submitted, and the md5 THAT blob has**. Those two
columns are checkable against each other, and `tools/arms_readme_guard.sh` is
the reader that checks them: it re-extracts each named blob and refuses if the
md5 does not match, and `--refresh` rewrites the rows from the repo.

HARNESS-CONSOL-10, 2026-09-07: this replaces a bare `md5` column that had no
reader and had already fallen out of step with the tracked file — the recorded
`7dfa4383…` was the blob at `6cbf814`, not the file, and nothing said so. A
stamped-but-unread field is a law 2 defect whichever way it is stamped.

<!-- ARMS TABLE BEGIN -- rows maintained by tools/arms_readme_guard.sh --refresh -->
| file | as submitted (commit) | md5 of THAT blob | what it is | produced by |
| --- | --- | --- | --- | --- |
| `c29_relock.sh` | `6cbf814` | `7dfa4383fda0630e604c73b83046a457` | The C29/B5 injection-ON arm: the CANONICAL `imprint-data/pixi.toml` (md5 `9711eb990bfe211d498d1635a60e0d07`) with the a3b2 cession as the ONE pack diff and nothing else. A copy of `c28-phase1/c28_relock.sh` with edits inside `### SUBSTITUTE` only — proved, not asserted. Its `EVIDENCE BEGIN/END` header is the inherited one from `p8_warm_inject.sh`; the arm-specific provenance is the `C29 PROVENANCE (2026-09-04)` block that opens the SUBSTITUTE region. | `LANE-C-WARM-LOG.md` §29 (job 5831726, node2352, `lock rc=1 wall=529s`) |
| `mh1_relock.sh` | `975fd2f` | `b67124720f591a9246fc63ff62402d54` | The merge-lane relock every merge campaign derives from — seven lanes have taken it by `git cat-file blob <sha>:harness/arms/mh1_relock.sh`. Versioned when the five task-only files were brought under the drift check. It is MAINTAINED, not frozen: the C31-4 sdist scoper (HARNESS-CONSOL-9) and this lane's `FAST_ENV` resolution rule both landed in it AFTER the commit named here, which is exactly why the column names a commit instead of asserting a checksum of the tip. | `LANE-SPEED-LOG.md` (HARNESS-SYNC lane, 2026-09-05) |
<!-- ARMS TABLE END -->
