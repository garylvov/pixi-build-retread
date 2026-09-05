# `harness/` — the versioned home of the retread relock/cert harness

**This directory is the source of truth.** The working copies under
`/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/` (`tools/`,
`p6-inode-cleanup/`, `p6b-instrumented/`, `p12-staging-lever/`,
`p4l-cert-p4k/artifacts/`) are *synced from here*. Sync direction is
**task-dir ← repo**: edit here, commit, copy out. The task tree is not a git
repository at all (CLAUDE.md law 7), so anything that exists only there is one
`rm` from gone — which is exactly the hazard this directory was created to
close.

While a campaign night is running, the task-dir copy is the live copy that jobs
execute; do not delete or move it. Reconcile the two at a natural break, from
here outward.

## Layout

| path | what it is |
| --- | --- |
| `phase_template/` | `phaseN_relock.sh`, `phaseN_cert.sh`, `cleanup.sh`, `README.md` — the starting point for a new relock/cert pair |
| `tools/` | `retread_fast_env.sh`, `b2_attribute.sh`, `manifest_audit.py`, `import_audit.py`, `env_version_delta.py`, `test_stage_mirror.sh`, the merge-queue driver `land.sh`, the binsnap ancestry guard (`binsnap_ancestry_guard.sh` + its `_test.sh` + the declared `binsnap_fixset.txt`) and the task-dir drift check (`harness_drift_check.sh` + `harness_drift_allowlist.txt`) |
| `inode-cleanup/` | the stale-root sweeper (`delete_stale_roots.v2.sh`), `delete_allowlisted.sh`, the census/characterize/verify scripts and their sbatch wrappers |
| `instrumented/` | the p6b instrumented relock (`p6b_relock.sh`, `.b2.sh` variant) plus `p6b_stamp.py` / `p6b_extract.py` |
| `verdict/cert_verdict.sh` | the typed-exit verdict gate |
| `manifests/a3b/` | the A3b-family pack-manifest cessions (`patch` diffs against `pypi-packs/<pack>/pixi.toml`) — see its README for the md5s and the apply matrix |
| `arms/` | derived relock harnesses kept verbatim because later arms are derived from them by SUBSTITUTE-only edits (`c29_relock.sh`, `mh1_relock.sh`) |

## How to derive a harness

Read `phase_template/README.md` — it is the procedure, in five steps. In short:
copy `phaseN_relock.sh` / `phaseN_cert.sh` into a new batch directory (never
edit the template in place), change **only** what lies between
`### SUBSTITUTE: BEGIN` and `### SUBSTITUTE: END`, `bash -n` both, then
`DRY_RUN=1 <newbatch>_cert.sh` before you submit anything.

## The rules that are easy to get wrong

* **Leftover tokens.** A derived harness must not still name the batch it was
  copied from. Each script runs a leftover-token self-check as its first action
  and `exit 9`s on a hit; it strips the `EVIDENCE`, `SUBSTITUTE` and
  `LEFTOVER-CHECK` regions and greps everything else, comments included, against
  `LEFTOVER_RE`. Set `LEFTOVER_RE` for the new batch — this guard has caught a
  defect on nearly every port.
* **Staging mirror.** `STAGE_METHOD=mirror` is the default: a persistent
  read-only mirror under `STAGE_MIRROR_ROOT`, keyed on
  `md5(pixi.toml) + imprint-data HEAD`, is built once per key and every later
  job pays only a fanned-out `cp -al`; a missing or mis-keyed mirror is rebuilt
  and any failure falls back to the `rsync` path and says so. Never share a
  writable workspace between jobs.
* **Cleanup never on `afterok`.** An in-job `rm -rf` of a job-scoped cache root
  sits on the critical path and has cost more wall time than the lock it
  followed. The relock removes nothing; the cert prints the roots, submits
  `cleanup.sh` with `--dependency=afterany:<cert job>`, and exits. `afterany`,
  so a RED cert still returns its disk — and the roots are printed before the
  cleanup job exists, so a diagnostician can hold them by cancelling it.
* **Memory from measured demand.** Request what `/usr/bin/time -v` measured, not
  what `sacct` MaxRSS reports — MaxRSS here is ~100 % of the cgroup cap for every
  job, which is reclaimable page cache, not demand. The binding constraint is the
  per-user QOS cap, so an inflated request buys pending, not speed.
* **`CERT_PARALLEL`.** The cert env loop runs N-wide, longest-first (default 4);
  `CERT_PARALLEL=1` restores the old serial behaviour exactly for an A/B. Raise
  it only against the measured per-env peaks, since the worst case is the *sum*
  across concurrent installs under the same QOS cap.
* **Task-dir drift.** The sync above is a snapshot, and snapshots rot: C31-4-1
  found four task copies silently BEHIND this directory, one of them 779 lines
  behind, with nothing on either side detecting it. `tools/harness_drift_check.sh
  <commit>` is the link — it md5s every mapped task file against
  `git cat-file blob <commit>:<path>` (the COMMIT, never the checkout, which
  concurrent lanes are editing) and exits 3 naming the offenders.
  `tools/harness_drift_allowlist.txt` is the only escape hatch and every line in
  it needs a reason. Both phase templates call it in their header behind
  `HARNESS_COMMIT`: set that variable to the harness commit a batch is meant to
  be and a stale task copy refuses in milliseconds; leave it unset and the
  templates print that the check is OFF — never silently skipped.

## What is deliberately not here

Scripts and READMEs only. No logs, no `artifacts/`, no `.tsv` outputs, no
`.bak-*` / `.pre-*` snapshots, no job-id-stamped one-off shells — those are
evidence of a particular run, not the harness.

`MANIFEST.md5` records the md5 of every file in this directory as committed.

2026-09-03 02:22 synced FROM task dir (one-time); from here the repo is the
source and the task dir is synced FROM it before each campaign.
