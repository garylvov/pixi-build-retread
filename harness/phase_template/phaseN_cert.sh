#!/usr/bin/env bash
### EVIDENCE BEGIN
# phaseN_cert.sh -- TEMPLATE for the CERT half of a two-phase retread batch.
# Chains off phaseN_relock.sh via the stamp that script writes only on lock rc=0.
#
# DERIVED BY SUBSTITUTION from the phase-2 harness that ran job 5597694. Deltas:
#
#   1. PERSISTENT CACHES. Sources tools/retread_fast_env.sh and calls
#      retread_fast_env "$WS" after the job-scoped env block. An install phase
#      does not re-solve, so the 41x that job 5598763 measured on a relock is
#      NOT claimed here -- what it buys the cert is the conda/pypi PACKAGE
#      downloads, already on disk from the relock. Treat it as a download
#      saving of unmeasured size, not a solve saving.
#
#   2. CLEANUP MOVED OFF THE CRITICAL PATH ENTIRELY. This script removes
#      nothing itself. After the verdict is computed and printed it submits
#      cleanup.sh with --dependency=afterany:<relock job>:<this job> and exits
#      -- BOTH phases, and `afterany` on both, because a cleanup chained behind
#      the cert alone never runs when the RELOCK fails (Slurm cancels the
#      afterok cert, the cleanup's dependency never releases, and both roots are
#      stranded forever: C18A/C18B, job 5759225). Better still, and what the
#      launcher should do, is submit ONE gated cleanup at DISPATCH with that
#      same dependency and record its JOB ID -- as CLEANUP_JOB= in the handoff
#      stamp, or as CLEANUP_AT_DISPATCH=<that job id> here -- so this script
#      DEFERS instead of becoming a second owner of the same roots. An `rm -rf`
#      of a job-scoped root inside the job cost 5152s (86 min) on job 5596128
#      against a 3679s lock, and 4795s + 2552s on job 5598763 -- while holding
#      16 CPUs and 100-160G of a per-user QOS capped at cpu 64 / mem 492G.
#      afterany (not afterok) so a RED cert still gets its disk back; the roots
#      a failed run needs for diagnosis are named in the cleanup job's log
#      before it touches them.
#
#   3. /usr/bin/time -v PER ENV, to its own file. Each install writes
#      $A/<env>.cert-install.time.txt, and the memory ledger reads peak RSS from
#      THAT file rather than grepping it back out of the install log. Measured
#      worst cert env: 1,475,216 K (env `gpu`, job 5597694 ledger) and
#      1,446,432 K (env `gpu`, job 5594284). sacct MaxRSS is unusable here --
#      it reports ~100% of the cgroup cap for every job regardless of phase.
#
#   4. THE ENV LOOP RUNS N-WIDE, LONGEST-FIRST (CERT_PARALLEL, default 4).
#      The pre-2026-09-02 loop was `for ENV in ...; do run_env "$ENV"; done` --
#      26 installs strictly one after another. Measured on the two certified
#      pairs (INEFFICIENCY-AUDIT-20260902.md section 1.4): the sum of the 26
#      install walls is 20,477s of a 22,247s phase (92.0%, job 5597694) and
#      19,332s of 21,288s (90.8%, job 5594284), while the 16-CPU node it holds
#      runs at TotalCPU/(16*wall) = 2.1% and 1.8% and the mean per-env
#      `Percent of CPU this job got` is 18.7% of ONE core. An LPT
#      (longest-processing-time-first) schedule over those same 26 measured
#      walls gives makespans of 20,477 / 10,241 / 5,123 / 4,773s at N =
#      1 / 2 / 4 / 6. Above N=6 the makespan is pinned by isaaclab-gpu-latest
#      alone (4,773s on 5597694), so N=6 is the floor and N=4 is the first
#      honest step. The installs are I/O bound (318 GB written per cert at
#      18.7% of a core), so NFS write bandwidth, not CPU or RAM, is the real
#      ceiling -- treat the LPT table as an upper bound.
#
#      MEASURED 2026-09-03, and the LPT table is a BAD forecast. Job 5658928 ran
#      all 26 envs at N=4 against the serial key 5597694: the sum of the install
#      walls went 20,477s -> 46,233s (2.26x the WORK) for a span of 13,111s
#      against the serial loop's 20,477s -- 1.56x wall, peak concurrency 4, mean
#      3.53, and 2.56x the 5,123s the table predicted. The worst inflations are
#      the SMALLEST envs (tensorboard-tools 15s -> 1,028s = 68x; test 24s ->
#      557s; cpu 31s -> 415s), which is what a write-bandwidth ceiling looks
#      like. Quote 3h38m of env loop instead of 5h41m, never "4x". The real
#      lever is UV_LINK_MODE, not width: on env `pace` with a warm uv cache,
#      copy = 1,062s / 23.78 GB written and hardlink = 397s / 0.75 GB (2.68x,
#      31.5x). Re-measured under fan-out 2026-09-03 and shipped: the link mode
#      now defaults to `hardlink` (see the CERT_UV_LINK_MODE block below), and
#      the WIDTH stays 4 -- hardlink at N=1 (job 5685818) spent 11,144s on 6 of
#      26 envs, where hardlink at N=4 (job 5685816) finished all 26 in a 9,057s
#      span. Width and link mode are separate wins; take both. See README.md.
#      Correctness at N=4 is CLEARED: the single row that diverged in 5658928
#      (pm-isaaclab RED-verify vs AMBER-repaired) reproduces at CERT_PARALLEL=1
#      under BOTH cache sitings (jobs 5674557, 5674558), so the fan-out is not
#      its cause. It is a staged-vs-relocked workspace difference -- which also
#      means a CERT_SERIAL_KEY from a relocked workspace does not score a staged
#      one.
#
#      WHAT IS AND IS NOT SHARED ACROSS THE N SUBSHELLS.
#        per env, already:  $A/<env>.cert-{install,probe,verify,state}.log,
#                           $A/<env>.cert-install.time.txt, $WS/.pixi/envs/<env>
#        per env, NEW:      $TMPDIR ($G/tmp/<env>), and the result/ledger rows,
#                           which are written to $A/rows.$J/<env>.{row,led,span}
#                           and concatenated in DECLARATION order at the end.
#                           So cert_results.tsv is byte-comparable with what the
#                           serial loop produced, whatever order the envs ran in.
#        shared, on purpose: PIXI_CACHE_DIR / RATTLER_CACHE_DIR / UV_CACHE_DIR
#                           (persistent, retread_fast_env) and the job-scoped
#                           RETREAD_BUILD_ROOT / _ARTIFACT_ROOT / _META_ROOT /
#                           _CACHE_DIR / _SHARED_CACHE_DIR / RETREAD_FAST_TMP_ROOT.
#                           Giving each env its own package cache would multiply
#                           108 GB of reads and 318 GB of writes by N and blow
#                           the inode quota, so they stay shared and the
#                           concurrency safety is: CERT_UV_LINK_MODE (set below
#                           -- `hardlink` by default since 2026-09-03; 5547450
#                           was a BUILD race, and this phase builds nothing,
#                           which the block below shows), pixi's own
#                           multi-env installs already share one cache inside a
#                           single process, and the backend carries guard tests
#                           for exactly this (`built_output_store::tests::
#                           concurrent_publishers_of_one_key_leave_exactly_one_entry`,
#                           `wheel::tests::concurrent_store_misses_coalesce_to_one_download`).
#                           This is a LEAD, not a proof: the proof is that a
#                           parallel run scores identical to a serial one under
#                           cert_verdict.sh. Do not raise CERT_PARALLEL on a new
#                           binary without re-running that comparison.
#        NOT shared:        no `cd` anywhere in this script -- the workspace is
#                           addressed only by --manifest-path.
#      CERT_PARALLEL=1 restores the old behaviour exactly.
#
# ---- MEMORY: 32G for the cert, 24G for the relock. Measured, not quoted. ----
#   worst cert env peak RSS  1,475,216 K  (job 5597694)   = 1.48 GB / 1.41 GiB
#   worst cert env peak RSS  1,446,432 K  (job 5594284)   = 1.45 GB / 1.38 GiB
#   relock peak, for scale   8,854,172 K  (job 5597671)   = 8.85 GB / 8.44 GiB
#   Every one of those is `/usr/bin/time -v` Maximum resident set size, from the
#   memory ledgers named above. At CERT_PARALLEL=6 the worst case is a SUM of
#   the six largest envs, and on the 5597694 ledger that is 1,475,216 +
#   1,355,544 + 1,343,420 + 1,333,200 + 1,221,588 + 1,034,344 = 7,763,312 K =
#   7.8 GB. 32G is >4x that, and leaves ~24 GB of page cache for a phase that
#   reads 108 GB and writes 318 GB. 24G on the relock is ~2.7x its 8.85 GB peak.
#   NEVER size either from sacct MaxRSS: it reports ~ReqMem for every job here
#   (a 160G job reports 167,772,480 K, a 100G job 104,858,688 K) because that is
#   reclaimable cgroup page cache, not demand. The binding constraint is the
#   per-user QOS cap (normal = cpu 64, mem 492G for the WHOLE user): at 160G we
#   fit 3 jobs, at 32G we fit ~15, and a 160G request is why job 5597889 pended
#   behind QOSMaxMemoryPerUser with a node sitting idle. Raise it only against a
#   /usr/bin/time -v row.
#
#       env -u SLURM_JOB_ID sbatch --partition=batch --qos=normal \
#           --cpus-per-task=16 --mem=32G --time=04:00:00 \
#           --job-name=<tag>-p2 --dependency=afterok:<relock job> \
#           --output=<shared path>/slurm-%j.out ./phaseN_cert.sh
#
#
# ---- EXACTLY ONE CLEANUP OWNER PER ROOT (measured 2026-09-04, tag AFINAL2) ---
#   The roots /oscar/data/stellex/glvov/retread/certAFINAL2-5769426 and
#   ws.AFINAL2-5769426 were handed to TWO cleanup jobs at once: the
#   dispatch-time gated cleanup 5770508, submitted by the launcher on
#   `--dependency=afterany:<p1>:<p2>`, and this script's OWN self-submitted
#   cleanup 5776646. Both released on the same dependency, both started
#   08:39:44 on node2343, and both walked the same two trees.
#   Two concurrent `rm -rf` walks of one tree unlink entries out from under
#   each other, so each one's rmdir of a parent finds children it cannot see.
#   BOTH returned rc=1 with pages of "Directory not empty"; 5776646 also hit
#   `rm: fts_read failed: Stale file handle`, which only a second walker can
#   produce. Both logged `exists_after=YES` for both roots. 590028 + 668715
#   entries were LEFT ON DISK after 2864 s and 3941 s of wall, and both jobs
#   still reported `CLEANUP DONE rc=0`.
#   THE RULE: one owner per root, chosen from a fact RECORDED ON DISK, and
#   named in the log by both branches. phaseN_relock.sh writes `CLEANUP_JOB=`
#   into the handoff stamp; `cleanup_owner` here reads it and returns
#   "dispatch <id>" or "self -", and `cleanup_submit_or_defer` either submits
#   exactly one cleanup or submits nothing and says whose job it defers to.
#   Guarded by phase_template/cleanup_owner_guard.sh.
# NEVER edit this file while a job is running it -- copy it aside first.
### EVIDENCE END
set -uo pipefail

### SUBSTITUTE: BEGIN -- MANIFEST, PROBES, EXPECT_*  (edit ONLY between these markers)
TAG=PHASEN                                   # MUST match the relock harness's TAG
T=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11
D=$T/phase-template-example                  # THIS harness's own directory
P1D=$T/phase-template-example                # the RELOCK harness's directory (stamp + arm probes)
                                             # set it to the relock dir when the two phases are split

# --- gates carried over from the relock phase --------------------------------
EXPECT_MANIFEST_LINES=1003
EXPECT_JETSON_ROWS=1
RESIDUAL_PATTERNS=()                         # same list the relock phase used; each must be 0
DELETED_FAMILIES="openmesh networkx pillow sentry-sdk numpy"   # informational lock occurrence rows

# --- probes ------------------------------------------------------------------
PROBES_CANON=$T/p1e-certify-lock/artifacts/probes.tsv
PROBES=$PROBES_CANON                         # the arm's copy -- MUST be the same file the relock gated
PROBE_TOKENS=()                              # module tokens that must be GONE from $PROBES
EXPECT_PROBE_ROWS=26

# --- every env must be probeable --------------------------------------------
# A5a (2026-09-03). An env whose name has no row in $PROBES gets an EMPTY $MODS,
# so both the TierA and the TierB probe are skipped. It was NOT silently green
# -- ARC stays at its 99 initialiser and the row scored RED-tierA -- but that
# label is a LIE: TierA never ran. Two changes, neither behind a flag:
#   1. this gate refuses the run before any install when an env has no row;
#   2. run_env scores such an env RED-probes-missing, an explicit state the
#      verdict gate reads as OUTCOME_DIFF (verdict is field 8, scored).
# REQUIRE_PROBE_ROW_PER_ENV=0 disables ONLY gate 1, so a guard run can reach
# state 2. Nothing else may set it.
REQUIRE_PROBE_ROW_PER_ENV=${REQUIRE_PROBE_ROW_PER_ENV:-1}

# --- the environments to certify ---------------------------------------------
# jetson is aarch64-only and is deliberately excluded; any manifest line that
# only binds jetson is therefore CERT-BLIND and relock rc=0 is its only evidence.
CERT_ENVS="pm-newton-gpu sage default cpu test tensorboard-tools gpu test-gpu \
unitree-rl-gym holosoma unitree-rl-lab-gpu groot-sonic-gpu viral-gpu uwlab-gpu \
flashsac-gpu hover-gpu robogen ros2-humble-gpu ros2-humble-cpu ros2-jazzy-gpu \
ros2-jazzy-cpu isaaclab-gpu-latest newton-gpu pm-isaaclab pm-mujoco pace"

# --- how wide to run the env loop, and in what order --------------------------
# CERT_PARALLEL   how many installs run at once. 1 = the pre-2026-09-02 serial
#                 loop, exactly. 4 is the measured first step; 6 is the LPT
#                 floor for this env set. Overridable from the environment so an
#                 A/B needs no edit:  CERT_PARALLEL=1 ./<harness>.sh
# CERT_WALL_TABLE a memory ledger from the last GREEN cert of this shape --
#                 columns env, install_rc, peak_rss_kb, wall_s. Used only to
#                 order the run longest-first (LPT), never to score anything.
#                 Empty or unreadable -> declaration order, and the schedule is
#                 then whatever the declaration happens to be, which on this env
#                 set costs ~1,000s against LPT. An env absent from the table
#                 sorts FIRST (unknown is treated as long, the safe side).
CERT_PARALLEL=${CERT_PARALLEL:-4}

# --- how uv puts wheels into the env prefixes -------------------------------
# CERT_UV_LINK_MODE  hardlink (default) | copy. This is exported AFTER
#                 retread_fast_env, which sets UV_LINK_MODE=copy and used to be
#                 the last word on it, so an A/B needs no edit:
#                   CERT_UV_LINK_MODE=copy ./<harness>.sh
#
# WHAT `copy` IS INSURANCE AGAINST, precisely. Job 5547450 died on
# "failed to hardlink file from <uv cache>/builds-v0/.tmp4QiHoL/lib/python3.11/
# site-packages/... to <uv cache>/archive-v0/...: No such file or directory"
# while uv-BUILDING the gym sdist with six concurrent builds: the tree that
# vanished was `builds-v0/.tmpXXXX`, a per-build ephemeral build environment a
# sibling reclaimed mid-link. uv documents UV_LINK_MODE as "the method to use
# when installing packages FROM THE GLOBAL CACHE" (uv 0.12.5, `uv help pip
# install`), and a cert installs a frozen lock: its source is `archive-v0`,
# content-addressed, and NOTHING in this campaign runs `uv cache clean|prune`
# (zero call sites in tools/ or any harness). The persistent cache's own
# buckets say the same -- archive-v0 4796 entries, builds-v0 and sdists-v9
# EMPTY. So the cert's install hardlinks cannot point at a reclaimable tree.
# The RELOCK is the phase that builds, and it keeps `copy` for that reason.
#
# MEASURED at fan-out, job 5685816 (26 envs, CERT_PARALLEL=4, hardlink, staged
# workspace, persistent caches) against 5658928 (same everything, copy):
#   env-loop span 13,111s -> 9,057s (-31%), sum of env walls 46,233s -> 32,421s,
#   writes 301.39 GB -> 128.03 GB by /usr/bin/time (302 GB -> 131.8 GB by sacct),
#   `pace` prefix 156,275 of 161,609 files at st_nlink>1 (it really hardlinked),
#   ZERO hardlink/EXDEV/busy/builds-v0 lines in all 26 install logs, and
#   cert_verdict.sh scores the two runs IDENTICAL: 26 rows, 0 differing, EXIT 0.
# The default is `hardlink` as of 2026-09-03 (C7 closeout), because the three
# things held against it were then measured and none survived:
#   * "two envs got ~2x SLOWER under hardlink" -- they did not. That was WIDTH,
#     not link mode. At CERT_PARALLEL=1 with hardlink (job 5685818) the two
#     accused envs are the FASTEST of all four corners: `pace` 553s (copy N=1
#     1,216s same node/day, copy N=4 1,290s, hardlink N=4 2,734s) and
#     `groot-sonic-gpu` 1,424s (1,529s / 1,779s / 3,440s). Hardlink turns an
#     install from a write into a refcount bump, so it gives back the most where
#     it is alone on the write path and the least at the N=4 bandwidth ceiling.
#   * "the hardlink N=1 corner is unmeasured" -- it is measured now. Every env
#     it has scored is IDENTICAL to copy at the same width on the same node the
#     same day (cert_verdict.sh, 0 differing), install_rc=0 throughout, and zero
#     hardlink/EXDEV/busy/builds-v0 lines in any install log at either width.
#   * "hardlinking couples every prefix to the SHARED persistent uv cache" -- it
#     does, and the coupling was checked rather than argued. Across the hardlink
#     certs, of 13,920 files in 60 archive-v0 entries that pre-date them, the
#     number whose CONTENT changed is 0; 12,557 had a ctime bump, which is the
#     link/unlink refcount churn, and every one is back at st_nlink=1 after the
#     prefixes were reclaimed. uv installs by unlink-and-create, so uv itself
#     cannot write through; a writer that rewrote a prefix file IN PLACE still
#     could, which is why that census is worth re-running when one appears.
# Use `copy` for a phase that BUILDS. This one does not -- see above.
CERT_UV_LINK_MODE=${CERT_UV_LINK_MODE:-hardlink}
CERT_WALL_TABLE=                             # e.g. $T/<lastgreen>/artifacts/memory_ledger.<job>.tsv

# --- optional SECOND verdict gate --------------------------------------------
# A results file from a previous run of THIS SAME LOCK, when one exists (an
# A/B of the harness itself, a re-cert of a lock already certified). Scored by
# the same cert_verdict.sh, and its exit is ANDed into the job's exit. Empty =
# the campaign baseline is the only gate, which is the normal case for a new
# batch whose lock has never been certified before.
CERT_SERIAL_KEY=

# --- toolchain + verdict gate ------------------------------------------------
PIXI=/users/glvov/.pixi/bin/pixi.real
SNAPDIR=$T/p4l-cert-p4k/artifacts/p4k-binsnap
SNAP=$SNAPDIR/pixi-build-retread
# OPTIONAL pin. Leave EMPTY and the gate DERIVES the sha from $SNAP at run
# time. Set it only to assert a specific binary, and then it MUST match.
# See the same note in phaseN_relock.sh: SNAP and a hand-written EXPECT_SHA
# were two coupled constants inside the SUBSTITUTE region, the leftover-token
# check strips that region by design, and a derivation that moved one and not
# the other killed job 5671529 exit 8 in 3 s. One constant cannot disagree
# with itself.
EXPECT_SHA_PIN=
UVBIN=/oscar/data/stellex/glvov/tasks/retread-cold-solve/verify_fixes/artifacts/uvbin
VERDICT=$T/p4l-cert-p4k/artifacts/cert_verdict.sh
CERT_BASELINE=$T/p4l-cert-p4k/artifacts/cert_results.5175534.att_corrected.tsv
FAST_ENV=$(dirname "$0")/../retread_fast_env.sh
[ -f "$FAST_ENV" ] || FAST_ENV=$T/tools/retread_fast_env.sh
CLEANUP=$(dirname "$0")/cleanup.sh
[ -f "$CLEANUP" ] || CLEANUP=$T/tools/phase_template/cleanup.sh
CLEANUP_SBATCH_ARGS="--partition=batch --qos=normal --cpus-per-task=1 --mem=4G --time=06:00:00"

# --- leftover-token self-check ------------------------------------------------
LEFTOVER_RE='bfinal|BFP1|BFP2|bfp1|bfp2|b1c|b1-phase|b1b-phase|b2-phase|b2b-phase|b3-phase|ctl-phase|eff-phase|/b1_|/b2_|/b3_|/ctl_|p5sab|P5SAB|p5t_abc|P5TABC|certB3P1|2cfec88d|57105d38'
### SUBSTITUTE: END

### LEFTOVER-CHECK BEGIN
# The match runs INSIDE awk, on the LINE, never on "FILENAME:LNO: line". Piping
# the annotated text to grep made the check match its own FILENAME: a harness in
# a directory named after a previous batch failed against itself, on every line,
# with the tokens nowhere in its body. A scan must not be able to match itself.
LEFT=$(awk '
  /^### EVIDENCE BEGIN/       {e=1} /^### EVIDENCE END/       {e=0; next} e {next}
  /^### SUBSTITUTE: BEGIN/    {s=1} /^### SUBSTITUTE: END/    {s=0; next} s {next}
  /^### LEFTOVER-CHECK BEGIN/ {l=1} /^### LEFTOVER-CHECK END/ {l=0; next} l {next}
  $0 ~ re {print FILENAME ":" FNR ": " $0}' re="$LEFTOVER_RE" "$0")
if [ -n "$LEFT" ]; then
  echo "### FATAL leftover-token self-check FAILED -- this harness still names a previous batch"
  printf '%s\n' "$LEFT"
  exit 9
fi
echo "### leftover-token self-check: clean"
### LEFTOVER-CHECK END

########## EXACTLY ONE CLEANUP OWNER PER ROOT ##########
# Two cleanup jobs were once handed the same two job-scoped roots and their
# `rm -rf` walks raced: both returned rc=1 "Directory not empty" and both roots
# were left on disk. The incident, with job ids and counts, is in the EVIDENCE
# header above. These two functions are the fix, and they decide from a RECORDED
# FACT -- the CLEANUP_JOB= line phaseN_relock.sh writes into the handoff stamp --
# rather than from anything about this job's environment.
#
# Both branches PRINT WHO OWNS THE ROOTS. A reader of the log never has to infer
# whether a cleanup exists.

cleanup_owner () {
  # Echoes two words: "<who> <jobid>". `dispatch <id>` means a cleanup job
  # already exists and owns these roots, so this cert job must submit nothing.
  # `self -` means nobody owns them yet and this cert job is the owner.
  #
  # Sources, in order:
  #   1. CLEANUP_JOB, from the phase-1 -> phase-2 handoff stamp
  #      ($P1D/artifacts/relock_env.sh), sourced below. The launcher submits
  #      the gated cleanup after both phases are queued, so the relock phase
  #      records the id there and it is on disk before this job starts.
  #   2. CLEANUP_AT_DISPATCH in this job's environment, for the operator who
  #      sets it directly. Set it to the cleanup's JOB ID. The legacy value
  #      `1` still defers, but can only print `unrecorded-id` -- a worse log
  #      line, and the reason the stamp is the preferred channel.
  # Unset, empty, 0 or `none` in both means NO dispatch cleanup is recorded.
  local id=""
  case "${CLEANUP_JOB:-}" in
    ''|0|none|NONE|unset) ;;
    *) id=$CLEANUP_JOB ;;
  esac
  if [ -z "$id" ]; then
    case "${CLEANUP_AT_DISPATCH:-0}" in
      ''|0|none|NONE) ;;
      1) id=unrecorded-id ;;
      *) id=$CLEANUP_AT_DISPATCH ;;
    esac
  fi
  if [ -n "$id" ]; then echo "dispatch $id"; else echo "self -"; fi
}

cleanup_submit_or_defer () {   # $1=dependency spec  $2..=roots
  local dep=$1; shift
  local owner who id clj clrc
  owner=$(cleanup_owner); who=${owner%% *}; id=${owner#* }
  if [ "$who" = dispatch ]; then
    echo "### CLEANUP OWNER: job $id (submitted at dispatch) -- roots: $*"
    echo "### cleanup NOT submitted here: exactly one cleanup owner per root; this cert job ${J:-?} defers to job $id"
    return 0
  fi
  if [ ! -x "$CLEANUP" ]; then
    echo "### CLEANUP OWNER: NOBODY -- $CLEANUP is missing or not executable; roots left on disk, clean them by hand:"
    echo "    $*"
    return 1
  fi
  # shellcheck disable=SC2086
  clj=$(env -u SLURM_JOB_ID sbatch --parsable $CLEANUP_SBATCH_ARGS \
        --job-name=${TAG}-cleanup --dependency=$dep \
        --output=$A/slurm-cleanup-%j.out "$CLEANUP" "$@" 2>&1); clrc=$?
  if [ "$clrc" = 0 ]; then
    echo "### CLEANUP OWNER: job $clj (submitted by this cert job ${J:-?}; no cleanup was recorded at dispatch) -- roots: $*"
    echo "### cleanup job submitted: $clj (--dependency=$dep) -- this job exits WITHOUT unlinking anything"
  else
    echo "### CLEANUP OWNER: NOBODY -- submit failed rc=$clrc output: $clj"
    echo "### RUN THIS BY HAND, it is the only thing that returns the inodes:"
    echo "    env -u SLURM_JOB_ID sbatch $CLEANUP_SBATCH_ARGS --job-name=${TAG}-cleanup $CLEANUP $*"
  fi
  return 0
}

J=${SLURM_JOB_ID:?missing Slurm job id}
A=$D/artifacts
STAMP=$P1D/artifacts/relock_env.sh
[ -f "$STAMP" ] || { echo "### FATAL relock handoff stamp missing: $STAMP (the relock phase did not reach lock rc=0)"; exit 2; }
# shellcheck disable=SC1090
. "$STAMP"
: "${WS:?stamp did not define WS}"; : "${LOCK:?stamp did not define LOCK}"
: "${EXPECT_LOCK_MD5:?stamp did not define EXPECT_LOCK_MD5}"
echo "### relock handoff: P1_JOB=${P1_JOB:-?} WS=$WS LOCK=$LOCK md5=$EXPECT_LOCK_MD5 P1_CACHE_ROOT=${P1_CACHE_ROOT:-unset}"
C=/oscar/data/stellex/glvov/retread/cert${TAG}P2-$J
RES=$A/cert_results.$J.tsv
COMPARE=$A/cert_results_vs_baseline.$J.tsv
LEDGER=$A/memory_ledger.$J.tsv
BLOG=$A/${TAG}P2-$J.backend.log

CQ=/oscar/runtime/bin/checkquota          # NOT on a batch job's default PATH: job 5611846 printed
[ -x "$CQ" ] || CQ=$(command -v checkquota 2>/dev/null || echo true)   # two EMPTY quota rows because of it
mkdir -p "$A"
exec > >(tee -a "$A/cert_${TAG}.$J.log") 2>&1
hostname; date -Is
echo "### ${TAG} CERT job=$J  envs=$(echo $CERT_ENVS | wc -w)  CERT_PARALLEL=$CERT_PARALLEL"
echo "### inode quota BEFORE:"; "$CQ" 2>/dev/null | grep -E 'data\+stellex|^Name' | head -4
printf '' > "$RES"
printf 'env\tinstall_rc\tpeak_rss_kb\twall_s\n' > "$LEDGER"

# ---- cheap gates BEFORE any install work ----
for f in "$SNAP" "$LOCK" "$PROBES" "$PROBES_CANON" "$VERDICT" "$CERT_BASELINE" "$FAST_ENV" "$WS/pixi.toml" "$WS/pixi.lock"; do
  [ -e "$f" ] || { echo "### FATAL missing required path: $f"; exit 2; }
done
GOT_SHA=$(sha256sum "$SNAP" | awk '{print $1}')
[ -n "$GOT_SHA" ] || { echo "### FATAL could not sha256sum $SNAP"; exit 2; }
if [ -n "$EXPECT_SHA_PIN" ]; then
  [ "$GOT_SHA" = "$EXPECT_SHA_PIN" ] || { echo "### FATAL backend snapshot sha mismatch got=$GOT_SHA want-pinned=$EXPECT_SHA_PIN"; exit 2; }
  echo "### backend snapshot sha PINNED and matched: $GOT_SHA"
else
  echo "### backend snapshot sha DERIVED from \$SNAP at run time: $GOT_SHA (no pin set)"
fi
EXPECT_SHA=$GOT_SHA
GOT_MD5=$(md5sum "$LOCK" | awk '{print $1}')
[ "$GOT_MD5" = "$EXPECT_LOCK_MD5" ] || { echo "### FATAL lock md5 mismatch got=$GOT_MD5 want=$EXPECT_LOCK_MD5"; exit 2; }
echo "### gates OK: snapshot sha + lock md5 + all required paths"
if [ "${#RESIDUAL_PATTERNS[@]}" -gt 0 ]; then
  echo "### manifest residual pin rows (want 0 each):"
  for pat in "${RESIDUAL_PATTERNS[@]}"; do printf '  %-40s %s\n' "$pat" "$(grep -c "$pat" "$WS/pixi.toml")"; done
fi
echo "### manifest lines (want $EXPECT_MANIFEST_LINES): $(wc -l < "$WS/pixi.toml")"
echo "### jetson LIVE rows (want $EXPECT_JETSON_ROWS): $(grep -c '^jetson = ' "$WS/pixi.toml")"
echo "### probes file IN USE: $PROBES"
echo "### probes rows (want $EXPECT_PROBE_ROWS): $(wc -l < "$PROBES")"
if [ "${#PROBE_TOKENS[@]}" -gt 0 ]; then
  for tok in "${PROBE_TOKENS[@]}"; do
    printf '  probe token %-20s arm=%s canonical=%s (arm must be 0)\n' "$tok" "$(grep -c "$tok" "$PROBES")" "$(grep -c "$tok" "$PROBES_CANON")"
  done
fi
echo "### lock occurrences for the deleted families (informational -- a family may legitimately survive at a different version):"
for p in $DELETED_FAMILIES; do
  printf '  %-12s %s   versions: %s\n' "$p" \
    "$(grep -cE "/${p}-[0-9]|name: ${p}$" "$LOCK")" \
    "$(grep -oE "/${p}-[0-9][^/]*" "$LOCK" | sort -u | tr '\n' ' ')"
done
PROBED_ENVS=0; UNPROBED=
for E in $CERT_ENVS; do
  if [ -n "$(awk -F'\t' -v e="$E" '$1==e{print $2}' "$PROBES")" ]; then
    PROBED_ENVS=$((PROBED_ENVS+1))
  else
    UNPROBED="$UNPROBED $E"
  fi
done
echo "### probed envs: $PROBED_ENVS of $(echo $CERT_ENVS | wc -w) (every env in CERT_ENVS must have a $PROBES row)"
if [ -n "$UNPROBED" ]; then
  for E in $UNPROBED; do echo "### probes row missing for $E"; done
  if [ "$REQUIRE_PROBE_ROW_PER_ENV" = 1 ]; then
    echo "### FATAL envs with no probes row cannot be certified:$UNPROBED"
    echo "###   add a row to $PROBES, or drop the env from CERT_ENVS. Running anyway"
    echo "###   would score each of them RED-probes-missing."
    exit 2
  fi
  echo "### REQUIRE_PROBE_ROW_PER_ENV=0 -- continuing; each unprobed env will score RED-probes-missing"
fi
echo "### baseline rows: $(wc -l < "$CERT_BASELINE")"

collect_artifacts() {
  set +e
  echo "### unconditional artifact collection $(date -Is)"
  [ -f "$SNAP" ] && sha256sum "$SNAP" > "$A/snapshot_sha256.$J.txt"
  [ -f "$LOCK" ] && sha256sum "$LOCK" > "$A/lock_sha256.$J.txt"
  [ -f "$RES" ] && sha256sum "$RES" > "$A/results_sha256.$J.txt"
  [ -f "$BLOG" ] && sed -r 's/\x1B\[[0-?]*[ -\/]*[@-~]//g' "$BLOG" > "$A/backend.$J.ansi_stripped.log"
  grep 'bench: conda_outputs total' "$A/backend.$J.ansi_stripped.log" > "$A/conda_outputs.$J.tsv" 2>/dev/null || true
  if [ -f "$RES" ]; then
    : > "$COMPARE"
    for baseline in "$CERT_BASELINE"; do
      printf 'baseline\t%s\n' "$baseline" >> "$COMPARE"
      awk -F'\t' 'NR==FNR { b[$1]=$0; next }
        { seen[$1]=1; if (!($1 in b)) { print $1 "\tNEW_ROW\tcurrent=" $0; next }
          n=split(b[$1],x,"\t"); status=(($2==x[2] && $4==x[4] && $5==x[5] && $6==x[6] && $7==x[7] && $8==x[8]) ? "MATCH_OUTCOME" : "OUTCOME_DIFF");
          if ($1=="hover-gpu" && $4==1 && x[4]==1) status=status "_KNOWN_HOVER_TIERA";
          print $1 "\t" status "\tcurrent=" $0 "\tbaseline=" b[$1] }
        END { for (e in b) if (!seen[e]) print e "\tNOT_RUN\tbaseline=" b[e] }' "$baseline" "$RES" | sort >> "$COMPARE"
    done
  fi
  sstat -j "$J.batch" --format=JobID,AveCPU,MaxRSS,MaxDiskRead,MaxDiskWrite > "$A/sstat_final.$J.txt" 2>&1 || true
}
trap collect_artifacts EXIT

cp "$LOCK" "$WS/pixi.lock"

########## ENV BLOCK -- job-scoped BUILD state, SHARED download+solve caches ##########
G=$C/install
for d in pixi rattler uv xdg-cache xdg-data retread-build retread-artifacts retread-meta retread-cache retread-shared pixi-home; do mkdir -p "$C/$d"; done
for d in home tmp scratch fast-tmp xdg-state xdg-config; do mkdir -p "$G/$d"; done
export PIXI_CACHE_DIR=$C/pixi RATTLER_CACHE_DIR=$C/rattler UV_CACHE_DIR=$C/uv PIXI_HOME=$C/pixi-home
export XDG_CACHE_HOME=$C/xdg-cache XDG_DATA_HOME=$C/xdg-data XDG_STATE_HOME=$G/xdg-state XDG_CONFIG_HOME=$G/xdg-config
export RETREAD_BUILD_ROOT=$C/retread-build RETREAD_ARTIFACT_ROOT=$C/retread-artifacts RETREAD_META_ROOT=$C/retread-meta RETREAD_CACHE_DIR=$C/retread-cache RETREAD_SHARED_CACHE_DIR=$C/retread-shared
export HOME=$G/home TMPDIR=$G/tmp RETREAD_SCRATCH_ROOT=$G/scratch RETREAD_FAST_TMP_ROOT=$G/fast-tmp
export PATH=/users/glvov/.pixi/bin:$UVBIN:/users/glvov/.local/bin:/usr/bin:/bin
export RETREAD_UV=$UVBIN/uv CONDA_OVERRIDE_CUDA=12 CONDA_OVERRIDE_GLIBC=2.35 UV_LOCK_TIMEOUT=3600 RETREAD_MAX_CONCURRENT_BUILDS=6 TOKIO_WORKER_THREADS=8 RAYON_NUM_THREADS=8
export UV_LINK_MODE=copy   # provisional; CERT_UV_LINK_MODE decides, after retread_fast_env
export PIXI_BUILD_RETREAD_LOG=pixi_build_retread=debug,warn RUST_BACKTRACE=1 PIXI_BUILD_BACKEND_OVERRIDE="pixi-build-retread=$SNAP"
unset RUST_LOG

# PERSISTENT CACHES -- after the job-scoped block (it overrides the three cache
# dirs) and after RETREAD_FAST_TMP_ROOT + SLURM_JOB_ID exist.
# shellcheck source=/dev/null
. "$FAST_ENV"
retread_fast_env "$WS" || { echo "### FATAL retread_fast_env refused"; exit 2; }

# LINK MODE -- last word, because retread_fast_env exports UV_LINK_MODE=copy.
case "$CERT_UV_LINK_MODE" in
  copy|hardlink) export UV_LINK_MODE=$CERT_UV_LINK_MODE ;;
  *) echo "### FATAL CERT_UV_LINK_MODE must be copy|hardlink, got '$CERT_UV_LINK_MODE'"; exit 2 ;;
esac

# --- C31-4: JOB-SCOPED sdist BUILD TREES, AND THE GUARD THAT READS THEM -------
# This block is why B-cert-4's `pm-newton-gpu` row measured a stale entry in a
# shared cache instead of the artefact under certification. The comment at the
# top of this file says the shared download caches "carry no resolution, only
# bytes keyed by url and hash" -- TRUE for `archive-v0`/`wheels-v6`/`simple-v*`,
# FALSE for `sdists-v9`, where uv builds a source distribution in place and a
# cmake project leaves a `CMakeCache.txt` naming the ABSOLUTE compiler paths of
# whichever workspace built it first. LANE-C-WARM-LOG 31.10-31.12 and 33.
# The build halves become job-local; the byte-keyed halves stay shared.
retread_scope_sdist_builds "$C" || { echo "### FATAL retread_scope_sdist_builds refused"; exit 2; }

# ITS READER, BEFORE THE FIRST INSTALL. Four hours of cert to reach a RED whose
# whole content was one absolute path is what this turns into a header refusal.
SDIST_GUARD=$(dirname "$FAST_ENV")/sdist_build_poison_guard.sh
[ -f "$SDIST_GUARD" ] || { echo "### FATAL sdist_build_poison_guard.sh missing next to $FAST_ENV"; exit 2; }
bash "$SDIST_GUARD" || { echo "### FATAL sdist build poison guard refused -- see the rows above"; exit 2; }
echo "### UV_LINK_MODE=$UV_LINK_MODE (CERT_UV_LINK_MODE=$CERT_UV_LINK_MODE) UV_CACHE_DIR=$UV_CACHE_DIR"

# Every env writes its OWN row files here; the driver concatenates them in
# DECLARATION order after the last one finishes. Nothing appends to a shared
# file while installs are in flight, so the parallel loop cannot interleave a
# row and cert_results.tsv stays byte-comparable with the serial format.
ROWD=$A/rows.$J
mkdir -p "$ROWD"

run_env() {
  ENV=$1
  ILOG=$A/$ENV.cert-install.log; PLOG=$A/$ENV.cert-probe.log; VLOG=$A/$ENV.cert-verify.log; SLOG=$A/$ENV.cert-state.log
  ITIME=$A/$ENV.cert-install.time.txt
  : > "$ILOG"; : > "$PLOG"; : > "$VLOG"; : > "$SLOG"; : > "$ITIME"
  # Own TMPDIR per env: the only env var this loop changes for parallelism.
  # $G/tmp was shared by all 26 serial installs; under fan-out a shared TMPDIR
  # is the classic collision site, and a per-env one costs nothing.
  export TMPDIR=$G/tmp/$ENV; mkdir -p "$TMPDIR"
  S=$(date +%s)
  timeout --foreground --kill-after=30s 25200s \
    /usr/bin/time -v -o "$ITIME" "$PIXI" install --manifest-path "$WS/pixi.toml" -e "$ENV" --frozen > "$ILOG" 2>&1
  IRC=$?; W=$(( $(date +%s)-S )); PREFIX=$WS/.pixi/envs/$ENV; SHR=$PREFIX/share/retread
  RSS=$(awk -F': ' '/Maximum resident set size/{print $2}' "$ITIME")
  { echo "env=$ENV install_rc=$IRC wall=${W}s peak_rss_kb=${RSS:-unknown}"; find "$SHR" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sort; } > "$SLOG" 2>&1
  BROKEN=$(find "$SHR" -maxdepth 1 -name '*.broken' -type f 2>/dev/null | wc -l); STATEF=$(find "$SHR" -maxdepth 1 -name '*.state' -type f 2>/dev/null | wc -l); INSTM=$(find "$SHR" -maxdepth 1 -name '*.installed' -type f 2>/dev/null | wc -l)
  ARC=99; BRC=99; VRC=99
  MODS=$(awk -F'\t' -v e="$ENV" '$1==e{print $2}' "$PROBES"); TIERB=$(awk -F'\t' -v e="$ENV" '$1==e{print $3}' "$PROBES")
  if [ "$IRC" = 0 ] && [ -n "$MODS" ]; then
    timeout --foreground --kill-after=15s 2400s "$PIXI" run --manifest-path "$WS/pixi.toml" -e "$ENV" --frozen python -c 'import importlib.util as u,sys; bad=[n for n in sys.argv[1].split() if u.find_spec(n) is None]; print("MISSING:",bad); sys.exit(1 if bad else 0)' "$MODS" >> "$PLOG" 2>&1; ARC=$?
    timeout --foreground --kill-after=15s 2400s "$PIXI" run --manifest-path "$WS/pixi.toml" -e "$ENV" --frozen python -c "$TIERB
print('TIERB_OK')" >> "$PLOG" 2>&1; BRC=$?
    RB=$PREFIX/bin/retread; VRC=0; SEEN=0
    if [ -x "$RB" ]; then for LJ in "$SHR"/retread-*.lock.json; do [ -e "$LJ" ] || continue; SEEN=$((SEEN+1)); timeout --foreground --kill-after=15s 2400s "$RB" verify --lock "$LJ" --prefix "$PREFIX" >> "$VLOG" 2>&1; R=$?; [ "$R" = 0 ] || VRC=$R; done; [ "$SEEN" = 0 ] && VRC=97; else [ "$INSTM" = 0 ] || VRC=98; fi
  else echo "SKIP install_rc=$IRC probes=$([ -n "$MODS" ] && echo present || echo missing)" >> "$PLOG"; fi
  if [ -z "$MODS" ]; then
    echo "### probes row missing for $ENV -- TierA and TierB never ran; this env is NOT certified"
    echo "PROBES-ROW-MISSING env=$ENV probes=$PROBES" >> "$PLOG"
  fi
  # Count repair ATTEMPTS, not lines mentioning one: the section header
  # '=== attempt N <ts> ===' is emitted exactly once per attempt, while the
  # older pattern also matched the inline '(repair attempt #N)' and so read
  # roughly DOUBLE. Verified on job 5168525: 2 matches, 1 actual attempt.
  ATT=$(find "$SHR" -maxdepth 1 -name '*.repair.log' -type f -exec cat {} + 2>/dev/null | grep -cE '^=== attempt [0-9]+ ' || true)
  # A successful repair DELETES the evidence for why it fired, so capture the
  # marker CONTENTS while they may still exist.
  for M in "$SHR"/*.broken "$SHR"/*.state; do
    [ -e "$M" ] || continue
    { echo "--- marker $(basename "$M") ---"; cat "$M"; } >> "$SLOG" 2>&1
  done
  find "$SHR" -maxdepth 1 -name '*.repair.log' -type f -exec sed -n '1,20p' {} + >> "$SLOG" 2>/dev/null || true
  if [ "$IRC" != 0 ]; then V=RED-install; elif [ -z "$MODS" ]; then V=RED-probes-missing; elif [ "$BROKEN" != 0 ]; then V=RED-broken-marker; elif [ "$STATEF" != 0 ]; then V=RED-stale-state; elif [ "$ARC" != 0 ]; then V=RED-tierA; elif [ "$BRC" != 0 ]; then V=RED-tierB; elif [ "$VRC" != 0 ]; then V=RED-verify; elif [ "$ATT" != 0 ]; then V=AMBER-repaired; else V=GREEN; fi
  # Atomic-ish row emission: write the row to a per-env temp file and rename it
  # into place, so a killed env leaves NO row rather than half a row (a missing
  # row is NOT_RUN under cert_verdict.sh, which is loud; a truncated row would
  # parse as something else).
  printf '%s\t%s\t%ss\t%s\t%s\t%s\t%s\t%s\n' "$ENV" "$IRC" "$W" "$ARC" "$BRC" "$VRC" "$ATT" "$V" > "$ROWD/$ENV.row.tmp"
  mv -f "$ROWD/$ENV.row.tmp" "$ROWD/$ENV.row"
  printf '%s\t%s\t%s\t%s\n' "$ENV" "$IRC" "${RSS:-unknown}" "$W" > "$ROWD/$ENV.led.tmp"
  mv -f "$ROWD/$ENV.led.tmp" "$ROWD/$ENV.led"
  # start/end epoch, so the concurrency ACTUALLY achieved is measurable after
  # the fact instead of assumed from CERT_PARALLEL.
  printf '%s\t%s\t%s\n' "$ENV" "$S" "$(( S + W ))" > "$ROWD/$ENV.span"
  echo "### $ENV install_rc=$IRC wall=${W}s peak_rss_kb=${RSS:-unknown} verdict=$V"
}

# Longest-first (LPT) over $CERT_WALL_TABLE. Ties break on name so the order is
# reproducible; an env the table does not name sorts first.
cert_run_order() {
  if [ -n "${CERT_WALL_TABLE:-}" ] && [ -r "$CERT_WALL_TABLE" ]; then
    for e in $CERT_ENVS; do
      w=$(awk -F'\t' -v e="$e" '$1==e && $4 ~ /^[0-9]+$/ {print $4; exit}' "$CERT_WALL_TABLE")
      printf '%s\t%s\n' "${w:-99999999}" "$e"
    done | sort -k1,1nr -k2,2 | cut -f2
  else
    printf '%s\n' $CERT_ENVS
  fi
}

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "### DRY_RUN=1 -- gates, env block and persistent caches are DONE; no env will be installed."
  echo "### CERT_PARALLEL=$CERT_PARALLEL  wall table=${CERT_WALL_TABLE:-<none, declaration order>}"
  echo "### link mode AT THE POINT OF USE: UV_LINK_MODE=$UV_LINK_MODE (asked for CERT_UV_LINK_MODE=$CERT_UV_LINK_MODE)"
  echo "### envs that WOULD run, in DISPATCH order (longest-first):"; cert_run_order | sed 's/^/  /'
  echo "### rows would be concatenated back in DECLARATION order:"; for ENV in $CERT_ENVS; do echo "  $ENV"; done
  echo "### verdict gate that WOULD score them: $VERDICT against $CERT_BASELINE"
  echo "### cleanup that WOULD be submitted: $CLEANUP with --dependency=afterany:${P1_JOB:-<p1>}:$J on $C ${P1_CACHE_ROOT:-} $WS"
  echo "### cleanup owner resolves to: $(cleanup_owner) (CLEANUP_JOB=${CLEANUP_JOB:-unset} CLEANUP_AT_DISPATCH=${CLEANUP_AT_DISPATCH:-0})"
  exit 0
fi

########## THE ENV LOOP -- $CERT_PARALLEL wide, longest-first ##########
case "$CERT_PARALLEL" in ''|*[!0-9]*|0) echo "### FATAL CERT_PARALLEL must be a positive integer, got '$CERT_PARALLEL'"; exit 2;; esac
LOOP_S=$(date +%s)
echo "### env loop: UV_LINK_MODE=$UV_LINK_MODE CERT_PARALLEL=$CERT_PARALLEL wall_table=${CERT_WALL_TABLE:-<none, declaration order>} start $(date -Is)"
echo "### dispatch order: $(cert_run_order | tr '\n' ' ')"
RUNNING=0
PIDS=()
for ENV in $(cert_run_order); do
  while [ "$RUNNING" -ge "$CERT_PARALLEL" ]; do wait -n; RUNNING=$((RUNNING-1)); done
  ( run_env "$ENV" ) &
  PIDS+=($!)
  RUNNING=$((RUNNING+1))
  echo "### dispatched $ENV (in flight $RUNNING/$CERT_PARALLEL) $(date -Is)"
done
# NEVER `wait` (bare) and NEVER `wait -n` here. This script starts with
# `exec > >(tee -a ...)`; under bash 5.1.8 that tee is a child that never exits,
# and BOTH forms then block on it forever:
#   * bare `wait` waits for ALL children -> job 5658928 wrote all 26 rows by
#     01:55:39 EDT and sat here until its 4 h limit killed it at 02:16 (TIMEOUT,
#     empty cert_results.tsv, no verdict, no cleanup submitted);
#   * `while [ $RUNNING -gt 0 ]; do wait -n; ... done` fails the same way once
#     the env subshells have ALREADY exited -- job 5674557 (one env, done
#     03:59:05) was still in it at 04:13 with `tee` as its only live child.
# Waiting on an EXPLICIT pid returns immediately for an exited child and 127 for
# one the throttle already reaped, so it can neither hang nor lose a status.
# The throttle above keeps `wait -n` on purpose: there it blocks until a real
# env completion, which is exactly what it is for.
for P in ${PIDS[@]+"${PIDS[@]}"}; do wait "$P" 2>/dev/null; done
LOOP_W=$(( $(date +%s) - LOOP_S ))
echo "### env loop DONE wall=${LOOP_W}s $(date -Is)"

# The 5547450 signature, hunted in EVERY install log. Cheap, and it is the only
# thing standing between a hardlink run and a silent bad link.
RACE=$(grep -nE 'failed to hardlink|EXDEV|Text file busy|builds-v0' "$A"/*.cert-install.log 2>/dev/null)
if [ -n "$RACE" ]; then
  echo "### HARDLINK RACE SIGNATURE PRESENT in an install log -- do not ship this link mode:"
  printf '%s\n' "$RACE" | head -60
  printf '%s\n' "$RACE" > "$A/hardlink_race.$J.txt"
else
  echo "### no hardlink/EXDEV/busy/builds-v0 line in any install log (link mode $UV_LINK_MODE)"
fi

# Bytes the installs actually wrote: /usr/bin/time -v "File system outputs", in
# 512-byte blocks, per env and totalled. Attributable, and independent of sacct.
: > "$A/fs_outputs.$J.tsv"
for ENV in $CERT_ENVS; do
  FSO=$(awk -F": " '/File system outputs/{print $2}' "$A/$ENV.cert-install.time.txt" 2>/dev/null)
  printf '%s\t%s\n' "$ENV" "${FSO:-na}" >> "$A/fs_outputs.$J.tsv"
done
awk -F'\t' '$2 ~ /^[0-9]+$/ { t+=$2 } END { printf "### install writes: %d blocks = %.2f GB (UV_LINK_MODE=%s)\n", t, t*512/1e9, mode }' mode="$UV_LINK_MODE" "$A/fs_outputs.$J.tsv"

# Reassemble in DECLARATION order. A missing row means that env's subshell died
# without finishing; it is left MISSING on purpose, so cert_verdict.sh reports
# NOT_RUN and exits 1 rather than a short file passing quietly.
MISSING=0
for ENV in $CERT_ENVS; do
  if [ -f "$ROWD/$ENV.row" ]; then cat "$ROWD/$ENV.row" >> "$RES"; else echo "### MISSING ROW for env $ENV -- its subshell produced none"; MISSING=$((MISSING+1)); fi
  [ -f "$ROWD/$ENV.led" ] && cat "$ROWD/$ENV.led" >> "$LEDGER"
done
echo "### rows assembled: $(wc -l < "$RES") of $(echo $CERT_ENVS | wc -w) (missing=$MISSING)"

# Concurrency ACTUALLY achieved, from the per-env start/end stamps -- not from
# CERT_PARALLEL, which is only what was asked for.
cat "$ROWD"/*.span 2>/dev/null | sort -k2,2n > "$A/env_spans.$J.tsv"
awk -F'\t' -v loop="$LOOP_W" '
  { s[NR]=$2; e[NR]=$3; sum+=$3-$2; if(!lo||$2<lo)lo=$2; if($3>hi)hi=$3; n=NR }
  END {
    if (!n) { print "### spans: none"; exit }
    peak=0
    for (i=1;i<=n;i++) { c=0; for (j=1;j<=n;j++) if (s[j]<=s[i] && e[j]>s[i]) c++; if (c>peak) peak=c }
    printf "### concurrency: peak=%d  mean=%.2f (sum of env walls %ds / span %ds)  loop wall %ds\n", peak, sum/(hi-lo), sum, hi-lo, loop }
' "$A/env_spans.$J.tsv"

# Evidence is collected BEFORE the gate. No green claim is made merely because
# an older baseline also contained a failure.
collect_artifacts

echo "### memory ledger (peak RSS per env, from /usr/bin/time -v):"
sort -t$'\t' -k3,3n "$LEDGER" | tail -5
if awk -F'\t' '$1=="hover-gpu" && $4==1 {ok=1} END{exit !ok}' "$RES"; then echo '### hover-gpu TierA rc=1 is the known pre-existing neural_wbc result'; fi

# Verdict via cert_verdict.sh. Typed exits: 0 identical, 1 a row differs or is
# missing, 2 SETUP FAILURE. A setup failure must never read as a pass -- the
# older `rg` gate did exactly that (rg is not on the PATH this script exports,
# so the `if` was command-not-found and fell through to success).
if [ ! -x "$VERDICT" ]; then
  echo "### FATAL verdict script missing or not executable: $VERDICT"
  VRC=2
else
  "$VERDICT" "$RES" "$CERT_BASELINE" | tee "$A/cert_verdict.$J.txt"
  VRC=${PIPESTATUS[0]}
fi
case "$VRC" in
  0) echo "### RESULT identical to certified baseline $CERT_BASELINE" ;;
  1) echo "### RESULT differs from certified baseline; inspect $A/cert_verdict.$J.txt; release decision reopens" ;;
  *) echo "### RESULT verdict could not be computed (rc=$VRC); this is NOT a pass" ;;
esac

# Second gate: the same lock's previous results, when the harness names one.
if [ -n "${CERT_SERIAL_KEY:-}" ]; then
  if [ -r "$CERT_SERIAL_KEY" ] && [ -x "$VERDICT" ]; then
    "$VERDICT" "$RES" "$CERT_SERIAL_KEY" | tee "$A/cert_verdict_vs_prior.$J.txt"
    KRC=${PIPESTATUS[0]}
  else
    echo "### FATAL second-gate key unreadable or verdict script missing: $CERT_SERIAL_KEY"
    KRC=2
  fi
  echo "### SECOND GATE vs $CERT_SERIAL_KEY -> exit $KRC"
  [ "$KRC" = 0 ] || { echo "### this run is NOT identical to the prior run of the same lock"; [ "$VRC" = 0 ] && VRC=$KRC; }
fi

########## CLEANUP -- SUBMITTED, NOT RUN. This is the last job of the chain. ##########
# afterany, so a RED cert still returns its disk. The roots are named here first
# so a diagnostician can hold them by cancelling the cleanup job before it runs.
ROOTS="$C ${P1_CACHE_ROOT:-} $WS"
echo "### cleanup roots: $ROOTS"
# THE DEPENDENCY IS `afterany` ON BOTH PHASES, and it is submitted at DISPATCH
# when it can be. A cleanup chained behind the CERT alone never runs when the
# RELOCK fails, because Slurm cancels the afterok cert and the cleanup's own
# dependency then never releases -- that is how `certC18A-5759225`/`ws.C18A-…`
# and the C18B pair were stranded (C18B-run.out: "the afterok dependency will
# not release; self-cleanup NOT run here by design"). Preferred shape, p6mbc's
# and b4u's (job 5769783, `afterany:5769781,afterany:5769782`): the LAUNCHER
# submits one gated cleanup with `--dependency=afterany:<p1>:<p2>` and sets
# CLEANUP_AT_DISPATCH=1 so this block only prints. When it was not submitted at
# dispatch, this block submits it here and includes the relock job in the
# dependency too, so the same job also covers a relock that never handed off.
CLEANUP_DEP=afterany:$J
[ -n "${P1_JOB:-}" ] && CLEANUP_DEP=afterany:${P1_JOB}:$J
# ONE owner. Defers when the stamp names a dispatch cleanup, submits exactly one
# when it does not, and names the owning job id either way.
# shellcheck disable=SC2086
cleanup_submit_or_defer "$CLEANUP_DEP" $ROOTS
echo "### inode quota AFTER (pre-cleanup):"; "$CQ" 2>/dev/null | grep -E 'data\+stellex' | head -2
echo "### ${TAG} CERT DONE verdict_rc=$VRC $(date -Is)"
exit "$VRC"
