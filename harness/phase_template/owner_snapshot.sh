#!/usr/bin/env bash
# owner_snapshot.sh -- freeze a cleanup owner's scripts into the JOB ROOT and
# generate the sbatch that runs the frozen copy. HARNESS-SYNC-5.
#
# THE DEFECT, MEASURED. A cleanup owner is submitted as
#   sbatch --wrap 'bash <task dir>/merge-h/cleanup_gated.sh <roots>'
# Slurm snapshots the TOP-LEVEL script -- here, the one-line wrap -- and nothing
# else, so `cleanup_gated.sh` and the `cleanup.sh` it calls are read from the
# task tree LIVE, for as long as the owner runs. det1f-cleanup 5999937 was four
# hours into a single 3,587,597-entry unlink with those reads open; det141-cleanup
# 6001240 sat beside it. harness_sync's read-set check is RIGHT to refuse rc 6 on
# that -- rewriting a file a reader on another NFS client has open unlinks its
# inode under them, bash reads an error, calls it EOF, and the job exits 0 having
# run half its rows -- but the CAUSE is that a reap measured in hours holds a
# live read of a SYNCED file. Nothing can install while it runs, and reaps like
# that run every night.
#
# THE FIX IS NOT TO WEAKEN THE REFUSAL. It is to stop the owner reading a synced
# file at all: copy what it will read into the job root at SUBMIT time and submit
# a real sbatch script that execs the copy. The job root is job-local and nothing
# syncs into it, so a later install cannot reach the reader, and the read-set
# check then sees paths under the job root rather than under tools/ and stops
# refusing -- correctly, because there is now nothing to protect.
#
# THE SET IS DERIVED, NOT LISTED. tools/script_refs.sh is the SAME parser
# harness_sync.sh uses for its read set. Deriving the snapshot from a different
# parser than the one that judges it is how a snapshot ends up missing exactly
# the file the job sources, so there is one parser and both call it. References
# are followed transitively; a reference the parser cannot resolve is a REFUSAL,
# not a shrug, because a snapshot with a hole is worse than no snapshot: the job
# would fall back to the live path with a row saying it was frozen.
#
#   usage: owner_snapshot.sh <job root> <script> [<script> ...] \
#                            [--arms <n>] --roots <root> [<root> ...]
#          owner_snapshot.sh <job root> <script> ... \
#                            --allow-underived --reason "<why this needs no wall>"
#
#   `--roots` IS REQUIRED (HARNESS-CONSOL-12). Without it the wall cannot be
#   derived and the owner would carry covers=0 / wall=0s placeholders that every
#   downstream comparison reads as measurements -- the shape det163's 24
#   self-continuing owners were generated in. The roots need not EXIST yet: an
#   absent root is estimated for the wall at submit and censused as zero inside
#   the owner. A caller that never submits the owner it generates says so with
#   `--allow-underived --reason`, which leaves a marker sidecar beside the job
#   root; that flag WITHOUT a reason is still a refusal.
#
#   Writes  <job root>/owner-snapshot/          the frozen scripts
#           <job root>/owner-snapshot/owner.sbatch   execs the FIRST script,
#                                                    passing "$@" through
#           <job root>/owner-snapshot/owner.wall     the DERIVED `--time=HH:MM:SS`
#                                                    for the caller's sbatch line
#   Prints  ### OWNER SNAPSHOT files=<n> root=<job root> src_commit=<sha> src_kind=<git|record> ...
#           ### OWNER SNAPSHOT froze file=<b> md5=<m> src_kind=<k> src_sha=<sha> dirty=<y/n>
#           ### OWNER SNAPSHOT wall=<s> ... (CLEANUP-WALL-1, below)
#   rc 0    frozen; rc 2 refused (and nothing was written that a caller may use)
#
# CLEANUP-WALL-1 (2026-09-07). THE WALL IS THE ONE PARAMETER OF A REAP THAT IS A
# FUNCTION OF THE WORK, AND IT WAS THE ONLY ONE NOBODY COMPUTED. Every cleanup
# owner in this campaign was submitted with a wall typed by hand at the submit
# site -- `--time=16:00:00` in det16_proof.sh's submit_owner(), `--time=06:00:00`
# in phaseN_cert.sh's CLEANUP_SBATCH_ARGS -- and a hand-typed wall is a constant
# in front of a variable quantity, so eventually it is too small. It was:
# det1f-cleanup 5999937 was given 6 h for 6.66 M entries, spent 17507 s removing
# certDET1F-5992569 (3,587,597 entries), started ws.DET1F-5992569 (3,068,868
# entries) at 03:14:59 and TIMED OUT 29 min later (`sacct -j 5999937`: TIMEOUT,
# Elapsed 06:00:04 against Timelimit 06:00:00), orphaning the second root.
#
# THE RATE IS MEASURED, NOT ASSUMED. Four cleanup rows, each with entries and
# wall printed by cleanup.sh itself:
#     5992050 certDET1F-5989192   953311 /  4492s = 212 entries/s
#     5992050 ws.DET1F-5989192    842018 /  4104s = 205 entries/s
#     5999937 certDET1F-5992569  3587597 / 17507s = 205 entries/s
#     6001240 certD141-6001140   2583379 / 12050s = 214 entries/s
#     6017160 certD16-6013332      44354 /   211s = 210 entries/s  (CLEANUP-WALL-2,
#       REAP-3: a 44 k root, two orders of magnitude smaller than the others, and
#       it lands in the SAME band -- which is the strongest evidence yet that this
#       is a per-entry cost and not a per-root one)
# Stable to within 5% across jobs, nodes and roots and across two orders of
# magnitude of root size, which is what makes it a planning constant rather than
# an observation. RE-CHECKED 2026-09-07 against that fifth row: the floor of 200
# still sits below every measurement (205 is the slowest) and is NOT raised --
# this term buys wall, so it must under-estimate the rate, never over-estimate
# it, and a wall derived from an optimistic rate is the defect this file exists
# to remove. It is lowered the day a slower row is measured, and only then.
#
# THE CENSUS COUNTS TOWARD THE WALL. cleanup.sh walks each root with
# `find "$r" | wc -l` and prints `### removing $r (entries=$N)` BEFORE it
# unlinks anything, so every reap pays a full metadata walk first; on NFS that
# is ~30 min per 3 M entries. It gets its own allowance rather than being hidden
# inside the margin, because a term that is not named cannot be re-measured.
#
# THE ROOTS USUALLY DO NOT EXIST YET. Owners are submitted BEFORE arm 1 (that is
# the point of an `afterany` owner), so at submit time a declared root is a name
# and not a tree. A root that IS present is censused; a root that is absent is
# sized from OWNER_ENTRIES_PER_ROOT_EST, which is this campaign's LARGEST
# measured root (certDET1F-5992569, 3,587,597 entries, rounded up).
#
# AND THE ESTIMATE IS RE-DERIVED INSIDE THE OWNER, WHICH IS THE ACTUATOR (law 9).
# An estimate that is only checked at submit is a guess with a nicer name. The
# generated owner.sbatch re-censuses the roots it was handed, compares against
# the entry count its wall was derived to cover, and if the real tree is bigger
# it prints `### OWNER WALL SHORT census=<n> covers=<n>` and RESUBMITS ITSELF for
# the remainder behind its own job id with a freshly derived wall -- so the
# reap removes what fits and a successor finishes the rest, instead of dying
# silently at a wall like 5999937 did. cleanup_gated.sh's ABSENT-root branch
# (MERGE-N-1) is what makes the successor safe: a root the first pass finished
# is a no-op exit 0 for the second, not a refusal.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB_ROOT=${1:-}; shift || true

########## CLEANUP-WALL-1: THE CONSTANTS, EACH CITED TO THE ROWS IT CAME FROM ##
# Every one of these is a MEASURED number or an explicit policy choice, and each
# is named so a later lane can re-measure it rather than re-guess it.
OWNER_UNLINK_RATE_PER_S=${OWNER_UNLINK_RATE_PER_S:-200}   # rows: 212 / 205 / 205 / 214 (jobs 5992050 x2, 5999937, 6001240)
OWNER_CENSUS_RATE_PER_S=${OWNER_CENSUS_RATE_PER_S:-2800}  # rows: 2807.7 / 2915.4 (clean walks, jobs 5999937 and 6001240); see below
# MEASURED 2026-09-07, not quoted. This read 1700 with the comment "~30 min per
# 3 M entries over NFS = 1667/s, rounded" -- a STEWARD'S SENTENCE, never a row.
# The walk that produces `entries=<N>` is cleanup.sh's `N=$(find "$r" | wc -l)`
# immediately before its `### removing` row, so the interval from the PREVIOUS
# root's `### removed` row to the NEXT root's `### removing` row is exactly one
# census walk and nothing else. Two such CLEAN intervals exist in the logs:
#   job 5999937 -- 3,068,868 entries in 1093 s = 2807.7/s (the slowest, and the
#     value below):
#       ### removed /oscar/.../certDET1F-5992569 rc=0 wall=17507s exists_after=no 2026-09-07T02:56:46-04:00
#       ### removing /oscar/.../ws.DET1F-5992569  (entries=3068868) start 2026-09-07T03:14:59-04:00
#   job 6001240 -- 2,154,463 entries in 739 s = 2915.4/s:
#       ### removed /oscar/.../certD141-6001140 rc=0 wall=12050s exists_after=no 2026-09-07T04:07:08-04:00
#       ### removing /oscar/.../ws.D141-6001140  (entries=2154463) start 2026-09-07T04:19:27-04:00
#   job 6013780 -- a FIRST-root interval, so it also contains hostname/date/
#     checkquota and is a LOWER BOUND, quoted only as corroboration: 3,068,868
#     entries in <=1031 s from the gate's own `2026-09-07T03:58:29-04:00` row =
#     >=2976.6/s:
#       ### removing /oscar/.../ws.DET1F-5992569  (entries=3068868) start 2026-09-07T04:15:40-04:00
# All five measurable walks land in 2807-2977/s. 2800 is below every one of them
# and is the FLOOR of the measurements, not their mean: this term buys wall, so
# it must under-estimate the rate, never over-estimate it. The old 1700 was
# 1.65x conservative for no measured reason, and a term nobody can re-derive
# from a row is the thing this constant block exists to abolish.
OWNER_WALL_MARGIN=${OWNER_WALL_MARGIN:-2}                 # 2x on the unlink term
# TWO metadata walks are paid per reap, not one: cleanup.sh censuses each root
# before it unlinks (`### removing $r (entries=$N)`), and the owner re-censuses
# at start to decide whether its wall still covers the tree. Named rather than
# folded into the margin so it can be re-measured on its own.
OWNER_CENSUS_WALKS=${OWNER_CENSUS_WALKS:-2}
OWNER_WALL_FLOOR_S=${OWNER_WALL_FLOOR_S:-3600}            # a tiny root still gets a usable wall
# The cap is OURS, not the scheduler's: `sinfo -p batch -o %l` = infinite and
# `sacctmgr show qos normal format=MaxWall` is EMPTY (both measured 2026-09-07).
# Past a day the right answer is a split, and the split exists -- see the
# self-continuation in the generated owner.sbatch.
OWNER_WALL_MAX_S=${OWNER_WALL_MAX_S:-86400}
# The largest root this campaign measured: certDET1F-5992569 = 3,587,597
# entries (job 5999937), rounded up. Used ONLY for a root that does not exist
# yet, which is the normal case because owners are submitted before arm 1.
OWNER_ENTRIES_PER_ARM_EST=${OWNER_ENTRIES_PER_ARM_EST:-3600000}
# CLEANUP-WALL-2 (2026-09-07, REAP-3 finding 2). THE CENSUS AND THE THING IT
# SIZES WERE WALKING DIFFERENT TREES. This census used `find "$r" -maxdepth 16`
# while the work it sizes -- cleanup.sh's `N=$(find "$r" | wc -l)` and the
# `rm -rf` behind it -- is UNBOUNDED, so every entry deeper than 16 was invisible
# to the wall and free to the reaper. Two measured disagreements from ONE reap
# (job 6017160, root certD16-6013332, a static tree, the two walks six minutes
# apart): owner_snapshot's `entries=44040` against cleanup.sh's `entries=44354`,
# and on the earlier root `423482` against `434860`. The cause is not a race, it
# is the depth: the smoke's `cp -al` mirror nests worktrees inside worktrees, and
# the SETUP-REFUSED branch on that same job refused at depth 8 naming
# `certD16-6013332/smk/w/.claude/worktrees/agent-ab947ce3406deed7e/assets/cad/
# H1_2_wrist_no_camera.STL`. A census that under-counts buys a wall that is too
# short, which is the exact failure mode (det1f-cleanup 5999937, TIMEOUT with a
# root half gone) this whole file exists to remove.
#
# THE FIX IS TO WALK WHAT THE REAPER WALKS. The census is now unbounded, the same
# `find "$r"` cleanup.sh runs. The old bounded count is still taken FOR ONE
# RELEASE and printed beside it whenever the two differ, so the size of the blind
# spot is on the record rather than in a lane's memory; set
# OWNER_CENSUS_COMPARE_DEPTH= (empty) to drop the second walk, and delete this
# block when the comparison has stopped being interesting.
OWNER_CENSUS_COMPARE_DEPTH=${OWNER_CENSUS_COMPARE_DEPTH-16}

# OWNER-EXPORT-1. The `--export` clause has ONE producer, tools/owner_export.sh,
# and it is sourced HERE so that `declare -f owner_export_clause` below ships it
# into the generated owner.sbatch alongside owner_wall_check. Not optional: an
# owner submitted with `--export=ALL,...` is held by Slurm with
# `user env retrieval failed requeued held` and nothing in this harness reads
# that reason, so refusing here is the loud failure law 9 asks for.
OE=$HERE/../tools/owner_export.sh
[ -f "$OE" ] || OE=$HERE/owner_export.sh
[ -f "$OE" ] || {
  echo "### OWNER SNAPSHOT REFUSED: no owner_export.sh -- the generated owner would have to"
  echo "###   build its own --export clause, and the last time two sites built that clause"
  echo "###   four owners sat held for days (REAP-3: 6013350 6013351 6014485 5841188)."
  exit 2; }
# shellcheck disable=SC1090
. "$OE"

# ONE derivation, defined once and SHIPPED into the generated owner.sbatch by
# `declare -f` below, so the number the submitter derives and the number the
# owner re-derives cannot drift apart into two implementations.
# CLEANUP-WALL-3 (2026-09-07). THE ABSENT-ROOT ESTIMATE IS A SUBMIT-TIME DEVICE
# AND IT WAS BEING USED AT RUN TIME. At submit the roots do not exist yet -- the
# owner is queued `--dependency=afterany:<relock job>` before the job that
# creates them has run -- so an absent root has to be ESTIMATED or the wall would
# be derived for an empty tree. Inside the OWNER that reasoning is inverted:
# there, absent means the tree is not there, so there is nothing to unlink and
# nothing to size. Measured: det163-cleanup-cont 6020524 censused
# `entries=14400000 present=0 absent=4` -- 14.4 million phantom entries for four
# roots that did not exist -- and every row downstream of that number was wrong.
# The generated owner.sbatch sets OWNER_CENSUS_ESTIMATE_ABSENT=0; the submitter
# leaves it at 1.
OWNER_CENSUS_ESTIMATE_ABSENT=${OWNER_CENSUS_ESTIMATE_ABSENT:-1}

owner_census () {                # $@ = declared roots; echoes "<entries> <present> <absent>"
  local r n nb tot=0 pres=0 abs=0
  for r in "$@"; do
    if [ -e "$r" ]; then
      # CLEANUP-WALL-2: the SAME walk cleanup.sh runs before it unlinks, so the
      # number that buys the wall and the number the reaper prints are one walk.
      n=$(find "$r" 2>/dev/null | wc -l)
      if [ -n "${OWNER_CENSUS_COMPARE_DEPTH:-}" ]; then
        nb=$(find "$r" -maxdepth "$OWNER_CENSUS_COMPARE_DEPTH" 2>/dev/null | wc -l)
        [ "$nb" = "$n" ] || echo "### OWNER CENSUS DEPTH $r unbounded=$n depth${OWNER_CENSUS_COMPARE_DEPTH}=$nb undercount=$((n - nb)) -- the bounded walk this census used to do would have bought a wall for $nb entries and the reaper would have unlinked $n" >&2
      fi
      tot=$((tot + n)); pres=$((pres + 1))
    else
      # CLEANUP-WALL-3: estimate only where an estimate is the only thing there
      # is -- at submit, before the tree exists.
      [ "$OWNER_CENSUS_ESTIMATE_ABSENT" = 0 ] ||
        tot=$((tot + OWNER_ENTRIES_PER_ARM_EST * OWNER_ARMS))
      abs=$((abs + 1))
    fi
  done
  echo "$tot $pres $abs"
}

owner_wall_derive () {           # $1 = entries; echoes "<wall seconds> <census allowance seconds>"
  local e=${1:-0} unlink census wall
  unlink=$(( (e + OWNER_UNLINK_RATE_PER_S - 1) / OWNER_UNLINK_RATE_PER_S ))
  census=$(( (e + OWNER_CENSUS_RATE_PER_S - 1) / OWNER_CENSUS_RATE_PER_S ))
  census=$(( census * OWNER_CENSUS_WALKS ))
  wall=$(( unlink * OWNER_WALL_MARGIN + census ))
  [ "$wall" -lt "$OWNER_WALL_FLOOR_S" ] && wall=$OWNER_WALL_FLOOR_S
  [ "$wall" -gt "$OWNER_WALL_MAX_S" ] && wall=$OWNER_WALL_MAX_S
  echo "$wall $census"
}

owner_wall_hms () {              # $1 = seconds -> HH:MM:SS, rounded UP to the minute
  local s=${1:-0}
  s=$(( ( (s + 59) / 60 ) * 60 ))
  printf '%02d:%02d:%02d\n' $(( s / 3600 )) $(( (s % 3600) / 60 )) $(( s % 60 ))
}

# THE OWNER'S OWN RE-DERIVATION. Shipped verbatim into the generated
# owner.sbatch by `declare -f`, so what the submitter computed and what the
# owner checks are one text and cannot diverge. It runs INSIDE the owner, at
# start, when the roots are real trees rather than names -- and when the tree is
# bigger than the wall was derived to cover it does not shrug and it does not
# die at the wall: it says so on one row and submits its own continuation for
# whatever it cannot finish. cleanup_gated.sh's ABSENT-root branch makes the
# second pass a no-op on any root the first one finished.
# CLEANUP-WALL-3 (2026-09-07). THE CONTINUATION WAS SUBMITTED BEFORE THE PASS RAN
# AND AGAINST A WALL THAT WAS NEVER DERIVED.
#
# THE DEFECT, MEASURED. `owner_wall_check` ran at job START, on the line above
# `exec bash <snap>/cleanup_gated.sh`, so its verdict could not depend on
# anything the pass did -- the gate had not spoken yet. Its whole test was
# `[ "$c" -gt "$OWNER_WALL_COVERS" ]`, and OWNER_WALL_COVERS is 0 for every owner
# whose submitter passed no `--roots` (det163_proof.sh's submit_owner calls
# `owner_snapshot.sh "$jr" "$gate"` and nothing else; the tool DID say so --
# `### OWNER SNAPSHOT wall=UNDERIVED roots=0` is in det163-6020526.out -- and
# nobody read it, which is law 9's detector with no actuator). Against covers=0
# ANY census is "short", so every owner submitted a continuation, and the
# continuation did it again, four deep, until the cap. Thirty det163-cleanup jobs
# on 2026-09-07 08:07-08:25, twenty-four of them continuations, every one of them
# reaching a pass that had nothing to do or refused:
#   6020506 -> `### OWNER WALL CENSUS entries=10800004 present=1 absent=3 covers=0 wall=0s`
#              `### OWNER WALL SHORT census=10800004 covers=0 derived=86400s`
#              `### OWNER WALL CONTINUATION submitted job=6020524` -- and only
#              THEN `### CLEANUP SETUP-REFUSED roots=1 removed=1`.
#   6020524 -> census entries=14400000 present=0 absent=4, continuation 6020542,
#              then `### NOTHING TO DO -- every root named is ABSENT`.
#   6020631/6020952/6021096 -> census present=1, continuation each time, then
#              `### JOB-FATAL NOT TAKEN` and `### CLEANUP REFUSED -- nothing deleted`
#              on all three; the tree they were "continuing" is one the gate
#              refuses on purpose, and no number of passes will change that.
#
# THE FIX. A continuation is earned by a pass, not predicted before one. It runs
# AFTER the gate and continues only when ALL of these hold:
#   * this pass actually removed something -- at least one `### removed` row. A
#     pass that removed nothing cannot be short of wall: it is done, or refused.
#   * something is left -- the post-pass census over PRESENT roots is > 0.
#   * the gate did not reach a terminal decision. NOTHING TO DO, CLEANUP REFUSED,
#     JOB-FATAL NOT TAKEN and a SETUP-REFUSED that left nothing behind are all
#     answers, not interruptions.
#   * the pass ended for WALL reasons, which requires the owner to HAVE a wall:
#     OWNER_WALL_S > 0 and the pass consumed OWNER_WALL_PRESSURE_NUM/DEN of it.
#     An owner with an underived wall cannot be short of it, and saying so out
#     loud is what turns the UNDERIVED row into an actuator.
# The depth cap stays as the last resort it always was, and still prints the
# hand-run line when it bites.
OWNER_WALL_PRESSURE_NUM=${OWNER_WALL_PRESSURE_NUM:-4}   # continue only past 4/5 of
OWNER_WALL_PRESSURE_DEN=${OWNER_WALL_PRESSURE_DEN:-5}   # the derived wall

# The pre-pass row. It MEASURES and it says so; it does not decide. Shipped into
# the generated owner.sbatch by `declare -f` with everything else.
owner_wall_census_row () {       # $@ = the roots this owner was handed
  local c pres abs hms
  read -r c pres abs < <(owner_census "$@")
  hms=$(owner_wall_hms "$OWNER_WALL_S")
  echo "### OWNER WALL CENSUS entries=$c present=$pres absent=$abs covers=$OWNER_WALL_COVERS wall=${OWNER_WALL_S}s ($hms) cont=$OWNER_CONT_N"
  if [ "$OWNER_WALL_S" -le 0 ]; then
    echo "### OWNER WALL UNDERIVED: this owner was generated with no --roots, so covers=$OWNER_WALL_COVERS"
    echo "###   and wall=0s are placeholders, not measurements. It will run its pass, but it"
    echo "###   CANNOT conclude it was cut short by a wall it never had, and it will not"
    echo "###   continue itself. Fix the submit site to pass --roots <root>... ."
  fi
  return 0
}

# The post-pass decision. $1 = the gate's rc, $2 = the pass log, $3.. = roots.
owner_continue_check () {
  local prc=$1 plog=$2; shift 2
  local removed=0 c pres abs new nc cj rc hms el reason=
  # NO `|| echo 0` HERE. `grep -c` PRINTS its count and THEN exits 1 when the
  # count is zero, so `$(grep -c ... || echo 0)` captures "0\n0" -- measured on
  # 2026-09-07: the row would read `removed=0 0` and every `[ "$removed" -gt 0 ]`
  # below would die "integer expression expected" into the owner's own stdout.
  # `grep -c` always prints a number, so the count needs no fallback; the `:=`
  # covers only the case where $plog is absent and grep never ran.
  [ -f "$plog" ] && removed=$(grep -c '^### removed ' "$plog" 2>/dev/null)
  case $removed in ''|*[!0-9]*) removed=0 ;; esac
  read -r c pres abs < <(owner_census "$@")
  el=${SECONDS:-0}
  echo "### OWNER PASS RESULT rc=$prc removed=$removed remaining=$c present=$pres absent=$abs elapsed=${el}s wall=${OWNER_WALL_S}s cont=$OWNER_CONT_N"

  # the terminal decisions, each named by the row that carries it
  if [ -f "$plog" ]; then
    grep -q '^### NOTHING TO DO'        "$plog" && reason='the gate said NOTHING TO DO -- every root ABSENT'
    [ -n "$reason" ] || { grep -q '^### CLEANUP REFUSED'   "$plog" && reason='the gate said CLEANUP REFUSED -- nothing deleted, and a refusal is an answer'; }
    [ -n "$reason" ] || { grep -q '^### JOB-FATAL NOT TAKEN' "$plog" && reason='the gate said JOB-FATAL NOT TAKEN -- a sealed store it refuses on purpose'; }
  fi
  [ -n "$reason" ] || [ "$removed" -gt 0 ] || reason="this pass removed nothing (no '### removed' row), so it was not cut short -- it was done or refused"
  [ -n "$reason" ] || [ "$c" -gt 0 ] || reason='nothing is left: the post-pass census over the present roots is 0'
  [ -n "$reason" ] || [ "$OWNER_WALL_S" -gt 0 ] || reason='this owner has no derived wall (wall=0s, covers=0), so it cannot have been short of one -- see the OWNER WALL UNDERIVED row'
  [ -n "$reason" ] || [ $(( el * OWNER_WALL_PRESSURE_DEN )) -ge $(( OWNER_WALL_S * OWNER_WALL_PRESSURE_NUM )) ] ||
    reason="it used ${el}s of a ${OWNER_WALL_S}s wall, under $OWNER_WALL_PRESSURE_NUM/$OWNER_WALL_PRESSURE_DEN of it, so the wall is not what stopped it"
  if [ -n "$reason" ]; then
    echo "### OWNER NO CONTINUATION: $reason."
    return 0
  fi

  read -r new nc < <(owner_wall_derive "$c")
  hms=$(owner_wall_hms "$new")
  echo "### OWNER WALL SHORT census=$c covers=$OWNER_WALL_COVERS derived=${new}s ($hms) rate=$OWNER_UNLINK_RATE_PER_S margin=$OWNER_WALL_MARGIN census_allow=$nc removed=$removed"
  if [ "$OWNER_CONT_N" -ge "$OWNER_CONT_MAX" ]; then
    echo "### OWNER WALL CONTINUATION CAP HIT depth=$OWNER_CONT_N max=$OWNER_CONT_MAX removed=$removed remaining=$c"
    echo "###   A chain that never ends is not an actuator either. RUN THIS BY HAND -- it is the"
    echo "###   only thing that returns the remaining $c inodes:"
    echo "    env -u SLURM_JOB_ID sbatch --partition=${SLURM_JOB_PARTITION:-batch} --qos=${SLURM_JOB_QOS:-normal} --cpus-per-task=1 --mem=4G --time=$hms $OWNER_SELF $*"
    return 0
  fi
  # OWNER-EXPORT-1: an EXPLICIT list, never `ALL`. The continuation inherits the
  # same contract its parent ran under, plus its own bumped counter -- and, since
  # CLEANUP-WALL-3, what its parent actually did, so the chain is readable from
  # any one of its logs.
  local EXPCL
  EXPCL=$(owner_export_clause "OWNER_CONT_N=$((OWNER_CONT_N + 1))" \
                              "OWNER_CONT_PARENT=${SLURM_JOB_ID:-0}" \
                              "OWNER_CONT_PARENT_REMOVED=$removed") || {
    echo "### OWNER WALL CONTINUATION REFUSED: the export clause could not be built (rows above)."
    echo "###   The remainder has NO owner. RUN THIS BY HAND once the offending value is fixed:"
    echo "    env -u SLURM_JOB_ID sbatch --partition=${SLURM_JOB_PARTITION:-batch} --qos=${SLURM_JOB_QOS:-normal} --cpus-per-task=1 --mem=4G --time=$hms $OWNER_SELF $*"
    return 0; }
  cj=$(env -u SLURM_JOB_ID sbatch --parsable \
        --partition="${SLURM_JOB_PARTITION:-batch}" --qos="${SLURM_JOB_QOS:-normal}" \
        --cpus-per-task=1 --mem=4G --time="$hms" \
        --job-name="${SLURM_JOB_NAME:-owner}-cont" \
        --output="$OWNER_OUT_DIR/owner-cont-%j.out" \
        --dependency=afterany:"${SLURM_JOB_ID:-0}" \
        "$EXPCL" \
        "$OWNER_SELF" "$@" 2>&1); rc=$?
  if [ "$rc" = 0 ]; then
    echo "### CONTINUATION depth=$((OWNER_CONT_N + 1)) parent=${SLURM_JOB_ID:-0} parent_removed=$removed job=$cj time=$hms remaining=$c"
    echo "### OWNER WALL CONTINUATION submitted job=$cj time=$hms after ${SLURM_JOB_ID:-0} -- this pass removed $removed and hit its wall, that one finishes the remaining $c"
  else
    echo "### OWNER WALL CONTINUATION FAILED rc=$rc output: $cj"
    echo "###   The remainder has NO owner. RUN THIS BY HAND:"
    echo "    env -u SLURM_JOB_ID sbatch --partition=${SLURM_JOB_PARTITION:-batch} --qos=${SLURM_JOB_QOS:-normal} --cpus-per-task=1 --mem=4G --time=$hms $OWNER_SELF $*"
  fi
  return 0
}

########## the argument split: scripts to freeze, roots to size the wall from ##
SCRIPTS=(); ROOTS=(); OWNER_ARMS=1; argmode=scripts
ALLOW_UNDERIVED=0; UNDERIVED_REASON=
while [ "$#" -gt 0 ]; do
  case $1 in
    --roots) argmode=roots ;;
    --arms)  shift; OWNER_ARMS=${1:-1}; [ "$OWNER_ARMS" -ge 1 ] 2>/dev/null || OWNER_ARMS=1 ;;
    --allow-underived) ALLOW_UNDERIVED=1 ;;
    --reason) shift; UNDERIVED_REASON=${1:-} ;;
    *) if [ "$argmode" = roots ]; then ROOTS+=("$1"); else SCRIPTS+=("$1"); fi ;;
  esac
  shift
done
[ -n "$JOB_ROOT" ] && [ "${#SCRIPTS[@]}" -ge 1 ] || {
  echo "### OWNER SNAPSHOT REFUSED: usage: owner_snapshot.sh <job root> <script> [<script>...] [--arms <n>] [--roots <root>...] [--allow-underived --reason \"<why>\"]"; exit 2; }
set -- "${SCRIPTS[@]}"

########## HARNESS-CONSOL-12: AN UNDERIVED WALL REFUSES AT SUBMIT #############
# A DETECTOR WITH NO ACTUATOR IS A DEFECT (law 9), AND THIS ONE COST 24 JOBS.
# Called with no `--roots` this tool used to print
#   ### OWNER SNAPSHOT wall=UNDERIVED roots=0
# and carry on. det163_proof.sh's submit_owner calls
# `owner_snapshot.sh "$jr" "$gate"` and nothing else; the row is in
# det163-6020526.out TWICE, nobody read it, and the owners it generated -- with
# OWNER_WALL_COVERS=0 and OWNER_WALL_S=0, placeholders that every downstream
# comparison read as measurements -- self-continued four deep, twenty-four times
# on 2026-09-07 between 08:07 and 08:25. CLEANUP-WALL-3 stopped the owner from
# ACTING on the placeholder. This stops it being MADE, which is the other half:
# the refusal names the actuator (`--roots <every root the arms will create>`)
# rather than describing the hazard, so the row a caller gets is the fix.
#
# THE OPT-OUT IS `--allow-older`'s, deliberately: some callers legitimately want
# a snapshot with no wall -- guards that exercise the READ SET and never submit
# the owner they generate -- and for those a refusal would be a wall in the wrong
# place. It costs an explicit argv flag AND a written reason, and it leaves a
# marker sidecar beside the job root, so an underived owner can always be traced
# to the caller that asked for one. `--allow-underived` WITHOUT `--reason` is
# still a refusal: an undocumented opt-out is the row nobody reads again.
if [ "${#ROOTS[@]}" -eq 0 ] && [ "$ALLOW_UNDERIVED" != 1 ]; then
  echo "### OWNER SNAPSHOT REFUSED: no --roots, so this owner's wall CANNOT be derived and it"
  echo "###   would carry OWNER_WALL_COVERS=0 and OWNER_WALL_S=0 -- placeholders that read as"
  echo "###   measurements to everything downstream. det1f-cleanup 5999937 TIMED OUT at a"
  echo "###   hand-typed 6 h with 3,068,868 entries left; det163's owners self-continued four"
  echo "###   deep against covers=0, 24 jobs on 2026-09-07 08:07-08:25."
  echo "###   PASS THE ROOTS -- every root the arms will create, whether or not they exist yet:"
  echo "###     owner_snapshot.sh $JOB_ROOT ${SCRIPTS[*]} --roots <root> [<root>...]"
  echo "###   The roots need not exist at submit: an absent root is ESTIMATED for the wall and"
  echo "###   censused as zero inside the owner (OWNER_CENSUS_ESTIMATE_ABSENT)."
  echo "###   If this caller never submits the owner it generates, say so and it proceeds:"
  echo "###     --allow-underived --reason \"<why this owner needs no wall>\""
  exit 2
fi
if [ "${#ROOTS[@]}" -eq 0 ] && [ -z "$UNDERIVED_REASON" ]; then
  echo "### OWNER SNAPSHOT REFUSED: --allow-underived WITHOUT --reason is still a refusal."
  echo "###   The marker sidecar exists so an underived owner can be traced to the caller that"
  echo "###   asked for one; an empty reason is the unread row this refusal replaced."
  echo "###     --allow-underived --reason \"<why this owner needs no wall>\""
  exit 2
fi

SR=$HERE/../tools/script_refs.sh
[ -f "$SR" ] || SR=$HERE/script_refs.sh
[ -f "$SR" ] || {
  echo "### OWNER SNAPSHOT REFUSED: no script_refs.sh -- the snapshot set would have to be"
  echo "###   guessed, and a guessed set is how the one file the job sources goes missing."
  exit 2; }
# shellcheck disable=SC1090
. "$SR"

SNAP=$JOB_ROOT/owner-snapshot
mkdir -p "$SNAP" || { echo "### OWNER SNAPSHOT REFUSED: cannot create $SNAP"; exit 2; }

# HARNESS-CONSOL-12: the marker sidecar, beside the job root exactly as
# `HARNESS_COMMIT.allow-older` sits beside its pin. An underived owner is
# permitted only when a caller wrote down why, and this file is where that
# sentence survives the job that carried it. Stale markers are removed rather
# than left to authorise a later run that did not ask.
AU_MARK_PATH=$JOB_ROOT/OWNER_SNAPSHOT.allow-underived
if [ "${#ROOTS[@]}" -eq 0 ]; then
  printf 'allow-underived roots=0 job_root=%s at=%s reason=%s\n' \
    "$JOB_ROOT" "$(date -Is)" "$(printf '%s' "$UNDERIVED_REASON" | tr '\n' ' ')" \
    > "$AU_MARK_PATH" || { echo "### OWNER SNAPSHOT REFUSED: cannot write $AU_MARK_PATH"; exit 2; }
else
  rm -f "$AU_MARK_PATH"
fi

########## DET-1-6-a: WHERE THE BYTES CAME FROM, NEVER A GUESS ################
# THE DEFECT, MEASURED. This file used to read `src_commit` out of
# `<job root>/../tools/.harness_synced_commit` and print it as the provenance of
# the frozen copy. That record is the TASK PIN -- what the task tree under
# `tools/` was last synced to -- and it says nothing about the bytes actually
# copied. det16_proof.sh froze the WORKTREE's files (at 6b3b669) while this row
# advertised src_commit=8108ca4, the pin, and the proof author had to add a
# hand-written PROVENANCE NOTE with two md5sums beside the row so a reader could
# not be misled by it. A row that needs a note beside it saying what it really
# means is a defect in the row.
#
# THE FIX IS TO IDENTIFY THE SOURCE, PER FILE, AND TO REFUSE WHEN IT CANNOT BE.
#   * a source directory inside a git repo, with the file TRACKED there:
#     kind=git, sha=`git rev-parse HEAD` of that tree, and its dirty state for
#     that file (a dirty file is NOT the commit it sits on, and saying so is the
#     whole point).
#   * a task copy -- the task tree is not a repo -- : kind=record, sha = the
#     `.harness_synced_commit` beside it, which for a task copy IS the identity
#     of those bytes rather than a pin over somebody else's.
#   * neither: REFUSE. An unidentifiable source is exactly the case the old
#     `src_commit=unknown` printed and carried on, and carrying on is how a
#     frozen copy of nobody-knows-what ends up in a job root for hours.
# The md5 of every frozen file is stamped too, in `md5sum -c` form, so the row
# cannot be misread even when the sha is right: bytes are the fact, the sha is
# the claim about them.
SRC_RECORD=; SRC_RECORD_PATH=
for c in "$JOB_ROOT/../tools/.harness_synced_commit" "$JOB_ROOT/tools/.harness_synced_commit" \
         "$(dirname -- "${SCRIPTS[0]}")/../.harness_synced_commit" \
         "$(dirname -- "${SCRIPTS[0]}")/../tools/.harness_synced_commit"; do
  [ -f "$c" ] && { SRC_RECORD=$(head -c 40 "$c"); SRC_RECORD_PATH=$c; break; }
done

owner_src_identity () {          # $1 = a file about to be frozen; echoes "<kind> <sha> <dirty>"
  local f=$1 d sha st r
  d=$(cd -- "$(dirname -- "$f")" 2>/dev/null && pwd) || { echo "none - -"; return 0; }
  if git -C "$d" rev-parse --git-dir >/dev/null 2>&1 &&
     git -C "$d" ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then
    sha=$(git -C "$d" rev-parse HEAD 2>/dev/null)
    st=$(git -C "$d" status --porcelain -- "$f" 2>/dev/null)
    if [ -n "$sha" ]; then
      if [ -n "$st" ]; then echo "git $sha yes"; else echo "git $sha no"; fi
      return 0
    fi
  fi
  for r in "$d/../tools/.harness_synced_commit" "$d/../.harness_synced_commit" \
           "$d/.harness_synced_commit" "$d/tools/.harness_synced_commit"; do
    [ -f "$r" ] && { echo "record $(head -c 40 "$r") -"; return 0; }
  done
  echo "none - -"
}

PROV=$SNAP/owner-snapshot.provenance
MD5S=$SNAP/owner-snapshot.md5
: > "$PROV"; : > "$MD5S"
SRC_KIND=; SRC_COMMIT=; SRC_DIRTY=
seen=" "
n=0
copy_one () {                    # $1 = absolute path of a script to freeze
  local src=$1 b=${1##*/} rb rp kind sha dirty m
  case "$seen" in *" $b "*) return 0;; esac
  [ -f "$src" ] || { echo "### OWNER SNAPSHOT REFUSED: $src does not exist"; return 2; }
  # IDENTIFY BEFORE COPYING. A copy whose source cannot be named is a copy
  # nobody can trace back, and this file exists to make the frozen bytes
  # traceable.
  read -r kind sha dirty < <(owner_src_identity "$src")
  if [ "$kind" = none ]; then
    echo "### OWNER SNAPSHOT REFUSED: cannot identify the source of $src."
    echo "###   Its directory is not a git worktree that tracks it, and there is no"
    echo "###   .harness_synced_commit beside it. Freezing it would put a copy of"
    echo "###   nobody-knows-what in the job root under a row claiming provenance."
    return 2
  fi
  cp -p "$src" "$SNAP/$b" || { echo "### OWNER SNAPSHOT REFUSED: cannot copy $src"; return 2; }
  m=$(md5sum "$SNAP/$b" | awk '{print $1}')
  printf '%s  %s\n' "$m" "$b" >> "$MD5S"
  printf '%s\t%s\t%s\t%s\t%s\n' "$b" "$m" "$kind" "$sha" "$dirty" >> "$PROV"
  echo "### OWNER SNAPSHOT froze file=$b md5=$m src_kind=$kind src_sha=$sha dirty=$dirty from=$src"
  [ -n "$SRC_KIND" ] || { SRC_KIND=$kind; SRC_COMMIT=$sha; SRC_DIRTY=$dirty; }
  seen="$seen$b "
  n=$((n+1))
  # follow what it reads, with the SAME parser the sync judges with
  # THREE fields since HARNESS-SYNC-7 -- the third is the provenance of the
  # second (`via=A->B`, `via=sibling`, or `-`).  It is read into its own
  # variable and not left to fall into $rp: two fields against a three-field
  # producer puts a literal tab and `via=...` on the end of the PATH, and every
  # `-f "$rp"` then fails, which is a snapshot silently missing files.
  while IFS=$'\t' read -r rb rp rvia; do
    [ -n "${rb:-}" ] || continue
    : "${rvia:=-}"
    if [ "$rb" = '?' ]; then
      echo "### OWNER SNAPSHOT REFUSED: $b contains a reference this parser cannot resolve."
      echo "###   A snapshot with a hole is worse than none: the job would read the LIVE"
      echo "###   path for that one file while every row said it was frozen."
      return 2
    fi
    if [ "${rp:--}" != "-" ] && [ -f "$rp" ]; then
      copy_one "$rp" || return 2
    elif [ -f "$(dirname -- "$src")/$rb" ]; then
      copy_one "$(dirname -- "$src")/$rb" || return 2
    fi
    # a reference to something that is not beside the source and not resolvable
    # to a path is not a task copy (a system tool, a flag) -- refs_of already
    # dropped those; anything left that we cannot FIND is reported, not guessed.
  done < <(refs_of_sibling_resolved "$src")
  return 0
}

FIRST=
for s in "$@"; do
  a=$s
  case "$a" in /*) ;; *) a=$(cd "$(dirname -- "$a")" 2>/dev/null && pwd)/$(basename -- "$a");; esac
  [ -n "$FIRST" ] || FIRST=${a##*/}
  copy_one "$a" || { rm -f "$SNAP/owner.sbatch"; exit 2; }
done

########## CLEANUP-WALL-1: DERIVE THE WALL, HERE, FROM THE DECLARED ROOTS ######
# The submitter derives; the OWNER re-derives (owner_wall_check, shipped into
# the generated sbatch below by `declare -f`, so there is exactly one text).
OWNER_ENTRIES=0; OWNER_PRESENT=0; OWNER_ABSENT=0
OWNER_WALL_S=0; OWNER_CENSUS_ALLOW_S=0; OWNER_WALL_HMS=
if [ "${#ROOTS[@]}" -ge 1 ]; then
  read -r OWNER_ENTRIES OWNER_PRESENT OWNER_ABSENT < <(owner_census "${ROOTS[@]}")
  read -r OWNER_WALL_S OWNER_CENSUS_ALLOW_S < <(owner_wall_derive "$OWNER_ENTRIES")
  OWNER_WALL_HMS=$(owner_wall_hms "$OWNER_WALL_S")
  printf -- '--time=%s\n' "$OWNER_WALL_HMS" > "$SNAP/owner.wall"
fi

# THE GENERATED SBATCH, AND IT CARRIES A LITERAL ABSOLUTE PATH ON PURPOSE.
# Slurm snapshots this top-level script at submit, so its bytes are frozen the
# moment sbatch returns; the path it names is under the JOB ROOT, which nothing
# installs into. Written literally rather than through a variable because
# script_refs.sh resolves a LITERAL absolute token and nothing else -- a
# `$SNAP/cleanup_gated.sh` here would read back as an unresolved path and the
# sync would go on refusing, which is the whole thing this removes.
{
  echo '#!/usr/bin/env bash'
  echo '# GENERATED by phase_template/owner_snapshot.sh -- do not edit.'
  echo "# frozen from src_commit=$SRC_COMMIT (src_kind=$SRC_KIND dirty=$SRC_DIRTY, task pin record=${SRC_RECORD:-none}) at $(date -Is)"
  # A directive, not the last word: a `--time=` on the caller's sbatch line
  # overrides it. It is here so an owner submitted with NO --time still gets the
  # derived one rather than the partition default (5 min, measured with
  # `sinfo -p batch -o %L`), which would kill every reap instantly.
  [ -n "$OWNER_WALL_HMS" ] && echo "#SBATCH --time=$OWNER_WALL_HMS"
  echo 'set -u'
  echo "echo \"### OWNER SNAPSHOT running frozen copy $SNAP/$FIRST src_commit=$SRC_COMMIT src_kind=$SRC_KIND\""
  # --- the re-derivation the owner performs on ITSELF (law 9's actuator) ------
  echo "OWNER_UNLINK_RATE_PER_S=$OWNER_UNLINK_RATE_PER_S"
  echo "OWNER_CENSUS_RATE_PER_S=$OWNER_CENSUS_RATE_PER_S"
  echo "OWNER_CENSUS_WALKS=$OWNER_CENSUS_WALKS"
  echo "OWNER_WALL_MARGIN=$OWNER_WALL_MARGIN"
  echo "OWNER_WALL_FLOOR_S=$OWNER_WALL_FLOOR_S"
  echo "OWNER_WALL_MAX_S=$OWNER_WALL_MAX_S"
  echo "OWNER_ENTRIES_PER_ARM_EST=$OWNER_ENTRIES_PER_ARM_EST"
  # CLEANUP-WALL-2: the owner re-censuses with the same unbounded walk and the
  # same one-release comparison depth the submitter used, or the two numbers
  # that are supposed to be one number are two again.
  echo "OWNER_CENSUS_COMPARE_DEPTH='$OWNER_CENSUS_COMPARE_DEPTH'"
  echo "OWNER_ARMS=$OWNER_ARMS"
  echo "OWNER_WALL_COVERS=$OWNER_ENTRIES"
  echo "OWNER_WALL_S=$OWNER_WALL_S"
  echo "OWNER_SELF=$SNAP/owner.sbatch"
  echo "OWNER_OUT_DIR=$SNAP"
  echo 'OWNER_CONT_N=${OWNER_CONT_N:-0}'
  echo 'OWNER_CONT_MAX=${OWNER_CONT_MAX:-4}'
  # OWNER-EXPORT-1: the declared name set travels as a LITERAL beside the
  # function, so the continuation exports what the submitter declared and not
  # whatever a later edit of owner_export.sh happens to say.
  echo "OWNER_EXPORT_VARS='$OWNER_EXPORT_VARS'"
  declare -f owner_export_clause
  # CLEANUP-WALL-3: inside the owner an absent root is an absent tree, not an
  # estimate (the estimate exists only for the submit, before the tree does).
  echo 'OWNER_CENSUS_ESTIMATE_ABSENT=0'
  echo "OWNER_WALL_PRESSURE_NUM=$OWNER_WALL_PRESSURE_NUM"
  echo "OWNER_WALL_PRESSURE_DEN=$OWNER_WALL_PRESSURE_DEN"
  echo 'OWNER_CONT_PARENT=${OWNER_CONT_PARENT:-0}'
  echo 'OWNER_CONT_PARENT_REMOVED=${OWNER_CONT_PARENT_REMOVED:-0}'
  echo 'echo "### CONTINUATION depth=$OWNER_CONT_N parent=$OWNER_CONT_PARENT parent_removed=$OWNER_CONT_PARENT_REMOVED"'
  declare -f owner_census owner_wall_derive owner_wall_hms owner_wall_census_row owner_continue_check
  # CLEANUP-WALL-3: MEASURE, RUN, THEN DECIDE -- in that order. This used to be
  # `owner_wall_check "$@"` followed by `exec bash <gate>`, which put the
  # continuation decision BEFORE the only thing that could inform it. The gate's
  # rc is preserved to the letter: a refusing owner still exits 2, it just no
  # longer spawns a successor to refuse again.
  echo 'owner_wall_census_row "$@"'
  echo "OWNER_PASS_LOG=\$OWNER_OUT_DIR/owner-pass-\${SLURM_JOB_ID:-0}.log"
  echo "bash $SNAP/$FIRST \"\$@\" 2>&1 | tee \"\$OWNER_PASS_LOG\""
  echo 'OWNER_PASS_RC=${PIPESTATUS[0]}'
  echo 'owner_continue_check "$OWNER_PASS_RC" "$OWNER_PASS_LOG" "$@"'
  echo 'exit "$OWNER_PASS_RC"'
} > "$SNAP/owner.sbatch" || { echo "### OWNER SNAPSHOT REFUSED: cannot write $SNAP/owner.sbatch"; exit 2; }
chmod +x "$SNAP/owner.sbatch"

echo "### OWNER SNAPSHOT files=$n root=$JOB_ROOT src_commit=$SRC_COMMIT src_kind=$SRC_KIND src_dirty=$SRC_DIRTY task_pin_record=${SRC_RECORD:-none}${SRC_RECORD_PATH:+ ($SRC_RECORD_PATH)}"
if [ "${#ROOTS[@]}" -ge 1 ]; then
  echo "### OWNER SNAPSHOT wall=$OWNER_WALL_S (--time=$OWNER_WALL_HMS) from entries=$OWNER_ENTRIES rate=$OWNER_UNLINK_RATE_PER_S margin=$OWNER_WALL_MARGIN census_allow=$OWNER_CENSUS_ALLOW_S roots=${#ROOTS[@]} present=$OWNER_PRESENT absent=$OWNER_ABSENT arms=$OWNER_ARMS"
  echo "### OWNER SNAPSHOT wall file: $SNAP/owner.wall ($(cat "$SNAP/owner.wall"))"
else
  echo "### OWNER SNAPSHOT wall=UNDERIVED roots=0 AUTHORISED reason=$UNDERIVED_REASON"
  echo "###   marker: $AU_MARK_PATH"
  echo "###   This owner keeps whatever --time the submit site types by hand, and it will run"
  echo "###   its pass but REFUSE to continue itself (CLEANUP-WALL-3: it cannot be short of a"
  echo "###   wall it never had). Pass --roots <root>... and the wall is derived instead."
fi
echo "### OWNER SNAPSHOT frozen: $(cd "$SNAP" && ls -1 *.sh 2>/dev/null | tr '\n' ' ')"
echo "### OWNER SNAPSHOT md5s: $MD5S ($(wc -l < "$MD5S") file(s)); provenance: $PROV"
echo "### OWNER SNAPSHOT sbatch: $SNAP/owner.sbatch -> $SNAP/$FIRST"
exit 0
