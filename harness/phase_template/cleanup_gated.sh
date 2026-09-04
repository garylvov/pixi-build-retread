#!/usr/bin/env bash
# HANDOFF §2 (coordinator ruling 2026-09-03, inode quota 75.6M rising ~2.76M/h):
# every relock/proof job submits an `afterany` cleanup for its OWN workspace and
# job-scoped cache root -- but only once the artifacts are safely in the task
# root, and never a root a running phase-2 still consumes.
#
# This is a GATE in front of tools/phase_template/cleanup.sh, not a replacement:
# the deletion itself, and the containment refusals that make it safe (path
# under /oscar/data/stellex/glvov/retread with a cert*/ws.* basename, or a
# per-arm isolated cache root .../agrescap/cache/retread-injection-on-<tag>,
# with the persistent .../agrescap/cache/retread refused by name), stay in that
# file. What is added here is the three conditions the ruling names.
#
#   1. THE ARTIFACTS ARE IN THE TASK ROOT. `$D/artifacts/<TAG>-<J>*.rc`,
#      `*.wall`, `*.lock.log` must all exist and be non-empty, and when the run
#      locked green (`rc` = 0) a certified lock must exist too. A run whose
#      artifacts never landed is a run whose evidence is still only in the
#      workspace, and deleting the workspace destroys it.
#   2. NO OTHER JOB OF OURS IS STILL RUNNING AGAINST THESE ROOTS. The relock job
#      id must appear in the root basename as a `-<jid>` token -- that is the
#      OWNERSHIP proof, and it is still required -- and no job id named anywhere
#      in the basename may be RUNNING/PENDING in `squeue`. (`afterany` already
#      waits for the producing job, so this catches a phase-2 or a sibling arm
#      that adopted the same root, which is the shape that took `ws.A3B-5697522`
#      out from under the A-final cert.)
#   3. NOTHING IS DELETED ON A REFUSAL. Exit 2, print why, leave every byte.
#
# 2026-09-04 ROOT FIX (inode sweep 2, jobs 5764454/5764455 deleted NOTHING).
# Two defects, both of them the same mistake -- treating a NAME SHAPE as the
# ownership proof instead of the job-id TOKEN inside it:
#
#   (a) the job id was read as `${r##*-}`, i.e. the LAST dash-separated field,
#       so a root had to END in its job id. Every root the oncert lanes mint
#       ends in `-ONCERT` (`certO7P6UA-5764452-ONCERT`, `ws.O7P6UA-5764452-ONCERT`),
#       and every per-arm cache root ends in an arm tag with no job id at all
#       (`retread-injection-on-o7p6ua`), so the whole class was permanently
#       un-reapable: `REFUSE: root ... does not end in a job id`. The id is now
#       required as a `-<jid>` TOKEN ANYWHERE in the basename, and every such
#       token found is queue-checked, not just the last one. The per-arm cache
#       class is exempt from the token test by construction and is proved
#       instead by its own prefix+pattern and by the relock job's terminality.
#   (b) the evidence gate looked for the exact name `<TAG>-<J>.rc` while the
#       oncert lanes write `<TAG>-<J>-ONCERT.rc`, so it reported MISSING
#       evidence that was sitting in the artifacts directory. The gate now
#       accepts `<TAG>-<J>*.rc` (and `.gz`, merge-h's 2026-09-04 gzip fix).
#
#   usage: env -u SLURM_JOB_ID sbatch --partition=batch --qos=normal \
#            --cpus-per-task=1 --mem=4G --time=16:00:00 \
#            --dependency=afterany:<relock job>:<cert job> \
#            --export=ALL,D=<harness dir>,TAG=<tag>,RJ=<relock job> \
#            --wrap 'bash <this file> <root> [<root> ...]'
#
#   afterany on BOTH phases, never afterok on the cert: a relock that fails its
#   own lock leaves a cert that Slurm cancels, and a cleanup chained behind the
#   cert alone then never runs at all. That is how C18A/C18B (job 5759225) were
#   stranded. See phase_template/README.md "Hazard 2".
set -uo pipefail
T=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11
CLEANUP=$(dirname "$0")/cleanup.sh
[ -f "$CLEANUP" ] || CLEANUP=$T/tools/phase_template/cleanup.sh
PERSISTENT_CACHE=/oscar/data/stellex/glvov/agrescap/cache/retread
ISO_CACHE_PREFIX=/oscar/data/stellex/glvov/agrescap/cache/retread-injection-on-
: "${D:?set D to the harness directory whose artifacts/ must already hold the evidence}"
: "${TAG:?set TAG}"
: "${RJ:?set RJ to the relock job id these roots belong to}"
case "$RJ" in ''|*[!0-9]*) echo "### REFUSE: RJ='$RJ' is not a job id"; exit 2;; esac
A=$D/artifacts
hostname; date -Is
echo "### CLEANUP GATE tag=$TAG relock_job=$RJ roots=$* "
fail=0

# --- condition 1: the evidence is in the task root ----------------------------
# A harness that GZIPS its lock log into the task root satisfies condition 1
# just as well as one that leaves it plain -- the evidence is in the task root
# either way. Before 2026-09-04 this loop demanded the plain name only, so any
# harness that gzips (p6m-b's does, and it is the shape the phase template
# has produced since the raw logs got large) could NEVER pass the gate and its
# roots could never be reclaimed. Accept `<name>` or `<name>.gz`.
# It also demanded the EXACT stem `<TAG>-<RJ>`, which no oncert lane writes --
# they all carry an arm suffix (`-ONCERT`, `-OFF`). The stem is a PREFIX now.
RCFILE=
for suffix in rc wall lock.log; do
  hit=
  for f in "$A/$TAG-$RJ"*".$suffix" "$A/$TAG-$RJ"*".$suffix.gz"; do
    [ -s "$f" ] || continue
    hit=$f
    case "$f" in
      *.gz) echo "### artifact present (gzipped): $f ($(stat -c%s "$f") B)";;
      *)    echo "### artifact present: $f ($(stat -c%s "$f") B)";;
    esac
    [ "$suffix" = rc ] && [ -z "$RCFILE" ] && case "$f" in *.gz) ;; *) RCFILE=$f;; esac
  done
  if [ -z "$hit" ]; then
    echo "### MISSING/EMPTY artifact: no $A/$TAG-$RJ*.$suffix (nor .gz)"; fail=1
  fi
done

LRC=$(cat "$RCFILE" 2>/dev/null | tr -d '[:space:]' || echo missing)
[ -n "$LRC" ] || LRC=missing
echo "### recorded lock rc=$LRC (from ${RCFILE:-<none>})"
if [ "$LRC" = 0 ]; then
  lockhit=
  for f in "$A/pixi.lock.cert" "$A/pixi.lock.$TAG-$RJ"*; do
    [ -s "$f" ] || continue
    lockhit=$f
    echo "### lock present: $f ($(stat -c%s "$f") B) md5 $(md5sum "$f" | awk '{print $1}')"
  done
  if [ -z "$lockhit" ]; then
    echo "### MISSING: a green run with no pixi.lock.cert and no pixi.lock.$TAG-$RJ* in the task root"; fail=1
  fi
fi

# --- condition 2: ownership, and nothing of ours still running on these roots --
for r in "$@"; do
  [ -n "$r" ] || continue
  base=${r##*/}
  case "${r%/}" in
    "$PERSISTENT_CACHE")
      echo "### REFUSE: $r is the PERSISTENT shared cache -- never a cleanup root"; fail=1; continue;;
  esac
  case "$r" in
    "$ISO_CACHE_PREFIX"?*)
      # A per-arm isolated cache root carries the ARM tag, never a job id. Its
      # ownership proof is its own name shape plus the terminality of RJ, which
      # is checked below with every other job id of this batch.
      echo "### root $r: per-arm isolated cache (no job-id token by construction); owner is relock job $RJ"
      jids=$RJ
      ;;
    *)
      # The ownership proof: the relock job id, as a `-<jid>` token, ANYWHERE in
      # the basename -- `certO7P6UA-5764452-ONCERT` proves 5764452 owns it just
      # as well as `certC18P1-5763080` proves 5763080 does.
      case "-$base-" in
        *"-$RJ-"*) ;;
        *) echo "### REFUSE: root $r does not carry relock job $RJ as a -<jid> token in its basename"; fail=1; continue;;
      esac
      jids=$(printf '%s\n' "$base" | grep -oE -- '-[0-9]{6,}' | tr -d - | sort -u)
      if [ -z "$jids" ]; then
        echo "### REFUSE: root $r has no -<jobid> token in its basename"; fail=1; continue
      fi
      ;;
  esac
  for jid in $jids; do
    st=$(squeue -j "$jid" -h -o '%t' 2>/dev/null | paste -sd, )
    if [ -n "$st" ]; then
      echo "### REFUSE: job $jid (named by root $r) is still in the queue: $st"; fail=1
    else
      echo "### root $r: its job $jid is no longer in the queue"
    fi
  done
done

if [ "$fail" -ne 0 ]; then
  echo "### CLEANUP REFUSED -- nothing deleted. Roots kept: $*"
  exit 2
fi

echo "### GATE PASSED -- handing $# root(s) to $CLEANUP"
exec bash "$CLEANUP" "$@"
