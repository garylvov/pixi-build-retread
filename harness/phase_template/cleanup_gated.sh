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
# 2026-09-04 ROOT FIX (jobs 5814670 and 5823482 refused, roots kept). A THIRD
# instance of the same mistake, this time in the LOCK half of condition 1:
#
#   (c) the green-run lock check looked for `pixi.lock.cert` and
#       `pixi.lock.<TAG>-<J>*` only -- the word `pixi` FIRST in both. Eight
#       harnesses of this campaign (c17c, c17w, c18a, c18b, c18c, c18p1, c18p2,
#       c21c) write the certified lock the other way round, with the tag+job
#       stem first: `<TAG>-<J>.pixi.lock.cert`. The gate could not see a lock
#       that was sitting in the artifacts directory and refused every one of
#       those runs. Measured on cleanup jobs 5814670 and 5823482, which both
#       printed `### MISSING: a green run with no pixi.lock.cert and no
#       pixi.lock.C21C-<J>* in the task root` while
#       `artifacts/C21C-5814669.pixi.lock.cert` (2758192 B) and
#       `artifacts/C21C-5823481.pixi.lock.cert` (2758170 B) existed and were
#       non-empty -- so ws.C21C-5814669, certC21C-5814669, ws.C21C-5823481 and
#       certC21C-5823481 were all stranded. Both refusals were CORRECT given
#       what the gate could see, and deleted nothing; the defect is that the
#       evidence was invisible to it. The stem-first shape
#       `<TAG>-<J>*pixi.lock.cert` is now accepted alongside the two it already
#       took. A green run with NO lock in ANY of the three shapes still refuses.
#       Reader: cleanup_lock_evidence_guard.sh.
#
#   usage: env -u SLURM_JOB_ID sbatch --partition=batch --qos=normal \
#            --cpus-per-task=1 --mem=4G --time=16:00:00 \
#            --dependency=afterany:<relock job>:<cert job> \
#            --export=ALL,D=<harness dir>,TAG=<tag>,RJ=<relock job> \
#            --wrap 'bash <this file> <root> [<root> ...]'
#
#   Since 2026-09-05 the three variables are DERIVED from the root arguments
#   when they are not exported -- see ROOT FIX (d) below -- so the `--export`
#   clause is now a way to OVERRIDE the derivation, not a way to make the gate
#   run at all. A `--wrap` with no `--export` is no longer a silent death.
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
# 2026-09-05 ROOT FIX (p6ad-4). A FOURTH instance of the same family, and this
# one is in the SUBMIT contract rather than in a name shape.
#
#   (d) `D`, `TAG` and `RJ` arrived ONLY through `--export`, and a `--wrap`
#       without that clause died on bash's own `${D:?...}` at line 85 -- before
#       the gate printed a single row, before the roots were even parsed, with a
#       message that names the variable and not the fix. TWO LANES HIT IT ON THE
#       SAME NIGHT: B-cert-4's cleanup 5841112 (boarded C31-2) and p6ad-4's
#       5879244, which stranded certP6AD4-5879243-{A,B} and ws.P6AD4-5879243-{A,B}.
#       Two independent lanes failing the same way on the same contract is a
#       harness defect, not two operator errors.
#
#       THE THREE VALUES ARE ALREADY IN THE ARGUMENTS. Every root this gate
#       accepts is `cert<TAG>-<JID>[-<arm>]` or `ws.<TAG>-<JID>[-<arm>]` -- that
#       is condition 2's OWNERSHIP proof, so the basename that proves ownership
#       also NAMES the tag and the job. `D` is the harness directory whose
#       `artifacts/` holds `<TAG>-<RJ>*`, and there is exactly one of those.
#       So an unset variable is now DERIVED and the derivation is PRINTED; an
#       explicitly exported value always wins, and nothing about the three
#       conditions below changes. If a value cannot be derived the gate still
#       refuses -- but it refuses with the exact `--export` clause to add, not
#       with a bash parameter-expansion error.
#
#       Reader: cleanup_gate_env_derivation_guard.sh, which runs this file with
#       no D/TAG/RJ at all and requires the derivation, runs it against a root
#       naming a harness that does not exist and requires the printed export
#       line, and runs the PREVIOUS version of the same file to show it dying at
#       line 85 -- the mutation that proves the guard can fail.
derive_from_roots() {
  # First root that parses wins; every root is required to agree later anyway,
  # because condition 2 queue-checks every job-id token it finds in each one.
  local r base rest
  for r in "$@"; do
    base=${r##*/}
    case "$base" in
      cert*) rest=${base#cert};;
      ws.*)  rest=${base#ws.};;
      *) continue;;
    esac
    # `<TAG>-<JID>` with an optional `-<arm>` tail: take the FIRST all-digit
    # dash-token as the job id and everything before it as the tag.
    local field i n tag="" jid=""
    n=$(awk -F- '{print NF}' <<<"$rest")
    for i in $(seq 2 "$n"); do
      field=$(awk -F- -v k="$i" '{print $k}' <<<"$rest")
      case "$field" in
        ''|*[!0-9]*) ;;
        *) jid=$field; tag=$(awk -F- -v k="$i" '{s=$1; for(j=2;j<k;j++) s=s"-"$j; print s}' <<<"$rest"); break;;
      esac
    done
    [ -n "$jid" ] || continue
    DERIVED_TAG=$tag; DERIVED_RJ=$jid; DERIVED_FROM=$r
    return 0
  done
  return 1
}
derive_harness_dir() {
  # The unique directory under the task root whose artifacts/ already holds the
  # evidence for this tag+job. `find -maxdepth` per HANDOFF section 2; a
  # full-tree walk here would cross every job root under the task directory.
  local tag=$1 rj=$2 hits
  hits=$(find "$T" -maxdepth 3 -type f -path "*/artifacts/$tag-$rj*" -printf '%h\n' 2>/dev/null | sort -u)
  [ "$(printf '%s\n' "$hits" | grep -c .)" = 1 ] || return 1
  printf '%s\n' "${hits%/artifacts}"
}
if [ -z "${D:-}" ] || [ -z "${TAG:-}" ] || [ -z "${RJ:-}" ]; then
  DERIVED_TAG=; DERIVED_RJ=; DERIVED_FROM=
  if derive_from_roots "$@"; then
    [ -n "${TAG:-}" ] || { TAG=$DERIVED_TAG; echo "### DERIVED TAG=$TAG from root $DERIVED_FROM"; }
    [ -n "${RJ:-}"  ] || { RJ=$DERIVED_RJ;  echo "### DERIVED RJ=$RJ from root $DERIVED_FROM"; }
  fi
  if [ -z "${D:-}" ] && [ -n "${TAG:-}" ] && [ -n "${RJ:-}" ]; then
    D=$(derive_harness_dir "$TAG" "$RJ") \
      && echo "### DERIVED D=$D (the one directory under $T whose artifacts/ holds $TAG-$RJ*)" \
      || D=
  fi
fi
if [ -z "${D:-}" ] || [ -z "${TAG:-}" ] || [ -z "${RJ:-}" ]; then
  echo "### REFUSE: could not derive the submit contract, and nothing was deleted."
  echo "###   D='${D:-<unset>}' TAG='${TAG:-<unset>}' RJ='${RJ:-<unset>}'"
  echo "###   Roots given: $*"
  echo "###   Add this to the sbatch line and resubmit (fill in what is <unset>):"
  echo "###     --export=ALL,D=<harness dir>,TAG=${TAG:-<tag>},RJ=${RJ:-<relock job>}"
  echo "###   D is the directory whose artifacts/ already holds ${TAG:-<tag>}-${RJ:-<job>}*.rc/.wall/.lock.log."
  exit 2
fi
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
  # Three accepted shapes, all of them a certified lock in the task root, and
  # nothing else. The third is the STEM-FIRST one (`<TAG>-<J>.pixi.lock.cert`,
  # and with an arm suffix `<TAG>-<J>-ONCERT.pixi.lock.cert`) written by c17c,
  # c17w, c18a/b/c/p1/p2 and c21c -- see defect (c) in the header. A run with a
  # lock in none of them is still MISSING and still refuses.
  lockhit=
  for f in "$A/pixi.lock.cert" "$A/pixi.lock.$TAG-$RJ"* "$A/$TAG-$RJ"*"pixi.lock.cert"; do
    [ -s "$f" ] || continue
    lockhit=$f
    echo "### lock present: $f ($(stat -c%s "$f") B) md5 $(md5sum "$f" | awk '{print $1}')"
  done
  if [ -z "$lockhit" ]; then
    echo "### MISSING: a green run with no pixi.lock.cert, no pixi.lock.$TAG-$RJ* and no $TAG-$RJ*pixi.lock.cert in the task root"; fail=1
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
      # A TWO-PHASE chain owns a root the relock job did NOT name: the cert
      # phase mints `cert<TAG>P2-<CERT JOB>`, whose only job-id token is the
      # CERT job. Demanding `-$RJ-` on every root refuses that set outright and
      # deletes nothing, so every two-phase cert leaks all three of its roots --
      # measured on job 5768460, which refused `certP6MBCP2-5768459` and kept
      # `certP6MBC-5768458` and `ws.P6MBC-5768458` with it. `OJ` names the
      # ADDITIONAL owner job ids of the same chain (space-separated, optional,
      # empty for every existing caller, so nothing else changes). Ownership is
      # still proved by a job-id token, and terminality is still checked below
      # for every id the basename carries -- OJ widens WHOSE token counts, it
      # does not skip the check.
      owner_hit=0
      for oj in $RJ ${OJ:-}; do
        case "-$base-" in *"-$oj-"*) owner_hit=1;; esac
      done
      if [ "$owner_hit" != 1 ]; then
        echo "### REFUSE: root $r carries none of the owner job ids ($RJ ${OJ:-}) as a -<jid> token in its basename"; fail=1; continue
      fi
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
