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
#   4. A ROOT THAT DOES NOT EXIST IS NOT A ROOT THAT WAS KEPT (MERGE-N-1,
#      2026-09-06). The refusal used to end `Roots kept: $*` over the ARGUMENT
#      LIST, so a gate that refused on missing evidence reported roots it had
#      never seen on disk as still occupying quota -- and a lane reading that row
#      goes looking for bytes that are not there. Every root is now classified
#      PRESENT or ABSENT before any condition runs. An absent root is announced,
#      is exempt from the ownership and queue checks (there is nothing to delete,
#      so there is nothing to own), and NEVER sets `fail`; a present root behaves
#      exactly as before. The refusal prints the two lists separately, and a call
#      in which EVERY root is absent is a no-op that exits 0 rather than handing
#      `cleanup.sh` a list of names.
#   5. A JOB THE PREAMBLE REFUSED BEFORE ARM 1 HAS NO EVIDENCE AND NEVER WILL
#      (CLEANUP-SEAM-1, 2026-09-06). Condition 1 waits for artifacts a job that
#      ran no arm cannot write, so it kept `certDET141-6000903` and
#      `ws.DET141-6000903` -- six empty directories -- forever. The branch above
#      the conditions is `setup_refused_check`; the full reasoning sits at it.
#   6. A JOB THAT DIED MID-ARM HAS NO EVIDENCE AND NEVER WILL EITHER
#      (CLEANUP-SEAM-2, 2026-09-07). Rule 5's branch requires EMPTY roots, which
#      is right for a preamble refusal and wrong for the next case along: job
#      6014471 staged, ran two arms, and both wrappers exited 7, so its roots
#      hold a store AND no `.rc`/`.wall`/`.lock.log` will ever be written. Its
#      owner 6014484 refused rc 2 and would have forever. The second branch is
#      `job_fatal_check`: a job-fatal row in the job's OWN stdout, sacct FAILED
#      or TIMEOUT for that job, nothing of ours still queued on the roots, and
#      no sealed subtree. The full reasoning sits at it.
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
#            --export=D=<harness-dir>,TAG=<tag>,RJ=<relock-job>,DRY_RUN=0,PATH=$PATH,HOME=$HOME \
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
#       line, runs it over a harness nested two deep and requires D anyway
#       (SWEEP-3-1), runs it over TWO harnesses holding the same evidence and
#       requires a refusal that names both, and runs PINNED PREVIOUS versions of
#       this same file -- ececead for the derivation, efe74a0 for the depth --
#       to show each defect reproducing. Those mutations are pinned to CONSTANTS,
#       not to HEAD: an arm that extracts `HEAD:<itself>` asserts against the fix
#       the moment the fix is committed, which is how arm B silently died between
#       d2ba3fd and 5891315 (13/2, not the 15/15 it was signed off at).
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
  #
  # SWEEP-3-1 (2026-09-05). The depth was 3, which reaches exactly one shape:
  # `<T>/<harness>/artifacts/<file>`. Merge and sweep lanes routinely nest the
  # harness one level deeper -- `<T>/c2-merged/a/artifacts/<file>`, a
  # per-candidate subdirectory under a batch directory -- and at depth 3 `find`
  # never sees the file, so `derive_harness_dir` returned 1 and the gate refused
  # with D unset. FOUR lanes with COMPLETE evidence were refused that way and
  # their roots could not be reclaimed. Depth 4 reaches both shapes.
  #
  # It is not widened past 4 on purpose: `<T>/<batch>/<cand>/artifacts/` is the
  # deepest layout any harness of this campaign writes, while depth 5 would start
  # matching artifacts trees COPIED inside a job root, and the uniqueness rule
  # below would then refuse cases that work today. Depth is a bound on the shapes
  # we accept, not a search budget.
  #
  # UNIQUENESS IS THE WHOLE POINT and it survives the widening: two candidate
  # directories is a REFUSAL that names both, never a guess. Naming one of two
  # would point this gate -- and `cleanup.sh` behind it -- at another lane's
  # evidence and delete another lane's roots. The list goes to stderr because
  # this function's stdout IS the derived value.
  local tag=$1 rj=$2 hits n
  hits=$(find "$T" -maxdepth 4 -type f -path "*/artifacts/$tag-$rj*" -printf '%h\n' 2>/dev/null | sort -u)
  n=$(printf '%s\n' "$hits" | grep -c .)
  if [ "$n" != 1 ]; then
    if [ "$n" -gt 1 ]; then
      echo "### AMBIGUOUS D: $n directories under $T hold $tag-$rj* -- naming NONE of them:" >&2
      printf '%s\n' "$hits" | sed 's|/artifacts$||; s|^|###   candidate: |' >&2
      echo "###   Pick one and pass it: --export=D=<harness-dir>,TAG=$tag,RJ=$rj,DRY_RUN=0,PATH=\$PATH,HOME=\$HOME" >&2
      echo "###   NO \`ALL\` in that clause -- OWNER-EXPORT-1: \`--export=ALL,...\` held 6013350/6013351/6014485/5841188." >&2
    fi
    return 1
  fi
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
  echo "###     --export=D=<harness-dir>,TAG=${TAG:-<tag>},RJ=${RJ:-<relock-job>},DRY_RUN=0,PATH=\$PATH,HOME=\$HOME"
  echo "###   NO \`ALL\` in that clause. \`--export=ALL,...\` makes Slurm retrieve the submitter's"
  echo "###   environment on the target node, and a failed retrieval HOLDS the job instead of"
  echo "###   refusing it: 6013350, 6013351, 6014485 and 5841188 all sat in \`user env retrieval"
  echo "###   failed requeued held\` (OWNER-EXPORT-1 / REAP-3). tools/owner_export.sh is the producer."
  echo "###   D is the directory whose artifacts/ already holds ${TAG:-<tag>}-${RJ:-<job>}*.rc/.wall/.lock.log."
  exit 2
fi
case "$RJ" in ''|*[!0-9]*) echo "### REFUSE: RJ='$RJ' is not a job id"; exit 2;; esac
A=$D/artifacts
hostname; date -Is
echo "### CLEANUP GATE tag=$TAG relock_job=$RJ roots=$* "
fail=0

# --- condition 0: which of these roots exist at all (MERGE-N-1) ---------------
PRESENT_ROOTS=(); ABSENT_ROOTS=()
for r in "$@"; do
  [ -n "$r" ] || continue
  if [ -e "$r" ]; then
    PRESENT_ROOTS+=("$r"); echo "### root $r: PRESENT on disk"
  else
    ABSENT_ROOTS+=("$r"); echo "### root $r: ABSENT -- it does not exist, so there is nothing to delete and nothing to keep (exit 0 for this root)"
  fi
done
echo "### ROOT CENSUS present=${#PRESENT_ROOTS[@]} absent=${#ABSENT_ROOTS[@]}"

# --- MERGE-T-2: ALL-ABSENT IS A NO-OP, AND IT HAS TO BE DECIDED HERE ---------
# MEASURED: job 5981195 printed `### ROOT CENSUS present=0 absent=2` and then
# `### CLEANUP REFUSED`, exit 2. The no-op branch existed -- it is the
# `${#PRESENT_ROOTS[@]} -eq 0` block at the bottom -- but it sat BELOW the
# `fail` check, so conditions 1 and 2 got to vote first: with every root already
# gone, a missing `.rc`/`.wall`/`.lock.log` artifact set `fail=1` and the run
# refused to do the nothing it had to do. A cleanup owner released on `afterany`
# for a chain that never created its roots (or whose roots were already reaped)
# then exits non-zero for the rest of time, and every watcher reads a terminal
# non-zero that means nothing.
#
# MERGE-N-1's rule is that an absent root is "nothing to delete and nothing to
# keep". With NO present roots there is nothing for the evidence gate to protect
# -- the gate exists to stop a root being deleted before its evidence is in the
# task root, and there is no root. So the decision moves ABOVE the conditions.
# The refusal is untouched for the case it was written for: one or more roots
# PRESENT without their evidence.
if [ "${#PRESENT_ROOTS[@]}" -eq 0 ]; then
  echo "### NOTHING TO DO -- every root named is ABSENT: ${ABSENT_ROOTS[*]:-<none>}."
  echo "###   Nothing to delete and nothing to keep (MERGE-N-1), so the evidence"
  echo "###   conditions have nothing to protect and are not run. Exiting 0 without"
  echo "###   calling $CLEANUP."
  exit 0
fi

# --- CLEANUP-SEAM-1: A JOB THE PREAMBLE REFUSED BEFORE ARM 1 -----------------
# MEASURED on det141-cleanup 6000916, the afterany owner of det141-proof 6000903.
# 6000903 was refused by `multiarm_preamble.sh` in ELEVEN SECONDS -- its own
# smoke's store root composed past the 256-byte pad -- so NO ARM RAN, and a job
# that never ran an arm never writes `<TAG>-<J>.rc`, `.wall` or `.lock.log`.
# Condition 1 then reported `### MISSING/EMPTY artifact:` three times, `###
# recorded lock rc=missing (from <none>)`, and `### CLEANUP REFUSED -- nothing
# deleted`, keeping `certDET141-6000903` and `ws.DET141-6000903`: SIX EMPTY
# DIRECTORIES, no data, no lock, nothing to protect. And it would refuse them
# again for the rest of time, because the evidence it waits for can never
# arrive.
#
# THIS IS A SEAM DEFECT IN A PAIR, NOT A BUG IN EITHER FILE. The preamble's
# SUCCESS CASE is refusing early and cheaply (that is the whole point of the
# smoke); this gate's evidence condition assumes a run that got far enough to
# produce evidence. Every future preamble refusal strands its roots the same way.
#
# THE BRANCH, and it sits ABOVE the conditions for the same reason MERGE-T-2's
# all-absent no-op does -- with nothing to protect, the evidence conditions get
# no vote:
#   (i)   the relock job's OWN STDOUT under $D carries a preamble refusal row,
#   (ii)  every present root holds NO file at all within the depth bound, and
#   (iii) no directory under it is SEALED (write stripped -- the shape
#         `multiarm_store_reap.sh` renames aside rather than deletes; there the
#         proof is dev:inode containment plus write-stripped directories, and
#         the containment half is `cleanup.sh`'s job on the line below).
# Then the roots are removed THROUGH `cleanup.sh`, which is where the
# containment refusals live -- this branch decides, it does not delete.
#
# A SEALED TREE IS THE REFUSAL THIS BRANCH KEEPS. Write-stripped directories
# mean a rattler-build store really provisioned under that root; a run that got
# that far is not a run the preamble refused before arm 1, whatever its stdout
# says, and it refuses exactly as it does today.
#
# Reader: cleanup_absent_root_guard.sh, arms S1-S4.
SETUP_REFUSED_RE='^### (PREAMBLE JOB REFUSED BEFORE ARM 1|PREAMBLE FATAL:|SMOKE SETUP_FAILED)'
SETUP_REFUSED_DEPTH=8

JOB_STDOUT_FALLBACK_DEPTH=3

job_stdout_files () {
  # The relock job's own stdout, found by its JOB ID -- `find -maxdepth` per
  # HANDOFF section 2, never a full-tree walk. Job names differ per lane
  # (`$D/logs/det141-6000903.out`), the job id does not.
  #
  # CLEANUP-SEAM-2 (2026-09-07). This searched `$D` at depth 2 and nothing else,
  # which assumes the harness dir and the job's log live together. det161b
  # splits them: `D` derives to the PER-ARM root `det161b-w1` (its `artifacts/`
  # is what holds `<TAG>-<RJ>*`, so that is the directory `derive_harness_dir`
  # is right to name), while the driver's stdout is `det161b-work/logs/
  # det161b-6014471.out`. The finder returned NOTHING for job 6014471, so any
  # branch keyed on the job's own stdout was silent on it. `$D` is still tried
  # FIRST and still wins, so nothing changes for a lane whose log is where it
  # always was; only when `$D` yields nothing does the search widen to the TASK
  # ROOT at depth 3, which is the shallowest depth that reaches
  # `<T>/<lane>-work/logs/<name>-<jid>.out`. A widening is ANNOUNCED, and the
  # JOB-FATAL branch below announces an EMPTY result too: a branch that says
  # nothing because it found no file is indistinguishable from one that says
  # nothing because the job was healthy, and that silence is what this note and
  # that row exist to make readable.
  local hits
  hits=$(find "$D" -maxdepth 2 -type f -name "*$RJ*.out" 2>/dev/null | sort)
  if [ -z "$hits" ]; then
    hits=$(find "$T" -maxdepth "$JOB_STDOUT_FALLBACK_DEPTH" -type f -name "*$RJ*.out" 2>/dev/null | sort)
    [ -z "$hits" ] || echo "### JOB STDOUT: none under D=$D at depth 2; fell back to $T at depth $JOB_STDOUT_FALLBACK_DEPTH and found $(printf '%s\n' "$hits" | grep -c .) file(s)" >&2
  fi
  printf '%s\n' "$hits"
}

setup_refused_check () {
  local outs row r n files sealed deep removed=0 crc
  outs=$(job_stdout_files)
  [ -n "$outs" ] || return 0
  row=$(printf '%s\n' "$outs" | while IFS= read -r f; do
          grep -m1 -E "$SETUP_REFUSED_RE" "$f" 2>/dev/null && break
        done)
  [ -n "$row" ] || return 0
  echo "### SETUP-REFUSED: the relock job $RJ refused itself before arm 1. Its own stdout says:"
  echo "###   $row"
  # Every present root must be provably empty AND unsealed, or this branch does
  # not fire at all and the refusal below stands.
  for r in "${PRESENT_ROOTS[@]+"${PRESENT_ROOTS[@]}"}"; do
    deep=$(find "$r" -mindepth "$SETUP_REFUSED_DEPTH" -maxdepth "$SETUP_REFUSED_DEPTH" 2>/dev/null | head -1)
    if [ -n "$deep" ]; then
      echo "### SETUP-REFUSED NOT TAKEN: $r still has entries at depth $SETUP_REFUSED_DEPTH ($deep) -- emptiness cannot be proved inside the depth bound, so the evidence conditions decide."
      return 0
    fi
    files=$(find "$r" -maxdepth "$SETUP_REFUSED_DEPTH" ! -type d 2>/dev/null | head -5)
    sealed=$(find "$r" -maxdepth "$SETUP_REFUSED_DEPTH" -type d ! -writable 2>/dev/null | head -5)
    n=$(find "$r" -maxdepth "$SETUP_REFUSED_DEPTH" -type d 2>/dev/null | wc -l)
    echo "### SETUP-REFUSED root $r: dirs=$n files=$(printf '%s\n' "$files" | grep -c .) sealed_dirs=$(printf '%s\n' "$sealed" | grep -c .)"
    if [ -n "$files" ]; then
      echo "### SETUP-REFUSED NOT TAKEN: $r holds file(s) -- a root with bytes in it is not a root a preamble refusal left behind:"
      printf '%s\n' "$files" | sed 's/^/###     /'
      return 0
    fi
    if [ -n "$sealed" ]; then
      echo "### SETUP-REFUSED NOT TAKEN: $r holds SEALED (write-stripped) director(ies) -- a provisioned store, so a run that reached the backend. Refusing exactly as before:"
      printf '%s\n' "$sealed" | sed 's/^/###     /'
      return 0
    fi
  done
  echo "### SETUP-REFUSED TAKEN: ${#PRESENT_ROOTS[@]} present root(s), no file, no sealed tree, and no evidence can ever arrive for a job that ran no arm. Handing them to $CLEANUP (which owns the containment refusals)."
  bash "$CLEANUP" "${PRESENT_ROOTS[@]}"; crc=$?
  for r in "${PRESENT_ROOTS[@]+"${PRESENT_ROOTS[@]}"}"; do
    if [ -e "$r" ]; then echo "### SETUP-REFUSED kept  $r (still on disk)"
    else echo "### SETUP-REFUSED removed $r"; removed=$((removed + 1)); fi
  done
  echo "### CLEANUP SETUP-REFUSED roots=${#PRESENT_ROOTS[@]} removed=$removed cleanup_rc=$crc"
  if [ "$crc" = 0 ] && [ "$removed" = "${#PRESENT_ROOTS[@]}" ]; then exit 0; fi
  echo "### SETUP-REFUSED INCOMPLETE -- $CLEANUP returned $crc and $removed of ${#PRESENT_ROOTS[@]} root(s) are gone. Law 9: this rc reaches Slurm."
  exit 2
}
setup_refused_check   # SETUP-REFUSED-BRANCH (MUTATION ANCHOR)

# --- CLEANUP-SEAM-2: A JOB THAT DIED MID-ARM, AFTER IT HAD ALREADY STAGED ----
# MEASURED on det161b-cleanup 6014484, the afterany owner of det161b-proof
# 6014471. 6014471 did NOT die in its preamble -- it got through the smoke, got
# through staging, ran arm W1 for 865 s and arm W2 for 86 s, and both arms died
# in `retread_scope_sdist_builds`:
#
#     retread_scope_sdist_builds: FATAL no byte-keyed bucket was symlinked --
#       the overlay would be a COLD cache, not an isolation of the build halves
#     FATAL: retread_scope_sdist_builds refused
#     ### ARM W1 WRAPPER EXIT rc=7 arm wall=865s 2026-09-07T04:41:35-04:00
#     ### ARM W2 WRAPPER EXIT rc=7 arm wall=86s 2026-09-07T04:43:01-04:00
#     ### D16 DET-1-6 PROOF DONE job_fatal=1 2026-09-07T04:43:03-04:00
#     ### DET161B_EXIT=1
#
# sacct agrees: `6014471 det161b-proof FAILED 1:0`. A wrapper that exits 7
# writes no `.rc`, no `.wall` and no `.lock.log`, so condition 1 reported all
# three MISSING and 6014484 printed `### CLEANUP REFUSED -- nothing deleted`,
# rc 2, keeping certD16-6014471, certD6A-6014471, ws.D6A-6014471 and the
# per-arm isolated cache -- and it would keep them for the rest of time,
# because the evidence it waits for can never arrive for a job that is over.
#
# THIS IS CLEANUP-SEAM-1's DEFECT ONE SEAM FURTHER ON, AND THE SEAM-1 BRANCH
# CANNOT COVER IT. `setup_refused_check` requires every present root to hold NO
# FILE AT ALL -- that is right for a job the preamble refused before arm 1, which
# left six empty directories. A job that died mid-arm has STAGED: its roots hold
# a pixi/rattler/uv store, an overlay, a workspace checkout. The emptiness proof
# is exactly what distinguishes the two cases, so widening seam 1 would delete
# the roots of jobs it was written to protect. This is a SECOND branch with a
# SECOND set of conditions, not a loosening of the first.
#
# THE THREE CONDITIONS, and each one is refusable on its own:
#
#   (i)   THE JOB'S OWN STDOUT CARRIES A JOB-FATAL ROW. Not any error, not a
#         stderr line: one of the four rows this campaign's drivers print to
#         DECLARE the job dead, each with a live producer, each derived by grep
#         and not invented here:
#           `### PREAMBLE JOB REFUSED BEFORE ARM 1. MULTIARM_JOB_FATAL=<nonzero>`
#               producer: tools/multiarm_preamble.sh, `multiarm_say "JOB REFUSED
#               BEFORE ARM 1. MULTIARM_JOB_FATAL=1"`.
#           `### ARM <label> WRAPPER EXIT rc=<nonzero>`
#               producer: the multi-arm drivers' `echo "### ARM $AL WRAPPER EXIT
#               rc=$rc arm wall=${W}s $(date -Is)"` -- det16_proof.sh,
#               det161_proof.sh, det161b_proof.sh, det162_proof.sh.
#           `### <TAG> ... PROOF DONE job_fatal=<nonzero>`
#               producer: `echo "### ${TAG} ... PROOF DONE job_fatal=$JOB_FATAL
#               $(date -Is)"` -- det1_proof.sh, det141_proof.sh, det16_proof.sh,
#               det161_proof.sh, det161b_proof.sh, det162_proof.sh.
#           `### <TAG>_EXIT=<nonzero>`
#               producer: the sbatch wrapper's `echo "### <TAG>_EXIT=$rc"` --
#               det141.sbatch, det16.sbatch, det161.sbatch, det161b.sbatch,
#               det162.sbatch, and the det1-work gate/mut sbatches.
#         ZERO IS NOT FATAL and the regex says so: `job_fatal=0`, `rc=0` and
#         `_EXIT=0` are the SUCCESS rows of the very same producers, and a
#         family that matched them would unlock the reaper on every green job
#         in the campaign.
#
#   (ii)  SLURM AGREES THE JOB DIED. `sacct -X` for RJ must be FAILED or
#         TIMEOUT. A ROW IN A LOG IS A CLAIM; THE ACCOUNTING RECORD IS THE FACT.
#         A driver that prints a fatal row and then exits 0 -- a swallowed rc,
#         which is the exact defect `driver_exit_guard.sh` exists for -- must NOT
#         unlock the reaper, because a job that reported success may have handed
#         its roots to a successor. The two halves are independent readers of the
#         same question and both have to say yes.
#         CLEANUP-SEAM-2-a (2026-09-07) WIDENS THIS LIST FROM TWO TO FOUR, and
#         the two additions are terminal failures of the same kind: a job the
#         scheduler killed. OUT_OF_MEMORY is the cgroup OOM killer -- the run is
#         over, its roots hold whatever it had staged, and no `.rc` will ever be
#         written; NODE_FAIL is the node dying under it, same story. Refusing
#         those two stranded exactly the roots this branch exists to reclaim,
#         and a lane reading the refusal would have gone looking for evidence
#         that cannot arrive.
#         WHAT IS DELIBERATELY NOT ON THE LIST, and each for its own reason:
#           CANCELLED -- an OPERATOR act, not a failure. sacct renders it
#             `CANCELLED by <uid>`, and the state is taken as the FIRST field,
#             so it is `CANCELLED` here and matches nothing. A cancel can be a
#             deliberate pause with a resubmit behind it, and reaping a paused
#             chain's roots is the one thing a cancel must not cost.
#           REQUEUED / RESIZING / SUSPENDED / PENDING / RUNNING -- not terminal
#             at all; the job may still write the evidence.
#           COMPLETED -- the case arm J2 exists for: a driver that swallowed its
#             rc and reported success may have handed its roots to a successor.
#         Arms J8 (OUT_OF_MEMORY), J9 (NODE_FAIL) and J10 (the four negatives)
#         are the readers.
#
#   (iii) NO ROOT HOLDS A SEALED (write-stripped) DIRECTORY. Same refusal, same
#         reasoning and the same depth bound as seam 1: `source_build.rs::
#         make_source_tree_read_only` strips `w` from DIRECTORIES, so a sealed
#         subtree means a rattler-build store really provisioned there, and
#         `multiarm_store_reap.sh` renames such a tree aside rather than deleting
#         it in-job. A sealed root is a reap question, not a strand question,
#         and it refuses exactly as it does today. The containment half of that
#         proof (dev:inode, not a string prefix) is NOT re-implemented here --
#         it lives in `multiarm_store_reap.sh` for the reap path and in
#         `cleanup.sh` for this one, which is why the branch DECIDES and
#         `cleanup.sh` DELETES.
#
# AND ONE MORE, WHICH SEAM 1 DID NOT NEED: NOTHING OF OURS IS STILL RUNNING ON
# THESE ROOTS. Seam 1's roots were empty, so an adopting sibling had nothing to
# lose; these roots hold staged data, which is precisely what a phase-2 or a
# sibling arm adopts. Condition 2's queue check sits BELOW this branch and would
# never run, so the branch runs it itself over every `-<jid>` token in every
# present root's basename. That is the check that would have saved
# `ws.A3B-5697522` from the A-final cert.
#
# WHOSE STDOUT, AND THE READER THAT MADE IT NECESSARY. `setup_refused_stdout`
# looked ONLY under `$D` at depth 2, which is right when the harness dir and the
# job log live together (`det141-work/logs/det141-6000903.out`). det161b splits
# them: its `D` derives to the PER-ARM root `det161b-w1`, whose only child is
# `artifacts/`, while the driver's stdout is in `det161b-work/logs/`. The
# finder returned nothing, so a branch keyed on the job's own stdout would have
# been silent on the very job it was written for. The lookup now falls back to
# the TASK ROOT at depth 3 -- bounded, HANDOFF section 2 -- and the files it
# searched are PRINTED, so a future silence is readable instead of invisible.
# MEASURED: at depth 3 the fallback finds five `*6014471*.out` files and only
# `det161b-work/logs/det161b-6014471.out` carries a row of the family; the four
# per-arm wrapper stdouts under `artifacts/` match nothing.
#
# Reader: cleanup_absent_root_guard.sh, arms J1-J5.
JOB_FATAL_RE='^### ((PREAMBLE JOB REFUSED BEFORE ARM 1\. MULTIARM_JOB_FATAL=|ARM [^ ]+ WRAPPER EXIT rc=|[A-Za-z0-9_]+_EXIT=)[0-9]*[1-9][0-9]*|.* PROOF DONE job_fatal=[0-9]*[1-9][0-9]*)( |$)'
JOB_FATAL_STATES='^(FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)$'
JOB_FATAL_SEAL_DEPTH=$SETUP_REFUSED_DEPTH

job_fatal_check () {
  local outs row st r sealed base jid jids qst removed=0 crc n
  outs=$(job_stdout_files)
  [ -n "$outs" ] || return 0
  row=$(printf '%s\n' "$outs" | while IFS= read -r f; do
          grep -m1 -E "$JOB_FATAL_RE" "$f" 2>/dev/null && break
        done)
  [ -n "$row" ] || return 0
  echo "### JOB-FATAL: the relock job $RJ declared itself dead in its own stdout:"
  echo "###   $row"
  # (ii) Slurm's own record, and it is allowed to overrule the row.
  st=$(sacct -j "$RJ" -X -n -o State 2>/dev/null | head -1 | awk '{print $1}')
  echo "### JOB-FATAL sacct state for job $RJ: '${st:-<none>}' (accepted: FAILED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL)"
  if ! printf '%s\n' "$st" | grep -qE "$JOB_FATAL_STATES"; then
    echo "### JOB-FATAL NOT TAKEN: the row is a CLAIM and sacct is the FACT -- '${st:-<none>}' is not a terminal-failure state, so a driver that printed a fatal row and still exited cleanly does not unlock the reaper. The evidence conditions decide."
    return 0
  fi
  # AND: nothing of ours still running on these roots.
  for r in "${PRESENT_ROOTS[@]+"${PRESENT_ROOTS[@]}"}"; do
    base=${r##*/}
    jids=$(printf '%s\n' "$base" | grep -oE -- '-[0-9]{6,}' | tr -d - | sort -u)
    for jid in $jids; do
      qst=$(squeue -j "$jid" -h -o '%t' 2>/dev/null | paste -sd, )
      if [ -n "$qst" ]; then
        echo "### JOB-FATAL NOT TAKEN: job $jid (named by root $r) is still in the queue: $qst. A root with staged data in it is exactly what a sibling arm adopts."
        return 0
      fi
    done
  done
  # (iii) a sealed subtree is a reap question, not a strand question.
  for r in "${PRESENT_ROOTS[@]+"${PRESENT_ROOTS[@]}"}"; do
    sealed=$(find "$r" -maxdepth "$JOB_FATAL_SEAL_DEPTH" -type d ! -writable 2>/dev/null | head -5)
    n=$(printf '%s\n' "$sealed" | grep -c .)
    echo "### JOB-FATAL root $r: sealed_dirs=$n within depth $JOB_FATAL_SEAL_DEPTH"
    if [ -n "$sealed" ]; then
      echo "### JOB-FATAL NOT TAKEN: $r holds SEALED (write-stripped) director(ies) -- a provisioned store, which multiarm_store_reap.sh renames aside rather than deleting in-job. Refusing exactly as before:"
      printf '%s\n' "$sealed" | sed 's/^/###     /'
      return 0
    fi
  done
  echo "### JOB-FATAL TAKEN: ${#PRESENT_ROOTS[@]} present root(s), job $RJ is $st, nothing of ours in the queue, no sealed tree, and no evidence can ever arrive for a job that is over. Handing them to $CLEANUP (which owns the containment refusals)."
  bash "$CLEANUP" "${PRESENT_ROOTS[@]}"; crc=$?
  for r in "${PRESENT_ROOTS[@]+"${PRESENT_ROOTS[@]}"}"; do
    if [ -e "$r" ]; then echo "### JOB-FATAL kept  $r (still on disk)"
    else echo "### JOB-FATAL removed $r"; removed=$((removed + 1)); fi
  done
  echo "### CLEANUP JOB-FATAL roots=${#PRESENT_ROOTS[@]} removed=$removed cleanup_rc=$crc fatal_row=\"$row\""
  if [ "$crc" = 0 ] && [ "$removed" = "${#PRESENT_ROOTS[@]}" ]; then exit 0; fi
  echo "### JOB-FATAL INCOMPLETE -- $CLEANUP returned $crc and $removed of ${#PRESENT_ROOTS[@]} root(s) are gone. Law 9: this rc reaches Slurm."
  exit 2
}
job_fatal_check   # JOB-FATAL-BRANCH (MUTATION ANCHOR)

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
for r in "${PRESENT_ROOTS[@]+"${PRESENT_ROOTS[@]}"}"; do
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
  echo "### CLEANUP REFUSED -- nothing deleted. Roots kept (PRESENT on disk): ${PRESENT_ROOTS[*]:-<none -- every root named was absent>}"
  [ "${#ABSENT_ROOTS[@]}" -eq 0 ] || echo "### Roots ABSENT (never existed or already reclaimed, nothing kept): ${ABSENT_ROOTS[*]}"
  exit 2
fi

echo "### GATE PASSED -- handing ${#PRESENT_ROOTS[@]} PRESENT root(s) to $CLEANUP (${#ABSENT_ROOTS[@]} absent, not passed)"
exec bash "$CLEANUP" "${PRESENT_ROOTS[@]}"
