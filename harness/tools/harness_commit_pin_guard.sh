#!/usr/bin/env bash
# GUARD for MERGE-N-2, 2026-09-06, HARNESS-FIX-2.
#
# THE DEFECT.  The harness pin reached a job ONLY through
# `sbatch --export=ALL,HARNESS_COMMIT=<sha>`, and in TWO CONSECUTIVE merge lanes
# that clause left the job `PENDING Reason=launch_failed_requeued_held` with
# "user env retrieval failed": Slurm re-runs the submitting user's login
# environment to build `ALL`, and when that retrieval times out the job is
# requeued AND HELD.  The job never starts, so nothing in its own log can say
# why, and the operator sees a merge lane that simply did not run.
#
# THE FIX.  `tools/harness_commit_resolve.sh` reads the pin from a file the job
# OWNS -- `<job root>/HARNESS_COMMIT`, written by the submitter into the harness
# directory whose `artifacts/` the run already writes to -- and falls back to the
# exported variable.  Both HARNESS-DRIFT blocks call it.  A file and an export
# that DISAGREE refuse and name both.
#
# ARMS -- the resolver alone, then the BLOCK as it sits in each template.
#   A   file only                     -> value, source=file
#   B   export only (the fallback)    -> value, source=export
#   C   file and export AGREEING      -> value, source=file
#   D   file and export DISAGREEING   -> rc 2, both values named
#   E   neither                       -> rc 0, empty stdout, announced unset
#   F   HARNESS_COMMIT_FILE overrides <job root>/HARNESS_COMMIT
#   G   a file with a comment and trailing whitespace yields the bare sha
#   H   THE ACCEPTANCE ARM: the HARNESS-DRIFT block of each template, run with
#       NO exported HARNESS_COMMIT at all and the file present, reaches the
#       drift check and passes it the sha from the file.  This is the fixture
#       submit MERGE-N-2 asks for.
#   I   MUTATION, PINNED TO A COMMIT CONSTANT ($N2_OLD): the SAME block from the
#       commit that carried the defect, same fixture -- it cannot see the file
#       and announces the drift check OFF.  Without this arm H proves nothing.
#   J   the block carries the resolver's refusal to the template's own exit code.
#
# NOTHING IS RUN THAT COULD LOCK.  Each arm extracts ONLY the region between
# `### HARNESS-DRIFT BEGIN` and `### HARNESS-DRIFT END`, prepends the two
# variables it reads ($FAST_ENV, $D) and appends `exit 0`.  `harness_drift_check.sh`
# is a STUB beside the stub $FAST_ENV, so the arms assert on what the block
# HANDED it, never on a real drift scan.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=${HARNESS_REPO:-}
[ -n "$REPO" ] || REPO=$(cd -- "$HERE/../.." && pwd)
if [ ! -d "$REPO/harness/tools" ] || ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  REPO=/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools
fi
[ -d "$REPO/harness/tools" ] || { echo "FATAL: no harness repo at $REPO"; exit 3; }
RESOLVE=$REPO/harness/tools/harness_commit_resolve.sh
[ -f "$RESOLVE" ] || { echo "FATAL: no resolver at $RESOLVE"; exit 3; }
N2_OLD=${N2_OLD:-6279978}   # the HARNESS_COMMIT that carried the defect
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
pass=0; fail=0
ok()  { echo "PASS  $*"; pass=$((pass+1)); }
bad() { echo "FAIL  $*"; fail=$((fail+1)); }
echo "### guard for $RESOLVE and the HARNESS-DRIFT block, mutation pinned to $N2_OLD"

SHA_FILE=aaaaaaa
SHA_ENV=bbbbbbb
JR=$W/jobroot; mkdir -p "$JR"

# ---------- the resolver on its own ------------------------------------------
run_resolve() { # run_resolve <logfile> [VAR=VAL ...] -- <job root>
  local log=$1; shift
  local -a envs=()
  while [ "${1:-}" != "--" ]; do envs+=("$1"); shift; done
  shift
  ( env -u HARNESS_COMMIT -u HARNESS_COMMIT_FILE "${envs[@]}" bash "$RESOLVE" "$@" ) \
    > "$log.out" 2> "$log.err"
  echo $?
}

printf '%s\n' "$SHA_FILE" > "$JR/HARNESS_COMMIT"
rcA=$(run_resolve "$W/A" -- "$JR")
[ "$rcA" = 0 ] && [ "$(cat "$W/A.out")" = "$SHA_FILE" ] && grep -q 'source=file:' "$W/A.err" \
  && ok "A: the file alone resolves the pin (rc=0, source=file)" \
  || bad "A: rc=$rcA out='$(cat "$W/A.out")' err='$(cat "$W/A.err")'"

rm -f "$JR/HARNESS_COMMIT"
rcB=$(run_resolve "$W/B" "HARNESS_COMMIT=$SHA_ENV" -- "$JR")
[ "$rcB" = 0 ] && [ "$(cat "$W/B.out")" = "$SHA_ENV" ] && grep -q 'source=export' "$W/B.err" \
  && ok "B: the export alone still works -- the fallback is intact" \
  || bad "B: rc=$rcB out='$(cat "$W/B.out")' err='$(cat "$W/B.err")'"

printf '%s\n' "$SHA_FILE" > "$JR/HARNESS_COMMIT"
rcC=$(run_resolve "$W/C" "HARNESS_COMMIT=$SHA_FILE" -- "$JR")
[ "$rcC" = 0 ] && [ "$(cat "$W/C.out")" = "$SHA_FILE" ] && grep -q 'source=file:' "$W/C.err" \
  && ok "C: file and export AGREEING resolve to the file" \
  || bad "C: rc=$rcC out='$(cat "$W/C.out")'"

rcD=$(run_resolve "$W/D" "HARNESS_COMMIT=$SHA_ENV" -- "$JR")
[ "$rcD" = 2 ] && ok "D: file and export DISAGREEING refuses with rc=2" || bad "D: rc=$rcD, want 2"
grep -q "$SHA_FILE" "$W/D.err" && grep -q "$SHA_ENV" "$W/D.err" \
  && ok "D: the refusal names BOTH values" || bad "D: the refusal does not name both: $(cat "$W/D.err")"

rm -f "$JR/HARNESS_COMMIT"
rcE=$(run_resolve "$W/E" -- "$JR")
[ "$rcE" = 0 ] && [ ! -s "$W/E.out" ] && grep -q 'HARNESS_COMMIT unset' "$W/E.err" \
  && ok "E: neither source is announced unset, never silently guessed" \
  || bad "E: rc=$rcE out='$(cat "$W/E.out")' err='$(cat "$W/E.err")'"

printf '%s\n' zzzzzzz > "$JR/HARNESS_COMMIT"
printf '%s\n' "$SHA_FILE" > "$W/elsewhere"
rcF=$(run_resolve "$W/F" "HARNESS_COMMIT_FILE=$W/elsewhere" -- "$JR")
[ "$rcF" = 0 ] && [ "$(cat "$W/F.out")" = "$SHA_FILE" ] \
  && ok "F: HARNESS_COMMIT_FILE overrides <job root>/HARNESS_COMMIT" \
  || bad "F: rc=$rcF out='$(cat "$W/F.out")'"

printf '# written by the submitter\n   %s  \n' "$SHA_FILE" > "$JR/HARNESS_COMMIT"
rcG=$(run_resolve "$W/G" -- "$JR")
[ "$rcG" = 0 ] && [ "$(cat "$W/G.out")" = "$SHA_FILE" ] \
  && ok "G: a comment and surrounding whitespace never reach the commit-ish" \
  || bad "G: rc=$rcG out='$(cat "$W/G.out")'"

# ---------- the BLOCK, as it sits in each template ----------------------------
# The stub environment: $FAST_ENV's DIRECTORY is where the block looks for both
# helpers, so the resolver under test is copied in beside a STUB drift check.
STUB=$W/stubtools; mkdir -p "$STUB"
: > "$STUB/retread_fast_env.sh"
cp "$RESOLVE" "$STUB/harness_commit_resolve.sh"
cat > "$STUB/harness_drift_check.sh" <<'STUBEOF'
echo "### [guard stub] drift check called with commit=$1"
exit 0
STUBEOF

# extract_block <source file> <dst> ; the region and nothing else.
extract_block() {
  { printf 'set -uo pipefail\nFAST_ENV=%s/retread_fast_env.sh\nD=%s\n' "$STUB" "$JR"
    sed -n '/^### HARNESS-DRIFT BEGIN/,/^### HARNESS-DRIFT END/p' "$1"
    printf 'exit 0\n'
  } > "$2"
}
run_block() { # run_block <script> <log> [VAR=VAL ...]
  local s=$1 log=$2; shift 2
  ( env -u HARNESS_COMMIT -u HARNESS_COMMIT_FILE "$@" bash "$s" ) > "$log" 2>&1
  echo $?
}

printf '%s\n' "$SHA_FILE" > "$JR/HARNESS_COMMIT"
for tpl in phaseN_relock.sh phaseN_cert.sh; do
  SRC=$REPO/harness/phase_template/$tpl
  extract_block "$SRC" "$W/blk-$tpl"
  rcH=$(run_block "$W/blk-$tpl" "$W/H-$tpl")
  if [ "$rcH" = 0 ] && grep -q "drift check called with commit=$SHA_FILE" "$W/H-$tpl"; then
    ok "H $tpl: NO --export at all, the file present -> the drift check runs on $SHA_FILE"
  else
    bad "H $tpl: rc=$rcH -- the block did not reach the drift check from the file"; sed -n '1,8p' "$W/H-$tpl"
  fi
  if grep -q "HARNESS_COMMIT unset" "$W/H-$tpl"; then
    bad "H $tpl: the block still announced the check OFF with the file sitting there"
  else
    ok "H $tpl: the check is NOT announced OFF -- the defect is gone"
  fi
done

# ---------- I: THE MUTATION, pinned to a commit constant ----------------------
for tpl in phaseN_relock.sh phaseN_cert.sh; do
  OLD=$W/OLD-$tpl
  if git -C "$REPO" show "$N2_OLD:harness/phase_template/$tpl" > "$OLD" 2>/dev/null && [ -s "$OLD" ]; then
    extract_block "$OLD" "$W/oblk-$tpl"
    rcI=$(run_block "$W/oblk-$tpl" "$W/I-$tpl")
    if grep -q "HARNESS_COMMIT unset -- harness drift check OFF" "$W/I-$tpl"; then
      ok "I $tpl: the pinned $N2_OLD block cannot see the file and runs UNCHECKED -- the defect, reproduced"
    else
      bad "I $tpl: $N2_OLD already read the file (rc=$rcI) -- WRONG PIN, arm H proves nothing"
    fi
  else
    bad "I $tpl: could not extract $N2_OLD:harness/phase_template/$tpl -- THE MUTATION ARM DID NOT RUN"
  fi
done

# ---------- J: the refusal reaches the template's own exit code ---------------
for tpl in phaseN_relock.sh phaseN_cert.sh; do
  want=6; [ "$tpl" = phaseN_cert.sh ] && want=2
  rcJ=$(run_block "$W/blk-$tpl" "$W/J-$tpl" "HARNESS_COMMIT=$SHA_ENV")
  if [ "$rcJ" = "$want" ] && grep -qi 'pin REFUSED' "$W/J-$tpl"; then
    ok "J $tpl: a disagreeing file+export refuses with the template's own rc=$want"
  else
    bad "J $tpl: rc=$rcJ, want $want; log: $(head -3 "$W/J-$tpl" | tr '\n' ' ')"
  fi
done

echo "### MERGE-N-2 pin guard: pass=$pass fail=$fail"
[ "$fail" = 0 ] || exit 1
exit 0
