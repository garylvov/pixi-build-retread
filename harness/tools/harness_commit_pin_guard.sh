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
#   K   THE WRITER HALF: `--write <job root> <sha>` creates the file the reader
#       reads, the two round-trip, and a sha that is not a commit in the harness
#       repo is refused at SUBMIT time -- when it is cheap -- writing nothing.
#   L   HARNESS-SYNC-2-1, THE READER HALF: with a sync record present, a pin
#       that is NOT the recorded commit is refused rc 4 with ONE `### PIN STALE`
#       line naming both shas and the `--write` command, BEFORE the drift check;
#       the recorded pin passes straight through to it; an `--allow-older` pin
#       still passes, because the writer leaves a marker sidecar the reader
#       honours -- and a marker naming a DIFFERENT sha does not authorise, nor
#       does one an ordinary write has since cancelled.  Its MUTATION is pinned
#       to $HS21_OLD: that reader accepts the stale pin in silence and the block
#       goes on to call the drift check on it, which is the eight-refusal shape
#       this arm exists to kill.
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
HS21_OLD=${HS21_OLD:-63a8521}  # HARNESS-SYNC-2-1: the reader BEFORE this fix
HS21_REC=${HS21_REC:-63a8521}  # and the commit arm L's fixture record names
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
# HARNESS-SYNC-2-1: the reader now compares the pin against the sync record of
# the task dir it is told about, and the DEFAULT is the live task dir -- which
# HAS a record.  Every arm below that is not about the record therefore runs
# against a task dir that has NONE, where that check announces itself OFF; the
# record's own arms (L) point HARNESS_TASK_DIR at their own fixture.
export HARNESS_TASK_DIR=$W/norecord-default; mkdir -p "$HARNESS_TASK_DIR/tools"
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

# ---- K: THE WRITER HALF -- a file nobody writes is a fallback that never fires
# DET-1-1: `--write` now reads `<task dir>/tools/.harness_synced_commit`. These
# arms are about the writer/reader round trip, NOT about the record, so they run
# against a task dir that HAS no record -- where the resolve-at-submit check
# announces itself OFF rather than guessing. The record's own arms live in
# harness_sync_guard.sh (arm G), beside the writer of the record.
KJR=$W/writejr
KTASK=$W/norecord; mkdir -p "$KTASK/tools"
kwrite () { HARNESS_TASK_DIR="$KTASK" bash "$RESOLVE" --write "$@"; }
N2_FULL=$(git -C "${HARNESS_REPO:-/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools}" \
            rev-parse --verify "${N2_OLD}^{commit}" 2>/dev/null || echo "$N2_OLD")
wrc=$(kwrite "$KJR" "$N2_OLD" > "$W/K1.log" 2>&1; echo $?)
if [ "$wrc" = 0 ] && [ "$(cat "$KJR/HARNESS_COMMIT" 2>/dev/null)" = "$N2_FULL" ]; then
  ok "K1: --write creates <job root>/HARNESS_COMMIT, as the FULL sha"
else
  bad "K1: --write rc=$wrc, file='$(cat "$KJR/HARNESS_COMMIT" 2>/dev/null)' want=$N2_FULL"
fi
grep -q 'no sync record' "$W/K1.log" \
  && ok "K1: with no record the resolve-at-submit check announces itself OFF (DET-1-1)" \
  || bad "K1: no record and no announcement -- a silent skip"
rrc=$(run_resolve "$W/K1r" -- "$KJR")
if [ "$rrc" = 0 ] && [ "$(cat "$W/K1r.out")" = "$N2_FULL" ]; then
  ok "K1: and the reader round-trips it -- writer and reader are the same file"
else
  bad "K1: round trip rc=$rrc out='$(cat "$W/K1r.out")'"
fi
wrc=$(kwrite "$W/nope" deadbee0 > "$W/K2.log" 2>&1; echo $?)
if [ "$wrc" = 2 ] && grep -q 'WRITE REFUSED' "$W/K2.log" && [ ! -f "$W/nope/HARNESS_COMMIT" ]; then
  ok "K2: --write refuses a sha that is not a commit, at SUBMIT time, and writes nothing"
else
  bad "K2: rc=$wrc, log: $(cat "$W/K2.log")"
fi


# ---- L: HARNESS-SYNC-2-1 -- THE READER REFUSES A STALE PIN TOO --------------
# The writer's rc-3 refusal only fires when a lane PASSES a sha.  A lane that
# copies the pin FILE bypasses it, and before this arm the job then died at its
# drift gate 3-6 s later with a table of files.  These arms run against a task
# dir with a REAL sync record, so the reader has something to compare against.
LTASK=$W/rectask; mkdir -p "$LTASK/tools"
LREC_FULL=$(git -C "$REPO" rev-parse --verify "${HS21_REC}^{commit}" 2>/dev/null || echo "")
if [ -z "$LREC_FULL" ] || [ -z "$N2_FULL" ] || [ "$LREC_FULL" = "$N2_FULL" ]; then
  bad "L: could not resolve the two commit constants (rec=$HS21_REC old=$N2_OLD) -- THE ARM DID NOT RUN"
else
printf '%s\n' "$LREC_FULL" > "$LTASK/tools/.harness_synced_commit"
LJR=$W/Ljobroot; mkdir -p "$LJR"
lread() { # lread <logfile> [VAR=VAL ...] -- <job root>
  local log=$1; shift
  local -a envs=()
  while [ "${1:-}" != "--" ]; do envs+=("$1"); shift; done
  shift
  ( env -u HARNESS_COMMIT -u HARNESS_COMMIT_FILE HARNESS_TASK_DIR="$LTASK" "${envs[@]}" \
      bash "$RESOLVE" "$@" ) > "$log.out" 2> "$log.err"
  echo $?
}

# L1  the recorded pin passes, and says so
printf '%s\n' "$LREC_FULL" > "$LJR/HARNESS_COMMIT"; rm -f "$LJR/HARNESS_COMMIT.allow-older"
rcL1=$(lread "$W/L1" -- "$LJR")
[ "$rcL1" = 0 ] && [ "$(cat "$W/L1.out")" = "$LREC_FULL" ] \
  && grep -q 'pin matches the sync record' "$W/L1.err" \
  && ok "L1: a pin that IS the recorded commit passes through to the drift check" \
  || bad "L1: rc=$rcL1 out='$(cat "$W/L1.out")' err='$(cat "$W/L1.err")'"

# L2  a stale pin: ONE line, both shas, the command, and rc 4
printf '%s\n' "$N2_FULL" > "$LJR/HARNESS_COMMIT"
rcL2=$(lread "$W/L2" -- "$LJR")
[ "$rcL2" = 4 ] && grep -q "^### PIN STALE pin=$N2_FULL synced=$LREC_FULL " "$W/L2.err" \
  && ok "L2: a STALE pin is refused rc 4 with the PIN STALE line naming both shas" \
  || bad "L2: rc=$rcL2 (want 4) err='$(head -2 "$W/L2.err")'"
grep -q -- "-- run: bash tools/harness_commit_resolve.sh --write $LJR" "$W/L2.err" \
  && ok "L2: and the refusal names the exact --write command, with THIS job root" \
  || bad "L2: the PIN STALE line does not name the --write command for $LJR"

# L3  the same refusal on the EXPORT fallback path, where there can be no marker
rm -f "$LJR/HARNESS_COMMIT"
rcL3=$(lread "$W/L3" "HARNESS_COMMIT=$N2_FULL" -- "$LJR")
[ "$rcL3" = 4 ] && grep -q '^### PIN STALE ' "$W/L3.err" \
  && ok "L3: a stale pin arriving by --export is refused the same way" \
  || bad "L3: rc=$rcL3 (want 4) err='$(head -2 "$W/L3.err")'"

# L4  --allow-older --reason writes the marker, and the reader HONOURS it
lwrite() { HARNESS_TASK_DIR="$LTASK" bash "$RESOLVE" --write "$@"; }
rm -f "$LJR/HARNESS_COMMIT" "$LJR/HARNESS_COMMIT.allow-older"
wL4=$(lwrite "$LJR" "$N2_OLD" --allow-older --reason "guard fixture: rerunning an old job shape" > "$W/L4.log" 2>&1; echo $?)
if [ "$wL4" = 0 ] && grep -q "^allow-older pin=$N2_FULL synced=$LREC_FULL .*reason=guard fixture" "$LJR/HARNESS_COMMIT.allow-older" 2>/dev/null; then
  ok "L4: --allow-older --reason leaves a marker sidecar naming the sha it authorises"
else
  bad "L4: write rc=$wL4 marker='$(cat "$LJR/HARNESS_COMMIT.allow-older" 2>/dev/null)'"
fi
rcL4=$(lread "$W/L4r" -- "$LJR")
[ "$rcL4" = 0 ] && [ "$(cat "$W/L4r.out")" = "$N2_FULL" ] \
  && grep -q 'ALLOW-OLDER honoured .*reason=guard fixture' "$W/L4r.err" \
  && ok "L4: and the reader HONOURS it -- a deliberate older pin still runs, reason in hand" \
  || bad "L4: rc=$rcL4 out='$(cat "$W/L4r.out")' err='$(cat "$W/L4r.err")'"

# L5  a marker naming a DIFFERENT sha authorises nothing
printf 'allow-older pin=%s synced=%s at=x reason=stale marker\n' "0000000000000000000000000000000000000000" "$LREC_FULL" \
  > "$LJR/HARNESS_COMMIT.allow-older"
rcL5=$(lread "$W/L5" -- "$LJR")
[ "$rcL5" = 4 ] && ok "L5: a marker for a DIFFERENT sha does not authorise this pin" \
  || bad "L5: rc=$rcL5 (want 4) err='$(head -2 "$W/L5.err")'"


# L5b  and the marker's pin= is read as a FIELD: a reason string that merely
# CONTAINS `pin=<this sha>` authorises nothing.
printf 'allow-older pin=%s synced=%s at=x reason=looks like pin=%s to a whole-line match\n' \
  "0000000000000000000000000000000000000000" "$LREC_FULL" "$N2_FULL" \
  > "$LJR/HARNESS_COMMIT.allow-older"
rcL5b=$(lread "$W/L5b" -- "$LJR")
[ "$rcL5b" = 4 ] && ok "L5b: a reason string containing pin=<sha> does not authorise itself" \
  || bad "L5b: rc=$rcL5b (want 4) err='$(head -2 "$W/L5b.err")'"

# L6  an ordinary write cancels the marker an earlier submission left
wL6=$(lwrite "$LJR" > "$W/L6.log" 2>&1; echo $?)
if [ "$wL6" = 0 ] && [ ! -f "$LJR/HARNESS_COMMIT.allow-older" ] && [ "$(cat "$LJR/HARNESS_COMMIT")" = "$LREC_FULL" ]; then
  ok "L6: an ordinary --write cancels the marker and resolves the pin from the record"
else
  bad "L6: rc=$wL6 marker_present=$([ -f "$LJR/HARNESS_COMMIT.allow-older" ] && echo yes || echo no) pin='$(cat "$LJR/HARNESS_COMMIT")'"
fi

# L7/L8  THE BLOCK, as each template carries it: the recorded pin reaches the
# drift check; the stale pin never does, and the refusal carries the template's
# own exit code.
for tpl in phaseN_relock.sh phaseN_cert.sh; do
  want=6; [ "$tpl" = phaseN_cert.sh ] && want=2
  printf '%s\n' "$LREC_FULL" > "$LJR/HARNESS_COMMIT"; rm -f "$LJR/HARNESS_COMMIT.allow-older"
  extract_block "$REPO/harness/phase_template/$tpl" "$W/Lblk-$tpl"
  sed -i "s#^D=.*#D=$LJR#" "$W/Lblk-$tpl"
  rcL7=$(run_block "$W/Lblk-$tpl" "$W/L7-$tpl" "HARNESS_TASK_DIR=$LTASK")
  [ "$rcL7" = 0 ] && grep -q "drift check called with commit=$LREC_FULL" "$W/L7-$tpl" \
    && ok "L7 $tpl: the recorded pin passes the reader and reaches the drift check" \
    || { bad "L7 $tpl: rc=$rcL7"; sed -n '1,6p' "$W/L7-$tpl"; }
  printf '%s\n' "$N2_FULL" > "$LJR/HARNESS_COMMIT"
  rcL8=$(run_block "$W/Lblk-$tpl" "$W/L8-$tpl" "HARNESS_TASK_DIR=$LTASK")
  if [ "$rcL8" = "$want" ] && grep -q '^### PIN STALE ' "$W/L8-$tpl" \
     && ! grep -q 'drift check called' "$W/L8-$tpl"; then
    ok "L8 $tpl: a stale pin refuses with the template's own rc=$want and NEVER reaches the drift check"
  else
    bad "L8 $tpl: rc=$rcL8 (want $want); log: $(head -3 "$W/L8-$tpl" | tr '\n' ' ')"
  fi
done

# L9  THE MUTATION, PINNED TO A COMMIT CONSTANT ($HS21_OLD): the reader as it
# stood before this fix, on the SAME stale fixture -- it accepts the pin and the
# block goes on to run the drift check on it, which is the drift table this arm
# exists to replace with one line.
OLDR=$W/oldresolve.sh
if git -C "$REPO" show "$HS21_OLD:harness/tools/harness_commit_resolve.sh" > "$OLDR" 2>/dev/null && [ -s "$OLDR" ]; then
  printf '%s\n' "$N2_FULL" > "$LJR/HARNESS_COMMIT"
  rcL9=$( ( env -u HARNESS_COMMIT -u HARNESS_COMMIT_FILE HARNESS_TASK_DIR="$LTASK" \
            bash "$OLDR" "$LJR" ) > "$W/L9.out" 2> "$W/L9.err"; echo $? )
  if [ "$rcL9" = 0 ] && [ "$(cat "$W/L9.out")" = "$N2_FULL" ] && ! grep -q 'PIN STALE' "$W/L9.err"; then
    ok "L9: the pinned $HS21_OLD reader ACCEPTS the stale pin in silence -- the defect, reproduced"
  else
    bad "L9: $HS21_OLD already refused (rc=$rcL9) -- WRONG PIN, arms L2/L8 prove nothing"
  fi
  OSTUB=$W/oldstub; mkdir -p "$OSTUB"; : > "$OSTUB/retread_fast_env.sh"
  cp "$OLDR" "$OSTUB/harness_commit_resolve.sh"; cp "$STUB/harness_drift_check.sh" "$OSTUB/"
  { printf 'set -uo pipefail\nFAST_ENV=%s/retread_fast_env.sh\nD=%s\n' "$OSTUB" "$LJR"
    sed -n '/^### HARNESS-DRIFT BEGIN/,/^### HARNESS-DRIFT END/p' "$REPO/harness/phase_template/phaseN_relock.sh"
    printf 'exit 0\n'; } > "$W/L9blk"
  rcL9b=$(run_block "$W/L9blk" "$W/L9b.log" "HARNESS_TASK_DIR=$LTASK")
  if grep -q "drift check called with commit=$N2_FULL" "$W/L9b.log"; then
    ok "L9: and with that reader the block runs the DRIFT CHECK on the stale sha -- the table, not the line"
  else
    bad "L9: the pre-fix block did not reach the drift check (rc=$rcL9b): $(head -3 "$W/L9b.log" | tr '\n' ' ')"
  fi
else
  bad "L9: could not extract $HS21_OLD:harness/tools/harness_commit_resolve.sh -- THE MUTATION ARM DID NOT RUN"
  bad "L9: (and its block half did not run either)"
fi
fi
echo "### MERGE-N-2 pin guard: pass=$pass fail=$fail"
[ "$fail" = 0 ] || exit 1
exit 0
