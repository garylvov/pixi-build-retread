#!/usr/bin/env bash
# cleanup_owner_guard.sh -- the reader for the "exactly ONE cleanup owner per
# root" rule.
#
# THE DEFECT IT WOULD HAVE CAUGHT. On 2026-09-04 the job-scoped roots
# `certAFINAL2-5769426` and `ws.AFINAL2-5769426` were handed to TWO cleanup jobs
# at once: the dispatch-time gated cleanup 5770508, submitted by the launcher on
# `--dependency=afterany:<p1>:<p2>`, and the cert phase's OWN self-submitted
# cleanup 5776646. Both released on the same dependency, both started 08:39:44
# on node2343, and both walked the same two trees. Two concurrent `rm -rf` walks
# of one tree unlink entries out from under each other, so each one's rmdir of a
# parent finds children it cannot see: BOTH returned rc=1 with pages of
# "Directory not empty", 5776646 also hit `rm: fts_read failed: Stale file
# handle` (which only a second walker can produce), both logged
# `exists_after=YES`, and 590028 + 668715 entries were LEFT ON DISK after 2864 s
# and 3941 s of wall -- while both jobs still printed `CLEANUP DONE rc=0`.
#
# WHAT THIS GUARD DOES. It extracts the REAL `cleanup_owner` and
# `cleanup_submit_or_defer` out of the shipped phaseN_cert.sh -- never a
# re-implementation -- puts a counting stub named `sbatch` first on PATH, and
# drives the real decision over both cases:
#
#   A. a handoff stamp recording CLEANUP_JOB=<id>  -> ZERO sbatch calls, and the
#      printed line NAMES that id as the owner.
#   A2. no stamp record but CLEANUP_AT_DISPATCH=<id> in the environment -> same.
#   A3. the legacy CLEANUP_AT_DISPATCH=1 -> still defers, prints `unrecorded-id`.
#   B. nothing recorded -> EXACTLY ONE sbatch call, and the printed line names
#      the submitted job id as the owner, with `--dependency=` intact.
#   C. NEGATIVE CONTROL: a pre-fix copy of `cleanup_owner`, with the stamp's
#      CLEANUP_JOB dropped as a source, MUST submit under case A -- that is the
#      second owner, reproduced. Without C, A could pass on a fixture that never
#      submits anything and the guard would be measuring nothing.
#   D. the WRITER half: the real resolution block out of phaseN_relock.sh must
#      resolve an id from the environment AND from the dispatch note file, since
#      a stamp line nothing writes is the same defect from the other end.
#
# Every case also asserts the log line NAMES an owner, because a silent branch
# is what left a reader unable to tell who was cleaning what.
#
# Falsification: drop `*) id=$CLEANUP_JOB ;;` from `cleanup_owner` and A goes
# RED with a real sbatch call recorded, while C stays green.
#
# Usage: cleanup_owner_guard.sh          (self-contained, needs only $TMPDIR)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CERT=$HERE/phaseN_cert.sh
RELOCK=$HERE/phaseN_relock.sh

W=$(mktemp -d "${TMPDIR:-/tmp}/cleanup-owner-guard.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
FAIL=0
fail () { echo "GUARD FAIL: $*"; FAIL=1; }
ok   () { echo "GUARD  ok : $*"; }

extract () {  # $1=file  $2=function name -> the function's text, verbatim
  awk -v fn="$2" '$0 ~ "^"fn" \\(\\) \\{" {p=1} p {print} p && /^\}$/ {exit}' "$1"
}

[ -f "$CERT" ] || { echo "GUARD FATAL: $CERT not found"; exit 2; }
OWNER=$(extract "$CERT" cleanup_owner)
SUBMIT=$(extract "$CERT" cleanup_submit_or_defer)
[ -n "$OWNER" ]  || { echo "GUARD FATAL: could not extract cleanup_owner from $CERT"; exit 2; }
[ -n "$SUBMIT" ] || { echo "GUARD FATAL: could not extract cleanup_submit_or_defer from $CERT"; exit 2; }
ok "extracted the real cleanup_owner ($(printf '%s\n' "$OWNER" | wc -l) lines) and cleanup_submit_or_defer ($(printf '%s\n' "$SUBMIT" | wc -l) lines) from phaseN_cert.sh"

########## the counting sbatch stub -- first on PATH ##########################
# The real function calls `env -u SLURM_JOB_ID sbatch --parsable ...`, so `env`
# resolves this through PATH exactly as it would resolve the real one.
mkdir -p "$W/bin"
cat > "$W/bin/sbatch" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$SBATCH_LOG"
echo 9999001
STUB
chmod +x "$W/bin/sbatch"

# the roots, and a cleanup script that must never actually run
ROOT_A=$W/roots/certGUARD-1
ROOT_B=$W/roots/ws.GUARD-1
mkdir -p "$ROOT_A" "$ROOT_B" "$W/art"
printf '#!/bin/sh\necho "GUARD BUG: the cleanup script was EXECUTED"\n' > "$W/cleanup.sh"
chmod +x "$W/cleanup.sh"

# $1=label  $2=the cleanup_owner text to use  $3..=env assignments
run_case () {
  local label=$1 owner_txt=$2; shift 2
  local drv=$W/$label.sh
  {
    echo 'set -u'
    echo "TAG=GUARD; J=777777; A=$W/art; CLEANUP=$W/cleanup.sh"
    echo 'CLEANUP_SBATCH_ARGS="--partition=batch --qos=normal --cpus-per-task=1 --mem=4G --time=06:00:00"'
    printf '%s\n' "$owner_txt"
    printf '%s\n' "$SUBMIT"
    echo 'cleanup_submit_or_defer "afterany:5769426:777777" "$@"'
  } > "$drv"
  : > "$W/$label.sbatch"
  SBATCH_LOG=$W/$label.sbatch PATH=$W/bin:$PATH \
    env "$@" bash "$drv" "$ROOT_A" "$ROOT_B" > "$W/$label.out" 2>&1
  echo $?
}

subs () { wc -l < "$W/$1.sbatch" | tr -d ' '; }   # how many sbatch calls
outp () { cat "$W/$1.out"; }

########## A. the stamp records a dispatch cleanup -> submit NOTHING ##########
RC=$(run_case A "$OWNER" CLEANUP_JOB=5770508)
N=$(subs A)
if [ "$N" = 0 ] && outp A | grep -q '^### CLEANUP OWNER: job 5770508 (submitted at dispatch)' \
   && outp A | grep -q 'defers to job 5770508'; then
  ok "A. stamp CLEANUP_JOB=5770508 -> 0 sbatch calls, log names job 5770508 as owner (rc=$RC)"
else
  fail "A. a stamped dispatch cleanup did NOT stop the self-submit: sbatch calls=$N rc=$RC"
  outp A | sed 's/^/GUARD:   /'
  [ "$N" != 0 ] && sed 's/^/GUARD:   sbatch /' "$W/A.sbatch"
fi

########## A2/A3. the environment channel #####################################
RC=$(run_case A2 "$OWNER" CLEANUP_AT_DISPATCH=5770508)
if [ "$(subs A2)" = 0 ] && outp A2 | grep -q '^### CLEANUP OWNER: job 5770508 (submitted at dispatch)'; then
  ok "A2. CLEANUP_AT_DISPATCH=5770508 -> 0 sbatch calls, log names job 5770508 (rc=$RC)"
else
  fail "A2. CLEANUP_AT_DISPATCH=<id> did not defer: sbatch calls=$(subs A2) rc=$RC"
  outp A2 | sed 's/^/GUARD:   /'
fi

RC=$(run_case A3 "$OWNER" CLEANUP_AT_DISPATCH=1)
if [ "$(subs A3)" = 0 ] && outp A3 | grep -q '^### CLEANUP OWNER: job unrecorded-id (submitted at dispatch)'; then
  ok "A3. legacy CLEANUP_AT_DISPATCH=1 -> 0 sbatch calls, log says the id was not recorded (rc=$RC)"
else
  fail "A3. legacy CLEANUP_AT_DISPATCH=1 did not defer, or did not say the id is unrecorded: sbatch calls=$(subs A3) rc=$RC"
  outp A3 | sed 's/^/GUARD:   /'
fi

########## B. nothing recorded -> EXACTLY ONE cleanup, and it says so #########
RC=$(run_case B "$OWNER")
N=$(subs B)
if [ "$N" = 1 ] && outp B | grep -q '^### CLEANUP OWNER: job 9999001 (submitted by this cert job 777777' \
   && grep -q -- '--dependency=afterany:5769426:777777' "$W/B.sbatch" \
   && grep -q -- "$ROOT_A" "$W/B.sbatch" && grep -q -- "$ROOT_B" "$W/B.sbatch"; then
  ok "B. nothing recorded -> exactly 1 sbatch call over both roots on --dependency=afterany:5769426:777777, log names job 9999001 as owner (rc=$RC)"
else
  fail "B. the no-owner case did not submit exactly one cleanup: sbatch calls=$N rc=$RC"
  outp B | sed 's/^/GUARD:   /'
  sed 's/^/GUARD:   sbatch /' "$W/B.sbatch"
fi

########## C. NEGATIVE CONTROL: the pre-fix owner MUST become a second owner ###
# The pre-fix shape read only the environment; a stamp that named 5770508 meant
# nothing to it. Feed it case A and it submits -- that is job 5776646.
PREFIX_OWNER=$(printf '%s\n' "$OWNER" | sed 's/^\( *\)\*) id=\$CLEANUP_JOB ;;$/\1*) ;;/')
if [ "$PREFIX_OWNER" = "$OWNER" ]; then
  fail "C. could not build the pre-fix cleanup_owner (the CLEANUP_JOB arm did not match) -- the control is vacuous"
else
  RC=$(run_case C "$PREFIX_OWNER" CLEANUP_JOB=5770508)
  N=$(subs C)
  if [ "$N" = 1 ]; then
    ok "C. the pre-fix owner ignores the stamp and submits a SECOND cleanup (1 sbatch call) -- fixture discriminates, A is not vacuous"
  else
    fail "C. the pre-fix owner ALSO submitted nothing ($N sbatch calls) -- case A proves nothing"
    outp C | sed 's/^/GUARD:   /'
  fi
fi

########## D. the WRITER half in phaseN_relock.sh #############################
if [ ! -f "$RELOCK" ]; then
  fail "D. $RELOCK not found -- the stamp writer is unchecked"
else
  RES=$(awk '/^CLEANUP_JOB_RECORD=""$/{p=1} p{print} p&&/^fi$/{exit}' "$RELOCK")
  if [ -z "$RES" ]; then
    fail "D. could not extract the CLEANUP_JOB_RECORD resolution block from phaseN_relock.sh"
  elif ! grep -q 'echo "CLEANUP_JOB=\$CLEANUP_JOB_RECORD"' "$RELOCK"; then
    fail "D. phaseN_relock.sh resolves an owner but never writes CLEANUP_JOB= into the handoff stamp -- the cert phase would never see it"
  else
    DRV=$W/relock.sh
    { echo 'set -u'; echo "A=\$1"; printf '%s\n' "$RES"; echo 'echo "RECORD=$CLEANUP_JOB_RECORD"'; } > "$DRV"
    mkdir -p "$W/d1" "$W/d2" "$W/d3"
    R1=$(CLEANUP_AT_DISPATCH=5770508 bash "$DRV" "$W/d1" | sed -n 's/^RECORD=//p')
    printf '5770508\n' > "$W/d2/cleanup_at_dispatch.jobid"
    R2=$(bash "$DRV" "$W/d2" | sed -n 's/^RECORD=//p')
    R3=$(bash "$DRV" "$W/d3" | sed -n 's/^RECORD=//p')
    if [ "$R1" = 5770508 ] && [ "$R2" = 5770508 ] && [ -z "$R3" ]; then
      ok "D. phaseN_relock.sh records the owner from the environment AND from the dispatch note, and records NOTHING when there is nothing (env=$R1 note=$R2 neither='$R3')"
    else
      fail "D. the stamp writer resolved wrong: env='$R1' note='$R2' neither='$R3' (want 5770508 / 5770508 / empty)"
    fi
  fi
fi

[ "$FAIL" = 0 ] && { echo "cleanup-owner guard: ALL PASS"; exit 0; }
echo "cleanup-owner guard: FAILED"; exit 1
