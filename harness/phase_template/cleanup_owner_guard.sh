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
#   E1/E2/E2b/E3 (CLEANUP-WALL-3, 2026-09-07). THE SECOND OWNER THAT THE FIRST
#      OWNER SUBMITS. A cleanup owner may continue itself past its wall, and on
#      2026-09-07 it did so 24 times in 18 minutes with nothing to continue. The
#      arms drive the REAL owner_snapshot.sh and the REAL cleanup_gated.sh and
#      assert an ABSENCE -- an empty sbatch calls file -- on a no-op pass and on
#      a refused one, with the refusal's rc 2 preserved, plus the mutation that
#      makes those absences falsifiable. The block sits at the bottom of this
#      file with its own long note.
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

################################################################################
# CLEANUP-WALL-3 (2026-09-07). THE SECOND OWNER, ARRIVING BY A DIFFERENT DOOR.
#
# Arms A-D above catch a second owner submitted by the DISPATCHER. This block
# catches a second owner submitted by the FIRST OWNER, which is the same defect
# and cost more: on 2026-09-07 08:07-08:25 thirty det163-cleanup jobs ran and
# TWENTY-FOUR were self-submitted continuations, none of which had anything to
# continue. `owner_wall_check` ran at job START, on the line above
# `exec bash <gate>`, so its verdict could not depend on anything the pass did;
# its whole test was `census > OWNER_WALL_COVERS`, and covers is 0 for every
# owner whose submitter passed no `--roots`, so every census was "short".
#   6020506 -> continuation 6020524, and only THEN `### CLEANUP SETUP-REFUSED`.
#   6020524 -> `entries=14400000 present=0 absent=4` (14.4 M phantom entries for
#              four roots that did not exist), continuation 6020542, then
#              `### NOTHING TO DO -- every root named is ABSENT`.
#   6020631/6020952/6021096 -> a continuation each, then `### JOB-FATAL NOT
#              TAKEN` and `### CLEANUP REFUSED` on all three: a tree the gate
#              refuses on purpose, which no number of passes will change.
#
# THE ARMS ASSERT AN ABSENCE, WHICH IS THE HALF cleanup_wall_guard NEVER HAD.
# "No continuation" is a BYTE FACT here -- a stub `sbatch` first on PATH appends
# its whole argv to a calls file, so the assertion is that the file is empty and
# never that a log line is missing. The REAL owner_snapshot.sh generates the
# owner and the REAL cleanup_gated.sh is the gate, so E1 and E2 measure the
# production verdicts and not a re-implementation of them.
#
#   E1  A NO-OP PASS DOES NOT CONTINUE. Every root ABSENT -> the gate's
#       `### NOTHING TO DO`, and the run-time census counts an absent root as
#       ZERO (entries=0, not the submit-time estimate -- 6020524's 14,400,000).
#   E2  A REFUSED PASS DOES NOT CONTINUE, AND STILL EXITS 2. A present root with
#       missing evidence -> `### CLEANUP REFUSED`, rc 2 unchanged. A fix that
#       silenced the refusal would be a worse defect than the chain.
#   E2b the same for `### JOB-FATAL NOT TAKEN`. Its producer is
#       cleanup_gated.sh's job-fatal branch (read by cleanup_absent_root_guard's
#       J arms); reproducing that branch needs a live sacct verdict, so E2b
#       fixtures the ROW and measures only what owner_continue_check does with
#       it -- which is this file's subject.
#   E3  MUTATION, so E1 can fail: on a COPY of owner_snapshot.sh the one line
#       that acts on the reason is cut, and E1's fixture MUST then continue.
################################################################################
SNAPTOOL=$HERE/owner_snapshot.sh
GATESRC=$HERE/cleanup_gated.sh
REFSRC=$HERE/../tools/script_refs.sh
if [ ! -f "$SNAPTOOL" ] || [ ! -f "$GATESRC" ]; then
  fail "E. owner_snapshot.sh or cleanup_gated.sh is missing -- the continuation arms did not run"
else
E=$W/cont; mkdir -p "$E/task/merge-h" "$E/task/tools" "$E/bin" "$E/harn/artifacts"
# A TASK-SHAPED tree: DET-1-6-a made owner_snapshot.sh REFUSE a source it cannot
# identify, so a fixture without `tools/.harness_synced_commit` is refused rc 2
# for a reason that has nothing to do with these arms (measured: job 6014994).
printf '4444444444444444444444444444444444444444\n' > "$E/task/tools/.harness_synced_commit"
cp "$GATESRC" "$E/task/merge-h/cleanup_gated.sh"
# The frozen set is the gate AND the cleanup.sh it sources. This one is a STUB
# that only prints: NOTHING IS EVER UNLINKED BY THIS GUARD, and the arms below
# are about whether a SECOND OWNER is submitted, never about deletion.
printf '#!/usr/bin/env bash\necho "### [guard stub] cleanup.sh WAS CALLED with: $*"\nexit 0\n' \
  > "$E/task/merge-h/cleanup.sh"
chmod +x "$E/task/merge-h/cleanup_gated.sh" "$E/task/merge-h/cleanup.sh"
cat > "$E/bin/sbatch" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$SBATCH_LOG"
echo 9990002
STUB
chmod +x "$E/bin/sbatch"
# D/TAG/RJ are exported EXPLICITLY: an exported value always wins over
# cleanup_gated.sh's derivation, so these arms do not depend on a harness
# directory existing under the live task root. The evidence is INCOMPLETE on
# purpose (no `.wall`), which is condition 1's refusal -- E2's shape.
ETAG=CW3GUARD$$; ERJ=99$$
printf '0\n' > "$E/harn/artifacts/$ETAG-$ERJ.rc"
printf 'x\n' > "$E/harn/artifacts/$ETAG-$ERJ.lock.log"
printf 'x\n' > "$E/harn/artifacts/$ETAG-$ERJ.pixi.lock.cert"

run_owner () {  # $1 = owner.sbatch  $2 = calls file  $3 = log  $4.. = roots
  local osb=$1 calls=$2 log=$3; shift 3
  : > "$calls"
  SBATCH_LOG=$calls PATH=$E/bin:$PATH SLURM_JOB_ID=8810001 SLURM_JOB_NAME=guard-cleanup \
    D=$E/harn TAG=$ETAG RJ=$ERJ DRY_RUN=0 \
    bash "$osb" "$@" > "$log" 2>&1
  echo $?
}
nocalls () {  # $1 = calls file -> 0 when there was no continuation
  [ ! -s "$1" ]
}

# ---- E1: every root ABSENT -> NOTHING TO DO -> no continuation --------------
EABS1=$E/roots/cert$ETAG-$ERJ-A     # names that were never created
EABS2=$E/roots/ws.$ETAG-$ERJ-B
[ -e "$EABS1" ] || [ -e "$EABS2" ] && fail "E1. the absent fixture exists on disk -- the arm would prove nothing"
mkdir -p "$E/jr1"
OWNER_WALL_FLOOR_S=60 bash "$SNAPTOOL" "$E/jr1" "$E/task/merge-h/cleanup_gated.sh" \
  --roots "$EABS1" "$EABS2" > "$E/E1.snap.log" 2>&1; rcE1S=$?
if [ "$rcE1S" != 0 ] || [ ! -f "$E/jr1/owner-snapshot/owner.sbatch" ]; then
  fail "E1. owner_snapshot.sh produced no owner (rc=$rcE1S)"; sed 's/^/GUARD:   /' "$E/E1.snap.log"
else
  rcE1=$(run_owner "$E/jr1/owner-snapshot/owner.sbatch" "$E/E1.calls" "$E/E1.log" "$EABS1" "$EABS2")
  # THE ESTIMATE IS A SUBMIT-TIME DEVICE. At submit the roots do not exist yet
  # (the owner queues `--dependency=afterany:<relock>` before the job that
  # creates them has run), so an absent root must be estimated or the wall would
  # be derived for an empty tree. INSIDE the owner that reasoning is inverted:
  # absent means there is no tree, so there is nothing to unlink and nothing to
  # size. 6020524 censused `entries=14400000 present=0 absent=4`.
  if grep -q '^### OWNER WALL CENSUS entries=0 present=0 absent=2 ' "$E/E1.log"; then
    ok "E1. the RUN-TIME census counts an absent root as ZERO: $(grep -m1 '^### OWNER WALL CENSUS' "$E/E1.log")"
  else
    fail "E1. the run-time census is not entries=0 present=0 absent=2 -- the submit-time estimate has leaked into the owner (6020524 printed entries=14400000 for four roots that did not exist)"
    grep '^### OWNER WALL CENSUS' "$E/E1.log" | sed 's/^/GUARD:   /'
  fi
  grep -q '^### NOTHING TO DO' "$E/E1.log" \
    && ok "E1. the REAL gate reached its no-op verdict: $(grep -m1 '^### NOTHING TO DO' "$E/E1.log")" \
    || { fail "E1. the gate did not print NOTHING TO DO -- the fixture is not the shape the arm claims"; sed 's/^/GUARD:   /' "$E/E1.log"; }
  grep -q '^### OWNER NO CONTINUATION: the gate said NOTHING TO DO' "$E/E1.log" \
    && ok "E1. and the owner says out loud why it will not continue: $(grep -m1 '^### OWNER NO CONTINUATION' "$E/E1.log")" \
    || { fail "E1. no '### OWNER NO CONTINUATION: the gate said NOTHING TO DO' row"; grep '^### OWNER' "$E/E1.log" | sed 's/^/GUARD:   /'; }
  if nocalls "$E/E1.calls"; then
    ok "E1. NO CONTINUATION as a byte fact: the sbatch calls file is empty"
  else
    fail "E1. a no-op pass submitted a successor -- this is 6020524's chain:"; sed 's/^/GUARD:   sbatch /' "$E/E1.calls"
  fi
  # The owner prints `### CONTINUATION depth=$OWNER_CONT_N ...` on every start,
  # so depth=0 is this owner announcing ITSELF; a SUCCESSOR is depth>=1.
  grep -qE '^### CONTINUATION depth=[1-9]' "$E/E1.log" \
    && { fail "E1. a depth>=1 CONTINUATION row was printed by a pass that did nothing"; grep '^### CONTINUATION' "$E/E1.log" | sed 's/^/GUARD:   /'; } \
    || ok "E1. and no successor link was announced (only this owner's own depth=0 row)"
  [ "$rcE1" = 0 ] && ok "E1. the owner exits the gate's rc (0) -- a no-op is not an error" \
                  || fail "E1. rc=$rcE1, but the gate's no-op branch exits 0"
fi

# ---- E2: a PRESENT root the gate refuses -> no continuation, rc still 2 -----
EPRES=$E/roots/cert$ETAG-$ERJ-C; mkdir -p "$EPRES"; : > "$EPRES/f1"
mkdir -p "$E/jr2"
OWNER_WALL_FLOOR_S=60 bash "$SNAPTOOL" "$E/jr2" "$E/task/merge-h/cleanup_gated.sh" \
  --roots "$EPRES" > "$E/E2.snap.log" 2>&1; rcE2S=$?
if [ "$rcE2S" != 0 ] || [ ! -f "$E/jr2/owner-snapshot/owner.sbatch" ]; then
  fail "E2. owner_snapshot.sh produced no owner (rc=$rcE2S)"; sed 's/^/GUARD:   /' "$E/E2.snap.log"
else
  rcE2=$(run_owner "$E/jr2/owner-snapshot/owner.sbatch" "$E/E2.calls" "$E/E2.log" "$EPRES")
  grep -q '^### CLEANUP REFUSED' "$E/E2.log" \
    && ok "E2. the REAL gate refused: $(grep -m1 '^### CLEANUP REFUSED' "$E/E2.log")" \
    || { fail "E2. the gate did not refuse -- the fixture is not the shape the arm claims"; sed 's/^/GUARD:   /' "$E/E2.log"; }
  if grep -qE '^### OWNER PASS RESULT rc=2 removed=0 remaining=[0-9]+ present=1 ' "$E/E2.log"; then
    ok "E2. the pass result says a refusal removed nothing: $(grep -m1 '^### OWNER PASS RESULT' "$E/E2.log")"
  else
    fail "E2. no '### OWNER PASS RESULT rc=2 removed=0 ... present=1' row"
    grep '^### OWNER PASS RESULT' "$E/E2.log" | sed 's/^/GUARD:   /'
  fi
  grep -q '^### OWNER NO CONTINUATION: the gate said CLEANUP REFUSED' "$E/E2.log" \
    && ok "E2. and it names the refusal as its reason for stopping: $(grep -m1 '^### OWNER NO CONTINUATION' "$E/E2.log")" \
    || { fail "E2. no '### OWNER NO CONTINUATION: the gate said CLEANUP REFUSED' row"; grep '^### OWNER' "$E/E2.log" | sed 's/^/GUARD:   /'; }
  if nocalls "$E/E2.calls"; then
    ok "E2. NO CONTINUATION as a byte fact: the sbatch calls file is empty"
  else
    fail "E2. a refused pass submitted a successor to refuse again -- 6020631/6020952/6021096:"; sed 's/^/GUARD:   sbatch /' "$E/E2.calls"
  fi
  # THE RC IS THE POINT OF THIS ARM. Silencing the refusal would be a worse
  # defect than the chain: law 9's rc is what reaches Slurm and the watcher.
  [ "$rcE2" = 2 ] \
    && ok "E2. AND THE RC IS UNCHANGED: the owner still exits 2, exactly as before the fix" \
    || fail "E2. the owner exited $rcE2 -- the fix swallowed the gate's refusal, which is worse than the chain it removes"
  [ -d "$EPRES" ] && ok "E2. nothing was deleted -- the refused root is still on disk" || fail "E2. THE ROOT IS GONE"
fi

# ---- E2b: the JOB-FATAL NOT TAKEN shape -------------------------------------
# THE FIXTURE PRINTS THE JOB-FATAL ROW AND NOTHING ELSE, ON PURPOSE. In
# production the two rows come together: `job_fatal_check`'s NOT-TAKEN branch
# `return 0`s and the evidence conditions below it then end at
# `### CLEANUP REFUSED` (6020631/6020952/6021096 carry both), and
# owner_continue_check tests CLEANUP REFUSED first, so a real log is attributed
# to the refusal. Both are terminal and the verdict is identical either way --
# but a fixture carrying both would exercise the refusal clause a second time
# and never the job-fatal one, so this arm isolates the clause it names. The
# first run of this arm (job 6021762) printed exactly that RED, which is the
# guard catching its own fixture.
mkdir -p "$E/jf"
printf '#!/usr/bin/env bash\necho "### JOB-FATAL NOT TAKEN: FIXTURE -- $* holds a SEALED (write-stripped) directory, a provisioned store the gate refuses on purpose."\nexit 2\n' \
  > "$E/task/merge-h/cleanup_jf.sh"
chmod +x "$E/task/merge-h/cleanup_jf.sh"
OWNER_WALL_FLOOR_S=60 bash "$SNAPTOOL" "$E/jf" "$E/task/merge-h/cleanup_jf.sh" \
  --roots "$EPRES" > "$E/E2b.snap.log" 2>&1
if [ ! -f "$E/jf/owner-snapshot/owner.sbatch" ]; then
  fail "E2b. owner_snapshot.sh produced no owner"; sed 's/^/GUARD:   /' "$E/E2b.snap.log"
else
  rcE2b=$(run_owner "$E/jf/owner-snapshot/owner.sbatch" "$E/E2b.calls" "$E/E2b.log" "$EPRES")
  grep -q '^### OWNER NO CONTINUATION: the gate said JOB-FATAL NOT TAKEN' "$E/E2b.log" \
    && ok "E2b. a JOB-FATAL NOT TAKEN pass does not continue: $(grep -m1 '^### OWNER NO CONTINUATION' "$E/E2b.log")" \
    || { fail "E2b. no '### OWNER NO CONTINUATION: the gate said JOB-FATAL NOT TAKEN' row"; grep '^### OWNER' "$E/E2b.log" | sed 's/^/GUARD:   /'; }
  nocalls "$E/E2b.calls" \
    && ok "E2b. and the sbatch calls file is empty" \
    || { fail "E2b. a sealed store the gate refuses on purpose got a successor:"; sed 's/^/GUARD:   sbatch /' "$E/E2b.calls"; }
  [ "$rcE2b" = 2 ] && ok "E2b. rc 2 preserved" || fail "E2b. rc=$rcE2b, want 2"
fi

# ---- E3: MUTATION -- E1 must be able to fail --------------------------------
# The fix's actuator is ONE line: `if [ -n "$reason" ]; then` inside
# owner_continue_check, which is what turns a named reason into a refusal to
# continue. Cut it on a COPY and the owner continues unconditionally, which is
# the pre-fix behaviour exactly. The line count is asserted first: an unmutated
# mutation is the classic green that proves nothing.
EMUT=$E/mut; mkdir -p "$EMUT/phase_template" "$EMUT/tools" "$E/jr3"
cp "$REFSRC" "$EMUT/tools/script_refs.sh"
cp "$HERE/../tools/owner_export.sh" "$EMUT/tools/owner_export.sh"
NANCH=$(grep -c '^  if \[ -n "\$reason" \]; then$' "$SNAPTOOL")
sed 's|^  if \[ -n "\$reason" \]; then$|  if false; then|' "$SNAPTOOL" > "$EMUT/phase_template/owner_snapshot.sh"
if [ "$NANCH" != 1 ]; then
  fail "E3. the mutation anchor matched $NANCH lines, not 1 -- the mutation is not the one arm E1 depends on"
elif cmp -s "$SNAPTOOL" "$EMUT/phase_template/owner_snapshot.sh"; then
  fail "E3. the mutation did not apply -- E1 is asserting against an unmutated file and cannot fail"
else
  OWNER_WALL_FLOOR_S=60 bash "$EMUT/phase_template/owner_snapshot.sh" "$E/jr3" \
    "$E/task/merge-h/cleanup_gated.sh" --roots "$EABS1" "$EABS2" > "$E/E3.snap.log" 2>&1
  if [ ! -f "$E/jr3/owner-snapshot/owner.sbatch" ]; then
    fail "E3. the mutant produced no owner -- the mutation arm did not run"; sed 's/^/GUARD:   /' "$E/E3.snap.log"
  else
    run_owner "$E/jr3/owner-snapshot/owner.sbatch" "$E/E3.calls" "$E/E3.log" "$EABS1" "$EABS2" >/dev/null
    NM=$( [ -f "$E/E3.calls" ] && wc -l < "$E/E3.calls" | tr -d ' ' || echo 0 )
    if [ "$NM" -ge 1 ] && grep -qE '^### CONTINUATION depth=1 ' "$E/E3.log"; then
      ok "E3. MUTATION REPRODUCED: with the reason ignored, E1's no-op fixture submits $NM successor(s) and announces depth=1 -- E1 CAN fail"
      sed 's/^/GUARD:   sbatch /' "$E/E3.calls"
    else
      fail "E3. the mutant did NOT continue (sbatch calls=$NM) -- E1 is an arm that cannot fail, which is the state cleanup_wall_guard was in before CLEANUP-WALL-3"
      grep -E '^### (OWNER|CONTINUATION)' "$E/E3.log" | sed 's/^/GUARD:   /'
    fi
  fi
fi
fi

[ "$FAIL" = 0 ] && { echo "cleanup-owner guard: ALL PASS"; exit 0; }
echo "cleanup-owner guard: FAILED"; exit 1
