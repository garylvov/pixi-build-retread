#!/usr/bin/env bash
# cleanup_wall_guard.sh -- the reader for CLEANUP-WALL-1 and -2: a cleanup owner
# `--time` is DERIVED from the work it was handed, and an owner whose wall does
# not cover what it finds says so and submits its own continuation.
#
# THE DEFECT IT WOULD HAVE CAUGHT. det1f-cleanup 5999937 was submitted with a
# hand-typed `--time=06:00:00` for two roots holding 6.66 M entries. It spent
# 17507 s removing certDET1F-5992569 (3,587,597 entries -- 205 entries/s),
# started ws.DET1F-5992569 (3,068,868 entries) at 03:14:59 and hit its wall
# 29 min later: `sacct -j 5999937` = TIMEOUT, Elapsed 06:00:04 against
# Timelimit 06:00:00. The second root was orphaned with no owner and no row
# saying so. Every submit site in the campaign typed its own constant --
# det16_proof.sh's submit_owner() `--time=16:00:00`, phaseN_cert.sh's
# CLEANUP_SBATCH_ARGS `--time=06:00:00` -- and a constant in front of a variable
# quantity is eventually too small.
#
# WHAT THIS GUARD DOES. It drives the REAL owner_snapshot.sh over fixture roots
# of two very different sizes and asserts the derivation is a FUNCTION of the
# entry count, not a constant with extra steps:
#
#   A. a 100-entry root and a 100,000-entry root get DIFFERENT walls, both
#      printed on an `### OWNER SNAPSHOT wall=` row, the big one strictly
#      larger, and the small one at the floor.
#   B. owner.wall carries a well-formed `--time=HH:MM:SS` the caller can paste.
#   C. THE ACTUATOR. A snapshot taken when the root held 100 entries, then run
#      by the generated owner.sbatch after the root grew to 100,000: it must
#      print `### OWNER WALL SHORT census=<n> covers=100` and RESUBMIT ITSELF
#      with a larger derived wall. `sbatch` is shimmed to a recorder, so the
#      resubmit is measured and nothing is queued. CLEANUP-WALL-3 added the
#      halves that made the continuation EARNED: C0 that the decision follows
#      the pass, and the `removed=`/parent rows on it.
#   C2. THE CAP stays loud -- at depth 4 it refuses the 5th link, prints the
#      hand-run line, and submits NOTHING.
#   C3. THE UNDERIVED WALL, which is the det163 shape: an owner generated with no
#      `--roots` has covers=0 and wall=0s, cannot be short of a wall it never
#      had, and must run its pass and submit NOTHING. Before CLEANUP-WALL-3 this
#      fixture continued on every run -- 24 of the 30 det163-cleanup jobs of
#      2026-09-07 08:07-08:25 were links of that chain.
#   D. MUTATION -- the guard must be able to fail. A copy of owner_snapshot.sh
#      with the derivation replaced by the floor gives the two roots of arm A
#      EQUAL walls. If D does not reproduce that, A proves nothing.
#   E. THE PRODUCTION CALL SITE (law 2). phaseN_cert.sh's real
#      cleanup_submit_or_defer must PASS --roots and CONSUME owner.wall. A
#      derivation with no caller is boarded debt, not a fix.
#   F. CLEANUP-WALL-2. The census walks what the REAPER walks. A tree DEEPER
#      than the old `-maxdepth 16` bound is censused at the UNBOUNDED count,
#      the wall is derived from THAT count, the disagreement with the bounded
#      walk is printed with its size, and a mutant with the bound restored
#      under-counts -- which is what makes F1/F2 arms and not assertions.
#
# Usage: cleanup_wall_guard.sh          (self-contained, needs only $TMPDIR)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SNAPTOOL=$HERE/owner_snapshot.sh
CERT=$HERE/phaseN_cert.sh
REFS=$HERE/../tools/script_refs.sh
[ -f "$SNAPTOOL" ] || { echo "GUARD FATAL: $SNAPTOOL not found"; exit 2; }
[ -f "$REFS" ]     || { echo "GUARD FATAL: $REFS not found"; exit 2; }

W=$(mktemp -d "${TMPDIR:-/tmp}/cleanup-wall-guard.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
FAIL=0
fail () { echo "GUARD FAIL: $*"; FAIL=1; }
ok   () { echo "GUARD  ok : $*"; }

# ---- fixtures ---------------------------------------------------------------
# The frozen "cleanup" is a stub: this guard measures the WALL, and a guard that
# actually unlinked something would be a guard that can destroy a root. It lives
# in a TASK-SHAPED tree -- `merge-h/` beside a `tools/.harness_synced_commit` --
# because DET-1-6-a made owner_snapshot.sh REFUSE a source it cannot identify,
# and a fixture that is neither a git worktree nor a task copy is neither
# faithful nor accepted (measured: job 6014994, rc 2 on every arm).
mkdir -p "$W/task/merge-h" "$W/task/tools"
printf '2222222222222222222222222222222222222222\n' > "$W/task/tools/.harness_synced_commit"
GATE=$W/task/merge-h/cleanup_gated.sh
# CLEANUP-WALL-3: the stub also emits the `### removed <root> rc=...` row, in
# the shape cleanup.sh line 108 emits it, because since CLEANUP-WALL-3 a
# continuation is EARNED by a pass that removed something and arm C's stub
# removed nothing. It still unlinks nothing -- it prints the row and returns.
printf '#!/usr/bin/env bash\necho "### FIXTURE CLEANUP ran with $*"\nfor r in "$@"; do echo "### removed $r rc=0 wall=0s exists_after=YES (FIXTURE: nothing was unlinked)"; done\nexit 0\n' > "$GATE"
chmod +x "$GATE"

mkroot () {   # $1 = path, $2 = how many files
  mkdir -p "$1" || return 2
  seq 1 "$2" | ( cd "$1" && xargs -n 500 touch )
  find "$1" -maxdepth 16 | wc -l
}
SMALL=$W/roots/certGUARD-1        # ~100 entries
BIG=$W/roots/ws.GUARD-1           # ~100000 entries
NS=$(mkroot "$SMALL" 100)
NB=$(mkroot "$BIG" 100000)
ok "fixtures: $SMALL has $NS entries, $BIG has $NB entries"

wall_of () {  # $1 = log file -> the wall in SECONDS off the OWNER SNAPSHOT row
  sed -n 's/^### OWNER SNAPSHOT wall=\([0-9][0-9]*\) .*/\1/p' "$1" | head -1
}

########## A. the wall is a function of the entry count ########################
# TWO regimes, and both matter. With the shipped 3600 s floor a 100,000-entry
# root is still SMALLER than the floor covers (2*100001/200 + 2*100001/2800 =
# 1074 s at the measured census rate, 1120 s at the old 1700), so a fixture at
# that size measures the floor and not the derivation
# -- the first run of this guard (job 6014847) printed exactly that and went
# red, which is the guard doing its job on itself. So: A1 measures the floor at
# the shipped constants, and A2 lowers the floor through its documented
# override so the derivation is what decides. A fixture big enough to clear a
# one-hour floor would need ~360,000 entries and would measure NFS, not this.
mkdir -p "$W/jr.small" "$W/jr.fsmall" "$W/jr.fbig"
bash "$SNAPTOOL" "$W/jr.small" "$GATE" --roots "$SMALL" > "$W/A1.small.log" 2>&1; rcS=$?
W1S=$(wall_of "$W/A1.small.log")
if [ "$rcS" = 0 ] && [ "$W1S" = 3600 ]; then
  ok "A1. at the shipped constants a 101-entry root lands on the 3600s FLOOR, which is what a floor is for"
else
  fail "A1. rc=$rcS wall='$W1S' (wanted the 3600s floor)"; sed 's/^/GUARD:   /' "$W/A1.small.log"
fi

OWNER_WALL_FLOOR_S=60 bash "$SNAPTOOL" "$W/jr.fsmall" "$GATE" --roots "$SMALL" > "$W/A.small.log" 2>&1; rcS=$?
OWNER_WALL_FLOOR_S=60 bash "$SNAPTOOL" "$W/jr.fbig"   "$GATE" --roots "$BIG"   > "$W/A.big.log"   2>&1; rcB=$?
WS=$(wall_of "$W/A.small.log"); WB=$(wall_of "$W/A.big.log")
# The PREDICTION, computed here independently of the tool: margin 2 on the
# unlink term at 200 entries/s, plus 2 census walks at 2800 entries/s (the
# MEASURED rate -- see the constant block in owner_snapshot.sh; it read 1700, a
# steward's sentence, until CLEANUP-WALL-2 measured five real walks), both
# rounded up. If the tool and this line disagree, one of them is wrong and the
# guard says so rather than accepting whatever was printed.
WANT=$(awk -v e="$NB" 'BEGIN{u=int((e+199)/200); c=int((e+2799)/2800); print u*2 + c*2}')
if [ "$rcS" = 0 ] && [ "$rcB" = 0 ] && [ -n "$WS" ] && [ -n "$WB" ]; then
  ok "A2. both snapshots printed a wall row: small=${WS}s big=${WB}s (floor overridden to 60s)"
  grep -h '^### OWNER SNAPSHOT wall=' "$W/A.small.log" "$W/A.big.log" | sed 's/^/GUARD:   /'
  if [ "$WB" -gt "$WS" ]; then
    ok "A2. the $NB-entry root gets a LARGER wall than the $NS-entry root (${WB}s > ${WS}s)"
  else
    fail "A2. the walls do not separate: small=${WS}s big=${WB}s -- the derivation is not reading the entry count"
  fi
  if [ "$WB" = "$WANT" ]; then
    ok "A2. and the big root's wall is EXACTLY the formula's ${WANT}s -- 2x unlink at 200/s plus 2 census walks at 2800/s"
  else
    fail "A2. the derived wall ${WB}s is not the predicted ${WANT}s"
  fi
else
  fail "A2. snapshot rc small=$rcS big=$rcB, wall rows small='$WS' big='$WB'"
  sed 's/^/GUARD:   /' "$W/A.small.log" "$W/A.big.log"
fi
########## B. owner.wall is a pasteable --time ################################
WF=$W/jr.fbig/owner-snapshot/owner.wall
if [ -s "$WF" ] && grep -qE '^--time=[0-9]{2,}:[0-9]{2}:[0-9]{2}$' "$WF"; then
  ok "B. owner.wall carries a well-formed wall: $(cat "$WF")"
else
  fail "B. owner.wall is missing or malformed: $( [ -f "$WF" ] && cat "$WF" || echo '<absent>' )"
fi
if grep -qE '^#SBATCH --time=[0-9]{2,}:[0-9]{2}:[0-9]{2}$' "$W/jr.fbig/owner-snapshot/owner.sbatch"; then
  ok "B. the generated owner.sbatch carries the derived #SBATCH --time directive too"
else
  fail "B. owner.sbatch has no #SBATCH --time directive -- an owner submitted without --time would take the 5-minute partition default"
fi

########## C. THE ACTUATOR: the owner re-derives and continues itself ##########
# The snapshot is taken while the root holds 100 entries; the root then GROWS to
# 100000 before the owner runs, which is exactly the shape of a real owner
# submitted before arm 1. The floor is lowered for the same reason as A2: at
# 3600 s both the original and the continuation would sit on the floor and the
# re-derivation would be invisible.
#
# CLEANUP-WALL-3 (2026-09-07) REWROTE THIS ARM, AND THE REASON IS THAT IT COULD
# ONLY SAY YES. Until today every arm here asserted that a continuation HAPPENS
# and not one asserted that one does NOT, so a guard that was all green sat over
# a runaway: on 2026-09-07 08:07-08:25 thirty det163-cleanup jobs ran, twenty-four
# of them self-submitted continuations, and every one of them reached a pass that
# had nothing to do (`### NOTHING TO DO`) or refused (`### CLEANUP REFUSED`,
# `### JOB-FATAL NOT TAKEN`) -- because the decision ran at job START, above
# `exec bash <gate>`, against `OWNER_WALL_COVERS` that is 0 for any owner whose
# submitter passed no `--roots`. Against covers=0 every census is "short".
# C now proves the YES half with the halves that were missing (the pass removed
# something, the census is non-zero, the wall is real and consumed); C2 proves
# the cap; C3 proves the NO half on the exact det163 shape. The three
# no-continuation arms over the gate's own verdicts live in
# cleanup_owner_guard.sh, whose subject -- exactly ONE owner per root -- is what
# a continuation chain violates 24 times over.
#
# OWNER_WALL_PRESSURE_NUM=0 is set at SNAPSHOT time (the generated owner.sbatch
# carries the value as a literal assignment, so setting it on the owner's own env
# would be overwritten). 0/5 of any wall is always reached, which makes this arm
# test the continuation CONDITION and not the clock: a fixture that had to burn
# 4/5 of a real wall would measure NFS.
mkdir -p "$W/jr.short" "$W/bin"
GROW=$W/roots/certGROW-2
mkroot "$GROW" 100 >/dev/null
OWNER_WALL_FLOOR_S=60 OWNER_WALL_PRESSURE_NUM=0 \
  bash "$SNAPTOOL" "$W/jr.short" "$GATE" --roots "$GROW" > "$W/C.snap.log" 2>&1
COVERS=$(sed -n 's/.*from entries=\([0-9][0-9]*\) .*/\1/p' "$W/C.snap.log" | head -1)
seq 101 100000 | ( cd "$GROW" && xargs -n 500 touch )
NG=$(find "$GROW" | wc -l)
cat > "$W/bin/sbatch" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$SBATCH_LOG"
echo 9990001
STUB
chmod +x "$W/bin/sbatch"
: > "$W/C.sbatch"
SBATCH_LOG=$W/C.sbatch PATH=$W/bin:$PATH SLURM_JOB_ID=8800001 SLURM_JOB_NAME=guard-cleanup \
  bash "$W/jr.short/owner-snapshot/owner.sbatch" "$GROW" > "$W/C.run.log" 2>&1; rcC=$?
# C0: the order. MEASURE, RUN, THEN DECIDE -- the pass's rows must appear BEFORE
# the continuation decision, because the decision reading them is the whole fix.
LN_PASS=$(grep -n '^### FIXTURE CLEANUP ran with' "$W/C.run.log" | head -1 | cut -d: -f1)
LN_DEC=$(grep -n '^### OWNER PASS RESULT ' "$W/C.run.log" | head -1 | cut -d: -f1)
if [ -n "$LN_PASS" ] && [ -n "$LN_DEC" ] && [ "$LN_DEC" -gt "$LN_PASS" ]; then
  ok "C0. the pass ran BEFORE the continuation decision (gate row line $LN_PASS, decision row line $LN_DEC) -- the pre-fix owner decided at line 0"
else
  fail "C0. the decision does not follow the pass (gate row line='$LN_PASS' decision row line='$LN_DEC') -- the continuation cannot be reading anything the pass did"
  sed 's/^/GUARD:   /' "$W/C.run.log"
fi
if grep -q "^### OWNER PASS RESULT rc=0 removed=1 remaining=$NG " "$W/C.run.log"; then
  ok "C. the pass result is MEASURED and printed: $(grep -m1 '^### OWNER PASS RESULT' "$W/C.run.log")"
else
  fail "C. no '### OWNER PASS RESULT rc=0 removed=1 remaining=$NG' row (rc=$rcC)"
  grep '^### OWNER PASS RESULT' "$W/C.run.log" | sed 's/^/GUARD:   /'
  sed 's/^/GUARD:   /' "$W/C.run.log"
fi
if grep -q "^### OWNER WALL SHORT census=$NG covers=$COVERS .* removed=1$" "$W/C.run.log"; then
  ok "C. the owner MEASURED the growth and said so, with what it removed on the row: $(grep -m1 '^### OWNER WALL SHORT' "$W/C.run.log")"
else
  fail "C. no '### OWNER WALL SHORT census=$NG covers=$COVERS ... removed=1' row (rc=$rcC)"
  sed 's/^/GUARD:   /' "$W/C.run.log"
fi
if grep -q '^### CONTINUATION depth=1 parent=8800001 parent_removed=1 ' "$W/C.run.log"; then
  ok "C. and the chain is readable from this one log: $(grep -m1 '^### CONTINUATION depth=1' "$W/C.run.log")"
else
  fail "C. no '### CONTINUATION depth=1 parent=8800001 parent_removed=1' row -- a chain whose links do not name their parent cannot be traced from any one of its logs"
  grep '^### CONTINUATION' "$W/C.run.log" | sed 's/^/GUARD:   /'
fi
NSUB=$(wc -l < "$W/C.sbatch" | tr -d ' ')
CONT_TIME=$(sed -n 's/.*--time=\([0-9:]*\).*/\1/p' "$W/C.sbatch" | head -1)
if [ "$NSUB" = 1 ] && grep -q -- "$W/jr.short/owner-snapshot/owner.sbatch" "$W/C.sbatch" \
   && grep -q -- '--dependency=afterany:8800001' "$W/C.sbatch" \
   && grep -q -- 'OWNER_CONT_N=1' "$W/C.sbatch" \
   && grep -q -- 'OWNER_CONT_PARENT=8800001' "$W/C.sbatch" \
   && grep -q -- 'OWNER_CONT_PARENT_REMOVED=1' "$W/C.sbatch"; then
  ok "C. it resubmitted ITSELF for the remainder: 1 sbatch call, --time=$CONT_TIME, behind afterany:8800001, OWNER_CONT_N=1, parent and parent_removed carried"
  sed 's/^/GUARD:   sbatch /' "$W/C.sbatch"
else
  fail "C. the continuation was not submitted as expected: sbatch calls=$NSUB"
  [ -s "$W/C.sbatch" ] && sed 's/^/GUARD:   sbatch /' "$W/C.sbatch"
fi
# OWNER-EXPORT-1 stays enforced on the row CLEANUP-WALL-3 rewrote: `--export=ALL`
# makes Slurm retrieve the submitter's environment on the target node, and a
# failed retrieval HOLDS the job -- 6013350/6013351/6014485/5841188.
if grep -q -- '--export=ALL' "$W/C.sbatch"; then
  fail "C. the continuation line carries --export=ALL -- that is the held-job shape OWNER-EXPORT-1 removed"
  sed 's/^/GUARD:   sbatch /' "$W/C.sbatch"
else
  ok "C. and the continuation line carries NO --export=ALL (OWNER-EXPORT-1 holds across the rewrite)"
fi
ORIG_TIME=$(sed -n 's/^--time=//p' "$W/jr.short/owner-snapshot/owner.wall")
if [ -n "$CONT_TIME" ] && [ "$CONT_TIME" != "$ORIG_TIME" ]; then
  ok "C. the continuation's wall ($CONT_TIME) is RE-DERIVED from the measured census, not a copy of the original ($ORIG_TIME)"
else
  fail "C. the continuation reused the original wall ($ORIG_TIME) -- a continuation with the SAME too-small wall is the timeout again"
fi
grep -q '^### FIXTURE CLEANUP ran with' "$W/C.run.log" \
  && ok "C. and it still ran the frozen cleanup for what DOES fit (remove what fits, continue the rest)" \
  || fail "C. the owner never reached the frozen cleanup -- a continuation that replaces the pass instead of extending it removes nothing"
if [ "$rcC" = 0 ]; then
  ok "C. the owner exits the GATE's rc (0) -- the continuation machinery does not invent an exit status"
else
  fail "C. the owner exited $rcC while the gate exited 0 -- the pass's rc was not preserved"
fi

# ---- C2, THE CAP: it stays loud, and it is the last resort, not the control --
: > "$W/C2.sbatch"
SBATCH_LOG=$W/C2.sbatch PATH=$W/bin:$PATH SLURM_JOB_ID=8800002 SLURM_JOB_NAME=guard-cleanup \
  OWNER_CONT_N=4 bash "$W/jr.short/owner-snapshot/owner.sbatch" "$GROW" > "$W/C2.run.log" 2>&1
C2SUB=$(wc -l < "$W/C2.sbatch" | tr -d ' ')
if grep -q "^### OWNER WALL CONTINUATION CAP HIT depth=4 max=4 removed=1 remaining=$NG$" "$W/C2.run.log" \
   && grep -q '^    env -u SLURM_JOB_ID sbatch --partition=' "$W/C2.run.log" \
   && [ "$C2SUB" = 0 ]; then
  ok "C2. at the cap it refuses the 5th link, prints the hand-run line, and submits nothing: $(grep -m1 'CAP HIT' "$W/C2.run.log")"
else
  fail "C2. the cap did not bite as expected (sbatch calls=$C2SUB)"
  grep -E 'CAP HIT|env -u SLURM_JOB_ID' "$W/C2.run.log" | sed 's/^/GUARD:   /'
  [ -s "$W/C2.sbatch" ] && sed 's/^/GUARD:   sbatch /' "$W/C2.sbatch"
fi

# ---- C3, THE UNDERIVED WALL: the det163 shape, exactly ----------------------
# An owner generated with NO `--roots` has OWNER_WALL_COVERS=0 and OWNER_WALL_S=0
# -- placeholders, not measurements -- and det163_proof.sh's submit_owner
# generated exactly that (`### OWNER SNAPSHOT wall=UNDERIVED roots=0` is in
# det163-6020526.out twice). Before CLEANUP-WALL-3 this fixture produced a
# continuation on EVERY run, four deep, which is the whole chain. It must now
# print the UNDERIVED row, say out loud that it will not continue, and submit
# nothing: that is what turns a row nobody read into an actuator (law 9).
mkdir -p "$W/jr.und"
OWNER_WALL_FLOOR_S=60 OWNER_WALL_PRESSURE_NUM=0 \
  bash "$SNAPTOOL" "$W/jr.und" "$GATE" > "$W/C3.snap.log" 2>&1; rcU=$?
if [ "$rcU" = 0 ] && [ -f "$W/jr.und/owner-snapshot/owner.sbatch" ]; then
  : > "$W/C3.sbatch"
  SBATCH_LOG=$W/C3.sbatch PATH=$W/bin:$PATH SLURM_JOB_ID=8800003 SLURM_JOB_NAME=guard-cleanup \
    bash "$W/jr.und/owner-snapshot/owner.sbatch" "$GROW" > "$W/C3.run.log" 2>&1
  C3SUB=$(wc -l < "$W/C3.sbatch" | tr -d ' ')
  grep -q '^### OWNER WALL UNDERIVED' "$W/C3.run.log" \
    && ok "C3. the owner NAMES its underived wall at run time: $(grep -m1 '^### OWNER WALL UNDERIVED' "$W/C3.run.log")" \
    || { fail "C3. no '### OWNER WALL UNDERIVED' row -- the placeholder is being read as a measurement"; sed 's/^/GUARD:   /' "$W/C3.run.log"; }
  grep -q '^### OWNER NO CONTINUATION: this owner has no derived wall' "$W/C3.run.log" \
    && ok "C3. and it REFUSES to continue on it: $(grep -m1 '^### OWNER NO CONTINUATION' "$W/C3.run.log")" \
    || { fail "C3. no '### OWNER NO CONTINUATION: this owner has no derived wall' row"; grep '^### OWNER' "$W/C3.run.log" | sed 's/^/GUARD:   /'; }
  if [ "$C3SUB" = 0 ]; then
    ok "C3. and the calls file is EMPTY -- no continuation, as a byte fact and not as a missing log line"
  else
    fail "C3. the underived owner submitted $C3SUB continuation(s) -- this is det163's chain, unfixed"
    sed 's/^/GUARD:   sbatch /' "$W/C3.sbatch"
  fi
  grep -q '^### FIXTURE CLEANUP ran with' "$W/C3.run.log" \
    && ok "C3. and it still RAN its pass -- an underived wall stops the continuation, never the cleanup" \
    || fail "C3. the underived owner never ran its pass"
else
  fail "C3. the snapshot tool would not generate an underived owner (rc=$rcU) -- C3 did not run"
  sed 's/^/GUARD:   /' "$W/C3.snap.log"
fi
########## D. MUTATION: derivation removed -> equal walls ######################
mkdir -p "$W/mut/phase_template" "$W/mut/tools" "$W/jr.mut.small" "$W/jr.mut.big"
cp "$REFS" "$W/mut/tools/script_refs.sh"
# OWNER-EXPORT-1: owner_snapshot.sh now REFUSES without owner_export.sh beside it
# (it must ship the clause producer into the generated sbatch by `declare -f`),
# so a mutant copy needs the same sibling or every arm below dies rc 2 for a
# reason that has nothing to do with the mutation.
cp "$HERE/../tools/owner_export.sh" "$W/mut/tools/owner_export.sh"
sed 's/^  wall=\$(( unlink \* OWNER_WALL_MARGIN + census ))$/  wall=$OWNER_WALL_FLOOR_S/' \
  "$SNAPTOOL" > "$W/mut/phase_template/owner_snapshot.sh"
if cmp -s "$SNAPTOOL" "$W/mut/phase_template/owner_snapshot.sh"; then
  fail "D. the mutation did not apply -- the derivation line was not found, so D is vacuous"
else
  OWNER_WALL_FLOOR_S=60 bash "$W/mut/phase_template/owner_snapshot.sh" "$W/jr.mut.small" "$GATE" --roots "$SMALL" > "$W/D.small.log" 2>&1
  OWNER_WALL_FLOOR_S=60 bash "$W/mut/phase_template/owner_snapshot.sh" "$W/jr.mut.big"   "$GATE" --roots "$BIG"   > "$W/D.big.log"   2>&1
  MS=$(wall_of "$W/D.small.log"); MB=$(wall_of "$W/D.big.log")
  if [ -n "$MS" ] && [ "$MS" = "$MB" ]; then
    ok "D. MUTATION REPRODUCED: with the derivation replaced by the floor both roots get ${MS}s -- so arm A is measuring the derivation and not the weather"
  else
    fail "D. the mutant still separated the walls (small=$MS big=$MB) -- arm A cannot be trusted"
  fi
fi

########## E. the production call site #########################################
if [ -f "$CERT" ]; then
  SUBMIT=$(awk '/^cleanup_submit_or_defer \(\) \{/{p=1} p{print} p&&/^\}$/{exit}' "$CERT")
  if printf '%s' "$SUBMIT" | grep -q -- '--roots "\$@"'; then
    ok "E. phaseN_cert.sh's cleanup_submit_or_defer hands the ROOTS to the snapshot tool"
  else
    fail "E. cleanup_submit_or_defer does not pass --roots -- the derivation has no production caller"
  fi
  if printf '%s' "$SUBMIT" | grep -q 'owner-snapshot/owner.wall'; then
    ok "E. and it CONSUMES owner.wall for the submitted --time"
  else
    fail "E. cleanup_submit_or_defer never reads owner.wall -- the derived wall is written and never read"
  fi
else
  fail "E. $CERT not found -- the production call site could not be checked"
fi


########## F. CLEANUP-WALL-2: the census walks what the REAPER walks ###########
# THE DEFECT, MEASURED ON JOB 6017160 (REAP-3). `owner_census` walked
# `find "$r" -maxdepth 16` while the work it sizes -- cleanup.sh's
# `N=$(find "$r" | wc -l)` and the `rm -rf` behind it -- is UNBOUNDED. On ONE
# reap of a STATIC tree the two disagreed twice: 44040 against 44354, and 423482
# against 434860. Not a race: the smoke's `cp -al` mirror nests worktrees inside
# worktrees, and the same job's SETUP-REFUSED branch named a real path at depth 8
# inside `.claude/worktrees/agent-.../assets/cad/`. A census that under-counts
# buys a wall too short, which is 5999937's TIMEOUT with a root half gone.
#
#   F1  a tree DEEPER than the old bound is censused at the UNBOUNDED count --
#       the same number `find "$r" | wc -l` gives, which is cleanup.sh's own walk.
#   F2  the wall DERIVED from it is the wall the unbounded count buys, not the
#       bounded one. F1 without F2 would pass a census that counts correctly and
#       then sizes off something else.
#   F3  the disagreement is PRINTED for this one release, with the undercount.
#   F4  MUTATION -- the bound restored on a COPY of the producer -> the census
#       under-counts and the derived wall is SMALLER. Without this, F1 and F2 are
#       green against any tree shallower than the bound and prove nothing.
DEEPGUARD=$W/roots/certDEEP-1
mkdir -p "$DEEPGUARD" || fail "F. could not make the deep fixture"
# 22 levels: past the old 16 bound with margin, and 40 files at the bottom so the
# undercount is a number and not a rounding artefact.
DPATH=$DEEPGUARD
for i in $(seq 1 22); do DPATH=$DPATH/d$i; done
mkdir -p "$DPATH"
seq 1 40 | ( cd "$DPATH" && xargs -n 40 touch )
seq 1 10 | ( cd "$DEEPGUARD" && xargs -n 10 touch )
F_UNB=$(find "$DEEPGUARD" | wc -l)
F_B16=$(find "$DEEPGUARD" -maxdepth 16 | wc -l)
if [ "$F_UNB" -gt "$F_B16" ]; then
  ok "F. NON-VACUITY: the fixture is deeper than 16 -- unbounded=$F_UNB depth16=$F_B16 undercount=$((F_UNB - F_B16))"
else
  fail "F. the fixture is NOT deeper than the old bound (unbounded=$F_UNB depth16=$F_B16) -- arms F1-F4 would prove nothing"
fi

FLOG=$W/F.log
( cd "$W/task" && bash "$SNAPTOOL" "$W/task/jobF" "$GATE" --roots "$DEEPGUARD" ) > "$FLOG" 2>&1
F_CENSUS=$(sed -n 's/^### OWNER SNAPSHOT wall=[0-9]* .* from entries=\([0-9][0-9]*\) .*/\1/p' "$FLOG" | head -1)
F_WALL=$(wall_of "$FLOG")
if [ "$F_CENSUS" = "$F_UNB" ]; then
  ok "F1. the census is the UNBOUNDED count, byte-equal to cleanup.sh's own walk (entries=$F_CENSUS)"
else
  fail "F1. census=$F_CENSUS but the reaper will walk $F_UNB -- the wall is being bought for a tree nobody removes"
  sed 's/^/      /' "$FLOG" | head -20
fi
# F2: the same derivation the file ships, computed here from the UNBOUNDED count.
F_WANT=$(( ( (F_UNB + 199) / 200 ) * 2 + ( (F_UNB + 2799) / 2800 ) * 2 ))
[ "$F_WANT" -lt 3600 ] && F_WANT=3600
if [ "$F_WALL" = "$F_WANT" ]; then
  ok "F2. the derived wall ${F_WALL}s is what the UNBOUNDED count buys at the shipped constants"
else
  fail "F2. wall=${F_WALL}s but the unbounded count $F_UNB buys ${F_WANT}s"
fi
if grep -q "### OWNER CENSUS DEPTH $DEEPGUARD unbounded=$F_UNB depth16=$F_B16 undercount=$((F_UNB - F_B16))" "$FLOG"; then
  ok "F3. the disagreement is PRINTED with its size, not left in a lane's memory"
else
  fail "F3. no '### OWNER CENSUS DEPTH' row naming unbounded=$F_UNB depth16=$F_B16"
  grep 'CENSUS DEPTH' "$FLOG" | sed 's/^/      /'
fi

# F4 MUTATION: restore the bound in a COPY and the census must under-count.
FMUTD=$W/mutF/phase_template; mkdir -p "$FMUTD" "$W/mutF/tools"
cp "$REFS" "$W/mutF/tools/script_refs.sh"
cp "$HERE/../tools/owner_export.sh" "$W/mutF/tools/owner_export.sh"
FMUT=$FMUTD/owner_snapshot.sh
sed 's@^      n=$(find "$r" 2>/dev/null | wc -l)$@      n=$(find "$r" -maxdepth 16 2>/dev/null | wc -l)@' "$SNAPTOOL" > "$FMUT"
if cmp -s "$SNAPTOOL" "$FMUT"; then
  fail "F4. the mutation edited nothing -- F1/F2 are asserting against an unmutated file and cannot fail"
else
  FMLOG=$W/F4.log
  ( cd "$W/task" && bash "$FMUT" "$W/task/jobF4" "$GATE" --roots "$DEEPGUARD" ) > "$FMLOG" 2>&1
  FM_CENSUS=$(sed -n 's/^### OWNER SNAPSHOT wall=[0-9]* .* from entries=\([0-9][0-9]*\) .*/\1/p' "$FMLOG" | head -1)
  if [ "$FM_CENSUS" = "$F_B16" ] && [ "$FM_CENSUS" != "$F_UNB" ]; then
    ok "F4. (mutation) the bound restored -> census=$FM_CENSUS, under-counting the reaper's $F_UNB by $((F_UNB - FM_CENSUS)). F1 CAN fail."
  else
    fail "F4. the bounded mutant censused $FM_CENSUS (bounded=$F_B16 unbounded=$F_UNB) -- the mutation did not change the count, so F1 proves nothing"
  fi
fi
echo
[ "$FAIL" = 0 ] && echo "CLEANUP-WALL-1/2 GUARD: ALL GREEN" || echo "CLEANUP-WALL-1/2 GUARD: SOME CHECKS FAILED"
exit "$FAIL"
