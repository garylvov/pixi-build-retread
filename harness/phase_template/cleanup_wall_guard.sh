#!/usr/bin/env bash
# cleanup_wall_guard.sh -- the reader for CLEANUP-WALL-1: a cleanup owner's
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
#      resubmit is measured and nothing is queued.
#   D. MUTATION -- the guard must be able to fail. A copy of owner_snapshot.sh
#      with the derivation replaced by the floor gives the two roots of arm A
#      EQUAL walls. If D does not reproduce that, A proves nothing.
#   E. THE PRODUCTION CALL SITE (law 2). phaseN_cert.sh's real
#      cleanup_submit_or_defer must PASS --roots and CONSUME owner.wall. A
#      derivation with no caller is boarded debt, not a fix.
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
# actually unlinked something would be a guard that can destroy a root.
printf '#!/usr/bin/env bash\necho "### FIXTURE CLEANUP ran with $*"\n' > "$W/cleanup_gated.sh"
chmod +x "$W/cleanup_gated.sh"

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
mkdir -p "$W/jr.small" "$W/jr.big"
bash "$SNAPTOOL" "$W/jr.small" "$W/cleanup_gated.sh" --roots "$SMALL" > "$W/A.small.log" 2>&1; rcS=$?
bash "$SNAPTOOL" "$W/jr.big"   "$W/cleanup_gated.sh" --roots "$BIG"   > "$W/A.big.log"   2>&1; rcB=$?
WS=$(wall_of "$W/A.small.log"); WB=$(wall_of "$W/A.big.log")
if [ "$rcS" = 0 ] && [ "$rcB" = 0 ] && [ -n "$WS" ] && [ -n "$WB" ]; then
  ok "A. both snapshots printed a wall row: small=${WS}s big=${WB}s"
  grep -h '^### OWNER SNAPSHOT wall=' "$W/A.small.log" "$W/A.big.log" | sed 's/^/GUARD:   /'
  if [ "$WB" -gt "$WS" ]; then
    ok "A. the 100000-entry root gets a LARGER wall than the 100-entry root (${WB}s > ${WS}s)"
  else
    fail "A. the walls do not separate: small=${WS}s big=${WB}s -- the derivation is not reading the entry count"
  fi
  if [ "$WS" = 3600 ]; then
    ok "A. the tiny root lands on the FLOOR (3600s), which is what a floor is for"
  else
    fail "A. the 100-entry root got ${WS}s, not the 3600s floor"
  fi
else
  fail "A. snapshot rc small=$rcS big=$rcB, wall rows small='$WS' big='$WB'"
  sed 's/^/GUARD:   /' "$W/A.small.log" "$W/A.big.log"
fi

########## B. owner.wall is a pasteable --time ################################
WF=$W/jr.big/owner-snapshot/owner.wall
if [ -s "$WF" ] && grep -qE '^--time=[0-9]{2,}:[0-9]{2}:[0-9]{2}$' "$WF"; then
  ok "B. owner.wall carries a well-formed wall: $(cat "$WF")"
else
  fail "B. owner.wall is missing or malformed: $( [ -f "$WF" ] && cat "$WF" || echo '<absent>' )"
fi
if grep -qE '^#SBATCH --time=[0-9]{2,}:[0-9]{2}:[0-9]{2}$' "$W/jr.big/owner-snapshot/owner.sbatch"; then
  ok "B. the generated owner.sbatch carries the derived #SBATCH --time directive too"
else
  fail "B. owner.sbatch has no #SBATCH --time directive -- an owner submitted without --time would take the 5-minute partition default"
fi

########## C. THE ACTUATOR: the owner re-derives and continues itself ##########
# The snapshot is taken while the root holds 100 entries; the root then GROWS to
# 100000 before the owner runs, which is exactly the shape of a real owner
# submitted before arm 1.
mkdir -p "$W/jr.short" "$W/bin"
GROW=$W/roots/certGROW-2
mkroot "$GROW" 100 >/dev/null
bash "$SNAPTOOL" "$W/jr.short" "$W/cleanup_gated.sh" --roots "$GROW" > "$W/C.snap.log" 2>&1
COVERS=$(sed -n 's/.*from entries=\([0-9][0-9]*\) .*/\1/p' "$W/C.snap.log" | head -1)
seq 101 100000 | ( cd "$GROW" && xargs -n 500 touch )
NG=$(find "$GROW" -maxdepth 16 | wc -l)
cat > "$W/bin/sbatch" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$SBATCH_LOG"
echo 9990001
STUB
chmod +x "$W/bin/sbatch"
: > "$W/C.sbatch"
SBATCH_LOG=$W/C.sbatch PATH=$W/bin:$PATH SLURM_JOB_ID=8800001 SLURM_JOB_NAME=guard-cleanup \
  bash "$W/jr.short/owner-snapshot/owner.sbatch" "$GROW" > "$W/C.run.log" 2>&1; rcC=$?
if grep -q "^### OWNER WALL SHORT census=$NG covers=$COVERS " "$W/C.run.log"; then
  ok "C. the owner MEASURED the growth and said so: $(grep -m1 '^### OWNER WALL SHORT' "$W/C.run.log")"
else
  fail "C. no '### OWNER WALL SHORT census=$NG covers=$COVERS' row (rc=$rcC)"
  sed 's/^/GUARD:   /' "$W/C.run.log"
fi
NSUB=$(wc -l < "$W/C.sbatch" | tr -d ' ')
CONT_TIME=$(sed -n 's/.*--time=\([0-9:]*\).*/\1/p' "$W/C.sbatch" | head -1)
if [ "$NSUB" = 1 ] && grep -q -- "$W/jr.short/owner-snapshot/owner.sbatch" "$W/C.sbatch" \
   && grep -q -- '--dependency=afterany:8800001' "$W/C.sbatch" \
   && grep -q -- 'OWNER_CONT_N=1' "$W/C.sbatch"; then
  ok "C. it resubmitted ITSELF for the remainder: 1 sbatch call, --time=$CONT_TIME, behind afterany:8800001, OWNER_CONT_N=1"
  sed 's/^/GUARD:   sbatch /' "$W/C.sbatch"
else
  fail "C. the continuation was not submitted as expected: sbatch calls=$NSUB"
  [ -s "$W/C.sbatch" ] && sed 's/^/GUARD:   sbatch /' "$W/C.sbatch"
fi
if [ -n "$CONT_TIME" ] && [ "$CONT_TIME" != "$(sed -n 's/^--time=//p' "$W/jr.short/owner-snapshot/owner.wall")" ]; then
  ok "C. the continuation's wall ($CONT_TIME) is re-derived, not a copy of the original ($(cat "$W/jr.short/owner-snapshot/owner.wall"))"
else
  fail "C. the continuation reused the original wall -- a continuation with the SAME too-small wall is the timeout again"
fi
grep -q '^### FIXTURE CLEANUP ran with' "$W/C.run.log" \
  && ok "C. and it still exec'd the frozen cleanup for what DOES fit (remove what fits, continue the rest)" \
  || fail "C. the owner never reached the frozen cleanup -- a continuation that replaces the pass instead of extending it removes nothing"

########## D. MUTATION: derivation removed -> equal walls ######################
mkdir -p "$W/mut/phase_template" "$W/mut/tools" "$W/jr.mut.small" "$W/jr.mut.big"
cp "$REFS" "$W/mut/tools/script_refs.sh"
sed 's/^  wall=\$(( unlink \* OWNER_WALL_MARGIN + census ))$/  wall=$OWNER_WALL_FLOOR_S/' \
  "$SNAPTOOL" > "$W/mut/phase_template/owner_snapshot.sh"
if cmp -s "$SNAPTOOL" "$W/mut/phase_template/owner_snapshot.sh"; then
  fail "D. the mutation did not apply -- the derivation line was not found, so D is vacuous"
else
  bash "$W/mut/phase_template/owner_snapshot.sh" "$W/jr.mut.small" "$W/cleanup_gated.sh" --roots "$SMALL" > "$W/D.small.log" 2>&1
  bash "$W/mut/phase_template/owner_snapshot.sh" "$W/jr.mut.big"   "$W/cleanup_gated.sh" --roots "$BIG"   > "$W/D.big.log"   2>&1
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

echo
[ "$FAIL" = 0 ] && echo "CLEANUP-WALL-1 GUARD: ALL GREEN" || echo "CLEANUP-WALL-1 GUARD: SOME CHECKS FAILED"
exit "$FAIL"
