#!/usr/bin/env bash
# cleanup_lock_evidence_guard.sh -- the reader for the lock-evidence half of
# condition 1 in cleanup_gated.sh.
#
# THE DEFECT IT WOULD HAVE CAUGHT. The gate's green-run check looked for the
# certified lock under two names only, `pixi.lock.cert` and
# `pixi.lock.<TAG>-<J>*`, i.e. the word `pixi` always FIRST. Eight harnesses in
# this campaign (c17c/c17w, c18a/b/c/p1/p2, c21c) write it the other way round,
# `<TAG>-<J>.pixi.lock.cert`, with the tag+job stem first -- so the gate could
# not see a lock that was sitting in the artifacts directory and refused every
# one of those runs. Measured on cleanup jobs 5814670 and 5823482, both of which
# printed `### MISSING: a green run with no pixi.lock.cert and no
# pixi.lock.C21C-<J>* in the task root` while
# `artifacts/C21C-5814669.pixi.lock.cert` (2758192 B) and
# `artifacts/C21C-5823481.pixi.lock.cert` (2758170 B) existed and were
# non-empty. Both refused correctly given what they could see -- nothing was
# deleted -- and both left their two roots stranded on disk.
#
# WHAT THIS GUARD DOES. It drives the REAL shipped `cleanup_gated.sh` -- a
# verbatim copy, proved byte-identical with `cmp` on direct file arguments,
# never a re-implementation -- over a throwaway fixture task root, with a stub
# `cleanup.sh` beside it so the gate's own `$(dirname "$0")/cleanup.sh`
# resolution can only ever reach a script that deletes nothing:
#
#   A. lock as `<TAG>-<J>.pixi.lock.cert`        -> GATE PASSED, roots handed on.
#   A2. lock as `<TAG>-<J>-ONCERT.pixi.lock.cert` -> GATE PASSED (the oncert
#      lanes carry an arm suffix on the stem, the same shape defect (b) fixed
#      for `.rc`).
#   B. lock as `pixi.lock.cert`                  -> GATE PASSED (no regression).
#   C. lock as `pixi.lock.<TAG>-<J>.gz`          -> GATE PASSED (no regression).
#   D. NON-VACUITY: a green run (`rc`=0) with NO lock in ANY shape -> the gate
#      must still REFUSE with exit 2 and hand the roots to nobody.
#
# Every case additionally asserts that both fixture roots still exist afterwards
# and that the stub cleanup ran (or did not run) as the verdict requires, since
# "nothing is deleted on a refusal" is condition 3 of the gate.
#
# Falsification: drop the `"$A/$TAG-$RJ"*"pixi.lock.cert"` term from the gate's
# lock loop and A and A2 go RED with the gate's own MISSING line, while B, C and
# D stay green.
#
# Usage: cleanup_lock_evidence_guard.sh          (self-contained, needs $TMPDIR)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GATE=$HERE/cleanup_gated.sh
[ -f "$GATE" ] || { echo "GUARD FATAL: $GATE not found"; exit 2; }

W=$(mktemp -d "${TMPDIR:-/tmp}/cleanup-lock-evidence-guard.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
FAIL=0
fail () { echo "GUARD FAIL: $*"; FAIL=1; }
ok   () { echo "GUARD  ok : $*"; }

########## containment: this guard may only ever touch $W ######################
# The gate ends in `exec bash <cleanup.sh> <roots>`, and cleanup.sh really does
# `rm -rf`. Two things keep that away from anything real: the copy of the gate
# lives in a directory whose sibling cleanup.sh is a stub, and every root passed
# in is under $W. Both are asserted, not assumed -- an unasserted containment
# claim is how a guard becomes the incident.
case $W in
  /*) ;;
  *) echo "GUARD FATAL: fixture root '$W' is not absolute"; exit 2;;
esac
case $W in
  /oscar/data/stellex/glvov/retread/*|/oscar/data/stellex/glvov/agrescap/*)
    echo "GUARD FATAL: fixture root $W is inside real campaign storage; refusing to run"; exit 2;;
esac

TPL=$W/tpl
mkdir -p "$TPL"
cp "$GATE" "$TPL/cleanup_gated.sh"
if cmp -s "$GATE" "$TPL/cleanup_gated.sh"; then
  ok "driving the REAL shipped gate: $TPL/cleanup_gated.sh is byte-identical to $GATE"
else
  echo "GUARD FATAL: the copy of the gate is not byte-identical to the shipped one"; exit 2
fi

# the stub the gate will exec instead of the real cleanup.sh
cat > "$TPL/cleanup.sh" <<STUB
#!/usr/bin/env bash
printf 'STUB CLEANUP REACHED:'
printf ' %s' "\$@"
printf '\n'
printf '%s\n' "\$@" >> "$W/handoff.log"
exit 0
STUB
chmod +x "$TPL/cleanup.sh"
grep -q 'STUB CLEANUP REACHED' "$TPL/cleanup.sh" \
  || { echo "GUARD FATAL: the stub cleanup.sh was not written"; exit 2; }
ok "stub cleanup.sh sits beside the copied gate -- the gate's own \$(dirname \$0)/cleanup.sh can only reach it"

########## the fixture task root ##############################################
TAG=GUARDLK
# A job id that Slurm will never know, so the gate's queue check reports "no
# longer in the queue" for it. Six-plus digits, as the gate's token regex wants.
RJ=99999901
D=$W/task
A=$D/artifacts
mkdir -p "$A"
ROOT_A=$W/roots/cert$TAG-$RJ
ROOT_B=$W/roots/ws.$TAG-$RJ

reset_fixture () {           # green run, all three artifacts, NO lock yet
  rm -rf "$A" "$W/roots" "$W/handoff.log"
  mkdir -p "$A" "$ROOT_A/sub" "$ROOT_B/sub"
  printf 'payload\n' > "$ROOT_A/sub/f"; printf 'payload\n' > "$ROOT_B/sub/f"
  printf '0\n'                > "$A/$TAG-$RJ.rc"
  printf '1234\n'             > "$A/$TAG-$RJ.wall"
  printf 'lock log lines...\n'> "$A/$TAG-$RJ.lock.log"
}

for r in "$ROOT_A" "$ROOT_B"; do
  case $r in
    "$W"/*) ;;
    *) echo "GUARD FATAL: fixture root $r escapes $W"; exit 2;;
  esac
done
ok "both fixture roots are inside $W (nothing real is reachable from this guard)"

# $1=label -> runs the real gate; sets GRC, GOUT
run_gate () {
  GOUT=$(D=$D TAG=$TAG RJ=$RJ bash "$TPL/cleanup_gated.sh" "$ROOT_A" "$ROOT_B" 2>&1)
  GRC=$?
}

roots_intact () { [ -s "$ROOT_A/sub/f" ] && [ -s "$ROOT_B/sub/f" ]; }

# $1=label  $2=description  ($3.. unused) -- expects a PASS that hands both roots on
expect_pass () {
  local label=$1 desc=$2
  run_gate
  if [ "$GRC" = 0 ] \
     && printf '%s\n' "$GOUT" | grep -q '^### GATE PASSED' \
     && printf '%s\n' "$GOUT" | grep -q 'STUB CLEANUP REACHED' \
     && printf '%s\n' "$GOUT" | grep -q "^### lock present: " \
     && roots_intact; then
    ok "$label. $desc -> GATE PASSED, both roots handed to the cleanup ($(printf '%s\n' "$GOUT" | sed -n 's/^### lock present: \([^ ]*\).*/\1/p' | sed "s|$A/||" | paste -sd, ))"
  else
    fail "$label. $desc -> the gate did NOT pass (rc=$GRC)"
    printf '%s\n' "$GOUT" | grep '^###' | sed 's/^/GUARD:   /'
  fi
}

########## A / A2 / B / C: every accepted lock shape must pass #################
reset_fixture
printf 'lock bytes\n' > "$A/$TAG-$RJ.pixi.lock.cert"
expect_pass A  "lock as <TAG>-<J>.pixi.lock.cert (c17/c18/c21 shape, jobs 5814670+5823482)"

reset_fixture
printf 'lock bytes\n' > "$A/$TAG-$RJ-ONCERT.pixi.lock.cert"
expect_pass A2 "lock as <TAG>-<J>-ONCERT.pixi.lock.cert (arm-suffixed stem)"

reset_fixture
printf 'lock bytes\n' > "$A/pixi.lock.cert"
expect_pass B  "lock as pixi.lock.cert (phaseN/mh1 shape)"

reset_fixture
printf 'lock bytes\n' > "$A/pixi.lock.$TAG-$RJ.gz"
expect_pass C  "lock as pixi.lock.<TAG>-<J>.gz"

########## D. NON-VACUITY: no lock in any shape must still REFUSE #############
reset_fixture
run_gate
if [ "$GRC" = 2 ] \
   && printf '%s\n' "$GOUT" | grep -q '^### MISSING: a green run with no ' \
   && printf '%s\n' "$GOUT" | grep -q '^### CLEANUP REFUSED -- nothing deleted' \
   && ! printf '%s\n' "$GOUT" | grep -q 'STUB CLEANUP REACHED' \
   && [ ! -s "$W/handoff.log" ] \
   && roots_intact; then
  ok "D. green run with NO lock in any shape -> still REFUSED (exit 2), cleanup never reached, both roots intact"
else
  fail "D. a green run with no lock at all did NOT refuse, or reached the cleanup (rc=$GRC, handoff=$( [ -s "$W/handoff.log" ] && echo yes || echo no ))"
  printf '%s\n' "$GOUT" | grep '^###' | sed 's/^/GUARD:   /'
fi

########## D2. the other half of condition 1 is untouched #####################
# A run missing an ARTIFACT must still refuse whatever the lock looks like --
# widening the lock check must not have widened anything else.
reset_fixture
rm -f "$A/$TAG-$RJ.wall"
printf 'lock bytes\n' > "$A/$TAG-$RJ.pixi.lock.cert"
run_gate
if [ "$GRC" = 2 ] && printf '%s\n' "$GOUT" | grep -q '^### MISSING/EMPTY artifact' && roots_intact; then
  ok "D2. a missing .wall still refuses even with a lock present (the artifact half is untouched)"
else
  fail "D2. a run missing its .wall was not refused (rc=$GRC)"
  printf '%s\n' "$GOUT" | grep '^###' | sed 's/^/GUARD:   /'
fi

########## D3. the ownership token is untouched ###############################
reset_fixture
printf 'lock bytes\n' > "$A/$TAG-$RJ.pixi.lock.cert"
BAD=$W/roots/cert$TAG-noid
mkdir -p "$BAD"
GOUT=$(D=$D TAG=$TAG RJ=$RJ bash "$TPL/cleanup_gated.sh" "$BAD" 2>&1); GRC=$?
if [ "$GRC" = 2 ] && printf '%s\n' "$GOUT" | grep -q '^### REFUSE: root .* carries none of the owner job ids' && [ -d "$BAD" ]; then
  ok "D3. a root with no -<jid> owner token still refuses even with a lock present (the ownership half is untouched)"
else
  fail "D3. a root with no owner token was not refused (rc=$GRC)"
  printf '%s\n' "$GOUT" | grep '^###' | sed 's/^/GUARD:   /'
fi

[ "$FAIL" = 0 ] && { echo "cleanup-lock-evidence guard: ALL PASS"; exit 0; }
echo "cleanup-lock-evidence guard: FAILED"; exit 1
