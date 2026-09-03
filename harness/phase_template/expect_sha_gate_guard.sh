#!/usr/bin/env bash
# Guard for the SNAP/EXPECT_SHA coupling defect (job 5671529, exit 8 in 3 s:
# "snapshot sha 1860e830... != 2dd790bf...").  The template used to carry TWO
# coupled toolchain constants inside the SUBSTITUTE region -- SNAP and a
# hand-written EXPECT_SHA -- and the leftover-token self-check cannot see a
# disagreement between them, because it strips that region by design.
#
# This drives the REAL phaseN_relock.sh / phaseN_cert.sh through a derivation,
# exactly as a phase harness is derived, and reads what the sha gate does.
#
#   A. pin EMPTY      -> gate DERIVES the sha and lets the run continue
#                        (the run then dies at a LATER gate, which is the proof
#                        it got past this one).
#   B. pin CORRECT    -> "PINNED and matched", run continues to the same place.
#   C. pin WRONG      -> refuses, naming both shas, at the sha gate.
#
# Falsification: restore the old `EXPECT_SHA=<literal>` + bare comparison and
# case A fails (the derived SNAP's sha will not equal the literal).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
FAIL=0
say() { printf '%s\n' "$*"; }
check() { # name expected_rc expected_substr actual_rc file
  if [ "$2" = "$4" ] && grep -qF -- "$3" "$5"; then say "  PASS  $1 (rc=$4)"; else
    say "  FAIL  $1: want rc=$2 and substring [$3]; got rc=$4"; sed -n '1,40p' "$5"; FAIL=1; fi; }

# A SNAP that is NOT the one any template ships a literal for.
printf '#!/bin/sh\necho guard-stub 0.0.0\n' > "$W/pixi-build-retread"; chmod +x "$W/pixi-build-retread"
REAL=$(sha256sum "$W/pixi-build-retread" | awk '{print $1}')
BOGUS=0000000000000000000000000000000000000000000000000000000000000000
say "guard SNAP sha256=$REAL"

# The cert half refuses before its sha gate without the relock phase's handoff
# stamp, so hand it a minimal one. Everything it declares is checked AFTER the
# sha gate, which is the gate under test.
mkdir -p "$W/harness/artifacts"
cat > "$W/harness/artifacts/relock_env.sh" <<STAMP
WS=$W/ws
LOCK=$W/ws/pixi.lock
EXPECT_LOCK_MD5=00000000000000000000000000000000
P1_JOB=999998
STAMP
# The cert's cheap-path loop runs BEFORE the sha gate and wants these to exist.
mkdir -p "$W/ws"; : > "$W/ws/pixi.toml"; : > "$W/ws/pixi.lock"

derive() { # src dst pin  -- substitutes ONLY the toolchain constants, as a derivation does
  sed -e "s|^SNAP=.*|SNAP=$W/pixi-build-retread|" \
      -e "s|^SNAPDIR=.*|SNAPDIR=$W|" \
      -e "s|^EXPECT_SHA_PIN=.*|EXPECT_SHA_PIN=$3|" \
      -e "s|^D=.*|D=$W/harness|" \
      -e "s|^P1D=.*|P1D=$W/harness|" "$1" > "$2"
  bash -n "$2" || { say "  FAIL  $2 does not parse"; FAIL=1; }
}

for TPL in phaseN_relock.sh phaseN_cert.sh; do
  say "== $TPL =="
  # C. a WRONG pin must refuse at the sha gate, naming both shas.
  derive "$HERE/$TPL" "$W/c.sh" "$BOGUS"
  SLURM_JOB_ID=999999 bash "$W/c.sh" > "$W/c.log" 2>&1; RC=$?
  check "wrong pin refuses, naming the pin" "$( [ "$TPL" = phaseN_relock.sh ] && echo 8 || echo 2 )" "$BOGUS" "$RC" "$W/c.log"
  grep -qF "$REAL" "$W/c.log" || { say "  FAIL  the refusal does not name the sha it actually got"; FAIL=1; }

  # B. a CORRECT pin passes the sha gate and the run continues.
  derive "$HERE/$TPL" "$W/b.sh" "$REAL"
  SLURM_JOB_ID=999999 bash "$W/b.sh" > "$W/b.log" 2>&1; RC=$?
  check "correct pin matches and continues" "$RC" "PINNED and matched" "$RC" "$W/b.log"
  grep -q 'FATAL.*sha' "$W/b.log" && { say "  FAIL  a matching pin still refused"; FAIL=1; }

  # A. an EMPTY pin DERIVES the sha -- the whole point: a SNAP swap cannot
  #    leave a stale sha behind, because there is no second constant to go stale.
  derive "$HERE/$TPL" "$W/a.sh" ""
  SLURM_JOB_ID=999999 bash "$W/a.sh" > "$W/a.log" 2>&1; RC=$?
  check "empty pin derives the sha at run time" "$RC" "DERIVED from \$SNAP at run time" "$RC" "$W/a.log"
  grep -q 'FATAL.*sha' "$W/a.log" && { say "  FAIL  the derived sha still produced a sha refusal"; FAIL=1; }
  grep -qF "$REAL" "$W/a.log" || { say "  FAIL  the derived sha was never printed"; FAIL=1; }
  # and it must reach a LATER gate, not stop here
  grep -q 'FATAL' "$W/a.log" || { say "  FAIL  the run did not reach any later gate -- test is not exercising the path"; FAIL=1; }
done

[ "$FAIL" = 0 ] && { say "EXPECT_SHA gate guard: ALL PASS"; exit 0; }
say "EXPECT_SHA gate guard: FAILED"; exit 1
