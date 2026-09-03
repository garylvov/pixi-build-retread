#!/usr/bin/env bash
# Guard for the leftover-token self-check, both directions.
#
# The check used to pipe "FILENAME:LNO: line" into grep, so it matched its OWN
# FILENAME: a harness derived into a directory named after a previous batch
# (p6b-c3b, here) failed against itself on every line, with the token nowhere in
# its body. That is a scan that can match itself, which is never allowed to be
# the thing deciding an exit code. The match now runs inside awk, on the LINE.
#
#   A. a harness whose PATH contains a leftover token, but whose BODY does not,
#      must pass.
#   B. a leftover token in the BODY, outside the three marked regions, must
#      still fail with exit 9 and name the line -- the check must not have been
#      weakened into uselessness.
#   C. a token inside a marked region (EVIDENCE / SUBSTITUTE / LEFTOVER-CHECK)
#      must still be exempt.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
FAIL=0
say() { printf '%s\n' "$*"; }

TOKEN=bfinal            # already in every template's LEFTOVER_RE
mkdir -p "$W/$TOKEN-batch"
run() { SLURM_JOB_ID=999999 bash "$1" 2>&1; }

for TPL in phaseN_relock.sh phaseN_cert.sh; do
  say "== $TPL =="
  # A. the token is in the PATH only.
  P="$W/$TOKEN-batch/$TPL"; cp "$HERE/$TPL" "$P"
  OUT=$(run "$P"); RC=$?
  if printf '%s' "$OUT" | grep -q 'leftover-token self-check: clean'; then
    say "  PASS  a path containing '$TOKEN' does not trip the check"
  else
    say "  FAIL  the check matched its own FILENAME (rc=$RC)"; printf '%s\n' "$OUT" | head -4; FAIL=1
  fi

  # B. the token in the BODY, outside every marked region, must still fail 9.
  P2="$W/$TOKEN-batch/body-$TPL"
  awk -v t="# $TOKEN leftover planted by the guard" \
      '{print} /^### LEFTOVER-CHECK END/{print t}' "$HERE/$TPL" > "$P2"
  OUT=$(run "$P2"); RC=$?
  if [ "$RC" = 9 ] && printf '%s' "$OUT" | grep -q 'leftover planted by the guard'; then
    say "  PASS  a token in the body still fails 9 and names the line"
  else
    say "  FAIL  a planted leftover was NOT caught (rc=$RC)"; FAIL=1
  fi

  # C. the same token inside the EVIDENCE region stays exempt.
  P3="$W/$TOKEN-batch/evid-$TPL"
  awk -v t="# $TOKEN cited deliberately in EVIDENCE" \
      '/^### EVIDENCE BEGIN/{print; print t; next} {print}' "$HERE/$TPL" > "$P3"
  OUT=$(run "$P3")
  if printf '%s' "$OUT" | grep -q 'leftover-token self-check: clean'; then
    say "  PASS  a deliberate citation inside EVIDENCE stays exempt"
  else
    say "  FAIL  the EVIDENCE region is no longer exempt"; FAIL=1
  fi
done

[ "$FAIL" = 0 ] && { say "leftover-check guard: ALL PASS"; exit 0; }
say "leftover-check guard: FAILED"; exit 1
