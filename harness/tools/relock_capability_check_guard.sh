#!/usr/bin/env bash
# relock_capability_check_guard.sh -- the reader for tools/relock_capability_check.sh.
#
# WHAT IT GUARDS, AND WHY THE TOOL EXISTS. det162_proof.sh's `--cold-proof-arm`
# gate greps its relock template for the row `sdist scoping SKIPPED (declared
# cold proof arm)`. At this tip that count is ZERO -- HARNESS-CONSOL-9 moved the
# branch into ONE producer, `retread_relock_scope_and_verify` -- so a verbatim
# copy of that gate REFUSES A CORRECT TREE. `relock_capability_check.sh` answers
# the same question by the pair that actually has to hold: the producer is
# EXECUTED, and the template is read for a call that FORWARDS AN ARGV to it.
#
# ARMS
#   A  it PASSES on every shipped template, for both capabilities. A capability
#      check that is red on the tree it ships with is not a check.
#   B  a fixture template with the call REMOVED -> rc 1, and the row says the
#      call is missing rather than something vaguer.
#   C  a fixture template that CALLS the function but drops the argv -> rc 1.
#      This is the shape a copy-paste actually produces and B would not catch it.
#   D  the stale-gate REGRESSION, stated as a measurement: det162's grep returns
#      0 on the very templates arm A passes. The tool and the grep DISAGREE on a
#      correct tree, and that disagreement is the whole reason for the tool -- if
#      the grep ever agrees again this arm says so and the tool can be retired.
#   E  MUTATION on the PRODUCER: the flag parse cut from a copy of
#      retread_fast_env.sh -> arm A's own templates must go rc 1. Without this,
#      A is green against a producer that ignores the flag entirely.
#   F  MUTATION on the OTHER half: a fixture template that reaches a working
#      producer but hardcodes `pixi lock -v` -> rc 1 for --frontend-rust-log,
#      because pixi's own -v overrides RUST_LOG and the announced filter is then
#      a lie.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
TOOL=$HERE/relock_capability_check.sh
FAST=$HERE/retread_fast_env.sh
[ -f "$TOOL" ] || { echo "FATAL no relock_capability_check.sh at $TOOL"; exit 3; }
[ -f "$FAST" ] || { echo "FATAL no retread_fast_env.sh at $FAST"; exit 3; }

W=$(mktemp -d "${TMPDIR:-/tmp}/relockcapguard.XXXXXX") || exit 3
trap 'rm -rf "$W"' EXIT
fail=0
say () { echo "### RELOCK-CAP GUARD $*"; }
ok  () { say "PASS $*"; }
bad () { say "FAIL $*"; fail=1; }

TARGETS=
for t in "$HERE/../phase_template/phaseN_relock.sh" "$HERE/../arms/mh1_relock.sh" \
         "$HERE/../arms/c29_relock.sh" "$HERE/../proof/hlgd_relock.sh" \
         "$HERE/../instrumented/p6b_relock.sh" "$HERE/../instrumented/p6b_relock.b2.sh"; do
  [ -f "$t" ] && TARGETS="$TARGETS $t"
done
N=$(printf '%s\n' $TARGETS | grep -c .)
[ "$N" -ge 1 ] || { say "REFUSED no relock template to read -- this guard would be green against nothing"; exit 3; }
say "targets=$N"

# ---- ARM A ------------------------------------------------------------------
for CAP in --cold-proof-arm --frontend-rust-log; do
  for t in $TARGETS; do
    out=$(bash "$TOOL" "$CAP" "$t" 2>&1); rc=$?
    if [ "$rc" = 0 ]; then ok "A $(basename "$t") $CAP -> rc 0: $out"
    else bad "A $(basename "$t") $CAP -> rc $rc: $out"; fi
  done
done

# ---- fixture templates ------------------------------------------------------
BASETPL=$HERE/../phase_template/phaseN_relock.sh
mkfix () {                     # $1 = name, $2.. = sed expressions
  local n=$1; shift
  local f=$W/$n.sh
  cp -f "$BASETPL" "$f"
  local e
  for e in "$@"; do sed -i "$e" "$f"; done
  echo "$f"
}
BNOCALL=$(mkfix nocall '/^[[:space:]]*retread_relock_scope_and_verify /d')
CNOARGV=$(mkfix noargv 's|^\([[:space:]]*\)retread_relock_scope_and_verify "\$C" "\$@"|\1retread_relock_scope_and_verify "$C"|')
FNOCALL=$(mkfix fnocall '/^[[:space:]]*retread_relock_frontend_log /d')
FHARDV=$(mkfix fhardv 's|"\$PIXI" lock \$LOCK_VERBOSITY|"$PIXI" lock -v|')

armfix () {                    # $1=label $2=cap $3=fixture $4=substring wanted
  local lb=$1 cap=$2 f=$3 want=$4 out rc
  out=$(bash "$TOOL" "$cap" "$f" 2>&1); rc=$?
  if [ "$rc" = 1 ] && printf '%s' "$out" | grep -qF -- "$want"; then
    ok "$lb rc 1 and the row names it: $out"
  else
    bad "$lb rc=$rc (wanted 1) / row does not carry '$want': $out"
  fi
}
cmp -s "$BASETPL" "$BNOCALL" && bad "B the fixture is unmutated -- arm B cannot fail" \
  || armfix "B" --cold-proof-arm "$BNOCALL" "never calls retread_relock_scope_and_verify"
cmp -s "$BASETPL" "$CNOARGV" && bad "C the fixture is unmutated -- arm C cannot fail" \
  || armfix "C" --cold-proof-arm "$CNOARGV" "WITHOUT forwarding an argv"
cmp -s "$BASETPL" "$FNOCALL" && bad "F0 the fixture is unmutated" \
  || armfix "F0" --frontend-rust-log "$FNOCALL" "never calls retread_relock_frontend_log"
cmp -s "$BASETPL" "$FHARDV" && bad "F the fixture is unmutated -- arm F cannot fail" \
  || armfix "F" --frontend-rust-log "$FHARDV" "hardcodes 'pixi lock -v'"

# ---- ARM D: the stale gate, stated as a measurement --------------------------
STALE=0
for t in $TARGETS; do
  c=$(grep -c -F 'sdist scoping SKIPPED' "$t")
  [ "$c" = 0 ] && STALE=$((STALE + 1))
done
if [ "$STALE" = "$N" ]; then
  ok "D det162's gate (grep -c -F 'sdist scoping SKIPPED' <template>) returns 0 on ALL $N templates that arm A just passed -- a file-coupled gate refusing a correct tree, which is exactly what this tool replaces"
else
  bad "D the grep found the row in $((N - STALE)) of $N templates. Either the branch has been copied BACK into a template -- two producers -- or this guard's premise has expired and the driver re-cut is no longer needed. Decide it, do not leave it."
fi

# ---- ARM E: MUTATION on the producer ----------------------------------------
EMUT=$W/fast_env_noflag.sh
sed 's|^\([[:space:]]*\)for a in "\$@"; do \[ "\$a" = --cold-proof-arm \] && cold=1; done|\1: # ARM E MUTATION: flag parse deleted|' "$FAST" > "$EMUT"
if cmp -s "$FAST" "$EMUT"; then
  bad "E the producer mutation edited nothing -- arm A is asserting against an unmutated producer and cannot fail"
else
  eout=$(RELOCK_FAST_ENV=$EMUT bash "$TOOL" --cold-proof-arm "$BASETPL" 2>&1); erc=$?
  if [ "$erc" = 1 ] && printf '%s' "$eout" | grep -qF -- 'the flag does NOT skip the scoping'; then
    ok "E (mutation) with the flag parse cut from the producer the SAME shipped template answers rc 1: $eout"
  else
    bad "E rc=$erc -- a producer that ignores the flag still answered YES, so arm A proves nothing: $eout"
  fi
fi

say "rc=$fail"
[ "$fail" = 0 ] && { say "ALL ARMS PASS"; exit 0; }
say "FAILED"; exit 1
