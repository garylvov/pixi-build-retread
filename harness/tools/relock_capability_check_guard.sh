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
#   A  it PASSES on every shipped template, for ALL THREE capabilities. A capability
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
for CAP in --cold-proof-arm --frontend-rust-log --env-seed; do
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

# ---- ARM S: the --env-seed capability, on fixtures that must answer NO -------
# HARNESS-CONSOL-12 (2026-09-07). Arm A above proves the check says YES on all
# six shipped templates; without these it would also say YES on a template that
# had lost the export, and a capability check that cannot answer NO is not a
# check. Each fixture removes exactly ONE of the three things the capability is:
# sourcing the one authority, calling it STRICT over a backend, refusing when it
# fails. Each is vacuity-checked against the base with `cmp -s`, the same way
# arms B/C/F0/F are, because a sed that stopped matching would otherwise read as
# a green.
SNOCALL=$(mkfix seed_nocall '/^env_seed_export "\$BACKEND" || exit 15$/d')
SOPT=$(mkfix seed_optional 's@^env_seed_export "\$BACKEND" || exit 15$@env_seed_export "$BACKEND" optional || exit 15@')
SNOREF=$(mkfix seed_norefuse 's@^env_seed_export "\$BACKEND" || exit 15$@env_seed_export "$BACKEND"@')
SNOSRC=$(mkfix seed_nosource '/^\. "\$ENV_SEED_LIB"$/d')
cmp -s "$BASETPL" "$SNOCALL" && bad "S1 the fixture is unmutated -- arm S1 cannot fail" \
  || armfix "S1" --env-seed "$SNOCALL" "never calls env_seed_export"
cmp -s "$BASETPL" "$SOPT" && bad "S2 the fixture is unmutated -- arm S2 cannot fail" \
  || armfix "S2" --env-seed "$SOPT" "OPTIONAL mode"
cmp -s "$BASETPL" "$SNOREF" && bad "S3 the fixture is unmutated -- arm S3 cannot fail" \
  || armfix "S3" --env-seed "$SNOREF" "does NOT refuse when it fails"
cmp -s "$BASETPL" "$SNOSRC" && bad "S4 the fixture is unmutated -- arm S4 cannot fail" \
  || armfix "S4" --env-seed "$SNOSRC" "never sources \$ENV_SEED_LIB"

# ---- ARM S5: THE MUTATION THAT NAMES THE BACK-PORT --------------------------
# mh1_relock.sh is a template this capability was ADDED to on 2026-09-07, and at
# 8f1dd88 `git cat-file blob 8f1dd88:harness/arms/mh1_relock.sh | grep -c seed`
# returned ZERO. Measured across the six shipped relock templates at that tip:
# retread_relock_frontend_log 6/6, retread_relock_scope_and_verify 6/6,
# env_seed_export 1/6 -- the two capabilities this tool already READ had been
# back-ported and the one it did not read had not. Cut the block back out of mh1
# and the check must say NO, or arm A's new green over mh1 proves nothing about
# mh1 in particular.
MH1=$HERE/../arms/mh1_relock.sh
if [ ! -f "$MH1" ]; then
  bad "S5 no mh1_relock.sh -- the back-port mutation did not run"
else
  MH1MUT=$W/mh1_seedcut.sh
  sed '/^env_seed_export "\$BACKEND" || exit 15$/d' "$MH1" > "$MH1MUT"
  if cmp -s "$MH1" "$MH1MUT"; then
    bad "S5 the mutation removed NOTHING from mh1_relock.sh -- either the back-port is absent or its call line changed shape, and arm A's mh1 --env-seed green is vacuous either way"
  else
    sout=$(bash "$TOOL" --env-seed "$MH1MUT" 2>&1); src=$?
    if [ "$src" = 1 ] && printf '%s' "$sout" | grep -qF -- 'never calls env_seed_export'; then
      ok "S5 (mutation) with the back-ported block cut back out of mh1 the check answers rc 1: $sout"
    else
      bad "S5 rc=$src (wanted 1) on an mh1 with the export cut: $sout"
    fi
  fi
fi

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
