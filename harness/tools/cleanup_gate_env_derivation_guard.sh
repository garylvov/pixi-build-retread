#!/usr/bin/env bash
# GUARD for cleanup_gated.sh ROOT FIX (d), 2026-09-05, p6ad-4.
#
# THE DEFECT IT GUARDS. `D`, `TAG` and `RJ` reached the gate only through
# `--export`, so an sbatch `--wrap` without that clause died on bash's own
# `${D:?...}` before the gate printed a row, before the roots were parsed, and
# with a message naming the variable rather than the fix. Two independent lanes
# hit it on one night -- B-cert-4's cleanup 5841112 (C31-2) and p6ad-4's 5879244,
# which stranded certP6AD4-5879243-{A,B} and ws.P6AD4-5879243-{A,B}. The three
# values are already carried by the root arguments, so they are now derived.
#
# NOTHING IS EVER DELETED BY THIS GUARD. Every arm points the gate at roots that
# DO NOT EXIST, so the run stops at the evidence or containment conditions long
# before `cleanup.sh` is reached. The guard asserts on what the gate PRINTS and
# on the values it derives, never on a deletion.
#
# ARMS
#   A  no D/TAG/RJ at all, over a fixture harness whose artifacts/ holds the
#      evidence -> the gate DERIVES all three and prints them. This is the fix.
#   B  the OLD file, byte-for-byte from git, same arguments -> dies on the bash
#      parameter expansion with no `### DERIVED` row anywhere. THE MUTATION:
#      without it arm A proves nothing, because a gate that always printed
#      `### DERIVED` would pass arm A for free.
#   C  explicit D/TAG/RJ WIN over the derivation: a TAG the roots do not name is
#      honoured, so the fix is additive and cannot override an operator.
#   D  underivable -- a root basename with no job-id token -> exit 2 and the
#      printed `--export=ALL,D=...` line, never a bash error.
#   E  derivable tag+job but NO harness directory holds the evidence -> exit 2
#      with the export line, and D reported unset.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
GATE=$REPO/harness/phase_template/cleanup_gated.sh
T=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11
WORK=${TMPDIR:-/tmp}/cleanup-gate-env-guard-$$
mkdir -p "$WORK"
pass=0; fail=0
ok()  { echo "PASS  $*"; pass=$((pass+1)); }
bad() { echo "FAIL  $*"; fail=$((fail+1)); }

[ -f "$GATE" ] || { echo "FATAL: no gate at $GATE"; exit 3; }
echo "### guard for $GATE"
echo "### work $WORK"

# A fixture harness directory under the TASK ROOT, because `derive_harness_dir`
# searches there by construction. Its tag is unique to this run so it can never
# collide with a live lane's artifacts.
TAG=GUARDENVDERIV$$
RJ=999$$
HDIR=$T/guard-envderiv-$$   # ONE level under the task root: that is where every harness
                            # directory of this campaign lives, and it is what
                            # `derive_harness_dir`s `-maxdepth 3` is sized for.
                            # A fixture nested deeper would be testing a layout
                            # no harness has.
mkdir -p "$HDIR/artifacts"
printf '0\n'   > "$HDIR/artifacts/$TAG-$RJ.rc"
printf '123\n' > "$HDIR/artifacts/$TAG-$RJ.wall"
printf 'x\n'   > "$HDIR/artifacts/$TAG-$RJ.lock.log"
printf 'x\n'   > "$HDIR/artifacts/$TAG-$RJ.pixi.lock.cert"
ROOT=/oscar/data/stellex/glvov/retread/cert$TAG-$RJ-A
[ -e "$ROOT" ] && { echo "FATAL: fixture root $ROOT exists on disk -- refusing to run"; exit 3; }
cleanup_fixture() { rm -rf "$HDIR" "$WORK"; }
trap cleanup_fixture EXIT

run_gate() {   # run_gate <logfile> <script> [VAR=VAL ...] -- <roots...>
  local log=$1 script=$2; shift 2
  local -a envs=()
  while [ "${1:-}" != "--" ]; do envs+=("$1"); shift; done
  shift
  ( env -u D -u TAG -u RJ "${envs[@]}" bash "$script" "$@" ) > "$log" 2>&1
  echo $?
}

# ---- ARM A: derivation, with nothing exported -------------------------------
rcA=$(run_gate "$WORK/A.log" "$GATE" -- "$ROOT")
grep -q "### DERIVED TAG=$TAG " "$WORK/A.log" \
  && ok "A: TAG derived from the root basename" || bad "A: TAG not derived"
grep -q "### DERIVED RJ=$RJ " "$WORK/A.log" \
  && ok "A: RJ derived from the root basename" || bad "A: RJ not derived"
grep -q "### DERIVED D=$HDIR " "$WORK/A.log" \
  && ok "A: D derived from the artifacts that hold the evidence" || bad "A: D not derived"
grep -q "### CLEANUP GATE tag=$TAG relock_job=$RJ" "$WORK/A.log" \
  && ok "A: the gate actually ran its conditions (rc=$rcA)" || bad "A: the gate never reached its conditions"
grep -qi "set D to the harness directory" "$WORK/A.log" \
  && bad "A: still died on the bash parameter expansion" || ok "A: no bash parameter-expansion death"

# ---- ARM B: THE MUTATION -- the previous version of the same file ------------
OLD=$WORK/cleanup_gated.OLD.sh
if git -C "$REPO" show "HEAD:harness/phase_template/cleanup_gated.sh" > "$OLD" 2>/dev/null && [ -s "$OLD" ]; then
  rcB=$(run_gate "$WORK/B.log" "$OLD" -- "$ROOT")
  if grep -q "### DERIVED" "$WORK/B.log"; then
    bad "B: the OLD file already derived -- this guard cannot fail and is worthless"
  else
    ok "B: the OLD file prints no ### DERIVED row (the guard can fail)"
  fi
  if grep -qi "set D to the harness directory" "$WORK/B.log"; then
    ok "B: the OLD file dies on the bash parameter expansion, rc=$rcB -- the defect, reproduced"
  else
    bad "B: the OLD file did not reproduce the defect (rc=$rcB) -- read $WORK/B.log"
  fi
else
  bad "B: could not extract HEAD:harness/phase_template/cleanup_gated.sh -- MUTATION ARM DID NOT RUN"
fi

# ---- ARM C: an explicit export WINS over the derivation ----------------------
rcC=$(run_gate "$WORK/C.log" "$GATE" "TAG=EXPLICITTAG" "RJ=$RJ" "D=$HDIR" -- "$ROOT")
grep -q "### CLEANUP GATE tag=EXPLICITTAG relock_job=$RJ" "$WORK/C.log" \
  && ok "C: an exported TAG overrides the derivation (rc=$rcC)" || bad "C: the derivation overrode an explicit export"
grep -q "### DERIVED" "$WORK/C.log" \
  && bad "C: derived something that was explicitly given" || ok "C: nothing derived when everything was given"

# ---- ARM D: underivable -- no job-id token in the basename -------------------
rcD=$(run_gate "$WORK/D.log" "$GATE" -- /oscar/data/stellex/glvov/retread/certNOJOBIDHERE)
[ "$rcD" = 2 ] && ok "D: an underivable root refuses with exit 2" || bad "D: exit $rcD, want 2"
grep -q -- "--export=ALL,D=" "$WORK/D.log" \
  && ok "D: the refusal prints the exact --export clause to add" || bad "D: the refusal does not print the export line"
grep -qi "set D to the harness directory" "$WORK/D.log" \
  && bad "D: refused with a bash parameter-expansion error" || ok "D: refused with a message, not a bash error"

# ---- ARM E: derivable tag+job, but no harness holds the evidence -------------
rcE=$(run_gate "$WORK/E.log" "$GATE" -- "/oscar/data/stellex/glvov/retread/certNOSUCHHARNESS$$-888$$-A")
[ "$rcE" = 2 ] && ok "E: no harness for the evidence refuses with exit 2" || bad "E: exit $rcE, want 2"
grep -q "D='<unset>'" "$WORK/E.log" \
  && ok "E: the refusal names D as the value it could not derive" || bad "E: the refusal does not name D"

# ---- nothing was deleted ----------------------------------------------------
[ -s "$HDIR/artifacts/$TAG-$RJ.rc" ] \
  && ok "no fixture evidence was removed by any arm" || bad "the guard deleted its own fixture"

echo "### cleanup_gate_env_derivation_guard: pass=$pass fail=$fail"
[ "$fail" -eq 0 ] || exit 1
exit 0
