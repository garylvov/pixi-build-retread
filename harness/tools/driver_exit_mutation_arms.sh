#!/usr/bin/env bash
# driver_exit_mutation_arms.sh -- HARNESS-EXIT-3. The arms that prove FAMILY B
# can still FAIL. A guard that cannot fail is a defect (CLAUDE.md law 3), and
# FAMILY B now has three ways to be silently inert that its own green run
# cannot distinguish from a healthy tree:
#
#   ARM 1  a wrapper reverted to the swallow shape is not caught, because the
#          discovery sweep did not reach it or the seam handed it no failure.
#   ARM 2  a payload the shim does not stub is EXECUTED instead of refused --
#          the STORE-REAP-2 defect, which is the reason this lane exists.
#   ARM 3  the ratchet is a suggestion: a baseline grown past its pin is
#          accepted and a new swallower is baselined away.
#
# Each arm MUTATES one thing and requires the guard's verdict to FLIP. An arm
# whose two sides agree is a FAILURE, not a pass.
#
#   usage: bash driver_exit_mutation_arms.sh [1|2|3|all]
#   rc 0 every arm flipped;  rc 1 an arm did not;  rc 4 fixture fatal.
set -uo pipefail

REPO=${HARNESS_REPO:-/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools}
TASK=${HARNESS_TASK_DIR:-/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11}
GUARD=${DEG_GUARD:-$TASK/tools/driver_exit_guard.sh}
WHICH=${1:-all}

[ -f "$GUARD" ] || { echo "ARMS FATAL: no guard at $GUARD"; exit 4; }

W=$(mktemp -d "${TMPDIR:-/tmp}/driver_exit_arms.XXXXXX") || { echo "ARMS FATAL: no temp dir"; exit 4; }
# THE MUTANT WRAPPER LIVES IN THE TASK TREE, because the thing under test is
# DISCOVERY -- a mutant outside the tree would prove nothing about it. It is
# removed on every exit path, including a kill, so the tree never keeps a
# deliberate swallower that the next lane would have to baseline.
MUTDIR=$TASK/harnessexit3/mutarm
trap 'rm -rf "$W" "$MUTDIR"' EXIT
echo "### driver_exit_mutation_arms  guard=$GUARD  task=$TASK  host=$(hostname)  $(date -Is)"

pass=0; fail=0
ok () { pass=$((pass + 1)); echo "  PASS  $*"; }
no () { fail=$((fail + 1)); echo "  FAIL  $*"; }

run_guard () {  # env assignments come from the caller; echoes rc, log in $W/$1.log
  local tag=$1; shift
  env "$@" bash "$GUARD" B >"$W/$tag.log" 2>&1
  echo $?
}

# ---------------------------------------------------------------- ARM 1 -----
# A wrapper reverted to the swallow shape must be CAUGHT -- found by discovery
# (it is written into the tree, not into a list), handed a failing payload, and
# named as a swallower outside the baseline.
if [ "$WHICH" = all ] || [ "$WHICH" = 1 ]; then
  echo "=== ARM 1 -- a wrapper reverted to the swallow shape"
  mkdir -p "$MUTDIR" || { echo "ARMS FATAL: cannot write $MUTDIR"; exit 4; }
  # the CONTROL half: the fixed shape, which must NOT be named.
  cat > "$MUTDIR/control.sbatch" <<'EOS'
#!/bin/bash
#SBATCH --job-name=he3-mutarm-control
set -u
bash /oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/tools/gate_build.sh
rc=$?
echo "### GATE_EXIT=$rc"
exit "$rc"
EOS
  # the MUTANT half: the pre-fix epilogue, byte for byte the family idiom.
  cat > "$MUTDIR/mutant.sbatch" <<'EOS'
#!/bin/bash
#SBATCH --job-name=he3-mutarm-mutant
set -u
bash /oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/tools/gate_build.sh
echo "### GATE_EXIT=$?"
EOS
  rc=$(run_guard arm1 DEG_EXPLORATORY=1 DEG_ONLY='harnessexit3/mutarm/*.sbatch')
  named_m=$(grep -c 'harnessexit3/mutarm/mutant.sbatch SWALLOWS' "$W/arm1.log")
  named_c=$(grep -c 'harnessexit3/mutarm/control.sbatch SWALLOWS' "$W/arm1.log")
  disc=$(grep -c 'harnessexit3/mutarm/mutant.sbatch' "$W/arm1.log")
  [ "$disc" -gt 0 ] && ok "ARM 1 discovery reached a wrapper that was never in any list" \
                    || no "ARM 1 the mutant was never even discovered (log $W/arm1.log)"
  [ "$named_m" -gt 0 ] && ok "ARM 1 MUTANT named as a swallower outside the baseline" \
                       || no "ARM 1 the reverted swallow shape was NOT caught (log $W/arm1.log)"
  [ "$named_c" -eq 0 ] && ok "ARM 1 CONTROL (same wrapper, rc re-raised) not named -- the arm is not vacuous" \
                       || no "ARM 1 VACUOUS: the fixed shape was named too, so the fixture cannot tell them apart"
  [ "$rc" -ne 0 ] && ok "ARM 1 the guard re-raised on the mutant: rc=$rc" \
                  || no "ARM 1 the guard named a swallower and still exited 0"
  rm -rf "$MUTDIR"
fi

# ---------------------------------------------------------------- ARM 2 -----
# A stub REMOVED from the shim dir must produce the UNCOVERED-PAYLOAD REFUSAL,
# never an execution. This is the STORE-REAP-2 defect as a fixture:
# sr2-work/check.sbatch's payload is `cargo check --all-targets`, and with the
# `cargo` stub gone the only acceptable outcome is a refusal.
if [ "$WHICH" = all ] || [ "$WHICH" = 2 ]; then
  echo "=== ARM 2 -- a stub removed from the shim dir"
  TARGET=sr2-work/check.sbatch
  [ -f "$TASK/$TARGET" ] || { echo "ARMS FATAL: no $TASK/$TARGET to drive"; exit 4; }
  rc_ctl=$(run_guard arm2ctl DEG_EXPLORATORY=1 DEG_ONLY="$TARGET")
  rc_mut=$(run_guard arm2mut DEG_EXPLORATORY=1 DEG_ONLY="$TARGET" DEG_DROP_STUBS=cargo)
  ctl_cargo=$(grep -c 'PAYLOAD.*cargo' "$W/arm2ctl.log")
  mut_ref=$(grep -c "REFUSE  B $TARGET: uncovered command cargo" "$W/arm2mut.log")
  mut_can=$(grep -c 'CANARY FIRED' "$W/arm2mut.log")
  grep -q "intercepted payload commands" "$W/arm2ctl.log" && \
    grep -qE '^### +1 cargo' "$W/arm2ctl.log" \
      && ok "ARM 2 CONTROL: the seam intercepted the real `cargo check` and stubbed it" \
      || no "ARM 2 CONTROL: no intercepted cargo in the recorder -- the arm below proves nothing (log $W/arm2ctl.log)"
  [ "$mut_ref" -gt 0 ] && ok "ARM 2 MUTANT: with the cargo stub gone the wrapper is REFUSED as uncovered, not run" \
                       || no "ARM 2 MUTANT: no uncovered refusal -- an unstubbed payload was not caught (log $W/arm2mut.log)"
  [ "$mut_can" -eq 0 ] && ok "ARM 2 MUTANT: no canary fired either -- nothing reached a real cargo" \
                       || no "ARM 2 MUTANT: THE CANARY FIRED -- a real payload was reachable"
fi

# ---------------------------------------------------------------- ARM 3 -----
# The ratchet must be a mechanism, not a manner. A baseline grown past its pin
# is a REFUSAL at rc 4, so a lane cannot baseline a new swallower away.
if [ "$WHICH" = all ] || [ "$WHICH" = 3 ]; then
  echo "=== ARM 3 -- the baseline may only shrink"
  cp -f "$TASK/tools/driver_exit_baseline.txt" "$W/grown.txt"
  echo "SWALLOW some/lane/that-we-would-rather-not-fix.sbatch   # exactly what the pin exists to stop" >> "$W/grown.txt"
  rc_ctl=$(run_guard arm3ctl DEG_EXPLORATORY=1 DEG_ONLY='harnessexit3/guard.sbatch')
  rc_mut=$(run_guard arm3mut DEG_EXPLORATORY=1 DEG_ONLY='harnessexit3/guard.sbatch' DEG_BASELINE="$W/grown.txt")
  [ "$rc_mut" -eq 4 ] && ok "ARM 3 MUTANT: a baseline one row past the pin REFUSES at rc 4" \
                      || no "ARM 3 MUTANT: a grown baseline was accepted (rc=$rc_mut, log $W/arm3mut.log)"
  [ "$rc_ctl" -ne 4 ] && ok "ARM 3 CONTROL: the same run on the real baseline does not refuse (rc=$rc_ctl)" \
                      || no "ARM 3 VACUOUS: the control refused too, so rc 4 says nothing about the pin"
fi

echo "### driver_exit_mutation_arms: pass=$pass fail=$fail  $(date -Is)"
[ "$fail" -eq 0 ] && echo "### MUTATION ARMS GREEN -- every arm flipped" \
                  || echo "### MUTATION ARMS RED -- an arm did not flip"
[ "$fail" -eq 0 ] || exit 1
exit 0
