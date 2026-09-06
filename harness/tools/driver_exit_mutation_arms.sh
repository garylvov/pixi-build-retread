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
#   ARM 4  HARNESS-SEAM-1: the by-path check decides coverage from a VARIABLE'S
#          NAME, so two wrappers naming the same binary get opposite verdicts.
#   ARM 5  a payload variable no list has ever held is refused instead of
#          scored, or a value nothing can resolve is passed instead of refused.
#   ARM 6  a payload invoked by its LITERAL path is executed instead of stubbed.
#
#   usage: bash driver_exit_mutation_arms.sh [1|2|3|4|5|6|all]
#   rc 0 every arm flipped;  rc 1 an arm did not;  rc 4 fixture fatal.
set -uo pipefail

REPO=${HARNESS_REPO:-/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools}
TASK=${HARNESS_TASK_DIR:-/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11}
GUARD=${DEG_GUARD:-$TASK/tools/driver_exit_guard.sh}
WHICH=${1:-all}

[ -f "$GUARD" ] || { echo "ARMS FATAL: no guard at $GUARD"; exit 4; }

# THE ARM LOGS ARE EVIDENCE, so DEG_ARMS_DIR keeps them. And NOTHING in this
# file uses a BACKTICK: the first version of ARM 2 wrote cargo check inside a
# double-quoted PASS message, which is command substitution -- the arms script
# ran a REAL cargo on the compute node while asserting that the guard does not.
# The lesson generalises: a file about not executing payloads must not contain
# a construct that executes one.
W=${DEG_ARMS_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/driver_exit_arms.XXXXXX")}
mkdir -p "$W" || { echo "ARMS FATAL: no work dir at $W"; exit 4; }
# THE MUTANT WRAPPER LIVES IN THE TASK TREE, because the thing under test is
# DISCOVERY -- a mutant outside the tree would prove nothing about it. It is
# removed on every exit path, including a kill, so the tree never keeps a
# deliberate swallower that the next lane would have to baseline.
MUTDIR=$TASK/harnessexit3/mutarm
MUTDIR2=$TASK/harnessseam1/mutarm
trap 'if [ -z "${DEG_ARMS_DIR:-}" ]; then rm -rf "$W"; fi; rm -rf "$MUTDIR" "$MUTDIR2"' EXIT
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
# sr2-work/check.sbatch's payload is cargo check --all-targets, and with the
# cargo stub gone the only acceptable outcome is a refusal.
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
      && ok "ARM 2 CONTROL: the seam intercepted the real cargo check and stubbed it" \
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

# ---------------------------------------------------------------- ARM 4 -----
# HARNESS-SEAM-1. THE DEFECT ITSELF, against the PINNED PRE-FIX BLOB.
# `driver_exit_scan.sh` decided by-path coverage from a hand-typed list of
# VARIABLE NAMES, so two wrappers whose payload variables resolve to THE SAME
# BINARY got opposite verdicts purely because of what the variable was called.
# STORE-REAP-3 hit it and renamed `TIP` to `RETREAD_BIN` to get past it.
# The arm runs the OLD file, extracted from the pinned commit constant, and
# requires it to disagree with itself; then it runs the NEW file and requires
# the same two tokens to agree, on identical resolved values.
SEAM_PREFIX_COMMIT=a9b6283a2c21c28c285ae6508f355fe8d4dd0656
if [ "$WHICH" = all ] || [ "$WHICH" = 4 ]; then
  echo "=== ARM 4 -- the pre-fix scanner decided by the variable's NAME"
  REPO_OK=1
  git -C "$REPO" rev-parse --verify "$SEAM_PREFIX_COMMIT^{commit}" >/dev/null 2>&1 \
    || { no "ARM 4 $SEAM_PREFIX_COMMIT is not a commit in $REPO"; REPO_OK=0; }
  if [ "$REPO_OK" = 1 ]; then
    mkdir -p "$W/seamarm"
    git -C "$REPO" cat-file blob "$SEAM_PREFIX_COMMIT:harness/tools/driver_exit_scan.sh" \
      > "$W/seamarm/old_scan.sh" 2>/dev/null
    BINP=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/binsnaps/integration-cbed649/pixi-build-retread
    for vn in TIP RETREAD_BIN; do
      {
        echo '#!/bin/bash'
        echo 'set -u'
        echo "$vn=$BINP"
        echo "\"\$$vn\" store-reap --store shadow --dry-run"
        echo 'rc=$?'
        echo 'exit "$rc"'
      } > "$W/seamarm/$vn.sbatch"
    done
    # THE OLD FILE. Its verdict function is deg_path_token_covered <token>.
    old_tip=$(bash -c '. "$1" >/dev/null 2>&1; deg_path_token_covered "$2"; echo $?' _ "$W/seamarm/old_scan.sh" '"$TIP"')
    old_rb=$(bash -c '. "$1" >/dev/null 2>&1; deg_path_token_covered "$2"; echo $?' _ "$W/seamarm/old_scan.sh" '"$RETREAD_BIN"')
    [ "$old_tip" != 0 ] && ok "ARM 4 PRE-FIX: '\$TIP' was REFUSED as uncovered (rc=$old_tip) -- the defect, reproduced from $SEAM_PREFIX_COMMIT" \
                        || no "ARM 4 PRE-FIX: '\$TIP' was accepted by the old scanner -- the arm is vacuous, the defect is not in this blob"
    [ "$old_rb" = 0 ] && ok "ARM 4 PRE-FIX: '\$RETREAD_BIN' was ACCEPTED on the same resolved binary -- name, not capability" \
                      || no "ARM 4 PRE-FIX: '\$RETREAD_BIN' was refused too, so the old verdicts do not differ and the arm proves nothing"
    # THE NEW FILE. Same two tokens, same resolved value, one verdict.
    new_tip=$(bash "$REPO/harness/tools/driver_exit_scan.sh" verdicts "$W/seamarm/TIP.sbatch")
    new_rb=$(bash "$REPO/harness/tools/driver_exit_scan.sh" verdicts "$W/seamarm/RETREAD_BIN.sbatch")
    tipv=$(printf '%s' "$new_tip" | cut -f1); tipp=$(printf '%s' "$new_tip" | cut -f3)
    rbv=$(printf '%s' "$new_rb" | cut -f1);  rbp=$(printf '%s' "$new_rb" | cut -f3)
    [ "$tipv" = COVERED ] && [ "$rbv" = COVERED ] \
      && ok "ARM 4 FIXED: both tokens are COVERED ($tipv / $rbv) -- the variable's name is not consulted" \
      || no "ARM 4 FIXED: verdicts still differ or are not COVERED (TIP=$tipv RETREAD_BIN=$rbv)"
    [ -n "$tipp" ] && [ "$tipp" = "$rbp" ] \
      && ok "ARM 4 the two verdicts are on IDENTICAL RESOLVED VALUES: $tipp" \
      || no "ARM 4 the resolved values differ ('$tipp' vs '$rbp') -- the comparison is not like for like"
  fi
fi

# ---------------------------------------------------------------- ARM 5 -----
# HARNESS-SEAM-1. COVERAGE IS DERIVED, so a payload variable NOBODY HAS EVER
# TYPED must be scored, and a variable whose value cannot be known statically
# must be REFUSED BY NAME rather than run. Both halves are driven through the
# live guard over a wrapper written into the tree.
if [ "$WHICH" = all ] || [ "$WHICH" = 5 ]; then
  echo "=== ARM 5 -- a payload variable nobody has thought of, and one that cannot be resolved"
  mkdir -p "$MUTDIR2" || { echo "ARMS FATAL: cannot write $MUTDIR2"; exit 4; }
  BINP=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/binsnaps/integration-cbed649/pixi-build-retread
  {
    echo '#!/bin/bash'
    echo '#SBATCH --job-name=hs1-mutarm-derived'
    echo 'set -u'
    echo "ZOGGLE_PAYLOAD_2026=$BINP"
    echo 'echo "### about to run the payload"'
    echo '"$ZOGGLE_PAYLOAD_2026" store-reap --store shadow --dry-run'
    echo 'rc=$?'
    echo 'echo "### VERB_RC=$rc"'
    echo 'exit "$rc"'
  } > "$MUTDIR2/derived.sbatch"
  {
    echo '#!/bin/bash'
    echo '#SBATCH --job-name=hs1-mutarm-unresolvable'
    echo 'set -u'
    echo 'ZOGGLE_PAYLOAD_2026=$(cat /dev/null)/pixi-build-retread'
    echo '"$ZOGGLE_PAYLOAD_2026" store-reap --store shadow --dry-run'
    echo 'rc=$?'
    echo 'exit "$rc"'
  } > "$MUTDIR2/unresolvable.sbatch"
  rc5=$(run_guard arm5 DEG_EXPLORATORY=1 DEG_ONLY='harnessseam1/mutarm/*.sbatch')
  d_bypath=$(grep -c 'BYPATH  B harnessseam1/mutarm/derived.sbatch' "$W/arm5.log")
  d_ref=$(grep -c 'REFUSE  B harnessseam1/mutarm/derived.sbatch' "$W/arm5.log")
  d_pay=$(grep -c 'PAYLOAD.*pixi-build-retread' "$W/arm5.log")
  u_ref=$(grep -c 'REFUSE  B harnessseam1/mutarm/unresolvable.sbatch: payload invoked by path through a variable that cannot be resolved' "$W/arm5.log")
  u_named=$(grep -c 'ZOGGLE_PAYLOAD_2026' "$W/arm5.log")
  can5=$(grep -c 'CANARY FIRED' "$W/arm5.log")
  [ "$d_bypath" -gt 0 ] && [ "$d_ref" -eq 0 ] \
    && ok "ARM 5 DERIVED: a variable named ZOGGLE_PAYLOAD_2026 -- in no list anywhere -- is COVERED and driven" \
    || no "ARM 5 DERIVED: the wrapper was not covered (bypath=$d_bypath refuse=$d_ref, log $W/arm5.log)"
  [ "$d_pay" -gt 0 ] && ok "ARM 5 DERIVED: the by-path invocation was INTERCEPTED (a pixi-build-retread PAYLOAD row exists)" \
                     || no "ARM 5 DERIVED: no PAYLOAD row -- the wrapper was covered on paper and reached nothing (log $W/arm5.log)"
  [ "$u_ref" -gt 0 ] && ok "ARM 5 UNRESOLVABLE: refused, not executed" \
                     || no "ARM 5 UNRESOLVABLE: no refusal -- an unknowable payload was passed or run (log $W/arm5.log)"
  [ "$u_named" -gt 0 ] && ok "ARM 5 UNRESOLVABLE: the refusal NAMES the variable" \
                       || no "ARM 5 UNRESOLVABLE: the refusal did not name ZOGGLE_PAYLOAD_2026, so nobody can act on it"
  [ "$can5" -eq 0 ] && ok "ARM 5 no canary fired -- nothing reached a real binary" \
                    || no "ARM 5 THE CANARY FIRED"
  rm -rf "$MUTDIR2"
fi

# ---------------------------------------------------------------- ARM 6 -----
# HARNESS-SEAM-1. A LITERAL by-path payload -- the real cargo, at the real
# path sr2-work/check.sbatch puts on its PATH -- must be STUBBED, and removing
# the cargo stub must turn that same wrapper into a REFUSAL. This is ARM 2's
# assertion for the half PATH and functions cannot reach.
if [ "$WHICH" = all ] || [ "$WHICH" = 6 ]; then
  echo "=== ARM 6 -- a payload invoked by its literal path"
  mkdir -p "$MUTDIR2" || { echo "ARMS FATAL: cannot write $MUTDIR2"; exit 4; }
  {
    echo '#!/bin/bash'
    echo '#SBATCH --job-name=hs1-mutarm-bypath-cargo'
    echo 'set -u'
    echo '/users/glvov/.cargo/bin/cargo check --all-targets -j 1'
    echo 'rc=$?'
    echo 'exit "$rc"'
  } > "$MUTDIR2/bypath_cargo.sbatch"
  rc6c=$(run_guard arm6ctl DEG_EXPLORATORY=1 DEG_ONLY='harnessseam1/mutarm/bypath_cargo.sbatch')
  rc6m=$(run_guard arm6mut DEG_EXPLORATORY=1 DEG_ONLY='harnessseam1/mutarm/bypath_cargo.sbatch' DEG_DROP_STUBS=cargo)
  c_pay=$(grep -cE 'PAYLOAD[[:space:]]+cargo' "$W/arm6ctl.log")
  c_can=$(grep -c 'CANARY FIRED' "$W/arm6ctl.log")
  m_ref=$(grep -c 'REFUSE  B harnessseam1/mutarm/bypath_cargo.sbatch' "$W/arm6mut.log")
  m_can=$(grep -c 'CANARY FIRED' "$W/arm6mut.log")
  [ "$c_pay" -gt 0 ] && ok "ARM 6 CONTROL: the LITERAL /users/glvov/.cargo/bin/cargo was intercepted and stubbed, not run" \
                     || no "ARM 6 CONTROL: no cargo PAYLOAD row -- the by-path seam did not intercept (log $W/arm6ctl.log)"
  [ "$c_can" -eq 0 ] && ok "ARM 6 CONTROL: no canary fired" \
                     || no "ARM 6 CONTROL: THE CANARY FIRED -- a real cargo was reachable"
  [ "$m_ref" -gt 0 ] && ok "ARM 6 MUTANT: with the cargo stub dropped the same wrapper is REFUSED, not run" \
                     || no "ARM 6 MUTANT: no refusal -- by-path coverage is not derived from the stub (log $W/arm6mut.log)"
  [ "$m_can" -eq 0 ] && ok "ARM 6 MUTANT: no canary fired either" \
                     || no "ARM 6 MUTANT: THE CANARY FIRED"
  rm -rf "$MUTDIR2"
fi

echo "### driver_exit_mutation_arms: pass=$pass fail=$fail  $(date -Is)"
[ "$fail" -eq 0 ] && echo "### MUTATION ARMS GREEN -- every arm flipped" \
                  || echo "### MUTATION ARMS RED -- an arm did not flip"
[ "$fail" -eq 0 ] || exit 1
exit 0
