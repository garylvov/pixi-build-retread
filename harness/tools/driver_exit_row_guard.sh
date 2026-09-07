#!/usr/bin/env bash
# driver_exit_row_guard.sh -- CLEANUP-SEAM-3. The reader for the EXIT trap in
# tools/retread_fast_env.sh (`retread_exit_row` / `retread_exit_row_install`).
#
#   usage: driver_exit_row_guard.sh          (HARNESS_REPO=<repo> to point it)
#
# FIXTURE-ONLY, AND THAT IS NOT A STYLE NOTE. Every arm here builds a THREE-LINE
# driver under $TMPDIR that sources the real library and exits; NOT ONE ARM RUNS
# A TEMPLATE. Guard job 6023585 is why the rule is written down: an arm of
# leftover_check_guard.sh ran `bash arms/mh1_relock.sh` with SLURM_JOB_ID forced
# to 999999, the template sailed past its gates into stage_build_mirror, and it
# spent eleven minutes building into the SHARED stage mirror under
# STAGE_MIRROR_ROOT -- leaving `85db7fdbbf51206a0cb57fa0d55e0e74.building.
# 999999-MH1-2247202` behind. A guard that can write production state is not
# fixture-only however green it prints. The one thing this file reads from the
# real tree it reads STATICALLY, with grep, and never executes (arm X8).
#
# WHAT IS UNDER TEST. mCB-relock 6022684 went FAILED 14:0 and its stdout carries
# no row from any of the four families `cleanup_gated.sh`'s JOB_FATAL_RE knew --
# `grep -c '_EXIT=' ` over mergeB31/logs/slurm-6022684.out returns 0 -- because
# the `### <TAG>_EXIT=<rc>` row's only producer was the sbatch WRAPPER, and a
# driver that ends on its own `exit 14` never reaches one. Owner mCB-cleanup
# 6023543 REFUSED (FAILED 2:0) and two roots were left with no reaper. The fix
# gives the row a producer that cannot be forgotten: an EXIT trap installed by
# the ONE library all six templates source.
#
# THE ARMS
#   X1  a driver that exits 14 prints `### <TAG>_EXIT=14`, and STILL EXITS 14
#   X2  a driver that exits 0 prints `_EXIT=0`, which the gate's own fatal
#       family must NOT match -- zero is not fatal
#   X3  CHAINING: a pre-existing EXIT trap still runs, and runs FIRST
#   X4  CHAINING through bash's quoting: a trap body containing a single quote
#       survives the unquote (this is where a naive `sed` implementation breaks)
#   X5  SILENCE: a script that never set TAG gets NO row -- guards source this
#       library too, and several of them write stdout another guard greps
#   X6  the RETREAD_EXIT_ROW=0 opt-out
#   X7  IDEMPOTENCE: sourcing the library twice installs one trap, not two
#   X8  STATIC, no execution: all six templates source this library, and each
#       sets TAG ABOVE the line that sources it (a TAG set below would print
#       `UNTAGGED_EXIT=`)
#   X9  the row X1 produced is matched by the REAL JOB_FATAL_RE, extracted from
#       cleanup_gated.sh rather than retyped here
#   X10 MUTATION: the `trap … EXIT` install line cut from a copy of the library
#       -> X1's driver prints no row at all. X1 can fail.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${HARNESS_REPO:-}
[ -n "$REPO" ] || REPO=$(cd -- "$HERE/../.." && pwd)
if [ ! -d "$REPO/harness/tools" ] || ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  REPO=/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools
fi
H=$REPO/harness
LIB=$H/tools/retread_fast_env.sh
GATE=$H/phase_template/cleanup_gated.sh
[ -f "$LIB" ]  || { echo "GUARD FATAL: no library at $LIB"; exit 2; }
[ -f "$GATE" ] || { echo "GUARD FATAL: no gate at $GATE"; exit 2; }
W=$(mktemp -d "${TMPDIR:-/tmp}/driver-exit-row-guard.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
pass=0; fail=0
ok()  { echo "GUARD  ok : $*"; pass=$((pass+1)); }
bad() { echo "GUARD FAIL: $*"; fail=$((fail+1)); }
echo "### driver_exit_row_guard for $LIB (CLEANUP-SEAM-3)"

# The fixture driver. It is the WHOLE shape a template has in common with this
# seam -- set TAG, source the library, exit -- and nothing else, so no arm here
# can reach a stage mirror, a cache, or a scheduler.
mk_driver () {   # mk_driver <out> <lib> <tag|-> <rc> [pre-source line]
  { printf '#!/usr/bin/env bash\n'
    printf 'set -uo pipefail\n'
    [ "$3" = - ] || printf 'TAG=%s\n' "$3"
    [ $# -ge 5 ] && printf '%s\n' "$5"
    printf '. "%s"\n' "$2"
    printf 'exit %s\n' "$4"
  } > "$1"
}

# ---- X1: the row, and the exit status -----------------------------------
mk_driver "$W/x1.sh" "$LIB" SEAM3X 14
OUT=$(bash "$W/x1.sh" 2>&1); RC=$?
if [ "$RC" = 14 ]; then
  ok "X1: the driver still exits 14 -- an EXIT trap must not move the status"
else
  bad "X1: rc=$RC, want 14 -- the trap changed the exit status"
fi
if printf '%s\n' "$OUT" | grep -qx '### SEAM3X_EXIT=14'; then
  ok "X1: it prints '### SEAM3X_EXIT=14' -- the row 6022684 never printed"
else
  bad "X1: no exit row. stdout was:"; printf '%s\n' "$OUT" | sed 's/^/      /'
fi

# ---- X2: zero is not fatal ------------------------------------------------
mk_driver "$W/x2.sh" "$LIB" SEAM3X 0
OUT0=$(bash "$W/x2.sh" 2>&1); RC=$?
if [ "$RC" = 0 ] && printf '%s\n' "$OUT0" | grep -qx '### SEAM3X_EXIT=0'; then
  ok "X2: a clean driver prints '_EXIT=0' (rc=$RC) -- the success row of the same producer"
else
  bad "X2: rc=$RC, row='$(printf '%s' "$OUT0" | tr '\n' '|')'"
fi

# ---- X3: the chain, and its order -----------------------------------------
# phaseN_cert.sh sets `trap collect_artifacts EXIT` BEFORE it sources the
# library. A bare `trap … EXIT` in the library would delete that silently.
mk_driver "$W/x3.sh" "$LIB" CHAINX 5 'prev () { echo "PREV-TRAP-RAN"; }; trap prev EXIT'
OUT=$(bash "$W/x3.sh" 2>&1); RC=$?
if [ "$RC" = 5 ] && printf '%s\n' "$OUT" | grep -q 'PREV-TRAP-RAN' \
   && printf '%s\n' "$OUT" | grep -qx '### CHAINX_EXIT=5'; then
  if [ "$(printf '%s\n' "$OUT" | grep -n 'PREV-TRAP-RAN' | cut -d: -f1)" \
       -lt "$(printf '%s\n' "$OUT" | grep -n 'CHAINX_EXIT' | cut -d: -f1)" ]; then
    ok "X3: a pre-existing EXIT trap survives AND runs first (rc=$RC)"
  else
    bad "X3: the pre-existing trap ran AFTER the row -- artifacts would be collected too late"
  fi
else
  bad "X3: the pre-existing trap was CLOBBERED (rc=$RC). stdout:"; printf '%s\n' "$OUT" | sed 's/^/      /'
fi

# ---- X4: chaining through bash's own quoting ------------------------------
# `trap -p` renders an embedded single quote as '\'' . An implementation that
# does not unquote it re-runs a broken body, or eval's a syntax error.
mk_driver "$W/x4.sh" "$LIB" QX 3 "trap 'echo \"IT'\\''S QUOTED\"' EXIT"
OUT=$(bash "$W/x4.sh" 2>&1); RC=$?
if [ "$RC" = 3 ] && printf '%s\n' "$OUT" | grep -qF "IT'S QUOTED" \
   && printf '%s\n' "$OUT" | grep -qx '### QX_EXIT=3'; then
  ok "X4: a trap body containing a single quote is unquoted correctly and both rows print"
else
  bad "X4: rc=$RC, the quoted chain broke. stdout:"; printf '%s\n' "$OUT" | sed 's/^/      /'
fi

# ---- X5: SILENCE for anything that is not a driver ------------------------
mk_driver "$W/x5.sh" "$LIB" - 7
OUT=$(bash "$W/x5.sh" 2>&1); RC=$?
if [ "$RC" = 7 ] && ! printf '%s\n' "$OUT" | grep -q '_EXIT='; then
  ok "X5: a script that never set TAG gets NO row (rc=$RC) -- a guard's stdout is another guard's fixture"
else
  bad "X5: rc=$RC, an untagged script printed: $(printf '%s' "$OUT" | tr '\n' '|')"
fi

# ---- X6: the opt-out -------------------------------------------------------
OUT=$(RETREAD_EXIT_ROW=0 bash "$W/x1.sh" 2>&1); RC=$?
if [ "$RC" = 14 ] && ! printf '%s\n' "$OUT" | grep -q '_EXIT='; then
  ok "X6: RETREAD_EXIT_ROW=0 suppresses the row and leaves the status alone (rc=$RC)"
else
  bad "X6: rc=$RC, the opt-out did not suppress: $(printf '%s' "$OUT" | tr '\n' '|')"
fi

# ---- X7: idempotence -------------------------------------------------------
# phaseN_relock.sh sources $FAST_ENV once, but a lane that re-sources it inside
# a function must not get two rows -- a second row is a second `$?` read, and
# the gate would see a row for a status nothing returned.
mk_driver "$W/x7.sh" "$LIB" IDEMX 6 ". \"$LIB\""
OUT=$(bash "$W/x7.sh" 2>&1); RC=$?
N=$(printf '%s\n' "$OUT" | grep -c '_EXIT=')
if [ "$RC" = 6 ] && [ "$N" = 1 ]; then
  ok "X7: sourcing the library twice installs ONE trap and prints ONE row (rc=$RC)"
else
  bad "X7: rc=$RC, rows=$N, want exactly 1: $(printf '%s' "$OUT" | tr '\n' '|')"
fi

# ---- X8: STATIC -- the six templates, read with grep, never executed -------
# The reader/writer law's production half: a trap in a library nobody sources
# is a zero-caller capability. Each template must source the library AND set TAG
# above that line, because TAG is read at exit and a TAG set below would leave
# the row reading `### UNTAGGED_EXIT=`.
TPLS='phase_template/phaseN_relock.sh phase_template/phaseN_cert.sh arms/mh1_relock.sh arms/c29_relock.sh proof/hlgd_relock.sh instrumented/p6b_relock.sh'
n_ok=0; n_tpl=0
for rel in $TPLS; do
  f=$H/$rel; n_tpl=$((n_tpl+1))
  [ -f "$f" ] || { bad "X8: no template at $f"; continue; }
  ls=$(grep -n '^[[:space:]]*\.[[:space:]]*"\$FAST_ENV"' "$f" | head -1 | cut -d: -f1)
  lt=$(grep -n '^[[:space:]]*TAG=' "$f" | head -1 | cut -d: -f1)
  if [ -n "$ls" ] && [ -n "$lt" ] && [ "$lt" -lt "$ls" ]; then
    n_ok=$((n_ok+1))
  else
    bad "X8: $rel sources the library at line '${ls:-<never>}' and sets TAG at line '${lt:-<never>}' -- TAG must come first"
  fi
done
[ "$n_ok" = "$n_tpl" ] && ok "X8: all $n_tpl templates source the library and set TAG above the source line (static; nothing executed)"

# ---- X9: the gate's OWN regex, extracted, not retyped ----------------------
GRE=$(grep -m1 "^JOB_FATAL_RE='" "$GATE" | sed "s/^JOB_FATAL_RE='//; s/'.*$//")
if [ -z "$GRE" ]; then
  bad "X9: could not extract JOB_FATAL_RE from $GATE -- X9 did not run"
else
  m14=$(printf '%s\n' "### SEAM3X_EXIT=14" | grep -cE "$GRE")
  m0=$(printf '%s\n'  "### SEAM3X_EXIT=0"  | grep -cE "$GRE")
  mf=$(printf '%s\n'  "### FATAL: the stage mirror was already dirty BEFORE this lock." | grep -cE "$GRE")
  mp=$(printf '%s\n'  "### PREAMBLE FATAL: SMOKE FIX rc=3 -- this binary did not reach the frontend" | grep -cE "$GRE")
  md=$(printf '%s\n'  "### FATAL two retread_fast_env.sh candidates DIFFER -- a vs b" | grep -cE "$GRE")
  if [ "$m14" = 1 ] && [ "$m0" = 0 ] && [ "$mf" = 1 ] && [ "$mp" = 0 ] && [ "$md" = 0 ]; then
    ok "X9: the gate's own fatal family takes _EXIT=14 and '### FATAL:', and refuses _EXIT=0, '### PREAMBLE FATAL:' and the colon-less '### FATAL two …'"
  else
    bad "X9: family verdicts wrong -- _EXIT=14:$m14(want 1) _EXIT=0:$m0(0) FATAL::$mf(1) PREAMBLE-FATAL:$mp(0) FATAL-no-colon:$md(0)"
  fi
fi

# ---- X10: THE MUTATION -----------------------------------------------------
MUT=$W/retread_fast_env.MUT.sh
sed 's/^  trap retread_exit_row EXIT  *# EXIT-ROW-INSTALL (MUTATION ANCHOR)$/  : # MUTATION: the trap is never installed/' "$LIB" > "$MUT"
nmut=$(diff "$LIB" "$MUT" | grep -c '^< ')
if [ "$nmut" -ne 1 ]; then
  bad "X10: the mutation changed $nmut line(s), want exactly 1 -- X1 cannot fail, so it proves nothing"
else
  mk_driver "$W/x10.sh" "$MUT" SEAM3X 14
  OUT=$(bash "$W/x10.sh" 2>&1); RC=$?
  if [ "$RC" = 14 ] && ! printf '%s\n' "$OUT" | grep -q '_EXIT='; then
    ok "X10: THE DEFECT, REPRODUCED -- with the install line cut the driver exits 14 in silence, exactly as 6022684 did"
  else
    bad "X10: the mutant still printed a row (rc=$RC) -- X1 cannot fail: $(printf '%s' "$OUT" | tr '\n' '|')"
  fi
fi

echo "### driver_exit_row_guard: pass=$pass fail=$fail -- $( [ "$fail" = 0 ] && echo PASS || echo FAIL )"
[ "$fail" = 0 ]
