#!/usr/bin/env bash
# fast_env_snapshot_gate_guard.sh -- C18-1-c.  The same-device gate in
# `retread_fast_env` must REFUSE when it cannot see its own subject, and must
# still export the store in the ordinary case.
#
# WHY THIS EXISTS.  Before C18-1-c the gate read
#     br_probe=${RETREAD_BUILD_ROOT:-$ws}
# so a caller that had not declared a build root got the store exported anyway,
# gated against the WORKSPACE -- a directory that is not where link(2) lands.
# On this filesystem both sit on device 48, so the defect was measured HARMLESS
# and would have stayed invisible until the day a harness put its build root on
# a different mount.  C18-1 boarded it as "an assumption wearing the clothes of
# a gate".  The fix refuses instead of guessing; this guard is the reader that
# says so, and ARM C is the mutation that proves the guard can fail.
#
#   usage: bash fast_env_snapshot_gate_guard.sh [<repo worktree>]
#   exits 0 on pass=N fail=0, 1 otherwise.  Touches no job root and no network.
#
# ARM C extracts the PRE-FIX file from a pinned commit constant, never from
# HEAD: a HEAD-relative mutation arm has a one-commit shelf life and this
# campaign has already been bitten by one (HARNESS-FIX-1, 13/2 not 15/15).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${1:-${HARNESS_REPO:-}}
if [ -z "$REPO" ]; then
  # the guard is synced into a task dir where $HERE/../.. is NOT a repo, so the
  # relative guess is TESTED before it is trusted (d8d98a4's lesson).
  cand=$(cd "$HERE/../.." 2>/dev/null && pwd)
  if [ -n "$cand" ] && git -C "$cand" rev-parse --git-dir >/dev/null 2>&1; then REPO=$cand; fi
fi
[ -n "$REPO" ] || { echo "FATAL: pass the harness worktree (or set HARNESS_REPO)"; exit 2; }
git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || { echo "FATAL: $REPO is not a git worktree"; exit 2; }

PREFIX_COMMIT=8e5b64a97cbed2b15961c92e4c8165f8f8a3c604   # the last commit carrying the guessing gate
PERSIST=/oscar/data/stellex/glvov/agrescap/cache/retread
# The scratch tree MUST live on the same device as the store, or every arm that
# expects an export fails for the WRONG reason: the first run of this guard put
# it in /tmp and arms A and C came back refused-on-EXDEV, which is the gate
# working correctly on a fixture that lied. Arm F is that case, on purpose.
TD=$(mktemp -d "$PERSIST/.gate-guard-XXXXXX") || exit 2
trap 'rm -rf "$TD"' EXIT
FIXED=$HERE/retread_fast_env.sh
[ -f "$FIXED" ] || { echo "FATAL: no retread_fast_env.sh beside this guard"; exit 2; }
git -C "$REPO" cat-file blob "$PREFIX_COMMIT:harness/tools/retread_fast_env.sh" > "$TD/prefix.sh" || {
  echo "FATAL: could not extract the pre-fix blob from $PREFIX_COMMIT"; exit 2; }
cmp -s "$FIXED" "$TD/prefix.sh" && { echo "FATAL: the fixed file and the pinned pre-fix blob are IDENTICAL -- this guard cannot fail"; exit 2; }

pass=0; fail=0
ok () { echo "PASS $1"; pass=$((pass+1)); }
no () { echo "FAIL $1"; fail=$((fail+1)); }

# Run the function in a SUBSHELL with a clean slate and report what it exported.
# RETREAD_FAST_TMP_ROOT/SLURM_JOB_ID are left unset on purpose: that branch
# shells out to python3, which is banned on the login node, and it is not what
# this guard is about.
run_fast_env () {   # $1 = file to source, rest = env assignments
  local file=$1; shift
  local ws=$TD/ws; mkdir -p "$ws"
  env -u RETREAD_GIT_SNAPSHOT_STORE -u RETREAD_BUILD_ROOT \
      -u RETREAD_FAST_TMP_ROOT -u SLURM_JOB_ID \
      RETREAD_PERSIST_CACHE_ROOT="$PERSIST" \
      "$@" bash -c '
        set -uo pipefail
        . "$1" || exit 3
        retread_fast_env "$2" >"$3/out.txt" 2>"$3/err.txt"
        rc=$?
        echo "RC=$rc"
        echo "STORE=${RETREAD_GIT_SNAPSHOT_STORE:-<unset>}"
      ' _ "$file" "$ws" "$TD"
}

echo "### ARM A -- build root declared and on the same device: the store IS exported"
A=$(run_fast_env "$FIXED" RETREAD_BUILD_ROOT="$TD/bldroot")
echo "$A" | sed 's/^/    /'
case "$A" in *"STORE=/oscar/data/stellex/glvov/agrescap/cache/retread/git-snapshots"*) ok "A: exported";; *) no "A: the ordinary case stopped working";; esac
grep -q 'dev .* == build root dev' "$TD/out.txt" && ok "A: printed the dev-match line" || no "A: no dev-match line"

echo "### ARM B -- RETREAD_BUILD_ROOT UNSET: the gate REFUSES and leaves the store off"
B=$(run_fast_env "$FIXED")
echo "$B" | sed 's/^/    /'
case "$B" in *"STORE=<unset>"*) ok "B: refused, store left unset";; *) no "B: exported the store with nothing to gate against";; esac
grep -q 'REFUSING the shared git snapshot store -- RETREAD_BUILD_ROOT is unset' "$TD/err.txt" \
  && ok "B: the refusal names RETREAD_BUILD_ROOT" || no "B: the refusal does not name the missing variable"

echo "### ARM C -- MUTATION: the pinned PRE-FIX file under arm B's conditions must EXPORT (the defect)"
C=$(run_fast_env "$TD/prefix.sh")
echo "$C" | sed 's/^/    /'
case "$C" in *"STORE=/oscar/data/stellex/glvov/agrescap/cache/retread/git-snapshots"*)
  ok "C: the pre-fix file reproduces the defect, so arm B is a real assertion";;
  *) no "C: the pre-fix file did NOT reproduce the defect -- arm B cannot fail and is worthless";; esac

echo "### ARM D -- explicit opt-out wins over everything"
D=$(run_fast_env "$FIXED" RETREAD_BUILD_ROOT="$TD/bldroot" RETREAD_FAST_ENV_GIT_SNAPSHOT_STORE=0)
case "$D" in *"STORE=<unset>"*) ok "D: opt-out honoured";; *) no "D: opt-out ignored";; esac

echo "### ARM E -- a preset store is left exactly alone"
E=$(env RETREAD_GIT_SNAPSHOT_STORE=$TD/preset RETREAD_BUILD_ROOT=$TD/bldroot \
     RETREAD_PERSIST_CACHE_ROOT="$PERSIST" \
     bash -c '. "$1"; retread_fast_env "$2" >/dev/null 2>&1; echo "STORE=$RETREAD_GIT_SNAPSHOT_STORE"' _ "$FIXED" "$TD/ws")
case "$E" in *"STORE=$TD/preset"*) ok "E: preset left alone";; *) no "E: clobbered a preset store ($E)";; esac


echo "### ARM F -- build root on a DIFFERENT device: the EXDEV refusal the fix must not have broken"
FTD=$(mktemp -d) || exit 2                     # /tmp: a different device from the store, deliberately
F=$(run_fast_env "$FIXED" RETREAD_BUILD_ROOT="$FTD/bldroot")
echo "$F" | sed 's/^/    /'
case "$F" in *"STORE=<unset>"*) ok "F: cross-device build root still refused";; *) no "F: exported across devices -- EXDEV would cost the hardlink farm";; esac
grep -q 'EXDEV would cost the hardlink farm' "$TD/err.txt" && ok "F: the refusal names EXDEV" || no "F: no EXDEV refusal line"
rm -rf "$FTD"
echo "### FAST_ENV SNAPSHOT GATE GUARD SUMMARY pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
