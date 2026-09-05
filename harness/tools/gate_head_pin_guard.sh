#!/usr/bin/env bash
# gate_head_pin_guard.sh -- the guard for L3-1b-5.
#
# THE CASE IT EXISTS FOR, in one sentence: a fixture worktree that is the pinned
# commit and clean AT ENTRY, and has MOVED by snapshot time, must be REFUSED --
# which is exactly what `gate_build.sh` could not do before L3-1b-5, and exactly
# how L3-1b's job 5918567 came to write `binsnaps/cand-9fc0d7d` holding a binary
# built from a different commit and exit 0.
#
# Everything here runs on a throwaway git repo in a temp dir: no cargo, no
# retread, no network, seconds not minutes. The two end-to-end arms drive the
# REAL `gate_build.sh` and assert it refuses BEFORE any build (no
# `release-build.log` written), which is the only reason they are affordable.
#
# TWO MUTATION ARMS, because a guard that cannot fail is a defect (law 3): a
# copy of `gate_head_pin.sh` with the HEAD comparison deleted must let the moved
# worktree through, and a copy with the dirty re-check deleted must let the
# dirtied worktree through. If either mutant still refuses, this guard is not
# measuring what it claims and says so.
#
#   usage: gate_head_pin_guard.sh [tools-dir]
#   rc 0 = every check passed; rc 1 = at least one FAILED
set -uo pipefail
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="${1:-$SELF_DIR}"
PIN="$TOOLS/gate_head_pin.sh"
GATE="$TOOLS/gate_build.sh"
[ -f "$PIN" ]  || { echo "FATAL: no gate_head_pin.sh at $PIN"; exit 2; }
[ -f "$GATE" ] || { echo "FATAL: no gate_build.sh at $GATE"; exit 2; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/gatepin.XXXXXX")" || exit 2
trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
ck () { # ck <label> <expected-rc> <actual-rc>
  if [ "$2" = "$3" ]; then echo "PASS  $1 (rc=$3)"; pass=$((pass+1))
  else echo "FAIL  $1 (rc=$3, wanted $2)"; fail=$((fail+1)); fi
}

# ---- the fixture: a real repo with two commits and a detached worktree -------
R=$TMP/repo
git init -q "$R"
git -C "$R" config user.email g@example.invalid
git -C "$R" config user.name  guard
echo one > "$R/f.txt"; git -C "$R" add f.txt; git -C "$R" commit -qm one
A=$(git -C "$R" rev-parse --short HEAD)
echo two > "$R/f.txt"; git -C "$R" commit -qam two
B=$(git -C "$R" rev-parse --short HEAD)
WT=$TMP/wt
git -C "$R" worktree add -q --detach "$WT" "$A"
echo "### fixture: A=$A B=$B worktree=$WT"

# ---- 1..3 the shape of the check itself -------------------------------------
bash "$PIN" >/dev/null 2>&1; ck "usage: no arguments is a refusal" 4 $?
bash "$PIN" "$WT" "$A" entry >/dev/null 2>&1; ck "clean worktree at the pin, entry phase" 0 $?
bash "$PIN" "$WT" "$B" entry >/dev/null 2>&1; ck "worktree at the WRONG commit is refused" 12 $?

# ---- 4 THE NAMED CASE: HEAD moves between entry and snapshot ----------------
bash "$PIN" "$WT" "$A" entry >/dev/null 2>&1; e1=$?
git -C "$WT" checkout -q --detach "$B"
bash "$PIN" "$WT" "$A" snapshot >/dev/null 2>&1; e2=$?
ck "HEAD MOVED between entry and snapshot: entry passes" 0 "$e1"
ck "HEAD MOVED between entry and snapshot: snapshot REFUSES" 12 "$e2"
git -C "$WT" checkout -q --detach "$A"

# ---- 5 the tree goes dirty between entry and snapshot -----------------------
bash "$PIN" "$WT" "$A" entry >/dev/null 2>&1; e3=$?
echo scribble >> "$WT/f.txt"
bash "$PIN" "$WT" "$A" snapshot >/dev/null 2>&1; e4=$?
ck "DIRTIED between entry and snapshot: entry passes" 0 "$e3"
ck "DIRTIED between entry and snapshot: snapshot REFUSES" 13 "$e4"
git -C "$WT" checkout -q -- f.txt

# ---- 6..7 the real gate_build.sh refuses before it builds anything ----------
D=$TMP/harness; mkdir -p "$D"
run_gate () { # run_gate <expect_head or empty>; echoes rc
  local eh="$1" rc
  rm -f "$D/artifacts/release-build.log"
  if [ -z "$eh" ]; then
    env -u EXPECT_HEAD WT="$WT" D="$D" EXPECT_PASS=1 JOBS=1 bash "$GATE" >"$TMP/gate.out" 2>&1; rc=$?
  else
    WT="$WT" D="$D" EXPECT_HEAD="$eh" EXPECT_PASS=1 JOBS=1 bash "$GATE" >"$TMP/gate.out" 2>&1; rc=$?
  fi
  echo "$rc"
}
rc=$(run_gate ""); nb=$([ -f "$D/artifacts/release-build.log" ] && echo 1 || echo 0)
refused=0; [ "$rc" -eq 0 ] && refused=1     # 0 = it refused, 1 = it ran anyway
ck "gate_build.sh with EXPECT_HEAD UNSET refuses (rc was $rc)" 0 "$refused"
ck "  ... and it refused before any release build" 0 "$nb"
rc=$(run_gate "$B"); nb=$([ -f "$D/artifacts/release-build.log" ] && echo 1 || echo 0)
ck "gate_build.sh pinned to the WRONG commit refuses at entry" 4 "$rc"
ck "  ... and it refused before any release build" 0 "$nb"

# ---- 8 the reader/writer check: TWO call sites, one of them at snapshot time -
n=$(grep -c 'gate_head_pin.sh\|"\$HEAD_PIN"' "$GATE")
sites=$(grep -c '^bash "\$HEAD_PIN"' "$GATE")
echo "### gate_build.sh mentions the pin on $n lines, $sites of them call sites"
ck "gate_build.sh calls the pin TWICE (entry + snapshot)" 2 "$sites"
grep -q 'HEAD_PIN" "\$WT" "\$EXPECT_HEAD" snapshot' "$GATE"
ck "the SNAPSHOT-time call site exists by name" 0 $?

# ---- 9..10 MUTATION ARMS: the guard must be able to fail --------------------
M1=$TMP/mut-nohead.sh
sed 's/^if \[ "\$ok" -ne 1 \]; then$/if false; then/' "$PIN" > "$M1"
grep -q '^if false; then' "$M1" || { echo "FATAL: mutation 1 did not apply"; exit 2; }
git -C "$WT" checkout -q --detach "$B"
bash "$M1" "$WT" "$A" snapshot >/dev/null 2>&1; m1=$?
git -C "$WT" checkout -q --detach "$A"
if [ "$m1" -eq 0 ]; then echo "PASS  MUTATION 1 (HEAD comparison deleted) lets the moved worktree through -- the check is load-bearing"; pass=$((pass+1))
else echo "FAIL  MUTATION 1 still refused (rc=$m1) -- arm 4 is not measuring the HEAD comparison"; fail=$((fail+1)); fi

M2=$TMP/mut-nodirty.sh
sed 's/^if \[ "\$DIRTY" -ne 0 \]; then$/if false; then/' "$PIN" > "$M2"
grep -q '^if false; then' "$M2" || { echo "FATAL: mutation 2 did not apply"; exit 2; }
echo scribble >> "$WT/f.txt"
bash "$M2" "$WT" "$A" snapshot >/dev/null 2>&1; m2=$?
git -C "$WT" checkout -q -- f.txt
if [ "$m2" -eq 0 ]; then echo "PASS  MUTATION 2 (dirty re-check deleted) lets the dirtied worktree through -- the check is load-bearing"; pass=$((pass+1))
else echo "FAIL  MUTATION 2 still refused (rc=$m2) -- arm 5 is not measuring the dirty re-check"; fail=$((fail+1)); fi

echo "### GATE_HEAD_PIN_GUARD passed=$pass failed=$fail"
[ "$fail" -eq 0 ] || { echo "### GUARD FAILED"; exit 1; }
echo "### GUARD PASSED"
exit 0
