#!/bin/bash
# merge-4-12-h/i/j: ONE test gate + release build + binsnap for a merge candidate.
#
# DERIVED BY SUBSTITUTION from merge-e/gate_build.sh (the script that gated
# c0a87d3) with FOUR additions, each of them a defect this campaign was bitten
# by and each of them a reader:
#
#   1. THE SUITE MUST HAVE PRINTED A RESULT LINE. C14.7(a): c14-gatep2 5741936
#      built the release binary clean, failed to compile the TEST binary, printed
#      no `test result:` line at all, walked an EMPTY failure list, reported
#      `in-suite failures: 0  isolated_red_count=0` and EXITED 0 WITH A BINSNAP.
#      A gate that cannot fail is a defect.
#   2. THE SPLIT MUST EQUAL WHAT WAS PREDICTED BEFORE THE RUN. $EXPECT_PASS is
#      stated in the submit command, and a mismatch is a refusal -- the "gate
#      split reconciling to the sum of fix guards" rule, enforced by the script
#      rather than by an agent reading a log.
#   3. THE BINSNAP GOES THROUGH tools/binsnap_ancestry_guard.sh (p6q-1/p6r), so
#      a candidate that silently dropped a fix a sibling merged cannot be
#      snapshotted.
#   4. EVERY TARGET MUST COMPILE, not just the lib. p6aa/LANE-C-WARM-LOG s27:
#      `tests/isaacsim_relax.rs` and `tests/wheel_fetch_live.rs` had not
#      compiled for an unknown number of merges -- their `RetreadConfig` struct
#      literals were missing six fields the struct had since gained -- and
#      NOTHING NOTICED, because this gate's only compile of test code is
#      `cargo test --lib`. An integration test that no gate builds is an
#      assertion nobody is running; the same hole covers examples and benches.
#      `cargo build --all-targets --keep-going` closes it for a few seconds of
#      debug codegen on top of a build the gate is doing anyway, and the
#      refusal names EVERY broken target rather than only the first.
#
# usage: WT=<worktree> D=<harness dir> SEED=<target to seed from> \
#        EXPECT_PASS=<n> [EXPECT_IGNORED=21] bash gate_build.sh
set -u
T=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11
: "${WT:?set WT to the candidate worktree}"
: "${D:?set D to this harness directory}"
: "${EXPECT_PASS:?state the expected pass count BEFORE the run}"
EXPECT_IGNORED=${EXPECT_IGNORED:-21}
SEED=${SEED:-}
A=$D/artifacts
mkdir -p "$A" "$A/privhome/home" "$A/privhome/cache"
export HOME="$A/privhome/home" XDG_CACHE_HOME="$A/privhome/cache"
export XDG_DATA_HOME="$A/privhome/home/.local/share" XDG_CONFIG_HOME="$A/privhome/home/.config"
export CARGO_HOME=/users/glvov/.cargo RUSTUP_HOME=/users/glvov/.rustup
export PATH="$CARGO_HOME/bin:$PATH"
export CARGO_TERM_COLOR=never
J=${JOBS:-8}
export CARGO_BUILD_JOBS=$J
unset RUST_LOG
cd "$WT" || exit 3
echo "### host=$(hostname) $(date -Is) HEAD=$(git rev-parse HEAD) short=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) JOBS=$J"
echo "### EXPECTED SPLIT (stated before the run): $EXPECT_PASS passed; 0 failed; $EXPECT_IGNORED ignored"

DIRTY=$(git status --porcelain | wc -l)
[ "$DIRTY" -eq 0 ] || { echo "### STOP: candidate worktree is dirty ($DIRTY paths) -- law 11"; exit 4; }

if [ ! -d "$WT/target" ] && [ -n "$SEED" ] && [ -d "$SEED" ]; then
  echo "### seeding target from $SEED"; S=$(date +%s)
  cp -a "$SEED" "$WT/target" && echo "### seed_ok wall=$(( $(date +%s) - S ))s" || echo "### seed_FAILED (continuing cold)"
fi

S=$(date +%s)
cargo build --release -j "$J" 2>&1 | tee "$A/release-build.log" | tail -30
brc=${PIPESTATUS[0]}
echo "### BUILD_RC=$brc wall=$(( $(date +%s) - S ))s $(date -Is)"
[ "$brc" -eq 0 ] || { echo "### STOP: release build failed"; exit 5; }

# (4) every target must COMPILE -- integration tests, examples, benches -- and
# not merely the lib. --keep-going so a refusal names all of them at once.
S=$(date +%s)
cargo build --all-targets --keep-going -j "$J" 2>&1 | tee "$A/all-targets-build.log" | tail -20
arc=${PIPESTATUS[0]}
echo "### ALLTARGETS_BUILD_RC=$arc wall=$(( $(date +%s) - S ))s $(date -Is)"
mapfile -t BROKEN < <(grep -oE 'could not compile `[^`]*` \([^)]*\)' "$A/all-targets-build.log" | sort -u)
echo "### broken_target_count=${#BROKEN[@]}"
if [ "$arc" -ne 0 ] || [ "${#BROKEN[@]}" -gt 0 ]; then
  echo "### STOP: a target does not compile -- the lib gate would never have seen this"
  if [ "${#BROKEN[@]}" -gt 0 ]; then
    for b in "${BROKEN[@]}"; do echo "###   BROKEN TARGET: $b"; done
  else
    echo "###   rc=$arc with no 'could not compile' line -- read $A/all-targets-build.log"
  fi
  exit 11
fi

S=$(date +%s)
timeout --foreground --kill-after=60s 5400s cargo test --lib -j "$J" 2>&1 | tee "$A/gate.log" | tail -25
trc=${PIPESTATUS[0]}
echo "### TEST_RC=$trc wall=$(( $(date +%s) - S ))s $(date -Is)"
SPLIT=$(grep -E '^test result:' "$A/gate.log" | tail -1)
echo "### GATE SPLIT (printed): $SPLIT"

# (1) a suite that printed nothing did not run
[ -n "$SPLIT" ] || { echo "### STOP: the suite printed no test-result line (TEST_RC=$trc) -- it did not run"; exit 6; }

# every failure re-run isolated
mapfile -t FAILED < <(sed -n '/^failures:$/,/^test result:/p' "$A/gate.log" | grep -E '^    [a-z]' | sed 's/^    //' | sort -u)
echo "### in-suite failures: ${#FAILED[@]}"
iso_bad=0
for t in "${FAILED[@]}"; do
  echo "### isolating $t"
  cargo test --lib -j "$J" -- --exact --test-threads=1 "$t" 2>&1 | tee "$A/iso-$t.log" | grep -E '^test result:'
  if grep -qE '^test result: ok\.' "$A/iso-$t.log"; then echo "### ISOLATED_GREEN $t"; else echo "### ISOLATED_RED $t"; iso_bad=$((iso_bad+1)); fi
done
echo "### isolated_red_count=$iso_bad"
[ "$iso_bad" -eq 0 ] || { echo "### STOP: a failure is real, not a race"; exit 7; }

# (2) the split must be the one predicted
GOT_PASS=$(echo "$SPLIT"    | sed -nE 's/^test result: ok\. ([0-9]+) passed.*/\1/p')
GOT_FAIL=$(echo "$SPLIT"    | sed -nE 's/.* ([0-9]+) failed.*/\1/p')
GOT_IGNORED=$(echo "$SPLIT" | sed -nE 's/.* ([0-9]+) ignored.*/\1/p')
echo "### split parsed: passed=$GOT_PASS failed=$GOT_FAIL ignored=$GOT_IGNORED"
[ "${GOT_PASS:-x}" = "$EXPECT_PASS" ] || { echo "### STOP: split $GOT_PASS != predicted $EXPECT_PASS -- the guard sum does not reconcile"; exit 8; }
[ "${GOT_FAIL:-x}" = "0" ] || { echo "### STOP: $GOT_FAIL failed in-suite"; exit 8; }
[ "${GOT_IGNORED:-x}" = "$EXPECT_IGNORED" ] || { echo "### STOP: ignored $GOT_IGNORED != $EXPECT_IGNORED"; exit 8; }

# (3) the ancestry guard, before any binsnap
bash "$T/tools/binsnap_ancestry_guard.sh" "$WT" HEAD || { echo "### STOP: BINSNAP REFUSED by the ancestry guard"; exit 9; }

SHORT=$(git rev-parse --short HEAD)
SNAP=$T/binsnaps/cand-$SHORT
mkdir -p "$SNAP"
cp -f "$WT/target/release/pixi-build-retread" "$SNAP/pixi-build-retread" || exit 10
git -C "$WT" rev-parse HEAD > "$SNAP/COMMIT"
sha256sum "$SNAP/pixi-build-retread" | awk '{print $1}' > "$SNAP/SHA256"
echo "### BINSNAP $SNAP sha256=$(cat "$SNAP/SHA256")"
ls -la "$SNAP"
exit 0
