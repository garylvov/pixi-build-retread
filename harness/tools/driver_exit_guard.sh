#!/usr/bin/env bash
# driver_exit_guard.sh -- HARNESS-EXIT-2. EVERY driver's own failure must reach
# Slurm's exit code (CLAUDE.md law 9), and this is the fixture that says so for
# the whole family instead of one lane's driver.
#
# WHY THIS EXISTS. HARNESS-EXIT-1 found FOUR jobs in one day that printed a
# refusal and reported `COMPLETED 0:0` to sacct -- c181-dryrun internal rc=8,
# c181-negctl rc=1, l3-2arm job_fatal=1, c181-twoarm reap_demo_ok=0. C18-1
# closed it in ITS OWN driver with `c181_driver_exit_guard.sh` (fixture pair
# bad 0:0 / good 7:0, jobs 5910974/5910975). The shape it caught was not one
# lane's mistake: it is the family idiom `bash <driver>` followed by
# `echo "### X_EXIT=$?"`, which captures the rc into a STRING and throws it
# away, and it was present in the merge-lane gate wrappers, the p6ad-4
# wrappers, the harness-fix wrappers, l3 and c181 alike. sacct state is not a
# success criterion for a harness that ends on an echo.
#
# WHAT IT ASSERTS, and why each shape is what it is.
#
#   FAMILY 0  the injection itself is not vacuous: the stub really exits 7.
#
#   FAMILY A  the VERSIONED drivers this lane fixed. Each is run -- the real
#             file, unmodified, over a throwaway root -- with a failure
#             injected into the thing it drives, and its process exit code must
#             be NON-ZERO. The mutation arm re-runs the IDENTICAL fixture over
#             the PRE-FIX BLOB, extracted from the pinned commit constant
#             $PREFIX_COMMIT (never HEAD:<path> -- HARNESS-FIX-1 learned that
#             the hard way when a HEAD-relative arm went dead the moment the
#             fix landed), and it must be ZERO. A pair that reports the same
#             code both ways is a FAILURE, not a pass: the fixture could not
#             tell the shapes apart.
#
#   FAMILY B  the WRAPPER shape, over every live `.sbatch` in the task tree.
#             The wrapper is run for real with `bash` shimmed on PATH: the
#             drift check is stubbed CLEAN so the wrapper reaches its payload,
#             and the payload is stubbed FAILING at 7. A fixed wrapper must
#             exit non-zero. The mutation arm is the pre-fix epilogue written
#             out as a literal constant below -- the exact two lines every one
#             of these files used to end on -- and must exit 0.
#
#   FAMILY C  the drivers that were ALREADY correct, locked so they stay that
#             way. Each is driven to a REAL refusal (a leftover token in the
#             phase templates' own self-check, a root the cleanup gate cannot
#             derive, a commit the drift check cannot find) and its exit code
#             must carry it.
#
#   usage: bash driver_exit_guard.sh            # everything
#          bash driver_exit_guard.sh A|B|C|0    # one family
#
# rc 0 every assertion passed;  rc 1 at least one failed;  rc 4 fixture fatal.
set -uo pipefail

# ---- the pinned pre-fix commit -------------------------------------------
# a83565b is the harness tip immediately BEFORE HARNESS-EXIT-2's fixes. It is a
# CONSTANT on purpose: the mutation arms must keep reproducing the defect after
# the fix lands, and `HEAD:<path>` stops doing that the instant it does.
PREFIX_COMMIT=a83565b039f4a8a6d98f745307a24ff07b2600b6

REPO=${HARNESS_REPO:-/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools}
TASK=${HARNESS_TASK_DIR:-/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11}
WHICH=${1:-all}

[ -d "$REPO" ] || { echo "GUARD FATAL: no repo at $REPO"; exit 4; }
git -C "$REPO" rev-parse --verify "$PREFIX_COMMIT^{commit}" >/dev/null 2>&1 || {
  echo "GUARD FATAL: $PREFIX_COMMIT is not a commit in $REPO"; exit 4; }

W=$(mktemp -d "${TMPDIR:-/tmp}/driver_exit_guard.XXXXXX") || { echo "GUARD FATAL: no temp dir"; exit 4; }
trap 'rm -rf "$W"' EXIT
echo "### driver_exit_guard  repo=$REPO  task=$TASK  prefix_commit=$PREFIX_COMMIT"
echo "### work=$W  family=$WHICH  host=$(hostname)  $(date -Is)"

pass=0; fail=0
ok  () { pass=$((pass + 1)); echo "  PASS  $*"; }
no  () { fail=$((fail + 1)); echo "  FAIL  $*"; }

# extract a pre-fix blob to a runnable file; echoes the path, empty on failure
prefix_blob () {  # $1 = repo-relative path
  local out="$W/prefix/$(echo "$1" | tr / _)"
  mkdir -p "$W/prefix"
  git -C "$REPO" cat-file blob "$PREFIX_COMMIT:$1" > "$out" 2>/dev/null || { echo ""; return; }
  chmod +x "$out"; echo "$out"
}

# ---------------------------------------------------------------- FAMILY 0 --
STUB=$W/stub_harness.sh
cat > "$STUB" <<'EOS'
#!/bin/sh
echo "### [guard stub] the driven thing is refusing on purpose"
exit 7
EOS
chmod +x "$STUB"

if [ "$WHICH" = all ] || [ "$WHICH" = 0 ]; then
  echo "=== FAMILY 0 -- the injection is not vacuous"
  "$STUB" >/dev/null 2>&1; s=$?
  [ "$s" -eq 7 ] && ok "the stub harness exits 7 (it is a real failure to inject)" \
                 || no "the stub harness exited $s, not 7 -- every arm below is meaningless"
fi

# ---------------------------------------------------------------- FAMILY A --
# run_pair <label> <repo-relative path> <runner-fn>
# The runner is called as: <runner-fn> <script-to-run> <arm-tag>
run_pair () {
  local label=$1 rel=$2 runner=$3 fixed=$REPO/harness/$2 old rcf rco
  old=$(prefix_blob "harness/$rel")
  [ -f "$fixed" ] || { no "$label: no such file $fixed"; return; }
  [ -n "$old" ]   || { no "$label: no blob at $PREFIX_COMMIT:harness/$rel"; return; }
  "$runner" "$fixed" fixed  >"$W/$label.fixed.log" 2>&1; rcf=$?
  "$runner" "$old"   prefix >"$W/$label.prefix.log" 2>&1; rco=$?
  [ "$rcf" -ne 0 ] && ok "$label FIXED re-raises: rc=$rcf" \
                   || no "$label FIXED swallowed its failure: rc=0 (log $W/$label.fixed.log)"
  if [ "$rco" -eq 0 ]; then
    ok "$label PRE-FIX ($PREFIX_COMMIT) swallows as it did: rc=0"
  elif [ "$rco" -eq "$rcf" ]; then
    no "$label VACUOUS: both shapes reported $rcf -- the fixture cannot tell them apart"
  else
    no "$label PRE-FIX reported $rco, not 0 -- read $W/$label.prefix.log before trusting the pair"
  fi
}

# --- A1 wheel_store_reclaim.sh: an entry it cannot fully return -------------
run_reclaim () {  # $1 script  $2 tag
  local r=$W/reclaim-$2/wheels/deadbeefcafe
  rm -rf "$W/reclaim-$2"; mkdir -p "$r"
  : > "$r/.pkg-1.0-py3-none-any.whl.retread-fill-v1.lock"
  # the lock is aged past the TTL so it is real debris, and a stray non-wheel
  # file makes the `rmdir` fail -- which is exactly the WARN that used to be
  # printed and then thrown away.
  touch -d '-3 hours' "$r/.pkg-1.0-py3-none-any.whl.retread-fill-v1.lock"
  : > "$r/SOMETHING-ELSE-IS-IN-HERE.txt"
  bash "$1" --apply "$W/reclaim-$2/wheels"
}

# --- A2 p6ad_negctl.sh: an arm whose mutation does not apply ----------------
run_negctl () {  # $1 script  $2 tag
  local base=$W/negctl-$2
  rm -rf "$base"; mkdir -p "$base/wt/src" "$base/arms" "$base/shim"
  : > "$base/wt/src/repodata.rs"
  for f in Cargo.toml Cargo.lock build.rs rust-toolchain.toml; do : > "$base/wt/$f"; done
  mkdir -p "$base/wt/recipe" "$base/wt/tests" "$base/wt/examples"
  # cargo always succeeds; python3 refuses ONLY arm A, so the LAST arm is clean
  # and a script that returns "whatever the last arm did" reports success.
  cat > "$base/shim/cargo" <<'EOS'
#!/bin/sh
echo "test result: ok. 0 passed; 0 failed"
exit 0
EOS
  cat > "$base/shim/python3" <<'EOS'
#!/bin/sh
for a in "$@"; do
  case "$a" in *mutA.py) echo "[guard shim] mutation A does not apply" >&2; exit 1;; esac
done
exit 0
EOS
  chmod +x "$base/shim/cargo" "$base/shim/python3"
  env PATH="$base/shim:$PATH" WT="$base/wt" ARMS="$base/arms" bash "$1"
}

# --- A3..A5 the p6af-2h probes: a pixi that produces no lock ----------------
run_probe () {  # $1 script  $2 tag  ($3 probe number, via PROBE_N)
  local base=$W/probe$PROBE_N-$2
  rm -rf "$base"; mkdir -p "$base/artifacts" "$base/shim"
  cat > "$base/shim/pixi" <<'EOS'
#!/bin/sh
case "${1:-}" in
  --version) echo "pixi 0.0.0-guard-shim"; exit 0;;
  config)    echo "[guard shim] no config"; exit 0;;
esac
echo "[guard shim] this pixi resolves nothing and writes no lock" >&2
exit 1
EOS
  chmod +x "$base/shim/pixi"
  env PATH="$base/shim:$PATH" SLURM_JOB_ID="guard$PROBE_N$2" TMPDIR="$base" bash "$1" "$base"
}
run_probe2 () { PROBE_N=2 run_probe "$@"; }
run_probe3 () { PROBE_N=3 run_probe "$@"; }
run_probe4 () { PROBE_N=4 run_probe "$@"; }

# --- A6 probe5: the binary it reads is not readable -------------------------
run_probe5 () {  # $1 script  $2 tag
  bash "$1" "$W/there-is-no-pixi-real-here"
}

if [ "$WHICH" = all ] || [ "$WHICH" = A ]; then
  echo "=== FAMILY A -- the versioned drivers, real file, injected failure"
  run_pair A1-wheel_store_reclaim tools/wheel_store_reclaim.sh      run_reclaim
  run_pair A2-p6ad_negctl         tools/p6ad_negctl.sh              run_negctl
  run_pair A3-p6af2h_probe2       tools/p6af2h_probe2_indexurl.sh   run_probe2
  run_pair A4-p6af2h_probe3       tools/p6af2h_probe3_indexurl.sh   run_probe3
  run_pair A5-p6af2h_probe4       tools/p6af2h_probe4_uvenv.sh      run_probe4
  run_pair A6-p6af2h_probe5       tools/p6af2h_probe5_tls.sh        run_probe5
fi

# ---------------------------------------------------------------- FAMILY B --
# Every live `.sbatch` in the task tree, run for real with `bash` shimmed.
WRAPPERS="
mergeB18/gate.sbatch
mergeB17/gate.sbatch
l3-work/l3.sbatch
c18-1-work/c181_dryrun.sbatch
c18-1-work/c181_negctl.sbatch
c18-1-work/c181_twoarm.sbatch
c18-1-work/c18_1_gate.sbatch
p6ad4-phase1/p6ad4.sbatch
p6ad4-work/mut.sbatch
p6ad4-work/gate.sbatch
p6ad4-work/guard.sbatch
harnessfix1/guards.sbatch
harnessfix1/baseline.sbatch
harnessfix1/verify.sbatch
harnessfix1/verify2.sbatch
"

mk_bash_shim () {
  mkdir -p "$W/bshim"
  cat > "$W/bshim/bash" <<'EOS'
#!/bin/sh
# The drift check is stubbed CLEAN so the wrapper REACHES its payload -- several
# of these wrappers refuse early on a dirty drift and that refusal would make
# the broken and the fixed shape agree, which is a vacuous fixture.
for a in "$@"; do
  case "$a" in
    *harness_drift_check.sh) echo "### [guard shim] drift check stubbed CLEAN"; exit 0;;
  esac
done
echo "### [guard shim] payload stubbed FAILING: $*"
exit 7
EOS
  chmod +x "$W/bshim/bash"
}

run_wrapper () {  # $1 = absolute wrapper path
  env PATH="$W/bshim:$PATH" SLURM_JOB_ID=999999 SLURM_JOB_NAME=driver-exit-guard \
      /bin/bash "$1"
}

if [ "$WHICH" = all ] || [ "$WHICH" = B ]; then
  echo "=== FAMILY B -- every live .sbatch wrapper, payload stubbed at 7"
  mk_bash_shim
  # the pinned pre-fix epilogue: the two lines every one of these files ended on
  PRE=$W/prefix_wrapper.sbatch
  {
    echo '#!/bin/bash'
    echo 'set -u'
    echo "bash $STUB"
    echo 'echo "### GATE_EXIT=$?"'
  } > "$PRE"
  run_wrapper "$PRE" >"$W/B0.log" 2>&1; s=$?
  [ "$s" -eq 0 ] && ok "B0 the pinned PRE-FIX wrapper epilogue still swallows: rc=0" \
                 || no "B0 the pinned PRE-FIX epilogue reported $s, not 0 -- every arm below is vacuous"
  for w in $WRAPPERS; do
    f=$TASK/$w
    [ -f "$f" ] || { no "B $w: no such file"; continue; }
    run_wrapper "$f" >"$W/$(echo "$w" | tr / _).log" 2>&1; s=$?
    [ "$s" -ne 0 ] && ok "B $w re-raises: rc=$s" \
                   || no "B $w swallowed a failing payload: rc=0"
  done
fi

# ---------------------------------------------------------------- FAMILY C --
# The drivers that were already right, driven to a REAL refusal.
if [ "$WHICH" = all ] || [ "$WHICH" = C ]; then
  echo "=== FAMILY C -- the already-correct drivers, locked at a real refusal"

  # C1/C2: the phase templates' own leftover-token self-check, which is the
  # first thing either of them runs. The copy is the real file plus ONE token,
  # so the refusal is the file's own and it fires before anything is created.
  for pair in "C1 phaseN_relock.sh" "C2 phaseN_cert.sh"; do
    set -- $pair
    tag=$1; base=$2
    src=$REPO/harness/phase_template/$base
    [ -f "$src" ] || { no "$tag: no such file $src"; continue; }
    cp -f "$src" "$W/$base"
    echo '# b1-phase -- driver_exit_guard: a leftover token, on purpose' >> "$W/$base"
    ( cd "$W" && bash "$W/$base" ) >"$W/$tag.log" 2>&1; s=$?
    [ "$s" -ne 0 ] && ok "$tag $base carries its leftover-token refusal to the exit code: rc=$s" \
                   || no "$tag $base printed a refusal and exited 0"
  done

  # C3: the cleanup gate, given a root whose job id it cannot derive.
  bash "$REPO/harness/phase_template/cleanup_gated.sh" "$W/a-root-with-no-jobid-token" \
      >"$W/C3.log" 2>&1; s=$?
  [ "$s" -ne 0 ] && ok "C3 cleanup_gated.sh carries its REFUSE to the exit code: rc=$s" \
                 || no "C3 cleanup_gated.sh refused and exited 0"

  # C4: the drift check, given a commit that is not one.
  bash "$REPO/harness/tools/harness_drift_check.sh" not-a-commit-at-all \
      >"$W/C4.log" 2>&1; s=$?
  [ "$s" -ne 0 ] && ok "C4 harness_drift_check.sh carries its FATAL to the exit code: rc=$s" \
                 || no "C4 harness_drift_check.sh printed FATAL and exited 0"

  # C5: the build gate, given a worktree that does not exist.
  ( WT=$W/no-such-worktree D=$W EXPECT_PASS=1 EXPECT_IGNORED=0 \
    bash "$REPO/harness/tools/gate_build.sh" ) >"$W/C5.log" 2>&1; s=$?
  [ "$s" -ne 0 ] && ok "C5 gate_build.sh carries its STOP to the exit code: rc=$s" \
                 || no "C5 gate_build.sh stopped and exited 0"
fi

echo "### driver_exit_guard: pass=$pass fail=$fail  $(date -Is)"
[ "$fail" -eq 0 ] && echo "### DRIVER EXIT GUARD GREEN -- every driver's own failure reaches its exit code" \
                  || echo "### DRIVER EXIT GUARD RED -- read the FAIL lines above"
# This guard is itself a driver. It re-raises.
[ "$fail" -eq 0 ] || exit 1
exit 0
