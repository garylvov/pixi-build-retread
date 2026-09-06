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
# An EXPLORATORY run is one whose numbers are NOT a verdict -- a first sweep
# with an empty baseline, a mutation arm, a narrowed DEG_ONLY. It says so on
# its FIRST line so a later reader cannot mistake it for the authoritative
# run, and it still prints the tally and still re-raises.
[ "${DEG_EXPLORATORY:-0}" = 1 ] && echo "### EXPLORATORY RUN -- these numbers are NOT a verdict"
echo "### driver_exit_guard  repo=$REPO  task=$TASK  prefix_commit=$PREFIX_COMMIT"
echo "### work=$W  family=$WHICH  host=$(hostname)  $(date -Is)"

pass=0; fail=0
# ---- THE TALLY IS NOT OPTIONAL ------------------------------------------
# HARNESS-EXIT-3, from its OWN defect: two discovery runs printed FAIL lines
# and then ended with neither the `pass=/fail=` tally nor the GREEN/RED line,
# and Slurm read COMPLETED 0:0 -- the exact shape this guard exists to refuse,
# on the guard itself. The cause was measured, not guessed: THE SCRIPT WAS
# EDITED WHILE IT WAS RUNNING. bash reads a script incrementally by BYTE
# OFFSET, so an edit that shifts the file makes a running shell resume at a
# stale offset -- here, onto the trailing `exit 0`. Two consequences, both
# permanent:
#   * a run executes a SNAPSHOT of this file (see the lane wrapper, which
#     copies the three driver_exit_*.sh into a run-scoped directory), and
#   * the tally prints from an EXIT TRAP, so a truncated, killed or
#     resumed-at-a-stale-offset run still says what it had counted and still
#     re-raises. A guard that can end silently is not a guard.
deg_tally_printed=0
deg_tally () {
  [ "$deg_tally_printed" -eq 0 ] || return 0
  deg_tally_printed=1
  echo "### driver_exit_guard: pass=$pass fail=$fail  $(date -Is)"
  [ "$fail" -eq 0 ] && echo "### DRIVER EXIT GUARD GREEN -- every driver's own failure reaches its exit code" \
                    || echo "### DRIVER EXIT GUARD RED -- read the FAIL lines above"
}
deg_on_exit () {
  local rc=$?
  deg_tally
  [ "$rc" -eq 0 ] && [ "$fail" -ne 0 ] && rc=1
  rm -rf "$W"
  exit "$rc"
}
trap deg_on_exit EXIT
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
# HARNESS-EXIT-3 REPLACED THIS FAMILY WHOLE. It was a HAND-TYPED LIST of
# seventeen wrappers driven by one `bash` shim. Both halves were defects:
#
#   the list -- STORE-REAP-2 wrote a NEW ad-hoc wrapper (`sr2-mut`, job
#   5966771) that swallowed its driver's rc and reported green, because a
#   wrapper nobody typed into the list is invisible to the guard that exists to
#   find it. The list is gone; the set is DISCOVERED.
#
#   HARNESS-SEAM-1 then removed the THIRD list, one layer down. The static
#   by-path check decided coverage from a hand-typed set of VARIABLE NAMES
#   (`DEG_SCAN_COVERED_VARS`), so a wrapper that called its payload variable
#   anything else was refused, and STORE-REAP-3's fix for that was to RENAME
#   ITS VARIABLE to one on the list. Coverage is now DERIVED: the value is
#   resolved statically from the wrapper's own text and the seam's own exported
#   environment, classified by the BASENAME of the resolved value against the
#   stubs the seam actually built, and the stub is BOUND OVER THE RESOLVED PATH
#   so the by-path invocation is intercepted instead of merely permitted. A
#   value that cannot be resolved statically is refused, naming the variable.
#
#   the shim -- it stubbed `bash` and nothing else, so a listed wrapper whose
#   payload is `cargo ...` or a retread binary was EXECUTED FOR REAL on the
#   guard's host. Measured: adding sr2-work/check.sbatch and census.sbatch ran
#   a real `cargo check` and a real `store-reap` dry run on the LOGIN NODE, and
#   then scored both as swallowers, which they are not. The shim is gone; every
#   payload goes through driver_exit_payload_shim.sh, and a payload the seam
#   does not cover is a REFUSAL, never an execution.
#
# THE RATCHET. Discovery over a tree this size finds swallowers that predate
# this lane and belong to lanes whose jobs are queued or running (a `.sbatch`
# is snapshotted at submit, so a RUNNING job is safe to edit but a PENDING one
# is not -- editing it is a silent change to a job that has not started). A
# guard that goes red on all of them is a guard nobody can keep green, and a
# permanently red guard is the same defect as no guard. So:
#
#   * driver_exit_baseline.txt names the wrappers KNOWN to swallow, and the
#     wrappers whose payload the seam cannot cover, at the commit that wrote it.
#   * the guard FAILS on a swallower that is NOT in the baseline. A new one is
#     fixed, never baselined.
#   * the baseline may only SHRINK. Its row counts at this commit are pinned in
#     the two constants below, and a baseline larger than the pin is a REFUSAL
#     (rc 4): somebody added a row instead of a fix.
#   * a baselined wrapper that now re-raises is printed as STALE so the next
#     pass deletes the row.
BASELINE=${DEG_BASELINE:-$REPO/harness/tools/driver_exit_baseline.txt}
# pinned at HARNESS-EXIT-3. LOWER THESE when you shrink the baseline; never raise.
BASELINE_MAX_SWALLOW=2
BASELINE_MAX_UNCOVERED=2

DEG_SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$DEG_SELF_DIR/driver_exit_payload_shim.sh"
. "$DEG_SELF_DIR/driver_exit_scan.sh"

# deg_run_wrapper <abs wrapper> <record dir> -- run one wrapper inside the seam.
# CONTAINMENT: the task tree is a tmpfs inside the sandbox, with only the
# wrapper itself bound back in read-only, so a wrapper's ordinary redirection
# (`find ... > "$A/tree-before.txt"`) cannot overwrite another lane's artifacts.
# This is additive to the payload seam, not a substitute for it.
deg_run_wrapper () {
  local f=$1 rec=$2 bindfile=${3:-} envs=() line binds=() bname bpath
  mkdir -p "$rec/tmp"
  while IFS= read -r line; do envs+=("$line"); done < <(deg_shim_env "$rec")
  # The task tree is a tmpfs so a wrapper's ordinary redirection cannot land on
  # another lane's artifacts. THE PERSISTENT STORES ARE TMPFS FOR A SECOND
  # REASON: several wrappers open with a `find` or a census over a store with
  # millions of entries, and a SAFE passthrough command that scans a real store
  # is how a run becomes a function of the NODE instead of of the wrapper's
  # bytes. Empty inside the sandbox, those scans return instantly. None of this
  # is a substitute for the payload seam; it is containment, and it is additive.
  # DEG_TMPFS_OK is built by deg_probe_tmpfs, because a mount that bwrap cannot
  # make fails the WHOLE sandbox at rc 1 -- which is how this lane briefly
  # scored 101 wrappers as re-raising with ZERO intercepted invocations, caught
  # only by the B0 non-vacuity arm.
  binds=(--dev-bind / / --tmpfs "$TASK" --bind "$W" "$W")
  [ -n "${DEG_TMPFS_OK:-}" ] && binds+=($DEG_TMPFS_OK)
  case "$f" in "$TASK"/*) binds+=(--ro-bind "$f" "$f");; esac
  # HARNESS-SEAM-1: THE BY-PATH HALF OF THE SEAM.
  # A wrapper that runs `"$RETREAD_BIN" store-reap ...` reaches its payload
  # through a PATH, and neither a function nor $PATH nor
  # command_not_found_handle is in that road. The old code exported eight
  # variable NAMES pointed at a stub and hoped the wrapper used one of them --
  # which a plain `RETREAD_BIN=<path>` on the wrapper's own first line simply
  # overwrites. So the seam places the stub WHERE THE WRAPPER WILL LOOK: the
  # scanner resolves the value statically, and each resolved path gets the
  # matching stub bound over it inside the sandbox. Nothing outside the sandbox
  # is touched, and a wrapper whose path the seam CANNOT bind is refused by
  # deg_probe_binds before it is started, never run uncovered.
  if [ -n "$bindfile" ] && [ -s "$bindfile" ]; then
    while IFS=$'\t' read -r bname bpath; do
      [ -n "$bname" ] && [ -n "$bpath" ] || continue
      binds+=(--ro-bind "$DEG_SHIM/$bname" "$bpath")
    done < "$bindfile"
  fi
  command timeout -k 5 "${DEG_WRAPPER_BUDGET_S:-60}" env "${envs[@]}" "$DEG_BWRAP" "${binds[@]}" /bin/bash "$f"
}

# deg_probe_binds <bindfile> -- CAN the seam actually place these stubs?
# A bind bwrap cannot make does not fail that bind, it ABORTS THE WHOLE SANDBOX
# at rc 1 -- the defect that once scored 101 wrappers as re-raising with zero
# intercepted invocations. So coverage is not a claim about a path, it is a
# MEASUREMENT: build the identical sandbox and run /bin/true in it. rc 0 means
# the stubs are placeable and the wrapper may be driven; anything else means the
# seam cannot cover this wrapper and it is refused.
deg_probe_binds () {   # $1 = bindfile ; rc 0 = the seam can place every stub
  local bindfile=$1 binds=() bname bpath
  binds=(--dev-bind / / --tmpfs "$TASK" --bind "$W" "$W")
  [ -n "${DEG_TMPFS_OK:-}" ] && binds+=($DEG_TMPFS_OK)
  while IFS=$'\t' read -r bname bpath; do
    [ -n "$bname" ] && [ -n "$bpath" ] || continue
    binds+=(--ro-bind "$DEG_SHIM/$bname" "$bpath")
  done < "$bindfile"
  "$DEG_BWRAP" "${binds[@]}" /bin/true >/dev/null 2>&1
}

# deg_probe_tmpfs -- keep only the extra tmpfs mounts this host's bwrap can
# actually make. `/users` is not visible under `--dev-bind / /` here (it is a
# separate mount), so asking for a tmpfs beneath it aborts the sandbox; a
# candidate that cannot be mounted is DROPPED and printed, never silently kept.
deg_probe_tmpfs () {
  local cand out=""
  for cand in ${DEG_TMPFS_EXTRA:-/oscar/data/stellex/glvov/caches /oscar/data/stellex/glvov/retread /users/glvov/.cache}; do
    [ -d "$cand" ] || { echo "###   tmpfs candidate $cand: absent, skipped"; continue; }
    if "$DEG_BWRAP" --dev-bind / / --tmpfs "$cand" /bin/true >/dev/null 2>&1; then
      out="$out --tmpfs $cand"; echo "###   tmpfs candidate $cand: MOUNTED"
    else
      echo "###   tmpfs candidate $cand: bwrap cannot mount it here, DROPPED"
    fi
  done
  DEG_TMPFS_OK=$out
}

if [ "$WHICH" = all ] || [ "$WHICH" = B ]; then
  echo "=== FAMILY B -- EVERY .sbatch in the task tree, payload seam, ratcheted"
  DEG_BWRAP=$(command -v bwrap 2>/dev/null); [ -n "$DEG_BWRAP" ] || { echo "GUARD FATAL: no bwrap -- FAMILY B refuses to run a wrapper uncontained"; exit 4; }
  [ -f "$BASELINE" ] || { echo "GUARD FATAL: no baseline at $BASELINE"; exit 4; }
  echo "### containment: the task tree is a tmpfs; probing the extra store mounts"
  deg_probe_tmpfs

  deg_build_shim "$W/seam" >/dev/null || { echo "GUARD FATAL: could not build the payload seam"; exit 4; }
  echo "### payload seam built at $W/seam  inject=$DEG_INJECT"

  # ---- DISCOVERY IS A SNAPSHOT, AND THE SNAPSHOT IS THE UNIT OF AGREEMENT.
  # Watcher-11's finding, and it is the right one: two runs that enumerate a
  # LIVE tree are not comparable, because lanes add and rewrite `.sbatch` files
  # between them, and a bucket that moves for that reason is indistinguishable
  # from a bucket that moves because the guard is flaky. So the set is
  # enumerated ONCE into a list of `md5<TAB>path`, the list's own md5 is
  # printed, and a run may be handed an existing list with DEG_SNAPSHOT.
  # A wrapper whose md5 no longer matches the snapshot is a REFUSAL, not a
  # reclassification: somebody rewrote it while the run was in flight, and no
  # verdict about its bytes is available.
  #
  # maxdepth 3 reaches every one of them; HARNESS-EXIT-3 measured 0/82/98/98/98
  # for depth 1..5 and 98 unbounded, so depth 3 is the floor that is also
  # complete. The unbounded control runs on every SNAPSHOT so a wrapper filed
  # deeper tomorrow is a loud discrepancy, not a silent miss.
  if [ -n "${DEG_SNAPSHOT:-}" ] && [ -f "$DEG_SNAPSHOT" ]; then
    cp -f "$DEG_SNAPSHOT" "$W/snapshot.tsv"
    echo "### FAMILY B SNAPSHOT: reused from $DEG_SNAPSHOT"
  else
    find "$TASK" -maxdepth 3 -name '*.sbatch' -type f 2>/dev/null | sort > "$W/discovered.txt"
    find "$TASK" -name '*.sbatch' -type f 2>/dev/null | sort > "$W/discovered_unbounded.txt"
    nd=$(grep -c . "$W/discovered.txt"); nu=$(grep -c . "$W/discovered_unbounded.txt")
    if [ "$nd" -ne "$nu" ]; then
      no "DISCOVERY depth 3 found $nd but the tree holds $nu -- raise the depth before trusting any row below"
    else
      ok "DISCOVERY depth 3 is complete: $nd = the unbounded count"
    fi
    : > "$W/snapshot.tsv"
    while IFS= read -r f; do
      printf '%s\t%s\n' "$(md5sum "$f" | awk '{print $1}')" "$f" >> "$W/snapshot.tsv"
    done < "$W/discovered.txt"
    [ -n "${DEG_SNAPSHOT_OUT:-}" ] && cp -f "$W/snapshot.tsv" "$DEG_SNAPSHOT_OUT"
  fi
  awk -F'\t' '{print $2}' "$W/snapshot.tsv" > "$W/discovered.txt"
  nd=$(grep -c . "$W/discovered.txt")
  SNAP_MD5=$(md5sum "$W/snapshot.tsv" | awk '{print $1}')
  echo "### FAMILY B SNAPSHOT: $nd wrappers, snapshot md5 $SNAP_MD5 -- two runs on this md5 MUST agree"

  # ---- the pinned pre-fix epilogue: the shape every fixed wrapper used to end
  # on. It is written out as a CONSTANT rather than read from a commit, so it
  # keeps reproducing the defect after every fix lands.
  PRE=$W/prefix_wrapper.sbatch
  {
    echo '#!/bin/bash'
    echo 'set -u'
    echo "bash $STUB"
    echo 'echo "### GATE_EXIT=$?"'
  } > "$PRE"
  deg_run_wrapper "$PRE" "$W/rec-B0" >"$W/B0.log" 2>&1; s=$?
  [ "$s" -eq 0 ] && ok "B0 the pinned PRE-FIX wrapper epilogue still swallows: rc=0" \
                 || no "B0 the pinned PRE-FIX epilogue reported $s, not 0 -- every arm below is vacuous"
  # A VACUOUS INJECTION PRODUCES NO BUCKETS. If the pre-fix epilogue does not
  # swallow, the seam did not reach the payload -- a broken sandbox, a missing
  # stub -- and every row below would be an artefact of that, not a verdict.
  # Measured on this lane: one bad tmpfs candidate aborted bwrap at rc 1 and
  # the run scored 101 wrappers as "re-raise" with ZERO intercepted
  # invocations. That must never reach a log as a bucket count.
  [ "$s" -eq 0 ] || { echo "GUARD REFUSES: the injection is vacuous (B0 rc=$s); no bucket below would mean anything"; exit 4; }

  # ---- the versioned shape lanes are told to copy is an ARM, not a document.
  LW=$REPO/harness/phase_template/lane_wrapper.sbatch
  if [ -f "$LW" ]; then
    deg_run_wrapper "$LW" "$W/rec-LW" >"$W/lane_wrapper.log" 2>&1; s=$?
    [ "$s" -ne 0 ] && ok "B phase_template/lane_wrapper.sbatch (the shape lanes copy) re-raises: rc=$s" \
                   || no "B phase_template/lane_wrapper.sbatch swallowed a failing payload: rc=0"
  else
    no "B phase_template/lane_wrapper.sbatch is missing -- lanes have no versioned shape to copy"
  fi

  # ---- the baseline, and its ratchet
  bl_swallow=$(grep -E '^SWALLOW[[:space:]]' "$BASELINE" | awk '{print $2}' | sort -u)
  bl_uncov=$(grep -E '^UNCOVERED[[:space:]]' "$BASELINE" | awk '{print $2}' | sort -u)
  n_bl_s=$(printf '%s\n' "$bl_swallow" | grep -c .); n_bl_u=$(printf '%s\n' "$bl_uncov" | grep -c .)
  echo "### BASELINE: $n_bl_s swallow rows (pin $BASELINE_MAX_SWALLOW), $n_bl_u uncovered rows (pin $BASELINE_MAX_UNCOVERED)"
  if [ "$n_bl_s" -gt "$BASELINE_MAX_SWALLOW" ] || [ "$n_bl_u" -gt "$BASELINE_MAX_UNCOVERED" ]; then
    echo "GUARD REFUSES: the baseline GREW ($n_bl_s/$n_bl_u against pins $BASELINE_MAX_SWALLOW/$BASELINE_MAX_UNCOVERED)."
    echo "GUARD REFUSES: a new swallower is FIXED, not baselined. rc 4."
    exit 4
  fi

  : > "$W/live_swallow.txt"; : > "$W/live_uncovered.txt"; : > "$W/live_timeout.txt"
  : > "$W/live_nopayload.txt"; live_reraise=0
  while IFS= read -r f; do
    rel=${f#"$TASK"/}
    # DEG_ONLY narrows the sweep to one wrapper. It exists for the mutation
    # arms, which must show ONE named wrapper flipping verdict, and it is never
    # set on a scoring run -- the whole point of this family is that the set is
    # discovered, not chosen.
    [ -z "${DEG_ONLY:-}" ] || case "$rel" in $DEG_ONLY) ;; *) continue;; esac
    # the snapshot is the unit of agreement: if these bytes are not the bytes
    # that were enumerated, there is no verdict to give about them.
    want=$(awk -F'\t' -v p="$f" '$2==p{print $1}' "$W/snapshot.tsv")
    have=$(md5sum "$f" 2>/dev/null | awk '{print $1}')
    if [ "$want" != "$have" ]; then
      no "B $rel CHANGED UNDER THE RUN (snapshot $want, now ${have:-absent}) -- rewritten mid-run, not classified"
      continue
    fi
    tag=$(printf '%s' "$rel" | tr / _)
    rec=$W/rec-$tag; mkdir -p "$rec"
    # STATIC HALF: a command invoked BY PATH cannot be reached by a PATH shim
    # or a function. HARNESS-SEAM-1: coverage is DERIVED, never listed. The
    # scanner resolves each path-position token from the wrapper's own
    # assignments and the seam's own exported environment, and classifies it by
    # the BASENAME of the resolved value against the stubs the seam has
    # actually built. Three outcomes, and two of them are refusals that name
    # what they refused on:
    #   COVERED    -- the seam has a stub for that basename; it is bound over
    #                 the resolved path and the wrapper is driven.
    #   UNCOVERED  -- resolved, but the seam has no stub for that basename
    #                 (a directory, a lane's own tool). Refused.
    #   UNRESOLVED -- the value cannot be known statically. Refused, NAMING THE
    #                 VARIABLE AND THE LINE. Never executed, never passed.
    unc=""; unres=""; : > "$rec/binds.tsv"
    while IFS=$'\t' read -r vk va vb vc; do
      case "$vk" in
        COVERED)    printf '%s\t%s\n' "$va" "$vb" >> "$rec/binds.tsv" ;;
        UNCOVERED)  unc="$unc $vb(basename '$va' is not a command the seam stubs)" ;;
        UNRESOLVED) unres="$unres [$va] at line $vb: $vc;" ;;
      esac
    done < <(deg_scan_path_verdicts "$f")
    if [ -n "$unres" ]; then
      echo "$rel" >> "$W/live_uncovered.txt"
      echo "  REFUSE  B $rel: payload invoked by path through a variable that cannot be resolved statically:$unres (NOT executed)"
      continue
    fi
    if [ -n "$unc" ]; then
      echo "$rel" >> "$W/live_uncovered.txt"
      echo "  REFUSE  B $rel: payload invoked by path, seam does not cover:$unc (NOT executed)"
      continue
    fi
    if [ -s "$rec/binds.tsv" ] && ! deg_probe_binds "$rec/binds.tsv"; then
      echo "$rel" >> "$W/live_uncovered.txt"
      echo "  REFUSE  B $rel: the seam cannot PLACE its stub at $(awk '{printf "%s ", $2}' "$rec/binds.tsv")-- bwrap refuses the bind (NOT executed)"
      continue
    fi
    [ -s "$rec/binds.tsv" ] && echo "  BYPATH  B $rel: $(awk '{printf "%s->%s ", $2, $1}' "$rec/binds.tsv")"
    deg_run_wrapper "$f" "$rec" "$rec/binds.tsv" >"$W/$tag.log" 2>&1; s=$?
    # RUNTIME HALF: a name the seam did not stub reached command_not_found_handle.
    if [ -s "$rec/uncovered.log" ]; then
      echo "$rel" >> "$W/live_uncovered.txt"
      echo "  REFUSE  B $rel: uncovered command $(awk '{print $2}' "$rec/uncovered.log" | sort -u | tr '\n' ' ')(refused at rc=$s, NOT executed)"
      continue
    fi
    # A TIMEOUT IS ITS OWN BUCKET, and it is the one bucket that is not a pure
    # function of the wrapper's bytes -- it depends on the node and the budget.
    # It is therefore never folded into a score: it is counted, listed, and
    # required to be EMPTY on the authoritative run. The determinism fix is
    # upstream of it -- the sandbox puts the real cache roots on tmpfs too, so
    # a wrapper's `find` over a store scans nothing and finishes.
    if [ "$s" -eq 124 ] || [ "$s" -eq 137 ]; then
      echo "$rel" >> "$W/live_timeout.txt"
      echo "  TIMEOUT B $rel: still running after ${DEG_WRAPPER_BUDGET_S:-60}s -- NOT SCORED (rc=$s)"
      continue
    fi
    # THE SCORING RULE. A wrapper is judged ONLY once the seam has actually
    # handed it a FAILING payload. Without that, a 0 says nothing: the sandbox
    # empties the stores, so a census whose loop finds no roots never reaches
    # any payload at all and exits 0 having demonstrated NOTHING. Scoring that
    # as a swallow is how a guard grows false rows -- four of them on this
    # lane's own first scoring run (p1b closure_metadata, char3, sr1 census,
    # sr2 bytes), every one of which re-raises correctly on a real failure.
    # PRECOND rows (the drift check, the harness-commit reader) are not
    # injected failures and do not count.
    inj=0
    [ -f "$rec/argv.log" ] && inj=$(grep -c "^PAYLOAD" "$rec/argv.log")
    if [ "${inj:-0}" -eq 0 ]; then
      echo "$rel" >> "$W/live_nopayload.txt"
      continue
    fi
    if [ "$s" -eq 0 ]; then echo "$rel" >> "$W/live_swallow.txt"
    else live_reraise=$((live_reraise + 1)); fi
  done < "$W/discovered.txt"

  n_live_s=$(grep -c . "$W/live_swallow.txt"); n_live_u=$(grep -c . "$W/live_uncovered.txt")
  n_live_t=$(grep -c . "$W/live_timeout.txt")
  n_live_n=$(grep -c . "$W/live_nopayload.txt")
  echo "### FAMILY B LIVE: $live_reraise re-raise, $n_live_s swallow, $n_live_u refused-uncovered, $n_live_t timed-out, $n_live_n no-payload-reached, of $nd discovered (budget ${DEG_WRAPPER_BUDGET_S:-60}s, snapshot $SNAP_MD5)"
  if [ "$n_live_t" -eq 0 ]; then
    ok "B no wrapper timed out -- every bucket below is a function of the wrapper's bytes"
  else
    no "B $n_live_t wrapper(s) timed out and are UNSCORED: $(tr '\n' ' ' < "$W/live_timeout.txt")-- raise DEG_WRAPPER_BUDGET_S or tmpfs the root they scan; a timing-dependent bucket is not a verdict"
  fi

  if [ "$n_live_n" -gt 0 ]; then
    echo "### NO PAYLOAD REACHED (unscored -- the seam never handed these a failing payload; a 0 from them proves nothing):"
    sed 's/^/###   /' "$W/live_nopayload.txt"
  fi
  # new swallowers -- the only thing that may turn this family red
  comm -23 <(sort -u "$W/live_swallow.txt") <(printf '%s\n' "$bl_swallow" | sort -u) > "$W/new_swallow.txt"
  comm -23 <(sort -u "$W/live_uncovered.txt") <(printf '%s\n' "$bl_uncov" | sort -u) > "$W/new_uncovered.txt"
  if [ -s "$W/new_swallow.txt" ]; then
    while IFS= read -r r; do no "B $r SWALLOWS a failing payload and is NOT in the baseline -- fix it, do not baseline it"; done < "$W/new_swallow.txt"
  else
    ok "B no wrapper outside the baseline swallows its payload ($n_live_s known, $live_reraise re-raise)"
  fi
  if [ -s "$W/new_uncovered.txt" ]; then
    while IFS= read -r r; do no "B $r invokes a payload the seam does not cover and is NOT in the baseline -- widen the seam or fix the wrapper"; done < "$W/new_uncovered.txt"
  else
    ok "B every wrapper outside the baseline was driven through the seam ($n_live_u known-uncovered)"
  fi
  # stale rows: printed, never fatal -- a fix by another lane must not turn this red
  comm -13 <(sort -u "$W/live_swallow.txt") <(printf '%s\n' "$bl_swallow" | sort -u) > "$W/stale.txt"
  if [ -s "$W/stale.txt" ]; then
    echo "### STALE BASELINE ROWS (these re-raise now -- delete them and lower the pin):"
    sed 's/^/###   /' "$W/stale.txt"
  fi

  # ---- the canary. Nothing in $DEG_CANARY nor $CARGO_HOME/bin may ever have
  # run: those stubs stand exactly where the REAL cargo/pixi/binary would be
  # found if the function seam and the PATH both leaked, and each writes a file.
  fired=$(find "$W" -maxdepth 2 -name 'CANARY_FIRED.*' -type f 2>/dev/null | wc -l)
  if [ "$fired" -eq 0 ]; then
    ok "B CANARY silent: no payload command reached a real PATH lookup in $nd wrapper runs"
  else
    no "B CANARY FIRED $fired times -- the payload seam LEAKED: $(find "$W" -maxdepth 2 -name 'CANARY_FIRED.*' -type f 2>/dev/null | tr '\n' ' ')"
  fi
  cat "$W"/rec-*/argv.log > "$W/argv-all.log" 2>/dev/null
  echo "### ARGV RECORDER: $(grep -c . "$W/argv-all.log" 2>/dev/null || echo 0) intercepted invocations, $W/argv-all.log"
  echo "### intercepted payload commands, by name:"
  awk -F'\t' '$1=="PAYLOAD"{print $2}' "$W/argv-all.log" 2>/dev/null | sort | uniq -c | sort -rn | sed 's/^/###   /'
  # ---- evidence. $W is a temp dir the EXIT trap removes, so a run that is
  # supposed to be quotable copies its decisive files out. Law 4: every gate
  # writes its evidence packet.
  if [ -n "${DEG_EVIDENCE_DIR:-}" ]; then
    mkdir -p "$DEG_EVIDENCE_DIR"
    cp -f "$W/snapshot.tsv" "$W/live_swallow.txt" "$W/live_uncovered.txt" \
          "$W/live_timeout.txt" "$W/live_nopayload.txt" "$W/new_swallow.txt" "$W/new_uncovered.txt" \
          "$W/argv-all.log" "$DEG_EVIDENCE_DIR/" 2>/dev/null
    for keep in ${DEG_EVIDENCE_LOGS:-}; do
      cp -f "$W/$(printf '%s' "$keep" | tr / _).log" "$DEG_EVIDENCE_DIR/" 2>/dev/null
      cp -f "$W/rec-$(printf '%s' "$keep" | tr / _)/argv.log" \
            "$DEG_EVIDENCE_DIR/argv.$(printf '%s' "$keep" | tr / _).log" 2>/dev/null
    done
    echo "### EVIDENCE written to $DEG_EVIDENCE_DIR"
  fi
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

deg_tally
[ "$fail" -eq 0 ] || exit 1
exit 0
