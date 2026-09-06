#!/usr/bin/env bash
# mutation_arms_guard.sh -- the fixture that proves `mutation_arms.sh` puts arm
# scratch UNDER THE JOB ROOT, removes it per arm, and REFUSES a RAM-backed
# scratch root.  L3-1b-1a-2.
#
# It needs no compiler: the fixture redefines `mut_arm_command`, which is the
# one seam the template has for exactly this.  So it runs in the guard set
# beside the shell guards rather than needing a 96 G cargo job.
#
# THE CHECKS, and each one is a claim `mutation_arms.sh` makes:
#   1. an arm's working directory is under the DECLARED scratch root, `<root>/<arm>`
#   2. it is NOT under `/tmp` or `$TMPDIR`
#   3. arm 1's directory is GONE by the time arm 2 starts -- the accumulation
#      that preceded job 5954481's OOM
#   4. no arm directory is left under the scratch root at `mut_done`
#   5. NEGATIVE CONTROL: a RAM-backed `JOB_ROOT` is REFUSED rc 4 and nothing is
#      created under it
#   6. NEGATIVE CONTROL: an unset `JOB_ROOT` is REFUSED rc 3
#   7. NON-VACUITY: the path used for check 5 really is a RAM filesystem, and
#      the path used for checks 1-4 really is not.  Without this the negative
#      control passes on any machine where /dev/shm happens to be a disk.
#   8. DET-1-2: TWO MATRICES WITH DIFFERENT `A_DIR` BUILD IN DIFFERENT SCRATCH
#      ROOTS, and each one's arm directory is under ITS OWN root.  Plus the
#      default (`A_DIR` unset) is still `$JOB_ROOT/mut`, so no existing lane
#      moves -- AND a `MUT_SCRATCH` inherited from a previous `mut_init`'s own
#      export is not mistaken for a caller declaration, which is the same defect
#      one level up and is what the first run of this fix actually did.
#   9. DET-1-2 MUTATION, pinned to the pre-fix commit $PREFIX: the template AS IT
#      WAS derives `MUT_SCRATCH="$JOB_ROOT/mut"` and IGNORES `A_DIR`, so the
#      second matrix builds in the first's live scratch -- the collision that
#      voided DET-1's runs 1 and 2.  Without this arm, check 8 could be a
#      property of the fixture.
#  10. DET-1-2: a scratch root held by a LIVE matrix is REFUSED rc 6 and names
#      the holder; a STALE lock (owner gone) is taken over, not a wedge.
#  11. DET-1-4: a RED BASE arm ends the matrix NON-ZERO and still prints the
#      `### MUT DONE bad=` footer -- measured failure det1-mut 5981304 printed
#      `BASE FAILED` then `MUT_EXIT=0` and Slurm said COMPLETED 0:0.
#  12. DET-1-4 MUTATION, pinned to $PREFIX: on the pre-fix template the same
#      fixture exits ZERO with no footer, so check 11 can fail.
#  13. DET-1-FIX: a mutation that matches ZERO lines is refused rc 99 BEFORE the
#      suite is run, and the refusal NAMES the pattern that found nothing --
#      the address separately from the substitution's LHS, so a reader can see
#      which half is wrong without diffing by hand.
#  14. DET-1-FIX MUTATION, pinned to $PREFIX: the pre-fix template refuses the
#      same arm rc 99 and names NOTHING, which is the silence det1-mut3 5983139
#      actually printed for arm M4.
#
#   rc 0  all of them hold
#   rc 1  a check failed (the row says which)
#   rc 2  the fixture could not be set up
#
# THE MUTATIONS ARE PINNED TO A COMMIT CONSTANT, NEVER `HEAD`: an arm that reads
# `HEAD:<the file it guards>` starts asserting the fix against itself the moment
# the fix is committed.
set -uo pipefail

# HARNESS-SEAM-1's tip, the last commit before DET-1-2/DET-1-4: at it,
# `mut_init` pins MUT_SCRATCH to "$JOB_ROOT/mut" and run_arm's GREEN branch is a
# bare `return 1`.
PREFIX=${MUT_GUARD_PREFIX:-873263ff429af36fb8be1259f681597f99533fda}
GREPO=${HARNESS_REPO:-/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools}

SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${1:-$SELF_DIR/mutation_arms.sh}"
[ -f "$TEMPLATE" ] || { echo "### GUARD FATAL: no template at $TEMPLATE"; exit 2; }

BASE="${MUT_GUARD_BASE:-/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/sr1-work/guardfix}"
RAM_ROOT="${MUT_GUARD_RAM_ROOT:-/dev/shm}"
rm -rf "$BASE"; mkdir -p "$BASE" || { echo "### GUARD FATAL: cannot create $BASE"; exit 2; }

bad=0
fail() { echo "### GUARD FAILED: $*"; bad=$((bad+1)); }
ok()   { echo "### GUARD ok: $*"; }

# shellcheck source=/dev/null
source "$TEMPLATE"

# ── 7. NON-VACUITY FIRST, so a green run cannot be a green machine ───────────
disk_t="$(mut_fstype "$BASE")"
ram_t="$(mut_fstype "$RAM_ROOT")"
echo "### GUARD fstypes: BASE=$BASE -> $disk_t ; RAM_ROOT=$RAM_ROOT -> $ram_t"
if mut_is_ram_fs "$BASE"; then
  fail "the POSITIVE fixture root $BASE is itself $disk_t (RAM) -- checks 1-4 would be testing the wrong thing"
else
  ok "the positive fixture root is on $disk_t, not RAM"
fi
if mut_is_ram_fs "$RAM_ROOT"; then
  ok "the negative control root $RAM_ROOT is $ram_t, so check 5 is not vacuous"
else
  fail "the negative control root $RAM_ROOT is $ram_t, NOT RAM -- check 5 would pass for the wrong reason. Set MUT_GUARD_RAM_ROOT to a tmpfs."
fi

# ── the fake worktree the arms copy from ─────────────────────────────────────
FAKE_WT="$BASE/wt"
mkdir -p "$FAKE_WT/src"
printf 'const MARKER: &str = "original";\n' > "$FAKE_WT/src/lib.rs"

# The seam: every arm records WHERE it ran and prints a `test result:` line, so
# the template's own "it did not run" refusal stays live.
PROBE="$BASE/where.txt"
: > "$PROBE"
mut_arm_command() {
  local dir="$1"
  printf '%s\t%s\n' "${MUT_ARM_NAME:-?}" "$dir" >> "$PROBE"
  # arm A2 records whether A1's directory still exists AT THE MOMENT A2 RUNS --
  # check 3, and it cannot be answered after the fact.
  if [ "${MUT_ARM_NAME:-}" = A2 ]; then
    printf 'A1_dir_present_during_A2\t%s\n' \
      "$([ -d "$MUT_SCRATCH/A1" ] && echo yes || echo no)" >> "$PROBE"
  fi
  if grep -q original "$dir/src/lib.rs"; then
    echo "test result: ok. 1 passed; 0 failed; 0 ignored"
    return 0
  fi
  echo "test result: FAILED. 0 passed; 1 failed; 0 ignored"
  return 101
}

# ── 6. NEGATIVE CONTROL: no JOB_ROOT ─────────────────────────────────────────
( unset JOB_ROOT; WT="$FAKE_WT"; mut_init >/dev/null 2>&1 )
rc=$?
[ "$rc" -eq 3 ] && ok "an unset JOB_ROOT is refused rc=3" \
                || fail "an unset JOB_ROOT gave rc=$rc, expected 3"

# ── 5. NEGATIVE CONTROL: a RAM-backed JOB_ROOT ───────────────────────────────
RAM_JOB_ROOT="$RAM_ROOT/mutguard-$$-$(date +%s)"
rm -rf "$RAM_JOB_ROOT"
out=$( JOB_ROOT="$RAM_JOB_ROOT"; WT="$FAKE_WT"; mut_init 2>&1 ); rc=$?
[ "$rc" -eq 4 ] && ok "a RAM-backed JOB_ROOT is refused rc=4" \
                || fail "a RAM-backed JOB_ROOT gave rc=$rc, expected 4; output: $out"
if [ -d "$RAM_JOB_ROOT/mut" ]; then
  fail "the refusal still created $RAM_JOB_ROOT/mut -- a refusal that writes is not a refusal"
else
  ok "the refusal created no scratch under the RAM root"
fi
rm -rf "$RAM_JOB_ROOT"

# ── 1-4. THE POSITIVE FIXTURE ────────────────────────────────────────────────
# PLAIN ASSIGNMENTS, not a `VAR=x mut_init` prefix: bash discards a prefix
# assignment when the function returns, and `run_arm` needs `$WT` afterwards.
JOB_ROOT="$BASE/job"
WT="$FAKE_WT"
A_DIR="$BASE/job/arts"
MUT_JOBS=1
mut_init || { echo "### GUARD FATAL: mut_init refused the disk-backed fixture"; exit 2; }
GUARDS=(dummy)
MUT_ARM_NAME=A1 run_arm A1 lib.rs "" GREEN \
  || fail "arm A1 (unmutated, declared GREEN) did not come out green"
MUT_ARM_NAME=A2 run_arm A2 lib.rs 's|original|mutated|' RED \
  || fail "arm A2 (mutated, declared RED) did not come out red"

# ── 5. THE PREFIX-ASSIGNMENT CALL SHAPE, WHICH IS THE ONE THIS FILE'S OWN
#      USAGE COMMENT SHOWS, MUST REACH THE ARMS. bash discards a prefix
#      assignment when a function returns, so `WT=... mut_init` used to leave
#      `run_arm` reading an unbound `$WT` and every lane copying the usage line
#      died in its FIRST arm under `set -u` (measured: sr2-mut 5966771, rc 1,
#      zero arms run). `mut_init` now records the value itself. The control
#      below is what makes this non-vacuous: `WT` is UNSET in the caller after
#      the prefix call, so an arm that passes proves the recording, not a
#      leftover variable.
( set -uo pipefail
  unset WT MUT_WT
  JOB_ROOT="$BASE/job-prefix" WT="$FAKE_WT" A_DIR="$BASE/job-prefix/arts" MUT_JOBS=1 \
    mut_init >/dev/null 2>&1 || { echo "### GUARD FATAL: prefix-form mut_init refused"; exit 2; }
  [ -z "${WT:-}" ] || echo "### note: this bash DID keep the prefix assignment; the arm below still proves the recording"
  GUARDS=(dummy)
  MUT_ARM_NAME=P1 run_arm P1 lib.rs "" GREEN >/dev/null 2>&1
) || fail "the prefix-assignment call shape (this file's documented usage) cannot run an arm"
if grep -q "^P1" "$PROBE"; then
  ok "the prefix-assignment call shape reaches the arms (arm P1 actually ran)"
else
  fail "arm P1 never ran -- the prefix-assignment call shape does not reach the arms"
fi
rm -rf "$BASE/job-prefix"

while IFS=$'\t' read -r who where; do
  case "$who" in
    A1|A2)
      # DET-1-2: the scratch root is the DECLARED one ($A_DIR here), not
      # $JOB_ROOT/mut. Both are under the job root, which is check 1's point.
      if [ "$where" = "$BASE/job/arts/$who" ]; then
        ok "arm $who ran in $where -- under the DECLARED scratch root, inside the job root (check 1)"
      else
        fail "arm $who ran in $where, not $BASE/job/arts/$who (check 1)"
      fi
      shadowed=0
      for shadow in "${TMPDIR:-/tmp}" /tmp; do
        if mut_path_inside "$where" "$shadow"; then
          fail "arm $who scratch $where is under $shadow (check 2)"
          shadowed=1
        fi
      done
      [ "$shadowed" -eq 0 ] && ok "arm $who scratch is under neither \$TMPDIR (${TMPDIR:-<unset>}) nor /tmp (check 2)"
      ;;
    A1_dir_present_during_A2)
      if [ "$where" = no ]; then
        ok "A1's directory was already gone when A2 started (check 3)"
      else
        fail "A1's directory still existed during A2 -- arms accumulate, which is the 5954481 shape (check 3)"
      fi
      ;;
  esac
done < "$PROBE"
grep -q '^A1_dir_present_during_A2' "$PROBE" \
  || fail "the A1-presence probe never ran -- check 3 is vacuous"

mut_done 0 || fail "mut_done refused a clean run"
left_dirs=$(find "$BASE/job/arts" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
if [ "$left_dirs" -eq 0 ]; then
  ok "no arm directory survived mut_done (check 4)"
else
  fail "$left_dirs arm director(ies) survived mut_done under $BASE/job/arts (check 4)"
fi
[ -f "$BASE/job/arts/.mut_lock" ] && fail "mut_done left the lock file behind" \
                                  || ok "mut_done removed the lock file"

# ── 8. DET-1-2: TWO MATRICES, TWO A_DIRs, TWO SCRATCH ROOTS ──────────────────
# The whole point: a lane that wants to keep run 1's logs points run 2 at mut2/
# and gets a SEPARATE scratch. Run them one after another in subshells, each
# recording where its arm actually built.
two_probe=$BASE/two.txt; : > "$two_probe"
run_matrix () {                # $1 = A_DIR (or "-" for the default) ; $2 = tag
  ( set -uo pipefail
    # shellcheck source=/dev/null
    source "$TEMPLATE"
    MTAG=$2
    mut_arm_command () {
      local dir="$1"
      printf '%s\t%s\t%s\n' "$MTAG" "$dir" "$MUT_SCRATCH" >> "$two_probe"
      echo "test result: ok. 1 passed; 0 failed; 0 ignored"; return 0
    }
    # MUT_SCRATCH is deliberately NOT unset: it is exported by the mut_init the
    # positive fixture already ran, and the template must treat its OWN derived
    # value as absent rather than as a declaration. If it does not, every
    # matrix here silently reuses the first one's root -- which is what the
    # first run of this fix actually did.
    JOB_ROOT="$BASE/two"; WT="$FAKE_WT"; MUT_JOBS=1
    if [ "$1" = "-" ]; then unset A_DIR; else A_DIR="$1"; fi
    mut_init >/dev/null 2>&1 || { echo "### MATRIX $2 mut_init refused rc=$?"; exit 9; }
    GUARDS=(dummy)
    run_arm X lib.rs "" GREEN >/dev/null 2>&1
    mut_done 0 >/dev/null 2>&1
  )
}
run_matrix "$BASE/two/mut"  M1 || fail "8: matrix M1 (A_DIR=.../mut) could not run"
run_matrix "$BASE/two/mut2" M2 || fail "8: matrix M2 (A_DIR=.../mut2) could not run"
run_matrix "-"              M3 || fail "8: matrix M3 (A_DIR unset) could not run"
s1=$(awk -F'\t' '$1=="M1"{print $3}' "$two_probe" | head -1)
s2=$(awk -F'\t' '$1=="M2"{print $3}' "$two_probe" | head -1)
s3=$(awk -F'\t' '$1=="M3"{print $3}' "$two_probe" | head -1)
d1=$(awk -F'\t' '$1=="M1"{print $2}' "$two_probe" | head -1)
d2=$(awk -F'\t' '$1=="M2"{print $2}' "$two_probe" | head -1)
echo "### GUARD two-matrix roots: M1=$s1 M2=$s2 M3(default)=$s3"
if [ -n "$s1" ] && [ -n "$s2" ] && [ "$s1" != "$s2" ] \
   && [ "$s1" = "$BASE/two/mut" ] && [ "$s2" = "$BASE/two/mut2" ]; then
  ok "8: two matrices with different A_DIR build in DIFFERENT scratch roots (DET-1-2)"
else
  fail "8: M1 root='$s1' M2 root='$s2' -- A_DIR did not move the scratch"
fi
if [ "$d1" = "$BASE/two/mut/X" ] && [ "$d2" = "$BASE/two/mut2/X" ]; then
  ok "8: each matrix's arm directory is under ITS OWN root ($d1 / $d2)"
else
  fail "8: arm dirs were '$d1' and '$d2'"
fi
[ "$s3" = "$BASE/two/mut" ] \
  && ok "8: with A_DIR unset the default is still \$JOB_ROOT/mut -- no existing lane moves" \
  || fail "8: the default scratch root became '$s3', not $BASE/two/mut"
# The three matrices above ran with MUT_SCRATCH still EXPORTED from the positive
# fixture's mut_init ($BASE/job/arts). None of them used it, which is the claim.
if [ "$s1" != "$BASE/job/arts" ] && [ "$s2" != "$BASE/job/arts" ] && [ "$s3" != "$BASE/job/arts" ]; then
  ok "8: an inherited MUT_SCRATCH that mut_init ITSELF derived is not mistaken for a declaration"
else
  fail "8: a matrix reused the previous mut_init's exported MUT_SCRATCH ($BASE/job/arts)"
fi

# ── 9. DET-1-2 MUTATION: the pre-fix template ignores A_DIR ──────────────────
PRE=$BASE/mutation_arms_prefix.sh
if git -C "$GREPO" cat-file blob "$PREFIX:harness/tools/mutation_arms.sh" > "$PRE" 2>/dev/null; then
  if grep -q 'MUT_SCRATCH="\$JOB_ROOT/mut"' "$PRE" && ! grep -q 'DET-1-2' "$PRE"; then
    ok "9: the pinned pre-fix template $PREFIX really is the pre-fix one"
  else
    fail "9: $PREFIX does not look like the pre-fix template -- the pin is wrong"
  fi
  pre_probe=$BASE/pre.txt; : > "$pre_probe"
  ( set -uo pipefail
    # shellcheck source=/dev/null
    source "$PRE"
    mut_arm_command () {
      printf '%s\t%s\n' "$MUT_SCRATCH" "$1" >> "$pre_probe"
      echo "test result: ok. 1 passed; 0 failed; 0 ignored"; return 0
    }
    JOB_ROOT="$BASE/pre"; WT="$FAKE_WT"; A_DIR="$BASE/pre/mut2"; MUT_JOBS=1
    mut_init >/dev/null 2>&1 || exit 9
    GUARDS=(dummy)
    run_arm X lib.rs "" GREEN >/dev/null 2>&1
  ) >/dev/null 2>&1
  pres=$(awk -F'\t' '{print $1}' "$pre_probe" | head -1)
  if [ "$pres" = "$BASE/pre/mut" ]; then
    ok "9: MUTATION -- the pre-fix template put A_DIR=.../mut2's arms in .../mut anyway (the DET-1 collision)"
  else
    fail "9: the pre-fix template built in '$pres' -- expected $BASE/pre/mut; the mutation did not reproduce"
  fi
else
  fail "9: could not read $PREFIX:harness/tools/mutation_arms.sh -- MUTATION ARM DID NOT RUN"
fi

# ── 10. DET-1-2: the lock ────────────────────────────────────────────────────
LOCKROOT=$BASE/lockjob/mut
mkdir -p "$LOCKROOT"
printf 'pid=%s\nhost=%s\njob=-\nwhen=now\n' "$$" "$(hostname)" > "$LOCKROOT/.mut_lock"
mkdir -p "$LOCKROOT/A1"                                   # arm dirs from the "live" matrix
out=$( set -uo pipefail
       JOB_ROOT="$BASE/lockjob"; WT="$FAKE_WT"; A_DIR="$LOCKROOT"; MUT_JOBS=1
       unset MUT_SCRATCH; mut_init 2>&1 ); rc=$?
[ "$rc" -eq 6 ] && ok "10: a scratch root held by a LIVE matrix is refused rc=6" \
                || fail "10: a live-locked scratch root gave rc=$rc, expected 6; output: $out"
printf '%s' "$out" | grep -q "pid=$$" \
  && ok "10: the refusal NAMES the holder's pid" || fail "10: the refusal does not name the holder"
printf '%s' "$out" | grep -q 'A_DIR=' \
  && ok "10: the refusal names the actuator (point A_DIR elsewhere)" \
  || fail "10: the refusal is a dead notice with no actuator"
# a STALE lock -- a pid that cannot be alive -- is taken over, not a wedge
printf 'pid=%s\nhost=%s\njob=-\nwhen=then\n' 4194304 "$(hostname)" > "$LOCKROOT/.mut_lock"
out=$( set -uo pipefail
       JOB_ROOT="$BASE/lockjob"; WT="$FAKE_WT"; A_DIR="$LOCKROOT"; MUT_JOBS=1
       unset MUT_SCRATCH; mut_init 2>&1 ); rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -q 'STALE'; then
  ok "10: a STALE lock is announced and taken over -- a crashed job does not wedge the next run"
else
  fail "10: a stale lock gave rc=$rc; output: $out"
fi

# ── 11/12. DET-1-4: a RED BASE arm is terminal, non-zero, and prints the footer
# The arm must run in a CHILD BASH, because the fix `exit`s -- which is the whole
# point: it does not leave the decision to the lane script that swallowed it.
mkbasefix () {   # $1 = template path, $2 = job root
  cat > "$BASE/basered.sh" <<EOSH
set -uo pipefail
source "$1"
mut_arm_command () { echo "test result: FAILED. 0 passed; 1 failed; 0 ignored"; return 101; }
JOB_ROOT="$2"; WT="$FAKE_WT"; A_DIR="$2/mut"; MUT_JOBS=1
mut_init >/dev/null 2>&1 || exit 9
GUARDS=(dummy)
bad=0
run_arm BASE lib.rs "" GREEN || bad=\$((bad+1))
echo "### the lane script SWALLOWED the return, exactly as det1_mutations.sh did"
exit 0
EOSH
  bash "$BASE/basered.sh" > "$BASE/basered.log" 2>&1; echo $?
}
rcB=$(mkbasefix "$TEMPLATE" "$BASE/basered-fix")
if [ "$rcB" -ne 0 ]; then
  ok "11: a RED BASE arm exits NON-ZERO (rc=$rcB) even though the lane script swallowed the return"
else
  fail "11: a RED BASE arm still exited 0 -- 5981304's COMPLETED 0:0 all over again"
fi
grep -q '### MUT DONE bad=1 reason=base-red' "$BASE/basered.log" \
  && ok "11: and it prints the MUT DONE footer through the same mut_done path" \
  || { fail "11: no 'MUT DONE bad=1 reason=base-red' footer"; sed 's/^/      /' "$BASE/basered.log"; }
grep -q 'the lane script SWALLOWED' "$BASE/basered.log" \
  && fail "11: execution continued past the red base -- the matrix is not ended" \
  || ok "11: nothing after the red base ran -- the matrix is ended, not reported as colours"
if [ -f "$PRE" ]; then
  rcP=$(mkbasefix "$PRE" "$BASE/basered-pre")
  if [ "$rcP" -eq 0 ] && ! grep -q 'MUT DONE' "$BASE/basered.log"; then
    ok "12: MUTATION -- the pre-fix template exits 0 with NO footer on the same fixture (5981304's shape)"
  else
    fail "12: the pre-fix template gave rc=$rcP with a footer -- check 11 proves nothing"
  fi
else
  fail "12: no pre-fix template to mutate against -- MUTATION ARM DID NOT RUN"
fi

# ── 13/14. DET-1-FIX: A ZERO-LINE MUTATION IS REFUSED **BEFORE THE SUITE RUNS**
#      AND THE REFUSAL NAMES THE PATTERN THAT FOUND NOTHING.
# Measured: det1-mut 5981742 and det1-mut3 5983139 both printed
# `### M4 FATAL: the mutation changed NOTHING` and not one word more, on two
# nodes ninety minutes apart, and the cause -- a range anchor written for a
# multi-line signature against a real ONE-LINE one -- had to be found by hand.
zero_probe=$BASE/zero.txt
run_zero_arm () {            # $1 = template ; $2 = job root ; prints the arm's output
  ( set -uo pipefail
    # shellcheck source=/dev/null
    source "$1"
    mut_arm_command () { printf 'RAN\t%s\n' "$1" >> "$zero_probe"; echo "test result: ok. 1 passed; 0 failed; 0 ignored"; return 0; }
    JOB_ROOT="$2"; WT="$FAKE_WT"; A_DIR="$2/mut"; MUT_JOBS=1
    unset MUT_SCRATCH
    mut_init >/dev/null 2>&1 || { echo "MUT_INIT_REFUSED"; exit 9; }
    GUARDS=(dummy)
    # The address names a function this fixture does not contain, exactly as
    # DET-1's M4 named a signature spelling `source_build.rs` does not contain.
    run_arm Z1 lib.rs '/^fn nosuchfn($/,/^}$/ s|^const MARKER.*$|const MARKER: \&str = "mutated";|' RED
    echo "RUN_ARM_RC=$?"
  ) 2>&1
}
: > "$zero_probe"
zout=$(run_zero_arm "$TEMPLATE" "$BASE/zeroarm-fix")
if printf '%s' "$zout" | grep -q 'RUN_ARM_RC=99'; then
  ok "13: a mutation that matches nothing is refused rc=99"
else
  fail "13: a zero-line mutation did not return 99; output: $zout"
fi
if grep -q '^RAN' "$zero_probe"; then
  fail "13: the arm's SUITE RAN anyway -- the refusal must come BEFORE the test command"
else
  ok "13: the refusal came BEFORE the suite ran (no arm command was invoked)"
fi
if printf '%s' "$zout" | grep -q 'ADDRESS /\^fn nosuchfn($/ matches 0 lines'; then
  ok "13: the refusal NAMES the pattern that failed to find its line"
else
  fail "13: the refusal does not name the absent pattern; output: $zout"
fi
if printf '%s' "$zout" | grep -q 'LHS |\^const MARKER.*| matches 1 line'; then
  ok "13: and it separates the pattern that DID match, so the reader knows which half is wrong"
else
  fail "13: the refusal does not report the substitution's LHS count; output: $zout"
fi
if [ -f "$PRE" ]; then
  : > "$zero_probe"
  zpre=$(run_zero_arm "$PRE" "$BASE/zeroarm-pre")
  if printf '%s' "$zpre" | grep -q 'RUN_ARM_RC=99' \
     && ! printf '%s' "$zpre" | grep -q 'ADDRESS /'; then
    ok "14: MUTATION -- the pre-fix template refuses rc=99 and names NOTHING, which is the silence 5983139 printed"
  else
    fail "14: the pre-fix template already named the pattern -- check 13 proves nothing; output: $zpre"
  fi
else
  fail "14: no pre-fix template to mutate against -- MUTATION ARM DID NOT RUN"
fi

echo "### MUTATION ARMS GUARD bad=$bad"
rm -rf "$BASE"
[ "$bad" -eq 0 ] || exit 1
exit 0
