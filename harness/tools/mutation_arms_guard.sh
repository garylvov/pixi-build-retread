#!/usr/bin/env bash
# mutation_arms_guard.sh -- the fixture that proves `mutation_arms.sh` puts arm
# scratch UNDER THE JOB ROOT, removes it per arm, and REFUSES a RAM-backed
# scratch root.  L3-1b-1a-2.
#
# It needs no compiler: the fixture redefines `mut_arm_command`, which is the
# one seam the template has for exactly this.  So it runs in the guard set
# beside the shell guards rather than needing a 96 G cargo job.
#
# SEVEN CHECKS, and each one is a claim `mutation_arms.sh` makes:
#   1. an arm's working directory is under `$JOB_ROOT/mut/<arm>`
#   2. it is NOT under `/tmp` or `$TMPDIR`
#   3. arm 1's directory is GONE by the time arm 2 starts -- the accumulation
#      that preceded job 5954481's OOM
#   4. nothing is left under the scratch root at `mut_done`
#   5. NEGATIVE CONTROL: a RAM-backed `JOB_ROOT` is REFUSED rc 4 and nothing is
#      created under it
#   6. NEGATIVE CONTROL: an unset `JOB_ROOT` is REFUSED rc 3
#   7. NON-VACUITY: the path used for check 5 really is a RAM filesystem, and
#      the path used for checks 1-4 really is not.  Without this the negative
#      control passes on any machine where /dev/shm happens to be a disk.
#
#   rc 0  all seven hold
#   rc 1  a check failed (the row says which)
#   rc 2  the fixture could not be set up
set -uo pipefail

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

while IFS=$'\t' read -r who where; do
  case "$who" in
    A1|A2)
      if [ "$where" = "$BASE/job/mut/$who" ]; then
        ok "arm $who ran in $where -- under the job root (check 1)"
      else
        fail "arm $who ran in $where, not $BASE/job/mut/$who (check 1)"
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
if [ -d "$BASE/job/mut" ]; then
  fail "the scratch root survived mut_done (check 4)"
else
  ok "the scratch root is gone after mut_done (check 4)"
fi

echo "### MUTATION ARMS GUARD bad=$bad"
rm -rf "$BASE"
[ "$bad" -eq 0 ] || exit 1
exit 0
