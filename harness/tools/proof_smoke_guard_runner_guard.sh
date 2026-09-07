#!/usr/bin/env bash
# proof_smoke_guard_runner_guard.sh -- THE GUARD FOR THE WAY RUNNERS CALL
# tools/proof_smoke_guard.sh.  HARNESS-CONSOL-13 item 3 (from HARNESS-CONSOL-8
# item 5 and the HARNESS-CONSOL-12 note).
#
# THE DEFECT. proof_smoke_guard.sh takes a MANDATORY `<job root>` -- its arms
# write their logs under `<job root>/artifacts` -- and it took it through
# `JOB_ROOT=${1:?usage: ...}`.  A runner that listed the guard and called it bare
# therefore got bash's own stderr diagnostic and an exit BEFORE the guard's first
# `### PSG` line, so the runner's log carried ZERO `### PSG` rows.  Every reader
# of that log greps `### PSG`, and zero rows reads exactly like "this guard was
# not in the list" -- which is how it stayed unrun.  A refusal that reaches
# nobody is a defect (law 9), so the refusal is now a ROW.
#
# THE SECOND HALF. Even called correctly, arms A-H run a REAL lock against the
# LIVE shared stage mirror and take its try-lock, so a runner could not include
# the guard without risking a collision with a running `*-proof` job.  Those arms
# are now opt-in (`PSG_HEAVY=1`) and stand down with a `### PSG SKIPPED heavy`
# row while a proof job is RUNNING.  A runner can now list the guard
# unconditionally: it gets the fixture arms every time, the heavy arms never by
# accident.
#
# THE ARMS:
#   R1  a BARE call prints `### PSG REFUSED reason=no-job-root` and exits
#       non-zero, and `grep -c '### PSG'` on that output is NOT 0
#   R2  with a job root and PSG_HEAVY unset: the `### PSG SKIPPED heavy` row is
#       printed, the FIXTURE arms actually RAN (pass rows > 0), and rc is 0
#   R3  the older PSG_FIXTURE_ONLY flag still skips, and names ITS OWN reason --
#       a runner pinned to the old flag is not silently re-armed
#   R4  MUTATION: the refusal block cut back to `${1:?...}` -- the bare call then
#       prints ZERO `### PSG` rows, which is the tree before this landing and is
#       why R1 can fail
#   R5  THE CALL-SITE SCAN: every line in the harness tree that INVOKES
#       proof_smoke_guard.sh must pass an argument.  The count of call sites is
#       PRINTED, because at this landing it is ZERO (every harness mention is
#       prose) and an arm whose subject does not exist must say so rather than
#       pass quietly.
#
#   usage: bash proof_smoke_guard_runner_guard.sh [<scratch job root>]
#          rc 0 all arms pass, rc 1 otherwise.  FIXTURE-ONLY: it never sets
#          PSG_HEAVY, so no live mirror and no lock is touched.
set -uo pipefail
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PSG="$SELF_DIR/proof_smoke_guard.sh"
HARNESS_ROOT="$(cd -- "$SELF_DIR/.." && pwd)"
[ -f "$PSG" ] || { echo "GUARD FATAL: no proof_smoke_guard.sh at $PSG"; exit 2; }

W=$(mktemp -d "${TMPDIR:-/tmp}/psg-runner-guard.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
pass=0; fail=0
ok()  { echo "PASS  $*"; pass=$((pass+1)); }
bad() { echo "FAIL  $*"; fail=$((fail+1)); }
chk() { if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1: got [$2] want [$3]"; fi; }

JR=${1:-$W/jobroot}
mkdir -p "$JR"
# the guard's own scratch stays inside OUR temp dir: nothing under the live
# retread root is created by this guard.
export PSG_SCRATCH="$W/scr"

# ---- R1: the bare call ------------------------------------------------------
r1=$(bash "$PSG" 2>&1); r1rc=$?
chk "R1 a bare call exits non-zero" "$([ "$r1rc" -ne 0 ] && echo yes || echo no)" "yes"
chk "R1 ... and names itself: ### PSG REFUSED reason=no-job-root" \
  "$(echo "$r1" | grep -c '^### PSG REFUSED reason=no-job-root$')" "1"
R1ROWS=$(echo "$r1" | grep -c '### PSG')
chk "R1 ... and grep -c '### PSG' is NOT 0 on a refusal (it was, and that is the defect)" \
  "$([ "$R1ROWS" -gt 0 ] && echo yes || echo no)" "yes"
echo "### PSGRG R1 measured rc=$r1rc psg_rows=$R1ROWS"

# ---- R2: a job root, heavy NOT armed ----------------------------------------
r2=$(bash "$PSG" "$JR" 2>&1); r2rc=$?
chk "R2 with a job root and PSG_HEAVY unset the run is rc 0" "$r2rc" "0"
chk "R2 the heavy arms stand down with a ROW that names the reason" \
  "$(echo "$r2" | grep -c '^### PSG SKIPPED heavy reason=not-armed')" "1"
R2PASS=$(echo "$r2" | grep -c '^### PSG PASS ')
chk "R2 NON-VACUITY: the FIXTURE arms really ran (pass rows > 0), so 'skipped' is not 'did nothing'" \
  "$([ "$R2PASS" -gt 0 ] && echo yes || echo no)" "yes"
chk "R2 no heavy arm banner appears" \
  "$(echo "$r2" | grep -cE '^########## PSG ARM (A|B|C|D|E|F|G|H|N|V) ')" "0"
echo "### PSGRG R2 measured rc=$r2rc pass_rows=$R2PASS"

# ---- R3: the older flag still works and names its own reason ----------------
r3=$(PSG_FIXTURE_ONLY=1 bash "$PSG" "$JR" 2>&1); r3rc=$?
chk "R3 PSG_FIXTURE_ONLY=1 is rc 0" "$r3rc" "0"
chk "R3 ... and the skip row names THAT reason, not the new one" \
  "$(echo "$r3" | grep -c '^### PSG SKIPPED heavy reason=fixture-only')" "1"

# ---- R4: THE MUTATION -------------------------------------------------------
# The refusal block cut back to the bash-diagnostic form.  Without this arm R1
# could be passing on a guard that never had the block.
MUT=$W/proof_smoke_guard_bare.sh
awk '
  index($0, "if [ \"$#\" -lt 1 ] || [ -z \"${1:-}\" ]; then") == 1 { skip = 1; next }
  skip && $0 == "fi" { skip = 0; print "JOB_ROOT=${1:?usage: proof_smoke_guard.sh <job root>}"; next }
  skip { next }
  $0 == "JOB_ROOT=$1" { next }
  { print }
' "$PSG" > "$MUT"
if cmp -s "$PSG" "$MUT" || ! grep -q '{1:?usage' "$MUT"; then
  bad "R4 the mutation did not apply -- the refusal block was not found, so R1 is vacuous"
else
  m=$(bash "$MUT" 2>&1); mrc=$?
  MROWS=$(echo "$m" | grep -c '### PSG')
  chk "R4 MUTATION: the bash-diagnostic form prints ZERO '### PSG' rows on a bare call" "$MROWS" "0"
  chk "R4 MUTATION: ... and still exits non-zero, so the rc alone never told the runners apart" \
    "$([ "$mrc" -ne 0 ] && echo yes || echo no)" "yes"
  echo "### PSGRG R4 measured mutant rc=$mrc psg_rows=$MROWS"
fi

# ---- R5: the call-site scan -------------------------------------------------
# A CALL is `bash .../proof_smoke_guard.sh <something>` or the script invoked
# directly -- never a `#` comment, which is what every harness mention is today.
# The pattern is built so it cannot match this guard's own scanning line (law
# 14): this guard's own file is dropped by name, and so is the GUARD ITSELF --
# measured, job 6022734: the first cut counted two "call sites" that were the
# guard's own `echo "### PSG   usage: proof_smoke_guard.sh <job root>"` and its
# own banner line. A file does not call itself, and a scan that reads its
# subject's prose as its subject's callers is the same class of miss as arm H4's
# in proof_smoke_guard.sh.
SCAN=$W/callsites.txt
grep -rn 'proof_smoke_guard\.sh' "$HARNESS_ROOT" --include='*.sh' --include='*.sbatch' 2>/dev/null \
  | grep -v 'proof_smoke_guard_runner_guard\.sh' \
  | grep -v '/proof_smoke_guard\.sh:' \
  | awk -F: '{ line = $0; sub(/^[^:]*:[^:]*:/, "", line)
               if (line ~ /^[[:space:]]*#/) next
               print }' > "$SCAN"
NCALL=$(wc -l < "$SCAN")
echo "### PSGRG R5 harness call sites of the guard (non-comment lines): $NCALL"
[ "$NCALL" -gt 0 ] && sed 's/^/### PSGRG   /' "$SCAN"
BARE=$(awk '{ if ($0 !~ /proof_smoke_guard\.sh[[:space:]]+[^[:space:]]/) print }' "$SCAN" | wc -l)
chk "R5 every harness call site passes a job root (bare call sites)" "$BARE" "0"
if [ "$NCALL" -eq 0 ]; then
  ok "R5 STATED PLAINLY: at this commit NO harness file invokes the guard -- every mention in tools/env_seed.sh, tools/proof_smoke.sh and phase_template/phaseN_relock.sh is PROSE. The scan is here so the first real call site is checked on the day it lands, and R1/R2 are what make including it safe."
fi

echo "### PSGRG GUARD $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
