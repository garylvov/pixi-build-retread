#!/usr/bin/env bash
# proof_smoke_guard.sh -- THE GUARD FOR proof_smoke.sh AND multiarm_preamble.sh.
# PROOF-SMOKE-1.  Law 3: a guard that cannot fail is a defect, so every arm
# below either runs the real thing against a real binary or runs it against a
# STUB built to fail in one named way, and the mutation arm proves that a driver
# WITHOUT the preamble lets a dead backend straight through.
#
#   usage: bash proof_smoke_guard.sh <job root>
#
#   <job root> is the lane's own work dir -- the one holding HARNESS_COMMIT.
#   Everything this guard creates lives under a SHORT scratch root it owns and
#   is removed on the way out.
#
#   PREDICTED: pass=14 fail=0  (state this in the sbatch before submitting)
#
# ── THE FOURTEEN CHECKS ────────────────────────────────────────────────────────
#   A1  the known-good binsnap reaches the frontend            REACHED_FRONTEND
#   A2  ... and proof_smoke.sh exits 0
#   B1  a stub that prints a panic and exits 1                 BACKEND_DIED
#   B2  ... and proof_smoke.sh exits 1
#   B3  ... and the stub's own panic line is QUOTED in the output
#   C1  a stub that sleeps forever                             TIMEOUT
#   C2  ... and proof_smoke.sh exits 2
#   C3  ... and the TIMEOUT is caused by the BACKEND half: that stub DOES score a
#       frontend row (the source-free `cpu` environment), so frontend_rows>=1
#       and backend_work_rows=0 is what holds the verdict -- the non-vacuity
#       control for requiring both halves.
#   D1  the length rule REFUSES a root long enough to compose > 256
#   D2  ... and PASSES the short root (the non-vacuity control: without D2,
#       "it refused" and "it always refuses" are the same picture)
#   E1  the preamble, unmutated, against the dying stub: NON-ZERO and a
#       `### SMOKE BACKEND_DIED` row
#   E2  the MUTATION -- the preamble with step 6 (the smoke) deleted, which is
#       every multi-arm driver written before this landing: it returns ZERO and
#       prints ZERO VERDICT rows and SAYS `smoke_ran=0` against the SAME dead
#       binary.  That is the measurement behind "a green gate is not a working
#       backend": without the smoke the job proceeds straight to its arms.
#   E3  the pin: at the PRE-FIX harness commit named below, neither
#       proof_smoke.sh nor multiarm_preamble.sh exists, so no driver at or
#       before it could have run a smoke.
#   F1  an unsatisfiable REQUIRED_UV is SETUP_FAILED rc=3 naming both versions --
#       the guard for the pin that arm A of 5993691 proved was missing.
set -uo pipefail

# THE MUTATION ARM'S COMMIT CONSTANT -- the harness tip immediately BEFORE this
# landing.  Pinned, never "HEAD~1": a relative ref moves under the next commit.
PSG_PREFIX_COMMIT=9923653ff5f73cfdf6b83209b670db911b5ede2f

JOB_ROOT=${1:?usage: proof_smoke_guard.sh <job root>}
T=${PSG_TASK:-/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11}
REPO=${PSG_REPO:-/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools}
MANIFEST=${PSG_MANIFEST:-/oscar/data/stellex/glvov/imprint-data/pixi.toml}
GOOD=${PSG_GOOD_BINSNAP:-$T/binsnaps/integration-569b0ac}
J=${SLURM_JOB_ID:-$$}
# SHORT, because the length rule is one of the things under test and a guard
# that cannot stage its own control is not a guard.
SCR=${PSG_SCRATCH:-/oscar/data/stellex/glvov/retread/psg$J}
OUT=$JOB_ROOT/artifacts
mkdir -p "$SCR" "$OUT" || { echo "### PSG FATAL cannot create $SCR / $OUT"; exit 3; }

pass=0; fail=0
chk () {  # chk <name> <condition-rc> <what was wanted> <what was seen>
  if [ "$2" = 0 ]; then pass=$((pass+1)); printf '### PSG PASS %-4s %s\n' "$1" "$3"
  else fail=$((fail+1)); printf '### PSG FAIL %-4s want: %s | got: %s\n' "$1" "$3" "$4"; fi
}

echo "### PSG proof_smoke_guard.sh  $(date -Is)  host=$(hostname -s) job=$J"
echo "### PSG scratch=$SCR  job_root=$JOB_ROOT"
echo "### PSG good binsnap=$GOOD"
echo "### PSG PREDICTED pass=14 fail=0"

# ---- the stubs --------------------------------------------------------------
# They live at a path ENDING `/pixi-build-retread` because the shim readback
# matches that exact shape -- a stub at any other basename would be refused by
# the shim check and never reach the verdict under test.
mkdir -p "$SCR/stubdie" "$SCR/stubsleep"
cat > "$SCR/stubdie/pixi-build-retread" <<'STUB'
#!/usr/bin/env bash
# bench: hermetic_provision   <- the witness string, so this stub can stand in
# for a FIX binary in the preamble arms below.
echo "thread 'main' panicked at src/psg_stub.rs:1:1:" >&2
echo "PROOF-SMOKE-GUARD stub backend refuses to serve" >&2
exit 1
STUB
cat > "$SCR/stubsleep/pixi-build-retread" <<'STUB'
#!/usr/bin/env bash
# a backend that never answers the handshake: the TIMEOUT arm.
exec sleep 100000
STUB
chmod +x "$SCR/stubdie/pixi-build-retread" "$SCR/stubsleep/pixi-build-retread"

# ONE staged workspace and ONE cache for every arm: staging is the expensive
# half and the question every arm asks is about the BINARY, not the tree.  Arm A
# pays the cold repodata fetch; the rest do not.
export SMOKE_WS=$SCR/w
export SMOKE_CACHE=$SCR/c
export SMOKE_ROOT=$SCR

# ---- A: the known-good binary ----------------------------------------------
echo ""; echo "########## PSG ARM A -- known good $GOOD ##########"
AOUT=$OUT/psg-$J-A.out
SMOKE_WALL=${PSG_WALL_A:-900} bash "$T/tools/proof_smoke.sh" "$GOOD" "$MANIFEST" "$JOB_ROOT" >"$AOUT" 2>&1
arc=$?
tail -25 "$AOUT"
grep -q '^### SMOKE REACHED_FRONTEND ' "$AOUT"; chk A1 $? "arm A prints '### SMOKE REACHED_FRONTEND'" "$(grep -m1 '^### SMOKE [A-Z_]* binary=' "$AOUT" || echo '<no verdict row>')"
[ "$arc" = 0 ]; chk A2 $? "arm A rc=0" "rc=$arc"

# ---- B: a stub that panics and exits 1 -------------------------------------
echo ""; echo "########## PSG ARM B -- stub that panics and exits 1 ##########"
BOUT=$OUT/psg-$J-B.out
SMOKE_WALL=${PSG_WALL_B:-120} bash "$T/tools/proof_smoke.sh" "$SCR/stubdie" "$MANIFEST" "$JOB_ROOT" >"$BOUT" 2>&1
brc=$?
tail -25 "$BOUT"
grep -q '^### SMOKE BACKEND_DIED ' "$BOUT"; chk B1 $? "arm B prints '### SMOKE BACKEND_DIED'" "$(grep -m1 '^### SMOKE [A-Z_]* binary=' "$BOUT" || echo '<no verdict row>')"
[ "$brc" = 1 ]; chk B2 $? "arm B rc=1" "rc=$brc"
grep -q 'PROOF-SMOKE-GUARD stub backend refuses to serve' "$BOUT"; chk B3 $? "arm B quotes the stub's own panic line" "$(grep -m1 '### SMOKE   ' "$BOUT" || echo '<nothing quoted>')"

# ---- C: a stub that sleeps --------------------------------------------------
echo ""; echo "########## PSG ARM C -- stub that sleeps (TIMEOUT) ##########"
COUT=$OUT/psg-$J-C.out
SMOKE_WALL=${PSG_WALL_C:-15} bash "$T/tools/proof_smoke.sh" "$SCR/stubsleep" "$MANIFEST" "$JOB_ROOT" >"$COUT" 2>&1
crc=$?
tail -25 "$COUT"
grep -q '^### SMOKE TIMEOUT ' "$COUT"; chk C1 $? "arm C prints '### SMOKE TIMEOUT'" "$(grep -m1 '^### SMOKE [A-Z_]* binary=' "$COUT" || echo '<no verdict row>')"
[ "$crc" = 2 ]; chk C2 $? "arm C rc=2" "rc=$crc"
# C3 IS WHY C IS A REAL CONTROL AND NOT AN ACCIDENT.  This stub DOES score a
# frontend row -- 5994177 measured it at four seconds, `resolve_pypi{group=cpu
# platform=linux-64-base}`, an environment with no source at all -- and it is
# the MISSING BACKEND HALF that holds the verdict at TIMEOUT.  Without this
# check, "C timed out" and "C never got anywhere" are the same picture.
CROW=$(grep -m1 'lock ended verdict=' "$COUT")
{ echo "$CROW" | grep -qE 'frontend_rows=[1-9]' && echo "$CROW" | grep -q 'backend_work_rows=0'; }
chk C3 $? "arm C's TIMEOUT is caused by the BACKEND half: frontend_rows>=1 AND backend_work_rows=0" "${CROW:-<no lock-ended row>}"

# ---- D: the length rule, RED and GREEN -------------------------------------
echo ""; echo "########## PSG ARM D -- the composed-prefix rule ##########"
LONGROOT=$SCR/$(printf 'p%.0s' $(seq 1 40))/$(printf 'q%.0s' $(seq 1 40))
DOUT=$OUT/psg-$J-D.out
bash -c 'PROOF_SMOKE_LIB=1 . "$1"; smoke_prefix_budget "$2" "long root"' _ "$T/tools/proof_smoke.sh" "$LONGROOT" >"$DOUT" 2>&1
drc=$?
bash -c 'PROOF_SMOKE_LIB=1 . "$1"; smoke_prefix_budget "$2" "short root"' _ "$T/tools/proof_smoke.sh" "$SCR/c/x" >>"$DOUT" 2>&1
drc2=$?
cat "$DOUT"
[ "$drc" != 0 ]; chk D1 $? "the rule REFUSES a root composing past 256 (len $(( ${#LONGROOT} + 100 + 92 )) )" "rc=$drc"
[ "$drc2" = 0 ]; chk D2 $? "the rule PASSES the short root -- the non-vacuity control" "rc=$drc2"

# ---- E: the preamble, and the mutation that removes its smoke --------------
echo ""; echo "########## PSG ARM E -- preamble vs preamble-without-the-smoke ##########"
cat > "$SCR/run_preamble.sh" <<'RUNNER'
#!/usr/bin/env bash
# $1 = preamble path, $2 = task dir, $3 = job root, $4 = cache root, $5 = binary
set -uo pipefail
source "$1"
MULTIARM_TAG=PSG
MULTIARM_JOB=${SLURM_JOB_ID:-$$}
MULTIARM_TASK=$2
MULTIARM_JOB_ROOT=$3
MULTIARM_CACHE_ROOT=$4
MULTIARM_ARMS=1
MULTIARM_WITNESS="bench: hermetic_provision"
MULTIARM_BINARIES=( "FIX=$5" )
MULTIARM_SMOKE_MANIFEST=$6
MULTIARM_SMOKE_WALL=${PSG_WALL_E:-120}
multiarm_preamble
rc=$?
echo "### RUNNER preamble rc=$rc"
exit $rc
RUNNER
chmod +x "$SCR/run_preamble.sh"

EOUT=$OUT/psg-$J-E-base.out
bash "$SCR/run_preamble.sh" "$T/tools/multiarm_preamble.sh" "$T" "$JOB_ROOT" "$SCR/mc" "$SCR/stubdie" "$MANIFEST" >"$EOUT" 2>&1
erc=$?
tail -20 "$EOUT"
{ [ "$erc" != 0 ] && grep -q '^### SMOKE BACKEND_DIED ' "$EOUT"; }
chk E1 $? "the unmutated preamble REFUSES the dying stub, non-zero, with a BACKEND_DIED row" "rc=$erc smoke_rows=$(grep -c '^### SMOKE ' "$EOUT")"

# THE MUTATION: delete step 6.  This is not a synthetic edit -- it is exactly
# the shape of every multi-arm driver written before this landing, which is what
# E3 pins.  A mutation that does not mutate proves nothing, so the substitution
# is COUNTED and the arm refuses if it changed no line.
sed 's/^  multiarm_smoke_all        || rc=\$?$/  : # MUTATION: step 6 deleted/' \
    "$T/tools/multiarm_preamble.sh" > "$SCR/multiarm_preamble.MUT.sh"
mutn=$(diff "$T/tools/multiarm_preamble.sh" "$SCR/multiarm_preamble.MUT.sh" | grep -c '^< ')
echo "### PSG mutation changed $mutn line(s) (must be exactly 1)"
if [ "$mutn" -ne 1 ]; then
  fail=$((fail+1)); echo "### PSG FAIL E2   the mutation changed $mutn lines, want 1 -- a mutation that does not mutate proves nothing"
else
  EMOUT=$OUT/psg-$J-E-mut.out
  bash "$SCR/run_preamble.sh" "$SCR/multiarm_preamble.MUT.sh" "$T" "$JOB_ROOT" "$SCR/mcm" "$SCR/stubdie" "$MANIFEST" >"$EMOUT" 2>&1
  emrc=$?
  tail -12 "$EMOUT"
  # THE READER IS THE VERDICT ROW, NOT ANY `### SMOKE` LINE.  5993691 E2 failed
  # on `smoke_rows=1` and the row it counted was `### SMOKE PREFIX BUDGET`, which
  # the preamble's step 4 prints out of the same library -- a check that counts
  # the wrong rows is a check that reports the wrong thing.
  smrows=$(grep -c '^### SMOKE [A-Z_]* binary=' "$EMOUT")
  { [ "$emrc" = 0 ] && [ "$smrows" -eq 0 ] && grep -q 'smoke_ran=0' "$EMOUT"; }
  chk E2 $? "the preamble WITHOUT the smoke lets the dead backend through: rc=0, ZERO verdict rows, and it SAYS smoke_ran=0" "rc=$emrc verdict_rows=$smrows smoke_ran_row=$(grep -c 'smoke_ran=0' "$EMOUT")"
fi

git -C "$REPO" cat-file -e "$PSG_PREFIX_COMMIT:harness/tools/proof_smoke.sh" 2>/dev/null; a=$?
git -C "$REPO" cat-file -e "$PSG_PREFIX_COMMIT:harness/tools/multiarm_preamble.sh" 2>/dev/null; b=$?
{ [ "$a" != 0 ] && [ "$b" != 0 ]; }
chk E3 $? "the pre-fix commit $PSG_PREFIX_COMMIT carries NEITHER file, so no driver at or before it could have smoked" "proof_smoke rc=$a multiarm_preamble rc=$b"

# ---- F: the uv preflight refusal -------------------------------------------
# 5993691 arm A reported BACKEND_DIED for the KNOWN-GOOD binsnap because the
# ambient uv is 0.11.29 and retread's `uv_closure::REQUIRED_UV` is 0.12.5: the
# backend printed `preflight: uv version mismatch` twenty times and the smoke
# blamed the binary.  The tool now pins the uv AND checks the version, and this
# arm is the guard for that check -- an impossible required version must be
# SETUP_FAILED (rc 3), never a verdict about the binary.
echo ""; echo "########## PSG ARM F -- the uv preflight refusal ##########"
FOUT=$OUT/psg-$J-F.out
SMOKE_REQUIRED_UV=99.99.99 SMOKE_WALL=60 bash "$T/tools/proof_smoke.sh" "$SCR/stubdie" "$MANIFEST" "$JOB_ROOT" >"$FOUT" 2>&1
frc=$?
tail -6 "$FOUT"
{ [ "$frc" = 3 ] && grep -q '^### SMOKE SETUP_FAILED ' "$FOUT" && grep -q 'retread.s preflight wants 99.99.99' "$FOUT"; }
chk F1 $? "an unsatisfiable REQUIRED_UV is SETUP_FAILED rc=3, naming both versions -- not a verdict about the binary" "rc=$frc $(grep -m1 '^### SMOKE [A-Z_]* binary=' "$FOUT" || echo '<no verdict row>')"

# ---- the scratch goes FIRST, and the verdict is the LAST thing on the page --
# ORDER MATTERS HERE AND IT IS NOT STYLE.  In `psmoke-guards` 5993691 the
# removal ran for FIVE MINUTES (a `cp -al` workspace of 44k entries over NFS)
# and the three rows after it -- including `### PSG FINAL` -- never reached the
# log, while the wrapper printed `### LANE_EXIT=0` over a `fail=4` summary: the
# HARNESS-EXIT-1 picture exactly.  So the long operation happens BEFORE the
# summary, and the verdict rows are the last thing written.
#
# AND NO `chmod -R` ON THE SCRATCH.  `$SCR/w` is a `cp -al` clone of the shared
# stage mirror, so a recursive chmod there changes the mode of the MIRROR'S OWN
# INODES -- p6x rule (2), "a hardlink clone is not isolation", read from the
# permissions side.  Nothing in this guard's scratch is sealed; `rm -rf` alone
# is correct, and its stderr is counted rather than thrown away (ORDER-1-1).
rmerr=$( rm -rf "$SCR" 2>&1 | wc -l )
[ "$rmerr" -eq 0 ] || fail=$((fail+1))

# ---- out --------------------------------------------------------------------
echo ""
echo "### PSG scratch removed errors=$rmerr root=$SCR"
echo "### PSG per-arm verdicts:"
for f in "$AOUT" "$BOUT" "$COUT" "$FOUT"; do
  printf '###   %-28s %s\n' "$(basename "$f")" "$(grep -m1 '^### SMOKE [A-Z_]* binary=' "$f" 2>/dev/null || echo '<none>')"
done
echo "### PSG SUMMARY pass=$pass fail=$fail (predicted pass=14 fail=0)"
echo "### PSG FINAL pass=$pass fail=$fail"
if [ "$fail" -eq 0 ]; then exit 0; fi
exit 1
