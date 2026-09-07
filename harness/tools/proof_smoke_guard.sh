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
#   PREDICTED: pass=39 fail=0 (state this in the sbatch before submitting).
#   PROOF-SMOKE-1-5 added SEVEN: N1-N7.  ARMS A-F STILL READ THE TASK COPY, so
#   A1/A2 remain the unfixed coin until the harness is synced -- read them as a
#   control on what a lane did NOT touch, never as a verdict.
#
#   N1  the NEW rule REFUSES the exact fixture root 6003336 ran and panicked on
#   N2  ... and the SAME row reports what the OLD rule said about it (240/16),
#       which is the non-vacuity control: the verdict INVERTED on one root
#   N3  ... and still PASSES a root with real headroom -- not "always refuses"
#   N4  MUTATION: remove the fast-tmp suffix term and the fixture root PASSES
#   N5  the MEASURED boundary from both sides: entry 167 composes to the pad
#       exactly, entry 168 trips the pad refusal at 256
#   N6  the fixed smoke PRINTS its fast-tmp mode -- the divergence is not silent
#   N7  THREE consecutive REACHED_FRONTEND in ONE job: the red/green/red coin
#       HARNESS-CONSOL-5 measured across 6003336/6003619/6003855 is retired
#
#   STAGE-MIRROR-1 adds EIGHT, S1-S8, and they are FIXTURE-ONLY: no binary and
#   no live mirror, so PSG_FIXTURE_ONLY=1 runs them alone in a cheap CPU job
#   rather than contending with a running relock for the shared mirror.
#   S1-S3  the staged probe-trace / audit / third_party egg-info copies each
#          come out of staging with link count 1 -- their own inodes
#   S4     a pack-directory file the NAME LIST does not carry is broken anyway,
#          by the pypi-packs/*/ sweep: the belt to the name list's braces
#   S5     NON-VACUITY: the wheel payload one level deeper is STILL shared, so
#          the break is targeted and not a disguised full copy
#   S6     writing the staged trace leaves the mirror BYTE-IDENTICAL
#   S7     MUTATION: smoke_break_one made a no-op -> links=2 and the mirror
#          CHANGES, which is what S1 and S6 would otherwise pass without
#   S8     smoke_stage CALLS it -- the function has a production call site
#
# ── THE THIRTY-NINE CHECKS ────────────────────────────────────────────────────
#   A1  the known-good binsnap reaches the frontend            REACHED_FRONTEND
#   A2  ... and proof_smoke.sh exits 0
#   B1  a stub that prints a panic and exits 1                 BACKEND_DIED
#   B2  ... and proof_smoke.sh exits 1
#   B3  ... and the stub's own panic line is QUOTED in the output
#   C1  a stub that sleeps forever                             TIMEOUT
#   C2  ... and proof_smoke.sh exits 2
#   C3  ... and the TIMEOUT is held by the BACKEND half, not by chance:
#       a sleeping backend serves ZERO `conda/outputs`, so backend_work_rows=0
#       is what holds the verdict, and both counters must be reported -- the
#       non-vacuity control for requiring both halves.  (The frontend half is
#       racy and measured so: 1 row in 5994177, 0 in 5994392, same stub.)
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
#   G1  PROOF-SMOKE-1-3: on 6000903's OWN roots (a six-character tag) the arms
#       fit and the SMOKE is not refused either.  Pure path arithmetic, no lock.
#   G2  ... and the same with an EIGHT-character tag.
#   G3  when the ARMS themselves overrun, the job still refuses and the refusal
#       names the ARM budget -- shortening the smoke would fix nothing there.
#   G4  MUTATION, pinned to $PSG_G_OLD (the commit 6000903 ran): on G1's fixture
#       the arms PASS and the smoke's own root is REFUSED, and there is no
#       symmetry row anywhere.  6000903, verbatim, so G1 can fail.
#   H1  PROOF-SMOKE-1-4: two arms of ONE job take the same stage mirror
#       sequentially -- the second ADOPTS the lock and both return 0, because
#       refusing a job its own lock would refuse every multi-arm driver in tree.
#   H2  a mirror held by a DIFFERENT, LIVE job refuses `### SMOKE STAGE BUSY`
#       rc 1 -- loudly, rather than staging beside it and sharing its inodes.
#   H3  ... and the BUSY row names the ACTUATOR: the holder's job id to chain
#       after.  A refusal that reaches nobody is a defect (law 9).
#   H4  ... and it is a REFUSAL, not a panic -- matched on the Rust panic's own
#       SHAPE, never on the word, because the first cut matched the BUSY row's
#       own prose explaining the panic it avoids (run 6003336) and caught its
#       own explanation. A pattern that can match the text it guards is law 14
#       one level up.
#   H5  THE NON-VACUITY CONTROL: a lock whose owner job is NOT in the queue is
#       RECLAIMED, loudly.  Without it a smoke killed mid-stage would wedge
#       every later smoke, which is a worse failure than the one being fixed --
#       and BUSY would be the only answer the lock knows, which is not a lock.
#   H6  THE MUTATION: with the acquire cut out of smoke_stage, the same fixture
#       stages straight past a live holder.  That is 6001840, and without this
#       arm H2 could be passing on the fixture rather than on the lock.
set -uo pipefail

# THE MUTATION ARM'S COMMIT CONSTANT -- the harness tip immediately BEFORE this
# landing.  Pinned, never "HEAD~1": a relative ref moves under the next commit.
PSG_PREFIX_COMMIT=9923653ff5f73cfdf6b83209b670db911b5ede2f

JOB_ROOT=${1:?usage: proof_smoke_guard.sh <job root>}
T=${PSG_TASK:-/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11}
REPO=${PSG_REPO:-/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools}
MANIFEST=${PSG_MANIFEST:-/oscar/data/stellex/glvov/imprint-data/pixi.toml}
GOOD=${PSG_GOOD_BINSNAP:-$T/binsnaps/integration-569b0ac}
# Arm S sources the proof_smoke.sh SITTING BESIDE THIS GUARD, so a fresh
# worktree of a commit tests THAT commit. Arms A-F still exec the task copy at
# $T/tools -- that split is deliberate and is what makes A1/A2 a control on what
# a lane did NOT sync.
PS_UNDER_TEST=${PSG_PROOF_SMOKE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/proof_smoke.sh}
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
echo "### PSG PREDICTED pass=39 fail=0 (24 was stale from PROOF-SMOKE-1-1 and had drifted through three landings; corrected here)"

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
# PROOF-SMOKE-1-4, AND THIS IS THE PRODUCTION CALL SITE FOR THE HOLD FLAG. This
# guard stages ONCE in arm A and then runs B..F against those hardlinks, so the
# mirror's inodes are in use for the guard's WHOLE life, not for arm A's. Arm A's
# smoke must therefore take the stage lock and NOT drop it at its own finish;
# this guard drops it on the way out. Without the flag arm A would release after
# itself and arms B..F would run unprotected -- which is the window 6001840 lost
# arm A in. Harmless against a task copy that predates the lock (the variable is
# simply unread there), which is what arms A-F still read today.
export SMOKE_STAGE_LOCK_HOLD=1
psg_release_stage_lock () {   # drop any lock in the live mirror root owned by THIS job
  local d o n=0
  while IFS= read -r d; do
    o=$(sed -n 's/^job=//p' "$d/owner" 2>/dev/null | head -1)
    [ "$o" = "$J" ] || continue
    rm -rf "$d" 2>/dev/null && n=$((n+1)) && echo "### PSG released stage lock $d (owner=$J)"
  done < <(find "${SMOKE_MIRROR_ROOT:-/oscar/data/stellex/glvov/agrescap/cache/retread/stage-mirror}" \
             -maxdepth 1 -name '.*.smoke-stage-v1.lock' -type d 2>/dev/null)
  echo "### PSG stage locks released by this job: $n"
}
trap 'psg_release_stage_lock' EXIT


# ---- S: STAGE-MIRROR-1 -- the smoke must not write through the shared mirror -
# FIXTURE-ONLY, no binary, no live mirror: the question is whether `cp -al` +
# `smoke_stage_break_links` leaves the workspace's copies on their own inodes,
# and a fixture answers it exactly. The function is SOURCED out of the shipped
# proof_smoke.sh (PROOF_SMOKE_LIB=1), so this arm tests the code that runs.
#
# WHAT IT WOULD HAVE CAUGHT, measured: D141 job 6001140 arm 3 quarantined the
# shared mirror over one file, pypi-packs/pm-newton-pack/retread-probe-trace-
# pm-newton-pack.json, whose mirror copy was rewritten at 00:29:24 inside job
# 6006079's window -- a psb-guard2 smoke whose own log records
# `### SMOKE stage: mirror HIT .../85db7fdbbf51206a0cb57fa0d55e0e74`.
echo ""; echo "########## PSG ARM S -- the smoke's hardlink break (fixture) ##########"
SFX=$(mktemp -d "${TMPDIR:-/tmp}/psg-arm-s.XXXXXX")
(
  set -u
  MIR=$SFX/mirror; WSX=$SFX/ws
  mkdir -p "$MIR/pypi-packs/pm-newton-pack" "$MIR/third_party/pkg.egg-info" "$MIR/pypi-packs/pm-newton-pack/dist"
  printf 'trace-v1\n'  > "$MIR/pypi-packs/pm-newton-pack/retread-probe-trace-pm-newton-pack.json"
  printf 'audit-v1\n'  > "$MIR/pypi-packs/pm-newton-pack/retread-audit-pm-newton-pack.json"
  printf 'sidecar-v1\n'> "$MIR/pypi-packs/pm-newton-pack/some-other-sidecar.txt"
  printf 'wheelbytes\n'> "$MIR/pypi-packs/pm-newton-pack/dist/pkg-1.0-py3-none-any.whl"
  printf 'reqs-v1\n'   > "$MIR/third_party/pkg.egg-info/requires.txt"
  cp -al "$MIR" "$WSX"
  BEFORE=$SFX/mirror.before.tsv; AFTER=$SFX/mirror.after.tsv
  find "$MIR" -mindepth 1 -printf '%y\t%s\t%P\n' | LC_ALL=C sort > "$BEFORE"
  PROOF_SMOKE_LIB=1 . "$PS_UNDER_TEST" || exit 90
  smoke_stage_break_links "$WSX" || exit 91
  # every path the lock can WRITE is now the workspace's own
  h () { stat -c %h "$1"; }
  printf 'S_TRACE=%s\n'   "$(h "$WSX/pypi-packs/pm-newton-pack/retread-probe-trace-pm-newton-pack.json")"
  printf 'S_AUDIT=%s\n'   "$(h "$WSX/pypi-packs/pm-newton-pack/retread-audit-pm-newton-pack.json")"
  printf 'S_SIDECAR=%s\n' "$(h "$WSX/pypi-packs/pm-newton-pack/some-other-sidecar.txt")"
  printf 'S_EGG=%s\n'     "$(h "$WSX/third_party/pkg.egg-info/requires.txt")"
  printf 'S_WHEEL=%s\n'   "$(h "$WSX/pypi-packs/pm-newton-pack/dist/pkg-1.0-py3-none-any.whl")"
  # the write the backend actually performs: truncate-and-write, in place
  printf 'trace-v2-longer\n' > "$WSX/pypi-packs/pm-newton-pack/retread-probe-trace-pm-newton-pack.json"
  find "$MIR" -mindepth 1 -printf '%y\t%s\t%P\n' | LC_ALL=C sort > "$AFTER"
  LC_ALL=C diff -q "$BEFORE" "$AFTER" >/dev/null && printf 'S_MIRROR=IDENTICAL\n' || printf 'S_MIRROR=CHANGED\n'
) > "$SFX/out.txt" 2>&1
SOUT=$(cat "$SFX/out.txt")
sfield () { printf '%s\n' "$SOUT" | sed -n "s/^$1=//p" | head -1; }
[ "$(sfield S_TRACE)" = 1 ]; chk S1 $? "the staged probe-trace json has link count 1 (its own inode)" "links=$(sfield S_TRACE) out=$(printf '%s' "$SOUT" | tr '\n' '|')"
[ "$(sfield S_AUDIT)" = 1 ]; chk S2 $? "the staged audit json has link count 1" "links=$(sfield S_AUDIT)"
[ "$(sfield S_EGG)" = 1 ];   chk S3 $? "the staged third_party egg-info file has link count 1" "links=$(sfield S_EGG)"
# the broader pypi-packs/*/ sweep: a name the literal list does NOT carry
[ "$(sfield S_SIDECAR)" = 1 ]; chk S4 $? "a pack-directory file the NAME LIST does not carry is broken anyway (the pypi-packs/*/ sweep)" "links=$(sfield S_SIDECAR)"
# the NON-VACUITY control: if everything were broken this arm would prove nothing
[ "$(sfield S_WHEEL)" = 2 ]; chk S5 $? "the wheel payload one level DEEPER is STILL shared (links=2) -- the break is targeted, not a full copy" "links=$(sfield S_WHEEL)"
[ "$(sfield S_MIRROR)" = IDENTICAL ]; chk S6 $? "writing the staged trace leaves the mirror BYTE-IDENTICAL" "mirror=$(sfield S_MIRROR)"
# ---- S-mut: a REAL code mutation, not a rearranged fixture ------------------
# `smoke_break_one` is turned into a no-op -- the function still runs, still
# counts, still prints its row, and breaks nothing. S1 and S6 must then go RED.
# Without this arm they would pass on a fixture that never had a hardlink.
PS_MUT=$SFX/proof_smoke.mut.sh
sed 's%^smoke_break_one () {%smoke_break_one () { return 0 ;  # MUTATION: the break is a no-op%' "$PS_UNDER_TEST" > "$PS_MUT"
if cmp -s "$PS_UNDER_TEST" "$PS_MUT"; then
  chk S7 1 "the mutation changes smoke_break_one" "the mutant is byte-identical -- the arm is vacuous"
else
(
  set -u
  MIR=$SFX/m2; WSX=$SFX/w2
  mkdir -p "$MIR/pypi-packs/pm-newton-pack"
  printf 'trace-v1\n' > "$MIR/pypi-packs/pm-newton-pack/retread-probe-trace-pm-newton-pack.json"
  cp -al "$MIR" "$WSX"
  B=$SFX/m2.before; A2=$SFX/m2.after
  find "$MIR" -mindepth 1 -printf '%y\t%s\t%P\n' | LC_ALL=C sort > "$B"
  PROOF_SMOKE_LIB=1 . "$PS_MUT" || exit 90
  smoke_stage_break_links "$WSX" >/dev/null || exit 91
  printf 'M_LINKS=%s\n' "$(stat -c %h "$WSX/pypi-packs/pm-newton-pack/retread-probe-trace-pm-newton-pack.json")"
  printf 'trace-v2-longer\n' > "$WSX/pypi-packs/pm-newton-pack/retread-probe-trace-pm-newton-pack.json"
  find "$MIR" -mindepth 1 -printf '%y\t%s\t%P\n' | LC_ALL=C sort > "$A2"
  LC_ALL=C diff -q "$B" "$A2" >/dev/null && printf 'M_MIRROR=IDENTICAL\n' || printf 'M_MIRROR=CHANGED\n'
) > "$SFX/mut.txt" 2>&1
MOUT=$(cat "$SFX/mut.txt")
mfield () { printf '%s\n' "$MOUT" | sed -n "s/^$1=//p" | head -1; }
[ "$(mfield M_LINKS)" = 2 ] && [ "$(mfield M_MIRROR)" = CHANGED ]
chk S7 $? "MUTATION: with smoke_break_one a no-op the staged copy keeps links=2 and the write CHANGES the mirror -- S1/S6 measure the break, not the fixture" "links=$(mfield M_LINKS) mirror=$(mfield M_MIRROR) out=$(printf '%s' "$MOUT" | tr '\n' '|')"
fi
grep -q '^  smoke_stage_break_links "\$ws" || return 1$' "$PS_UNDER_TEST"
chk S8 $? "smoke_stage CALLS smoke_stage_break_links -- the function has a production call site" "no call site found"
rm -rf "$SFX"

if [ -n "${PSG_FIXTURE_ONLY:-}" ]; then
  # A cheap CPU job can run the fixture arms alone. The heavy arms below stage
  # against the LIVE shared mirror and take its try-lock; a lane that only
  # changed a fixture-testable function should not have to pay that, or contend
  # with a running relock for the mirror.
  echo "### PSG FIXTURE-ONLY: skipping arms A-N (they stage against the live mirror)"
  echo "### PSG SUMMARY pass=$pass fail=$fail (FIXTURE-ONLY subset)"
  rmdir "$SCR" 2>/dev/null
  echo "### PSG FINAL pass=$pass fail=$fail"
  [ "$fail" -eq 0 ] && exit 0 || exit 1
fi
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
# C3 IS WHY C IS A REAL CONTROL AND NOT AN ACCIDENT: the verdict is held by the
# MISSING BACKEND HALF, and the row has to say so.  THE FRONTEND HALF IS RACY AND
# WAS MEASURED TO BE: the same sleeping stub at the same 15 s cap scored
# frontend_rows=1 in 5994177 (the source-free `cpu` environment resolving) and
# frontend_rows=0 in 5994392, so an assertion on it would be a flaky guard.  What
# is NOT racy is that a sleeping backend serves ZERO `conda/outputs` -- so C3
# asserts backend_work_rows=0 and that BOTH counters are reported at all.
CROW=$(grep -m1 "lock ended verdict=" "$COUT")
{ echo "$CROW" | grep -q 'backend_work_rows=0' && echo "$CROW" | grep -q 'frontend_rows='; }
chk C3 $? "arm C's TIMEOUT row reports BOTH counters and backend_work_rows=0 -- the backend half is what holds the verdict" "${CROW:-<no lock-ended row>}"

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

# ---- N: PROOF-SMOKE-1-5, the root the budget was not modelling --------------
# These arms read the REPO copy (like G1-G4 and H), because the task copy is the
# unsynced 8108ca4 and is the CONTROL on what this lane did not touch.
echo ""; echo "########## PSG ARM N -- PROOF-SMOKE-1-5: the fast-tmp store root ##########"
NSMOKE=$REPO/harness/tools/proof_smoke.sh
NOUT=$OUT/psg-$J-N.out
# THE FIXTURE IS THE EXACT SHAPE THAT SAID headroom=16 AND THEN PANICKED.
# 6003336/6003855 ran with XDG_CACHE_HOME=<...>/retread/psg<jobid>/c/x, 48 bytes,
# and a fast-tmp root of <...>/retread/psg<jobid>/f, 46 bytes.  Nothing here
# touches the filesystem: the budget is string arithmetic and these are strings.
NFIX=/oscar/data/stellex/glvov/retread/psgFIX0001/c/x
NFIXFAST=/oscar/data/stellex/glvov/retread/psgFIX0001/f
: > "$NOUT"
bash -c 'PROOF_SMOKE_LIB=1 . "$1"; smoke_prefix_budget "$2" "N1 fixture (the 6003336 shape)" "$3"' \
     _ "$NSMOKE" "$NFIX" "$NFIXFAST" >>"$NOUT" 2>&1
n1rc=$?
bash -c 'PROOF_SMOKE_LIB=1 . "$1"; smoke_prefix_budget "$2" "N3 real headroom, no fast-tmp" ""' \
     _ "$NSMOKE" "$SCR/c/x" >>"$NOUT" 2>&1
n3rc=$?
cat "$NOUT"
[ "$n1rc" != 0 ]; chk N1 $? "the NEW rule REFUSES the fixture root that 6003336 ran and panicked on" "rc=$n1rc"
# NON-VACUITY, AND IT IS THE WHOLE POINT: the OLD function, on the SAME root,
# said composed=240 with 16 bytes to spare.  The new row carries that number
# itself so the inversion is on one page instead of across two logs.
grep -q 'OLD RULE .* said composed=240 headroom=16' "$NOUT"
chk N2 $? "the same row reports the OLD rule's verdict on the SAME root: composed=240 headroom=16" \
          "$(grep -m1 'OLD RULE' "$NOUT" || echo '<no OLD RULE row>')"
[ "$n3rc" = 0 ]; chk N3 $? "the NEW rule still PASSES a root with real headroom -- it does not just always refuse" "rc=$n3rc"

# THE MUTATION: delete the per-namespace suffix term, so the fast-tmp candidate
# composes with the SHORT rule and the fixture root sails through.
sed 's#fe=$(smoke_prefix_fasttmp_entry "$fast")#fe=$(smoke_prefix_entry "$fast")#' \
    "$NSMOKE" > "$SCR/proof_smoke.NMUT.sh"
nmutn=$(diff "$NSMOKE" "$SCR/proof_smoke.NMUT.sh" | grep -c '^< ')
bash -c 'PROOF_SMOKE_LIB=1 . "$1"; smoke_prefix_budget "$2" "N4 MUTATION" "$3"' \
     _ "$SCR/proof_smoke.NMUT.sh" "$NFIX" "$NFIXFAST" >"$OUT/psg-$J-N4.out" 2>&1
n4rc=$?
cat "$OUT/psg-$J-N4.out"
[ "$nmutn" = 1 ] && [ "$n4rc" = 0 ]
chk N4 $? "MUTATION: with the fast-tmp suffix term removed (1 changed line) the fixture root PASSES -- the term is what refuses" \
          "changed_lines=$nmutn rc=$n4rc"

# THE BOUNDARY, BOTH SIDES, against the MEASURED 167.  Asserted on the hard
# refusal's own row, not on rc: at 167 the headroom band still objects (255-8),
# and that is a different sentence from the pad overrun.
NR67=$(printf 'x%.0s' $(seq 1 67)); NR68=$(printf 'x%.0s' $(seq 1 68))
bash -c 'PROOF_SMOKE_LIB=1 . "$1"; smoke_prefix_budget "/$2" "N5 entry 167" ""' _ "$NSMOKE" "${NR67:1}" >"$OUT/psg-$J-N5.out" 2>&1
bash -c 'PROOF_SMOKE_LIB=1 . "$1"; smoke_prefix_budget "/$2" "N5 entry 168" ""' _ "$NSMOKE" "${NR68:1}" >>"$OUT/psg-$J-N5.out" 2>&1
cat "$OUT/psg-$J-N5.out"
n5b=$(grep -c 'PREFIX REFUSAL: composed 256' "$OUT/psg-$J-N5.out")
[ "$(grep -c 'entry=167 composed=255' "$OUT/psg-$J-N5.out")" = 1 ] && [ "$n5b" = 1 ]
chk N5 $? "the MEASURED boundary: entry 167 composes to exactly the pad (255) and entry 168 trips the pad refusal at 256" \
          "entry167_rows=$(grep -c 'entry=167 composed=255' "$OUT/psg-$J-N5.out") pad_refusals_at_256=$n5b"

# ---- N6/N7: THE COIN, RUN THREE TIMES ---------------------------------------
# HARNESS-CONSOL-5 measured arm A red/green/red on ONE binary across three solo
# jobs and drew the standing consequence that a single arm-A result carries no
# information.  THAT IS THE THING THIS ARM HAS TO RETIRE, and the only way to
# retire it is three runs in ONE job on the FIXED smoke.  They share the staged
# workspace and the cache, so runs 2 and 3 are cheap.
echo ""; echo "########## PSG ARM N7 -- the FIXED smoke, three times, one job ##########"
n7pass=0
for r in 1 2 3; do
  RO=$OUT/psg-$J-N7-$r.out
  SMOKE_WALL=${PSG_WALL_A:-900} bash "$NSMOKE" "$GOOD" "$MANIFEST" "$JOB_ROOT" >"$RO" 2>&1
  rrc=$?
  echo "### PSG N7 run $r rc=$rrc $(grep -m1 '^### SMOKE [A-Z_]* binary=' "$RO" || echo '<no verdict row>')"
  grep -m1 '^### SMOKE lock ended' "$RO" | cut -c1-200 | sed 's/^/### PSG   /'
  grep -m1 '^### SMOKE PREFIX BUDGET smoke store root ' "$RO" | cut -c1-220 | sed 's/^/### PSG   /'
  [ "$rrc" = 0 ] && grep -q '^### SMOKE REACHED_FRONTEND ' "$RO" && n7pass=$((n7pass+1))
done
grep -q '^### SMOKE fast-tmp mode=off' "$OUT/psg-$J-N7-1.out"
chk N6 $? "the fixed smoke PRINTS its fast-tmp mode -- the divergence from the driver is on the page, not silent" \
          "$(grep -m1 'fast-tmp mode=' "$OUT/psg-$J-N7-1.out" || echo '<no mode row>')"
[ "$n7pass" = 3 ]
chk N7 $? "THREE consecutive REACHED_FRONTEND on one binary -- the coin HARNESS-CONSOL-5 measured (red/green/red) is gone" \
          "reached_frontend=$n7pass of 3"

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

# ---- G: the smoke root against the ARMS' roots (PROOF-SMOKE-1-3) ------------
# THE DEFECT, MEASURED, not imagined: det141-proof 6000903 died in ELEVEN
# SECONDS. Its three arm cache homes `$C/x<N>` composed to 247 (headroom 9) and
# PASSED; its smoke's DEFAULT store root `$C/smk/c/x` composed to 252 (headroom
# 4) and was REFUSED. The smoke exists to ask the CHEAP question before the arms
# commit -- a smoke root LONGER than an arm root makes it ask a HARDER one, and
# every driver re-cut onto this preamble with a six-character tag inherits that.
#
# THESE ARMS ARE PURE ARITHMETIC ON PATH LENGTHS: no binary, no stage, no lock,
# nothing created on disk. The roots are NAMES, measured at their real byte
# lengths under the real `/oscar/data/stellex/glvov/retread` base, and G1's is
# 6000903's own root, character for character.
#
# THE SUBJECT IS THE REPO COPY, deliberately, and it is the same choice
# `cleanup_absent_root_guard.sh` makes: this guard runs BEFORE the sync that
# installs the fix into `$T/tools/`, so a G arm reading the task copy would test
# the file the fix is not in yet. Arms A-F keep reading the task copies, which
# is what makes them a control on the unchanged behaviour.
echo ""; echo "########## PSG ARM G -- the smoke root vs the ARMS' roots ##########"
PSG_G_PRE=${PSG_G_PRE:-$REPO/harness/tools/multiarm_preamble.sh}
PSG_G_SMK=${PSG_G_SMK:-$REPO/harness/tools/proof_smoke.sh}
# THE MUTATION'S COMMIT CONSTANT: the harness tip this lane opened on, the one
# 6000903 ran. Pinned, never HEAD~1.
PSG_G_OLD=${PSG_G_OLD:-8108ca48b21533d92a9371c8ef289877fd311eb8}
RB=/oscar/data/stellex/glvov/retread

cat > "$SCR/run_prefix.sh" <<'GRUNNER'
#!/usr/bin/env bash
# $1 smoke lib, $2 preamble, $3 cache root, $4 arm count, $5 new|old,
# $6 (old only) the smoke store root the OLD default composes.
set -uo pipefail
PROOF_SMOKE_LIB=1 . "$1" || exit 9
# shellcheck source=/dev/null
. "$2" || exit 9
MULTIARM_CACHE_ROOT=$3
MULTIARM_ARMS=$4
arms_rc=0
for n in $(seq 1 "$4"); do
  multiarm_prefix_check "$(multiarm_arm_cache_home "$n")" "arm $n cache home" || arms_rc=1
done
if [ "$5" = new ]; then sroot=$(multiarm_smoke_store_root); else sroot=$6; fi
smoke_rc=0
smoke_prefix_budget "$sroot" "smoke store root" || smoke_rc=1
echo "### PSG-G ARMS_RC=$arms_rc SMOKE_RC=$smoke_rc mode=$5 smoke_root=$sroot"
sym_rc=9
if declare -F multiarm_smoke_prefix_symmetry >/dev/null 2>&1; then
  sym_rc=0; multiarm_smoke_prefix_symmetry || sym_rc=1
else
  echo "### PSG-G no multiarm_smoke_prefix_symmetry in $2 -- a pre-fix preamble"
fi
echo "### PSG-G SYM_RC=$sym_rc"
GRUNNER
chmod +x "$SCR/run_prefix.sh"

grun () {  # grun <log> <smoke lib> <preamble> <cache root> <arms> <mode> [old root]
  # `SMOKE_CACHE`/`SMOKE_ROOT`/`SMOKE_WS` are EXPORTED above for arms A-F, and
  # `multiarm_smoke_cache_home` honours an override by design -- so a G arm that
  # inherited them would measure the guard's own scratch instead of the fixture,
  # and would pass whatever the preamble did. They are stripped per call.
  local log=$1; shift
  env -u SMOKE_CACHE -u SMOKE_ROOT -u SMOKE_WS bash "$SCR/run_prefix.sh" "$@" >"$log" 2>&1
  cat "$log"
}
gval () { sed -n "s/.*$2=\([^ ]*\).*/\1/p" "$1" | head -1; }

# G1: 6000903's own shape -- a SIX-character tag, three arms.
G1C=$RB/certDET141-6000903
G1LOG=$OUT/psg-$J-G1.out
grun "$G1LOG" "$PSG_G_SMK" "$PSG_G_PRE" "$G1C" 3 new
{ [ "$(gval "$G1LOG" ARMS_RC)" = 0 ] && [ "$(gval "$G1LOG" SMOKE_RC)" = 0 ] && [ "$(gval "$G1LOG" SYM_RC)" = 0 ]; }
chk G1 $? "a 6-character tag: arms fit AND the smoke is NOT refused (6000903's exact roots)" \
  "arms=$(gval "$G1LOG" ARMS_RC) smoke=$(gval "$G1LOG" SMOKE_RC) sym=$(gval "$G1LOG" SYM_RC)"

# G2: an EIGHT-character tag, on a root two bytes shorter, so the arms still
# clear the headroom band and the question stays about the SMOKE.
G2C=$RB/gDET14100-6000903
G2LOG=$OUT/psg-$J-G2.out
grun "$G2LOG" "$PSG_G_SMK" "$PSG_G_PRE" "$G2C" 3 new
{ [ "$(gval "$G2LOG" ARMS_RC)" = 0 ] && [ "$(gval "$G2LOG" SMOKE_RC)" = 0 ] && [ "$(gval "$G2LOG" SYM_RC)" = 0 ]; }
chk G2 $? "an 8-character tag: arms fit AND the smoke is NOT refused" \
  "arms=$(gval "$G2LOG" ARMS_RC) smoke=$(gval "$G2LOG" SMOKE_RC) sym=$(gval "$G2LOG" SYM_RC)"

# G3: THE ARMS THEMSELVES OVERRUN. The job must still refuse, and the refusal
# must name the ARM budget -- shortening the smoke would fix nothing here.
G3C=$RB/certAVERYLONGTAGINDEED-6000903
G3LOG=$OUT/psg-$J-G3.out
grun "$G3LOG" "$PSG_G_SMK" "$PSG_G_PRE" "$G3C" 3 new
{ [ "$(gval "$G3LOG" ARMS_RC)" = 1 ] && [ "$(gval "$G3LOG" SMOKE_RC)" = 1 ] \
  && grep -q 'SMOKE PREFIX SYMMETRY .*max_arm_composed=' "$G3LOG" \
  && grep -q 'arm 3 cache home' "$G3LOG"; }
chk G3 $? "arms that overrun refuse the job AND the symmetry row names the ARM budget" \
  "arms=$(gval "$G3LOG" ARMS_RC) smoke=$(gval "$G3LOG" SMOKE_RC) sym_row=$(grep -c 'SMOKE PREFIX SYMMETRY' "$G3LOG")"

# G4: THE MUTATION -- 6000903's own preamble, extracted from the commit it ran,
# on G1's fixture. The OLD default suffix is READ OUT of the old file rather
# than restated here: a guard that hardcodes the thing it is mutating away from
# stops testing the moment that default moves.
G4PRE=$SCR/multiarm_preamble.$PSG_G_OLD.sh
G4SMK=$SCR/proof_smoke.$PSG_G_OLD.sh
G4LOG=$OUT/psg-$J-G4.out
if git -C "$REPO" show "$PSG_G_OLD:harness/tools/multiarm_preamble.sh" > "$G4PRE" 2>/dev/null && [ -s "$G4PRE" ] \
   && git -C "$REPO" show "$PSG_G_OLD:harness/tools/proof_smoke.sh" > "$G4SMK" 2>/dev/null && [ -s "$G4SMK" ]; then
  OLDSUF=$(grep -oE 'SMOKE_CACHE:-\$MULTIARM_CACHE_ROOT[^}]*' "$G4PRE" | head -1 | sed 's/.*MULTIARM_CACHE_ROOT//')
  echo "### PSG-G the pinned $PSG_G_OLD preamble's smoke cache default is \$MULTIARM_CACHE_ROOT$OLDSUF"
  if [ -z "$OLDSUF" ]; then
    fail=$((fail+1)); echo "### PSG FAIL G4   could not read the old smoke-cache default out of $PSG_G_OLD -- WRONG PIN, G1 proves nothing"
  else
    grun "$G4LOG" "$G4SMK" "$G4PRE" "$G1C" 3 old "$G1C$OLDSUF/x"
    { [ "$(gval "$G4LOG" ARMS_RC)" = 0 ] && [ "$(gval "$G4LOG" SMOKE_RC)" = 1 ] && [ "$(gval "$G4LOG" SYM_RC)" = 9 ]; }
    chk G4 $? "THE DEFECT, REPRODUCED: at $PSG_G_OLD the arms PASS and the smoke's own root is REFUSED, with no symmetry row anywhere -- 6000903 verbatim" \
      "arms=$(gval "$G4LOG" ARMS_RC) smoke=$(gval "$G4LOG" SMOKE_RC) sym=$(gval "$G4LOG" SYM_RC)"
  fi
else
  fail=$((fail+1)); echo "### PSG FAIL G4   could not extract $PSG_G_OLD's preamble/proof_smoke -- THE MUTATION DID NOT RUN"
fi

# ---- H: PROOF-SMOKE-1-4 -- the shared stage mirror is taken under a try-lock -
# THE HAZARD, AND A CORRECTION TO THE READING THAT COMMISSIONED THIS ARM.
# The story was a removal test: 6001839 (14 checks, node2311) and 6001840 (18
# checks, node2320) STARTED IN THE SAME SECOND, 22:31:16, and the second lost
# arm A at 130 s -- `### SMOKE BACKEND_DIED`, `A2 rc=1`, quoting `end byte index
# 18446744073709551591 is out of bounds for string of length 260`,
# frontend_rows=0 backend_work_rows=16 -- while the chained rerun 6002138, with
# no sibling psg job alive, scored 18/0. Concurrency was declared confirmed.
# **THAT READING IS WITHDRAWN, BY THIS GUARD'S OWN NEXT RUN.** 6003336 asked
# `squeue` for a sibling psg job before starting, printed `no sibling psg job
# alive -- this run holds the stage mirror alone`, and arm A went RED anyway
# with the identical signature and the identical declared budget row
# (`entry=148 composed=240 pad=256 headroom=16`) that the GREEN control 6002138
# printed. The cause is named by proof_smoke.sh's own detector --
# `reason=PREFIX_PANIC_256`, a composed build prefix of 260 against a 256 pad --
# and the budget check UNDERCOUNTS IT BY TWENTY BYTES. That root defect is
# BOARDED and is NOT what these arms test.
# WHAT THESE ARMS DO TEST is a real and separately evidenced hazard: `cp -al`
# out of $SMOKE_MIRROR_ROOT/$key hands out THE MIRROR'S OWN inodes, so two jobs
# staging from one mirror hold the same files and one in-place write reaches
# both plus the mirror. The live mirror root carries three `.DIRTY-<jobid>`
# quarantines written by phaseN_relock.sh's own reader, so the write-through has
# fired before. The lock serialises that; it does not, and is not claimed to,
# fix arm A.
# These arms run against a FIXTURE mirror root under this guard's own scratch --
# they never touch the live mirror, and they source proof_smoke.sh as a LIB so
# no smoke is run and no binary is involved.
echo ""; echo "########## PSG ARM H -- the stage mirror try-lock (PROOF-SMOKE-1-4) ##########"
HDIR=$SCR/h
mkdir -p "$HDIR/mirror" "$HDIR/src"
printf '[project]\nname = "psg-h"\n' > "$HDIR/src/pixi.toml"
# THE REPO COPY, DELIBERATELY, and for the same reason arms G1-G4 read it: the
# sync that would install this fix into $T is refused while other lanes' jobs
# are live, so the task copy is still the pre-fix bytes and asserting the fix
# against it would assert the fix against its own absence. Arms A-F keep reading
# $T (they are the control on what this lane did not touch).
HSMOKE=${PSG_SMOKE_REPO:-$REPO/harness/tools/proof_smoke.sh}
echo "### PSG H reads $HSMOKE (repo copy: the fix is not synced into \$T)"
hsetup () {   # source proof_smoke.sh (or a mutant, \$1) as a LIB over the fixture
  export SMOKE_MIRROR_ROOT=$HDIR/mirror
  export SMOKE_SRC_WS=$HDIR/src
  export PROOF_SMOKE_LIB=1
  # shellcheck source=/dev/null
  . "${1:-$HSMOKE}" >/dev/null 2>&1
}
hlock () {  # echo the lock path the LIB itself spells -- never a second copy of it
  ( hsetup
    smoke_stage_lock_path "$(smoke_stage_key)" )
}
HLOCK=$(hlock)
# h1: TWO acquisitions from ONE job, sequentially, both succeed. A driver that
# smokes N binaries in one job must not refuse itself -- refusing here would
# refuse every multi-arm driver in the tree, which is a worse defect than the
# one under repair.
H1OUT=$SCR/h1.out
( hsetup

  K=$(smoke_stage_key)
  smoke_stage_lock_acquire "$K"; echo "first_rc=$?"
  smoke_stage_lock_acquire "$K"; echo "second_rc=$?"
  smoke_stage_lock_release ) > "$H1OUT" 2>&1
grep -q 'first_rc=0' "$H1OUT" && grep -q 'second_rc=0' "$H1OUT" && grep -q 'ADOPTED' "$H1OUT"
chk H1 $? "two arms of ONE job take the same mirror sequentially (second ADOPTS, both rc 0)" "$(grep -E 'first_rc|second_rc' "$H1OUT" | tr '\n' ' ')"
rm -rf "$HLOCK"
# h2: a lock PRE-HELD by a DIFFERENT, LIVE job. The holder named in the owner
# file is THIS guard's own job id, because the liveness check asks squeue and the
# only job this guard is entitled to ask about is its own; the ASKER is a
# different id, so this is exactly the "the holder is alive, refuse" branch.
# The refusal must be the BUSY row, non-zero, and NOT a panic.
mkdir -p "$HLOCK"
{ echo "job=$J"; echo "host=elsewhere"; echo "pid=1"; echo "at=$(date -Is)"; } > "$HLOCK/owner"
H2OUT=$SCR/h2.out
( hsetup

  SLURM_JOB_ID=$(( J + 1 ))          # a DIFFERENT job asking for a LIVE holder's lock
  smoke_stage_lock_acquire "$(smoke_stage_key)"; echo "busy_rc=$?" ) > "$H2OUT" 2>&1
grep -q '### SMOKE STAGE BUSY' "$H2OUT" && grep -q 'busy_rc=1' "$H2OUT"
chk H2 $? "a mirror held by a LIVE foreign job refuses '### SMOKE STAGE BUSY' rc 1, loudly" "$(grep -E 'busy_rc|STAGE BUSY' "$H2OUT" | head -1)"
grep -q "ACTUATOR: chain this job after $J" "$H2OUT"
chk H3 $? "and the BUSY row names the ACTUATOR -- the holder's job id to chain after" "$(grep -m1 ACTUATOR "$H2OUT" || echo '<no actuator row>')"
# THE PATTERN IS THE PANIC'S OWN SHAPE, NOT THE WORD "panic". The first cut
# grepped `panic|out of bounds` and went RED on run 6003336 by matching the
# REFUSAL'S OWN PROSE -- the BUSY row explained itself by naming the panic it
# exists to avoid, so the arm caught its own explanation. A guard whose pattern
# can match the text it guards is the law-14 defect one level up, and it is
# fixed the same way: build the pattern so it cannot match the guard's subject
# saying the word.
grep -qE "thread '[^']*' panicked|panicked at |is out of bounds for string of length" "$H2OUT"
[ $? -ne 0 ]
chk H4 $? "and it is a REFUSAL, not a panic (no Rust panic frame anywhere in the refusal)" "$(grep -m1 -E "panicked at |out of bounds for string of length" "$H2OUT" || echo none)"
# h5: THE NON-VACUITY CONTROL. BUSY must not be the only answer the lock knows:
# a lock whose owner job is NOT in the queue is reclaimed, loudly, so a smoke
# killed mid-stage cannot wedge every later smoke.
{ echo "job=99999999"; echo "host=gone"; echo "pid=1"; echo "at=$(date -Is)"; } > "$HLOCK/owner"
H5OUT=$SCR/h5.out
( hsetup

  SLURM_JOB_ID=$(( J + 2 ))
  smoke_stage_lock_acquire "$(smoke_stage_key)"; echo "stale_rc=$?"
  smoke_stage_lock_release ) > "$H5OUT" 2>&1
grep -q '### SMOKE STAGE LOCK STALE' "$H5OUT" && grep -q 'stale_rc=0' "$H5OUT"
chk H5 $? "a lock whose owner job is NOT in the queue is RECLAIMED loudly, not waited on forever" "$(grep -E 'stale_rc|LOCK STALE' "$H5OUT" | head -1)"
rm -rf "$HLOCK"
# h6: THE MUTATION. With the acquire cut out of smoke_stage, h2's fixture
# PROCEEDS past a live holder -- which is 6001840. Without this arm h2 could be
# passing on the fixture rather than on the lock.
HMUT=$SCR/proof_smoke_nolock.sh
sed 's@^  smoke_stage_lock_acquire "$key" || return 2$@  :@' "$HSMOKE" > "$HMUT"
if bash -n "$HMUT" 2>/dev/null && ! grep -q 'smoke_stage_lock_acquire "$key" || return 2' "$HMUT"; then
  mkdir -p "$HLOCK"
  { echo "job=$J"; echo "host=elsewhere"; echo "pid=1"; echo "at=$(date -Is)"; } > "$HLOCK/owner"
  H6OUT=$SCR/h6.out
  ( hsetup "$HMUT"
    SLURM_JOB_ID=$(( J + 3 ))
    mkdir -p "$HDIR/ws6"
    smoke_stage "$HDIR/ws6"; echo "mut_stage_rc=$?" ) > "$H6OUT" 2>&1
  ! grep -q '### SMOKE STAGE BUSY' "$H6OUT"
  chk H6 $? "MUTATION -- with the acquire cut, staging PROCEEDS past a live holder (6001840), so H2 CAN fail" "$(grep -E 'mut_stage_rc|STAGE BUSY' "$H6OUT" | head -1)"
  rm -rf "$HLOCK"
else
  chk H6 1 "MUTATION -- the no-lock mutant builds and runs" "the mutant did not build: MUTATION ARM DID NOT RUN"
fi

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
#
# AND IT GOES THROUGH `multiarm_store_reap.sh`, NOT A BARE `rm -rf`.  MEASURED
# on 5994392: a bare `rm -rf "$SCR"` printed **1592** errors, because a smoke
# that reaches the backend leaves rattler-build's SEALED conda environments
# under it -- `make_source_tree_read_only` strips write from DIRECTORIES, and
# unlinking an entry is a write to its parent (ORDER-1-1's whole finding, met
# here from the other end).  `reap_delete` is the versioned answer: rename
# FIRST, re-prove containment by dev:inode on the RENAMED path, chmod only
# that, then remove SYNCHRONOUSLY with stderr COUNTED.  The workspace `w` is
# handled separately and WITHOUT a chmod, because it is a `cp -al` clone of the
# shared stage mirror and a recursive chmod there would change the mode of the
# MIRROR'S OWN inodes (p6x rule 2).
REAP_JOB_ROOT=$SCR
# shellcheck source=/dev/null
. "$T/tools/multiarm_store_reap.sh"
rmerr=0
if reap_init; then
  # the hardlink clone first, plain and unchmodded
  if [ -d "$SCR/w" ]; then
    n=$( rm -rf "$SCR/w" 2>&1 | wc -l ); rmerr=$((rmerr + n))
  fi
  # PLAIN FILES FIRST, AND reap_delete GETS DIRECTORIES ONLY.  MEASURED on
  # 5994853: `reap_delete` REFUSED `run_preamble.sh` and
  # `multiarm_preamble.MUT.sh` as "NOT strictly under the declared job root" --
  # its containment proof is a dev:inode walk through `..`, which is a statement
  # about directories, and handing it a regular file asks it a question it
  # cannot answer.  That refusal is the tool being right; the caller was wrong.
  find "$SCR" -maxdepth 1 -type f -delete 2>/dev/null
  for d in "$SCR"/*; do
    [ -d "$d" ] || continue
    reap_delete "$d" psg || rmerr=$((rmerr + 1))
  done
  reap_done || rmerr=$((rmerr + 1))
  rmdir "$SCR" 2>/dev/null
else
  echo "### PSG the reap context refused this guard's own scratch root -- leaving $SCR in place"
  rmerr=$((rmerr + 1))
fi
[ "$rmerr" -eq 0 ] || fail=$((fail+1))

# ---- out --------------------------------------------------------------------
echo ""
echo "### PSG scratch removed errors=$rmerr root=$SCR (via multiarm_store_reap: aside=${REAP_ASIDE:-?} deleted=${REAP_DELETED:-?} refused=${REAP_REFUSED:-?})"
echo "### PSG per-arm verdicts:"
for f in "$AOUT" "$BOUT" "$COUT" "$FOUT"; do
  printf '###   %-28s %s\n' "$(basename "$f")" "$(grep -m1 '^### SMOKE [A-Z_]* binary=' "$f" 2>/dev/null || echo '<none>')"
done
echo "### PSG SUMMARY pass=$pass fail=$fail (predicted pass=39 fail=0)"
echo "### PSG FINAL pass=$pass fail=$fail"
if [ "$fail" -eq 0 ]; then exit 0; fi
exit 1
