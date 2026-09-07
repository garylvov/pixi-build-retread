#!/usr/bin/env bash
# cold_proof_arm_guard.sh -- THE READER OF C31-4'S ONE RELOCK-SIDE PRODUCER.
#
# WHAT IT GUARDS. `tools/retread_fast_env.sh` defines
# `retread_relock_scope_and_verify <job root> ["$@"]`: it scopes the uv sdist
# build trees into the job root (C31-4), SKIPS that scoping when the forwarded
# argv carries `--cold-proof-arm` (DET-1-6-2), and runs
# `tools/sdist_build_poison_guard.sh` either way. Every relock template in this
# harness must reach C31-4 through that function and through nothing else.
#
# WHY THE FLAG EXISTS. `retread_scope_sdist_builds` refuses when it symlinked no
# byte-keyed bucket, which is exactly the state of a DECLARED-COLD PROOF ARM's
# own empty `RETREAD_PERSIST_CACHE_ROOT` (measured: job 6014471 arm W1,
# `symlinked=0 ... shared_links_seen=0` then `FATAL no byte-keyed bucket was
# symlinked`). That refusal is correct for a production relock and wrong for
# such an arm, and pre-seeding the root is not the fix -- the bucket a cold
# proof must not share is `sdists-v9`, the one bucket the scoper never links.
# So the shape is declared ON ARGV.
#
# ── WHAT THIS GUARD LOOKED LIKE YESTERDAY, AND WHY IT CHANGED ────────────────
# HARNESS-CONSOL-8 shipped this file with a CLASSIFIER: a target that did not
# call the scoper was reported NOT-APPLICABLE, because three of the four relock
# templates had never received C31-4 at all. Measured then by `grep -c`:
#
#     phase_template/phaseN_relock.sh   scoper=3  poison_guard=2
#     arms/mh1_relock.sh                scoper=0  poison_guard=0
#     arms/c29_relock.sh                scoper=0  poison_guard=0
#     proof/hlgd_relock.sh              scoper=0  poison_guard=0
#
# and that row said "the day mh1_relock.sh gains the C31-4 scoper, this guard
# flips it to APPLICABLE". HARNESS-CONSOL-9 landed the back-port, so all four
# are applicable and THE CLASSIFIER IS GONE. It has to go: with it in place a
# template that LOST the call would go quiet (N/A) instead of red, and the
# regression this campaign just paid for would be undetectable. Absence is now a
# FAILURE, which is the only shape in which arm 4 is a guard at all.
#
# ARMS
#   1  GLOBAL. The real scoper against a real empty shared cache MUST refuse
#      and MUST print the `no byte-keyed bucket` FATAL. If this goes green the
#      flag is solving a problem that no longer exists.
#   H2 GLOBAL. `retread_relock_scope_and_verify` with `--cold-proof-arm` in the
#      forwarded argv MUST print the SKIPPED row, MUST NOT call the scoper, and
#      MUST still run the poison guard (a cold arm keeps the shared, poisonable
#      caches, so it needs the reader more, not less).
#   H3 GLOBAL MUTATION ARM. The same function without the flag MUST NOT print
#      SKIPPED and MUST call the scoper. An H2 that passes with H3 also passing
#      is not a gate: it would pass a function that skips unconditionally.
#   HP GLOBAL. The same function, poison planted in the caches the guard reads,
#      MUST refuse in BOTH shapes. Without this the poison guard could be
#      commented out of the producer and H2/H3 would not notice.
#   4  PER TARGET. The template calls `retread_relock_scope_and_verify`, and
#      FORWARDS AN ARGV to it -- a call that drops the argv silently loses the
#      flag. It must also NOT carry its own `COLD_PROOF_ARM` parse: a second
#      copy of the branch is the drift this consolidation removed.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
FAST=$HERE/retread_fast_env.sh
# Explicit targets win (the caller passes paths); otherwise the shipped set.
if [ "$#" -gt 0 ]; then
  TARGETS="$*"
else
  TARGETS=
  for t in "$HERE/../phase_template/phaseN_relock.sh" "$HERE/../arms/mh1_relock.sh" \
           "$HERE/../arms/c29_relock.sh" "$HERE/../proof/hlgd_relock.sh" \
           "$HERE/../instrumented/p6b_relock.sh" "$HERE/../instrumented/p6b_relock.b2.sh"; do
    [ -f "$t" ] && TARGETS="$TARGETS $t"
  done
fi
W=$(mktemp -d "${TMPDIR:-/tmp}/coldproof.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
fail=0
say () { echo "### COLD-PROOF-ARM GUARD $*"; }

[ -n "${TARGETS// /}" ] || { say "FATAL no relock template to read"; exit 3; }
[ -f "$FAST" ] || { say "FATAL retread_fast_env.sh not found next to this guard ($FAST)"; exit 3; }

# ---- ARM 1 (GLOBAL): the refusal the flag exists to bypass, reproduced live --
( set +e
  # shellcheck source=/dev/null
  . "$FAST"
  mkdir -p "$W/shared/uv" "$W/shared/pixi" "$W/job"     # EMPTY, like a proof arm's own root
  export UV_CACHE_DIR=$W/shared/uv PIXI_CACHE_DIR=$W/shared/pixi
  retread_scope_sdist_builds "$W/job" > "$W/arm1.out" 2>&1
  echo $? > "$W/arm1.rc" )
A1RC=$(cat "$W/arm1.rc" 2>/dev/null || echo NONE)
if [ "$A1RC" != 0 ] && grep -q 'no byte-keyed bucket was symlinked' "$W/arm1.out"; then
  say "ARM1 PASS the scoper still refuses an empty shared cache (rc=$A1RC, FATAL row present)"
else
  say "ARM1 FAIL the scoper did NOT refuse an empty shared cache (rc=$A1RC) -- the flag now bypasses nothing:"
  sed 's/^/###   /' "$W/arm1.out"; fail=1
fi

# ---- ARMS H2/H3/HP (GLOBAL): the producer itself -----------------------------
# The scoper is STUBBED (redefined after sourcing) so the arm reads the branch,
# not the overlay machinery -- that half is guarded by sdist_build_scope_guard.
# The POISON GUARD IS REAL and runs against fixture cache roots. Nothing here
# touches the shared cache or runs `pixi lock`; the whole arm is fixture-only.
mk_fix () {  # mk_fix <root> <compiler path>
  local root=$1 cc=$2
  local d=$root/uv-cache/sdists-v9/pypi/openmesh/1.2.1/lv_fixture/src/build-setuptools/temp
  mkdir -p "$d"
  printf 'CMAKE_CXX_COMPILER:FILEPATH=%s\n' "$cc" > "$d/CMakeCache.txt"
}
run_helper () {  # run_helper <outfile> <fixture pixi root> [extra argv...]
  local out=$1 pixi=$2; shift 2
  ( set +e
    # shellcheck source=/dev/null
    . "$FAST"
    retread_scope_sdist_builds () { echo "SCOPER-WAS-CALLED"; return 0; }
    mkdir -p "$pixi"
    export UV_CACHE_DIR=$pixi/uv-cache PIXI_CACHE_DIR=$pixi
    retread_relock_scope_and_verify /nonexistent/job/root "$@"
    echo $? > "$out.rc" ) > "$out" 2>&1
}
GOODCC=""
for c in /usr/bin/c++ /usr/bin/g++ /bin/sh; do [ -e "$c" ] && { GOODCC=$c; break; }; done
[ -n "$GOODCC" ] || { say "FATAL no existing tool to build a clean fixture from"; exit 3; }
DEADCC=/oscar/data/stellex/glvov/retread/ws.GUARD-DOES-NOT-EXIST/bin/x86_64-conda-linux-gnu-c++
[ -e "$DEADCC" ] && { say "FATAL the fixture's 'dead' compiler path exists: $DEADCC"; exit 3; }

mk_fix "$W/clean" "$GOODCC"
mk_fix "$W/poisoned" "$DEADCC"

run_helper "$W/h2.out" "$W/clean" TAG /root --cold-proof-arm
H2RC=$(cat "$W/h2.out.rc" 2>/dev/null || echo NONE)
if [ "$H2RC" = 0 ] && grep -q 'sdist scoping SKIPPED (declared cold proof arm)' "$W/h2.out" \
   && ! grep -q 'SCOPER-WAS-CALLED' "$W/h2.out" && grep -q 'cmakecache_files=' "$W/h2.out"; then
  say "ARMH2 PASS with the flag: SKIPPED row printed, scoper NOT called, poison guard still walked the caches (rc=$H2RC)"
else
  say "ARMH2 FAIL (rc=$H2RC):"; sed 's/^/###   /' "$W/h2.out"; fail=1
fi

run_helper "$W/h3.out" "$W/clean" TAG /root
H3RC=$(cat "$W/h3.out.rc" 2>/dev/null || echo NONE)
if [ "$H3RC" = 0 ] && ! grep -q 'sdist scoping SKIPPED' "$W/h3.out" \
   && grep -q 'SCOPER-WAS-CALLED' "$W/h3.out" && grep -q 'sdist builds scoped' "$W/h3.out"; then
  say "ARMH3 PASS (mutation) without the flag: no SKIPPED row, the scoper WAS called, the scoped row printed (rc=$H3RC)"
else
  say "ARMH3 FAIL (rc=$H3RC) the producer behaves the same with and without the flag -- it is not a gate:"
  sed 's/^/###   /' "$W/h3.out"; fail=1
fi

for shape in cold hot; do
  if [ "$shape" = cold ]; then run_helper "$W/hp.$shape.out" "$W/poisoned" TAG /root --cold-proof-arm
  else                         run_helper "$W/hp.$shape.out" "$W/poisoned" TAG /root; fi
  HPRC=$(cat "$W/hp.$shape.out.rc" 2>/dev/null || echo NONE)
  if [ "$HPRC" = 7 ] && grep -q 'sdist build poison guard refused' "$W/hp.$shape.out" \
     && grep -q "$DEADCC" "$W/hp.$shape.out"; then
    say "ARMHP PASS $shape shape: a poisoned sdist tree is REFUSED (rc=$HPRC) and the refusal names the missing tool"
  else
    say "ARMHP FAIL $shape shape (rc=$HPRC, want 7): the poison guard did not refuse through the producer:"
    sed 's/^/###   /' "$W/hp.$shape.out"; fail=1
  fi
done

# ---- ARM 4, once per target: the template REACHES the producer ---------------
napp=0
for TPL in $TARGETS; do
  TN=$(basename "$TPL")
  if [ ! -f "$TPL" ]; then say "ARM4 FAIL template absent: $TPL"; fail=1; continue; fi
  napp=$((napp + 1))
  CALL=$(grep -E '^[[:space:]]*retread_relock_scope_and_verify[[:space:]]' "$TPL" | head -1)
  if [ -z "$CALL" ]; then
    say "ARM4 FAIL $TN never calls retread_relock_scope_and_verify -- this template does not scope its sdist build trees and does not run the poison guard. C31-4 was back-ported into every relock template by HARNESS-CONSOL-9; losing the call is a regression, not a variant."
    fail=1; continue
  fi
  # The forwarded argv is what carries --cold-proof-arm. A call with only the
  # job root silently drops the flag, so the third field must name an argv.
  if printf '%s\n' "$CALL" | grep -q '@'; then
    say "ARM4 PASS $TN reaches the producer and forwards an argv:$(printf '%s' "$CALL" | sed 's/^[[:space:]]*/ /')"
  else
    say "ARM4 FAIL $TN calls the producer WITHOUT forwarding an argv -- --cold-proof-arm can never reach it:$(printf '%s' "$CALL" | sed 's/^[[:space:]]*/ /')"
    fail=1
  fi
  NOWN=$(grep -c 'COLD_PROOF_ARM' "$TPL")
  if [ "$NOWN" = 0 ]; then
    say "ARM4 PASS $TN keeps no second copy of the flag branch (COLD_PROOF_ARM=$NOWN)"
  else
    say "ARM4 FAIL $TN carries its own COLD_PROOF_ARM parse ($NOWN line(s)) -- two producers of one branch is the drift this consolidation removed"
    fail=1
  fi
done

# ---- ARM 5 (DET-1-6-3): the frontend log filter, EXECUTED, then MUTATED ------
# WHY IT IS HERE. A cold proof arm's whole point is that it BUILDS rather than
# replays, and the only evidence of a build is the `uv_distribution` span
# `build_metadata{dist=...}` in the frontend log. Job 6015646 arm W1 asserted
# exactly that and read ZERO -- and was WRONG about why: the log had zero
# `build_metadata` rows FOR EVERY DIST while carrying 18390 DEBUG rows, because
# the relock template did `unset RUST_LOG` and then ran `pixi lock -v`, whose
# own filter overrides RUST_LOG. A criterion whose producer is switched off one
# function above it is a criterion with no live producer (doctrine law 2), and
# this arm is that producer's reader.
#
# IT EXECUTES THE REAL BYTES. `retread_relock_frontend_log` is sourced out of
# retread_fast_env.sh and CALLED, in both shapes; nothing here re-implements its
# logic, because a guard that greps for the logic it expects passes a file that
# says the right thing and does the wrong one.
A5F='uv_distribution=debug,pixi=info,warn'

# 5a DEFAULT SHAPE: no flag must be byte-for-byte today's behaviour.
A5A=$( set +e
  # shellcheck source=/dev/null
  . "$FAST" >/dev/null 2>&1
  retread_relock_frontend_log --cold-proof-arm
  echo "RESULT rust_log=${RUST_LOG-<unset>} lock_verbosity=${LOCK_VERBOSITY-<UNSET-VAR>}" )
A5AR=$(printf '%s\n' "$A5A" | grep -F 'RESULT ')
if [ "$A5AR" = 'RESULT rust_log=<unset> lock_verbosity=-v' ] \
   && printf '%s\n' "$A5A" | grep -qF "### INSTRUMENTATION: frontend RUST_LOG unset, lock verbosity '-v'"; then
  say "ARM5a PASS no flag = today's behaviour exactly ($A5AR)"
else
  say "ARM5a FAIL no flag must leave RUST_LOG unset and LOCK_VERBOSITY=-v, so every production relock is byte-unaffected. Got: $A5AR"
  printf '%s\n' "$A5A" | sed 's/^/###     /'
  fail=1
fi

# 5b DECLARED SHAPE: the filter is exported AND the -v is dropped. Both, or the
# spans still never print -- that is the whole lesson of 6015646 W1.
A5B=$( set +e
  # shellcheck source=/dev/null
  . "$FAST" >/dev/null 2>&1
  retread_relock_frontend_log --cold-proof-arm "--frontend-rust-log=$A5F"
  echo "RESULT rust_log=${RUST_LOG-<unset>} lock_verbosity=${LOCK_VERBOSITY-<UNSET-VAR>}" )
A5BR=$(printf '%s\n' "$A5B" | grep -F 'RESULT ')
if [ "$A5BR" = "RESULT rust_log=$A5F lock_verbosity=" ] \
   && printf '%s\n' "$A5B" | grep -qF "### INSTRUMENTATION: frontend RUST_LOG=$A5F" \
   && printf '%s\n' "$A5B" | grep -qF '### INSTRUMENTATION: lock verbosity flag DROPPED'; then
  say "ARM5b PASS --frontend-rust-log= exports the filter AND empties LOCK_VERBOSITY ($A5BR)"
else
  say "ARM5b FAIL the declared filter must reach RUST_LOG *and* drop pixi's -v (pixi's own -v overrides RUST_LOG). Got: $A5BR"
  printf '%s\n' "$A5B" | sed 's/^/###     /'
  fail=1
fi

# 5c MUTATION. Delete the argv parse from a COPY of the producer and 5b must go
# red. A guard that cannot fail is a defect, so this arm fails when the mutant
# PASSES.
A5MUT=$W/fast_env_mutant.sh
sed 's/^\([[:space:]]*\)case "\$a" in --frontend-rust-log=.*$/\1: # ARM5c MUTATION: argv parse deleted/' "$FAST" > "$A5MUT"
if cmp -s "$FAST" "$A5MUT"; then
  say "ARM5c FAIL the mutation edited nothing -- this arm is asserting against an unmutated file and cannot fail"
  fail=1
else
  A5C=$( set +e
    # shellcheck source=/dev/null
    . "$A5MUT" >/dev/null 2>&1
    retread_relock_frontend_log --cold-proof-arm "--frontend-rust-log=$A5F"
    echo "RESULT rust_log=${RUST_LOG-<unset>} lock_verbosity=${LOCK_VERBOSITY-<UNSET-VAR>}" )
  A5CR=$(printf '%s\n' "$A5C" | grep -F 'RESULT ')
  if [ "$A5CR" = "RESULT rust_log=$A5F lock_verbosity=" ]; then
    say "ARM5c FAIL the mutant with NO argv parse still produced the declared filter -- 5b proves nothing. Got: $A5CR"
    fail=1
  else
    say "ARM5c PASS (mutation) argv parse deleted -> the declared filter does NOT appear ($A5CR)"
  fi
fi

# 5d PER TARGET, ADOPTION. A template that adopted the call must NOT also keep a
# hardcoded `lock -v`: half an adoption is worse than none, because the row says
# the filter was set while pixi quietly overrides it. A template that has not
# adopted it is REPORTED, not failed -- the back-port is boarded debt, named in
# the row -- but a template that adopted it and kept `-v` is a hard failure.
for TPL in $TARGETS; do
  TN=$(basename "$TPL")
  [ -f "$TPL" ] || continue
  ADOPT=$(grep -cE '^[[:space:]]*retread_relock_frontend_log[[:space:]]' "$TPL")
  HARDV=$(grep -cE '"\$PIXI" lock[[:space:]]+-v' "$TPL")
  # THE SECOND HALF-ADOPTION, AND IT IS THE ONE THE INSTRUMENTED TEMPLATES
  # ACTUALLY HAD. A template can adopt the call, pass `$LOCK_VERBOSITY` to pixi,
  # and STILL lie -- by re-assigning LOCK_VERBOSITY after the producer set it.
  # `p6b_relock.sh` shipped `LOCK_VERBOSITY=-vvv` beside an exported
  # FRONTEND_RUST_LOG and a row announcing that filter, and pixi's own -vvv
  # overrode it: the row was false for as long as that template has existed.
  # So in an ADOPTING template the producer must be the LAST writer -- any
  # `LOCK_VERBOSITY=` assignment below the call line is a second producer.
  CALLLINE=$(grep -nE '^[[:space:]]*retread_relock_frontend_log[[:space:]]' "$TPL" | head -1 | cut -d: -f1)
  LVAFTER=0
  [ -n "$CALLLINE" ] && LVAFTER=$(awk -v c="$CALLLINE" 'NR>c && /^[[:space:]]*LOCK_VERBOSITY=/ {n++} END{print n+0}' "$TPL")
  if [ "$ADOPT" -ge 1 ] && [ "$HARDV" = 0 ] && [ "$LVAFTER" != 0 ]; then
    say "ARM5d FAIL $TN calls retread_relock_frontend_log and then RE-ASSIGNS LOCK_VERBOSITY $LVAFTER time(s) below the call -- a second producer of the one value the call exists to own, and the announced filter would be overridden in silence"
    fail=1
  elif [ "$ADOPT" -ge 1 ] && [ "$HARDV" = 0 ]; then
    say "ARM5d PASS $TN calls retread_relock_frontend_log, passes \$LOCK_VERBOSITY to pixi lock (no hardcoded -v), and never re-assigns it below the call"
  elif [ "$ADOPT" -ge 1 ] && [ "$HARDV" != 0 ]; then
    say "ARM5d FAIL $TN calls retread_relock_frontend_log but STILL hardcodes 'pixi lock -v' ($HARDV line(s)) -- pixi's -v overrides RUST_LOG, so the filter it just announced is a lie"
    fail=1
  elif grep -qE '^[[:space:]]*unset RUST_LOG[[:space:]]*$' "$TPL"; then
    say "ARM5d NOT-ADOPTED $TN still does 'unset RUST_LOG' + 'pixi lock -v' and cannot log uv_distribution spans. BOARDED DEBT (DET-1-6-3): back-port the retread_relock_frontend_log call, one line after its retread_relock_scope_and_verify call, and swap 'lock -v' for 'lock \$LOCK_VERBOSITY'. Sized: 2 lines per template, $TN."
  else
    say "ARM5d NOT-ADOPTED $TN (no 'unset RUST_LOG' of its own; nothing to back-port yet)"
  fi
done

say "TARGETS checked=$napp"
[ "$napp" -ge 1 ] || { say "REFUSED no target -- this guard would be green against nothing"; exit 3; }
[ "$fail" = 0 ] && { say "ALL ARMS PASS"; exit 0; }
say "REFUSED"; exit 1
