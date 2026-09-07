#!/usr/bin/env bash
# cold_proof_arm_guard.sh -- THE READER OF phaseN_relock.sh's `--cold-proof-arm`.
#
# DET-1-6-2. `retread_scope_sdist_builds` refuses when it symlinked no
# byte-keyed bucket, which is exactly the state of a DECLARED-COLD PROOF ARM's
# own empty `RETREAD_PERSIST_CACHE_ROOT` (measured: job 6014471 arm W1,
# `symlinked=0 ... shared_links_seen=0` then `FATAL no byte-keyed bucket was
# symlinked`). `phaseN_relock.sh` therefore takes `--cold-proof-arm` on ARGV and
# prints `### stage: sdist scoping SKIPPED (declared cold proof arm)` instead of
# calling the scoper. THIS guard is the criterion's live producer:
#
#   ARM 1  the REAL scoper, real empty shared cache  -> MUST refuse (rc != 0)
#          and MUST print the `no byte-keyed bucket` FATAL. If this arm goes
#          green the flag is solving a problem that no longer exists. GLOBAL:
#          it reads retread_fast_env.sh, not a template.
#   ARM 2  the template's flag block, argv WITH the flag  -> MUST print SKIPPED
#          and MUST NOT call the scoper.
#   ARM 3  MUTATION ARM. the same block, argv WITHOUT the flag -> MUST NOT print
#          SKIPPED. A guard whose arm 2 passes with arm 3 also passing is not a
#          guard: it would pass a block that prints SKIPPED unconditionally.
#   ARM 4  the flag block is ACTUALLY PRESENT in the template it claims to
#          guard. Without this the whole file can go green against nothing.
#
# ── HARNESS-CONSOL-8, AND A PREMISE FALSIFIED BEFORE IT WAS IMPLEMENTED ──────
# This guard was handed a single template path. It is a TARGETS LIST now, for
# the reason item 3 of that lane established: `arms/mh1_relock.sh` is the
# DERIVATION SOURCE for every merge lane's relock script, so a transformation
# that reaches only `phase_template/phaseN_relock.sh` reaches ONE of the relock
# scripts this campaign actually runs.
#
# BUT THE FLAG MUST NOT SIMPLY BE ADDED TO THE OTHERS, AND THE ROWS SAY WHY.
# MEASURED 2026-09-07 by `grep -c` on all four relock templates in the tree:
#
#     phase_template/phaseN_relock.sh   scoper=3  poison_guard=2
#     arms/mh1_relock.sh                scoper=0  poison_guard=0
#     arms/c29_relock.sh                scoper=0  poison_guard=0
#     proof/hlgd_relock.sh              scoper=0  poison_guard=0
#
# THREE OF THE FOUR NEVER CALL `retread_scope_sdist_builds` AT ALL, and never
# run `sdist_build_poison_guard.sh` either. A `--cold-proof-arm` branch in a
# file with no scoper to skip is a flag with no producer -- the same defect as a
# stamped-but-unread directive, in the other direction. So the guard CLASSIFIES
# each target instead of demanding the flag of all of them:
#
#   * a target that CALLS the scoper MUST carry the flag and is put through
#     arms 2, 3 and 4 in full;
#   * a target that does NOT call the scoper is NOT-APPLICABLE and says so on
#     its own row -- and that row is the READER of the divergence. The day
#     mh1_relock.sh (or c29, or hlgd) gains the C31-4 scoper, this guard flips
#     it to APPLICABLE and demands the flag in the same run. A silent skip would
#     let the back-port land without one.
#
# That divergence is BOARDED, not fixed here: back-porting C31-4's scoper and
# its poison guard into the template seven live merge lanes derive from is not a
# guard's change to make, and not one to make while those lanes are running.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
FAST=$HERE/retread_fast_env.sh
# Explicit targets win (the caller passes paths); otherwise the shipped set.
if [ "$#" -gt 0 ]; then
  TARGETS="$*"
else
  TARGETS=
  for t in "$HERE/../phase_template/phaseN_relock.sh" "$HERE/../arms/mh1_relock.sh" \
           "$HERE/../arms/c29_relock.sh" "$HERE/../proof/hlgd_relock.sh"; do
    [ -f "$t" ] && TARGETS="$TARGETS $t"
  done
fi
W=$(mktemp -d "${TMPDIR:-/tmp}/coldproof.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
fail=0
say () { echo "### COLD-PROOF-ARM GUARD $*"; }

[ -n "${TARGETS// /}" ] || { say "FATAL no relock template to read"; exit 3; }

# ---- ARM 1 (GLOBAL): the refusal the flag exists to bypass, reproduced live --
if [ -f "$FAST" ]; then
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
else
  say "ARM1 FAIL retread_fast_env.sh not found next to this guard ($FAST)"; fail=1
fi

# ---- ARMS 2-4, once per target, applicability decided from the file ---------
napp=0; nna=0
for TPL in $TARGETS; do
  TN=$(basename "$TPL")
  if [ ! -f "$TPL" ]; then say "ARM4 FAIL template absent: $TPL"; fail=1; continue; fi
  NSCOPE=$(grep -c 'retread_scope_sdist_builds' "$TPL")
  if [ "$NSCOPE" = 0 ]; then
    nna=$((nna + 1))
    say "ARM4 N/A $TN never calls retread_scope_sdist_builds (scoper=0) -- there is nothing for --cold-proof-arm to skip, so the flag is NOT required here. BOARDED: this template did not receive C31-4's scoper or its sdist poison guard. If it ever does, this row flips to APPLICABLE and arms 2-4 run."
    if grep -q -- '--cold-proof-arm' "$TPL"; then
      say "ARM4 FAIL $TN carries --cold-proof-arm but no scoper -- a flag with no producer"; fail=1
    fi
    continue
  fi
  napp=$((napp + 1))
  if grep -q -- '--cold-proof-arm' "$TPL" && grep -q 'sdist scoping SKIPPED (declared cold proof arm)' "$TPL"; then
    say "ARM4 PASS $TN calls the scoper (scoper=$NSCOPE) and carries the flag and its row"
  else
    say "ARM4 FAIL $TN CALLS the scoper but carries no --cold-proof-arm branch -- a cold proof arm through it dies at the refusal"; fail=1; continue
  fi

  # The block is lifted from the template by its own two markers, so this guard
  # reads the SHIPPED text and not a copy that can drift away from it.
  awk '/^# --- `--cold-proof-arm`/{s=1} s{print} s&&/^fi$/{exit}' "$TPL" > "$W/block.$TN.sh"
  if ! grep -q 'COLD_PROOF_ARM=0' "$W/block.$TN.sh"; then
    say "ARM2/3 FAIL could not lift the flag block out of $TN"; fail=1; continue
  fi
  cat > "$W/harness.$TN.sh" <<'EOS'
C=/nonexistent/job/root
UV_CACHE_DIR=/nonexistent/uv
PIXI_CACHE_DIR=/nonexistent/pixi
retread_scope_sdist_builds () { echo "SCOPER-WAS-CALLED"; return 0; }
EOS
  cat "$W/block.$TN.sh" >> "$W/harness.$TN.sh"
  bash "$W/harness.$TN.sh" TAG /root --cold-proof-arm > "$W/arm2.$TN.out" 2>&1; A2RC=$?
  bash "$W/harness.$TN.sh" TAG /root                  > "$W/arm3.$TN.out" 2>&1; A3RC=$?
  if [ "$A2RC" = 0 ] && grep -q 'sdist scoping SKIPPED (declared cold proof arm)' "$W/arm2.$TN.out" \
     && ! grep -q 'SCOPER-WAS-CALLED' "$W/arm2.$TN.out"; then
    say "ARM2 PASS $TN with the flag: SKIPPED row printed and the scoper was NOT called"
  else
    say "ARM2 FAIL $TN (rc=$A2RC):"; sed 's/^/###   /' "$W/arm2.$TN.out"; fail=1
  fi
  if ! grep -q 'sdist scoping SKIPPED' "$W/arm3.$TN.out" && grep -q 'SCOPER-WAS-CALLED' "$W/arm3.$TN.out"; then
    say "ARM3 PASS $TN (mutation) without the flag: no SKIPPED row and the scoper WAS called"
  else
    say "ARM3 FAIL $TN (rc=$A3RC) the block behaves the same with and without the flag -- it is not a gate:"
    sed 's/^/###   /' "$W/arm3.$TN.out"; fail=1
  fi
done

say "TARGETS applicable=$napp not_applicable=$nna"
[ "$napp" -ge 1 ] || { say "REFUSED no target calls the scoper -- this guard would be green against nothing"; exit 3; }
[ "$fail" = 0 ] && { say "ALL ARMS PASS"; exit 0; }
say "REFUSED"; exit 1
