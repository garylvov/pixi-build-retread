#!/usr/bin/env bash
# relock_capability_check.sh -- ONE READER for the question every proof driver
# asks and every one of them has answered by hand: "does this relock template
# support <capability>?"
#
# THE DEFECT (DET-1-6-3 finding 1). det162_proof.sh's `--cold-proof-arm` gate
# answers that question with
#
#     grep -c -F 'sdist scoping SKIPPED' "$RELOCK_SRC"
#
# and at this tip that count is ZERO -- not because the capability is gone but
# because HARNESS-CONSOL-9 MOVED the branch out of every template and into ONE
# producer, `retread_relock_scope_and_verify` in tools/retread_fast_env.sh. A
# verbatim copy of that gate would print `### FATAL the worktree phaseN_relock.sh
# has NO --cold-proof-arm branch` against a CORRECT tree. A driver gate coupled
# to a FILE rather than to a BEHAVIOUR goes stale the first time the behaviour
# moves, and it fails in the worst direction: it refuses good trees, so the next
# lane's instinct is to weaken the gate.
#
# WHAT THIS CHECKS INSTEAD -- THE PAIR THAT ACTUALLY HAS TO HOLD.
#   1. THE PRODUCER IS LIVE. The function is SOURCED and CALLED, in both shapes,
#      and its real rows are read. Nothing here greps for the logic it expects:
#      a file that says the right thing and does the wrong one is precisely what
#      DET-1-6-3 spent a lane on.
#   2. THE TEMPLATE REACHES IT. The template calls the function AND FORWARDS AN
#      ARGV to it -- a call that drops the argv silently loses the flag, which is
#      the same defect one level down.
# Either half alone is green on a tree that cannot do the thing.
#
#   usage: relock_capability_check.sh <--capability> <relock template path>
#          capabilities: --cold-proof-arm      --frontend-rust-log
#
#   Prints  ### RELOCK CAPABILITY <cap> template=<name> = YES|NO (<why>)
#   rc 0    the template can do it;  rc 1 it cannot;  rc 2 usage/environment.
#
# FOR DRIVER AUTHORS. This replaces the hand-rolled grep, and it is the ONLY
# thing a driver should ask:
#
#     bash "$T/tools/relock_capability_check.sh" --cold-proof-arm "$RELOCK_SRC" \
#       || { echo "### FATAL this relock template cannot declare a cold proof arm"; exit 2; }
#
# Its own reader is tools/relock_capability_check_guard.sh, which runs it against
# the shipped templates AND against fixture templates with the call removed or
# the argv dropped, and mutates the producer.
set -uo pipefail

CAP=; TPL=
while [ "$#" -gt 0 ]; do
  case $1 in
    --cold-proof-arm|--frontend-rust-log) CAP=$1 ;;
    -*) echo "### RELOCK CAPABILITY REFUSED: unknown capability '$1' (known: --cold-proof-arm --frontend-rust-log)" >&2; exit 2 ;;
    *)  TPL=$1 ;;
  esac
  shift
done
[ -n "$CAP" ] && [ -n "$TPL" ] || {
  echo "### RELOCK CAPABILITY REFUSED: usage: relock_capability_check.sh <--cold-proof-arm|--frontend-rust-log> <relock template>" >&2
  exit 2; }
[ -f "$TPL" ] || { echo "### RELOCK CAPABILITY REFUSED: no such template: $TPL" >&2; exit 2; }

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FAST=${RELOCK_FAST_ENV:-$HERE/retread_fast_env.sh}
[ -f "$FAST" ] || {
  echo "### RELOCK CAPABILITY REFUSED: no retread_fast_env.sh at $FAST -- the producer cannot be executed, and a capability answered without executing it is the grep this file replaces" >&2
  exit 2; }
TN=$(basename -- "$TPL")

say_no () { echo "### RELOCK CAPABILITY $CAP template=$TN = NO ($1)"; exit 1; }
say_yes () { echo "### RELOCK CAPABILITY $CAP template=$TN = YES ($1)"; exit 0; }

# ---- HALF 1: the template reaches the producer, WITH an argv -----------------
# `<fn> "$C" "$@"`, `<fn> "$@"`, `<fn> "$@" "--flag=..."`, `<fn> "${ARR[@]}"` all
# forward; `<fn> "$C"` alone does not, and that is a real shape -- a copy that
# drops the argv keeps the call and loses the flag.
reaches () {                 # $1 = function name -> rc 0 if called WITH an argv
  local fn=$1 call
  call=$(grep -nE "^[[:space:]]*$fn([[:space:]]|\$)" "$TPL" | head -1)
  [ -n "$call" ] || return 2
  printf '%s' "$call" | grep -qE '"\$@"|"\$\{[A-Za-z_][A-Za-z0-9_]*\[@\]\}"' || return 3
  return 0
}

# ---- HALF 2: the producer is EXECUTED, in both shapes ------------------------
case $CAP in
--cold-proof-arm)
  FN=retread_relock_scope_and_verify
  reaches "$FN"; r=$?
  [ "$r" = 2 ] && say_no "the template never calls $FN, so nothing it is passed can reach the flag"
  [ "$r" = 3 ] && say_no "the template calls $FN WITHOUT forwarding an argv -- the call is there and the flag can never arrive"
  # The flag's whole observable effect is that the scoper is SKIPPED and said so.
  # Run it in a throwaway job root against a shape that would otherwise refuse.
  W=$(mktemp -d "${TMPDIR:-/tmp}/relockcap.XXXXXX") || exit 2
  trap 'rm -rf "$W"' EXIT
  ON=$( set +e; . "$FAST" >/dev/null 2>&1
        UV_CACHE_DIR=$W/uv PIXI_CACHE_DIR=$W/pixi \
          "$FN" "$W/jr" --cold-proof-arm 2>&1 )
  OFF=$( set +e; . "$FAST" >/dev/null 2>&1
         UV_CACHE_DIR=$W/uv PIXI_CACHE_DIR=$W/pixi \
           "$FN" "$W/jr" 2>&1 )
  printf '%s\n' "$ON" | grep -q 'sdist scoping SKIPPED (declared cold proof arm)' \
    || say_no "$FN is defined but the flag does NOT skip the scoping -- the producer's branch is gone, and the template's call cannot make up for it"
  if printf '%s\n' "$OFF" | grep -q 'sdist scoping SKIPPED'; then
    say_no "$FN skips the scoping with NO flag on the argv -- it skips unconditionally, so 'supports the flag' would be true of a function that ignores it"
  fi
  say_yes "the template forwards an argv to $FN, the flag makes it print the SKIPPED row, and without the flag it does not"
  ;;
--frontend-rust-log)
  FN=retread_relock_frontend_log
  reaches "$FN"; r=$?
  [ "$r" = 2 ] && say_no "the template never calls $FN, so the frontend filter has no producer here and any vacuity assertion over uv_distribution spans would read absent, not fail"
  [ "$r" = 3 ] && say_no "the template calls $FN WITHOUT forwarding an argv -- --frontend-rust-log= can never arrive"
  # The other half of the ONE control: pixi's own -v overrides RUST_LOG, so a
  # template that hardcodes it cannot log the spans however good the filter is.
  grep -qE '"\$PIXI" lock[[:space:]]+-v' "$TPL" \
    && say_no "the template hardcodes 'pixi lock -v', which sets pixi's own tracing filter and OVERRIDES RUST_LOG -- the filter it announces would be a lie"
  CALLLINE=$(grep -nE "^[[:space:]]*$FN([[:space:]]|\$)" "$TPL" | head -1 | cut -d: -f1)
  LVAFTER=$(awk -v c="$CALLLINE" 'NR>c && /^[[:space:]]*LOCK_VERBOSITY=/ {n++} END{print n+0}' "$TPL")
  [ "$LVAFTER" = 0 ] \
    || say_no "the template re-assigns LOCK_VERBOSITY $LVAFTER time(s) BELOW the $FN call -- a second producer of the value the call exists to own"
  F='uv_distribution=debug,warn'
  DECL=$( set +e; . "$FAST" >/dev/null 2>&1
          "$FN" "--frontend-rust-log=$F" >/dev/null 2>&1
          echo "rust_log=${RUST_LOG-<unset>} lock_verbosity=${LOCK_VERBOSITY-<UNSET-VAR>}" )
  DEF=$( set +e; . "$FAST" >/dev/null 2>&1
         "$FN" >/dev/null 2>&1
         echo "rust_log=${RUST_LOG-<unset>} lock_verbosity=${LOCK_VERBOSITY-<UNSET-VAR>}" )
  [ "$DECL" = "rust_log=$F lock_verbosity=" ] \
    || say_no "the producer did not export the declared filter AND empty the verbosity flag -- it printed '$DECL'; BOTH or the spans still never print"
  [ "$DEF" = 'rust_log=<unset> lock_verbosity=-v' ] \
    || say_no "with NO flag the producer is not today's behaviour ('$DEF', wanted rust_log=<unset> lock_verbosity=-v) -- adopting it would change every production relock"
  say_yes "the template forwards an argv to $FN, passes \$LOCK_VERBOSITY to pixi lock and never re-assigns it, and the producer exports the filter and drops the -v"
  ;;
esac
