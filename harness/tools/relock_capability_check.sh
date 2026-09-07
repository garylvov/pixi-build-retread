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
#          capabilities: --cold-proof-arm      --frontend-rust-log      --env-seed
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
    --cold-proof-arm|--frontend-rust-log|--env-seed) CAP=$1 ;;
    -*) echo "### RELOCK CAPABILITY REFUSED: unknown capability '$1' (known: --cold-proof-arm --frontend-rust-log --env-seed)" >&2; exit 2 ;;
    *)  TPL=$1 ;;
  esac
  shift
done
[ -n "$CAP" ] && [ -n "$TPL" ] || {
  echo "### RELOCK CAPABILITY REFUSED: usage: relock_capability_check.sh <--cold-proof-arm|--frontend-rust-log|--env-seed> <relock template>" >&2
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
--env-seed)
  # HALF 1 IS DIFFERENT IN SHAPE HERE, AND IT HAS TO BE. The other two
  # capabilities are "does the template FORWARD ITS ARGV to the producer"; this
  # one is "does the template SOURCE the one authority and call it in STRICT
  # mode over its backend". There is no flag to forward -- the seed is asked of
  # the binary, never typed -- so the argv test would be vacuous and is replaced
  # by the three things that actually make the export travel.
  FN=env_seed_export
  ESL=${RELOCK_ENV_SEED_LIB:-$HERE/env_seed.sh}
  [ -f "$ESL" ] || say_no "no env_seed.sh at $ESL -- the producer cannot be executed, and a capability answered without executing it is the grep this file replaces"
  # (1) the template SOURCES the library rather than carrying a literal seed. A
  #     literal here is a second authority that drifts from the backend constant.
  grep -qE '^[[:space:]]*\.[[:space:]]+"\$ENV_SEED_LIB"' "$TPL" \
    || say_no "the template never sources \$ENV_SEED_LIB -- tools/env_seed.sh is the ONE authority and a wrapper that does not source it either has no seed or has a second one"
  grep -qE '^[[:space:]]*(export[[:space:]]+)?PYTHONHASHSEED=' "$TPL" \
    && say_no "the template ASSIGNS PYTHONHASHSEED itself -- a literal seed in a wrapper is exactly the second authority DET-1-6-1 removed; the value is asked of the binary"
  # (2) it CALLS the function, over a binary, in STRICT mode. `optional` is for
  #     control arms whose binaries predate the verb; a relock wrapper that means
  #     to certify a lock must not lock under a random seed.
  CALL=$(grep -nE "^[[:space:]]*$FN[[:space:]]" "$TPL" | head -1)
  [ -n "$CALL" ] || say_no "the template never calls $FN, so the pinned seed never reaches the shell that launches pixi -- the DET-1-4-1 defect, unclosed"
  printf '%s' "$CALL" | grep -qE "$FN[[:space:]]+\"\\\$[A-Za-z_][A-Za-z0-9_]*\"" \
    || say_no "the template calls $FN without passing a backend binary -- the seed would be asked of nothing ($CALL)"
  printf '%s' "$CALL" | grep -q 'optional' \
    && say_no "the template calls $FN in OPTIONAL mode -- a verb-less binary would then lock under the ambient seed, which is the control-arm contract and not a relock's"
  # (3) it REFUSES on failure. A wrapper that exports nothing and locks anyway is
  #     the defect wearing the fix's call site.
  printf '%s' "$CALL" | grep -qE '\|\|[[:space:]]*(exit|return)[[:space:]]' \
    || say_no "the template calls $FN but does NOT refuse when it fails ($CALL) -- locking on after a failed seed export is the exact defect this block exists to close"

  # ---- HALF 2: the producer is EXECUTED, in BOTH shapes ---------------------
  # Two fixture "binaries": one carrying the verb marker and answering `env-seed`,
  # one that predates the verb. A capability answered by grepping the template
  # alone would pass against a library whose function had been gutted.
  W=$(mktemp -d "${TMPDIR:-/tmp}/relockcap-seed.XXXXXX") || exit 2
  trap 'rm -rf "$W"' EXIT
  ( . "$ESL" >/dev/null 2>&1; printf '%s' "${ENV_SEED_MARKER:-}" ) > "$W/marker" 2>/dev/null
  MARK=$(cat "$W/marker")
  [ -n "$MARK" ] || say_no "env_seed.sh defines no ENV_SEED_MARKER -- the verb is detected by that marker, and without it the producer cannot tell a new binary from an old one"
  printf '#!/usr/bin/env bash\n# %s\n[ "${1:-}" = env-seed ] && { echo 0; exit 0; }\nexit 0\n' "$MARK" > "$W/newbin"
  printf '#!/usr/bin/env bash\nexit 0\n' > "$W/oldbin"
  chmod +x "$W/newbin" "$W/oldbin"
  NEWOUT=$( set +e; . "$ESL" >/dev/null 2>&1
            unset PYTHONHASHSEED
            "$FN" "$W/newbin" >/dev/null 2>&1; r=$?
            echo "rc=$r seed=${PYTHONHASHSEED-<unset>}" )
  [ "$NEWOUT" = 'rc=0 seed=0' ] \
    || say_no "over a verb-carrying binary the producer did not export the pinned seed -- it printed '$NEWOUT', wanted 'rc=0 seed=0'"
  STRICTOUT=$( set +e; . "$ESL" >/dev/null 2>&1
               unset PYTHONHASHSEED
               "$FN" "$W/oldbin" >/dev/null 2>&1; r=$?
               echo "rc=$r seed=${PYTHONHASHSEED-<unset>}" )
  [ "$STRICTOUT" = 'rc=15 seed=<unset>' ] \
    || say_no "over a VERB-LESS binary STRICT mode did not refuse -- it printed '$STRICTOUT', wanted 'rc=15 seed=<unset>'; a strict mode that passes anything is not a mode"
  OPTOUT=$( set +e; . "$ESL" >/dev/null 2>&1
            unset PYTHONHASHSEED
            "$FN" "$W/oldbin" optional >/dev/null 2>&1; r=$?
            echo "rc=$r seed=${PYTHONHASHSEED-<unset>}" )
  [ "$OPTOUT" = 'rc=0 seed=<unset>' ] \
    || say_no "OPTIONAL mode over a verb-less binary is not today's behaviour ('$OPTOUT', wanted 'rc=0 seed=<unset>') -- the control arms depend on it running unseeded rather than refusing"
  say_yes "the template sources tools/env_seed.sh, calls $FN over a backend in STRICT mode and refuses on failure, and the producer exports the pin for a verb-carrying binary, refuses a verb-less one in strict, and runs it unseeded in optional"
  ;;
esac
