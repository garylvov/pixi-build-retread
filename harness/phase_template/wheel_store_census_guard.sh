#!/usr/bin/env bash
# wheel_store_census_guard.sh -- the reader for the "which wheel store did the
# lock actually read" census line.
#
# THE DEFECT IT WOULD HAVE CAUGHT. The relock harnesses carried a block that
# announced a JOB-SCOPED wheel store and then, since p6i re-enabled the SHARED
# export at 15:35 on 09-03, printed "unset" followed by the shared path. It was
# a dead letter: nothing in it called the seed, and every proof after p6i read
# the fill-lock-poisoned SHARED store while its log said the store was isolated.
# A log line that names the wrong store is worse than no line, because the
# forensics on top of it are all wrong too.
#
# WHAT IT ASSERTS, against the REAL block and the REAL census function lifted
# out of each shipped harness:
#
#   A. WHEEL_STORE_SEED UNSET  -> the census names the SHARED store, scope=SHARED,
#                                 and no job-scoped directory is created.
#   B. WHEEL_STORE_SEED SET    -> the block calls the real
#                                 `retread_seed_wheel_store`, the store is
#                                 TRULY job-scoped (its wheels are separate
#                                 inodes from the shared store's), and the
#                                 census names the JOB-SCOPED path.
#   C. NEGATIVE CONTROL: with the seed set but the export of
#                        RETREAD_WHEEL_STORE removed -- the exact shape of the
#                        dead letter, a claim without an effect -- the census
#                        MUST fall back to naming the SHARED store. If it still
#                        said JOB-SCOPED, the line would be reading the harness's
#                        intent instead of the resolved path, and A and B would
#                        both be vacuous.
#
# The census resolves the store the way `courier::wheel_store_root_with` does
# (RETREAD_WHEEL_STORE -> XDG_CACHE_HOME -> HOME/.cache, then retread/wheels),
# at the point of use. That is what makes C fail rather than pass.
#
# Falsification: derive the scope word from $WHEEL_STORE_SEED instead of from
# the resolved path and C goes RED; drop the `export RETREAD_WHEEL_STORE=` from
# the seed branch and B goes RED.
#
# Usage: wheel_store_census_guard.sh        (self-contained, needs only $TMPDIR)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
# Every harness PRESENT is checked; one that is not shipped in this checkout is
# skipped by name, never silently.
TARGETS=$HERE/phaseN_relock.sh
for extra in "$ROOT/proof/hlgd_relock.sh" "$ROOT/../p18-hardlink-guard/hlgd_relock.sh"; do
  [ -f "$extra" ] && TARGETS="$TARGETS $extra"
done
# harness/ tree: tools/retread_fast_env.sh. Task checkout: the templates sit in
# <task>/tools/phase_template/, so the fast-env is one level up from HERE.
FAST_ENV=
for c in "$ROOT/tools/retread_fast_env.sh" "$ROOT/retread_fast_env.sh"; do
  [ -f "$c" ] && { FAST_ENV=$c; break; }
done
[ -n "$FAST_ENV" ] || { echo "GUARD FATAL: no retread_fast_env.sh under $ROOT"; exit 2; }

W=$(mktemp -d "${TMPDIR:-/tmp}/wheel-store-census-guard.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
FAIL=0
fail () { echo "GUARD FAIL: $*"; FAIL=1; }
ok   () { echo "GUARD  ok : $*"; }

########## a fixture shared store, shaped like the real one ####################
PERSIST=$W/persist
SHARED=$PERSIST/wheels
mkdir -p "$SHARED/aaa111" "$SHARED/bbb222"
printf 'wheel-one\n' > "$SHARED/aaa111/pkg_one-1.0-py3-none-any.whl"
printf 'wheel-two\n' > "$SHARED/bbb222/pkg_two-2.0-py3-none-any.whl"
# the poison the seed exists to escape -- it must NOT reach a job-scoped store
: > "$SHARED/bbb222/.pkg_two-2.0-py3-none-any.whl.retread-fill-v1.lock"
SHARED_INO=$(stat -c %i "$SHARED/aaa111/pkg_one-1.0-py3-none-any.whl")

extract () {  # $1=file $2=function name -> the function text, verbatim
  awk -v fn="$2" '$0 ~ "^"fn" \\(\\) \\{" {p=1} p {print} p && /^\}$/ {exit}' "$1"
}
extract_seed_block () {  # the real `if [ -n "${WHEEL_STORE_SEED..." ... fi block
  awk '/^if \[ -n "\$\{WHEEL_STORE_SEED:-\}" \]; then/ {p=1} p {print} p && /^fi$/ {exit}' "$1"
}

# what a census line claims, parsed out of the harness's own stdout
census_field () { printf '%s\n' "$1" | sed -n 's/.*WHEEL STORE IN USE ([^)]*): //p' \
                                     | tr ' ' '\n' | sed -n "s/^$2=//p" | head -1; }

for TPL in $TARGETS; do
  NAME=$(basename "$TPL")
  echo "GUARD: == $NAME =="
  [ -f "$TPL" ] || { fail "$NAME: not found at $TPL"; continue; }

  BLK=$(extract_seed_block "$TPL")
  IN_USE=$(extract "$TPL" wheel_store_in_use)
  CENSUS=$(extract "$TPL" wheel_store_census)
  [ -n "$BLK" ]    || { fail "$NAME: no WHEEL_STORE_SEED block -- a lane that sets it is ignored silently"; continue; }
  [ -n "$IN_USE" ] || { fail "$NAME: no wheel_store_in_use -- nothing resolves the store at the point of use"; continue; }
  [ -n "$CENSUS" ] || { fail "$NAME: no wheel_store_census -- the lock log never says which store it read"; continue; }
  case $BLK in
    *retread_seed_wheel_store*) ok "$NAME: the seed branch calls retread_seed_wheel_store (not a dead letter)" ;;
    *) fail "$NAME: the WHEEL_STORE_SEED branch does NOT call retread_seed_wheel_store -- it is a claim with no effect" ;;
  esac

  DRV=$W/$NAME.drv.sh
  { echo 'set -u'
    echo '. "$FAST_ENV_PATH"'                 # the real retread_seed_wheel_store
    echo 'C=$JOBROOT'                          # the block's job-scoped cache root
    printf '%s\n' "$BLK"
    printf '%s\n' "$IN_USE"
    printf '%s\n' "$CENSUS"
    echo "wheel_store_census 'GUARD'"
  } > "$DRV"

  # ---- A. seed UNSET -> the census must name the SHARED store ---------------
  JR=$W/$NAME.a; mkdir -p "$JR"
  OUT=$(FAST_ENV_PATH=$FAST_ENV JOBROOT=$JR \
        RETREAD_PERSIST_CACHE_ROOT=$PERSIST RETREAD_WHEEL_STORE=$SHARED \
        bash "$DRV" 2>&1)
  if [ "$(census_field "$OUT" scope)" = SHARED ] && [ "$(census_field "$OUT" path)" = "$SHARED" ] \
     && [ ! -d "$JR/wheels-seeded" ]; then
    ok "$NAME: A. seed unset -> scope=SHARED path=$SHARED, no job-scoped store made"
  else
    fail "$NAME: A. seed unset but the census does not name the shared store"
    printf '%s\n' "$OUT" | tail -4 | sed 's/^/GUARD:   /'
  fi

  # ---- B. seed SET -> truly job-scoped, and the census says so --------------
  JR=$W/$NAME.b; mkdir -p "$JR"
  OUT=$(FAST_ENV_PATH=$FAST_ENV JOBROOT=$JR WHEEL_STORE_SEED=$SHARED \
        RETREAD_PERSIST_CACHE_ROOT=$PERSIST RETREAD_WHEEL_STORE=$SHARED \
        bash "$DRV" 2>&1)
  JOBSTORE=$JR/wheels-seeded
  SEEDED_WHL=$JOBSTORE/aaa111/pkg_one-1.0-py3-none-any.whl
  if [ "$(census_field "$OUT" scope)" = JOB-SCOPED ] \
     && [ "$(census_field "$OUT" path)" = "$JOBSTORE" ] \
     && [ -f "$SEEDED_WHL" ] \
     && [ "$(stat -c %i "$SEEDED_WHL")" != "$SHARED_INO" ] \
     && [ "$(find "$JOBSTORE" -name '.*.retread-fill-v1.lock' | wc -l)" = 0 ]; then
    ok "$NAME: B. seed set -> scope=JOB-SCOPED path=$JOBSTORE, own inodes, 0 fill locks"
  else
    fail "$NAME: B. WHEEL_STORE_SEED did not produce a real job-scoped store the census names"
    printf '%s\n' "$OUT" | tail -6 | sed 's/^/GUARD:   /'
  fi

  # ---- C. NEGATIVE CONTROL: the dead letter -- claim without effect ---------
  # Strip the one line that makes the seed take effect. The block still PRINTS
  # "job-scoped"; the census must still report SHARED, because it resolves the
  # store instead of believing the harness.
  DEAD=$W/$NAME.dead.sh
  grep -v '^  export RETREAD_WHEEL_STORE=\$WHEEL_STORE_JOBSCOPED$' "$DRV" > "$DEAD"
  if cmp -s "$DRV" "$DEAD"; then
    fail "$NAME: C. could not build the dead-letter control (the export line moved) -- control is vacuous"
  else
    JR=$W/$NAME.c; mkdir -p "$JR"
    OUT=$(FAST_ENV_PATH=$FAST_ENV JOBROOT=$JR WHEEL_STORE_SEED=$SHARED \
          RETREAD_PERSIST_CACHE_ROOT=$PERSIST RETREAD_WHEEL_STORE=$SHARED \
          bash "$DEAD" 2>&1)
    if printf '%s' "$OUT" | grep -q 'job-scoped' \
       && [ "$(census_field "$OUT" scope)" = SHARED ] \
       && [ "$(census_field "$OUT" path)" = "$SHARED" ]; then
      ok "$NAME: C. dead-letter control: block still claims job-scoped, census correctly says SHARED"
    else
      fail "$NAME: C. the census believed the harness instead of resolving the store -- A and B are vacuous"
      printf '%s\n' "$OUT" | tail -4 | sed 's/^/GUARD:   /'
    fi
  fi
done

[ "$FAIL" = 0 ] && { echo "wheel-store-census guard: ALL PASS"; exit 0; }
echo "wheel-store-census guard: FAILED"; exit 1
