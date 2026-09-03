#!/usr/bin/env bash
# Reclaim ABANDONED fill locks in the content-addressed wheel store.
#
# THE DEFECT THIS WORKS AROUND IS IN CODE AND IS BOARDED, NOT FIXED HERE.
# `courier` fills an entry as `<sha256>/<name>.whl` under a sidecar
# `.<name>.whl.retread-fill-v1.lock`. If the filling process dies between
# taking the lock and renaming the wheel into place, the directory survives
# with a lock and NO wheel. A reader then finds the directory, opens the
# missing wheel, and ABORTS the whole index chain:
#
#   PyPI exact wheel index chain aborted on `https://pypi.nvidia.com` because
#   the failure was not a package miss: opening <store>/<sha>/<name>.whl
#   for SHA-256: No such file or directory (os error 2)
#
# An absent wheel IS a miss and the chain should fall through and re-fetch.
# Until the reader does that, one crashed job poisons the shared store for
# every later job -- which is what happened at 01:43-01:44 on 2026-09-03 and
# then killed jobs 5685024 and 5686431 (both bisect arms) hours later.
#
# This reclaims ONLY the debris: a directory holding a fill lock, no wheel,
# and whose lock is older than TTL_MIN (so a live filler is never disturbed).
# Anything else is left alone and reported.
#
#   usage: wheel_store_reclaim.sh [--apply] [--ttl-min N] [store]
#          default is a DRY RUN.
set -uo pipefail
APPLY=0; TTL_MIN=60
STORE=${RETREAD_WHEEL_STORE:-/oscar/data/stellex/glvov/agrescap/cache/retread/wheels}
while [ $# -gt 0 ]; do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --ttl-min) TTL_MIN=$2; shift 2 ;;
    *) STORE=$1; shift ;;
  esac
done
case "$STORE" in */wheels) ;; *) echo "REFUSING: $STORE does not end in /wheels"; exit 2;; esac
[ -d "$STORE" ] || { echo "REFUSING: $STORE is not a directory"; exit 2; }

TOTAL=0; POISON=0; YOUNG=0; RECLAIMED=0
for d in "$STORE"/*/; do
  [ -d "$d" ] || continue
  TOTAL=$((TOTAL+1))
  # A directory with ANY wheel in it is a real entry. Never touched.
  [ -n "$(find "$d" -maxdepth 1 -name '*.whl' -print -quit 2>/dev/null)" ] && continue
  lock=$(find "$d" -maxdepth 1 -name '.*.retread-fill-v1.lock' -print -quit 2>/dev/null)
  [ -n "$lock" ] || { echo "  LEFT (no wheel, no lock -- not our debris): $d"; continue; }
  POISON=$((POISON+1))
  if [ -n "$(find "$lock" -newermt "-${TTL_MIN} minutes" 2>/dev/null)" ]; then
    YOUNG=$((YOUNG+1))
    echo "  KEPT (lock younger than ${TTL_MIN}m -- a filler may be live): $d"
    continue
  fi
  echo "  RECLAIM $(stat -c '%y' "$lock" | cut -c1-16)  $d"
  if [ "$APPLY" = 1 ]; then
    rm -f "$lock" && rmdir "$d" 2>/dev/null
    [ -d "$d" ] && echo "    WARN: directory not empty after lock removal, left in place" || RECLAIMED=$((RECLAIMED+1))
  fi
done
echo "store=$STORE entries=$TOTAL poisoned=$POISON kept_young=$YOUNG reclaimed=$RECLAIMED apply=$APPLY"
[ "$APPLY" = 1 ] || echo "DRY RUN -- pass --apply to reclaim"
