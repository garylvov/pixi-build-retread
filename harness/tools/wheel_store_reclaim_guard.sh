#!/usr/bin/env bash
# Guard for wheel_store_reclaim.sh. Builds a fake store with one of each shape
# and asserts the tool touches EXACTLY the abandoned debris.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d)/wheels; mkdir -p "$W"; trap 'rm -rf "$(dirname "$W")"' EXIT
FAIL=0; say(){ printf '%s\n' "$*"; }
mk(){ mkdir -p "$W/$1"; }
# (a) a REAL entry: wheel present, and a lock beside it (a fill that finished)
mk aaa; : > "$W/aaa/pkg-1.0.whl"; : > "$W/aaa/.pkg-1.0.whl.retread-fill-v1.lock"
touch -d '3 hours ago' "$W/aaa/.pkg-1.0.whl.retread-fill-v1.lock"
# (b) ABANDONED: lock, no wheel, old   -> the only thing that may be reclaimed
mk bbb; : > "$W/bbb/.pkg-2.0.whl.retread-fill-v1.lock"; touch -d '3 hours ago' "$W/bbb/.pkg-2.0.whl.retread-fill-v1.lock"
# (c) IN FLIGHT: lock, no wheel, fresh -> a live filler, must be kept
mk ccc; : > "$W/ccc/.pkg-3.0.whl.retread-fill-v1.lock"
# (d) neither wheel nor lock           -> not our debris, must be left
mk ddd; : > "$W/ddd/README"
"$HERE/wheel_store_reclaim.sh" --apply --ttl-min 60 "$W" > "$W/../out.txt" 2>&1
cat "$W/../out.txt" | sed 's/^/  /'
chk(){ [ "$1" = "$2" ] || { say "  FAIL $3: want $1 got $2"; FAIL=1; }; }
[ -f "$W/aaa/pkg-1.0.whl" ] || { say "  FAIL a REAL entry's wheel was removed"; FAIL=1; }
[ -f "$W/aaa/.pkg-1.0.whl.retread-fill-v1.lock" ] || { say "  FAIL a real entry's lock was removed"; FAIL=1; }
[ -d "$W/bbb" ] && { say "  FAIL the abandoned entry was NOT reclaimed"; FAIL=1; }
[ -f "$W/ccc/.pkg-3.0.whl.retread-fill-v1.lock" ] || { say "  FAIL an IN-FLIGHT fill was reclaimed -- that races a live filler"; FAIL=1; }
[ -f "$W/ddd/README" ] || { say "  FAIL an unrelated directory was touched"; FAIL=1; }
grep -q 'reclaimed=1' "$W/../out.txt" || { say "  FAIL expected exactly one reclaim"; FAIL=1; }
grep -q 'kept_young=1' "$W/../out.txt" || { say "  FAIL expected exactly one in-flight keep"; FAIL=1; }
# it must refuse a store path that is not a wheel store
"$HERE/wheel_store_reclaim.sh" --apply /tmp >/dev/null 2>&1 && { say "  FAIL it did not refuse a non-wheels path"; FAIL=1; }
[ "$FAIL" = 0 ] && { say "wheel-store reclaim guard: ALL PASS"; exit 0; }
say "wheel-store reclaim guard: FAILED"; exit 1
