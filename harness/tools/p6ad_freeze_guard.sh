#!/usr/bin/env bash
# p6ad freeze guard. Proves, on real repodata, that a frozen snapshot yields the
# SAME universe digest twice back to back, and that the guard is not vacuous
# because a snapshot with one document changed yields a DIFFERENT one.
#
#   SNAP=<path to pixi-build-retread> bash p6ad_freeze_guard.sh [work dir]
#
# Phase 3 does NOT wait for conda-forge to publish something. A guard whose
# non-vacuity depends on upstream moving inside a twenty-minute window is not a
# guard; it is a coin flip. It mutates one document in a SECOND snapshot and
# requires the digest to move, which is the same property a real refresh has.
set -e
SNAP=${SNAP:?SNAP=<path to pixi-build-retread> required}
SRC=${SRC:-/oscar/data/stellex/glvov/agrescap/cache/retread/rattler}
WORK=${1:-${TMPDIR:-/tmp}/p6ad-freeze-$$}
FAST_ENV=${FAST_ENV:-/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools/harness/tools/retread_fast_env.sh}
. "$FAST_ENV"

mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
fail=0
say () { echo "p6ad-freeze-guard: $*"; }

say "PHASE 1 -- freeze one snapshot"
# NOT in a pipeline: the function EXPORTS, and a pipeline runs it in a subshell
# where the exports die with the stage. The first run of this guard caught
# exactly that, which is why the census goes to a file and is cat'd afterwards.
RETREAD_FREEZE_BINARY=$SNAP retread_freeze_repodata "$SRC" "$WORK/frozen" > "$WORK/freeze.log" 2>&1 \
  || { cat "$WORK/freeze.log"; say "FATAL the freeze itself failed"; exit 3; }
cat "$WORK/freeze.log"
grep -q '^### repodata frozen ' "$WORK/freeze.log" || { say "FATAL no census line"; exit 3; }
# The freeze's own exclusion rule, checked on the snapshot it just made. FETCH
# LOCKS are the thing that must never be copied -- a lock from another job's
# in-flight fetch is exactly the poison `retread_seed_wheel_store` excludes for
# wheels. Content-hash MEMOS are a different case and must NOT be asserted away
# here: the job-header read the function itself just ran WRITES memos into the
# snapshot, correctly, describing the snapshot's own inodes. The first run of
# this guard failed on that and the assertion was wrong, not the freeze.
if find "$WORK/frozen/retread-repodata" -maxdepth 1 -name '.*retread-fetch-v1.lock' | grep -q .; then
  say "FAIL a fetch lock reached the frozen snapshot"; fail=1
else
  say "PASS  no fetch lock reached the frozen snapshot ($(find "$WORK/frozen/retread-repodata" -maxdepth 1 -name '.*retread-universe-v1.json' | wc -l) locally-written content-hash memos, which is correct)"
fi

say "PHASE 2 -- two back-to-back reads of the SAME frozen snapshot"
a=$("$SNAP" repodata-universe --cache-root "$WORK/frozen" 2>/dev/null)
b=$("$SNAP" repodata-universe --cache-root "$WORK/frozen" 2>/dev/null)
da=${a##*digest=}; da=${da%% *}
db=${b##*digest=}; db=${db%% *}
say "read A digest=$da"
say "read B digest=$db"
if [ "$da" = "$db" ] && [ -n "$da" ]; then
  say "PASS  a frozen snapshot reads the same universe twice"
else
  say "FAIL  a frozen snapshot must read the same universe twice ($da vs $db)"; fail=1
fi
# And the exports the freeze made are the ones retread reads.
[ "$RATTLER_CACHE_DIR" = "$WORK/frozen" ] || { say "FAIL RATTLER_CACHE_DIR=$RATTLER_CACHE_DIR"; fail=1; }
[ "$RETREAD_REPODATA_FROZEN" = 1 ] || { say "FAIL RETREAD_REPODATA_FROZEN=$RETREAD_REPODATA_FROZEN"; fail=1; }

say "PHASE 3 -- NON-VACUITY: a second snapshot with one document moved"
cp -a "$WORK/frozen" "$WORK/moved"
victim=$(ls -S "$WORK/moved/retread-repodata"/*.json | tail -1)   # the SMALLEST document
say "mutating $(basename "$victim")"
python3 - "$victim" <<'PY'
import sys
p=sys.argv[1]
data=bytearray(open(p,'rb').read())
data.extend(b' ')          # one byte, appended: a real refresh moves far more
open(p,'wb').write(bytes(data))
PY
# `cp -a` above carries any memo phase 2's reads left behind, and those memos
# describe the SOURCE inodes. That is exactly the case the stat tuple exists to
# catch, so the digest below must still move -- but say so rather than leave it
# implicit, because it is the one place in this guard where a memo is present
# and must be ignored.
say "memos carried into the mutated copy: $(find "$WORK/moved/retread-repodata" -maxdepth 1 -name '.*retread-universe-v1.json' | wc -l) (must not change the answer)"
c=$("$SNAP" repodata-universe --cache-root "$WORK/moved" 2>/dev/null)
dc=${c##*digest=}; dc=${dc%% *}
say "moved  digest=$dc"
if [ "$dc" != "$da" ] && [ -n "$dc" ]; then
  say "PASS  one changed document moves the universe digest"
else
  say "FAIL  a changed document MUST move the digest ($da vs $dc)"; fail=1
fi

say "PHASE 4 -- a frozen backend refuses to invent a pair it does not hold"
mkdir -p "$WORK/empty/retread-repodata"
if "$SNAP" repodata-universe --cache-root "$WORK/empty" >/dev/null 2>&1; then
  say "FAIL an empty snapshot must not report a universe"; fail=1
else
  say "PASS  an empty snapshot is refused, not reported as a universe"
fi

[ "$fail" = 0 ] && say "GUARD GREEN" || say "GUARD RED"
exit "$fail"
