#!/usr/bin/env bash
# p6af-2a GUARD -- the lazy, sha-verified package half of the frozen channel
# mirror, and the 404 reader that would have caught p6af-2 in 90 seconds.
#
# It runs the org, not a structure check: a real server on a real port, real
# HTTP requests, one real upstream fetch, and the reader asserted on the access
# log the server actually wrote.  Six arms, and TWO of them must be RED:
#
#   A  shard-index 404          -> the reader must TOLERATE it (p6af.1 fallback)
#   B  repodata.json 200        -> the static half still works
#   C  a package IN the frozen document, absent locally
#                               -> filled from upstream, sha-verified, 200
#   D  reader over A+B+C        -> PASS  (0 stray 404s)
#   E  a package NOT in the frozen document
#                               -> 404 + a NORECORD row, and the reader goes RED
#                                  naming it.  This is the "fixture with one
#                                  missing package" the brief asks for, and it
#                                  is the exact shape of job 5855746's death.
#   F  the SAME package as C with the frozen record's sha256 mutated
#                               -> REFUSED (never served), SHA-MISMATCH logged.
#                                  Without F, C proves only that the bytes
#                                  arrived, not that anything checked them.
#
# NEEDS THE NETWORK (arm C fetches ~27 KB from prefix.dev), so it runs on a
# compute node, never on the login node.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/retread_fast_env.sh"

W=${1:-${TMPDIR:-/tmp}/p6af2a-guard-$$}
PORT=${2:-$(( 21000 + (${SLURM_JOB_ID:-$$} % 3000) ))}
mkdir -p "$W" || exit 2
ROOT=$W/mirror
STORE=$W/pkgstore
CHAN=prefix.dev__conda-forge
SUB=noarch
PKG=colorama-0.4.6-pyhd8ed1ab_1.conda
GHOST=p6af2a-no-such-package-1.0-h0000000_0.conda
UPSTREAM=https://prefix.dev/conda-forge/$SUB/$PKG
FAIL=0
say () { printf '### %s\n' "$*"; }
# Every RED dumps the server's own two logs. Guard job 5858488 failed arm C with
# no reason on stdout at all -- the reason (a 403 from the upstream) was in the
# fetch log, on a compute node, in a $TMPDIR that the next job would not have.
# A guard that says FAIL without saying why costs a whole second job to read.
red () {
  printf '### GUARD FAIL %s\n' "$*" >&2
  FAIL=1
  printf '###   fetch log:\n'; sed 's/^/###     /' "${RETREAD_MIRROR_FETCH_LOG:-/dev/null}" 2>/dev/null | tail -20
  printf '###   server log:\n'; sed 's/^/###     /' "${RETREAD_MIRROR_ACCESS_LOG:-/dev/null}" 2>/dev/null | tail -20
}

mkdir -p "$ROOT/$CHAN/$SUB" || exit 2

# ---- the frozen document, built from the real archive ------------------------
# The sha256 in the record is computed from the bytes upstream is serving right
# now rather than hardcoded, so the guard cannot rot into a false RED the next
# time conda-forge republishes.  Arm F is what makes arm C non-vacuous.
say "fetching $UPSTREAM to build the frozen record"
curl -fsSL "$UPSTREAM" -o "$W/$PKG" || { echo "### GUARD ABORT: cannot reach $UPSTREAM (network?)" >&2; exit 3; }
SHA=$(sha256sum "$W/$PKG" | cut -d' ' -f1)
SIZE=$(stat -c %s "$W/$PKG")
say "record: $PKG size=$SIZE sha256=$SHA"
write_doc () {   # $1 = the sha256 to put in the record
  cat > "$ROOT/$CHAN/$SUB/repodata.json" <<JSON
{"info":{"subdir":"$SUB"},"packages":{},"packages.conda":{
 "$PKG":{"build":"pyhd8ed1ab_1","build_number":1,"depends":[],"license":"BSD-3-Clause",
  "md5":"00000000000000000000000000000000","name":"colorama","noarch":"python",
  "run_exports":{"weak":[]},"sha256":"$1","size":$SIZE,"subdir":"noarch",
  "timestamp":0,"version":"0.4.6"}}}
JSON
}
write_doc "$SHA"

# ---- serve, with the package store ON ---------------------------------------
export RETREAD_MIRROR_FETCH_LOG=$W/fetch.log
: > "$RETREAD_MIRROR_FETCH_LOG"
retread_serve_channel_mirror "$ROOT" "$PORT" "$STORE" || { echo "### GUARD ABORT: server never came up" >&2; exit 3; }
LOG=$RETREAD_MIRROR_ACCESS_LOG
U=$RETREAD_MIRROR_URL
trap 'kill "$RETREAD_MIRROR_PID" 2>/dev/null' EXIT

# ---- A: the whole fallback CHAIN the protocol depends on ---------------------
# Three names, not one. p6af.1's summary counted only the shard index, and a
# reader written from that summary goes RED on a perfect run: two full canonical
# solves (5855746, 5858679) show 16 of EACH of these per run, because the freeze
# writes the plain document and nothing else. All three must be tolerated, and
# arm E is what keeps that tolerance from becoming a hole.
for n in repodata_shards.msgpack.zst repodata.json.zst repodata.json.bz2; do
  A=$(curl -s -o /dev/null -w '%{http_code}' "$U/$CHAN/$SUB/$n")
  say "A fallback $n -> $A (want 404)"
  [ "$A" = 404 ] || red "A: $n answered $A, not 404"
done

# ---- B: the classic document -------------------------------------------------
B=$(curl -s -o /dev/null -w '%{http_code}' "$U/$CHAN/$SUB/repodata.json")
say "B repodata.json -> $B (want 200)"
[ "$B" = 200 ] || red "B: repodata.json answered $B, not 200"

# ---- C: the package, absent locally, filled from upstream --------------------
[ -e "$ROOT/$CHAN/$SUB/$PKG" ] && red "C precondition: the package was already in the mirror"
C=$(curl -s -o "$W/served.conda" -w '%{http_code}' "$U/$CHAN/$SUB/$PKG")
say "C package -> $C (want 200)"
[ "$C" = 200 ] || red "C: package answered $C, not 200"
if [ -f "$W/served.conda" ]; then
  GOT=$(sha256sum "$W/served.conda" | cut -d' ' -f1)
  [ "$GOT" = "$SHA" ] || red "C: served bytes sha256=$GOT != $SHA"
fi
grep -q "^\[.*\] PKGFETCH .*$PKG " "$RETREAD_MIRROR_FETCH_LOG" \
  || red "C: no PKGFETCH row for $PKG in $RETREAD_MIRROR_FETCH_LOG"

# ---- D: the reader, over a clean log ----------------------------------------
if retread_assert_mirror_no_stray_404 "$LOG"; then
  say "D reader PASS over A+B+C (as it must: every 404 so far is a fallback name)"
else
  red "D: the reader went RED on a clean log -- it is over-strict"
fi

# ---- E: the fixture with one missing package --------------------------------
E=$(curl -s -o /dev/null -w '%{http_code}' "$U/$CHAN/$SUB/$GHOST")
say "E package absent from the frozen document -> $E (want 404)"
[ "$E" = 404 ] || red "E: a package the frozen universe never declared answered $E, not 404"
grep -q "NORECORD .*$GHOST" "$RETREAD_MIRROR_FETCH_LOG" \
  || red "E: no NORECORD row for $GHOST -- the refusal was silent"
if retread_assert_mirror_no_stray_404 "$LOG"; then
  red "E: THE READER IS VACUOUS -- it passed a log containing a package 404"
else
  say "E reader RED as required, and it named the request"
fi

# ---- E2: `.tar.bz2` is not `.bz2` -------------------------------------------
# The tolerated fallback name `repodata.json.bz2` and the conda archive suffix
# `.tar.bz2` are one character class apart. Matched as a SUFFIX rather than as a
# whole basename, the gate would wave a missing package through -- so the miss is
# probed and the counter is read: the fallback tally must not have absorbed it.
curl -s -o /dev/null "$U/$CHAN/$SUB/p6af2a-ghost-1.0-h0.tar.bz2"
E2=$(retread_assert_mirror_no_stray_404 "$LOG" 2>&1 | grep -c 'p6af2a-ghost-1.0-h0\.tar\.bz2')
say "E2 missing .tar.bz2 package named by the reader -> $E2 (want 1)"
[ "$E2" = 1 ] || red "E2: a missing .tar.bz2 PACKAGE was not named -- the .bz2 rule is a suffix rule, not a whole-name rule"

# ---- F: mutation control -- the sha check must not be decorative -------------
kill "$RETREAD_MIRROR_PID" 2>/dev/null; wait "$RETREAD_MIRROR_PID" 2>/dev/null
rm -rf "$STORE" "$ROOT/$CHAN/$SUB/$PKG"
write_doc "0000000000000000000000000000000000000000000000000000000000000000"
PORT2=$(( PORT + 1 ))
: > "$RETREAD_MIRROR_FETCH_LOG"
retread_serve_channel_mirror "$ROOT" "$PORT2" "$STORE" || { echo "### GUARD ABORT: second server never came up" >&2; exit 3; }
trap 'kill "$RETREAD_MIRROR_PID" 2>/dev/null' EXIT
F=$(curl -s -o /dev/null -w '%{http_code}' "$RETREAD_MIRROR_URL/$CHAN/$SUB/$PKG")
say "F package with a mutated frozen sha256 -> $F (want 404: fetched, checked, refused, never stored)"
[ "$F" = 404 ] || red "F: a package whose bytes disagree with the frozen record was SERVED ($F)"
grep -q 'SHA-MISMATCH' "$RETREAD_MIRROR_FETCH_LOG" || red "F: no SHA-MISMATCH row -- nothing checked the bytes"
[ -e "$STORE/$CHAN/$SUB/$PKG" ] && red "F: the mismatching archive was written into the package store"

kill "$RETREAD_MIRROR_PID" 2>/dev/null
if [ "$FAIL" = 0 ]; then echo "### p6af2a_pkg_mirror_guard PASS (work dir $W)"; exit 0; fi
echo "### p6af2a_pkg_mirror_guard FAILED (work dir $W kept)" >&2; exit 1
