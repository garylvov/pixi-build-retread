#!/bin/bash
# p6af_channel_mirror_guard.sh -- the guard for retread_freeze_channel_mirror.
#
# It runs the org, not a structure check: three real `pixi lock` runs of a real
# manifest against real conda-forge documents.
#
#   ARM 0  NON-VACUITY. Network blocked, NO mirror. MUST FAIL. If this passes,
#          the block is not blocking and every other arm is meaningless.
#   ARM 1  Network blocked, mirror. MUST succeed.
#   ARM 2  Same again, fresh pixi cache and fresh HOME. MUST succeed and MUST be
#          BYTE-IDENTICAL to arm 1.
#   ASSERT the arm-1 lock carries at least one `run_exports` block -- the direct
#          reader for the merge step. Drop the merge and this line goes to 0
#          while every other assertion still passes.
#
# Usage:  p6af_channel_mirror_guard.sh <workdir> <mirror root> <port>
# The mirror root must already have been built by retread_freeze_channel_mirror.
set -u
W=${1:?workdir}; MIRROR=${2:?mirror root}; PORT=${3:-18791}
PIXI=${PIXI_BIN:-/users/glvov/.pixi/bin/pixi}
mkdir -p "$W" || exit 2

# The probe manifest names ONLY conda-forge, because that is the channel whose
# documents the mirror is being tested for. It is not the canonical manifest and
# does not pretend to be: a canonical relock is a separate, hour-long proof.
mkdir -p "$W/ws"
cat > "$W/ws/pixi.toml" <<'TOML'
[workspace]
name = "p6af-guard"
channels = ["https://prefix.dev/conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = "3.11.*"
TOML

( cd "$MIRROR" && exec python3 -m http.server "$PORT" --bind 127.0.0.1 ) > "$W/httpd.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in 1 2 3 4 5 6 7 8 9 10; do curl -fs -o /dev/null "http://127.0.0.1:$PORT/" && break; sleep 1; done
curl -fsS -o /dev/null "http://127.0.0.1:$PORT/" || { echo "GUARD FATAL mirror server never answered"; exit 2; }

arm () { # $1 name  $2 mirror|nomirror
  local a=$1 m=$2
  local H=$W/home-$a
  rm -rf "$H" "$W/cache-$a" "$W/ws/pixi.lock" "$W/ws/.pixi"
  mkdir -p "$H/.pixi" "$W/cache-$a"
  if [ "$m" = mirror ]; then
    printf '[mirrors]\n"https://prefix.dev/conda-forge" = ["http://127.0.0.1:%s/prefix.dev__conda-forge"]\n' "$PORT" > "$H/.pixi/config.toml"
  else
    : > "$H/.pixi/config.toml"
  fi
  local t0 rc
  t0=$(date +%s)
  ( cd "$W/ws" && env HOME="$H" PIXI_CACHE_DIR="$W/cache-$a" \
      NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
      HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9 \
      https_proxy=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 all_proxy=http://127.0.0.1:9 \
      PIXI_SHIM_NO_STALE_CHECK=1 "$PIXI" lock ) > "$W/$a.log" 2>&1
  rc=$?
  [ -f "$W/ws/pixi.lock" ] && cp "$W/ws/pixi.lock" "$W/pixi.lock.$a"
  echo "### GUARD arm=$a mirror=$m rc=$rc wall=$(( $(date +%s)-t0 ))s md5=$(md5sum "$W/pixi.lock.$a" 2>/dev/null | cut -d' ' -f1)"
  return $rc
}

fail=0
arm A0_nomirror nomirror
if [ $? -eq 0 ]; then echo "GUARD FAIL A0: a lock SUCCEEDED with the network blocked and no mirror -- the block is not blocking"; fail=1
else echo "GUARD OK A0: no-mirror lock refused with the network blocked (non-vacuity)"; fi

arm A1_mirror mirror || { echo "GUARD FAIL A1: offline lock against the mirror failed"; fail=1; }
arm A2_mirror mirror || { echo "GUARD FAIL A2: offline lock against the mirror failed"; fail=1; }

m1=$(md5sum "$W/pixi.lock.A1_mirror" 2>/dev/null | cut -d' ' -f1)
m2=$(md5sum "$W/pixi.lock.A2_mirror" 2>/dev/null | cut -d' ' -f1)
if [ -n "$m1" ] && [ "$m1" = "$m2" ]; then echo "GUARD OK identity: two offline locks on one mirror are byte-identical ($m1)"
else echo "GUARD FAIL identity: A1=$m1 A2=$m2"; fail=1; fi

re=$(grep -c 'run_exports' "$W/pixi.lock.A1_mirror" 2>/dev/null)
[ -n "$re" ] || re=0
if [ "$re" -gt 0 ]; then echo "GUARD OK run_exports: the mirror lock carries $re run_exports blocks"
else echo "GUARD FAIL run_exports: 0 blocks -- the mirror was built without the sharded-cache merge"; fail=1; fi

echo "### GUARD VERDICT $([ $fail -eq 0 ] && echo PASS || echo FAIL)"
exit $fail
