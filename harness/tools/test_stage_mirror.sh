#!/usr/bin/env bash
# test_stage_mirror.sh -- a REAL round trip of the staging functions the template
# now uses, on a fixture tree shaped like imprint-data. It builds a mirror,
# stages a workspace out of it, breaks the in-place-written links, then SIMULATES
# the lock: it appends to a progress log, truncates a probe trace, rewrites a
# .retread-cache stamp, and atomically replaces a wheel -- and asserts the mirror
# survived all of it. Then it MUTATES the harness (skips stage_break_links) and
# asserts stage_verify_mirror actually fails, because a guard that cannot fail is
# not a guard.
#
#   bash test_stage_mirror.sh
#
# Runs anywhere on /oscar; needs no Slurm, no pixi, no lock.
set -uo pipefail

TMPL=${1:-/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/tools/phase_template/phaseN_relock.sh}
ROOT=$(mktemp -d "${TMPDIR:-/tmp}/stage-mirror-test.XXXXXX")
trap 'chmod -R u+w "$ROOT" 2>/dev/null; rm -rf "$ROOT"' EXIT
FAIL=0
ok   () { echo "  PASS $*"; }
bad  () { echo "  FAIL $*"; FAIL=1; }
check() { if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1: got '$2' want '$3'"; fi; }

########## the fixture: the shape that matters, not the size ##########
SRC=$ROOT/src
mkdir -p "$SRC/pypi-packs/demo-pack/wheels/demo/.retread-source-wheels/v12/aa/bb" \
         "$SRC/pypi-packs/demo-pack/wheels/.retread-wheel-fetch/v1/sha256/cc" \
         "$SRC/third_party/ProtoMotions/deep/er" "$SRC/.git/objects/pack" \
         "$SRC/.pixi" "$SRC/src"
echo 'name = "demo"' > "$SRC/pixi.toml"
echo 'detached = true' > "$SRC/.pixi/config.toml"
P=$SRC/pypi-packs/demo-pack
echo '{"probe":"old"}'                > "$P/retread-probe-trace-demo-pack.json"
echo '{"audit":"old"}'                > "$P/retread-audit-demo-pack.json"
echo 'progress line 1'                > "$P/retread-progress-demo-pack.log"
echo '{"lock":"old"}'                 > "$P/retread-demo-pack.target-abc.lock.json"
head -c 300000 /dev/urandom           > "$P/wheels/.retread-wheel-fetch/v1/sha256/cc/demo-1.0-py3-none-any.whl"
echo 'stamp v1'                       > "$P/wheels/.retread-wheel-fetch/v1/sha256/cc/demo-1.0-py3-none-any.whl.retread-cache"
head -c 200000 /dev/urandom           > "$P/wheels/demo/.retread-source-wheels/v12/aa/bb/demo-1.0.injected.whl"
echo 'tp' > "$SRC/third_party/ProtoMotions/deep/er/file.txt"
echo 'gitobj' > "$SRC/.git/objects/pack/p.pack"
echo 'srcfile' > "$SRC/src/mod.py"      # depth 2 -- the shallow-file regression
echo 'rootfile' > "$SRC/AGENTS.md"     # depth 1 -- ditto
mkdir -p "$SRC/test"; echo 't' > "$SRC/test/test_a.py"
CLEANED=$ROOT/pixi.toml.cleaned
echo 'name = "demo-under-test"' > "$CLEANED"

########## pull the staging functions out of the template ##########
# Everything from the STAGE_METHOD assignment down to (not including) the
# `if [ ! -e "$WS/.cert-staged" ]` driver, which is what a job runs.
FUNCS=$ROOT/stage_funcs.sh
awk '/^STAGE_METHOD=/{f=1} f&&/^if \[ ! -e "\$WS\/\.cert-staged" \]/{exit} f{print}' "$TMPL" > "$FUNCS"
grep -q 'stage_build_mirror ()'  "$FUNCS" || { echo "FATAL: could not extract stage_* from $TMPL"; exit 2; }
grep -q 'stage_verify_mirror ()' "$FUNCS" || { echo "FATAL: no stage_verify_mirror in $TMPL"; exit 2; }
echo "### extracted $(grep -c '^stage_[a-z_]* ()' "$FUNCS") stage functions from $TMPL"

run_stage () {   # $1 = "break" | "nobreak"
  local mode=$1
  SRC_WS=$SRC J=TEST$$ A=$ROOT/artifacts WS=$ROOT/ws.$mode
  mkdir -p "$A"
  STAGE_MIRROR_ROOT=$ROOT/mirrors
  . "$FUNCS"
  STAGE_MIRROR_ROOT=$ROOT/mirrors      # the snippet re-asserts the real root; override after sourcing
  STAGE_KEY=$(stage_key)
  STAGE_MIRROR=$STAGE_MIRROR_ROOT/$STAGE_KEY
  if ! { [ -f "$STAGE_MIRROR/.stage-mirror-key" ] && grep -qx "key=$STAGE_KEY" "$STAGE_MIRROR/.stage-mirror-key"; }; then
    mkdir -p "$STAGE_MIRROR_ROOT"
    stage_build_mirror "$STAGE_MIRROR" "$STAGE_KEY" || return 1
  fi
  stage_mirror_hit "$STAGE_MIRROR" || return 1
  rm -rf "$WS/.pixi"; mkdir -p "$WS/.pixi"; cp "$SRC_WS/.pixi/config.toml" "$WS/.pixi/config.toml"
  rm -f "$WS/pixi.toml"; cp "$CLEANED" "$WS/pixi.toml"
  [ "$mode" = break ] && stage_break_links
  return 0
}

simulate_lock () {   # exactly the four in-place writers the source audit named
  local w=$1
  echo 'progress line 2'   >> "$w/pypi-packs/demo-pack/retread-progress-demo-pack.log"   # append
  echo '{"probe":"NEW and longer"}' > "$w/pypi-packs/demo-pack/retread-probe-trace-demo-pack.json"  # truncate
  echo '{"audit":"NEW"}'   > "$w/pypi-packs/demo-pack/retread-audit-demo-pack.json"      # truncate
  echo 'stamp v2 longer'   > "$w/pypi-packs/demo-pack/wheels/.retread-wheel-fetch/v1/sha256/cc/demo-1.0-py3-none-any.whl.retread-cache"
  # and one ATOMIC writer, which must be harmless even while hardlinked
  local t=$w/pypi-packs/demo-pack/wheels/demo/.retread-source-wheels/v12/aa/bb
  head -c 200000 /dev/urandom > "$t/.tmp.whl" && mv -f "$t/.tmp.whl" "$t/demo-1.0.injected.whl"
  echo 'lockfile' > "$w/pixi.lock"
}

echo "=== 1. mirror build + hit + break-links, then a simulated lock ==="
run_stage break || { echo "FATAL: staging failed"; exit 2; }
MIR=$ROOT/mirrors/$(cd "$ROOT/mirrors" && ls)
WSB=$ROOT/ws.break

check "workspace has the source tree"      "$([ -f "$WSB/pypi-packs/demo-pack/retread-audit-demo-pack.json" ] && echo yes)" yes
check "third_party came across"            "$([ -f "$WSB/third_party/ProtoMotions/deep/er/file.txt" ] && echo yes)" yes
check "DEPTH-1 file came across"           "$([ -f "$WSB/AGENTS.md" ] && echo yes)" yes
check "DEPTH-2 file came across"           "$([ -f "$WSB/src/mod.py" ] && echo yes)" yes
check "DEPTH-2 file in another dir"        "$([ -f "$WSB/test/test_a.py" ] && echo yes)" yes
check "no file is missing vs the mirror"   "$( ( cd "$MIR" && find . -type f | grep -vF '.stage-mirror-' | sort ) > "$ROOT/m.lst"; ( cd "$WSB" && find . -type f | sort ) > "$ROOT/w.lst"; comm -23 "$ROOT/m.lst" "$ROOT/w.lst" | wc -l )" 0
check ".git came across"                   "$([ -f "$WSB/.git/objects/pack/p.pack" ] && echo yes)" yes
check "manifest under test is installed"   "$(cat "$WSB/pixi.toml")" 'name = "demo-under-test"'
check "manifest is NOT shared with mirror" "$(stat -c %h "$WSB/pixi.toml")" 1
check "big wheel IS shared with mirror"    "$(stat -c %h "$WSB/pypi-packs/demo-pack/wheels/demo/.retread-source-wheels/v12/aa/bb/demo-1.0.injected.whl")" 2
check "progress log is NOT shared"         "$(stat -c %h "$WSB/pypi-packs/demo-pack/retread-progress-demo-pack.log")" 1
check "probe trace is NOT shared"          "$(stat -c %h "$WSB/pypi-packs/demo-pack/retread-probe-trace-demo-pack.json")" 1
check "audit json is NOT shared"           "$(stat -c %h "$WSB/pypi-packs/demo-pack/retread-audit-demo-pack.json")" 1
check ".retread-cache stamp is NOT shared" "$(stat -c %h "$WSB/pypi-packs/demo-pack/wheels/.retread-wheel-fetch/v1/sha256/cc/demo-1.0-py3-none-any.whl.retread-cache")" 1

simulate_lock "$WSB"
OUT=$( . "$FUNCS" >/dev/null 2>&1; A=$ROOT/artifacts J=TEST$$ TAG=T; stage_verify_mirror "$MIR" )
echo "$OUT" | sed 's/^/    /'
check "mirror INTACT after a simulated lock" "$(echo "$OUT" | grep -c 'mirror INTACT')" 1
check "mirror probe trace still the old bytes" "$(cat "$MIR/pypi-packs/demo-pack/retread-probe-trace-demo-pack.json")" '{"probe":"old"}'
check "mirror progress log still one line"     "$(wc -l < "$MIR/pypi-packs/demo-pack/retread-progress-demo-pack.log")" 1
check "mirror was NOT quarantined"             "$([ -d "$MIR" ] && echo yes)" yes
check "SOURCE tree untouched (probe trace)"    "$(cat "$SRC/pypi-packs/demo-pack/retread-probe-trace-demo-pack.json")" '{"probe":"old"}'
check "SOURCE tree untouched (progress log)"   "$(wc -l < "$SRC/pypi-packs/demo-pack/retread-progress-demo-pack.log")" 1

echo "=== 2. THE GUARD MUST BE ABLE TO FAIL: same run WITHOUT stage_break_links ==="
run_stage nobreak || { echo "FATAL: staging failed"; exit 2; }
WSN=$ROOT/ws.nobreak
check "unbroken probe trace IS shared with the mirror" "$([ "$(stat -c %h "$WSN/pypi-packs/demo-pack/retread-probe-trace-demo-pack.json")" -ge 2 ] && echo yes)" yes
simulate_lock "$WSN"
OUT2=$( . "$FUNCS" >/dev/null 2>&1; A=$ROOT/artifacts J=TEST$$ TAG=T; stage_verify_mirror "$MIR" )
echo "$OUT2" | sed 's/^/    /'
check "verify DETECTS the poisoned mirror" "$(echo "$OUT2" | grep -c 'the mirror CHANGED under this job')" 1
check "and QUARANTINES it"                 "$(ls -d "$MIR".DIRTY-* >/dev/null 2>&1 && echo yes)" yes

echo
if [ "$FAIL" = 0 ]; then echo "ALL GREEN"; else echo "SOME CHECKS FAILED"; fi
exit "$FAIL"
