#!/usr/bin/env bash
# sdist_build_scope_guard.sh -- the guard for C31-4's two halves.
#
# HALF ONE, the reader: `sdist_build_poison_guard.sh` must go RED on a fixture
# carrying one poisoned CMakeCache.txt and GREEN on a byte copy of the same
# fixture with the compiler path repaired. Both directions are asserted, because
# a guard that cannot fail is a defect and a guard that always fails is worse.
#
# HALF TWO, the writer: `retread_scope_sdist_builds` must give the job its OWN
# empty `sdists-v9` and `builds-v0` in BOTH uv caches while leaving every
# byte-keyed bucket a symlink into the shared cache -- and it must REFUSE an
# overlay on a different filesystem, because uv renames `<cache>/.tmpXXXX` into
# `archive-v0` and that rename is EXDEV across devices (measured, job 5869427).
#
# Usage: sdist_build_scope_guard.sh
#   Self-contained. Needs a writable $TMPDIR. Set SCOPE_GUARD_XDEV to a
#   directory on a DIFFERENT filesystem to exercise the EXDEV negative control;
#   without it that control is SKIPPED and said to be skipped, never silently
#   counted as a pass.
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
READER=$HERE/sdist_build_poison_guard.sh
FAST_ENV=$HERE/retread_fast_env.sh
[ -x "$READER" ] || [ -f "$READER" ] || { echo "GUARD FATAL: no sdist_build_poison_guard.sh at $READER"; exit 2; }
[ -f "$FAST_ENV" ] || { echo "GUARD FATAL: no retread_fast_env.sh at $FAST_ENV"; exit 2; }
# shellcheck source=/dev/null
. "$FAST_ENV"
command -v retread_scope_sdist_builds >/dev/null \
  || { echo "GUARD FATAL: retread_scope_sdist_builds is not defined by $FAST_ENV"; exit 2; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/sdist-scope-guard.XXXXXX") || exit 2
trap 'rm -rf "$WORK"' EXIT
echo "GUARD: work dir $WORK"

FAIL=0
fail () { echo "GUARD FAIL: $*"; FAIL=1; }
ok   () { echo "GUARD  ok : $*"; }

# A real compiler to point the clean fixture at, chosen by measurement.
GOODCC=""
for c in /usr/bin/c++ /usr/bin/g++ /bin/sh; do [ -e "$c" ] && { GOODCC=$c; break; }; done
[ -n "$GOODCC" ] || { echo "GUARD FATAL: no existing tool to build a clean fixture from"; exit 2; }
echo "GUARD: clean fixture will name $GOODCC"

DEAD=/oscar/data/stellex/glvov/retread/ws.GUARD-DOES-NOT-EXIST/.pixi/envs/x/bin/x86_64-conda-linux-gnu-c++
[ -e "$DEAD" ] && { echo "GUARD FATAL: the fixture's 'dead' path exists: $DEAD"; exit 2; }

mk_cache () {  # mk_cache <root> <compiler path>
  local root=$1 cc=$2
  local d=$root/sdists-v9/pypi/openmesh/1.2.1/lv_fixture/src/build-setuptools/temp.linux-x86_64-cpython-312
  mkdir -p "$d" "$root/archive-v0" "$root/wheels-v6" "$root/builds-v0"
  cat > "$d/CMakeCache.txt" <<EOF
# This is the CMakeCache file, fixture.
CMAKE_C_COMPILER:FILEPATH=$cc
CMAKE_CXX_COMPILER:FILEPATH=$cc
CMAKE_CXX_COMPILER_AR:FILEPATH=CMAKE_CXX_COMPILER_AR-NOTFOUND
CMAKE_BUILD_TYPE:STRING=Release
EOF
  # a second, healthy sibling -- the pm-mujoco control from 31.12
  local s=$root/sdists-v9/pypi/openmesh/1.2.1/lv_fixture/src/build-setuptools/temp.linux-x86_64-cpython-310
  mkdir -p "$s"
  printf 'CMAKE_CXX_COMPILER:FILEPATH=%s\n' "$GOODCC" > "$s/CMakeCache.txt"
}

########## HALF ONE: THE READER #################################################
echo "GUARD: === HALF ONE: the reader, RED on poison / GREEN on a clean copy ==="

POISON=$WORK/cache-poisoned
mk_cache "$POISON" "$DEAD"
OUT=$WORK/reader-red.out
bash "$READER" "$POISON" > "$OUT" 2>&1; RC=$?
sed 's/^/GUARD:   /' "$OUT"
[ "$RC" -eq 3 ] && ok "poisoned fixture: reader exit 3 (RED)" || fail "poisoned fixture: reader exit $RC, expected 3"
grep -q "poisoned_files=1" "$OUT" && ok "poisoned fixture: exactly 1 of 2 CMakeCache files flagged (the 3.10 sibling is healthy, 31.12's control)" \
  || fail "poisoned fixture: expected poisoned_files=1, census line was: $(grep TOTAL "$OUT")"
grep -q "cmakecache_files=2" "$OUT" && ok "poisoned fixture: both CMakeCache files were actually walked -- the RED is not vacuous" \
  || fail "poisoned fixture: the walk did not find 2 CMakeCache files"
grep -q "$DEAD" "$OUT" && ok "poisoned fixture: the refusal names the missing tool" || fail "poisoned fixture: refusal did not name $DEAD"

# The CLEAN COPY: byte copy of the same fixture, one path repaired. rsync -aW,
# never cp -al -- a hardlinked 'copy' would share the poisoned inode.
CLEAN=$WORK/cache-clean
rsync -aW "$POISON/" "$CLEAN/" >/dev/null 2>&1 || { fail "could not rsync the fixture"; }
sed -i "s|=$DEAD|=$GOODCC|" "$CLEAN/sdists-v9/pypi/openmesh/1.2.1/lv_fixture/src/build-setuptools/temp.linux-x86_64-cpython-312/CMakeCache.txt"
sed -i "s|^CMAKE_C_COMPILER:FILEPATH=.*|CMAKE_C_COMPILER:FILEPATH=$GOODCC|" "$CLEAN/sdists-v9/pypi/openmesh/1.2.1/lv_fixture/src/build-setuptools/temp.linux-x86_64-cpython-312/CMakeCache.txt"
OUT=$WORK/reader-green.out
bash "$READER" "$CLEAN" > "$OUT" 2>&1; RC=$?
sed 's/^/GUARD:   /' "$OUT"
[ "$RC" -eq 0 ] && ok "clean copy: reader exit 0 (GREEN)" || fail "clean copy: reader exit $RC, expected 0"
grep -q "cmakecache_files=2" "$OUT" && ok "clean copy: the GREEN is not vacuous -- 2 files walked" || fail "clean copy: walk found no files, the GREEN would be vacuous"

# NEGATIVE CONTROL on the skip rules: a `*-NOTFOUND` value must NOT be a poison,
# or every cmake cache on earth would be red.
OUT=$WORK/reader-notfound.out
printf 'CMAKE_CXX_COMPILER:FILEPATH=CMAKE_CXX_COMPILER-NOTFOUND\n' \
  > "$CLEAN/sdists-v9/pypi/openmesh/1.2.1/lv_fixture/src/build-setuptools/temp.linux-x86_64-cpython-310/CMakeCache.txt"
bash "$READER" "$CLEAN" > "$OUT" 2>&1; RC=$?
[ "$RC" -eq 0 ] && ok "a *-NOTFOUND value is not treated as poison" || fail "a *-NOTFOUND value went RED (exit $RC)"

# NEGATIVE CONTROL on double counting: naming one root twice (which a harness
# does whenever UV_CACHE_DIR and PIXI_CACHE_DIR/uv-cache resolve to one
# directory) must not double the census.
OUT=$WORK/reader-dedupe.out
bash "$READER" "$POISON" "$POISON" "$POISON/." > "$OUT" 2>&1
grep -q "TOTAL cmakecache_files=2 " "$OUT" && ok "a root named three times is counted once" \
  || fail "root dedupe: census was $(grep TOTAL "$OUT")"

########## HALF TWO: THE WRITER #################################################
echo "GUARD: === HALF TWO: retread_scope_sdist_builds ==="

SH=$WORK/shared
mk_cache "$SH/uv" "$DEAD"
mk_cache "$SH/pixi/uv-cache" "$DEAD"
mkdir -p "$SH/pixi/pkgs" "$SH/pixi/repodata"
printf 'shared-pkg\n' > "$SH/pixi/pkgs/warm.txt"
printf 'Signature: 8a477f597d28d172789f06886806bc55\n' > "$SH/uv/CACHEDIR.TAG"

JOBROOT=$WORK/jobroot
( export UV_CACHE_DIR=$SH/uv PIXI_CACHE_DIR=$SH/pixi
  retread_scope_sdist_builds "$JOBROOT"
  echo "SCOPED_UV=$UV_CACHE_DIR"
  echo "SCOPED_PIXI=$PIXI_CACHE_DIR"
) > "$WORK/scope.out" 2>&1
RC=$?
sed 's/^/GUARD:   /' "$WORK/scope.out"
[ "$RC" -eq 0 ] && ok "scope: rc=0" || fail "scope: rc=$RC, expected 0"

for d in "$JOBROOT/uv-overlay/sdists-v9" "$JOBROOT/uv-overlay/builds-v0" \
         "$JOBROOT/pixi-overlay/uv-cache/sdists-v9" "$JOBROOT/pixi-overlay/uv-cache/builds-v0"; do
  if [ -L "$d" ]; then fail "scope: $d is a SYMLINK -- the build half is not job-local"
  elif [ ! -d "$d" ]; then fail "scope: $d does not exist"
  elif [ -n "$(find "$d" -mindepth 1 -maxdepth 1 -print -quit)" ]; then fail "scope: $d is NOT empty"
  else ok "scope: $(basename "$(dirname "$d")")/$(basename "$d") is a real, empty, job-local directory"; fi
done

# THE POISON MUST BE OUT OF REACH, and this is the assertion that ties the two
# halves together: the reader run against the SCOPED roots must be green while
# the same reader against the SHARED roots is red.
OUT=$WORK/reader-after-scope.out
bash "$READER" "$JOBROOT/uv-overlay" "$JOBROOT/pixi-overlay/uv-cache" > "$OUT" 2>&1; RC=$?
sed 's/^/GUARD:   /' "$OUT"
[ "$RC" -eq 0 ] && ok "reader against the SCOPED roots: GREEN" || fail "reader against the SCOPED roots: exit $RC"
bash "$READER" "$SH/uv" "$SH/pixi/uv-cache" > "$WORK/reader-shared.out" 2>&1; RC=$?
[ "$RC" -eq 3 ] && ok "reader against the SHARED roots: still RED -- the fix isolates, it does not delete" \
  || fail "reader against the SHARED roots: exit $RC, expected 3 (if this is 0 the fixture stopped being poisoned)"

# The byte-keyed halves must still be SHARED, or the overlay is just a cold cache.
for b in archive-v0 wheels-v6; do
  for ov in "$JOBROOT/uv-overlay" "$JOBROOT/pixi-overlay/uv-cache"; do
    if [ -L "$ov/$b" ]; then ok "scope: $(basename "$ov")/$b is a symlink -> $(readlink "$ov/$b")"
    else fail "scope: $ov/$b is not a symlink into the shared cache"; fi
  done
done
[ -L "$JOBROOT/pixi-overlay/pkgs" ] && ok "scope: pixi pkgs stays shared by symlink" || fail "scope: pixi pkgs is not a symlink"
[ -f "$JOBROOT/uv-overlay/CACHEDIR.TAG" ] && [ ! -L "$JOBROOT/uv-overlay/CACHEDIR.TAG" ] \
  && ok "scope: regular files are COPIED, not symlinked (nothing writes through them)" \
  || fail "scope: CACHEDIR.TAG was not copied as a regular file"

# NEGATIVE CONTROL 1: an overlay that already carries build state must be emptied.
JOBROOT2=$WORK/jobroot2
mkdir -p "$JOBROOT2/uv-overlay/sdists-v9/leftover"
( export UV_CACHE_DIR=$SH/uv PIXI_CACHE_DIR=$SH/pixi; retread_scope_sdist_builds "$JOBROOT2" ) >/dev/null 2>&1
if [ -e "$JOBROOT2/uv-overlay/sdists-v9/leftover" ]; then fail "scope: a pre-existing build tree SURVIVED into the overlay"
else ok "scope: a pre-existing build tree is cleared, not adopted"; fi

# NEGATIVE CONTROL 2: EXDEV. Only runs if the operator names a second filesystem.
if [ -n "${SCOPE_GUARD_XDEV:-}" ] && [ -d "$SCOPE_GUARD_XDEV" ]; then
  d1=$(stat -c %d "$SH/uv"); d2=$(stat -c %d "$SCOPE_GUARD_XDEV")
  if [ "$d1" = "$d2" ]; then
    echo "GUARD  ?? : SCOPE_GUARD_XDEV=$SCOPE_GUARD_XDEV is on the SAME device as the fixture; EXDEV control SKIPPED"
  else
    X=$SCOPE_GUARD_XDEV/sdist-scope-guard-xdev.$$
    ( export UV_CACHE_DIR=$SH/uv PIXI_CACHE_DIR=$SH/pixi; retread_scope_sdist_builds "$X" ) > "$WORK/xdev.out" 2>&1
    RC=$?
    rm -rf "$X"
    [ "$RC" -ne 0 ] && ok "EXDEV control: an overlay on another filesystem is REFUSED (rc=$RC)" \
      || fail "EXDEV control: a cross-device overlay was ACCEPTED -- uv's rename into archive-v0 would die"
    grep -q "different filesystem" "$WORK/xdev.out" && ok "EXDEV control: the refusal names the reason" || fail "EXDEV control: refusal did not name the filesystem"
  fi
else
  echo "GUARD  ?? : EXDEV control SKIPPED (set SCOPE_GUARD_XDEV to a dir on another filesystem, e.g. /tmp, to run it)"
fi

echo "GUARD: ==============================="
[ "$FAIL" -eq 0 ] && { echo "GUARD: ALL CHECKS PASSED"; exit 0; }
echo "GUARD: FAILURES ABOVE"; exit 1
