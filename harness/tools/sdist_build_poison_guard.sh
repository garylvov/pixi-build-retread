#!/usr/bin/env bash
# sdist_build_poison_guard.sh -- THE READER for `retread_scope_sdist_builds`.
#
# Run this in the phase template BEFORE any install. It walks every
# `CMakeCache.txt` REACHABLE BY THIS JOB inside a uv sdist build tree and
# REFUSES if one names a compiler that does not exist on this filesystem.
#
# WHY IT EXISTS (C31-4, LANE-C-WARM-LOG 31.10-31.12 and 33). A `CMakeCache.txt`
# records the ABSOLUTE compiler paths of the run that created it. uv builds an
# sdist IN PLACE under `<uv cache>/sdists-v9/pypi/<name>/<version>/<rev>/src/`,
# so with a shared uv cache the build's success depends on which workspace last
# built it. B-cert-4 spent four hours reaching a RED-install whose whole content
# was one absolute path into `ws.MN1-5761731`, a workspace reaped weeks earlier.
# This guard turns that into a refusal in the job header, before anything runs.
#
# WHAT IT CHECKS, and deliberately nothing more: the entries whose value IS a
# tool this build will exec --
#     CMAKE_{C,CXX,Fortran,CUDA,ASM}_COMPILER:FILEPATH
#     CMAKE_MAKE_PROGRAM:FILEPATH
# A `*-NOTFOUND` value is cmake's own "absent and known absent" and is skipped;
# a relative value is skipped (it resolves against PATH at build time, so its
# existence here proves nothing either way). Everything else must be an absolute
# path that exists.
#
# Usage:
#   sdist_build_poison_guard.sh [<uv cache root> ...]
# With no arguments it checks the two uv caches a retread job can reach:
#   $UV_CACHE_DIR   and   $PIXI_CACHE_DIR/uv-cache
# (pixi 0.73.0 does not read UV_CACHE_DIR -- 0 occurrences in the binary -- so
# the second one is a separate cache and is the one that was actually poisoned.)
#
# Exit codes:  0 clean   2 usage/undeterminable   3 POISON FOUND
set -u

ROOTS=()
if [ "$#" -gt 0 ]; then
  ROOTS=("$@")
else
  [ -n "${UV_CACHE_DIR:-}" ] && ROOTS+=("$UV_CACHE_DIR")
  [ -n "${PIXI_CACHE_DIR:-}" ] && ROOTS+=("$PIXI_CACHE_DIR/uv-cache")
fi
if [ "${#ROOTS[@]}" -eq 0 ]; then
  echo "sdist_build_poison_guard: FATAL no roots -- pass them, or export UV_CACHE_DIR / PIXI_CACHE_DIR first" >&2
  exit 2
fi

TOTAL_FILES=0 TOTAL_REFS=0 PRESENT=0 ABSENT=0 POISONED=0
BAD_ROWS=""

# DEDUPE BY REALPATH. `$UV_CACHE_DIR` and `$PIXI_CACHE_DIR/uv-cache` can name the
# same directory (a fixture, or a harness that points both at one root), and
# counting one tree twice would double every number in the census line.
SEEN=""
UNIQ=()
for r in "${ROOTS[@]}"; do
  rp=$(readlink -f "$r" 2>/dev/null) || rp=$r
  case "$SEEN" in *"|$rp|"*) continue;; esac
  SEEN="$SEEN|$rp|"
  UNIQ+=("$r")
done
ROOTS=("${UNIQ[@]}")

for root in "${ROOTS[@]}"; do
  sd=$root/sdists-v9
  if [ ! -d "$sd" ]; then
    echo "### sdist_poison_guard root=$root sdists-v9 ABSENT (nothing to check)"
    continue
  fi
  n=0
  while IFS= read -r f; do
    n=$((n+1)); TOTAL_FILES=$((TOTAL_FILES+1))
    bad_here=0
    while IFS= read -r p; do
      [ -n "$p" ] || continue
      case "$p" in
        *-NOTFOUND) continue;;
        /*) ;;
        *) continue;;
      esac
      TOTAL_REFS=$((TOTAL_REFS+1))
      if [ -e "$p" ]; then
        PRESENT=$((PRESENT+1))
      else
        ABSENT=$((ABSENT+1)); bad_here=1
        BAD_ROWS="$BAD_ROWS
  MISSING TOOL  $p
      named by  $f"
      fi
    done < <(grep -hE '^CMAKE_(C|CXX|Fortran|CUDA|ASM)_COMPILER:FILEPATH=|^CMAKE_MAKE_PROGRAM:FILEPATH=' "$f" 2>/dev/null | sed 's/^[^=]*=//')
    [ "$bad_here" = 1 ] && POISONED=$((POISONED+1))
  done < <(find "$sd" -maxdepth 10 -type f -name CMakeCache.txt 2>/dev/null)
  echo "### sdist_poison_guard root=$root cmakecache_files=$n"
done

echo "### sdist_poison_guard TOTAL cmakecache_files=$TOTAL_FILES tool_refs=$TOTAL_REFS present=$PRESENT absent=$ABSENT poisoned_files=$POISONED"

if [ "$POISONED" -gt 0 ]; then
  echo "sdist_build_poison_guard: REFUSING -- $POISONED cached CMakeCache.txt file(s) reachable by this job name $ABSENT tool path(s) that do not exist." >&2
  # shellcheck disable=SC2001
  echo "$BAD_ROWS" >&2
  echo "sdist_build_poison_guard: a CMakeCache.txt is a resolution of the BUILD ENVIRONMENT, not bytes keyed by url and hash. Call retread_scope_sdist_builds <job root> after retread_fast_env so this job builds sdists in its OWN tree. Do NOT delete from the shared cache to get past this." >&2
  exit 3
fi
echo "sdist_build_poison_guard: clean"
exit 0
