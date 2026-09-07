#!/usr/bin/env bash
# stage_mirror.sh -- THE ONE AUTHORITY for "does this tree share inodes with the
# read-only canonical source", sourced by every writer and every reader of the
# shared stage mirror. PROOF-SMOKE-1-7.
#
# WHAT WENT WRONG, MEASURED ON det161b-proof 6014471 (2026-09-07). Two rows,
# adjacent, in D16-6014471-W1.out:
#     ### stage: no mirror manifest -- cannot check inode disjointness
#     ### stage: mirror shares inodes with /oscar/data/stellex/glvov/imprint-data -- quarantining and rebuilding
# The second row is a LIE, and the first says why: the check never ran. The
# mirror it quarantined as `…74.SRCLINKED-6014471` was published at 03:55:59 by
# smoke job 6013332 (`.stage-mirror-key`: built_by_job=6013332), and a
# smoke-published mirror carries a `.stage-mirror-key` and NO
# `.stage-mirror-manifest.tsv` -- read-only `find -maxdepth 1` on the quarantine
# confirms it, against the relock-built `…74` and `…74.SRCLINKED-5691063` which
# both carry a 5.19 MB manifest. phaseN_relock.sh's disjointness check reads
# that manifest to know what to sample, so on a smoke-published mirror it
# returned 1 for "cannot check" -- the SAME rc it returns for "hardlinked" --
# and the caller printed the hardlink story. 10.72 GB were rebuilt on a verdict
# nobody had measured, and the mirror was renamed after a defect it did not have.
#
# THREE THINGS ARE WRONG THERE AND THIS FILE FIXES ALL THREE.
#
#   1. THE CHECK DEPENDED ON THE WRITER'S BOOKKEEPING. A tree is walkable; there
#      is no reason to ask a manifest what a tree contains. The check here
#      ENUMERATES the tree itself, so it works on any mirror whoever built it.
#
#   2. IT SAMPLED. Both halves sampled 50 files -- proof_smoke.sh's publish gate
#      through `shuf -n 50` over its own find, phaseN_relock.sh's adopt gate
#      through `shuf -n 50` over the manifest -- and a publish gated on 50 of
#      44,113 files is gated on nothing: one hardlinked file in a mirror is
#      enough to write through into imprint-data from every job staged off it,
#      and a 50-file sample finds it about one time in 900. Enumeration is also
#      CHEAPER than the sample it replaces: two `find -printf` walks, no per-file
#      `stat` fork, against 100 forks for 50 files.
#
#   3. IT COMPARED INODE NUMBERS ALONE. An inode number is only meaningful
#      beside its device: two files on different filesystems may share one and
#      be unrelated. The key here is (device, inode).
#
# AND "CANNOT CHECK" IS ITS OWN ANSWER: rc 2, never folded into rc 1, so no
# caller can ever again print a hardlink verdict for a check that did not run.
#
#   . "$(dirname "$0")/stage_mirror.sh"
#
#   stage_mirror_inode_check <tree> <source>   rc 0 disjoint, 1 SHARED, 2 cannot check
#   stage_mirror_census <tree>                 the census a mirror publishes

STAGE_MIRROR_SHARED_SHOW=${STAGE_MIRROR_SHARED_SHOW:-10}   # how many shared paths to print

stage_mirror_census () {         # $1 = tree -> the census, minus the mirror's own stamps
  # Same shape and the same LC_ALL=C pin as phaseN_relock.sh's stage_manifest --
  # a census written under one locale and re-walked under another is the
  # false-FATAL census_collation_guard.sh exists for. The two are asserted
  # IDENTICAL on a fixture by stage_mirror_guard.sh rather than shared as text,
  # because stage_manifest is extracted verbatim by two other guards.
  find "$1" -mindepth 1 -xdev -printf '%y\t%s\t%T@\t%P\n' | grep -vF '.stage-mirror-' | LC_ALL=C sort
}

stage_mirror_inode_check () {    # $1 = tree, $2 = source tree
  local t=$1 s=$2 wd n shared
  if [ ! -d "$t" ] || [ ! -d "$s" ]; then
    echo "### stage: mirror-vs-source inode check CANNOT RUN: tree='$t' source='$s' -- one of them is not a directory."
    echo "###   This is rc 2 and NOT a hardlink verdict. A caller that treats it as one"
    echo "###   quarantines a mirror for a defect nobody measured (job 6014471)."
    return 2
  fi
  wd=$(mktemp -d "${TMPDIR:-/tmp}/stage-mirror-inode.XXXXXX") || {
    echo "### stage: mirror-vs-source inode check CANNOT RUN: no writable TMPDIR"; return 2; }
  # (device, inode) keyed by path relative to each root. ONE walk each, no
  # per-file stat fork: %D is the device and %i the inode, and a pair is only
  # the same file when BOTH match.
  find "$t" -xdev -type f -printf '%P\t%D\t%i\n' 2>/dev/null | grep -vF '.stage-mirror-' | LC_ALL=C sort > "$wd/tree"
  find "$s" -xdev -type f -printf '%P\t%D\t%i\n' 2>/dev/null | LC_ALL=C sort > "$wd/src"
  if [ ! -s "$wd/tree" ] || [ ! -s "$wd/src" ]; then
    echo "### stage: mirror-vs-source inode check CANNOT RUN: tree files=$(wc -l < "$wd/tree") source files=$(wc -l < "$wd/src")"
    echo "###   An empty side makes the check VACUOUS, and a vacuous check that returns"
    echo "###   0 is worse than no check. rc 2, and still not a hardlink verdict."
    rm -rf "$wd"; return 2
  fi
  awk -F'\t' 'NR==FNR{d[$1]=$2; i[$1]=$3; next}
              ($1 in d){ n++; if (d[$1]==$2 && i[$1]==$3) { s++; if (s<=show) print "###   SHARED INODE dev=" $2 " ino=" $3 "  " $1 } }
              END{ print "@@" (n+0) "\t" (s+0) }' show="$STAGE_MIRROR_SHARED_SHOW" \
      "$wd/src" "$wd/tree" > "$wd/out"
  grep -v '^@@' "$wd/out"
  n=$(sed -n 's/^@@\([0-9]*\)\t.*/\1/p' "$wd/out")
  shared=$(sed -n 's/^@@[0-9]*\t\([0-9]*\)$/\1/p' "$wd/out")
  rm -rf "$wd"
  echo "### stage: mirror-vs-source inode check: enumerated ${n:-0} shared ${shared:-0} (want 0)"
  if [ "${shared:-1}" != 0 ]; then
    echo "### stage: the tree is HARDLINKED to $s. Every job staged from it writes"
    echo "###        through into the read-only canonical tree."
    return 1
  fi
  return 0
}
