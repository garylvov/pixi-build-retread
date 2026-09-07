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
#   stage_mirror_reap_building <parent> [minage_m]  remove UNFINISHED .building temps
#   stage_mirror_repromote <dirty dir>         rename a false-positive quarantine back

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

# ── STAGE-MIRROR-3: the two things that can be done to a mirror parent ───────
#
# WHY THESE LIVE HERE AND NOT IN THE REAPER. The mirror parent
# `$STAGE_MIRROR_ROOT` is a DECLARED PERSISTENT root: `cleanup.sh` refuses every
# path under `agrescap/cache/retread` BY NAME, and `multiarm_store_reap.sh`
# refuses it the same way. That containment is correct and is not being
# loosened -- a general-purpose reaper let loose on the mirror parent is how a
# 10.72 GB mirror gets deleted on a verdict nobody measured, twice. What is
# needed instead is a verb narrow enough that it cannot express the dangerous
# act: each one below can touch exactly ONE class of name, proves its
# precondition by measurement, and prints a counted footer.
#
# THE TWO ARTEFACTS, both left by 2026-09-07 and both measured:
#
#   1. `<key>.building.<jid>-<TAG>-<pid>` -- a PARTIAL cp -al tree that was
#      never renamed into place. `85db7fdbbf51206a0cb57fa0d55e0e74.building.
#      999999-MH1-2247202` is one: hc14-guard 6023585 ran `bash
#      arms/mh1_relock.sh` with SLURM_JOB_ID forced to 999999, the template ran
#      for real, and the temp is what it got to before it was killed. 12908
#      entries against the finished mirror's 44117, and NO `.stage-mirror-key`
#      -- an unfinished build by construction, since the key is written last.
#      Note the job id: 999999 is not a job, it is a guard's placeholder, and a
#      non-existent id must read as NOT RUNNING rather than as unknown.
#
#   2. `<key>.DIRTY-<jid>-<TAG>-<epoch>` -- a mirror quarantined by a PRE-LOCK
#      verify. mCB-relock 6022684 quarantined `…74` at 09:42 as
#      `…74.DIRTY-6022684-MCB-1788788544` on a diff that was PURE REORDERING:
#      STAGE-MIRROR-2's locale bug, a census written under one collation and
#      re-walked under another. Measured at 10:26 with the one-producer census
#      below: a fresh `LC_ALL=C` walk of that quarantine is 44117 rows, and
#      `diff` against its own stored `.stage-mirror-manifest.tsv` is EMPTY --
#      md5 41faec6000ed5e269232e7d0b0e7c35b for both. It was never dirty. With
#      no live key present, re-promoting it saves a ~15 minute rebuild; the
#      alternative is to let the next relock rebuild 10.72 GB to get back a tree
#      we can prove we already have.
#
# NEITHER VERB DELETES A MIRROR. `reap_building` removes only unfinished temps,
# and `repromote` only RENAMES. Neither will act while the name it would touch
# could still belong to a running job, and neither will act on a bare key.

stage_mirror_reap_building () {   # $1 = mirror parent, $2 = min age minutes (default 60)
  local parent=$1 minage=${2:-60} d b jid qst age now n_seen=0 n_removed=0 n_refused=0
  [ -d "$parent" ] || { echo "### stage-reap: no mirror parent at $parent"; return 2; }
  now=$(date +%s)
  # -maxdepth 1 -mindepth 1: only direct children of the parent, never a walk.
  while IFS= read -r d; do
    [ -n "$d" ] || continue
    b=${d##*/}
    # THE NAME IS THE WHOLE PERMISSION. Anything that is not `<key>.building.<jid>-…`
    # is not this verb's business -- a bare key, a .DIRTY, a .SRCLINKED all fall
    # through untouched, and there is no flag that widens this.
    case "$b" in
      *.building.[0-9]*-*) ;;
      *) continue ;;
    esac
    n_seen=$((n_seen+1))
    jid=${b#*.building.}; jid=${jid%%-*}
    # A jid that no scheduler knows returns EMPTY here, which is NOT RUNNING.
    # 999999 is exactly that case and it must not read as "cannot tell".
    qst=$(squeue -j "$jid" -h -o '%t' 2>/dev/null | paste -sd,)
    if [ -n "$qst" ]; then
      echo "### stage-reap REFUSED $b: job $jid is still in the queue ($qst) -- it may be building this very temp"
      n_refused=$((n_refused+1)); continue
    fi
    age=$(( (now - $(stat -c %Y "$d")) / 60 ))
    if [ "$age" -lt "$minage" ]; then
      echo "### stage-reap REFUSED $b: mtime is ${age}m old, younger than the ${minage}m floor"
      n_refused=$((n_refused+1)); continue
    fi
    # A finished mirror carries a key. If one is here the name lied about being
    # a temp, and this verb has no business with a finished tree.
    if [ -f "$d/.stage-mirror-key" ]; then
      echo "### stage-reap REFUSED $b: it carries a .stage-mirror-key, so it is a FINISHED mirror wearing a temp's name"
      n_refused=$((n_refused+1)); continue
    fi
    echo "### stage-reap removing $b (job $jid not in queue, mtime ${age}m, entries $(find "$d" -maxdepth 6 2>/dev/null | wc -l), no key)"
    rm -rf -- "$d" && n_removed=$((n_removed+1))
  done <<EOF
$(find "$parent" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | LC_ALL=C sort)
EOF
  echo "### STAGE-REAP BUILDING parent=$parent seen=$n_seen removed=$n_removed refused=$n_refused minage=${minage}m"
  return 0
}

stage_mirror_repromote () {   # $1 = the .DIRTY-<...> directory to promote back
  local d=$1 parent b key live wd rc
  [ -d "$d" ] || { echo "### stage-repromote: no directory at $d"; return 2; }
  parent=$(cd -- "$(dirname -- "$d")" && pwd); b=${d##*/}
  case "$b" in
    *.DIRTY-*) ;;
    *) echo "### stage-repromote REFUSED $b: only a .DIRTY-<...> quarantine can be promoted"; return 2 ;;
  esac
  key=${b%%.DIRTY-*}
  live=$parent/$key
  # A LIVE KEY IS AN ABSOLUTE REFUSAL. Promoting over one would replace a mirror
  # some running job may be staged from, which is the accident this whole family
  # of files exists to prevent.
  if [ -e "$live" ]; then
    echo "### stage-repromote REFUSED $b: a live key already exists at $live -- nothing to restore"
    return 1
  fi
  [ -f "$d/.stage-mirror-manifest.tsv" ] || {
    echo "### stage-repromote REFUSED $b: no .stage-mirror-manifest.tsv, so the quarantine verdict cannot be re-tested"
    return 1; }
  wd=$(mktemp -d "${TMPDIR:-/tmp}/stage-repromote.XXXXXX") || return 2
  # ONE PRODUCER. This is `stage_mirror_census`, the same function every writer
  # and reader uses, so a promotion cannot be decided by a second implementation
  # of "what is in this tree" -- which is the defect STAGE-MIRROR-2 was.
  stage_mirror_census "$d" > "$wd/now.tsv"
  LC_ALL=C sort -- "$d/.stage-mirror-manifest.tsv" > "$wd/stored.tsv"
  if ! diff -q -- "$wd/stored.tsv" "$wd/now.tsv" >/dev/null 2>&1; then
    echo "### stage-repromote REFUSED $b: the tree does NOT match its own stored manifest -- it really is dirty. Diff head:"
    diff -- "$wd/stored.tsv" "$wd/now.tsv" 2>/dev/null | head -10 | sed 's/^/###   /'
    echo "### STAGE-REPROMOTE promoted=0 refused=1 key=$key stored_rows=$(wc -l < "$wd/stored.tsv") now_rows=$(wc -l < "$wd/now.tsv")"
    rm -rf "$wd"; return 1
  fi
  echo "### stage-repromote $b matches its stored manifest EXACTLY ($(wc -l < "$wd/now.tsv") rows, md5 $(md5sum "$wd/now.tsv" | awk '{print $1}')) -- the quarantine was a false positive"
  rm -rf "$wd"
  mv -T -- "$d" "$live" || { echo "### stage-repromote FAILED to rename $b -> $key"; return 1; }
  echo "### STAGE-REPROMOTE promoted=1 refused=0 key=$key from=$b entries=$(grep '^entries=' "$live/.stage-mirror-key" 2>/dev/null | cut -d= -f2)"
  return 0
}
