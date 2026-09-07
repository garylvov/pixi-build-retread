#!/usr/bin/env bash
# store_census_snapshot_guard.sh -- the reader for phaseN_relock.sh's frozen
# store census (READERS-2-1).
#
# THE DEFECT IT WOULD HAVE CAUGHT. B29's relock judged its git-snapshot landing
# criterion against the SHARED persistent store root, read LIVE at judgement
# time. Two runs of the same relock over the same landing reported census_rows 5
# and then 3 -- the landing did not change, the store did, because other jobs
# write it while a lock is running. A criterion whose input can move under it is
# a coin, and this one had been flipping in silence.
#
# WHAT IT ASSERTS, against the REAL functions lifted out of the shipped
# template:
#
#   A. a fixture store with 5 generation directories -> the snapshot row says
#      rows=5 and the file it names exists and lists them.
#   B. ADD a sixth generation AFTER the snapshot -> the criterion still judges 5
#      and the RELEASE row carries BOTH numbers (frozen=5 live=6 delta=1). This
#      is the whole point: the verdict is frozen, the drift is still reported.
#   C. a label nobody snapshotted -> REFUSES, non-zero, naming its actuator. A
#      silent fall back to a live read is the defect reintroduced by omission,
#      so "no snapshot" must never be "read the store instead".
#   D. an EMPTY store -> the criterion refuses, so "nothing over-age" and
#      "nowhere to look" cannot be confused. Without D, A and B would both pass
#      on a judge that always says ok.
#
#   MUTATION. Rewrite store_census_judge_present to read the LIVE root instead
#   of the frozen file -- the exact code that was there before -- and arm B must
#   go RED (it judges 6). If B still passed, B would be proving nothing.
#
# Usage: store_census_snapshot_guard.sh [<template>]   (needs only $TMPDIR)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TPL=${1:-$HERE/phaseN_relock.sh}
[ -f "$TPL" ] || { echo "GUARD FATAL: no template at $TPL"; exit 2; }

W=$(mktemp -d "${TMPDIR:-/tmp}/store-census-guard.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
FAIL=0
fail () { echo "GUARD FAIL: $*"; FAIL=1; }
ok   () { echo "GUARD  ok : $*"; }

extract () {
  awk -v fn="$2" '$0 ~ "^"fn" \\(\\) \\{" {p=1} p {print} p && /^\}$/ {exit}' "$1"
}

FNS=
for f in store_census_live_rows store_census_snapshot store_census_frozen_rows \
         store_census_release store_census_judge_present; do
  t=$(extract "$TPL" "$f")
  [ -n "$t" ] || { echo "GUARD FATAL: $TPL has no $f -- the census has no frozen producer"; exit 2; }
  FNS="$FNS$t"$'\n'
done
ONELINER=$(grep -m1 '^store_census_file () ' "$TPL")
[ -n "$ONELINER" ] || { echo "GUARD FATAL: $TPL has no store_census_file"; exit 2; }

########## fixture store: 5 generation dirs + the reap try-lock FILE beside them
STORE=$W/store/canonical-git-sources
mkdir -p "$STORE"
for v in v1 v2 v3 v4 v5; do mkdir -p "$STORE/$v"; done
# The reaper's walk skips this; so must the count, or the criterion would
# disagree with the walk it judges.
: > "$STORE/.canonical-git-sources.reap-v1.lock"

mk_driver () {   # $1=path  $2=extra function text (mutations) 
  { echo 'set -u'
    echo 'A=$JOBROOT; TAG=GUARD; J=0'
    printf '%s\n' "$ONELINER"
    printf '%s\n' "$FNS"
    [ -n "${2:-}" ] && printf '%s\n' "$2"
    echo 'set +u'
    echo '"$@"'; } > "$1"
}
mk_driver "$W/drv.sh"
# THE MUTATION: the criterion reads the live root again.
MUTFN='store_census_judge_present () {
  local label=$1 live
  live=$(store_census_live_rows "$MUT_ROOT")
  echo "### CENSUS CRITERION label=$label ok: frozen generation_dirs=$live (judged from the snapshot file, never from a live re-read)"
}'
mk_driver "$W/drv.mut.sh" "$MUTFN"

JR=$W/jobroot; mkdir -p "$JR"
JRM=$W/jobroot-mut; mkdir -p "$JRM"

########## A. the snapshot names its file and counts the generations ##########
OUT=$(JOBROOT=$JR bash "$W/drv.sh" store_census_snapshot git-snapshots "$STORE" 2>&1)
CFILE=$(printf '%s\n' "$OUT" | sed -n 's/^### CENSUS SNAPSHOT rows=[0-9]* file=//p' | head -1)
if printf '%s\n' "$OUT" | grep -q '^### CENSUS SNAPSHOT rows=5 file=' \
   && [ -n "$CFILE" ] && [ -f "$CFILE" ] && [ "$(grep -c '^d' "$CFILE")" = 5 ]; then
  ok "A: 5 generation dirs -> '### CENSUS SNAPSHOT rows=5 file=$CFILE', and the file lists exactly those 5"
else
  fail "A: out: $(printf '%s' "$OUT" | tr '\n' '|') cfile='$CFILE'"
fi
# the try-lock FILE beside the generations must not be counted as one
if [ -f "$CFILE" ] && grep -q 'reap-v1.lock' "$CFILE" && [ "$(grep -c '^d' "$CFILE")" = 5 ]; then
  ok "A2: the reap try-lock FILE is listed in the snapshot but is NOT counted as a generation"
else
  fail "A2: the snapshot's treatment of the non-directory entry is wrong"
fi

########## B. the store grows AFTER the snapshot; the verdict must not #########
mkdir -p "$STORE/v6"
OUTJ=$(JOBROOT=$JR bash "$W/drv.sh" store_census_judge_present git-snapshots 2>&1); RCJ=$?
OUTR=$(JOBROOT=$JR bash "$W/drv.sh" store_census_release git-snapshots "$STORE" 2>&1)
if [ "$RCJ" = 0 ] && printf '%s\n' "$OUTJ" | grep -q 'frozen generation_dirs=5' \
   && printf '%s\n' "$OUTR" | grep -q '^### CENSUS RELEASE label=git-snapshots frozen=5 live=6 delta=1 file='; then
  ok "B: a 6th generation appears after the snapshot -- the criterion still judges 5, and one RELEASE row carries frozen=5 live=6 delta=1"
else
  fail "B: rc=$RCJ judge: $(printf '%s' "$OUTJ" | tr '\n' '|') release: $(printf '%s' "$OUTR" | tr '\n' '|')"
fi

########## B-mut. judge-from-live restored -> B must go RED ###################
OUTM=$(JOBROOT=$JR MUT_ROOT=$STORE bash "$W/drv.mut.sh" store_census_judge_present git-snapshots 2>&1)
if printf '%s\n' "$OUTM" | grep -q 'frozen generation_dirs=6'; then
  ok "B-mut: with the criterion reading the LIVE root it reports 6 -- so B's 5 is the frozen file talking, not an accident"
else
  fail "B-mut: the live-reading mutant did not report 6: $(printf '%s' "$OUTM" | tr '\n' '|')"
fi

########## C. a label nobody snapshotted -> REFUSE, never fall back ###########
OUTC=$(JOBROOT=$JRM bash "$W/drv.sh" store_census_judge_present git-snapshots 2>&1); RCC=$?
if [ "$RCC" != 0 ] && printf '%s\n' "$OUTC" | grep -q 'FATAL CENSUS' \
   && printf '%s\n' "$OUTC" | grep -q 'ACTUATOR:' \
   && ! printf '%s\n' "$OUTC" | grep -q 'CENSUS CRITERION .* ok'; then
  ok "C: an un-snapshotted label REFUSES (rc=$RCC), names its actuator, and does NOT quietly read the live store"
else
  fail "C: rc=$RCC out: $(printf '%s' "$OUTC" | tr '\n' '|')"
fi

########## D. an empty store -> the criterion refuses #########################
EMPTY=$W/empty/canonical-git-sources; mkdir -p "$EMPTY"
JRE=$W/jobroot-empty; mkdir -p "$JRE"
JOBROOT=$JRE bash "$W/drv.sh" store_census_snapshot git-snapshots "$EMPTY" >/dev/null 2>&1
OUTD=$(JOBROOT=$JRE bash "$W/drv.sh" store_census_judge_present git-snapshots 2>&1); RCD=$?
if [ "$RCD" != 0 ] && printf '%s\n' "$OUTD" | grep -q 'REFUSED: the frozen listing counted ZERO'; then
  ok "D: an empty store is REFUSED (rc=$RCD) -- 'nothing over-age' and 'nowhere to look' stay distinct, so the judge is not a constant ok"
else
  fail "D: rc=$RCD out: $(printf '%s' "$OUTD" | tr '\n' '|')"
fi

########## E. the template actually CALLS it before the lock ##################
SNAP_LN=$(grep -n '^store_census_snapshot git-snapshots ' "$TPL" | head -1 | cut -d: -f1)
LOCK_LN=$(grep -n '^/usr/bin/time -v -o "\$LTIME" "\$PIXI" lock -v' "$TPL" | head -1 | cut -d: -f1)
REL_LN=$(grep -n '^store_census_release git-snapshots ' "$TPL" | head -1 | cut -d: -f1)
if [ -n "$SNAP_LN" ] && [ -n "$LOCK_LN" ] && [ -n "$REL_LN" ] \
   && [ "$SNAP_LN" -lt "$LOCK_LN" ] && [ "$REL_LN" -gt "$LOCK_LN" ]; then
  ok "E: the template snapshots BEFORE the lock and releases AFTER it -- the capability has a production call site, in the right order"
else
  fail "E: snapshot/lock/release ordering in the template is wrong (snap=$SNAP_LN lock=$LOCK_LN release=$REL_LN)"
fi

echo "### store_census_snapshot_guard: $( [ "$FAIL" = 0 ] && echo PASS || echo FAIL )"
exit "$FAIL"
