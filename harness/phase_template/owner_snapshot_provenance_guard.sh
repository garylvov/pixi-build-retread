#!/usr/bin/env bash
# owner_snapshot_provenance_guard.sh -- the reader for DET-1-6-a: the
# `### OWNER SNAPSHOT` row names WHERE THE FROZEN BYTES CAME FROM, and it is
# never the task pin standing in for a sha nobody read.
#
# THE DEFECT IT WOULD HAVE CAUGHT. owner_snapshot.sh read `src_commit` out of
# `<job root>/../tools/.harness_synced_commit` -- the TASK PIN -- and printed it
# as the provenance of files it had copied from somewhere else entirely.
# det16_proof.sh froze the WORKTREE's cleanup_gated.sh (6b3b669) while the row
# said src_commit=8108ca4, and the proof author had to write a hand-typed
# PROVENANCE NOTE with two md5sums underneath it so a reader would not be
# misled. A row that needs a correction printed beside it is the defect.
#
# THE ARMS, AND ARM A IS THAT INCIDENT AS A FIXTURE:
#   A. a git-worktree source WITH a task pin record beside it (det16's exact
#      shape): the row must carry src_kind=git and the WORKTREE's HEAD, and it
#      must ALSO print the pin as task_pin_record so neither fact is hidden.
#   B. a task-copy source (not a repo) with a record: src_kind=record and the
#      record's sha -- for a task copy the record IS the identity of the bytes.
#   C. a source that is neither: REFUSED, rc 2. The old code printed
#      `src_commit=unknown` and carried on, which puts a frozen copy of
#      nobody-knows-what in a job root for the hours a reap runs.
#   D. MUTATION: with the git branch of owner_src_identity disabled the SAME
#      arm-A fixture reports the RECORD's sha -- the original defect, reproduced
#      on demand. Without D, A could pass against a tool that always printed
#      whatever it found first.
#   E. the md5 half: every frozen file is stamped, `md5sum -c` is clean inside
#      the snapshot dir, and the stamps match the SOURCE files byte for byte.
#      A sha is a claim about bytes; the md5s are the bytes.
#
# Usage: owner_snapshot_provenance_guard.sh      (self-contained, needs $TMPDIR)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SNAPTOOL=$HERE/owner_snapshot.sh
REFS=$HERE/../tools/script_refs.sh
[ -f "$SNAPTOOL" ] || { echo "GUARD FATAL: $SNAPTOOL not found"; exit 2; }
[ -f "$REFS" ]     || { echo "GUARD FATAL: $REFS not found"; exit 2; }

W=$(mktemp -d "${TMPDIR:-/tmp}/owner-prov-guard.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
FAIL=0
fail () { echo "GUARD FAIL: $*"; FAIL=1; }
ok   () { echo "GUARD  ok : $*"; }

PIN=1111111111111111111111111111111111111111    # the TASK PIN: deliberately NOT any real sha

# ---- the gate the owner freezes, in its real shape: it sources a sibling -----
mkgate () {   # $1 = directory to write the pair into
  mkdir -p "$1"
  printf '#!/bin/bash\nCLEANUP=$(dirname "$0")/cleanup.sh\nbash "$CLEANUP"\n' > "$1/cleanup_gated.sh"
  printf '#!/bin/bash\necho cleanup\n' > "$1/cleanup.sh"
}
field () {    # $1 = log, $2 = field name -> its value off the files= row
  sed -n "s/^### OWNER SNAPSHOT files=.* $2=\([^ ]*\).*/\1/p" "$1" | head -1
}

########## A. a git worktree source, with a pin record beside it ##############
GR=$W/A/repo
mkgate "$GR/harness/phase_template"
git -C "$GR" init -q 2>/dev/null
git -C "$GR" config user.email guard@example.invalid
git -C "$GR" config user.name  guard
git -C "$GR" add -A >/dev/null 2>&1
git -C "$GR" commit -q -m v1 >/dev/null 2>&1
HEADA=$(git -C "$GR" rev-parse HEAD)
# the pin record the OLD code would have printed instead, in BOTH places it
# looked: beside the JOB ROOT (which is what the files= row reports as
# task_pin_record) and beside the SOURCE tree, which is where owner_src_identity
# falls back to when it cannot see git. Without the second one arm D's mutant
# has nothing to fall back TO and refuses instead of reproducing the defect
# (measured: job 6014994, arm D red with src_kind='').
mkdir -p "$W/A/jobroot" "$W/A/tools" "$GR/harness/tools"
printf '%s\n' "$PIN" > "$W/A/tools/.harness_synced_commit"
printf '%s\n' "$PIN" > "$GR/harness/tools/.harness_synced_commit"
bash "$SNAPTOOL" "$W/A/jobroot/jr" "$GR/harness/phase_template/cleanup_gated.sh" > "$W/A.log" 2>&1; rcA=$?
KA=$(field "$W/A.log" src_kind); SA=$(field "$W/A.log" src_commit)
if [ "$rcA" = 0 ] && [ "$KA" = git ] && [ "$SA" = "$HEADA" ]; then
  ok "A. a git-worktree source is named by its OWN HEAD: src_kind=git src_commit=$SA"
else
  fail "A. rc=$rcA src_kind='$KA' src_commit='$SA' (wanted git / $HEADA)"
  sed 's/^/GUARD:   /' "$W/A.log"
fi
grep -q "^### OWNER SNAPSHOT files=2 root=" "$W/A.log" \
  && ok "A. and it still froze the gate AND the cleanup.sh it sources (files=2)" \
  || fail "A. the snapshot set changed shape -- files=2 row absent"
if grep -q 'src_dirty=no' "$W/A.log"; then
  ok "A. the committed source is reported CLEAN (src_dirty=no)"
else
  fail "A. src_dirty was not reported no on a freshly committed tree"
fi
# and the dirty case, because a dirty file is NOT the commit it sits on
printf '#!/bin/bash\necho cleanup CHANGED\n' > "$GR/harness/phase_template/cleanup.sh"
bash "$SNAPTOOL" "$W/A/jobroot/jr2" "$GR/harness/phase_template/cleanup_gated.sh" > "$W/A2.log" 2>&1
if grep -q '^### OWNER SNAPSHOT froze file=cleanup.sh .* dirty=yes' "$W/A2.log"; then
  ok "A. an UNCOMMITTED source file is stamped dirty=yes, not silently attributed to HEAD"
else
  fail "A. a dirty source file was not flagged"
  grep '^### OWNER SNAPSHOT froze' "$W/A2.log" | sed 's/^/GUARD:   /'
fi

########## B. a task copy: not a repo, identified by its record ###############
mkgate "$W/B/task/merge-h"
mkdir -p "$W/B/task/tools" "$W/B/jobroot"
printf '%s\n' "$PIN" > "$W/B/task/tools/.harness_synced_commit"
bash "$SNAPTOOL" "$W/B/jobroot/jr" "$W/B/task/merge-h/cleanup_gated.sh" > "$W/B.log" 2>&1; rcB=$?
KB=$(field "$W/B.log" src_kind); SB=$(field "$W/B.log" src_commit)
if [ "$rcB" = 0 ] && [ "$KB" = record ] && [ "$SB" = "$PIN" ]; then
  ok "B. a task copy is named by the record beside it: src_kind=record src_commit=$SB"
else
  fail "B. rc=$rcB src_kind='$KB' src_commit='$SB' (wanted record / $PIN)"
  sed 's/^/GUARD:   /' "$W/B.log"
fi

########## C. neither -> REFUSE ###############################################
mkgate "$W/C/orphan"
mkdir -p "$W/C/jobroot"
bash "$SNAPTOOL" "$W/C/jobroot/jr" "$W/C/orphan/cleanup_gated.sh" > "$W/C.log" 2>&1; rcC=$?
if [ "$rcC" = 2 ] && grep -q 'REFUSED: cannot identify the source' "$W/C.log"; then
  ok "C. an unidentifiable source is REFUSED rc 2, not frozen under a guessed row"
else
  fail "C. rc=$rcC -- an unidentifiable source was accepted"
  sed 's/^/GUARD:   /' "$W/C.log"
fi

########## D. MUTATION: the git branch disabled -> the record wins #############
mkdir -p "$W/mut/phase_template" "$W/mut/tools" "$W/D/jobroot"
cp "$REFS" "$W/mut/tools/script_refs.sh"
sed 's|^  if git -C "\$d" rev-parse --git-dir >/dev/null 2>&1 &&$|  if false \&\&|' \
  "$SNAPTOOL" > "$W/mut/phase_template/owner_snapshot.sh"
if cmp -s "$SNAPTOOL" "$W/mut/phase_template/owner_snapshot.sh"; then
  fail "D. the mutation did not apply -- owner_src_identity's git branch was not found, so D is vacuous"
else
  # arm A's fixture again: a git worktree with a pin record beside the JOB ROOT
  bash "$W/mut/phase_template/owner_snapshot.sh" "$W/A/jobroot/jrmut" \
       "$GR/harness/phase_template/cleanup_gated.sh" > "$W/D.log" 2>&1
  KD=$(field "$W/D.log" src_kind); SD=$(field "$W/D.log" src_commit)
  if [ "$KD" = record ] && [ "$SD" = "$PIN" ]; then
    ok "D. MUTATION REPRODUCED: without the git branch the SAME fixture advertises the pin ($SD) for bytes that came from $HEADA -- DET-1-6-a exactly"
  else
    fail "D. the mutant did not reproduce the defect (src_kind='$KD' src_commit='$SD') -- arm A proves nothing"
    sed 's/^/GUARD:   /' "$W/D.log"
  fi
fi

########## E. the md5 half #####################################################
SNAP=$W/A/jobroot/jr/owner-snapshot
if [ -s "$SNAP/owner-snapshot.md5" ] && ( cd "$SNAP" && md5sum -c owner-snapshot.md5 >/dev/null 2>&1 ); then
  ok "E. every frozen file is stamped and md5sum -c is clean inside the snapshot dir ($(wc -l < "$SNAP/owner-snapshot.md5") files)"
else
  fail "E. owner-snapshot.md5 is missing or does not verify"
  [ -f "$SNAP/owner-snapshot.md5" ] && sed 's/^/GUARD:   /' "$SNAP/owner-snapshot.md5"
fi
SRCMD5=$(md5sum "$GR/harness/phase_template/cleanup_gated.sh" | awk '{print $1}')
if grep -q "^$SRCMD5  cleanup_gated.sh$" "$SNAP/owner-snapshot.md5" 2>/dev/null; then
  ok "E. the stamp matches the SOURCE file byte for byte ($SRCMD5)"
else
  fail "E. the stamped md5 does not match the source file"
fi
if [ -s "$SNAP/owner-snapshot.provenance" ] \
   && [ "$(wc -l < "$SNAP/owner-snapshot.provenance")" = "$(wc -l < "$SNAP/owner-snapshot.md5")" ]; then
  ok "E. the provenance table has one row per frozen file"
else
  fail "E. the provenance table and the md5 table disagree about how many files were frozen"
fi

echo
[ "$FAIL" = 0 ] && echo "DET-1-6-a GUARD: ALL GREEN" || echo "DET-1-6-a GUARD: SOME CHECKS FAILED"
exit "$FAIL"
