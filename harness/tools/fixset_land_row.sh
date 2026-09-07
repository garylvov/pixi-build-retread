#!/usr/bin/env bash
# fixset_land_row.sh -- add ONE landed row to the declared binsnap fix set and
# leave its TWO homes provably identical at a nameable commit.  MERGE-K-2.
#
# THE DEFECT.  `land.sh` appended the landed row to the TASK copy only:
#
#     grep -qE "^${FIXSET_ADD%% *} " "$T/tools/binsnap_fixset.txt" || \
#       printf '%s\n' "$FIXSET_ADD" >> "$T/tools/binsnap_fixset.txt"
#
# The fix set has two homes -- that task copy, which `binsnap_ancestry_guard.sh`
# reads, and `harness/tools/binsnap_fixset.txt`, which is the versioned one.
# `harness_drift_check.sh` md5s every mapped task file against
# `git cat-file blob <HARNESS_COMMIT>:<path>`, and `tools/binsnap_fixset.txt` is
# mapped.  So EVERY landing put the two copies exactly one row apart, and the
# next job to run behind `$HARNESS_COMMIT` died at its drift gate on a file no
# human had touched.  `efe74a0` closed the gap for B17 BY HAND, which is a
# snapshot, not a link -- the same shape C31-4-1 already paid for once, where
# four task copies sat silently behind the repo and nothing on either side
# noticed.
#
# THE SHAPE, AND WHY THIS ONE.  Two shapes were available.  The minimal one --
# write both files and PRINT the commit command for a human to run -- leaves the
# repo dirty and the sync depending on somebody reading a line of output at
# 03:00; a landing is unattended by construction, and `$HARNESS_COMMIT` cannot
# advance until that command runs, so every job in the gap still refuses.  It
# converts a silent break into a loud break that still needs a human.  So this
# script takes the other shape: it COMMITS the row in the worktree BY PATH and
# then RE-EXTRACTS the task copy FROM THAT COMMIT with `git cat-file blob`.  The
# task copy is therefore not a parallel write that happens to match -- it is the
# commit's own bytes, which is the only definition of "identical" the drift check
# accepts.  The new commit is printed as the `HARNESS_COMMIT` the next job must
# carry, and it exists BEFORE the fast-forward, so a refusal here is a refusal
# with nothing moved.
#
# WHY COMMITTING FROM A LANDING IS SAFE HERE.  It is `git commit -- <one path>`,
# a path-limited commit: concurrent lanes editing other files in this worktree
# are neither staged nor swept in, and this is NOT `merge`/`stash`/`checkout`/
# `reset`, none of which are valid on a dirty tree (CLAUDE.md law 11).  The one
# thing a path-limited commit could still sweep in is somebody else's UNCOMMITTED
# edit to the fix set itself, so that is refused explicitly below.  Nothing is
# pushed: `HARNESS_COMMIT` is read locally, and pushing the harness mirror is a
# separate, deliberate act.
#
#   usage: fixset_land_row.sh <harness-repo> <task-dir> "<sha> <name>"
#
#   rc 0  the row is present in both homes, they are byte-identical, and the
#         commit that says so is printed
#   rc 2  REFUSED before touching anything (bad arguments, missing file, the two
#         copies already diverged, or an uncommitted edit to the versioned copy)
#   rc 3  the commit or the re-extraction failed -- the repo copy is restored
#   rc 3  ALSO: harness_sync.sh refused, so the task copies were NOT advanced.
#         Its rc is printed and EXPLAINED -- rc 4 a PENDING job pinned to another
#         commit, rc 6 a RUNNING job of ours still READING a file this install
#         would rename over (HARNESS-SYNC-3-1). The sync's own rows are captured
#         to `<task>/.fixset_land_sync.out` and re-printed under the FATAL. This
#         script NEVER passes --force; overriding a read-set refusal is a
#         deliberate, hand-proved act, never an unattended landing's.
#
# Reader: land_fixset_sync_guard.sh, which lands a row into a THROWAWAY fixture
# repo and asserts the two copies are byte-identical afterwards, and replays the
# pre-MERGE-K-2 append from the pinned commit to show that same assertion failing.
set -uo pipefail
export PATH=/users/glvov/.pixi/bin:/users/glvov/.local/bin:$PATH   # git-lfs, or the commit dies

REPO="${1:?usage: fixset_land_row.sh <harness-repo> <task-dir> \"<sha> <name>\"}"
TASK="${2:?usage: fixset_land_row.sh <harness-repo> <task-dir> \"<sha> <name>\"}"
ROW="${3:?usage: fixset_land_row.sh <harness-repo> <task-dir> \"<sha> <name>\"}"

RREL=harness/tools/binsnap_fixset.txt
RPATH=$REPO/$RREL
TPATH=$TASK/tools/binsnap_fixset.txt
KEY=${ROW%% *}
MREL=harness/MANIFEST.md5
MPATH=$REPO/$MREL
MKEY=tools/binsnap_fixset.txt
# MERGE-L-1. `md5sum -c MANIFEST.md5` has been DIRTY in the harness worktree
# since the first landing that used this helper (8e5b64a: one file changed,
# MANIFEST.md5 untouched) -- the manifest is the reader that says the tree is
# what it claims, and a landing that silently invalidates it hands the next
# lane a failure it did not cause. The row is rewritten and committed in the
# SAME path-limited commit as the fix set, so the two can never disagree.

case "$KEY" in
  ''|"$ROW") echo "### FIXSET REFUSE: row '$ROW' is not '<sha> <name>'"; exit 2;;
esac
[ -f "$RPATH" ] || { echo "### FIXSET REFUSE: no versioned fix set at $RPATH"; exit 2; }
[ -f "$TPATH" ] || { echo "### FIXSET REFUSE: no task fix set at $TPATH"; exit 2; }
git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || {
  echo "### FIXSET REFUSE: $REPO is not a git repository"; exit 2; }

# The two copies must ALREADY agree, or this landing would bury an existing
# divergence under a new row. Direct file arguments -- a piped compare is a known
# false-mismatch source in this environment (CLAUDE.md law 15).
if ! cmp -s "$RPATH" "$TPATH"; then
  echo "### FIXSET REFUSE: the two fix-set copies already differ, and nothing was appended."
  echo "###   repo $RPATH md5=$(md5sum "$RPATH" | awk '{print $1}')"
  echo "###   task $TPATH md5=$(md5sum "$TPATH" | awk '{print $1}')"
  echo "###   Reconcile them from a commit first, never by editing one side:"
  echo "###     git -C $REPO cat-file blob <commit>:$RREL > $TPATH"
  exit 2
fi

# A path-limited commit would sweep in an uncommitted edit somebody else is
# holding in this file. Refuse instead of stealing it.
if ! git -C "$REPO" diff --quiet -- "$RREL" "$MREL" || ! git -C "$REPO" diff --cached --quiet -- "$RREL" "$MREL"; then
  echo "### FIXSET REFUSE: $RREL has an uncommitted edit in $REPO -- committing the landed row"
  echo "###   would sweep somebody else's work into a merge-queue commit. Commit or drop it first."
  git -C "$REPO" status --porcelain -- "$RREL"
  exit 2
fi

if grep -qE "^${KEY} " "$RPATH"; then
  echo "### FIXSET already carries $KEY -- nothing appended (idempotent re-run)"
  HC=$(git -C "$REPO" rev-parse HEAD)
else
  BACKUP=$(mktemp "${TMPDIR:-/tmp}/fixset_land_row.XXXXXX") || exit 3
  cp -f "$RPATH" "$BACKUP" || exit 3
  printf '%s\n' "$ROW" >> "$RPATH"
  if [ -f "$MPATH" ] && grep -qE "  ${MKEY}\$" "$MPATH"; then
    NEWMD5=$(md5sum "$RPATH" | awk '{print $1}')
    TMPM=$(mktemp "${TMPDIR:-/tmp}/fixset_manifest.XXXXXX") || exit 3
    awk -v k="$MKEY" -v m="$NEWMD5" '{ if ($2 == k) print m "  " k; else print }' "$MPATH" > "$TMPM" \
      && mv -f "$TMPM" "$MPATH" || { echo "### FIXSET FATAL: could not rewrite $MPATH"; cp -f "$BACKUP" "$RPATH"; exit 3; }
    MPATHS=("$RREL" "$MREL")
  else
    echo "### FIXSET WARN: no $MKEY row in $MPATH -- committing the fix set alone"
    MPATHS=("$RREL")
  fi
  MSG=$(mktemp "${TMPDIR:-/tmp}/fixset_land_msg.XXXXXX") || exit 3
  {
    printf 'harness: carry %s into the versioned fix set (MERGE-K-2)\n\n' "$KEY"
    printf 'Appended by land.sh as part of the landing, so the task copy can be\n'
    printf 're-extracted from this commit and HARNESS_COMMIT can advance to it.\n\n'
    printf 'row: %s\n' "$ROW"
  } > "$MSG"
  # -F, never -m: a `-m` message with punctuation in it has bitten this campaign.
  if ! git -C "$REPO" commit -q -F "$MSG" -- "${MPATHS[@]}"; then
    echo "### FIXSET FATAL: the commit failed -- restoring $RPATH and touching nothing else"
    cp -f "$BACKUP" "$RPATH"; rm -f "$BACKUP" "$MSG"; exit 3
  fi
  rm -f "$BACKUP" "$MSG"
  HC=$(git -C "$REPO" rev-parse HEAD)
  echo "### FIXSET appended $KEY and committed it as $HC"
fi

# LAND-SYNC-1. The task copy is installed BY THE ONE WRITER, not by this script.
#
# This block used to be a bare `git cat-file blob "$HC:$RREL" > "$TPATH"`, which
# is precisely the shape HARNESS-SYNC-1 abolished one layer up: it writes ONE
# task file from a commit, records nothing, and knows nothing about the queue.
# The consequences were both real and both observed on 2026-09-06. (a) A landing
# left `tools/.harness_synced_commit` naming the PREVIOUS commit while
# `tools/binsnap_fixset.txt` carried the new one, so `harness_sync.sh --check`
# reported the fix set as EDITED -- a landing making the drift reader accuse the
# next lane of a hand edit nobody made. (b) The rest of the mapped set stayed at
# whatever commit it was already at, so a job pinned to the printed
# `HARNESS_COMMIT` could still die at its drift gate on a file this landing
# never touched.
#
# Calling the writer fixes both by construction: it installs THE WHOLE MAPPED
# SET at `$HC`, md5-verifies every file against the blob, rename-installs so a
# live job keeps its inode, records `.harness_synced_commit`, and prints the
# drift line from the new state.
#
# AND IT CAN REFUSE, WHICH IS THE POINT, NOT A REGRESSION. `harness_sync.sh`
# exits 4 when a PENDING job of ours is pinned to a different commit -- a
# landing that would strand a queued job now says so, by name, at the moment it
# happens, instead of that job dying three hours later on "drift". land.sh
# treats a non-zero here as its exit-11 refusal WITH NOTHING MOVED except the
# harness commit, which is idempotent: re-running the landing after the queue
# drains finds the row already present and takes the `already carries` arm.
SYNC=$TASK/tools/harness_sync.sh
[ -f "$SYNC" ] || SYNC=$REPO/harness/tools/harness_sync.sh
[ -f "$SYNC" ] || { echo "### FIXSET FATAL: no harness_sync.sh at $TASK/tools or $REPO/harness/tools"; exit 3; }
# HARNESS-SYNC-3-1. The sync's own rows are CAPTURED, not just streamed, because
# the refusal this block has to explain is a per-file row printed BEFORE the
# summary -- and a landing is unattended, so "it was on stdout somewhere" is not
# a reader. `tee` into the job root keeps them on stdout AND on disk; the rows
# are then re-read from that file with `grep`, never `tail`, because the rows
# that matter are at the TOP of a refusal and a tail would cut exactly them.
SYNCLOG=$TASK/.fixset_land_sync.out
HARNESS_TASK_DIR="$TASK" HARNESS_REPO="$REPO" bash "$SYNC" "$HC" 2>&1 | tee "$SYNCLOG"
src=${PIPESTATUS[0]}
if [ "$src" != 0 ]; then
  echo "### FIXSET FATAL: harness_sync.sh $HC exited $src -- the task copies were NOT advanced to this landing's commit."
  echo "###   the sync's own rows are captured at $SYNCLOG"
  case "$src" in
    4)
      echo "###   rc 4 means a PENDING job of ours is pinned to a different commit and this sync would strand it;"
      echo "###   let it start or drain, then re-run the landing (the fix-set row is idempotent)."
      # HARNESS-SYNC-4-1. The sync now computes the READ-SET check too before it
      # exits 4, so an rc-4 refusal may ALSO carry rc-6 rows. Printing them here
      # is the reader half: without it the lander is told "drain the pinned job",
      # drains it, re-runs, and is handed a second wait it was never shown. That
      # is exactly what this landing did on 2026-09-06T22:41.
      if grep -q '### SYNC REFUSED rc=6' "$SYNCLOG" 2>/dev/null; then
        RJIDS4=$(sed -n 's/.*### SYNC REFUSED rc=6 running=\([0-9][0-9]*\).*/\1/p' "$SYNCLOG" | sort -u | tr '\n' ' ')
        echo "###   AND rc 6 IS OWED TOO -- the same run's read-set check ALSO refused. Repinning or"
        echo "###   draining the rc-4 job does NOT make this sync legal on its own. Its rows:"
        grep -F -- '### SYNC REFUSED rc=6' "$SYNCLOG" | sed 's/^/###     from harness_sync: /'
        echo "###   ACTUATOR: job(s) ${RJIDS4:-<none named -- read the rows above>} must ALSO finish (a"
        echo "###   state=PENDING one must RUN and finish, not merely start) before the landing is re-run."
      else
        echo "###   (the same run's read-set check did NOT refuse: no rc-6 row is on the page.)"
      fi
      ;;
    6)
      RJIDS=$(sed -n 's/.*### SYNC REFUSED rc=6 running=\([0-9][0-9]*\).*/\1/p' "$SYNCLOG" | sort -u | tr '\n' ' ')   # RC6-MSG
      grep -F -- '### SYNC REFUSED' "$SYNCLOG" | sed 's/^/###   from harness_sync: /'                                 # RC6-MSG
      echo "###   rc 6 means a job of ours would READ a file this sync would install. Each row carries"              # RC6-MSG
      echo "###   state=, and the two states are refused for two different reasons:"                                  # RC6-MSG
      echo "###   state=RUNNING -- a RUNNING job of ours is still READING that file right now. The install"           # RC6-MSG
      echo "###   is a rename from THIS node; the reader is on another NFS client, so its inode"                      # RC6-MSG
      echo "###   is unlinked under it, bash reads an error, calls it EOF, and that job exits 0 half-run."             # RC6-MSG
      echo "###   state=PENDING -- the job has not STARTED, so it would run whatever bytes this sync leaves"           # RC6-MSG
      echo "###   behind rather than the ones it was submitted against. An afterany cleanup owner submitted"           # RC6-MSG
      echo "###   with --wrap owns no pin dir, so rc 4 never saw it either (HARNESS-SYNC-4)."                          # RC6-MSG
      echo "###   ACTUATOR: wait for job(s) ${RJIDS:-<none named -- read the rows above>} to FINISH (a state=PENDING"   # RC6-MSG
      echo "###   job must run AND finish, not merely start) and re-run"                                              # RC6-MSG
      echo "###   the landing (the fix-set row is idempotent); or, ONLY after proving by hand that the install"        # RC6-MSG
      echo "###   set and that job's read set are disjoint, re-run the sync with --force --reason \"<why>\"."          # RC6-MSG
      echo "###   THIS SCRIPT NEVER FORCES ON ITS OWN: a landing that overrides a read-set refusal unattended"         # RC6-MSG
      echo "###   is the truncation this refusal exists to prevent."                                                   # RC6-MSG
      ;;
    *)
      echo "###   rc $src is neither 4 (a PENDING job pinned elsewhere) nor 6 (a RUNNING or PENDING job reading an"
      echo "###   installed file) -- read the captured rows at $SYNCLOG before re-running anything."
      ;;
  esac
  exit 3
fi
if ! cmp "$RPATH" "$TPATH"; then
  echo "### FIXSET FATAL: the two copies differ AFTER the sync -- read them both by hand"; exit 3
fi
MD5=$(md5sum "$TPATH" | awk '{print $1}')
ROWS=$(grep -cE '^[0-9a-f]{7,40} ' "$TPATH")
echo "### FIXSET SYNCED rows=$ROWS md5=$MD5 repo=$RPATH task=$TPATH"
echo "### FIXSET HARNESS_COMMIT=$HC  <-- every job after this landing must carry THIS commit"
echo "###   drift check:  bash $TASK/tools/harness_drift_check.sh $HC"
exit 0
