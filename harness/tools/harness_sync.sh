#!/usr/bin/env bash
# harness_sync.sh -- THE ONE WRITER OF TASK COPIES.  HARNESS-SYNC-1.
#
# WHY THIS EXISTS.  `harness_drift_check.sh` made a stale task copy REFUSE
# instead of certifying the wrong harness, and that half works.  What it never
# fixed is WHO WRITES the task copies: three lanes now edit one shared task-dir
# harness, each re-extracting whatever files it happened to care about with a
# bare `git cat-file blob <its own commit>:<path> > <task path>`, and the drift
# check only surfaces the collision AFTERWARDS, in somebody else's job.  That is
# exactly what the 08:45-09:00 refusals in ticks 499-501 were: jobs pinned to
# `9463cf3` died at their drift gate because another lane had re-extracted task
# copies from a NEWER commit while they sat in the queue.  Nobody edited those
# jobs' files by hand and nobody did anything wrong; there was simply no writer
# that knew about the queue.
#
# SO THIS IS THAT WRITER, AND IT KNOWS THREE THINGS THE BARE `cat-file` DID NOT.
#   1. THE WHOLE SET, not the files one lane remembered.  A sync leaves the task
#      dir AT a commit, so the next job's drift gate passes for every file, not
#      for the two that lane touched.
#   2. THE QUEUE.  A job that is PENDING has NOT snapshotted anything -- its
#      pin file already names a commit, and moving the task dir off that commit
#      kills it the moment it starts.  So a sync that would strand a queued job
#      REFUSES (rc 4) and NAMES it, before writing one byte.  AND AFTERWARDS
#      (DET-1-1) it prints the PIN REPORT: every job root whose pin no longer
#      names the synced commit, classed PENDING / RUNNING / no-job, because the
#      rc-4 refusal is a PRE-condition and cannot see a job submitted a minute
#      LATER off a stale pin -- which is how MERGE-T lost a relock.
#   3. RENAME-INSTALL.  Every file goes down as a temp file in the TARGET
#      directory and then `mv -f` over the name.  bash reads a script
#      INCREMENTALLY: overwriting a file a running job is executing feeds it the
#      second half of a different script.  A rename leaves the running process
#      second half of a different script.  A rename leaves the running process
#      on the old inode -- BUT ONLY WHEN THE RENAMER AND THE READER SIT ON THE
#      SAME NFS CLIENT.  A sync driven from the login node never does: NFS
#      silly-rename protects only the RENAMING client's own opens, so a job
#      reading that file on another node has its inode unlinked underneath it,
#      bash's next incremental read returns an error, bash treats the error as
#      EOF and EXITS with the status of its last completed command -- rc 0, the
#      first rows only, and no error anywhere.  PROOF-SMOKE-1-1 measured exactly
#      that (fixture job 5995889, five arms) and it is the signature of 5993691.
#      So the rename is NECESSARY AND NOT SUFFICIENT, and the thing that makes a
#      sync safe is the PRE-INSTALL READ-SET REFUSAL below (rc 6), not the mv.
#
# AND IT NAMES AN IN-PLACE EDIT BEFORE A JOB DIES FOR IT (`--check`).  The drift
# check answers "does this match the commit I was told to be?"; `--check`
# answers the question a lane actually has at 03:00 -- "has anybody hand-edited
# a task copy since the last sync?" -- against `<task dir>/tools/.harness_synced_commit`,
# which this script writes and nothing else does.  Both phase templates run it
# as the FIRST line of their drift block, so the refusal names the edited FILE
# and the command that fixes it instead of saying "drift".
#
#   usage: harness_sync.sh <commit> [--add <task path>]... [--force] [--reason "<why>"]
#          harness_sync.sh --check [<commit>]          # no writes, ever
#
#   env overrides (argv wins):
#     HARNESS_TASK_DIR   default /oscar/data/stellex/glvov/agrescap/tasks/retread-4-11
#     HARNESS_REPO       default /oscar/data/stellex/glvov/agrescap/worktrees/harness-tools
#     HARNESS_SQUEUE     the squeue to ask about PENDING jobs (default `squeue`)
#
#   --running-list <file>  TEST-ONLY, the guard's shim for the rc-6 check. Rows
#          `<jid> <state> <name> <workdir> <sbatch path or -> [<job root>]`,
#          used INSTEAD of the live `squeue -t R,PD` + `scontrol show job`.
#          `<state>` is RUNNING or PENDING and is REQUIRED -- a row whose second
#          field is neither is a FATAL rc 2, not a row silently read as a name,
#          because that is how a stale four-field row from before HARNESS-SYNC-4
#          would have shifted every later field by one and quietly emptied the
#          read set. Nothing in production passes this flag.
#
#   rc 0  synced (or --check found nothing edited).  INCLUDING THE NO-OP: when
#         every mapped task copy already holds the commit's bytes and no `--add`
#         names a new file, the install set is EMPTY, and an empty install set
#         is its own verdict -- `### SYNC NO-OP commit=<sha> files=<n>
#         installed=0 unchanged=<n>`, rc 0, a pin mismatch TOLERATED on
#         `reason=nothing-installed`, and `### RECORD ADVANCED <old> -> <new>
#         bytes-identical` if the record was behind the disk.  Until
#         HARNESS-SYNC-9 this was an rc-4 REFUSAL: the read-set block is guarded
#         on a non-empty install set, so nothing was examined, and rc 4 read
#         "not examined" as "unsafe".  A sync that installs nothing cannot
#         strand anything.  (B30's landing, 2026-09-07T07:47: 21 rows, 3 jobs,
#         every one `read-set-not-examined`, nothing moved.)
#   rc 2  FATAL: bad arguments, no repo, no such commit, no mapping library
#   rc 3  --check: at least one task copy differs from the last synced commit
#   rc 4  REFUSED: a PENDING job of ours is pinned to a different commit AND
#         that job would actually be hurt by the move (HARNESS-SYNC-6).  The pin
#         alone is no longer enough: the same read set rc 6 computes is asked
#         about every pin-mismatched job, and it refuses only when
#           reason=reads-installed-file       its read set intersects the install set
#           reason=read-set-undeterminable    its script or a reference is unreadable
#           reason=read-set-not-examined      it never reached a determinate read
#                                             set -- OUT OF SCOPE. "We did not
#                                             look" is not "we looked and it was
#                                             clean". The OTHER way to an empty
#                                             read set, an EMPTY INSTALL SET, is
#                                             NOT this reason any more: it is
#                                             reason=nothing-installed on a
#                                             TOLERATED row (HARNESS-SYNC-9, and
#                                             see rc 0's NO-OP above).
#         A job whose read set is DETERMINATE and disjoint is TOLERATED on a
#         `### PIN MISMATCH tolerated jid=<j> reason=<why>` row -- never in
#         silence -- with reason=reads-only-job-root for the owner-snapshot shape
#         (HARNESS-SYNC-5: it execs a frozen copy under its own job root and
#         reads nothing of ours, so it neither runs a drift gate against this
#         task dir nor executes a byte this sync writes) and
#         reason=read-set-disjoint otherwise.  6015658/6015659 det162-cleanup
#         are the case that forced this: two snapshot owners, `afterany` on a
#         proof that could run to 13:05, holding the whole merge queue on a pin
#         neither of them ever read.
#   rc 5  a file failed to install or failed its md5 verification
#   rc 6  REFUSED: a job of ours -- RUNNING **or PENDING** -- would READ a file
#         this sync would install.  Every refusal row carries `state=`, and the
#         two states are refused for two DIFFERENT reasons that happen to have
#         the same answer (wait for the job):
#           state=RUNNING  the job is reading that file NOW, on another node.  A
#             cross-NFS-client rename is NOT atomic for it -- its inode is
#             unlinked under it, bash calls the read error EOF, and it exits 0
#             having run half its rows.  (HARNESS-SYNC-3.)
#           state=PENDING  the job has not started, so it will read the bytes
#             that are on disk WHENEVER Slurm starts it -- not the ones it was
#             submitted against.  A sync landing between the parent's end and
#             the dependent's start silently swaps the script it runs, and
#             NOTHING anywhere reports that.  rc 4 does not cover this: rc 4 is
#             a PIN DIR check, and a `sbatch --wrap 'bash .../cleanup_gated.sh'`
#             cleanup owner has no pin dir at all -- it was invisible to rc 4
#             for having no pin and to rc 6 for not being RUNNING.
#             (HARNESS-SYNC-4.)
#         Override: --force --reason "<why>".
#
#   BOTH REFUSAL SETS ARE ALWAYS COMPUTED AND ALWAYS PRINTED (HARNESS-SYNC-4-1).
#   Until this lane the rc-4 block `exit 4`ed where it stood, BEFORE the read-set
#   computation ran at all -- so when both applied the operator was handed the
#   rc-4 rows and told "let the queued job drain", drained it, re-ran, and was
#   THEN handed rc 6 and a second wait it had been given no way to see coming.
#   Two serial waits for one queue state, and the second one invisible: that is
#   MERGE-U's landing 2026-09-06T22:41 exactly (refused rc 4 on det141-cleanup
#   6001240 with det141-proof 6001140 still RUNNING and unread beside it).
#   So: the pin check and the read-set check BOTH run, BOTH print their rows,
#   and the exit is decided afterwards.
#     PRECEDENCE: rc 4 wins when both apply.  Not because it is worse -- it is
#     the CHEAPER one to clear (a pinned PENDING job can be repinned with
#     harness_commit_resolve.sh --write, where a read-set hit can only be
#     waited out) -- but because a caller that switches on the rc must get a
#     STABLE answer, and `fixset_land_row.sh` already explains 4 and 6
#     differently.  The rc-6 rows are on the page either way, and the rc-4 arm
#     of that explainer re-prints them, so nothing is stamped-but-unread.
#   Neither check writes a byte in either case: the install is still downstream
#   of both.
#
# THE MAPPING IS NOT DUPLICATED HERE.  A writer with its own copy of the table
# would eventually install a set its own checker does not read, which is the
# same class of defect one level up.  `HARNESS_DRIFT_LIB=1 . harness_drift_check.sh`
# defines `map_of`, `harness_is_evidence` and `HARNESS_SCAN_DIRS` and returns
# without running; that file is the ONE home of the table.
#
# WHAT IT DELIBERATELY DOES NOT DO ON ITS OWN: create task files that do not
# exist yet.  The commit carries files that map into a scanned directory but are
# absent from the task dir (`tools/land.sh` is one, and the task dir's copy of it
# lives at `merge-h/land.sh` by the basename rule -- installing the tools/ name
# would manufacture a second, wrong copy).  The drift check does not read absent
# files either, so inventing them all here would make the writer and the checker
# disagree about the set.  So they are REPORTED, and a new file enters the task
# dir only when a human NAMES it: `--add tools/<f>`, once, and from then on it is
# an ordinary member of the set.  That keeps this the only writer -- the
# alternative is a lane creating the file with a bare `cat-file`, which is the
# defect -- while leaving the choice of what exists explicit and auditable.
#
# Reader: harness_sync_guard.sh.
set -uo pipefail

SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="${HARNESS_TASK_DIR:-/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11}"
REPO="${HARNESS_REPO:-/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools}"
SQUEUE="${HARNESS_SQUEUE:-squeue}"
RECORD_REL="tools/.harness_synced_commit"
RECORD="$TASK_DIR/$RECORD_REL"
# READ BEFORE ANY MODE RUNS, because the no-op arm reports whether the record
# MOVED, and it cannot tell that after it has already been overwritten.
REC_OLD=
[ -f "$RECORD" ] && REC_OLD=$(sed -e 's/#.*//' -e 's/[[:space:]]//g' "$RECORD" | grep -m1 . || true)

MODE=sync; COMMIT=; FORCE=0; REASON=; ADDS=; RUNLIST=
while [ $# -gt 0 ]; do
  case "$1" in
    --check)  MODE=check;;
    --force)  FORCE=1;;
    --add)    shift; [ -n "${1:-}" ] || { echo "### SYNC FATAL: --add needs a task-relative path" >&2; exit 2; }
              ADDS="$ADDS $1";;
    --add=*)  ADDS="$ADDS ${1#--add=}";;
    --running-list) shift; RUNLIST="${1:-}";;
    --running-list=*) RUNLIST="${1#--running-list=}";;
    --reason) shift; REASON="${1:-}";;
    --reason=*) REASON="${1#--reason=}";;
    -*)       echo "### SYNC FATAL: unknown flag '$1'" >&2; exit 2;;
    *)        [ -z "$COMMIT" ] || { echo "### SYNC FATAL: two commits given ('$COMMIT' and '$1')" >&2; exit 2; }
              COMMIT="$1";;
  esac
  shift
done

[ -d "$TASK_DIR" ] || { echo "### SYNC FATAL: no such task dir $TASK_DIR" >&2; exit 2; }
[ -d "$REPO" ]     || { echo "### SYNC FATAL: no such repo $REPO" >&2; exit 2; }
DRIFT="$SELF_DIR/harness_drift_check.sh"
[ -f "$DRIFT" ] || { echo "### SYNC FATAL: no harness_drift_check.sh beside $0 -- it owns the mapping" >&2; exit 2; }
# shellcheck disable=SC1090
HARNESS_DRIFT_LIB=1 . "$DRIFT" || { echo "### SYNC FATAL: could not source the mapping from $DRIFT" >&2; exit 2; }
command -v map_of >/dev/null 2>&1 || {
  echo "### SYNC FATAL: $DRIFT is too old to source -- it defines no map_of (HARNESS-SYNC-1)" >&2; exit 2; }

# `--check` with no argument means "against whatever the last sync installed".
if [ -z "$COMMIT" ] && [ "$MODE" = check ]; then
  if [ -f "$RECORD" ]; then
    COMMIT=$(sed -e 's/#.*//' -e 's/[[:space:]]//g' "$RECORD" | grep -m1 . || true)
  fi
  [ -n "$COMMIT" ] || {
    echo "### SYNC CHECK FATAL: no commit given and no record at $RECORD" >&2
    echo "###   Nothing has been synced by harness_sync.sh yet, so there is no 'last synced" >&2
    echo "###   commit' to compare against. Run: harness_sync.sh <commit>" >&2
    exit 2; }
fi
[ -n "$COMMIT" ] || { echo "### SYNC FATAL: usage: harness_sync.sh <commit> [--force] [--reason \"<why>\"]" >&2; exit 2; }

SHA="$(git -C "$REPO" rev-parse --verify "${COMMIT}^{commit}" 2>/dev/null)" || {
  echo "### SYNC FATAL: $COMMIT is not a commit in $REPO" >&2; exit 2; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/harness_sync.XXXXXX")" || {
  echo "### SYNC FATAL: could not make a temp dir" >&2; exit 2; }
trap 'rm -rf "$TMP"' EXIT

# ---- the mapped set, from the checker's own table --------------------------
# One `<task-relative path>|<repo path>` row per file the drift check will read.
SET="$TMP/set.txt"; : > "$SET"
ABSENT="$TMP/absent.txt"; : > "$ABSENT"
for d in $HARNESS_SCAN_DIRS; do
  [ -d "$TASK_DIR/$d" ] || continue
  while IFS= read -r f; do
    b=$(basename -- "$f")
    harness_is_evidence "$b" && continue
    trel="$d/$b"
    w=$(map_of "$trel")
    printf '%s|%s\n' "$trel" "$w" >> "$SET"
  done < <(find "$TASK_DIR/$d" -maxdepth 1 -type f | sort)
done
# The other direction, reported only: files the commit carries that map into a
# scanned directory and are not in the task dir.
while IFS= read -r w; do
  case "$w" in
    harness/tools/*)          trel="tools/${w#harness/tools/}";;
    harness/phase_template/*) trel="tools/phase_template/${w#harness/phase_template/}";;
    *) continue;;
  esac
  harness_is_evidence "$(basename -- "$trel")" && continue
  case "$trel" in */*/*/*) continue;; esac      # nested, not a maxdepth-1 file
  [ -e "$TASK_DIR/$trel" ] || printf '%s|%s\n' "$trel" "$w" >> "$ABSENT"
done < <(git -C "$REPO" ls-tree -r --name-only "$SHA" harness/tools harness/phase_template 2>/dev/null | sort)

# ---- --add: the one way a NEW file enters the task dir ----------------------
# Named explicitly, one path at a time, and only into a scanned directory with a
# blob behind it. After the first --add the file is an ordinary member of the
# set and every later sync carries it without being told.
if [ "$MODE" != check ]; then
  for a in $ADDS; do
    case "$a" in
      tools/*|merge-h/*) ;;
      *) echo "### SYNC FATAL: --add $a is not under a scanned directory ($HARNESS_SCAN_DIRS)" >&2; exit 2;;
    esac
    aw=$(map_of "$a")
    if [ -z "$aw" ] || ! git -C "$REPO" cat-file -e "$SHA:$aw" 2>/dev/null; then
      echo "### SYNC FATAL: --add $a maps to '${aw:-<nothing>}', which $SHA does not carry" >&2; exit 2
    fi
    if grep -qF -- "$a|" "$SET"; then
      echo "### SYNC --add $a is already in the task dir -- it is synced either way"
      continue
    fi
    mkdir -p "$(dirname -- "$TASK_DIR/$a")" || { echo "### SYNC FATAL: could not make the directory for $a" >&2; exit 2; }
    printf '%s|%s\n' "$a" "$aw" >> "$SET"
    echo "### SYNC --add $a will be CREATED from $SHA:$aw"
    grep -vF -- "$a|" "$ABSENT" > "$TMP/absent2" 2>/dev/null || : > "$TMP/absent2"
    mv -f "$TMP/absent2" "$ABSENT"
  done
fi

TOTAL=$(wc -l < "$SET")

# ---- --check: read-only, and it names the FILE ------------------------------
if [ "$MODE" = check ]; then
  echo "### harness sync CHECK: task=$TASK_DIR against last-synced commit=$SHA"
  edited=0; okc=0
  while IFS='|' read -r trel w; do
    if ! git -C "$REPO" cat-file blob "$SHA:$w" > "$TMP/blob" 2>/dev/null; then
      echo "SYNC CHECK no-blob  $trel  (nothing at $SHA:$w)"; edited=$((edited + 1)); continue
    fi
    tmd5=$(md5sum "$TASK_DIR/$trel" | awk '{print $1}')
    bmd5=$(md5sum "$TMP/blob"       | awk '{print $1}')
    if [ "$tmd5" = "$bmd5" ]; then okc=$((okc + 1)); else
      echo "SYNC CHECK EDITED   $trel  task=$tmd5 $SHA:$w=$bmd5"; edited=$((edited + 1))
    fi
  done < "$SET"
  echo "### SYNC CHECK SUMMARY commit=$SHA checked=$TOTAL clean=$okc edited=$edited"
  # LANE-CLOSE CHECKLIST, second row (HARNESS-CONSOL-10). "Every task copy is
  # still the commit's bytes" says nothing about whether that COMMIT left this
  # disk. Seventeen commits from four lanes lived only under agrescap/worktrees
  # on 2026-09-07 because no reader asked. It never changes this mode's rc --
  # the cert treats a non-0/3 rc as "inconclusive" -- so the row is the actor.
  # HARNESS-SYNC-8 FINDING 2: THE SYNC INSTALLED A CALLER WITHOUT ITS CALLEE.
  # This resolved ONE hop, `$(dirname "$0")`, and a NEW file enters the task dir
  # only when it is `--add`ed by name -- so the moment the newly-installed task
  # copy of this script ran, it called a path that has never existed there and
  # printed `### PUSH LAG branch=? unpushed=?`. A criterion that can only answer
  # `?` has no live producer (law 2), and "unpushed=0" is a stated acceptance
  # criterion for a landing. `script_refs.sh` two blocks down has survived the
  # same absence for the same reason: it resolves THREE hops, ending at the repo.
  # This now does the same, and `$SELF_DIR` (from BASH_SOURCE) rather than `$0`,
  # because `$0` is not knowable when the file is Slurm's snapshot
  # (HARNESS-SYNC-7-1). An `--add` of tools/harness_push_check.sh is still the
  # right end state -- the fallback is what keeps the criterion measurable until
  # then, not a reason to skip it.
  PUSH_CHECK=$SELF_DIR/harness_push_check.sh
  [ -f "$PUSH_CHECK" ] || PUSH_CHECK=$TASK_DIR/tools/harness_push_check.sh
  [ -f "$PUSH_CHECK" ] || PUSH_CHECK=$REPO/harness/tools/harness_push_check.sh
  if [ -f "$PUSH_CHECK" ]; then
    echo "### PUSH LAG reader: $PUSH_CHECK"
    HARNESS_REPO=$REPO bash "$PUSH_CHECK" || true
  else
    echo "### PUSH LAG branch=? unpushed=? -- no harness_push_check.sh at $SELF_DIR,"
    echo "###   at $TASK_DIR/tools, nor at $REPO/harness/tools. This criterion has NO"
    echo "###   producer and must not be reported as satisfied."
  fi
  # THE OTHER DIRECTION, IN THE READ-ONLY MODE TOO. The absent set was reported
  # only by the WRITING mode, so a lane that ran `--check` -- the mode you run
  # precisely when you are not installing -- could not see that the commit
  # carries files this task dir does not have. That is how harness_push_check.sh
  # was missing for a full lane without anybody being told.
  if [ -s "$ABSENT" ]; then
    echo "### SYNC ABSENT ($(wc -l < "$ABSENT")): in $SHA, not in the task dir, NOT installed --"
    echo "###   the drift check does not read them either, so the writer and the checker agree."
    echo "###   To bring one in:  harness_sync.sh $SHA --add <task path>"
    sed 's/^/###   /' "$ABSENT"
  else
    echo "### SYNC ABSENT (0): the task dir carries every file $SHA maps into a scanned directory"
  fi
  if [ "$edited" -gt 0 ]; then
    echo "### SYNC CHECK REFUSED -- the file(s) named above were written by something"
    echo "###   other than harness_sync.sh since the last sync. Do not re-edit them in"
    echo "###   place: commit the change in $REPO and then"
    echo "###     bash $TASK_DIR/tools/harness_sync.sh <commit>"
    exit 3
  fi
  echo "### SYNC CHECK CLEAN -- every task copy is still $SHA's bytes"
  exit 0
fi

# ---- the queue, BEFORE a single byte is written -----------------------------
# A PENDING job has snapshotted nothing. Its pin file already names a commit and
# moving the task dir off that commit kills it at its own drift gate the moment
# it starts -- which is precisely ticks 499-501. The job roots are found by the
# `<lane dir>/HARNESS_COMMIT` convention harness_commit_resolve.sh --write uses:
# a pin dir matches a job when the names are equal, when the job name extends the
# dir name, or when the dir name starts with the job name's leading token
# (`sr3-gate` -> `sr3-work`).
# The two refusal verdicts, decided here and ACTED ON at the bottom of the
# checks (HARNESS-SYNC-4-1).  0 means "this check did not refuse".
SYNC_RC4=0
SYNC_RC6=0

PINDIRS="$TMP/pindirs.txt"
find "$TASK_DIR" -maxdepth 2 -name HARNESS_COMMIT -type f -printf '%h\n' 2>/dev/null | sort > "$PINDIRS"

REFUSALS="$TMP/refusals.txt"; : > "$REFUSALS"
PDROWS="$TMP/pd.txt"; : > "$PDROWS"
"$SQUEUE" -u glvov -h -t PD -o '%i %j' > "$PDROWS" 2>/dev/null || : > "$PDROWS"
while read -r jid jname; do
  [ -n "${jname:-}" ] || continue
  stem=${jname%%-*}
  while read -r pd; do
    b=$(basename -- "$pd")
    match=0
    [ "$b" = "$jname" ] && match=1
    [ "${jname#"$b"-}" != "$jname" ] && match=1
    [ "${b#"$stem"}" != "$b" ] && match=1
    [ "$match" = 1 ] || continue
    pin=$(sed -e 's/#.*//' -e 's/[[:space:]]//g' "$pd/HARNESS_COMMIT" | grep -m1 . || true)
    [ -n "$pin" ] || continue
    pinsha=$(git -C "$REPO" rev-parse --verify "${pin}^{commit}" 2>/dev/null || echo "$pin")
    [ "$pinsha" = "$SHA" ] && continue
    printf '%s %s %s pinned=%s\n' "$jid" "$jname" "$pd/HARNESS_COMMIT" "$pinsha" >> "$REFUSALS"
  done < "$PINDIRS"
done < "$PDROWS"

# THE VERDICT ON THESE ROWS IS DEFERRED TO `rc4_verdict` BELOW (HARNESS-SYNC-6).
# It cannot be decided here any more, because it now depends on the READ SET,
# which is computed further down.  Nothing is printed yet either: the rows keep
# their place at the TOP of a refusal page, ahead of the rc-6 family, because
# that is the order every consumer of this output was written against.
rc4_verdict () {   # called AFTER the read-set computation; prints the rc-4 family
if [ -s "$REF_REFUSE" ]; then
  echo "### SYNC would strand these PENDING jobs -- they are pinned to another commit:"
  # `state=PENDING` on every row, because these rows now share a page with the
  # rc-6 rows and only the state field says which check produced which.  Every
  # row here is PENDING by construction: $PDROWS is `squeue -t PD`.
  sed 's/^/###   state=PENDING /' "$REF_REFUSE"
  if [ "$FORCE" != 1 ]; then
    echo "### SYNC REFUSED (rc 4). A queued job has snapshotted NOTHING: syncing the task"
    echo "###   dir to $SHA kills it at its own drift gate the moment it starts."
    echo "###   Let them run, repin them (harness_commit_resolve.sh --write <job root> $SHA),"
    echo "###   or re-run with --force --reason \"<why this is the right call>\"."
    SYNC_RC4=4   # HARNESS-SYNC-4-1 DEFERRED EXIT
  elif [ -z "$REASON" ]; then
    echo "### SYNC REFUSED (rc 4) -- --force WITHOUT --reason is still a refusal. The rows"
    echo "###   above are what you are overriding; say why, and it goes in the log row:"
    echo "###     harness_sync.sh $COMMIT --force --reason \"<why>\""
    SYNC_RC4=4   # HARNESS-SYNC-4-1 DEFERRED EXIT
  else
    # The COUNT IS OF JOBS, not of rows.  One queued job matching two pin dirs
    # writes two rows, and `wc -l` would call that two jobs; the operator copies
    # this line into a lane log row, so it has to be the number they can check
    # against `squeue`.  (HARNESS-SYNC-4 replaced an `awk '{print $3}'` here that
    # printed the literal REFUSED for every input; a count that is merely
    # PLAUSIBLE is how that survived, so it is asserted by the guard now.)
    echo "### SYNC FORCED over $(awk '{print $1}' "$REF_REFUSE" | sort -u | wc -l) pinned PENDING job(s) reason=$REASON"
    echo "###   ^ copy this line into the lane log row for this sync."
  fi
  echo "### SYNC rc-4 rows=$(wc -l < "$REF_REFUSE") jobs=$(awk '{print $1}' "$REF_REFUSE" | sort -u | wc -l) refusing=$( [ "$SYNC_RC4" = 0 ] && echo no || echo yes)"
  echo "###   -- the read-set check ALSO RAN (HARNESS-SYNC-4-1): if it also"
  echo "###   refuses, its rows are printed here too and you get BOTH waits at once."
fi
# THE TOLERATED ROWS ARE PRINTED WHETHER OR NOT ANYTHING REFUSED (HARNESS-SYNC-6).
# A pin mismatch that this sync decided NOT to refuse for is exactly the kind of
# thing that must not be silent: it is a judgement the machinery made on the
# operator's behalf, and the PIN REPORT at the bottom only names pins AFTER a
# successful sync.  One row per (job, pin dir), with the reason that cleared it.
if [ -s "$REF_TOL" ]; then
  cat "$REF_TOL"
  echo "###   ^ PIN MISMATCH tolerated: these PENDING jobs are pinned to another commit, but"
  echo "###   their READ SET does not intersect this sync's install set, so moving the task"
  echo "###   dir cannot change a byte they will execute.  rc 4 used to refuse for them on"
  echo "###   the pin alone (HARNESS-SYNC-6)."
fi
}

# ---- THE READ SET OF EVERY RUNNING JOB, BEFORE A SINGLE BYTE (HARNESS-SYNC-3) -
# rc 6, and it is the guard the rename is NOT.  `mv -f` over a name is atomic
# for readers ON THE RENAMING CLIENT.  It is not atomic for a reader on another
# node: NFS silly-rename hides the old inode behind `.nfsXXXX` only for opens
# THIS client holds, so the job reading the script on node2xxx has its file
# unlinked, bash's next incremental read errors, bash treats the error as EOF,
# and the job EXITS 0 having run only the rows it had already parsed.  That is
# 5993691 (three tail rows lost, `### LANE_EXIT=0` over `fail=4`) and it is what
# PROOF-SMOKE-1-1 reproduced deliberately in job 5995889.  The top-level sbatch
# is immune -- Slurm snapshots it at submit -- but everything it reaches by
# path (`bash "$T/tools/<f>"`, `source .../tools/<f>`) is not.
#
# So: the INSTALL SET (the files whose bytes would actually change) is intersected
# with the READ SET of every job of ours, and a non-empty intersection is
# a REFUSAL, before the first byte.  A job whose read set cannot be
# DETERMINED is treated as reading EVERYTHING -- an unreadable job is not a safe
# job (law 9).
#
# AND "EVERY JOB" MEANS PENDING TOO (HARNESS-SYNC-4).  The first cut asked
# `squeue -t R` only, and that left a hole with a live example sitting in it:
# det141-cleanup 6001240, an `afterany:6001140` owner submitted as
# `sbatch --wrap 'bash <task dir>/merge-h/cleanup_gated.sh <roots>'`.  It has NO
# PIN DIR, so the rc-4 pin check could not see it; it is not RUNNING, so this
# check could not see it either; and a sync landing in the window between its
# parent finishing and Slurm starting it would have silently changed the bytes
# of the cleanup it then runs -- with no drift refusal, no rc 4, no rc 6 and
# nothing in any log.  A PENDING job's script is READABLE the same way a running
# one's is (`scontrol write batch_script <jid> -` works in state PD, measured on
# 6001240), so it is parsed by the same `refs_of` and refused by the same rc.
# The row prints `state=PENDING` so the operator knows the job it must wait for
# has not started yet, rather than looking for it on a node.
INSTSET="$TMP/instset.txt"; : > "$INSTSET"
# THE FILES THAT HAVE NO BLOB AT ALL ARE COUNTED, NOT SKIPPED (HARNESS-SYNC-9).
# A mapped task file with no mapping (`$w` empty) or no blob at `$SHA` used to
# `continue` out of this loop and vanish, which was harmless while this loop's
# only product was the install set -- but the install loop at the bottom calls
# exactly that file `SYNC NO-BLOB` and exits 5. So an empty install set alone
# does NOT mean "this sync would do nothing and succeed"; it means that only
# when nothing is unmapped or blobless too. The no-op short-circuit below reads
# BOTH files, because a no-op that swallowed a would-be rc 5 would be the same
# defect it exists to fix, pointed the other way.
NOBLOB="$TMP/noblob.txt"; : > "$NOBLOB"
while IFS='|' read -r trel w; do
  if [ -z "$w" ] || ! git -C "$REPO" cat-file blob "$SHA:$w" > "$TMP/preblob" 2>/dev/null; then
    printf '%s\n' "$trel" >> "$NOBLOB"; continue
  fi
  if [ -f "$TASK_DIR/$trel" ] &&
     [ "$(md5sum "$TASK_DIR/$trel" | awk '{print $1}')" = "$(md5sum "$TMP/preblob" | awk '{print $1}')" ]; then
    continue
  fi
  printf '%s\n' "$trel" >> "$INSTSET"
done < "$SET"

# ---- A SYNC THAT INSTALLS NOTHING CANNOT STRAND ANYTHING (HARNESS-SYNC-9) ----
# THE DEFECT, AND IT HELD B30 FOR A MORNING. When the task dir is ALREADY at the
# requested commit the install set is EMPTY. The read-set block below is guarded
# on `[ -s "$INSTSET" ]`, so it never runs, so no job ever reaches a determinate
# read set, so `$RSDET` is empty -- and the rc-4 refinement then reads every
# pin-mismatched PENDING job as `read-set-not-examined` and REFUSES. That is
# HARNESS-SYNC-6's own comment: "out of the read-set scope, OR THE INSTALL SET
# WAS EMPTY SO NO JOB WAS EXAMINED AT ALL". Law 9's "we did not look is not we
# looked and it was clean" is the right reading when there is something to look
# FOR. Here there is not: not one byte will be renamed over, so no reader can
# lose an inode and no queued job's drift gate can move under it. Measured:
# B30's landing 2026-09-07T07:47 refused rc 4 with 21 rows over three jobs, ALL
# of them `read-set-not-examined`, `rc4_tolerated=0`, and NOTHING MOVED -- a
# sync refusing to do nothing, and a merge queue held by it.
#
# SO THE EMPTY INSTALL SET IS ITS OWN VERDICT, and it is a TOLERANCE, not a
# skip: the pin-mismatched jobs are still named on `### PIN MISMATCH tolerated`
# rows with `reason=nothing-installed`, because a judgement the machinery makes
# on the operator's behalf must never be silent (HARNESS-SYNC-6). Every refusal
# for a NON-EMPTY install set -- rc 4 and rc 6 alike -- is untouched: `$NOOP` is
# 0 in every one of those runs.
#
# WHY IT DOES NOT `exit 0` HERE. The tail of this script is the sync's evidence
# packet -- the ABSENT report, the PIN REPORT, the drift line from the new state
# -- and every one of them is still true and still owed on a no-op run. (The
# guard's own arm A2 is the live proof: its second sync is a no-op by
# construction and asserts the ABSENT report.) So the no-op sets a flag, the
# refusal verdict reads it, the install loop runs and writes nothing because
# every file is `SYNC same`, and the one shared tail prints.
NOOP=0
if [ ! -s "$INSTSET" ] && [ ! -s "$NOBLOB" ]; then   # HARNESS-SYNC-9 NO-OP SHORT-CIRCUIT
  NOOP=1
  echo "### SYNC NO-OP commit=$SHA files=$TOTAL installed=0 unchanged=$TOTAL"
  echo "###   Every mapped task copy is ALREADY $SHA's bytes and no --add names a file the"
  echo "###   task dir lacks, so the install set is EMPTY. Nothing is renamed over, so no"
  echo "###   RUNNING job can lose an inode (rc 6) and no PENDING job's pin can be stranded"
  echo "###   (rc 4): a pin mismatch is TOLERATED below on reason=nothing-installed rather"
  echo "###   than refused on read-set-not-examined. A NON-EMPTY install set refuses exactly"
  echo "###   as before."
fi

# THE READ-SET PARSER LIVES IN tools/script_refs.sh (HARNESS-SYNC-5), because
# phase_template/owner_snapshot.sh must copy exactly the files a job will read
# and two copies of that parser is how a snapshot ends up missing one. It emits
# `<basename>\t<abs path or ->`, or `?\t-` for a reference it cannot resolve.
SR=$(dirname -- "$0")/script_refs.sh
[ -f "$SR" ] || SR=$TASK_DIR/tools/script_refs.sh
[ -f "$SR" ] || SR=$REPO/harness/tools/script_refs.sh
[ -f "$SR" ] || { echo "### SYNC FATAL: no script_refs.sh beside $0, at $TASK_DIR/tools, nor at $REPO/harness/tools -- the read-set check has no parser and a silent install is worse than a refusal" >&2; exit 2; }
# shellcheck disable=SC1090
. "$SR"

RSHITS="$TMP/readset.txt"; : > "$RSHITS"
RSDET="$TMP/readset_determinate.txt"; : > "$RSDET"   # HARNESS-SYNC-6
if [ -s "$INSTSET" ]; then
  RUNROWS="$TMP/run.txt"; : > "$RUNROWS"
  if [ -n "$RUNLIST" ]; then
    [ -f "$RUNLIST" ] || { echo "### SYNC FATAL: --running-list $RUNLIST does not exist" >&2; exit 2; }
    grep -v '^[[:space:]]*$' "$RUNLIST" > "$RUNROWS" || :
    # THE STATE COLUMN IS REQUIRED AND VALIDATED (HARNESS-SYNC-4).  Before this
    # lane a row was `<jid> <name> <workdir> <script>`; state was inserted at
    # field 2, so a stale row would read `<name>` as the state and shift every
    # later field by one -- an empty read set and a SILENT install, which is the
    # exact class of defect this whole block exists to stop.  Refuse loudly.
    while read -r _j st _rest; do
      case "${st:-}" in
        RUNNING|PENDING) ;;
        *) echo "### SYNC FATAL: --running-list $RUNLIST row for job ${_j:-?} has state='${st:-}';" >&2
           echo "###   rows are '<jid> <state> <name> <workdir> <sbatch or -> [<job root>]'" >&2
           echo "###   and <state> must be RUNNING or PENDING (HARNESS-SYNC-4)." >&2
           exit 2;;
      esac
    done < "$RUNROWS"
  else
    "$SQUEUE" -u glvov -h -t R,PD -o '%i %T %j %Z' > "$TMP/rq.txt" 2>/dev/null || : > "$TMP/rq.txt"
    while read -r jid jstate jname wd; do
      [ -n "${jid:-}" ] || continue
      cmd=$(scontrol show job "$jid" 2>/dev/null | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
      jroot="${wd:--}"
      [ -f "${cmd:-}" ] && jroot=$(dirname -- "$cmd")
      # THE SUBMITTED SCRIPT FROM SLURM'S OWN SNAPSHOT, not from the filesystem.
      # `sbatch --wrap` and a heredoc submission both leave `Command=(null)` and
      # no file on disk (5992050 is one), and reading that as "no script" would
      # refuse every sync for the life of the job. `scontrol write batch_script`
      # hands back the exact bytes Slurm is running.
      if scontrol write batch_script "$jid" "$TMP/sb.$jid" >/dev/null 2>&1 && [ -s "$TMP/sb.$jid" ]; then
        script="$TMP/sb.$jid"
      elif [ -f "${cmd:-}" ]; then
        script="$cmd"
      else
        script=-
      fi
      printf '%s %s %s %s %s %s\n' "$jid" "${jstate:-RUNNING}" "${jname:--}" "${wd:--}" "$script" "$jroot" >> "$RUNROWS"
    done < "$TMP/rq.txt"
  fi
  while read -r jid jstate jname wd script jroot; do
    [ -n "${jid:-}" ] || continue
    if [ -z "${jroot:-}" ]; then
      if [ "${script:--}" != "-" ] && [ -f "$script" ]; then jroot=$(dirname -- "$script"); else jroot=${wd:--}; fi
    fi
    # IN SCOPE: the job works in this task dir, its job root is in it, or it owns
    # a pin dir here by the same name rule the PIN REPORT uses.
    scope=0
    case "${wd:-}"    in "$TASK_DIR"|"$TASK_DIR"/*) scope=1;; esac
    case "${jroot:-}" in "$TASK_DIR"|"$TASK_DIR"/*) scope=1;; esac
    if [ "$scope" = 0 ] && [ -n "${jname:-}" ]; then
      stem=${jname%%-*}
      while read -r pd; do
        b=$(basename -- "$pd")
        [ "$b" = "$jname" ] && scope=1
        [ "${jname#"$b"-}" != "$jname" ] && scope=1
        [ "${b#"$stem"}" != "$b" ] && scope=1
      done < "$PINDIRS"
    fi
    [ "$scope" = 1 ] || continue
    # THE READ SET: the job's own sbatch, plus one level down -- the driver
    # scripts it names that sit in the job root.  Rows are `<basename>\t<path>`
    # (HARNESS-SYNC-5); the recursion still walks by BASENAME because a job-root
    # driver is found by name, and the PATH column is what the hit test uses.
    RS="$TMP/rs.$jid.txt"; : > "$RS"
    if [ -z "${script:-}" ] || [ "$script" = "-" ] || [ ! -f "$script" ]; then
      printf '?\t-\n' > "$RS"
    else
      refs_of_sibling_resolved "$script" > "$RS"
      while IFS= read -r rb; do
        [ "$rb" = '?' ] && continue
        # A reference that resolved to a LITERAL path is followed THERE; one
        # that did not is looked for in the job root, as before.
        rp=$(awk -F'\t' -v b="$rb" '$1==b && $2!="-"{print $2; exit}' "$RS")
        if [ -n "$rp" ] && [ -f "$rp" ]; then
          refs_of_sibling_resolved "$rp" >> "$RS"
        elif [ -f "$jroot/$rb" ]; then
          refs_of_sibling_resolved "$jroot/$rb" >> "$RS"
        fi
      done < <(cut -f1 "$RS" | sort -u)
    fi
    if cut -f1 "$RS" | grep -qx '?'; then
      why=unresolved-reference
      [ -f "${script:-/nonexistent}" ] || why=no-sbatch-found
      while IFS= read -r trel; do
        printf '### SYNC REFUSED rc=6 running=%s state=%s file=%s reason=%s\n' \
          "$jid" "${jstate:-RUNNING}" "$(basename -- "$trel")" "$why" >> "$RSHITS"
      done < "$INSTSET"
      continue
    fi
    # HARNESS-SYNC-6.  THIS job's read set is DETERMINATE -- every reference in
    # it resolved.  Record that, because the rc-4 verdict below needs to tell
    # "we looked and it reads nothing of ours" apart from "we never looked",
    # and law 9 forbids reading the second as the first.  The second column says
    # whether every resolved reference lives under the job's OWN root, which is
    # the owner-snapshot shape (HARNESS-SYNC-5) and the reason worth printing.
    ojr=$(awk -F'\t' -v jr="$jroot/" '$2!="-" && index($2,jr)!=1 {print "no"; exit}' "$RS")
    printf '%s %s\n' "$jid" "${ojr:-yes}" >> "$RSDET"
    # THE HIT TEST, HARNESS-SYNC-5, AND THE ADDITION IS THE `elsewhere` BRANCH.
    # Matching by basename alone made a job that runs a JOB-LOCAL SNAPSHOT of
    # cleanup_gated.sh collide with the task copy of that name, and pin the
    # whole harness for as long as it ran -- hours, for a reap of millions of
    # entries (det1f-cleanup 5999937, det141-cleanup 6001240). A reference is a
    # hit when the basename matches AND at least one of its rows either has an
    # UNRESOLVED path (law 9: what cannot be read is assumed dangerous) or
    # resolves to exactly the file this sync would rewrite. If every row for
    # that basename resolves somewhere else, the job is reading a different
    # file with the same name and the sync is safe for it -- which is what the
    # owner snapshot exists to make true.
    while IFS= read -r trel; do
      tb=$(basename -- "$trel")
      cut -f1 "$RS" | grep -qxF -- "$tb" || continue
      # HARNESS-SYNC-7: the verdict now carries the PATH it resolved to and the
      # VARIABLE CHAIN that produced it, because a cleared refusal that cannot
      # be audited is a refusal cleared on trust.
      vinfo=$(awk -F'\t' -v b="$tb" -v want="$TASK_DIR/$trel" '
        $1==b { seen=1
                if ($2=="-") unresolved=1
                else if ($2==want) exact=1
                else { other=1; if (op=="") { op=$2; ov=($3==""?"-":$3) } } }
        END { if (!seen) v="none"; else if (unresolved) v="unresolved";
              else if (exact) v="exact"; else v="elsewhere";
              printf "%s\t%s\t%s", v, (op==""?"-":op), (ov==""?"-":ov) }' "$RS")
      verdict=${vinfo%%$'\t'*}
      vrest=${vinfo#*$'\t'}; rpath=${vrest%%$'\t'*}; rvia=${vrest#*$'\t'}
      case "$verdict" in
        elsewhere)
          # WHICH KIND of elsewhere, named rather than lumped: a path computed
          # from a literal variable chain says `resolved-literal` and prints the
          # chain; a path outside this task tree cannot be a read of anything
          # this sync installs, and says so.
          rmatch=resolved-elsewhere
          case "$rvia" in via=-|-|'') ;; *) rmatch=resolved-literal;; esac
          rscope=reads-outside-install-set
          case "$rpath" in "$TASK_DIR"/*) rscope=reads-another-task-path;; esac
          printf '###   read-set OK job=%s file=%s match=%s %s path=%s reason=%s -- every reference to that basename resolves elsewhere (owner snapshot); not a read of %s\n' \
            "$jid" "$tb" "$rmatch" "$rvia" "$rpath" "$rscope" "$TASK_DIR/$trel"
          continue;;
        none) continue;;
      esac
      printf '### SYNC REFUSED rc=6 running=%s state=%s file=%s reason=read-by-%s match=%s\n' \
        "$jid" "${jstate:-RUNNING}" "$tb" "$jname" "$verdict" >> "$RSHITS"
    done < "$INSTSET"
  done < "$RUNROWS"
fi

# ---- THE rc-4 REFINEMENT (HARNESS-SYNC-6) -----------------------------------
# WHAT WAS WRONG WITH THE PIN TEST ALONE.  rc 4 asks ONE question -- "does a
# PENDING job own a pin dir naming a different commit?" -- and refuses on the
# answer.  That was right when every queued job read the task harness live.  It
# is not right any more, because HARNESS-SYNC-5's owner snapshot exists exactly
# to make a queued job STOP reading the task tree: it copies the gate and what
# the gate sources into the job's own root and submits an sbatch that `exec`s
# the frozen copy by literal absolute path.  Such a job cannot be hurt by this
# sync -- not its drift gate, because it does not run one, and not its bytes,
# because none of them are ours.  Refusing for it is a refusal with no injury
# behind it, and this campaign has paid for that twice: 6015658/6015659
# det162-cleanup are snapshot owners, `afterany` on a proof that can run to
# 13:05, and they held the whole merge queue on a pin they never read.
#
# SO THE TEST IS NOW THE PIN **AND** THE READ SET, and the read set is the SAME
# one rc 6 computes -- the same `refs_of_sibling_resolved` parser over
# `scontrol write batch_script` plus one level of drivers in the job root.  Two
# outcomes refuse and one tolerates:
#   * the job appears in $RSHITS -- its read set intersects the install set, OR
#     it was undeterminable (law 9: an unreadable job is not a safe job).  rc 4.
#   * the job never reached a determinate read set at all -- out of the read-set
#     scope, or the install set was empty so no job was examined.  ALSO rc 4:
#     "we did not look" is not "we looked and it was clean", and reading it as
#     the latter is precisely how 6001240 became invisible (HARNESS-SYNC-4).
#   * the job HAS a determinate read set and is not in $RSHITS -- tolerated, on
#     a named row, never in silence.
# The tolerated row's reason distinguishes the shape that motivated this from
# the general case, because they are cleared for different reasons and a reader
# sizing the next owner needs to know which one they have.
REF_REFUSE="$TMP/refusals_refusing.txt"; : > "$REF_REFUSE"
REF_TOL="$TMP/refusals_tolerated.txt";   : > "$REF_TOL"
while read -r r_jid r_jname r_pinfile r_pinned; do
  [ -n "${r_jid:-}" ] || continue
  if grep -q " running=$r_jid " "$RSHITS" 2>/dev/null; then
    # WHY it is in $RSHITS decides the reason, because the two are cleared
    # differently: an intersecting reader has to finish, an UNDETERMINABLE one
    # may only need a readable script (or an owner snapshot).  A row ending
    # `reason=unresolved-reference` or `reason=no-sbatch-found` is the second.
    if grep -F " running=$r_jid " "$RSHITS" | grep -qvE 'reason=(unresolved-reference|no-sbatch-found)$'; then
      r_why=reads-installed-file
    else
      r_why=read-set-undeterminable
    fi
    printf '%s %s %s %s reason=%s\n' \
      "$r_jid" "$r_jname" "$r_pinfile" "$r_pinned" "$r_why" >> "$REF_REFUSE"
    continue
  fi
  r_det=$(awk -v j="$r_jid" '$1==j{print $2; exit}' "$RSDET")
  if [ -z "$r_det" ]; then   # HARNESS-SYNC-6 TOLERANCE BRANCH
    # NOT EXAMINED is not CLEAN -- the job was out of the read-set scope. Law 9.
    #
    # EXCEPT WHEN THERE WAS NOTHING TO EXAMINE (HARNESS-SYNC-9). The other way
    # to reach an empty `$RSDET` is an EMPTY INSTALL SET: the read-set block is
    # guarded on `[ -s "$INSTSET" ]` and never ran, so no job was examined --
    # not because we declined to look, but because there is nothing this sync
    # would write for a job to read. "We did not look" and "there was nothing
    # to look for" have opposite verdicts and this branch used to give them the
    # same one, which is how B30's landing refused rc 4 having moved nothing.
    # Tolerated, on a NAMED row like every other cleared pin mismatch.
    if [ "$NOOP" = 1 ]; then
      printf '### PIN MISMATCH tolerated jid=%s reason=nothing-installed name=%s pin=%s %s\n' \
        "$r_jid" "$r_jname" "$r_pinfile" "$r_pinned" >> "$REF_TOL"
      continue
    fi
    printf '%s %s %s %s reason=read-set-not-examined\n' \
      "$r_jid" "$r_jname" "$r_pinfile" "$r_pinned" >> "$REF_REFUSE"
    continue
  fi
  r_why=read-set-disjoint
  [ "$r_det" = yes ] && r_why=reads-only-job-root
  # `$r_pinned` is field 4 of a $REFUSALS row and ALREADY reads `pinned=<sha>`
  # (see the printf that writes $REFUSALS).  Prefixing it again printed
  # `pinned=pinned=8108ca4...` on the first production run of this refinement,
  # 2026-09-07 -- harmless to a human, poison to any reader that parses the
  # field, which is the whole point of putting the sha on the row.  Interpolated
  # bare, and the guard now asserts the field EXACTLY.
  printf '### PIN MISMATCH tolerated jid=%s reason=%s name=%s pin=%s %s\n' \
    "$r_jid" "$r_why" "$r_jname" "$r_pinfile" "$r_pinned" >> "$REF_TOL"
done < "$REFUSALS"
rc4_verdict

if [ -s "$RSHITS" ]; then
  sort -u "$RSHITS"
  if [ "$FORCE" != 1 ] || [ -z "$REASON" ]; then
    echo "### SYNC REFUSED (rc 6). The file(s) named are ones this sync would REWRITE, and"
    echo "###   each row's state= says which way that job would be hurt:"
    echo "###   state=RUNNING -- the job is READING that file NOW, ON ANOTHER NFS CLIENT."
    echo "###     The rename-install does NOT protect a reader on another node: its inode is"
    echo "###     unlinked under it, bash reads an error, calls it EOF, and the job exits 0"
    echo "###     having run half its rows."
    echo "###   state=PENDING -- the job has NOT STARTED. It will run whatever bytes are on"
    echo "###     disk when Slurm starts it, not the ones it was submitted against, and a"
    echo "###     dependency owner starting after this sync would silently run a different"
    echo "###     script with nothing anywhere reporting it. (HARNESS-SYNC-4.)"
    echo "###   Wait for the job(s) -- a PENDING one has to RUN and FINISH, not just start --"
    echo "###   or, if you have MEASURED that none of them will read the file, re-run with:"
    echo "###     harness_sync.sh $COMMIT --force --reason \"<why this is the right call>\""
    [ "$FORCE" = 1 ] && [ -z "$REASON" ] && echo "###   (--force WITHOUT --reason is still a refusal.)"
    SYNC_RC6=6   # HARNESS-SYNC-4-1 DEFERRED EXIT
  else
    RS_MARK="force-readset commit=$SHA at=$(date -Is) hits=$(sort -u "$RSHITS" | wc -l) reason=$(printf '%s' "$REASON" | tr '\n' ' ')"
    printf '%s\n' "$RS_MARK" > "$RECORD.force-readset" || {
      echo "### SYNC FATAL: could not write $RECORD.force-readset" >&2; exit 2; }
    echo "### SYNC FORCED over the read set of $(sort -u "$RSHITS" | sed -n 's/.* running=\([^ ]*\) .*/\1/p' | sort -u | wc -l) RUNNING/PENDING job(s) reason=$REASON"
    echo "###   marker: $RECORD.force-readset"
    echo "###   ^ copy this line into the lane log row for this sync."
  fi
fi

# ---- THE COMBINED REFUSAL (HARNESS-SYNC-4-1) -------------------------------
# Both checks have now RUN and both have PRINTED.  This is the only place either
# one exits, so an operator holding an rc-4 refusal is holding the rc-6 rows too
# and can clear one queue state instead of two.
if [ "$SYNC_RC4" != 0 ] || [ "$SYNC_RC6" != 0 ]; then
  echo "### SYNC REFUSED -- rc4_rows=$(wc -l < "$REF_REFUSE") rc6_rows=$(sort -u "$RSHITS" | wc -l) rc4_tolerated=$(wc -l < "$REF_TOL")"
  if [ "$SYNC_RC4" != 0 ] && [ "$SYNC_RC6" != 0 ]; then
    echo "###   BOTH CHECKS REFUSED. Exiting rc 4 (rc 4 takes precedence -- it is the"
    echo "###   cheaper one to clear, by repin), but the rc-6 rows above are OWED TOO:"
    echo "###   repinning the queued job does NOT make this sync legal. Clear both."
    exit 4
  fi
  [ "$SYNC_RC4" != 0 ] && { echo "###   rc 4 only: no job's read set intersects the install set."; exit 4; }
  echo "###   rc 6 only: no PENDING job of ours is pinned to another commit."
  exit 6
fi

# ---- install: temp file in the TARGET dir, then mv -f -----------------------
echo "### harness sync: task=$TASK_DIR repo=$REPO commit=$SHA files=$TOTAL"
inst=0; same=0; failed=0
while IFS='|' read -r trel w; do
  dst="$TASK_DIR/$trel"
  if [ -z "$w" ] || ! git -C "$REPO" cat-file blob "$SHA:$w" > "$TMP/blob" 2>/dev/null; then
    echo "SYNC NO-BLOB   $trel  (nothing at $SHA:${w:-<no mapping>})"; failed=$((failed + 1)); continue
  fi
  bmd5=$(md5sum "$TMP/blob" | awk '{print $1}')
  if [ -f "$dst" ] && [ "$(md5sum "$dst" | awk '{print $1}')" = "$bmd5" ]; then
    echo "SYNC same      $trel  $bmd5"; same=$((same + 1)); continue
  fi
  # The temp file must be in the TARGET directory: `mv` across filesystems is a
  # copy, and a copy is not the atomic replacement a live task dir needs.
  stage="$(mktemp "$(dirname -- "$dst")/.harness_sync.XXXXXX")" || {
    echo "SYNC FAIL      $trel  (could not stage beside it)"; failed=$((failed + 1)); continue; }
  if ! cp -f "$TMP/blob" "$stage"; then
    echo "SYNC FAIL      $trel  (could not write the stage file)"; rm -f "$stage"; failed=$((failed + 1)); continue
  fi
  mode=$(git -C "$REPO" ls-tree "$SHA" -- "$w" | awk '{print $1}')
  case "$mode" in 100755) chmod 755 "$stage";; *) chmod 644 "$stage";; esac
  if ! mv -f "$stage" "$dst"; then
    echo "SYNC FAIL      $trel  (rename-install failed)"; rm -f "$stage"; failed=$((failed + 1)); continue
  fi
  # Verify the INSTALLED file, with direct file arguments (law 15): a piped
  # compare is a known false-mismatch source here.
  gmd5=$(md5sum "$dst" | awk '{print $1}')
  if [ "$gmd5" = "$bmd5" ]; then
    echo "SYNC installed $trel  $bmd5"; inst=$((inst + 1))
  else
    echo "SYNC VERIFY FAILED $trel  on-disk=$gmd5 $SHA:$w=$bmd5"; failed=$((failed + 1))
  fi
done < "$SET"

if [ -s "$ABSENT" ]; then
  echo "### SYNC ABSENT ($(wc -l < "$ABSENT")): in $SHA, not in the task dir, NOT installed --"
  echo "###   the drift check does not read them either, so the writer and the checker agree."
  echo "###   To bring one in, NAME it -- and this writer still installs it, not a bare"
  echo "###   cat-file:  harness_sync.sh $COMMIT --add <task path>"
  sed 's/^/###   /' "$ABSENT"
fi

echo "### SYNC SUMMARY commit=$SHA files=$TOTAL installed=$inst unchanged=$same failed=$failed"
if [ "$failed" -gt 0 ]; then
  echo "### SYNC FAILED -- $failed file(s) did not install or did not verify. The record is"
  echo "###   NOT advanced, so --check still compares against the last good sync."
  exit 5
fi

# The record is written LAST and only on success: it is the claim "every task
# copy is this commit's bytes", and a half-finished sync must not make it.
printf '%s\n' "$SHA" > "$TMP/record" && mv -f "$TMP/record" "$RECORD" || {
  echo "### SYNC FAILED -- could not write $RECORD"; exit 5; }
echo "### SYNC RECORDED $RECORD = $SHA"
# THE HONEST STATE OF A NO-OP WHOSE RECORD WAS BEHIND (HARNESS-SYNC-9). If every
# mapped copy already matched $SHA, the task dir was AT two commits at once for
# the mapped set, and the record naming the older one was simply the older true
# statement. Moving it installed nothing and hid nothing -- but it is a change to
# what `--check` compares against tomorrow, so it is stated, with the word that
# says WHY it was safe.
if [ "$NOOP" = 1 ] && [ "${REC_OLD:-}" != "$SHA" ]; then
  echo "### RECORD ADVANCED ${REC_OLD:-<none>} -> $SHA bytes-identical"
  echo "###   nothing was installed: every mapped task copy already held these bytes, so the"
  echo "###   record was behind the disk, not the disk behind the record."
fi

# ---- THE PIN REPORT: every job root that is NOT at the commit just synced ----
# DET-1-1.  The rc-4 refusal above is a PRE-condition and it can only see jobs
# that are ALREADY queued; MERGE-T's relock was submitted 99 s after DET-1's sync
# with a pin copied from earlier in the session, and no refusal could have fired.
# So the sync also says, AFTERWARDS, which pin files no longer name the task
# dir's own commit -- because that is the list of jobs that will die at a drift
# gate, and it costs one squeue to print.  Three classes and they are not the
# same problem:
#   PENDING  the rc-4 rows above, forced through.  These die on their next start.
#   RUNNING  the job ran its drift gate against the harness on disk at start:
#            past its DRIFT gate -- drift only. Whether it is still READING an
#            installed file is a DIFFERENT question, answered BEFORE the install
#            by the rc-6 read-set check above. Named here for the record.
#   no job   a stale pin left over from a finished lane. Harmless until somebody
#            submits behind it -- which is exactly what happened -- so the line
#            says the actuator: rewrite it AT SUBMIT.
ALLROWS="$TMP/all.txt"; : > "$ALLROWS"
"$SQUEUE" -u glvov -h -o '%i %j %T' > "$ALLROWS" 2>/dev/null || : > "$ALLROWS"
pin_total=0; pin_match=0; pin_pd=0; pin_run=0; pin_stale=0
PINLINES="$TMP/pinlines.txt"; : > "$PINLINES"
while read -r pd; do
  [ -n "${pd:-}" ] || continue
  pin=$(sed -e 's/#.*//' -e 's/[[:space:]]//g' "$pd/HARNESS_COMMIT" 2>/dev/null | grep -m1 . || true)
  [ -n "$pin" ] || continue
  pin_total=$((pin_total + 1))
  pinsha=$(git -C "$REPO" rev-parse --verify "${pin}^{commit}" 2>/dev/null || echo "$pin")
  if [ "$pinsha" = "$SHA" ]; then pin_match=$((pin_match + 1)); continue; fi
  b=$(basename -- "$pd")
  jid=; jname=; jstate=
  while read -r cid cname cstate; do
    [ -n "${cname:-}" ] || continue
    stem=${cname%%-*}
    match=0
    [ "$b" = "$cname" ] && match=1
    [ "${cname#"$b"-}" != "$cname" ] && match=1
    [ "${b#"$stem"}" != "$b" ] && match=1
    [ "$match" = 1 ] || continue
    jid=$cid; jname=$cname; jstate=$cstate
    [ "$cstate" = PENDING ] && break      # a PENDING match is the one that matters
  done < "$ALLROWS"
  case "${jstate:-}" in
    PENDING)
      pin_pd=$((pin_pd + 1))
      printf '###   PENDING  %s pinned=%s job=%s %s -- IT WILL DIE at its drift gate. Repin: harness_commit_resolve.sh --write %s\n' \
        "$pd/HARNESS_COMMIT" "$pinsha" "$jid" "$jname" "$pd" >> "$PINLINES";;
    "")
      pin_stale=$((pin_stale + 1))
      printf '###   STALE    %s pinned=%s no job of ours matches -- stale pin, rewrite AT SUBMIT: harness_commit_resolve.sh --write %s\n' \
        "$pd/HARNESS_COMMIT" "$pinsha" "$pd" >> "$PINLINES";;
    *)
      pin_run=$((pin_run + 1))
      printf '###   %-8s %s pinned=%s job=%s %s -- past its drift gate (drift only; read set checked pre-install)\n' \
        "$jstate" "$pd/HARNESS_COMMIT" "$pinsha" "$jid" "$jname" >> "$PINLINES";;
  esac
done < "$PINDIRS"
echo "### SYNC PIN REPORT commit=$SHA -- job roots whose pin is not this commit:"
if [ -s "$PINLINES" ]; then cat "$PINLINES"; else echo "###   (none: every pin file names $SHA)"; fi
echo "### SYNC PIN SUMMARY commit=$SHA pins=$pin_total match=$pin_match pending=$pin_pd running=$pin_run stale=$pin_stale"

# The drift line from the NEW state, from the same checker the jobs run.
echo "### the drift line from the new state:"
bash "$DRIFT" "$SHA" "$TASK_DIR" "$REPO" 2>&1 | tail -2
exit 0
