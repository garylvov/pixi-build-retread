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
#          `<jid> <name> <workdir> <sbatch path or ->`, used INSTEAD of the live
#          `squeue -t R` + `scontrol show job`. Nothing in production passes it.
#
#   rc 0  synced (or --check found nothing edited)
#   rc 2  FATAL: bad arguments, no repo, no such commit, no mapping library
#   rc 3  --check: at least one task copy differs from the last synced commit
#   rc 4  REFUSED: a PENDING job of ours is pinned to a different commit
#   rc 5  a file failed to install or failed its md5 verification
#   rc 6  REFUSED: a RUNNING job of ours may still be READING a file this sync
#         would install.  A cross-NFS-client rename is NOT atomic for a reader
#         on another node -- it truncates that reader's script silently.
#         (HARNESS-SYNC-3.)  Override: --force --reason "<why>".
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

if [ -s "$REFUSALS" ]; then
  echo "### SYNC would strand these PENDING jobs -- they are pinned to another commit:"
  sed 's/^/###   /' "$REFUSALS"
  if [ "$FORCE" != 1 ]; then
    echo "### SYNC REFUSED (rc 4). A queued job has snapshotted NOTHING: syncing the task"
    echo "###   dir to $SHA kills it at its own drift gate the moment it starts."
    echo "###   Let them run, repin them (harness_commit_resolve.sh --write <job root> $SHA),"
    echo "###   or re-run with --force --reason \"<why this is the right call>\"."
    exit 4
  fi
  if [ -z "$REASON" ]; then
    echo "### SYNC REFUSED (rc 4) -- --force WITHOUT --reason is still a refusal. The rows"
    echo "###   above are what you are overriding; say why, and it goes in the log row:"
    echo "###     harness_sync.sh $COMMIT --force --reason \"<why>\""
    exit 4
  fi
  echo "### SYNC FORCED over $(wc -l < "$REFUSALS") pinned PENDING job(s) reason=$REASON"
  echo "###   ^ copy this line into the lane log row for this sync."
fi

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
# with the READ SET of every RUNNING job of ours, and a non-empty intersection is
# a REFUSAL, before the first byte.  A running job whose read set cannot be
# DETERMINED is treated as reading EVERYTHING -- an unreadable job is not a safe
# job (law 9).
INSTSET="$TMP/instset.txt"; : > "$INSTSET"
while IFS='|' read -r trel w; do
  [ -n "$w" ] || continue
  git -C "$REPO" cat-file blob "$SHA:$w" > "$TMP/preblob" 2>/dev/null || continue
  if [ -f "$TASK_DIR/$trel" ] &&
     [ "$(md5sum "$TASK_DIR/$trel" | awk '{print $1}')" = "$(md5sum "$TMP/preblob" | awk '{print $1}')" ]; then
    continue
  fi
  printf '%s\n' "$trel" >> "$INSTSET"
done < "$SET"

# Every `bash|source|.` reference in a script, as a BASENAME.  A reference whose
# basename is a variable is resolved from a same-file assignment (`FAST_ENV=$T/
# tools/retread_fast_env.sh` then `. "$FAST_ENV"`, which is det1_proof2.sh); one
# that still cannot be resolved prints `?` and makes the whole job unknown.
refs_of () {                      # $1 = script path
  local f=$1
  grep -hoE '(^|[[:space:]])(bash|source|\.)[[:space:]]+[^[:space:];&|)]+' "$f" 2>/dev/null \
  | awk '{print $NF}' | tr -d '\042\047' | while IFS= read -r tok; do
      local b=${tok##*/} v r
      case "$b" in
        *'$'*)
          v=${b#*\$}; v=${v#\{}; v=${v%%[^A-Za-z0-9_]*}
          [ -n "$v" ] || { printf '?\n'; continue; }
          r=$(sed -n "s/^[[:space:]]*$v=.*\/\([A-Za-z0-9_.-]*\.\(sh\|sbatch\|bash\)\).*/\1/p" "$f" | head -1)
          if [ -n "$r" ]; then printf '%s\n' "$r"; else printf '?\n'; fi;;
        *.sh|*.sbatch|*.bash) printf '%s\n' "$b";;
        *) ;;                     # `bash -c`, `. /etc/profile`, flags: not a task copy
      esac
    done
}

RSHITS="$TMP/readset.txt"; : > "$RSHITS"
if [ -s "$INSTSET" ]; then
  RUNROWS="$TMP/run.txt"; : > "$RUNROWS"
  if [ -n "$RUNLIST" ]; then
    [ -f "$RUNLIST" ] || { echo "### SYNC FATAL: --running-list $RUNLIST does not exist" >&2; exit 2; }
    grep -v '^[[:space:]]*$' "$RUNLIST" > "$RUNROWS" || :
  else
    "$SQUEUE" -u glvov -h -t R -o '%i %j %Z' > "$TMP/rq.txt" 2>/dev/null || : > "$TMP/rq.txt"
    while read -r jid jname wd; do
      [ -n "${jid:-}" ] || continue
      cmd=$(scontrol show job "$jid" 2>/dev/null | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
      printf '%s %s %s %s\n' "$jid" "${jname:--}" "${wd:--}" "${cmd:--}" >> "$RUNROWS"
    done < "$TMP/rq.txt"
  fi
  while read -r jid jname wd script; do
    [ -n "${jid:-}" ] || continue
    # IN SCOPE: the job works in this task dir, runs a script from it, or owns a
    # pin dir here by the same name rule the PIN REPORT uses.
    scope=0
    case "${wd:-}"     in "$TASK_DIR"|"$TASK_DIR"/*) scope=1;; esac
    case "${script:-}" in "$TASK_DIR"|"$TASK_DIR"/*) scope=1;; esac
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
    # scripts it names that sit in the job root.
    RS="$TMP/rs.$jid.txt"; : > "$RS"
    if [ -z "${script:-}" ] || [ "$script" = "-" ] || [ ! -f "$script" ]; then
      printf '?\n' > "$RS"
    else
      refs_of "$script" > "$RS"
      jroot=$(dirname -- "$script")
      while IFS= read -r rb; do
        [ "$rb" = '?' ] && continue
        [ -f "$jroot/$rb" ] || continue
        refs_of "$jroot/$rb" >> "$RS"
      done < <(sort -u "$RS")
    fi
    if grep -qx '?' "$RS"; then
      why=unresolved-reference
      [ -f "${script:-/nonexistent}" ] || why=no-sbatch-found
      while IFS= read -r trel; do
        printf '### SYNC REFUSED rc=6 running=%s file=%s reason=%s\n' \
          "$jid" "$(basename -- "$trel")" "$why" >> "$RSHITS"
      done < "$INSTSET"
      continue
    fi
    while IFS= read -r trel; do
      tb=$(basename -- "$trel")
      grep -qxF -- "$tb" "$RS" || continue
      printf '### SYNC REFUSED rc=6 running=%s file=%s reason=read-by-%s\n' \
        "$jid" "$tb" "$(basename -- "$script")" >> "$RSHITS"
    done < "$INSTSET"
  done < "$RUNROWS"
fi

if [ -s "$RSHITS" ]; then
  sort -u "$RSHITS"
  if [ "$FORCE" != 1 ] || [ -z "$REASON" ]; then
    echo "### SYNC REFUSED (rc 6). The job(s) above are RUNNING ON ANOTHER NFS CLIENT and"
    echo "###   the file(s) named are ones this sync would REWRITE. The rename-install does"
    echo "###   NOT protect a reader on another node: its inode is unlinked under it, bash"
    echo "###   reads an error, calls it EOF, and the job exits 0 having run half its rows."
    echo "###   Wait for the job(s), or -- if you have MEASURED that none of them will read"
    echo "###   the file again -- re-run with:"
    echo "###     harness_sync.sh $COMMIT --force --reason \"<why this is the right call>\""
    [ "$FORCE" = 1 ] && [ -z "$REASON" ] && echo "###   (--force WITHOUT --reason is still a refusal.)"
    exit 6
  fi
  RS_MARK="force-readset commit=$SHA at=$(date -Is) hits=$(sort -u "$RSHITS" | wc -l) reason=$(printf '%s' "$REASON" | tr '\n' ' ')"
  printf '%s\n' "$RS_MARK" > "$RECORD.force-readset" || {
    echo "### SYNC FATAL: could not write $RECORD.force-readset" >&2; exit 2; }
  echo "### SYNC FORCED over the read set of $(sort -u "$RSHITS" | awk '{print $3}' | sort -u | wc -l) RUNNING job(s) reason=$REASON"
  echo "###   marker: $RECORD.force-readset"
  echo "###   ^ copy this line into the lane log row for this sync."
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
