#!/usr/bin/env bash
# owner_snapshot.sh -- freeze a cleanup owner's scripts into the JOB ROOT and
# generate the sbatch that runs the frozen copy. HARNESS-SYNC-5.
#
# THE DEFECT, MEASURED. A cleanup owner is submitted as
#   sbatch --wrap 'bash <task dir>/merge-h/cleanup_gated.sh <roots>'
# Slurm snapshots the TOP-LEVEL script -- here, the one-line wrap -- and nothing
# else, so `cleanup_gated.sh` and the `cleanup.sh` it calls are read from the
# task tree LIVE, for as long as the owner runs. det1f-cleanup 5999937 was four
# hours into a single 3,587,597-entry unlink with those reads open; det141-cleanup
# 6001240 sat beside it. harness_sync's read-set check is RIGHT to refuse rc 6 on
# that -- rewriting a file a reader on another NFS client has open unlinks its
# inode under them, bash reads an error, calls it EOF, and the job exits 0 having
# run half its rows -- but the CAUSE is that a reap measured in hours holds a
# live read of a SYNCED file. Nothing can install while it runs, and reaps like
# that run every night.
#
# THE FIX IS NOT TO WEAKEN THE REFUSAL. It is to stop the owner reading a synced
# file at all: copy what it will read into the job root at SUBMIT time and submit
# a real sbatch script that execs the copy. The job root is job-local and nothing
# syncs into it, so a later install cannot reach the reader, and the read-set
# check then sees paths under the job root rather than under tools/ and stops
# refusing -- correctly, because there is now nothing to protect.
#
# THE SET IS DERIVED, NOT LISTED. tools/script_refs.sh is the SAME parser
# harness_sync.sh uses for its read set. Deriving the snapshot from a different
# parser than the one that judges it is how a snapshot ends up missing exactly
# the file the job sources, so there is one parser and both call it. References
# are followed transitively; a reference the parser cannot resolve is a REFUSAL,
# not a shrug, because a snapshot with a hole is worse than no snapshot: the job
# would fall back to the live path with a row saying it was frozen.
#
#   usage: owner_snapshot.sh <job root> <script> [<script> ...]
#
#   Writes  <job root>/owner-snapshot/          the frozen scripts
#           <job root>/owner-snapshot/owner.sbatch   execs the FIRST script,
#                                                    passing "$@" through
#   Prints  ### OWNER SNAPSHOT files=<n> root=<job root> src_commit=<sha>
#   rc 0    frozen; rc 2 refused (and nothing was written that a caller may use)
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB_ROOT=${1:-}; shift || true
[ -n "$JOB_ROOT" ] && [ "$#" -ge 1 ] || {
  echo "### OWNER SNAPSHOT REFUSED: usage: owner_snapshot.sh <job root> <script> [<script>...]"; exit 2; }

SR=$HERE/../tools/script_refs.sh
[ -f "$SR" ] || SR=$HERE/script_refs.sh
[ -f "$SR" ] || {
  echo "### OWNER SNAPSHOT REFUSED: no script_refs.sh -- the snapshot set would have to be"
  echo "###   guessed, and a guessed set is how the one file the job sources goes missing."
  exit 2; }
# shellcheck disable=SC1090
. "$SR"

SNAP=$JOB_ROOT/owner-snapshot
mkdir -p "$SNAP" || { echo "### OWNER SNAPSHOT REFUSED: cannot create $SNAP"; exit 2; }

# The commit the SOURCE tree is at, so a frozen copy can be traced back to what
# it was frozen from. Read at the point of use, never remembered.
SRC_COMMIT=unknown
for c in "$JOB_ROOT/../tools/.harness_synced_commit" "$JOB_ROOT/tools/.harness_synced_commit" \
         "$(dirname -- "$1")/../.harness_synced_commit" "$(dirname -- "$1")/../tools/.harness_synced_commit"; do
  [ -f "$c" ] && { SRC_COMMIT=$(head -c 40 "$c"); break; }
done

seen=" "
n=0
copy_one () {                    # $1 = absolute path of a script to freeze
  local src=$1 b=${1##*/} rb rp
  case "$seen" in *" $b "*) return 0;; esac
  [ -f "$src" ] || { echo "### OWNER SNAPSHOT REFUSED: $src does not exist"; return 2; }
  cp -p "$src" "$SNAP/$b" || { echo "### OWNER SNAPSHOT REFUSED: cannot copy $src"; return 2; }
  seen="$seen$b "
  n=$((n+1))
  # follow what it reads, with the SAME parser the sync judges with
  while IFS=$'\t' read -r rb rp; do
    [ -n "${rb:-}" ] || continue
    if [ "$rb" = '?' ]; then
      echo "### OWNER SNAPSHOT REFUSED: $b contains a reference this parser cannot resolve."
      echo "###   A snapshot with a hole is worse than none: the job would read the LIVE"
      echo "###   path for that one file while every row said it was frozen."
      return 2
    fi
    if [ "${rp:--}" != "-" ] && [ -f "$rp" ]; then
      copy_one "$rp" || return 2
    elif [ -f "$(dirname -- "$src")/$rb" ]; then
      copy_one "$(dirname -- "$src")/$rb" || return 2
    fi
    # a reference to something that is not beside the source and not resolvable
    # to a path is not a task copy (a system tool, a flag) -- refs_of already
    # dropped those; anything left that we cannot FIND is reported, not guessed.
  done < <(refs_of_sibling_resolved "$src")
  return 0
}

FIRST=
for s in "$@"; do
  a=$s
  case "$a" in /*) ;; *) a=$(cd "$(dirname -- "$a")" 2>/dev/null && pwd)/$(basename -- "$a");; esac
  [ -n "$FIRST" ] || FIRST=${a##*/}
  copy_one "$a" || { rm -f "$SNAP/owner.sbatch"; exit 2; }
done

# THE GENERATED SBATCH, AND IT CARRIES A LITERAL ABSOLUTE PATH ON PURPOSE.
# Slurm snapshots this top-level script at submit, so its bytes are frozen the
# moment sbatch returns; the path it names is under the JOB ROOT, which nothing
# installs into. Written literally rather than through a variable because
# script_refs.sh resolves a LITERAL absolute token and nothing else -- a
# `$SNAP/cleanup_gated.sh` here would read back as an unresolved path and the
# sync would go on refusing, which is the whole thing this removes.
{
  echo '#!/usr/bin/env bash'
  echo '# GENERATED by phase_template/owner_snapshot.sh -- do not edit.'
  echo "# frozen from src_commit=$SRC_COMMIT at $(date -Is)"
  echo 'set -u'
  echo "echo \"### OWNER SNAPSHOT running frozen copy $SNAP/$FIRST src_commit=$SRC_COMMIT\""
  echo "exec bash $SNAP/$FIRST \"\$@\""
} > "$SNAP/owner.sbatch" || { echo "### OWNER SNAPSHOT REFUSED: cannot write $SNAP/owner.sbatch"; exit 2; }
chmod +x "$SNAP/owner.sbatch"

echo "### OWNER SNAPSHOT files=$n root=$JOB_ROOT src_commit=$SRC_COMMIT"
echo "### OWNER SNAPSHOT frozen: $(cd "$SNAP" && ls -1 *.sh 2>/dev/null | tr '\n' ' ')"
echo "### OWNER SNAPSHOT sbatch: $SNAP/owner.sbatch -> $SNAP/$FIRST"
exit 0
