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
#   usage: owner_snapshot.sh <job root> <script> [<script> ...] \
#                            [--arms <n>] [--roots <root> [<root> ...]]
#
#   Writes  <job root>/owner-snapshot/          the frozen scripts
#           <job root>/owner-snapshot/owner.sbatch   execs the FIRST script,
#                                                    passing "$@" through
#           <job root>/owner-snapshot/owner.wall     the DERIVED `--time=HH:MM:SS`
#                                                    for the caller's sbatch line
#   Prints  ### OWNER SNAPSHOT files=<n> root=<job root> src_commit=<sha> src_kind=<git|record> ...
#           ### OWNER SNAPSHOT froze file=<b> md5=<m> src_kind=<k> src_sha=<sha> dirty=<y/n>
#           ### OWNER SNAPSHOT wall=<s> ... (CLEANUP-WALL-1, below)
#   rc 0    frozen; rc 2 refused (and nothing was written that a caller may use)
#
# CLEANUP-WALL-1 (2026-09-07). THE WALL IS THE ONE PARAMETER OF A REAP THAT IS A
# FUNCTION OF THE WORK, AND IT WAS THE ONLY ONE NOBODY COMPUTED. Every cleanup
# owner in this campaign was submitted with a wall typed by hand at the submit
# site -- `--time=16:00:00` in det16_proof.sh's submit_owner(), `--time=06:00:00`
# in phaseN_cert.sh's CLEANUP_SBATCH_ARGS -- and a hand-typed wall is a constant
# in front of a variable quantity, so eventually it is too small. It was:
# det1f-cleanup 5999937 was given 6 h for 6.66 M entries, spent 17507 s removing
# certDET1F-5992569 (3,587,597 entries), started ws.DET1F-5992569 (3,068,868
# entries) at 03:14:59 and TIMED OUT 29 min later (`sacct -j 5999937`: TIMEOUT,
# Elapsed 06:00:04 against Timelimit 06:00:00), orphaning the second root.
#
# THE RATE IS MEASURED, NOT ASSUMED. Four cleanup rows, each with entries and
# wall printed by cleanup.sh itself:
#     5992050 certDET1F-5989192   953311 /  4492s = 212 entries/s
#     5992050 ws.DET1F-5989192    842018 /  4104s = 205 entries/s
#     5999937 certDET1F-5992569  3587597 / 17507s = 205 entries/s
#     6001240 certD141-6001140   2583379 / 12050s = 214 entries/s
# Stable to within 5% across jobs, nodes and roots, which is what makes it a
# planning constant rather than an observation. OWNER_UNLINK_RATE_PER_S is set
# BELOW the slowest measured row on purpose: a wall derived from an optimistic
# rate is the defect this file exists to remove.
#
# THE CENSUS COUNTS TOWARD THE WALL. cleanup.sh walks each root with
# `find "$r" | wc -l` and prints `### removing $r (entries=$N)` BEFORE it
# unlinks anything, so every reap pays a full metadata walk first; on NFS that
# is ~30 min per 3 M entries. It gets its own allowance rather than being hidden
# inside the margin, because a term that is not named cannot be re-measured.
#
# THE ROOTS USUALLY DO NOT EXIST YET. Owners are submitted BEFORE arm 1 (that is
# the point of an `afterany` owner), so at submit time a declared root is a name
# and not a tree. A root that IS present is censused; a root that is absent is
# sized from OWNER_ENTRIES_PER_ROOT_EST, which is this campaign's LARGEST
# measured root (certDET1F-5992569, 3,587,597 entries, rounded up).
#
# AND THE ESTIMATE IS RE-DERIVED INSIDE THE OWNER, WHICH IS THE ACTUATOR (law 9).
# An estimate that is only checked at submit is a guess with a nicer name. The
# generated owner.sbatch re-censuses the roots it was handed, compares against
# the entry count its wall was derived to cover, and if the real tree is bigger
# it prints `### OWNER WALL SHORT census=<n> covers=<n>` and RESUBMITS ITSELF for
# the remainder behind its own job id with a freshly derived wall -- so the
# reap removes what fits and a successor finishes the rest, instead of dying
# silently at a wall like 5999937 did. cleanup_gated.sh's ABSENT-root branch
# (MERGE-N-1) is what makes the successor safe: a root the first pass finished
# is a no-op exit 0 for the second, not a refusal.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB_ROOT=${1:-}; shift || true

########## CLEANUP-WALL-1: THE CONSTANTS, EACH CITED TO THE ROWS IT CAME FROM ##
# Every one of these is a MEASURED number or an explicit policy choice, and each
# is named so a later lane can re-measure it rather than re-guess it.
OWNER_UNLINK_RATE_PER_S=${OWNER_UNLINK_RATE_PER_S:-200}   # rows: 212 / 205 / 205 / 214 (jobs 5992050 x2, 5999937, 6001240)
OWNER_CENSUS_RATE_PER_S=${OWNER_CENSUS_RATE_PER_S:-2800}  # rows: 2807.7 / 2915.4 (clean walks, jobs 5999937 and 6001240); see below
# MEASURED 2026-09-07, not quoted. This read 1700 with the comment "~30 min per
# 3 M entries over NFS = 1667/s, rounded" -- a STEWARD'S SENTENCE, never a row.
# The walk that produces `entries=<N>` is cleanup.sh's `N=$(find "$r" | wc -l)`
# immediately before its `### removing` row, so the interval from the PREVIOUS
# root's `### removed` row to the NEXT root's `### removing` row is exactly one
# census walk and nothing else. Two such CLEAN intervals exist in the logs:
#   job 5999937 -- 3,068,868 entries in 1093 s = 2807.7/s (the slowest, and the
#     value below):
#       ### removed /oscar/.../certDET1F-5992569 rc=0 wall=17507s exists_after=no 2026-09-07T02:56:46-04:00
#       ### removing /oscar/.../ws.DET1F-5992569  (entries=3068868) start 2026-09-07T03:14:59-04:00
#   job 6001240 -- 2,154,463 entries in 739 s = 2915.4/s:
#       ### removed /oscar/.../certD141-6001140 rc=0 wall=12050s exists_after=no 2026-09-07T04:07:08-04:00
#       ### removing /oscar/.../ws.D141-6001140  (entries=2154463) start 2026-09-07T04:19:27-04:00
#   job 6013780 -- a FIRST-root interval, so it also contains hostname/date/
#     checkquota and is a LOWER BOUND, quoted only as corroboration: 3,068,868
#     entries in <=1031 s from the gate's own `2026-09-07T03:58:29-04:00` row =
#     >=2976.6/s:
#       ### removing /oscar/.../ws.DET1F-5992569  (entries=3068868) start 2026-09-07T04:15:40-04:00
# All five measurable walks land in 2807-2977/s. 2800 is below every one of them
# and is the FLOOR of the measurements, not their mean: this term buys wall, so
# it must under-estimate the rate, never over-estimate it. The old 1700 was
# 1.65x conservative for no measured reason, and a term nobody can re-derive
# from a row is the thing this constant block exists to abolish.
OWNER_WALL_MARGIN=${OWNER_WALL_MARGIN:-2}                 # 2x on the unlink term
# TWO metadata walks are paid per reap, not one: cleanup.sh censuses each root
# before it unlinks (`### removing $r (entries=$N)`), and the owner re-censuses
# at start to decide whether its wall still covers the tree. Named rather than
# folded into the margin so it can be re-measured on its own.
OWNER_CENSUS_WALKS=${OWNER_CENSUS_WALKS:-2}
OWNER_WALL_FLOOR_S=${OWNER_WALL_FLOOR_S:-3600}            # a tiny root still gets a usable wall
# The cap is OURS, not the scheduler's: `sinfo -p batch -o %l` = infinite and
# `sacctmgr show qos normal format=MaxWall` is EMPTY (both measured 2026-09-07).
# Past a day the right answer is a split, and the split exists -- see the
# self-continuation in the generated owner.sbatch.
OWNER_WALL_MAX_S=${OWNER_WALL_MAX_S:-86400}
# The largest root this campaign measured: certDET1F-5992569 = 3,587,597
# entries (job 5999937), rounded up. Used ONLY for a root that does not exist
# yet, which is the normal case because owners are submitted before arm 1.
OWNER_ENTRIES_PER_ARM_EST=${OWNER_ENTRIES_PER_ARM_EST:-3600000}

# ONE derivation, defined once and SHIPPED into the generated owner.sbatch by
# `declare -f` below, so the number the submitter derives and the number the
# owner re-derives cannot drift apart into two implementations.
owner_census () {                # $@ = declared roots; echoes "<entries> <present> <absent>"
  local r n tot=0 pres=0 abs=0
  for r in "$@"; do
    if [ -e "$r" ]; then
      n=$(find "$r" -maxdepth 16 2>/dev/null | wc -l)
      tot=$((tot + n)); pres=$((pres + 1))
    else
      tot=$((tot + OWNER_ENTRIES_PER_ARM_EST * OWNER_ARMS)); abs=$((abs + 1))
    fi
  done
  echo "$tot $pres $abs"
}

owner_wall_derive () {           # $1 = entries; echoes "<wall seconds> <census allowance seconds>"
  local e=${1:-0} unlink census wall
  unlink=$(( (e + OWNER_UNLINK_RATE_PER_S - 1) / OWNER_UNLINK_RATE_PER_S ))
  census=$(( (e + OWNER_CENSUS_RATE_PER_S - 1) / OWNER_CENSUS_RATE_PER_S ))
  census=$(( census * OWNER_CENSUS_WALKS ))
  wall=$(( unlink * OWNER_WALL_MARGIN + census ))
  [ "$wall" -lt "$OWNER_WALL_FLOOR_S" ] && wall=$OWNER_WALL_FLOOR_S
  [ "$wall" -gt "$OWNER_WALL_MAX_S" ] && wall=$OWNER_WALL_MAX_S
  echo "$wall $census"
}

owner_wall_hms () {              # $1 = seconds -> HH:MM:SS, rounded UP to the minute
  local s=${1:-0}
  s=$(( ( (s + 59) / 60 ) * 60 ))
  printf '%02d:%02d:%02d\n' $(( s / 3600 )) $(( (s % 3600) / 60 )) $(( s % 60 ))
}

# THE OWNER'S OWN RE-DERIVATION. Shipped verbatim into the generated
# owner.sbatch by `declare -f`, so what the submitter computed and what the
# owner checks are one text and cannot diverge. It runs INSIDE the owner, at
# start, when the roots are real trees rather than names -- and when the tree is
# bigger than the wall was derived to cover it does not shrug and it does not
# die at the wall: it says so on one row and submits its own continuation for
# whatever it cannot finish. cleanup_gated.sh's ABSENT-root branch makes the
# second pass a no-op on any root the first one finished.
owner_wall_check () {            # $@ = the roots this owner was handed
  local c pres abs new nc cj rc hms
  read -r c pres abs < <(owner_census "$@")
  hms=$(owner_wall_hms "$OWNER_WALL_S")
  echo "### OWNER WALL CENSUS entries=$c present=$pres absent=$abs covers=$OWNER_WALL_COVERS wall=${OWNER_WALL_S}s ($hms) cont=$OWNER_CONT_N"
  [ "$c" -gt "$OWNER_WALL_COVERS" ] || return 0
  read -r new nc < <(owner_wall_derive "$c")
  hms=$(owner_wall_hms "$new")
  echo "### OWNER WALL SHORT census=$c covers=$OWNER_WALL_COVERS derived=${new}s ($hms) rate=$OWNER_UNLINK_RATE_PER_S margin=$OWNER_WALL_MARGIN census_allow=$nc"
  if [ "$OWNER_CONT_N" -ge "$OWNER_CONT_MAX" ]; then
    echo "### OWNER WALL CONTINUATION REFUSED: this is continuation $OWNER_CONT_N of at most $OWNER_CONT_MAX."
    echo "###   A chain that never ends is not an actuator either. RUN THIS BY HAND -- it is the"
    echo "###   only thing that returns the remaining inodes:"
    echo "    env -u SLURM_JOB_ID sbatch --partition=${SLURM_JOB_PARTITION:-batch} --qos=${SLURM_JOB_QOS:-normal} --cpus-per-task=1 --mem=4G --time=$hms $OWNER_SELF $*"
    return 0
  fi
  cj=$(env -u SLURM_JOB_ID sbatch --parsable \
        --partition="${SLURM_JOB_PARTITION:-batch}" --qos="${SLURM_JOB_QOS:-normal}" \
        --cpus-per-task=1 --mem=4G --time="$hms" \
        --job-name="${SLURM_JOB_NAME:-owner}-cont" \
        --output="$OWNER_OUT_DIR/owner-cont-%j.out" \
        --dependency=afterany:"${SLURM_JOB_ID:-0}" \
        --export=ALL,OWNER_CONT_N=$((OWNER_CONT_N + 1)) \
        "$OWNER_SELF" "$@" 2>&1); rc=$?
  if [ "$rc" = 0 ]; then
    echo "### OWNER WALL CONTINUATION submitted job=$cj time=$hms after ${SLURM_JOB_ID:-0} -- this pass removes what fits, that one finishes the remainder"
  else
    echo "### OWNER WALL CONTINUATION FAILED rc=$rc output: $cj"
    echo "###   The remainder has NO owner. RUN THIS BY HAND:"
    echo "    env -u SLURM_JOB_ID sbatch --partition=${SLURM_JOB_PARTITION:-batch} --qos=${SLURM_JOB_QOS:-normal} --cpus-per-task=1 --mem=4G --time=$hms $OWNER_SELF $*"
  fi
  return 0
}

########## the argument split: scripts to freeze, roots to size the wall from ##
SCRIPTS=(); ROOTS=(); OWNER_ARMS=1; argmode=scripts
while [ "$#" -gt 0 ]; do
  case $1 in
    --roots) argmode=roots ;;
    --arms)  shift; OWNER_ARMS=${1:-1}; [ "$OWNER_ARMS" -ge 1 ] 2>/dev/null || OWNER_ARMS=1 ;;
    *) if [ "$argmode" = roots ]; then ROOTS+=("$1"); else SCRIPTS+=("$1"); fi ;;
  esac
  shift
done
[ -n "$JOB_ROOT" ] && [ "${#SCRIPTS[@]}" -ge 1 ] || {
  echo "### OWNER SNAPSHOT REFUSED: usage: owner_snapshot.sh <job root> <script> [<script>...] [--arms <n>] [--roots <root>...]"; exit 2; }
set -- "${SCRIPTS[@]}"

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

########## DET-1-6-a: WHERE THE BYTES CAME FROM, NEVER A GUESS ################
# THE DEFECT, MEASURED. This file used to read `src_commit` out of
# `<job root>/../tools/.harness_synced_commit` and print it as the provenance of
# the frozen copy. That record is the TASK PIN -- what the task tree under
# `tools/` was last synced to -- and it says nothing about the bytes actually
# copied. det16_proof.sh froze the WORKTREE's files (at 6b3b669) while this row
# advertised src_commit=8108ca4, the pin, and the proof author had to add a
# hand-written PROVENANCE NOTE with two md5sums beside the row so a reader could
# not be misled by it. A row that needs a note beside it saying what it really
# means is a defect in the row.
#
# THE FIX IS TO IDENTIFY THE SOURCE, PER FILE, AND TO REFUSE WHEN IT CANNOT BE.
#   * a source directory inside a git repo, with the file TRACKED there:
#     kind=git, sha=`git rev-parse HEAD` of that tree, and its dirty state for
#     that file (a dirty file is NOT the commit it sits on, and saying so is the
#     whole point).
#   * a task copy -- the task tree is not a repo -- : kind=record, sha = the
#     `.harness_synced_commit` beside it, which for a task copy IS the identity
#     of those bytes rather than a pin over somebody else's.
#   * neither: REFUSE. An unidentifiable source is exactly the case the old
#     `src_commit=unknown` printed and carried on, and carrying on is how a
#     frozen copy of nobody-knows-what ends up in a job root for hours.
# The md5 of every frozen file is stamped too, in `md5sum -c` form, so the row
# cannot be misread even when the sha is right: bytes are the fact, the sha is
# the claim about them.
SRC_RECORD=; SRC_RECORD_PATH=
for c in "$JOB_ROOT/../tools/.harness_synced_commit" "$JOB_ROOT/tools/.harness_synced_commit" \
         "$(dirname -- "${SCRIPTS[0]}")/../.harness_synced_commit" \
         "$(dirname -- "${SCRIPTS[0]}")/../tools/.harness_synced_commit"; do
  [ -f "$c" ] && { SRC_RECORD=$(head -c 40 "$c"); SRC_RECORD_PATH=$c; break; }
done

owner_src_identity () {          # $1 = a file about to be frozen; echoes "<kind> <sha> <dirty>"
  local f=$1 d sha st r
  d=$(cd -- "$(dirname -- "$f")" 2>/dev/null && pwd) || { echo "none - -"; return 0; }
  if git -C "$d" rev-parse --git-dir >/dev/null 2>&1 &&
     git -C "$d" ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then
    sha=$(git -C "$d" rev-parse HEAD 2>/dev/null)
    st=$(git -C "$d" status --porcelain -- "$f" 2>/dev/null)
    if [ -n "$sha" ]; then
      if [ -n "$st" ]; then echo "git $sha yes"; else echo "git $sha no"; fi
      return 0
    fi
  fi
  for r in "$d/../tools/.harness_synced_commit" "$d/../.harness_synced_commit" \
           "$d/.harness_synced_commit" "$d/tools/.harness_synced_commit"; do
    [ -f "$r" ] && { echo "record $(head -c 40 "$r") -"; return 0; }
  done
  echo "none - -"
}

PROV=$SNAP/owner-snapshot.provenance
MD5S=$SNAP/owner-snapshot.md5
: > "$PROV"; : > "$MD5S"
SRC_KIND=; SRC_COMMIT=; SRC_DIRTY=
seen=" "
n=0
copy_one () {                    # $1 = absolute path of a script to freeze
  local src=$1 b=${1##*/} rb rp kind sha dirty m
  case "$seen" in *" $b "*) return 0;; esac
  [ -f "$src" ] || { echo "### OWNER SNAPSHOT REFUSED: $src does not exist"; return 2; }
  # IDENTIFY BEFORE COPYING. A copy whose source cannot be named is a copy
  # nobody can trace back, and this file exists to make the frozen bytes
  # traceable.
  read -r kind sha dirty < <(owner_src_identity "$src")
  if [ "$kind" = none ]; then
    echo "### OWNER SNAPSHOT REFUSED: cannot identify the source of $src."
    echo "###   Its directory is not a git worktree that tracks it, and there is no"
    echo "###   .harness_synced_commit beside it. Freezing it would put a copy of"
    echo "###   nobody-knows-what in the job root under a row claiming provenance."
    return 2
  fi
  cp -p "$src" "$SNAP/$b" || { echo "### OWNER SNAPSHOT REFUSED: cannot copy $src"; return 2; }
  m=$(md5sum "$SNAP/$b" | awk '{print $1}')
  printf '%s  %s\n' "$m" "$b" >> "$MD5S"
  printf '%s\t%s\t%s\t%s\t%s\n' "$b" "$m" "$kind" "$sha" "$dirty" >> "$PROV"
  echo "### OWNER SNAPSHOT froze file=$b md5=$m src_kind=$kind src_sha=$sha dirty=$dirty from=$src"
  [ -n "$SRC_KIND" ] || { SRC_KIND=$kind; SRC_COMMIT=$sha; SRC_DIRTY=$dirty; }
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

########## CLEANUP-WALL-1: DERIVE THE WALL, HERE, FROM THE DECLARED ROOTS ######
# The submitter derives; the OWNER re-derives (owner_wall_check, shipped into
# the generated sbatch below by `declare -f`, so there is exactly one text).
OWNER_ENTRIES=0; OWNER_PRESENT=0; OWNER_ABSENT=0
OWNER_WALL_S=0; OWNER_CENSUS_ALLOW_S=0; OWNER_WALL_HMS=
if [ "${#ROOTS[@]}" -ge 1 ]; then
  read -r OWNER_ENTRIES OWNER_PRESENT OWNER_ABSENT < <(owner_census "${ROOTS[@]}")
  read -r OWNER_WALL_S OWNER_CENSUS_ALLOW_S < <(owner_wall_derive "$OWNER_ENTRIES")
  OWNER_WALL_HMS=$(owner_wall_hms "$OWNER_WALL_S")
  printf -- '--time=%s\n' "$OWNER_WALL_HMS" > "$SNAP/owner.wall"
fi

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
  echo "# frozen from src_commit=$SRC_COMMIT (src_kind=$SRC_KIND dirty=$SRC_DIRTY, task pin record=${SRC_RECORD:-none}) at $(date -Is)"
  # A directive, not the last word: a `--time=` on the caller's sbatch line
  # overrides it. It is here so an owner submitted with NO --time still gets the
  # derived one rather than the partition default (5 min, measured with
  # `sinfo -p batch -o %L`), which would kill every reap instantly.
  [ -n "$OWNER_WALL_HMS" ] && echo "#SBATCH --time=$OWNER_WALL_HMS"
  echo 'set -u'
  echo "echo \"### OWNER SNAPSHOT running frozen copy $SNAP/$FIRST src_commit=$SRC_COMMIT src_kind=$SRC_KIND\""
  # --- the re-derivation the owner performs on ITSELF (law 9's actuator) ------
  echo "OWNER_UNLINK_RATE_PER_S=$OWNER_UNLINK_RATE_PER_S"
  echo "OWNER_CENSUS_RATE_PER_S=$OWNER_CENSUS_RATE_PER_S"
  echo "OWNER_CENSUS_WALKS=$OWNER_CENSUS_WALKS"
  echo "OWNER_WALL_MARGIN=$OWNER_WALL_MARGIN"
  echo "OWNER_WALL_FLOOR_S=$OWNER_WALL_FLOOR_S"
  echo "OWNER_WALL_MAX_S=$OWNER_WALL_MAX_S"
  echo "OWNER_ENTRIES_PER_ARM_EST=$OWNER_ENTRIES_PER_ARM_EST"
  echo "OWNER_ARMS=$OWNER_ARMS"
  echo "OWNER_WALL_COVERS=$OWNER_ENTRIES"
  echo "OWNER_WALL_S=$OWNER_WALL_S"
  echo "OWNER_SELF=$SNAP/owner.sbatch"
  echo "OWNER_OUT_DIR=$SNAP"
  echo 'OWNER_CONT_N=${OWNER_CONT_N:-0}'
  echo 'OWNER_CONT_MAX=${OWNER_CONT_MAX:-4}'
  declare -f owner_census owner_wall_derive owner_wall_hms owner_wall_check
  echo 'owner_wall_check "$@"'
  echo "exec bash $SNAP/$FIRST \"\$@\""
} > "$SNAP/owner.sbatch" || { echo "### OWNER SNAPSHOT REFUSED: cannot write $SNAP/owner.sbatch"; exit 2; }
chmod +x "$SNAP/owner.sbatch"

echo "### OWNER SNAPSHOT files=$n root=$JOB_ROOT src_commit=$SRC_COMMIT src_kind=$SRC_KIND src_dirty=$SRC_DIRTY task_pin_record=${SRC_RECORD:-none}${SRC_RECORD_PATH:+ ($SRC_RECORD_PATH)}"
if [ "${#ROOTS[@]}" -ge 1 ]; then
  echo "### OWNER SNAPSHOT wall=$OWNER_WALL_S (--time=$OWNER_WALL_HMS) from entries=$OWNER_ENTRIES rate=$OWNER_UNLINK_RATE_PER_S margin=$OWNER_WALL_MARGIN census_allow=$OWNER_CENSUS_ALLOW_S roots=${#ROOTS[@]} present=$OWNER_PRESENT absent=$OWNER_ABSENT arms=$OWNER_ARMS"
  echo "### OWNER SNAPSHOT wall file: $SNAP/owner.wall ($(cat "$SNAP/owner.wall"))"
else
  echo "### OWNER SNAPSHOT wall=UNDERIVED roots=0 -- this caller passed no --roots, so the"
  echo "###   owner keeps whatever --time the submit site types by hand. That is the shape"
  echo "###   that killed det1f-cleanup 5999937 (TIMEOUT at 06:00:04 with a root half gone);"
  echo "###   pass --roots <root>... and the wall is derived and printed instead."
fi
echo "### OWNER SNAPSHOT frozen: $(cd "$SNAP" && ls -1 *.sh 2>/dev/null | tr '\n' ' ')"
echo "### OWNER SNAPSHOT md5s: $MD5S ($(wc -l < "$MD5S") file(s)); provenance: $PROV"
echo "### OWNER SNAPSHOT sbatch: $SNAP/owner.sbatch -> $SNAP/$FIRST"
exit 0
