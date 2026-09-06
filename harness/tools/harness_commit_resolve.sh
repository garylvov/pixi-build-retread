#!/usr/bin/env bash
# harness_commit_resolve.sh -- say which harness commit this job is pinned to,
# and say WHERE that answer came from.  MERGE-N-2.
#
#   usage: HC=$(bash harness_commit_resolve.sh "<job root>") || exit ...
#          bash harness_commit_resolve.sh --write <job root> [<sha>] \
#               [--allow-older --reason "<why>"]
#
#   stdout  the commit-ish, or nothing when there is none to find
#   stderr  one `### HARNESS_COMMIT ...` provenance row, always
#   rc 0    resolved, or deliberately unset (the caller announces the check OFF)
#   rc 2    REFUSED: the file and the export disagree; or --write got a sha that
#           is not a commit, or no job root
#   rc 3    --write REFUSED: the sha is not the commit the task copies actually
#           ARE (DET-1-1), and no `--allow-older --reason` was given
#   rc 4    READ REFUSED (HARNESS-SYNC-2-1): the pin this job would run under is
#           not the commit the task copies actually ARE -- `.harness_synced_commit`
#           -- and no `--allow-older` marker authorises it.  WHY 4: 0/2/3 are
#           already taken above, and 2 and 6 are the phase templates' OWN FATAL
#           exits, so neither could be told apart from a template refusal; 4
#           already means "a pin disagrees with the sync record" on the WRITER
#           side of this seam, in harness_sync.sh, and this is the same fact
#           read from the other end.
#
# WHY THIS EXISTS, AND IT IS A SCHEDULER DEFECT, NOT A STYLE PREFERENCE.
# The pin reached the job ONLY through `sbatch --export=ALL,HARNESS_COMMIT=<sha>`,
# and in TWO CONSECUTIVE merge lanes that clause put the job into
#
#     JobState=PENDING Reason=launch_failed_requeued_held
#     ... "user env retrieval failed"
#
# -- Slurm re-runs the submitting user's login environment to build `ALL`, and
# when that retrieval times out the job is requeued AND HELD.  Nothing about the
# harness is wrong in that state and nothing in the job's own log says so,
# because the job never started.  A value the job needs is therefore carried in
# a FILE THE JOB OWNS, written by the submitter into the harness directory
# whose `artifacts/` the run is already writing to, and read here.
#
# THE ORDER, AND WHY THE FILE WINS.  `<job root>/HARNESS_COMMIT` first, the
# exported variable second.  The file is the artefact of the submission that
# actually happened; an exported value can be inherited from the SUBMITTING
# SHELL through `ALL` without anyone intending it, which is exactly how a merge
# lane opened on a stale `87366e7` while READERS-1 had already landed past it.
# The export is kept as the fallback so every existing caller keeps working
# unchanged -- this is additive, and a job that only exports still runs.
#
# AND IT REFUSES ON DISAGREEMENT.  Two different values means the submitter
# wrote one pin and exported another, and neither is more likely to be right.
# Guessing here would pin a three-hour relock to a harness nobody chose, so it
# names both and refuses (law 9: never coerce an error to a default).
set -uo pipefail

# ---- THE WRITER HALF (reader/writer law) ------------------------------------
# A file nobody writes is a fallback that never fires, so the submitter's half
# lives in this same file and there is exactly one command to remember:
#
#   bash harness_commit_resolve.sh --write <job root> [<sha>]
#
# It REFUSES a sha that is not a commit in the harness repo. That check costs
# nothing at SUBMIT time and is the only moment it is cheap: the alternative is
# a three-hour relock refusing at its drift gate on a typo.
#
# ── DET-1-1: THE PIN IS RESOLVED AT SUBMIT, NOT COPIED FROM EARLIER ──────────
# MEASURED, 2026-09-06.  DET-1 committed a harness change and installed it
# through the ONE writer at 14:06:19; `tools/.harness_synced_commit` said
# `873263f` from 14:06:20 onwards.  MERGE-T's `mBZ-relock` 5981194 was SUBMITTED
# at 14:07:59 -- ninety-nine seconds LATER -- carrying a pin file that still said
# `b7b699b`, because the lane had written that pin earlier in its session and the
# pin was a COPY of a value, not a resolution.  `harness_sync.sh`'s rc-4 refusal
# could not have caught it: that refusal reads `squeue -t PD` before it writes,
# and at 14:06 the job did not exist in any queue.  A guard that only looks at
# jobs ALREADY queued cannot see a job submitted a minute later, so "sync then
# submit" and "submit then sync" are indistinguishable to it and whoever submits
# second loses a three-hour relock at its drift gate.
#
# The fix is on this side of the seam, because THIS is the moment the pin is
# decided.  `--write` now reads the record `harness_sync.sh` writes -- the ONE
# statement of what the task copies actually ARE -- and:
#   * with NO sha argument it RESOLVES the pin from that record.  This is the
#     shape every submit line should use: `--write <job root>` immediately
#     before its `sbatch`, so the pin is whatever the task dir is at that
#     instant and cannot be a value carried from earlier in the session.
#   * with a sha argument that is NOT the recorded one it REFUSES rc 3 and names
#     both, because that is exactly the copied-stale-pin shape above.
#   * `--allow-older --reason "<why>"` proceeds anyway and prints the reason for
#     the lane log row -- a deliberate pin to an older harness is legitimate
#     (re-running an old job shape), a silent one is the defect.
#   * with NO record at all the check announces itself OFF rather than guessing;
#     a task dir that has never been synced has nothing to compare against.
if [ "${1:-}" = --write ]; then
  shift
  JR=; WSHA=; ALLOW_OLDER=0; WREASON=
  while [ $# -gt 0 ]; do
    case "$1" in
      --allow-older)   ALLOW_OLDER=1;;
      --allow-older=*) ALLOW_OLDER=1; WREASON="${1#--allow-older=}";;
      --reason)        shift; WREASON="${1:-}";;
      --reason=*)      WREASON="${1#--reason=}";;
      -*) echo "### HARNESS_COMMIT WRITE REFUSED: unknown flag '$1'" >&2; exit 2;;
      *)  if   [ -z "$JR" ];   then JR="$1"
          elif [ -z "$WSHA" ]; then WSHA="$1"
          else echo "### HARNESS_COMMIT WRITE REFUSED: too many arguments ('$1')" >&2; exit 2; fi;;
    esac
    shift
  done
  if [ -z "$JR" ]; then
    echo "### HARNESS_COMMIT WRITE REFUSED: usage: harness_commit_resolve.sh --write <job root> [<sha>] [--allow-older --reason \"<why>\"]" >&2
    exit 2
  fi
  REPO=${HARNESS_REPO:-/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools}
  W_TASK_DIR=${HARNESS_TASK_DIR:-/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11}
  W_RECORD=${HARNESS_SYNCED_RECORD:-$W_TASK_DIR/tools/.harness_synced_commit}

  # What the task copies ACTUALLY are, from the record harness_sync.sh writes
  # last and only on success.  Nothing else writes this file.
  SYNCED=; SYNCED_SHA=
  if [ -f "$W_RECORD" ]; then
    SYNCED=$(sed -e 's/#.*//' -e 's/[[:space:]]//g' "$W_RECORD" | grep -m1 . || true)
  fi
  if [ -n "$SYNCED" ]; then
    SYNCED_SHA=$(git -C "$REPO" rev-parse --verify "${SYNCED}^{commit}" 2>/dev/null || printf '%s' "$SYNCED")
  fi

  if [ -z "$WSHA" ]; then
    if [ -z "$SYNCED_SHA" ]; then
      echo "### HARNESS_COMMIT WRITE REFUSED: no sha given and no sync record at $W_RECORD" >&2
      echo "###   Nothing has been synced by harness_sync.sh, so there is no commit to" >&2
      echo "###   resolve the pin FROM. Run: harness_sync.sh <commit>, then retry." >&2
      exit 3
    fi
    WSHA=$SYNCED_SHA
    echo "### HARNESS_COMMIT resolved at submit: $WSHA source=record:$W_RECORD (DET-1-1)"
  fi

  if ! git -C "$REPO" rev-parse --verify "${WSHA}^{commit}" >/dev/null 2>&1; then
    echo "### HARNESS_COMMIT WRITE REFUSED: '$WSHA' is not a commit in $REPO" >&2
    exit 2
  fi
  WSHA_FULL=$(git -C "$REPO" rev-parse --verify "${WSHA}^{commit}")

  if [ -z "$SYNCED_SHA" ]; then
    echo "### HARNESS_COMMIT WRITE: no sync record at $W_RECORD -- the resolve-at-submit"
    echo "###   check is OFF for this write (DET-1-1). The pin is taken on trust."
  elif [ "$WSHA_FULL" != "$SYNCED_SHA" ]; then
    if [ "$ALLOW_OLDER" != 1 ] || [ -z "$WREASON" ]; then
      echo "### HARNESS_COMMIT WRITE REFUSED (rc 3): the pin is NOT what the task copies are." >&2
      echo "###   asked to pin : $WSHA_FULL" >&2
      echo "###   task dir IS  : $SYNCED_SHA   ($W_RECORD)" >&2
      echo "###   A pin copied from earlier in the session is the DET-1-1 shape: the job" >&2
      echo "###   dies at its own drift gate hours later. Drop the sha and let it resolve:" >&2
      echo "###     harness_commit_resolve.sh --write $JR" >&2
      echo "###   or say why an older harness is the right call, and it goes in the log row:" >&2
      echo "###     harness_commit_resolve.sh --write $JR $WSHA --allow-older --reason \"<why>\"" >&2
      [ "$ALLOW_OLDER" = 1 ] && [ -z "$WREASON" ] && \
        echo "###   (--allow-older WITHOUT --reason is still a refusal.)" >&2
      exit 3
    fi
    # HARNESS-SYNC-2-1: the reader refuses a stale pin too, so a deliberate
    # older pin needs a MARKER or this write would create a job that refuses
    # itself.  One sidecar line naming the very sha it authorises.
    AO_MARK="allow-older pin=$WSHA_FULL synced=$SYNCED_SHA at=$(date -Is) reason=$(printf '%s' "$WREASON" | tr '\n' ' ')"
    echo "### HARNESS_COMMIT WRITE ALLOWED-OLDER pin=$WSHA_FULL synced=$SYNCED_SHA reason=$WREASON"
    echo "###   ^ copy this line into the lane log row for this submission."
  fi

  mkdir -p "$JR" || exit 2
  printf '%s\n' "$WSHA_FULL" > "$JR/HARNESS_COMMIT" || exit 2
  if [ -n "${AO_MARK:-}" ]; then
    printf '%s\n' "$AO_MARK" > "$JR/HARNESS_COMMIT.allow-older" || exit 2
    echo "### HARNESS_COMMIT ALLOW-OLDER marker: $JR/HARNESS_COMMIT.allow-older"
    echo "###   $AO_MARK"
  else
    # An ordinary write cancels any marker an earlier submission left here.
    rm -f "$JR/HARNESS_COMMIT.allow-older"
  fi
  echo "### HARNESS_COMMIT written: $JR/HARNESS_COMMIT = $WSHA_FULL (no --export needed)"
  exit 0
fi
JOB_ROOT="${1:-}"
HC_FILE="${HARNESS_COMMIT_FILE:-}"
[ -n "$HC_FILE" ] || { [ -n "$JOB_ROOT" ] && HC_FILE=$JOB_ROOT/HARNESS_COMMIT; }

FROM_FILE=
if [ -n "$HC_FILE" ] && [ -f "$HC_FILE" ]; then
  # First non-blank, non-comment line, whitespace stripped.  A file written by
  # `echo` carries a newline and a file written by a here-doc may carry a
  # comment; neither may become part of a commit-ish.
  FROM_FILE=$(sed -e 's/#.*//' -e 's/[[:space:]]//g' "$HC_FILE" | grep -m1 . || true)
fi
FROM_ENV="${HARNESS_COMMIT:-}"

if [ -n "$FROM_FILE" ] && [ -n "$FROM_ENV" ] && [ "$FROM_FILE" != "$FROM_ENV" ]; then
  echo "### HARNESS_COMMIT REFUSED: $HC_FILE says '$FROM_FILE' and the environment says '$FROM_ENV'" >&2
  echo "###   Two pins, no rule for choosing between them. Delete one and resubmit." >&2
  exit 2
fi


# ── HARNESS-SYNC-2-1: THE READER REFUSES A STALE PIN TOO, BEFORE THE DRIFT ───
# CHECK EVER RUNS.  The writer's rc-3 refusal only fires when a lane PASSES a
# sha; a lane that copies the pin FILE (a `cp`, an sbatch that writes it inline,
# a job root cloned from an older one) bypasses the writer entirely and the job
# then dies at its drift gate three to six seconds later with a table of files
# and no statement of the actual fault.  The pin and the record are two strings
# and comparing them costs nothing, so the eight-refusal shape becomes ONE line
# that names both shas and the command that fixes it.
#
# THE MARKER, and why it is a SIDECAR and not a second line in the pin file.
# `--allow-older --reason "<why>"` is a legitimate deliberate pin to an older
# harness, and the reader must not undo the writer's decision -- so the writer
# leaves `<pin file>.allow-older`, one line, naming the very sha it authorises:
#
#   allow-older pin=<full sha> synced=<sha at write time> at=<iso> reason=<why>
#
# It is honoured ONLY when its `pin=` equals the sha resolved here, so a marker
# left behind by an earlier submission cannot authorise a different pin, and the
# writer deletes it on any ordinary write.  A SIDECAR rather than a second line
# because the pin file stays exactly one sha: every existing consumer -- and a
# bare `cat <job root>/HARNESS_COMMIT` is one -- is unchanged, and a lane that
# copies only the pin file leaves the marker behind and is REFUSED, which is the
# right answer for a copied pin.
PIN="${FROM_FILE:-$FROM_ENV}"
if [ -n "$PIN" ]; then
  R_REPO=${HARNESS_REPO:-/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools}
  R_TASK_DIR=${HARNESS_TASK_DIR:-/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11}
  R_RECORD=${HARNESS_SYNCED_RECORD:-$R_TASK_DIR/tools/.harness_synced_commit}
  R_SYNCED=
  if [ -f "$R_RECORD" ]; then
    R_SYNCED=$(sed -e 's/#.*//' -e 's/[[:space:]]//g' "$R_RECORD" | grep -m1 . || true)
  fi
  if [ -z "$R_SYNCED" ]; then
    # No record = nothing to compare against.  Announced, never silent (law 9).
    echo "### HARNESS_COMMIT stale-pin check OFF: no sync record at $R_RECORD (HARNESS-SYNC-2-1)" >&2
  else
    R_SYNCED_SHA=$(git -C "$R_REPO" rev-parse --verify "${R_SYNCED}^{commit}" 2>/dev/null || printf '%s' "$R_SYNCED")
    R_PIN_SHA=$(git -C "$R_REPO" rev-parse --verify "${PIN}^{commit}" 2>/dev/null || printf '%s' "$PIN")
    if [ "$R_PIN_SHA" = "$R_SYNCED_SHA" ]; then
      echo "### HARNESS_COMMIT pin matches the sync record $R_RECORD (HARNESS-SYNC-2-1)" >&2
    else
      R_MARK=
      if [ -n "$FROM_FILE" ] && [ -n "$HC_FILE" ] && [ -f "$HC_FILE.allow-older" ]; then
        # Read the marker's `pin=` as a FIELD, not by pattern-matching the whole
        # line: a `reason=` string is free text and could otherwise contain a
        # `pin=<sha>` of its own and authorise itself.
        R_MARK_PIN=$(sed -n 's/^allow-older[[:space:]]\{1,\}pin=\([^[:space:]]\{1,\}\).*/\1/p' "$HC_FILE.allow-older" | head -1)
        if [ -n "$R_MARK_PIN" ] && [ "$R_MARK_PIN" = "$R_PIN_SHA" ]; then
          R_MARK=$(grep -m1 '^allow-older[[:space:]]' "$HC_FILE.allow-older" || true)
        fi
      fi
      if [ -n "$R_MARK" ]; then
        echo "### HARNESS_COMMIT ALLOW-OLDER honoured pin=$R_PIN_SHA synced=$R_SYNCED_SHA -- $R_MARK" >&2
      else
        echo "### PIN STALE pin=$R_PIN_SHA synced=$R_SYNCED_SHA -- run: bash tools/harness_commit_resolve.sh --write ${JOB_ROOT:-<job root>}" >&2
        echo "###   The pin this job would run under is NOT what the task copies are." >&2
        echo "###   pin    : $R_PIN_SHA   (${HC_FILE:-<export>})" >&2
        echo "###   task IS: $R_SYNCED_SHA   ($R_RECORD)" >&2
        echo "###   Refusing HERE, in the job's first seconds, instead of at the drift" >&2
        echo "###   gate with a table of files (HARNESS-SYNC-2-1). A deliberate older" >&2
        echo "###   pin is written with --allow-older --reason \"<why>\", which leaves the" >&2
        echo "###   marker this reader honours." >&2
        exit 4
      fi
    fi
  fi
fi
if [ -n "$FROM_FILE" ]; then
  echo "### HARNESS_COMMIT=$FROM_FILE source=file:$HC_FILE (the job owns this file; no --export needed)" >&2
  printf '%s\n' "$FROM_FILE"
  exit 0
fi
if [ -n "$FROM_ENV" ]; then
  echo "### HARNESS_COMMIT=$FROM_ENV source=export (no $HC_FILE; --export is the FALLBACK path -- MERGE-N-2)" >&2
  printf '%s\n' "$FROM_ENV"
  exit 0
fi
echo "### HARNESS_COMMIT unset: no ${HC_FILE:-<job root>/HARNESS_COMMIT} and nothing exported" >&2
exit 0
