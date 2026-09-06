#!/usr/bin/env bash
# harness_commit_resolve.sh -- say which harness commit this job is pinned to,
# and say WHERE that answer came from.  MERGE-N-2.
#
#   usage: HC=$(bash harness_commit_resolve.sh "<job root>") || exit ...
#
#   stdout  the commit-ish, or nothing when there is none to find
#   stderr  one `### HARNESS_COMMIT ...` provenance row, always
#   rc 0    resolved, or deliberately unset (the caller announces the check OFF)
#   rc 2    REFUSED: the file and the export disagree
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
#   bash harness_commit_resolve.sh --write <job root> <sha>
#
# It REFUSES a sha that is not a commit in the harness repo. That check costs
# nothing at SUBMIT time and is the only moment it is cheap: the alternative is
# a three-hour relock refusing at its drift gate on a typo.
if [ "${1:-}" = --write ]; then
  JR=${2:?usage: harness_commit_resolve.sh --write <job root> <sha>}
  SHA=${3:?usage: harness_commit_resolve.sh --write <job root> <sha>}
  REPO=${HARNESS_REPO:-/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools}
  if ! git -C "$REPO" rev-parse --verify "${SHA}^{commit}" >/dev/null 2>&1; then
    echo "### HARNESS_COMMIT WRITE REFUSED: '$SHA' is not a commit in $REPO" >&2
    exit 2
  fi
  mkdir -p "$JR" || exit 2
  printf '%s\n' "$SHA" > "$JR/HARNESS_COMMIT" || exit 2
  echo "### HARNESS_COMMIT written: $JR/HARNESS_COMMIT = $SHA (no --export needed)"
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
