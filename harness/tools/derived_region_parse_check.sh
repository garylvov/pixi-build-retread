#!/usr/bin/env bash
# derived_region_parse_check.sh -- REFUSE a derived phase wrapper whose
# SUBSTITUTE region leaves the shell's quoting state changed, or whose FAST_ENV
# hop inside that region no longer reproduces the tip template's.
#
#   usage: derived_region_parse_check.sh <derived script> [<end marker>] [<tip template>]
#   rc 0  the region closes every quote it opens AND its FAST_ENV hop is the tip's
#   rc 3  REFUSED: the region does not close its quotes, and the refusal names the line
#   rc 4  FATAL: no file, no end marker in it, or no tip template to compare against
#   rc 5  REFUSED: the region carries a STALE FAST_ENV hop (DET-1-6-3b)
#
# BOTH refusals are about the SAME blind spot: the derivation gate the drivers
# run diffs the derived file against its template with this region STRIPPED, so
# nothing that lives INSIDE the region is ever compared to anything. Two things
# have shipped through that hole and cost a job each -- an unbalanced quote
# (6014197) and a stale FAST_ENV hop (6020526).
#
# ── WHAT IT WOULD HAVE CAUGHT, MEASURED, ON A JOB THAT PAID FOR IT ───────────
# det161-proof 6014197. Its SUBSTITUTE region carried
#
#     D=${2:?arg 2: this arm's job root, which must hold HARNESS_COMMIT and artifacts/}
#
# and the apostrophe in `arm's` is INSIDE a `${var:?word}` expansion, where it is
# a quote character and not a letter. Bash opened a single-quoted span at that
# line and closed it on the next apostrophe in the file -- 109 lines later, in
# `grep -E 'data\+stellex|^Name'`. Everything between was swallowed into one
# word: the backticked `` `ALL` `` in a COMMENT about `--export=ALL` was
# command-substituted (`line 282: ALL: command not found`), the whole GATES block
# was executed as one command name, and `FAST_ENV` was still unbound at line 289
# because its assignment had been eaten. The wrapper exited rc 1 with a 0-second
# arm and no lock, on BOTH arms.
#
# ── AND WHY NOTHING CAUGHT IT ────────────────────────────────────────────────
# `bash -n` ON THE WHOLE FILE PASSES. Measured on the exact broken wrapper and on
# its template: both rc 0. The stray apostrophe is re-balanced by a later one, so
# the file parses; it just parses into different commands than it reads as. The
# driver's own `bash -n "$RELOCK"` gate was therefore green on a wrapper that
# could not run, and the derivation gate that diffs the file against the template
# with this region STRIPPED cannot see inside the region by construction.
#
# ── THE CHECK, AND WHY IT IS A TRUNCATION AND NOT A PATTERN ──────────────────
# Parse the file TRUNCATED AT `### SUBSTITUTE: END`. A region that closes every
# quote it opens parses on its own; one that does not gives bash's own
# `unexpected EOF while looking for matching` naming the OPENING line. That is a
# statement about the shell's real quoting rules, so it catches an unbalanced
# `"`, `'`, backtick, `$(`, heredoc or `${` alike -- not just the apostrophe that
# happened to be the first instance. A regex for "apostrophe inside `${:?}`"
# would have been a rule fitted to one job's log.
#
# Reader: tools/derived_region_parse_check_guard.sh, whose arm D is the control
# that the SAME fixture's full-file `bash -n` passes.
set -uo pipefail

SRC=${1:?usage: derived_region_parse_check.sh <derived script> [<end marker>]}
END_MARK=${2:-'### SUBSTITUTE: END'}

[ -f "$SRC" ] || { echo "### DERIVED REGION PARSE FATAL: no file at $SRC"; exit 4; }

END_LN=$(grep -n -m1 -F -x -- "$END_MARK" "$SRC" | cut -d: -f1)
if [ -z "$END_LN" ]; then
  echo "### DERIVED REGION PARSE FATAL: $SRC has no line '$END_MARK'."
  echo "###        This check cannot say anything about a file with no region, and"
  echo "###        passing silently would make it decorative."
  echo "###        ACTUATOR: derive the wrapper from a template that carries the markers."
  exit 4
fi

TMP=$(mktemp "${TMPDIR:-/tmp}/derived-region-parse.XXXXXX.sh") || exit 4
trap 'rm -f "$TMP"' EXIT
head -n "$END_LN" "$SRC" > "$TMP"

if ERR=$(bash -n "$TMP" 2>&1); then
  echo "### DERIVED REGION PARSE: clean -- $SRC parses standalone through line $END_LN ($END_MARK)"
else
  echo "### DERIVED REGION PARSE REFUSED: $SRC does not parse when truncated at line $END_LN."
  echo "###        The SUBSTITUTE region opens a quote it never closes, so the lines"
  echo "###        BELOW the region are swallowed into it and run as something other"
  echo "###        than what they read as. \`bash -n\` on the WHOLE file does NOT catch"
  echo "###        this -- a later quote re-balances it -- which is why this check"
  echo "###        truncates. Bash's own message names the OPENING line:"
  printf '%s\n' "$ERR" | sed "s@$TMP@$SRC@g" | sed 's/^/###   /'
  echo "###        ACTUATOR: close the quote at the line named above. The usual cause"
  echo "###        is an apostrophe in prose inside a \${var:?message} expansion, where"
  echo "###        it is a quote character: write the message without the apostrophe."
  exit 3
fi

### FAST_ENV-HOP BEGIN
# ── THE SECOND BLIND SPOT OF THE SAME REGION (DET-1-6-3b, job 6020526) ───────
# The derivation gate every driver runs diffs a derived wrapper against its
# template WITH THIS REGION STRIPPED, and reports "0 diff lines". So does the
# leftover-token self-check: it strips the region by construction. A stale
#
#     FAST_ENV=$(dirname "$0")/../retread_fast_env.sh
#
# hop -- the pre-2026-09-07 form, a path that has never existed in the harness
# repo -- therefore sat INSIDE the region of det163's wrapper and was reported
# as 0 diff lines, while at run time it fell through to the TASK-dir copy. Job
# 6020526 ran a producer-less task copy for exactly that reason and nothing in
# its log said which bytes it had imported.
#
# `bash -n` cannot see it (the line parses), the stripped diff cannot see it
# (the line is inside the stripped region), and fast_env_resolution_guard.sh
# cannot see it either -- that guard reads the harness repo's own templates,
# and a derived wrapper lives in a task directory it never walks. This is the
# only reader that sees a DERIVED file's hop.
#
# THE RULE: the hop is not a pattern to match, it is a COPY to reproduce. Every
# FAST_ENV assignment and every `source`/`.` of retread_fast_env.sh in the
# derived file, comments stripped, must be BYTE-IDENTICAL to the tip template's,
# in the same order. That is the ONE resolution rule HARNESS-CONSOL-10 installed
# (fast_env_resolution_guard.sh CANON + FALLBACK) without this check having to
# restate it: whatever the tip says is the rule, and a derived file says it too
# or it is refused. A regex for the one dead path would be a rule fitted to one
# job's log, and would pass the NEXT wrong path.
TPL=${3:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../phase_template/phaseN_relock.sh}
if [ ! -f "$TPL" ]; then
  echo "### DERIVED FAST_ENV hop=FATAL: no tip template at $TPL"
  echo "###        A check that passes a file it cannot compare is decorative."
  echo "###        ACTUATOR: pass the tip template as arg 3."
  exit 4
fi

fast_env_candidates () {   # $1 = file; the hop lines, comments and indent stripped
  sed -e 's/[[:space:]]*#.*$//' -e 's/[[:space:]]*$//' -e 's/^[[:space:]]*//' -- "$1" \
  | grep -E '^(FAST_ENV[A-Za-z0-9_]*=|\[ -f "\$FAST_ENV" \][[:space:]]*\|\|[[:space:]]*FAST_ENV[A-Za-z0-9_]*=|(\.|source)[[:space:]]+[^[:space:]]*retread_fast_env\.sh)' \
  || true
}

TIP_C=$(fast_env_candidates "$TPL")
DRV_C=$(fast_env_candidates "$SRC")
TIP_MD5=$(printf '%s\n' "$TIP_C" | md5sum | awk '{print $1}')
DRV_MD5=$(printf '%s\n' "$DRV_C" | md5sum | awk '{print $1}')

if [ "$TIP_MD5" = "$DRV_MD5" ]; then
  echo "### DERIVED FAST_ENV hop=ok tip=$TIP_MD5 derived=$DRV_MD5"
  [ -n "$TIP_C" ] || echo "###        (neither file carries a FAST_ENV hop -- nothing to reproduce)"
else
  echo "### DERIVED FAST_ENV hop=STALE tip=$TIP_MD5 derived=$DRV_MD5"
  echo "###        $SRC does not reproduce the tip template's FAST_ENV resolution."
  echo "###        The stripped diff CANNOT see this: the hop is inside the"
  echo "###        SUBSTITUTE region, which that diff removes before comparing."
  echo "###        TIP ($TPL):"
  printf '%s\n' "$TIP_C" | sed 's/^/###   tip| /'
  echo "###        DERIVED ($SRC):"
  printf '%s\n' "$DRV_C" | sed 's/^/###   drv| /'
  echo "###        ACTUATOR: copy the tip lines verbatim into the derived wrapper."
  echo "###        A hop naming ../retread_fast_env.sh (no tools/) resolves to the"
  echo "###        TASK-dir copy at run time -- job 6020526's producer-less run."
  exit 5
fi
### FAST_ENV-HOP END

exit 0
