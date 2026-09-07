#!/usr/bin/env bash
# derived_region_parse_check.sh -- REFUSE a derived phase wrapper whose
# SUBSTITUTE region leaves the shell's quoting state changed.
#
#   usage: derived_region_parse_check.sh <derived script> [<end marker>]
#   rc 0  the region closes every quote it opens
#   rc 3  REFUSED: it does not, and the refusal names the line
#   rc 4  FATAL: no file, or no end marker in it
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
  exit 0
fi

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
