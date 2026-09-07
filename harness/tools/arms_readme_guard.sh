#!/usr/bin/env bash
# arms_readme_guard.sh -- the reader `harness/arms/README.md`'s table never had.
#
# WHY. The table used to carry a bare `md5` column. `grep -rn '7dfa4383'` over
# the repo found NO consumer, and the value had already stopped matching the
# tracked file -- it was the md5 of the blob at commit 6cbf814, which is a
# perfectly good fact that nothing said and nothing checked. CLAUDE.md law 2:
# a stamped field with no reader is the same defect as a gate criterion with no
# producer. So the column became TWO columns that can be checked against each
# other -- the commit an arm was submitted at, and the md5 of THAT blob -- and
# this is the check.
#
#   usage: arms_readme_guard.sh [--refresh]
#
# Default: read-only. For every row it re-extracts `<commit>:harness/arms/<file>`
# and refuses if the blob is missing or its md5 is not the recorded one. It also
# refuses when a file in arms/ has NO row (a missing row is how a table stops
# describing the directory) and when a row names a file that is gone.
#
#   --refresh  rewrites each row's md5 from the commit it already names. It does
#              NOT invent commits and it does NOT re-point a row at the tip:
#              re-anchoring an arm to a newer submission is an editorial act
#              with a job id behind it, not a refresh.
#
# rc 0 all rows check out; rc 1 a row is wrong (or a file has none); rc 2 the
# guard could not run at all. Every refusal names the row.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=${HARNESS_REPO:-}
[ -n "$REPO" ] || REPO=$(cd -- "$HERE/../.." && pwd)
README=${ARMS_README:-$REPO/harness/arms/README.md}
ARMS_DIR=$(dirname -- "$README")
MODE=check
[ "${1:-}" = "--refresh" ] && MODE=refresh
[ -f "$README" ] || { echo "GUARD FATAL: no arms README at $README"; exit 2; }
git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || { echo "GUARD FATAL: $REPO is not a git repo"; exit 2; }
grep -q '^<!-- ARMS TABLE BEGIN' "$README" && grep -q '^<!-- ARMS TABLE END' "$README" \
  || { echo "GUARD FATAL: $README has no ARMS TABLE BEGIN/END markers -- nothing to read"; exit 2; }

W=$(mktemp -d "${TMPDIR:-/tmp}/armsreadme.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
fail=0; rows=0; seen=""
say() { printf 'GUARD: %s\n' "$*"; }

# The rows, between the markers, skipping the header and separator lines.
awk '/^<!-- ARMS TABLE BEGIN/{t=1; next} /^<!-- ARMS TABLE END/{t=0} t' "$README" \
  | grep '^| `' > "$W/rows" || true
[ -s "$W/rows" ] || { say "REFUSED: the table between the markers has no rows"; exit 1; }

: > "$W/newrows"
while IFS= read -r row; do
  rows=$((rows + 1))
  FILE=$(printf '%s\n' "$row" | awk -F'|' '{print $2}' | tr -d ' `')
  COMMIT=$(printf '%s\n' "$row" | awk -F'|' '{print $3}' | tr -d ' `')
  MD5=$(printf '%s\n' "$row" | awk -F'|' '{print $4}' | tr -d ' `')
  seen="$seen $FILE"
  if [ ! -f "$ARMS_DIR/$FILE" ]; then
    say "FAIL row names \`$FILE\`, which is not in $ARMS_DIR -- a row for a file that is gone"
    fail=1; printf '%s\n' "$row" >> "$W/newrows"; continue
  fi
  if ! git -C "$REPO" cat-file blob "$COMMIT:harness/arms/$FILE" > "$W/blob" 2>/dev/null; then
    say "FAIL \`$FILE\`: no blob at $COMMIT:harness/arms/$FILE -- the commit this row names does not carry the file"
    fail=1; printf '%s\n' "$row" >> "$W/newrows"; continue
  fi
  GOT=$(md5sum "$W/blob" | awk '{print $1}')
  if [ "$GOT" = "$MD5" ]; then
    say "OK   \`$FILE\` $COMMIT blob md5=$GOT"
    printf '%s\n' "$row" >> "$W/newrows"
  elif [ "$MODE" = refresh ]; then
    say "REFRESH \`$FILE\` $COMMIT md5 $MD5 -> $GOT"
    printf '%s\n' "$row" | awk -v m="$GOT" -F'|' 'BEGIN{OFS="|"} {$4=" `" m "` "; print}' >> "$W/newrows"
  else
    say "FAIL \`$FILE\`: recorded md5 $MD5 but $COMMIT:harness/arms/$FILE is $GOT"
    fail=1; printf '%s\n' "$row" >> "$W/newrows"
  fi
done < "$W/rows"

# A file with no row: the table stops describing the directory the moment an arm
# is added without one, and nothing else would ever say so.
for f in "$ARMS_DIR"/*.sh; do
  [ -e "$f" ] || continue
  b=$(basename -- "$f")
  case " $seen " in *" $b "*) ;; *) say "FAIL \`$b\` is in arms/ and has NO row in the table"; fail=1 ;; esac
done

if [ "$MODE" = refresh ]; then
  awk -v rowfile="$W/newrows" '
    /^<!-- ARMS TABLE BEGIN/{print; t=1; next}
    /^<!-- ARMS TABLE END/{t=0}
    t && /^\| `/ { if (!done) { while ((getline l < rowfile) > 0) print l; done=1 } next }
    {print}' "$README" > "$W/README.new"
  cp "$W/README.new" "$README"
  say "REFRESHED $README ($rows row(s))"
fi

say "arms README: rows=$rows fail=$fail"
[ "$fail" = 0 ] || exit 1
exit 0
