#!/usr/bin/env bash
# moved_rows_match.sh -- count the MOVED ROWS whose PACKAGE NAME matches a
# pattern in `env_version_delta.py` output.  MERGE-M-3.
#
# WHY THIS EXISTS.  Every merge lane since B15 printed this, inherited from
# mergeB17/B19's analyze.sh:
#
#   echo "### isaac rows anywhere in the moved list vs the control:"
#   python3 env_version_delta.py "$CTL" "$NEW" \
#     | sed -n '/FULL PER-ENV VERSION DRIFT/,$p' | grep -icE 'isaac'
#
# and on MERGE-M's B20 it printed **2 on a moved list that was EMPTY**.  The
# grep runs over the WHOLE drift block, which is mostly the per-ENVIRONMENT
# header lines, and two of this workspace's environments are named
# `pm-isaaclab` and `isaaclab-gpu-latest`.  The number every lane read as
# "isaacsim moved" was the count of environments whose NAME contains the string.
#
# A row in that block is one of exactly three shapes, and only the second is a
# moved row:
#     "  <env>                    moved=N  (baseline pkgs=N new pkgs=N)"
#     "      <package>            <old>              -> <new>"
#     "      ... N more"
# The moved rows are the six-space-indented ones carrying " -> ", and the match
# is applied to the PACKAGE FIELD alone -- not to the versions either, because a
# version string can carry anything.
#
# IT ALSO REPORTS TRUNCATION, which the old one-liner could not.
# `env_version_delta.py` prints at most 40 rows per environment and then
# "... N more".  When that line is present the count is a LOWER BOUND, and this
# reader says so instead of letting a lane read a floor as a total.
#
#   usage: moved_rows_match.sh <extended-regex> [file]      (stdin if no file)
#          moved_rows_match.sh 'isaac' drift.txt
#
#   rc 0  the count was printed (whatever it is -- this reader counts, it does
#         not judge; the criterion belongs to the caller)
#   rc 2  the input carried no `FULL PER-ENV VERSION DRIFT` block at all, so
#         there was nothing to count (SETUP FAILURE, never a verdict)
set -uo pipefail
PAT="${1:?usage: moved_rows_match.sh <extended-regex> [file]}"
SRC="${2:--}"

IN=$(mktemp); trap 'rm -f "$IN"' EXIT
if [ "$SRC" = "-" ]; then cat > "$IN"; else
  [ -r "$SRC" ] || { echo "MOVED-ROWS-MATCH FATAL: cannot read $SRC" >&2; exit 2; }
  cat -- "$SRC" > "$IN"
fi
grep -q 'FULL PER-ENV VERSION DRIFT' "$IN" || {
  echo "MOVED-ROWS-MATCH FATAL: no 'FULL PER-ENV VERSION DRIFT' block in the input" >&2; exit 2; }

sed -n '/FULL PER-ENV VERSION DRIFT/,$p' "$IN" \
| awk -v pat="$PAT" '
    # a moved row: exactly six leading spaces, a package field, then " -> "
    /^      / && / -> / {
      rows++
      line = $0
      sub(/^      /, "", line)
      pkg = line
      sub(/[ \t].*$/, "", pkg)
      if (tolower(pkg) ~ tolower(pat)) { hits++; matched[hits] = $0 }
      next
    }
    /^      \.\.\. [0-9]+ more$/ { split($0, m, " "); truncated += m[2]; truncenvs++ }
    END {
      printf "### MOVED ROWS MATCHING /%s/ (package field only): %d of %d moved rows\n", pat, hits+0, rows+0
      for (i = 1; i <= hits+0; i++) print matched[i]
      if (truncenvs+0 > 0)
        printf "### MOVED-ROWS-MATCH WARNING: env_version_delta.py truncated %d environment(s), hiding %d further moved rows -- this count is a LOWER BOUND\n", truncenvs, truncated
    }
'
