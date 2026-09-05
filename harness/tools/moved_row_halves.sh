#!/usr/bin/env bash
# moved_row_halves.sh -- classify every MOVED package row of a pixi lock pair as
# CONDA or PyPI, PER ENVIRONMENT.  MERGE-K-1.
#
# WHY THIS EXISTS, and it is a landing criterion that could not be checked.
# HANDOFF section 0(4) (tick 439) refuses a landing whose moved list carries a
# pypi row, and section 2's "11:11 09-05 RULE (steward, MERGE-K)" says such a
# row is never excluded by name -- it must be attributed by a same-generation
# control.  Both rules need to know WHICH HALF each moved row lives in, and
# until this file nothing could say.  The reader every merge lane used printed
# a census of the WHOLE produced lock:
#
#     package=anyio  conda_env_rows=14  pypi_env_rows=1
#
# which counts occurrences, not moves.  On MERGE-K's own B17 list -- 13 moves,
# all of them `anyio 4.15.0 -> 4.15.1` -- that line is true and useless: twelve
# of the thirteen moves were conda and ONE, in `pm-newton-gpu`, was a
# files.pythonhosted.org WHEEL.  A lane reading the census alone would have
# called the list conda-only and landed through the criterion.  MERGE-K found it
# by walking both locks by hand; this is that walk, versioned.
#
#   usage: moved_row_halves.sh <baseline lock> <new lock>
#
#   rc 0  no moved row is pypi  (the tick-439 shape)
#   rc 1  at least one moved row IS pypi -- each one printed, count in the
#         summary.  NOT an error: it is the signal the steward's rule says must
#         then be attributed by a control.  A caller that treats rc 1 as a
#         crash has misread it; a caller that ignores it has skipped the gate.
#   rc 2  a file is missing or unreadable, or a lock declares no environments
#         (SETUP FAILURE, never a verdict -- the p4l cert_verdict.sh convention)
#
# IT IS A SECOND, INDEPENDENT IMPLEMENTATION OF THE MOVED SET, and that is
# deliberate: `env_version_delta.py` computes the same thing in python from the
# same two files, so the two totals must agree and a disagreement is a finding
# about one of them.  This one is awk so it runs where python is banned.
#
# THE PARSE, stated because a wrong split here is a silent wrong verdict:
#   conda  <name>-<version>-<build>.conda | .tar.bz2  -> strip ext, then the
#          LAST two dash fields are build and version and the rest is the name
#          (conda names contain dashes: `font-ttf-dejavu-sans-mono`).
#   pypi   <name>-<version>-<pytag>-<abi>-<plat>.whl  -> the FIRST two dash
#          fields are name and version (wheel filenames escape dashes in both).
#          <name>-<version>.tar.gz | .zip             -> strip ext, rsplit once.
#   Any `#fragment` or `?query` is cut first, and a `direct+` URL prefix is
#   ignored -- the jetson env carries one (`torch-2.5.0a0%2B…nv24.08…whl`).
set -uo pipefail
BASE="${1:?usage: moved_row_halves.sh <baseline lock> <new lock>}"
NEW="${2:?usage: moved_row_halves.sh <baseline lock> <new lock>}"
for f in "$BASE" "$NEW"; do
  [ -r "$f" ] || { echo "MOVED-HALVES FATAL: cannot read $f" >&2; exit 2; }
done

# one lock -> "env<TAB>name<TAB>half<TAB>version", one row per distinct triple
extract() {
  awk '
    function basename(u,   n, a) { n = split(u, a, "/"); return a[n] }
    function strip(u,   s) { s = u; sub(/[#?].*$/, "", s); sub(/^direct\+/, "", s); return s }
    function conda_nv(b,   s, n, a, i, name, ver) {
      s = b; sub(/\.conda$/, "", s); sub(/\.tar\.bz2$/, "", s)
      n = split(s, a, "-"); if (n < 3) return ""
      ver = a[n-1]; name = a[1]
      for (i = 2; i <= n-2; i++) name = name "-" a[i]
      return name "\t" ver
    }
    function pypi_nv(b,   s, n, a, i, name, ver) {
      if (b ~ /\.whl$/) { n = split(b, a, "-"); if (n < 2) return ""; return a[1] "\t" a[2] }
      s = b; sub(/\.tar\.gz$/, "", s); sub(/\.zip$/, "", s); sub(/\.tar\.xz$/, "", s)
      n = split(s, a, "-"); if (n < 2) return ""
      ver = a[n]; name = a[1]
      for (i = 2; i <= n-1; i++) name = name "-" a[i]
      return name "\t" ver
    }
    /^environments:$/            { inenv = 1; next }
    inenv && /^[a-z]/            { inenv = 0 }
    inenv && /^  [A-Za-z0-9_][A-Za-z0-9._-]*:$/ { env = $1; sub(":", "", env); next }
    inenv && env != "" && /^      - (conda|pypi): / {
      half = ($0 ~ /- conda: /) ? "conda" : "pypi"
      url = $3
      nv = (half == "conda") ? conda_nv(basename(strip(url))) : pypi_nv(basename(strip(url)))
      if (nv == "") next
      print env "\t" nv "\t" half
    }
  ' "$1" | awk -F'\t' '{ print $1 "\t" $2 "\t" $4 "\t" $3 }' | sort -u
}

B=$(mktemp); N=$(mktemp); trap 'rm -f "$B" "$N"' EXIT
extract "$BASE" > "$B"
extract "$NEW"  > "$N"
bn=$(wc -l < "$B"); nn=$(wc -l < "$N")
echo "### MOVED-HALVES baseline=$BASE rows=$bn"
echo "### MOVED-HALVES new=$NEW rows=$nn"
[ "$bn" -gt 0 ] && [ "$nn" -gt 0 ] || { echo "MOVED-HALVES FATAL: a lock declares no environment package rows" >&2; exit 2; }

# join on (env, name, half); a version that differs is a MOVE
awk -F'\t' '
  NR == FNR { k = $1 SUBSEP $2 SUBSEP $3; b[k] = $4; next }
  {
    k = $1 SUBSEP $2 SUBSEP $3
    if (!(k in b)) next                 # appeared: a count change, not this readers job
    if (b[k] == $4) next
    printf "  MOVED env=%s package=%s half=%s %s -> %s\n", $1, $2, $3, b[k], $4
    total++; if ($3 == "pypi") pypi++; else conda++
  }
  END {
    printf "### MOVED-HALVES SUMMARY moved=%d conda=%d pypi=%d\n", total+0, conda+0, pypi+0
    if (pypi+0 > 0) printf "### MOVED-HALVES PYPI ROWS PRESENT -- tick-439 refuses this list until a same-generation control attributes them (HANDOFF section 2, 11:11 09-05 RULE)\n"
    else            printf "### MOVED-HALVES NO PYPI ROWS -- the conda-only shape tick-439 accepts\n"
    exit (pypi+0 > 0) ? 1 : 0
  }
' "$B" "$N"
