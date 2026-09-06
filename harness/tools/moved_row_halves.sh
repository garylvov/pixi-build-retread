#!/usr/bin/env bash
# moved_row_halves.sh -- read the WHOLE package-set delta of a pixi lock pair,
# PER ENVIRONMENT, and classify every changed row as CONDA or PyPI.  MERGE-K-1,
# widened by MERGE-M-4.
#
# WHY THIS EXISTS, and it is a landing criterion that could not be checked.
# HANDOFF section 0(4) (tick 439) refuses a landing whose moved list carries a
# pypi row OR whose per-env package COUNT changes, and section 2's "11:11 09-05
# RULE (steward, MERGE-K)" says such a row is never excluded by name -- it must
# be attributed by a same-generation control.  Both rules need to know WHICH
# HALF each changed row lives in, and until this file nothing could say.  The
# reader every merge lane used printed a census of the WHOLE produced lock:
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
# MERGE-M-4, AND IT IS WHY THIS FILE WAS REWRITTEN RATHER THAN EXTENDED.  The
# first version joined the two locks on (env, package, half) and SKIPPED any key
# missing from one side -- `if (!(k in b)) next`, commented "appeared: a count
# change, not this reader's job".  A REMOVED package has no new half to put in a
# column, so it was invisible: on MERGE-M's B21 candidate, a lock that had LOST
# `exceptiongroup` from FOUR environments, this reader printed
# `moved=1 conda=1 pypi=0` and EXITED 0 -- the conda-only shape tick 439
# accepts.  Its own row counts showed the loss plainly (11304 -> 11300) and its
# summary did not.  It had been the second opinion on every landing since B17.
# A removal and an addition are now first-class HALVES of their own -- `old -> -`
# and `- -> new` -- counted, printed, and REFUSED.
#
#   usage: moved_row_halves.sh <baseline lock> <new lock>
#
#   rc 0  nothing this criterion refuses: no pypi row anywhere in the changed
#         list and no environment's package count moved.  (Version moves inside
#         the conda half with counts held are the shape tick 439 accepts.)
#   rc 1  REFUSED -- at least one of: a changed row is pypi; an environment's
#         package count changed (a removal, an addition, or an environment that
#         exists on only one side).  Every such row is printed and the reasons
#         are named on the `REFUSE` line.  NOT an error: it is the signal the
#         steward's rule says must then be attributed by a control.  A caller
#         that treats rc 1 as a crash has misread it; a caller that ignores it
#         has skipped the gate.
#   rc 2  a file is missing or unreadable, or a lock declares no environments
#         (SETUP FAILURE, never a verdict -- the p4l cert_verdict.sh convention)
#
# IT IS A SECOND, INDEPENDENT IMPLEMENTATION OF THE CHANGED SET, and that is
# deliberate: `env_version_delta.py` computes the same thing in python from the
# same two files, so the two totals must agree and a disagreement is a finding
# about one of them.  This one is awk so it runs where python is banned.
#
# AGREEING WITH IT IS A PROPERTY OF THE KEY, NOT OF THE PARSE, and the key is
# therefore copied on purpose while the parse is deliberately not:
# `env_version_delta.py` keys a package by (env, NORMALIZED NAME) alone -- name
# lowercased with `_` folded to `-`, half not in the key, LAST row in file order
# winning when one name is carried by two urls in one env -- and emits one row
# per key whose version differs, counting a vanished package as `old -> -` and a
# new one as `- -> new`.  This file keys and folds identically, so its
# `TOTAL rows` is directly comparable with that script's
# `### total moved rows across all envs`.  Anything else would make the second
# opinion unable to disagree meaningfully.
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

# one lock -> "env<TAB>name<TAB>version<TAB>half", IN FILE ORDER (the order is
# load-bearing: the last url for a name inside one env is the one that counts,
# which is what the python reader's dict assignment does).
extract() {
  awk '
    function basename(u,   n, a) { n = split(u, a, "/"); return a[n] }
    function strip(u,   s) { s = u; sub(/[#?].*$/, "", s); sub(/^direct\+/, "", s); return s }
    function norm(n,   s) { s = tolower(n); gsub(/_/, "-", s); return s }
    function conda_nv(b,   s, n, a, i, name, ver) {
      s = b; sub(/\.conda$/, "", s); sub(/\.tar\.bz2$/, "", s)
      n = split(s, a, "-"); if (n < 3) return ""
      ver = a[n-1]; name = a[1]
      for (i = 2; i <= n-2; i++) name = name "-" a[i]
      return norm(name) "\t" ver
    }
    function pypi_nv(b,   s, n, a, i, name, ver) {
      if (b ~ /\.whl$/) { n = split(b, a, "-"); if (n < 2) return ""; return norm(a[1]) "\t" a[2] }
      s = b; sub(/\.tar\.gz$/, "", s); sub(/\.zip$/, "", s); sub(/\.tar\.xz$/, "", s)
      n = split(s, a, "-"); if (n < 2) return ""
      ver = a[n]; name = a[1]
      for (i = 2; i <= n-1; i++) name = name "-" a[i]
      return norm(name) "\t" ver
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
  ' "$1"
}

# fold one extract to ONE entry per (env, name): last url in file order wins the
# version, and a name carried by BOTH halves inside one env is labelled
# `conda+pypi` so it can never be read as pure conda.
fold() {
  awk -F'\t' '
    { k = $1 SUBSEP $2
      if (!(k in ver)) { ord[++n] = k; e[k] = $1; p[k] = $2 }
      ver[k] = $3
      if ($4 == "pypi") pypi[k] = 1; else conda[k] = 1 }
    END { for (i = 1; i <= n; i++) { k = ord[i]
        h = (conda[k] && pypi[k]) ? "conda+pypi" : (pypi[k] ? "pypi" : "conda")
        print e[k] "\t" p[k] "\t" ver[k] "\t" h } }
  ' "$1"
}

B=$(mktemp); N=$(mktemp); A=$(mktemp); trap 'rm -f "$B" "$N" "$A"' EXIT
extract "$BASE" | fold /dev/stdin > "$B"
extract "$NEW"  | fold /dev/stdin > "$N"
bn=$(wc -l < "$B"); nn=$(wc -l < "$N")
echo "### MOVED-HALVES baseline=$BASE rows=$bn"
echo "### MOVED-HALVES new=$NEW rows=$nn"
[ "$bn" -gt 0 ] && [ "$nn" -gt 0 ] || { echo "MOVED-HALVES FATAL: a lock declares no environment package rows" >&2; exit 2; }

# ONE sorted stream, both sides tagged, grouped by (env, package).  A group of
# two is a package present on both sides; a group of one is a REMOVAL or an
# ADDITION, which is precisely what the pre-MERGE-M-4 join dropped on the floor.
{ awk -F'\t' '{ print $1 "\t" $2 "\tB\t" $3 "\t" $4 }' "$B"
  awk -F'\t' '{ print $1 "\t" $2 "\tN\t" $3 "\t" $4 }' "$N"
} | LC_ALL=C sort -t"$(printf '\t')" -k1,1 -k2,2 -k3,3 > "$A"

LC_ALL=C awk -F'\t' '
  function flush(   h, kind, old, new) {
    if (key == "") return
    if (haveb && haven) {
      if (bver == nver) return
      h = (bhalf == nhalf) ? bhalf : "conda+pypi"
      kind = "MOVED  "; old = bver; new = nver; emoved[env]++
    } else if (haveb) {
      h = bhalf; kind = "REMOVED"; old = bver; new = "-"; eremoved[env]++
    } else {
      h = nhalf; kind = "ADDED  "; old = "-"; new = nver; eadded[env]++
    }
    rows++
    if (h ~ /pypi/) apypi++; else aconda++
    if (kind == "MOVED  ") { if (h ~ /pypi/) mpypi++; else mconda++ }
    printf "  %s env=%s package=%s half=%s %s -> %s\n", kind, env, pkg, h, old, new
  }
  {
    k = $1 SUBSEP $2
    if (k != key) { flush(); key = k; env = $1; pkg = $2; haveb = 0; haven = 0 }
    if ($3 == "B") { haveb = 1; bver = $4; bhalf = $5; benv[$1]++ }
    else           { haven = 1; nver = $4; nhalf = $5; nenv[$1]++ }
    seenenv[$1] = 1
  }
  END {
    flush()
    ne = 0
    for (e in seenenv) { ne++; elist[ne] = e }
    for (i = 2; i <= ne; i++) { v = elist[i]; j = i-1
      while (j >= 1 && elist[j] > v) { elist[j+1] = elist[j]; j-- }
      elist[j+1] = v }
    print "### MOVED-HALVES PER-ENV COUNTS (base_pkgs/new_pkgs are distinct (env,package) keys, directly comparable with env_version_delta.py)"
    for (i = 1; i <= ne; i++) {
      e = elist[i]
      bp = benv[e]+0; np = nenv[e]+0; d = np - bp
      if (d != 0) countenvs++
      printf "  ENV %-26s base_pkgs=%-5d new_pkgs=%-5d delta=%+d  moved=%d removed=%d added=%d\n", \
        e, bp, np, d, emoved[e]+0, eremoved[e]+0, eadded[e]+0
      tmoved += emoved[e]+0; tremoved += eremoved[e]+0; tadded += eadded[e]+0
    }
    printf "### MOVED-HALVES SUMMARY moved=%d conda=%d pypi=%d removed=%d added=%d\n", \
      tmoved+0, mconda+0, mpypi+0, tremoved+0, tadded+0
    printf "### MOVED-HALVES HALVES OVER ALL CHANGED ROWS rows=%d conda=%d pypi=%d  (a REMOVED or an ADDED pypi package is a pypi row too, and the refusal below reads THIS line)\n", \
      rows+0, aconda+0, apypi+0
    printf "### MOVED-HALVES TOTAL rows=%d envs=%d count_changed_envs=%d  (rows must equal env_version_delta.py \"total moved rows across all envs\")\n", \
      rows+0, ne, countenvs+0
    reasons = ""
    if (apypi+0 > 0)     reasons = "pypi_rows"
    if (countenvs+0 > 0) reasons = (reasons == "" ? "" : reasons ",") "package_count_change"
    if (reasons != "") {
      printf "### MOVED-HALVES REFUSE reasons=%s -- tick-439 refuses this list until a same-generation control attributes it (HANDOFF section 2, 11:11 09-05 RULE)\n", reasons
      exit 1
    }
    print "### MOVED-HALVES CLEAN no pypi row and no package-count change -- the shape tick-439 accepts"
    exit 0
  }
' "$A"
