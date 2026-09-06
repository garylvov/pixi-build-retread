#!/usr/bin/env bash
# arm_lock_sha_count.sh -- count the DISTINCT lock shas of a NAMED set of proof
# arms, and name them in the same breath.  L3-1b-7.
#
# WHY THIS EXISTS.  Every two-arm/three-arm proof harness on this campaign
# (`l3_twoarm.sh`, `l31_twoarm.sh`, `l31b_tworun.sh`, `c32_coldprofile.sh`,
# `c181_twoarm.sh` -- five copies of the same line) ended its identity block
# with:
#
#   NSHA=$(awk '… if(kv["shims"]=="0" && kv["lock_rc"]=="0") print kv["lock_sha"]' rows | sort -u | wc -l)
#   echo "  distinct CANONICAL-arm lock shas (arms 1 and 2): $NSHA"
#
# The filter is `shims == 0`, which is NOT "arms 1 and 2".  In `l31b-proof`
# 5919981 all three arms carried `shims=0`, so the counter counted arm 3 -- the
# control, whose manifest differs BY DESIGN -- and printed **2 directly beneath
# two lines showing arms 1 and 2 with the byte-identical sha
# `b7dba246…` at 2 732 160 bytes each**.  In `l31b1-proof` 5933357 it printed
# **3** while still naming two arms.  The block's own header calls a difference
# between arms 1 and 2 "a CACHE-STATE effect … evidence in its own right", so a
# criterion meant to be read as a finding was being computed over the wrong set:
# it can never print 1, and on a GREEN pair it reports against the result.
#
# The fix is not a better filter, it is that THE COUNTER COUNTS EXACTLY THE ARMS
# IT NAMES.  The arms are arguments; the label is derived from them; an arm that
# is missing from the rows file, or whose `lock_rc` is not 0, is REFUSED rather
# than quietly dropped, because a silently smaller set is how this defect read
# as a number in the first place.
#
#   usage: arm_lock_sha_count.sh <rows file> <arm> [<arm> ...]
#          arm_lock_sha_count.sh "$A/L31B-$J.rows.txt" 1 2
#
# The rows file is the harness's own per-arm row file: one line per arm, of
# `key=value` fields separated by spaces, carrying at least `arm=`, `lock_rc=`
# and `lock_sha=` (`label=` and `lock_bytes=` are printed when present).
#
#   rc 0  the count was printed
#   rc 2  a named arm is absent from the rows file, appears twice, or did not
#         lock (`lock_rc != 0`), or the file is unreadable -- SETUP FAILURE,
#         never a verdict.  The message names the arm.
set -uo pipefail
ROWS="${1:?usage: arm_lock_sha_count.sh <rows file> <arm> [<arm> ...]}"
shift
[ "$#" -ge 1 ] || { echo "ARM-SHA FATAL: name at least one arm" >&2; exit 2; }
[ -r "$ROWS" ] || { echo "ARM-SHA FATAL: cannot read $ROWS" >&2; exit 2; }

ARMS="$*"
awk -v armlist="$ARMS" '
  function kv_of(line,   i, n, f, p) {
    delete kv
    n = split(line, f, /[ \t]+/)
    for (i = 1; i <= n; i++) { if (split(f[i], p, "=") == 2) kv[p[1]] = p[2] }
  }
  BEGIN { nw = split(armlist, want, /[ \t]+/); label = want[1]
          for (i = 2; i <= nw; i++) label = label "," want[i] }
  {
    kv_of($0)
    if (!("arm" in kv)) next
    a = kv["arm"]
    seen[a]++
    rc[a] = ("lock_rc" in kv) ? kv["lock_rc"] : "MISSING"
    sha[a] = ("lock_sha" in kv) ? kv["lock_sha"] : "MISSING"
    bytes[a] = ("lock_bytes" in kv) ? kv["lock_bytes"] : "-"
    lab[a] = ("label" in kv) ? kv["label"] : "-"
  }
  END {
    bad = 0
    for (i = 1; i <= nw; i++) {
      a = want[i]
      if (!(a in seen))    { printf "ARM-SHA FATAL: arm %s is not in the rows file\n", a > "/dev/stderr"; bad = 1; continue }
      if (seen[a] > 1)     { printf "ARM-SHA FATAL: arm %s appears %d times in the rows file\n", a, seen[a] > "/dev/stderr"; bad = 1; continue }
      if (rc[a] != "0")    { printf "ARM-SHA FATAL: arm %s has lock_rc=%s, so it has no lock sha to count\n", a, rc[a] > "/dev/stderr"; bad = 1; continue }
      if (sha[a] == "MISSING" || sha[a] == "") { printf "ARM-SHA FATAL: arm %s carries no lock_sha\n", a > "/dev/stderr"; bad = 1; continue }
    }
    if (bad) exit 2
    for (i = 1; i <= nw; i++) {
      a = want[i]
      printf "  arm=%s %-16s sha=%s bytes=%s\n", a, lab[a], sha[a], bytes[a]
      if (!(sha[a] in distinct)) { distinct[sha[a]] = 1; nd++ }
    }
    printf "  distinct lock shas (arms %s): %d  (counted over exactly those %d arm(s), no other arm is in the set)\n", label, nd, nw
    exit 0
  }
' "$ROWS"
