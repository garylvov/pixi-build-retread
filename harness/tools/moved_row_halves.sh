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
#         exists on only one side); or (HARNESS-CONSOL-13) a conda BUILD STRING
#         changed in only SOME of the environments carrying that package, or in
#         more than one old->new pair.  Every such row is printed and the reasons
#         are named on the `REFUSE` line.  NOT an error: it is the signal the
#         steward's rule says must then be attributed by a control.  A caller
#         that treats rc 1 as a crash has misread it; a caller that ignores it
#         has skipped the gate.
#   rc 2  a file is missing or unreadable, or a lock declares no environments
#         (SETUP FAILURE, never a verdict -- the p4l cert_verdict.sh convention)
#
#   A BUILD-STRING-ONLY change with identical per-env coverage stays rc 0 -- an
#   upstream rebuild sweeping every environment is an environmental fact, not a
#   resolution change -- but it is NEVER SILENT: `### BUILD-STRING CHANGES <n>`
#   and its `### BUILD-STRING SUMMARY` row are the table HANDOFF section 2's
#   HARNESS-CONSOL-13 rule reads before landing.  Before this section a rebuild
#   was invisible to every reader in the tree, because every one of them is
#   keyed on VERSION.
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

# one lock -> "env<TAB>name<TAB>version<TAB>build<TAB>half", IN FILE ORDER (the
# BUILD column is `-` for every pypi row: a wheel has no conda build string, and
# a column that invented one would make the build-string reader below compare
# two things that are not the same kind of fact).  (the order is
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
      return norm(name) "\t" ver "\t" a[n]
    }
    function pypi_nv(b,   s, n, a, i, name, ver) {
      if (b ~ /\.whl$/) { n = split(b, a, "-"); if (n < 2) return ""; return norm(a[1]) "\t" a[2] "\t-" }
      s = b; sub(/\.tar\.gz$/, "", s); sub(/\.zip$/, "", s); sub(/\.tar\.xz$/, "", s)
      n = split(s, a, "-"); if (n < 2) return ""
      ver = a[n]; name = a[1]
      for (i = 2; i <= n-1; i++) name = name "-" a[i]
      return norm(name) "\t" ver "\t-"
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
# version AND the build string, and a name carried by BOTH halves inside one env
# is labelled `conda+pypi` so it can never be read as pure conda.
# OUT: env<TAB>name<TAB>version<TAB>half<TAB>build -- half stays in column 4 so
# every reader written against the pre-BUILD shape still reads what it read.
fold() {
  awk -F'\t' '
    { k = $1 SUBSEP $2
      if (!(k in ver)) { ord[++n] = k; e[k] = $1; p[k] = $2 }
      ver[k] = $3; bld[k] = $4
      if ($5 == "pypi") pypi[k] = 1; else conda[k] = 1 }
    END { for (i = 1; i <= n; i++) { k = ord[i]
        h = (conda[k] && pypi[k]) ? "conda+pypi" : (pypi[k] ? "pypi" : "conda")
        print e[k] "\t" p[k] "\t" ver[k] "\t" h "\t" bld[k] } }
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
{ awk -F'\t' '{ print $1 "\t" $2 "\tB\t" $3 "\t" $4 "\t" $5 }' "$B"
  awk -F'\t' '{ print $1 "\t" $2 "\tN\t" $3 "\t" $4 "\t" $5 }' "$N"
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
MRH_RC=$?

# ── THE BUILD-STRING READER (HARNESS-CONSOL-13, CRIT-1) ─────────────────────
# WHY IT EXISTS. Everything above this line is keyed on VERSION. A conda package
# rebuilt upstream keeps its version and changes only its BUILD STRING --
# `pkg-config 0.29.2 h1114479_1012` -> `h1114479_1013` between B30's landed
# proof lock (mergeB30/artifacts/pixi.lock.cert) and det162's W1 lock
# (det162-work/artifacts/pixi.lock.D17-6015646-W1) -- and to every reader above
# that pair is IDENTICAL: no move, no removal, no addition, no count change.
# `### MOVED-HALVES CLEAN` is printed and the landing criterion reads it and
# lands. The bytes in the environment are NOT the bytes that were proved, and
# nothing in the tree said so. This section is that missing half.
#
# THE CRITERION IT SERVES, and it is a rule that lands with this producer:
# HANDOFF section 2 now reads THIS table -- land only when it is EMPTY, or when
# every changed row is present in ALL the environments that carry the package,
# identically (one and the same old->new pair). A rebuild that swept every
# environment at once is an environmental fact about the channel; a rebuild that
# reached SOME environments is a RESOLUTION difference wearing a rebuild's
# clothes, and those are not the same event.
#
# THE rc CONTRACT, stated because it is deliberately NOT symmetric with the
# refusals above: a build-string-only change with identical per-env coverage
# leaves rc 0 -- refusing every upstream rebuild would refuse most weeks -- but
# a PARTIAL change (identical_per_env=no) is rc 1, because that is the shape
# that is not an environmental rebuild. The rc above still wins if it refused.
#
# WHAT IT DELIBERATELY DOES NOT DO: pypi rows carry no build string (the BUILD
# column is `-` for them) and a package whose VERSION moved is already the row
# walk's business, so only (env, name, version)-identical CONDA pairs are
# compared here. That keeps the two readers from double-counting one event.
BS_OUT=$(LC_ALL=C awk -F'\t' '
  function flush(   k) {
    if (key == "") return
    if (haveb && haven && bhalf == "conda" && nhalf == "conda" && bver == nver) {
      k = pkg SUBSEP bver
      carry[k]++
      if (bbld != nbld && bbld != "-" && nbld != "-") {
        rows++
        chg[pkg SUBSEP bver SUBSEP bbld SUBSEP nbld]++
        pkgchg[k]++
        envseen[env] = 1
        line[++nl] = sprintf("  BUILD env=%s package=%s version=%s %s -> %s", env, pkg, bver, bbld, nbld)
      }
    }
  }
  {
    k = $1 SUBSEP $2
    if (k != key) { flush(); key = k; env = $1; pkg = $2; haveb = 0; haven = 0 }
    if ($3 == "B") { haveb = 1; bver = $4; bhalf = $5; bbld = $6 }
    else           { haven = 1; nver = $4; nhalf = $5; nbld = $6 }
  }
  END {
    flush()
    identical = "yes"
    for (c in chg) { split(c, f, SUBSEP); pairs[f[1] SUBSEP f[2]]++ }
    printf "### BUILD-STRING CHANGES %d\n", rows+0
    for (i = 1; i <= nl; i++) print line[i]
    for (p in pkgchg) {
      split(p, g, SUBSEP)
      pk = (pairs[p] == 1) ? "one" : "MANY"
      ident = (pkgchg[p] == carry[p] && pairs[p] == 1) ? "yes" : "no"
      if (ident == "no") identical = "no"
      roll[++nr] = sprintf("  BUILD-PKG package=%s version=%s changed_envs=%d envs_carrying=%d distinct_build_pairs=%s identical=%s", \
        g[1], g[2], pkgchg[p], carry[p], pk, ident)
    }
    for (i = 2; i <= nr; i++) { v = roll[i]; j = i-1
      while (j >= 1 && roll[j] > v) { roll[j+1] = roll[j]; j-- }
      roll[j+1] = v }
    for (i = 1; i <= nr; i++) print roll[i]
    ne = 0; for (e in envseen) ne++
    printf "### BUILD-STRING SUMMARY build_string_changed=%d envs=%d identical_per_env=%s\n", rows+0, ne, identical
    if (rows+0 == 0) {
      print "### BUILD-STRING NOTE no conda row kept its version and changed its build -- nothing for the section-2 rule to read"
      exit 0
    }
    if (identical == "yes") {
      print "### BUILD-STRING NOTE every changed row reached EVERY environment carrying the package, as one old->new pair: an upstream rebuild, which section 2 lands"
      exit 0
    }
    print "### BUILD-STRING REFUSE a build-string change that reached only SOME of the environments carrying the package is a RESOLUTION difference, not an environmental rebuild (HANDOFF section 2)"
    exit 1
  }
' "$A")
BS_RC=$?
printf '%s\n' "$BS_OUT"

# ── THE REORDER CLASSIFIER (MERGE-U-1), BACK-PORTED HERE ────────────────────
# WHERE IT CAME FROM AND WHY IT LIVES HERE NOW. Every merge lane's `analyze.sh`
# is a task-dir file copied forward from the previous lane -- mergeB16 through
# mergeB30, sixteen copies, and `grep -rn requires-dist harness/` finds NOTHING:
# the analyzer has no harness home at all, so a defect fixed in one copy is
# fixed in exactly one copy and every later lane inherits the broken one. This
# reader IS the harness file those analyzers already call (`bash
# $T/tools/moved_row_halves.sh "$CTL" "$NEW"`, four times in mergeB30/analyze.sh
# alone) and it takes the SAME two locks, so the classification lands here and
# gets a call site for free rather than needing a new file nobody invokes.
#
# THE DEFECT. B29's analyzer classified a changed line as a `requires-dist` line
# by grepping the LITERAL STRING `requires-dist` ON THE CHANGED LINE. In a
# pixi.lock `requires-dist:` is a KEY and the things that reorder under it are
# the LIST ITEMS beneath it, which never carry the key's own text. So on a
# textbook gym 0.26.2 reorder it scored `0 requires-dist / 28 NOT` and would
# have told its reader the predicted shape did not reproduce -- when all 28
# changed lines were gym's own `extra == 'all'` / `extra == 'testing'` items.
# MERGE-U measured that on B30's real candidate and fixed it lane-locally; this
# is that fix, versioned.
#
# BOTH COUNTS ARE PRINTED SIDE BY SIDE, deliberately: the defect stays VISIBLE
# rather than being silently repaired, so a reader comparing this output with
# any B29-era analyze.sh log can see why the two disagree.
#
# AND THE FIX'S OWN NARROWNESS IS PRINTED TOO, rather than left to be discovered
# the way the first one was: keying on the `extra ==` marker classifies a
# requires-dist item that carries an extra marker and MISSES one that does not
# (`- numpy>=1.18.0`). On the gym block every one of the 28 carries a marker, so
# the fix is right about the case it was measured on and no wider; the
# `no_extra_marker` count is what a later reorder somewhere else would show up
# in, and it is on the page so nobody has to rediscover the same class of miss.
echo "### MOVED-HALVES REORDER CLASSIFICATION (MERGE-U-1) baseline=$BASE new=$NEW"
MRH_RAW=$(LC_ALL=C diff "$BASE" "$NEW" | grep -cE '^[<>]')
MRH_SRT=$(LC_ALL=C diff <(LC_ALL=C sort "$BASE") <(LC_ALL=C sort "$NEW") | grep -cE '^[<>]')
MRH_D=$(mktemp); LC_ALL=C diff "$BASE" "$NEW" | grep -E '^[<>]' > "$MRH_D"
echo "  changed lines (RAW, order-sensitive)                          : $MRH_RAW"
echo "  changed lines with both sides SORTED (order-insensitive)      : $MRH_SRT"
echo "  OLD count -- changed lines carrying the LITERAL requires-dist : $(grep -cE 'requires-dist' "$MRH_D")   <-- MERGE-U-1: reads 0 on a REAL gym reorder, which is the defect"
echo "  FIXED count -- changed dependency ITEMS carrying an 'extra ==': $(grep -cE '^[<>][[:space:]]*-[[:space:]].*extra ==' "$MRH_D")"
echo "  of those, gym's own extras (all / testing)                    : $(grep -cE "^[<>][[:space:]]*-[[:space:]].*extra == .(all|testing)." "$MRH_D")"
echo "  dependency items with NO extra marker (the fix's blind spot)  : $(grep -E '^[<>][[:space:]]*-[[:space:]]' "$MRH_D" | grep -cvE 'extra ==')"
echo "  changed lines that are NOT dependency items                   : $(grep -cvE '^[<>][[:space:]]*-[[:space:]]' "$MRH_D")"
if [ "$MRH_RAW" -gt 0 ] && [ "$MRH_SRT" -eq 0 ]; then
  echo "  VERDICT: raw=$MRH_RAW sorted=0 -- THE SAME BYTES IN A DIFFERENT ORDER. No package moved,"
  echo "           no row was added or removed. This is a LEAD about emission-order stability and"
  echo "           NOT a moved row; the rc above is what tick-439 reads, and this section does not"
  echo "           change it."
elif [ "$MRH_RAW" -eq 0 ]; then
  echo "  VERDICT: raw=0 -- byte-identical. No reorder appears in this pair."
else
  echo "  VERDICT: raw=$MRH_RAW sorted=$MRH_SRT -- a SORTED delta is a RESOLUTION change, not a reorder."
fi
rm -f "$MRH_D"
# THE rc IS THE WORST OF THE TWO READERS, and the version reader wins a tie:
# a list that the row walk already refuses is refused for ITS reason, and a list
# the row walk accepts can still be refused by the build-string half alone.
if [ "$MRH_RC" -ne 0 ]; then exit "$MRH_RC"; fi
exit "$BS_RC"
