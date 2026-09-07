#!/usr/bin/env bash
# manifest_regen.sh -- THE generator for harness/MANIFEST.md5, and the reader
# that says the tracked file is what this generator would write.
#
#   usage: manifest_regen.sh <harness dir> --check    rc 0 in step, rc 6 drift
#          manifest_regen.sh <harness dir> --write    rewrite MANIFEST.md5
#          manifest_regen.sh <harness dir>            print the rows to stdout
#
# ── WHY THIS EXISTS (HARNESS-CONSOL-13's caveat, measured) ───────────────────
# MANIFEST.md5 had NO generator. `README.md` says it "records the md5 of every
# file in this directory as committed", and lanes kept it true by rewriting
# individual ROWS IN PLACE (tools/fixset_land_row.sh does exactly that, for the
# one row it owns). Nobody could regenerate the whole file, because the order it
# was in reproduced under NOTHING -- not `sort`, not `LC_ALL=C sort`, not `find`
# order. Measured on the file before the reorder commit: `LC_ALL=C sort -k2`
# moved rows (`README.md` and `arms/README.md` to the top, `inode-cleanup/
# census.sh` down two) while the two versions were SET-EQUAL, i.e. the old order
# was a case-insensitive locale collation nothing in this repo reproduced. The
# consequence was not cosmetic: any wholesale regeneration produced a diff of
# ~dozens of moved lines with no content change, which is indistinguishable from
# a real change in review, so the file could only ever be edited by hand -- and
# a hand-edited manifest is a reader that drifts from what it reads.
#
# ── THE ORDER, DOCUMENTED ONCE ──────────────────────────────────────────────
# Rows are the `md5sum` output for the tracked files of <harness dir>, in
# `LC_ALL=C sort` order OF THE PATHS. That is: bytes, not locale; the path, not
# the digest; git's file list, not `find`'s (which would sweep artifacts, logs
# and `.bak-*` snapshots the README says are deliberately not here).
#
# ── THE TWO EXCLUSIONS, AND WHY EACH IS ONE ─────────────────────────────────
#   MANIFEST.md5        a file cannot carry its own digest.
#   tools/__pycache__/  tracked `.pyc` bytes are the interpreter's, rebuilt on
#                       any run, so a row for one is a guaranteed false drift.
# These were not chosen here: they are the exclusions the hand-maintained file
# ALREADY had, read off it. The tracked set minus these two is exactly the 137
# rows the file carried, byte-for-byte, which is why the reorder commit could
# change line order and nothing else.
#
# Reader: tools/manifest_regen_guard.sh (arm C swaps two rows -> RED).
set -uo pipefail

H=${1:?usage: manifest_regen.sh <harness dir> [--check|--write]}
MODE=${2:-}
[ -d "$H" ] || { echo "### MANIFEST REGEN FATAL: no directory at $H"; exit 4; }
H=$(cd -- "$H" && pwd)
git -C "$H" rev-parse --git-dir >/dev/null 2>&1 \
  || { echo "### MANIFEST REGEN FATAL: $H is not inside a git repository -- the row set is git's file list, and there is no substitute that excludes artifacts"; exit 4; }

EXCLUDE_RE='^(MANIFEST\.md5|tools/__pycache__/)'

manifest_rows () {   # stdout: the manifest, in the documented order
  ( cd -- "$H" || exit 4
    git ls-files -- . \
      | grep -Ev "$EXCLUDE_RE" \
      | LC_ALL=C sort \
      | tr '\n' '\0' \
      | xargs -0 --no-run-if-empty md5sum -- )
}

TMP=$(mktemp "${TMPDIR:-/tmp}/manifest-regen.XXXXXX") || exit 4
trap 'rm -f "$TMP"' EXIT
manifest_rows > "$TMP" || { echo "### MANIFEST REGEN FATAL: could not hash the tracked files of $H"; exit 4; }
N=$(wc -l < "$TMP")
# A generator that emits nothing would make --check pass against an empty file.
[ "$N" -gt 0 ] || { echo "### MANIFEST REGEN FATAL: the tracked file list of $H is empty"; exit 4; }

case "$MODE" in
  --write)
    cp -- "$TMP" "$H/MANIFEST.md5" || exit 4
    echo "### MANIFEST REGEN wrote=$H/MANIFEST.md5 rows=$N md5=$(md5sum "$H/MANIFEST.md5" | awk '{print $1}')"
    ;;
  --check)
    M=$H/MANIFEST.md5
    [ -f "$M" ] || { echo "### MANIFEST REGEN DRIFT: no MANIFEST.md5 at $M"; exit 6; }
    # DIRECT FILE ARGUMENTS (CLAUDE.md law 15): a piped compare is a known
    # false-mismatch source in this environment.
    if cmp -s -- "$TMP" "$M"; then
      echo "### MANIFEST REGEN check=ok rows=$N md5=$(md5sum "$M" | awk '{print $1}')"
    else
      echo "### MANIFEST REGEN check=DRIFT rows_generated=$N rows_tracked=$(wc -l < "$M")"
      echo "###        The tracked MANIFEST.md5 is not what this generator writes."
      echo "###        ORDER counts: rows are LC_ALL=C sorted BY PATH, so a row"
      echo "###        moved but unchanged is drift too, and is reported here."
      diff -- "$M" "$TMP" | head -40 | sed 's/^/###   /'
      echo "###        ACTUATOR: manifest_regen.sh $H --write, and commit the result."
      exit 6
    fi
    ;;
  '')
    cat -- "$TMP"
    ;;
  *)
    echo "### MANIFEST REGEN FATAL: unknown mode '$MODE' (want --check, --write, or nothing)"
    exit 4
    ;;
esac
exit 0
