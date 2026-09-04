#!/usr/bin/env bash
# census_collation_guard.sh -- the reader for the LC_ALL=C pin on the stage
# mirror census.
#
# THE DEFECT IT WOULD HAVE CAUGHT. `stage_manifest` writes the mirror census
# once, in the job that BUILDS the mirror, and `stage_verify_mirror` re-walks it
# later, in a DIFFERENT job, and compares the two with `diff`. Both ends used a
# bare `sort`. glibc's en_US.UTF-8 collation ignores `_`, `-` and case at the
# primary level where C compares bytes, so two jobs that inherited different
# locales sort the SAME file set into different line orders and the diff is
# non-empty on a tree nothing wrote to. `stage_verify_mirror` reads a non-empty
# diff as "a hardlinked input was written through", quarantines the shared
# mirror and sets MIRROR_DIRTY=1 -> the harness exits 12.
#
# That is exactly what job 5752248 (`ml1`) did: a FALSE FATAL exit 12 whose two
# censuses differ by ZERO lines once both are C-sorted, which quarantined the
# shared stage mirror and cost the next job 459 s / 62 GB of re-staging.
#
# WHAT THIS GUARD DOES. It builds a fixture tree whose names differ ONLY in the
# collation-sensitive characters (`_` vs `-`, upper vs lower), extracts the REAL
# `stage_manifest` and `stage_verify_mirror` out of each shipped harness, and:
#
#   A. writes the census under en_US.UTF-8 and re-walks it under C, and asserts
#      the two agree -- i.e. the pin makes the census locale-independent.
#   B. drives the REAL `stage_verify_mirror` under en_US.UTF-8 against a
#      manifest written under C, and asserts it reports INTACT and does NOT
#      quarantine (this is the ml1 path end to end).
#   C. NEGATIVE CONTROL: the same fixture through an UNPINNED copy of
#      `stage_manifest` (LC_ALL=C stripped) MUST disagree across the two
#      locales. Without C the two asserts above could pass on a fixture that
#      simply does not discriminate, and a guard that cannot fail is a defect.
#
# Falsification: drop `LC_ALL=` from the sort in `stage_manifest` and A and B go
# RED while C stays green.
#
# Usage: census_collation_guard.sh          (self-contained, needs only $TMPDIR)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
# The harness tree ships proof/hlgd_relock.sh next to phase_template/; a task
# checkout may carry only the template. Every harness PRESENT is checked; a
# missing one is skipped by name, never silently.
TARGETS=$HERE/phaseN_relock.sh
for extra in "$ROOT/proof/hlgd_relock.sh" "$ROOT/../p18-hardlink-guard/hlgd_relock.sh"; do
  [ -f "$extra" ] && TARGETS="$TARGETS $extra"
done

# The one locale this whole defect turns on. If the box does not have it there
# is nothing to test and saying PASS would be a lie.
ALT=en_US.UTF-8
locale -a 2>/dev/null | grep -qx 'en_US.utf8' \
  || { echo "GUARD FATAL: $ALT is not installed; this guard cannot run"; exit 2; }

W=$(mktemp -d "${TMPDIR:-/tmp}/census-collation-guard.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
FAIL=0
fail () { echo "GUARD FAIL: $*"; FAIL=1; }
ok   () { echo "GUARD  ok : $*"; }

########## the fixture: names that C and en_US.UTF-8 order DIFFERENTLY #########
# Under C these sort A-b < A_b < a-b < a_b (byte order: '-'=0x2d < '_'=0x5f,
# upper before lower). Under en_US.UTF-8 the punctuation and the case are
# ignored at the primary level, so they come out a-b, a_b, A-b, A_b. Same set,
# different line order, and `diff` sees every line as changed.
FIX=$W/payload
mkdir -p "$FIX/pkg_one" "$FIX/pkg-one" "$FIX/PKG_one" "$FIX/PKG-one"
for d in pkg_one pkg-one PKG_one PKG-one; do
  printf 'payload\n' > "$FIX/$d/wheel.whl"
done
printf 'x\n' > "$FIX/a_b"; printf 'x\n' > "$FIX/a-b"
printf 'x\n' > "$FIX/A_b"; printf 'x\n' > "$FIX/A-b"
# EVERY entry gets the SAME size and the SAME mtime on purpose. The census line
# is "%y\t%s\t%T@\t%P" and `sort` sorts the WHOLE line, so if sizes or mtimes
# differ they decide the order and the NAME -- the only collation-sensitive
# field -- never gets compared. That would make this fixture pass whether or not
# the pin is there, and the guard would be measuring nothing.
find "$FIX" -mindepth 1 -exec touch -h -d '2026-01-01T00:00:00' {} +

# prove the fixture discriminates AT ALL before trusting any verdict off it
NAMES=$W/names
( cd "$FIX" && find . -mindepth 1 -printf '%P\n' ) > "$NAMES"
if [ "$(LC_ALL=C sort "$NAMES" | md5sum)" = "$(LC_ALL=$ALT sort "$NAMES" | md5sum)" ]; then
  fail "the fixture does not collate differently under C and $ALT -- this guard would be vacuous"
else
  ok "fixture collates differently under C and $ALT (the guard can fail)"
fi

########## drive the REAL functions out of each shipped harness ################
extract () {  # $1=file  $2=function name -> the function's text, verbatim
  awk -v fn="$2" '$0 ~ "^"fn" \\(\\) \\{" {p=1} p {print} p && /^\}$/ {exit}' "$1"
}

for TPL in $TARGETS; do
  NAME=$(basename "$TPL")
  echo "GUARD: == $NAME =="
  [ -f "$TPL" ] || { fail "$NAME: not found at $TPL"; continue; }

  SM=$(extract "$TPL" stage_manifest)
  [ -n "$SM" ] || { fail "$NAME: could not extract stage_manifest"; continue; }
  case $SM in
    *"LC_ALL=C sort"*) ok "$NAME: stage_manifest carries the LC_ALL=C pin" ;;
    *) fail "$NAME: stage_manifest has NO LC_ALL=C pin on its sort" ;;
  esac

  # ---- A. census written under $ALT must equal census written under C -------
  DRV=$W/$NAME.manifest.sh
  { echo 'set -u'; printf '%s\n' "$SM"; echo 'stage_manifest "$1"'; } > "$DRV"
  LC_ALL=$ALT bash "$DRV" "$FIX" > "$W/$NAME.alt.tsv" 2>"$W/$NAME.alt.err"
  LC_ALL=C    bash "$DRV" "$FIX" > "$W/$NAME.c.tsv"   2>"$W/$NAME.c.err"
  if [ ! -s "$W/$NAME.alt.tsv" ] || [ ! -s "$W/$NAME.c.tsv" ]; then
    fail "$NAME: stage_manifest produced an empty census -- the driver is broken"
    sed 's/^/GUARD:   /' "$W/$NAME.alt.err" "$W/$NAME.c.err"
  elif LC_ALL=C diff -q "$W/$NAME.alt.tsv" "$W/$NAME.c.tsv" >/dev/null; then
    ok "$NAME: A. census under $ALT == census under C ($(wc -l < "$W/$NAME.c.tsv") entries)"
  else
    fail "$NAME: A. the census DEPENDS ON THE LOCALE -- $(LC_ALL=C diff "$W/$NAME.alt.tsv" "$W/$NAME.c.tsv" | grep -c '^[<>]') differing lines"
    LC_ALL=C diff "$W/$NAME.alt.tsv" "$W/$NAME.c.tsv" | head -8 | sed 's/^/GUARD:   /'
  fi

  # ---- B. the REAL stage_verify_mirror, cross-locale, must say INTACT ------
  SV=$(extract "$TPL" stage_verify_mirror)
  [ -n "$SV" ] || { fail "$NAME: could not extract stage_verify_mirror"; continue; }
  MIR=$W/$NAME.mirror
  rm -rf "$MIR"; cp -a "$FIX" "$MIR"
  # the census stamp is written by the BUILDING job, here under C
  LC_ALL=C bash "$DRV" "$MIR" > "$MIR/.stage-mirror-manifest.tsv"
  printf 'key=guard\n' > "$MIR/.stage-mirror-key"
  VDRV=$W/$NAME.verify.sh
  { echo 'set -u'; echo 'A=$2; TAG=GUARD; J=999999; MIRROR_DIRTY=0'
    printf '%s\n' "$SM"; printf '%s\n' "$SV"
    echo 'stage_verify_mirror "$1"'; echo 'exit $MIRROR_DIRTY'; } > "$VDRV"
  mkdir -p "$W/$NAME.art"
  # the VERIFYING job is a different job and inherits a different locale
  VOUT=$(LC_ALL=$ALT bash "$VDRV" "$MIR" "$W/$NAME.art" 2>&1); VRC=$?
  if [ "$VRC" = 0 ] && printf '%s' "$VOUT" | grep -q 'mirror INTACT' \
     && [ ! -e "$MIR.DIRTY-999999" ]; then
    ok "$NAME: B. stage_verify_mirror under $ALT reports INTACT, no quarantine"
  else
    fail "$NAME: B. FALSE FATAL reproduced (rc=$VRC, quarantined=$( [ -e "$MIR.DIRTY-999999" ] && echo yes || echo no ))"
    printf '%s\n' "$VOUT" | head -8 | sed 's/^/GUARD:   /'
    rm -rf "$MIR.DIRTY-999999"
  fi

  # ---- C. NEGATIVE CONTROL: unpin it and the same fixture MUST disagree ----
  UDRV=$W/$NAME.unpinned.sh
  sed 's/LC_ALL=C sort/sort/' "$DRV" > "$UDRV"
  LC_ALL=$ALT bash "$UDRV" "$FIX" > "$W/$NAME.u.alt.tsv"
  LC_ALL=C    bash "$UDRV" "$FIX" > "$W/$NAME.u.c.tsv"
  if LC_ALL=C diff -q "$W/$NAME.u.alt.tsv" "$W/$NAME.u.c.tsv" >/dev/null; then
    fail "$NAME: C. the UNPINNED census also agreed -- asserts A and B are vacuous here"
  else
    ok "$NAME: C. unpinned census disagrees across locales by $(LC_ALL=C diff "$W/$NAME.u.alt.tsv" "$W/$NAME.u.c.tsv" | grep -c '^[<>]') lines (the pin is what fixes it)"
  fi
done

[ "$FAIL" = 0 ] && { echo "census-collation guard: ALL PASS"; exit 0; }
echo "census-collation guard: FAILED"; exit 1
