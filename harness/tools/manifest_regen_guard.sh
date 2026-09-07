#!/usr/bin/env bash
# manifest_regen_guard.sh -- the reader for tools/manifest_regen.sh.
#
#   usage: manifest_regen_guard.sh          (HARNESS_REPO=<repo> to point it)
#   Self-contained: one read-only pass over the real harness tree, everything
#   else a fixture git repo built under $TMPDIR. No cluster resource, no binary.
#
# THE ARMS:
#   A  the REAL tracked harness/MANIFEST.md5 is in step: --check rc 0
#   B  DETERMINISM: two regenerations of the real tree are byte-identical
#   C  MUTATION, and it is the caveat this tool was written for: SWAP TWO ROWS
#      in a fixture manifest, change nothing else, and --check goes RED rc 6.
#      Order is part of the file, so a reorder is drift and says so.
#   D  a changed DIGEST is drift too (the ordinary case), rc 6
#   E  the exclusions are real AND minimal: no MANIFEST.md5 row, no
#      __pycache__ row, and every OTHER tracked file has exactly one row
#   F  NON-VACUITY on C/D: the unmutated fixture passes --check rc 0
#   G  an absent manifest is DRIFT rc 6, never a silent pass
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TOOL=$HERE/manifest_regen.sh
REPO=${HARNESS_REPO:-}
[ -n "$REPO" ] || REPO=$(cd -- "$HERE/../.." && pwd)
H=$REPO/harness
[ -f "$TOOL" ] || { echo "GUARD FATAL: no tool at $TOOL"; exit 2; }
[ -d "$H/tools" ] || { echo "GUARD FATAL: no harness tree at $H"; exit 2; }
W=$(mktemp -d "${TMPDIR:-/tmp}/manifest-regen-guard.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
pass=0; fail=0
ok()  { echo "GUARD  ok : $*"; pass=$((pass+1)); }
bad() { echo "GUARD FAIL: $*"; fail=$((fail+1)); }

########## A. the real file is in step ########################################
OUT=$(bash "$TOOL" "$H" --check 2>&1); RC=$?
if [ "$RC" = 0 ] && printf '%s\n' "$OUT" | grep -q '^### MANIFEST REGEN check=ok'; then
  ok "A: the tracked $H/MANIFEST.md5 is byte-equal to what the generator writes ($(printf '%s' "$OUT" | tr -d '\n'))"
else
  bad "A: --check on the real tree rc=$RC"; printf '%s\n' "$OUT" | sed 's/^/      /'
fi

########## B. determinism #####################################################
bash "$TOOL" "$H" > "$W/g1.txt" 2>/dev/null
bash "$TOOL" "$H" > "$W/g2.txt" 2>/dev/null
if [ -s "$W/g1.txt" ] && cmp -s -- "$W/g1.txt" "$W/g2.txt"; then
  ok "B: two regenerations are byte-identical ($(wc -l < "$W/g1.txt") rows)"
else
  bad "B: two regenerations of the same tree differ -- the order is not deterministic"
fi

########## E. the exclusions are real, and nothing else is excluded ###########
( cd "$H" && git ls-files -- . ) | LC_ALL=C sort > "$W/tracked.txt"
awk '{print substr($0, index($0,"  ")+2)}' "$W/g1.txt" | LC_ALL=C sort > "$W/gen_paths.txt"
grep -Ev '^(MANIFEST\.md5|tools/__pycache__/)' "$W/tracked.txt" > "$W/expect_paths.txt"
if cmp -s -- "$W/expect_paths.txt" "$W/gen_paths.txt"; then
  ok "E: the generated rows are EXACTLY the tracked files minus MANIFEST.md5 and tools/__pycache__/ ($(wc -l < "$W/gen_paths.txt") of $(wc -l < "$W/tracked.txt") tracked)"
else
  bad "E: the generated path set is not tracked-minus-exclusions"
  diff -- "$W/expect_paths.txt" "$W/gen_paths.txt" | head -10 | sed 's/^/      /'
fi
if [ "$(wc -l < "$W/gen_paths.txt")" -ge 100 ]; then
  ok "E2: the generator emitted $(wc -l < "$W/gen_paths.txt") rows -- it is not green against nothing"
else
  bad "E2: only $(wc -l < "$W/gen_paths.txt") rows generated for a tree of $(wc -l < "$W/tracked.txt") tracked files"
fi

########## the fixture: a real little git repo, so --check can be driven ######
FX=$W/fx
mkdir -p "$FX/tools" "$FX/arms" "$FX/tools/__pycache__"
printf 'alpha\n'  > "$FX/README.md"
printf 'bravo\n'  > "$FX/arms/README.md"
printf 'chas\n'   > "$FX/arms/a_relock.sh"
printf 'delta\n'  > "$FX/tools/zz_last.sh"
printf 'echo\n'   > "$FX/tools/aa_first.sh"
printf 'pycbytes\n' > "$FX/tools/__pycache__/x.cpython-39.pyc"
( cd "$FX" && git init -q . && git add -A && git -c user.email=g@x -c user.name=g commit -qm f ) >/dev/null 2>&1
if [ -d "$FX/.git" ]; then ok "fixture: a 6-file git repo at \$TMPDIR/fx"; else bad "fixture: could not git init"; fi

OUT=$(bash "$TOOL" "$FX" --write 2>&1); RC=$?
if [ "$RC" = 0 ] && [ -f "$FX/MANIFEST.md5" ]; then
  ok "fixture: --write produced $(wc -l < "$FX/MANIFEST.md5") rows"
else
  bad "fixture: --write rc=$RC; $OUT"
fi
cp -- "$FX/MANIFEST.md5" "$W/fx.pristine"

########## F. NON-VACUITY: the unmutated fixture passes #######################
OUT=$(bash "$TOOL" "$FX" --check 2>&1); RC=$?
if [ "$RC" = 0 ]; then
  ok "F: NON-VACUITY -- the freshly written fixture manifest passes --check rc 0"
else
  bad "F: rc=$RC on an unmutated fixture; $OUT"
fi

########## C. THE MUTATION: swap two rows, change nothing else ################
# The whole caveat, reproduced: the SET is identical, only the ORDER moved.
awk 'NR==1{a=$0; next} NR==2{print $0; print a; next} {print}' "$W/fx.pristine" > "$FX/MANIFEST.md5"
if ! cmp -s -- "$FX/MANIFEST.md5" "$W/fx.pristine" \
   && cmp -s <(LC_ALL=C sort "$FX/MANIFEST.md5") <(LC_ALL=C sort "$W/fx.pristine"); then
  ok "C0: the mutant differs from the pristine file ONLY in line order (sorted, they are identical)"
else
  bad "C0: the row swap did not produce an order-only mutation -- C measures nothing"
fi
OUT=$(bash "$TOOL" "$FX" --check 2>&1); RC=$?
if [ "$RC" = 6 ] && printf '%s\n' "$OUT" | grep -q 'check=DRIFT'; then
  ok "C: MUTATION -- two swapped rows go RED rc 6, so the documented order is enforced and not merely described"
else
  bad "C: rc=$RC on an order-only mutation (want 6)"; printf '%s\n' "$OUT" | sed 's/^/      /'
fi

########## D. a changed digest is drift too ###################################
sed '1s/^./0/' "$W/fx.pristine" > "$FX/MANIFEST.md5"
OUT=$(bash "$TOOL" "$FX" --check 2>&1); RC=$?
if [ "$RC" = 6 ]; then
  ok "D: a single altered digest character goes RED rc 6"
else
  bad "D: rc=$RC on an altered digest (want 6)"
fi

########## G. an absent manifest is drift, not a pass #########################
rm -f "$FX/MANIFEST.md5"
OUT=$(bash "$TOOL" "$FX" --check 2>&1); RC=$?
if [ "$RC" = 6 ] && printf '%s\n' "$OUT" | grep -q 'no MANIFEST.md5'; then
  ok "G: an absent manifest is DRIFT rc 6, loudly -- not a silent pass"
else
  bad "G: rc=$RC with no manifest present (want 6)"
fi

echo "### manifest_regen_guard: pass=$pass fail=$fail -- $( [ "$fail" = 0 ] && echo PASS || echo FAIL )"
[ "$fail" = 0 ]
