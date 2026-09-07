#!/usr/bin/env bash
# quarantine_collision_guard.sh -- the reader for DET-1-6-b: a quarantine name
# two arms of one job can both produce is not a quarantine.
#
# THE DEFECT. Every quarantine in phaseN_relock.sh was `mv "$x" "$x.<KIND>-$J"`,
# one name per JOB -- and a multi-arm job runs several arms under one job id.
# The second arm's `mv` finds a DIRECTORY at that name and, being mv, moves the
# tree INSIDE it: the first quarantine now contains the second, the outer
# manifest describes neither, and a reader looking for `<mirror>.DIRTY-<job>`
# finds two different failures nested. The live mirror root already carries four
# `.DIRTY-<jobid>` and one `.SRCLINKED-<jobid>`, so this is a shape that fires.
# The same name-per-job mistake was in the BUILD temp (`$m.building.$J`, in both
# phaseN_relock.sh and proof_smoke.sh), where the next arm's `rm -rf` deletes the
# previous arm's half-built mirror -- 10.72 GB of real bytes.
#
# THE ARMS:
#   A. two quarantines of the same mirror path in ONE job, from two arm tags:
#      two SIBLING directories, and neither inside the other.
#   B. the collision itself, made deterministic by shimming `date` to a fixed
#      epoch so both calls compute the SAME name: the second must REFUSE and
#      leave the tree where it is. Three fields make a collision unlikely; only
#      the check makes nesting impossible.
#   C. MUTATION -- the pre-fix line, verbatim (`mv "$1" "$1.$2-$J"`): the same
#      fixture must NEST, or arm A is measuring nothing.
#   D. law 2, the call sites: no `.<KIND>-$J` rename survives in
#      phaseN_relock.sh, all three sites go through stage_quarantine, and both
#      build temps carry a pid.
#
# The REAL functions are extracted from phaseN_relock.sh with the same awk
# test_stage_mirror.sh uses -- never a re-implementation, except in arm C where
# the re-implementation IS the defect being reproduced.
#
# Usage: quarantine_collision_guard.sh          (self-contained, needs $TMPDIR)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# HARNESS-CONSOL-8 (2026-09-07). THE FILE UNDER TEST WAS ONE HARDCODED SCALAR,
# and `arms/mh1_relock.sh` carries this transformation in full -- `stage_quarantine
# () {` with the `$src.$kind-$J-$arm-<epoch>` name, the rc-3 collision refusal,
# three call sites and the `b="$1.building.$J-$arm-$$"` build temp -- while no
# guard read it for any of them. A transformation with no reader on a file is
# exactly a half-landed change (law 2), and this one is worse than an unread
# file: `arms/` is the DERIVATION SOURCE for every merge lane's relock script
# (README.md, "kept verbatim because later arms are derived from them by
# SUBSTITUTE-only edits"), and seven merge lanes have `git cat-file blob
# <sha>:harness/arms/mh1_relock.sh` in the lane log in two days. A defect that
# reaches this file is copied forward into every one of them.
#
# So the target is a LIST, in the same shape census_collation_guard.sh and
# wheel_store_census_guard.sh already use, and a target that is not there is
# skipped rather than fatal -- `arms/` is additive and a checkout without it must
# still run the arms it does have.
TARGETS=$HERE/phaseN_relock.sh
for extra in "$HERE/../arms/mh1_relock.sh"; do
  [ -f "$extra" ] && TARGETS="$TARGETS $extra"
done
SMOKE=$HERE/../tools/proof_smoke.sh
for f in $TARGETS; do [ -f "$f" ] || { echo "GUARD FATAL: $f not found"; exit 2; }; done

W=$(mktemp -d "${TMPDIR:-/tmp}/quarantine-guard.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
FAIL=0
fail () { echo "GUARD FAIL: $*"; FAIL=1; }
ok   () { echo "GUARD  ok : $*"; }


# ONE PASS PER TARGET. Everything from here to `done` is the per-target body:
# the extraction, arms A-C over it, and arm D's three greps. Each target gets
# its own work directory so two targets' fixtures cannot collide, and every row
# names the file it read.
for TPL in $TARGETS; do
TN=$(basename "$TPL")
WT=$W/$TN; mkdir -p "$WT"
echo "===== DET-1-6-b TARGET: $TN ====="
FUNCS=$WT/stage_funcs.sh
awk '/^STAGE_METHOD=/{f=1} f&&/^if \[ ! -e "\$WS\/\.cert-staged" \]/{exit} f{print}' "$TPL" > "$FUNCS"
# PROOF-SMOKE-1-7 landed after this guard was written: the extracted block now
# SOURCES tools/stage_mirror.sh and refuses (exit 14) without it, which is how
# job 6015434 turned every arm here into rc 14. Providing a dependency is the
# guard's job; the sibling candidate in that block resolves to whatever sits
# beside the extract.
for cand in "$HERE/../tools/stage_mirror.sh" "$HERE/stage_mirror.sh"; do
  [ -f "$cand" ] && { cp "$cand" "$(dirname -- "$FUNCS")/stage_mirror.sh"; break; }
done
[ -f "$(dirname -- "$FUNCS")/stage_mirror.sh" ] \
  || { echo "GUARD FATAL: no stage_mirror.sh beside $TPL -- the extracted staging block cannot be sourced"; exit 2; }
grep -q '^stage_quarantine () {' "$FUNCS" || { echo "GUARD FATAL: no stage_quarantine in $TPL"; exit 2; }
ok "extracted $(grep -c '^stage_[a-z_]* ()' "$FUNCS") stage functions from $TN"

mkmirror () { mkdir -p "$1/pypi-packs"; printf 'payload\n' > "$1/pypi-packs/f.txt"; }
sibs ()     { find "$(dirname "$1")" -maxdepth 1 -name "$(basename "$1").DIRTY-*" | sort; }

########## A. two arms of one job, same mirror path #############################
M=$WT/A/mirror
mkdir -p "$WT/A"
mkmirror "$M"
OUT1=$( . "$FUNCS" >/dev/null 2>&1; J=7770001 TAG=D6A; stage_quarantine "$M" DIRTY )
mkmirror "$M"
OUT2=$( . "$FUNCS" >/dev/null 2>&1; J=7770001 TAG=D6B; stage_quarantine "$M" DIRTY )
printf '%s\n%s\n' "$OUT1" "$OUT2" | sed 's/^/GUARD:   /'
N=$(sibs "$M" | wc -l)
NEST=$(find "$WT/A" -maxdepth 3 -path "*mirror.DIRTY-*/mirror" | wc -l)
if [ "$N" = 2 ] && [ "$NEST" = 0 ]; then
  ok "A. two arms of job 7770001 produced TWO SIBLING quarantines, neither inside the other"
  sibs "$M" | sed 's/^/GUARD:   /'
else
  fail "A. quarantines=$N nested=$NEST (wanted 2 siblings, 0 nested)"
  find "$WT/A" -maxdepth 3 | sed 's/^/GUARD:   /'
fi
case "$OUT1" in *"$M.DIRTY-7770001-D6A-"*) ok "A. the name carries job id AND arm tag AND a timestamp" ;;
  *) fail "A. the quarantine name does not carry job+arm+timestamp: $OUT1" ;; esac

########## B. a real collision REFUSES rather than nests ########################
mkdir -p "$WT/bin" "$WT/B"
printf '#!/usr/bin/env bash\necho 1757000000\n' > "$WT/bin/date"; chmod +x "$WT/bin/date"
MB=$WT/B/mirror
mkmirror "$MB"
OB1=$( . "$FUNCS" >/dev/null 2>&1; PATH=$WT/bin:$PATH J=7770002 TAG=D6A; stage_quarantine "$MB" DIRTY )
mkmirror "$MB"
OB2=$( . "$FUNCS" >/dev/null 2>&1; PATH=$WT/bin:$PATH J=7770002 TAG=D6A; stage_quarantine "$MB" DIRTY ); rcB2=$?
printf '%s\n%s\n' "$OB1" "$OB2" | sed 's/^/GUARD:   /'
NB=$(sibs "$MB" | wc -l)
if [ "$rcB2" = 3 ] && printf '%s' "$OB2" | grep -q 'QUARANTINE NAME COLLISION' \
   && [ "$NB" = 1 ] && [ -d "$MB" ]; then
  ok "B. a genuine name collision REFUSES rc 3, leaves the tree at $MB, and does not nest"
else
  fail "B. rc=$rcB2 quarantines=$NB mirror_still_there=$( [ -d "$MB" ] && echo yes || echo no )"
fi

########## C. MUTATION: the pre-fix line, and it must NEST ######################
MUT=$WT/mut_funcs.sh
awk '/^stage_quarantine \(\) \{/{p=1; print "stage_quarantine () {  # PRE-FIX, verbatim"; print "  mv \"$1\" \"$1.$2-$J\" 2>/dev/null && echo \"### stage: quarantined -> $1.$2-$J\""; print "}"; next} p&&/^}$/{p=0; next} p{next} {print}' "$FUNCS" > "$MUT"
if ! grep -q 'PRE-FIX, verbatim' "$MUT"; then
  fail "C. the mutant could not be built -- stage_quarantine was not found, so C is vacuous"
else
  MC=$WT/C/mirror
  mkdir -p "$WT/C"
  mkmirror "$MC"
  ( . "$MUT" >/dev/null 2>&1; J=7770003 TAG=D6A; stage_quarantine "$MC" DIRTY ) >/dev/null
  mkmirror "$MC"
  ( . "$MUT" >/dev/null 2>&1; J=7770003 TAG=D6B; stage_quarantine "$MC" DIRTY ) >/dev/null
  if [ -d "$MC.DIRTY-7770003/mirror" ]; then
    ok "C. MUTATION REPRODUCED: with the pre-fix line the second arm's tree is NESTED at $MC.DIRTY-7770003/mirror -- arm A is measuring the fix"
  else
    fail "C. the pre-fix line did not nest -- arm A proves nothing"
    find "$WT/C" -maxdepth 3 | sed 's/^/GUARD:   /'
  fi
fi

########## D. law 2: every call site goes through the helper ####################
LEFT=$(grep -nE 'mv "\$[A-Za-z_]+" "\$[A-Za-z_]+\.(DIRTY|SRCLINKED|stale)-\$J"' "$TPL")
if [ -z "$LEFT" ]; then
  ok "D. no job-id-only quarantine rename survives in $TN"
else
  fail "D. a hand-rolled quarantine rename is still there:"; printf '%s\n' "$LEFT" | sed 's/^/GUARD:   /'
fi
NCALL=$(grep -c 'stage_quarantine "' "$TPL")
if [ "$NCALL" -ge 3 ]; then
  ok "D. all three quarantine sites in $TN call stage_quarantine ($NCALL call sites)"
else
  fail "D. only $NCALL stage_quarantine call sites in $TN -- one of the three was missed"
fi
if grep -q 'b="\$1.building.\$J-\$arm-\$\$"' "$TPL"; then
  ok "D. $TN's mirror build temp carries the arm tag and this process's pid"
else
  fail "D. the build temp in $TN is still one name per JOB -- the next arm's rm -rf deletes it"
fi
done   # TARGETS -- the per-target arms end here; what follows reads only $SMOKE

if [ -f "$SMOKE" ]; then
  if grep -q 'b=\$m.building.\$jid-' "$SMOKE"; then
    ok "D. proof_smoke.sh's publish temp carries the arm tag and pid too"
  else
    fail "D. proof_smoke.sh still builds into \$m.building.\$jid -- the same collision on the publish path"
  fi
else
  fail "D. $SMOKE not found -- the publish path could not be checked"
fi

echo
[ "$FAIL" = 0 ] && echo "DET-1-6-b GUARD: ALL GREEN" || echo "DET-1-6-b GUARD: SOME CHECKS FAILED"
exit "$FAIL"
