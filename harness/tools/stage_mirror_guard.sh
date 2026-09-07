#!/usr/bin/env bash
# stage_mirror_guard.sh -- the reader for PROOF-SMOKE-1-7: ONE enumerating
# authority decides whether a staged tree is hardlinked into the read-only
# canonical tree, the publisher is gated on it, and "cannot check" is never
# printed as "shares inodes".
#
# THE INCIDENT, MEASURED. det161b-proof 6014471 printed these two rows, adjacent,
# in D16-6014471-W1.out:
#     ### stage: no mirror manifest -- cannot check inode disjointness
#     ### stage: mirror shares inodes with /oscar/data/stellex/glvov/imprint-data -- quarantining and rebuilding
# The mirror it renamed `…74.SRCLINKED-6014471` was published 03:55:59 by smoke
# job 6013332 (its .stage-mirror-key says built_by_job=6013332), and a
# smoke-published mirror carried a key and NO .stage-mirror-manifest.tsv -- the
# file phaseN_relock.sh's disjointness check read to decide what to SAMPLE. That
# check returned 1 for "cannot check", the same rc as "hardlinked", so the caller
# printed the hardlink story and 10.72 GB were rebuilt on a verdict nobody had
# measured. Both halves also sampled 50 files, of 44,113: one hardlinked file is
# enough to write through into imprint-data from every job staged off the mirror,
# and 50 of 44,113 finds it about one time in 900.
#
# THE ARMS:
#   A. a `cp -al` mirror (the defect it exists to catch): the authority returns
#      1, the row says `enumerated <n> shared <m>` with m>0, and the SMOKE's
#      real publish gate -- sourced out of proof_smoke.sh, not re-implemented --
#      refuses it.
#   B. a real copy: rc 0, `shared 0`, and the RELOCK's real gate, extracted from
#      phaseN_relock.sh, agrees on the same tree. One rule, two callers.
#   C. THE 6014471 REGRESSION: a real-copy mirror with NO
#      .stage-mirror-manifest.tsv must PASS, because the tree is walkable and
#      asking its builder's bookkeeping was the defect. And a genuinely
#      unrunnable check answers rc 2 with no hardlink claim on the page.
#   D. MUTATION: sampling restored, deterministically (`head -n 50` over a
#      fixture whose one hardlink sorts last). The linked tree then PUBLISHES --
#      the defect on demand. Without D, arm A proves nothing.
#   E. the publish writes the CENSUS as well as the key, on a real
#      smoke_stage_build_mirror round trip, and stage_mirror_census agrees
#      byte-for-byte with phaseN_relock.sh's stage_manifest on one fixture.
#
# Usage: stage_mirror_guard.sh          (self-contained, needs $TMPDIR + rsync)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LIB=$HERE/stage_mirror.sh
SMOKE=$HERE/proof_smoke.sh
TPL=$HERE/../phase_template/phaseN_relock.sh
for f in "$LIB" "$SMOKE" "$TPL"; do [ -f "$f" ] || { echo "GUARD FATAL: $f not found"; exit 2; }; done

W=$(mktemp -d "${TMPDIR:-/tmp}/stage-mirror-guard.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
FAIL=0
fail () { echo "GUARD FAIL: $*"; FAIL=1; }
ok   () { echo "GUARD  ok : $*"; }

. "$LIB"

# ---- the fixture SOURCE: stands in for imprint-data --------------------------
SRC=$W/imprint-data
mkdir -p "$SRC/pypi-packs/demo" "$SRC/third_party/PM/deep"
printf 'name = "demo"\n' > "$SRC/pixi.toml"
i=0; while [ "$i" -lt 200 ]; do printf 'f%s\n' "$i" > "$SRC/pypi-packs/demo/f$i.txt"; i=$((i+1)); done
printf 'egg\n' > "$SRC/third_party/PM/deep/payload.bin"
# the one file arm D's sampled mutant must miss: it sorts LAST, so a `head -n 50`
# sample provably never reaches it. A `shuf` here would make arm D a coin toss,
# and a guard that flips a coin is not a guard.
printf 'linked\n' > "$SRC/zzz_linked.bin"
NSRC=$(find "$SRC" -type f | wc -l)
ok "fixture source: $NSRC files under $SRC"

########## A. a cp -al mirror is REFUSED, by the authority and by the smoke ####
LINKED=$W/mirror.linked
cp -al "$SRC" "$LINKED"
OUTA=$(stage_mirror_inode_check "$LINKED" "$SRC"); rcA=$?
printf '%s\n' "$OUTA" | tail -3 | sed 's/^/GUARD:   /'
NA=$(printf '%s' "$OUTA" | sed -n 's/.*enumerated \([0-9]*\) shared \([0-9]*\).*/\1 \2/p')
set -- $NA
if [ "$rcA" = 1 ] && [ "${1:-0}" = "$NSRC" ] && [ "${2:-0}" -gt 0 ]; then
  ok "A. the cp -al mirror is REFUSED rc 1, and the row ENUMERATES all $NSRC files with ${2} shared"
else
  fail "A. rc=$rcA enumerated='${1:-}' shared='${2:-}' (wanted rc 1, enumerated $NSRC, shared > 0)"
fi
OUTAS=$( PROOF_SMOKE_LIB=1 . "$SMOKE" >/dev/null 2>&1
         SMOKE_SRC_WS=$SRC; smoke_stage_assert_mirror_disjoint "$LINKED" ); rcAS=$?
if [ "$rcAS" != 0 ] && printf '%s' "$OUTAS" | grep -q 'enumerated'; then
  ok "A. the SMOKE's own publish gate refuses it too (rc=$rcAS), through the same enumeration"
else
  fail "A. the smoke's publish gate did not refuse a cp -al tree (rc=$rcAS)"
  printf '%s\n' "$OUTAS" | sed 's/^/GUARD:   /'
fi

########## B. a real copy passes, and the RELOCK's gate agrees #################
REAL=$W/mirror.real
cp -a "$SRC" "$REAL"
OUTB=$(stage_mirror_inode_check "$REAL" "$SRC"); rcB=$?
printf '%s\n' "$OUTB" | sed 's/^/GUARD:   /'
if [ "$rcB" = 0 ] && printf '%s' "$OUTB" | grep -q "enumerated $NSRC shared 0"; then
  ok "B. a real copy passes rc 0 with all $NSRC files enumerated and 0 shared"
else
  fail "B. rc=$rcB on a real copy"
fi
RDJ=$(awk '/^stage_assert_mirror_disjoint \(\) \{/{p=1} p{print} p&&/^\}$/{exit}' "$TPL")
OUTBR=$( . "$LIB" >/dev/null 2>&1; eval "$RDJ"; SRC_WS=$SRC; stage_assert_mirror_disjoint "$REAL" ); rcBR=$?
if [ "$rcBR" = 0 ] && printf '%s' "$OUTBR" | grep -q 'enumerated'; then
  ok "B. phaseN_relock.sh's gate accepts the same tree through the same authority (rc 0)"
else
  fail "B. the relock gate disagreed with the smoke gate on one tree (rc=$rcBR)"
  printf '%s\n' "$OUTBR" | sed 's/^/GUARD:   /'
fi

########## C. the 6014471 regression, and rc 2 is not a verdict ################
rm -f "$REAL/.stage-mirror-manifest.tsv"
OUTC=$( . "$LIB" >/dev/null 2>&1; eval "$RDJ"; SRC_WS=$SRC; stage_assert_mirror_disjoint "$REAL" ); rcC=$?
if [ "$rcC" = 0 ]; then
  ok "C. a mirror with NO .stage-mirror-manifest.tsv is CHECKED and PASSES -- 6014471's quarantine cannot recur"
else
  fail "C. rc=$rcC -- a manifest-less mirror is still unjudgeable, which is the incident"
  printf '%s\n' "$OUTC" | sed 's/^/GUARD:   /'
fi
OUTC2=$(stage_mirror_inode_check "$W/no-such-tree" "$SRC"); rcC2=$?
if [ "$rcC2" = 2 ] && printf '%s' "$OUTC2" | grep -q 'CANNOT RUN' \
   && ! printf '%s' "$OUTC2" | grep -q 'HARDLINKED'; then
  ok "C. an unrunnable check answers rc 2 and says CANNOT RUN, with no hardlink claim on the page"
else
  fail "C. rc=$rcC2 -- 'cannot check' is still being reported as something else"
  printf '%s\n' "$OUTC2" | sed 's/^/GUARD:   /'
fi

########## D. MUTATION: sampling restored -> the linked tree publishes #########
MUT=$W/stage_mirror.sampled.sh
sed "s@  LC_ALL=C sort > \"\$wd/tree\"@  LC_ALL=C sort | head -n 50 > \"\$wd/tree\"@" "$LIB" > "$MUT"
if cmp -s "$LIB" "$MUT"; then
  fail "D. the mutation did not apply -- the enumeration line was not found, so D is vacuous"
else
  OUTD=$( . "$MUT" >/dev/null 2>&1; stage_mirror_inode_check "$LINKED" "$SRC" ); rcD=$?
  ND=$(printf '%s' "$OUTD" | sed -n 's/.*enumerated \([0-9]*\) shared \([0-9]*\).*/\1 \2/p')
  set -- $ND
  if [ "$rcD" = 0 ] && [ "${2:-1}" = 0 ]; then
    ok "D. MUTATION REPRODUCED: a 50-file sample of the SAME cp -al tree reports shared 0 and PUBLISHES it (enumerated ${1:-?}) -- arm A is measuring the enumeration"
  else
    fail "D. the sampled mutant still caught it (rc=$rcD shared=${2:-?}) -- arm A proves nothing"
  fi
fi

########## E. the publish writes the census, and the census agrees #############
MR=$W/mirrors
mkdir -p "$MR"
OUTE=$( PROOF_SMOKE_LIB=1 . "$SMOKE" >/dev/null 2>&1
        SMOKE_SRC_WS=$SRC; SMOKE_MIRROR_ROOT=$MR
        smoke_stage_build_mirror "$MR/k1" k1 ); rcE=$?
if [ "$rcE" = 0 ] && [ -s "$MR/k1/.stage-mirror-manifest.tsv" ] && [ -s "$MR/k1/.stage-mirror-key" ]; then
  ok "E. a real smoke publish writes BOTH stamps: key ($(wc -l < "$MR/k1/.stage-mirror-key") lines) and census ($(wc -l < "$MR/k1/.stage-mirror-manifest.tsv") rows)"
else
  fail "E. rc=$rcE key=$( [ -s "$MR/k1/.stage-mirror-key" ] && echo yes || echo no ) census=$( [ -s "$MR/k1/.stage-mirror-manifest.tsv" ] && echo yes || echo no )"
  printf '%s\n' "$OUTE" | tail -5 | sed 's/^/GUARD:   /'
fi
SM=$(awk '/^stage_manifest \(\) \{/{p=1} p{print} p&&/^\}$/{exit}' "$TPL")
{ echo 'set -u'; printf '%s\n' "$SM"; echo 'stage_manifest "$1"'; } > "$W/sm.sh"
bash "$W/sm.sh" "$REAL" > "$W/sm.out" 2>/dev/null
( . "$LIB" >/dev/null 2>&1; stage_mirror_census "$REAL" ) > "$W/smc.out" 2>/dev/null
if [ -s "$W/sm.out" ] && cmp -s "$W/sm.out" "$W/smc.out"; then
  ok "E. stage_mirror_census and phaseN_relock.sh's stage_manifest agree byte for byte ($(wc -l < "$W/sm.out") rows)"
else
  fail "E. the two censuses DISAGREE -- a mirror published by one and verified by the other would look written-through"
  diff "$W/sm.out" "$W/smc.out" 2>/dev/null | head -5 | sed 's/^/GUARD:   /'
fi

echo
[ "$FAIL" = 0 ] && echo "PROOF-SMOKE-1-7 GUARD: ALL GREEN" || echo "PROOF-SMOKE-1-7 GUARD: SOME CHECKS FAILED"
exit "$FAIL"
