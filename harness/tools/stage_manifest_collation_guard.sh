#!/usr/bin/env bash
# stage_manifest_collation_guard.sh -- STAGE-MIRROR-2. A COLLATION DIFFERENCE
# MUST NEVER READ AS A CHANGED MIRROR.
#
#   usage: stage_manifest_collation_guard.sh   (HARNESS_REPO=<repo> to point it)
#   Self-contained: fixtures only, no cluster resource, no live mirror touched.
#
# ── WHAT IT COST ────────────────────────────────────────────────────────────
# The stage-mirror census is written ONCE by the building job and re-walked
# LATER by a DIFFERENT job, and the two are compared. glibc's en_US.UTF-8
# collation ignores `_`, `-` and case at the primary level, so two jobs with
# different locales sort the same file set into different ORDERS. phaseN,
# hlgd and tools/stage_mirror.sh's census all pinned `LC_ALL=C sort`;
# arms/mh1_relock.sh -- the file every merge lane's relock is cut from -- ended
# in a bare `| sort`. mCB-relock 6022684 therefore declared `the mirror CHANGED
# under this job` and QUARANTINED the shared mirror on a tree nothing had
# touched: 44117 rows on both sides, identical md5 once C-sorted. A quarantine
# is not a warning -- the next job pays a full re-stage.
#
# ── THE ARMS ────────────────────────────────────────────────────────────────
#   A  STATIC: every stage_manifest / stage_mirror_census in the repo ends in
#      `LC_ALL=C sort`, and no bare `| sort` survives in any of them.
#   B  ONE RULE, NOT FOUR COPIES: the listing command is byte-identical in every
#      template and in tools/stage_mirror.sh.
#   C  THE DEFECT, REPRODUCED AND THEN NOT: a stored manifest in en_US.UTF-8
#      order versus a C-sorted live listing of the SAME set. The naive `diff`
#      the old reader ran is NON-EMPTY (so the fixture really reproduces
#      6022684) and the set comparison the new reader runs is EMPTY.
#   D  A REAL CHANGE STILL CHANGES: drop one row from the live side and the set
#      comparison must be non-empty. Without D, C is satisfied by a reader that
#      says INTACT unconditionally.
#   E  MUTATION: the bare `| sort` restored in a copy of mh1_relock.sh -> arm A
#      goes RED, so A is a real assertion.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=${HARNESS_REPO:-}
[ -n "$REPO" ] || REPO=$(cd -- "$HERE/../.." && pwd)
H=$REPO/harness
[ -d "$H/tools" ] || { echo "GUARD FATAL: no harness tree at $H"; exit 2; }
W=$(mktemp -d "${TMPDIR:-/tmp}/stagecoll.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
pass=0; fail=0
ok()  { echo "GUARD  ok : $*"; pass=$((pass+1)); }
bad() { echo "GUARD FAIL: $*"; fail=$((fail+1)); }

LISTERS="arms/mh1_relock.sh proof/hlgd_relock.sh phase_template/phaseN_relock.sh tools/stage_mirror.sh"
CANON="find \"\$1\" -mindepth 1 -xdev -printf '%y\\t%s\\t%T@\\t%P\\n' | grep -vF '.stage-mirror-' | LC_ALL=C sort"

########## A and B: one rule, pinned, in every copy ###########################
echo "GUARD: === A/B: one listing command, LC_ALL=C pinned, in every copy ==="
N=0
for rel in $LISTERS; do
  f=$H/$rel
  [ -f "$f" ] || { bad "A: no file at $rel"; continue; }
  LINE=$(grep -m1 -F -- "-printf '%y\\t%s\\t%T@\\t%P\\n'" "$f" | sed 's/^[[:space:]]*//')
  if [ -z "$LINE" ]; then bad "A: $rel has no census listing line at all"; continue; fi
  N=$((N+1))
  case "$LINE" in
    *"| LC_ALL=C sort") ok "A: $rel pins LC_ALL=C on the census sort" ;;
    *) bad "A: $rel ends its census listing with '${LINE##*| }' -- a census written under one collation and re-walked under another diffs non-empty on a tree nothing touched (6022684)" ;;
  esac
  if [ "$LINE" = "$CANON" ]; then
    ok "B: $rel's listing command is byte-identical to the one rule"
  else
    bad "B: $rel's listing command diverges from the one rule; got: $LINE"
  fi
done
[ "$N" -ge 4 ] && ok "A/B read $N copies of the listing (not green against nothing)" \
               || bad "A/B found only $N listings -- expected at least 4"

########## the fixture: one set, two orders ###################################
# Names chosen so en_US.UTF-8 and C really do disagree: C sorts by byte, so `_`
# (0x5f) lands after uppercase and before lowercase, while the UTF-8 collation
# ignores it at the primary level.
cat > "$W/rows" <<'EOS'
f	10	1.0	a_b
f	10	1.0	ab
f	10	1.0	A_b
f	10	1.0	Ab
f	10	1.0	a-b
f	10	1.0	aB
EOS
LC_ALL=C sort "$W/rows" > "$W/live"                       # this job's walk
LC_ALL=en_US.UTF-8 sort "$W/rows" > "$W/stored" 2>/dev/null || cp "$W/rows" "$W/stored"
if cmp -s -- "$W/stored" "$W/live"; then
  bad "fixture: the two collations agree on this set, so C measures nothing -- the fixture, not the code, is wrong"
else
  ok "fixture: en_US.UTF-8 and C really do order this set differently ($(diff "$W/stored" "$W/live" | grep -c '^[<>]') diff lines)"
fi

########## C: the old reader vs the new one, same two files ###################
if diff -q "$W/stored" "$W/live" >/dev/null 2>&1; then
  bad "C: the naive diff (the OLD reader) found no difference -- the fixture does not reproduce 6022684"
else
  ok "C: the OLD reader's plain diff calls this pair CHANGED -- 6022684, reproduced"
fi
LC_ALL=C sort -- "$W/stored" > "$W/s.c"; LC_ALL=C sort -- "$W/live" > "$W/l.c"
if cmp -s -- "$W/s.c" "$W/l.c"; then
  ok "C: the NEW reader (both sides LC_ALL=C sorted, then cmp) calls the same pair INTACT"
else
  bad "C: the set comparison still reports a change on one set in two orders"
fi

########## D: a REAL change still changes #####################################
grep -v '	ab$' "$W/live" > "$W/live.missing"
LC_ALL=C sort -- "$W/live.missing" > "$W/lm.c"
if cmp -s -- "$W/s.c" "$W/lm.c"; then
  bad "D: dropping a row did NOT change the verdict -- the new reader is blind, which is worse than the old one"
else
  ok "D: one row genuinely absent still reads as CHANGED, so C is not a reader that always says INTACT"
fi

########## E: MUTATION -- the bare sort restored ##############################
MSRC=$H/arms/mh1_relock.sh
if [ -f "$MSRC" ]; then
  sed "s@| grep -vF '.stage-mirror-' | LC_ALL=C sort\$@| grep -vF '.stage-mirror-' | sort@" "$MSRC" > "$W/mh1.mut"
  ML=$(grep -m1 -F -- "-printf '%y\\t%s\\t%T@\\t%P\\n'" "$W/mh1.mut" | sed 's/^[[:space:]]*//')
  case "$ML" in
    *"| LC_ALL=C sort") bad "E: could not build the bare-sort mutant of mh1_relock.sh -- arm A proves nothing" ;;
    *"| sort")          ok  "E: MUTATION -- with the bare \`| sort\` restored the mutant's listing line is '${ML##*| }', which arm A's case rejects" ;;
    *)                  bad "E: the mutant's listing line is unrecognisable: $ML" ;;
  esac
else
  bad "E: no arms/mh1_relock.sh to mutate"
fi

########## F: every reader C-sorts BOTH sides before comparing ################
for rel in arms/mh1_relock.sh proof/hlgd_relock.sh phase_template/phaseN_relock.sh; do
  f=$H/$rel
  [ -f "$f" ] || continue
  if grep -q '^  LC_ALL=C sort -- "\$m/\.stage-mirror-manifest\.tsv" > "\$stored"$' "$f" \
     && grep -q '^  LC_ALL=C sort -- "\$now" > "\$live"$' "$f" \
     && grep -q '^  if cmp -s -- "\$stored" "\$live"; then$' "$f"; then
    ok "F: $rel's stage_verify_mirror C-sorts BOTH sides and compares the sets"
  else
    bad "F: $rel's stage_verify_mirror still compares the stored bytes directly -- an older writer's order reads as a change"
  fi
done

echo "### stage_manifest_collation_guard: pass=$pass fail=$fail -- $( [ "$fail" = 0 ] && echo PASS || echo FAIL )"
[ "$fail" = 0 ]
