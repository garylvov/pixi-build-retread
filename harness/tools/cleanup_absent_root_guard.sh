#!/usr/bin/env bash
# GUARD for MERGE-N-1, 2026-09-06, HARNESS-FIX-2.
#
# THE DEFECT.  `cleanup_gated.sh` ended every refusal with
#
#     ### CLEANUP REFUSED -- nothing deleted. Roots kept: $*
#
# over the ARGUMENT LIST.  A root that had never existed -- a name a submitter
# typed, a root a previous sweep already reclaimed -- was therefore reported as
# KEPT, i.e. as bytes and inodes still sitting on a quota that this campaign
# watches.  A lane reading that row goes looking for something that is not
# there, and the inode sweeps this gate exists to serve are the readers of it.
#
# THE FIX.  Condition 0 classifies every root PRESENT or ABSENT before any other
# condition runs.  An absent root is announced, is exempt from the ownership and
# queue checks (nothing to delete means nothing to own), never sets `fail`, and
# is never handed to `cleanup.sh`.  A refusal prints the two lists SEPARATELY.
# A call in which every root is absent is a no-op that exits 0.
#
# ARMS -- and BOTH fixtures are RED on the pinned old file.
#   A1  a root that EXISTS, evidence missing -> refuse 2, "Roots kept (PRESENT on
#       disk)" names it, no ABSENT line, and the root is still on disk after.
#   A2  a root that DOES NOT EXIST, evidence missing -> refuse 2, the kept list
#       says every root named was absent, and the ABSENT line names it.
#   A3  ONE OF EACH in one call -> each root lands in exactly one list.
#   A4  an absent root with COMPLETE evidence -> exit 0, "NOTHING TO DO", and
#       `cleanup.sh` IS NEVER CALLED (a stub proves it).
#   M1  MUTATION, PINNED ($N1_OLD): A1's fixture on the old gate prints the bare
#       `Roots kept:` and never the new marker.
#   M2  MUTATION: A2's fixture on the old gate reports an absent root as KEPT --
#       the defect, verbatim.
#   M3  MUTATION: A4's fixture on the old gate reaches GATE PASSED and calls
#       `cleanup.sh` on a root that does not exist.
#
# NOTHING IS EVER DELETED BY THIS GUARD.  The gate under test is run from a COPY
# in a temp dir beside a STUB `cleanup.sh` that only prints a marker, so the real
# deletion machinery is not on the path at all; the one root that EXISTS is a
# directory inside this guard's own temp dir, never under
# /oscar/data/stellex/glvov/retread.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=${HARNESS_REPO:-}
[ -n "$REPO" ] || REPO=$(cd -- "$HERE/../.." && pwd)
if [ ! -d "$REPO/harness/tools" ] || ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  REPO=/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools
fi
[ -d "$REPO/harness/tools" ] || { echo "FATAL: no harness repo at $REPO"; exit 3; }
SRC=$REPO/harness/phase_template/cleanup_gated.sh
[ -f "$SRC" ] || { echo "FATAL: no gate at $SRC"; exit 3; }
N1_OLD=${N1_OLD:-6279978}   # the HARNESS_COMMIT that carried the defect
T=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11
W=$(mktemp -d); pass=0; fail=0
ok()  { echo "PASS  $*"; pass=$((pass+1)); }
bad() { echo "FAIL  $*"; fail=$((fail+1)); }
echo "### guard for $SRC (MERGE-N-1), mutation pinned to $N1_OLD"

# The gate under test, and the OLD one, each beside a STUB cleanup.sh.
STUBMARK='### [guard stub] cleanup.sh WAS CALLED with:'
mk_bed() {  # mk_bed <dir> <gate source file>
  mkdir -p "$1"
  cp "$2" "$1/cleanup_gated.sh"
  printf '#!/usr/bin/env bash\necho "%s $*"\nexit 0\n' "$STUBMARK" > "$1/cleanup.sh"
}
NEWBED=$W/new; mk_bed "$NEWBED" "$SRC"
OLDBED=
OLDF=$W/cleanup_gated.$N1_OLD.sh
if git -C "$REPO" show "$N1_OLD:harness/phase_template/cleanup_gated.sh" > "$OLDF" 2>/dev/null && [ -s "$OLDF" ]; then
  OLDBED=$W/old; mk_bed "$OLDBED" "$OLDF"
else
  bad "M: could not extract $N1_OLD:harness/phase_template/cleanup_gated.sh -- THE MUTATION ARMS DID NOT RUN"
fi

# Evidence fixtures under the task root, where derive_harness_dir looks.
mk_evidence() {  # mk_evidence <harness dir> <tag> <rj> <complete: yes|no>
  mkdir -p "$1/artifacts"
  printf '0\n'   > "$1/artifacts/$2-$3.rc"
  printf 'x\n'   > "$1/artifacts/$2-$3.lock.log"
  printf 'x\n'   > "$1/artifacts/$2-$3.pixi.lock.cert"
  [ "$4" = yes ] && printf '123\n' > "$1/artifacts/$2-$3.wall"   # omit .wall => condition 1 fails
  return 0
}
RJ=99$$                       # a job id no queue will know
TAG_BAD=GUARDN1BAD$$;  HD_BAD=$T/guard-n1bad-$$;  mk_evidence "$HD_BAD"  "$TAG_BAD"  "$RJ" no
TAG_OK=GUARDN1OK$$;    HD_OK=$T/guard-n1ok-$$;    mk_evidence "$HD_OK"   "$TAG_OK"   "$RJ" yes
trap 'rm -rf "$HD_BAD" "$HD_OK" "$W"' EXIT

# The PRESENT root lives inside this guard's temp dir, never under the real
# retread root -- the gate only ever reads its BASENAME.
PRESENT=$W/roots/cert$TAG_BAD-$RJ-A;  mkdir -p "$PRESENT"
ABSENT=$W/roots/cert$TAG_BAD-$RJ-B;   [ -e "$ABSENT" ] && { echo "FATAL: $ABSENT exists"; exit 3; }
PRESENT_OK=$W/roots/cert$TAG_OK-$RJ-A
ABSENT_OK=$W/roots/cert$TAG_OK-$RJ-B; [ -e "$ABSENT_OK" ] && { echo "FATAL: $ABSENT_OK exists"; exit 3; }

run() {  # run <bed> <log> <roots...>
  local bed=$1 log=$2; shift 2
  ( env -u D -u TAG -u RJ -u OJ bash "$bed/cleanup_gated.sh" "$@" ) > "$log" 2>&1
  echo $?
}

# ---- A1: a root that EXISTS, evidence missing -------------------------------
rc=$(run "$NEWBED" "$W/A1.log" "$PRESENT")
[ "$rc" = 2 ] && ok "A1: a present root with missing evidence refuses (rc=2)" || bad "A1: rc=$rc, want 2"
grep -qF "Roots kept (PRESENT on disk): $PRESENT" "$W/A1.log" \
  && ok "A1: the kept list names the root that really is on disk" \
  || bad "A1: kept list wrong: $(grep -F 'Roots kept' "$W/A1.log")"
grep -q "Roots ABSENT" "$W/A1.log" \
  && bad "A1: an absent list was printed for a root that exists" \
  || ok "A1: no ABSENT line when every root is present"
[ -d "$PRESENT" ] && ok "A1: nothing was deleted -- the root is still on disk" || bad "A1: THE ROOT IS GONE"

# ---- A2: a root that DOES NOT EXIST, evidence missing ------------------------
rc=$(run "$NEWBED" "$W/A2.log" "$ABSENT")
[ "$rc" = 2 ] && ok "A2: missing evidence still refuses (rc=2)" || bad "A2: rc=$rc, want 2"
grep -q "Roots kept (PRESENT on disk): <none -- every root named was absent>" "$W/A2.log" \
  && ok "A2: an absent root is NOT reported as kept -- MERGE-N-1, fixed" \
  || bad "A2: still claims to have kept it: $(grep -F 'Roots kept' "$W/A2.log")"
grep -qF "Roots ABSENT (never existed or already reclaimed, nothing kept): $ABSENT" "$W/A2.log" \
  && ok "A2: the absent root is named in its own list" \
  || bad "A2: no ABSENT line naming $ABSENT"

# ---- A3: one of each in one call --------------------------------------------
rc=$(run "$NEWBED" "$W/A3.log" "$PRESENT" "$ABSENT")
grep -qF "Roots kept (PRESENT on disk): $PRESENT" "$W/A3.log" \
  && ok "A3: the mixed call keeps only the present root (rc=$rc)" \
  || bad "A3: kept list wrong: $(grep -F 'Roots kept' "$W/A3.log")"
grep -qF "Roots ABSENT (never existed or already reclaimed, nothing kept): $ABSENT" "$W/A3.log" \
  && ok "A3: and reports only the absent one as absent" \
  || bad "A3: absent list wrong: $(grep -F 'Roots ABSENT' "$W/A3.log")"

# ---- A4: an absent root with COMPLETE evidence -------------------------------
rc=$(run "$NEWBED" "$W/A4.log" "$ABSENT_OK")
[ "$rc" = 0 ] && ok "A4: an absent root with complete evidence exits 0" || bad "A4: rc=$rc, want 0"
grep -q "NOTHING TO DO -- every root named is ABSENT" "$W/A4.log" \
  && ok "A4: and says so instead of claiming a reclaim" || bad "A4: no NOTHING TO DO row"
grep -qF "$STUBMARK" "$W/A4.log" \
  && bad "A4: cleanup.sh was called on a root that does not exist" \
  || ok "A4: cleanup.sh was NOT called -- nothing was handed a name with no bytes"

# ---- M1/M2/M3: THE MUTATION --------------------------------------------------
if [ -n "$OLDBED" ]; then
  rc=$(run "$OLDBED" "$W/M1.log" "$PRESENT")
  grep -q "Roots kept (PRESENT on disk)" "$W/M1.log" \
    && bad "M1: the pinned $N1_OLD gate already classifies -- WRONG PIN, A1 proves nothing" \
    || ok "M1: the pinned $N1_OLD gate prints no PRESENT/ABSENT classification (rc=$rc)"
  grep -qE '^### CLEANUP REFUSED -- nothing deleted\. Roots kept: ' "$W/M1.log" \
    && ok "M1: it prints the old undifferentiated 'Roots kept:' row" \
    || bad "M1: the old row is not there either -- read $W/M1.log"

  rc=$(run "$OLDBED" "$W/M2.log" "$ABSENT")
  grep -qF "Roots kept: $ABSENT" "$W/M2.log" \
    && ok "M2: THE DEFECT, REPRODUCED -- the old gate reports a root that never existed as KEPT (rc=$rc)" \
    || bad "M2: $N1_OLD did not reproduce MERGE-N-1 (rc=$rc) -- read $W/M2.log"
  grep -q "Roots ABSENT" "$W/M2.log" \
    && bad "M2: the old gate already distinguished absent roots" \
    || ok "M2: and it says nothing at all about the root being absent"

  rc=$(run "$OLDBED" "$W/M3.log" "$ABSENT_OK")
  grep -qF "$STUBMARK" "$W/M3.log" \
    && ok "M3: the old gate hands an ABSENT root to cleanup.sh (rc=$rc) -- the second half of the defect" \
    || bad "M3: the old gate did not call cleanup.sh (rc=$rc) -- read $W/M3.log"
fi

echo "### MERGE-N-1 absent-root guard: pass=$pass fail=$fail"
[ "$fail" = 0 ] || exit 1
exit 0
