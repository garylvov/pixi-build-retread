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
#   A2  MERGE-T-2, and this arm used to assert the opposite: a call in which
#       EVERY root is absent is a NO-OP -- exit 0, census printed, no refusal,
#       `cleanup.sh` never called -- whatever the evidence looks like. Job
#       5981195 printed `ROOT CENSUS present=0 absent=2` then `CLEANUP REFUSED`
#       because the no-op branch sat BELOW the evidence conditions.
#   A3  ONE OF EACH in one call -> each root lands in exactly one list.
#   A4  an absent root with COMPLETE evidence -> exit 0, "NOTHING TO DO", and
#       `cleanup.sh` IS NEVER CALLED (a stub proves it).
#   M1  MUTATION, PINNED ($N1_OLD): A1's fixture on the old gate prints the bare
#       `Roots kept:` and never the new marker.
#   M2  MUTATION: A2's fixture on the old gate reports an absent root as KEPT --
#       the defect, verbatim.
#   M3  MUTATION: A4's fixture on the old gate reaches GATE PASSED and calls
#       `cleanup.sh` on a root that does not exist.
#   M4  MUTATION for MERGE-T-2, pinned to $T2_OLD -- the file 5981195 ran, which
#       HAS the classification and STILL refuses rc 2 on an all-absent call.
#   M5  the fix is SCOPED: a PRESENT root with missing evidence refuses on the
#       old file and on the new one alike.
#
# ── CLEANUP-SEAM-1 (2026-09-06), the same file, one seam further on ──────────
#   S1  a job whose stdout carries a preamble refusal row, with EMPTY roots ->
#       removed through `cleanup.sh`, footer `### CLEANUP SETUP-REFUSED roots=2
#       removed=2`, each path named, the refusal row it acted on quoted.
#   S2  the same, but one root holds a SEALED (write-stripped) subtree -> refused
#       exactly as today, `cleanup.sh` never called, nothing deleted.
#   S3  a present root whose job printed NO refusal row -> the old refusal,
#       verbatim, and the branch says nothing at all. The fix is scoped.
#   S4  MUTATION: the branch's anchor line cut (counted, must be exactly 1) ->
#       S1's fixture is stranded again, rc 2, no footer. S1 can fail.
#
# ── CLEANUP-SEAM-2 (2026-09-07), the same file, one seam further on again ────
#   J1-J7  a job that died MID-ARM, after staging: its roots hold BYTES, so the
#       seam-1 branch declines and a SECOND branch decides. The full arm list
#       sits at the J block below.
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

# ---- A2: EVERY root absent, evidence MISSING -- MERGE-T-2 --------------------
# This arm asserted rc=2 until 2026-09-06. It was wrong, and job 5981195 is what
# says so: `### ROOT CENSUS present=0 absent=2` then `### CLEANUP REFUSED`,
# exit 2, from a cleanup owner released on `afterany` for a chain whose roots
# were already gone. MERGE-N-1's rule is that an absent root is nothing to delete
# and nothing to keep; with NO present root the evidence conditions have nothing
# to protect, so an all-absent call is a NO-OP and exits 0 whatever the evidence
# looks like. The refusal keeps the case it was written for -- A1, a root that IS
# on disk without its evidence.
rc=$(run "$NEWBED" "$W/A2.log" "$ABSENT")
[ "$rc" = 0 ] && ok "A2: an ALL-ABSENT call is a no-op and exits 0 even with evidence missing (MERGE-T-2)" \
              || { bad "A2: rc=$rc, want 0 -- this is 5981195's exit 2"; sed 's/^/      /' "$W/A2.log"; }
grep -qF "NOTHING TO DO -- every root named is ABSENT: $ABSENT" "$W/A2.log" \
  && ok "A2: it prints the census verdict and names the absent root" \
  || bad "A2: no NOTHING TO DO row naming $ABSENT"
grep -q 'CLEANUP REFUSED' "$W/A2.log" \
  && bad "A2: it still refuses -- an owner job goes terminal non-zero for doing nothing" \
  || ok "A2: and it does NOT refuse -- no meaningless non-zero for a watcher to read"
grep -qF "$STUBMARK" "$W/A2.log" \
  && bad "A2: cleanup.sh was called with no root on disk" \
  || ok "A2: cleanup.sh was NOT called"
grep -q '### ROOT CENSUS present=0 absent=1' "$W/A2.log" \
  && ok "A2: the census is printed before the verdict, as 5981195 printed it" \
  || bad "A2: no ROOT CENSUS row"

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

# ---- M4/M5: THE MERGE-T-2 MUTATION, pinned to its own pre-fix commit ---------
# $N1_OLD predates the PRESENT/ABSENT classification entirely, so it cannot show
# what MERGE-T-2 changed. The file 5981195 actually ran is the one to mutate
# against: it HAS the classification and HAS the no-op branch, and still exits 2
# because the branch sits below the `fail` check.
T2_OLD=${T2_OLD:-873263ff429af36fb8be1259f681597f99533fda}
T2F=$W/cleanup_gated.$T2_OLD.sh
if git -C "$REPO" show "$T2_OLD:harness/phase_template/cleanup_gated.sh" > "$T2F" 2>/dev/null && [ -s "$T2F" ]; then
  T2BED=$W/t2old; mk_bed "$T2BED" "$T2F"
  grep -q 'MERGE-T-2' "$T2F" \
    && bad "M4: $T2_OLD already carries the MERGE-T-2 fix -- WRONG PIN, A2 proves nothing" \
    || ok "M4: the pinned $T2_OLD gate is the pre-fix one (no MERGE-T-2 block)"
  rc=$(run "$T2BED" "$W/M4.log" "$ABSENT")
  if [ "$rc" = 2 ] && grep -q 'CLEANUP REFUSED' "$W/M4.log"; then
    ok "M4: THE DEFECT, REPRODUCED -- the pre-fix gate REFUSES rc=2 on an all-absent call (5981195)"
  else
    bad "M4: the pre-fix gate gave rc=$rc with no refusal -- A2 cannot fail; read $W/M4.log"
  fi
  grep -q '### ROOT CENSUS present=0 absent=1' "$W/M4.log" \
    && ok "M4: and it printed the same census first, exactly as 5981195's stdout reads" \
    || bad "M4: the pre-fix gate printed no census -- the pin is not the file 5981195 ran"
  # AND THE OTHER DIRECTION: the fix must not have loosened the refusal that
  # matters. A PRESENT root without its evidence still refuses on BOTH files.
  rc=$(run "$T2BED" "$W/M5old.log" "$PRESENT"); rcn=$(run "$NEWBED" "$W/M5new.log" "$PRESENT")
  [ "$rc" = 2 ] && [ "$rcn" = 2 ] \
    && ok "M5: a PRESENT root with missing evidence still refuses on both files (old=$rc new=$rcn) -- the fix is scoped" \
    || bad "M5: present-root refusal moved: old=$rc new=$rcn"
else
  bad "M4: could not extract $T2_OLD:harness/phase_template/cleanup_gated.sh -- THE MERGE-T-2 MUTATION DID NOT RUN"
fi

# ---- S1-S4: CLEANUP-SEAM-1, the preamble-refused job ------------------------
# 6000916 refused `certDET141-6000903` and `ws.DET141-6000903` forever: 6000903
# was refused by the preamble in 11 s, ran NO arm, and so could never write the
# `.rc`/`.wall`/`.lock.log` condition 1 waits for. Six empty directories, kept
# by a gate that was doing exactly what it was written to do.
#
# THE FIXTURES ARE THE REAL SHAPE, not a sketch: the harness dir holds the ONE
# artifact such a job does write (`<TAG>-<J>.reap-owed.txt`, which is what
# `det141-work/artifacts/` actually held) so `derive_harness_dir` still finds D
# while every condition-1 artifact is missing, and the stdout carries the three
# rows 6000903 printed, verbatim.
SR_ROW1='### SMOKE SETUP_FAILED binary=eb032bf58ca94cdafc2ebcd4fdef66ad07bf4b25bd57b73f50b9e91c9aaa7575 wall=0s'
SR_ROW2='### PREAMBLE FATAL: SMOKE FIX rc=3 -- this binary did not reach the frontend'
SR_ROW3='### PREAMBLE JOB REFUSED BEFORE ARM 1. MULTIARM_JOB_FATAL=1'

mk_refused_harness () {   # mk_refused_harness <dir> <tag> <rj> <refused: yes|no>
  mkdir -p "$1/artifacts" "$1/logs"
  printf '%s\n' "$2-$3 owed roots" > "$1/artifacts/$2-$3.reap-owed.txt"
  if [ "$4" = yes ]; then
    printf '%s\n%s\n%s\n' "$SR_ROW1" "$SR_ROW2" "$SR_ROW3" > "$1/logs/srguard-$3.out"
  else
    printf '### PREAMBLE CLEAN: pin+drift ok\n### ARM 1 lock rc=0\n' > "$1/logs/srguard-$3.out"
  fi
}
TAG_SR=GUARDSR$$;  HD_SR=$T/guard-sr-$$;  mk_refused_harness "$HD_SR" "$TAG_SR" "$RJ" yes
TAG_NR=GUARDNR$$;  HD_NR=$T/guard-nr-$$;  mk_refused_harness "$HD_NR" "$TAG_NR" "$RJ" no
trap 'chmod -R u+w "$W" 2>/dev/null; rm -rf "$HD_BAD" "$HD_OK" "$HD_SR" "$HD_NR" "$W"' EXIT

# A bed whose stub cleanup.sh REALLY removes, so S1 measures a removal instead of
# asserting one. It can only ever reach this guard's own temp dir: the stub
# refuses any path that is not under $W, which no real root ever is.
mk_bed_rm () {
  mkdir -p "$1"
  cp "$2" "$1/cleanup_gated.sh"
  { printf '#!/usr/bin/env bash\n'
    printf 'echo "%s $*"\n' "$STUBMARK"
    printf 'rc=0\nfor r in "$@"; do case "$r" in %s/*) rm -rf "$r";; *) echo "### [guard stub] REFUSED outside the guard temp dir: $r"; rc=1;; esac; done\nexit $rc\n' "$W"
  } > "$1/cleanup.sh"
}
RMBED=$W/newrm; mk_bed_rm "$RMBED" "$SRC"

mk_empty_root () { mkdir -p "$1/smk/c/x" "$1/smk/c/d"; }

# ---- S1: refused before arm 1, empty roots -> REMOVED ------------------------
S1A=$W/roots/cert$TAG_SR-$RJ;  mk_empty_root "$S1A"
S1B=$W/roots/ws.$TAG_SR-$RJ;   mkdir -p "$S1B"
rc=$(run "$RMBED" "$W/S1.log" "$S1A" "$S1B")
[ "$rc" = 0 ] && ok "S1: a preamble-refused job's empty roots are reclaimed (rc=0)" \
  || { bad "S1: rc=$rc, want 0 -- this is 6000916's permanent refusal"; sed 's/^/      /' "$W/S1.log"; }
grep -qF '### CLEANUP SETUP-REFUSED roots=2 removed=2' "$W/S1.log" \
  && ok "S1: the footer counts both roots and both removals" \
  || bad "S1: footer wrong: $(grep -F 'SETUP-REFUSED roots=' "$W/S1.log" || echo '<no footer>')"
grep -qF "### SETUP-REFUSED removed $S1A" "$W/S1.log" && grep -qF "### SETUP-REFUSED removed $S1B" "$W/S1.log" \
  && ok "S1: and it names each removed path" \
  || bad "S1: per-path removal rows missing"
# THE ROW IT QUOTES IS THE FIRST MATCH IN THE FILE, NOT THE LAST, and job
# 6001818 is what says so: this arm asserted `$SR_ROW3` and went RED while the
# branch had fired correctly, because `grep -m1` over a stdout that carries all
# three rows returns `SMOKE SETUP_FAILED` -- the row printed FIRST. Asserting a
# later row would be asserting an implementation detail the gate never promised.
grep -qF "$SR_ROW1" "$W/S1.log" \
  && ok "S1: the decision QUOTES the relock job's own refusal row (the first one its stdout carries)" \
  || bad "S1: the refusal row it acted on is not on the page: $(grep -m1 'SETUP-REFUSED: the relock job' "$W/S1.log" || echo '<no decision row>')"
{ [ ! -e "$S1A" ] && [ ! -e "$S1B" ]; } \
  && ok "S1: both roots are really gone from disk" || bad "S1: a root survived"
grep -q 'CLEANUP REFUSED' "$W/S1.log" \
  && bad "S1: it still printed the old refusal" || ok "S1: no stranding refusal was printed"

# ---- S2: refused before arm 1, but a SEALED subtree -> refuse as today -------
S2A=$W/roots/cert$TAG_SR-$RJ-S2; mk_empty_root "$S2A"; chmod a-w "$S2A/smk/c/x"
rc=$(run "$NEWBED" "$W/S2.log" "$S2A")
[ "$rc" = 2 ] && ok "S2: a SEALED (write-stripped) subtree still refuses (rc=2)" || bad "S2: rc=$rc, want 2"
grep -q 'SETUP-REFUSED NOT TAKEN' "$W/S2.log" && grep -q 'SEALED' "$W/S2.log" \
  && ok "S2: and it says WHY -- a provisioned store is not a job that ran no arm" \
  || bad "S2: no sealed-tree refusal row: $(grep -c 'SETUP-REFUSED' "$W/S2.log") SETUP-REFUSED rows"
grep -qF "$STUBMARK" "$W/S2.log" && bad "S2: cleanup.sh was called over a sealed tree" \
  || ok "S2: cleanup.sh was NOT called"
[ -d "$S2A" ] && ok "S2: the root is still on disk" || bad "S2: THE SEALED ROOT WAS DELETED"
chmod -R u+w "$S2A" 2>/dev/null

# ---- S3: NO refusal row -> the old refusal, unchanged ------------------------
S3A=$W/roots/cert$TAG_NR-$RJ; mk_empty_root "$S3A"
rc=$(run "$NEWBED" "$W/S3.log" "$S3A")
[ "$rc" = 2 ] && ok "S3: a present root whose job printed NO refusal row still refuses (rc=2) -- the fix is scoped" \
  || bad "S3: rc=$rc, want 2 -- the branch fired on a job that was never refused"
grep -q 'SETUP-REFUSED' "$W/S3.log" \
  && bad "S3: the branch spoke at all for a job with no refusal row" \
  || ok "S3: the branch is silent and condition 1 decides, exactly as before"
grep -q '### CLEANUP REFUSED -- nothing deleted' "$W/S3.log" \
  && ok "S3: and the old refusal is printed verbatim" || bad "S3: no old refusal row"
[ -d "$S3A" ] && ok "S3: nothing was deleted" || bad "S3: THE ROOT IS GONE"

# ---- S4: THE MUTATION -- the branch cut out of the new file ------------------
# Not a pinned old commit: this defect is being fixed in the same commit the
# guard lands in, so there is no prior file that HAS the fix to revert. The
# mutation is the anchor line, and it is COUNTED -- a mutation that does not
# mutate proves nothing (PSG E2's rule).
MUTF=$W/cleanup_gated.MUT.sh
sed 's/^setup_refused_check   # SETUP-REFUSED-BRANCH (MUTATION ANCHOR)$/: # MUTATION: the setup-refused branch is cut/' "$SRC" > "$MUTF"
mutn=$(diff "$SRC" "$MUTF" | grep -c '^< ')
if [ "$mutn" -ne 1 ]; then
  bad "S4: the mutation changed $mutn line(s), want exactly 1 -- S1 cannot fail, so it proves nothing"
else
  MUTBED=$W/mut; mk_bed_rm "$MUTBED" "$MUTF"
  S4A=$W/roots/cert$TAG_SR-$RJ-S4; mk_empty_root "$S4A"
  rc=$(run "$MUTBED" "$W/S4.log" "$S4A")
  { [ "$rc" = 2 ] && grep -q '### CLEANUP REFUSED -- nothing deleted' "$W/S4.log"; } \
    && ok "S4: THE DEFECT, REPRODUCED -- with the branch cut, 6000916's permanent refusal comes straight back (rc=$rc)" \
    || bad "S4: the mutant did not reproduce the stranding (rc=$rc) -- S1 cannot fail; read $W/S4.log"
  grep -qF '### CLEANUP SETUP-REFUSED' "$W/S4.log" \
    && bad "S4: the mutant still printed the footer -- the anchor is not the branch" \
    || ok "S4: and the mutant prints no SETUP-REFUSED footer at all"
  [ -d "$S4A" ] && ok "S4: the mutant left the root stranded, which is the whole finding" || bad "S4: the mutant deleted it anyway"
fi


# ── J1-J6: CLEANUP-SEAM-2 (2026-09-07), the job that died MID-ARM ────────────
# 6014484 refused certD16-6014471, certD6A-6014471, ws.D6A-6014471 and the
# per-arm isolated cache forever. 6014471 was NOT refused by its preamble -- it
# staged, ran arm W1 for 865 s and arm W2 for 86 s, and both wrappers exited 7
# in `retread_scope_sdist_builds`. A wrapper that exits 7 writes no `.rc`, no
# `.wall` and no `.lock.log`, so condition 1 reported all three MISSING.
#
# THE FIXTURES CARRY THE DISTINGUISHING PROPERTY, not a sketch of it: a
# seam-2 root HOLDS FILES. That is exactly why `setup_refused_check` cannot be
# widened to cover this case and why J1 asserts the seam-1 branch printing
# `SETUP-REFUSED NOT TAKEN` before seam 2 decides -- two branches, two
# conditions, and the arms prove the first one still declines.
#
#   J1  fatal row + sacct FAILED + unsealed roots WITH FILES -> removed through
#       `cleanup.sh`, footer `### CLEANUP JOB-FATAL roots=2 removed=2 ...`
#       quoting the row it acted on, both paths named, both really gone.
#   J2  the same fatal row, but sacct says COMPLETED -> REFUSED. A row in a log
#       is a claim; the accounting record is the fact, and a driver that
#       swallowed its rc must not unlock the reaper.
#   J3  the same fatal row, sacct FAILED, but one root holds a SEALED
#       (write-stripped) subtree -> refused, `cleanup.sh` never called.
#   J4  ZERO IS NOT FATAL: a stdout whose only rows are `WRAPPER EXIT rc=0`,
#       `job_fatal=0` and `_EXIT=0` -- the SUCCESS rows of the very same
#       producers -- with sacct FAILED. The branch must be SILENT, or every
#       green job in the campaign is reapable by its own success rows.
#   J5  THE det161b STDOUT SHAPE. `D` derives to the PER-ARM root (artifacts/
#       only) while the driver's stdout lives in a DIFFERENT directory under the
#       task root. The old `find "$D" -maxdepth 2` found nothing, so a branch
#       keyed on the job's own stdout was silent on the one job it was written
#       for. The fallback must find it, ANNOUNCE the widening, and reap.
#   J8/J9  CLEANUP-SEAM-2-a: sacct OUT_OF_MEMORY and NODE_FAIL are terminal
#       failures too -- one arm per added state, each with its own shim.
#   J10 the NEGATIVES, one arm per state: COMPLETED, RUNNING, PENDING and the
#       two-word `CANCELLED by <uid>` all refuse. A widening without these is
#       just "any state at all".
#   J7  BOTH families in one stdout with bytes in the root: seam 1 declines OUT
#       LOUD on the files and seam 2 decides. The ordering guarantee.
#   J6  MUTATION: the branch's anchor line cut (counted, exactly 1) -> J1's
#       fixture is stranded again, rc 2, no footer. J1 can fail.
JF_ROW='### ARM W1 WRAPPER EXIT rc=7 arm wall=865s 2026-09-07T04:41:35-04:00'
RJ_J5=98$$                    # its own job id: the $T fallback must not see the
                              # S-arm fixtures' logs, which share $RJ

mk_fatal_harness () {   # mk_fatal_harness <dir> <tag> <rj> <fatal: yes|no|nolog>
  mkdir -p "$1/artifacts"
  printf '%s\n' "$2-$3 owed roots" > "$1/artifacts/$2-$3.reap-owed.txt"
  [ "$4" = nolog ] && return 0
  mkdir -p "$1/logs"
  if [ "$4" = yes ]; then
    { printf '### SMOKE stage: PUBLISHED key=85db7fdb wall=879s\n'
      printf '%s\n' "$JF_ROW"
      printf '### %s DET-1-6 PROOF DONE job_fatal=1 2026-09-07T04:43:03-04:00\n' "$2"
      printf '### JFGUARD_EXIT=1\n'
    } > "$1/logs/jfguard-$3.out"
  else
    # The SUCCESS rows of the same four producers. Nothing here may match.
    { printf '### SMOKE stage: PUBLISHED key=85db7fdb wall=879s\n'
      printf '### ARM W1 WRAPPER EXIT rc=0 arm wall=865s 2026-09-07T04:41:35-04:00\n'
      printf '### %s DET-1-6 PROOF DONE job_fatal=0 2026-09-07T04:43:03-04:00\n' "$2"
      printf '### JFGUARD_EXIT=0\n'
      printf '### PREAMBLE JOB REFUSED BEFORE ARM 1. MULTIARM_JOB_FATAL=0\n'
    } > "$1/logs/jfguard-$3.out"
  fi
}
# A root of the shape seam 1 must DECLINE: it holds bytes.
mk_staged_root () { mkdir -p "$1/pixi/pkgs" "$1/g/fast-tmp"; printf 'staged\n' > "$1/pixi/pkgs/blob.tar"; }
# The `sacct` the gate will find on PATH. It is the ONLY thing shimmed, and only
# for the J arms: the A/M/S arms keep the real one, which returns EMPTY for a job
# id no scheduler ever issued -- verified at $RJ before this guard was written.
mk_sacct () { mkdir -p "$1"; printf '#!/usr/bin/env bash\necho "%s"\n' "$2" > "$1/sacct"; chmod +x "$1/sacct"; }
runp() {  # runp <bindir> <bed> <log> <roots...>
  local bin=$1 bed=$2 log=$3; shift 3
  ( PATH="$bin:$PATH" env -u D -u TAG -u RJ -u OJ bash "$bed/cleanup_gated.sh" "$@" ) > "$log" 2>&1
  echo $?
}
BIN_FAILED=$W/bin-failed;  mk_sacct "$BIN_FAILED" FAILED
BIN_DONE=$W/bin-done;      mk_sacct "$BIN_DONE"   COMPLETED

TAG_J1=GUARDJ1$$;  HD_J1=$T/guard-j1-$$;  mk_fatal_harness "$HD_J1" "$TAG_J1" "$RJ" yes
TAG_J2=GUARDJ2$$;  HD_J2=$T/guard-j2-$$;  mk_fatal_harness "$HD_J2" "$TAG_J2" "$RJ" yes
TAG_J3=GUARDJ3$$;  HD_J3=$T/guard-j3-$$;  mk_fatal_harness "$HD_J3" "$TAG_J3" "$RJ" yes
TAG_J4=GUARDJ4$$;  HD_J4=$T/guard-j4-$$;  mk_fatal_harness "$HD_J4" "$TAG_J4" "$RJ" no
TAG_J5=GUARDJ5$$;  HD_J5=$T/guard-j5-$$;  mk_fatal_harness "$HD_J5" "$TAG_J5" "$RJ_J5" nolog
HD_J5LOG=$T/guard-j5log-$$;  mkdir -p "$HD_J5LOG/logs"
{ printf '%s\n' "$JF_ROW"; } > "$HD_J5LOG/logs/det-$RJ_J5.out"
trap 'chmod -R u+w "$W" 2>/dev/null; rm -rf "$HD_BAD" "$HD_OK" "$HD_SR" "$HD_NR" "$HD_J1" "$HD_J2" "$HD_J3" "$HD_J4" "$HD_J5" "$HD_J5LOG" "$W"' EXIT

# ---- J1: died mid-arm, roots hold bytes, sacct FAILED -> REMOVED ------------
J1A=$W/roots/cert$TAG_J1-$RJ;      mk_staged_root "$J1A"
J1B=$W/roots/ws.$TAG_J1-$RJ;       mk_staged_root "$J1B"
rc=$(runp "$BIN_FAILED" "$RMBED" "$W/J1.log" "$J1A" "$J1B")
[ "$rc" = 0 ] && ok "J1: a job that died MID-ARM has its roots reclaimed (rc=0)" \
  || { bad "J1: rc=$rc, want 0 -- this is 6014484's permanent refusal"; sed 's/^/      /' "$W/J1.log"; }
grep -qF '### CLEANUP JOB-FATAL roots=2 removed=2' "$W/J1.log" \
  && ok "J1: the footer counts both roots and both removals" \
  || bad "J1: footer wrong: $(grep -F 'CLEANUP JOB-FATAL' "$W/J1.log" || echo '<no footer>')"
grep -qF "fatal_row=\"$JF_ROW\"" "$W/J1.log" \
  && ok "J1: and the footer QUOTES the row it acted on" \
  || bad "J1: the footer does not quote the fatal row: $(grep -F 'CLEANUP JOB-FATAL' "$W/J1.log" || echo '<none>')"
# MEASURED RED, job 6016160: this asserted `SETUP-REFUSED NOT TAKEN` and went
# red while the branch had worked perfectly. Seam 1 does not decline a mid-arm
# death -- it never speaks at all, because its regex is
# PREAMBLE/SMOKE-refusal-only and a driver that ran two arms prints none of
# those rows. Seam 1 returns at its own `[ -n "$row" ] || return 0`, exactly as
# arm S3 requires it to for any job it was not written for. The SCOPING property
# is what this row is really about, so it asserts the silence.
grep -q 'SETUP-REFUSED' "$W/J1.log" \
  && bad "J1: seam 1 spoke for a job that printed no preamble refusal row: $(grep -m1 'SETUP-REFUSED' "$W/J1.log")" \
  || ok "J1: seam 1 is SILENT -- its regex is preamble-refusal-only, so a mid-arm death is not its case at all"
{ [ ! -e "$J1A" ] && [ ! -e "$J1B" ]; } \
  && ok "J1: both roots are really gone from disk" || bad "J1: a root survived"

# ---- J2: the same row, but sacct says COMPLETED -> REFUSE -------------------
J2A=$W/roots/cert$TAG_J2-$RJ; mk_staged_root "$J2A"
rc=$(runp "$BIN_DONE" "$NEWBED" "$W/J2.log" "$J2A")
[ "$rc" = 2 ] && ok "J2: a fatal ROW with a COMPLETED accounting record refuses (rc=2)" \
  || bad "J2: rc=$rc, want 2 -- a lying row unlocked the reaper"
grep -q 'JOB-FATAL NOT TAKEN' "$W/J2.log" && grep -q "is not a terminal-failure state" "$W/J2.log" \
  && ok "J2: and it says WHY -- the row is a claim, sacct is the fact" \
  || bad "J2: no sacct refusal row: $(grep -m1 'JOB-FATAL' "$W/J2.log" || echo '<no JOB-FATAL row at all>')"
grep -qF "$STUBMARK" "$W/J2.log" && bad "J2: cleanup.sh was called on a job Slurm says completed" \
  || ok "J2: cleanup.sh was NOT called"
[ -d "$J2A" ] && ok "J2: the root is still on disk" || bad "J2: THE ROOT WAS DELETED"

# ---- J3: fatal row + FAILED, but a SEALED subtree -> REFUSE -----------------
J3A=$W/roots/cert$TAG_J3-$RJ; mk_staged_root "$J3A"; chmod a-w "$J3A/g/fast-tmp"
rc=$(runp "$BIN_FAILED" "$NEWBED" "$W/J3.log" "$J3A")
[ "$rc" = 2 ] && ok "J3: a SEALED (write-stripped) subtree still refuses (rc=2)" || bad "J3: rc=$rc, want 2"
grep -q 'JOB-FATAL NOT TAKEN' "$W/J3.log" && grep -q 'SEALED' "$W/J3.log" \
  && ok "J3: and it says WHY -- a provisioned store is a reap question, not a strand question" \
  || bad "J3: no sealed-tree refusal row"
grep -qF "$STUBMARK" "$W/J3.log" && bad "J3: cleanup.sh was called over a sealed tree" \
  || ok "J3: cleanup.sh was NOT called"
[ -d "$J3A" ] && ok "J3: the sealed root is still on disk" || bad "J3: THE SEALED ROOT WAS DELETED"
chmod -R u+w "$J3A" 2>/dev/null

# ---- J4: ZERO IS NOT FATAL --------------------------------------------------
J4A=$W/roots/cert$TAG_J4-$RJ; mk_staged_root "$J4A"
rc=$(runp "$BIN_FAILED" "$NEWBED" "$W/J4.log" "$J4A")
[ "$rc" = 2 ] && ok "J4: a stdout carrying only the SUCCESS rows of the same producers refuses (rc=2)" \
  || bad "J4: rc=$rc, want 2 -- rc=0/job_fatal=0/_EXIT=0 matched the fatal family"
grep -q 'JOB-FATAL' "$W/J4.log" \
  && bad "J4: the branch spoke for a job whose rows are all zero: $(grep -m1 'JOB-FATAL' "$W/J4.log")" \
  || ok "J4: the branch is SILENT -- zero is not fatal, and condition 1 decides"
[ -d "$J4A" ] && ok "J4: nothing was deleted" || bad "J4: THE ROOT IS GONE"

# ---- J5: the det161b stdout shape -- D has no logs/ --------------------------
J5A=$W/roots/cert$TAG_J5-$RJ_J5; mk_staged_root "$J5A"
rc=$(runp "$BIN_FAILED" "$RMBED" "$W/J5.log" "$J5A")
[ "$rc" = 0 ] && ok "J5: the job's stdout is found outside D and the root is reclaimed (rc=0)" \
  || { bad "J5: rc=$rc, want 0 -- the finder is still blind to det161b's shape"; sed 's/^/      /' "$W/J5.log"; }
grep -q "### JOB STDOUT: none under D=" "$W/J5.log" \
  && ok "J5: and the widening is ANNOUNCED, not silent" \
  || bad "J5: no fallback announcement row"
grep -qF '### CLEANUP JOB-FATAL roots=1 removed=1' "$W/J5.log" \
  && ok "J5: the footer counts the one root" || bad "J5: footer wrong"
[ ! -e "$J5A" ] && ok "J5: the root is really gone" || bad "J5: the root survived"

# ---- J7: THE OVERLAP -- both families in one stdout, and roots with BYTES ----
# The arm J1 was reaching for. A stdout can carry BOTH a preamble refusal row and
# a job-fatal row (the preamble's own row is in both families by construction:
# `MULTIARM_JOB_FATAL=1`), and then the ONLY thing that separates the two
# branches is the state of the roots. With bytes in them, seam 1 must DECLINE
# out loud -- `SETUP-REFUSED NOT TAKEN ... holds file(s)` -- and seam 2 must then
# decide. That is the ordering guarantee: the more specific branch votes first
# and hands over, rather than the two racing on the same fixture.
TAG_J7=GUARDJ7$$;  HD_J7=$T/guard-j7-$$
mkdir -p "$HD_J7/artifacts" "$HD_J7/logs"
printf '%s\n' "$TAG_J7-$RJ owed roots" > "$HD_J7/artifacts/$TAG_J7-$RJ.reap-owed.txt"
{ printf '%s\n' "$SR_ROW3"; printf '%s\n' "$JF_ROW"; } > "$HD_J7/logs/j7guard-$RJ.out"
J7A=$W/roots/cert$TAG_J7-$RJ; mk_staged_root "$J7A"
rc=$(runp "$BIN_FAILED" "$RMBED" "$W/J7.log" "$J7A")
[ "$rc" = 0 ] && ok "J7: with BOTH families in one stdout and bytes in the root, the roots are still reclaimed (rc=0)" \
  || { bad "J7: rc=$rc, want 0"; sed 's/^/      /' "$W/J7.log"; }
grep -q 'SETUP-REFUSED NOT TAKEN' "$W/J7.log" && grep -q 'holds file(s)' "$W/J7.log" \
  && ok "J7: seam 1 DECLINED OUT LOUD on the bytes -- the more specific branch votes first and hands over" \
  || bad "J7: seam 1 did not decline on the files: $(grep -m1 'SETUP-REFUSED' "$W/J7.log" || echo '<silent>')"
grep -qF '### CLEANUP JOB-FATAL roots=1 removed=1' "$W/J7.log" \
  && ok "J7: and seam 2 decided, with its own footer" || bad "J7: no JOB-FATAL footer"
rm -rf "$HD_J7"


# ---- J8/J9: the OTHER terminal failures the scheduler can hand us -----------
# CLEANUP-SEAM-2-a. The list was FAILED/TIMEOUT only, so a job the cgroup OOM
# killer took, or one whose node died under it, refused and stranded its roots
# for the same reason 6014484 did -- the evidence can never arrive and the gate
# waits for it anyway. These two arms are one per added state, each with its own
# shim, because a list widened without a reader per entry is a list nobody can
# tell is wired up.
for ST in OUT_OF_MEMORY NODE_FAIL; do
  BINST=$W/bin-$ST; mk_sacct "$BINST" "$ST"
  JSA=$W/roots/cert$TAG_J1-$RJ-$ST; mk_staged_root "$JSA"
  rc=$(runp "$BINST" "$RMBED" "$W/J8.$ST.log" "$JSA")
  [ "$rc" = 0 ] && ok "J8/J9: sacct $ST is a terminal failure and its roots are reclaimed (rc=0)" \
    || { bad "J8/J9: rc=$rc for sacct $ST, want 0 -- a scheduler-killed job still strands"; sed 's/^/      /' "$W/J8.$ST.log"; }
  grep -qF '### CLEANUP JOB-FATAL roots=1 removed=1' "$W/J8.$ST.log" \
    && ok "J8/J9: and $ST prints the JOB-FATAL footer" \
    || bad "J8/J9: no footer for $ST: $(grep -m1 'JOB-FATAL' "$W/J8.$ST.log" || echo '<no JOB-FATAL row>')"
done

# ---- J10: the NEGATIVES, one arm per state ---------------------------------
# The widening must not become "any state at all". COMPLETED is J2's case
# (a swallowed rc); RUNNING and PENDING are not terminal, so the job may still
# write the evidence; CANCELLED is an OPERATOR act with a resubmit possibly
# behind it, and reaping a paused chain's roots is the one thing a cancel must
# not cost. sacct renders it `CANCELLED by <uid>` and the gate takes the FIRST
# field, so the shim prints the two-word form to prove the field split too.
for ST in COMPLETED RUNNING PENDING "CANCELLED by 1234"; do
  SLUG=${ST%% *}
  BINST=$W/bin-neg-$SLUG; mk_sacct "$BINST" "$ST"
  JNA=$W/roots/cert$TAG_J1-$RJ-N$SLUG; mk_staged_root "$JNA"
  rc=$(runp "$BINST" "$NEWBED" "$W/J10.$SLUG.log" "$JNA")
  if [ "$rc" = 2 ] && grep -q 'JOB-FATAL NOT TAKEN' "$W/J10.$SLUG.log" \
     && ! grep -qF "$STUBMARK" "$W/J10.$SLUG.log" && [ -d "$JNA" ]; then
    ok "J10: sacct '$ST' is NOT job-fatal -- refused rc 2, cleanup.sh never called, root still on disk"
  else
    bad "J10: sacct '$ST' gave rc=$rc (want 2) cleanup_called=$(grep -qF "$STUBMARK" "$W/J10.$SLUG.log" && echo yes || echo no) root_present=$( [ -d "$JNA" ] && echo yes || echo no)"
    sed 's/^/      /' "$W/J10.$SLUG.log"
  fi
done
# ---- J6: THE MUTATION -- the branch cut out of the new file ------------------
MUTJ=$W/cleanup_gated.MUTJ.sh
sed 's/^job_fatal_check   # JOB-FATAL-BRANCH (MUTATION ANCHOR)$/: # MUTATION: the job-fatal branch is cut/' "$SRC" > "$MUTJ"
mutj=$(diff "$SRC" "$MUTJ" | grep -c '^< ')
if [ "$mutj" -ne 1 ]; then
  bad "J6: the mutation changed $mutj line(s), want exactly 1 -- J1 cannot fail, so it proves nothing"
else
  MUTJBED=$W/mutj; mk_bed_rm "$MUTJBED" "$MUTJ"
  J6A=$W/roots/cert$TAG_J1-$RJ-J6; mk_staged_root "$J6A"
  rc=$(runp "$BIN_FAILED" "$MUTJBED" "$W/J6.log" "$J6A")
  { [ "$rc" = 2 ] && grep -q '### CLEANUP REFUSED -- nothing deleted' "$W/J6.log"; } \
    && ok "J6: THE DEFECT, REPRODUCED -- with the branch cut, 6014484's permanent refusal comes straight back (rc=$rc)" \
    || bad "J6: the mutant did not reproduce the stranding (rc=$rc) -- J1 cannot fail; read $W/J6.log"
  grep -qF '### CLEANUP JOB-FATAL' "$W/J6.log" \
    && bad "J6: the mutant still printed the footer -- the anchor is not the branch" \
    || ok "J6: and the mutant prints no JOB-FATAL footer at all"
  [ -d "$J6A" ] && ok "J6: the mutant left the root stranded, which is the whole finding" || bad "J6: the mutant deleted it anyway"
fi
echo "### MERGE-N-1 absent-root guard: pass=$pass fail=$fail"
[ "$fail" = 0 ] || exit 1
exit 0
