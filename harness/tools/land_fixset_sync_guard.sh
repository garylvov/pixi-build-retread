#!/usr/bin/env bash
# GUARD for the land.sh fix-set sync, MERGE-K-2, 2026-09-05.
#
# THE DEFECT IT GUARDS. The fix set has two homes -- the TASK copy
# `tools/binsnap_fixset.txt` that `binsnap_ancestry_guard.sh` reads, and the
# versioned `harness/tools/binsnap_fixset.txt` that `harness_drift_check.sh`
# md5s every task file against. `land.sh` appended the landed row to the task
# copy ONLY, so the two diverged by one row at every landing and any job that
# then ran behind `$HARNESS_COMMIT` refused at its drift gate on a file no human
# had touched. efe74a0 re-synced it by hand for B17. This guard is the reader
# for the replacement: `fixset_land_row.sh`.
#
# NOTHING TOUCHES THE LIVE TREE. Every arm runs against a THROWAWAY git repo and
# a THROWAWAY task dir under $WORK; the live task fix set is md5'd before the
# first arm and after the last one and the guard fails if it moved. `land.sh`
# itself is never executed -- it pushes and it fast-forwards `integration/4.12`.
# What is executed is the helper land.sh now delegates to, plus a STATIC check
# that land.sh really delegates and no longer appends on its own.
#
# ARMS
#   A  a fixture landing through `fixset_land_row.sh` -> the row is in both
#      copies, the two are BYTE-IDENTICAL, the task copy is the new commit's own
#      blob, and the new HARNESS_COMMIT is printed. This is the fix.
#   B  THE MUTATION: the pre-MERGE-K-2 append, taken verbatim out of the pinned
#      $LAND_OLD blob, replayed on an identical fixture -> the two copies end up
#      DIFFERENT. Without this arm, arm A's identity assertion could be passing
#      because the fixture is trivially identical, and would prove nothing. The
#      arm also asserts the extraction is faithful (the old blob really contains
#      that append, and really never names the versioned copy) and that the
#      CURRENT land.sh no longer contains it.
#   C  idempotence -- landing the SAME row twice appends nothing the second time,
#      makes no second commit, and still leaves the copies identical. A merge
#      lane re-runs land.sh after a transient refusal; that must not double-row
#      the fix set.
#   D  the copies ALREADY differ -> REFUSE, append nothing, commit nothing. A
#      landing must not bury an existing drift under a new row.
#   E  an uncommitted edit to the versioned copy -> REFUSE. A path-limited commit
#      would otherwise sweep another lane's work into a merge-queue commit.
#   J  HARNESS-SYNC-3-1: the sync refuses rc 6 (a RUNNING job of ours can still
#      READ a file this install would rename over) -> the landing FATALs AND
#      SAYS SO AS rc 6: it re-prints harness_sync's own `### SYNC REFUSED` rows
#      from the file it captured, names the refusing job id in the actuator, and
#      does NOT repeat the rc-4 "a PENDING job is pinned elsewhere" story. J2 is
#      the mutation: with the tagged rc-6 lines cut out the same fixture leaves
#      every one of those assertions RED.
#
# THE MUTATION IS PINNED TO A COMMIT CONSTANT, NEVER `HEAD`: an arm that reads
# `HEAD:<the file it guards>` starts asserting the fix against itself the moment
# the fix is committed, which is exactly how arm B of
# cleanup_gate_env_derivation_guard.sh silently died between d2ba3fd and 5891315.
set -uo pipefail
export PATH=/users/glvov/.pixi/bin:/users/glvov/.local/bin:$PATH
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
# WHERE THE REPO IS. `$HERE/../..` is the repo when this guard runs from the
# harness worktree -- but the task-dir copy the drift check syncs lives at
# `<task>/tools/`, where `$HERE/../..` is `/oscar/data/stellex/glvov/agrescap/tasks`
# and every arm died `FATAL: no gate at .../tasks/harness/...`. A synced copy that
# can only ever FATAL is a file with no reader. `HARNESS_REPO` wins (same variable
# harness_drift_check.sh uses for the same thing); otherwise the relative guess is
# TESTED and the campaign worktree is the fallback. The guard always exercises the
# VERSIONED file -- the drift check is what proves the task copy equals it.
REPO=${HARNESS_REPO:-}
[ -n "$REPO" ] || REPO=$(cd -- "$HERE/../.." && pwd)
if [ ! -d "$REPO/harness/tools" ] || ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  REPO=/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools
fi
[ -d "$REPO/harness/tools" ] || { echo "FATAL: no harness repo at $REPO"; exit 3; }
HELPER=$REPO/harness/tools/fixset_land_row.sh
LAND=$REPO/harness/tools/land.sh
RREL=harness/tools/binsnap_fixset.txt
LAND_OLD=${LAND_OLD:-efe74a0}   # the last commit carrying the task-copy-only append
HELPER_OLD=${HELPER_OLD:-8e5b64a}  # MERGE-L-1: the last commit whose helper left MANIFEST.md5 stale
LIVE_TASK_FIXSET=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/tools/binsnap_fixset.txt
WORK=${TMPDIR:-/tmp}/land-fixset-sync-guard-$$
mkdir -p "$WORK"
pass=0; fail=0
ok()  { echo "PASS  $*"; pass=$((pass+1)); }
bad() { echo "FAIL  $*"; fail=$((fail+1)); }
trap 'rm -rf "$WORK"' EXIT

[ -f "$HELPER" ] || { echo "FATAL: no helper at $HELPER"; exit 3; }
[ -f "$LAND" ]   || { echo "FATAL: no land.sh at $LAND"; exit 3; }
echo "### guard for $HELPER (and the land.sh call site)"
echo "### work $WORK  mutation pin $LAND_OLD"
LIVE_BEFORE=$(md5sum "$LIVE_TASK_FIXSET" | awk '{print $1}')

ROW1='deadbee fixture-row-one'
ROW2='cafef00 fixture-row-two'

# A throwaway pair of homes: a real git repo laid out like the harness worktree,
# and a task dir laid out like the task tree. Both start from the same bytes,
# which is the state a real landing starts from.
mkfixture () {   # mkfixture <name> ; echoes "<repo> <taskdir>"
  local n=$1 r=$WORK/$1/repo t=$WORK/$1/task
  mkdir -p "$r/harness/tools" "$t/tools"
  printf '1111111 base-row-a\n2222222 base-row-b\n' > "$r/$RREL"
  cp -f "$r/$RREL" "$t/tools/binsnap_fixset.txt"
  git -C "$r" init -q
  git -C "$r" config user.email guard@example.invalid
  git -C "$r" config user.name  'fixset guard'
  # MERGE-L-1: the fixture carries a MANIFEST.md5 with a row for the fix set,
  # because that row is the thing every landing used to leave stale.
  { md5sum "$r/$RREL" | awk '{print $1 "  tools/binsnap_fixset.txt"}'; } > "$r/harness/MANIFEST.md5"
  # LAND-SYNC-1: the fixture carries the WRITER and its mapping library in both
  # homes, because the landing now installs the task copy THROUGH harness_sync.sh
  # instead of writing it itself. Without them the helper cannot run at all, and
  # a fixture that cannot exercise the call site would guard nothing. The empty
  # allowlist is seeded BEFORE the landing on purpose: created afterwards it is
  # itself an unsynced write and `--check` correctly calls it edited.
  cp -f "$REPO/harness/tools/harness_sync.sh" "$REPO/harness/tools/harness_drift_check.sh" "$r/harness/tools/"
  : > "$r/harness/tools/harness_drift_allowlist.txt"
  local f
  for f in binsnap_fixset.txt harness_sync.sh harness_drift_check.sh harness_drift_allowlist.txt; do
    cp -f "$r/harness/tools/$f" "$t/tools/$f"
  done
  git -C "$r" add -- "$RREL" harness/MANIFEST.md5 harness/tools/harness_sync.sh \
      harness/tools/harness_drift_check.sh harness/tools/harness_drift_allowlist.txt
  git -C "$r" commit -q -m base -- "$RREL" harness/MANIFEST.md5 harness/tools/harness_sync.sh \
      harness/tools/harness_drift_check.sh harness/tools/harness_drift_allowlist.txt
  echo "$r $t"
}

# ---- ARM A: a fixture landing -----------------------------------------------
read -r RA TA < <(mkfixture A)
BASE_COMMIT=$(git -C "$RA" rev-parse HEAD)
bash "$HELPER" "$RA" "$TA" "$ROW1" > "$WORK/A.log" 2>&1; rcA=$?
[ "$rcA" = 0 ] && ok "A: the fixture landing succeeded (rc=0)" || { bad "A: rc=$rcA"; cat "$WORK/A.log"; }
cmp "$RA/$RREL" "$TA/tools/binsnap_fixset.txt" \
  && ok "A: the two copies are BYTE-IDENTICAL after the landing" \
  || bad "A: the two copies differ after the landing -- the defect is not fixed"
grep -qE "^deadbee fixture-row-one$" "$RA/$RREL" \
  && ok "A: the landed row reached the VERSIONED copy" || bad "A: the versioned copy never got the row"
NEWC=$(git -C "$RA" rev-parse HEAD)
[ "$NEWC" != "$BASE_COMMIT" ] && ok "A: the row was committed ($BASE_COMMIT -> $NEWC)" \
  || bad "A: no commit was made, so HARNESS_COMMIT cannot advance"
git -C "$RA" cat-file blob "$NEWC:$RREL" > "$WORK/A.blob" 2>/dev/null
cmp "$WORK/A.blob" "$TA/tools/binsnap_fixset.txt" \
  && ok "A: the task copy IS the commit's blob (not a parallel write that matched)" \
  || bad "A: the task copy is not the commit's blob"
grep -q "### FIXSET HARNESS_COMMIT=$NEWC" "$WORK/A.log" \
  && ok "A: it prints the new HARNESS_COMMIT the next job must carry" \
  || bad "A: it does not print the new HARNESS_COMMIT"
[ -z "$(git -C "$RA" status --porcelain -- "$RREL")" ] \
  && ok "A: it leaves the versioned copy clean, not dirty for a human to finish" \
  || bad "A: it left the versioned copy uncommitted"

# ---- ARM B: THE MUTATION -- the pre-MERGE-K-2 append, replayed ---------------
OLDLAND=$WORK/land.OLD.sh
if git -C "$REPO" show "$LAND_OLD:harness/tools/land.sh" > "$OLDLAND" 2>/dev/null && [ -s "$OLDLAND" ]; then
  # Faithfulness of the extraction, asserted before it is used.
  grep -q 'printf .*FIXSET_ADD.* >> "\$T/tools/binsnap_fixset.txt"' "$OLDLAND" \
    && ok "B: the pinned $LAND_OLD land.sh really appends to the TASK copy only" \
    || bad "B: $LAND_OLD does not carry the task-only append -- WRONG PIN, the mutation is not the defect"
  grep -q "$RREL" "$OLDLAND" \
    && bad "B: the pinned old land.sh already names the versioned copy" \
    || ok "B: the pinned old land.sh never names $RREL -- that is the defect"
  # Replay it on a fresh, identical fixture. `$T` is the fixture task dir, so the
  # live tree is untouchable by construction.
  read -r RB TB < <(mkfixture B)
  ( T=$TB; FIXSET_ADD=$ROW1
    grep -qE "^${FIXSET_ADD%% *} " "$T/tools/binsnap_fixset.txt" || \
      printf '%s\n' "$FIXSET_ADD" >> "$T/tools/binsnap_fixset.txt" )
  cmp -s "$RB/$RREL" "$TB/tools/binsnap_fixset.txt" \
    && bad "B: the old append left the copies identical -- arm A cannot fail and is worthless" \
    || ok "B: the old append leaves the two copies DIFFERENT (arm A's assertion can fail)"
else
  bad "B: could not extract $LAND_OLD:harness/tools/land.sh -- MUTATION ARM DID NOT RUN"
fi
# ---- ARM B (static): the CURRENT land.sh delegates and no longer appends -----
# The mention of the file in land.sh's prose is fine; a REDIRECT into it is not.
grep -qE '>>.*binsnap_fixset\.txt' "$LAND" \
  && bad "B: the current land.sh still appends to binsnap_fixset.txt itself" \
  || ok "B: the current land.sh contains no append into binsnap_fixset.txt"
grep -q 'fixset_land_row.sh' "$LAND" \
  && ok "B: the current land.sh delegates to fixset_land_row.sh" \
  || bad "B: the current land.sh has no call site for the helper (reader/writer law)"

# ---- ARM C: idempotence ------------------------------------------------------
read -r RC TC < <(mkfixture C)
bash "$HELPER" "$RC" "$TC" "$ROW2" > "$WORK/C1.log" 2>&1
C1=$(git -C "$RC" rev-parse HEAD)
bash "$HELPER" "$RC" "$TC" "$ROW2" > "$WORK/C2.log" 2>&1; rcC=$?
C2=$(git -C "$RC" rev-parse HEAD)
[ "$rcC" = 0 ] && [ "$(grep -c '^cafef00 ' "$RC/$RREL")" = 1 ] \
  && ok "C: landing the same row twice leaves exactly one copy of it" \
  || bad "C: the row was duplicated or the re-run refused (rc=$rcC)"
[ "$C1" = "$C2" ] && ok "C: the second run makes no second commit" || bad "C: it committed twice ($C1 -> $C2)"
cmp "$RC/$RREL" "$TC/tools/binsnap_fixset.txt" \
  && ok "C: the copies are still identical after the re-run" || bad "C: the re-run diverged the copies"

# ---- ARM D: the copies already differ -> REFUSE ------------------------------
read -r RD TD < <(mkfixture D)
printf '9999999 somebody-else-edited-this\n' >> "$TD/tools/binsnap_fixset.txt"
DBEFORE=$(md5sum "$RD/$RREL" | awk '{print $1}')
bash "$HELPER" "$RD" "$TD" "$ROW1" > "$WORK/D.log" 2>&1; rcD=$?
[ "$rcD" = 2 ] && ok "D: pre-existing divergence refuses with exit 2" || bad "D: exit $rcD, want 2"
[ "$(md5sum "$RD/$RREL" | awk '{print $1}')" = "$DBEFORE" ] \
  && ok "D: it appended nothing while refusing" || bad "D: it appended anyway"
grep -q "cat-file blob" "$WORK/D.log" \
  && ok "D: the refusal prints the re-extract command that reconciles them" \
  || bad "D: the refusal does not say how to reconcile"

# ---- ARM E: an uncommitted edit to the versioned copy -> REFUSE --------------
read -r RE TE < <(mkfixture E)
printf '8888888 another-lane-is-holding-this\n' >> "$RE/$RREL"
cp -f "$RE/$RREL" "$TE/tools/binsnap_fixset.txt"   # copies agree; the REPO is dirty
bash "$HELPER" "$RE" "$TE" "$ROW1" > "$WORK/E.log" 2>&1; rcE=$?
[ "$rcE" = 2 ] && ok "E: an uncommitted edit to the versioned copy refuses with exit 2" \
  || bad "E: exit $rcE, want 2 -- it would have swept another lane's edit into a merge commit"
grep -qE '^8888888 ' "$RE/$RREL" && [ "$(git -C "$RE" log --oneline | wc -l)" = 1 ] \
  && ok "E: the other lane's edit is still uncommitted and untouched" \
  || bad "E: it committed or clobbered the other lane's edit"

# ---- the live tree was never touched ----------------------------------------
[ "$(md5sum "$LIVE_TASK_FIXSET" | awk '{print $1}')" = "$LIVE_BEFORE" ] \
  && ok "the LIVE task fix set is unchanged ($LIVE_BEFORE)" \
  || bad "THE GUARD MODIFIED THE LIVE FIX SET -- was $LIVE_BEFORE"


# ---- ARM F: MERGE-L-1 -- the landing must leave MANIFEST.md5 IN STEP ---------
# The defect this arm exists for is on the record: commit 8e5b64a, the first
# real landing through this helper, changed `harness/tools/binsnap_fixset.txt`
# and NOTHING else, so `md5sum -c MANIFEST.md5` in the harness worktree went
# dirty for a file nobody had hand-edited -- and the next lane inherits a
# failure it did not cause. F1 asserts the manifest is clean after a landing;
# F2 is the MUTATION: the PINNED pre-fix helper on the same fixture must leave
# it dirty, or F1 cannot fail and is worthless.
read -r RF TF < <(mkfixture F)
bash "$HELPER" "$RF" "$TF" "$ROW1" > "$WORK/F.log" 2>&1
( cd "$RF/harness" && md5sum -c MANIFEST.md5 >/dev/null 2>&1 ) \
  && ok "F1: md5sum -c MANIFEST.md5 is clean after the landing" \
  || { bad "F1: the landing left MANIFEST.md5 stale -- MERGE-L-1 is not fixed"; sed 's/^/      /' "$WORK/F.log"; }
[ -z "$(git -C "$RF" status --porcelain -- harness/MANIFEST.md5)" ] \
  && ok "F1: the manifest row was COMMITTED, not left dirty for a human" \
  || bad "F1: the manifest row is uncommitted after the landing"

if git -C "$REPO" cat-file blob "$HELPER_OLD:harness/tools/fixset_land_row.sh" > "$WORK/helper_old.sh" 2>/dev/null; then
  read -r RG TG < <(mkfixture G)
  bash "$WORK/helper_old.sh" "$RG" "$TG" "$ROW1" > "$WORK/G.log" 2>&1
  if ( cd "$RG/harness" && md5sum -c MANIFEST.md5 >/dev/null 2>&1 ); then
    bad "F2: the PINNED pre-fix helper ALSO left the manifest clean -- F1 cannot fail"
  else
    ok "F2: the pinned pre-fix helper reproduces the stale manifest, so F1 is a real assertion"
  fi
else
  bad "F2: could not extract $HELPER_OLD:harness/tools/fixset_land_row.sh -- MUTATION ARM DID NOT RUN"
fi
# ---- ARM H: LAND-SYNC-1 -- a landing leaves the task dir SYNCED, not EDITED --
# THE DEFECT, observed live on 2026-09-06 immediately after B28 landed. This
# helper installed the task copy with a bare
#     git cat-file blob "$HC:harness/tools/binsnap_fixset.txt" > "$TPATH"
# which is the exact shape HARNESS-SYNC-1 abolished one level up: it writes ONE
# file, records nothing in `tools/.harness_synced_commit`, and knows nothing
# about the queue. So `harness_sync.sh --check` -- the FIRST line of both phase
# templates' drift block -- reported `tools/binsnap_fixset.txt` as EDITED after
# every landing, i.e. the landing itself made the drift reader accuse the next
# lane of a hand edit nobody made.
#
# H1 is the fix: after a fixture landing, `--check` reads CLEAN and the recorded
# commit IS the landing's commit. H2 is the MUTATION and it is what makes H1
# mean anything -- the PINNED pre-fix helper, replayed on an identical fixture,
# must leave `--check` REFUSING. A guard that cannot fail is a defect.
read -r RH TH < <(mkfixture H)
bash "$HELPER" "$RH" "$TH" "$ROW1" > "$WORK/H.log" 2>&1; rcH=$?
HC_H=$(git -C "$RH" rev-parse HEAD)
[ "$rcH" = 0 ] && ok "H1: the landing succeeded through the writer (rc=0)" \
  || { bad "H1: rc=$rcH"; sed 's/^/      /' "$WORK/H.log"; }
grep -q '### SYNC SUMMARY' "$WORK/H.log" \
  && ok "H1: the landing ran harness_sync.sh (its SUMMARY row is in the log)" \
  || bad "H1: no SYNC SUMMARY row -- the landing did not go through the ONE writer"
[ -f "$TH/tools/.harness_synced_commit" ] && [ "$(cat "$TH/tools/.harness_synced_commit")" = "$HC_H" ] \
  && ok "H1: .harness_synced_commit records the landing's own commit $HC_H" \
  || bad "H1: the recorded sync commit is '$(cat "$TH/tools/.harness_synced_commit" 2>/dev/null)', want $HC_H"
HARNESS_TASK_DIR="$TH" HARNESS_REPO="$RH" bash "$TH/tools/harness_sync.sh" --check > "$WORK/H.check" 2>&1
rcHC=$?
grep -q '### SYNC CHECK CLEAN' "$WORK/H.check" && [ "$rcHC" = 0 ] \
  && ok "H1: harness_sync.sh --check reads CLEAN after the landing" \
  || { bad "H1: --check is not clean after the landing (rc=$rcHC)"; sed 's/^/      /' "$WORK/H.check"; }
# COMMENT-STRIPPED, for the same reason arm B strips prose from land.sh: the fix
# is DOCUMENTED in the helper by quoting the very line it removed, and a grep
# over the whole file would match that explanation and fail forever.
grep -vE "^[[:space:]]*#" "$HELPER" | grep -qE 'cat-file blob "\$HC:\$RREL"' \
  && bad "H1: the helper still re-extracts the task copy itself" \
  || ok "H1: the helper no longer writes the task copy with a bare cat-file"

if [ -s "$WORK/helper_old.sh" ]; then
  read -r RI TI < <(mkfixture I)
  bash "$WORK/helper_old.sh" "$RI" "$TI" "$ROW1" > "$WORK/I.log" 2>&1
  # The pre-fix helper never wrote the marker, so seed it with the fixture's
  # BASE commit -- which is the live shape: the task dir was last synced at some
  # earlier commit and the landing moved one file out from under it.
  git -C "$RI" rev-parse 'HEAD^' > "$TI/tools/.harness_synced_commit" 2>/dev/null \
    || git -C "$RI" rev-parse HEAD > "$TI/tools/.harness_synced_commit"
  HARNESS_TASK_DIR="$TI" HARNESS_REPO="$RI" bash "$TI/tools/harness_sync.sh" --check > "$WORK/I.check" 2>&1
  if grep -q '### SYNC CHECK CLEAN' "$WORK/I.check"; then
    bad "H2: the PINNED pre-fix helper ALSO left --check clean -- H1 cannot fail and is worthless"
  else
    ok "H2: the pinned pre-fix helper leaves --check REFUSING, so H1 is a real assertion"
  fi
else
  bad "H2: no pinned pre-fix helper blob ($HELPER_OLD) -- MUTATION ARM DID NOT RUN"
fi

# ---- ARM J: HARNESS-SYNC-3-1 -- an rc-6 refusal must be EXPLAINED AS rc 6 ----
# THE DEFECT. `harness_sync.sh` grew a PRE-INSTALL refusal (rc 6: a RUNNING job
# of ours can still READ, by path, a file this sync would rename over, and a
# cross-NFS-client rename truncates that reader and turns its exit into 0).
# The landing path picked the refusal up for free -- `fixset_land_row.sh` treats
# any non-zero from the sync as `### FIXSET FATAL` -- but the FATAL block's
# explanation named ONLY rc 4, so a lander who hit rc 6 was told a PENDING job
# was pinned elsewhere (false) and never saw the actuator. rc 6 had a producer
# and a consumer; what it lacked was a consumer that could EXPLAIN it.
#
# THE SHIM. Nothing here submits a job or waits for one. The helper resolves
# `$TASK/tools/harness_sync.sh` FIRST, so a fixture task dir carrying a
# two-line stand-in that prints one real-shaped refusal row and exits 6 presents
# the landing with exactly the condition, deterministically. `--running-list`
# (harness_sync's own test-only flag) is the equivalent one level down; here the
# thing under test is the MESSAGE, so the sync itself is the part to stub.
#
# J1 is the fix. J2 is the MUTATION -- the same fixture run against a copy of
# the helper with the rc-6 lines cut out (they carry an `RC6-MSG` tag for
# exactly this) must go RED on every one of J1's message assertions, or J1
# cannot fail and is worthless.
mkshim () {   # mkshim <taskdir>  -- a harness_sync.sh that refuses rc 6
  cat > "$1/tools/harness_sync.sh" <<'SHIM'
#!/usr/bin/env bash
echo "### SYNC REFUSED rc=6 running=999999 file=x.sh reason=read-by-fixture-job"
echo "### SYNC REFUSED (rc 6). The job(s) above are RUNNING ON ANOTHER NFS CLIENT."
exit 6
SHIM
  chmod 755 "$1/tools/harness_sync.sh"
}
rc6_msgcheck () {   # rc6_msgcheck <log> -- echoes "<named> <actuator> <rows>", 1 = present
  local L=$1 a=0 b=0 c=0
  grep -q 'rc 6 means a RUNNING job of ours is still READING' "$L" && a=1
  grep -q 'ACTUATOR: wait for job(s) 999999' "$L" && b=1
  grep -q 'from harness_sync: ### SYNC REFUSED rc=6 running=999999 file=x.sh' "$L" && c=1
  echo "$a $b $c"
}

read -r RJ TJ < <(mkfixture J)
JBASE=$(md5sum "$TJ/tools/binsnap_fixset.txt" | awk '{print $1}')
mkshim "$TJ"
bash "$HELPER" "$RJ" "$TJ" "$ROW1" > "$WORK/J.log" 2>&1; rcJ=$?
read -r JA JB JC < <(rc6_msgcheck "$WORK/J.log")
[ "$rcJ" = 3 ] && ok "J1: a landing over an rc-6 refusal FATALs (rc=3), it does not proceed" \
  || { bad "J1: rc=$rcJ, want 3"; sed 's/^/      /' "$WORK/J.log"; }
grep -q '### FIXSET FATAL: harness_sync.sh .* exited 6' "$WORK/J.log" \
  && ok "J1: the FATAL line names the sync's own rc 6" || bad "J1: the FATAL line does not name rc 6"
[ "$JA" = 1 ] && ok "J1: it says rc 6 = a RUNNING job of ours is still READING an installed file" \
  || bad "J1: the FATAL text never explains what rc 6 MEANS"
[ "$JB" = 1 ] && ok "J1: it prints the ACTUATOR and QUOTES the refusing job id 999999" \
  || bad "J1: the actuator is missing or does not name the job from the refusal row"
[ "$JC" = 1 ] && ok "J1: it re-prints harness_sync's own SYNC REFUSED row from the captured file" \
  || bad "J1: the refusal rows are not re-printed under the FATAL"
grep -q 'rc 4 means a PENDING job' "$WORK/J.log" \
  && bad "J1: an rc-6 refusal is STILL explained as rc 4 -- the defect is not fixed" \
  || ok "J1: the rc-4 explanation is NOT printed for an rc-6 refusal"
[ -f "$TJ/.fixset_land_sync.out" ] && grep -q '### SYNC REFUSED rc=6' "$TJ/.fixset_land_sync.out" \
  && ok "J1: the sync's output was CAPTURED to the job root, not only streamed" \
  || bad "J1: no captured sync output at $TJ/.fixset_land_sync.out"
[ "$(md5sum "$TJ/tools/binsnap_fixset.txt" | awk '{print $1}')" = "$JBASE" ] \
  && ok "J1: the refusal INSTALLED NOTHING -- the task fix set is still its pre-landing bytes" \
  || bad "J1: the task copy was written despite the refusal"
[ ! -f "$TJ/tools/.harness_synced_commit" ] \
  && ok "J1: no .harness_synced_commit was recorded, so no job is told a false HARNESS_COMMIT" \
  || bad "J1: a sync commit was recorded even though the sync refused"
# The row IS committed before the sync runs -- that ordering is the helper's own
# design ("a refusal here is a refusal with nothing moved EXCEPT the harness
# commit", which is idempotent on re-run). Asserted, not assumed, so a future
# reorder is caught here rather than in a landing.
[ -n "$(git -C "$RJ" log --oneline -1 --format=%H)" ] && git -C "$RJ" show --stat --oneline HEAD | grep -q 'binsnap_fixset.txt' \
  && ok "J1: the fix-set row is committed BEFORE the sync (idempotent re-run), as the helper documents" \
  || bad "J1: the commit ordering changed -- the FATAL text's 'nothing moved except the harness commit' is now false"
# COMMENT- AND echo-STRIPPED, for the reason arm B strips prose from land.sh and
# H1 strips comments from the helper: the fix DOCUMENTS `--force --reason` as the
# operator's actuator and PRINTS it as advice, so a naive grep over the whole
# file would match the very advice it wants and assert the opposite of the truth.
# What must not exist is an EXECUTED `--force`.
grep -vE "^[[:space:]]*#" "$HELPER" | grep -vE "^[[:space:]]*echo " | grep -q -- '--force' \
  && bad "J1: the helper itself passes --force to the sync -- an unattended override of a read-set refusal" \
  || ok "J1: the helper NEVER forces the sync itself (--force appears only in comments and printed advice)"
grep -c 'rc 4 means a PENDING job of ours is pinned' "$HELPER" | grep -qx 1 \
  && ok "J1: the rc-4 explanation is still in the helper, unchanged" \
  || bad "J1: the rc-4 explanation was lost while adding rc 6"

# ---- ARM J2: THE MUTATION ----------------------------------------------------
grep -v 'RC6-MSG' "$HELPER" > "$WORK/helper_norc6.sh"
CUT=$(( $(grep -c '' "$HELPER") - $(grep -c '' "$WORK/helper_norc6.sh") ))
[ "$CUT" -gt 0 ] && ok "J2: the mutation really removes the rc-6 message ($CUT tagged lines cut)" \
  || bad "J2: the RC6-MSG tag matched nothing -- THE MUTATION DID NOT RUN"
if bash -n "$WORK/helper_norc6.sh" 2>/dev/null; then
  read -r RK TK < <(mkfixture K)
  mkshim "$TK"
  bash "$WORK/helper_norc6.sh" "$RK" "$TK" "$ROW1" > "$WORK/K.log" 2>&1; rcK=$?
  read -r KA KB KC < <(rc6_msgcheck "$WORK/K.log")
  [ "$rcK" = 3 ] && ok "J2: the mutant still FATALs (rc=3), so the ONLY difference is the message" \
    || bad "J2: the mutant exited $rcK, not 3 -- the cut changed more than the message"
  [ "$KA$KB$KC" = "000" ] \
    && ok "J2: with the rc-6 text cut, ALL THREE of J1's message assertions go RED (named=$KA actuator=$KB rows=$KC)" \
    || bad "J2: the mutant still satisfies J1 (named=$KA actuator=$KB rows=$KC) -- J1 cannot fail and is worthless"
else
  bad "J2: the mutant does not parse -- MUTATION ARM DID NOT RUN"
fi
echo "### land_fixset_sync_guard: pass=$pass fail=$fail"
[ "$fail" -eq 0 ] || exit 1
exit 0
