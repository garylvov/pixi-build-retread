#!/usr/bin/env bash
# GUARD for harness_sync.sh, the single writer of task copies. HARNESS-SYNC-1,
# 2026-09-06.
#
# THE DEFECT IT GUARDS. Three lanes write one shared task-dir harness. Each one
# re-extracted the files it cared about with a bare
# `git cat-file blob <its commit>:<path> > <task path>`, and the only thing that
# noticed the collision was somebody else's job, hours later, at its drift gate:
# the 08:45-09:00 refusals in ticks 499-501 are jobs pinned to `9463cf3` dying
# because another lane had moved the task dir to a newer commit while they sat
# PENDING. `harness_sync.sh` is the replacement writer -- whole set, queue-aware,
# rename-install, verified by md5 -- and this is its reader.
#
# NOTHING TOUCHES THE LIVE TREE. Every arm runs against a THROWAWAY git repo and
# a THROWAWAY task dir under $WORK, reached through HARNESS_REPO/HARNESS_TASK_DIR,
# and `squeue` is a STUB SCRIPT the arm writes -- no job is ever submitted and no
# real queue is ever consulted. The live task dir's own record file is md5'd
# before the first arm and after the last one and the guard fails if it moved.
#
# ARMS
#   A  a sync onto a clean fixture installs and VERIFIES every mapped file, the
#      executable bit survives, the merge-h basename rule is honoured, the drift
#      check then says CLEAN, and the record names the commit.
#   B  a PENDING job pinned to the OLDER commit makes the sync REFUSE rc 4,
#      naming the job id, having written NOTHING. `--force` alone still refuses.
#      `--force --reason` proceeds and prints the reason for the log row.
#   C  an in-place edit after a sync is NAMED by `--check` (rc 3), with the
#      file's path and the `harness_sync.sh <commit>` command in the message.
#   D  THE MUTATION, pinned to the pre-fix commit $PREFIX: at that commit there
#      IS no harness_sync.sh, and the procedure the drift refusal printed --
#      a bare `git cat-file blob` per file -- moves the same fixture with NO
#      refusal at all, stranding the same queued job. Without this arm, arm B's
#      refusal could be a property of the fixture rather than of the fix.
#   E  a SECOND mutation: harness_sync.sh with its queue block cut out must make
#      arm B's assertion go RED, or arm B cannot fail and is worthless.
#   F  static: the mapping is SOURCED from the drift check, not copied -- the
#      check defines map_of when sourced with HARNESS_DRIFT_LIB, and the writer
#      contains no map_of of its own; and both phase templates run `--check`
#      as the first line of their drift block.
#
# THE MUTATION IS PINNED TO A COMMIT CONSTANT, NEVER `HEAD`: an arm that reads
# `HEAD:<the file it guards>` starts asserting the fix against itself the moment
# the fix is committed.
set -uo pipefail
export PATH=/users/glvov/.pixi/bin:/users/glvov/.local/bin:$PATH

PREFIX=9a154a6857997f2c48707252a267d341dfccc009   # HARNESS-EXIT-3's tip: before this lane

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=${HARNESS_REPO:-}
[ -n "$REPO" ] || REPO=$(cd -- "$HERE/../.." && pwd)
if [ ! -d "$REPO/harness/tools" ] || ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  REPO=/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools
fi
[ -d "$REPO/harness/tools" ] || { echo "FATAL: no harness repo at $REPO"; exit 3; }
SYNC=$HERE/harness_sync.sh
[ -f "$SYNC" ] || SYNC=$REPO/harness/tools/harness_sync.sh
[ -f "$SYNC" ] || { echo "FATAL: no harness_sync.sh at $HERE or $REPO/harness/tools"; exit 3; }
DRIFT=$(dirname -- "$SYNC")/harness_drift_check.sh
[ -f "$DRIFT" ] || { echo "FATAL: no harness_drift_check.sh beside $SYNC"; exit 3; }

LIVE_RECORD=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/tools/.harness_synced_commit
LIVE_BEFORE=$( [ -f "$LIVE_RECORD" ] && md5sum "$LIVE_RECORD" | awk '{print $1}' || echo absent )

WORK=$(mktemp -d "${TMPDIR:-/tmp}/harness_sync_guard.XXXXXX") || exit 3
trap 'rm -rf "$WORK"' EXIT
pass=0; fail=0
ok()  { echo "PASS  $*"; pass=$((pass+1)); }
bad() { echo "FAIL  $*"; fail=$((fail+1)); }

# ---- the fixture ------------------------------------------------------------
# A miniature of the real shape: harness/{tools,phase_template,arms} in the repo,
# tools/, tools/phase_template/ and merge-h/ in the task dir, one file per
# mapping rule including the merge-h basename rule and one executable.
mkfixture () {                        # $1 = tag; echoes "<repo> <task> <v1> <v2>"
  local tag=$1 R=$WORK/$1/repo T=$WORK/$1/task
  mkdir -p "$R/harness/tools" "$R/harness/phase_template" "$R/harness/arms"
  git -C "$R" init -q 2>/dev/null
  git -C "$R" config user.email guard@example.invalid
  git -C "$R" config user.name  guard
  printf 'v1 tool\n'      > "$R/harness/tools/a_tool.sh"
  printf 'v1 exec\n'      > "$R/harness/tools/an_exec.sh";  chmod 755 "$R/harness/tools/an_exec.sh"
  printf 'v1 template\n'  > "$R/harness/phase_template/a_phase.sh"
  printf 'v1 gated\n'     > "$R/harness/phase_template/cleanup_gated.sh"
  printf 'v1 arm\n'       > "$R/harness/arms/an_arm.sh"
  printf '# allow nothing\n' > "$R/harness/tools/harness_drift_allowlist.txt"
  cp -f "$DRIFT" "$R/harness/tools/harness_drift_check.sh"
  cp -f "$SYNC"  "$R/harness/tools/harness_sync.sh"
  git -C "$R" add -A >/dev/null 2>&1
  git -C "$R" commit -q -m v1 >/dev/null 2>&1
  local v1; v1=$(git -C "$R" rev-parse HEAD)
  printf 'v2 tool CHANGED\n'     > "$R/harness/tools/a_tool.sh"
  printf 'v2 exec CHANGED\n'     > "$R/harness/tools/an_exec.sh"
  printf 'v2 template CHANGED\n' > "$R/harness/phase_template/a_phase.sh"
  printf 'v2 gated CHANGED\n'    > "$R/harness/phase_template/cleanup_gated.sh"
  printf 'v2 arm CHANGED\n'      > "$R/harness/arms/an_arm.sh"
  git -C "$R" add -A >/dev/null 2>&1
  git -C "$R" commit -q -m v2 >/dev/null 2>&1
  local v2; v2=$(git -C "$R" rev-parse HEAD)
  # the task dir starts life at v1, the way a task dir always does
  mkdir -p "$T/tools/phase_template" "$T/merge-h"
  git -C "$R" cat-file blob "$v1:harness/tools/a_tool.sh"              > "$T/tools/a_tool.sh"
  git -C "$R" cat-file blob "$v1:harness/tools/an_exec.sh"             > "$T/tools/an_exec.sh"
  git -C "$R" cat-file blob "$v1:harness/phase_template/a_phase.sh"    > "$T/tools/phase_template/a_phase.sh"
  git -C "$R" cat-file blob "$v1:harness/phase_template/cleanup_gated.sh" > "$T/merge-h/cleanup_gated.sh"
  git -C "$R" cat-file blob "$v1:harness/arms/an_arm.sh"               > "$T/merge-h/an_arm.sh"
  cp -f "$DRIFT" "$T/tools/harness_drift_check.sh"
  cp -f "$SYNC"  "$T/tools/harness_sync.sh"
  cp -f "$R/harness/tools/harness_drift_allowlist.txt" "$T/tools/harness_drift_allowlist.txt"
  echo "$R $T $v1 $v2"
}
runsync () {  # $1=repo $2=task $3=stub-or-'-' ; rest = argv for harness_sync.sh
  local R=$1 T=$2 stub=$3; shift 3
  HARNESS_REPO="$R" HARNESS_TASK_DIR="$T" HARNESS_SQUEUE="$stub" \
    bash "$T/tools/harness_sync.sh" "$@" 2>&1
}
mkstub () {   # $1=path $2=body-line...  a squeue that prints exactly what it is told
  local p=$1; shift
  { echo '#!/usr/bin/env bash'; for l in "$@"; do printf 'echo %q\n' "$l"; done; } > "$p"
  chmod 755 "$p"
}

# ---- A: a sync onto a clean tree -------------------------------------------
read -r RA TA V1A V2A < <(mkfixture A)
mkstub "$WORK/A_squeue"                      # no PENDING jobs at all
LOGA=$WORK/A.log
runsync "$RA" "$TA" "$WORK/A_squeue" "$V2A" > "$LOGA" 2>&1; rcA=$?
[ "$rcA" -eq 0 ] && ok "A: sync onto a clean tree rc=0" || { bad "A: sync rc=$rcA"; sed 's/^/      /' "$LOGA"; }
amiss=0
for pair in "tools/a_tool.sh|harness/tools/a_tool.sh" \
            "tools/an_exec.sh|harness/tools/an_exec.sh" \
            "tools/phase_template/a_phase.sh|harness/phase_template/a_phase.sh" \
            "merge-h/cleanup_gated.sh|harness/phase_template/cleanup_gated.sh" \
            "merge-h/an_arm.sh|harness/arms/an_arm.sh"; do
  t=${pair%%|*}; w=${pair##*|}
  git -C "$RA" cat-file blob "$V2A:$w" > "$WORK/A.blob" 2>/dev/null
  if cmp -s "$WORK/A.blob" "$TA/$t"; then :; else bad "A: $t is not $V2A:$w after the sync"; amiss=1; fi
done
[ "$amiss" -eq 0 ] && ok "A: every mapped file is the commit's own bytes, merge-h basename rule included"
grep -q 'SYNC installed tools/a_tool.sh' "$LOGA" \
  && ok "A: the sync NAMES each file it installed" || bad "A: no per-file installed row"
[ -x "$TA/tools/an_exec.sh" ] && ok "A: the executable bit survived the rename-install" \
                              || bad "A: an_exec.sh lost its mode"
[ "$(cat "$TA/tools/.harness_synced_commit" 2>/dev/null)" = "$V2A" ] \
  && ok "A: the record names the synced commit" || bad "A: record is '$(cat "$TA/tools/.harness_synced_commit" 2>/dev/null)' not $V2A"
grep -q 'DRIFT CLEAN' "$LOGA" && ok "A: the drift line printed from the NEW state is CLEAN" \
                              || { bad "A: no DRIFT CLEAN line"; sed 's/^/      /' "$LOGA"; }
HARNESS_DRIFT_ALLOWLIST=$TA/tools/harness_drift_allowlist.txt \
  bash "$TA/tools/harness_drift_check.sh" "$V2A" "$TA" "$RA" >/dev/null 2>&1 \
  && ok "A: an INDEPENDENT drift check agrees the task dir is $V2A" \
  || bad "A: the independent drift check still refuses after a sync"

# ---- B: a PENDING job pinned to the older commit ---------------------------
read -r RB TB V1B V2B < <(mkfixture B)
mkdir -p "$TB/lane1"; printf '%s\n' "$V1B" > "$TB/lane1/HARNESS_COMMIT"
mkstub "$WORK/B_squeue" "9999999 lane1-relock"
BEFORE_B=$(md5sum "$TB/tools/a_tool.sh" | awk '{print $1}')
LOGB=$WORK/B.log
runsync "$RB" "$TB" "$WORK/B_squeue" "$V2B" > "$LOGB" 2>&1; rcB=$?
[ "$rcB" -eq 4 ] && ok "B: a PENDING job pinned to $V1B makes the sync refuse rc 4" \
                 || { bad "B: rc=$rcB, wanted 4"; sed 's/^/      /' "$LOGB"; }
grep -q '9999999' "$LOGB" && ok "B: the refusal NAMES the job id" || bad "B: the refusal does not name 9999999"
grep -q 'lane1/HARNESS_COMMIT' "$LOGB" && ok "B: the refusal names the pin file it read" \
                                       || bad "B: the refusal does not name the pin file"
[ "$(md5sum "$TB/tools/a_tool.sh" | awk '{print $1}')" = "$BEFORE_B" ] \
  && ok "B: NOTHING was written -- the queue is consulted before the first byte" \
  || bad "B: the sync wrote a task copy and THEN refused"
runsync "$RB" "$TB" "$WORK/B_squeue" "$V2B" --force > "$WORK/B2.log" 2>&1; rcB2=$?
[ "$rcB2" -eq 4 ] && ok "B: --force WITHOUT a reason is still a refusal" || bad "B: bare --force returned $rcB2"
[ "$(md5sum "$TB/tools/a_tool.sh" | awk '{print $1}')" = "$BEFORE_B" ] \
  && ok "B: bare --force wrote nothing either" || bad "B: bare --force wrote task copies"
runsync "$RB" "$TB" "$WORK/B_squeue" "$V2B" --force --reason "guard fixture" > "$WORK/B3.log" 2>&1; rcB3=$?
[ "$rcB3" -eq 0 ] && ok "B: --force --reason proceeds" || { bad "B: --force --reason rc=$rcB3"; sed 's/^/      /' "$WORK/B3.log"; }
grep -q 'SYNC FORCED .*reason=guard fixture' "$WORK/B3.log" \
  && ok "B: the forced sync prints the reason for the log row" || bad "B: no SYNC FORCED reason line"

# ---- C: an in-place edit is named by --check --------------------------------
printf 'somebody hand-edited this\n' >> "$TA/tools/a_tool.sh"
runsync "$RA" "$TA" "$WORK/A_squeue" --check > "$WORK/C.log" 2>&1; rcC=$?
[ "$rcC" -eq 3 ] && ok "C: --check refuses rc 3 after an in-place edit" \
                 || { bad "C: --check rc=$rcC, wanted 3"; sed 's/^/      /' "$WORK/C.log"; }
grep -q 'SYNC CHECK EDITED   tools/a_tool.sh' "$WORK/C.log" \
  && ok "C: --check NAMES the edited file" || bad "C: --check did not name tools/a_tool.sh"
grep -q 'harness_sync.sh <commit>' "$WORK/C.log" \
  && ok "C: --check says what to run, not just 'drift'" || bad "C: --check prints no fix command"
runsync "$RA" "$TA" "$WORK/A_squeue" --check > "$WORK/C0.log" 2>&1
grep -q 'no commit given' "$WORK/C0.log" && bad "C: --check lost the recorded commit" \
  || ok "C: --check with no argument used the recorded commit, not an argument"
runsync "$RA" "$TA" "$WORK/A_squeue" "$V2A" > "$WORK/C2.log" 2>&1; rcC2=$?
runsync "$RA" "$TA" "$WORK/A_squeue" --check > "$WORK/C3.log" 2>&1; rcC3=$?
[ "$rcC2" -eq 0 ] && [ "$rcC3" -eq 0 ] \
  && ok "C: re-syncing repairs the edit and --check goes quiet again" \
  || { bad "C: repair rc=$rcC2 recheck rc=$rcC3"; sed 's/^/      /' "$WORK/C3.log"; }

# ---- D: THE MUTATION -- the pre-fix world, pinned to $PREFIX ----------------
if git -C "$REPO" cat-file -e "$PREFIX:harness/tools/harness_sync.sh" 2>/dev/null; then
  bad "D: $PREFIX already carries harness_sync.sh -- the pin is not the pre-fix world"
else
  ok "D: at the pinned pre-fix commit $PREFIX there is no harness_sync.sh at all"
fi
read -r RD TD V1D V2D < <(mkfixture D)
mkdir -p "$TD/lane1"; printf '%s\n' "$V1D" > "$TD/lane1/HARNESS_COMMIT"
BEFORE_D=$(md5sum "$TD/tools/a_tool.sh" | awk '{print $1}')
# The pre-fix procedure, verbatim from what the drift refusal printed:
#   git -C <repo> cat-file blob <sha>:<repo path> > <task dir>/<task path>
git -C "$RD" cat-file blob "$V2D:harness/tools/a_tool.sh" > "$TD/tools/a_tool.sh" 2> "$WORK/D.log"
rcD=$?
[ "$rcD" -eq 0 ] && ok "D: the pre-fix bare cat-file succeeded -- rc 0, no refusal to give" \
                 || bad "D: the pre-fix procedure failed for an unrelated reason"
[ -s "$WORK/D.log" ] && bad "D: the pre-fix procedure printed something -- it should be silent" \
                     || ok "D: the pre-fix procedure said NOTHING about the queued job"
[ "$(md5sum "$TD/tools/a_tool.sh" | awk '{print $1}')" != "$BEFORE_D" ] \
  && ok "D: it moved the task copy anyway, stranding lane1 (pinned $V1D) -- ticks 499-501" \
  || bad "D: the pre-fix procedure did not move the file, so the arm proves nothing"
grep -q "$V1D" "$TD/lane1/HARNESS_COMMIT" \
  && ok "D: and lane1's pin still names the OLD commit nobody told it about" \
  || bad "D: the fixture pin is not what the arm assumes"

# ---- E: cut the queue block out and arm B must go red -----------------------
MUT=$WORK/harness_sync_mutant.sh
awk '/^# ---- the queue, BEFORE a single byte is written/ {skip=1}
     /^# ---- install: temp file in the TARGET dir/       {skip=0}
     !skip' "$SYNC" > "$MUT"
if bash -n "$MUT" 2>/dev/null && ! grep -q 'SYNC REFUSED (rc 4)' "$MUT"; then
  read -r RE TE V1E V2E < <(mkfixture E)
  mkdir -p "$TE/lane1"; printf '%s\n' "$V1E" > "$TE/lane1/HARNESS_COMMIT"
  cp -f "$MUT" "$TE/tools/harness_sync.sh"
  mkstub "$WORK/E_squeue" "9999999 lane1-relock"
  HARNESS_REPO="$RE" HARNESS_TASK_DIR="$TE" HARNESS_SQUEUE="$WORK/E_squeue" \
    bash "$TE/tools/harness_sync.sh" "$V2E" > "$WORK/E.log" 2>&1; rcE=$?
  [ "$rcE" -ne 4 ] && ok "E: with the queue block cut out the sync does NOT refuse (rc=$rcE) -- arm B can fail" \
                   || bad "E: the mutant STILL refused rc 4 -- arm B is not testing the queue block"
else
  bad "E: could not build the queue-block mutant -- MUTATION ARM DID NOT RUN"
fi

# ---- F: static -- one mapping, and the templates call --check ---------------
( REPO=$REPO SHA=$PREFIX HARNESS_DRIFT_LIB=1 . "$DRIFT" >/dev/null 2>&1
  command -v map_of >/dev/null 2>&1 && command -v harness_is_evidence >/dev/null 2>&1 ) \
  && ok "F: harness_drift_check.sh sources as a library and defines the mapping" \
  || bad "F: HARNESS_DRIFT_LIB=1 . harness_drift_check.sh defines no map_of"
grep -qE '^\s*map_of \(\)' "$SYNC" \
  && bad "F: harness_sync.sh has its OWN map_of -- the table is duplicated" \
  || ok "F: harness_sync.sh defines no map_of; it sources the checker's table"
for tpl in phaseN_relock.sh phaseN_cert.sh; do
  P=$REPO/harness/phase_template/$tpl
  if [ -f "$P" ] && awk '/### HARNESS-DRIFT/{d=1} d && /harness_sync.sh/ && /--check/{found=1} END{exit !found}' "$P"; then
    ok "F: $tpl runs harness_sync.sh --check inside its drift block"
  else
    bad "F: $tpl does not run harness_sync.sh --check in its drift block"
  fi
done

LIVE_AFTER=$( [ -f "$LIVE_RECORD" ] && md5sum "$LIVE_RECORD" | awk '{print $1}' || echo absent )
[ "$LIVE_BEFORE" = "$LIVE_AFTER" ] && ok "live task dir untouched: $LIVE_RECORD md5 $LIVE_BEFORE" \
                                   || bad "THE GUARD MOVED THE LIVE RECORD: $LIVE_BEFORE -> $LIVE_AFTER"

echo "### harness_sync_guard: pass=$pass fail=$fail"
[ "$fail" -eq 0 ] || exit 1
exit 0
