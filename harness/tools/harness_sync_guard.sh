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
#   G  DET-1-1, THE ONE THAT COST A RELOCK: a pin is written, a sync then moves
#      the record, and the SUBMIT HELPER re-resolves -- `--write <job root>` with
#      NO sha -- so the job's pin equals the SYNCED commit and not the value the
#      lane was carrying. A sha that is NOT the synced one is refused rc 3 and
#      names both; `--allow-older --reason` proceeds and prints the reason.
#      MUTATION, pinned to $PREFIX_RESOLVE: the pre-fix `--write <jr> <sha>`
#      writes the STALE pin, rc 0, silently -- which is 5981194 exactly.
#   H  DET-1-1's other half: after a sync, the PIN REPORT names every job root
#      whose pin is not the synced commit, classed PENDING / RUNNING / no-job,
#      each with its actuator. The rc-4 refusal is a PRE-condition and cannot see
#      a job submitted 99 s LATER; this line is what a lane reads afterwards.
#   I  HARNESS-SYNC-3: a RUNNING job of ours whose read set contains a file this
#      sync would INSTALL makes the sync REFUSE rc 6 BEFORE the first byte, the
#      destination md5 unchanged and still not the commit's. A disjoint read set
#      installs. A job whose sbatch cannot be found reads EVERYTHING. `--force`
#      alone still refuses; `--force --reason` proceeds and leaves the marker.
#      MUTATION: with the read-set check cut out, the same fixture INSTALLS the
#      file arm a1 saved -- without which a1 cannot fail.
#   L  HARNESS-SYNC-4: a PENDING job of ours reads too, and NOTHING saw it. An
#      `afterany` cleanup owner submitted with `sbatch --wrap 'bash <task
#      dir>/merge-h/cleanup_gated.sh …'` (6001240 is one, live) owns no PIN DIR,
#      so rc 4 misses it, and is not RUNNING, so arm I's rc 6 missed it too: a
#      sync landing between its parent's end and its own start rewrote the script
#      it would then run, silently. So the read set is computed for state PD as
#      well as R, refused with the SAME rc 6 and a `state=` field on every row.
#      A DISJOINT PENDING read set still installs (b2) -- otherwise a task dir
#      holding 25 queued jobs could never be synced again. MUTATION: cutting the
#      PD half of the filter (`-t R`) makes the queued job invisible again and
#      the same fixture INSTALLS, so b4 can fail. And the `--running-list` state
#      column is REQUIRED (b6): a stale four-field row is a FATAL rc 2, because
#      read one field out of step it yields an EMPTY read set and a silent
#      install -- the very defect this block exists to prevent.
#   M  HARNESS-SYNC-4-1: rc 4 exited where it stood, UPSTREAM of the read-set
#      computation, so when both refusals applied the operator was shown only
#      the rc-4 rows, told to drain the queued job, and handed rc 6 on the
#      re-run -- two serial waits for one queue state (MERGE-U's landing,
#      2026-09-06T22:41). Both sets are now computed and BOTH printed, with the
#      exit decided afterwards and rc 4 taking precedence. m1 asserts both row
#      families in ONE rc-4 run; m2 is the non-vacuity control (rc 6 alone is
#      still 6 and prints no rc-4 row); m3 is the MUTATION -- restore the early
#      `exit 4` and m1's rc-6 rows vanish, so m1 can fail. m4 covers the
#      `--force` summary's JOB COUNT, which no arm had ever read: HARNESS-SYNC-4
#      replaced an `awk '{print $3}'` there that printed the literal REFUSED for
#      every input, in the line the operator is told to copy into a lane log row.
#   N  HARNESS-SYNC-6: rc 4 is the PIN **AND** THE READ SET. A pin-mismatched
#      PENDING job that reads only its OWN job root (the HARNESS-SYNC-5 owner
#      snapshot) is TOLERATED on a named row instead of refusing (n1); one that
#      reads an installed file still refuses rc 4 with its rc-6 rows beside it
#      (n2); one whose read set is undeterminable, or was never examined, still
#      refuses (n3, n3b -- law 9). n4 is the mutation.
#   R  HARNESS-SYNC-7: a path built from LITERAL variables is a KNOWN path.
#      det162-proof 6015646 blocked the queue as a RUNNING rc-6 hit on two files
#      it never read -- `WT=<literal>`, `WTH=$WT/harness`,
#      `MOVED_ROWS=$WTH/tools/moved_row_halves.sh` -- a chain whose value is in
#      the text. r1 is that shape, resolved, installing, and PRINTING the chain
#      and the absolute path; r2 is the control (the same shape landing INSIDE
#      the task tools/ still refuses rc 6 match=exact); r3 keeps a `$( )` hop
#      unresolved and refusing (law 9: the expansion adds resolutions, it does
#      not invent them); r4 is the mutation; r5 is the REGRESSION ARM for this
#      lane's own first-cut defect -- `$(dirname "$0")` in a SLURM SNAPSHOT is
#      NOT the job's $0, and substituting it cleared TEN live refusals against a
#      /tmp path in the 07:16 dry run.
#   J  static: the header no longer claims the rename is "the only safe way to
#      write into a live task dir" -- it is safe only when renamer and reader are
#      the SAME NFS client -- and the PIN REPORT no longer calls a RUNNING job
#      SAFE on the strength of its drift gate.
#   F  static: the mapping is SOURCED from the drift check, not copied -- the
#      check defines map_of when sourced with HARNESS_DRIFT_LIB, and the writer
#      contains no map_of of its own; and both phase templates run `--check`
#      as the first line of their drift block.
#   S  HARNESS-SYNC-8: THE FORCE PATH AGAINST ITS OWN POST-CONDITIONS. A
#      fixture force, then BOTH gates (`--check` and harness_drift_check.sh) run
#      on the task dir the force left, plus the push-lag fallback and the
#      read-only mode's ABSENT report. Every arm before this one asserted what
#      --force DOES; none asserted the state it LEAVES, which is how the marker
#      it writes came to red-line both gates permanently. See the block above
#      the arm for the measured rows.
#   T  HARNESS-SYNC-9: A NO-OP SYNC MUST NOT REFUSE ITSELF. With the task dir
#      already AT the commit the install set is EMPTY, the read-set block never
#      runs, and rc 4 read the resulting empty read set as "not examined" and
#      refused -- 21 rows over three jobs on B30's landing, nothing moved. An
#      empty install set is now its own verdict: rc 0, a NO-OP row, the pin
#      TOLERATED on reason=nothing-installed. t3 is the non-vacuity control and
#      t4 the blobless hole (an empty install set is not a no-op when a mapped
#      file has no blob -- that run still owes rc 5).
#
# THE MUTATION IS PINNED TO A COMMIT CONSTANT, NEVER `HEAD`: an arm that reads
# `HEAD:<the file it guards>` starts asserting the fix against itself the moment
# the fix is committed.
set -uo pipefail
export PATH=/users/glvov/.pixi/bin:/users/glvov/.local/bin:$PATH

PREFIX=9a154a6857997f2c48707252a267d341dfccc009   # HARNESS-EXIT-3's tip: before this lane
# DET-1's own last harness commit: `--write` exists there and takes the sha ON
# TRUST, which is the defect arm G's mutation reproduces.
PREFIX_RESOLVE=873263ff429af36fb8be1259f681597f99533fda

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
  cp -f "$(dirname -- "$SYNC")/harness_push_check.sh" "$R/harness/tools/harness_push_check.sh" 2>/dev/null
  cp -f "$(dirname -- "$SYNC")/script_refs.sh" "$R/harness/tools/script_refs.sh" 2>/dev/null
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
  cp -f "$(dirname -- "$SYNC")/harness_push_check.sh" "$T/tools/harness_push_check.sh" 2>/dev/null
  cp -f "$(dirname -- "$SYNC")/script_refs.sh" "$T/tools/script_refs.sh" 2>/dev/null
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

# ---- A2: a NEW file enters only when it is NAMED ---------------------------
# `harness/arms/an_arm.sh` is in the commit and reachable through the merge-h
# basename rule, but `merge-h/a_new_arm.sh` does not exist in the task dir. A
# sync must NOT invent it, and `--add` must create it -- from this writer, never
# from a bare cat-file.
printf 'v2 brand new\n' > "$RA/harness/tools/a_new_tool.sh"
git -C "$RA" add -A >/dev/null 2>&1; git -C "$RA" commit -q -m v3 >/dev/null 2>&1
V3A=$(git -C "$RA" rev-parse HEAD)
runsync "$RA" "$TA" "$WORK/A_squeue" "$V3A" > "$WORK/A2.log" 2>&1
[ -e "$TA/tools/a_new_tool.sh" ] && bad "A2: the sync INVENTED a task file nobody named" \
  || ok "A2: a file the commit carries but the task dir lacks is NOT invented"
grep -q 'SYNC ABSENT' "$WORK/A2.log" && grep -q -- '--add <task path>' "$WORK/A2.log" \
  && ok "A2: the absent report names the actuator (--add), it is not a dead notice" \
  || { bad "A2: the absent report has no actuator"; sed 's/^/      /' "$WORK/A2.log"; }
runsync "$RA" "$TA" "$WORK/A_squeue" "$V3A" --add tools/a_new_tool.sh > "$WORK/A3.log" 2>&1; rcA3=$?
git -C "$RA" cat-file blob "$V3A:harness/tools/a_new_tool.sh" > "$WORK/A3.blob"
[ "$rcA3" -eq 0 ] && cmp -s "$WORK/A3.blob" "$TA/tools/a_new_tool.sh" \
  && ok "A2: --add creates the named file from the commit's own bytes" \
  || { bad "A2: --add rc=$rcA3 did not create the file correctly"; sed 's/^/      /' "$WORK/A3.log"; }
runsync "$RA" "$TA" "$WORK/A_squeue" "$V3A" --add tools/no_such_thing.sh > "$WORK/A4.log" 2>&1; rcA4=$?
[ "$rcA4" -eq 2 ] && ok "A2: --add of a path the commit does not carry is FATAL, not invented" \
                  || bad "A2: --add of a nonexistent blob returned $rcA4"
V2A=$V3A                      # arm C compares against the record, which now says v3

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

# ---- G: DET-1-1 -- the pin is RESOLVED AT SUBMIT, not copied ----------------
# The sequence that lost MERGE-T's relock, in a fixture: lane writes a pin at v1
# (14:0x), a sync moves the task dir to v2 (14:06), the lane submits (14:07:59).
RESOLVE=$HERE/harness_commit_resolve.sh
[ -f "$RESOLVE" ] || RESOLVE=$REPO/harness/tools/harness_commit_resolve.sh
if [ ! -f "$RESOLVE" ]; then
  bad "G: no harness_commit_resolve.sh at $HERE or $REPO/harness/tools -- ARM DID NOT RUN"
else
read -r RG TG V1G V2G < <(mkfixture G)
mkstub "$WORK/G_squeue"                       # nothing queued: the sync is free to move
JRG=$TG/laneG
mkdir -p "$JRG"
wr () {  # run the submit helper against the FIXTURE task dir and repo
  HARNESS_REPO="$RG" HARNESS_TASK_DIR="$TG" bash "$RESOLVE" --write "$@" 2>&1
}
# 1. the lane writes its pin while the task dir is still v1
wr "$JRG" "$V1G" > "$WORK/G1.log" 2>&1; rcG1=$?
[ "$rcG1" -eq 0 ] && [ "$(cat "$JRG/HARNESS_COMMIT")" = "$V1G" ] \
  && ok "G: the pin is written at v1 while the task dir IS v1" \
  || { bad "G: could not write the v1 pin (rc=$rcG1)"; sed 's/^/      /' "$WORK/G1.log"; }
# 2. a sync moves the record to v2 -- exactly DET-1's 14:06:19 install
runsync "$RG" "$TG" "$WORK/G_squeue" "$V2G" > "$WORK/G2.log" 2>&1; rcG2=$?
[ "$rcG2" -eq 0 ] && [ "$(cat "$TG/tools/.harness_synced_commit")" = "$V2G" ] \
  && ok "G: the sync moved the record to v2 with nothing queued to refuse for" \
  || { bad "G: the sync rc=$rcG2 did not move the record"; sed 's/^/      /' "$WORK/G2.log"; }
# 3. THE FIX: re-resolving at submit gives the SYNCED commit, not the carried one
wr "$JRG" > "$WORK/G3.log" 2>&1; rcG3=$?
if [ "$rcG3" -eq 0 ] && [ "$(cat "$JRG/HARNESS_COMMIT")" = "$V2G" ]; then
  ok "G: --write with NO sha RESOLVES the pin at submit -- the job's pin is the synced commit $V2G"
else
  bad "G: re-resolve rc=$rcG3 left the pin at '$(cat "$JRG/HARNESS_COMMIT")', wanted $V2G"
  sed 's/^/      /' "$WORK/G3.log"
fi
grep -q 'resolved at submit' "$WORK/G3.log" \
  && ok "G: and it says WHERE the answer came from (the record), not just the sha" \
  || bad "G: the resolve printed no provenance row"
# 4. a CARRIED sha -- the copied stale pin -- is refused rc 3 and names both
printf '%s\n' "$V2G" > "$JRG/HARNESS_COMMIT"
wr "$JRG" "$V1G" > "$WORK/G4.log" 2>&1; rcG4=$?
[ "$rcG4" -eq 3 ] && ok "G: a pin that is NOT the synced commit is REFUSED rc 3 at submit" \
                  || { bad "G: a stale pin gave rc=$rcG4, wanted 3"; sed 's/^/      /' "$WORK/G4.log"; }
grep -q "$V1G" "$WORK/G4.log" && grep -q "$V2G" "$WORK/G4.log" \
  && ok "G: the refusal names BOTH the asked-for pin and what the task dir IS" \
  || bad "G: the refusal does not name both shas"
[ "$(cat "$JRG/HARNESS_COMMIT")" = "$V2G" ] \
  && ok "G: the refusal wrote NOTHING -- the pin file still names the synced commit" \
  || bad "G: the refusal overwrote the pin anyway"
# 5. --allow-older WITHOUT a reason is still a refusal; with one it proceeds
wr "$JRG" "$V1G" --allow-older > "$WORK/G5.log" 2>&1; rcG5=$?
[ "$rcG5" -eq 3 ] && ok "G: --allow-older WITHOUT --reason is still a refusal" \
                  || bad "G: bare --allow-older returned $rcG5"
wr "$JRG" "$V1G" --allow-older --reason "guard fixture: rerunning an old job shape" > "$WORK/G6.log" 2>&1; rcG6=$?
[ "$rcG6" -eq 0 ] && [ "$(cat "$JRG/HARNESS_COMMIT")" = "$V1G" ] \
  && ok "G: --allow-older --reason proceeds and pins the older commit deliberately" \
  || { bad "G: --allow-older --reason rc=$rcG6"; sed 's/^/      /' "$WORK/G6.log"; }
grep -q 'ALLOWED-OLDER .*reason=guard fixture' "$WORK/G6.log" \
  && ok "G: and it prints the reason for the lane log row" || bad "G: no ALLOWED-OLDER reason line"
# 6. THE MUTATION: the pre-fix writer, pinned to a commit constant
PREW=$WORK/harness_commit_resolve_prefix.sh
if git -C "$REPO" cat-file blob "$PREFIX_RESOLVE:harness/tools/harness_commit_resolve.sh" > "$PREW" 2>/dev/null; then
  if grep -q 'DET-1-1' "$PREW"; then
    bad "G: $PREFIX_RESOLVE already carries the DET-1-1 fix -- the pin is not the pre-fix world"
  else
    ok "G: the pinned pre-fix writer $PREFIX_RESOLVE takes the sha on trust (no DET-1-1 block)"
  fi
  printf '%s\n' "$V2G" > "$JRG/HARNESS_COMMIT"
  HARNESS_REPO="$RG" HARNESS_TASK_DIR="$TG" bash "$PREW" --write "$JRG" "$V1G" > "$WORK/G7.log" 2>&1
  rcG7=$?
  if [ "$rcG7" -eq 0 ] && [ "$(cat "$JRG/HARNESS_COMMIT")" = "$V1G" ]; then
    ok "G: MUTATION -- the pre-fix writer stamped the STALE pin $V1G, rc 0, over a task dir that is $V2G (5981194)"
  else
    bad "G: the pre-fix writer gave rc=$rcG7 pin='$(cat "$JRG/HARNESS_COMMIT")' -- the mutation did not reproduce"
  fi
  grep -qi 'refus' "$WORK/G7.log" && bad "G: the pre-fix writer refused something -- it is not the pre-fix world" \
                                  || ok "G: and it refused NOTHING, which is why nobody saw it"
else
  bad "G: could not read $PREFIX_RESOLVE:harness/tools/harness_commit_resolve.sh -- MUTATION ARM DID NOT RUN"
fi
fi

# ---- H: DET-1-1 -- the PIN REPORT the sync prints afterwards ----------------
# Three job roots, three classes: one PENDING (forced through), one RUNNING
# (safe), one with no job at all (a stale pin from a finished lane).
read -r RH TH V1H V2H < <(mkfixture H)
mkdir -p "$TH/lanepd" "$TH/lanerun" "$TH/lanedead"
printf '%s\n' "$V1H" > "$TH/lanepd/HARNESS_COMMIT"
printf '%s\n' "$V1H" > "$TH/lanerun/HARNESS_COMMIT"
printf '%s\n' "$V1H" > "$TH/lanedead/HARNESS_COMMIT"
# The stub answers BOTH shapes the sync asks for: `-t PD -o '%i %j'` for the
# refusal, and `-o '%i %j %T'` for the report.
cat > "$WORK/H_squeue" <<'EOSTUB'
#!/usr/bin/env bash
pd=0; for a in "$@"; do [ "$a" = PD ] && pd=1; done
if [ "$pd" = 1 ]; then
  echo "7000001 lanepd-relock"
else
  echo "7000001 lanepd-relock PENDING"
  echo "7000002 lanerun-relock RUNNING"
fi
EOSTUB
chmod 755 "$WORK/H_squeue"
runsync "$RH" "$TH" "$WORK/H_squeue" "$V2H" --force --reason "guard fixture H" > "$WORK/H.log" 2>&1; rcH=$?
[ "$rcH" -eq 0 ] && ok "H: the forced sync completed so there is a post-sync state to report" \
                 || { bad "H: forced sync rc=$rcH"; sed 's/^/      /' "$WORK/H.log"; }
grep -q "PENDING  $TH/lanepd/HARNESS_COMMIT" "$WORK/H.log" \
  && ok "H: the pin report names the PENDING job root that will die at its drift gate" \
  || { bad "H: no PENDING row for lanepd"; grep '^### SYNC PIN' "$WORK/H.log" | sed 's/^/      /'; }
grep -qE "RUNNING .*$TH/lanerun/HARNESS_COMMIT.*past its drift gate" "$WORK/H.log" \
  && ok "H: a RUNNING job's stale pin is named for the record, drift only, not as a refusal" \
  || { bad "H: no RUNNING/drift-gate row for lanerun"; grep '^### SYNC PIN' "$WORK/H.log" | sed 's/^/      /'; }
grep -q "STALE    $TH/lanedead/HARNESS_COMMIT" "$WORK/H.log" \
  && ok "H: a pin with no job of ours behind it is named STALE" \
  || { bad "H: no STALE row for lanedead"; grep '^### SYNC PIN' "$WORK/H.log" | sed 's/^/      /'; }
grep -q 'rewrite AT SUBMIT: harness_commit_resolve.sh --write' "$WORK/H.log" \
  && ok "H: the STALE row carries its actuator -- rewrite at submit, not 'stale'" \
  || bad "H: the STALE row is a dead notice with no actuator"
grep -qE '^### SYNC PIN SUMMARY .*pending=1 running=1 stale=1' "$WORK/H.log" \
  && ok "H: the summary partitions the pins exactly: pending=1 running=1 stale=1" \
  || { bad "H: the pin summary does not partition"; grep 'SYNC PIN SUMMARY' "$WORK/H.log" | sed 's/^/      /'; }
# and the mutation for THIS half: arm A synced with no pin dirs at all and must
# say so rather than printing nothing, or the report cannot be read as evidence.
grep -q 'SYNC PIN REPORT' "$LOGA" && grep -q 'every pin file names' "$LOGA" \
  && ok "H: a sync with no drifted pins says so explicitly (arm A's log)" \
  || { bad "H: arm A's sync printed no pin report at all"; }

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

# ---- I: HARNESS-SYNC-3 -- the READ SET of a RUNNING job, before the install --
# The rename-install is atomic only for readers on the RENAMING NFS client. A
# sync driven from the login node is never that client, so a RUNNING job on a
# compute node that is still READING an installed script has its inode unlinked,
# bash calls the read error EOF, and the job exits 0 with half its rows -- which
# is 5993691 and what PROOF-SMOKE-1-1 reproduced in 5995889. `--running-list` is
# the TEST-ONLY shim that feeds the check a job list instead of squeue+scontrol.
mkjob () {          # $1 = task dir; writes an sbatch + a driver that reads $2
  local T=$1 reads=$2
  mkdir -p "$T/jobroot"
  { echo '#!/bin/bash'; echo "T=$T"
    echo 'bash "$T/jobroot/driver.sh"'; } > "$T/jobroot/j.sbatch"
  { echo '#!/bin/bash'; echo "T=$T"
    echo "source \"\$T/$reads\""; } > "$T/jobroot/driver.sh"
}
# a1: the driver reads a file this sync WOULD rewrite -> rc 6, nothing written
read -r RI TI V1I V2I < <(mkfixture I)
mkstub "$WORK/I_squeue"
mkjob "$TI" tools/a_tool.sh
printf '8000001 RUNNING laneI %s %s\n' "$TI" "$TI/jobroot/j.sbatch" > "$WORK/I_run.txt"
git -C "$RI" cat-file blob "$V2I:harness/tools/a_tool.sh" > "$WORK/I.blob"
BEFORE_I=$(md5sum "$TI/tools/a_tool.sh" | awk '{print $1}')
WANT_I=$(md5sum "$WORK/I.blob" | awk '{print $1}')
[ "$BEFORE_I" != "$WANT_I" ] \
  && ok "I(a1): NON-VACUITY -- tools/a_tool.sh is $BEFORE_I and $V2I says $WANT_I, so it IS in the install set" \
  || bad "I(a1): the fixture file already matches the commit -- the arm would prove nothing"
runsync "$RI" "$TI" "$WORK/I_squeue" "$V2I" --running-list "$WORK/I_run.txt" > "$WORK/I.log" 2>&1; rcI=$?
[ "$rcI" -eq 6 ] && ok "I(a1): a RUNNING job whose driver reads an installed file makes the sync refuse rc 6" \
                 || { bad "I(a1): rc=$rcI, wanted 6"; sed 's/^/      /' "$WORK/I.log"; }
grep -q '^### SYNC REFUSED rc=6 running=8000001 state=RUNNING file=a_tool.sh' "$WORK/I.log" \
  && ok "I(a1): the refusal row NAMES the job id and the file and the STATE" \
  || { bad "I(a1): no 'SYNC REFUSED rc=6 running=8000001 state=RUNNING file=a_tool.sh' row"; sed 's/^/      /' "$WORK/I.log"; }
AFTER_I=$(md5sum "$TI/tools/a_tool.sh" | awk '{print $1}')
[ "$AFTER_I" = "$BEFORE_I" ] && [ "$AFTER_I" != "$WANT_I" ] \
  && ok "I(a1): NOTHING was written -- the file is still $AFTER_I, still not $V2I's bytes" \
  || bad "I(a1): the sync wrote the file it refused over ($BEFORE_I -> $AFTER_I)"
# a2: a RUNNING job whose read set is DISJOINT installs normally
read -r RI2 TI2 V1I2 V2I2 < <(mkfixture I2)
mkstub "$WORK/I2_squeue"
mkjob "$TI2" tools/harness_drift_check.sh          # same bytes at v1 and v2: not installed
printf '8000002 RUNNING laneI2 %s %s\n' "$TI2" "$TI2/jobroot/j.sbatch" > "$WORK/I2_run.txt"
runsync "$RI2" "$TI2" "$WORK/I2_squeue" "$V2I2" --running-list "$WORK/I2_run.txt" > "$WORK/I2.log" 2>&1; rcI2=$?
git -C "$RI2" cat-file blob "$V2I2:harness/tools/a_tool.sh" > "$WORK/I2.blob"
[ "$rcI2" -eq 0 ] && cmp -s "$WORK/I2.blob" "$TI2/tools/a_tool.sh" \
  && ok "I(a2): a RUNNING job with a DISJOINT read set does not block the sync (rc 0, installed)" \
  || { bad "I(a2): rc=$rcI2 -- a disjoint read set must not refuse"; sed 's/^/      /' "$WORK/I2.log"; }
grep -q 'SYNC REFUSED rc=6' "$WORK/I2.log" && bad "I(a2): it refused anyway" \
                                           || ok "I(a2): and it printed no rc-6 row at all"
# a3: a RUNNING job whose sbatch cannot be found reads EVERYTHING (law 9)
read -r RI3 TI3 V1I3 V2I3 < <(mkfixture I3)
mkstub "$WORK/I3_squeue"
printf '8000003 RUNNING laneI3 %s -\n' "$TI3" > "$WORK/I3_run.txt"
BEFORE_I3=$(md5sum "$TI3/tools/a_tool.sh" | awk '{print $1}')
runsync "$RI3" "$TI3" "$WORK/I3_squeue" "$V2I3" --running-list "$WORK/I3_run.txt" > "$WORK/I3.log" 2>&1; rcI3=$?
[ "$rcI3" -eq 6 ] && ok "I(a3): a RUNNING job with NO discoverable sbatch is treated as reading everything -- rc 6" \
                  || { bad "I(a3): rc=$rcI3, wanted 6"; sed 's/^/      /' "$WORK/I3.log"; }
grep -q 'reason=no-sbatch-found' "$WORK/I3.log" \
  && ok "I(a3): and the row says WHY it could not be determined" || bad "I(a3): no reason=no-sbatch-found"
[ "$(md5sum "$TI3/tools/a_tool.sh" | awk '{print $1}')" = "$BEFORE_I3" ] \
  && ok "I(a3): nothing written there either" || bad "I(a3): it wrote after refusing"
# a4: --force --reason proceeds and leaves the marker, like --allow-older does
read -r RI4 TI4 V1I4 V2I4 < <(mkfixture I4)
mkstub "$WORK/I4_squeue"
mkjob "$TI4" tools/a_tool.sh
printf '8000004 RUNNING laneI4 %s %s\n' "$TI4" "$TI4/jobroot/j.sbatch" > "$WORK/I4_run.txt"
runsync "$RI4" "$TI4" "$WORK/I4_squeue" "$V2I4" --running-list "$WORK/I4_run.txt" --force > "$WORK/I4a.log" 2>&1; rcI4a=$?
[ "$rcI4a" -eq 6 ] && ok "I(a4): --force WITHOUT --reason is still a refusal" || bad "I(a4): bare --force returned $rcI4a"
runsync "$RI4" "$TI4" "$WORK/I4_squeue" "$V2I4" --running-list "$WORK/I4_run.txt" \
        --force --reason "guard fixture I4" > "$WORK/I4.log" 2>&1; rcI4=$?
git -C "$RI4" cat-file blob "$V2I4:harness/tools/a_tool.sh" > "$WORK/I4.blob"
[ "$rcI4" -eq 0 ] && cmp -s "$WORK/I4.blob" "$TI4/tools/a_tool.sh" \
  && ok "I(a4): --force --reason proceeds and installs" \
  || { bad "I(a4): --force --reason rc=$rcI4"; sed 's/^/      /' "$WORK/I4.log"; }
MARK=$TI4/tools/.harness_synced_commit.force-readset
[ -f "$MARK" ] && grep -q 'reason=guard fixture I4' "$MARK" \
  && ok "I(a4): the override left a marker naming the reason: $(cat "$MARK")" \
  || bad "I(a4): no .force-readset marker with the reason at $MARK"
# a5: THE MUTATION -- cut the read-set check out and a1 must go RED
MUT3=$WORK/harness_sync_noreadset.sh
awk '/^# ---- THE READ SET OF EVERY RUNNING JOB, BEFORE A SINGLE BYTE/ {skip=1}
     /^# ---- install: temp file in the TARGET dir/                    {skip=0}
     !skip' "$SYNC" > "$MUT3"
if bash -n "$MUT3" 2>/dev/null && ! grep -q 'SYNC REFUSED rc=6' "$MUT3"; then
  read -r RI5 TI5 V1I5 V2I5 < <(mkfixture I5)
  mkstub "$WORK/I5_squeue"
  mkjob "$TI5" tools/a_tool.sh
  printf '8000005 RUNNING laneI5 %s %s\n' "$TI5" "$TI5/jobroot/j.sbatch" > "$WORK/I5_run.txt"
  cp -f "$MUT3" "$TI5/tools/harness_sync.sh"
  git -C "$RI5" cat-file blob "$V2I5:harness/tools/a_tool.sh" > "$WORK/I5.blob"
  HARNESS_REPO="$RI5" HARNESS_TASK_DIR="$TI5" HARNESS_SQUEUE="$WORK/I5_squeue" \
    bash "$TI5/tools/harness_sync.sh" "$V2I5" > "$WORK/I5.log" 2>&1; rcI5=$?
  if [ "$rcI5" -ne 6 ] && cmp -s "$WORK/I5.blob" "$TI5/tools/a_tool.sh"; then
    ok "I(a5): MUTATION -- with the read-set check cut out the sync rc=$rcI5 INSTALLS the file arm a1 saved, so a1 can fail"
  else
    bad "I(a5): the mutant still refused (rc=$rcI5) or did not install -- ARM a1 IS NOT TESTING THE READ-SET CHECK"
  fi
else
  bad "I(a5): could not build the read-set mutant -- MUTATION ARM DID NOT RUN"
fi

# a1b: a `--wrap` shaped script -- ONE bash line, no driver -- is parsed too.
# 5992050 is exactly this shape: BatchFlag=1, Command=(null), one `bash <path>`.
read -r RI6 TI6 V1I6 V2I6 < <(mkfixture I6)
mkstub "$WORK/I6_squeue"
mkdir -p "$TI6/jobroot"
{ echo '#!/bin/sh'; echo '# This script was created by sbatch --wrap.'; echo
  echo "bash $TI6/tools/a_tool.sh arg1 arg2"; } > "$TI6/jobroot/wrap.sh"
printf '8000006 RUNNING laneI6 %s %s\n' "$TI6" "$TI6/jobroot/wrap.sh" > "$WORK/I6_run.txt"
runsync "$RI6" "$TI6" "$WORK/I6_squeue" "$V2I6" --running-list "$WORK/I6_run.txt" > "$WORK/I6.log" 2>&1; rcI6=$?
[ "$rcI6" -eq 6 ] && grep -q 'file=a_tool.sh' "$WORK/I6.log" \
  && ok "I(a1b): a --wrap one-liner naming an installed file refuses rc 6 too (the 5992050 shape)" \
  || { bad "I(a1b): rc=$rcI6 -- the --wrap shape was not read"; sed 's/^/      /' "$WORK/I6.log"; }

# a1c: a reference reached through a COMMAND SUBSTITUTION. 5992569's own sbatch
# opens `HC=$(bash "$T/tools/harness_commit_resolve.sh" "$P")`; the first cut
# required whitespace or line-start before the verb, so `$(bash` did not match
# and harness_commit_resolve.sh silently left that job's read set. Found by
# applying the rule by hand before syncing with it.
read -r RI7 TI7 V1I7 V2I7 < <(mkfixture I7)
mkstub "$WORK/I7_squeue"
mkdir -p "$TI7/jobroot"
{ echo '#!/bin/bash'; echo "HC=\$(bash \"$TI7/tools/a_tool.sh\" arg) || exit 6"; } > "$TI7/jobroot/sub.sbatch"
printf '8000007 RUNNING laneI7 %s %s\n' "$TI7" "$TI7/jobroot/sub.sbatch" > "$WORK/I7_run.txt"
runsync "$RI7" "$TI7" "$WORK/I7_squeue" "$V2I7" --running-list "$WORK/I7_run.txt" > "$WORK/I7.log" 2>&1; rcI7=$?
[ "$rcI7" -eq 6 ] && grep -q 'file=a_tool.sh' "$WORK/I7.log" \
  && ok "I(a1c): a reference inside \$( ) is in the read set too (5992569's harness_commit_resolve.sh line)" \
  || { bad "I(a1c): rc=$rcI7 -- a command substitution hid the reference"; sed 's/^/      /' "$WORK/I7.log"; }

# a1d: the BARE DOT, both ways. det1_proof2.sh sources ONE file through a
# variable (`FAST_ENV=$T/tools/retread_fast_env.sh` ... `. "$FAST_ENV"`) and
# writes `$(grep -c . "$OB")` four times. A pattern that takes both makes that
# job undeterminable -- and an undeterminable job refuses EVERY sync for its
# whole run, which is a rule nobody can work with.
read -r RI8 TI8 V1I8 V2I8 < <(mkfixture I8)
mkstub "$WORK/I8_squeue"
mkdir -p "$TI8/jobroot"
{ echo '#!/bin/bash'; echo "FAST_ENV=$TI8/tools/a_tool.sh"; echo '. "$FAST_ENV"'; } > "$TI8/jobroot/dot.sbatch"
printf '8000008 RUNNING laneI8 %s %s\n' "$TI8" "$TI8/jobroot/dot.sbatch" > "$WORK/I8_run.txt"
runsync "$RI8" "$TI8" "$WORK/I8_squeue" "$V2I8" --running-list "$WORK/I8_run.txt" > "$WORK/I8.log" 2>&1; rcI8=$?
[ "$rcI8" -eq 6 ] && grep -q 'file=a_tool.sh reason=read-by-laneI8' "$WORK/I8.log" \
  && ok "I(a1d): \`. \"\$FAST_ENV\"\` is RESOLVED to its file and refuses BY NAME, not as 'undeterminable'" \
  || { bad "I(a1d): rc=$rcI8 -- the variable source was not resolved"; sed 's/^/      /' "$WORK/I8.log"; }
read -r RI9 TI9 V1I9 V2I9 < <(mkfixture I9)
mkstub "$WORK/I9_squeue"
mkdir -p "$TI9/jobroot"
{ echo '#!/bin/bash'; echo 'OB=/tmp/some.rows.txt'; echo 'OB_N=$(grep -c . "$OB")'
  echo 'HP_N=$(grep -c . "$HPROV")'; echo "bash \"$TI9/tools/harness_drift_check.sh\" x"; } > "$TI9/jobroot/grep.sbatch"
printf '8000009 RUNNING laneI9 %s %s\n' "$TI9" "$TI9/jobroot/grep.sbatch" > "$WORK/I9_run.txt"
runsync "$RI9" "$TI9" "$WORK/I9_squeue" "$V2I9" --running-list "$WORK/I9_run.txt" > "$WORK/I9.log" 2>&1; rcI9=$?
git -C "$RI9" cat-file blob "$V2I9:harness/tools/a_tool.sh" > "$WORK/I9.blob"
[ "$rcI9" -eq 0 ] && cmp -s "$WORK/I9.blob" "$TI9/tools/a_tool.sh" \
  && ok "I(a1d): \`grep -c . \"\$OB\"\` is an ARGUMENT, not a source -- the job stays determinable and the sync proceeds" \
  || { bad "I(a1d): rc=$rcI9 -- a grep argument was read as a dot-source and refused the sync"; sed 's/^/      /' "$WORK/I9.log"; }


# ---- L: HARNESS-SYNC-4 -- a PENDING job READS TOO, and rc 4 never saw it ----
# THE HOLE, WITH A LIVE EXAMPLE IN IT. det141-cleanup 6001240 is an
# `afterany:6001140` owner submitted as
# `sbatch --wrap 'bash <task dir>/merge-h/cleanup_gated.sh <roots>'`. It owns NO
# PIN DIR, so the rc-4 pin check cannot see it. It is not RUNNING, so the rc-6
# read-set check (arms I) could not see it either. A sync landing in the window
# between 6001140 finishing and Slurm starting 6001240 would therefore have
# rewritten cleanup_gated.sh underneath it, with no drift refusal, no rc 4, no
# rc 6 and nothing in any log -- the job would simply have run a script nobody
# chose for it. HARNESS-PREAMBLE-1 deferred its own sync BY HAND for exactly
# this reason, which is the tell that the rule was missing from the machinery.
# So the read set is now computed for state PD as well as R, and the refusal row
# carries `state=` because "wait for job X" means something different when X has
# not started: the operator must wait for it to RUN and FINISH, not just start.
mkpdjob () {        # $1 = task dir; writes a --wrap-shaped script reading $2
  local T=$1 reads=$2
  mkdir -p "$T/jobroot"
  { echo '#!/bin/sh'; echo '# This script was created by sbatch --wrap.'; echo
    echo "bash $T/$reads /some/cert /some/ws"; } > "$T/jobroot/wrap.sh"
}
# b1: a PENDING job whose --wrap line names a file this sync WOULD rewrite
read -r RL TL V1L V2L < <(mkfixture L)
mkstub "$WORK/L_squeue"
mkpdjob "$TL" tools/a_tool.sh
printf '8000011 PENDING laneL %s %s\n' "$TL" "$TL/jobroot/wrap.sh" > "$WORK/L_run.txt"
git -C "$RL" cat-file blob "$V2L:harness/tools/a_tool.sh" > "$WORK/L.blob"
BEFORE_L=$(md5sum "$TL/tools/a_tool.sh" | awk '{print $1}')
WANT_L=$(md5sum "$WORK/L.blob" | awk '{print $1}')
[ "$BEFORE_L" != "$WANT_L" ] \
  && ok "L(b1): NON-VACUITY BEFORE -- tools/a_tool.sh is $BEFORE_L and $V2L says $WANT_L, so it IS in the install set" \
  || bad "L(b1): the fixture file already matches the commit -- the arm would prove nothing"
runsync "$RL" "$TL" "$WORK/L_squeue" "$V2L" --running-list "$WORK/L_run.txt" > "$WORK/L.log" 2>&1; rcL=$?
[ "$rcL" -eq 6 ] && ok "L(b1): a PENDING job whose script reads an installed file refuses rc 6 -- the SAME rc as RUNNING" \
                 || { bad "L(b1): rc=$rcL, wanted 6"; sed 's/^/      /' "$WORK/L.log"; }
grep -q '^### SYNC REFUSED rc=6 running=8000011 state=PENDING file=a_tool.sh reason=read-by-laneL' "$WORK/L.log" \
  && ok "L(b1): and the row says state=PENDING, so the operator knows the job has NOT STARTED" \
  || { bad "L(b1): no 'rc=6 running=8000011 state=PENDING file=a_tool.sh reason=read-by-laneL' row"; sed 's/^/      /' "$WORK/L.log"; }
AFTER_L=$(md5sum "$TL/tools/a_tool.sh" | awk '{print $1}')
[ "$AFTER_L" = "$BEFORE_L" ] && [ "$AFTER_L" != "$WANT_L" ] \
  && ok "L(b1): NON-VACUITY AFTER -- nothing was written; the file is still $AFTER_L and still not $V2L's bytes" \
  || bad "L(b1): the sync wrote the file it refused over ($BEFORE_L -> $AFTER_L)"
# b2: a PENDING job whose read set is DISJOINT must NOT block the sync. Without
# this arm the fix could be "refuse whenever anything is queued", which in a task
# dir carrying 25 held PD jobs would mean the harness could never be synced again.
read -r RL2 TL2 V1L2 V2L2 < <(mkfixture L2)
mkstub "$WORK/L2_squeue"
mkpdjob "$TL2" tools/harness_drift_check.sh        # same bytes at v1 and v2: not installed
printf '8000012 PENDING laneL2 %s %s\n' "$TL2" "$TL2/jobroot/wrap.sh" > "$WORK/L2_run.txt"
runsync "$RL2" "$TL2" "$WORK/L2_squeue" "$V2L2" --running-list "$WORK/L2_run.txt" > "$WORK/L2.log" 2>&1; rcL2=$?
git -C "$RL2" cat-file blob "$V2L2:harness/tools/a_tool.sh" > "$WORK/L2.blob"
[ "$rcL2" -eq 0 ] && cmp -s "$WORK/L2.blob" "$TL2/tools/a_tool.sh" \
  && ok "L(b2): a PENDING job with a DISJOINT read set does not block the sync (rc 0, installed)" \
  || { bad "L(b2): rc=$rcL2 -- a disjoint PENDING read set must not refuse"; sed 's/^/      /' "$WORK/L2.log"; }
grep -q 'SYNC REFUSED rc=6' "$WORK/L2.log" && bad "L(b2): it refused anyway" \
                                           || ok "L(b2): and it printed no rc-6 row at all"
# b3: a PENDING job whose script cannot be read is treated as reading EVERYTHING
read -r RL3 TL3 V1L3 V2L3 < <(mkfixture L3)
mkstub "$WORK/L3_squeue"
printf '8000013 PENDING laneL3 %s -\n' "$TL3" > "$WORK/L3_run.txt"
BEFORE_L3=$(md5sum "$TL3/tools/a_tool.sh" | awk '{print $1}')
runsync "$RL3" "$TL3" "$WORK/L3_squeue" "$V2L3" --running-list "$WORK/L3_run.txt" > "$WORK/L3.log" 2>&1; rcL3=$?
[ "$rcL3" -eq 6 ] && ok "L(b3): a PENDING job with NO readable script is treated as reading everything -- rc 6 (law 9)" \
                  || { bad "L(b3): rc=$rcL3, wanted 6"; sed 's/^/      /' "$WORK/L3.log"; }
grep -q 'state=PENDING file=a_tool.sh reason=no-sbatch-found' "$WORK/L3.log" \
  && ok "L(b3): and the row carries BOTH the state and the reason it could not be determined" \
  || { bad "L(b3): no 'state=PENDING ... reason=no-sbatch-found' row"; sed 's/^/      /' "$WORK/L3.log"; }
[ "$(md5sum "$TL3/tools/a_tool.sh" | awk '{print $1}')" = "$BEFORE_L3" ] \
  && ok "L(b3): nothing written there either" || bad "L(b3): it wrote after refusing"
# b4: THE LIVE PATH. `--running-list` shims the job LIST, so b1-b3 prove the
# state PLUMBING and nothing about the `squeue -t` filter the list stands in for.
# This arm drives the REAL squeue call through a stub that behaves like squeue --
# it honours `-t` and answers only for the format carrying %T -- so a PENDING job
# only reaches the check if the filter actually asks for PD.
mkstub_states () {  # $1=path $2=RUNNING rows $3=PENDING rows; models `squeue -t`
  local p=$1 rtext=$2 ptext=$3
  [ -n "$rtext" ] && printf '%s\n' "$rtext" > "$p.R" || : > "$p.R"
  [ -n "$ptext" ] && printf '%s\n' "$ptext" > "$p.PD" || : > "$p.PD"
  { echo '#!/usr/bin/env bash'
    echo 'fmt=; t=R'
    echo 'while [ $# -gt 0 ]; do'
    echo '  case "$1" in'
    echo '    -t) shift; t=${1:-};;  -t?*) t=${1#-t};;'
    echo '    -o) shift; fmt=${1:-};; -o?*) fmt=${1#-o};;'
    echo '  esac; shift'
    echo 'done'
    echo '# only the read-set query asks for %T; the rc-4 pin query must see nothing here'
    echo 'case "$fmt" in *%T*) ;; *) exit 0;; esac'
    echo "case \",\$t,\" in *,R,*)  [ -s '$p.R' ]  && cat '$p.R';;  esac"
    echo "case \",\$t,\" in *,PD,*) [ -s '$p.PD' ] && cat '$p.PD';; esac"
    echo 'exit 0'
  } > "$p"
  chmod 755 "$p"
}
read -r RL4 TL4 V1L4 V2L4 < <(mkfixture L4)
mkstub_states "$WORK/L4_squeue" "" "8000014 PENDING laneL4 $TL4"
git -C "$RL4" cat-file blob "$V2L4:harness/tools/a_tool.sh" > "$WORK/L4.blob"
BEFORE_L4=$(md5sum "$TL4/tools/a_tool.sh" | awk '{print $1}')
WANT_L4=$(md5sum "$WORK/L4.blob" | awk '{print $1}')
[ "$BEFORE_L4" != "$WANT_L4" ] \
  && ok "L(b4): NON-VACUITY -- a_tool.sh is $BEFORE_L4, $V2L4 says $WANT_L4, so it IS in the install set" \
  || bad "L(b4): the fixture already matches the commit -- the live-path arm would prove nothing"
runsync "$RL4" "$TL4" "$WORK/L4_squeue" "$V2L4" > "$WORK/L4.log" 2>&1; rcL4=$?
[ "$rcL4" -eq 6 ] && ok "L(b4): THE LIVE squeue PATH lists PENDING jobs -- a queued job of ours refuses rc 6 with no --running-list at all" \
                  || { bad "L(b4): rc=$rcL4, wanted 6 -- the production squeue query does not reach PENDING jobs"; sed 's/^/      /' "$WORK/L4.log"; }
grep -q 'state=PENDING' "$WORK/L4.log" \
  && ok "L(b4): and the live-path row carries state=PENDING too" \
  || { bad "L(b4): the live-path refusal row has no state=PENDING"; sed 's/^/      /' "$WORK/L4.log"; }
[ "$(md5sum "$TL4/tools/a_tool.sh" | awk '{print $1}')" = "$BEFORE_L4" ] \
  && ok "L(b4): and it wrote nothing" || bad "L(b4): it wrote after refusing"
# b5: THE MUTATION, and it cuts exactly one thing: the PD half of the state
# filter. With `-t R` the queued job is never listed, the read set is empty, and
# the same fixture INSTALLS -- which is what made 6001240 invisible. Without this
# arm, b4 could be passing on the stub rather than on the filter.
# The sanity check targets the CODE line (`-t R,PD -o`), not the bare string: the
# header comment ALSO says `squeue -t R,PD`, and a first cut that asserted the
# string was absent everywhere never built the mutant at all -- guard run 6002444,
# "MUTATION ARM DID NOT RUN". That is the arm catching its own defect.
MUT4=$WORK/harness_sync_runningonly.sh
sed 's/-t R,PD -o/-t R -o/' "$SYNC" > "$MUT4"
if bash -n "$MUT4" 2>/dev/null && ! grep -q -- "-t R,PD -o" "$MUT4" && grep -q -- "-t R -o" "$MUT4"; then
  read -r RL5 TL5 V1L5 V2L5 < <(mkfixture L5)
  mkstub_states "$WORK/L5_squeue" "" "8000015 PENDING laneL5 $TL5"
  cp -f "$MUT4" "$TL5/tools/harness_sync.sh"
  git -C "$RL5" cat-file blob "$V2L5:harness/tools/a_tool.sh" > "$WORK/L5.blob"
  HARNESS_REPO="$RL5" HARNESS_TASK_DIR="$TL5" HARNESS_SQUEUE="$WORK/L5_squeue" \
    bash "$TL5/tools/harness_sync.sh" "$V2L5" > "$WORK/L5.log" 2>&1; rcL5=$?
  if [ "$rcL5" -ne 6 ] && cmp -s "$WORK/L5.blob" "$TL5/tools/a_tool.sh"; then
    ok "L(b5): MUTATION -- with the PD half of the state filter cut (\`-t R\`) the sync rc=$rcL5 INSTALLS over the queued job, so b4 CAN fail"
  else
    bad "L(b5): the \`-t R\` mutant still refused (rc=$rcL5) or did not install -- ARM b4 IS NOT TESTING THE PD STATE FILTER"
  fi
else
  bad "L(b5): could not build the -t R mutant -- MUTATION ARM DID NOT RUN"
fi
# b6: the state column is REQUIRED, and a row without it is a FATAL, not a row
# read one field out of step. A four-field row is what every --running-list
# fixture looked like before this lane; read with the new parser it would make
# `<name>` the state, shift the script one field left, and hand the check an
# EMPTY read set -- a silent install, which is the defect this block exists for.
read -r RL6 TL6 V1L6 V2L6 < <(mkfixture L6)
mkstub "$WORK/L6_squeue"
mkjob "$TL6" tools/a_tool.sh
printf '8000016 laneL6 %s %s\n' "$TL6" "$TL6/jobroot/j.sbatch" > "$WORK/L6_run.txt"   # the OLD four-field shape
BEFORE_L6=$(md5sum "$TL6/tools/a_tool.sh" | awk '{print $1}')
runsync "$RL6" "$TL6" "$WORK/L6_squeue" "$V2L6" --running-list "$WORK/L6_run.txt" > "$WORK/L6.log" 2>&1; rcL6=$?
[ "$rcL6" -eq 2 ] && ok "L(b6): a --running-list row with no state column is FATAL rc 2, not silently misparsed" \
                  || { bad "L(b6): rc=$rcL6, wanted 2 -- a stale four-field row was accepted"; sed 's/^/      /' "$WORK/L6.log"; }
grep -q "state='laneL6'" "$WORK/L6.log" \
  && ok "L(b6): and the FATAL NAMES the field it read and the row shape it wanted" \
  || { bad "L(b6): the FATAL does not name the bad state field"; sed 's/^/      /' "$WORK/L6.log"; }
[ "$(md5sum "$TL6/tools/a_tool.sh" | awk '{print $1}')" = "$BEFORE_L6" ] \
  && ok "L(b6): and it wrote nothing before refusing" || bad "L(b6): it wrote before the FATAL"

# ---- M: HARNESS-SYNC-4-1 -- BOTH refusal sets are computed and BOTH printed --
# THE DEFECT, WITH A DATED LANDING IN IT. The rc-4 pin block `exit 4`ed where it
# stood, upstream of the entire read-set computation. So when both applied the
# operator saw only the rc-4 rows, was told to let the queued job drain, drained
# it, re-ran -- and was THEN handed rc 6 and a second wait nothing had let them
# see coming. That is MERGE-U's landing at 2026-09-06T22:41: refused rc 4 naming
# det141-cleanup 6001240, with det141-proof 6001140 RUNNING beside it and its
# read-set hit never computed. Two serial waits for ONE queue state.
# Now both checks run, both print, and the exit is decided after both. rc 4 wins
# when both apply -- it is the cheaper one to clear (repin) and a caller
# switching on the rc needs a stable answer -- and the summary says the rc-6
# rows are owed too, so the precedence is not silently a dismissal.
mkboth () {         # $1 = task dir; a pin dir at $2 plus a driver reading $3
  local T=$1 pin=$2 reads=$3
  mkdir -p "$T/lane1"; printf '%s\n' "$pin" > "$T/lane1/HARNESS_COMMIT"
  mkjob "$T" "$reads"
}
# m1: BOTH. A PENDING job pinned to v1 (rc 4) AND a RUNNING job whose driver
# reads a_tool.sh (rc 6), in one fixture, one invocation.
read -r RM TM V1M V2M < <(mkfixture M)
mkboth "$TM" "$V1M" tools/a_tool.sh
mkstub "$WORK/M_squeue" "9999901 lane1-relock"          # answers the rc-4 pin query
printf '8000021 RUNNING laneM %s %s\n' "$TM" "$TM/jobroot/j.sbatch" > "$WORK/M_run.txt"
BEFORE_M=$(md5sum "$TM/tools/a_tool.sh" | awk '{print $1}')
git -C "$RM" cat-file blob "$V2M:harness/tools/a_tool.sh" > "$WORK/M.blob"
WANT_M=$(md5sum "$WORK/M.blob" | awk '{print $1}')
[ "$BEFORE_M" != "$WANT_M" ] \
  && ok "M(m1): NON-VACUITY -- a_tool.sh is $BEFORE_M and $V2M says $WANT_M, so it IS in the install set" \
  || bad "M(m1): the fixture already matches the commit -- neither check would have a subject"
runsync "$RM" "$TM" "$WORK/M_squeue" "$V2M" --running-list "$WORK/M_run.txt" > "$WORK/M.log" 2>&1; rcM=$?
[ "$rcM" -eq 4 ] && ok "M(m1): with BOTH refusals live the sync exits rc 4 (the documented precedence)" \
                 || { bad "M(m1): rc=$rcM, wanted 4"; sed 's/^/      /' "$WORK/M.log"; }
grep -q 'state=PENDING 9999901' "$WORK/M.log" \
  && ok "M(m1): the rc-4 family is printed AND carries state= like the rc-6 family does" \
  || { bad "M(m1): no 'state=PENDING 9999901' rc-4 row"; sed 's/^/      /' "$WORK/M.log"; }
grep -q 'SYNC REFUSED rc=6 running=8000021 state=RUNNING file=a_tool.sh' "$WORK/M.log" \
  && ok "M(m1): THE FIX -- the rc-6 rows are on the page too, in the SAME run that refused rc 4" \
  || { bad "M(m1): the rc-6 rows are missing -- the early exit is still upstream of the read set"; sed 's/^/      /' "$WORK/M.log"; }
grep -q 'BOTH CHECKS REFUSED' "$WORK/M.log" \
  && ok "M(m1): and the summary SAYS both refused, so rc 4 does not read as an all-clear on rc 6" \
  || { bad "M(m1): no BOTH CHECKS REFUSED summary"; sed 's/^/      /' "$WORK/M.log"; }
[ "$(md5sum "$TM/tools/a_tool.sh" | awk '{print $1}')" = "$BEFORE_M" ] \
  && ok "M(m1): and running BOTH checks still wrote nothing" || bad "M(m1): it wrote while refusing"
# m2: the non-vacuity control for m1's rc. rc-6 ALONE must still be rc 6, and
# must print NO rc-4 row -- otherwise m1's `4` could be the script's only answer.
read -r RM2 TM2 V1M2 V2M2 < <(mkfixture M2)
mkstub "$WORK/M2_squeue"                                 # nothing pinned, nothing queued
mkjob "$TM2" tools/a_tool.sh
printf '8000022 RUNNING laneM2 %s %s\n' "$TM2" "$TM2/jobroot/j.sbatch" > "$WORK/M2_run.txt"
runsync "$RM2" "$TM2" "$WORK/M2_squeue" "$V2M2" --running-list "$WORK/M2_run.txt" > "$WORK/M2.log" 2>&1; rcM2=$?
[ "$rcM2" -eq 6 ] && ok "M(m2): rc 6 ALONE is still rc 6 -- the deferred exit did not collapse both rcs into 4" \
                  || { bad "M(m2): rc=$rcM2, wanted 6"; sed 's/^/      /' "$WORK/M2.log"; }
grep -q 'would strand these PENDING jobs' "$WORK/M2.log" \
  && { bad "M(m2): an rc-4 row was printed with nothing pinned"; sed 's/^/      /' "$WORK/M2.log"; } \
  || ok "M(m2): and it prints NO rc-4 row -- the families are reported separately, not merged"
grep -q 'rc 6 only' "$WORK/M2.log" \
  && ok "M(m2): the summary says which single check refused" || bad "M(m2): no 'rc 6 only' summary line"
# m3: THE MUTATION, and it cuts exactly one thing -- the deferred exit becomes
# the early `exit 4` again. m1's fixture must then LOSE its rc-6 rows. Without
# this arm m1's rc-6 assertion could be passing on the fixture rather than on
# the reordering.
MUT5=$WORK/harness_sync_earlyexit4.sh
sed 's/SYNC_RC4=4   # HARNESS-SYNC-4-1 DEFERRED EXIT/exit 4/' "$SYNC" > "$MUT5"
if bash -n "$MUT5" 2>/dev/null && ! grep -q 'SYNC_RC4=4   # HARNESS-SYNC-4-1 DEFERRED EXIT' "$MUT5"; then
  read -r RM3 TM3 V1M3 V2M3 < <(mkfixture M3)
  mkboth "$TM3" "$V1M3" tools/a_tool.sh
  mkstub "$WORK/M3_squeue" "9999903 lane1-relock"
  printf '8000023 RUNNING laneM3 %s %s\n' "$TM3" "$TM3/jobroot/j.sbatch" > "$WORK/M3_run.txt"
  cp -f "$MUT5" "$TM3/tools/harness_sync.sh"
  HARNESS_REPO="$RM3" HARNESS_TASK_DIR="$TM3" HARNESS_SQUEUE="$WORK/M3_squeue" \
    bash "$TM3/tools/harness_sync.sh" "$V2M3" --running-list "$WORK/M3_run.txt" > "$WORK/M3.log" 2>&1; rcM3=$?
  if [ "$rcM3" -eq 4 ] && ! grep -q 'SYNC REFUSED rc=6' "$WORK/M3.log"; then
    ok "M(m3): MUTATION -- with the early \`exit 4\` restored the same fixture prints NO rc-6 row (rc=$rcM3), so m1 CAN fail"
  else
    bad "M(m3): the early-exit mutant still printed the rc-6 rows (rc=$rcM3) -- ARM m1 IS NOT TESTING THE DEFERRED EXIT"
  fi
else
  bad "M(m3): could not build the early-exit-4 mutant -- MUTATION ARM DID NOT RUN"
fi
# m4: the `--force` summary's JOB COUNT, which had no reader at all. HARNESS-
# SYNC-4 replaced an `awk '{print $3}'` here that printed the literal REFUSED
# for every input -- a count nobody ever asserted, in a line the operator is
# told to copy into a lane log row. Two pinned jobs must print TWO.
read -r RM4 TM4 V1M4 V2M4 < <(mkfixture M4)
mkdir -p "$TM4/lane1" "$TM4/lane2"
printf '%s\n' "$V1M4" > "$TM4/lane1/HARNESS_COMMIT"
printf '%s\n' "$V1M4" > "$TM4/lane2/HARNESS_COMMIT"
mkstub "$WORK/M4_squeue" "9999904 lane1-relock" "9999905 lane2-relock"
runsync "$RM4" "$TM4" "$WORK/M4_squeue" "$V2M4" > "$WORK/M4a.log" 2>&1; rcM4a=$?
[ "$rcM4a" -eq 4 ] && [ "$(grep -c 'state=PENDING 999990' "$WORK/M4a.log")" -eq 2 ] \
  && ok "M(m4): NON-VACUITY -- the fixture really does produce TWO rc-4 rows for TWO jobs" \
  || { bad "M(m4): rc=$rcM4a with $(grep -c 'state=PENDING 999990' "$WORK/M4a.log") rows, wanted rc 4 and 2"; sed 's/^/      /' "$WORK/M4a.log"; }
runsync "$RM4" "$TM4" "$WORK/M4_squeue" "$V2M4" --force --reason "guard fixture m4" > "$WORK/M4.log" 2>&1; rcM4=$?
grep -q '### SYNC FORCED over 2 pinned PENDING job(s) reason=guard fixture m4' "$WORK/M4.log" \
  && ok "M(m4): the forced summary COUNTS the jobs it overrode -- 'over 2 pinned PENDING job(s)'" \
  || { bad "M(m4): the count is wrong: $(grep -m1 'SYNC FORCED over' "$WORK/M4.log" || echo '<no SYNC FORCED line>')"; sed 's/^/      /' "$WORK/M4.log"; }

# ---- K: the live read set comes from SLURM'S OWN SNAPSHOT, not from disk ----
# `--running-list` shims the job LIST; it cannot shim the one live call the list
# is built from. `sbatch --wrap` and heredoc submissions leave `Command=(null)`
# and NO file on disk -- 5992050, RUNNING in this task dir, is one -- so a check
# that read only `Command=` would have called every such job undeterminable and
# refused every sync for its whole life. This arm runs the real call against THIS
# guard's own job, which is the only running job it is entitled to ask about.
grep -q 'scontrol write batch_script' "$SYNC" \
  && ok "K: harness_sync.sh takes the running job's script from Slurm's snapshot, not from Command= alone" \
  || bad "K: harness_sync.sh never calls scontrol write batch_script -- Command=(null) jobs would refuse forever"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if scontrol write batch_script "$SLURM_JOB_ID" "$WORK/self.sbatch" >/dev/null 2>&1 && [ -s "$WORK/self.sbatch" ]; then
    ok "K: scontrol write batch_script $SLURM_JOB_ID dumped $(wc -c < "$WORK/self.sbatch") bytes -- the mechanism works on a RUNNING job"
  else
    bad "K: scontrol write batch_script failed on this guard's own job -- the live read set has no source"
  fi
  grep -q 'harness_sync_guard.sh' "$WORK/self.sbatch" 2>/dev/null \
    && ok "K: and the bytes are THIS job's real submitted script (it names harness_sync_guard.sh)" \
    || bad "K: the dump does not name harness_sync_guard.sh -- it is not this job's script"
else
  bad "K: no SLURM_JOB_ID -- the live-snapshot arm DID NOT RUN (run this guard under sbatch)"
fi

# ---- J: the header no longer claims the rename is sufficient ---------------
[ "$(grep -c 'is the only safe way to write into a live task dir' "$SYNC")" -eq 0 ] \
  && ok "J: the retracted claim ('the only safe way to write into a live task dir') is GONE" \
  || bad "J: harness_sync.sh still claims the rename is the only safe way"
grep -qi 'same NFS client' "$SYNC" \
  && ok "J: and the header states the real condition -- renamer and reader on the same NFS client" \
  || bad "J: the header does not name the same-NFS-client condition"
grep -q 'read set checked pre-install' "$SYNC" \
  && ok "J: the PIN REPORT's RUNNING row no longer says SAFE; it says drift only, read set checked pre-install" \
  || bad "J: the PIN REPORT still labels a RUNNING job SAFE without qualification"

LIVE_AFTER=$( [ -f "$LIVE_RECORD" ] && md5sum "$LIVE_RECORD" | awk '{print $1}' || echo absent )
[ "$LIVE_BEFORE" = "$LIVE_AFTER" ] && ok "live task dir untouched: $LIVE_RECORD md5 $LIVE_BEFORE" \
                                   || bad "THE GUARD MOVED THE LIVE RECORD: $LIVE_BEFORE -> $LIVE_AFTER"


# ---- P: HARNESS-SYNC-5 -- an owner that reads JOB-LOCAL bytes ---------------
# THE DEFECT. A cleanup owner is submitted as
#   sbatch --wrap 'bash <task dir>/merge-h/cleanup_gated.sh <roots>'
# Slurm snapshots the TOP-LEVEL script -- the one-line wrap -- and nothing else,
# so the gate and the cleanup.sh it calls are read from the task tree LIVE for
# the owner's whole life. det1f-cleanup 5999937 spent four hours inside ONE
# 3,587,597-entry unlink with those reads open and det141-cleanup 6001240 sat
# beside it; for every one of those hours any install touching those files was
# refused rc 6. The refusal is RIGHT and stays; what changes is that the owner
# stops reading a synced file, so there is nothing left to protect.
#
#   p1  an owner running a JOB-ROOT SNAPSHOT installs (rc 0) even though the
#       install set contains a file of the SAME BASENAME
#   p2  the MUTATION, and it is the OLD SHAPE: the same fixture, the same
#       install set, the owner reading the TASK path -> rc 6, still. Without p2,
#       p1 would pass just as well on a read-set check that had stopped
#       refusing anything at all.
#   p3  owner_snapshot.sh copies the gate AND what the gate sources, and prints
#       its row
#   p4  a reference the parser cannot resolve is a REFUSAL, not a partial
#       snapshot -- a snapshot with a hole is worse than none, because the job
#       reads the LIVE path for that one file while every row says it is frozen
SNAPTOOL=$(dirname -- "$SYNC")/../phase_template/owner_snapshot.sh
[ -f "$SNAPTOOL" ] || SNAPTOOL=$REPO/harness/phase_template/owner_snapshot.sh
if [ ! -f "$SNAPTOOL" ]; then
  bad "P: no owner_snapshot.sh -- HARNESS-SYNC-5's submitter half is absent"
else
mkowner () {   # $1 = task dir, $2 = repo; echoes the amended v2 sha
  # The gate's REAL shape, quoted from phase_template/cleanup_gated.sh: it names
  # `cleanup.sh` through a variable assigned from its own directory, which is
  # what the sibling rule has to get right.
  local T=$1 R=$2
  printf '#!/bin/bash\nCLEANUP=$(dirname "$0")/cleanup.sh\nbash "$CLEANUP"\n' > "$T/merge-h/cleanup_gated.sh"
  printf '#!/bin/bash\necho cleanup\n' > "$T/merge-h/cleanup.sh"
  mkdir -p "$T/jobroot"
  # The repo must carry a cleanup.sh too, or the sync maps the task file to a
  # blob that does not exist and fails rc 5 for a reason that has nothing to do
  # with the read set -- which is exactly what job 6009975 caught.
  printf 'v2 cleanup CHANGED\n' > "$R/harness/phase_template/cleanup.sh"
  git -C "$R" add -A >/dev/null 2>&1
  git -C "$R" commit -q --amend --no-edit >/dev/null 2>&1
  # DET-1-6-a: a REAL task tree carries `tools/.harness_synced_commit` -- that
  # record is what identifies a task copy's bytes, and owner_snapshot.sh now
  # REFUSES a source it cannot identify at all. A fixture without the record was
  # a fixture less faithful than production, and it is the only thing that made
  # the old `src_commit=unknown` row look acceptable.
  git -C "$R" rev-parse HEAD > "$T/tools/.harness_synced_commit"
  git -C "$R" rev-parse HEAD
}
# ---- p3 first: the snapshot itself, because p1 depends on it working --------
read -r RP TP V1P V2P < <(mkfixture P)
V2P=$(mkowner "$TP" "$RP")
PSNAP=$WORK/P_snapshot.log
bash "$SNAPTOOL" "$TP/jobroot" "$TP/merge-h/cleanup_gated.sh" > "$PSNAP" 2>&1; rcP3=$?
if [ "$rcP3" -eq 0 ] \
   && grep -qE '^### OWNER SNAPSHOT files=2 root='"$TP"'/jobroot src_commit=' "$PSNAP" \
   && [ -f "$TP/jobroot/owner-snapshot/cleanup_gated.sh" ] \
   && [ -f "$TP/jobroot/owner-snapshot/cleanup.sh" ] \
   && [ -f "$TP/jobroot/owner-snapshot/owner.sbatch" ]; then
  ok "P(p3): owner_snapshot froze the gate AND the cleanup.sh it sources (files=2) and wrote owner.sbatch"
else
  bad "P(p3): rc=$rcP3"; sed 's/^/      /' "$PSNAP"
fi
# CLEANUP-WALL-3 (2026-09-07): this read `exec bash <frozen>` until the owner
# stopped EXEC'ing its gate. It cannot exec any more -- the continuation decision
# now runs AFTER the pass and needs the pass's rc and log, and an `exec` replaces
# the shell that would make it. The claim this arm actually holds is unchanged
# and is the one restated here: the frozen copy is named by a LITERAL absolute
# path, because a `$SNAP/...` there reads back unresolved and the sync goes on
# refusing. The rc is preserved by `exit "$OWNER_PASS_RC"`, asserted below.
grep -qF "bash $TP/jobroot/owner-snapshot/cleanup_gated.sh" "$TP/jobroot/owner-snapshot/owner.sbatch" \
  && ok "P(p3): the generated sbatch names the frozen copy by LITERAL absolute path (a variable there would read back unresolved and go on refusing)" \
  || { bad "P(p3): owner.sbatch does not run the frozen copy by literal path"; sed 's/^/      /' "$TP/jobroot/owner-snapshot/owner.sbatch"; }
grep -q '^exit "\$OWNER_PASS_RC"$' "$TP/jobroot/owner-snapshot/owner.sbatch" \
  && ok "P(p3): and it exits the PASS's rc -- the owner no longer exec's the gate, so the rc has to be carried by hand or every refusal would read as success" \
  || { bad "P(p3): owner.sbatch does not end in exit \"\$OWNER_PASS_RC\" -- the gate's rc is dropped"; sed 's/^/      /' "$TP/jobroot/owner-snapshot/owner.sbatch"; }
# ---- p1: the sync now installs over that owner ------------------------------
mkstub "$WORK/P_squeue"
printf '8000051 RUNNING laneP %s %s %s\n' "$TP" "$TP/jobroot/owner-snapshot/owner.sbatch" "$TP/jobroot/owner-snapshot" > "$WORK/P_run.txt"
git -C "$RP" cat-file blob "$V2P:harness/phase_template/cleanup_gated.sh" > "$WORK/P.blob"
BEFORE_P=$(md5sum "$TP/merge-h/cleanup_gated.sh" | awk '{print $1}')
WANT_P=$(md5sum "$WORK/P.blob" | awk '{print $1}')
[ "$BEFORE_P" != "$WANT_P" ] \
  && ok "P(p1): NON-VACUITY -- merge-h/cleanup_gated.sh is $BEFORE_P and $V2P says $WANT_P, so it IS in the install set" \
  || bad "P(p1): the fixture gate already matches the commit -- the arm would prove nothing"
runsync "$RP" "$TP" "$WORK/P_squeue" "$V2P" --running-list "$WORK/P_run.txt" > "$WORK/P1.log" 2>&1; rcP1=$?
if [ "$rcP1" -eq 0 ] && cmp -s "$WORK/P.blob" "$TP/merge-h/cleanup_gated.sh"; then
  ok "P(p1): an owner running the JOB-ROOT snapshot no longer blocks the sync -- rc 0 and cleanup_gated.sh installed, with a live owner of the same basename in the queue"
else
  bad "P(p1): rc=$rcP1 -- the snapshot did not clear the read-set refusal"; sed 's/^/      /' "$WORK/P1.log"
fi
grep -q 'read-set OK job=8000051 file=cleanup_gated.sh' "$WORK/P1.log" \
  && ok "P(p1): and it SAYS SO -- the row names the job and the basename it cleared, rather than clearing it in silence" \
  || { bad "P(p1): no 'read-set OK' row naming the job"; sed 's/^/      /' "$WORK/P1.log"; }
# ---- p2: the MUTATION -- the old shape, reading the task path ---------------
read -r RP2 TP2 V1P2 V2P2 < <(mkfixture P2)
V2P2=$(mkowner "$TP2" "$RP2")
mkstub "$WORK/P2_squeue"
{ echo '#!/bin/bash'; echo "bash $TP2/merge-h/cleanup_gated.sh /oscar/data/stellex/glvov/retread/certX-1"; } > "$TP2/jobroot/wrap.sbatch"
printf '8000052 RUNNING laneP2 %s %s %s\n' "$TP2" "$TP2/jobroot/wrap.sbatch" "$TP2/jobroot" > "$WORK/P2_run.txt"
BEFORE_P2=$(md5sum "$TP2/merge-h/cleanup_gated.sh" | awk '{print $1}')
runsync "$RP2" "$TP2" "$WORK/P2_squeue" "$V2P2" --running-list "$WORK/P2_run.txt" > "$WORK/P2.log" 2>&1; rcP2=$?
AFTER_P2=$(md5sum "$TP2/merge-h/cleanup_gated.sh" | awk '{print $1}')
if [ "$rcP2" -eq 6 ] && [ "$AFTER_P2" = "$BEFORE_P2" ] \
   && grep -q 'SYNC REFUSED rc=6 running=8000052 .*file=cleanup_gated.sh' "$WORK/P2.log"; then
  ok "P(p2): MUTATION -- the SAME owner submitted the OLD way, reading the task path, is STILL refused rc 6 and nothing was written. p1 is the snapshot clearing it, not the check going blind."
else
  bad "P(p2): rc=$rcP2 before=$BEFORE_P2 after=$AFTER_P2 -- the old shape must still refuse"; sed 's/^/      /' "$WORK/P2.log"
fi
# ---- p4: a hole in the snapshot is a refusal --------------------------------
read -r RP4 TP4 V1P4 V2P4 < <(mkfixture P4)
mkowner "$TP4" "$RP4" >/dev/null
printf '#!/bin/bash\n. "$UNSET_SOMETHING"\n' >> "$TP4/merge-h/cleanup_gated.sh"
bash "$SNAPTOOL" "$TP4/jobroot" "$TP4/merge-h/cleanup_gated.sh" > "$WORK/P4.log" 2>&1; rcP4=$?
if [ "$rcP4" -ne 0 ] && grep -q 'OWNER SNAPSHOT REFUSED' "$WORK/P4.log" \
   && [ ! -f "$TP4/jobroot/owner-snapshot/owner.sbatch" ]; then
  ok "P(p4): a reference the parser cannot resolve REFUSES the snapshot (rc=$rcP4) and leaves NO owner.sbatch for a caller to submit"
else
  bad "P(p4): rc=$rcP4 -- an unresolvable reference must refuse, not freeze a partial set"; sed 's/^/      /' "$WORK/P4.log"
fi
fi

# ---- Q: HARNESS-CONSOL-10 -- --check names the PUSH LAG, not just the drift --
# The lane-close checklist gained a second row. On 2026-09-07 seventeen commits
# from four lanes existed only under agrescap/worktrees while --check reported
# CLEAN, because "clean" only ever meant "the task copies are the commit's
# bytes". Two arms: the row is emitted at all (q1), and it carries a REAL
# number read from a REAL remote (q2). q3 is the mutation.
read -r RQ TQ vQ1 vQ2 <<< "$(mkfixture Q)"
mkstub "$WORK/Q_squeue" ""
printf '%s\n' "$vQ1" > "$TQ/tools/.harness_synced_commit"
runsync "$RQ" "$TQ" "$WORK/Q_squeue" --check > "$WORK/Q1.log" 2>&1
grep -q '^### PUSH LAG branch=' "$WORK/Q1.log" \
  && ok "Q(q1): --check prints a PUSH LAG row beside the SYNC CHECK SUMMARY" \
  || { bad "Q(q1): --check printed no PUSH LAG row"; sed 's/^/      /' "$WORK/Q1.log"; }
grep -q '^### SYNC CHECK SUMMARY' "$WORK/Q1.log" \
  && ok "Q(q1): the drift summary is still printed -- the new row did not replace it" \
  || bad "Q(q1): the SYNC CHECK SUMMARY row is gone"

# q2: a real bare remote, one commit pushed, TWO left behind. The number has to
# be 2 and the rc has to be 1, or the row is decoration.
git init -q --bare "$WORK/Q_remote.git" 2>/dev/null
git -C "$RQ" remote add private "$WORK/Q_remote.git" 2>/dev/null
QBR=$(git -C "$RQ" symbolic-ref --quiet --short HEAD)
git -C "$RQ" push -q private "$vQ1:refs/heads/$QBR" 2>/dev/null
printf 'v3 tool\n' > "$RQ/harness/tools/a_tool.sh"
git -C "$RQ" add -A >/dev/null 2>&1; git -C "$RQ" commit -q -m v3 >/dev/null 2>&1
PC=$TQ/tools/harness_push_check.sh
HARNESS_REPO="$RQ" bash "$PC" > "$WORK/Q2.log" 2>&1; rcQ2=$?
if [ "$rcQ2" -eq 1 ] && grep -q "^### PUSH LAG branch=$QBR unpushed=2 " "$WORK/Q2.log"; then
  ok "Q(q2): two commits behind the remote -> unpushed=2 rc=1"
else
  bad "Q(q2): rc=$rcQ2, wanted 1 with unpushed=2"; sed 's/^/      /' "$WORK/Q2.log"
fi
grep -q 'fast-forward: bash -c .*push private' "$WORK/Q2.log" \
  && ok "Q(q2): the lag row names the command that fixes it" \
  || bad "Q(q2): the lag row does not say what to run"
git -C "$RQ" push -q private "$QBR" 2>/dev/null
HARNESS_REPO="$RQ" bash "$PC" > "$WORK/Q2b.log" 2>&1; rcQ2b=$?
if [ "$rcQ2b" -eq 0 ] && grep -q "unpushed=0 " "$WORK/Q2b.log"; then
  ok "Q(q2): after the push, unpushed=0 rc=0 -- the reader can go quiet"
else
  bad "Q(q2): after a real push rc=$rcQ2b, wanted 0"; sed 's/^/      /' "$WORK/Q2b.log"
fi
# a branch on NO remote is not 0 unpushed, it is ALL of them
git -C "$RQ" checkout -q -b orphan-q 2>/dev/null
HARNESS_REPO="$RQ" bash "$PC" > "$WORK/Q2c.log" 2>&1; rcQ2c=$?
if [ "$rcQ2c" -eq 1 ] && grep -q 'unpushed=ALL .*remote=ABSENT' "$WORK/Q2c.log"; then
  ok "Q(q2): a branch the remote has never seen reports ALL, not 0"
else
  bad "Q(q2): a branch absent from the remote gave rc=$rcQ2c"; sed 's/^/      /' "$WORK/Q2c.log"
fi
git -C "$RQ" checkout -q "$QBR" 2>/dev/null

# q3: THE MUTATION -- a checker that always says 0 must make q2 red.
sed -e 's|^\[ "\$N" -eq 0 \] && exit 0$|[ 1 -eq 1 ] \&\& exit 0|' "$PC" > "$WORK/Q_mut.sh"
git -C "$RQ" reset -q --hard "$vQ2" 2>/dev/null
HARNESS_REPO="$RQ" bash "$WORK/Q_mut.sh" > "$WORK/Q3.log" 2>&1; rcQ3=$?
if [ "$rcQ3" -eq 0 ]; then
  ok "Q(q3): the mutation (always exit 0) is REACHABLE -- q2's rc=1 is a real assertion"
else
  bad "Q(q3): the mutation did not change the rc (got $rcQ3) -- q2 cannot fail"
fi
# ---- N: HARNESS-SYNC-6 -- rc 4 is the PIN **AND** THE READ SET --------------
# THE DEFECT, WITH THE QUEUE STATE THAT PROVED IT. rc 4 asked ONE question --
# "does a PENDING job own a pin dir naming another commit?" -- and refused on the
# answer alone. HARNESS-SYNC-5 then built the owner snapshot for the express
# purpose of making a queued job STOP reading the task tree: it freezes the gate
# and what the gate sources into the job's own root and submits an sbatch that
# `exec`s the frozen copy by literal absolute path. Such a job runs no drift gate
# against this task dir and executes not one byte this sync writes -- and rc 4
# went on refusing for it anyway. On 2026-09-07 that was 6015658/6015659
# det162-cleanup: two snapshot owners, `afterany` on a proof that could run to
# 13:05, holding the entire merge queue on a pin neither of them ever read.
#   n1  a pin-mismatched PENDING SNAPSHOT OWNER -> TOLERATED on a named row, rc 0,
#       and the file with the colliding basename is actually installed
#   n2  a pin-mismatched PENDING `--wrap` READER of an installed file -> still
#       rc 4, and the rc-6 rows are on the same page (HARNESS-SYNC-4-1 intact)
#   n3  a pin-mismatched PENDING job whose script cannot be read -> still rc 4,
#       reason=read-set-undeterminable (law 9), and n3b: a job the read-set scope
#       never reached -> reason=read-set-not-examined, because "we did not look"
#       is not "we looked and it was clean"
#   n4  THE MUTATION: the tolerance branch forced unreachable -> n1's fixture
#       refuses rc 4 again, so n1 CAN fail
mksnapowner () {  # $1 = task dir; freezes a gate + its cleanup under <task>/jobroot/owner-snapshot
  local T=$1 S=$1/jobroot/owner-snapshot
  mkdir -p "$S"
  printf '#!/bin/bash\nCLEANUP=$(dirname "$0")/cleanup.sh\nbash "$CLEANUP"\n' > "$S/cleanup_gated.sh"
  printf '#!/bin/bash\necho frozen cleanup\n' > "$S/cleanup.sh"
  # The literal absolute path is the point: a variable here reads back
  # unresolved and the job goes on refusing (P(p3) asserts the same thing).
  { echo '#!/bin/bash'; echo "exec bash $S/cleanup_gated.sh /some/cert /some/ws"; } > "$S/owner.sbatch"
  chmod 755 "$S/cleanup_gated.sh" "$S/cleanup.sh" "$S/owner.sbatch"
}
# ---- n1: the snapshot owner is TOLERATED ------------------------------------
read -r RN TN V1N V2N < <(mkfixture N)
mkdir -p "$TN/lane1"; printf '%s\n' "$V1N" > "$TN/lane1/HARNESS_COMMIT"
mksnapowner "$TN"
mkstub "$WORK/N1_squeue" "9000001 lane1-relock"      # the rc-4 pin query sees it
printf '9000001 PENDING lane1-relock %s %s %s\n' \
  "$TN" "$TN/jobroot/owner-snapshot/owner.sbatch" "$TN/jobroot/owner-snapshot" > "$WORK/N1_run.txt"
git -C "$RN" cat-file blob "$V2N:harness/phase_template/cleanup_gated.sh" > "$WORK/N1.blob"
BEFORE_N1=$(md5sum "$TN/merge-h/cleanup_gated.sh" | awk '{print $1}')
WANT_N1=$(md5sum "$WORK/N1.blob" | awk '{print $1}')
[ "$BEFORE_N1" != "$WANT_N1" ] \
  && ok "N(n1): NON-VACUITY -- merge-h/cleanup_gated.sh is $BEFORE_N1 and $V2N says $WANT_N1, so the colliding basename IS in the install set" \
  || bad "N(n1): the fixture gate already matches the commit -- the arm would prove nothing"
runsync "$RN" "$TN" "$WORK/N1_squeue" "$V2N" --running-list "$WORK/N1_run.txt" > "$WORK/N1.log" 2>&1; rcN1=$?
[ "$rcN1" -eq 0 ] && ok "N(n1): THE FIX -- a pin-mismatched PENDING SNAPSHOT OWNER no longer refuses the sync (rc 0)" \
  || { bad "N(n1): rc=$rcN1, wanted 0 -- the pin alone is still refusing"; sed 's/^/      /' "$WORK/N1.log"; }
grep -q '^### PIN MISMATCH tolerated jid=9000001 reason=reads-only-job-root' "$WORK/N1.log" \
  && ok "N(n1): and it SAYS SO on a named row -- tolerated jid=9000001 reason=reads-only-job-root, never in silence" \
  || { bad "N(n1): no 'PIN MISMATCH tolerated jid=9000001 reason=reads-only-job-root' row"; sed 's/^/      /' "$WORK/N1.log"; }
# THE WHOLE ROW, FIELD BY FIELD, because a row whose sha field is malformed is a
# row no reader can parse -- and the first production run of this refinement
# printed exactly that: `pinned=pinned=8108ca4...`, the `pinned=` prefix applied
# twice because $REFUSALS field 4 already carries it. The arm above matched a
# PREFIX and sailed straight past it, which is how a field defect survives a
# guard. Asserted whole now.
grep -qE "^### PIN MISMATCH tolerated jid=9000001 reason=reads-only-job-root name=lane1-relock pin=$TN/lane1/HARNESS_COMMIT pinned=$V1N\$" "$WORK/N1.log" \
  && ok "N(n1): and the WHOLE row parses -- name=, pin= and a single well-formed pinned=<sha> naming $V1N" \
  || { bad "N(n1): the tolerated row's fields are malformed"; grep 'PIN MISMATCH tolerated' "$WORK/N1.log" | sed 's/^/      /'; }
grep -q 'SYNC REFUSED (rc 4)' "$WORK/N1.log" \
  && { bad "N(n1): it printed the rc-4 refusal anyway"; sed 's/^/      /' "$WORK/N1.log"; } \
  || ok "N(n1): and no rc-4 refusal is printed for it"
cmp -s "$WORK/N1.blob" "$TN/merge-h/cleanup_gated.sh" \
  && ok "N(n1): the install actually HAPPENED -- merge-h/cleanup_gated.sh is now $V2N's bytes" \
  || bad "N(n1): rc 0 but the file was not installed"
grep -q 'read-set OK job=9000001 file=cleanup_gated.sh' "$WORK/N1.log" \
  && ok "N(n1): the rc-6 side cleared it too, by the owner-snapshot rule (HARNESS-SYNC-5), and said which basename" \
  || bad "N(n1): no 'read-set OK' row -- n1 may be passing for the wrong reason"
# ---- n2: a pin-mismatched PENDING READER still refuses, WITH its rc-6 rows ---
read -r RN2 TN2 V1N2 V2N2 < <(mkfixture N2)
mkdir -p "$TN2/lane1"; printf '%s\n' "$V1N2" > "$TN2/lane1/HARNESS_COMMIT"
mkdir -p "$TN2/jobroot"
{ echo '#!/bin/sh'; echo '# This script was created by sbatch --wrap.'; echo
  echo "bash $TN2/tools/a_tool.sh /some/cert"; } > "$TN2/jobroot/wrap.sh"
mkstub "$WORK/N2_squeue" "9000002 lane1-relock"
printf '9000002 PENDING lane1-relock %s %s %s\n' "$TN2" "$TN2/jobroot/wrap.sh" "$TN2/jobroot" > "$WORK/N2_run.txt"
BEFORE_N2=$(md5sum "$TN2/tools/a_tool.sh" | awk '{print $1}')
runsync "$RN2" "$TN2" "$WORK/N2_squeue" "$V2N2" --running-list "$WORK/N2_run.txt" > "$WORK/N2.log" 2>&1; rcN2=$?
[ "$rcN2" -eq 4 ] && ok "N(n2): a pin-mismatched PENDING job that DOES read an installed file still refuses rc 4" \
  || { bad "N(n2): rc=$rcN2, wanted 4 -- the refinement is too wide"; sed 's/^/      /' "$WORK/N2.log"; }
grep -q '^###   state=PENDING 9000002 lane1-relock .*reason=reads-installed-file' "$WORK/N2.log" \
  && ok "N(n2): and the rc-4 row now carries the REASON it refused -- reads-installed-file" \
  || { bad "N(n2): the rc-4 row has no reason= field"; sed 's/^/      /' "$WORK/N2.log"; }
grep -q 'SYNC REFUSED rc=6 running=9000002 state=PENDING file=a_tool.sh' "$WORK/N2.log" \
  && ok "N(n2): the rc-6 rows are STILL on the same page (HARNESS-SYNC-4-1 survives the refinement)" \
  || { bad "N(n2): the rc-6 rows are gone -- the refinement broke SYNC-4-1"; sed 's/^/      /' "$WORK/N2.log"; }
grep -q 'BOTH CHECKS REFUSED' "$WORK/N2.log" \
  && ok "N(n2): and the summary still says BOTH refused, with rc 4 keeping precedence" \
  || bad "N(n2): no BOTH CHECKS REFUSED summary"
grep -q 'PIN MISMATCH tolerated' "$WORK/N2.log" \
  && bad "N(n2): it tolerated a job that reads an installed file" \
  || ok "N(n2): and no tolerated row was printed for it"
[ "$(md5sum "$TN2/tools/a_tool.sh" | awk '{print $1}')" = "$BEFORE_N2" ] \
  && ok "N(n2): nothing was written" || bad "N(n2): it wrote while refusing"
# ---- n3: undeterminable is a REFUSAL, and it says which kind ----------------
read -r RN3 TN3 V1N3 V2N3 < <(mkfixture N3)
mkdir -p "$TN3/lane1"; printf '%s\n' "$V1N3" > "$TN3/lane1/HARNESS_COMMIT"
mkstub "$WORK/N3_squeue" "9000003 lane1-relock"
printf '9000003 PENDING lane1-relock %s -\n' "$TN3" > "$WORK/N3_run.txt"
runsync "$RN3" "$TN3" "$WORK/N3_squeue" "$V2N3" --running-list "$WORK/N3_run.txt" > "$WORK/N3.log" 2>&1; rcN3=$?
[ "$rcN3" -eq 4 ] && ok "N(n3): a pin-mismatched PENDING job with NO readable script still refuses rc 4 (law 9)" \
  || { bad "N(n3): rc=$rcN3, wanted 4"; sed 's/^/      /' "$WORK/N3.log"; }
grep -q '^###   state=PENDING 9000003 lane1-relock .*reason=read-set-undeterminable' "$WORK/N3.log" \
  && ok "N(n3): and the row says WHICH kind of refusal it is -- read-set-undeterminable, not 'reads an installed file'" \
  || { bad "N(n3): the reason is missing or wrong"; sed 's/^/      /' "$WORK/N3.log"; }
# n3b: the job the read-set scope never reached at all. Arm B's fixture is this
# shape already (a bare `<jid> <name>` squeue stub leaves the read-set query a
# row with no workdir), so the reason has to be the NOT-EXAMINED one.
grep -q 'reason=read-set-not-examined' "$LOGB" \
  && ok "N(n3b): a pin-mismatched job the read-set scope never reached refuses as read-set-not-examined, not as clean" \
  || { bad "N(n3b): arm B's job was not classed read-set-not-examined"; grep -E '^###   state=PENDING' "$LOGB" | sed 's/^/      /'; }
# ---- n4: THE MUTATION -------------------------------------------------------
# Cut exactly one thing: the branch that lets a determinate, disjoint read set
# reach the tolerated row. n1's fixture must then refuse rc 4 again.
MUT6=$WORK/harness_sync_notolerance.sh
sed 's|.*# HARNESS-SYNC-6 TOLERANCE BRANCH$|  if true; then|' "$SYNC" > "$MUT6"
if bash -n "$MUT6" 2>/dev/null && ! grep -q 'HARNESS-SYNC-6 TOLERANCE BRANCH' "$MUT6"; then
  read -r RN4 TN4 V1N4 V2N4 < <(mkfixture N4)
  mkdir -p "$TN4/lane1"; printf '%s\n' "$V1N4" > "$TN4/lane1/HARNESS_COMMIT"
  mksnapowner "$TN4"
  mkstub "$WORK/N4_squeue" "9000004 lane1-relock"
  printf '9000004 PENDING lane1-relock %s %s %s\n' \
    "$TN4" "$TN4/jobroot/owner-snapshot/owner.sbatch" "$TN4/jobroot/owner-snapshot" > "$WORK/N4_run.txt"
  cp -f "$MUT6" "$TN4/tools/harness_sync.sh"
  HARNESS_REPO="$RN4" HARNESS_TASK_DIR="$TN4" HARNESS_SQUEUE="$WORK/N4_squeue" \
    bash "$TN4/tools/harness_sync.sh" "$V2N4" --running-list "$WORK/N4_run.txt" > "$WORK/N4.log" 2>&1; rcN4=$?
  if [ "$rcN4" -eq 4 ] && ! grep -q 'PIN MISMATCH tolerated' "$WORK/N4.log"; then
    ok "N(n4): MUTATION -- with the tolerance branch cut the SAME snapshot owner refuses rc 4 again (rc=$rcN4) and no tolerated row is printed, so n1 CAN fail"
  else
    bad "N(n4): the mutant gave rc=$rcN4 and still tolerated -- ARM n1 IS NOT TESTING THE REFINEMENT"
    sed 's/^/      /' "$WORK/N4.log"
  fi
else
  bad "N(n4): could not build the no-tolerance mutant -- MUTATION ARM DID NOT RUN"
fi


# ---- R: HARNESS-SYNC-7 -- a path built from LITERAL variables is a KNOWN path -
# THE DEFECT, and it cost the merge queue a whole morning. det162-proof 6015646
# sat as a RUNNING rc-6 blocker on `moved_row_halves.sh` and
# `multiarm_preamble.sh` while reading NEITHER task copy: its driver assigns
# `WT=<a literal worktree path>`, then `WTH=$WT/harness`, then
# `MOVED_ROWS=$WTH/tools/moved_row_halves.sh`, and sources
# `"$WTH/tools/multiarm_preamble.sh"`. Every hop is a literal in the SAME file,
# so the absolute path is as knowable from the text as `/abs/path` is -- the
# parser simply stopped at the `$` and the check then assumed the worst about a
# path it could have computed. That is not caution, it is an unread fact, and
# law 9 does not licence refusing on one.
#
#   r1  THE det162 SHAPE: a three-hop chain to a tree OUTSIDE the task dir is
#       RESOLVED, the sync INSTALLS (rc 0), and the row SAYS how -- with the
#       variable chain and the absolute path, so the clearance can be audited.
#   r2  THE CONTROL, and without it r1 is worthless: the SAME chain shape whose
#       literal points INTO the task tools/ resolves to the file the sync would
#       rewrite and still REFUSES rc 6, match=exact.
#   r3  a hop assigned from a COMMAND SUBSTITUTION is not knowable from the text
#       and stays unresolved -- refused rc 6, match=unresolved (law 9).
#   r4  THE MUTATION: with the chain branch cut, r1's fixture refuses again, so
#       r1 CAN fail.
mkchainjob () {   # $1 = job root dir, $2 = file name, $3.. = body lines
  local jr=$1 fn=$2; shift 2
  mkdir -p "$jr"
  { echo '#!/bin/bash'; for l in "$@"; do printf '%s\n' "$l"; done; } > "$jr/$fn"
  chmod 755 "$jr/$fn"
}

# ---- r1: the det162 chain, resolving OUTSIDE the task tree ------------------
read -r RR1 TR1 V1R1 V2R1 < <(mkfixture R1)
R1OUT=$WORK/R1_outside
mkdir -p "$R1OUT/harness/tools"
printf 'outside tool\n' > "$R1OUT/harness/tools/a_tool.sh"
printf 'outside exec\n' > "$R1OUT/harness/tools/an_exec.sh"
mkchainjob "$TR1/jobroot" r1.sbatch \
  "WT=$R1OUT" \
  'WTH=$WT/harness' \
  'A_TOOL=$WTH/tools/a_tool.sh' \
  'bash "$A_TOOL"' \
  'source "$WTH/tools/an_exec.sh"'
mkstub "$WORK/R1_squeue"
printf '8000071 RUNNING laneR1 %s %s %s\n' "$TR1" "$TR1/jobroot/r1.sbatch" "$TR1/jobroot" > "$WORK/R1_run.txt"
git -C "$RR1" cat-file blob "$V2R1:harness/tools/a_tool.sh" > "$WORK/R1.blob"
BEFORE_R1=$(md5sum "$TR1/tools/a_tool.sh" | awk '{print $1}')
WANT_R1=$(md5sum "$WORK/R1.blob" | awk '{print $1}')
[ "$BEFORE_R1" != "$WANT_R1" ] \
  && ok "R(r1): NON-VACUITY -- tools/a_tool.sh is $BEFORE_R1 and $V2R1 says $WANT_R1, so it IS in the install set and the job DOES name that basename" \
  || bad "R(r1): the fixture tool already matches the commit -- the arm would prove nothing"
runsync "$RR1" "$TR1" "$WORK/R1_squeue" "$V2R1" --running-list "$WORK/R1_run.txt" > "$WORK/R1.log" 2>&1; rcR1=$?
if [ "$rcR1" -eq 0 ] && cmp -s "$WORK/R1.blob" "$TR1/tools/a_tool.sh"; then
  ok "R(r1): a three-hop LITERAL chain to a tree outside the task dir no longer blocks the sync -- rc 0 and a_tool.sh installed with a live RUNNING job naming that basename"
else
  bad "R(r1): rc=$rcR1 -- the det162 shape must resolve and install"; sed 's/^/      /' "$WORK/R1.log"
fi
if grep -q 'read-set OK job=8000071 file=a_tool.sh match=resolved-literal via=A_TOOL->WTH->WT ' "$WORK/R1.log"; then
  ok "R(r1): and it SAYS HOW -- the row carries match=resolved-literal and the whole chain A_TOOL->WTH->WT, not a bare clearance"
else
  bad "R(r1): no row naming match=resolved-literal with the A_TOOL->WTH->WT chain"; grep 'read-set OK' "$WORK/R1.log" | sed 's/^/      /'
fi
if grep -q "read-set OK job=8000071 file=a_tool.sh .*path=$R1OUT/harness/tools/a_tool.sh reason=reads-outside-install-set" "$WORK/R1.log"; then
  ok "R(r1): the ABSOLUTE resolved path is printed and classed reads-outside-install-set -- the reason the refusal was wrong, on the record"
else
  bad "R(r1): the row does not print the absolute path with reason=reads-outside-install-set"; grep 'read-set OK' "$WORK/R1.log" | sed 's/^/      /'
fi
grep -q 'read-set OK job=8000071 file=an_exec.sh match=resolved-literal via=WTH->WT ' "$WORK/R1.log" \
  && ok "R(r1): the TWO-hop form (\`source \"\$WTH/tools/an_exec.sh\"\`, det162's multiarm_preamble shape) resolves too, via=WTH->WT" \
  || { bad "R(r1): the two-hop source form did not resolve"; grep 'read-set OK' "$WORK/R1.log" | sed 's/^/      /'; }

# ---- r2: THE CONTROL -- the same chain, pointing INTO the task tools/ --------
read -r RR2 TR2 V1R2 V2R2 < <(mkfixture R2)
mkchainjob "$TR2/jobroot" r2.sbatch \
  "TT=$TR2" \
  'TTOOLS=$TT/tools' \
  'A_TOOL=$TTOOLS/a_tool.sh' \
  'bash "$A_TOOL"'
mkstub "$WORK/R2_squeue"
printf '8000072 RUNNING laneR2 %s %s %s\n' "$TR2" "$TR2/jobroot/r2.sbatch" "$TR2/jobroot" > "$WORK/R2_run.txt"
BEFORE_R2=$(md5sum "$TR2/tools/a_tool.sh" | awk '{print $1}')
runsync "$RR2" "$TR2" "$WORK/R2_squeue" "$V2R2" --running-list "$WORK/R2_run.txt" > "$WORK/R2.log" 2>&1; rcR2=$?
AFTER_R2=$(md5sum "$TR2/tools/a_tool.sh" | awk '{print $1}')
if [ "$rcR2" -eq 6 ] && [ "$AFTER_R2" = "$BEFORE_R2" ] \
   && grep -q 'SYNC REFUSED rc=6 running=8000072 .*file=a_tool.sh .*match=exact' "$WORK/R2.log"; then
  ok "R(r2): CONTROL -- the SAME literal-chain shape resolving INTO the task tools/ is refused rc 6 match=exact and nothing was written. r1 is the chain being COMPUTED, not the check going blind."
else
  bad "R(r2): rc=$rcR2 before=$BEFORE_R2 after=$AFTER_R2 -- a chain that lands on the installed file must still refuse"; sed 's/^/      /' "$WORK/R2.log"
fi

# ---- r3: a hop assigned from a COMMAND SUBSTITUTION stays unresolved ---------
read -r RR3 TR3 V1R3 V2R3 < <(mkfixture R3)
mkchainjob "$TR3/jobroot" r3.sbatch \
  'ROOT=$(cd /tmp && pwd)' \
  'bash "$ROOT/tools/a_tool.sh"'
mkstub "$WORK/R3_squeue"
printf '8000073 RUNNING laneR3 %s %s %s\n' "$TR3" "$TR3/jobroot/r3.sbatch" "$TR3/jobroot" > "$WORK/R3_run.txt"
BEFORE_R3=$(md5sum "$TR3/tools/a_tool.sh" | awk '{print $1}')
runsync "$RR3" "$TR3" "$WORK/R3_squeue" "$V2R3" --running-list "$WORK/R3_run.txt" > "$WORK/R3.log" 2>&1; rcR3=$?
AFTER_R3=$(md5sum "$TR3/tools/a_tool.sh" | awk '{print $1}')
if [ "$rcR3" -eq 6 ] && [ "$AFTER_R3" = "$BEFORE_R3" ] \
   && grep -q 'SYNC REFUSED rc=6 running=8000073 .*file=a_tool.sh .*match=unresolved' "$WORK/R3.log"; then
  ok "R(r3): a hop whose value is a COMMAND SUBSTITUTION is not in the text, stays unresolved and still refuses rc 6 -- the expansion adds resolutions, it does not invent them"
else
  bad "R(r3): rc=$rcR3 before=$BEFORE_R3 after=$AFTER_R3 -- a \$( ) hop must not be guessed"; sed 's/^/      /' "$WORK/R3.log"
fi

# ---- r4: THE MUTATION -- cut the chain branch and r1 must go red -------------
SRSRC=$(dirname -- "$SYNC")/script_refs.sh
MUT7=$WORK/script_refs_nochain.sh
if [ -f "$SRSRC" ]; then
  sed 's|^\([[:space:]]*\)if ex=.*# HARNESS-SYNC-7 CHAIN BRANCH$|\1if false; then|' "$SRSRC" > "$MUT7"
else
  : > "$MUT7"
fi
if [ -s "$MUT7" ] && bash -n "$MUT7" 2>/dev/null && ! grep -q 'HARNESS-SYNC-7 CHAIN BRANCH' "$MUT7"; then
  read -r RR4 TR4 V1R4 V2R4 < <(mkfixture R4)
  R4OUT=$WORK/R4_outside
  mkdir -p "$R4OUT/harness/tools"
  printf 'outside tool\n' > "$R4OUT/harness/tools/a_tool.sh"
  mkchainjob "$TR4/jobroot" r4.sbatch \
    "WT=$R4OUT" \
    'WTH=$WT/harness' \
    'A_TOOL=$WTH/tools/a_tool.sh' \
    'bash "$A_TOOL"'
  cp -f "$MUT7" "$TR4/tools/script_refs.sh"
  mkstub "$WORK/R4_squeue"
  printf '8000074 RUNNING laneR4 %s %s %s\n' "$TR4" "$TR4/jobroot/r4.sbatch" "$TR4/jobroot" > "$WORK/R4_run.txt"
  runsync "$RR4" "$TR4" "$WORK/R4_squeue" "$V2R4" --running-list "$WORK/R4_run.txt" > "$WORK/R4.log" 2>&1; rcR4=$?
  if [ "$rcR4" -eq 6 ] && grep -q 'SYNC REFUSED rc=6 running=8000074 .*file=a_tool.sh .*match=unresolved' "$WORK/R4.log"; then
    ok "R(r4): MUTATION -- with the chain branch cut, the SAME det162 shape refuses rc 6 match=unresolved again (rc=$rcR4), so r1 CAN fail"
  else
    bad "R(r4): the mutant gave rc=$rcR4 and did not refuse on a_tool.sh -- ARM r1 IS NOT TESTING THE RESOLUTION"
    sed 's/^/      /' "$WORK/R4.log"
  fi
else
  bad "R(r4): could not build the no-chain mutant of script_refs.sh -- MUTATION ARM DID NOT RUN"
fi

# ---- r5: THE SNAPSHOT $0 TRAP, and it is a REGRESSION ARM ------------------
# HARNESS-SYNC-7 shipped its first cut with `$(dirname "$0")` substituted by the
# dirname of the file being parsed, on the reasoning that `$0` is knowable from
# the text. IT IS NOT, HERE. The sync parses SLURM'S OWN SNAPSHOT (`scontrol
# write batch_script` into the sync's temp dir); the job's real `$0` is a script
# on a compute node. The live dry run at 07:16 09-07 measured the cost exactly:
# TEN held readers of `tools/retread_fast_env.sh`, every one of them
# `FAST_ENV=$(dirname "$0")/../retread_fast_env.sh`, were CLEARED against
# `/tmp/harness_sync.XXXXXX/../retread_fast_env.sh` -- a path that does not
# exist and that nothing will ever read. Ten missed refusals, from a resolution
# that looked safer than the `-` it replaced.
#
# This arm is that fixture: a job whose script is handed to the sync from a
# DIFFERENT DIRECTORY than the one it will run in, naming an installed file
# through `$(dirname "$0")`. It must REFUSE.
read -r RR5 TR5 V1R5 V2R5 < <(mkfixture R5)
mkdir -p "$TR5/jobroot" "$WORK/R5_slurmtmp"
mkchainjob "$WORK/R5_slurmtmp" r5.sbatch \
  'FAST=$(dirname "$0")/../tools/a_tool.sh' \
  'bash "$FAST"'
mkstub "$WORK/R5_squeue"
# THE POINT OF THE ARM: the SCRIPT path the sync reads is the snapshot in
# $WORK/R5_slurmtmp, while the job root is $TR5/jobroot -- the production shape.
printf '8000075 PENDING laneR5 %s %s %s\n' "$TR5" "$WORK/R5_slurmtmp/r5.sbatch" "$TR5/jobroot" > "$WORK/R5_run.txt"
BEFORE_R5=$(md5sum "$TR5/tools/a_tool.sh" | awk '{print $1}')
runsync "$RR5" "$TR5" "$WORK/R5_squeue" "$V2R5" --running-list "$WORK/R5_run.txt" > "$WORK/R5.log" 2>&1; rcR5=$?
AFTER_R5=$(md5sum "$TR5/tools/a_tool.sh" | awk '{print $1}')
if [ "$rcR5" -eq 6 ] && [ "$AFTER_R5" = "$BEFORE_R5" ] \
   && grep -q 'SYNC REFUSED rc=6 running=8000075 .*file=a_tool.sh .*match=unresolved' "$WORK/R5.log"; then
  ok "R(r5): REGRESSION ARM -- \$(dirname \"\$0\") in a SLURM SNAPSHOT is NOT resolved against the snapshot's temp dir; the reader still refuses rc 6 match=unresolved and nothing was written"
else
  bad "R(r5): rc=$rcR5 before=$BEFORE_R5 after=$AFTER_R5 -- the snapshot \$0 trap is BACK: ten live refusals were cleared by this once"
  grep -E 'read-set OK|SYNC REFUSED rc=6' "$WORK/R5.log" | sed 's/^/      /'
fi
if grep -q 'read-set OK job=8000075 .*path=/tmp/' "$WORK/R5.log" || grep -q "read-set OK job=8000075 .*path=$WORK/R5_slurmtmp" "$WORK/R5.log"; then
  bad "R(r5): a read-set OK row for 8000075 names a path under the SNAPSHOT's directory -- that is the wrong path, and a wrong path here is a MISSED REFUSAL"
else
  ok "R(r5): and no cleared row anywhere names a path under the snapshot's own directory"
fi


# ---- S: HARNESS-SYNC-8 -- THE FORCE PATH AGAINST ITS OWN POST-CONDITIONS -----
# THE DEFECT, MEASURED 2026-09-07 AFTER THE ONE AUTHORISED FORCE TO 955d086.
# `--force` writes its audit marker to `"$RECORD.force-readset"`, i.e. into the
# SCANNED directory `tools/`, where no commit can ever carry a blob for it. The
# skip arm in harness_drift_check.sh matched the record by EXACT NAME, so the
# record was skipped and its sidecar was not:
#     bash tools/harness_sync.sh --check        -> rc 3  checked=86 clean=85 edited=1
#       SYNC CHECK no-blob  tools/.harness_synced_commit.force-readset
#     bash tools/harness_drift_check.sh 955d086 -> rc 3  ok=84 mismatch=0 missing=1
# `mismatch=0` in both: NOTHING had drifted; the marker was the entire refusal,
# and it red-lines both gates permanently from the moment a force succeeds. The
# phase templates run `--check` as the first line of their drift block and
# "--check clean" is an acceptance criterion for a merge-queue landing, so the
# force written to unblock B30 is what then blocked it.
#
# WHY THE GUARD DID NOT CATCH IT, WHICH IS THE REAL LESSON: every existing arm
# asserted what --force DOES (it installs) and none asserted the state it LEAVES.
# A writer is not guarded until its own readers are run against its output.
#
#   s1  NON-VACUITY + the control: the fixture really does need the force (rc 6
#       without it, nothing written).
#   s2  the force installs (rc 0) and the marker IS on disk.
#   s3  THE POST-CONDITION: `--check` rc 0 CLEAN and `harness_drift_check.sh`
#       rc 0 CLEAN on that same task dir, with the marker still there.
#   s4  the marker is NOT deleted -- it is the audit record of an authorised
#       force and quieting the gate by removing it is the wrong fix.
#   s5  MUTATION: the pre-fix EXACT-NAME skip arm restored -> `--check` rc 3
#       with the no-blob row. Without this, s3 passes for free.
#   s6  HARNESS-SYNC-8 FINDING 2: a task dir with NO tools/harness_push_check.sh
#       still prints a REAL `### PUSH LAG` row -- the three-hop fallback -- and
#       never `branch=? unpushed=?`, which is a criterion with no producer.
#   s7  `--check` reports the ABSENT set. It was reported only by the WRITING
#       mode, which is why harness_push_check.sh was missing for a whole lane
#       with nobody told.
read -r RS TS V1S V2S < <(mkfixture S)
mkchainjob "$TS/jobroot" s.sbatch \
  "TT=$TS" \
  'TTOOLS=$TT/tools' \
  'bash "$TTOOLS/a_tool.sh"'
mkstub "$WORK/S_squeue"
printf '8000081 RUNNING laneS %s %s %s\n' "$TS" "$TS/jobroot/s.sbatch" "$TS/jobroot" > "$WORK/S_run.txt"
BEFORE_S=$(md5sum "$TS/tools/a_tool.sh" | awk '{print $1}')
runsync "$RS" "$TS" "$WORK/S_squeue" "$V2S" --running-list "$WORK/S_run.txt" > "$WORK/S1.log" 2>&1; rcS1=$?
AFTER_S1=$(md5sum "$TS/tools/a_tool.sh" | awk '{print $1}')
if [ "$rcS1" -eq 6 ] && [ "$AFTER_S1" = "$BEFORE_S" ]; then
  ok "S(s1): NON-VACUITY -- without --force this fixture refuses rc 6 and writes nothing, so the force path below is really exercised"
else
  bad "S(s1): rc=$rcS1 before=$BEFORE_S after=$AFTER_S1 -- wanted rc 6; the force arms would prove nothing"
  sed 's/^/      /' "$WORK/S1.log"
fi

SMARK=$TS/tools/.harness_synced_commit.force-readset
runsync "$RS" "$TS" "$WORK/S_squeue" "$V2S" --running-list "$WORK/S_run.txt" \
        --force --reason "guard fixture: the single reader is a fixture job" > "$WORK/S2.log" 2>&1; rcS2=$?
if [ "$rcS2" -eq 0 ] && grep -q '### SYNC FORCED over the read set of' "$WORK/S2.log"; then
  ok "S(s2): --force --reason installs over the read set, rc 0"
else
  bad "S(s2): rc=$rcS2 -- the authorised force did not install"; sed 's/^/      /' "$WORK/S2.log"
fi
if [ -s "$SMARK" ]; then
  ok "S(s2): the audit marker is on disk: $(cat "$SMARK")"
else
  bad "S(s2): no force-readset marker was written -- the force left no audit record and s3 would pass for free"
fi

runsync "$RS" "$TS" "$WORK/S_squeue" --check > "$WORK/S3c.log" 2>&1; rcS3c=$?
if [ "$rcS3c" -eq 0 ] && grep -q '### SYNC CHECK CLEAN' "$WORK/S3c.log"; then
  ok "S(s3): after an authorised force, \`--check\` is rc 0 CLEAN -- the writer's own reader tolerates what it wrote"
else
  bad "S(s3): --check rc=$rcS3c after a force. THE FORCE PATH RED-LINES ITS OWN GATE (HARNESS-SYNC-8 finding 1):"
  grep -E 'no-blob|SYNC CHECK SUMMARY|SYNC CHECK REFUSED' "$WORK/S3c.log" | sed 's/^/      /'
fi
HARNESS_REPO="$RS" HARNESS_TASK_DIR="$TS" bash "$TS/tools/harness_drift_check.sh" "$V2S" \
  > "$WORK/S3d.log" 2>&1; rcS3d=$?
if [ "$rcS3d" -eq 0 ] && grep -q 'missing=0' "$WORK/S3d.log"; then
  ok "S(s3): and harness_drift_check.sh is rc 0 with missing=0 on the same task dir -- both gates, not one"
else
  bad "S(s3): drift rc=$rcS3d after a force -- the marker is being read as a task copy that went missing:"
  grep -E 'DRIFT SUMMARY|DRIFT REFUSED|no-blob' "$WORK/S3d.log" | sed 's/^/      /'
fi
[ -s "$SMARK" ] \
  && ok "S(s4): and the marker is STILL on disk after both gates ran -- the gate was fixed, the audit record was not deleted" \
  || bad "S(s4): the marker is gone -- something quieted the gate by deleting the evidence of the force"

# s5 MUTATION: restore the pre-fix EXACT-NAME skip arm and the same force must
# red-line --check again. A guard that cannot fail is a defect.
read -r RS5 TS5 V1S5 V2S5 < <(mkfixture S5)
mkchainjob "$TS5/jobroot" s5.sbatch "TT=$TS5" 'TTOOLS=$TT/tools' 'bash "$TTOOLS/a_tool.sh"'
mkstub "$WORK/S5_squeue"
printf '8000082 RUNNING laneS5 %s %s %s\n' "$TS5" "$TS5/jobroot/s5.sbatch" "$TS5/jobroot" > "$WORK/S5_run.txt"
#
# THE MUTATION IS APPLIED **AFTER** THE FORCE, AND THAT ORDER IS THE ARM.
# Job 6019634 measured the first cut of this arm doing it the other way round
# and it went green for a false reason: `--force` performs the whole sync, which
# INSTALLS `tools/harness_drift_check.sh` from the commit -- so a mutation
# written into the task copy beforehand is overwritten by the very sync it is
# meant to change. The gate being mutated runs AFTER the force, so the mutation
# belongs there too. (rc 3 alone would then be ambiguous -- the mutated task
# copy is itself an EDITED file -- so the arm asserts the SPECIFIC no-blob row
# for the marker, not the exit code.)
S5MARK=$TS5/tools/.harness_synced_commit.force-readset
if ! grep -q '|\.harness_synced_commit\.\*)' "$TS5/tools/harness_drift_check.sh"; then
  bad "S(s5): the harness_drift_check.sh this guard ships beside does NOT carry the .harness_synced_commit.* glob -- this tree is the PRE-FIX state, s3 above cannot be mutated, and a forced sync here permanently red-lines both gates"
else
  runsync "$RS5" "$TS5" "$WORK/S5_squeue" "$V2S5" --running-list "$WORK/S5_run.txt" \
          --force --reason "guard fixture mutation" > "$WORK/S5f.log" 2>&1; rcS5f=$?
  cp -f "$TS5/tools/harness_drift_check.sh" "$WORK/S5.drift.before"
  sed -i 's/|\.harness_synced_commit|\.harness_synced_commit\.\*)/|.harness_synced_commit)/' \
      "$TS5/tools/harness_drift_check.sh"
  if cmp -s "$WORK/S5.drift.before" "$TS5/tools/harness_drift_check.sh"; then
    bad "S(s5): the mutation edited nothing -- this arm is asserting against an unmutated file and cannot fail"
  elif [ ! -s "$S5MARK" ]; then
    bad "S(s5): the force wrote no marker, so there is nothing for the mutated gate to trip over and s5 measures nothing (force rc=$rcS5f)"
  else
    runsync "$RS5" "$TS5" "$WORK/S5_squeue" --check > "$WORK/S5c.log" 2>&1; rcS5c=$?
    if [ "$rcS5f" -eq 0 ] && [ "$rcS5c" -eq 3 ] \
       && grep -q 'SYNC CHECK no-blob  tools/.harness_synced_commit.force-readset' "$WORK/S5c.log"; then
      ok "S(s5): MUTATION -- with the exact-name skip arm restored, the marker the force wrote makes --check refuse rc 3 with its own no-blob row. s3 CAN fail; this is the state 955d086 was left in."
    else
      bad "S(s5): force rc=$rcS5f check rc=$rcS5c -- the pre-fix shape did NOT reproduce the no-blob refusal on the marker, so s3 proves nothing"
      grep -E 'no-blob|SYNC CHECK SUMMARY' "$WORK/S5c.log" | sed 's/^/      /'
    fi
  fi
fi

# s6: the callee is ABSENT from the task dir and the criterion must still have a
# producer. This is the exact state 955d086's sync left behind: a NEW file enters
# only via --add, so the freshly-installed caller called a path that never existed.
read -r RS6 TS6 V1S6 V2S6 < <(mkfixture S6)
rm -f "$TS6/tools/harness_push_check.sh"
mkstub "$WORK/S6_squeue"
runsync "$RS6" "$TS6" "$WORK/S6_squeue" "$V2S6" > "$WORK/S6s.log" 2>&1
runsync "$RS6" "$TS6" "$WORK/S6_squeue" --check > "$WORK/S6.log" 2>&1; rcS6=$?
S6ROW=$(grep -m1 '### PUSH LAG' "$WORK/S6.log" || true)
if [ -n "$S6ROW" ] && ! printf '%s' "$S6ROW" | grep -q 'branch=?'; then
  ok "S(s6): with tools/harness_push_check.sh ABSENT the fallback still produces a real row: $S6ROW"
else
  bad "S(s6): the push-lag criterion has no producer when the callee is absent (row: '${S6ROW:-<none>}') -- 'unpushed=0' cannot then be reported as satisfied"
fi
grep -q 'PUSH LAG reader:' "$WORK/S6.log" \
  && ok "S(s6): and it NAMES which copy answered -- a clearance whose source is unknown is not auditable" \
  || bad "S(s6): no '### PUSH LAG reader:' row -- the hop that answered is not on the record"

# s7: --check reports the absent set (it was a write-mode-only report).
if grep -q '### SYNC ABSENT' "$WORK/S6.log"; then
  ok "S(s7): \`--check\` reports the ABSENT set: $(grep -m1 '### SYNC ABSENT' "$WORK/S6.log")"
else
  bad "S(s7): --check printed no ABSENT report -- a lane running the read-only mode cannot see that the commit carries files this task dir does not have"
fi

# ---- T: HARNESS-SYNC-9 -- A NO-OP SYNC MUST NOT REFUSE ITSELF ---------------
# THE DEFECT, WITH THE LANDING IT HELD. B30's landing 2026-09-07T07:47 refused
# rc 4 having moved NOTHING: the task dir was ALREADY at the requested commit,
# so the install set was EMPTY, so the read-set block (guarded on
# `[ -s "$INSTSET" ]`) never ran, so `$RSDET` was empty -- and rc 4's refinement
# read every pin-mismatched PENDING job as `read-set-not-examined` and refused.
# 21 rows, three jobs, `rc4_tolerated=0`, and not one byte would have moved for
# any of them. HARNESS-SYNC-6's own comment names the case ("or the install set
# was empty so no job was examined at all") and gives it law 9's verdict, which
# is the right verdict when there is something to look FOR and the wrong one
# when there is not. A sync that installs nothing cannot strand anything.
#   t1  at-commit fixture + a pin-mismatched PENDING job -> rc 0, the NO-OP row,
#       the pin TOLERATED on reason=nothing-installed (never in silence), no
#       rc-4 refusal, and nothing written
#   t2  the record BEHIND the disk on a no-op -> `### RECORD ADVANCED <old> ->
#       <new> bytes-identical` and the record file actually moves
#   t3  NON-VACUITY: the SAME fixture with ONE file differing -> the old rc 4,
#       reason=read-set-not-examined, unchanged. The fix is not a hole.
#   t4  the blobless-file hole: an empty INSTALL set is not a no-op when a mapped
#       task file has no blob at the commit -- that run still owes rc 5, and a
#       no-op reading only `$INSTSET` would have swallowed it
#   t5  THE MUTATION: the short-circuit cut -> t1's fixture refuses rc 4 again,
#       so t1 CAN fail
read -r RT TT V1T V2T < <(mkfixture T)
mkstub "$WORK/T_none"                                  # no jobs at all
runsync "$RT" "$TT" "$WORK/T_none" "$V2T" > "$WORK/T0.log" 2>&1; rcT0=$?
[ "$rcT0" -eq 0 ] && ok "T(t0): the SETUP sync (clean tree -> $V2T) rc=0, so t1 starts AT the commit" \
  || { bad "T(t0): the setup sync rc=$rcT0 -- arm T's fixture is not at the commit and the arm proves nothing"; sed 's/^/      /' "$WORK/T0.log"; }
# t1: now pin a queued job to the OLDER commit and sync to the SAME commit again.
mkdir -p "$TT/lane1"; printf '%s\n' "$V1T" > "$TT/lane1/HARNESS_COMMIT"
mkstub "$WORK/T_squeue" "9000010 lane1-relock"
BEFORE_T=$(md5sum "$TT/tools/a_tool.sh" | awk '{print $1}')
runsync "$RT" "$TT" "$WORK/T_squeue" "$V2T" > "$WORK/T1.log" 2>&1; rcT1=$?
[ "$rcT1" -eq 0 ] \
  && ok "T(t1): THE FIX -- a sync whose install set is EMPTY no longer refuses for a pin-mismatched PENDING job (rc 0)" \
  || { bad "T(t1): rc=$rcT1, wanted 0 -- a sync that installs nothing is still refusing"; sed 's/^/      /' "$WORK/T1.log"; }
grep -qE "^### SYNC NO-OP commit=$V2T files=[0-9]+ installed=0 unchanged=[0-9]+\$" "$WORK/T1.log" \
  && ok "T(t1): and it SAYS it did nothing, in one parseable row: $(grep -m1 '^### SYNC NO-OP' "$WORK/T1.log")" \
  || { bad "T(t1): no well-formed '### SYNC NO-OP commit=$V2T files=<n> installed=0 unchanged=<n>' row"; grep -m1 'SYNC NO-OP' "$WORK/T1.log" | sed 's/^/      /'; }
grep -qE "^### PIN MISMATCH tolerated jid=9000010 reason=nothing-installed name=lane1-relock pin=$TT/lane1/HARNESS_COMMIT pinned=$V1T\$" "$WORK/T1.log" \
  && ok "T(t1): the pin mismatch is TOLERATED on a whole, parseable row -- reason=nothing-installed, never in silence" \
  || { bad "T(t1): the tolerated row is missing or malformed"; grep 'PIN MISMATCH' "$WORK/T1.log" | sed 's/^/      /'; }
grep -q 'reason=read-set-not-examined' "$WORK/T1.log" \
  && { bad "T(t1): it still classed the job read-set-not-examined -- the empty install set is still being read as 'we did not look'"; sed 's/^/      /' "$WORK/T1.log"; } \
  || ok "T(t1): and NOT as read-set-not-examined -- there was nothing to examine, which is not the same as declining to look"
grep -q 'SYNC REFUSED' "$WORK/T1.log" \
  && { bad "T(t1): a refusal was printed on a run that installs nothing"; sed 's/^/      /' "$WORK/T1.log"; } \
  || ok "T(t1): and no SYNC REFUSED row of any kind is on the page"
[ "$(md5sum "$TT/tools/a_tool.sh" | awk '{print $1}')" = "$BEFORE_T" ] \
  && ok "T(t1): and rc 0 wrote nothing -- every file was already the commit's bytes" \
  || bad "T(t1): the no-op run changed a task copy"
grep -q '### SYNC PIN REPORT' "$WORK/T1.log" \
  && ok "T(t1): the evidence packet survives the no-op -- the PIN REPORT still prints (it is not an early exit)" \
  || { bad "T(t1): the no-op skipped the PIN REPORT, so a lane loses the list of job roots that will die"; sed 's/^/      /' "$WORK/T1.log"; }
# t2: the record BEHIND the bytes. The disk is at $V2T; say the record is at $V1T.
printf '%s\n' "$V1T" > "$TT/tools/.harness_synced_commit"
runsync "$RT" "$TT" "$WORK/T_squeue" "$V2T" > "$WORK/T2.log" 2>&1; rcT2=$?
grep -qE "^### RECORD ADVANCED $V1T -> $V2T bytes-identical\$" "$WORK/T2.log" \
  && ok "T(t2): a no-op whose RECORD was behind says so -- '### RECORD ADVANCED $V1T -> $V2T bytes-identical'" \
  || { bad "T(t2): rc=$rcT2 and no RECORD ADVANCED row"; grep -E 'RECORD|NO-OP' "$WORK/T2.log" | sed 's/^/      /'; }
[ "$(cat "$TT/tools/.harness_synced_commit")" = "$V2T" ] \
  && ok "T(t2): and the record file actually moved to $V2T, so --check compares against the truth" \
  || bad "T(t2): the record still says '$(cat "$TT/tools/.harness_synced_commit")'"
grep -q 'RECORD ADVANCED' "$WORK/T1.log" \
  && bad "T(t2): t1's run claimed RECORD ADVANCED with the record ALREADY at the commit -- the row is unconditional" \
  || ok "T(t2): and the row is NOT printed when the record was already right (t1 has none)"
# t3: NON-VACUITY. One file differs, so the install set is not empty, and the
# SAME pinned job must produce the SAME rc 4 it always did.
git -C "$RT" cat-file blob "$V1T:harness/tools/a_tool.sh" > "$TT/tools/a_tool.sh"
BEFORE_T3=$(md5sum "$TT/tools/a_tool.sh" | awk '{print $1}')
runsync "$RT" "$TT" "$WORK/T_squeue" "$V2T" > "$WORK/T3.log" 2>&1; rcT3=$?
[ "$rcT3" -eq 4 ] \
  && ok "T(t3): NON-VACUITY -- with ONE file differing the same pinned PENDING job still refuses rc 4" \
  || { bad "T(t3): rc=$rcT3, wanted 4 -- the no-op arm has widened into a hole"; sed 's/^/      /' "$WORK/T3.log"; }
grep -q '^###   state=PENDING 9000010 lane1-relock .*reason=read-set-not-examined' "$WORK/T3.log" \
  && ok "T(t3): and with the unchanged reason -- read-set-not-examined, exactly as before HARNESS-SYNC-9" \
  || { bad "T(t3): the rc-4 row's reason changed"; grep -E '^###   state=PENDING' "$WORK/T3.log" | sed 's/^/      /'; }
grep -q 'SYNC NO-OP' "$WORK/T3.log" \
  && bad "T(t3): it printed a NO-OP row for a run with a non-empty install set" \
  || ok "T(t3): and no NO-OP row is printed when there IS something to install"
[ "$(md5sum "$TT/tools/a_tool.sh" | awk '{print $1}')" = "$BEFORE_T3" ] \
  && ok "T(t3): and it wrote nothing while refusing" || bad "T(t3): it wrote while refusing"
# t4: THE BLOBLESS HOLE. Nothing to install, but a mapped task file has no blob
# at the commit -- the install loop calls that NO-BLOB and exits 5, so this run
# is NOT a no-op. Reading only $INSTSET would have turned a rc 5 into a rc 0.
read -r RT4 TT4 V1T4 V2T4 < <(mkfixture T4)
mkstub "$WORK/T4_squeue"
runsync "$RT4" "$TT4" "$WORK/T4_squeue" "$V2T4" > "$WORK/T4s.log" 2>&1
printf 'nobody committed this\n' > "$TT4/tools/an_orphan.sh"
runsync "$RT4" "$TT4" "$WORK/T4_squeue" "$V2T4" > "$WORK/T4.log" 2>&1; rcT4=$?
if [ "$rcT4" -eq 5 ] && ! grep -q 'SYNC NO-OP' "$WORK/T4.log"; then
  ok "T(t4): a mapped task file with NO BLOB at the commit is NOT a no-op -- rc 5 is still owed and no NO-OP row is printed"
else
  bad "T(t4): rc=$rcT4 (wanted 5) or a NO-OP row was printed -- the short-circuit swallowed a would-be rc 5"
  grep -E 'NO-OP|NO-BLOB|SUMMARY' "$WORK/T4.log" | sed 's/^/      /'
fi
# t5: THE MUTATION -- cut exactly the short-circuit and t1's fixture must refuse.
MUT9=$WORK/harness_sync_nonoop.sh
sed 's|.*# HARNESS-SYNC-9 NO-OP SHORT-CIRCUIT$|if false; then|' "$SYNC" > "$MUT9"
if bash -n "$MUT9" 2>/dev/null && ! grep -q 'HARNESS-SYNC-9 NO-OP SHORT-CIRCUIT' "$MUT9"; then
  read -r RT5 TT5 V1T5 V2T5 < <(mkfixture T5)
  mkstub "$WORK/T5_none"
  runsync "$RT5" "$TT5" "$WORK/T5_none" "$V2T5" > "$WORK/T5s.log" 2>&1
  mkdir -p "$TT5/lane1"; printf '%s\n' "$V1T5" > "$TT5/lane1/HARNESS_COMMIT"
  mkstub "$WORK/T5_squeue" "9000011 lane1-relock"
  cp -f "$MUT9" "$TT5/tools/harness_sync.sh"
  HARNESS_REPO="$RT5" HARNESS_TASK_DIR="$TT5" HARNESS_SQUEUE="$WORK/T5_squeue" \
    bash "$TT5/tools/harness_sync.sh" "$V2T5" > "$WORK/T5.log" 2>&1; rcT5=$?
  if [ "$rcT5" -eq 4 ] && grep -q 'reason=read-set-not-examined' "$WORK/T5.log" \
     && ! grep -q 'SYNC NO-OP' "$WORK/T5.log"; then
    ok "T(t5): MUTATION -- with the short-circuit cut the SAME at-commit fixture refuses rc 4 read-set-not-examined again (rc=$rcT5), so t1 CAN fail"
  else
    bad "T(t5): the mutant gave rc=$rcT5 and did not reproduce the refusal -- ARM t1 IS NOT TESTING THE SHORT-CIRCUIT"
    sed 's/^/      /' "$WORK/T5.log"
  fi
else
  bad "T(t5): could not build the no-short-circuit mutant -- MUTATION ARM DID NOT RUN"
fi

echo "### harness_sync_guard: pass=$pass fail=$fail"
[ "$fail" -eq 0 ] || exit 1
exit 0
