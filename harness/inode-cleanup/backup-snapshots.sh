#!/bin/bash
set -u
export PATH=/users/glvov/.local/bin:$PATH
ROOT=/oscar/data/stellex/glvov
SRC=$ROOT/retread-src
WTS=$ROOT/agrescap/worktrees
BASE=8722c4a2a2dd594f1fb43cb113608d770413a391
TASK=$ROOT/agrescap/tasks/retread-4-11/p6-inode-cleanup

do_one() {
  local NAME=$1 DATE=$2
  local D=$ROOT/$NAME
  local BR=snapshot/$NAME-$DATE
  local W=$WTS/snap-$NAME-$DATE
  echo "############ $NAME -> $BR ############"
  [ -e "$W" ] && { echo "SKIP: $W exists"; return 1; }
  git -C "$SRC" worktree add --quiet -b "$BR" "$W" "$BASE" || { echo "FAIL worktree add"; return 1; }
  mkdir -p "$W/snapshot"
  rsync -a \
    --exclude='.git' \
    --exclude='target/' \
    --exclude='cargo-target' \
    --exclude='cargo-target-*' \
    --exclude='.pixi/' \
    --exclude='envs/' \
    --exclude='node_modules/' \
    "$D/" "$W/snapshot/" || { echo "FAIL rsync"; return 1; }
  echo "--- copied size / inodes ---"
  du -sh "$W/snapshot"; du --inodes -s "$W/snapshot"

  # manifest: worktree registration + HEAD, so the pointers survive deletion
  {
    echo "# Snapshot manifest: $D"
    echo
    echo "Copied 2026-09-02 by agrescap/tasks/retread-4-11/p6-inode-cleanup/backup-snapshots.sh"
    echo "Base commit for this branch: $BASE"
    echo
    echo "## Registered git worktrees found inside this snapshot"
    echo
    printf '%-60s %-46s %s\n' "PATH (relative)" "HEAD" "OWNING REPO"
    find "$D" -mindepth 1 -maxdepth 3 -name .git -type f | sort | while read -r g; do
      wdir=$(dirname "$g")
      printf '%-60s %-46s %s\n' "${wdir#$D/}" "$(git -C "$wdir" rev-parse HEAD 2>/dev/null)" "$(sed 's|gitdir: ||; s|/.git/worktrees/.*||' "$g")"
    done
    echo
    echo "## Excluded from this commit (rebuildable)"
    echo "- target/ and cargo-target*/ : cargo build output, rebuild with 'cargo build'"
    echo "- .pixi/ and envs/          : pixi environments, rebuild with 'pixi install' from the committed manifest+lock"
    echo "- .git pointer files        : worktree registrations, recorded in the table above"
  } > "$W/snapshot/SNAPSHOT-MANIFEST.md"

  ( cd "$W" && git add -A ) || { echo "FAIL git add"; return 1; }
  echo "--- staged entry count ---"
  ( cd "$W" && git status --porcelain | wc -l )
  echo "--- 10 largest staged files (bytes) ---"
  ( cd "$W" && git diff --cached --name-only -z | xargs -0 -I{} du -b "{}" 2>/dev/null | sort -n | tail -10 )
  echo "--- any staged file over 50MB? ---"
  ( cd "$W" && git diff --cached --name-only -z | xargs -0 -I{} du -b "{}" 2>/dev/null | awk '$1>52428800' )

  cat > "$TASK/msg-$NAME.txt" <<MSG
snapshot: preserve $D (plain copy, $DATE) as git history

Source path : $D
Snapshot date: 2026-08-${DATE:6:2}
Base commit : $BASE
  ("fix: give the drift fixture the origin field the merge required", 2026-08-12)

Every source tree inside this snapshot is a REGISTERED git worktree of
retread-src (or of imprint-data) detached at its base commit, not an
untracked plain copy. The whole snapshot is committed verbatim under
snapshot/ so nothing that is not already reachable from git history is
lost; SNAPSHOT-MANIFEST.md records each worktree's path, HEAD and owning
repo.

Excluded, because each is mechanically rebuildable:
  target/, cargo-target*/  cargo build output -> 'cargo build'
  .pixi/, envs/            pixi environments  -> 'pixi install' from the
                           committed manifest + lock
  .git                     worktree pointer files; the registrations they
                           encode are in SNAPSHOT-MANIFEST.md instead
MSG
  ( cd "$W" && git commit --quiet -F "$TASK/msg-$NAME.txt" ) || { echo "FAIL commit"; return 1; }
  echo "--- commit ---"
  git -C "$W" log -1 --format='%H %ci %s'
  echo "--- files added vs base ---"
  git -C "$W" diff --name-only "$BASE" HEAD | wc -l
  echo
}

do_one retread-product-fix      20260816
do_one retread-recovery         20260814
do_one retread-product-baseline 20260815
echo "############ ALL DONE ############"
git -C "$SRC" branch --list 'snapshot/*' -v
