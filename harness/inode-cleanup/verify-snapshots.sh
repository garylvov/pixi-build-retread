#!/bin/bash
set -u
ROOT=/oscar/data/stellex/glvov
WTS=$ROOT/agrescap/worktrees
verify() {
  local NAME=$1 DATE=$2
  local W=$WTS/snap-$NAME-$DATE
  echo "############ VERIFY $NAME ############"
  echo "--- git status in backup worktree (expect empty) ---"
  git -C "$W" status --porcelain
  echo "    [$(git -C "$W" status --porcelain | wc -l) entries]"
  echo "--- diff -rq source vs committed copy (expect only excluded/.git noise) ---"
  diff -rq --exclude=target --exclude='cargo-target' --exclude='cargo-target-*' \
       --exclude=.pixi --exclude=envs --exclude=.git --exclude=SNAPSHOT-MANIFEST.md \
       "$ROOT/$NAME" "$W/snapshot" 2>&1 | head -40
  echo "    [$(diff -rq --exclude=target --exclude='cargo-target' --exclude='cargo-target-*' --exclude=.pixi --exclude=envs --exclude=.git --exclude=SNAPSHOT-MANIFEST.md "$ROOT/$NAME" "$W/snapshot" 2>&1 | wc -l) diff lines]"
  echo "--- commit reachable ---"
  git -C "$ROOT/retread-src" rev-parse "snapshot/$NAME-$DATE"
  echo "--- files in commit under snapshot/ ---"
  git -C "$ROOT/retread-src" ls-tree -r --name-only "snapshot/$NAME-$DATE" -- snapshot | wc -l
  echo
}
verify retread-product-fix      20260816
verify retread-recovery         20260814
verify retread-product-baseline 20260815
echo "############ VERIFY DONE ############"
