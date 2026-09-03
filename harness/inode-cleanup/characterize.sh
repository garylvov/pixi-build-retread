#!/bin/bash
# Characterise the three plain-copy snapshot dirs. Read-only.
set -u
ROOT=/oscar/data/stellex/glvov
OUT=$ROOT/agrescap/tasks/retread-4-11/p6-inode-cleanup
export PATH=/users/glvov/.local/bin:$PATH

for SNAP in retread-product-fix retread-recovery retread-product-baseline; do
  D=$ROOT/$SNAP
  echo "################ SNAPSHOT $SNAP ################"
  echo "--- total du -sh ---"; du -sh "$D"
  echo "--- total inodes ---"; du --inodes -s "$D"
  echo "--- per-child bytes+inodes ---"
  find "$D" -mindepth 1 -maxdepth 1 | sort | while read -r c; do
    printf '%-70s %10s %10s\n' "${c#$D/}" "$(du -sh "$c" | cut -f1)" "$(du --inodes -s "$c" | cut -f1)"
  done
  echo "--- worktree candidates (dirs with .git) and their state ---"
  find "$D" -mindepth 1 -maxdepth 3 -name .git | sort | while read -r g; do
    W=$(dirname "$g")
    echo "=== WORKTREE ${W#$ROOT/} ==="
    echo "  gitdir: $(cat "$g" 2>/dev/null)"
    echo "  HEAD:   $(git -C "$W" rev-parse HEAD 2>&1)"
    echo "  desc:   $(git -C "$W" log -1 --format='%H %ci %s' 2>&1)"
    echo "  --- parts of this worktree ---"
    for part in target .pixi envs .git-artifacts node_modules; do
      [ -e "$W/$part" ] && printf '    %-16s %10s %10s inodes\n' "$part" "$(du -sh "$W/$part"|cut -f1)" "$(du --inodes -s "$W/$part"|cut -f1)"
    done
    echo "  --- source size excluding target/.pixi/envs ---"
    du -sh --exclude=target --exclude=.pixi --exclude=envs "$W" 2>/dev/null | sed 's/^/    /'
    du --inodes -s --exclude=target --exclude=.pixi --exclude=envs "$W" 2>/dev/null | sed 's/^/    /'
    echo "  --- git status --porcelain (tracked diffs + untracked, ignored excluded) ---"
    git -C "$W" status --porcelain 2>&1 | head -300
    echo "  --- status line count ---"
    echo "    $(git -C "$W" status --porcelain 2>/dev/null | wc -l) entries"
  done
  echo
done
echo "################ DONE ################"
