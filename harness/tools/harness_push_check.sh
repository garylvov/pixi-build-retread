#!/usr/bin/env bash
# harness_push_check.sh -- "how many commits exist ONLY on this disk?"
#
# WHY THIS EXISTS. On 2026-09-07 HARNESS-CONSOL-9 found `private/harness/
# tools-20260902` at 6b3b669 while the worktree tip was d530f4b: SEVENTEEN
# commits from four lanes existed only under `agrescap/worktrees`, which
# CLAUDE.md law 7 says is one `rm` from gone. Nothing printed that state, so
# nothing could notice it. This is the reader.
#
#   usage: harness_push_check.sh [--repo <dir>] [--remote <name>] [--branch <b>]
#
# It prints ONE row and its rc IS the verdict, so a checklist can branch on it:
#
#   ### PUSH LAG branch=<b> unpushed=<n> remote=<name> tip=<sha> remote=<sha> src=<how>
#
#   rc 0  n = 0, everything on this disk is also on the remote
#   rc 1  n > 0, there are commits only here -- the row names how many
#   rc 2  cannot tell (no such repo/remote/branch, or the remote is unreachable
#         AND there is no remote-tracking ref to fall back on). NEVER silent:
#         law 9 -- an unanswerable question is announced, not defaulted to 0.
#
# `src=ls-remote` means the number was read from the REMOTE just now; `src=
# tracking` means the network read failed and the answer came from the local
# remote-tracking ref, which can be stale in the SAFE direction only if nobody
# else pushed. The row always says which, because a lag number nobody can date
# is the same defect as no row at all.
set -uo pipefail
REPO=${HARNESS_REPO:-/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools}
REMOTE=private
BRANCH=
while [ $# -gt 0 ]; do
  case "$1" in
    --repo)   REPO=$2; shift 2 ;;
    --remote) REMOTE=$2; shift 2 ;;
    --branch) BRANCH=$2; shift 2 ;;
    *) echo "### PUSH LAG UNKNOWN ARG '$1'" >&2; exit 2 ;;
  esac
done

git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || {
  echo "### PUSH LAG branch=? unpushed=? remote=$REMOTE tip=? remote=? src=none  (not a git repo: $REPO)"; exit 2; }
[ -n "$BRANCH" ] || BRANCH=$(git -C "$REPO" symbolic-ref --quiet --short HEAD 2>/dev/null)
[ -n "$BRANCH" ] || {
  echo "### PUSH LAG branch=DETACHED unpushed=? remote=$REMOTE tip=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) remote=? src=none"; exit 2; }
TIP=$(git -C "$REPO" rev-parse "$BRANCH" 2>/dev/null) || {
  echo "### PUSH LAG branch=$BRANCH unpushed=? remote=$REMOTE tip=? remote=? src=none  (no such branch)"; exit 2; }

SRC=ls-remote
RTIP=$(git -C "$REPO" ls-remote --heads "$REMOTE" "$BRANCH" 2>/dev/null | awk '{print $1}' | head -1)
if [ -z "$RTIP" ]; then
  SRC=tracking
  RTIP=$(git -C "$REPO" rev-parse --quiet --verify "refs/remotes/$REMOTE/$BRANCH" 2>/dev/null)
fi
if [ -z "$RTIP" ]; then
  echo "### PUSH LAG branch=$BRANCH unpushed=ALL remote=$REMOTE tip=${TIP:0:7} remote=ABSENT src=$SRC"
  echo "###   the branch is on NO remote -- every commit on it exists only on this disk (law 7)"
  exit 1
fi
# `<remote>..<tip>` is exactly "reachable from the tip and not from the remote",
# which is the set a push would send. It counts a diverged branch's own commits
# too, which is right: those are also only here.
N=$(git -C "$REPO" rev-list --count "$RTIP..$TIP" 2>/dev/null)
[ -n "$N" ] || {
  echo "### PUSH LAG branch=$BRANCH unpushed=? remote=$REMOTE tip=${TIP:0:7} remote=${RTIP:0:7} src=$SRC  (remote sha not in this repo -- fetch first)"; exit 2; }
echo "### PUSH LAG branch=$BRANCH unpushed=$N remote=$REMOTE tip=${TIP:0:7} remote=${RTIP:0:7} src=$SRC"
[ "$N" -eq 0 ] && exit 0
if git -C "$REPO" merge-base --is-ancestor "$RTIP" "$TIP" 2>/dev/null; then
  echo "###   fast-forward: bash -c 'git -C $REPO push $REMOTE $BRANCH'"
else
  echo "###   DIVERGED from $REMOTE/$BRANCH -- do NOT force; reconcile first"
fi
exit 1
