#!/usr/bin/env bash
# gate_head_pin.sh -- REFUSE unless a gate's worktree is still the exact commit
# the gate was submitted against, and still clean.  L3-1b-5.
#
# WHY THIS EXISTS, and it is one lane's measured defect and not a hypothetical.
# `gate_build.sh` read `git rev-parse --short HEAD` in TWO places -- once in its
# opening `### host=...` row and once at the very end to NAME the binsnap
# directory -- and checked `git status --porcelain` exactly ONCE, at entry.
# Between those two reads sits a release build, an all-targets build and a
# `cargo test --lib`: on this campaign that is fifteen to twenty-five minutes in
# which any lane holding the same worktree can commit.  L3-1b job 5918073 did
# exactly that to itself (a commit landed at 17:10 into a worktree the gate had
# read as `dirty=0` at 17:06:49) and job 5918567 did worse -- it WROTE
# `binsnaps/cand-9fc0d7d` with sha256 `eefd705a...` for a binary that was NOT
# built from `9fc0d7d`.  Nothing failed.  The mislabelled binsnap was caught
# only because that lane re-ran a pinned gate and compared the two shas by hand.
#
# A GATE IS PINNED TO A COMMIT, not to a directory another actor can move.  So
# the pin is checked at BOTH ends of the build and this file is the check, in
# one place, called twice, so the two calls cannot drift apart.
#
#   usage: gate_head_pin.sh <worktree> <expect-short-head> <phase-label>
#
#   rc 0   HEAD is <expect-short-head> and the tree is clean
#   rc 4   usage / not a git worktree
#   rc 12  HEAD is NOT <expect-short-head>  (the worktree moved, or the gate was
#          submitted against the wrong commit -- both are refusals)
#   rc 13  HEAD matches but the tree is DIRTY
#
# It prints one row per call, phase-labelled, so the evidence packet carries the
# pin at entry AND at snapshot time rather than a single unrepeatable reading.
set -uo pipefail

WT="${1:-}"; EXPECT="${2:-}"; PHASE="${3:-}"
if [ -z "$WT" ] || [ -z "$EXPECT" ] || [ -z "$PHASE" ]; then
  echo "### HEAD PIN REFUSED (usage): gate_head_pin.sh <worktree> <expect-short-head> <phase-label>" >&2
  exit 4
fi
[ -d "$WT" ] || { echo "### HEAD PIN REFUSED ($PHASE): no such worktree $WT" >&2; exit 4; }
GOT=$(git -C "$WT" rev-parse --short HEAD 2>/dev/null) || {
  echo "### HEAD PIN REFUSED ($PHASE): $WT is not a git worktree" >&2; exit 4; }

# The comparison is on the SHORT form the caller stated, in both directions, so
# a caller that states a full sha and a repo that abbreviates to seven still
# reconcile -- but nothing shorter than what git itself prints is accepted.
LONG=$(git -C "$WT" rev-parse HEAD 2>/dev/null)
case "$LONG" in
  "$EXPECT"*) ok=1 ;;
  *) ok=0 ;;
esac
[ "$GOT" = "$EXPECT" ] && ok=1
if [ "$ok" -ne 1 ]; then
  echo "### HEAD PIN REFUSED ($PHASE): $WT is at $GOT ($LONG), the gate was pinned to $EXPECT" >&2
  exit 12
fi

DIRTY=$(git -C "$WT" status --porcelain | wc -l)
if [ "$DIRTY" -ne 0 ]; then
  echo "### HEAD PIN REFUSED ($PHASE): $WT is at $GOT as pinned but DIRTY ($DIRTY paths) -- law 11" >&2
  git -C "$WT" status --porcelain | head -20 >&2
  exit 13
fi
echo "### HEAD PIN ok ($PHASE): $WT HEAD=$GOT ($LONG) dirty=0 pinned=$EXPECT"
exit 0
