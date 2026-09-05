#!/usr/bin/env bash
# Land ONE merge-queue step: ff integration/4.12 to a candidate whose GATE and
# whose canonical RELOCK are both green, binsnap it through the ancestry guard,
# extend the declared fix set with the branch tip that landed, and push to
# `private` only.
#
# Nothing here decides anything: the caller passes the two job ids and this
# script REFUSES unless sacct says both COMPLETED 0:0. `origin` is never a
# push target on this campaign; the guard below checks the remote by name.
#
#   usage: CAND=<sha> GATE_JOB=<id> RELOCK_JOB=<id> FIXSET_ADD="<sha> <name>" \
#          bash land.sh
set -uo pipefail
export PATH=/users/glvov/.pixi/bin:/users/glvov/.local/bin:$PATH   # git-lfs, or push dies
T=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11
REPO=/oscar/data/stellex/glvov/retread-src
: "${CAND:?}"; : "${GATE_JOB:?}"; : "${RELOCK_JOB:?}"; : "${FIXSET_ADD:?}"

for j in "$GATE_JOB" "$RELOCK_JOB"; do
  row=$(sacct -j "$j" -X -n -o State,ExitCode --parsable2)
  echo "### sacct $j: $row"
  case "$row" in COMPLETED\|0:0) ;; *) echo "### REFUSE: job $j is not COMPLETED 0:0"; exit 2;; esac
done

FULL=$(git -C "$REPO" rev-parse --verify "${CAND}^{commit}") || exit 3
OLD=$(git -C "$REPO" rev-parse --verify integration/4.12) || exit 3
git -C "$REPO" merge-base --is-ancestor "$OLD" "$FULL" || {
  echo "### REFUSE: $CAND is not a fast-forward of integration/4.12 ($OLD)"; exit 4; }

SHORT=$(git -C "$REPO" rev-parse --short "$FULL")
SNAPC=$T/binsnaps/cand-$SHORT
[ -f "$SNAPC/pixi-build-retread" ] || { echo "### REFUSE: no candidate binsnap at $SNAPC"; exit 5; }

# the fix set the NEXT candidate must carry now includes what this step landed
grep -qE "^${FIXSET_ADD%% *} " "$T/tools/binsnap_fixset.txt" || \
  printf '%s\n' "$FIXSET_ADD" >> "$T/tools/binsnap_fixset.txt"

# ff, then re-run the guard against the LANDED tip with the extended fix set
# `integration/4.12` is CHECKED OUT in its own worktree, so `branch -f` refuses
# (and would be wrong -- it would leave that worktree's index describing a tree
# nobody is on). The branch is advanced by a FAST-FORWARD-ONLY merge inside that
# worktree, which is a refusal if anything is dirty (law 11) or if the candidate
# is not a descendant.
IWT=/oscar/data/stellex/glvov/retread-integration-4-12
[ "$(git -C "$IWT" rev-parse --abbrev-ref HEAD)" = "integration/4.12" ] || {
  echo "### REFUSE: $IWT is not on integration/4.12"; exit 6; }
D0=$(git -C "$IWT" status --porcelain | wc -l)
[ "$D0" -eq 0 ] || { echo "### REFUSE: the integration worktree is dirty ($D0 paths) -- law 11"; exit 6; }
git -C "$IWT" merge --ff-only "$FULL" || exit 6
bash "$T/tools/binsnap_ancestry_guard.sh" "$REPO" integration/4.12 || {
  echo "### REFUSE: the landed tip does not carry the declared fix set"; exit 7; }

SNAPI=$T/binsnaps/integration-$SHORT
mkdir -p "$SNAPI"
cp -f "$SNAPC/pixi-build-retread" "$SNAPI/pixi-build-retread" || exit 8
git -C "$REPO" rev-parse HEAD >/dev/null
echo "$FULL" > "$SNAPI/COMMIT"
sha256sum "$SNAPI/pixi-build-retread" | awk '{print $1}' > "$SNAPI/SHA256"
# direct file arguments only -- a piped compare is a known false-mismatch source here
cmp "$SNAPC/pixi-build-retread" "$SNAPI/pixi-build-retread" || { echo "### REFUSE: binsnap copy differs"; exit 9; }
echo "### BINSNAP $SNAPI sha256=$(cat "$SNAPI/SHA256") (byte-identical to $SNAPC)"

git -C "$REPO" push private "$FULL:refs/heads/integration/4.12" 2>&1 | tail -5
prc=${PIPESTATUS[0]}
echo "### push rc=$prc"
[ "$prc" -eq 0 ] || exit 10
echo "### private now at: $(git -C "$REPO" ls-remote private refs/heads/integration/4.12)"
echo "### origin still carries none of our refs:"
git -C "$REPO" ls-remote origin 'refs/heads/integration/*' 'refs/heads/fix/*' 2>&1 | head -3
echo "### LANDED integration/4.12 $OLD -> $FULL"
