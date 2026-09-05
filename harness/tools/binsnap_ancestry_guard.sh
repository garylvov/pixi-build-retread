#!/usr/bin/env bash
# p6r / boarded p6q-1 — refuse a binsnap of a branch that is missing a fix a
# sibling lane already merged.
#
# `627de7f` (p6n-b) was gated, binsnapped and ARMED TWICE while silently
# regressing p6j: it forked before p6j landed, so the p6j projection was simply
# absent and the emitted `pillow` row went back to carrying every bundled
# wheel's exact Requires-Dist. Nothing in the build or the arm harness noticed.
# `git merge-base --is-ancestor` over the declared fix set does notice, in
# milliseconds, before a 6-22 minute release build and a 3-hour arm.
#
#   usage: binsnap_ancestry_guard.sh <repo-or-worktree> <commit-ish> [fixset-file]
#
# The fix set is one `<sha> <name>` per line (`#` comments ignored). It
# defaults to tools/binsnap_fixset.txt beside this script. Each entry must be
# an ANCESTOR of the candidate; a missing one is a refusal, rc=2.
#
# The integration tip is reported, never required: a candidate lane forks from
# the tip's PARENTS, so the tip's own merge commit is legitimately absent from
# every branch built on it. What must be present is the FIXES the tip carries.
set -uo pipefail

REPO="${1:?usage: binsnap_ancestry_guard.sh <repo> <commit-ish> [fixset-file]}"
CAND="${2:?usage: binsnap_ancestry_guard.sh <repo> <commit-ish> [fixset-file]}"
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FIXSET="${3:-$SELF_DIR/binsnap_fixset.txt}"

[ -d "$REPO" ] || { echo "ANCESTRY FATAL: no such repo $REPO" >&2; exit 3; }
[ -f "$FIXSET" ] || { echo "ANCESTRY FATAL: no fix set at $FIXSET" >&2; exit 3; }

CAND_SHA="$(git -C "$REPO" rev-parse --verify "${CAND}^{commit}" 2>/dev/null)" || {
  echo "ANCESTRY FATAL: $CAND is not a commit in $REPO" >&2; exit 3; }

missing=0
checked=0
while read -r sha name rest; do
  case "$sha" in ''|\#*) continue;; esac
  checked=$((checked + 1))
  if ! git -C "$REPO" rev-parse --verify --quiet "${sha}^{commit}" >/dev/null; then
    echo "ANCESTRY FATAL: declared fix $sha ($name) is not a commit in $REPO" >&2
    exit 3
  fi
  if git -C "$REPO" merge-base --is-ancestor "$sha" "$CAND_SHA"; then
    echo "ANCESTRY ok      $sha $name"
    continue
  fi

  # NOT an ancestor. That is not yet a refusal: a fix reaches a branch as a
  # CHERRY-PICK just as legitimately as it reaches it by merge, and this
  # campaign does both. C13 `5584ce6` carries C12 as `e0ae761`; C15 and C16 are
  # built on C13, so an ancestor-only guard refuses every descendant of C13 for
  # a fix all of them demonstrably contain. Refusing a branch that HAS the fix
  # is the same defect as accepting one that does not.
  #
  # `git cherry <upstream> <head> <limit>` answers exactly this question: it
  # prints `- <sha>` when an equivalent patch (same `git patch-id`) is already
  # in <upstream>, and `+ <sha>` when it is not. Restricting it to the single
  # commit (`<fix>` with limit `<fix>^`) keeps it O(1) walks, not O(history).
  #
  # A MERGE commit has no patch-id, so equivalence cannot be decided for one;
  # such an entry stays ancestor-only and says so rather than passing quietly.
  parents=$(git -C "$REPO" rev-list --parents -n 1 "$sha" | wc -w)
  if [ "$parents" -gt 2 ]; then
    echo "ANCESTRY MISSING $sha $name ${rest} (merge commit: no patch-id, ancestor-only)" >&2
    missing=$((missing + 1))
    continue
  fi
  eq=$(git -C "$REPO" cherry "$CAND_SHA" "$sha" "${sha}^" 2>/dev/null | head -1)
  case "$eq" in
    -*) echo "ANCESTRY ok      $sha $name (as an equivalent patch: git cherry says the candidate already carries it)";;
    *)  echo "ANCESTRY MISSING $sha $name ${rest}" >&2
        missing=$((missing + 1));;
  esac
done < "$FIXSET"

if [ "$checked" -eq 0 ]; then
  echo "ANCESTRY FATAL: fix set $FIXSET declares nothing -- a guard that cannot fail" >&2
  exit 3
fi

if [ "$missing" -gt 0 ]; then
  echo "ANCESTRY REFUSED candidate=$CAND_SHA missing=$missing of $checked declared fixes" >&2
  echo "  merge the missing fix (or rebase this branch onto the integration tip) before snapshotting." >&2
  exit 2
fi

echo "ANCESTRY ACCEPTED candidate=$CAND_SHA fixes=$checked"
exit 0
