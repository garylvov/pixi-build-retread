#!/usr/bin/env bash
# Guard for the guard. It must REFUSE 627de7f (the branch that was armed twice
# without p6j) and ACCEPT 2f74345 (the same lineage with p6j merged in).
set -uo pipefail
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
G="$SELF_DIR/binsnap_ancestry_guard.sh"
REPO="${1:-/oscar/data/stellex/glvov/agrescap/worktrees/fix-p6q-pillow-origin}"
rc_all=0

echo "--- must REFUSE 627de7f (p6j missing) ---"
"$G" "$REPO" 627de7f; rc=$?
if [ "$rc" -eq 2 ]; then echo "PASS refused 627de7f"; else echo "FAIL 627de7f rc=$rc (expected 2)"; rc_all=1; fi

echo "--- must ACCEPT 2f74345 (p6j merged) ---"
# Against ITS OWN fix set, not today's. The default set grows every time the
# merge queue lands a step, so a branch cut before those steps legitimately
# fails it -- which is the guard working, not a regression. This case exists to
# assert the GUARD's behaviour, so it declares the four fixes that were the set
# when 2f74345 was cut.
OWN="$(mktemp)"
printf '4b3103b p6i shared-cache-atomicity\n8e71eda p6k closure-resolve-only\n5562f7c p6j constrains-origin-coverage\nf0b7bee p6k-b wheel-fingerprint-ctime\n' > "$OWN"
"$G" "$REPO" 2f74345 "$OWN"; rc=$?
rm -f "$OWN"
if [ "$rc" -eq 0 ]; then echo "PASS accepted 2f74345"; else echo "FAIL 2f74345 rc=$rc (expected 0)"; rc_all=1; fi

echo "--- non-vacuity: an empty fix set is itself a refusal ---"
EMPTY="$(mktemp)"; printf '# nothing declared\n' > "$EMPTY"
"$G" "$REPO" 2f74345 "$EMPTY"; rc=$?
rm -f "$EMPTY"
if [ "$rc" -eq 3 ]; then echo "PASS empty fix set refused"; else echo "FAIL empty fix set rc=$rc (expected 3)"; rc_all=1; fi

echo "--- a CHERRY-PICKED fix satisfies the set: C15 c1f26a8 carries C12 5f71a14 as e0ae761 ---"
# Ancestor-only, this REFUSED every descendant of C13 for a fix all of them
# demonstrably contain -- C13 5584ce6 carries C12 as the cherry-pick e0ae761,
# and C15/C16 are built on C13. Refusing a branch that HAS the fix is the same
# defect as accepting one that does not.
CP="$(mktemp)"; printf '5f71a14 c12-git-source-inject-spans\n' > "$CP"
"$G" "${REPO_MAIN:-/oscar/data/stellex/glvov/retread-src}" c1f26a8 "$CP"; rc=$?
if [ "$rc" -eq 0 ]; then echo "PASS accepted c1f26a8 through the cherry-pick"; else echo "FAIL c1f26a8 rc=$rc (expected 0)"; rc_all=1; fi

echo "--- and equivalence must NOT be a free pass: 627de7f still has no p6j, by patch-id either ---"
PJ="$(mktemp)"; printf '5562f7c p6j constrains-origin-coverage\n' > "$PJ"
"$G" "${REPO_MAIN:-/oscar/data/stellex/glvov/retread-src}" 627de7f "$PJ"; rc=$?
rm -f "$CP" "$PJ"
if [ "$rc" -eq 2 ]; then echo "PASS 627de7f still refused with equivalence on"; else echo "FAIL 627de7f rc=$rc (expected 2)"; rc_all=1; fi

echo "=== binsnap ancestry guard: $([ $rc_all -eq 0 ] && echo GREEN || echo RED) ==="
exit $rc_all
