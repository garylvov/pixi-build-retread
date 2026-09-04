#!/usr/bin/env bash
# wheel_store_seed_guard.sh -- the reader for `retread_seed_wheel_store`.
#
# It BUILDS a fixture wheel store, RUNS the real function on it, and asserts the
# four things the function promises: no fill lock reaches the destination, no
# quarantine dir reaches it, every destination wheel is its OWN inode with link
# count 1, and the source store is byte- and inode-identical afterwards.
#
# Then it runs a NEGATIVE CONTROL: it seeds a second destination with `cp -al`
# and re-runs the function over that destination. rsync's quick check skips
# every entry (same size, same mtime, -a preserved both), so the hardlinks
# survive, and the function MUST refuse with a link-count complaint. Without
# this half the link-count assert could be vacuous -- a check that cannot fail
# is a defect.
#
# Usage: wheel_store_seed_guard.sh          (self-contained, needs only a $TMPDIR)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FAST_ENV=$HERE/retread_fast_env.sh
[ -f "$FAST_ENV" ] || { echo "GUARD FATAL: no retread_fast_env.sh next to me at $FAST_ENV"; exit 2; }
# shellcheck source=/dev/null
. "$FAST_ENV"
command -v retread_seed_wheel_store >/dev/null \
  || { echo "GUARD FATAL: retread_seed_wheel_store is not defined by $FAST_ENV"; exit 2; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/wheel-store-seed-guard.XXXXXX") || exit 2
trap 'rm -rf "$WORK"' EXIT
echo "GUARD: work dir $WORK"

FAIL=0
fail () { echo "GUARD FAIL: $*"; FAIL=1; }
ok   () { echo "GUARD  ok : $*"; }

########## the fixture store ###################################################
SRC=$WORK/persist-store
mkdir -p "$SRC/aaa111" "$SRC/bbb222" "$SRC/ccc333.quarantine-5762227-17"
printf 'wheel-one-bytes\n'  > "$SRC/aaa111/pkg_one-1.0-py3-none-any.whl"
printf 'record-one\n'       > "$SRC/aaa111/record.json"
printf 'wheel-two-bytes\n'  > "$SRC/bbb222/pkg_two-2.0-py3-none-any.whl"
: >                           "$SRC/bbb222/.pkg_two-2.0-py3-none-any.whl.retread-fill-v1.lock"
printf 'poison\n'           > "$SRC/ccc333.quarantine-5762227-17/pkg_three-3.0-py3-none-any.whl"
echo "GUARD: fixture source:"
find "$SRC" -mindepth 1 | sort | sed 's/^/GUARD:   /'

SRC_BEFORE=$(find "$SRC" -printf '%P %y %s %n %m\n' 2>/dev/null | sort)
SRC_INO_ONE=$(stat -c %i "$SRC/aaa111/pkg_one-1.0-py3-none-any.whl")
SRC_INO_TWO=$(stat -c %i "$SRC/bbb222/pkg_two-2.0-py3-none-any.whl")

########## POSITIVE: the real seed #############################################
echo "GUARD: === POSITIVE: retread_seed_wheel_store (rsync -aW byte copy) ==="
DST=$WORK/jobscoped-store
OUT=$WORK/positive.out
retread_seed_wheel_store "$SRC" "$DST" > "$OUT" 2>&1
RC=$?
sed 's/^/GUARD:   /' "$OUT"

[ "$RC" -eq 0 ] || fail "positive seed returned rc=$RC, expected 0"
[ "$RC" -eq 0 ] && ok "positive seed rc=0"

grep -q '^### wheel_store seeded src=.* dst=.* entries=.* wheels=.* fill_locks=.* hardlinks=.* quarantines=.* rc=.* wall=.*s$' "$OUT" \
  && ok "census line printed in the declared shape" \
  || fail "no census line of the declared shape in the positive output"

[ "$(grep -c '^### wheel_store seeded ' "$OUT")" -eq 1 ] \
  && ok "exactly ONE census line" \
  || fail "expected exactly one census line, got $(grep -c '^### wheel_store seeded ' "$OUT")"

N_LOCK=$(find "$DST" -mindepth 2 -maxdepth 2 -name '.*.retread-fill-v1.lock' 2>/dev/null | wc -l)
[ "$N_LOCK" -eq 0 ] && ok "no fill-lock sidecar in the destination" \
                    || fail "$N_LOCK fill-lock sidecar(s) reached the destination"

N_QUAR=$(find "$DST" -mindepth 1 -maxdepth 2 -name '*.quarantine-*' 2>/dev/null | wc -l)
[ "$N_QUAR" -eq 0 ] && ok "no quarantine dir in the destination" \
                    || fail "$N_QUAR quarantine entr(y/ies) reached the destination"

N_WHL=$(find "$DST" -mindepth 2 -maxdepth 2 -name '*.whl' 2>/dev/null | wc -l)
[ "$N_WHL" -eq 2 ] && ok "both non-quarantined wheels arrived (n=$N_WHL)" \
                   || fail "expected 2 wheels in the destination, found $N_WHL"

N_MULTI=$(find "$DST" -mindepth 2 -maxdepth 2 -name '*.whl' -printf '%n\n' 2>/dev/null | { grep -vc '^1$' || true; })
[ "${N_MULTI:-0}" -eq 0 ] && ok "every destination wheel has link count 1" \
                          || fail "$N_MULTI destination wheel(s) have link count != 1"

DST_INO_ONE=$(stat -c %i "$DST/aaa111/pkg_one-1.0-py3-none-any.whl" 2>/dev/null)
DST_INO_TWO=$(stat -c %i "$DST/bbb222/pkg_two-2.0-py3-none-any.whl" 2>/dev/null)
[ -n "$DST_INO_ONE" ] && [ "$DST_INO_ONE" != "$SRC_INO_ONE" ] \
  && ok "wheel one is a DIFFERENT inode (src=$SRC_INO_ONE dst=$DST_INO_ONE)" \
  || fail "wheel one shares the source inode ($SRC_INO_ONE)"
[ -n "$DST_INO_TWO" ] && [ "$DST_INO_TWO" != "$SRC_INO_TWO" ] \
  && ok "wheel two is a DIFFERENT inode (src=$SRC_INO_TWO dst=$DST_INO_TWO)" \
  || fail "wheel two shares the source inode ($SRC_INO_TWO)"

cmp -s "$SRC/aaa111/pkg_one-1.0-py3-none-any.whl" "$DST/aaa111/pkg_one-1.0-py3-none-any.whl" \
  && ok "wheel one copied byte-for-byte" || fail "wheel one bytes differ after the seed"

[ -f "$DST/aaa111/record.json" ] && ok "record sidecar carried across (warmth, not just wheels)" \
                                 || fail "record.json did not reach the destination"

SRC_AFTER=$(find "$SRC" -printf '%P %y %s %n %m\n' 2>/dev/null | sort)
[ "$SRC_BEFORE" = "$SRC_AFTER" ] && ok "SOURCE STORE UNMODIFIED (same entries, sizes, link counts, modes)" \
                                 || { fail "the source store changed across the seed"; diff <(echo "$SRC_BEFORE") <(echo "$SRC_AFTER") | sed 's/^/GUARD:   /'; }
[ "$(stat -c %i "$SRC/aaa111/pkg_one-1.0-py3-none-any.whl")" = "$SRC_INO_ONE" ] \
  && ok "source wheel one kept its inode" || fail "source wheel one was replaced"

########## NEGATIVE CONTROL: cp -al, which the check MUST catch #################
echo "GUARD: === NEGATIVE CONTROL: destination seeded with cp -al ==="
BAD=$WORK/cp-al-store
cp -al "$SRC" "$BAD" || { echo "GUARD FATAL: cp -al of the fixture failed (no hardlink support here?)"; exit 2; }
BAD_N=$(find "$BAD" -mindepth 2 -maxdepth 2 -name '*.whl' -printf '%n\n' 2>/dev/null | { grep -vc '^1$' || true; })
echo "GUARD:   cp -al produced $BAD_N wheel(s) with link count != 1 (the precondition for this control)"
[ "${BAD_N:-0}" -gt 0 ] || fail "cp -al did not actually hardlink -- the negative control proves nothing"

NOUT=$WORK/negative.out
retread_seed_wheel_store "$SRC" "$BAD" > "$NOUT" 2>&1
NRC=$?
sed 's/^/GUARD:   /' "$NOUT"

[ "$NRC" -ne 0 ] && ok "negative control REFUSED (rc=$NRC) -- the link-count check is not vacuous" \
                 || fail "negative control returned 0: the link-count check DID NOT FIRE on a cp -al store"
grep -q 'link count != 1' "$NOUT" \
  && ok "negative control refused for the RIGHT reason (link count)" \
  || fail "negative control refusal did not name the link count"
grep -q '^### wheel_store seeded .*hardlinks=[1-9]' "$NOUT" \
  && ok "census line reported hardlinks>0 on the cp -al store" \
  || fail "census line did not report a non-zero hardlink count on the cp -al store"

echo
if [ "$FAIL" -eq 0 ]; then echo "GUARD: PASS"; exit 0; else echo "GUARD: FAIL"; exit 1; fi
