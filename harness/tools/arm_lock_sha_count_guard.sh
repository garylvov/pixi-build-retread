#!/usr/bin/env bash
# Guard for tools/arm_lock_sha_count.sh (L3-1b-7).  Doctrine law 3.
#
# Arm A is `l31b-proof` 5919981's own rows, to the sha: three arms, all
# `shims=0`, arms 1 and 2 byte-identical and arm 3 different by design.  The
# reader must print 1.  Arm Z runs the PRE-FIX one-liner over the SAME rows and
# requires it to print 2 -- the number the proof log actually carried -- so the
# finding is reproduced, not asserted.
#
#   usage: bash arm_lock_sha_count_guard.sh   rc 0 all arms pass, rc 1 otherwise
set -uo pipefail
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
R="$SELF_DIR/arm_lock_sha_count.sh"
[ -f "$R" ] || { echo "GUARD FATAL: no reader at $R"; exit 1; }
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
pass=0; fail=0
ok()  { echo "PASS  $*"; pass=$((pass+1)); }
bad() { echo "FAIL  $*"; fail=$((fail+1)); }
chk() { if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1: got [$2] want [$3]"; fi; }
n_of() { echo "$1" | sed -n 's/^  distinct lock shas (arms [^)]*): \([0-9]*\) .*/\1/p'; }
label_of() { echo "$1" | sed -n 's/^  distinct lock shas (arms \([^)]*\)): .*/\1/p'; }

S12=b7dba246055737796ef6c3b123984a269aae0237f547c6d52acfdb212142c759
S3=37c6148cadf0413c981463418457c2c8198e43b402117e03fbed48c771b4dd7a

# ---- arm A: l31b-proof 5919981's shape.  Two identical shas -> 1.
cat > "$WORK/a.rows" <<EOF
arm=1 label=cold-run1 cache=cold shims=0 lock_rc=0 lock_wall=1966s lock_sha=$S12 lock_bytes=2732160
arm=2 label=warm-run2 cache=adopt1 shims=0 lock_rc=0 lock_wall=1835s lock_sha=$S12 lock_bytes=2732160
arm=3 label=cold-ctl cache=cold shims=0 lock_rc=0 lock_wall=2560s lock_sha=$S3 lock_bytes=2732313
EOF
out=$(bash "$R" "$WORK/a.rows" 1 2); rc=$?
chk "A rc 0" "$rc" "0"
chk "A two identical shas over the two named arms count 1" "$(n_of "$out")" "1"
chk "A the label is derived from the arms actually counted" "$(label_of "$out")" "1,2"
chk "A the control arm is not in the set" "$(echo "$out" | grep -c "$S3")" "0"
chk "A both counted arms are named with their sha" "$(echo "$out" | grep -c "sha=$S12")" "2"

# ---- arm Z: THE PRE-FIX ONE-LINER, RUN OVER THE SAME ROWS.  It filters on
# `shims=="0" && lock_rc=="0"`, which is all three arms, so it prints 2 -- the
# number l31b-proof 5919981 printed directly beneath two identical shas.
oldn=$(awk '{delete kv; for(i=1;i<=NF;i++){split($i,p,"=");kv[p[1]]=p[2]} if(kv["shims"]=="0"&&kv["lock_rc"]=="0") print kv["lock_sha"]}' "$WORK/a.rows" | sort -u | wc -l)
if [ "$oldn" -eq 2 ]; then
  ok "Z the pre-fix counter prints 2 over two identical shas -- L3-1b-7 reproduced on the proof's own rows"
else
  bad "Z the pre-fix counter did NOT reproduce L3-1b-7: it printed [$oldn], expected 2"
fi

# ---- arm B: two DIFFERENT shas over the two named arms count 2
cat > "$WORK/b.rows" <<EOF
arm=1 label=cold-run1 cache=cold shims=0 lock_rc=0 lock_sha=0110ce46bfcced3f42ecf40d4ec4fa3c lock_bytes=2732160
arm=2 label=warm-run2 cache=adopt1 shims=0 lock_rc=0 lock_sha=0e2036cca3f75f0a9ba87b242111878b lock_bytes=2732313
arm=3 label=cold-ctl cache=cold shims=0 lock_rc=0 lock_sha=a3e591a3aaaaaaaaaaaaaaaaaaaaaaaa lock_bytes=2732313
EOF
out=$(bash "$R" "$WORK/b.rows" 1 2)
chk "B two different shas count 2" "$(n_of "$out")" "2"
chk "B and it is still 2 arms, not 3 -- l31b1-proof printed 3 here" \
  "$(echo "$out" | grep -c '^  arm=')" "2"

# ---- arm C: the set is whatever is named, including the control
out=$(bash "$R" "$WORK/a.rows" 1 2 3)
chk "C naming three arms counts three arms" "$(n_of "$out")" "2"
chk "C and says so in the label" "$(label_of "$out")" "1,2,3"

# ---- arm D: a missing arm is REFUSED, never silently dropped -- which is the
# whole class of defect this file replaces.
err=$(bash "$R" "$WORK/a.rows" 1 2 9 2>&1 >/dev/null); rc=$?
chk "D naming an absent arm is rc 2" "$rc" "2"
chk "D and the message names it" "$(echo "$err" | grep -c 'arm 9 is not in the rows file')" "1"

# ---- arm E: an arm that did not lock has no sha to count and is refused
cat > "$WORK/e.rows" <<EOF
arm=1 label=cold-run1 shims=0 lock_rc=0 lock_sha=$S12 lock_bytes=2732160
arm=2 label=warm-run2 shims=0 lock_rc=1 lock_sha= lock_bytes=0
EOF
err=$(bash "$R" "$WORK/e.rows" 1 2 2>&1 >/dev/null); rc=$?
chk "E an arm with lock_rc!=0 is rc 2" "$rc" "2"
chk "E and the message names the rc" "$(echo "$err" | grep -c 'arm 2 has lock_rc=1')" "1"

# ---- arm F: a duplicated arm row is refused (a re-run appended, not replaced)
cat "$WORK/a.rows" > "$WORK/f.rows"; head -1 "$WORK/a.rows" >> "$WORK/f.rows"
err=$(bash "$R" "$WORK/f.rows" 1 2 2>&1 >/dev/null); rc=$?
chk "F a duplicated arm row is rc 2" "$rc" "2"
chk "F and the message counts the duplicates" "$(echo "$err" | grep -c 'arm 1 appears 2 times')" "1"

# ---- arm G: an unreadable rows file and a missing arm list are setup failures
bash "$R" "$WORK/nope.rows" 1 2 >/dev/null 2>&1
chk "G an unreadable rows file is rc 2" "$?" "2"
bash "$R" "$WORK/a.rows" >/dev/null 2>&1
chk "G naming no arm at all is rc 2" "$?" "2"

# ---- arm H: THE MUTATION.  Put the `shims` filter back and the counter stops
# counting the arms it names -- arm A's 1 becomes the reported 2.
MUT="$WORK/mutant.sh"
sed 's@      if (!(a in seen))    {@      if (0)               {@; s@^    for (i = 1; i <= nw; i++) {$@    nw = 0; for (a in seen) if (rc[a] == "0") want[++nw] = a\n    for (i = 1; i <= nw; i++) {@' "$R" > "$MUT"
if cmp -s "$R" "$MUT"; then
  bad "H the mutation changed nothing -- this arm is vacuous"
else
  mout=$(bash "$MUT" "$WORK/a.rows" 1 2 2>&1)
  if [ "$(n_of "$mout")" = "2" ]; then
    ok "H a mutant that counts every locking arm instead of the named ones prints 2 over two identical shas -- L3-1b-7, reproduced and caught"
  else
    bad "H the mutant did not reproduce the defect: it printed [$(n_of "$mout")]"
  fi
fi

echo "### ARM LOCK SHA COUNT GUARD $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
