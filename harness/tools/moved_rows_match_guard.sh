#!/usr/bin/env bash
# Guard for tools/moved_rows_match.sh (MERGE-M-3).  Doctrine law 3.
#
# Arm A is the defect itself, as a fixture: a drift block with TWO environments
# whose NAMES contain "isaac" (`pm-isaaclab` and `isaaclab-gpu-latest`) and NOT
# ONE isaac package moved.  The reader must print 0.  Arm Z runs the OLD
# one-liner over the same fixture and REQUIRES it to print 2, so the arm proves
# the finding rather than asserting it.
#
#   usage: bash moved_rows_match_guard.sh    rc 0 all arms pass, rc 1 otherwise
set -uo pipefail
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
R="$SELF_DIR/moved_rows_match.sh"
[ -f "$R" ] || { echo "GUARD FATAL: no reader at $R"; exit 1; }
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
pass=0; fail=0
ok()  { echo "PASS  $*"; pass=$((pass+1)); }
bad() { echo "FAIL  $*"; fail=$((fail+1)); }
chk() { if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1: got [$2] want [$3]"; fi; }
count() { echo "$1" | sed -n 's/^### MOVED ROWS MATCHING .*: \([0-9]*\) of .*/\1/p'; }
total() { echo "$1" | sed -n 's/^### MOVED ROWS MATCHING .*: [0-9]* of \([0-9]*\) moved rows$/\1/p'; }

# The fixture is byte-shaped like env_version_delta.py's own output:
#   "  %-24s moved=%d  (baseline pkgs=%d new pkgs=%d)"
#   "      %-38s %-18s -> %s"
mkfix() { printf '%s\n' "$@"; }

# ---- arm A: MERGE-M-3's OWN CASE -- an EMPTY moved list, two isaac-named envs
cat > "$WORK/a.txt" <<'EOF'
### baseline=x
### new     =y
### envs: baseline=27 new=27

### FULL PER-ENV VERSION DRIFT
  default                  moved=0  (baseline pkgs=266 new pkgs=266)
  isaaclab-gpu-latest      moved=0  (baseline pkgs=812 new pkgs=812)
  pm-isaaclab              moved=0  (baseline pkgs=410 new pkgs=410)

### total moved rows across all envs: 0
EOF
out=$(bash "$R" 'isaac' "$WORK/a.txt"); rc=$?
chk "A rc 0" "$rc" "0"
chk "A an empty moved list with two isaac-NAMED envs counts 0" "$(count "$out")" "0"
chk "A and it says there were 0 moved rows to match against" "$(total "$out")" "0"
chk "A no row is echoed" "$(echo "$out" | grep -c ' -> ')" "0"

# ---- arm Z: THE OLD ONE-LINER, RUN.  It must print 2 on the same fixture --
# the number every merge lane since B15 read as "isaacsim rows moved".
oldcount=$(sed -n '/FULL PER-ENV VERSION DRIFT/,$p' "$WORK/a.txt" | grep -icE 'isaac')
if [ "$oldcount" = "2" ]; then
  ok "Z the pre-fix one-liner counts 2 on a moved list of 0 -- MERGE-M-3 reproduced, not asserted"
else
  bad "Z the pre-fix one-liner did NOT reproduce MERGE-M-3: it printed [$oldcount], expected 2"
fi

# ---- arm B: real isaac PACKAGE rows are counted, and only those
cat > "$WORK/b.txt" <<'EOF'
### FULL PER-ENV VERSION DRIFT
  isaaclab-gpu-latest      moved=3  (baseline pkgs=812 new pkgs=812)
      isaacsim-kernel                        4.5.0.0            -> 4.5.1.0
      isaaclab                               0.40.9             -> 0.40.10
      anyio                                  4.15.0             -> 4.15.1
  pace                     moved=1  (baseline pkgs=292 new pkgs=292)
      numpy                                  1.26.4             -> 1.26.5
### total moved rows across all envs: 4
EOF
out=$(bash "$R" 'isaac' "$WORK/b.txt")
chk "B two isaac PACKAGE rows out of four moved rows" "$(count "$out")" "2"
chk "B the total is the moved rows, not the block lines" "$(total "$out")" "4"
chk "B the matching rows are echoed for the reader" "$(echo "$out" | grep -c 'isaacsim-kernel')" "1"
chk "B a non-isaac package in an isaac env is not counted" "$(echo "$out" | grep -c 'anyio')" "0"

# ---- arm C: the match is the PACKAGE field, never the version
cat > "$WORK/c.txt" <<'EOF'
### FULL PER-ENV VERSION DRIFT
  pace                     moved=1  (baseline pkgs=292 new pkgs=292)
      somepkg                                1.0+isaac          -> 1.1+isaac
EOF
out=$(bash "$R" 'isaac' "$WORK/c.txt")
chk "C a version string containing the pattern is not a match" "$(count "$out")" "0"

# ---- arm D: TRUNCATION is reported, so a floor is never read as a total
cat > "$WORK/d.txt" <<'EOF'
### FULL PER-ENV VERSION DRIFT
  robogen                  moved=88  (baseline pkgs=256 new pkgs=256)
      isaacsim-kernel                        4.5.0.0            -> 4.5.1.0
      ... 87 more
EOF
out=$(bash "$R" 'isaac' "$WORK/d.txt")
chk "D the visible match is counted" "$(count "$out")" "1"
chk "D the truncation warning names the hidden rows" \
  "$(echo "$out" | grep -c 'hiding 87 further moved rows')" "1"
chk "D the '... N more' line is not itself counted as a moved row" "$(total "$out")" "1"

# ---- arm E: no drift block at all is rc 2, a setup failure and never a count
printf 'nothing here\n' > "$WORK/e.txt"
bash "$R" 'isaac' "$WORK/e.txt" >/dev/null 2>&1
chk "E rc 2 when there is no drift block" "$?" "2"
bash "$R" 'isaac' "$WORK/nope.txt" >/dev/null 2>&1
chk "E rc 2 when the file does not exist" "$?" "2"

# ---- arm F: stdin is the same reader
out=$(bash "$R" 'isaac' < "$WORK/b.txt")
chk "F stdin gives the same answer as the file argument" "$(count "$out")" "2"

# ---- arm G: THE MUTATION.  Widen the row test back to the whole block and the
# env names come back -- arm A's 0 becomes the old 2.
MUT="$WORK/mutant.sh"
sed -e 's@^    /\^      / \&\& / -> / {@    /./ {@' \
    -e 's@tolower(pkg) ~ tolower(pat)@tolower($0) ~ tolower(pat)@' "$R" > "$MUT"
if cmp -s "$R" "$MUT"; then
  bad "G the mutation changed nothing -- this arm is vacuous"
else
  mout=$(bash "$MUT" 'isaac' "$WORK/a.txt" 2>&1)
  if [ "$(count "$mout")" = "2" ]; then
    ok "G a mutant that treats every line as a moved row prints the old 2 on an empty list -- the defect, reproduced and caught"
  else
    bad "G the mutant did not reproduce the defect: it printed [$(count "$mout")]"
  fi
fi

echo "### MOVED ROWS MATCH GUARD $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
