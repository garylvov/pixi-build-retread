#!/usr/bin/env bash
# fast_env_resolution_guard.sh -- ONE rule for which retread_fast_env.sh a
# harness imports, and the reader that keeps it one.
#
# CLAUDE.md law 7 hazard (b): two copies of one module are the NORMAL state
# here, and PYTHONPATH -- or, for these shell harnesses, the order of the
# FAST_ENV candidates -- alone decides which a process runs. Until 2026-09-07
# four of the seven harnesses that source it named
# `$(dirname "$0")/../retread_fast_env.sh`, a path that has never existed in the
# harness repo, so they ALWAYS fell through to the task-dir copy while
# `instrumented/p6b_relock.sh` named the correct `../tools/` form and read the
# repo copy. Two halves of one campaign, importing different bytes, with nothing
# in any job log saying so.
#
# THE RULE, and all three parts are asserted below:
#   FAST_ENV=$(dirname "$0")/../tools/retread_fast_env.sh
#   [ -f "$FAST_ENV" ] || FAST_ENV=$T/tools/retread_fast_env.sh
#   ... and then the resolved path is PRINTED as `### FAST_ENV resolved=<path>`
#       and two candidates that DIFFER by md5 REFUSE.
#
# HALF ONE is static across every template that carries a FAST_ENV line. HALF
# TWO drives the real block, cut out of a real template, over four fixtures:
# both copies present and identical, both present and DIFFERENT, only the
# task-dir copy, and the dead legacy path present as a decoy.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=${HARNESS_REPO:-}
[ -n "$REPO" ] || REPO=$(cd -- "$HERE/../.." && pwd)
H=$REPO/harness
[ -d "$H/tools" ] || { echo "GUARD FATAL: no harness tree at $H"; exit 2; }
W=$(mktemp -d "${TMPDIR:-/tmp}/fastenvres.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
pass=0; fail=0
ok()  { echo "GUARD  ok : $*"; pass=$((pass+1)); }
bad() { echo "GUARD FAIL: $*"; fail=$((fail+1)); }

CANON='FAST_ENV=$(dirname "$0")/../tools/retread_fast_env.sh'
FALLBACK='[ -f "$FAST_ENV" ] || FAST_ENV=$T/tools/retread_fast_env.sh'

########## HALF ONE: every template resolves it the SAME way ###################
echo "GUARD: === HALF ONE: one rule, in every harness that sources it ==="
NTPL=0
for f in $(grep -rl '^FAST_ENV=\$(dirname\|^FAST_ENV=\$T/' --include='*.sh' "$H/arms" "$H/proof" "$H/instrumented" "$H/phase_template" 2>/dev/null | sort); do
  REL=${f#"$H"/}; NTPL=$((NTPL+1))
  FIRST=$(grep -m1 '^FAST_ENV=' "$f" | sed 's/[[:space:]]*#.*$//')
  SECOND=$(grep -m1 '^\[ -f "\$FAST_ENV" \] || FAST_ENV=' "$f")
  if [ "$FIRST" = "$CANON" ]; then ok "$REL: first candidate is the canonical tools/ path"
  else bad "$REL: first candidate is '$FIRST', not '$CANON' -- a second resolution order is a second copy waiting to be imported"; fi
  if [ "$SECOND" = "$FALLBACK" ]; then ok "$REL: falls back to the task-dir copy, verbatim"
  else bad "$REL: fallback line is '$SECOND', not '$FALLBACK'"; fi
  # The dead path is the specific defect this guard was written for. It must not
  # come back, in any file, in any form.
  if grep -q 'dirname "\$0")/\.\./retread_fast_env\.sh' "$f"; then
    bad "$REL: still names ../retread_fast_env.sh -- that path does not exist in this repo, so this harness can only ever take the fallback"
  else ok "$REL: does not name the dead ../retread_fast_env.sh path"; fi
  if grep -q '^echo "### FAST_ENV resolved=\$FAST_ENV"$' "$f"; then
    ok "$REL: PRINTS which copy it resolved"
  else bad "$REL: never prints '### FAST_ENV resolved=' -- the answer is unrecoverable from its job log"; fi
done
[ "$NTPL" -ge 5 ] && ok "HALF ONE read $NTPL harnesses (the guard is not green against nothing)" \
                  || bad "HALF ONE found only $NTPL harnesses with a FAST_ENV line -- expected at least 5"
[ -f "$H/tools/retread_fast_env.sh" ] && ok "the canonical copy really is at tools/retread_fast_env.sh" \
  || bad "tools/retread_fast_env.sh does not exist -- the rule names a path that is not there"
[ -f "$H/retread_fast_env.sh" ] && bad "harness/retread_fast_env.sh EXISTS -- the dead path is dead no longer, and the old first candidate would now win" \
  || ok "harness/retread_fast_env.sh does not exist, so the old first candidate was always a no-op"

########## HALF TWO: drive the REAL block ######################################
# Cut from a real template rather than retyped, so a change to the block that
# this guard does not see cannot exist.
echo "GUARD: === HALF TWO: the block itself, over four fixtures ==="
SRC=$H/arms/mh1_relock.sh
[ -f "$SRC" ] || SRC=$H/phase_template/phaseN_relock.sh
BLK=$W/block.sh
{ echo 'set -uo pipefail'
  echo 'T=$FIXT'
  grep -m1 '^FAST_ENV=' "$SRC"
  grep -m1 '^\[ -f "\$FAST_ENV" \] || FAST_ENV=' "$SRC"
  awk '/^FAST_ENV_ALT=/{p=1} p{print} /^echo "### FAST_ENV resolved=/{if(p)exit}' "$SRC"
} > "$BLK"
grep -q '^FAST_ENV_ALT=' "$BLK" && grep -q 'FAST_ENV resolved=' "$BLK" \
  && ok "the block was extracted from $(basename "$SRC") ($(wc -l < "$BLK") lines), not retyped" \
  || { bad "could not extract the resolution block from $SRC -- HALF TWO cannot run"; echo "GUARD: fast_env_resolution: pass=$pass fail=$fail"; exit 1; }

mkfix () {  # $1 tag, $2 repo-copy-content ('' = absent), $3 task-copy-content, $4 legacy-copy ('' = absent)
  local d=$W/$1
  mkdir -p "$d/harness/tools" "$d/task/tools"
  [ -n "$2" ] && printf '%s\n' "$2" > "$d/harness/tools/retread_fast_env.sh"
  printf '%s\n' "$3" > "$d/task/tools/retread_fast_env.sh"
  [ -n "$4" ] && printf '%s\n' "$4" > "$d/harness/retread_fast_env.sh"
  mkdir -p "$d/harness/arms"
  cp "$BLK" "$d/harness/arms/probe.sh"
  echo "$d"
}
runfix () { ( cd "$1/harness/arms" && FIXT=$1/task bash ./probe.sh ) > "$2" 2>&1; echo $?; }

# a1: both copies present and IDENTICAL -> the repo copy wins, and it says so.
D=$(mkfix a1 'X=1' 'X=1' ''); RC=$(runfix "$D" "$W/a1.log")
if [ "$RC" = 0 ] && grep -q "^### FAST_ENV resolved=\./\.\./tools/retread_fast_env\.sh$" "$W/a1.log"; then
  ok "a1: both copies present and identical -> resolves the REPO copy and prints it"
else bad "a1: rc=$RC"; sed 's/^/      /' "$W/a1.log"; fi

# a2: the two copies DIFFER -> refuse, naming both md5s. This is the whole point:
# path order must not get to pick a winner silently.
D=$(mkfix a2 'X=1' 'X=2' ''); RC=$(runfix "$D" "$W/a2.log")
if [ "$RC" != 0 ] && grep -q 'candidates DIFFER' "$W/a2.log" && grep -qc '[0-9a-f]\{32\}' "$W/a2.log"; then
  ok "a2: two DIFFERENT copies REFUSE (rc=$RC) and the refusal carries both md5s"
else bad "a2: rc=$RC -- two divergent copies must refuse, not choose"; sed 's/^/      /' "$W/a2.log"; fi

# a3: only the task-dir copy exists (the shape every merge lane actually runs in,
# where the derived script sits in a task dir with no harness/tools beside it).
D=$(mkfix a3 '' 'X=1' ''); RC=$(runfix "$D" "$W/a3.log")
if [ "$RC" = 0 ] && grep -q "^### FAST_ENV resolved=.*/task/tools/retread_fast_env.sh$" "$W/a3.log"; then
  ok "a3: no repo copy -> the task-dir fallback resolves, and the log names it"
else bad "a3: rc=$RC"; sed 's/^/      /' "$W/a3.log"; fi

# a4: THE DECOY. The dead legacy path is present and holds DIFFERENT bytes. The
# old first candidate would have imported it. The rule must not.
D=$(mkfix a4 'X=1' 'X=1' 'X=LEGACY'); RC=$(runfix "$D" "$W/a4.log")
if [ "$RC" = 0 ] && grep -q "^### FAST_ENV resolved=\./\.\./tools/retread_fast_env\.sh$" "$W/a4.log" \
   && ! grep -q "resolved=.*\.\./retread_fast_env\.sh$" "$W/a4.log"; then
  ok "a4: a legacy ../retread_fast_env.sh sitting there with different bytes is NOT imported"
else bad "a4: rc=$RC -- the decoy was picked up"; sed 's/^/      /' "$W/a4.log"; fi

echo "GUARD: fast_env_resolution: harnesses=$NTPL pass=$pass fail=$fail"
[ "$fail" = 0 ] || exit 1
exit 0
