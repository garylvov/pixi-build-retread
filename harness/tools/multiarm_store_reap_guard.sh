#!/usr/bin/env bash
# multiarm_store_reap_guard.sh -- the fixture that proves `multiarm_store_reap.sh`
# (a) leaves a SEALED tree intact on the rename path and lists it for the cleanup
# owner, (b) removes one with ZERO errors on the chmod path, (c) REFUSES a path
# that is not under the job's own root -- by inode, not by string -- and (d) that
# the PRE-FIX idiom it replaces really does produce the 42k/exit-0 shape on the
# same fixture.  ORDER-1-1.
#
# It needs no compiler and no retread binary: the seal is reproduced from C18's
# own rule (`source_build.rs::make_source_tree_read_only` -- depth-first, every
# node INCLUDING DIRECTORIES, `mode & !0o222`), which is the whole cause.  So it
# runs in the guard set beside the other shell guards on 1 CPU.
#
# THE CHECKS, and each one is a claim `multiarm_store_reap.sh` makes:
#   1. NON-VACUITY: the fixture really is sealed -- its directories have no `w`
#      bit -- and an UNSEALED copy of the identical tree is removed by a plain
#      `rm -rf` with ZERO errors.  Without this every later check could pass on
#      a tree that was never sealed.
#   2. PRE-FIX MUTATION: the idiom the thirteen task drivers carry
#      (`mv $X $X.reap; rm -rf $X.reap` inside a function whose caller keeps
#      going) produces $PREFIX_ERRS `Permission denied` lines, ZERO
#      `Directory not empty` lines, leaves ALL $FIXTURE_FILES files in place,
#      and the enclosing driver still exits 0.  That is order1-proof 5980023 in
#      miniature.
#   3. RENAME PATH: `reap_aside` on the same sealed fixture renames it to
#      `<path>.reap-<arm label>`, runs NO `rm` (errors 0), leaves all
#      $FIXTURE_FILES files and the seal itself untouched, and puts the new path
#      in the handoff list for the cleanup owner.  `reap_done` rc 0.
#   4. CHMOD PATH: `reap_delete` on the same sealed fixture removes it with
#      chmod_err 0, rm_err 0, `exists_after=no`, and `reap_done` rc 0.
#   5. REFUSAL, plain: a directory outside the declared job root is refused
#      rc 6, is STILL THERE afterwards, and `reap_done` returns 6.
#   6. REFUSAL IS MEASURED, NOT A PREFIX -- both directions, which is the only
#      way to show it:
#        (a) a target whose STRING begins with the job root but which really
#            lives outside it (reached through a symlinked ancestor) is REFUSED;
#            a `case "$t" in "$JOB_ROOT"/*)` test would have accepted it.
#        (b) a target whose STRING does NOT begin with the job root but which
#            really IS inside it (reached through a symlinked ancestor pointing
#            in) is ACCEPTED; the same prefix test would have refused it.
#   7. THE PERSISTENT STORE, both directions: a target that CONTAINS a declared
#      persistent root is refused, and a `REAP_JOB_ROOT` at or under one is
#      refused by `reap_init` rc 5.
#   8. `reap_init` refusals: unset job root rc 3, non-directory rc 3.
#   9. A reap attempted with no `reap_init` is refused rc 3.
#
#   rc 0  all of them hold
#   rc 1  a check failed (the row says which)
#   rc 2  the fixture could not be set up
#
# THE MUTATION IS PINNED TO A COMMIT CONSTANT, NEVER `HEAD`.  $PREFIX is the last
# harness commit BEFORE this file existed; check 2 asserts against git that
# `harness/tools/multiarm_store_reap.sh` is genuinely absent there, so "pre-fix"
# is proved rather than asserted, and the behavioural arm is the idiom lifted
# verbatim from `order1-work/order1_proof.sh` (which is task-only and therefore
# has no commit of its own -- that absence is the reason this lane exists).
set -uo pipefail

PREFIX=${REAP_GUARD_PREFIX:-09a0035be9c3fcdc3bf425bfa4b4836dad466d82}
GREPO=${HARNESS_REPO:-/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools}

SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${1:-$SELF_DIR/multiarm_store_reap.sh}"
[ -f "$TEMPLATE" ] || { echo "### GUARD FATAL: no template at $TEMPLATE"; exit 2; }

BASE="${REAP_GUARD_BASE:-/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/order1-1-work/guardfix}"

# The fixture's shape, and the numbers every check reconciles to.  MEASURED on a
# compute node before they were written here (order1-1-work/logs, probe job in
# the ORDER-1-1 rows of LANE-SPEED-LOG.md), not derived on paper.
FIXTURE_FILES=51        # 50 payload files + source.json
FIXTURE_DIRS=9          # root + canonical-git-sources + v3 + ident + refstate + repo + a + b + c
PREFIX_ERRS=51          # MEASURED: one per FILE.  rm never attempts the rmdir of a directory it failed to empty, so the 8 subdirectories add no rows -- which is why 5980023 shows 42245 "Permission denied" and ZERO "Directory not empty".

fail=0
say() { echo "### REAPGUARD $*"; }
ck()  { # ck <name> <got> <want>
  if [ "$2" = "$3" ]; then say "OK   $1: $2"; else say "FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi
}

rm_hard() { chmod -R u+w "$1" 2>/dev/null; rm -rf "$1" 2>/dev/null; }
rm_hard "$BASE"
mkdir -p "$BASE" || { echo "### GUARD FATAL: cannot create $BASE"; exit 2; }
trap 'rm_hard "$BASE"' EXIT

# ── the fixture, and the seal, exactly as C18 makes one ─────────────────────
mk_tree() {
  local R="$1" d f
  mkdir -p "$R/canonical-git-sources/v3/ident/refstate/repo" || return 2
  printf '{"schema":"retread-canonical-git-seal-v1"}\n' > "$R/canonical-git-sources/v3/ident/refstate/source.json"
  for f in 1 2 3 4 5; do printf 'top %s\n' "$f" > "$R/canonical-git-sources/v3/ident/refstate/repo/top$f.txt"; done
  for d in a b c; do
    mkdir -p "$R/canonical-git-sources/v3/ident/refstate/repo/$d" || return 2
    for f in $(seq 1 15); do printf '%s %s\n' "$d" "$f" > "$R/canonical-git-sources/v3/ident/refstate/repo/$d/f$f.txt"; done
  done
  return 0
}
# `make_source_tree_read_only`: depth-first, symlinks skipped, `mode & !0o222`
# on FILES AND DIRECTORIES alike.  The directory half is the cause.
seal() {
  local p m
  while IFS= read -r -d '' p; do
    m=$(stat -c '%a' -- "$p" 2>/dev/null) || continue
    chmod "$(printf '%o' $(( 8#$m & ~8#222 )))" -- "$p" 2>/dev/null
  done < <(find "$1" -depth ! -type l -print0 2>/dev/null)
}
nfiles() { find "$1" ! -type d 2>/dev/null | wc -l | tr -d ' '; }
ndirs()  { find "$1" -type d 2>/dev/null | wc -l | tr -d ' '; }
sealed_root() { stat -c '%a' -- "$1/canonical-git-sources/v3/ident/refstate/repo" 2>/dev/null; }

# ── CHECK 1: NON-VACUITY ────────────────────────────────────────────────────
V=$BASE/vac; mk_tree "$V" || exit 2
ck "1a fixture file count"        "$(nfiles "$V")" "$FIXTURE_FILES"
UNSEALED_ERR=$(rm -rf "$V" 2>&1 | wc -l | tr -d ' ')
ck "1b unsealed rm error lines"   "$UNSEALED_ERR" "0"
ck "1c unsealed tree gone"        "$([ -e "$V" ] && echo yes || echo no)" "no"
S=$BASE/vac2; mk_tree "$S" || exit 2; seal "$S"
SM=$(sealed_root "$S")
case "$SM" in *[2367]) say "FAIL 1d sealed repo dir still has a w bit: $SM"; fail=$((fail+1)) ;;
              *) say "OK   1d sealed repo dir mode: $SM (no w bit anywhere)" ;; esac
rm_hard "$S"

# ── CHECK 2: THE PRE-FIX MUTATION, pinned ───────────────────────────────────
if git -C "$GREPO" cat-file -e "$PREFIX:harness/tools/multiarm_store_reap.sh" 2>/dev/null; then
  say "FAIL 2a $PREFIX already carries multiarm_store_reap.sh -- PREFIX is not a pre-fix commit"; fail=$((fail+1))
else
  say "OK   2a $PREFIX has no harness/tools/multiarm_store_reap.sh (pre-fix, proved from git)"
fi
# The idiom AS THE THIRTEEN DRIVERS CARRY IT, inside a function whose caller keeps
# going -- which is exactly how 5980023 reached `job_fatal=0`.
prefix_arm() {
  local X="$1"
  if mv "$X" "$X.reap" 2>/dev/null; then
    rm -rf "$X.reap" 2>"$BASE/prefix.err"
    echo "### ARM cold root reaped: $X.reap"
  fi
  return 0
}
prefix_driver() {
  prefix_arm "$1"
  echo "### PROOF DONE job_fatal=0"
  return 0
}
P=$BASE/pre; mk_tree "$P" || exit 2; seal "$P"
prefix_driver "$P" >/dev/null
PRC=$?
PE=$(wc -l < "$BASE/prefix.err" 2>/dev/null | tr -d ' ')
PDENY=$(grep -c 'Permission denied' "$BASE/prefix.err" 2>/dev/null)
PNOTEMPTY=$(grep -c 'Directory not empty' "$BASE/prefix.err" 2>/dev/null)
ck "2b pre-fix driver exit status"    "$PRC" "0"
ck "2c pre-fix error lines"           "$PE" "$PREFIX_ERRS"
ck "2d pre-fix all Permission denied" "$PDENY" "$PREFIX_ERRS"
ck "2e pre-fix Directory-not-empty"   "$PNOTEMPTY" "0"
ck "2f pre-fix tree SURVIVES"         "$(nfiles "$P.reap")" "$FIXTURE_FILES"
rm_hard "$P.reap"

# ── the template under test ─────────────────────────────────────────────────
# shellcheck disable=SC1090
source "$TEMPLATE" || { echo "### GUARD FATAL: cannot source $TEMPLATE"; exit 2; }

J=$BASE/job; mkdir -p "$J" || exit 2
REAP_PERSISTENT_ROOTS="$BASE/persist"; mkdir -p "$REAP_PERSISTENT_ROOTS" || exit 2

# ── CHECK 9: no reap_init ───────────────────────────────────────────────────
REAP_INIT_DONE=0
N=$J/noinit; mk_tree "$N" >/dev/null 2>&1
reap_aside "$N" a0 >/dev/null 2>&1; ck "9 reap before reap_init" "$?" "3"
rm_hard "$N"

# ── CHECK 8: reap_init refusals ─────────────────────────────────────────────
( REAP_JOB_ROOT=""            reap_init >/dev/null 2>&1 ); ck "8a init with no job root"      "$?" "3"
( REAP_JOB_ROOT="$J/nothere"  reap_init >/dev/null 2>&1 ); ck "8b init on a missing dir"      "$?" "3"
touch "$J/afile"
( REAP_JOB_ROOT="$J/afile"    reap_init >/dev/null 2>&1 ); ck "8c init on a regular file"     "$?" "3"
# CHECK 7b: a job root at/under a persistent root
mkdir -p "$REAP_PERSISTENT_ROOTS/inside"
( REAP_JOB_ROOT="$REAP_PERSISTENT_ROOTS/inside" reap_init >/dev/null 2>&1 ); ck "7b init under the persistent root" "$?" "5"
( REAP_JOB_ROOT="$REAP_PERSISTENT_ROOTS"        reap_init >/dev/null 2>&1 ); ck "7c init AT the persistent root"    "$?" "5"

REAP_JOB_ROOT="$J"
reap_init >/dev/null || { echo "### GUARD FATAL: reap_init refused the fixture job root"; exit 2; }

# ── CHECK 3: THE RENAME PATH ON A SEALED TREE ───────────────────────────────
A=$J/git-snapshots-cold-a1; mk_tree "$A" || exit 2; seal "$A"
A_MODE=$(sealed_root "$A")
reap_aside "$A" a1 >/dev/null; ARC=$?
ck "3a reap_aside rc"                "$ARC" "0"
ck "3b renamed with the arm label"   "$([ -d "$A.reap-a1" ] && echo yes || echo no)" "yes"
ck "3c original name is free"        "$([ -e "$A" ] && echo yes || echo no)" "no"
ck "3d tree INTACT"                  "$(nfiles "$A.reap-a1")" "$FIXTURE_FILES"
ck "3e seal untouched"               "$(sealed_root "$A.reap-a1")" "$A_MODE"
ck "3f no rm ran (error count)"      "$REAP_ERRORS" "0"
reap_handoff_list "$J/REAP-OWED.txt" >/dev/null
ck "3g handoff names the new path"   "$(grep -c "^$A.reap-a1\$" "$J/REAP-OWED.txt")" "1"
reap_done >/dev/null; ck "3h reap_done rc after the rename path" "$?" "0"
rm_hard "$A.reap-a1"

# ── CHECK 4: THE CHMOD PATH ON A SEALED TREE ────────────────────────────────
REAP_JOB_ROOT="$J"; reap_init >/dev/null
B2=$J/git-snapshots-cold-a2; mk_tree "$B2" || exit 2; seal "$B2"
DOUT=$(reap_delete "$B2" a2 2>&1); DRC=$?
ck "4a reap_delete rc"               "$DRC" "0"
ck "4b tree is gone"                 "$([ -e "$B2.reap-a2" ] && echo yes || echo no)" "no"
ck "4c row says exists_after=no"     "$(printf '%s\n' "$DOUT" | grep -c 'exists_after=no')" "1"
ck "4d chmod_err=0 rm_err=0"         "$(printf '%s\n' "$DOUT" | grep -c 'chmod_err=0 rm_err=0 errors=0')" "1"
ck "4e error count"                  "$REAP_ERRORS" "0"
reap_done >/dev/null; ck "4f reap_done rc after the chmod path" "$?" "0"

# ── CHECK 5: A PLAIN OUTSIDE PATH IS REFUSED ────────────────────────────────
REAP_JOB_ROOT="$J"; reap_init >/dev/null
O=$BASE/outside; mk_tree "$O" || exit 2
reap_aside  "$O" a3 >/dev/null 2>&1; ck "5a reap_aside on an outside path"  "$?" "6"
reap_delete "$O" a3 >/dev/null 2>&1; ck "5b reap_delete on an outside path" "$?" "6"
ck "5c the outside tree is untouched" "$(nfiles "$O")" "$FIXTURE_FILES"
ck "5d refusal counted"               "$REAP_REFUSED" "2"
reap_done >/dev/null; ck "5e reap_done rc after a refusal" "$?" "6"

# ── CHECK 6: THE REFUSAL IS MEASURED, NOT A STRING PREFIX ───────────────────
REAP_JOB_ROOT="$J"; reap_init >/dev/null
# (a) string says inside, inode says outside
ln -s "$BASE/elsewhere" "$J/link-out"
mkdir -p "$BASE/elsewhere"; E=$BASE/elsewhere/tree; mk_tree "$E" || exit 2
reap_aside "$J/link-out/tree" a4 >/dev/null 2>&1
ck "6a prefix says inside, inode says outside -> REFUSED" "$?" "6"
ck "6b that tree is untouched" "$(nfiles "$E")" "$FIXTURE_FILES"
# (b) string says outside, inode says inside
REAP_JOB_ROOT="$J"; reap_init >/dev/null
I=$J/really-inside; mk_tree "$I" || exit 2; seal "$I"
ln -s "$J" "$BASE/link-in"
reap_aside "$BASE/link-in/really-inside" a5 >/dev/null 2>&1
ck "6c prefix says outside, inode says inside -> ACCEPTED" "$?" "0"
ck "6d it really moved" "$([ -d "$I.reap-a5" ] && echo yes || echo no)" "yes"
rm_hard "$I.reap-a5"

# ── CHECK 7a: A TARGET THAT CONTAINS THE PERSISTENT ROOT ────────────────────
REAP_JOB_ROOT="$J"; reap_init >/dev/null
W=$J/wrapper; mkdir -p "$W/sub" || exit 2
REAP_PERSISTENT_ROOTS="$W/sub"
reap_delete "$W" a6 >/dev/null 2>&1; ck "7a target CONTAINS a persistent root -> REFUSED" "$?" "6"
ck "7a2 the wrapper is untouched" "$([ -d "$W/sub" ] && echo yes || echo no)" "yes"
REAP_PERSISTENT_ROOTS="$BASE/persist"

say "SUMMARY failures=$fail"
[ "$fail" -eq 0 ] || exit 1
exit 0
