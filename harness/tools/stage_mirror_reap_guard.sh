#!/usr/bin/env bash
# stage_mirror_reap_guard.sh -- STAGE-MIRROR-3. The reader for the two verbs in
# tools/stage_mirror.sh: `stage_mirror_reap_building` and
# `stage_mirror_repromote`.
#
#   usage: stage_mirror_reap_guard.sh          (HARNESS_REPO=<repo> to point it)
#
# FIXTURE-ONLY. Every mirror parent here is a handful of files under $TMPDIR.
# The guard NEVER touches $STAGE_MIRROR_ROOT, and it must not: the whole reason
# these verbs exist is that the real mirror parent is a declared PERSISTENT root
# that `cleanup.sh` and `multiarm_store_reap.sh` refuse by name. `squeue` is
# SHIMMED for the running-job arms, the same way cleanup_absent_root_guard.sh
# shims `sacct`, so no arm depends on the real queue holding a particular job.
#
# WHAT THESE VERBS ARE FOR, both measured on 2026-09-07:
#   * `85db7fdbbf51206a0cb57fa0d55e0e74.building.999999-MH1-2247202` -- a partial
#     `cp -al` tree left when hc14-guard 6023585 ran a template for real with the
#     job id forced to 999999. 12908 entries, no `.stage-mirror-key` (the key is
#     written last, so its absence IS the proof the build never finished), and a
#     job id that no scheduler ever issued.
#   * `…74.DIRTY-6022684-MCB-1788788544` -- quarantined by mCB-relock 6022684 on
#     a diff that was pure reordering (STAGE-MIRROR-2's locale bug). A fresh
#     LC_ALL=C census of it is byte-identical to its own stored manifest.
#
# THE ARMS
#   R1  an OLD .building temp whose jid is not in the queue, with no key -> REMOVED
#   R2  the same temp while its jid IS in the queue (shimmed) -> REFUSED
#   R3  a NON-EXISTENT jid reads as NOT RUNNING, not as "cannot tell" -- this is
#       the 999999 case and the verb is useless without it
#   R4  a bare live key in the same parent is never even considered
#   R5  a .building temp that carries a .stage-mirror-key -> REFUSED (a finished
#       mirror wearing a temp's name)
#   R6  a temp younger than the age floor -> REFUSED
#   R7  a .DIRTY quarantine in the same parent is untouched by the reaper
#   P1  a .DIRTY whose tree matches its stored manifest -> PROMOTED to the bare key
#   P2  ... but never when a live key already exists -> REFUSED
#   P3  a .DIRTY whose tree does NOT match -> REFUSED, and it keeps its name
#   P4  a name that is not a .DIRTY -> REFUSED
#   M1  MUTATION: the queue check cut -> R2's running temp is destroyed. R2 can fail.
#   M2  MUTATION: the manifest comparison cut -> P3 promotes a genuinely dirty
#       tree. P3 can fail.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${HARNESS_REPO:-}
[ -n "$REPO" ] || REPO=$(cd -- "$HERE/../.." && pwd)
if [ ! -d "$REPO/harness/tools" ] || ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  REPO=/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools
fi
LIB=$REPO/harness/tools/stage_mirror.sh
[ -f "$LIB" ] || { echo "GUARD FATAL: no library at $LIB"; exit 2; }
W=$(mktemp -d "${TMPDIR:-/tmp}/stage-mirror-reap-guard.XXXXXX") || exit 2
trap 'chmod -R u+w "$W" 2>/dev/null; rm -rf "$W"' EXIT
pass=0; fail=0
ok()  { echo "GUARD  ok : $*"; pass=$((pass+1)); }
bad() { echo "GUARD FAIL: $*"; fail=$((fail+1)); }
echo "### stage_mirror_reap_guard for $LIB (STAGE-MIRROR-3)"

# A squeue that reports the given jid RUNNING and everything else EMPTY, which
# is exactly what the real one does for an id no scheduler ever issued.
mk_squeue () {  # mk_squeue <bindir> <jid that is "running"|->
  mkdir -p "$1"
  { printf '#!/usr/bin/env bash\n'
    printf 'want=%s\n' "$2"
    printf 'for a in "$@"; do case "$a" in -j) n=1;; *) [ "${n:-}" = 1 ] && { j=$a; n=0; };; esac; done\n'
    printf '[ "${j:-}" = "$want" ] && echo R\nexit 0\n'
  } > "$1/squeue"
  chmod +x "$1/squeue"
}
BIN_NONE=$W/bin-none; mk_squeue "$BIN_NONE" -
BIN_RUN=$W/bin-run;   mk_squeue "$BIN_RUN"  4242424

# A tiny mirror tree, plus the manifest a finished mirror publishes.
mk_tree () {   # mk_tree <dir>
  mkdir -p "$1/pkgs/a" "$1/src"
  printf 'one\n' > "$1/pkgs/a/one.txt"
  printf 'two\n' > "$1/src/two.txt"
}
publish () {   # publish <dir> <key> <jid> -- writes the manifest then the key, as the writer does
  ( . "$LIB"; stage_mirror_census "$1" > "$1/.stage-mirror-manifest.tsv" )
  { printf 'key=%s\n' "$2"; printf 'built_by_job=%s\n' "$3"
    printf 'entries=%s\n' "$(wc -l < "$1/.stage-mirror-manifest.tsv")"; } > "$1/.stage-mirror-key"
}
age_out () { find "$1" -maxdepth 0 -exec touch -d '3 hours ago' {} + ; }

reap () {  # reap <bindir> <parent> [minage]
  local bin=$1 parent=$2 minage=${3:-60}
  ( PATH="$bin:$PATH"; . "$LIB"; stage_mirror_reap_building "$parent" "$minage" ) 2>&1
}
promote () {  # promote <dir>
  ( . "$LIB"; stage_mirror_repromote "$1" ) 2>&1
}

KEY=85db7fdbbf51206a0cb57fa0d55e0e74

########## R1/R3/R4/R7: one parent holding one of everything ##################
P1D=$W/parent1; mkdir -p "$P1D"
T_OLD=$P1D/$KEY.building.999999-MH1-2247202; mk_tree "$T_OLD"; age_out "$T_OLD"
LIVE=$P1D/$KEY;                              mk_tree "$LIVE";  publish "$LIVE" "$KEY" 6014471
DIRTYD=$P1D/$KEY.DIRTY-6022684-MCB-1788788544; mk_tree "$DIRTYD"; publish "$DIRTYD" "$KEY" 6022684
OUT=$(reap "$BIN_NONE" "$P1D")
if grep -q "### STAGE-REAP BUILDING parent=$P1D seen=1 removed=1 refused=0" <<<"$OUT"; then
  ok "R1/R3: the 999999 temp is removed and the footer counts it -- a job id no scheduler issued reads as NOT RUNNING"
else
  bad "R1/R3: footer wrong:"; sed 's/^/      /' <<<"$OUT"
fi
[ ! -e "$T_OLD" ] && ok "R1: the temp is really gone" || bad "R1: the temp survived"
[ -d "$LIVE" ] && [ -f "$LIVE/.stage-mirror-key" ] \
  && ok "R4: the bare live key in the same parent was never considered (seen=1, not 3)" \
  || bad "R4: THE LIVE KEY WAS TOUCHED"
[ -d "$DIRTYD" ] && ok "R7: the .DIRTY quarantine is not the reaper's business either" || bad "R7: the quarantine was removed"

########## R2: the jid IS in the queue ########################################
P2D=$W/parent2; mkdir -p "$P2D"
T_RUN=$P2D/$KEY.building.4242424-MH1-9; mk_tree "$T_RUN"; age_out "$T_RUN"
OUT=$(reap "$BIN_RUN" "$P2D")
if grep -q 'stage-reap REFUSED' <<<"$OUT" && grep -q 'still in the queue' <<<"$OUT" \
   && grep -q 'removed=0 refused=1' <<<"$OUT" && [ -d "$T_RUN" ]; then
  ok "R2: a temp whose job is still queued is REFUSED and left on disk"
else
  bad "R2: a RUNNING job's temp was not protected:"; sed 's/^/      /' <<<"$OUT"
fi

########## R5: a temp carrying a key ##########################################
P3D=$W/parent3; mkdir -p "$P3D"
T_KEY=$P3D/$KEY.building.999998-MH1-9; mk_tree "$T_KEY"; publish "$T_KEY" "$KEY" 999998; age_out "$T_KEY"
OUT=$(reap "$BIN_NONE" "$P3D")
if grep -q 'FINISHED mirror wearing a temp' <<<"$OUT" && grep -q 'removed=0 refused=1' <<<"$OUT" && [ -d "$T_KEY" ]; then
  ok "R5: a .building name over a tree that HAS a key is refused -- the key is the finished-ness proof"
else
  bad "R5: a keyed tree was reaped:"; sed 's/^/      /' <<<"$OUT"
fi

########## R6: too young ######################################################
P4D=$W/parent4; mkdir -p "$P4D"
T_NEW=$P4D/$KEY.building.999997-MH1-9; mk_tree "$T_NEW"
OUT=$(reap "$BIN_NONE" "$P4D")
if grep -q 'younger than the' <<<"$OUT" && grep -q 'removed=0 refused=1' <<<"$OUT" && [ -d "$T_NEW" ]; then
  ok "R6: a temp younger than the age floor is refused -- a build that just started is not a leftover"
else
  bad "R6: the age floor did not hold:"; sed 's/^/      /' <<<"$OUT"
fi

########## P1: the promotion ##################################################
P5D=$W/parent5; mkdir -p "$P5D"
D_OK=$P5D/$KEY.DIRTY-6022684-MCB-1788788544; mk_tree "$D_OK"; publish "$D_OK" "$KEY" 6022684
OUT=$(promote "$D_OK"); rc=$?
if [ "$rc" = 0 ] && grep -q "### STAGE-REPROMOTE promoted=1 refused=0 key=$KEY" <<<"$OUT" \
   && [ -d "$P5D/$KEY" ] && [ ! -e "$D_OK" ]; then
  ok "P1: a quarantine matching its own stored manifest is promoted back to the bare key"
else
  bad "P1: rc=$rc, promotion failed:"; sed 's/^/      /' <<<"$OUT"
fi
grep -q 'the quarantine was a false positive' <<<"$OUT" \
  && ok "P1: and it says WHY, with the row count and md5 it compared" \
  || bad "P1: no false-positive row"

########## P2: a live key already there #######################################
P6D=$W/parent6; mkdir -p "$P6D"
mk_tree "$P6D/$KEY"; publish "$P6D/$KEY" "$KEY" 6014471
D_CLASH=$P6D/$KEY.DIRTY-6022684-MCB-1; mk_tree "$D_CLASH"; publish "$D_CLASH" "$KEY" 6022684
OUT=$(promote "$D_CLASH"); rc=$?
if [ "$rc" = 1 ] && grep -q 'a live key already exists' <<<"$OUT" && [ -d "$D_CLASH" ] && [ -d "$P6D/$KEY" ]; then
  ok "P2: promotion over a LIVE key is refused -- a running job may be staged from it"
else
  bad "P2: rc=$rc, a live key was overwritten:"; sed 's/^/      /' <<<"$OUT"
fi

########## P3: genuinely dirty ################################################
P7D=$W/parent7; mkdir -p "$P7D"
D_BAD=$P7D/$KEY.DIRTY-6022684-MCB-2; mk_tree "$D_BAD"; publish "$D_BAD" "$KEY" 6022684
printf 'written through\n' > "$D_BAD/src/three.txt"     # the thing a quarantine is FOR
OUT=$(promote "$D_BAD"); rc=$?
if [ "$rc" = 1 ] && grep -q 'does NOT match its own stored manifest' <<<"$OUT" \
   && grep -q 'promoted=0 refused=1' <<<"$OUT" && [ -d "$D_BAD" ] && [ ! -e "$P7D/$KEY" ]; then
  ok "P3: a tree that really changed is REFUSED and keeps its quarantine name"
else
  bad "P3: rc=$rc, a dirty tree was promoted:"; sed 's/^/      /' <<<"$OUT"
fi

########## P4: not a quarantine at all ########################################
P8D=$W/parent8; mkdir -p "$P8D"; mk_tree "$P8D/$KEY"; publish "$P8D/$KEY" "$KEY" 1
OUT=$(promote "$P8D/$KEY"); rc=$?
if [ "$rc" = 2 ] && grep -q 'only a .DIRTY' <<<"$OUT" && [ -d "$P8D/$KEY" ]; then
  ok "P4: a bare key is not a promotable name (rc=2)"
else
  bad "P4: rc=$rc, a non-quarantine was accepted:"; sed 's/^/      /' <<<"$OUT"
fi

########## M1: MUTATION -- the queue check cut ################################
MUT1=$W/stage_mirror.M1.sh
sed 's/^    if \[ -n "\$qst" \]; then$/    if false; then/' "$LIB" > "$MUT1"
m1=$(diff "$LIB" "$MUT1" | grep -c '^< ')
if [ "$m1" -ne 1 ]; then
  bad "M1: the mutation changed $m1 line(s), want exactly 1 -- R2 cannot fail"
else
  P9D=$W/parent9; mkdir -p "$P9D"
  T9=$P9D/$KEY.building.4242424-MH1-9; mk_tree "$T9"; age_out "$T9"
  OUT=$( PATH="$BIN_RUN:$PATH"; . "$MUT1"; stage_mirror_reap_building "$P9D" 60 2>&1 )
  if [ ! -e "$T9" ]; then
    ok "M1: THE DEFECT, REPRODUCED -- without the queue check a RUNNING job's temp is destroyed under it"
  else
    bad "M1: the mutant still refused -- R2 proves nothing:"; sed 's/^/      /' <<<"$OUT"
  fi
fi

########## M2: MUTATION -- the manifest comparison cut ########################
MUT2=$W/stage_mirror.M2.sh
sed 's|^  if ! diff -q -- "\$wd/stored.tsv" "\$wd/now.tsv" >/dev/null 2>&1; then$|  if false; then|' "$LIB" > "$MUT2"
m2=$(diff "$LIB" "$MUT2" | grep -c '^< ')
if [ "$m2" -ne 1 ]; then
  bad "M2: the mutation changed $m2 line(s), want exactly 1 -- P3 cannot fail"
else
  P10D=$W/parent10; mkdir -p "$P10D"
  D10=$P10D/$KEY.DIRTY-6022684-MCB-3; mk_tree "$D10"; publish "$D10" "$KEY" 6022684
  printf 'written through\n' > "$D10/src/three.txt"
  OUT=$( . "$MUT2"; stage_mirror_repromote "$D10" 2>&1 )
  if [ -d "$P10D/$KEY" ]; then
    ok "M2: THE DEFECT, REPRODUCED -- without the comparison a genuinely dirty tree is promoted back into service"
  else
    bad "M2: the mutant still refused -- P3 proves nothing:"; sed 's/^/      /' <<<"$OUT"
  fi
fi

echo "### stage_mirror_reap_guard: pass=$pass fail=$fail -- $( [ "$fail" = 0 ] && echo PASS || echo FAIL )"
[ "$fail" = 0 ]
