#!/usr/bin/env bash
# proof_smoke.sh -- ONE BOUNDED LOCK THAT ANSWERS "IS THIS BINARY A WORKING
# BUILD BACKEND", BEFORE A LANE COMMITS THREE HOURS TO FOUR COLD ARMS.
# PROOF-SMOKE-1, boarded at tick 519.
#
#   usage: bash proof_smoke.sh <binsnap> <manifest> <job root>
#
#     <binsnap>   a binsnap DIRECTORY (containing pixi-build-retread) or the
#                 binary itself.  Both are accepted; the sha256 printed is
#                 always the sha of the BINARY.
#     <manifest>  the manifest to lock.  The canonical one in production.
#     <job root>  where this smoke writes its artifacts and its log.
#
#   verdicts, and the ONE row every path prints last:
#
#     ### SMOKE REACHED_FRONTEND binary=<sha> wall=<s>     rc 0
#     ### SMOKE BACKEND_DIED     binary=<sha> wall=<s>     rc 1
#     ### SMOKE TIMEOUT          binary=<sha> wall=<s>     rc 2
#     ### SMOKE SETUP_FAILED     binary=<sha> wall=<s>     rc 3
#
# ── WHY IT EXISTS, MEASURED AND NOT ARGUED ───────────────────────────────────
#
# DET-1-FIX's candidate `3f2095a` passed its gate `1868 passed; 0 failed; 21
# ignored` -- the number it PREDICTED -- and then, as a build backend, died in
# every arm that ran it.  `det1f-proof2` **5989192** (node, 09-06 17:2x):
#
#   arm 1 FIX  rc=1 wall= 44 s   resolve_pypi rows = 0    lock_sha=NONE
#   arm 2 FIX  rc=1 wall= 49 s   resolve_pypi rows = 0    lock_sha=NONE
#   arm 4 FIX  rc=1 wall= 46 s   resolve_pypi rows = 0    lock_sha=NONE
#   arm 3 CTL  rc=0 wall=2566 s  resolve_pypi rows = 173  (same node, same hour)
#
# Three hours of queue and three cold arms bought one fact that forty-five
# seconds would have bought: THE CANDIDATE CANNOT SERVE A LOCK.  A green gate is
# not a working backend, and nothing in this campaign's harness asked the
# question before the expensive part.  This file asks it.
#
# ── THE DISCRIMINATOR, AND IT IS A MEASUREMENT ───────────────────────────────
#
# `grep -c resolve_pypi` over the two logs of that one job: **0** on the dead
# FIX arm (118 lines total) and **173** on the live CTL arm (68 426 lines).  The
# first frontend row on the CTL arm is at 17:36:51 against `### lock start
# 2026-09-06T17:33:32`, i.e. **199 s on a fully cold cache** -- so the frontend
# marker both SEPARATES the two outcomes and ARRIVES EARLY, and a smoke does not
# have to finish a lock to answer the question.  The moment a frontend row
# appears this script KILLS the lock and returns REACHED_FRONTEND.  The bounded
# wall is therefore a ceiling that a healthy binary never reaches, not a budget
# it spends.
#
# WHAT "REACHED THE FRONTEND" DOES AND DOES NOT CLAIM.  It claims the backend
# negotiated, answered the frontend's build-dispatch calls for at least one
# source, and the resolver started work.  It does NOT claim the lock will
# finish, that the lock will be correct, or that a later environment will not
# hit a different backend path.  It is a smoke test.  Its whole value is that it
# is CHEAP and that it fails in the way the expensive run would.
#
# ── THE TWO NAMED DEATHS IT DETECTS BY NAME ──────────────────────────────────
#
# (1) THE 256-BYTE PREFIX PANIC (DET-1-FIX-1).  `rattler-build debug setup` pads
#     its build prefix to a fixed 256 bytes; at a composed length of 260 the
#     `256 - len` underflows and it panics in
#     `rattler_build_core/src/types/directories.rs` with
#     `end byte index 18446744073709551615 is out of bounds for string of
#     length 260`.  Measured: the parent proof's store root composed to exactly
#     256 and provisioned; the re-cut's was four bytes longer and every FIX arm
#     died.  This script reports that death as `reason=PREFIX_PANIC_256`, so a
#     lane never again reads a path-length accident as a code defect.
#
# (2) THE PRE-LOCK REFUSAL DET-1-FIX-1 ADDED.  The same check, run BEFORE the
#     lock: this script composes the longest hermetic entry path its own store
#     root would produce and refuses at > 256 with `reason=PREFIX_REFUSAL`,
#     because a smoke run from a root too long to provision would report
#     BACKEND_DIED against a perfectly good binary.  A guard that blames the
#     wrong thing is worse than no guard.  It is a SETUP_FAILED, not a verdict
#     on the binary -- rc 3, and the row says which root to shorten.
#
# ── WHAT IS DELIBERATELY NOT HERE ────────────────────────────────────────────
#
# NO `pkill`, NO `pgrep`, NO cmdline scan of any kind (CLAUDE.md laws 12 and
# 14).  The lock runs under `setsid` in its own process group and is killed by
# that pgid; the kill is then VERIFIED by walking `/proc/<pid>/stat` field 5
# (the pgrp) -- a numeric field, so no pattern this script uses can match this
# script.  A kill that reports success while a process survives is settled by
# measurement, never by an exit code.
#
# NO write to the canonical source tree.  The workspace is staged from the
# keyed stage mirror (a REAL copy of the source, never a hardlink into it) or,
# when no mirror matches, by rsync.  `SMOKE_WS` lets a caller hand in a
# workspace it already staged, which is what makes smoking N binaries in one
# job cost one staging.
set -uo pipefail

PROOF_SMOKE_VERSION=1

SMOKE_SRC_WS=${SMOKE_SRC_WS:-/oscar/data/stellex/glvov/imprint-data}
SMOKE_PIXI=${SMOKE_PIXI:-/users/glvov/.pixi/bin/pixi.real}
SMOKE_WALL=${SMOKE_WALL:-900}
SMOKE_POLL=${SMOKE_POLL:-2}
SMOKE_MIRROR_ROOT=${SMOKE_MIRROR_ROOT:-/oscar/data/stellex/glvov/agrescap/cache/retread/stage-mirror}
SMOKE_STAGE_PAR=${SMOKE_STAGE_PAR:-16}
# The short root.  SHORT IS THE POINT (see the prefix panic above): every byte
# spent here is a byte the hermetic entry path cannot have.
SMOKE_ROOT_BASE=${SMOKE_ROOT_BASE:-/oscar/data/stellex/glvov/retread}
SMOKE_RUST_LOG=${SMOKE_RUST_LOG:-uv_distribution=debug,pixi=info,pixi_core=info,pixi_command_dispatcher=info,warn}
SMOKE_BACKEND_LOG=${SMOKE_BACKEND_LOG:-pixi_build_retread=debug,warn}
# The rattler-build padding target and the fixed tail the composed prefix adds
# to the store entry path.  Both MEASURED on 5989192 / DET1-5981957: 164+92=256
# provisioned, 168+92=260 panicked.
SMOKE_PREFIX_PAD=${SMOKE_PREFIX_PAD:-256}
SMOKE_PREFIX_TAIL=${SMOKE_PREFIX_TAIL:-92}
SMOKE_PREFIX_HEADROOM=${SMOKE_PREFIX_HEADROOM:-8}

# THE FRONTEND MARKERS.  `resolve_pypi{` is the span the resolver opens per
# environment; `Preparing metadata for:` is uv building an sdist's metadata
# inside it; the "assumed to be installed by conda" row is the resolver's own
# first INFO line.  Any one of them means the frontend moved.
SMOKE_FRONTEND_RE=${SMOKE_FRONTEND_RE:-'resolve_pypi\{|Preparing metadata for:|are assumed to be installed by conda|Found static `pyproject\.toml` for:'}
# The first line worth quoting when nothing reached the frontend.
SMOKE_ERROR_RE=${SMOKE_ERROR_RE:-'panicked at|^Error|ERROR|error\[|error:|× |failed with status'}

smoke_say () { echo "### SMOKE $*"; }

# ---- the verdict row, and it is printed on EVERY exit path ------------------
SMOKE_VERDICT=SETUP_FAILED
SMOKE_BINSHA=UNKNOWN
SMOKE_T0=$(date +%s)
SMOKE_RC=3
smoke_finish () {
  local wall=$(( $(date +%s) - SMOKE_T0 ))
  echo "### SMOKE $SMOKE_VERDICT binary=$SMOKE_BINSHA wall=${wall}s"
  exit "$SMOKE_RC"
}

smoke_setup_failed () {
  echo "### SMOKE SETUP FATAL: $*"
  SMOKE_VERDICT=SETUP_FAILED; SMOKE_RC=3; smoke_finish
}

# ── the composed-prefix budget.  Also exported for the preamble to reuse. ─────
# Returns 0 when the root is short enough, 1 when it is not.  PRINTS the two
# measured numbers either way, because "it passed" is only readable if the
# margin is on the page.
smoke_prefix_budget () {
  local root=$1 label=${2:-store root} entry len composed
  # the longest entry this store root can produce: <root>/retread/
  # hermetic-build-envs/v8/env-<64 hex>
  entry="$root/retread/hermetic-build-envs/v8/env-$(printf 'a%.0s' $(seq 1 64))"
  len=${#entry}
  composed=$(( len + SMOKE_PREFIX_TAIL ))
  echo "### SMOKE PREFIX BUDGET $label entry=$len composed=$composed pad=$SMOKE_PREFIX_PAD headroom=$(( SMOKE_PREFIX_PAD - composed )) root=$root"
  if [ "$composed" -gt "$SMOKE_PREFIX_PAD" ]; then
    echo "### SMOKE PREFIX REFUSAL: composed $composed > $SMOKE_PREFIX_PAD."
    echo "### SMOKE   rattler-build pads its build prefix to $SMOKE_PREFIX_PAD and panics on the underflow"
    echo "### SMOKE   in rattler_build_core/src/types/directories.rs before anything provisions."
    echo "### SMOKE   SHORTEN THE ROOT, NOT THE CHECK: $root"
    return 1
  fi
  if [ "$composed" -gt $(( SMOKE_PREFIX_PAD - SMOKE_PREFIX_HEADROOM )) ]; then
    echo "### SMOKE PREFIX NOTE: only $(( SMOKE_PREFIX_PAD - composed )) bytes of headroom under the pad target (want > $SMOKE_PREFIX_HEADROOM)"
    echo "### SMOKE PREFIX REFUSAL: the margin is inside the headroom band and the parent proof sat EXACTLY on the boundary without knowing."
    return 1
  fi
  return 0
}

# ── kill a process GROUP and PROVE it is gone (laws 12 and 14) ───────────────
# Selection is by pgid read from /proc/<pid>/stat field 5.  There is no cmdline
# pattern anywhere in here, so this scan cannot match itself.
smoke_pgid_members () {
  local want=$1 p pid st
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    st=$(cat "$p/stat" 2>/dev/null) || continue
    # field 5 is pgrp; comm (field 2) may contain spaces and parens, so cut at
    # the LAST ') ' first -- GREEDY, because only comm can contain one.
    st=${st##*') '}
    set -- $st
    [ "${3:-}" = "$want" ] || continue
    printf '%s\n' "$pid"
  done
}

smoke_kill_pgid () {
  local pgid=$1 n i
  [ -n "$pgid" ] && [ "$pgid" != 0 ] || return 0
  kill -TERM -- "-$pgid" 2>/dev/null
  for i in 1 2 3 4 5 6 7 8 9 10; do
    n=$(smoke_pgid_members "$pgid" | wc -l)
    [ "$n" -eq 0 ] && break
    sleep 1
  done
  n=$(smoke_pgid_members "$pgid" | wc -l)
  if [ "$n" -ne 0 ]; then
    echo "### SMOKE kill: SIGTERM left $n member(s) of pgid $pgid alive -- escalating to SIGKILL (law 12: SIGTERM is not sufficient by default)"
    kill -KILL -- "-$pgid" 2>/dev/null
    for i in 1 2 3 4 5; do
      n=$(smoke_pgid_members "$pgid" | wc -l); [ "$n" -eq 0 ] && break; sleep 1
    done
  fi
  n=$(smoke_pgid_members "$pgid" | wc -l)
  echo "### SMOKE kill: pgid=$pgid survivors_after_kill=$n (0 means the kill took, measured from /proc and not from an exit code)"
  [ "$n" -eq 0 ]
}

# ── staging ─────────────────────────────────────────────────────────────────
smoke_stage_key () {
  printf '%s %s' \
    "$(md5sum "$SMOKE_SRC_WS/pixi.toml" | awk '{print $1}')" \
    "$(git -C "$SMOKE_SRC_WS" rev-parse HEAD 2>/dev/null || echo nogit)" \
  | md5sum | awk '{print $1}'
}

smoke_stage () {                 # $1 = workspace to create
  local ws=$1 key mirror S
  key=$(smoke_stage_key)
  mirror=$SMOKE_MIRROR_ROOT/$key
  mkdir -p "$ws" || return 1
  if [ -f "$mirror/.stage-mirror-key" ] && grep -qx "key=$key" "$mirror/.stage-mirror-key"; then
    echo "### SMOKE stage: mirror HIT $mirror (key $key)"
    S=$(date +%s)
    ( cd "$mirror" && find . -mindepth 1 -maxdepth 2 -type d -printf '%P\n' ) \
      | grep -vF '.stage-mirror-' | sed "s|^|$ws/|" | tr '\n' '\0' \
      | xargs -0 -r -n 64 -P "$SMOKE_STAGE_PAR" mkdir -p || return 1
    # TWO finds, not one expression: -mindepth/-maxdepth are GLOBAL options in
    # GNU find, so a combined expression silently drops every shallow file.
    { ( cd "$mirror" && find . -mindepth 1 -maxdepth 2 ! -type d -printf '%P\n' )
      ( cd "$mirror" && find . -mindepth 3 -maxdepth 3            -printf '%P\n' ) } \
      | grep -vF '.stage-mirror-' | tr '\n' '\0' \
      | xargs -0 -r -I{} -P "$SMOKE_STAGE_PAR" cp -al "$mirror/{}" "$ws/{}" || return 1
    echo "### SMOKE stage: cp -al wall=$(( $(date +%s) - S ))s"
  else
    echo "### SMOKE stage: NO mirror for key $key -- rsync path (one-time cost)"
    S=$(date +%s)
    rsync -a --exclude '/.pixi/' --exclude '/pixi.lock' --exclude '/pixi.lock.*' \
             --exclude '/logs/' --exclude '/results/' --exclude '/scratchpad/' \
             --exclude '/third_party/' "$SMOKE_SRC_WS/" "$ws/" || return 1
    cp -al "$SMOKE_SRC_WS/third_party" "$ws/third_party" || return 1
    echo "### SMOKE stage: rsync+cp -al wall=$(( $(date +%s) - S ))s"
  fi
  # the per-run writable bits, never shared with the mirror
  rm -rf "$ws/.pixi"; mkdir -p "$ws/.pixi"
  [ -f "$SMOKE_SRC_WS/.pixi/config.toml" ] && cp "$SMOKE_SRC_WS/.pixi/config.toml" "$ws/.pixi/config.toml"
  return 0
}

# ---- the sourceable half ends here ------------------------------------------
# `PROOF_SMOKE_LIB=1 . proof_smoke.sh` gives a caller the length rule
# (`smoke_prefix_budget`) and the kill-and-prove helpers WITHOUT running a
# smoke.  `multiarm_preamble.sh` uses it so the 256-byte rule has exactly ONE
# implementation: two copies of a length rule is how the rule drifts.
if [ -n "${PROOF_SMOKE_LIB:-}" ]; then
  return 0 2>/dev/null || exit 0
fi

# ---- arguments --------------------------------------------------------------
BINSNAP=${1:-}; MANIFEST=${2:-}; JOB_ROOT=${3:-}
[ -n "$BINSNAP" ] && [ -n "$MANIFEST" ] && [ -n "$JOB_ROOT" ] \
  || { echo "### SMOKE usage: proof_smoke.sh <binsnap> <manifest> <job root>"; exit 3; }

BIN=$BINSNAP
[ -d "$BIN" ] && BIN=$BIN/pixi-build-retread
[ -x "$BIN" ] || smoke_setup_failed "no executable backend at $BIN"
[ -f "$MANIFEST" ] || smoke_setup_failed "no manifest at $MANIFEST"
mkdir -p "$JOB_ROOT" || smoke_setup_failed "cannot create job root $JOB_ROOT"
SMOKE_BINSHA=$(sha256sum "$BIN" | awk '{print $1}')

J=${SLURM_JOB_ID:-$$}
# A short, UNIQUE, job-owned root.  `smk` + jobid + a 4-hex nonce keeps two
# smokes in one job from colliding without spending bytes on a label.
NONCE=$(od -An -N2 -tx1 /dev/urandom 2>/dev/null | tr -d ' \n'); NONCE=${NONCE:-0000}
ROOT=${SMOKE_ROOT:-$SMOKE_ROOT_BASE/smk$J-$NONCE}
CACHE=${SMOKE_CACHE:-$ROOT/c}
WS=${SMOKE_WS:-$ROOT/w}
LOGDIR=$JOB_ROOT/artifacts
mkdir -p "$LOGDIR" "$ROOT" "$CACHE" || smoke_setup_failed "cannot create $ROOT"

STEM=smoke-$J-${SMOKE_BINSHA:0:7}
LLOG=$LOGDIR/$STEM.lock.log
BLOG=$LOGDIR/$STEM.backend.log
SHIM=$LOGDIR/$STEM.backend-shim.sh
: > "$LLOG"; : > "$BLOG"

echo "### SMOKE proof_smoke.sh v$PROOF_SMOKE_VERSION  $(date -Is)  host=$(hostname -s) job=$J"
echo "### SMOKE binary=$BIN sha256=$SMOKE_BINSHA"
echo "### SMOKE manifest=$MANIFEST md5=$(md5sum "$MANIFEST" | awk '{print $1}')"
echo "### SMOKE root=$ROOT ws=$WS cache=$CACHE wall_cap=${SMOKE_WALL}s"

# ---- the pre-lock prefix refusal (DET-1-FIX-1), BEFORE anything is staged ----
export XDG_CACHE_HOME=$CACHE/x
export XDG_DATA_HOME=$CACHE/d
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" || smoke_setup_failed "cannot create $XDG_CACHE_HOME"
smoke_prefix_budget "$XDG_CACHE_HOME" "smoke store root" || {
  echo "### SMOKE reason=PREFIX_REFUSAL -- this is a refusal about the ROOT, not a verdict about the binary."
  SMOKE_VERDICT=SETUP_FAILED; SMOKE_RC=3; smoke_finish
}

# ---- the workspace ----------------------------------------------------------
if [ -n "${SMOKE_WS:-}" ] && [ -f "$WS/pixi.toml" ]; then
  echo "### SMOKE stage: REUSING the caller's workspace $WS (SMOKE_WS was set and it is staged)"
else
  smoke_stage "$WS" || smoke_setup_failed "could not stage a workspace at $WS"
fi
rm -f "$WS/pixi.toml"; cp "$MANIFEST" "$WS/pixi.toml" || smoke_setup_failed "cannot install the manifest"
# A lock that already exists would let pixi answer without ever instantiating a
# backend, and the smoke would report REACHED_FRONTEND for a binary it never ran.
rm -f "$WS"/pixi.lock "$WS"/pixi.lock.* 2>/dev/null

# ---- the backend shim, which READS BACK ITS OWN EXEC LINE (DET-1-4-1) -------
cat > "$SHIM" <<SHIMEOF
#!/usr/bin/env bash
unset RUST_LOG
exec 2> >(tee -a "$BLOG" >&2)
exec "$BIN" "\$@"
SHIMEOF
chmod +x "$SHIM"
# The matcher is the BACKEND exec, not "the first line starting with exec": the
# shim's first such line is the stderr redirect.  Exactly one line may exec a
# pixi-build-retread path and it must be THIS smoke's binary.
SHIM_EXEC_N=$(grep -c -E '^exec "[^"]*/pixi-build-retread" "\$@"$' "$SHIM")
SHIM_EXEC=$(grep -m1 -E '^exec "[^"]*/pixi-build-retread" "\$@"$' "$SHIM")
echo "### SMOKE shim exec lines=$SHIM_EXEC_N line: ${SHIM_EXEC:-<none>}"
[ "$SHIM_EXEC_N" -eq 1 ] || smoke_setup_failed "the shim has $SHIM_EXEC_N backend exec lines, want exactly 1"
[ "$SHIM_EXEC" = "exec \"$BIN\" \"\$@\"" ] || smoke_setup_failed "the shim does not exec this smoke's binary (want exec \"$BIN\" \"\$@\", got $SHIM_EXEC)"
echo "### SMOKE shim ok: it execs THIS smoke's binary"

export PIXI_BUILD_BACKEND_OVERRIDE="pixi-build-retread=$SHIM"
export PIXI_CACHE_DIR=$CACHE/pixi
export RATTLER_CACHE_DIR=$CACHE/rattler
export UV_CACHE_DIR=$CACHE/uv
mkdir -p "$PIXI_CACHE_DIR" "$RATTLER_CACHE_DIR" "$UV_CACHE_DIR"
CACHE_STATE=cold
[ "$(ls -1U "$PIXI_CACHE_DIR" | wc -l)" -eq 0 ] || CACHE_STATE=warm
echo "### SMOKE cache state=$CACHE_STATE (job-scoped; a second smoke in the same job root reuses it on purpose)"
export RUST_LOG=$SMOKE_RUST_LOG
export PIXI_BUILD_RETREAD_LOG=$SMOKE_BACKEND_LOG

# ---- THE LOCK, in its own process group, polled ------------------------------
cd "$WS" || smoke_setup_failed "cannot cd $WS"
S=$(date +%s)
setsid "$SMOKE_PIXI" lock >>"$LLOG" 2>&1 &
LPID=$!
LPGID=$(awk '{sub(/^.*\) /,""); print $3}' "/proc/$LPID/stat" 2>/dev/null)
LPGID=${LPGID:-$LPID}
MYPGID=$(awk '{sub(/^.*\) /,""); print $3}' "/proc/$$/stat" 2>/dev/null)
echo "### SMOKE lock started pid=$LPID pgid=$LPGID (this script's pgid=$MYPGID) at $(date -Is)"
if [ "$LPGID" = "${MYPGID:-none}" ]; then
  # setsid did not take.  Killing this group would kill the smoke itself, so the
  # kill is degraded to the single pid and SAID SO rather than done silently.
  echo "### SMOKE WARNING: setsid did not move the lock into its own process group -- the kill will target pid $LPID only, and a surviving backend child is possible."
  LPGID=
fi

VERDICT=
while :; do
  ELAPSED=$(( $(date +%s) - S ))
  if grep -a -q -E "$SMOKE_FRONTEND_RE" "$LLOG" 2>/dev/null; then
    VERDICT=REACHED_FRONTEND; break
  fi
  if ! kill -0 "$LPID" 2>/dev/null; then
    # one last read: the frontend row may have landed in the same instant the
    # process exited, and a race that reports BACKEND_DIED for a healthy binary
    # would cost a lane a whole re-cut.
    sleep 1
    if grep -a -q -E "$SMOKE_FRONTEND_RE" "$LLOG" 2>/dev/null; then VERDICT=REACHED_FRONTEND
    else VERDICT=BACKEND_DIED; fi
    break
  fi
  if [ "$ELAPSED" -ge "$SMOKE_WALL" ]; then VERDICT=TIMEOUT; break; fi
  sleep "$SMOKE_POLL"
done
WALL=$(( $(date +%s) - S ))
cd / || true

# THE KILL COMES BEFORE THE WAIT.  On REACHED_FRONTEND and on TIMEOUT the lock
# is still running and still holds the node; `wait` first would block for the
# forty-three minutes the smoke exists to avoid.
if [ "$VERDICT" != BACKEND_DIED ]; then
  if [ -n "$LPGID" ]; then
    smoke_kill_pgid "$LPGID" || echo "### SMOKE WARNING: the lock's process group did not die -- report this, do not ignore it"
  else
    kill -TERM "$LPID" 2>/dev/null; sleep 2; kill -KILL "$LPID" 2>/dev/null
    echo "### SMOKE kill: degraded single-pid kill of $LPID (no private process group)"
  fi
fi
wait "$LPID" 2>/dev/null; LRC=$?

FE_ROWS=$(grep -a -c -E "$SMOKE_FRONTEND_RE" "$LLOG" 2>/dev/null); FE_ROWS=${FE_ROWS:-0}
echo "### SMOKE lock ended verdict=$VERDICT wall=${WALL}s lock_rc=$LRC frontend_rows=$FE_ROWS log_lines=$(wc -l < "$LLOG")"

case "$VERDICT" in
  REACHED_FRONTEND)
    echo "### SMOKE first frontend row:"
    grep -a -m1 -E "$SMOKE_FRONTEND_RE" "$LLOG" | cut -c1-300 | sed 's/^/### SMOKE   /'
    SMOKE_VERDICT=REACHED_FRONTEND; SMOKE_RC=0 ;;
  BACKEND_DIED)
    # the named deaths first, by name, then the first ERROR/panic line verbatim
    REASON=UNCLASSIFIED
    if grep -a -q -E 'directories\.rs|is out of bounds for string of length|18446744073709551615' "$LLOG" "$BLOG" 2>/dev/null; then
      REASON=PREFIX_PANIC_256
    elif grep -a -q -F 'the hermetic entry path is' "$LLOG" "$BLOG" 2>/dev/null; then
      REASON=PREFIX_REFUSAL
    fi
    echo "### SMOKE BACKEND_DIED reason=$REASON lock_rc=$LRC frontend_rows=0"
    if [ "$REASON" = PREFIX_PANIC_256 ]; then
      echo "### SMOKE   THIS IS DET-1-FIX-1 AND IT IS A PATH-LENGTH ACCIDENT, NOT A CODE DEFECT."
      echo "### SMOKE   rattler-build pads its build prefix to $SMOKE_PREFIX_PAD; the composed path overran it."
      echo "### SMOKE   Shorten the store root and re-smoke before touching the branch."
      grep -a -m1 -E 'is out of bounds for string of length' "$LLOG" "$BLOG" 2>/dev/null | cut -c1-300 | sed 's/^/### SMOKE   panic| /'
    fi
    echo "### SMOKE first ERROR/panic line (backend log first, then the lock log):"
    FIRST=$(grep -a -m1 -E "$SMOKE_ERROR_RE" "$BLOG" 2>/dev/null)
    [ -n "$FIRST" ] || FIRST=$(grep -a -m1 -E "$SMOKE_ERROR_RE" "$LLOG" 2>/dev/null)
    [ -n "$FIRST" ] || FIRST=$(tail -1 "$LLOG" 2>/dev/null)
    printf '### SMOKE   %s\n' "$(printf '%s' "$FIRST" | cut -c1-300)"
    echo "### SMOKE lock log tail:"
    tail -12 "$LLOG" | cut -c1-200 | sed 's/^/### SMOKE   /'
    SMOKE_VERDICT=BACKEND_DIED; SMOKE_RC=1 ;;
  TIMEOUT)
    echo "### SMOKE TIMEOUT: ${SMOKE_WALL}s elapsed with ZERO frontend rows and the lock still alive."
    echo "### SMOKE   A healthy binary reached the frontend in 199 s on a fully cold cache (measured, 5989192 arm 3)."
    echo "### SMOKE lock log tail:"
    tail -12 "$LLOG" | cut -c1-200 | sed 's/^/### SMOKE   /'
    SMOKE_VERDICT=TIMEOUT; SMOKE_RC=2 ;;
esac

echo "### SMOKE artifacts: $LLOG $BLOG $SHIM"
smoke_finish
