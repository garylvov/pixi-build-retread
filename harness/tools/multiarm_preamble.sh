#!/usr/bin/env bash
# multiarm_preamble.sh -- THE VERSIONED OPENING OF EVERY MULTI-ARM PROOF DRIVER.
# PROOF-SMOKE-1.
#
# Same reason `mutation_arms.sh`, `multiarm_store_reap.sh` and
# `phase_template/lane_wrapper.sbatch` exist: FOURTEEN task-only multi-arm
# drivers today (the thirteen ORDER-1-1 named plus `det1-work/det1_proof2.sh`)
# open with the same six checks, every one of them descended from the one before
# it BY COPY, and `agrescap/tasks` is not a git repository (CLAUDE.md law 7).  A
# fix in one copy is stranded in that copy -- which is exactly how DET-1-4-1's
# `BACKEND` bug survived into a proof, and how DET-1-FIX burned three cold arms
# on a path-length accident a forty-second check would have named.
#
#   usage, at the very top of a lane's driver, AFTER its own constants:
#
#     source "$T/tools/multiarm_preamble.sh"
#     MULTIARM_TAG=DET1F
#     MULTIARM_JOB=$J
#     MULTIARM_TASK=$T
#     MULTIARM_JOB_ROOT=$D                  # the dir holding HARNESS_COMMIT
#     MULTIARM_CACHE_ROOT=$C                # this job's ONE cache root
#     MULTIARM_ARMS=4
#     MULTIARM_WITNESS="bench: hermetic_provision"
#     MULTIARM_BINARIES=( "FIX=$SNAP_FIX" "CTL=$SNAP_CTL" )
#     MULTIARM_SMOKE_MANIFEST=$CANON_MANIFEST
#     multiarm_preamble || exit $?
#
#   and, inside the arm loop:
#
#     WS=$(multiarm_arm_ws "$AN")           # ws.<TAG>-<jid>/a<N>
#     export XDG_CACHE_HOME=$(multiarm_arm_cache_home "$AN")
#     multiarm_write_shim "$AN" "$SNAP" "$SHIM" "$BLOG" || return 21
#
# ── THE SIX THINGS IT DOES, AND WHAT EACH ONE COST WHEN IT WAS MISSING ───────
#
# 1. PIN AND DRIFT.  `harness_commit_resolve.sh` then `harness_drift_check.sh`.
#    det1_proof.sh -- the campaign's longest-running multi-arm driver -- carried
#    NEITHER, so it ran whatever the task dir happened to hold.
#
# 2. THE `strings` WITNESS TABLE, BEFORE ANY ARM.  The two binaries are
#    separated BY THEIR BYTES, not by the arm plan's labels: a fix-only string
#    the lane NAMES must be present in every FIX binsnap and ABSENT from every
#    CTL binsnap, or the whole job refuses.  DET-1-4-1: a driver bound `BACKEND`
#    ONCE before the arm loop while the loop rebound `SNAP` per arm, so all
#    three shims execed the FIX binary and the job reported `ctl_rows=71` for a
#    row `strings` counts ZERO times in the control binary.  A witness table
#    makes that job refuse instead of publishing.
#
# 3. PER-ARM SHIMS FROM ONE TEMPLATE, READ BACK.  The shim interpolates THE
#    ARM'S binary and the written file is then re-read: exactly one line may
#    exec a `pixi-build-retread` path and it must be this arm's.  The matcher is
#    THE BACKEND EXEC, not "the first line starting with exec" -- the shim's
#    first such line is the stderr redirect, and a reader that got that wrong
#    aborted all four arms of `det1f-proof` 5988493 with rc=21.  A reader that
#    refuses a correct shim is the same class of defect as one that accepts a
#    wrong one.
#
# 4. THE COMPOSED-PREFIX LENGTH CHECK (DET-1-FIX-1).  `rattler-build debug
#    setup` pads its build prefix to a fixed 256 bytes; the hermetic store entry
#    path plus 92 fixed bytes is what gets padded.  MEASURED: 164+92 = 256
#    provisioned and locked; 168+92 = 260 underflowed `256 - len` and panicked
#    in `rattler_build_core/src/types/directories.rs` before ANY FIX arm
#    provisioned.  The parent proof sat EXACTLY on the boundary and nobody knew.
#    Every arm's store root is measured here, before the job stages anything.
#
# 5. ARM WORKSPACE NAMING.  `<base>/ws.<TAG>-<jid>/a<N>` -- one per-job PARENT,
#    not four job-named siblings in the shared `retread/` directory.  That is
#    ORDER-1-1 and it is not tidying: `multiarm_store_reap.sh` refuses any path
#    that is not STRICTLY under a root the job DECLARES as its own, proved by a
#    dev:inode walk through `..`, and "the shared retread directory" is not a
#    root this job may declare.  The flat form `ws.<TAG>-<jid>-a<N>` is kept as
#    the arm's printed IDENTITY (`multiarm_arm_id`) so log readers and every
#    existing grep still find it, while the PATH stays inside a root the reaper
#    will accept.
#
# 6. A SMOKE LOCK FOR EVERY DISTINCT BINARY, BEFORE ARM 1.  This is the whole
#    point (tick 519).  `det1f-proof2` **5989192** committed three hours and
#    four cold arms to a candidate that dies as a build backend in 44 seconds:
#    its gate was green at 1868/0/21 and its three FIX arms produced
#    `frontend=0 prep_metadata=0 lock_sha=NONE` while the CTL arm locked rc=0 in
#    2566 s on the same node in the same hour.  A GREEN GATE IS NOT A WORKING
#    BACKEND.  `proof_smoke.sh` asks the cheap question first, and if any
#    distinct binary does not reach the frontend the WHOLE JOB refuses:
#    `MULTIARM_JOB_FATAL=1`, a non-zero return, and every root this job created
#    handed to the `afterany` cleanup owner rather than deleted in-job.
#
# ── THE ONE OFF SWITCH, AND IT IS LOUD ───────────────────────────────────────
# `MULTIARM_SMOKE=0` skips step 6 and is REFUSED unless `MULTIARM_SMOKE_REASON`
# is also set; the reason is printed in the header and again in the summary, so
# it lands in the job's own log and in whatever a reader greps.  A silent skip
# would make this file decorative within two lanes.
#
# Sourcing this file defines functions and touches nothing.  `multiarm_preamble`
# is what acts.
set -uo pipefail

MULTIARM_PREAMBLE_VERSION=1

MULTIARM_WS_BASE=${MULTIARM_WS_BASE:-/oscar/data/stellex/glvov/retread}
MULTIARM_SMOKE=${MULTIARM_SMOKE:-1}
MULTIARM_SMOKE_REASON=${MULTIARM_SMOKE_REASON:-}
MULTIARM_SMOKE_WALL=${MULTIARM_SMOKE_WALL:-900}
MULTIARM_JOB_FATAL=0
MULTIARM_SMOKE_ROOTS=()

multiarm_say () { echo "### PREAMBLE $*"; }

multiarm_fatal () {
  MULTIARM_JOB_FATAL=1
  echo "### PREAMBLE FATAL: $*"
}

# ---- 1. pin and drift -------------------------------------------------------
multiarm_pin_and_drift () {
  local hc rc drc
  local resolve=$MULTIARM_TASK/tools/harness_commit_resolve.sh
  local drift=$MULTIARM_TASK/tools/harness_drift_check.sh
  [ -f "$resolve" ] || { multiarm_fatal "no harness_commit_resolve.sh at $resolve"; return 6; }
  [ -f "$drift" ]   || { multiarm_fatal "no harness_drift_check.sh at $drift"; return 6; }
  hc=$(bash "$resolve" "$MULTIARM_JOB_ROOT"); rc=$?
  if [ "$rc" -ne 0 ]; then
    multiarm_fatal "harness commit pin REFUSED rc=$rc (rc 4 = the pin is not the commit the task copies ARE -- run: bash tools/harness_commit_resolve.sh --write $MULTIARM_JOB_ROOT)"
    return 6
  fi
  [ -n "$hc" ] || { multiarm_fatal "no harness pin resolved for $MULTIARM_JOB_ROOT"; return 6; }
  multiarm_say "HARNESS_COMMIT=$hc"
  bash "$drift" "$hc" 2>&1 | tail -3
  drc=${PIPESTATUS[0]}
  multiarm_say "DRIFT_RC=$drc"
  [ "$drc" -eq 0 ] || { multiarm_fatal "the task-dir harness is not $hc"; return 6; }
  MULTIARM_HARNESS_COMMIT=$hc
  return 0
}

# ---- 2. the strings witness table ------------------------------------------
# Every entry of MULTIARM_BINARIES is `<ROLE>=<path>`.  A role beginning FIX
# must carry the witness at least once; a role beginning CTL must carry it zero
# times.  Any other role is refused rather than silently unchecked.
multiarm_witness_table () {
  local ent role path n bin sha ok=1
  local -A seen=()
  [ -n "${MULTIARM_WITNESS:-}" ] || { multiarm_fatal "MULTIARM_WITNESS is unset -- a witness table with no string is vacuous"; return 9; }
  [ "${#MULTIARM_BINARIES[@]}" -gt 0 ] || { multiarm_fatal "MULTIARM_BINARIES is empty"; return 9; }
  multiarm_say "WITNESS TABLE for '$MULTIARM_WITNESS'"
  printf '### PREAMBLE   %-6s %-12s %-8s %s\n' ROLE COUNT WANT BINARY
  for ent in "${MULTIARM_BINARIES[@]}"; do
    role=${ent%%=*}; path=${ent#*=}
    bin=$path; [ -d "$bin" ] && bin=$bin/pixi-build-retread
    if [ ! -x "$bin" ]; then multiarm_fatal "role $role: no executable binary at $bin"; ok=0; continue; fi
    sha=$(sha256sum "$bin" | awk '{print $1}')
    n=$(strings -a "$bin" 2>/dev/null | grep -c -F "$MULTIARM_WITNESS")
    n=${n:-0}
    case "$role" in
      FIX*) printf '### PREAMBLE   %-6s %-12s %-8s %s sha256=%s\n' "$role" "$n" '>=1' "$bin" "$sha"
            if [ "$n" -lt 1 ]; then
              multiarm_fatal "role $role does not carry the witness string -- either the wrong binsnap is pinned or the branch lost the row, and every CTL assertion would then be vacuous"
              ok=0
            fi ;;
      CTL*) printf '### PREAMBLE   %-6s %-12s %-8s %s sha256=%s\n' "$role" "$n" '0' "$bin" "$sha"
            if [ "$n" -ne 0 ]; then
              multiarm_fatal "role $role CARRIES the witness string -- it is not a control for this row and the negative control would be vacuous"
              ok=0
            fi ;;
      *)    multiarm_fatal "role '$role' is neither FIX* nor CTL* -- this table would not check it, and an unchecked binary in a witness table is worse than no table"
            ok=0 ;;
    esac
    if [ -n "${seen[$sha]:-}" ] && [ "${seen[$sha]}" != "$role" ]; then
      multiarm_fatal "roles ${seen[$sha]} and $role are THE SAME BYTES (sha256=$sha) -- a control whose arms run the same binary is not a control"
      ok=0
    fi
    seen[$sha]=$role
  done
  [ "$ok" = 1 ]
}

# ---- 3. the per-arm shim, written from ONE template and read back -----------
# usage: multiarm_write_shim <arm> <binary> <shim path> <backend log>
multiarm_write_shim () {
  local an=$1 bin=$2 shim=$3 blog=$4 n line
  [ -d "$bin" ] && bin=$bin/pixi-build-retread
  [ -x "$bin" ] || { echo "### ARM $an FATAL: no executable backend at $bin"; return 21; }
  mkdir -p "$(dirname "$shim")" "$(dirname "$blog")" || return 21
  cat > "$shim" <<SHIMEOF
#!/usr/bin/env bash
unset RUST_LOG
exec 2> >(tee -a "$blog" >&2)
exec "$bin" "\$@"
SHIMEOF
  chmod +x "$shim" || return 21
  # THE MATCHER IS THE BACKEND EXEC, NOT "THE FIRST LINE STARTING WITH exec".
  n=$(grep -c -E '^exec "[^"]*/pixi-build-retread" "\$@"$' "$shim")
  line=$(grep -m1 -E '^exec "[^"]*/pixi-build-retread" "\$@"$' "$shim")
  echo "### ARM $an SHIM backend exec lines=$n line: ${line:-<none>}"
  if [ "$n" -ne 1 ]; then
    echo "### ARM $an FATAL: the backend shim has $n lines that exec a pixi-build-retread path, want exactly 1."
    sed 's/^/### ARM '"$an"'   shim| /' "$shim"
    return 21
  fi
  if [ "$line" != "exec \"$bin\" \"\$@\"" ]; then
    echo "### ARM $an FATAL: the backend shim does not exec this arm's binary."
    echo "### ARM $an   want: exec \"$bin\" \"\$@\""
    echo "### ARM $an   got:  $line"
    echo "### ARM $an   This is DET-1-4-1 exactly: a driver shipped three shims all naming the FIX binary and reported the third as a control."
    return 21
  fi
  echo "### ARM $an SHIM ok: it execs THIS arm's binary ($bin)"
  return 0
}

# ---- 4. the composed-prefix budget -----------------------------------------
# One implementation, in proof_smoke.sh, sourced here.  Two copies of a length
# rule is how the rule drifts.
multiarm_prefix_check () {
  local root=$1 label=${2:-store root}
  smoke_prefix_budget "$root" "$label"
}

multiarm_prefix_check_all () {
  local n ok=1 home
  for n in $(seq 1 "${MULTIARM_ARMS:-1}"); do
    home=$(multiarm_arm_cache_home "$n")
    multiarm_prefix_check "$home" "arm $n cache home" || { multiarm_fatal "arm $n's store root composes past the 256-byte pad -- SHORTEN THE ROOT, NOT THE CHECK"; ok=0; }
  done
  [ "$ok" = 1 ]
}

# ---- 5. names --------------------------------------------------------------
multiarm_ws_root ()       { printf '%s/ws.%s-%s' "$MULTIARM_WS_BASE" "$MULTIARM_TAG" "$MULTIARM_JOB"; }
multiarm_arm_ws ()        { printf '%s/a%s' "$(multiarm_ws_root)" "$1"; }
multiarm_arm_id ()        { printf 'ws.%s-%s-a%s' "$MULTIARM_TAG" "$MULTIARM_JOB" "$1"; }
# SHORT ON PURPOSE (see 4): `x<N>` is unique per arm and fourteen characters
# shorter than the `xdg-cache-<scope>` name that panicked.
multiarm_arm_cache_home () { printf '%s/x%s' "$MULTIARM_CACHE_ROOT" "$1"; }

# ---- 6. the smoke, for every DISTINCT binary, before arm 1 -----------------
multiarm_smoke_all () {
  local ent role path bin sha rc ok=1 out
  local -A done=()
  local smoke=$MULTIARM_TASK/tools/proof_smoke.sh
  if [ "$MULTIARM_SMOKE" != 1 ]; then
    if [ -z "$MULTIARM_SMOKE_REASON" ]; then
      multiarm_fatal "MULTIARM_SMOKE=0 with no MULTIARM_SMOKE_REASON -- a silent skip makes this preamble decorative"
      return 9
    fi
    multiarm_say "SMOKE SKIPPED ON PURPOSE. reason=$MULTIARM_SMOKE_REASON"
    return 0
  fi
  [ -f "$smoke" ] || { multiarm_fatal "no proof_smoke.sh at $smoke"; return 9; }
  [ -n "${MULTIARM_SMOKE_MANIFEST:-}" ] || { multiarm_fatal "MULTIARM_SMOKE_MANIFEST is unset"; return 9; }
  for ent in "${MULTIARM_BINARIES[@]}"; do
    role=${ent%%=*}; path=${ent#*=}
    bin=$path; [ -d "$bin" ] && bin=$bin/pixi-build-retread
    sha=$(sha256sum "$bin" 2>/dev/null | awk '{print $1}')
    [ -n "$sha" ] || { multiarm_fatal "role $role: cannot hash $bin"; ok=0; continue; }
    if [ -n "${done[$sha]:-}" ]; then
      multiarm_say "SMOKE $role: same bytes as ${done[$sha]} (sha256=$sha) -- one smoke per DISTINCT binary, already done"
      continue
    fi
    done[$sha]=$role
    multiarm_say "SMOKE $role $bin -- ONE bounded lock, wall cap ${MULTIARM_SMOKE_WALL}s, before this job commits to its arms"
    # SMOKE_WS is deliberately shared across the smokes of one job: staging is
    # the expensive half and the question each smoke asks is about the BINARY.
    # The caches are shared for the same reason -- the first smoke pays the cold
    # repodata fetch, the rest do not.
    SMOKE_WALL=$MULTIARM_SMOKE_WALL \
    SMOKE_ROOT=${SMOKE_ROOT:-$MULTIARM_CACHE_ROOT/smk} \
    SMOKE_WS=${SMOKE_WS:-$MULTIARM_CACHE_ROOT/smk/w} \
    SMOKE_CACHE=${SMOKE_CACHE:-$MULTIARM_CACHE_ROOT/smk/c} \
      bash "$smoke" "$bin" "$MULTIARM_SMOKE_MANIFEST" "$MULTIARM_JOB_ROOT"
    rc=$?
    if [ "$rc" -ne 0 ]; then
      multiarm_fatal "SMOKE $role rc=$rc -- this binary did not reach the frontend, so this job will NOT spend its arms on it. Read the '### SMOKE' rows above for the verdict and the quoted first error."
      ok=0
    else
      multiarm_say "SMOKE $role REACHED_FRONTEND"
    fi
  done
  MULTIARM_SMOKE_ROOTS+=("$MULTIARM_CACHE_ROOT/smk")
  [ "$ok" = 1 ]
}

# ---- the whole preamble -----------------------------------------------------
multiarm_preamble () {
  local v rc=0
  for v in MULTIARM_TAG MULTIARM_JOB MULTIARM_TASK MULTIARM_JOB_ROOT \
           MULTIARM_CACHE_ROOT MULTIARM_ARMS MULTIARM_WITNESS MULTIARM_SMOKE_MANIFEST; do
    if [ -z "${!v:-}" ]; then
      multiarm_fatal "$v is unset -- the preamble refuses rather than guessing"
      return 9
    fi
  done
  echo "################################################################################"
  multiarm_say "multiarm_preamble.sh v$MULTIARM_PREAMBLE_VERSION  tag=$MULTIARM_TAG job=$MULTIARM_JOB arms=$MULTIARM_ARMS  $(date -Is)"
  multiarm_say "ws root  : $(multiarm_ws_root)   (arm N -> $(multiarm_arm_ws N), id $(multiarm_arm_id N))"
  multiarm_say "cache root: $MULTIARM_CACHE_ROOT"
  echo "################################################################################"

  # proof_smoke.sh is SOURCED for its length rule (step 4) before it is RUN for
  # its smoke (step 6).  Sourcing it defines functions and runs nothing: it
  # reaches its argument parse only when it has arguments, and it is sourced
  # with none, so the usage branch would exit -- hence the subshell-free guard
  # below, which sets the sentinel the script honours.
  if ! declare -F smoke_prefix_budget >/dev/null 2>&1; then
    PROOF_SMOKE_LIB=1 . "$MULTIARM_TASK/tools/proof_smoke.sh" || {
      multiarm_fatal "could not source proof_smoke.sh for its length rule"; return 9; }
  fi

  multiarm_pin_and_drift    || rc=$?
  multiarm_witness_table    || rc=$?
  multiarm_prefix_check_all || rc=$?
  multiarm_smoke_all        || rc=$?

  if [ "$MULTIARM_JOB_FATAL" != 0 ]; then
    echo "################################################################################"
    multiarm_say "JOB REFUSED BEFORE ARM 1. MULTIARM_JOB_FATAL=1"
    multiarm_say "Nothing this job created is deleted here: every root below is handed to the afterany cleanup owner."
    local r
    for r in "${MULTIARM_SMOKE_ROOTS[@]+"${MULTIARM_SMOKE_ROOTS[@]}"}" "$MULTIARM_CACHE_ROOT" "$(multiarm_ws_root)"; do
      [ -n "$r" ] && echo "### PREAMBLE owed root: $r"
    done
    # If the lane has already sourced multiarm_store_reap.sh and run reap_init,
    # its own handoff file is written too -- that is the file the afterany owner
    # reads.  Absent it, the rows above are the record.
    if declare -F reap_handoff_list >/dev/null 2>&1; then
      reap_handoff_list "$MULTIARM_JOB_ROOT/artifacts/${MULTIARM_TAG}-${MULTIARM_JOB}.reap-owed.txt" || true
    fi
    echo "################################################################################"
    return "${rc:-9}"
  fi
  multiarm_say "PREAMBLE CLEAN: pin+drift ok, witness table ok, prefix budget ok for $MULTIARM_ARMS arm(s), every distinct binary REACHED_FRONTEND."
  return 0
}
