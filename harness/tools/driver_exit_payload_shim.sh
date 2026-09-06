#!/usr/bin/env bash
# driver_exit_payload_shim.sh -- HARNESS-EXIT-3. The per-wrapper PAYLOAD SEAM.
#
# WHY THIS EXISTS. HARNESS-EXIT-2's FAMILY B drove each listed `.sbatch` with a
# single `bash` shim on PATH. That works only for a wrapper whose payload is
# invoked AS `bash <driver>`. STORE-REAP-2 measured the consequence the moment
# it added two wrappers whose payloads are `cargo check` and a retread binary:
# the guard did not test them, it RAN THEM FOR REAL on the login node, and then
# scored them as swallowers, which they were not. A guard that executes the
# thing it is auditing is worse than no guard.
#
# WHAT THIS BUILDS. A world in which a wrapper can be run to completion and
# CANNOT reach a real payload, by four independent mechanisms, so that no single
# one of them is load-bearing:
#
#   1. FUNCTIONS. A bash function beats PATH always, including after a wrapper
#      does `export PATH="$CARGO_HOME/bin:$PATH"` -- which `sr2-work/check.sbatch`
#      really does, and which defeats a PATH-only shim. The preamble is handed
#      to the wrapper's own bash through BASH_ENV, so every nested `bash` gets
#      it too.
#   2. PATH. Set to shim:safebin:canary and nothing else, so a name we did not
#      stub is NOT FOUND rather than executed.
#   3. command_not_found_handle. A name that reaches PATH lookup and misses is
#      RECORDED and the run REFUSES (rc 99). An uncovered payload is a refusal,
#      never an execution -- that is the whole point of this file.
#   4. CANARY. The last PATH entry, and $CARGO_HOME/bin, hold stubs named for
#      the real payload commands which WRITE A FILE and exit 0. They are
#      unreachable while (1)-(3) hold; if any of them ever fires, the guard says
#      so and fails, and the file is the evidence that the seam leaked.
#
# Every stub RECORDS ITS ARGV to $DEG_REC/argv.log, so a run can prove what it
# intercepted rather than asserting it.
#
# CONTAINMENT is separate and additive: the caller runs the wrapper under
# `bwrap --tmpfs <task dir>` so a wrapper's ordinary redirections cannot
# overwrite another lane's artifacts. See driver_exit_guard.sh FAMILY B.
#
# usage:  . driver_exit_payload_shim.sh ; deg_build_shim <root-dir>
#         deg_run_wrapper <wrapper-abs-path> <per-run-record-dir>
#
# The payload injection code. 7, as in FAMILY 0.
DEG_INJECT=${DEG_INJECT:-7}

# Names whose invocation IS the payload: stubbed FAILING at $DEG_INJECT.
DEG_PAYLOAD_NAMES="bash sh cargo rustc pixi uv conda mamba micromamba
python python3 make cmake srun sbatch salloc env timeout nohup xargs
ssh scp docker apptainer singularity retread pixi-build-retread"

# Names that are not the payload but are DESTRUCTIVE or would block: recorded
# and neutralised at 0. `sleep` is here because two hold wrappers end on
# `sleep 43200` and a guard is not going to wait twelve hours to score them.
DEG_NEUTRAL_NAMES="rm mv rsync scancel scontrol squeue sacct sinfo git
curl wget sync sleep"

# Names passed through to the real tool: reading, printing, arithmetic. These
# cannot start a build, a solve, or a job.
DEG_SAFE_NAMES="cat echo printf sed awk grep egrep fgrep find sort uniq comm
head tail cut tr wc mkdir cp ln chmod touch stat readlink realpath dirname
basename mktemp date hostname sha256sum md5sum cmp diff column jq test [
true false seq expr bc nproc id uname tar gzip gunzip xz ls df getconf
paste join fold split tsort numfmt od xxd base64 cksum sha1sum sha512sum b2sum
nl rev tac shuf pr printenv which gawk mawk stdbuf tput expand unexpand
csplit pwd file"

# The names a canary stands in for: the real payloads. If one of these runs,
# something above leaked.
DEG_CANARY_NAMES="cargo pixi python python3 uv conda retread pixi-build-retread
srun sbatch rustc"

# Variables a wrapper uses to name a payload BINARY BY PATH. A PATH shim cannot
# reach those, so they are pointed at the payload stub instead. A path-position
# token naming any other variable is UNCOVERED and its wrapper is refused.
DEG_OVERRIDE_VARS="BIN PIXI RETREAD RETREAD_BIN PIXI_BIN BINARY EXE CARGO"

deg_build_shim () {   # $1 = root dir (created)
  local root=$1
  DEG_ROOT=$root
  DEG_SHIM=$root/shim; DEG_SAFEBIN=$root/safebin; DEG_CANARY=$root/canary
  DEG_HOME=$root/canaryhome
  mkdir -p "$DEG_SHIM" "$DEG_SAFEBIN" "$DEG_CANARY" "$DEG_HOME/bin" || return 4

  local n
  for n in $DEG_PAYLOAD_NAMES; do
    cat > "$DEG_SHIM/$n" <<'EOS'
#!/bin/sh
n=${0##*/}
printf 'PAYLOAD\t%s\t%s\n' "$n" "$*" >> "$DEG_REC/argv.log"
echo "### [payload shim] $n $* -- STUBBED FAILING rc=${DEG_INJECT:-7}"
exit ${DEG_INJECT:-7}
EOS
    chmod +x "$DEG_SHIM/$n"
  done

  # `bash` is the one payload stub with exceptions, and each is deliberate.
  # The drift check and the harness-commit reader are PRECONDITIONS, not
  # payloads: several wrappers refuse early on a dirty drift, and a refusal
  # that fires before the payload makes the broken and the fixed shape agree,
  # which is a vacuous fixture. HARNESS-EXIT-2 learned that with the drift
  # check; STORE-REAP-2's `sr2-work/gate.sbatch` adds the resolver, which is
  # read into `HC=$(bash ...)` and gates on being non-empty.
  cat > "$DEG_SHIM/bash" <<'EOS'
#!/bin/sh
printf 'PAYLOAD\tbash\t%s\n' "$*" >> "$DEG_REC/argv.log"
for a in "$@"; do
  case "$a" in
    *harness_drift_check.sh)
      echo "### [guard shim] drift check stubbed CLEAN"; exit 0;;
    *harness_commit_resolve.sh)
      echo "${DEG_FAKE_COMMIT:-0000000}"; exit 0;;
  esac
done
echo "### [payload shim] bash $* -- STUBBED FAILING rc=${DEG_INJECT:-7}"
exit ${DEG_INJECT:-7}
EOS
  chmod +x "$DEG_SHIM/bash"

  for n in $DEG_NEUTRAL_NAMES; do
    cat > "$DEG_SHIM/$n" <<'EOS'
#!/bin/sh
n=${0##*/}
printf 'NEUTRAL\t%s\t%s\n' "$n" "$*" >> "$DEG_REC/argv.log"
exit 0
EOS
    chmod +x "$DEG_SHIM/$n"
  done

  # `tee` passes stdin through but writes NO file: a wrapper's log tee must not
  # land on another lane's artifact even inside the sandbox.
  cat > "$DEG_SHIM/tee" <<'EOS'
#!/bin/sh
printf 'NEUTRAL\ttee\t%s\n' "$*" >> "$DEG_REC/argv.log"
exec cat
EOS
  chmod +x "$DEG_SHIM/tee"

  # safebin: the real tool, by symlink, resolved from the GUARD's PATH.
  local p
  for n in $DEG_SAFE_NAMES; do
    p=$(command -v "$n" 2>/dev/null) || continue
    [ -n "$p" ] && ln -sf "$p" "$DEG_SAFEBIN/$n"
  done

  for n in $DEG_CANARY_NAMES; do
    cat > "$DEG_CANARY/$n" <<'EOS'
#!/bin/sh
n=${0##*/}
: > "$DEG_REC/CANARY_FIRED.$n"
printf 'CANARY\t%s\t%s\n' "$n" "$*" >> "$DEG_REC/argv.log"
echo "### [CANARY] $n reached PATH lookup -- the payload seam LEAKED" >&2
exit 0
EOS
    chmod +x "$DEG_CANARY/$n"
    cp -f "$DEG_CANARY/$n" "$DEG_HOME/bin/$n"
  done

  # the preamble: BASH_ENV, read by every non-interactive bash we start.
  {
    echo '# driver_exit_payload_shim preamble -- functions beat PATH, always.'
    for n in $DEG_PAYLOAD_NAMES $DEG_NEUTRAL_NAMES tee; do
      printf '%s () { "%s/%s" "$@"; }\n' "$n" "$DEG_SHIM" "$n"
    done
    cat <<'EOS'
command_not_found_handle () {
  printf 'UNCOVERED\t%s\n' "$1" >> "$DEG_REC/uncovered.log"
  echo "### [guard] UNCOVERED PAYLOAD: $1 -- refusing rather than executing" >&2
  exit 99
}
EOS
  } > "$root/preamble.sh"

  deg_drop_stubs
  printf '%s\n' "$root/preamble.sh"
}

# deg_shim_env <record-dir> -- echoes the env assignments for one wrapper run.
deg_shim_env () {
  local rec=$1 v
  printf '%s\n' \
    "PATH=$DEG_SHIM:$DEG_SAFEBIN:$DEG_CANARY" \
    "BASH_ENV=$DEG_ROOT/preamble.sh" \
    "DEG_REC=$rec" \
    "DEG_INJECT=$DEG_INJECT" \
    "DEG_FAKE_COMMIT=0000000" \
    "CARGO_HOME=$DEG_HOME" \
    "RUSTUP_HOME=$DEG_HOME" \
    "TMPDIR=$rec/tmp"
  for v in $DEG_OVERRIDE_VARS; do printf '%s=%s\n' "$v" "$DEG_SHIM/pixi-build-retread"; done
}

# DEG_DROP_STUBS -- the MUTATION ARM for the seam itself. A guard whose shim
# silently lacks a stub would run that payload for real; naming a stub here
# deletes it from the shim dir AND from the preamble, and the wrapper that
# needs it must then come back as an UNCOVERED REFUSAL rather than as a score.
# A seam that keeps scoring with a stub removed is not covering that payload.
deg_drop_stubs () {
  local n
  for n in ${DEG_DROP_STUBS:-}; do
    rm -f "$DEG_SHIM/$n" "$DEG_CANARY/$n" "$DEG_HOME/bin/$n"
    grep -v "^$n () {" "$DEG_ROOT/preamble.sh" > "$DEG_ROOT/preamble.tmp" \
      && mv "$DEG_ROOT/preamble.tmp" "$DEG_ROOT/preamble.sh"
    echo "### [seam mutation] stub '$n' REMOVED from the shim dir and the preamble"
  done
}
