#!/usr/bin/env bash
# env_seed.sh -- THE ONE AUTHORITY for putting the pinned interpreter hash seed
# into the shell that launches pixi.  SOURCED, never executed: it defines
# `ENV_SEED_MARKER` and `env_seed_export` and runs nothing.
#
#     . <this file>
#     env_seed_export "$BACKEND"            # STRICT   -- a verb-less binary is refused
#     env_seed_export "$BIN" optional       # OPTIONAL -- a verb-less binary runs unseeded
#
# ── WHY THE SEED IS EXPORTED BY THE CALLER AND NOT SET BY THE BACKEND ────────
# DET-1-4-1 (job 6001140, node1802) locked one manifest three times on one node
# with ONE binary -- binsnaps/cand-3f2095a -- varying nothing but this variable
# in the launching shell:
#
#   arm 1  PYTHONHASHSEED=0   gym requires_dist md5 e569ebf5...  \  cmp rc=0
#   arm 2  PYTHONHASHSEED=0   gym requires_dist md5 e569ebf5...  /  RAW 0 SORTED 0
#   arm 3  unset              gym requires_dist md5 bd63668b...     RAW 28 SORTED 0
#
# A raw delta of 28 with a SORTED delta of 0 is a pure reordering: no package
# moved, no count changed (`moved_row_halves.sh` rc=0, moved=0, on all three
# pairs).  That binary ALREADY calls `apply_reproducible_python_hash_seed` on
# all three of its own doors, and the block moved anyway -- because the process
# that builds `gym` 0.26.2's `requires_dist` is an in-process PEP 517 child of
# the pixi FRONTEND, spawned by pixi's embedded uv, which the backend never
# execs and therefore cannot pin.  The only channel that reaches it is the
# environment pixi itself was launched with.  That is this export.
#
# ── WHY THIS IS A FILE AND NOT A BLOCK IN phaseN_relock.sh (DET-1-6-1) ───────
# 58717bd put `env_seed_export` inside `phase_template/phaseN_relock.sh`, where
# the ARMS reach it and NOTHING ELSE DOES.  det16-proof 6013332 then paid the
# whole price of that: its preamble smoked the FIX binary
# `binsnaps/cand-c0ccc0d` through `tools/proof_smoke.sh`, which launches its own
# `pixi lock` and had never heard of the export, so c0ccc0d's new `preflight()`
# refused at second zero and the job was REFUSED BEFORE ARM 1 -- after paying
# 879 s to publish the stage mirror.  The first line of that job's backend log
# is this, verbatim:
#
#     preflight: $PYTHONHASHSEED must be `0` in the environment that launched pixi
#
# A wrapper and a smoke that disagree about the environment pixi runs in ask
# different questions, and the cheap one (the smoke) then refuses the expensive
# one's binary for a reason that is about the smoke.  One authority, sourced by
# both, is the root fix.  Readers: `phase_template/env_seed_export_guard.sh`
# (the function, every arm) and `tools/proof_smoke_guard.sh` arms V1-V4 (the
# smoke's call site, against real binaries).
#
# ── THE TWO MODES, AND WHY `optional` IS NOT A LOOPHOLE ─────────────────────
# STRICT is the relock wrapper's mode: a lane that means to certify a lock must
# not lock under a random seed, and a binary that cannot state the seed is a
# binary this wrapper will not guess for.
#
# OPTIONAL is the SMOKE's mode, and the reason is the control arms.  Every
# known-good control in the tree is a PRE-FIX binary -- `integration-569b0ac` is
# the one `proof_smoke_guard.sh` arms A and N7 run -- and those binaries carry
# NO `env-seed` verb and NO `preflight` to satisfy.  Refusing them would refuse
# every control arm and leave the smoke able to smoke only the binary under
# test, which is the opposite of a control.  So `optional` relaxes EXACTLY ONE
# case, the verb's ABSENCE, and says so on its own row.  A binary that HAS the
# verb and then answers badly (non-zero, or empty) is still refused in both
# modes: that is a defect in the binary, not an old binary.
#
# ── THE VALUE IS ASKED OF THE BINARY, NEVER TYPED HERE ───────────────────────
# A literal `0` in this file would be a second authority for the seed, free to
# drift from the Rust constant the backend's preflight compares against; the
# first time they disagreed, every lock would refuse and the two `0`s would both
# look right.  `retread env-seed` prints `uv_closure::REPRODUCIBLE_PYTHON_HASH_
# SEED` and nothing else, and it is handled before the preflight precisely so it
# can be called to SATISFY that preflight.
#
# ── THE MARKER, AND WHY THE VERB IS NEVER PROBED BY RUNNING IT ───────────────
# `main.rs` handles `env-seed` at the TOP of `main`; a binary built BEFORE
# fix/det1-env-seed matches no verb, falls through to the automatic preflight
# and STARTS THE JSON-RPC TRANSPORT -- so `<old binary> env-seed` inside `$(...)`
# does not fail, it BLOCKS on stdin, and the caller would hang at second zero
# with no output rather than decide.  The detection is therefore STATIC, the
# same shape and for the same reason as `store_reap_census.sh`'s: a string that
# exists only in a binary carrying the verb, and nothing is executed until it is
# found.  Measured, not assumed: `grep -a -c -F` of this marker is 1 in
# `binsnaps/cand-c0ccc0d` (the verb's own binsnap) and 0 in
# `binsnaps/cand-3f2095a` (its parent).
ENV_SEED_MARKER='retread env-seed: writing stdout: '
env_seed_export () {             # $1 = backend binary; $2 = strict|optional
  local bin=${1:-} mode=${2:-strict} seed
  if [ -z "$bin" ] || [ ! -x "$bin" ]; then
    echo "### FATAL ENV SEED: backend '${bin:-<unset>}' is not an executable binary."
    echo "###        ACTUATOR: point the backend path at the binsnap this run is meant to use."
    return 15
  fi
  if ! grep -a -q -F -- "$ENV_SEED_MARKER" "$bin"; then
    if [ "$mode" = optional ]; then
      # NOT a silent skip: the row says the binary was asked, what the answer
      # was, and what the ambient seed therefore is, so a reader can tell an
      # unseeded CONTROL from an unseeded accident.
      echo "### ENV SEED not-applicable binary lacks verb: $bin"
      echo "###        This binary predates fix/det1-env-seed: no \`env-seed\` verb and no"
      echo "###        preflight to satisfy, so it runs with the ambient seed"
      echo "###        PYTHONHASHSEED='${PYTHONHASHSEED:-<absent from the environment>}'."
      echo "###        Refusing here would refuse every known-good control arm."
      return 0
    fi
    echo "### FATAL ENV SEED: $bin does NOT carry the \`env-seed\` verb, and this"
    echo "###        caller will not guess the seed on its behalf -- a literal here"
    echo "###        is a second authority that drifts from the backend's constant."
    echo "###        It is also NOT probed by running it: an old binary answers an"
    echo "###        unknown verb by starting the JSON-RPC transport and blocking on"
    echo "###        stdin, so this refusal is what stops a silent hang."
    echo "###        ACTUATOR: rebuild/point at a binsnap at or after fix/det1-env-seed."
    return 15
  fi
  if ! seed=$("$bin" env-seed); then
    echo "### FATAL ENV SEED: \`$bin env-seed\` exited non-zero; refusing to lock"
    echo "###        without a pinned interpreter hash seed -- an unset seed reorders"
    echo "###        requires_dist lines and the lock's bytes stop being a function"
    echo "###        of its resolution (DET-1-4-1, job 6001140)."
    echo "###        ACTUATOR: run \`$bin env-seed\` by hand and fix what it reports."
    return 15
  fi
  if [ -z "$seed" ]; then
    echo "### FATAL ENV SEED: \`$bin env-seed\` printed nothing. Exporting an EMPTY"
    echo "###        PYTHONHASHSEED is not the same as exporting the pin -- CPython"
    echo "###        treats empty as unset, i.e. random, so it would look like a pin"
    echo "###        and behave like the defect. Refusing."
    echo "###        ACTUATOR: run \`$bin env-seed\` by hand and fix what it reports."
    return 15
  fi
  export PYTHONHASHSEED=$seed
  echo "### ENV SEED exported PYTHONHASHSEED=$seed source=$bin env-seed"
}
