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
# marker separates those two outcomes and ARRIVES EARLY, and a smoke does not
# have to finish a lock to answer the question.  BUT A FRONTEND ROW ALONE IS NOT
# ENOUGH -- 5994177 arm C scored one in four seconds with a backend that only
# slept, because the `cpu` environment has no source at all -- so the verdict
# needs BOTH a frontend row AND a `conda/outputs` row in the BACKEND log (14 on
# the known-good binary, 0 on the sleeping stub, 0 on the dying stub).  The
# moment both appear this script KILLS the lock and returns REACHED_FRONTEND, so
# the bounded wall is a ceiling a healthy binary never reaches, not a budget it
# spends.
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
# (1) THE PREFIX PANIC (DET-1-FIX-1).  `rattler-build debug setup` pads its host
#     prefix to a fixed target and slices a placeholder string by `pad - len`;
#     when the composed path is longer than the pad that subtraction underflows
#     in usize and it panics in
#     `rattler_build_core/src/types/directories.rs`.  This script reports that
#     death as `reason=PREFIX_PANIC_256`, so a lane never again reads a
#     path-length accident as a code defect.
#
#     READ THE PANIC CORRECTLY, because this campaign misread it once and sized
#     a fix to the wrong number.  `end byte index 18446744073709551591 is out of
#     bounds for string of length 260`: the **260 IS THE PLACEHOLDER**, built by
#     repeating a ten-byte template until it clears the pad -- 26 * 10 = 260 for
#     every input, always, and it says NOTHING about any path.  The path is in
#     the **INDEX**: `2^64 - 18446744073709551591 = 25` is how many bytes the
#     composed prefix overran the pad by.  PROOF-SMOKE-1-5 measured that against
#     two roots eight bytes apart (job 6004714) and got deficits of exactly 25
#     and 33, which is what makes the boundary a measurement and not an opinion.
#
# (2) THE PRE-LOCK REFUSAL DET-1-FIX-1 ADDED, AND THE ROOT IT WAS MISSING.  The
#     same check, run BEFORE the lock: this script composes the longest hermetic
#     entry path its store root would produce and refuses with
#     `reason=PREFIX_REFUSAL`, because a smoke run from a root too long to
#     provision would report BACKEND_DIED against a perfectly good binary.  A
#     guard that blames the wrong thing is worse than no guard.  It is a
#     SETUP_FAILED, not a verdict on the binary -- rc 3.
#
#     PROOF-SMOKE-1-5: it modelled ONE of the TWO store roots a binary can
#     reach, and the one it modelled was not the one that panicked.  See the
#     note over `smoke_prefix_entry`.  The refusal now composes both and judges
#     the worse; and because the fast-tmp candidate cannot fit under any root on
#     this filesystem, the smoke disengages fast-tmp and says so, which is what
#     turns arm A from a coin flip back into a verdict about the binary.
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
# THE PINNED uv, AND IT IS NOT OPTIONAL.  MEASURED, on this guard's own first
# real run (`psmoke-guards` 5993691 arm A): with the ambient uv on PATH the
# KNOWN-GOOD binsnap `integration-569b0ac` died in 3 s and the smoke reported
# BACKEND_DIED against a binary that is fine.  The backend log said why, twenty
# times over: `preflight: uv version mismatch  wanted: 0.12.5  got: 0.11.29`.
# retread's `uv_closure::REQUIRED_UV` refuses any other uv, and every driver
# satisfies it from the harness's own pinned uv -- so a smoke that does not is
# a guard that blames the wrong thing, which is worse than no guard.  The
# version is CHECKED below, not assumed, and a mismatch is SETUP_FAILED (rc 3),
# never a verdict about the binary.
SMOKE_UVBIN=${SMOKE_UVBIN:-/oscar/data/stellex/glvov/tasks/retread-cold-solve/verify_fixes/artifacts/uvbin}
SMOKE_REQUIRED_UV=${SMOKE_REQUIRED_UV:-0.12.5}
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
# to the store entry path.  PROOF-SMOKE-1-5, job 6004714: only the DIFFERENCE of
# these two is measured, and it is 167 -- the longest hermetic entry path that
# does not panic.  See the long note above `smoke_prefix_entry` for how 167 was
# read off two arms whose roots differ by eight bytes, and why the 255/88 split
# is the corroborated one.  The previous pair (256/92 -> a boundary of 164) came
# from 5989192 / DET1-5981957 and was three bytes out on a root that was itself
# 44 bytes short of the one the binary used, which is why it passed a root that
# panicked every time it was reached.
SMOKE_PREFIX_PAD=${SMOKE_PREFIX_PAD:-255}
SMOKE_PREFIX_TAIL=${SMOKE_PREFIX_TAIL:-88}
SMOKE_PREFIX_HEADROOM=${SMOKE_PREFIX_HEADROOM:-8}

# THE FRONTEND MARKERS.  `resolve_pypi{` is the span the resolver opens per
# environment; `Preparing metadata for:` is uv building an sdist's metadata
# inside it; the "assumed to be installed by conda" row is the resolver's own
# first INFO line.  Any one of them means the frontend moved.
# AND THE BACKEND HALF, WHICH IS NOT OPTIONAL AND WAS THE THIRD CORRECTION.
# MEASURED on `psmoke-guards` 5994177: the SLEEPING stub scored a frontend row
# in FOUR SECONDS -- `INFO resolve_pypi{group=cpu platform=linux-64-base}` --
# because the `cpu` environment has no path or git source and resolves without
# ever asking the backend for anything.  A frontend row alone therefore proves
# nothing about the backend, and the smoke handed REACHED_FRONTEND to a backend
# that had not answered a single call.  The backend half is one row it CANNOT
# print without serving a real build-dispatch request: `rpc request
# method=conda/outputs`.  Counted on four logs: known-good binary 14, sleeping
# stub 0, dying stub 0, and the DET-1-FIX candidate 30 (it served, then panicked
# -- and that one is caught by the process exit, which is the point of having
# three verdicts rather than one).  REACHED_FRONTEND now requires BOTH halves.
SMOKE_BACKEND_WORK_RE=${SMOKE_BACKEND_WORK_RE:-'conda/outputs'}
SMOKE_FRONTEND_RE=${SMOKE_FRONTEND_RE:-'resolve_pypi\{|Preparing metadata for:|are assumed to be installed by conda|Found static `pyproject\.toml` for:'}
# The first line worth quoting when nothing reached the frontend.
SMOKE_ERROR_RE=${SMOKE_ERROR_RE:-'panicked at|^Error|ERROR|error\[|error:|× |failed with status'}

smoke_say () { echo "### SMOKE $*"; }

# THE VERDICT'S POSITIVE TEST, IN ONE PLACE.  Both halves or neither.
smoke_both_halves () {
  grep -a -q -E "$SMOKE_FRONTEND_RE" "$LLOG" 2>/dev/null || return 1
  grep -a -q -E "$SMOKE_BACKEND_WORK_RE" "$BLOG" 2>/dev/null || return 1
  return 0
}

# ---- the verdict row, and it is printed on EVERY exit path ------------------
SMOKE_VERDICT=SETUP_FAILED
SMOKE_BINSHA=UNKNOWN
SMOKE_T0=$(date +%s)
SMOKE_RC=3
smoke_finish () {
  local wall=$(( $(date +%s) - SMOKE_T0 ))
  smoke_stage_lock_release
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
#
# PROOF-SMOKE-1-3 (2026-09-06): the ARITHMETIC is split out of the VERDICT.
# `multiarm_preamble.sh` has to COMPARE two roots -- the smoke's store root
# against the LONGEST arm's -- before either is judged, and the only way to do
# that with the rule as it stood was to re-derive the entry path beside it, i.e.
# a second copy of a length rule.  `smoke_prefix_composed` is that one number,
# `smoke_prefix_budget` is the one verdict, and both read `smoke_prefix_entry`.
#
# PROOF-SMOKE-1-5 (2026-09-07): THE RULE HAD THE RIGHT SHAPE AND THE WRONG ROOT,
# and the number that exposed it had been misread.  All of the below is measured
# on job 6004714 (two arms, roots eight bytes apart) plus the two red psg runs.
#
# (1) `composed=260` WAS NEVER A PATH LENGTH.  The panic reads `end byte index
#     18446744073709551591 is out of bounds for string of length 260`.  260 is
#     the PLACEHOLDER rattler-build builds by repeating a ten-byte template
#     until it clears its pad target -- 26 * 10 = 260 for every input, forever.
#     The number that carries the path is the INDEX: 2^64 - 25, i.e. `pad -
#     composed` underflowed by 25.  Both reds (6003336, 6003855) printed the
#     SAME index, and job 6004714's two arms printed 2^64-25 at entry 148 and
#     2^64-33 at entry 156 -- the deficit moved by EXACTLY the eight bytes the
#     root moved.  So the overrun is a clean linear function of the root and the
#     boundary is a single measured number, below.
#
# (2) THE ROOT WAS WRONG BY 44 BYTES.  `smoke_prefix_entry` composes
#     `<root>/retread/hermetic-build-envs/...`, which is
#     `courier::persistent_store_root_with` -- correct for a binary that carries
#     the L3-1b-4 flip (det141-proof 6001140's arms measured on disk at
#     `<arm cache home>/retread/hermetic-build-envs` = 81 bytes, exactly this
#     rule).  It is NOT correct for a binary whose hermetic store still goes
#     through `courier::retread_cache_root()`, which `fasttmp::
#     backend_env_override` redirects into the job namespace.  MEASURED ON DISK
#     while 6004714 ran, not modelled:
#       <fast-tmp>/retread-glvov/79ff79765a52/job-6004714/caches/retread/
#       hermetic-build-envs/v8   = 123 bytes (arm A) and 131 (arm B, +8)
#     and the entry leaf is named by the lock files left behind
#     (`.env-a383a406...lock`, 64 hex), so the real entry was 192, not 148.
#
# (3) THE MEASURED BOUNDARY.  Arm A: entry 192, deficit 25.  Arm B: entry 200,
#     deficit 33.  Both give the same answer: THE LARGEST HERMETIC ENTRY PATH
#     THAT DOES NOT PANIC IS 167 BYTES.  `SMOKE_PREFIX_PAD - SMOKE_PREFIX_TAIL`
#     is that 167 and nothing else; only their DIFFERENCE is measured, and the
#     255/88 split below is the corroborated one -- 88 is
#     `/rattler-output` + `/bld` + `/rattler-build_` +
#     `retread-hermetic-build-environment` (the ONE package
#     `render_debug_recipe` emits) + `_` + a ten-digit stamp + `/host_env`, and
#     255 is the only pad target for which the placeholder buffer is 260.
#
# (4) SO THE "PER-PACKAGE, EMISSION-ORDER" READING IS FALSIFIED.  The recipe has
#     exactly one output; there is no order for it to be noisy about, and both
#     reds printed the same constant.
smoke_prefix_entry () {
  # the longest entry a PERSISTENT store root can produce (the L3-1b-4 root):
  # <root>/retread/hermetic-build-envs/v8/env-<64 hex>
  printf '%s/retread/hermetic-build-envs/v8/env-%s' "$1" "$(printf 'a%.0s' $(seq 1 64))"
}
# ...and the longest entry the FAST-TMP JOB NAMESPACE produces, which is the one
# that actually panicked.  Every component is measured, not guessed:
# `retread-$USER` from `fasttmp::user_namespace_component`, a twelve-hex
# workspace digest from `fasttmp::workspace_hash` (six bytes rendered, measured
# as `79ff79765a52`), `job-<id>` from `fasttmp::current_job_component`, and
# `caches/retread` from the `RETREAD_CACHE_DIR` redirect.
smoke_prefix_fasttmp_entry () {
  printf '%s/retread-%s/%s/job-%s/caches/retread/hermetic-build-envs/v8/env-%s' \
    "$1" "${USER:-u}" "$(printf 'a%.0s' $(seq 1 12))" "${SLURM_JOB_ID:-$$}" \
    "$(printf 'a%.0s' $(seq 1 64))"
}
# THE ONE COMPOSER, AND IT TAKES THE WORST CANDIDATE, because the smoke does not
# get to know which store root the binary under test will choose -- that is
# decided by whether the binary carries L3-1b-4, which is precisely the thing a
# smoke is run to find out.  A budget that models one of two reachable roots is
# a budget that passes the other, which is what happened.
smoke_prefix_composed () {
  local root=$1 fast=${2-${RETREAD_FAST_TMP_ROOT:-}} e best entry
  entry=$(smoke_prefix_entry "$root"); best=${#entry}
  if [ -n "$fast" ]; then
    e=$(smoke_prefix_fasttmp_entry "$fast")
    [ "${#e}" -gt "$best" ] && best=${#e}
  fi
  printf '%s' "$(( best + SMOKE_PREFIX_TAIL ))"
}
smoke_prefix_budget () {
  local root=$1 label=${2:-store root} fast=${3-${RETREAD_FAST_TMP_ROOT:-}}
  local pe fe plen flen len composed oldcomposed which
  pe=$(smoke_prefix_entry "$root");            plen=${#pe}
  len=$plen; which="persistent $root"
  if [ -n "$fast" ]; then
    fe=$(smoke_prefix_fasttmp_entry "$fast");  flen=${#fe}
    if [ "$flen" -gt "$len" ]; then len=$flen; which="fast-tmp $fast"; fi
  else
    flen=NOT_MODELLED
  fi
  composed=$(( len + SMOKE_PREFIX_TAIL ))
  # THE OLD NUMBER STAYS ON THE PAGE.  A rule that changes silently is a rule
  # nobody can audit, and this one changed by enough to invert a verdict.
  oldcomposed=$(( plen + 92 ))
  echo "### SMOKE PREFIX BUDGET $label entry=$len composed=$composed pad=$SMOKE_PREFIX_PAD headroom=$(( SMOKE_PREFIX_PAD - composed )) root=$root worst=$which"
  echo "### SMOKE PREFIX BUDGET $label   candidates: persistent entry=$plen | fast-tmp entry=$flen ${fast:+root=$fast}"
  echo "### SMOKE PREFIX BUDGET $label   OLD RULE (pre PROOF-SMOKE-1-5, persistent root + tail 92) said composed=$oldcomposed headroom=$(( 256 - oldcomposed )) -- DELTA=$(( composed - oldcomposed ))"
  if [ -z "$fast" ]; then
    echo "### SMOKE PREFIX NOTE: the fast-tmp candidate is NOT modelled here because RETREAD_FAST_TMP_ROOT is unset AT THIS POINT."
    echo "### SMOKE   If the caller exports one LATER (det141_proof.sh exports \$G/fast-tmp long after it calls the preamble), this row is a FLOOR, not the answer."
  fi
  if [ "$composed" -gt "$SMOKE_PREFIX_PAD" ]; then
    echo "### SMOKE PREFIX REFUSAL: composed $composed > $SMOKE_PREFIX_PAD."
    echo "### SMOKE   rattler-build pads its build prefix to $SMOKE_PREFIX_PAD and panics on the underflow"
    echo "### SMOKE   in rattler_build_core/src/types/directories.rs before anything provisions."
    echo "### SMOKE   The longest hermetic entry that survives is $(( SMOKE_PREFIX_PAD - SMOKE_PREFIX_TAIL )) bytes (MEASURED, job 6004714); this one is $len."
    echo "### SMOKE   THE OVERRUN IS IN: $which"
    if [ -n "$fast" ] && [ "$which" != "persistent $root" ]; then
      echo "### SMOKE   AND IT CANNOT BE FIXED BY SHORTENING THE ROOT: the fast-tmp namespace adds a FIXED $(( flen - ${#fast} )) bytes,"
      echo "### SMOKE   so no root on this filesystem composes short enough while the store is redirected there."
      echo "### SMOKE   THE ACTUATOR IS RETREAD_FAST_TMP=off for the smoke (fasttmp.rs reads it), which is what this script now does."
    else
      echo "### SMOKE   SHORTEN THE ROOT, NOT THE CHECK: $root"
    fi
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

# ── THE STAGE MIRROR IS SHARED STATE, AND TWO SMOKES MUST NOT HOLD IT AT ONCE ──
# PROOF-SMOKE-1-4.
#
# READ THIS FIRST: THE CONCURRENCY READING THIS LOCK WAS COMMISSIONED FOR IS
# **WITHDRAWN**, AND THE MEASUREMENT THAT WITHDREW IT IS THIS LANE'S OWN.
# The story was a removal test: 6001839 (14 checks, node2311) and 6001840 (18
# checks, node2320) started in the SAME SECOND and 6001840 lost arm A, while the
# chained rerun 6002138, with no sibling psg job alive, scored 18/0 -- so
# "concurrency" was declared confirmed by removal.  THEN THIS LANE RAN
# `proof_smoke_guard.sh` AS **6003336**, WHICH ASKED `squeue` FOR A SIBLING psg
# JOB BEFORE STARTING AND PRINTED `no sibling psg job alive -- this run holds
# the stage mirror alone`, AND ARM A WENT RED ANYWAY, with the identical
# signature:
#   ### SMOKE BACKEND_DIED reason=PREFIX_PANIC_256 lock_rc=1
#         frontend_rows=0 backend_work_rows=16
#   ### SMOKE   panic| ... end byte index 18446744073709551591 is out of bounds
#         for string of length 260
# Same binary sha (6796560...), same `psg$J/c/x` root shape, same mirror key
# 85db7fd..., and the SAME declared budget row as both 09-06 jobs AND as the
# GREEN control 6002138: `entry=148 composed=240 pad=256 headroom=16`.  A run
# ALONE reproduced the failure that a run alone was supposed to rule out, so the
# removal test was CONFOUNDED and concurrency is not the cause.
#
# WHAT THE CAUSE ACTUALLY IS, named by THIS FILE'S OWN detector rather than
# inferred: `reason=PREFIX_PANIC_256`.  rattler-build pads its build prefix to
# 256 and `256 - len` underflows above it.  The composed string was 260 while
# the pre-lock budget check declared 240 with 16 bytes of headroom -- so THE
# BUDGET RULE UNDERCOUNTS BY TWENTY BYTES for this manifest, and the refusal
# that exists precisely to stop this passed a root that then panicked.  The
# budget models the hermetic STORE ENTRY (root + 100 + SMOKE_PREFIX_TAIL); it
# does not model the per-package BUILD prefix rattler-build composes, which
# varies with which package the resolver reaches first -- which is the emission
# -order instability DET-1 is about, and which is why this fails INTERMITTENTLY
# (five of six psmoke runs red on 09-06, one green) and looks like contention
# when it is not.  **THAT IS THE ROOT DEFECT AND THIS LANE DID NOT FIX IT.**  It
# is boarded, not smuggled into a lock: closing it means teaching
# `smoke_prefix_budget` a build prefix it does not currently model, against a
# measurement this lane has not taken, and a guess would put a second wrong
# number where one already is.
#
# SO WHY DOES THE LOCK STAY?  Because it guards a DIFFERENT hazard that has its
# own evidence and never depended on the concurrency reading:
#
# WHAT IS ACTUALLY SHARED, because the coordinator's phrase "one shared stage
# mirror" is nearly right and the nearly matters. The two jobs' stage PATHS were
# never shared: `SCR=.../psg$J` and `ROOT=$SMOKE_ROOT_BASE/smk$J-$NONCE` both
# carry the job id, and both jobs printed the SAME prefix row
# (`entry=148 composed=240 pad=256 headroom=16`) because their job ids are the
# same LENGTH, not because they were the same path. What is shared is one level
# down: `cp -al` from `$SMOKE_MIRROR_ROOT/$key` gives every staged file THE
# MIRROR'S OWN INODE, so two jobs staging from one mirror hold the same inodes,
# and a single in-place write by either reaches the mirror and the other job's
# workspace at once. The rsync fallback is worse still -- it `cp -al`s
# `$SMOKE_SRC_WS/third_party` straight out of the canonical tree.
#
# THE CHOICE, AND WHY IT IS THE SMALLER TRUE FIX. Making the stage per-job in
# CONTENT means a real copy instead of the hardlink farm, which is precisely the
# cost the mirror exists to avoid, and the stage is already per-job in PATH so
# there is nothing to move. Serialising access to the mirror is what the removal
# test actually validated (6002138 passed BECAUSE it ran alone), it is one
# `mkdir` on the acquire path, and it turns a corrupted share into a loud
# refusal that names its owner. So: a NON-BLOCKING dot-sidecar try-lock beside
# the mirror, the same shape the wheel store uses for `.<wheel>.whl.retread-
# fill-v1.lock` -- `mkdir` rather than a file because mkdir is the create-or-fail
# primitive that is atomic on NFS, where a noclobber redirect is not.
#
# WHAT THIS DOES **NOT** FIX, boarded rather than blind-fixed: the write-through
# itself. `phaseN_relock.sh` already carries the three readers for it --
# `stage_break_hardlinks` (cp -p + mv -f over every `-links +1` file),
# `stage_assert_mirror_disjoint` (the mirror must share no inode with
# $SMOKE_SRC_WS) and `stage_verify_mirror` (quarantine to `.DIRTY-$J`, set
# MIRROR_DIRTY, exit 12) -- and `smoke_stage` has NONE of them. The live mirror
# root carries three `.DIRTY-<jobid>` quarantines and one `.SRCLINKED-<jobid>`,
# so that hole has fired before. Carrying those three into the smoke is a
# SEPARATE change against a defect this lane did not measure, and it is boarded
# as debt, not smuggled in here.
SMOKE_STAGE_LOCK=${SMOKE_STAGE_LOCK:-1}
# TTL is the backstop, not the mechanism: the mechanism is the owner job's
# LIVENESS, asked of squeue at the point of use (law 5). A lock whose owner job
# is not in the queue is dead however new it is; the TTL only covers a lock
# whose owner id cannot be resolved at all.
SMOKE_STAGE_LOCK_TTL=${SMOKE_STAGE_LOCK_TTL:-14400}
# Set by a caller that stages ONCE and then runs many arms against those
# hardlinks (proof_smoke_guard.sh is the one in tree): the lock then spans the
# CALLER's life, not this one invocation's, and the caller removes it.
SMOKE_STAGE_LOCK_HOLD=${SMOKE_STAGE_LOCK_HOLD:-0}
SMOKE_STAGE_LOCK_PATH=
SMOKE_STAGE_LOCK_MINE=0          # 1 only if THIS process created the lock

smoke_stage_lock_path () {     # $1 = mirror key; THE one place this path is spelled
  printf '%s\n' "$SMOKE_MIRROR_ROOT/.$1.smoke-stage-v1.lock"
}

smoke_stage_lock_owner () {      # $1 = lock dir; echoes the owner job id or ''
  sed -n 's/^job=//p' "$1/owner" 2>/dev/null | head -1
}

smoke_stage_lock_acquire () {    # $1 = mirror key; rc 0 held, rc 1 BUSY
  local key=$1 lock owner age now mine
  [ "$SMOKE_STAGE_LOCK" = 1 ] || { echo "### SMOKE stage lock: DISABLED (SMOKE_STAGE_LOCK=0) -- concurrent smokes share the mirror's inodes"; return 0; }
  mkdir -p "$SMOKE_MIRROR_ROOT" 2>/dev/null
  lock=$SMOKE_MIRROR_ROOT/.$key.smoke-stage-v1.lock
  mine=${SLURM_JOB_ID:-pid$$}
  if mkdir "$lock" 2>/dev/null; then
    { echo "job=$mine"; echo "host=$(hostname -s)"; echo "pid=$$"; echo "at=$(date -Is)"; } > "$lock/owner"
    SMOKE_STAGE_LOCK_PATH=$lock; SMOKE_STAGE_LOCK_MINE=1
    echo "### SMOKE stage lock: TAKEN $lock owner=$mine"
    return 0
  fi
  owner=$(smoke_stage_lock_owner "$lock")
  # RE-ENTRANT WITHIN A JOB. Two smokes in one job are serialised by the shell
  # that runs them; they share a job id, and refusing the second would refuse
  # every multi-arm driver in the tree.
  if [ -n "$owner" ] && [ "$owner" = "$mine" ]; then
    SMOKE_STAGE_LOCK_PATH=$lock; SMOKE_STAGE_LOCK_MINE=0
    echo "### SMOKE stage lock: ADOPTED $lock -- already held by THIS job ($mine)"
    return 0
  fi
  # A DEAD OWNER IS RECLAIMED, LOUDLY. A smoke killed mid-stage would otherwise
  # wedge every later smoke until the TTL, which is a worse failure than the one
  # the lock exists to stop.
  now=$(date +%s); age=$(( now - $(stat -c %Y "$lock" 2>/dev/null || echo "$now") ))
  if [ -n "$owner" ] && [ -z "$(squeue -h -j "$owner" -o '%i' 2>/dev/null)" ]; then
    echo "### SMOKE STAGE LOCK STALE: $lock owner=$owner is not in the queue (age=${age}s) -- RECLAIMING"
    rm -rf "$lock" 2>/dev/null
    smoke_stage_lock_acquire "$key"; return $?
  fi
  if [ "$age" -gt "$SMOKE_STAGE_LOCK_TTL" ]; then
    echo "### SMOKE STAGE LOCK STALE: $lock age=${age}s > TTL ${SMOKE_STAGE_LOCK_TTL}s owner='${owner:-<unreadable>}' -- RECLAIMING"
    rm -rf "$lock" 2>/dev/null
    smoke_stage_lock_acquire "$key"; return $?
  fi
  echo "### SMOKE STAGE BUSY: $lock is held by job ${owner:-<unreadable>} (age=${age}s), which is LIVE."
  echo "###   The stage mirror hands out its OWN inodes via cp -al, so staging beside that job"
  echo "###   would give both workspaces the same files and an in-place write by either would"
  echo "###   reach the other and the mirror, and the live mirror root already carries three"
  echo "###   .DIRTY-<jobid> quarantines from phaseN_relock.sh's reader, so that is not hypothetical."
  echo "###   ACTUATOR: chain this job after ${owner:-the holder} (--dependency=afterany:${owner:-<jobid>}),"
  echo "###   or re-run once it is terminal. SMOKE_STAGE_LOCK=0 disables the lock deliberately."
  sed 's/^/###   owner: /' "$lock/owner" 2>/dev/null
  return 1
}

smoke_stage_lock_release () {
  [ "$SMOKE_STAGE_LOCK_MINE" = 1 ] || return 0
  [ "$SMOKE_STAGE_LOCK_HOLD" = 1 ] && { echo "### SMOKE stage lock: HELD past this smoke (SMOKE_STAGE_LOCK_HOLD=1) -- the caller releases $SMOKE_STAGE_LOCK_PATH"; return 0; }
  [ -n "$SMOKE_STAGE_LOCK_PATH" ] || return 0
  rm -rf "$SMOKE_STAGE_LOCK_PATH" 2>/dev/null \
    && echo "### SMOKE stage lock: RELEASED $SMOKE_STAGE_LOCK_PATH"
  SMOKE_STAGE_LOCK_MINE=0
  return 0
}

smoke_stage () {                 # $1 = workspace to create
  local ws=$1 key mirror S
  key=$(smoke_stage_key)
  mirror=$SMOKE_MIRROR_ROOT/$key
  # BEFORE a single inode is handed out. A BUSY mirror is a SETUP refusal, never
  # a verdict about the binary: the caller turns rc 1 here into SETUP_FAILED rc 3.
  smoke_stage_lock_acquire "$key" || return 2
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

# The lock must come off even when this process is killed rather than finished:
# a smoke SIGKILLed mid-run would otherwise leave the mirror BUSY until another
# job's liveness check reclaimed it. `smoke_finish` releases on every ordinary
# path and this covers the rest; both are idempotent.
trap 'smoke_stage_lock_release' EXIT

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
# PROOF-SMOKE-1-5.  THE FAST-TMP ROOT IS DECIDED HERE, ABOVE THE BUDGET, because
# the budget cannot model a root it has not been told about -- and the root it
# was not being told about is the one that panicked.  (It used to be exported
# two hundred lines below, after the refusal it belongs to.)
RETREAD_FAST_TMP_ROOT=$ROOT/f
# AND FAST-TMP IS DISENGAGED FOR THE SMOKE, WHICH IS A ROOT FIX AND NOT A
# WORKAROUND, and the reason is arithmetic rather than preference.  MEASURED
# (job 6004714, and on disk while it ran): the fast-tmp job namespace inserts a
# FIXED 146 bytes between the FAST-TMP ROOT and the hermetic entry -- and the
# fast-tmp root is `$ROOT/f`, so 148 from the smoke root itself; both numbers
# appear below and they are the same measurement counted from two places --
# (`/retread-$USER/<12 hex>/job-<id>/caches/retread/hermetic-build-envs/v8/env-
# <64 hex>`), so the entry is `len(root) + 148` and the boundary is 167 -- which
# needs a root of 19 bytes, when `/oscar/data/stellex/glvov/retread` alone is 33.
# NO ROOT ON THIS FILESYSTEM CAN SATISFY IT.  A smoke that cannot be given a
# passing root is not a guard, it is a coin: 6003336 red, 6003619 green, 6003855
# red on ONE binary, and the green one never reached the hermetic build at all
# (zero `hermetic` rows in its backend log) -- it won a race to the frontend and
# killed the lock before the panic could happen.  Disengaging fast-tmp puts the
# store back on `persistent_store_root`, where the budget's rule is the true one
# and the verdict is about the binary again.  IT IS PRINTED, because it is a
# real difference from the driver's environment and a reader must see it.
export RETREAD_FAST_TMP=${SMOKE_FAST_TMP:-off}
echo "### SMOKE fast-tmp mode=$RETREAD_FAST_TMP (PROOF-SMOKE-1-5: 'off' keeps the hermetic store off the +148-byte job namespace, which no root here can afford; the DRIVER still runs with it on)"
# The fast-tmp candidate is modelled EXACTLY WHEN the store can reach it: with
# the mode off it cannot, so passing it would refuse every root for a chain that
# is not in play.  One call, one verdict.
SMOKE_PREFIX_FAST_ARG=$RETREAD_FAST_TMP_ROOT
[ "$RETREAD_FAST_TMP" = off ] && SMOKE_PREFIX_FAST_ARG=
smoke_prefix_budget "$XDG_CACHE_HOME" "smoke store root" "$SMOKE_PREFIX_FAST_ARG" || {
  echo "### SMOKE reason=PREFIX_REFUSAL -- this is a refusal about the ROOT, not a verdict about the binary."
  SMOKE_VERDICT=SETUP_FAILED; SMOKE_RC=3; smoke_finish
}

# ---- the workspace ----------------------------------------------------------
if [ -n "${SMOKE_WS:-}" ] && [ -f "$WS/pixi.toml" ]; then
  echo "### SMOKE stage: REUSING the caller's workspace $WS (SMOKE_WS was set and it is staged)"
else
  smoke_stage "$WS"
  case $? in
    0) ;;
    2) echo "### SMOKE reason=STAGE_BUSY -- this is a refusal about the shared stage mirror, not a"
       echo "###   verdict about the binary. Nothing was staged and no inode was shared."
       SMOKE_VERDICT=SETUP_FAILED; SMOKE_RC=3; smoke_finish;;
    *) smoke_setup_failed "could not stage a workspace at $WS";;
  esac
fi
# THE PATHS THIS SMOKE ACTUALLY OWNS, measured after staging. The composed
# budget is NOT recomputed here -- `smoke_prefix_budget` above is the one
# implementation of that rule and a second copy is how the rule drifts (this
# file says so about the LIB seam and it applies to itself). What is printed is
# the raw lengths, so a reader of PROOF-SMOKE-1-4's panic (`string of length
# 260`, against two jobs that both declared composed=240) can tell a root
# overrun from a string that never was a root of this job's at all.
echo "### SMOKE stage paths: ws_len=${#WS} root_len=${#ROOT} cache_len=${#XDG_CACHE_HOME} lock=${SMOKE_STAGE_LOCK_PATH:-<none>}"
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

# ---- THE BACKEND'S ENVIRONMENT, the same one every driver gives it ----------
# A smoke whose environment differs from the production driver's answers a
# different question.  These are the knobs the drivers set, and the uv is
# CHECKED rather than trusted.
export PATH=/users/glvov/.pixi/bin:$SMOKE_UVBIN:/users/glvov/.local/bin:/usr/bin:/bin
export RETREAD_UV=$SMOKE_UVBIN/uv
[ -x "$RETREAD_UV" ] || smoke_setup_failed "RETREAD_UV $RETREAD_UV missing -- retread's preflight refuses any other uv"
UVVER=$("$RETREAD_UV" --version 2>&1 | awk '{print $2}')
echo "### SMOKE uv: $RETREAD_UV -> $UVVER (retread's uv_closure::REQUIRED_UV wants $SMOKE_REQUIRED_UV)"
[ "$UVVER" = "$SMOKE_REQUIRED_UV" ] || smoke_setup_failed "uv is $UVVER, retread's preflight wants $SMOKE_REQUIRED_UV -- every backend call would fail 'preflight: uv version mismatch' and the smoke would blame the binary"
# FAST-TMP GETS A JOB-SCOPED ROOT ON DISK, WHICH IS WHAT EVERY DRIVER GIVES IT.
# MEASURED on `psmoke-guards` 5993931 arm A, with the uv fixed and fast-tmp left
# at its default: `fast-tmp backend engage: retread fast-tmp budget too small:
# estimated need 85899345920 bytes > budget 77309411328 bytes from SLURM memory
# environment` -- 80 GiB wanted against a 72 GiB budget derived from --mem=96G,
# because an unrooted fast-tmp is charged to RAM.  Every driver exports a DISK
# root (det1_proof2.sh: `RETREAD_FAST_TMP_ROOT=$G/fast-tmp` under its own cache
# root) and none of them meets this budget.  Raising --mem would be treating the
# symptom; the smoke's environment must be the driver's.
export RETREAD_SCRATCH_ROOT=$ROOT/s
# PROOF-SMOKE-1-5: the VALUE is chosen far above, beside the budget that has to
# model it; this line is now only the export, and it must stay in agreement.
export RETREAD_FAST_TMP_ROOT
export XDG_STATE_HOME=$ROOT/t
export XDG_CONFIG_HOME=$ROOT/n
mkdir -p "$RETREAD_SCRATCH_ROOT" "$RETREAD_FAST_TMP_ROOT" "$XDG_STATE_HOME" "$XDG_CONFIG_HOME" \
  || smoke_setup_failed "cannot create the job-scoped scratch/fast-tmp roots under $ROOT"
echo "### SMOKE fast-tmp root (DISK, job-scoped, ONE LETTER ON PURPOSE): $RETREAD_FAST_TMP_ROOT"
# AND THE SECOND MEASUREMENT, WHICH IS WHY THESE NAMES ARE ONE LETTER.  With
# `$CACHE/g/fast-tmp` (57 bytes) the KNOWN-GOOD binary `integration-569b0ac`
# died at `directories.rs:181` with `string of length 260` -- 5994392 arm A,
# `BACKEND_DIED reason=PREFIX_PANIC_256 backend_work_rows=16` at 75 s, on a
# binary that is fine.  MEASURED on that job: the hermetic store does NOT live
# under XDG_CACHE_HOME at all when fast-tmp is on; it lives at
# `<fast-tmp>/retread-$USER/<12 hex>/job-<jid>/caches/retread/hermetic-build-envs`
# -- 131 bytes of directory before the entry.  `$ROOT/f` is twelve bytes shorter
# than `$CACHE/g/fast-tmp` and buys exactly that back.
# PROOF-SMOKE-1-2 IS CLOSED (PROOF-SMOKE-1-5), AND THE REASON IT STAYED OPEN IS
# WORTH KEEPING.  It read: "modelling the fast-tmp chain was tried on paper and
# REJECTED -- the arithmetic (root + 100 + 92) predicts 284 for a configuration
# that is MEASURED AT 260, so a modelled refusal would refuse working jobs."
# THAT REJECTION WAS CAUSED BY THE SAME MISREADING THE FIX IS ABOUT: 260 was
# never a measurement of the path, it is rattler-build's placeholder buffer.  The
# configuration was measured at 280 (entry 192 + tail 88), the paper arithmetic
# said 284, and the four bytes between them are the tail's own error (92 vs the
# real 88).  The model was RIGHT to within four bytes and was thrown out on a
# number that meant something else -- which is the whole argument for reading
# the index and not the length.  The fast-tmp chain is now modelled, from a
# measurement: `/retread-$USER/<12 hex>/job-<id>/caches/retread/hermetic-build-
# envs/v8/env-<64 hex>` = a FIXED 146 bytes from the fast-tmp root, confirmed on disk while job 6004714
# ran (123 bytes to `.../v8` on arm A, 131 on arm B, the entry leaf named by the
# `.env-<64 hex>.lock` files left behind).
export CONDA_OVERRIDE_CUDA=12
export CONDA_OVERRIDE_GLIBC=2.35
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export UV_LINK_MODE=copy
export UV_LOCK_TIMEOUT=${SMOKE_UV_LOCK_TIMEOUT:-3600}
export RUST_BACKTRACE=1

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
MYPGID=$(awk '{sub(/^.*\) /,""); print $3}' "/proc/$$/stat" 2>/dev/null)
# READ THE PGID IN A LOOP, NOT ONCE.  `setsid` calls setsid(2) AFTER the shell
# has forked and returned the pid, so a single read right here loses the race
# and reports the parent's group -- which is what 5993691 printed on all three
# of its arms.  Five one-second reads, and only then a verdict.
LPGID=
for _i in 1 2 3 4 5; do
  LPGID=$(awk '{sub(/^.*\) /,""); print $3}' "/proc/$LPID/stat" 2>/dev/null)
  [ -n "$LPGID" ] && [ "$LPGID" != "${MYPGID:-none}" ] && break
  kill -0 "$LPID" 2>/dev/null || break
  sleep 1
done
LPGID=${LPGID:-$LPID}
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
  # BOTH HALVES.  A frontend row alone is scored by a backend that never
  # answered (5994177 arm C: the `cpu` environment resolves with no source at
  # all), and a backend row alone says nothing about the resolver.
  if smoke_both_halves; then
    VERDICT=REACHED_FRONTEND; break
  fi
  if ! kill -0 "$LPID" 2>/dev/null; then
    # one last read: the rows may have landed in the same instant the process
    # exited, and a race that reports BACKEND_DIED for a healthy binary would
    # cost a lane a whole re-cut.
    sleep 1
    if smoke_both_halves; then VERDICT=REACHED_FRONTEND
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
BE_ROWS=$(grep -a -c -E "$SMOKE_BACKEND_WORK_RE" "$BLOG" 2>/dev/null); BE_ROWS=${BE_ROWS:-0}
echo "### SMOKE lock ended verdict=$VERDICT wall=${WALL}s lock_rc=$LRC frontend_rows=$FE_ROWS backend_work_rows=$BE_ROWS log_lines=$(wc -l < "$LLOG")"

case "$VERDICT" in
  REACHED_FRONTEND)
    echo "### SMOKE first frontend row (and backend_work_rows=$BE_ROWS, both halves required):"
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
    [ "$BE_ROWS" -eq 0 ] && [ "$REASON" = UNCLASSIFIED ] && REASON=NO_BACKEND_WORK
    echo "### SMOKE BACKEND_DIED reason=$REASON lock_rc=$LRC frontend_rows=$FE_ROWS backend_work_rows=$BE_ROWS"
    if [ "$REASON" = PREFIX_PANIC_256 ]; then
      echo "### SMOKE   THIS IS DET-1-FIX-1 AND IT IS A PATH-LENGTH ACCIDENT, NOT A CODE DEFECT."
      echo "### SMOKE   rattler-build pads its build prefix to $SMOKE_PREFIX_PAD; the composed path overran it."
      echo "### SMOKE   Shorten the store root and re-smoke before touching the branch."
      grep -a -m1 -E 'is out of bounds for string of length' "$LLOG" "$BLOG" 2>/dev/null | cut -c1-300 | sed 's/^/### SMOKE   panic| /'
    fi
    # THE FIRST ERROR LINE PLUS THE TWO AFTER IT.  A rust panic's first line is
    # only its LOCATION -- `thread 'main' panicked at src/foo.rs:1:1:` -- and the
    # message is on the NEXT line.  5993691 arm B quoted the header and dropped
    # the sentence, which is the half a reader needs.
    echo "### SMOKE first ERROR/panic line and the two after it (backend log first, then the lock log):"
    FIRST=$(grep -a -m1 -A2 -E "$SMOKE_ERROR_RE" "$BLOG" 2>/dev/null)
    [ -n "$FIRST" ] || FIRST=$(grep -a -m1 -A2 -E "$SMOKE_ERROR_RE" "$LLOG" 2>/dev/null)
    [ -n "$FIRST" ] || FIRST=$(tail -3 "$LLOG" 2>/dev/null)
    printf '%s\n' "$FIRST" | cut -c1-300 | sed 's/^/### SMOKE   /'
    echo "### SMOKE lock log tail:"
    tail -12 "$LLOG" | cut -c1-200 | sed 's/^/### SMOKE   /'
    SMOKE_VERDICT=BACKEND_DIED; SMOKE_RC=1 ;;
  TIMEOUT)
    echo "### SMOKE TIMEOUT: ${SMOKE_WALL}s elapsed with frontend_rows=$FE_ROWS backend_work_rows=$BE_ROWS (BOTH must be non-zero) and the lock still alive."
    echo "### SMOKE   A healthy binary reached the frontend in 199 s on a fully cold cache (measured, 5989192 arm 3)."
    echo "### SMOKE lock log tail:"
    tail -12 "$LLOG" | cut -c1-200 | sed 's/^/### SMOKE   /'
    SMOKE_VERDICT=TIMEOUT; SMOKE_RC=2 ;;
esac

echo "### SMOKE artifacts: $LLOG $BLOG $SHIM"
smoke_finish
