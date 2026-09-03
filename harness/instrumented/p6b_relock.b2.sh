#!/usr/bin/env bash
### EVIDENCE BEGIN
# phaseN_relock.sh -- TEMPLATE for the RELOCK half of a two-phase retread batch.
#
# Copy this file (and phaseN_cert.sh + cleanup.sh) into a new phase directory,
# edit ONLY the SUBSTITUTE block, run `bash -n`, run the script's own
# leftover-token self-check (it runs on every start and exits 9 on a hit), and
# submit. README.md next to this file is the five-step recipe.
#
# DERIVED BY SUBSTITUTION from the phase-1/phase-2 pair that ran jobs 5597671 /
# 5597694 (the last certified pair on this campaign). Three deltas, each of them
# measured, not asserted:
#
#   1. PERSISTENT CACHES. This template sources tools/retread_fast_env.sh and
#      calls retread_fast_env "$WS" right after the job-scoped env block, so
#      PIXI_CACHE_DIR / RATTLER_CACHE_DIR / UV_CACHE_DIR live under
#      agrescap/cache/retread/ and the route-probe verdict cache is symlinked
#      out of the job-scoped fast-tmp namespace. Job 5598763 measured the same
#      manifest, same node, same job, three arms:
#          arm A  cold, defaults          rc=0  lock wall 2865s
#          arm B  cold, persistent caches emptied at start  rc=0  2633s
#          arm C  arm B again, caches WARM rc=0  lock wall   69s
#      41x. Lock RESOLUTION identical in all three (pypi names 174, conda names
#      1707, pypi urls 213, conda urls 2584; env_version_delta.py moved=0 over
#      all 27 envs, both comparisons). See HAZARDS in retread_fast_env.sh for
#      the two byte-level warm-vs-cold deltas -- they are not resolution changes.
#
#   2. NO SELF-CLEANUP HERE. The predecessor of this file removed its own
#      job-scoped cache root before exiting, which put an NFS `rm -rf` of a
#      ~1M-inode tree on the afterok critical path: job 5596128 spent 5152s
#      (86 min) in that epilogue against a 3679s lock, holding its cert
#      successor and 160G of QOS the whole time. Job 5598763's epilogue took
#      4795s + 2552s = 2.0h for two roots. Rule (HANDOFF section 2): extract
#      artifacts, exit, and clean up from the LAST job of the chain. So the
#      relock phase removes NOTHING; the cert phase submits cleanup.sh with
#      --dependency=afterany and exits.
#
#   3. /usr/bin/time -v ALWAYS, to its own file. The lock is wrapped with
#      `-o "$A/$TAG-$J.lock.time.txt"` so real peak RSS is a first-class
#      artifact instead of a line buried in a 40k-line lock log. sacct MaxRSS is
#      UNUSABLE on this filesystem -- every job reports ~100% of its cgroup cap
#      (a 120G job reports ~125.8M K, a 160G job ~167.8M K) regardless of what
#      it did. That is reclaimable page cache, not demand.
#
# ---- MEMORY: ask for 72G here, 100G for the cert. Measured, not quoted. ----
#   relock peak process RSS   8,854,172 K   (job 5597671, /usr/bin/time -v)
#   relock peak process RSS   8,829,764 K   (job 5594283, same instrument)
#   relock peak process RSS   4,704,452 K   (job 5598763 cold arm A)
#   relock peak process RSS   2,869,056 K   (job 5598763 WARM arm C)
#   worst cert env peak RSS   1,475,216 K   (env `gpu`, job 5597694 ledger)
#   worst cert env peak RSS   1,446,432 K   (env `gpu`, job 5594284 ledger)
#   Campaign convention writes those as GB decimal: 8.85 / 8.83 / 4.70 / 2.87
#   for the relocks and 1.48 / 1.45 for the cert envs (1.41 / 1.38 GiB).
#   72G is >8x the worst relock measurement. 100G for the cert buys headroom
#   for 26 sequential installs plus page cache without tripping the per-user
#   QOS cap (normal = cpu 64, mem 492G for the WHOLE user -- a 160G request is
#   what left job 5597889 pending behind QOSMaxMemoryPerUser while a node sat
#   idle). Raise it only against a measurement, never against a sacct row.
#
#       env -u SLURM_JOB_ID sbatch --partition=batch --qos=normal \
#           --cpus-per-task=16 --mem=72G --time=03:00:00 \
#           --job-name=<tag>-p1 --output=<shared path>/slurm-%j.out \
#           ./phaseN_relock.sh
#
# ---- p6b DELTA: THIS IS AN INSTRUMENTED RELOCK, NOT A MEASUREMENT RELOCK ----
# Derived from tools/phase_template/phaseN_relock.sh by substitution. Its job is
# to make ONE relock emit uv resolver tracing, which no artifact this campaign
# holds contains. Five deltas beyond the SUBSTITUTE block, each measured here
# before it was written down:
#
#   1. VERBOSITY. `pixi lock --help` on pixi 0.73.0: "-v for warnings, -vv for
#      info, -vvv for debug, -vvvv for trace". So -vv is INFO and does NOT reach
#      a debug-level uv target; -vvv is the one that does. The binary carries the
#      filter template `apple_codesign=off,pixi=<lvl>,pixi_command_dispatcher=<lvl>,
#      pixi_core=<lvl>,rattler_upload=<lvl>,uv_resolver=<lvl>,resolvo=<lvl>`
#      (read out of `strings /users/glvov/.pixi/bin/pixi.real`), so `uv_resolver`
#      IS in pixi's own default filter and a plain -vvv already turns it to debug.
#      `uv_client` and `uv_distribution` are NOT in that template; they are
#      reachable only through RUST_LOG.
#
#   2. RUST_LOG ON THE FRONTEND, AND IT REPLACES THE DEFAULT FILTER. Measured on
#      a throwaway 1-line workspace on the login node: `RUST_LOG=pixi_core=trace,
#      pixi=trace pixi.real lock` with NO -v flag printed TRACE/DEBUG pixi_core
#      rows, which the default (warn) filter would never have emitted. So RUST_LOG
#      is honoured and overrides -v. That is why the RUST_LOG below re-names the
#      pixi targets it needs -- dropping them would silence the very rows the
#      timeline is built from.
#
#   3. THE FRONTEND'S OWN ROWS CARRY NO TIMESTAMP. Only the backend's rows do
#      (they come from the backend's tracing through the stderr shim). Every
#      "timeline" this campaign has built came off backend rows for that reason.
#      An instrumented run must stamp the frontend at the pipe, so the lock log
#      is piped through p6b_stamp.py and the lock's rc is taken from
#      PIPESTATUS[0], never from the pipeline.
#
#   4. THE LOG DOES NOT GO TO NFS WHILE IT IS BEING WRITTEN. uv_resolver=debug on
#      a 27-env relock is a large multi-GB stream; 842 MB of warnings to /oscar
#      jammed writeback and took a node out for 9 h on 2026-08-31. The lock and
#      backend logs are written to node-local storage and gzipped into artifacts/
#      at the end, with the extractor run against the local copies first.
#
#   5. BACKEND LOGGING STAYS ON PIXI_BUILD_RETREAD_LOG AND RUST_LOG IS UNSET FOR
#      IT. The backend is a child process and would otherwise inherit the
#      frontend's RUST_LOG, which is exactly the control HANDOFF section 1 forbids
#      for it. The shim unsets RUST_LOG before exec'ing the backend.
#
# NOT SUBMITTED. The canonical manifest does not lock until the *_debug_cpython
# blocker (p5x) is fixed; submitting before that buys a 47 s rc=1.
# NEVER edit this file while a job is running it -- copy it aside first.
### EVIDENCE END
set -uo pipefail

### SUBSTITUTE: BEGIN -- MANIFEST, PROBES, EXPECT_*  (edit ONLY between these markers)
# Every campaign-specific constant in this harness lives here. Nothing below
# this block names a previous batch; the self-check right after it enforces that.

TAG=P6B                                      # short batch tag; roots become certP6B-<job> / ws.P6B-<job>
T=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11
D=$T/p6b-instrumented                        # THIS harness's own directory (artifacts land in $D/artifacts)

# --- the manifest under test -------------------------------------------------
# The CANONICAL manifest, unmodified: this run measures where the time goes, so
# it must lock exactly what a production relock locks. b1-scratch/pixi.toml.orig
# is a byte copy of imprint-data/pixi.toml, so the diff gate below is 0/0.
SRC_WS=/oscar/data/stellex/glvov/imprint-data           # READ-ONLY canonical source tree
CLEANED=$T/b1-scratch/pixi.toml.orig                    # the scratch manifest this batch locks
EXPECT_CLEANED_MD5=9711eb990bfe211d498d1635a60e0d07     # md5sum of $CLEANED
EXPECT_MANIFEST_LINES=1003                              # wc -l of $CLEANED
EXPECT_DEL=0                                            # diff SRC_WS/pixi.toml CLEANED : '< ' lines
EXPECT_ADD=0                                            # diff SRC_WS/pixi.toml CLEANED : '> ' lines
EXPECT_ENVS=27                                          # envs the manifest declares AND the lock must carry
EXPECT_JETSON_ROWS=1                                    # live `jetson = ` rows (0 disables the jetson env)

# --- residual-pin gate: one pattern per deleted pin family, each must be 0 ----
RESIDUAL_PATTERNS=()                                    # nothing is deleted here: the manifest is canonical

# --- probes ------------------------------------------------------------------
PROBES_CANON=$T/p1e-certify-lock/artifacts/probes.tsv    # canonical, operator-gated, NEVER edited
PROBES_ARM=$PROBES_CANON                                 # unchanged: no pin is deleted in this batch
PROBE_TOKENS=()                                          # module tokens that must be GONE from $PROBES_ARM

# --- instruments -------------------------------------------------------------
ATTRIB=$T/tools/b2_attribute.sh                          # whole-file occurrence delta (secondary, blind by design)
EVD=$T/b3-phase1/env_version_delta.py                    # PER-ENV PER-PACKAGE version delta (primary CHECK 1)
EVD_PACKAGES="openmesh networkx pillow sentry-sdk numpy" # packages CHECK 1 adjudicates
WATCH_PACKAGES="gxx_linux-64 cmake"                      # observed, not touched by this batch
BASE_LOCK=$SRC_WS/pixi.lock                              # baseline for the occurrence delta

# --- toolchain ---------------------------------------------------------------
PIXI=/users/glvov/.pixi/bin/pixi.real                    # bypass the flock shim
### SUBSTITUTE ME BEFORE SUBMITTING: the p5x binsnap.
# This run must use the binary that FIXES the *_debug_cpython blocker, because
# the canonical manifest does not lock without it. The dir pattern is
# `binsnaps/fix-p5x-debug-cpython/pixi-build-retread`; that path already exists
# but holds a PRE-GREEN build whose sha256 at 2026-09-02 18:55 EDT was
# 67ba7131061ade2ad3fbbcf70913aa39167bf1a1fcaec3f12909b5ea9af6cf28. When p5x is
# green, re-stamp BOTH lines below from `sha256sum $SNAP` -- the gate at the top
# of the run refuses a mismatch, which is the point.
SNAP=$T/binsnaps/integration-44233cf/pixi-build-retread
EXPECT_SHA=2dd790bf7e7769bdb4e060497020b4b2484253bbdfa89fe71e5b6c164ee1ee84
UVBIN=/oscar/data/stellex/glvov/tasks/retread-cold-solve/verify_fixes/artifacts/uvbin
FAST_ENV=$(dirname "$0")/../tools/retread_fast_env.sh    # persistent caches; fallback below
[ -f "$FAST_ENV" ] || FAST_ENV=$T/tools/retread_fast_env.sh

# --- INSTRUMENTATION: the whole reason this harness exists -------------------
# Verbosity that reaches uv. -vv is INFO (see the EVIDENCE header); -vvv is the
# debug level, and `uv_resolver` sits in pixi's own default filter template.
LOCK_VERBOSITY=-vvv
# RUST_LOG REPLACES that default filter, so it has to re-name every target the
# extractor reads, not just the uv ones. `uv_client`/`uv_distribution` are the
# targets that say whether a resolve is fetching or thinking; `pixi_uv` is NOT a
# tracing target in this binary -- `strings` shows only the crate names
# `pixi_uv_context` and `pixi_uv_conversions`, so those are what is named here.
FRONTEND_RUST_LOG=apple_codesign=off,uv_resolver=debug,uv_client=info,uv_distribution=info,pixi_uv_context=debug,pixi_uv_conversions=debug,pixi=info,pixi_core=info,pixi_command_dispatcher=info,resolvo=info
# The backend's control surface, never RUST_LOG (HANDOFF section 1). This value
# is byte-identical to the reference runs 5611846 / 5618074 so the backend rows
# stay comparable with them; a bare `debug` also works but turns every
# dependency crate to debug and breaks that comparison.
BACKEND_LOG_FILTER=pixi_build_retread=debug,warn
# Line stamper: the frontend's rows carry no timestamp of their own.
STAMPER=$D/p6b_stamp.py
EXTRACTOR=$D/p6b_extract.py
# Where the multi-GB logs are WRITTEN (node-local). Copied to $A gzipped at the end.
LOCAL_LOG_ROOT=/tmp/retread-p6b

# --- leftover-token self-check ------------------------------------------------
# Names of PREVIOUS batches. A hit anywhere outside the three marked regions is
# a botched derivation, which is what HANDOFF section 2's grep exists to catch.
LEFTOVER_RE='p5w|P5W|de58240|p5x|P5X|67ba7131|0d867265|SUBSTITUTE_ME_SHA256|bfinal|BFP1|BFP2|bfp1|bfp2|b1c|b1-phase|b1b-phase|b2-phase|b2b-phase|b3-phase|ctl-phase|eff-phase|/b1_|/b2_|/b3_|/ctl_|p5sab|P5SAB|p5t_abc|P5TABC|p5u|P5U|p5z|P5Z|p6a|P6A|certB3P1|2cfec88d|57105d38|phase-template-example|PHASEN'
### SUBSTITUTE: END

### LEFTOVER-CHECK BEGIN
# Strips the three marked regions (this one included) and fails on any survivor.
# Comments are NOT exempt: a stale path in a comment has misled a reader on this
# campaign before. Deliberate evidence citations belong in the EVIDENCE region.
LEFT=$(awk '
  /^### EVIDENCE BEGIN/       {e=1} /^### EVIDENCE END/       {e=0; next} e {next}
  /^### SUBSTITUTE: BEGIN/    {s=1} /^### SUBSTITUTE: END/    {s=0; next} s {next}
  /^### LEFTOVER-CHECK BEGIN/ {l=1} /^### LEFTOVER-CHECK END/ {l=0; next} l {next}
  {print FILENAME ":" FNR ": " $0}' "$0" | grep -E "$LEFTOVER_RE")
if [ -n "$LEFT" ]; then
  echo "### FATAL leftover-token self-check FAILED -- this harness still names a previous batch"
  printf '%s\n' "$LEFT"
  exit 9
fi
echo "### leftover-token self-check: clean (regex $LEFTOVER_RE)"
### LEFTOVER-CHECK END

J=${SLURM_JOB_ID:?missing Slurm job id}
A=$D/artifacts
C=/oscar/data/stellex/glvov/retread/cert${TAG}-$J        # job-scoped cache root
G=$C/g
WS=/oscar/data/stellex/glvov/retread/ws.${TAG}-$J        # pristine workspace

CQ=/oscar/runtime/bin/checkquota          # NOT on a batch job's default PATH: job 5611846 printed
[ -x "$CQ" ] || CQ=$(command -v checkquota 2>/dev/null || echo true)   # two EMPTY quota rows because of it
mkdir -p "$A"
hostname; date -Is
echo "### ${TAG} RELOCK JOB=$J NODE=${SLURM_JOB_NODELIST:-none} nproc=$(nproc) mem=$(free -g|awk '/^Mem:/{print $2"G"}') glibc=$(ldd --version|head -1)"
echo "### inode quota BEFORE:"; "$CQ" 2>/dev/null | grep -E 'data\+stellex|^Name' | head -4

########## 0. GATES ##########
case "$WS" in /oscar/data/stellex/glvov/retread/ws.${TAG}-*) ;; *) echo "FATAL bad WS $WS"; exit 4;; esac
case "$C"  in /oscar/data/stellex/glvov/retread/cert${TAG}-*) ;; *) echo "FATAL bad C $C";  exit 4;; esac
[ -f "$SNAP" ] || { echo "FATAL: pre-made snapshot $SNAP missing"; exit 8; }
GOT_SHA=$(sha256sum "$SNAP" | awk '{print $1}')
[ "$GOT_SHA" = "$EXPECT_SHA" ] || { echo "FATAL: snapshot sha $GOT_SHA != $EXPECT_SHA"; exit 8; }
echo "### backend snapshot OK: $SNAP sha256=$GOT_SHA"
ls -l "$SNAP"; "$SNAP" --version 2>&1 | head -2
[ -f "$FAST_ENV" ] || { echo "FATAL: persistent-cache snippet $FAST_ENV missing"; exit 8; }
[ -f "$CLEANED" ] || { echo "FATAL: manifest under test $CLEANED missing"; exit 9; }
echo "### manifest md5: $(md5sum "$CLEANED")"
GOT_CM=$(md5sum "$CLEANED" | awk '{print $1}')
[ "$GOT_CM" = "$EXPECT_CLEANED_MD5" ] || { echo "FATAL: manifest md5 $GOT_CM != $EXPECT_CLEANED_MD5"; exit 9; }
for f in "$PROBES_ARM" "$PROBES_CANON" "$ATTRIB" "$EVD" "$BASE_LOCK"; do
  [ -e "$f" ] || { echo "FATAL: missing required path $f"; exit 9; }
done
if [ "${#RESIDUAL_PATTERNS[@]}" -gt 0 ]; then
  echo "### scratch manifest residual pin rows (want 0 each):"
  RESID_BAD=0
  for pat in "${RESIDUAL_PATTERNS[@]}"; do
    n=$(grep -c "$pat" "$CLEANED")
    printf '  %-40s %s\n' "$pat" "$n"
    [ "$n" = 0 ] || RESID_BAD=1
  done
  [ "$RESID_BAD" = 0 ] || { echo "FATAL: a deleted pin still has a live row in $CLEANED"; exit 9; }
fi
echo "### canonical-vs-scratch manifest diff (want EXACTLY $EXPECT_DEL deleted, $EXPECT_ADD added):"
diff "$SRC_WS/pixi.toml" "$CLEANED"
DEL=$(diff "$SRC_WS/pixi.toml" "$CLEANED" | grep -c '^< ')
ADD=$(diff "$SRC_WS/pixi.toml" "$CLEANED" | grep -c '^> ')
echo "### manifest diff counts: deleted=$DEL (want $EXPECT_DEL) added=$ADD (want $EXPECT_ADD)"
[ "$DEL" = "$EXPECT_DEL" ] && [ "$ADD" = "$EXPECT_ADD" ] || { echo "FATAL: manifest diff is not exactly the staged deletions"; exit 9; }
if [ "${#PROBE_TOKENS[@]}" -gt 0 ]; then
  echo "### probe-token gate on $PROBES_ARM (want 0 each -- clean BY CONSTRUCTION):"
  PROBE_BAD=0
  for tok in "${PROBE_TOKENS[@]}"; do
    n=$(grep -c "$tok" "$PROBES_ARM")
    printf '  %-24s arm=%s canonical=%s\n' "$tok" "$n" "$(grep -c "$tok" "$PROBES_CANON")"
    [ "$n" = 0 ] || PROBE_BAD=1
  done
  [ "$PROBE_BAD" = 0 ] || { echo "FATAL: a deleted pin's module is still named in $PROBES_ARM -- this reproduces job 5346167's find_spec RED-tierA by construction"; exit 9; }
fi
BACKEND=$SNAP

########## 1. WORKSPACE ws.${TAG}-$J -- FROM SCRATCH, from the canonical tree ##########
if [ ! -e "$WS/.cert-staged" ]; then
  if [ -d "$WS" ]; then
    mv "$WS" "$WS.trash.$$"
    ( chmod -R u+w "$WS.trash.$$" >/dev/null 2>&1; rm -rf "$WS.trash.$$" ) &
    echo "### moved pre-existing $WS aside"
  fi
  mkdir -p "$WS"
  echo "### stage 1/3: rsync small set from $SRC_WS (excl .pixi, third_party, data/log dirs, ALL pixi.lock*)"
  S=$(date +%s)
  rsync -a --info=stats2 \
    --exclude '/.pixi/' --exclude '/third_party/' \
    --exclude '/assets/' --exclude '/groot-sonic-data/' --exclude '/logs/' \
    --exclude '/results/' --exclude '/scratchpad/' --exclude '/scratch_rescue/' \
    --exclude '/.pytest_cache/' --exclude '/pixi.lock' --exclude '/pixi.lock.*' \
    "$SRC_WS/" "$WS/"
  echo "### rsync rc=$? wall=$(( $(date +%s) - S ))s"
  mkdir -p "$WS/.pixi"
  cp "$SRC_WS/.pixi/config.toml" "$WS/.pixi/config.toml"
  echo "### stage 2/3: cp -al third_party (hardlink, read-only share)"
  S=$(date +%s)
  cp -al "$SRC_WS/third_party" "$WS/third_party"
  echo "### cp -al third_party rc=$? wall=$(( $(date +%s) - S ))s"
  echo "### stage 3/3: install the manifest under test as the root manifest"
  cp "$CLEANED" "$WS/pixi.toml"
  touch "$WS/.cert-staged"
fi

# Reuse hygiene: a re-entered workspace can keep .pixi/{meta-v0,scratch-v0,envs}
# from a previous attempt. Move them aside so the WORKSPACE side is cold too.
STAMP=$(date +%s)
for d in meta-v0 scratch-v0 envs; do
  if [ -e "$WS/.pixi/$d" ]; then
    mkdir -p "$A/attic-${TAG}-$J"
    mv "$WS/.pixi/$d" "$A/attic-${TAG}-$J/.pixi-$d.$STAMP"
    echo "### moved stale $WS/.pixi/$d -> $A/attic-${TAG}-$J/.pixi-$d.$STAMP"
  fi
done
if ls "$WS"/pixi.lock* >/dev/null 2>&1; then
  mkdir -p "$A/attic-${TAG}-$J"
  for f in "$WS"/pixi.lock*; do mv "$f" "$A/attic-${TAG}-$J/$(basename "$f").$STAMP"; echo "### moved PARTIAL $f aside"; done
fi
echo "### workspace-local retread state after reset: $(ls -A "$WS/.pixi" | tr '\n' ' ')"

echo "### staged files: $(find "$WS" | wc -l)  size: $(du -sh --exclude=third_party "$WS" | cut -f1) (+third_party hardlinked)"
echo "### .git present: $([ -d "$WS/.git" ] && echo yes || echo NO)  modules: $(ls "$WS/.git/modules" 2>/dev/null | wc -l)"
echo "### manifest md5 (staged vs source):"; md5sum "$WS/pixi.toml" "$CLEANED"
if [ "$(md5sum < "$WS/pixi.toml")" != "$(md5sum < "$CLEANED")" ]; then
  echo "### FATAL: staged manifest is not $CLEANED"; exit 3
fi
echo "### manifest lines: $(wc -l < "$WS/pixi.toml") (want $EXPECT_MANIFEST_LINES)"
echo "### jetson LIVE (uncommented) rows: $(grep -c '^jetson = ' "$WS/pixi.toml") (want $EXPECT_JETSON_ROWS)"
echo "### pixi.lock present (want NO): $(ls "$WS"/pixi.lock* 2>/dev/null | wc -l) file(s)"
rm -f "$WS"/pixi.lock "$WS"/pixi.lock.* 2>/dev/null
echo "### path deps present:"
grep -oE 'path *= *"[^"]+"' "$WS/pixi.toml" | sed 's/.*"\(.*\)"/\1/' | sort -u | \
  while read -r d; do printf '  %-60s %s\n' "$d" "$([ -e "$WS/$d" ] && echo OK || echo MISSING)"; done
echo "### env list from the staged manifest:"
"$PIXI" workspace environment list --manifest-path "$WS/pixi.toml" 2>&1 | tee "$A/${TAG}-$J.envlist.txt" | tail -40
echo "### env count from manifest: $(grep -cE '^- ' "$A/${TAG}-$J.envlist.txt" 2>/dev/null) (want $EXPECT_ENVS; raw lines $(wc -l < "$A/${TAG}-$J.envlist.txt"))"

########## 2. ENV BLOCK -- job-scoped BUILD state, SHARED download+solve caches ##########
for d in pixi rattler uv xdg-cache xdg-data retread-build retread-artifacts \
         retread-meta retread-cache retread-shared pixi-home; do mkdir -p "$C/$d"; done
for d in home tmp scratch fast-tmp xdg-state xdg-config; do mkdir -p "$G/$d"; done
export PIXI_CACHE_DIR=$C/pixi
export RATTLER_CACHE_DIR=$C/rattler
export UV_CACHE_DIR=$C/uv
export XDG_CACHE_HOME=$C/xdg-cache
export XDG_DATA_HOME=$C/xdg-data
export RETREAD_BUILD_ROOT=$C/retread-build
export RETREAD_ARTIFACT_ROOT=$C/retread-artifacts
export RETREAD_META_ROOT=$C/retread-meta
export RETREAD_CACHE_DIR=$C/retread-cache
export RETREAD_SHARED_CACHE_DIR=$C/retread-shared
export PIXI_HOME=$C/pixi-home
export HOME=$G/home
export TMPDIR=$G/tmp
export RETREAD_SCRATCH_ROOT=$G/scratch
export RETREAD_FAST_TMP_ROOT=$G/fast-tmp
export XDG_STATE_HOME=$G/xdg-state
export XDG_CONFIG_HOME=$G/xdg-config
export RETREAD_MAX_CONCURRENT_BUILDS=6
export TOKIO_WORKER_THREADS=8
export RAYON_NUM_THREADS=8
export PATH=/users/glvov/.pixi/bin:$UVBIN:/users/glvov/.local/bin:/usr/bin:/bin
export RETREAD_UV=$UVBIN/uv
[ -x "$RETREAD_UV" ] || { echo "FATAL: RETREAD_UV $RETREAD_UV missing"; exit 7; }
echo "### uv: $RETREAD_UV -> $("$RETREAD_UV" --version 2>&1)  (command -v uv: $(command -v uv))"
export CONDA_OVERRIDE_CUDA=12
export CONDA_OVERRIDE_GLIBC=2.35
export UV_LOCK_TIMEOUT=3600
export UV_LINK_MODE=copy   # job 5547450: NFS hardlink race under concurrent uv builds
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PIXI_BUILD_RETREAD_LOG=$BACKEND_LOG_FILTER
# INSTRUMENTED RUN: the frontend gets RUST_LOG so uv's own targets reach the log
# at all. The BACKEND must not inherit it -- it is a child process and its
# control surface is PIXI_BUILD_RETREAD_LOG (HANDOFF section 1). The shim below
# unsets RUST_LOG before exec'ing it, which is the only place that can.
export RUST_LOG=$FRONTEND_RUST_LOG
export RUST_BACKTRACE=1
echo "### INSTRUMENTATION: frontend RUST_LOG=$RUST_LOG"
echo "### INSTRUMENTATION: frontend verbosity $LOCK_VERBOSITY (pixi 0.73: -vvv = debug)"
echo "### INSTRUMENTATION: backend PIXI_BUILD_RETREAD_LOG=$PIXI_BUILD_RETREAD_LOG (RUST_LOG unset for it)"
[ -f "$STAMPER" ]   || { echo "FATAL: line stamper $STAMPER missing"; exit 8; }
[ -f "$EXTRACTOR" ] || { echo "FATAL: extractor $EXTRACTOR missing"; exit 8; }

# PERSISTENT CACHES -- must come AFTER the job-scoped block above (it overrides
# the three cache dirs) and AFTER RETREAD_FAST_TMP_ROOT + SLURM_JOB_ID exist,
# because the verdict-cache symlink is placed at a path derived from both.
# shellcheck source=/dev/null
. "$FAST_ENV"
retread_fast_env "$WS" || { echo "FATAL: retread_fast_env refused"; exit 7; }
# B2 DELTA (2026-09-02): this run measures WHERE THE TIME GOES on a cold-store
# relock, and it runs concurrently with the B1 store measurement against the same
# manifest. Sharing the store would let a concurrent publish serve this run's
# conda_outputs and delete the very breakdown this harness exists to produce --
# and would equally dirty B1's number. So this run gets its OWN job-scoped store
# root. The variable is still SET, so the store code path is exercised.
export RETREAD_BUILT_OUTPUT_STORE=$C/built-outputs-isolated
mkdir -p "$RETREAD_BUILT_OUTPUT_STORE"
echo "### store isolation: RETREAD_BUILT_OUTPUT_STORE=$RETREAD_BUILT_OUTPUT_STORE (job-scoped, cold by design)"

# Node-local log root: uv_resolver=debug on 27 envs is a multi-GB stream and it
# must not be written straight onto NFS while it grows (2026-08-31: 842 MB of
# warnings to /oscar jammed writeback and took a node out for 9 h). Gzipped into
# $A at the end; if node-local storage is unavailable we fall back to $A and say
# so loudly rather than silently writing GBs to NFS.
L=$LOCAL_LOG_ROOT-$J
if mkdir -p "$L" 2>/dev/null && [ -w "$L" ]; then
  echo "### instrumented logs are written node-local: $L (gzipped into $A at the end)"
else
  L=$A
  echo "### WARNING: no node-local log root; writing multi-GB logs straight to NFS at $A"
fi

# backend stderr shim (pixi 0.73 swallows backend stderr behind its expect() panic)
BLOG=$L/${TAG}-$J.backend.log
: > "$BLOG"
SHIM=$A/${TAG}-$J.backend-shim.sh
cat > "$SHIM" <<SHIMEOF
#!/usr/bin/env bash
unset RUST_LOG
exec 2> >(tee -a "$BLOG" >&2)
exec "$BACKEND" "\$@"
SHIMEOF
chmod +x "$SHIM"
export PIXI_BUILD_BACKEND_OVERRIDE="pixi-build-retread=$SHIM"
echo "### backend shim: $SHIM -> $BACKEND ; stderr tee -> $BLOG"
echo "### pixi.real --version: $($PIXI --version)"
echo "### PIXI_BUILD_BACKEND_OVERRIDE=$PIXI_BUILD_BACKEND_OVERRIDE"
env | grep -E '^(HOME|PIXI_|RATTLER_|UV_|XDG_|TMPDIR|RETREAD_|CONDA_OVERRIDE)' | sort
echo "### persistent cache sizes BEFORE the lock:"
du -sh "${RETREAD_PERSIST_CACHE_ROOT:-/oscar/data/stellex/glvov/agrescap/cache/retread}"/* 2>/dev/null

########## 3. LOCK ($EXPECT_ENVS envs, no pre-existing pixi.lock) ##########
cd "$WS" || exit 5
LLOG=$L/${TAG}-$J.lock.log
LTIME=$A/${TAG}-$J.lock.time.txt
echo "### lock start $(date -Is)"
S=$(date +%s)
# The frontend's own rows carry NO timestamp -- only the backend's do. Every row
# is stamped at the pipe so the uv tracing can be put on a timeline at all, and
# the lock's rc comes from PIPESTATUS[0]: a pipeline reports the LAST stage's rc
# and that has misread a run on this campaign before.
/usr/bin/time -v -o "$LTIME" "$PIXI" lock $LOCK_VERBOSITY 2>&1 \
  | python3 -u "$STAMPER" > "$LLOG"
LRC=${PIPESTATUS[0]}
LW=$(( $(date +%s) - S ))
echo "### lock rc=$LRC wall=${LW}s end $(date -Is)"
echo "$LRC" > "$A/${TAG}-$J.rc"; echo "$LW" > "$A/${TAG}-$J.wall"
echo "### /usr/bin/time -v (lock) -> $LTIME"
grep -E 'Elapsed \(wall|Maximum resident set size|User time|System time|Percent of CPU' "$LTIME" | sed 's/^/  /'
LRSS=$(awk -F': ' '/Maximum resident set size/{print $2}' "$LTIME")
echo "### lock peak RSS: ${LRSS:-UNKNOWN} kbytes  (72G request = $( [ -n "$LRSS" ] && awk -v r="$LRSS" 'BEGIN{printf "%.0fx", 72*1024*1024/r}' || echo '?') headroom)"
echo "### persistent cache sizes AFTER the lock:"
du -sh "${RETREAD_PERSIST_CACHE_ROOT:-/oscar/data/stellex/glvov/agrescap/cache/retread}"/* 2>/dev/null

########## 4. FIRST-CUT EVIDENCE ##########
if [ -f "$WS/pixi.lock" ]; then
  cp "$WS/pixi.lock" "$A/pixi.lock.cert"
  echo "### pixi.lock.cert saved: $(stat -c%s "$A/pixi.lock.cert") bytes  md5: $(md5sum < "$A/pixi.lock.cert")"
  echo "### envs in the produced lock (want $EXPECT_ENVS): $(awk '/^environments:/{f=1;next} f&&/^[a-z]/{exit} f&&/^  [A-Za-z0-9][A-Za-z0-9._-]*:$/{c++} END{print c+0}' "$A/pixi.lock.cert")"
  echo "### jetson env in the produced lock: $(awk '/^environments:/{f=1;next} f&&/^[a-z]/{exit} f&&/^  jetson:$/{c++} END{print c+0}' "$A/pixi.lock.cert") (want $EXPECT_JETSON_ROWS)"
  # Name/url sets -- the cheap identity instrument for comparing two locks.
  # Extraction copied verbatim from the A/B/C harness of job 5598763, which is
  # the run these four counts were validated against (pypi names 174, conda
  # names 1707, pypi urls 213, conda urls 2584 on the canonical manifest).
  LK=$A/pixi.lock.cert
  grep -aoE '^\s+- pypi: \S+'  "$LK" | awk '{print $3}' | sort -u > "$A/${TAG}-$J.pypi-urls.txt"
  grep -aoE '^\s+- conda: \S+' "$LK" | awk '{print $3}' | sort -u > "$A/${TAG}-$J.conda-urls.txt"
  grep -aoE '^- pypi: \S+'  "$LK" | sed 's|.*/||; s|-[0-9].*||'      | grep -v '^$' | sort -u > "$A/${TAG}-$J.pypi-names.txt"
  grep -aoE '^- conda: \S+' "$LK" | sed 's|.*/||; s|-[^-]*-[^-]*$||' | grep -v '^$' | sort -u > "$A/${TAG}-$J.conda-names.txt"
  echo "### name/url sets: pypi names=$(wc -l < "$A/${TAG}-$J.pypi-names.txt") conda names=$(wc -l < "$A/${TAG}-$J.conda-names.txt") pypi urls=$(wc -l < "$A/${TAG}-$J.pypi-urls.txt") conda urls=$(wc -l < "$A/${TAG}-$J.conda-urls.txt")"
else
  echo "### NO pixi.lock produced"
fi
echo "### COUNTERS (all must be 0):"
for pat in 'retread rpc error' 'courier inputs changed' '0 exact matches' \
           'run dependencies differ' 'panicked'; do
  printf '  %-28s lock.log=%s backend.log=%s\n' "$pat" \
    "$(grep -c "$pat" "$LLOG" 2>/dev/null)" "$(grep -c "$pat" "$BLOG" 2>/dev/null)"
done
echo "### POSITIVE SIGNALS:"
for pat in 'advertised identity: loaded' 'ownership: name=' 'owner=env-pypi' \
           'pypi-declared' 'native provider:' 'declared-pypi bound check'; do
  printf '  %-28s lock.log=%s backend.log=%s\n' "$pat" \
    "$(grep -c "$pat" "$LLOG" 2>/dev/null)" "$(grep -c "$pat" "$BLOG" 2>/dev/null)"
done
echo "### ROUTE PROBE CACHE (a WARM run executes a handful of probes, a cold one ~315):"
for pat in 'route probe cache: hit' 'route probe cache: opened' 'route probe'; do
  printf '  %-32s lock.log=%s backend.log=%s\n' "$pat" \
    "$(grep -c "$pat" "$LLOG" 2>/dev/null)" "$(grep -c "$pat" "$BLOG" 2>/dev/null)"
done
echo "  probes EXECUTED (sum of probes= on the finished spans): $(grep -oE 'bundle route probes finished[^\n]*probes=[0-9]+' "$BLOG" 2>/dev/null | grep -oE 'probes=[0-9]+' | awk -F= '{s+=$2} END{print s+0}')"
grep -nE 'route probe cache' "$BLOG" 2>/dev/null | head -10
echo "### learned-conda-fact YIELDS (WARN expected, non-fatal):"
printf '  learned-fact-yield count: lock.log=%s backend.log=%s\n' \
  "$(grep -c 'learned conda fact.*yield' "$LLOG" 2>/dev/null)" "$(grep -c 'learned conda fact.*yield' "$BLOG" 2>/dev/null)"
echo "### uv closure pass B failures (want 0):"
printf '  %-32s lock.log=%s backend.log=%s\n' 'uv closure pass B failed' \
  "$(grep -c 'uv closure pass B failed' "$LLOG" 2>/dev/null)" "$(grep -c 'uv closure pass B failed' "$BLOG" 2>/dev/null)"
echo "### WARNs (should be 0 -- overrides exported):"
grep -nE 'WARN.*(CONDA_OVERRIDE|virtual package|auto-set)' "$LLOG" "$BLOG" 2>/dev/null | head -10
echo "### backend.log bytes: $(stat -c%s "$BLOG" 2>/dev/null)"
echo "### backend.log first ERROR/panic lines:"
grep -nE 'ERROR|panic|error:|thread .* panicked' "$BLOG" 2>/dev/null | head -20
echo "### backend.log tail:"; tail -40 "$BLOG" 2>/dev/null
echo "### lock.log tail:"; tail -40 "$LLOG" 2>/dev/null

########## 4b. INSTRUMENTATION HARVEST -- the reason this run exists ##########
echo "### uv tracing rows actually captured (0 here means the run was NOT instrumented):"
for t in uv_resolver uv_client uv_distribution pixi_uv_context pixi_uv_conversions resolvo; do
  printf '  %-22s lock.log=%s backend.log=%s\n' "$t" \
    "$(grep -c "$t" "$LLOG" 2>/dev/null)" "$(grep -c "$t" "$BLOG" 2>/dev/null)"
done
echo "### raw log sizes: lock=$(stat -c%s "$LLOG" 2>/dev/null) backend=$(stat -c%s "$BLOG" 2>/dev/null) bytes"
echo "### extractor: $EXTRACTOR"
python3 "$EXTRACTOR" --lock "$LLOG" --backend "$BLOG" \
   --out "$A/${TAG}-$J.extract.txt" 2>&1 | tail -80
echo "### extract written: $A/${TAG}-$J.extract.txt ($(stat -c%s "$A/${TAG}-$J.extract.txt" 2>/dev/null) bytes)"
echo "### gzipping the raw logs into $A (they are the evidence; the extract is only a reading of them)"
for f in "$LLOG" "$BLOG"; do
  [ -f "$f" ] || continue
  gzip -c "$f" > "$A/$(basename "$f").gz" && echo "  $(basename "$f").gz $(stat -c%s "$A/$(basename "$f").gz") bytes"
done
# The node-local originals are removed here ON PURPOSE: they live on the node's
# own disk and nothing else will ever collect them, and this is not the NFS
# rm -rf that section 7 forbids.
if [ "$L" != "$A" ]; then rm -f "$LLOG" "$BLOG"; rmdir "$L" 2>/dev/null; fi

########## 5. TWO CHECKS on every deleted pin ##########
echo "### CHECK 1/2 -- per-env per-package VERSION delta (primary) + occurrence delta (secondary)"
if [ -f "$A/pixi.lock.cert" ]; then
  if [ -x "$EVD" ]; then
    "$EVD" "$BASE_LOCK" "$A/pixi.lock.cert" $EVD_PACKAGES 2>&1 | sed 's/^/  /'
  else
    echo "  MISSING $EVD -- occurrence counts below are the only CHECK-1 evidence, and they are BLIND to same-count-different-version"
  fi
  echo "  --- $ATTRIB (shared-pin drift) ---"
  "$ATTRIB" "$A/pixi.lock.cert" 2>&1 | sed 's/^/  /'
  echo "  --- deleted-pin families: whole-file occurrence counts (secondary; blind by design) ---"
  for p in $EVD_PACKAGES; do
    printf '  %-16s baseline=%s arm=%s\n' "$p" \
      "$(grep -cE "/${p}-[0-9]|name: ${p}$" "$BASE_LOCK" 2>/dev/null)" \
      "$(grep -cE "/${p}-[0-9]|name: ${p}$" "$A/pixi.lock.cert" 2>/dev/null)"
    echo "    baseline versions: $(grep -oE "/${p}-[0-9][^/]*" "$BASE_LOCK" 2>/dev/null | sort -u | tr '\n' ' ')"
    echo "    arm      versions: $(grep -oE "/${p}-[0-9][^/]*" "$A/pixi.lock.cert" 2>/dev/null | sort -u | tr '\n' ' ')"
  done
  echo "  --- watched packages, NOT touched by this batch ---"
  for p in $WATCH_PACKAGES; do
    printf '  %-16s baseline=%s arm=%s\n' "$p" \
      "$(grep -cE "/${p}-[0-9]|name: ${p}$" "$BASE_LOCK" 2>/dev/null)" \
      "$(grep -cE "/${p}-[0-9]|name: ${p}$" "$A/pixi.lock.cert" 2>/dev/null)"
  done
else
  echo "  SKIPPED: no pixi.lock.cert produced (lock rc=$LRC)"
fi
echo "### CHECK 2/2 -- probes grep"
echo "  canonical $PROBES_CANON (untouched, operator-gated)"
echo "  arm copy  $PROBES_ARM"
diff "$PROBES_CANON" "$PROBES_ARM" | sed 's/^/  /'

########## 6. HANDOFF TO THE CERT PHASE ##########
if [ "$LRC" = 0 ] && [ -f "$A/pixi.lock.cert" ]; then
  {
    echo "# written by phaseN_relock.sh (${TAG}) job $J $(date -Is)"
    echo "P1_JOB=$J"
    echo "WS=$WS"
    echo "P1_CACHE_ROOT=$C"
    echo "LOCK=$A/pixi.lock.cert"
    echo "EXPECT_LOCK_MD5=$(md5sum < "$A/pixi.lock.cert" | awk '{print $1}')"
  } > "$A/relock_env.sh"
  echo "### cert-phase handoff written:"; cat "$A/relock_env.sh"
else
  echo "### NO cert-phase handoff written (lock rc=$LRC) -- the afterok dependency will not release"
fi

########## 7. NO SELF-CLEANUP HERE -- ON PURPOSE ##########
# See the EVIDENCE header: an `rm -rf` of a job-scoped root on the afterok path
# cost job 5596128 5152s of held QOS. The cert phase submits cleanup.sh with
# --dependency=afterany and exits; that job removes $C and $WS.
echo "### self-cleanup NOT run here by design -- roots left for the cert phase and cleanup.sh: $C $WS"
echo "### inode quota AFTER:"; "$CQ" 2>/dev/null | grep -E 'data\+stellex' | head -2
echo "### ${TAG} RELOCK DONE lock_rc=$LRC wall=${LW}s peak_rss_kb=${LRSS:-unknown} $(date -Is)"
exit "$LRC"
