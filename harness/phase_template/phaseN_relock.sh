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
#   4. STAGING IS HARDLINKED OUT OF A PERSISTENT MIRROR, not rsync'd per job.
#      Every relock used to copy the "small set" out of imprint-data before it
#      could solve anything -- 9,175 regular files / 4,910 dirs / 62,261,385,682
#      bytes -- measured at 422s (job 5611846), 534s (5650823) and 572s
#      (5655631), plus a `cp -al third_party` at 212-254s. 12-14 minutes of pure
#      harness overhead in front of a lock that finishes in 69s warm.
#
#      WHAT THAT 62 GB IS (census of imprint-data, job 5658374):
#          pypi-packs      1,370 files   52.31 GB   the local path-source packs
#          .git            6,207 files    9.93 GB   never opened by the lock
#          everything else ~1,600 files    0.04 GB
#
#      WHAT THE LOCK ACTUALLY READS. This filesystem mounts relatime, so on a
#      workspace rsync -a staged (atime = staging, mtime = the source's old
#      mtime) the first read moves atime, and `find -printf '%A@ %T@'` after the
#      lock is a read-set detector. Job 5650823's workspace: 254 files /
#      2.31 GB, every one of them under pypi-packs, plus pixi.toml and
#      .pixi/config.toml (created by `cp`, so atime==mtime and relatime hides
#      them). ZERO reads in .git, src, test, docs, humble_ws, jazzy_ws,
#      packages, patches, plans, scripts, step_back, tools, wbc_push.
#
#      So the copy is ~27x the bytes the lock opens, and the mirror turns the
#      per-job cost into `cp -al`, which writes directory entries, not bytes.
#      Timings and the equivalence proof: LANE-SPEED-LOG.md, "staging lever".
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
#
# ---- EXACTLY ONE CLEANUP OWNER PER ROOT (measured 2026-09-04, tag AFINAL2) ---
#   The roots /oscar/data/stellex/glvov/retread/certAFINAL2-5769426 and
#   ws.AFINAL2-5769426 were cleaned by TWO jobs at once: the dispatch-time
#   gated cleanup 5770508 and the cert phase's OWN self-submitted cleanup
#   5776646. Both released on the same dependency, both started 08:39:44 on
#   node2343, and both walked the same trees. Two concurrent `rm -rf` walks of
#   one tree unlink entries out from under each other: BOTH returned rc=1 with
#   pages of "Directory not empty" (5776646 also logged `rm: fts_read failed:
#   Stale file handle`), both logged `exists_after=YES`, and 590028 + 668715
#   entries were LEFT ON DISK after 2864 s and 3941 s of wall.
#   Nothing on disk told the cert phase that an owner already existed. Section 6
#   now records one: `CLEANUP_JOB=` in the handoff stamp names the dispatch
#   cleanup's job id, and phaseN_cert.sh's `cleanup_owner` defers to it instead
#   of submitting a second owner. Empty means nobody owns them and the cert
#   phase submits and owns exactly one, as it always did.
#   Guarded by phase_template/cleanup_owner_guard.sh.
# NEVER edit this file while a job is running it -- copy it aside first.
### EVIDENCE END
set -uo pipefail

### SUBSTITUTE: BEGIN -- MANIFEST, PROBES, EXPECT_*  (edit ONLY between these markers)
# Every campaign-specific constant in this harness lives here. Nothing below
# this block names a previous batch; the self-check right after it enforces that.

TAG=PHASEN                                   # short batch tag; roots become certPHASEN-<job> / ws.PHASEN-<job>
T=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11
D=$T/phase-template-example                  # THIS harness's own directory (artifacts land in $D/artifacts)

# --- the manifest under test -------------------------------------------------
SRC_WS=/oscar/data/stellex/glvov/imprint-data           # READ-ONLY canonical source tree
CLEANED=$T/b1-scratch/pixi.toml.EXAMPLE                 # the scratch manifest this batch locks
EXPECT_CLEANED_MD5=00000000000000000000000000000000     # md5sum of $CLEANED
EXPECT_MANIFEST_LINES=1003                              # wc -l of $CLEANED
EXPECT_DEL=0                                            # diff SRC_WS/pixi.toml CLEANED : '< ' lines
EXPECT_ADD=0                                            # diff SRC_WS/pixi.toml CLEANED : '> ' lines
EXPECT_ENVS=27                                          # envs the manifest declares AND the lock must carry
EXPECT_JETSON_ROWS=1                                    # live `jetson = ` rows (0 disables the jetson env)

# --- residual-pin gate: one pattern per deleted pin family, each must be 0 ----
RESIDUAL_PATTERNS=()                                    # e.g. ('^openmesh = ' '^pillow = "==10.4.0"')

# --- probes ------------------------------------------------------------------
PROBES_CANON=$T/p1e-certify-lock/artifacts/probes.tsv    # canonical, operator-gated, NEVER edited
PROBES_ARM=$PROBES_CANON                                 # this batch's copy; point it at a corrected copy
                                                         # whenever a deleted pin also names a probe module
PROBE_TOKENS=()                                          # module tokens that must be GONE from $PROBES_ARM

# --- instruments -------------------------------------------------------------
ATTRIB=$T/tools/b2_attribute.sh                          # whole-file occurrence delta (secondary, blind by design)
EVD=$T/b3-phase1/env_version_delta.py                    # PER-ENV PER-PACKAGE version delta (primary CHECK 1)
EVD_PACKAGES="openmesh networkx pillow sentry-sdk numpy" # packages CHECK 1 adjudicates
WATCH_PACKAGES="gxx_linux-64 cmake"                      # observed, not touched by this batch
BASE_LOCK=$SRC_WS/pixi.lock                              # baseline for the occurrence delta

# --- toolchain ---------------------------------------------------------------
PIXI=/users/glvov/.pixi/bin/pixi.real                    # bypass the flock shim
SNAP=$T/p4l-cert-p4k/artifacts/p4k-binsnap/pixi-build-retread
# OPTIONAL pin. Leave EMPTY and the gate DERIVES the sha from $SNAP at run
# time. Set it only to assert a specific binary, and then it MUST match.
# It used to be a mandatory second constant beside SNAP, and on 2026-09-03 a
# derivation substituted SNAP and not it, so job 5671529 died exit 8 in 3 s
# ("snapshot sha 1860e830... != 2dd790bf..."). The leftover-token self-check
# cannot see that: both values live INSIDE this SUBSTITUTE region, which the
# check strips by design. One constant cannot disagree with itself.
EXPECT_SHA_PIN=
UVBIN=/oscar/data/stellex/glvov/tasks/retread-cold-solve/verify_fixes/artifacts/uvbin
FAST_ENV=$(dirname "$0")/../retread_fast_env.sh          # persistent caches; fallback below
[ -f "$FAST_ENV" ] || FAST_ENV=$T/tools/retread_fast_env.sh

# --- leftover-token self-check ------------------------------------------------
# Names of PREVIOUS batches. A hit anywhere outside the three marked regions is
# a botched derivation, which is what HANDOFF section 2's grep exists to catch.
LEFTOVER_RE='bfinal|BFP1|BFP2|bfp1|bfp2|b1c|b1-phase|b1b-phase|b2-phase|b2b-phase|b3-phase|ctl-phase|eff-phase|/b1_|/b2_|/b3_|/ctl_|p5sab|P5SAB|p5t_abc|P5TABC|certB3P1|2cfec88d|57105d38'
### SUBSTITUTE: END

### LEFTOVER-CHECK BEGIN
# Strips the three marked regions (this one included) and fails on any survivor.
# Comments are NOT exempt: a stale path in a comment has misled a reader on this
# campaign before. Deliberate evidence citations belong in the EVIDENCE region,
# or -- MERGE-N-4, when the citation has to sit beside the code it explains --
# between a `### CITATION BEGIN` / `### CITATION END` pair, which is stripped
# exactly like the other three. The pair is a DELIBERATE, per-site opt-out: a
# botched derivation never carries one, so the check still catches every real
# leftover. Blanket-exempting COMMENTS was rejected -- a stale path in a comment
# is the defect this check was written for.
# The match runs INSIDE awk, on the LINE, never on "FILENAME:LNO: line". Piping
# the annotated text to grep made the check match its own FILENAME: a harness in
# a directory named after a previous batch failed against itself, on every line,
# with the tokens nowhere in its body. A scan must not be able to match itself.
LEFT=$(awk '
  /^### EVIDENCE BEGIN/       {e=1} /^### EVIDENCE END/       {e=0; next} e {next}
  /^### SUBSTITUTE: BEGIN/    {s=1} /^### SUBSTITUTE: END/    {s=0; next} s {next}
  /^### LEFTOVER-CHECK BEGIN/ {l=1} /^### LEFTOVER-CHECK END/ {l=0; next} l {next}
  /^### CITATION BEGIN/       {c=1} /^### CITATION END/       {c=0; next} c {next}
  $0 ~ re {print FILENAME ":" FNR ": " $0}' re="$LEFTOVER_RE" "$0")
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
[ -n "$GOT_SHA" ] || { echo "FATAL: could not sha256sum $SNAP"; exit 8; }
if [ -n "$EXPECT_SHA_PIN" ]; then
  [ "$GOT_SHA" = "$EXPECT_SHA_PIN" ] || { echo "FATAL: snapshot sha $GOT_SHA != pinned $EXPECT_SHA_PIN"; exit 8; }
  echo "### backend snapshot sha PINNED and matched"
else
  echo "### backend snapshot sha DERIVED from \$SNAP at run time (no pin set)"
fi
EXPECT_SHA=$GOT_SHA
echo "### backend snapshot OK: $SNAP sha256=$GOT_SHA"
ls -l "$SNAP"; "$SNAP" --version 2>&1 | head -2
[ -f "$FAST_ENV" ] || { echo "FATAL: persistent-cache snippet $FAST_ENV missing"; exit 8; }

### HARNESS-DRIFT BEGIN
# HARNESS-SYNC-1 (2026-09-06).  THE FIRST QUESTION IS NOT "IS THIS THE RIGHT
# COMMIT" BUT "HAS ANYBODY HAND-EDITED A TASK COPY SINCE THE LAST SYNC".  The
# drift check below can only say "not $HARNESS_COMMIT"; `harness_sync.sh --check`
# names the FILE, because it compares against the commit the SINGLE WRITER last
# installed (`tools/.harness_synced_commit`) -- so a direct edit is named here,
# by path, with the command that fixes it, instead of killing a job three hours
# later for "drift".  No writer, or no record yet = ANNOUNCED, never silent.
SYNC_CHECK=$(dirname "$FAST_ENV")/harness_sync.sh
if [ -f "$SYNC_CHECK" ]; then
  SYNC_OUT=$(HARNESS_TASK_DIR=$T bash "$SYNC_CHECK" --check 2>&1); SYNC_RC=$?
  echo "$SYNC_OUT"
  if [ "$SYNC_RC" -eq 3 ]; then
    echo "FATAL: a task copy named above was EDITED IN PLACE, not synced."
    echo "       Commit it in the harness repo, then run: harness_sync.sh <commit>"
    exit 6
  fi
  [ "$SYNC_RC" -eq 0 ] || echo "### harness_sync.sh --check inconclusive (rc=$SYNC_RC) -- the drift check below still runs"
else
  echo "### harness_sync.sh missing next to $FAST_ENV -- in-place-edit check OFF for this run"
fi
# C31-4-1d.  The task tree is not a git repo, so a task copy of this harness can
# silently fall behind the versioned one -- C31-4-1 found four that had, one of
# them 779 lines behind.  Set HARNESS_COMMIT to the harness commit this batch is
# meant to be, and a stale copy REFUSES here in milliseconds instead of
# certifying the wrong harness three hours from now.  Unset = the check is
# announced as OFF; it is never silently skipped.
# MERGE-N-2.  The pin used to arrive ONLY through
# `sbatch --export=ALL,HARNESS_COMMIT=<sha>`, and that clause put jobs into
# `launch_failed_requeued_held` / "user env retrieval failed" in TWO CONSECUTIVE
# merge lanes -- Slurm re-runs the submitter's login environment to build `ALL`.
# The job never starts, so nothing in ITS log can say why.  The pin is now read
# from `$D/HARNESS_COMMIT`, a file the job OWNS, written by the submitter beside
# the `artifacts/` this run already writes to; the export is kept as the FALLBACK,
# so every existing caller keeps working and a submit with NO --export at all
# still reaches this check.  A file and an export that DISAGREE refuse, naming
# both -- see tools/harness_commit_resolve.sh.
HC_RESOLVE=$(dirname "$FAST_ENV")/harness_commit_resolve.sh
if [ -f "$HC_RESOLVE" ]; then
  # HARNESS-SYNC-2-1: the resolver has TWO refusals now -- rc 2 (the file and
  # the export disagree) and rc 4 (PIN STALE: the pin is not the commit the task
  # copies ARE).  Name the rc so this FATAL cannot mis-state which one fired.
  HARNESS_COMMIT=$(bash "$HC_RESOLVE" "$D"); HC_RC=$?
  [ "$HC_RC" -eq 0 ] || {
    echo "FATAL harness commit pin REFUSED rc=$HC_RC -- see the resolver line above (2 = the file and the export disagree, 4 = PIN STALE)"; exit 6; }
else
  echo "### harness_commit_resolve.sh missing next to $FAST_ENV -- falling back to the exported pin"
  HARNESS_COMMIT="${HARNESS_COMMIT:-}"
fi
if [ -n "$HARNESS_COMMIT" ]; then
  echo "### HARNESS_COMMIT=$HARNESS_COMMIT -- checking the task-dir harness against it"
  DRIFT_CHECK=$(dirname "$FAST_ENV")/harness_drift_check.sh
  [ -f "$DRIFT_CHECK" ] || { echo "FATAL: harness_drift_check.sh missing next to $FAST_ENV"; exit 6; }
  bash "$DRIFT_CHECK" "$HARNESS_COMMIT" || {
    echo "FATAL: harness drift check REFUSED -- the task-dir harness is not $HARNESS_COMMIT"; exit 6; }
else
  echo "### HARNESS_COMMIT unset -- harness drift check OFF for this run"
fi
### HARNESS-DRIFT END
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
# STAGING -- two paths, chosen by STAGE_METHOD just below.
#
#   STAGE_METHOD=mirror (default)
#       A PERSISTENT read-only stage mirror at $STAGE_MIRROR_ROOT/<key>/, keyed
#       on md5($SRC_WS/pixi.toml) + the source tree's git HEAD. rsync'd out of
#       $SRC_WS ONCE per key; every later job pays only `cp -al` of the mirror,
#       which writes directory entries, not bytes. Guards: mirror absent or key
#       mismatch -> rebuild it; rebuild fails -> fall back to the rsync path.
#   STAGE_METHOD=rsync
#       the pre-p12 path, kept selectable and byte-identical to what it was.
#
# SAFETY. $SRC_WS is READ-ONLY and the mirror is hardlink-shared across jobs, so
# nothing the lock WRITES may be a live hardlink into either. Measured write set
# inside a workspace (same job-5650823 comparison, mtime newer than staging):
#   $WS/pixi.lock ; $WS/.pixi/** ; and inside each $WS/pypi-packs/<pack>/ the
#   sidecars retread-probe-trace-*.json, retread-audit-*.json,
#   retread-progress-*.log, retread-*.target-*.lock.json (all four came back
#   with CHANGED SIZES), a handful of .retread-wheel-fetch/ entries, and 230
#   brand-new files under .retread-source-wheels/ and .retread-autodata/.
# stage_break_links() gives every pre-existing pack sidecar and every
# .retread-wheel-fetch/.retread-source-wheels entry its own inode before the
# lock starts; stage_verify_mirror() re-walks the mirror manifest afterwards, so
# a write that escaped the break-list is caught rather than silently poisoning
# the mirror for the next batch. That pair is the reader for this writer.
#
# 2026-09-03. That was NOT ENOUGH, twice over, and both halves are now fixed.
# (i) stage_build_mirror `cp -al`'d $SRC_WS/third_party into the mirror, so the
#     mirror was not a copy of imprint-data, it WAS imprint-data, and every
#     workspace hardlinked out of it wrote straight into the read-only canonical
#     tree. The mirror is built with rsync now, and stage_assert_mirror_disjoint
#     samples 50 files and refuses a mirror that shares an inode with $SRC_WS.
# (ii) stage_break_links PRUNED third_party, and third_party is where the
#     *.egg-info/*.txt files setuptools rewrites in place live. It no longer
#     prunes it; STAGE_TP_WRITABLE is the pattern list.
# The verify now runs BEFORE the lock as well as after, src_tp_fingerprint reads
# $SRC_WS directly across the lock, and a job that trips either one exits 12 no
# matter what the lock returned -- the old code printed FATAL-CLASS and exited 0.
STAGE_METHOD=mirror
STAGE_MIRROR_ROOT=/oscar/data/stellex/glvov/agrescap/cache/retread/stage-mirror
# PROOF-SMOKE-1-7: the ONE inode-disjointness authority, sourced by this file and
# by phase_template/phaseN_relock.sh. Two implementations of "does this tree
# share inodes with imprint-data" is how a publish gated on 50 sampled files
# handed a mirror to a reader that judged it by a different rule (job 6014471).
STAGE_MIRROR_LIB=$(dirname -- "${BASH_SOURCE[0]}")/stage_mirror.sh
# the task layout is $T/tools/phase_template beside $T/tools, the repo layout is
# harness/phase_template beside harness/tools -- both spellings, and neither guessed.
[ -f "$STAGE_MIRROR_LIB" ] || STAGE_MIRROR_LIB=$(dirname -- "${BASH_SOURCE[0]}")/../stage_mirror.sh
[ -f "$STAGE_MIRROR_LIB" ] || STAGE_MIRROR_LIB=$(dirname -- "${BASH_SOURCE[0]}")/../tools/stage_mirror.sh
[ -f "$STAGE_MIRROR_LIB" ] || STAGE_MIRROR_LIB=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/tools/stage_mirror.sh
if [ -f "$STAGE_MIRROR_LIB" ]; then
  . "$STAGE_MIRROR_LIB"
else
  echo "### stage: FATAL -- stage_mirror.sh is missing, and it is the only thing that"
  echo "###        decides whether a mirror is hardlinked into the canonical tree."
  exit 14
fi
STAGE_PAR=16          # cp -al here is NFS-RPC-latency bound, not CPU bound
STAGE_RSYNC_EXCLUDES=( --exclude '/.pixi/' --exclude '/third_party/'
  --exclude '/assets/' --exclude '/groot-sonic-data/' --exclude '/logs/'
  --exclude '/results/' --exclude '/scratchpad/' --exclude '/scratch_rescue/'
  --exclude '/.pytest_cache/' --exclude '/pixi.lock' --exclude '/pixi.lock.*' )

# 2026-09-03: THE WRITE-SET UNDER third_party, derived empirically rather than
# guessed. On job 5673296's finished mirror-staged workspace,
#     find $WS -type f -links +1 -newer $WS/.cert-staged
# returned 157 files. Exactly SEVEN of them still shared an inode with the stage
# mirror -- and, because the mirror itself was `cp -al`'d out of $SRC_WS, with
# /oscar/data/stellex/glvov/imprint-data as well (links=40 at the time of
# measurement, i.e. 40 trees on one inode):
#     third_party/ProtoMotions/protomotions.egg-info/{SOURCES,top_level,dependency_links}.txt
#     third_party/pace-sim2real/source/pace_sim2real/pace_sim2real.egg-info/{SOURCES,top_level,requires,dependency_links}.txt
# setuptools `egg_info` rewrites those in place whenever the editable path
# dependencies are built. The other 150 are .pixi/bld and pypi-packs wheels
# hardlinked into the SHARED WHEEL STORE, not the mirror, by temp-file-plus-
# rename writers -- those are safe and must stay shared, they are the payload.
# The pattern list below is deliberately wider than the measured seven, because
# egg-link / dist-info / __pycache__ are the same class of in-place
# setuptools/pip write; `-size -1M` keeps it from ever copying a payload blob.
# (Measured on imprint-data: third_party is 25,178 files / 10.72 GB, of which
# 18,930 / 140.7 MB are under 64 KiB. Breaking links for ALL small files was
# considered and rejected: 18,930 cp+mv round trips over NFS per job, against a
# measured write-set of 7. The mirror-is-a-real-copy fix below is what makes
# imprint-data unreachable; this list is the second layer, and it also protects
# the mirror on the rsync fallback path.)
# -size -1048576c, NOT -size -1M: find rounds -size up to whole units, so
# `-size -1M` means "rounds to less than one megabyte-block" and matches NOTHING
# at all. The first cut of this list used -1M and the guard test caught it.
STAGE_TP_WRITABLE=( -size -1048576c '(' -path '*.egg-info/*' -o -name '*.egg-link'
  -o -path '*.dist-info/*' -o -path '*/__pycache__/*' -o -name '*.pth' ')' )
MIRROR_DIRTY=0        # set by stage_verify_mirror; forces a non-zero job exit
SRC_WRITTEN=0         # set by the direct reader on $SRC_WS; same

stage_key () {   # md5(manifest) + git HEAD of the SOURCE tree, not of $CLEANED:
  printf '%s %s' \
    "$(md5sum "$SRC_WS/pixi.toml" | awk '{print $1}')" \
    "$(git -C "$SRC_WS" rev-parse HEAD 2>/dev/null || echo nogit)" \
  | md5sum | awk '{print $1}'
}

stage_rsync_path () {            # the pre-p12 path, unchanged
  echo "### stage(rsync) 1/2: rsync small set from $SRC_WS"
  local S=$(date +%s)
  rsync -a --info=stats2 "${STAGE_RSYNC_EXCLUDES[@]}" "$SRC_WS/" "$WS/"
  echo "### rsync rc=$? wall=$(( $(date +%s) - S ))s"
  echo "### stage(rsync) 2/2: cp -al third_party (hardlink, read-only share)"
  S=$(date +%s)
  cp -al "$SRC_WS/third_party" "$WS/third_party"
  echo "### cp -al third_party rc=$? wall=$(( $(date +%s) - S ))s"
}

stage_manifest () {              # what the mirror holds, minus its own two stamp files.
  # -mindepth 1 drops the mirror root, whose mtime moves whenever a stamp file
  # is written; -F because no real path here contains ".stage-mirror-".
  # LC_ALL=C ON THE SORT IS LOAD-BEARING, NOT HYGIENE. This census is written
  # once by the building job and re-walked later by a DIFFERENT job, and the two
  # are compared with `diff`. glibc's en_US.UTF-8 collation ignores `_`, `-` and
  # case at the primary level, so two jobs that inherited different locales sort
  # the same file set into different orders and the diff is non-empty on a tree
  # nothing touched. That is what `ml1` 5752248 did: a FALSE FATAL exit 12 that
  # quarantined the shared stage mirror and cost the next job 459 s / 62 GB of
  # re-staging, while both censuses diff to 0 lines once C-sorted. Every writer
  # AND every reader of this census pins LC_ALL=C; the pin is per-command so it
  # is greppable and cannot be lost when the function is moved.
  # Reader: phase_template/census_collation_guard.sh.
  find "$1" -mindepth 1 -xdev -printf '%y\t%s\t%T@\t%P\n' | grep -vF '.stage-mirror-' | LC_ALL=C sort
}

stage_build_mirror () {          # ONE-TIME per key. Returns non-zero on failure.
  # DET-1-6-b, the same collision from the other end: the temp used to be
  # `$1.building.$J`, one name per JOB, and a job runs several arms. Arm 2's
  # `rm -rf` on that name deletes arm 1's half-built mirror -- 10.72 GB of real
  # bytes -- and the two then race into one rename. The name now carries the arm
  # tag and THIS PROCESS's pid, so it cannot be another arm's, and the cleanup
  # path names the temp this call actually created (STAGE_BUILD_TMP) instead of
  # reconstructing a name that might be someone else's.
  local m=$1 key=$2 arm b
  arm=${STAGE_QUARANTINE_ARM:-${TAG:-arm}}
  b="$1.building.$J-$arm-$$"
  STAGE_BUILD_TMP=$b
  echo "### stage(mirror): BUILDING $m (key $key) -- this is the once-per-key cost"
  rm -rf "$b" 2>/dev/null
  mkdir -p "$b" || return 1
  local S=$(date +%s)
  rsync -a --info=stats2 "${STAGE_RSYNC_EXCLUDES[@]}" "$SRC_WS/" "$b/" || return 1
  echo "### mirror rsync wall=$(( $(date +%s) - S ))s"
  S=$(date +%s)
  # THE MIRROR IS A REAL COPY OF $SRC_WS -- NEVER A HARDLINK INTO IT.
  # This was `cp -al "$SRC_WS/third_party" "$b/third_party"` until 2026-09-03,
  # and that single line is what wrote the read-only canonical tree: every
  # workspace hardlinked out of the mirror was also hardlinked into
  # imprint-data, so setuptools' in-place egg_info rewrite landed there.
  # Hardlinks are only ever mirror -> workspace, one direction, one hop.
  # Cost of the correctness: +10.72 GB of real bytes ONCE PER MIRROR KEY.
  mkdir -p "$b/third_party" || return 1
  rsync -a --info=stats2 "$SRC_WS/third_party/" "$b/third_party/" || return 1
  echo "### mirror rsync third_party (REAL COPY, not cp -al) wall=$(( $(date +%s) - S ))s"
  # The key file is written BEFORE the manifest, and both are excluded from it,
  # so the manifest describes only the shared payload and never itself.
  { echo "key=$key"; echo "src=$SRC_WS"
    echo "pixi_toml_md5=$(md5sum "$SRC_WS/pixi.toml" | awk '{print $1}')"
    echo "git_head=$(git -C "$SRC_WS" rev-parse HEAD 2>/dev/null || echo nogit)"
    echo "built_by_job=$J"; echo "built_at=$(date -Is)"; } > "$b/.stage-mirror-key" || return 1
  stage_manifest "$b" > "$b/.stage-mirror-manifest.tsv" || return 1
  echo "entries=$(wc -l < "$b/.stage-mirror-manifest.tsv")" >> "$b/.stage-mirror-key"
  # concurrent builders: -T refuses to move INTO an existing directory, so
  # the loser discards its copy and adopts the winner's mirror.
  mv -T "$b" "$m" 2>/dev/null || { rm -rf "$b"; [ -f "$m/.stage-mirror-key" ] || return 1; }
  echo "### stage(mirror): built $m entries=$(grep '^entries=' "$m/.stage-mirror-key" | cut -d= -f2)"
}

stage_assert_mirror_disjoint () { # the mirror must share NO inode with $SRC_WS
  # The reader for the "real copy, not cp -al" writer above. If any mirror file
  # shares an inode with $SRC_WS then every workspace staged from the mirror is
  # hardlinked into the read-only canonical tree, and an in-place write inside a
  # job lands in imprint-data.
  #
  # PROOF-SMOKE-1-7. This read `$m/.stage-mirror-manifest.tsv` to know what to
  # sample, and sampled 50 of its rows. Both halves were defects. It could not
  # check a mirror somebody else built without that file -- proof_smoke.sh's
  # publish wrote only `.stage-mirror-key` -- and it returned 1 for "cannot
  # check", the same rc as "hardlinked", so the caller printed a hardlink
  # verdict for a check that never ran and quarantined a 10.72 GB mirror as
  # SRCLINKED (job 6014471, mirror published 03:55:59 by smoke job 6013332).
  # The check is now the ONE enumeration in tools/stage_mirror.sh, which walks
  # the tree instead of asking its builder's bookkeeping, keys on (device,
  # inode) rather than an inode number alone, and answers rc 2 for "cannot
  # check" so no caller can confuse the two again.
  stage_mirror_inode_check "$1" "$SRC_WS"
}

stage_mirror_hit () {            # cp -al the mirror into $WS, fanned out
  # A flat `cp -al $m $WS` is ONE serial walk and measured 230s (job 5658374
  # arm C); fanning out over the mirror's 80 top-level entries barely helped
  # (211s, arm D) because third_party is a single entry holding 25,178 files.
  # So the fan-out unit is a DEPTH-3 entry -- 722 of them, the largest holding
  # 2,926 files -- and the two directory levels above them are pre-created.
  # `cp -al` preserves a directory's mode and mtime; the levels we create by
  # hand would lose theirs, so they are restored from the mirror afterwards.
  local m=$1 S=$(date +%s)
  mkdir -p "$WS" || return 1
  ( cd "$m" && find . -mindepth 1 -maxdepth 2 -type d -printf '%P\n' ) \
    | grep -vF '.stage-mirror-' | sed "s|^|$WS/|" | tr '\n' '\0' \
    | xargs -0 -r -n 64 -P "$STAGE_PAR" mkdir -p || return 1
  # TWO finds, not one expression: -mindepth/-maxdepth are global OPTIONS in
  # GNU find, not tests, so `\( -mindepth 3 -o ! -type d \)` silently applies
  # -mindepth 3 to the whole walk and DROPS every depth-1 and depth-2 file --
  # which is exactly the bug job 5661215 caught (the staged tree came back
  # without AGENTS.md, .git/HEAD, test/*.py and every other shallow file).
  { ( cd "$m" && find . -mindepth 1 -maxdepth 2 ! -type d -printf '%P\n' )
    ( cd "$m" && find . -mindepth 3 -maxdepth 3            -printf '%P\n' ) } \
    | grep -vF '.stage-mirror-' | tr '\n' '\0' \
    | xargs -0 -r -I{} -P "$STAGE_PAR" cp -al "$m/{}" "$WS/{}" || return 1
  ( cd "$m" && find . -mindepth 1 -maxdepth 2 -type d -printf '%P\n' ) \
    | grep -vF '.stage-mirror-' \
    | while IFS= read -r d; do chmod --reference="$m/$d" "$WS/$d"; touch -r "$m/$d" "$WS/$d"; done
  chmod --reference="$m" "$WS"; touch -r "$m" "$WS"
  echo "### stage(mirror) cp -al wall=$(( $(date +%s) - S ))s"
}

stage_break_links () {           # give every file the lock writes IN PLACE its own inode
  # Source audit of retread v4.10.90 (worktree fix-p6a-strict-single-pass).
  # Every writer that produces a .whl goes temp-file -> rename
  # (`materialize_validated_wheel` in src/source_build.rs, `fetch_wheel` /
  # `atomic_owned_copy` in src/wheel.rs, `wheel::commit_atomic_write` for the
  # inject/relax paths), and the only removals are unlinks -- a rename or an
  # unlink replaces a directory entry and leaves a hardlinked twin's inode
  # alone, so the multi-GB wheel payloads are SAFE to share with the mirror.
  # Exactly four writers go through the inode and would corrupt the mirror:
  #     retread-progress-*.log      status::log, OpenOptions .append(true)
  #     retread-probe-trace-*.json  write_probe_trace, tokio::fs::write
  #     retread-audit*.json         build_bundle_audit site, tokio::fs::write
  #     *.retread-cache             write_relaxed_wheel_cache_stamp, fs::write
  # Those are what this breaks (363 files / 25 MB on the current source tree).
  # retread-*.lock.json is temp+rename and would be safe; it is broken anyway
  # because it is small and it is the file a reader is most likely to mistake
  # for shared state.
  local n=0 S=$(date +%s) f
  while IFS= read -r f; do
    cp -p "$f" "$f.stagetmp.$$" 2>/dev/null || continue
    mv -f "$f.stagetmp.$$" "$f" && n=$((n+1))
  done < <(find "$WS" -path "$WS/third_party" -prune -o -type f -links +1 \
             \( -name 'retread-progress-*.log' -o -name 'retread-probe-trace-*.json' \
                -o -name 'retread-audit*.json'  -o -name 'retread-*.lock.json' \
                -o -name '*.retread-cache' \) -print 2>/dev/null)
  # third_party is NO LONGER PRUNED. Its *.egg-info/*.txt files are rewritten in
  # place by setuptools during the lock, and pruning them here is what let the
  # write reach the mirror -- and, before the mirror became a real copy, the
  # canonical tree. This sweep runs on BOTH staging paths: the rsync fallback
  # `cp -al`s third_party straight out of $SRC_WS and needs it even more.
  local t=0
  while IFS= read -r f; do
    cp -p "$f" "$f.stagetmp.$$" 2>/dev/null || continue
    mv -f "$f.stagetmp.$$" "$f" && t=$((t+1))
  done < <(find "$WS/third_party" -type f -links +1 "${STAGE_TP_WRITABLE[@]}" -print 2>/dev/null)
  echo "### stage: broke $n in-place-written hardlink(s) + $t under third_party, wall=$(( $(date +%s) - S ))s"
  echo "### stage: files still sharing an inode with the mirror (expected -- all atomic-rename writers): $(find "$WS" -path "$WS/third_party" -prune -o -type f -links +1 -print 2>/dev/null | wc -l)"
  echo "### stage: third_party files still sharing an inode (expected -- large immutable payloads): $(find "$WS/third_party" -type f -links +1 -print 2>/dev/null | wc -l)"
}

src_tp_fingerprint () {          # a DIRECT reader on the read-only canonical tree
  # stage_verify_mirror watches the mirror; this watches $SRC_WS itself, so a
  # write that reaches imprint-data by any route at all -- including one that
  # never touches the mirror -- is caught by the job that did it.
  find "$SRC_WS/third_party" -type f "${STAGE_TP_WRITABLE[@]}" \
    -printf '%i %T@ %s %P\n' 2>/dev/null | LC_ALL=C sort
}

# DET-1-6-b: A QUARANTINE NAME THAT TWO ARMS OF ONE JOB CAN BOTH PRODUCE IS NOT
# A QUARANTINE. Every quarantine here used to be `mv "$x" "$x.<KIND>-$J"`, and a
# multi-arm job runs several arms under ONE job id: the second arm's `mv` finds
# a DIRECTORY already at that name and, being mv, moves the tree INSIDE it. The
# first quarantine then contains the second, the outer manifest describes
# neither, and the reader that goes looking for `<mirror>.DIRTY-<job>` finds a
# tree whose contents are two different failures nested. The live mirror root
# already carries four `.DIRTY-<jobid>` and one `.SRCLINKED-<jobid>`, so this is
# a shape that fires, not a hypothesis.
#
# The name now carries job id, ARM TAG and a timestamp, and -- because three
# fields still cannot make a collision impossible, only unlikely -- the target
# is CHECKED and a collision REFUSES rather than nests. A refusal leaves the
# tree where it is, which is recoverable; a nest is not.
stage_quarantine () {            # $1 = path to move aside, $2 = kind (DIRTY|SRCLINKED|stale)
  local src=$1 kind=$2 arm dst
  arm=${STAGE_QUARANTINE_ARM:-${TAG:-arm}}      # read at the point of use, never remembered
  dst="$src.$kind-${J:-nojob}-$arm-$(date +%s)"
  if [ -e "$dst" ]; then
    echo "### stage: QUARANTINE NAME COLLISION -- $dst already exists."
    echo "###   REFUSING to move: an mv of a directory onto an existing directory NESTS it,"
    echo "###   and a quarantine inside a quarantine describes neither failure. $src is"
    echo "###   left exactly where it is. Move it aside by hand and say which arm made it."
    return 3
  fi
  if mv "$src" "$dst" 2>/dev/null; then
    echo "### stage: quarantined -> $dst"
    return 0
  fi
  echo "### stage: QUARANTINE FAILED -- could not mv $src to $dst; the tree is untouched"
  return 3
}

stage_verify_mirror () {         # the READER for stage_build_mirror's writer
  local m=$1
  [ -f "$m/.stage-mirror-manifest.tsv" ] || { echo "### stage: no mirror manifest at $m -- cannot verify"; return 0; }
  local now=$A/${TAG}-$J.stage-mirror-now.tsv
  stage_manifest "$m" > "$now"
  if LC_ALL=C diff -q "$m/.stage-mirror-manifest.tsv" "$now" >/dev/null; then
    echo "### stage: mirror INTACT ($m)"
  else
    echo "### stage: FATAL-CLASS -- the mirror CHANGED under this job. A hardlinked"
    echo "###        input was written through. Quarantining the mirror; the next"
    echo "###        job rebuilds it. Diff head:"
    LC_ALL=C diff "$m/.stage-mirror-manifest.tsv" "$now" | head -20
    stage_quarantine "$m" DIRTY
    MIRROR_DIRTY=1
  fi
}

if [ ! -e "$WS/.cert-staged" ]; then
  if [ -d "$WS" ]; then
    mv "$WS" "$WS.trash.$$"
    ( chmod -R u+w "$WS.trash.$$" >/dev/null 2>&1; rm -rf "$WS.trash.$$" ) &
    echo "### moved pre-existing $WS aside"
  fi
  STAGE_USED=$STAGE_METHOD
  STAGE_MIRROR=
  if [ "$STAGE_METHOD" = mirror ]; then
    STAGE_KEY=$(stage_key)
    STAGE_MIRROR=$STAGE_MIRROR_ROOT/$STAGE_KEY
    echo "### stage: method=mirror key=$STAGE_KEY mirror=$STAGE_MIRROR"
    if [ -f "$STAGE_MIRROR/.stage-mirror-key" ] &&
       grep -qx "key=$STAGE_KEY" "$STAGE_MIRROR/.stage-mirror-key"; then
      echo "### stage: mirror key MATCHES -- warm path"
    else
      [ -e "$STAGE_MIRROR" ] && { echo "### stage: mirror key MISMATCH -- rebuilding"; stage_quarantine "$STAGE_MIRROR" stale || { echo "### stage: FATAL -- a stale mirror that cannot be moved aside would be REBUILT INTO, and the rebuild would inherit its files"; exit 13; }; }
      mkdir -p "$STAGE_MIRROR_ROOT"
      stage_build_mirror "$STAGE_MIRROR" "$STAGE_KEY" || {
        echo "### stage: mirror build FAILED -- falling back to the rsync path"
        rm -rf "${STAGE_BUILD_TMP:-}" 2>/dev/null; STAGE_USED=rsync; STAGE_MIRROR=; }
    fi
    if [ -n "$STAGE_MIRROR" ]; then
      # Before a single file is hardlinked out of it: is the mirror a real copy?
      # A mirror built by an older template is `cp -al`'d to $SRC_WS; adopting
      # it would re-open the hole, so it is quarantined and rebuilt here.
      stage_assert_mirror_disjoint "$STAGE_MIRROR" || {
        echo "### stage: mirror shares inodes with $SRC_WS -- quarantining and rebuilding"
        stage_quarantine "$STAGE_MIRROR" SRCLINKED || { echo "### stage: FATAL -- a source-linked mirror that cannot be moved aside would be rebuilt INTO, handing this job the very inodes the check refused"; exit 13; }
        stage_build_mirror "$STAGE_MIRROR" "$STAGE_KEY" \
          && stage_assert_mirror_disjoint "$STAGE_MIRROR" \
          || { echo "### stage: FATAL -- could not produce a mirror disjoint from $SRC_WS"; exit 13; }
      }
    fi
    if [ -n "$STAGE_MIRROR" ]; then
      stage_mirror_hit "$STAGE_MIRROR" || {
        echo "### stage: mirror hit FAILED -- falling back to the rsync path"
        rm -rf "$WS"; mkdir -p "$WS"; STAGE_USED=rsync; STAGE_MIRROR=; }
    fi
  fi
  if [ "$STAGE_USED" = rsync ]; then
    mkdir -p "$WS"
    stage_rsync_path
  fi
  echo "### stage: install the per-job WRITABLE bits (never shared with the mirror)"
  rm -rf "$WS/.pixi"; mkdir -p "$WS/.pixi"
  cp "$SRC_WS/.pixi/config.toml" "$WS/.pixi/config.toml"
  rm -f "$WS/pixi.toml"; cp "$CLEANED" "$WS/pixi.toml"
  rm -f "$WS"/pixi.lock "$WS"/pixi.lock.* 2>/dev/null
  # UNCONDITIONAL: the rsync fallback path hardlinks third_party out of $SRC_WS
  # too, so it needs the break sweep exactly as much as the mirror path does.
  stage_break_links
  echo "### stage: method actually used = $STAGE_USED"
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

# No `du` here either -- audit item C6, and it walks the staged tree over NFS.
echo "### staged files: $(find "$WS" | wc -l)  (third_party hardlinked from the mirror)"
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
# job 5547450: the failing hardlink's SOURCE was <uv cache>/builds-v0/.tmpXXXX,
# a per-build ephemeral build environment that one of six concurrent uv BUILDS
# reclaimed while another linked out of it. THIS phase is the one that builds,
# so `copy` stays here and is not up for the CERT's argument: the cert installs
# a frozen lock and links out of the content-addressed `archive-v0` instead
# (CERT_UV_LINK_MODE in phaseN_cert.sh, measured under fan-out 2026-09-03).
export UV_LINK_MODE=copy
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PIXI_BUILD_RETREAD_LOG=pixi_build_retread=debug,warn
unset RUST_LOG
export RUST_BACKTRACE=1

# PERSISTENT CACHES -- must come AFTER the job-scoped block above (it overrides
# the three cache dirs) and AFTER RETREAD_FAST_TMP_ROOT + SLURM_JOB_ID exist,
# because the verdict-cache symlink is placed at a path derived from both.
# shellcheck source=/dev/null
. "$FAST_ENV"
retread_fast_env "$WS" || { echo "FATAL: retread_fast_env refused"; exit 7; }

# --- C31-4: JOB-SCOPED sdist BUILD TREES, AND THE GUARD THAT READS THEM -------
# retread_fast_env just pointed UV_CACHE_DIR and PIXI_CACHE_DIR at the SHARED
# persistent root. For the byte-keyed buckets that is the whole 41x win and it
# stays. For `sdists-v9` and `builds-v0` it is a CORRECTNESS BUG: uv builds a
# source distribution IN PLACE inside `sdists-v9`, so a cmake project leaves a
# `CMakeCache.txt` there holding the ABSOLUTE compiler paths of whichever
# workspace built it first, and the next job inherits them. B-cert-4's
# `pm-newton-gpu` RED-install was exactly that -- a compiler path into
# `ws.MN1-5761731`, reaped weeks earlier -- while `pm-mujoco` built the SAME
# sdist green in the same job off the healthy python-3.10 sibling directory.
# LANE-C-WARM-LOG 31.10-31.12 and 33.
#
# `retread_scope_sdist_builds` gives this job its own empty build halves in BOTH
# uv caches (pixi 0.73.0 does not read UV_CACHE_DIR; its uv cache hangs off
# PIXI_CACHE_DIR, and THAT is the one that was poisoned) and leaves every
# byte-keyed bucket a symlink into the shared cache.
# --- `--cold-proof-arm`: THE ONE SHAPE THIS FUNCTION CANNOT SERVE ------------
# DET-1-6-2, measured on job 6014471 arm W1. `retread_scope_sdist_builds`
# shares the byte-keyed buckets by SYMLINKING every top-level entry of the
# shared uv cache EXCEPT `sdists-v9`/`builds-v0`, and then REFUSES when it
# symlinked none: "no byte-keyed bucket was symlinked -- the overlay would be a
# COLD cache, not an isolation of the build halves". That refusal is correct
# for a production relock and WRONG for a DECLARED-COLD PROOF ARM, which is
# given its OWN EMPTY `RETREAD_PERSIST_CACHE_ROOT` on purpose -- a shared cache
# would let arm 2 replay arm 1's prepared metadata and make the proof vacuous.
# For such an arm there is nothing to symlink BY CONSTRUCTION, and 6014471's
# census row read `symlinked=0 copied=0 build_dirs_local=4/4
# build_dirs_nonempty=0 shared_links_seen=0` before the refusal.
# PRE-SEEDING THE ROOT IS NOT THE FIX EITHER: the bucket a cold proof must not
# share is exactly `sdists-v9`, and `sdists-v9` is the one bucket this function
# never links -- it `rm -rf`s and re-creates it empty. So the caller declares
# the shape ON ARGV, not in the environment, which is where a proof's own state
# leaks between its arms.
# THE GUARD THAT READS THIS FLAG: tools/cold_proof_arm_guard.sh -- it runs the
# scoper against an empty shared cache with the flag and without it, and refuses
# unless the flagged run prints the SKIPPED row and the unflagged run refuses.
COLD_PROOF_ARM=0
for _cpa in "$@"; do [ "$_cpa" = --cold-proof-arm ] && COLD_PROOF_ARM=1; done
unset _cpa
if [ "$COLD_PROOF_ARM" = 1 ]; then
  echo "### stage: sdist scoping SKIPPED (declared cold proof arm)"
  echo "### stage: cold proof arm keeps UV_CACHE_DIR=$UV_CACHE_DIR PIXI_CACHE_DIR=$PIXI_CACHE_DIR unscoped -- the poison guard below still reads them"
else
  retread_scope_sdist_builds "$C" || { echo "FATAL: retread_scope_sdist_builds refused"; exit 7; }
fi

# ITS READER, and it runs BEFORE any install. A criterion with no live producer
# is a defect; this one names a file and a path and refuses on them.
SDIST_GUARD=$(dirname "$FAST_ENV")/sdist_build_poison_guard.sh
[ -f "$SDIST_GUARD" ] || { echo "FATAL: sdist_build_poison_guard.sh missing next to $FAST_ENV"; exit 7; }
bash "$SDIST_GUARD" || { echo "FATAL: sdist build poison guard refused -- see the rows above"; exit 7; }

# --- OPTIONAL: JOB-SCOPED WHEEL STORE, SEEDED FROM THE PERSISTENT ONE ---------
# retread_fast_env just exported RETREAD_WHEEL_STORE=<persist root>/wheels, the
# SHARED store. That is the right default and it is what buys index authority
# (see the wheel-store block in retread_fast_env.sh). But a lane whose lock is
# being killed by a stale `.<wheel>.whl.retread-fill-v1.lock` sidecar in that
# shared store needs the store's WARMTH without its POISON, and the way to get
# that is a job-scoped store SEEDED by a byte copy.
#
# To turn it on, a harness sets WHEEL_STORE_SEED to the store to seed FROM
# before it reaches this point, e.g.
#     WHEEL_STORE_SEED=$RETREAD_PERSIST_CACHE_ROOT/wheels
# and this block redirects RETREAD_WHEEL_STORE at a fresh job-scoped copy.
#
# THE SEED IS `retread_seed_wheel_store`, AND `cp -al` IS REFUSED FOR THIS.
# A `cp -al` of the wheel store is wrong twice over:
#   (a) a hardlink shares the inode, so making one bumps the ctime of files
#       other live lanes are reading and any write through the "isolated" copy
#       lands in the shared store (the p6k-b burn);
#   (b) `cp -al` copies the directory whole, fill locks included, so the store
#       that was supposed to be clean carries exactly the poison it was
#       isolated from.
# `retread_seed_wheel_store` does an `rsync -aW` byte copy with the fill locks
# and quarantine dirs excluded, prints a census line, and REFUSES (non-zero) if
# a fill lock, a quarantine dir, or a wheel with link count != 1 reached the
# destination. That link-count assert is the reader for this whole comment.
#
# NOTE: the `cp -al` calls elsewhere in this template stage $SRC_WS/third_party
# and the build mirror. Those are a different argument (read-only shared trees
# with no lock sidecars in them) and are NOT covered by this refusal.
if [ -n "${WHEEL_STORE_SEED:-}" ]; then
  WHEEL_STORE_JOBSCOPED=${WHEEL_STORE_JOBSCOPED:-$C/wheels-seeded}
  rm -rf "$WHEEL_STORE_JOBSCOPED"
  echo "### wheel store: seeding job-scoped store from $WHEEL_STORE_SEED (byte copy; cp -al is refused here)"
  retread_seed_wheel_store "$WHEEL_STORE_SEED" "$WHEEL_STORE_JOBSCOPED" \
    || { echo "FATAL: retread_seed_wheel_store refused"; exit 7; }
  export RETREAD_WHEEL_STORE=$WHEEL_STORE_JOBSCOPED
  echo "### wheel store: RETREAD_WHEEL_STORE=$RETREAD_WHEEL_STORE (job-scoped; seed wall is OUTSIDE the lock span)"
else
  echo "### wheel store: SHARED, RETREAD_WHEEL_STORE=$RETREAD_WHEEL_STORE (set WHEEL_STORE_SEED to isolate)"
fi

# --- WHICH WHEEL STORE THE LOCK ACTUALLY READ --------------------------------
# WHY THIS EXISTS. Until 2026-09-04 the block above was the only thing a lane
# log said about the wheel store, and after p6i re-enabled the SHARED export at
# 15:35 09-03 it was a DEAD LETTER: it announced a job-scoped store and then
# printed the shared path, so every proof since read the fill-lock-poisoned
# shared store while its log claimed otherwise. A claim about which store was
# read is worthless unless it is RESOLVED THE WAY THE BACKEND RESOLVES IT, at
# the moment the lock runs -- not read back off the variable the harness set.
#
# `wheel_store_in_use` is `courier::wheel_store_root_with` in shell:
# RETREAD_WHEEL_STORE -> XDG_CACHE_HOME -> HOME/.cache, then join retread/wheels.
# The scope word is derived by COMPARING that resolved path to the shared store,
# never from WHEEL_STORE_SEED -- so a seed that silently failed to take effect
# prints SHARED, which is the whole point.
# Reader: phase_template/wheel_store_census_guard.sh.
wheel_store_in_use () {
  if [ -n "${RETREAD_WHEEL_STORE:-}" ]; then printf '%s\n' "$RETREAD_WHEEL_STORE"
  elif [ -n "${XDG_CACHE_HOME:-}" ];    then printf '%s\n' "$XDG_CACHE_HOME/retread/wheels"
  else                                       printf '%s\n' "${HOME:-/nonexistent}/.cache/retread/wheels"
  fi
}
wheel_store_census () {          # $1 = when ("BEFORE LOCK" / "AFTER LOCK")
  local s shared scope
  s=$(wheel_store_in_use)
  shared=${RETREAD_PERSIST_CACHE_ROOT:-/oscar/data/stellex/glvov/agrescap/cache/retread}/wheels
  if [ "$s" = "$shared" ]; then scope=SHARED; else scope=JOB-SCOPED; fi
  printf '### WHEEL STORE IN USE (%s): scope=%s path=%s shared=%s entries=%s fill_locks=%s\n' \
    "$1" "$scope" "$s" "$shared" \
    "$(ls -1U "$s" 2>/dev/null | wc -l)" \
    "$(find "$s" -mindepth 2 -maxdepth 2 -name '.*.retread-fill-v1.lock' 2>/dev/null | wc -l)"
}

# backend stderr shim (pixi 0.73 swallows backend stderr behind its expect() panic)
BLOG=$A/${TAG}-$J.backend.log
: > "$BLOG"
SHIM=$A/${TAG}-$J.backend-shim.sh
cat > "$SHIM" <<SHIMEOF
#!/usr/bin/env bash
exec 2> >(tee -a "$BLOG" >&2)
exec "$BACKEND" "\$@"
SHIMEOF
chmod +x "$SHIM"
export PIXI_BUILD_BACKEND_OVERRIDE="pixi-build-retread=$SHIM"
echo "### backend shim: $SHIM -> $BACKEND ; stderr tee -> $BLOG"
echo "### pixi.real --version: $($PIXI --version)"
echo "### PIXI_BUILD_BACKEND_OVERRIDE=$PIXI_BUILD_BACKEND_OVERRIDE"

# --- THE INTERPRETER HASH SEED, EXPORTED INTO THE SHELL THAT LAUNCHES PIXI ----
# DET-1-5, and it is HERE rather than in the backend because the backend cannot
# reach it. DET-1-4-1 (job 6001140, node1802) locked this manifest three times
# on one node with ONE binary -- binsnaps/cand-3f2095a -- varying nothing but
# this variable in the launching shell:
#
#   arm 1  PYTHONHASHSEED=0   gym requires_dist md5 e569ebf5...  \  cmp rc=0
#   arm 2  PYTHONHASHSEED=0   gym requires_dist md5 e569ebf5...  /  RAW 0 SORTED 0
#   arm 3  unset              gym requires_dist md5 bd63668b...     RAW 28 SORTED 0
#
# A raw delta of 28 with a SORTED delta of 0 is a pure reordering: no package
# moved, no count changed (`moved_row_halves.sh` rc=0, moved=0, on all three
# pairs). That binary ALREADY calls `apply_reproducible_python_hash_seed` on
# all three of its own doors, and the block moved anyway -- because the process
# that builds `gym` 0.26.2's `requires_dist` is an in-process PEP 517 child of
# the pixi FRONTEND, spawned by pixi's embedded uv, which the backend never
# execs and therefore cannot pin. The only channel that reaches it is the
# environment pixi itself was launched with. That is this export.
#
# THE FUNCTION LIVES IN `tools/env_seed.sh`, NOT HERE (DET-1-6-1). It was
# written here by 58717bd, where the ARMS reach it and nothing else does --
# and det16-proof 6013332 paid for that: its preamble smokes each binary
# through `tools/proof_smoke.sh`, which launches its OWN `pixi lock` and had
# never heard of the export, so c0ccc0d's preflight refused at second zero and
# the job was REFUSED BEFORE ARM 1 (`### PREAMBLE FATAL: SMOKE FIX rc=1`) after
# paying 879 s to publish the stage mirror. The smoke and the wrapper must run
# pixi in the SAME environment or the cheap one refuses the expensive one's
# binary for a reason that is about itself. The library carries the DET-1-4-1
# measurement, the reason the value is asked of the binary rather than typed,
# and the reason the verb is detected STATICALLY by a marker instead of being
# probed by running it (an old binary answers an unknown verb by starting the
# JSON-RPC transport and blocking on stdin -- a hang, not a refusal).
# Readers: phase_template/env_seed_export_guard.sh, tools/proof_smoke_guard.sh
# arms V1-V4.
#
# The path is resolved the way `store_reap_census.sh` below is: beside this
# script first (a derived wrapper in a worktree gets that worktree's library),
# then the task copy. It REFUSES rather than continuing seedless -- a missing
# library must not degrade into the exact defect this block exists to close.
ENV_SEED_LIB=$(dirname "$0")/../tools/env_seed.sh
[ -f "$ENV_SEED_LIB" ] || ENV_SEED_LIB=$T/tools/env_seed.sh
if [ ! -f "$ENV_SEED_LIB" ]; then
  echo "### FATAL ENV SEED: no env_seed.sh beside this wrapper ($(dirname "$0")/../tools/)"
  echo "###        nor at $T/tools/env_seed.sh. Locking without the pinned seed is the"
  echo "###        DET-1-4-1 defect, so this wrapper refuses instead of continuing."
  echo "###        ACTUATOR: sync tools/env_seed.sh, or run from a worktree that has it."
  exit 15
fi
# shellcheck source=/dev/null
. "$ENV_SEED_LIB"
env_seed_export "$BACKEND" || exit 15

# THE GUARD ARM, and it is the reader for the export above. The backend's
# preflight refuses when the ambient seed is absent or not the constant, so
# `retread preflight` here is the same check the lock itself will make -- run
# at second zero, on this job's clock, instead of surfacing as a refused
# `initialize` forty minutes into a staged lock. A wrapper that exported the
# variable and never checked that anything reads it is exactly the
# stamped-but-unread directive law 2 forbids.
if ! "$BACKEND" preflight; then
  echo "### FATAL ENV SEED: the backend refused its own preflight AFTER this wrapper"
  echo "###        exported PYTHONHASHSEED=$PYTHONHASHSEED. The export and the check"
  echo "###        disagree; that is a defect in one of them, not a condition to"
  echo "###        lock through."
  exit 15
fi
echo "### backend preflight ok (uv version + ambient PYTHONHASHSEED)"

env | grep -E '^(HOME|PIXI_|PYTHONHASHSEED|RATTLER_|UV_|XDG_|TMPDIR|RETREAD_|CONDA_OVERRIDE)' | sort
# --- persistent-cache census (NEVER `du` this tree) ----------------------------
# `du -sh <persist cache root>/*` walks the uv cache and the 69 GB stage mirror
# over NFS. Job 5678087 sat in it for 26+ minutes in D-state BEFORE its lock
# started -- half an hour of a 3 h wall spent on a diagnostic. Audit item C6
# already ruled "no `du` over the store in harnesses"; the template still had
# two calls. What the lane logs actually quote off these lines is the ENTRY
# COUNT ("store entries before 0 / after 225"), never the byte total, so count
# the top-level entries and never descend.
cache_census() {
  local root=${RETREAD_PERSIST_CACHE_ROOT:-/oscar/data/stellex/glvov/agrescap/cache/retread}
  local d n
  for d in "$root"/*; do
    [ -e "$d" ] || continue
    n=$(ls -1U "$d" 2>/dev/null | wc -l)
    printf '  %8s top-level entries  %s\n' "$n" "$d"
  done
}

echo "### persistent cache census BEFORE the lock (entry counts; du is banned here):"
cache_census

# --- STORE-REAP-2: the persistent-store census, the PRODUCTION READER of
# `retread store-reap` -------------------------------------------------------
# STORE-REAP-1-1 (law 2): the three persistent-store reapers have exactly one
# caller, the backend's `initialize`, and the root IT resolves is job-scoped by
# the ENV BLOCK above -- so the reapers can never reach the roots that actually
# hold the over-age entries, and their capability had no reader that runs. This
# is that reader. It is DRY RUN, it renames nothing, and the census script has
# no way to be told otherwise (tools/store_reap_census_guard.sh asserts it).
# Its rc is deliberately not checked: a census must never fail a relock.
STORE_REAP_CENSUS=$(dirname "$0")/../tools/store_reap_census.sh
[ -f "$STORE_REAP_CENSUS" ] || STORE_REAP_CENSUS=$T/tools/store_reap_census.sh
if [ -f "$STORE_REAP_CENSUS" ]; then
  BACKEND=$BACKEND bash "$STORE_REAP_CENSUS" "BEFORE LOCK"
else
  echo "### STORE-REAP CENSUS (BEFORE LOCK): SKIPPED -- no store_reap_census.sh beside this template nor at \$T/tools"
fi


# --- READERS-2-1: A LANDING CRITERION MUST HAVE A FROZEN PRODUCER ------------
# THE DEFECT. B29's relock judged its git-snapshot landing criterion against the
# SHARED persistent store root read LIVE at judgement time
# ($RETREAD_PERSIST_CACHE_ROOT/git-snapshots/canonical-git-sources, exported by
# retread_fast_env's C18-1 default-on flip). Two runs of the same relock over
# the same landing reported census_rows 5 and then 3 -- not because the landing
# changed, but because other jobs write that root while a lock is running. A
# criterion whose input can move under it is not a criterion; it is a coin, and
# it had been flipping in silence.
#
# THE FIX, AND WHY IT IS A FILE AND NOT A VARIABLE. Freeze the listing to a file
# in the JOB ROOT before the lock, and judge against the FILE. A shell variable
# would freeze the number too, but it dies with the process, cannot be audited
# after the fact, and cannot be handed to the phase-2 cert or to an analyzer
# that runs hours later in a different job -- which is exactly where these
# criteria are read. The frozen file is the evidence packet's copy of the input
# the verdict was computed from.
#
# BOTH NUMBERS, ONE RELEASE. store_census_release prints frozen AND live on one
# line. Hiding the live number would trade one blindness for another: a store
# that grew by six generations under the lock is a fact the packet should carry,
# it is just not the fact the verdict is allowed to depend on.
#
# A MISSING SNAPSHOT IS A REFUSAL, NOT A FALLBACK. store_census_frozen_rows
# returns non-zero and prints FATAL when it is asked for a label nobody
# snapshotted, so a criterion can never quietly degrade back to reading the
# live store -- which is the whole defect, reintroduced by omission.
# Reader: phase_template/store_census_snapshot_guard.sh.
store_census_file () { printf '%s\n' "$A/${TAG}-$J.census-$1.tsv"; }

store_census_live_rows () {      # $1 = subtree root; generation DIRECTORIES only
  # Directories only, on purpose: the reap try-lock beside the generations is a
  # FILE and the reaper's walk skips it, so counting it would make the criterion
  # disagree with the walk it is judging.
  find "$1" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l
}

store_census_snapshot () {       # $1 = label, $2 = subtree root. BEFORE THE LOCK.
  local label=$1 root=$2 out n
  out=$(store_census_file "$label")
  find "$root" -maxdepth 1 -mindepth 1 -printf '%y\t%f\n' 2>/dev/null | LC_ALL=C sort > "$out"
  n=$(grep -c '^d' "$out")
  printf '%s\n' "$n" > "$out.rows"
  echo "### CENSUS SNAPSHOT rows=$n file=$out"
  echo "### CENSUS SNAPSHOT label=$label root=$root taken=$(date -Is)"
}

store_census_frozen_rows () {    # $1 = label -> echoes the FROZEN count, or REFUSES
  local label=$1 out
  out=$(store_census_file "$label")
  if [ ! -f "$out.rows" ]; then
    echo "### FATAL CENSUS: no snapshot was taken for label='$label', so there is" >&2
    echo "###        nothing frozen to judge against. Refusing rather than falling" >&2
    echo "###        back to a live read of a store other jobs are writing --" >&2
    echo "###        that fallback IS the READERS-2-1 defect." >&2
    echo "###        ACTUATOR: call store_census_snapshot '$label' <root> before the lock." >&2
    return 1
  fi
  cat "$out.rows"
}

store_census_release () {        # $1 = label, $2 = subtree root. BOTH numbers, one line.
  local label=$1 root=$2 frozen live
  frozen=$(store_census_frozen_rows "$label") || return 1
  live=$(store_census_live_rows "$root")
  echo "### CENSUS RELEASE label=$label frozen=$frozen live=$live delta=$((live - frozen)) file=$(store_census_file "$label")"
}

store_census_judge_present () {  # $1 = label. THE CRITERION, judged from the FILE.
  # "the store the reapers were asked to walk was PRESENT when this lock
  # started". Judged frozen, because a store that appears or empties AFTER the
  # lock must not be able to flip a verdict about the lock.
  local label=$1 frozen
  frozen=$(store_census_frozen_rows "$label") || return 1
  if [ "$frozen" -eq 0 ]; then
    echo "### CENSUS CRITERION label=$label REFUSED: the frozen listing counted ZERO"
    echo "###        generation directories, so every reaper row this run prints is"
    echo "###        over a store that was not there -- 'nothing over-age' and"
    echo "###        'nowhere to look' are not the same reading."
    echo "###        ACTUATOR: confirm RETREAD_GIT_SNAPSHOT_STORE points at the"
    echo "###        persistent root (retread_fast_env exports it) before reading"
    echo "###        any reap row from this job."
    return 1
  fi
  echo "### CENSUS CRITERION label=$label ok: frozen generation_dirs=$frozen (judged from the snapshot file, never from a live re-read)"
}

# THE PRODUCTION CALL SITE (law 2: a capability with no caller is the same
# defect as a criterion with no producer). The git-snapshot store is the SHARED
# persistent one and it is the root every merge lane's census criterion is about.
GIT_SNAPSHOT_GENS=${RETREAD_GIT_SNAPSHOT_STORE:-${RETREAD_PERSIST_CACHE_ROOT:-/oscar/data/stellex/glvov/agrescap/cache/retread}/git-snapshots}/canonical-git-sources
store_census_snapshot git-snapshots "$GIT_SNAPSHOT_GENS"
store_census_judge_present git-snapshots || echo "### CENSUS CRITERION git-snapshots refused (reported; a census must never fail a relock, and the refusal above names the actuator)"
########## 3. LOCK ($EXPECT_ENVS envs, no pre-existing pixi.lock) ##########
cd "$WS" || exit 5
LLOG=$A/${TAG}-$J.lock.log
LTIME=$A/${TAG}-$J.lock.time.txt

# stage_verify_mirror runs TWICE. HERE, as a guard that the mirror this job was
# staged from is itself clean before a single byte is locked -- a mirror already
# dirty pre-lock means an EARLIER job wrote through it and the inputs this job
# staged are not the ones the mirror was built from, which is not something to
# discover three hours later. And again after the lock, as the reader for this
# job's own writes.
if [ -n "${STAGE_MIRROR:-}" ] && [ -d "${STAGE_MIRROR:-/nonexistent}" ]; then
  echo "### stage: PRE-LOCK mirror verify"
  stage_verify_mirror "$STAGE_MIRROR"
  [ "$MIRROR_DIRTY" = 0 ] || {
    echo "### FATAL: the stage mirror was already dirty BEFORE this lock."
    echo "###        Refusing to lock on inputs that are not the ones it was built from."
    exit 14; }
fi
# The direct reader on the canonical tree, taken across the lock.
SRC_FP_BEFORE=$A/${TAG}-$J.src-third_party.before.txt
SRC_FP_AFTER=$A/${TAG}-$J.src-third_party.after.txt
src_tp_fingerprint > "$SRC_FP_BEFORE"
echo "### source-tree write guard: fingerprinted $(wc -l < "$SRC_FP_BEFORE") in-place-writable files under $SRC_WS/third_party"

wheel_store_census 'BEFORE LOCK'
echo "### lock start $(date -Is)"
S=$(date +%s)
/usr/bin/time -v -o "$LTIME" "$PIXI" lock -v > "$LLOG" 2>&1
LRC=$?
LW=$(( $(date +%s) - S ))
echo "### lock rc=$LRC wall=${LW}s end $(date -Is)"
echo "$LRC" > "$A/${TAG}-$J.rc"; echo "$LW" > "$A/${TAG}-$J.wall"
wheel_store_census 'AFTER LOCK'
store_census_release git-snapshots "$GIT_SNAPSHOT_GENS" || true

# The READER for the stage mirror's writer. A relock that wrote through a
# hardlink into the shared mirror has poisoned it for every later batch, and the
# only way that can be known is to re-walk it. Empty $STAGE_MIRROR (rsync path,
# or a mirror that was never used) makes this a no-op.
[ -n "${STAGE_MIRROR:-}" ] && [ -d "${STAGE_MIRROR:-/nonexistent}" ] && stage_verify_mirror "$STAGE_MIRROR"
src_tp_fingerprint > "$SRC_FP_AFTER"
if LC_ALL=C diff -q "$SRC_FP_BEFORE" "$SRC_FP_AFTER" >/dev/null; then
  echo "### source-tree write guard: $SRC_WS/third_party UNCHANGED across the lock"
else
  echo "### source-tree write guard: FATAL -- THIS JOB WROTE $SRC_WS. Diff:"
  LC_ALL=C diff "$SRC_FP_BEFORE" "$SRC_FP_AFTER" | head -40
  SRC_WRITTEN=1
fi
echo "### /usr/bin/time -v (lock) -> $LTIME"
grep -E 'Elapsed \(wall|Maximum resident set size|User time|System time|Percent of CPU' "$LTIME" | sed 's/^/  /'
LRSS=$(awk -F': ' '/Maximum resident set size/{print $2}' "$LTIME")
echo "### lock peak RSS: ${LRSS:-UNKNOWN} kbytes  (72G request = $( [ -n "$LRSS" ] && awk -v r="$LRSS" 'BEGIN{printf "%.0fx", 72*1024*1024/r}' || echo '?') headroom)"
echo "### persistent cache census AFTER the lock (entry counts; du is banned here):"
cache_census

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
  grep -aoE '^\s+- pypi: \S+'  "$LK" | awk '{print $3}' | LC_ALL=C sort -u > "$A/${TAG}-$J.pypi-urls.txt"
  grep -aoE '^\s+- conda: \S+' "$LK" | awk '{print $3}' | LC_ALL=C sort -u > "$A/${TAG}-$J.conda-urls.txt"
  grep -aoE '^- pypi: \S+'  "$LK" | sed 's|.*/||; s|-[0-9].*||'      | grep -v '^$' | LC_ALL=C sort -u > "$A/${TAG}-$J.pypi-names.txt"
  grep -aoE '^- conda: \S+' "$LK" | sed 's|.*/||; s|-[^-]*-[^-]*$||' | grep -v '^$' | LC_ALL=C sort -u > "$A/${TAG}-$J.conda-names.txt"
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
LC_ALL=C diff "$PROBES_CANON" "$PROBES_ARM" | sed 's/^/  /'

########## 6. HANDOFF TO THE CERT PHASE ##########
# WHO OWNS THE JOB-SCOPED ROOTS, recorded here so the cert phase never guesses.
# Two cleanup jobs once raced on the same two roots and left both on disk; the
# measurement is in the EVIDENCE header above. The rule is one owner per root,
# and this is where the owner is written down.
#
# Two sources, in order, both facts rather than environment guesses:
#   1. CLEANUP_AT_DISPATCH in THIS relock job's environment -- set it to the
#      cleanup's JOB ID. The legacy value `1` is honoured but records only
#      `unrecorded-id`, which makes for a worse cert log line.
#   2. $A/cleanup_at_dispatch.jobid, the dispatch note. The launcher submits the
#      gated cleanup only AFTER both phases are queued, so the id cannot be in
#      this job's environment; it writes the id into that file instead, any time
#      before this job reaches the handoff.
# Nothing recorded => CLEANUP_JOB= is written empty, and the cert phase submits
# and owns its own cleanup exactly as it did before this rule existed.
CLEANUP_JOB_RECORD=""
case "${CLEANUP_AT_DISPATCH:-0}" in
  ''|0|none|NONE) ;;
  1) CLEANUP_JOB_RECORD=unrecorded-id ;;
  *) CLEANUP_JOB_RECORD=$CLEANUP_AT_DISPATCH ;;
esac
if [ -z "$CLEANUP_JOB_RECORD" ] && [ -r "$A/cleanup_at_dispatch.jobid" ]; then
  CLEANUP_JOB_RECORD=$(tr -dc '0-9_' < "$A/cleanup_at_dispatch.jobid" | head -c 32)
fi
if [ -n "$CLEANUP_JOB_RECORD" ]; then
  echo "### CLEANUP OWNER AT DISPATCH: job $CLEANUP_JOB_RECORD -- recorded in the handoff stamp; the cert phase will DEFER to it"
else
  echo "### CLEANUP OWNER AT DISPATCH: NONE RECORDED -- the cert phase will submit and own exactly one cleanup"
fi
if [ "$LRC" = 0 ] && [ -f "$A/pixi.lock.cert" ] && [ "$MIRROR_DIRTY" = 0 ] && [ "$SRC_WRITTEN" = 0 ]; then
  {
    echo "# written by phaseN_relock.sh (${TAG}) job $J $(date -Is)"
    echo "P1_JOB=$J"
    echo "WS=$WS"
    echo "P1_CACHE_ROOT=$C"
    echo "LOCK=$A/pixi.lock.cert"
    echo "EXPECT_LOCK_MD5=$(md5sum < "$A/pixi.lock.cert" | awk '{print $1}')"
    echo "CLEANUP_JOB=$CLEANUP_JOB_RECORD"
  } > "$A/relock_env.sh"
  echo "### cert-phase handoff written:"; cat "$A/relock_env.sh"
else
  echo "### NO cert-phase handoff written (lock rc=$LRC mirror_dirty=$MIRROR_DIRTY src_written=$SRC_WRITTEN) -- the afterok dependency will not release"
fi

########## 7. NO SELF-CLEANUP HERE -- ON PURPOSE ##########
# See the EVIDENCE header: an `rm -rf` of a job-scoped root on the afterok path
# cost job 5596128 5152s of held QOS. Cleanup is a separate 1-CPU job hung on
# `--dependency=afterany:<this job>:<cert job>` -- BOTH phases, `afterany` on
# both. Submitted at DISPATCH by the launcher, whose job id section 6 records
# as CLEANUP_JOB= in the handoff stamp, or -- when nothing is recorded -- by
# the cert phase with the relock job included in the dependency. EXACTLY ONE
# of the two, never both: two owners racing on one root leaves it on disk.
#
# NEVER behind the cert alone. When this relock fails its own lock the block
# above writes no handoff, Slurm cancels the afterok cert, and a cleanup that
# depends only on the cert never releases -- so BOTH roots below leak forever.
# That is exactly what stranded `certC18A-5759225` / `ws.C18A-5759225` and the
# C18B pair, whose run log reads "the afterok dependency will not release" and
# "self-cleanup NOT run here by design" on the same page.
echo "### self-cleanup NOT run here by design -- roots left to the afterany:<relock>:<cert> cleanup job: $C $WS"
echo "### if no such cleanup job exists for THIS relock, that is the C18 defect: submit one now over those roots"
echo "### inode quota AFTER:"; "$CQ" 2>/dev/null | grep -E 'data\+stellex' | head -2
echo "### ${TAG} RELOCK DONE lock_rc=$LRC wall=${LW}s peak_rss_kb=${LRSS:-unknown} mirror_dirty=$MIRROR_DIRTY src_written=$SRC_WRITTEN $(date -Is)"
# A job that mutated a shared input can never report rc=0. Until 2026-09-03 the
# post-lock FATAL-CLASS was a printed line and nothing else: job 5673296
# quarantined the mirror AND wrote imprint-data, and still exited 0.
if [ "$MIRROR_DIRTY" != 0 ] || [ "$SRC_WRITTEN" != 0 ]; then
  echo "### FATAL: this job wrote through a hardlink into a SHARED INPUT"
  echo "###        (mirror_dirty=$MIRROR_DIRTY src_written=$SRC_WRITTEN). Exiting 12"
  echo "###        REGARDLESS of lock rc=$LRC -- the lock's own result is not the question."
  exit 12
fi
exit "$LRC"
