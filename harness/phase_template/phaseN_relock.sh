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
# campaign before. Deliberate evidence citations belong in the EVIDENCE region.
# The match runs INSIDE awk, on the LINE, never on "FILENAME:LNO: line". Piping
# the annotated text to grep made the check match its own FILENAME: a harness in
# a directory named after a previous batch failed against itself, on every line,
# with the tokens nowhere in its body. A scan must not be able to match itself.
LEFT=$(awk '
  /^### EVIDENCE BEGIN/       {e=1} /^### EVIDENCE END/       {e=0; next} e {next}
  /^### SUBSTITUTE: BEGIN/    {s=1} /^### SUBSTITUTE: END/    {s=0; next} s {next}
  /^### LEFTOVER-CHECK BEGIN/ {l=1} /^### LEFTOVER-CHECK END/ {l=0; next} l {next}
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
STAGE_METHOD=mirror
STAGE_MIRROR_ROOT=/oscar/data/stellex/glvov/agrescap/cache/retread/stage-mirror
STAGE_PAR=16          # cp -al here is NFS-RPC-latency bound, not CPU bound
STAGE_RSYNC_EXCLUDES=( --exclude '/.pixi/' --exclude '/third_party/'
  --exclude '/assets/' --exclude '/groot-sonic-data/' --exclude '/logs/'
  --exclude '/results/' --exclude '/scratchpad/' --exclude '/scratch_rescue/'
  --exclude '/.pytest_cache/' --exclude '/pixi.lock' --exclude '/pixi.lock.*' )

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
  find "$1" -mindepth 1 -xdev -printf '%y\t%s\t%T@\t%P\n' | grep -vF '.stage-mirror-' | sort
}

stage_build_mirror () {          # ONE-TIME per key. Returns non-zero on failure.
  local m=$1 key=$2 b="$1.building.$J"
  echo "### stage(mirror): BUILDING $m (key $key) -- this is the once-per-key cost"
  rm -rf "$b" 2>/dev/null
  mkdir -p "$b" || return 1
  local S=$(date +%s)
  rsync -a --info=stats2 "${STAGE_RSYNC_EXCLUDES[@]}" "$SRC_WS/" "$b/" || return 1
  echo "### mirror rsync wall=$(( $(date +%s) - S ))s"
  S=$(date +%s)
  cp -al "$SRC_WS/third_party" "$b/third_party" || return 1
  echo "### mirror cp -al third_party wall=$(( $(date +%s) - S ))s"
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
  echo "### stage: broke $n in-place-written hardlink(s), wall=$(( $(date +%s) - S ))s"
  echo "### stage: files still sharing an inode with the mirror (expected -- all atomic-rename writers): $(find "$WS" -path "$WS/third_party" -prune -o -type f -links +1 -print 2>/dev/null | wc -l)"
}

stage_verify_mirror () {         # the READER for stage_build_mirror's writer
  local m=$1
  [ -f "$m/.stage-mirror-manifest.tsv" ] || { echo "### stage: no mirror manifest at $m -- cannot verify"; return 0; }
  local now=$A/${TAG}-$J.stage-mirror-now.tsv
  stage_manifest "$m" > "$now"
  if diff -q "$m/.stage-mirror-manifest.tsv" "$now" >/dev/null; then
    echo "### stage: mirror INTACT ($m)"
  else
    echo "### stage: FATAL-CLASS -- the mirror CHANGED under this job. A hardlinked"
    echo "###        input was written through. Quarantining the mirror; the next"
    echo "###        job rebuilds it. Diff head:"
    diff "$m/.stage-mirror-manifest.tsv" "$now" | head -20
    mv "$m" "$m.DIRTY-$J" 2>/dev/null && echo "### stage: quarantined -> $m.DIRTY-$J"
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
      [ -e "$STAGE_MIRROR" ] && { echo "### stage: mirror key MISMATCH -- rebuilding"; mv "$STAGE_MIRROR" "$STAGE_MIRROR.stale-$J"; }
      mkdir -p "$STAGE_MIRROR_ROOT"
      stage_build_mirror "$STAGE_MIRROR" "$STAGE_KEY" || {
        echo "### stage: mirror build FAILED -- falling back to the rsync path"
        rm -rf "$STAGE_MIRROR.building.$J" 2>/dev/null; STAGE_USED=rsync; STAGE_MIRROR=; }
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
  [ -n "$STAGE_MIRROR" ] && stage_break_links
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
export PIXI_BUILD_RETREAD_LOG=pixi_build_retread=debug,warn
unset RUST_LOG
export RUST_BACKTRACE=1

# PERSISTENT CACHES -- must come AFTER the job-scoped block above (it overrides
# the three cache dirs) and AFTER RETREAD_FAST_TMP_ROOT + SLURM_JOB_ID exist,
# because the verdict-cache symlink is placed at a path derived from both.
# shellcheck source=/dev/null
. "$FAST_ENV"
retread_fast_env "$WS" || { echo "FATAL: retread_fast_env refused"; exit 7; }

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
env | grep -E '^(HOME|PIXI_|RATTLER_|UV_|XDG_|TMPDIR|RETREAD_|CONDA_OVERRIDE)' | sort
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

########## 3. LOCK ($EXPECT_ENVS envs, no pre-existing pixi.lock) ##########
cd "$WS" || exit 5
LLOG=$A/${TAG}-$J.lock.log
LTIME=$A/${TAG}-$J.lock.time.txt
echo "### lock start $(date -Is)"
S=$(date +%s)
/usr/bin/time -v -o "$LTIME" "$PIXI" lock -v > "$LLOG" 2>&1
LRC=$?
LW=$(( $(date +%s) - S ))
echo "### lock rc=$LRC wall=${LW}s end $(date -Is)"
echo "$LRC" > "$A/${TAG}-$J.rc"; echo "$LW" > "$A/${TAG}-$J.wall"

# The READER for the stage mirror's writer. A relock that wrote through a
# hardlink into the shared mirror has poisoned it for every later batch, and the
# only way that can be known is to re-walk it. Empty $STAGE_MIRROR (rsync path,
# or a mirror that was never used) makes this a no-op.
[ -n "${STAGE_MIRROR:-}" ] && stage_verify_mirror "$STAGE_MIRROR"
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
