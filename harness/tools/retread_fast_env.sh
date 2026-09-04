#!/usr/bin/env bash
# retread_fast_env.sh -- persistent shared caches for retread relocks.
#
# MEASURED, job 5598763 (node1820, 16 cpu, 72G, harness p5t-ab/p5t_abcb.sh,
# post-merge binsnap sha a5ed78a1… = integration/4.12 a01c49f, one manifest:
# the canonical imprint-data pixi.toml md5 9711eb99…, 27 envs, three arms in
# one job so they are within-job comparable):
#
#   arm A  defaults, cold, probes serial            rc=0  wall 2865s
#   arm B  parallel probes + these caches, COLD     rc=0  wall 2633s
#   arm C  arm B again, caches WARM                 rc=0  wall   69s
#
#   backend span window  1753.2s -> 1587.1s ->  62.4s
#   route probes union    805.0s ->  828.2s ->  32.7s   (probes 315/315/5)
#   frontend zero-backend 1111.8s -> 1045.9s ->   6.6s
#
# Lock RESOLUTION is identical across all three arms: pypi names 174, conda names
# 1707, pypi urls 213, conda urls 2584, and env_version_delta.py reports 0 moved
# version rows in all 27 envs for both A-vs-B and B-vs-C. The BYTES are not
# identical -- A-vs-B reorders gym's requires_dist extras (30 diff lines,
# set-identical, sorted diff = 0), and B-vs-C moves the protomotions-deps-pack
# build hash and adds one `run_exports: {}` line. See HAZARDS.
#
# SO: the whole win is CACHE WARMTH, and it is 41x. The parallel-probe flag is
# NOT part of it and is NOT set here -- see below.
#
# HOW TO USE: after a harness has exported its job-scoped roots, source this and
# call the function with the workspace the arm will lock in:
#
#     . /oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/tools/retread_fast_env.sh
#     retread_fast_env "$WS"
#
# It must be called AFTER RETREAD_FAST_TMP_ROOT and SLURM_JOB_ID are set: the
# verdict-cache symlink is placed at a path derived from both.
#
# WHAT IT CHANGES, AND WHAT EACH ONE BOUGHT
#   UV_CACHE_DIR         the frontend's PyPI resolution cache, and the big one.
#                        Zero-backend frontend time 1111.8s -> 6.6s once warm.
#                        The cache reaches 15 GB after one relock.
#   PIXI_CACHE_DIR       pixi's own cache (21 GB after one relock).
#   RATTLER_CACHE_DIR    conda repodata (856 MB).
#   the verdict SYMLINK  route-probe verdicts. These do NOT follow
#                        RETREAD_CACHE_DIR: fast-tmp redirects the retread cache
#                        into a JOB-scoped namespace
#                          $RETREAD_FAST_TMP_ROOT/retread-$USER/
#                          <sha256(realpath ws)[:12]>/job-$SLURM_JOB_ID/caches/retread/
#                        (fasttmp::namespace(); verified against the
#                        "route probe cache: opened" rows of jobs 5548874 and
#                        5598763). Pre-placing a symlink there is the only way
#                        to make verdicts outlive a job without turning fast-tmp
#                        off. 13 verdict files, 551 KB, 315 probes -> 5.
#
# WHAT IT DOES NOT CHANGE
#   RETREAD_BUILD_ROOT / RETREAD_ARTIFACT_ROOT / RETREAD_META_ROOT /
#   RETREAD_SCRATCH_ROOT / HOME / TMPDIR stay job-scoped. Only rebuildable
#   download+solve caches are shared; build state is not.
#
# WHY RETREAD_PARALLEL_PROBES IS NOT SET HERE
#   It was measured and it does nothing. Arm B ran with the flag on (the backend
#   confirmed it: "experimental parallel probe solves enabled" x17) against arm
#   A's "parallel probe solves disabled" x17, both cold, same node, same job,
#   same 315 probes -- and the route-probe span union went 805.0s -> 828.2s,
#   i.e. 2.9% WORSE, inside the noise. It is not a disqualified flag (lock
#   content was identical), it is simply not a win, and it carries the v4.10.46
#   silent-process-exit history documented on
#   `thread_budget::parallel_probes_enabled`. Turn it on only for a deliberate
#   experiment:  RETREAD_FAST_ENV_PARALLEL_PROBES=1 retread_fast_env "$WS"
#
# HAZARDS
#   * Concurrent jobs share these directories. This function sets
#     UV_LINK_MODE=copy, which is the right DEFAULT but not a law -- read this
#     before overriding it. Job 5547450's race was a uv BUILD: six concurrent
#     builds, and the link that failed had its SOURCE in
#     `<uv cache>/builds-v0/.tmpXXXX`, a per-build ephemeral build environment a
#     sibling reclaimed mid-link. uv documents UV_LINK_MODE as the method used
#     "when installing packages FROM THE GLOBAL CACHE"; an INSTALL of a frozen
#     lock links out of `archive-v0`, which is content-addressed, which nothing
#     in this campaign prunes (`uv cache clean|prune`: zero call sites), and
#     which the cert never leaves -- the persistent cache's builds-v0 and
#     sdists-v9 are EMPTY after every relock and cert we have run. So: keep
#     `copy` for any phase that BUILDS (the relock), and decide the CERT's mode
#     with the harness knob CERT_UV_LINK_MODE, which is exported after this
#     function precisely because this line used to be the last word. Measured
#     under fan-out, job 5685816 vs 5658928: hardlink is 1.45x on the env loop
#     and 2.35x on bytes with identical cert_verdict.sh rows and zero race
#     lines. Default unchanged; see tools/phase_template/README.md.
#   * INODES. One cold relock leaves ~36 GB here on a filesystem whose soft
#     quota is ~95M/100M inodes. The tree is rebuildable, so deleting it is
#     always safe and only ever slow:  rm -rf /oscar/data/stellex/glvov/agrescap/cache/retread
#   * A WARM run reuses the workspace's own .pixi state too, not just these
#     caches. Arm C's 69s is "same workspace, manifest unchanged, pixi.lock
#     deleted" -- the realistic re-lock, not a from-scratch build.
#   * Arm C's lock differed from arm B's in exactly one place: the
#     protomotions-deps-pack conda_source build hash (5e31933f -> 784d2b1c) plus
#     one added `run_exports: {}` line. Name sets and every per-env version are
#     identical; this is a rebuilt local path-source pack, not a resolution
#     change. Watch it if a cert ever reads that hash.

RETREAD_PERSIST_CACHE_ROOT=${RETREAD_PERSIST_CACHE_ROOT:-/oscar/data/stellex/glvov/agrescap/cache/retread}

retread_fast_env () {
  local ws=${1:?retread_fast_env: pass the workspace directory}
  local root=$RETREAD_PERSIST_CACHE_ROOT
  case "$root" in
    /oscar/data/stellex/glvov/agrescap/cache/retread*) ;;
    *) echo "retread_fast_env: REFUSING unexpected cache root $root" >&2; return 2;;
  esac
  local d
  for d in uv rattler pixi verdicts built-outputs wheels; do mkdir -p "$root/$d" || return 2; done

  export PIXI_CACHE_DIR=$root/pixi
  export RATTLER_CACHE_DIR=$root/rattler
  export UV_CACHE_DIR=$root/uv
  export UV_LINK_MODE=copy          # safe default; a CERT overrides it AFTER this call

  # SHARED BUILT-OUTPUT STORE (retread >= the p5w commit; older binaries ignore
  # this variable entirely, so setting it here is safe for every harness).
  # conda/outputs is >90% of the backend window and its two existing memos die
  # with the job -- the disk one is written under a fasttmp JOB-scoped cache dir
  # and keyed on the manifest mtime plus the pack's absolute path. This store is
  # keyed on content only, so a workspace staged at a new path can adopt a
  # previously computed result. The supported control is the pack config key
  # `retread-built-output-store`; this env var is the fallback, used here so the
  # harness never has to edit a pack manifest (which would move every pack build
  # hash and make the identity check meaningless).
  # TRADE: an entry freezes the resolution that produced it, exactly as a lock
  # file does. Delete the directory to force fresh resolution; it is rebuildable.
  export RETREAD_BUILT_OUTPUT_STORE=$root/built-outputs

  # THE IMPORT->DISTRIBUTION INDEX AUTHORITY. `courier::wheel_store_root_with`
  # resolves RETREAD_WHEEL_STORE -> XDG_CACHE_HOME -> HOME/.cache, then joins
  # "retread"/"wheels". Every harness on this campaign sets XDG_CACHE_HOME and
  # HOME job-scoped, so the wheel store has started EMPTY in every job we have
  # ever run and auto_imports naming ran on the curated map plus the fallback
  # only. (The reason recorded in LANE-C-WARM-LOG section 7 -- "it lives under
  # RETREAD_BUILD_ROOT / RETREAD_ARTIFACT_ROOT" -- was wrong twice over: those
  # two names do not exist in the backend at all, and coverage was never 0.)
  #
  # MEASURED, section 9.4, one job, three arms over the same store:
  #     arm OFF     store 0 entries before -> 225 after     8 of 309 indexed
  #     arm ON      store 225 before       -> 233 after   222 of 411 indexed = 54%
  # 222 of 411 against 5 of 317 in job 5655631. Names like evdev, hid,
  # fast-simplification, alphashape and newton crossed from LEAD to INJECTABLE
  # purely because the index could name them -- the lead/injectable partition
  # is a function of the store's contents, not of the code.
  #
  # SAFE TO SHARE, and the reason is structural, not empirical: the store is a
  # BLOB store keyed by content. Entries are <sha256>/<filename>, written
  # temp+rename, so two jobs writing the same wheel write the same bytes to the
  # same address and a partial write is never visible under a final name. The
  # courier's own doc comment says blob stores stay shared. Nothing here is
  # resolution state -- a wheel's sha256 does not depend on who fetched it.
  #
  # HELD until fix/p6d-digest-regression merged (that lane was root-causing the
  # hermetic-toolchain failure and every harness sources this file, p6d's own
  # confirmation relock included). p6d merged 2026-09-03; the hold is lifted.
  #
  # RESIDUAL, boarded not hidden (p6c's doc comment carries it too): the index
  # is a genuine naming authority whose content is not knowable at cache-key
  # time, so two injection-ON runs with differently warm stores can still share
  # a decision key. It cannot cause ON/OFF confusion -- the gate is in the key --
  # but it means an injection-ON verdict is a function of store warmth.
  # DISABLED 2026-09-03 07:30 -- the shared store is a self-poisoning
  # writer/reader pair and this export put it in EVERY harness.
  #   writer: `wheel::acquire_wheel_store_fill_lock` creates a zero-byte
  #           `.<wheel>.whl.retread-fill-v1.lock` in `<sha256>/` BEFORE the
  #           wheel exists, so a lock-without-wheel is the normal in-flight
  #           state -- and a crash or a slow fill leaves it there.
  #   reader: the PyPI index chain opens the store path and treats ENOENT as
  #           "the failure was not a package miss", so it ABORTS the whole
  #           chain instead of falling through and re-fetching.
  # Measured 2026-09-03: 6 abandoned entries from a job that died at 01:43-01:44
  # killed jobs 5685024 and BOTH arms of bisect 5686431 hours later; job 5688724
  # then died on an entry whose lock it had created itself 8 minutes earlier,
  # with the store growing 198 -> 428 entries mid-run. A job can poison itself.
  # Reclaiming the debris (tools/wheel_store_reclaim.sh) does not fix it,
  # because it is produced continuously while lanes run.
  #
  # RE-ENABLED 2026-09-03 15:35 -- p6i merged, and BOTH conditions the disable
  # named are met by `fix/p6i-shared-cache-atomicity-b` @ 4b3103b, now in
  # integration/4.12 = c0a87d3:
  #   writer: publication was ALREADY temp-in-the-same-dir + flush + sync_all +
  #           rename (`wheel::atomic_owned_copy`); what was missing is that the
  #           fill-lock placeholder is now UNSELECTABLE as a wheel --
  #           `wheel::WHEEL_STORE_FILL_LOCK_SUFFIX` /
  #           `is_wheel_store_fill_lock_name` own the name, and
  #           `handler::auto_imports_store_wheels` filters the sidecar AND any
  #           zero-length entry.
  #   reader: `pypi::classify_wheel_store_path` decides Absent / ZeroLength /
  #           FillInProgress from the FILESYSTEM, and
  #           `handler::wheel_store_second_chance` waits <=30 s
  #           (`pypi::WHEEL_STORE_FILL_WAIT_SECS`), retries the index ONCE, then
  #           records a MISS and falls through. An error naming no raced store
  #           wheel is still fatal, so "everything became a miss" is excluded.
  # THE PROOF, and it is a lock and not a unit test: job 5719938 (`ME2`) relocked
  # the canonical 27-env manifest from a FRESH workspace against THIS store while
  # it held 696 top-level entries and 487 fill-lock sidecars, including the two
  # lock-without-wheel entries that killed jobs 5685024, 5686431 (both arms) and
  # 5688724. rc=0, wall 1764 s, 27/27 envs, `not a package miss` = 0,
  # `wheel store: entry never filled` = 0, and its lock is BYTE-IDENTICAL
  # (md5 4bcddf2e0819b61c66f983ae642dbad5) to job 5719937's, which locked the
  # same manifest with a JOB-SCOPED store.
  # The prize this buys back: index authority 222/411 = 54% with a warm store
  # against 5/317 without (LANE-C-WARM-LOG 9.4).
  export RETREAD_WHEEL_STORE=$root/wheels

  # Measured as no help (805.0s -> 828.2s route-probe union). Opt-in only.
  if [ "${RETREAD_FAST_ENV_PARALLEL_PROBES:-0}" = 1 ]; then
    export RETREAD_PARALLEL_PROBES=1
    echo "retread_fast_env: RETREAD_PARALLEL_PROBES=1 (experiment; measured as no win)"
  else
    unset RETREAD_PARALLEL_PROBES
  fi

  if [ -n "${RETREAD_FAST_TMP_ROOT:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    local nsh nsdir vlink
    nsh=$(python3 -c "import hashlib,os,sys;print(hashlib.sha256(os.path.realpath(sys.argv[1]).encode()).hexdigest()[:12])" "$ws") || return 2
    nsdir=$RETREAD_FAST_TMP_ROOT/retread-${USER}/$nsh/job-${SLURM_JOB_ID}/caches/retread
    vlink=$nsdir/retread-route-probe-verdicts
    mkdir -p "$nsdir" || return 2
    if [ ! -L "$vlink" ]; then rm -rf "$vlink"; ln -s "$root/verdicts" "$vlink" || return 2; fi
    echo "retread_fast_env: verdict cache $vlink -> $(readlink -f "$vlink")"
  else
    echo "retread_fast_env: RETREAD_FAST_TMP_ROOT or SLURM_JOB_ID unset; verdict cache left job-scoped" >&2
  fi

  echo "retread_fast_env: PIXI_CACHE_DIR=$PIXI_CACHE_DIR"
  echo "retread_fast_env: RATTLER_CACHE_DIR=$RATTLER_CACHE_DIR"
  echo "retread_fast_env: UV_CACHE_DIR=$UV_CACHE_DIR"
  echo "retread_fast_env: RETREAD_BUILT_OUTPUT_STORE=$RETREAD_BUILT_OUTPUT_STORE"
}

########## retread_seed_wheel_store -- THE ONE WAY TO SEED A WHEEL STORE #######
# Seed a JOB-SCOPED wheel store from a PERSISTENT one, as a BYTE COPY.
#
#     retread_seed_wheel_store <src_store> <dst_store>
#
# WHY A JOB-SCOPED STORE AT ALL. `wheel::acquire_wheel_store_fill_lock` writes a
# zero-byte `.<wheel>.whl.retread-fill-v1.lock` sidecar BEFORE the wheel exists.
# If the fill never completes, `handler::wheel_store_second_chance` waits
# WHEEL_STORE_FILL_WAIT_SECS on that sidecar and then returns an Err that
# propagates into the PyPI index-chain aggregate and takes the WHOLE lock down
# (measured: job 5762227, rc=1 at 308 s, on a fill lock dated 26 h earlier).
# The persistent store holds hundreds of these sidecars. So a lane that wants
# the store's WARMTH without its POISON copies the wheels and record sidecars
# and leaves the fill locks and the quarantine dirs behind.
#
# WHY `rsync -aW` AND NEVER `cp -al`. Two independent reasons, and each one on
# its own is fatal:
#   (a) A hardlink shares the INODE. Creating one bumps the ctime of an inode
#       that other live lanes are reading out of the shared store, and any write
#       through the "isolated" copy is a write to the shared store. That is the
#       p6k-b burn on this campaign.
#   (b) `cp -al` copies the WHOLE directory, fill locks included, so the store
#       that was supposed to be clean carries the exact poison it was isolated
#       from. `--exclude` on rsync is what actually drops them.
# `-W` forces whole-file transfer: no delta algorithm, no partial-file linking,
# a real byte copy on every entry.
#
# THE SOURCE STORE IS READ-ONLY TO THIS FUNCTION. It never writes, deletes,
# chmods or relinks anything under <src_store>. The only thing it touches is
# <dst_store>, which it creates.
#
# rc 24 ("some files vanished before they could be transferred") is ACCEPTED: a
# concurrent lane publishing into the shared store races the walk, and the
# entries that vanished were by definition not part of the warm set. Every
# other non-zero rsync rc is fatal.
#
# It prints exactly ONE census line and then ASSERTS on it. Returns non-zero if
# any fill lock or quarantine dir survived, if no wheel arrived (the seed is not
# warm, so the caller gained nothing and should know), or if ANY destination
# wheel has a link count other than 1 -- that last one is the direct, positive
# reader for the "byte copy, not hardlink" claim above.
retread_seed_wheel_store () {
  local src=${1:-} dst=${2:-}
  if [ -z "$src" ] || [ -z "$dst" ]; then
    echo "retread_seed_wheel_store: usage: retread_seed_wheel_store <src_store> <dst_store>" >&2
    return 2
  fi
  if [ ! -d "$src" ]; then
    echo "retread_seed_wheel_store: FATAL source store is not a directory: $src" >&2
    return 2
  fi
  mkdir -p "$dst" || { echo "retread_seed_wheel_store: FATAL cannot create $dst" >&2; return 2; }

  local t0 rc wall
  t0=$(date +%s)
  rsync -aW --exclude='.*.retread-fill-v1.lock' --exclude='*.quarantine-*' "$src/" "$dst/"
  rc=$?
  wall=$(( $(date +%s) - t0 ))
  if [ "$rc" != 0 ] && [ "$rc" != 24 ]; then
    echo "retread_seed_wheel_store: FATAL rsync rc=$rc src=$src dst=$dst (only 0 and 24 are OK)" >&2
    return 3
  fi

  local entries wheels locks quars hard
  entries=$(ls -1U "$dst" 2>/dev/null | wc -l)
  wheels=$(find "$dst" -mindepth 2 -maxdepth 2 -name '*.whl' 2>/dev/null | wc -l)
  locks=$(find "$dst" -mindepth 2 -maxdepth 2 -name '.*.retread-fill-v1.lock' 2>/dev/null | wc -l)
  quars=$(find "$dst" -mindepth 1 -maxdepth 2 -name '*.quarantine-*' 2>/dev/null | wc -l)
  # `grep -c` EXITS 1 when the count is zero, which is the healthy case here, so
  # the pipeline is guarded and the empty-output case is normalised to 0.
  hard=$(find "$dst" -mindepth 2 -maxdepth 2 -name '*.whl' -printf '%n\n' 2>/dev/null | { grep -vc '^1$' || true; })
  [ -n "$hard" ] || hard=0

  echo "### wheel_store seeded src=$src dst=$dst entries=$entries wheels=$wheels fill_locks=$locks hardlinks=$hard quarantines=$quars rc=$rc wall=${wall}s"

  local bad=0
  if [ "$locks" -ne 0 ]; then
    echo "retread_seed_wheel_store: FATAL $locks fill-lock sidecar(s) survived into $dst -- the seed carries the poison it was isolating from" >&2
    bad=1
  fi
  if [ "$quars" -ne 0 ]; then
    echo "retread_seed_wheel_store: FATAL $quars quarantine entr(y/ies) survived into $dst" >&2
    bad=1
  fi
  if [ "$wheels" -le 0 ]; then
    echo "retread_seed_wheel_store: FATAL no wheels under $dst -- the seed is not warm, the caller gained nothing" >&2
    bad=1
  fi
  if [ "$hard" -ne 0 ]; then
    echo "retread_seed_wheel_store: FATAL $hard wheel(s) under $dst have a link count != 1 -- a HARDLINK survived, this store is not isolated from $src" >&2
    bad=1
  fi
  [ "$bad" -eq 0 ] || return 4
  return 0
}
