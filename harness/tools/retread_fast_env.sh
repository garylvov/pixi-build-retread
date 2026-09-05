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

  # C18 -- THE CANONICAL GIT SNAPSHOT STORE. NOT SET HERE, DELIBERATELY, AND
  # THIS NOTE IS THE READER FOR IT.
  #   The trees under `<store>/canonical-git-sources/v3/<repository identity>/
  #   <ref state>/` are what `ensure_canonical_git_snapshot` clones, normalizes,
  #   seals and publishes. They lived under `courier::retread_cache_root()`,
  #   which consults `fasttmp::backend_env_override("RETREAD_CACHE_DIR")` first,
  #   and fasttmp redirects that key into
  #     $RETREAD_FAST_TMP_ROOT/retread-$USER/<workspace hash>/job-$SLURM_JOB_ID/caches/retread
  #   -- so the store was JOB-SCOPED in every job of this campaign (C17.1) and
  #   every relock paid twelve clones. Nobody decided that about Git snapshots:
  #   RETREAD_CACHE_DIR is on fasttmp's scratch-cache list and the one root
  #   deliberately exempted is the wheel blob store above.
  #   MEASURED (LANE-SPEED-LOG C18.5, jobs 5763080 -> 5763081, 8 CPU/72G, SAME
  #   node2350, one store): `canonical_git_snapshot` 992.0s / 54 rows -> 24.4s /
  #   55 rows, `span_path="clone"` 12 rows -> ZERO, and the two pixi.lock files
  #   are BYTE-IDENTICAL (md5 b69a046c1a034868ff6475c2115bbfc6).
  #   TO TURN IT ON, a harness exports the fallback AFTER calling this function:
  #     export RETREAD_GIT_SNAPSHOT_STORE=$RETREAD_PERSIST_CACHE_ROOT/git-snapshots
  #   (the supported control is the pack config key `retread-git-snapshot-store`;
  #   the env var exists so a harness never edits a pack manifest, which would
  #   move that pack's build hash -- the same argument as the built-output store).
  #   NOT SET HERE FOR THREE REASONS, each of which must be closed first:
  #     (a) CLOSED 2026-09-04 09:20. C18 is MERGED: `integration/4.12` = 75a3357
  #         carries `0e4eced` (fix set 11), gate 1715/0/21. Binaries from that tip
  #         onward read the key; older binsnaps still ignore it, so a harness
  #         pinned to an older binsnap must not quote a saving from setting it.
  #         MEASURED ACROSS JOBS AND NODES, not within one job (which is all C18's
  #         own proof showed) -- `bench: canonical_git_snapshot`, ANSI-stripped,
  #         from the BACKEND log, never the lock.log (which carries only ~6 of the
  #         ~55 rows and will read 0 clones on a run that cloned twelve times):
  #             MB3 5765718  no store        55 rows  12 clone  910 829 ms
  #             MB5 5771099  store, 25 dirs  54 rows   1 clone   27 066 ms
  #             MB5 5771101  store, 27 dirs  54 rows   0 clone   17 456 ms
  #         MB3 is the non-vacuity control and lands on C18's own cold reference
  #         (12 clones / 992 043 ms) almost exactly. All three locks are
  #         resolution-identical: `env_version_delta` moved=0 over all 27 envs in
  #         every pairing, including against MI1 5749049 and C18P1 5763080.
  #         STILL NOT EXPORTED HERE, because of (c) below: this line turns the
  #         store on for EVERY lane at once, and nothing deletes from it. The
  #         production call site is `mergeB5/mb5_relock.sh`, which exports it and
  #         GATES on `stat -c %d` first, and that is the shape to copy.
  #     (b) HARDLINK/EXDEV. The private per-entry build tree is hardlinked out of
  #         the canonical tree and `link(2)` returns EXDEV across filesystems, so
  #         a store on a different device from RETREAD_BUILD_ROOT silently costs
  #         the farm and falls back to a full checkout. Every harness here has
  #         both on hpcnfs:/oscar, but a harness that sets this must GATE on
  #         `stat -c %d` agreeing, not assume it. (C16, merged 09-04, deletes the
  #         farm -- re-read this clause against that tip before trusting it.)
  #     (c) NOTHING REAPS IT. **This is now the ONLY thing between the key and a
  #         default-on export here, and it is boarded as C18-1.** Twelve full worktrees per manifest revision, a new
  #         entry per commit of any git source, no eviction anywhere, on a
  #         filesystem whose inode quota read 100.00% on 2026-09-04. See the
  #         INODES hazard above: rm -rf of this subtree is always safe, only slow.
  #
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

########## retread_freeze_repodata -- A PER-QUEUE FROZEN CONDA UNIVERSE ########
# Snapshot the repodata cache ONCE per job into a job-scoped root, point retread
# at it, and stop every refresh for the life of the job.
#
#     retread_freeze_repodata <src cache root> <dst cache root>
#
# THE DEFECT IT CLOSES (MH-1, boarded 23:50 09-03; p6ac-1, boarded 20:45 09-04).
# `retread_fast_env` points RATTLER_CACHE_DIR at ONE persistent root that every
# lane shares, and `repodata::build_sparse` refreshes any document older than
# `REPODATA_TTL` (30 minutes). So a relock that runs for an hour resolves its
# first bundles against one conda universe and its later ones against another,
# and two proof relocks hours apart are not comparable at all. MEASURED tonight
# while p6ad was being built: `pixi-build-retread repodata-universe` read the
# shared cache's conda-forge linux-64 document at 638 500 357 B, and fifteen
# minutes later at 638 505 040 B with a different sha256. Nothing in the lock
# says which one it used. MH.4 charges this class one 75-minute control; C22-3
# is the same family with a 503 as the trigger.
#
# WHAT IT DOES
#   1. `rsync -aW` the repodata DOCUMENTS out of <src>/retread-repodata into
#      <dst>/retread-repodata. Documents only: the `.…retread-fetch-v1.lock`
#      sidecars and the `.…retread-universe-v1.json` memos are excluded, for the
#      same reason `retread_seed_wheel_store` excludes fill locks -- a snapshot
#      must not carry another job's in-flight state. No refresh, no network.
#   2. Export RATTLER_CACHE_DIR=<dst>, which is exactly what
#      `repodata::cache_root_from` reads, so retread consults the snapshot.
#   3. Export RETREAD_REPODATA_FROZEN=1, so `repodata::build_sparse` never
#      refreshes and a pair MISSING from the snapshot is a loud not-consulted
#      rather than a quiet fetch that would silently unfreeze the universe.
#   4. Print the job header with the universe digest, computed BY THE BINARY
#      (`pixi-build-retread repodata-universe`), not by this script. One
#      implementation of the fold: a shell that folded `sha256sum` its own way
#      would be a second comparison rule, and the first time the two disagreed
#      the disagreement would read as a moved universe.
#
# WHY `rsync -aW` AND NEVER `cp -al`, unchanged from `retread_seed_wheel_store`:
# a hardlink shares the inode, so a write through the "frozen" copy is a write
# to the shared cache and creating one bumps the ctime of an inode other live
# lanes are reading (the p6k-b burn). `-W` forces a whole-file byte copy.
#
# COST, and it is the honest trade: the two conda-forge documents are ~892 MB,
# so a freeze costs one 892 MB copy per job and ~0.9 GB of scratch. Measured
# sha256 of the pair on node2341: 4.01 s + 1.52 s.
#
# WHAT IT DOES **NOT** FREEZE -- READ THIS BEFORE CLAIMING A FROZEN RUN.
#   * PIXI's own repodata is a DIFFERENT CACHE and this function does not touch
#     it. Retread reads `<RATTLER_CACHE_DIR>/retread-repodata/*.json` (full
#     decompressed documents, 892 MB on the shared root); pixi reads
#     `<PIXI_CACHE_DIR>/repodata/*.shards-cache-v1` (the sharded protocol, 2 MB
#     of index on the same root). Neither reads the other's files.
#   * Pixi 0.73.0 HAS NO OFFLINE OR NO-REFRESH CONTROL. Measured, not recalled:
#     `pixi lock --help` offers `--json --check --dry-run` and nothing else, and
#     the supported config keys are `cache.root`, `cache.repodata`, `mirrors`,
#     `repodata-config.{disable-bzip2,disable-sharded,disable-zstd}`, … with no
#     offline key anywhere in the list. `cache.repodata` (or PIXI_CACHE_DIR)
#     REDIRECTS pixi's repodata cache; it does not stop the gateway
#     revalidating it over the network on the next solve, so a redirected cache
#     is reuse, not a freeze.
#   * `--frozen` IS NOT THIS. In pixi it means "install from the lock without
#     re-solving", so it does not apply to a relock at all; using the word for
#     both is how this gets miscommunicated.
#   * The ONE mechanism that genuinely freezes pixi is `mirrors`. p6ad wrote
#     that this means a `file://` snapshot including the shard index. BOTH
#     HALVES OF THAT SENTENCE ARE WRONG and p6af measured why: pixi refuses a
#     `file://` mirror outright ("URL scheme is not allowed"), and against a
#     mirror it falls back from the shard index to plain `repodata.json`, so no
#     shard index is needed at all. The shipped mechanism is
#     `retread_freeze_channel_mirror` + `retread_serve_channel_mirror` below.
#   So: a frozen run freezes the half that decides the vendored set (retread's
#   route probes and co-solves) and records the other half's provenance rather
#   than controlling it.
#
# It prints exactly ONE census line and then ASSERTS on it: non-zero if no
# document arrived, if a fetch lock or a memo survived, or if any destination
# document has a link count other than 1 -- the direct positive reader for the
# "byte copy, not hardlink" claim above.
retread_freeze_repodata () {
  local src=${1:-} dst=${2:-}
  if [ -z "$src" ] || [ -z "$dst" ]; then
    echo "retread_freeze_repodata: usage: retread_freeze_repodata <src cache root> <dst cache root>" >&2
    return 2
  fi
  local srcd=$src/retread-repodata dstd=$dst/retread-repodata
  if [ ! -d "$srcd" ]; then
    echo "retread_freeze_repodata: FATAL source has no repodata cache: $srcd" >&2
    return 2
  fi
  mkdir -p "$dstd" || { echo "retread_freeze_repodata: FATAL cannot create $dstd" >&2; return 2; }

  local t0 rc wall
  t0=$(date +%s)
  # rc 24 ("some files vanished") is ACCEPTED for the same reason as in
  # retread_seed_wheel_store: a concurrent lane republishing a document races
  # the walk, and a document that vanished was not part of the frozen set.
  # RULE ORDER IS LOAD-BEARING: rsync takes the FIRST matching rule, and the
  # content-hash memo is named `.<doc>.json.retread-universe-v1.json`, i.e. it
  # ends in `.json`. `--include='*.json'` first would therefore copy the memos
  # in, and a memo copied in describes the SOURCE file's stat tuple -- which is
  # exactly the poisoned-memo case the retread guards refuse. Dotfiles are
  # excluded FIRST.
  # rc 24 ("some files vanished") AND rc 23 ("some files/attrs were not
  # transferred") are both the CONCURRENT-REPUBLISH RACE, not a broken copy.
  # Measured 2026-09-04, job 5853901: another lane republished
  # `conda_forge--noarch--373f6e9e6cea9d02.json` mid-walk and the sender got
  # `Stale file handle (116)` -> rc 23, and the whole freeze refused. `invalidate`
  # + `write_atomic` make a document a NEW INODE on every republish, so a reader
  # holding the old handle sees ESTALE; the file is there and readable on the very
  # next open. So both codes RETRY rather than fail, and the name-set check below
  # is what makes the retry safe: a document that is in the source and not in the
  # snapshot is fatal no matter what rsync returned.
  local attempt
  for attempt in 1 2 3; do
    rsync -aW --exclude='.*' --exclude='*.part.*' --include='*.json' \
          --exclude='*' "$srcd/" "$dstd/"
    rc=$?
    case "$rc" in
      0) break;;
      23|24) echo "retread_freeze_repodata: rsync rc=$rc on attempt $attempt (concurrent republish); retrying" >&2;;
      *) break;;
    esac
  done
  wall=$(( $(date +%s) - t0 ))
  if [ "$rc" != 0 ] && [ "$rc" != 23 ] && [ "$rc" != 24 ]; then
    echo "retread_freeze_repodata: FATAL rsync rc=$rc src=$srcd dst=$dstd (only 0, 23 and 24 are OK)" >&2
    return 3
  fi
  # THE READER FOR THE RETRY. Every document name the source has must be in the
  # snapshot; a tolerated rc is only tolerable because this cannot pass without it.
  local missing
  missing=$(comm -23 <(cd "$srcd" && ls -1 *.json 2>/dev/null | sort) \
                     <(cd "$dstd" && ls -1 *.json 2>/dev/null | sort))
  if [ -n "$missing" ]; then
    echo "retread_freeze_repodata: FATAL document(s) in $srcd that never reached $dstd:" >&2
    printf '%s\n' "$missing" >&2
    return 3
  fi

  local docs locks memos hard bytes
  docs=$(find "$dstd" -maxdepth 1 -type f -name '*.json' ! -name '.*' 2>/dev/null | wc -l)
  locks=$(find "$dstd" -maxdepth 1 -name '.*retread-fetch-v1.lock' 2>/dev/null | wc -l)
  memos=$(find "$dstd" -maxdepth 1 -name '.*retread-universe-v1.json' 2>/dev/null | wc -l)
  bytes=$(du -sb "$dstd" 2>/dev/null | cut -f1)
  hard=$(find "$dstd" -maxdepth 1 -type f -name '*.json' -printf '%n\n' 2>/dev/null | { grep -vc '^1$' || true; })
  [ -n "$hard" ] || hard=0

  echo "### repodata frozen src=$srcd dst=$dstd docs=$docs bytes=$bytes fetch_locks=$locks memos=$memos hardlinks=$hard rc=$rc wall=${wall}s"

  local bad=0
  if [ "$docs" -le 0 ]; then
    echo "retread_freeze_repodata: FATAL no repodata documents under $dstd -- there is nothing to freeze" >&2
    bad=1
  fi
  if [ "$locks" -ne 0 ]; then
    echo "retread_freeze_repodata: FATAL $locks fetch-lock sidecar(s) survived into $dstd" >&2
    bad=1
  fi
  if [ "$memos" -ne 0 ]; then
    echo "retread_freeze_repodata: FATAL $memos content-hash memo(s) survived into $dstd -- a memo copied in describes the SOURCE file's stat tuple, not this one" >&2
    bad=1
  fi
  if [ "$hard" -ne 0 ]; then
    echo "retread_freeze_repodata: FATAL $hard document(s) under $dstd have a link count != 1 -- a HARDLINK survived, this snapshot is not isolated from $srcd" >&2
    bad=1
  fi
  [ "$bad" -eq 0 ] || return 4

  export RATTLER_CACHE_DIR=$dst
  export RETREAD_REPODATA_FROZEN=1
  echo "retread_freeze_repodata: RATTLER_CACHE_DIR=$RATTLER_CACHE_DIR RETREAD_REPODATA_FROZEN=1"

  # THE JOB HEADER. Printed by the binary, so it uses the same fold the backend
  # rows use. A binary that cannot print it is not fatal to the freeze -- the
  # freeze is the rsync and the two exports -- but it IS reported, because a
  # frozen run whose header nobody can read is a run nobody can explain later.
  local snap=${RETREAD_FREEZE_BINARY:-}
  if [ -n "$snap" ] && [ -x "$snap" ]; then
    "$snap" repodata-universe --cache-root "$dst" || \
      echo "retread_freeze_repodata: WARNING $snap repodata-universe failed; no universe digest in this job header" >&2
  else
    echo "retread_freeze_repodata: WARNING RETREAD_FREEZE_BINARY unset or not executable; no universe digest in this job header" >&2
  fi
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

######## retread_freeze_channel_mirror -- FREEZING PIXI'S HALF OF THE UNIVERSE ##
# p6af. `retread_freeze_repodata` above freezes RETREAD's conda universe and
# says, correctly, that pixi's half cannot be frozen and that the only candidate
# mechanism is `mirrors`. It also says that mechanism is a `file://` snapshot.
# THAT SECOND CLAIM IS WRONG AND IS CORRECTED HERE, measured on the hold
# (job 5846309, node1829, pixi 0.73.0):
#
#   * `mirrors` -> `file:///…` is REFUSED. pixi routes a mirror through its HTTP
#     client, and the client rejects the scheme:
#         builder error for url (file:///…/conda-forge/linux-64/
#           repodata_shards.msgpack.zst)  ...  URL scheme is not allowed
#     A `file://` CHANNEL (in the manifest) works fine -- but changing the
#     manifest's channels changes the URLs recorded in the lock, so it can never
#     produce a lock comparable to a network one. `mirrors` is transparent: the
#     lock keeps `https://prefix.dev/conda-forge/` whatever the mirror is.
#   * `mirrors` -> `http://127.0.0.1:<port>/…` WORKS, offline. Measured with the
#     outside blocked by a dead proxy (HTTPS_PROXY/HTTP_PROXY/ALL_PROXY =
#     http://127.0.0.1:9, NO_PROXY=127.0.0.1) and the non-vacuity control run
#     first: the same lock with NO mirror fails in 4 s on
#     `repodata_shards.msgpack.zst` with `Connection refused (os error 111)`.
#   * PIXI FALLS BACK TO repodata.json. Against the mirror it asks for exactly
#     two URLs per subdir -- `repodata_shards.msgpack.zst` (404) then
#     `repodata.json` (200) -- so a static mirror needs NO shard index and NO
#     shards. That is the whole protocol requirement.
#
# THE ONE THING A CLASSIC DOCUMENT DOES NOT CARRY IS `run_exports`, AND THE
# CERTIFIED LOCK HAS 2432 OF THEM. prefix.dev publishes run_exports only inside
# the sharded protocol: `<subdir>/run_exports.json` is a 404 on every channel we
# use, and retread's own `retread-repodata/*.json` snapshots contain the string
# `run_exports` zero times. A mirror built from those documents alone locks
# correctly -- the conda URL set was IDENTICAL to the network lock, 0 diff lines
# -- but the lock differs in exactly the run_exports blocks (84 diff lines on a
# 48-package probe manifest, every one of them a run_exports line).
# pixi DOES honour a `run_exports` key found in a classic repodata.json record:
# injecting `{"noarch": ["p6af-probe-marker"]}` into one record put the marker
# in the lock. So the freeze CARRIES THE FIELD ACROSS from the sharded cache,
# and then:
#
#   MEASURED IDENTITY (job 5846309): two offline mirror locks and one same-window
#   network lock all md5 373d726c8153ac4247905faf3362dd53, diff 0 lines.
#   Wall 8-9 s offline.
#
#     retread_freeze_channel_mirror <lock> <dst mirror root> <pixi repodata cache>
#
# <pixi repodata cache> is a `<PIXI_CACHE_DIR>/repodata` directory that has been
# warmed BY A NETWORK LOCK OF THIS SAME WORKSPACE -- that is where the shards,
# and therefore the run_exports, come from. The intended shape of a freeze job is
# one network lock into a job-local PIXI_CACHE_DIR, then this call over its
# cache: the mirror is then guaranteed to carry run_exports for every name that
# solve touched, which is every name the lock records.
#
# WHAT IT COSTS, measured per channel/subdir (content-length from the live
# channels, 2026-09-04, the seven channels the certified lock declares):
#     conda-forge         linux-64 638 505 040   noarch 253 787 755   aarch64 296 780 114
#     robostack-jazzy     linux-64  10 421 274   noarch         174   aarch64   9 872 784
#     robostack-humble    linux-64   5 192 968   noarch         174   aarch64   4 479 803
#     nvidia              linux-64   3 194 395   noarch     449 167   aarch64   2 887 727
#     pytorch (anaconda)  linux-64   1 367 363   noarch      52 669   aarch64      44 757
#     pytorch (prefix)    linux-64   1 230 019   noarch      67 490   aarch64      19 251
#     pixi-build-backends linux-64     513 431   noarch      30 797   aarch64     478 632
#   linux-64 + noarch only   914 812 716 B (872 MiB)
#   + linux-aarch64        1 229 375 784 B (1.15 GiB)
# One mirror per QUEUE, not per job: every lane points at the same frozen root.
#
# WHY IT DOES NOT SHIP A TRIMMED UNIVERSE, even though a trim is ~700x smaller
# (conda-forge's 1794 lock records are 1 270 523 B against 892 292 795 B, and a
# 33 KB trim of the probe manifest still locked byte-identically): a trimmed
# document answers "no such package" to retread's route probes, and the route
# probe verdicts are what decide the vendored set (p6ac). A universe that is
# smaller than the one the certified lock was solved against is a DIFFERENT
# universe, which is the C22-3 / p6ac-1 divergence mechanism, not a saving.
# `pixi_trim_repodata.py` exists to MEASURE that cost; nothing calls it in a
# production path.
#
# WHAT IT STILL DOES NOT FREEZE, boarded not hidden:
#   * run_exports coverage is the freeze job's shard coverage. A later solve in
#     the queue that reaches a package NAME the reference lock never selected
#     gets that package's record from the full document but with no
#     run_exports, where the network would have supplied one. It cannot fail --
#     the universe is complete -- but that one record's block can differ. The
#     reader is the census line: `shards_present` vs `shards_absent`.
#   * The PyPI side. uv's index is not mirrored by anything here.
retread_freeze_channel_mirror () {
  local lock=${1:-} dst=${2:-} pxcache=${3:-}
  if [ -z "$lock" ] || [ -z "$dst" ] || [ -z "$pxcache" ]; then
    echo "retread_freeze_channel_mirror: usage: retread_freeze_channel_mirror <lock> <dst mirror root> <pixi repodata cache dir>" >&2
    return 2
  fi
  [ -f "$lock" ]     || { echo "retread_freeze_channel_mirror: FATAL no lock at $lock" >&2; return 2; }
  [ -d "$pxcache" ]  || { echo "retread_freeze_channel_mirror: FATAL no pixi repodata cache at $pxcache" >&2; return 2; }
  case "$dst" in
    /oscar/data/stellex/glvov/agrescap/cache/retread/*)
      echo "retread_freeze_channel_mirror: REFUSING to write inside the shared persistent cache: $dst" >&2; return 2;;
  esac
  local here; here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local merge=$here/pixi_merge_run_exports.py
  [ -f "$merge" ] || { echo "retread_freeze_channel_mirror: FATAL missing $merge" >&2; return 2; }

  # The channels are the lock's own, and the subdirs are the lock's own platforms
  # plus noarch. Nothing is guessed and nothing is configured twice.
  local chans subdirs
  chans=$(grep -o '^ *- url: https://[^ ]*' "$lock" | sed 's/.*url: //' | sed 's:/*$::' | sort -u)
  # The subdirs come from the RECORD URLs, not from the `platforms:` block: a
  # lock whose platform name equals its subdir omits the `subdir:` key entirely
  # (measured -- a `subdir:`-only rule built a mirror with noarch and nothing
  # else, and the lock then failed on the first linux-64 package).
  subdirs=$( { grep -o '\- conda: https\?://[^ ]*' "$lock" |
                 sed 's|.*://||' | awk -F/ '{print $(NF-1)}'; echo noarch; } | sort -u)
  [ -n "$chans" ] || { echo "retread_freeze_channel_mirror: FATAL no channel urls in $lock" >&2; return 2; }

  mkdir -p "$dst" || return 2
  # p6af-2. The shard cache only indexes a (channel, subdir) pair some solve on
  # this machine actually fetched: the canonical workspace declares 21 pairs and
  # the shared cache carried 19, so a strict merge aborts the whole freeze on two
  # pairs that contribute no records to the lock either. With
  # RETREAD_MIRROR_ALLOW_MISSING_INDEX=1 such a pair is mirrored WITHOUT
  # run_exports and counted in `pairs_no_index` on the census line below -- the
  # reader, so a pair that lost its run_exports can never be silent. Default is
  # STRICT, unchanged.
  local mflags=""
  [ "${RETREAD_MIRROR_ALLOW_MISSING_INDEX:-0}" = 1 ] && mflags=--allow-missing-index
  local t0 c s name base tmp bytes docs=0 fetched=0 total=0 noidx=0
  t0=$(date +%s)
  for c in $chans; do
    # The mirror directory is keyed on HOST AND PATH, never on the last segment:
    # this workspace declares BOTH https://prefix.dev/pytorch and
    # https://conda.anaconda.org/pytorch, and a last-segment key silently merges
    # two different channels into one directory.
    name=$(printf '%s' "$c" | sed 's|^https\?://||; s|/|__|g')
    base=/${c#*://*/}                       # the channel's path, e.g. /conda-forge
    for s in $subdirs; do
      mkdir -p "$dst/$name/$s" || return 2
      tmp=$dst/$name/$s/.repodata.json.raw
      # Always fetched from the channel itself. Retread's own snapshot holds the
      # same bytes, but its file names key on the channel's LAST SEGMENT and so
      # cannot tell the two pytorch channels apart -- reusing it would be a
      # silent wrong-channel read.
      curl -fsSL "$c/$s/repodata.json" -o "$tmp" || {
        echo "retread_freeze_channel_mirror: FATAL fetch failed $c/$s/repodata.json" >&2; return 3; }
      fetched=$((fetched+1))
      # The channel HOST is passed as well as its path: two of this workspace's
      # channels have the SAME path (`/pytorch`) on different hosts, and the
      # shard index keys on the path alone.
      python3 "$merge" "$tmp" "$dst/$name/$s/repodata.json" "$pxcache" "$base/$s/" \
        "$(printf '%s' "$c" | sed -e 's|^https\?://||' -e 's|/.*$||')" $mflags \
        2> "$dst/$name/$s/.merge.stderr" \
        || { echo "retread_freeze_channel_mirror: FATAL run_exports merge failed for $name/$s" >&2
             cat "$dst/$name/$s/.merge.stderr" >&2; return 3; }
      cat "$dst/$name/$s/.merge.stderr" >&2
      grep -q 'index=absent' "$dst/$name/$s/.merge.stderr" && noidx=$((noidx+1))
      rm -f "$tmp"
      bytes=$(stat -c %s "$dst/$name/$s/repodata.json")
      total=$((total+bytes)); docs=$((docs+1))
    done
  done

  # ONE digest over the mirror, computed the same way every time: sha256 of the
  # sorted "<relative path> <sha256>" table. A digest over a `find` order would
  # not be a comparison rule at all.
  local digest
  digest=$( (cd "$dst" && find . -name repodata.json -printf '%P\n' | sort | while read -r p; do
              echo "$p $(sha256sum "$p" | cut -d' ' -f1)"; done) | sha256sum | cut -d' ' -f1)
  echo "### channel_mirror frozen dst=$dst docs=$docs fetched=$fetched pairs_no_index=$noidx bytes=$total wall=$(( $(date +%s)-t0 ))s digest=$digest"
  if [ "$docs" -eq 0 ]; then
    echo "retread_freeze_channel_mirror: FATAL no documents written" >&2; return 4
  fi
  export RETREAD_CHANNEL_MIRROR=$dst
  export RETREAD_CHANNEL_MIRROR_DIGEST=$digest
  return 0
}

######## retread_serve_channel_mirror / retread_pixi_mirror_config ############
# The mirror is served over loopback HTTP because `mirrors` will not take a
# file:// URL (see above). Both halves must run in the SAME shell as the lock:
# an `srun --overlap` step reaps detached children, so a server started in one
# step is gone by the next.
#
#     retread_serve_channel_mirror <mirror root> <port>      # sets RETREAD_MIRROR_URL/_PID
#     retread_pixi_mirror_config   <lock> <job HOME>         # writes $HOME/.pixi/config.toml
#
# THE CONFIG GOES IN A JOB-LOCAL HOME AND NOWHERE ELSE. imprint-data's
# `.pixi/config.toml` is read-only to this campaign and carries
# `run-post-link-scripts` + `[concurrency]`; pixi MERGES system < global < local,
# so `[mirrors]` in the job HOME's global config lands beside the workspace's own
# keys without touching them. Measured: with imprint-data's config.toml copied
# verbatim into the staged workspace, `pixi config list` showed all three keys
# and the lock was still md5 373d726c8153ac4247905faf3362dd53.
retread_serve_channel_mirror () {
  local root=${1:-} port=${2:-}
  [ -d "$root" ] || { echo "retread_serve_channel_mirror: FATAL no mirror at $root" >&2; return 2; }
  [ -n "$port" ] || { echo "retread_serve_channel_mirror: usage: <mirror root> <port>" >&2; return 2; }
  ( cd "$root" && exec python3 -m http.server "$port" --bind 127.0.0.1 ) > "${TMPDIR:-/tmp}/channel-mirror-$port.log" 2>&1 &
  RETREAD_MIRROR_PID=$!
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    curl -fs -o /dev/null "http://127.0.0.1:$port/" && break
    sleep 1
  done
  curl -fsS -o /dev/null "http://127.0.0.1:$port/" || {
    echo "retread_serve_channel_mirror: FATAL server on $port never answered" >&2
    kill "$RETREAD_MIRROR_PID" 2>/dev/null; return 3; }
  RETREAD_MIRROR_URL=http://127.0.0.1:$port
  export RETREAD_MIRROR_PID RETREAD_MIRROR_URL
  echo "### channel_mirror serving $RETREAD_MIRROR_URL pid=$RETREAD_MIRROR_PID root=$root"
}

######## retread_serve_conda_deny_proxy -- BLOCK THE CHANNELS, NOT THE WORLD ###
# p6af blocked pixi with a dead proxy plus a NO_PROXY allowlist of everything
# pixi legitimately needs. On the canonical workspace that allowlist has to name
# the conda->PyPI mapping service, three find-links pages, the GitHub release
# host and whatever GitHub redirects release assets to -- and every omission is a
# dead job AFTER the conda half has already resolved (jobs 5853300, 5854363,
# 5855234, each one a different missing host). State the actual requirement
# instead: pixi must not reach a CONDA CHANNEL. `conda_deny_proxy.py` refuses
# exactly those hosts and tunnels the rest, and its log names every host pixi
# asked for, which is a positive record rather than the absence of an error.
#
#     retread_serve_conda_deny_proxy <deny host,host,...> <port> <log path>
#         # sets RETREAD_DENY_PROXY_URL / _PID / _LOG
#
# The caller still exports the proxy variables itself, and must keep
# NO_PROXY=127.0.0.1 so the loopback channel mirror is reached directly.
retread_serve_conda_deny_proxy () {
  local deny=${1:-} port=${2:-} logp=${3:-}
  [ -n "$deny" ] && [ -n "$port" ] && [ -n "$logp" ] || {
    echo "retread_serve_conda_deny_proxy: usage: <deny hosts> <port> <log path>" >&2; return 2; }
  local here; here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local prox=$here/conda_deny_proxy.py
  [ -f "$prox" ] || { echo "retread_serve_conda_deny_proxy: FATAL missing $prox" >&2; return 2; }
  : > "$logp"
  python3 "$prox" "$port" "$deny" "$logp" &
  RETREAD_DENY_PROXY_PID=$!
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    grep -q 'proxy listening' "$logp" && break
    sleep 1
  done
  grep -q 'proxy listening' "$logp" || {
    echo "retread_serve_conda_deny_proxy: FATAL proxy never came up (see $logp)" >&2
    kill "$RETREAD_DENY_PROXY_PID" 2>/dev/null; return 3; }
  RETREAD_DENY_PROXY_URL=http://127.0.0.1:$port
  RETREAD_DENY_PROXY_LOG=$logp
  export RETREAD_DENY_PROXY_PID RETREAD_DENY_PROXY_URL RETREAD_DENY_PROXY_LOG
  echo "### deny_proxy serving $RETREAD_DENY_PROXY_URL pid=$RETREAD_DENY_PROXY_PID deny=$deny log=$logp"
}

retread_pixi_mirror_config () {
  local lock=${1:-} jobhome=${2:-} pixibin=${3:-}
  [ -f "$lock" ] || { echo "retread_pixi_mirror_config: FATAL no lock at $lock" >&2; return 2; }
  [ -n "$jobhome" ] || { echo "retread_pixi_mirror_config: usage: <lock> <job HOME>" >&2; return 2; }
  [ -n "${RETREAD_MIRROR_URL:-}" ] || { echo "retread_pixi_mirror_config: FATAL RETREAD_MIRROR_URL unset -- serve the mirror first" >&2; return 2; }
  case "$jobhome" in
    /users/glvov|/users/glvov/*) echo "retread_pixi_mirror_config: REFUSING to write the real HOME: $jobhome" >&2; return 2;;
  esac
  # WHERE PIXI ACTUALLY LOOKS, and `$HOME/.pixi/config.toml` ALONE IS NOT IT.
  # p6af measured the config injection on a spike that let PIXI_HOME default, so
  # the job HOME was the right place. Every relock harness on this campaign
  # EXPORTS a job-scoped PIXI_HOME (and XDG_CONFIG_HOME), and pixi reads its
  # global config out of those, not out of `$HOME/.pixi`. Measured, job 5851478:
  # the config was written, the seven mirror bases all answered 200, and the lock
  # still went straight to `https://prefix.dev/conda-forge/linux-64/
  # repodata_shards.msgpack.zst` and died on the dead proxy in 3 s. So the same
  # `[mirrors]` block is written to every job-local location pixi consults, and
  # the `pixi config list` check below is the reader that refuses if pixi still
  # cannot see it.
  local dests="$jobhome/.pixi/config.toml"
  [ -n "${PIXI_HOME:-}" ] && case "$PIXI_HOME" in /users/glvov|/users/glvov/*) ;; *) dests="$dests $PIXI_HOME/config.toml";; esac
  [ -n "${XDG_CONFIG_HOME:-}" ] && case "$XDG_CONFIG_HOME" in /users/glvov|/users/glvov/*) ;; *) dests="$dests $XDG_CONFIG_HOME/pixi/config.toml";; esac
  mkdir -p "$jobhome/.pixi" || return 2
  # THE KEY MUST BE THE SAME ONE retread_freeze_channel_mirror BUILT: host AND
  # path, scheme stripped, slashes to `__`. Until 2026-09-04 the sed program in
  # here was written inside a double-quoted string as `s|^https\\?://||`, which
  # reaches sed as `\\?` -- a literal backslash followed by a literal `?` -- so
  # the scheme was NOT stripped and every mirror URL named a directory called
  # `https:____prefix.dev__conda-forge` that the freeze had never created. Every
  # request would have 404'd. It was never caught because the guard writes its
  # own `[mirrors]` line inline and so never called this function. The reader is
  # the 200-check at the bottom of this function, which is only possible because
  # the mirror is already being served when the config is written.
  local names=""
  { echo "[mirrors]"
    grep -o '^ *- url: https://[^ ]*' "$lock" | sed 's/.*url: //' | sed 's:/*$::' | sort -u |
      while read -r c; do
        echo "\"$c\" = [\"$RETREAD_MIRROR_URL/$(printf '%s' "$c" | sed -e 's|^https\?://||' -e 's|/|__|g')\"]"
      done
  } > "$jobhome/.pixi/config.toml"
  local dst
  for dst in $dests; do
    mkdir -p "$(dirname "$dst")" || return 2
    [ "$dst" = "$jobhome/.pixi/config.toml" ] || cp "$jobhome/.pixi/config.toml" "$dst" || return 2
    echo "### channel_mirror config $dst channels=$(grep -c '^"' "$dst")"
  done

  # READER. Every mirror base this config names must be a directory the server
  # actually serves. A config that points at a directory the freeze never wrote
  # is not a mis-configuration that shows up later as a slow lock -- it is a
  # lock that cannot resolve a single package, an hour into a job.
  local bad=0 base
  names=$(sed -n 's/.*= \["\([^"]*\)"\].*/\1/p' "$jobhome/.pixi/config.toml")
  for base in $names; do
    if curl -fs -o /dev/null "$base/"; then
      echo "### channel_mirror config check 200 $base/"
    else
      echo "retread_pixi_mirror_config: FATAL mirror base not served: $base/ (the freeze never wrote this directory)" >&2
      bad=1
    fi
  done
  [ "$bad" -eq 0 ] || return 3

  # THE READER FOR THE WHOLE INJECTION. A config pixi does not read is a config
  # that does not exist, and the only authority on what pixi reads is pixi.
  if [ -n "$pixibin" ] && [ -x "$pixibin" ]; then
    local listed
    listed=$("$pixibin" config list 2>&1)
    printf '%s\n' "$listed" | sed 's/^/  pixi config list: /'
    if printf '%s' "$listed" | grep -q "$RETREAD_MIRROR_URL"; then
      echo "### channel_mirror config VISIBLE to pixi ($RETREAD_MIRROR_URL found in \`pixi config list\`)"
    else
      echo "retread_pixi_mirror_config: FATAL pixi does not see the mirrors block -- \`pixi config list\` never names $RETREAD_MIRROR_URL, so the lock would go to the network" >&2
      return 4
    fi
  else
    echo "retread_pixi_mirror_config: WARNING no pixi binary passed; the mirrors block is NOT confirmed visible to pixi" >&2
  fi
}
