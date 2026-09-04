#!/usr/bin/env bash
### EVIDENCE BEGIN
# p8_warm_inject.sh -- WARM-STORE injection-ON validation, two sequential arms.
#
# DERIVED BY SUBSTITUTION from tools/phase_template/phaseN_relock.sh (the
# certified relock half, itself derived from jobs 5597671/5597694) with the
# two-arm run_arm() shape lifted from p4n-wire-phase1b/p4n_wire_phase1b.sh
# (the A/B harness that ran jobs 5549254 .. 5569638). Everything the template
# does for ONE arm this script does TWICE, sequentially, on one node, against
# one manifest, with the SAME backend binary.
#
#   arm OFF : canonical manifest, persistent caches, RETREAD_AUTO_IMPORTS UNSET
#   arm ON  : identical, plus RETREAD_AUTO_IMPORTS=1  (exact string "1";
#             mod.rs var("RETREAD_AUTO_IMPORTS").map(|v| v=="1").unwrap_or(false))
#
# WHY THIS RUN EXISTS. HANDOFF section 6 records that every injection-ON failure
# to date was measured against a COLD store: index authority was 1-4% of rows
# (0.9% of arm-B injected rows; 13/610 even in a full arm-A run; 529/529
# indexed=false in the earliest probe), so 99%+ of observed failures exercised
# the PEP-503 fallback naming path that a warm production store does not take.
# The recorded exit-ramp criterion for flipping the gate was a WARM-STORE
# validation, which was impossible until the persistent caches under
# agrescap/cache/retread/ existed. They exist now (retread_fast_env.sh; job
# 5598763 arm C locked in 69s warm vs 2865s cold; the template's own smoke
# 5611846 locked a FRESH workspace against those warm caches in 2366s).
#
# CACHE SAFETY -- THE ONE NEW MECHANISM HERE.
# The shared warm store must not be poisoned by injection-derived state. So:
#   * arm OFF uses the shared root  agrescap/cache/retread/            (normal use)
#   * arm ON  uses an ISOLATED root agrescap/cache/retread-injection-on/
#     seeded AFTER arm OFF finishes, so arm ON starts from a store at least as
#     warm as arm OFF's. Seeding is:
#       - uv, pixi, rattler : `cp -al` (hardlink). These hold content-addressed
#         downloads and index metadata; they are injection-independent, and uv
#         writes via tmp+rename so an in-place refresh cannot write THROUGH a
#         hardlink into the shared copy. Hardlinks make the seed cheap in bytes.
#       - verdicts          : real `cp -a`. This is the ONE semantically loaded
#         cache (route-probe verdicts keyed on (validity_key, digest)) and the
#         one an injected closure could plausibly dirty, so it gets a genuine
#         copy with no shared inodes.
#     retread_fast_env.sh's own guard accepts this root: its case pattern is
#     /oscar/data/stellex/glvov/agrescap/cache/retread* .
#   * The isolated root is removed at the very end of this job (nothing depends
#     on this job, so there is no afterok epilogue hazard of the kind that cost
#     job 5596128 5152 s). If the job is killed first, the tree is rebuildable
#     and safe to delete by hand.
#
# NO SELF-CLEANUP of the job-scoped cert*/ws.* roots here: a cleanup job is
# submitted separately if wanted. This job removes only its OWN isolated cache
# seed.
#
# /usr/bin/time -v per arm, to its own file. sacct MaxRSS is unusable on this
# filesystem (every job reports ~100% of its cgroup cap).
#
# MEMORY: 72G. Measured relock peak process RSS 8.85 / 8.83 / 4.70 GB cold,
# 2.87 GB warm (jobs 5597671, 5594283, 5598763 arms A and C). Two arms run
# SEQUENTIALLY, so the peak is one arm's, not the sum.
#
#     env -u SLURM_JOB_ID sbatch --partition=batch --qos=normal \
#         --cpus-per-task=16 --mem=72G --time=03:00:00 \
#         --job-name=p8-warm-inject \
#         --output=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p8-warm-inject/slurm-%j.out \
#         ./p8_warm_inject.sh
#
# NEVER edit this file while a job is running it -- copy it aside first.
#
# P11WI PROVENANCE (2026-09-02, 19:20 EDT). This file is a COPY of
# p8-warm-inject/p8_warm_inject.sh (job 5638422) with EXACTLY four
# substitutions inside the SUBSTITUTE block and nothing else:
#   TAG   -> P16WI                       (so job roots cannot collide)
#   D     -> $T/p16-warm-reinject
#   SNAP  -> $T/binsnaps/fix-p5x-debug-cpython/pixi-build-retread
#   EXPECT_SHA / SNAP_COMMIT -> 67ba7131.../e7cd52d
# LANE-BLOCKER-DEBUGCPYTHON-LOG.md section 9 directs exactly this: re-run
# 5638422's harness UNCHANGED against the fixed binsnap. 5638422's A/B was void
# because its gate-OFF baseline was red for the *_debug_cpython reason, which
# has nothing to do with the injection gate. The isolated ON-arm cache root is
# re-seeded at run time by seed_isolated_cache() (rm -rf then cp -al from the
# CURRENT shared store), so ON starts as warm as OFF.
#
# P12WI DIFFERS FROM P11WI IN ONE MORE WAY, and it is a measured harness fix,
# not a change to the experiment. P11WI (job 5655631) spent ~10 min in each of
# its four `du -sh "$CACHEROOT"/*` calls over a 71 G uv + 26 G pixi NFS store.
# That output gates nothing, it is not read by any comparison, and it is what
# made the job look wedged for an hour. Replaced by
# `timeout 60 du --inodes -s`. The seed's `cp -al` now also prints its own
# per-tree wall, because in 5655631 it was the single longest phase (>2 h for
# the uv tree) and no row said so while it ran.
#
# P14WI KEEPS P13WI'S ISOLATION STRATEGY. seed_isolated_cache no longer
# hardlink-clones 232,759 directories over ~113 minutes; it symlinks the three
# content-addressed download caches to the shared store and real-copies only the
# two trees that can hold an arm-dependent value. The full argument, with the
# code expressions that decide each tree, is in the comment above that function.
# Net effect: the ON arm's setup drops from ~113 min to seconds, the ON arm
# starts EXACTLY as warm as OFF instead of approximately so, and the poisoning
# surface SHRINKS -- no earlier copy isolated built-outputs at all.
#
# OA6 PROVENANCE (2026-09-03). A COPY of the previous a3b arm's harness with
# substitutions inside the SUBSTITUTE block ONLY: TAG, D, CERT_MANIFEST,
# EXPECT_CERT_MD5/LINES/ADD, ISO_CACHE, LEFTOVER_RE. Pack diffs, binsnap, OFF
# baseline and every gate are unchanged.
#
# P6N PROVENANCE (2026-09-03). A COPY of the a3c arm's harness (oncert-a3c/
# oa6_relock.sh, job 5722719) with SUBSTITUTE-block edits: TAG, D,
# CERT_MANIFEST, EXPECT_CERT_MD5/LINES/DEL/ADD, ISO_CACHE, LEFTOVER_RE, and
# SNAP/EXPECT_SHA_PIN/SNAP_COMMIT left as MARKED PLACEHOLDERS for the p6n
# binsnap. Pack diffs (a3b + a3b2), the job-scoped wheel store, 72G/16cpu and
# every gate are unchanged.
#
# THE MANIFEST IS THE a3b CERTIFIED PACKET, NOT THE a3c FIXTURE. a3c added a
# hand declaration `tensorboard = ">=2.8,<2.21"` to probe ONE edge. p6n is a
# BACKEND fix (a second workspace-conda-fact pass) and must be measured against
# the manifest with NO hand declaration, or the fix and the fixture cannot be
# told apart. So CERT_MANIFEST is b1-scratch/pixi.toml.a3b: 9 deleted / 10 added
# against canonical, md5 744db6f17947dd8dee78871ff2732b18, 1004 lines -- all
# four numbers recomputed here with the same commands the gates below run, not
# copied from a3c.
#
# TWO ROOT FIXES CARRIED BY THIS COPY, both outside the SUBSTITUTE block,
# because job 5722719 proved the first one is a live defect:
#
#  (a) THE ISO_CACHE GUARD WENT STALE AND KILLED 5722719. seed_isolated_cache()
#      and the epilogue each carried a HARD-CODED literal cache path
#      (...retread-injection-on-oa5) while the SUBSTITUTE block set ISO_CACHE to
#      ...-oa6. Every file gate passed, the job reached the node, and it died at
#      `FATAL: refusing to seed unexpected root .../retread-injection-on-oa6`
#      after the whole preflight had said all clear. The guard's real job is to
#      stop an `rm -rf` outside the cache tree; pinning it to one arm's exact
#      suffix does not add safety, it only adds a second copy of a knob that the
#      SUBSTITUTE block already owns. Both call sites now call ONE function,
#      iso_cache_guard(), which DERIVES the permitted root from $SHARED_CACHE
#      and $ISO_CACHE -- there is no second literal left to keep in sync -- and
#      section 0 calls the SAME function as a dry seed check, so DRY_RUN=1
#      exercises the very guard that killed 5722719 without creating, copying
#      or removing anything.
#
#  (b) THE SNAPSHOT GATE NOW UNDERSTANDS AN UNFILLED PLACEHOLDER. With SNAP
#      unset the old gate exits 8 at the very first check, so none of the
#      manifest, residual-pin, probe or pack-diff gates could run at all and a
#      harness could not be validated before its binary existed. The gate now
#      recognises the exact string __P6N_BINSNAP__: under DRY_RUN=1 it says so
#      loudly and continues to every other gate; in a REAL run it is a FATAL,
#      so the placeholder can never reach a node. Filling it in is three lines
#      in the SUBSTITUTE block: SNAP, EXPECT_SHA_PIN, SNAP_COMMIT.
#
# THIS ARM IS A MEASUREMENT FIXTURE, NOT A PROPOSED MANIFEST. The clean-manifest
# ruling stands and no hand declaration is being proposed for protobuf or for
# tensorboard. This arm exists to isolate ONE edge and prove the diagnosis
# below by removing it.
#
# WHAT THE PREVIOUS ARM MEASURED. 24 of 27 environments resolved; viral-gpu
# died on
#   isaaclab-viral-pack would constrain
#   protobuf >4.21.0,!=5.26.0,!=5.28.0,!=5.29.0,>=6.31.1,<8
# The >=6.31.1 clause has exactly one origin in that job's backend log:
# injection auto-bundles tensorboard into the pack ("auto-bundled into
# isaaclab-viral-pack dep=tensorboard version=2.21.0") and PyPI tensorboard
# 2.21.0 declares protobuf>=6.31.1,<8.0.0. The gate-OFF lock of the SAME
# manifest emits protobuf >=5,!=5.26.0,!=5.28.0,!=5.29.0,<8 -- it does NOT omit
# the entry; it emits a band the installed conda protobuf 5.29.3 satisfies.
#
# WHY CONDA CANNOT MEET >=6.31.1 HERE, measured by standalone solve probe:
# viral-gpu's declared conda set plus the pack's binding constrains solves and
# picks protobuf 5.29.3; add protobuf >=6.31.1,<8 and it FAILS, naming
# ray-core ==2.49.1 -> libgrpc >=1.71.0,<1.72.0a0 -> libgrpc 1.71.0 ->
# libprotobuf >=5.29.3,<5.29.4.0a0, against conda protobuf 6.31.1+ which
# constrains libprotobuf ==6.31.1+. ray-core ==2.49.1 is this pack's own
# [package.build.config.retread-overrides] ray = "==2.49.1".
#
# The fixture bounds tensorboard to what the conda provider can serve
# (conda-forge's newest tensorboard is 2.20.0, protobuf >=3.19.6,!=4.24.0;
# there is no conda 2.21.0 at all). If viral-gpu then resolves and a lock is
# emitted, the tensorboard-2.21 edge is proved to be the whole of the failure
# and nothing else in the 27-env workspace blocks an injection-ON lock.
### EVIDENCE END
set -uo pipefail

### SUBSTITUTE: BEGIN -- MANIFEST, PROBES, EXPECT_*  (edit ONLY between these markers)

# C29 PROVENANCE (2026-09-04). A COPY of c28-phase1/c28_relock.sh (the C28/B5
# injection-ON-on-canonical arm, job 5829952) with edits INSIDE THIS BLOCK AND
# NOWHERE ELSE. Everything outside these markers -- staging, the pack-diff
# preflight, the iso-cache guard, the arm, the strict-gate reporting, the B4
# gate, the handoff -- is byte-identical to that file, which is itself a
# SUBSTITUTE-only copy of bcert-phase1/bcert_relock.sh.
#
# WHAT THIS ARM ASKS, and the ONE variable that separates it from C28.
# LANE-C-WARM-LOG 28.2/28.6: injection-ON on the CANONICAL manifest with NO pack
# diffs is RED. Exactly one environment failed, pm-isaaclab, on exactly one
# name, typing-extensions: two co-resident packs each advertising a genuine
# declared band, isaaclab-2.3x-pack `~=4.12,>=4.12.2,<4.13` against
# protomotions-deps-pack `>4.5.0,>=4.15.0,<5`, empty intersection. isaaclab-gpu-
# latest and viral-gpu were NEVER REACHED, because the lock aborts on the first
# pypi solve failure (C28-2). C28-1 sized the remedy in one line: the A3b2
# cession, isaaclab-2.3x-pack's retread-drop-deps gaining typing-extensions.
#
# This arm applies THAT AND NOTHING ELSE:
#
#     manifest  = the CANONICAL imprint-data/pixi.toml, byte-for-byte, md5
#                 9711eb990bfe211d498d1635a60e0d07, UNCHANGED from C28
#     pack diffs= ONE, the A3b2 cession. No A3b manifest packet, no A3b pack
#                 diffs, no second pass.
#     injection = ON  (RETREAD_AUTO_IMPORTS=1)
#     strict    = OFF via the env harness door, and the arm PRINTS why
#     stores    = job-scoped, wheel store left EMPTY
#     binsnap   = p6ab-616cfec, the same binary C28 measured
#
# THE ONE THING THIS ARM HAD TO BUILD, AND WHY IT IS NOT A SECOND VARIABLE.
# a3b-work/isaaclab-2.3x-pack.pixi.toml.a3b2.diff (md5
# 11ab83a65d10e7250db497836a621126) DOES NOT APPLY to the canonical pack. It was
# cut against the A3b-PATCHED pack, whose retread-drop-deps already reads
# ["pydantic","click","wandb","platformdirs","fsspec"]; the canonical pack reads
# ["pydantic","click"]. Measured, not assumed: `patch -p0` on a scratch copy of
# imprint-data/pypi-packs/isaaclab-2.3x-pack/pixi.toml (md5
# 74d3aaa6374154be07b42d37fe0b3f32) on node1948, 2026-09-04 --
# `Hunk #1 FAILED at 60. 1 out of 1 hunk FAILED`, rc=1, file unchanged. So the
# A3b2 CESSION was rebased onto canonical as
# a3b-work/isaaclab-2.3x-pack.pixi.toml.a3b2only.diff. Its cession comment text
# is carried verbatim from the a3b2 diff; the only functional line it changes is
# retread-drop-deps, canonical ["pydantic","click"] -> ["pydantic","click",
# "typing-extensions"], and NOTHING ELSE. Measured on the same node:
# `diff canonical patched | grep -vE '^> #'` yields exactly that one pair of
# lines. wandb, platformdirs and fsspec are DELIBERATELY still emitted by this
# pack, exactly as canonical emits them -- carrying them would be the A3b pack
# diff, which this arm must not apply.
#
# It is a MEASUREMENT FIXTURE, not a proposed manifest and not a cert
# precondition. No afterok cert is chained behind it: the question is whether
# a3b2 alone turns canonical injection-ON green and, if not, what the next
# failure is, and a failing arm answers that as well as a passing one.

TAG=C29P1                                    # roots become cert${TAG}-<job>-<ARM> / ws.${TAG}-<job>-<ARM>; verified absent under /oscar/data/stellex/glvov/retread before submission, so this arm cannot collide with any prior one
T=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11  # task root
D=$T/c29-phase1                              # THIS harness's own directory
SRC_WS=/oscar/data/stellex/glvov/imprint-data           # READ-ONLY canonical source tree
CLEANED=$T/b1-scratch/pixi.toml.orig                    # the preserved canonical copy; md5-gated below
EXPECT_CLEANED_MD5=9711eb990bfe211d498d1635a60e0d07
EXPECT_MANIFEST_LINES=1003
EXPECT_DEL=0                                            # canonical == under test: zero diff
EXPECT_ADD=0
EXPECT_ENVS=27
EXPECT_JETSON_ROWS=1

# --- THE ARM'S MANIFEST: THE CANONICAL ONE AGAIN, FROM A SECOND PATH ---------
# Unchanged from C28. The arm stages $CERT_MANIFEST. Here that is a FRESH `cp`
# of $SRC_WS/pixi.toml taken at harness-build time into this harness's own
# directory, and it is a DIFFERENT FILE from $CLEANED on purpose: the two
# manifest gates below then prove, from two independent paths, that the bytes
# the arm will solve are the canonical bytes. Recomputed here from THIS copy:
#
#   md5sum $D/pixi.toml.canon        -> 9711eb990bfe211d498d1635a60e0d07
#   wc -l < $D/pixi.toml.canon       -> 1003
#   diff $SRC_WS/pixi.toml $D/pixi.toml.canon | grep -c '^< '  -> 0
#   diff $SRC_WS/pixi.toml $D/pixi.toml.canon | grep -c '^> '  -> 0
CERT_MANIFEST=$D/pixi.toml.canon
# ONE PACK DIFF, THE A3b2 CESSION, REBASED ONTO THE CANONICAL PACK.
# Both md5s below were measured on node1948 by reproducing the preflight's own
# sequence (cp the canonical pack to scratch, `patch -p0`, md5sum) -- they are
# not copied from any doc:
#   diff file md5   5c116bc032f62e38e1a822b44284ef06
#   canonical pack  74d3aaa6374154be07b42d37fe0b3f32
#   result md5      25e00bb4f886c7de4e37bfd24d21df53   (173 lines)
# protomotions-deps-pack gets NO diff: it declares the surviving band
# `>4.5.0,>=4.15.0,<5` and changes nothing, and an empty patch is a gate that
# cannot fail. PACK_DIFFS2 is EMPTY: the a3b2 second pass exists only because
# the a3b first pass came before it, and there is no first pass here.
PACK_DIFF_DIR=$T/a3b-work
PACK_DIFFS=(
  "isaaclab-2.3x-pack:5c116bc032f62e38e1a822b44284ef06:25e00bb4f886c7de4e37bfd24d21df53"
)
PACK_DIFF_SUFFIX=a3b2only                               # file name: <pack>.pixi.toml.<suffix>.diff
PACK_DIFF_SUFFIX2=none
PACK_DIFFS2=()
EXPECT_CERT_MD5=9711eb990bfe211d498d1635a60e0d07
EXPECT_CERT_LINES=1003
EXPECT_CERT_DEL=0                                       # canonical -> arm manifest: identical
EXPECT_CERT_ADD=0
# THE RESIDUAL-PIN GATE, INVERTED, AND STILL NON-VACUOUS. Unchanged from C28.
# Every pre-C28 arm asserted these tokens were GONE, because its manifest
# deleted them. This arm asserts they are ALL PRESENT: nine matching lines,
# measured `grep -cE <re> $SRC_WS/pixi.toml` -> 9. Those nine are precisely the
# declared pins the certified A3b packet removes and this arm keeps, so the gate
# fails loudly if anyone hands this harness the certified packet by mistake,
# which is the substitution error it exists to catch.
CERT_RESIDUAL_RE='^pillow = "==10\.4\.0"$|^networkx = "==3\.4\.2"$|^sentry-sdk = "==2\.0\.0"$|^openmesh = "==1\.2\.1"$|^numpy = ">=1\.21\.1"$'
EXPECT_CERT_RESIDUAL=9
PROBES=$T/bfinal-phase1/probes.bfinal.tsv               # the bfinal probe set (26 rows); phase 1 only counts it

# --- THERE IS NO TARGET LOCK. THE PIN IS THE GATE-OFF PROOF, AS A BASELINE ---
# Unchanged from C28. This arm reproduces nothing: no injection-ON lock of the
# CANONICAL manifest has ever existed, so a byte-identity target would be a
# fiction. What is pinned below is the newest gate-OFF tip proof of the SAME
# manifest -- mergeB10's MBA-5817831, binsnap cand-1e8a06e, canonical manifest
# md5 9711eb99..., no pack diffs, lock_rc=0, wall 3727 s -- and the B4 gate is
# used here as an INSTRUMENT, not as an acceptance test:
#   * the pin proves the baseline file on disk is still the artefact it names
#     (that check is real and fatal if the baseline moved);
#   * the fresh lock is compared to it and the sorted-line diff, the url-set
#     sizes and the per-env env_version_delta are PRINTED -- which is the
#     number the report needs: the list of packages injection added or moved on
#     TODAY's manifest, per environment;
#   * the acceptance rule will then almost certainly REFUSE (exit 5), because
#     an injection-ON lock is not expected to equal a gate-OFF one. That
#     refusal is EXPECTED and costs nothing: no cert is chained behind this
#     job, the fresh lock is already written into $D/artifacts before the gate
#     runs, and the refusal prints the delta rather than hiding it.
# Stated here, before the run, so the exit code is read as designed and not as
# a surprise.
PRESERVED_LOCK=$T/mergeB10/artifacts/pixi.lock.cert
EXPECT_LOCK_SHA256=386baecfbac50bea2d1de7864ab6c1c9c5ef2eaed5b1cacccc3e7c74fb104db2
EXPECT_LOCK_BYTES=2758192

# --- instruments -------------------------------------------------------------
EVD=$T/b3-phase1/env_version_delta.py
# Unchanged from C28: every package 17-22 named as a failure class, so the delta
# table itself carries the evidence for the report.
EVD_PACKAGES="pillow networkx packaging numpy typing-extensions filelock fsspec protobuf tensorboard sentry-sdk lxml platformdirs wandb"
BASE_LOCK=$SRC_WS/pixi.lock
# THE GATE-OFF SEMANTIC BASELINE: the same file as the pin above. There is no
# OFF arm in this job, and mergeB10 is the newest gate-OFF lock of exactly this
# manifest with exactly no pack diffs, so it is the only valid comparison. Note
# what that makes the delta mean here: it is injection-ON-plus-a3b2 against
# gate-OFF-plus-nothing, so a moved row is attributable to injection OR to the
# cession, and the cession moves no installed byte by its own argument.
OFF_BASELINE=$T/mergeB10/artifacts/pixi.lock.cert
OFF_BASELINE_MD5=e566308965704232044c0d0a44cfb692

# --- toolchain ---------------------------------------------------------------
PIXI=/users/glvov/.pixi/bin/pixi.real                    # bypass the flock shim
# --- BINSNAP: the p6ab snapshot, UNCHANGED FROM C28 -- and that is the point.
# --- C28 measured the red on this exact binary; changing it here would make the
# --- pack diff and the binary two variables instead of one. sha256sum and the
# --- commit file were re-read from the snapshot directory, not copied.
SNAP=$T/binsnaps/p6ab-616cfec/pixi-build-retread
EXPECT_SHA_PIN=6ef2f5209022738f99cc785ece1c52970aca708cdae9d7b317542a78161464c5
SNAP_COMMIT=616cfec
UVBIN=/oscar/data/stellex/glvov/tasks/retread-cold-solve/verify_fixes/artifacts/uvbin
FAST_ENV=$T/tools/retread_fast_env.sh

# --- cache roots: shared for arm OFF, isolated clone for arm ON --------------
SHARED_CACHE=/oscar/data/stellex/glvov/agrescap/cache/retread
# Its OWN isolated root, verified absent before submission, so this job cannot
# collide with any queued or running arm.
ISO_CACHE=/oscar/data/stellex/glvov/agrescap/cache/retread-injection-on-c29p1

# --- THE WHEEL STORE: EMPTY, job-scoped ---------------------------------------
# The store is job-scoped under $XDG_CACHE_HOME and starts EMPTY every job (see
# the block in run_arm). `none` leaves it that way, so no gate-OFF built output
# and no earlier arm's wheel can stand in for one this arm had to produce.
WHEEL_STORE_SEED=none

# --- THE STRICT ZERO-GATE: OFF, DELIBERATELY, AND WHY IT IS THE ENV DOOR -----
# `retread-auto-imports-strict` is DEFAULT ON whenever injection is on, and it
# REFUSES a request that dropped a detected root. This arm must LAND whatever
# the manifest produces so the failure text can be read; a strict refusal would
# replace that text with a refusal notice. Strict is therefore OFF, and the arm
# REPORTS what strict would have refused -- C28 got INDETERMINATE there because
# the run died before the summary printed, and reaching that summary is one of
# the things a green arm here would buy.
#
# WHY THE ENV OVERRIDE AND NOT THE CONFIG KEY, measured rather than preferred.
# `config::AUTO_IMPORTS_STRICT_KEY` = "retread-auto-imports-strict" is a
# `[package.build.config]` key: the backend reads it from `params.configuration`,
# i.e. from EACH PACK'S OWN manifest. Setting it there means editing pack
# manifests, and this arm's variable is ONE pack edit -- the a3b2 cession line
# and its comment, and nothing else. Turning strict off through the config key
# would put a second, unrelated edit into the same file the measurement is
# about. The backend's own comment on `auto_imports_strict_decision` names the
# env override as the harness door for precisely this case, so that is the door
# used, and it is named here rather than left to be discovered from a log line.
AUTO_IMPORTS_STRICT=0

# --- leftover-token self-check ------------------------------------------------
# WIDENED with the immediate predecessor's tokens: C28P1 / c28p1, jobs 5829952
# and 5829955, the c28 harness directory and file name, and the placeholder
# pack-diff directory `dev-null-no-pack-diffs` that C28 used to say "no diffs"
# and that this arm must not still name.
# KEPT ON PURPOSE, though this arm legitimately uses them: `a3b-work` and
# `a3b2`. The self-check SKIPS this SUBSTITUTE block, so the declarations above
# are invisible to it; an occurrence of either token anywhere in the BODY would
# still be a stale literal of the kind that killed job 5722719, and the guard
# should keep catching it. NOT added, for the same reasons C28 stated: bare
# `c28`, because this block's provenance text truthfully records where the file
# came from; bare `bcert`, because the body's handoff line records the
# provenance string `bcert_relock.sh` on purpose; and `A3b`, because the body's
# pack-diff FATAL messages are named that and are now REACHABLE and correct.
LEFTOVER_RE='5829952|5829955|C28P1|c28p1|c28-phase1|c28_relock|dev-null-no-pack-diffs|5769781|5769782|5779261|5779262|B4P1U|b4p1u|B4P1Z|b4p1z|5799527|5799528|a3b-work|a3b2|pixi.toml.a3b|744db6f1|bc897c83|p6u-work|5764452|5764453|26ac32b|69db5f83|O7P6UA|O7P6UB|o7ua_|o7ub_|oncert-p6ua|oncert-p6ub|p6u-26ac32b|5757173|5757174|d23fe7b|5ef5a275|O7P6TA|O7P6TB|o7ta_|o7tb_|oncert-p6ta|oncert-p6tb|p6t-d23fe7b|5752280|5752281|5752282|5748915|0dcda13|950a5f89|O7P6S|o7s_|oncert-p6s|p6s-0dcda13|o7p6s|5745086|4ee963d|151aff46|O7P6R|o7r_|oncert-p6r|p6r-4ee963d|5742776|2f74345|ce5a9d9e|O7P6Q|o7q_|oncert-p6q|p6q-2f74345|5739415|627de7f|17a25da8|O7P6NB|o7b_|oncert-p6nb|p6n-b-627de7f|5727660|af53707|ad3c912d|OA6|oa6|oncert-a3c|a3c|5722719|731209d7|OA5|oa5|oncert-a3b5|5720294|OCA3B|OA4|oa4_|oncert-a3b4|P6GOC|p6g-oncert|p6j-oncert|P6JOC|5562f7c|e1e7065|p16|P16WI|p16_warm|67ba7131|p5x|P5X|e7cd52d|bfinal|BFP1|BFP2|bfp1|bfp2|b1c|b1-phase|b1b-phase|b2-phase|b2b-phase|b3-phase|ctl-phase|eff-phase|/b1_|/b2_|/b3_|/ctl_|p5sab|P5SAB|p5t_abc|P5TABC|certB3P1|PHASEN|phaseN|2cfec88d|57105d38|P4NW1B|p4nw1b|P8WI|p8-warm|p8_warm|a5ed78a1|a01c49f'
### SUBSTITUTE: END

### LEFTOVER-CHECK BEGIN
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

# iso_cache_guard <caller> -- THE ONE PLACE the isolation root's shape is decided.
#
# WHY IT EXISTS. The predecessor arm passed every DRY_RUN gate, reached a node,
# and died in 4 s with rc=4 on "refusing to seed unexpected root", printed by
# seed_isolated_cache itself -- NOT by tools/retread_fast_env.sh, whose own
# guard is the far wider case pattern
# /oscar/data/stellex/glvov/agrescap/cache/retread* and would have accepted the
# same path. That harness carried the permitted root name TWICE as a hard-coded
# literal (seed + epilogue), both still naming the arm before it, while the
# SUBSTITUTE block had renamed ISO_CACHE. A knob copied into a second place is a
# knob that will go stale, and this one cost a node slot. The full account is in
# the EVIDENCE block at the top of this file.
#
# THE FIX. The expected root is DERIVED from $SHARED_CACHE and $ISO_CACHE here
# and nowhere else: the root must sit under "<shared store>-injection-on-<arm>"
# and must not BE the shared store. That is the whole safety property an
# `rm -rf "$ISO_CACHE"` needs, and it cannot go stale when TAG changes.
# It is called THREE times: once in section 0 as a dry check (so DRY_RUN=1
# exercises this guard without creating or copying anything), once by
# seed_isolated_cache, once by the epilogue before the rm.
iso_cache_guard() {
  local who=$1
  if [ "$ISO_CACHE" = "$SHARED_CACHE" ]; then
    echo "FATAL ($who): ISO_CACHE is the SHARED store $ISO_CACHE -- refusing"; exit 4
  fi
  case "$ISO_CACHE" in
    "$SHARED_CACHE"-injection-on-?*) ;;
    *) echo "FATAL ($who): refusing unexpected isolation root $ISO_CACHE (want ${SHARED_CACHE}-injection-on-<arm>)"; exit 4;;
  esac
  echo "### ISO_CACHE GUARD ($who): $ISO_CACHE accepted -- derived from SHARED_CACHE=$SHARED_CACHE, distinct from it"
}

# DRY_RUN=1 runs every gate that reads only files -- snapshot sha, both manifest
# gates, the residual-pin gate, the probe set, and the PACK-DIFF PREFLIGHT below
# -- then exits 0 WITHOUT touching a cache, a workspace or the backend. Two jobs
# (5697524, 5707111) died in 2-3 s inside exactly these gates because there was
# no way to run them anywhere but a compute node. There is now.
DRY_RUN=${DRY_RUN:-0}
if [ "$DRY_RUN" = "1" ]; then
  J=DRYRUN
  echo "### DRY_RUN=1 -- file gates only, no cache, no workspace, no backend"
else
  J=${SLURM_JOB_ID:?missing Slurm job id}
fi
A=$D/artifacts
P=${TAG}-${J}

CQ=/oscar/runtime/bin/checkquota
[ -x "$CQ" ] || CQ=$(command -v checkquota 2>/dev/null || echo true)
mkdir -p "$A"
hostname; date -Is
echo "### ${TAG} WARM-STORE INJECTION A/B JOB=$J NODE=${SLURM_JOB_NODELIST:-none} nproc=$(nproc) mem=$(free -g|awk '/^Mem:/{print $2"G"}') glibc=$(ldd --version|head -1)"
echo "### inode quota BEFORE:"; "$CQ" 2>/dev/null | grep -E 'data\+stellex|^Name' | head -4

########## 0. GATES ##########
case "$SRC_WS" in /oscar/data/stellex/glvov/imprint-data) ;; *) echo "FATAL bad SRC_WS $SRC_WS"; exit 4;; esac
# BINSNAP GATE, PLACEHOLDER-AWARE. The arm is written before its binary exists,
# so SNAP may still be the literal __P6N_BINSNAP__. Under DRY_RUN=1 that is
# announced and the run continues to every OTHER file gate -- the whole reason
# the dry run exists. In a REAL run it is FATAL, so an unfilled placeholder can
# never reach a node. Nothing here can be bypassed by accident: the branch keys
# on the exact placeholder string and on DRY_RUN, and both are printed.
if [ "$SNAP" = "__P6N_BINSNAP__" ]; then
  echo "### BINSNAP GATE DEFERRED: SNAP is still the unfilled placeholder __P6N_BINSNAP__"
  echo "###   fill SNAP / EXPECT_SHA_PIN / SNAP_COMMIT in the SUBSTITUTE block before submitting"
  [ "$DRY_RUN" = "1" ] || { echo "FATAL: placeholder SNAP in a REAL run -- fill SNAP/EXPECT_SHA_PIN/SNAP_COMMIT first"; exit 8; }
  EXPECT_SHA=PLACEHOLDER
else
  [ -f "$SNAP" ] || { echo "FATAL: pre-made snapshot $SNAP missing"; exit 8; }
  GOT_SHA=$(sha256sum "$SNAP" | awk '{print $1}')
  [ -n "$GOT_SHA" ] || { echo "FATAL: could not sha256sum $SNAP"; exit 8; }
  case "$EXPECT_SHA_PIN" in
    __P6N_EXPECT_SHA__) echo "FATAL: SNAP was filled in but EXPECT_SHA_PIN is still a placeholder"; exit 8;; esac
  case "$SNAP_COMMIT" in
    __P6N_SNAP_COMMIT__) echo "FATAL: SNAP was filled in but SNAP_COMMIT is still a placeholder"; exit 8;; esac
  if [ -n "$EXPECT_SHA_PIN" ]; then
    [ "$GOT_SHA" = "$EXPECT_SHA_PIN" ] || { echo "FATAL: snapshot sha $GOT_SHA != pinned $EXPECT_SHA_PIN"; exit 8; }
    echo "### backend snapshot sha PINNED and matched"
  else
    echo "### backend snapshot sha DERIVED from \$SNAP at run time (no pin set)"
  fi
  EXPECT_SHA=$GOT_SHA
  echo "### backend snapshot OK: $SNAP sha256=$GOT_SHA commit=$SNAP_COMMIT"
  ls -l "$SNAP"; "$SNAP" --version 2>&1 | head -2
fi
[ -f "$FAST_ENV" ] || { echo "FATAL: persistent-cache snippet $FAST_ENV missing"; exit 8; }
[ -f "$CLEANED" ] || { echo "FATAL: manifest under test $CLEANED missing"; exit 9; }
echo "### manifest md5: $(md5sum "$CLEANED")"
GOT_CM=$(md5sum "$CLEANED" | awk '{print $1}')
[ "$GOT_CM" = "$EXPECT_CLEANED_MD5" ] || { echo "FATAL: manifest md5 $GOT_CM != $EXPECT_CLEANED_MD5"; exit 9; }
for f in "$EVD" "$BASE_LOCK"; do
  [ -e "$f" ] || { echo "FATAL: missing required path $f"; exit 9; }
done
echo "### canonical-vs-under-test manifest diff (want EXACTLY $EXPECT_DEL deleted, $EXPECT_ADD added):"
diff "$SRC_WS/pixi.toml" "$CLEANED"
DEL=$(diff "$SRC_WS/pixi.toml" "$CLEANED" | grep -c '^< ')
ADD=$(diff "$SRC_WS/pixi.toml" "$CLEANED" | grep -c '^> ')
echo "### manifest diff counts: deleted=$DEL (want $EXPECT_DEL) added=$ADD (want $EXPECT_ADD)"
[ "$DEL" = "$EXPECT_DEL" ] && [ "$ADD" = "$EXPECT_ADD" ] || { echo "FATAL: manifest is not the canonical one"; exit 9; }
[ -f "$CERT_MANIFEST" ] || { echo "FATAL: certified manifest $CERT_MANIFEST missing"; exit 9; }
GOT_CERT=$(md5sum "$CERT_MANIFEST" | awk '{print $1}')
[ "$GOT_CERT" = "$EXPECT_CERT_MD5" ] || { echo "FATAL: cert manifest md5 $GOT_CERT != $EXPECT_CERT_MD5"; exit 9; }
echo "### certified manifest md5: $GOT_CERT lines: $(wc -l < "$CERT_MANIFEST") (want $EXPECT_CERT_LINES)"
[ "$(wc -l < "$CERT_MANIFEST")" = "$EXPECT_CERT_LINES" ] || { echo "FATAL: cert manifest line count"; exit 9; }
echo "### canonical-vs-certified diff (want EXACTLY $EXPECT_CERT_DEL deleted, $EXPECT_CERT_ADD added):"
diff "$SRC_WS/pixi.toml" "$CERT_MANIFEST"
CDEL=$(diff "$SRC_WS/pixi.toml" "$CERT_MANIFEST" | grep -c '^< ')
CADD=$(diff "$SRC_WS/pixi.toml" "$CERT_MANIFEST" | grep -c '^> ')
echo "### cert diff counts: deleted=$CDEL (want $EXPECT_CERT_DEL) added=$CADD (want $EXPECT_CERT_ADD)"
[ "$CDEL" = "$EXPECT_CERT_DEL" ] && [ "$CADD" = "$EXPECT_CERT_ADD" ] || { echo "FATAL: certified manifest is not the 9-line packet"; exit 9; }
CRES=$(grep -cE "$CERT_RESIDUAL_RE" "$CERT_MANIFEST")
echo "### RESIDUAL-PIN GATE: deleted tokens still present in the certified manifest = $CRES (want $EXPECT_CERT_RESIDUAL)"
[ "$CRES" = "$EXPECT_CERT_RESIDUAL" ] || { echo "FATAL: a deleted pin survives in the certified manifest"; exit 9; }
echo "### and the SAME tokens in the canonical manifest (want > 0, or this gate is vacuous): $(grep -cE "$CERT_RESIDUAL_RE" "$CLEANED")"
[ "$(grep -cE "$CERT_RESIDUAL_RE" "$CLEANED")" -gt 0 ] || { echo "FATAL: residual gate is vacuous -- canonical has none of these tokens either"; exit 9; }
[ -f "$PROBES" ] || { echo "FATAL: probe set $PROBES missing"; exit 9; }
echo "### probe set: $PROBES rows=$(wc -l < "$PROBES")"
[ -d "$SHARED_CACHE" ] || { echo "FATAL: shared warm cache $SHARED_CACHE missing -- this run is pointless cold"; exit 9; }
# DRY SEED CHECK. seed_isolated_cache runs deep into a real job; its guard is
# the one thing about it decidable from strings alone, so decide it here, under
# DRY_RUN too. Creates nothing, copies nothing, removes nothing.
iso_cache_guard "dry-seed preflight"
echo "### ISO_CACHE exists_now=$([ -e "$ISO_CACHE" ] && echo YES || echo no)  (the seed rm -rf's and rebuilds it)"
[ -x "$UVBIN/uv" ] || { echo "FATAL: uv $UVBIN/uv missing"; exit 7; }
BACKEND=$SNAP
########## 0b. PACK-DIFF PREFLIGHT ##########
# stage_ws applies these diffs on the compute node, 10+ minutes into the job,
# after the whole workspace is staged. Every failure mode there -- a missing
# diff, a changed diff, a diff that no longer applies, a result that is not the
# file we reasoned about -- is knowable HERE, from files alone, in a second.
# This reproduces stage_ws's exact sequence (patch -p0 in the pack directory,
# first pass then second pass) on a scratch copy and asserts the same md5s.
PRE=$(mktemp -d "${TMPDIR:-/tmp}/packdiff-preflight.XXXXXX") || { echo "FATAL: mktemp"; exit 9; }
PRE_BAD=0
for spec in "${PACK_DIFFS[@]}" ${PACK_DIFFS2[@]+"${PACK_DIFFS2[@]}"}; do
  pk=${spec%%:*}
  [ -d "$PRE/$pk" ] && continue
  mkdir -p "$PRE/$pk"
  cp "$SRC_WS/pypi-packs/$pk/pixi.toml" "$PRE/$pk/pixi.toml" || { echo "FATAL: no source pack $pk"; exit 9; }
done
pre_apply() {
  local sfx=$1; shift
  local spec pk rest dmd5 rmd5 df got
  for spec in "$@"; do
    pk=${spec%%:*}; rest=${spec#*:}; dmd5=${rest%%:*}; rmd5=${rest##*:}
    df=$PACK_DIFF_DIR/$pk.pixi.toml.$sfx.diff
    if [ ! -f "$df" ]; then echo "  PREFLIGHT FAIL $pk/$sfx: diff $df missing"; PRE_BAD=$((PRE_BAD+1)); continue; fi
    got=$(md5sum "$df" | awk '{print $1}')
    if [ "$got" != "$dmd5" ]; then echo "  PREFLIGHT FAIL $pk/$sfx: diff md5 $got != $dmd5"; PRE_BAD=$((PRE_BAD+1)); continue; fi
    if ! ( cd "$PRE/$pk" && patch -p0 --force --no-backup-if-mismatch pixi.toml < "$df" >/dev/null 2>&1 ); then
      echo "  PREFLIGHT FAIL $pk/$sfx: patch does not apply"; PRE_BAD=$((PRE_BAD+1)); continue; fi
    got=$(md5sum "$PRE/$pk/pixi.toml" | awk '{print $1}')
    if [ "$got" != "$rmd5" ]; then echo "  PREFLIGHT FAIL $pk/$sfx: result md5 $got != $rmd5"; PRE_BAD=$((PRE_BAD+1)); continue; fi
    echo "  PREFLIGHT OK   $pk/$sfx: diff $dmd5 -> result $got"
  done
}
echo "### PACK-DIFF PREFLIGHT (dry application on a scratch copy of $SRC_WS/pypi-packs):"
pre_apply "$PACK_DIFF_SUFFIX"  "${PACK_DIFFS[@]}"
if [ ${#PACK_DIFFS2[@]} -gt 0 ]; then
  pre_apply "$PACK_DIFF_SUFFIX2" "${PACK_DIFFS2[@]}"
else
  echo "  second pass: NOT CONFIGURED (PACK_DIFFS2 empty) -- first pass only"
fi
rm -rf "$PRE"
[ "$PRE_BAD" = 0 ] || { echo "FATAL: $PRE_BAD pack-diff preflight failure(s) -- fix before the job burns a node"; exit 9; }
echo "### PACK-DIFF PREFLIGHT: all clear"

if [ "$DRY_RUN" = "1" ]; then
  echo "### DRY_RUN complete: every file gate passed. Nothing was staged, cached or locked."
  exit 0
fi

echo "### shared warm cache contents BEFORE anything:"
for d in uv pixi rattler verdicts; do
  printf '  %-10s %s entries\n' "$d" "$(ls -A "$SHARED_CACHE/$d" 2>/dev/null | wc -l)"
done

########## HELPERS ##########

# stage_ws <ws> -- pristine workspace, canonical manifest, no pixi.lock.
# Same function for both arms, so staging cannot bias the comparison.
# ARM_MANIFEST / ARM_MD5 / ARM_LINES are set immediately before each run_arm
# call; stage_ws installs whichever manifest the arm is testing.
ARM_MANIFEST=$CLEANED
ARM_MD5=$EXPECT_CLEANED_MD5
ARM_LINES=$EXPECT_MANIFEST_LINES

stage_ws() {
  local WS=$1
  case "$WS" in /oscar/data/stellex/glvov/retread/ws.${TAG}-*) ;; *) echo "FATAL bad WS $WS"; exit 4;; esac
  if [ -d "$WS" ]; then
    mv "$WS" "$WS.trash.$$"
    ( chmod -R u+w "$WS.trash.$$" >/dev/null 2>&1; rm -rf "$WS.trash.$$" ) &
    echo "### moved pre-existing $WS aside"
  fi
  mkdir -p "$WS"
  echo "### stage 1/3: rsync small set from $SRC_WS"
  local S; S=$(date +%s)
  rsync -a --info=stats2 \
    --exclude '/.pixi/' --exclude '/third_party/' \
    --exclude '/assets/' --exclude '/groot-sonic-data/' --exclude '/logs/' \
    --exclude '/results/' --exclude '/scratchpad/' --exclude '/scratch_rescue/' \
    --exclude '/.pytest_cache/' --exclude '/pixi.lock' --exclude '/pixi.lock.*' \
    --exclude '/.cert-staged' \
    "$SRC_WS/" "$WS/"
  echo "### rsync rc=$? wall=$(( $(date +%s) - S ))s"
  mkdir -p "$WS/.pixi"
  cp "$SRC_WS/.pixi/config.toml" "$WS/.pixi/config.toml"
  echo "### stage 2/3: cp -al third_party (hardlink, read-only share)"
  S=$(date +%s)
  cp -al "$SRC_WS/third_party" "$WS/third_party"
  echo "### cp -al third_party rc=$? wall=$(( $(date +%s) - S ))s"
  # --- THIRD_PARTY EGG-INFO BREAK-LINKS (hazard found 2026-09-03) -----------
  # `cp -al third_party` shares INODES with $SRC_WS, and setuptools rewrites
  # third_party/*/*.egg-info/{SOURCES,top_level,dependency_links}.txt during a
  # build -- so a staged relock writes THROUGH the hardlink into imprint-data.
  # Measured on the live tree at 05:53 today: link count 42 on
  # third_party/ProtoMotions/protomotions.egg-info/SOURCES.txt with mtime
  # 05:27, i.e. written by a job of this batch. `stage_break_links()` prunes
  # third_party by design, so it does not cover this.
  # STAGE_METHOD=rsync does NOT close it either: BOTH staging paths in the
  # template `cp -al "$SRC_WS/third_party"`. Give these files their own inodes.
  # Pattern list taken from harness/tools-20260902 @ 6ad024b, which derived it
  # empirically: of 157 files a finished workspace rewrote, exactly SEVEN still
  # shared an inode with imprint-data, all under *.egg-info/. The list is wider
  # than those seven because .egg-link / .dist-info / __pycache__ / .pth are the
  # same class of in-place setuptools/pip write. `-size -1048576c` and NOT
  # `-size -1M`: find rounds -size up to whole units, so `-1M` matches NOTHING.
  # (The other half of 6ad024b -- making the stage MIRROR a real copy -- does not
  # apply here: this harness takes the rsync path and never builds or reads the
  # mirror. It `cp -al`s third_party straight out of $SRC_WS, which is the same
  # exposure, and this list is what closes it.)
  STAGE_TP_WRITABLE=( -size -1048576c '(' -path '*.egg-info/*' -o -name '*.egg-link'
    -o -path '*.dist-info/*' -o -path '*/__pycache__/*' -o -name '*.pth' ')' )
  EGGBROKE=0
  while IFS= read -r f; do
    [ -f "$f" ] || continue
    cp -p "$f" "$f.breaklink.$$" && mv -f "$f.breaklink.$$" "$f" && EGGBROKE=$((EGGBROKE+1))
  done < <(find "$WS/third_party" -type f -links +1 "${STAGE_TP_WRITABLE[@]}" 2>/dev/null)
  echo "### third_party egg-info break-links: $EGGBROKE file(s) given their own inode"
  REMAIN=$(find "$WS/third_party" -type f -links +1 "${STAGE_TP_WRITABLE[@]}" 2>/dev/null | wc -l)
  [ "$REMAIN" = 0 ] || { echo "### FATAL: $REMAIN third_party egg-info files still share an inode with $SRC_WS"; exit 3; }
  echo "### stage 3/4: install the manifest under test as the root manifest"
  cp "$ARM_MANIFEST" "$WS/pixi.toml"

  echo "### stage 4/4: apply the PACK diffs to the STAGED packs (never imprint-data)"
  declare -A SRC_MD5_BEFORE=()
  for spec in "${PACK_DIFFS[@]}" ${PACK_DIFFS2[@]+"${PACK_DIFFS2[@]}"}; do
    pk=${spec%%:*}; SRC_MD5_BEFORE[$pk]=$(md5sum "$SRC_WS/pypi-packs/$pk/pixi.toml" | awk '{print $1}')
  done
  for spec in "${PACK_DIFFS[@]}" ${PACK_DIFFS2[@]+"${PACK_DIFFS2[@]}"}; do
    pk=${spec%%:*}; rest=${spec#*:}; dmd5=${rest%%:*}; rmd5=${rest##*:}
    sfx=$PACK_DIFF_SUFFIX
    for s2 in ${PACK_DIFFS2[@]+"${PACK_DIFFS2[@]}"}; do [ "$s2" = "$spec" ] && sfx=$PACK_DIFF_SUFFIX2; done
    df=$PACK_DIFF_DIR/$pk.pixi.toml.$sfx.diff; tgt=$WS/pypi-packs/$pk/pixi.toml
    [ -f "$df" ] && [ -f "$tgt" ] || { echo "### FATAL A3b: $df or $tgt missing"; exit 9; }
    got=$(md5sum "$df" | awk '{print $1}')
    [ "$got" = "$dmd5" ] || { echo "### FATAL A3b: diff md5 $got != $dmd5 for $pk"; exit 9; }
    ino_s=$(stat -c%i "$SRC_WS/pypi-packs/$pk/pixi.toml"); ino_t=$(stat -c%i "$tgt")
    [ "$ino_s" != "$ino_t" ] || { echo "### FATAL A3b: $tgt shares the SOURCE INODE -- patching would edit imprint-data"; exit 9; }
    ( cd "$WS/pypi-packs/$pk" && patch -p0 --force --no-backup-if-mismatch pixi.toml < "$df" ) || { echo "### FATAL A3b: patch failed for $pk"; exit 9; }
    got=$(md5sum "$tgt" | awk '{print $1}')
    [ "$got" = "$rmd5" ] || { echo "### FATAL A3b: patched $pk md5 $got != $rmd5"; exit 9; }
    echo "###   $pk patched with $sfx diff, result md5 $got"
  done
  for spec in "${PACK_DIFFS[@]}" ${PACK_DIFFS2[@]+"${PACK_DIFFS2[@]}"}; do
    pk=${spec%%:*}; now=$(md5sum "$SRC_WS/pypi-packs/$pk/pixi.toml" | awk '{print $1}')
    [ "$now" = "${SRC_MD5_BEFORE[$pk]}" ] || { echo "### FATAL A3b: SOURCE TREE MOVED for $pk"; exit 9; }
    echo "###   imprint-data/pypi-packs/$pk/pixi.toml UNCHANGED ($now)"
  done
  rm -f "$WS"/pixi.lock "$WS"/pixi.lock.* 2>/dev/null
  echo "### manifest md5 (staged vs source):"; md5sum "$WS/pixi.toml" "$ARM_MANIFEST"
  if [ "$(md5sum < "$WS/pixi.toml")" != "$(md5sum < "$ARM_MANIFEST")" ]; then
    echo "### FATAL: staged manifest is not $ARM_MANIFEST"; exit 3
  fi
  [ "$(md5sum < "$WS/pixi.toml" | awk '{print $1}')" = "$ARM_MD5" ] || { echo "### FATAL: staged manifest md5 != $ARM_MD5"; exit 3; }
  echo "### manifest lines: $(wc -l < "$WS/pixi.toml") (want $ARM_LINES)"
  echo "### jetson LIVE rows: $(grep -c '^jetson = ' "$WS/pixi.toml") (want $EXPECT_JETSON_ROWS)"
  echo "### pixi.lock present (want 0): $(ls "$WS"/pixi.lock* 2>/dev/null | wc -l) file(s)"
  echo "### staged files: $(find "$WS" | wc -l)  size: $(du -sh --exclude=third_party "$WS" | cut -f1) (+third_party hardlinked)"
  echo "### path deps present:"
  grep -oE 'path *= *"[^"]+"' "$WS/pixi.toml" | sed 's/.*"\(.*\)"/\1/' | sort -u | \
    while read -r d; do printf '  %-60s %s\n' "$d" "$([ -e "$WS/$d" ] && echo OK || echo MISSING)"; done
}

# lock_names <lockfile> <out_prefix> -- pypi + conda distribution NAME sets.
# Extractor copied verbatim from the A/B harness of the wire lane, so the two
# arms are parsed by identical code.
lock_names() {
  local L=$1 OUT=$2
  awk '/^packages:/{f=1;next} f&&/^- pypi: /{p=1;next} f&&/^- conda: /{p=0} f&&p&&/^  name: /{print $2; p=0}' "$L" \
    | sort -u > "$OUT.pypi.txt"
  awk '/^packages:/{f=1;next} f&&/^- conda: /{n=split($3,q,"/"); b=q[n]; sub(/\.(conda|tar\.bz2)$/,"",b); m=split(b,r,"-"); if(m<3) next; s=r[1]; for(i=2;i<=m-2;i++) s=s"-"r[i]; print s}' "$L" \
    | sort -u > "$OUT.conda.txt"
  printf '  pypi names=%s  conda names=%s\n' "$(wc -l < "$OUT.pypi.txt")" "$(wc -l < "$OUT.conda.txt")"
}

# seed_isolated_cache -- build $ISO_CACHE as a MINIMAL isolation root.
#
# WHAT CHANGED FROM THE EARLIER COPIES OF THIS HARNESS, AND THE ARGUMENT.
# Jobs 5638422 and 5655631 hardlink-cloned uv+pixi+rattler with `cp -al`.
# MEASURED in job 5655631: 232,759 directories, ~113 minutes of NFS wall (2,180 dirs/min), and
# it is inode-bound, not byte-bound, because cp -al copies no data. That cost is
# unnecessary. Only two of the five trees under the shared root can hold a value
# that DIFFERS between the arms:
#
#   ISOLATE  verdicts/        route-probe verdicts. route_probe_cache.rs: the
#                             FILE key is validity_key(channels, python, subdir,
#                             policy_fields = channel-priority /
#                             system-requirements / virtual-packages /
#                             workspace-deps / workspace-pypi-providers) and the
#                             ENTRY key is probe_digest(stage, universe, specs).
#                             RETREAD_AUTO_IMPORTS is in NEITHER, while the spec
#                             set is exactly what injection changes -- so ON
#                             writes Sat/Unsat DECISIONS about injected names
#                             into the file OFF reads, at OFF's address. Second
#                             hazard: RouteProbeCache::persist rewrites the whole
#                             file from an in-memory snapshot, so two arms drop
#                             each other's entries. 13 files, 2.7 M.
#   ISOLATE  built-outputs/   handler::built_output_store_key_for_outputs builds
#                             the key from SCHEMA + backend_build_identity() +
#                             conda_outputs_cache_key_for_target(...) +
#                             source_identity + workspace/source manifest
#                             digests. No input is the injection gate, yet the
#                             stored payload is the POST-injection
#                             CondaOutputsResult. So ON publishes an injected
#                             outputs.json under the exact key a later OFF run
#                             looks up, and OFF adopts it with no resolve and no
#                             probes -- silently, since the log says only
#                             "built_output_store hit". 14 entries, 118 K.
#                             (NOT inert under THIS binsnap: the merged
#                             integration binary this job runs carries the
#                             built-output store, and p6c put the auto-imports
#                             gate into its key with a SCHEMA bump, so a
#                             gate-ON entry can no longer land at a gate-OFF
#                             address. Isolated anyway: the next binsnap
#                             makes it live and a guard that only works by
#                             accident is not a guard.)
#
#   SHARE    uv/ pixi/ rattler/   every bucket is named by an artifact's own
#                             identity, so ON can only ADD files and an added
#                             file is byte-identical to what OFF would have
#                             fetched had it asked:
#                               uv/simple-v24/pypi/<name>.rkyv    project name
#                               uv/wheels-v6/{pypi,url}/<name>/   dist filename
#                               uv/archive-v0/<hash>/             content hash
#                               uv/interpreter-v4/<h>/            interpreter
#                               uv/sdists-v9, uv/builds-v0        per-sdist (empty)
#                               pixi/pkgs/<name>-<ver>-<build>    build string
#                                 already folds a content hash -- the store holds
#                                 isaaclab-2.3x-pack-0.54.2-py311_h2cb6c52e99_loose_5
#                                 and ..._h7761845cc2_loose_5 side by side, so an
#                                 injected pack build is ADDITIVE, never an
#                                 overwrite
#                               pixi/repodata/*.shards-cache-v1   channel shard
#                               pixi/backends-v0, conda-pypi-mapping, uv-cache
#                               rattler/retread-repodata/         named by
#                                 repodata::disk_cache_path = hash(channel_url|subdir)
#                             uv resolves in memory; there is no solved-manifest
#                             bucket anywhere in its layout.
#
# HOW THE ISOLATION IS EXPRESSED. $ISO_CACHE/{uv,pixi,rattler} are SYMLINKS to
# the shared trees and $ISO_CACHE/{verdicts,built-outputs} are real `cp -a`
# copies. retread_fast_env then exports UV_CACHE_DIR/PIXI_CACHE_DIR/
# RATTLER_CACHE_DIR through the symlinks (transparent) and
# RETREAD_BUILT_OUTPUT_STORE plus the verdict symlink into the real copies.
# Nothing else in the harness changes, the ON arm starts EXACTLY as warm as OFF
# rather than merely "at least as warm", and the epilogue rm -rf removes links,
# never the shared trees behind them.
seed_isolated_cache() {
  echo "### seeding MINIMAL isolation root $ISO_CACHE from $SHARED_CACHE  $(date -Is)"
  iso_cache_guard "seed"
  rm -rf "$ISO_CACHE"
  mkdir -p "$ISO_CACHE" || exit 4
  local S; S=$(date +%s)
  local d
  # SHARED, read-write, via symlink: content-addressed download caches.
  for d in uv pixi rattler; do
    mkdir -p "$SHARED_CACHE/$d" || exit 4
    ln -s "$SHARED_CACHE/$d" "$ISO_CACHE/$d" || { echo "FATAL: ln -s $d failed"; exit 4; }
    printf '  shared-by-symlink %-8s -> %s\n' "$d" "$(readlink -f "$ISO_CACHE/$d")"
  done
  # THE WHEEL STORE: hardlink-cloned, not symlinked. It is content-addressed, so
  # sharing it would be safe; it is cloned only so the ON arm's writes cannot
  # change what a later OFF baseline reads, and so this run can report the ON
  # arm's own entry count separately. It holds one relock's worth of wheels, not
  # 71 G, so the clone is cheap -- its wall is printed below.
  if [ -d "$SHARED_CACHE/wheels" ]; then
    local WS0; WS0=$(date +%s)
    cp -al "$SHARED_CACHE/wheels" "$ISO_CACHE/wheels" || { echo "FATAL: cp -al wheels failed"; exit 4; }
    printf '  hardlink-cloned   %-8s %s top-level entries, wall=%ss\n' wheels \
      "$(ls -A "$ISO_CACHE/wheels" | wc -l)" "$(( $(date +%s) - WS0 ))"
  else
    mkdir -p "$ISO_CACHE/wheels"; echo "  wheels absent in shared store; created empty"
  fi

  # ISOLATED, real copies: the two trees that store a DECISION, not a download.
  for d in verdicts built-outputs; do
    if [ -d "$SHARED_CACHE/$d" ]; then
      cp -a "$SHARED_CACHE/$d" "$ISO_CACHE/$d" || { echo "FATAL: cp -a $d failed"; exit 4; }
    else
      mkdir -p "$ISO_CACHE/$d"
    fi
    printf '  real-copied       %-8s %s files\n' "$d" "$(find "$ISO_CACHE/$d" -type f | wc -l)"
  done
  echo "  SHARED-INODE CHECK (the isolated trees must share NO inode with the shared store):"
  for d in verdicts built-outputs; do
    printf '    %-14s overlap=%s (want 0)\n' "$d" \
      "$(comm -12 <(find "$SHARED_CACHE/$d" -type f -printf '%i\n' 2>/dev/null | sort -u) \
                  <(find "$ISO_CACHE/$d"    -type f -printf '%i\n' 2>/dev/null | sort -u) | grep -c .)"
  done
  echo "  SHARE CHECK (the download trees MUST resolve to the shared store):"
  for d in uv pixi rattler; do
    printf '    %-14s %s (want %s)\n' "$d" "$(readlink -f "$ISO_CACHE/$d")" "$SHARED_CACHE/$d"
    [ "$(readlink -f "$ISO_CACHE/$d")" = "$SHARED_CACHE/$d" ] || { echo "FATAL: $d does not resolve to the shared store"; exit 4; }
  done
  echo "### seed wall=$(( $(date +%s) - S ))s  $(date -Is)   (job 5655631 spent ~113 min in this function)"
}

# run_arm <OFF|ON> <cache_root> <auto_imports_value_or_empty>
run_arm() {
  local ARM=$1 CACHEROOT=$2 AI=$3
  local C=/oscar/data/stellex/glvov/retread/cert${TAG}-${J}-${ARM}
  local G=$C/g
  local WS=/oscar/data/stellex/glvov/retread/ws.${TAG}-${J}-${ARM}
  case "$C" in /oscar/data/stellex/glvov/retread/cert${TAG}-*) ;; *) echo "FATAL bad C $C"; exit 4;; esac
  echo
  echo "################################################################"
  echo "### ARM $ARM  RETREAD_AUTO_IMPORTS=${AI:-<unset>}  cache=$CACHEROOT  start $(date -Is)"
  echo "################################################################"
  stage_ws "$WS"

  ##### ENV BLOCK -- job-scoped BUILD state, SHARED/ISOLATED download+solve caches #####
  local d
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
  export CONDA_OVERRIDE_CUDA=12
  export CONDA_OVERRIDE_GLIBC=2.35
  export UV_LOCK_TIMEOUT=3600
  export UV_LINK_MODE=copy
  export OMNI_KIT_ACCEPT_EULA=YES
  export PRIVACY_CONSENT=Y
  export PIXI_BUILD_RETREAD_LOG=pixi_build_retread=debug,warn
  unset RUST_LOG
  export RUST_BACKTRACE=1

  # PERSISTENT CACHES -- after the job-scoped block, after RETREAD_FAST_TMP_ROOT.
  export RETREAD_PERSIST_CACHE_ROOT=$CACHEROOT
  # shellcheck source=/dev/null
  . "$FAST_ENV"
  retread_fast_env "$WS" || { echo "FATAL: retread_fast_env refused"; exit 7; }

  # THE WHEEL STORE -- the import->distribution INDEX AUTHORITY, and the reason
  # every injection run so far reported ~0% indexed naming.
  # courier::wheel_store_root_with resolves RETREAD_WHEEL_STORE -> XDG_CACHE_HOME
  # -> HOME/.cache, then joins "retread"/"wheels". retread_fast_env.sh does NOT
  # set RETREAD_WHEEL_STORE, and run_arm sets XDG_CACHE_HOME and HOME under the
  # job-scoped cert root -- so the store has started EMPTY in every job ever run,
  # and auto_imports_dry rows could only be indexed=false. (Job 5655631 arm ON,
  # which ran further than any predecessor, still managed only 5 indexed=true of
  # 317 -- and all five appear late, as the store fills mid-run.)
  # Pointing it at $CACHEROOT/wheels makes the OFF arm FILL the store and the ON
  # arm READ it, which is the only way the index-authority question can be
  # measured at all.
  # WHY SHARING A BLOB STORE ACROSS ARMS IS SAFE, unlike verdicts/built-outputs:
  # entries are <sha256>/<filename>, written tmp+rename, and the courier doc
  # comment states "Blob stores stay SHARED; only envs/scratch are job-local".
  # The name IS the content, so injection can only ADD entries and an added entry
  # is byte-identical to what the other arm would have written. Nothing here
  # records a decision.
  # NOT LANDED IN tools/retread_fast_env.sh ON PURPOSE. Every harness sources
  # that file, including the fix/p6d confirmation relock, and p6d is root-causing
  # a regression in the built-wheel DIGEST path (a named-git wheel judged "not
  # built" on the merged binary). Until p6d names which digest the store keys on,
  # the export stays local to this harness. The ready diff for fast-env is in
  # the Lane C handoff draft, marked "land after p6d".
  # WHEEL-STORE CHANGE (2026-09-03, per the p6i wheel-store class that
  # killed job 5697525): UNSET RETREAD_WHEEL_STORE rather than
  # pointing it at $CACHEROOT/wheels. The shared RETREAD_WHEEL_STORE landing in
  # tools/retread_fast_env.sh is REVERTED (HANDOFF §0) because a fill lock can
  # be created before the wheel exists and a reader that finds the directory
  # and not the wheel ABORTS instead of missing. courier::wheel_store_root_with
  # falls back to XDG_CACHE_HOME (job-scoped to $C/xdg-cache above) joined with
  # "retread"/"wheels", so the store starts empty every job.
  unset RETREAD_WHEEL_STORE
  local WHEEL_STORE_DIAG=$XDG_CACHE_HOME/retread/wheels
  mkdir -p "$WHEEL_STORE_DIAG" || exit 4
  echo "### ARM $ARM RETREAD_WHEEL_STORE=<unset, job-scoped via XDG_CACHE_HOME> diag_path=$WHEEL_STORE_DIAG entries_before=$(ls -A "$WHEEL_STORE_DIAG" 2>/dev/null | wc -l)"

  # THE WHEEL-STORE CONDITION OF THE TARGET ARM. The store is job-scoped and
  # starts empty (see above); WHEEL_STORE_SEED=none leaves it empty, which is
  # the condition the target lock was produced under. A seed path hardlink-clones
  # a store instead, and then deletes every fill-lock sidecar, every dotfile and
  # every zero-length entry FROM THE SEEDED COPY -- a reader that finds a shard
  # directory holding a fill lock and no wheel ABORTS instead of missing. The
  # source store is only READ (cp -al), never written.
  if [ "${WHEEL_STORE_SEED:-none}" != "none" ]; then
    if [ ! -d "$WHEEL_STORE_SEED" ]; then
      echo "FATAL: WHEEL_STORE_SEED=$WHEEL_STORE_SEED does not exist"; exit 4
    fi
    WSS0=$(date +%s)
    cp -al "$WHEEL_STORE_SEED"/. "$WHEEL_STORE_DIAG"/ || { echo "FATAL: wheel-store seed cp -al failed"; exit 4; }
    find "$WHEEL_STORE_DIAG" -mindepth 2 -maxdepth 2 \( -name '.*' -o -name '*.retread-fill-v1.lock' -o -size 0 \) -delete
    echo "### ARM $ARM WHEEL-STORE PRE-SEEDED from $WHEEL_STORE_SEED shards=$(ls -A "$WHEEL_STORE_DIAG" | wc -l) wheels=$(find "$WHEEL_STORE_DIAG" -mindepth 2 -maxdepth 2 -name '*.whl' | wc -l) wall=$(( $(date +%s) - WSS0 ))s"
  else
    echo "### ARM $ARM WHEEL-STORE NOT seeded (WHEEL_STORE_SEED=none) shards=$(ls -A "$WHEEL_STORE_DIAG" 2>/dev/null | wc -l) wheels=$(find "$WHEEL_STORE_DIAG" -mindepth 2 -maxdepth 2 -name '*.whl' 2>/dev/null | wc -l)"
  fi

  # THE ONE VARIABLE UNDER TEST.
  if [ -n "$AI" ]; then export RETREAD_AUTO_IMPORTS="$AI"; else unset RETREAD_AUTO_IMPORTS; fi
  echo "### ARM $ARM gate: RETREAD_AUTO_IMPORTS=${RETREAD_AUTO_IMPORTS:-<unset>}"

  # THE STRICT ZERO-GATE. Declared as AUTO_IMPORTS_STRICT in the SUBSTITUTE
  # block, where the reason it is OFF and the reason it is the ENV door rather
  # than the manifest key are both written out. Both facts are PRINTED here as
  # well, because sbatch snapshots this script at submission and stdout is the
  # only record of what the job actually decided.
  if [ -n "${AUTO_IMPORTS_STRICT:-}" ]; then
    export RETREAD_AUTO_IMPORTS_STRICT="$AUTO_IMPORTS_STRICT"
  else
    unset RETREAD_AUTO_IMPORTS_STRICT
  fi
  echo "### ARM $ARM gate: RETREAD_AUTO_IMPORTS_STRICT=${RETREAD_AUTO_IMPORTS_STRICT:-<unset, i.e. backend default = ON with injection>}"
  echo "### ARM $ARM strict is OFF ON PURPOSE: this phase MEASURES what the lock installs,"
  echo "###   so it must land the lock. The config key retread-auto-imports-strict=false is"
  echo "###   a [package.build.config] key read from each PACK'S OWN manifest -- the exact"
  echo "###   bytes the pinned pack diffs gate and the exact bytes the target lock was"
  echo "###   solved against -- so it cannot turn this gate off without changing the"
  echo "###   artifact under measurement. The env override is the backend's declared"
  echo "###   harness door for a measurement arm; the strict gate itself is still owed."

  local BLOG=$A/$P-$ARM.backend.log
  : > "$BLOG"
  local SHIM=$A/$P-$ARM.backend-shim.sh
  cat > "$SHIM" <<SHIMEOF
#!/usr/bin/env bash
exec 2> >(tee -a "$BLOG" >&2)
exec "$BACKEND" "\$@"
SHIMEOF
  chmod +x "$SHIM"
  export PIXI_BUILD_BACKEND_OVERRIDE="pixi-build-retread=$SHIM"
  echo "### ARM $ARM backend shim: $SHIM -> $BACKEND ; stderr tee -> $BLOG"
  echo "### ARM $ARM env:"; env | grep -E '^(HOME|PIXI_|RATTLER_|UV_|XDG_|TMPDIR|RETREAD_|CONDA_OVERRIDE)' | sort | sed 's/^/  /'
  echo "### ARM $ARM env count from manifest: $("$PIXI" workspace environment list --manifest-path "$WS/pixi.toml" 2>&1 | grep -cE '^- ') (want $EXPECT_ENVS)"
  echo "### ARM $ARM persistent cache TOP-LEVEL ENTRIES BEFORE the lock (du -sh over 97G of NFS cost ~10min x4 in job 5655631 and gated nothing):"
  timeout 60 du --inodes -s "$CACHEROOT"/* 2>/dev/null | sed 's/^/  /' || echo "  (du --inodes timed out at 60s; not a gate)"

  ##### THE LOCK #####
  cd "$WS" || exit 5
  local LLOG=$A/$P-$ARM.lock.log
  local LTIME=$A/$P-$ARM.lock.time.txt
  echo "### ARM $ARM lock start $(date -Is)"
  local S LRC LW; S=$(date +%s)
  /usr/bin/time -v -o "$LTIME" "$PIXI" lock -v > "$LLOG" 2>&1
  LRC=$?
  LW=$(( $(date +%s) - S ))
  echo "### ARM $ARM lock rc=$LRC wall=${LW}s end $(date -Is)"
  echo "$ARM $LRC $LW" >> "$A/$P.arm-walls.txt"
  echo "$LRC" > "$A/$P-$ARM.rc"; echo "$LW" > "$A/$P-$ARM.wall"
  grep -E 'Elapsed \(wall|Maximum resident set size|User time|System time|Percent of CPU' "$LTIME" | sed 's/^/  /'
  local LRSS; LRSS=$(awk -F': ' '/Maximum resident set size/{print $2}' "$LTIME")
  echo "### ARM $ARM lock peak RSS: ${LRSS:-UNKNOWN} kbytes"
  # THE DECISIVE LINES. Every previous arm needed a human to open a 40k-line
  # lock log to learn WHICH name killed it. `pixi lock` prints one `Error:`
  # block per failure and the LAST one carries the verdict, so print it here,
  # with the name it names, and count how many blocks there were.
  if [ "$LRC" != "0" ]; then
    local LSTRIP=$A/$P-$ARM.lock.stripped.log
    sed -e 's/\x1b\[[0-9;]*m//g' "$LLOG" > "$LSTRIP" 2>/dev/null
    echo "### ARM $ARM FAILED. \`Error:\` blocks in the lock log: $(grep -ac '^ *.*Error:' "$LSTRIP")"
    echo "### ARM $ARM FINAL Error: block (the verdict):"
    awk '/Error:/{n=NR} {l[NR]=$0} END{ if(n){ for(i=n;i<=NR && i<n+60;i++) print "  "l[i] } else print "  (no Error: line found)" }' "$LSTRIP"
    echo "### ARM $ARM every package name named by a \`would constrain\` / \`would require\` row:"
    grep -aoE '(would constrain|would require) [A-Za-z0-9._-]+' "$LSTRIP" | sort | uniq -c | sort -rn | head -20 | sed 's/^/  /'
  else
    echo "### ARM $ARM lock SUCCEEDED (rc=0)."
  fi
  echo "### ARM $ARM RETREAD_WHEEL_STORE entries_after (job-scoped, diag_path=$WHEEL_STORE_DIAG)=$(ls -A "$WHEEL_STORE_DIAG" 2>/dev/null | wc -l)  (the index authority's corpus)"
  echo "### ARM $ARM persistent cache TOP-LEVEL ENTRIES AFTER the lock:"
  timeout 60 du --inodes -s "$CACHEROOT"/* 2>/dev/null | sed 's/^/  /' || echo "  (du --inodes timed out at 60s; not a gate)"

  ##### AUTO-IMPORT EXTRACT (the whole point of the ON arm) #####
  local SLOG=$A/$P-$ARM.backend.stripped.log
  sed -e 's/\x1b\[[0-9;]*m//g' "$BLOG" 2>/dev/null > "$SLOG"
  echo "### ARM $ARM ANSI-stripped backend log: bytes=$(stat -c%s "$SLOG" 2>/dev/null)"
  local DRY=$A/$P-$ARM.auto-imports.txt
  grep -E 'auto_imports' "$SLOG" > "$DRY" 2>/dev/null
  echo "### ARM $ARM auto_imports rows: $(wc -l < "$DRY")"
  printf '  auto_imports_dry rows  %s\n' "$(grep -c 'auto_imports_dry:' "$DRY" 2>/dev/null)"
  printf '  auto_imports_dry summ  %s\n' "$(grep -cF 'auto_imports_dry: summary' "$DRY" 2>/dev/null)"
  printf '  auto_imports_lead rows %s\n' "$(grep -cF 'auto_imports_lead:' "$DRY" 2>/dev/null)"
  printf '  INJECTABLE rows        %s\n' "$(grep -cF 'detected requirement (INJECTABLE)' "$DRY" 2>/dev/null)"
  printf '  SKIPPED rows           %s\n' "$(grep -cF 'detected requirement SKIPPED' "$DRY" 2>/dev/null)"
  printf '  bundles injecting      %s\n' "$(grep -cF 'auto_imports: injecting detected requirements' "$DRY" 2>/dev/null)"
  printf '  injected=true rows     %s\n' "$(grep -cF 'injected=true' "$DRY" 2>/dev/null)"
  echo "### ARM $ARM SKIP REASONS (the guard doing its job):"
  grep -F 'detected requirement SKIPPED' "$DRY" 2>/dev/null \
    | grep -oE 'skip_reason="[^"]*"' | sort | uniq -c | sort -rn | sed 's/^/  /'
  echo "### ARM $ARM distinct SKIPPED module names, by reason -> $A/$P-$ARM.skipped-by-reason.txt"
  grep -F 'detected requirement SKIPPED' "$DRY" 2>/dev/null \
    | sed -nE 's/.*module=([^ ]+).*skip_reason="([^"]*)".*/\2\t\1/p' | sort -u \
    > "$A/$P-$ARM.skipped-by-reason.txt"
  awk -F'\t' '{m[$1]=m[$1]" "$2} END{for(k in m) printf "  [%s]%s\n", k, m[k]}' \
    "$A/$P-$ARM.skipped-by-reason.txt" | head -40
  echo "### ARM $ARM distinct LEAD modules: $(grep -F 'auto_imports_lead:' "$DRY" 2>/dev/null | grep -oE ' module=[^ ]+' | sed 's/ module=//' | sort -u | tee "$A/$P-$ARM.lead-names.txt" | wc -l)"
  sed 's/^/    ~ /' "$A/$P-$ARM.lead-names.txt" 2>/dev/null | head -100
  echo "### ARM $ARM distinct INJECTABLE root names: $(grep -F 'detected requirement (INJECTABLE)' "$DRY" 2>/dev/null | grep -oE ' root=[^ ]+' | sed 's/ root=//' | sort -u | tee "$A/$P-$ARM.injectable-names.txt" | wc -l)"
  sed 's/^/    /' "$A/$P-$ARM.injectable-names.txt" 2>/dev/null | head -100
  echo "### ARM $ARM indexed-vs-fallback naming (THE COLD-STORE CAVEAT, measured):"
  grep -F 'detected requirement (INJECTABLE)' "$DRY" 2>/dev/null | grep -oE 'indexed=[a-z]+' | sort | uniq -c | sed 's/^/  /'
  echo "### ARM $ARM roots actually injected, per bundle:"
  grep -F 'auto_imports: injecting detected requirements' "$DRY" 2>/dev/null \
    | sed -nE 's/.*bundle=([^ ]+).*injected=([0-9]+).*declared_roots=([0-9]+).*/  bundle=\1 injected=\2 declared_roots=\3/p' | head -60
  # THE SUPPRESSED-ROOT TABLE. One row per environment that shipped WITHOUT a
  # detected import, plus the request-wide counter the zero-gate reads.
  # `auto_imports_suppressed_roots=` is emitted UNCONDITIONALLY (INFO with a
  # zero when nothing was dropped), so a MISSING row is a dead run, not a clean
  # one -- which is why the verdict below distinguishes the two.
  echo "### ARM $ARM SUPPRESSED-ROOT ROWS (one per env):"
  grep -F 'auto_imports_suppressed env=' "$SLOG" 2>/dev/null \
    | sed -nE 's/.*(auto_imports_suppressed env=[^ ]+ roots=\[[^]]*\] reason=[^ ]+).*/  \1/p' \
    | sort -u | tee "$A/$P-$ARM.suppressed-envs.txt" | head -40
  printf '  suppressed-env rows %s\n' "$(wc -l < "$A/$P-$ARM.suppressed-envs.txt")"
  echo "### ARM $ARM SUPPRESSED-ROOT COUNTERS (the zero-gate reads these):"
  grep -oE 'auto_imports_suppressed_roots=[0-9]+ auto_imports_suppressed_all=[a-z]+ auto_imports_backoffs=[0-9]+' "$SLOG" 2>/dev/null \
    | sort | uniq -c | sed 's/^/  /' | tee "$A/$P-$ARM.suppressed-counters.txt"
  SUP_TOTAL=$(grep -oE 'auto_imports_suppressed_roots=[0-9]+' "$SLOG" 2>/dev/null | awk -F= '{s+=$2} END{print s+0}')
  SUP_ROWS=$(grep -cE 'auto_imports_suppressed_roots=[0-9]+' "$SLOG" 2>/dev/null)
  if [ "${SUP_ROWS:-0}" -eq 0 ]; then
    echo "  ### ARM $ARM ZERO-GATE: NO COUNTER ROW AT ALL -- absence is not a zero (the run may have died before the summary). VERDICT: INDETERMINATE"
  elif [ "${SUP_TOTAL:-0}" -eq 0 ]; then
    echo "  ### ARM $ARM ZERO-GATE: auto_imports_suppressed_roots total=0 across $SUP_ROWS rows. VERDICT: strict WOULD HAVE PASSED this lock"
  else
    echo "  ### ARM $ARM ZERO-GATE: auto_imports_suppressed_roots total=$SUP_TOTAL across $SUP_ROWS rows. VERDICT: strict WOULD HAVE REFUSED this lock, naming the envs above"
  fi
  echo "### ARM $ARM CARRIED SUPPRESSION into conda/build_v1:"
  grep -cF 'CARRYING the advertising pass' "$SLOG" 2>/dev/null | sed 's/^/  carry rows /'
  grep -F 'ABI INVARIANT IN conda/build_v1' "$SLOG" 2>/dev/null | cut -c1-300 | sed 's/^/    /' | head -5
  echo "### ARM $ARM BACK-OFFS:"
  printf '  RESOLVE BACK-OFF lines %s\n' "$(grep -cF 'RESOLVE BACK-OFF' "$SLOG" 2>/dev/null)"
  grep -F 'RESOLVE BACK-OFF' "$SLOG" 2>/dev/null | cut -c1-240 | sed 's/^/    /' | head -8
  printf '  ABI BACK-OFF lines     %s\n' "$(grep -cF 'ABI BACK-OFF' "$SLOG" 2>/dev/null)"
  grep -F 'ABI BACK-OFF' "$SLOG" 2>/dev/null \
    | sed -nE 's/.*bundle=([^ ]+).*dropped_roots=([^ ]*).*/  bundle=\1 dropped=\2/p' | sort -u | head -20
  printf '  BACK-OFF SUMMARY lines %s\n' "$(grep -cF 'BACK-OFF SUMMARY' "$SLOG" 2>/dev/null)"
  grep -F 'BACK-OFF SUMMARY' "$SLOG" 2>/dev/null | sed 's/^/    /' | head -5
  # The loose-constraint emission (commit b1b4aaa). The log constant is
  # INJECTED_CONSTRAINT_LOG = "retread-inject-constraint" (mod.rs), emitted with
  # fields bundle= package= emitted= source= decided= at the sole render point,
  # plus two fallback rows ("not representable" / "conflicting constraints").
  echo "### ARM $ARM CONSTRAINT EMISSIONS for injected members (the loose-constraint fix):"
  grep -F 'retread-inject-constraint' "$SLOG" 2>/dev/null > "$A/$P-$ARM.inject-constraints.txt"
  printf '  retread-inject-constraint rows %s\n' "$(wc -l < "$A/$P-$ARM.inject-constraints.txt")"
  echo "  by source= (manifest | loose):"
  grep -oE 'source="?[a-z]+' "$A/$P-$ARM.inject-constraints.txt" 2>/dev/null | sort | uniq -c | sed 's/^/    /'
  echo "  distinct package/emitted/source triples (want NO '==' emitted):"
  sed -nE 's/.*package="?([^" ]+)"?.*emitted="?([^"]*)"?.*source="?([a-z]+)"?.*/  \1\t\2\t\3/p' \
    "$A/$P-$ARM.inject-constraints.txt" 2>/dev/null | sort -u | head -60 | sed 's/^/  /'
  printf '  rows emitting an EXACT pin (==) -- the 5555157 failure shape, want 0: %s\n' \
    "$(grep -c 'emitted=[^ ]*==' "$A/$P-$ARM.inject-constraints.txt" 2>/dev/null)"
  echo "### ARM $ARM PILLOW-CLASS CHECK (the 5555157 kill: pillow==11.3.0 run-dep vs env pace needing 10.4.0):"
  grep -nE 'pillow' "$SLOG" 2>/dev/null | grep -E 'constraint|inject|run.dep|==' | cut -c1-240 | head -20 | sed 's/^/    /'
  grep -nE 'pillow' "$LLOG" 2>/dev/null | grep -E 'error|conflict|cannot|unsat|==' | cut -c1-240 | head -20 | sed 's/^/    /'
  echo "### ARM $ARM route refusals (conda side unavailable):"
  grep -oE 'route refused for [^ ]+ — [^"]{0,90}' "$SLOG" 2>/dev/null | sort -u | head -20 | sed 's/^/    /'

  ##### LOCK EVIDENCE #####
  if [ -f "$WS/pixi.lock" ]; then
    cp "$WS/pixi.lock" "$A/pixi.lock.$P-$ARM"
    echo "### ARM $ARM pixi.lock saved: $(stat -c%s "$A/pixi.lock.$P-$ARM") bytes md5: $(md5sum < "$A/pixi.lock.$P-$ARM")"
    echo "### ARM $ARM envs in lock (want $EXPECT_ENVS): $(awk '/^environments:/{f=1;next} f&&/^[a-z]/{exit} f&&/^  [A-Za-z0-9][A-Za-z0-9._-]*:$/{c++} END{print c+0}' "$A/pixi.lock.$P-$ARM")"
    echo "### ARM $ARM lock NAME SETS:"
    lock_names "$A/pixi.lock.$P-$ARM" "$A/$P-$ARM.locknames"
    local LK=$A/pixi.lock.$P-$ARM
    grep -aoE '^\s+- pypi: \S+'  "$LK" | awk '{print $3}' | sort -u > "$A/$P-$ARM.pypi-urls.txt"
    grep -aoE '^\s+- conda: \S+' "$LK" | awk '{print $3}' | sort -u > "$A/$P-$ARM.conda-urls.txt"
    echo "### ARM $ARM url sets: pypi urls=$(wc -l < "$A/$P-$ARM.pypi-urls.txt") conda urls=$(wc -l < "$A/$P-$ARM.conda-urls.txt")"
  else
    echo "### ARM $ARM NO pixi.lock produced"
    : > "$A/$P-$ARM.locknames.pypi.txt"; : > "$A/$P-$ARM.locknames.conda.txt"
    : > "$A/$P-$ARM.pypi-urls.txt"; : > "$A/$P-$ARM.conda-urls.txt"
  fi

  echo "### ARM $ARM COUNTERS (all must be 0):"
  for pat in 'retread rpc error' 'courier inputs changed' '0 exact matches' \
             'run dependencies differ' 'panicked'; do
    printf '  %-28s lock.log=%s backend.log=%s\n' "$pat" \
      "$(grep -c "$pat" "$LLOG" 2>/dev/null)" "$(grep -c "$pat" "$SLOG" 2>/dev/null)"
  done
  echo "### ARM $ARM ROUTE PROBE CACHE (a WARM run executes a handful, a cold one ~315):"
  for pat in 'route probe cache: hit' 'route probe cache: opened'; do
    printf '  %-32s backend.log=%s\n' "$pat" "$(grep -c "$pat" "$SLOG" 2>/dev/null)"
  done
  echo "  probes EXECUTED: $(grep -oE 'bundle route probes finished[^\n]*probes=[0-9]+' "$SLOG" 2>/dev/null | grep -oE 'probes=[0-9]+' | awk -F= '{s+=$2} END{print s+0}')"
  grep -nE 'route probe cache' "$SLOG" 2>/dev/null | head -6 | sed 's/^/  /'
  echo "### ARM $ARM ERROR line histogram:"
  grep -oE 'ERROR .*' "$SLOG" 2>/dev/null | sed 's/[0-9]\{3,\}/N/g' | sort | uniq -c | sort -rn | head -15 | sed 's/^/  /'
  echo "### ARM $ARM THE EXACT FAILURE (lock.log error block), if any:"
  grep -nE '^\s*(×|help:|caused by|error)' "$LLOG" 2>/dev/null | cut -c1-300 | head -40 | sed 's/^/  /'
  echo "### ARM $ARM lock.log tail:"; tail -40 "$LLOG" 2>/dev/null | sed 's/^/  /'
  echo "### ARM $ARM DONE rc=$LRC wall=${LW}s peak_rss_kb=${LRSS:-unknown} $(date -Is)"
  return 0
}

: > "$A/$P.arm-walls.txt"

########## ONE ARM: ONCERT. The OFF and ON arms of job 5668401 are NOT re-run --
########## they answered their question (B1 RESULT (0,1,1)) and the finding this
########## job exists for is a p6g CODE change, read off the ONCERT arm alone.
########## The shared wheel store carries the index warmth the dropped OFF arm
########## used to deposit; it is now populated (225+ entries) and shared by
########## tools/retread_fast_env.sh, so the isolated cache still gets seeded ONCE --
########## job 5668401 called seed_isolated_cache twice and its own instrument
########## caught the second call cloning 2 of 225 entries (LANE-C-WARM-LOG 9.5).
##########
# THE POINT OF THIS JOB. Job 5655631 gave the first valid
# injection A/B: arm OFF locked clean (rc=0, 3340 s, 27 envs) and arm ON failed
# in 478 s on `isaaclab-2.3x-pack 0.54.2 would constrain pillow
# !=8.3.*,>=8.3.2,==11.3.0` against environment unitree-rl-lab-gpu's
# `pillow ==10.4.0`. That 10.4.0 comes from
# [feature.isaaclab-unitree.dependencies], and the certified 9-line packet
# DELETES it -- along with the [feature.pace.dependencies] copy that killed job
# 5555157. So this arm asks the question the OFF/ON pair cannot: does
# injection-ON lock clean once the hand-pins the packet removes are gone? A
# clean lock here turns Lane C from "blocked" into the instrument that found
# those pins. A failure here is just as informative and must not be explained
# away: it would mean the class survives the packet.
seed_isolated_cache
ARM_MANIFEST=$CERT_MANIFEST; ARM_MD5=$EXPECT_CERT_MD5; ARM_LINES=$EXPECT_CERT_LINES
run_arm ONCERT "$ISO_CACHE" "1"

########## THE p6g FINDING -- the row this job exists to read ##############
# p6g gave the constrains slot its first READER: log_final_bundle_outputs now
# prints what each pack advertises, and every projection/omission WARNs naming
# the pack, the dropped bound and the workspace counterparties. Before p6g the
# bound that killed two arms was visible only in the consuming solver's error.
# ANSI HAZARD (LANE-C-WARM-LOG 12.6): the backend log is colour-escaped and the
# field names are SPLIT by the escapes, so `grep -F 'package=fsspec'` finds ZERO
# on the raw file and every row on the stripped one. run_arm already writes the
# stripped copy; every grep below reads THAT.
BL=$A/$P-ONCERT.backend.stripped.log
[ -f "$BL" ] || BL=$A/$P-ONCERT.backend.log
echo
echo "################################################################"
echo "### p6g CONSTRAINS FINDING  (backend log: ${BL:-<none>})"
echo "################################################################"
if [ -n "${BL:-}" ] && [ -f "$BL" ]; then
  echo "### every 'bundle constrains emitted' row:"
  grep -a 'bundle constrains emitted' "$BL" | cut -c1-400 | sed 's/^/  /'
  echo "### isaaclab-2.3x-pack's row (THE question: is fsspec absent from it?):"
  grep -a 'bundle constrains emitted' "$BL" | grep -a 'isaaclab-2.3x-pack' | cut -c1-600 | sed 's/^/  /'
  echo "### fsspec anywhere in an emitted-constrains row:"
  grep -ac 'bundle constrains emitted.*fsspec' "$BL" | sed 's/^/  count=/'
  echo "### p6g projection / omission WARNs:"
  grep -aE 'constrains .*(projected|omitted|dropped)' "$BL" | cut -c1-400 | head -40 | sed 's/^/  /'
  echo "### auto-routed fsspec rows (the two closure picks that collided):"
  grep -a 'auto-routed fsspec' "$BL" | cut -c1-300 | sed 's/^/  /'
  echo "### retread-constrains-discipline rows: total=$(grep -ac 'retread-constrains-discipline' "$BL")  naming fsspec=$(grep -a 'retread-constrains-discipline' "$BL" | grep -ac fsspec)  naming typing-extensions=$(grep -a 'retread-constrains-discipline' "$BL" | grep -ac 'typing.extensions')"
  echo "### the fsspec discipline rows (the p6j finding, verbatim):"
  grep -a 'retread-constrains-discipline' "$BL" | grep -a fsspec | cut -c1-400 | sort -u | head -20 | sed 's/^/  /'
  echo "### typing-extensions in every emitted-constrains row (the class 5716354 died on):"
  grep -a 'bundle constrains emitted' "$BL" | grep -a 'typing.extensions' | cut -c1-800 | sed 's/^/  /'
  echo "### the typing-extensions discipline rows:"
  grep -a 'retread-constrains-discipline' "$BL" | grep -a 'typing.extensions' | cut -c1-400 | sort -u | head -20 | sed 's/^/  /'
else
  echo "  NO BACKEND LOG -- the finding cannot be read from this run."
fi

########## COMPARISON ##########
echo
echo "################################################################"
echo "### OFF/ON COMPARISON  $(date -Is)"
echo "################################################################"
echo "### wall time per arm (arm rc wall_s):"; sed 's/^/  /' "$A/$P.arm-walls.txt"
# This harness runs ONE arm (ONCERT). The OFF/ON comparison below has no inputs
# and must say so rather than print empty diffs that read as findings.
if ! awk '$1=="OFF"{f=1} END{exit !f}' "$A/$P.arm-walls.txt"; then
  echo "  SINGLE-ARM RUN: no OFF arm in this job, so the OFF-vs-ON comparison"
  echo "  below is SKIPPED. The finding is the ONCERT arm alone."
  SKIP_AB=1
fi
if [ "${SKIP_AB:-0}" != "1" ]; then
awk '{w[$1]=$3; r[$1]=$2} END{
  if (w["OFF"]!="" && w["ON"]!="")
    printf "  OFF rc=%s %ss   ON rc=%s %ss   delta=%+ds (%.1f%%)\n", r["OFF"], w["OFF"], r["ON"], w["ON"], w["ON"]-w["OFF"], 100.0*(w["ON"]-w["OFF"])/w["OFF"];
}' "$A/$P.arm-walls.txt"

for KIND in pypi conda; do
  FA=$A/$P-OFF.locknames.$KIND.txt
  FB=$A/$P-ON.locknames.$KIND.txt
  echo "### LOCK $KIND NAME SETS:  OFF=$(wc -l < "$FA" 2>/dev/null)  ON=$(wc -l < "$FB" 2>/dev/null)"
  comm -13 "$FA" "$FB" > "$A/$P.diff.$KIND.added.txt" 2>/dev/null
  comm -23 "$FA" "$FB" > "$A/$P.diff.$KIND.removed.txt" 2>/dev/null
  echo "###   ADDED by injection (ON not OFF): $(wc -l < "$A/$P.diff.$KIND.added.txt")"
  sed 's/^/      + /' "$A/$P.diff.$KIND.added.txt" | head -120
  echo "###   REMOVED by injection (OFF not ON): $(wc -l < "$A/$P.diff.$KIND.removed.txt")"
  sed 's/^/      - /' "$A/$P.diff.$KIND.removed.txt" | head -120
done
echo "### URL COUNTS: OFF pypi=$(wc -l < "$A/$P-OFF.pypi-urls.txt") conda=$(wc -l < "$A/$P-OFF.conda-urls.txt")  ON pypi=$(wc -l < "$A/$P-ON.pypi-urls.txt") conda=$(wc -l < "$A/$P-ON.conda-urls.txt")"

echo "### PER-ENV PER-PACKAGE VERSION DELTA, OFF vs ON (the semantic diff):"
if [ -f "$A/pixi.lock.$P-OFF" ] && [ -f "$A/pixi.lock.$P-ON" ] && [ -x "$EVD" ]; then
  "$EVD" "$A/pixi.lock.$P-OFF" "$A/pixi.lock.$P-ON" $EVD_PACKAGES 2>&1 | sed 's/^/  /'
  echo "### and the same instrument OFF vs the imprint-data baseline lock:"
  "$EVD" "$BASE_LOCK" "$A/pixi.lock.$P-OFF" $EVD_PACKAGES 2>&1 | sed 's/^/  /'
  echo "### and ON vs the imprint-data baseline lock:"
  "$EVD" "$BASE_LOCK" "$A/pixi.lock.$P-ON" $EVD_PACKAGES 2>&1 | sed 's/^/  /'
else
  echo "  SKIPPED: one arm produced no lock (see rc rows above)"
fi

echo "### LEAD MODULE SETS across arms (should be identical -- detection is gate-independent):"
if diff -q "$A/$P-OFF.lead-names.txt" "$A/$P-ON.lead-names.txt" >/dev/null 2>&1; then
  echo "  IDENTICAL ($(wc -l < "$A/$P-OFF.lead-names.txt") modules)"
else
  diff "$A/$P-OFF.lead-names.txt" "$A/$P-ON.lead-names.txt" | sed 's/^/    /' | head -40
fi
echo "### INJECTABLE NAME SETS across arms:"
if diff -q "$A/$P-OFF.injectable-names.txt" "$A/$P-ON.injectable-names.txt" >/dev/null 2>&1; then
  echo "  IDENTICAL ($(wc -l < "$A/$P-OFF.injectable-names.txt") names) -- the gate changes only whether they are USED"
else
  diff "$A/$P-OFF.injectable-names.txt" "$A/$P-ON.injectable-names.txt" | sed 's/^/    /' | head -40
fi
echo "### SANITY: arm OFF must show 0 injecting lines:"
printf '  arm OFF injecting-lines=%s (want 0)  injected=true=%s (want 0)\n' \
  "$(grep -cF 'auto_imports: injecting detected requirements' "$A/$P-OFF.auto-imports.txt" 2>/dev/null)" \
  "$(grep -cF 'injected=true' "$A/$P-OFF.auto-imports.txt" 2>/dev/null)"
fi   # end SKIP_AB
echo "### lock md5 (this run produced the ONCERT arm only):"
md5sum "$A/pixi.lock.$P-ONCERT" 2>/dev/null | sed 's/^/  /'
echo
echo "################################################################"
echo "### THE THIRD-ARM QUESTION: does injection-ON lock clean on the CERTIFIED manifest?"
echo "################################################################"
sed 's/^/  /' "$A/$P.arm-walls.txt"
if [ -f "$A/pixi.lock.$P-ONCERT" ]; then
  echo "### ONCERT vs the gate-OFF baseline $OFF_BASELINE (want md5 $OFF_BASELINE_MD5):"
  if [ -f "$OFF_BASELINE" ]; then
    printf '  baseline md5 now: %s\n' "$(md5sum "$OFF_BASELINE" | awk '{print $1}')"
    "$EVD" "$OFF_BASELINE" "$A/pixi.lock.$P-ONCERT" $EVD_PACKAGES 2>&1 | sed 's/^/  /'
    echo "### NAME-SET counts, baseline vs this arm:"
    lock_names "$OFF_BASELINE"          "$A/$P-OFFBASE.locknames"
    lock_names "$A/pixi.lock.$P-ONCERT" "$A/$P-ONCERT.locknames"
    for KIND in pypi conda; do
      echo "###   $KIND: baseline=$(wc -l < "$A/$P-OFFBASE.locknames.$KIND.txt") oncert=$(wc -l < "$A/$P-ONCERT.locknames.$KIND.txt")"
      comm -13 "$A/$P-OFFBASE.locknames.$KIND.txt" "$A/$P-ONCERT.locknames.$KIND.txt" > "$A/$P.gate-added.$KIND.txt"
      comm -23 "$A/$P-OFFBASE.locknames.$KIND.txt" "$A/$P-ONCERT.locknames.$KIND.txt" > "$A/$P.gate-removed.$KIND.txt"
      echo "###     added by the gate:   $(wc -l < "$A/$P.gate-added.$KIND.txt")"
      sed 's/^/        + /' "$A/$P.gate-added.$KIND.txt" | head -80
      echo "###     removed by the gate: $(wc -l < "$A/$P.gate-removed.$KIND.txt")"
      sed 's/^/        - /' "$A/$P.gate-removed.$KIND.txt" | head -80
    done
  else
    echo "  BASELINE MISSING: $OFF_BASELINE -- no semantic diff is possible."
  fi
  echo "### ONCERT lock: pillow rows"
  grep -aoE "pillow-[0-9][^-]*" "$A/pixi.lock.$P-ONCERT" | sort | uniq -c | sed 's/^/  /'
else
  echo "  ONCERT produced NO lock -- read its rc and its error block above."
fi

########## CACHE-SAFETY EPILOGUE ##########
echo "### CACHE SAFETY: removing the isolated injection-ON cache root."
iso_cache_guard "epilogue"
S=$(date +%s); rm -rf "$ISO_CACHE"
echo "  rm -rf $ISO_CACHE rc=$? wall=$(( $(date +%s) - S ))s  exists_now=$([ -e "$ISO_CACHE" ] && echo YES || echo no)"
echo "  POST-DELETE PROOF the shared trees survived (rm -rf removes a symlink, never its target):"
for d in uv pixi rattler verdicts built-outputs wheels; do
  printf '    %-14s %s\n' "$d" "$([ -d "$SHARED_CACHE/$d" ] && echo "present, $(ls -A "$SHARED_CACHE/$d" | wc -l) entries" || echo "*** MISSING ***")"
done
echo "### shared warm cache AFTER the whole run (arm OFF wrote here; arm ON did not):"
for d in uv pixi rattler verdicts; do
  printf '  %-10s %s entries\n' "$d" "$(ls -A "$SHARED_CACHE/$d" 2>/dev/null | wc -l)"
done
echo "### job-scoped roots LEFT for a separate cleanup job (no self-cleanup on purpose):"
ls -d /oscar/data/stellex/glvov/retread/cert${TAG}-${J}-* /oscar/data/stellex/glvov/retread/ws.${TAG}-${J}-* 2>/dev/null | sed 's/^/  /'
echo "### inode quota AFTER:"; "$CQ" 2>/dev/null | grep -E 'data\+stellex' | head -2

ROFF=$(awk '$1=="OFF"{print $2}' "$A/$P.arm-walls.txt")
RON=$(awk '$1=="ON"{print $2}' "$A/$P.arm-walls.txt")
RONC=$(awk '$1=="ONCERT"{print $2}' "$A/$P.arm-walls.txt")
echo "### ${TAG} DONE off_rc=${ROFF:-?} on_rc=${RON:-?} certarm_rc=${RONC:-?} $(date -Is)"
# The job's own exit code is NOT the finding. arm ON is EXPECTED to fail on the
# canonical manifest (job 5655631 measured that), so a nonzero job rc here is
# consistent with the hypothesis rather than evidence against it. The finding is
# the pair (on_rc, certarm_rc): on_rc=1 with oncert_rc=0 means the packet's
# deletions are exactly what stood between injection-ON and a clean lock.
# SINGLE-ARM EXIT. The predecessor required ROFF=0 AND RONC=0, and no OFF arm
# runs here, so ROFF was always empty and the job could not report success even
# on a clean lock. The ONCERT arm is the whole job; its rc is the job rc.
if [ "${RONC:-1}" != "0" ]; then
  echo "### B4 RELOCK FAILED: arm rc=${RONC:-?} -- no lock to gate, no cert-phase handoff written."
  exit 1
fi

########## B4 REPRODUCTION GATE + CERT-PHASE HANDOFF ##########
# ADDED to the arm script this harness was derived from (that script is a
# one-shot relock instrument; it writes no handoff and gates nothing about the
# lock's identity). Two jobs, both of which must PRINT in stdout because sbatch
# snapshots the script at submission and nothing here can be checked afterwards:
#
#   1. THE HARD GATE. The lock this run produced must be byte-for-byte the
#      preserved artifact named in the SUBSTITUTE region. sha256sum is called
#      with DIRECT FILE ARGUMENTS, never through a pipe (project CLAUDE.md law
#      15: piped cmp/md5sum give false mismatches on this box). A mismatch is
#      FATAL: it prints the size delta, a sorted-line diff and the per-env
#      per-package version delta, then exits nonzero so the afterok cert phase
#      never releases and no different artifact gets certified by accident.
#   2. THE HANDOFF. On a pass, the stamp the cert phase sources. Same five keys
#      the two-phase template's relock writes: P1_JOB, WS, P1_CACHE_ROOT, LOCK,
#      EXPECT_LOCK_MD5. run_arm declares its workspace and cache root `local`,
#      so both are recomputed here from the SAME expressions run_arm uses; the
#      existence check below is what proves the recomputation is right.
FRESH_LOCK=$A/pixi.lock.$P-ONCERT
WS_ARM=/oscar/data/stellex/glvov/retread/ws.${TAG}-${J}-ONCERT
C_ARM=/oscar/data/stellex/glvov/retread/cert${TAG}-${J}-ONCERT

echo "### B4 GATE: the relock's lock vs the preserved injection-ON artifact"
if [ ! -f "$FRESH_LOCK" ]; then
  echo "### FATAL B4 GATE: the arm reported rc=0 but produced no lock at $FRESH_LOCK"; exit 3
fi
if [ ! -f "$PRESERVED_LOCK" ]; then
  echo "### FATAL B4 GATE: the preserved lock is missing: $PRESERVED_LOCK"; exit 3
fi
echo "### B4 GATE sha256 (direct file arguments, no pipe):"
sha256sum "$FRESH_LOCK" "$PRESERVED_LOCK" | sed 's/^/  /'
FRESH_SHA=$(sha256sum "$FRESH_LOCK" | awk '{print $1}')
PRES_SHA=$(sha256sum "$PRESERVED_LOCK" | awk '{print $1}')
FRESH_BYTES=$(stat -c%s "$FRESH_LOCK"); PRES_BYTES=$(stat -c%s "$PRESERVED_LOCK")
echo "### B4 GATE fresh     $FRESH_LOCK      $FRESH_SHA  $FRESH_BYTES bytes"
echo "### B4 GATE preserved $PRESERVED_LOCK  $PRES_SHA  $PRES_BYTES bytes"
echo "### B4 GATE pinned    $EXPECT_LOCK_SHA256  $EXPECT_LOCK_BYTES bytes"
if [ "$PRES_SHA" != "$EXPECT_LOCK_SHA256" ] || [ "$PRES_BYTES" != "$EXPECT_LOCK_BYTES" ]; then
  echo "### FATAL B4 GATE: the PRESERVED file on disk is not the pinned artifact."
  echo "###   The preserved copy moved or was overwritten -- stop and re-establish it before certifying anything."
  exit 4
fi
echo "### B4 GATE: the preserved file matches its pin, so the pin is live, not decorative."
if [ "$FRESH_SHA" != "$PRES_SHA" ]; then
  echo "### B4 GATE FAILED -- this relock did NOT reproduce the preserved lock."
  echo "###   byte delta: $(( FRESH_BYTES - PRES_BYTES ))"
  DRIFT=$A/$P.lock_drift_sorted.diff
  diff <(sort "$FRESH_LOCK") <(sort "$PRESERVED_LOCK") > "$DRIFT" 2>&1
  echo "###   sorted-line diff -> $DRIFT ($(grep -c '^[<>]' "$DRIFT" 2>/dev/null) changed lines; 0 would mean pure reordering)"
  echo "###   name/url set sizes (resolution, not byte order):"
  for f in "$FRESH_LOCK" "$PRESERVED_LOCK"; do
    printf '###     %-72s pypi-urls=%s conda-urls=%s\n' "$f" \
      "$(grep -aoE '^\s+- pypi: \S+'  "$f" | awk '{print $3}' | sort -u | wc -l)" \
      "$(grep -aoE '^\s+- conda: \S+' "$f" | awk '{print $3}' | sort -u | wc -l)"
  done
  EVDOUT=$A/$P.env_version_delta.target_vs_fresh.txt
  echo "###   env_version_delta target -> fresh (the delta to report, per environment):"
  if [ -f "$EVD" ]; then
    python3 "$EVD" "$PRESERVED_LOCK" "$FRESH_LOCK" $EVD_PACKAGES > "$EVDOUT" 2>&1
    sed 's/^/     /' "$EVDOUT"
  else
    echo "     (instrument missing: $EVD)"; : > "$EVDOUT"
  fi
  echo "###   auto-import rows this run emitted (store state at injection time):"
  grep -aE 'auto_imports(_dry|_lead)?:' "$A/$P-ONCERT.backend.stripped.log" 2>/dev/null \
    | grep -aE 'injecting detected requirements|BACK-OFF|indexing the wheel store' | cut -c1-220 | head -20 | sed 's/^/     /'

  # ---- THE ACCEPTANCE RULE, declared in the SUBSTITUTE region and APPLIED here.
  # Two ways a non-identical lock is still the same resolution, and nothing else
  # is accepted:
  #   (1) every changed line is a `purls:` `?source=` provenance annotation --
  #       which name-mapping artefact the process had cached, not what it chose;
  #   (2) env_version_delta reports `total moved rows across all envs: 0` --
  #       every selected version in every environment is identical.
  # Both counts are computed here and PRINTED, so the rule that decided this run
  # is in stdout and not only in a doc.
  CHANGED=$(grep -c '^[<>]' "$DRIFT" 2>/dev/null || true)
  NONANNOT=$(grep '^[<>]' "$DRIFT" 2>/dev/null | grep -vcE '^[<>][[:space:]]*- pkg:[^[:space:]]*\?source=' || true)
  MOVED=$(grep -F 'total moved rows across all envs:' "$EVDOUT" 2>/dev/null | grep -oE '[0-9]+$' | tail -1)
  echo "### B4 GATE RULE INPUTS: changed_lines=${CHANGED:-?} non_annotation_changed_lines=${NONANNOT:-?} env_version_delta_moved_rows=${MOVED:-<none>}"
  ACCEPT=
  if [ "${CHANGED:-0}" -gt 0 ] && [ "${NONANNOT:-1}" -eq 0 ]; then
    ACCEPT="annotation-only (every changed line is a ?source= purl annotation)"
  elif [ -n "${MOVED:-}" ] && [ "$MOVED" -eq 0 ]; then
    ACCEPT="zero-moved-rows (env_version_delta: 0 moved rows across all envs)"
  fi
  if [ -z "$ACCEPT" ]; then
    echo "### FATAL B4 GATE: refusing to hand a workspace built from a DIFFERENT resolution to the cert phase."
    echo "###   Neither acceptance rule holds: the changed lines are not all ?source= annotations,"
    echo "###   and env_version_delta did not report 0 moved rows (it reported '${MOVED:-<no line>}')."
    echo "###   No handoff stamp is written, so the afterok cert phase will not release. Report the delta above."
    exit 5
  fi
  GATE_STATE="ACCEPTED-WITH-RECORDED-DELTA: $ACCEPT"
  echo "### B4 GATE ACCEPTED WITH A RECORDED DELTA -- rule applied: $ACCEPT"
  echo "###   fresh  $FRESH_SHA"
  echo "###   target $PRES_SHA"
  echo "###   THE CERT PROCEEDS ON THE LOCK THIS RUN PRODUCED, not the target: the handoff"
  echo "###   below names $FRESH_LOCK, so nothing is certified that this workspace did not build."
else
  GATE_STATE="PASSED-byte-identical"
  echo "### B4 GATE PASSED: the relock reproduced the target lock byte-for-byte ($FRESH_SHA)"
fi

for p in "$WS_ARM" "$C_ARM"; do
  [ -d "$p" ] || { echo "### FATAL B4 handoff: recomputed arm root does not exist: $p"; exit 6; }
done
{
  echo "# written by bcert_relock.sh (${TAG}) job $J $(date -Is)"
  echo "P1_JOB=$J"
  echo "WS=$WS_ARM"
  echo "P1_CACHE_ROOT=$C_ARM"
  echo "LOCK=$FRESH_LOCK"
  echo "EXPECT_LOCK_MD5=$(md5sum < "$FRESH_LOCK" | awk '{print $1}')"
} > "$A/relock_env.sh"
echo "### cert-phase handoff written:"; cat "$A/relock_env.sh" | sed 's/^/  /'
echo "### roots LEFT for the cert phase and its own afterany cleanup: $C_ARM $WS_ARM"
echo "### ${TAG} RELOCK DONE arm_rc=$RONC gate=$GATE_STATE $(date -Is)"
exit 0
