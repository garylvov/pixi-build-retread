# `tools/phase_template/` — the fast, correct starting point for a new relock/cert pair
**Versioned copy: retread repo branch `harness/tools-20260902` → `harness/`; sync direction task-dir ← repo (one-time repo ← task-dir catch-up 2026-09-03 02:22; from here the repo is the source and the task dir is synced FROM it before each campaign).**

Three files:

| file | what it is |
|---|---|
| `phaseN_relock.sh` | the RELOCK half. Stages a pristine workspace, gates the manifest, locks under **persistent** caches, writes `artifacts/relock_env.sh` for the cert. Removes nothing. |
| `phaseN_cert.sh` | the CERT half. Reads that stamp, installs + probes + verifies every x86 env, scores against the certified baseline, then **submits** `cleanup.sh` and exits. Removes nothing itself. |
| `cleanup_gated.sh` | the GATE in front of `cleanup.sh`, and what a lane actually submits. Refuses (exit 2, nothing deleted) unless the evidence is in the task root, the root basename carries the relock job as a `-<jid>` token, and no job id in the basename is still queued. |
| `cleanup.sh` | a 1-CPU/4G job whose whole purpose is `rm -rf` on job-scoped roots. Refuses anything that is not `/oscar/data/stellex/glvov/retread/{cert*,ws.*}` or `agrescap/cache/retread-injection-on-<tag>`; refuses the persistent `agrescap/cache/retread` by name. `DRY_RUN=1` unlinks nothing. |

### The three relock shapes, and which one you are actually running

| shape | what it means | lock wall | evidence |
|---|---|---|---|
| **cold** | fresh workspace, empty caches | **2865 s** | job 5598763 arm A |
| **fresh workspace, warm shared caches** | the normal derived-harness run: this template on a new batch | **2366 s** | job 5611846 (this template's own smoke) |
| **same-workspace re-lock, warm caches** | `.pixi` state kept, only `pixi.lock` deleted | **69 s** | job 5598763 arm C |

The 41× is real and it is arm C's shape. **A newly derived harness does not get
it**, because it stages a pristine workspace, and the persistent cache holds
*downloads and solved metadata*, not *built local packs*. The smoke measured the
gap precisely: 14 `conda_outputs` bundle builds (identical to the cold arms; arm
C ran 1), 315 route probes executed (identical to cold; arm C ran 5), backend
window 1362.6 s, frontend head+tail 1003.4 s. Net −17 % against cold, not −97 %.
The verdict cache was correctly shared and correctly *missed*: its entries are
keyed on `(validity_key, digest)`, and a rebuilt local pack changes the digest.
**That is the next speed ticket** — make the local pack build reproducible or
cacheable across workspaces, and the fresh-workspace shape collapses toward
arm C's.

Evidence pointer for everything below: **job `5598763`** (node1820, 16 cpu, 72G) —
artifacts in `p5t-ab/artifacts/p5tabc-5598763.*`, narrative in `LANE-SPEED-LOG.md`
(section "2026-09-02 05:22–05:35 EDT"), draft handoff text in
`p5t-ab/HANDOFF-SPEED-DRAFT.md`. That one job measured cold-defaults **2865 s**,
cold-with-persistent-caches **2633 s**, and warm-persistent-caches **69 s** on the
same node with the same manifest and the same binary.

---

## Derive a new pair in five steps

1. **Copy, don't edit in place.**
   ```
   mkdir -p <T>/<newbatch>/artifacts <T>/<newbatch>/logs
   cp tools/phase_template/phaseN_relock.sh <T>/<newbatch>/<newbatch>_relock.sh
   cp tools/phase_template/phaseN_cert.sh   <T>/<newbatch>/<newbatch>_cert.sh
   ```
   `<T>` is `/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11`. `cleanup.sh`
   stays where it is — both copies find it by path, and nothing in it is
   campaign-specific.

   > **The binsnap is now ONE constant, not two.** Set `SNAP` (and the cert's
   > `SNAPDIR`) and stop. `EXPECT_SHA_PIN` is EMPTY by default and the gate
   > derives the sha from `$SNAP` with `sha256sum` at run time, so a SNAP swap
   > cannot leave a stale sha behind. Set `EXPECT_SHA_PIN` only to assert a
   > specific binary, and then it MUST match or the run refuses.
   >
   > This is the fix for the defect that killed job **5671529** (exit 8 in 3 s,
   > `snapshot sha 1860e830… != 2dd790bf…`): `SNAP` and a hand-written
   > `EXPECT_SHA` were two coupled constants and a derivation moved only the
   > first. The leftover-token self-check structurally cannot see it — both
   > values live INSIDE the SUBSTITUTE region, which the check strips by
   > design, and the stale sha was not in `LEFTOVER_RE`. Same exit-8-in-3-s
   > signature as p5w run 1 and run 2.
   >
   > Guard: `tools/phase_template/expect_sha_gate_guard.sh` derives both
   > templates three ways (pin empty / correct / wrong) and reads what the gate
   > does. Restoring the old literal-`EXPECT_SHA` shape fails it — measured, 10
   > FAIL rows including *"the derived sha still produced a sha refusal"*.

2. **Edit ONLY between `### SUBSTITUTE: BEGIN` and `### SUBSTITUTE: END`.**
   Everything a new batch changes lives in that one block: `TAG`, `D`, the
   manifest and its md5/line-count/diff-count gates, the residual-pin patterns,
   the probes file and its forbidden module tokens, the env list, the binsnap
   (`SNAP`/`SNAPDIR` — its sha is DERIVED, see the note above), the instrument
   paths, and `LEFTOVER_RE`. Set `TAG` to the same
   value in both files, and point the cert's `P1D` at the relock's directory
   when the two phases live in different dirs.

   `FAST_ENV` defaults to `$(dirname "$0")/../retread_fast_env.sh` and falls
   back to `$T/tools/retread_fast_env.sh`, so it resolves from either the
   template directory or a sibling batch directory. Leave it alone unless you
   moved the snippet.

3. **`bash -n` both, then let the scripts check themselves.**
   ```
   bash -n <newbatch>_relock.sh && bash -n <newbatch>_cert.sh
   bash tools/phase_template/expect_sha_gate_guard.sh    # after any gate edit
   bash tools/phase_template/leftover_check_guard.sh     # after any check edit
   bash tools/phase_template/wedge_triage_guard.sh       # after any triage edit
   bash tools/phase_template/census_collation_guard.sh   # after any stage-census edit
   bash tools/phase_template/wheel_store_census_guard.sh # after any wheel-store edit
   bash tools/phase_template/cleanup_owner_guard.sh     # after any cleanup-ownership edit
   ```

   > **The stage census is C-sorted at both ends, and that is load-bearing.**
   > `stage_manifest` is written by the job that builds the mirror and re-walked
   > by a later, different job, and the two are compared with `diff`. glibc's
   > `en_US.UTF-8` collation ignores `_`, `-` and case where C compares bytes,
   > so two jobs with different inherited locales sort the same file set into
   > different orders and `stage_verify_mirror` reads the non-empty diff as a
   > write-through: it quarantines the shared mirror and the harness exits 12.
   > `ml1` 5752248 did exactly that — a FALSE FATAL whose two censuses differ by
   > 0 lines once C-sorted, which cost the next job 459 s / 62 GB of re-staging.
   > Every `sort`/`diff` on that path carries an explicit `LC_ALL=C` prefix so
   > the pin is greppable and cannot be lost in a refactor.
   > Guard: `census_collation_guard.sh`.

   > **The lock log says which wheel store it actually read.** The old
   > "job-scoped wheel store" block became a dead letter when p6i re-enabled the
   > shared export: it announced a job-scoped store and printed the shared path,
   > so every proof after 15:35 09-03 read the fill-lock-poisoned shared store
   > while its log claimed isolation. Now `WHEEL_STORE_SEED` actually calls
   > `retread_seed_wheel_store`, the no-seed branch names the shared store
   > plainly, and `wheel_store_census` prints
   > `### WHEEL STORE IN USE (BEFORE|AFTER LOCK): scope=… path=…` on both sides
   > of the lock, resolving the store the way `courier::wheel_store_root_with`
   > does rather than reading back the variable the harness set — so a seed that
   > silently failed to take effect prints `SHARED`.
   > Guard: `wheel_store_census_guard.sh`.

   > **Exactly ONE cleanup owner per root, and the log names it.** A cleanup
   > submitted at dispatch and the cert phase's own self-submitted cleanup are
   > two owners of the same job-scoped roots, and they run at the same instant
   > because they hang on the same dependency. Two concurrent `rm -rf` walks of
   > one tree unlink entries out from under each other, so each one's rmdir of a
   > parent finds children it cannot see. **Measured 2026-09-04, tag AFINAL2:**
   > the dispatch-time gated cleanup **5770508** and the cert's self-submitted
   > **5776646** both released on `afterany:5769426:5769500`, both started
   > 08:39:44 on node2343, and both walked
   > `/oscar/data/stellex/glvov/retread/certAFINAL2-5769426` (590,028 entries)
   > and `ws.AFINAL2-5769426` (668,715 entries). **Both returned `rc=1` with
   > pages of `Directory not empty`**; 5776646 also logged
   > `rm: fts_read failed: Stale file handle`, which only a second walker can
   > produce. Both logged `exists_after=YES` for both roots, both still printed
   > `CLEANUP DONE rc=0`, and **both roots were left on disk** after 2864 s and
   > 3941 s of wall.
   > The owner is now a recorded fact, not an environment guess.
   > `phaseN_relock.sh` resolves it from `CLEANUP_AT_DISPATCH` or from the
   > dispatch note `<artifacts>/cleanup_at_dispatch.jobid` and writes
   > `CLEANUP_JOB=<id>` into the phase-1 → phase-2 handoff stamp
   > (`relock_env.sh`, alongside `P1_JOB=`/`WS=`/`LOCK=`/`EXPECT_LOCK_MD5=`).
   > `phaseN_cert.sh`'s `cleanup_owner` reads it and returns `dispatch <id>` or
   > `self -`, and `cleanup_submit_or_defer` either submits **nothing** and
   > prints
   > `### CLEANUP OWNER: job <id> (submitted at dispatch) -- roots: …`, or
   > submits **exactly one** and prints
   > `### CLEANUP OWNER: job <id> (submitted by this cert job <J>; no cleanup was recorded at dispatch) -- roots: …`.
   > Neither branch is silent: a reader of the log can name the owning job
   > without inferring anything.
   > Guard: `cleanup_owner_guard.sh`.

   > **The check matches the LINE, never "FILENAME:LNO: line".** It used to
   > annotate first and pipe the annotated text to `grep`, so it matched its own
   > FILENAME: a harness derived into `p6b-c3b/` failed against itself on every
   > line, with the token nowhere in its body. Fixed 2026-09-03 by moving the
   > match inside `awk`. A scan must not be able to match itself.
   Each script runs a leftover-token self-check as its *first* action and
   `exit 9`s on a hit. It strips three marked regions — `EVIDENCE` (the header),
   `SUBSTITUTE` (the constants), `LEFTOVER-CHECK` (itself) — and greps
   everything else, comments included, for the names of previous batches. This
   replaces the by-hand grep in HANDOFF §2, which had caught a defect on nearly
   every port. It is a guard that can fail: injecting `# bfinal` on one code
   line of an otherwise clean copy makes the script exit 9 and print the
   offending line.

4. **Dry-run the cert without installing anything.**
   ```
   DRY_RUN=1 <newbatch>_cert.sh
   ```
   runs every gate, the whole env block, and `retread_fast_env`, then prints the
   env order, the verdict gate it would use, and the cleanup it would submit —
   and exits 0. Use it after any substitution; it catches a bad stamp path, a
   bad probes file, a missing baseline, or a broken cache-siting in seconds.

5. **Submit. The memory numbers in each header are measured, not inherited.**
   ```
   env -u SLURM_JOB_ID sbatch --partition=batch --qos=normal --cpus-per-task=16 \
       --mem=24G  --time=03:00:00 --job-name=<tag>-p1 \
       --output=<T>/<newbatch>/logs/slurm-%j.out ./<newbatch>_relock.sh
   env -u SLURM_JOB_ID sbatch --partition=batch --qos=normal --cpus-per-task=16 \
       --mem=32G  --time=04:00:00 --job-name=<tag>-p2 --dependency=afterok:<p1 job> \
       --output=<T>/<newbatch>/logs/slurm-%j.out ./<newbatch>_cert.sh
   env -u SLURM_JOB_ID sbatch --partition=batch --qos=normal --cpus-per-task=1 \
       --mem=4G --time=16:00:00 --job-name=<tag>-cleanup \
       --dependency=afterany:<p1 job>:<p2 job> \
       --export=ALL,D=<T>/<newbatch>,TAG=<tag>,RJ=<p1 job> \
       --output=<T>/<newbatch>/logs/slurm-cleanup-%j.out \
       --wrap 'bash <T>/tools/phase_template/cleanup_gated.sh <cert root> <cache root> <ws root>'
   ```
   **Submit all THREE, and the third one now, not later.** The cleanup is
   `afterany` on *both* phases, so it also reclaims the roots of a relock that
   failed its own lock — the case that stranded C18A/C18B. Then **record that
   cleanup's job id** so the cert phase defers to it instead of becoming a
   second owner: either `echo <cleanup job id> > <p1 artifacts>/cleanup_at_dispatch.jobid`
   before the relock reaches its handoff, or export
   `CLEANUP_AT_DISPATCH=<cleanup job id>` into the relock and the cert. The
   legacy `CLEANUP_AT_DISPATCH=1` still defers but logs `unrecorded-id`, which
   is strictly worse to read. Record NOTHING and the cert submits and owns one
   itself, which is correct — what is never correct is both. See "Hazard 2".
   **24G relock / 32G cert, and the cert asks for 4 hours, not 8** — both changed
   2026-09-02 with the parallel env loop (next section). Sizing, all of it
   `/usr/bin/time -v` `Maximum resident set size`, never `sacct`: worst relock
   peak **8,854,172 K = 8.85 GB** (job 5597671), worst single cert *env*
   **1,475,216 K = 1.48 GB** (env `gpu`, job 5597694), and at `CERT_PARALLEL=6`
   the six largest envs of that ledger sum to **7,763,312 K = 7.8 GB**. So 24G is
   2.7× the relock peak and 32G is >4× the worst parallel cert, with the rest
   left as page cache for a phase that reads 108 GB and writes 318 GB. `sacct`
   MaxRSS is unusable here — every job reports ~100 % of its cgroup cap
   (160G → 167,772,480 K, 100G → 104,858,688 K) regardless of what it did, which
   is reclaimable page cache, not demand. The binding constraint is the per-user
   QOS cap (`normal` = cpu 64, mem 492G for the *whole* user): a 160G request is
   why job 5597889 pended behind `QOSMaxMemoryPerUser` with a node sitting idle,
   and 32G takes the pair from 3 concurrent jobs to ~15.

---

## Hazard 1 — a warm lock is not byte-identical to a cold one

The resolution is identical; the bytes are not. Job 5598763 compared its warm
arm C against its cold arm B — same node, same job, same manifest, same binary:

* name sets identical: pypi names 174, conda names 1707, pypi urls 213, conda
  urls 2584, zero one-sided members;
* `b3-phase1/env_version_delta.py` reports **0 moved version rows across all 27
  envs**;
* but `diff` is 10 lines, from exactly two causes:
  1. the **local path-source pack build hash** —
     `protomotions-deps-pack[5e31933f]` → `[784d2b1c]`, appearing in both the
     env row and the package row. A rebuilt local pack, not a resolution change.
  2. **one added `run_exports: {}` line** on the `future-1.0.0` conda record.
     Warm rattler carries a `run_exports` record that cold rattler does not
     (2554 such lines cold, 2555 warm).

So: **compare locks by name-set and by `env_version_delta.py`, never by md5.**
If a cert ever keys on that pack hash, it will read a warm relock as a change
when nothing resolved differently. (A third, older byte delta lives in the same
family and is unrelated to warmth: `gym`'s `requires_dist` extras come out in a
nondeterministic order — `diff` 30 lines, `diff <(sort A) <(sort B)` 0 lines.)

Caveat on the 41×: arm C was a realistic **re-lock** — same workspace, `.pixi`
state kept, manifest unchanged, `pixi.lock` deleted. It ran 1 `conda_outputs`
against arm A/B's 14. A fresh workspace with a warm shared cache is a different,
larger number: **2366 s** (job 5611846), and see the table at the top.

That same smoke also settles which side of the two byte deltas a fresh workspace
lands on. Job 5611846 vs cold arm A: raw `diff` 30 lines, **sorted diff 0** —
only the known `gym` ordering. Job 5611846 vs warm arm C: 10 raw / 8 sorted, the
pack hash and the `run_exports` line, with the smoke carrying arm A's
`[5e31933f]`. So the `[784d2b1c]` hash was an artifact of arm C rebuilding the
pack *in place*, not of cache warmth. Name sets and `env_version_delta.py`
(0 moved rows over all 27 envs) are identical across all three.

## Hazard 2 — cleanup never goes on the `afterok` path

An `rm -rf` of a job-scoped cache root on this filesystem takes tens of minutes:
**5152 s** in job 5596128's in-job epilogue (against a 3679 s lock, while holding
16 CPUs and 160G of a 492G per-user QOS and blocking its cert successor), and
**4795 s + 2552 s** for job 5598763's two roots. It is slow, not wedged — decide
by falling entry counts, never by an exit code.

Hence the rule (HANDOFF §2): extract artifacts, exit, clean up from a separate
1-CPU job. In this template the relock removes nothing at all and the cert
removes nothing at all; one cleanup job is hung behind **both** phases:

    --dependency=afterany:<relock job>:<cert job>

**`afterany` on BOTH, and never `afterok` on the cert alone.** Two reasons, and
the second one cost us two 450k-entry roots:

* `afterany` on the cert, so a RED cert still returns its disk.
* **Both phases named**, because a relock that fails its own lock writes no
  handoff, Slurm cancels the `afterok` cert, and a cleanup that names only the
  cert then has a dependency that never releases. `certC18A-5759225`,
  `ws.C18A-5759225` and the C18B pair are stranded exactly that way — job
  5759225's log carries "the afterok dependency will not release" and
  "self-cleanup NOT run here by design" on the same page, and no cleanup job for
  5759225 exists in `sacct` at all. Naming both jobs fixes it: a cancelled cert
  is terminal, so `afterany` releases and the roots come back.

Who submits it. Preferred, and what p6mbc and b4u do (job 5769783 shows
`afterany:5769781,afterany:5769782`): the **launcher** submits one *gated*
cleanup at dispatch, right after it has both job ids, and **records that
cleanup's job id** — as `CLEANUP_AT_DISPATCH=<id>` or in
`<p1 artifacts>/cleanup_at_dispatch.jobid` — so the relock stamps `CLEANUP_JOB=`
and the cert defers, printing the owning job id and submitting nothing.
Fallback: the cert submits it, and includes `$P1_JOB` from the relock stamp in
the dependency. **Never both** — jobs 5770508 and 5776646 both owned the AFINAL2
roots, raced, and left them on disk. Either way the roots are printed before the cleanup job exists, so a
diagnostician who wants to keep them can hold them by cancelling it.

    env -u SLURM_JOB_ID sbatch --partition=batch --qos=normal \
        --cpus-per-task=1 --mem=4G --time=16:00:00 --job-name=<tag>-cleanup \
        --dependency=afterany:<p1 job>:<p2 job> \
        --export=ALL,D=<harness dir>,TAG=<tag>,RJ=<p1 job> \
        --output=<T>/<newbatch>/logs/slurm-cleanup-%j.out \
        --wrap 'bash <T>/tools/phase_template/cleanup_gated.sh <root> ...'

`cleanup_gated.sh` is the gate, `cleanup.sh` is the deletion. The gate checks
three things and exits 2 without unlinking a byte if any fails: the evidence is
in the task root (`<TAG>-<RJ>*.rc`/`*.wall`/`*.lock.log`, plain or `.gz`, and a
certified lock when the run was green), the root basename carries the relock job
as a `-<jid>` **token** (the ownership proof), and no job id named in the
basename is still in `squeue`.

> **The token, not the suffix.** Until 2026-09-04 the gate read the job id as the
> last dash-separated field, so a root had to *end* in its job id. Every root an
> oncert lane mints ends in `-ONCERT`, so the whole class was permanently
> un-reapable — `p6ua-cleanup` 5764454 and `p6ub-cleanup` 5764455 printed
> `REFUSE: root … does not end in a job id` for six roots and deleted nothing.
> The same gate asked for `O7P6UA-5764452.rc` while the lane had written
> `O7P6UA-5764452-ONCERT.rc`, and so called present evidence missing.

`cleanup.sh` refuses any path outside `/oscar/data/stellex/glvov/retread/` whose
basename is not `cert*`/`ws.*`, and under `agrescap/cache/` accepts only
`retread-injection-on-<tag>` — the per-arm isolated cache roots, which carry no
job id in their name by construction. The persistent
`agrescap/cache/retread` is refused **by name**, before any prefix test runs.
`DRY_RUN=1` prints what it would remove (counting to depth 8 only) and unlinks
nothing.

---

## The persistent cache

`retread_fast_env "$WS"` (from `tools/retread_fast_env.sh`, sourced by both
halves after the job-scoped env block) repoints `PIXI_CACHE_DIR`,
`RATTLER_CACHE_DIR` and `UV_CACHE_DIR` at
`/oscar/data/stellex/glvov/agrescap/cache/retread/{pixi,rattler,uv}`, sets
`UV_LINK_MODE=copy` (a default the CERT half may override AFTER this call —
see "`UV_LINK_MODE`" below), and places a symlink for the route-probe verdict cache at
the path `fasttmp::namespace()` computes —
`$RETREAD_FAST_TMP_ROOT/retread-$USER/<sha256(realpath ws)[:12]>/job-$SLURM_JOB_ID/caches/retread/retread-route-probe-verdicts`
— which is the only way to persist verdicts without turning fast-tmp off.
Build state (`RETREAD_BUILD_ROOT`, `RETREAD_ARTIFACT_ROOT`, `RETREAD_META_ROOT`,
`RETREAD_SCRATCH_ROOT`, `HOME`, `TMPDIR`) stays job-scoped. It deliberately does
**not** set `RETREAD_PARALLEL_PROBES`: arm B measured that flag at −2.9 % on the
route-probe span union, i.e. no win.

It is rebuildable, so deleting it is always safe and only ever slow. It is also
not free: measured 2026-09-02 06:16 EDT it held **385,580 inodes**
(pixi 254,269 · uv 131,275 · rattler 22 · verdicts 14) on a filesystem that was
already `INODE_SOFT_EXCEEDED` at 103.8M against a 100M soft limit with 20 days of
grace left. One further relock through it (job 5611846) took it to **496,934**
(pixi 256,940 · uv 239,958 — uv nearly doubled). That is ~0.5 % of the soft
limit, which is a good trade for the speed — but it grows per run, so it is a
standing cost the operator should re-measure, not assume.

    rm -rf /oscar/data/stellex/glvov/agrescap/cache/retread   # always safe, only ever slow

---

## Staging: the workspace is hardlinked out of a persistent mirror

Before p12 every relock began by copying imprint-data's "small set" into a fresh
workspace: **9,175 regular files, 4,910 directories, 62,261,385,682 bytes**,
measured at **422 s** (job 5611846), **534 s** (5650823) and **572 s** (5655631),
followed by a `cp -al third_party` at **212–254 s**. Twelve to fourteen minutes
of pure harness overhead in front of a lock that finishes in 69 s warm.

### What that 62 GB is, and what the lock reads of it

| subtree | files | apparent bytes | opened by the lock |
|---|---|---|---|
| `pypi-packs/` | 1,370 | 52.31 GB | yes — the whole read set lives here |
| `.git/` | 6,207 | 9.93 GB | **never** |
| everything else | ~1,600 | 0.04 GB | **never** |

The read set was measured, not guessed. This filesystem mounts `relatime`, and
`rsync -a` leaves a staged file with `atime` = staging time and `mtime` = the
source's much older mtime, so the first read moves `atime` and
`find -printf '%A@ %T@'` after the lock is a read-set detector. Job 5650823's
workspace came back with **254 files / 2.31 GB read**, every one of them under
`pypi-packs/`, plus `pixi.toml` and `.pixi/config.toml` (created with `cp`, so
`atime == mtime` and `relatime` hides them). Zero reads in `.git`, `src`, `test`,
`docs`, `humble_ws`, `jazzy_ws`, `packages`, `patches`, `plans`, `scripts`,
`step_back`, `tools`, `wbc_push`.

### The two staging paths

Three plain variables sit just above the staging block in `phaseN_relock.sh`:

```
STAGE_METHOD=mirror     # mirror | rsync
STAGE_MIRROR_ROOT=/oscar/data/stellex/glvov/agrescap/cache/retread/stage-mirror
STAGE_PAR=16
```

* **`mirror` (the default).** A persistent read-only mirror lives at
  `$STAGE_MIRROR_ROOT/<key>/`, where `<key>` is
  `md5( md5($SRC_WS/pixi.toml) + " " + git -C $SRC_WS rev-parse HEAD )`. It is
  rsync'd out of `$SRC_WS` **once per key**; every later job pays only
  `cp -al`, which writes directory entries, not bytes, fanned out `STAGE_PAR`
  ways because the cost here is NFS RPC latency and not CPU.
  Guards: no mirror, or a `.stage-mirror-key` that does not carry the computed
  key → the old mirror is moved to `<mirror>.stale-<job>` and a new one is
  built; a failed build or a failed `cp -al` falls back to the rsync path and
  says so. Two jobs that both miss race on `mv -T`, which refuses to move into
  an existing directory, so the loser discards its copy and adopts the winner's.
* **`rsync`.** The pre-p12 path, unchanged and still selectable.

### Why hardlinking the packs is safe, and what is copied anyway

A hardlink is only safe for a file nobody writes **through the inode**. The
retread source (v4.10.90) was read for every writer that touches a pack:

| writer | mechanism | hardlink-safe? |
|---|---|---|
| `materialize_validated_wheel` (`src/source_build.rs`) | temp file → `std::fs::rename` | yes |
| `fetch_wheel`, `atomic_owned_copy` (`src/wheel.rs`) | `.part` / `.copy` sibling → rename | yes |
| the inject / relax / autodata writers | `wheel::create_atomic_tmp` + `commit_atomic_write` | yes |
| the courier lock `retread-*.target-*.lock.json` | `.json.tmp` → rename | yes |
| `is_fresh`, `fetch_wheel` eviction | `remove_file` (unlink only) | yes |
| `status::log` → `retread-progress-*.log` | `OpenOptions … .append(true)` | **no** |
| `write_probe_trace` → `retread-probe-trace-*.json` | `tokio::fs::write` (truncate) | **no** |
| the audit site → `retread-audit*.json` | `tokio::fs::write` (truncate) | **no** |
| `write_relaxed_wheel_cache_stamp` → `*.retread-cache` | `std::fs::write` (truncate) | **no** |

`stage_break_links()` gives every file in the last four rows its own inode
before the lock starts — **363 files, 25 MB** on the current source tree, which
is seconds. The multi-GB wheel payloads stay shared. Nothing outside
`pypi-packs/` needs breaking: retread writes `.retread/auto-overrides.json`
atomically, does not create the `.pixi/bld` / `.pixi/envs` symlinks inside a
Slurm job at all, and never writes to `third_party/`, `packages/`, `src/` or any
`.git` (its destructive git work happens in a `git clone --no-local` under
`retread_cache_root()/path-git-metadata/v1/`).

The pack build hash that lands in `pixi.lock` is not a directory-content hash:
`RetreadLock::compute_inputs_hash_for_target` (`src/lock.rs`) hashes a declared
input list (`retread-inputs-v7`, sorted entry specs, index URLs, relax policy,
`target.resolution_identity()`, `EMIT_EPOCH`, `courier::config_fingerprint`),
and retread declares `input_globs: Default::default()` so pixi hashes no pack
bytes either. Staging by hardlink therefore cannot move a build hash.

### The guard that makes this a reader/writer pair

`stage_verify_mirror()` runs after the lock, re-walks the mirror against the
`.stage-mirror-manifest.tsv` written when it was built, and on any difference
prints a FATAL-CLASS block and renames the mirror to `<mirror>.DIRTY-<job>` so
the next job rebuilds it. A write that escapes the break-list is caught instead
of silently poisoning every later batch.

`p12-staging-lever/test_stage_mirror.sh` runs the real functions on a fixture
tree, simulates all four in-place writers plus one atomic one, and asserts the
mirror survives — then runs the same thing **without** `stage_break_links` and
asserts the verifier fails and quarantines. A guard that cannot fail is not a
guard, so that second arm is the point of the test.

    bash p12-staging-lever/test_stage_mirror.sh     # no Slurm, no pixi, no lock

### What it measures out to

| arm | seconds | job |
|---|---|---|
| the old path: rsync 557 + `cp -al third_party` 202 | **759** | 5658374 A |
| cold mirror build, once per key (`.git` included) | 799 | 5658792 |
| warm hit, flat serial `cp -al` | 230 | 5658374 C |
| warm hit, fan-out over the 80 top-level entries | 211 | 5658374 D |
| **warm hit, fan-out over depth-3 entries + break-links — the default** | **75** | 5661756 |

**759 s → 75 s.** Fanning out over top-level entries bought almost nothing
because `third_party` is a single entry holding 25,178 files; splitting at
depth 3 (722 units, largest 2,926 files) is what does it.

Equivalence against a real rsync stage of the same source tree, job 5661756:
`diff -rq` is **0 lines** for `pypi-packs`, `.git`, and every other subtree, and
0 for the root-level files. The structural manifest differs only in the per-job
writable bits plus 322 directory mtimes — the directories `stage_break_links`
wrote into, which the lock bumps on the old path too.

**A note on the fan-out, because it shipped broken once.** The first version
enumerated its units with
`find . -mindepth 1 -maxdepth 3 \( -mindepth 3 -o ! -type d \)`. `-mindepth` is a
global option in GNU find, not a test, so it applied to the whole walk and every
depth-1 and depth-2 FILE was silently dropped — `AGENTS.md`, `.git/HEAD`,
`.git/index`, all 205 `test/*.py`. The tree still had 44,000 entries and every
`path =` dependency still resolved, which is exactly why only a `diff -rq`
against a real rsync stage caught it. The fix is two separate `find` calls, and
`test_stage_mirror.sh` now carries three shallow-file checks plus a
"no file is missing vs the mirror" `comm`, all four of which fail against the
broken version.

### Costs

The mirror is **69 G / 44,113 entries** and lives beside the pixi/uv/rattler
caches under `agrescap/cache/retread/`. It is rebuildable: deleting it is always
safe and costs the next job 799 s. A staged workspace is **34,072 files /
9,746 dirs**, of which only **365 files have link count 1** — the 363 broken
sidecars plus `pixi.toml` and `.pixi/config.toml`. Its private cost is therefore
about 25 MB and 9,750 inodes, not 62 GB and 44,000.

`.git` is still mirrored even though the lock never opens it. That is
deliberate — it keeps the staged tree structurally identical to what the rsync
path produced and keeps `### .git present: yes` meaning what it meant. Dropping
it is worth a measured 248 s off the mirror build and ~40 % of every `cp -al`,
and is the obvious next lever, behind its own variable.

---

## The cert env loop is `CERT_PARALLEL` wide — measured, and NOT shippable at 4

`phaseN_cert.sh` dispatches its 26 installs `CERT_PARALLEL` at a time
(default **4**; `CERT_PARALLEL=1` reproduces the pre-2026-09-02 serial loop
exactly), longest-first from `CERT_WALL_TABLE` — a previous cert's
`memory_ledger.<job>.tsv`, used only for ordering, never for scoring. An env the
table does not name sorts first. Set it in the SUBSTITUTE block:

    CERT_PARALLEL=${CERT_PARALLEL:-4}
    CERT_WALL_TABLE=$T/<last green cert>/artifacts/memory_ledger.<job>.tsv

Each env gets its own subshell, its own `TMPDIR`, and its own row files under
`artifacts/rows.<job>/<env>.{row,led,span}`. Nothing appends to a shared file
while installs are in flight; the rows are concatenated in DECLARATION order
after the last env finishes, so `cert_results.tsv` is byte-comparable with the
serial format whatever order the envs ran in. A subshell that dies leaves no row,
`cert_verdict.sh` reports `NOT_RUN`, and the job exits 1 — a short results file
never passes quietly. The `.span` files give the concurrency ACTUALLY achieved
(`artifacts/env_spans.<job>.tsv`, and the job prints peak and mean width).

`CERT_SERIAL_KEY` (optional) names a results file from a previous run of the SAME
lock; it is scored by a second `cert_verdict.sh` call whose exit is ANDed into the
job's.

### NEVER `wait` and NEVER `wait -n` after the dispatch loop (fixed 2026-09-03)

This script starts with `exec > >(tee -a "$A/cert_$TAG.$J.log") 2>&1`. Under
bash 5.1.8 that `tee` is a child of the script that never exits, and every
argument-less wait then blocks on it forever. Both forms were measured failing:

* the original bare `wait` waits for ALL children — job **`5658928`** wrote all
  26 env rows by 01:55:39 EDT and sat in it until its 4 h limit killed it at
  02:16: `TIMEOUT`, an empty `cert_results.tsv`, no verdict, no cleanup job
  submitted, and 38 minutes of a 16-CPU hold spent doing nothing;
* the first fix, `while [ $RUNNING -gt 0 ]; do wait -n; RUNNING=$((RUNNING-1)); done`,
  fails the same way when the env subshells have ALREADY exited — job
  **`5674557`** (one env, finished 03:59:05) was still sitting in it at 04:13
  with `tee` as its only live child.

The loop now records each subshell's pid and waits on it by name:

    PIDS=()
    ( run_env "$ENV" ) & PIDS+=($!)
    ...
    for P in ${PIDS[@]+"${PIDS[@]}"}; do wait "$P" 2>/dev/null; done

`wait <pid>` returns immediately for an exited child and 127 for one the
throttle already reaped, so it can neither hang nor lose a status. The throttle
inside the dispatch loop keeps `wait -n` deliberately: there it blocks until a
real env completion, which is what it is for. **This bites at every
`CERT_PARALLEL`, 1 included** — the serial setting still runs each env as
`( run_env ) &`.

When a run does hang this way its rows are all on disk: reassemble
`cert_results.tsv` by `cat`-ing `artifacts/rows.<job>/<env>.row` in
`CERT_ENVS` declaration order and score it with `cert_verdict.sh` by hand.

### Every env must have a probes row (A5a, 2026-09-03)

An env whose name has no row in `$PROBES` gets an empty `$MODS`, so the TierA and
TierB probes are both skipped. That was never a silent pass — `ARC` stays at its
`99` initialiser and the row scored `RED-tierA` — but the label was a lie: TierA
never ran, and nothing said so. Two changes, neither behind a flag:

1. a gate before any install refuses a run in which an env of `CERT_ENVS` has no
   probes row, printing `### probes row missing for <env>` per env and the count
   line `### probed envs: N of M`;
2. `run_env` scores such an env **`RED-probes-missing`**, an explicit verdict the
   gate reads as `OUTCOME_DIFF` (verdict is field 8, and field 8 is scored).

`REQUIRE_PROBE_ROW_PER_ENV=0` disables only (1), so a guard run can reach (2).
The guard is `p12b-cert-attrib/probes_guard.sh`: it runs the real harness three
times against a real workspace — doctored probes file expecting exit 2, doctored
file with the gate off expecting a `RED-probes-missing` row that
`cert_verdict.sh` calls `OUTCOME_DIFF`, and the real 26-row file expecting `cpu`
GREEN and the gate silent. It fails if the three edits are reverted.

### What the first full proof run measured

Job **`5658928`**, `CERT_PARALLEL=4`, 16 CPU / 32 G, node2320, against the
already-certified bfinal lock with `bfinal-phase2/artifacts/cert_results.5597694.tsv`
(serial, 16 CPU, same lock, same manifest, same binsnap) as the answer key. All
26 rows landed; they had to be reassembled by hand from `rows.5658928/` because
of the `wait` hang above.

1. **One scored field diverged, and it is NOT the fan-out.** `pm-isaaclab` came
   back `RED-verify` against the key's `AMBER-repaired`; every other row matched
   both the campaign baseline `5175534` and the serial key. Inside that one env
   prefix `isaaclab-2.3x-pack` pins `lxml==7.0.0b1`, `platformdirs==4.11.7`,
   `wandb==0.29.0` and the co-resident `protomotions-deps-pack` pins
   `7.0.0a3` / `4.11.3` / `0.28.2`. **The pins are read from the two packs' own
   `retread-<bundle>.target-<hash>.lock.json` sidecars** (`src/coresident_pins.rs`,
   `coresident_pins()` / `explain_divergent_pins()`), and both sidecars carry the
   SAME target hashes in the serial run and in the parallel one. The serial run
   had the identical disagreement and still scored AMBER: its repair converged,
   this one's was REFUSED. Attribution runs: `p12b-cert-attrib/`.
2. **4 wide is 1.56× fast and 2.26× expensive.** Sum of the 26 install walls
   **20 477 s serial → 46 233 s parallel = 2.26×**; span 13 111 s against the
   serial loop's 20 477 s, peak concurrency 4, mean 3.53. The LPT table predicted
   5 123 s at N=4; the measured span is **2.56× that**. Quote the honest number:
   **3 h 38 m of env loop instead of 5 h 41 m**, bought with 2.26× the work.
   Worst inflations are the SMALL envs — `tensorboard-tools` 15 s → 1 028 s
   (68×), `test` 24 s → 557 s, `cpu` 31 s → 415 s — which is what a bandwidth
   ceiling looks like, not a CPU one.
3. **Peak per-env RSS under 4-wide: 1 249 420 K = 1.25 GB** (`ros2-humble-gpu`),
   under the 1 475 216 K serial worst case. The 32 G request stands; 160 G never
   was justified.
4. **Persistent caches did NOT cut the cert's I/O.** `5658928` (persistent
   caches) read 484 072 M and wrote 309 404 M; `5597694` (job-scoped caches) read
   483 548 M and wrote 314 679 M. −1.7 % of writes. The 318 GB is the env
   prefixes being written, not package cache repopulation — so audit item S13's
   premise is wrong and the lever is `UV_LINK_MODE`, not cache siting.



---

## `UV_LINK_MODE` — what `copy` is actually insurance against, and what it costs

`retread_fast_env` exports `UV_LINK_MODE=copy` and, until 2026-09-03, that was
the last word on it for every phase. The rule came from job **`5547450`**, and
the rule was a generalisation of something narrower than anyone had written down.

**What `5547450` actually hit.** From its own backend log, the failure is a
source BUILD, not an install:

    uv-building wheel from sdist .../gym-0.26.2.tar.gz for gym: uv ["build", ...] failed
      Caused by: Failed to install requirements from `build-system.requires`
      Caused by: Failed to install build dependencies
      Caused by: Failed to install: setuptools-80.10.2-py3-none-any.whl
      Caused by: failed to hardlink file from
        <job uv cache>/builds-v0/.tmp4QiHoL/lib/python3.11/site-packages/pkg_resources/tests/test_working_set.py
        to <job uv cache>/archive-v0/2Xwd2LTx6I0HEYJm/pkg_resources/tests/test_working_set.py
        : No such file or directory (os error 2)

Both endpoints are uv cache buckets, and the one that vanished is
`builds-v0/.tmp4QiHoL` — a per-build ephemeral build environment, created and
reclaimed by one of the six concurrent builds
(`RETREAD_MAX_CONCURRENT_BUILDS=6`). **Builds hardlinked out of a reclaimable
temp tree.** That is the whole hazard, and it belongs to the phase that builds.

**Installs do not link from there.** uv documents the flag (uv 0.12.5,
`uv help pip install`) as *"The method to use when installing packages from the
global cache"*, and warns that clearing the cache breaks installed packages —
i.e. the source is the cache. A cert installs a FROZEN lock: it resolves
nothing and builds nothing, so its source is `archive-v0`, content-addressed.
Nothing reclaims it: **`uv cache clean` and `uv cache prune` have zero call
sites** in `tools/` or in any harness in the task tree, and the persistent
cache's own buckets show the shape plainly — `archive-v0` 4 796 entries,
`builds-v0` and `sdists-v9` **empty** after every relock and cert run against it.

**So the split is by PHASE, and it is now expressed that way.**

* `phaseN_relock.sh` keeps `UV_LINK_MODE=copy`, hard, with the reason in the
  line above it. That phase builds, concurrently; it is the phase `5547450` was.
* `phaseN_cert.sh` gets **`CERT_UV_LINK_MODE`** (`hardlink` default since the
  N=1 corner landed, below | `copy`),
  exported AFTER `retread_fast_env` and printed twice — once where it is decided
  and once at the point of use, in the `### env loop:` line — because a knob
  exported before the fast env would be a silent no-op. The loop then greps
  every install log for `failed to hardlink|EXDEV|Text file busy|builds-v0` and
  says so out loud either way, and totals the installs' `File system outputs`.
  Guard: `link_mode_guard.sh`, which drives the REAL template to `DRY_RUN=1`
  four times (hardlink survives a copy-setting fast env; the SHIPPED default
  line -- no substitution -- is hardlink; `copy` is still reachable; `symlink`
  refuses with exit 2). Falsified by mutation each time it changed: setting the
  default back to `copy` fails case B while case D still passes.

### What `hardlink` measured under fan-out (job `5685816`, 2026-09-03)

26 envs, `CERT_PARALLEL=4`, persistent caches, staged workspace, the certified
994-line manifest and `ae312ea0…` lock, against job `5658928` which is the same
run in every respect except the link mode.

| | copy `5658928` | hardlink `5685816` |
|---|---|---|
| env-loop span | 13 111 s | **9 057 s** (−31 %) |
| sum of env walls | 46 233 s | **32 421 s** (−30 %) |
| install writes (`/usr/bin/time -v`) | 301.39 GB | **128.03 GB** (2.35×) |
| job MaxDiskWrite (sacct) | 302 GB | **131.8 GB** |
| peak / mean concurrency | 4 / 3.53 | 4 / 3.58 |
| worst per-env RSS | 1.25 GB | 0.97 GB |
| `pace` prefix files at `st_nlink>1` | ~72 k | **156 275 of 161 609** |
| race lines in 26 install logs | — | **0** |
| `cert_verdict.sh` hardlink vs copy | | **26 rows, 0 differing, EXIT 0** |

Both runs score the same single `OUTCOME_DIFF` against the campaign baseline and
the `5597694` serial key — `pm-isaaclab` `RED-verify` vs `AMBER-repaired` — which
is the staged-vs-relocked pack-ownership residue the attribution lane isolated,
reproduces at `CERT_PARALLEL=1`, and has nothing to do with the link mode.

### The N=1 corner, and why it flipped the default (2026-09-03, C7 closeout)

The three objections above were the flip conditions. Job `5685818` (hardlink,
`CERT_PARALLEL=1`) answered all three, and the first one turned out to be the
opposite of what it looked like.

**The two "unexplained slow envs" were WIDTH, not link mode.** `5685818` ran
alongside `5674559` (copy, `CERT_PARALLEL=1`) on the SAME node, the same day,
against the same persistent cache, in the same dispatch order — so on the envs
both have scored, the link mode is the only difference. This is FINAL at 8
shared envs, not a snapshot: the copy job hit its 7 h wall clock at 12/26, and
the hardlink job wedged on `robogen` at 8/26 (backend asleep holding its own
job-scoped git-clone lock, prefix empty, CPU counters frozen -- upstream of
anything a link mode controls, and it never reached wheel placement). The ratios
held from the first four envs to the eighth:

| env | hardlink N=1 | copy N=1 | wall | hardlink GB | copy GB | bytes |
|---|---|---|---|---|---|---|
| `isaaclab-gpu-latest` | 3 842 s | 5 569 s | 0.69x | 26.93 | 48.34 | 0.56x |
| `pm-isaaclab` | 1 199 s | 3 255 s | 0.37x | 23.49 | 13.49 | 1.74x |
| `unitree-rl-lab-gpu` | 618 s | 1 105 s | 0.56x | 1.18 | 12.90 | 0.09x |
| `hover-gpu` | 1 350 s | 1 655 s | 0.82x | 24.01 | 24.35 | 0.99x |
| `groot-sonic-gpu` | **1 424 s** | 1 529 s | 0.93x | 3.40 | 14.79 | 0.23x |
| `pace` | **553 s** | 1 216 s | 0.45x | 0.78 | 12.59 | 0.06x |
| `viral-gpu` | 1 336 s | 1 541 s | 0.87x | 1.24 | 24.79 | 0.05x |
| `sage` | 676 s | 844 s | 0.80x | 6.49 | 11.52 | 0.56x |
| **8 envs** | **10 998 s** | **16 714 s** | **0.66x** | **87.52** | **162.77** | **0.54x** |

`pace` and `groot-sonic-gpu` are the two envs the fan-out run accused. At N=1
they are the FASTEST of all four corners — `pace` 553 s against 1 216 s (copy
N=1, same node/day), 1 290 s (copy N=4) and 2 734 s (hardlink N=4);
`groot-sonic-gpu` 1 424 s against 1 529 / 1 779 / 3 440. Nothing about the link
mode slowed them down; running four installs into one write path did.

That also rehabilitates the single-env test. It measured `pace` at concurrency 1
and got 397 s / 0.75 GB against copy's 1 062 s / 23.78 GB; the N=1 corner
reproduces it almost exactly (553 s / 0.78 GB). The test was never wrong — it
was **width-specific**, and quoting it at N=4 was the error. Hardlink converts an
install from a write into a refcount bump, so it gives back the most when it is
alone on the write path and the least against the N=4 bandwidth ceiling.

The one row that does not fit: `pm-isaaclab` wrote MORE under hardlink (23.49 GB
vs 13.49) while running 2.7x faster. Unexplained, and named here rather than
smoothed over.

**Gates at N=1 are clean.** Every env `5685818` has scored is identical to copy
at the same width on the same node the same day — `cert_verdict.sh` over the 8
common rows: **0 differing, EXIT 0** — and identical to hardlink at N=4 on the
same rows (**0 differing, EXIT 0**). `install_rc=0` throughout, and the race scan
over every install log at BOTH widths finds zero `hardlink|EXDEV|Text file
busy|Stale file handle|No such file` lines. The only `OUTCOME_DIFF` any corner
scores against the campaign baseline is the one `pm-isaaclab` pack-ownership
residue, which reproduces at `CERT_PARALLEL=1` under copy as well.

**The shared-cache coupling was censused, not argued.** Of 13 920 files in 60
`archive-v0` entries that pre-date the hardlink certs, the number whose CONTENT
changed is **0**. 12 557 had a ctime bump — that is the link/unlink refcount
churn — and every one is back at `st_nlink=1` now the prefixes are reclaimed.
The entry count went 4 796 -> 8 222 by ADDING entries; none were rewritten. uv
installs by unlink-and-create so uv itself cannot write through. A writer that
rewrote a prefix file IN PLACE still could, so re-run that census if one appears.

### The four corners, and what ships

| | copy | hardlink |
|---|---|---|
| **N=1** | `5597694` 20 477 s walls / 307.3 GB / 1.48 GB RSS; `5674559` today | `5685818` **0.66x wall and 0.54x bytes** of copy N=1 over 8 shared envs |
| **N=4** | `5658928` span 13 111 s / walls 46 233 s / 302 GB / 1.25 GB RSS | `5685816` span **9 057 s** / walls 32 421 s / **128 GB** / 0.97 GB RSS |

Two independent levers, and **both ship**:

* **`CERT_UV_LINK_MODE=hardlink`** — it wins at BOTH widths (0.66x wall at N=1,
  0.69x span at N=4), the gates are identical to copy at both, and the byte drop
  is 2.35x at N=4 and 1.86x at N=1. Well clear of the 527 s noise floor: 5 716 s
  and 75.2 GB saved over eight envs at N=1, 4 054 s of span at N=4.
* **`CERT_PARALLEL=4`** — width is still worth more than the link mode. Hardlink
  at N=1 spent **11 144 s on 6 of 26 envs**, where hardlink at N=4 finished all
  26 inside a **9 057 s** span. N=4 already wins by >2 000 s with 20 envs still
  to run, four times the noise floor. Take both levers; they do not overlap.

The relock keeps `copy`, unchanged: the split is by phase, and that phase builds.

## Before you call anything wedged: `wedge_triage.sh <pid> [<jobid>]`

Read-only, signals nothing, and refuses to say `WEDGED` until it has ruled out
the two states that look identical to one. It prints in order and stops at the
first verdict that applies: **(a)** `dmesg -T | grep -i lockd` — if the last line
is `not responding` with no later `OK`, `NFS-LOCK-OUTAGE (since <time>)`, and the
answer is WAIT, because the mount is hard and the process resumes on its own;
**(b)** every `pipe:` fd of the pid and its threads resolved to the holder of the
other end — a `pipe_read` thread whose write end is held by a live ancestor or
child is `IDLE-ON-RPC-CHANNEL`, the pixi→backend build-RPC stdin with no request
in flight, which is what a correct idle backend looks like; **(c)** `/proc/locks`
for the pid family with READ/WRITE, where a READ row is retread's documented
shared clone lease (`source_build.rs`, `process_clone_locks`) and blocks nobody;
**(d)** two samples 120 s apart of `utime+stime`, `rchar/wchar`, the wchan
histogram and the newest file under the job's artifact dir — only if every
counter AND the output are flat, and neither (a) nor (b) fired, does it print
`WEDGED` together with the exact `kill -TERM <pid>` line, which it never runs.
Exit codes 10/11/12/0. Guard: `wedge_triage_guard.sh`, which drives the real
script against real processes (a `coproc` pipe, an idle `sleep`) and substitutes
only `WT_DMESG_CMD` and `WT_SAMPLE_SECS`. It exists because on 2026-09-03 a
44 m 53 s site NLM outage on node2347 was diagnosed as a backend hang and a
healthy 60-minute install came within five minutes of being SIGTERMed — while on
2026-09-02 the same two symptoms twice meant a real wedge where killing was
right. Pass `WT_ARTIFACT_DIR` for a cert job: `scontrol` WorkDir is the task
directory, not the per-env `artifacts/` the install log grows in.

Two facts that belong next to it. **uv takes a per-sdist flock on the shared
persistent uv cache**, so concurrent jobs building the same path source
(`protomotions`, `pace-sim2real`) serialize on
`…/uv-cache/sdists-v9/path/<hash>/.lock` by design — expected, not a defect. But
that lock lives on the same NFS, so **under a lockd outage the ordinary wait
becomes a 3 600 s timeout that kills the job**: that is exactly how `hlgd-proof`
`5688009` died on 2026-09-03. And **`lockd` is server-side** — when
`hpcnfs.ccv.brown.edu` stops answering it stalls NLM locks on EVERY node at once
(measured the same afternoon on node2347, node2407, node1936 and node1827), so
the outage explaining a uv lock timeout may only be visible from a different host
than the job's. When a uv "waiting for lock" row is the last thing in a log,
`wedge_triage.sh` prints the lockd view for you.
