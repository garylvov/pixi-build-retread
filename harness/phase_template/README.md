# `tools/phase_template/` — the fast, correct starting point for a new relock/cert pair

Three files:

| file | what it is |
|---|---|
| `phaseN_relock.sh` | the RELOCK half. Stages a pristine workspace, gates the manifest, locks under **persistent** caches, writes `artifacts/relock_env.sh` for the cert. Removes nothing. |
| `phaseN_cert.sh` | the CERT half. Reads that stamp, installs + probes + verifies every x86 env, scores against the certified baseline, then **submits** `cleanup.sh` and exits. Removes nothing itself. |
| `cleanup.sh` | a 1-CPU/4G job whose whole purpose is `rm -rf` on job-scoped roots. Refuses anything that is not `/oscar/data/stellex/glvov/retread/{cert*,ws.*}`. |

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

2. **Edit ONLY between `### SUBSTITUTE: BEGIN` and `### SUBSTITUTE: END`.**
   Everything a new batch changes lives in that one block: `TAG`, `D`, the
   manifest and its md5/line-count/diff-count gates, the residual-pin patterns,
   the probes file and its forbidden module tokens, the env list, the binsnap
   and its sha, the instrument paths, and `LEFTOVER_RE`. Set `TAG` to the same
   value in both files, and point the cert's `P1D` at the relock's directory
   when the two phases live in different dirs.

   `FAST_ENV` defaults to `$(dirname "$0")/../retread_fast_env.sh` and falls
   back to `$T/tools/retread_fast_env.sh`, so it resolves from either the
   template directory or a sibling batch directory. Leave it alone unless you
   moved the snippet.

3. **`bash -n` both, then let the scripts check themselves.**
   ```
   bash -n <newbatch>_relock.sh && bash -n <newbatch>_cert.sh
   ```
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
       --mem=72G  --time=03:00:00 --job-name=<tag>-p1 \
       --output=<T>/<newbatch>/logs/slurm-%j.out ./<newbatch>_relock.sh
   env -u SLURM_JOB_ID sbatch --partition=batch --qos=normal --cpus-per-task=16 \
       --mem=100G --time=08:00:00 --job-name=<tag>-p2 --dependency=afterok:<p1 job> \
       --output=<T>/<newbatch>/logs/slurm-%j.out ./<newbatch>_cert.sh
   ```
   72G relock / 100G cert. Worst relock peak RSS ever measured with
   `/usr/bin/time -v` is **8,854,172 K** (job 5597671); worst cert *env* is
   **1,475,216 K** (env `gpu`, job 5597694). `sacct` MaxRSS is unusable here —
   every job reports ~100 % of its cgroup cap regardless of what it did, which
   is reclaimable page cache, not demand. The binding constraint is the per-user
   QOS cap (`normal` = cpu 64, mem 492G for the *whole* user): a 160G request is
   why job 5597889 pended behind `QOSMaxMemoryPerUser` with a node sitting idle.

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

Hence the rule (HANDOFF §2): extract artifacts, exit, clean up from the last job
of the chain. In this template the relock removes nothing at all, and the cert
computes its verdict, prints the roots, submits `cleanup.sh` with
`--dependency=afterany:<cert job>`, and exits. `afterany`, not `afterok`, so a
RED cert still returns its disk; and because the roots are printed before the
cleanup job exists, a diagnostician who wants to keep them can hold them by
cancelling that cleanup job.

`cleanup.sh` refuses any path outside `/oscar/data/stellex/glvov/retread/` and
any basename that is not `cert*` or `ws.*`. The persistent cache is outside that
prefix by construction and is never touched by it.

---

## The persistent cache

`retread_fast_env "$WS"` (from `tools/retread_fast_env.sh`, sourced by both
halves after the job-scoped env block) repoints `PIXI_CACHE_DIR`,
`RATTLER_CACHE_DIR` and `UV_CACHE_DIR` at
`/oscar/data/stellex/glvov/agrescap/cache/retread/{pixi,rattler,uv}`, sets
`UV_LINK_MODE=copy`, and places a symlink for the route-probe verdict cache at
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
