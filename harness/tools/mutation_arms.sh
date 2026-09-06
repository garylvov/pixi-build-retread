#!/usr/bin/env bash
# mutation_arms.sh -- THE VERSIONED MUTATION-ARM TEMPLATE.  L3-1b-1a-2.
#
# Law 3: "every change lands with its guard test; a guard that cannot fail is a
# defect".  The way this campaign shows a guard CAN fail is a mutation matrix --
# one arm per single-variable mutation of the code under guard, each arm a FRESH
# COPY of the worktree so the worktree itself is never dirtied and law 11 never
# comes near it.  Every lane that has done this (L3-1b-1, L3-1b-1a, p6ad-4,
# p6ad-4-3, MERGE-N-5) wrote its own `mutations.sh` BY COPYING THE PREVIOUS
# LANE'S COPY OUT OF A TASK DIRECTORY -- and `agrescap/tasks` is not a git
# repository (CLAUDE.md law 7), so the whole derivation chain lived in
# unversioned files, one `rm` from gone, with every lane's bug fix stranded in
# the copy that fixed it.  L3-1b-1a fixed the arm-cleanup defect in ITS copy and
# BOARDED that `l31b1-work/mutations.sh` still had it.  This file ends that: the
# template is versioned here, a lane SOURCES it and declares only its arms, and
# a fix lands once.
#
#   usage, from a lane's own matrix script:
#
#     source "$T/tools/mutation_arms.sh"
#     JOB_ROOT=$T/<lane>-work  WT=<worktree>  A_DIR=$JOB_ROOT/mut  MUT_JOBS=1 \
#       mut_init
#
#   A SECOND, CONCURRENT MATRIX POINTS `A_DIR` SOMEWHERE ELSE AND GETS ITS OWN
#   SCRATCH -- that is what `A_DIR` is FOR (see defect 3 below):
#
#     JOB_ROOT=$T/<lane>-work  WT=<worktree>  A_DIR=$JOB_ROOT/mut2  MUT_JOBS=1 \
#       mut_init
#     GUARDS=( name_of_guard_one name_of_guard_two )
#     run_arm BASE src/foo.rs ""            GREEN || bad=$((bad+1))
#     run_arm M1   src/foo.rs 's|a|b|'      RED   || bad=$((bad+1))
#     mut_done "$bad"
#
# ── THE TWO DEFECTS THIS TEMPLATE EXISTS TO MAKE UNREPEATABLE ────────────────
#
# (1) ARM SCRATCH LANDED IN `/tmp`, AND `/tmp` IS NOT ALWAYS A DISK.  The
#     inherited `run_arm` put every arm's full worktree-plus-`target/` copy under
#     `${SLURM_TMPDIR:-/tmp}` and removed an arm directory only at the START of
#     an arm OF THE SAME NAME, so nine differently-named arms accumulated nine
#     copies.  Job 5954481 died `OUT_OF_MEMORY 0:125` in arm M7 at `MaxRSS
#     100662600K` = 96.0 GiB exactly against `ReqMem 96G`.
#
#     THE HONEST HISTORY, because a fix built on a wrong diagnosis is the next
#     defect: the FIRST diagnosis of 5954481 was "/tmp is RAM-backed and nine
#     accumulated copies are the cause", and `df -PT` on that node says `/tmp`
#     is DISK, so on THAT node the copies cost inodes and bytes and not one byte
#     of RSS.  The OOM was `JOBS=4` -- four concurrent rustc codegen jobs in one
#     arm, when one retread test build already needs the whole 96 G on its own.
#     BOTH are real and this template holds both: `MUT_JOBS` defaults to 1, the
#     arm directory is removed on EVERY exit path, the scratch root is the JOB
#     ROOT ON DISK rather than anything temp-shaped, and -- because the first
#     diagnosis was wrong ONLY on this cluster's nodes and would be right on a
#     node whose `/tmp` is `tmpfs` -- `mut_init` MEASURES the filesystem and
#     REFUSES rather than assuming either way.
#
# (2) NOTHING MEASURED THE FILESYSTEM AT THE POINT OF USE.  CLAUDE.md law 5.
#     `mut_init` prints `df -PT` of the real scratch and `stat -f` of its
#     filesystem type, and refuses on `tmpfs`/`ramfs`/`devtmpfs`.
#
# (3) DET-1-2, MEASURED 2026-09-06.  `mut_init` DERIVED `MUT_SCRATCH` FROM
#     `$JOB_ROOT` AND IGNORED `A_DIR` ENTIRELY, so the documented way to keep a
#     first run's arm logs -- point the second matrix at `mut2/` -- moved the
#     ARTIFACTS and left the second matrix building its arm directories in the
#     FIRST run's live scratch.  DET-1 hit exactly this: run 1 (5981304, A_DIR
#     `mut`) and run 2 (5981742, A_DIR `mut2`) shared `$JOB_ROOT/mut`, and both
#     runs were declared dead evidence rather than mined.  A third matrix could
#     not be submitted at all while the first two were alive.  Two halves of the
#     fix, and both are needed -- one makes the collision impossible to reach by
#     accident, the other makes it impossible to reach on purpose:
#       * THE SCRATCH ROOT FOLLOWS THE DECLARATION.  `MUT_SCRATCH` if the lane
#         set one; else `A_DIR`, which is the knob every lane already reaches
#         for; else `$JOB_ROOT/mut`, unchanged for every caller that declares
#         neither.  `mut_init` PRINTS the root and WHERE IT CAME FROM.
#       * A LOCK FILE WITH A PID.  A scratch root in use by a live matrix is
#         REFUSED rc 6, naming the pid, the host, the slurm job and the arm
#         directories already in it.  A lock whose owner is gone is announced as
#         STALE and taken over -- a crashed job must not wedge the next run.
#
# ── REFUSALS (nothing is created before they are all passed) ─────────────────
#   rc 3  `JOB_ROOT` or `WT` unset, or `JOB_ROOT` cannot be created
#   rc 4  the scratch root is on a RAM-backed filesystem, or it resolves under a
#         RAM-backed `/tmp` or `$TMPDIR`
#   rc 5  `WT` is unset or is not a directory
#   rc 6  the chosen scratch root is held by a LIVE matrix (DET-1-2)
# and from `run_arm`:
#   rc 99 the mutation changed nothing -- a mutation that does not mutate proves
#         nothing
#   rc 98 the arm printed no `test result:` line -- it did not run
#   rc 1  the arm's colour is not the one declared
# and, DET-1-4, NOT a return but an EXIT: a RED BASE arm ends the whole matrix
# through `mut_done 1 base-red` with rc `MUT_BASE_FAIL_RC` (default 1), because
# 5981304 printed `BASE FAILED` and then `MUT_EXIT=0` / `COMPLETED 0:0` -- the
# detector fired into a lane script that swallowed the return.
#
# Sourcing this file defines functions and touches nothing.  `mut_init` is what
# acts.
set -uo pipefail

# Filesystem types that are RAM.  A scratch root on one of these charges every
# byte an arm writes to the job's RSS, which is the failure mode above.
MUT_RAM_FSTYPES="tmpfs ramfs devtmpfs"

# One place that answers "what filesystem is this path on".  `stat -f -c %T`
# first because it names the type directly; `df -PT` as the fallback and as the
# row a reader actually greps.
mut_fstype() {
  local p="$1" t=""
  t="$(stat -f -c %T "$p" 2>/dev/null)" || t=""
  if [ -z "$t" ]; then
    t="$(df -PT "$p" 2>/dev/null | awk 'NR==2 {print $2}')"
  fi
  printf '%s' "${t:-unknown}"
}

mut_is_ram_fs() {
  local t; t="$(mut_fstype "$1")"
  case " $MUT_RAM_FSTYPES " in
    *" $t "*) return 0 ;;
    *)        return 1 ;;
  esac
}

# Is `$1` inside `$2`, as resolved paths?  Used to catch a JOB_ROOT that is
# itself a symlink into /tmp.
mut_path_inside() {
  local child parent
  child="$(readlink -f -- "$1" 2>/dev/null)" || return 1
  parent="$(readlink -f -- "$2" 2>/dev/null)" || return 1
  [ -n "$child" ] && [ -n "$parent" ] || return 1
  case "$child/" in "$parent"/*) return 0 ;; *) return 1 ;; esac
}

# Declare where the arms run, prove it is not RAM, and print the proof.
# Exports MUT_SCRATCH, MUT_JOBS, A_DIR.
mut_init() {
  # Spelled out rather than as `${JOB_ROOT:?...}` on purpose: in a
  # non-interactive shell that construct EXITS the shell with rc 1 instead of
  # returning, so the refusal could neither be tested nor distinguished from any
  # other failure. A refusal has to be a return code a guard can assert on.
  if [ -z "${JOB_ROOT:-}" ]; then
    echo "### MUT REFUSED: set JOB_ROOT to this lane's work directory ON DISK (L3-1b-1a-2)"
    return 3
  fi
  if [ -z "${WT:-}" ]; then
    echo "### MUT REFUSED: set WT to the worktree the arms copy from"
    return 5
  fi
  [ -d "$WT" ] || { echo "### MUT REFUSED: WT is not a directory: $WT"; return 5; }
  # STORE-REAP-2 ROOT FIX. `run_arm` used to read $WT directly, and bash
  # DISCARDS a `WT=... mut_init` prefix assignment when the function returns --
  # so the shape this file own USAGE COMMENT showed (`JOB_ROOT=... WT=...
  # mut_init`) initialised fine and then died `WT: unbound variable` in the
  # first arm, under `set -u`, after the fixture had already been built.
  # Measured: sr2-mut 5966771, rc 1, zero arms run. The guard knew the quirk and
  # worked around it with plain assignments; the template did not, so every
  # lane that copied the usage line inherited the trap. `mut_init` now OWNS the
  # value: it records it in a global the arms read, and both call shapes work.
  MUT_WT="$WT"
  mkdir -p "$JOB_ROOT" 2>/dev/null || {
    echo "### MUT REFUSED: cannot create JOB_ROOT $JOB_ROOT"; return 3; }
  [ -d "$JOB_ROOT" ] || { echo "### MUT REFUSED: JOB_ROOT is not a directory: $JOB_ROOT"; return 3; }

  # DET-1-2: the scratch root FOLLOWS THE DECLARATION, and says where it came
  # from. Explicit `MUT_SCRATCH` first, then `A_DIR` (the knob a lane already
  # moves when it wants a second matrix), then the job-root default -- which is
  # what every caller that declares neither keeps getting.
  #
  # AND `mut_init` MUST NOT READ BACK ITS OWN EXPORT.  It exports `MUT_SCRATCH`
  # and `A_DIR`, so a SECOND `mut_init` in the same shell -- or in a subshell of
  # it, which is how a guard and a two-matrix lane both do this -- would see its
  # predecessor's derived value sitting in the "the caller declared this"
  # position and IGNORE the new declaration entirely.  That is the DET-1-2 defect
  # again one level up, and it was measured on the very first run of this fix:
  # the guard's prefix-assignment arm inherited `MUT_SCRATCH` from the arm before
  # it, kept the FIRST matrix's root, and was refused rc 6 by its own live lock.
  # So each derived value is remembered, and a value that is byte-equal to what
  # this function last derived is treated as ABSENT, not as a declaration.
  if [ -n "${MUT_SCRATCH:-}" ] && [ "${MUT_SCRATCH:-}" = "${MUT_SCRATCH_DERIVED:-}" ]; then
    MUT_SCRATCH=
  fi
  if [ -n "${A_DIR:-}" ] && [ "${A_DIR:-}" = "${MUT_ADIR_DERIVED:-}" ]; then
    A_DIR=
  fi
  local scratch_src
  if [ -n "${MUT_SCRATCH:-}" ]; then
    scratch_src=MUT_SCRATCH
  elif [ -n "${A_DIR:-}" ]; then
    MUT_SCRATCH="$A_DIR"; scratch_src=A_DIR
  else
    MUT_SCRATCH="$JOB_ROOT/mut"; scratch_src=JOB_ROOT-default
  fi
  # `MUT_ADIR_DERIVED` is recorded ONLY when this function invented the value.
  # An `A_DIR` the CALLER set is the caller's, and re-declaring the same one in a
  # second matrix must keep meaning what it says.
  if [ -z "${A_DIR:-}" ]; then
    A_DIR="$JOB_ROOT/mut-artifacts"; MUT_ADIR_DERIVED="$A_DIR"
  else
    MUT_ADIR_DERIVED=
  fi
  MUT_SCRATCH_DERIVED="$MUT_SCRATCH"
  MUT_JOBS="${MUT_JOBS:-1}"
  MUT_LOCK="$MUT_SCRATCH/.mut_lock"

  # THE REFUSALS, all measured, and BEFORE anything is created under the root.
  local jt; jt="$(mut_fstype "$JOB_ROOT")"
  echo "### MUT scratch root=$MUT_SCRATCH from=$scratch_src fstype=$jt (DET-1-2: A_DIR moves the scratch, not just the logs)"
  echo "### MUT df -PT JOB_ROOT: $(df -PT "$JOB_ROOT" 2>/dev/null | tail -1)"
  if mut_is_ram_fs "$JOB_ROOT"; then
    echo "### MUT REFUSED: JOB_ROOT $JOB_ROOT is on $jt, which is RAM."
    echo "### MUT   Every byte an arm writes would count against the job's RSS."
    echo "### MUT   Job 5954481 died OUT_OF_MEMORY at MaxRSS 100662600K = 96.0 GiB."
    return 4
  fi
  # ...and the same refusal for the two places a scratch root drifts back to,
  # but ONLY when they really are RAM here -- on this cluster's batch nodes
  # `/tmp` measured as DISK, and refusing a disk-backed /tmp on a stored belief
  # would be exactly the "measure, don't quote" defect in the other direction.
  local shadow
  for shadow in "${TMPDIR:-/tmp}" /tmp; do
    [ -d "$shadow" ] || continue
    if mut_path_inside "$JOB_ROOT" "$shadow" && mut_is_ram_fs "$shadow"; then
      echo "### MUT REFUSED: JOB_ROOT $JOB_ROOT resolves under $shadow, which is $(mut_fstype "$shadow")."
      return 4
    fi
  done

  # ── DET-1-2 THE LOCK, BEFORE ANY ARM DIRECTORY IS CREATED ──────────────────
  # Two matrices in one scratch root delete each other's arm directories --
  # `run_arm` does `rm -rf "$MUT_SCRATCH/$name"` on entry and on every exit -- so
  # the loser's colours are not wrong, they are MEANINGLESS, and that is worse.
  # The lock names the pid, the host and the slurm job so the refusal can say
  # WHO holds it rather than "busy".
  if [ -f "$MUT_LOCK" ]; then
    local l_pid l_host l_job l_when l_live armdirs
    l_pid=$(sed -n 's/^pid=//p'  "$MUT_LOCK" | head -1)
    l_host=$(sed -n 's/^host=//p' "$MUT_LOCK" | head -1)
    l_job=$(sed -n 's/^job=//p'  "$MUT_LOCK" | head -1)
    l_when=$(sed -n 's/^when=//p' "$MUT_LOCK" | head -1)
    armdirs=$(find "$MUT_SCRATCH" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
    l_live=0
    # The pid only means anything on the host that minted it, and a pid scan
    # must not be able to match the scanner (law 14): `kill -0` on a NAMED pid
    # cannot.
    if [ -n "$l_pid" ] && [ "$l_host" = "$(hostname)" ] && kill -0 "$l_pid" 2>/dev/null; then
      l_live=1
    fi
    # Across nodes the slurm job id is the only honest liveness signal.
    if [ "$l_live" = 0 ] && [ -n "$l_job" ] && [ "$l_job" != "-" ] && command -v squeue >/dev/null 2>&1; then
      if [ -n "$(squeue -h -j "$l_job" -o '%T' 2>/dev/null)" ]; then l_live=1; fi
    fi
    if [ "$l_live" = 1 ]; then
      echo "### MUT REFUSED: scratch root $MUT_SCRATCH is HELD BY A LIVE MATRIX (DET-1-2)"
      echo "### MUT   holder: pid=$l_pid host=$l_host job=$l_job since=$l_when"
      echo "### MUT   arm directories already in it: $armdirs"
      echo "### MUT   Two matrices in one scratch root delete each other's arm dirs, which"
      echo "### MUT   is how DET-1's runs 1 and 2 both became dead evidence. Point this run"
      echo "### MUT   somewhere else: A_DIR=\$JOB_ROOT/mut2 (or MUT_SCRATCH=<path>)."
      return 6
    fi
    echo "### MUT lock at $MUT_LOCK is STALE (pid=$l_pid host=$l_host job=$l_job since=$l_when) -- taking it over"
  fi

  mkdir -p "$MUT_SCRATCH" "$A_DIR" || {
    echo "### MUT REFUSED: cannot create $MUT_SCRATCH / $A_DIR"; return 3; }
  printf 'pid=%s\nhost=%s\njob=%s\nwhen=%s\nscratch_from=%s\n' \
    "$$" "$(hostname)" "${SLURM_JOB_ID:--}" "$(date -Is)" "$scratch_src" > "$MUT_LOCK"
  echo "### MUT lock $MUT_LOCK pid=$$ job=${SLURM_JOB_ID:--}"
  echo "### MUT INIT ok host=$(hostname) $(date -Is) scratch=$MUT_SCRATCH artifacts=$A_DIR MUT_JOBS=$MUT_JOBS"
  echo "### MUT worktree WT=$WT HEAD=$(git -C "$WT" rev-parse --short HEAD 2>/dev/null) dirty=$(git -C "$WT" status --porcelain 2>/dev/null | wc -l)"
  export MUT_SCRATCH A_DIR MUT_JOBS MUT_LOCK MUT_SCRATCH_DERIVED MUT_ADIR_DERIVED
  return 0
}

# What one arm actually runs, in the arm's own directory.  A lane redefines this
# ONLY when its arms are not `cargo test --lib`; the guard for this template
# redefines it so the guard needs no compiler.
mut_arm_command() {
  local dir="$1"
  ( cd "$dir" && timeout --foreground --kill-after=60s "${MUT_ARM_TIMEOUT:-2400}s" \
      cargo test --lib -j "$MUT_JOBS" -- "${GUARDS[@]}" )
}

# run_arm <name> <file relative to src/> <sed expression or ""> <GREEN|RED>
run_arm() {
  local name="$1" file="$2" sedexpr="$3" expect="$4"
  local dir="$MUT_SCRATCH/$name"
  # ON EVERY EXIT PATH.  Nine full `target/` copies left behind is a real
  # inode-quota defect on a filesystem this campaign has driven to its soft
  # limit twice, and it is a defect regardless of whether the scratch is RAM.
  arm_done() { local rc="$1"; rm -rf "$dir"; return "$rc"; }
  rm -rf "$dir"; mkdir -p "$dir" || { echo "### $name FATAL: cannot create $dir"; return 97; }
  cp -a "$MUT_WT"/. "$dir"/ 2>/dev/null
  rm -rf "$dir/target"
  [ -d "$MUT_WT/target" ] && cp -a "$MUT_WT/target" "$dir/target" 2>/dev/null
  if [ -n "$sedexpr" ]; then
    sed -i "$sedexpr" "$dir/src/$file"
    if cmp -s "$MUT_WT/src/$file" "$dir/src/$file"; then
      echo "### $name FATAL: the mutation changed NOTHING -- a mutation that does not mutate proves nothing"
      arm_done 99; return 99
    fi
  fi
  mut_arm_command "$dir" > "$A_DIR/$name.log" 2>&1
  local rc=$?
  local split; split=$(grep -E '^test result:' "$A_DIR/$name.log" | tail -1)
  echo "### $name file=$file rc=$rc expect=$expect split=${split:-<none printed>}"
  grep -E "^test .*(${MUT_ROW_FILTER:-.})" "$A_DIR/$name.log" | sed "s/^/###   $name /"
  [ -n "$split" ] || { echo "### $name FATAL: no test-result line -- it did not run"; arm_done 98; return 98; }
  if [ "$expect" = GREEN ]; then
    if [ "$rc" -ne 0 ]; then
      echo "### $name FAILED: expected GREEN, got rc=$rc"
      arm_done 1
      # ── DET-1-4: A RED BASE ARM IS A TERMINAL EVENT, AND IT USED TO REACH
      #    NOBODY. MEASURED: det1-mut 5981304 printed `### BASE FAILED: expected
      #    GREEN, got rc=101`, then `### MUT_EXIT=0`, NO `MUT DONE` footer at
      #    all, and Slurm recorded COMPLETED 0:0. The detector fired and the
      #    actuator was a lane script that happened to swallow the return
      #    (law 9). It cannot be left to the caller: a base that does not build
      #    makes every mutant colour in the matrix MEANINGLESS -- a mutant that
      #    "goes RED" against a base that is already red proves nothing -- so the
      #    template ends the run itself, through the same `mut_done` path every
      #    other exit takes, so the footer and the arm-directory count are always
      #    printed and the job's rc is always non-zero.
      if [ "$name" = "${MUT_BASE_ARM:-BASE}" ]; then
        echo "### MUT BASE RED -- the matrix is VOID: every mutant colour would be measured"
        echo "### MUT   against a base that does not build. Ending here rather than reporting"
        echo "### MUT   colours nobody can read (law 9)."
        mut_done 1 base-red
        exit "${MUT_BASE_FAIL_RC:-1}"
      fi
      return 1
    fi
  else
    [ "$rc" -ne 0 ] || { echo "### $name FAILED: expected RED, the mutation passed the guards"; arm_done 1; return 1; }
  fi
  echo "### $name OK"
  arm_done 0
  return 0
}

# Close out: prove nothing was left behind, and exit on the tally.
mut_done() {
  local bad="${1:-0}" reason="${2:-}"
  local left; left=$(find "$MUT_SCRATCH" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
  echo "### MUT arm directories left behind: $left (must be 0)"
  # DET-1-2: the scratch root can now BE the artifacts directory (`A_DIR`), so a
  # blanket `rm -rf "$MUT_SCRATCH"` would delete the very arm logs a lane moved
  # `A_DIR` in order to keep. Remove what this template created -- the arm
  # directories and the lock -- and take the root itself only when nothing else
  # is in it.
  find "$MUT_SCRATCH" -maxdepth 1 -mindepth 1 -type d -exec rm -rf {} + 2>/dev/null
  rm -f "${MUT_LOCK:-$MUT_SCRATCH/.mut_lock}"
  rmdir "$MUT_SCRATCH" 2>/dev/null
  echo "### MUT DONE bad=$bad${reason:+ reason=$reason} $(date -Is)"
  [ "$bad" -eq 0 ] && [ "$left" -eq 0 ]
}
