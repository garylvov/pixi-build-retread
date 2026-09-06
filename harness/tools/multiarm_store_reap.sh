#!/usr/bin/env bash
# multiarm_store_reap.sh -- THE ONE WAY A MULTI-ARM DRIVER RETIRES A JOB-SCOPED
# STORE BETWEEN ARMS.  ORDER-1-1.
#
# Same reason `mutation_arms.sh` and `phase_template/lane_wrapper.sbatch` exist:
# THIRTEEN task-only drivers carry a hand-copied `mv $X $X.reap; ( rm -rf
# $X.reap ) &` block today, every one of them descended from the one before it
# by copy, and `agrescap/tasks` is not a git repository (CLAUDE.md law 7).  A fix
# in one copy is stranded in that copy.  So the block lives here, a lane SOURCES
# it, and a fix lands once.
#
#   usage, from a lane's own arm function:
#
#     source "$T/tools/multiarm_store_reap.sh"
#     REAP_JOB_ROOT=$C reap_init            # $C = this job's own root
#     ...
#     reap_aside  "$WS"                     "a$AN"   # the DEFAULT: rename + hand over
#     reap_delete "$C/git-snapshots-cold-$ARM_SCOPE" "a$AN"   # only if bytes are owed NOW
#     ...
#     reap_handoff_list "$A/reap-owed.txt"  # what the cleanup owner must finish
#     reap_done || rc=$?                    # the ACTUATOR -- non-zero reaches Slurm
#
# ── THE DEFECT IT EXISTS TO MAKE UNREPEATABLE ────────────────────────────────
#
# ORDER-1's `order1-proof` **5980023** (node2409, COMPLETED 02:44:54, ExitCode
# 0:0) printed **42 252** lines of
#
#   rm: cannot remove '/oscar/data/stellex/glvov/retread/certORDER1-5980023/
#       git-snapshots-cold-a<N>.reap/canonical-git-sources/v3/<identity>/
#       <ref state>/repo/<file>': Permission denied
#
# -- 11 964 for each of arms 1, 2, 3 (identical counts: three complete failures
# over the same tree) and 6 360 for arm 4, cut short when the job ended -- and
# then printed `### ORDER1 PROOF DONE job_fatal=0` and exited 0.  Forty-two
# thousand refusals that reached no actuator: CLAUDE.md law 9 exactly.
#
# ── THE CAUSE, WHICH IS THE SEAL AND NOT THE FILES ───────────────────────────
#
# The tree being reaped is C18's canonical Git snapshot store.  `source_build.rs
# ::make_source_tree_read_only` walks the published tree and, for EVERY node --
# **directories included** -- sets `metadata.permissions().mode() & !0o222`.
# Stripping `w` from a FILE does not stop `unlink`; stripping `w` from its
# PARENT DIRECTORY does, because unlinking an entry is a write to the directory.
# So `rm -rf` fails on every leaf with EACCES and never even reaches the
# `rmdir`: measured on 5980023, `Permission denied` 42 245, `Directory not
# empty` **0**.  That is the signature -- a seal, not a busy tree, not NFS.
#
# The seal is also VALIDATED on the next hit (C18-1: "a seal whose root regained
# a write bit ... is REFUSED fail-closed, never repaired"), which is why the
# answer is not "stop sealing" and not "chmod the store".  It is: DO NOT DELETE
# A SEALED STORE IN-JOB.  Rename it aside and let the job's own `afterany`
# cleanup owner -- which runs after the arms, with the whole job's wall clock,
# and is already submitted by every one of these drivers -- finish it.
#
# ── THE FOUR RULES, and each is a function below ─────────────────────────────
#   1. `reap_aside` is the DEFAULT and it never runs `rm`.  It renames with the
#      ARM LABEL (so a four-arm job's four retirements are distinguishable) and
#      records the path for the cleanup owner.  A rename cannot fail on a sealed
#      tree: the seal is on the tree's OWN nodes, and the rename is a write to
#      the PARENT, which is the job root.
#   2. `reap_delete` exists for the case the bytes are genuinely owed before the
#      next arm stages.  It renames FIRST, then `chmod -R u+w` the RENAMED path
#      -- re-proving containment AFTER the rename, so the chmod can only ever
#      touch this job's own root -- then removes it SYNCHRONOUSLY with stderr
#      captured and COUNTED.  Not backgrounded: `( rm -rf ) &` is what threw the
#      rc away in the first place, and an error count is the whole point.
#   3. EVERY reap REFUSES a path that is not under the job's own root, and the
#      refusal is MEASURED -- `stat` dev:inode walked up through `..`, not a
#      string prefix.  The order1 driver's guard was `case "$cr" in
#      /oscar/.../cache/retread*)`, a prefix test that (a) a symlink walks
#      straight past and (b) never fired anyway, because the paths it was
#      handed were job-scoped.  A prefix test answers the wrong question: the
#      question is not "does this look like the shared store", it is "is this
#      MINE".
#   4. `reap_done` is the ACTUATOR.  Non-zero if any `rm` reported an error or
#      any path was refused, so the driver's `exit "$rc"` carries it to Slurm.
#
# ── WHAT IS NOT HERE, DELIBERATELY ───────────────────────────────────────────
# No background delete of any kind.  ORDER-1's comment argued the last arm's
# background `rm` is always cut short by the job ending and so must ALSO be
# handed to the cleanup owner -- which was true, and is the admission that the
# background delete was never the mechanism that reclaimed anything.  The
# cleanup owner is.
#
# Nothing in here reads `SLURM_JOB_ID`.  The job root is DECLARED by the caller
# and proved by inode; deriving it from the environment would make the one
# refusal that matters depend on a variable a fixture cannot honestly set.
set -uo pipefail

# ── STATE, all of it visible to the caller ──────────────────────────────────
REAP_JOB_ROOT="${REAP_JOB_ROOT:-}"   # declared by the caller, proved by reap_init
REAP_HANDED=()                        # renamed-aside paths the cleanup owner owes
REAP_ERRORS=0                         # rm/chmod error LINES counted this job
REAP_DELETED=0                        # trees reap_delete actually removed
REAP_ASIDE=0                          # trees renamed aside
REAP_REFUSED=0                        # paths refused by the containment test
REAP_INIT_DONE=0

# The roots that must NEVER be handed to this file, in either direction: a
# target under one of them, or a target that CONTAINS one of them.  Only the
# PERSISTENT cache root is listed -- `RETREAD_WHEEL_STORE` and
# `RETREAD_GIT_SNAPSHOT_STORE` are deliberately absent, because a truly-cold arm
# points both of those AT ITS OWN JOB ROOT and reaping them is the legitimate
# case this file is for.  Extra roots: `REAP_PERSISTENT_ROOTS`, space separated.
REAP_PERSISTENT_DEFAULT="${RETREAD_PERSIST_CACHE_ROOT:-/oscar/data/stellex/glvov/agrescap/cache/retread}"

# ── THE MEASUREMENT ─────────────────────────────────────────────────────────

# dev:inode of a path, following symlinks (the same resolution `mv`, `chmod`
# and `rm` will do).  Empty output means the path is not there.
reap_devino() { stat -c '%d:%i' -- "$1" 2>/dev/null; }

# Is $1 STRICTLY under $2?  Walk up from $1 through `..` comparing dev:inode.
# The kernel resolves `..` after resolving each symlink, so a symlinked or
# bind-mounted path is answered by where it REALLY is -- which a `case` prefix
# test cannot do.  rc 0 strictly under, rc 1 not under (or equal), rc 2 the
# ancestor could not be stated at all.
reap_is_under() {
  local child="$1" want cur d up depth=0
  want="$(reap_devino "$2")"
  [ -n "$want" ] || return 2
  cur="$(reap_devino "$1")"
  [ -n "$cur" ] || return 1
  [ "$cur" = "$want" ] && return 1          # equal is NOT under
  cur="$1"
  while [ "$depth" -lt 256 ]; do
    cur="$cur/.."
    d="$(reap_devino "$cur")"
    [ -n "$d" ] || return 1
    [ "$d" = "$want" ] && return 0
    up="$(reap_devino "$cur/..")"
    [ -n "$up" ] || return 1
    [ "$up" = "$d" ] && return 1            # `..` is itself: the filesystem root
    depth=$((depth + 1))
  done
  return 1
}

# ── INIT ────────────────────────────────────────────────────────────────────
# Declares and PROVES the job's own root.  Spelled out rather than
# `${REAP_JOB_ROOT:?}` for the reason mutation_arms.sh spells its own out: that
# construct EXITS a non-interactive shell instead of returning, so the refusal
# could not be asserted on.
#   rc 0 ready   rc 3 no usable job root   rc 5 the job root is itself off-limits
reap_init() {
  if [ -z "${REAP_JOB_ROOT:-}" ]; then
    echo "### REAP REFUSED: set REAP_JOB_ROOT to THIS JOB'S OWN root (ORDER-1-1)"
    return 3
  fi
  if [ ! -d "$REAP_JOB_ROOT" ]; then
    echo "### REAP REFUSED: REAP_JOB_ROOT is not a directory: $REAP_JOB_ROOT"
    return 3
  fi
  local p
  for p in $REAP_PERSISTENT_DEFAULT ${REAP_PERSISTENT_ROOTS:-}; do
    [ -e "$p" ] || continue
    if [ "$(reap_devino "$p")" = "$(reap_devino "$REAP_JOB_ROOT")" ] \
       || reap_is_under "$REAP_JOB_ROOT" "$p"; then
      echo "### REAP REFUSED: REAP_JOB_ROOT $REAP_JOB_ROOT is at or under the PERSISTENT root $p"
      return 5
    fi
  done
  REAP_HANDED=(); REAP_ERRORS=0; REAP_DELETED=0; REAP_ASIDE=0; REAP_REFUSED=0
  REAP_INIT_DONE=1
  echo "### REAP init job_root=$REAP_JOB_ROOT devino=$(reap_devino "$REAP_JOB_ROOT") persistent=$REAP_PERSISTENT_DEFAULT"
  return 0
}

# The shared refusal.  Every reap goes through it and it answers ONE question by
# measurement: is this path strictly inside the root this job declared, and does
# it contain nothing that outlives the job?
#   rc 0 may be reaped   rc 3 not initialised   rc 6 refused
reap_check_target() {
  local t="$1" why=""
  if [ "$REAP_INIT_DONE" != 1 ]; then
    echo "### REAP REFUSED: reap_init has not run"; return 3
  fi
  if [ -L "$t" ]; then
    why="it is a SYMLINK; renaming or deleting it acts on the link, not the tree"
  elif [ ! -e "$t" ]; then
    why="it does not exist"
  elif ! reap_is_under "$t" "$REAP_JOB_ROOT"; then
    why="it is NOT strictly under the declared job root $REAP_JOB_ROOT (dev:inode walk, not a string prefix)"
  else
    local p
    for p in $REAP_PERSISTENT_DEFAULT ${REAP_PERSISTENT_ROOTS:-}; do
      [ -e "$p" ] || continue
      if [ "$(reap_devino "$p")" = "$(reap_devino "$t")" ] || reap_is_under "$p" "$t"; then
        why="the PERSISTENT root $p is at or under it"; break
      fi
    done
  fi
  [ -z "$why" ] && return 0
  REAP_REFUSED=$((REAP_REFUSED + 1))
  echo "### REAP REFUSED $t -- $why"
  return 6
}

# ── RULE 1: THE DEFAULT.  RENAME ASIDE, HAND IT OVER, RUN NO `rm`. ──────────
# reap_aside <path> [<arm label>]
reap_aside() {
  local t="$1" label="${2:-}" dst
  reap_check_target "$t" || return $?
  dst="$t.reap"; [ -n "$label" ] && dst="$t.reap-$label"
  if [ -e "$dst" ]; then
    dst="$dst-$$"
  fi
  if ! mv -- "$t" "$dst" 2>&1; then
    REAP_ERRORS=$((REAP_ERRORS + 1))
    echo "### REAP could not rename $t -> $dst; left in place for the cleanup owner"
    REAP_HANDED+=("$t")
    return 1
  fi
  REAP_HANDED+=("$dst")
  REAP_ASIDE=$((REAP_ASIDE + 1))
  echo "### REAP ASIDE arm=${label:-none} $t -> $dst (OWED to this job's afterany cleanup owner; no rm ran)"
  return 0
}

# ── RULE 2: ONLY WHEN THE BYTES ARE OWED BEFORE THE NEXT ARM STAGES ─────────
# reap_delete <path> [<arm label>]
# rename -> re-prove containment -> chmod -R u+w -> synchronous rm -> COUNT.
# Whatever survives is handed to the cleanup owner as well, so a partial delete
# still has an owner.  rc 0 gone with zero errors; rc 1 errors (counted); rc 6
# refused.
reap_delete() {
  local t="$1" label="${2:-}" dst errf chmod_err rm_err n=0
  reap_check_target "$t" || return $?
  dst="$t.reap"; [ -n "$label" ] && dst="$t.reap-$label"
  [ -e "$dst" ] && dst="$dst-$$"
  if ! mv -- "$t" "$dst" 2>&1; then
    REAP_ERRORS=$((REAP_ERRORS + 1))
    echo "### REAP could not rename $t -> $dst; nothing removed, left for the cleanup owner"
    REAP_HANDED+=("$t"); return 1
  fi
  # RE-PROVE IT AFTER THE RENAME.  The chmod is the one destructive act in this
  # file that a wrong path would make catastrophic on a SHARED store, so it does
  # not inherit the pre-rename answer.
  if ! reap_is_under "$dst" "$REAP_JOB_ROOT"; then
    REAP_REFUSED=$((REAP_REFUSED + 1))
    echo "### REAP REFUSED after rename: $dst is not under $REAP_JOB_ROOT -- no chmod, no rm"
    REAP_HANDED+=("$dst"); return 6
  fi
  errf="$(mktemp "${TMPDIR:-/tmp}/reap-err-XXXXXX")" || {
    REAP_ERRORS=$((REAP_ERRORS + 1))
    echo "### REAP could not open an error file; $dst handed to the cleanup owner unremoved"
    REAP_HANDED+=("$dst"); return 1; }
  # C18's seal strips 0o222 from DIRECTORIES too, and an unwritable directory is
  # what makes `unlink` fail.  `u+w` and nothing else: group and other stay as
  # the seal left them, and `chmod -R` does not traverse symlinks by default.
  chmod -R u+w -- "$dst" 2>>"$errf"
  chmod_err=$(wc -l < "$errf"); chmod_err=${chmod_err:-0}
  rm -rf -- "$dst" 2>>"$errf"
  n=$(wc -l < "$errf"); n=${n:-0}
  rm_err=$((n - chmod_err))
  if [ "$n" -gt 0 ]; then
    sed 's/^/### REAP ERR /' "$errf" | head -20
    [ "$n" -gt 20 ] && echo "### REAP ERR ... $((n - 20)) further error lines suppressed"
  fi
  rm -f "$errf"
  REAP_ERRORS=$((REAP_ERRORS + n))
  if [ -e "$dst" ]; then
    REAP_HANDED+=("$dst")
    echo "### REAP DELETE arm=${label:-none} $dst SURVIVED chmod_err=$chmod_err rm_err=$rm_err errors=$n -- handed to the cleanup owner"
    return 1
  fi
  REAP_DELETED=$((REAP_DELETED + 1))
  echo "### REAP DELETE arm=${label:-none} $dst removed chmod_err=$chmod_err rm_err=$rm_err errors=$n exists_after=no"
  [ "$n" -eq 0 ] || return 1
  return 0
}

# ── THE HANDOFF: what the cleanup owner is owed, as a file it can read ──────
# reap_handoff_list [<path>]   -- default <job root>/REAP-OWED.txt
reap_handoff_list() {
  local out="${1:-$REAP_JOB_ROOT/REAP-OWED.txt}" p
  : > "$out" || { echo "### REAP could not write the handoff list $out"; return 1; }
  for p in ${REAP_HANDED+"${REAP_HANDED[@]}"}; do printf '%s\n' "$p" >> "$out"; done
  echo "### REAP HANDOFF $out rows=${#REAP_HANDED[@]}"
  for p in ${REAP_HANDED+"${REAP_HANDED[@]}"}; do echo "### REAP OWED $p"; done
  return 0
}

# ── RULE 4: THE ACTUATOR ────────────────────────────────────────────────────
# rc 0 clean; rc 1 an rm/chmod reported errors; rc 6 something was refused.
# A driver calls this and lets the rc reach `exit`.
reap_done() {
  echo "### REAP DONE aside=$REAP_ASIDE deleted=$REAP_DELETED owed=${#REAP_HANDED[@]} errors=$REAP_ERRORS refused=$REAP_REFUSED"
  if [ "$REAP_REFUSED" -gt 0 ]; then
    echo "### REAP FATAL: $REAP_REFUSED path(s) refused -- a driver handed this file something that is not its own"
    return 6
  fi
  if [ "$REAP_ERRORS" -gt 0 ]; then
    echo "### REAP FATAL: $REAP_ERRORS error line(s) from chmod/rm -- law 9: this rc must reach Slurm"
    return 1
  fi
  return 0
}
