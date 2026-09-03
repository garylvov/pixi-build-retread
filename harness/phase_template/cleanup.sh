#!/usr/bin/env bash
# cleanup.sh -- remove job-scoped retread roots from a job of its OWN, so no
# measured phase ever pays for an NFS `rm -rf`.
#
# WHY THIS FILE EXISTS. Unlinking a job-scoped cache root here is slow and it is
# the slowness of the filesystem, not of a wedge: 5152s on job 5596128 (against
# a 3679s lock, while holding 160G of a 492G per-user QOS), and 4795s + 2552s
# for the two roots of job 5598763. As an in-job epilogue that time sits on the
# afterok critical path and blocks the successor. As a separate 1-CPU/4G job it
# costs nothing anybody is waiting for.
#
# USAGE (the cert phase submits this for you):
#     env -u SLURM_JOB_ID sbatch --partition=batch --qos=normal \
#         --cpus-per-task=1 --mem=4G --time=06:00:00 \
#         --dependency=afterany:<cert job> cleanup.sh <root> [<root> ...]
#
# REFUSAL. Only paths under /oscar/data/stellex/glvov/retread/ whose basename
# starts with `cert` or `ws.` are removed. Anything else is printed and skipped.
# The canonical source tree, the persistent cache under agrescap/cache/retread,
# and every artifacts/ directory are outside that pattern by construction.
#
# The persistent cache is NEVER touched here. It is what makes the next relock
# 69s instead of 2865s (job 5598763). It is rebuildable, so deleting it is
# always safe and only ever slow -- but that is a deliberate operator act:
#     rm -rf /oscar/data/stellex/glvov/agrescap/cache/retread
set -uo pipefail

CQ=/oscar/runtime/bin/checkquota          # NOT on a batch job's default PATH: job 5611846 printed
[ -x "$CQ" ] || CQ=$(command -v checkquota 2>/dev/null || echo true)   # two EMPTY quota rows because of it
ALLOWED_PREFIX=/oscar/data/stellex/glvov/retread

hostname; date -Is
echo "### CLEANUP job=${SLURM_JOB_ID:-none} roots=$#"
if [ "$#" = 0 ]; then echo "### nothing to do (no roots passed)"; exit 0; fi
echo "### inode quota BEFORE:"; "$CQ" 2>/dev/null | grep -E 'data\+stellex|^Name' | head -4

RC=0
for r in "$@"; do
  [ -n "$r" ] || continue
  base=$(basename "$r")
  case "$r" in
    "$ALLOWED_PREFIX"/*) ;;
    *) echo "### REFUSED (outside $ALLOWED_PREFIX): $r"; RC=1; continue;;
  esac
  case "$base" in
    cert*|ws.*) ;;
    *) echo "### REFUSED (basename is not cert*/ws.*): $r"; RC=1; continue;;
  esac
  case "$r" in
    */../*|*/..) echo "### REFUSED (path traversal): $r"; RC=1; continue;;
  esac
  if [ ! -e "$r" ]; then echo "### already gone: $r"; continue; fi
  N=$(find "$r" 2>/dev/null | wc -l)
  echo "### removing $r  (entries=$N) start $(date -Is)"
  S=$(date +%s)
  chmod -R u+w "$r" >/dev/null 2>&1
  rm -rf "$r"
  echo "### removed $r rc=$? wall=$(( $(date +%s) - S ))s exists_after=$([ -e "$r" ] && echo YES || echo no) $(date -Is)"
done

echo "### inode quota AFTER:"; "$CQ" 2>/dev/null | grep -E 'data\+stellex' | head -2
echo "### CLEANUP DONE rc=$RC $(date -Is)"
exit "$RC"
