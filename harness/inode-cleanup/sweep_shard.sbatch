#!/bin/bash
# sweep_shard.sbatch <shard.tsv>
# One independent 1-CPU/4G cleanup job for ~5 stale retread roots.
# Deletion is done by delete_stale_roots.sh, which re-runs a LIVE sacct/squeue
# preflight per root and refuses anything that is not a plain cert*/ws.* child
# of /oscar/data/stellex/glvov/retread.
set -uo pipefail
HERE=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6-inode-cleanup
CQ=/oscar/runtime/bin/checkquota
SHARD="$1"
RECENT="${2:-7200}"   # optional: shrink the mtime guard when the owning job is proven terminal
hostname; date -Is
echo "### SWEEP job=${SLURM_JOB_ID:-none} shard=$SHARD"
echo "### inode quota BEFORE:"; "$CQ" 2>/dev/null | grep -E 'data\+stellex' | head -2
INVENTORY="$HERE/$SHARD" DELETE=1 RECENT_SECS="$RECENT" bash "$HERE/${SWEEP_SCRIPT:-delete_stale_roots.v2.sh}"
RC=$?
echo "### inode quota AFTER:"; "$CQ" 2>/dev/null | grep -E 'data\+stellex' | head -2
echo "### SWEEP DONE rc=$RC $(date -Is)"
exit $RC
