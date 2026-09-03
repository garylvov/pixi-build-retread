#!/bin/bash
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --export=NONE
#SBATCH --output=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6-inode-cleanup/logs/slurm-%j.out
# Usage: sbatch f_clean.sbatch <abs-path-to-allowlist.tsv>
#
# ROOT FIX 2026-09-02 21:35 EDT: the first cut passed the allowlist as
# `--export=ALL,F_ALLOWLIST=...`.  That makes Slurm retrieve the submitting
# user's environment on the compute node, which failed for job 5649933 --
# it was requeued HELD with Reason=user_env_retrieval_failed_requeued_held,
# no output, dependency already satisfied.  A positional ARGUMENT needs no
# env retrieval, so the allowlist now arrives as $1 and --export=NONE is
# declared outright.  (Args over env vars is also the standing preference.)
set -uo pipefail
B=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6-inode-cleanup
AL="${1:-${F_ALLOWLIST:-}}"
[ -n "$AL" ] && [ -r "$AL" ] || { echo "FATAL: no readable allowlist (arg1=$AL)"; exit 2; }
echo "### host=$(hostname) job=$SLURM_JOB_ID allowlist=$AL start=$(date -Is)"
echo "### inode quota BEFORE:"; /oscar/runtime/bin/checkquota 2>/dev/null | grep 'data+stellex'
ALLOWLIST="$AL" DELETE=1 bash $B/delete_allowlisted.sh
rc=$?
echo "### delete_allowlisted rc=$rc"
echo "### inode quota AFTER:"; /oscar/runtime/bin/checkquota 2>/dev/null | grep 'data+stellex'
exit $rc
