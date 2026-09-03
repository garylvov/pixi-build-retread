#!/bin/bash
#SBATCH --job-name=p6-d-clean
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=05:00:00
#SBATCH --output=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6-inode-cleanup/logs/slurm-%j.out
set -uo pipefail
B=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6-inode-cleanup
echo "### host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
echo "### inode quota BEFORE:"; /oscar/runtime/bin/checkquota 2>/dev/null | grep 'data+stellex'
ALLOWLIST=$B/allowlist-d.tsv DELETE=1 bash $B/delete_allowlisted.sh
rc=$?
echo "### delete_allowlisted rc=$rc"
echo "### inode quota AFTER:"; /oscar/runtime/bin/checkquota 2>/dev/null | grep 'data+stellex'
exit $rc
