#!/bin/bash
#SBATCH --job-name=p6-install-v2
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:20:00
#SBATCH --output=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6-inode-cleanup/logs/slurm-%j.out
# Installs delete_stale_roots.v2.sh over delete_stale_roots.sh (keeping a .bak).
# The --dependency=afterany on every v1-invoking sweep job IS the gate: this job
# cannot start until all of them are terminal, so no running bash is reading the
# file being replaced.  A runtime re-check backs the dependency up.
set -uo pipefail
B=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6-inode-cleanup
V1=$B/delete_stale_roots.sh; V2=$B/delete_stale_roots.v2.sh
echo "### install_v2 job=$SLURM_JOB_ID start=$(date -Is) host=$(hostname)"
LIVE=$(squeue -u glvov -h -t RUNNING,PENDING,CONFIGURING,COMPLETING \
       -o "%i %j" | awk '$2 ~ /^p6-(sweep-[0-9]+|fu-)/ {print $1}' | tr '\n' ' ')
if [ -n "$LIVE" ]; then
  echo "REFUSE: v1-invoking sweep jobs still non-terminal: $LIVE"; exit 4; fi
echo "gate OK: no p6-sweep-*/p6-fu-* job is live"
cmp -s "$V1" "$V2" && { echo "already installed (v1 == v2), nothing to do"; exit 0; }
cp -p "$V1" "$V1.bak-preinstall-$(date +%Y%m%dT%H%M%S)" || exit 1
# rename-swap, not truncate-in-place: leaves any open inode untouched
cp -p "$V2" "$V1.new" && mv -f "$V1.new" "$V1" || exit 1
if cmp -s "$V1" "$V2"; then echo "INSTALLED: delete_stale_roots.sh now == v2 ($(date -Is))"; exit 0
else echo "FAILED: post-install compare mismatch"; exit 1; fi
