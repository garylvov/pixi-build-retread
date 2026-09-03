#!/bin/bash
#SBATCH --job-name=p6-measure
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=03:00:00
#SBATCH --output=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6-inode-cleanup/logs/slurm-%j.out
echo "### measure start $(date -Is) host=$(hostname)"
for p in /oscar/data/stellex/glvov/caches /oscar/data/stellex/glvov/glvov-envs; do
  echo "=== $p"; timeout 5400 du --inodes -d1 "$p" 2>/dev/null | sort -k1,1nr
done
echo "### measure done $(date -Is)"
