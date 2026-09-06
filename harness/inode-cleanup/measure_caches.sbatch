#!/bin/bash
#SBATCH --job-name=p6-measure
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=03:00:00
#SBATCH --output=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6-inode-cleanup/logs/slurm-%j.out
set -uo pipefail
echo "### measure start $(date -Is) host=$(hostname)"
# HARNESS-EXIT-3 (law 9). This job's whole output is a `du` per root, and a
# `timeout 5400 du` that gives up returns 124 with a SHORT answer, not no
# answer -- so the old shape printed a partial census and reported COMPLETED
# 0:0. The worst rc of the roots is carried to the exit code; a measurement
# that did not finish is a failure that reaches Slurm.
worst=0
for p in /oscar/data/stellex/glvov/caches /oscar/data/stellex/glvov/glvov-envs; do
  echo "=== $p"
  timeout 5400 du --inodes -d1 "$p" 2>/dev/null | sort -k1,1nr
  drc=${PIPESTATUS[0]}
  echo "### DU_RC=$drc root=$p"
  [ "$drc" -gt "$worst" ] && worst=$drc
done
echo "### measure done $(date -Is) worst_du_rc=$worst"
exit "$worst"
