#!/bin/bash
set -u
B=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6-inode-cleanup
J=${SLURM_JOB_ID:-manual2}
echo "start $(date) host=$(hostname)"
cat > $B/.du2-$J.sh <<'EOF'
#!/bin/bash
p="$1"; t="${2:-5400}"
out=$(timeout "$t" du --inodes -s "$p" 2>/dev/null); rc=$?
if [ $rc -eq 124 ]; then printf 'TIMEOUT\t%s\tTIMEOUT-partial\n' "$p"
elif [ -z "$out" ]; then printf '0\t%s\tunreadable-or-empty\n' "$p"
else printf '%s\t%s\tok\n' "$(echo "$out"|awk '{print $1}')" "$p"; fi
EOF
chmod +x $B/.du2-$J.sh
for T in /oscar/data/stellex/glvov/agrescap; do
  n=$(basename $T)
  : > $B/census-$n-$J.tsv
  find "$T" -mindepth 1 -maxdepth 1 -print0 2>/dev/null | xargs -0 -P 8 -I{} $B/.du2-$J.sh "{}" 5400 >> $B/census-$n-$J.tsv
  sort -k1,1nr $B/census-$n-$J.tsv -o $B/census-$n-$J.tsv
  echo "=== $n done $(date)"; cat $B/census-$n-$J.tsv
done
# second level for agrescap/tasks and agrescap/worktrees etc if they dominate
for sub in tasks worktrees preserved evidence canonical cache trash; do
  d=/oscar/data/stellex/glvov/agrescap/$sub
  [ -d "$d" ] || continue
  echo "=== L2 $sub $(date)"
  find "$d" -mindepth 1 -maxdepth 1 -print0 2>/dev/null | xargs -0 -P 8 -I{} $B/.du2-$J.sh "{}" 3600 >> $B/census-agrescap-l2-$J.tsv
done
sort -k1,1nr $B/census-agrescap-l2-$J.tsv -o $B/census-agrescap-l2-$J.tsv 2>/dev/null
head -40 $B/census-agrescap-l2-$J.tsv
echo "done $(date)"
