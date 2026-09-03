#!/bin/bash
set -u
B=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6-inode-cleanup
J=${SLURM_JOB_ID:-manual3}
echo "start $(date) host=$(hostname)"
cat > $B/.du3-$J.sh <<'EOF'
#!/bin/bash
p="$1"; t="${2:-2700}"
out=$(timeout "$t" du --inodes -s "$p" 2>/dev/null); rc=$?
if [ $rc -eq 124 ]; then printf 'TIMEOUT\t%s\tTIMEOUT-partial\n' "$p"
elif [ -z "$out" ]; then printf '0\t%s\tunreadable-or-empty\n' "$p"
else printf '%s\t%s\tok\n' "$(echo "$out"|awk '{print $1}')" "$p"; fi
EOF
chmod +x $B/.du3-$J.sh

echo "=== A: retread per-root  $(date)"
find /oscar/data/stellex/glvov/retread -mindepth 1 -maxdepth 1 -print0 2>/dev/null \
  | xargs -0 -P 4 -I{} $B/.du3-$J.sh "{}" 1200 > $B/.rr-$J.tsv
while IFS=$'\t' read -r n p s; do
  base=$(basename "$p"); jid=$(echo "$base" | grep -oE '[0-9]{6,9}' | tail -1); st="-"
  if [ -n "$jid" ]; then st=$(sacct -X -n -j "$jid" -o State 2>/dev/null | head -1 | tr -d ' '); [ -z "$st" ] && st="NOT-IN-SACCT"; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$n" "$p" "$s" "${jid:--}" "$st"
done < $B/.rr-$J.tsv | sort -k1,1nr > $B/census-retread-$J.tsv
echo "retread roots: $(wc -l < $B/census-retread-$J.tsv)  snapshot $(date)"
awk -F'\t' '$1~/^[0-9]+$/{t+=$1} END{print "retread numeric total: "t}' $B/census-retread-$J.tsv
head -40 $B/census-retread-$J.tsv

echo "=== B: agrescap per-child  $(date)"
find /oscar/data/stellex/glvov/agrescap -mindepth 1 -maxdepth 1 -print0 2>/dev/null \
  | xargs -0 -P 4 -I{} $B/.du3-$J.sh "{}" 2700 > $B/census-agrescap-$J.tsv
sort -k1,1nr $B/census-agrescap-$J.tsv -o $B/census-agrescap-$J.tsv
cat $B/census-agrescap-$J.tsv

echo "=== C: group share /oscar/data/stellex/*  $(date)"
find /oscar/data/stellex -mindepth 1 -maxdepth 1 -print0 2>/dev/null \
  | xargs -0 -P 4 -I{} $B/.du3-$J.sh "{}" 1800 > $B/census-group-$J.tsv
sort -k1,1nr $B/census-group-$J.tsv -o $B/census-group-$J.tsv
cat $B/census-group-$J.tsv
echo "=== done $(date)"
/oscar/runtime/bin/checkquota 2>&1 | grep -E 'data\+stellex|Used_Inodes'
