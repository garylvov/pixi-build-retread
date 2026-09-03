#!/bin/bash
# Read-only inode census. Deletes nothing.
set -u
B=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6-inode-cleanup
J=${SLURM_JOB_ID:-manual}
R=/oscar/data/stellex/glvov

echo "=== census start $(date) job=$J host=$(hostname)"
echo "=== checkquota"
/oscar/runtime/bin/checkquota 2>&1 | tee $B/census-quota-$J.txt
echo "=== (a) done"

# helper: du --inodes -s one path with timeout
cat > $B/.du1-$J.sh <<'EOF'
#!/bin/bash
p="$1"; t="${2:-2400}"
out=$(timeout "$t" du --inodes -s "$p" 2>/dev/null)
rc=$?
if [ $rc -eq 124 ]; then
  # partial: count what we can quickly via find with a cap? report as TIMEOUT with unknown
  printf '%s\t%s\t%s\n' "TIMEOUT" "$p" "TIMEOUT-partial"
elif [ -z "$out" ]; then
  printf '%s\t%s\t%s\n' "0" "$p" "unreadable-or-empty"
else
  printf '%s\t%s\t%s\n' "$(echo "$out" | awk '{print $1}')" "$p" "ok"
fi
EOF
chmod +x $B/.du1-$J.sh

echo "=== (b) top-level census of $R"
: > $B/census-top-$J.tsv
find "$R" -mindepth 1 -maxdepth 1 -print0 2>/dev/null \
  | xargs -0 -P 8 -I{} $B/.du1-$J.sh "{}" 2400 >> $B/census-top-$J.tsv
sort -k1,1nr $B/census-top-$J.tsv -o $B/census-top-$J.tsv
echo "top-level done $(date)"
cat $B/census-top-$J.tsv

echo "=== (c) level-2 for top 6"
: > $B/census-l2-$J.tsv
for e in $(awk -F'\t' '$1 ~ /^[0-9]+$/ {print $2}' $B/census-top-$J.tsv | head -6); do
  echo "--- L2 $e" >> $B/census-l2-$J.tsv
  timeout 1800 du --inodes -d1 "$e" 2>/dev/null | sort -k1,1nr >> $B/census-l2-$J.tsv \
    || echo -e "TIMEOUT\t$e" >> $B/census-l2-$J.tsv
done
echo "l2 done $(date)"

echo "=== (d) retread per-root"
: > $B/census-retread-$J.tsv
if [ -d "$R/retread" ]; then
  find "$R/retread" -mindepth 1 -maxdepth 1 -print0 2>/dev/null \
    | xargs -0 -P 8 -I{} $B/.du1-$J.sh "{}" 1200 > $B/.retread-raw-$J.tsv
  # annotate with embedded job id + sacct state
  while IFS=$'\t' read -r n p s; do
    base=$(basename "$p")
    jid=$(echo "$base" | grep -oE '[0-9]{6,9}' | tail -1)
    st="-"
    if [ -n "$jid" ]; then
      st=$(sacct -X -n -j "$jid" -o State 2>/dev/null | head -1 | tr -d ' ')
      [ -z "$st" ] && st="NOT-IN-SACCT"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$n" "$p" "$s" "${jid:--}" "$st"
  done < $B/.retread-raw-$J.tsv | sort -k1,1nr > $B/census-retread-$J.tsv
  echo "retread roots: $(wc -l < $B/census-retread-$J.tsv) snapshot $(date)"
fi

echo "=== (e) group share: /oscar/data/stellex/*"
: > $B/census-group-$J.tsv
find /oscar/data/stellex -mindepth 1 -maxdepth 1 -print0 2>/dev/null \
  | xargs -0 -P 8 -I{} $B/.du1-$J.sh "{}" 1800 >> $B/census-group-$J.tsv
sort -k1,1nr $B/census-group-$J.tsv -o $B/census-group-$J.tsv
cat $B/census-group-$J.tsv

echo "=== final checkquota"
/oscar/runtime/bin/checkquota 2>&1 | tail -20
echo "=== census done $(date)"
