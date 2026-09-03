#!/bin/bash
# delete_stale_roots.sh -- reclaim inodes from unreclaimed job-scoped scratch roots
# under /oscar/data/stellex/glvov/retread/.
#
# DRY-RUN IS THE DEFAULT. Nothing is removed unless you arm it:
#     DELETE=1 bash delete_stale_roots.sh
#
# Safety properties:
#   * Operates ONLY on roots classified DELETABLE in inventory.tsv (same directory).
#   * Re-runs a live sacct/squeue preflight at execution time for every embedded job
#     id and SKIPS any root whose job is not in a terminal state.  The inventory is a
#     snapshot; this preflight is the authority.
#   * Refuses any path that is not a direct child of $RETREAD (no traversal, no globs).
#   * rm -rf --one-file-system, so a stray mount inside a root cannot be crossed.
#   * Never touches the task-dir artifacts (which live outside /retread/) or the
#     legacy retread-product-* / retread-recovery trees.
#
# Env knobs:
#   DELETE=1       arm real deletion (default 0 = dry run)
#   HOLD="a b c"   extra space-separated root names to skip regardless of class
#   RECENT_SECS=N  skip roots modified within N seconds (default 7200 = 2h)
#   INVENTORY=P    read a different inventory/shard TSV (same 7-column schema)

set -uo pipefail

RETREAD=/oscar/data/stellex/glvov/retread
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INVENTORY="${INVENTORY:-$HERE/inventory.tsv}"   # override to run one shard of the sweep

DELETE="${DELETE:-0}"
HOLD="${HOLD:-}"
RECENT_SECS="${RECENT_SECS:-7200}"

# Job states that mean "finished, safe to reclaim".
TERMINAL_RE='^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT)'

[ -r "$INVENTORY" ] || { echo "FATAL: cannot read $INVENTORY" >&2; exit 1; }

if [ "$DELETE" = "1" ]; then
  echo "### ARMED -- roots will be REMOVED ###"
else
  echo "### DRY RUN -- nothing will be removed.  Re-run with DELETE=1 to arm. ###"
fi
echo

# ---------------------------------------------------------------- preflight ---
# One sacct call for every embedded job id, plus squeue for anything still queued.
mapfile -t IDS < <(awk -F'\t' 'NR>1 && $6=="DELETABLE" && $2!="-" {print $2}' "$INVENTORY" | sort -u)

declare -A JOBSTATE
if [ "${#IDS[@]}" -gt 0 ]; then
  IDLIST=$(printf '%s,' "${IDS[@]}"); IDLIST="${IDLIST%,}"
  while IFS='|' read -r j s; do
    [ -n "$j" ] && JOBSTATE["$j"]="${s%% *}"
  done < <(sacct -j "$IDLIST" -X --format=JobID,State -n -P 2>/dev/null)
fi

# squeue overrides sacct: anything currently queued/running is live, full stop.
while read -r j s; do
  [ -n "$j" ] && JOBSTATE["$j"]="$s"
done < <(squeue -u glvov -h -o "%i %T" 2>/dev/null)

NOW=$(date +%s)
n_del=0; n_skip=0; n_missing=0

while IFS=$'\t' read -r root jid state mtime inodes class reason; do
  [ "$class" = "DELETABLE" ] || continue

  # --- hold list ---
  for h in $HOLD; do
    if [ "$h" = "$root" ]; then
      echo "SKIP  $root -- on HOLD list"; n_skip=$((n_skip+1)); continue 2
    fi
  done

  # --- path sanity: must be a plain direct child of RETREAD ---
  case "$root" in
    */*|.|..|"") echo "SKIP  $root -- not a plain child name"; n_skip=$((n_skip+1)); continue ;;
  esac
  # --- pattern guard: only job-scoped cert*/ws.* roots, same rule as cleanup.sh ---
  case "$root" in
    cert*|ws.*) ;;
    *) echo "SKIP  $root -- basename is not cert*/ws.*"; n_skip=$((n_skip+1)); continue ;;
  esac
  path="$RETREAD/$root"
  if [ ! -d "$path" ]; then
    echo "GONE  $path -- already absent"; n_missing=$((n_missing+1)); continue
  fi
  if [ -L "$path" ]; then
    echo "SKIP  $path -- is a symlink"; n_skip=$((n_skip+1)); continue
  fi

  # --- live-job preflight ---
  if [ "$jid" != "-" ]; then
    live="${JOBSTATE[$jid]:-}"
    if [ -z "$live" ]; then
      echo "SKIP  $root -- job $jid not found by sacct/squeue (fail closed)"
      n_skip=$((n_skip+1)); continue
    fi
    if ! [[ "$live" =~ $TERMINAL_RE ]]; then
      echo "SKIP  $root -- job $jid is $live (non-terminal)"
      n_skip=$((n_skip+1)); continue
    fi
  fi

  # --- recency preflight ---
  mt=$(stat -c %Y "$path" 2>/dev/null || echo 0)
  if [ "$((NOW - mt))" -lt "$RECENT_SECS" ]; then
    echo "SKIP  $root -- modified $(( (NOW-mt)/60 ))m ago (< ${RECENT_SECS}s)"
    n_skip=$((n_skip+1)); continue
  fi

  # --- act ---
  if [ "$DELETE" = "1" ]; then
    echo "RM    $path  (job $jid $state, ~$inodes inodes) start $(date -Is)"
    t0=$(date +%s)
    # Materialized git checkouts inside these roots contain read-only DIRECTORIES,
    # and rm cannot unlink a file whose parent directory is not writable: the v1
    # script left a ~14k-entry stub per root with rc=1 (jobs 5631958-5631963,
    # 2026-09-02).  cleanup.sh always had this chmod; this script did not.
    chmod -R u+w "$path" >/dev/null 2>&1
    rm -rf --one-file-system "$path"; rc=$?
    if [ -e "$path" ]; then
      echo "  RETRY $path -- still present after first rm; re-chmod and re-rm"
      chmod -R u+w "$path" >/dev/null 2>&1
      rm -rf --one-file-system "$path"; rc=$?
    fi
    echo "  DONE  $path rc=$rc wall=$(( $(date +%s) - t0 ))s exists_after=$([ -e "$path" ] && echo YES || echo no) $(date -Is)"
  else
    echo "WOULD-RM $path  (job $jid $state, ~$inodes inodes)"
  fi
  n_del=$((n_del+1))
done < <(tail -n +2 "$INVENTORY")

echo
echo "---- summary ----"
echo "deleted (or would delete): $n_del"
echo "skipped by preflight:      $n_skip"
echo "already absent:            $n_missing"
[ "$DELETE" = "1" ] || echo "(dry run -- re-run with DELETE=1 to actually reclaim)"
