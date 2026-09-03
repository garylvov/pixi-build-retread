#!/bin/bash
# delete_allowlisted.sh -- remove ONLY paths listed byte-for-byte in an explicit
# allowlist file.  Written for p6-inode-cleanup classes B (cargo target/ trees)
# and C (merged, clean git worktrees), which live OUTSIDE
# /oscar/data/stellex/glvov/retread/ and so cannot use cleanup.sh's cert*/ws.*
# pattern guard.
#
# DRY RUN IS THE DEFAULT.  Arm with DELETE=1.
#
#   ALLOWLIST=<file> DELETE=1 bash delete_allowlisted.sh [path ...]
#
# With no path arguments every line of the allowlist is acted on.
# With path arguments, each argument must appear as a WHOLE LINE of the
# allowlist, compared byte-for-byte; anything else is REFUSED and the script
# exits non-zero without touching it.  There is no globbing, no prefix match,
# no realpath normalisation that could smuggle a parent in.
#
# Allowlist line syntax:  <abs-path> TAB <kind>
#   kind=tree      -> chmod -R u+w then rm -rf --one-file-system
#   kind=worktree  -> git -C <parent-repo> worktree remove --force <abs-path>
#                     (parent repo read from the worktree's own .git file)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALLOWLIST="${ALLOWLIST:-$HERE/allowlist.tsv}"
DELETE="${DELETE:-0}"

[ -r "$ALLOWLIST" ] || { echo "FATAL: cannot read allowlist $ALLOWLIST" >&2; exit 2; }

# ---- load the allowlist into an exact-match map -------------------------------
declare -A ALLOWED_KIND
while IFS=$'\t' read -r p kind; do
  case "$p" in ''|'#'*) continue ;; esac
  case "$p" in /*) ;; *) echo "FATAL: allowlist entry not absolute: $p" >&2; exit 2 ;; esac
  ALLOWED_KIND["$p"]="$kind"
done < "$ALLOWLIST"

if [ "$#" -gt 0 ]; then TARGETS=( "$@" ); else
  mapfile -t TARGETS < <(awk -F'\t' '$1 !~ /^#/ && NF {print $1}' "$ALLOWLIST")
fi

if [ "$DELETE" = "1" ]; then echo "### ARMED -- listed paths will be REMOVED ###"
else echo "### DRY RUN -- nothing removed.  Re-run with DELETE=1 to arm. ###"; fi
echo "### allowlist: $ALLOWLIST ($((${#ALLOWED_KIND[@]})) entries)"
echo

rc_all=0; n_del=0; n_refused=0; n_missing=0

for path in "${TARGETS[@]}"; do
  kind="${ALLOWED_KIND[$path]:-}"
  if [ -z "$kind" ]; then
    echo "REFUSE  $path -- not present byte-for-byte in the allowlist"
    n_refused=$((n_refused+1)); rc_all=3; continue
  fi
  if [ -L "$path" ]; then
    echo "REFUSE  $path -- is a symlink"; n_refused=$((n_refused+1)); rc_all=3; continue
  fi
  if [ ! -e "$path" ]; then
    echo "GONE    $path -- already absent"; n_missing=$((n_missing+1)); continue
  fi

  if [ "$DELETE" != "1" ]; then
    echo "WOULD-$kind  $path"; n_del=$((n_del+1)); continue
  fi

  t0=$(date +%s)
  case "$kind" in
    tree)
      echo "RM      $path  start $(date -Is)"
      chmod -R u+w "$path" >/dev/null 2>&1
      rm -rf --one-file-system "$path"; rc=$?
      if [ -e "$path" ]; then
        echo "  RETRY $path"; chmod -R u+w "$path" >/dev/null 2>&1
        rm -rf --one-file-system "$path"; rc=$?
      fi
      ;;
    worktree)
      repo=$(sed -n 's|^gitdir: ||p' "$path/.git" 2>/dev/null)
      repo="${repo%%/.git/worktrees/*}"
      if [ -z "$repo" ] || [ ! -d "$repo/.git" ]; then
        echo "REFUSE  $path -- could not read parent repo from $path/.git"
        n_refused=$((n_refused+1)); rc_all=3; continue
      fi
      echo "WT-RM   $path  (repo $repo) start $(date -Is)"
      git -C "$repo" worktree remove --force "$path"; rc=$?
      if [ -e "$path" ]; then
        echo "  RETRY $path -- worktree remove left it; chmod + rm + prune"
        chmod -R u+w "$path" >/dev/null 2>&1
        rm -rf --one-file-system "$path"; rc=$?
        git -C "$repo" worktree prune
      fi
      ;;
    *) echo "REFUSE  $path -- unknown kind '$kind'"; n_refused=$((n_refused+1)); rc_all=3; continue ;;
  esac
  echo "  DONE  $path rc=$rc wall=$(( $(date +%s) - t0 ))s exists_after=$([ -e "$path" ] && echo YES || echo no) $(date -Is)"
  [ "$rc" = "0" ] || rc_all=1
  n_del=$((n_del+1))
done

echo
echo "---- summary ----"
echo "acted on:  $n_del"
echo "refused:   $n_refused"
echo "absent:    $n_missing"
exit $rc_all
