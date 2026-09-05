#!/usr/bin/env bash
# harness_drift_check.sh -- REFUSE to run on a task-dir harness that is not the
# named commit.  C31-4-1d.
#
# WHY THIS EXISTS.  `harness/` is the versioned home; the task tree
# `/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/` is a NON-git working
# copy that jobs actually execute (harness/README.md, CLAUDE.md law 7).  C31-4-1
# reconciled the two by hand and found four task copies silently BEHIND the
# repo -- `retread_fast_env.sh` was 381 lines against the repo's 1160.  That
# reconciliation was a SNAPSHOT, not a link: nothing detected the divergence
# while it existed and nothing would detect the next one.  This script is the
# link.  Wired into the phase-template header behind $HARNESS_COMMIT, a stale
# task copy REFUSES instead of running for three hours and certifying the wrong
# harness.
#
#   usage: harness_drift_check.sh <commit-ish> [task-dir] [repo] [allowlist]
#
#   env overrides (argv wins):
#     HARNESS_TASK_DIR   default /oscar/data/stellex/glvov/agrescap/tasks/retread-4-11
#     HARNESS_REPO       default /oscar/data/stellex/glvov/agrescap/worktrees/harness-tools
#     HARNESS_DRIFT_ALLOWLIST  default harness_drift_allowlist.txt beside this script
#
#   rc 0  every mapped task file matches the commit (or is allowlisted)
#   rc 3  DRIFT: at least one file differs from the commit or has no blob there
#   rc 4  FATAL: no repo, no such commit, no allowlist file
#
# EVERY COMPARISON IS AGAINST `git cat-file blob <commit>:<path>`, NEVER against
# the repo CHECKOUT.  The worktree is edited by concurrent lanes; reading the
# checkout would compare the task copy against somebody's half-written file.
# Blobs are written to a temp file and md5summed with a DIRECT FILE ARGUMENT --
# a piped `md5sum` is not trustworthy in this environment.
#
# THE ALLOWLIST IS LOAD-BEARING, so it is a file with reasons in it and not a
# constant in this script.  It names the task files a lane DELIBERATELY left
# unsynced (a live process holds one, say).  Anything not named there must
# match.  Its own fixture proves it: forcing the allowlist to cover everything
# turns the RED case green.
set -uo pipefail

COMMIT="${1:?usage: harness_drift_check.sh <commit-ish> [task-dir] [repo] [allowlist]}"
TASK_DIR="${2:-${HARNESS_TASK_DIR:-/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11}}"
REPO="${3:-${HARNESS_REPO:-/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools}}"
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ALLOWLIST="${4:-${HARNESS_DRIFT_ALLOWLIST:-$SELF_DIR/harness_drift_allowlist.txt}}"

[ -d "$TASK_DIR" ] || { echo "DRIFT FATAL: no such task dir $TASK_DIR" >&2; exit 4; }
[ -d "$REPO" ]     || { echo "DRIFT FATAL: no such repo $REPO" >&2; exit 4; }
[ -f "$ALLOWLIST" ] || { echo "DRIFT FATAL: no allowlist at $ALLOWLIST" >&2; exit 4; }

SHA="$(git -C "$REPO" rev-parse --verify "${COMMIT}^{commit}" 2>/dev/null)" || {
  echo "DRIFT FATAL: $COMMIT is not a commit in $REPO" >&2; exit 4; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/harness_drift.XXXXXX")" || {
  echo "DRIFT FATAL: could not make a temp dir" >&2; exit 4; }
trap 'rm -rf "$TMP"' EXIT

echo "### harness drift check: task=$TASK_DIR repo=$REPO commit=$SHA"
echo "### allowlist: $ALLOWLIST"

# ---- the allowlist, as task-relative paths -------------------------------
ALLOWED=""
while read -r p rest; do
  case "$p" in ''|\#*) continue;; esac
  ALLOWED="$ALLOWED $p"
done < "$ALLOWLIST"

# ---- the mapped set ------------------------------------------------------
# task tools/<f>                -> harness/tools/<f>
# task tools/phase_template/<f> -> harness/phase_template/<f>
# task merge-h/<f>              -> by basename, searched phase_template, arms,
#                                  tools (the §34 basename rule: merge-h's
#                                  cleanup_gated.sh is the phase_template file
#                                  and its gate_build.sh is the tools file).
# `.bak-*`, `.pre-*` and __pycache__ are run evidence, not harness, and are
# excluded here exactly as harness/README.md excludes them from the repo.
map_of () {
  case "$1" in
    tools/phase_template/*) echo "harness/phase_template/${1#tools/phase_template/}"; return;;
    tools/*)                echo "harness/tools/${1#tools/}"; return;;
    merge-h/*)
      local b=${1#merge-h/} d
      for d in phase_template arms tools; do
        if git -C "$REPO" cat-file -e "$SHA:harness/$d/$b" 2>/dev/null; then
          echo "harness/$d/$b"; return
        fi
      done
      echo "harness/tools/$b"; return;;
  esac
  echo ""
}

checked=0; okc=0; allowc=0; mismatch=0; missing=0
BAD="$TMP/bad.txt"; : > "$BAD"

scan_dir () {  # $1 = task-relative dir
  local rel="$1" abs="$TASK_DIR/$1" f base trel wpath tmd5 bmd5
  [ -d "$abs" ] || return 0
  while IFS= read -r f; do
    base=$(basename "$f")
    case "$base" in *.bak-*|*.pre-*|*.pyc|*~) continue;; esac
    trel="$rel/$base"
    case " $ALLOWED " in
      *" $trel "*) echo "DRIFT allow    $trel"; allowc=$((allowc + 1)); continue;;
    esac
    checked=$((checked + 1))
    wpath=$(map_of "$trel")
    if [ -z "$wpath" ]; then
      echo "DRIFT MISSING  $trel  (no mapping rule)"; missing=$((missing + 1))
      echo "$trel  no-mapping" >> "$BAD"; continue
    fi
    if ! git -C "$REPO" cat-file blob "$SHA:$wpath" > "$TMP/blob" 2>/dev/null; then
      echo "DRIFT MISSING  $trel  (no blob at $SHA:$wpath)"; missing=$((missing + 1))
      echo "$trel  no-blob:$wpath" >> "$BAD"; continue
    fi
    tmd5=$(md5sum "$f"        | awk '{print $1}')
    bmd5=$(md5sum "$TMP/blob" | awk '{print $1}')
    if [ "$tmd5" = "$bmd5" ]; then
      echo "DRIFT ok       $trel  $tmd5"; okc=$((okc + 1))
    else
      echo "DRIFT MISMATCH $trel  task=$tmd5 $wpath=$bmd5"; mismatch=$((mismatch + 1))
      echo "$trel  mismatch:$wpath" >> "$BAD"
    fi
  done < <(find "$abs" -maxdepth 1 -type f | sort)
}

scan_dir tools
scan_dir tools/phase_template
scan_dir merge-h

echo "### DRIFT SUMMARY commit=$SHA checked=$checked ok=$okc allowed=$allowc mismatch=$mismatch missing=$missing"

if [ "$mismatch" -gt 0 ] || [ "$missing" -gt 0 ]; then
  echo "### DRIFT REFUSED -- the task-dir harness is NOT $SHA. Offending files:"
  cat "$BAD"
  echo "### Fix it from the commit, not from the checkout:"
  echo "###   git -C $REPO cat-file blob $SHA:<repo path> > $TASK_DIR/<task path>"
  echo "### or, if the divergence is deliberate, name the file in $ALLOWLIST with a reason."
  exit 3
fi
echo "### DRIFT CLEAN -- the task-dir harness is $SHA"
exit 0
