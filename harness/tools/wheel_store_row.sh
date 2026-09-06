#!/usr/bin/env bash
# wheel_store_row.sh -- print the ONE row a relock harness owes about its wheel
# store: the path it RESOLVED, where that path came from, and whether that is
# shared or job-scoped.  MERGE-M-1.
#
# WHY THIS EXISTS.  Every merge-proof harness on this campaign printed this row
# by hand:
#
#   ### WHEEL STORE: job-scoped, under XDG_CACHE_HOME=… (RETREAD_WHEEL_STORE unset: /oscar/…/agrescap/cache/retread/wheels)
#
# and the parenthesis is the `${RETREAD_WHEEL_STORE:-<unset>}` expansion, so it
# PRINTED A VALUE -- the variable was set, the store was the SHARED one, and the
# row's own prose called it job-scoped and called the variable unset.  The block
# comment above it said `tools/retread_fast_env.sh` keeps the shared export
# COMMENTED OUT; that export was RE-ENABLED on 2026-09-03 15:35 when p6i merged
# (`export RETREAD_WHEEL_STORE=$root/wheels`, and the comment block right above
# it says so at length) and the row was never updated.  B18, B19, B20 and B21
# all printed the identical wrong row, which is why the four proofs stayed
# comparable with each other and why nothing was re-run for it -- but a reader
# that states a conclusion its own value contradicts is the defect class this
# campaign keeps paying for.  MERGE-M-1.
#
# THE FIX IS NOT A BETTER SENTENCE, IT IS A DERIVED ONE: this file RESOLVES the
# store the way the product does and reports the resolution, its PROVENANCE and
# its SCOPE, and it never asserts a scope it was not given the means to decide.
#
# The resolution mirrors `courier::wheel_store_root_with` exactly:
#   1. RETREAD_WHEEL_STORE, non-empty after trimming  -> that path, USED WHOLE
#      (the product does not join "retread/wheels" onto it).
#   2. XDG_CACHE_HOME -> $XDG_CACHE_HOME/retread/wheels
#   3. HOME           -> $HOME/.cache/retread/wheels
#   4. neither        -> the product falls back to a temp dir; this reader says
#      so and refuses to name a path it cannot derive.
#
#   usage: wheel_store_row.sh [job-scope root]
#
# The optional argument is the directory that IS this job's scope -- $C in every
# harness of this campaign.  Given it, the row reports scope=job-scoped when the
# resolved store lives under it and scope=shared when it does not.  Without it
# the row reports scope=undeclared, which is honest: no harness can tell the two
# apart from the path alone, and guessing is how MERGE-M-1 happened.
#
#   rc 0  the row was printed
#   rc 2  the store could not be derived at all (no RETREAD_WHEEL_STORE, no
#         XDG_CACHE_HOME, no HOME) -- SETUP FAILURE, never a verdict
set -uo pipefail
JOB_ROOT="${1:-}"

RWS="${RETREAD_WHEEL_STORE:-}"
XDG="${XDG_CACHE_HOME:-}"
HOMEDIR="${HOME:-}"

# trim, the way `.filter(|s| !s.trim().is_empty())` does
rws_trimmed="${RWS#"${RWS%%[![:space:]]*}"}"
rws_trimmed="${rws_trimmed%"${rws_trimmed##*[![:space:]]}"}"

if [ -n "$rws_trimmed" ]; then
  resolved="$rws_trimmed"; provenance="RETREAD_WHEEL_STORE"
elif [ -n "$XDG" ]; then
  resolved="$XDG/retread/wheels"; provenance="XDG_CACHE_HOME"
elif [ -n "$HOMEDIR" ]; then
  resolved="$HOMEDIR/.cache/retread/wheels"; provenance="HOME"
else
  echo "### WHEEL STORE: UNDERIVABLE -- RETREAD_WHEEL_STORE, XDG_CACHE_HOME and HOME are all unset; the product would fall back to a temp dir" >&2
  exit 2
fi

# canonicalise for the containment test only; the row prints the path as derived
scope="undeclared"
if [ -n "$JOB_ROOT" ]; then
  jr=$(cd -- "$JOB_ROOT" 2>/dev/null && pwd -P) || jr="$JOB_ROOT"
  case "$resolved/" in
    "$jr"/*) scope="job-scoped" ;;
    *)       scope="shared" ;;
  esac
fi

exists="absent"
[ -d "$resolved" ] && exists="present"

printf '### WHEEL STORE: resolved=%s provenance=%s scope=%s exists=%s\n' \
  "$resolved" "$provenance" "$scope" "$exists"
printf '###   inputs: RETREAD_WHEEL_STORE=%s XDG_CACHE_HOME=%s HOME=%s job_scope_root=%s\n' \
  "${RWS:-(unset)}" "${XDG:-(unset)}" "${HOMEDIR:-(unset)}" "${JOB_ROOT:-(not declared)}"
