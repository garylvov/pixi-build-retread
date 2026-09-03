#!/usr/bin/env bash
# Reader/writer guard for the wheel store (LANE-C-WARM-LOG 8.6 / 9.4).
#
# The WRITER is tools/retread_fast_env.sh, which exports RETREAD_WHEEL_STORE.
# The READER is the backend's `courier::wheel_store_root_with`. They live in
# different languages and different repos-worth of file, and the last time
# nobody checked the pair, the recorded reason for 0 % index coverage named two
# environment variables (RETREAD_BUILD_ROOT, RETREAD_ARTIFACT_ROOT) that do not
# exist in the backend at all.
#
# So this asserts BOTH halves against the real files:
#   1. sourcing the real retread_fast_env.sh exports RETREAD_WHEEL_STORE,
#      it is <persist cache root>/wheels, and the directory exists;
#   2. the real courier.rs returns that value VERBATIM -- no join. If the
#      backend ever starts appending "retread/wheels" to the override the way
#      it does to XDG_CACHE_HOME, the harness would silently point at
#      <root>/wheels/retread/wheels and coverage would go back to ~2 %.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${1:-/oscar/data/stellex/glvov/agrescap/worktrees/merge-4-12-c/src/courier.rs}
# retread_fast_env allowlists its cache root, so the guard must run under a
# real subdirectory of that root rather than /tmp.
W=$(mktemp -d /oscar/data/stellex/glvov/agrescap/cache/retread-guard.XXXXXX)
# ...and the allowlist prefix-matches "…/cache/retread", so name it accordingly.
ROOT=$W/retread; mkdir -p "$ROOT"
trap 'rm -rf "$W"' EXIT
FAIL=0
say() { printf '%s\n' "$*"; }
bad() { say "  FAIL  $*"; FAIL=1; }

# --- half 1: the WRITER -------------------------------------------------------
OUT=$(
  set -u
  export RETREAD_PERSIST_CACHE_ROOT="$ROOT/persist"
  unset RETREAD_WHEEL_STORE RETREAD_FAST_TMP_ROOT SLURM_JOB_ID 2>/dev/null || true
  # shellcheck disable=SC1090
  . "$HERE/retread_fast_env.sh" >/dev/null 2>&1
  retread_fast_env "$W/ws" >/dev/null 2>&1
  printf '%s\n' "${RETREAD_WHEEL_STORE:-<UNSET>}"
)
say "writer exported RETREAD_WHEEL_STORE=$OUT"
[ "$OUT" = "$ROOT/persist/wheels" ] || bad "writer must export <persist root>/wheels, got [$OUT]"
[ -d "$ROOT/persist/wheels" ] || bad "writer must CREATE the store directory"
# it must be EXPORTED, not merely set: the backend is a child process.
EXPORTED=$(
  export RETREAD_PERSIST_CACHE_ROOT="$ROOT/persist2"
  unset RETREAD_WHEEL_STORE
  # shellcheck disable=SC1090
  . "$HERE/retread_fast_env.sh" >/dev/null 2>&1
  retread_fast_env "$W/ws" >/dev/null 2>&1
  env | grep -c '^RETREAD_WHEEL_STORE='
)
[ "$EXPORTED" = 1 ] || bad "RETREAD_WHEEL_STORE must reach a CHILD's environment (env| grep = $EXPORTED)"

# --- half 2: the READER -------------------------------------------------------
[ -f "$SRC" ] || { say "  SKIP  reader half: $SRC not readable"; [ "$FAIL" = 0 ] && exit 0 || exit 1; }
BRANCH=$(awk '/fn wheel_store_root_with/{f=1} f{print} f&&/^}/{exit}' "$SRC" |
         awk '/RETREAD_WHEEL_STORE/{f=1} f{print} f&&/}/{exit}')
printf '%s\n' "$BRANCH" | grep -q 'return std::path::PathBuf::from(dir);' \
  || bad "the reader no longer returns RETREAD_WHEEL_STORE verbatim -- the writer's value would be rewritten. Branch was:
$BRANCH"
printf '%s\n' "$BRANCH" | grep -q '\.join(' \
  && bad "the reader now JOINS onto RETREAD_WHEEL_STORE; the harness would point at <root>/wheels/retread/wheels"

[ "$FAIL" = 0 ] && { say "wheel-store reader/writer guard: ALL PASS"; exit 0; }
say "wheel-store reader/writer guard: FAILED"; exit 1
