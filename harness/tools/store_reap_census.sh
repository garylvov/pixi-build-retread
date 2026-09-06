#!/usr/bin/env bash
# store_reap_census.sh -- THE HARNESS HALF OF `retread store-reap`.  STORE-REAP-2.
#
# WHAT IT IS FOR (law 2, and the debt it closes).  STORE-REAP-1 boarded
# STORE-REAP-1-1: the persistent-store reapers (C18-1 git snapshots, L3-1b
# shadow, L3-1b-1 built wheels, and -- since L3-1b-3B -- build-requirements and
# strict-wheel-attestations, the last two walked by ONE reaper because they are
# the same `<root>/<dir>/<version>/<identity>/<marker>` shape) have exactly one
# production caller --
# the backend's `initialize` -- and every relock this harness runs job-scopes
# `XDG_CACHE_HOME` to `$C/xdg-cache`, so the root that caller resolves is a
# fresh empty directory the job's own cleanup deletes.  The reapers therefore
# print `reason="store-absent"` on day 1 and on day 14 alike, and the over-age
# entries in the REAL persistent roots have no process that will ever open
# them.  `retread store-reap` is the missing call site; THIS file is the
# missing reader, so every lane job carries the census in its own log instead
# of the number existing only in a lane report nobody re-reads.
#
#   usage:  BACKEND=<retread binary> bash store_reap_census.sh <when>
#
#   env:
#     BACKEND            REQUIRED, the retread binary this job is running
#     STORE_REAP_ROOTS   space-separated roots to census.  Default below.
#     STORE_REAP_BYTES   set to 1 to charge bytes (a FULL walk of every
#                        selected entry).  OFF by default: this runs on every
#                        lane job and a census that costs minutes is a census
#                        lanes will delete.
#
#   rc 0  the census ran (including when a store refused -- see below), OR the
#         binary does not carry the verb yet, which prints ONE row saying so
#         and skips.  Every binsnap older than STORE-REAP-2 is in that case,
#         and it must never read as a census that found nothing.
#   rc 3  no usable BACKEND.  A census that silently prints nothing is worse
#         than one that refuses: the row would read as "no over-age entries".
#
# IT IS DRY RUN AND IT CANNOT BE ANYTHING ELSE.  The command line is built
# here, in full, with `--dry-run` stated explicitly even though it is the
# verb's default, and there is NO passthrough of caller arguments -- so no
# caller of this script, and no future edit to a phase template, can turn a
# census into a reap by adding a word.  `store_reap_census_guard.sh` asserts
# that `--apply` appears nowhere in this file and that the stub backend it runs
# is handed `--dry-run` for every root.  The FIRST real `--apply` against a
# shared store is an operator decision made from these rows, not a harness
# behaviour.
#
# WHY THE ROOTS ARE NAMED AND NOT DERIVED.  Inside a relock, `$HOME` and
# `$XDG_CACHE_HOME` are both job-scoped, so the verb's own default root -- the
# product's `courier::persistent_store_root` formula -- resolves to the job's
# empty cache and censuses nothing.  The roots that hold the orphaned bytes are
# reachable only by being named.  Naming them here, versioned, is the honest
# form: the list is reviewable and one edit changes every lane at once.
set -uo pipefail

WHEN=${1:-unspecified}
: "${BACKEND:=}"

# The persistent roots this campaign actually writes.  `caches/rtcache` is the
# shared one every hand-run and every `RETREAD_*_STORE`-pointed job has used;
# `~/.cache/retread` is what the product's own default formula resolves to for
# a process that is NOT job-scoped -- the operator's own `pixi lock`, and the
# only root a default `store-reap` would reach.
STORE_REAP_ROOTS=${STORE_REAP_ROOTS:-"/oscar/data/stellex/glvov/caches/rtcache /users/glvov/.cache/retread"}
STORE_REAP_BYTES=${STORE_REAP_BYTES:-0}

if [ -z "$BACKEND" ] || [ ! -x "$BACKEND" ]; then
  echo "### STORE-REAP CENSUS ($WHEN): REFUSED -- BACKEND='${BACKEND:-<unset>}' is not an executable retread binary"
  exit 3
fi

# THE VERB IS NOT IN EVERY BINARY, AND A BINARY WITHOUT IT MUST NOT BE PROBED
# BY RUNNING IT.  Every binsnap in flight while STORE-REAP-2 was written
# (dcf417e, 498db14, B26's merge) predates the verb, and `retread store-reap`
# on such a binary does NOT fail: `main.rs` matches no verb, falls through to
# the automatic preflight and STARTS THE JSON-RPC TRANSPORT, so the census
# would block a relock on a backend waiting for stdin.  The detection is
# therefore STATIC -- a marker string that exists only in a binary carrying the
# verb -- and nothing is executed until it is found.  An absent verb is ONE row
# and exit 0; it must never look like a census that found nothing.
STORE_REAP_MARKER='store-reap: --store '
if ! grep -a -q -F -- "$STORE_REAP_MARKER" "$BACKEND"; then
  echo "### store_reap_census: verb absent in this binary ($BACKEND sha256=$(sha256sum "$BACKEND" | cut -c1-16)) -- census skipped"
  exit 0
fi

echo "### STORE-REAP CENSUS ($WHEN) backend=$BACKEND bytes=$STORE_REAP_BYTES roots=$STORE_REAP_ROOTS"
bytes_flag=()
[ "$STORE_REAP_BYTES" = 1 ] && bytes_flag=(--bytes)

for root in $STORE_REAP_ROOTS; do
  if [ ! -d "$root" ]; then
    echo "### STORE-REAP CENSUS ($WHEN) root=$root ABSENT -- nothing to census"
    continue
  fi
  "$BACKEND" store-reap --store all --dry-run --root "$root" "${bytes_flag[@]+"${bytes_flag[@]}"}" 2>&1 |
    sed "s|^|### STORE-REAP ($WHEN) |"
  rc=${PIPESTATUS[0]}
  # rc 7 is the verb's "a live writer held a store's try-lock" refusal.  It is
  # reported and it does NOT fail the job: housekeeping deferring to a relock
  # is correct behaviour, and a census is not the job's purpose.  It must not
  # be silent either -- a refused store scanned nothing, and `scanned=0` on a
  # store that refused looks exactly like an empty store.
  echo "### STORE-REAP CENSUS ($WHEN) root=$root rc=$rc$([ "$rc" = 7 ] && echo ' (a store REFUSED: a live writer holds its try-lock; nothing was scanned there)')"
done
echo "### STORE-REAP CENSUS ($WHEN) DONE"
exit 0
