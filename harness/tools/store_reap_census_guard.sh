#!/usr/bin/env bash
# store_reap_census_guard.sh -- the guard for store_reap_census.sh.  STORE-REAP-2.
#
# The census is the PRODUCTION READER of `retread store-reap` (law 2), and the
# one property it must never lose is that it cannot mutate a shared store.  A
# guard that only checked the happy path would pass on a file that had grown an
# `--apply`, so the assertions here are:
#
#   1. `--apply` appears NOWHERE in the census script, and every invocation it
#      makes carries `--dry-run` explicitly.
#   2. It fans out over EVERY named root, one invocation each, and skips an
#      absent root instead of invoking against it.
#   3. No BACKEND, or a non-executable one, is rc 3 -- a refusal, never a
#      silent empty census that reads as "no over-age entries".
#   4. A store refusal (the verb's rc 7) is REPORTED and does not fail the job.
#   5. `--bytes` is off by default and on with STORE_REAP_BYTES=1.
#
# Every check has a NON-VACUITY control: the stub backend records its argv, and
# the guard proves the recording works before asserting on what is absent from
# it.
#
#   usage: bash store_reap_census_guard.sh [path to store_reap_census.sh]
#   rc 0   all checks pass
#   rc 1   at least one check failed (the count is printed)
set -uo pipefail

SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CENSUS="${1:-$SELF_DIR/store_reap_census.sh}"
[ -f "$CENSUS" ] || { echo "### GUARD FATAL: no census script at $CENSUS"; exit 4; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/store-reap-census-guard.XXXXXX") || exit 4
trap 'rm -rf "$WORK"' EXIT
bad=0
chk() { # chk <name> <got> <want>
  if [ "$2" = "$3" ]; then echo "### ok   $1"; else
    echo "### FAIL $1: got [$2] want [$3]"; bad=$((bad+1)); fi
}
chk_has() { # chk_has <name> <haystack file> <needle>
  if grep -qF -- "$3" "$2"; then echo "### ok   $1"; else
    echo "### FAIL $1: [$3] not found in $2"; bad=$((bad+1)); fi
}

# ---- the stub backend: records argv, exits with whatever STUB_RC says --------
STUB=$WORK/stub-retread
ARGV=$WORK/argv.txt
cat > "$STUB" <<'STUBEOF'
#!/usr/bin/env bash
# carries the verb marker the census probes for statically: store-reap: --store 
printf '%s\n' "$*" >> "$ARGV_FILE"
echo "### store-reap TOTAL roots=1 stores=5 mode=dry-run scanned=0 would_evict=0 bytes=0 refused=false"
exit "${STUB_RC:-0}"
STUBEOF
chmod +x "$STUB"

mkdir -p "$WORK/rootA" "$WORK/rootB"

run_census() { # run_census <roots> [extra env assignments already exported]
  : > "$ARGV"
  ARGV_FILE=$ARGV BACKEND=$STUB STORE_REAP_ROOTS="$1" \
    STORE_REAP_BYTES="${BYTES:-0}" STUB_RC="${RC:-0}" \
    bash "$CENSUS" GUARD > "$WORK/out.txt" 2>&1
  echo $?
}

# ---- 1. the script itself can never apply -----------------------------------
# COMMENTS ARE NOT CODE. The census file EXPLAINS at length why it never
# applies, so a plain `grep -F -- --apply` over the whole file is a check that
# can only ever fail; the assertion is over the EXECUTABLE lines. Its
# non-vacuity control is right below it: the same check over a copy that grew
# an `--apply` in code must come out non-zero.
code_only() { grep -v '^[[:space:]]*#' "$1"; }
chk "no executable line of the census carries --apply" \
  "$(code_only "$CENSUS" | grep -cF -- '--apply')" 0
sed 's|--store all --dry-run|--store all --apply|' "$CENSUS" > "$WORK/applied.sh"
cmp -s "$CENSUS" "$WORK/applied.sh" && { echo "### GUARD FATAL: the control mutation changed NOTHING"; exit 9; }
chk "NON-VACUITY: a census that grew an --apply IS caught" \
  "$(code_only "$WORK/applied.sh" | grep -cF -- '--apply')" 1
chk_has   "the census script states --dry-run explicitly" "$CENSUS" "--dry-run"

# ---- 2. fan-out, and the recording works (non-vacuity) ----------------------
rc=$(run_census "$WORK/rootA $WORK/rootB")
chk "two present roots exit 0" "$rc" 0
chk "two present roots produce two invocations" "$(wc -l < "$ARGV")" 2
chk_has "invocation 1 names rootA" "$ARGV" "--root $WORK/rootA"
chk_has "invocation 2 names rootB" "$ARGV" "--root $WORK/rootB"
chk "every invocation carries --dry-run" \
  "$(grep -c -- '--dry-run' "$ARGV")" 2
chk "no invocation carries --apply" \
  "$(grep -c -- '--apply' "$ARGV")" 0
chk "every invocation asks for ALL stores, never a frozen subset" \
  "$(grep -c -- '--store all' "$ARGV")" 2

# ---- 3. an absent root is skipped, not invoked against ----------------------
rc=$(run_census "$WORK/rootA $WORK/does-not-exist")
chk "an absent root does not fail the census" "$rc" 0
chk "an absent root produces NO invocation" "$(wc -l < "$ARGV")" 1
chk_has "an absent root is announced" "$WORK/out.txt" "ABSENT"

# ---- 4. no backend is a refusal, not an empty census ------------------------
: > "$ARGV"
ARGV_FILE=$ARGV BACKEND= STORE_REAP_ROOTS="$WORK/rootA" bash "$CENSUS" GUARD > "$WORK/out.txt" 2>&1
chk "an unset BACKEND is rc 3" "$?" 3
chk_has "the refusal says so" "$WORK/out.txt" "REFUSED"
ARGV_FILE=$ARGV BACKEND=$WORK/not-a-binary STORE_REAP_ROOTS="$WORK/rootA" \
  bash "$CENSUS" GUARD > "$WORK/out.txt" 2>&1
chk "a non-executable BACKEND is rc 3" "$?" 3

# ---- 5. a store refusal (rc 7) is reported and does not fail the job --------
RC=7
rc=$(run_census "$WORK/rootA")
RC=0
chk "a store refusal does not fail the census" "$rc" 0
chk_has "a store refusal is reported with its rc" "$WORK/out.txt" "rc=7"
chk_has "a store refusal explains itself" "$WORK/out.txt" "REFUSED"

# ---- 6. bytes is opt-in -----------------------------------------------------
run_census "$WORK/rootA" >/dev/null
chk "bytes is OFF by default" "$(grep -c -- '--bytes' "$ARGV")" 0
BYTES=1
run_census "$WORK/rootA" >/dev/null
BYTES=0
chk "STORE_REAP_BYTES=1 turns it on" "$(grep -c -- '--bytes' "$ARGV")" 1


# ---- 7. a binary WITHOUT the verb is one row and a skip, never an empty census
# Every binsnap older than STORE-REAP-2 lacks `store-reap`, and on such a binary
# the verb is not an error -- `main.rs` falls through to the JSON-RPC transport
# and WAITS ON STDIN. The census must therefore detect it STATICALLY and must
# not run it at all. The pair of arms is the non-vacuity control for each other:
# the same census, the same roots, one stub with the marker and one without.
OLD=$WORK/old-retread
cat > "$OLD" <<'OLDEOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$ARGV_FILE"
sleep 600
OLDEOF
chmod +x "$OLD"
: > "$ARGV"
ARGV_FILE=$ARGV BACKEND=$OLD STORE_REAP_ROOTS="$WORK/rootA" \
  timeout 30 bash "$CENSUS" GUARD > "$WORK/out.txt" 2>&1
chk "a binary without the verb exits 0" "$?" 0
chk "a binary without the verb is NEVER executed" "$(wc -l < "$ARGV")" 0
chk_has "it prints the verb-absent row" "$WORK/out.txt" "verb absent in this binary"
chk_has "the row says the census was skipped" "$WORK/out.txt" "census skipped"
chk "the verb-absent row does NOT print a census summary" \
  "$(grep -c 'STORE-REAP CENSUS' "$WORK/out.txt")" 0
# NON-VACUITY: the marker-carrying stub, same call, DOES census.
: > "$ARGV"
ARGV_FILE=$ARGV BACKEND=$STUB STORE_REAP_ROOTS="$WORK/rootA" \
  bash "$CENSUS" GUARD > "$WORK/out.txt" 2>&1
chk "NON-VACUITY: a binary WITH the verb is executed" "$(wc -l < "$ARGV")" 1
chk "NON-VACUITY: and prints the census, not the absent row" \
  "$(grep -c 'verb absent in this binary' "$WORK/out.txt")" 0

echo "### GUARD bad=$bad"
[ "$bad" -eq 0 ] || echo "### GUARD FAILED"
exit $([ "$bad" -eq 0 ] && echo 0 || echo 1)
