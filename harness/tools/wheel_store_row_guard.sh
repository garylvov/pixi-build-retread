#!/usr/bin/env bash
# Guard for tools/wheel_store_row.sh (MERGE-M-1).  Doctrine law 3.
#
# The arm that matters is B: the EXACT environment B18-B21 ran in --
# RETREAD_WHEEL_STORE set by `retread_fast_env` to the shared persistent store
# while XDG_CACHE_HOME is job-scoped -- where the old hand-written row said
# "job-scoped … RETREAD_WHEEL_STORE unset" and printed the shared path in the
# same breath.  The row must now say shared, name RETREAD_WHEEL_STORE as the
# provenance, and print the store the product would actually open.
#
#   usage: bash wheel_store_row_guard.sh    rc 0 all arms pass, rc 1 otherwise
set -uo pipefail
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROW="$SELF_DIR/wheel_store_row.sh"
[ -f "$ROW" ] || { echo "GUARD FATAL: no reader at $ROW"; exit 1; }
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
pass=0; fail=0
ok()  { echo "PASS  $*"; pass=$((pass+1)); }
bad() { echo "FAIL  $*"; fail=$((fail+1)); }
chk() { if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1: got [$2] want [$3]"; fi; }

C="$WORK/job"                 # the job scope root, $C in every harness here
SHARED="$WORK/shared/retread/wheels"
mkdir -p "$C/xdg-cache" "$SHARED"

field() { echo "$1" | sed -n "s/.*[ #]$2=\([^ ]*\).*/\1/p" | head -1; }
run() { env -u RETREAD_WHEEL_STORE -u XDG_CACHE_HOME -u HOME "$@" bash "$ROW" "${JOBARG-}" 2>&1; }

# ---- arm A: no RETREAD_WHEEL_STORE, XDG_CACHE_HOME job-scoped -> job-scoped
JOBARG="$C"
out=$(run XDG_CACHE_HOME="$C/xdg-cache" HOME="$C/home")
chk "A resolved is XDG_CACHE_HOME/retread/wheels" "$(field "$out" resolved)" "$C/xdg-cache/retread/wheels"
chk "A provenance is XDG_CACHE_HOME"              "$(field "$out" provenance)" "XDG_CACHE_HOME"
chk "A scope is job-scoped"                       "$(field "$out" scope)" "job-scoped"

# ---- arm B: MERGE-M-1's OWN CASE.  RETREAD_WHEEL_STORE set to the shared
# store, XDG_CACHE_HOME job-scoped.  The old row called this "job-scoped
# … RETREAD_WHEEL_STORE unset" and printed the shared path in the parenthesis.
out=$(run RETREAD_WHEEL_STORE="$SHARED" XDG_CACHE_HOME="$C/xdg-cache" HOME="$C/home")
chk "B resolved is the shared store, used whole"  "$(field "$out" resolved)" "$SHARED"
chk "B provenance names the variable that won"    "$(field "$out" provenance)" "RETREAD_WHEEL_STORE"
chk "B scope is shared, NOT job-scoped"           "$(field "$out" scope)" "shared"
chk "B the row never says the variable is unset" \
  "$(echo "$out" | grep -ci 'unset:')" "0"
chk "B the inputs line carries the value verbatim" \
  "$(echo "$out" | grep -c "RETREAD_WHEEL_STORE=$SHARED")" "1"
chk "B an existing store is reported present"     "$(field "$out" exists)" "present"

# ---- arm C: HOME fallback when neither of the first two is set
out=$(run HOME="$C/home")
chk "C resolved is HOME/.cache/retread/wheels" "$(field "$out" resolved)" "$C/home/.cache/retread/wheels"
chk "C provenance is HOME"                     "$(field "$out" provenance)" "HOME"
chk "C a store that does not exist yet is reported absent" "$(field "$out" exists)" "absent"

# ---- arm D: no job scope declared -> the row refuses to guess a scope
JOBARG=""
out=$(run RETREAD_WHEEL_STORE="$SHARED" XDG_CACHE_HOME="$C/xdg-cache" HOME="$C/home")
chk "D without a declared job root the scope is undeclared, not guessed" \
  "$(field "$out" scope)" "undeclared"
JOBARG="$C"

# ---- arm E: an empty / whitespace RETREAD_WHEEL_STORE is NOT a value, exactly
# as `.filter(|s| !s.trim().is_empty())` decides it in the product.
out=$(run RETREAD_WHEEL_STORE="   " XDG_CACHE_HOME="$C/xdg-cache" HOME="$C/home")
chk "E a whitespace-only variable falls through to XDG_CACHE_HOME" \
  "$(field "$out" provenance)" "XDG_CACHE_HOME"

# ---- arm F: nothing derivable is rc 2, a setup failure and never a row
env -u RETREAD_WHEEL_STORE -u XDG_CACHE_HOME -u HOME bash "$ROW" >/dev/null 2>&1
chk "F no inputs at all is rc 2" "$?" "2"

# ---- arm G: THE MUTATION.  A reader that hard-wires the scope the way the old
# hand-written row did must be caught: it reports job-scoped in arm B's shared
# environment.  A guard that cannot fail is a defect.
MUT="$WORK/mutant.sh"
sed 's/^scope="undeclared"$/scope="job-scoped"/; s/^    \*)       scope="shared" ;;$/    *)       scope="job-scoped" ;;/' "$ROW" > "$MUT"
if cmp -s "$ROW" "$MUT"; then
  bad "G the mutation changed nothing -- this arm is vacuous"
else
  mout=$(env -u RETREAD_WHEEL_STORE -u XDG_CACHE_HOME -u HOME \
    RETREAD_WHEEL_STORE="$SHARED" XDG_CACHE_HOME="$C/xdg-cache" HOME="$C/home" \
    bash "$MUT" "$C" 2>&1)
  if [ "$(field "$mout" scope)" = "job-scoped" ]; then
    ok "G the hard-wired-scope mutant calls the SHARED store job-scoped -- the MERGE-M-1 row, reproduced and caught"
  else
    bad "G the mutant did not reproduce the defect: scope=[$(field "$mout" scope)]"
  fi
fi

echo "### WHEEL STORE ROW GUARD $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
