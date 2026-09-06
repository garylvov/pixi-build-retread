#!/usr/bin/env bash
# Guard for tools/moved_row_halves.sh (MERGE-K-1, widened by MERGE-M-4).
# Doctrine law 3: every change lands with its guard test, and a guard that
# cannot fail is a defect -- so arm D MUTATES the reader and REQUIRES the
# mutation to be caught, and arm H runs the PINNED PRE-FIX FILE ITSELF and
# requires it to fail the way MERGE-M-4 says it failed on B21.
#
# Nothing here touches a real lock, a job root or the network: every arm builds
# its own fixture pair under a temp dir and deletes it.  The fixtures deliberately
# carry the four filename shapes that a naive parse gets wrong -- a conda name
# with dashes in it, a wheel, an sdist tarball and a `direct+` URL -- because a
# wrong split here is a silent wrong VERDICT on a landing, not a crash.
#
#   usage: bash moved_row_halves_guard.sh          rc 0 all arms pass, rc 1 otherwise
#
#   OLD_READER=<path>   arm H's pre-MERGE-M-4 file.  Default: extracted from the
#                       harness worktree at the pinned commit named by
#                       OLD_READER_COMMIT (default 87366e7, the HARNESS_COMMIT
#                       MERGE-M closed on and the last commit carrying the
#                       defect).  A missing old reader FAILS the arm; it never
#                       silently skips.
#   EVD=<path>          arm G's env_version_delta.py.  Default: the sibling
#                       tools/ copy, else b3-phase1/env_version_delta.py beside
#                       the task tree.  A missing one FAILS the arm.
set -uo pipefail
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
READER="$SELF_DIR/moved_row_halves.sh"
[ -x "$READER" ] || [ -f "$READER" ] || { echo "GUARD FATAL: no reader at $READER"; exit 1; }
OLD_READER_COMMIT="${OLD_READER_COMMIT:-87366e7}"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
pass=0; fail=0
ok()   { echo "PASS  $*"; pass=$((pass+1)); }
bad()  { echo "FAIL  $*"; fail=$((fail+1)); }
chk()  { if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1: got [$2] want [$3]"; fi; }

# $1 out file; remaining args: "env|half|url" triples
mklock() {
  local out=$1; shift
  { echo "version: 7"
    echo "environments:"
    local cur=""
    for spec in "$@"; do
      local e=${spec%%|*}; local rest=${spec#*|}
      local h=${rest%%|*}; local u=${rest#*|}
      if [ "$e" != "$cur" ]; then
        echo "  $e:"; echo "    packages:"; echo "      p1:"; cur=$e
      fi
      echo "      - $h: $u"
    done
    echo "packages:"
  } > "$out"
}
CF=https://prefix.dev/conda-forge/noarch
PH=https://files.pythonhosted.org/packages/aa/bb/ccdd
sumline() { echo "$1" | sed -n 's/^### MOVED-HALVES SUMMARY //p'; }
totline() { echo "$1" | sed -n 's/^### MOVED-HALVES TOTAL rows=\([0-9]*\) .*/\1/p'; }

# ---- arm A: one pypi move among conda moves -> rc 1, and the pypi row is NAMED
mklock "$WORK/a.base" \
  "envA|conda|$CF/anyio-4.15.0-pyh5ded981_0.conda" \
  "envA|conda|$CF/attrs-26.1.0-pyhcf101f3_0.conda" \
  "envB|pypi|$PH/widget-1.2.0-py3-none-any.whl"
mklock "$WORK/a.new" \
  "envA|conda|$CF/anyio-4.15.1-pyh5ded981_0.conda" \
  "envA|conda|$CF/attrs-26.2.0-pyhcf101f3_0.conda" \
  "envB|pypi|$PH/widget-1.2.1-py3-none-any.whl"
out=$(bash "$READER" "$WORK/a.base" "$WORK/a.new"); rc=$?
chk "A rc is 1 when a moved row is pypi" "$rc" "1"
chk "A summary counts the halves apart" \
  "$(sumline "$out")" "moved=3 conda=2 pypi=1 removed=0 added=0"
chk "A names the pypi row by env and package" \
  "$(echo "$out" | grep -c 'MOVED   env=envB package=widget half=pypi 1.2.0 -> 1.2.1')" "1"
chk "A refuses for the pypi reason and NOT for a count change" \
  "$(echo "$out" | sed -n 's/^### MOVED-HALVES REFUSE reasons=\([a-z_,]*\) .*/\1/p')" "pypi_rows"
chk "A per-env counts are held" \
  "$(echo "$out" | grep -c 'delta=+0')" "2"

# ---- arm B: conda-only version step, counts held -> rc 0 and the accepting line
mklock "$WORK/b.base" "envA|conda|$CF/anyio-4.15.0-pyh5ded981_0.conda" \
                      "envA|conda|$CF/attrs-26.1.0-pyhcf101f3_0.conda"
mklock "$WORK/b.new"  "envA|conda|$CF/anyio-4.15.1-pyh5ded981_0.conda" \
                      "envA|conda|$CF/attrs-26.2.0-pyhcf101f3_0.conda"
out=$(bash "$READER" "$WORK/b.base" "$WORK/b.new"); rc=$?
chk "B rc is 0 on a conda-only version step with counts held" "$rc" "0"
chk "B summary" "$(sumline "$out")" "moved=2 conda=2 pypi=0 removed=0 added=0"
chk "B prints the accepting sentence" "$(echo "$out" | grep -c 'MOVED-HALVES CLEAN')" "1"
chk "B TOTAL rows equals the moved count" "$(totline "$out")" "2"

# ---- arm C: the four filename shapes a naive split gets wrong
mklock "$WORK/c.base" \
  "envA|conda|$CF/font-ttf-dejavu-sans-mono-2.37-hab24e00_0.tar.bz2" \
  "envA|pypi|$PH/gym-0.26.2.tar.gz" \
  "envA|pypi|$PH/uniplot-0.23.2-py2.py3-none-any.whl" \
  "envA|pypi|direct+https://developer.download.nvidia.com/x/torch-2.5.0a0%2B872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl#sha256=dead"
mklock "$WORK/c.new" \
  "envA|conda|$CF/font-ttf-dejavu-sans-mono-2.38-hab24e00_0.tar.bz2" \
  "envA|pypi|$PH/gym-0.26.3.tar.gz" \
  "envA|pypi|$PH/uniplot-0.23.2-py2.py3-none-any.whl" \
  "envA|pypi|direct+https://developer.download.nvidia.com/x/torch-2.5.0a0%2B872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl#sha256=dead"
out=$(bash "$READER" "$WORK/c.base" "$WORK/c.new"); rc=$?
chk "C a dashed conda name survives the split whole" \
  "$(echo "$out" | grep -c 'package=font-ttf-dejavu-sans-mono half=conda 2.37 -> 2.38')" "1"
chk "C an sdist tarball is parsed and classified pypi" \
  "$(echo "$out" | grep -c 'package=gym half=pypi 0.26.2 -> 0.26.3')" "1"
chk "C an unmoved wheel and an unmoved direct+ URL produce no row" \
  "$(echo "$out" | grep -cE 'package=(uniplot|torch) ')" "0"
chk "C rc is 1 (the sdist move is a pypi row)" "$rc" "1"
chk "C the count is held, so the refusal is the pypi one alone" \
  "$(echo "$out" | sed -n 's/^### MOVED-HALVES REFUSE reasons=\([a-z_,]*\) .*/\1/p')" "pypi_rows"

# ---- arm R: MERGE-M-4 ITSELF -- a REMOVAL, in exactly the shape B21 produced.
# `exceptiongroup` vanishes from one env and nothing else changes.  The pre-fix
# reader called this list conda-only and exited 0 (arm H runs that file).
mklock "$WORK/r.base" \
  "envA|conda|$CF/anyio-4.15.1-pyh5ded981_0.conda" \
  "envA|conda|$CF/exceptiongroup-1.3.1-pyhd8ed1ab_0.conda" \
  "envB|conda|$CF/exceptiongroup-1.3.1-pyhd8ed1ab_0.conda"
mklock "$WORK/r.new" \
  "envA|conda|$CF/anyio-4.15.1-pyh5ded981_0.conda" \
  "envB|conda|$CF/exceptiongroup-1.3.1-pyhd8ed1ab_0.conda"
out=$(bash "$READER" "$WORK/r.base" "$WORK/r.new"); rc=$?
chk "R a removal is rc 1, never 0" "$rc" "1"
chk "R the removal is printed as its own half, old -> -" \
  "$(echo "$out" | grep -c 'REMOVED env=envA package=exceptiongroup half=conda 1.3.1 -> -')" "1"
chk "R the summary counts it as removed, not as moved" \
  "$(sumline "$out")" "moved=0 conda=0 pypi=0 removed=1 added=0"
chk "R the refusal names the count change" \
  "$(echo "$out" | sed -n 's/^### MOVED-HALVES REFUSE reasons=\([a-z_,]*\) .*/\1/p')" "package_count_change"
chk "R the losing env's count is printed and the untouched one is not flagged" \
  "$(echo "$out" | grep -c 'ENV envA .* base_pkgs=2 .* new_pkgs=1 .* delta=-1')" "1"
chk "R exactly one env changed count" \
  "$(echo "$out" | sed -n 's/.*count_changed_envs=\([0-9]*\).*/\1/p')" "1"
chk "R TOTAL rows is 1" "$(totline "$out")" "1"

# ---- arm S: an ADDITION is the mirror image and is refused the same way
mklock "$WORK/s.base" "envA|conda|$CF/anyio-4.15.1-pyh5ded981_0.conda"
mklock "$WORK/s.new"  "envA|conda|$CF/anyio-4.15.1-pyh5ded981_0.conda" \
                      "envA|pypi|$PH/newthing-9.9.9-py3-none-any.whl"
out=$(bash "$READER" "$WORK/s.base" "$WORK/s.new"); rc=$?
chk "S an addition is rc 1" "$rc" "1"
chk "S the addition is printed as - -> new" \
  "$(echo "$out" | grep -c 'ADDED   env=envA package=newthing half=pypi - -> 9.9.9')" "1"
chk "S the summary counts it as added" \
  "$(sumline "$out")" "moved=0 conda=0 pypi=0 removed=0 added=1"
chk "S an ADDED pypi package counts as a pypi row for the refusal" \
  "$(echo "$out" | sed -n 's/^### MOVED-HALVES REFUSE reasons=\([a-z_,]*\) .*/\1/p')" "pypi_rows,package_count_change"
chk "S the all-rows halves line separates it from the moved halves" \
  "$(echo "$out" | sed -n 's/^### MOVED-HALVES HALVES OVER ALL CHANGED ROWS \(rows=[0-9]* conda=[0-9]* pypi=[0-9]*\).*/\1/p')" \
  "rows=1 conda=0 pypi=1"

# ---- arm T: a WHOLE ENVIRONMENT appearing is a count change in that env
mklock "$WORK/t.base" "envA|conda|$CF/anyio-4.15.1-pyh5ded981_0.conda"
mklock "$WORK/t.new"  "envA|conda|$CF/anyio-4.15.1-pyh5ded981_0.conda" \
                      "envZ|conda|$CF/attrs-26.1.0-pyhcf101f3_0.conda"
out=$(bash "$READER" "$WORK/t.base" "$WORK/t.new"); rc=$?
chk "T a new environment is rc 1" "$rc" "1"
chk "T the new env is listed with base_pkgs=0" \
  "$(echo "$out" | grep -c 'ENV envZ .* base_pkgs=0 .* new_pkgs=1 .* delta=+1')" "1"

# ---- arm G: THE AGREEMENT CLAIM, EXECUTED.  The file's own header says its
# TOTAL rows must equal env_version_delta.py's total, so the guard runs both on
# the same fixtures rather than asserting it in prose.
EVD="${EVD:-}"
if [ -z "$EVD" ]; then
  for c in "$SELF_DIR/env_version_delta.py" "$SELF_DIR/../../b3-phase1/env_version_delta.py" \
           "$SELF_DIR/../b3-phase1/env_version_delta.py"; do
    [ -f "$c" ] && { EVD=$c; break; }
  done
fi
if [ -z "$EVD" ] || [ ! -f "$EVD" ] || ! command -v python3 >/dev/null 2>&1; then
  bad "G cannot run the agreement arm: EVD=[$EVD] python3=$(command -v python3 || echo none)"
else
  for pair in a r s t c; do
    mine=$(totline "$(bash "$READER" "$WORK/$pair.base" "$WORK/$pair.new")")
    theirs=$(python3 "$EVD" "$WORK/$pair.base" "$WORK/$pair.new" \
             | sed -n 's/^### total moved rows across all envs: //p')
    chk "G pair $pair: TOTAL rows agrees with env_version_delta.py" "$mine" "$theirs"
  done
fi

# ---- arm D: THE MUTATIONS.  Three, because they corrupt the reader at
# different depths and a guard that only knows one of them is part of a guard.
# The assertion is the property that matters, not a particular summary string:
# a mutant must FAIL to report the row AND must exit 0, i.e. it would let the
# landing through.  The observed signature is printed so a future reader of this
# guard's log can see WHICH way each mutant broke.
mutate_and_check() {
  local tag=$1 sedexpr=$2 base=$3 new=$4 wantgrep=$5
  local mutant="$WORK/mutant-$tag.sh"
  sed "$sedexpr" "$READER" > "$mutant"
  if cmp -s "$READER" "$mutant"; then
    bad "D/$tag the mutation changed nothing -- this arm is vacuous"; return
  fi
  local mout mrc mhit
  mout=$(bash "$mutant" "$base" "$new"); mrc=$?
  mhit=$(echo "$mout" | grep -c "$wantgrep")
  if [ "$mhit" -eq 0 ] && [ "$mrc" -eq 0 ]; then
    ok "D/$tag the mutant loses [$wantgrep] and exits 0 -- it would LAND it; summary [$(sumline "$mout")]"
  else
    bad "D/$tag unexpected: rc=$mrc hits=$mhit summary=[$(sumline "$mout")]"
  fi
}
#   M1  the classification itself is hard-wired to "conda".  It also drags the
#       PARSE with it (the wheel is then split by the conda rule), so the pypi
#       row is not merely mislabelled, it disappears.
mutate_and_check M1 's/half = (\$0 ~ \/- conda: \/) ? "conda" : "pypi"/half = "conda"/' \
  "$WORK/a.base" "$WORK/a.new" 'half=pypi'
#   M2  the parse is left correct and only the emitted LABEL is forced to
#       "conda" -- the surgical "cannot classify" mutant.
mutate_and_check M2 's/if (\$4 == "pypi") pypi\[k\] = 1; else conda\[k\] = 1/conda[k] = 1/' \
  "$WORK/a.base" "$WORK/a.new" 'half=pypi'
#   M3  MERGE-M-4's OWN MUTANT: restore the pre-fix behaviour by dropping any
#       group that is not present on both sides.  This is the defect that let
#       B21's four vanished packages through, stated as a one-line mutation.
mutate_and_check M3 's/^    } else if (haveb) {$/    } else if (haveb) { return/' \
  "$WORK/r.base" "$WORK/r.new" 'REMOVED'

# ---- arm H: THE PINNED PRE-FIX FILE, run as it shipped.  Not a mutant of the
# current reader -- the actual blob at OLD_READER_COMMIT, which was the second
# opinion on every landing from B17 to B21.  It must MISS arm R's removal and
# exit 0, which is exactly the MERGE-M-4 finding; if it does not, the finding
# was wrong and this guard says so.
OLD_READER="${OLD_READER:-}"
if [ -z "$OLD_READER" ]; then
  WT=/oscar/data/stellex/glvov/agrescap/worktrees/harness-tools
  if git -C "$WT" cat-file -e "$OLD_READER_COMMIT:harness/tools/moved_row_halves.sh" 2>/dev/null; then
    OLD_READER="$WORK/old-reader.sh"
    git -C "$WT" show "$OLD_READER_COMMIT:harness/tools/moved_row_halves.sh" > "$OLD_READER"
  fi
fi
if [ -z "$OLD_READER" ] || [ ! -s "$OLD_READER" ]; then
  bad "H no pre-fix reader to run (OLD_READER unset and $OLD_READER_COMMIT not reachable) -- the arm FAILS rather than skipping"
else
  hout=$(bash "$OLD_READER" "$WORK/r.base" "$WORK/r.new"); hrc=$?
  hrem=$(echo "$hout" | grep -c 'REMOVED')
  hsum=$(sumline "$hout")
  if [ "$hrc" -eq 0 ] && [ "$hrem" -eq 0 ]; then
    ok "H the pinned $OLD_READER_COMMIT reader misses the removal and exits 0 -- MERGE-M-4 reproduced; its summary was [$hsum]"
  else
    bad "H the pinned $OLD_READER_COMMIT reader did NOT reproduce MERGE-M-4: rc=$hrc removed_rows=$hrem summary=[$hsum]"
  fi
fi

# ---- arm E: setup failures are rc 2 and never a verdict
bash "$READER" "$WORK/nope.lock" "$WORK/a.new" >/dev/null 2>&1
chk "E a missing baseline is rc 2, not a verdict" "$?" "2"
printf 'version: 7\npackages:\n' > "$WORK/empty.lock"
bash "$READER" "$WORK/empty.lock" "$WORK/a.new" >/dev/null 2>&1
chk "E a lock with no environments is rc 2" "$?" "2"

# ---- arm F: an unchanged pair changes nothing
out=$(bash "$READER" "$WORK/a.base" "$WORK/a.base"); rc=$?
chk "F identical locks: rc 0" "$rc" "0"
chk "F identical locks: nothing in any column" \
  "$(sumline "$out")" "moved=0 conda=0 pypi=0 removed=0 added=0"
chk "F identical locks: TOTAL rows 0" "$(totline "$out")" "0"

echo "### MOVED-HALVES GUARD $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
