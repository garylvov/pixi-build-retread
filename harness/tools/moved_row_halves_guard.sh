#!/usr/bin/env bash
# Guard for tools/moved_row_halves.sh (MERGE-K-1).  Doctrine law 3: every change
# lands with its guard test, and a guard that cannot fail is a defect -- so arm D
# MUTATES the reader and REQUIRES the mutation to be caught.
#
# Nothing here touches a real lock, a job root or the network: every arm builds
# its own fixture pair under a temp dir and deletes it.  The fixtures deliberately
# carry the four filename shapes that a naive parse gets wrong -- a conda name
# with dashes in it, a wheel, an sdist tarball and a `direct+` URL -- because a
# wrong split here is a silent wrong VERDICT on a landing, not a crash.
#
#   usage: bash moved_row_halves_guard.sh          rc 0 all arms pass, rc 1 otherwise
set -uo pipefail
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
READER="$SELF_DIR/moved_row_halves.sh"
[ -x "$READER" ] || [ -f "$READER" ] || { echo "GUARD FATAL: no reader at $READER"; exit 1; }
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
  "$(echo "$out" | sed -n 's/^### MOVED-HALVES SUMMARY //p')" "moved=3 conda=2 pypi=1"
chk "A names the pypi row by env and package" \
  "$(echo "$out" | grep -c 'MOVED env=envB package=widget half=pypi 1.2.0 -> 1.2.1')" "1"
chk "A prints the refusal sentence, not the accept one" \
  "$(echo "$out" | grep -c 'PYPI ROWS PRESENT')" "1"

# ---- arm B: conda-only -> rc 0 and the accepting sentence
mklock "$WORK/b.base" "envA|conda|$CF/anyio-4.15.0-pyh5ded981_0.conda" \
                      "envA|conda|$CF/attrs-26.1.0-pyhcf101f3_0.conda"
mklock "$WORK/b.new"  "envA|conda|$CF/anyio-4.15.1-pyh5ded981_0.conda" \
                      "envA|conda|$CF/attrs-26.2.0-pyhcf101f3_0.conda"
out=$(bash "$READER" "$WORK/b.base" "$WORK/b.new"); rc=$?
chk "B rc is 0 on a conda-only moved list" "$rc" "0"
chk "B summary" "$(echo "$out" | sed -n 's/^### MOVED-HALVES SUMMARY //p')" "moved=2 conda=2 pypi=0"
chk "B prints the accepting sentence" "$(echo "$out" | grep -c 'NO PYPI ROWS')" "1"

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

# ---- arm D: THE MUTATIONS.  Two, because they corrupt the reader at different
# depths and a guard that only knows one of them is half a guard.  The assertion
# in both cases is the property that matters, not a particular summary string:
# a mutant must FAIL to report the pypi row AND must exit 0, i.e. it would let
# the landing through.  The observed signature is printed so a future reader of
# this guard's log can see WHICH way each mutant broke.
#   M1  the classification itself is hard-wired to "conda".  It also drags the
#       PARSE with it (the wheel is then split by the conda rule), so the row is
#       not merely mislabelled, it disappears -- moved falls from 3 to 2.
#   M2  the parse is left correct and only the emitted LABEL is forced to
#       "conda".  This is the surgical "cannot classify" mutant: the row is
#       still found and still moved, and it is called conda.
mutate_and_check() {
  local tag=$1
  local sedexpr=$2
  local mutant="$WORK/mutant-$tag.sh"
  sed "$sedexpr" "$READER" > "$mutant"
  if cmp -s "$READER" "$mutant"; then
    bad "D/$tag the mutation changed nothing -- this arm is vacuous"; return
  fi
  local mout mrc msum mpypi
  mout=$(bash "$mutant" "$WORK/a.base" "$WORK/a.new"); mrc=$?
  msum=$(echo "$mout" | sed -n 's/^### MOVED-HALVES SUMMARY //p')
  mpypi=$(echo "$mout" | grep -c 'half=pypi')
  if [ "$msum" = "moved=3 conda=2 pypi=1" ]; then
    bad "D/$tag the mutant still gets it right -- the sed missed the code it aimed at"
  elif [ "$mpypi" -eq 0 ] && [ "$mrc" -eq 0 ]; then
    ok "D/$tag the mutant loses the pypi row and exits 0 -- it would LAND it; summary [$msum]"
  else
    bad "D/$tag unexpected: rc=$mrc pypi_rows=$mpypi summary=[$msum]"
  fi
}
mutate_and_check M1 's/half = (\$0 ~ \/- conda: \/) ? "conda" : "pypi"/half = "conda"/'
mutate_and_check M2 's/{ print \$1 "\\t" \$2 "\\t" \$4 "\\t" \$3 }/{ print $1 "\\t" $2 "\\tconda\\t" $3 }/'
# ---- arm E: setup failures are rc 2 and never a verdict
bash "$READER" "$WORK/nope.lock" "$WORK/a.new" >/dev/null 2>&1
chk "E a missing baseline is rc 2, not a verdict" "$?" "2"
printf 'version: 7\npackages:\n' > "$WORK/empty.lock"
bash "$READER" "$WORK/empty.lock" "$WORK/a.new" >/dev/null 2>&1
chk "E a lock with no environments is rc 2" "$?" "2"

# ---- arm F: an unchanged pair moves nothing
out=$(bash "$READER" "$WORK/a.base" "$WORK/a.base"); rc=$?
chk "F identical locks: rc 0" "$rc" "0"
chk "F identical locks: moved=0" \
  "$(echo "$out" | sed -n 's/^### MOVED-HALVES SUMMARY //p')" "moved=0 conda=0 pypi=0"

echo "### MOVED-HALVES GUARD $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
