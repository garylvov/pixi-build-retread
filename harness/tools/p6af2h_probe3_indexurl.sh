#!/usr/bin/env bash
# p6af-2h PROBE 3 -- probe2 said `pypi-config.index-url` moved NOTHING, and a
# no-effect result is worthless without a NON-VACUITY control: "the knob does
# nothing" and "the config was never read" look identical from the lock.
#
# Three things this probe adds that probe2 lacked:
#   1. `pixi config list` under the arm's own PIXI_HOME -- proof the file is read.
#   2. A fixture whose manifest declares NO [pypi-options] at all, so the config
#      is the only thing that could name an index.
#   3. A dead port. If `index-url` is honoured on that fixture, the lock MUST
#      fail (nothing can serve `idna`); if it locks anyway from pypi.org, the
#      key is inert for locking. Either answer is decisive; a silent pass is not.
set -uo pipefail
D=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6af2h-phase1
A=$D/artifacts
W=${SLURM_TMPDIR:-/tmp}/p6af2h-probe3-${SLURM_JOB_ID:-x}
PIXI=$(command -v pixi || echo /users/glvov/.pixi/bin/pixi)
mkdir -p "$W" "$A"
echo "### P6AF2H PROBE3 start $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "### pixi: $PIXI -> $("$PIXI" --version 2>&1)"

mk_ws () {  # $1 dir  $2 with|without pypi-options
  mkdir -p "$1"
  {
    echo '[workspace]'
    echo 'name = "p6af2h-probe3"'
    echo 'channels = ["https://prefix.dev/conda-forge"]'
    echo 'platforms = ["linux-64"]'
    echo
    if [ "$2" = with ]; then
      echo '[pypi-options]'
      echo 'extra-index-urls = ["https://pypi.org/simple"]'
      echo
    fi
    echo '[dependencies]'
    echo 'python = "3.11.*"'
    echo
    echo '[pypi-dependencies]'
    echo 'idna = "*"'
  } > "$1/pixi.toml"
}

arm () {  # $1 name  $2 with|without  $3 config body
  local a=$1 opts=$2 body=$3 rc t0 H WS
  H=$W/home-$a; WS=$W/ws-$a
  rm -rf "$H" "$WS" "$W/cache-$a"
  mkdir -p "$H/.pixi" "$H/.config" "$W/cache-$a"
  mk_ws "$WS" "$opts"
  printf '%s' "$body" > "$H/.pixi/config.toml"
  echo "### ARM $a manifest pypi-options=$opts config.toml:"
  sed 's/^/###     /' "$H/.pixi/config.toml"
  echo "### ARM $a NON-VACUITY CONTROL -- what pixi itself says it loaded:"
  env HOME="$H" PIXI_HOME="$H/.pixi" XDG_CONFIG_HOME="$H/.config" \
      "$PIXI" config list 2>&1 | sed 's/^/###     /'
  t0=$(date +%s)
  ( cd "$WS" && env HOME="$H" PIXI_HOME="$H/.pixi" XDG_CONFIG_HOME="$H/.config" \
      PIXI_CACHE_DIR="$W/cache-$a" "$PIXI" lock ) > "$W/$a.log" 2>&1
  rc=$?
  echo "### ARM $a lock rc=$rc wall=$(( $(date +%s) - t0 ))s"
  if [ -f "$WS/pixi.lock" ]; then
    cp "$WS/pixi.lock" "$A/probe3-$a.pixi.lock"
    echo "### ARM $a indexes=[$(awk '/^ *indexes:/{p=1;next} p&&/^ *- /{printf "%s ",$2;next} p{p=0}' "$WS/pixi.lock")] md5=$(md5sum "$WS/pixi.lock" | awk '{print $1}')"
    grep -c '127.0.0.1' "$WS/pixi.lock" | sed 's/^/###   ARM '"$a"' lines naming 127.0.0.1: /'
  else
    echo "### ARM $a NO LOCK -- tail:"; tail -15 "$W/$a.log" | sed 's/^/###     /'
  fi
}

arm D without ''
arm E without '[pypi-config]
index-url = "http://127.0.0.1:59999/simple"
'
arm F with '[pypi-config]
index-url = "http://127.0.0.1:59999/simple"
'
# G: the SAME dead-port index url, but declared where the canonical manifest
#    declares its own -- IN THE MANIFEST. This is the control that says whether
#    the lock records what the manifest names (and so whether a mirror declared
#    this way would corrupt the canonical lock).
H=$W/home-G; WS=$W/ws-G
rm -rf "$H" "$WS" "$W/cache-G"; mkdir -p "$H/.pixi" "$H/.config" "$W/cache-G"
mk_ws "$WS" without
sed -i 's|^\[dependencies\]|[pypi-options]\nindex-url = "https://pypi.org/simple"\nextra-index-urls = ["https://py.mujoco.org"]\n\n[dependencies]|' "$WS/pixi.toml"
echo "### ARM G manifest declares index-url + extra-index-urls ITSELF:"; sed 's/^/###     /' "$WS/pixi.toml"
( cd "$WS" && env HOME="$H" PIXI_HOME="$H/.pixi" XDG_CONFIG_HOME="$H/.config" \
    PIXI_CACHE_DIR="$W/cache-G" "$PIXI" lock ) > "$W/G.log" 2>&1
echo "### ARM G lock rc=$?"
[ -f "$WS/pixi.lock" ] && { cp "$WS/pixi.lock" "$A/probe3-G.pixi.lock"; \
  echo "### ARM G indexes=[$(awk '/^ *indexes:/{p=1;next} p&&/^ *- /{printf "%s ",$2;next} p{p=0}' "$WS/pixi.lock")]"; } \
  || { echo "### ARM G NO LOCK -- tail:"; tail -15 "$W/G.log" | sed 's/^/###     /'; }

rm -rf "$W"
echo "### P6AF2H PROBE3 DONE $(date -Is)"
