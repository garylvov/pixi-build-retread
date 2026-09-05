#!/usr/bin/env bash
# p6af-2h PROBE 4 -- the ONE remaining shape that would make a PyPI freeze cheap.
#
# probe3 settled that pixi 0.73.0's `pypi-config.index-url` is READ (config list
# echoes it) and INERT for `pixi lock`, and that a manifest-declared index-url is
# RECORDED IN THE LOCK. So a mirror declared either way is useless: one does
# nothing, the other corrupts the artefact we are trying to reproduce.
#
# The win condition, if it exists, is an override pixi does NOT record but its
# EMBEDDED uv DOES obey -- exactly the transparency `[mirrors]` gives the conda
# half. uv's own env names are the candidates. A dead port is the discriminator:
#   honoured  -> the lock FAILS (nothing serves `idna`)
#   inert     -> the lock succeeds and records https://pypi.org/simple
# Both answers are decisive; neither is a silent pass.
#
# It also records which of these names the pixi binary contains at all, and
# whether the TLS knobs a terminating proxy would need are present -- that is
# the sizing input for the fallback design.
set -uo pipefail
D=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6af2h-phase1
A=$D/artifacts
W=${SLURM_TMPDIR:-/tmp}/p6af2h-probe4-${SLURM_JOB_ID:-x}
PIXI=$(command -v pixi || echo /users/glvov/.pixi/bin/pixi)
PIXIREAL=$(dirname "$(readlink -f "$PIXI")")/pixi.real
[ -x "$PIXIREAL" ] || PIXIREAL=$(readlink -f "$PIXI")
mkdir -p "$W" "$A"
echo "### P6AF2H PROBE4 start $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "### pixi: $PIXI -> $("$PIXI" --version 2>&1); real binary $PIXIREAL"

echo "### ENV NAMES PRESENT IN THE pixi BINARY (strings, exact match)"
for n in UV_INDEX_URL UV_EXTRA_INDEX_URL UV_DEFAULT_INDEX UV_INDEX UV_OFFLINE \
         UV_NO_CACHE UV_CACHE_DIR UV_HTTP_TIMEOUT UV_INSECURE_HOST \
         SSL_CERT_FILE SSL_CERT_DIR SSL_CLIENT_CERT REQUESTS_CA_BUNDLE \
         CURL_CA_BUNDLE PIP_INDEX_URL NATIVE_TLS UV_NATIVE_TLS; do
  c=$(strings -n 4 "$PIXIREAL" | grep -cx "$n")
  echo "###   $n present=$c"
done

mk_ws () {
  mkdir -p "$1"
  printf '%s\n' '[workspace]' 'name = "p6af2h-probe4"' \
    'channels = ["https://prefix.dev/conda-forge"]' 'platforms = ["linux-64"]' \
    '' '[dependencies]' 'python = "3.11.*"' '' '[pypi-dependencies]' 'idna = "*"' \
    > "$1/pixi.toml"
}

arm () {  # $1 name  $2... env assignments
  local a=$1; shift
  local rc t0 H WS
  H=$W/home-$a; WS=$W/ws-$a
  rm -rf "$H" "$WS" "$W/cache-$a"; mkdir -p "$H/.pixi" "$H/.config" "$W/cache-$a"
  mk_ws "$WS"
  echo "### ARM $a env: $*"
  t0=$(date +%s)
  ( cd "$WS" && env HOME="$H" PIXI_HOME="$H/.pixi" XDG_CONFIG_HOME="$H/.config" \
      PIXI_CACHE_DIR="$W/cache-$a" "$@" "$PIXI" lock ) > "$W/$a.log" 2>&1
  rc=$?
  echo "### ARM $a lock rc=$rc wall=$(( $(date +%s) - t0 ))s"
  if [ -f "$WS/pixi.lock" ]; then
    echo "### ARM $a indexes=[$(awk '/^ *indexes:/{p=1;next} p&&/^ *- /{printf "%s ",$2;next} p{p=0}' "$WS/pixi.lock")] md5=$(md5sum "$WS/pixi.lock" | awk '{print $1}')"
    echo "### ARM $a idna row: $(grep -m1 'idna' "$WS/pixi.lock" | sed 's/^ *//')"
  else
    echo "### ARM $a NO LOCK -- tail:"; tail -12 "$W/$a.log" | sed 's/^/###     /'
  fi
}

DEAD=http://127.0.0.1:59999/simple
arm base
arm uv_index_url        UV_INDEX_URL="$DEAD"
arm uv_default_index    UV_DEFAULT_INDEX="$DEAD"
arm uv_extra_index      UV_EXTRA_INDEX_URL="$DEAD"
arm pip_index_url       PIP_INDEX_URL="$DEAD"
# and the control that PROVES a dead endpoint really is fatal when it IS used:
# point the CONDA side at a dead mirror the same way p6af does, which is known
# to work, so a rc=0 above cannot be read as "the fixture never needed the net".
echo "### CONTROL: the same dead endpoint via the conda [mirrors] key, which IS honoured"
H=$W/home-ctl; WS=$W/ws-ctl
rm -rf "$H" "$WS" "$W/cache-ctl"; mkdir -p "$H/.pixi" "$H/.config" "$W/cache-ctl"
mk_ws "$WS"
printf '[mirrors]\n"https://prefix.dev/conda-forge" = ["http://127.0.0.1:59999/conda-forge"]\n' > "$H/.pixi/config.toml"
( cd "$WS" && env HOME="$H" PIXI_HOME="$H/.pixi" XDG_CONFIG_HOME="$H/.config" \
    PIXI_CACHE_DIR="$W/cache-ctl" "$PIXI" lock ) > "$W/ctl.log" 2>&1
echo "### CONTROL lock rc=$? (want NON-ZERO: proves a dead endpoint IS fatal when the knob is live)"
tail -6 "$W/ctl.log" | sed 's/^/###     /'

rm -rf "$W"
echo "### P6AF2H PROBE4 DONE $(date -Is)"
