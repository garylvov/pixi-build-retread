#!/usr/bin/env bash
# p6af-2h STEP 3 PROBE -- the design question, answered by running it, not argued.
#
# For CONDA, p6af's freeze works because `[mirrors]` is a TRANSPARENT url-mapping
# layer: the lock still records the real channel url. The question this probe
# settles is whether pixi 0.73.0 has anything of that shape for PyPI.
#
#   pixi config supported keys (from `pixi config set nonexistent.key 1`):
#     mirrors, pypi-config, pypi-config.index-url, pypi-config.extra-index-urls,
#     pypi-config.allow-insecure-host, proxy-config.{http,https,non-proxy-hosts}
#
#   and the canonical lock RECORDS the index list -- `indexes:` blocks naming
#   https://pypi.org/simple, https://pypi.nvidia.com/, https://py.mujoco.org/.
#
# ARM A: no pypi-config at all           -> baseline `indexes:` block
# ARM B: pypi-config.index-url = a LOCAL http url that does not even exist
#        -> if the `indexes:` block in the produced lock names the local url,
#           a pypi "mirror" done this way CHANGES THE LOCK, and the p6af design
#           cannot be reused for PyPI. If the block is unchanged, it can.
# ARM C: pypi-config.extra-index-urls  -> does a config-level extra index reach
#        a manifest that already declares its own extra-index-urls?
#
# Nothing here is a workaround and nothing is left running: it is three locks of
# a two-package fixture in a scratch dir, read-only against everything of ours.
set -uo pipefail
D=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6af2h-phase1
A=$D/artifacts
W=${SLURM_TMPDIR:-/tmp}/p6af2h-probe2-${SLURM_JOB_ID:-x}
PIXI=$(command -v pixi || echo /users/glvov/.pixi/bin/pixi)
mkdir -p "$W/ws" "$A"
echo "### P6AF2H PROBE2 start $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "### pixi: $PIXI -> $("$PIXI" --version 2>&1)"

cat > "$W/ws/pixi.toml" <<'TOML'
[workspace]
name = "p6af2h-probe2"
channels = ["https://prefix.dev/conda-forge"]
platforms = ["linux-64"]

[pypi-options]
extra-index-urls = ["https://pypi.org/simple", "https://py.mujoco.org"]

[dependencies]
python = "3.11.*"

[pypi-dependencies]
idna = "*"
TOML
echo "### FIXTURE MANIFEST (mirrors the canonical manifest's shape: extra-index-urls declared IN the manifest)"
sed 's/^/###   /' "$W/ws/pixi.toml"

arm () {  # $1 name  $2 config body (may be empty)
  local a=$1 body=$2 rc t0 H
  H=$W/home-$a
  rm -rf "$H" "$W/cache-$a" "$W/ws/pixi.lock" "$W/ws/.pixi"
  mkdir -p "$H/.pixi" "$H/.config" "$W/cache-$a"
  printf '%s' "$body" > "$H/.pixi/config.toml"
  echo "### ARM $a config.toml:"
  sed 's/^/###     /' "$H/.pixi/config.toml"
  t0=$(date +%s)
  ( cd "$W/ws" && env HOME="$H" PIXI_HOME="$H/.pixi" XDG_CONFIG_HOME="$H/.config" \
      PIXI_CACHE_DIR="$W/cache-$a" "$PIXI" lock ) > "$W/$a.log" 2>&1
  rc=$?
  echo "### ARM $a lock rc=$rc wall=$(( $(date +%s) - t0 ))s"
  if [ -f "$W/ws/pixi.lock" ]; then
    cp "$W/ws/pixi.lock" "$A/probe2-$a.pixi.lock"
    echo "### ARM $a indexes: block, verbatim"
    awk '/^ *indexes:/{p=1;print "###     "$0;next} p&&/^ *- /{print "###     "$0;next} p{p=0}' "$W/ws/pixi.lock"
    echo "### ARM $a lock md5 $(md5sum "$W/ws/pixi.lock" | awk '{print $1}')"
  else
    echo "### ARM $a NO LOCK PRODUCED -- tail of its log:"
    tail -20 "$W/$a.log" | sed 's/^/###     /'
  fi
}

arm A ''
arm B '[pypi-config]
index-url = "http://127.0.0.1:59999/simple"
'
arm C '[pypi-config]
extra-index-urls = ["http://127.0.0.1:59999/simple"]
'

echo "### PROBE2 COMPARISON"
for a in A B C; do
  f=$A/probe2-$a.pixi.lock
  [ -f "$f" ] && echo "###   arm=$a md5=$(md5sum "$f" | awk '{print $1}') indexes=[$(awk '/^ *indexes:/{p=1;next} p&&/^ *- /{printf "%s ",$2;next} p{p=0}' "$f")]" \
              || echo "###   arm=$a NO LOCK"
done
rm -rf "$W"
echo "### P6AF2H PROBE2 DONE $(date -Is)"
