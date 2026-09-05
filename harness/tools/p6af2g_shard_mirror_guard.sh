#!/bin/bash
# p6af2g_shard_mirror_guard.sh -- the reader for the SHARD-INDEX channel mirror.
#
# IT RUNS THE ORG, NOT A STRUCTURE CHECK. Every arm is a real HTTP conversation
# with the real `channel_mirror_server.py` over a real frozen mirror of real
# channels, and the last arm is a real `pixi lock`.
#
#   ARM 1  PROTOCOL REPLAY, in the order `rattler_repodata_gateway` performs it:
#            GET <pair>/repodata_shards.msgpack.zst          200, sha == frozen
#            GET <shards dir>/<sha>.msgpack.zst              200, sha256(body)==name
#            GET <pair>/<a package named by that shard>      200, filled from the
#                                                            SHARD's record
#            GET <sharded pair>/repodata.json                404 + CLASSIC-DOWNGRADE
#            GET <classic pair>/repodata.json                200
#          The two `repodata.json` lines are the fixture's two-pair requirement:
#          the sharded pair must have NO classic document to fall back to, and
#          the classic-only pair must still be served.
#   ARM 2  MUTATION -- a shard sha the frozen index does not name.  MUST 404 and
#          MUST write a NOSHARD row.  If this passes as a 200 the allow-list is
#          decorative and the mirror is a proxy.
#   ARM 3  MUTATION -- a shard whose CONTENT does not hash to its name.  A local
#          upstream is stood up that answers `<sha>.msgpack.zst` with a different
#          shard's bytes and the pair's `shards_base_url` is repointed at it.
#          MUST 404 and MUST write a SHARD-SHA-MISMATCH row.  No network, no
#          forged msgpack.
#   ARM 4  MUTATION -- one byte flipped in a FROZEN INDEX FILE on disk.  The
#          server MUST REFUSE TO START, because an index that moved between the
#          freeze and the serve is a different universe.
#   ARM 5  LIVE PIXI.  A real `pixi lock` of a conda-forge probe manifest with
#          the network dead, `[mirrors]` pointing at this mirror INCLUDING the
#          `shards.prefix.dev` key, and a COLD pixi cache.  MUST succeed, the
#          access log MUST show `repodata_shards.msgpack.zst 200` and shard
#          200s, and MUST show ZERO `repodata.json` reads for the conda-forge
#          pairs -- that zero is the whole claim of the lane.
#   ARM 0  NON-VACUITY for arm 5: the same lock with the network dead and NO
#          mirror MUST FAIL.  Without it arm 5 proves only that something
#          answered.
#
# Usage:  p6af2g_shard_mirror_guard.sh <workdir> [port base]
set -u
W=${1:?workdir}; PB=${2:-19310}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIXI=${PIXI_BIN:-/users/glvov/.pixi/bin/pixi.real}
[ -x "$PIXI" ] || PIXI=/users/glvov/.pixi/bin/pixi
rm -rf "$W"; mkdir -p "$W" || exit 2
fail=0
ok ()   { echo "GUARD OK   $*"; }
bad ()  { echo "GUARD FAIL $*"; fail=1; }

# Sourced HARD, not `|| true`: `retread_shard_mirror.sh` calls
# `retread_pixi_mirror_config` out of this file rather than re-implementing it,
# and a swallowed source error would surface as a confusing "command not found"
# three arms later.  Only two lines of it run at source time (a
# RETREAD_PERSIST_CACHE_ROOT default); everything else is a function.
# shellcheck source=/dev/null
. "$HERE/retread_fast_env.sh" || { echo "GUARD FATAL cannot source retread_fast_env.sh"; exit 2; }
# shellcheck source=/dev/null
. "$HERE/retread_shard_mirror.sh" || { echo "GUARD FATAL cannot source retread_shard_mirror.sh"; exit 2; }

# ---- the fixture lock. TWO channels, and it is a real lock's shape, not a
# ---- pixi.lock: `shard_mirror._pairs_from_lock` reads exactly these two line
# ---- forms and nothing else, so a synthetic one is honest here.
# robostack-humble is in it for ONE reason, measured (probe 5870870): its
# `noarch` index is published with an EMPTY `shards` map, so the freeze records
# that pair as `classic` and serves its 174-byte document.  That is the
# classic-only half of the two-pair fixture, and it is a real channel doing a
# real thing rather than a directory the guard invented.
cat > "$W/fixture.lock" <<'LOCK'
  channels:
      - url: https://prefix.dev/conda-forge/
      - url: https://prefix.dev/robostack-humble/
  packages:
      - conda: https://prefix.dev/conda-forge/linux-64/xz-5.8.1-hbcc6ac9_2.conda
LOCK

MIRROR=$W/mirror
retread_freeze_shard_mirror "$W/fixture.lock" "$MIRROR" || { echo "GUARD FATAL freeze refused"; exit 2; }
echo "### GUARD frozen index bytes: $(find "$MIRROR" -maxdepth 3 -name 'repodata_shards.msgpack.zst' -printf '%s\n' | awk '{s+=$1} END{print s+0}')"

serve () {  # $1 port -> sets SRV; rc!=0 if the server refuses to come up
  RETREAD_MIRROR_FETCH_LOG=$W/pkgfetch-$1.log
  export RETREAD_MIRROR_FETCH_LOG
  : > "$RETREAD_MIRROR_FETCH_LOG"
  python3 "$HERE/channel_mirror_server.py" "$MIRROR" "$1" "$W/pkgs-$1" \
    > "$W/httpd-$1.log" 2>&1 &
  SRV=$!
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do curl -fs -o /dev/null "http://127.0.0.1:$1/" && return 0; sleep 1; done
  return 1
}

P1=$PB
serve "$P1" || { echo "GUARD FATAL server never answered on $P1"; sed "s/^/  /" "$W/httpd-$P1.log"; exit 2; }
trap 'kill $SRV ${UPS:-} 2>/dev/null' EXIT
U=http://127.0.0.1:$P1
SLUG=prefix.dev__conda-forge
SUB=linux-64

# ---------------------------------------------------------------- ARM 1 ----
python3 "$HERE/shard_mirror.py" probe "$MIRROR" "$SLUG" "$SUB" > "$W/probe.txt" || {
  echo "GUARD FATAL probe refused"; exit 2; }
sed 's/^/  probe /' "$W/probe.txt"
val () { awk -F'\t' -v k="$1" '$1==k{print $2}' "$W/probe.txt"; }
IDXSHA=$(val index_sha256); SHSHA=$(val shard_sha); SHPATH=$(val shard_path)
SHNAME=$(val shard_name)

code=$(curl -s -o "$W/idx.bin" -w '%{http_code}' "$U/$SLUG/$SUB/repodata_shards.msgpack.zst")
got=$(sha256sum "$W/idx.bin" | cut -d' ' -f1)
if [ "$code" = 200 ] && [ "$got" = "$IDXSHA" ]; then
  ok "A1 index: $SLUG/$SUB served 200 and sha256 matches the freeze ($got)"
else bad "A1 index: code=$code sha=$got want=$IDXSHA"; fi

code=$(curl -s -o "$W/shard.bin" -w '%{http_code}' "$U/$SHPATH")
got=$(sha256sum "$W/shard.bin" | cut -d' ' -f1)
if [ "$code" = 200 ] && [ "$got" = "$SHSHA" ]; then
  ok "A1 shard: $SHNAME served 200 from $SHPATH, sha256 of the COMPRESSED body == the index's name"
else bad "A1 shard: code=$code sha=$got want=$SHSHA path=$SHPATH"; fi

PKG=$(python3 "$HERE/shard_mirror.py" shard-files "$MIRROR" "$SHPATH" 2>/dev/null | head -1 | cut -f2)
if [ -n "$PKG" ]; then
  code=$(curl -s -o "$W/pkg.bin" -w '%{http_code}' "$U/$SLUG/$SUB/$PKG")
  if [ "$code" = 200 ]; then
    ok "A1 package: $PKG filled from the SHARD's record ($(stat -c%s "$W/pkg.bin") B) with no classic document in the mirror"
  else bad "A1 package: $PKG code=$code (see $W/pkgfetch-$P1.log)"; fi
else bad "A1 package: could not name a package out of the served shard"; fi

code=$(curl -s -o /dev/null -w '%{http_code}' "$U/$SLUG/$SUB/repodata.json")
if [ "$code" = 404 ] && grep -q "CLASSIC-DOWNGRADE $SLUG/$SUB" "$W/pkgfetch-$P1.log"; then
  ok "A1 downgrade: the SHARDED pair has no classic document (404) and the attempt is a named row"
else bad "A1 downgrade: code=$code, CLASSIC-DOWNGRADE row present=$(grep -c CLASSIC-DOWNGRADE "$W/pkgfetch-$P1.log")"; fi

code=$(curl -s -o /dev/null -w '%{http_code}' "$U/prefix.dev__robostack-humble/noarch/repodata.json")
if [ "$code" = 200 ]; then ok "A1 classic pair: robostack-humble/noarch (empty shard index upstream) is still served classically"
else bad "A1 classic pair: robostack-humble/noarch repodata.json code=$code"; fi

# ---------------------------------------------------------------- ARM 2 ----
GHOST=0000000000000000000000000000000000000000000000000000000000000042
code=$(curl -s -o /dev/null -w '%{http_code}' "$U/$(dirname "$SHPATH")/$GHOST.msgpack.zst")
if [ "$code" != 200 ] && grep -q "NOSHARD .*$GHOST" "$W/pkgfetch-$P1.log"; then
  ok "A2 mutation: a shard sha the frozen index does not name is refused ($code) with a NOSHARD row"
else bad "A2 mutation: a foreign shard sha answered $code (NOSHARD rows=$(grep -c NOSHARD "$W/pkgfetch-$P1.log"))"; fi

# ---------------------------------------------------------------- ARM 3 ----
# A local upstream that answers <real sha>.msgpack.zst with a DIFFERENT shard's
# bytes.  The mirror must compute the sha of what it got and refuse.
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
UPD=$W/fake-upstream/conda-forge
mkdir -p "$UPD"
SH2=$(python3 "$HERE/shard_mirror.py" probe "$MIRROR" "$SLUG" noarch 2>/dev/null | awk -F'\t' '$1=="shard_sha"{print $2}')
SH2PATH=$(python3 "$HERE/shard_mirror.py" probe "$MIRROR" "$SLUG" noarch 2>/dev/null | awk -F'\t' '$1=="shard_path"{print $2}')
cp "$W/shard.bin" "$UPD/$SH2.msgpack.zst"      # linux-64's bytes under noarch's name
UP=$((PB+1))
( cd "$W/fake-upstream" && exec python3 -m http.server "$UP" --bind 127.0.0.1 ) > "$W/upstream.log" 2>&1 &
UPS=$!
for i in 1 2 3 4 5; do curl -fs -o /dev/null "http://127.0.0.1:$UP/" && break; sleep 1; done
python3 "$HERE/shard_mirror.py" set-shards-base "$MIRROR" "$SLUG" noarch "http://127.0.0.1:$UP/conda-forge/" | sed 's/^/  /'
rm -f "$MIRROR/$SH2PATH"
P2=$((PB+2)); serve "$P2" || { echo "GUARD FATAL server never came up on $P2"; exit 2; }
code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$P2/$SH2PATH")
if [ "$code" != 200 ] && grep -q 'SHARD-SHA-MISMATCH' "$W/pkgfetch-$P2.log"; then
  ok "A3 mutation: a shard whose CONTENT does not hash to its name is refused ($code) with a SHARD-SHA-MISMATCH row"
else bad "A3 mutation: tampered shard answered $code (mismatch rows=$(grep -c 'SHARD-SHA-MISMATCH' "$W/pkgfetch-$P2.log"))"; fi
kill "$SRV" ${UPS:-} 2>/dev/null; wait "$SRV" 2>/dev/null

# ---------------------------------------------------------------- ARM 4 ----
# Restore the state file the mutation above rewrote, then move ONE BYTE of a
# frozen index and require the server to refuse to start.
retread_freeze_shard_mirror "$W/fixture.lock" "$MIRROR" > /dev/null || { echo "GUARD FATAL re-freeze refused"; exit 2; }
printf 'x' >> "$MIRROR/$SLUG/$SUB/repodata_shards.msgpack.zst"
P3=$((PB+3))
python3 "$HERE/channel_mirror_server.py" "$MIRROR" "$P3" "$W/pkgs-$P3" > "$W/httpd-$P3.log" 2>&1 &
SRV=$!
sleep 3
if kill -0 "$SRV" 2>/dev/null && curl -fs -o /dev/null "http://127.0.0.1:$P3/"; then
  bad "A4 mutation: the server started over a frozen index that had MOVED on disk"
  kill "$SRV" 2>/dev/null
else
  if grep -q 'moved: sha' "$W/httpd-$P3.log"; then
    ok "A4 mutation: an appended byte in a frozen index makes the server REFUSE to start"
  else
    bad "A4 mutation: the server did not start, but not for the stated reason:"; sed 's/^/    /' "$W/httpd-$P3.log" | tail -5
  fi
fi
wait "$SRV" 2>/dev/null

# ---------------------------------------------------------------- ARM 5 ----
retread_freeze_shard_mirror "$W/fixture.lock" "$MIRROR" > /dev/null || { echo "GUARD FATAL re-freeze refused"; exit 2; }
P4=$((PB+4)); serve "$P4" || { echo "GUARD FATAL server never came up on $P4"; exit 2; }
RETREAD_MIRROR_URL=http://127.0.0.1:$P4
RETREAD_MIRROR_ACCESS_LOG=$W/httpd-$P4.log
RETREAD_SHARD_MIRROR=$MIRROR
export RETREAD_MIRROR_URL RETREAD_MIRROR_ACCESS_LOG RETREAD_SHARD_MIRROR
mkdir -p "$W/ws"
cat > "$W/ws/pixi.toml" <<'TOML'
[workspace]
name = "p6af2g-guard"
channels = ["https://prefix.dev/conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = "3.11.*"
TOML

pixi_arm () {   # $1 name  $2 mirror|nomirror
  local a=$1 m=$2 rc t0 H
  # NOT one `local` line: bash declares every name in a `local` statement before
  # it assigns any of them, so `H=$W/home-$a` on the same line reads a declared
  # but unset `a` and dies under `set -u`.
  H=$W/home-$a
  rm -rf "$H" "$W/cache-$a" "$W/ws/pixi.lock" "$W/ws/.pixi"
  mkdir -p "$H/.pixi" "$W/cache-$a"
  # PER ARM, never once at the top: pixi reads its global config out of
  # PIXI_HOME/XDG_CONFIG_HOME and NOT out of `$HOME/.pixi` (job 5851478 --
  # the config was written, every mirror base answered 200, and the lock still
  # went straight to prefix.dev).  Pointing both at the arm's own home is what
  # keeps the no-mirror arm genuinely un-mirrored.
  export PIXI_HOME=$H/.pixi XDG_CONFIG_HOME=$H/.config
  if [ "$m" = mirror ]; then
    HOME=$H retread_pixi_shard_mirror_config "$W/fixture.lock" "$H" "$PIXI" \
      > "$W/config-$a.log" 2>&1 \
      || { echo "GUARD FATAL config refused"; sed 's/^/    /' "$W/config-$a.log"; return 90; }
    sed 's/^/  config /' "$H/.pixi/config.toml"
  else
    : > "$H/.pixi/config.toml"
  fi
  t0=$(date +%s)
  ( cd "$W/ws" && env HOME="$H" PIXI_HOME="$H/.pixi" XDG_CONFIG_HOME="$H/.config" \
      PIXI_CACHE_DIR="$W/cache-$a" \
      NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
      HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9 \
      https_proxy=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 all_proxy=http://127.0.0.1:9 \
      PIXI_SHIM_NO_STALE_CHECK=1 "$PIXI" lock ) > "$W/$a.log" 2>&1
  rc=$?
  [ -f "$W/ws/pixi.lock" ] && cp "$W/ws/pixi.lock" "$W/pixi.lock.$a"
  echo "### GUARD pixi arm=$a mirror=$m rc=$rc wall=$(( $(date +%s)-t0 ))s"
  return $rc
}

pixi_arm A0_nomirror nomirror
if [ $? -eq 0 ]; then bad "A0 non-vacuity: a lock SUCCEEDED with the network dead and no mirror"
else ok "A0 non-vacuity: with the network dead and no mirror the lock is refused"; fi

LINES0=$(wc -l < "$RETREAD_MIRROR_ACCESS_LOG")
if pixi_arm A5_mirror mirror; then
  ok "A5 live pixi: an offline lock against the SHARD mirror succeeded"
else bad "A5 live pixi: the lock failed -- tail:"; tail -15 "$W/A5_mirror.log" | sed 's/^/    /'; fi
tail -n +$((LINES0+1)) "$RETREAD_MIRROR_ACCESS_LOG" > "$W/A5.access.log"
SH200=$(grep -cE '"GET /[^"]*/[0-9a-f]{64}\.msgpack\.zst HTTP[^"]*" 200' "$W/A5.access.log")
IDX200=$(grep -cE '"GET /[^"]*repodata_shards\.msgpack\.zst HTTP[^"]*" 200' "$W/A5.access.log")
CLS=$(grep -cE '"GET /prefix\.dev__conda-forge/[^"]*/repodata\.json HTTP' "$W/A5.access.log")
echo "### GUARD A5 access log: shard 200=$SH200 shard-index 200=$IDX200 conda-forge repodata.json requests=$CLS"
[ "$IDX200" -gt 0 ] && ok "A5 protocol: pixi fetched the shard INDEX from the mirror ($IDX200 pairs)" \
                    || bad "A5 protocol: pixi never fetched a shard index from the mirror"
[ "$SH200" -gt 0 ]  && ok "A5 protocol: pixi fetched $SH200 SHARDS from the mirror" \
                    || bad "A5 protocol: pixi fetched no shards -- it is not speaking the sharded protocol here"
[ "$CLS" -eq 0 ]    && ok "A5 protocol: ZERO classic-document reads for conda-forge -- no downgrade" \
                    || bad "A5 protocol: $CLS classic repodata.json request(s) for conda-forge -- pixi downgraded"
RE=$(grep -c 'run_exports' "$W/pixi.lock.A5_mirror" 2>/dev/null); [ -n "$RE" ] || RE=0
[ "$RE" -gt 0 ] && ok "A5 run_exports: the shard-mirror lock carries $RE run_exports blocks, from the shards themselves" \
                || bad "A5 run_exports: 0 blocks in the lock"
retread_assert_shard_mirror_clean "$W/A5.access.log" "$MIRROR" "$W/pkgfetch-$P4.log" || bad "A5 gate: retread_assert_shard_mirror_clean refused"

echo "### GUARD VERDICT $([ $fail -eq 0 ] && echo PASS || echo FAIL)"
exit $fail
