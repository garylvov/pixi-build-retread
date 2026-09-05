#!/usr/bin/env bash
# p6af-2g: the SHARD-INDEX channel mirror -- the shell half.
#
# WHY IT IS A SEPARATE FILE AND NOT MORE OF `retread_fast_env.sh`.
# `agrescap/worktrees/harness-tools` is ONE worktree on ONE shared branch and
# several lanes commit into it at once; `retread_fast_env.sh` is the file every
# one of them touches, and C31-4-1 is reconciling its task-directory copy while
# this is being written. A new capability that can live in its own file does.
# Source it AFTER `retread_fast_env.sh` -- `retread_pixi_shard_mirror_config`
# calls `retread_pixi_mirror_config` rather than re-implementing it.
#
# WHAT IT IS FOR, in one paragraph, because the sizing is the argument.
# p6af's frozen mirror serves the CLASSIC `repodata.json` -- 1 229 375 784 B for
# the 21 pairs this workspace declares -- and its `run_exports` are merged in
# from whatever shards a pixi cache happened to hold. p6af-2a measured that
# coverage at 1247 of conda-forge/linux-64's 13956 names and `env_version_delta`
# moved 16 rows; p6af-2e re-ran it with the coverage complete for THAT solve and
# the same 16 rows moved, which killed the coverage explanation. The surviving
# one is protocol shape: pixi asks for `repodata_shards.msgpack.zst`, the mirror
# 404s it, pixi falls back to a document that cannot carry `run_exports` at all.
# MEASURED (probe 5870870): the shard INDEXES for all 21 pairs are 2 023 293 B
# -- 1.93 MiB against 1.15 GiB -- and passing the shards themselves through
# lazily, each verified against the sha256 the frozen index names, makes the
# served universe complete BY CONSTRUCTION instead of by luck.
#
#     retread_freeze_shard_mirror       <lock> <dst mirror root>
#     retread_pixi_shard_mirror_config  <lock> <job HOME> [pixi binary]
#     retread_assert_shard_mirror_clean <access log> <mirror root>
#
# The server is the SAME `retread_serve_channel_mirror`: `channel_mirror_server.py`
# arms its sharded half off `INDEX-STATE.json` in the mirror root, so a p6af-shaped
# static mirror is byte-for-byte unaffected and its guard still passes.

######## retread_freeze_shard_mirror ##########################################
# Fetches one frozen shard index per (channel, subdir) FROM THE CHANNEL, stores
# it VERBATIM, and records its sha256. Verbatim matters: the served index is
# then the document prefix.dev published, its sha256 is prefix.dev's sha256, and
# no msgpack ENCODER has to be written and trusted for `channel_relations`, for
# the two `/pytorch` pairs whose shas are msgpack ARRAYS, or for the
# `packages.whl` key rattler 0.25.5 has no field for.
#
# THE NETWORK IS STILL OPEN WHEN THIS RUNS, by construction: it is called before
# the deny proxy is armed, exactly where `retread_freeze_channel_mirror` is.
retread_freeze_shard_mirror () {
  local lock=${1:-} dst=${2:-}
  if [ -z "$lock" ] || [ -z "$dst" ]; then
    echo "retread_freeze_shard_mirror: usage: <lock> <dst mirror root>" >&2; return 2
  fi
  [ -f "$lock" ] || { echo "retread_freeze_shard_mirror: FATAL no lock at $lock" >&2; return 2; }
  case "$dst" in
    /oscar/data/stellex/glvov/agrescap/cache/retread/*)
      echo "retread_freeze_shard_mirror: REFUSING to write inside the shared persistent cache: $dst" >&2
      return 2;;
  esac
  local here; here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local py=$here/shard_mirror.py
  [ -f "$py" ] || { echo "retread_freeze_shard_mirror: FATAL missing $py" >&2; return 2; }
  local t0; t0=$(date +%s)
  python3 "$py" freeze "$lock" "$dst" || {
    echo "retread_freeze_shard_mirror: FATAL freeze refused" >&2; return 3; }
  # ONE digest over the frozen indexes, computed the same way every time: the
  # sha256 of the sorted "<relative path> <sha256>" table. A digest over a
  # `find` order would not be a comparison rule at all. Same shape as
  # RETREAD_CHANNEL_MIRROR_DIGEST so the two are readable side by side.
  local digest
  digest=$( (cd "$dst" && find . -maxdepth 3 -name repodata_shards.msgpack.zst -printf '%P\n' |
               sort | while read -r p; do echo "$p $(sha256sum "$p" | cut -d' ' -f1)"; done) |
            sha256sum | cut -d' ' -f1)
  echo "### shard_mirror frozen dst=$dst wall=$(( $(date +%s)-t0 ))s digest=$digest"
  export RETREAD_SHARD_MIRROR=$dst
  export RETREAD_SHARD_MIRROR_DIGEST=$digest
  return 0
}

######## retread_pixi_shard_mirror_config #####################################
# THE ONE THING THE SHARDED PROTOCOL NEEDS THAT THE CLASSIC ONE DID NOT.
# MEASURED (probe 5870870): prefix.dev publishes
#     shards_base_url = "https://shards.prefix.dev/<channel>/"
# -- an ABSOLUTE url on a DIFFERENT HOST -- and
# `rattler_networking::mirror_middleware::MirrorMiddleware::handle` maps by
# STRING PREFIX against the configured channel key. So a shard url does not
# begin with `https://prefix.dev/conda-forge` and would never reach the loopback
# mirror: pixi would fetch every shard straight from `shards.prefix.dev`, a host
# the p6af-2a/2e deny list does not name, and the run would look offline while
# being anything but. This function adds `https://shards.prefix.dev/<channel>`
# to `[mirrors]` as a key of its own, derived from the FROZEN INDEXES rather
# than hard-coded, so a channel that moves its shards is picked up by the freeze
# and not by a human remembering to.
#
# The channel half is `retread_pixi_mirror_config`'s, called here and not
# duplicated; this appends to what it wrote, in every destination it wrote to,
# and then re-runs both of its readers over the result.
retread_pixi_shard_mirror_config () {
  local lock=${1:-} jobhome=${2:-} pixibin=${3:-}
  [ -n "${RETREAD_SHARD_MIRROR:-}" ] || {
    echo "retread_pixi_shard_mirror_config: FATAL RETREAD_SHARD_MIRROR unset -- freeze first" >&2
    return 2; }
  retread_pixi_mirror_config "$lock" "$jobhome" "$pixibin" || return $?
  local here; here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local dests="$jobhome/.pixi/config.toml"
  [ -n "${PIXI_HOME:-}" ] && case "$PIXI_HOME" in /users/glvov|/users/glvov/*) ;; *) dests="$dests $PIXI_HOME/config.toml";; esac
  [ -n "${XDG_CONFIG_HOME:-}" ] && case "$XDG_CONFIG_HOME" in /users/glvov|/users/glvov/*) ;; *) dests="$dests $XDG_CONFIG_HOME/pixi/config.toml";; esac
  local added=0 line kind url slug dst
  while IFS=$'\t' read -r kind url slug; do
    [ "$kind" = shards ] || continue
    line="\"$url\" = [\"$RETREAD_MIRROR_URL/$slug\"]"
    for dst in $dests; do
      grep -qxF "$line" "$dst" || printf '%s\n' "$line" >> "$dst"
    done
    added=$((added+1))
    echo "### shard_mirror config key $url -> $RETREAD_MIRROR_URL/$slug"
  done < <(python3 "$here/shard_mirror.py" keys "$RETREAD_SHARD_MIRROR")
  if [ "$added" -eq 0 ]; then
    # Not an error by itself -- a channel whose shards_base_url is relative
    # needs no extra key -- but it is exactly the shape of a silent no-op, so it
    # is a named row rather than silence.
    echo "### shard_mirror config: no separate shard host; every shards_base_url sits under a channel key"
  fi
  # READER ONE: every base the config now names must be served.
  local bad=0 base
  for base in $(sed -n 's/.*= \["\([^"]*\)"\].*/\1/p' "$jobhome/.pixi/config.toml"); do
    if curl -fs -o /dev/null "$base/"; then
      echo "### shard_mirror config check 200 $base/"
    else
      echo "retread_pixi_shard_mirror_config: FATAL mirror base not served: $base/" >&2
      bad=1
    fi
  done
  [ "$bad" -eq 0 ] || return 3
  # READER TWO: the only authority on what pixi reads is pixi.
  if [ -n "$pixibin" ] && [ -x "$pixibin" ]; then
    local listed; listed=$("$pixibin" config list 2>&1)
    printf '%s\n' "$listed" | sed 's/^/  pixi config list: /'
    if [ "$added" -gt 0 ] && ! printf '%s' "$listed" | grep -q 'shards\.'; then
      echo "retread_pixi_shard_mirror_config: FATAL pixi does not see the shard-host mirror keys -- every shard would go straight to the channel's shard host" >&2
      return 4
    fi
  fi
  return 0
}

######## retread_assert_shard_mirror_clean ####################################
# THE READER FOR THE WHOLE LANE, and it asserts the POSITIVE, not the absence of
# an error. Over the access log of a solve against a shard mirror:
#   * every pair the state calls `sharded` must have been answered 200 for its
#     `repodata_shards.msgpack.zst`;
#   * ZERO `repodata.json` reads for those pairs -- a single one is pixi
#     downgrading to the classic document, which is the exact defect this lane
#     exists to remove, and the mirror does not even hold that file;
#   * shard requests are COUNTED and their bytes summed, because "the mirror was
#     used" is a number here and not an adjective;
#   * a `NOSHARD`, `SHARD-SHA-MISMATCH` or `SHARD-MISPLACED` row anywhere is red.
#
#     retread_assert_shard_mirror_clean <access log> <mirror root> [pkgfetch log]
retread_assert_shard_mirror_clean () {
  local log=${1:-} root=${2:-} fetchlog=${3:-}
  [ -f "$log" ] || { echo "retread_assert_shard_mirror_clean: FATAL no access log at $log" >&2; return 2; }
  [ -f "$root/INDEX-STATE.json" ] || {
    echo "retread_assert_shard_mirror_clean: FATAL no INDEX-STATE.json under $root" >&2; return 2; }
  local bad=0
  # The status is read from the field AFTER the quoted request line, never by
  # grepping a number anywhere on the line -- a path can contain `200`.
  local prog='function base(p,  n,a){n=split(p,a,"/"); return a[n]}
    /"(GET|HEAD) /{split($2,r," "); split($3,c," "); b=base(r[2]);
      if (b ~ /^[0-9a-f]{64}\.msgpack\.zst$/) { if (c[1]=="200") sh++; else shbad++; next }
      if (b=="repodata_shards.msgpack.zst") { if (c[1]=="200") idx200[r[2]]++; else idx404[r[2]]++; next }
      if (b=="repodata.json") { if (c[1]=="200") cls200[r[2]]++; else cls404[r[2]]++ } }
    END{ printf "SHARD200 %d\nSHARDBAD %d\n", sh+0, shbad+0
         for (k in idx200) printf "IDX200 %s %d\n", k, idx200[k]
         for (k in idx404) printf "IDX404 %s %d\n", k, idx404[k]
         for (k in cls200) printf "CLS200 %s %d\n", k, cls200[k]
         for (k in cls404) printf "CLS404 %s %d\n", k, cls404[k] }'
  local out; out=$(awk -F'"' "$prog" "$log")
  local sh200 shbad
  sh200=$(printf '%s\n' "$out" | sed -n 's/^SHARD200 //p')
  shbad=$(printf '%s\n' "$out" | sed -n 's/^SHARDBAD //p')
  echo "### shard_mirror_gate shard requests 200=$sh200 non-200=$shbad"
  printf '%s\n' "$out" | sed -n 's/^IDX200 /###   shard index 200 /p' | sort
  printf '%s\n' "$out" | sed -n 's/^IDX404 /###   shard index 404 /p' | sort
  # Which pairs the state calls sharded, straight out of the state file -- the
  # gate must not carry its own idea of that.
  local pairs; pairs=$(python3 -c '
import json,sys
st=json.load(open(sys.argv[1]))
for r in st["pairs"]:
    print("%s\t%s/%s" % (r["mode"], r["slug"], r["subdir"]))' "$root/INDEX-STATE.json")
  local mode pair n
  while IFS=$'\t' read -r mode pair; do
    [ "$mode" = sharded ] || continue
    n=$(printf '%s\n' "$out" | grep -c "^IDX200 /$pair/repodata_shards.msgpack.zst ")
    if [ "$n" -eq 0 ]; then
      # Not fatal on its own: a pair the solve never touches is never asked for.
      echo "###   pair $pair sharded, index NOT requested by this solve"
    fi
    if printf '%s\n' "$out" | grep -q "^CLS[0-9]* /$pair/repodata.json "; then
      echo "### shard_mirror_gate FAIL: pixi asked for the CLASSIC document of SHARDED pair $pair -- it downgraded" >&2
      bad=1
    fi
  done <<< "$pairs"
  if [ "${shbad:-0}" -ne 0 ]; then
    echo "### shard_mirror_gate FAIL: $shbad shard request(s) did not answer 200" >&2
    bad=1
  fi
  if [ -n "$fetchlog" ] && [ -f "$fetchlog" ]; then
    local refused
    refused=$(grep -c -e NOSHARD -e SHARD-SHA-MISMATCH -e SHARD-MISPLACED -e SHARDFETCH-FAIL "$fetchlog")
    local fetched bytes
    fetched=$(grep -c '^\[.*\] SHARDFETCH ' "$fetchlog")
    bytes=$(awk '/ SHARDFETCH /{for(i=1;i<=NF;i++) if ($i ~ /^bytes=/){sub("bytes=","",$i); s+=$i}} END{print s+0}' "$fetchlog")
    echo "### shard_mirror_gate shards fetched=$fetched bytes=$bytes refusals=$refused"
    if [ "$refused" -ne 0 ]; then
      echo "### shard_mirror_gate FAIL: $refused shard refusal row(s) in $fetchlog" >&2
      grep -e NOSHARD -e SHARD-SHA-MISMATCH -e SHARD-MISPLACED -e SHARDFETCH-FAIL "$fetchlog" |
        head -20 | sed 's/^/###   /' >&2
      bad=1
    fi
  fi
  if [ "$bad" -eq 0 ]; then
    echo "### shard_mirror_gate PASS: the sharded protocol was served and nothing downgraded"
    return 0
  fi
  return 1
}
