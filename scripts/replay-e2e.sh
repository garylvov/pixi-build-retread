#!/usr/bin/env bash
# Phase 2.6 empirical seal: lukewarm replay e2e for the example packs.
# For each pack: COLD produce (schema-9 lock, no lock present) -> save ->
# LUKEWARM nuke (caches + wheels + envs, KEEP .pixi/config.toml + the lock)
# -> replay -> assert replay fires + lock byte-identical + import works.
#
# Run from the retread repo root. Uses the LOCAL v2.7.0 backend (the example
# packs pin file:///.../local-channel + version="*").
set -uo pipefail

REPO=/home/garylvov/projects/pixi-build-retread
PIXI=/home/garylvov/.pixi/bin/pixi
LOG=/tmp/p26
mkdir -p "$LOG"
cd "$REPO"

# ---- rebuild local v2.7.0 backend once ----
echo "=== REBUILD local backend (v2.7.0) ==="
bash scripts/rebuild-local.sh >"$LOG/rebuild.log" 2>&1
echo "rebuild rc=$? ; advertised: $(grep -o 'pixi-build-retread-[0-9.]*' local-channel/linux-64/repodata.json | sort -u | tr '\n' ' ')"

nuke_caches() {
  rm -rf ~/.cache/uv ~/.cache/retread \
         ~/.cache/rattler/cache/retread-probes \
         ~/.cache/rattler/cache/retread-repodata \
         ~/.cache/rattler/cache/retread-git-clones \
         ~/.cache/rattler/cache/bld 2>/dev/null
}

# run_pack <name> <consumer_dir> <env> <pack_subdir> <lock_name> <import_py>
run_pack() {
  local name="$1" consumer="$2" env="$3" packsub="$4" lockname="$5" importpy="$6"
  local packdir="$consumer/$packsub"
  local lock="$packdir/$lockname"
  echo ""
  echo "############################################################"
  echo "### PACK: $name  (consumer=$consumer env=$env)"
  echo "############################################################"

  # ---------- COLD produce ----------
  echo "--- [$name] COLD produce ---"
  nuke_caches
  rm -rf "$consumer/.pixi/envs" "$consumer/.pixi/bld" "$consumer/.pixi/meta-v0" \
         "$consumer/.pixi/artifacts-v0" 2>/dev/null
  rm -rf "$packdir/wheels" 2>/dev/null
  rm -f "$lock" 2>/dev/null
  ( cd "$consumer" && OMNI_KIT_ACCEPT_EULA=YES "$PIXI" install -e "$env" ) \
      >"$LOG/${name}_cold.log" 2>&1
  local coldrc=$?
  if [[ ! -f "$lock" ]]; then
    echo "  [$name] COLD FAILED rc=$coldrc -- no lock produced. tail:"; tail -8 "$LOG/${name}_cold.log"; return 1
  fi
  local coldschema; coldschema=$(grep -o '"schema"[: ]*[0-9]*' "$lock" | head -1 | grep -o '[0-9]*')
  local gitwheels; gitwheels=$(grep -c '"git_source"' "$lock")
  local sdistwheels; sdistwheels=$(grep -c '"sdist_source"' "$lock")
  echo "  [$name] cold rc=$coldrc | schema=$coldschema | git_source=$gitwheels | sdist_source=$sdistwheels"
  cp "$lock" "$LOG/${name}.cold.lock.json"

  # ---------- LUKEWARM replay ----------
  echo "--- [$name] LUKEWARM replay ---"
  nuke_caches
  rm -rf "$consumer/.pixi/envs" "$consumer/.pixi/bld" "$consumer/.pixi/meta-v0" \
         "$consumer/.pixi/artifacts-v0" 2>/dev/null
  rm -rf "$packdir/wheels" 2>/dev/null
  local wheels_before; wheels_before=$(find "$packdir/wheels" -name '*.whl' 2>/dev/null | wc -l)
  ( cd "$consumer" && OMNI_KIT_ACCEPT_EULA=YES "$PIXI" install -e "$env" ) \
      >"$LOG/${name}_warm.log" 2>&1
  local warmrc=$?
  local wheels_after; wheels_after=$(find "$packdir/wheels" -name '*.whl' 2>/dev/null | wc -l)

  # ---------- assertions ----------
  local replay_hit; replay_hit=$(grep -ciE "build_v1.*(replay|replayed|re-materializ)|replayed from lock" "$LOG/${name}_warm.log")
  local derive_ab; derive_ab=$(grep -ciE "auto-bundled" "$LOG/${name}_warm.log")
  local derive_solve; derive_solve=$(grep -ciE "resolvo solve finished" "$LOG/${name}_warm.log")
  local shared_err; shared_err=$(grep -ciE "shares a git checkout" "$LOG/${name}_warm.log")
  local byteid="NO"
  if diff -q "$LOG/${name}.cold.lock.json" "$lock" >/dev/null 2>&1; then byteid="YES"; fi

  echo "  [$name] warm rc=$warmrc | replay_hit=$replay_hit | auto-bundled=$derive_ab | resolvo-solve=$derive_solve | shared-checkout-err=$shared_err"
  echo "  [$name] wheels before=$wheels_before after=$wheels_after"
  echo "  *** [$name] LOCK BYTE-IDENTICAL: $byteid ***"
  if [[ "$byteid" == "NO" ]]; then
    echo "  [$name] DIFF (cold vs replay):"; diff "$LOG/${name}.cold.lock.json" "$lock" | head -40
  fi

  # ---------- import ----------
  local imp="SKIP"
  if [[ -n "$importpy" ]]; then
    if ( cd "$consumer" && OMNI_KIT_ACCEPT_EULA=YES "$PIXI" run -e "$env" python -c "$importpy" ) >"$LOG/${name}_import.log" 2>&1; then
      imp="OK"
    else
      imp="FAIL"; echo "  [$name] import tail:"; tail -5 "$LOG/${name}_import.log"
    fi
  fi
  echo "  [$name] import: $imp"
  echo "SUMMARY[$name]: cold_schema=$coldschema git=$gitwheels sdist=$sdistwheels replay_hit=$replay_hit derive=$((derive_ab+derive_solve)) shared_err=$shared_err byteid=$byteid import=$imp wheels=$wheels_before->$wheels_after"
}

# genesis (light, git-source regression), isaac6 (index regression),
# gigastrap-isaac (HEAVY, decisive gym sdist fix + grizzly acceptance target)
run_pack genesis  "$REPO/examples/genesis"   default     genesis-pack retread-genesis-pack.lock.json  "import genesis"
run_pack isaac6   "$REPO/examples/isaac6"    gpu         isaac-pack   retread-isaac-pack-6.lock.json  "import isaacsim"
run_pack isaac    "$REPO/examples/gigastrap" isaaclab-gpu isaac-pack  retread-isaac-pack.lock.json    "import isaacsim; import isaaclab"

echo ""
echo "=== ALL DONE ==="
grep '^SUMMARY' "$LOG"/*.log 2>/dev/null
