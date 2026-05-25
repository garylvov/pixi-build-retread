#!/usr/bin/env bash
# Nuke every cache that turns "edit code -> rebuild -> still see old error"
# into a guaranteed trap, rebuild the local conda channel, and verify the
# new version is what gets advertised. Run this in the retread repo root
# after bumping `Cargo.toml` + `recipe/recipe.yaml` to a new version.
#
# Usage:
#   bash scripts/rebuild-local.sh
#
# Caches this nukes (and WHY each one matters -- the README's "Iteration
# gotchas" and HANDOFF's invariant #5 explain in more depth):
#   - local-channel/linux-64/pixi-build-retread-*.conda
#     The previous artifact bytes. rattler-build is happy to leave these.
#   - local-channel/linux-64/repodata.json
#     rattler-build APPENDS to this; deletion forces full regen. Without
#     this, the channel keeps advertising the old version to pixi.
#   - ~/.cache/rattler/cache/backends-v0/pixi-build-retread-*
#     Where pixi caches the retread EXECUTABLE keyed by build hash. If
#     the hash collides (or pixi misses the version bump), pixi reuses
#     the old binary even after the channel advertises a new one.
#   - ~/.cache/rattler/cache/retread-git-clones
#     retread's clone cache. Stale after any layout change (v0.13.3
#     moved <slug>-<rev>/ to <slug>/<sha12>/), or after a half-broken
#     checkout (ENAMETOOLONG, network blip). Cheap to redo -- shallow
#     clones with --filter=blob:none.
#   - $CONSUMER_PROJECT/.pixi/{meta-v0,bld}/isaac* (if CONSUMER_PROJECT
#     env var is set), plus the rattler-cache bld/{metadata,source_metadata}
#     for isaac. Project-local pixi caches that hold built-output metadata
#     keyed by the old retread emission.
#
# What this script does NOT delete (intentionally):
#   - local-channel/noarch/repodata.json -- empty placeholder that
#     rattler-build still scans during build-env resolution. Deleting
#     it produces `could not find subdir 'noarch'`. The script will
#     recreate it if missing (one-time recovery).

set -euo pipefail

# Resolve repo root from the script location, so the script works from
# any cwd (and any user, since no absolute path is hardcoded).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_CHANNEL="$REPO_ROOT/local-channel"
RECIPE="$REPO_ROOT/recipe/recipe.yaml"
VARIANT_CONFIG="$REPO_ROOT/recipe/variants.yaml"

# Find rattler-build (and implicitly cargo). Strategies, in order:
#   1. rattler-build already on PATH  -> use whatever's there
#   2. RETREAD_TOOLS_PATH env var set -> use that bin dir
#   3. `pixi exec --spec rattler-build`-equivalent: ask pixi to give us
#      a transient env that has it. Works on any machine that has pixi
#      installed (which is implied if you're building a pixi backend),
#      no source-checkout assumption.
# Last resort: helpful error pointing at the two override knobs.
if command -v rattler-build >/dev/null 2>&1; then
    : # already on PATH, nothing to do
elif [[ -n "${RETREAD_TOOLS_PATH:-}" && -x "$RETREAD_TOOLS_PATH/rattler-build" ]]; then
    export PATH="$RETREAD_TOOLS_PATH:$PATH"
elif command -v pixi >/dev/null 2>&1; then
    # `pixi global bin-dir` returns the dir where pixi-global tools live.
    # If the user installed rattler-build globally (`pixi global install
    # rattler-build`), it's on this PATH.
    PIXI_BIN="$(pixi global bin-dir 2>/dev/null || true)"
    if [[ -n "$PIXI_BIN" && -x "$PIXI_BIN/rattler-build" ]]; then
        export PATH="$PIXI_BIN:$PATH"
    else
        echo "[rebuild-local] pixi is installed but rattler-build is not on PATH and not in pixi global." >&2
        echo "[rebuild-local]   install it once:  pixi global install rattler-build" >&2
        echo "[rebuild-local]   (or set RETREAD_TOOLS_PATH=/abs/bin/dir if you have it elsewhere)" >&2
        exit 1
    fi
else
    echo "[rebuild-local] ERROR: rattler-build not found and no pixi to bootstrap it." >&2
    echo "[rebuild-local]   put rattler-build on PATH, or:" >&2
    echo "[rebuild-local]     pixi global install rattler-build" >&2
    echo "[rebuild-local]   alternatively, set RETREAD_TOOLS_PATH=/abs/bin/dir if you have it elsewhere." >&2
    exit 1
fi

# Pull the version straight out of Cargo.toml -- single source of truth.
# (Recipe must match; the verify step at the end catches divergence.)
VERSION="$(grep -m1 '^version = ' "$REPO_ROOT/Cargo.toml" | sed -E 's/version = "([^"]+)"/\1/')"
RECIPE_VERSION="$(grep -m1 '  version: ' "$RECIPE" | sed -E 's/.*version: "([^"]+)".*/\1/')"
if [[ "$VERSION" != "$RECIPE_VERSION" ]]; then
    echo "[rebuild-local] ERROR: Cargo.toml version ($VERSION) != recipe.yaml version ($RECIPE_VERSION)" >&2
    echo "[rebuild-local]   bump both to the same value before rebuilding" >&2
    exit 1
fi
echo "[rebuild-local] building pixi-build-retread $VERSION"

# (1) Nuke local-channel artifacts + stale repodata. PRESERVE noarch/.
echo "[rebuild-local] nuking local-channel artifacts + linux-64 repodata"
rm -f  "$LOCAL_CHANNEL"/linux-64/pixi-build-retread-*.conda \
       "$LOCAL_CHANNEL"/linux-64/repodata.json

# (2) Ensure noarch placeholder exists -- rattler-build requires it.
if [[ ! -f "$LOCAL_CHANNEL/noarch/repodata.json" ]]; then
    echo "[rebuild-local] noarch/repodata.json missing; recreating placeholder"
    mkdir -p "$LOCAL_CHANNEL/noarch"
    echo '{"info":{"subdir":"noarch"},"packages":{},"packages.conda":{},"repodata_version":2}' \
        > "$LOCAL_CHANNEL/noarch/repodata.json"
fi

# (3) Nuke the global pixi backend cache (cached executable).
echo "[rebuild-local] nuking pixi backend cache"
rm -rf ~/.cache/rattler/cache/backends-v0/pixi-build-retread-*

# (4) Nuke retread's own git-clone cache (stale after layout changes).
echo "[rebuild-local] nuking retread git-clone cache"
rm -rf ~/.cache/rattler/cache/retread-git-clones

# (4b) Nuke retread's probe + repodata caches. v0.22.0+ caches conda
#      repodata.json[.zst] per (channel, subdir) under retread-repodata;
#      the older v0.13-v0.21 path stored per-package probe results
#      under retread-probes. Both 30-min TTL; nuking them on every
#      rebuild forces probes to re-query upstream so the new version's
#      probe logic isn't masked by a result computed by the old logic.
echo "[rebuild-local] nuking retread probe + repodata caches"
rm -rf ~/.cache/rattler/cache/retread-probes \
       ~/.cache/rattler/cache/retread-repodata

# (5) Nuke the consumer project's pixi caches AND retread audit/trace
#     files if pointed at one. Skip silently when CONSUMER_PROJECT
#     isn't set -- not everyone has a downstream workspace they're
#     iterating against. The audit + trace deletion is what makes
#     `cat retread-probe-trace-*.json | head -3` a reliable check
#     of "did this run actually invoke retread?" -- without nuking,
#     stale files from earlier runs masquerade as fresh data.
if [[ -n "${CONSUMER_PROJECT:-}" ]]; then
    echo "[rebuild-local] nuking consumer caches under $CONSUMER_PROJECT"
    rm -rf "$CONSUMER_PROJECT/.pixi/meta-v0/"isaac* \
           "$CONSUMER_PROJECT/.pixi/bld/"isaac* || true
    rm -rf ~/.cache/rattler/cache/bld/metadata-v0/isaac* \
           ~/.cache/rattler/cache/bld/source_metadata-v0/isaac* || true
    # Nuke stale retread audit + probe-trace files in every source
    # package under the consumer (anything containing pixi.toml at
    # max-depth 2). Best-effort -- skip silently if find errors.
    find "$CONSUMER_PROJECT" -maxdepth 3 -name "retread-audit-*.json" -delete 2>/dev/null || true
    find "$CONSUMER_PROJECT" -maxdepth 3 -name "retread-probe-trace-*.json" -delete 2>/dev/null || true
    # Nuke per-entry wheel caches under each source pack's wheels/
    # dir. These accumulated suffix-stack wheels from pre-v0.13.7
    # (`*.injected.autodata.injected.autodata.whl`) waste disk and
    # confuse cache-reuse heuristics when iterating on the pipeline.
    find "$CONSUMER_PROJECT" -maxdepth 3 -type d -name "wheels" \
        -exec rm -rf {} + 2>/dev/null || true
fi

# (6) Rebuild. Variant config is included so multi-python recipes work
#     transparently; recipes without variants ignore the flag.
echo "[rebuild-local] running rattler-build"
RATTLER_ARGS=(
    --recipe "$RECIPE"
    --output-dir "$LOCAL_CHANNEL"
    --target-platform linux-64
)
if [[ -f "$VARIANT_CONFIG" ]]; then
    RATTLER_ARGS+=(--variant-config "$VARIANT_CONFIG")
fi
rattler-build build "${RATTLER_ARGS[@]}"

# (7) Verify -- if the channel doesn't advertise the version we just
#     built, something went sideways (probably a stale repodata.json
#     under a different subdir).
echo "[rebuild-local] verifying channel advertises $VERSION"
ADVERTISED="$(grep -o 'pixi-build-retread-[0-9.]*' "$LOCAL_CHANNEL/linux-64/repodata.json" | sort -u)"
echo "[rebuild-local]   channel sees: $ADVERTISED"
if ! echo "$ADVERTISED" | grep -q "pixi-build-retread-$VERSION\$"; then
    echo "[rebuild-local] WARN: channel does not advertise $VERSION; pixi will pick a different version" >&2
    exit 1
fi

echo "[rebuild-local] done. Next step:"
if [[ -n "${CONSUMER_PROJECT:-}" ]]; then
    echo "    cd $CONSUMER_PROJECT && pixi install"
else
    # CONSUMER_PROJECT not set -- script skipped the per-workspace
    # nuke. Be NOISY about what the user still has to clean by hand,
    # because skipping any of these is the #1 reason "edit code ->
    # rebuild -> still see old error" persists.
    echo ""
    echo "[rebuild-local] WARN: CONSUMER_PROJECT not set; per-workspace caches were NOT nuked." >&2
    echo "[rebuild-local] Before re-solving in your workspace, manually run:" >&2
    echo "" >&2
    echo "    rm -rf \\" >&2
    echo "      <your-workspace>/.pixi/meta-v0/<pack-name>* \\" >&2
    echo "      <your-workspace>/.pixi/bld/<pack-name>* \\" >&2
    echo "      ~/.cache/rattler/cache/bld/metadata-v0/<pack-name>* \\" >&2
    echo "      ~/.cache/rattler/cache/bld/source_metadata-v0/<pack-name>* \\" >&2
    echo "      <your-workspace>/<source-pack-dir>/retread-audit-*.json \\" >&2
    echo "      <your-workspace>/<source-pack-dir>/retread-probe-trace-*.json" >&2
    echo "" >&2
    echo "[rebuild-local] OR re-run this script with CONSUMER_PROJECT=/abs/path to do it automatically:" >&2
    echo "    CONSUMER_PROJECT=/abs/path bash scripts/rebuild-local.sh" >&2
    echo "" >&2
    echo "    cd <your-workspace> && pixi install"
    echo "    (set CONSUMER_PROJECT=/path/to/workspace to also nuke its pixi caches automatically)"
fi
