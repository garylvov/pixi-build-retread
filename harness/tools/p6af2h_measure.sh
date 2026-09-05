#!/usr/bin/env bash
# p6af-2h measurement job. No lock, no relock, no arm: it reads the surviving
# shared uv simple-index cache, re-requests every page it names, and HEADs the
# canonical lock's pypi artefact urls. Read-only against everything of ours.
set -uo pipefail
D=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6af2h-phase1
A=$D/artifacts
mkdir -p "$A"
echo "### P6AF2H MEASURE start $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none}"

UVBIN=/oscar/data/stellex/glvov/tasks/retread-cold-solve/verify_fixes/artifacts/uvbin/uv
echo "### uv binary: $UVBIN -> $("$UVBIN" --version 2>&1)"
echo "### pixi: $(/users/glvov/.pixi/bin/pixi --version 2>&1)"

# --- STEP 2 evidence from the uv binary itself, since no uv crate source exists
#     on this box (checked: ~/.cargo/registry/src/index.crates.io-*/ has
#     rattler_repodata_gateway-0.25.5 and rattler_conda_types-0.42.2 and NO uv-*).
echo "### UV BINARY STRINGS -- simple-index cache bucket names and revalidation vocabulary"
strings -n 5 "$UVBIN" | grep -E '^simple-v[0-9]+$|^wheels-v[0-9]+$|^sdists-v[0-9]+$|^archive-v[0-9]+$|^interpreter-v[0-9]+$|^builds-v[0-9]+$' | sort -u | sed 's/^/###   bucket /'
strings -n 6 "$UVBIN" | grep -iE '^(if-none-match|if-modified-since|cache-control|no-store|no-cache|must-revalidate|max-age|stale-while-revalidate|immutable)$' | sort -u | sed 's/^/###   header-token /'
strings -n 8 "$UVBIN" | grep -E 'vnd\.pypi\.simple' | sort -u | head -5 | sed 's/^/###   accept /'
strings -n 8 "$UVBIN" | grep -iE 'CachePolicy|cache policy|revalidat' | sort -u | head -20 | sed 's/^/###   policy /'

echo "### ON-DISK CACHE SHAPE (find -maxdepth, never a listing)"
for R in /oscar/data/stellex/glvov/agrescap/cache/retread/pixi/uv-cache \
         /oscar/data/stellex/glvov/agrescap/cache/retread/uv; do
  echo "###   root=$R"
  find "$R" -maxdepth 1 -mindepth 1 -type d -printf '###     %f\n' | sort
  for S in "$R"/simple-v*; do
    [ -d "$S" ] || continue
    echo "###     $(basename "$S") subdirs: $(find "$S" -maxdepth 1 -mindepth 1 -type d | wc -l) files: $(find "$S" -maxdepth 2 -type f -name '*.rkyv' | wc -l)"
  done
done

LOCKURLS=/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6af2g-phase1/artifacts/P6AF2G-5872517-N.pypi-urls.txt
PY=$(command -v python3)
echo "### python3: $PY -> $("$PY" --version 2>&1)"
"$PY" "$D/measure_pypi.py" "$A" "$LOCKURLS" 2>&1 | tee "$A/measure.out"
RC=${PIPESTATUS[0]}
echo "### P6AF2H MEASURE DONE rc=$RC $(date -Is)"
exit "$RC"
