#!/usr/bin/env bash
# B2 attribution: per-package lock-occurrence delta, baseline vs B2.
# The same test that adjudicated B1 (zarr/numcodecs/openmesh 2->0 or 4->0 = sole provider).
# Usage: b2_attribute.sh <b2 pixi.lock.cert>
set -uo pipefail
B2=${1:?usage: b2_attribute.sh <b2-lock>}
BASE=/oscar/data/stellex/glvov/imprint-data/pixi.lock
[ -f "$B2" ] || { echo "FATAL missing B2 lock: $B2"; exit 2; }
[ -f "$BASE" ] || { echo "FATAL missing baseline lock: $BASE"; exit 2; }
printf 'sizes: baseline %s  b2 %s\n\n' "$(stat -c%s "$BASE")" "$(stat -c%s "$B2")"
printf '%-16s %9s %9s %8s  %s\n' package baseline b2 delta verdict
for p in cyclonedds networkx etils opencv-python mujoco onnxruntime gym pillow \
         transformers moviepy tensordict sentry-sdk dm_control; do
  b=$(grep -cE "/${p}-[0-9]|name: ${p}$" "$BASE")
  n=$(grep -cE "/${p}-[0-9]|name: ${p}$" "$B2")
  if   [ "$n" -eq 0 ] && [ "$b" -gt 0 ]; then v="SOLE-PROVIDER (pin is load-bearing)"
  elif [ "$n" -eq "$b" ];                then v="unchanged -> pin REDUNDANT candidate"
  else                                        v="CHANGED -> read the versions"
  fi
  printf '%-16s %9s %9s %8s  %s\n' "$p" "$b" "$n" "$((n-b))" "$v"
done
echo
echo "=== version drift for anything that survived ==="
for p in cyclonedds networkx etils opencv-python mujoco onnxruntime gym pillow \
         transformers moviepy tensordict sentry-sdk dm_control; do
  vb=$(grep -oE "/${p}-[0-9][^-]*" "$BASE" | sed "s#/${p}-##" | sort -u | tr '\n' ',')
  vn=$(grep -oE "/${p}-[0-9][^-]*" "$B2"   | sed "s#/${p}-##" | sort -u | tr '\n' ',')
  [ "$vb" = "$vn" ] || printf '  %-16s baseline=[%s] b2=[%s]\n' "$p" "$vb" "$vn"
done
