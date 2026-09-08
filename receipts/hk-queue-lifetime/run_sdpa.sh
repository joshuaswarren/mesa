#!/bin/sh
# run_sdpa.sh <venv> <icd.json> <tag> <pytest-nodeid> [extra env...]
VENV="$1"; ICD="$2"; TAG="$3"; NODE="$4"; shift 4
cd "$HOME/src/mlx-coopmat-dense-20260908/.work/mlx/python/tests" || exit 9
T0=$(date +%s)
env DEVICE=gpu MLX_OMARCHY_ALLOW_NON_APPLE=1 VK_ICD_FILENAMES="$ICD" "$@" \
  flock /tmp/m1-gpu.lock timeout 900 "$VENV/bin/python" -m pytest -q -p no:cacheprovider "$NODE" \
  > "$HOME/hkql/sdpa-$TAG.log" 2>&1
RC=$?
echo "rc=$RC wall=$(( $(date +%s) - T0 ))" >> "$HOME/hkql/sdpa-$TAG.log"
echo "rc=$RC wall=$(( $(date +%s) - T0 ))"
tail -4 "$HOME/hkql/sdpa-$TAG.log" | cut -c1-300
