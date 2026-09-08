#!/bin/sh
# run_py.sh <venv> <icd.json> <tag> <pytest args...>
VENV="$1"; ICD="$2"; TAG="$3"; shift 3
cd "$HOME/src/mlx-coopmat-dense-20260908/.work/mlx/python/tests" || exit 9
T0=$(date +%s)
env DEVICE=gpu MLX_OMARCHY_ALLOW_NON_APPLE=1 VK_ICD_FILENAMES="$ICD" \
  flock /tmp/m1-gpu.lock timeout 900 "$VENV/bin/python" -m pytest -q -p no:cacheprovider "$@" \
  > "$HOME/hkql/py-$TAG.log" 2>&1
RC=$?
echo "rc=$RC wall=$(( $(date +%s) - T0 ))" >> "$HOME/hkql/py-$TAG.log"
echo "== $TAG rc=$RC wall=$(( $(date +%s) - T0 )) : $(grep -E 'passed|failed|error' "$HOME/hkql/py-$TAG.log" | tail -1 | cut -c1-160)"
