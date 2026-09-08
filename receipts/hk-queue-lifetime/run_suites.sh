#!/bin/sh
# run_suites.sh <icd.json> <tag>
ICD="$1"; TAG="$2"
D="$HOME/src/mlx-coopmat-dense-20260908/.work/build/tests/omarchy"
for t in omarchy_runtime_tests omarchy_fast_ops_tests omarchy_fast_regression_tests omarchy_primitive_tests omarchy_reduce_ops_tests; do
  if [ ! -x "$D/$t" ]; then echo "== $t: MISSING"; continue; fi
  T0=$(date +%s)
  env VK_ICD_FILENAMES="$ICD" AGX_SIMDMAT=1 MLX_OMARCHY_ALLOW_NON_APPLE=1 flock /tmp/m1-gpu.lock timeout 900 "$D/$t" > "$HOME/hkql/suite-$TAG-$t.log" 2>&1
  RC=$?
  echo "== $t rc=$RC wall=$(( $(date +%s) - T0 )) : $(grep -E 'test cases:' "$HOME/hkql/suite-$TAG-$t.log" | tail -1) | $(grep -E 'assertions:' "$HOME/hkql/suite-$TAG-$t.log" | tail -1)"
done
