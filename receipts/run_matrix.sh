#!/bin/sh
# usage: run_matrix.sh <icd.json> [extra env passthrough via environment]
# Runs every advertised shape x layout x mode through cmshape under the GPU lock.
ICD="$1"
cd "$HOME/src/cmshape" || exit 1
pass=0; fail=0
for shape in "8 8 8" "16 16 16"; do
  for types in "f32 f32 f32 f32" "f16 f16 f32 f32" "f16 f16 f16 f16"; do
    for lay in row col; do
      for mode in identity ones randint randf; do
        out=$(VK_ICD_FILENAMES="$ICD" AGX_SIMDMAT=1 flock /tmp/m1-gpu.lock ./cmshape $shape $types $lay $mode 2>&1)
        rc=$?
        echo "$out"
        if [ $rc -eq 0 ]; then pass=$((pass+1)); else fail=$((fail+1)); fi
      done
    done
  done
done
echo "SUMMARY pass=$pass fail=$fail"
