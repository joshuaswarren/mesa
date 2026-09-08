#!/bin/sh
# usage: run_seeds.sh <icd.json> : randint with several seeds for every shape/layout
ICD="$1"
cd "$HOME/src/cmshape" || exit 1
pass=0; fail=0
for shape in "8 8 8" "16 16 16"; do
  for types in "f32 f32 f32 f32" "f16 f16 f32 f32" "f16 f16 f16 f16"; do
    for lay in row col; do
      for seed in 1 7 12345 99991; do
        out=$(VK_ICD_FILENAMES="$ICD" AGX_SIMDMAT=1 flock /tmp/m1-gpu.lock ./cmshape $shape $types $lay randint $seed 2>&1)
        rc=$?
        if [ $rc -eq 0 ]; then pass=$((pass+1)); else fail=$((fail+1)); echo "$out"; fi
      done
    done
  done
done
echo "SEEDS pass=$pass fail=$fail"
