#!/bin/bash
# usage: suites.sh <tag> <icd.json>   (runs in background, logs to ~/log)
tag=$1; icd=$2
cd ~/src/mlx-coopmat-dense-20260908/.work/build/tests/omarchy
for s in primitive eq_math complex_ops fast_ops runtime reduce_ops; do
  log=~/log/pm-suite-$tag-$s.log
  start=$(date +%s)
  VK_ICD_FILENAMES=$icd AGX_SIMDMAT=1 timeout 1800 flock /tmp/m1-gpu.lock ./omarchy_${s}_tests > $log 2>&1
  rc=$?
  end=$(date +%s)
  echo "$s rc=$rc secs=$((end-start)) $(grep -E '^\[doctest\] (test cases|assertions)' $log | tr '\n' ' ')" >> ~/log/pm-suite-$tag-summary.log
done
echo DONE >> ~/log/pm-suite-$tag-summary.log
