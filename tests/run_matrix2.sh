#!/bin/sh
# Build: cc -O2 -ffp-contract=off cmshape2.c -lvulkan -lm -o cmshape2
# Usage: run_matrix2.sh /absolute/path/to/icd.json
ICD="${1:?usage: run_matrix2.sh <icd.json>}"
cd "$(dirname "$0")" || exit 1
pass=0; fail=0
run() {
  desc="$1"; shift
  out=$(VK_DRIVER_FILES="$ICD" VK_ICD_FILENAMES="$ICD" AGX_SIMDMAT=1 MESA_SHADER_CACHE_DISABLE=true flock -w 30 /tmp/m1-gpu.lock timeout --kill-after=5s 120 ./cmshape2 "$@" 2>&1)
  rc=$?
  echo "$out"
  if [ $rc -eq 0 ]; then pass=$((pass+1)); else fail=$((fail+1)); echo "^^ FAIL: $desc" >&2; fi
}

echo "=== HW baseline (lsize 32, must always pass) ==="
for shape in "8 8 8" "16 16 16"; do
  for types in "f32 f32 f32 f32" "f16 f16 f32 f32" "f16 f16 f16 f16"; do
    for lay in row col; do
      run "hw $shape $types $lay" $shape $types $lay randint 7 32
    done
  done
done

echo "=== workgroup sizes (SW path after fix) ==="
for lsize in 1 8 16 48 33; do
  run "w $lsize f32" 8 8 8 f32 f32 f32 f32 row randint 7 $lsize
  run "w $lsize f16" 16 16 16 f16 f16 f16 f16 row randint 7 $lsize
done

echo "=== pointer forms ==="
run "ptr u16 f16" 8 8 8 f16 f16 f16 f16 row randint 7 32 u16
run "ptr u16 f32" 8 8 8 f32 f32 f32 f32 row randint 7 32 u16
run "ptr u8 f16"  8 8 8 f16 f16 f16 f16 row randint 7 32 u8
run "ptr u16 f32 col" 16 16 16 f32 f32 f32 f32 col randint 7 32 u16

echo "=== precise contraction (NoContraction must survive) ==="
run "precise l32" 8 8 8 f32 f32 f32 f32 row precise 7 32 f32
run "precise l16" 8 8 8 f32 f32 f32 f32 row precise 7 16 f32
echo "=== divergent component updates ==="
run "divergent l16" 8 8 8 f16 f16 f32 f32 row divergent 7 16 u16
run "divergent l33" 8 8 8 f16 f16 f32 f32 col divergent 7 33 u16
echo "SUMMARY pass=$pass fail=$fail"
test "$fail" -eq 0
