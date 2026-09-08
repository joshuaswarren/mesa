#!/bin/sh
# gdb_sdpa.sh <venv> <icd.json> <tag> [extra env...]
VENV="$1"; ICD="$2"; TAG="$3"; shift 3
cd "$HOME/src/mlx-coopmat-dense-20260908/.work/mlx/python/tests" || exit 9
env DEVICE=gpu MLX_OMARCHY_ALLOW_NON_APPLE=1 VK_ICD_FILENAMES="$ICD" "$@" \
  flock /tmp/m1-gpu.lock timeout 900 gdb -q -batch \
  -ex "handle SIGBUS stop print" -ex run -ex "bt 30" -ex "info registers pc sp x0 x1 x2 x3 x8 x19 x20 x21" -ex "x/6i \$pc" -ex "p \$_siginfo._sifields._sigfault.si_addr" -ex "info sharedlibrary" \
  --args "$VENV/bin/python" -m pytest -q -p no:cacheprovider -s "test_fast_sdpa.py::TestFastSDPA::test_sdpa" \
  > "$HOME/hkql/gdb-$TAG.log" 2>&1
echo rc=$?
grep -n "signal SIG\|^#[0-9]" "$HOME/hkql/gdb-$TAG.log" | head -40 | cut -c1-220
grep -n "^pc \|^sp \|^x0 \|^x1 \|^x2 \|^x3 \|^x8 \|^x19\|^x20\|^x21\|si_addr\|=> 0x" "$HOME/hkql/gdb-$TAG.log" | cut -c1-160
