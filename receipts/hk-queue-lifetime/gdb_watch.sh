#!/bin/sh
# gdb_watch.sh <venv> <icd.json> <tag> <addr>
VENV="$1"; ICD="$2"; TAG="$3"; ADDR="$4"; shift 4
cd "$HOME/src/mlx-coopmat-dense-20260908/.work/mlx/python/tests" || exit 9
cat > /tmp/hkql-watch.gdb <<GEOF
set pagination off
handle SIGBUS stop print
set breakpoint pending on
break mlx::core::eval
run
delete 1
watch *(unsigned long*)$ADDR
commands
silent
printf "WATCH HIT value=0x%lx\n", *(unsigned long*)$ADDR
bt 8
continue
end
continue
bt 6
info registers pc x0
GEOF
env DEVICE=gpu MLX_OMARCHY_ALLOW_NON_APPLE=1 VK_ICD_FILENAMES="$ICD" "$@" \
  flock /tmp/m1-gpu.lock timeout 900 gdb -q -batch -x /tmp/hkql-watch.gdb \
  --args "$VENV/bin/python" -m pytest -q -p no:cacheprovider "test_fast_sdpa.py::TestFastSDPA::test_sdpa" \
  > "$HOME/hkql/gdbwatch-$TAG.log" 2>&1
echo rc=$?
grep -c "WATCH HIT" "$HOME/hkql/gdbwatch-$TAG.log"
