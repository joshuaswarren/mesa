#!/usr/bin/env bash
# DEC-32 G14C barrier-mask bisect sweep (jwm2, the M2 Max).
# For each mask arm: rebuild nothing (the driver reads AGX_CDM_BARRIER_MASK
# at runtime per wAQ/cdm-barrier-bisect d4a1bed), run the turnaround
# benchmark, then the 2B/4B decode 5-pair rows, then the coherency stress.
# Usage: cdm_bisect_jwm2.sh DRIVER_ICD_JSON [ARMS...]
#   ARMS default: 0xfffff 0x160 0x020 0x040 0x100 0x060 0x1c0 nocdmbarrier
set -uo pipefail
ICD="${1:?usage: cdm_bisect_jwm2.sh DRIVER_ICD_JSON [ARMS...]}"
shift
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(0xfffff 0x160 0x020 0x040 0x100 0x060 0x1c0 nocdmbarrier)
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VK_ICD_FILENAMES="$ICD"
export MLX_OMARCHY_FOREIGN_ICD="$ICD"
export MLX_OMARCHY_ALLOW_NON_APPLE=1

echo "=== DEC-32 G14C barrier bisect, boot $(cat /proc/sys/kernel/random/boot_id | cut -c1-8), $(date -u +%FT%TZ) ==="
for arm in "${ARMS[@]}"; do
  case "$arm" in
    nocdmbarrier) export AGX_CDM_BARRIER_MASK=0x160; export HK_NOCDMBARRIER=1 ;;
    *)            export AGX_CDM_BARRIER_MASK="$arm"; unset HK_NOCDMBARRIER || true ;;
  esac
  echo "--- arm $arm ---"
  "$KIT_DIR/cdm_turnaround_vk" "$KIT_DIR/bw.spv" 512 262144 8 2>&1 | tail -2
  # the 2B/4B decode 5-pair rows + the coherency stress run here through the
  # omarchy-mlx Vulkan backend; wire to the host's model paths before the run.
  # dec13_decode.py <model> <pairs>
  "$KIT_DIR/dec13_decode.py" 2b 5 2>&1 | tail -3
  "$KIT_DIR/dec13_decode.py" 4b 5 2>&1 | tail -3
  "$KIT_DIR/coherency_stress.sh" "$arm" 2>&1 | tail -2
done
echo "=== bisect sweep complete $(date -u +%FT%TZ) ==="
