/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef AGX_M5_VA_H
#define AGX_M5_VA_H

#include <stdbool.h>
#include <stdint.h>

/* M5 uses compact TA/deflake pointers relative to this architectural base.
 * Their 32-bit fields require driver resources in the first 4 GiB. Reserve
 * a larger driver interval there, away from both data/sparse heaps and USC.
 * Keep this contract in sync with the kernel's m5_vm.rs.
 */
#define AGX_M5_DRIVER_BASE UINT64_C(0x1000000000)
#define AGX_M5_DRIVER_SIZE UINT64_C(0x800000000)
#define AGX_M5_DATA_START UINT64_C(0x4000000000)
#define AGX_M5_DATA_END UINT64_C(0x8000000000)
#define AGX_M5_SHADER_BASE UINT64_C(0x10000000000)
#define AGX_M5_SHADER_SIZE UINT64_C(0x100000000)

struct agx_m5_va_layout {
   uint64_t driver_start, driver_end;
   uint64_t data_start, data_end;
   uint64_t shader_start, shader_end;
};

/* VDM PPP references are relative to the architectural GART base. */
static inline bool
agx_m5_vdm_compact(uint64_t address, uint64_t *out)
{
   if (!out || address < AGX_M5_DRIVER_BASE || (address & 3) ||
       address - AGX_M5_DRIVER_BASE >= (UINT64_C(1) << 39))
      return false;
   *out = address - AGX_M5_DRIVER_BASE;
   return true;
}

static inline bool
agx_m5_va_layout(uint64_t vm_start, uint64_t vm_end,
                 uint64_t kernel_min_size, uint64_t shader_base,
                 struct agx_m5_va_layout *out)
{
   if (!out || vm_start >= vm_end || vm_start > AGX_M5_DRIVER_BASE ||
       shader_base < vm_start || (shader_base & (AGX_M5_SHADER_SIZE - 1)) ||
       shader_base > UINT64_MAX - AGX_M5_SHADER_SIZE ||
       shader_base + AGX_M5_SHADER_SIZE > vm_end || AGX_M5_DATA_END > vm_end)
      return false;

   uint64_t size = kernel_min_size > AGX_M5_DRIVER_SIZE
                     ? kernel_min_size : AGX_M5_DRIVER_SIZE;
   if (size > UINT64_MAX - 0x3fff)
      return false;
   size = (size + 0x3fff) & ~UINT64_C(0x3fff);
   if (size > AGX_M5_DATA_START - AGX_M5_DRIVER_BASE)
      return false;

   const uint64_t driver_end = AGX_M5_DRIVER_BASE + size;
   const uint64_t shader_end = shader_base + AGX_M5_SHADER_SIZE;
   if ((shader_base < driver_end && shader_end > AGX_M5_DRIVER_BASE) ||
       (shader_base < AGX_M5_DATA_END && shader_end > AGX_M5_DATA_START))
      return false;

   *out = (struct agx_m5_va_layout) {
      .driver_start = AGX_M5_DRIVER_BASE, .driver_end = driver_end,
      .data_start = AGX_M5_DATA_START, .data_end = AGX_M5_DATA_END,
      .shader_start = shader_base, .shader_end = shader_end,
   };
   return true;
}

#endif
