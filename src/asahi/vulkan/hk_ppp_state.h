/* SPDX-License-Identifier: MIT */
#pragma once

#include "agx_m5_va.h"
#include "agx_ppp.h"

/* Vulkan pools hold full GPU VAs. G17G and G18G VDM PPP pointers, like
 * their VDM roots, encode an offset from the architectural GART base.
 * Native PPP records refer to the full mapped VA after adding that base.
 * Shader/Ioto and PPP-contained 48-bit scissor bases remain full addresses.
 * G17P's existing low-window allocator already returns its compact address.
 */
static inline bool
hk_ppp_emit_state(enum agx_chip chip, uint8_t **out, uint64_t gpu_base,
                   size_t size)
{
   if (agx_chip_hal300_path(chip)) {
      if (!agx_m5_vdm_compact(gpu_base, &gpu_base))
         return false;
   }
   return agx_ppp_emit_state(chip, out, gpu_base, size);
}
