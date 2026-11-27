/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <assert.h>
#include <stdint.h>

#include "agx_chip.h"

/*
 * The depth/stencil pixel dimensions the ISP is told about. AGX2 packs them
 * through CR_ISP_ZLS_PIXELS, but the AGX3 generations do not agree: G17P
 * keeps the 15-bit pair (a native 256x256 render publishes 0x007f80ff) while
 * the M5 parts (G17G, G17S, G17C) widen the second field to 16 bits (the
 * same render publishes 0x00ff00ff), and so does G18G (the J873 256x256 depth and stencil renders
 * publish 0x00ff00ff). All treat zero as the inactive encoding rather than a
 * minus-one 1x1 surface, so a render with no depth or stencil storage
 * publishes zero on each. Honeykrisp's hk_drm_*_zls_pixels() are the
 * witnesses for these three rules; this is the same packing where Gallium
 * and the Vulkan driver can share it.
 */
static inline uint32_t
agx_zls_pixels(enum agx_chip chip, uint32_t width, uint32_t height,
               uint64_t depth_base, uint64_t depth_meta, uint64_t stencil_base,
               uint64_t stencil_meta)
{
   if ((depth_base | depth_meta | stencil_base | stencil_meta) == 0)
      return 0;

   const unsigned shift = agx_chip_hal300_path(chip) ? 16 : 15;
   assert(width >= 1 && width <= (1u << shift));
   assert(height >= 1 && height <= (1u << shift));
   return (width - 1) | ((height - 1) << shift);
}
