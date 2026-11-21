/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded, target-aware AGX3 VDM memory-barrier representation.
 */

#pragma once

#ifndef __OPENCL_VERSION__
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_VDM_BARRIER_CORE_LENGTH 4u
/* Bit 3 of the barrier word is the USC cache invalidate (the genxml
 * VDM_BARRIER `USC cache inval` field, which the AGX2 path sets). */
#define AGX_VDM_BARRIER_CORE_USC_INVAL 0x8u
/* Apple's captures use 0x21000960, without the invalidate. */
#define AGX_VDM_BARRIER_CORE_PLAIN  0x21000960u

/* The render-start barrier word for a chip. It carries the USC cache
 * invalidate: the render-start barrier must make the USC see memory written
 * since the previous render. Without it the raster vertex shader of an
 * emulated geometry draw can read stale contents of recycled command-pool
 * memory written by the geometry pre-pass; on G17G (J704, 26A428) the tiler
 * then bins nothing and the fragment kick of that render hangs until the
 * firmware restarts the GPU.
 *
 * A kernel that reports DRM_ASAHI_FEATURE_VDM_USC_INVALIDATE
 * (kernel_admits_inval) admits the invalidate on every chip. Older kernels
 * admit it on every chip except G15S, G15C (M3 Pro/Max) and G16G (M4), whose
 * backends refuse a render unless its VDM stream starts with exactly
 * 0x21000960 (EINVAL), so on those kernels those three chips keep the plain
 * word.
 *
 * That chip list is kept as the fallback, rather than setting the invalidate
 * only when the kernel reports the feature: every other chip already runs
 * with the invalidate on kernels that predate the feature, and requiring it
 * would take the fix above away from G17G on all of them.
 */
static inline uint32_t
agx_vdm_barrier_core_word(enum agx_chip chip, bool kernel_admits_inval)
{
   if (kernel_admits_inval)
      return AGX_VDM_BARRIER_CORE_PLAIN | AGX_VDM_BARRIER_CORE_USC_INVAL;

   switch (chip) {
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G16G:
      return AGX_VDM_BARRIER_CORE_PLAIN;
   default:
      return AGX_VDM_BARRIER_CORE_PLAIN | AGX_VDM_BARRIER_CORE_USC_INVAL;
   }
}

struct agx_vdm_barrier_core {
   uint8_t bytes[AGX_VDM_BARRIER_CORE_LENGTH];
   uint8_t length;
};

bool agx_vdm_barrier_core_supported(enum agx_chip chip);

bool agx_vdm_barrier_core_pack(enum agx_chip chip, bool kernel_admits_inval,
                               struct agx_vdm_barrier_core *out);

bool agx_vdm_barrier_core_unpack(enum agx_chip chip, const void *bytes,
                                 size_t size);

#ifdef __cplusplus
}
#endif
