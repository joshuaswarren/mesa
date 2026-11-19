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
/* Apple's captures use 0x21000960, without the invalidate. We set it on every
 * chip that uses this word: the render-start barrier must make the USC see
 * memory written since the previous render. Without it the raster vertex
 * shader of an emulated geometry draw can read stale contents of recycled
 * command-pool memory written by the geometry pre-pass; on G17G (J704, 26A428)
 * the tiler then bins nothing and the fragment kick of that render hangs until
 * the firmware restarts the GPU. */
#define AGX_VDM_BARRIER_CORE_WORD   (0x21000960u | AGX_VDM_BARRIER_CORE_USC_INVAL)

struct agx_vdm_barrier_core {
   uint8_t bytes[AGX_VDM_BARRIER_CORE_LENGTH];
   uint8_t length;
};

bool agx_vdm_barrier_core_supported(enum agx_chip chip);

bool agx_vdm_barrier_core_pack(enum agx_chip chip,
                               struct agx_vdm_barrier_core *out);

bool agx_vdm_barrier_core_unpack(enum agx_chip chip, const void *bytes,
                                 size_t size);

#ifdef __cplusplus
}
#endif
