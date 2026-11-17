/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded, target-aware AGX3 VDM memory-barrier representation.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_VDM_BARRIER_CORE_LENGTH 4u
#define AGX_VDM_BARRIER_CORE_WORD   UINT32_C(0x21000960)
/* The same barrier with bit 3 set (genxml VDM_BARRIER `USC cache inval`).
 * G17G needs it: without it the raster vertex shader of an emulated geometry
 * draw can read stale contents of recycled command-pool memory written by the
 * geometry pre-pass, the tiler bins nothing and the fragment kick of that
 * empty tile hangs until the firmware restarts the GPU (J704, 26A428). */
#define AGX_VDM_BARRIER_CORE_WORD_USC_INVAL UINT32_C(0x21000968)

/* The render-start memory-barrier word for `chip`. */
static inline uint32_t
agx_vdm_barrier_core_word(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17G ? AGX_VDM_BARRIER_CORE_WORD_USC_INVAL
                                : AGX_VDM_BARRIER_CORE_WORD;
}

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
