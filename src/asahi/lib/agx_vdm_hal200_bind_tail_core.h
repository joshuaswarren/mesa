/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded G17P VertexProgramVariant bind-tail representation.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_VDM_HAL200_BIND_COUNT 8u
#define AGX_VDM_HAL200_BIND_TAIL_LENGTH 68u
#define AGX_VDM_HAL200_BIND_HEADER_CONTROL UINT32_C(0x00000500)
#define AGX_VDM_HAL200_BIND_VIEWPORT_CONTROL_PAIR 4u

struct agx_vdm_hal200_bind_pair {
   uint32_t context_offset;
   uint32_t control;
};

struct agx_vdm_hal200_bind_tail {
   uint8_t bytes[AGX_VDM_HAL200_BIND_TAIL_LENGTH];
   uint8_t length;
};

/*
 * Append this 68-byte tail immediately after the 28-byte G17P
 * VertexProgramVariant prefix.  The result is the native 0x60-byte encoder
 * prefix whose draw record begins at byte 0x60:
 *
 *   u32 header control = 0x500
 *   8 x { u32 render-context-relative address, u32 control }
 *
 * The tail is a run of PPP state records {size_words << 8, address}: each
 * control is the size of the record the following address names, the header
 * control sizes slot 0, and the last control sizes the 8-byte draw-state
 * record whose address follows the tail.  Every slot but the viewport has a
 * fixed length.  The viewport payload grows by 32 bytes per viewport, so its
 * control (pair 4) is 2 + 8 * viewport_count words.
 */
bool agx_vdm_hal200_bind_tail_pack(
   enum agx_chip chip,
   const struct agx_vdm_hal200_bind_pair pairs[AGX_VDM_HAL200_BIND_COUNT],
   struct agx_vdm_hal200_bind_tail *out);

bool agx_vdm_hal200_bind_tail_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_hal200_bind_pair pairs[AGX_VDM_HAL200_BIND_COUNT]);

/* The pair 4 control for a viewport payload of the given length, or zero. */
uint32_t agx_vdm_hal200_bind_viewport_control(size_t viewport_bytes);

#ifdef __cplusplus
}
#endif
