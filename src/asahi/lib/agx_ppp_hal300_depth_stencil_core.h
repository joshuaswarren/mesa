/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded HAL300 PPP depth/stencil state-block representation.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_PPP_HAL300_DEPTH_STENCIL_CORE_LENGTH 24u
#define AGX_PPP_HAL300_DEPTH_STENCIL_HEADER UINT32_C(0x00000001)

struct agx_ppp_hal300_depth_stencil_fields {
   uint32_t fragment_control_raw;
   uint32_t fragment_front_face_raw;
   uint32_t fragment_back_face_raw;
   uint32_t fragment_front_stencil_raw;
   uint32_t fragment_back_stencil_raw;
};

struct agx_ppp_hal300_depth_stencil_core {
   uint8_t bytes[AGX_PPP_HAL300_DEPTH_STENCIL_CORE_LENGTH];
   uint8_t length;
};

/*
 * Pack/unpack the 6-word HAL300 PPP depth/stencil block: header (bit 0)
 * followed by fragment control, front face, back face, front stencil, and
 * back stencil (note: back face precedes front stencil on HAL300, swapped
 * relative to G16).
 *
 * Admitted: G17G/X (HAL300) and G18G (J873 native renders); closed: G15/G16
 * and G17P/G17A.
 */
bool agx_ppp_hal300_depth_stencil_core_supported(enum agx_chip chip);

bool agx_ppp_hal300_depth_stencil_core_pack(
   enum agx_chip chip,
   const struct agx_ppp_hal300_depth_stencil_fields *fields,
   struct agx_ppp_hal300_depth_stencil_core *out);

bool agx_ppp_hal300_depth_stencil_core_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_hal300_depth_stencil_fields *fields);

#ifdef __cplusplus
}
#endif
