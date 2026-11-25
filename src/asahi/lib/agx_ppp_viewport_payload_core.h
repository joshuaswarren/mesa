/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded, target-aware AGX3 viewport-only PPP payload representation.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS 16u
#define AGX_PPP_VIEWPORT_PAYLOAD_MIN_LENGTH    40u
#define AGX_PPP_VIEWPORT_PAYLOAD_MAX_LENGTH    520u
#define AGX_PPP_VIEWPORT_PAYLOAD_GEN6_BASE_HEADER UINT32_C(0x00000c00)
#define AGX_PPP_VIEWPORT_PAYLOAD_GEN7_BASE_HEADER UINT32_C(0x00003000)

struct agx_ppp_viewport_transform_raw {
   uint32_t translate_x;
   uint32_t scale_x;
   uint32_t translate_y;
   uint32_t scale_y;
   uint32_t translate_z;
   uint32_t scale_z;
};

struct agx_ppp_viewport_payload_fields {
   uint8_t viewport_count;
   uint64_t region_clip_raw[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS];
   uint32_t viewport_control_raw;
   struct agx_ppp_viewport_transform_raw
      viewports[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS];
};

struct agx_ppp_viewport_payload_core {
   uint8_t bytes[AGX_PPP_VIEWPORT_PAYLOAD_MAX_LENGTH];
   uint16_t length;
};

/*
 * Pack/unpack a viewport-only PPP payload: header (region-clip + viewport
 * selector bits plus viewport_count - 1 in bits 12..15 on Gen6 or 14..17 on
 * Gen7), viewport_count raw 8-byte region-clip records, raw viewport control,
 * and viewport_count 24-byte translate/scale XYZ transform records.
 *
 * Admitted: G15G/X, G16G/X, G17P (Gen6), G17G/X, G18G (Gen7, J873 native
 * renders); closed: G16P, G17A (no test rows).
 */
bool agx_ppp_viewport_payload_core_supported(enum agx_chip chip);

bool agx_ppp_viewport_payload_core_pack(
   enum agx_chip chip, const struct agx_ppp_viewport_payload_fields *fields,
   struct agx_ppp_viewport_payload_core *out);

bool agx_ppp_viewport_payload_core_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_viewport_payload_fields *fields);

#ifdef __cplusplus
}
#endif
