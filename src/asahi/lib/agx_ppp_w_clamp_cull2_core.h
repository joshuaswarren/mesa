/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded, target-aware AGX3 PPP W-clamp/Cull-2 payload representation.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_PPP_W_CLAMP_CULL2_CORE_LENGTH 12u
#define AGX_PPP_W_CLAMP_CULL2_GEN6_HEADER UINT32_C(0x00410000)
#define AGX_PPP_W_CLAMP_CULL2_GEN7_HEADER UINT32_C(0x00840002)

struct agx_ppp_w_clamp_cull2_fields {
   uint32_t w_clamp_raw;
   uint32_t cull2_raw;
};

struct agx_ppp_w_clamp_cull2_core {
   uint8_t bytes[AGX_PPP_W_CLAMP_CULL2_CORE_LENGTH];
   uint8_t length;
};

/*
 * Pack/unpack only the exact three-word PPP payload copied by the inspected
 * Apple fixed-layout render-state paths:
 *
 *   u32 exact W-clamp + Cull-2 selector header
 *   u32 lossless W-clamp payload
 *   u32 lossless Cull-2 payload
 *
 * G15G/G15X, G16G/G16X, and G17P emit 0x00410000.  G17G/G17X emit
 * 0x00840002.  On all seven the default record is the first two words
 * followed by the third, exactly 12 bytes.
 *
 * Their default complete witnesses are:
 *
 *   Gen6: 00 00 41 00 00 00 00 00 80 00 00 00
 *   Gen7: 02 00 84 00 00 00 00 00 80 00 00 00
 *
 * Dynamic paths copy both payload words without masking.  They remain raw so
 * every bit round-trips without assigning an unsupported meaning to Cull-2
 * bit 7 or to the fixed Gen7 header bit 1.
 *
 * The target allowlist contains only identities directly served by the
 * inspected G15G/G15X, G16G/G16X, G17P, and G17G/G17X plugins, plus G18G,
 * whose J873 native renders carry the Gen7 default witness.  This helper
 * does not allocate PPP memory, emit the surrounding VDM pointer, lower API
 * state, submit a command, admit a device, or advertise a capability.
 */
bool agx_ppp_w_clamp_cull2_core_pack(
   enum agx_chip chip, const struct agx_ppp_w_clamp_cull2_fields *fields,
   struct agx_ppp_w_clamp_cull2_core *out);

bool agx_ppp_w_clamp_cull2_core_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_w_clamp_cull2_fields *fields);

#ifdef __cplusplus
}
#endif
