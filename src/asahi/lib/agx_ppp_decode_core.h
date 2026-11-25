/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx_chip.h"
#include "agx_ppp_hal300_depth_bias_scissor_core.h"
#include "agx_ppp_hal300_depth_stencil_core.h"
#include "agx_ppp_viewport_payload_core.h"
#include "agx_ppp_w_clamp_cull2_core.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Supported whole-payload AGX3 VDM PPP State Update forms. */
enum agx_ppp_decoded_record_kind {
   AGX_PPP_DECODED_RECORD_INVALID = 0,
   AGX_PPP_DECODED_RECORD_W_CLAMP_CULL2,
   AGX_PPP_DECODED_RECORD_VIEWPORT,
   AGX_PPP_DECODED_RECORD_HAL300_DEPTH_STENCIL,
   AGX_PPP_DECODED_RECORD_HAL300_DEPTH_BIAS_SCISSOR_BASES,
};

struct agx_ppp_decoded_record {
   enum agx_ppp_decoded_record_kind kind;
   size_t length;
   union {
      struct agx_ppp_w_clamp_cull2_fields w_clamp_cull2;
      struct agx_ppp_viewport_payload_fields viewport;
      struct agx_ppp_hal300_depth_stencil_fields hal300_depth_stencil;
      struct agx_ppp_hal300_depth_bias_scissor_bases_fields
         hal300_depth_bias_scissor_bases;
   } data;
};

/* Decode exactly one complete bounded PPP payload.  The size must come from
 * the surrounding VDM pointer; prefixes and appended bytes fail closed. */
bool agx_ppp_decode_one(enum agx_chip chip, const void *payload, size_t size,
                        struct agx_ppp_decoded_record *out);

#ifdef __cplusplus
}
#endif
