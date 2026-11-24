/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_ppp_w_clamp_cull2_core.h"
#include "agx_bytes.h"

#include <string.h>

static bool
agx_ppp_w_clamp_cull2_header(enum agx_chip chip, uint32_t *header)
{
   if (!agx_chip_has_record(chip, AGX_RECORD_PPP_W_CLAMP_CULL2)) {
      *header = 0;
      return false;
   }

   /* PPPEncoderGen7 on the HAL300 path, PPPEncoderGen6 before it. */
   *header = agx_chip_hal300_path(chip) ? AGX_PPP_W_CLAMP_CULL2_GEN7_HEADER
                                        : AGX_PPP_W_CLAMP_CULL2_GEN6_HEADER;
   return true;
}

bool
agx_ppp_w_clamp_cull2_core_pack(
   enum agx_chip chip, const struct agx_ppp_w_clamp_cull2_fields *fields,
   struct agx_ppp_w_clamp_cull2_core *out)
{
   uint32_t header = 0;

   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !fields || !agx_ppp_w_clamp_cull2_header(chip, &header))
      return false;

   agx_store_le32(&out->bytes[0], header);
   agx_store_le32(&out->bytes[4], fields->w_clamp_raw);
   agx_store_le32(&out->bytes[8], fields->cull2_raw);
   out->length = AGX_PPP_W_CLAMP_CULL2_CORE_LENGTH;
   return true;
}

bool
agx_ppp_w_clamp_cull2_core_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_w_clamp_cull2_fields *fields)
{
   uint32_t expected_header = 0;

   if (fields)
      memset(fields, 0, sizeof(*fields));

   if (!bytes || !fields || size != AGX_PPP_W_CLAMP_CULL2_CORE_LENGTH ||
       !agx_ppp_w_clamp_cull2_header(chip, &expected_header))
      return false;

   const uint8_t *raw = bytes;
   if (agx_load_le32(raw) != expected_header)
      return false;

   fields->w_clamp_raw = agx_load_le32(&raw[4]);
   fields->cull2_raw = agx_load_le32(&raw[8]);
   return true;
}
