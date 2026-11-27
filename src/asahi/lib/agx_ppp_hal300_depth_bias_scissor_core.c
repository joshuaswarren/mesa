/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_ppp_hal300_depth_bias_scissor_core.h"
#include "agx_bytes.h"

#include <string.h>

static bool
base_is_representable(uint64_t base)
{
   return (base >> 48) == 0 && (base & UINT64_C(3)) == 0;
}

static void
pack_base(uint8_t *dst, uint64_t base)
{
   agx_store_le32(&dst[0], (uint32_t)((base >> 16) & UINT64_C(0xffff0000)));
   agx_store_le32(&dst[4], (uint32_t)base & UINT32_C(0xfffffffc));
}

static bool
unpack_base(const uint8_t *src, uint64_t *base)
{
   const uint32_t high = agx_load_le32(&src[0]);
   const uint32_t low = agx_load_le32(&src[4]);

   if ((high & UINT32_C(0x0000ffff)) != 0 || (low & UINT32_C(3)) != 0)
      return false;

   *base = ((uint64_t)high << 16) | low;
   return true;
}

bool
agx_ppp_hal300_depth_bias_scissor_core_supported(enum agx_chip chip)
{
   return agx_chip_has_record(chip, AGX_RECORD_PPP_HAL300_DEPTH_BIAS_SCISSOR);
}

bool
agx_ppp_hal300_depth_bias_record_pack(
   enum agx_chip chip,
   const struct agx_ppp_hal300_depth_bias_record_fields *fields,
   struct agx_ppp_hal300_depth_bias_record *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !fields ||
       !agx_ppp_hal300_depth_bias_scissor_core_supported(chip))
      return false;

   agx_store_le32(&out->bytes[0], fields->depth_bias_raw);
   agx_store_le32(&out->bytes[4], fields->slope_scale_raw);
   agx_store_le32(&out->bytes[8], fields->clamp_raw);
   out->length = AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH;
   return true;
}

bool
agx_ppp_hal300_depth_bias_record_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_hal300_depth_bias_record_fields *fields)
{
   if (fields)
      memset(fields, 0, sizeof(*fields));

   if (!bytes || !fields || size != AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH ||
       !agx_ppp_hal300_depth_bias_scissor_core_supported(chip))
      return false;

   const uint8_t *raw = bytes;
   fields->depth_bias_raw = agx_load_le32(&raw[0]);
   fields->slope_scale_raw = agx_load_le32(&raw[4]);
   fields->clamp_raw = agx_load_le32(&raw[8]);
   return true;
}

bool
agx_ppp_hal300_scissor_record_pack(
   enum agx_chip chip,
   const struct agx_ppp_hal300_scissor_record_fields *fields,
   struct agx_ppp_hal300_scissor_record *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !fields ||
       !agx_ppp_hal300_depth_bias_scissor_core_supported(chip))
      return false;

   agx_store_le16(&out->bytes[0], fields->max_x);
   agx_store_le16(&out->bytes[2], fields->min_x);
   agx_store_le16(&out->bytes[4], fields->max_y);
   agx_store_le16(&out->bytes[6], fields->min_y);
   agx_store_le32(&out->bytes[8], fields->min_z_raw);
   agx_store_le32(&out->bytes[12], fields->max_z_raw);
   out->length = AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH;
   return true;
}

bool
agx_ppp_hal300_scissor_record_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_hal300_scissor_record_fields *fields)
{
   if (fields)
      memset(fields, 0, sizeof(*fields));

   if (!bytes || !fields || size != AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH ||
       !agx_ppp_hal300_depth_bias_scissor_core_supported(chip))
      return false;

   const uint8_t *raw = bytes;
   fields->max_x = agx_load_le16(&raw[0]);
   fields->min_x = agx_load_le16(&raw[2]);
   fields->max_y = agx_load_le16(&raw[4]);
   fields->min_y = agx_load_le16(&raw[6]);
   fields->min_z_raw = agx_load_le32(&raw[8]);
   fields->max_z_raw = agx_load_le32(&raw[12]);
   return true;
}

bool
agx_ppp_hal300_depth_bias_scissor_bases_pack(
   enum agx_chip chip,
   const struct agx_ppp_hal300_depth_bias_scissor_bases_fields *fields,
   struct agx_ppp_hal300_depth_bias_scissor_bases *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !fields ||
       !agx_ppp_hal300_depth_bias_scissor_core_supported(chip) ||
       !base_is_representable(fields->depth_bias_base) ||
       !base_is_representable(fields->scissor_base))
      return false;

   agx_store_le32(&out->bytes[0], AGX_PPP_HAL300_DEPTH_BIAS_SCISSOR_BASES_HEADER);
   pack_base(&out->bytes[4], fields->depth_bias_base);
   pack_base(&out->bytes[12], fields->scissor_base);
   out->length = AGX_PPP_HAL300_DEPTH_BIAS_SCISSOR_BASES_LENGTH;
   return true;
}

bool
agx_ppp_hal300_depth_bias_scissor_bases_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_hal300_depth_bias_scissor_bases_fields *fields)
{
   if (fields)
      memset(fields, 0, sizeof(*fields));

   if (!bytes || !fields ||
       size != AGX_PPP_HAL300_DEPTH_BIAS_SCISSOR_BASES_LENGTH ||
       !agx_ppp_hal300_depth_bias_scissor_core_supported(chip))
      return false;

   const uint8_t *raw = bytes;
   if (agx_load_le32(&raw[0]) != AGX_PPP_HAL300_DEPTH_BIAS_SCISSOR_BASES_HEADER)
      return false;

   uint64_t depth_bias_base;
   uint64_t scissor_base;
   if (!unpack_base(&raw[4], &depth_bias_base) ||
       !unpack_base(&raw[12], &scissor_base))
      return false;

   fields->depth_bias_base = depth_bias_base;
   fields->scissor_base = scissor_base;
   return true;
}
