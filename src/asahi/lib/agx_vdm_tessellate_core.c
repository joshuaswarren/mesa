/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_tessellate_core.h"
#include "agx_bytes.h"

#include <string.h>

#define AGX_VDM_TESSELLATE_DIRECT_FIXED UINT32_C(0xb2f00c00)
#define AGX_VDM_TESSELLATE_INSTANCE_STRIDE_MAX UINT32_C(0x7ffff)

/*
 * This covers only the exact nine-dword direct Tessellate record constructed
 * by the pinned G15, HAL200, and HAL300 render plugins.  The factor-buffer VA
 * is 48-bit and split across header[7:0], word 1, and word 8[7:0].  All three
 * inspected encoder families use byte-identical records, so the bytes cannot
 * classify the originating generation.
 *
 * This is a host representation/validation surface.  It does not emit into a
 * live VDM stream or enable tessellation lowering, scheduling, submission,
 * device admission, or an API capability.
 */
static void
agx_vdm_tessellate_store_le32(uint8_t *dst, uint32_t value)
{
   dst[0] = (uint8_t)value;
   dst[1] = (uint8_t)(value >> 8);
   dst[2] = (uint8_t)(value >> 16);
   dst[3] = (uint8_t)(value >> 24);
}

bool
agx_vdm_tessellate_direct_target_supported(enum agx_chip chip)
{
   /* G18G: no J873 native capture contains this record yet. */
   return agx_chip_has_record(chip, AGX_RECORD_VDM_TESSELLATE);
}

bool
agx_vdm_tessellate_direct_pack(
   enum agx_chip chip, const struct agx_vdm_tessellate_direct_info *info,
   struct agx_vdm_tessellate_direct *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info || !agx_vdm_tessellate_direct_target_supported(chip) ||
       info->factor_buffer >= (UINT64_C(1) << 48) ||
       info->instance_stride_word > AGX_VDM_TESSELLATE_INSTANCE_STRIDE_MAX)
      return false;

   const uint32_t header =
      AGX_VDM_TESSELLATE_DIRECT_FIXED |
      (uint32_t)((info->factor_buffer >> 32) & UINT64_C(0xff));

   agx_store_le32(&out->bytes[0], header);
   agx_store_le32(&out->bytes[4],
                                 (uint32_t)info->factor_buffer);
   agx_store_le32(&out->bytes[8], info->patch_count);
   agx_store_le32(&out->bytes[12], info->instance_count);
   agx_store_le32(&out->bytes[16], info->base_patch);
   agx_store_le32(&out->bytes[20], info->base_instance);
   agx_store_le32(&out->bytes[24],
                                 info->instance_stride_word);
   agx_store_le32(&out->bytes[28],
                                 info->factor_buffer_size_minus_one);
   agx_store_le32(
      &out->bytes[32],
      (uint32_t)((info->factor_buffer >> 40) & UINT64_C(0xff)));
   out->length = AGX_VDM_TESSELLATE_DIRECT_LENGTH;
   return true;
}

bool
agx_vdm_tessellate_direct_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_tessellate_direct_info *info)
{
   if (info)
      memset(info, 0, sizeof(*info));

   if (!bytes || !info || !agx_vdm_tessellate_direct_target_supported(chip) ||
       size != AGX_VDM_TESSELLATE_DIRECT_LENGTH)
      return false;

   const uint8_t *raw = bytes;
   const uint32_t header = agx_load_le32(&raw[0]);
   const uint32_t high_word = agx_load_le32(&raw[32]);
   const uint32_t instance_stride =
      agx_load_le32(&raw[24]);

   if ((header & UINT32_C(0xffffff00)) !=
          AGX_VDM_TESSELLATE_DIRECT_FIXED ||
       high_word > UINT32_C(0xff) ||
       instance_stride > AGX_VDM_TESSELLATE_INSTANCE_STRIDE_MAX)
      return false;

   info->factor_buffer =
      (uint64_t)agx_load_le32(&raw[4]) |
      ((uint64_t)(header & UINT32_C(0xff)) << 32) |
      ((uint64_t)high_word << 40);
   info->patch_count = agx_load_le32(&raw[8]);
   info->instance_count = agx_load_le32(&raw[12]);
   info->base_patch = agx_load_le32(&raw[16]);
   info->base_instance = agx_load_le32(&raw[20]);
   info->instance_stride_word = instance_stride;
   info->factor_buffer_size_minus_one =
      agx_load_le32(&raw[28]);
   return true;
}

#define AGX_VDM_TESSELLATE_INDIRECT_FIXED UINT32_C(0xb4300c00)

bool
agx_vdm_tessellate_indirect_pack(
   enum agx_chip chip, const struct agx_vdm_tessellate_indirect_info *info,
   struct agx_vdm_tessellate_indirect *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info || chip != AGX_CHIP_G17P ||
       info->factor_buffer >= (UINT64_C(1) << 48) ||
       info->indirect_buffer >= (UINT64_C(1) << 48) ||
       (info->indirect_buffer & 3) ||
       info->instance_stride_word > AGX_VDM_TESSELLATE_INSTANCE_STRIDE_MAX)
      return false;

   agx_vdm_tessellate_store_le32(
      &out->bytes[0],
      AGX_VDM_TESSELLATE_INDIRECT_FIXED |
         (uint32_t)((info->factor_buffer >> 32) & UINT64_C(0xff)));
   agx_vdm_tessellate_store_le32(&out->bytes[4],
                                 (uint32_t)info->factor_buffer);
   agx_vdm_tessellate_store_le32(&out->bytes[8],
                                 (uint32_t)(info->indirect_buffer >> 32));
   agx_vdm_tessellate_store_le32(&out->bytes[12],
                                 (uint32_t)info->indirect_buffer);
   agx_vdm_tessellate_store_le32(&out->bytes[16],
                                 info->instance_stride_word);
   agx_vdm_tessellate_store_le32(&out->bytes[20],
                                 info->factor_buffer_size_minus_one);
   agx_vdm_tessellate_store_le32(
      &out->bytes[24],
      (uint32_t)((info->factor_buffer >> 40) & UINT64_C(0xff)));
   out->length = AGX_VDM_TESSELLATE_INDIRECT_CORE_LENGTH;
   return true;
}

bool
agx_vdm_tessellate_indirect_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_tessellate_indirect_info *info)
{
   if (info)
      memset(info, 0, sizeof(*info));

   if (!bytes || !info || !agx_chip_native_agx3(chip) ||
       size != AGX_VDM_TESSELLATE_INDIRECT_CORE_LENGTH)
      return false;

   const uint8_t *raw = bytes;
   const uint32_t header = agx_load_le32(&raw[0]);
   const uint32_t indirect_high = agx_load_le32(&raw[8]);
   const uint32_t indirect_low = agx_load_le32(&raw[12]);
   const uint32_t instance_stride = agx_load_le32(&raw[16]);
   const uint32_t high_word = agx_load_le32(&raw[24]);

   if ((header & UINT32_C(0xffffff00)) != AGX_VDM_TESSELLATE_INDIRECT_FIXED ||
       indirect_high > UINT32_C(0xffff) || (indirect_low & 3) ||
       instance_stride > AGX_VDM_TESSELLATE_INSTANCE_STRIDE_MAX ||
       high_word > UINT32_C(0xff))
      return false;

   info->factor_buffer =
      (uint64_t)agx_load_le32(&raw[4]) |
      ((uint64_t)(header & UINT32_C(0xff)) << 32) |
      ((uint64_t)high_word << 40);
   info->indirect_buffer = ((uint64_t)indirect_high << 32) | indirect_low;
   info->instance_stride_word = instance_stride;
   info->factor_buffer_size_minus_one =
      agx_load_le32(&raw[20]);
   return true;
}

#define AGX_VDM_TESSELLATION_STATE_HEADER UINT32_C(0x40000090)
#define AGX_VDM_TESSELLATION_STATE_FIXED  (UINT32_C(1) << 21)
#define AGX_VDM_TESSELLATION_STATE_ZERO   (UINT32_C(0xffff) | (UINT32_C(1) << 22))

bool
agx_vdm_tessellation_state_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_tessellation_state_info *info)
{
   if (info)
      memset(info, 0, sizeof(*info));

   if (!bytes || !info || !agx_vdm_tessellate_direct_target_supported(chip) ||
       size != AGX_VDM_TESSELLATION_STATE_LENGTH)
      return false;

   const uint8_t *raw = bytes;
   const uint32_t word1 = agx_load_le32(&raw[4]);
   const uint32_t word2 = agx_load_le32(&raw[8]);
   if (agx_load_le32(&raw[0]) !=
          AGX_VDM_TESSELLATION_STATE_HEADER ||
       !(word1 & AGX_VDM_TESSELLATION_STATE_FIXED) ||
       (word1 & AGX_VDM_TESSELLATION_STATE_ZERO) || word2 > UINT16_MAX)
      return false;

   info->partition = (enum agx_vdm_tessellation_partition)(word1 >> 30);
   info->counterclockwise = (word1 >> 29) & 1;
   const unsigned odd =
      info->partition == AGX_VDM_TESSELLATION_PARTITION_FRACTIONAL_ODD;
   const unsigned stored = (word1 >> 23) & 0x3f;
   if (stored + 1 + odd > 64)
      return false;
   info->max_factor = (uint8_t)(stored + 1 + odd);
   info->quads = (word1 >> 20) & 1;
   info->float_factors = (word1 >> 19) & 1;
   info->step = (enum agx_vdm_tessellation_step)((word1 >> 17) & 3);
   info->program_flag = (word1 >> 16) & 1;
   info->scale_half = (uint16_t)word2;
   return true;
}
