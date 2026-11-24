/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_stream_link_core.h"
#include "agx_bytes.h"

#include <string.h>

/*
 * Direct (non-returning) 8-byte VDM Stream Link:
 *   word 0[7:0]:   target[39:32]
 *   word 0[28:8]:  0 (returning-call bit 28 fails closed)
 *   word 0[31:29]: VDM block type 4
 *   word 1[31:0]:  target[31:0]
 * Target remains 40-bit through HAL300; callers own address-domain
 * translation (e.g. render-context-relative offset on G17P vs absolute).
 */
bool
agx_vdm_stream_link_core_supported(enum agx_chip chip)
{
   /* G18G: no J873 native capture contains this record yet. */
   return agx_chip_has_record(chip, AGX_RECORD_VDM_STREAM_LINK);
}

bool
agx_vdm_stream_link_core_pack(enum agx_chip chip, uint64_t target,
                              struct agx_vdm_stream_link_core *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !agx_vdm_stream_link_core_supported(chip) ||
       target >= (UINT64_C(1) << 40))
      return false;

   const uint32_t word0 =
      AGX_VDM_STREAM_LINK_CORE_HEADER | (uint32_t)(target >> 32);
   const uint32_t word1 = (uint32_t)target;

   agx_store_le32(&out->bytes[0], word0);
   agx_store_le32(&out->bytes[4], word1);
   out->length = AGX_VDM_STREAM_LINK_CORE_LENGTH;
   return true;
}

bool
agx_vdm_stream_link_core_unpack(enum agx_chip chip, const void *bytes,
                                size_t size, uint64_t *target)
{
   if (target)
      *target = 0;

   if (!bytes || !target || size != AGX_VDM_STREAM_LINK_CORE_LENGTH ||
       !agx_vdm_stream_link_core_supported(chip))
      return false;

   const uint8_t *raw = bytes;
   const uint32_t word0 = agx_load_le32(&raw[0]);
   if ((word0 & AGX_VDM_STREAM_LINK_CORE_HEADER_MASK) !=
       AGX_VDM_STREAM_LINK_CORE_HEADER)
      return false;

   const uint32_t word1 = agx_load_le32(&raw[4]);
   *target = ((uint64_t)(word0 & UINT32_C(0xff)) << 32) | word1;
   return true;
}
