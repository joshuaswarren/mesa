/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_stream_link_core.h"
#include "agx_bytes.h"

#include <string.h>

/*
 * Pack/unpack the exact direct (non-returning) eight-byte VDM Stream Link
 * emitted by the inspected Apple VDMEncoderGen5 and VDMEncoderGen6 paths:
 *
 *   word 0 bits  0..7: target address bits 32..39
 *   word 0 bits  8..28: zero
 *   word 0 bits 29..31: VDM block type 4 (Stream Link)
 *   word 1 bits  0..31: target address bits 0..31
 *
 * Unlike other AGX3 command-stream pointers, this target deliberately remains
 * 40-bit through HAL300. On G17P the encoded value is a render-context-relative
 * offset, while the neighboring generations use the address domain selected
 * by their own command-stream ABI. This byte packer preserves the supplied
 * value; the caller owns that generation-specific translation. The
 * returning-call form at word 0 bit 28 is not present in the pinned AGX3
 * emitters and therefore fails closed here.
 *
 * The target allowlist contains only identities directly served by the
 * inspected G15G/G15X, G16G/G16X, G17P, and G17G/G17X Apple Metal plugins.
 * G16P and G17A remain closed despite sharing neighboring command-stream
 * generations.  This helper does not emit a live command or enable command
 * allocation, submission, or device admission.
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
