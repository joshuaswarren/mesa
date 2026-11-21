/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_ppp_viewport_payload_core.h"
#include "agx_bytes.h"

#include <string.h>

static bool
agx_ppp_viewport_header_layout(enum agx_chip chip, uint32_t *base_header,
                               unsigned *count_shift)
{
   if (!agx_chip_has_record(chip, AGX_RECORD_PPP_VIEWPORT)) {
      *base_header = 0;
      *count_shift = 0;
      return false;
   }

   /* PPPEncoderGen7 on the HAL300 path, PPPEncoderGen6 before it. */
   const bool gen7 = agx_chip_hal300_path(chip);
   *base_header = gen7 ? AGX_PPP_VIEWPORT_PAYLOAD_GEN7_BASE_HEADER
                       : AGX_PPP_VIEWPORT_PAYLOAD_GEN6_BASE_HEADER;
   *count_shift = gen7 ? 14 : 12;
   return true;
}

/*
 * Pack/unpack only the viewport-only PPP payload emitted by the inspected
 * Apple fixed-layout render-state paths:
 *
 *   u32 PPP header (region-clip + viewport bits set)
 *   viewport_count raw 8-byte region-clip records
 *   u32 raw viewport-control record
 *   viewport_count raw 24-byte viewport transform records
 *
 * The six transform words are directly ordered as translate/scale X, Y, Z.
 * They remain integer bit patterns so NaNs and every other floating-point
 * payload round-trip losslessly.  Region-clip and viewport-control semantics
 * are deliberately not inferred here and remain raw.
 *
 * G15, G16, and G17P PPPEncoderGen6 place viewport_count - 1 in header bits
 * 12..15.  G17G/G17X PPPEncoderGen7 place it in bits 14..17.  All other
 * header bits are fixed to the exact viewport-only form, so mixed or extended
 * PPP updates fail closed.  The region-clip and viewport selectors themselves
 * move from bits 10/11 to bits 12/13 at Gen7, so even a one-viewport packet is
 * generation-specific.
 *
 * The target allowlist contains only identities directly served by the
 * inspected G15G/G15X, G16G/G16X, G17P, and G17G/G17X plugins.  This helper
 * does not allocate or emit a live PPP update, follow a VDM pointer, submit a
 * command, or admit a device.
 */
bool
agx_ppp_viewport_payload_core_supported(enum agx_chip chip)
{
   uint32_t base_header;
   unsigned count_shift;
   return agx_ppp_viewport_header_layout(chip, &base_header, &count_shift);
}

static size_t
agx_ppp_viewport_payload_length(unsigned viewport_count)
{
   return 8u + (32u * viewport_count);
}

bool
agx_ppp_viewport_payload_core_pack(
   enum agx_chip chip, const struct agx_ppp_viewport_payload_fields *fields,
   struct agx_ppp_viewport_payload_core *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !fields || !agx_ppp_viewport_payload_core_supported(chip) ||
       fields->viewport_count == 0 ||
       fields->viewport_count > AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS)
      return false;

   uint32_t base_header;
   unsigned shift;
   if (!agx_ppp_viewport_header_layout(chip, &base_header, &shift))
      return false;

   const uint32_t header = base_header |
                           ((uint32_t)(fields->viewport_count - 1) << shift);
   size_t offset = 0;

   agx_store_le32(&out->bytes[offset], header);
   offset += 4;

   for (unsigned i = 0; i < fields->viewport_count; ++i) {
      agx_store_le64(&out->bytes[offset],
                                  fields->region_clip_raw[i]);
      offset += 8;
   }

   agx_store_le32(&out->bytes[offset],
                               fields->viewport_control_raw);
   offset += 4;

   for (unsigned i = 0; i < fields->viewport_count; ++i) {
      const struct agx_ppp_viewport_transform_raw *viewport =
         &fields->viewports[i];
      const uint32_t words[6] = {
         viewport->translate_x, viewport->scale_x,     viewport->translate_y,
         viewport->scale_y,     viewport->translate_z, viewport->scale_z,
      };

      for (unsigned w = 0; w < 6; ++w) {
         agx_store_le32(&out->bytes[offset], words[w]);
         offset += 4;
      }
   }

   if (offset != agx_ppp_viewport_payload_length(fields->viewport_count)) {
      memset(out, 0, sizeof(*out));
      return false;
   }

   out->length = (uint16_t)offset;
   return true;
}

bool
agx_ppp_viewport_payload_core_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_viewport_payload_fields *fields)
{
   if (fields)
      memset(fields, 0, sizeof(*fields));

   if (!bytes || !fields || !agx_ppp_viewport_payload_core_supported(chip) ||
       size < sizeof(uint32_t))
      return false;

   const uint8_t *raw = bytes;
   uint32_t base_header;
   unsigned shift;
   if (!agx_ppp_viewport_header_layout(chip, &base_header, &shift))
      return false;

   const uint32_t count_mask = UINT32_C(0xf) << shift;
   const uint32_t header = agx_load_le32(raw);

   if ((header & ~count_mask) != base_header)
      return false;

   const unsigned viewport_count = ((header & count_mask) >> shift) + 1;
   if (size != agx_ppp_viewport_payload_length(viewport_count))
      return false;

   fields->viewport_count = (uint8_t)viewport_count;
   size_t offset = 4;

   for (unsigned i = 0; i < viewport_count; ++i) {
      fields->region_clip_raw[i] = agx_load_le64(&raw[offset]);
      offset += 8;
   }

   fields->viewport_control_raw = agx_load_le32(&raw[offset]);
   offset += 4;

   for (unsigned i = 0; i < viewport_count; ++i) {
      struct agx_ppp_viewport_transform_raw *viewport = &fields->viewports[i];
      uint32_t *const words[6] = {
         &viewport->translate_x, &viewport->scale_x,     &viewport->translate_y,
         &viewport->scale_y,     &viewport->translate_z, &viewport->scale_z,
      };

      for (unsigned w = 0; w < 6; ++w) {
         *words[w] = agx_load_le32(&raw[offset]);
         offset += 4;
      }
   }

   if (offset != size) {
      memset(fields, 0, sizeof(*fields));
      return false;
   }

   return true;
}
