/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_hal200_bind_tail_core.h"
#include "agx_bytes.h"
#include "agx_ppp_viewport_payload_core.h"

#include <string.h>

static const uint32_t controls[AGX_VDM_HAL200_BIND_COUNT] = {
   UINT32_C(0x700), UINT32_C(0x500), UINT32_C(0x700), UINT32_C(0x500),
   UINT32_C(0xa00), UINT32_C(0x300), UINT32_C(0x200), UINT32_C(0x200),
};

uint32_t
agx_vdm_hal200_bind_viewport_control(size_t viewport_bytes)
{
   if (viewport_bytes < AGX_PPP_VIEWPORT_PAYLOAD_MIN_LENGTH ||
       viewport_bytes > AGX_PPP_VIEWPORT_PAYLOAD_MAX_LENGTH ||
       (viewport_bytes - 8) % 32 != 0)
      return 0;

   return (uint32_t)(viewport_bytes / 4) << 8;
}

/* Whether a pair's control is one the native tail can carry. */
static bool
agx_vdm_hal200_bind_control_valid(unsigned pair, uint32_t control)
{
   if (pair >= AGX_VDM_HAL200_BIND_COUNT)
      return false;

   if (pair != AGX_VDM_HAL200_BIND_VIEWPORT_CONTROL_PAIR)
      return control == controls[pair];

   return (control & 0xff) == 0 &&
          agx_vdm_hal200_bind_viewport_control((control >> 8) * 4) == control;
}

bool
agx_vdm_hal200_bind_tail_pack(
   enum agx_chip chip,
   const struct agx_vdm_hal200_bind_pair pairs[AGX_VDM_HAL200_BIND_COUNT],
   struct agx_vdm_hal200_bind_tail *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !pairs || !agx_chip_native_agx3(chip))
      return false;

   agx_store_le32(out->bytes, AGX_VDM_HAL200_BIND_HEADER_CONTROL);
   for (unsigned i = 0; i < AGX_VDM_HAL200_BIND_COUNT; ++i) {
      if (pairs[i].context_offset == 0 ||
          (pairs[i].context_offset & UINT32_C(3)) != 0 ||
          !agx_vdm_hal200_bind_control_valid(i, pairs[i].control)) {
         memset(out, 0, sizeof(*out));
         return false;
      }

      const size_t offset = 4 + (8 * i);
      agx_store_le32(&out->bytes[offset], pairs[i].context_offset);
      agx_store_le32(&out->bytes[offset + 4], pairs[i].control);
   }

   out->length = AGX_VDM_HAL200_BIND_TAIL_LENGTH;
   return true;
}

bool
agx_vdm_hal200_bind_tail_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_hal200_bind_pair pairs[AGX_VDM_HAL200_BIND_COUNT])
{
   if (pairs)
      memset(pairs, 0,
             sizeof(struct agx_vdm_hal200_bind_pair) *
                AGX_VDM_HAL200_BIND_COUNT);

   if (!bytes || !pairs || !agx_chip_native_agx3(chip) ||
       size != AGX_VDM_HAL200_BIND_TAIL_LENGTH)
      return false;

   const uint8_t *raw = bytes;
   if (agx_load_le32(raw) != AGX_VDM_HAL200_BIND_HEADER_CONTROL)
      return false;

   for (unsigned i = 0; i < AGX_VDM_HAL200_BIND_COUNT; ++i) {
      const size_t offset = 4 + (8 * i);
      pairs[i].context_offset = agx_load_le32(&raw[offset]);
      pairs[i].control = agx_load_le32(&raw[offset + 4]);
      if (pairs[i].context_offset == 0 ||
          (pairs[i].context_offset & UINT32_C(3)) != 0 ||
          !agx_vdm_hal200_bind_control_valid(i, pairs[i].control)) {
         memset(pairs, 0,
                sizeof(struct agx_vdm_hal200_bind_pair) *
                   AGX_VDM_HAL200_BIND_COUNT);
         return false;
      }
   }

   return true;
}
