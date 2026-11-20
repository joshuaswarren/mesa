/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_hal200_direct_draw_core.h"
#include "agx_bytes.h"
#include "agx_abi.h"
#include "agx_vdm_index_list_core.h"

#include <string.h>

static bool
primitive_supported(uint8_t primitive)
{
   switch (primitive) {
   case 0:
   case 1:
   case 3:
   case 5:
   case 6:
   case 9:
   case 11:
   case 13:
   case 14:
   case 15:
      return true;
   default:
      return false;
   }
}

bool
agx_vdm_hal200_direct_draw_body_pack(
   enum agx_chip chip, const struct agx_vdm_hal200_direct_draw_fields *fields,
   struct agx_vdm_hal200_direct_draw_body *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !fields || !agx_chip_native_agx3(chip) ||
       !primitive_supported(fields->primitive) || fields->vertex_count == 0 ||
       fields->instance_count == 0)
      return false;

   agx_store_le32(&out->bytes[0],
              (AGX_VDM_HAL200_DIRECT_DRAW_OPCODE << 16) |
                 ((uint32_t)fields->primitive << 8));
   agx_store_le32(&out->bytes[4], fields->vertex_count);
   agx_store_le32(&out->bytes[8], fields->instance_count);
   agx_store_le32(&out->bytes[12], fields->vertex_start);
   out->length = AGX_VDM_HAL200_DIRECT_DRAW_BODY_LENGTH;
   return true;
}

bool
agx_vdm_hal200_direct_draw_pack(
   enum agx_chip chip, const struct agx_vdm_hal200_direct_draw_fields *fields,
   struct agx_vdm_hal200_direct_draw *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   struct agx_vdm_hal200_direct_draw_body body;
   if (!out || !fields || fields->draw_state_context_offset == 0 ||
       (fields->draw_state_context_offset & UINT32_C(3)) != 0 ||
       !agx_vdm_hal200_direct_draw_body_pack(chip, fields, &body))
      return false;

   agx_store_le32(out->bytes, fields->draw_state_context_offset);
   memcpy(out->bytes + AGX_VDM_HAL200_DRAW_STATE_POINTER_LENGTH, body.bytes,
          body.length);
   out->length = AGX_VDM_HAL200_DIRECT_DRAW_LENGTH;
   return true;
}

bool
agx_vdm_hal200_draw_emit(enum agx_chip chip, const void *packed,
                         size_t packed_size, bool render_state_emitted,
                         void *stream, size_t stream_capacity,
                         size_t *emitted_size)
{
   if (emitted_size)
      *emitted_size = 0;

   if (!agx_chip_native_agx3(chip) || !packed || !stream || !emitted_size ||
       (packed_size != AGX_VDM_HAL200_DIRECT_DRAW_LENGTH &&
        packed_size != AGX_VDM_HAL200_INDEXED_DRAW_LENGTH &&
        packed_size != AGX_VDM_HAL200_INDIRECT_NONINDEXED_DRAW_LENGTH &&
        packed_size != AGX_VDM_HAL200_INDIRECT_INDEXED_DRAW_LENGTH))
      return false;

   const size_t skip = render_state_emitted
                          ? 0
                          : AGX_VDM_HAL200_DRAW_STATE_POINTER_LENGTH;
   const size_t size = packed_size - skip;
   if (stream_capacity < size)
      return false;

   memcpy(stream, (const uint8_t *)packed + skip, size);
   *emitted_size = size;
   return true;
}

bool
agx_vdm_hal200_index_range(uint64_t address, uint64_t range_B,
                           uint32_t first_index, uint32_t count,
                           uint8_t index_size_B, uint64_t *relocated,
                           uint32_t *size_word)
{
   if (relocated)
      *relocated = 0;
   if (size_word)
      *size_word = 0;

   const uint64_t limit = UINT64_C(1) << 48;
   const uint64_t offset_B = (uint64_t)first_index * index_size_B;
   const uint64_t count_B = (uint64_t)count * index_size_B;
   /* Null indices use the canonical zero page with a one-dword native bound.
    * Repeated zero indices must not reduce the draw count or widen that bound.
    * Ordinary buffers still require their complete requested span in range.
    */
   const bool zero_indices = address == AGX_ZERO_PAGE_ADDRESS &&
                             range_B == 4 && first_index == 0;
   if (!relocated || !size_word || !address ||
       (index_size_B != 1 && index_size_B != 2 && index_size_B != 4) ||
       (address & (index_size_B - 1)) ||
       address >= limit || !range_B ||
       range_B > limit - address || !count || offset_B >= range_B ||
       (!zero_indices && count_B > range_B - offset_B))
      return false;

   /* A native direct indexed draw rounds the supplied byte length up to
    * dwords, then subtracts one. Keep arithmetic 64-bit so firstIndex and a
    * large (> 4 GiB) range cannot wrap before validation, and clamp remaining
    * windows above the 32-bit dword ceiling to UINT32_MAX dwords (which still
    * covers every possible uint32_t index count). The window starts at the
    * dword holding the first index, so a range that ends inside a dword needs
    * that dword counted from there. */
   *relocated = address + offset_B;
   const uint64_t rem_B = range_B - offset_B;
   const uint64_t max_dwords_B = (uint64_t)UINT32_MAX * 4;
   const uint64_t dwords =
      rem_B >= max_dwords_B
         ? (uint64_t)UINT32_MAX
         : (rem_B + (*relocated & 3) + 3) >> 2;
   *size_word = (uint32_t)((dwords > UINT32_MAX ? UINT32_MAX : dwords) - 1);
   return true;
}

bool
agx_vdm_hal200_indexed_draw_body_pack(
   enum agx_chip chip, const struct agx_vdm_hal200_indexed_draw_fields *fields,
   struct agx_vdm_hal200_indexed_draw_body *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   uint64_t address;
   uint32_t size_word;
   if (!out || !fields || !agx_chip_native_agx3(chip) ||
       fields->instance_count == 0 ||
       !primitive_supported(fields->primitive) ||
       !agx_vdm_hal200_index_range(
          fields->index_buffer, fields->index_buffer_range_B,
          fields->first_index, fields->index_count, fields->index_size_B,
          &address, &size_word))
      return false;

   struct agx_vdm_index_list_direct_words packet;
   if (!agx_vdm_index_list_direct_pack_words(
          chip, fields->primitive, true, fields->restart_enable,
          fields->index_size_B == 1 ? AGX_VDM_INDEX_LIST_INDEX_SIZE_U8
          : fields->index_size_B == 2 ? AGX_VDM_INDEX_LIST_INDEX_SIZE_U16
                                    : AGX_VDM_INDEX_LIST_INDEX_SIZE_U32, address,
          fields->index_count, fields->instance_count,
          (uint32_t)fields->base_vertex, size_word, &packet) || packet.count != 7)
      return false;

   /* Restart state precedes every native indexed draw, even with restart
    * disabled; baseVertex is copied as a raw signed word. */
   agx_store_le32(out->bytes, UINT32_C(0x40000001));
   const uint32_t restart_max = fields->index_size_B == 1 ? UINT32_C(0xff)
                                 : fields->index_size_B == 2 ? UINT32_C(0xffff)
                                                            : UINT32_MAX;
   agx_store_le32(out->bytes + 4, fields->restart_index < restart_max
                                 ? fields->restart_index : restart_max);
   for (unsigned i = 0; i < packet.count; ++i)
      agx_store_le32(out->bytes + 8 + 4 * i, packet.words[i]);
   out->length = AGX_VDM_HAL200_INDEXED_DRAW_BODY_LENGTH;
   return true;
}

bool
agx_vdm_hal200_indexed_draw_pack(
   enum agx_chip chip, const struct agx_vdm_hal200_indexed_draw_fields *fields,
   struct agx_vdm_hal200_indexed_draw *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   struct agx_vdm_hal200_indexed_draw_body body;
   if (!out || !fields || !fields->draw_state_context_offset ||
       (fields->draw_state_context_offset & 3) ||
       !agx_vdm_hal200_indexed_draw_body_pack(chip, fields, &body))
      return false;

   agx_store_le32(out->bytes, fields->draw_state_context_offset);
   memcpy(out->bytes + AGX_VDM_HAL200_DRAW_STATE_POINTER_LENGTH, body.bytes,
          body.length);
   out->length = AGX_VDM_HAL200_INDEXED_DRAW_LENGTH;
   return true;
}

bool
agx_vdm_hal200_indirect_draw_pack(
   enum agx_chip chip, const struct agx_vdm_hal200_indirect_draw_fields *fields,
   struct agx_vdm_hal200_indirect_draw *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !fields || !agx_chip_native_agx3(chip) ||
       !fields->draw_state_context_offset ||
       (fields->draw_state_context_offset & 3) || !fields->indirect_buffer)
      return false;

   enum agx_vdm_index_list_index_size index_size;
   switch (fields->index_size_B) {
   case 1:
      index_size = AGX_VDM_INDEX_LIST_INDEX_SIZE_U8;
      break;
   case 2:
      index_size = AGX_VDM_INDEX_LIST_INDEX_SIZE_U16;
      break;
   case 4:
      index_size = AGX_VDM_INDEX_LIST_INDEX_SIZE_U32;
      break;
   default:
      index_size = AGX_VDM_INDEX_LIST_INDEX_SIZE_U8;
      if (fields->indexed)
         return false;
      break;
   }

   uint32_t size_word = 0;
   if (fields->indexed) {
      const uint64_t limit = UINT64_C(1) << 48;
      if (!fields->index_buffer ||
          (fields->index_buffer & (fields->index_size_B - 1)) ||
          fields->index_buffer >= limit ||
          !fields->index_buffer_range_B ||
          fields->index_buffer_range_B > limit - fields->index_buffer)
         return false;
      const uint64_t dwords = (fields->index_buffer_range_B +
                               (fields->index_buffer & 3) + 3) >> 2;
      if (dwords == 0 || dwords > UINT32_MAX)
         return false;
      size_word = (uint32_t)(dwords - 1);
   }

   struct agx_vdm_index_list_indirect_words packet;
   if (!agx_vdm_index_list_indirect_pack_words(
          chip, fields->primitive, fields->indexed, fields->restart_enable,
          index_size, fields->index_buffer, fields->indirect_buffer,
          fields->restart_index, size_word, &packet))
      return false;

   agx_store_le32(out->bytes, fields->draw_state_context_offset);
   for (unsigned i = 0; i < packet.count; ++i)
      agx_store_le32(out->bytes + 4 + 4 * i, packet.words[i]);
   out->length = 4 + 4 * packet.count;

   return out->length == (fields->indexed
                             ? AGX_VDM_HAL200_INDIRECT_INDEXED_DRAW_LENGTH
                             : AGX_VDM_HAL200_INDIRECT_NONINDEXED_DRAW_LENGTH);
}
