/* SPDX-License-Identifier: MIT */
#pragma once

#include "agx_va_runtime_hole.h"
#include "agx_vdm_hal200_bind_tail_core.h"
#include "agx_ppp_viewport_payload_core.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Fixed HAL200 draw-state header words shared between encoder and decoder. */
#define AGX_G17P_DRAW_STATE_SELECTOR_WORD0 UINT32_C(0x00000100)
#define AGX_G17P_VARYING_RECORD_WORD0      UINT32_C(0x10040000)

#define AGX_G17P_BIND_GROUP_BYTES 0x80u
#define AGX_G17P_VIEWPORT_RECORD_BYTES 40u
#define AGX_G17P_VIEWPORT_RECORD_MAX_BYTES                              \
   AGX_PPP_VIEWPORT_PAYLOAD_MAX_LENGTH
#define AGX_G17P_VARYING_RECORD_BYTES 20u
#define AGX_G17P_VARYING_RECORD_OFFSET                                  \
   (AGX_G17P_BIND_GROUP_BYTES + AGX_G17P_VIEWPORT_RECORD_BYTES)
#define AGX_G17P_BIND_STATE_BYTES                                        \
   (AGX_G17P_VARYING_RECORD_OFFSET + AGX_G17P_VARYING_RECORD_BYTES)

struct agx_g17p_bind_state_upload {
   struct agx_vdm_hal200_bind_pair pairs[AGX_VDM_HAL200_BIND_COUNT];
   struct agx_vdm_hal200_bind_tail tail;
};

static inline size_t
agx_g17p_bind_state_size(size_t viewport_size)
{
   if (viewport_size < AGX_PPP_VIEWPORT_PAYLOAD_MIN_LENGTH ||
       viewport_size > AGX_G17P_VIEWPORT_RECORD_MAX_BYTES ||
       ((viewport_size - 8u) % 32u) != 0)
      return 0;

   return AGX_G17P_BIND_GROUP_BYTES + viewport_size +
          AGX_G17P_VARYING_RECORD_BYTES;
}

/* Compact state lives in the retained +8 MiB..+32 MiB VDM aperture. */
static inline bool
agx_g17p_draw_state_range_valid(uint64_t compact, size_t size)
{
   return size != 0 && compact >= AGX_G17P_VDM_ALIAS_OFFSET_START &&
          compact < AGX_G17P_VDM_ALIAS_OFFSET_END && (compact & 3) == 0 &&
          size <= AGX_G17P_VDM_ALIAS_OFFSET_END - compact;
}

static inline bool
agx_g17p_ppp_indices_pack(size_t scissor_bytes, size_t depth_bias_bytes,
                          uint8_t output[8])
{
   /* HAL200 selector bit8 is followed by a scissor index in the low half and
    * a depth-bias index in the high half. The enables live elsewhere. */
   if (!output || scissor_bytes == 0 || depth_bias_bytes == 0 ||
       scissor_bytes % 16 != 0 || depth_bias_bytes % 12 != 0)
      return false;

   const size_t scissor_index = scissor_bytes / 16 - 1;
   const size_t depth_bias_index = depth_bias_bytes / 12 - 1;
   if (scissor_index > UINT16_MAX || depth_bias_index > UINT16_MAX)
      return false;

   const uint8_t record[8] = {
      (uint8_t)AGX_G17P_DRAW_STATE_SELECTOR_WORD0,
      (uint8_t)(AGX_G17P_DRAW_STATE_SELECTOR_WORD0 >> 8),
      (uint8_t)(AGX_G17P_DRAW_STATE_SELECTOR_WORD0 >> 16),
      (uint8_t)(AGX_G17P_DRAW_STATE_SELECTOR_WORD0 >> 24),
      (uint8_t)scissor_index, (uint8_t)(scissor_index >> 8),
      (uint8_t)depth_bias_index, (uint8_t)(depth_bias_index >> 8),
   };
   memcpy(output, record, sizeof(record));
   return true;
}

static inline bool
agx_g17p_upload_bind_state(void *cpu, size_t capacity, uint64_t compact,
                           const uint8_t *bind_group,
                           const uint8_t *viewport, size_t viewport_size,
                           uint32_t counts_32, uint32_t counts_16,
                           struct agx_g17p_bind_state_upload *out)
{
   const size_t required = agx_g17p_bind_state_size(viewport_size);
   const size_t varying_offset = AGX_G17P_BIND_GROUP_BYTES + viewport_size;
   if (!cpu || !bind_group || !viewport || !out ||
       !required || capacity < required ||
       ((counts_32 | counts_16) & UINT32_C(0xff000000)) != 0 ||
       !agx_g17p_draw_state_range_valid(compact, required))
      return false;

   const uint32_t group = (uint32_t)compact;
   struct agx_g17p_bind_state_upload encoded = {
      .pairs = {
         {group + varying_offset, 0x700},
         {group + 0x00, 0x500},
         {group + 0x1c, 0x700},
         {group + 0x30, 0x500},
         {group + 0x4c,
          agx_vdm_hal200_bind_viewport_control(viewport_size)},
         {group + AGX_G17P_BIND_GROUP_BYTES, 0x300},
         {group + 0x60, 0x200},
         {group + 0x6c, 0x200},
      },
   };
   if (!agx_vdm_hal200_bind_tail_pack(AGX_CHIP_G17P, encoded.pairs,
                                       &encoded.tail))
      return false;

   memcpy(cpu, bind_group, AGX_G17P_BIND_GROUP_BYTES);
   memcpy((uint8_t *)cpu + AGX_G17P_BIND_GROUP_BYTES, viewport,
          viewport_size);

   /* Native vertex varying record: separate f32/f16 varying counts and
    * two auxiliary words, zero for the admitted user-varying layout. */
   const uint32_t words[5] = {
      AGX_G17P_VARYING_RECORD_WORD0, counts_32, counts_16, 0, 0,
   };
   uint8_t *varyings = (uint8_t *)cpu + varying_offset;
   for (unsigned i = 0; i < 5; ++i) {
      for (unsigned byte = 0; byte < 4; ++byte)
         varyings[4 * i + byte] = words[i] >> (8 * byte);
   }

   *out = encoded;
   return true;
}

static inline bool
agx_g17p_ppp_indices_base_pack(size_t scissor_bytes,
                               size_t depth_bias_bytes,
                               unsigned viewport_count, uint8_t output[8])
{
   if (!output || viewport_count == 0 ||
       viewport_count > AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS ||
       scissor_bytes % 16 != 0 || depth_bias_bytes % 12 != 0 ||
       scissor_bytes / 16 < viewport_count ||
       depth_bias_bytes / 12 < viewport_count)
      return false;

   const size_t scissor_index = scissor_bytes / 16 - viewport_count;
   const size_t depth_bias_index = depth_bias_bytes / 12 - viewport_count;
   if (scissor_index > UINT16_MAX || depth_bias_index > UINT16_MAX)
      return false;

   const uint8_t record[8] = {
      (uint8_t)AGX_G17P_DRAW_STATE_SELECTOR_WORD0,
      (uint8_t)(AGX_G17P_DRAW_STATE_SELECTOR_WORD0 >> 8),
      (uint8_t)(AGX_G17P_DRAW_STATE_SELECTOR_WORD0 >> 16),
      (uint8_t)(AGX_G17P_DRAW_STATE_SELECTOR_WORD0 >> 24),
      (uint8_t)scissor_index, (uint8_t)(scissor_index >> 8),
      (uint8_t)depth_bias_index, (uint8_t)(depth_bias_index >> 8),
   };
   memcpy(output, record, sizeof(record));
   return true;
}

static inline bool
agx_g17p_upload_draw_indices(void *cpu, size_t capacity, uint64_t compact,
                             size_t scissor_bytes, size_t depth_bias_bytes,
                             unsigned viewport_count, uint32_t *offset)
{
   uint8_t indices[8];
   if (!cpu || !offset || capacity < sizeof(indices) ||
       !agx_g17p_draw_state_range_valid(compact, sizeof(indices)) ||
       !agx_g17p_ppp_indices_base_pack(scissor_bytes, depth_bias_bytes,
                                       viewport_count, indices))
      return false;

   memcpy(cpu, indices, sizeof(indices));
   *offset = (uint32_t)compact;
   return true;
}
