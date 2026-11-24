/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#ifndef HK_PARTIAL_MRT_HSR_H
#define HK_PARTIAL_MRT_HSR_H

#include <stdbool.h>
#include <stdint.h>
#include "agx_linker.h"
#include "agx_pack.h"
#include "util/format/u_format.h"

/*
 * Classify a Vulkan LOAD_OP_CLEAR colour value into one of the constant
 * epilog clear-fill codes so a partial-MRT draw can materialize an unwritten
 * cleared attachment directly in its epilog and keep OPAQUE/PUNCH_THROUGH HSR.
 *
 * Uses exact bit comparisons (so -0.0f is rejected and integer channels match
 * exact 0 / 1 values).
 */
static inline enum agx_epilog_clear_fill
hk_partial_mrt_classify_clear_value(enum pipe_format fmt,
                                    const uint32_t clear_u32[4])
{
   if (fmt == PIPE_FORMAT_NONE || util_format_is_depth_or_stencil(fmt) ||
       util_format_get_nr_components(fmt) == 0)
      return AGX_EPILOG_CLEAR_FILL_NONE;

   const uint32_t one_bits =
      util_format_is_pure_integer(fmt) ? 1u : 0x3f800000u;

   if (clear_u32[0] == 0u && clear_u32[1] == 0u &&
       clear_u32[2] == 0u && clear_u32[3] == 0u)
      return AGX_EPILOG_CLEAR_FILL_ZERO;

   if (clear_u32[0] == 0u && clear_u32[1] == 0u &&
       clear_u32[2] == 0u && clear_u32[3] == one_bits)
      return AGX_EPILOG_CLEAR_FILL_BLACK_OPAQUE;

   if (clear_u32[0] == one_bits && clear_u32[1] == one_bits &&
       clear_u32[2] == one_bits && clear_u32[3] == one_bits)
      return AGX_EPILOG_CLEAR_FILL_WHITE;

   return AGX_EPILOG_CLEAR_FILL_NONE;
}

/*
 * Select the effective hardware pass type (AGX_PASS_TYPE_*) for a graphics
 * draw and advance the render stream's pending opaque colour-attachment write
 * mask (`*opaque_color_written`).
 *
 * On AGX3, an OPAQUE or PUNCH_THROUGH draw covering a tile allows the ISP to
 * skip the tile's background (BG) shader as well as earlier opaque draws. A
 * partial-MRT draw that leaves an attachment untouched therefore must stay
 * TRANSLUCENT whenever any untouched attachment has live contents — whether
 * from the stream's BG shader (LOAD_OP_LOAD, LOAD_OP_CLEAR not materialized
 * by epilog clear-fill, or bg.partial on a continuation stream), an earlier
 * user draw / clear (`preserved_color_mask`), or a pending opaque draw in the
 * current HSR batch (`*opaque_color_written`).
 *
 * When `((preserved_color_mask | *opaque_color_written) & ~color_written) == 0`,
 * every untouched attachment is LOAD_OP_DONT_CARE with no prior writes, so
 * promoting the draw to OPAQUE / PUNCH_THROUGH is safe even across hardware
 * partial renders (SPM).
 *
 * `force_translucent` (for vkCmdClearAttachments meta clears) commits the clear
 * immediately to tile memory and clears `*opaque_color_written`.
 */
static inline uint8_t
hk_partial_mrt_hsr_select_pass_type(uint8_t linked_pass_type,
                                    bool partial_mrt_eligible,
                                    bool tag_write_disable,
                                    bool rasterizer_discard,
                                    bool force_translucent,
                                    bool promote_pt_for_full_quads,
                                    uint8_t color_written,
                                    uint8_t preserved_color_mask,
                                    uint8_t *opaque_color_written)
{
   if (rasterizer_discard)
      return linked_pass_type;

   uint8_t pass_type = linked_pass_type;

   if (force_translucent && !tag_write_disable) {
      if (pass_type == AGX_PASS_TYPE_OPAQUE)
         pass_type = AGX_PASS_TYPE_TRANSLUCENT;
      else if (pass_type == AGX_PASS_TYPE_PUNCH_THROUGH)
         pass_type = AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH;
   } else if (partial_mrt_eligible && !tag_write_disable &&
              color_written != 0 &&
              ((preserved_color_mask | *opaque_color_written) &
               ~color_written) == 0) {
      if (pass_type == AGX_PASS_TYPE_TRANSLUCENT)
         pass_type = AGX_PASS_TYPE_OPAQUE;
      else if (pass_type == AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH)
         pass_type = AGX_PASS_TYPE_PUNCH_THROUGH;
   }

   if (pass_type == AGX_PASS_TYPE_PUNCH_THROUGH && promote_pt_for_full_quads)
      pass_type = AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH;

   if (tag_write_disable)
      return pass_type;

   if (pass_type == AGX_PASS_TYPE_OPAQUE ||
       pass_type == AGX_PASS_TYPE_PUNCH_THROUGH) {
      *opaque_color_written |= color_written;
   } else if (pass_type == AGX_PASS_TYPE_TRANSLUCENT ||
              pass_type == AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH) {
      *opaque_color_written = 0;
   }

   return pass_type;
}

#endif /* HK_PARTIAL_MRT_HSR_H */
