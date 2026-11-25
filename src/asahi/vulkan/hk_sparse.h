/* SPDX-License-Identifier: MIT */
#pragma once

#include "asahi/lib/agx_chip.h"
#include "asahi/layout/layout.h"

/* Sparse binds have already converted texel coordinates to format elements.
 * Do not divide block-compressed coordinates by the block extent again.
 */
static inline uint64_t
hk_sparse_image_tile_offset_B(enum agx_chip chip, const struct ail_layout *layout,
                              unsigned level, unsigned x_el, unsigned y_el,
                              unsigned z)
{
   (void)chip;
   assert(x_el % layout->tilesize_el[level].width_el == 0);
   assert(y_el % layout->tilesize_el[level].height_el == 0);
   return z * layout->layer_stride_B +
          ail_get_tiled_element_B(layout, level, x_el, y_el, 0);
}
