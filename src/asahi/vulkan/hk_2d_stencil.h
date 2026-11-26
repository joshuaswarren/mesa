/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#ifndef HK_2D_STENCIL_H
#define HK_2D_STENCIL_H

#include <stdbool.h>
#include <stdint.h>

#include "nir.h"
#include "util/blake3/blake3.h"

struct hk_descriptor_state;
struct hk_device;
struct vk_descriptor_set_layout;

/* Maximum supported 2D box-blur / stereo-SAD radius (16 -> up to 33x33). */
#define HK_2D_STENCIL_MAX_RADIUS 16u

enum hk_2d_stencil_kind {
   HK_2D_STENCIL_BOX_BLUR_U8 = 0,
   HK_2D_STENCIL_STEREO_SAD_U8 = 1,
   HK_2D_STENCIL_SOBEL_3X3_U8 = 2,
   HK_2D_STENCIL_FAST9_U8 = 3,
};

struct hk_2d_stencil_binding {
   uint32_t set;
   uint32_t binding;
   uint32_t descriptor_offset_B;
   blake3_hash set_layout_hash;
};

struct hk_2d_stencil_plan {
   uint8_t kind;
   uint8_t pad[3];
   uint16_t tile_w;
   uint16_t tile_h;
   uint32_t orig_wg_size[3];

   /* Push-constant byte offsets (all 32-bit int32 fields). */
   uint16_t width_pc_offset_B;
   uint16_t height_pc_offset_B;
   uint16_t stride_pc_offset_B;
   uint16_t radius_pc_offset_B;
   uint16_t d_min_pc_offset_B;
   uint16_t d_max_pc_offset_B;
   uint16_t cost_stride_pc_offset_B;
   uint16_t border_pc_offset_B;
   uint16_t threshold_pc_offset_B;
   uint16_t pad2;

   /* Up to 6 SSBO bindings:
    * - BOX_BLUR_U8:   [0]=src(u8), [1]=dst(u8)
    * - STEREO_SAD_U8: [0]=left(u8), [1]=right(u8), [2]=prev_disp(u8),
    *                  [3]=prev_cost(u32), [4]=dst_disp(u8), [5]=dst_cost(u32)
    * - SOBEL_3X3_U8:  [0]=src(u8), [1]=dir(i32), [2]=mag(f32)
    * - FAST9_U8:      [0]=src(u8), [1]=dst_score(f32), [2]=dst_count(u32),
    *                  [3]=dst_pts(ivec2)
    */
   struct hk_2d_stencil_binding bufs[6];
};

bool hk_nir_match_2d_stencil(
   const nir_shader *orig_nir, uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *set_layouts,
   struct hk_2d_stencil_plan *plan_out);

nir_shader *hk_nir_build_tiled_2d_stencil(
   void *mem_ctx, const nir_shader *orig_nir,
   const struct hk_2d_stencil_plan *plan,
   nir_address_format ssbo_addr_format);

bool hk_2d_stencil_validate_dispatch_core(
   bool alias_free, const struct hk_descriptor_state *desc,
   const struct hk_2d_stencil_plan *plan, uint32_t base_x, uint32_t base_y,
   uint32_t base_z, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
   uint32_t *out_groups_x, uint32_t *out_groups_y);

bool hk_2d_stencil_validate_dispatch(
   const struct hk_device *dev, const struct hk_descriptor_state *desc,
   const struct hk_2d_stencil_plan *plan, uint32_t base_x, uint32_t base_y,
   uint32_t base_z, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
   uint32_t *out_groups_x, uint32_t *out_groups_y);

#endif /* HK_2D_STENCIL_H */
