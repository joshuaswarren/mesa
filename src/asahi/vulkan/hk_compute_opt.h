/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#ifndef HK_COMPUTE_OPT_H
#define HK_COMPUTE_OPT_H

#include <stdbool.h>
#include <stdint.h>

#include "nir.h"
#include "util/blake3/blake3.h"

struct hk_descriptor_state;
struct hk_device;
struct vk_descriptor_set_layout;

enum hk_compute_opt_kind {
   HK_COMPUTE_OPT_RGBA8_TO_GRAY_U8 = 0,
   HK_COMPUTE_OPT_CANNY_HYSTERESIS_U8 = 1,
   HK_COMPUTE_OPT_CANNY_NMS_U8 = 2,
   HK_COMPUTE_OPT_CANNY_FINALIZE_U8 = 3,
   HK_COMPUTE_OPT_BILINEAR_RESIZE_BORDER_U8 = 4,
};

/* Exact float32 lerp algebra recorded in plan->flags for
 * HK_COMPUTE_OPT_BILINEAR_RESIZE_BORDER_U8 so the emitted kernel reproduces
 * the matched shader's rounding bit-for-bit. */
enum hk_resize_lerp_mode {
   HK_RESIZE_LERP_ONE_MINUS_ADD_A = 0,
   HK_RESIZE_LERP_ONE_MINUS_ADD_B = 1,
   HK_RESIZE_LERP_ONE_MINUS_FFMA_A = 2,
   HK_RESIZE_LERP_ONE_MINUS_FFMA_B = 3,
   HK_RESIZE_LERP_DIFF_FFMA = 4,
   HK_RESIZE_LERP_DIFF_ADD = 5,
};

struct hk_compute_opt_binding {
   uint32_t set;
   uint32_t binding;
   uint32_t descriptor_offset_B;
   blake3_hash set_layout_hash;
};

struct hk_compute_opt_plan {
   uint8_t kind;
   uint8_t flags;
   uint16_t tile_w;
   uint16_t tile_h;
   uint16_t pad0;
   uint32_t orig_wg_size[3];

   /* Push-constant byte offsets (all 32-bit fields). */
   uint16_t pc_off[8];

   /* Kernel-specific scalar parameters captured verbatim from matched NIR.
    * - RGBA8_TO_GRAY_U8:    u32[0..2] = (w_r, w_g, w_b), u32[3] = shr
    * - CANNY_NMS_U8:        u32[0] = dx_lut, u32[1] = dy_lut,
    *                        u32[2] = low_thresh bits, u32[3] = high_thresh bits
    * - CANNY_FINALIZE_U8:   u32[0] = state_thresh, flags bit 0 = is_f2u32
    */
   uint32_t params[8];

   /* Up to 4 SSBO bindings:
    * - RGBA8_TO_GRAY_U8:    [0]=src(u32), [1]=dst(u8)
    * - CANNY_HYSTERESIS_U8: [0]=src(u8), [1]=dst(u8)
    * - CANNY_NMS_U8:        [0]=dir(i32), [1]=mag(f32), [2]=dst(u8)
    * - CANNY_FINALIZE_U8:   [0]=state(u8), [1]=mag(f32), [2]=dst(u8)
    */
   struct hk_compute_opt_binding bufs[4];
};

bool hk_nir_match_compute_opt(
   const nir_shader *orig_nir, uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *set_layouts,
   struct hk_compute_opt_plan *plan_out);

nir_shader *hk_nir_build_compute_opt(
   void *mem_ctx, const nir_shader *orig_nir,
   const struct hk_compute_opt_plan *plan,
   nir_address_format ssbo_addr_format);

bool hk_compute_opt_validate_dispatch_core(
   bool alias_free, const struct hk_descriptor_state *desc,
   const struct hk_compute_opt_plan *plan, uint32_t base_x, uint32_t base_y,
   uint32_t base_z, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
   uint32_t max_groups_x, uint32_t max_groups_y, uint32_t *out_groups_x,
   uint32_t *out_groups_y);

bool hk_compute_opt_validate_dispatch(
   const struct hk_device *dev, const struct hk_descriptor_state *desc,
   const struct hk_compute_opt_plan *plan, uint32_t base_x, uint32_t base_y,
   uint32_t base_z, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
   uint32_t *out_groups_x, uint32_t *out_groups_y);

#endif /* HK_COMPUTE_OPT_H */
