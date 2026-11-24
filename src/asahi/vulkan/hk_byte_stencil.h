/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#ifndef HK_BYTE_STENCIL_H
#define HK_BYTE_STENCIL_H

#include <stdbool.h>
#include <stdint.h>

#include "nir.h"
#include "util/blake3/blake3.h"

struct hk_descriptor_state;
struct hk_device;
struct vk_descriptor_set_layout;

/* Maximum supported stencil diameter (2 * 28 + 1 = 57). Keeps shared memory
 * staging compact (<= 7.75 KiB vertical, <= 3.0 KiB horizontal) so threadgroup
 * occupancy stays full. */
#define HK_BYTE_STENCIL_MAX_DIAMETER 57u
#define HK_BYTE_STENCIL_MAX_RADIUS   28u

enum hk_byte_stencil_axis {
   HK_BYTE_STENCIL_HORIZ = 0,
   HK_BYTE_STENCIL_VERT = 1,
};

enum hk_byte_stencil_store_conv {
   HK_BYTE_STENCIL_CONV_F2I32 = 0,
   HK_BYTE_STENCIL_CONV_F2U32 = 1,
};

struct hk_byte_stencil_binding {
   uint32_t set;
   uint32_t binding;
   uint32_t descriptor_offset_B;
   blake3_hash set_layout_hash;
};

struct hk_byte_stencil_plan {
   uint8_t axis;
   uint8_t store_conv;
   uint8_t is_grid_stride;
   uint8_t pad;
   uint16_t tile_w;
   uint16_t tile_h;
   uint32_t orig_wg_size[3];
   uint16_t width_pc_offset_B;
   uint16_t height_pc_offset_B;
   uint16_t stride_pc_offset_B;
   uint16_t diameter_pc_offset_B;
   struct hk_byte_stencil_binding src;
   struct hk_byte_stencil_binding dst;
   struct hk_byte_stencil_binding weights;
};

bool hk_nir_match_byte_stencil(
   const nir_shader *orig_nir, uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *set_layouts,
   struct hk_byte_stencil_plan *plan_out);

nir_shader *hk_nir_build_tiled_byte_stencil(
   void *mem_ctx, const nir_shader *orig_nir,
   const struct hk_byte_stencil_plan *plan,
   nir_address_format ssbo_addr_format);

bool hk_byte_stencil_validate_dispatch_core(
   bool alias_free, const struct hk_descriptor_state *desc,
   const struct hk_byte_stencil_plan *plan, uint32_t base_x, uint32_t base_y,
   uint32_t base_z, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
   uint32_t *out_groups_x, uint32_t *out_groups_y);

bool hk_byte_stencil_validate_dispatch(
   const struct hk_device *dev, const struct hk_descriptor_state *desc,
   const struct hk_byte_stencil_plan *plan, uint32_t base_x, uint32_t base_y,
   uint32_t base_z, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
   uint32_t *out_groups_x, uint32_t *out_groups_y);

#endif /* HK_BYTE_STENCIL_H */
