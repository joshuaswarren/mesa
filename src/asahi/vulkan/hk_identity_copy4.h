/* SPDX-License-Identifier: MIT */
#ifndef HK_IDENTITY_COPY4_H
#define HK_IDENTITY_COPY4_H

#include <stdbool.h>
#include <stdint.h>
#include "util/u_math.h"

struct nir_shader;

#define HK_COMPUTE_OPT_PUSH_BYTES 64

static inline uint32_t
hk_im2col_push_offset(uint32_t push_bytes)
{
   return ALIGN_POT(push_bytes, 4u);
}

struct hk_im2col_push {
   uint32_t cols;
   uint32_t rows_y;
   uint32_t kh;
};

static inline bool
hk_identity_copy4_supported_local(const uint16_t local[3])
{
   return local[0] > 0 && local[1] > 0 && local[2] == 1 &&
          (uint32_t)local[0] * local[1] <= 1024;
}

enum hk_identity_copy4_kind {
   HK_IDENTITY_COPY4_IM2COL_1X1 = 0,
   HK_IDENTITY_COPY4_LINEAR_COPY = 1,
   HK_IDENTITY_COPY4_LINEAR_OFFSET = 2,
   HK_IDENTITY_COPY4_LINEAR_FADD = 3,
   HK_IDENTITY_COPY4_PAD4 = 4,
};

struct hk_identity_copy4 {
   uint8_t kind;
   uint8_t src_count;
   uint8_t src_set, src1_set, dst_set, pad_set;
   uint16_t src_binding, src1_binding, dst_binding, pad_binding;
   uint16_t border_width_offset, border_height_offset;
   uint16_t input_width_offset, input_height_offset, kernel_width_offset;
   uint16_t patches_width_offset, patches_height_offset;
   uint16_t stride_width_offset, stride_height_offset, channels_offset;
   uint16_t count_offset, src_offset_offset, dst_offset_offset, push_bytes;
   uint16_t in_shape_off[4];
   uint16_t out_shape_off[4];
   uint32_t pad_idx[4];
   uint16_t local[3];
   uint32_t fadd_fp_math_ctrl;
   uint32_t fadd_float_controls_execution_mode;
   bool im2col_vec4_ok;
   bool fadd_meta_valid;
};

/* Fast pre-filter before cloning/lowering NIR to check whether a compute
 * shader can possibly match any identity_copy4 pattern.
 */
bool hk_nir_copy_identity4_may_match(const struct nir_shader *orig);

/* Pre-descriptor NIR after explicit push IO. The returned alternate requires
 * the direct, zero-base, packed-vec4 dispatch and storage certificate external
 * to this matcher. The original and unsuccessful metadata remain unchanged.
 */
struct nir_shader *hk_nir_copy_identity4(const struct nir_shader *orig,
                                         struct hk_identity_copy4 *info);

/* Dual-output packed vec4 alternate for certified LINEAR_FADD shaders: writes
 * each computed vec4 to both the original destination SSBO and a second 16B-
 * aligned destination VA passed in push constants at
 * hk_dual_write_push_offset(info->push_bytes). Returns NULL for other kinds.
 */
struct nir_shader *hk_nir_copy_identity4_fadd_dual(const struct nir_shader *orig,
                                                   struct hk_identity_copy4 *info);

/* Packed vec4 alternate for certified 2D im2col shaders (both unit-stride and
 * non-unit-stride paths) when channels, kernel_width, input_width, and
 * patches_width are multiples of 4.
 */
struct hk_pack_loop_bound;
struct nir_shader *hk_nir_im2col_vec4_with_bound(
   const struct nir_shader *orig,
   struct hk_pack_loop_bound *loop_bound_out);
struct nir_shader *hk_nir_im2col_vec4(const struct nir_shader *orig);

#endif
