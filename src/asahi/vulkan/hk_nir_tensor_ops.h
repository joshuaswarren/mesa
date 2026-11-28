/* SPDX-License-Identifier: MIT */
#ifndef HK_NIR_TENSOR_OPS_H
#define HK_NIR_TENSOR_OPS_H

#include <stdbool.h>
#include <stdint.h>
#include "hk_nir_match_pointwise_clamp.h"

struct nir_shader;

enum hk_tensor_op_kind {
   HK_TENSOR_OP_NONE = 0,
   HK_TENSOR_OP_CONCAT_X = 1,
   HK_TENSOR_OP_CONCAT_Y = 2,
   HK_TENSOR_OP_RESAMPLE_NEAREST = 3,
};

struct hk_tensor_op_info {
   uint8_t kind;
   uint8_t src_set;
   uint8_t dst_set;
   uint8_t pad0;
   uint16_t src_binding;
   uint16_t dst_binding;
   uint16_t push_bytes;
   uint16_t local[3];

   /* CONCAT_X:
    *   x_lim = dim_q * dim_c, y_lim = (has_dim_n ? dim_n * dim_h : dim_h)
    *   q = gx / dim_c, r = gx % dim_c
    *   src_idx = gy * x_lim + gx
    *   dst_idx = q * dst_stride_q + gy * dst_stride_y + dst_off_x + r
    *
    * CONCAT_Y:
    *   x_lim = (has_dim_w ? dim_w * dim_c : dim_c), y_lim = dim_n * dim_h
    *   n = gy / dim_h, h = gy % dim_h
    *   src_idx = gy * x_lim + gx
    *   dst_idx = (n * dst_stride_n + dst_y_off + h) * x_lim + gx
    *
    * RESAMPLE_NEAREST:
    *   x_lim = out_w * ch, y_lim = out_h
    *   x = gx / ch, c = gx % ch
    *   sx = f2i32(ffloor(i2f32(x) * scale_x)), sy = f2i32(ffloor(i2f32(gy) * scale_y))
    *   src_idx = (sy * in_w + sx) * ch + c
    *   dst_idx = (gy * out_w + x) * ch + c
    */
   uint16_t dim_n_off;         /* UINT16_MAX if unused */
   uint16_t dim_h_off;         /* out_h for RESAMPLE_NEAREST */
   uint16_t dim_q_off;         /* dim_w for CONCAT_Y (UINT16_MAX if unused), out_w for RESAMPLE */
   uint16_t dim_c_off;         /* channels / inner divisor */
   uint16_t dst_stride_y_off;  /* dst_stride_n for CONCAT_Y, in_w for RESAMPLE */
   uint16_t dst_off_x_off;     /* UINT16_MAX if zero; dst_y_off for CONCAT_Y */
   uint16_t dst_stride_q_off;  /* scale_x for RESAMPLE */
   uint16_t scale_y_off;       /* scale_y for RESAMPLE */
};

/* Fast pre-filter before cloning/lowering NIR to check whether a compute
 * shader can possibly match any tensor-op pattern.
 */
bool hk_nir_tensor_op_may_match(const struct nir_shader *orig);

/* Pre-descriptor NIR after explicit push IO. Builds up to two zero-division
 * vectorized compute alternates (`alt_primary_out`, optional `alt_secondary_out`)
 * for certified 3D tensor concat or nearest-neighbor resample shaders. */
bool hk_nir_match_and_build_tensor_op(const struct nir_shader *orig,
                                      struct hk_tensor_op_info *info_out,
                                      struct nir_shader **alt_primary_out,
                                      struct nir_shader **alt_secondary_out);

/* Post-pass over a DUAL_NONE cooperative-matrix GEMM NIR shader that fuses a
 * trailing element-wise FP32 residual addition (and optional linear copy) at
 * every output store. The residual and destination addresses are the
 * struct hk_dual_fadd_push addresses in push constants plus the store's byte
 * offset from the output's first element, so they never depend on the
 * output descriptor's address. */
bool hk_nir_apply_gemm_fadd(struct nir_shader *nir,
                            const struct hk_nir_gemm_output *info,
                            enum hk_compute_pair_dual_write dual_write);

#endif
