/* SPDX-License-Identifier: MIT */
#ifndef HK_NIR_MATCH_POINTWISE_CLAMP_H
#define HK_NIR_MATCH_POINTWISE_CLAMP_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include "compiler/nir/nir_defines.h"
#include "util/u_math.h"

struct nir_shader;

struct hk_nir_clamp_buffer {
   uint32_t set;
   uint32_t binding;
   uint32_t offset_B;
};

/* General FP32 pointwise clamp epilogue:
 *   - lower only:  fmax(x, lower)
 *   - upper only:  fmin(x, upper)
 *   - both, !upper_first: fmin(fmax(x, lower), upper)
 *   - both,  upper_first: fmax(fmin(x, upper), lower)
 */
struct hk_nir_clamp_epilogue {
   bool has_lower;
   bool has_upper;
   bool upper_first;
   uint32_t lower_bits;
   uint32_t upper_bits;
   uint32_t max_fp_math_ctrl;
   uint32_t min_fp_math_ctrl;
   uint32_t float_controls_execution_mode;
};

bool hk_nir_clamp_epilogue_valid(const struct hk_nir_clamp_epilogue *e);

static inline bool
hk_nir_clamp_epilogue_equal(const struct hk_nir_clamp_epilogue *a,
                            const struct hk_nir_clamp_epilogue *b)
{
   if (!a || !b)
      return a == b;
   if (a->has_lower != b->has_lower || a->has_upper != b->has_upper ||
       a->float_controls_execution_mode != b->float_controls_execution_mode)
      return false;
   if (a->has_lower && a->has_upper && a->upper_first != b->upper_first)
      return false;
   if (a->has_lower && (a->lower_bits != b->lower_bits ||
                        a->max_fp_math_ctrl != b->max_fp_math_ctrl))
      return false;
   if (a->has_upper && (a->upper_bits != b->upper_bits ||
                        a->min_fp_math_ctrl != b->min_fp_math_ctrl))
      return false;
   return true;
}

struct hk_nir_pointwise_clamp {
   struct hk_nir_clamp_buffer input;
   struct hk_nir_clamp_buffer output;
   uint32_t count_offset_B;
   struct hk_nir_clamp_epilogue epilogue;
   uint32_t local_size_x;
};

/*
 * Recognizes, on pre- or post-preprocess compute NIR before Vulkan descriptor
 * lowering, a signed gx < push_count guard and exactly one FP32 SSBO output:
 * fmax(x, lo), fmin(x, hi), fmin(fmax(x, lo), hi), or fmax(fmin(x, hi), lo).
 * Both arrays have stride four.
 */
bool hk_nir_match_pointwise_clamp(const struct nir_shader *shader,
                                  struct hk_nir_pointwise_clamp *match);

struct hk_nir_depthwise_output {
   struct hk_nir_clamp_buffer output;
   struct hk_nir_clamp_buffer reads[3]; /* 0: input, 1: weights, 2: bias (optional) */
   uint32_t width_offset_B, height_offset_B, channels_offset_B;
   uint32_t input_width_offset_B, input_height_offset_B;
   uint32_t kernel_width_offset_B, kernel_height_offset_B;
   uint32_t stride_x_offset_B, stride_y_offset_B;
   uint32_t dil_x_offset_B, dil_y_offset_B;
   uint32_t pad_x_offset_B, pad_y_offset_B;
   uint32_t has_bias_offset_B; /* UINT32_MAX for biasless depthwise */
   uint32_t push_bytes;
   uint32_t fmul_fp_ctrl, fadd_fp_ctrl, bias_fadd_fp_ctrl;
   uint32_t out_res_type, read_res_type[3];
   uint32_t desc_comps;
   uint32_t local_size[3];
   uint32_t float_controls_execution_mode;
   bool has_bias_buffer;
   bool vec4_valid;
};

struct hk_nir_gemm_output {
   struct hk_nir_clamp_buffer output;
   struct hk_nir_clamp_buffer reads[3]; /* 0: A, 1: B, 2: bias (optional) */
   uint32_t m_offset_B, n_offset_B, k_offset_B, bias_mode_offset_B;
   uint32_t push_bytes;
   uint32_t out_res_type, read_res_type[3];
   uint32_t desc_comps;
   uint32_t local_size[3];
   uint32_t rows_per_thread;
   uint32_t float_controls_execution_mode;
   uint8_t bias_mode_mask; /* bitmask of supported bias_mode arms (0..2) */
   bool has_bias_mode;
   bool has_bias_buffer;
   bool is_guarded;
   bool cmat_valid;
};

enum hk_compute_pair_role {
   HK_COMPUTE_PAIR_NONE = 0,
   HK_COMPUTE_PAIR_DEPTHWISE = 1,
   HK_COMPUTE_PAIR_CLAMP = 2,
   HK_COMPUTE_PAIR_GEMM = 3,
   HK_COMPUTE_PAIR_COPY4 = 4,
   HK_COMPUTE_PAIR_SELF_COPY = 5,
   HK_COMPUTE_PAIR_PAD4 = 6,
   HK_COMPUTE_PAIR_FADD = 7,
   HK_COMPUTE_PAIR_IM2COL = 8,
};

enum hk_compute_pair_dual_write {
   HK_COMPUTE_PAIR_DUAL_NONE = 0,
   HK_COMPUTE_PAIR_DUAL_COPY = 1,
   HK_COMPUTE_PAIR_DUAL_PAD4 = 2,
   HK_COMPUTE_PAIR_DUAL_FADD_INPLACE = 3,
   HK_COMPUTE_PAIR_DUAL_FADD_OUT = 4,
   HK_COMPUTE_PAIR_DUAL_FADD_COPY_INPLACE = 5,
   HK_COMPUTE_PAIR_DUAL_FADD_COPY_OUT = 6,
   HK_COMPUTE_PAIR_DUAL_FADD_EP = 7,
   HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY = 8,
   HK_COMPUTE_PAIR_DUAL_COPY2 = 9,
   HK_COMPUTE_PAIR_DUAL_GEMM_DW = 10,
};

static inline bool
hk_compute_pair_dual_is_fadd(enum hk_compute_pair_dual_write dual)
{
   return dual >= HK_COMPUTE_PAIR_DUAL_FADD_INPLACE &&
          dual <= HK_COMPUTE_PAIR_DUAL_FADD_COPY_OUT;
}

static inline bool
hk_compute_pair_dual_is_fadd_ep(enum hk_compute_pair_dual_write mode)
{
   return mode == HK_COMPUTE_PAIR_DUAL_FADD_EP ||
          mode == HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY;
}

struct hk_nir_fadd_epilogue {
   uint32_t fp_math_ctrl;
   uint32_t float_controls_execution_mode;
   bool prod_is_rhs;
};

static inline bool
hk_nir_fadd_epilogue_equal(const struct hk_nir_fadd_epilogue *a,
                           const struct hk_nir_fadd_epilogue *b)
{
   if (!a || !b)
      return a == b;
   return a->fp_math_ctrl == b->fp_math_ctrl &&
          a->float_controls_execution_mode == b->float_controls_execution_mode &&
          a->prod_is_rhs == b->prod_is_rhs;
}

static inline uint32_t
hk_dual_write_push_offset(uint32_t push_bytes)
{
   return ALIGN_POT(push_bytes, 8u);
}

#define HK_DUAL_WRITE_FLAG_SKIP_C        (1u << 0)
#define HK_DUAL_WRITE_FLAG_SKIP_FADD_DST (1u << 1)

struct hk_dual_copy_push {
   uint32_t dst2_va_lo;
   uint32_t dst2_va_hi;
   uint32_t flags;
   uint32_t _pad;
};

struct hk_dual_copy2_push {
   uint32_t dst2_va_lo;
   uint32_t dst2_va_hi;
   uint32_t dst3_va_lo;
   uint32_t dst3_va_hi;
   uint32_t flags;
   uint32_t _pad;
};

struct hk_dual_fadd_ep_push {
   uint32_t add_src_va_lo;
   uint32_t add_src_va_hi;
   uint32_t fadd_dst_va_lo;
   uint32_t fadd_dst_va_hi;
   uint32_t flags;
   uint32_t _pad;
};

struct hk_dual_fadd_ep_copy_push {
   uint32_t add_src_va_lo;
   uint32_t add_src_va_hi;
   uint32_t fadd_dst_va_lo;
   uint32_t fadd_dst_va_hi;
   uint32_t dst2_va_lo;
   uint32_t dst2_va_hi;
   uint32_t flags;
   uint32_t _pad;
};

/* Push words of the cooperative-matrix GEMM residual-FADD post-pass
 * (hk_nir_apply_gemm_fadd()): the GPU addresses of the FADD residual input,
 * the FADD destination and the copy destination for the output element at
 * hk_nir_gemm_output::output.offset_B. Each is the address itself, as in the
 * other dual writes, never a difference from the output's address. */
struct hk_dual_fadd_push {
   uint32_t res_va_lo;
   uint32_t res_va_hi;
   uint32_t dst1_va_lo;
   uint32_t dst1_va_hi;
   uint32_t dst2_va_lo;
   uint32_t dst2_va_hi;
};

struct hk_dual_pad4_push {
   uint32_t dst2_va_lo;
   uint32_t dst2_va_hi;
   uint32_t pad_va_lo;
   uint32_t pad_va_hi;
   uint32_t in_h;
   uint32_t in_w;
   uint32_t out_h;
   uint32_t out_w;
   uint32_t pad_idx[4];
   uint32_t flags;
   uint32_t _pad;
};

#define HK_GEMM_DW_FLAG_SKIP_C       HK_DUAL_WRITE_FLAG_SKIP_C
#define HK_GEMM_DW_FLAG_HAS_BIAS     (1u << 1)
#define HK_GEMM_DW_FLAG_HAS_DW_DST2  (1u << 2)
#define HK_GEMM_DW_FLAG_SKIP_PAD_DST (1u << 3)
#define HK_GEMM_DW_FLAG_SKIP_DW_DST1 (1u << 4)
#define HK_GEMM_DW_FLAG_SKIP_DW_DST2 (1u << 5)

struct hk_dual_gemm_dw_push {
   uint32_t pad_dst_va_lo;
   uint32_t pad_dst_va_hi;
   uint32_t pad_va_lo;
   uint32_t pad_va_hi;
   uint32_t dw_wt_va_lo;
   uint32_t dw_wt_va_hi;
   uint32_t dw_bias_va_lo;
   uint32_t dw_bias_va_hi;
   uint32_t dw_dst1_va_lo;
   uint32_t dw_dst1_va_hi;
   uint32_t dw_dst2_va_lo;
   uint32_t dw_dst2_va_hi;
   uint32_t in_wh;         /* lo16: in_w, hi16: in_h (multiples of 8) */
   uint32_t out_wh;        /* lo16: pad_out_w (in_w+1), hi16: pad_out_h (in_h+1) */
   uint32_t pad_idx[4];
   uint32_t tiles_x_m_prime;
   uint32_t tiles_x_udiv_sh; /* b0: sh1, b1..5: sh2 for tiles_x = in_w >> 3 */
   uint32_t flags;
   uint32_t _pad;
};

#define HK_GEMM_DW_PROJ_FLAG_HAS_BIAS  (1u << 0)
#define HK_GEMM_DW_PROJ_FLAG_HAS_COPY  (1u << 1)
#define HK_GEMM_DW_PROJ_FLAG_SKIP_C    (1u << 2)
#define HK_GEMM_DW_PROJ_FLAG_SKIP_COPY (1u << 3)

struct hk_dual_gemm_dw_proj_push {
   uint32_t proj_b_va_lo;
   uint32_t proj_b_va_hi;
   uint32_t proj_bias_va_lo;
   uint32_t proj_bias_va_hi;
   uint32_t proj_c_va_lo;
   uint32_t proj_c_va_hi;
   uint32_t proj_copy_va_lo;
   uint32_t proj_copy_va_hi;
   uint32_t proj_n;
   uint32_t proj_flags;
};

#define HK_IM2COL_A_FLAG_VEC4    (1u << 0)
#define HK_IM2COL_A_FLAG_COPY1X1 (1u << 1)

struct hk_im2col_a_push {
   uint32_t src_va_lo;
   uint32_t src_va_hi;
   uint32_t dst_va_lo;
   uint32_t dst_va_hi;
   uint32_t iw;
   uint32_t ih;
   uint32_t kw;
   uint32_t cols;
   uint32_t step_x;
   uint32_t sh;
   uint32_t pad_x;
   uint32_t pad_y;
   uint32_t cols_m_prime;
   uint32_t kw_m_prime;
   uint32_t udiv_shifts; /* b0: cols_sh1, b1..5: cols_sh2, b8: kw_sh1, b9..13: kw_sh2 */
   uint32_t flags;
};

static inline void
hk_im2col_a_fast_udiv_params(uint32_t d, uint32_t *m_prime_out,
                             uint32_t *sh1_out, uint32_t *sh2_out)
{
   if (d <= 1u) {
      *m_prime_out = 1u;
      *sh1_out = 0u;
      *sh2_out = 0u;
      return;
   }
   const uint32_t l = 32u - (uint32_t)__builtin_clz(d - 1u);
   const uint64_t rem_hi = (1ull << l) - (uint64_t)d;
   *m_prime_out = (uint32_t)(((rem_hi << 32) / (uint64_t)d) + 1ull);
   *sh1_out = 1u;
   *sh2_out = l - 1u;
}

static inline void
hk_im2col_a_push_init_udiv(struct hk_im2col_a_push *ip)
{
   uint32_t c_sh1 = 0, c_sh2 = 0, k_sh1 = 0, k_sh2 = 0;
   hk_im2col_a_fast_udiv_params(ip->cols, &ip->cols_m_prime, &c_sh1, &c_sh2);
   hk_im2col_a_fast_udiv_params(ip->kw, &ip->kw_m_prime, &k_sh1, &k_sh2);
   ip->udiv_shifts = c_sh1 | (c_sh2 << 1) | (k_sh1 << 8) | (k_sh2 << 9);
}

#define HK_DW_A_FLAG_HAS_BIAS (1u << 0)
#define HK_DW_A_FLAG_HAS_DST1 (1u << 1)
#define HK_DW_A_FLAG_HAS_DST2 (1u << 2)
#define HK_DW_A_FLAG_STRIDE2  (1u << 3)

struct hk_dw_a_push {
   uint32_t in_va_lo;
   uint32_t in_va_hi;
   uint32_t wt_va_lo;
   uint32_t wt_va_hi;
   uint32_t bias_va_lo;
   uint32_t bias_va_hi;
   uint32_t dst1_va_lo;
   uint32_t dst1_va_hi;
   uint32_t dst2_va_lo;
   uint32_t dst2_va_hi;
   uint32_t in_wh;         /* lo16: in_w, hi16: in_h */
   uint32_t out_wh;        /* lo16: out_w, hi16: out_h */
   uint32_t pad_flags;     /* b0..7: (int8_t)px, b8..15: (int8_t)py, b16..23: flags */
   uint32_t out_w_m_prime;
   uint32_t out_w_udiv_sh; /* b0: sh1, b1..5: sh2 */
   uint32_t _pad;
};

#define HK_EXP_DW_A_FLAG_HAS_BIAS     (1u << 0)
#define HK_EXP_DW_A_FLAG_SKIP_EXP_C   (1u << 1)
#define HK_EXP_DW_A_FLAG_SKIP_IM2_DST (1u << 2)

struct hk_exp_dw_a_push {
   uint32_t exp_a_va_lo;
   uint32_t exp_a_va_hi;
   uint32_t exp_b_va_lo;
   uint32_t exp_b_va_hi;
   uint32_t exp_bias_va_lo;
   uint32_t exp_bias_va_hi;
   uint32_t im2_dst_va_lo;
   uint32_t im2_dst_va_hi;
   uint32_t exp_k;            /* K_in of expand GEMM (% 4 == 0 for 1x1; kw*kh for im2col) */
   uint32_t tiles_x_m_prime;  /* fast udiv for tiles_x = in_w >> 3 */
   uint32_t tiles_x_udiv_sh;  /* b0: sh1, b1..5: sh2 for tiles_x; b8: kw_sh1, b9..13: kw_sh2 */
   uint32_t flags;
   uint32_t im2_iw_ih;        /* lo16: iw, hi16: ih */
   uint32_t im2_kw_kh;        /* lo16: kw, hi16: kh */
   uint32_t im2_step_sh;      /* lo16: step_x, hi16: sh */
   uint32_t im2_pad_xy;       /* lo16: pad_x, hi16: pad_y */
   uint32_t im2_kw_m_prime;   /* fast udiv for im2_kw */
   uint32_t _pad;
};

struct hk_nir_dw_a_cfg {
   uint32_t fmul_fp_ctrl;
   uint32_t fadd_fp_ctrl;
   uint32_t bias_fadd_fp_ctrl;
   bool has_epilogue;
   struct hk_nir_clamp_epilogue epilogue;
   bool has_expand;
   bool has_exp_im2col;
   bool has_exp_epilogue;
   struct hk_nir_clamp_epilogue exp_epilogue;
   bool has_proj;
   uint8_t proj_tile_n; /* 16, 24, 32, or 64 */
   bool has_proj_epilogue;
   struct hk_nir_clamp_epilogue proj_epilogue;
};

static inline bool
hk_nir_dw_a_cfg_equal(const struct hk_nir_dw_a_cfg *a,
                      const struct hk_nir_dw_a_cfg *b)
{
   if (!a || !b)
      return a == b;
   return a->fmul_fp_ctrl == b->fmul_fp_ctrl &&
          a->fadd_fp_ctrl == b->fadd_fp_ctrl &&
          a->bias_fadd_fp_ctrl == b->bias_fadd_fp_ctrl &&
          a->has_epilogue == b->has_epilogue &&
          (!a->has_epilogue ||
           hk_nir_clamp_epilogue_equal(&a->epilogue, &b->epilogue)) &&
          a->has_expand == b->has_expand &&
          a->has_exp_im2col == b->has_exp_im2col &&
          a->has_exp_epilogue == b->has_exp_epilogue &&
          (!a->has_exp_epilogue ||
           hk_nir_clamp_epilogue_equal(&a->exp_epilogue, &b->exp_epilogue)) &&
          a->has_proj == b->has_proj &&
          (!a->has_proj ||
           (a->proj_tile_n == b->proj_tile_n &&
            a->has_proj_epilogue == b->has_proj_epilogue &&
            (!a->has_proj_epilogue ||
             hk_nir_clamp_epilogue_equal(&a->proj_epilogue, &b->proj_epilogue))));
}

static inline void
hk_exp_dw_a_push_init_udiv(struct hk_exp_dw_a_push *ep, uint32_t tiles_x)
{
   uint32_t sh1 = 0, sh2 = 0, kw_sh1 = 0, kw_sh2 = 0;
   hk_im2col_a_fast_udiv_params(tiles_x, &ep->tiles_x_m_prime, &sh1, &sh2);
   const uint32_t kw = ep->im2_kw_kh & 0xffffu;
   if (kw > 0u)
      hk_im2col_a_fast_udiv_params(kw, &ep->im2_kw_m_prime, &kw_sh1, &kw_sh2);
   ep->tiles_x_udiv_sh = sh1 | (sh2 << 1) | (kw_sh1 << 8) | (kw_sh2 << 9);
}

static inline void
hk_dw_a_push_init_udiv(struct hk_dw_a_push *dp, uint32_t out_w)
{
   uint32_t sh1 = 0, sh2 = 0;
   hk_im2col_a_fast_udiv_params(out_w, &dp->out_w_m_prime, &sh1, &sh2);
   dp->out_w_udiv_sh = sh1 | (sh2 << 1);
}

static inline void
hk_dual_gemm_dw_push_init_udiv(struct hk_dual_gemm_dw_push *dp, uint32_t tiles_x)
{
   uint32_t sh1 = 0, sh2 = 0;
   hk_im2col_a_fast_udiv_params(tiles_x, &dp->tiles_x_m_prime, &sh1, &sh2);
   dp->tiles_x_udiv_sh = sh1 | (sh2 << 1);
}

static inline uint32_t
hk_dual_write_push_size(enum hk_compute_pair_dual_write dual_write)
{
   switch (dual_write) {
   case HK_COMPUTE_PAIR_DUAL_COPY:
      return sizeof(struct hk_dual_copy_push);
   case HK_COMPUTE_PAIR_DUAL_COPY2:
      return sizeof(struct hk_dual_copy2_push);
   case HK_COMPUTE_PAIR_DUAL_PAD4:
      return sizeof(struct hk_dual_pad4_push);
   case HK_COMPUTE_PAIR_DUAL_FADD_EP:
      return sizeof(struct hk_dual_fadd_ep_push);
   case HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY:
      return sizeof(struct hk_dual_fadd_ep_copy_push);
   case HK_COMPUTE_PAIR_DUAL_GEMM_DW:
      return sizeof(struct hk_dual_gemm_dw_push);
   default:
      return 0u;
   }
}

static inline uint32_t
hk_im2col_a_push_offset(uint32_t push_bytes,
                        enum hk_compute_pair_dual_write dual_write)
{
   const uint32_t base =
      dual_write != HK_COMPUTE_PAIR_DUAL_NONE
         ? hk_dual_write_push_offset(push_bytes) +
              hk_dual_write_push_size(dual_write)
         : push_bytes;
   return ALIGN_POT(base, 8u);
}

static inline uint32_t
hk_dw_a_push_offset(uint32_t push_bytes,
                    enum hk_compute_pair_dual_write dual_write)
{
   return hk_im2col_a_push_offset(push_bytes, dual_write);
}

static inline uint32_t
hk_exp_dw_a_push_offset(uint32_t push_bytes,
                        enum hk_compute_pair_dual_write dual_write)
{
   return ALIGN_POT(hk_dw_a_push_offset(push_bytes, dual_write) +
                       sizeof(struct hk_dw_a_push),
                    8u);
}

static inline uint32_t
hk_dual_gemm_dw_proj_push_offset(uint32_t push_bytes)
{
   return ALIGN_POT(hk_dual_write_push_offset(push_bytes) +
                       sizeof(struct hk_dual_gemm_dw_push),
                    8u);
}

#define HK_COMPUTE_PAIR_INFO_VERSION 5u

struct hk_compute_pair_info {
   uint32_t version;
   uint32_t role;
   union {
      struct hk_nir_depthwise_output depthwise;
      struct hk_nir_gemm_output gemm;
      struct hk_nir_pointwise_clamp clamp;
      struct {
         uint32_t push_bytes;
         uint32_t fp_math_ctrl;
         uint32_t float_controls_execution_mode;
         bool meta_valid;
      } fadd;
   };
};

/* Proves the guarded output and the input, weight, and optional bias read
 * addresses, in that order. The original indexing and FP32 arithmetic stay
 * intact. */
bool hk_nir_match_depthwise_output(const struct nir_shader *shader,
                                   nir_address_format storage_format,
                                   struct hk_nir_depthwise_output *match);

/* Proves a tiled row-major GEMM output (arbitrary rows_per_thread in 1..16,
 * 0..3 bias_mode arms, optional bias buffer) and its A, B, and optional bias
 * read bindings. */
bool hk_nir_match_gemm_output(const struct nir_shader *shader,
                              nir_address_format storage_format,
                              struct hk_nir_gemm_output *match);

struct hk_nir_direct_shape {
   const void *push;
   uint32_t push_size_B;
   uint32_t groups[3];
   uint32_t base_group[3];
};

bool hk_nir_depthwise_producer_footprint(
   const struct hk_nir_depthwise_output *producer,
   const struct hk_nir_direct_shape *producer_launch,
   uint64_t producer_ends[4], uint64_t *out_elems_out);

bool hk_nir_gemm_producer_footprint(
   const struct hk_nir_gemm_output *producer,
   const struct hk_nir_direct_shape *producer_launch,
   uint64_t producer_ends[4], uint32_t *m_out, uint32_t *n_out);

bool hk_nir_pointwise_clamp_footprint(
   const struct hk_nir_pointwise_clamp *consumer,
   const struct hk_nir_direct_shape *consumer_launch,
   uint64_t consumer_ends[2]);

bool hk_nir_depthwise_clamp_footprint(
   const struct hk_nir_depthwise_output *producer,
   const struct hk_nir_pointwise_clamp *consumer,
   const struct hk_nir_direct_shape *producer_launch,
   const struct hk_nir_direct_shape *consumer_launch,
   uint64_t producer_ends[4], uint64_t consumer_ends[2]);

bool hk_nir_gemm_clamp_footprint(
   const struct hk_nir_gemm_output *producer,
   const struct hk_nir_pointwise_clamp *consumer,
   const struct hk_nir_direct_shape *producer_launch,
   const struct hk_nir_direct_shape *consumer_launch,
   uint64_t producer_ends[4], uint64_t consumer_ends[2]);

/* General epilogue splicer for any matched producer NIR that still carries
 * Vulkan descriptor derefs: clones the producer and wraps every scalar FP32
 * store to `output` with `epilogue`, preserving exact fp_math_ctrl and
 * float_controls_execution_mode. */
struct nir_shader *
hk_nir_splice_epilogue(const struct nir_shader *producer,
                       struct hk_nir_clamp_buffer output,
                       const struct hk_nir_clamp_epilogue *epilogue);

/* Build a packed vec4 2-pixel-per-thread (horizontal pair (ox, ox+1) with Z
 * carrying output row oy) depthwise convolution NIR alternate from a certified
 * depthwise producer (with vec4_valid == true). When `epilogue` is non-NULL,
 * folds that exact clamp epilogue into both pixel outputs; handles both biased
 * and biasless depthwise kernels. */
struct hk_pack_loop_bound;

struct nir_shader *
hk_nir_build_depthwise_vec4(const struct nir_shader *producer,
                            const struct hk_nir_depthwise_output *info,
                            const struct hk_nir_clamp_epilogue *epilogue,
                            enum hk_compute_pair_dual_write dual_write);

bool
hk_nir_bind_depthwise_loop_bounds(struct nir_shader *nir,
                                  const struct hk_nir_depthwise_output *info,
                                  struct hk_pack_loop_bound *loop_bound_out);

/* Build a Direct-B 4 KiB XOR-swizzled Shared-A f32matmad8x8 GEMM NIR alternate
 * (tile_n == 16, 32, or 64) from a certified row-major GEMM producer (with
 * cmat_valid == true). When `epilogue` is non-NULL, folds that exact clamp
 * epilogue into output writeback; when `dual_write` is non-zero, also writes
 * the fused copy4 or pad4 destination buffer directly from registers. */
struct nir_shader *
hk_nir_build_gemm_cmat_tiled_with_bound(
   const struct nir_shader *producer,
   const struct hk_nir_gemm_output *info,
   uint32_t tile_m,
   uint32_t tile_n,
   const struct hk_nir_clamp_epilogue *epilogue,
   enum hk_compute_pair_dual_write dual_write,
   const struct hk_nir_fadd_epilogue *fadd_ep,
   bool im2col_a,
   const struct hk_nir_dw_a_cfg *dw_a,
   const struct hk_nir_dw_a_cfg *gemm_dw,
   struct hk_pack_loop_bound *loop_bound_out);

struct nir_shader *
hk_nir_build_gemm_cmat_with_bound(
   const struct nir_shader *producer,
   const struct hk_nir_gemm_output *info,
   uint32_t tile_n,
   const struct hk_nir_clamp_epilogue *epilogue,
   enum hk_compute_pair_dual_write dual_write,
   const struct hk_nir_fadd_epilogue *fadd_ep,
   bool im2col_a,
   const struct hk_nir_dw_a_cfg *dw_a,
   const struct hk_nir_dw_a_cfg *gemm_dw,
   struct hk_pack_loop_bound *loop_bound_out);

struct nir_shader *
hk_nir_build_gemm_cmat(const struct nir_shader *producer,
                       const struct hk_nir_gemm_output *info,
                       uint32_t tile_n,
                       const struct hk_nir_clamp_epilogue *epilogue,
                       enum hk_compute_pair_dual_write dual_write);

/* M-tile height and K-step for each cooperative-matrix GEMM variant (all 128
 * threads / 4 subgroups sharing a 4 KiB XOR-swizzled shared A tile):
 *   tile_n == 16: 64x16, K_step=16 (4x1 SGs, 2x2 acc/SG, pipe_b)
 *   tile_n == 32: 64x32, K_step=16 (2x2 SGs, 4x2 acc/SG, pipe_b)
 *   tile_n == 64: 32x64, K_step=32 (2x2 SGs, 2x4 acc/SG) */
static inline uint32_t
hk_gemm_cmat_tile_m(uint32_t tile_n)
{
   return tile_n == 64u ? 32u : 64u;
}

static inline uint32_t
hk_gemm_cmat_k_step(uint32_t tile_n)
{
   return tile_n == 64u ? 32u : 16u;
}

/* Choose cooperative-matrix tile_n (16, 32, or 64) for a GEMM dispatch.
 * - N <= 16: cmat16 (32x16, K_step=32) avoids empty N-columns.
 * - Exact multiples of 64 (with K >= 32), 32, or 16 (< 64, e.g. N=48): pick
 *   the widest exact tile to avoid N-padding waste and enable unguarded
 *   specialization whenever M%tile_m==0, K%8==0.
 * - Non-multiple N > 32 with even ceil(N/32) and K >= 32: cmat64 (32x64,
 *   K_step=32) halves workgroup barriers and doubles M-parallelism without
 *   extra N-padding waste.
 * - Small-grid high-K GEMMs where cmat64 (g64 <= 16 with K >= 64, or
 *   g64 <= 28 with K >= 128) or cmat32 (g32 <= 32 with K >= 64) would
 *   under-fill shader cores: cmat16 (32x16, K_step=32) multiplies workgroup
 *   count by up to 4x and halves K-loop barriers vs cmat32.
 * - Otherwise: cmat32 (64x32, K_step=16). */
static inline uint32_t
hk_gemm_cmat_select_tile_n(uint64_t m_u, uint32_t n_u, uint32_t k_u,
                           bool has16, bool has32, bool has64)
{
   const uint32_t n_tiles32 = (n_u + 31u) / 32u;
   if (n_u <= 16u && has16)
      return 16u;
   uint32_t tile_n;
   if ((n_u % 64u) == 0u && k_u >= 32u && has64)
      tile_n = 64u;
   else if ((n_u % 32u) == 0u && has32)
      tile_n = 32u;
   else if ((n_u % 16u) == 0u && n_u < 64u && has16)
      return 16u;
   else if (n_u > 32u && (n_tiles32 % 2u) == 0u && k_u >= 32u && has64)
      tile_n = 64u;
   else if (has32)
      tile_n = 32u;
   else if (has64)
      return 64u;
   else
      return has16 ? 16u : 0u;

   if (has16 && tile_n == 64u) {
      const uint64_t g64 = ((m_u + 31ull) / 32ull) * ((uint64_t)(n_u + 63u) / 64ull);
      if ((g64 <= 16ull && k_u >= 64u) || (g64 <= 28ull && k_u >= 128u))
         return 16u;
   } else if (has16 && tile_n == 32u) {
      const uint64_t g32 = ((m_u + 63ull) / 64ull) * (uint64_t)n_tiles32;
      if (g32 <= 32ull && k_u >= 64u)
         return 16u;
   }
   return tile_n;
}

/* Return true when a guarded GEMM dispatch with validated descriptor ranges
 * may safely specialize to the unguarded cooperative-matrix variant (which
 * executes K_step main iterations plus a uniform K%8==0 tail). */
static inline bool
hk_gemm_cmat_can_specialize_unguarded(bool is_guarded, uint64_t m_u,
                                      uint64_t n_u, uint64_t k_u,
                                      uint32_t tile_n, uint32_t a_offset_B,
                                      uint64_t a_base_addr,
                                      uint64_t a_base_offset)
{
   if (!is_guarded || m_u == 0 || n_u == 0 || k_u == 0)
      return false;
   if (tile_n != 16u && tile_n != 24u && tile_n != 32u && tile_n != 48u &&
       tile_n != 64u)
      return false;
   const uint32_t tile_m = hk_gemm_cmat_tile_m(tile_n);
   return (m_u % tile_m) == 0 && (n_u % tile_n) == 0 && (k_u % 8u) == 0 &&
          (a_offset_B % 16u) == 0 && ((a_base_addr | a_base_offset) % 16ull) == 0;
}

/* Validate dispatch dimensions for cooperative-matrix GEMM replacement.
 * Unguarded GEMM executes ceil(K/32) unconditional 32-wide tiles, so K must
 * be a multiple of 32 (not merely 16) to match the original shader exactly. */
static inline bool
hk_gemm_cmat_dispatch_dims_ok(bool is_guarded, uint64_t m_u, uint32_t n_u,
                              uint32_t k_u, const uint32_t grid[3])
{
   if (n_u == 0 || k_u == 0 || m_u == 0 || grid[2] != 1)
      return false;
   if (is_guarded) {
      const uint64_t need_x = ((uint64_t)n_u + 31ull) / 32ull;
      const uint64_t need_y = (m_u + 31ull) / 32ull;
      return (uint64_t)grid[0] >= need_x &&
             (uint64_t)grid[1] >= need_y &&
             (uint64_t)grid[0] * 32ull <= (uint64_t)INT32_MAX &&
             (uint64_t)grid[1] * 32ull <= (uint64_t)INT32_MAX;
   }
   return (n_u % 32u) == 0 && (k_u % 32u) == 0 && (m_u % 64ull) == 0 &&
          grid[1] != 0 && (uint64_t)grid[0] * 32ull == (uint64_t)n_u &&
          (uint64_t)grid[1] * 32ull == m_u && m_u <= (uint64_t)INT32_MAX;
}

#endif
