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
};

enum hk_compute_pair_dual_write {
   HK_COMPUTE_PAIR_DUAL_NONE = 0,
   HK_COMPUTE_PAIR_DUAL_COPY = 1,
   HK_COMPUTE_PAIR_DUAL_PAD4 = 2,
};

static inline uint32_t
hk_dual_write_push_offset(uint32_t push_bytes)
{
   return ALIGN_POT(push_bytes, 8u);
}

struct hk_dual_copy_push {
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
};

#define HK_COMPUTE_PAIR_INFO_VERSION 5u

struct hk_compute_pair_info {
   uint32_t version;
   uint32_t role;
   union {
      struct hk_nir_depthwise_output depthwise;
      struct hk_nir_gemm_output gemm;
      struct hk_nir_pointwise_clamp clamp;
      uint32_t fadd_push_bytes;
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
struct nir_shader *
hk_nir_build_depthwise_vec4(const struct nir_shader *producer,
                            const struct hk_nir_depthwise_output *info,
                            const struct hk_nir_clamp_epilogue *epilogue,
                            enum hk_compute_pair_dual_write dual_write);

/* Build a Direct-B 4 KiB XOR-swizzled Shared-A f32matmad8x8 GEMM NIR alternate
 * (tile_n == 16, 32, or 64) from a certified row-major GEMM producer (with
 * cmat_valid == true). When `epilogue` is non-NULL, folds that exact clamp
 * epilogue into output writeback; when `dual_write` is non-zero, also writes
 * the fused copy4 or pad4 destination buffer directly from registers. */
struct nir_shader *
hk_nir_build_gemm_cmat(const struct nir_shader *producer,
                       const struct hk_nir_gemm_output *info,
                       uint32_t tile_n,
                       const struct hk_nir_clamp_epilogue *epilogue,
                       enum hk_compute_pair_dual_write dual_write);

/* M-tile height for each cooperative-matrix GEMM variant:
 *   tile_n == 16: 32x16, K_step=32 (4 KiB shared A, 128 threads)
 *   tile_n == 32: 64x32, K_step=16 (4 KiB shared A, 128 threads)
 *   tile_n == 64: 32x64, K_step=32 (4 KiB shared A, 128 threads) */
static inline uint32_t
hk_gemm_cmat_tile_m(uint32_t tile_n)
{
   return tile_n == 32u ? 64u : 32u;
}

static inline uint32_t
hk_gemm_cmat_k_step(uint32_t tile_n)
{
   return tile_n == 32u ? 16u : 32u;
}

/* Choose cooperative-matrix tile_n (16, 32, or 64) for a guarded GEMM dispatch.
 * - N <= 16: cmat16 (32x16, K_step=32) avoids empty N-columns.
 * - N > 32 with even ceil(N/32) and K >= 32: cmat64 (32x64, K_step=32) halves
 *   workgroup barriers and doubles M-parallelism without extra N-padding waste.
 * - Otherwise: cmat32 (64x32, K_step=16). */
static inline uint32_t
hk_gemm_cmat_select_tile_n(uint32_t n_u, uint32_t k_u,
                           bool has16, bool has32, bool has64)
{
   const uint32_t n_tiles32 = (n_u + 31u) / 32u;
   if (n_u <= 16u && has16)
      return 16u;
   if (n_u > 32u && (n_tiles32 % 2u) == 0u && k_u >= 32u && has64)
      return 64u;
   if (has32)
      return 32u;
   if (has64)
      return 64u;
   if (has16)
      return 16u;
   return 0u;
}

/* Return true when a guarded GEMM dispatch with validated descriptor ranges
 * may safely specialize to the unguarded cooperative-matrix variant. */
static inline bool
hk_gemm_cmat_can_specialize_unguarded(bool is_guarded, uint64_t m_u,
                                      uint64_t n_u, uint64_t k_u,
                                      uint32_t tile_n, uint32_t a_offset_B,
                                      uint64_t a_base_addr,
                                      uint64_t a_base_offset)
{
   if (!is_guarded || m_u == 0 || n_u == 0 || k_u == 0)
      return false;
   if (tile_n != 16u && tile_n != 32u && tile_n != 64u)
      return false;
   const uint32_t tile_m = hk_gemm_cmat_tile_m(tile_n);
   const uint32_t k_step = hk_gemm_cmat_k_step(tile_n);
   return (m_u % tile_m) == 0 && (n_u % tile_n) == 0 && (k_u % k_step) == 0 &&
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
