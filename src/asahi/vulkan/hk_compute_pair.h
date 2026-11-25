/* SPDX-License-Identifier: MIT */
#pragma once

#include "hk_descriptor_set.h"
#include "hk_nir_match_pointwise_clamp.h"
#include "hk_pack_loop_bound.h"
#include "hk_private.h"
#include "hk_shader.h"
#include "util/simple_mtx.h"
#include "util/u_inlines.h"
#include "vk_pipeline.h"

struct blob;
struct blob_reader;
struct hk_cs;
struct hk_device;
struct hk_shader;
struct nir_shader;
struct vk_descriptor_set_layout;

#define HK_COMPUTE_PAIR_MAX_VARIANTS 64u

enum hk_compute_pair_alt_kind {
   HK_COMPUTE_PAIR_ALT_NONE = 0,
   HK_COMPUTE_PAIR_ALT_DEPTHWISE_VEC4 = 1,
   HK_COMPUTE_PAIR_ALT_GEMM_CMAT32 = 2,
   HK_COMPUTE_PAIR_ALT_GEMM_CMAT64 = 3,
   HK_COMPUTE_PAIR_ALT_GEMM_CMAT16 = 4,
   HK_COMPUTE_PAIR_ALT_GEMM_CMAT16_UNGUARDED = 5,
   HK_COMPUTE_PAIR_ALT_GEMM_CMAT32_UNGUARDED = 6,
   HK_COMPUTE_PAIR_ALT_GEMM_CMAT64_UNGUARDED = 7,
   HK_COMPUTE_PAIR_ALT_LINEAR_FADD = 8,
   HK_COMPUTE_PAIR_ALT_GEMM_CMAT24_UNGUARDED = 9,
   HK_COMPUTE_PAIR_ALT_GEMM_CMAT48_UNGUARDED = 10,
   HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64 = 11,
   HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64_UNGUARDED = 12,
};

static inline bool
hk_compute_pair_alt_is_gemm_cmat(enum hk_compute_pair_alt_kind kind)
{
   return kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT16 ||
          kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT32 ||
          kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT64 ||
          kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64 ||
          kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT16_UNGUARDED ||
          kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT24_UNGUARDED ||
          kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT32_UNGUARDED ||
          kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT48_UNGUARDED ||
          kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT64_UNGUARDED ||
          kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64_UNGUARDED;
}

static inline uint32_t
hk_compute_pair_alt_gemm_tile_n(enum hk_compute_pair_alt_kind kind)
{
   switch (kind) {
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT16:
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT16_UNGUARDED:
      return 16;
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT24_UNGUARDED:
      return 24;
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT32:
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT32_UNGUARDED:
      return 32;
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT48_UNGUARDED:
      return 48;
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT64:
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT64_UNGUARDED:
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64:
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64_UNGUARDED:
      return 64;
   default:
      return 0;
   }
}

static inline uint32_t
hk_compute_pair_alt_gemm_tile_m(enum hk_compute_pair_alt_kind kind)
{
   switch (kind) {
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT64:
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT64_UNGUARDED:
      return 32;
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT16:
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT16_UNGUARDED:
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT24_UNGUARDED:
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT32:
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT32_UNGUARDED:
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT48_UNGUARDED:
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64:
   case HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64_UNGUARDED:
      return 64;
   default:
      return 0;
   }
}

static inline uint32_t
hk_compute_pair_alt_gemm_k_step(enum hk_compute_pair_alt_kind kind)
{
   return (kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT64 ||
           kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT64_UNGUARDED)
             ? 32u : 16u;
}

static inline bool
hk_compute_pair_alt_is_gemm_unguarded(enum hk_compute_pair_alt_kind kind)
{
   return kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT16_UNGUARDED ||
          kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT24_UNGUARDED ||
          kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT32_UNGUARDED ||
          kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT48_UNGUARDED ||
          kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT64_UNGUARDED ||
          kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64 ||
          kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64_UNGUARDED;
}

static inline bool
hk_compute_pair_alt_is_gemm_unguarded_for(enum hk_compute_pair_alt_kind kind,
                                          bool producer_is_guarded)
{
   return hk_compute_pair_alt_is_gemm_unguarded(kind) ||
          (!producer_is_guarded && hk_compute_pair_alt_is_gemm_cmat(kind));
}

struct hk_compute_pair_variant {
   bool has_epilogue;
   struct hk_nir_clamp_epilogue epilogue;
   bool has_fadd;
   struct hk_nir_fadd_epilogue fadd;
   bool im2col_a;
   bool has_dw_a;
   struct hk_nir_dw_a_cfg dw_a;
   bool has_gemm_dw;
   struct hk_nir_dw_a_cfg gemm_dw;
   enum hk_compute_pair_alt_kind alt_kind;
   enum hk_compute_pair_dual_write dual_write;
   struct hk_shader *shader;
};

/* Refcounted producer state shared between hk_api_shader and any recorded
 * command-buffer dispatch that may lazily fuse with a consumer clamp at submit
 * time. Holds references on the descriptor set layouts and owns its cloned
 * preprocessed NIR + lazily compiled fused shader variants, so it remains
 * valid even if the application destroys the pipeline/layouts after recording.
 */
struct hk_compute_pair_producer {
   struct pipe_reference ref;
   void (*destroy)(struct hk_compute_pair_producer *p);
   struct hk_device *dev;
   simple_mtx_t lock;
   struct hk_compute_pair_info info;
   struct nir_shader *nir;
   VkShaderCreateFlagsEXT flags;
   struct vk_pipeline_robustness_state robustness;
   VkShaderStageFlags next_stage_mask;
   uint32_t features;
   uint32_t set_layout_count;
   struct vk_descriptor_set_layout *set_layouts[HK_MAX_SETS];
   struct hk_compute_pair_variant variants[HK_COMPUTE_PAIR_MAX_VARIANTS];
   uint32_t variant_count;
};

static inline struct hk_compute_pair_producer *
hk_compute_pair_producer_ref(struct hk_compute_pair_producer *p)
{
   if (p)
      p_atomic_inc(&p->ref.count);
   return p;
}

static inline void
hk_compute_pair_producer_unref(struct hk_compute_pair_producer *p)
{
   if (p && pipe_reference(&p->ref, NULL))
      p->destroy(p);
}

/* The producer state of a depthwise or GEMM compute shader that `info`
 * compiles from `nir`, its NIR before hk_lower_nir(). NULL on failure. */
struct hk_compute_pair_producer *
hk_compute_pair_producer_create(struct hk_device *dev,
                                const struct vk_shader_compile_info *info,
                                uint32_t features,
                                const struct hk_compute_pair_info *pair,
                                const struct nir_shader *nir);

/* A depthwise or GEMM producer goes into a shader binary with what its fused
 * variants compile from: its NIR, compile options and set layouts.
 * Serialization returns false for any other producer, and for one whose set
 * layouts have immutable or embedded samplers; the shader then loads without
 * one, and is not fused at submission. */
bool hk_compute_pair_producer_serialize(struct blob *blob,
                                        const struct hk_compute_pair_producer *p);

/* The producer of the pair `pair` that hk_compute_pair_producer_serialize()
 * wrote, or NULL. */
struct hk_compute_pair_producer *
hk_compute_pair_producer_deserialize(struct hk_device *dev,
                                     struct blob_reader *blob,
                                     const struct hk_compute_pair_info *pair);

struct hk_shader *
hk_compute_pair_get_or_compile_fused(struct hk_device *dev,
                                     struct hk_compute_pair_producer *prod,
                                     const struct hk_nir_clamp_epilogue *epilogue,
                                     enum hk_compute_pair_alt_kind alt_kind,
                                     enum hk_compute_pair_dual_write dual_write,
                                     const struct hk_nir_fadd_epilogue *fadd_ep,
                                     bool im2col_a,
                                     const struct hk_nir_dw_a_cfg *dw_a,
                                     const struct hk_nir_dw_a_cfg *gemm_dw);

void hk_compute_pair_info_serialize(struct blob *blob,
                                    const struct hk_compute_pair_info *info);
bool hk_compute_pair_info_deserialize(struct blob_reader *blob,
                                      struct hk_compute_pair_info *info);

/* An immutable record of one original direct dispatch. Carries numeric backing
 * identities, the uploaded compute root/set VAs for submit-time fused CDM
 * emission, and a refcounted handle to the producer compilation state (for
 * producer roles only; NULL for clamps).
 */
struct hk_compute_pair_copy4 {
   uint64_t src_start;
   uint64_t dst_start;
   uint64_t bytes;
};

struct hk_compute_pair_pad4 {
   uint32_t in_shape[4];
   uint32_t out_shape[4];
   uint32_t pad_idx[4];
   uint64_t in_bytes;
   uint64_t out_bytes;
   uint64_t pad_bytes;
};

struct hk_compute_pair_im2col {
   uint64_t src_bytes;
   uint64_t dst_bytes;
   uint32_t iw;
   uint32_t ih;
   uint32_t kw;
   uint32_t cols;
   uint32_t step_x;
   uint32_t sh;
   uint32_t pad_x;
   uint32_t pad_y;
   uint32_t pw;
   uint32_t ph;
};

struct hk_compute_pair_dispatch {
   struct hk_compute_pair_info info;
   union {
      struct hk_compute_pair_copy4 copy4;
      struct hk_compute_pair_pad4 pad4;
      struct hk_compute_pair_im2col im2col;
   } extra;
   struct hk_storage_backing bindings[4];
   uint32_t groups[3];       /* Original API workgroup counts (for footprint) */
   uint32_t exec_threads[3]; /* Effective global thread counts (for CDM emit) */
   uint16_t exec_local[3];   /* Effective logical workgroup size */
   uint64_t root_va;
   uint64_t root_sets[HK_MAX_SETS];
   uint8_t push[HK_MAX_PUSH_SIZE];
   uint64_t cs_group_count_addr;
   uint32_t cs_base_group[3];
   uint64_t printf_buffer_address;
   enum hk_compute_pair_alt_kind alt_kind;
   bool overlap_safe;
   uint64_t pack_loop_params[HK_PACK_LOOP_PARAM_COUNT];
   struct hk_compute_pair_producer *producer;
};

bool hk_fused_compute_shader_eligible(const struct hk_device *dev,
                                      struct hk_shader *s,
                                      const struct hk_compute_pair_dispatch *p);

size_t hk_fused_compute_ioto_bound(const struct hk_device *dev,
                                   struct hk_shader *s,
                                   const struct hk_compute_pair_dispatch *p);

bool hk_emit_fused_compute_dispatch(struct hk_device *dev, struct hk_cs *cs,
                                    struct hk_shader *s,
                                    const struct hk_compute_pair_dispatch *p,
                                    uint8_t *ioto_map, uint64_t ioto_va,
                                    size_t ioto_cap, size_t *ioto_used_out);

static inline bool
hk_copy4_intervals_safe(const struct hk_storage_backing *src, uint64_t src_start,
                        const struct hk_storage_backing *dst, uint64_t dst_start,
                        uint64_t bytes, bool allow_exact_inplace)
{
   const uint64_t s_bo0 = src->offset + src_start, s_bo1 = s_bo0 + bytes;
   const uint64_t d_bo0 = dst->offset + dst_start, d_bo1 = d_bo0 + bytes;
   const uint64_t s_va0 = src->address + src_start, s_va1 = s_va0 + bytes;
   const uint64_t d_va0 = dst->address + dst_start, d_va1 = d_va0 + bytes;

   if (allow_exact_inplace &&
       src->handle == dst->handle && s_bo0 == d_bo0 && s_va0 == d_va0)
      return true;

   const bool bo_overlap =
      src->handle == dst->handle && s_bo0 < d_bo1 && d_bo0 < s_bo1;
   const bool va_overlap = s_va0 < d_va1 && d_va0 < s_va1;
   return !bo_overlap && !va_overlap;
}

static inline bool
hk_copy4_classify_intervals(bool kind_allows_inplace, unsigned nsrc,
                            const struct hk_storage_backing backings[3],
                            uint64_t src_start, uint64_t dst_start, uint64_t bytes,
                            bool *same_src0_dst_out)
{
   const bool same_src0_dst =
      nsrc == 1 && backings[0].handle == backings[2].handle &&
      backings[0].offset + src_start == backings[2].offset + dst_start;
   const bool allow_inplace = same_src0_dst || kind_allows_inplace;
   if (!hk_copy4_intervals_safe(&backings[0], src_start, &backings[2], dst_start,
                                bytes, allow_inplace))
      return false;
   if (nsrc == 2 &&
       !hk_copy4_intervals_safe(&backings[1], src_start, &backings[2], dst_start,
                                bytes, allow_inplace))
      return false;
   if (same_src0_dst_out)
      *same_src0_dst_out = same_src0_dst;
   return true;
}
