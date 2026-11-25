/* SPDX-License-Identifier: MIT */
#pragma once

#include "hk_descriptor_set.h"
#include "hk_nir_match_pointwise_clamp.h"
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

#define HK_COMPUTE_PAIR_MAX_VARIANTS 16u

enum hk_compute_pair_alt_kind {
   HK_COMPUTE_PAIR_ALT_NONE = 0,
   HK_COMPUTE_PAIR_ALT_DEPTHWISE_VEC4 = 1,
   HK_COMPUTE_PAIR_ALT_GEMM_CMAT32 = 2,
   HK_COMPUTE_PAIR_ALT_GEMM_CMAT64 = 3,
   HK_COMPUTE_PAIR_ALT_GEMM_CMAT16 = 4,
};

struct hk_compute_pair_variant {
   struct hk_nir_clamp_epilogue epilogue;
   enum hk_compute_pair_alt_kind alt_kind;
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

struct hk_shader *
hk_compute_pair_get_or_compile_fused(struct hk_device *dev,
                                     struct hk_compute_pair_producer *prod,
                                     const struct hk_nir_clamp_epilogue *epilogue,
                                     enum hk_compute_pair_alt_kind alt_kind);

void hk_compute_pair_info_serialize(struct blob *blob,
                                    const struct hk_compute_pair_info *info);
bool hk_compute_pair_info_deserialize(struct blob_reader *blob,
                                      struct hk_compute_pair_info *info);

/* An immutable record of one original direct dispatch. Carries numeric backing
 * identities, the uploaded compute root/set VAs for submit-time fused CDM
 * emission, and a refcounted handle to the producer compilation state (for
 * producer roles only; NULL for clamps).
 */
struct hk_compute_pair_dispatch {
   struct hk_compute_pair_info info;
   struct hk_storage_backing bindings[4];
   uint32_t groups[3];       /* Original API workgroup counts (for footprint) */
   uint32_t exec_threads[3]; /* Effective global thread counts (for CDM emit) */
   uint16_t exec_local[3];   /* Effective logical workgroup size */
   uint64_t root_va;
   uint64_t root_sets[HK_MAX_SETS];
   uint8_t push[HK_MAX_PUSH_SIZE];
   enum hk_compute_pair_alt_kind alt_kind;
   struct hk_compute_pair_producer *producer;
};

bool hk_fused_compute_shader_eligible(const struct hk_device *dev,
                                      struct hk_shader *s,
                                      const struct hk_compute_pair_dispatch *p);

bool hk_emit_fused_compute_dispatch(struct hk_device *dev, struct hk_cs *cs,
                                    struct hk_shader *s,
                                    const struct hk_compute_pair_dispatch *p,
                                    uint8_t *ioto_map, uint64_t ioto_va,
                                    size_t ioto_cap, size_t *ioto_used_out);
