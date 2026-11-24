/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef HK_AGX3_AS_LAYOUT_H
#define HK_AGX3_AS_LAYOUT_H
#include <stdbool.h>
#include <stdint.h>
#include "hk_agx3_as_format.h"

/* Encoding scratch aliases the common builder's finished sort workspace.
 * All offsets are relative to the first byte after vk_ir_header. */
struct hk_agx3_as_layout {
   uint32_t nodes, jobs, envelope, state, node_status, envelope_status;
   /* An update's dispatches of inner jobs, one per depth (see
    * HK_AGX3_REFIT_LEVELS). */
   uint32_t refit_levels;
   uint32_t encode_bytes, node_bound, slot_bound;
   uint64_t as_bytes;
};

static inline bool
hk_agx3_as_layout_init_format(enum hk_agx3_as_format format, uint32_t leaves,
                              struct hk_agx3_as_layout *out)
{
   if (!out)
      return false;
   const uint32_t alignment = hk_agx3_as_image_align(format);
   if (!leaves) {
      *out = (struct hk_agx3_as_layout){.slot_bound = 3,
         .as_bytes = 192 + (alignment == HK_AGX3_GEN1_IMAGE_ALIGN ? 0 : alignment - 1)};
      return true;
   }
   const uint64_t nodes = (uint64_t)leaves + (leaves > 1 ? leaves : 2) - 1;
   /* A tree has jobs-1 child-root slots. Each leaf cluster contributes its
    * triangles plus ceil(triangles/4) QTB headers. With at most leaves leaf
    * jobs and leaves-1 inner jobs, end_slot <= 4*leaves+1 (including slots0..2).
    */
   const uint64_t slots = (uint64_t)leaves * 4 + 1;
   const uint64_t jobs = nodes * 16;
   const uint64_t envelope = jobs + nodes * 64;
   const uint64_t state = envelope + 48;
   /* GEN1 keeps the indirect encode arguments after the plan state. */
   const uint64_t status = state + hk_agx3_as_plan_tail(format);
   const uint64_t refit_levels = (status + nodes * 4 + 4 + 7) & ~UINT64_C(7);
   /* GEN1's current refit uses per-level dispatches. The M5 backend keeps
    * its independent IR-topology refit and needs no level-dispatch tail. */
   const uint64_t end = refit_levels +
      (format == HK_AGX3_AS_FORMAT_GEN1 ? 16 * 12 : 0);
   if (nodes > UINT32_MAX / 8 || slots > UINT32_MAX || end > UINT32_MAX)
      return false;
   *out = (struct hk_agx3_as_layout){
      .nodes = 0, .jobs = jobs, .envelope = envelope, .state = state,
      .node_status = status, .envelope_status = status + nodes * 4,
      .refit_levels = refit_levels,
      .encode_bytes = end, .node_bound = nodes, .slot_bound = slots,
      .as_bytes = slots * HK_AGX3_AS_SLOT_B + 184 +
                  (alignment == HK_AGX3_GEN1_IMAGE_ALIGN ? 0 : alignment - 1),
   };
   return true;
}
/* The nonempty envelope publishes readiness at native AS byte0xa0. Empty
 * metadata clears that word and publishes NaN bounds so parent builders
 * discard references to empty BLAS. This is not a hardware empty-node ABI. */
static inline bool
hk_agx3_as_layout_init(uint32_t leaves, struct hk_agx3_as_layout *out)
{
   return hk_agx3_as_layout_init_format(HK_AGX3_AS_FORMAT_M5, leaves, out);
}

static inline bool
hk_gen1_as_layout_init(uint32_t leaves, struct hk_agx3_as_layout *out)
{
   return hk_agx3_as_layout_init_format(HK_AGX3_AS_FORMAT_GEN1, leaves, out);
}

static inline void
hk_agx3_as_empty_header(uint32_t header[28])
{
   for (unsigned i = 0; i < 28; ++i)
      header[i] = i >= 4 && i < 10 ? 0x7fc00000u : 0;
}

/* GEN1 refit scheduling policy.
 *
 * Updates at or below HK_AGX3_REFIT_FUSED_MAX_LEAVES run every inner level
 * plus envelope finalization in one direct-dispatch workgroup
 * (hk_agx3_refit_top), avoiding per-level indirect dispatches and barriers.
 * Larger updates still run wide lower levels multi-workgroup, then finish the
 * top HK_AGX3_REFIT_TOP_LEVELS levels (at most 1 + 8 + 64 = 73 inner jobs)
 * and envelope in one hk_agx3_refit_top workgroup.
 */
#define HK_AGX3_REFIT_FUSED_MAX_LEAVES    8192u
#define HK_AGX3_REFIT_TOP_LEVELS          3u
#define HK_AGX3_REFIT_TOP_LANES           256u
#define HK_AGX3_REFIT_PREPARE_GROUP_SIZE  64u
#define HK_AGX3_REFIT_PREPARE_MAX_THREADS 4096u
#define HK_AGX3_REFIT_COPY_MAX_THREADS    65536u

static inline bool
hk_agx3_refit_fused_all(uint32_t leaf_node_count)
{
   return leaf_node_count <= HK_AGX3_REFIT_FUSED_MAX_LEAVES;
}

static inline uint32_t
hk_agx3_refit_top_levels(uint32_t leaf_node_count)
{
   return hk_agx3_refit_fused_all(leaf_node_count) ? 16u : HK_AGX3_REFIT_TOP_LEVELS;
}

static inline uint32_t
hk_agx3_refit_prepare_threads(uint32_t leaf_bound, uint32_t node_bound,
                              uint64_t retained_bytes, bool copy, bool instances)
{
   const uint64_t group = HK_AGX3_REFIT_PREPARE_GROUP_SIZE;
   uint64_t threads;
   if (copy) {
      threads = node_bound > leaf_bound ? node_bound : leaf_bound;
      const uint64_t words = retained_bytes / 4;
      if (words > threads)
         threads = words;
      if (threads > HK_AGX3_REFIT_COPY_MAX_THREADS)
         threads = HK_AGX3_REFIT_COPY_MAX_THREADS;
   } else if (instances) {
      threads = leaf_bound;
      if (threads > HK_AGX3_REFIT_COPY_MAX_THREADS)
         threads = HK_AGX3_REFIT_COPY_MAX_THREADS;
   } else if (hk_agx3_refit_fused_all(leaf_bound)) {
      threads = group;
   } else {
      threads = node_bound;
      if (threads > HK_AGX3_REFIT_PREPARE_MAX_THREADS)
         threads = HK_AGX3_REFIT_PREPARE_MAX_THREADS;
   }
   if (threads < group)
      threads = group;
   return (uint32_t)((threads + group - 1) & ~(group - 1));
}

static inline uint32_t
hk_agx3_refit_direct_leaf_invocations(uint32_t leaf_node_count)
{
   const uint32_t jobs_per_group = HK_AGX3_GEN1_NODE_GROUP_LANES / 8u;
   const uint32_t groups =
      leaf_node_count ? (leaf_node_count + jobs_per_group - 1u) / jobs_per_group : 1u;
   return groups * HK_AGX3_GEN1_NODE_GROUP_LANES;
}

/* Single-geometry, in-place small/medium triangle updates load vertices
 * directly inside the leaf-job encoder, skipping prepare, triangle placement
 * scratch writes/reads, and the intermediate barrier. */
static inline bool
hk_agx3_refit_uses_direct_triangles(bool skips_leaves, bool in_place,
                                    uint32_t geometry_count,
                                    uint32_t primitive_count,
                                    uint32_t leaf_node_count)
{
   return skips_leaves && in_place && geometry_count == 1u &&
          leaf_node_count > 0u && primitive_count == leaf_node_count &&
          hk_agx3_refit_fused_all(leaf_node_count);
}
#endif
