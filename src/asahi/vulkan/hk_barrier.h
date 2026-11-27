/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "vulkan/vulkan_core.h"

/* Reasons are independent bits: one split may have several causes. */
enum hk_barrier_split_reason {
   HK_BARRIER_SPLIT_GLOBAL,
   HK_BARRIER_SPLIT_TEXTURE,
   HK_BARRIER_SPLIT_SHADER,
   HK_BARRIER_SPLIT_SIDE_STREAM,
   HK_BARRIER_SPLIT_VERTEX,
   HK_BARRIER_SPLIT_COMPUTE_TRANSFER,
   HK_BARRIER_SPLIT_LAYOUT,
   HK_BARRIER_SPLIT_OWNERSHIP,
   HK_BARRIER_SPLIT_UNKNOWN,
   HK_BARRIER_SPLIT_COUNT,
};

struct hk_barrier_state {
   /* Cumulative across physical streams of the API render pass instance. */
   bool shader_writes;
   bool pre_graphics, post_graphics;
   bool attachments_alias;
   bool tilebuffer_spilled;
   /* Color input attachments can read on-chip tile memory in this pass. */
   bool color_input_local_read;
   /* At least one depth or stencil attachment may have pending writes. */
   bool depth_stencil_writes;
   /* Attachment images which may have pending writes, not read-only loads.
    * The first color_attachment_count entries are color attachments; any
    * remaining entries are depth or stencil attachments. */
   uint32_t color_attachment_count;
   uint32_t attachment_count;
   const VkImage *attachments;
};

/* CPU-only classification. Zero means the existing tile/stream ordering
 * satisfies the dependency; nonzero requires ending the physical render.
 * When color_input_local_read is set, BY_REGION color input attachment reads
 * stay in tile memory; depth/stencil and non-tile reads still split.
 * When used_local_read is non-NULL, it is set to true iff the barrier stayed
 * in the render pass (return value 0) and relied on color_input_local_read to
 * admit an INPUT_ATTACHMENT_READ on a written color attachment.
 */
uint32_t hk_barrier_classify_ex(const VkDependencyInfo *info,
                                const struct hk_barrier_state *state,
                                bool *used_local_read);

static inline uint32_t
hk_barrier_classify(const VkDependencyInfo *info,
                    const struct hk_barrier_state *state)
{
   return hk_barrier_classify_ex(info, state, NULL);
}

/* Whether a kept local-read barrier must split the physical render pass before
 * emitting a draw whose fragment shader reads input attachments via the DRAM
 * texture path rather than on-chip tile memory. */
static inline bool
hk_barrier_local_read_draw_needs_split(bool kept_local_read_barrier,
                                       bool in_meta,
                                       bool fs_has_input_attachments,
                                       bool uses_local_read_tile)
{
   return kept_local_read_barrier && !in_meta && fs_has_input_attachments &&
          !uses_local_read_tile;
}


/* Whether a sync2 dependency declares only local compute-to-compute storage
 * buffer ordering (no image barriers, no queue-family ownership transfer, no
 * non-compute stages, no non-storage accesses, no unknown pNext extensions).
 * Because Vulkan storage access bits also cover storage images, callers must
 * pair this with stream-level image tracking before narrowing the CDM barrier. */
bool hk_dependency_is_pure_compute_ssbo(const VkDependencyInfo *dep);

const char *hk_barrier_split_reason_name(enum hk_barrier_split_reason reason);

bool hk_barrier_image_may_alias(VkImageCreateFlags2KHR flags,
                                VkImageTiling tiling,
                                VkExternalMemoryHandleTypeFlags external);

bool hk_barrier_attachment_may_write(VkImageLayout layout,
                                     VkImageAspectFlags aspect,
                                     VkAttachmentLoadOp load_op,
                                     VkResolveModeFlagBits resolve_mode);
