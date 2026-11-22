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
 */
uint32_t hk_barrier_classify(const VkDependencyInfo *info,
                             const struct hk_barrier_state *state);

const char *hk_barrier_split_reason_name(enum hk_barrier_split_reason reason);

bool hk_barrier_image_may_alias(VkImageCreateFlags2KHR flags,
                                VkImageTiling tiling,
                                VkExternalMemoryHandleTypeFlags external);

bool hk_barrier_attachment_may_write(VkImageLayout layout,
                                     VkImageAspectFlags aspect,
                                     VkAttachmentLoadOp load_op,
                                     VkResolveModeFlagBits resolve_mode);
