/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "asahi/lib/agx_chip.h"
#include "vulkan/vulkan_core.h"
#include "hk_sync.h"
#include <stdbool.h>
#include <stdint.h>

/* Whether consecutive 2x/4x MSAA render-pass instances may continue in one
 * render stream.
 *
 * Both HAL200 (G16G, G17P) and HAL300/M5 (G17G, G17S, G17C) share the
 * partial-render per-sample store/reload programs required by render-pass
 * continuation. On chips whose hardware multisample background reload drops
 * sample bytes above AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE on
 * uncovered tiles (agx_chip_uncovered_tile_ms_reload_defect), wide
 * multisample tilebuffers stay unmerged so a continuation split or partial
 * render never reloads wide samples through bg.partial.
 */
static inline bool
hk_msaa_pass_merge_supported(enum agx_chip chip, uint32_t chip_id,
                             unsigned samples, unsigned sample_size_B)
{
   if (samples != 2 && samples != 4)
      return false;
   if (!agx_chip_native_hal200(chip) && !agx_chip_g17_m5(chip))
      return false;
   if (sample_size_B > AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE &&
       agx_chip_uncovered_tile_ms_reload_defect(chip, chip_id, samples))
      return false;
   return true;
}

/* Whether render-pass continuation may elide non-attachment layout transitions
 * and prove attachment-only layouts exclude descriptor reads. Shared across
 * HAL200 and M5. */
static inline bool
hk_pass_merge_layout_proof_supported(enum agx_chip chip)
{
   return agx_chip_native_hal200(chip) || agx_chip_g17_m5(chip);
}

/* How one barrier of a dependency recorded between two render pass instances
 * relates to continuing the first instance's tile pass with the second, in
 * one command buffer (hk_m5_pass_merge_begin()) or across the command
 * buffers of one submission (hk_render_pack.h). Ordered from weakest. */
enum hk_m5_merge_dep {
   /* It does not order the second instance's graphics work after the first
    * instance's, or only for attachment operations, which the tile pass
    * orders. */
   HK_M5_MERGE_DEP_LOCAL,
   /* It orders the second instance's work after the first instance's beyond
    * attachment operations (up to ALL_COMMANDS -> ALL_COMMANDS with every
    * memory access). One tile pass runs the second instance's vertex work
    * before the first instance's fragment work and stores the attachments
    * only at its end, so this holds only for a merge that is
    * memory-equivalent by construction, see hk_m5_ordered_refusal(). */
   HK_M5_MERGE_DEP_ORDERED,
   /* A layout transition or queue family ownership transfer between two
    * graphics scopes: never continued across. */
   HK_M5_MERGE_DEP_END,
};

static inline enum hk_m5_merge_dep
hk_m5_pass_merge_dependency(VkPipelineStageFlags2 src_stages,
                            VkPipelineStageFlags2 dst_stages,
                            VkAccessFlags2 dst_access, bool buffer,
                            bool transition)
{
   const VkPipelineStageFlags2 local_stages =
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
   const VkAccessFlags2 local_access =
      VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
      VK_ACCESS_2_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT;

   const bool from_gfx = hk_sync_kinds_src(src_stages) & HK_SYNC_GFX;
   const bool to_gfx =
      hk_sync_kinds_dst(dst_stages) & (HK_SYNC_GFX | HK_SYNC_GFX_FRAG);
   if (!from_gfx || !to_gfx)
      return HK_M5_MERGE_DEP_LOCAL;

   if (!buffer && !(dst_stages & ~local_stages) &&
       !(dst_access & ~local_access))
      return HK_M5_MERGE_DEP_LOCAL;

   return transition ? HK_M5_MERGE_DEP_END : HK_M5_MERGE_DEP_ORDERED;
}
/* Whether a barrier's first scope covers graphics work outside the
 * framebuffer-local attachment stages while its second scope reaches graphics
 * work. Such a barrier still classifies as HK_M5_MERGE_DEP_LOCAL when its
 * second scope is attachment-only (e.g. FRAGMENT_SHADER ->
 * COLOR_ATTACHMENT_OUTPUT), which is safe while continuing the same
 * attachments but would break a WAR/WAW barrier before a final-pass resolve
 * writing a separate image. */
static inline bool
hk_m5_pass_merge_non_local_src(VkPipelineStageFlags2 src_stages,
                               VkPipelineStageFlags2 dst_stages)
{
   const VkPipelineStageFlags2 local_stages =
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

   const bool from_non_local_gfx =
      hk_sync_kinds_src(src_stages & ~local_stages) & HK_SYNC_GFX;
   const bool to_gfx =
      hk_sync_kinds_dst(dst_stages) & (HK_SYNC_GFX | HK_SYNC_GFX_FRAG);
   return from_non_local_gfx && to_gfx;
}
/* Whether a vkCmdPipelineBarrier2 dependency's first synchronization scope
 * covers fragment-shader image reads (either directly or through a logically
 * later graphics or global stage). Folding a trailing vkCmdResolveImage2 into
 * the preceding render pass's EOT when its fragment shaders read images would
 * otherwise allow finished tiles to write the resolve destination while other
 * tiles still sample it. */
static inline bool
hk_resolve_fold_barrier_orders_fs(const VkDependencyInfo *info)
{
   const VkPipelineStageFlags2 fs_stages =
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
      VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT |
      VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;

   for (uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
      if (info->pMemoryBarriers[i].srcStageMask & fs_stages)
         return true;
   }
   for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; ++i) {
      if (info->pBufferMemoryBarriers[i].srcStageMask & fs_stages)
         return true;
   }
   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i) {
      if (info->pImageMemoryBarriers[i].srcStageMask & fs_stages)
         return true;
   }
   return false;
}

static inline bool
hk_resolve_fold_fs_hazard(bool fs_reads_images, bool fs_barrier_seen)
{
   return fs_reads_images && fs_barrier_seen;
}

/* Resulting rendering flags when hk_agx3_resume_suspended_pass() reopens a
 * suspended render pass instance in place in the same command buffer. */
static inline VkRenderingFlagBits
hk_agx3_resumed_render_flags(VkRenderingFlagBits prev_flags,
                             VkRenderingFlags resume_flags)
{
   return (prev_flags & ~VK_RENDERING_SUSPENDING_BIT) |
          (resume_flags &
           (VK_RENDERING_SUSPENDING_BIT |
            VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT));
}

/* Whether a render pass instance must stay off G17P direct colour resolve
 * (and preserve_resolve_outside). A pass that is suspending or resuming cannot
 * direct-resolve, and a pass reopened in-CB after beginning suspended
 * (resumed_in_cb) lacked begin-time direct-resolve setup (loaded_resolve ->
 * process_empty_tiles, fragment attachment tracking, and sync footprint), so
 * end-of-pass BG/EOT rebuilds must keep it on the post-pass resolve path. */
static inline bool
hk_render_forbids_direct_resolve(VkRenderingFlags flags, bool resumed_in_cb)
{
   return resumed_in_cb ||
          (flags & (VK_RENDERING_SUSPENDING_BIT |
                    VK_RENDERING_RESUMING_BIT)) != 0;
}

/* Whether a pipeline barrier can make render pass attachment writes from an
 * earlier instance available to later shaders. A barrier whose source scope
 * excludes attachment-output stages or attachment/memory writes cannot make
 * attachment writes visible, so a later instance still cannot read the first
 * instance's attachments through descriptors or aliases without racing. */
static inline bool
hk_m5_barrier_makes_att_visible(VkPipelineStageFlags2 src_stages,
                                VkAccessFlags2 src_access)
{
   const VkPipelineStageFlags2 att_stages =
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
      VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
   const VkAccessFlags2 att_access =
      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
      VK_ACCESS_2_MEMORY_WRITE_BIT;
   return (src_stages & att_stages) != 0 && (src_access & att_access) != 0;
}
