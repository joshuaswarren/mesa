/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_barrier.h"

#define REASON(r) (1u << HK_BARRIER_SPLIT_##r)

bool
hk_barrier_attachment_may_write(VkImageLayout layout, VkImageAspectFlags aspect,
                                VkAttachmentLoadOp load_op,
                                VkResolveModeFlagBits resolve_mode)
{
   /* A clear changes contents even if subsequent draws are read-only. Keep
    * unknown/non-load operations and resolves conservative as well. */
   if (load_op != VK_ATTACHMENT_LOAD_OP_LOAD ||
       resolve_mode != VK_RESOLVE_MODE_NONE)
      return true;

   const bool depth = aspect == VK_IMAGE_ASPECT_DEPTH_BIT;
   const bool stencil = aspect == VK_IMAGE_ASPECT_STENCIL_BIT;
   if (!depth && !stencil)
      return true;

   switch (layout) {
   case VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL:
   case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
      return false;
   case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL:
   case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL:
      return !depth;
   case VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL:
   case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL:
      return !stencil;
   default:
      return true;
   }
}

bool
hk_barrier_image_may_alias(VkImageCreateFlags2KHR flags, VkImageTiling tiling,
                           VkExternalMemoryHandleTypeFlags external)
{
   /* External-memory images behave as if ALIAS were set. Linear images can
    * share a host-accessible representation without ALIAS, and modifiers
    * can be linear too. Only ordinary optimal images admit the proof. */
   return (flags &
           (VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_SPARSE_BINDING_BIT)) ||
          tiling != VK_IMAGE_TILING_OPTIMAL || external != 0;
}

static const VkPipelineStageFlags2 framebuffer_stages =
   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
   VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

static const VkPipelineStageFlags2 vertex_stages =
   VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
   VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |
   VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
   VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
   VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |
   VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT |
   VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT |
   VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT |
   VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT |
   VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
   VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;

static const VkPipelineStageFlags2 external_stages =
   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
   VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT |
   VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT |
   VK_PIPELINE_STAGE_2_CLEAR_BIT;

static const VkAccessFlags2 attachment_access =
   VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
   VK_ACCESS_2_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT |
   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

static const VkAccessFlags2 attachment_writes =
   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

static const VkAccessFlags2 shader_writes =
   VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
   VK_ACCESS_2_TRANSFORM_FEEDBACK_WRITE_BIT_EXT |
   VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT;

static const VkAccessFlags2 image_access =
   VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
   VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
   VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
   VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT |
   VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
   VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_HOST_WRITE_BIT;

/* Deliberately a whitelist. New access types must not silently become no-ops. */
static const VkAccessFlags2 read_access =
   VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT |
   VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT |
   VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
   VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
   VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
   VK_ACCESS_2_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT |
   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
   VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_HOST_READ_BIT |
   VK_ACCESS_2_MEMORY_READ_BIT |
   VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT;

static VkPipelineStageFlags2
expand_stages(VkPipelineStageFlags2 stages, bool source)
{
   const VkPipelineStageFlags2 all =
      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT |
      (source ? VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT
              : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT);
   if (stages & all)
      return vertex_stages | framebuffer_stages | external_stages;

   if (stages & VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT)
      stages |= vertex_stages | framebuffer_stages;

   return stages & ~(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
                     VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT |
                     VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
}

static uint32_t
classify_scope(const struct hk_barrier_state *state, VkDependencyFlags flags,
               VkPipelineStageFlags2 src, VkAccessFlags2 src_access,
               VkPipelineStageFlags2 dst, VkAccessFlags2 dst_access,
               bool attachment, bool ds_attachment, bool buffer,
               bool *used_local_read)
{
   src = expand_stages(src, true);
   dst = expand_stages(dst, false);

   const VkPipelineStageFlags2 known =
      vertex_stages | framebuffer_stages | external_stages;

   /* Non-graphics work is not normally legal within a render pass. Keep a
    * conservative fallback for internal users and future extensions. */
   uint32_t reasons = 0;
   if ((src | dst) & external_stages)
      reasons |= REASON(COMPUTE_TRANSFER);
   if ((src | dst) & ~known)
      reasons |= REASON(UNKNOWN);

   /* Vulkan framebuffer regions constrain execution as well as memory.
    * A global fragment dependency cannot be implemented by tile ordering.
    * Fragment -> vertex also cannot run in one tiling/fragment job. */
   if ((src & framebuffer_stages) && dst &&
       (!(flags & VK_DEPENDENCY_BY_REGION_BIT) || (dst & ~framebuffer_stages)))
      reasons |= REASON(GLOBAL);

   /* There is no established vertex-to-vertex execution fence inside a
    * native render. Also, future software draws could otherwise append work
    * before this render's source scope. Do not rely on submission order.
    * Vertex -> fragment is already ordered by the tiling/fragment boundary.
    */
   if (src && (dst & vertex_stages))
      reasons |= REASON(VERTEX);

   /* A side stream alone is not a hazard. Only dependencies targeting
    * software vertex work can require ending it: appending that work to a
    * stream ahead of this render would put it before its source scope.
    * Framebuffer-local dependencies never order those software stages. */
   if (src && (dst & vertex_stages) &&
       (state->pre_graphics || state->post_graphics))
      reasons |= REASON(SIDE_STREAM);

   const VkAccessFlags2 known_access =
      read_access | attachment_writes | shader_writes |
      VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
      VK_ACCESS_2_HOST_WRITE_BIT;
   if ((src_access | dst_access) & ~known_access)
      reasons |= REASON(UNKNOWN);
   if (src_access &
       (VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT))
      reasons |= REASON(COMPUTE_TRANSFER);

   /* Visibility also exposes writes made available by earlier dependencies.
    * A read-only/empty source access mask alone does not prove that a store
    * is unnecessary (e.g. local CW -> CR, then CR -> input read). Use only
    * positive knowledge of what this pass can write.
    * No buffer/image binding history is needed to prove an unrelated image
    * unwritten when all shaders are read-only and attachments cannot alias.
    */
   const bool publishes_shader =
      (src_access & (shader_writes | VK_ACCESS_2_MEMORY_WRITE_BIT)) != 0;
   bool writes_shader =
      state->shader_writes && (publishes_shader || dst_access);
   /* Internal side streams write private working memory. Their existence
    * alone does not make an attachment dependency a storage dependency. */
   writes_shader |=
      (state->pre_graphics || state->post_graphics) && publishes_shader;
   bool writes_attachment =
      attachment &&
      ((src_access & (attachment_writes | VK_ACCESS_2_MEMORY_WRITE_BIT)) ||
       (dst_access & image_access) || (state->attachments_alias && dst_access));

   if (writes_shader)
      reasons |= REASON(SHADER);

   if (writes_attachment) {
      /* Fixed-function attachment operations observe the preceding pixel's
       * attachment value within the tile pass. When color_input_local_read is
       * set, BY_REGION color input attachment reads also read on-chip tile
       * memory directly (with a draw-time fallback split if any draw falls back
       * to texture fetches). Depth/stencil writes and generic shader/sampled
       * reads still require a memory store. */
      const bool ds_src_write =
         (src_access & VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) != 0 ||
         (state->depth_stencil_writes &&
          ((src_access & VK_ACCESS_2_MEMORY_WRITE_BIT) != 0 ||
           (src & (VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)) != 0));
      const bool allow_color_input_local_read =
         state->color_input_local_read && !ds_attachment && !ds_src_write;
      const VkAccessFlags2 local_dst_access =
         attachment_access |
         (allow_color_input_local_read ? VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT
                                       : 0);
      const bool local_attachment =
         !buffer && !state->tilebuffer_spilled && !state->attachments_alias &&
         (flags & VK_DEPENDENCY_BY_REGION_BIT) &&
         !(src & ~framebuffer_stages) && !(dst & ~framebuffer_stages) &&
         dst_access && !(dst_access & ~local_dst_access);
      if (!local_attachment)
         reasons |= REASON(TEXTURE);
      else if (used_local_read && allow_color_input_local_read &&
               (dst_access & VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT))
         *used_local_read = true;
   }

   return reasons;
}

uint32_t
hk_barrier_classify_ex(const VkDependencyInfo *info,
                       const struct hk_barrier_state *state,
                       bool *used_local_read)
{
   const VkDependencyFlags known_flags = VK_DEPENDENCY_BY_REGION_BIT |
                                         VK_DEPENDENCY_VIEW_LOCAL_BIT |
                                         VK_DEPENDENCY_FEEDBACK_LOOP_BIT_EXT;
   uint32_t reasons = (info->pNext || (info->dependencyFlags & ~known_flags))
                         ? REASON(UNKNOWN)
                         : 0;
   bool any_local_read = false;

   for (uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
      const VkMemoryBarrier2 *b = &info->pMemoryBarriers[i];
      reasons |= classify_scope(
         state, info->dependencyFlags, b->srcStageMask, b->srcAccessMask,
         b->dstStageMask, b->dstAccessMask,
         state->attachment_count || state->attachments_alias, false, false,
         &any_local_read);
      if (b->pNext)
         reasons |= REASON(UNKNOWN);
   }

   for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; ++i) {
      const VkBufferMemoryBarrier2 *b = &info->pBufferMemoryBarriers[i];
      reasons |= classify_scope(
         state, info->dependencyFlags, b->srcStageMask, b->srcAccessMask,
         b->dstStageMask, b->dstAccessMask, state->attachments_alias, false,
         true, &any_local_read);
      if (b->srcQueueFamilyIndex != b->dstQueueFamilyIndex)
         reasons |= REASON(OWNERSHIP);
      if (b->pNext)
         reasons |= REASON(UNKNOWN);
   }

   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i) {
      const VkImageMemoryBarrier2 *b = &info->pImageMemoryBarriers[i];
      bool attachment = state->attachments_alias;
      bool ds_attachment =
         (b->subresourceRange.aspectMask &
          (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0;
      for (uint32_t a = 0; a < state->attachment_count; ++a) {
         if (b->image == state->attachments[a]) {
            attachment = true;
            if (a >= state->color_attachment_count)
               ds_attachment = true;
         }
      }
      reasons |= classify_scope(state, info->dependencyFlags, b->srcStageMask,
                                b->srcAccessMask, b->dstStageMask,
                                b->dstAccessMask, attachment, ds_attachment,
                                false, &any_local_read);
      /* Equal layouts, including UNDEFINED, are not transitions (sync2).
       * Keep all real transitions conservative; layout-specific data movement
       * is handled by the existing render continuation path. */
      if (b->oldLayout != b->newLayout)
         reasons |= REASON(LAYOUT);
      if (b->srcQueueFamilyIndex != b->dstQueueFamilyIndex)
         reasons |= REASON(OWNERSHIP);
      if (b->pNext)
         reasons |= REASON(UNKNOWN);
   }

   if (used_local_read)
      *used_local_read = (reasons == 0) && any_local_read;
   return reasons;
}

const char *
hk_barrier_split_reason_name(enum hk_barrier_split_reason reason)
{
   static const char *const names[HK_BARRIER_SPLIT_COUNT] = {
      "framebuffer_global",
      "attachment_memory",
      "shader_memory",
      "side_stream",
      "vertex_order",
      "compute_transfer",
      "layout",
      "ownership",
      "unknown",
   };
   return names[reason];
}

bool
hk_dependency_is_pure_compute_ssbo(const VkDependencyInfo *dep)
{
   if (!dep || dep->pNext != NULL || dep->imageMemoryBarrierCount != 0)
      return false;

   if (dep->memoryBarrierCount == 0 && dep->bufferMemoryBarrierCount == 0)
      return false;

   const VkPipelineStageFlags2 required_stage =
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
   const VkAccessFlags2 allowed_access =
      VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
      VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

   for (uint32_t i = 0; i < dep->memoryBarrierCount; ++i) {
      const VkMemoryBarrier2 *b = &dep->pMemoryBarriers[i];
      if (b->pNext != NULL ||
          b->srcStageMask != required_stage ||
          b->dstStageMask != required_stage ||
          (b->srcAccessMask & ~allowed_access) != 0 ||
          (b->dstAccessMask & ~allowed_access) != 0)
         return false;
   }

   for (uint32_t i = 0; i < dep->bufferMemoryBarrierCount; ++i) {
      const VkBufferMemoryBarrier2 *b = &dep->pBufferMemoryBarriers[i];
      if (b->pNext != NULL ||
          b->srcQueueFamilyIndex != b->dstQueueFamilyIndex ||
          b->srcQueueFamilyIndex == VK_QUEUE_FAMILY_EXTERNAL ||
          b->srcQueueFamilyIndex == VK_QUEUE_FAMILY_FOREIGN_EXT ||
          b->srcStageMask != required_stage ||
          b->dstStageMask != required_stage ||
          (b->srcAccessMask & ~allowed_access) != 0 ||
          (b->dstAccessMask & ~allowed_access) != 0)
         return false;
   }

   return true;
}
