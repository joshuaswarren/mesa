/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_ray_tracing_core.h"

#include <string.h>

bool
hk_rt_group_from_create_info(struct hk_ray_tracing_group *out,
                             VkRayTracingShaderGroupTypeKHR type,
                             uint32_t general_shader,
                             uint32_t closest_hit_shader,
                             uint32_t any_hit_shader,
                             uint32_t intersection_shader)
{
   struct hk_ray_tracing_group group = {
      .recursive_index = VK_SHADER_UNUSED_KHR,
      .ahit_index = VK_SHADER_UNUSED_KHR,
      .isec_index = VK_SHADER_UNUSED_KHR,
   };

   switch (type) {
   case VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR:
      group.recursive_index = general_shader;
      break;

   case VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR:
      group.recursive_index = closest_hit_shader;
      group.ahit_index = any_hit_shader;
      break;

   case VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR:
      group.recursive_index = closest_hit_shader;
      /* An any-hit shader in a procedural group is only reachable from the
       * intersection shader's reportIntersectionEXT, so it is meaningless
       * without one.  The specification requires an intersection shader in a
       * procedural group; drop the any-hit rather than emit an unreachable
       * call if a layer ever hands us a group without one. */
      if (intersection_shader != VK_SHADER_UNUSED_KHR) {
         group.isec_index = intersection_shader;
         group.ahit_index = any_hit_shader;
      }
      break;

   default:
      return false;
   }

   *out = group;
   return true;
}

struct hk_ray_tracing_group
hk_rt_group_rebase(struct hk_ray_tracing_group src, uint32_t stage_base)
{
   struct hk_ray_tracing_group dst = src;

   if (src.recursive_index != VK_SHADER_UNUSED_KHR)
      dst.recursive_index = stage_base + src.recursive_index;
   if (src.ahit_index != VK_SHADER_UNUSED_KHR)
      dst.ahit_index = stage_base + src.ahit_index;
   if (src.isec_index != VK_SHADER_UNUSED_KHR)
      dst.isec_index = stage_base + src.isec_index;

   return dst;
}

void
hk_rt_write_group_handle(void *dst, struct hk_ray_tracing_group_handle handle)
{
   uint8_t *bytes = dst;

   memset(bytes, 0, HK_RAY_TRACING_GROUP_HANDLE_SIZE);

   bytes[0] = (uint8_t)(handle.index >> 0);
   bytes[1] = (uint8_t)(handle.index >> 8);
   bytes[2] = (uint8_t)(handle.index >> 16);
   bytes[3] = (uint8_t)(handle.index >> 24);
}

static uint32_t
max2(uint32_t a, uint32_t b)
{
   return a > b ? a : b;
}

static uint32_t
max3(uint32_t a, uint32_t b, uint32_t c)
{
   return max2(max2(a, b), c);
}

uint32_t
hk_rt_pipeline_stack_size(const struct hk_rt_stack_sizes *sizes,
                          uint32_t recursion_depth)
{
   const uint32_t first_level =
      max3(sizes->closest_hit, sizes->miss,
           sizes->intersection + sizes->any_hit);
   const uint32_t deeper_level = max2(sizes->closest_hit, sizes->miss);

   uint32_t total = sizes->raygen;

   if (recursion_depth >= 1) {
      total += first_level;
      total += (recursion_depth - 1) * deeper_level;
   }

   /* A callable may be invoked from any stage and may itself call another; the
    * specification bounds the chain only through maxRayRecursionDepth's
    * required minimum of 31, so reserve that many frames. */
   total += 31 * sizes->callable;

   return total;
}

static uint32_t
align16(uint32_t x)
{
   return (x + 15) & ~15u;
}

uint32_t
hk_rt_continuation_stack_size(const struct hk_rt_stack_sizes *sizes,
                              uint32_t recursion_depth, bool has_callables,
                              uint32_t traversing_frame, uint32_t *max_frame)
{
   const uint32_t chit_miss = align16(max2(sizes->closest_hit, sizes->miss));
   const uint32_t callable = align16(sizes->callable);
   const uint32_t traversal =
      align16(sizes->intersection) + align16(sizes->any_hit);

   uint32_t leaf = max3(chit_miss, callable, traversal);
   leaf = max2(leaf, align16(sizes->raygen));
   if (traversing_frame)
      leaf = max2(leaf, align16(traversing_frame) + traversal);

   uint32_t total = HK_RT_RETURN_RECORD_SIZE;
   total += align16(sizes->raygen) + HK_RT_RETURN_RECORD_SIZE;
   total += recursion_depth * (chit_miss + HK_RT_RETURN_RECORD_SIZE);
   if (has_callables)
      total += 31 * (callable + HK_RT_RETURN_RECORD_SIZE);
   total += leaf;

   *max_frame = leaf;
   return total;
}

uint32_t
hk_rt_group_stack_size(const struct hk_rt_stack_sizes *sizes,
                       const struct hk_ray_tracing_group *group,
                       VkShaderGroupShaderKHR which,
                       uint32_t recursive_stage_size)
{
   switch (which) {
   case VK_SHADER_GROUP_SHADER_GENERAL_KHR:
   case VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR:
      /* Both name the group's recursive shader; the caller resolves its stage
       * so we do not have to know whether it is raygen, miss, callable or
       * closest hit. */
      return group->recursive_index == VK_SHADER_UNUSED_KHR
                ? 0
                : recursive_stage_size;

   case VK_SHADER_GROUP_SHADER_ANY_HIT_KHR:
      return group->ahit_index == VK_SHADER_UNUSED_KHR ? 0 : sizes->any_hit;

   case VK_SHADER_GROUP_SHADER_INTERSECTION_KHR:
      return group->isec_index == VK_SHADER_UNUSED_KHR ? 0
                                                       : sizes->intersection;

   default:
      return 0;
   }
}

void
hk_rt_launch_group_count(uint32_t width, uint32_t height, uint32_t depth,
                         uint32_t out[3])
{
   /* Round up without overflowing: a launch of 0xffffffff rays is refused
    * by maxRayDispatchInvocationCount long before it gets here, but the
    * arithmetic must not wrap even so.  2D/3D launches (height > 1) use 8x4
    * tiles per 32-lane workgroup; 1D launches (height <= 1) keep 32x1 strips. */
   if (height > 1) {
      out[0] = (width >> HK_RT_TILE_2D_X_LOG2) +
               ((width & (HK_RT_TILE_2D_X - 1)) ? 1u : 0u);
      out[1] = (height >> HK_RT_TILE_2D_Y_LOG2) +
               ((height & (HK_RT_TILE_2D_Y - 1)) ? 1u : 0u);
      out[2] = depth;
   } else {
      out[0] = (width >> HK_RT_WORKGROUP_SIZE_X_LOG2) +
               ((width & (HK_RT_WORKGROUP_SIZE_X - 1)) ? 1u : 0u);
      out[1] = height;
      out[2] = depth;
   }
}
