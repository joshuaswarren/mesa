/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Host unit tests for hk_ray_tracing_core launch group decomposition and the
 * 2D 8x4 / 1D 32x1 invocation-id mapping shared with hk_ray_tracing_pipeline.c
 * and libagx/draws.cl across direct and indirect dispatch paths.
 */

#include "hk_ray_tracing_core.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#endif

/* Exact mirror of libagx_trace_rays_group_count() in src/asahi/libagx/draws.cl
 * used by libagx_trace_rays_indirect_setup, libagx_trace_rays_select_setup,
 * and libagx_trace_rays_select_launch. */
static void
indirect_group_count(uint32_t width, uint32_t height, uint32_t depth,
                     uint32_t local_size_x_log2,
                     VkDispatchIndirectCommand *out)
{
   if (local_size_x_log2 == 5u && height > 1u) {
      out->x = (width >> 3u) + ((width & 7u) ? 1u : 0u);
      out->y = (height >> 2u) + ((height & 3u) ? 1u : 0u);
      out->z = depth;
      return;
   }
   const uint32_t mask = (1u << local_size_x_log2) - 1u;
   out->x = (width >> local_size_x_log2) + ((width & mask) ? 1u : 0u);
   out->y = height;
   out->z = depth;
}

/* Host model of libagx_trace_rays_select_setup() for multi-kernel indirect
 * dispatch: only the kernel matching `selected_handle` receives the tiled
 * group count; every other kernel receives (0, 0, 0). */
static void
indirect_select_setup(uint32_t width, uint32_t height, uint32_t depth,
                      const uint32_t *handles, uint32_t kernel_count,
                      uint32_t selected_handle,
                      VkDispatchIndirectCommand *out_groups)
{
   VkDispatchIndirectCommand groups;
   indirect_group_count(width, height, depth, HK_RT_WORKGROUP_SIZE_X_LOG2,
                        &groups);
   for (uint32_t k = 0; k < kernel_count; ++k) {
      const bool run = handles[k] == selected_handle;
      out_groups[k] = (VkDispatchIndirectCommand){
         .x = run ? groups.x : 0u,
         .y = run ? groups.y : 0u,
         .z = run ? groups.z : 0u,
      };
   }
}

static void
check_groups(uint32_t w, uint32_t h, uint32_t d,
             uint32_t exp_x, uint32_t exp_y, uint32_t exp_z)
{
   uint32_t out[3] = {0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu};
   hk_rt_launch_group_count(w, h, d, out);
   assert(out[0] == exp_x);
   assert(out[1] == exp_y);
   assert(out[2] == exp_z);

   /* Indirect single-kernel setup must produce the identical grid. */
   VkDispatchIndirectCommand ind = {0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu};
   indirect_group_count(w, h, d, HK_RT_WORKGROUP_SIZE_X_LOG2, &ind);
   assert(ind.x == exp_x);
   assert(ind.y == exp_y);
   assert(ind.z == exp_z);

   /* Multi-kernel indirect select setup must route the exact grid only to the
    * selected raygen kernel. */
   const uint32_t handles[3] = {11u, 22u, 33u};
   VkDispatchIndirectCommand sel[3];
   indirect_select_setup(w, h, d, handles, 3, handles[1], sel);
   assert(sel[0].x == 0 && sel[0].y == 0 && sel[0].z == 0);
   assert(sel[1].x == exp_x && sel[1].y == exp_y && sel[1].z == exp_z);
   assert(sel[2].x == 0 && sel[2].y == 0 && sel[2].z == 0);
}

/* Verify that every (x, y, z) in [0, w) x [0, h) x [0, d) is visited exactly
 * once by the shader's 8x4 (h > 1) or 32x1 (h <= 1) mapping over the
 * threadgroup grid produced by both direct hk_rt_launch_group_count() and
 * indirect libagx_trace_rays_group_count(), and that the exact number of
 * trailing partial-tile lanes are retired out of bounds. */
static void
check_bijection(uint32_t w, uint32_t h, uint32_t d)
{
   uint32_t groups[3];
   hk_rt_launch_group_count(w, h, d, groups);

   VkDispatchIndirectCommand ind;
   indirect_group_count(w, h, d, HK_RT_WORKGROUP_SIZE_X_LOG2, &ind);
   assert(ind.x == groups[0] && ind.y == groups[1] && ind.z == groups[2]);

   const size_t total = (size_t)w * (size_t)h * (size_t)d;
   const size_t launched = (size_t)groups[0] * (size_t)groups[1] *
                           (size_t)groups[2] * HK_RT_WORKGROUP_SIZE_X;
   assert(launched >= total);

   uint8_t *seen = calloc(total ? total : 1, 1);
   assert(seen != NULL);

   size_t in_bounds = 0, oob = 0;
   const bool tile_2d = h > 1;
   for (uint32_t gz = 0; gz < groups[2]; ++gz) {
      for (uint32_t gy = 0; gy < groups[1]; ++gy) {
         for (uint32_t gx = 0; gx < groups[0]; ++gx) {
            for (uint32_t lid = 0; lid < HK_RT_WORKGROUP_SIZE_X; ++lid) {
               const uint32_t x = tile_2d
                  ? ((gx << HK_RT_TILE_2D_X_LOG2) +
                     (lid & (HK_RT_TILE_2D_X - 1)))
                  : ((gx << HK_RT_WORKGROUP_SIZE_X_LOG2) + lid);
               const uint32_t y = tile_2d
                  ? ((gy << HK_RT_TILE_2D_Y_LOG2) +
                     (lid >> HK_RT_TILE_2D_X_LOG2))
                  : gy;
               const uint32_t z = gz;

               if (x >= w || y >= h || z >= d) {
                  oob++;
                  continue;
               }

               const size_t idx =
                  ((size_t)z * (size_t)h + (size_t)y) * (size_t)w + (size_t)x;
               assert(seen[idx] == 0);
               seen[idx] = 1;
               in_bounds++;
            }
         }
      }
   }

   assert(in_bounds == total);
   assert(oob == launched - total);
   for (size_t i = 0; i < total; ++i)
      assert(seen[i] == 1);

   free(seen);
}

/* Group indices and slot stages, as creation and the binary/cache loader
 * check them (hk_rt_group_indices_valid(), hk_rt_group_stages_valid()). */
static void
check_group_validation(void)
{
   const VkShaderStageFlagBits stages[] = {
      VK_SHADER_STAGE_RAYGEN_BIT_KHR,       VK_SHADER_STAGE_MISS_BIT_KHR,
      VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,  VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
      VK_SHADER_STAGE_INTERSECTION_BIT_KHR, VK_SHADER_STAGE_CALLABLE_BIT_KHR,
   };
   const uint32_t n = ARRAY_SIZE(stages);
   const uint32_t unused = VK_SHADER_UNUSED_KHR;
   const struct hk_ray_tracing_group valid[] = {
      {.recursive_index = 0, .ahit_index = unused, .isec_index = unused},
      {.recursive_index = 1, .ahit_index = unused, .isec_index = unused},
      {.recursive_index = 5, .ahit_index = unused, .isec_index = unused},
      {.recursive_index = 2, .ahit_index = 3, .isec_index = unused},
      {.recursive_index = 2, .ahit_index = 3, .isec_index = 4},
      {.recursive_index = unused, .ahit_index = unused, .isec_index = 4},
      {.recursive_index = unused, .ahit_index = unused, .isec_index = unused},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(valid); ++i) {
      assert(hk_rt_group_indices_valid(&valid[i], n));
      assert(hk_rt_group_stages_valid(&valid[i], n, stages));
   }

   /* Every slot: one past the end, further, and the largest index that is
    * not VK_SHADER_UNUSED_KHR. */
   const uint32_t past[] = {n, n + 1, unused - 1};
   for (unsigned i = 0; i < ARRAY_SIZE(past); ++i) {
      for (unsigned slot = 0; slot < 3; ++slot) {
         struct hk_ray_tracing_group g = valid[4];
         uint32_t *index = slot == 0   ? &g.recursive_index
                           : slot == 1 ? &g.ahit_index
                                       : &g.isec_index;
         *index = past[i];
         assert(!hk_rt_group_indices_valid(&g, n));
         assert(!hk_rt_group_stages_valid(&g, n, stages));
      }
   }

   /* No index is in range of an empty stage array. */
   assert(!hk_rt_group_indices_valid(&valid[0], 0));
   assert(hk_rt_group_indices_valid(&valid[6], 0));

   /* In range, but a shader of a stage the slot does not take. */
   const struct hk_ray_tracing_group wrong[] = {
      {.recursive_index = 3, .ahit_index = unused, .isec_index = unused},
      {.recursive_index = 4, .ahit_index = unused, .isec_index = unused},
      {.recursive_index = 2, .ahit_index = 0, .isec_index = unused},
      {.recursive_index = 2, .ahit_index = 4, .isec_index = unused},
      {.recursive_index = 2, .ahit_index = 3, .isec_index = 2},
      {.recursive_index = 2, .ahit_index = 3, .isec_index = 3},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(wrong); ++i) {
      assert(hk_rt_group_indices_valid(&wrong[i], n));
      assert(!hk_rt_group_stages_valid(&wrong[i], n, stages));
   }
}

int
main(void)
{
   check_group_validation();

   /* Zero dimensions produce zero workgroups on the zero axis. */
   check_groups(0, 0, 0, 0, 0, 0);
   check_groups(0, 1080, 1, 0, 270, 1);
   check_groups(1920, 0, 1, 60, 0, 1);
   check_groups(1920, 1080, 0, 240, 270, 0);

   /* 1D launches (height == 1) use flat 32x1 strips. */
   check_groups(1, 1, 1, 1, 1, 1);
   check_groups(31, 1, 1, 1, 1, 1);
   check_groups(32, 1, 1, 1, 1, 1);
   check_groups(33, 1, 1, 2, 1, 1);
   check_groups(100, 1, 7, 4, 1, 7);

   /* 2D/3D launches (height > 1) use 8x4 tiles, including partial edge tiles
    * at heights 2 (partial single tile row), 4 (full single tile row), and 5
    * (first two-row launch with a 1-pixel partial bottom edge). */
   check_groups(1, 2, 1, 1, 1, 1);
   check_groups(7, 2, 1, 1, 1, 1);
   check_groups(8, 2, 1, 1, 1, 1);
   check_groups(9, 2, 1, 2, 1, 1);
   check_groups(17, 2, 2, 3, 1, 2);

   check_groups(1, 4, 1, 1, 1, 1);
   check_groups(7, 4, 1, 1, 1, 1);
   check_groups(8, 4, 1, 1, 1, 1);
   check_groups(9, 4, 1, 2, 1, 1);
   check_groups(17, 4, 2, 3, 1, 2);

   check_groups(1, 5, 1, 1, 2, 1);
   check_groups(7, 5, 1, 1, 2, 1);
   check_groups(8, 5, 1, 1, 2, 1);
   check_groups(9, 5, 1, 2, 2, 1);
   check_groups(17, 5, 2, 3, 2, 2);

   check_groups(17, 9, 3, 3, 3, 3);
   check_groups(1920, 1080, 1, 240, 270, 1);
   check_groups(2560, 1440, 1, 320, 360, 1);

   /* Top-of-range non-wrapping arithmetic. */
   check_groups(0xffffffffu, 1, 1, 0x08000000u, 1, 1);
   check_groups(0xffffffffu, 0xffffffffu, 1, 0x20000000u, 0x40000000u, 1);
   check_groups(0xffffffe0u, 1, 1, 0x07ffffffu, 1, 1);
   check_groups(0xfffffff8u, 0xfffffffcu, 2, 0x1fffffffu, 0x3fffffffu, 2);

   /* Exact bijective coverage of shader launch IDs across 1D, 2D, and 3D. */
   check_bijection(0, 5, 1);
   check_bijection(5, 0, 1);
   check_bijection(1, 1, 1);
   check_bijection(31, 1, 1);
   check_bijection(32, 1, 1);
   check_bijection(65, 1, 3);

   /* Exhaustive partial edge tile matrix at heights 2, 4, and 5 across
    * partial and tile-aligned widths and 2D/3D depths. */
   const uint32_t edge_widths[] = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33};
   const uint32_t edge_heights[] = {2, 4, 5};
   for (unsigned hi = 0; hi < sizeof(edge_heights) / sizeof(edge_heights[0]); ++hi) {
      for (unsigned wi = 0; wi < sizeof(edge_widths) / sizeof(edge_widths[0]); ++wi) {
         check_bijection(edge_widths[wi], edge_heights[hi], 1);
         check_bijection(edge_widths[wi], edge_heights[hi], 2);
      }
   }

   check_bijection(5, 3, 1);
   check_bijection(17, 9, 3);
   check_bijection(64, 36, 2);

   printf("HK_RAY_TRACING_CORE_PASS\n");
   return 0;
}
