/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Verify HAL200/AGX3B image_copy_buffer format mapping, alignment/tiling
 * geometry planners, and live drm-shim compute dispatch. */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "asahi/layout/layout.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/driconf.h"
#include "util/u_inlines.h"
#include "util/xmlconfig.h"
#include "agx_public.h"
#include "agx_state.h"

#define CHECK(expr)                                                            \
   do {                                                                        \
      if (!(expr)) {                                                           \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr);            \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static void
check_block_buf2img_formats(void)
{
   /* Supported 8-bit 1/4-channel (UNORM8 PBE) and 32-bit 1-channel (R32_UINT
    * PBE) formats map to their canonical PBE view format. */
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8_UNORM) ==
         PIPE_FORMAT_R8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8_UINT) ==
         PIPE_FORMAT_R8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8_SNORM) ==
         PIPE_FORMAT_R8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8G8B8A8_UNORM) ==
         PIPE_FORMAT_R8G8B8A8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8G8B8A8_SRGB) ==
         PIPE_FORMAT_R8G8B8A8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_B8G8R8A8_UNORM) ==
         PIPE_FORMAT_R8G8B8A8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8G8B8A8_SNORM) ==
         PIPE_FORMAT_R8G8B8A8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8G8B8A8_SINT) ==
         PIPE_FORMAT_R8G8B8A8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8G8B8A8_UINT) ==
         PIPE_FORMAT_R8G8B8A8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R32_FLOAT) ==
         PIPE_FORMAT_R32_UINT);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R32_SINT) ==
         PIPE_FORMAT_R32_UINT);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R32_UINT) ==
         PIPE_FORMAT_R32_UINT);

   /* Formats without an AGX3B explicit tile-store encoding, depth/stencil,
    * block-compressed, 16-byte, and packed formats decline cleanly. */
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8G8_UNORM) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R16_FLOAT) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R16G16_FLOAT) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R16G16B16A16_FLOAT) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R32G32_FLOAT) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_Z32_FLOAT) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_S8_UINT) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_DXT1_RGBA) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R32G32B32A32_FLOAT) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R10G10B10A2_UNORM) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_B5G6R5_UNORM) ==
         PIPE_FORMAT_NONE);
}

static void
check_alignment_and_tiled_plan(void)
{
   struct pipe_box full_2048 = {.x = 0, .y = 0, .z = 0, .width = 2048, .height = 2048, .depth = 1};
   CHECK(agx_g17p_block_buf2img_aligned(&full_2048, 2048, 2048));

   struct pipe_box full_npot = {.x = 0, .y = 0, .z = 0, .width = 300, .height = 200, .depth = 1};
   CHECK(agx_g17p_block_buf2img_aligned(&full_npot, 300, 200));

   struct pipe_box sub_aligned = {.x = 32, .y = 64, .z = 0, .width = 96, .height = 64, .depth = 1};
   CHECK(agx_g17p_block_buf2img_aligned(&sub_aligned, 256, 256));

   struct pipe_box sub_edge = {.x = 256, .y = 128, .z = 0, .width = 44, .height = 72, .depth = 1};
   CHECK(agx_g17p_block_buf2img_aligned(&sub_edge, 300, 200));

   struct pipe_box sub_unaligned_x = {.x = 5, .y = 0, .z = 0, .width = 64, .height = 64, .depth = 1};
   CHECK(!agx_g17p_block_buf2img_aligned(&sub_unaligned_x, 256, 256));

   struct pipe_box sub_unaligned_w = {.x = 0, .y = 0, .z = 0, .width = 50, .height = 64, .depth = 1};
   CHECK(!agx_g17p_block_buf2img_aligned(&sub_unaligned_w, 256, 256));

   /* Uncompressed GPU-tiled 2048x2048 RGBA8 layout. */
   struct ail_layout layout = {
      .width_px = 2048,
      .height_px = 2048,
      .depth_px = 1,
      .sample_count_sa = 1,
      .levels = 1,
      .tiling = AIL_TILING_GPU,
      .compressed = false,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   ail_make_miptree(&layout);

   uint32_t groups[3] = {0};
   uint32_t per_thread = 0;
   CHECK(agx_g17p_tiled_buf2img_plan(&layout, 0, &full_2048, 0, 2048 * 4,
                                     2048 * 2048 * 4, groups, &per_thread));
   CHECK(per_thread == 4);
   CHECK(groups[0] > 0 && groups[1] > 0 && groups[2] > 0);

   /* Unaligned buffer offset for 4B format is rejected cleanly. */
   CHECK(!agx_g17p_tiled_buf2img_plan(&layout, 0, &full_2048, 2, 2048 * 4,
                                      2048 * 2048 * 4, groups, &per_thread));

   /* Compressed level declines the uncompressed tiled planner. */
   struct ail_layout comp = layout;
   comp.compressed = true;
   ail_make_miptree(&comp);
   CHECK(!agx_g17p_tiled_buf2img_plan(&comp, 0, &full_2048, 0, 2048 * 4,
                                      2048 * 2048 * 4, groups, &per_thread));
}

static void
check_shim_paths(void)
{
   int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   CHECK(fd >= 0);

   static const driOptionDescription options[] = {
#include "driinfo_asahi.h"
   };
   driOptionCache info, cache;
   driParseOptionInfo(&info, options, ARRAY_SIZE(options));
   struct pipe_screen_config config = {
      .options = &cache,
      .options_info = &info,
   };
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   CHECK(screen != NULL);
   struct agx_device *dev = agx_device(screen);
   struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
   CHECK(pctx != NULL);
   struct agx_context *ctx = agx_context(pctx);

   /* 1. Verify image_copy_buffer is wired on HAL200 and dispatches compute. */
   if (agx_chip_native_agx3(dev->chip)) {
      CHECK(pctx->image_copy_buffer != NULL);

      struct pipe_resource tex_templ = {
         .target = PIPE_TEXTURE_2D,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .width0 = 256,
         .height0 = 256,
         .depth0 = 1,
         .array_size = 1,
         .last_level = 0,
         .bind = PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET,
      };
      struct pipe_resource *tex = screen->resource_create(screen, &tex_templ);
      CHECK(tex != NULL);

      struct pipe_resource buf_templ = {
         .target = PIPE_BUFFER,
         .format = PIPE_FORMAT_R8_UNORM,
         .width0 = 256 * 256 * 4,
         .height0 = 1,
         .depth0 = 1,
         .array_size = 1,
         .bind = PIPE_BIND_SAMPLER_VIEW,
      };
      struct pipe_resource *pbo = screen->resource_create(screen, &buf_templ);
      CHECK(pbo != NULL);

      /* Full aligned upload */
      struct pipe_box full = {.width = 256, .height = 256, .depth = 1};
      pctx->image_copy_buffer(pctx, tex, pbo, 0, 256 * 4, 0, 0, &full);
      CHECK(ctx->batch != NULL && agx_batch_is_compute(ctx->batch));
      uint8_t *cdm_after_full = ctx->batch->cdm.current;
      CHECK(cdm_after_full > (uint8_t *)agx_bo_map(ctx->batch->cdm.bo));
      CHECK(agx_resource_valid(agx_resource(tex), 0));

      /* Partial unaligned sub-rect upload into valid compressed texture */
      struct pipe_box partial = {.x = 7, .y = 9, .width = 45, .height = 37, .depth = 1};
      pctx->image_copy_buffer(pctx, tex, pbo, 0, 45 * 4, 0, 0, &partial);
      CHECK(ctx->batch != NULL && agx_batch_is_compute(ctx->batch));
      CHECK(ctx->batch->cdm.current > cdm_after_full);

      /* A 3D texture also bound as a shader image stays uncompressed and
       * GPU-tiled, so it takes the Morton compute path. */
      struct pipe_resource tex3d_templ = {
         .target = PIPE_TEXTURE_3D,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .width0 = 64,
         .height0 = 64,
         .depth0 = 2,
         .array_size = 1,
         .last_level = 0,
         .bind = PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_SHADER_IMAGE,
      };
      struct pipe_resource *tex3d = screen->resource_create(screen, &tex3d_templ);
      CHECK(tex3d != NULL);
      CHECK(agx_resource(tex3d)->layout.tiling == AIL_TILING_GPU);
      CHECK(!agx_resource(tex3d)->layout.compressed);
      uint8_t *cdm_before_3d = ctx->batch->cdm.current;
      struct pipe_box box3d = {.width = 64, .height = 64, .depth = 2};
      pctx->image_copy_buffer(pctx, tex3d, pbo, 0, 64 * 4, 64 * 64 * 4, 0, &box3d);
      CHECK(ctx->batch != NULL && agx_batch_is_compute(ctx->batch));
      CHECK(ctx->batch->cdm.current > cdm_before_3d);
      CHECK(agx_resource_valid(agx_resource(tex3d), 0));

      pctx->flush(pctx, NULL, 0);
      pipe_resource_reference(&tex3d, NULL);
      pipe_resource_reference(&pbo, NULL);
      pipe_resource_reference(&tex, NULL);
   }

   pctx->destroy(pctx);
   screen->destroy(screen);
   close(fd);
}

int
main(int argc, char **argv)
{
   check_block_buf2img_formats();
   check_alignment_and_tiled_plan();

   if (argc > 1 && !strcmp(argv[1], "--unit-only"))
      return 0;

   check_shim_paths();
   return 0;
}
