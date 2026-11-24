/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hk_cmd_buffer.h"

static unsigned failures = 0;

#define CHECK(cond, fmt, ...)                                                  \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL %s:%d: " fmt "\n", __FILE__, __LINE__,          \
                 ##__VA_ARGS__);                                               \
         failures++;                                                           \
      }                                                                        \
   } while (0)

static struct hk_cs
make_attachmentless_cs(uint32_t w, uint32_t h, uint32_t layers)
{
   struct hk_cs cs = {0};
   cs.cr.width = w;
   cs.cr.height = h;
   cs.cr.layers = layers;
   cs.area_x1 = w;
   cs.area_y1 = h;
   for (unsigned i = 0; i < ARRAY_SIZE(cs.tib.logical_format); ++i)
      cs.tib.logical_format[i] = PIPE_FORMAT_NONE;
   return cs;
}

static void
test_empty_attachmentless(void)
{
   struct hk_cs cs = make_attachmentless_cs(16384, 16384, 2048);
   CHECK(hk_attachmentless_render(&cs), "empty pass is attachmentless");

   uint32_t w = 0, h = 0, layers = 0;
   hk_attachmentless_vdm_bounds(&cs, &w, &h, &layers);
   CHECK(w == 1 && h == 1 && layers == 1,
         "empty attachmentless collapses to 1x1x1, got %ux%ux%u", w, h, layers);
}

static void
test_vkd3d_bounded_layer_zero(void)
{
   struct hk_cs cs = make_attachmentless_cs(16384, 16384, 2048);
   cs.stats.cmds = 3;
   cs.layer_zero_draws = 3;
   cs.attachmentless_bounded_draws = 3;
   cs.attachmentless_max_x = 1280;
   cs.attachmentless_max_y = 720;

   uint32_t w = 0, h = 0, layers = 0;
   hk_attachmentless_vdm_bounds(&cs, &w, &h, &layers);
   CHECK(w == 1280 && h == 720 && layers == 1,
         "vkd3d attachmentless bounds to 1280x720x1, got %ux%ux%u", w, h, layers);
}

static void
test_rasterizer_discard_clamps_to_1x1(void)
{
   struct hk_cs cs = make_attachmentless_cs(16384, 16384, 2048);
   cs.stats.cmds = 2;
   cs.layer_zero_draws = 2;
   cs.attachmentless_bounded_draws = 2;
   cs.attachmentless_max_x = 0;
   cs.attachmentless_max_y = 0;

   uint32_t w = 0, h = 0, layers = 0;
   hk_attachmentless_vdm_bounds(&cs, &w, &h, &layers);
   CHECK(w == 1 && h == 1 && layers == 1,
         "empty scissor / rasterizer discard clamps to 1x1x1, got %ux%ux%u",
         w, h, layers);
}

static void
test_multiview_keeps_layers_but_bounds_xy(void)
{
   struct hk_cs cs = make_attachmentless_cs(16384, 16384, 4);
   cs.stats.cmds = 2;
   cs.layer_zero_unknown = true;
   cs.attachmentless_bounded_draws = 2;
   cs.attachmentless_max_x = 640;
   cs.attachmentless_max_y = 480;

   uint32_t w = 0, h = 0, layers = 0;
   hk_attachmentless_vdm_bounds(&cs, &w, &h, &layers);
   CHECK(w == 640 && h == 480 && layers == 4,
         "unknown layers keeps 4 layers while bounding XY to 640x480, got %ux%ux%u",
         w, h, layers);
}

static void
test_untracked_emitter_disqualifies_both(void)
{
   struct hk_cs cs = make_attachmentless_cs(16384, 16384, 8);
   cs.stats.cmds = 4;
   cs.layer_zero_draws = 3;
   cs.attachmentless_bounded_draws = 3;
   cs.attachmentless_max_x = 640;
   cs.attachmentless_max_y = 480;

   uint32_t w = 0, h = 0, layers = 0;
   hk_attachmentless_vdm_bounds(&cs, &w, &h, &layers);
   CHECK(w == 16384 && h == 16384 && layers == 8,
         "untracked emitter falls back to full 16384x16384x8, got %ux%ux%u",
         w, h, layers);
}

static void
test_attachments_and_empty_tiles_excluded(void)
{
   struct hk_cs cs = make_attachmentless_cs(1920, 1080, 6);
   cs.stats.cmds = 1;
   cs.layer_zero_draws = 1;
   cs.attachmentless_bounded_draws = 1;
   cs.attachmentless_max_x = 100;
   cs.attachmentless_max_y = 100;

   /* Colour attachment */
   cs.tib.logical_format[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
   CHECK(!hk_attachmentless_render(&cs), "colour attachment not attachmentless");
   uint32_t w = 0, h = 0, layers = 0;
   hk_attachmentless_vdm_bounds(&cs, &w, &h, &layers);
   CHECK(w == 1920 && h == 1080 && layers == 6,
         "colour attachment keeps 1920x1080x6, got %ux%ux%u", w, h, layers);
   cs.tib.logical_format[0] = PIPE_FORMAT_NONE;

   /* Depth attachment */
   cs.cr.depth.buffer = 0x1000;
   CHECK(!hk_attachmentless_render(&cs), "depth attachment not attachmentless");
   cs.cr.depth.buffer = 0;

   /* Stencil attachment */
   cs.cr.stencil.buffer = 0x2000;
   CHECK(!hk_attachmentless_render(&cs), "stencil attachment not attachmentless");
   cs.cr.stencil.buffer = 0;

   /* Process empty tiles */
   cs.cr.process_empty_tiles = true;
   CHECK(!hk_attachmentless_render(&cs), "process_empty_tiles not attachmentless");
   cs.cr.process_empty_tiles = false;

   /* Render area still clamps bounded draws */
   cs.area_x1 = 64;
   cs.area_y1 = 32;
   hk_attachmentless_vdm_bounds(&cs, &w, &h, &layers);
   CHECK(w == 64 && h == 32 && layers == 1,
         "area_x1/y1 still clamps to 64x32x1, got %ux%ux%u", w, h, layers);
}

int
main(void)
{
   test_empty_attachmentless();
   test_vkd3d_bounded_layer_zero();
   test_rasterizer_discard_clamps_to_1x1();
   test_multiview_keeps_layers_but_bounds_xy();
   test_untracked_emitter_disqualifies_both();
   test_attachments_and_empty_tiles_excluded();

   if (failures) {
      fprintf(stderr, "test-hk-attachmentless: %u failure(s)\n", failures);
      return EXIT_FAILURE;
   }
   printf("test-hk-attachmentless: all checks passed\n");
   return EXIT_SUCCESS;
}
