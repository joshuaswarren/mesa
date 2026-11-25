/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Under the no-op DRM shim with the M5 kernel's submission checks
 * (ASAHI_SHIM_CHIP=g17s-kernel, ASAHI_SHIM_M5_EXPECT_VALID=1), check which
 * private 2D colour and depth images Gallium compresses on M5
 * (AGX_CAP_GEN10_2D_COMPRESSION), and that the renders it makes of them at
 * 1x, 2x and 4x are ones the kernel admits.
 *
 * 2x/4x Z32_FLOAT depth compresses only where the kernel reports
 * DRM_ASAHI_FEATURE_M5_DEPTH_MSAA_COMPRESSION. With
 * ASAHI_SHIM_NO_M5_MSAA_ZLS_COMPRESSION=1 the shim reports a kernel without
 * it, whose render validation refuses a single-layer multisample render with
 * compressed ZLS with EOPNOTSUPP, so a compressed multisample depth buffer
 * fails the run there. This test does not execute GPU work or compare
 * rendered pixels.
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "asahi/lib/agx_chip_caps.h"
#include "drm-uapi/asahi_drm.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/driconf.h"
#include "util/u_inlines.h"
#include "util/u_simple_shaders.h"
#include "util/xmlconfig.h"
#include "agx_public.h"
#include "agx_state.h"

#define CHECK(expr) do { \
   if (!(expr)) { \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
      exit(1); \
   } \
} while (0)

#define SIZE 64

static struct pipe_resource *
create(struct pipe_screen *screen, enum pipe_format format, unsigned samples,
       unsigned bind)
{
   struct pipe_resource templ = {
      .target = PIPE_TEXTURE_2D, .format = format, .width0 = SIZE,
      .height0 = SIZE, .depth0 = 1, .array_size = 1, .nr_samples = samples,
      .nr_storage_samples = samples, .bind = bind | PIPE_BIND_SAMPLER_VIEW,
   };
   struct pipe_resource *res = screen->resource_create(screen, &templ);
   CHECK(res);
   return res;
}

static bool
compressed(struct pipe_resource *res)
{
   return agx_resource(res)->layout.compressed;
}

static void
flush_and_wait(struct pipe_screen *screen, struct pipe_context *pipe)
{
   struct pipe_fence_handle *fence = NULL;
   pipe->flush(pipe, &fence, 0);
   CHECK(fence);
   CHECK(screen->fence_finish(screen, pipe, fence, OS_TIMEOUT_INFINITE));
   screen->fence_reference(screen, &fence, NULL);
}

/* Clear a colour and depth attachment of this sample count in one render,
 * then load both and draw a depth-tested triangle into them in a second. */
static void
render(struct pipe_screen *screen, struct pipe_context *pipe,
       struct pipe_resource *colour, struct pipe_resource *depth)
{
   struct pipe_framebuffer_state fb = {
      .width = SIZE, .height = SIZE, .layers = 1, .nr_cbufs = 1,
      .samples = colour->nr_samples,
      .cbufs[0] = {.texture = colour, .format = colour->format},
      .zsbuf = {.texture = depth, .format = depth->format},
   };
   pipe->set_framebuffer_state(pipe, &fb);

   const union pipe_color_union clear = {.f = {0.25f, 0.5f, 0.75f, 1.0f}};
   pipe->clear(pipe, PIPE_CLEAR_COLOR0 | PIPE_CLEAR_DEPTH, 0x1, 0, NULL,
               &clear, 0.5, 0);
   flush_and_wait(screen, pipe);

   struct pipe_draw_info info = {
      .mode = MESA_PRIM_TRIANGLES, .instance_count = 1,
   };
   struct pipe_draw_start_count_bias range = {.count = 3};
   pipe->draw_vbo(pipe, &info, 0, NULL, &range, 1);
   flush_and_wait(screen, pipe);

   struct pipe_framebuffer_state none = {0};
   pipe->set_framebuffer_state(pipe, &none);
}

int
main(void)
{
   const char *chip = getenv("ASAHI_SHIM_CHIP");
   CHECK(chip && !strncmp(chip, "g17s", 4));
   const char *no_msaa_zls = getenv("ASAHI_SHIM_NO_M5_MSAA_ZLS_COMPRESSION");
   const bool msaa_zls_expected = !(no_msaa_zls && !strcmp(no_msaa_zls, "1"));

   int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   CHECK(fd >= 0);
   static const driOptionDescription descriptions[] = {
#include "driinfo_asahi.h"
   };
   driOptionCache info, options;
   driParseOptionInfo(&info, descriptions, ARRAY_SIZE(descriptions));
   struct pipe_screen_config config = {.options = &options,
                                       .options_info = &info};
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   CHECK(screen);
   const struct agx_device *dev = agx_device(screen);
   CHECK(agx_chip_has_cap(dev->chip, AGX_CAP_GEN10_2D_COMPRESSION));
   const bool msaa_zls =
      dev->params.features & DRM_ASAHI_FEATURE_M5_DEPTH_MSAA_COMPRESSION;
   CHECK(msaa_zls == msaa_zls_expected);

   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   CHECK(pipe);

   /* A passthrough vertex shader, a fragment shader writing the
    * interpolated colour and a full-screen triangle, drawn with the depth
    * test on. */
   const float vertices[3][8] = {
      {-1, -1, 0.25f, 1, 1, 0, 0, 1},
      { 3, -1, 0.25f, 1, 0, 1, 0, 1},
      {-1,  3, 0.25f, 1, 0, 0, 1, 1},
   };
   struct pipe_resource *buffer = pipe_buffer_create(
      screen, PIPE_BIND_VERTEX_BUFFER, PIPE_USAGE_DEFAULT, sizeof(vertices));
   CHECK(buffer);
   pipe_buffer_write(pipe, buffer, 0, sizeof(vertices), vertices);
   struct pipe_vertex_buffer vertex_buffer = {.buffer.resource = buffer};
   pipe->set_vertex_buffers(pipe, 1, &vertex_buffer);
   struct pipe_vertex_element elements[2] = {
      {.src_format = PIPE_FORMAT_R32G32B32A32_FLOAT, .src_stride = 32},
      {.src_offset = 16, .src_format = PIPE_FORMAT_R32G32B32A32_FLOAT,
       .src_stride = 32},
   };
   void *vertex_elements = pipe->create_vertex_elements_state(pipe, 2, elements);
   CHECK(vertex_elements);
   pipe->bind_vertex_elements_state(pipe, vertex_elements);
   const enum tgsi_semantic semantics[] = {TGSI_SEMANTIC_POSITION,
                                           TGSI_SEMANTIC_GENERIC};
   const unsigned indices[] = {0, 0};
   void *vs = util_make_vertex_passthrough_shader(pipe, 2, semantics, indices,
                                                  false);
   void *fs = util_make_fragment_cloneinput_shader(
      pipe, 1, TGSI_SEMANTIC_GENERIC, TGSI_INTERPOLATE_PERSPECTIVE);
   CHECK(vs && fs);
   pipe->bind_vs_state(pipe, vs);
   pipe->bind_fs_state(pipe, fs);

   struct pipe_blend_state blend = {0};
   blend.rt[0].colormask = PIPE_MASK_RGBA;
   void *blend_state = pipe->create_blend_state(pipe, &blend);
   struct pipe_depth_stencil_alpha_state zsa = {
      .depth_enabled = true, .depth_writemask = true,
      .depth_func = PIPE_FUNC_LESS,
   };
   void *zsa_state = pipe->create_depth_stencil_alpha_state(pipe, &zsa);
   struct pipe_rasterizer_state raster = {
      .front_ccw = true, .depth_clip_near = true, .depth_clip_far = true,
      .clip_halfz = true, .multisample = true,
   };
   void *raster_state = pipe->create_rasterizer_state(pipe, &raster);
   CHECK(blend_state && zsa_state && raster_state);
   pipe->bind_blend_state(pipe, blend_state);
   pipe->bind_depth_stencil_alpha_state(pipe, zsa_state);
   pipe->bind_rasterizer_state(pipe, raster_state);
   pipe->set_sample_mask(pipe, ~0);
   struct pipe_viewport_state viewport = {
      .scale = {SIZE / 2, SIZE / 2, 1}, .translate = {SIZE / 2, SIZE / 2, 0},
   };
   pipe->set_viewport_states(pipe, 0, 1, &viewport);

   static const unsigned sample_counts[] = {1, 2, 4};
   for (unsigned i = 0; i < ARRAY_SIZE(sample_counts); ++i) {
      const unsigned samples = sample_counts[i];
      struct pipe_resource *colour =
         create(screen, PIPE_FORMAT_R8G8B8A8_UNORM, samples,
                PIPE_BIND_RENDER_TARGET);
      struct pipe_resource *z32 =
         create(screen, PIPE_FORMAT_Z32_FLOAT, samples,
                PIPE_BIND_DEPTH_STENCIL);
      struct pipe_resource *z16 =
         create(screen, PIPE_FORMAT_Z16_UNORM, samples,
                PIPE_BIND_DEPTH_STENCIL);

      /* Colour compresses at every sample count. Z32_FLOAT compresses at 1x,
       * and at 2x/4x only where the kernel admits compressed multisample
       * ZLS. Z16_UNORM compresses at 1x only. */
      CHECK(compressed(colour));
      CHECK(compressed(z32) == (samples == 1 || msaa_zls));
      CHECK(compressed(z16) == (samples == 1));

      render(screen, pipe, colour, z32);
      render(screen, pipe, colour, z16);
      printf("%s: %ux colour %s, Z32_FLOAT %s, Z16_UNORM %s\n", chip, samples,
             compressed(colour) ? "compressed" : "plain",
             compressed(z32) ? "compressed" : "plain",
             compressed(z16) ? "compressed" : "plain");

      pipe_resource_reference(&colour, NULL);
      pipe_resource_reference(&z32, NULL);
      pipe_resource_reference(&z16, NULL);
   }

   /* Beyond the 2D single-level shape, and below 16x16 pixels, nothing is
    * compressed. */
   struct pipe_resource templ = {
      .target = PIPE_TEXTURE_2D_ARRAY, .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = SIZE, .height0 = SIZE, .depth0 = 1, .array_size = 2,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
   };
   struct pipe_resource *res = screen->resource_create(screen, &templ);
   CHECK(res && !compressed(res));
   pipe_resource_reference(&res, NULL);
   templ.target = PIPE_TEXTURE_2D;
   templ.array_size = 1;
   templ.last_level = 2;
   res = screen->resource_create(screen, &templ);
   CHECK(res && !compressed(res));
   pipe_resource_reference(&res, NULL);
   templ.last_level = 0;
   templ.width0 = templ.height0 = 8;
   res = screen->resource_create(screen, &templ);
   CHECK(res && !compressed(res));
   pipe_resource_reference(&res, NULL);
   templ.format = PIPE_FORMAT_Z32_FLOAT;
   templ.bind = PIPE_BIND_DEPTH_STENCIL | PIPE_BIND_SAMPLER_VIEW;
   templ.target = PIPE_TEXTURE_2D_ARRAY;
   templ.array_size = 2;
   templ.width0 = templ.height0 = SIZE;
   res = screen->resource_create(screen, &templ);
   CHECK(res && !compressed(res));
   pipe_resource_reference(&res, NULL);

   pipe->bind_vs_state(pipe, NULL);
   pipe->bind_fs_state(pipe, NULL);
   pipe->delete_vs_state(pipe, vs);
   pipe->delete_fs_state(pipe, fs);
   pipe->bind_blend_state(pipe, NULL);
   pipe->delete_blend_state(pipe, blend_state);
   pipe->bind_depth_stencil_alpha_state(pipe, NULL);
   pipe->delete_depth_stencil_alpha_state(pipe, zsa_state);
   pipe->bind_rasterizer_state(pipe, NULL);
   pipe->delete_rasterizer_state(pipe, raster_state);
   pipe->bind_vertex_elements_state(pipe, NULL);
   pipe->delete_vertex_elements_state(pipe, vertex_elements);
   pipe->set_vertex_buffers(pipe, 0, NULL);
   pipe_resource_reference(&buffer, NULL);
   pipe->destroy(pipe);
   screen->destroy(screen);
   printf("%s: M5 compression policy ok (multisample ZLS %s)\n", chip,
          msaa_zls ? "admitted" : "refused");
   return 0;
}
