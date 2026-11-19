/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Under the no-op DRM shim, check which render batches a depth-only
 * framebuffer may be merged into (agx_try_zs_merge). G17G draws a framebuffer
 * with several colour attachments or layers in passes that rebind it, so a
 * merge into such a batch must be refused there and kept everywhere else.
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/driconf.h"
#include "util/u_inlines.h"
#include "util/xmlconfig.h"
#include "agx_public.h"
#include "agx_state.h"

#define CHECK(expr) do { \
   if (!(expr)) { \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
      exit(1); \
   } \
} while (0)

static struct pipe_resource *
create(struct pipe_screen *screen, enum pipe_format format, unsigned layers,
       unsigned bind)
{
   struct pipe_resource templ = {
      .target = layers > 1 ? PIPE_TEXTURE_2D_ARRAY : PIPE_TEXTURE_2D,
      .format = format, .width0 = 64, .height0 = 64, .depth0 = 1,
      .array_size = layers, .bind = bind,
   };
   struct pipe_resource *res = screen->resource_create(screen, &templ);
   CHECK(res);
   return res;
}

static struct pipe_surface
surface(struct pipe_resource *res, unsigned layers)
{
   return (struct pipe_surface){
      .texture = res, .format = res->format, .last_layer = layers - 1,
   };
}

/* Start a batch on a framebuffer with nr_cbufs colour attachments, then bind
 * its depth attachment alone and report whether that merged. */
static bool
merges(struct pipe_screen *screen, unsigned nr_cbufs, unsigned layers)
{
   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   CHECK(pipe);
   struct agx_context *ctx = agx_context(pipe);

   struct pipe_resource *zs =
      create(screen, PIPE_FORMAT_Z32_FLOAT, layers, PIPE_BIND_DEPTH_STENCIL);
   struct pipe_resource *colour[2] = {0};
   struct pipe_framebuffer_state fb = {
      .width = 64, .height = 64, .layers = layers, .nr_cbufs = nr_cbufs,
      .zsbuf = surface(zs, layers),
   };
   for (unsigned i = 0; i < nr_cbufs; ++i) {
      colour[i] = create(screen, PIPE_FORMAT_R8G8B8A8_UNORM, layers,
                         PIPE_BIND_RENDER_TARGET);
      fb.cbufs[i] = surface(colour[i], layers);
   }

   pipe->set_framebuffer_state(pipe, &fb);
   CHECK(agx_get_batch(ctx));

   struct pipe_framebuffer_state depth_only = {
      .width = 64, .height = 64, .layers = layers,
      .zsbuf = surface(zs, layers),
   };
   pipe->set_framebuffer_state(pipe, &depth_only);
   bool merged = ctx->zs_merge;

   /* A merged state's colour targets are the batch's; unmerged, none. */
   CHECK(merged == (ctx->framebuffer.nr_cbufs == nr_cbufs));

   pipe->destroy(pipe);
   for (unsigned i = 0; i < nr_cbufs; ++i)
      pipe_resource_reference(&colour[i], NULL);
   pipe_resource_reference(&zs, NULL);
   return merged;
}

int
main(void)
{
   const char *chip = getenv("ASAHI_SHIM_CHIP");
   CHECK(chip);
   const bool split = !strcmp(chip, "g17s");

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
   CHECK(agx_chip_g17_m5(agx_device(screen)->chip) == split);

   CHECK(merges(screen, 1, 1));
   CHECK(merges(screen, 2, 1) == !split);
   CHECK(merges(screen, 1, 2) == !split);

   screen->destroy(screen);
   printf("%s: zs merge ok\n", chip);
   return 0;
}
