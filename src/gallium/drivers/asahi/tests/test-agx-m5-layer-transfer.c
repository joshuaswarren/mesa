/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Under the no-op DRM shim, check how the M5 profile draws into a layered
 * framebuffer. G17G draws a layered framebuffer a layer at a time, so a draw
 * whose vertex shader writes gl_Layer must take the CPU vertex stages
 * (agx_swtnl_applies), which draw each primitive into the layer it selects.
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "asahi/compiler/agx_compile.h"
#include "compiler/nir/nir_builder.h"
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

static void *
create_vs(struct pipe_context *pipe, bool layer)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "layer split VS");
   nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), nir_imm_int(&b, 0),
                    .base = 0, .write_mask = 0xf,
                    .io_semantics.location = VARYING_SLOT_POS);
   if (layer) {
      nir_store_output(&b, nir_load_instance_id(&b), nir_imm_int(&b, 0),
                       .base = 1, .write_mask = 0x1,
                       .src_type = nir_type_int32,
                       .io_semantics.location = VARYING_SLOT_LAYER);
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   struct pipe_shader_state state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = b.shader,
   };
   void *vs = pipe->create_vs_state(pipe, &state);
   CHECK(vs);
   return vs;
}

static struct pipe_resource *
create_colour(struct pipe_screen *screen, unsigned layers)
{
   struct pipe_resource templ = {
      .target = layers > 1 ? PIPE_TEXTURE_2D_ARRAY : PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 64,
      .height0 = 64,
      .depth0 = 1,
      .array_size = layers,
      .last_level = 2,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
   };
   struct pipe_resource *res = screen->resource_create(screen, &templ);
   CHECK(res);
   return res;
}

/* Bind a framebuffer over layers 0..layers-1 of a 2D array texture and the
 * given vertex shader, and report whether the draw takes the CPU stages. */
static bool
takes_cpu_stages(struct pipe_context *pipe, struct pipe_resource *colour,
                 unsigned layers, void *vs)
{
   struct pipe_framebuffer_state fb = {
      .width = 64,
      .height = 64,
      .layers = layers,
      .nr_cbufs = 1,
   };
   fb.cbufs[0] = (struct pipe_surface){
      .texture = colour,
      .format = colour->format,
      .level = 1,
      .last_layer = layers - 1,
   };
   pipe->set_framebuffer_state(pipe, &fb);
   pipe->bind_vs_state(pipe, vs);

   /* A draw waits for an asynchronously compiled shader before it asks
    * (agx_draw_vbo); this test asks directly, so wait here too. */
   struct agx_context *ctx = agx_context(pipe);
   agx_shader_wait_ready(ctx->stage[MESA_SHADER_VERTEX].shader);
   return agx_swtnl_applies(ctx);
}

int
main(void)
{
   const char *chip_name = getenv("ASAHI_SHIM_CHIP");
   CHECK(chip_name);

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

   const enum agx_chip chip = agx_device(screen)->chip;
   const bool m5 = agx_chip_g17_m5(chip);
   CHECK(m5 == !strcmp(chip_name, "g17s"));

   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   CHECK(pipe);

   const struct pipe_vertex_element element = {
      .src_format = PIPE_FORMAT_R32G32B32A32_FLOAT,
      .src_stride = 16,
   };
   void *velems = pipe->create_vertex_elements_state(pipe, 1, &element);
   CHECK(velems);
   pipe->bind_vertex_elements_state(pipe, velems);

   void *vs_layer = create_vs(pipe, true);
   void *vs_plain = create_vs(pipe, false);

   struct pipe_resource *array = create_colour(screen, 4);

   /* Layered framebuffer: only a vertex shader that writes gl_Layer, and
    * only on the M5 profile, which draws such a framebuffer per layer. */
   CHECK(takes_cpu_stages(pipe, array, 4, vs_layer) == m5);
   CHECK(!takes_cpu_stages(pipe, array, 4, vs_plain));

   /* One layer: the vertex shader's layer has nowhere else to go. */
   CHECK(!takes_cpu_stages(pipe, array, 1, vs_layer));

   struct pipe_framebuffer_state none = {0};
   pipe->set_framebuffer_state(pipe, &none);
   pipe->bind_vs_state(pipe, NULL);
   pipe->bind_vertex_elements_state(pipe, NULL);
   pipe->delete_vs_state(pipe, vs_layer);
   pipe->delete_vs_state(pipe, vs_plain);
   pipe->delete_vertex_elements_state(pipe, velems);
   pipe_resource_reference(&array, NULL);
   pipe->destroy(pipe);
   screen->destroy(screen);

   printf("%s: layer split ok\n", chip_name);
   return 0;
}
