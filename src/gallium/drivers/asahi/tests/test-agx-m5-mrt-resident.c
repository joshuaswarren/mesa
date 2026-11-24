/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Under the no-op DRM shim, check that the M5 profile keeps every colour
 * attachment of a single-sample framebuffer of four RGBA8 attachments in the
 * tilebuffer: one clear and one draw go into one batch, the layout spills
 * nothing, the background and end-of-tile programs are the multi-target ones
 * and the M5 submission gate accepts the render. A framebuffer the
 * multi-target programs do not cover keeps the previous paths.
 *
 * Then, on eight framebuffer slots of mixed integer and normalized formats
 * with two empty slots, every fragment program must still link: a clear
 * through a quad that masks all attachments but one, and a draw with a
 * partial colour mask per attachment. On M5 each masked attachment keeps its
 * contents with a tile load/store pair, so neither epilog stores the
 * attachment at byte 0 of the tile. This test does not execute GPU work or
 * compare rendered pixels.
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "asahi/compiler/agx_compile.h"
#include "asahi/lib/agx_bg_eot.h"
#include "compiler/nir/nir_builder.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/driconf.h"
#include "util/format/u_format.h"
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

#define NR_CBUFS 4
#define COLOUR_BUFFERS (PIPE_CLEAR_COLOR0 | PIPE_CLEAR_COLOR1 | \
                        PIPE_CLEAR_COLOR2 | PIPE_CLEAR_COLOR3)

static struct pipe_resource *
create_target(struct pipe_screen *screen, enum pipe_format format,
              unsigned samples)
{
   const struct pipe_resource templ = {
      .target = PIPE_TEXTURE_2D, .format = format,
      .width0 = 64, .height0 = 64, .depth0 = 1, .array_size = 1,
      .nr_samples = samples, .nr_storage_samples = samples,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
   };
   struct pipe_resource *rsrc = screen->resource_create(screen, &templ);
   CHECK(rsrc);
   return rsrc;
}

static void
bind_targets(struct pipe_context *pipe, struct pipe_resource **targets,
             unsigned count)
{
   struct pipe_framebuffer_state fb = {
      .width = 64, .height = 64, .layers = 1, .nr_cbufs = count,
   };
   for (unsigned i = 0; i < count; ++i) {
      fb.cbufs[i] = (struct pipe_surface){
         .texture = targets[i], .format = targets[i]->format,
      };
   }
   pipe->set_framebuffer_state(pipe, &fb);
}

static void
draw(struct pipe_context *pipe)
{
   struct pipe_draw_info info = {
      .mode = MESA_PRIM_TRIANGLES, .instance_count = 1,
   };
   struct pipe_draw_start_count_bias range = {.count = 3};
   pipe->draw_vbo(pipe, &info, 0, NULL, &range, 1);
}

/* Eight slots of mixed formats, two of them empty, that all fit in the
 * tile: a draw buffer set of the indexed draw buffer conformance tests. */
static const enum pipe_format mixed_formats[8] = {
   PIPE_FORMAT_R8_SINT,          PIPE_FORMAT_R16G16_UINT,
   PIPE_FORMAT_NONE,             PIPE_FORMAT_R10G10B10A2_UINT,
   PIPE_FORMAT_R8_UNORM,         PIPE_FORMAT_R16_UINT,
   PIPE_FORMAT_NONE,             PIPE_FORMAT_R32G32B32A32_SINT,
};

/* st/mesa's clear through a quad: one colour from constant buffer 0,
 * written to every colour buffer. */
static void *
create_clear_fs(struct pipe_context *pipe)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "clear quad FS");
   b.shader->info.io_lowered = true;
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *colour = nir_load_ubo(&b, 4, 32, zero, zero, .align_mul = 16,
                                  .align_offset = 0, .range = 16);
   nir_store_output(&b, colour, zero, .write_mask = 0xf,
                    .src_type = nir_type_float32,
                    .io_semantics.location = FRAG_RESULT_COLOR);
   b.shader->info.num_ubos = 1;
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   struct pipe_shader_state state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = b.shader,
   };
   void *fs = pipe->create_fs_state(pipe, &state);
   CHECK(fs);
   return fs;
}

/* One output per slot, typed by the slot's format. */
static void *
create_typed_fs(struct pipe_context *pipe, const enum pipe_format *formats)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "typed MRT FS");
   b.shader->info.io_lowered = true;
   nir_def *coord = nir_load_frag_coord(&b);
   nir_def *x = nir_fmul_imm(&b, nir_channel(&b, coord, 0), 1.0 / 64);
   nir_def *y = nir_fmul_imm(&b, nir_channel(&b, coord, 1), 1.0 / 64);
   nir_def *v[4] = {x, y, nir_fsub_imm(&b, 1.0, x), nir_fsub_imm(&b, 1.0, y)};
   for (unsigned rt = 0; rt < 8; ++rt) {
      nir_def *value =
         nir_vec4(&b, v[rt % 4], v[(rt + 1) % 4], v[(rt + 2) % 4],
                  v[(rt + 3) % 4]);
      nir_alu_type type = nir_type_float32;
      if (util_format_is_pure_uint(formats[rt])) {
         value = nir_f2u32(&b, value);
         type = nir_type_uint32;
      } else if (util_format_is_pure_sint(formats[rt])) {
         value = nir_f2i32(&b, value);
         type = nir_type_int32;
      }
      nir_store_output(&b, value, nir_imm_int(&b, 0), .write_mask = 0xf,
                       .src_type = type,
                       .io_semantics.location = FRAG_RESULT_DATA0 + rt);
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   struct pipe_shader_state state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = b.shader,
   };
   void *fs = pipe->create_fs_state(pipe, &state);
   CHECK(fs);
   return fs;
}

/* Draw with these colour masks and check that the draw reaches the batch. */
static void
draw_masked(struct pipe_context *pipe, const unsigned *masks)
{
   struct agx_context *ctx = agx_context(pipe);
   struct pipe_blend_state blend = {.independent_blend_enable = true};
   for (unsigned rt = 0; rt < 8; ++rt)
      blend.rt[rt].colormask = masks[rt];
   void *blend_state = pipe->create_blend_state(pipe, &blend);
   CHECK(blend_state);
   pipe->bind_blend_state(pipe, blend_state);

   const unsigned refused = ctx->skipped_no_program;
   draw(pipe);
   CHECK(ctx->skipped_no_program == refused);
   CHECK(ctx->batch && ctx->batch->g17g_mrt_resident);
   CHECK(!ctx->batch->draws_refused && !ctx->batch->native_rejected);
   pipe->flush(pipe, NULL, 0);

   pipe->bind_blend_state(pipe, NULL);
   pipe->delete_blend_state(pipe, blend_state);
}

static void
check_mixed_masked(struct pipe_screen *screen, struct pipe_context *pipe)
{
   struct pipe_resource *targets[8] = {0};
   struct pipe_framebuffer_state fb = {
      .width = 64, .height = 64, .layers = 1, .nr_cbufs = 8,
   };
   for (unsigned rt = 0; rt < 8; ++rt) {
      if (mixed_formats[rt] == PIPE_FORMAT_NONE)
         continue;
      targets[rt] = create_target(screen, mixed_formats[rt], 1);
      fb.cbufs[rt] = (struct pipe_surface){
         .texture = targets[rt], .format = targets[rt]->format,
      };
   }
   pipe->set_framebuffer_state(pipe, &fb);
   CHECK(agx_g17g_mrt_resident(agx_context(pipe)));

   /* A clear through a quad of each attachment in turn, with all channels
    * and with one. */
   const float colour[4] = {0.25f, 0.5f, 0.75f, 1.0f};
   struct pipe_constant_buffer cb = {
      .buffer_size = sizeof(colour),
      .user_buffer = colour,
   };
   pipe->set_constant_buffer(pipe, MESA_SHADER_FRAGMENT, 0, &cb);
   void *clear_fs = create_clear_fs(pipe);
   pipe->bind_fs_state(pipe, clear_fs);
   for (unsigned rt = 0; rt < 8; ++rt) {
      if (mixed_formats[rt] == PIPE_FORMAT_NONE)
         continue;
      static const unsigned clear_masks[] = {PIPE_MASK_R, PIPE_MASK_RGBA};
      for (unsigned i = 0; i < ARRAY_SIZE(clear_masks); ++i) {
         unsigned masks[8] = {0};
         masks[rt] = clear_masks[i];
         draw_masked(pipe, masks);
      }
   }

   /* A draw of every slot under a partial colour mask each. */
   void *typed_fs = create_typed_fs(pipe, mixed_formats);
   pipe->bind_fs_state(pipe, typed_fs);
   const unsigned partial[8] = {
      PIPE_MASK_B, PIPE_MASK_B, PIPE_MASK_B, PIPE_MASK_B,
      PIPE_MASK_B, PIPE_MASK_B, PIPE_MASK_B, PIPE_MASK_B,
   };
   draw_masked(pipe, partial);
   const unsigned full[8] = {
      PIPE_MASK_RGBA, PIPE_MASK_RGBA, PIPE_MASK_RGBA, PIPE_MASK_RGBA,
      PIPE_MASK_RGBA, PIPE_MASK_RGBA, PIPE_MASK_RGBA, PIPE_MASK_RGBA,
   };
   draw_masked(pipe, full);

   pipe->bind_fs_state(pipe, NULL);
   pipe->delete_fs_state(pipe, clear_fs);
   pipe->delete_fs_state(pipe, typed_fs);
   struct pipe_framebuffer_state none = {0};
   pipe->set_framebuffer_state(pipe, &none);
   for (unsigned rt = 0; rt < 8; ++rt)
      pipe_resource_reference(&targets[rt], NULL);
}

int
main(void)
{
   const char *chip = getenv("ASAHI_SHIM_CHIP");
   CHECK(chip && !strncmp(chip, "g17s", 4));
   int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   CHECK(fd >= 0);
   static const driOptionDescription descriptions[] = {
#include "driinfo_asahi.h"
   };
   driOptionCache info, options;
   driParseOptionInfo(&info, descriptions, ARRAY_SIZE(descriptions));
   struct pipe_screen_config config = {.options = &options, .options_info = &info};
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   CHECK(screen);
   CHECK(agx_chip_g17_m5(agx_device(screen)->chip));
   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   CHECK(pipe);
   struct agx_context *ctx = agx_context(pipe);

   struct pipe_resource *targets[NR_CBUFS];
   for (unsigned i = 0; i < NR_CBUFS; ++i)
      targets[i] = create_target(screen, PIPE_FORMAT_R8G8B8A8_UNORM, 1);
   bind_targets(pipe, targets, NR_CBUFS);
   CHECK(agx_g17g_mrt_resident(ctx));

   const float vertices[3][8] = {
      {-1, -1, 0, 1, 1, 0, 0, 1},
      { 3, -1, 0, 1, 0, 1, 0, 1},
      {-1,  3, 0, 1, 0, 0, 1, 1},
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
   /* One colour output per attachment. */
   void *fs = util_make_fragment_cloneinput_shader(
      pipe, NR_CBUFS, TGSI_SEMANTIC_GENERIC, TGSI_INTERPOLATE_PERSPECTIVE);
   CHECK(vs && fs);
   pipe->bind_vs_state(pipe, vs);
   pipe->bind_fs_state(pipe, fs);

   struct pipe_blend_state blend = {.independent_blend_enable = true};
   for (unsigned i = 0; i < NR_CBUFS; ++i)
      blend.rt[i].colormask = PIPE_MASK_RGBA;
   void *blend_state = pipe->create_blend_state(pipe, &blend);
   struct pipe_depth_stencil_alpha_state zs = {0};
   void *zs_state = pipe->create_depth_stencil_alpha_state(pipe, &zs);
   struct pipe_rasterizer_state raster = {
      .front_ccw = true, .depth_clip_near = true, .depth_clip_far = true,
      .clip_halfz = true,
   };
   void *raster_state = pipe->create_rasterizer_state(pipe, &raster);
   CHECK(blend_state && zs_state && raster_state);
   pipe->bind_blend_state(pipe, blend_state);
   pipe->bind_depth_stencil_alpha_state(pipe, zs_state);
   pipe->bind_rasterizer_state(pipe, raster_state);
   struct pipe_viewport_state viewport = {
      .scale = {32, 32, 1}, .translate = {32, 32, 0},
   };
   pipe->set_viewport_states(pipe, 0, 1, &viewport);

   /* One background clear of every attachment, in one batch. */
   const union pipe_color_union colour = {.f = {0.25f, 0.5f, 0.75f, 1.0f}};
   pipe->clear(pipe, COLOUR_BUFFERS, 0xffff, 0xff, NULL, &colour, 0, 0);
   struct agx_batch *batch = ctx->batch;
   CHECK(batch && batch->key.nr_cbufs == NR_CBUFS);
   CHECK((batch->clear & COLOUR_BUFFERS) == COLOUR_BUFFERS);
   CHECK(batch->initialized && batch->g17g_mrt_resident);

   const struct agx_tilebuffer_layout *tib = &batch->tilebuffer_layout;
   CHECK(!agx_tilebuffer_spills(tib));
   CHECK(tib->nr_samples == 1 && tib->sample_size_B == 16 &&
         tib->tile_size == 32 * 32);
   for (unsigned i = 0; i < NR_CBUFS; ++i) {
      CHECK(tib->logical_format[i] == PIPE_FORMAT_R8G8B8A8_UNORM);
      CHECK(agx_tilebuffer_offset_B(tib, i) == 4 * i);
   }

   /* The draw runs once, into the same batch. */
   const unsigned refused = ctx->skipped_no_program;
   draw(pipe);
   CHECK(ctx->batch == batch && batch->draws == 1);
   CHECK(ctx->skipped_no_program == refused && !batch->draws_refused);
   CHECK(!batch->native_rejected);

   /* Background, end-of-tile and partial-render programs all build. */
   for (unsigned store = 0; store < 2; ++store) {
      for (unsigned partial = 0; partial < 2; ++partial) {
         struct asahi_bg_eot program = agx_build_bg_eot(batch, store, partial);
         printf("store=%u partial=%u usc=%#llx\n", store, partial,
                (unsigned long long)program.usc);
         CHECK(program.usc && !batch->native_rejected);
      }
   }

   /* The end of tile is the native multi-target store of four targets:
    * eight staged halves per target. */
   struct agx_bg_eot_key key = {.tib = *tib, .mixed_mrt = true};
   for (unsigned i = 0; i < NR_CBUFS; ++i)
      key.op[i] = AGX_EOT_STORE;
   struct agx_bg_eot_shader *eot = agx_get_bg_eot_shader(&ctx->bg_eot, &key);
   CHECK(eot && eot->info.agx3_staging_known);
   CHECK(eot->info.agx3_staging_halfs == 8 * NR_CBUFS);

   /* With ASAHI_SHIM_M5_EXPECT_VALID, a render the M5 gate refuses fails
    * the run at exit. */
   pipe->flush(pipe, NULL, 0);

   /* Multisampled attachments keep the previous paths. */
   struct pipe_resource *msaa[2] = {
      create_target(screen, PIPE_FORMAT_R8G8B8A8_UNORM, 4),
      create_target(screen, PIPE_FORMAT_R8G8B8A8_UNORM, 4),
   };
   bind_targets(pipe, msaa, 2);
   CHECK(!agx_g17g_mrt_resident(ctx));

   /* Back to the single-sample attachments: resident again. */
   bind_targets(pipe, targets, NR_CBUFS);
   CHECK(agx_g17g_mrt_resident(ctx));

   check_mixed_masked(screen, pipe);

   pipe->bind_vs_state(pipe, NULL);
   pipe->bind_fs_state(pipe, NULL);
   pipe->delete_vs_state(pipe, vs);
   pipe->delete_fs_state(pipe, fs);
   pipe->bind_vertex_elements_state(pipe, NULL);
   pipe->delete_vertex_elements_state(pipe, vertex_elements);
   pipe->bind_blend_state(pipe, NULL);
   pipe->delete_blend_state(pipe, blend_state);
   pipe->bind_depth_stencil_alpha_state(pipe, NULL);
   pipe->delete_depth_stencil_alpha_state(pipe, zs_state);
   pipe->bind_rasterizer_state(pipe, NULL);
   pipe->delete_rasterizer_state(pipe, raster_state);
   struct pipe_framebuffer_state none = {0};
   pipe->set_framebuffer_state(pipe, &none);
   for (unsigned i = 0; i < 2; ++i)
      pipe_resource_reference(&msaa[i], NULL);
   for (unsigned i = 0; i < NR_CBUFS; ++i)
      pipe_resource_reference(&targets[i], NULL);
   pipe_resource_reference(&buffer, NULL);
   pipe->destroy(pipe);
   screen->destroy(screen);
   driDestroyOptionInfo(&info);
   close(fd);
   puts("M5 resident MRT checks passed (no GPU execution)");
   return 0;
}
