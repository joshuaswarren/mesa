/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Under the no-op DRM shim, check that the M5 profile admits the draws
 * st/mesa's blit-based texture transfers make: a pixel-buffer upload
 * (st_pbo_create_vs and st_pbo_get_upload_fs) and download
 * (st_pbo_get_download_fs) into a single layer. Their vertex shader writes
 * gl_Layer from the instance ID for every transfer, layered or not. Each
 * draw must reach the GPU: none may be dropped by the M5 draw profile, and
 * with ASAHI_SHIM_M5_EXPECT_VALID the kernel checks must accept the render.
 * This test does not execute GPU work or compare rendered pixels.
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

/* st_pbo_create_vs with st->pbo.layers: the position from the vertex buffer
 * and gl_Layer from the instance ID. */
static void *
create_pbo_vs(struct pipe_context *pipe)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "st/pbo VS");
   b.shader->info.io_lowered = true;
   nir_def *pos = nir_load_input(&b, 4, 32, nir_imm_int(&b, 0),
                                 .io_semantics.location = VERT_ATTRIB_POS);
   nir_store_output(&b, pos, nir_imm_int(&b, 0), .write_mask = 0xf,
                    .io_semantics.location = VARYING_SLOT_POS);
   nir_store_output(&b, nir_load_instance_id(&b), nir_imm_int(&b, 0),
                    .write_mask = 0x1, .src_type = nir_type_int32,
                    .io_semantics.location = VARYING_SLOT_LAYER);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   struct pipe_shader_state state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = b.shader,
   };
   void *vs = pipe->create_vs_state(pipe, &state);
   CHECK(vs);
   return vs;
}

/* The texel fetch of both transfer fragment shaders. */
static nir_def *
fetch(nir_builder *b, enum glsl_sampler_dim dim, nir_def *coord)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, dim == GLSL_SAMPLER_DIM_BUF ? 1 : 2);
   tex->op = nir_texop_txf;
   tex->sampler_dim = dim;
   tex->coord_components = coord->num_components;
   tex->dest_type = nir_type_float32;
   tex->texture_index = 0;
   tex->sampler_index = 0;
   tex->can_speculate = true;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   if (dim != GLSL_SAMPLER_DIM_BUF)
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return &tex->def;
}

/* st_pbo_create_upload_fs/download_fs for a 2D texture: the pixel buffer
 * address from gl_FragCoord and the parameters in constant buffer 0
 * (param = [-xoffset + skip_pixels, -yoffset, stride, image_height]). */
static void *
create_pbo_fs(struct pipe_context *pipe, bool download)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      download ? "st/pbo download FS" : "st/pbo upload FS");
   b.shader->info.io_lowered = true;
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *param = nir_load_ubo(&b, 4, 32, zero, zero, .align_mul = 16,
                                 .align_offset = 0, .range = 16);
   nir_def *coord = nir_load_frag_coord(&b);
   nir_def *offset_pos = nir_iadd(&b, nir_trim_vector(&b, param, 2),
                                  nir_f2i32(&b, nir_trim_vector(&b, coord, 2)));
   nir_def *pbo_addr =
      nir_iadd(&b, nir_channel(&b, offset_pos, 0),
               nir_imul(&b, nir_channel(&b, offset_pos, 1),
                        nir_channel(&b, param, 2)));

   if (download) {
      nir_def *texel = fetch(&b, GLSL_SAMPLER_DIM_2D,
                             nir_f2i32(&b, nir_trim_vector(&b, coord, 2)));
      nir_image_store(&b, zero, nir_vec4(&b, pbo_addr, zero, zero, zero),
                      zero, texel, zero, .image_dim = GLSL_SAMPLER_DIM_BUF,
                      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
                      .access = ACCESS_NON_READABLE,
                      .src_type = nir_type_float32);
      b.shader->info.num_images = 1;
      BITSET_SET(b.shader->info.images_used, 0);
   } else {
      nir_def *texel = fetch(&b, GLSL_SAMPLER_DIM_BUF, pbo_addr);
      nir_store_output(&b, texel, zero, .write_mask = 0xf,
                       .src_type = nir_type_float32,
                       .io_semantics.location = FRAG_RESULT_COLOR);
   }

   b.shader->info.num_textures = 1;
   b.shader->info.num_ubos = 1;
   BITSET_SET(b.shader->info.textures_used, 0);
   BITSET_SET(b.shader->info.textures_used_by_txf, 0);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   struct pipe_shader_state state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = b.shader,
   };
   void *fs = pipe->create_fs_state(pipe, &state);
   CHECK(fs);
   return fs;
}

/* st_pbo_draw: a rectangle as a four-vertex strip, one instance per
 * layer. */
static void
draw_rectangle(struct pipe_context *pipe, struct agx_context *ctx)
{
   const unsigned refused = ctx->skipped_no_program;
   struct pipe_draw_info info = {
      .mode = MESA_PRIM_TRIANGLE_STRIP,
      .instance_count = 1,
   };
   struct pipe_draw_start_count_bias range = {.count = 4};
   pipe->draw_vbo(pipe, &info, 0, NULL, &range, 1);
   CHECK(ctx->skipped_no_program == refused);
   CHECK(ctx->batch && ctx->batch->draws == 1);
   CHECK(!ctx->batch->draws_refused && !ctx->batch->native_rejected);
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
   struct pipe_screen_config config = {.options = &options,
                                       .options_info = &info};
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   CHECK(screen);
   CHECK(agx_chip_g17_m5(agx_device(screen)->chip));
   /* st/mesa takes these paths only for blit-based transfers, and writes
    * gl_Layer only where vertex shaders may. */
   CHECK(screen->caps.texture_transfer_modes & PIPE_TEXTURE_TRANSFER_BLIT);
   CHECK(screen->caps.vs_instanceid && screen->caps.vs_layer_viewport);
   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   CHECK(pipe);
   struct agx_context *ctx = agx_context(pipe);

   const float vertices[4][4] = {
      {-1, -1, 0, 1}, {1, -1, 0, 1}, {-1, 1, 0, 1}, {1, 1, 0, 1},
   };
   struct pipe_resource *vbuf = pipe_buffer_create(
      screen, PIPE_BIND_VERTEX_BUFFER, PIPE_USAGE_STREAM, sizeof(vertices));
   CHECK(vbuf);
   pipe_buffer_write(pipe, vbuf, 0, sizeof(vertices), vertices);
   struct pipe_vertex_buffer vertex_buffer = {.buffer.resource = vbuf};
   pipe->set_vertex_buffers(pipe, 1, &vertex_buffer);
   const struct pipe_vertex_element element = {
      .src_format = PIPE_FORMAT_R32G32B32A32_FLOAT,
      .src_stride = 16,
   };
   void *velems = pipe->create_vertex_elements_state(pipe, 1, &element);
   CHECK(velems);
   pipe->bind_vertex_elements_state(pipe, velems);

   /* st_init_pbo_helpers' blend, rasterizer and depth state. */
   struct pipe_blend_state blend = {0};
   blend.rt[0].colormask = PIPE_MASK_RGBA;
   void *blend_state = pipe->create_blend_state(pipe, &blend);
   struct pipe_rasterizer_state raster = {
      .half_pixel_center = 1, .depth_clip_near = 1, .depth_clip_far = 1,
   };
   void *raster_state = pipe->create_rasterizer_state(pipe, &raster);
   struct pipe_depth_stencil_alpha_state zs = {0};
   void *zs_state = pipe->create_depth_stencil_alpha_state(pipe, &zs);
   CHECK(blend_state && raster_state && zs_state);
   pipe->bind_blend_state(pipe, blend_state);
   pipe->bind_rasterizer_state(pipe, raster_state);
   pipe->bind_depth_stencil_alpha_state(pipe, zs_state);
   struct pipe_viewport_state viewport = {
      .scale = {32, 32, 1}, .translate = {32, 32, 0},
   };
   pipe->set_viewport_states(pipe, 0, 1, &viewport);

   /* param = [0, 0, stride 64, image height 64]. */
   const int32_t param[4] = {0, 0, 64, 64};
   struct pipe_constant_buffer cb = {
      .buffer_size = sizeof(param),
      .user_buffer = param,
   };
   pipe->set_constant_buffer(pipe, MESA_SHADER_FRAGMENT, 0, &cb);

   const struct pipe_resource texture_templ = {
      .target = PIPE_TEXTURE_2D, .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 64, .height0 = 64, .depth0 = 1, .array_size = 1,
      .nr_samples = 1, .nr_storage_samples = 1,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
   };
   struct pipe_resource *texture =
      screen->resource_create(screen, &texture_templ);
   CHECK(texture);
   /* As st/mesa allocates a pixel buffer object: a plain buffer, bound as
    * a texel buffer or an image only through views. */
   struct pipe_resource *pbo = pipe_buffer_create(
      screen, PIPE_BIND_SAMPLER_VIEW, PIPE_USAGE_STREAM, 64 * 64 * 4);
   CHECK(pbo);

   void *vs = create_pbo_vs(pipe);
   void *upload_fs = create_pbo_fs(pipe, false);
   void *download_fs = create_pbo_fs(pipe, true);
   pipe->bind_vs_state(pipe, vs);

   /* Upload: the pixel buffer as a texel buffer, drawn into the texture. */
   struct pipe_sampler_view buffer_templ = {
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .target = PIPE_BUFFER,
      .swizzle_r = PIPE_SWIZZLE_X, .swizzle_g = PIPE_SWIZZLE_Y,
      .swizzle_b = PIPE_SWIZZLE_Z, .swizzle_a = PIPE_SWIZZLE_W,
      .u.buf = {.offset = 0, .size = 64 * 64 * 4},
   };
   struct pipe_sampler_view *buffer_view =
      pipe->create_sampler_view(pipe, pbo, &buffer_templ);
   CHECK(buffer_view);
   pipe->set_sampler_views(pipe, MESA_SHADER_FRAGMENT, 0, 1, 0, &buffer_view);
   struct pipe_framebuffer_state fb = {
      .width = 64, .height = 64, .layers = 1, .nr_cbufs = 1,
   };
   fb.cbufs[0] = (struct pipe_surface){
      .texture = texture, .format = texture->format,
   };
   pipe->set_framebuffer_state(pipe, &fb);
   pipe->bind_fs_state(pipe, upload_fs);
   draw_rectangle(pipe, ctx);
   pipe->flush(pipe, NULL, 0);

   /* Download: the texture fetched, written to the pixel buffer as an
    * image, with no attachment. */
   struct pipe_sampler_view texture_templ_view = {
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .target = PIPE_TEXTURE_2D,
      .swizzle_r = PIPE_SWIZZLE_X, .swizzle_g = PIPE_SWIZZLE_Y,
      .swizzle_b = PIPE_SWIZZLE_Z, .swizzle_a = PIPE_SWIZZLE_W,
   };
   struct pipe_sampler_view *texture_view =
      pipe->create_sampler_view(pipe, texture, &texture_templ_view);
   CHECK(texture_view);
   pipe->set_sampler_views(pipe, MESA_SHADER_FRAGMENT, 0, 1, 0, &texture_view);
   const struct pipe_image_view image = {
      .resource = pbo,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .access = PIPE_IMAGE_ACCESS_WRITE,
      .shader_access = PIPE_IMAGE_ACCESS_WRITE,
      .u.buf = {.offset = 0, .size = 64 * 64 * 4},
   };
   pipe->set_shader_images(pipe, MESA_SHADER_FRAGMENT, 0, 1, 0, &image);
   struct pipe_framebuffer_state empty = {
      .width = 64, .height = 64, .layers = 1, .samples = 1,
   };
   pipe->set_framebuffer_state(pipe, &empty);
   pipe->bind_fs_state(pipe, download_fs);
   draw_rectangle(pipe, ctx);

   /* With ASAHI_SHIM_M5_EXPECT_VALID, a render the M5 kernel checks refuse
    * fails the run at exit. */
   pipe->flush(pipe, NULL, 0);

   pipe->set_shader_images(pipe, MESA_SHADER_FRAGMENT, 0, 0, 1, NULL);
   pipe->set_sampler_views(pipe, MESA_SHADER_FRAGMENT, 0, 0, 1, NULL);
   pipe_sampler_view_reference(&buffer_view, NULL);
   pipe_sampler_view_reference(&texture_view, NULL);
   pipe->set_framebuffer_state(pipe, &(struct pipe_framebuffer_state){0});
   pipe->bind_vs_state(pipe, NULL);
   pipe->bind_fs_state(pipe, NULL);
   pipe->delete_vs_state(pipe, vs);
   pipe->delete_fs_state(pipe, upload_fs);
   pipe->delete_fs_state(pipe, download_fs);
   pipe->bind_vertex_elements_state(pipe, NULL);
   pipe->delete_vertex_elements_state(pipe, velems);
   pipe->bind_blend_state(pipe, NULL);
   pipe->delete_blend_state(pipe, blend_state);
   pipe->bind_rasterizer_state(pipe, NULL);
   pipe->delete_rasterizer_state(pipe, raster_state);
   pipe->bind_depth_stencil_alpha_state(pipe, NULL);
   pipe->delete_depth_stencil_alpha_state(pipe, zs_state);
   pipe_resource_reference(&texture, NULL);
   pipe_resource_reference(&pbo, NULL);
   pipe_resource_reference(&vbuf, NULL);
   pipe->destroy(pipe);
   screen->destroy(screen);
   driDestroyOptionInfo(&info);
   close(fd);
   puts("M5 PBO transfer draws admitted (no GPU execution)");
   return 0;
}
